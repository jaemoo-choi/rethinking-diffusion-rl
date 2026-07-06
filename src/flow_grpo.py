# SPDX-License-Identifier: Apache-2.0
"""Flow-GRPO training entrypoint (yifan123/flow_grpo).

Standalone alternative to train.py that swaps the ELBO-ratio policy loss for
the original Flow-GRPO PPO-clipped objective: ratios are computed from the
per-step log-probabilities of the SDE flow sampler (solver="flow"), exactly
as in the upstream paper.

Pair with any model's ``*_flow_grpo`` config (e.g.
``config/wan2_1.py:wan2_1_flow_grpo`` or
``config/skyreels_i2v.py:skyreels_flow_grpo``), which force
``sample.solver="flow"``, ``sample.deterministic=False`` and
``sample.noise_level`` = eta (upstream default 0.7), while keeping our shift /
num_steps / dataset / reward. Drive from ``scripts/<model>_flow_grpo.sh``
with ``METHOD=flow_grpo ADV=standard BETA=0 SCALE=1.0``. Everything else
(dataset, reward, model, optimizer, LoRA, distributed setup, EMA, resume)
reuses the existing repo plumbing, so the entrypoint is model-agnostic via
``build_model_utils`` (sd3 / flux / wan2_1 / hunyuan / skyreels_i2v /
wan2_2_i2v / worldplay). NOTE: wan2_2_a14b_i2v needs the separate
sequential-offload entrypoint train_wan2_2_a14b.py and is not supported here.
"""

from collections import defaultdict
import gc
import math
import os
import signal
from concurrent import futures
import time
import zipfile

import numpy as np
import torch
import torch.distributed as dist
import wandb
from functools import partial

# This module lives in src/ but is launched as a script (torchrun src/flow_grpo.py),
# so the repo root is not on sys.path by default. Add it so `config`, `src`, and
# `src.dataloader` resolve regardless of how the script is invoked.
import sys as _sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)

from config.paths import WANDB_ENTITY
import tqdm
from src.ema import EMAModuleWrapper
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast
from absl import app, flags
import logging
from src.utils import *
from src.diffusers_patch.solver import flow_grpo_step
from src.dataloader import *


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# Flow-GRPO PPO clip range. Matches the upstream repo default
# (yifan123/flow_grpo, scripts/single_node/grpo_*.sh: --clip_range 1e-4
# at the loss level corresponds to a ratio clip of 1±1e-4 for log-prob
# ratios; we use 1e-3 to match this codebase's existing "grpo" method
# in src/utils/base.py:411).
FLOW_GRPO_CLIP_EPS = 1e-3


# Per-sample conditioning keys that each model's v_pred_fn reads from its
# `samples` arg during the training pass. train.py passes the FULL sample batch
# to compute_loss; flow_grpo.py instead builds a slim v_pred_samples per
# timestep, so we must forward these verbatim or conditioned/I2V models silently
# lose their conditioning (e.g. skyreels_i2v: first_frame_latents=None -> the
# 16ch->32ch channel-concat never happens -> shape error / wrong velocity).
# These keys are per-sample (NOT time-shuffled), so column-slicing does not
# apply; pass them whole. The tuple is a superset — `if k in batch` makes it a
# no-op for unconditioned models (sd3/flux/wan2_1, which read only timesteps).
_PASSTHRU_VPRED_KEYS = (
    "first_frame_latents",     # skyreels_i2v
    "prompt_attention_mask",   # skyreels_i2v / hunyuan / worldplay
    "i2v_condition",           # wan2_2_i2v
    "i2v_first_frame_mask",    # wan2_2_i2v
    "action_ids",              # worldplay
)


def _flow_grpo_log_prob(model_output, latents, prev_sample, sigmas, indices, eta):
    """Vectorised flow-GRPO Gaussian log-prob, plus the artefacts needed for KL.

    Returns ``(log_prob, prev_sample_mean, sigma_dev_dt)``:

    - ``log_prob``: per-row log p(prev_sample | latents, theta), reduced over
      the latent feature dims by mean (matches solver.flow_grpo_step).
    - ``prev_sample_mean``: the policy-mean μ_θ(x_t, v_θ) used by the
      upstream Gaussian-mean KL formula in train_wan2_1.py.
    - ``sigma_dev_dt``: ``std_dev_t * sqrt(-dt)`` — policy-independent step
      std (depends only on the sampler / schedule), reusable for both
      branches of the KL.

    Per-row indexing in ``indices`` lets each batch row live at a different
    sampler step (train.py's per-row time permutation).
    """
    device = model_output.device
    sigmas = sigmas.to(device)
    sigma = sigmas[indices]                          # (B,)
    sigma_prev = sigmas[indices + 1]                 # (B,)
    sigma_max = float(sigmas[1].item())

    # Broadcast over latent dims (B, 1, 1, ..., 1).
    expand_shape = [model_output.shape[0]] + [1] * (model_output.ndim - 1)
    sigma_b = sigma.view(*expand_shape)
    sigma_prev_b = sigma_prev.view(*expand_shape)
    dt = sigma_prev_b - sigma_b  # neg dt

    # Guard the sigma->1 singularity in 1/(1-sigma) the SAME way the sampler does
    # (solver.flow_grpo_step): clamp sigma to sigma_max (= sigmas[1]). An exact
    # `sigma == 1` test misfires for UniPCMultistepScheduler (wan2.1/wan2.2),
    # whose sigmas[0] ~= 0.999875 != 1 -> std_dev_t explodes (~89*eta) and the
    # trainer's top-step log-prob diverges from the stored sampler log-prob,
    # corrupting the importance ratio. clamp matches the solver exactly.
    sigma_eff = torch.clamp(sigma_b, max=sigma_max)
    std_dev_t = torch.sqrt(sigma_b / (1 - sigma_eff)) * eta
    sigma_dev_dt = std_dev_t * torch.sqrt(-1 * dt)

    prev_sample_mean = (
        latents * (1 + std_dev_t ** 2 / (2 * sigma_b) * dt)
        + model_output * (1 + std_dev_t ** 2 * (1 - sigma_b) / (2 * sigma_b)) * dt
    )

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * sigma_dev_dt ** 2)
        - torch.log(sigma_dev_dt)
        - 0.5 * math.log(2 * math.pi)
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return log_prob, prev_sample_mean, sigma_dev_dt


def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    # --- WandB Init (only on main process) ---
    log_dir = config.save_dir
    wandb_run_id = None

    wandb_run_id_file = os.path.join(log_dir, "wandb_run_id.txt")
    if os.path.exists(wandb_run_id_file):
        with open(wandb_run_id_file) as _f:
            wandb_run_id = _f.read().strip() or None
    elif os.path.exists(os.path.join(log_dir, "checkpoints/training_state.pt")):
        training_state = torch.load(os.path.join(log_dir, "checkpoints/training_state.pt"), map_location="cpu")
        wandb_run_id = training_state.get("wandb_run_id")

    if is_main_process(rank):
        os.makedirs(log_dir, exist_ok=True)
        if wandb_run_id:
            try:
                wandb.init(entity=WANDB_ENTITY, project=(getattr(config, "project", None) or config.name), name=config.run_name,
                           config=config.to_dict(), dir=log_dir, id=wandb_run_id, resume="allow")
                logger.info(f"Resumed wandb run with id: {wandb_run_id}")
            except Exception as e:
                logger.warning(f"Failed to resume wandb run {wandb_run_id}: {e}")
                wandb.init(entity=WANDB_ENTITY, project=(getattr(config, "project", None) or config.name), name=config.run_name,
                           config=config.to_dict(), dir=log_dir)
                logger.info(f"Created new wandb run with id: {wandb.run.id}")
        else:
            wandb.init(entity=WANDB_ENTITY, project=(getattr(config, "project", None) or config.name), name=config.run_name,
                       config=config.to_dict(), dir=log_dir)
            logger.info(f"Created new wandb run with id: {wandb.run.id}")

        with open(wandb_run_id_file, "w") as _f:
            _f.write(wandb.run.id)

        # Clean wandb shutdown on SLURM termination (TIMEOUT / scancel / resubmit):
        # close the run so it reopens as "running" (not "crashed") on the next
        # allocation, then RE-RAISE the signal so SLURM still records the job as
        # terminated-by-signal (not exit-0 COMPLETED) and auto_resubmit resubmits.
        def _wandb_clean_shutdown(signum, _frame):
            try:
                wandb.finish(exit_code=0, quiet=True)
            except Exception:
                pass
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        for _sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(_sig, _wandb_clean_shutdown)
            except Exception:
                pass

    logger.info(f"\n{config}")

    # Flow-GRPO requires a stochastic flow sampler.
    assert config.sample.solver == "flow", (
        f"flow_grpo.py requires sample.solver='flow', got '{config.sample.solver}'."
    )
    assert not config.sample.deterministic, (
        "flow_grpo.py requires sample.deterministic=False (stochastic SDE step)."
    )
    eta = float(config.sample.noise_level)
    if is_main_process(rank):
        logger.info(f"[flow_grpo] eta={eta}, clip_eps={FLOW_GRPO_CLIP_EPS}")

    set_seed(config.seed, rank)

    base_model = getattr(config, "base_model", "sd3")

    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=(mixed_precision_dtype == torch.float16))

    model_utils = build_model_utils(
        base_model,
        config=config,
        logger=logger,
        wandb_module=wandb,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    pipeline, text_encoders, tokenizers = model_utils.get_pretrained_model(
        mixed_precision_dtype=mixed_precision_dtype,
        enable_amp=enable_amp,
    )
    (
        transformer_ddp,
        transformer_trainable_parameters,
        old_transformer_trainable_parameters,
        transformer_lora_config,
    ) = model_utils.set_adapter(pipeline, local_rank)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if getattr(config, "gradient_checkpointing", False):
        pipeline.transformer.enable_gradient_checkpointing()
        if is_main_process(rank):
            logger.info("Gradient checkpointing enabled on transformer")

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    train_dataloader, test_dataloader = get_dataloaders(
        config=config, world_size=world_size, rank=rank, num_workers=0,
    )
    train_sampler = train_dataloader.batch_sampler

    neg_prompt_embed, neg_pooled_prompt_embed = model_utils.get_negative_prompt_embeddings(text_encoders, tokenizers)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    if neg_pooled_prompt_embed is not None:
        sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
        train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)
    else:
        sample_neg_pooled_prompt_embeds = None
        train_neg_pooled_prompt_embeds = None

    stat_tracker = PerPromptStatTracker(config.sample.global_std)
    if config.resume_from:
        training_state_path = os.path.join(log_dir, "checkpoints/training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location="cpu")
            if "stat_tracker_history_prompts" in training_state:
                stat_tracker.history_prompts = training_state["stat_tracker_history_prompts"]
                logger.info(f"Loaded stat_tracker with {len(stat_tracker.history_prompts)} historical prompts")

    executor = futures.ThreadPoolExecutor(max_workers=8)
    sample_log_executor = futures.ThreadPoolExecutor(max_workers=1)
    sample_log_future = None

    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running Flow-GRPO training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info(f"  Total samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (parallel/dist/accum) = {total_train_batch_size}")

    va_dim_weights = getattr(config, "videoalign_dimension_weights", None) or None
    _va_video_fps = getattr(config, "video_fps", None)
    reward_fn = get_reward_fn(device, config.reward_fn,
                              videoalign_dimension_weights=va_dim_weights, video_fps=_va_video_fps)
    # eval_reward_fn intentionally not instantiated — flow_grpo.py has no
    # eval block (mirroring train.py where the eval_fn call is commented
    # out), and a second VideoAlign instance costs ~14 GB host RAM that
    # pushes the cache-save / row-permutation peak over the cgroup limit
    # on 8-GPU H200 placements.

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from != "" or os.path.exists(os.path.join(log_dir, "checkpoints/lora_current_policy/adapter_model.safetensors")):
        try:
            config.resume_from = os.path.join(log_dir, "checkpoints")
        except Exception:
            assert config.resume_from
        logger.info(f"Resuming from {config.resume_from}")

        lora_current_path = os.path.join(config.resume_from, "lora_current_policy")
        if os.path.exists(lora_current_path):
            transformer_ddp.module.load_adapter(lora_current_path, adapter_name="default", is_trainable=True)

        lora_old_path = os.path.join(config.resume_from, "lora_old_policy")
        if os.path.exists(lora_old_path):
            transformer_ddp.module.load_adapter(lora_old_path, adapter_name="old", is_trainable=False)

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            except (RuntimeError, EOFError, zipfile.BadZipFile) as e:
                logger.warning(f"Failed to load optimizer state from {opt_path}: {e}")

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            try:
                scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))
            except (RuntimeError, EOFError, zipfile.BadZipFile) as e:
                logger.warning(f"Failed to load scaler state from {scaler_path}: {e}")

        training_state_path = os.path.join(config.resume_from, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            global_step = training_state["global_step"]
            first_epoch = training_state.get("epoch", 0)
            logger.info(f"Loaded global_step: {global_step}, epoch: {first_epoch}")

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)
        if config.resume_from:
            lora_ema_path = os.path.join(config.resume_from, "lora_ema_policy")
            if os.path.exists(lora_ema_path):
                transformer_ddp.module.add_adapter("temp_ema_load", transformer_lora_config)
                transformer_ddp.module.load_adapter(lora_ema_path, adapter_name="temp_ema_load", is_trainable=False)
                transformer_ddp.module.set_adapter("temp_ema_load")
                temp_ema_params = [p for p in transformer_ddp.module.parameters() if p.requires_grad]
                for ema_param, loaded_param in zip(ema.ema_parameters, temp_ema_params):
                    ema_param.data.copy_(loaded_param.data)
                transformer_ddp.module.set_adapter("default")
                transformer_ddp.module.delete_adapter("temp_ema_load")

    if world_size > 1:
        dist.barrier()

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    if not config.resume_from:
        for src_param, tgt_param in zip(
            transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
        ):
            tgt_param.data.copy_(src_param.detach().data)

    ####################
    #### START LOOP ####
    ####################
    for epoch in range(first_epoch, config.num_epochs):
        epoch_start_time = time.time()
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        ##################
        #### SAMPLING ####
        ##################
        # Cache fast-path (mirrors train.py:340-366): on resume, if the
        # samples for this epoch were persisted by a previous run, skip the
        # whole sampling phase and load the cached collated_samples.
        _sample_cache_enabled = getattr(config, "save_sample_cache", True) and not config.debug
        _cache_hit = False
        sampling_time = 0.0
        peak_sampling_gb = 0.0
        sigma_schedule = None
        if _sample_cache_enabled and epoch == first_epoch and config.resume_from:
            _cached = load_sample_cache(config.save_dir, epoch, rank, world_size, logger)
            if _cached is not None:
                _cache_hit = True
                if is_main_process(rank):
                    logger.info(f"Epoch {epoch}: loaded sample cache, skipping sampling phase")
                # sigma_schedule must always be on device; pull it out of the
                # cache before the conditional move-to-device pass below.
                sigma_schedule = _cached.pop("_sigma_schedule").to(device).float()
                pipeline.vae.to("cpu")
                if hasattr(reward_fn, "to_cpu"):
                    reward_fn.to_cpu()
                # Cache is always saved from CPU. If this run keeps samples
                # on GPU during training (cpu_offload_samples=False), move
                # them back; otherwise leave on CPU like the normal path.
                if not getattr(config, "cpu_offload_samples", False):
                    for _k, _v in list(_cached.items()):
                        if isinstance(_v, torch.Tensor):
                            _cached[_k] = _v.to(device)
                collated_samples = _cached
                gc.collect()
                torch.cuda.empty_cache()

        if not _cache_hit:
            sampling_start_time = time.time()
            torch.cuda.reset_peak_memory_stats(device)
            pipeline.vae.to(device)
            if hasattr(reward_fn, "to_gpu"):
                reward_fn.to_gpu()
            pipeline.transformer.eval()
            samples_data_list = []

            for i in tqdm(
                range(config.sample.num_batches_per_epoch),
                desc=f"Epoch {epoch}: sampling",
                disable=not is_main_process(rank),
                position=0,
            ):
                transformer_ddp.module.set_adapter("default")
                if hasattr(train_sampler, "set_epoch") and isinstance(train_sampler, DistributedKRepeatSampler):
                    train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

                prompts, prompt_metadata = next(train_iter)
                batch_inputs = model_utils.extract_dataloader_metadata(
                    prompts, prompt_metadata, text_encoders, tokenizers,
                )
                prompt_embeds = batch_inputs["prompt_embeds"]
                pooled_prompt_embeds = batch_inputs["pooled_prompt_embeds"]
                prompt_ids = batch_inputs["prompt_ids"]

                if i == 0 and is_main_process(rank) and not config.debug:
                    # Rolling checkpoint (overwritten every save_freq epochs) —
                    # what resume reads (<save_dir>/checkpoints/).
                    if epoch % config.save_freq == 0:
                        model_utils.save_ckpt(
                            config.save_dir, transformer_ddp, global_step, ema,
                            transformer_trainable_parameters, optimizer, scaler, epoch, stat_tracker,
                        )

                    # Permanent epoch snapshots: kept forever, every
                    # num_epochs // 10 epochs. Written to
                    # <save_dir>/permanent/epoch_<N>/checkpoints/ (loadable by
                    # evaluation.py like the rolling ckpt). The FINAL trained
                    # model is saved separately after the loop.
                    perm_interval = max(1, config.num_epochs // 10)
                    if epoch % perm_interval == 0:
                        model_utils.save_ckpt(
                            config.save_dir, transformer_ddp, global_step, ema,
                            transformer_trainable_parameters, optimizer, scaler, epoch, stat_tracker,
                            ckpt_subdir=os.path.join("permanent", f"epoch_{epoch}", "checkpoints"),
                        )

                offload_during_decode = getattr(config, "offload_transformer_during_decode", False)

                transformer_ddp.module.set_adapter("old")
                with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                    with torch.no_grad():
                        sample_kwargs = model_utils.build_sample_kwargs(
                            prompt_embeds, pooled_prompt_embeds,
                            sample_neg_prompt_embeds, sample_neg_pooled_prompt_embeds,
                            len(prompts),
                        )
                        if offload_during_decode:
                            sample_kwargs["skip_decode"] = True
                        images, latents, log_probs = model_utils.pipeline_with_logprob(pipeline, **sample_kwargs)

                transformer_ddp.module.set_adapter("default")

                latents = torch.stack(latents, dim=1)  # (B, num_steps+1, *)
                log_probs = torch.stack(log_probs, dim=1)  # (B, num_steps)
                latents_clean = latents[:, -1].clone()
                latents_trajectory = latents

                if sigma_schedule is None:
                    sigma_schedule = pipeline.scheduler.sigmas.detach().to(device).float()

                if offload_during_decode:
                    pipeline.transformer.to("cpu")
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        images = model_utils.decode_latents(pipeline, latents_clean)

                timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)
                rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
                time.sleep(0)
                last_images = images
                last_prompts = prompts
                del images

                if offload_during_decode and i < config.sample.num_batches_per_epoch - 1:
                    pipeline.transformer.to(device)
                    torch.cuda.empty_cache()

                sample_data = {
                    "prompt_ids": prompt_ids,
                    "prompts_text": prompts,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents_trajectory[:, :-1].contiguous(),
                    "next_latents": latents_trajectory[:, 1:].contiguous(),
                    "log_probs": log_probs,
                    "latents_clean": latents_clean,
                    "rewards_future": rewards_future,
                }
                del latents_trajectory
                model_utils.collect_sample_artifacts(sample_data)
                samples_data_list.append(sample_data)

            # Collect rewards
            for sample_item in tqdm(
                samples_data_list, desc="Waiting for rewards", disable=not is_main_process(rank), position=0
            ):
                rewards, _ = sample_item["rewards_future"].result()
                sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
                del sample_item["rewards_future"]

            # Collate
            def _collate_field(key, samples):
                first = samples[0][key]
                if first is None:
                    return None
                if isinstance(first, list):
                    result = []
                    for s in samples:
                        result.extend(s[key])
                    return result
                if isinstance(first, dict):
                    return {sk: torch.cat([s[key][sk] for s in samples], dim=0) for sk in first}
                return torch.cat([s[key] for s in samples], dim=0)

            collated_samples = {k: _collate_field(k, samples_data_list) for k in samples_data_list[0].keys()}
            del samples_data_list

            sampling_time = time.time() - sampling_start_time
            peak_sampling_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if is_main_process(rank):
                logger.info(f"Epoch {epoch}: Sampling took {sampling_time:.2f}s (peak VRAM: {peak_sampling_gb:.2f} GiB)")

            # Sample logging (videos to wandb)
            if epoch % config.sample_log_freq == 0 and is_main_process(rank):
                if sample_log_future is None or sample_log_future.done():
                    num_log = len(last_images) if hasattr(last_images, "__len__") else last_images.shape[0]
                    rewards_to_log = collated_samples["rewards"]["avg"][-num_log:].cpu()
                    sample_log_future = sample_log_executor.submit(
                        model_utils.log_train_samples, last_images, last_prompts, rewards_to_log, global_step,
                    )

            num_train_timesteps = int(config.sample.num_steps)
            collated_samples["rewards"]["avg"] = (
                collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
            )

            gathered_rewards_dict = {}
            for key, value_tensor in collated_samples["rewards"].items():
                gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

            _pending_sampling_log = {}
            if is_main_process(rank):
                for _k, _v in gathered_rewards_dict.items():
                    _pending_sampling_log[f"reward_{_k}"] = float(_v.mean())
                    if "_accuracy" not in _k:
                        _pending_sampling_log[f"reward_{_k}_std"] = float(_v.std())

            if config.per_prompt_stat_tracking:
                if collated_samples.get("prompt_ids") is not None:
                    prompt_ids_all = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
                    prompts_all_decoded = pipeline.tokenizer.batch_decode(
                        prompt_ids_all.cpu().numpy(), skip_special_tokens=True
                    )
                else:
                    import itertools
                    local_prompts = collated_samples["prompts_text"]
                    all_prompts_nested = [None] * world_size
                    dist.all_gather_object(all_prompts_nested, local_prompts)
                    prompts_all_decoded = list(itertools.chain.from_iterable(all_prompts_nested))
                advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"], config.train)
                if is_main_process(rank):
                    group_size, trained_prompt_num = stat_tracker.get_stats()
                    zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all_decoded, gathered_rewards_dict)
                    _pending_sampling_log.update({
                        "group_size": float(group_size),
                        "trained_prompt_num": float(trained_prompt_num),
                        "zero_std_ratio": float(zero_std_ratio),
                        "reward_std_mean": float(reward_std_mean),
                    })
                stat_tracker.clear()
            else:
                raise NotImplementedError

            samples_per_gpu = collated_samples["timesteps"].shape[0]
            avg_rewards_all = gathered_rewards_dict["avg"]
            if advantages.ndim == 1:
                advantages = advantages[:, None]
            assert advantages.shape[0] == world_size * samples_per_gpu
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
            collated_samples["rewards"] = torch.from_numpy(
                avg_rewards_all.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)

            if is_main_process(rank):
                _adv_flat_np = np.asarray(advantages, dtype=np.float64).reshape(-1)
                _adv_clip_max = float(getattr(config.train, "adv_clip_max", 5.0))
                _adv_frac_saturated = (
                    float(np.mean(np.abs(_adv_flat_np) >= _adv_clip_max)) if _adv_clip_max > 0 else 0.0
                )
                _pending_sampling_log.update({
                    "advantages_mean": float(_adv_flat_np.mean()),
                    "advantages_max": float(_adv_flat_np.max()),
                    "advantages_min": float(_adv_flat_np.min()),
                    "advantages_std": float(_adv_flat_np.std()),
                    "advantages_abs_mean": float(np.abs(_adv_flat_np).mean()),
                    "advantages_frac_saturated": _adv_frac_saturated,
                })

            collated_samples.pop("prompt_ids", None)
            collated_samples.pop("prompts_text", None)
            del last_images, last_prompts

            # Stash the precomputed sampling-side wandb log on rank 0 so it
            # survives the sample cache. We fold this into the first training
            # step's wandb.log below — keeping all sampling+training metrics
            # in one log call at the same global_step lets wandb commit them
            # together instead of dropping the standalone calls.
            if is_main_process(rank):
                collated_samples["_pending_sampling_log"] = dict(_pending_sampling_log)

            if getattr(config, "offload_transformer_during_decode", False):
                if next(pipeline.transformer.parameters()).device.type == "cpu":
                    pipeline.transformer.to(device)
                    torch.cuda.empty_cache()

            pipeline.vae.to("cpu")
            if hasattr(reward_fn, "to_cpu"):
                reward_fn.to_cpu()

            if getattr(config, "cpu_offload_samples", False):
                for k, v in collated_samples.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        collated_samples[k] = v.cpu()
            gc.collect()
            torch.cuda.empty_cache()

            # Persist samples for fast resume next time. sigma_schedule
            # piggy-backs as a sentinel key (popped on load above).
            if _sample_cache_enabled:
                _save_dict = dict(collated_samples)
                _save_dict["_sigma_schedule"] = sigma_schedule.cpu()
                save_sample_cache(config.save_dir, epoch, rank, world_size, _save_dict, logger)

        # Pop the sampling-side wandb log (rewards/group/advantages) and flush
        # immediately — if SLURM kills the job between sampling and the first
        # gradient update, deferring to the training loop would drop the dict.
        pending_sampling_log = collated_samples.pop("_pending_sampling_log", None) or {}
        if is_main_process(rank) and pending_sampling_log:
            pending_sampling_log["epoch"] = epoch
            wandb.log(pending_sampling_log, step=global_step)

        ##################
        #### TRAINING ####
        ##################
        total_batch_size, num_steps_stored = collated_samples["timesteps"].shape
        # latents tensor stores num_steps entries (we sliced [:, :-1]); sanity check.
        assert collated_samples["latents"].shape[1] == num_steps_stored
        num_train_timesteps = max(1, int(num_steps_stored * config.train.timestep_fraction))
        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size

        training_start_time = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        transformer_ddp.train()

        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps
        current_accumulated_steps = 0
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm_device = "cpu" if getattr(config, "cpu_offload_samples", False) else device
            perm = torch.randperm(total_batch_size, device=perm_device)
            shuffled_samples = {k: v[perm] for k, v in collated_samples.items()}

            perms_time = torch.stack([
                torch.randperm(num_steps_stored, device=perm_device) for _ in range(total_batch_size)
            ])
            time_shuffle_keys = ["timesteps", "latents", "next_latents", "log_probs"]
            row_idx = torch.arange(total_batch_size, device=perm_device)[:, None]
            for key in time_shuffle_keys:
                shuffled_samples[key] = shuffled_samples[key][row_idx, perms_time]

            # advantages is per-sample (broadcast across timesteps); align by row only.
            shuffled_samples["advantages"] = shuffled_samples["advantages"][row_idx, perms_time[:, :1]]

            training_batch_size = total_batch_size // num_batches
            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0, disable=not is_main_process(rank),
            ):
                if getattr(config, "cpu_offload_samples", False):
                    train_sample_batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in train_sample_batch.items()
                    }
                embeds = train_sample_batch["prompt_embeds"]
                pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # Pre-shuffled per-sample sampler-step indices for this micro-batch.
                # `perms_time` lives at the outer scope; slice the same rows we
                # used to build samples_batched_list so the (xt, x_{t-1}, t,
                # logp_old) tuple stays self-consistent.
                start = i * training_batch_size
                end = (i + 1) * training_batch_size
                step_idx_batch = perms_time[start:end].to(device)  # (B, num_steps_stored)

                for j_idx in tqdm(
                    range(num_train_timesteps), desc="Timestep", position=1, leave=False,
                    disable=not is_main_process(rank),
                ):
                    # Timesteps already shuffled in-place on the dict — column j_idx
                    # is exactly the j_idx-th drawn sampler step for each row.
                    xt = train_sample_batch["latents"][:, j_idx]
                    next_x = train_sample_batch["next_latents"][:, j_idx]
                    old_log_prob = train_sample_batch["log_probs"][:, j_idx]
                    advantages = train_sample_batch["advantages"][:, 0]
                    sampler_step = step_idx_batch[:, j_idx]  # (B,) — original sampler index per sample

                    # samples dict consumed by v_pred_fn. Forward the per-sample
                    # conditioning keys (no-op for sd3/flux/wan2_1, required for
                    # I2V/conditioned models) plus the time-sliced timesteps —
                    # v_pred_fn is called with index=0 and reads timesteps[:, 0].
                    v_pred_samples = {
                        k: train_sample_batch[k]
                        for k in _PASSTHRU_VPRED_KEYS
                        if k in train_sample_batch
                    }
                    v_pred_samples["timesteps"] = train_sample_batch["timesteps"][:, j_idx:j_idx + 1]

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        # --- Current policy velocity (only forward with grad) ---
                        transformer_ddp.module.set_adapter("default")
                        cur_v = model_utils.v_pred_fn(
                            transformer_ddp, xt, v_pred_samples, 0,
                            embeds, pooled_embeds,
                            train_neg_prompt_embeds, train_neg_pooled_prompt_embeds,
                            "train",
                        ).float()

                        # --- Reference policy (LoRA disabled, frozen) ---
                        with torch.no_grad():
                            with transformer_ddp.module.disable_adapter():
                                ref_v = model_utils.v_pred_fn(
                                    transformer_ddp, xt, v_pred_samples, 0,
                                    embeds, pooled_embeds,
                                    train_neg_prompt_embeds, train_neg_pooled_prompt_embeds,
                                    "ref",
                                ).float()
                            transformer_ddp.module.set_adapter("default")

                    # Gradients flow only through cur_v (xt and next_x are stored; sigmas are constants).
                    # Stored ``old_log_prob`` was produced under the "old" adapter at sampling
                    # time and that adapter is frozen for the duration of this epoch — use directly.
                    new_log_prob, mu_cur, sigma_dev_dt = _flow_grpo_log_prob(
                        cur_v, xt.float(), next_x.float(),
                        sigma_schedule, sampler_step, eta,
                    )
                    with torch.no_grad():
                        # sigma_dev_dt is policy-independent; only μ_ref differs.
                        _, mu_ref, _ = _flow_grpo_log_prob(
                            ref_v, xt.float(), next_x.float(),
                            sigma_schedule, sampler_step, eta,
                        )

                    # --- PPO clipped policy loss ---
                    advantages_clipped = advantages.clamp(-config.train.adv_clip_max, config.train.adv_clip_max)
                    log_ratio = new_log_prob - old_log_prob.float()
                    ratio = log_ratio.exp()
                    unclipped = -advantages_clipped * ratio
                    clipped = -advantages_clipped * torch.clamp(
                        ratio, 1.0 - FLOW_GRPO_CLIP_EPS, 1.0 + FLOW_GRPO_CLIP_EPS,
                    )
                    policy_loss = torch.maximum(unclipped, clipped).mean()

                    # --- Upstream flow_grpo per-step Gaussian-mean KL ---
                    # train_wan2_1.py: ((μ_θ - μ_ref)² / (2 σ_dev² Δt²)).mean()
                    kl_div = (
                        (mu_cur - mu_ref.detach()) ** 2 / (2.0 * sigma_dev_dt ** 2)
                    ).mean(dim=tuple(range(1, mu_cur.ndim)))
                    kl_loss = kl_div.mean()
                    loss = policy_loss + config.train.beta * kl_loss

                    scaled_loss = loss / effective_grad_accum_steps
                    is_last_accum = (current_accumulated_steps + 1) % effective_grad_accum_steps == 0
                    if is_last_accum:
                        if mixed_precision_dtype == torch.float16:
                            scaler.scale(scaled_loss).backward()
                        else:
                            scaled_loss.backward()
                    else:
                        with transformer_ddp.no_sync():
                            if mixed_precision_dtype == torch.float16:
                                scaler.scale(scaled_loss).backward()
                            else:
                                scaled_loss.backward()
                    current_accumulated_steps += 1

                    with torch.no_grad():
                        info_accumulated["loss"].append(loss.detach())
                        info_accumulated["policy_loss"].append(policy_loss.detach())
                        info_accumulated["kl_loss"].append(kl_loss.detach())
                        info_accumulated["ratio_mean"].append(ratio.detach().mean())
                        info_accumulated["ratio_max"].append(ratio.detach().max())
                        info_accumulated["log_ratio_abs_mean"].append(log_ratio.detach().abs().mean())
                        info_accumulated["clipfrac"].append(
                            ((ratio.detach() - 1.0).abs() > FLOW_GRPO_CLIP_EPS).float().mean()
                        )

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(
                            transformer_ddp.module.parameters(), config.train.max_grad_norm,
                        )
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {
                            k: torch.mean(torch.stack(v_list)).item()
                            for k, v_list in info_accumulated.items()
                        }
                        info_tensor = torch.tensor(
                            [log_info[k] for k in sorted(log_info.keys())], device=device,
                        )
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {
                            k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))
                        }
                        if is_main_process(rank):
                            wandb.log({
                                "grad_norm": norm.item(),
                                "gradient_update_times": gradient_update_times,
                                "epoch": epoch,
                                "inner_epoch": inner_epoch,
                                **reduced_log_info,
                            }, step=global_step)

                        global_step += 1
                        info_accumulated = defaultdict(list)

                if (
                    config.train.ema and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        training_time = time.time() - training_start_time
        epoch_time = time.time() - epoch_start_time
        peak_training_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        if is_main_process(rank):
            logger.info(f"Epoch {epoch}: Training {training_time:.2f}s (peak VRAM: {peak_training_gb:.2f} GiB)")
            wandb.log({
                "timing/sampling_time": sampling_time,
                "timing/training_time": training_time,
                "timing/epoch_time_without_eval": epoch_time,
                "memory/peak_sampling_gib": peak_sampling_gb,
                "memory/peak_training_gib": peak_training_gb,
            }, step=global_step)

        if world_size > 1:
            dist.barrier()

        # Update the old policy via the same decay schedule used in train.py.
        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(
                    tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay)
                )

        # Training phase of this epoch finished — the cache is no longer
        # needed; the next epoch's cache is the authoritative one for any
        # future resume. Mirrors train.py:957-959.
        if _sample_cache_enabled:
            clear_sample_cache(config.save_dir, epoch, rank, world_size)

    # Reached only on full-run completion. In-loop saves run at epoch start, so
    # persist the FINAL trained model here — rolling + permanent epoch_<N>.
    if is_main_process(rank) and not config.debug:
        for _final_subdir in ("checkpoints", os.path.join("permanent", f"epoch_{config.num_epochs}", "checkpoints")):
            model_utils.save_ckpt(
                config.save_dir, transformer_ddp, global_step, ema,
                transformer_trainable_parameters, optimizer, scaler,
                config.num_epochs, stat_tracker, ckpt_subdir=_final_subdir,
            )
        logger.info(f"Saved FINAL checkpoint (epoch={config.num_epochs}) after full training run")

    if is_main_process(rank):
        if sample_log_future is not None:
            try:
                sample_log_future.result(timeout=30)
            except Exception as e:
                logger.warning(f"Final async sample logging task did not complete cleanly: {e}")
        wandb.finish()
    sample_log_executor.shutdown(wait=False)
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
