# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import gc
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

# This module lives in src/ but is launched as a script (torchrun src/train.py),
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
from src.dataloader import *



os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


tqdm = partial(tqdm.tqdm, dynamic_ncols=True)
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


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

    # Load wandb run ID: check a lightweight dedicated file first (written immediately
    # after wandb.init so it persists even if the job dies before the first checkpoint), then fall back to training_state.pt for backward compatibility.
    wandb_run_id_file = os.path.join(log_dir, "wandb_run_id.txt")
    if os.path.exists(wandb_run_id_file):
        with open(wandb_run_id_file) as _f: wandb_run_id = _f.read().strip() or None
    elif os.path.exists(os.path.join(log_dir, "checkpoints/training_state.pt")):
        training_state = torch.load(os.path.join(log_dir, "checkpoints/training_state.pt"), map_location="cpu")
        wandb_run_id = training_state.get("wandb_run_id")

    if is_main_process(rank):
        os.makedirs(log_dir, exist_ok=True)
        if wandb_run_id:
            # Resume the existing wandb run so all resubmits log to the same page.
            try:
                wandb.init(entity=WANDB_ENTITY, project=(getattr(config, "project", None) or config.name), name=config.run_name, config=config.to_dict(), dir=log_dir, id=wandb_run_id, resume="allow")
                logger.info(f"Resumed wandb run with id: {wandb_run_id}")
            except Exception as e:
                logger.warning(f"Failed to resume wandb run {wandb_run_id}: {e}")
                logger.info("Creating new wandb run instead...")
                wandb.init(entity=WANDB_ENTITY, project=(getattr(config, "project", None) or config.name), name=config.run_name, config=config.to_dict(), dir=log_dir)
                logger.info(f"Created new wandb run with id: {wandb.run.id}")
        else:
            # First run for this config — create a fresh wandb run.
            wandb.init(entity=WANDB_ENTITY, project=(getattr(config, "project", None) or config.name), name=config.run_name, config=config.to_dict(), dir=log_dir)
            logger.info(f"Created new wandb run with id: {wandb.run.id}")

        # Persist the run ID immediately after init so subsequent jobs (including those that fail before saving a checkpoint) always resume the same wandb page.
        with open(wandb_run_id_file, "w") as _f:
            _f.write(wandb.run.id)

        # Clean wandb shutdown on SLURM termination (wall-time TIMEOUT / scancel /
        # auto-resubmit). SLURM sends SIGTERM (then SIGKILL after KillWait) when a
        # job ends early; without an explicit finish, the run is left open and the
        # server marks it "crashed" once the heartbeat lapses — so every resubmit
        # shows as a dead red run even though training continues on the next
        # allocation. We close the run cleanly here (so it reopens as "running"
        # next time via resume="allow"), then RE-RAISE the original signal so the
        # process still dies by signal: SLURM must record TIMEOUT/CANCELLED, not an
        # exit-0 COMPLETED, otherwise auto_resubmit would treat the run as finished
        # and stop resubmitting it.
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

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    base_model = getattr(config, "base_model", "sd3")

    # --- Mixed Precision Setup ---
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

    # --- Load pipeline and models ---
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

    # Enable gradient checkpointing to reduce activation memory (~60-70% reduction)
    # at the cost of ~30% more compute. Only for large video models (hunyuan, skyreels).
    if getattr(config, "gradient_checkpointing", False):
        pipeline.transformer.enable_gradient_checkpointing()
        if is_main_process(rank):
            logger.info("Gradient checkpointing enabled on transformer")

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    train_dataloader, test_dataloader = get_dataloaders(
        config=config,
        world_size=world_size,
        rank=rank,
        num_workers=0,
    )
    train_sampler = train_dataloader.batch_sampler

    # --- Prompt Embeddings ---
    neg_prompt_embed, neg_pooled_prompt_embed = model_utils.get_negative_prompt_embeddings(text_encoders, tokenizers,)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    # Pooled embeddings may be None (e.g., HunyuanVideo 1.5 has no CLIP pooling)
    if neg_pooled_prompt_embed is not None:
        sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
        train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)
    else:
        sample_neg_pooled_prompt_embeds = None
        train_neg_pooled_prompt_embeds = None

    stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # Optional noise-region reward: separate reward weightings for high- vs
    # low-noise timesteps (config.reward_fn_{high,low}_noise, both dicts over the
    # SAME scored components). Uses an isolated per-prompt stat tracker so the
    # main reward/advantage logging path stays byte-identical. Off unless both
    # dicts are set — every existing config is unaffected.
    _reward_fn_high_noise = getattr(config, "reward_fn_high_noise", None)
    _reward_fn_low_noise = getattr(config, "reward_fn_low_noise", None)
    _noise_region_reward = _reward_fn_high_noise is not None and _reward_fn_low_noise is not None
    if _noise_region_reward:
        stat_tracker_region = PerPromptStatTracker(config.sample.global_std)
        logger.info(
            "Noise-region reward ON (t>=%s high): high=%s low=%s",
            getattr(config.train, "high_noise_threshold", 0.9),
            _reward_fn_high_noise, _reward_fn_low_noise,
        )
    # Load stat_tracker state if resuming
    if config.resume_from:
        training_state_path = os.path.join(log_dir, "checkpoints/training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location="cpu")
            if "stat_tracker_history_prompts" in training_state:
                stat_tracker.history_prompts = training_state["stat_tracker_history_prompts"]
                logger.info(f"Loaded stat_tracker with {len(stat_tracker.history_prompts)} historical prompts")

    # Reward computation is now Phase-2 sync (A14B-style); no async executor
    # needed. sample_log_executor is still used for fire-and-forget rank-0 logs.
    sample_log_executor = futures.ThreadPoolExecutor(max_workers=1)  # Async sample logging on rank0
    sample_log_future = None

    # Train!
    samples_per_epoch = config.sample.train_batch_size * world_size * config.sample.num_batches_per_epoch
    total_train_batch_size = config.train.batch_size * world_size * config.train.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"Training Method: {config.train.method}")
    logger.info(f"Advantage: {config.train.adv}")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}")
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    # Pass config.video_fps so VideoAlign writes temp mp4s at the real
    # generation fps (preserves playback-speed semantics for the reward model).
    va_dim_weights = getattr(config, 'videoalign_dimension_weights', None) or None
    _va_video_fps = getattr(config, "video_fps", None)
    reward_fn = get_reward_fn(device, config.reward_fn, videoalign_dimension_weights=va_dim_weights, video_fps=_va_video_fps)

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    
    # Check for new checkpoint format (separate policy directories)
    if config.resume_from != "" or os.path.exists(os.path.join(log_dir, "checkpoints/lora_current_policy/adapter_model.safetensors")):
        try:
            config.resume_from = os.path.join(log_dir, "checkpoints")
        except:
            assert config.resume_from
        
        logger.info(f"Resuming from {config.resume_from} (new format with separate policies)")
        
        # Load current policy adapter
        lora_current_path = os.path.join(config.resume_from, "lora_current_policy")
        if os.path.exists(lora_current_path):
            transformer_ddp.module.load_adapter(lora_current_path, adapter_name="default", is_trainable=True)
            logger.info(f"Loaded current policy from {lora_current_path}")
        
        # Load old policy adapter
        lora_old_path = os.path.join(config.resume_from, "lora_old_policy")
        if os.path.exists(lora_old_path):
            transformer_ddp.module.load_adapter(lora_old_path, adapter_name="old", is_trainable=False)
            logger.info(f"Loaded old policy from {lora_old_path}")
        
        # Load optimizer and scaler
        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            try:
                # Load to CPU first to avoid OOM, optimizer will handle device placement
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
                logger.info("Loaded optimizer state")
            except (RuntimeError, EOFError, zipfile.BadZipFile) as e:
                logger.warning(f"Failed to load optimizer state from {opt_path}: {e}")
                logger.warning("Starting with fresh optimizer state. Consider removing corrupted checkpoint.")

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            try:
                scaler.load_state_dict(torch.load(scaler_path, map_location="cpu"))
                logger.info("Loaded scaler state")
            except (RuntimeError, EOFError, zipfile.BadZipFile) as e:
                logger.warning(f"Failed to load scaler state from {scaler_path}: {e}")
                logger.warning("Starting with fresh scaler state")
        
        # Load global_step, epoch, and stat_tracker state
        training_state_path = os.path.join(config.resume_from, "training_state.pt")
        if os.path.exists(training_state_path):
            training_state = torch.load(training_state_path, map_location=device)
            global_step = training_state["global_step"]
            first_epoch = training_state.get("epoch", 0)
            logger.info(f"Loaded global_step: {global_step}, epoch: {first_epoch}")
            # wandb_run_id is already loaded earlier for wandb.init

    # Initialize EMA
    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)
        
        # Load EMA policy from checkpoint if it exists (new format only)
        if config.resume_from:
            lora_ema_path = os.path.join(config.resume_from, "lora_ema_policy")
            if os.path.exists(lora_ema_path):
                # Create a temporary adapter to load EMA weights
                transformer_ddp.module.add_adapter("temp_ema_load", transformer_lora_config)
                transformer_ddp.module.load_adapter(lora_ema_path, adapter_name="temp_ema_load", is_trainable=False)
                transformer_ddp.module.set_adapter("temp_ema_load")
                
                # Get the parameters from the temp_ema_load adapter
                temp_ema_params = [p for p in transformer_ddp.module.parameters() if p.requires_grad]
                
                # Copy EMA weights to the EMA wrapper's ema_parameters (correct attribute name)
                for ema_param, loaded_param in zip(ema.ema_parameters, temp_ema_params):
                    ema_param.data.copy_(loaded_param.data)
                
                # Clean up: switch back to default adapter and delete temp adapter
                transformer_ddp.module.set_adapter("default")
                transformer_ddp.module.delete_adapter("temp_ema_load")
                
                logger.info(f"Loaded EMA policy from {lora_ema_path}")
            else:
                logger.info("No EMA checkpoint found, EMA will start from current policy weights")

    time_fraction = float(getattr(config.train, "time_fraction", 1.0))
    assert 0.0 < time_fraction <= 1.0, f"time_fraction must be in (0, 1], got {time_fraction}"
    active_region = max(1, int(config.sample.num_steps * time_fraction))
    time_fraction_start_idx = config.sample.num_steps - active_region
    num_train_timesteps = max(1, int(active_region * config.train.timestep_fraction))
    if is_main_process(rank):
        logger.info(
            f"time_fraction={time_fraction} -> start_idx={time_fraction_start_idx}, "
            f"active_region={active_region}, num_train_timesteps={num_train_timesteps}"
        )

    # Synchronize all ranks after checkpoint loading to prevent race conditions:
    # rank 0 could start saving new checkpoints (overwriting files) while slower
    # ranks are still loading the same files via mmap, causing SIGBUS.
    if world_size > 1:
        dist.barrier()

    # --- Sync global_step with wandb counter on resume ---
    # The checkpoint saves global_step at epoch start, but the previous run
    # may have logged more training steps before crashing.  Auto-resubmit can
    # also cause multiple resumes, each advancing wandb's internal counter
    # further.  If global_step falls behind the counter, every subsequent
    # wandb.log(..., step=global_step) is silently dropped — including the
    # per-epoch reward breakdown.  Fix: advance global_step to match.
    if config.resume_from and is_main_process(rank):
        if wandb.run is not None:
            try:
                _wb_step = wandb.run.step  # next expected step
            except AttributeError:
                _wb_step = global_step
            if global_step < _wb_step:
                logger.info(
                    f"wandb counter ({_wb_step}) ahead of checkpoint "
                    f"global_step ({global_step}); advancing to match"
                )
                global_step = _wb_step
    if config.resume_from and world_size > 1:
        _gs = torch.tensor([global_step], device=device, dtype=torch.long)
        dist.broadcast(_gs, src=0)
        global_step = int(_gs.item())

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    # Only copy current policy to old policy if NOT resuming from checkpoint
    # When resuming, we've already loaded separate old policy weights
    if not config.resume_from:
        for src_param, tgt_param in zip(
            transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
        ):
            tgt_param.data.copy_(src_param.detach().data)
            assert src_param is not tgt_param
        logger.info("Initialized old policy from current policy (fresh training)")
    else:
        logger.info("Using loaded old policy from checkpoint (not copying from current)")


    # Timing data from the previous epoch, merged into the next epoch's
    # combined wandb.log call.  Logging timing as a separate wandb.log
    # at the end of the epoch advances the step counter and silently eats
    # the next epoch's per-epoch reward log (same step, already committed).
    _pending_timing = {}

    ####################
    #### START LOOP ####
    ####################
    for epoch in range(first_epoch, config.num_epochs):
        epoch_start_time = time.time()
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # --- Sample-cache fast-path ---
        # On resume, if the previous job persisted this epoch's samples, skip sampling and go straight to training. See save_sample_cache load_sample_cache in src/utils/__init__.py.
        _sample_cache_enabled = getattr(config, "save_sample_cache", True) and not config.debug
        _cache_hit = False
        if _sample_cache_enabled and epoch == first_epoch and config.resume_from:
            _cached = load_sample_cache(config.save_dir, epoch, rank, world_size, logger)
            if _cached is not None:
                _cache_hit = True
                if is_main_process(rank):
                    logger.info(f"Epoch {epoch}: loaded sample cache, skipping sampling phase")
                pipeline.vae.to("cpu")
                if hasattr(reward_fn, 'to_cpu'):
                    reward_fn.to_cpu()
                # Cache is always saved from CPU. If this run keeps samples on
                # GPU during training (cpu_offload_samples=False), move them
                # back; otherwise leave on CPU like the normal path.
                if not getattr(config, "cpu_offload_samples", False):
                    for _k, _v in list(_cached.items()):
                        if isinstance(_v, torch.Tensor):
                            _cached[_k] = _v.to(device)
                collated_samples = _cached
                sampling_time = 0.0
                peak_sampling_gb = 0.0
                gc.collect()
                torch.cuda.empty_cache()

                # Re-emit per-epoch reward/advantage log on cache-resume.
                # The non-cache path's wandb.log lives inside `if not
                # _cache_hit:` below, so without this every resume cycle
                # leaves a hole in the wandb reward curve.
                # `gather_tensor_to_all` uses NCCL all_gather, which needs
                # CUDA tensors. With cpu_offload_samples=True the cached
                # tensors live on CPU, so move a temporary copy to device
                # for the gather (don't mutate _cached itself — downstream
                # training code expects the offload-side placement).
                _rewards_gathered = (
                    gather_tensor_to_all(_cached["rewards"].to(device), world_size).cpu().numpy()
                )
                _advantages_gathered = (
                    gather_tensor_to_all(_cached["advantages"].to(device), world_size).cpu().numpy()
                )
                if is_main_process(rank):
                    _resume_log = {
                        "epoch": epoch,
                        "reward/avg": float(_rewards_gathered.mean()),
                        "reward/avg_std": float(_rewards_gathered.std()),
                        "advantages_mean": float(_advantages_gathered.mean()),
                        "advantages_max": float(_advantages_gathered.max()),
                        "advantages_min": float(_advantages_gathered.min()),
                        "advantages_std": float(_advantages_gathered.std()),
                        "advantages_abs_mean": float(np.abs(_advantages_gathered).mean()),
                    }
                    # Re-emit the per-component reward breakdown from the sidecar
                    # so cache-hit resume epochs don't leave a hole in the
                    # per-component reward curves (the cache only stores the
                    # averaged reward). None for caches written before the
                    # sidecar existed.
                    _bd = load_reward_breakdown(config.save_dir, epoch)
                    if _bd:
                        _resume_log.update(_bd)
                    # global_step was synced to wandb's counter above, so
                    # step=global_step is safe and won't desync the counter.
                    wandb.log(_resume_log, step=global_step)

        ##################
        #### SAMPLING ####
        ##################
        if not _cache_hit:
            sampling_start_time = time.time()
            torch.cuda.reset_peak_memory_stats(device)
            # Phase 1: foundation on GPU, reward models stay on CPU. The reward
            # forward pass is deferred until the Phase-2 reward block below; this
            # avoids the OOM from the old interleaved layout (HPSv3 + VideoAlign
            # ~42 GB of weights + foundation ~15 GB + per-step transient
            # activations ~10-15 GB > 80 GB H100).
            #
            # Bring the WHOLE foundation back to GPU here (not just VAE).
            # Phase 2 below moves transformer/VAE/text_encoder all to CPU; the
            # end-of-Phase-2 block only restores transformer for training.
            # Without restoring text_encoder here, epoch>=1 sampling hits a
            # device-mismatch crash inside `pipeline.encode_prompt` because
            # the tokenized inputs land on GPU while text_encoder is still CPU.
            pipeline.vae.to(device)
            if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
                pipeline.text_encoder.to(device)
            if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
                pipeline.transformer.to(device)
            if hasattr(reward_fn, 'to_cpu'):
                reward_fn.to_cpu()
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

                # Each utils class extracts its own conditioning (precomputed embeddings, action ids, reference images, ...) from the
                # dataloader yield and stashes any model-specific state on itself for build_sample_kwargs to pick up.
                batch_inputs = model_utils.extract_dataloader_metadata(
                    prompts, prompt_metadata, text_encoders, tokenizers,
                )
                prompt_embeds = batch_inputs["prompt_embeds"]
                pooled_prompt_embeds = batch_inputs["pooled_prompt_embeds"]
                prompt_ids = batch_inputs["prompt_ids"]

                # if i == 0 and epoch % config.eval_freq == 0 and not config.debug:
                #     model_utils.eval_fn(
                #         pipeline,
                #         test_dataloader,
                #         text_encoders,
                #         tokenizers,
                #         global_step,
                #         reward_fn,
                #         executor,
                #         mixed_precision_dtype,
                #         ema,
                #         transformer_trainable_parameters,
                #         tqdm,
                #     )

                if i == 0 and is_main_process(rank) and not config.debug:
                    # Rolling checkpoint: overwritten every save_freq epochs.
                    # This is what resume reads (<save_dir>/checkpoints/).
                    if epoch % config.save_freq == 0:
                        model_utils.save_ckpt(
                            config.save_dir,
                            transformer_ddp,
                            global_step,
                            ema,
                            transformer_trainable_parameters,
                            optimizer,
                            scaler,
                            epoch,
                            stat_tracker,
                        )

                    # Permanent epoch snapshots: kept forever (never overwritten)
                    # every num_epochs // 10 epochs. Written to
                    # <save_dir>/permanent/epoch_<N>/checkpoints/ so each is
                    # loadable by evaluation.py exactly like the rolling ckpt
                    # (point eval at <save_dir>/permanent/epoch_<N>). The FINAL
                    # trained model is saved separately after the loop.
                    perm_interval = max(1, config.num_epochs // 10)
                    if epoch % perm_interval == 0:
                        model_utils.save_ckpt(
                            config.save_dir,
                            transformer_ddp,
                            global_step,
                            ema,
                            transformer_trainable_parameters,
                            optimizer,
                            scaler,
                            epoch,
                            stat_tracker,
                            ckpt_subdir=os.path.join("permanent", f"epoch_{epoch}", "checkpoints"),
                        )

                offload_during_decode = getattr(config, "offload_transformer_during_decode", False)
    
                transformer_ddp.module.set_adapter("old")
                # When time_fraction < 1.0, disable the LoRA layers before sampling
                # so the first (1 - time_fraction) fraction of steps runs through
                # the base model. run_sampling() re-enables them at the boundary.
                if time_fraction_start_idx > 0:
                    transformer_ddp.module.disable_adapter_layers()
                with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                    with torch.no_grad():
                        sample_kwargs = model_utils.build_sample_kwargs(
                            prompt_embeds,
                            pooled_prompt_embeds,
                            sample_neg_prompt_embeds,
                            sample_neg_pooled_prompt_embeds,
                            len(prompts),
                        )
                        if offload_during_decode:
                            sample_kwargs["skip_decode"] = True
                        if time_fraction_start_idx > 0:
                            sample_kwargs["time_fraction_start_idx"] = time_fraction_start_idx
                            sample_kwargs["peft_model_for_toggle"] = transformer_ddp.module
                        images, latents, _ = model_utils.pipeline_with_logprob(pipeline, **sample_kwargs)

                # Re-enable adapter layers after sampling so the training phase
                # (which expects set_adapter + disable_adapter context manager semantics) operates on enabled LoRA layers.
                if time_fraction_start_idx > 0:
                    transformer_ddp.module.enable_adapter_layers()
                transformer_ddp.module.set_adapter("default")
    
                latents = torch.stack(latents, dim=1)
                latents_clean = latents[:, -1].clone()
                if config.train.xt_from_latents:
                    latents_trajectory = latents  # (B, num_steps+1, ...), keep for training
                else:
                    del latents  # Free intermediate latents (only final needed)
    
                # Offload transformer to CPU before VAE decode to free VRAM
                if offload_during_decode:
                    pipeline.transformer.to("cpu")
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        images = model_utils.decode_latents(pipeline, latents_clean)
    
                timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)
                # A14B-style sequentialization: stash decoded images on CPU and
                # defer reward computation to Phase 2 (after all sampling). This
                # frees the GPU window between batches for transformer and VAE
                # only — reward models never share VRAM with sampling.
                images_cpu = images.detach().to("cpu", copy=True)
                # Keep last batch for logging, free the rest
                last_images = images
                last_prompts = prompts
                del images
    
                # Reload transformer to GPU after VAE decode. We always reload
                # (including after the final batch) so Phase 3 starts with the
                # transformer fully resident on GPU. Skipping the reload on the
                # last batch — the "optimization" the original code attempted —
                # crashes Wan2.2-I2V-A14B's Phase-3 setup: `PeftModel.to(device)`
                # there does not reliably recurse into every wrapped submodule
                # (e.g. `patch_embedding`), so the first training forward sees a
                # CPU weight on a CUDA input. One extra `.to(device)` per epoch
                # is negligible compared to the freed VRAM during decode.
                if offload_during_decode:
                    pipeline.transformer.to(device)
                    torch.cuda.empty_cache()
    
                # Per-sample sampler-timestep index, shuffled in lockstep with
                # `timesteps` during training. Lets us bucket metrics by the
                # original position in the sampler trajectory even after the
                # per-sample random time permutation.
                timestep_idx = (
                    torch.arange(timesteps.shape[1], device=device, dtype=torch.long)
                    .unsqueeze(0).expand(timesteps.shape[0], -1).contiguous()
                )
    
                sample_data = {
                    "prompt_ids": prompt_ids,
                    "prompts_text": prompts,  # raw text for stat tracking (used when prompt_ids is None)
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "timestep_idx": timestep_idx,
                    "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "latents_clean": latents_clean,
                    # Phase-2 inputs stashed on CPU (A14B-style swap).
                    "images_cpu": images_cpu,
                    "reward_prompts": prompts,
                    "reward_prompt_metadata": prompt_metadata,
                }
                if config.train.xt_from_latents:
                    # Drop the final clean latent so index aligns with `timesteps`
                    # (sampler returns num_steps+1 entries, timesteps has num_steps).
                    # After this slice, latents[:, j] is the sampler input at sigma=timesteps[:, j].
                    sample_data["latents"] = latents_trajectory[:, :-1].contiguous()
                    del latents_trajectory
                # Per-model conditioning produced by pipeline_with_logprob
                # (first_frame_latents, i2v_condition, prompt_attention_mask,
                # action_ids, ...) is appended by each utils class.
                model_utils.collect_sample_artifacts(sample_data)

                # Drop the initial (adapter-disabled) prefix so training only
                # iterates over timesteps where the fine-tuned adapter was active.
                # `timestep_idx` retains absolute sampler positions so downstream
                # per-sampler-timestep bucketing still works.
                if time_fraction_start_idx > 0:
                    for _k in ("timesteps", "timestep_idx", "next_timesteps"):
                        sample_data[_k] = sample_data[_k][:, time_fraction_start_idx:].contiguous()
                    if config.train.xt_from_latents and "latents" in sample_data:
                        sample_data["latents"] = sample_data["latents"][:, time_fraction_start_idx:].contiguous()
                samples_data_list.append(sample_data)
    
            # ─── Phase 2: reward computation ───────────────────────────────
            # Foundation off GPU, rewards onto GPU. With offload_during_decode
            # the transformer is already on CPU after the last batch; bring VAE
            # and text_encoder along so the entire 41 GB reward stack fits on
            # H100 without contention.
            if hasattr(pipeline, "transformer") and pipeline.transformer is not None:
                pipeline.transformer.to("cpu")
            if hasattr(pipeline, "vae") and pipeline.vae is not None:
                pipeline.vae.to("cpu")
            if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
                pipeline.text_encoder.to("cpu")
            torch.cuda.empty_cache()
            if hasattr(reward_fn, 'to_gpu'):
                reward_fn.to_gpu()

            reward_wait_per_batch = []
            for sample_item in tqdm(
                samples_data_list, desc="Scoring rewards", disable=not is_main_process(rank), position=0
            ):
                _rw_t0 = time.perf_counter()
                images_gpu = sample_item.pop("images_cpu").to(device, non_blocking=True)
                rewards, reward_metadata = reward_fn(
                    images_gpu,
                    sample_item.pop("reward_prompts"),
                    sample_item.pop("reward_prompt_metadata"),
                    only_strict=True,
                )
                reward_wait_per_batch.append(time.perf_counter() - _rw_t0)
                sample_item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
                del images_gpu

            # End of Phase 2: rewards back to CPU, transformer back to GPU
            # for the upcoming training phase. VAE and text_encoder stay on
            # CPU (training only needs the transformer; the post-sampling
            # cleanup below would otherwise immediately re-offload VAE).
            if hasattr(reward_fn, 'to_cpu'):
                reward_fn.to_cpu()
            torch.cuda.empty_cache()
            if hasattr(pipeline, "transformer"):
                pipeline.transformer.to(device)

            # Per-rank reward-wait spread: if one rank is consistently many
            # minutes slower than its peers, that's the precursor to the
            # 30-min NCCL allgather hang (VideoAlign Qwen2-VL forward
            # contending with WAN sampling on a shared GPU). The hung rank
            # never reaches the gather, so this only warns *before* the cliff,
            # not at it.
            if reward_wait_per_batch:
                _local_total = float(sum(reward_wait_per_batch))
                _local_max = float(max(reward_wait_per_batch))
                _local_stats = torch.tensor([_local_total, _local_max], device=device, dtype=torch.float32)
                _all_stats = gather_tensor_to_all(_local_stats, world_size).reshape(world_size, 2)
                if is_main_process(rank):
                    totals = _all_stats[:, 0].tolist()
                    maxes = _all_stats[:, 1].tolist()
                    spread = max(totals) - min(totals)
                    logger.info(
                        "[reward latency] per-rank total(s)=[%s] batch-max(s)=[%s] spread=%.1fs",
                        ", ".join(f"{t:.1f}" for t in totals),
                        ", ".join(f"{m:.1f}" for m in maxes),
                        spread,
                    )
    
            # Collate samples
            def _collate_field(key, samples):
                first = samples[0][key]
                if first is None:
                    return None
                if isinstance(first, list):
                    # Non-tensor fields (e.g., prompts_text): concatenate lists
                    result = []
                    for s in samples:
                        result.extend(s[key])
                    return result
                if isinstance(first, dict):
                    return {sk: torch.cat([s[key][sk] for s in samples], dim=0) for sk in first}
                return torch.cat([s[key] for s in samples], dim=0)
    
            collated_samples = {
                k: _collate_field(k, samples_data_list)
                for k in samples_data_list[0].keys()
            }
            del samples_data_list  # Free per-batch sample dicts (now collated)
    
            sampling_time = time.time() - sampling_start_time
            peak_sampling_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            if is_main_process(rank):
                logger.info(f"Epoch {epoch}: Sampling took {sampling_time:.2f} seconds (peak VRAM: {peak_sampling_gb:.2f} GiB)")
    
            # Logging images (main process, async to avoid blocking collectives on other ranks)
            if epoch % config.sample_log_freq == 0 and is_main_process(rank):
                if sample_log_future is None or sample_log_future.done():
                    num_log = len(last_images) if hasattr(last_images, '__len__') else last_images.shape[0]
                    rewards_to_log = collated_samples["rewards"]["avg"][-num_log:].cpu()
                    sample_log_future = sample_log_executor.submit(
                        model_utils.log_train_samples,
                        last_images,
                        last_prompts,
                        rewards_to_log,
                        global_step,
                    )
                else:
                    logger.warning(
                        "Skipping train sample logging at epoch %s because previous logging task is still running.",
                        epoch,
                    )
            collated_samples["rewards"]["avg"] = (
                collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
            )

            # Region rewards: re-aggregate the already-scored per-component
            # rewards under the high- and low-noise weight dicts (no extra
            # reward-model compute). Each becomes [N, num_train_timesteps] so it
            # gathers / distributes exactly like "avg".
            if _noise_region_reward:
                for _rk, _wts in (("avg_low_noise", _reward_fn_low_noise),
                                  ("avg_high_noise", _reward_fn_high_noise)):
                    _combined = None
                    for _cname, _w in _wts.items():
                        if _cname not in collated_samples["rewards"]:
                            raise KeyError(
                                f"Region reward component '{_cname}' is not scored. "
                                f"Add it to config.reward_fn (scored: "
                                f"{sorted(collated_samples['rewards'].keys())})."
                            )
                        _term = collated_samples["rewards"][_cname] * float(_w)
                        _combined = _term if _combined is None else _combined + _term
                    collated_samples["rewards"][_rk] = (
                        _combined.unsqueeze(1).repeat(1, num_train_timesteps)
                    )

            # Gather rewards across processes
            gathered_rewards_dict = {}
            for key, value_tensor in collated_samples["rewards"].items():
                gathered_rewards_dict[key] = gather_tensor_to_all(value_tensor, world_size).numpy()

            # --- Build combined per-epoch wandb log dict ---
            # wandb silently drops data when wandb.log() is called multiple
            # times at the same step= value (the first call commits the step
            # and advances the internal counter; subsequent calls at the same
            # step see counter > step and are discarded). Accumulate ALL
            # per-epoch data (rewards, stat-tracker, advantages, timing from
            # the previous epoch) into one dict and log it in a single call.
            _epoch_log = {"epoch": epoch}
            _epoch_log.update(_pending_timing)
            _pending_timing = {}

            if is_main_process(rank):
                for _k, _v in gathered_rewards_dict.items():
                    if "_strict_accuracy" in _k or "_accuracy" in _k:
                        _epoch_log[f"reward/{_k}"] = _v.mean()
                        continue
                    _epoch_log[f"reward/{_k}"] = _v.mean()
                    _epoch_log[f"reward/{_k}_std"] = _v.std()

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
                if getattr(config.train, "whiten_rewards", False):
                    # Per-component reward whitening: standardize each reward component
                    # across the gathered batch (unit std) BEFORE the weighted sum, so a
                    # component's raw scale no longer dominates the advantage (e.g. the
                    # bounded [0,1] dynamic_degree_bump vs the wider-range HPS/VideoAlign
                    # terms). Weights then act as true importance shares. Opt-in via
                    # config.train.whiten_rewards (default off → identical to before).
                    _wsum = None
                    for _cname, _w in config.reward_fn.items():
                        if _cname not in gathered_rewards_dict:
                            raise KeyError(
                                f"whiten_rewards: reward component '{_cname}' not in gathered "
                                f"rewards ({sorted(gathered_rewards_dict.keys())})"
                            )
                        _comp = gathered_rewards_dict[_cname].astype(np.float64)
                        _comp = _comp.reshape(_comp.shape[0], -1).mean(axis=1)  # [N] per-sample
                        _z = (_comp - _comp.mean()) / (_comp.std() + 1e-8)
                        _term = float(_w) * _z
                        _wsum = _term if _wsum is None else _wsum + _term
                    _whitened_avg = np.repeat(_wsum[:, None], num_train_timesteps, axis=1)  # [N,T]
                    advantages = stat_tracker.update(prompts_all_decoded, _whitened_avg, config.train)
                else:
                    advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"], config.train)

                # Region advantages: group-normalize the two region rewards in a
                # single multi-column update (the tracker normalizes per-prompt
                # over axis 0, keeping trailing columns). Column 0 = low-noise,
                # 1 = high-noise. Isolated tracker, cleared each epoch.
                if _noise_region_reward:
                    _region_rewards = np.stack(
                        [gathered_rewards_dict["avg_low_noise"][:, 0],
                         gathered_rewards_dict["avg_high_noise"][:, 0]],
                        axis=1,
                    )  # [N, 2] per-sample scalars
                    region_advantages = stat_tracker_region.update(
                        prompts_all_decoded, _region_rewards, config.train
                    )  # [N, 2]
                    stat_tracker_region.clear()

                if is_main_process(rank):
                    group_size, trained_prompt_num = stat_tracker.get_stats()
                    zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all_decoded, gathered_rewards_dict)
                    # mean_top_* must reflect the RAW aggregate reward. When
                    # whiten_rewards is on, stat_tracker holds the zero-mean
                    # whitened surrogate used for advantages, so its mean_top_100
                    # is ≡0 and the other percentiles measure z-score tails, not
                    # reward magnitude. Recompute from the raw reward via a
                    # throwaway tracker (update() only touches its own instance).
                    if getattr(config.train, "whiten_rewards", False):
                        _reward_log_tracker = PerPromptStatTracker(config.sample.global_std)
                        _reward_log_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"], config.train)
                    else:
                        _reward_log_tracker = stat_tracker
                    _epoch_log.update({
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "reward/zero_std_ratio": zero_std_ratio,
                        "reward/std_mean": reward_std_mean,
                        "reward/mean_top_100": _reward_log_tracker.get_mean_of_top_rewards(100),
                        "reward/mean_top_75": _reward_log_tracker.get_mean_of_top_rewards(75),
                        "reward/mean_top_50": _reward_log_tracker.get_mean_of_top_rewards(50),
                        "reward/mean_top_25": _reward_log_tracker.get_mean_of_top_rewards(25),
                        "reward/mean_top_10": _reward_log_tracker.get_mean_of_top_rewards(10),
                    })
                stat_tracker.clear()
            else:
                raise NotImplementedError

            # Distribute advantages back to processes
            samples_per_gpu = collated_samples["timesteps"].shape[0]
            avg_rewards_all = gathered_rewards_dict["avg"]
            if advantages.ndim == 1:
                advantages = advantages[:, None]

            if advantages.shape[0] == world_size * samples_per_gpu:
                collated_samples["advantages"] = torch.from_numpy(
                    advantages.reshape(world_size, samples_per_gpu, -1)[rank]
                ).to(device)
                collated_samples["rewards"] = torch.from_numpy(
                    avg_rewards_all.reshape(world_size, samples_per_gpu, -1)[rank]
                ).to(device)
                # Region advantages -> this rank, broadcast across the timestep
                # axis (per-sample scalar; compute_loss selects per step by the
                # actual noise level, so the timestep axis stays constant here).
                if _noise_region_reward:
                    for _key, _col in (("advantages_low_noise", 0), ("advantages_high_noise", 1)):
                        _a = region_advantages[:, _col:_col + 1]  # [N, 1]
                        collated_samples[_key] = torch.from_numpy(
                            _a.reshape(world_size, samples_per_gpu, -1)[rank]
                        ).to(device).repeat(1, num_train_timesteps)
            else:
                assert False

            # Per-sample prompt-group ids: a stable integer per distinct prompt,
            # aligned with prompts_all_decoded (same rank-major order as the
            # gathered advantages). The pwvm loss uses these to restrict its
            # cross-sample velocity matching to same-prompt siblings within a
            # micro-batch. Constant along the timestep axis, so this is a
            # per-sample scalar that rides the sample shuffle (not the time
            # shuffle) like `advantages`.
            _group_lut = {}
            _group_ids_all = np.array(
                [_group_lut.setdefault(_p, len(_group_lut)) for _p in prompts_all_decoded],
                dtype=np.int64,
            )
            collated_samples["prompt_group_ids"] = torch.from_numpy(
                _group_ids_all.reshape(world_size, samples_per_gpu)[rank]
            ).to(device)

            if is_main_process(rank):
                _adv_flat_np = np.asarray(advantages, dtype=np.float64).reshape(-1)
                _adv_clip_max = float(getattr(config.train, "adv_clip_max", 5.0))
                _adv_frac_saturated = (
                    float(np.mean(np.abs(_adv_flat_np) >= _adv_clip_max))
                    if _adv_clip_max > 0 else 0.0
                )
                logger.info(f"Advantages abs mean: {np.abs(_adv_flat_np).mean()}")
                _epoch_log.update({
                    "advantages_mean": float(_adv_flat_np.mean()),
                    "advantages_max": float(_adv_flat_np.max()),
                    "advantages_min": float(_adv_flat_np.min()),
                    "advantages_std": float(_adv_flat_np.std()),
                    "advantages_abs_mean": float(np.abs(_adv_flat_np).mean()),
                    "advantages_frac_saturated": _adv_frac_saturated,
                })
                # Single wandb.log for all per-epoch data.
                wandb.log(_epoch_log, step=global_step)
            collated_samples.pop("prompt_ids", None)
            collated_samples.pop("prompts_text", None)
            del last_images, last_prompts  # Free logging references
    
            # Ensure transformer is back on GPU after sampling (may be on CPU if
            # offload_transformer_during_decode was used for the last batch)
            if getattr(config, "offload_transformer_during_decode", False):
                if next(pipeline.transformer.parameters()).device.type == "cpu":
                    pipeline.transformer.to(device)
                    torch.cuda.empty_cache()
    
            # Free GPU memory before training: offload VAE and reward models
            # (only transformer needed for training phase)
            pipeline.vae.to("cpu")
            if hasattr(reward_fn, 'to_cpu'):
                reward_fn.to_cpu()
    
            # Optionally move sample data to CPU — only 1 micro-batch is on GPU at a time
            # during training. Frees ~10-15 GB but adds CPU-GPU transfer per micro-batch.
            if getattr(config, "cpu_offload_samples", False):
                for k, v in collated_samples.items():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        collated_samples[k] = v.cpu()
            gc.collect()
            torch.cuda.empty_cache()

            # Persist the post-sampling dict so a job timing out mid-training
            # can resume straight into the training phase next time.
            if _sample_cache_enabled:
                save_sample_cache(config.save_dir, epoch, rank, world_size, collated_samples, logger)
                # Persist the per-component reward breakdown next to the cache so
                # a cache-hit resume can re-emit the full reward curves (the cache
                # itself only keeps the averaged reward). Rank-0 only.
                if is_main_process(rank):
                    _bd = {k: v for k, v in _epoch_log.items()
                           if k.startswith("reward/") and k not in ("reward/avg", "reward/avg_std")}
                    if "group_size" in _epoch_log:
                        _bd["group_size"] = _epoch_log["group_size"]
                    if "trained_prompt_num" in _epoch_log:
                        _bd["trained_prompt_num"] = _epoch_log["trained_prompt_num"]
                    if _bd:
                        save_reward_breakdown(config.save_dir, epoch, _bd)

        ##################
        #### TRAINING ####
        ##################
        total_batch_size, num_timesteps = collated_samples["timesteps"].shape
        num_batches = config.sample.num_batches_per_epoch * config.sample.train_batch_size // config.train.batch_size
        training_start_time = time.time()
        torch.cuda.reset_peak_memory_stats(device)
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps
        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            # When samples are offloaded to CPU, indices must also be on CPU
            perm_device = "cpu" if getattr(config, "cpu_offload_samples", False) else device
            perm = torch.randperm(total_batch_size, device=perm_device)
            shuffled_samples = {k: v[perm] for k, v in collated_samples.items()}

            perms_time = torch.stack([torch.randperm(num_timesteps, device=perm_device) for _ in range(total_batch_size)])
            time_shuffle_keys = ["timesteps", "timestep_idx", "next_timesteps"]
            if config.train.xt_from_latents:
                time_shuffle_keys.append("latents")
            for key in time_shuffle_keys:
                shuffled_samples[key] = shuffled_samples[key][torch.arange(total_batch_size, device=perm_device)[:, None], perms_time]

            training_batch_size = total_batch_size // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(list)  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                # Move micro-batch to GPU if samples were offloaded to CPU
                if getattr(config, "cpu_offload_samples", False):
                    train_sample_batch = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in train_sample_batch.items()
                    }
                embeds = train_sample_batch["prompt_embeds"]
                pooled_embeds = train_sample_batch["pooled_prompt_embeds"]

                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx

                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        loss, loss_terms = model_utils.compute_loss(
                            transformer_ddp,
                            train_sample_batch,
                            j_idx,
                            embeds,
                            pooled_embeds,
                            train_neg_prompt_embeds,
                            train_neg_pooled_prompt_embeds,
                        )

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    # DDP all-reduces on every backward by default; wrap all
                    # but the final accumulation-cycle backward in no_sync()
                    # to match accelerator.accumulate() semantics used by
                    # Flow-GRPO. Final grads are numerically identical.
                    is_last_accum = (current_accumulated_steps + 1) % effective_grad_accum_steps == 0
                    if is_last_accum:
                        if mixed_precision_dtype == torch.float16:
                            scaler.scale(scaled_loss).backward()  # one accumulation
                        else:
                            scaled_loss.backward()
                    else:
                        with transformer_ddp.no_sync():
                            if mixed_precision_dtype == torch.float16:
                                scaler.scale(scaled_loss).backward()
                            else:
                                scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(transformer_ddp.module.parameters(), config.train.max_grad_norm)
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {k: torch.mean(torch.stack(v_list)).item() for k, v_list in info_accumulated.items()}
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info.keys())], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {k: info_tensor[ki].item() for ki, k in enumerate(sorted(log_info.keys()))}

                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "gradient_norm/grad_norm": norm.item(),
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                },
                                step=global_step,
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(list)  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        training_time = time.time() - training_start_time
        epoch_time = time.time() - epoch_start_time
        peak_training_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        if is_main_process(rank):
            logger.info(f"Epoch {epoch}: Training took {training_time:.2f} seconds (peak VRAM: {peak_training_gb:.2f} GiB)")
            logger.info(f"Epoch {epoch}: Total epoch time (without eval) {epoch_time:.2f} seconds")
            # Save timing for the next epoch's combined wandb.log call.
            # A separate wandb.log here would advance the step counter and
            # silently eat the next epoch's per-epoch reward log.
            _pending_timing = {
                "timing/sampling_time": sampling_time,
                "timing/training_time": training_time,
                "timing/epoch_time_without_eval": epoch_time,
                "memory/peak_sampling_gib": peak_sampling_gb,
                "memory/peak_training_gib": peak_training_gb,
            }

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

        # Training phase of this epoch finished successfully — the sample
        # cache for this epoch is no longer needed. Delete it so the next
        # epoch's cache is the authoritative one for any future resume.
        if _sample_cache_enabled:
            clear_sample_cache(config.save_dir, epoch, rank, world_size)

    # Reached only when the epoch loop completes the full run (a wall-time/crash
    # kill terminates the process before here). The in-loop saves run at the
    # START of each epoch, so the last epoch's updates are never checkpointed by
    # them — persist the FINAL trained model now, as both the rolling checkpoint
    # and a permanent epoch_<num_epochs> snapshot.
    if is_main_process(rank) and not config.debug:
        for _final_subdir in ("checkpoints", os.path.join("permanent", f"epoch_{config.num_epochs}", "checkpoints")):
            model_utils.save_ckpt(
                config.save_dir,
                transformer_ddp,
                global_step,
                ema,
                transformer_trainable_parameters,
                optimizer,
                scaler,
                config.num_epochs,
                stat_tracker,
                ckpt_subdir=_final_subdir,
            )
        logger.info(f"Saved FINAL checkpoint (epoch={config.num_epochs}) after full training run")

    if is_main_process(rank):
        if sample_log_future is not None:
            try:
                sample_log_future.result(timeout=30)
            except Exception as e:
                logger.warning(f"Final async sample logging task did not complete cleanly: {e}")
        # Flush the last epoch's timing data that was deferred.
        if _pending_timing:
            wandb.log(_pending_timing)
        wandb.finish()
    sample_log_executor.shutdown(wait=False)
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)