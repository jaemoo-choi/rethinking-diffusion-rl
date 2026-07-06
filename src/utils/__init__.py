import os
import shutil
import numpy as np
import random
import json
from datetime import timedelta
import torch.distributed as dist
import torch
import src.reward


import numpy as np
from collections import deque
import torch


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Compatibility helper for third-party imports (e.g., DINO via torch.hub)."""
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


# ELBO computation
def compute_elbo(t_expanded, v_prediction, xt, x0, noise, method, guidance_scale=1.0):
    if method == "adaptive":
        forward_x0_prediction = xt - t_expanded * v_prediction
        with torch.no_grad():
            weight_factor = (
                torch.abs(forward_x0_prediction.double() - x0.double())
                .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                .clip(min=0.00001)
            )
        elbo = - ((forward_x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))
        # G-normalize so ELBO magnitude is independent of classifier-free guidance. 
        # Under cached_uncond, v_θ - v_old = G·Δv_text, so err and thus adaptive ELBO scale linearly in G; dividing by G makes
        # forward_ELBO, old_ELBO, and logratio G-invariant, keeping the advantage/correction balance in pepg_noratio stable across CFG.
        return elbo / guidance_scale
    
    elif method == 't':
        forward_x0_prediction = xt - t_expanded * v_prediction
        return - ((forward_x0_prediction - x0) ** 2).mean(dim=tuple(range(1, x0.ndim)))

    elif method == 'uniform':
        elbo = - (((noise - x0 - v_prediction))**2).mean(dim=tuple(range(1, x0.ndim)))
        # Uniform ELBO is -||v - (ε - x_0)||², which scales as G² under CFG
        # (v - v_true ∝ G). Divide by G² to keep magnitude G-invariant, matching the 1/G correction applied to adaptive.
        return elbo / (guidance_scale * guidance_scale)
    
    elif method == 'exact':
        weights = (1.0 - t_expanded) / (t_expanded + 1e-3) 
        return - (weights * ((noise - x0 - v_prediction))**2).mean(dim=tuple(range(1, x0.ndim)))


# ─── Drifting reward target (docs/drifting_reward_impl_plan.md) ──────────────
# Network-free cross-sample drift target δ_i for the DMD-shaped wrapper
#   L_i = ½‖x̂0^θ − sg[x̂0^θ + w·δ_i]‖²   (w = native adaptive-ELBO normalizer).
# Modes A/B are t-independent (depend only on the clean endpoints x0 and the
# clipped advantage A), so they are PRECOMPUTED once per K-group at advantage
# time and cached. Mode S (self-sample) is t-dependent (uses x̂0^θ), so it is
# built inside the loss, not here. These helpers are pure / additive — no
# existing method calls them.
def median_dist(feat):
    """Median of the off-diagonal pairwise Euclidean distances of ``feat``
    ([K, D]); the median-heuristic bandwidth τ for the Gaussian kernel. Returns
    a 0-d tensor, floored at 1e-6 so τ=0 (identical samples) can't divide-by-0."""
    with torch.no_grad():
        d = torch.cdist(feat, feat)  # [K, K]
        K = d.shape[0]
        if K < 2:
            return torch.tensor(1.0, device=d.device, dtype=d.dtype)
        iu = torch.triu_indices(K, K, offset=1, device=d.device)
        vals = d[iu[0], iu[1]]
        m = torch.median(vals)
        return m.clamp(min=1e-6)


def subset_bary(ell, A, X, mask):
    """|A|-weighted, kernel-localized barycenter of a column subset (Loss B).

    softmax over the masked columns of ``ell[:, mask] + log|A[mask]|`` applied to
    ``X[mask]``. ``ell`` [K,K] kernel logits, ``A`` [K] advantage, ``X`` [K,D]
    endpoints, ``mask`` [K] bool. Returns [K, D]; all-zero if the subset is empty."""
    if mask.sum() == 0:
        return torch.zeros_like(X)
    cols = mask.nonzero(as_tuple=True)[0]
    logits = ell[:, cols] + torch.log(A[cols].abs().clamp(min=1e-8))[None, :]  # [K, |mask|]
    wts = torch.softmax(logits, dim=1)
    return wts @ X[cols]


def compute_drift_target(x0, adv, groups, tau, mode, loo=True, phi=None):
    """Per-sample drift target δ for modes A/B (docs/drifting_reward_impl_plan.md).

    Args:
        x0:     [N, *feat] detached clean endpoints (full latents).
        adv:    [N] per-sample advantage, ALREADY clipped to ±adv_clip_max.
        groups: list of index tensors/arrays, one per distinct prompt (the
                K-repeat group); buckets of `prompt_group_ids`.
        tau:    fixed bandwidth, or None → per-group median heuristic.
        mode:   "A" (additive tilt: softmax(ℓ+A) − softmax(ℓ)) or
                "B" (winner−loser split; degenerate group → fall back to A, #7).
        loo:    leave-one-out — exclude self from the kernel (#6).
        phi:    optional feature map [N, *]; None → identity (raw latent).

    Returns δ with the same shape as x0; gradient never flows through it (the
    caller detaches / it is built from detached endpoints).
    """
    delta = torch.zeros_like(x0)
    flat = x0.flatten(1).float()  # [N, D]
    phi_flat = phi.flatten(1).float() if phi is not None else None
    for g in groups:
        if len(g) < 2:
            continue  # singleton group → no cross-sample signal, leave δ=0
        X = flat[g]                       # [K, D]
        A = adv[g].float()                # [K]
        feat = X if phi_flat is None else phi_flat[g]
        tau_g = tau if tau else median_dist(feat)
        ell = -torch.cdist(feat, feat) ** 2 / (2 * tau_g ** 2)  # canonical Gaussian
        if loo:
            ell.fill_diagonal_(float("-inf"))
        if mode == "A":
            dz = ((ell + A[None, :]).softmax(1) - ell.softmax(1)) @ X
        elif mode == "B":
            pos, neg = A > 0, A < 0
            if pos.sum() == 0 or neg.sum() == 0:
                # Degenerate (all-winner or all-loser) group: fall back to A.
                dz = ((ell + A[None, :]).softmax(1) - ell.softmax(1)) @ X
            else:
                dz = subset_bary(ell, A, X, pos) - subset_bary(ell, A, X, neg)
        else:
            raise ValueError(f"Unknown drift mode {mode!r} (expected 'A' or 'B')")
        delta[g] = dz.view(-1, *x0.shape[1:]).to(delta.dtype)
    return delta


# Advantage computation per prompt
class PerPromptStatTracker:
    def __init__(self, global_std=False):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

    # exp reward is for rwr
    def update(self, prompts, rewards, config, exp=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        adv_method = config.adv
        advantages = np.empty_like(rewards) * 0.0
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(hash(prompt))  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[prompts == prompt]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)
            
            if adv_method == "standard":
                if self.global_std:
                    std = np.std(rewards, axis=0, keepdims=True) + 1e-4  # Use global std of all rewards
                else:
                    std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
                advantages[prompts == prompt] = config.scale * (prompt_rewards - mean) / std

            elif adv_method == "exact":
                beta, alpha = config.beta, config.alpha
                if beta != 0:
                    advantages[prompts == prompt] = beta * (prompt_rewards - mean) / alpha
                else:
                    advantages[prompts == prompt] = config.scale * (prompt_rewards - mean)

            elif adv_method == "diffusionnft":
                # DiffusionNFT (arXiv:2509.16117) treats reward NOT as an advantage
                # but as an optimality probability r ∈ [0,1] that convexly mixes the
                # positive / negative regression branches (r and 1-r sum to 1).
                # Faithful to the reference impl (NVlabs/DiffusionNFT
                # scripts/train_nft_sd3.py, default adv_mode="all"): take the plain
                # group-relative advantage A = (R-μ)/σ_g, clip to ±M, then linearly
                # map to [0,1]:  r = clip( clip(A,-M,M)/M / 2 + 0.5, 0, 1 ),
                # with M = config.adv_clip_max. (A=0 → r=0.5; A=+M → r=1; A=-M → r=0.)
                if self.global_std:
                    std = np.std(rewards, axis=0, keepdims=True) + 1e-4
                else:
                    std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4
                A = (prompt_rewards - mean) / std
                M = float(getattr(config, "adv_clip_max", 5.0))
                A = np.clip(A, -M, M)
                advantages[prompts == prompt] = np.clip(A / M / 2.0 + 0.5, 0.0, 1.0)

            else:
                raise NotImplementedError(f"Advantage method {adv_method} not implemented.")

        return advantages

    def get_stats(self):
        avg_group_size = sum(len(v) for v in self.stats.values()) / len(self.stats) if self.stats else 0
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        if not per_prompt_top_means:
            return 0.0

        return np.mean(per_prompt_top_means)

def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    torch.cuda.set_device(lock_rank)
    timeout_seconds = os.getenv("NCCL_TIMEOUT_SECONDS")
    kwargs = {"rank": rank, "world_size": world_size}
    if timeout_seconds:
        kwargs["timeout"] = timedelta(seconds=int(timeout_seconds))
    dist.init_process_group("nccl", **kwargs)
    

def cleanup_distributed():
    dist.destroy_process_group()

def is_main_process(rank):
    return rank == 0

def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def get_reward_fn(device, reward_config, videoalign_dimension_weights=None, video_fps=None):
    return getattr(src.reward, "multi_score")(
        device, reward_config,
        videoalign_dimension_weights=videoalign_dimension_weights,
        video_fps=video_fps,
    )


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    elif decay_type == 3:
        flat = 10
        uprate = 0.01
        uphold = 0.5
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def save_ckpt(
    logger, wandb, save_dir, transformer_ddp, global_step, rank, ema, transformer_trainable_parameters, config, optimizer, scaler, epoch=None, stat_tracker=None, ckpt_subdir="checkpoints"
):
    # ckpt_subdir selects where, under save_dir, the checkpoint is written.
    # Default "checkpoints" is the rolling checkpoint (overwritten each epoch,
    # used for resume). Pass e.g. "permanent/epoch_<N>/checkpoints" to write a
    # permanent epoch snapshot that is never overwritten.
    if is_main_process(rank):
        save_root = os.path.join(save_dir, ckpt_subdir)
        os.makedirs(save_root, exist_ok=True)

        model_to_save = transformer_ddp.module

        # Save current policy (default adapter) - without EMA
        save_root_lora_current = os.path.join(save_root, "lora_current_policy")
        os.makedirs(save_root_lora_current, exist_ok=True)
        model_to_save.set_adapter("default")
        model_to_save.save_pretrained(save_root_lora_current)
        logger.info(f"Saved current policy adapter to {save_root_lora_current}")

        # Save old policy adapter
        save_root_lora_old = os.path.join(save_root, "lora_old_policy")
        os.makedirs(save_root_lora_old, exist_ok=True)
        model_to_save.set_adapter("old")
        model_to_save.save_pretrained(save_root_lora_old)
        logger.info(f"Saved old policy adapter to {save_root_lora_old}")

        # Save EMA policy (if enabled)
        if config.train.ema and ema is not None:
            save_root_lora_ema = os.path.join(save_root, "lora_ema_policy")
            os.makedirs(save_root_lora_ema, exist_ok=True)
            model_to_save.set_adapter("default")
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
            model_to_save.save_pretrained(save_root_lora_ema)
            ema.copy_temp_to(transformer_trainable_parameters)
            logger.info(f"Saved EMA policy adapter to {save_root_lora_ema}")

        # Restore to default adapter
        model_to_save.set_adapter("default")

        # Save optimizer and scaler
        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))
        
        # Save global_step, epoch, stat_tracker state, and wandb run id for proper resumption
        wandb_run_id = wandb.run.id if wandb.run else None
        training_state = {
            "global_step": global_step,
            "wandb_run_id": wandb_run_id
        }
        if epoch is not None:
            training_state["epoch"] = epoch
        if stat_tracker is not None:
            training_state["stat_tracker_history_prompts"] = stat_tracker.history_prompts
        torch.save(training_state, os.path.join(save_root, "training_state.pt"))

        # Save training config for evaluation
        import json
        config_dict = config.to_dict()
        with open(os.path.join(save_root, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        logger.info(f"Saved training config to {save_root}/config.json")

        logger.info(f"Saved all checkpoints to {save_root} at step {global_step}")


def _sample_cache_dir(save_dir, epoch):
    return os.path.join(save_dir, "checkpoints", "sample_cache", f"epoch_{epoch}")


def _move_samples_to_cpu(samples_dict):
    """Return a dict with all CUDA tensors moved to CPU (nested dicts handled)."""
    out = {}
    for k, v in samples_dict.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu() if v.is_cuda else v
        elif isinstance(v, dict):
            out[k] = _move_samples_to_cpu(v)
        else:
            out[k] = v
    return out


def save_sample_cache(save_dir, epoch, rank, world_size, samples_dict, logger):
    """Persist per-rank samples shard for fast resume. Atomic via temp+rename
    and a rank-0 _COMPLETE marker written only after every rank has saved."""
    cache_dir = _sample_cache_dir(save_dir, epoch)
    if is_main_process(rank):
        os.makedirs(cache_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    samples_cpu = _move_samples_to_cpu(samples_dict)
    payload = {"world_size": world_size, "rank": rank, "epoch": epoch, "samples": samples_cpu}
    tmp_path = os.path.join(cache_dir, f"samples_rank_{rank}.pt.tmp")
    final_path = os.path.join(cache_dir, f"samples_rank_{rank}.pt")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, final_path)

    if world_size > 1:
        dist.barrier()
    if is_main_process(rank):
        with open(os.path.join(cache_dir, "_COMPLETE"), "w") as f:
            f.write(str(epoch))
        logger.info(f"Saved sample cache for epoch {epoch} -> {cache_dir}")


def load_sample_cache(save_dir, epoch, rank, world_size, logger):
    """Return the samples dict for this rank, or None if the cache is missing
    / incomplete / built for a different world_size."""
    cache_dir = _sample_cache_dir(save_dir, epoch)
    marker = os.path.join(cache_dir, "_COMPLETE")
    shard = os.path.join(cache_dir, f"samples_rank_{rank}.pt")
    if not os.path.exists(marker) or not os.path.exists(shard):
        return None
    try:
        payload = torch.load(shard, map_location="cpu")
    except Exception as e:
        if is_main_process(rank):
            logger.warning(f"Failed to load sample cache {shard}: {e}. Re-sampling.")
        return None
    if payload.get("world_size") != world_size:
        if is_main_process(rank):
            logger.warning(
                f"Sample cache world_size={payload.get('world_size')} != current {world_size}. Re-sampling."
            )
        return None
    return payload["samples"]


def clear_sample_cache(save_dir, epoch, rank, world_size):
    """Remove the epoch_N cache directory after training has consumed it."""
    if world_size > 1:
        dist.barrier()
    if is_main_process(rank):
        shutil.rmtree(_sample_cache_dir(save_dir, epoch), ignore_errors=True)


def save_reward_breakdown(save_dir, epoch, breakdown):
    """Persist the per-epoch reward breakdown scalars alongside the sample cache.

    The sample cache stores only the *averaged* reward tensor (the per-component
    rewards dict is collapsed to ``rewards["avg"]`` before caching), so a
    cache-hit resume cannot reconstruct the per-component breakdown
    (``reward/videoalign_ta``, ``reward/mean_top_*``, ``reward/zero_std_ratio``,
    …). Write those already-computed scalars to a small JSON sidecar so the
    cache-hit path can re-emit the full breakdown to wandb instead of leaving a
    hole in every per-component reward curve. Rank-0 only; best-effort."""
    import json
    cache_dir = _sample_cache_dir(save_dir, epoch)
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = os.path.join(cache_dir, "reward_breakdown.json.tmp")
    final_path = os.path.join(cache_dir, "reward_breakdown.json")
    with open(tmp_path, "w") as f:
        json.dump({k: float(v) for k, v in breakdown.items()}, f)
    os.replace(tmp_path, final_path)


def load_reward_breakdown(save_dir, epoch):
    """Return the per-epoch reward-breakdown sidecar dict, or None if absent
    (older caches written before this sidecar existed)."""
    import json
    path = os.path.join(_sample_cache_dir(save_dir, epoch), "reward_breakdown.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def build_model_utils(base_model, *, config, logger, wandb_module, device, rank, world_size):
    """Instantiate the TrainingUtils class for ``config.base_model``.

    Imports are lazy so each model's heavy diffusers dependencies are not
    pulled in at training start.
    """
    if base_model == "sd3":
        from .sd3 import SD3TrainingUtils as cls
    elif base_model == "flux":
        from .flux import FluxTrainingUtils as cls
    else:
        raise ValueError(f"Unsupported config.base_model={base_model}.")
    return cls(
        config=config,
        logger=logger,
        wandb_module=wandb_module,
        device=device,
        rank=rank,
        world_size=world_size,
    )
