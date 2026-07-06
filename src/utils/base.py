"""Base class for all model training utilities.

Captures the common skeleton shared across SD3, Flux, WAN2.1, HunyuanVideo,
SkyReels-I2V, and WorldPlay.  Derived classes override model-specific methods
(get_pretrained_model, v_pred_fn, pipeline_with_logprob, text-embedding
helpers, logging, eval_fn) and provide their LoRA configuration via
get_lora_config().

Loss computation (policy loss + KL regularization) is implemented here so
that train.py can call model_utils.compute_loss() instead of inlining the
math.
"""

import math
from abc import ABC, abstractmethod

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, PeftModel, get_peft_model

from . import compute_elbo, save_ckpt as base_save_ckpt


class BaseTrainingUtils(ABC):
    """Shared training utility base class for all diffusion models."""

    def __init__(self, config, logger, wandb_module, device, rank, world_size):
        self.config = config
        self.logger = logger
        self.wandb = wandb_module
        self.device = device
        self.rank = rank
        self.world_size = world_size

    # ------------------------------------------------------------------
    # Abstract methods – must be implemented by every derived class
    # ------------------------------------------------------------------

    @abstractmethod
    def get_pretrained_model(self, mixed_precision_dtype, enable_amp):
        """Load the pretrained pipeline.

        Returns:
            (pipeline, text_encoders, tokenizers)
        """
        ...

    @abstractmethod
    def get_lora_config(self):
        """Return a ``peft.LoraConfig`` appropriate for this model."""
        ...

    def _maybe_override_flow_shift(self, scheduler):
        """Re-instantiate ``scheduler`` with ``config.sample.shift`` if set.

        Flow-matching schedulers expose the trajectory-warping shift under
        either ``"shift"`` (FlowMatchEulerDiscreteScheduler) or ``"flow_shift"``
        (UniPCMultistepScheduler with ``use_flow_sigmas=True``). When the
        per-model config sets ``config.sample.shift``, override whichever key
        the loaded scheduler exposes; otherwise return the scheduler unchanged.
        """
        shift = getattr(self.config.sample, "shift", None)
        if shift is None:
            return scheduler
        cfg = dict(scheduler.config)
        overrides = {}
        if "shift" in cfg:
            overrides["shift"] = float(shift)
        if "flow_shift" in cfg:
            overrides["flow_shift"] = float(shift)
        if not overrides:
            return scheduler
        return type(scheduler).from_config(scheduler.config, **overrides)

    @abstractmethod
    def compute_text_embeddings(self, prompts, text_encoders, tokenizers, max_sequence_length=128):
        """Encode text prompts into embeddings."""
        ...

    @abstractmethod
    def get_negative_prompt_embeddings(self, text_encoders, tokenizers):
        """Return negative-prompt embeddings (neg_embeds, neg_pooled_embeds)."""
        ...

    @abstractmethod
    def get_prompt_embeddings(self, prompts, text_encoders, tokenizers):
        """Return prompt embeddings (embeds, pooled_embeds)."""
        ...

    @abstractmethod
    def build_sample_kwargs(self, prompt_embeds, pooled_prompt_embeds,
                            sample_neg_prompt_embeds, sample_neg_pooled_prompt_embeds,
                            batch_size):
        """Build kwargs dict for ``pipeline_with_logprob``."""
        ...

    @abstractmethod
    def v_pred_fn(self, transformer_ddp, xt, samples, index, embeds,
                  pooled_embeds, neg_embeds, neg_pooled_embeds,
                  sample_type="train", cached_uncond=None):
        """Velocity prediction for the model's transformer.

        ``cached_uncond`` optionally supplies a pre-computed CFG
        unconditional prediction so the same forward is not repeated across
        old / current / ref calls in compute_loss. Models that do not
        implement sharing can ignore this argument.
        """
        ...

    def _compute_shared_uncond(self, transformer_ddp, xt, samples, j_idx,
                               neg_embeds, neg_pooled_embeds):
        """Compute one CFG uncond pass to share across old/current/ref.

        Default: return ``None`` (no sharing; each v_pred_fn computes its
        own uncond). Override per-model when the transformer signature and
        CFG formulation allow a single shared forward.
        """
        return None

    @abstractmethod
    def pipeline_with_logprob(self, pipeline, **kwargs):
        """Run the patched pipeline and return (images, latents, log_probs)."""
        ...

    @abstractmethod
    def decode_latents(self, pipeline, latents):
        """Decode raw latents into images/videos using the pipeline's VAE.

        This mirrors the VAE decode logic from the patched pipeline, allowing
        decode to happen after the transformer has been offloaded from GPU.

        Args:
            pipeline: The diffusion pipeline (VAE must be on GPU).
            latents: Raw latents from the last denoising step.

        Returns:
            Decoded images (list of PIL images) or video tensors.
        """
        ...

    @abstractmethod
    def save_sample(self, data, output_path):
        """Save a single image/video sample to disk."""
        ...

    @abstractmethod
    def log_train_samples(self, images, prompts, rewards_to_log, global_step):
        """Log training samples to wandb."""
        ...

    @abstractmethod
    def _log_eval_outputs(self, final_rewards, images, prompts, global_step):
        """Log evaluation outputs to wandb."""
        ...

    @abstractmethod
    def eval_fn(self, pipeline, test_dataloader, text_encoders, tokenizers,
                global_step, reward_fn, executor, mixed_precision_dtype,
                ema, transformer_trainable_parameters, tqdm_fn):
        """Run evaluation loop."""
        ...

    # ------------------------------------------------------------------
    # Concrete shared methods
    # ------------------------------------------------------------------

    def get_tokenizer_max_length(self):
        return 256

    # ------------------------------------------------------------------
    # Hooks for the training loop
    #
    # These hide model-specific dataloading and sampling-artifact handling
    # from train.py. The default implementations cover the simple
    # text-only path (SD3 / Flux / WAN2.1); models with extra
    # conditioning (precomputed embeddings, reference images, action ids,
    # I2V conditioning tensors, etc.) override one or both.
    # ------------------------------------------------------------------

    def extract_dataloader_metadata(self, prompts, prompt_metadata, text_encoders, tokenizers):
        """Extract per-batch model inputs from a dataloader yield.

        Encodes prompts (or unpacks pre-computed embeddings from
        ``prompt_metadata``) and stashes any model-specific conditioning
        on ``self`` so that the subsequent ``build_sample_kwargs`` /
        ``pipeline_with_logprob`` calls can pick it up.

        Returns:
            dict with ``prompt_embeds``, ``pooled_prompt_embeds``,
            ``prompt_ids`` (the latter is ``None`` when prompts are not
            tokenized — e.g. precomputed-embedding mode).
        """
        prompt_embeds, pooled_prompt_embeds = self.get_prompt_embeddings(
            prompts, text_encoders, tokenizers,
        )
        prompt_ids = tokenizers[0](
            prompts,
            padding="max_length",
            max_length=self.get_tokenizer_max_length(),
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_ids": prompt_ids,
        }

    def collect_sample_artifacts(self, sample_data):
        """Append model-specific fields to ``sample_data`` after sampling.

        Default: no-op. Override in models that need to forward
        per-sample conditioning produced inside ``pipeline_with_logprob``
        (first-frame latents, I2V condition tensors, attention masks,
        action ids, ...) into the per-sample dict consumed by training.
        """
        return

    def maybe_extract_boundary_anchor(self, latents_full):
        """Return ``(x_boundary, t_boundary)`` for re-anchoring the velocity
        target away from ``x_N`` (the fully-denoised final latent), or
        ``(None, None)`` if this model does not need it.

        Called from the training entrypoint with the full sampling
        trajectory ``latents_full`` of shape ``(B, num_steps+1, ...)``
        before the trajectory tensor is freed. Wan2.2-I2V-A14B overrides
        this to return the latent at the MoE handoff (high-noise expert's
        actual rollout endpoint) when ``train_expert == "high"``; all other
        models inherit this no-op.
        """
        return None, None

    def set_adapter(self, pipeline, local_rank):
        """Configure LoRA adapters (default + old) and wrap in DDP."""
        if not self.config.use_lora:
            raise NotImplementedError("Current training loop expects LoRA adapters.")

        transformer = pipeline.transformer.to(self.device)
        transformer_lora_config = self.get_lora_config()

        if self.config.train.lora_path:
            transformer = PeftModel.from_pretrained(
                transformer, self.config.train.lora_path,
            )
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)

        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")

        transformer_ddp = DDP(
            transformer,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        transformer_ddp.module.set_adapter("default")
        transformer_trainable_parameters = list(
            filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
        )
        transformer_ddp.module.set_adapter("old")
        old_transformer_trainable_parameters = list(
            filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
        )
        transformer_ddp.module.set_adapter("default")

        return (
            transformer_ddp,
            transformer_trainable_parameters,
            old_transformer_trainable_parameters,
            transformer_lora_config,
        )

    def save_ckpt(self, save_dir, transformer_ddp, global_step, ema,
                  transformer_trainable_parameters, optimizer, scaler,
                  epoch=None, stat_tracker=None, ckpt_subdir="checkpoints"):
        base_save_ckpt(
            self.logger, self.wandb, save_dir, transformer_ddp, global_step,
            self.rank, ema, transformer_trainable_parameters, self.config,
            optimizer, scaler, epoch, stat_tracker, ckpt_subdir=ckpt_subdir,
        )

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_policy_loss(self, forward_ELBO, old_ELBO, advantages):
        """Compute the policy gradient loss.

        Supports: epg, epg_noratio, pepg, pepg_noratio, par, par_noratio, grpo.
        """
        method = self.config.train.method
        detached_logratio = forward_ELBO.detach() - old_ELBO
        detached_ratio = detached_logratio.exp()

        if method == "epg":
            return -(detached_ratio * advantages * forward_ELBO).mean()
        elif method == "epg_noratio":
            return -(advantages * forward_ELBO).mean()
        elif method == "pepg":
            return -(detached_ratio * (advantages - detached_logratio) * forward_ELBO).mean()
        elif method == "pepg_noratio":
            return -((advantages - detached_logratio) * forward_ELBO).mean()
        elif method == "par":
            return 0.5 * (detached_ratio * ((advantages - (forward_ELBO - old_ELBO)) ** 2)).mean()
        elif method == "par_noratio":
            return 0.5 * (((advantages - (forward_ELBO - old_ELBO)) ** 2)).mean()
        elif method == "grpo":
            assert "standard" == self.config.train.adv
            ratio = torch.exp(forward_ELBO - old_ELBO)
            advantages = torch.clamp(advantages, -5.0, 5.0)
            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(ratio, 1.0 - 1e-3, 1.0 + 1e-3)
            return torch.mean(torch.maximum(unclipped_loss, clipped_loss))
        else:
            raise NotImplementedError(f"Unknown policy method: {method}")

    def compute_kl_loss(self, forward_prediction, ref_forward_prediction,
                        forward_ELBO, old_ELBO, t_expanded, xt, x0, noise):
        """Compute the KL-divergence regularization loss.

        Supports: girsanov, kl1, kl1_importance, kl2, kl2_importance, kl3,
        kl3_importance.
        """
        kl_method = self.config.train.kl_method
        elbo_method = self.config.train.elbo

        if kl_method == "girsanov":
            return ((forward_prediction - ref_forward_prediction) ** 2).mean(
                dim=tuple(range(1, x0.ndim))
            )

        # All other variants need ref_ELBO
        else:
            ref_ELBO = compute_elbo(
                t_expanded, ref_forward_prediction, xt, x0, noise, elbo_method,
                guidance_scale=self.config.train.ref_guidance_scale,
            )

            if kl_method == "kl1":
                return forward_ELBO - ref_ELBO
            elif kl_method == "kl1_importance":
                return (forward_ELBO - old_ELBO).exp() * (forward_ELBO - ref_ELBO)
            elif kl_method == "kl2":
                return 0.5 * (forward_ELBO - ref_ELBO) ** 2
            elif kl_method == "kl2_importance":
                return 0.5 * (forward_ELBO - old_ELBO).exp() * (forward_ELBO - ref_ELBO) ** 2
            elif kl_method == "kl3":
                return (forward_ELBO - ref_ELBO).exp() - 1 - (forward_ELBO - ref_ELBO)
            elif kl_method == "kl3_importance":
                return (forward_ELBO - old_ELBO).exp() * (
                    (forward_ELBO - ref_ELBO).exp() - 1 - (forward_ELBO - ref_ELBO)
                )
            else:
                raise NotImplementedError(f"Unknown KL method: {kl_method}")

    def compute_pwvm_loss(self, forward_prediction, xt, x0, t_expanded,
                          advantages, prompt_group_ids):
        """Posterior-weighted cross-sample velocity matching loss.

        For each anchor i, softly match v_theta(x_t^i, t_i) against every clean
        endpoint x_0^j in the same prompt group, with pair weight proportional
        to the forward-kernel likelihood p(x_t^i | x_0^j) (interpreted as a
        non-parametric posterior over endpoints) and the candidate's advantage
        A^j. See docs/posterior_weighted_velocity_matching.md.

        Candidates are restricted to the current micro-batch. Self (i=j) is
        included, so in the degenerate case of no same-prompt siblings the
        loss collapses to the standard advantage-weighted per-trajectory flow
        matching objective.
        """
        if prompt_group_ids is None:
            raise ValueError(
                "pwvm method requires train_sample_batch['prompt_group_ids']; "
                "ensure per_prompt_stat_tracking is enabled."
            )

        B = forward_prediction.shape[0]
        feat_dims = tuple(range(1, x0.ndim))
        feat_numel = 1
        for d in x0.shape[1:]:
            feat_numel *= int(d)

        x0_f = x0.float()
        xt_f = xt.float()
        v_f = forward_prediction.float()
        t_col = t_expanded.float().view(B, *([1] * (x0.ndim - 1)))
        t_flat = t_col.view(B).clamp_min(1e-6)

        # --- Posterior weights w_ij (detached) ---
        with torch.no_grad():
            # ||xt^i - (1-t^i) x0^j||^2 via inner-product expansion — avoids
            # materialising a [B, B, *feat] tensor.
            xt_flat = xt_f.reshape(B, feat_numel)
            x0_flat = x0_f.reshape(B, feat_numel)
            one_minus_t_vec = (1.0 - t_flat).view(B, 1)

            xt_sq = xt_flat.pow(2).sum(dim=1)                 # [B], ||xt^i||^2
            x0_sq = x0_flat.pow(2).sum(dim=1)                 # [B], ||x0^j||^2
            cross = xt_flat @ x0_flat.t()                     # [B, B], <xt^i, x0^j>
            # ||xt^i - (1-t^i) x0^j||^2 = ||xt^i||^2
            #   - 2 (1-t^i) <xt^i, x0^j> + (1-t^i)^2 ||x0^j||^2
            sq_dist = (
                xt_sq.view(B, 1)
                - 2.0 * one_minus_t_vec * cross
                + one_minus_t_vec.pow(2) * x0_sq.view(1, B)
            ).clamp_min(0.0)

            same_prompt_mask = (
                prompt_group_ids.view(B, 1) == prompt_group_ids.view(1, B)
            )

            tau_mode = getattr(self.config.train, "pwvm_tau_mode", "median")
            if tau_mode == "median":
                # Per-row median over same-prompt candidates only. Push masked
                # entries to +inf (they end up at the tail after sort), then
                # gather the (lower) median by valid count.
                sentinel = torch.finfo(sq_dist.dtype).max
                masked_dist = sq_dist.masked_fill(~same_prompt_mask, sentinel)
                sorted_dist, _ = masked_dist.sort(dim=1)          # [B, B]
                valid_counts = same_prompt_mask.long().sum(dim=1)  # [B]
                median_idx = ((valid_counts - 1) // 2).clamp_min(0)
                median_dist = sorted_dist.gather(
                    1, median_idx.unsqueeze(1)
                ).squeeze(1)
                pwvm_scale_cfg = float(
                    getattr(self.config.train, "pwvm_scale", 1.0)
                )
                # 2 tau^2 = pwvm_scale * median / ln(2); the median entry sits
                # at logit = -ln(2) (weight 1/2 relative to the row minimum).
                denom = (
                    pwvm_scale_cfg * median_dist / math.log(2.0)
                ).clamp_min(1e-8)
                logits = -sq_dist / denom.view(B, 1)
            else:
                tau = float(getattr(self.config.train, "pwvm_tau", 1.0))
                logits = -sq_dist / (2.0 * tau * tau)

            logits = logits.masked_fill(~same_prompt_mask, float("-inf"))
            row_max = logits.amax(dim=1, keepdim=True)
            exp_logits = torch.where(
                same_prompt_mask,
                (logits - row_max).exp(),
                torch.zeros_like(logits),
            )
            row_sum = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-20)
            w_ij = exp_logits / row_sum  # [B, B]

            adv_clip = float(getattr(self.config.train, "adv_clip_max", 5.0))
            adv_row = advantages.detach().float().view(1, B).clamp(
                -adv_clip, adv_clip
            )
            pair_weights = w_ij * adv_row                     # [B, B]
            S_i = pair_weights.sum(dim=1)                     # [B]
            N_i = (pair_weights * x0_sq.view(1, B)).sum(dim=1)  # [B]
            # M_i[feat] = sum_j pair_weights[i,j] * x0[j,feat]
            M_flat = pair_weights @ x0_flat                   # [B, feat_numel]
            M_i = M_flat.view_as(x0_f)

        # Factored expansion of sum_j pair_weights[i,j] * ||v^i - (xt^i - x0^j)/t^i||^2
        # Let r_i = v^i - xt^i / t^i, then for each j:
        #   v^i - (xt^i - x0^j)/t^i = r_i + x0^j / t^i
        #   ||.||^2 = ||r_i||^2 + (2/t^i) <r_i, x0^j> + (1/t^i^2) ||x0^j||^2
        # Summing over j with pair_weights[i,j]:
        #   = S_i * ||r_i||^2 + (2/t^i) * <r_i, M_i> + (1/t^i^2) * N_i
        r_i = v_f - xt_f / t_col.clamp_min(1e-6)
        r_sq_sum = r_i.pow(2).sum(dim=feat_dims)              # [B]
        r_dot_M = (r_i * M_i).sum(dim=feat_dims)              # [B]

        inv_t = 1.0 / t_flat
        loss_per_anchor = (
            S_i * r_sq_sum
            + 2.0 * inv_t * r_dot_M
            + inv_t.pow(2) * N_i
        ) / float(feat_numel)                                 # matches per_pair .mean()
        loss = loss_per_anchor.mean()

        with torch.no_grad():
            diag_mask = torch.eye(B, dtype=w_ij.dtype, device=w_ij.device)
            self_weight_mean = (w_ij * diag_mask).sum(dim=1).mean()
            ess_per_sample = 1.0 / w_ij.pow(2).sum(dim=1).clamp_min(1e-12)
            pwvm_info = {
                "pwvm_self_weight": self_weight_mean,
                "pwvm_ess": ess_per_sample.mean(),
                "pwvm_loss": loss.detach(),
                # Per-sample ESS — train.py pops this and buckets by the
                # original sampler-timestep index to log ESS vs time.
                "pwvm_ess_per_sample": ess_per_sample,
            }
        return loss, pwvm_info

    def compute_loss(self, transformer_ddp, train_sample_batch, j_idx,
                     embeds, pooled_embeds, neg_embeds, neg_pooled_embeds):
        """Full loss computation for one timestep.

        Orchestrates old-policy, current-policy, and reference-model forward
        passes, then computes the policy loss and KL regularization.

        This method should be called inside a ``torch.cuda.amp.autocast``
        context by train.py.

        Returns:
            loss: Scalar training loss (policy + beta * KL).
            loss_terms: Dict of detached scalars for logging.
        """
        config = self.config

        # When the MoE high-noise expert is the trainable one (Wan2.2 A14B
        # I2V with train_expert="high"), anchor the velocity target to the
        # latent at the MoE handoff (x_boundary) instead of the fully-
        # denoised x_N. The high-noise expert's actual rollout endpoint is
        # x_boundary; the frozen low-noise expert then continues to x_N (used
        # for reward, untouched here). Rescaled flow:
        #   target_v = (x_t - x_boundary) / (t - t_boundary)
        #   predicted_x_boundary = x_t - (t - t_boundary) * v_pred
        # Re-uses the standard compute_elbo / compute_kl_loss / etc. by
        # feeding them (dt, x_boundary) in place of (t, x_N).
        use_boundary_anchor = (
            getattr(config, "train_expert", None) == "high"
            and "latents_boundary" in train_sample_batch
            and "t_boundary" in train_sample_batch
        )
        if use_boundary_anchor:
            x0 = train_sample_batch["latents_boundary"]
            t_boundary = train_sample_batch["t_boundary"].to(x0.device).float()
            t = (train_sample_batch["timesteps"][:, j_idx] / 1000.0 - t_boundary).clamp(min=1e-6)
        else:
            x0 = train_sample_batch["latents_clean"]
            t = train_sample_batch["timesteps"][:, j_idx] / 1000.0
        t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

        if getattr(config.train, "xt_from_latents", False) and "latents" in train_sample_batch:
            xt = train_sample_batch["latents"][:, j_idx].to(x0.dtype)
            # Effective noise consistent with xt = (1-t)*x0 + t*noise (in the
            # rescaled-time coordinate when use_boundary_anchor is on). Only
            # used by compute_elbo 'uniform' / 'exact'; 'adaptive' / 't'
            # ignore it.
            noise = ((xt.float() - (1 - t_expanded) * x0.float()) / t_expanded.clamp(min=1e-6)).to(x0.dtype)
        else:
            noise = torch.randn_like(x0.float())
            xt = (1 - t_expanded) * x0 + t_expanded * noise

        # --- Shared CFG uncond (computed once, reused across old/current/ref) ---
        # Default: None. Wan2.1 overrides _compute_shared_uncond to return a
        # single base-model uncond forward, saving 2 identical forwards per
        # timestep when CFG is active.
        cached_uncond = self._compute_shared_uncond(
            transformer_ddp, xt, train_sample_batch, j_idx,
            neg_embeds, neg_pooled_embeds,
        )

        # --- Old policy (detached) ---
        transformer_ddp.module.set_adapter("old")
        with torch.no_grad():
            old_prediction = self.v_pred_fn(
                transformer_ddp, xt, train_sample_batch, j_idx,
                embeds, pooled_embeds, neg_embeds, neg_pooled_embeds, "sample",
                cached_uncond=cached_uncond,
            )
            old_ELBO = compute_elbo(
                t_expanded, old_prediction, xt, x0, noise, config.train.elbo,
                guidance_scale=config.train.guidance_scale,
            )

        # --- Current policy (trainable) ---
        transformer_ddp.module.set_adapter("default")
        forward_prediction = self.v_pred_fn(
            transformer_ddp, xt, train_sample_batch, j_idx,
            embeds, pooled_embeds, neg_embeds, neg_pooled_embeds, "train",
            cached_uncond=cached_uncond,
        )
        forward_ELBO = compute_elbo(
            t_expanded, forward_prediction, xt, x0, noise, config.train.elbo,
            guidance_scale=config.train.guidance_scale,
        )

        # --- Reference model (frozen, no adapter) ---
        with torch.no_grad():
            if config.use_lora:
                with transformer_ddp.module.disable_adapter():
                    ref_forward_prediction = self.v_pred_fn(
                        transformer_ddp, xt, train_sample_batch, j_idx,
                        embeds, pooled_embeds, neg_embeds, neg_pooled_embeds, "ref",
                        cached_uncond=cached_uncond,
                    )
                transformer_ddp.module.set_adapter("default")
            else:
                assert False

        # Noise-region reward: when the run defines separate high/low-noise
        # reward weightings (config.reward_fn_{high,low}_noise), train.py stores
        # two per-sample advantage tensors. Select per step by the *actual*
        # sampler noise level t = timesteps/1000 — robust to the per-sample time
        # shuffle, since these advantages are constant along the timestep axis.
        # t >= high_noise_threshold is the high-noise region. Falls back to the
        # single advantage when the two tensors are absent (all existing runs).
        if ("advantages_high_noise" in train_sample_batch
                and "advantages_low_noise" in train_sample_batch):
            thr = float(getattr(config.train, "high_noise_threshold", 0.9))
            noise_level = train_sample_batch["timesteps"][:, j_idx] / 1000.0
            advantages = torch.where(
                noise_level >= thr,
                train_sample_batch["advantages_high_noise"][:, j_idx],
                train_sample_batch["advantages_low_noise"][:, j_idx],
            )
        else:
            advantages = train_sample_batch["advantages"][:, j_idx]

        # --- RAM / DiffusionNFT: velocity-space regression losses ---
        # These two methods do not use the ELBO log-prob / policy-gradient path.
        # They regress the trainable velocity v^θ (= forward_prediction) directly
        # against a reward-shaped target, with their own implicit regularization
        # (RAM anchors on v^ref inside the target; DiffusionNFT anchors on v^old),
        # so they bypass compute_policy_loss and the separate β·KL term.
        if config.train.method in ("ram", "diffusionnft"):
            feat_dims = tuple(range(1, x0.ndim))
            # Flow-matching target velocity v* = ε - x0 = noise - x0 (in the
            # rescaled-time coordinate when use_boundary_anchor is on).
            v_star = (noise - x0).to(forward_prediction.dtype)
            # Per-sample reward signal (shape [B]).
            r_sample = advantages.to(forward_prediction.dtype)

            if config.train.method == "ram":
                # RAM — Reinforce Adjoint Matching (arXiv:2605.10759), matching the
                # reference impl (AndreasBergmeister/ram scripts/training_sd3.py):
                #   reward_direction = ε - x0 = v*
                #   scaled_adv       = ram_reward_multiplier · advantage
                #   target = v^ref + scaled_adv · (v* - v^old)
                #   L      = || v^θ - sg[target] ||²
                # The residual uses the *old* (lagged) velocity, not v^θ, so the
                # whole target is detached and the gradient flows only through v^θ.
                mult = float(getattr(config.train, "ram_reward_multiplier", 1.0))
                r = (mult * r_sample).view(-1, *([1] * (len(x0.shape) - 1)))
                target = (
                    ref_forward_prediction + r * (v_star - old_prediction)
                ).detach()
                loss = ((forward_prediction - target) ** 2).mean(dim=feat_dims).mean()
            else:  # diffusionnft
                # DiffusionNFT — Theorem 3.2 / Eq. (5) (arXiv:2509.16117), a faithful
                # port of NVlabs/DiffusionNFT scripts/train_nft_sd3.py. The implicit
                # positive / negative policies blend the trainable v^θ with the frozen
                # sampling policy v^old (detached):
                #   v⁺ = γ·v^θ + (1-γ)·v^old,   v⁻ = (1+γ)·v^old - γ·v^θ
                # The branch losses are x0-space flow-matching MSEs with per-branch
                # ADAPTIVE weighting (the same form as compute_elbo "adaptive"), and
                # r ∈ [0,1] (from the 'diffusionnft' adv branch) convexly mixes them:
                #   L_policy = mean[ (r·L⁺ + (1-r)·L⁻) / γ ] · M
                # Plus a Girsanov KL anchor to the reference, weighted by β:
                #   L = L_policy + β·mean(||v^θ - v^ref||²)
                # γ = config.train.diffusionnft_beta (their config.beta); M =
                # config.train.adv_clip_max; β = config.train.beta.
                gamma = float(getattr(config.train, "diffusionnft_beta", 0.1))
                M = float(getattr(config.train, "adv_clip_max", 5.0))
                v_plus = gamma * forward_prediction + (1.0 - gamma) * old_prediction
                v_minus = (1.0 + gamma) * old_prediction - gamma * forward_prediction

                def _adaptive_x0_loss(v_pred):
                    x0_hat = xt - t_expanded * v_pred
                    with torch.no_grad():
                        wf = (
                            torch.abs(x0_hat.double() - x0.double())
                            .mean(dim=feat_dims, keepdim=True)
                            .clip(min=1e-5)
                        )
                    return ((x0_hat - x0) ** 2 / wf).mean(dim=feat_dims)

                positive_loss = _adaptive_x0_loss(v_plus)
                negative_loss = _adaptive_x0_loss(v_minus)
                ori_policy_loss = (
                    r_sample * positive_loss / gamma
                    + (1.0 - r_sample) * negative_loss / gamma
                )
                policy_loss = (ori_policy_loss * M).mean()
                kl_div = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                    dim=feat_dims
                )
                loss = policy_loss + config.train.beta * kl_div.mean()

            with torch.no_grad():
                loss_terms = {
                    "policy_loss": loss.detach(),
                    "total_loss": loss.detach(),
                    "x0_norm": torch.mean(x0 ** 2),
                    "forward_elbo_mean": forward_ELBO.mean(),
                    "old_elbo_mean": old_ELBO.mean(),
                    "kl_div": ((forward_prediction - ref_forward_prediction) ** 2)
                    .mean(dim=feat_dims).mean(),
                    "adv_mean": advantages.to(forward_ELBO.dtype).mean(),
                }
            return loss, loss_terms

        # --- DMD-shaped reward-fine-tuning losses (Stage 1; docs/dmd_loss.md §4) ---
        # Both variants write the update in DMD's exact gradient-injection wrapper
        # (§4.3): a plain ½ MSE whose target is sg[x̂0^θ + w·d], so the gradient IS
        #   ∇ = -(w·d)·∂x̂0^θ/∂θ   (x̂0^θ - target = -w·d in value; target detached).
        # No whole-loss 1/W scaling (c=1). w is the per-sample DMD/adaptive-ELBO
        # normalizer 1/(mean|x̂0^θ - x0| + 1e-5) — step size set by the signal, not
        # the t/sample-dependent residual scale. The β anchor pulls x̂0^θ toward the
        # frozen few-step init's x̂0 (= ref here; init≈base) and lives INSIDE d, so
        # this path bypasses compute_policy_loss + the separate β·KL term (exactly
        # like the ram / diffusionnft branches above). The two methods differ ONLY
        # in the reward coefficient on the data direction (x0 - x̂0^θ):
        #   dmd_epg: coef = A          (signed group-relative advantage; REINFORCE-like)
        #   dmd_nft: coef = 2σ(A) - 1  (bounded contrastive optimality; r<½ repels)
        if config.train.method in ("dmd_epg", "dmd_nft"):
            feat_dims = tuple(range(1, x0.ndim))
            # x̂0^θ (trainable) and x̂0^init (frozen few-step init ≈ ref, detached).
            x0_hat = xt - t_expanded * forward_prediction
            x0_hat_init = (xt - t_expanded * ref_forward_prediction).detach()
            # Reward coefficient on (x0 - x̂0^θ), shape [B,1,1,...] for broadcasting.
            A = advantages.to(x0_hat.dtype).view(-1, *([1] * (len(x0.shape) - 1)))
            if config.train.method == "dmd_epg":
                coef = A
            else:  # dmd_nft — r = σ(A) ∈ [0,1] -> contrastive sign 2r-1 ∈ [-1,1]
                coef = 2.0 * torch.sigmoid(A) - 1.0
            with torch.no_grad():
                # DMD "nice scaling": divide by per-sample residual-to-x0 (double
                # precision + 1e-5 clip, matching compute_elbo "adaptive").
                resid = (
                    torch.abs(x0_hat.detach().double() - x0.double())
                    .mean(dim=feat_dims, keepdim=True)
                    .clip(min=1e-5)
                )
                w = (1.0 / resid).to(x0_hat.dtype)
                d = coef * (x0 - x0_hat.detach()) + config.train.beta * (
                    x0_hat_init - x0_hat.detach()
                )
                target = x0_hat.detach() + w * d  # sg[x̂0^θ + w·d]
            loss = 0.5 * ((x0_hat - target) ** 2).mean(dim=feat_dims).mean()

            with torch.no_grad():
                loss_terms = {
                    "policy_loss": loss.detach(),
                    "total_loss": loss.detach(),
                    "x0_norm": torch.mean(x0 ** 2),
                    "forward_elbo_mean": forward_ELBO.mean(),
                    "old_elbo_mean": old_ELBO.mean(),
                    "kl_div": ((forward_prediction - ref_forward_prediction) ** 2)
                    .mean(dim=feat_dims).mean(),
                    "adv_mean": advantages.to(forward_ELBO.dtype).mean(),
                    "dmd_w_mean": w.mean().to(forward_ELBO.dtype),
                    "dmd_coef_mean": coef.mean().to(forward_ELBO.dtype),
                }
            return loss, loss_terms

        # --- PWVM: posterior-weighted cross-sample velocity matching ---
        # Like ram / diffusionnft, this is a velocity-space regression loss
        # that bypasses the ELBO policy-gradient + separate β·KL path. It
        # softly matches v^θ(x_t^i, t_i) against every clean endpoint x_0^j in
        # the same prompt group, weighted by the forward-kernel posterior and
        # the candidate advantage. See docs/posterior_weighted_velocity_matching.md.
        if config.train.method == "pwvm":
            loss, pwvm_info = self.compute_pwvm_loss(
                forward_prediction, xt, x0, t_expanded, advantages,
                train_sample_batch.get("prompt_group_ids"),
            )
            # Per-sample ESS is a [B] tensor; train.py's generic loss-term
            # accumulation only handles scalars, so drop it here.
            pwvm_info.pop("pwvm_ess_per_sample", None)
            with torch.no_grad():
                feat_dims = tuple(range(1, x0.ndim))
                pwvm_info.update({
                    "policy_loss": loss.detach(),
                    "total_loss": loss.detach(),
                    "x0_norm": torch.mean(x0 ** 2),
                    "forward_elbo_mean": forward_ELBO.mean(),
                    "old_elbo_mean": old_ELBO.mean(),
                    "kl_div": ((forward_prediction - ref_forward_prediction) ** 2)
                    .mean(dim=feat_dims).mean(),
                    "adv_mean": advantages.to(forward_ELBO.dtype).mean(),
                })
            return loss, pwvm_info

        # --- Policy loss ---
        if config.train.method == "sppo":
            policy_loss = (advantages * forward_ELBO + ((forward_prediction - old_prediction) / config.train.guidance_scale) ** 2  ).mean()
        else:
            policy_loss = self.compute_policy_loss(forward_ELBO, old_ELBO, advantages)

        # --- KL regularization ---
        kl_div_loss = self.compute_kl_loss(
            forward_prediction, ref_forward_prediction,
            forward_ELBO, old_ELBO, t_expanded, xt, x0, noise,
        )

        kl_div_loss_mean = torch.mean(kl_div_loss)
        loss = policy_loss + config.train.beta * kl_div_loss_mean

        # --- Auxiliary differentiable dynamic-degree loss (opt-in) ---
        # Off by default. When `config.train.dynamic_loss_lambda > 0`, decode
        # a flow-matching x0 estimate from the trainable forward velocity,
        # run RAFT optical flow, and add a motion-quality term. Gradient
        # flows: dynamic_loss → decoded_video → x0_hat → forward_prediction
        # → LoRA adapter. Cost: ~30 s VAE decode + ~3 s RAFT per microbatch
        # at 53×480×832 bsz=8. Requires self.pipeline.vae on GPU (lazily
        # restored if currently on CPU).
        dyn_loss_value = None
        dyn_lambda = float(getattr(config.train, "dynamic_loss_lambda", 0.0) or 0.0)
        if dyn_lambda > 0.0 and self.pipeline is not None:
            try:
                from src.reward.dynamic_degree_scorer import get_shared_dynamic_degree_scorer
                # Bring VAE to the same device as x0_hat. train.py:765 moves
                # vae to CPU before training; restore on first call here.
                _vae = getattr(self.pipeline, "vae", None)
                if _vae is not None and next(_vae.parameters()).device != x0.device:
                    _vae.to(x0.device)
                # Flow-matching x0 estimate from velocity prediction.
                # x_t = (1-t)*x0 + t*noise, v = dx/dt = noise - x0,
                # so x0_hat = x_t - t * v.
                x0_hat = xt - t_expanded * forward_prediction
                decoded = self.decode_latents(self.pipeline, x0_hat)
                # decoded shape varies per model; normalize to [B,T,C,H,W] in [0,1]
                if decoded.ndim == 5 and decoded.shape[1] == 3:
                    # WAN: [B, C, T, H, W] -> [B, T, C, H, W]
                    decoded = decoded.transpose(1, 2).contiguous()
                # Map to [0,1] (Wan VAE outputs roughly [-1, 1])
                decoded = (decoded.float().clamp(-1, 1) + 1.0) * 0.5

                scorer = get_shared_dynamic_degree_scorer(
                    dtype=torch.float32, device=x0.device,
                )
                num_pairs = int(getattr(config.train, "dynamic_loss_num_pairs", 8))
                flow_score = scorer.score_video_with_grad(decoded, num_pairs=num_pairs)

                kind = str(getattr(config.train, "dynamic_loss_kind", "neg"))
                if kind == "neg":
                    dyn_loss_value = -dyn_lambda * flow_score.mean()
                elif kind == "hinge":
                    from src.reward.dynamic_degree_scorer import DynamicDegreeScorer
                    thres = DynamicDegreeScorer.vbench_threshold(
                        int(config.height), int(config.width),
                    )
                    dyn_loss_value = dyn_lambda * torch.relu(thres - flow_score).mean()
                elif kind == "target":
                    tgt = float(getattr(config.train, "dynamic_loss_target", 0.0))
                    dyn_loss_value = dyn_lambda * torch.relu(tgt - flow_score).mean()
                else:
                    raise ValueError(
                        f"Unknown dynamic_loss_kind={kind!r}; "
                        "expected 'neg' | 'hinge' | 'target'."
                    )
                loss = loss + dyn_loss_value
            except Exception as _exc:
                # Don't crash training on a bad aux-loss config — log and skip.
                self.logger.warning(
                    f"dynamic_loss skipped due to error: {_exc!r}"
                )
                dyn_loss_value = None

        # --- Logging terms ---
        # Inside no_grad: .detach() is redundant (nothing has grad here).
        # Cache repeated squared-diff and mean tensors to avoid recomputation.
        with torch.no_grad():
            _elbo_diff = forward_ELBO - old_ELBO
            _adv = advantages.to(forward_ELBO.dtype)
            _adv_elbo = _adv * forward_ELBO
            _old_dev_sq = (forward_prediction - old_prediction) ** 2
            _kl_pair_sq = (forward_prediction - ref_forward_prediction) ** 2
            _old_kl_pair_sq = (old_prediction - ref_forward_prediction) ** 2
            _feat_dims = tuple(range(1, x0.ndim))
            # CFG-amplification diagnostic: the shared-uncond design causes
            # (v_current - v_ref) = G * (v_current_text - v_ref_text), so KL
            # scales as G^2. cfg_scale_sq_inv normalizes by 1/G^2 so values
            # stay comparable when sweeping G — a jump across CFG values is a
            # sign that adapter state (not CFG blend) is the source of drift.
            _g_train = float(config.train.guidance_scale)
            _cfg_scale_sq_inv = 1.0 / (_g_train * _g_train) if _g_train > 1.0 else 1.0
            # Loss-component diagnostic: report what the optimizer actually
            # sees. `correction_term` is what `pepg_noratio` subtracts from the
            # advantage (logratio * forward_ELBO); if its abs dominates
            # `adv_elbo_abs_mean`, the regularizer is steering the update.
            _kl_term_weighted = config.train.beta * kl_div_loss_mean
            _correction_term = (_elbo_diff * forward_ELBO).mean()
            # Direction diagnostic for pepg_noratio: the per-sample weight on
            # `forward_ELBO` is `(adv - logratio)`. Cosine with pure `adv`
            # says whether the update still points toward reward or has been
            # redirected by the CFG-amplified logratio term. ~+1 = aligned,
            # ~0 = orthogonal, ~-1 = anti-aligned (update steers away from
            # reward). Scalar per-batch; cheap and bypasses recomputing grads.
            _eff_weight = _adv - _elbo_diff
            _adv_norm = _adv.norm()
            _eff_norm = _eff_weight.norm()
            _cosine_eff_vs_adv = (_adv * _eff_weight).sum() / (
                _adv_norm * _eff_norm + 1e-8
            )
            loss_terms = {
                "x0_norm": torch.mean(x0 ** 2),
                "x0_norm_max": torch.max(x0 ** 2),
                "old_deviate": _old_dev_sq.mean(),
                "old_deviate_max": _old_dev_sq.max(),
                "policy_loss": policy_loss.detach(),
                "kl_div_loss": kl_div_loss_mean.detach(),
                "kl_div": _kl_pair_sq.mean(dim=_feat_dims).mean(),
                "old_kl_div": _old_kl_pair_sq.mean(dim=_feat_dims).mean(),
                "kl_div_cfg_normalized": _kl_pair_sq.mean(dim=_feat_dims).mean() * _cfg_scale_sq_inv,
                "kl_term_weighted_abs": _kl_term_weighted.abs(),
                "policy_loss_abs": policy_loss.detach().abs(),
                "correction_term_mean": _correction_term,
                "correction_term_abs_mean": _correction_term.abs(),
                "policy_dir_cosine_adv": _cosine_eff_vs_adv,
                "eff_weight_abs_mean": _eff_weight.abs().mean(),
                "forward_elbo_mean": forward_ELBO.mean(),
                "forward_elbo_std": forward_ELBO.std(unbiased=False),
                "old_elbo_mean": old_ELBO.mean(),
                "elbo_diff_std": _elbo_diff.std(unbiased=False),
                "adv_elbo_mean": _adv_elbo.mean(),
                "adv_elbo_abs_mean": _adv_elbo.abs().mean(),
                "total_loss": loss.detach(),
            }
            if dyn_loss_value is not None:
                loss_terms["dynamic_loss"] = dyn_loss_value.detach()

        return loss, loss_terms
