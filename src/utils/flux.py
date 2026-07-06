from collections import defaultdict
import os
import tempfile
import time

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast as torch_autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from diffusers import FluxPipeline
from peft import LoraConfig

import src.reward
from src.diffusers_patch.pipeline_with_logprob_flux import pipeline_with_logprob as patched_pipeline_with_logprob
from . import gather_tensor_to_all, is_main_process
from .base import BaseTrainingUtils


class FluxTrainingUtils(BaseTrainingUtils):
    """Training utilities for Flux model.

    Flux differs from SD3 in:
    - 2 text encoders (text_encoder, text_encoder_2) instead of 3
    - 2 tokenizers instead of 3
    - Different LoRA target modules
    - Returns (image, latents, latent_image_ids, text_ids, log_probs) from pipeline
    - Uses bf16 for training (fp16 inference cannot produce valid images)
    - Simpler sample/train split: no separate CFG guidance for training
    """

    def __init__(self, config, logger, wandb_module, device, rank, world_size):
        super().__init__(config, logger, wandb_module, device, rank, world_size)
        self._pipeline = None  # set in get_pretrained_model; used by v_pred_fn

    def get_pretrained_model(self, mixed_precision_dtype, enable_amp):
        """Load Flux pipeline with frozen VAE and text encoders."""
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        pipeline = FluxPipeline.from_pretrained(
            self.config.pretrained.model,
            token=hf_token,
            local_files_only=False,
        )
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
        pipeline.transformer.requires_grad_(not self.config.use_lora)

        text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
        tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

        pipeline.safety_checker = None
        pipeline.set_progress_bar_config(
            position=1,
            disable=not is_main_process(self.rank),
            leave=False,
            desc="Timestep",
            dynamic_ncols=True,
        )

        text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
        pipeline.vae.to(self.device, dtype=torch.float32)
        pipeline.text_encoder.to(self.device, dtype=text_encoder_dtype)
        pipeline.text_encoder_2.to(self.device, dtype=text_encoder_dtype)

        self._pipeline = pipeline

        return pipeline, text_encoders, tokenizers

    def get_lora_config(self):
        return LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=[
                "attn.to_k",
                "attn.to_q",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "ff.net.0.proj",
                "ff.net.2",
                "ff_context.net.0.proj",
                "ff_context.net.2",
            ],
        )

    def compute_text_embeddings(self, prompts, text_encoders, tokenizers, max_sequence_length=128):
        """Compute text embeddings using FluxPipeline.encode_prompt.

        Flux uses heterogeneous text encoders (CLIP + T5). CLIP uses 77
        positional tokens while T5 uses a larger sequence length, so we
        delegate to pipeline.encode_prompt rather than manual tokenization.
        """
        with torch.no_grad():
            if self._pipeline is None:
                raise RuntimeError("Flux pipeline is not initialized before compute_text_embeddings.")

            prompt_embeds, pooled_prompt_embeds, _text_ids = self._pipeline.encode_prompt(
                prompt=prompts,
                prompt_2=prompts,
                device=self.device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
            )

            prompt_embeds = prompt_embeds.to(self.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.device)
            return prompt_embeds, pooled_prompt_embeds

    def get_negative_prompt_embeddings(self, text_encoders, tokenizers):
        return self.compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128)

    def get_prompt_embeddings(self, prompts, text_encoders, tokenizers):
        return self.compute_text_embeddings(prompts, text_encoders, tokenizers, max_sequence_length=128)

    def build_sample_kwargs(
        self,
        prompt_embeds,
        pooled_prompt_embeds,
        sample_neg_prompt_embeds,
        sample_neg_pooled_prompt_embeds,
        batch_size,
    ):
        """Build kwargs for Flux pipeline sampling.

        Flux does not use classifier-free guidance via negative prompts;
        guidance is handled internally via guidance_embeds.
        """
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "num_inference_steps": self.config.sample.num_steps,
            "guidance_scale": self.config.sample.guidance_scale,
            "output_type": "pt",
            "height": self.config.resolution,
            "width": self.config.resolution,
            "noise_level": self.config.sample.noise_level,
            "deterministic": self.config.sample.deterministic,
            "solver": self.config.sample.solver,
        }

    def v_pred_fn(
        self,
        transformer_ddp,
        xt,
        samples,
        index,
        embeds,
        pooled_embeds,
        neg_embeds,
        neg_pooled_embeds,
        sample_type="train",
        cached_uncond=None,
    ):
        """Velocity prediction for Flux.

        Flux transformer requires img_ids and txt_ids (positional encodings)
        plus an optional guidance embedding for FLUX.1-dev.
        """
        batch_size, seq_len, _ = xt.shape

        h = w = int(seq_len ** 0.5)
        img_ids = torch.zeros(h, w, 3, device=xt.device, dtype=xt.dtype)
        img_ids[..., 1] = torch.arange(h, device=xt.device)[:, None]
        img_ids[..., 2] = torch.arange(w, device=xt.device)[None, :]
        img_ids = img_ids.reshape(seq_len, 3)

        text_seq_len = embeds.shape[1]
        txt_ids = torch.zeros(text_seq_len, 3, device=xt.device, dtype=xt.dtype)

        guidance = None
        if self._pipeline is not None and self._pipeline.transformer.config.guidance_embeds:
            guidance = torch.full(
                [batch_size],
                self.config.sample.guidance_scale,
                device=xt.device,
                dtype=torch.float32,
            )

        prediction = transformer_ddp(
            hidden_states=xt,
            timestep=samples["timesteps"][:, index] / 1000,
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            return_dict=False,
        )[0]

        return prediction

    def pipeline_with_logprob(self, pipeline, **kwargs):
        """Flux pipeline returns 5-tuple; normalise to 3-tuple."""
        images, all_latents, _latent_image_ids, _text_ids, all_log_probs = patched_pipeline_with_logprob(
            pipeline, **kwargs
        )
        return images, all_latents, all_log_probs

    def decode_latents(self, pipeline, latents):
        """Decode raw latents into images using Flux VAE.

        Flux uses packed latents that must be unpacked before VAE decode.
        """
        height = self.config.resolution
        width = self.config.resolution
        latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
        latents = latents.to(dtype=pipeline.vae.dtype)
        scaling_factor = getattr(pipeline.vae.config, "scaling_factor", None)
        shift_factor = getattr(pipeline.vae.config, "shift_factor", None)
        if scaling_factor is not None and shift_factor is not None:
            latents = (latents / scaling_factor) + shift_factor
        image = pipeline.vae.decode(latents, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(image, output_type="pt")
        return image

    def save_sample(self, image_tensor, output_path):
        """Save a single image sample to disk."""
        img_data = image_tensor.float().cpu().clamp(0, 1)
        img_np = (img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        pil.save(output_path)
        return output_path

    def log_train_samples(self, images, prompts, rewards_to_log, global_step):
        if not is_main_process(self.rank):
            return

        images_to_log = images.float().cpu()
        with tempfile.TemporaryDirectory() as tmpdir:
            num_to_log = min(15, len(images_to_log))
            for idx in range(num_to_log):
                img_data = images_to_log[idx].clamp(0, 1)
                img_np = (img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                pil = Image.fromarray(img_np)
                pil = pil.resize((self.config.resolution, self.config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            self.wandb.log(
                {
                    "images": [
                        self.wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompts[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                        )
                        for idx in range(num_to_log)
                    ],
                },
                step=global_step,
            )

    def _log_eval_outputs(self, final_rewards, images, prompts, global_step):
        if not is_main_process(self.rank):
            return

        def _safe_reward_mean(value):
            arr = np.asarray(value)
            valid = arr[arr != -10]
            if valid.size == 0:
                return -10.0
            return float(np.mean(valid))

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((self.config.resolution, self.config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_samples_to_log)]

            self.wandb.log(
                {
                    "eval_images": [
                        self.wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{key}": _safe_reward_mean(value) for key, value in final_rewards.items()},
                },
                step=global_step,
            )

    def eval_fn(
        self,
        pipeline,
        test_dataloader,
        text_encoders,
        tokenizers,
        global_step,
        reward_fn,
        executor,
        mixed_precision_dtype,
        ema,
        transformer_trainable_parameters,
        tqdm_fn,
    ):
        if self.config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        pipeline.transformer.eval()

        all_rewards = defaultdict(list)

        test_sampler = (
            DistributedSampler(test_dataloader.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            if self.world_size > 1
            else None
        )
        eval_loader = DataLoader(
            test_dataloader.dataset,
            batch_size=self.config.sample.test_batch_size,
            sampler=test_sampler,
            collate_fn=test_dataloader.collate_fn,
            num_workers=test_dataloader.num_workers,
        )

        for test_batch in tqdm_fn(
            eval_loader,
            desc="Eval: ",
            disable=not is_main_process(self.rank),
            position=0,
        ):
            prompts, prompt_metadata = test_batch
            prompt_embeds, pooled_prompt_embeds = self.compute_text_embeddings(
                prompts, text_encoders, tokenizers, max_sequence_length=128
            )

            with torch_autocast(
                enabled=(self.config.mixed_precision in ["fp16", "bf16"]),
                dtype=mixed_precision_dtype,
            ):
                with torch.no_grad():
                    images, _, _ = self.pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=self.config.sample.eval_num_steps,
                        guidance_scale=self.config.sample.guidance_scale,
                        output_type="pt",
                        height=self.config.resolution,
                        width=self.config.resolution,
                        noise_level=0,
                    )

            rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
            time.sleep(0)
            rewards, _ = rewards_future.result()

            for key, value in rewards.items():
                rewards_tensor = torch.as_tensor(value, device=self.device).float()
                gathered_value = gather_tensor_to_all(rewards_tensor, self.world_size)
                all_rewards[key].append(gathered_value.numpy())

        if is_main_process(self.rank):
            final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}
            self._log_eval_outputs(final_rewards, images, prompts, global_step)

        if self.config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)

        if self.world_size > 1:
            dist.barrier()
