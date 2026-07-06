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
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig

import src.reward
from src.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from src.diffusers_patch.pipeline_with_logprob_sd3 import pipeline_with_logprob as patched_pipeline_with_logprob
from . import gather_tensor_to_all, is_main_process
from .base import BaseTrainingUtils


class SD3TrainingUtils(BaseTrainingUtils):
    def get_pretrained_model(self, mixed_precision_dtype, enable_amp):
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        pipeline = StableDiffusion3Pipeline.from_pretrained(
            self.config.pretrained.model,
            token=hf_token,
            local_files_only=False,
        )
        # Honor config.sample.shift (flow-matching trajectory warp) if set.
        # SD3 uses FlowMatchEulerDiscreteScheduler (key "shift"); no-op when
        # config.sample.shift is unset (default ODE/SDE configs).
        pipeline.scheduler = self._maybe_override_flow_shift(pipeline.scheduler)
        pipeline.vae.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)
        pipeline.text_encoder_2.requires_grad_(False)
        pipeline.text_encoder_3.requires_grad_(False)
        pipeline.transformer.requires_grad_(not self.config.use_lora)

        text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
        tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

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
        pipeline.text_encoder_3.to(self.device, dtype=text_encoder_dtype)

        return pipeline, text_encoders, tokenizers

    def get_lora_config(self):
        return LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=[
                "attn.add_k_proj",
                "attn.add_q_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "attn.to_k",
                "attn.to_out.0",
                "attn.to_q",
                "attn.to_v",
            ],
        )

    def compute_text_embeddings(self, prompt, text_encoders, tokenizers, max_sequence_length=128):
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders,
                tokenizers,
                prompt,
                max_sequence_length,
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
        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_prompt_embeds": sample_neg_prompt_embeds[:batch_size],
            "negative_pooled_prompt_embeds": sample_neg_pooled_prompt_embeds[:batch_size],
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
        if sample_type == "train":
            guidance_scale = self.config.train.guidance_scale
        elif sample_type == "sample":
            guidance_scale = self.config.sample.guidance_scale
        elif sample_type == "ref":
            guidance_scale = self.config.train.ref_guidance_scale
        else:
            raise ValueError(f"Unknown sample_type: {sample_type}")

        if guidance_scale > 1.0:
            embeds = torch.cat([neg_embeds, embeds])
            pooled_embeds = torch.cat([neg_pooled_embeds, pooled_embeds])

        prediction = transformer_ddp(
            hidden_states=torch.cat([xt] * 2) if guidance_scale > 1.0 else xt,
            timestep=torch.cat([samples["timesteps"][:, index]] * 2)
            if guidance_scale > 1.0
            else samples["timesteps"][:, index],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]

        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = prediction.chunk(2)
            prediction = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        return prediction

    def pipeline_with_logprob(self, pipeline, **kwargs):
        return patched_pipeline_with_logprob(pipeline, **kwargs)

    def decode_latents(self, pipeline, latents):
        """Decode raw latents into images using SD3 VAE."""
        latents = latents.to(dtype=pipeline.vae.dtype)
        scaling_factor = getattr(pipeline.vae.config, "scaling_factor", None)
        shift_factor = getattr(pipeline.vae.config, "shift_factor", None)
        if scaling_factor is not None and shift_factor is not None:
            latents = (latents / scaling_factor) + shift_factor
        image = pipeline.vae.decode(latents, return_dict=False)[0]
        image = pipeline.image_processor.postprocess(image, output_type="pt")
        return image

    def save_sample(self, image_tensor, output_path):
        """Save a single image sample to disk.

        Args:
            image_tensor: Tensor of shape [C, H, W] with values in [0, 1]
            output_path: Path to save the image (should end with .jpg)
        """
        img_np = (image_tensor.detach().to(dtype=torch.float32).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        pil.save(output_path)
        return output_path

    def log_train_samples(self, images, prompts, rewards_to_log, global_step):
        if not is_main_process(self.rank):
            return

        images_to_log = images.cpu()
        with tempfile.TemporaryDirectory() as tmpdir:
            num_to_log = min(15, len(images_to_log))
            for idx in range(num_to_log):
                img_data = images_to_log[idx]
                pil = Image.fromarray((img_data.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
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

        neg_prompt_embed, neg_pooled_prompt_embed = self.compute_text_embeddings(
            [""], text_encoders, tokenizers, max_sequence_length=128
        )

        sample_neg_prompt_embeds = neg_prompt_embed.repeat(self.config.sample.test_batch_size, 1, 1)
        sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(self.config.sample.test_batch_size, 1)

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
            current_batch_size = len(prompt_embeds)
            if current_batch_size < len(sample_neg_prompt_embeds):
                current_sample_neg_prompt_embeds = sample_neg_prompt_embeds[:current_batch_size]
                current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:current_batch_size]
            else:
                current_sample_neg_prompt_embeds = sample_neg_prompt_embeds
                current_sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds

            with torch_autocast(
                enabled=(self.config.mixed_precision in ["fp16", "bf16"]),
                dtype=mixed_precision_dtype,
            ):
                with torch.no_grad():
                    images, _, _ = self.pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=current_sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=current_sample_neg_pooled_prompt_embeds,
                        num_inference_steps=self.config.sample.eval_num_steps,
                        guidance_scale=self.config.sample.guidance_scale,
                        output_type="pt",
                        height=self.config.resolution,
                        width=self.config.resolution,
                        noise_level=self.config.sample.noise_level,
                        deterministic=True,
                        solver="flow",
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
