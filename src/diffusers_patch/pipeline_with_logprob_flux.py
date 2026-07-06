"""Flux pipeline with log-probability tracking.

Patched version of FluxPipeline.__call__ that returns all intermediate
latents and per-step log-probs for GRPO training.
Flux uses guidance embeddings (not CFG) and returns additional latent_image_ids
and text_ids needed for the Flux transformer architecture.
Supports all solvers (flow, dance, ddim, dpm1, dpm2).
"""

from typing import List, Optional, Union

import numpy as np
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from .solver import run_sampling


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    guidance_scale: float = 3.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    noise_level: float = 0.7,
    deterministic: bool = False,
    solver: str = "flow",
    skip_decode: bool = False,
):
    """Run Flux denoising with log-probability tracking.

    Args:
        self: FluxPipeline instance.
        noise_level: eta for stochastic solvers (ignored when deterministic=True).
        deterministic: If True, use deterministic sampling (eta=0).
        solver: One of "flow", "dance", "ddim", "dpm1", "dpm2".
        skip_decode: If True, skip VAE decode and return None for images.

    Returns:
        (images, all_latents, latent_image_ids, text_ids, all_log_probs):
        Decoded images (or None if skip_decode), list of intermediate latents,
        spatial position IDs, text position IDs, and list of per-step
        log-probabilities.
    """
    height = height or self.default_sample_size * self.vae_scale_factor
    width = width or self.default_sample_size * self.vae_scale_factor

    # 1. Check inputs
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = None
    self._current_timestep = None
    self._interrupt = False

    # 2. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    joint_attn_kwargs = getattr(self, "joint_attention_kwargs", None)
    lora_scale = (
        joint_attn_kwargs.get("scale", None)
        if joint_attn_kwargs is not None
        else None
    )

    # 3. Encode prompts
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )

    # 4. Prepare latent variables (Flux uses packed latents with image_ids)
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, latent_image_ids = self.prepare_latents(
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )

    # 5. Prepare timesteps (Flux uses shifted sigma schedule)
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if (
        hasattr(self.scheduler.config, "use_flow_sigmas")
        and self.scheduler.config.use_flow_sigmas
    ):
        sigmas = None
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    self._num_timesteps = len(timesteps)
    sigmas = self.scheduler.sigmas.float()

    # 6. Define velocity prediction with guidance embedding
    def v_pred_fn(z, sigma):
        latent_model_input = z
        # Flux uses guidance embeddings, not CFG
        if self.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=device, dtype=torch.float32
            )
            guidance = guidance.expand(latent_model_input.shape[0])
        else:
            guidance = None
        # Flux expects float timestep (sigma directly), not sigma * 1000
        timesteps = torch.full(
            [latent_model_input.shape[0]],
            sigma,
            device=z.device,
            dtype=latent_model_input.dtype,
        )
        joint_attn_kwargs = getattr(self, "joint_attention_kwargs", None)
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timesteps,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attn_kwargs,
            return_dict=False,
        )[0]
        return noise_pred

    # 7. Denoising loop
    latents, all_latents, all_log_probs = run_sampling(
        v_pred_fn, latents, sigmas, solver, deterministic, noise_level
    )

    # 8. Unpack and VAE decode (skipped when skip_decode=True to allow transformer offload)
    if not skip_decode:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = latents.to(dtype=self.vae.dtype)
        scaling_factor = getattr(self.vae.config, "scaling_factor", None)
        shift_factor = getattr(self.vae.config, "shift_factor", None)
        if scaling_factor is not None and shift_factor is not None:
            latents = (latents / scaling_factor) + shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)
    else:
        image = None

    self.maybe_free_model_hooks()

    return image, all_latents, latent_image_ids, text_ids, all_log_probs
