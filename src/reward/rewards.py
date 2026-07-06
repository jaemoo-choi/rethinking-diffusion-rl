from PIL import Image
import io
import numpy as np
import torch
import gc
import logging
import os
import sys
import json
import socket
import subprocess
import tempfile
import time
import atexit
from collections import defaultdict


logger = logging.getLogger(__name__)


# Default location of the pip overlay that shadows the training env's
# transformers 4.46 with 4.53.2 + qwen_vl_utils for the Qwen2.5-VL judges.
# The sbatch scripts set VLM_OVERLAY_PYTHONPATH; this is the fallback.
_DEFAULT_VLM_OVERLAY = os.path.join(
    os.environ.get("SCRATCH", os.path.expanduser("~/scratch")), "vs2_overlay"
)


def _is_cuda_oom(error):
    return isinstance(error, torch.OutOfMemoryError) or "out of memory" in str(error).lower()


def _build_score_function(score_name, factory, device):
    if "device" not in factory.__code__.co_varnames:
        return factory()

    target_device = torch.device(device) if not isinstance(device, torch.device) else device

    try:
        return factory(target_device)
    except Exception as error:
        if target_device.type != "cuda" or not _is_cuda_oom(error):
            raise

        logger.warning(
            "CUDA OOM while initializing reward scorer '%s' on %s; retrying on CPU.",
            score_name,
            target_device,
        )
        gc.collect()
        torch.cuda.empty_cache()
        return factory(torch.device("cpu"))


def _to_bchw_float01(images):
    """Convert image/video batch to B,C,H,W float in [0, 1].

    For videos, the first frame is selected.
    Supports torch/numpy in channel-first or channel-last layouts.
    """
    if isinstance(images, torch.Tensor):
        x = images.detach().cpu()
        if x.ndim == 5:
            # B,T,C,H,W or B,T,H,W,C
            if x.shape[2] == 3:
                x = x[:, 0]
            elif x.shape[-1] == 3:
                x = x[:, 0].permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 5D tensor shape: {tuple(x.shape)}")
        elif x.ndim == 4:
            # B,C,H,W or B,H,W,C
            if x.shape[1] == 3:
                pass
            elif x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 4D tensor shape: {tuple(x.shape)}")
        else:
            raise ValueError(f"Unsupported tensor rank for images/videos: {x.ndim}")

        x = x.float()
        if x.numel() > 0 and x.max().item() > 1.0:
            x = x / 255.0
        return x.clamp(0, 1)

    if isinstance(images, np.ndarray):
        x = images
        if x.ndim == 5:
            # B,T,C,H,W or B,T,H,W,C
            if x.shape[2] == 3:
                x = x[:, 0]
            elif x.shape[-1] == 3:
                x = x[:, 0].transpose(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 5D ndarray shape: {x.shape}")
        elif x.ndim == 4:
            # B,C,H,W or B,H,W,C
            if x.shape[1] == 3:
                pass
            elif x.shape[-1] == 3:
                x = x.transpose(0, 3, 1, 2)
            else:
                raise ValueError(f"Unsupported 4D ndarray shape: {x.shape}")
        else:
            raise ValueError(f"Unsupported ndarray rank for images/videos: {x.ndim}")

        x = torch.from_numpy(x).float()
        if x.numel() > 0 and x.max().item() > 1.0:
            x = x / 255.0
        return x.clamp(0, 1)

    raise ValueError(f"Unsupported input type for image conversion: {type(images)}")


def _to_bhwc_uint8(images):
    bchw = _to_bchw_float01(images)
    arr = (bchw * 255).round().to(torch.uint8).numpy()
    return arr.transpose(0, 2, 3, 1)


def _to_bthwc_uint8(images):
    """Convert image/video batch to B,T,H,W,C uint8.

    For image batches, T=1.
    """
    if isinstance(images, torch.Tensor):
        x = images.detach().cpu()
        if x.ndim == 5:
            # B,T,C,H,W or B,T,H,W,C
            if x.shape[2] == 3:
                pass
            elif x.shape[-1] == 3:
                x = x.permute(0, 1, 4, 2, 3)
            else:
                raise ValueError(f"Unsupported 5D tensor shape for videos: {tuple(x.shape)}")
        elif x.ndim == 4:
            # B,C,H,W or B,H,W,C
            if x.shape[1] == 3:
                x = x.unsqueeze(1)
            elif x.shape[-1] == 3:
                x = x.permute(0, 3, 1, 2).unsqueeze(1)
            else:
                raise ValueError(f"Unsupported 4D tensor shape for images/videos: {tuple(x.shape)}")
        else:
            raise ValueError(f"Unsupported tensor rank for videos: {x.ndim}")

        x = x.float()
        if x.numel() > 0 and x.max().item() <= 1.0:
            x = x * 255.0
        x = x.clamp(0, 255).round().to(torch.uint8)
        return x.permute(0, 1, 3, 4, 2).numpy()

    if isinstance(images, np.ndarray):
        x = images
        if x.ndim == 5:
            # B,T,C,H,W or B,T,H,W,C
            if x.shape[2] == 3:
                x = x.transpose(0, 1, 3, 4, 2)
            elif x.shape[-1] == 3:
                pass
            else:
                raise ValueError(f"Unsupported 5D ndarray shape for videos: {x.shape}")
        elif x.ndim == 4:
            # B,C,H,W or B,H,W,C
            if x.shape[1] == 3:
                x = x.transpose(0, 2, 3, 1)
            elif x.shape[-1] == 3:
                pass
            else:
                raise ValueError(f"Unsupported 4D ndarray shape for images/videos: {x.shape}")
            x = np.expand_dims(x, axis=1)
        else:
            raise ValueError(f"Unsupported ndarray rank for videos: {x.ndim}")

        if x.dtype != np.uint8:
            if np.issubdtype(x.dtype, np.floating) and x.size > 0 and np.max(x) <= 1.0:
                x = x * 255.0
            x = np.clip(x, 0, 255).astype(np.uint8)
        return x

    raise ValueError(f"Unsupported input type for video conversion: {type(images)}")


# ---------------------------------------------------------------------------
# Qwen2.5-VL judge rewards (VideoScore2, UnifiedReward-2.0) via overlay worker
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def jpeg_incompressibility():
    def _fn(images, prompts, metadata):
        images = _to_bhwc_uint8(images)
        images = [Image.fromarray(image) for image in images]
        buffers = [io.BytesIO() for _ in images]
        for image, buffer in zip(images, buffers):
            image.save(buffer, format="JPEG", quality=95)
        sizes = [buffer.tell() / 1000 for buffer in buffers]
        return np.array(sizes), {}

    return _fn


def jpeg_compressibility():
    jpeg_fn = jpeg_incompressibility()

    def _fn(images, prompts, metadata):
        rew, meta = jpeg_fn(images, prompts, metadata)
        return -rew / 500, meta

    return _fn


def aesthetic_score(device):
    from .aesthetic_scorer import AestheticScorer

    scorer = AestheticScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = (_to_bchw_float01(images) * 255).round().clamp(0, 255).to(torch.uint8)
        scores = scorer(images)
        return scores, {}

    return _fn


def clip_score(device):
    from .clip_scorer import ClipScorer

    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        images = _to_bchw_float01(images)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def hpsv2_score(device):
    from .hpsv2_scorer import HPSv2Scorer

    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = _to_bchw_float01(images)
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def pickscore_score(device):
    from .pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = _to_bhwc_uint8(images)
        images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def imagereward_score(device):
    from .imagereward_scorer import ImageRewardScorer

    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        images = _to_bhwc_uint8(images)
        images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def geneval_score(device):
    from .gen_eval import load_geneval

    batch_size = 64
    compute_geneval = load_geneval(device)

    def _fn(images, prompts, metadatas, only_strict):
        del prompts
        images = _to_bhwc_uint8(images)
        images_batched = np.array_split(images, np.ceil(len(images) / batch_size))
        metadatas_batched = np.array_split(metadatas, np.ceil(len(metadatas) / batch_size))
        all_scores = []
        all_rewards = []
        all_strict_rewards = []
        all_group_strict_rewards = []
        all_group_rewards = []
        for image_batch, metadata_batched in zip(images_batched, metadatas_batched):
            pil_images = [Image.fromarray(image) for image in image_batch]

            data = {
                "images": pil_images,
                "metadatas": list(metadata_batched),
                "only_strict": only_strict,
            }
            scores, rewards, strict_rewards, group_rewards, group_strict_rewards = compute_geneval(**data)

            all_scores += scores
            all_rewards += rewards
            all_strict_rewards += strict_rewards
            all_group_strict_rewards.append(group_strict_rewards)
            all_group_rewards.append(group_rewards)
        all_group_strict_rewards_dict = defaultdict(list)
        all_group_rewards_dict = defaultdict(list)
        for current_dict in all_group_strict_rewards:
            for key, value in current_dict.items():
                all_group_strict_rewards_dict[key].extend(value)
        all_group_strict_rewards_dict = dict(all_group_strict_rewards_dict)

        for current_dict in all_group_rewards:
            for key, value in current_dict.items():
                all_group_rewards_dict[key].extend(value)
        all_group_rewards_dict = dict(all_group_rewards_dict)

        return all_scores, all_rewards, all_strict_rewards, all_group_rewards_dict, all_group_strict_rewards_dict

    return _fn


def ocr_score(device):
    from .ocr import OcrScorer

    scorer = OcrScorer()

    def _fn(images, prompts, metadata):
        images = _to_bhwc_uint8(images)
        scores = scorer(images, prompts)
        # change tensor to list
        return scores, {}

    return _fn


def unifiedreward_score_sglang(device):
    import asyncio
    from openai import AsyncOpenAI
    import base64
    from io import BytesIO
    import re

    def pil_image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base64_qwen = f"data:image;base64,{encoded_image_text}"
        return base64_qwen

    def _extract_scores(text_outputs):
        scores = []
        pattern = r"Final Score:\s*([1-5](?:\.\d+)?)"
        for text in text_outputs:
            match = re.search(pattern, text)
            if match:
                try:
                    scores.append(float(match.group(1)))
                except ValueError:
                    scores.append(0.0)
            else:
                scores.append(0.0)
        return scores

    client = AsyncOpenAI(base_url="http://127.0.0.1:17140/v1", api_key="flowgrpo")

    async def evaluate_image(prompt, image):
        question = f"<image>\nYou are given a text caption and a generated image based on that caption. Your task is to evaluate this image based on two key criteria:\n1. Alignment with the Caption: Assess how well this image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of this image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nBased on the above criteria, assign a score from 1 to 5 after 'Final Score:'.\nYour task is provided as follows:\nText Caption: [{prompt}]"
        images_base64 = pil_image_to_base64(image)
        response = await client.chat.completions.create(
            model="UnifiedReward-7b-v1.5",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": images_base64},
                        },
                        {
                            "type": "text",
                            "text": question,
                        },
                    ],
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    async def evaluate_batch_image(images, prompts):
        tasks = [evaluate_image(prompt, img) for prompt, img in zip(prompts, images)]
        results = await asyncio.gather(*tasks)
        return results

    def _fn(images, prompts, metadata):
        # 处理Tensor类型转换
        images = _to_bhwc_uint8(images)

        # 转换为PIL Image并调整尺寸
        images = [Image.fromarray(image).resize((512, 512)) for image in images]

        # 执行异步批量评估
        text_outputs = asyncio.run(evaluate_batch_image(images, prompts))
        score = _extract_scores(text_outputs)
        score = [sc / 5.0 for sc in score]
        return score, {}

    return _fn


class MultiScorer:
    """Callable multi-reward scorer with GPU memory management.

    Supports offloading reward models to CPU during the training phase
    (when only the transformer is needed) and reloading to GPU for scoring.

    Usage:
        reward_fn = multi_score(device, score_dict)
        scores = reward_fn(images, prompts, metadata)  # normal scoring
        reward_fn.to_cpu()   # offload to CPU before training
        reward_fn.to_gpu()   # reload to GPU before next sampling
    """

    def __init__(self, device, score_dict, videoalign_dimension_weights=None, video_fps=None):
        score_functions = {
            "geneval": geneval_score,
            "ocr": ocr_score,
            "clipscore": clip_score,
            "aesthetic": aesthetic_score,
            "pickscore": pickscore_score,
            "imagereward": imagereward_score,
            "hpsv2": hpsv2_score,
            "unifiedreward": unifiedreward_score_sglang,
            "jpeg_compressibility": jpeg_compressibility,
        }
        self.device = device
        self.score_dict = score_dict
        self.score_fns = {}
        for score_name, weight in score_dict.items():
            self.score_fns[score_name] = _build_score_function(
                score_name, score_functions[score_name], device
            )

    def __call__(self, images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}

        for score_name, weight in self.score_dict.items():
            if score_name == "geneval":
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = self.score_fns[score_name](
                    images, prompts, metadata, only_strict
                )
                score_details["accuracy"] = rewards
                score_details["strict_accuracy"] = strict_rewards
                for key, value in group_strict_rewards.items():
                    score_details[f"{key}_strict_accuracy"] = value
                for key, value in group_rewards.items():
                    score_details[f"{key}_accuracy"] = value
            else:
                scores, rewards = self.score_fns[score_name](images, prompts, metadata)
                if isinstance(rewards, dict) and rewards:
                    for dim_key, dim_vals in rewards.items():
                        score_details[f"{score_name}_{dim_key}"] = dim_vals
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, {}

    def _get_scorer_modules(self):
        """Recursively collect every nn.Module reachable under each score_fn.

        score_fns come in two shapes (and a hybrid we have to handle):
        - Closures over an nn.Module (e.g. video_hpsv3_score, video_dino_ia_score):
          the module sits in the closure cells.
        - Class instances with __call__ that hold an nn.Module via attribute
          (e.g. videoalign_score). videoalign in particular is an nn.Module
          ITSELF that wraps a *non*-nn.Module helper (VideoVLMRewardInference)
          which in turn holds the actual ~14 GB Qwen2VLRewardModelBT as a plain
          attribute — `nn.Module.to()` will NOT recurse into it because the
          helper is not registered as a child module.

        So we cannot stop descending when we hit an nn.Module: there may be
        another nn.Module hidden inside its `__dict__` via a plain Python
        attribute (e.g. self.inferencer.model). Walk through everything,
        deduplicating with an id-set to avoid cycles. Skip Tensor leaves —
        they have no useful children and pull in tons of internal state.
        """
        modules = []
        seen = set()

        def _walk(obj):
            if obj is None:
                return
            oid = id(obj)
            if oid in seen:
                return
            seen.add(oid)
            # Tensor / Parameter leaves: no useful structure, skip.
            if isinstance(obj, torch.Tensor):
                return
            if isinstance(obj, torch.nn.Module):
                modules.append(obj)
                # IMPORTANT: keep descending. nn.Module.to() only walks
                # _modules / _parameters / _buffers; it misses non-Module
                # attributes that themselves contain nn.Modules.
            # Closure cells (for `def`-style score_fns).
            closure = getattr(obj, "__closure__", None)
            if closure:
                for cell in closure:
                    try:
                        _walk(cell.cell_contents)
                    except ValueError:
                        pass  # empty cell
            # Instance attributes (for class-style score_fns / wrappers).
            inst_dict = getattr(obj, "__dict__", None)
            if isinstance(inst_dict, dict):
                for v in inst_dict.values():
                    _walk(v)

        for score_fn in self.score_fns.values():
            _walk(score_fn)
        return modules

    def to_cpu(self):
        """Offload all reward model weights to CPU to free GPU memory."""
        for module in self._get_scorer_modules():
            module.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def to_gpu(self):
        """Reload all reward model weights to GPU for scoring."""
        for module in self._get_scorer_modules():
            module.to(self.device)


def multi_score(device, score_dict, videoalign_dimension_weights=None, video_fps=None):
    return MultiScorer(
        device,
        score_dict,
        videoalign_dimension_weights=videoalign_dimension_weights,
        video_fps=video_fps,
    )


def main():
    import torchvision.transforms as transforms

    image_paths = [
        "test_cases/nasa.jpg",
    ]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    images = torch.stack([transform(Image.open(image_path).convert("RGB")) for image_path in image_paths])
    prompts = [
        'A astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
    ]
    metadata = {}  # Example metadata
    score_dict = {"unifiedreward": 1.0}
    # Initialize the multi_score function with a device and score_dict
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scoring_fn = multi_score(device, score_dict)
    # Get the scores
    scores, _ = scoring_fn(images, prompts, metadata)
    # Print the scores
    print("Scores:", scores)


if __name__ == "__main__":
    main()
