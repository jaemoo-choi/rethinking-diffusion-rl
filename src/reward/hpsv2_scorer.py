import os
import torch
import torch.nn as nn
from torchvision.transforms import Normalize, Compose, Resize, CenterCrop, InterpolationMode
import numpy as np
from PIL import Image

from hpsv2.src.open_clip import create_model, get_tokenizer
from .reward_ckpt_path import CKPT_PATH

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def image_transform_tensor(
    image_size: int,
    mean: tuple = None,
    std: tuple = None,
):
    """Match DanceGRPO / OpenCLIP's default val preprocessing:
    Resize(shortest_side) -> CenterCrop -> Normalize.
    """
    mean = mean or OPENAI_DATASET_MEAN
    std = std or OPENAI_DATASET_STD

    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    transforms = [
        Resize(image_size, interpolation=InterpolationMode.BICUBIC, antialias=True),
        CenterCrop(image_size),
        Normalize(mean=mean, std=std),
    ]
    return Compose(transforms)


class HPSv2Scorer(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        self.dtype = dtype
        self.device = device
        model = create_model(
            "ViT-H-14",
            os.path.join(CKPT_PATH, "open_clip_pytorch_model.bin"),
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            output_dict=True,
        )

        image_mean = getattr(model.visual, "image_mean", None)
        image_std = getattr(model.visual, "image_std", None)
        image_size = model.visual.image_size
        if isinstance(image_size, tuple):
            image_size = image_size[0]
        preprocess_val = image_transform_tensor(
            image_size,
            mean=image_mean,
            std=image_std,
        )

        self.model = model.to(device)
        self.preprocess_val = preprocess_val
        checkpoint = torch.load(os.path.join(CKPT_PATH, "HPS_v2.1_compressed.pt"), map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.processor = get_tokenizer("ViT-H-14")
        self.eval()

    @torch.no_grad()
    def __call__(self, images, prompts):
        """Score a batch of images.

        Args:
            images: tensor of shape [B, C, H, W] float in [0, 1].
            prompts: list of strings, length B.
        """
        image = self.preprocess_val(images.to(self.dtype).to(device=self.device, non_blocking=True))
        # Process the prompt
        text = self.processor(prompts).to(device=self.device, non_blocking=True)
        with torch.amp.autocast("cuda"):
            outputs = self.model(image, text)
            image_features, text_features = outputs["image_features"], outputs["text_features"]
            logits_per_image = image_features @ text_features.T
            hps_score = torch.diagonal(logits_per_image, 0)
        return hps_score.contiguous()

    @torch.no_grad()
    def score_video(self, video_frames, prompts, num_sample_frames=4):
        """Score a batch of videos by averaging HPSv2 over sampled frames.

        Following BranchGRPO's per-frame scoring pattern, this uniformly samples
        frames from each video and averages the per-frame HPSv2 scores.

        Args:
            video_frames: tensor of shape [B, T, C, H, W] float in [0, 1].
            prompts: list of strings, length B.
            num_sample_frames: number of frames to uniformly sample per video.
        """
        B, T, C, H, W = video_frames.shape
        n_frames = min(num_sample_frames, T)
        indices = torch.linspace(0, T - 1, n_frames).round().long()

        all_scores = torch.zeros(B, device=self.device)
        for fi in indices:
            frame_batch = video_frames[:, fi]  # [B, C, H, W]
            scores = self(frame_batch, prompts)
            all_scores += scores
        all_scores /= n_frames
        return all_scores


def main():
    scorer = HPSv2Scorer(dtype=torch.float32, device="cuda")

    images = [
        "test_cases/nasa.jpg",
        "test_cases/hello world.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts = [
        'An astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
        'New York Skyline with "Hello World" written with fireworks on the sky',
    ]
    images = [np.array(img) for img in pil_images]
    images = np.array(images)
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    images = torch.tensor(images, dtype=torch.uint8) / 255.0
    print(scorer(images, prompts))


if __name__ == "__main__":
    main()
