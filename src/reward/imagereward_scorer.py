import os
from PIL import Image
import torch
import ImageReward as RM
from config.paths import HF_HOME


class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = (
            RM.load(
                "ImageReward-v1.0",
                device=device,
                download_root=os.path.join(HF_HOME, "ImageReward"),
            )
            .eval()
            .to(dtype=dtype)
        )
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prompts, images):
        _, rewards = self.model.inference_rank(prompts, images)
        batch_size = len(prompts)
        rewards_tensor = torch.as_tensor(rewards, device=self.device, dtype=torch.float32).flatten()

        # ImageReward output can be:
        # - scalar (batch_size == 1)
        # - vector of length B
        # - flattened BxB matrix (pairwise format)
        if rewards_tensor.numel() == batch_size * batch_size:
            rewards_tensor = torch.diagonal(rewards_tensor.reshape(batch_size, batch_size), 0)
        elif rewards_tensor.numel() == batch_size:
            pass
        elif batch_size == 1 and rewards_tensor.numel() >= 1:
            rewards_tensor = rewards_tensor[:1]
        else:
            # Some ImageReward versions can return a scalar even for batched input.
            # Fall back to per-sample scoring for robust behavior across versions.
            fallback_scores = []
            for prompt, image in zip(prompts, images):
                score = self.model.score(prompt, image)
                fallback_scores.append(float(score))
            rewards_tensor = torch.as_tensor(fallback_scores, device=self.device, dtype=torch.float32)

        return rewards_tensor.to(dtype=self.dtype).contiguous()


# Usage example
def main():
    scorer = ImageRewardScorer(device="cuda", dtype=torch.float32)

    images = [
        "test_cases/nasa.jpg",
        "test_cases/hello world.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts = [
        'An astronaut’s glove floating in zero-g with "NASA 2049" on the wrist',
        'New York Skyline with "Hello World" written with fireworks on the sky',
    ]
    print(scorer(prompts, pil_images))


if __name__ == "__main__":
    main()
