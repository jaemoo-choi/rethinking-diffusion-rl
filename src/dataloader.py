import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch
import json
from PIL import Image


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_testloader(config, world_size, rank, num_workers=0):
    if config.prompt_fn in ["general_ocr", "text"]:
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError(f"Prompt function not supported with dataset: {config.prompt_fn}")

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_dataloader


def get_dataloaders(config, world_size, rank, num_workers=0):
    """Build train/test dataloaders from config.

    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    if config.prompt_fn in ["general_ocr", "text"]:
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError(f"Prompt function not supported with dataset: {config.prompt_fn}")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader

