#!/usr/bin/env python3
"""Download PickScore models to HF cache"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.paths import HF_HOME, HF_TOKEN_PATHS
from transformers import AutoProcessor, AutoModel

# Set cache directory
cache_dir = HF_HOME
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir

# Check for HF token
token_files = HF_TOKEN_PATHS
token = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)
if token:
    print("Using HuggingFace token from environment")
else:
    for token_file in token_files:
        if os.path.exists(token_file):
            with open(token_file, "r") as f:
                token = f.read().strip()
            print(f"Using HuggingFace token from {token_file}")
            break

if not token:
    print("No HF token found, downloading public models only")

print("\nDownloading laion/CLIP-ViT-H-14-laion2B-s32B-b79K (processor)...")
try:
    processor = AutoProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        cache_dir=cache_dir,
        token=token,
        local_files_only=False,
    )
    print("✓ Processor downloaded successfully")
except Exception as e:
    print(f"✗ Failed to download processor: {e}")

print("\nDownloading yuvalkirstain/PickScore_v1 (model)...")
try:
    model = AutoModel.from_pretrained(
        "yuvalkirstain/PickScore_v1",
        cache_dir=cache_dir,
        token=token,
        local_files_only=False,
    )
    print("✓ Model downloaded successfully")
except Exception as e:
    print(f"✗ Failed to download model: {e}")

print("\n✓ Download complete! Models cached in:", cache_dir)
