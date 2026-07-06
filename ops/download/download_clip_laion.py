#!/usr/bin/env python3
"""Download LAION CLIP model for PickScore"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.paths import HF_HOME
from transformers import AutoProcessor, AutoModel

# Set cache directory
cache_dir = HF_HOME
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir

print("Downloading laion/CLIP-ViT-H-14-laion2B-s32B-b79K...")
print("This model is needed for PickScore processor\n")

try:
    print("Step 1: Downloading processor...")
    processor = AutoProcessor.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        cache_dir=cache_dir
    )
    print("✓ Processor downloaded\n")
    
    print("Step 2: Downloading model weights...")
    model = AutoModel.from_pretrained(
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        cache_dir=cache_dir
    )
    print("✓ Model downloaded\n")
    
    print("✓ Successfully downloaded laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    print(f"✓ Cached in: {cache_dir}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nNote: This model requires internet access to download from HuggingFace.")
