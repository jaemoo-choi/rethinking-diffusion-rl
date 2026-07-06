#!/usr/bin/env python3
"""Download ImageReward model weights into HF cache."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.paths import HF_HOME

import ImageReward as RM


cache_dir = HF_HOME
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_dir, "datasets")

print("Downloading ImageReward-v1.0...")
print(f"HF_HOME={cache_dir}")

try:
    model = RM.load(
        "ImageReward-v1.0",
        device="cpu",
        download_root=os.path.join(cache_dir, "ImageReward"),
    )
    _ = model
    print("✓ ImageReward downloaded successfully")
except Exception as e:
    print(f"✗ Failed to download ImageReward: {e}")
