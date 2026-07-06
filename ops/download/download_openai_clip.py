#!/usr/bin/env python3
"""Download OpenAI CLIP for AestheticScorer and ClipScorer"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.paths import HF_HOME
from transformers import CLIPModel, CLIPProcessor

# Set cache directory
cache_dir = HF_HOME
os.environ["HF_HOME"] = cache_dir
os.environ["HF_DATASETS_CACHE"] = cache_dir

print("Downloading openai/clip-vit-large-patch14...")
print("This model is needed for AestheticScorer and ClipScorer\n")

try:
    print("Step 1: Downloading CLIP model...")
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
        cache_dir=cache_dir
    )
    print("✓ CLIP model downloaded\n")
    
    print("Step 2: Downloading CLIP processor...")
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14",
        cache_dir=cache_dir
    )
    print("✓ CLIP processor downloaded\n")
    
    print("✓ Successfully downloaded openai/clip-vit-large-patch14")
    print(f"✓ Cached in: {cache_dir}")
    
except Exception as e:
    print(f"✗ Error: {e}")
