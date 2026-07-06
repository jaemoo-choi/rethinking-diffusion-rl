"""config/paths.py — Single source of truth for all external path dependencies (Python).

Shell scripts should source config/paths.sh instead. This module provides
the same canonical defaults for Python code that may run without the shell wrapper.
"""
import os

# Repository root (this file lives at <repo>/config/paths.py)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HuggingFace cache directory (defaults to <repo>/hf_cache; override via HF_HOME)
HF_HOME = os.environ.get("HF_HOME", os.path.join(_REPO_ROOT, "hf_cache"))

# HuggingFace token search paths
HF_TOKEN_PATHS = [
    os.path.expanduser("~/.cache/huggingface/token"),
    os.path.expanduser("~/.huggingface/token"),
]

# Weights & Biases entity (set WANDB_ENTITY in your environment)
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "")
