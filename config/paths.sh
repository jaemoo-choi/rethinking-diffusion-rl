#!/usr/bin/env bash
# config/paths.sh — Single source of truth for all external path dependencies.
# Source this file in every shell script instead of hardcoding paths.
# All values use ${VAR:-default} so environment overrides still work.

# ── Repository root (this file lives at <repo>/config/paths.sh) ─
_PATHS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(dirname "${_PATHS_DIR}")}"

# ── User scratch directory ────────────────────────────────────
export SCRATCH="${SCRATCH:-${HOME}/scratch}"

# ── Conda ─────────────────────────────────────────────────────
export CONDA_SH="${CONDA_SH:-${SCRATCH}/anaconda3/etc/profile.d/conda.sh}"
export CONDA_ENV_IMAGE="${CONDA_ENV_IMAGE:-${SCRATCH}/anaconda3/envs/image_elbo}"
export CONDA_ENV_VIDEO="${CONDA_ENV_VIDEO:-${SCRATCH}/anaconda3/envs/video_elbo}"
export CONDA_ENV_WORLDCOMPASS="${CONDA_ENV_WORLDCOMPASS:-${SCRATCH}/anaconda3/envs/worldcompass_elbo}"
export CONDA_ENV_NFT="${CONDA_ENV_NFT:-nft}"

# ── HuggingFace cache ────────────────────────────────────────
export HF_HOME="${HF_HOME:-${REPO_ROOT}/hf_cache}"
# Do NOT set TRANSFORMERS_CACHE — it is deprecated and, when set to HF_HOME
# directly, bypasses the HF_HUB_CACHE={HF_HOME}/hub layout, causing
# LocalEntryNotFoundError in offline mode. HF_HOME alone routes transformers
# to the correct {HF_HOME}/hub cache.
unset TRANSFORMERS_CACHE
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_PREFER_LOCAL_FILES="${HF_PREFER_LOCAL_FILES:-1}"

# ── HuggingFace token ────────────────────────────────────────
HF_TOKEN_FILE="${HF_TOKEN_FILE:-${HOME}/.cache/huggingface/token}"
if [[ -f "${HF_TOKEN_FILE}" ]]; then
    export HUGGING_FACE_HUB_TOKEN=$(cat "${HF_TOKEN_FILE}")
    export HF_TOKEN=$(cat "${HF_TOKEN_FILE}")
fi

# ── Weights & Biases ─────────────────────────────────────────
export WANDB_ENTITY="${WANDB_ENTITY:-}"
