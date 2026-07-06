#!/bin/bash
#SBATCH --job-name=dl-sd3_5
#SBATCH --qos=coe-grade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --time=02:00:00
#SBATCH --output=./sbatch_logs/dl-sd3_5-O
#SBATCH --error=./sbatch_logs/dl-sd3_5-E

# Download stabilityai/stable-diffusion-3.5-medium into HF_HOME.
# Gated model — requires:
#   1. `export HF_TOKEN=hf_...` before submission (token is inherited by SLURM)
#   2. License accepted at https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
#
# Usage: export HF_TOKEN=hf_xxx; sbatch ops/download_sd3_5.sh

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${REPO_DIR}/config/paths.sh"
# Explicitly un-gate HF for the duration of this job.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# HF_TOKEN must be inherited from the submitting shell.
if [ -z "${HF_TOKEN}" ]; then
    echo "ERROR: HF_TOKEN is empty. Run 'export HF_TOKEN=hf_...' before sbatch." >&2
    exit 2
fi
# Mirror to HUGGING_FACE_HUB_TOKEN (older libs) + persist to the token file so
# subsequent offline jobs pick it up automatically via paths.sh.
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
mkdir -p "${HOME}/.cache/huggingface"
printf '%s' "${HF_TOKEN}" > "${HOME}/.cache/huggingface/token"
chmod 600 "${HOME}/.cache/huggingface/token"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_IMAGE}"

cd "${REPO_DIR}"

python <<'PY'
import os
from huggingface_hub import snapshot_download

cache_dir = os.path.join(os.environ["HF_HOME"], "hub")
model_id = "stabilityai/stable-diffusion-3.5-medium"

print(f"Downloading {model_id} to {cache_dir} ...")
path = snapshot_download(model_id, cache_dir=cache_dir, repo_type="model")
print(f"Done: {path}")

# Offline sanity check
os.environ["HF_HUB_OFFLINE"] = "1"
import torch
from diffusers import StableDiffusion3Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
print("Pipeline loaded offline OK")
print("  transformer:", type(pipe.transformer).__name__)
print("  vae:", type(pipe.vae).__name__)
print("  text_encoder:", type(pipe.text_encoder).__name__)
PY
