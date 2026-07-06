#!/bin/bash
#SBATCH --job-name=dl-qwen2vl
#SBATCH --qos=coe-grade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --output=./sbatch_logs/dl-qwen2vl-O
#SBATCH --error=./sbatch_logs/dl-qwen2vl-E

# Download Qwen/Qwen2-VL-7B-Instruct (~15 GB) — backbone of HPSv3 reward.
# Without this cached, HF_HUB_OFFLINE=1 will reject HPSv3 init.

set -euo pipefail
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
source "${REPO_DIR}/config/paths.sh"
export HF_HUB_OFFLINE=0
source "${CONDA_SH}"
conda activate "${CONDA_ENV_VIDEO}"

python <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id="Qwen/Qwen2-VL-7B-Instruct", repo_type="model")
print("Qwen2-VL-7B-Instruct cached at:", p)
PY
