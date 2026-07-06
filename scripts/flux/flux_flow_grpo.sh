#!/bin/bash
#SBATCH --job-name=flux_flowgrpo
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:8
#SBATCH --mem-per-cpu=128G
#SBATCH --time=02:00:00
#SBATCH --output=./sbatch_logs/train-%x-O
#SBATCH --error=./sbatch_logs/train-%x-E
#SBATCH --exclude=atl1-1-03-012-3-0,atl1-1-03-012-28-0,atl1-1-03-014-9-0,atl1-1-03-014-16-0,atl1-1-03-010-20-0,atl1-1-03-011-13-0,atl1-1-03-013-26-0,atl1-1-03-012-18-0,atl1-1-03-015-2-0,atl1-1-03-011-18-0,atl1-1-03-014-2-0,atl1-1-03-015-9-0,atl1-1-03-015-30-0,atl1-1-03-017-30-0,atl1-1-03-010-15-0,atl1-1-03-011-23-0,atl1-1-03-011-28-0,atl1-1-03-014-23-0

# ORIGINAL Flow-GRPO (yifan123/flow_grpo) on FLUX + GenEval (image).
# Same dataset/reward/geometry as flux_geneval (num_steps=10); sampler -> flow SDE.
# Uses flow_grpo.py. Image guidance kept at flow_grpo image default. NOISE_LEVEL = eta.
export NUM_GPUS=8
export MIN_GPUS=4
export ALLOW_GPU_6=false
export FORCE_TIER=H200:8
export TASK="geneval"
export MAX_EPOCHS=360
export METHOD="flow_grpo"
export ADV="standard"
export KL_METHOD="girsanov"
export BETA=0
export ALPHA=0.0001
export SCALE=1.0
export ELBO="adaptive"
export DECAY_TYPE=0
export MAX_GRAD_NORM=1.0
export GRADIENT_STEP_PER_EPOCH=2
export GUIDANCE_SCALE=3.5
export TRAIN_GUIDANCE_SCALE=3.5
export REF_GUIDANCE_SCALE=3.5
export LEARNING_RATE=5e-5
export NOISE_LEVEL=0.7
export XT_FROM_LATENTS=0
export NCCL_TIMEOUT_SECONDS=1800
export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_DIR}/config/paths.sh"

source "${CONDA_SH}"
conda activate "${CONDA_ENV_IMAGE}"

MASTER_PORT=$((29500 + RANDOM % 1000))
srun torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NUM_GPUS}" src/flow_grpo.py \
    --config config/flux.py:flux_geneval_flow_grpo
