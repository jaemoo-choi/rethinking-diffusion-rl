#!/bin/bash
#SBATCH --job-name=sd3_gen_pepg
#SBATCH --qos=coe-grade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H200:8
#SBATCH --mem-per-cpu=128G
#SBATCH --time=02:00:00
#SBATCH --output=./sbatch_logs/train-%x-O
#SBATCH --error=./sbatch_logs/train-%x-E
#SBATCH --exclude=atl1-1-03-012-3-0,atl1-1-03-012-28-0,atl1-1-03-014-9-0,atl1-1-03-014-16-0,atl1-1-03-010-20-0,atl1-1-03-011-13-0,atl1-1-03-013-26-0,atl1-1-03-012-18-0,atl1-1-03-015-2-0,atl1-1-03-011-18-0,atl1-1-03-014-2-0,atl1-1-03-015-9-0,atl1-1-03-015-30-0,atl1-1-03-017-30-0,atl1-1-03-010-15-0,atl1-1-03-011-23-0,atl1-1-03-011-28-0,atl1-1-03-014-23-0

# SD3.5-medium on GenEval, standard 10-step DPM2 ODE sampling (config sd3_geneval),
# trained for 360 epochs with the PEPG policy objective:
#     -(dr * (A - dlogr) * fELBO)
# Proximal EPG: adds the -dlogr correction, dlogr = fELBO.detach() - oELBO.
#
# Shared backdrop reproduces the reference paper's elbo_ode PEPG recipe
# (jaemoo-choi/pce-dm@nft-pce-dm) — ONLY `METHOD` differs across the 5 scripts:
#   ADV="exact"  => reference "no-proximal": A = BETA*(R-mean)/ALPHA = 0.1*(R-mean)
#                   (NOT std-normalized; "no-proximal" is unnamed in our tracker,
#                    "exact" is our formula-equivalent branch).
#   BETA=1e-4    => Girsanov KL coefficient AND the advantage numerator (dual-use,
#                   exactly as upstream base.py reuses BETA in both places).
#   ALPHA=1e-3, DECAY_TYPE=1, MAX_GRAD_NORM=5.0, GRADIENT_STEP_PER_EPOCH=1,
#   LEARNING_RATE=3e-4, guidance held at 1.0 (no CFG, matching elbo_ode ODE).
export NUM_GPUS=8
# MIN_GPUS=6 (not 4): the K-repeat sampler needs num_replicas*bsz % k(=24) == 0.
# 8gpu->bsz9->72 and 6gpu->bsz8->48 both divide by 24; 4gpu->bsz8->32 does NOT.
export MIN_GPUS=6
export ALLOW_GPU_6=true
export TASK="geneval"
export MAX_EPOCHS=360
export METHOD="pepg"
export ADV="exact"
export KL_METHOD="girsanov"
export BETA=0.0001
export ALPHA=0.001
export SCALE=1.0
export ELBO="adaptive"
export DECAY_TYPE=1
export MAX_GRAD_NORM=5.0
export GRADIENT_STEP_PER_EPOCH=1
export GUIDANCE_SCALE=1.0
export TRAIN_GUIDANCE_SCALE=1.0
export REF_GUIDANCE_SCALE=1.0
export LEARNING_RATE=3e-4
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
srun torchrun --master_port="${MASTER_PORT}" --nproc_per_node="${NUM_GPUS}" src/train.py \
    --config config/sd3.py:sd3_geneval
