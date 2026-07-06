#!/bin/bash
# image_elbo Full Environment Setup
# Based on README_IMAGE_ELBO_SETUP.md — all known fixes applied.
#
# Prerequisites: run on a GPU node
#   srun --time=4:00:00 --gres=gpu:h100:1 --qos=coe-grade --pty --cpus-per-task=4 --mem-per-cpu=6G bash
#
# Usage:
#   cd <repo_root>
#   bash ops/setup_image_elbo.sh

set -e

# ── Core paths ────────────────────────────────────────────────────────────────
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO}/config/paths.sh"
CONDA_ROOT=$SCRATCH/anaconda3
ENV_PATH=$CONDA_ROOT/envs/image_elbo

export PIP_CACHE_DIR=$SCRATCH/pip-cache
export TMPDIR=$SCRATCH/tmp
export TEMP=$SCRATCH/tmp
export TMP=$SCRATCH/tmp

# ── Helpers ───────────────────────────────────────────────────────────────────
PY="$ENV_PATH/bin/python"

load_cuda() {
    module load cuda/12.6
    CUDA_HOME=$(which nvcc | sed 's|/bin/nvcc||')
    export CUDA_HOME
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export TORCH_CUDA_ARCH_LIST="9.0"
    echo "  CUDA_HOME : $CUDA_HOME"
    echo "  ARCH_LIST : $TORCH_CUDA_ARCH_LIST"
}

# =========================================
# Step 1 — Environment variables
# =========================================
echo ""
echo "========================================="
echo "Step 1: Scratch environment variables"
echo "========================================="
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
unset CONDA_EXE CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PYTHON_EXE
export CONDA_NO_PLUGINS=true
echo "  ENV_PATH : $ENV_PATH"
echo "  REPO     : $REPO"
echo "✓ Done"

# =========================================
# Step 1.5 — Hard reset
# =========================================
echo ""
echo "========================================="
echo "Step 1.5: Hard reset"
echo "========================================="
rm -rf "$ENV_PATH"
rm -rf "$TMPDIR"/* "$PIP_CACHE_DIR"/*
rm -rf "$REPO"/mmcv "$REPO"/mmcv_src
rm -rf "$REPO"/mmdetection "$REPO"/mmdetection_src
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
echo "✓ Done"

# =========================================
# Step 2 — Create conda env
# =========================================
echo ""
echo "========================================="
echo "Step 2: Create image_elbo env (Python 3.10.16)"
echo "========================================="
"$CONDA_ROOT/bin/conda" create -y -p "$ENV_PATH" python=3.10.16
"$PY" -V
echo "✓ Done"

# =========================================
# Step 3 — Base project dependencies
# =========================================
echo ""
echo "========================================="
echo "Step 3: Base project dependencies"
echo "========================================="
cd "$REPO"
# Pin pip and setuptools to versions compatible with legacy build systems (MMCV 1.7.2, MMDet 2.28.2)
"$PY" -m pip install -q "pip==22.3.1" "setuptools<70" wheel
"$PY" -m pip install -q -U ninja
"$PY" -m pip install -q psutil
"$PY" -m pip install -e .
"$PY" -m pip install -q -U openmim mmengine
echo "✓ Done"

# =========================================
# Step 4 — flash-attn  (non-blocking)
# =========================================
echo ""
echo "========================================="
echo "Step 4: flash-attn (non-blocking)"
echo "========================================="
load_cuda
nvcc --version
if "$PY" -m pip install --no-build-isolation --no-cache-dir flash-attn 2>&1; then
    echo "✓ flash-attn installed"
else
    echo "⚠ flash-attn failed — continuing as recommended by README"
fi

# =========================================
# Step 5 — Reward checkpoints
# =========================================
echo ""
echo "========================================="
echo "Step 5: Reward checkpoints"
echo "========================================="
mkdir -p "$REPO/reward_ckpts"
cd "$REPO/reward_ckpts"
wget -qc https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth
wget -qc https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
wget -qc https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
wget -qc https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt
cd "$REPO"
echo "✓ Done"

# =========================================
# Step 6 — MMCV v1.7.2 with CUDA ops
# =========================================
echo ""
echo "========================================="
echo "Step 6: Build MMCV v1.7.2 with CUDA ops  (~15-30 min)"
echo "========================================="
load_cuda
git clone https://github.com/open-mmlab/mmcv.git
cd "$REPO/mmcv"
git fetch --tags -q
git checkout v1.7.2
# Use setup.py develop directly — avoids all pip PEP-517/660 version conflicts
MMCV_WITH_OPS=1 FORCE_CUDA=1 "$PY" setup.py develop 2>&1 | tee "$REPO/mmcv_build.log"
cd "$REPO"
mv mmcv mmcv_src
echo "✓ Done"

# =========================================
# Step 7 — Verify mmcv._ext
# =========================================
echo ""
echo "========================================="
echo "Step 7: Verify mmcv._ext"
echo "========================================="
# Run from repo root (not source tree) so import resolution matches real usage
cd "$REPO"
"$PY" -c "
import mmcv
print('mmcv version:', mmcv.__version__)
print('mmcv path   :', mmcv.__file__)
import mmcv._ext
print('mmcv._ext   :', mmcv._ext.__file__)
print('✓ mmcv._ext OK')
"
cd "$REPO"

# =========================================
# Step 8 — MMDetection v2.28.2
# =========================================
echo ""
echo "========================================="
echo "Step 8: MMDetection v2.28.2"
echo "========================================="
git clone https://github.com/open-mmlab/mmdetection.git
cd "$REPO/mmdetection"
git fetch --tags -q
git checkout v2.28.2
"$PY" setup.py develop 2>&1 | tee "$REPO/mmdet_build.log"
cd "$REPO"

# Verify mmdet import immediately to catch editable-link/path issues early
echo ""
echo "-----------------------------------------"
echo "Post-Step 8 check: Verify mmdet import"
echo "-----------------------------------------"
if "$PY" -c "import mmdet; print('mmdet:', mmdet.__version__, mmdet.__file__)"; then
    echo "✓ mmdet import OK"
else
    echo "✗ mmdet import failed after Step 8"
    echo "  Python executable: $PY"
    echo "  Last lines of mmdet build log:"
    tail -n 80 "$REPO/mmdet_build.log" || true
    exit 1
fi

echo "✓ Done"

# =========================================
# Step 9 — Integration check
# =========================================
echo ""
echo "========================================="
echo "Step 9: Final integration check"
echo "========================================="
cd "$REPO"
"$PY" -c "
import sys
import mmcv, mmdet
print('python:', sys.executable)
print('mmcv :', mmcv.__version__, mmcv.__file__)
import mmcv._ext; print('mmcv._ext:', mmcv._ext.__file__)
print('mmdet:', mmdet.__version__, mmdet.__file__)
print('✓ Integration check passed!')
"
cd "$REPO"

# =========================================
# Step 10 — Extra reward packages
# =========================================
echo ""
echo "========================================="
echo "Step 10: Extra reward-related packages"
echo "========================================="
"$PY" -m pip install open-clip-torch clip-benchmark
"$PY" -m pip install paddlepaddle-gpu==2.6.2
"$PY" -m pip install paddleocr==2.9.1
"$PY" -m pip install python-Levenshtein
"$PY" -m pip install hpsv2x==1.2.0
"$PY" -m pip install image-reward
"$PY" -m pip install git+https://github.com/openai/CLIP.git
echo "✓ Done"

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo ""
echo "Activate with:"
echo "  source $SCRATCH/anaconda3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_PATH"
