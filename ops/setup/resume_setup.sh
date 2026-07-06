#!/bin/bash
# Resume setup from Step 4 (torch already installed)
# Run this on a GPU node:
#   srun --time=4:00:00 --gres=gpu:h200:1 --qos=coe-grade --pty --cpus-per-task=4 --mem-per-cpu=6G bash
#   cd <repo_root>
#   bash ops/resume_setup.sh

set -e

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO}/config/paths.sh"
CONDA_ROOT=$SCRATCH/anaconda3
ENV_PATH=$CONDA_ROOT/envs/image_elbo

export PIP_CACHE_DIR=$SCRATCH/pip-cache
export TMPDIR=$SCRATCH/tmp
export TEMP=$SCRATCH/tmp
export TMP=$SCRATCH/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

echo "ENV_PATH: $ENV_PATH"
"$ENV_PATH/bin/python" -V

# ---- pin pip to a version that still supports legacy builds ----
"$ENV_PATH/bin/python" -m pip install -q "pip==22.3.1" "setuptools<70" wheel
echo "✓ pip pinned to 22.3.1, setuptools<70"

# =========================================
# Step 4: flash-attn  (non-blocking)
# =========================================
echo ""
echo "========================================="
echo "Step 4: flash-attn"
echo "========================================="
module load cuda/12.6
CUDA_HOME=$(which nvcc | sed 's|/bin/nvcc||')
export CUDA_HOME PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0"
echo "CUDA_HOME: $CUDA_HOME"
nvcc --version

if "$ENV_PATH/bin/python" -m pip install --no-build-isolation --no-cache-dir flash-attn 2>&1; then
    echo "✓ flash-attn installed"
else
    echo "⚠ flash-attn failed — continuing as recommended by README"
fi

# =========================================
# Step 5: Reward checkpoints
# =========================================
echo ""
echo "========================================="
echo "Step 5: Download reward checkpoints"
echo "========================================="
mkdir -p "$REPO/reward_ckpts"
cd "$REPO/reward_ckpts"
wget -qc https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/refs/heads/main/sac+logos+ava1-l14-linearMSE.pth
wget -qc https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth
wget -qc https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
wget -qc https://huggingface.co/xswu/HPSv2/resolve/main/HPS_v2.1_compressed.pt
cd "$REPO"
echo "✓ Checkpoints downloaded"

# =========================================
# Step 6: MMCV v1.7.2 with CUDA ops
# =========================================
echo ""
echo "========================================="
echo "Step 6: Build MMCV v1.7.2"
echo "========================================="
# Re-export CUDA (module state can reset)
module load cuda/12.6
CUDA_HOME=$(which nvcc | sed 's|/bin/nvcc||')
export CUDA_HOME PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="9.0"
echo "CUDA_HOME: $CUDA_HOME"

# Clean any previous partial clone
rm -rf "$REPO/mmcv" "$REPO/mmcv_src"

git clone https://github.com/open-mmlab/mmcv.git "$REPO/mmcv"
cd "$REPO/mmcv"
git fetch --tags -q
git checkout v1.7.2

echo "Building (this takes ~15-30 min)…"
# Use setup.py develop directly — avoids all pip PEP 660 / --no-use-pep517 version issues
MMCV_WITH_OPS=1 FORCE_CUDA=1 "$ENV_PATH/bin/python" setup.py develop 2>&1 | tee "$REPO/mmcv_build.log"

cd "$REPO"
mv mmcv mmcv_src
echo "✓ MMCV v1.7.2 built"

# =========================================
# Step 7: Verify mmcv._ext
# =========================================
echo ""
echo "========================================="
echo "Step 7: Verify mmcv._ext"
echo "========================================="
cd "$REPO/mmcv_src"   # run from inside so mmcv package is found via .egg-link
"$ENV_PATH/bin/python" -c "
import mmcv
print('mmcv version:', mmcv.__version__, '|', mmcv.__file__)
import mmcv._ext
print('mmcv._ext OK:', mmcv._ext.__file__)
"
cd "$REPO"

# =========================================
# Step 8: MMDetection v2.28.2
# =========================================
echo ""
echo "========================================="
echo "Step 8: MMDetection v2.28.2"
echo "========================================="
rm -rf "$REPO/mmdetection" "$REPO/mmdetection_src"
git clone https://github.com/open-mmlab/mmdetection.git "$REPO/mmdetection"
cd "$REPO/mmdetection"
git fetch --tags -q
git checkout v2.28.2
"$ENV_PATH/bin/python" setup.py develop 2>&1 | tee "$REPO/mmdet_build.log"
cd "$REPO"
mv mmdetection mmdetection_src
echo "✓ MMDetection v2.28.2 installed"

# =========================================
# Step 9: Integration check
# =========================================
echo ""
echo "========================================="
echo "Step 9: Integration check"
echo "========================================="
cd "$REPO/mmcv_src"
"$ENV_PATH/bin/python" -c "
import mmcv, mmdet
print('mmcv:', mmcv.__version__, mmcv.__file__)
import mmcv._ext; print('mmcv._ext:', mmcv._ext.__file__)
print('mmdet:', mmdet.__version__, mmdet.__file__)
print('✓ Integration check passed!')
"
cd "$REPO"

# =========================================
# Step 10: Extra reward packages
# =========================================
echo ""
echo "========================================="
echo "Step 10: Extra packages"
echo "========================================="
"$ENV_PATH/bin/python" -m pip install open-clip-torch clip-benchmark
"$ENV_PATH/bin/python" -m pip install paddlepaddle-gpu==2.6.2
"$ENV_PATH/bin/python" -m pip install paddleocr==2.9.1
"$ENV_PATH/bin/python" -m pip install python-Levenshtein
"$ENV_PATH/bin/python" -m pip install hpsv2x==1.2.0
"$ENV_PATH/bin/python" -m pip install image-reward
"$ENV_PATH/bin/python" -m pip install git+https://github.com/openai/CLIP.git
echo "✓ All extra packages installed"

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo "Activate with:"
echo "  source $SCRATCH/anaconda3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_PATH"
