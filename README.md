# Rethinking the Design Space of Reinforcement Learning for Diffusion Models

### On the Importance of Likelihood Estimation Beyond Loss Design

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![ICML](https://img.shields.io/badge/ICML-2026-8A2BE2)](https://icml.cc/)

This is the official implementation of **Rethinking the Design Space of Reinforcement Learning for Diffusion Models**.

<p align="center">
  <img src="assets/training_curves_eval_reward.png" width="62%" />
  <img src="assets/concept.png" width="34%" />
</p>

> **Training efficiency and design-space analysis for reward-based diffusion fine-tuning.**
> *(Left)* GenEval performance across training time for various fine-tuning methods on SD3.5-Medium.
> *(Right)* Conceptual summary of the design space considered in this work, highlighting policy-gradient loss design, likelihood estimation, and sampling strategy.

We provide a systematic analysis of the RL design space for diffusion/flow models by disentangling three factors: **(i)** policy-gradient objectives, **(ii)** likelihood estimators, and **(iii)** rollout sampling schemes. We show that adopting an **evidence lower bound (ELBO) based likelihood estimator, computed only from the final generated sample**, is the dominant factor enabling effective, efficient, and stable RL optimization — outweighing the impact of the specific policy-gradient loss functional. On SD 3.5 Medium, our method improves the GenEval score from **0.24 to 0.95 in 90 GPU hours**, which is **4.6× more efficient than FlowGRPO** and **2× more efficient than the SOTA method DiffusionNFT** without reward hacking.

## Update

**[2026]** Paper accepted to **ICML 2026**.

**[2026]** Code released for RL fine-tuning of SD3 and Flux.

## Environment Installation

The image training stack lives in a single conda env (`image_elbo`). It is pinned to specific versions because the image reward models depend on legacy builds of **MMCV 1.7.2** and **MMDetection 2.28.2** (do not upgrade these).

**Core versions:** Python 3.10.16 · CUDA 12.6 · PyTorch 2.6.0 · torchvision 0.21.0 · transformers 4.40.0 · diffusers 0.33.1 · accelerate 1.4.0 · peft 0.10.0 · deepspeed 0.16.4 · numpy 1.26.4 · tokenizers 0.19.1

**One-shot setup script** (run on a GPU node — it builds MMCV/MMDet CUDA ops and downloads reward checkpoints):

```bash
# Allocate a GPU node first, e.g.:
#   srun --time=4:00:00 --gres=gpu:h100:1 --pty --cpus-per-task=4 --mem-per-cpu=6G bash

git clone https://github.com/<your-org>/rethinking-diffusion-rl.git
cd rethinking-diffusion-rl
bash ops/setup/setup_image_elbo.sh
```

The script:
1. Creates the `image_elbo` conda env (Python 3.10.16) and `pip install -e .` (see `setup.py`).
2. Installs `flash-attn` (non-blocking; the run continues if the build fails).
3. Downloads reward checkpoints into `reward_ckpts/` (Aesthetic predictor, Mask2Former for GenEval, LAION CLIP-ViT-H-14, HPS-v2.1).
4. Builds **MMCV 1.7.2** and **MMDetection 2.28.2** with CUDA ops.
5. Installs extra reward packages (open-clip, PaddleOCR 2.9.1 for OCR reward, HPSv2, ImageReward, OpenAI CLIP).

**Activate the env** (paths come from `config/paths.sh` — the single source of truth for `HF_HOME`, conda paths, and W&B entity; source it in every shell script instead of hardcoding paths):

```bash
source config/paths.sh
source "${CONDA_SH}"
conda activate "${CONDA_ENV_IMAGE}"
```

> **HuggingFace cache:** all weights are cached at `HF_HOME` (defined in `config/paths.sh`) in **offline mode** by default. Never download to `~/.cache`.

## Project Structure

```
rethinking-diffusion-rl/
├── setup.py                        # Package + pinned dependencies (image_elbo)
├── config/
│   ├── base.py                     # Base config (env-var driven)
│   ├── paths.py / paths.sh         # Shared HF_HOME / conda env paths
│   ├── sd3.py                      # SD3 base + ODE/SDE named configs
│   └── flux.py                     # Flux base + ODE named configs
├── src/
│   ├── train.py                    # Distributed training entrypoint
│   ├── flow_grpo.py                # GRPO training loop
│   ├── dataloader.py               # All dataset classes and dataloaders
│   ├── ema.py  prompts.py
│   ├── utils/                      # Per-model pipeline utils (uniform interface)
│   │   ├── __init__.py             #   compute_elbo(), PerPromptStatTracker, distributed helpers
│   │   ├── base.py  sd3.py  flux.py
│   ├── diffusers_patch/            # Patched pipelines with log-prob tracking
│   │   ├── solver.py               #   Shared ODE/SDE solvers (flow/dance/ddim/dpm)
│   │   └── pipeline_with_logprob_{sd3,flux}.py
│   └── reward/                     # All reward scorers (see below)
├── dataset/                        # Training/evaluation prompt sets
│   ├── geneval/  geneval_unseen_objects/  ocr/  pickscore/  drawbench/
├── ops/                            # Env setup + reward/base-model download scripts
│   ├── setup/                      #   setup_image_elbo.sh
│   └── download/                   #   download_sd3_5.sh, reward downloads
├── scripts/                        # SLURM training scripts (bucketed per model)
│   ├── sd3/  flux/
├── reward_ckpts/                   # Pre-trained reward model weights
└── logs/                           # Checkpoints & training outputs
```

## Reward Functions

All reward scoring lives in `src/reward/`; `multi_score()` in `rewards.py` aggregates any weighted combination of scorers per config.

| Reward | File | Description |
|--------|------|-------------|
| **GenEval** | `gen_eval.py` | Compositional generation benchmark (MMDetection object/attribute/relation eval). |
| **CLIP** | `clip_scorer.py` | Text–image alignment via CLIP cosine similarity. |
| **Aesthetic** | `aesthetic_scorer.py` | Learned aesthetic quality predictor on CLIP features. |
| **PickScore** | `pickscore_scorer.py` | Human-preference-aligned scoring model. |
| **ImageReward** | `imagereward_scorer.py` | BLIP-based human-preference reward. |
| **HPS-v2** | `hpsv2_scorer.py` | Human Preference Score v2. |
| **OCR** | `ocr.py` | Text-rendering accuracy via PaddleOCR detection. |
| **UnifiedReward** | `unifiedreward_scorer.py` | Multi-aspect unified reward model. |

## Training

Training is launched with `torchrun`. Each config name maps to a function in `config/<model>.py` (dispatched via `get_config`). ODE variants come first; SDE variants are suffixed `_sde`.

```bash
source config/paths.sh
source "${CONDA_SH}" && conda activate "${CONDA_ENV_IMAGE}"

# SD 3.5 Medium on GenEval with ELBO-based likelihood estimation
torchrun --nproc_per_node=8 src/train.py --config config/sd3.py:sd3_geneval
```

Representative SD3 configs (`config/sd3.py`):

| Config | Reward | Notes |
|--------|--------|-------|
| `sd3_geneval` | GenEval | ELBO likelihood, ODE sampling (default) |
| `sd3_geneval_sde` | GenEval | ELBO likelihood, SDE sampling |
| `sd3_ocr` | OCR | Text rendering |
| `sd3_pickscore` | PickScore | Human preference |
| `sd3_hpsv2` | HPS-v2 | Human preference |
| `sd3_multi_reward` | Multiple | Weighted combination |
| `sd3_pickscore_flow_grpo` | PickScore | FlowGRPO (trajectory-based) baseline |
| `sd3_geneval_diffusionnft_4step_shift5` | GenEval | DiffusionNFT baseline, 4-step |

Flux (`config/flux.py`: `flux_geneval`, `flux_geneval_flow_grpo`, …) follows the same pattern. On the cluster, ready-made SLURM scripts live in `scripts/<model>/` (e.g. `scripts/sd3/sd3_geneval_epg.sh`).

## Results

<p align="center">
  <img src="assets/first_exceed_095_bar.png" width="70%" />
</p>

> **Training time comparison on GenEval.** Total GPU hours (8× H100) required to reach a GenEval score of 0.95. ELBO-based likelihood estimation substantially reduces training cost compared to trajectory-based approaches, and ODE sampling further improves efficiency at the same target performance.

<p align="center">
  <img src="assets/all_comparisons4.png" width="90%" />
</p>

> Qualitative comparison between baseline benchmarks and our model.

## Acknowledgement

This codebase builds upon the diffusion RL fine-tuning ecosystem, including [Flow-GRPO](https://github.com/yifan123/flow_grpo), [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT), and [DanceGRPO](https://github.com/XueZeyue/DanceGRPO). Reward models are adapted from [GenEval](https://github.com/djghosh13/geneval), [PickScore](https://github.com/yuvalkirstain/PickScore), [ImageReward](https://github.com/THUDM/ImageReward), [HPSv2](https://github.com/tgxs002/HPSv2), and [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR). We thank the authors for open-sourcing their work.

## Citation

If you find our code or paper useful, please consider citing:

```bibtex
@inproceedings{choi2026rethinking,
  title     = {Rethinking the Design Space of Reinforcement Learning for Diffusion Models: On the Importance of Likelihood Estimation Beyond Loss Design},
  author    = {Choi, Jaemoo and Zhu, Yuchen and Guo, Wei and Molodyk, Petr and Yuan, Bo and Bai, Jinbin and Xin, Yi and Tao, Molei and Chen, Yongxin},
  booktitle = {Forty-third International Conference on Machine Learning},
  year      = {2026}
}
```
