import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


# ============================================
# SD3 ODE base config
# ============================================
def _sd3_ode_base_config(n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name="", num_image_per_prompt=24):
    config = base.get_config()
    assert dataset in ["pickscore", "ocr", "geneval", "geneval_unseen_objects"]

    config.base_model = "sd3"
    # Image fine-tuning default: 360 epochs. Override per-run with env MAX_EPOCHS
    # (e.g. the 4-step runs set MAX_EPOCHS=10000).
    config.num_epochs = int(os.getenv("MAX_EPOCHS", "360"))
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    bsz = 9 if n_gpus % 8 == 0 else 8

    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.resolution = 512

    config.sample.num_image_per_prompt = num_image_per_prompt
    num_groups = 48
    n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
    assert n_batch_per_epoch % gradient_step_per_epoch == 0
    config.sample.train_batch_size = bsz
    config.sample.num_batches_per_epoch = n_batch_per_epoch
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // gradient_step_per_epoch

    if dataset in ["geneval", "geneval_unseen_objects"]:
        config.sample.test_batch_size = 14
    else:
        config.sample.test_batch_size = 16
    config.prompt_fn = "geneval" if dataset in ["geneval", "geneval_unseen_objects"] else "text"

    config.name = name
    config.save_dir = f"logs/{name}/{config.run_name}"
    config.reward_fn = reward_fn

    config.decay_type = int(os.getenv("DECAY_TYPE"))
    config.train.adv_mode = "all"

    config.sample.deterministic = True
    config.sample.solver = "dpm2"
    config.sample.noise_level = 0.7

    return config


# ---------- SD3 ODE configs ----------

def sd3_ocr():
    config = _sd3_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="ocr",
        reward_fn={"ocr": 1.0},
        name="ocr",
    )
    return config


def sd3_geneval():
    config = _sd3_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="geneval",
        reward_fn={"geneval": 1.0},
        name="geneval",
    )
    return config


def sd3_geneval_unseen_objects():
    config = _sd3_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="geneval_unseen_objects",
        reward_fn={"geneval": 1.0},
        name="geneval_unseen_objects",
    )
    return config


def sd3_pickscore():
    config = _sd3_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="pickscore",
        reward_fn={"pickscore": 1.0},
        name="pickscore",
    )
    return config


def sd3_hpsv2():
    config = _sd3_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="pickscore",
        reward_fn={"hpsv2": 1.0},
        name="hpsv2",
    )
    return config


def sd3_multi_reward():
    config = _sd3_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="pickscore",
        reward_fn={"pickscore": 1.0, "hpsv2": 1.0, "clipscore": 1.0},
        name="multi_reward",
    )
    config.sample.num_steps = 25
    return config


def sd3_ocr_general_reward():
    config = _sd3_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="ocr",
        reward_fn={"pickscore": 1.0, "hpsv2": 1.0, "clipscore": 1.0, "ocr": 1.0},
        name="multi_reward",
    )
    config.sample.num_steps = 25
    return config


def sd3_geneval_4step_shift12():
    """SD3.5-medium on GenEval, 4-step DPM2 sampling with flow-matching
    shift=12.0 (few-step, high-shift).

    Identical dataset (``geneval``) / reward (``geneval``) / recipe to
    ``sd3_geneval``; the ONLY differences are ``num_steps`` 10->4 and the
    flow-matching ``shift`` -> 12.0 (consumed by SD3TrainingUtils via
    BaseTrainingUtils._maybe_override_flow_shift after pipeline load). Distinct
    ``config.name`` so checkpoints land in ``logs/sd3_geneval_4step_shift12/``.
    Drive from ``scripts/sd3_geneval_4step_shift12_pepg.sh``
    (``METHOD=pepg_noratio``, ``XT_FROM_LATENTS=1``).
    """
    config = _sd3_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="geneval",
        reward_fn={"geneval": 1.0},
        name="sd3_geneval_4step_shift12",
    )
    config.sample.num_steps = 4
    config.sample.eval_num_steps = 4
    config.sample.shift = 12.0
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return config


def sd3_geneval_diffusionnft_4step_shift12():
    """DiffusionNFT contrastive flow-matching loss on the 4-step / shift=12.0
    SD3 GenEval recipe (see ``sd3_geneval_4step_shift12``).

    Same dataset / reward / geometry (num_steps=4, shift=12.0) as
    ``sd3_geneval_4step_shift12``; only the training objective differs. Drive
    from ``scripts/sd3_geneval_4step_shift12_diffusionnft_decay2.sh``
    with ``METHOD=diffusionnft ADV=diffusionnft`` (reward mapped to r∈[0,1] in
    PerPromptStatTracker), ``DIFFUSIONNFT_BETA`` = the contrastive guidance
    strength γ, ``BETA`` = the Girsanov KL coefficient (≈1e-4), ``DECAY_TYPE=2``
    KL warmup, ``XT_FROM_LATENTS=1``. Checkpoints land in
    ``logs/sd3_geneval_diffusionnft_4step_shift12/``.
    """
    config = sd3_geneval_4step_shift12()
    config.name = "sd3_geneval_diffusionnft_4step_shift12"
    config.train.diffusionnft_beta = float(os.getenv("DIFFUSIONNFT_BETA", "0.1"))
    config.run_name = config.run_name + f"_nftg{config.train.diffusionnft_beta}"
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return config


def sd3_geneval_4step_shift5():
    """SD3.5-medium on GenEval, 4-step DPM2 sampling with flow-matching
    shift=5.0.

    Identical to ``sd3_geneval_4step_shift12`` except the flow-matching
    ``shift`` is 5.0 instead of 12.0. Distinct ``config.name`` so checkpoints
    land in ``logs/sd3_geneval_4step_shift5/``. Drive from
    ``scripts/sd3_geneval_4step_shift5_pepg_noratio_std_beta0.sh``.
    """
    config = sd3_geneval_4step_shift12()
    config.name = "sd3_geneval_4step_shift5"
    config.sample.shift = 5.0
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return config


def sd3_geneval_diffusionnft_4step_shift5():
    """DiffusionNFT contrastive flow-matching loss on the 4-step / shift=5.0
    SD3 GenEval recipe (see ``sd3_geneval_4step_shift5``).

    Same as ``sd3_geneval_diffusionnft_4step_shift12`` but with shift=5.0.
    Drive from ``scripts/sd3_geneval_4step_shift5_diffusionnft_beta0.sh``.
    Checkpoints land in ``logs/sd3_geneval_diffusionnft_4step_shift5/``.
    """
    config = sd3_geneval_4step_shift5()
    config.name = "sd3_geneval_diffusionnft_4step_shift5"
    config.train.diffusionnft_beta = float(os.getenv("DIFFUSIONNFT_BETA", "0.1"))
    config.run_name = config.run_name + f"_nftg{config.train.diffusionnft_beta}"
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return config


# ============================================
# SD3 SDE base config
# ============================================
def _sd3_sde_base_config(n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name=""):
    config = base.get_config()
    assert dataset in ["pickscore", "ocr", "geneval", "geneval_unseen_objects"]

    config.base_model = "sd3"
    # Image fine-tuning default: 360 epochs. Override per-run with env MAX_EPOCHS
    # (e.g. the 4-step runs set MAX_EPOCHS=10000).
    config.num_epochs = int(os.getenv("MAX_EPOCHS", "360"))
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 40
    config.sample.eval_num_steps = 40
    config.resolution = 512
    bsz = 9 if n_gpus % 8 == 0 else 8
    bsz = int(os.getenv("BATCH_SIZE", bsz))

    config.sample.num_image_per_prompt = 24
    num_groups = 48

    while True:
        if bsz < 1:
            assert False, "Cannot find a proper batch size."
        if (
            num_groups * config.sample.num_image_per_prompt % (n_gpus * bsz) == 0
            and bsz * n_gpus % config.sample.num_image_per_prompt == 0
        ):
            n_batch_per_epoch = num_groups * config.sample.num_image_per_prompt // (n_gpus * bsz)
            if n_batch_per_epoch % gradient_step_per_epoch == 0:
                config.sample.train_batch_size = bsz
                config.sample.num_batches_per_epoch = n_batch_per_epoch
                config.train.batch_size = config.sample.train_batch_size
                config.train.gradient_accumulation_steps = (
                    config.sample.num_batches_per_epoch // gradient_step_per_epoch
                )
                break
        bsz -= 1

    if dataset in ["geneval", "geneval_unseen_objects"]:
        config.sample.test_batch_size = 14
    else:
        config.sample.test_batch_size = 16
    if n_gpus > 32:
        config.sample.test_batch_size = config.sample.test_batch_size // 2

    config.prompt_fn = "geneval" if dataset in ["geneval", "geneval_unseen_objects"] else "general_ocr"

    config.reward_fn = reward_fn

    config.decay_type = int(os.getenv("DECAY_TYPE"))
    config.beta = 1.0
    config.train.adv_mode = "all"

    config.sample.guidance_scale = float(os.getenv("GUIDANCE_SCALE"))
    config.num_steps = 40 if config.sample.guidance_scale == 1 else 10
    config.sample.deterministic = False
    config.sample.solver = "flow"
    config.sample.noise_level = float(os.getenv("NOISE_LEVEL"))
    config.run_name = f'{config.run_name}_noiselvl{config.sample.noise_level}_guidance{config.sample.guidance_scale}'
    config.save_dir = f"logs/{config.run_name}"
    return config


# ---------- SD3 SDE configs ----------

def sd3_ocr_sde():
    config = _sd3_sde_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="ocr",
        reward_fn={"ocr": 1.0},
        name="ocr",
    )
    config.beta = 0.1
    return config


def sd3_geneval_sde():
    config = _sd3_sde_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="geneval",
        reward_fn={"geneval": 1.0},
        name="geneval",
    )
    return config


def sd3_pickscore_sde():
    config = _sd3_sde_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="pickscore",
        reward_fn={"pickscore": 1.0},
        name="pickscore",
    )
    return config


def sd3_hpsv2_sde():
    config = _sd3_sde_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="pickscore",
        reward_fn={"hpsv2": 1.0},
        name="hpsv2",
    )
    return config


def sd3_geneval_unseen_objects_sde():
    config = _sd3_sde_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="geneval_unseen_objects",
        reward_fn={"geneval": 1.0},
        name="geneval_unseen_objects",
    )
    return config


def sd3_multi_reward_sde():
    config = _sd3_sde_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="pickscore",
        reward_fn={"pickscore": 1.0, "hpsv2": 1.0, "clipscore": 1.0},
        name="multi_reward",
    )
    config.sample.num_steps = 40
    config.beta = 0.1
    return config


def sd3_pickscore_flow_grpo():
    """SD3 PickScore with the ORIGINAL Flow-GRPO objective (yifan123/flow_grpo).

    Thin alias over ``sd3_pickscore_sde`` (which already uses the stochastic
    flow SDE sampler, solver='flow', noise_level from NOISE_LEVEL); only the
    name/folder differ so the ``flow_grpo.py`` run gets its own save_dir. Run
    via ``scripts/sd3_flow_grpo.sh`` (``METHOD=flow_grpo ADV=standard
    BETA=0 SCALE=1.0``).
    """
    config = sd3_pickscore_sde()
    config.name = "sd3_pickscore_flow_grpo"
    # Flow-GRPO evaluates at the SAME step count it trains at (no denoising
    # reduction): eval_num_steps == num_steps, matching the video configs.
    config.sample.eval_num_steps = config.sample.num_steps
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return config
