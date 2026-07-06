import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


# ============================================
# Flux ODE base config
# ============================================
def _flux_ode_base_config(n_gpus=1, gradient_step_per_epoch=1, dataset="pickscore", reward_fn={}, name="", num_image_per_prompt=24):
    config = base.get_config()
    assert dataset in ["pickscore", "ocr", "geneval", "geneval_unseen_objects"]

    config.base_model = "flux"
    # Image fine-tuning default: 360 epochs. Override per-run with env MAX_EPOCHS.
    config.num_epochs = int(os.getenv("MAX_EPOCHS", "360"))
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    bsz = 3 if n_gpus % 8 == 0 else 4

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


# ---------- Flux ODE configs ----------

def flux_geneval():
    config = _flux_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="geneval",
        reward_fn={"geneval": 1.0},
        name="geneval",
    )
    return config


def flux_geneval_bf16():
    config = _flux_ode_base_config(
        n_gpus=int(os.getenv("NUM_GPUS")),
        gradient_step_per_epoch=int(os.getenv("GRADIENT_STEP_PER_EPOCH")),
        dataset="geneval",
        reward_fn={"geneval": 1.0},
        name="geneval",
    )
    config.mixed_precision = "bf16"
    return config


def flux_geneval_flow_grpo():
    """FLUX GenEval with the ORIGINAL Flow-GRPO objective (yifan123/flow_grpo).

    Same dataset / reward / geometry (num_steps=10) as ``flux_geneval``; only
    the SAMPLER becomes the stochastic flow SDE (flux's base hardcodes dpm2/
    deterministic, so the override is mandatory). Run via
    ``scripts/flux_flow_grpo.sh``. ``NOISE_LEVEL`` sets eta (default 0.7).
    """
    config = flux_geneval()
    config.name = "flux_geneval_flow_grpo"
    config.mixed_precision = "bf16"
    config.sample.solver = "flow"
    config.sample.deterministic = False
    config.sample.noise_level = float(os.getenv("NOISE_LEVEL", "0.7"))
    # Flow-GRPO evaluates at the SAME step count it trains at (no denoising
    # reduction): eval_num_steps == num_steps (=10), matching the video configs.
    config.sample.eval_num_steps = config.sample.num_steps
    config.run_name = config.run_name + f"_eta{config.sample.noise_level}"
    config.save_dir = f"logs/{config.name}/{config.run_name}"
    return config
