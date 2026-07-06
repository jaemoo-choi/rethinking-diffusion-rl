import ml_collections
import os

# task = "geneval"

task = os.getenv("TASK")
method = os.getenv("METHOD")
adv = os.getenv("ADV")
beta = float(os.getenv("BETA", "0.0001"))
alpha = float(os.getenv("ALPHA", "0.0001"))
scale = float(os.getenv("SCALE", "1.0"))
elbo = os.getenv("ELBO")
decay = int(os.getenv("DECAY_TYPE"))
max_grad_norm = float(os.getenv("MAX_GRAD_NORM"))
gradient_step_per_epoch = int(os.getenv("GRADIENT_STEP_PER_EPOCH"))
resume_from = os.getenv("RESUME_FROM", "")
kl_method = os.getenv("KL_METHOD", "girsanov")
learning_rate = float(os.getenv("LEARNING_RATE", "3e-4"))
guidance_scale = float(os.getenv("GUIDANCE_SCALE",1))
ref_guidance_scale = float(os.getenv("REF_GUIDANCE_SCALE",1))
train_guidance_scale = float(os.getenv("TRAIN_GUIDANCE_SCALE",1))
xt_from_latents = os.getenv("XT_FROM_LATENTS", "0") == "1"
time_fraction = float(os.getenv("TIME_FRACTION", "1.0"))

num_inner_epochs = 1
_lr_tag = f"_lr{learning_rate}" if learning_rate != 3e-4 else ""
_xtlat_tag = "_xtlat" if xt_from_latents else ""
_scale_tag = f"_scale{scale}" if scale != 1.0 else ""
_tf_tag = f"_tf{time_fraction}" if time_fraction != 1.0 else ""
run_name = f"{method}_{adv}_{kl_method}_inep{num_inner_epochs}_{task}_beta{beta}_alpha{alpha}_{elbo}_decay{decay}_gradclip{max_grad_norm}_gradstep{gradient_step_per_epoch}_guidance{guidance_scale}_{train_guidance_scale}_{ref_guidance_scale}{_lr_tag}{_xtlat_tag}{_scale_tag}{_tf_tag}"

def get_config():
    config = ml_collections.ConfigDict()

    config.task = task

    ###### General ######
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = run_name
    config.debug = False
    # random seed for reproducibility.
    config.seed = 96
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples. Override per-run with env MAX_EPOCHS; per-model helpers may set a
    # different family default. Default 500 preserves the historical effective
    # cap (train.py previously hard-clamped the loop to 500).
    config.num_epochs = int(os.getenv("MAX_EPOCHS", "500"))
    # number of epochs between saving model checkpoints.
    config.save_freq = 1
    config.eval_freq = 10
    config.sample_log_freq = 10
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "no"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = resume_from
    # whether or not to use LoRA.
    config.use_lora = True
    config.dataset = ""

    ###### Video / Spatial Geometry ######
    # Used by WAN2.1 (and future video models).
    # For image models these are ignored; use config.resolution instead.
    config.frames = 33          # number of video frames (must satisfy VAE constraint, e.g. 4k+1)
    config.height = 480         # spatial height  (must be multiple of 16 for latent alignment)
    config.width = 832          # spatial width   (must be multiple of 16 for latent alignment)
    config.video_fps = 16       # output video fps (used for saving / reward scoring)

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = ""
    # revision of the model to load.
    pretrained.revision = ""

    ###### Sampling ######
    config.sample = sample = ml_collections.ConfigDict()
    
    # number of sampler inference steps.
    sample.num_steps = 40
    sample.eval_num_steps = 40
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = guidance_scale
    # batch size (per GPU!) to use for sampling.
    sample.train_batch_size = 1
    sample.num_image_per_prompt = 1
    sample.test_batch_size = 1
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample.num_batches_per_epoch = 2
    # Whether use all samples in a batch to compute std
    sample.global_std = True
    # noise level
    sample.noise_level = 1.0

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    train.adv = adv 
    train.method = method 
    train.elbo = elbo
    train.kl_method = kl_method
    # train guidance_scale
    train.guidance_scale = train_guidance_scale
    train.ref_guidance_scale = ref_guidance_scale
    # batch size (per GPU!) to use for training.
    train.batch_size = 1
    # learning rate.
    train.learning_rate = learning_rate
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = gradient_step_per_epoch
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = max_grad_norm
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.num_inner_epochs = num_inner_epochs
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = 0.99
    # Fraction of the END of the sampler trajectory where the LoRA adapter is
    # enabled during sampling AND loss is computed during training.
    # 1.0 = original behaviour (adapter active for all steps).
    # 0.6 = base model for first 40% of steps, adapter + training on last 60%.
    # Env-driven via TIME_FRACTION (read at module top); also stamped into
    # run_name as _tf<value> when != 1.0.
    train.time_fraction = time_fraction
    # kl ratio
    train.beta = beta
    train.alpha = alpha
    # advantage scaling factor (applied in "standard" advantage mode where
    # there is no β/α to set the magnitude).
    train.scale = scale
    # pretrained lora path
    train.lora_path = None
    train.ema = True
    # If True, store the full sampling trajectory in the sample dict and use
    # latents[:, j_idx] as xt at training time instead of re-interpolating
    # xt = (1-t)*x0 + t*noise. Keeps training on-policy w.r.t. the actual
    # (stochastic) sampler trajectory. Combine with cpu_offload_samples=True
    # for video models — trajectory memory grows ~num_steps+1x.
    train.xt_from_latents = xt_from_latents

    # Auxiliary differentiable dynamic-degree loss (RAFT-flow-magnitude
    # objective added to total loss before backward). Off by default
    # (lambda=0). When > 0, requires VAE on GPU during training and adds
    # ~1-3 s per microbatch. Three formulations:
    #   "neg":    - lambda * mean(top5_flow)        (unbounded maximize)
    #   "hinge":  + lambda * relu(thres - flow)     (penalty below VBench thresh)
    #   "target": + lambda * relu(target - flow)    (soft target above thresh)
    # See src/reward/dynamic_degree_scorer.py for the underlying scorer.
    train.dynamic_loss_lambda = float(os.getenv("DYNAMIC_LOSS_LAMBDA", "0.0"))
    train.dynamic_loss_kind = os.getenv("DYNAMIC_LOSS_KIND", "neg")
    train.dynamic_loss_target = float(os.getenv("DYNAMIC_LOSS_TARGET", "0.0"))
    train.dynamic_loss_num_pairs = int(os.getenv("DYNAMIC_LOSS_NUM_PAIRS", "8"))

    ###### Prompt Function ######
    # prompt function to use. see `prompts.py` for available prompt functions.
    config.prompt_fn = ""
    # kwargs to pass to the prompt function.
    config.prompt_fn_kwargs = {}

    ###### Reward Function ######
    # reward function to use. see `rewards.py` for available reward functions.
    config.reward_fn = ml_collections.ConfigDict()
    # Custom dimension weights for VideoAlign reward (e.g. {"VQ": 0.3, "MQ": 0.3, "TA": 0.4}).
    # Empty dict means use default equal-weight sum (Overall = VQ + MQ + TA).
    config.videoalign_dimension_weights = ml_collections.ConfigDict()
    config.save_dir = ""

    ###### Per-Prompt Stat Tracking ######
    config.per_prompt_stat_tracking = True

    ###### Memory Optimization ######
    # gradient_checkpointing: recompute activations during backward to save VRAM.
    # Saves ~60-70% activation memory at ~30% more compute. May introduce minor
    # numerical differences vs. no checkpointing. Recommended for large video models.
    config.gradient_checkpointing = False
    # cpu_offload_samples: store collated training samples on CPU and load
    # micro-batches to GPU on demand. Saves ~10-15 GB but adds CPU-GPU transfer
    # overhead per micro-batch. Recommended for large video models.
    config.cpu_offload_samples = False
    # offload_transformer_during_decode: move the transformer to CPU before VAE
    # decode, freeing its VRAM for the decoder. Adds ~2 PCIe transfers per batch
    # (~0.2s for 1.3B, ~1.6s for 13B models). Recommended for video models where
    # the transformer dominates VRAM and VAE decode is memory-intensive.
    config.offload_transformer_during_decode = False
    # save_sample_cache: after each epoch's sampling phase, persist the
    # collated samples (latents, embeds, advantages, rewards) to
    # {save_dir}/checkpoints/sample_cache/epoch_{N}/. On resume, if the cache
    # for the resuming epoch exists, skip sampling and go straight to training.
    # Cache is cleared after the epoch's training phase completes.
    config.save_sample_cache = True

    return config
