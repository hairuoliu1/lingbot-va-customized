# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_flexiv_cfg import va_flexiv_cfg

# Legend: [INF] inference-only, [TRN] training-only, [BOTH] shared by train/infer.
va_flexiv_train_cfg = EasyDict(__name__="Config: VA flexiv train")
# [TRN] Start from base flexiv config and add training-only fields.
va_flexiv_train_cfg.update(va_flexiv_cfg)

# [TRN] Root path containing one or multiple LeRobot repos with extracted latents.
va_flexiv_train_cfg.dataset_path = "/inspire/hdd/global_user/liuhairuo-253208120281/cache/huggingface/lerobot"
# [TRN] Experiment folder name; outputs are saved under save_root/exp_name/.
va_flexiv_train_cfg.exp_name = "test"
# [TRN] Auto resume from save_root/exp_name/train_state if it exists.
va_flexiv_train_cfg.auto_resume = True
# [TRN] Optional explicit checkpoint directory for resume; None uses auto_resume.
va_flexiv_train_cfg.resume_from = None
# [TRN] Cached empty text embedding used by classifier-free guidance in training.
va_flexiv_train_cfg.empty_emb_path = "/inspire/hdd/global_user/liuhairuo-253208120281/lingbot-va/empty_emb.pt"
# [TRN] Enable TensorBoard logging on rank 0.
va_flexiv_train_cfg.enable_tensorboard = True
# [TRN] Number of DataLoader workers.
va_flexiv_train_cfg.load_worker = 16
# [TRN] Number of workers used when building sub-datasets in MultiLatentLeRobotDataset.
va_flexiv_train_cfg.dataset_init_worker = 32
# [TRN] Save checkpoint every N optimizer steps.
va_flexiv_train_cfg.save_interval = 100
# [TRN] Keep interval checkpoints (and latest). Set 0 to keep all checkpoints.
va_flexiv_train_cfg.keep_interval = 1000
# [TRN] Run garbage collection every N steps.
va_flexiv_train_cfg.gc_interval = 100
# [TRN] Probability of dropping condition (CFG training).
va_flexiv_train_cfg.cfg_prob = 0.1
# [TRN] AdamW learning rate.
va_flexiv_train_cfg.learning_rate = 1e-5
# [TRN] AdamW beta1.
va_flexiv_train_cfg.beta1 = 0.9
# [TRN] AdamW beta2.
va_flexiv_train_cfg.beta2 = 0.95
# [TRN] AdamW weight decay.
va_flexiv_train_cfg.weight_decay = 0.1
# [TRN] LR warmup steps before constant schedule.
va_flexiv_train_cfg.warmup_steps = 10
# [TRN] Per-GPU micro-batch size.
va_flexiv_train_cfg.batch_size = 1
# [TRN] Number of gradient accumulation steps.
va_flexiv_train_cfg.gradient_accumulation_steps = 1
# [TRN] Total optimization steps.
va_flexiv_train_cfg.num_steps = 5000
# [TRN] Random seed used by DistributedSampler shuffling.
va_flexiv_train_cfg.sampler_seed = 42
# [TRN] Number of diffusion timesteps used by training schedulers.
va_flexiv_train_cfg.train_num_timesteps = 1000
# [TRN] Probability of adding extra noise to latent condition branch.
va_flexiv_train_cfg.latent_noisy_cond_prob = 0.5
# [TRN] Probability of adding extra noise to action condition branch.
va_flexiv_train_cfg.action_noisy_cond_prob = 0.0
# [TRN] Minimum sampled chunk_size for train-time attention mask.
va_flexiv_train_cfg.chunk_size_min = 1
# [TRN] Maximum sampled chunk_size for train-time attention mask. Set min=max to disable random sampling.
va_flexiv_train_cfg.chunk_size_max = 2
# [TRN] Minimum sampled window_size for train-time block attention mask.
va_flexiv_train_cfg.window_size_min = 4
# [TRN] Maximum sampled window_size for train-time block attention mask. Set min=max to disable random sampling.
va_flexiv_train_cfg.window_size_max = 32
# [TRN] Gradient clipping threshold (L2 norm).
va_flexiv_train_cfg.max_grad_norm = 2.0
# [TRN] AdamW epsilon for numerical stability.
va_flexiv_train_cfg.adam_eps = 1e-8
# [TRN] Enable periodic preview video export from training latents.
va_flexiv_train_cfg.enable_preview_video = True
# [TRN] Save one preview video every N optimizer steps.
va_flexiv_train_cfg.preview_video_interval = 100
# [TRN] Enable validation loop.
va_flexiv_train_cfg.enable_validation = True
# [TRN] Validation split ratio from full dataset.
va_flexiv_train_cfg.val_split_ratio = 0.01
# [TRN] Run validation every N optimizer steps.
va_flexiv_train_cfg.val_interval = 100
# [TRN] Number of validation dataloader workers.
va_flexiv_train_cfg.val_worker = 16
# [TRN] Max validation batches per eval (-1 means full validation set).
va_flexiv_train_cfg.val_eval_num_batches = -1
