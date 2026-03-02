# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

# Legend: [INF] inference-only, [TRN] training-only, [BOTH] shared by train/infer.
va_flexiv_cfg = EasyDict(__name__="Config: VA flexiv")
va_flexiv_cfg.update(va_shared_cfg)
# [INF] Server mode: keep an online VA service running.
va_shared_cfg.infer_mode = "server"

# [BOTH] Base WAN model checkpoint path.
va_flexiv_cfg.wan22_pretrained_model_name_or_path = "/inspire/hdd/global_user/liuhairuo-253208120281/model/robbyant/lingbot-va-base"

# [BOTH] Temporal attention window (in frames/chunks).
va_flexiv_cfg.attn_window = 30
# [BOTH] Number of frames generated/consumed per rollout chunk.
va_flexiv_cfg.frame_chunk_size = 4
# [BOTH] Environment adapter type, "none" means generic image/action pipeline.
va_flexiv_cfg.env_type = "none"

# Per-camera resolution after `script/extract_lerobot_latents.py` preprocessing
# with default `--max-area 256*256` on 640x480 videos.
# [BOTH] Latent frame height after preprocessing.
va_flexiv_cfg.height = 224
# [BOTH] Latent frame width after preprocessing.
va_flexiv_cfg.width = 288
# [BOTH] Fixed action head dimension expected by the model.
va_flexiv_cfg.action_dim = 30
# 30fps source -> ~10fps sampled frames in extractor => 12 action steps per latent frame.
va_flexiv_cfg.action_per_frame = 12
# [BOTH] Camera keys to load latent features from dataset observations.
va_flexiv_cfg.obs_cam_keys = [
    "observation.images.cam1",
    "observation.images.cam2",
    "observation.images.cam3",
]

# [BOTH] CFG scale for video latent generation branch.
va_flexiv_cfg.guidance_scale = 5
# [BOTH] CFG scale for action branch.
va_flexiv_cfg.action_guidance_scale = 1

# [BOTH] Diffusion sampling steps for video branch.
va_flexiv_cfg.num_inference_steps = 5
# [INF] Execute all planned video steps when set to -1.
va_flexiv_cfg.video_exec_step = -1
# [BOTH] Diffusion sampling steps for action branch.
va_flexiv_cfg.action_num_inference_steps = 10

# [BOTH] Flow matching scheduler shift for video branch.
va_flexiv_cfg.snr_shift = 5.0
# [BOTH] Flow matching scheduler shift for action branch.
va_flexiv_cfg.action_snr_shift = 1.0

# Model action head is fixed 30-dim:
# [left_eef(7), right_eef(7), left_joints(7), right_joints(7), left_gripper(1), right_gripper(1)].
# Dataset action is 16-dim and reordered in dataset loader to:
# [left_eef(7), left_gripper(1), right_eef(7), right_gripper(1)].
# Map source 16 dims -> target 30 dims.
va_flexiv_cfg.used_action_channel_ids = [
    0, 1, 2, 3, 4, 5, 6,      # left eef
    28,                        # left gripper
    7, 8, 9, 10, 11, 12, 13,   # right eef
    29,                        # right gripper
]
# [BOTH] Inverse index map for projecting model outputs back to source action order.
inverse_used_action_channel_ids = [
    len(va_flexiv_cfg.used_action_channel_ids)
] * va_flexiv_cfg.action_dim
for i, j in enumerate(va_flexiv_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_flexiv_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

# [BOTH] Action normalization strategy used by dataset and model IO.
va_flexiv_cfg.action_norm_method = "quantiles"
# [BOTH] Per-channel lower quantile from source dataset action space.
_src_action_min = [
    -0.27296513319015503, -0.3545406758785248, 0.11597598344087601,
    -0.799770176410675, -0.8600184321403503, -0.9667750597000122,
    -0.6736516356468201, 0.0, -0.2811152935028076, -0.5322191119194031,
    0.12220173329114914, -0.6555060148239136, -0.9254750609397888,
    -0.9986172914505005, -0.931738555431366, 0.0,
]
# [BOTH] Per-channel upper quantile from source dataset action space.
_src_action_max = [
    1.0360140800476074, 0.6931226849555969, 0.6360237002372742,
    0.8910780549049377, 0.9769880771636963, 0.9999993443489075,
    0.7667854428291321, 0.08500000089406967, 1.2115975618362427,
    0.3647098243236542, 0.6301588416099548, 0.8612079620361328,
    0.9975616931915283, 0.9999995231628418, 0.7407299280166626,
    0.08500000089406967,
]
# [BOTH] Build target 30-dim quantile stats and fill non-used dims with defaults.
_q01 = [0.0] * va_flexiv_cfg.action_dim
_q99 = [1.0] * va_flexiv_cfg.action_dim
for _src_i, _dst_i in enumerate(va_flexiv_cfg.used_action_channel_ids):
    _q01[_dst_i] = _src_action_min[_src_i]
    _q99[_dst_i] = _src_action_max[_src_i]

va_flexiv_cfg.norm_stat = {
    # [BOTH] Lower quantile per target action channel.
    "q01": _q01,
    # [BOTH] Upper quantile per target action channel.
    "q99": _q99,
}
