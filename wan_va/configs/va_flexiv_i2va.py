# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_flexiv_cfg import va_flexiv_cfg

# Legend: [INF] inference-only, [TRN] training-only, [BOTH] shared by train/infer.
va_flexiv_i2va_cfg = EasyDict(__name__="Config: VA flexiv i2va")
# [INF] Start from base flexiv config and override i2va-only fields.
va_flexiv_i2va_cfg.update(va_flexiv_cfg)

# [INF] Directory containing initial images named by camera key, e.g. cam1.png.
va_flexiv_i2va_cfg.input_img_path = "example/flexiv"
# [INF] Number of rollout chunks to autoregressively generate.
va_flexiv_i2va_cfg.num_chunks_to_infer = 10
# [INF] Text instruction for video+action generation.
va_flexiv_i2va_cfg.prompt = "Pick up the cloth and wipe the table surface."
# [INF] Switch inference branch to image-to-video-action mode.
va_flexiv_i2va_cfg.infer_mode = "i2va"
