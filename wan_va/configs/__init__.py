# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_flexiv_cfg import va_flexiv_cfg
from .va_flexiv_i2va import va_flexiv_i2va_cfg
from .va_flexiv_train_cfg import va_flexiv_train_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'flexiv': va_flexiv_cfg,
    'flexiv_train': va_flexiv_train_cfg,
    'franka': va_franka_cfg,
    'flexiv_i2av': va_flexiv_i2va_cfg,
    'flexiv_i2va': va_flexiv_i2va_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
}