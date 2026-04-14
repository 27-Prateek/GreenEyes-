# import os
# from pathlib import Path

# # ── Paths ─────────────────────────────────────────────────────────────────────
# ROOT_DIR        = Path(__file__).parent.parent
# DATA_RAW        = ROOT_DIR / "data" / "raw"
# DATA_PROCESSED  = ROOT_DIR / "data" / "processed"
# CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
# RESULTS_DIR     = ROOT_DIR / "results"

# # ── Data ──────────────────────────────────────────────────────────────────────
# RANDOM_SEED    = 42
# VAL_YEAR       = 2019
# TEST_YEAR      = 2020
# WINDOW_SIZE    = 168
# FORECAST_HOURS = [1, 24, 72]

# POLLUTANT_COLS = [
#     'PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
#     'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
# ]

# AQI_CATEGORY_MAP = {
#     'Good': 0, 'Satisfactory': 1, 'Moderate': 2,
#     'Poor': 3, 'Very Poor': 4, 'Severe': 5,
# }
# AQI_CATEGORIES = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

# # ── Preprocessing ─────────────────────────────────────────────────────────────
# SAVGOL_WINDOW    = 49
# SAVGOL_WINDOW_D  = 7
# SAVGOL_POLYORDER = 2
# SHORT_GAP_HOURS  = 3
# MEDIUM_GAP_HOURS = 6

# # ── Model Architecture ────────────────────────────────────────────────────────
# N_FEATURES       = None
# N_CITIES         = 26
# HIDDEN_DIM       = 128
# WAVENET_LAYERS   = [8, 5, 3]
# KERNEL_SIZE      = 3
# LSTM_LAYERS      = 2
# LSTM_DROPOUT     = 0.2
# GNN_HEADS        = 4
# N_CATEGORIES     = 6
# DROPOUT          = 0.15
# CITY_GRAPH_KM    = 500

# # ── Training ──────────────────────────────────────────────────────────────────
# EPOCHS           = 50
# BATCH_SIZE       = 128
# LEARNING_RATE    = 1e-3
# WEIGHT_DECAY     = 1e-4
# WARMUP_EPOCHS    = 8
# GRAD_CLIP        = 1.0
# PATIENCE         = 10
# LOSS_ALPHA       = 0.75
# LOSS_BETA        = 0.25

# # ── Evaluation ────────────────────────────────────────────────────────────────
# CONFORMAL_ALPHA  = 0.10

# # ── Reproducibility ───────────────────────────────────────────────────────────
# import torch
# import numpy as np
# import random

# def set_seed(seed: int = RANDOM_SEED):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark     = False





# configs/config.py
# ============================================================
# AQI-Sense India — Master Configuration
# Tuned for RTX 3050 Laptop GPU (4 GB VRAM)
# ============================================================

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent
DATA_RAW        = ROOT_DIR / "data" / "raw"
DATA_PROCESSED  = ROOT_DIR / "data" / "processed"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR     = ROOT_DIR / "results"

# ── Data ──────────────────────────────────────────────────────────────────────
RANDOM_SEED    = 42
VAL_YEAR       = 2019          # ← matches your current config
TEST_YEAR      = 2020          # ← matches your current config
WINDOW_SIZE    = 168           # 7 days of hourly data
FORECAST_HOURS = [1, 24, 72]

POLLUTANT_COLS = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
    'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene'
]

AQI_CATEGORY_MAP = {
    'Good': 0, 'Satisfactory': 1, 'Moderate': 2,
    'Poor': 3, 'Very Poor': 4, 'Severe': 5,
}
AQI_CATEGORIES = ['Good', 'Satisfactory', 'Moderate', 'Poor', 'Very Poor', 'Severe']

# ── Preprocessing ─────────────────────────────────────────────────────────────
SAVGOL_WINDOW    = 49          # must be odd; 49h ≈ 2 days smoothing window
SAVGOL_WINDOW_D  = 7
SAVGOL_POLYORDER = 2
SHORT_GAP_HOURS  = 3
MEDIUM_GAP_HOURS = 6

# ── Model Architecture ────────────────────────────────────────────────────────
N_FEATURES       = None        # set automatically after preprocessing
N_CITIES         = 26
HIDDEN_DIM       = 64          # ← reduced from 128; fits 4 GB VRAM comfortably
                               #   increase to 128 if you enable mixed precision
WAVENET_LAYERS   = [8, 5, 3]  # same as paper
KERNEL_SIZE      = 3
LSTM_LAYERS      = 2
LSTM_DROPOUT     = 0.2
GNN_HEADS        = 4
N_CATEGORIES     = 6
DROPOUT          = 0.15
CITY_GRAPH_KM    = 500

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS           = 4
BATCH_SIZE       = 32          # ← safe for 4 GB VRAM with HIDDEN_DIM=64
                               #   use GRAD_ACCUM_STEPS=4 for effective batch of 128
GRAD_ACCUM_STEPS = 4           # ← gradient accumulation: effective batch = 32×4 = 128
LEARNING_RATE    = 1e-3
WEIGHT_DECAY     = 1e-4
WARMUP_EPOCHS    = 8
GRAD_CLIP        = 1.0
PATIENCE         = 15
LOSS_ALPHA       = 0.75        # regression loss weight
LOSS_BETA        = 0.25        # classification loss weight

# ── Mixed Precision ───────────────────────────────────────────────────────────
USE_AMP          = True        # automatic mixed precision — halves VRAM usage
                               # lets you use HIDDEN_DIM=128 if desired

# ── Evaluation ────────────────────────────────────────────────────────────────
CONFORMAL_ALPHA  = 0.10        # 90% conformal prediction intervals

# ── Reproducibility ───────────────────────────────────────────────────────────
import torch
import numpy as np
import random

def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False