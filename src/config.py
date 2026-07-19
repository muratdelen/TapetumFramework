"""Configuration file for RetinexTapetum."""

import os
import torch


def env_to_bool(name: str, default: bool = False) -> bool:
    """Safely convert an environment variable to a boolean value."""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"", "0", "false", "no", "n", "off", "none", "null"}:
        return False
    raise ValueError(f"Invalid boolean environment variable: {name}={value!r}")


PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, os.pardir))

DATA_NAME = os.environ.get("RETINEX_DATA_NAME", "LOL-v2")
_data_variant_env = os.environ.get("RETINEX_DATA_VARIANT", "Real_captured")
DATA_VARIANT = None if str(_data_variant_env).strip().lower() in {"", "none", "null"} else str(_data_variant_env).strip()

VALID_DATA_NAMES = {"LOL-v1", "LOL-v2", "UHD-LL down4", "SICE", "LoLI-Street"}
VALID_DATA_VARIANTS = {"Real_captured", "Synthetic", None}

if DATA_NAME not in VALID_DATA_NAMES:
    raise ValueError(f"Invalid DATA_NAME: {DATA_NAME!r}. Valid options: {sorted(VALID_DATA_NAMES)}")
if DATA_VARIANT not in VALID_DATA_VARIANTS:
    raise ValueError(f"Invalid DATA_VARIANT: {DATA_VARIANT!r}. Valid options: Real_captured, Synthetic, None")

if DATA_VARIANT is None:
    LOCAL_DATA_ROOT = os.path.join(WORKSPACE_ROOT, "datasets", DATA_NAME)
else:
    LOCAL_DATA_ROOT = os.path.join(WORKSPACE_ROOT, "datasets", DATA_NAME, DATA_VARIANT)

COLAB_BASE_DATA_ROOT = "/content/drive/MyDrive/TAPETUM/datasets"
if DATA_VARIANT is None:
    COLAB_DATA_ROOT = os.path.join(COLAB_BASE_DATA_ROOT, DATA_NAME)
else:
    COLAB_DATA_ROOT = os.path.join(COLAB_BASE_DATA_ROOT, DATA_NAME, DATA_VARIANT)

DATA_ROOT = COLAB_DATA_ROOT if os.path.exists(COLAB_DATA_ROOT) else LOCAL_DATA_ROOT

if DATA_VARIANT is None:
    RUN_ROOT = os.path.join(WORKSPACE_ROOT, DATA_NAME, "RetinexTapetum")
else:
    RUN_ROOT = os.path.join(WORKSPACE_ROOT, DATA_NAME, f"RetinexTapetum-{DATA_VARIANT}")

TRAIN_LOW_DIR = os.path.join(DATA_ROOT, "Train", "Low")
TRAIN_HIGH_DIR = os.path.join(DATA_ROOT, "Train", "Normal")
TEST_LOW_DIR = os.path.join(DATA_ROOT, "Test", "Low")
TEST_HIGH_DIR = os.path.join(DATA_ROOT, "Test", "Normal")
USE_TRAIN_VAL_SPLIT = True
VAL_RATIO = 0.10
SPLIT_SEED = 42
VAL_LOW_DIR = os.path.join(DATA_ROOT, "Val", "Low")
VAL_HIGH_DIR = os.path.join(DATA_ROOT, "Val", "Normal")

CKPT_DIR = os.path.join(RUN_ROOT, "checkpoints")
RESULT_DIR = os.path.join(RUN_ROOT, "results", "Test")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
BATCH_SIZE = 2
NUM_WORKERS = 2
CROP_SIZE = 256
EPOCHS = 120
LR = 0.00046948014087030786
MIN_LR = 2.12826357198832e-05
GRAD_CLIP_NORM = 3.0
SHOW_PROGRESS_BARS = False
TRAIN_LOG_INTERVAL = 50
VAL_LOG_INTERVAL = 25

LAMBDA_MAX = 1.35
LAMBDA_INIT = 0.0

W_L1 = 1.0
W_SSIM = 0.35
W_COLOR = 0.06
W_CHROMA = 0.12
W_ATTN = 0.012
W_EDGE = 0.04
W_DARK_NOISE = 0.01

USE_LPIPS_LOSS = True
LPIPS_NET = "alex"
W_LPIPS = 0.08
LPIPS_LOSS_RESIZE = 256
LPIPS_METRIC_RESIZE = 0
BEST_MODEL_METRIC = "psnr"

W_RECON_LOW = 1.0
W_RECON_HIGH = 0.85
W_REFLECT = 0.06
W_SMOOTH_LOW = 0.08
W_SMOOTH_HIGH = 0.08
W_SMOOTH_ENH = 0.06

PATIENCE = 45
SEED = 42
BASE_CHANNELS = 128
RESUME_TRAINING = env_to_bool("RETINEX_TAPETUM_RESUME", default=False)
RESUME_CKPT_NAME = os.environ.get("RETINEX_TAPETUM_RESUME_CKPT", "last.pth")

COLOR_RESTORE = False
COLOR_RESTORE_STRENGTH = 0.0
COLOR_RESTORE_EPS = 1e-4
