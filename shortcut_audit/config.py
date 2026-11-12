"""
Central configuration for the shortcut-learning audit.

EDIT ONLY THIS FILE (or set the environment variable BRISC_ROOT) before running anything.
All other scripts import their paths and hyper-parameters from here.
"""
import os
from pathlib import Path

# --------------------------------------------------------------------------------------
# 1. WHERE IS THE DATA?
# --------------------------------------------------------------------------------------
# Folder that contains `classification_task/` and `segmentation_task/` (BRISC 2025).
# Windows example: r"D:\Datasets\Medical\brisc2025"
DATA_ROOT = Path(os.environ.get("BRISC_ROOT", r"D:\Datasets\Medical\brisc2025"))

CLASSIFICATION_DIR = DATA_ROOT / "classification_task"   # <split>/<class>/<image>.jpg
SEGMENTATION_DIR = DATA_ROOT / "segmentation_task"       # <split>/images|masks/...

# Splits that may exist under classification_task. Missing ones are ignored.
#   TRAIN_SPLIT : used for fitting models / probes
#   VAL_SPLIT   : used for choosing the best epoch (if it does not exist, 10 % of train is held out)
#   TEST_SPLIT  : used for EVERY reported number (never trained on)
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
TUMOR_CLASSES = ["glioma", "meningioma", "pituitary"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# --------------------------------------------------------------------------------------
# 2. WHERE DO RESULTS GO?
# --------------------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
OUT_DIR = Path(os.environ.get("AUDIT_OUT", PROJECT_DIR / "outputs"))
INDEX_CSV = OUT_DIR / "index.csv"            # one row per image (path, class, split, mask path, ...)
FEATURES_CSV = OUT_DIR / "features.csv"      # morphometric / location features
RUNS_DIR = OUT_DIR / "runs"                  # trained models + per-image predictions
RESULTS_DIR = OUT_DIR / "results"            # tables / figures used in the report

# --------------------------------------------------------------------------------------
# 3. EXPERIMENT SETTINGS
# --------------------------------------------------------------------------------------
SEEDS = [0, 1, 2]            # >= 3 seeds so that every number comes with a spread
IMAGE_SIZE = int(os.environ.get('AUDIT_SIZE', 224))
BATCH_SIZE = 32
EPOCHS = int(os.environ.get('AUDIT_EPOCHS', 10))
LR = float(os.environ.get('AUDIT_LR', 1e-4))
NUM_WORKERS = 0              # keep 0 on Windows/PyCharm unless you know you need more
PRETRAINED = os.environ.get('AUDIT_PRETRAINED','1')=='1'            # ImageNet weights through timm (needs internet the first time)

# Architectures used in the study (timm names): one CNN family, one classic CNN, one transformer.
ARCHS = ["efficientnet_b0", "resnet50", "vit_small_patch16_224"]

# Intervention strengths: how many pixels the tumor mask is dilated before it is erased.
DILATION_SWEEP = [0, 5, 10, 20]
N_BOOTSTRAP = 1000

# Perceptual-hash threshold (Hamming distance out of 256 bits) for "near duplicate".
NEAR_DUP_HAMMING = 10

# Brain segmentation threshold (grey level) used for geometry features.
BRAIN_THRESHOLD = 15

for _d in (OUT_DIR, RUNS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
