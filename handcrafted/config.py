"""Edit this file before running the stages from PyCharm."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("BRISC_ROOT", r"D:\Datasets\Medical\brisc2025"))
OUTPUT_DIR = Path(os.environ.get("BRISC_HANDCRAFTED_OUT", HERE / "outputs"))

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"
TUMOR_CLASSES = ("glioma", "meningioma", "pituitary")
ALL_CLASSES = ("glioma", "meningioma", "no_tumor", "pituitary")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# Optional CSV with sample_id (dataset-relative image path) and patient_id columns.
PATIENT_CSV = None
DHASH_MAX_DISTANCE = 10  # out of 256 bits, as in shortcut_audit
BRAIN_THRESHOLD = 15
RING_EDGES = (0, 3, 7, 14, 25)  # pixels at a 256-pixel short side
TEXTURE_BINS = 32
TEXTURE_SENSITIVITY_BINS = (16, 64)
QC_IMAGES_PER_CLASS = 2

# Empty selection runs all named experiments and F01–F25 individually.
EXPERIMENT_IDS = ()
MODELS = ("logistic", "lda", "svm", "random_forest", "xgboost", "knn")
TUNE_MODELS = True
LDA_MAX_COMPONENTS = 128  # Training-fold PCA for wide LDA inputs. Set to None for full covariance.
CV_FOLDS = 4
MODEL_JOBS = 2
SEED = 42
BOOTSTRAPS = 500

VOCABULARY_SIZES = (32,)  # Add 64, 128 or 256 to compare dictionary sizes in F23.
PCA_COMPONENTS = (32,)  # candidates for the optional mask and appearance PCA controls
RUN_EXHAUSTIVE_SUBSETS = False  # 255 combinations plus an intercept-only baseline
SUBSET_MODEL = "svm"
RUN_TEXTURE_SENSITIVITY = True
PERMUTATION_REPEATS = 5

CNN_SEEDS = (42, 43, 44)
CNN_RUN_FOUR_CLASS = False
CNN_EPOCHS = 15
CNN_SIZE = 128
CNN_BATCH_SIZE = 32
CNN_DEVICE = "auto"  # auto, cpu, or cuda
CNN_CPU_THREADS = 4

ROBUSTNESS_IMAGES_PER_CLASS = 5
ROBUSTNESS_GROUPS = ("F04", "F05", "F06", "F07", "F12", "F15", "F19", "F20", "F21")
