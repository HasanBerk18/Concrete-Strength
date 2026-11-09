"""Central configuration: paths, column names, split fractions, seeds and search grids."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_XLS = ROOT / "data" / "raw" / "Concrete_Data.xls"
RESULTS_DIR = ROOT / "results"

# Column order follows the UCI file (Yeh, 1998).
RAW_COLUMNS = [
    "cement", "slag", "fly_ash", "water", "superplasticizer",
    "coarse_agg", "fine_agg", "age", "strength",
]
MIX_COLUMNS = RAW_COLUMNS[:7]          # identical values here = same mix
TARGET = "strength"
# Model inputs: 7 mix components + log(age) + engineered ratios.
FEATURES = MIX_COLUMNS + ["log_age", "w_c", "w_b"]

# Grouped (by mix ID) train / validation / test split.
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)
MAIN_SEED = 42
REPEAT_SEEDS = (42, 7, 123, 2024, 31415)

# Curing-age and strength groups for the error breakdown.
AGE_BINS = (0, 7, 28, 100, 365)
AGE_LABELS = ("1-7 d", "14-28 d", "56-100 d", "120-365 d")
STRENGTH_BINS = (0, 20, 40, 60, float("inf"))
STRENGTH_LABELS = ("<20 MPa", "20-40 MPa", "40-60 MPa", ">60 MPa")

# Display order; also fixes each model's colour in every figure.
MODEL_ORDER = ["LinearRegression", "RandomForest", "SVR", "MLP-A", "MLP-B", "MLP-C"]
MLP_NAMES = {"MLP-A", "MLP-B", "MLP-C"}

# ---------------- ANN settings ----------------
MLP_ARCHS = {
    "MLP-A": (32,),
    "MLP-B": (128, 64, 32),
    "MLP-C": (128, 64, 32),   # MLP-B + dropout + L2
}
MAX_EPOCHS = 2000
PATIENCE = 50                  # early stopping on validation loss
ANN_GRID = {"lr": [1e-3, 3e-3], "batch_size": [16, 32]}
# MLP-C reuses MLP-B's best lr/batch size and tunes only the regularisation.
REG_GRID = {"dropout": [0.1, 0.2], "l2": [1e-4, 1e-3]}

# ---------------- Conventional baselines ----------------
RF_GRID = {
    "n_estimators": [300],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [1.0, 0.5, "sqrt"],
}
SVR_GRID = {
    "C": [10, 30, 100, 300],
    "gamma": ["scale", 0.05, 0.1, 0.3],
    "epsilon": [0.5, 1.0, 2.0],
}

# --fast smoke-test overrides
FAST_MAX_EPOCHS = 150
FAST_PATIENCE = 15
