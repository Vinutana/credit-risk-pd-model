"""
config.py — Centralised Configuration for Credit Risk Modeling
==============================================================
All paths, hyperparameters, constants and logging settings live here.
No hardcoded paths anywhere else in the codebase.

Usage
-----
    from src.config import RAW_DATA_PATH, MODEL_PATH, LOGGER
"""

import logging
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Project Layout
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR        = PROJECT_ROOT / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
ARTIFACT_DIR    = PROJECT_ROOT / "artifacts"
REPORTS_DIR     = PROJECT_ROOT / "reports"
LOG_DIR         = PROJECT_ROOT / "logs"

# Create directories at import time so callers never need to mkdir
for _d in [PROCESSED_DIR, ARTIFACT_DIR, REPORTS_DIR, LOG_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Data Paths
# ─────────────────────────────────────────────────────────────────────────────
RAW_DATA_PATH       = RAW_DATA_DIR  / "application_train.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "model_data.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Artifact Versioning
# ─────────────────────────────────────────────────────────────────────────────
MODEL_VERSION = "v2"

MODEL_PATH          = ARTIFACT_DIR / f"xgb_credit_risk_model_{MODEL_VERSION}.pkl"
ENCODER_PATH        = ARTIFACT_DIR / f"label_encoders_{MODEL_VERSION}.pkl"
IMPUTER_PATH        = ARTIFACT_DIR / f"median_imputer_{MODEL_VERSION}.pkl"
FEATURE_PATH        = ARTIFACT_DIR / f"credit_risk_feature_names_{MODEL_VERSION}.pkl"
THRESHOLD_PATH      = ARTIFACT_DIR / f"optimal_threshold_{MODEL_VERSION}.pkl"
PREDICTION_PATH     = ARTIFACT_DIR / f"test_predictions_{MODEL_VERSION}.csv"
METRIC_PATH         = ARTIFACT_DIR / f"model_metrics_{MODEL_VERSION}.csv"
SHAP_IMPORTANCE_PATH= ARTIFACT_DIR / f"shap_feature_importance_{MODEL_VERSION}.csv"
MODEL_COMPARISON_PATH = ARTIFACT_DIR / "model_comparison.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Model Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "n_estimators"      : 1000,          # high ceiling; early stopping will cut it
    "max_depth"         : 6,
    "learning_rate"     : 0.05,
    "subsample"         : 0.8,
    "colsample_bytree"  : 0.8,
    "min_child_weight"  : 5,
    "reg_alpha"         : 0.1,
    "reg_lambda"        : 1.0,
    "random_state"      : 42,
    "eval_metric"       : "auc",
    "tree_method"       : "hist",        # faster on large datasets
    "n_jobs"            : -1,
}

EARLY_STOPPING_ROUNDS   = 50
CV_FOLDS                = 5
TEST_SIZE               = 0.20
RANDOM_STATE            = 42

# ─────────────────────────────────────────────────────────────────────────────
# Risk Band Thresholds (PD Score → Label)
# ─────────────────────────────────────────────────────────────────────────────
RISK_BAND_THRESHOLDS = {
    "Low Risk"       : (0.00, 0.10),
    "Medium Risk"    : (0.10, 0.30),
    "High Risk"      : (0.30, 0.50),
    "Very High Risk" : (0.50, 1.00),
}

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
def _build_logger(name: str = "credit_risk") -> logging.Logger:
    """Create and return a production-grade logger writing to console + file."""
    logger = logging.getLogger(name)
    if logger.handlers:                  # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    # Console handler (INFO+)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (DEBUG+)
    fh = logging.FileHandler(LOG_DIR / "credit_risk.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


LOGGER = _build_logger()