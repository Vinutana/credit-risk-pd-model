"""
utils.py — Shared Helper Utilities
====================================
Reusable, well-tested utility functions for artifact I/O, metrics
formatting, and data validation used across the entire src package.

All functions include type hints and docstrings suitable for a
production codebase.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, List

import joblib
import numpy as np
import pandas as pd

from src.config import LOGGER

# ─────────────────────────────────────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────────────────────────────────────

def timed(func):
    """Decorator: logs the wall-clock time taken by a function call."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        LOGGER.info(f"{func.__qualname__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ─────────────────────────────────────────────────────────────────────────────
# Artifact I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_artifact(obj: Any, filepath: Path | str, label: str = "object") -> None:
    """
    Persist *obj* to *filepath* using joblib.

    Parameters
    ----------
    obj      : Any Python object (model, list, dict, …)
    filepath : Destination path.  Parent directories are created automatically.
    label    : Human-readable name used in the log message.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)
    LOGGER.info(f"Saved {label} → {filepath}")


def load_artifact(filepath: Path | str, label: str = "object") -> Any:
    """
    Load a joblib-pickled artifact.

    Raises
    ------
    FileNotFoundError if the artifact does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Artifact not found: {filepath}\n"
            "Run the training pipeline first to generate this artifact."
        )
    obj = joblib.load(filepath)
    LOGGER.info(f"Loaded {label} ← {filepath}")
    return obj


def save_dataframe(df: pd.DataFrame, filepath: Path | str, label: str = "dataframe") -> None:
    """Save a DataFrame to CSV (no index)."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    LOGGER.info(f"Saved {label} ({df.shape[0]:,} rows × {df.shape[1]} cols) → {filepath}")


def load_dataframe(filepath: Path | str, label: str = "dataframe") -> pd.DataFrame:
    """Load a CSV into a DataFrame."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    df = pd.read_csv(filepath)
    LOGGER.info(f"Loaded {label} ({df.shape[0]:,} rows × {df.shape[1]} cols) ← {filepath}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Data Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_columns(df: pd.DataFrame, required_cols: List[str], context: str = "") -> None:
    """
    Assert that *df* contains every column in *required_cols*.

    Raises
    ------
    ValueError with a helpful message listing the missing columns.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        ctx = f" [{context}]" if context else ""
        raise ValueError(
            f"Missing required columns{ctx}: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )


def validate_no_target_in_features(X: pd.DataFrame) -> None:
    """Raise if TARGET column accidentally ended up in the feature matrix."""
    if "TARGET" in X.columns:
        raise ValueError(
            "Column 'TARGET' found in feature matrix X. "
            "Drop it before training."
        )


def check_dataframe_dtypes(df: pd.DataFrame, context: str = "") -> None:
    """Log a warning for any object-dtype columns remaining in *df*."""
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        ctx = f" [{context}]" if context else ""
        LOGGER.warning(
            f"Object-dtype columns found{ctx}; model may reject these: {obj_cols}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics Formatting
# ─────────────────────────────────────────────────────────────────────────────

def gini_from_auc(auc: float) -> float:
    """Gini coefficient = 2 × AUC − 1 (standard credit risk convention)."""
    return round(2 * auc - 1, 4)


def format_metrics_table(metrics: dict) -> pd.DataFrame:
    """
    Convert a {metric_name: value} dict into a tidy two-column DataFrame
    for display or CSV export.
    """
    return pd.DataFrame(
        {"Metric": list(metrics.keys()), "Value": list(metrics.values())}
    )


def log_metrics(metrics: dict, prefix: str = "") -> None:
    """Log every metric in *metrics* at INFO level."""
    tag = f"[{prefix}] " if prefix else ""
    for name, val in metrics.items():
        LOGGER.info(f"{tag}{name}: {val:.4f}" if isinstance(val, float) else f"{tag}{name}: {val}")


# ─────────────────────────────────────────────────────────────────────────────
# Risk Banding
# ─────────────────────────────────────────────────────────────────────────────

def assign_risk_band(pd_score: float) -> str:
    """
    Map a PD probability to a human-readable risk band.

    Bands
    -----
    [0.00, 0.10) → Low Risk
    [0.10, 0.30) → Medium Risk
    [0.30, 0.50) → High Risk
    [0.50, 1.00] → Very High Risk
    """
    if pd_score < 0.10:
        return "Low Risk"
    elif pd_score < 0.30:
        return "Medium Risk"
    elif pd_score < 0.50:
        return "High Risk"
    return "Very High Risk"


def assign_recommendation(pd_score: float) -> str:
    """Map a PD probability to a lending recommendation."""
    if pd_score < 0.10:
        return "Approve"
    elif pd_score < 0.30:
        return "Manual Review"
    elif pd_score < 0.50:
        return "High Risk Review"
    return "Reject"