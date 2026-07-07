"""
preprocess.py — Data Preprocessing Pipeline
=============================================
Handles all data-cleaning steps that must happen BEFORE feature
engineering.  Key design principles:

1. **No target contamination** — TARGET is excluded from all
   imputation and outlier-capping operations.
2. **Fitted artifacts persist** — MedianImputer statistics and
   categorical encoders are fitted on TRAINING data only and saved
   as artifacts.  At inference time the same fitted objects are loaded,
   eliminating training-serving skew.
3. **Label encoding for categoricals** — XGBoost handles label-encoded
   integers natively and more efficiently than one-hot encoded data.
4. **Anomaly replacement before imputation** — The DAYS_EMPLOYED=365243
   anomaly is replaced with NaN first so it participates in imputation.

Public API
----------
    fit_and_save_preprocessors(df_train) -> (imputer_map, encoder_map)
    transform(df, imputer_map, encoder_map) -> pd.DataFrame
    preprocess_data(df, fit=False) -> pd.DataFrame   ← convenience wrapper
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import ENCODER_PATH, IMPUTER_PATH, LOGGER
from src.utils import save_artifact, load_artifact

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Columns that must NEVER be touched by imputation / capping
_PROTECTED_COLS = {"TARGET", "SK_ID_CURR"}

# XGBoost does not accept string dtypes
_CATEGORICAL_FILL_VALUE = "Missing"   # fills NaN in categoricals before encoding


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Anomaly Replacement
# ─────────────────────────────────────────────────────────────────────────────

def replace_days_employed_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the DAYS_EMPLOYED = 365,243 anomaly with NaN.

    This value appears for applicants with no employment history and is
    a data entry sentinel, not a real observation.  Replacing it with NaN
    lets the median imputer handle it cleanly.
    """
    if "DAYS_EMPLOYED" not in df.columns:
        LOGGER.debug("DAYS_EMPLOYED not found; skipping anomaly replacement.")
        return df

    n_anomaly = (df["DAYS_EMPLOYED"] == 365_243).sum()
    if n_anomaly:
        LOGGER.info(f"Replacing {n_anomaly:,} DAYS_EMPLOYED=365243 anomalies with NaN.")
    df = df.copy()
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365_243, np.nan)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Missing Value Imputation (FITTED on training data)
# ─────────────────────────────────────────────────────────────────────────────

def fit_median_imputer(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute per-column medians from TRAINING data.

    Returns a dict {column_name: median_value} for numeric columns only.
    This dict is persisted so inference time uses the same values.
    """
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in _PROTECTED_COLS
    ]
    imputer_map: dict[str, float] = {}
    for col in numeric_cols:
        imputer_map[col] = float(df[col].median())

    LOGGER.info(f"Fitted median imputer on {len(imputer_map)} numeric columns.")
    return imputer_map


def apply_median_imputer(df: pd.DataFrame, imputer_map: dict[str, float]) -> pd.DataFrame:
    """Fill numeric NaN values using the pre-fitted imputer map."""
    df = df.copy()
    for col, median_val in imputer_map.items():
        if col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing:
                df[col] = df[col].fillna(median_val)
    return df


def apply_categorical_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN in categorical (object) columns with the sentinel 'Missing'."""
    df = df.copy()
    cat_cols = [
        c for c in df.select_dtypes(include="object").columns
        if c not in _PROTECTED_COLS
    ]
    for col in cat_cols:
        n_missing = df[col].isna().sum()
        if n_missing:
            df[col] = df[col].fillna(_CATEGORICAL_FILL_VALUE)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Outlier Capping / Winsorisation (FITTED on training data)
# ─────────────────────────────────────────────────────────────────────────────

def fit_outlier_caps(
    df: pd.DataFrame,
    lower: float = 0.01,
    upper: float = 0.99,
) -> dict[str, tuple[float, float]]:
    """
    Compute per-column (lower_bound, upper_bound) from TRAINING data.

    TARGET and SK_ID_CURR are excluded.
    Returns a dict {column_name: (lower_bound, upper_bound)}.
    """
    numeric_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in _PROTECTED_COLS
    ]
    caps: dict[str, tuple[float, float]] = {}
    for col in numeric_cols:
        caps[col] = (float(df[col].quantile(lower)), float(df[col].quantile(upper)))

    LOGGER.info(f"Fitted outlier caps on {len(caps)} numeric columns (p{int(lower*100)}/p{int(upper*100)}).")
    return caps


def apply_outlier_caps(df: pd.DataFrame, caps: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Winsorise numeric columns using the pre-fitted cap bounds."""
    df = df.copy()
    for col, (lo, hi) in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Label Encoding for Categoricals (FITTED on training data)
# ─────────────────────────────────────────────────────────────────────────────

def fit_label_encoders(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """
    Fit one label encoder per categorical column.

    Each encoder is stored as {category_value: integer_code}.
    The special value 'Missing' (from apply_categorical_fill) always maps
    to 0 so unseen values at inference time can be safely mapped to 0 too.

    Returns a dict {column_name: {value: code, ...}}.
    """
    cat_cols = [
        c for c in df.select_dtypes(include="object").columns
        if c not in _PROTECTED_COLS
    ]
    encoders: dict[str, dict[str, int]] = {}
    for col in cat_cols:
        unique_vals = df[col].dropna().unique().tolist()
        # Ensure 'Missing' maps to 0
        mapping: dict[str, int] = {_CATEGORICAL_FILL_VALUE: 0}
        code = 1
        for v in sorted(unique_vals):
            if v != _CATEGORICAL_FILL_VALUE:
                mapping[v] = code
                code += 1
        encoders[col] = mapping

    LOGGER.info(f"Fitted label encoders on {len(encoders)} categorical columns.")
    return encoders


def apply_label_encoders(df: pd.DataFrame, encoders: dict[str, dict[str, int]]) -> pd.DataFrame:
    """
    Apply pre-fitted label encoders.

    Unseen categories (present at inference but not in training) are
    mapped to 0 (same as 'Missing') rather than raising an error.
    """
    df = df.copy()
    for col, mapping in encoders.items():
        if col not in df.columns:
            continue
        # Cast to str to handle any edge-case dtype mismatches
        df[col] = df[col].astype(str).map(mapping).fillna(0).astype(int)

    LOGGER.debug(f"Applied label encoders to {len(encoders)} columns.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Composite Public API
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_save_preprocessors(df_train: pd.DataFrame) -> tuple[dict, dict, dict]:
    """
    Fit all preprocessing artifacts on training data and save them.

    Returns
    -------
    (imputer_map, cap_map, encoder_map)

    Each is also persisted to disk via src.config artifact paths.
    """
    LOGGER.info("Fitting preprocessing artifacts on training data …")

    # Anomaly replacement is stateless (no fitting needed)
    df_clean = replace_days_employed_anomaly(df_train)
    df_clean = apply_categorical_fill(df_clean)

    imputer_map = fit_median_imputer(df_clean)
    df_clean    = apply_median_imputer(df_clean, imputer_map)

    cap_map  = fit_outlier_caps(df_clean)
    df_clean = apply_outlier_caps(df_clean, cap_map)

    encoder_map = fit_label_encoders(df_clean)

    # Bundle everything into one artifact for simplicity
    preprocessors = {
        "imputer_map" : imputer_map,
        "cap_map"     : cap_map,
        "encoder_map" : encoder_map,
    }
    save_artifact(preprocessors, IMPUTER_PATH, label="preprocessors (imputer+caps+encoders)")

    # Also save encoders separately so predict.py can load just encoders
    save_artifact(encoder_map, ENCODER_PATH, label="label encoders")

    LOGGER.info("All preprocessing artifacts saved.")
    return imputer_map, cap_map, encoder_map


def load_preprocessors() -> tuple[dict, dict, dict]:
    """Load the fitted preprocessors from disk."""
    preprocessors = load_artifact(IMPUTER_PATH, label="preprocessors")
    return (
        preprocessors["imputer_map"],
        preprocessors["cap_map"],
        preprocessors["encoder_map"],
    )


def transform(
    df            : pd.DataFrame,
    imputer_map   : dict,
    cap_map       : dict,
    encoder_map   : dict,
) -> pd.DataFrame:
    """
    Apply the full preprocessing transform using pre-fitted artifacts.

    Safe to call on both training AND inference data.

    Parameters
    ----------
    df          : Raw DataFrame (may contain TARGET and SK_ID_CURR).
    imputer_map : Medians fitted on training data.
    cap_map     : Outlier bounds fitted on training data.
    encoder_map : Label encoder mappings fitted on training data.

    Returns
    -------
    Cleaned, encoded DataFrame (TARGET and SK_ID_CURR preserved if present).
    """
    LOGGER.info(f"Transforming dataframe of shape {df.shape} …")

    df = replace_days_employed_anomaly(df)
    df = apply_categorical_fill(df)
    df = apply_median_imputer(df, imputer_map)
    df = apply_outlier_caps(df, cap_map)
    df = apply_label_encoders(df, encoder_map)

    LOGGER.info(f"Preprocessing complete → shape {df.shape}")
    return df


def preprocess_data(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    Convenience wrapper used by the pipeline script and Streamlit app.

    Parameters
    ----------
    df  : Input DataFrame.
    fit : If True, fits and saves preprocessors (training mode).
          If False, loads existing preprocessors and transforms (inference mode).
    """
    if fit:
        imputer_map, cap_map, encoder_map = fit_and_save_preprocessors(df)
    else:
        imputer_map, cap_map, encoder_map = load_preprocessors()

    return transform(df, imputer_map, cap_map, encoder_map)