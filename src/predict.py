"""
predict.py — Inference / Prediction Pipeline
=============================================
Provides functions for single-customer and batch scoring.

The inference pipeline mirrors the training pipeline EXACTLY:
    Raw customer data
    → replace_days_employed_anomaly
    → apply_categorical_fill
    → apply_median_imputer  (fitted statistics from training)
    → apply_outlier_caps    (fitted bounds from training)
    → apply_label_encoders  (fitted mappings from training)
    → engineer_features
    → reindex to training feature order
    → XGBClassifier.predict_proba
    → risk band + recommendation

No training-serving skew is possible because the same fitted
preprocessor artifacts are used at inference time.

Usage
-----
    from src.predict import score_dataframe, predict_single

    # Batch scoring:
    scored = score_dataframe(raw_df)

    # Single customer dict:
    result = predict_single({"AMT_INCOME_TOTAL": 200000, ...})
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import (
    ENCODER_PATH,
    FEATURE_PATH,
    IMPUTER_PATH,
    LOGGER,
    MODEL_PATH,
    THRESHOLD_PATH,
)
from src.feature_engineering import engineer_features
from src.preprocess import (
    apply_categorical_fill,
    apply_label_encoders,
    apply_median_imputer,
    apply_outlier_caps,
    replace_days_employed_anomaly,
)
from src.utils import assign_recommendation, assign_risk_band, load_artifact


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Loading (cached at module level for performance)
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """Load the trained XGBoost model."""
    return load_artifact(MODEL_PATH, label="XGBoost model")


def load_features() -> list[str]:
    """Load the ordered list of training feature names."""
    return load_artifact(FEATURE_PATH, label="feature names")


def load_preprocessors() -> tuple[dict, dict, dict]:
    """Load the fitted preprocessor artifacts."""
    preprocessors = load_artifact(IMPUTER_PATH, label="preprocessors")
    return (
        preprocessors["imputer_map"],
        preprocessors["cap_map"],
        preprocessors["encoder_map"],
    )


def load_threshold() -> float:
    """Load the optimal decision threshold."""
    return float(load_artifact(THRESHOLD_PATH, label="optimal threshold"))


# ─────────────────────────────────────────────────────────────────────────────
# Core Inference Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_inference(
    df            : pd.DataFrame,
    imputer_map   : dict,
    cap_map       : dict,
    encoder_map   : dict,
) -> pd.DataFrame:
    """
    Apply the EXACT same preprocessing steps used during training.

    This function is the anti-skew contract: any change to training
    preprocessing must be mirrored here.

    Parameters
    ----------
    df            : Raw customer DataFrame (one or more rows).
    imputer_map   : Medians fitted on training data.
    cap_map       : Outlier bounds fitted on training data.
    encoder_map   : Label encoder mappings fitted on training data.

    Returns
    -------
    Preprocessed + feature-engineered DataFrame aligned to training columns.
    """
    df = df.copy()

    # Step 1: Anomaly replacement (stateless)
    df = replace_days_employed_anomaly(df)

    # Step 2: Fill categoricals
    df = apply_categorical_fill(df)

    # Step 3: Impute numerics with TRAINING medians
    df = apply_median_imputer(df, imputer_map)

    # Step 4: Winsorise with TRAINING bounds
    df = apply_outlier_caps(df, cap_map)

    # Step 5: Encode categoricals with TRAINING mappings
    df = apply_label_encoders(df, encoder_map)

    # Step 6: Feature engineering (stateless — same at train & inference)
    df = engineer_features(df, drop_ids=True)

    return df


def score_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Score a batch DataFrame and return it with prediction columns appended.

    Parameters
    ----------
    raw_df : DataFrame with the same columns as application_train.csv.

    Returns
    -------
    A copy of raw_df with the following columns appended:
        Prediction           : Binary class (0/1)
        Probability_Default  : Raw probability of default (0–1)
        PD                   : Probability as percentage (0–100)
        Risk_Band            : Low / Medium / High / Very High Risk
        Recommendation       : Approve / Manual Review / High Risk Review / Reject
    """
    LOGGER.info(f"Scoring batch of {len(raw_df):,} customers …")

    # Load artifacts
    model         = load_model()
    feature_names = load_features()
    imputer_map, cap_map, encoder_map = load_preprocessors()
    threshold     = load_threshold()

    # Preprocess & engineer features
    df_processed = preprocess_for_inference(raw_df, imputer_map, cap_map, encoder_map)

    # Align to training feature order (fill any missing columns with 0)
    missing_cols = set(feature_names) - set(df_processed.columns)
    if missing_cols:
        LOGGER.warning(f"Filling {len(missing_cols)} missing columns with 0: {list(missing_cols)[:5]} …")
    df_model = df_processed.reindex(columns=feature_names, fill_value=0)

    # Predict
    y_prob = model.predict_proba(df_model)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Append output columns to a copy of the ORIGINAL (raw) dataframe
    result = raw_df.copy()
    result["Prediction"]          = y_pred
    result["Probability_Default"] = np.round(y_prob, 6)
    result["PD"]                  = np.round(y_prob * 100, 2)
    result["Risk_Band"]           = [assign_risk_band(p) for p in y_prob]
    result["Recommendation"]      = [assign_recommendation(p) for p in y_prob]

    LOGGER.info(
        f"Scoring complete. "
        f"Risk distribution: "
        + "  ".join(
            f"{band}={count}"
            for band, count in result["Risk_Band"].value_counts().items()
        )
    )
    return result


def predict_single(customer_dict: dict) -> dict:
    """
    Score a single customer provided as a Python dictionary.

    Parameters
    ----------
    customer_dict : Column-name → value mapping for one customer.

    Returns
    -------
    {
        "prediction"           : 0 or 1,
        "probability_default"  : float,
        "pd_percentage"        : float,
        "risk_band"            : str,
        "recommendation"       : str,
    }
    """
    df = pd.DataFrame([customer_dict])
    scored = score_dataframe(df)
    row = scored.iloc[0]
    return {
        "prediction"          : int(row["Prediction"]),
        "probability_default" : float(row["Probability_Default"]),
        "pd_percentage"       : float(row["PD"]),
        "risk_band"           : str(row["Risk_Band"]),
        "recommendation"      : str(row["Recommendation"]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point — Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo: score one customer using approximate typical values
    sample = {
        "AMT_INCOME_TOTAL"   : 200_000,
        "AMT_CREDIT"         : 500_000,
        "AMT_ANNUITY"        : 25_000,
        "AMT_GOODS_PRICE"    : 450_000,
        "CNT_FAM_MEMBERS"    : 2,
        "DAYS_EMPLOYED"      : -2_500,
        "DAYS_BIRTH"         : -15_000,
        "EXT_SOURCE_1"       : 0.55,
        "EXT_SOURCE_2"       : 0.60,
        "EXT_SOURCE_3"       : 0.58,
    }
    try:
        result = predict_single(sample)
        LOGGER.info(f"Sample prediction: {result}")
        print("\nSample customer prediction:")
        for k, v in result.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        LOGGER.error(f"Prediction demo failed: {exc}", exc_info=True)
        sys.exit(1)