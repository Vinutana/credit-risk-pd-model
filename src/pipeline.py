"""
pipeline.py — End-to-End Data Preparation Pipeline
====================================================
Orchestrates the full sequence:
    Raw CSV → Preprocessing (fit) → Feature Engineering → Processed CSV

This script is the SINGLE entry point for data preparation.  Running it
guarantees that:
1. Preprocessing artifacts (imputer, caps, encoders) are fitted on the
   training data and persisted.
2. The processed CSV written to disk is consistent with what the model
   training script expects.

Usage
-----
    python -m src.pipeline

    # or from the project root:
    python src/pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly OR as part of the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LOGGER, RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.feature_engineering import engineer_features
from src.preprocess import preprocess_data
from src.utils import load_dataframe, save_dataframe, timed


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

@timed
def load_raw_data() -> "pd.DataFrame":  # noqa: F821
    """Load the raw training CSV from the configured path."""
    import pandas as pd
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data not found at: {RAW_DATA_PATH}\n"
            "Download the Home Credit Default Risk dataset from Kaggle and place\n"
            "application_train.csv in data/raw/"
        )
    df = pd.read_csv(RAW_DATA_PATH)
    LOGGER.info(f"Raw dataset loaded → shape {df.shape}")
    return df


@timed
def run_pipeline() -> None:
    """
    Execute the full data preparation pipeline.

    Steps
    -----
    1. Load raw data
    2. Preprocess (fit artifacts + transform)
    3. Engineer features
    4. Save processed dataset
    """
    LOGGER.info("=" * 60)
    LOGGER.info("CREDIT RISK PIPELINE — START")
    LOGGER.info("=" * 60)

    # ── Step 1: Load ─────────────────────────────────────────────────────────
    df = load_raw_data()
    n_raw_rows, n_raw_cols = df.shape

    # ── Step 2: Preprocess (FIT mode — fits and saves all artifacts) ─────────
    LOGGER.info("Step 2/4 — Preprocessing (FIT mode) …")
    df = preprocess_data(df, fit=True)
    LOGGER.info(f"After preprocessing: shape {df.shape}")

    # ── Step 3: Feature Engineering ──────────────────────────────────────────
    LOGGER.info("Step 3/4 — Feature engineering …")
    # Keep SK_ID_CURR in the processed file (useful for downstream joins),
    # but mark it so train.py knows to drop it before model training.
    df = engineer_features(df, drop_ids=False)
    LOGGER.info(f"After feature engineering: shape {df.shape}")

    # ── Step 4: Save ─────────────────────────────────────────────────────────
    LOGGER.info("Step 4/4 — Saving processed dataset …")
    save_dataframe(df, PROCESSED_DATA_PATH, label="processed dataset")

    LOGGER.info("=" * 60)
    LOGGER.info(
        f"PIPELINE COMPLETE  "
        f"| {n_raw_rows:,} rows × {n_raw_cols} raw cols "
        f"→ {df.shape[1]} processed cols"
    )
    LOGGER.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:
        LOGGER.error(f"Pipeline failed: {exc}", exc_info=True)
        sys.exit(1)