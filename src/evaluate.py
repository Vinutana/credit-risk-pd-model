"""
evaluate.py — Model Evaluation Script
=======================================
Loads the saved model and test predictions and computes a comprehensive
suite of credit-risk metrics including:

- ROC-AUC & Gini coefficient
- PR-AUC
- KS Statistic (industry-standard for scorecard validation)
- Precision, Recall, F1 (at optimal threshold)
- Classification report
- Confusion matrix

All metrics are logged and saved to artifacts/model_metrics_vX.csv.

Usage
-----
    python -m src.evaluate
    python src/evaluate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    LOGGER,
    METRIC_PATH,
    MODEL_PATH,
    PREDICTION_PATH,
    THRESHOLD_PATH,
)
from src.utils import (
    format_metrics_table,
    gini_from_auc,
    load_artifact,
    log_metrics,
    save_dataframe,
    timed,
)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_test_predictions() -> pd.DataFrame:
    """
    Load the test prediction file saved by train.py.

    Columns: actual, predicted_probability, predicted_class
    """
    if not PREDICTION_PATH.exists():
        raise FileNotFoundError(
            f"Test predictions not found: {PREDICTION_PATH}\n"
            "Run `python src/train.py` first."
        )
    df = pd.read_csv(PREDICTION_PATH)
    LOGGER.info(f"Loaded test predictions → {len(df):,} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Metric Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Kolmogorov-Smirnov statistic: maximum separation between the cumulative
    score distributions of defaulters and non-defaulters.

    Industry standard for scorecard validation (typical range: 0.3 – 0.6).
    """
    ks_stat = ks_2samp(y_prob[y_true == 0], y_prob[y_true == 1]).statistic
    return float(ks_stat)


@timed
def evaluate_model() -> pd.DataFrame:
    """
    Compute and report all evaluation metrics.

    Returns
    -------
    DataFrame with columns [Metric, Value].
    """
    LOGGER.info("=" * 60)
    LOGGER.info("CREDIT RISK EVALUATION — START")
    LOGGER.info("=" * 60)

    # Load artifacts
    pred_df   = load_test_predictions()
    threshold = load_artifact(THRESHOLD_PATH, label="optimal threshold")

    y_true = pred_df["actual"].values
    y_prob = pred_df["predicted_probability"].values
    y_pred = (y_prob >= threshold).astype(int)

    LOGGER.info(f"Evaluation threshold: {threshold:.4f}")
    LOGGER.info(f"Test set:  {len(y_true):,} rows | {y_true.mean()*100:.1f}% default rate")

    # ── Core metrics ─────────────────────────────────────────────────────────
    roc_auc   = roc_auc_score(y_true, y_prob)
    gini      = gini_from_auc(roc_auc)
    pr_auc    = average_precision_score(y_true, y_prob)
    ks        = compute_ks_statistic(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    metrics = {
        "ROC_AUC"  : round(roc_auc,   4),
        "Gini"     : round(gini,       4),
        "PR_AUC"   : round(pr_auc,     4),
        "KS"       : round(ks,         4),
        "Precision": round(precision,  4),
        "Recall"   : round(recall,     4),
        "F1"       : round(f1,         4),
        "Threshold": round(threshold,  4),
    }

    # ── Log & display ─────────────────────────────────────────────────────────
    log_metrics(metrics, prefix="EVAL")

    LOGGER.info("\n" + "=" * 40)
    LOGGER.info("Classification Report")
    LOGGER.info("=" * 40)
    LOGGER.info("\n" + classification_report(y_true, y_pred, target_names=["Non-Default", "Default"]))

    LOGGER.info("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    LOGGER.info(f"\n{cm}")
    tn, fp, fn, tp = cm.ravel()
    LOGGER.info(
        f"  True Negatives  (correctly rejected non-defaults): {tn:,}\n"
        f"  False Positives (non-defaults incorrectly flagged): {fp:,}\n"
        f"  False Negatives (missed defaults — credit losses!): {fn:,}\n"
        f"  True Positives  (correctly identified defaults):    {tp:,}\n"
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    metrics_df = format_metrics_table(metrics)
    save_dataframe(metrics_df, METRIC_PATH, label="model metrics")

    LOGGER.info("=" * 60)
    LOGGER.info(
        f"EVALUATION COMPLETE  "
        f"| ROC-AUC={roc_auc:.4f}  Gini={gini:.4f}  KS={ks:.4f}"
    )
    LOGGER.info("=" * 60)

    return metrics_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        evaluate_model()
    except Exception as exc:
        LOGGER.error(f"Evaluation failed: {exc}", exc_info=True)
        sys.exit(1)