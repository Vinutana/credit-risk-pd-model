"""
train.py — Model Training Script
==================================
Trains an XGBoost PD (Probability of Default) model with:

- Stratified k-fold cross-validation for robust performance estimates
- Early stopping to prevent overfitting
- Scale-pos-weight for class imbalance handling
- Optimal decision threshold selection via PR curve
- All artifacts saved with version tags

Usage
-----
    python -m src.train
    python src/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
from xgboost import XGBClassifier

from src.config import (
    LOGGER,
    PROCESSED_DATA_PATH,
    MODEL_PATH,
    FEATURE_PATH,
    THRESHOLD_PATH,
    XGBOOST_PARAMS,
    EARLY_STOPPING_ROUNDS,
    CV_FOLDS,
    TEST_SIZE,
    RANDOM_STATE,
)
from src.utils import (
    save_artifact,
    save_dataframe,
    validate_no_target_in_features,
    check_dataframe_dtypes,
    log_metrics,
    timed,
)
from src.config import PREDICTION_PATH


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Preparation
# ─────────────────────────────────────────────────────────────────────────────

def load_processed_data() -> pd.DataFrame:
    """Load the processed dataset produced by pipeline.py."""
    if not PROCESSED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found: {PROCESSED_DATA_PATH}\n"
            "Run `python src/pipeline.py` first."
        )
    df = pd.read_csv(PROCESSED_DATA_PATH)
    LOGGER.info(f"Loaded processed data → shape {df.shape}")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split processed data into feature matrix X and target vector y.

    Drops:
    - TARGET (the label)
    - SK_ID_CURR (customer ID — not a predictor)
    """
    if "TARGET" not in df.columns:
        raise ValueError("Column 'TARGET' not found in processed dataset.")

    drop_cols = [c for c in ["TARGET", "SK_ID_CURR"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["TARGET"]

    validate_no_target_in_features(X)
    check_dataframe_dtypes(X, context="after feature engineering")

    LOGGER.info(f"Feature matrix X: {X.shape}  |  Target y distribution:")
    LOGGER.info(f"  Class 0 (non-default): {(y == 0).sum():,}  ({(y == 0).mean()*100:.1f}%)")
    LOGGER.info(f"  Class 1 (default):     {(y == 1).sum():,}  ({(y == 1).mean()*100:.1f}%)")

    return X, y


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Compute scale_pos_weight for XGBoost class imbalance handling.

    Formula: count(negatives) / count(positives)
    Applied only to y_train (not the whole dataset) to avoid leakage.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw = n_neg / n_pos
    LOGGER.info(f"scale_pos_weight = {spw:.2f}  (neg={n_neg:,}, pos={n_pos:,})")
    return spw


# ─────────────────────────────────────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

@timed
def cross_validate_model(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Run stratified k-fold CV to get unbiased performance estimates.

    Returns a dict with mean and std of ROC-AUC and PR-AUC across folds.
    """
    LOGGER.info(f"Starting {CV_FOLDS}-fold stratified cross-validation …")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    aucs, pr_aucs = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        spw = calculate_scale_pos_weight(y_tr)
        params = {**XGBOOST_PARAMS, "scale_pos_weight": spw}

        clf = XGBClassifier(**params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=False,
        )

        y_prob = clf.predict_proba(X_val)[:, 1]
        fold_auc   = roc_auc_score(y_val, y_prob)
        fold_prauc = average_precision_score(y_val, y_prob)
        aucs.append(fold_auc)
        pr_aucs.append(fold_prauc)

        LOGGER.info(
            f"  Fold {fold_idx}/{CV_FOLDS}: "
            f"ROC-AUC={fold_auc:.4f}  PR-AUC={fold_prauc:.4f}  "
            f"best_iter={clf.best_iteration}"
        )

    cv_results = {
        "cv_roc_auc_mean"  : float(np.mean(aucs)),
        "cv_roc_auc_std"   : float(np.std(aucs)),
        "cv_pr_auc_mean"   : float(np.mean(pr_aucs)),
        "cv_pr_auc_std"    : float(np.std(pr_aucs)),
    }
    log_metrics(cv_results, prefix="CV")
    return cv_results


# ─────────────────────────────────────────────────────────────────────────────
# Final Model Training
# ─────────────────────────────────────────────────────────────────────────────

@timed
def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val  : pd.DataFrame,
    y_val  : pd.Series,
) -> XGBClassifier:
    """
    Train the final XGBoost model on the full training set with early stopping
    against the hold-out validation set.

    Parameters
    ----------
    X_train, y_train : Full training partition.
    X_val, y_val     : Hold-out validation set used only for early stopping.

    Returns
    -------
    Fitted XGBClassifier with the optimal number of trees.
    """
    spw    = calculate_scale_pos_weight(y_train)
    params = {**XGBOOST_PARAMS, "scale_pos_weight": spw}

    LOGGER.info("Training final XGBoost model …")
    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=100,
    )
    LOGGER.info(f"Final model trained.  Best iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Threshold Selection
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true: pd.Series, y_prob: np.ndarray) -> float:
    """
    Select the decision threshold that maximises the F1 score on the
    validation set.

    Using the default 0.5 threshold is inappropriate for imbalanced
    datasets.  The PR curve gives the Pareto frontier of precision vs.
    recall; we pick the point that maximises F1.

    Returns
    -------
    Optimal threshold (float between 0 and 1).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # F1 = 2 * P * R / (P + R)  — avoid division by zero
    with np.errstate(invalid="ignore"):
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0,
        )
    best_idx       = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    best_f1        = float(f1_scores[best_idx])

    LOGGER.info(
        f"Optimal threshold: {best_threshold:.4f}  "
        f"(F1={best_f1:.4f}  P={precision[best_idx]:.4f}  R={recall[best_idx]:.4f})"
    )
    return best_threshold


# ─────────────────────────────────────────────────────────────────────────────
# Artifact Saving
# ─────────────────────────────────────────────────────────────────────────────

def save_training_artifacts(
    model          : XGBClassifier,
    feature_names  : list[str],
    threshold      : float,
    X_test         : pd.DataFrame,
    y_test         : pd.Series,
    y_prob         : np.ndarray,
) -> None:
    """Persist all training outputs to the artifacts directory."""
    save_artifact(model,         MODEL_PATH,    label="XGBoost model")
    save_artifact(feature_names, FEATURE_PATH,  label="feature names list")
    save_artifact(threshold,     THRESHOLD_PATH, label="optimal threshold")

    # Save test predictions for Notebook 7 and evaluation
    pred_df = pd.DataFrame({
        "actual"               : y_test.values,
        "predicted_probability": y_prob,
        "predicted_class"      : (y_prob >= threshold).astype(int),
    })
    save_dataframe(pred_df, PREDICTION_PATH, label="test predictions")
    LOGGER.info("All training artifacts saved successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

@timed
def main() -> None:
    LOGGER.info("=" * 60)
    LOGGER.info("CREDIT RISK TRAINING — START")
    LOGGER.info("=" * 60)

    # ── Load & prepare data ──────────────────────────────────────────────────
    df         = load_processed_data()
    X, y       = prepare_features(df)

    # ── Train / validation / test split (stratified) ─────────────────────────
    # We use a 60/20/20 split:
    #   Train  (60%) → model learning
    #   Val    (20%) → early stopping + threshold selection
    #   Test   (20%) → final hold-out evaluation (never seen during training)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.25,               # 0.25 × 0.80 = 0.20 of original
        random_state=RANDOM_STATE,
        stratify=y_trainval,
    )
    LOGGER.info(
        f"Split sizes:  train={len(X_train):,}  val={len(X_val):,}  test={len(X_test):,}"
    )

    # ── Cross-validation (uses X_trainval to keep test set unseen) ───────────
    cv_results = cross_validate_model(X_trainval, y_trainval)

    # ── Train final model ────────────────────────────────────────────────────
    model = train_final_model(X_train, y_train, X_val, y_val)

    # ── Threshold optimisation on validation set ─────────────────────────────
    y_prob_val = model.predict_proba(X_val)[:, 1]
    threshold  = find_optimal_threshold(y_val, y_prob_val)

    # ── Test set evaluation ──────────────────────────────────────────────────
    y_prob_test = model.predict_proba(X_test)[:, 1]
    test_roc_auc = roc_auc_score(y_test, y_prob_test)
    test_pr_auc  = average_precision_score(y_test, y_prob_test)
    log_metrics(
        {"test_roc_auc": test_roc_auc, "test_pr_auc": test_pr_auc},
        prefix="TEST",
    )

    # ── Save artifacts ────────────────────────────────────────────────────────
    feature_names = X.columns.tolist()
    save_training_artifacts(
        model, feature_names, threshold, X_test, y_test, y_prob_test
    )

    LOGGER.info("=" * 60)
    LOGGER.info(
        f"TRAINING COMPLETE  "
        f"| CV ROC-AUC={cv_results['cv_roc_auc_mean']:.4f} ± {cv_results['cv_roc_auc_std']:.4f}"
        f"  | Test ROC-AUC={test_roc_auc:.4f}"
    )
    LOGGER.info("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOGGER.error(f"Training failed: {exc}", exc_info=True)
        sys.exit(1)