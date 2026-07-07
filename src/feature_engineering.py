"""
feature_engineering.py — Feature Engineering for Credit Risk Modeling
=======================================================================
Creates domain-specific features from the Home Credit raw/preprocessed
data.  All functions are stateless (no fitting required) and can be
applied identically at training and inference time.

Feature Groups
--------------
1. Financial burden ratios        (credit/income, annuity/income, …)
2. EXT_SOURCE aggregate features  (most predictive group)
3. Age & employment features      (stability signals)
4. Document & request counts      (behavioural signals)
5. Household financial capacity   (income per family member)

ID Handling
-----------
SK_ID_CURR is a row identifier, not a predictor.  It is dropped at the
END of this module so it is never included in the model feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LOGGER


# ─────────────────────────────────────────────────────────────────────────────
# 1. Financial Burden Ratios
# ─────────────────────────────────────────────────────────────────────────────

def create_credit_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    AMT_CREDIT / AMT_INCOME_TOTAL

    Measures how large the loan is relative to annual income.
    Higher ratio → greater financial burden → higher default risk.
    Guard: use np.where to avoid division by zero cleanly.
    """
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = np.where(
        df["AMT_INCOME_TOTAL"] > 0,
        df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"],
        0.0,
    )
    return df


def create_annuity_income_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    AMT_ANNUITY / AMT_INCOME_TOTAL

    Monthly repayment as a fraction of income — a debt-service coverage proxy.
    """
    df = df.copy()
    df["ANNUITY_INCOME_RATIO"] = np.where(
        df["AMT_INCOME_TOTAL"] > 0,
        df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"],
        0.0,
    )
    return df


def create_goods_credit_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    AMT_GOODS_PRICE / AMT_CREDIT

    Loan-to-value proxy: how much of the credit is for the actual goods.
    Values < 1 suggest the credit exceeds the goods price (risk signal).
    """
    df = df.copy()
    df["GOODS_CREDIT_RATIO"] = np.where(
        df["AMT_CREDIT"] > 0,
        df["AMT_GOODS_PRICE"] / df["AMT_CREDIT"],
        0.0,
    )
    return df


def create_annuity_credit_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    AMT_ANNUITY / AMT_CREDIT

    Implicit interest rate proxy: higher → shorter/more expensive loan.
    """
    df = df.copy()
    df["ANNUITY_CREDIT_RATIO"] = np.where(
        df["AMT_CREDIT"] > 0,
        df["AMT_ANNUITY"] / df["AMT_CREDIT"],
        0.0,
    )
    return df


def create_credit_per_person(df: pd.DataFrame) -> pd.DataFrame:
    """AMT_CREDIT per household member (family financial exposure)."""
    df = df.copy()
    family = df["CNT_FAM_MEMBERS"].clip(lower=1)   # at least 1 person
    df["CREDIT_PER_PERSON"] = df["AMT_CREDIT"] / family
    return df


def create_income_per_person(df: pd.DataFrame) -> pd.DataFrame:
    """AMT_INCOME_TOTAL per household member (family disposable income)."""
    df = df.copy()
    family = df["CNT_FAM_MEMBERS"].clip(lower=1)
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / family
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXT_SOURCE Aggregate Features  (top predictors in Home Credit data)
# ─────────────────────────────────────────────────────────────────────────────

def create_ext_source_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate EXT_SOURCE_1/2/3 (external credit bureau scores).

    These are the strongest individual predictors of default in the Home
    Credit dataset.  We create:

    - EXT_SOURCE_MEAN   : average of available scores
    - EXT_SOURCE_MIN    : worst single score
    - EXT_SOURCE_MAX    : best single score
    - EXT_SOURCE_STD    : volatility across scores (high std → instability)
    - EXT_SOURCE_NANSUM : count of non-null scores (more data → lower uncertainty)
    """
    df = df.copy()
    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]

    if not ext_cols:
        LOGGER.warning("No EXT_SOURCE columns found; skipping EXT_SOURCE features.")
        return df

    ext_df = df[ext_cols]
    df["EXT_SOURCE_MEAN"]   = ext_df.mean(axis=1)
    df["EXT_SOURCE_MIN"]    = ext_df.min(axis=1)
    df["EXT_SOURCE_MAX"]    = ext_df.max(axis=1)
    df["EXT_SOURCE_STD"]    = ext_df.std(axis=1).fillna(0)
    df["EXT_SOURCE_NANSUM"] = ext_df.notna().sum(axis=1)   # count of available scores

    # Interaction: product of all three scores (zero if any score is 0)
    if len(ext_cols) == 3:
        df["EXT_SOURCE_PRODUCT"] = (
            df["EXT_SOURCE_1"].fillna(0)
            * df["EXT_SOURCE_2"].fillna(0)
            * df["EXT_SOURCE_3"].fillna(0)
        )

    LOGGER.info(f"Created EXT_SOURCE aggregate features from {ext_cols}.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Age & Employment Features
# ─────────────────────────────────────────────────────────────────────────────

def create_employment_age_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    |DAYS_EMPLOYED| / |DAYS_BIRTH|

    Employment tenure as a fraction of the applicant's age.
    0 → newly employed or unemployed; 1 → employed their entire adult life.

    Note: Both columns are stored as NEGATIVE integers in the raw data
    (days before the application date).  We use abs() to get positive values.
    """
    df = df.copy()
    age_abs = np.abs(df["DAYS_BIRTH"])
    empl_abs = np.abs(df["DAYS_EMPLOYED"])
    df["EMPLOYMENT_AGE_RATIO"] = np.where(
        age_abs > 0,
        empl_abs / age_abs,
        0.0,
    )
    return df


def create_age_years(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applicant age in years (absolute, positive).

    DAYS_BIRTH is stored as a negative integer.  Converting to years makes
    the feature human-interpretable and SHAP plots more readable.
    """
    df = df.copy()
    df["AGE_YEARS"] = np.abs(df["DAYS_BIRTH"]) / 365.25
    return df


def create_employment_years(df: pd.DataFrame) -> pd.DataFrame:
    """Employment tenure in years (DAYS_EMPLOYED converted, NaN-safe)."""
    df = df.copy()
    df["EMPLOYMENT_YEARS"] = np.abs(df["DAYS_EMPLOYED"]) / 365.25
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Document & Credit Bureau Request Features
# ─────────────────────────────────────────────────────────────────────────────

def create_document_flags_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count of submitted FLAG_DOCUMENT_* flags per applicant.

    Submitting more supporting documents may correlate with lower risk
    (thorough applicants) or higher risk (compensating for poor scores).
    The count captures this signal in a single integer.
    """
    doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
    if not doc_cols:
        return df
    df = df.copy()
    df["DOCUMENT_COUNT"] = df[doc_cols].sum(axis=1)
    LOGGER.info(f"Created DOCUMENT_COUNT from {len(doc_cols)} FLAG_DOCUMENT_* columns.")
    return df


def create_credit_bureau_inquiry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate credit bureau inquiry counts.

    Many inquiries in a short window signal credit-seeking behaviour
    (associated with financial stress).

    - BUREAU_INQUIRY_TOTAL : sum of all time-window inquiry counts
    - BUREAU_INQUIRY_RECENT: inquiries in the last day + hour (most urgent signal)
    """
    inquiry_cols = [
        c for c in df.columns
        if c.startswith("AMT_REQ_CREDIT_BUREAU_")
    ]
    if not inquiry_cols:
        return df

    df = df.copy()
    df["BUREAU_INQUIRY_TOTAL"] = df[inquiry_cols].sum(axis=1)

    recent_cols = [c for c in inquiry_cols if c.endswith(("_HOUR", "_DAY"))]
    if recent_cols:
        df["BUREAU_INQUIRY_RECENT"] = df[recent_cols].sum(axis=1)

    LOGGER.info(f"Created bureau inquiry features from {len(inquiry_cols)} columns.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. ID Column Removal
# ─────────────────────────────────────────────────────────────────────────────

def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop identifier columns that must NOT enter the model feature matrix.

    SK_ID_CURR is a customer ID — including it would cause data leakage
    (the model could memorise IDs) and degrade generalisation.
    """
    id_cols = [c for c in ["SK_ID_CURR"] if c in df.columns]
    if id_cols:
        df = df.drop(columns=id_cols)
        LOGGER.info(f"Dropped identifier columns: {id_cols}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master Feature Engineering Function
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, drop_ids: bool = True) -> pd.DataFrame:
    """
    Apply the complete feature engineering pipeline.

    Parameters
    ----------
    df       : Preprocessed DataFrame (post preprocess.py).
    drop_ids : If True (default), SK_ID_CURR is removed.
               Set to False only when you need SK_ID_CURR for join/output.

    Returns
    -------
    DataFrame with all engineered features added and ID columns removed.
    """
    LOGGER.info(f"Starting feature engineering on shape {df.shape} …")

    # ── 1. Financial burden ratios ──────────────────────────────────────────
    df = create_credit_income_ratio(df)
    df = create_annuity_income_ratio(df)
    df = create_goods_credit_ratio(df)
    df = create_annuity_credit_ratio(df)
    df = create_credit_per_person(df)
    df = create_income_per_person(df)

    # ── 2. EXT_SOURCE aggregates ────────────────────────────────────────────
    df = create_ext_source_features(df)

    # ── 3. Age & employment ─────────────────────────────────────────────────
    df = create_employment_age_ratio(df)
    df = create_age_years(df)
    df = create_employment_years(df)

    # ── 4. Behavioural signals ──────────────────────────────────────────────
    df = create_document_flags_count(df)
    df = create_credit_bureau_inquiry_features(df)

    # ── 5. Drop IDs ─────────────────────────────────────────────────────────
    if drop_ids:
        df = drop_id_columns(df)

    n_new = df.shape[1]
    LOGGER.info(f"Feature engineering complete → shape {df.shape}")
    return df