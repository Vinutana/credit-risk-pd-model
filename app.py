"""
app.py — Credit Risk Scoring System (Streamlit)
================================================
Production-grade Streamlit application for batch credit risk scoring.

Features
--------
- CSV upload with column validation
- Automatic preprocessing & feature engineering (using fitted training artifacts)
- Batch probability-of-default (PD) scoring
- Risk band classification (Low / Medium / High / Very High)
- Portfolio summary dashboard with metrics & charts
- Downloadable scored CSV

Output Columns
--------------
All original customer columns are preserved, plus:
    Prediction           : 0 (non-default) / 1 (default)
    Probability_Default  : PD probability [0, 1]
    PD                   : PD as percentage [0, 100]
    Risk_Band            : Categorical risk tier
    Recommendation       : Business lending decision

Run
---
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the Python path when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Scoring System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Cached Artifact Loading (loaded once per Streamlit session)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artifacts …")
def load_scoring_components():
    """
    Load and cache all inference artifacts.

    Using @st.cache_resource means these are loaded ONCE per server session
    instead of on every user upload — critical for large models.
    """
    from src.predict import load_model, load_features, load_preprocessors, load_threshold
    model         = load_model()
    feature_names = load_features()
    imputers      = load_preprocessors()  # (imputer_map, cap_map, encoder_map)
    threshold     = load_threshold()
    return model, feature_names, imputers, threshold


def check_artifacts_available() -> bool:
    """Return True only if all required model artifacts exist on disk."""
    from src.config import MODEL_PATH, FEATURE_PATH, IMPUTER_PATH, THRESHOLD_PATH
    missing = [p for p in [MODEL_PATH, FEATURE_PATH, IMPUTER_PATH, THRESHOLD_PATH] if not p.exists()]
    if missing:
        st.error(
            "⚠️ **Model artifacts not found.**\n\n"
            "Please run the training pipeline first:\n"
            "```bash\n"
            "python src/pipeline.py   # data preparation\n"
            "python src/train.py      # model training\n"
            "```\n\n"
            f"Missing: {[str(p) for p in missing]}"
        )
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Scoring Helper
# ─────────────────────────────────────────────────────────────────────────────

def run_batch_scoring(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full inference pipeline on a batch DataFrame.

    Uses cached artifacts — no re-loading on every call.
    """
    from src.predict import preprocess_for_inference
    import numpy as np
    from src.utils import assign_risk_band, assign_recommendation

    model, feature_names, (imputer_map, cap_map, encoder_map), threshold = load_scoring_components()

    df_proc = preprocess_for_inference(raw_df, imputer_map, cap_map, encoder_map)

    # Warn about unexpected missing columns
    missing_cols = set(feature_names) - set(df_proc.columns)
    if missing_cols:
        st.warning(
            f"⚠️ {len(missing_cols)} columns expected by the model are missing from "
            f"the uploaded file and will be filled with 0.\n"
            f"First 5 missing: {sorted(missing_cols)[:5]}"
        )

    df_model = df_proc.reindex(columns=feature_names, fill_value=0)
    y_prob   = model.predict_proba(df_model)[:, 1]
    y_pred   = (y_prob >= threshold).astype(int)

    result = raw_df.copy()
    result["Prediction"]          = y_pred
    result["Probability_Default"] = np.round(y_prob, 6)
    result["PD"]                  = np.round(y_prob * 100, 2)
    result["Risk_Band"]           = [assign_risk_band(p)       for p in y_prob]
    result["Recommendation"]      = [assign_recommendation(p)  for p in y_prob]

    return result


# ─────────────────────────────────────────────────────────────────────────────
# UI — Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-building.png", width=64)
    st.title("Credit Risk System")
    st.markdown("---")
    st.markdown(
        """
        **Model:** XGBoost Classifier  
        **Dataset:** Home Credit Default Risk  
        **Target:** Probability of Default (PD)
        """
    )
    st.markdown("---")
    st.markdown(
        """
        **Risk Band Thresholds**
        | Band | PD Range |
        |------|----------|
        | 🟢 Low Risk | < 10% |
        | 🟡 Medium Risk | 10% – 30% |
        | 🟠 High Risk | 30% – 50% |
        | 🔴 Very High Risk | ≥ 50% |
        """
    )
    st.markdown("---")
    st.markdown(
        """
        **Output Columns Added**
        - `Prediction` (0/1)
        - `Probability_Default` (0–1)
        - `PD` (percentage)
        - `Risk_Band`
        - `Recommendation`
        """
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI — Main Content
# ─────────────────────────────────────────────────────────────────────────────

st.title("🏦 Credit Risk Scoring System")
st.markdown(
    """
    Upload a customer dataset in CSV format to generate **Probability of Default (PD)** scores,
    risk band classifications, and lending recommendations for each customer.

    The system applies the same preprocessing and feature engineering pipeline
    used during model training — no manual feature preparation required.
    """
)

# Early check — fail fast if artifacts are missing
if not check_artifacts_available():
    st.stop()

# Load components upfront (triggers cache if already loaded)
try:
    load_scoring_components()
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# File Upload
# ─────────────────────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    label="📂 Upload Customer Dataset (CSV)",
    type=["csv"],
    help="Upload a CSV file with the same column structure as application_train.csv. "
         "The TARGET column will be ignored if present.",
)

if uploaded_file is None:
    st.info("📄 Please upload a CSV file to begin scoring.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Data Preview
# ─────────────────────────────────────────────────────────────────────────────

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Could not read the uploaded file: {e}")
    st.stop()

st.subheader("📋 Uploaded Dataset Preview")
col_a, col_b, col_c = st.columns(3)
col_a.metric("Customers", f"{len(raw_df):,}")
col_b.metric("Columns",   f"{raw_df.shape[1]}")
has_target = "TARGET" in raw_df.columns
col_c.metric("Has TARGET column", "✅ Yes" if has_target else "❌ No")

with st.expander("Preview first 5 rows"):
    st.dataframe(raw_df.head(), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("🔄 Running credit risk scoring pipeline …"):
    try:
        scored_df = run_batch_scoring(raw_df)
    except Exception as e:
        st.error(f"❌ Scoring failed: {e}")
        st.exception(e)
        st.stop()

st.success(f"✅ Successfully scored **{len(scored_df):,}** customers.")

# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Summary Dashboard
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📊 Portfolio Summary")

band_counts = scored_df["Risk_Band"].value_counts()
ordered_bands = ["Low Risk", "Medium Risk", "High Risk", "Very High Risk"]

c1, c2, c3, c4 = st.columns(4)
band_colours   = ["🟢", "🟡", "🟠", "🔴"]

for col, band, emoji in zip([c1, c2, c3, c4], ordered_bands, band_colours):
    count = band_counts.get(band, 0)
    pct   = count / len(scored_df) * 100 if len(scored_df) > 0 else 0
    col.metric(f"{emoji} {band}", f"{count:,}", f"{pct:.1f}%")

# Key statistics
st.markdown("#### Key Statistics")
sk1, sk2, sk3, sk4 = st.columns(4)
sk1.metric("Mean PD",   f"{scored_df['PD'].mean():.1f}%")
sk2.metric("Median PD", f"{scored_df['PD'].median():.1f}%")
sk3.metric("Max PD",    f"{scored_df['PD'].max():.1f}%")
sk4.metric("% Predicted Default",
           f"{(scored_df['Prediction'] == 1).mean()*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("📈 Risk Distribution")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("**Customer Count by Risk Band**")
    risk_summary = (
        scored_df["Risk_Band"]
        .value_counts()
        .reindex(ordered_bands, fill_value=0)
        .reset_index()
    )
    risk_summary.columns = ["Risk Band", "Count"]
    st.bar_chart(risk_summary.set_index("Risk Band"), use_container_width=True)

with chart_col2:
    st.markdown("**Recommendation Distribution**")
    rec_summary = scored_df["Recommendation"].value_counts().reset_index()
    rec_summary.columns = ["Recommendation", "Count"]
    st.bar_chart(rec_summary.set_index("Recommendation"), use_container_width=True)

# PD distribution histogram
st.markdown("**PD Score Distribution**")
st.bar_chart(
    scored_df["PD"].round(0).value_counts().sort_index(),
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Scored Data Table
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("🗂 Scored Customer Data")

output_cols = ["Prediction", "Probability_Default", "PD", "Risk_Band", "Recommendation"]
other_cols  = [c for c in scored_df.columns if c not in output_cols]
display_df  = scored_df[output_cols + other_cols]

# Colour-code Risk_Band column via pandas Styler
def colour_band(val):
    colours = {
        "Low Risk"       : "background-color: #d4edda; color: #155724",
        "Medium Risk"    : "background-color: #fff3cd; color: #856404",
        "High Risk"      : "background-color: #fde8d8; color: #7c3700",
        "Very High Risk" : "background-color: #f8d7da; color: #721c24",
    }
    return colours.get(val, "")

styled = display_df.head(100).style.applymap(colour_band, subset=["Risk_Band"])
st.dataframe(styled, use_container_width=True, height=400)

if len(scored_df) > 100:
    st.caption(f"Showing first 100 of {len(scored_df):,} rows. Download the full dataset below.")

# ─────────────────────────────────────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("⬇️ Download Results")

csv_data = display_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label     = "📥 Download Scored Dataset (CSV)",
    data      = csv_data,
    file_name = "credit_risk_predictions.csv",
    mime      = "text/csv",
    use_container_width=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; color:#6c757d; font-size:0.85rem;'>
    Credit Risk Scoring System · XGBoost · Home Credit Default Risk Dataset ·
    Built for portfolio demonstration
    </div>
    """,
    unsafe_allow_html=True,
)