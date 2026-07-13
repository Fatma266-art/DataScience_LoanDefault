"""
Loan Default Prediction — Streamlit GUI
========================================
Based on the "Give Me Some Credit" Kaggle dataset.

Usage:
    streamlit run app.py

If model_artifacts/ does not exist, the app trains a lightweight demo model
on synthetic data so you can explore the interface immediately.
Place the real artifacts (from the notebook) in model_artifacts/ for accurate predictions.
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* Top banner */
        .main-header {
            background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
            padding: 2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            color: white;
        }
        .main-header h1 { margin: 0; font-size: 2rem; }
        .main-header p  { margin: .4rem 0 0; opacity: .85; font-size: 1rem; }

        /* Result cards */
        .result-safe {
            background: #E8F5E9;
            border-left: 6px solid #2E7D32;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
        }
        .result-risk {
            background: #FFEBEE;
            border-left: 6px solid #C62828;
            border-radius: 8px;
            padding: 1.2rem 1.5rem;
        }
        .metric-card {
            background: #F5F5F5;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .metric-card h3 { margin: 0; font-size: 1.6rem; color: #1565C0; }
        .metric-card p  { margin: .2rem 0 0; color: #555; font-size: .85rem; }

        /* Section label */
        .section-label {
            font-size: .75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .08em;
            color: #1565C0;
            margin-bottom: .4rem;
        }

        /* Info banner */
        .demo-banner {
            background: #FFF8E1;
            border-left: 4px solid #F9A825;
            border-radius: 6px;
            padding: .8rem 1rem;
            font-size: .88rem;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Demo-model builder (no real artifacts present)
# ─────────────────────────────────────────────
ARTIFACTS_DIR = "model_artifacts"
DEMO_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
DEMO_THR_PATH   = os.path.join(ARTIFACTS_DIR, "best_threshold.pkl")
DEMO_FEAT_PATH  = os.path.join(ARTIFACTS_DIR, "feature_columns.pkl")
DEMO_INFO_PATH  = os.path.join(ARTIFACTS_DIR, "model_info.pkl")


def build_demo_model():
    """Train a lightweight XGBoost on synthetic data as a fallback."""
    from sklearn.datasets import make_classification
    from xgboost import XGBClassifier

    np.random.seed(42)
    n = 5000
    X_syn, y_syn = make_classification(
        n_samples=n, n_features=10,
        weights=[0.93, 0.07], random_state=42
    )
    feature_cols = [
        "RevolvingUtilizationRate", "Age", "Late30to59Days", "DebtRatio",
        "MonthlyIncome", "Times90DaysLate", "TotalLatePayments",
        "IncomePerDependent", "CreditRiskScore", "AgeGroup_encoded",
    ]
    model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42,
                          scale_pos_weight=13, eval_metric="aucpr")
    model.fit(X_syn, y_syn)

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(model,        DEMO_MODEL_PATH)
    joblib.dump(0.35,         DEMO_THR_PATH)
    joblib.dump(feature_cols, DEMO_FEAT_PATH)
    info = {
        "model_name": "XGB — Demo (synthetic data)",
        "threshold": 0.35,
        "feature_columns": feature_cols,
        "metrics": {
            "precision": 0.0, "recall": 0.0,
            "f1_score": 0.0, "roc_auc": 0.0, "auc_pr": 0.0,
        },
        "is_demo": True,
    }
    joblib.dump(info, DEMO_INFO_PATH)
    return model, 0.35, feature_cols, info


# ─────────────────────────────────────────────
# Load artifacts (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_artifacts():
    if not os.path.exists(DEMO_MODEL_PATH):
        return build_demo_model()
    model     = joblib.load(DEMO_MODEL_PATH)
    threshold = joblib.load(DEMO_THR_PATH)
    features  = joblib.load(DEMO_FEAT_PATH)
    info      = joblib.load(DEMO_INFO_PATH) if os.path.exists(DEMO_INFO_PATH) else {}
    return model, threshold, features, info


model, threshold, feature_cols, model_info = load_artifacts()
is_demo = model_info.get("is_demo", False)

# ─────────────────────────────────────────────
# All possible derived-feature names
# ─────────────────────────────────────────────
DERIVED = {"TotalLatePayments", "IncomePerDependent",
           "DebtBurden", "CreditRiskScore", "AgeGroup_encoded"}


def build_features(raw: dict) -> pd.DataFrame:
    """Add derived features then return a DataFrame with exactly feature_cols."""
    d = dict(raw)
    d["TotalLatePayments"]  = d["Late30to59Days"] + d.get("Late60to89Days", 0) + d["Times90DaysLate"]
    d["IncomePerDependent"] = d["MonthlyIncome"] / (d["NumberOfDependents"] + 1)
    d["DebtBurden"]         = d["DebtRatio"] * d["MonthlyIncome"]
    d["CreditRiskScore"]    = d["RevolvingUtilizationRate"] * (d["TotalLatePayments"] + 1)
    age = d["Age"]
    if   age < 25:  d["AgeGroup_encoded"] = 0
    elif age < 35:  d["AgeGroup_encoded"] = 1
    elif age < 45:  d["AgeGroup_encoded"] = 2
    elif age < 55:  d["AgeGroup_encoded"] = 3
    elif age < 65:  d["AgeGroup_encoded"] = 4
    else:           d["AgeGroup_encoded"] = 5

    # Only keep columns the model actually needs
    row = {c: d.get(c, 0.0) for c in feature_cols}
    return pd.DataFrame([row])[feature_cols]


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>🏦 Loan Default Predictor</h1>
        <p>Enter borrower information to predict the probability of financial distress within 2 years.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if is_demo:
    st.markdown(
        """<div class="demo-banner">
        ⚠️ <strong>Demo mode:</strong> No trained model was found in <code>model_artifacts/</code>.
        A lightweight demo model trained on synthetic data is being used.
        Run the notebook and copy <code>model_artifacts/</code> here for real predictions.
        </div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Sidebar — model info
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown(f"**Model:** `{model_info.get('model_name', 'Unknown')}`")
    st.markdown(f"**Decision threshold:** `{threshold:.4f}`")
    st.markdown(f"**Features used:** `{len(feature_cols)}`")

    metrics = model_info.get("metrics", {})
    if metrics and not is_demo:
        st.markdown("---")
        st.markdown("**Test-set performance**")
        cols_s = st.columns(2)
        cols_s[0].metric("F1 Score",  f"{metrics.get('f1_score',  0):.3f}")
        cols_s[1].metric("AUC-PR",    f"{metrics.get('auc_pr',    0):.3f}")
        cols_s[0].metric("Precision", f"{metrics.get('precision', 0):.3f}")
        cols_s[1].metric("Recall",    f"{metrics.get('recall',    0):.3f}")
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

    st.markdown("---")
    st.markdown(
        "**Feature legend**\n"
        "- *RevolvingUtilization*: credit used / limit\n"
        "- *DebtRatio*: monthly debt / income\n"
        "- *Late X days*: times past due in last 2 yrs\n"
        "- *OpenCreditLines*: # open accounts\n"
        "- *RealEstateLoans*: # RE / home loans\n"
        "- *NumberOfDependents*: financial dependents"
    )

# ─────────────────────────────────────────────
# Input form — two columns
# ─────────────────────────────────────────────
st.subheader("📋 Borrower Information")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<p class="section-label">Personal & Income</p>', unsafe_allow_html=True)
    age             = st.slider("Age (years)", 18, 109, 45, help="Borrower's age in years")
    monthly_income  = st.number_input("Monthly Income (USD)", min_value=0.0, max_value=50_000.0,
                                      value=5_000.0, step=100.0,
                                      help="Gross monthly income in USD")
    num_dependents  = st.slider("Number of Dependents", 0, 10, 0,
                                help="Number of financial dependents (children, spouse, etc.)")
    debt_ratio      = st.slider("Debt Ratio", 0.0, 1.0, 0.35, 0.01,
                                help="Monthly debt payments ÷ monthly income")
    revolving_util  = st.slider("Revolving Utilization Rate", 0.0, 1.0, 0.30, 0.01,
                                help="Total credit used ÷ total credit limit (0 to 1)")

with col2:
    st.markdown('<p class="section-label">Credit & Payment History</p>', unsafe_allow_html=True)
    late_30_59  = st.slider("Late 30–59 Days (count)", 0, 20, 0,
                            help="Times 30–59 days past due in last 2 years")
    late_60_89  = st.slider("Late 60–89 Days (count)", 0, 20, 0,
                            help="Times 60–89 days past due in last 2 years")
    times_90    = st.slider("Times 90+ Days Late (count)", 0, 20, 0,
                            help="Times 90+ days past due in last 2 years")
    open_credit = st.slider("Open Credit Lines", 0, 30, 5,
                            help="Number of open loans and lines of credit")
    real_estate = st.slider("Real Estate Loans", 0, 10, 1,
                            help="Number of mortgage and real estate loans")

# ─────────────────────────────────────────────
# Predict button
# ─────────────────────────────────────────────
st.markdown("")
predict_btn = st.button("🔍  Predict Default Risk", type="primary", use_container_width=True)

if predict_btn:
    raw_input = {
        "RevolvingUtilizationRate": revolving_util,
        "Age":                age,
        "Late30to59Days":     late_30_59,
        "DebtRatio":          debt_ratio,
        "MonthlyIncome":      monthly_income,
        "OpenCreditLines":    open_credit,
        "Times90DaysLate":    times_90,
        "RealEstateLoans":    real_estate,
        "Late60to89Days":     late_60_89,
        "NumberOfDependents": num_dependents,
    }

    X_input = build_features(raw_input)
    prob    = model.predict_proba(X_input)[0][1]
    pred    = int(prob >= threshold)

    # ── Result banner ──
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    r1, r2, r3 = st.columns([1.5, 1, 1])

    with r1:
        if pred == 1:
            st.markdown(
                f"""<div class="result-risk">
                    <h2 style="color:#C62828; margin:0">⚠️ HIGH RISK — Default Likely</h2>
                    <p style="margin:.5rem 0 0; color:#555">
                      This borrower has a high probability of financial distress within 2 years.
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="result-safe">
                    <h2 style="color:#2E7D32; margin:0">✅ LOW RISK — No Default Expected</h2>
                    <p style="margin:.5rem 0 0; color:#555">
                      This borrower is unlikely to experience financial distress within 2 years.
                    </p>
                </div>""",
                unsafe_allow_html=True,
            )

    with r2:
        st.markdown(
            f"""<div class="metric-card">
                <h3>{prob*100:.1f}%</h3>
                <p>Default Probability</p>
            </div>""",
            unsafe_allow_html=True,
        )

    with r3:
        st.markdown(
            f"""<div class="metric-card">
                <h3 style="color:{'#C62828' if pred==1 else '#2E7D32'}">
                    {'Default' if pred==1 else 'No Default'}
                </h3>
                <p>Prediction (thr = {threshold:.2f})</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Probability gauge ──
    st.markdown("")
    gauge_pct = prob * 100
    color     = "#C62828" if prob >= threshold else "#2E7D32"
    bar_width = max(2, gauge_pct)

    st.markdown("**Default Probability Gauge**")
    st.markdown(
        f"""
        <div style="background:#E0E0E0; border-radius:8px; height:24px; width:100%; overflow:hidden;">
            <div style="background:{color}; width:{bar_width:.1f}%; height:100%;
                        border-radius:8px; transition: width .4s;
                        display:flex; align-items:center; justify-content:flex-end; padding-right:8px;">
                <span style="color:white; font-weight:bold; font-size:.85rem;">{gauge_pct:.1f}%</span>
            </div>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:.75rem; color:#777; margin-top:.2rem;">
            <span>0%  (Safe)</span>
            <span style="color:{color}; font-weight:bold;">Threshold: {threshold*100:.0f}%</span>
            <span>100% (High Risk)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Derived features breakdown ──
    st.markdown("")
    with st.expander("🔬 Show derived features used by the model"):
        total_late     = late_30_59 + late_60_89 + times_90
        income_per_dep = monthly_income / (num_dependents + 1)
        debt_burden    = debt_ratio * monthly_income
        credit_risk    = revolving_util * (total_late + 1)
        age_group      = ["<25","25-35","35-45","45-55","55-65","65+"][
            0 if age<25 else 1 if age<35 else 2 if age<45 else 3 if age<55 else 4 if age<65 else 5
        ]

        feat_df = pd.DataFrame({
            "Derived Feature": [
                "TotalLatePayments",
                "IncomePerDependent",
                "DebtBurden (monthly)",
                "CreditRiskScore",
                "AgeGroup (encoded)",
            ],
            "Value": [
                f"{total_late}",
                f"${income_per_dep:,.2f}",
                f"${debt_burden:,.2f}",
                f"{credit_risk:.4f}",
                f"{age_group}",
            ],
            "Description": [
                "Sum of all 3 late-payment counts",
                "MonthlyIncome ÷ (Dependents + 1)",
                "DebtRatio × MonthlyIncome",
                "RevolvingUtil × (TotalLate + 1)",
                "Age bracket (0–5 ordinal)",
            ],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    # ── Risk breakdown ──
    st.markdown("")
    with st.expander("📈 Risk factor summary"):
        risks = {
            "High revolving utilization (> 0.7)":   revolving_util > 0.7,
            "Any payments 90+ days late":           times_90 > 0,
            "Multiple late payments (30-89 days)":  (late_30_59 + late_60_89) > 2,
            "High debt ratio (> 0.5)":              debt_ratio > 0.5,
            "Low income per dependent (< $2,000)":  income_per_dep < 2000,
            "Young borrower (< 30 years old)":      age < 30,
        }
        flag_rows = [
            {"Factor": k, "Status": "🔴 Risk" if v else "🟢 OK"}
            for k, v in risks.items()
        ]
        st.dataframe(pd.DataFrame(flag_rows), use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:.8rem;'>"
    "Loan Default Predictor · Based on the Give Me Some Credit dataset (Kaggle) · "
    "Fundamentals of Data Science Project"
    "</p>",
    unsafe_allow_html=True,
)