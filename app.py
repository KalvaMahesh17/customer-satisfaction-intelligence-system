"""
Customer Satisfaction Intelligence System
Web Application built with Streamlit
Course: IT503PC Machine Learning | Lab: IT505PC Machine Learning Lab
Author: College B.Tech Project
"""

import os
import json
import joblib
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import (
    load_dataset,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    VALID_TARGET_CLASSES,
    inspect_dataset_summary,
)
from src.analysis import (
    CLASS_COLOR_MAP,
    plot_target_distribution_plotly,
    plot_service_factor_vs_target_plotly,
    plot_numerical_distributions_plotly,
    plot_correlation_heatmap_plotly,
    get_service_factor_summary,
    get_correlation_matrix,
)
from src.explainability import (
    explain_single_prediction_shap,
    plot_local_explanation_plotly,
    plot_feature_importance_plotly,
)

# =====================================================================
# STREAMLIT PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Customer Satisfaction Intelligence System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CUSTOM CSS STYLING FOR PREMIUM SAAS UI (GeneX Reference)
# =====================================================================
st.markdown(
    """
    <style>
    /* Google Fonts: Plus Jakarta Sans for UI + JetBrains Mono for numerals */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #12233F;
    }

    /* Light / Calm SaaS workspace canvas (Restrained #F4F7FB) */
    .stApp {
        background-color: #F4F7FB;
        background-image: none !important;
    }

    /* Clean Main Container Spacing */
    .main .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3.5rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 1440px !important;
    }

    /* =====================================================================
       SIDEBAR - DEEP NAVY PROFESSIONAL SAAS SHELL (GeneX reference)
       ===================================================================== */
    [data-testid="stSidebar"] {
        background-color: #0B1727 !important;
        border-right: 1px solid #16263D !important;
        box-shadow: 4px 0 24px rgba(11, 23, 39, 0.08) !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        background-color: #0B1727 !important;
        padding-top: 1.2rem !important;
    }

    /* Sidebar branding header */
    .sidebar-brand-container {
        padding: 0.5rem 0.6rem 1.2rem 0.6rem;
        border-bottom: 1px solid #1A2E4B;
        margin-bottom: 1.2rem;
    }

    .sidebar-brand-title {
        font-size: 1.12rem;
        font-weight: 800;
        letter-spacing: 0.04em;
        color: #FFFFFF;
        line-height: 1.25;
        text-transform: uppercase;
        margin: 0;
    }

    .sidebar-brand-title span {
        color: #0EA5E9;
    }

    .sidebar-brand-sub {
        font-size: 0.76rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 0.35rem;
        letter-spacing: 0.02em;
    }

    .sidebar-nav-label {
        font-size: 0.68rem;
        font-weight: 700;
        color: #475569;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0 0.6rem 0.6rem 0.6rem;
    }

    /* Streamlit Radio styling in Dark Navy Sidebar */
    [data-testid="stSidebar"] .stRadio > div {
        gap: 0.3rem !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        background-color: transparent !important;
        border-radius: 10px !important;
        padding: 0.55rem 0.85rem !important;
        font-weight: 500 !important;
        font-size: 0.88rem !important;
        color: #94A3B8 !important;
        transition: all 0.18s ease-in-out !important;
        border: 1px solid transparent !important;
        cursor: pointer !important;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        background-color: #13243C !important;
        color: #F8FAFC !important;
    }

    /* Highlight for selected radio button */
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"],
    [data-testid="stSidebar"] .stRadio label:has(input:checked) {
        background-color: #182D4B !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-left: 3px solid #0EA5E9 !important;
        border-color: transparent transparent transparent #0EA5E9 !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15) !important;
    }

    /* Sidebar Academic Info Card */
    .sidebar-academic-card {
        background-color: #0E1D31;
        border: 1px solid #1E3352;
        border-radius: 12px;
        padding: 0.95rem 1rem;
        margin-top: 1.4rem;
    }

    .sidebar-academic-header {
        font-size: 0.68rem;
        font-weight: 700;
        color: #38BDF8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.55rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .sidebar-academic-content {
        font-size: 0.78rem;
        color: #94A3B8;
        line-height: 1.6;
    }

    .sidebar-academic-content b {
        color: #E2E8F0;
    }

    .sidebar-academic-footer {
        margin-top: 0.7rem;
        padding-top: 0.5rem;
        border-top: 1px solid #1A2E48;
        font-size: 0.7rem;
        color: #64748B;
    }

    /* =====================================================================
       TOP HEADER AREA (COMPACT SAAS CARD)
       ===================================================================== */
    .saas-header-card {
        background: #FFFFFF;
        border: 1px solid #E4EAF2;
        border-radius: 16px;
        padding: 1.25rem 1.6rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 20px rgba(15, 39, 71, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .saas-header-left h1 {
        font-size: 1.5rem !important;
        font-weight: 800 !important;
        color: #0F2747 !important;
        margin: 0 !important;
        letter-spacing: -0.02em !important;
        line-height: 1.25 !important;
    }

    .saas-header-left p {
        font-size: 0.9rem !important;
        color: #475569 !important;
        margin: 0.25rem 0 0.35rem 0 !important;
        font-weight: 500 !important;
    }

    .saas-header-left .sub-tagline {
        font-size: 0.78rem;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .saas-header-left .sub-tagline span {
        display: inline-block;
        background: #F1F5F9;
        color: #334155;
        padding: 0.15rem 0.5rem;
        border-radius: 6px;
        margin-right: 0.4rem;
        font-size: 0.72rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 0.45rem 0.9rem;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        white-space: nowrap;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10B981;
    }

    /* =====================================================================
       KPI SAAS CARDS
       ===================================================================== */
    .kpi-card-saas {
        background: #FFFFFF;
        border: 1px solid #E4EAF2;
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 4px 18px rgba(15, 39, 71, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .kpi-card-saas:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(15, 39, 71, 0.08);
    }

    .kpi-label-saas {
        font-size: 0.72rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    .kpi-indicator-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }

    .kpi-value-saas {
        font-size: 1.75rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 0.35rem;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .kpi-value-saas.champion-text {
        font-size: 1.15rem !important;
        color: #0F2747 !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .kpi-sub-saas {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 0.3rem;
        font-weight: 500;
    }

    /* =====================================================================
       WHITE CONTENT & ANALYTICS CARDS
       ===================================================================== */
    .content-card-saas {
        background: #FFFFFF;
        border: 1px solid #E4EAF2;
        border-radius: 16px;
        padding: 1.35rem 1.5rem;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 20px rgba(15, 39, 71, 0.04);
    }

    .card-header-eyebrow {
        font-size: 0.7rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-bottom: 0.2rem;
    }

    .card-header-title {
        font-size: 1.12rem;
        font-weight: 700;
        color: #0F2747;
        letter-spacing: -0.01em;
        margin-bottom: 0.2rem;
    }

    .card-header-desc {
        font-size: 0.82rem;
        color: #64748B;
        margin-bottom: 0.85rem;
        font-weight: 400;
    }

    /* Result Badges */
    .badge-satisfied {
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }

    .badge-neutral {
        background: #FFFBEB;
        color: #92400E;
        border: 1px solid #FDE68A;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }

    .badge-dissatisfied {
        background: #FFF1F2;
        color: #9F1239;
        border: 1px solid #FECDD3;
        padding: 0.6rem 1.2rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }

    /* =====================================================================
       STREAMLIT BUTTONS & INPUT CONTROLS
       ===================================================================== */
    div.stButton > button, div.stDownloadButton > button {
        background: #0F2747 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        border: 1px solid #0F2747 !important;
        border-radius: 10px !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 2px 8px rgba(15, 39, 71, 0.12) !important;
        transition: all 0.18s ease-in-out !important;
    }

    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background: #183B68 !important;
        border-color: #183B68 !important;
        box-shadow: 0 4px 14px rgba(15, 39, 71, 0.2) !important;
        transform: translateY(-1px) !important;
    }

    div.stButton > button:active, div.stDownloadButton > button:active {
        transform: translateY(0px) !important;
    }

    /* Form and Inputs */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        color: #0F172A !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    }

    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="input"] > div:focus-within {
        border-color: #0EA5E9 !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.15) !important;
    }

    /* Streamlit Tabs - Clean Modern Segmented Bar */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: #E8EEF5 !important;
        border-radius: 12px !important;
        padding: 0.3rem !important;
        border-bottom: none !important;
        margin-bottom: 1.2rem !important;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        color: #64748B !important;
        padding: 0.45rem 1rem !important;
        transition: all 0.15s ease !important;
    }

    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #0F2747 !important;
        box-shadow: 0 2px 8px rgba(15, 39, 71, 0.08) !important;
    }

    /* Dataframe and Tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #E4EAF2 !important;
        border-radius: 14px !important;
        background: #FFFFFF !important;
        box-shadow: 0 2px 12px rgba(15, 39, 71, 0.03) !important;
        overflow: hidden !important;
    }

    /* Metric Widgets */
    div[data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E4EAF2 !important;
        border-radius: 14px !important;
        padding: 0.95rem 1.15rem !important;
        box-shadow: 0 2px 10px rgba(15, 39, 71, 0.03) !important;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        color: #64748B !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }

    div[data-testid="stMetricValue"] {
        font-size: 1.55rem !important;
        font-weight: 800 !important;
        color: #0F172A !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# CACHED DATA & MODEL LOADERS (HIGH PERFORMANCE)
# =====================================================================
@st.cache_data
def get_cached_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource
def get_cached_models() -> Dict[str, Any]:
    best_model = joblib.load("models/saved_model_pipeline.pkl")
    all_models = joblib.load("models/all_models.pkl")
    splits = joblib.load("models/test_split.pkl")
    return {
        "best_model": best_model,
        "all_models": all_models,
        "splits": splits,
    }


@st.cache_data
def get_cached_metadata() -> Dict[str, Any]:
    with open("models/model_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open("models/feature_metadata.json", "r", encoding="utf-8") as f:
        f_meta = json.load(f)
    return {"model": meta, "feature": f_meta}


@st.cache_data
def get_cached_permutation_importance() -> pd.DataFrame:
    path = "outputs/results/permutation_importance.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    # Compute on the fly if not cached
    models = get_cached_models()
    from src.explainability import compute_global_permutation_importance
    df_imp = compute_global_permutation_importance(
        models["best_model"],
        models["splits"]["X_test"],
        models["splits"]["y_test"],
    )
    df_imp.to_csv(path, index=False)
    return df_imp


# Load global artifacts safely
try:
    df_raw = get_cached_dataset()
    models_dict = get_cached_models()
    metadata = get_cached_metadata()
    model_meta = metadata["model"]
    feature_meta = metadata["feature"]
    best_pipeline = models_dict["best_model"]
    all_pipelines = models_dict["all_models"]
    X_test_ref = models_dict["splits"]["X_test"]
    y_test_ref = models_dict["splits"]["y_test"]
except Exception as e:
    st.error(f"Error loading system artifacts: {e}. Please ensure src/train_model.py was executed.")
    st.stop()


# =====================================================================
# SIDEBAR NAVIGATION & ACADEMIC INFO (GeneX SAAS STYLE)
# =====================================================================
st.sidebar.markdown(
    """
    <div class="sidebar-brand-container">
        <h2 class="sidebar-brand-title">CUSTOMER <span>INTELLIGENCE</span></h2>
        <div class="sidebar-brand-sub">Supervised ML Analytics Platform</div>
    </div>
    <div class="sidebar-nav-label">Navigation</div>
    """,
    unsafe_allow_html=True,
)

menu_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Dashboard",
        "📊 Data Analysis",
        "🤖 Model Performance",
        "🔮 Customer Prediction",
        "🧪 Scenario Simulator",
        "📂 Upload & Batch Prediction",
        "ℹ️ About Project",
    ],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown(
    f"""
    <div class="sidebar-academic-card">
        <div class="sidebar-academic-header">
            <span style="width: 5px; height: 5px; border-radius: 50%; background: #38BDF8; display: inline-block;"></span>
            Academic Project
        </div>
        <div class="sidebar-academic-content">
            <div><b>Course:</b> IT503PC — Machine Learning</div>
            <div><b>Lab:</b> IT505PC — ML Lab</div>
            <div><b>Champion:</b> <code style="color: #38BDF8; background: #132742; padding: 0.1rem 0.3rem; border-radius: 4px;">{model_meta['best_model_name']}</code></div>
            <div><b>Metric:</b> 5-Fold CV Macro F1</div>
            <div><b>Seed:</b> <code style="color: #34D399; background: #132742; padding: 0.1rem 0.3rem; border-radius: 4px;">42</code></div>
        </div>
        <div class="sidebar-academic-footer">
            B.Tech Academic System
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =====================================================================
# PAGE 1: 🏠 DASHBOARD
# =====================================================================
if menu_selection == "🏠 Dashboard":
    st.markdown(
        """
        <div class="saas-header-card">
            <div class="saas-header-left">
                <h1>Customer Satisfaction Intelligence System</h1>
                <p>Customer Satisfaction Classification and Service Analysis</p>
                <div class="sub-tagline">
                    <span>Supervised Machine Learning Analytics Platform</span>
                    <span>IT503PC / IT505PC Curriculum</span>
                </div>
            </div>
            <div>
                <span class="status-pill">
                    <span class="status-dot"></span>
                    ML SYSTEM READY
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI cards
    total_cust = len(df_raw)
    sat_count = (df_raw[TARGET_COLUMN] == "Satisfied").sum()
    neu_count = (df_raw[TARGET_COLUMN] == "Neutral").sum()
    dis_count = (df_raw[TARGET_COLUMN] == "Dissatisfied").sum()
    best_m_name = model_meta["best_model_name"]
    best_cv_f1 = model_meta["cv_scores"][best_m_name]["cv_macro_f1"]
    best_test_acc = model_meta["test_results"][best_m_name]["test_accuracy"]

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f"""
            <div class="kpi-card-saas">
                <div class="kpi-label-saas">
                    <span class="kpi-indicator-dot" style="background: #0EA5E9;"></span>
                    Total Records
                </div>
                <div class="kpi-value-saas">{total_cust:,}</div>
                <div class="kpi-sub-saas">Synthetic cohort</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="kpi-card-saas">
                <div class="kpi-label-saas">
                    <span class="kpi-indicator-dot" style="background: #0D9488;"></span>
                    Satisfied
                </div>
                <div class="kpi-value-saas" style="color: #0D9488;">{sat_count}</div>
                <div class="kpi-sub-saas">33.3% of cohort</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="kpi-card-saas">
                <div class="kpi-label-saas">
                    <span class="kpi-indicator-dot" style="background: #D97706;"></span>
                    Neutral
                </div>
                <div class="kpi-value-saas" style="color: #D97706;">{neu_count}</div>
                <div class="kpi-sub-saas">33.3% of cohort</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="kpi-card-saas">
                <div class="kpi-label-saas">
                    <span class="kpi-indicator-dot" style="background: #E11D48;"></span>
                    Dissatisfied
                </div>
                <div class="kpi-value-saas" style="color: #E11D48;">{dis_count}</div>
                <div class="kpi-sub-saas">33.3% of cohort</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f"""
            <div class="kpi-card-saas">
                <div class="kpi-label-saas">
                    <span class="kpi-indicator-dot" style="background: #2563EB;"></span>
                    Champion Model
                </div>
                <div class="kpi-value-saas champion-text">{best_m_name}</div>
                <div class="kpi-sub-saas">CV F1: <b>{best_cv_f1:.3f}</b> · Acc: <b>{best_test_acc*100:.1f}%</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # Visual Section: Two columns
    dash_col1, dash_col2 = st.columns([1, 1.2])

    with dash_col1:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Satisfaction Overview</div>
                <div class="card-header-title">Customer Satisfaction Distribution</div>
                <div class="card-header-desc">Distribution across the three target classes (Balanced benchmark)</div>
            """,
            unsafe_allow_html=True,
        )
        fig_dist = plot_target_distribution_plotly(df_raw)
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with dash_col2:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Benchmark Evaluation</div>
                <div class="card-header-title">Model Comparison (CV Macro F1 vs Test Accuracy)</div>
                <div class="card-header-desc">Stratified 5-Fold Cross-Validation on training data vs untouched test set</div>
            """,
            unsafe_allow_html=True,
        )
        comp_df = pd.DataFrame(model_meta["comparison_table"])
        
        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Bar(
                name="CV Macro F1",
                x=comp_df["Model"],
                y=comp_df["CV Macro F1"],
                marker_color="#2563EB",
                text=[f"{v:.3f}" for v in comp_df["CV Macro F1"]],
                textposition="outside",
                textfont=dict(size=11, color="#0F172A", family="Plus Jakarta Sans, Inter, sans-serif"),
                width=0.36,
            )
        )
        fig_comp.add_trace(
            go.Bar(
                name="Test Accuracy",
                x=comp_df["Model"],
                y=comp_df["Test Accuracy"],
                marker_color="#0D9488",
                text=[f"{v*100:.1f}%" for v in comp_df["Test Accuracy"]],
                textposition="outside",
                textfont=dict(size=11, color="#0F172A", family="Plus Jakarta Sans, Inter, sans-serif"),
                width=0.36,
            )
        )
        fig_comp.update_layout(
            barmode="group",
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
            height=340,
            margin=dict(l=20, r=20, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
            yaxis=dict(range=[0, 1.15], gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Key Service Factors Overview
    st.markdown(
        """
        <div class="content-card-saas">
            <div class="card-header-eyebrow">Service Impact</div>
            <div class="card-header-title">Key Influential Service Factors (Permutation Importance)</div>
            <div class="card-header-desc">Mean decrease in test generalization Macro F1 when factor values are randomly permuted</div>
        """,
        unsafe_allow_html=True,
    )
    imp_df = get_cached_permutation_importance()
    fig_imp = plot_feature_importance_plotly(imp_df.head(6), title="")
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Platform Intelligence Workflow
    st.markdown(
        """
        <div class="content-card-saas">
            <div class="card-header-eyebrow">Executive Summary</div>
            <div class="card-header-title">Platform Intelligence Workflow</div>
            <div class="card-header-desc">How this supervised ML platform answers the 4 core business questions:</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem; margin-top: 1rem;">
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem;">
                    <div style="font-weight: 700; color: #0F2747; font-size: 0.9rem; margin-bottom: 0.3rem;">1. WHAT is customer satisfaction?</div>
                    <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5;">Supervised multiclass classification into <b>Satisfied</b>, <b>Neutral</b>, and <b>Dissatisfied</b> using the validated champion model.</div>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem;">
                    <div style="font-weight: 700; color: #0F2747; font-size: 0.9rem; margin-bottom: 0.3rem;">2. WHY did the model predict that?</div>
                    <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5;">Local explainability decomposes feature contributions pushing toward or against customer satisfaction.</div>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem;">
                    <div style="font-weight: 700; color: #0F2747; font-size: 0.9rem; margin-bottom: 0.3rem;">3. WHAT IF service factors improve?</div>
                    <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5;">Real-time Scenario Simulator evaluates the effect of operational adjustments on predicted class probabilities.</div>
                </div>
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1rem;">
                    <div style="font-weight: 700; color: #0F2747; font-size: 0.9rem; margin-bottom: 0.3rem;">4. WHERE should managers intervene?</div>
                    <div style="font-size: 0.82rem; color: #64748B; line-height: 1.5;">Customer Rescue Mode benchmarks 4 operational intervention packages to identify the highest-impact recovery path.</div>
                </div>
            </div>
            <div style="margin-top: 1.2rem; font-size: 0.76rem; color: #94A3B8; border-top: 1px solid #F1F5F9; padding-top: 0.8rem;">
                <b>Academic Disclosure:</b> The training dataset (1,500 records) is synthetic and generated strictly for academic evaluation within the IT503PC / IT505PC curriculum.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# PAGE 2: 📊 DATA ANALYSIS
# =====================================================================
elif menu_selection == "📊 Data Analysis":
    st.markdown(
        """
        <div class="saas-header-card">
            <div class="saas-header-left">
                <h1>Data Analysis & Service Insights</h1>
                <p>Exploratory Data Analysis and statistical relationships between service dimensions and satisfaction.</p>
                <div class="sub-tagline">
                    <span>11 Service Features</span>
                    <span>1,500 Customer Records</span>
                    <span>Balanced Distribution</span>
                </div>
            </div>
            <div>
                <span class="status-pill">
                    <span class="status-dot"></span>
                    SYNTHETIC COHORT
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    eda_tabs = st.tabs([
        "📋 Dataset Overview",
        "🎯 Satisfaction Distribution",
        "📈 Feature Analysis",
        "⚖️ Service Factors vs Satisfaction",
        "🔥 Correlation Matrix",
        "🌟 Global Feature Importance",
    ])

    # Tab 1: Dataset Overview
    with eda_tabs[0]:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Schema & Integrity</div>
                <div class="card-header-title">Dataset Overview & Summary Metrics</div>
                <div class="card-header-desc">Synthetic cohort structured for supervised multiclass customer satisfaction classification</div>
            """,
            unsafe_allow_html=True,
        )
        col_ov1, col_ov2, col_ov3, col_ov4 = st.columns(4)
        col_ov1.metric("Total Records", f"{len(df_raw):,}")
        col_ov2.metric("Total Features", len(FEATURE_COLUMNS))
        col_ov3.metric("Target Classes", len(VALID_TARGET_CLASSES))
        col_ov4.metric("Missing Cells", int(df_raw.isnull().sum().sum()))

        st.write("")
        st.markdown("**Data Preview (Top 10 records):**")
        st.dataframe(df_raw.head(10), use_container_width=True)

        st.write("")
        st.markdown("**Missing Value Breakdown:**")
        missing_df = df_raw.isnull().sum().reset_index()
        missing_df.columns = ["Column Name", "Missing Count"]
        missing_df["Missing Percentage"] = ((missing_df["Missing Count"] / len(df_raw)) * 100).round(2).astype(str) + "%"
        missing_df["Data Type"] = [str(df_raw[c].dtype) for c in missing_df["Column Name"]]
        st.dataframe(missing_df, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 2: Target Distribution
    with eda_tabs[1]:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Target Benchmark</div>
                <div class="card-header-title">Satisfaction Level Target Distribution</div>
                <div class="card-header-desc">Evenly distributed across 3 classes (500 records each), establishing a balanced benchmark</div>
            """,
            unsafe_allow_html=True,
        )
        col_t1, col_t2 = st.columns([1.5, 1])
        with col_t1:
            fig_t = plot_target_distribution_plotly(df_raw)
            st.plotly_chart(fig_t, use_container_width=True)
        with col_t2:
            st.markdown(
                """
                <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; margin-top: 1rem;">
                    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.6rem;">Class Breakdown</div>
                    <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #E2E8F0; font-size: 0.85rem;">
                        <span style="color: #0D9488; font-weight: 600;">● Satisfied</span>
                        <span style="font-weight: 700; color: #0F172A;">500 (33.33%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; border-bottom: 1px solid #E2E8F0; font-size: 0.85rem;">
                        <span style="color: #D97706; font-weight: 600;">● Neutral</span>
                        <span style="font-weight: 700; color: #0F172A;">500 (33.33%)</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; padding: 0.4rem 0; font-size: 0.85rem;">
                        <span style="color: #E11D48; font-weight: 600;">● Dissatisfied</span>
                        <span style="font-weight: 700; color: #0F172A;">500 (33.33%)</span>
                    </div>
                    <div style="margin-top: 1rem; font-size: 0.78rem; color: #64748B; line-height: 1.5; border-top: 1px solid #E2E8F0; padding-top: 0.8rem;">
                        <b>Academic Baseline:</b> A naive baseline classifier achieves exactly <b>33.33% Accuracy</b> and <b>0.1667 Macro F1</b>. Real ML models must significantly outperform this threshold.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 3: Feature Analysis
    with eda_tabs[2]:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Distribution Telemetry</div>
                <div class="card-header-title">Individual Feature Distributions</div>
                <div class="card-header-desc">Inspect statistical spread, outliers, and density across customer cohorts</div>
            """,
            unsafe_allow_html=True,
        )
        selected_feat = st.selectbox(
            "Select Numerical Feature to Inspect:",
            NUMERICAL_FEATURES,
            index=0,
        )
        fig_num = plot_numerical_distributions_plotly(df_raw, selected_feat)
        st.plotly_chart(fig_num, use_container_width=True)

        col_f1, col_f2, col_f3 = st.columns(3)
        col_f1.metric("Minimum Value", f"{df_raw[selected_feat].min():.1f}")
        col_f2.metric("Mean Value", f"{df_raw[selected_feat].mean():.2f}")
        col_f3.metric("Maximum Value", f"{df_raw[selected_feat].max():.1f}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 4: Service Factors vs Satisfaction
    with eda_tabs[3]:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Comparative Inspection</div>
                <div class="card-header-title">Service Factor Relationships with Satisfaction</div>
                <div class="card-header-desc">Observed patterns indicate model-relevant associations without establishing causal guarantees</div>
            """,
            unsafe_allow_html=True,
        )
        selected_factor = st.selectbox(
            "Select Service Factor to Compare Across Classes:",
            NUMERICAL_FEATURES,
            index=0,
        )
        fig_fac = plot_service_factor_vs_target_plotly(df_raw, selected_factor)
        st.plotly_chart(fig_fac, use_container_width=True)

        st.markdown("**Mean Numerical Values Grouped by Satisfaction Level:**")
        summary_table = get_service_factor_summary(df_raw)
        st.dataframe(summary_table, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 5: Correlation Matrix
    with eda_tabs[4]:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Collinearity Analysis</div>
                <div class="card-header-title">Pearson Correlation Matrix (Numerical Factors)</div>
                <div class="card-header-desc">Pairwise linear correlation coefficients across all numerical service dimensions</div>
            """,
            unsafe_allow_html=True,
        )
        fig_corr = plot_correlation_heatmap_plotly(df_raw)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Tab 6: Global Feature Importance
    with eda_tabs[5]:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Global Attribution</div>
                <div class="card-header-title">Global Permutation Feature Importance</div>
                <div class="card-header-desc">Permutation importance evaluates the drop in Macro F1 on held-out test data when features are shuffled</div>
            """,
            unsafe_allow_html=True,
        )
        imp_df_all = get_cached_permutation_importance()
        fig_all_imp = plot_feature_importance_plotly(imp_df_all, title="")
        st.plotly_chart(fig_all_imp, use_container_width=True)
        st.dataframe(imp_df_all, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# PAGE 3: 🤖 MODEL PERFORMANCE
# =====================================================================
elif menu_selection == "🤖 Model Performance":
    st.markdown(
        """
        <div class="saas-header-card">
            <div class="saas-header-left">
                <h1>Model Performance & Evaluation</h1>
                <p>Stratified 5-Fold Cross-Validation on training data vs untouched held-out test set evaluation.</p>
                <div class="sub-tagline">
                    <span>5 Evaluated Algorithms</span>
                    <span>Metric: CV Macro F1</span>
                    <span>Generalization Diagnosis</span>
                </div>
            </div>
            <div>
                <span class="status-pill">
                    <span class="status-dot"></span>
                    EVALUATION BENCHMARK
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Champion Model Banner Card
    best_m_name = model_meta["best_model_name"]
    best_cv_f1 = model_meta["cv_scores"][best_m_name]["cv_macro_f1"]
    best_test_acc = model_meta["test_results"][best_m_name]["test_accuracy"]
    best_gap = model_meta["test_results"][best_m_name]["generalization_gap"]

    st.markdown(
        f"""
        <div class="content-card-saas" style="border-left: 4px solid #2563EB;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div class="card-header-eyebrow">Validated Champion Model</div>
                    <div style="font-size: 1.35rem; font-weight: 800; color: #0F2747;">{best_m_name}</div>
                    <div style="font-size: 0.85rem; color: #64748B; margin-top: 0.2rem;">Selected via highest mean CV Macro F1 on training set</div>
                </div>
                <div style="display: flex; gap: 1.5rem;">
                    <div>
                        <div style="font-size: 0.7rem; font-weight: 700; color: #64748B; text-transform: uppercase;">CV Macro F1</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #2563EB;">{best_cv_f1:.4f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Test Accuracy</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #0D9488;">{best_test_acc*100:.2f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.7rem; font-weight: 700; color: #64748B; text-transform: uppercase;">Generalization Gap</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #059669;">{best_gap*100:.2f}%</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Model comparison table
    st.markdown(
        """
        <div class="content-card-saas">
            <div class="card-header-eyebrow">Algorithm Matrix</div>
            <div class="card-header-title">Full Model Evaluation Benchmark</div>
            <div class="card-header-desc">Complete comparison across training, cross-validation, and held-out test set metrics</div>
        """,
        unsafe_allow_html=True,
    )
    comp_df = pd.DataFrame(model_meta["comparison_table"])

    def highlight_best(row):
        return ['background-color: #ECFDF5; font-weight: 700; color: #065F46;' if row["Is Best Model"] else '' for _ in row]

    st.dataframe(
        comp_df.style.apply(highlight_best, axis=1),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Visual Comparisons
    perf_col1, perf_col2 = st.columns(2)
    with perf_col1:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Test Metrics</div>
                <div class="card-header-title">Test Set Metrics Comparison</div>
                <div class="card-header-desc">Accuracy, Macro Precision, Recall, and F1 on untouched test data</div>
            """,
            unsafe_allow_html=True,
        )
        fig_metrics = go.Figure()
        for metric_name, color in [
            ("Test Accuracy", "#0F2747"),
            ("Test Precision", "#2563EB"),
            ("Test Recall", "#D97706"),
            ("Test Macro F1", "#0D9488"),
        ]:
            fig_metrics.add_trace(
                go.Bar(
                    name=metric_name,
                    x=comp_df["Model"],
                    y=comp_df[metric_name],
                    marker_color=color,
                    text=[f"{v:.3f}" for v in comp_df[metric_name]],
                    textposition="outside",
                    textfont=dict(size=10, family="Plus Jakarta Sans, Inter, sans-serif"),
                    width=0.2,
                )
            )
        fig_metrics.update_layout(
            barmode="group",
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
            height=360,
            margin=dict(l=20, r=20, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
            yaxis=dict(range=[0, 1.15], gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with perf_col2:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Diagnostics</div>
                <div class="card-header-title">Train vs Test Accuracy (Overfitting Diagnostic)</div>
                <div class="card-header-desc">Identifies algorithms suffering from high variance vs stable generalization</div>
            """,
            unsafe_allow_html=True,
        )
        fig_gap = go.Figure()
        fig_gap.add_trace(
            go.Bar(
                name="Train Accuracy",
                x=comp_df["Model"],
                y=comp_df["Train Accuracy"],
                marker_color="#93C5FD",
                text=[f"{v*100:.1f}%" for v in comp_df["Train Accuracy"]],
                textposition="outside",
                textfont=dict(size=10, color="#0F172A", family="Plus Jakarta Sans, Inter, sans-serif"),
                width=0.35,
            )
        )
        fig_gap.add_trace(
            go.Bar(
                name="Test Accuracy",
                x=comp_df["Model"],
                y=comp_df["Test Accuracy"],
                marker_color="#0F2747",
                text=[f"{v*100:.1f}%" for v in comp_df["Test Accuracy"]],
                textposition="outside",
                textfont=dict(size=10, color="#0F172A", family="Plus Jakarta Sans, Inter, sans-serif"),
                width=0.35,
            )
        )
        fig_gap.update_layout(
            barmode="group",
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
            height=360,
            margin=dict(l=20, r=20, t=10, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
            yaxis=dict(range=[0, 1.25], gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        )
        st.plotly_chart(fig_gap, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Confusion Matrix & Classification Report
    st.markdown(
        """
        <div class="content-card-saas">
            <div class="card-header-eyebrow">Detailed Model Inspection</div>
            <div class="card-header-title">Confusion Matrix & Classification Report</div>
            <div class="card-header-desc">Inspect per-class precision, recall, and false positive patterns across classes</div>
        """,
        unsafe_allow_html=True,
    )
    
    selected_eval_model = st.selectbox(
        "Select Model to Inspect:",
        list(model_meta["confusion_matrices"].keys()),
        index=1,  # Logistic Regression
    )

    cm_data = np.array(model_meta["confusion_matrices"][selected_eval_model])
    cr_data = model_meta["classification_reports"][selected_eval_model]

    cm_col, cr_col = st.columns([1, 1.2])

    with cm_col:
        st.markdown(f"**Confusion Matrix ({selected_eval_model})**")
        fig_cm = px.imshow(
            cm_data,
            labels=dict(x="Predicted Class", y="True Class", color="Customer Count"),
            x=VALID_TARGET_CLASSES,
            y=VALID_TARGET_CLASSES,
            text_auto=True,
            color_continuous_scale=[[0, "#F8FAFC"], [1, "#0D9488"]],
        )
        fig_cm.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
            height=340,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with cr_col:
        st.markdown(f"**Per-Class Classification Report ({selected_eval_model})**")
        cr_records = []
        for cls in VALID_TARGET_CLASSES:
            if cls in cr_data:
                cr_records.append({
                    "Class": cls,
                    "Precision": round(cr_data[cls]["precision"], 3),
                    "Recall": round(cr_data[cls]["recall"], 3),
                    "F1-Score": round(cr_data[cls]["f1-score"], 3),
                    "Support": int(cr_data[cls]["support"]),
                })
        cr_df = pd.DataFrame(cr_records)
        st.dataframe(cr_df, use_container_width=True)

        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.8rem; font-size: 0.82rem; color: #475569;">
                <div><b>Overall Accuracy:</b> <code>{cr_data.get('accuracy', 0):.3f}</code></div>
                <div><b>Macro Average F1:</b> <code>{cr_data.get('macro avg', {}).get('f1-score', 0):.3f}</code></div>
                <div><b>Weighted Average F1:</b> <code>{cr_data.get('weighted avg', {}).get('f1-score', 0):.3f}</code></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Best Model Selection Rationale
    st.markdown(
        f"""
        <div class="content-card-saas">
            <div class="card-header-eyebrow">Selection Rule</div>
            <div class="card-header-title">Champion Model Selection Rationale</div>
            <div style="font-size: 0.85rem; color: #334155; line-height: 1.6; margin-top: 0.5rem;">
                <p><b>Predefined Rule:</b> The champion model is selected using <b>Highest Mean Macro F1 from Stratified 5-Fold Cross-Validation</b> on training data (tie-breaker: Mean CV Accuracy).</p>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 0.8rem; margin: 1rem 0;">
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.8rem;">
                        <span style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Champion Model</span>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #0F2747;">{model_meta['best_model_name']}</div>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.8rem;">
                        <span style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Mean CV Macro F1</span>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #2563EB;">{model_meta['cv_scores'][model_meta['best_model_name']]['cv_macro_f1']:.4f}</div>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.8rem;">
                        <span style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Test Accuracy</span>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #0D9488;">{model_meta['test_results'][model_meta['best_model_name']]['test_accuracy']*100:.2f}%</div>
                    </div>
                    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.8rem;">
                        <span style="font-size: 0.72rem; color: #64748B; font-weight: 700; text-transform: uppercase;">Generalization Gap</span>
                        <div style="font-size: 1.1rem; font-weight: 800; color: #059669;">{model_meta['test_results'][model_meta['best_model_name']]['generalization_gap']*100:.2f}%</div>
                    </div>
                </div>
                <div style="background: #F1F5F9; border-left: 3px solid #2563EB; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.82rem; color: #334155; margin-top: 0.8rem;">
                    <b>Academic Viva Defense Note:</b> Complex non-linear models like KNN (Train 100%, Test 59.3%) and Random Forest (Train 99.5%, Test 61.0%) suffered from high variance and overfitting on this dataset. Logistic Regression provided a regularized decision boundary with only a 3.08% generalization gap between training and testing, achieving both the highest CV Macro F1 and the most stable test generalization.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# PAGE 4: 🔮 CUSTOMER PREDICTION
# =====================================================================
elif menu_selection == "🔮 Customer Prediction":
    st.markdown(
        """
        <div class="saas-header-card">
            <div class="saas-header-left">
                <h1>Customer Satisfaction Prediction</h1>
                <p>Predict satisfaction level for an individual customer using the validated best model with local factor explainability.</p>
                <div class="sub-tagline">
                    <span>Champion: Logistic Regression</span>
                    <span>SHAP Feature Attribution</span>
                    <span>Cross-Model Consensus</span>
                </div>
            </div>
            <div>
                <span class="status-pill">
                    <span class="status-dot"></span>
                    CHAMPION MODEL ACTIVE
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick presets for viva and demo ease
    st.markdown(
        """
        <div class="content-card-saas" style="padding: 1rem 1.4rem; margin-bottom: 1.2rem;">
            <div class="card-header-eyebrow">Quick Customer Presets</div>
            <div style="font-size: 0.82rem; color: #64748B; margin-bottom: 0.6rem;">Load predefined customer profiles for live demonstration and defense:</div>
        """,
        unsafe_allow_html=True,
    )
    p_col1, p_col2, p_col3 = st.columns(3)
    preset_choice = None
    if p_col1.button("✨ Load Satisfied Customer", use_container_width=True):
        preset_choice = "satisfied"
    if p_col2.button("⚖️ Load Neutral Customer", use_container_width=True):
        preset_choice = "neutral"
    if p_col3.button("⚠️ Load Dissatisfied Customer", use_container_width=True):
        preset_choice = "dissatisfied"
    st.markdown("</div>", unsafe_allow_html=True)

    # Default values based on preset
    defaults = {
        "Service_Type": "Hospitality",
        "Customer_Type": "Regular",
        "Visit_Frequency": "Frequent",
        "Service_Quality": 4.0,
        "Staff_Behaviour": 4.0,
        "Response_Time_Min": 10.0,
        "Waiting_Time_Min": 8.0,
        "Price_Value": 4.0,
        "Product_Quality": 4.0,
        "Complaint_Resolution": 4.0,
        "Ease_of_Process": 4.0,
    }

    if preset_choice == "satisfied":
        defaults.update({
            "Service_Quality": 5.0,
            "Staff_Behaviour": 5.0,
            "Response_Time_Min": 5.0,
            "Waiting_Time_Min": 5.0,
            "Price_Value": 4.5,
            "Product_Quality": 5.0,
            "Complaint_Resolution": 4.5,
            "Ease_of_Process": 5.0,
        })
    elif preset_choice == "neutral":
        defaults.update({
            "Service_Quality": 3.0,
            "Staff_Behaviour": 3.0,
            "Response_Time_Min": 18.0,
            "Waiting_Time_Min": 15.0,
            "Price_Value": 3.0,
            "Product_Quality": 3.0,
            "Complaint_Resolution": 3.0,
            "Ease_of_Process": 3.0,
        })
    elif preset_choice == "dissatisfied":
        defaults.update({
            "Service_Quality": 1.5,
            "Staff_Behaviour": 1.5,
            "Response_Time_Min": 32.0,
            "Waiting_Time_Min": 28.0,
            "Price_Value": 2.0,
            "Product_Quality": 2.0,
            "Complaint_Resolution": 1.5,
            "Ease_of_Process": 2.0,
        })

    # Input Form wrapped in SaaS card
    st.markdown(
        """
        <div class="content-card-saas">
            <div class="card-header-eyebrow">Customer Intake Form</div>
            <div class="card-header-title">Customer Service Experience Attributes</div>
            <div class="card-header-desc">Input demographic context, waiting times, and experience ratings (1 to 5)</div>
        """,
        unsafe_allow_html=True,
    )
    with st.form("customer_prediction_form"):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #0F2747; text-transform: uppercase; margin-bottom: 0.5rem;'>Customer Profile</div>", unsafe_allow_html=True)
            service_type = st.selectbox(
                "Service Type",
                feature_meta["categorical_categories"]["Service_Type"],
                index=feature_meta["categorical_categories"]["Service_Type"].index(defaults["Service_Type"]),
            )
            customer_type = st.selectbox(
                "Customer Type",
                feature_meta["categorical_categories"]["Customer_Type"],
                index=feature_meta["categorical_categories"]["Customer_Type"].index(defaults["Customer_Type"]),
            )
            visit_freq = st.selectbox(
                "Visit Frequency",
                feature_meta["categorical_categories"]["Visit_Frequency"],
                index=feature_meta["categorical_categories"]["Visit_Frequency"].index(defaults["Visit_Frequency"]),
            )

        with f_col2:
            st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #0F2747; text-transform: uppercase; margin-bottom: 0.5rem;'>Core Service & Timings</div>", unsafe_allow_html=True)
            serv_quality = st.slider("Service Quality (1-5)", 1.0, 5.0, float(defaults["Service_Quality"]), 0.5)
            staff_behav = st.slider("Staff Behaviour (1-5)", 1.0, 5.0, float(defaults["Staff_Behaviour"]), 0.5)
            resp_time = st.number_input("Response Time (minutes)", min_value=1.0, max_value=60.0, value=float(defaults["Response_Time_Min"]), step=1.0)
            wait_time = st.number_input("Waiting Time (minutes)", min_value=1.0, max_value=60.0, value=float(defaults["Waiting_Time_Min"]), step=1.0)

        with f_col3:
            st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #0F2747; text-transform: uppercase; margin-bottom: 0.5rem;'>Experience & Value</div>", unsafe_allow_html=True)
            price_val = st.slider("Price Value (1-5)", 1.0, 5.0, float(defaults["Price_Value"]), 0.5)
            prod_quality = st.slider("Product Quality (1-5)", 1.0, 5.0, float(defaults["Product_Quality"]), 0.5)
            comp_res = st.slider("Complaint Resolution (1-5)", 1.0, 5.0, float(defaults["Complaint_Resolution"]), 0.5)
            ease_proc = st.slider("Ease of Process (1-5)", 1.0, 5.0, float(defaults["Ease_of_Process"]), 0.5)

        submit_pred = st.form_submit_button("🎯 Predict Customer Satisfaction", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Process prediction
    if submit_pred or ("current_customer" in st.session_state and preset_choice):
        customer_record = pd.DataFrame([{
            "Service_Type": service_type,
            "Customer_Type": customer_type,
            "Visit_Frequency": visit_freq,
            "Service_Quality": serv_quality,
            "Staff_Behaviour": staff_behav,
            "Response_Time_Min": resp_time,
            "Waiting_Time_Min": wait_time,
            "Price_Value": price_val,
            "Product_Quality": prod_quality,
            "Complaint_Resolution": comp_res,
            "Ease_of_Process": ease_proc,
        }])
        st.session_state["current_customer"] = customer_record
    elif "current_customer" not in st.session_state:
        st.session_state["current_customer"] = pd.DataFrame([defaults])

    active_customer = st.session_state["current_customer"]

    # Run predictions
    pred_class = best_pipeline.predict(active_customer)[0]
    pred_probs = best_pipeline.predict_proba(active_customer)[0]
    classes = list(best_pipeline.classes_)

    # Results section
    res_col1, res_col2 = st.columns([1, 1.2])

    with res_col1:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Model Classification</div>
                <div class="card-header-title">Predicted Satisfaction Outcome</div>
            """,
            unsafe_allow_html=True,
        )
        if pred_class == "Satisfied":
            st.markdown('<div class="badge-satisfied">● SATISFIED</div>', unsafe_allow_html=True)
        elif pred_class == "Neutral":
            st.markdown('<div class="badge-neutral">● NEUTRAL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-dissatisfied">● DISSATISFIED</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown("<div style='font-size: 0.78rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 0.4rem;'>Predicted Class Probabilities</div>", unsafe_allow_html=True)
        prob_df = pd.DataFrame({
            "Class": classes,
            "Probability": pred_probs,
        })
        fig_prob = px.bar(
            prob_df,
            x="Class",
            y="Probability",
            color="Class",
            color_discrete_map=CLASS_COLOR_MAP,
            text=[f"{p*100:.1f}%" for p in pred_probs],
        )
        fig_prob.update_traces(
            textposition="outside",
            textfont=dict(size=11, color="#0F172A", family="Plus Jakarta Sans, Inter, sans-serif"),
            marker=dict(line=dict(width=0)),
            width=0.45,
        )
        fig_prob.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
            showlegend=False,
            height=260,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
            yaxis=dict(range=[0, 1.15], gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        )
        st.plotly_chart(fig_prob, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with res_col2:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Ensemble Verification</div>
                <div class="card-header-title">Multi-Algorithm Consensus</div>
            """,
            unsafe_allow_html=True,
        )
        consensus_preds = {}
        for m_name, pipe in all_pipelines.items():
            if m_name != "Dummy Baseline":
                consensus_preds[m_name] = pipe.predict(active_customer)[0]

        agree_count = sum(1 for p in consensus_preds.values() if p == pred_class)
        total_ml_models = len(consensus_preds)

        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.8rem 1rem; margin-bottom: 0.9rem; font-size: 0.85rem; color: #0F172A;">
                <b>Consensus Agreement:</b> <span style="color: #2563EB; font-weight: 700;">{agree_count} / {total_ml_models} algorithms</span> agree with <b>{pred_class}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        cons_records = [{"Algorithm": k, "Prediction": v, "Status": "Agrees" if v == pred_class else "Diverges"} for k, v in consensus_preds.items()]
        st.dataframe(pd.DataFrame(cons_records), use_container_width=True)

        st.markdown(
            """
            <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 0.5rem;">
                Official project prediction strictly adheres to the validated Best Model (Logistic Regression).
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Local explainability section
    st.markdown(
        """
        <div class="content-card-saas">
            <div class="card-header-eyebrow">Explainable AI (XAI)</div>
            <div class="card-header-title">Local Factor Attribution (Why did the model predict this?)</div>
            <div class="card-header-desc">Feature attribution analysis identifying which factors increased or decreased the likelihood of this prediction</div>
        """,
        unsafe_allow_html=True,
    )
    explanation = explain_single_prediction_shap(
        best_pipeline,
        active_customer,
        X_test_ref,
        predicted_class=pred_class,
    )
    fig_local = plot_local_explanation_plotly(explanation)
    st.plotly_chart(fig_local, use_container_width=True)

    st.write("")
    if st.button("🧪 Open What-If Scenario Simulator with this Customer", use_container_width=True):
        st.session_state["scenario_customer"] = active_customer
        st.info("Customer transferred to Scenario Simulator! Navigate to '🧪 Scenario Simulator' in the sidebar.")
    st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# PAGE 5: 🧪 WHAT-IF SCENARIO SIMULATOR
# =====================================================================
elif menu_selection == "🧪 Scenario Simulator":
    st.markdown(
        """
        <div class="saas-header-card">
            <div class="saas-header-left">
                <h1>What-If Satisfaction Simulator & Customer Rescue</h1>
                <p>Simulate service adjustments in real-time and benchmark targeted operational recovery strategies.</p>
                <div class="sub-tagline">
                    <span>Dynamic What-If Modeling</span>
                    <span>Class Probability Shifts</span>
                    <span>Customer Rescue Analysis</span>
                </div>
            </div>
            <div>
                <span class="status-pill">
                    <span class="status-dot"></span>
                    INTERACTIVE ENGINE
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Retrieve customer from state or create baseline
    if "scenario_customer" not in st.session_state:
        if "current_customer" in st.session_state:
            st.session_state["scenario_customer"] = st.session_state["current_customer"].copy()
        else:
            st.session_state["scenario_customer"] = pd.DataFrame([{
                "Service_Type": "Hospitality",
                "Customer_Type": "Regular",
                "Visit_Frequency": "Frequent",
                "Service_Quality": 2.5,
                "Staff_Behaviour": 2.5,
                "Response_Time_Min": 25.0,
                "Waiting_Time_Min": 22.0,
                "Price_Value": 2.5,
                "Product_Quality": 2.5,
                "Complaint_Resolution": 2.5,
                "Ease_of_Process": 2.5,
            }])

    base_cust = st.session_state["scenario_customer"].iloc[0]

    # Baseline Prediction
    base_pred = best_pipeline.predict(st.session_state["scenario_customer"])[0]
    base_probs = best_pipeline.predict_proba(st.session_state["scenario_customer"])[0]
    classes = list(best_pipeline.classes_)

    scen_col1, scen_col2 = st.columns(2)

    with scen_col1:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Current Baseline</div>
                <div class="card-header-title">Customer Starting State</div>
                <div class="card-header-desc">Initial recorded service inputs and model prediction</div>
            """,
            unsafe_allow_html=True,
        )
        st.write(f"- **Service Quality:** `{base_cust['Service_Quality']}` | **Staff Behaviour:** `{base_cust['Staff_Behaviour']}`")
        st.write(f"- **Response Time:** `{base_cust['Response_Time_Min']} min` | **Waiting Time:** `{base_cust['Waiting_Time_Min']} min`")
        st.write(f"- **Price Value:** `{base_cust['Price_Value']}` | **Complaint Resolution:** `{base_cust['Complaint_Resolution']}`")
        
        st.markdown("<div style='margin-top: 1rem; font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase;'>Current Model Outcome</div>", unsafe_allow_html=True)
        if base_pred == "Satisfied":
            st.markdown('<div class="badge-satisfied">● SATISFIED</div>', unsafe_allow_html=True)
        elif base_pred == "Neutral":
            st.markdown('<div class="badge-neutral">● NEUTRAL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-dissatisfied">● DISSATISFIED</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with scen_col2:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Interactive Simulation</div>
                <div class="card-header-title">Simulate Service Adjustments</div>
                <div class="card-header-desc">Move sliders to evaluate real-time impact on customer satisfaction</div>
            """,
            unsafe_allow_html=True,
        )
        sim_serv_qual = st.slider("Adjust Service Quality", 1.0, 5.0, float(base_cust["Service_Quality"]), 0.5)
        sim_staff_beh = st.slider("Adjust Staff Behaviour", 1.0, 5.0, float(base_cust["Staff_Behaviour"]), 0.5)
        sim_wait_time = st.slider("Adjust Waiting Time (min)", 1.0, 45.0, float(base_cust["Waiting_Time_Min"]), 1.0)
        sim_resp_time = st.slider("Adjust Response Time (min)", 1.0, 45.0, float(base_cust["Response_Time_Min"]), 1.0)
        sim_comp_res = st.slider("Adjust Complaint Resolution", 1.0, 5.0, float(base_cust["Complaint_Resolution"]), 0.5)
        st.markdown("</div>", unsafe_allow_html=True)

    # Simulated row
    sim_row = st.session_state["scenario_customer"].copy()
    sim_row["Service_Quality"] = sim_serv_qual
    sim_row["Staff_Behaviour"] = sim_staff_beh
    sim_row["Waiting_Time_Min"] = sim_wait_time
    sim_row["Response_Time_Min"] = sim_resp_time
    sim_row["Complaint_Resolution"] = sim_comp_res

    # Run model on simulated row
    sim_pred = best_pipeline.predict(sim_row)[0]
    sim_probs = best_pipeline.predict_proba(sim_row)[0]

    # Comparison section
    comp_col1, comp_col2 = st.columns([1, 1.2])

    with comp_col1:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Simulated Outcome</div>
                <div class="card-header-title">Model Response to Adjustments</div>
            """,
            unsafe_allow_html=True,
        )
        if sim_pred == "Satisfied":
            st.markdown('<div class="badge-satisfied">● SATISFIED</div>', unsafe_allow_html=True)
        elif sim_pred == "Neutral":
            st.markdown('<div class="badge-neutral">● NEUTRAL</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-dissatisfied">● DISSATISFIED</div>', unsafe_allow_html=True)

        st.write("")
        if base_pred != sim_pred:
            st.success(f"Outcome shifted from **{base_pred}** to **{sim_pred}** under modeled adjustments!")
        else:
            st.info(f"Outcome remains **{sim_pred}**. Adjust factors further to cross decision boundaries.")
        st.markdown("</div>", unsafe_allow_html=True)

    with comp_col2:
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-header-eyebrow">Probability Telemetry</div>
                <div class="card-header-title">Before vs. After Class Probabilities</div>
            """,
            unsafe_allow_html=True,
        )
        prob_comp_df = pd.DataFrame({
            "Class": classes * 2,
            "Probability": list(base_probs) + list(sim_probs),
            "Scenario": ["Before (Baseline)"] * 3 + ["After (Simulated)"] * 3,
        })
        fig_prob_comp = px.bar(
            prob_comp_df,
            x="Class",
            y="Probability",
            color="Scenario",
            barmode="group",
            color_discrete_map={"Before (Baseline)": "#94A3B8", "After (Simulated)": "#0D9488"},
            text=[f"{p*100:.1f}%" for p in prob_comp_df["Probability"]],
        )
        fig_prob_comp.update_traces(
            textposition="outside",
            textfont=dict(size=10, color="#0F172A", family="Plus Jakarta Sans, Inter, sans-serif"),
            width=0.35,
        )
        fig_prob_comp.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
            height=280,
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
            yaxis=dict(range=[0, 1.15], gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        )
        st.plotly_chart(fig_prob_comp, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Customer Rescue Mode
    st.markdown(
        """
        <div class="content-card-saas">
            <div class="card-header-eyebrow">Managerial Action</div>
            <div class="card-header-title">Customer Rescue Strategy Analysis</div>
            <div class="card-header-desc">Benchmarking standard operational intervention packages to identify the highest modeled recovery probability:</div>
        """,
        unsafe_allow_html=True,
    )
    rescue_scenarios = [
        ("Speed Up Service", {"Waiting_Time_Min": max(3.0, base_cust["Waiting_Time_Min"] * 0.4), "Response_Time_Min": max(5.0, base_cust["Response_Time_Min"] * 0.5)}),
        ("Train Staff / Hospitality", {"Staff_Behaviour": min(5.0, base_cust["Staff_Behaviour"] + 2.0), "Service_Quality": min(5.0, base_cust["Service_Quality"] + 1.5)}),
        ("Prioritize Complaint Resolution", {"Complaint_Resolution": 5.0, "Ease_of_Process": min(5.0, base_cust["Ease_of_Process"] + 1.5)}),
        ("Comprehensive Service Overhaul", {"Service_Quality": 4.5, "Staff_Behaviour": 4.5, "Waiting_Time_Min": 6.0, "Response_Time_Min": 7.0, "Complaint_Resolution": 4.5}),
    ]

    rescue_results = []
    for s_name, adjustments in rescue_scenarios:
        test_r = st.session_state["scenario_customer"].copy()
        for k, v in adjustments.items():
            test_r[k] = v
        r_pred = best_pipeline.predict(test_r)[0]
        r_probs = best_pipeline.predict_proba(test_r)[0]
        sat_prob = r_probs[classes.index("Satisfied")]

        rescue_results.append({
            "Intervention Package": s_name,
            "Modeled Outcome": r_pred,
            "Satisfied Probability": f"{sat_prob*100:.1f}%",
            "Adjustments Made": ", ".join([f"{k}: {v}" for k, v in adjustments.items()]),
        })

    rescue_df = pd.DataFrame(rescue_results)
    st.dataframe(rescue_df, use_container_width=True)
    st.markdown(
        """
        <div style="font-size: 0.74rem; color: #94A3B8; margin-top: 0.6rem;">
            Notice: Model-based scenario analysis represents algorithmic decision boundaries and does not constitute guaranteed operational advice.
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# PAGE 6: 📂 UPLOAD & BATCH PREDICTION
# =====================================================================
elif menu_selection == "📂 Upload & Batch Prediction":
    st.markdown(
        """
        <div class="saas-header-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div class="saas-header-title">Upload Restaurant Data</div>
                    <div class="saas-header-subtitle">Analyze and predict satisfaction for a compatible customer dataset.</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 500; margin-top: 0.25rem;">Automated Schema Validation • Batch Inference Pipeline • External Evaluation</div>
                </div>
                <div class="saas-status-badge">
                    <span class="saas-status-dot"></span>
                    BATCH ENGINE READY
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    template_path = "data/prediction_template.csv"
    template_bytes = None
    if os.path.exists(template_path):
        with open(template_path, "rb") as f:
            template_bytes = f.read()

    # Upload Section Card
    st.markdown(
        """
        <div class="content-card-saas">
            <div class="card-eyebrow">BATCH INGESTION</div>
            <div class="card-title">Customer Dataset Upload</div>
            <div class="card-desc">Upload a CSV dataset containing customer experience ratings. Use our standard template to guarantee schema compatibility.</div>
        """,
        unsafe_allow_html=True,
    )

    up_top1, up_top2 = st.columns([1.5, 1])
    with up_top1:
        uploaded_file = st.file_uploader(
            "Select CSV file for batch inference:",
            type=["csv"],
            key="batch_file_uploader",
        )
    with up_top2:
        st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
        if template_bytes:
            st.download_button(
                label="📥 Download Template CSV",
                data=template_bytes,
                file_name="customer_prediction_template.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            up_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        missing_required = [c for c in FEATURE_COLUMNS if c not in up_df.columns]
        has_target = TARGET_COLUMN in up_df.columns
        has_id = "Customer_ID" in up_df.columns

        # Dataset Health Card
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-eyebrow">VERIFICATION</div>
                <div class="card-title">Dataset Health & Compatibility</div>
                <div class="card-desc">Validating schema integrity, required attributes, and numerical boundaries against the Champion Model pipeline.</div>
            """,
            unsafe_allow_html=True,
        )

        h_col1, h_col2, h_col3, h_col4 = st.columns(4)
        h_col1.metric("Rows (Records)", f"{len(up_df):,}")
        h_col2.metric("Columns", f"{len(up_df.columns)}")
        h_col3.metric("Missing Required", len(missing_required))
        h_col4.metric(
            "Compatibility",
            "Verified" if not missing_required else "Incompatible",
        )

        if missing_required:
            st.error(
                f"❌ **Incompatible Dataset**: Prediction stopped because required feature meanings are missing: `{missing_required}`. "
                "The model only supports datasets with compatible service features. Please download the template above."
            )
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()
        else:
            st.success("✅ **Schema Verified**: All 11 required predictive features are present and compatible with the preprocessing pipeline!")

        st.markdown(
            """
            <div style="font-size: 0.85rem; font-weight: 700; color: #0F2747; margin-top: 1rem; margin-bottom: 0.5rem;">
                Sample Preview (First 5 Rows):
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(up_df.head(5), use_container_width=True)

        warnings = []
        for feat in NUMERICAL_FEATURES:
            if "Time" not in feat:
                if (up_df[feat].dropna() < 1).any() or (up_df[feat].dropna() > 5).any():
                    warnings.append(f"'{feat}' contains values outside the standard 1-5 rating range.")
            else:
                if (up_df[feat].dropna() < 0).any():
                    warnings.append(f"'{feat}' contains negative time values.")

        if warnings:
            st.warning("⚠️ **Data Quality Warnings:**\n" + "\n".join([f"- {w}" for w in warnings]))

        st.markdown("</div>", unsafe_allow_html=True)

        # Batch Prediction Section
        st.markdown(
            """
            <div class="content-card-saas">
                <div class="card-eyebrow">MODEL INFERENCE</div>
                <div class="card-title">Run Batch Prediction</div>
                <div class="card-desc">Execute multiclass classification across all uploaded records using the Champion Logistic Regression pipeline.</div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🚀 Execute Batch Predictions", use_container_width=True):
            with st.spinner("Processing through ColumnTransformer pipeline and generating predictions..."):
                X_batch = up_df[FEATURE_COLUMNS]
                preds = best_pipeline.predict(X_batch)
                probs = best_pipeline.predict_proba(X_batch)
                classes = list(best_pipeline.classes_)

                results_df = up_df.copy()
                results_df["Predicted_Satisfaction"] = preds
                for i, cls in enumerate(classes):
                    results_df[f"Probability_{cls}"] = probs[:, i].round(3)

                st.session_state["batch_results"] = results_df

        st.markdown("</div>", unsafe_allow_html=True)

        if "batch_results" in st.session_state:
            res_df = st.session_state["batch_results"]

            st.markdown(
                """
                <div class="content-card-saas">
                    <div class="card-eyebrow">PREDICTION RESULTS</div>
                    <div class="card-title">Batch Prediction Summary & Export</div>
                    <div class="card-desc">Class breakdown and predicted probability distributions for the uploaded cohort.</div>
                """,
                unsafe_allow_html=True,
            )

            b_counts = res_df["Predicted_Satisfaction"].value_counts().reindex(VALID_TARGET_CLASSES, fill_value=0).reset_index()
            b_counts.columns = ["Satisfaction Level", "Count"]

            b_col1, b_col2 = st.columns([1.3, 1])
            with b_col1:
                fig_b = px.bar(
                    b_counts,
                    x="Satisfaction Level",
                    y="Count",
                    color="Satisfaction Level",
                    color_discrete_map=CLASS_COLOR_MAP,
                    text="Count",
                )
                fig_b.update_traces(
                    textposition="outside",
                    textfont=dict(size=12, color="#0F2747", family="Plus Jakarta Sans, Inter, sans-serif"),
                    marker=dict(line=dict(width=1, color="#FFFFFF")),
                )
                fig_b.update_layout(
                    template="plotly_white",
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#0F2747"),
                    height=290,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                    xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#F1F5F9"),
                    yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#F1F5F9"),
                )
                st.plotly_chart(fig_b, use_container_width=True)

            with b_col2:
                st.markdown(
                    """
                    <div style="font-size: 0.85rem; font-weight: 700; color: #0F2747; margin-bottom: 0.75rem;">
                        Cohort Breakdown
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                for _, row in b_counts.iterrows():
                    pct = (row["Count"] / len(res_df)) * 100 if len(res_df) > 0 else 0
                    color = CLASS_COLOR_MAP.get(row['Satisfaction Level'], '#0F2747')
                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0.6rem; margin-bottom: 0.4rem; background: #F8FAFC; border-radius: 8px; border: 1px solid #E2E8F0;">
                            <div style="display: flex; align-items: center; gap: 0.5rem; font-size: 0.82rem; font-weight: 600; color: #0F2747;">
                                <span style="width: 8px; height: 8px; border-radius: 50%; background: {color}; display: inline-block;"></span>
                                {row['Satisfaction Level']}
                            </div>
                            <div style="font-size: 0.82rem; font-weight: 700; color: #475569;">
                                {row['Count']} <span style="font-size: 0.75rem; color: #94A3B8; font-weight: 500;">({pct:.1f}%)</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv_bytes,
                    file_name="customer_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown(
                """
                <div style="font-size: 0.85rem; font-weight: 700; color: #0F2747; margin-top: 1.25rem; margin-bottom: 0.5rem;">
                    Predictions Table (First 20 Records):
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(res_df.head(20), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # External Labeled Evaluation
            if has_target:
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

                y_true = up_df[TARGET_COLUMN]
                y_pred = res_df["Predicted_Satisfaction"]

                ext_acc = accuracy_score(y_true, y_pred)
                ext_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
                ext_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
                ext_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

                st.markdown(
                    """
                    <div class="content-card-saas">
                        <div class="card-eyebrow">EXTERNAL VALIDATION</div>
                        <div class="card-title">External Labeled Dataset Evaluation</div>
                        <div class="card-desc">Since ground-truth labels (Satisfaction_Level) were supplied, performance metrics are computed directly against external real-world distribution.</div>
                    """,
                    unsafe_allow_html=True,
                )

                ecol1, ecol2, ecol3, ecol4 = st.columns(4)
                ecol1.metric("External Accuracy", f"{ext_acc*100:.2f}%")
                ecol2.metric("External Precision", f"{ext_prec:.3f}")
                ecol3.metric("External Recall", f"{ext_rec:.3f}")
                ecol4.metric("External Macro F1", f"{ext_f1:.3f}")

                ext_cm = confusion_matrix(y_true, y_pred, labels=VALID_TARGET_CLASSES)
                fig_ext_cm = px.imshow(
                    ext_cm,
                    labels=dict(x="Predicted Class", y="Ground Truth Class"),
                    x=VALID_TARGET_CLASSES,
                    y=VALID_TARGET_CLASSES,
                    text_auto=True,
                    color_continuous_scale="Tealgrn",
                )
                fig_ext_cm.update_layout(
                    template="plotly_white",
                    paper_bgcolor="#FFFFFF",
                    plot_bgcolor="#FFFFFF",
                    font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#0F2747"),
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_ext_cm, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


# =====================================================================
# PAGE 7: ℹ️ ABOUT PROJECT
# =====================================================================
elif menu_selection == "ℹ️ About Project":
    st.markdown(
        """
        <div class="saas-header-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div class="saas-header-title">About the Project</div>
                    <div class="saas-header-subtitle">Customer Satisfaction Intelligence System • Project Documentation & Technical Architecture</div>
                    <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 500; margin-top: 0.25rem;">IT503PC Machine Learning | IT505PC ML Lab | B.Tech Academic Project</div>
                </div>
                <div class="saas-status-badge">
                    <span class="saas-status-dot"></span>
                    SYSTEM SPECIFICATION
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("📌 1. Project Overview & Objective", expanded=True):
        st.markdown(
            """
            <div style="font-size: 0.9rem; color: #334155; line-height: 1.6;">
                In competitive service environments such as restaurants, retail, and hospitality, customer retention depends on the proactive identification of service friction and actionable recovery workflows.<br><br>
                The <strong>Customer Satisfaction Intelligence System</strong> addresses this challenge through end-to-end Machine Learning:
                <ul style="margin-top: 0.5rem; margin-bottom: 0.5rem;">
                    <li><strong>Multiclass Classification:</strong> Accurately categorizes customer perception into <strong>Satisfied</strong>, <strong>Neutral</strong>, and <strong>Dissatisfied</strong>.</li>
                    <li><strong>Transparent Explainability:</strong> Exposes global feature importance and individual prediction attribution without black-box opacity.</li>
                    <li><strong>Interactive Simulation:</strong> Enables managers to run what-if simulations to calculate the ROI of operational improvements.</li>
                    <li><strong>Batch Validation:</strong> Offers automated schema verification and inference for multi-branch datasets.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🤖 2. Machine Learning Algorithms Implemented", expanded=False):
        st.markdown(
            """
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                <ol style="margin-left: 1.2rem; padding-left: 0;">
                    <li style="margin-bottom: 0.6rem;"><strong>Dummy Classifier (Baseline):</strong> Predicts the most frequent class to establish a minimal benchmark (33.33% accuracy on balanced 3-class distribution).</li>
                    <li style="margin-bottom: 0.6rem;"><strong>Logistic Regression:</strong> Multinomial linear classifier trained with L2 regularization (C=0.1). Selected as Champion Model due to optimal test generalization and highest Cross-Validation Macro F1 (0.6388).</li>
                    <li style="margin-bottom: 0.6rem;"><strong>Decision Tree Classifier:</strong> Non-linear decision rules tuned across depth limits and split criteria to expose interpretable decision paths.</li>
                    <li style="margin-bottom: 0.6rem;"><strong>K-Nearest Neighbors (KNN):</strong> Distance-based instance learning evaluated across neighbor counts (K=11) and distance metrics.</li>
                    <li style="margin-bottom: 0.6rem;"><strong>Random Forest Classifier:</strong> Ensemble of 150 bagging decision trees evaluated with bootstrap aggregation.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("⚙️ 3. Preprocessing & Validation Integrity", expanded=False):
        st.markdown(
            """
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                <ul style="margin-left: 1.2rem; padding-left: 0;">
                    <li style="margin-bottom: 0.5rem;"><strong>Pipeline Architecture:</strong> Built using <code>sklearn.compose.ColumnTransformer</code> inside end-to-end <code>Pipeline</code> objects fitted strictly on the training partition to eliminate data leakage.</li>
                    <li style="margin-bottom: 0.5rem;"><strong>Numerical Pipeline:</strong> Imputed via median and normalized with <code>StandardScaler</code>.</li>
                    <li style="margin-bottom: 0.5rem;"><strong>Categorical Pipeline:</strong> Imputed via most frequent and encoded with <code>OneHotEncoder(handle_unknown='ignore')</code>.</li>
                    <li style="margin-bottom: 0.5rem;"><strong>Partitioning:</strong> 80/20 train/test split with stratified sampling (<code>random_state=42</code>). Stratified 5-Fold Cross-Validation was performed exclusively on the training set for hyperparameter tuning.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🏆 4. Champion Model Selection Rule", expanded=False):
        st.markdown(
            """
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                <p>Model selection followed strict academic evaluation criteria:</p>
                <ul style="margin-left: 1.2rem; padding-left: 0;">
                    <li><strong>Primary Metric:</strong> Mean 5-Fold Cross-Validation Macro F1-Score on the training set. Macro F1 treats all three classes with equal importance, preventing majority-class bias.</li>
                    <li><strong>Tie-Breaker:</strong> Mean Cross-Validation Accuracy.</li>
                    <li><strong>Champion Selection:</strong> <strong>Logistic Regression</strong> achieved the highest Cross-Validation Macro F1 (<strong>0.6388</strong>) and generalized cleanly on the unseen test set (<strong>62.67% Test Accuracy, 0.6270 Test Macro F1</strong>).</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("⚠️ 5. Project Limitations & Ethical Disclosures", expanded=False):
        st.markdown(
            """
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                <ul style="margin-left: 1.2rem; padding-left: 0;">
                    <li style="margin-bottom: 0.5rem;"><strong>Synthetic Cohort:</strong> The primary dataset of 1,500 records is synthetically synthesized for academic evaluation. Production deployments must calibrate on verified in-situ guest responses.</li>
                    <li style="margin-bottom: 0.5rem;"><strong>Correlation vs Causation:</strong> Model feature attributions indicate statistical association within the training manifold, not guaranteed operational causality.</li>
                    <li style="margin-bottom: 0.5rem;"><strong>Schema Guardrails:</strong> Input features must strictly match the designated semantic schema; arbitrary customer surveys without matching ratings are prevented from producing corrupted inferences.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🔮 6. Future Scope & Roadmap", expanded=False):
        st.markdown(
            """
            <div style="font-size: 0.88rem; color: #334155; line-height: 1.6;">
                <ul style="margin-left: 1.2rem; padding-left: 0;">
                    <li style="margin-bottom: 0.5rem;">Real-time API webhooks for integration with Point of Sale (POS) and table tablet feedback kiosks.</li>
                    <li style="margin-bottom: 0.5rem;">Multi-modal NLP sentiment analysis on free-form diner reviews using distilled Transformer encoders.</li>
                    <li style="margin-bottom: 0.5rem;">Cloud-native automated MLOps pipelines with data drift detection and dynamic active learning loops.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
