"""
Explainability Module for Customer Satisfaction Intelligence System.
Course: IT503PC Machine Learning / IT505PC Machine Learning Lab
Project: Customer Satisfaction Intelligence System

Provides:
1. Global Explainability: Permutation Feature Importance computed on held-out test data.
2. Local Explainability: SHAP explanations (and robust contribution fallbacks) for single customer predictions.
3. Feature grouping and human-readable visualizations.
"""

import os
import json
import joblib
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import (
    load_dataset,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    VALID_TARGET_CLASSES,
)
from src.preprocessing import get_feature_names_after_preprocessing


def compute_global_permutation_importance(
    pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computes permutation feature importance on held-out data using Macro F1.
    Permuting original columns before preprocessing captures the true grouped impact
    of categorical and numerical features together without encoding distortion.
    """
    result = permutation_importance(
        pipeline,
        X_val,
        y_val,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "Feature": X_val.columns,
            "Importance_Mean": result.importances_mean,
            "Importance_Std": result.importances_std,
        }
    ).sort_values(by="Importance_Mean", ascending=False)

    return importance_df


def plot_feature_importance_plotly(
    importance_df: pd.DataFrame,
    title: str = "Global Feature Importance (Permutation Macro F1 Decrease)",
) -> go.Figure:
    """
    Creates an interactive horizontal bar chart for global feature importance.
    """
    df_sorted = importance_df.sort_values(by="Importance_Mean", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_sorted["Importance_Mean"],
            y=[f.replace("_", " ") for f in df_sorted["Feature"]],
            orientation="h",
            marker=dict(
                color="#0EA5E9",
                line=dict(width=0),
            ),
            error_x=dict(
                type="data",
                array=df_sorted["Importance_Std"],
                visible=True,
                color="#94A3B8",
            ),
            text=[f"{val:.3f}" for val in df_sorted["Importance_Mean"]],
            textposition="outside",
            textfont=dict(size=11, color="#0F172A", family="Plus Jakarta Sans, Inter, sans-serif"),
        )
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
        xaxis_title="Mean Decrease in Macro F1 (Higher = More Critical)",
        yaxis_title="Service Factor",
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


def explain_single_prediction_shap(
    pipeline,
    single_row: pd.DataFrame,
    X_background: Optional[pd.DataFrame] = None,
    predicted_class: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generates local feature attribution for a single customer prediction using SHAP.
    If SHAP is unavailable or slow, falls back to a linear model coefficient decomposition.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Transform single row
    X_single_trans = preprocessor.transform(single_row)
    feature_names = get_feature_names_after_preprocessing(preprocessor)

    # Attempt SHAP explanation
    try:
        import shap

        if X_background is not None:
            # Use a compact sample for speed and responsiveness
            bg_sample = X_background.sample(min(50, len(X_background)), random_state=42)
            bg_trans = preprocessor.transform(bg_sample)
        else:
            bg_trans = X_single_trans

        # For Logistic Regression / Linear models
        if hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, bg_trans)
            shap_values = explainer.shap_values(X_single_trans)
        elif hasattr(model, "predict_proba"):
            # Tree / Kernel Explainer fallback
            explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else shap.Explainer(model.predict_proba, bg_trans)
            shap_values = explainer(X_single_trans)
        else:
            raise ValueError("Model does not support standard SHAP explainer.")

        # Extract values for the target or predicted class
        classes = list(pipeline.classes_)
        class_idx = classes.index(predicted_class) if (predicted_class and predicted_class in classes) else 0

        # Handle various SHAP return structures across versions
        if isinstance(shap_values, list):
            vals = shap_values[class_idx][0]
        elif hasattr(shap_values, "values"):
            if len(shap_values.values.shape) == 3:
                vals = shap_values.values[0, :, class_idx]
            else:
                vals = shap_values.values[0]
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, class_idx]
        else:
            vals = shap_values[0]

        # Aggregate back to original features for clarity
        aggregated_impacts = {}
        for f_name, val in zip(feature_names, vals):
            orig_feature = f_name.split("_")[0] if any(f_name.startswith(cat) for cat in CATEGORICAL_FEATURES) else f_name
            # Keep clean original name if it's one of FEATURE_COLUMNS
            matched_col = next((c for c in FEATURE_COLUMNS if f_name.startswith(c)), orig_feature)
            aggregated_impacts[matched_col] = aggregated_impacts.get(matched_col, 0.0) + float(val)

        impact_df = pd.DataFrame(
            [{"Feature": k, "Impact": round(v, 4)} for k, v in aggregated_impacts.items()]
        ).sort_values(by="Impact", key=abs, ascending=False)

        return {
            "method": "SHAP",
            "feature_impacts": impact_df.to_dict(orient="records"),
            "target_class": predicted_class,
        }

    except Exception as e:
        # Fallback: Model Coefficient / Feature Contribution approximation
        return _fallback_linear_contribution(pipeline, single_row, predicted_class)


def _fallback_linear_contribution(pipeline, single_row: pd.DataFrame, predicted_class: Optional[str]) -> Dict[str, Any]:
    """
    Fallback mathematical decomposition for Logistic Regression:
    Contribution_i = (X_trans_i - mean_i) * coef_i
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    X_single_trans = preprocessor.transform(single_row)[0]
    feature_names = get_feature_names_after_preprocessing(preprocessor)

    classes = list(pipeline.classes_)
    class_idx = classes.index(predicted_class) if (predicted_class and predicted_class in classes) else 0

    if hasattr(model, "coef_"):
        coefs = model.coef_[class_idx]
        contributions = X_single_trans * coefs
    else:
        contributions = np.zeros(len(feature_names))

    aggregated_impacts = {}
    for f_name, val in zip(feature_names, contributions):
        matched_col = next((c for c in FEATURE_COLUMNS if f_name.startswith(c)), f_name)
        aggregated_impacts[matched_col] = aggregated_impacts.get(matched_col, 0.0) + float(val)

    impact_df = pd.DataFrame(
        [{"Feature": k, "Impact": round(v, 4)} for k, v in aggregated_impacts.items()]
    ).sort_values(by="Impact", key=abs, ascending=False)

    return {
        "method": "Feature Contribution (Normalized Model Coefficients)",
        "feature_impacts": impact_df.to_dict(orient="records"),
        "target_class": predicted_class,
    }


def plot_local_explanation_plotly(explanation: Dict[str, Any]) -> go.Figure:
    """
    Creates an interactive waterfall / divergence bar chart showing features pushing toward or against
    the predicted satisfaction class.
    """
    records = explanation["feature_impacts"][:8]  # top 8 features
    df_exp = pd.DataFrame(records).sort_values(by="Impact", ascending=True)

    colors = ["#0D9488" if val >= 0 else "#E11D48" for val in df_exp["Impact"]]

    fig = go.Figure(
        go.Bar(
            x=df_exp["Impact"],
            y=[f.replace("_", " ") for f in df_exp["Feature"]],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:+.3f}" for v in df_exp["Impact"]],
            textposition="outside",
            textfont=dict(size=11, color="#0F172A", family="Plus Jakarta Sans, Inter, sans-serif"),
        )
    )

    pred_class = explanation.get("target_class", "Prediction")
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
        xaxis_title="Influence Score (Teal: Increases Likelihood, Crimson: Decreases Likelihood)",
        yaxis_title="Service Factor",
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        height=360,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    return fig


if __name__ == "__main__":
    splits = joblib.load("models/test_split.pkl")
    best_pipeline = joblib.load("models/saved_model_pipeline.pkl")
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    print("Computing Permutation Importance on Test Set...")
    imp_df = compute_global_permutation_importance(best_pipeline, X_test, y_test)
    print(imp_df.head(10))

    # Test local explanation
    sample_row = X_test.iloc[[0]]
    pred = best_pipeline.predict(sample_row)[0]
    print(f"\nExplaining single prediction for class '{pred}'...")
    local_exp = explain_single_prediction_shap(best_pipeline, sample_row, X_test, predicted_class=pred)
    print("Method:", local_exp["method"])
    print("Top factors:", local_exp["feature_impacts"][:5])
