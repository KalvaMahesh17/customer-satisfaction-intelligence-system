"""
Exploratory Data Analysis and Service Analysis Module.
Course: IT503PC Machine Learning / IT505PC Machine Learning Lab
Project: Customer Satisfaction Intelligence System

Computes real statistical summaries and generates clean, professional visualizations
for service factor relationships, customer satisfaction distributions, and correlations.
"""

import os
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import (
    load_dataset,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    VALID_TARGET_CLASSES,
)

# Futuristic illuminated color palette inspired by LUMA SPACE
# Clean enterprise SaaS palette matching GeneX reference
CLASS_COLOR_MAP = {
    "Dissatisfied": "#E11D48",  # Sophisticated coral / crimson
    "Neutral": "#D97706",       # Refined amber
    "Satisfied": "#0D9488",     # Clean enterprise teal / emerald
}
PRIMARY_COLOR = "#0F172A"
CYAN_ACCENT = "#0EA5E9"
TEAL_ACCENT = "#0D9488"


def get_dataset_overview_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary overview statistics for the dataset overview card.
    """
    return {
        "total_records": len(df),
        "total_features": len(df.columns) - 1,  # excluding target
        "missing_records": int(df.isnull().any(axis=1).sum()),
        "missing_cells": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_counts": df[TARGET_COLUMN].value_counts().to_dict(),
        "target_proportions": (df[TARGET_COLUMN].value_counts(normalize=True) * 100).round(2).to_dict(),
    }


def get_service_factor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes mean ratings and service times grouped by Satisfaction Level.
    Uses careful language: 'Associated with', 'Observed average'.
    """
    summary = df.groupby(TARGET_COLUMN)[NUMERICAL_FEATURES].mean().loc[VALID_TARGET_CLASSES]
    summary = summary.round(2)
    return summary


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Pearson correlation matrix for numerical service features.
    """
    return df[NUMERICAL_FEATURES].corr().round(2)


# =====================================================================
# PLOTLY INTERACTIVE FIGURE GENERATORS (FOR STREAMLIT UI)
# =====================================================================

def plot_target_distribution_plotly(df: pd.DataFrame) -> go.Figure:
    """
    Interactive Bar Chart showing customer satisfaction distribution with clean enterprise styling.
    """
    counts = df[TARGET_COLUMN].value_counts().reindex(VALID_TARGET_CLASSES).reset_index()
    counts.columns = ["Satisfaction Level", "Customer Count"]

    fig = px.bar(
        counts,
        x="Satisfaction Level",
        y="Customer Count",
        color="Satisfaction Level",
        color_discrete_map=CLASS_COLOR_MAP,
        text="Customer Count",
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=12, color="#0F172A", family="Plus Jakarta Sans, Inter, sans-serif"),
        marker=dict(line=dict(width=0)),
        width=0.45,
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
        showlegend=False,
        yaxis_title="Number of Customers",
        xaxis_title="Satisfaction Level",
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0", range=[0, 600]),
        margin=dict(l=40, r=20, t=20, b=40),
        height=340,
    )
    return fig


def plot_service_factor_vs_target_plotly(
    df: pd.DataFrame,
    factor_col: str,
    title: Optional[str] = None,
) -> go.Figure:
    """
    Interactive Grouped Bar or Box chart for a specific service factor against satisfaction.
    """
    is_time_feature = "Time" in factor_col

    if is_time_feature:
        # Box plot for continuous time features
        fig = px.box(
            df,
            x=TARGET_COLUMN,
            y=factor_col,
            color=TARGET_COLUMN,
            category_orders={TARGET_COLUMN: VALID_TARGET_CLASSES},
            color_discrete_map=CLASS_COLOR_MAP,
            points="outliers",
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
            showlegend=False,
            xaxis_title="Satisfaction Level",
            yaxis_title=f"{factor_col.replace('_', ' ')} (minutes)",
            xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
            yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
            margin=dict(l=40, r=20, t=20, b=40),
            height=360,
        )
    else:
        # Grouped bar chart of mean rating
        means = df.groupby(TARGET_COLUMN)[factor_col].mean().reindex(VALID_TARGET_CLASSES).reset_index()
        means.columns = ["Satisfaction Level", f"Mean {factor_col}"]

        fig = px.bar(
            means,
            x="Satisfaction Level",
            y=f"Mean {factor_col}",
            color="Satisfaction Level",
            color_discrete_map=CLASS_COLOR_MAP,
            text=f"Mean {factor_col}",
        )
        fig.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            textfont=dict(size=12, color="#0F172A", family="Plus Jakarta Sans, Inter, sans-serif"),
            marker=dict(line=dict(width=0)),
            width=0.45,
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
            showlegend=False,
            xaxis_title="Satisfaction Level",
            yaxis_title=f"Average Rating (1 to 5)",
            yaxis=dict(range=[0, 5.5], gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
            xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
            margin=dict(l=40, r=20, t=20, b=40),
            height=360,
        )

    return fig


def plot_numerical_distributions_plotly(df: pd.DataFrame, feature_col: str) -> go.Figure:
    """
    Histogram and KDE distribution for a chosen numerical feature.
    """
    fig = px.histogram(
        df,
        x=feature_col,
        color=TARGET_COLUMN,
        category_orders={TARGET_COLUMN: VALID_TARGET_CLASSES},
        color_discrete_map=CLASS_COLOR_MAP,
        barmode="overlay",
        opacity=0.8,
        marginal="box",
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
        xaxis_title=feature_col.replace('_', ' '),
        yaxis_title="Count",
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0", showline=True, linecolor="#E2E8F0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=30, b=40),
        height=380,
    )
    return fig


def plot_correlation_heatmap_plotly(df: pd.DataFrame) -> go.Figure:
    """
    Interactive Correlation Heatmap for numerical service features.
    """
    corr = get_correlation_matrix(df)
    labels = [c.replace("_", " ") for c in corr.columns]

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#0F2747"],
                [0.5, "#F8FAFC"],
                [1.0, "#0EA5E9"]
            ],
            zmin=-1.0,
            zmax=1.0,
            text=corr.values,
            texttemplate="%{text:.2f}",
            hoverongaps=False,
        )
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans, Inter, sans-serif", color="#1E293B"),
        height=520,
        margin=dict(l=60, r=30, t=30, b=60),
    )
    return fig


# =====================================================================
# STATIC MATPLOTLIB / SEABORN FIGURE EXPORTERS (SAVED TO DISK)
# =====================================================================

def save_eda_figures(output_dir: str = "outputs/figures") -> None:
    """
    Generates and saves static PNG figures into outputs/figures/ for documentation/academic report.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_dataset()

    sns.set_theme(style="whitegrid", font="sans-serif")

    # 1. Target distribution
    plt.figure(figsize=(7, 4.5))
    counts = df[TARGET_COLUMN].value_counts().reindex(VALID_TARGET_CLASSES)
    palette = [CLASS_COLOR_MAP[c] for c in VALID_TARGET_CLASSES]
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=palette, legend=False)
    plt.title("Satisfaction Level Distribution (N=1,500)", fontsize=13, weight="bold", pad=12)
    plt.xlabel("Satisfaction Level", fontsize=11)
    plt.ylabel("Number of Customers", fontsize=11)
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2.0, p.get_height() + 10),
                    ha="center", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "satisfaction_distribution.png"), dpi=200)
    plt.close()

    # 2. Key service factors vs target
    key_factors = ["Service_Quality", "Staff_Behaviour", "Waiting_Time_Min", "Complaint_Resolution"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, factor in enumerate(key_factors):
        if "Time" in factor:
            sns.boxplot(data=df, x=TARGET_COLUMN, y=factor, order=VALID_TARGET_CLASSES,
                        hue=TARGET_COLUMN, palette=palette, legend=False, ax=axes[idx])
            axes[idx].set_ylabel(f"{factor.replace('_', ' ')} (min)", fontsize=10)
        else:
            sns.barplot(data=df, x=TARGET_COLUMN, y=factor, order=VALID_TARGET_CLASSES,
                        hue=TARGET_COLUMN, palette=palette, legend=False, ax=axes[idx], errorbar=None)
            axes[idx].set_ylabel("Mean Rating (1-5)", fontsize=10)
            axes[idx].set_ylim(0, 5.2)

        axes[idx].set_title(f"{factor.replace('_', ' ')} vs Satisfaction", fontsize=11, weight="bold")
        axes[idx].set_xlabel("Satisfaction Level", fontsize=10)

    plt.suptitle("Service Factor Relationships Across Customer Satisfaction Levels", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "service_factors_vs_satisfaction.png"), dpi=200)
    plt.close()

    # 3. Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = get_correlation_matrix(df)
    clean_labels = [c.replace("_", " ") for c in corr.columns]
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f",
                xticklabels=clean_labels, yticklabels=clean_labels, cbar_kws={'label': 'Pearson Correlation'})
    plt.title("Correlation Matrix of Numerical Service Features", fontsize=13, weight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"), dpi=200)
    plt.close()

    print(f"EDA figures successfully saved to '{output_dir}'.")


if __name__ == "__main__":
    save_eda_figures()
