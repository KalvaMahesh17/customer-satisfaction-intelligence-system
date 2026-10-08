"""
Data Loader Module for Customer Satisfaction Intelligence System.
Handles dataset loading, structure validation, and schema checks.
"""

import os
from typing import Tuple, Dict, Any, Optional
import pandas as pd


EXPECTED_COLUMNS = [
    "Customer_ID",
    "Service_Type",
    "Customer_Type",
    "Visit_Frequency",
    "Service_Quality",
    "Staff_Behaviour",
    "Response_Time_Min",
    "Waiting_Time_Min",
    "Price_Value",
    "Product_Quality",
    "Complaint_Resolution",
    "Ease_of_Process",
    "Satisfaction_Level",
]

FEATURE_COLUMNS = [col for col in EXPECTED_COLUMNS if col not in ["Customer_ID", "Satisfaction_Level"]]
CATEGORICAL_FEATURES = ["Service_Type", "Customer_Type", "Visit_Frequency"]
NUMERICAL_FEATURES = [
    "Service_Quality",
    "Staff_Behaviour",
    "Response_Time_Min",
    "Waiting_Time_Min",
    "Price_Value",
    "Product_Quality",
    "Complaint_Resolution",
    "Ease_of_Process",
]
TARGET_COLUMN = "Satisfaction_Level"
VALID_TARGET_CLASSES = ["Dissatisfied", "Neutral", "Satisfied"]


def load_dataset(filepath: str = "data/customer_satisfaction_dataset.csv") -> pd.DataFrame:
    """
    Loads the primary training dataset and performs initial validation.
    
    Raises:
        FileNotFoundError: If dataset path is invalid.
        ValueError: If file is empty or corrupted.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please ensure 'customer_satisfaction_dataset.csv' "
            "exists in the 'data/' directory."
        )

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file '{filepath}': {e}")

    if df.empty:
        raise ValueError(f"The dataset at '{filepath}' is empty.")

    return df


def inspect_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a comprehensive dataset inspection dictionary for Phase 0 and EDA.
    """
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {k: str(v) for k, v in df.dtypes.items()},
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df[TARGET_COLUMN].value_counts().to_dict() if TARGET_COLUMN in df else None,
        "unique_categories": {c: sorted(df[c].dropna().unique().tolist()) for c in cat_cols if c != "Customer_ID"},
        "numerical_ranges": {
            c: {"min": float(df[c].min()), "max": float(df[c].max()), "mean": float(df[c].mean())}
            for c in num_cols
        },
    }
    return summary


if __name__ == "__main__":
    df = load_dataset()
    info = inspect_dataset_summary(df)
    print("=" * 60)
    print("PHASE 0: DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"Total Rows: {info['num_rows']}")
    print(f"Total Columns: {info['num_columns']}")
    print(f"Columns: {info['column_names']}")
    print(f"Missing Values: {info['total_missing']}")
    print(f"Duplicate Rows: {info['duplicate_rows']}")
    print(f"Target Distribution: {info['target_distribution']}")
    print("\nCategorical Unique Values:")
    for cat, vals in info['unique_categories'].items():
        print(f"  - {cat}: {vals}")
    print("\nNumerical Ranges:")
    for num, rng in info['numerical_ranges'].items():
        print(f"  - {num}: min={rng['min']}, max={rng['max']}, mean={rng['mean']:.2f}")
    print("=" * 60)
