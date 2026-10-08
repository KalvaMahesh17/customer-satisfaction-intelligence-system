"""
Preprocessing Pipeline Module for Customer Satisfaction Intelligence System.
Constructs leak-free ColumnTransformer pipelines for numerical and categorical features.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.data_loader import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, TARGET_COLUMN


def build_preprocessor(
    numerical_features: List[str] = NUMERICAL_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that handles:
    1. Numerical features: Median imputation + StandardScaler
    2. Categorical features: Most-frequent imputation + OneHotEncoder(handle_unknown='ignore')
    
    Returns:
        ColumnTransformer ready to be used in an sklearn Pipeline.
    """
    # Numerical pipeline: impute missing with median, then standardize
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical pipeline: impute missing with most frequent, then one-hot encode
    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def get_feature_names_after_preprocessing(
    preprocessor: ColumnTransformer,
    numerical_features: List[str] = NUMERICAL_FEATURES,
    categorical_features: List[str] = CATEGORICAL_FEATURES,
) -> List[str]:
    """
    Extracts the feature names produced by the ColumnTransformer after fitting.
    Useful for explainability and feature importance analysis.
    """
    feature_names = []
    # Add numerical feature names
    feature_names.extend(numerical_features)

    # Add one-hot encoded categorical feature names
    try:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_encoded_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
        feature_names.extend(cat_encoded_names)
    except Exception:
        # Fallback if get_feature_names_out isn't fitted yet
        feature_names.extend([f"{cat}_encoded" for cat in categorical_features])

    return feature_names


if __name__ == "__main__":
    from src.data_loader import load_dataset
    df = load_dataset()
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X)
    names = get_feature_names_after_preprocessing(preprocessor)
    print("Preprocessing Test Successful!")
    print(f"Original features: {X.shape[1]}")
    print(f"Transformed features: {X_trans.shape[1]}")
    print(f"Transformed feature names ({len(names)}):")
    print(names)
