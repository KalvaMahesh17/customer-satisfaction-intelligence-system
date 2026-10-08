"""
Model Training, Tuning, Cross-Validation, and Selection Pipeline.
Course: IT503PC Machine Learning / IT505PC Machine Learning Lab
Project: Customer Satisfaction Intelligence System

Follows rigorous ML best practices:
1. Leak-free ColumnTransformer fitted on training data only.
2. 80/20 Stratified train/test split (random_state=42).
3. 5-Fold Stratified Cross-Validation on training data.
4. Predefined Best Model Rule: Highest Mean CV Macro F1 (tie-breaker: Mean CV Accuracy).
5. Final evaluation on untouched test set.
6. Saves model artifacts and metadata.
"""

import os
import json
import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.data_loader import (
    load_dataset,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    VALID_TARGET_CLASSES,
)
from src.preprocessing import build_preprocessor, get_feature_names_after_preprocessing


RANDOM_STATE = 42


def get_model_hyperparameter_grids() -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """
    Defines classification algorithms and small, focused hyperparameter tuning grids
    tailored for academic clarity and fast, robust execution.
    """
    models = {
        "Dummy Baseline": (
            DummyClassifier(strategy="most_frequent"),
            {},
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["lbfgs"],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE),
            {
                "model__max_depth": [3, 5, 8, 12, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [3, 5, 7, 9, 11],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],  # 1: Manhattan, 2: Euclidean
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE),
            {
                "model__n_estimators": [50, 100, 150],
                "model__max_depth": [5, 8, 12, None],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
            },
        ),
    }
    return models


def train_and_evaluate_all_models() -> Dict[str, Any]:
    """
    Executes the end-to-end training, cross-validation, tuning, and evaluation pipeline.
    """
    print("=" * 70)
    print("CUSTOMER SATISFACTION INTELLIGENCE SYSTEM — MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load dataset
    print("[1/7] Loading and validating dataset...")
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    print(f"      Total records: {len(df)}")
    print(f"      Features: {len(FEATURE_COLUMNS)} columns (Customer_ID dropped)")
    print(f"      Target: '{TARGET_COLUMN}' with classes {VALID_TARGET_CLASSES}")

    # 2. Train/Test Split (80% train, 20% test, stratified)
    print("\n[2/7] Splitting data into 80% train and 20% test sets (stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"      Training set size: {len(X_train)} samples")
    print(f"      Test set size: {len(X_test)} samples (kept untouched until final evaluation)")

    # 3. Stratified 5-Fold Cross-Validation Setup
    print("\n[3/7] Setting up 5-Fold Stratified Cross-Validation on training data...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_configs = get_model_hyperparameter_grids()
    fitted_pipelines = {}
    cv_scores = {}
    best_params = {}
    test_results = {}
    train_accuracies = {}

    # 4. Model Training and Hyperparameter Tuning using CV
    print("\n[4/7] Training models and tuning hyperparameters on training data...")
    for model_name, (estimator, param_grid) in model_configs.items():
        print(f"\n      --> Training & Tuning: {model_name}...")
        preprocessor = build_preprocessor()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        if param_grid:
            # GridSearchCV using macro F1 as scoring metric
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="f1_macro",
                cv=cv,
                n_jobs=-1,
                refit=True,
                return_train_score=False,
            )
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            mean_cv_macro_f1 = float(grid_search.best_score_)
            best_tuned_params = grid_search.best_params_
        else:
            # For Baseline DummyClassifier
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_tuned_params = {}

            # Calculate CV score for baseline
            from sklearn.model_selection import cross_val_score
            cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
            mean_cv_macro_f1 = float(np.mean(cv_f1))

        # Also compute CV Accuracy for tie-breaking
        from sklearn.model_selection import cross_val_score
        cv_acc = cross_val_score(best_pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        mean_cv_acc = float(np.mean(cv_acc))

        # Train accuracy for generalization gap
        y_train_pred = best_pipeline.predict(X_train)
        train_acc = float(accuracy_score(y_train, y_train_pred))

        fitted_pipelines[model_name] = best_pipeline
        cv_scores[model_name] = {
            "cv_macro_f1": mean_cv_macro_f1,
            "cv_accuracy": mean_cv_acc,
        }
        best_params[model_name] = best_tuned_params
        train_accuracies[model_name] = train_acc

        print(f"          Mean CV Macro F1: {mean_cv_macro_f1:.4f}")
        print(f"          Mean CV Accuracy: {mean_cv_acc:.4f}")
        print(f"          Train Accuracy:   {train_acc:.4f}")
        if best_tuned_params:
            print(f"          Best Params:      {best_tuned_params}")

    # 5. Best Model Selection (Predefined Rule: Highest Mean CV Macro F1)
    print("\n[5/7] Selecting Best Model using Predefined Selection Rule...")
    print("      Rule: Highest Mean CV Macro F1 on Training Data (Tie-breaker: Mean CV Accuracy).")
    
    # Exclude dummy baseline from candidate best models
    candidate_models = [m for m in model_configs.keys() if m != "Dummy Baseline"]
    sorted_candidates = sorted(
        candidate_models,
        key=lambda m: (cv_scores[m]["cv_macro_f1"], cv_scores[m]["cv_accuracy"]),
        reverse=True,
    )
    best_model_name = sorted_candidates[0]
    best_pipeline = fitted_pipelines[best_model_name]

    print(f"      *** SELECTED BEST MODEL: {best_model_name} ***")
    print(f"      CV Macro F1: {cv_scores[best_model_name]['cv_macro_f1']:.4f}")
    print(f"      CV Accuracy: {cv_scores[best_model_name]['cv_accuracy']:.4f}")

    # 6. Final Evaluation on Untouched Test Set
    print("\n[6/7] Evaluating all models on the untouched 20% test set (300 samples)...")
    comparison_records = []
    confusion_matrices = {}
    classification_reports = {}

    for model_name, pipeline in fitted_pipelines.items():
        y_test_pred = pipeline.predict(X_test)

        test_acc = float(accuracy_score(y_test, y_test_pred))
        test_prec = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
        test_rec = float(recall_score(y_test, y_test_pred, average="macro", zero_division=0))
        test_f1 = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_test_pred, labels=VALID_TARGET_CLASSES).tolist()
        confusion_matrices[model_name] = cm

        # Classification report dict
        cr = classification_report(y_test, y_test_pred, labels=VALID_TARGET_CLASSES, output_dict=True, zero_division=0)
        classification_reports[model_name] = cr

        train_acc = train_accuracies[model_name]
        gen_gap = round(train_acc - test_acc, 4)

        test_results[model_name] = {
            "test_accuracy": test_acc,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "test_macro_f1": test_f1,
            "train_accuracy": train_acc,
            "generalization_gap": gen_gap,
        }

        comparison_records.append(
            {
                "Model": model_name,
                "CV Macro F1": round(cv_scores[model_name]["cv_macro_f1"], 4),
                "CV Accuracy": round(cv_scores[model_name]["cv_accuracy"], 4),
                "Train Accuracy": round(train_acc, 4),
                "Test Accuracy": round(test_acc, 4),
                "Test Precision": round(test_prec, 4),
                "Test Recall": round(test_rec, 4),
                "Test Macro F1": round(test_f1, 4),
                "Generalization Gap": gen_gap,
                "Is Best Model": (model_name == best_model_name),
            }
        )

    comparison_df = pd.DataFrame(comparison_records)
    print("\n" + comparison_df.to_string(index=False))

    # 7. Save Model Artifacts, Metadata, and Feature Data
    print("\n[7/7] Saving model artifacts and metadata...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    # Save best model pipeline
    best_model_path = "models/saved_model_pipeline.pkl"
    joblib.dump(best_pipeline, best_model_path)
    print(f"      Saved best model pipeline to: {best_model_path}")

    # Save all models for Model Consensus feature
    all_models_path = "models/all_models.pkl"
    joblib.dump(fitted_pipelines, all_models_path)
    print(f"      Saved all model pipelines to: {all_models_path}")

    # Save test dataset splits for evaluation and explainability
    splits_path = "models/test_split.pkl"
    joblib.dump({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}, splits_path)
    print(f"      Saved train/test splits to: {splits_path}")

    # Save comparison dataframe to CSV
    csv_results_path = "outputs/results/model_comparison.csv"
    comparison_df.to_csv(csv_results_path, index=False)
    print(f"      Saved model comparison table to: {csv_results_path}")

    # Extract feature names after preprocessing
    fitted_preprocessor = best_pipeline.named_steps["preprocessor"]
    transformed_feature_names = get_feature_names_after_preprocessing(fitted_preprocessor)

    # Prepare model metadata
    model_metadata = {
        "best_model_name": best_model_name,
        "selection_rule": "Highest mean cross-validation Macro F1 from Stratified 5-Fold CV on training data",
        "random_state": RANDOM_STATE,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "target_classes": VALID_TARGET_CLASSES,
        "cv_scores": cv_scores,
        "test_results": test_results,
        "best_hyperparameters": best_params,
        "confusion_matrices": confusion_matrices,
        "classification_reports": classification_reports,
        "comparison_table": comparison_records,
    }

    metadata_path = "models/model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=4)
    print(f"      Saved model metadata to: {metadata_path}")

    # Prepare feature metadata
    feature_metadata = {
        "expected_columns": FEATURE_COLUMNS,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "transformed_feature_names": transformed_feature_names,
        "target_column": TARGET_COLUMN,
        "target_classes": VALID_TARGET_CLASSES,
        "numerical_ranges": {
            c: {
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "mean": round(float(df[c].mean()), 2),
                "median": round(float(df[c].median()), 2),
            }
            for c in NUMERICAL_FEATURES
        },
        "categorical_categories": {
            c: sorted(df[c].dropna().unique().tolist())
            for c in CATEGORICAL_FEATURES
        },
    }

    feature_meta_path = "models/feature_metadata.json"
    with open(feature_meta_path, "w", encoding="utf-8") as f:
        json.dump(feature_metadata, f, indent=4)
    print(f"      Saved feature metadata to: {feature_meta_path}")

    print("=" * 70)
    print("TRAINING PIPELINE COMPLETE AND SUCCESSFUL!")
    print("=" * 70)

    return model_metadata


if __name__ == "__main__":
    train_and_evaluate_all_models()
