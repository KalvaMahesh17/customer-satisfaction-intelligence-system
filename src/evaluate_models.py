"""
Model Evaluation Script for Customer Satisfaction Intelligence System.
Loads saved models and test datasets, generates comprehensive evaluation reports,
confusion matrices, and train vs test generalization diagnostics.
"""

import os
import json
import joblib
from typing import Dict, Any
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from src.data_loader import VALID_TARGET_CLASSES


def evaluate_saved_models(
    splits_path: str = "models/test_split.pkl",
    all_models_path: str = "models/all_models.pkl",
    metadata_path: str = "models/model_metadata.json",
) -> Dict[str, Any]:
    """
    Evaluates all saved model pipelines on the test split and returns evaluation details.
    """
    if not os.path.exists(splits_path) or not os.path.exists(all_models_path):
        raise FileNotFoundError("Model artifacts not found. Please run src/train_model.py first.")

    splits = joblib.load(splits_path)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    models_dict = joblib.load(all_models_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    evaluation_summary = {}

    for name, pipeline in models_dict.items():
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        macro_prec = precision_score(y_test, y_test_pred, average="macro", zero_division=0)
        macro_rec = recall_score(y_test, y_test_pred, average="macro", zero_division=0)
        macro_f1 = f1_score(y_test, y_test_pred, average="macro", zero_division=0)

        cm = confusion_matrix(y_test, y_test_pred, labels=VALID_TARGET_CLASSES)
        cr_dict = classification_report(
            y_test, y_test_pred, labels=VALID_TARGET_CLASSES, output_dict=True, zero_division=0
        )
        cr_text = classification_report(
            y_test, y_test_pred, labels=VALID_TARGET_CLASSES, zero_division=0
        )

        evaluation_summary[name] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "test_precision_macro": float(macro_prec),
            "test_recall_macro": float(macro_rec),
            "test_macro_f1": float(macro_f1),
            "generalization_gap": float(train_acc - test_acc),
            "confusion_matrix": cm.tolist(),
            "classification_report_dict": cr_dict,
            "classification_report_text": cr_text,
        }

    return evaluation_summary


if __name__ == "__main__":
    results = evaluate_saved_models()
    print("=" * 60)
    print("MODEL EVALUATION & GENERALIZATION REPORT")
    print("=" * 60)
    for model_name, metrics in results.items():
        print(f"\n--- Model: {model_name} ---")
        print(f"  Train Accuracy:      {metrics['train_accuracy']:.4f}")
        print(f"  Test Accuracy:       {metrics['test_accuracy']:.4f}")
        print(f"  Generalization Gap:  {metrics['generalization_gap']:.4f}")
        print(f"  Test Macro Precision:{metrics['test_precision_macro']:.4f}")
        print(f"  Test Macro Recall:   {metrics['test_recall_macro']:.4f}")
        print(f"  Test Macro F1:       {metrics['test_macro_f1']:.4f}")
        print("  Confusion Matrix (Dissatisfied, Neutral, Satisfied):")
        print(np.array(metrics['confusion_matrix']))
    print("=" * 60)
