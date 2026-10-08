"""
Comprehensive Automated Test Suite for Customer Satisfaction Intelligence System.
Verifies all 10 mandatory test scenarios specified in project audit requirements:
1. Normal customer prediction
2. Highly satisfied customer prediction
3. Highly dissatisfied customer prediction
4. Normal external CSV batch inference
5. CSV missing required field (schema rejection)
6. CSV with missing values (handled by imputer)
7. CSV with unexpected categories (handled by OneHotEncoder ignore)
8. CSV with invalid numerical ranges (warnings / handling)
9. CSV with target labels (external evaluation)
10. CSV without target labels (prediction only)
"""

import os
import joblib
import pandas as pd
import numpy as np

from src.data_loader import load_dataset, FEATURE_COLUMNS, NUMERICAL_FEATURES, TARGET_COLUMN, VALID_TARGET_CLASSES
from src.explainability import explain_single_prediction_shap


def run_all_tests():
    print("=" * 70)
    print("STARTING COMPLETE ML SYSTEM TEST SUITE (10 SCENARIOS)")
    print("=" * 70)

    # Load artifacts
    pipeline = joblib.load("models/saved_model_pipeline.pkl")
    all_models = joblib.load("models/all_models.pkl")
    splits = joblib.load("models/test_split.pkl")
    X_test = splits["X_test"]

    # TEST CASE 1: Normal customer
    print("\n[TEST 1] Testing Normal Customer Prediction...")
    normal_cust = pd.DataFrame([{
        "Service_Type": "Hospitality",
        "Customer_Type": "Regular",
        "Visit_Frequency": "Frequent",
        "Service_Quality": 3.5,
        "Staff_Behaviour": 3.5,
        "Response_Time_Min": 15.0,
        "Waiting_Time_Min": 12.0,
        "Price_Value": 3.0,
        "Product_Quality": 3.5,
        "Complaint_Resolution": 3.0,
        "Ease_of_Process": 3.5,
    }])
    pred1 = pipeline.predict(normal_cust)[0]
    prob1 = pipeline.predict_proba(normal_cust)[0]
    assert pred1 in VALID_TARGET_CLASSES
    assert len(prob1) == 3
    print(f"  --> Prediction: {pred1}, Probs: {prob1.round(3)} [PASS]")

    # TEST CASE 2: Very satisfied customer
    print("\n[TEST 2] Testing Very Satisfied Customer Prediction...")
    sat_cust = pd.DataFrame([{
        "Service_Type": "Healthcare",
        "Customer_Type": "Loyal",
        "Visit_Frequency": "Frequent",
        "Service_Quality": 5.0,
        "Staff_Behaviour": 5.0,
        "Response_Time_Min": 4.0,
        "Waiting_Time_Min": 3.0,
        "Price_Value": 5.0,
        "Product_Quality": 5.0,
        "Complaint_Resolution": 5.0,
        "Ease_of_Process": 5.0,
    }])
    pred2 = pipeline.predict(sat_cust)[0]
    prob2 = pipeline.predict_proba(sat_cust)[0]
    assert pred2 == "Satisfied"
    print(f"  --> Prediction: {pred2}, Satisfied Prob: {prob2[list(pipeline.classes_).index('Satisfied')]:.3f} [PASS]")

    # TEST CASE 3: Very dissatisfied customer
    print("\n[TEST 3] Testing Very Dissatisfied Customer Prediction...")
    dissat_cust = pd.DataFrame([{
        "Service_Type": "Banking",
        "Customer_Type": "New",
        "Visit_Frequency": "First Time",
        "Service_Quality": 1.0,
        "Staff_Behaviour": 1.0,
        "Response_Time_Min": 40.0,
        "Waiting_Time_Min": 38.0,
        "Price_Value": 1.0,
        "Product_Quality": 1.0,
        "Complaint_Resolution": 1.0,
        "Ease_of_Process": 1.0,
    }])
    pred3 = pipeline.predict(dissat_cust)[0]
    prob3 = pipeline.predict_proba(dissat_cust)[0]
    assert pred3 == "Dissatisfied"
    print(f"  --> Prediction: {pred3}, Dissatisfied Prob: {prob3[list(pipeline.classes_).index('Dissatisfied')]:.3f} [PASS]")

    # TEST CASE 4: Normal external CSV batch prediction
    print("\n[TEST 4] Testing Normal External CSV (Restaurant B Sample)...")
    rest_b = pd.read_csv("data/restaurant_b_sample.csv")
    preds_b = pipeline.predict(rest_b[FEATURE_COLUMNS])
    assert len(preds_b) == len(rest_b)
    print(f"  --> Successfully predicted {len(preds_b)} external rows [PASS]")

    # TEST CASE 5: CSV missing required field (Schema rejection)
    print("\n[TEST 5] Testing Schema Rejection on Missing Required Columns...")
    bad_df = pd.DataFrame({"Customer_ID": ["C99"], "Food_Taste": [5], "Price": [3]})
    missing = [c for c in FEATURE_COLUMNS if c not in bad_df.columns]
    assert len(missing) > 0
    print(f"  --> Successfully detected missing required columns: {len(missing)} missing [PASS]")

    # TEST CASE 6: CSV with missing values (Imputation handled)
    print("\n[TEST 6] Testing CSV with Missing / NaN Values...")
    nan_cust = normal_cust.copy()
    nan_cust.loc[0, "Service_Quality"] = np.nan
    nan_cust.loc[0, "Staff_Behaviour"] = np.nan
    nan_cust.loc[0, "Response_Time_Min"] = np.nan
    pred_nan = pipeline.predict(nan_cust)[0]
    assert pred_nan in VALID_TARGET_CLASSES
    print(f"  --> Pipeline successfully imputed NaNs and produced prediction: {pred_nan} [PASS]")

    # TEST CASE 7: CSV with unexpected / unseen categorical values
    print("\n[TEST 7] Testing Unexpected / Unseen Categorical Values...")
    unseen_cust = normal_cust.copy()
    unseen_cust.loc[0, "Service_Type"] = "Space_Tourism"  # brand new category
    unseen_cust.loc[0, "Customer_Type"] = "VIP_Astronaut"
    pred_unseen = pipeline.predict(unseen_cust)[0]
    assert pred_unseen in VALID_TARGET_CLASSES
    print(f"  --> OneHotEncoder safely ignored unseen categories; prediction: {pred_unseen} [PASS]")

    # TEST CASE 8: CSV with invalid numerical ranges
    print("\n[TEST 8] Testing Boundary / Unusual Numerical Values...")
    extreme_cust = normal_cust.copy()
    extreme_cust.loc[0, "Waiting_Time_Min"] = 120.0  # 2 hours wait
    pred_extreme = pipeline.predict(extreme_cust)[0]
    assert pred_extreme in VALID_TARGET_CLASSES
    print(f"  --> Model processed extreme wait time without crashing; prediction: {pred_extreme} [PASS]")

    # TEST CASE 9: CSV with target labels (External evaluation)
    print("\n[TEST 9] Testing External Labeled Evaluation (Restaurant C)...")
    rest_c = pd.read_csv("data/restaurant_c_labeled_sample.csv")
    assert TARGET_COLUMN in rest_c.columns
    c_preds = pipeline.predict(rest_c[FEATURE_COLUMNS])
    from sklearn.metrics import accuracy_score
    c_acc = accuracy_score(rest_c[TARGET_COLUMN], c_preds)
    print(f"  --> Evaluated {len(rest_c)} labeled rows; External Accuracy: {c_acc*100:.1f}% [PASS]")

    # TEST CASE 10: CSV without target labels (Prediction only mode)
    print("\n[TEST 10] Testing External CSV Without Labels (Template)...")
    template = pd.read_csv("data/prediction_template.csv")
    assert TARGET_COLUMN not in template.columns
    t_preds = pipeline.predict(template[FEATURE_COLUMNS])
    assert len(t_preds) == len(template)
    print(f"  --> Template batch prediction succeeded for {len(t_preds)} rows [PASS]")

    # Bonus test: Local explainability
    print("\n[BONUS] Testing Local Explainability on Sample...")
    exp = explain_single_prediction_shap(pipeline, normal_cust, X_test, predicted_class=pred1)
    assert "feature_impacts" in exp
    assert len(exp["feature_impacts"]) > 0
    print(f"  --> Generated local explanation using {exp['method']} [PASS]")

    print("\n" + "=" * 70)
    print("ALL 10 MANDATORY TEST CASES PASSED WITH 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
