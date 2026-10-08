# Customer Satisfaction Intelligence System
### Customer Satisfaction Classification and Service Analysis

**Course:** IT503PC – Machine Learning  
**Lab:** IT505PC – Machine Learning Lab  
**Type:** Supervised Multiclass Classification + Service Analysis + Explainable ML + Interactive Web Application  
**Tech Stack:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit, Joblib, SHAP  

---

## 1. Problem Statement & Vision
In competitive service industries (such as restaurants, hospitality, retail, e-commerce, and banking), customer retention hinges on understanding service quality gaps before customers churn. Traditional feedback forms often capture ratings without explaining what specific factors drive dissatisfaction or providing ways to simulate service interventions.

This project builds an end-to-end **Customer Satisfaction Intelligence Platform** answering four central questions:
1. **WHAT** is the customer's satisfaction level? (*Satisfied*, *Neutral*, *Dissatisfied*)
2. **WHY** did the model make that prediction? (Local factor attribution & explainability)
3. **WHAT** happens if service factors are improved? (Real-time What-If scenario simulation)
4. **WHAT** operational areas deserve immediate manager attention? (Customer rescue mode)

---

## 2. Real-World Restaurant / Service Scenario
Imagine a restaurant or service establishment collecting feedback across customer touchpoints. 
- Historical feedback records are used to train machine learning classifiers.
- When a new customer completes their visit, the manager enters their service factors (e.g. staff behaviour, waiting time, price value).
- The system predicts the customer's satisfaction tier, shows agreement across multiple models (**Model Consensus**), and explains which factors influenced the prediction.
- If dissatisfied, the manager tests improvements using the **What-If Simulator** (e.g., *"What if waiting time dropped from 30 to 10 minutes?"*).
- Later, a neighboring branch (e.g., "Restaurant B") uploads a new batch CSV of customers. The system validates the schema, predicts satisfaction for each row, and provides a downloadable results CSV.

---

## 3. Dataset & Academic Disclosure
- **File:** `data/customer_satisfaction_dataset.csv`
- **Total Records:** 1,500 customer records
- **Target Distribution:** 500 Satisfied, 500 Neutral, 500 Dissatisfied (Perfectly balanced 1:1:1)
- **Target Column:** `Satisfaction_Level`
- **Identifier:** `Customer_ID` (strictly excluded from predictive modeling to prevent data leakage)

> **Important Disclosure:** *The training dataset is synthetic and created specifically for academic demonstration in IT503PC / IT505PC courses.*

### Features:
| Column | Type | Description |
| :--- | :--- | :--- |
| `Customer_ID` | String | Unique synthetic identifier (excluded from model) |
| `Service_Type` | Categorical | Banking, E-Commerce, Healthcare, Hospitality, Retail, Telecom |
| `Customer_Type` | Categorical | Loyal, New, Regular |
| `Visit_Frequency` | Categorical | First Time, Frequent, Occasional |
| `Service_Quality` | Numerical | Rating from 1 (Very Poor) to 5 (Excellent) |
| `Staff_Behaviour` | Numerical | Rating from 1 (Very Poor) to 5 (Excellent) |
| `Response_Time_Min` | Numerical | Approximate response time in minutes |
| `Waiting_Time_Min` | Numerical | Approximate waiting time in minutes |
| `Price_Value` | Numerical | Perceived value for money rating (1–5) |
| `Product_Quality` | Numerical | Product/food quality rating (1–5) |
| `Complaint_Resolution` | Numerical | Rating for complaint handling (1–5) |
| `Ease_of_Process` | Numerical | Ease of service/checkout rating (1–5) |
| `Satisfaction_Level` | Target | **Dissatisfied**, **Neutral**, **Satisfied** |

---

## 4. Machine Learning Methodology

### A. Preprocessing Pipeline (`src/preprocessing.py`)
To prevent data leakage, all preprocessing is encapsulated within an `sklearn.compose.ColumnTransformer` fitted strictly on training data:
- **Numerical Features:** Imputed with median values using `SimpleImputer(strategy='median')` and scaled using `StandardScaler()`.
- **Categorical Features:** Imputed with mode using `SimpleImputer(strategy='most_frequent')` and encoded using `OneHotEncoder(handle_unknown='ignore')`.

### B. Train/Test Split
- **Split Ratio:** 80% Training (1,200 samples) / 20% Testing (300 samples).
- **Stratification:** `stratify=y` to preserve exact class balance across splits.
- **Random Seed:** `random_state=42` for exact reproducibility.
- **Integrity Rule:** The test set was kept completely untouched during model selection and hyperparameter tuning.

### C. Models Implemented
1. **DummyClassifier (Baseline):** Most frequent class strategy (achieves 33.33% accuracy, 0.1667 Macro F1 on balanced 3-class target).
2. **Logistic Regression:** Multinomial linear classification with L2 regularization ($C=0.1$).
3. **Decision Tree Classifier:** Non-linear decision tree tuned over `max_depth`, `min_samples_split`, and `min_samples_leaf`.
4. **K-Nearest Neighbors (KNN):** Distance-based instance classifier tuned over `n_neighbors`, distance weighting, and distance metrics.
5. **Random Forest Classifier:** Bagging ensemble of 150 decision trees.

### D. 5-Fold Stratified Cross-Validation & Tuning
Cross-validation was conducted exclusively on the 80% training split using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` and `GridSearchCV(scoring='f1_macro')`.

### E. Best Model Selection Rule
Defined prior to test evaluation:
- **Primary Metric:** Highest Mean Cross-Validation Macro F1 on training data.
- **Tie-Breaker:** Highest Mean Cross-Validation Accuracy.
- **Champion Selected:** **Logistic Regression** (Mean CV Macro F1: `0.6388`, CV Accuracy: `64.25%`).

---

## 5. Experimental Results (Untouched Test Set)

| Model | CV Macro F1 | CV Accuracy | Train Accuracy | Test Accuracy | Test Precision | Test Recall | Test Macro F1 | Generalization Gap | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** | 0.1667 | 0.3333 | 0.3333 | 0.3333 | 0.1111 | 0.3333 | 0.1667 | 0.0000 | Baseline |
| **Logistic Regression** | **0.6388** | **0.6425** | 0.6575 | **0.6267** | **0.6205** | **0.6267** | **0.6231** | **0.0308** | **Best Model** |
| **Decision Tree** | 0.4915 | 0.4933 | 0.7792 | 0.4733 | 0.4745 | 0.4733 | 0.4738 | 0.3058 | Overfit |
| **KNN** | 0.5679 | 0.5700 | 1.0000 | 0.5933 | 0.5929 | 0.5933 | 0.5931 | 0.4067 | Overfit |
| **Random Forest** | 0.5957 | 0.6033 | 0.9950 | 0.6100 | 0.6017 | 0.6100 | 0.6047 | 0.3850 | Overfit |

### Viva Insight on Model Selection:
While KNN and Random Forest fit the training set almost perfectly (99.5% - 100% train accuracy), their test performance dropped significantly to ~59-61% (generalization gap of ~38-40%). In contrast, **Logistic Regression achieved the lowest generalization gap (only 3.08%) and the highest Cross-Validation Macro F1 (0.6388)**, proving it is the most stable and reliable classifier for this dataset.

---

## 6. Key Application Features

1. **Dashboard:** High-level executive KPI cards, class breakdown charts, model performance comparison, and top service drivers.
2. **Data Analysis:** Interactive EDA including distribution plots, correlation heatmap, and service factor cross-tabulations.
3. **Model Performance:** Algorithm comparison table, train vs. test diagnostics, per-model interactive confusion matrix, and classification report.
4. **Customer Prediction:** Interactive customer input with presets, live prediction badge, probability distribution, **Model Consensus** (voting across all 4 algorithms), and **Prediction Explainability** (feature attribution).
5. **What-If Simulator & Rescue Mode:** Real-time parameter tweaking with before vs. after comparison, plus automated **Customer Rescue Analysis** evaluating standard service intervention packages.
6. **Upload & Batch Prediction:** Upload external CSV, automated schema and range validation, batch prediction with CSV export, and external labeled evaluation (if ground-truth targets exist).
7. **About Page:** Complete academic documentation, methodology details, disclosures, limitations, and future work.

---

## 7. Project Structure
```
customer-satisfaction-intelligence-system/
│
├── data/
│   ├── customer_satisfaction_dataset.csv
│   ├── customer_satisfaction_data_dictionary.csv
│   ├── prediction_template.csv
│   ├── restaurant_b_sample.csv
│   └── restaurant_c_labeled_sample.csv
│
├── models/
│   ├── saved_model_pipeline.pkl
│   ├── all_models.pkl
│   ├── test_split.pkl
│   ├── model_metadata.json
│   └── feature_metadata.json
│
├── outputs/
│   ├── figures/
│   └── results/
│       ├── model_comparison.csv
│       └── permutation_importance.csv
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── evaluate_models.py
│   ├── analysis.py
│   └── explainability.py
│
├── app.py
├── requirements.txt
├── README.md
└── VIVA_NOTES.md
```

---

## 8. Installation & Execution Guide

### Prerequisites
- Python 3.9 to 3.14
- Recommended: Virtual environment (`venv` or `conda`)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Re-train Models & Run Evaluation
```bash
# Trains models, performs 5-fold CV, and saves artifacts to models/
python -m src.train_model

# Evaluates saved models on untouched test split
python -m src.evaluate_models

# Generates EDA report figures into outputs/figures/
python -m src.analysis
```

### Step 3: Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 9. Limitations & Ethical Considerations
- **Synthetic Data:** The dataset is synthetic; real restaurant operations encounter non-linear psychological nuances and unobserved confounders (e.g., weather, ambient music, parking availability).
- **Correlation vs. Causation:** Model explainability reflects algorithmic associations learned by the model, not proven causal facts.
- **Dataset Compatibility:** External CSV prediction requires compatible feature schemas; arbitrary data formats are rejected by design to prevent erroneous inferences.

---

## 10. Future Scope
- Integration with point-of-sale (POS) tablet terminals for real-time customer feedback capture.
- Incorporating NLP sentiment analysis on open-ended text reviews.
- MLOps deployment on cloud containers (Docker / AWS / GCP) with automated model monitoring and drift detection.
