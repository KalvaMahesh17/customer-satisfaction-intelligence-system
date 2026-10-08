# Customer Satisfaction Intelligence System
## Academic Viva Voce Preparation Guide & Comprehensive Q&A

**Course:** IT503PC – Machine Learning  
**Lab:** IT505PC – Machine Learning Lab  
**Student Level:** B.Tech Undergraduate  

---

### 1. Fundamentals of Machine Learning

#### Q1: What is Machine Learning?
**Answer:** Machine Learning is a branch of artificial intelligence where algorithms learn patterns, statistical dependencies, and decision boundaries directly from historical data without being explicitly programmed with rigid deterministic rules.

#### Q2: What is Supervised Learning?
**Answer:** Supervised learning is a paradigm where every training example consists of an input vector of features $X$ and an associated ground-truth label $y$. The algorithm's objective is to learn a mapping function $f(X) \approx y$ that generalizes accurately to unseen test instances.

#### Q3: What is Classification?
**Answer:** Classification is a supervised learning task where the output variable $y$ is discrete and categorical (e.g., categories, tiers, or classes), as opposed to regression where the target is a continuous numerical quantity.

#### Q4: What is Multiclass Classification?
**Answer:** Multiclass classification involves predicting one target label from three or more discrete mutually exclusive classes. In our project, the target has three distinct classes: **Dissatisfied**, **Neutral**, and **Satisfied**.

#### Q5: Why did you choose customer satisfaction as your problem domain?
**Answer:** Customer satisfaction directly impacts business retention, churn prevention, and revenue. Automating satisfaction classification allows service managers to diagnose operational bottlenecks (such as long waiting times or poor staff conduct) before customers disengage.

#### Q6: Why use three classes instead of binary (Satisfied vs Dissatisfied)?
**Answer:** Real-world feedback is rarely black-and-white. The "Neutral" class captures passive, borderline customers who have not experienced major failures but are at risk of switching to a competitor. Identifying the Neutral segment provides high-value retention opportunities.

#### Q7: What is a Feature?
**Answer:** A feature is an individual measurable property or characteristic of a phenomenon being observed (e.g., `Waiting_Time_Min`, `Staff_Behaviour`, `Service_Quality`). Features form the input vector $X$.

#### Q8: What is the Target?
**Answer:** The target ($y$) is the dependent variable or outcome that the machine learning model is trained to predict. In our project, `Satisfaction_Level` is the target column.

#### Q9: Why is the dataset synthetic?
**Answer:** Real commercial customer data often contains proprietary business information, personally identifiable information (PII), or strict non-disclosure restrictions. A synthetic dataset generated for academic demonstration allows full transparency, reproducibility, and rigorous experimentation without privacy concerns.

---

### 2. Data Preprocessing & Pipeline Integrity

#### Q10: What is Preprocessing and why is it necessary?
**Answer:** Raw data often contains missing entries, disparate measurement scales, and non-numerical text categories. Preprocessing cleans, transforms, and standardizes data into an optimal numerical representation suitable for mathematical modeling.

#### Q11: How did you handle Missing Values and why?
**Answer:**
- Numerical features were imputed using the **median** via `SimpleImputer(strategy='median')` because the median is robust to outliers and skewed distributions.
- Categorical features were imputed using the **mode (most frequent)** category via `SimpleImputer(strategy='most_frequent')`.

#### Q12: What is Categorical Encoding and which technique did you use?
**Answer:** Machine learning algorithms operate on numerical vectors. We used **One-Hot Encoding** (`OneHotEncoder(handle_unknown='ignore', sparse_output=False)`), which converts each distinct category into a binary column ($0$ or $1$) without imposing an artificial ordinal relationship.

#### Q13: Why did you apply Feature Scaling?
**Answer:** Distance-based algorithms (such as KNN) and gradient-based linear models (such as Logistic Regression) are sensitive to feature magnitudes. Features with large ranges (e.g., waiting time in minutes: 1–45) would otherwise dominate features with smaller scales (ratings: 1–5). `StandardScaler()` standardizes features to zero mean and unit variance ($z = \frac{x - \mu}{\sigma}$).

#### Q14: What is `Pipeline` in Scikit-learn?
**Answer:** A `Pipeline` chains together sequential data transformation steps and a final estimator. It ensures that during cross-validation and testing, transformations are fitted strictly on training folds and merely applied to validation folds.

#### Q15: What is `ColumnTransformer`?
**Answer:** A `ColumnTransformer` applies different transformation pipelines in parallel to distinct subsets of columns (e.g., scaling numerical features while one-hot encoding categorical features) and concatenates their outputs into a unified feature matrix.

#### Q16: What is Data Leakage and how did you prevent it?
**Answer:** Data leakage occurs when information from outside the training dataset (such as test set statistics) inadvertently influences model training. We prevented data leakage by:
1. Splitting data into train and test sets *before* any transformation.
2. Fitting the `ColumnTransformer` and imputers *only on the training split*.
3. Dropping `Customer_ID` so the model cannot memorize unique customer keys.

---

### 3. Splitting, Cross-Validation & Algorithms

#### Q17: Why did you choose an 80/20 Train/Test Split?
**Answer:** An 80/20 split is an industry standard that balances giving the model sufficient historical data to learn complex patterns (1,200 samples) while retaining a statistically significant, representative test set (300 samples) to evaluate generalization.

#### Q18: What is Stratification (`stratify=y`) and why is it crucial?
**Answer:** Stratified sampling ensures that the proportion of target classes in both the training and test splits exactly matches the overall population distribution (exactly 33.3% Dissatisfied, 33.3% Neutral, 33.3% Satisfied in both splits).

#### Q19: What is 5-Fold Stratified Cross-Validation?
**Answer:** In 5-fold CV, the training dataset is partitioned into 5 equal, stratified folds. The model is trained on 4 folds and validated on the remaining 1 fold; this process repeats 5 times so every fold serves as validation once. The average validation score provides a robust estimate of generalization error.

#### Q20: Why did you include `DummyClassifier`?
**Answer:** `DummyClassifier(strategy='most_frequent')` serves as a naive baseline. On a balanced 3-class problem, predicting the most frequent class yields exactly **33.33% Accuracy** and **0.1667 Macro F1**. Any worthwhile machine learning model must significantly outperform this baseline.

#### Q21: How does Logistic Regression work for multiclass classification?
**Answer:** Multinomial Logistic Regression models the log-odds of each class as a linear combination of input features and applies the **Softmax function** to convert raw logits into normalized probabilities summing to 1.

#### Q22: How does a Decision Tree Classifier work?
**Answer:** A Decision Tree recursively partitions feature space using orthogonal decision rules chosen to maximize impurity reduction (e.g., Gini Impurity). It is highly interpretable but susceptible to overfitting if grown too deep.

#### Q23: How does K-Nearest Neighbors (KNN) work?
**Answer:** KNN is a non-parametric, instance-based "lazy learner." Given a query sample, it computes distances (Euclidean or Manhattan) to all training points, identifies the $K$ closest neighbors, and assigns the majority class (or distance-weighted vote).

#### Q24: What is Ensemble Learning and Bagging?
**Answer:** Ensemble learning combines predictions from multiple individual models to produce a stronger, more generalized predictor. **Bagging (Bootstrap Aggregating)** trains diverse models in parallel on random bootstrap samples with replacement and averages their predictions. **Random Forest** is an ensemble of bagged decision trees with random feature subsampling.

#### Q25: Why do we tune Hyperparameters?
**Answer:** Hyperparameters govern the model's learning process and structural complexity (e.g., tree depth, number of neighbors, regularization strength $C$). Tuning via `GridSearchCV` discovers the configuration that balances bias and variance for optimal validation performance.

---

### 4. Evaluation Metrics & Generalization

#### Q26: What is Overfitting and the Generalization Gap?
**Answer:** Overfitting occurs when a model memorizes idiosyncrasies and noise in the training set rather than learning generalizable underlying patterns. The **Generalization Gap** is the difference between training accuracy and testing accuracy ($\text{Gap} = \text{Acc}_{\text{train}} - \text{Acc}_{\text{test}}$). A large gap (e.g., >20%) indicates significant overfitting.

#### Q27: What is Accuracy and why is it insufficient on its own?
**Answer:** Accuracy is the ratio of correct predictions to total predictions ($\frac{TP+TN}{Total}$). While intuitive, on imbalanced problems it can mask complete failure on minority classes. Even on balanced datasets, it does not reveal per-class trade-offs between false positives and false negatives.

#### Q28: What is Precision and Recall?
**Answer:**
- **Precision:** Of all samples predicted as class $C$, what fraction was actually class $C$? ($\frac{TP}{TP + FP}$). Focuses on avoiding false alarms.
- **Recall (Sensitivity):** Of all actual class $C$ samples, what fraction did the model correctly identify? ($\frac{TP}{TP + FN}$). Focuses on avoiding misses.

#### Q29: What is Macro F1 and why is it our Primary Metric?
**Answer:** The F1-score is the harmonic mean of precision and recall ($2 \cdot \frac{P \cdot R}{P + R}$). **Macro F1** computes the F1-score independently for each class and calculates the unweighted arithmetic mean:
$$\text{Macro F1} = \frac{F1_{\text{Dissatisfied}} + F1_{\text{Neutral}} + F1_{\text{Satisfied}}}{3}$$
It treats all classes equally and penalizes models that fail on any individual satisfaction category.

#### Q30: What is a Confusion Matrix?
**Answer:** A Confusion Matrix is an $N \times N$ contingency table (where $N=3$ here) contrasting true classes along the rows against predicted classes along the columns. Diagonal entries represent correct classifications; off-diagonal entries reveal specific misclassification patterns (e.g., Dissatisfied misclassified as Neutral).

#### Q31: How was the Best Model selected?
**Answer:**
1. **Predefined Rule:** The champion model was selected strictly based on **Highest Mean Cross-Validation Macro F1** across 5-fold CV on training data (tie-breaker: Mean CV Accuracy).
2. **Result:** **Logistic Regression** won with a Mean CV Macro F1 of **0.6388** and CV Accuracy of **64.25%**.
3. **Academic Justification:** Logistic Regression demonstrated a minimal generalization gap (only **3.08%** difference between train and test accuracy), whereas complex non-linear models like KNN (train 100%, test 59.3%) and Random Forest (train 99.5%, test 61.0%) suffered from high variance and severe overfitting.

---

### 5. Explainability & Application Architecture

#### Q32: How is the model saved and loaded?
**Answer:** The entire fitted Scikit-learn Pipeline (including the ColumnTransformer preprocessor and the trained classifier) is serialized to disk as a binary `.pkl` file using `joblib.dump()`. In the Streamlit app, it is loaded into memory using `@st.cache_resource` for instant, leak-free inference.

#### Q33: What is Global Permutation Feature Importance?
**Answer:** Permutation importance measures how much model performance (Macro F1) degrades when values in a single feature column are randomly shuffled on the held-out test set. If shuffling a feature causes a large drop in F1, the model depends heavily on that feature.

#### Q34: What is Local Explainability and SHAP?
**Answer:** Local explainability answers *“Why did the model assign this specific label to this particular customer?”* **SHAP (SHapley Additive exPlanations)** is based on cooperative game theory, attributing additive contribution values to each feature that explain the delta between the baseline expected prediction and the customer's specific output probability.

#### Q35: Why can machine learning not prove causality?
**Answer:** Machine learning models identify statistical correlations and joint probability distributions in observational data ($P(Y \mid X)$). They do not account for unmeasured confounders or counterfactual interventions ($P(Y \mid \text{do}(X))$). For example, improving recorded staff behavior in a simulator predicts higher modeled satisfaction, but cannot guarantee real customer emotional response if other operational factors fail.

#### Q36: How does the What-If Simulator work?
**Answer:** When the user adjusts service sliders (e.g., reducing waiting time), the app constructs a new input dataframe, passes it through the exact saved preprocessor and trained best model pipeline, and computes the updated prediction and probability distribution in real time without hardcoded heuristics.

#### Q37: How does CSV batch upload validation work?
**Answer:** Before running inference on an uploaded CSV, the system validates:
1. **Schema Check:** Verifies all 11 required predictive feature columns exist. If columns are missing or incompatible, prediction halts with an informative error message.
2. **Range Check:** Flags out-of-bounds ratings (<1 or >5) and negative minutes.
3. **Unseen Categories:** Safely handled by `OneHotEncoder(handle_unknown='ignore')` by assigning zeroes to unseen dummy columns without crashing.
4. **External Evaluation:** If ground-truth `Satisfaction_Level` labels exist in the uploaded file, the system computes external Accuracy, Precision, Recall, Macro F1, and a Confusion Matrix on the independent dataset.
