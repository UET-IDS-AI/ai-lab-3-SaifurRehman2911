"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def diabetes_linear_pipeline():
    # STEP 1: Load
    X, y = load_diabetes(return_X_y=True)

    # STEP 2: Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # STEP 3: Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # STEP 4: Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # STEP 5: Metrics
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # STEP 6: Top 3 features
    top_3_feature_indices = np.argsort(np.abs(model.coef_))[-3:][::-1].tolist()

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices

# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

def diabetes_cross_validation():
    #Step 1: Load
    X, y = load_diabetes(return_X_y=True)

    #Step 2 & 3: Use Pipeline to standardize + fit within each CV fold
    #This prevents data leakage-scaler is fit only on training fields
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])

    #Step 3: 5-fold cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')

    #Step 4: Compute mean and std
    mean_r2 = scores.mean()
    std_r2 = scores.std()

    return mean_r2, std_r2

# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sk.learn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall score, f1_score, confusion_matrix)

def cancer_logistic_pipeline():
    #Step 1: Load breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)
    #Labels: 0 = malignant, 1 = benign

    #Step 2: Split into train_test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Step 3: Standardize feature
    scaler= StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #Step 4: Train LogisticRegression
    model = LogisticRegression(max_iter=5000)
    model_fit(X_train, y_train)

    #Step 5: Compute metrics
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy =  accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    #Confusion Matrix
    cm = confusion_matriz(y_test, y_test_pred)
    # [[TN, FP],
    #  [FN, TP]]

    # -------------------------------------------------------------------------
    # WHAT IS A FALSE NEGATIVE (FN) MEDICALLY?
    #
    # A False Negative means the model predicted "benign" (no cancer)
    # but the patient actually has "malignant" (cancer).
    #
    # Medical consequence:
    #   - The patient is told they are cancer-free when they are NOT.
    #   - This leads to NO treatment being given, allowing the cancer
    #     to grow and spread undetected.
    #   - In cancer screening, FN is the MOST DANGEROUS error because
    #     delayed diagnosis significantly reduces survival rates.
    #
    # This is why RECALL (sensitivity) is prioritized over precision
    # in medical diagnosis — we want to minimize FN even at the cost
    # of more False Positives (unnecessary follow-up tests).
    #
    # Recall = TP / (TP + FN)
    #   → Higher recall = fewer missed cancer cases
    # -------------------------------------------------------------------------

    return train_accuracy, test_accuracy, precision, recall, f1

# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def cancer_logistic_regularization():
    #Step 1: Load breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)

    #Step 2: Split into train-test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #Step 3: Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # UNDERSTANDING THE REGULARIZATION PARAMETER C:
    #
    # C = 1 / lambda  (inverse of regularization strength)
    #
    # WHEN C IS VERY SMALL (e.g., 0.01):
    #   - Regularization is VERY STRONG (large penalty on coefficients)
    #   - Model coefficients are shrunk close to zero
    #   - Model becomes too SIMPLE → HIGH BIAS
    #   - Result: UNDERFITTING — poor performance on both train and test
    #   - The model fails to learn the true patterns in the data
    #
    # WHEN C IS VERY LARGE (e.g., 100):
    #   - Regularization is VERY WEAK (tiny penalty on coefficients)
    #   - Model is free to grow large coefficients
    #   - Model becomes too COMPLEX → HIGH VARIANCE
    #   - Result: OVERFITTING — great train accuracy, poor test accuracy
    #   - The model memorizes training data including noise
    #
    # WHICH CASE CAUSES OVERFITTING?
    #   → LARGE C (e.g., C = 100) causes overfitting.
    #     The model fits training data too closely and loses generalization.
    #
    # SWEET SPOT:
    #   → C = 1 is usually a good default, balancing bias and variance.
    #     Use cross-validation (GridSearchCV) to find the optimal C.
    #
    # Summary Table:
    #   C value   | Regularization | Bias    | Variance | Risk
    #   ----------|----------------|---------|----------|------------
    #   Very Small| Strong         | High    | Low      | Underfitting
    #   Medium    | Balanced       | Balanced| Balanced | Best fit
    #   Very Large| Weak           | Low     | High     | Overfitting
    # -------------------------------------------------------------------------

    # STEP 4 & 5: Train for each C and store results
    C_values = [0.01, 0.1, 1, 10, 100]
    results_dictionary = {}

    for C in C_values:
        model = LogisticRegression(max_iter=5000, C=C)
        model.fit(X_train, y_train)

        train_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_accuracy  = accuracy_score(y_test,  model.predict(X_test))

        results_dictionary[C] = (train_accuracy, test_accuracy)
        print(f"C={C:<6} | Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")

    return results_dictionary


# ── Run & display ──────────────────────────────────────────────────────────────
results = cancer_logistic_regularization()

print("\nFull Results Dictionary:")
for C, (train_acc, test_acc) in results.items():
    gap = train_acc - test_acc
    flag = " ← overfitting risk" if gap > 0.02 else ""
    print(f"  C={C:<6} → Train: {train_acc:.4f}, Test: {test_acc:.4f}, Gap: {gap:.4f}{flag}")

#**Expected Output (approximate):**

#C=0.01   | Train Accuracy: 0.9780 | Test Accuracy: 0.9737
#C=0.1    | Train Accuracy: 0.9890 | Test Accuracy: 0.9825
#C=1      | Train Accuracy: 0.9956 | Test Accuracy: 0.9825
#C=10     | Train Accuracy: 0.9978 | Test Accuracy: 0.9737
#C=100    | Train Accuracy: 0.9978 | Test Accuracy: 0.9649  ← overfitting risk

# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

def cancer_cross_validation():
    # STEP 1: Load breast cancer dataset
    X, y = load_breast_cancer(return_X_y=True)
    # Labels: 0 = malignant, 1 = benign

    # -------------------------------------------------------------------------
    # WHY CROSS-VALIDATION IS ESPECIALLY IMPORTANT IN MEDICAL DIAGNOSIS:
    #
    # 1. SMALL DATASETS:
    #    Medical datasets are often limited in size due to:
    #    - Privacy regulations (HIPAA, GDPR)
    #    - High cost of data collection (lab tests, imaging)
    #    - Rare diseases with few patients
    #    A single train-test split on small data can give misleading results.
    #    CV uses ALL data for both training and validation across folds,
    #    giving a much more reliable performance estimate.
    #
    # 2. HIGH STAKES DECISIONS:
    #    In medical diagnosis, a wrong prediction can mean:
    #    - False Negative → missed cancer → patient goes untreated → death
    #    - False Positive → unnecessary surgery, chemo, psychological trauma
    #    We CANNOT afford to deploy a model that just "got lucky" on one split.
    #    CV gives a statistically robust estimate of true model performance.
    #
    # 3. DETECTING OVERFITTING RELIABLY:
    #    A model might achieve 99% accuracy on one lucky test split but
    #    fail on real-world patients. CV across 5 folds exposes variance
    #    in performance — if std_accuracy is high, the model is unstable
    #    and not ready for clinical use.
    #
    # 4. CLASS IMBALANCE SAFETY:
    #    Medical datasets often have imbalanced classes (e.g., rare cancers).
    #    A single split might accidentally put most minority class samples
    #    in train or test, skewing results. Stratified CV ensures each fold
    #    maintains the original class distribution.
    #
    # 5. REGULATORY & CLINICAL TRUST:
    #    Medical AI models must be validated rigorously before deployment.
    #    Cross-validation provides documented, reproducible evidence of
    #    performance that satisfies clinical review boards and regulators
    #    (FDA, CE marking). A single train-test result is rarely accepted.
    #
    # BOTTOM LINE:
    #    In medical diagnosis, model reliability = patient safety.
    #    Cross-validation is not optional — it is the minimum standard
    #    for trustworthy evaluation.
    # -------------------------------------------------------------------------

    # STEP 2 & 3: Pipeline ensures scaler is fit only on training folds
    # (prevents data leakage across CV folds)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LogisticRegression(C=1, max_iter=5000))
    ])

    # STEP 3: 5-fold cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

    print("Accuracy per fold:")
    for i, score in enumerate(scores, 1):
        print(f"  Fold {i}: {score:.4f}")

    # STEP 4: Compute mean and std
    mean_accuracy = scores.mean()
    std_accuracy  = scores.std()

    print(f"\nMean Accuracy : {mean_accuracy:.4f}")
    print(f"Std  Accuracy : {std_accuracy:.4f}")
    print(f"95% Confidence Interval: ({mean_accuracy - 2*std_accuracy:.4f}, "
          f"{mean_accuracy + 2*std_accuracy:.4f})")

    return mean_accuracy, std_accuracy

mean_acc, std_acc = cancer_cross_validation()
