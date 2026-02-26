"""
Linear & Logistic Regression Lab
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

def diabetes_linear_pipeline():

    # Load
    data = load_diabetes()
    X, y = data.data, data.target

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # Metrics
    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    # Top 3 features by absolute coefficient
    coefs = np.abs(model.coef_)
    top_3_feature_indices = list(np.argsort(coefs)[-3:][::-1])

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices


# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():

    data = load_diabetes()
    X, y = data.data, data.target

    # Scale whole dataset for CV
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LinearRegression()

    scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    mean_r2 = np.mean(scores)
    std_r2 = np.std(scores)

    # Comments:
    # Std represents stability of performance across folds.
    # CV lowers variance risk by averaging multiple train/test splits.

    return mean_r2, std_r2


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    cm = confusion_matrix(y_test, test_pred)

    # Comment:
    # False Negative = patient actually has cancer but model predicts healthy.
    # This is dangerous because disease goes untreated.

    return train_accuracy, test_accuracy, precision, recall, f1


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    C_values = [0.01, 0.1, 1, 10, 100]
    results = {}

    for c in C_values:
        model = LogisticRegression(max_iter=5000, C=c)
        model.fit(X_train, y_train)

        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))

        results[c] = (train_acc, test_acc)

    # Comments:
    # Small C -> strong regularization -> simpler model -> possible underfitting.
    # Large C -> weak regularization -> complex model.
    # Very large C can cause overfitting.

    return results


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():

    data = load_breast_cancer()
    X, y = data.data, data.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    model = LogisticRegression(C=1, max_iter=5000)

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)

    # Comment:
    # In medical diagnosis, mistakes are costly.
    # CV ensures results are stable and not dependent on one lucky split.

    return mean_accuracy, std_accuracy
