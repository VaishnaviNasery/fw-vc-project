"""
Dataset loaders for the FW-VC project.

All loaders standardize features, append a bias column, and return
(X_train, X_test, y_train, y_test) as numpy arrays with integer labels.
"""

import numpy as np
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils.numerical import sanitize


def _prepare(X_tr, X_te, y_tr, y_te, name):
    """Standardize, append bias, sanitize, and print summary."""
    scaler = StandardScaler()
    X_tr = sanitize(scaler.fit_transform(X_tr))
    X_te = sanitize(scaler.transform(X_te))
    X_tr = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
    X_te = np.hstack([X_te, np.ones((X_te.shape[0], 1))])
    print(f"\nLoaded {name} | train: {X_tr.shape}  test: {X_te.shape}")
    print(f"  Class balance — train: {np.mean(y_tr):.3f}  test: {np.mean(y_te):.3f}")
    return X_tr, X_te, y_tr.astype(int), y_te.astype(int)


def load_digits_binary(test_size=0.2, random_state=42):
    """
    sklearn digits dataset (8x8 images, 64 features, n=1797).
    Binary classification: digit 0 vs. all others (~10% positive rate).
    Used as an MNIST-proxy for image classification experiments.
    """
    ds = load_digits()
    X = ds.data.astype(np.float64)
    y = (ds.target == 0).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    return _prepare(X_tr, X_te, y_tr, y_te, "Digits (MNIST-proxy) | digit 0 vs. rest")


def load_breast_cancer_binary(test_size=0.2, random_state=42):
    """
    Wisconsin Breast Cancer dataset (30 features, n=569).
    Binary classification: malignant (1) vs. benign (0).
    Serves as the tabular benchmark in place of the Adult dataset.
    """
    ds = load_breast_cancer()
    X = ds.data.astype(np.float64)
    y = 1 - ds.target.astype(int)   # flip: malignant=1, benign=0
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    return _prepare(X_tr, X_te, y_tr, y_te, "Breast Cancer | malignant vs. benign")
