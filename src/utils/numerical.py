"""
Numerical utility functions shared across the project.
Handles numerical stability, gradient clipping, and metric computation.
"""

import numpy as np
from sklearn.metrics import accuracy_score


def sanitize(a, pos_cap=1e8, neg_cap=-1e8):
    """Replace NaN/Inf with finite values to prevent divergence."""
    return np.nan_to_num(
        np.asarray(a, dtype=np.float64),
        nan=0.0, posinf=pos_cap, neginf=neg_cap
    )


def sigmoid(z):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def logistic_loss_and_grad(X, y, w):
    """Binary cross-entropy loss and its gradient."""
    p = sigmoid(X @ w)
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    grad = (X.T @ (p - y)) / X.shape[0]
    return float(loss), sanitize(grad)


def clip_gradient(g, max_norm=5.0):
    """Clip gradient to have at most max_norm L2 norm."""
    norm = np.linalg.norm(g)
    return g * (max_norm / (norm + 1e-12)) if norm > max_norm else g


def minibatch_grad(X, y, w, batch_size):
    """Compute stochastic gradient on a random mini-batch."""
    idx = np.random.choice(X.shape[0], min(batch_size, X.shape[0]), replace=False)
    _, g = logistic_loss_and_grad(X[idx], y[idx], w)
    return clip_gradient(g, max_norm=5.0)


def predict(X, w):
    """Predict binary labels from logistic regression weights."""
    return (sigmoid(X @ sanitize(w)) >= 0.5).astype(int)


def sparsity_ratio(w, threshold=1e-6):
    """Fraction of weight coordinates that are effectively zero."""
    return float(np.sum(np.abs(w) <= threshold) / len(w))


def nnz(w, threshold=1e-6):
    """Number of nonzero weight coordinates."""
    return int(np.sum(np.abs(w) > threshold))


def compute_metrics(X_tr, y_tr, X_te, y_te, w):
    """Compute the full evaluation metric dictionary for a weight vector."""
    train_loss, _ = logistic_loss_and_grad(X_tr, y_tr, w)
    test_loss, _  = logistic_loss_and_grad(X_te, y_te, w)
    return {
        "train_loss": train_loss,
        "test_loss":  test_loss,
        "train_acc":  accuracy_score(y_tr, predict(X_tr, w)),
        "test_acc":   accuracy_score(y_te, predict(X_te, w)),
        "sparsity":   sparsity_ratio(w),
        "nnz":        nnz(w),
        "l1_norm":    float(np.linalg.norm(w, 1)),
        "l2_norm":    float(np.linalg.norm(w, 2)),
    }
