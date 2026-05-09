"""
Baseline optimizers: SGD, Adam, and Projected SGD with L1 constraint.
"""

import time
import numpy as np
import pandas as pd

from src.utils.numerical import sanitize, minibatch_grad, clip_gradient, compute_metrics
from src.optimizers.constraints import project_onto_l1_ball


def _record(history, epoch, start, w, X_tr, y_tr, X_te, y_te, extra=None):
    m = compute_metrics(X_tr, y_tr, X_te, y_te, w)
    m["epoch"]   = epoch + 1
    m["runtime"] = time.time() - start
    if extra:
        m.update(extra)
    history.append(m)


def train_sgd(X_tr, y_tr, X_te, y_te, epochs=40, batch_size=64, lr=0.05):
    """
    Unconstrained mini-batch Stochastic Gradient Descent.

    Update rule:
        w_{t+1} = w_t - lr * g_t

    No constraint is enforced. Serves as the dense accuracy upper bound.
    """
    w     = np.zeros(X_tr.shape[1])
    hist  = []
    start = time.time()
    steps = max(1, X_tr.shape[0] // batch_size)

    for ep in range(epochs):
        for _ in range(steps):
            g = minibatch_grad(X_tr, y_tr, w, batch_size)
            w = sanitize(w - lr * g)
        _record(hist, ep, start, w, X_tr, y_tr, X_te, y_te)

    return w, pd.DataFrame(hist)


def train_adam(X_tr, y_tr, X_te, y_te,
               epochs=40, batch_size=64, lr=0.01,
               beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam optimizer (Kingma & Ba, 2015) with bias correction.

    Adaptive moment estimation; serves as the adaptive unconstrained baseline.
    """
    d = X_tr.shape[1]
    w, m, v, t = np.zeros(d), np.zeros(d), np.zeros(d), 0
    hist  = []
    start = time.time()
    steps = max(1, X_tr.shape[0] // batch_size)

    for ep in range(epochs):
        for _ in range(steps):
            t   += 1
            g    = minibatch_grad(X_tr, y_tr, w, batch_size)
            m    = beta1 * m + (1 - beta1) * g
            v    = beta2 * v + (1 - beta2) * g**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            step  = clip_gradient(lr * m_hat / (np.sqrt(v_hat) + eps), max_norm=1.0)
            w     = sanitize(w - step)
        _record(hist, ep, start, w, X_tr, y_tr, X_te, y_te)

    return w, pd.DataFrame(hist)


def train_projected_sgd(X_tr, y_tr, X_te, y_te,
                        epochs=40, batch_size=64, lr=0.05, R=5.0):
    """
    Projected SGD with Euclidean projection onto the L1 ball.

    Update rule:
        w_{t+1} = proj_{||w||_1 <= R}(w_t - lr * g_t)

    Projection costs O(d log d) per step via isotonic soft-thresholding.
    Serves as the constrained sparse baseline.
    """
    w     = np.zeros(X_tr.shape[1])
    hist  = []
    start = time.time()
    steps = max(1, X_tr.shape[0] // batch_size)

    for ep in range(epochs):
        for _ in range(steps):
            g = minibatch_grad(X_tr, y_tr, w, batch_size)
            w = project_onto_l1_ball(sanitize(w - lr * g), R=R)
        _record(hist, ep, start, w, X_tr, y_tr, X_te, y_te)

    return w, pd.DataFrame(hist)
