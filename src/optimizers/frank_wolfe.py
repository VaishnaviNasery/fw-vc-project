"""
Frank-Wolfe optimizers for L1-constrained logistic regression.

Two variants are implemented:

1. SFW-Paper: Stochastic Frank-Wolfe with the classical open-loop step size
   schedule gamma_t = 2/(t+2), following Jaggi (2013) and Hazan & Kale (2012).
   Serves as the projection-free reference implementation.

2. FW-VC: Our proposed Frank-Wolfe with Variance-Controlled adaptive step size.
   Replaces the fixed schedule with an EMA-based adaptive rule, motivated by
   the instability of fixed step sizes in high-variance stochastic settings.

Both methods:
  - Use the L1 LMO: O(d) per step, no projection required.
  - Preserve feasibility automatically via convex combination update.
  - Track the Frank-Wolfe duality gap as a convergence diagnostic.
"""

import time
import numpy as np
import pandas as pd

from src.utils.numerical import sanitize, minibatch_grad, compute_metrics
from src.optimizers.constraints import l1_lmo, frank_wolfe_gap


def _record(history, epoch, start, w, X_tr, y_tr, X_te, y_te, extra=None):
    m = compute_metrics(X_tr, y_tr, X_te, y_te, w)
    m["epoch"]   = epoch + 1
    m["runtime"] = time.time() - start
    if extra:
        m.update(extra)
    history.append(m)


def train_sfw_paper(X_tr, y_tr, X_te, y_te,
                    epochs=40, batch_size=64, R=5.0,
                    gamma_rule="open_loop", gamma_const=0.1):
    """
    Stochastic Frank-Wolfe with classical step size schedule.

    Step size options:
        open_loop : gamma_t = 2 / (t + 2)   [Jaggi 2013, Theorem 1]
        constant  : gamma_t = gamma_const

    Update:
        s_t = LMO(g_t, R)
        x_{t+1} = (1 - gamma_t) x_t + gamma_t s_t

    Parameters
    ----------
    gamma_rule : str
        'open_loop' or 'constant'
    gamma_const : float
        Step size used when gamma_rule='constant'.
    """
    w         = np.zeros(X_tr.shape[1])
    hist      = []
    start     = time.time()
    steps     = max(1, X_tr.shape[0] // batch_size)
    global_t  = 0
    gamma_log = []
    gap_log   = []

    for ep in range(epochs):
        ep_gammas, ep_gaps = [], []

        for _ in range(steps):
            global_t += 1
            g = minibatch_grad(X_tr, y_tr, w, batch_size)
            s = l1_lmo(g, R=R)

            if gamma_rule == "open_loop":
                gamma = 2.0 / (global_t + 2.0)
            elif gamma_rule == "constant":
                gamma = gamma_const
            else:
                raise ValueError("gamma_rule must be 'open_loop' or 'constant'")

            gap = frank_wolfe_gap(g, w, s)
            w   = sanitize((1.0 - gamma) * w + gamma * s)
            ep_gammas.append(gamma)
            ep_gaps.append(gap)

        mean_gamma = float(np.mean(ep_gammas))
        mean_gap   = float(np.mean(ep_gaps))
        gamma_log.append(mean_gamma)
        gap_log.append(mean_gap)
        _record(hist, ep, start, w, X_tr, y_tr, X_te, y_te,
                extra={"gamma": mean_gamma, "fw_gap": mean_gap})

    return w, pd.DataFrame(hist), gamma_log, gap_log


def train_fw_vc(X_tr, y_tr, X_te, y_te,
                epochs=40, batch_size=64, R=5.0,
                beta=0.9, alpha=0.5,
                gamma_max=0.3, gamma_min=1e-3, eps=1e-8):
    """
    FW-VC: Frank-Wolfe with Variance-Controlled adaptive step size.

    Core innovation over SFW-Paper:
        Instead of a pre-defined schedule, the step size adapts to
        observed gradient magnitude via an exponential moving average (EMA):

            v_t   = beta * v_{t-1} + (1 - beta) * ||g_t||^2
            gamma_t = clip( alpha / sqrt(v_t),  gamma_min,  gamma_max )

    Intuition:
        Large gradients (high v_t) => small gamma_t (conservative step).
        Small gradients (low  v_t) => large gamma_t (more aggressive step).

    The convex combination update preserves L1 feasibility automatically:
        x_{t+1} = (1 - gamma_t) x_t + gamma_t s_t

    Parameters
    ----------
    beta      : float   EMA decay factor (0 < beta < 1)
    alpha     : float   Step size scale factor
    gamma_max : float   Upper clip on gamma_t (prevents instability)
    gamma_min : float   Lower clip on gamma_t (ensures Sigma gamma_t diverges)
    eps       : float   Numerical stabilizer for sqrt denominator
    """
    w         = np.zeros(X_tr.shape[1])
    v_ema     = 0.0
    hist      = []
    start     = time.time()
    steps     = max(1, X_tr.shape[0] // batch_size)
    gamma_log = []
    gap_log   = []

    for ep in range(epochs):
        ep_gammas, ep_gaps = [], []

        for _ in range(steps):
            g     = minibatch_grad(X_tr, y_tr, w, batch_size)
            v_ema = beta * v_ema + (1 - beta) * float(np.dot(g, g))
            gamma = float(np.clip(alpha / (np.sqrt(v_ema) + eps),
                                  gamma_min, gamma_max))
            s     = l1_lmo(g, R=R)
            gap   = frank_wolfe_gap(g, w, s)
            w     = sanitize((1.0 - gamma) * w + gamma * s)
            ep_gammas.append(gamma)
            ep_gaps.append(gap)

        mean_gamma = float(np.mean(ep_gammas))
        mean_gap   = float(np.mean(ep_gaps))
        gamma_log.append(mean_gamma)
        gap_log.append(mean_gap)
        _record(hist, ep, start, w, X_tr, y_tr, X_te, y_te,
                extra={"gamma": mean_gamma, "v_ema": v_ema, "fw_gap": mean_gap})

    return w, pd.DataFrame(hist), gamma_log, gap_log
