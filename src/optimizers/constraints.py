"""
Constrained optimization primitives for L1-ball problems.

Provides:
  - Euclidean projection onto the L1 ball (for Projected SGD)
  - Linear Minimization Oracle over the L1 ball (for Frank-Wolfe methods)
  - Frank-Wolfe gap computation (stationarity measure)
"""

import numpy as np
from src.utils.numerical import sanitize


def project_onto_l1_ball(v, R=1.0):
    """
    Euclidean projection onto { x : ||x||_1 <= R }.

    Uses the O(d log d) isotonic soft-thresholding algorithm
    of Duchi et al. (2008).

    Parameters
    ----------
    v : ndarray
        Point to project.
    R : float
        L1 ball radius.

    Returns
    -------
    ndarray
        Projected point satisfying ||result||_1 <= R.
    """
    v = sanitize(v)
    if np.linalg.norm(v, 1) <= R:
        return v.copy()
    u    = np.abs(v)
    s    = np.sort(u)[::-1]
    cssv = np.cumsum(s)
    rho  = np.where(s * np.arange(1, len(s) + 1) > (cssv - R))[0][-1]
    theta = (cssv[rho] - R) / (rho + 1.0)
    return sanitize(np.sign(v) * np.maximum(u - theta, 0.0))


def l1_lmo(g, R):
    """
    Linear Minimization Oracle over the L1 ball:
        s* = argmin_{||s||_1 <= R} <g, s>

    Closed-form solution: place all budget R on the coordinate
    with the largest absolute gradient value, pointing opposite
    to the gradient direction. This is an O(d) operation.

    Parameters
    ----------
    g : ndarray
        Gradient vector.
    R : float
        L1 ball radius.

    Returns
    -------
    ndarray
        Sparse direction s* (one nonzero entry).
    """
    g = sanitize(g)
    s = np.zeros_like(g)
    j = np.argmax(np.abs(g))
    if g[j] != 0:
        s[j] = -R * np.sign(g[j])
    return s


def frank_wolfe_gap(g, w, s):
    """
    Frank-Wolfe duality gap (primal gap proxy):
        gap_t = <g_t, w_t - s_t>

    A gap of zero certifies first-order stationarity under
    the constraint. Smaller gap => closer to optimality.

    Parameters
    ----------
    g : ndarray  gradient at current iterate
    w : ndarray  current iterate
    s : ndarray  LMO solution

    Returns
    -------
    float
    """
    return float(np.dot(g, w - s))
