"""
Frank-Wolfe Variance-Controlled Sparse Stochastic Gradient Descent
for Efficient Machine Learning Optimization

COMPSCI 651 – Optimization Final Project
Authors: Jiewen Luo, Vaishnavi Nasery, Rufina Lourdes

Paper-based extension:
    Pokutta, Spiegel, Zimmer. Deep Neural Network Training with Frank-Wolfe.
    arXiv:2010.07243

Datasets:
    1. sklearn Digits: digit 0 vs. rest
    2. sklearn Breast Cancer: malignant vs. benign
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Directories
# ─────────────────────────────────────────────
SAVE_DIR_RESULTS = "results"
SAVE_DIR_PLOTS = "plots"
os.makedirs(SAVE_DIR_RESULTS, exist_ok=True)
os.makedirs(SAVE_DIR_PLOTS, exist_ok=True)

np.random.seed(42)

# ─────────────────────────────────────────────
# Numerical Utilities
# ─────────────────────────────────────────────
def sanitize(a, pos_cap=1e8, neg_cap=-1e8):
    return np.nan_to_num(
        np.asarray(a, dtype=np.float64),
        nan=0.0,
        posinf=pos_cap,
        neginf=neg_cap
    )


def sigmoid(z):
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def logistic_loss_and_grad(X, y, w):
    """Binary cross-entropy loss and gradient."""
    n = X.shape[0]
    p = sigmoid(X @ w)
    eps = 1e-12
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    grad = (X.T @ (p - y)) / n
    return float(loss), sanitize(grad)


def clip_gradient(g, max_norm=5.0):
    norm = np.linalg.norm(g)
    if norm > max_norm:
        return g * (max_norm / (norm + 1e-12))
    return g


def minibatch_grad(X, y, w, batch_size):
    idx = np.random.choice(X.shape[0], min(batch_size, X.shape[0]), replace=False)
    _, g = logistic_loss_and_grad(X[idx], y[idx], w)
    return clip_gradient(g, max_norm=5.0)


def predict(X, w):
    return (sigmoid(X @ sanitize(w)) >= 0.5).astype(int)


def sparsity_ratio(w, threshold=1e-6):
    return float(np.sum(np.abs(w) <= threshold) / len(w))


def nnz(w, threshold=1e-6):
    return int(np.sum(np.abs(w) > threshold))


def compute_metrics(X_tr, y_tr, X_te, y_te, w):
    train_loss, _ = logistic_loss_and_grad(X_tr, y_tr, w)
    test_loss, _ = logistic_loss_and_grad(X_te, y_te, w)

    return {
        "train_loss": train_loss,
        "test_loss": test_loss,
        "train_acc": accuracy_score(y_tr, predict(X_tr, w)),
        "test_acc": accuracy_score(y_te, predict(X_te, w)),
        "sparsity": sparsity_ratio(w),
        "nnz": nnz(w),
        "l1_norm": float(np.linalg.norm(w, 1)),
        "l2_norm": float(np.linalg.norm(w, 2)),
    }


# ─────────────────────────────────────────────
# Constrained Optimization Primitives
# ─────────────────────────────────────────────
def project_onto_l1_ball(v, R=1.0):
    """Euclidean projection onto {x : ||x||_1 <= R}."""
    v = sanitize(v)
    if np.linalg.norm(v, 1) <= R:
        return v.copy()

    u = np.abs(v)
    s = np.sort(u)[::-1]
    cssv = np.cumsum(s)
    rho = np.where(s * np.arange(1, len(s) + 1) > (cssv - R))[0][-1]
    theta = (cssv[rho] - R) / (rho + 1.0)
    return sanitize(np.sign(v) * np.maximum(u - theta, 0.0))


def l1_lmo(g, R):
    """
    Linear Minimization Oracle over L1 ball:
        s = argmin_{||s||_1 <= R} <g, s>
    """
    g = sanitize(g)
    s = np.zeros_like(g)
    j = np.argmax(np.abs(g))
    if g[j] != 0:
        s[j] = -R * np.sign(g[j])
    return s


def frank_wolfe_gap(g, w, s):
    """
    FW gap:
        gap = <g, w - s>
    Smaller gap means closer to first-order stationarity under the constraint.
    """
    return float(np.dot(g, w - s))


# ─────────────────────────────────────────────
# Dataset Loaders
# ─────────────────────────────────────────────
def _prepare(X_tr, X_te, y_tr, y_te, name):
    scaler = StandardScaler()
    X_tr = sanitize(scaler.fit_transform(X_tr))
    X_te = sanitize(scaler.transform(X_te))

    # Bias term is also constrained here. If you want unconstrained bias,
    # separate it from w and do not include it in LMO/projection.
    X_tr = np.hstack([X_tr, np.ones((X_tr.shape[0], 1))])
    X_te = np.hstack([X_te, np.ones((X_te.shape[0], 1))])

    print(f"\nLoaded {name} | train: {X_tr.shape} test: {X_te.shape}")
    print(f"Class balance train={np.mean(y_tr):.3f}, test={np.mean(y_te):.3f}")

    return X_tr, X_te, y_tr.astype(int), y_te.astype(int)


def load_digits_binary(test_size=0.2, random_state=42):
    ds = load_digits()
    X = ds.data.astype(np.float64)
    y = (ds.target == 0).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return _prepare(
        X_tr, X_te, y_tr, y_te,
        "Digits | task: digit 0 vs. rest"
    )


def load_breast_cancer_binary(test_size=0.2, random_state=42):
    ds = load_breast_cancer()
    X = ds.data.astype(np.float64)
    y = 1 - ds.target.astype(int)  # malignant=1, benign=0

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return _prepare(
        X_tr, X_te, y_tr, y_te,
        "Breast Cancer | task: malignant vs. benign"
    )


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
def _record(history, epoch, start, w, X_tr, y_tr, X_te, y_te, extra=None):
    m = compute_metrics(X_tr, y_tr, X_te, y_te, w)
    m["epoch"] = epoch + 1
    m["runtime"] = time.time() - start
    if extra:
        m.update(extra)
    history.append(m)


# ─────────────────────────────────────────────
# Optimizers
# ─────────────────────────────────────────────
def train_sgd(X_tr, y_tr, X_te, y_te, epochs=40, batch_size=64, lr=0.05):
    w = np.zeros(X_tr.shape[1])
    hist = []
    start = time.time()
    steps = max(1, X_tr.shape[0] // batch_size)

    for ep in range(epochs):
        for _ in range(steps):
            g = minibatch_grad(X_tr, y_tr, w, batch_size)
            w = sanitize(w - lr * g)

        _record(hist, ep, start, w, X_tr, y_tr, X_te, y_te)

    return w, pd.DataFrame(hist)


def train_adam(
    X_tr, y_tr, X_te, y_te,
    epochs=40,
    batch_size=64,
    lr=0.01,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8
):
    d = X_tr.shape[1]
    w = np.zeros(d)
    m = np.zeros(d)
    v = np.zeros(d)
    t = 0

    hist = []
    start = time.time()
    steps = max(1, X_tr.shape[0] // batch_size)

    for ep in range(epochs):
        for _ in range(steps):
            t += 1
            g = minibatch_grad(X_tr, y_tr, w, batch_size)

            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g ** 2)

            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)

            step = lr * m_hat / (np.sqrt(v_hat) + eps)
            step = clip_gradient(step, max_norm=1.0)

            w = sanitize(w - step)

        _record(hist, ep, start, w, X_tr, y_tr, X_te, y_te)

    return w, pd.DataFrame(hist)


def train_projected_sgd(
    X_tr, y_tr, X_te, y_te,
    epochs=40,
    batch_size=64,
    lr=0.05,
    R=5.0
):
    w = np.zeros(X_tr.shape[1])
    hist = []
    start = time.time()
    steps = max(1, X_tr.shape[0] // batch_size)

    for ep in range(epochs):
        for _ in range(steps):
            g = minibatch_grad(X_tr, y_tr, w, batch_size)
            w = project_onto_l1_ball(w - lr * g, R=R)

        _record(hist, ep, start, w, X_tr, y_tr, X_te, y_te)

    return w, pd.DataFrame(hist)


def train_sfw_paper(
    X_tr, y_tr, X_te, y_te,
    epochs=40,
    batch_size=64,
    R=5.0,
    gamma_rule="open_loop",
    gamma_const=0.1
):
    """
    Paper-style Stochastic Frank-Wolfe.

    Update:
        g_t = stochastic minibatch gradient
        s_t = argmin_{s in C} <g_t, s>
        w_{t+1} = (1 - gamma_t) w_t + gamma_t s_t

    Supported gamma rules:
        open_loop: gamma_t = 2 / (t + 2)
        constant:  gamma_t = gamma_const

    This is projection-free. The feasible set is the L1 ball.
    """
    w = np.zeros(X_tr.shape[1])
    hist = []
    start = time.time()
    steps = max(1, X_tr.shape[0] // batch_size)

    global_t = 0
    gamma_log = []
    gap_log = []

    for ep in range(epochs):
        ep_gammas = []
        ep_gaps = []

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

            w = sanitize((1.0 - gamma) * w + gamma * s)

            ep_gammas.append(gamma)
            ep_gaps.append(gap)

        mean_gamma = float(np.mean(ep_gammas))
        mean_gap = float(np.mean(ep_gaps))

        gamma_log.append(mean_gamma)
        gap_log.append(mean_gap)

        _record(
            hist, ep, start, w, X_tr, y_tr, X_te, y_te,
            extra={
                "gamma": mean_gamma,
                "fw_gap": mean_gap,
            }
        )

    return w, pd.DataFrame(hist), gamma_log, gap_log


def train_fw_vc(
    X_tr, y_tr, X_te, y_te,
    epochs=40,
    batch_size=64,
    R=5.0,
    beta=0.9,
    alpha=0.5,
    gamma_max=0.3,
    gamma_min=1e-3,
    eps=1e-8
):
    """
    FW-VC: your variance-controlled Frank-Wolfe variant.

    Difference from paper-style SFW:
        Paper-style SFW uses a pre-defined/open-loop step-size.
        FW-VC uses EMA of gradient norm to adapt gamma_t.
    """
    w = np.zeros(X_tr.shape[1])
    v_ema = 0.0
    hist = []
    start = time.time()
    steps = max(1, X_tr.shape[0] // batch_size)

    gamma_log = []
    gap_log = []

    for ep in range(epochs):
        ep_gammas = []
        ep_gaps = []

        for _ in range(steps):
            g = minibatch_grad(X_tr, y_tr, w, batch_size)

            v_ema = beta * v_ema + (1 - beta) * float(np.dot(g, g))
            gamma = float(np.clip(
                alpha / (np.sqrt(v_ema) + eps),
                gamma_min,
                gamma_max
            ))

            s = l1_lmo(g, R=R)
            gap = frank_wolfe_gap(g, w, s)

            w = sanitize((1.0 - gamma) * w + gamma * s)

            ep_gammas.append(gamma)
            ep_gaps.append(gap)

        mean_gamma = float(np.mean(ep_gammas))
        mean_gap = float(np.mean(ep_gaps))

        gamma_log.append(mean_gamma)
        gap_log.append(mean_gap)

        _record(
            hist, ep, start, w, X_tr, y_tr, X_te, y_te,
            extra={
                "gamma": mean_gamma,
                "v_ema": v_ema,
                "fw_gap": mean_gap,
            }
        )

    return w, pd.DataFrame(hist), gamma_log, gap_log


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
COLORS = {
    "SGD": "#e74c3c",
    "Adam": "#3498db",
    "Projected SGD": "#2ecc71",
    "SFW-Paper": "#f39c12",
    "FW-VC": "#9b59b6",
}

STYLES = {
    "SGD": "-",
    "Adam": "--",
    "Projected SGD": "-.",
    "SFW-Paper": "-",
    "FW-VC": ":",
}


def _ax_plot(ax, histories, metric, ylabel, title):
    for name, hist in histories.items():
        if metric not in hist.columns:
            continue

        ax.plot(
            hist["epoch"],
            hist[metric],
            label=name,
            color=COLORS[name],
            linestyle=STYLES[name],
            linewidth=2.2
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_main_dashboard(histories, gamma_logs, dataset_name, save_dir):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"Frank-Wolfe Optimizers vs Baselines — {dataset_name}",
        fontsize=15,
        fontweight="bold",
        y=1.01
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0])
    ax5 = fig.add_subplot(gs[1, 1])
    ax6 = fig.add_subplot(gs[1, 2])

    _ax_plot(ax1, histories, "train_loss", "Cross-Entropy Loss", "Training Loss")
    _ax_plot(ax2, histories, "test_loss", "Cross-Entropy Loss", "Test Loss")
    _ax_plot(ax3, histories, "train_acc", "Accuracy", "Training Accuracy")
    _ax_plot(ax4, histories, "test_acc", "Accuracy", "Test Accuracy")
    _ax_plot(ax5, histories, "sparsity", "Sparsity Ratio", "Weight Sparsity")

    for name, gamma_log in gamma_logs.items():
        epochs = np.arange(1, len(gamma_log) + 1)
        ax6.plot(
            epochs,
            gamma_log,
            label=name,
            color=COLORS[name],
            linestyle=STYLES[name],
            linewidth=2.2,
            marker="o",
            markersize=3
        )

    ax6.set_xlabel("Epoch", fontsize=11)
    ax6.set_ylabel("Step Size gamma", fontsize=11)
    ax6.set_title("Frank-Wolfe Step Size", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"{dataset_name}_dashboard.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dashboard -> {path}")


def plot_sparsity_bar(final_df, dataset_name, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(
        f"Final Sparsity & Accuracy — {dataset_name}",
        fontsize=13,
        fontweight="bold"
    )

    names = final_df["optimizer"].tolist()
    colors = [COLORS[n] for n in names]

    ax = axes[0]
    bars = ax.bar(names, final_df["sparsity"], color=colors, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Sparsity Ratio", fontsize=11)
    ax.set_title("Weight Sparsity", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)

    for bar, val in zip(bars, final_df["sparsity"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax = axes[1]
    bars = ax.bar(names, final_df["test_acc"], color=colors, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title("Test Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=20)

    for bar, val in zip(bars, final_df["test_acc"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    fig.tight_layout()
    path = os.path.join(save_dir, f"{dataset_name}_sparsity_accuracy_bar.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved bar chart -> {path}")


def plot_fw_gap(histories, dataset_name, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))

    for name in ["SFW-Paper", "FW-VC"]:
        if name in histories and "fw_gap" in histories[name].columns:
            ax.plot(
                histories[name]["epoch"],
                histories[name]["fw_gap"],
                label=name,
                color=COLORS[name],
                linestyle=STYLES[name],
                linewidth=2.2
            )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Frank-Wolfe Gap", fontsize=11)
    ax.set_title(f"FW Gap — {dataset_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, f"{dataset_name}_fw_gap.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved FW gap plot -> {path}")


def plot_convergence_stability(histories, dataset_name, save_dir):
    window = 5
    fig, ax = plt.subplots(figsize=(8, 4))

    for name, hist in histories.items():
        series = pd.Series(hist["train_loss"].values).rolling(window).std()
        ax.plot(
            hist["epoch"],
            series,
            label=name,
            color=COLORS[name],
            linestyle=STYLES[name],
            linewidth=2
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(f"Rolling Std of Train Loss, window={window}", fontsize=11)
    ax.set_title(f"Convergence Stability — {dataset_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, f"{dataset_name}_stability.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stability plot -> {path}")


def plot_sensitivity(sensitivity_df, dataset_name, save_dir):
    params = sensitivity_df["param_name"].unique()
    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4), sharey=False)

    if len(params) == 1:
        axes = [axes]

    fig.suptitle(
        f"FW-VC Hyperparameter Sensitivity — {dataset_name}",
        fontsize=13,
        fontweight="bold"
    )

    for ax, param in zip(axes, params):
        sub = sensitivity_df[sensitivity_df["param_name"] == param]

        ax.plot(
            sub["param_value"].astype(str),
            sub["test_acc"],
            marker="o",
            color=COLORS["FW-VC"],
            linewidth=2
        )

        ax.set_title(param, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Test Accuracy", fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(save_dir, f"{dataset_name}_sensitivity.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sensitivity plot -> {path}")


# ─────────────────────────────────────────────
# Sensitivity Analysis
# ─────────────────────────────────────────────
def sensitivity_analysis(
    X_tr, y_tr, X_te, y_te,
    dataset_name,
    save_dir,
    base_cfg,
    epochs=30
):
    print(f"\nRunning FW-VC sensitivity analysis on {dataset_name}...")
    rows = []

    sweep = {
        "R (L1 radius)": ("R", [1.0, 2.0, 5.0, 10.0, 20.0]),
        "beta (EMA)": ("beta", [0.7, 0.8, 0.9, 0.95, 0.99]),
        "alpha (scale)": ("alpha", [0.1, 0.3, 0.5, 1.0, 2.0]),
        "gamma_max": ("gamma_max", [0.05, 0.1, 0.2, 0.3, 0.5]),
    }

    for param_name, (key, values) in sweep.items():
        for val in values:
            cfg = dict(base_cfg)
            cfg[key] = val

            _, hist, _, _ = train_fw_vc(
                X_tr, y_tr, X_te, y_te,
                epochs=epochs,
                batch_size=cfg["batch_size"],
                R=cfg["R"],
                beta=cfg["beta"],
                alpha=cfg["alpha"],
                gamma_max=cfg["gamma_max"],
                gamma_min=cfg["gamma_min"],
                eps=cfg["eps"]
            )

            rows.append({
                "param_name": param_name,
                "param_value": val,
                "test_acc": hist["test_acc"].iloc[-1],
                "test_loss": hist["test_loss"].iloc[-1],
                "sparsity": hist["sparsity"].iloc[-1],
                "nnz": hist["nnz"].iloc[-1],
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, f"{dataset_name}_sensitivity.csv"), index=False)
    plot_sensitivity(df, dataset_name, save_dir)
    return df


# ─────────────────────────────────────────────
# Experiment Runner
# ─────────────────────────────────────────────
def run_experiment(dataset_name="digits"):
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT: {dataset_name.upper()}")
    print(f"{'=' * 70}")

    if dataset_name == "digits":
        X_tr, X_te, y_tr, y_te = load_digits_binary()

        cfg = dict(
            R=3.0,
            beta=0.9,
            alpha=1.0,
            gamma_max=0.1,
            gamma_min=1e-3,
            eps=1e-8,
            batch_size=64
        )

        sgd_lr = 0.05
        adam_lr = 0.01
        psgd_lr = 0.05
        epochs = 40

    elif dataset_name == "breast_cancer":
        X_tr, X_te, y_tr, y_te = load_breast_cancer_binary()

        cfg = dict(
            R=5.0,
            beta=0.95,
            alpha=2.0,
            gamma_max=0.1,
            gamma_min=1e-3,
            eps=1e-8,
            batch_size=32
        )

        sgd_lr = 0.05
        adam_lr = 0.01
        psgd_lr = 0.05
        epochs = 40

    else:
        raise ValueError("dataset_name must be 'digits' or 'breast_cancer'")

    batch_size = cfg["batch_size"]
    tag = dataset_name

    histories = {}
    gamma_logs = {}
    gap_logs = {}

    print("\n[1/5] SGD ...")
    _, hist_sgd = train_sgd(
        X_tr, y_tr, X_te, y_te,
        epochs=epochs,
        batch_size=batch_size,
        lr=sgd_lr
    )
    histories["SGD"] = hist_sgd

    print("[2/5] Adam ...")
    _, hist_adam = train_adam(
        X_tr, y_tr, X_te, y_te,
        epochs=epochs,
        batch_size=batch_size,
        lr=adam_lr
    )
    histories["Adam"] = hist_adam

    print("[3/5] Projected SGD ...")
    _, hist_psgd = train_projected_sgd(
        X_tr, y_tr, X_te, y_te,
        epochs=epochs,
        batch_size=batch_size,
        lr=psgd_lr,
        R=cfg["R"]
    )
    histories["Projected SGD"] = hist_psgd

    print("[4/5] SFW-Paper ...")
    _, hist_sfw, sfw_gamma_log, sfw_gap_log = train_sfw_paper(
        X_tr, y_tr, X_te, y_te,
        epochs=epochs,
        batch_size=batch_size,
        R=cfg["R"],
        gamma_rule="open_loop"
    )
    histories["SFW-Paper"] = hist_sfw
    gamma_logs["SFW-Paper"] = sfw_gamma_log
    gap_logs["SFW-Paper"] = sfw_gap_log

    print("[5/5] FW-VC ...")
    _, hist_fw, fw_gamma_log, fw_gap_log = train_fw_vc(
        X_tr, y_tr, X_te, y_te,
        epochs=epochs,
        batch_size=batch_size,
        R=cfg["R"],
        beta=cfg["beta"],
        alpha=cfg["alpha"],
        gamma_max=cfg["gamma_max"],
        gamma_min=cfg["gamma_min"],
        eps=cfg["eps"]
    )
    histories["FW-VC"] = hist_fw
    gamma_logs["FW-VC"] = fw_gamma_log
    gap_logs["FW-VC"] = fw_gap_log

    # Final comparison table
    final_rows = []
    for name, hist in histories.items():
        row = hist.iloc[-1].copy().to_dict()
        row["optimizer"] = name
        final_rows.append(row)

    final_df = pd.DataFrame(final_rows)

    cols = [
        "optimizer",
        "train_loss",
        "test_loss",
        "train_acc",
        "test_acc",
        "sparsity",
        "nnz",
        "l1_norm",
        "l2_norm",
        "runtime",
        "gamma",
        "fw_gap",
    ]

    final_df = final_df[[c for c in cols if c in final_df.columns]]

    print(f"\n{'─' * 80}")
    print(f"Final Results — {dataset_name}")
    print(f"{'─' * 80}")
    print(final_df.to_string(index=False))

    # Save CSVs
    final_df.to_csv(
        os.path.join(SAVE_DIR_RESULTS, f"{tag}_final.csv"),
        index=False
    )

    for name, hist in histories.items():
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        hist.to_csv(
            os.path.join(SAVE_DIR_RESULTS, f"{tag}_{safe_name}_history.csv"),
            index=False
        )

    # Plots
    print("\nGenerating plots...")
    plot_main_dashboard(histories, gamma_logs, tag, SAVE_DIR_PLOTS)
    plot_sparsity_bar(final_df, tag, SAVE_DIR_PLOTS)
    plot_convergence_stability(histories, tag, SAVE_DIR_PLOTS)
    plot_fw_gap(histories, tag, SAVE_DIR_PLOTS)

    # Sensitivity only for FW-VC
    sens_df = sensitivity_analysis(
        X_tr, y_tr, X_te, y_te,
        dataset_name=tag,
        save_dir=SAVE_DIR_PLOTS,
        base_cfg=cfg,
        epochs=30
    )

    return histories, final_df, gamma_logs, sens_df


# ─────────────────────────────────────────────
# Cross-dataset summary
# ─────────────────────────────────────────────
def print_combined_summary(results):
    print(f"\n{'=' * 80}")
    print("COMBINED SUMMARY ACROSS DATASETS")
    print(f"{'=' * 80}")

    for ds, (_, final_df, _, _) in results.items():
        print(f"\n── {ds.upper()} ──")
        keep = [
            "optimizer",
            "test_loss",
            "test_acc",
            "sparsity",
            "nnz",
            "l1_norm",
            "runtime",
        ]
        keep = [c for c in keep if c in final_df.columns]
        print(final_df[keep].to_string(index=False))


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    all_results = {}

    print("\n===== DATASET 1: Digits =====")
    all_results["digits"] = run_experiment("digits")

    print("\n===== DATASET 2: Breast Cancer =====")
    all_results["breast_cancer"] = run_experiment("breast_cancer")

    print_combined_summary(all_results)

    print("\nDone. Results saved in ./results/, plots saved in ./plots/")