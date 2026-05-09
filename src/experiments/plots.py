"""
Plotting module for the FW-VC project.
All plots are saved to the plots/ directory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

COLORS = {
    "SGD":           "#e74c3c",
    "Adam":          "#3498db",
    "Projected SGD": "#2ecc71",
    "SFW-Paper":     "#f39c12",
    "FW-VC":         "#9b59b6",
}
STYLES = {
    "SGD":           "-",
    "Adam":          "--",
    "Projected SGD": "-.",
    "SFW-Paper":     "-",
    "FW-VC":         ":",
}


def _ax_plot(ax, histories, metric, ylabel, title):
    for name, hist in histories.items():
        if metric not in hist.columns:
            continue
        ax.plot(hist["epoch"], hist[metric],
                label=name, color=COLORS[name],
                linestyle=STYLES[name], linewidth=2.2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_main_dashboard(histories, gamma_logs, dataset_name):
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Frank-Wolfe Optimizers vs Baselines — {dataset_name}",
                 fontsize=15, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    _ax_plot(axes[0], histories, "train_loss", "Cross-Entropy Loss", "Training Loss")
    _ax_plot(axes[1], histories, "test_loss",  "Cross-Entropy Loss", "Test Loss")
    _ax_plot(axes[2], histories, "train_acc",  "Accuracy",           "Training Accuracy")
    _ax_plot(axes[3], histories, "test_acc",   "Accuracy",           "Test Accuracy")
    _ax_plot(axes[4], histories, "sparsity",   "Sparsity Ratio",     "Weight Sparsity")

    ax6 = axes[5]
    for name, gl in gamma_logs.items():
        ax6.plot(np.arange(1, len(gl) + 1), gl,
                 label=name, color=COLORS[name],
                 linestyle=STYLES[name], linewidth=2.2,
                 marker="o", markersize=3)
    ax6.set_xlabel("Epoch"); ax6.set_ylabel("Step Size γ")
    ax6.set_title("Frank-Wolfe Step Size", fontsize=12, fontweight="bold")
    ax6.legend(fontsize=9); ax6.grid(True, alpha=0.3)

    path = os.path.join(PLOTS_DIR, f"{dataset_name}_dashboard.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_sparsity_bar(final_df, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Final Sparsity & Accuracy — {dataset_name}",
                 fontsize=13, fontweight="bold")
    names  = final_df["optimizer"].tolist()
    colors = [COLORS[n] for n in names]

    for ax, metric, label in zip(axes,
                                  ["sparsity", "test_acc"],
                                  ["Sparsity Ratio (↑ sparser)", "Test Accuracy"]):
        bars = ax.bar(names, final_df[metric], color=colors,
                      edgecolor="black", linewidth=0.7)
        ax.set_ylabel(label, fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.tick_params(axis="x", rotation=20)
        for bar, val in zip(bars, final_df[metric]):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{dataset_name}_sparsity_bar.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_fw_gap(histories, dataset_name):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in ["SFW-Paper", "FW-VC"]:
        if name in histories and "fw_gap" in histories[name].columns:
            ax.plot(histories[name]["epoch"], histories[name]["fw_gap"],
                    label=name, color=COLORS[name],
                    linestyle=STYLES[name], linewidth=2.2)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Frank-Wolfe Gap")
    ax.set_title(f"FW Duality Gap — {dataset_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{dataset_name}_fw_gap.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_convergence_stability(histories, dataset_name, window=5):
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, hist in histories.items():
        series = pd.Series(hist["train_loss"].values).rolling(window).std()
        ax.plot(hist["epoch"], series, label=name,
                color=COLORS[name], linestyle=STYLES[name], linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Rolling Std of Train Loss (w={window})")
    ax.set_title(f"Convergence Stability — {dataset_name}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{dataset_name}_stability.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_sensitivity(sensitivity_df, dataset_name):
    params = sensitivity_df["param_name"].unique()
    fig, axes = plt.subplots(1, len(params), figsize=(5 * len(params), 4))
    if len(params) == 1:
        axes = [axes]
    fig.suptitle(f"FW-VC Hyperparameter Sensitivity — {dataset_name}",
                 fontsize=13, fontweight="bold")
    for ax, param in zip(axes, params):
        sub = sensitivity_df[sensitivity_df["param_name"] == param]
        ax.plot(sub["param_value"].astype(str), sub["test_acc"],
                marker="o", color=COLORS["FW-VC"], linewidth=2)
        ax.set_title(param, fontsize=11, fontweight="bold")
        ax.set_xlabel("Value"); ax.set_ylabel("Test Accuracy")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, f"{dataset_name}_sensitivity.png")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")
