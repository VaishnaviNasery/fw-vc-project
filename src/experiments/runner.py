"""
Experiment runner: orchestrates dataset loading, optimizer training,
metric collection, result saving, and sensitivity analysis.
"""

import os
import numpy as np
import pandas as pd

from src.data import load_digits_binary, load_breast_cancer_binary
from src.optimizers import (
    train_sgd, train_adam, train_projected_sgd,
    train_sfw_paper, train_fw_vc
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Dataset + hyperparameter configurations ────────────────────────────────
CONFIGS = {
    "digits": {
        "loader":      load_digits_binary,
        "epochs":      40,
        "batch_size":  64,
        "sgd_lr":      0.05,
        "adam_lr":     0.01,
        "psgd_lr":     0.05,
        "fw_vc": dict(R=3.0, beta=0.9, alpha=1.0,
                      gamma_max=0.1, gamma_min=1e-3, eps=1e-8),
    },
    "breast_cancer": {
        "loader":      load_breast_cancer_binary,
        "epochs":      40,
        "batch_size":  32,
        "sgd_lr":      0.05,
        "adam_lr":     0.01,
        "psgd_lr":     0.05,
        "fw_vc": dict(R=5.0, beta=0.95, alpha=2.0,
                      gamma_max=0.1, gamma_min=1e-3, eps=1e-8),
    },
}


def run_experiment(dataset_name):
    """
    Run all five optimizers on the given dataset and return results.

    Returns
    -------
    histories : dict[str, DataFrame]
    final_df  : DataFrame
    gamma_logs: dict[str, list]
    """
    print(f"\n{'='*70}\nEXPERIMENT: {dataset_name.upper()}\n{'='*70}")

    cfg = CONFIGS[dataset_name]
    X_tr, X_te, y_tr, y_te = cfg["loader"]()
    ep  = cfg["epochs"]
    bs  = cfg["batch_size"]
    R   = cfg["fw_vc"]["R"]

    histories  = {}
    gamma_logs = {}
    gap_logs   = {}

    print("\n[1/5] SGD")
    _, h = train_sgd(X_tr, y_tr, X_te, y_te,
                     epochs=ep, batch_size=bs, lr=cfg["sgd_lr"])
    histories["SGD"] = h

    print("[2/5] Adam")
    _, h = train_adam(X_tr, y_tr, X_te, y_te,
                      epochs=ep, batch_size=bs, lr=cfg["adam_lr"])
    histories["Adam"] = h

    print("[3/5] Projected SGD")
    _, h = train_projected_sgd(X_tr, y_tr, X_te, y_te,
                                epochs=ep, batch_size=bs,
                                lr=cfg["psgd_lr"], R=R)
    histories["Projected SGD"] = h

    print("[4/5] SFW-Paper")
    _, h, gl, gapl = train_sfw_paper(X_tr, y_tr, X_te, y_te,
                                      epochs=ep, batch_size=bs,
                                      R=R, gamma_rule="open_loop")
    histories["SFW-Paper"]  = h
    gamma_logs["SFW-Paper"] = gl
    gap_logs["SFW-Paper"]   = gapl

    print("[5/5] FW-VC")
    _, h, gl, gapl = train_fw_vc(X_tr, y_tr, X_te, y_te,
                                  epochs=ep, batch_size=bs,
                                  **cfg["fw_vc"])
    histories["FW-VC"]  = h
    gamma_logs["FW-VC"] = gl
    gap_logs["FW-VC"]   = gapl

    # Build final summary table
    rows = []
    for name, hist in histories.items():
        row = hist.iloc[-1].copy().to_dict()
        row["optimizer"] = name
        rows.append(row)
    final_df = pd.DataFrame(rows)

    ordered_cols = ["optimizer", "train_loss", "test_loss", "train_acc",
                    "test_acc", "sparsity", "nnz", "l1_norm", "l2_norm",
                    "runtime", "gamma", "fw_gap"]
    final_df = final_df[[c for c in ordered_cols if c in final_df.columns]]

    print(f"\n{'─'*80}\nFinal Results — {dataset_name}\n{'─'*80}")
    print(final_df.to_string(index=False))

    # Save
    final_df.to_csv(os.path.join(RESULTS_DIR, f"{dataset_name}_final.csv"), index=False)
    for name, hist in histories.items():
        safe = name.lower().replace(" ", "_").replace("-", "_")
        hist.to_csv(
            os.path.join(RESULTS_DIR, f"{dataset_name}_{safe}_history.csv"),
            index=False)

    return histories, final_df, gamma_logs, gap_logs, X_tr, y_tr, X_te, y_te


def sensitivity_analysis(X_tr, y_tr, X_te, y_te,
                         dataset_name, base_cfg, epochs=30):
    """
    One-at-a-time hyperparameter sensitivity sweep for FW-VC.
    Returns a DataFrame of (param_name, param_value, test_acc, sparsity).
    """
    print(f"\n  Sensitivity analysis on {dataset_name}...")
    sweep = {
        "R (L1 radius)": ("R",        [1.0, 2.0, 5.0, 10.0, 20.0]),
        "beta (EMA)":    ("beta",     [0.7, 0.8, 0.9, 0.95, 0.99]),
        "alpha (scale)": ("alpha",    [0.1, 0.3, 0.5, 1.0, 2.0]),
        "gamma_max":     ("gamma_max",[0.05, 0.1, 0.2, 0.3, 0.5]),
    }
    rows = []
    for param_name, (key, values) in sweep.items():
        for val in values:
            cfg = dict(base_cfg)
            cfg[key] = val
            _, hist, _, _ = train_fw_vc(
                X_tr, y_tr, X_te, y_te,
                epochs=epochs,
                batch_size=cfg.get("batch_size", 64),
                R=cfg["R"], beta=cfg["beta"], alpha=cfg["alpha"],
                gamma_max=cfg["gamma_max"], gamma_min=cfg["gamma_min"],
                eps=cfg["eps"])
            rows.append({
                "param_name":  param_name,
                "param_value": val,
                "test_acc":    hist["test_acc"].iloc[-1],
                "test_loss":   hist["test_loss"].iloc[-1],
                "sparsity":    hist["sparsity"].iloc[-1],
                "nnz":         hist["nnz"].iloc[-1],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f"{dataset_name}_sensitivity.csv"), index=False)
    return df
