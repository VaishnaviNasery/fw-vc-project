"""
Entry point for the FW-VC project.

Runs all five optimizers on both datasets, generates plots,
performs sensitivity analysis, and saves all results to disk.

Usage:
    python run.py
"""

import numpy as np
from src.experiments.runner import run_experiment, sensitivity_analysis, CONFIGS
from src.experiments.plots import (
    plot_main_dashboard, plot_sparsity_bar, plot_fw_gap,
    plot_convergence_stability, plot_sensitivity
)

np.random.seed(42)

DATASETS = ["digits", "breast_cancer"]

if __name__ == "__main__":
    all_results = {}

    for dataset in DATASETS:
        print(f"\n{'='*70}\n  RUNNING: {dataset.upper()}\n{'='*70}")

        (histories, final_df, gamma_logs,
         gap_logs, X_tr, y_tr, X_te, y_te) = run_experiment(dataset)

        print(f"\n  Generating plots for {dataset}...")
        plot_main_dashboard(histories, gamma_logs, dataset)
        plot_sparsity_bar(final_df, dataset)
        plot_fw_gap(histories, dataset)
        plot_convergence_stability(histories, dataset)

        base_cfg = {**CONFIGS[dataset]["fw_vc"],
                    "batch_size": CONFIGS[dataset]["batch_size"]}
        sens_df = sensitivity_analysis(
            X_tr, y_tr, X_te, y_te,
            dataset_name=dataset,
            base_cfg=base_cfg,
            epochs=30)
        plot_sensitivity(sens_df, dataset)

        all_results[dataset] = final_df

    print(f"\n{'='*70}\n  COMBINED SUMMARY\n{'='*70}")
    for ds, df in all_results.items():
        print(f"\n  {ds.upper()}")
        keep = ["optimizer", "test_loss", "test_acc",
                "sparsity", "nnz", "l1_norm", "runtime"]
        keep = [c for c in keep if c in df.columns]
        print(df[keep].to_string(index=False))

    print("\nDone. Results saved in ./results/  |  Plots saved in ./plots/")
