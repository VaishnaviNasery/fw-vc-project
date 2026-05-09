# FW-VC: Frank-Wolfe with Variance-Controlled Step Size

**COMPSCI 651 – Optimization | University of Massachusetts Amherst**  
Jiewen Luo · Vaishnavi Nasery · Rufina Lourdes Rajesh

## Overview

This repository contains the implementation and experiments for our course project on
**projection-free sparse stochastic optimization**. We propose **FW-VC**, a variant of
the Frank-Wolfe algorithm that replaces the standard open-loop step size schedule with
an adaptive, variance-controlled rule based on an exponential moving average (EMA) of
observed gradient norms. The method is evaluated on L1-constrained binary logistic
regression tasks against SGD, Adam, Projected SGD, and the classical Stochastic
Frank-Wolfe (SFW-Paper) baseline.

## Project Structure

```
fw_vc_project/
├── run.py                        # Main entry point — runs all experiments
├── requirements.txt
├── results/                      # CSV outputs (auto-generated)
├── plots/                        # PNG figures (auto-generated)
└── src/
    ├── data/
    │   └── datasets.py           # Dataset loaders (Digits, Breast Cancer)
    ├── optimizers/
    │   ├── constraints.py        # L1 projection, LMO, FW gap
    │   ├── baselines.py          # SGD, Adam, Projected SGD
    │   └── frank_wolfe.py        # SFW-Paper and FW-VC (proposed method)
    ├── experiments/
    │   ├── runner.py             # Experiment orchestration + sensitivity analysis
    │   └── plots.py              # All visualization functions
    └── utils/
        └── numerical.py          # Shared numerical utilities
```

## Setup

```bash
git clone https://github.com/<your-username>/fw_vc_project.git
cd fw_vc_project
pip install -r requirements.txt
python run.py
```

All results are saved to `results/` and all plots to `plots/`.

## Method

**FW-VC** solves the constrained stochastic optimization problem:

```
min  f(x) = E[f(x, xi)]    subject to  ||x||_1 <= R
```

At each iteration t:
1. Compute stochastic gradient `g_t` on a mini-batch
2. Update EMA of gradient norm: `v_t = beta * v_{t-1} + (1 - beta) * ||g_t||^2`
3. Compute adaptive step size: `gamma_t = clip(alpha / sqrt(v_t), gamma_min, gamma_max)`
4. Run LMO: `s_t = -R * e_j`,  where `j = argmax_i |g_{t,i}|`
5. Update: `x_{t+1} = (1 - gamma_t) * x_t + gamma_t * s_t`

The convex combination in step 5 preserves L1 feasibility automatically,
requiring no projection.

## Datasets

| Dataset         | Task                      | Features | Samples |
|----------------|---------------------------|----------|---------|
| Digits (MNIST-proxy) | Digit 0 vs. rest   | 64       | 1,797   |
| Breast Cancer   | Malignant vs. benign      | 30       | 569     |

## Key Results

**Digits dataset:**

| Method        | Test Acc | Sparsity | NNZ/65 |
|---------------|----------|----------|--------|
| SGD           | 99.4%    | 4.6%     | 62     |
| Adam          | 99.7%    | 4.6%     | 62     |
| Projected SGD | 90.8%    | 73.8%    | 17     |
| SFW-Paper     | 98.3%    | 41.5%    | 38     |
| **FW-VC**     | **93.1%**| **75.4%**| **16** |

FW-VC achieves the highest sparsity while maintaining competitive accuracy over the other constrained method (Projected SGD).

## Requirements

```
numpy
pandas
matplotlib
scikit-learn
```

## Citation

If you use this code, please cite:

```
Jaggi, M. (2013). Revisiting Frank-Wolfe: Projection-Free Sparse Convex Optimization. ICML.
Hazan, E., & Kale, S. (2012). Projection-free Online Learning. ICML.
```
