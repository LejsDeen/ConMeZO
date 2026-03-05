# Synthetic Quadratic Benchmark

This directory contains a lightweight synthetic experiment that compares **MeZO** and **ConMeZO** on an ill-conditioned quadratic objective. It serves as a controlled setting to illustrate the convergence advantage of cone-constrained perturbations without the overhead of LLM fine-tuning.

## Problem Setup

The objective is a general quadratic $f(x) = \tfrac{1}{2} x^\top A x$ where $A$ is a positive-definite diagonal matrix with geometrically spaced eigenvalues and a condition number of $d^3$ ($d = 1000$).

## Files

| File | Description |
|:---|:---|
| `algorithms.py` | MeZO and ConMeZO optimizer implementations |
| `problems.py` | Quadratic problem definitions |
| `run_quadratic_simulation.py` | Main script: runs multi-trial HPO and long-run evaluation |
| `requirements.txt` | Python dependencies (NumPy only) |

## Usage

```bash
pip install -r requirements.txt

# Quick run with default hyperparameters (η=0.01, α_m=0.01, θ=1.30)
python run_quadratic_simulation.py

# Full two-stage hyperparameter search (slower)
python run_quadratic_simulation.py --full-hpo
```

### Output

The script saves aggregated convergence curves to `mezo_conmezo_longrun_curves.npz` (override with `--save-data <path>`). The `.npz` file contains:

| Key | Description |
|:---|:---|
| `mean_mezo` / `se_mezo` | Mean and standard error of $f(x_t)$ for MeZO |
| `mean_con` / `se_con` | Mean and standard error of $f(x_t)$ for ConMeZO |
| `initial_f` | Initial objective value $f(x_0)$ |
| `eta_mezo`, `eta_con`, `alpha_con`, `theta_con` | Best hyperparameters used |
