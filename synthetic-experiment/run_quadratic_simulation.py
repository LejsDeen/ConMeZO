#!/usr/bin/env python3
"""
MeZO vs ConMeZO on an ill-conditioned general quadratic: run trials and save curves to a .npz.

Usage:
  python run_quadratic_simulation.py
  python run_quadratic_simulation.py --save-data out.npz --full-hpo
"""

from __future__ import annotations

import argparse
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from algorithms import OptimizerFactory
from problems import ProblemFactory


class ExperimentConfig:
    def __init__(
        self,
        dimension: int = 20,
        mu: float = 0.01,
        iterations: int = 500,
        x_init_norm: float = 10.0,
        seed: Optional[int] = None,
    ):
        self.dimension = dimension
        self.mu = mu
        self.iterations = iterations
        self.x_init_norm = x_init_norm
        self.seed = seed


class HPOConfig:
    def __init__(
        self,
        eta_grid: Optional[np.ndarray] = None,
        alpha_m_grid: Optional[List[float]] = None,
        theta_grid: Optional[List[float]] = None,
    ):
        self.eta_grid = eta_grid if eta_grid is not None else np.logspace(-5, -0.5, num=15)
        self.alpha_m_grid = alpha_m_grid if alpha_m_grid is not None else [
            0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 0.95, 0.99, 0.999
        ]
        self.theta_grid = theta_grid if theta_grid is not None else [
            0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.4, 1.57
        ]


class ExperimentResult:
    def __init__(
        self,
        algorithm: str,
        params: Dict[str, Any],
        f_history: np.ndarray,
        grad_norm_history: np.ndarray,
    ):
        self.algorithm = algorithm
        self.params = params
        self.f_history = f_history
        self.grad_norm_history = grad_norm_history

    def is_converged(self) -> bool:
        return not np.all(np.isnan(self.f_history))


def run_single_trial(problem, x_init, m_init, config, optimizer_type, params, trial_seed):
    np.random.seed(trial_seed)
    if optimizer_type == "mezo":
        optimizer = OptimizerFactory.create_optimizer("mezo", mu=config.mu, verbose=False)
        f_hist, grad_hist = optimizer.optimize(
            x_init, problem.objective_function, config.iterations, params["eta"],
            gradient_func=problem.gradient,
        )
    elif optimizer_type == "conmezo":
        optimizer = OptimizerFactory.create_optimizer("conmezo", mu=config.mu, verbose=False)
        f_hist, grad_hist = optimizer.optimize(
            x_init,
            problem.objective_function,
            config.iterations,
            params["eta"],
            alpha_m=params["alpha_m"],
            theta=params["theta"],
            m_init=m_init,
            gradient_func=problem.gradient,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    return ExperimentResult(optimizer_type, params, f_hist, grad_hist)


def run_multitrial_hpo(problem, x_init, m_init, config, hpo_config, num_trials=5, base_seed=42):
    mezo_multitrial_results: Dict = {}
    conmezo_multitrial_results: Dict = {}

    print(f"\n--- MeZO Multi-Trial HPO (d={config.dimension}) ---")
    total_runs = len(hpo_config.eta_grid) * num_trials
    current_run = 0
    t0 = time.time()
    for eta in hpo_config.eta_grid:
        params = {"eta": eta}
        params_key = (eta,)
        mezo_multitrial_results[params_key] = []
        for _trial in range(num_trials):
            current_run += 1
            trial_seed = base_seed + current_run
            print(
                f"Running MeZO ({current_run}/{total_runs}): eta={eta:.7f}",
                end="\r",
            )
            mezo_multitrial_results[params_key].append(
                run_single_trial(problem, x_init, m_init, config, "mezo", params, trial_seed)
            )
    print(f"\nMeZO multi-trial HPO complete in {time.time() - t0:.1f}s")

    print(f"\n--- ConMeZO Multi-Trial HPO (d={config.dimension}) ---")
    total_runs = (
        len(hpo_config.alpha_m_grid)
        * len(hpo_config.eta_grid)
        * len(hpo_config.theta_grid)
        * num_trials
    )
    current_run = 0
    t0 = time.time()
    for alpha_m, eta, theta in product(
        hpo_config.alpha_m_grid, hpo_config.eta_grid, hpo_config.theta_grid
    ):
        params = {"eta": eta, "alpha_m": alpha_m, "theta": theta}
        params_key = (eta, alpha_m, theta)
        conmezo_multitrial_results[params_key] = []
        for _trial in range(num_trials):
            current_run += 1
            trial_seed = base_seed + current_run + 100000
            print(
                f"Running ConMeZO ({current_run}/{total_runs}): "
                f"eta={eta:.7f}, alpha_m={alpha_m:.2f}, theta={theta:.2f}",
                end="\r",
            )
            conmezo_multitrial_results[params_key].append(
                run_single_trial(problem, x_init, m_init, config, "conmezo", params, trial_seed)
            )
    print(f"\nConMeZO multi-trial HPO complete in {time.time() - t0:.1f}s")

    return mezo_multitrial_results, conmezo_multitrial_results


def aggregate_trials(multitrial_results: Dict) -> Tuple[Dict, Dict, object]:
    mean_histories: Dict = {}
    se_histories: Dict = {}
    best_final_value = float("inf")
    best_params_key = None

    for params_key, results_list in multitrial_results.items():
        all_histories = []
        for result in results_list:
            if result.is_converged():
                all_histories.append(result.f_history)
        if len(all_histories) == 0:
            continue
        histories_array = np.array(all_histories)
        mean_hist = np.nanmean(histories_array, axis=0)
        se_hist = np.nanstd(histories_array, axis=0) / np.sqrt(len(all_histories))
        mean_histories[params_key] = mean_hist
        se_histories[params_key] = se_hist
        final_mean = mean_hist[-1] if not np.isnan(mean_hist[-1]) else float("inf")
        if final_mean < best_final_value:
            best_final_value = final_mean
            best_params_key = params_key
    return mean_histories, se_histories, best_params_key


def main():
    parser = argparse.ArgumentParser(description="Compute MeZO vs ConMeZO curves on general quadratic.")
    parser.add_argument(
        "--full-hpo",
        action="store_true",
        help="Stage-1 grid HPO (10^4 iter/config) then long run. "
        "Default: fixed η=0.01, α_m=0.01, θ=1.30 — long run only.",
    )
    parser.add_argument(
        "--save-data",
        type=str,
        default="mezo_conmezo_longrun_curves.npz",
        help="Output path for aggregated curves (.npz).",
    )
    args = parser.parse_args()

    dim = 10**3
    mu = 0.01
    x_init_norm = 10.0
    seed = 42
    num_trials = 5
    base_seed = 42

    problem = ProblemFactory.create_problem(
        "general_quadratic", dim, condition_number=dim**3, seed=seed
    )
    x_initial = problem.generate_initial_point(x_init_norm, seed)
    m_initial = np.zeros(dim)
    initial_f = problem.objective_function(x_initial)

    if args.full_hpo:
        config_hpo = ExperimentConfig(
            dimension=dim, mu=mu, iterations=10**4, x_init_norm=x_init_norm, seed=seed
        )
        hpo_config = HPOConfig(
            eta_grid=10.0 ** np.arange(0, -5, -1),
            alpha_m_grid=[0.01, 0.05, 0.1, 0.2],
            theta_grid=[1.2, 1.3, 1.4, 1.5],
        )
        mezo_mt, con_mt = run_multitrial_hpo(
            problem, x_initial, m_initial, config_hpo, hpo_config, num_trials, base_seed
        )
        mezo_means, mezo_ses, best_mezo_key = aggregate_trials(mezo_mt)
        con_means, con_ses, best_con_key = aggregate_trials(con_mt)
        if best_mezo_key is None or best_con_key is None:
            raise SystemExit("HPO did not produce valid best configs.")
        eta_m = best_mezo_key[0]
        eta_c = best_con_key[0]
        alpha_c = best_con_key[1]
        theta_c = best_con_key[2]
    else:
        eta_m = eta_c = 0.01
        alpha_c = 0.01
        theta_c = 1.30

    config_long = ExperimentConfig(
        dimension=dim, mu=mu, iterations=10**5, x_init_norm=x_init_norm, seed=seed
    )
    hpo_long = HPOConfig(
        eta_grid=[eta_m],
        alpha_m_grid=[alpha_c],
        theta_grid=[theta_c],
    )
    mezo_mt2, con_mt2 = run_multitrial_hpo(
        problem, x_initial, m_initial, config_long, hpo_long, num_trials, base_seed
    )
    mezo_means2, mezo_ses2, best_mezo_key2 = aggregate_trials(mezo_mt2)
    con_means2, con_ses2, best_con_key2 = aggregate_trials(con_mt2)
    if best_mezo_key2 is None or best_con_key2 is None:
        raise SystemExit("Long-run aggregation failed.")

    mm = mezo_means2[best_mezo_key2]
    sm = mezo_ses2[best_mezo_key2]
    mc = con_means2[best_con_key2]
    sc = con_ses2[best_con_key2]

    np.savez(
        args.save_data,
        mean_mezo=mm,
        se_mezo=sm,
        mean_con=mc,
        se_con=sc,
        initial_f=initial_f,
        eta_mezo=best_mezo_key2[0],
        eta_con=best_con_key2[0],
        alpha_con=best_con_key2[1],
        theta_con=best_con_key2[2],
    )
    print(f"Saved curve bundle to {args.save_data}")


if __name__ == "__main__":
    main()
