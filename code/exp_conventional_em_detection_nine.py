"""Conventional EM detection benchmarks on the eight real datasets.

For each dataset, fit homogeneous (m=1) and homoscedastic two-component
(m=2) normal mixture regressions using the same Stage-1 predictor design as
the manuscript.
Two detector-only decisions are reported:

* EM-BIC selects m=2 when BIC_2 < BIC_1.
* EM-LR rejects homogeneity using a fixed-design parametric-bootstrap
  likelihood-ratio p-value.  A chi-square reference is intentionally not used
  because the mixture null is nonregular.

The m=2 likelihood is fitted by conventional multi-start EM with a variance
shared by the two components.  The common-variance specification avoids the
unbounded-likelihood degeneracy of an unconstrained component-specific fit and
matches the common-noise mixture model used in the paper's detection theory.

Outputs:
  results/conventional_em_detection_nine_summary.csv
  results/conventional_em_detection_nine_bootstrap.csv
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
RESULTS_DIR = os.path.join(ROOT, "results")
sys.path.insert(0, THIS_DIR)
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import exp_realworld_K2_rigorous as R  # noqa: E402


LOADERS = [
    R.load_taylor,
    R.load_iris,
    R.load_penguins,
    R.load_atlanta,
    R.load_power,
    R.load_nematodes,
    R.load_ca_housing,
    R.load_n_deposition,
]

LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class NullFit:
    beta: np.ndarray
    variance: float
    loglik: float


@dataclass
class EMFit:
    pi: np.ndarray
    beta: np.ndarray
    variance: np.ndarray
    loglik: float
    iterations: int
    restart: int


def finite_standardized_xy(y, x):
    y = np.asarray(y, dtype=float).reshape(-1)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != y.size and x.shape[1] == y.size:
        x = x.T
    if x.shape[0] != y.size:
        raise ValueError(f"incompatible y and x shapes: {y.shape}, {x.shape}")
    keep = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y = y[keep]
    x = x[keep]
    x_sd = x.std(axis=0)
    if np.any(x_sd <= 1e-12):
        raise ValueError("a Stage-1 predictor is constant")
    y_sd = float(y.std())
    if y_sd <= 1e-12:
        raise ValueError("the response is constant")
    x = (x - x.mean(axis=0)) / x_sd
    y = (y - y.mean()) / y_sd
    return y, np.column_stack([np.ones(y.size), x]), keep


def fit_null(y, design):
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    variance = max(float(np.mean(resid * resid)), 1e-12)
    loglik = float(-0.5 * np.sum(
        LOG_2PI + np.log(variance) + resid * resid / variance
    ))
    return NullFit(beta=beta, variance=variance, loglik=loglik)


def log_components(y, design, pi, beta, variance):
    residual2 = (y[:, None] - design @ beta.T) ** 2
    return (np.log(np.maximum(pi, 1e-300))[None, :]
            - 0.5 * (LOG_2PI + np.log(variance)[None, :]
                     + residual2 / variance[None, :]))


def mixture_loglik(y, design, pi, beta, variance):
    log_c = log_components(y, design, pi, beta, variance)
    mx = np.max(log_c, axis=1)
    return float(np.sum(mx + np.log(np.exp(log_c - mx[:, None]).sum(axis=1))))


def responsibilities(y, design, pi, beta, variance):
    log_c = log_components(y, design, pi, beta, variance)
    mx = np.max(log_c, axis=1, keepdims=True)
    weights = np.exp(log_c - mx)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def weighted_regression(design, y, weights, fallback):
    weights = np.maximum(weights, 0.0)
    xtwx = design.T @ (weights[:, None] * design)
    xtwy = design.T @ (weights * y)
    try:
        beta = np.linalg.solve(xtwx, xtwy)
    except np.linalg.LinAlgError:
        try:
            beta = np.linalg.lstsq(xtwx, xtwy, rcond=None)[0]
        except np.linalg.LinAlgError:
            beta = fallback.copy()
    if not np.all(np.isfinite(beta)):
        beta = fallback.copy()
    return beta


def initial_states(y, design, null, n_restarts, seed):
    p = design.shape[1]
    resid_sd = max(float(np.sqrt(null.variance)), 1e-4)
    starts = []

    def add(delta, pi=(0.5, 0.5), variance_scale=1.0):
        starts.append((
            np.asarray(pi, dtype=float),
            np.vstack([null.beta - delta, null.beta + delta]),
            np.full(2, null.variance * float(variance_scale)),
        ))

    delta = np.zeros(p)
    delta[0] = 0.7 * resid_sd
    add(delta)
    add(delta, pi=(0.3, 0.7), variance_scale=0.8)
    for col in range(1, p):
        delta = np.zeros(p)
        delta[col] = 0.65 * resid_sd
        add(delta)
        delta = np.zeros(p)
        delta[0] = 0.35 * resid_sd
        delta[col] = 0.5 * resid_sd
        add(delta, pi=(0.4, 0.6))

    rng = np.random.default_rng(seed)
    while len(starts) < n_restarts:
        scale = np.full(p, 0.45 * resid_sd)
        scale[0] = 0.7 * resid_sd
        delta = rng.normal(0.0, scale)
        pi1 = float(rng.uniform(0.25, 0.75))
        var_scale = float(np.exp(rng.normal(0.0, 0.3)))
        add(delta, pi=(pi1, 1.0 - pi1), variance_scale=var_scale)
    return starts[:n_restarts]


def fit_two_component_em(y, design, n_restarts=8, max_iter=250,
                         tol=1e-8, seed=20260817):
    null = fit_null(y, design)
    variance_floor = 1e-10 * null.variance
    pi_floor = 1e-4
    best = None

    for restart, (pi, beta, variance) in enumerate(
            initial_states(y, design, null, n_restarts, seed)):
        previous = mixture_loglik(y, design, pi, beta, variance)
        iterations = 0
        for iteration in range(1, max_iter + 1):
            weights = responsibilities(y, design, pi, beta, variance)
            n_eff = weights.sum(axis=0)
            pi_new = np.maximum(n_eff / y.size, pi_floor)
            pi_new /= pi_new.sum()
            beta_new = np.empty_like(beta)
            component_sse = np.empty(2)
            for j in range(2):
                beta_new[j] = weighted_regression(
                    design, y, weights[:, j], beta[j]
                )
                resid = y - design @ beta_new[j]
                component_sse[j] = float(
                    np.sum(weights[:, j] * resid * resid)
                )
            common_variance = max(
                float(component_sse.sum() / y.size), variance_floor
            )
            variance_new = np.full(2, common_variance)
            current = mixture_loglik(
                y, design, pi_new, beta_new, variance_new
            )
            if current + 1e-7 < previous:
                break
            pi, beta, variance = pi_new, beta_new, variance_new
            iterations = iteration
            if abs(current - previous) <= tol * (1.0 + abs(previous)):
                previous = current
                break
            previous = current

        fitted = EMFit(
            pi=pi.copy(), beta=beta.copy(), variance=variance.copy(),
            loglik=float(previous), iterations=iterations, restart=restart,
        )
        if best is None or fitted.loglik > best.loglik:
            best = fitted
    return null, best


def detection_statistics(y, design, n_restarts, max_iter, tol, seed):
    null, alternative = fit_two_component_em(
        y, design, n_restarts=n_restarts, max_iter=max_iter,
        tol=tol, seed=seed,
    )
    lr = max(2.0 * (alternative.loglik - null.loglik), 0.0)
    p = design.shape[1]
    parameter_difference = p + 1  # (2p+2) - (p+1), common variance
    delta_bic = lr - parameter_difference * math.log(y.size)  # BIC1-BIC2
    return null, alternative, lr, delta_bic, parameter_difference


_BOOT_CONTEXT = None


def init_boot_worker(design, null_beta, null_variance, n_restarts,
                     max_iter, tol, fit_seed):
    global _BOOT_CONTEXT
    _BOOT_CONTEXT = (
        design, null_beta, null_variance, n_restarts, max_iter, tol, fit_seed,
    )


def boot_worker(task):
    replicate, noise_seed = task
    (design, null_beta, null_variance, n_restarts,
     max_iter, tol, fit_seed) = _BOOT_CONTEXT
    rng = np.random.default_rng(noise_seed)
    yb = design @ null_beta + rng.normal(
        0.0, math.sqrt(null_variance), design.shape[0]
    )
    _, _, lr, delta_bic, _ = detection_statistics(
        yb, design, n_restarts=n_restarts, max_iter=max_iter,
        tol=tol, seed=fit_seed + 37 * replicate,
    )
    return replicate, lr, delta_bic


def stage1_predictors(dataset):
    x = np.asarray(dataset["x"], dtype=float)
    if x.ndim == 2 and x.shape[1] > 1:
        return x, "full x design"
    focal = np.asarray(dataset.get("sp_x", x), dtype=float)
    return focal, dataset.get("sp_x_label", "x")


def run_dataset(dataset, n_boot, observed_restarts, bootstrap_restarts,
                max_iter, tol, base_seed, jobs):
    y_raw = np.asarray(dataset["y"], dtype=float)
    x_raw, predictor_label = stage1_predictors(dataset)
    y, design, keep = finite_standardized_xy(y_raw, x_raw)
    null, alternative, lr, delta_bic, parameter_difference = detection_statistics(
        y, design, n_restarts=observed_restarts, max_iter=max_iter,
        tol=tol, seed=base_seed,
    )

    boot_lr = np.empty(n_boot)
    boot_delta_bic = np.empty(n_boot)
    tasks = [(b, base_seed + 100_003 + b) for b in range(n_boot)]
    if n_boot and jobs > 1:
        context = mp.get_context("spawn")
        with context.Pool(
            processes=jobs,
            initializer=init_boot_worker,
            initargs=(design, null.beta, null.variance, bootstrap_restarts,
                      max_iter, tol, base_seed + 500_000),
        ) as pool:
            for done, (b, lr_b, bic_b) in enumerate(
                    pool.imap_unordered(boot_worker, tasks), start=1):
                boot_lr[b] = lr_b
                boot_delta_bic[b] = bic_b
                if done % 25 == 0 or done == n_boot:
                    print(f"  bootstrap {done}/{n_boot}", flush=True)
    else:
        init_boot_worker(
            design, null.beta, null.variance, bootstrap_restarts,
            max_iter, tol, base_seed + 500_000,
        )
        for done, task in enumerate(tasks, start=1):
            b, lr_b, bic_b = boot_worker(task)
            boot_lr[b] = lr_b
            boot_delta_bic[b] = bic_b
            if done % 25 == 0 or done == n_boot:
                print(f"  bootstrap {done}/{n_boot}", flush=True)

    if n_boot:
        exceedances = int(np.sum(boot_lr >= lr - 1e-10))
        p_boot = (exceedances + 1.0) / (n_boot + 1.0)
        critical95 = float(np.quantile(boot_lr, 0.95))
    else:
        p_boot = np.nan
        critical95 = np.nan

    slope_columns = alternative.beta[:, 1:] if alternative.beta.shape[1] > 1 else np.empty((2, 0))
    summary = {
        "dataset": dataset["name"],
        "status": "ok",
        "n": y.size,
        "n_removed_nonfinite": int(y_raw.size - y.size),
        "predictor_design": predictor_label,
        "x_dim": design.shape[1] - 1,
        "observed_restarts": observed_restarts,
        "bootstrap_restarts": bootstrap_restarts,
        "max_iter": max_iter,
        "fit_seed": base_seed,
        "bootstrap_B": n_boot,
        "loglik_m1": null.loglik,
        "loglik_m2": alternative.loglik,
        "LR_stat": lr,
        "LR_crit95_boot": critical95,
        "LR_p_boot": p_boot,
        "LR_reject_5pct": bool(p_boot <= 0.05) if n_boot else np.nan,
        "BIC1_minus_BIC2": delta_bic,
        "BIC_select_m2": bool(delta_bic > 0.0),
        "parameter_difference": parameter_difference,
        "pi1": float(alternative.pi[0]),
        "common_variance_std": float(alternative.variance[0]),
        "slope1_std": " ".join(f"{v:.10g}" for v in slope_columns[0]),
        "slope2_std": " ".join(f"{v:.10g}" for v in slope_columns[1]),
        "em_iterations_best": alternative.iterations,
        "em_restart_best": alternative.restart,
        "variance_model": "common across components",
        "source": dataset.get("source", ""),
    }
    boot_rows = [
        {
            "dataset": dataset["name"],
            "replicate": b + 1,
            "LR_stat": boot_lr[b],
            "BIC1_minus_BIC2": boot_delta_bic[b],
        }
        for b in range(n_boot)
    ]
    return summary, boot_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=199)
    parser.add_argument("--observed-restarts", type=int, default=10)
    parser.add_argument("--bootstrap-restarts", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main():
    global RESULTS_DIR
    args = parse_args()
    RESULTS_DIR = os.path.abspath(args.out_dir)
    wanted = set(args.datasets) if args.datasets else None
    summaries = []
    all_boot = []
    start = time.time()

    for index, loader in enumerate(LOADERS):
        try:
            dataset = loader()
            if wanted is not None and dataset["name"] not in wanted:
                continue
            print(f"[{dataset['name']}] loading complete", flush=True)
            summary, boot_rows = run_dataset(
                dataset,
                n_boot=args.boot,
                observed_restarts=args.observed_restarts,
                bootstrap_restarts=args.bootstrap_restarts,
                max_iter=args.max_iter,
                tol=args.tol,
                base_seed=args.seed + 10_000 * index,
                jobs=args.jobs,
            )
            summaries.append(summary)
            all_boot.extend(boot_rows)
            lr_decision = (
                f"p={summary['LR_p_boot']:.4f}"
                if args.boot else "p=NA"
            )
            print(
                f"[{dataset['name']}] n={summary['n']} "
                f"LR={summary['LR_stat']:.3f}, {lr_decision}; "
                f"BIC1-BIC2={summary['BIC1_minus_BIC2']:.3f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            name = getattr(loader, "__name__", "unknown")
            print(f"[{name}] ERROR: {type(exc).__name__}: {exc}", flush=True)
            summaries.append({"dataset": name, "status": f"error: {exc}"})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{args.suffix}" if args.suffix else ""
    summary_path = os.path.join(
        RESULTS_DIR, f"conventional_em_detection_nine_summary{suffix}.csv"
    )
    boot_path = os.path.join(
        RESULTS_DIR, f"conventional_em_detection_nine_bootstrap{suffix}.csv"
    )
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(all_boot).to_csv(boot_path, index=False)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {boot_path}", flush=True)
    print(f"elapsed_seconds={time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
