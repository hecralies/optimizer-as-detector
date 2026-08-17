"""Kasahara--Shimotsu (2015) modified penalized-EM test on eight datasets.

This implements their formal test of homogeneity (m=1) against a
two-component normal mixture regression (m=2), specialized to the scalar
focal relationship used by the manuscript's real-data table.  It follows
Section 5 and Section 7.1 of the paper:

* initial restricted penalized MLE with alpha_1 = alpha_2 = 0.5;
* penalty p_n(v; v_hat) = -a_n {v_hat/v + log(v/v_hat) - 1};
* a_n = 2.2 for dim(X)=1 and sigma_j >= 0.01 sigma_hat;
* one and two unrestricted generalized-EM updates, reported as K=2 and K=3;
* parametric-bootstrap calibration under the fitted homogeneous regression.

The current ordinary EM/LR benchmark files are not modified.  Outputs:
  results/kasahara_shimotsu_em_nine_summary.csv
  results/kasahara_shimotsu_em_nine_bootstrap.csv
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
from scipy.optimize import minimize


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
RESULTS_DIR = os.path.join(ROOT, "results")
sys.path.insert(0, THIS_DIR)
# Only the data loaders are imported from the shared experiment module.  The
# benchmark itself does not use its Numba SGD kernels, so disabling their JIT
# avoids unnecessary compilation during startup.
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

PENALTY_BY_DIM = {1: 2.2, 2: 3.1, 3: 5.4, 4: 8.3}
LOG_2PI = math.log(2.0 * math.pi)


@dataclass
class NullFit:
    beta: np.ndarray
    variance: float
    loglik: float


@dataclass
class MixtureState:
    pi: np.ndarray
    beta: np.ndarray
    variance: np.ndarray
    loglik: float
    penalized_loglik: float
    iterations: int


def _finite_standardized_xy(y, x):
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
        raise ValueError("the focal regressor contains a constant column")
    y_sd = float(y.std())
    if y_sd <= 1e-12:
        raise ValueError("the response is constant")
    x = (x - x.mean(axis=0)) / x_sd
    y = (y - y.mean()) / y_sd
    design = np.column_stack([np.ones(y.size), x])
    return y, design, keep


def fit_null(y, design):
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    resid = y - design @ beta
    variance = max(float(np.mean(resid * resid)), 1e-12)
    loglik = float(-0.5 * np.sum(LOG_2PI + np.log(variance) + resid * resid / variance))
    return NullFit(beta=beta, variance=variance, loglik=loglik)


def _log_components(y, design, state):
    means = design @ state.beta.T
    residual2 = (y[:, None] - means) ** 2
    return (np.log(np.maximum(state.pi, 1e-300))[None, :]
            - 0.5 * (LOG_2PI + np.log(state.variance)[None, :]
                     + residual2 / state.variance[None, :]))


def _loglik(y, design, state):
    log_c = _log_components(y, design, state)
    mx = np.max(log_c, axis=1)
    return float(np.sum(mx + np.log(np.exp(log_c - mx[:, None]).sum(axis=1))))


def _penalty(variance, variance_hat, a_n):
    ratio = variance_hat / variance
    return float(-a_n * np.sum(ratio + np.log(variance / variance_hat) - 1.0))


def _evaluate(y, design, state, variance_hat, a_n):
    state.loglik = _loglik(y, design, state)
    state.penalized_loglik = state.loglik + _penalty(state.variance, variance_hat, a_n)
    return state


def _responsibilities(y, design, state):
    log_c = _log_components(y, design, state)
    mx = np.max(log_c, axis=1, keepdims=True)
    weights = np.exp(log_c - mx)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights


def _weighted_regression(design, y, weights, fallback):
    sqrt_w = np.sqrt(np.maximum(weights, 0.0))
    wx = design * sqrt_w[:, None]
    wy = y * sqrt_w
    try:
        beta = np.linalg.lstsq(wx, wy, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = fallback.copy()
    if not np.all(np.isfinite(beta)):
        beta = fallback.copy()
    return beta


def _em_update(y, design, state, variance_hat, a_n, fixed_pi):
    weights = _responsibilities(y, design, state)
    n_eff = weights.sum(axis=0)
    if fixed_pi:
        pi = np.array([0.5, 0.5])
    else:
        pi = np.clip(n_eff / y.size, 1e-6, 1.0 - 1e-6)
        pi /= pi.sum()
    beta = np.empty_like(state.beta)
    variance = np.empty(2)
    floor = 1e-4 * variance_hat  # sigma_j >= 0.01 sigma_hat
    for j in range(2):
        beta[j] = _weighted_regression(design, y, weights[:, j], state.beta[j])
        resid = y - design @ beta[j]
        sse = float(np.sum(weights[:, j] * resid * resid))
        # Closed-form maximizer of weighted Gaussian Q plus equation (22).
        variance[j] = max((sse + 2.0 * a_n * variance_hat)
                          / (n_eff[j] + 2.0 * a_n), floor)
    updated = MixtureState(pi=pi, beta=beta, variance=variance,
                           loglik=np.nan, penalized_loglik=np.nan,
                           iterations=state.iterations + 1)
    return _evaluate(y, design, updated, variance_hat, a_n)


def _initial_states(y, design, null, n_restarts, seed):
    """Deterministic structured starts plus reproducible random perturbations."""
    q = design.shape[1]
    beta0 = null.beta
    resid = y - design @ beta0
    resid_sd = max(float(resid.std()), 1e-4)
    starts = []

    def add_pair(delta):
        starts.append(np.vstack([beta0 - delta, beta0 + delta]))

    delta = np.zeros(q)
    delta[0] = 0.65 * resid_sd
    add_pair(delta)
    for col in range(1, q):
        delta = np.zeros(q)
        delta[col] = 0.65 * resid_sd
        add_pair(delta)
        delta2 = np.zeros(q)
        delta2[0] = 0.35 * resid_sd
        delta2[col] = 0.5 * resid_sd
        add_pair(delta2)
    delta = np.zeros(q)
    delta[0] = 1.1 * resid_sd
    add_pair(delta)

    rng = np.random.default_rng(seed)
    while len(starts) < n_restarts:
        scale = np.full(q, 0.45 * resid_sd)
        scale[0] = 0.7 * resid_sd
        perturb = rng.normal(0.0, scale)
        add_pair(perturb)
    return starts[:n_restarts]


def restricted_penalized_fit(y, design, null, a_n, n_restarts=8,
                             max_iter=150, tol=1e-9, seed=2015):
    best = None
    starts = _initial_states(y, design, null, n_restarts, seed)
    for beta in starts:
        state = MixtureState(
            pi=np.array([0.5, 0.5]), beta=beta,
            variance=np.array([null.variance, null.variance]),
            loglik=np.nan, penalized_loglik=np.nan, iterations=0,
        )
        state = _evaluate(y, design, state, null.variance, a_n)
        previous = state.penalized_loglik
        for _ in range(max_iter):
            candidate = _em_update(y, design, state, null.variance, a_n, fixed_pi=True)
            change = candidate.penalized_loglik - previous
            state = candidate
            previous = state.penalized_loglik
            if abs(change) <= tol * (1.0 + abs(previous)):
                break
        if best is None or state.penalized_loglik > best.penalized_loglik:
            best = state

    # The restricted estimate is a penalized MLE, not merely a fixed number of
    # EM steps.  Near homogeneity, ordinary EM can approach that maximum very
    # slowly.  Refine the best EM basin with an analytic-gradient optimizer.
    p = design.shape[1]
    floor = 1e-4 * null.variance

    def unpack(params):
        beta = params[:2 * p].reshape(2, p)
        variance = np.exp(params[2 * p:2 * p + 2])
        return beta, variance

    def objective(params):
        beta, variance = unpack(params)
        means = design @ beta.T
        residual = y[:, None] - means
        log_c = (-math.log(2.0)
                 - 0.5 * (LOG_2PI + np.log(variance)[None, :]
                          + residual * residual / variance[None, :]))
        mx = np.max(log_c, axis=1, keepdims=True)
        exp_c = np.exp(log_c - mx)
        denom = exp_c.sum(axis=1, keepdims=True)
        weights = exp_c / denom
        loglik = float(np.sum(mx[:, 0] + np.log(denom[:, 0])))
        penalized = loglik + _penalty(variance, null.variance, a_n)

        grad_beta = np.empty((2, p))
        for j in range(2):
            grad_beta[j] = design.T @ (
                weights[:, j] * residual[:, j] / variance[j]
            )
        grad_logv = np.sum(
            weights * (-0.5 + 0.5 * residual * residual / variance[None, :]),
            axis=0,
        ) + a_n * (null.variance / variance - 1.0)
        gradient = np.concatenate([grad_beta.ravel(), grad_logv])
        return -penalized, -gradient

    params0 = np.concatenate([best.beta.ravel(), np.log(best.variance)])
    bounds = [(None, None)] * (2 * p) + [(math.log(floor), None)] * 2
    optimized = minimize(
        objective, params0, method="L-BFGS-B", jac=True, bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-12, "gtol": 1e-7, "maxls": 50},
    )
    beta_opt, variance_opt = unpack(optimized.x)
    refined = MixtureState(
        pi=np.array([0.5, 0.5]), beta=beta_opt, variance=variance_opt,
        loglik=np.nan, penalized_loglik=np.nan,
        iterations=best.iterations + int(getattr(optimized, "nit", 0)),
    )
    refined = _evaluate(y, design, refined, null.variance, a_n)
    if np.isfinite(refined.penalized_loglik) and (
            refined.penalized_loglik >= best.penalized_loglik - 1e-8):
        return refined
    return best


def modified_em_statistics(y, design, n_restarts=8, max_iter=150,
                           tol=1e-9, seed=2015):
    null = fit_null(y, design)
    x_dim = design.shape[1] - 1
    if x_dim not in PENALTY_BY_DIM:
        raise ValueError("Kasahara--Shimotsu report tuning constants only for dim(X)=1,...,4")
    a_n = PENALTY_BY_DIM[x_dim]
    restricted = restricted_penalized_fit(
        y, design, null, a_n, n_restarts=n_restarts,
        max_iter=max_iter, tol=tol, seed=seed,
    )
    # Their restricted estimate is theta^(1); one free update gives K=2.
    state_k2 = _em_update(y, design, restricted, null.variance, a_n, fixed_pi=False)
    state_k3 = _em_update(y, design, state_k2, null.variance, a_n, fixed_pi=False)
    stat_k2 = max(2.0 * (state_k2.loglik - null.loglik), 0.0)
    stat_k3 = max(2.0 * (state_k3.loglik - null.loglik), 0.0)
    return {
        "null": null,
        "restricted": restricted,
        "k2": state_k2,
        "k3": state_k3,
        "stat_k2": stat_k2,
        "stat_k3": stat_k3,
        "a_n": a_n,
    }


def fit_penalized_two_component_alternative(y, design, fitted_test,
                                            max_em_iter=300, tol=1e-9):
    """Fit the post-rejection m=2 model for descriptive component recovery.

    The finite K=2/K=3 iterates define the formal KS statistic.  Component
    assignments used by a downstream screening procedure instead come from a
    converged penalized alternative fit, initialized at the K=3 state.
    """
    null = fitted_test["null"]
    a_n = fitted_test["a_n"]
    state = fitted_test["k3"]
    previous = state.penalized_loglik
    for _ in range(max_em_iter):
        candidate = _em_update(
            y, design, state, null.variance, a_n, fixed_pi=False,
        )
        change = candidate.penalized_loglik - previous
        state = candidate
        previous = state.penalized_loglik
        if abs(change) <= tol * (1.0 + abs(previous)):
            break

    p = design.shape[1]
    floor = 1e-4 * null.variance

    def unpack(params):
        eta = float(params[0])
        pi1 = 1.0 / (1.0 + math.exp(-np.clip(eta, -30.0, 30.0)))
        beta = params[1:1 + 2 * p].reshape(2, p)
        variance = np.exp(params[1 + 2 * p:1 + 2 * p + 2])
        return np.array([pi1, 1.0 - pi1]), beta, variance

    def objective(params):
        pi, beta, variance = unpack(params)
        means = design @ beta.T
        residual = y[:, None] - means
        log_c = (np.log(pi)[None, :]
                 - 0.5 * (LOG_2PI + np.log(variance)[None, :]
                          + residual * residual / variance[None, :]))
        mx = np.max(log_c, axis=1, keepdims=True)
        exp_c = np.exp(log_c - mx)
        denom = exp_c.sum(axis=1, keepdims=True)
        weights = exp_c / denom
        loglik = float(np.sum(mx[:, 0] + np.log(denom[:, 0])))
        penalized = loglik + _penalty(variance, null.variance, a_n)

        grad_eta = float(np.sum(weights[:, 0] - pi[0]))
        grad_beta = np.empty((2, p))
        for j in range(2):
            grad_beta[j] = design.T @ (
                weights[:, j] * residual[:, j] / variance[j]
            )
        grad_logv = np.sum(
            weights * (-0.5 + 0.5 * residual * residual / variance[None, :]),
            axis=0,
        ) + a_n * (null.variance / variance - 1.0)
        gradient = np.concatenate([[grad_eta], grad_beta.ravel(), grad_logv])
        return -penalized, -gradient

    eta0 = math.log(state.pi[0] / state.pi[1])
    params0 = np.concatenate([[eta0], state.beta.ravel(), np.log(state.variance)])
    bounds = [(-12.0, 12.0)] + [(None, None)] * (2 * p) + [
        (math.log(floor), None), (math.log(floor), None),
    ]
    optimized = minimize(
        objective, params0, method="L-BFGS-B", jac=True, bounds=bounds,
        options={"maxiter": 600, "ftol": 1e-12, "gtol": 1e-7, "maxls": 50},
    )
    pi_opt, beta_opt, variance_opt = unpack(optimized.x)
    refined = MixtureState(
        pi=pi_opt, beta=beta_opt, variance=variance_opt,
        loglik=np.nan, penalized_loglik=np.nan,
        iterations=state.iterations + int(getattr(optimized, "nit", 0)),
    )
    refined = _evaluate(y, design, refined, null.variance, a_n)
    if np.isfinite(refined.penalized_loglik) and (
            refined.penalized_loglik >= state.penalized_loglik - 1e-8):
        state = refined
    return state, _responsibilities(y, design, state)


def _component_slopes(state):
    return float(state.beta[0, 1]), float(state.beta[1, 1])


_BOOT_CONTEXT = None


def _init_boot_worker(y_mean, design, null_beta, null_variance,
                      n_restarts, max_iter, tol, fit_seed):
    global _BOOT_CONTEXT
    _BOOT_CONTEXT = (y_mean, design, null_beta, null_variance,
                     n_restarts, max_iter, tol, fit_seed)


def _boot_worker(task):
    replicate, noise_seed = task
    (y_mean, design, null_beta, null_variance,
     n_restarts, max_iter, tol, fit_seed) = _BOOT_CONTEXT
    rng = np.random.default_rng(noise_seed)
    yb = y_mean + rng.normal(0.0, math.sqrt(null_variance), design.shape[0])
    fitted = modified_em_statistics(
        yb, design, n_restarts=n_restarts, max_iter=max_iter,
        tol=tol, seed=fit_seed,
    )
    return replicate, fitted["stat_k2"], fitted["stat_k3"]


def run_dataset(dataset, n_boot, n_restarts, max_iter, tol, base_seed, jobs,
                use_full_x=False):
    y_raw = np.asarray(dataset["y"], dtype=float)
    focal = np.asarray(dataset["x"] if use_full_x
                       else dataset.get("sp_x", dataset["x"]), dtype=float)
    if not use_full_x and focal.ndim == 2 and focal.shape[1] > 1:
        focal = focal[:, :1]
    y, design, keep = _finite_standardized_xy(y_raw, focal)
    observed = modified_em_statistics(
        y, design, n_restarts=n_restarts, max_iter=max_iter,
        tol=tol, seed=base_seed,
    )
    null = observed["null"]
    boot_k2 = np.empty(n_boot)
    boot_k3 = np.empty(n_boot)
    tasks = [(b, base_seed + 100_003 + b) for b in range(n_boot)]
    if n_boot and jobs > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=jobs,
            initializer=_init_boot_worker,
            initargs=(design @ null.beta, design, null.beta, null.variance,
                      n_restarts, max_iter, tol, base_seed),
        ) as pool:
            for done, (b, stat_k2, stat_k3) in enumerate(
                    pool.imap_unordered(_boot_worker, tasks), start=1):
                boot_k2[b] = stat_k2
                boot_k3[b] = stat_k3
                if done % 25 == 0 or done == n_boot:
                    print(f"  bootstrap {done}/{n_boot}", flush=True)
    else:
        _init_boot_worker(design @ null.beta, design, null.beta, null.variance,
                          n_restarts, max_iter, tol, base_seed)
        for done, task in enumerate(tasks, start=1):
            b, stat_k2, stat_k3 = _boot_worker(task)
            boot_k2[b] = stat_k2
            boot_k3[b] = stat_k3
            if done % 25 == 0 or done == n_boot:
                print(f"  bootstrap {done}/{n_boot}", flush=True)

    ge_k2 = int(np.sum(boot_k2 >= observed["stat_k2"] - 1e-10))
    ge_k3 = int(np.sum(boot_k3 >= observed["stat_k3"] - 1e-10))
    p_k2 = (ge_k2 + 1.0) / (n_boot + 1.0) if n_boot else np.nan
    p_k3 = (ge_k3 + 1.0) / (n_boot + 1.0) if n_boot else np.nan
    crit_k2 = float(np.quantile(boot_k2, 0.95)) if n_boot else np.nan
    crit_k3 = float(np.quantile(boot_k3, 0.95)) if n_boot else np.nan
    boot_rows = [
        {"dataset": dataset["name"], "replicate": b + 1,
         "stat_K2": boot_k2[b], "stat_K3": boot_k3[b]}
        for b in range(n_boot)
    ]
    b2_1, b2_2 = _component_slopes(observed["k2"])
    b3_1, b3_2 = _component_slopes(observed["k3"])
    summary = {
        "dataset": dataset["name"],
        "status": "ok",
        "n": y.size,
        "n_removed_nonfinite": int(y_raw.size - y.size),
        "focal_x": "full x design" if use_full_x else dataset.get("sp_x_label", "x"),
        "x_dim": design.shape[1] - 1,
        "a_n": observed["a_n"],
        "restricted_solver": "EM warm starts + L-BFGS-B refinement",
        "n_restarts": n_restarts,
        "em_warmup_maxiter": max_iter,
        "fit_seed": base_seed,
        "bootstrap_B": n_boot,
        "stat_K2": observed["stat_k2"],
        "p_boot_K2": p_k2,
        "crit95_K2": crit_k2,
        "reject_5pct_K2": bool(p_k2 <= 0.05) if n_boot else np.nan,
        "pi1_K2": float(observed["k2"].pi[0]),
        "slope1_K2_std": b2_1,
        "slope2_K2_std": b2_2,
        "slopes_reverse_K2": bool(b2_1 * b2_2 < 0.0),
        "stat_K3": observed["stat_k3"],
        "p_boot_K3": p_k3,
        "crit95_K3": crit_k3,
        "reject_5pct_K3": bool(p_k3 <= 0.05) if n_boot else np.nan,
        "pi1_K3": float(observed["k3"].pi[0]),
        "slope1_K3_std": b3_1,
        "slope2_K3_std": b3_2,
        "slopes_reverse_K3": bool(b3_1 * b3_2 < 0.0),
        "restricted_iterations": observed["restricted"].iterations,
        "source": dataset.get("source", ""),
    }
    return summary, boot_rows


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=199)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--full-x", action="store_true",
                        help="use each dataset's full Stage 1 predictor design")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="optional dataset names, for example Taylor Iris")
    parser.add_argument("--suffix", default="",
                        help="optional output suffix, useful for smoke tests")
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
                dataset, n_boot=args.boot, n_restarts=args.restarts,
                max_iter=args.max_iter, tol=args.tol,
                base_seed=args.seed + 10_000 * index, jobs=args.jobs,
                use_full_x=args.full_x,
            )
            summaries.append(summary)
            all_boot.extend(boot_rows)
            print(
                f"[{dataset['name']}] n={summary['n']} "
                f"K2={summary['stat_K2']:.3f}, p={summary['p_boot_K2']:.4f}; "
                f"K3={summary['stat_K3']:.3f}, p={summary['p_boot_K3']:.4f}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            name = getattr(loader, "__name__", "unknown")
            print(f"[{name}] ERROR: {type(exc).__name__}: {exc}", flush=True)
            summaries.append({"dataset": name, "status": f"error: {exc}"})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{args.suffix}" if args.suffix else ""
    summary_path = os.path.join(RESULTS_DIR, f"kasahara_shimotsu_em_nine_summary{suffix}.csv")
    boot_path = os.path.join(RESULTS_DIR, f"kasahara_shimotsu_em_nine_bootstrap{suffix}.csv")
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    pd.DataFrame(all_boot).to_csv(boot_path, index=False)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {boot_path}", flush=True)
    print(f"elapsed_seconds={time.time() - start:.1f}", flush=True)


if __name__ == "__main__":
    main()
