"""
Reproducible Type I error experiments for the competitive-SGD detector.

This script intentionally does not hard-code any reported rejection rates.
It writes:
  - results/type1_rigorous_raw.csv
  - results/type1_rigorous_summary.csv

Default parameters match the PNAS draft:
  T = n = 200,000, eta = 0.005, sigma = 1
  d=1: 200 independent trials with closed-form one-sided p-values
  d>=2: 100 calibration runs + 100 independent test runs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import norm

from exp_synthetic_h1_robust import robust_component_scale


@njit(cache=False)
def _sgd_scalar_centered(x, y, idx, eta, init_gap):
    sxx = 0.0
    sxy = 0.0
    n = x.shape[0]
    for i in range(n):
        sxx += x[i] * x[i]
        sxy += x[i] * y[i]
    beta_ols = sxy / sxx
    theta0 = beta_ols + init_gap
    theta1 = beta_ols - init_gap

    for t in range(idx.shape[0]):
        i = idx[t]
        xi = x[i]
        yi = y[i]
        r0 = yi - theta0 * xi
        r1 = yi - theta1 * xi
        if r0 * r0 <= r1 * r1:
            theta0 += eta * r0 * xi
        else:
            theta1 += eta * r1 * xi
    return theta0, theta1


@njit(cache=False)
def _sgd_multi_centered(x, y, idx, eta, beta_ols, init_scale):
    d = x.shape[1]
    theta0 = beta_ols.copy()
    theta1 = beta_ols.copy()
    theta0[0] += init_scale
    theta1[0] -= init_scale

    for t in range(idx.shape[0]):
        i = idx[t]
        r0 = y[i]
        r1 = y[i]
        for j in range(d):
            r0 -= theta0[j] * x[i, j]
            r1 -= theta1[j] * x[i, j]
        if r0 * r0 <= r1 * r1:
            for j in range(d):
                theta0[j] += eta * r0 * x[i, j]
        else:
            for j in range(d):
                theta1[j] += eta * r1 * x[i, j]
    return theta0, theta1


def center_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_c = y - y.mean()
    if x.ndim == 1:
        x_c = x - x.mean()
    else:
        x_c = x - x.mean(axis=0)
    return x_c, y_c


def ols_scalar(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.dot(x, y) / np.dot(x, x))


def ols_multi(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    sigma_x = x.T @ x / x.shape[0]
    return np.linalg.solve(sigma_x, x.T @ y / x.shape[0])


def component_scale(x: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Use the robust quadratic-intercept scale."""
    sigma_hat, _ = robust_component_scale(x, y)
    return sigma_hat, "quadratic intercept"


def run_scalar_stat_and_p(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    eta: float,
    t_sgd: int,
    rng_sgd: np.random.Generator,
    init_scale: float,
) -> tuple[float, float, float, float, float]:
    x, y = center_xy(x_raw, y_raw)
    beta_ols = ols_scalar(x, y)
    resid = y - beta_ols * x
    sigma_hat, _ = component_scale(x, y)
    sx = float(np.mean(x**2))

    idx = rng_sgd.integers(0, x.shape[0], size=t_sgd, dtype=np.int64)
    theta0, theta1 = _sgd_scalar_centered(
        x.astype(np.float64), y.astype(np.float64), idx, eta, init_scale * sigma_hat
    )

    st = abs(theta0 - theta1) * np.sqrt(sx) / sigma_hat
    kappa_eps = float(np.mean(np.abs(resid)) / sigma_hat)
    kappa_x = float(np.mean(np.abs(x)) / np.sqrt(sx))
    mu0 = 2.0 * kappa_eps * kappa_x
    e_star = float(np.mean(np.abs(resid)) * np.mean(np.abs(x)) / sx)
    d_hat = float(np.mean((resid - e_star * x) ** 2 * x**2 * (resid * x >= 0)))
    # Only one piece wins each round.  The half-gap innovation therefore has
    # covariance D/2, giving Var(S_T)=2*eta*D/sigma^2 at first order.
    sigma_t = float(np.sqrt(2.0 * eta * d_hat / sigma_hat**2))
    p_value = float(1.0 - norm.cdf((st - mu0) / sigma_t))
    return st, p_value, mu0, sigma_t, sigma_hat


def run_multi_stat(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    eta: float,
    t_sgd: int,
    rng_sgd: np.random.Generator,
    init_scale: float,
) -> tuple[float, float]:
    x, y = center_xy(x_raw, y_raw)
    beta_ols = ols_multi(x, y)
    resid = y - x @ beta_ols
    sigma_hat, _ = component_scale(x, y)
    sigma_x = x.T @ x / x.shape[0]
    idx = rng_sgd.integers(0, x.shape[0], size=t_sgd, dtype=np.int64)
    theta0, theta1 = _sgd_multi_centered(
        x.astype(np.float64), y.astype(np.float64), idx, eta, beta_ols.astype(np.float64),
        init_scale * sigma_hat
    )
    diff = theta0 - theta1
    st = float(np.sqrt(diff @ sigma_x @ diff) / sigma_hat)
    return st, sigma_hat


def gen_data(setting: str, n: int, d: int, rho: float, sigma: float, rng: np.random.Generator):
    theta = np.ones(d)
    if setting == "gaussian":
        if d == 1:
            x = rng.standard_normal(n)
            eps = rng.standard_normal(n) * sigma
            y = theta[0] * x + eps
        else:
            x = rng.standard_normal((n, d))
            eps = rng.standard_normal(n) * sigma
            y = x @ theta + eps
    elif setting == "binary":
        if d != 1:
            raise ValueError("binary setting is only defined for d=1")
        x = rng.choice(np.array([-1.0, 1.0]), size=n)
        eps = rng.standard_normal(n) * sigma
        y = theta[0] * x + eps
    elif setting == "endogenous":
        if d == 1:
            cov = np.array([[1.0, rho * sigma], [rho * sigma, sigma**2]])
            sample = rng.standard_normal((n, 2)) @ np.linalg.cholesky(cov).T
            x = sample[:, 0]
            eps = sample[:, 1]
            y = theta[0] * x + eps
        else:
            cov = np.eye(d + 1)
            cov[d, d] = sigma**2
            cov[0, d] = rho * sigma
            cov[d, 0] = rho * sigma
            sample = rng.standard_normal((n, d + 1)) @ np.linalg.cholesky(cov).T
            x = sample[:, :d]
            eps = sample[:, d]
            y = x @ theta + eps
    else:
        raise ValueError(f"unknown setting: {setting}")
    return x, y


def scalar_experiment(args, label: str, setting: str, rho: float, seed_base: int):
    rows = []
    for rep in range(args.n_trials_d1):
        rng_data = np.random.default_rng(seed_base + rep)
        rng_sgd = np.random.default_rng(seed_base + 50_000 + rep)
        x, y = gen_data(setting, args.n_data, 1, rho, args.sigma, rng_data)
        st, p_value, mu0, sigma_t, sigma_hat = run_scalar_stat_and_p(
            x, y, args.eta, args.t_sgd, rng_sgd, args.init_scale
        )
        rows.append(
            {
                "label": label,
                "setting": setting,
                "d": 1,
                "rho": rho,
                "role": "test",
                "rep": rep,
                "S_T": st,
                "p_value": p_value,
                "mu0": mu0,
                "sigma_T": sigma_t,
                "sigma_hat": sigma_hat,
                "cutoff_95": np.nan,
                "cutoff_99": np.nan,
            }
        )
    return rows


def multi_experiment(args, label: str, setting: str, d: int, rho: float, seed_base: int):
    rows = []
    cal_stats = []
    for rep in range(args.n_cal):
        rng_data = np.random.default_rng(seed_base + rep)
        rng_sgd = np.random.default_rng(seed_base + 50_000 + rep)
        x, y = gen_data(setting, args.n_data, d, rho, args.sigma, rng_data)
        st, sigma_hat = run_multi_stat(x, y, args.eta, args.t_sgd, rng_sgd, args.init_scale)
        cal_stats.append(st)
        rows.append(
            {
                "label": label,
                "setting": setting,
                "d": d,
                "rho": rho,
                "role": "calibration",
                "rep": rep,
                "S_T": st,
                "p_value": np.nan,
                "mu0": np.nan,
                "sigma_T": np.nan,
                "sigma_hat": sigma_hat,
                "cutoff_95": np.nan,
                "cutoff_99": np.nan,
            }
        )

    cal_stats = np.array(cal_stats)
    cutoff_95 = float(np.quantile(cal_stats, 0.95, method="higher"))
    cutoff_99 = float(np.quantile(cal_stats, 0.99, method="higher"))

    for rep in range(args.n_test):
        rng_data = np.random.default_rng(seed_base + 100_000 + rep)
        rng_sgd = np.random.default_rng(seed_base + 150_000 + rep)
        x, y = gen_data(setting, args.n_data, d, rho, args.sigma, rng_data)
        st, sigma_hat = run_multi_stat(x, y, args.eta, args.t_sgd, rng_sgd, args.init_scale)
        rows.append(
            {
                "label": label,
                "setting": setting,
                "d": d,
                "rho": rho,
                "role": "test",
                "rep": rep,
                "S_T": st,
                "p_value": np.nan,
                "mu0": np.nan,
                "sigma_T": np.nan,
                "sigma_hat": sigma_hat,
                "cutoff_95": cutoff_95,
                "cutoff_99": cutoff_99,
            }
        )
    return rows


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for keys, g in raw.groupby(["label", "setting", "d", "rho"], sort=False):
        label, setting, d, rho = keys
        test = g[g["role"] == "test"].copy()
        if int(d) == 1:
            rej05 = float((test["p_value"] < 0.05).mean())
            rej01 = float((test["p_value"] < 0.01).mean())
            cutoff95 = np.nan
            cutoff99 = np.nan
            median_p = float(test["p_value"].median())
        else:
            cutoff95 = float(test["cutoff_95"].iloc[0])
            cutoff99 = float(test["cutoff_99"].iloc[0])
            rej05 = float((test["S_T"] > cutoff95).mean())
            rej01 = float((test["S_T"] > cutoff99).mean())
            median_p = np.nan
        n_test = int(len(test))
        se05 = float(np.sqrt(0.05 * 0.95 / n_test))
        summaries.append(
            {
                "label": label,
                "setting": setting,
                "d": int(d),
                "rho": float(rho),
                "n_test": n_test,
                "mean_S_T": float(test["S_T"].mean()),
                "sd_S_T": float(test["S_T"].std(ddof=1)),
                "median_p": median_p,
                "cutoff_95": cutoff95,
                "cutoff_99": cutoff99,
                "rej_0.05": rej05,
                "rej_0.01": rej01,
                "mc_se_at_0.05": se05,
            }
        )
    return pd.DataFrame(summaries)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-data", type=int, default=200_000)
    parser.add_argument("--t-sgd", type=int, default=200_000)
    parser.add_argument("--eta", type=float, default=0.005)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--n-trials-d1", type=int, default=200)
    parser.add_argument("--n-cal", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.n_data = min(args.n_data, 20_000)
        args.t_sgd = min(args.t_sgd, 20_000)
        args.n_trials_d1 = min(args.n_trials_d1, 10)
        args.n_cal = min(args.n_cal, 10)
        args.n_test = min(args.n_test, 10)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "type1_rigorous_raw.csv"
    summary_path = args.out_dir / "type1_rigorous_summary.csv"
    config_path = args.out_dir / "type1_rigorous_config.json"

    config = vars(args).copy()
    config["out_dir"] = str(config["out_dir"])
    config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    settings = [
        ("No endogeneity: Gaussian x", "gaussian", 1, 0.0, 1_000),
        ("No endogeneity: Gaussian x", "gaussian", 2, 0.0, 3_000),
        ("No endogeneity: Gaussian x", "gaussian", 5, 0.0, 4_000),
        ("Endogeneity", "endogenous", 1, 0.0, 5_000),
        ("Endogeneity", "endogenous", 1, 0.5, 6_000),
        ("Endogeneity", "endogenous", 1, 0.9, 7_000),
        ("Endogeneity", "endogenous", 2, 0.0, 8_000),
        ("Endogeneity", "endogenous", 2, 0.5, 9_000),
        ("Endogeneity", "endogenous", 2, 0.9, 10_000),
        ("Endogeneity", "endogenous", 5, 0.0, 11_000),
        ("Endogeneity", "endogenous", 5, 0.5, 12_000),
        ("Endogeneity", "endogenous", 5, 0.9, 13_000),
    ]

    print(
        f"Type I rigorous run: n={args.n_data}, T={args.t_sgd}, eta={args.eta}, "
        f"d1_trials={args.n_trials_d1}, cal={args.n_cal}, test={args.n_test}",
        flush=True,
    )
    t0 = time.time()
    rows = []
    for label, setting, d, rho, seed_base in settings:
        started = time.time()
        if d == 1:
            new_rows = scalar_experiment(args, label, setting, rho, seed_base)
        else:
            new_rows = multi_experiment(args, label, setting, d, rho, seed_base)
        rows.extend(new_rows)
        raw = pd.DataFrame(rows)
        raw.to_csv(raw_path, index=False)
        summary = summarize(raw)
        summary.to_csv(summary_path, index=False)
        last = summary.iloc[-1]
        print(
            f"{label:28s} d={d} rho={rho:.1f}: "
            f"rej05={last['rej_0.05']:.3f}, rej01={last['rej_0.01']:.3f}, "
            f"mean_ST={last['mean_S_T']:.3f} [{time.time() - started:.1f}s]",
            flush=True,
        )

    print(f"Saved {raw_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    print(f"Done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
