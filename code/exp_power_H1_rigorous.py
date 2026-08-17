"""
Reproducible H1 gap and power experiments for the competitive-SGD detector.

Writes:
  - results/power_H1_rigorous_raw.csv
  - results/power_H1_rigorous_summary.csv
  - results/power_H1_rigorous_config.json

Every statistic is normalized by the two-fold cross-fitted
quadratic-intercept component-scale estimator.  Every rejection decision uses
an empirical one-sided upper null critical value calibrated under the matching
dimension and endogeneity setting.

Default parameters match the paper experiment:
  T = n = 200,000, eta = 0.005, sigma = 1, Delta = 0.5
  no endogeneity: d in {1,2,5,10}
  endogeneity: rho in {0,0.5,0.9}, d in {1,2,5}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

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


def scalar_stat(x_raw, y_raw, eta, t_sgd, rng_sgd, init_scale):
    x, y = center_xy(x_raw, y_raw)
    beta_ols = ols_scalar(x, y)
    sigma_hat, _ = robust_component_scale(x, y)
    sx = float(np.mean(x**2))
    idx = rng_sgd.integers(0, len(x), size=t_sgd, dtype=np.int64)
    theta0, theta1 = _sgd_scalar_centered(
        x.astype(np.float64), y.astype(np.float64), idx, eta, init_scale * sigma_hat
    )
    st = abs(theta0 - theta1) * np.sqrt(sx) / sigma_hat
    return st, sigma_hat


def multi_stat(x_raw, y_raw, eta, t_sgd, rng_sgd, init_scale):
    x, y = center_xy(x_raw, y_raw)
    beta_ols = ols_multi(x, y)
    sigma_hat, _ = robust_component_scale(x, y)
    sigma_x = x.T @ x / x.shape[0]
    idx = rng_sgd.integers(0, x.shape[0], size=t_sgd, dtype=np.int64)
    theta0, theta1 = _sgd_multi_centered(
        x.astype(np.float64), y.astype(np.float64), idx, eta, beta_ols.astype(np.float64),
        init_scale * sigma_hat
    )
    diff = theta0 - theta1
    st = float(np.sqrt(diff @ sigma_x @ diff) / sigma_hat)
    return st, sigma_hat


def gen_h0(n, d, rho, sigma, rng):
    theta = np.ones(d)
    if d == 1:
        if rho == 0.0:
            x = rng.standard_normal(n)
            eps = rng.standard_normal(n) * sigma
        else:
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
    return x, y


def gen_h1(n, d, rho, sigma, delta, rng):
    if d == 1:
        if rho == 0.0:
            x = rng.standard_normal(n)
            eps = rng.standard_normal(n) * sigma
        else:
            cov = np.array([[1.0, rho * sigma], [rho * sigma, sigma**2]])
            sample = rng.standard_normal((n, 2)) @ np.linalg.cholesky(cov).T
            x = sample[:, 0]
            eps = sample[:, 1]
        groups = rng.integers(0, 2, size=n)
        slopes = np.where(groups == 0, -delta, delta)
        y = slopes * x + eps
    else:
        cov = np.eye(d + 1)
        cov[d, d] = sigma**2
        cov[0, d] = rho * sigma
        cov[d, 0] = rho * sigma
        sample = rng.standard_normal((n, d + 1)) @ np.linalg.cholesky(cov).T
        x = sample[:, :d]
        eps = sample[:, d]
        theta0 = np.zeros(d)
        theta1 = np.zeros(d)
        theta0[0] = -delta
        theta1[0] = delta
        groups = rng.integers(0, 2, size=n)
        y = np.where(groups == 0, x @ theta0, x @ theta1) + eps
    return x, y


def h0_cutoffs(args, d, rho, seed_base):
    rows = []
    stats = []
    for rep in range(args.n_cal):
        rng_data = np.random.default_rng(seed_base + rep)
        rng_sgd = np.random.default_rng(seed_base + 50_000 + rep)
        x, y = gen_h0(args.n_data, d, rho, args.sigma, rng_data)
        if d == 1:
            st, sigma_hat = scalar_stat(
                x, y, args.eta, args.t_sgd, rng_sgd, args.init_scale
            )
        else:
            st, sigma_hat = multi_stat(
                x, y, args.eta, args.t_sgd, rng_sgd, args.init_scale
            )
        stats.append(st)
        rows.append(
            {
                "scenario": "H0 calibration",
                "d": d,
                "rho": rho,
                "rep": rep,
                "S_T": st,
                "p_value": np.nan,
                "sigma_hat": sigma_hat,
                "cutoff_95": np.nan,
                "cutoff_99": np.nan,
                "reject_05": np.nan,
                "reject_01": np.nan,
            }
        )
    stats = np.array(stats)
    cutoff_95 = float(np.quantile(stats, 0.95, method="higher"))
    cutoff_99 = float(np.quantile(stats, 0.99, method="higher"))
    return cutoff_95, cutoff_99, rows


def h1_rows(args, scenario, d, rho, cutoff_95, cutoff_99, seed_base):
    rows = []
    for rep in range(args.n_h1):
        rng_data = np.random.default_rng(seed_base + 100_000 + rep)
        rng_sgd = np.random.default_rng(seed_base + 150_000 + rep)
        x, y = gen_h1(args.n_data, d, rho, args.sigma, args.delta, rng_data)
        if d == 1:
            st, sigma_hat = scalar_stat(
                x, y, args.eta, args.t_sgd, rng_sgd, args.init_scale
            )
        else:
            st, sigma_hat = multi_stat(x, y, args.eta, args.t_sgd, rng_sgd, args.init_scale)
        p_value = np.nan
        reject_05 = st > cutoff_95
        reject_01 = st > cutoff_99
        rows.append(
            {
                "scenario": scenario,
                "d": d,
                "rho": rho,
                "rep": rep,
                "S_T": st,
                "p_value": p_value,
                "sigma_hat": sigma_hat,
                "scale_estimator": "cross-fitted quadratic intercept",
                "cutoff_95": cutoff_95,
                "cutoff_99": cutoff_99,
                "reject_05": bool(reject_05),
                "reject_01": bool(reject_01),
            }
        )
    return rows


def summarize(raw):
    h1 = raw[raw["scenario"] != "H0 calibration"].copy()
    summaries = []
    for keys, g in h1.groupby(["scenario", "d", "rho"], sort=False):
        scenario, d, rho = keys
        summaries.append(
            {
                "scenario": scenario,
                "d": int(d),
                "rho": float(rho),
                "n_h1": int(len(g)),
                "mean_S_T": float(g["S_T"].mean()),
                "sd_S_T": float(g["S_T"].std(ddof=1)),
                "min_S_T": float(g["S_T"].min()),
                "cutoff_95": float(g["cutoff_95"].dropna().iloc[0])
                if g["cutoff_95"].notna().any()
                else np.nan,
                "cutoff_99": float(g["cutoff_99"].dropna().iloc[0])
                if g["cutoff_99"].notna().any()
                else np.nan,
                "power_0.05": float(g["reject_05"].mean()),
                "power_0.01": float(g["reject_01"].mean()),
            }
        )
    return pd.DataFrame(summaries)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-data", type=int, default=200_000)
    parser.add_argument("--t-sgd", type=int, default=200_000)
    parser.add_argument("--eta", type=float, default=0.005)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--n-h1", type=int, default=100)
    parser.add_argument("--n-cal", type=int, default=500)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.n_data = min(args.n_data, 20_000)
        args.t_sgd = min(args.t_sgd, 20_000)
        args.n_h1 = min(args.n_h1, 10)
        args.n_cal = min(args.n_cal, 20)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "power_H1_rigorous_raw.csv"
    summary_path = args.out_dir / "power_H1_rigorous_summary.csv"
    config_path = args.out_dir / "power_H1_rigorous_config.json"

    config = vars(args).copy()
    config["out_dir"] = str(config["out_dir"])
    config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    settings = [
        ("No endogeneity", 1, 0.0, 1_000),
        ("No endogeneity", 2, 0.0, 2_000),
        ("No endogeneity", 5, 0.0, 3_000),
        ("No endogeneity", 10, 0.0, 4_000),
        ("Endogeneity", 1, 0.0, 5_000),
        ("Endogeneity", 1, 0.5, 6_000),
        ("Endogeneity", 1, 0.9, 7_000),
        ("Endogeneity", 2, 0.0, 8_000),
        ("Endogeneity", 2, 0.5, 9_000),
        ("Endogeneity", 2, 0.9, 10_000),
        ("Endogeneity", 5, 0.0, 11_000),
        ("Endogeneity", 5, 0.5, 12_000),
        ("Endogeneity", 5, 0.9, 13_000),
    ]

    print(
        f"H1 rigorous run: n={args.n_data}, T={args.t_sgd}, eta={args.eta}, "
        f"delta={args.delta}, h1={args.n_h1}, cal={args.n_cal}",
        flush=True,
    )
    t0 = time.time()
    rows = []
    for scenario, d, rho, seed_base in settings:
        started = time.time()
        cutoff_95, cutoff_99, cal_rows = h0_cutoffs(args, d, rho, seed_base)
        rows.extend(cal_rows)
        rows.extend(h1_rows(args, scenario, d, rho, cutoff_95, cutoff_99, seed_base))

        raw = pd.DataFrame(rows)
        raw.to_csv(raw_path, index=False)
        summary = summarize(raw)
        summary.to_csv(summary_path, index=False)
        last = summary.iloc[-1]
        print(
            f"{scenario:15s} d={d:2d} rho={rho:.1f}: "
            f"S_T={last['mean_S_T']:.3f}+/-{last['sd_S_T']:.3f}, "
            f"power05={last['power_0.05']:.3f} [{time.time() - started:.1f}s]",
            flush=True,
        )

    print(f"Saved {raw_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    print(f"Done in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
