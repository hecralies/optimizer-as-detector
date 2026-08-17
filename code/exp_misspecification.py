"""
Misspecification robustness experiments for the competitive-SGD detector.

Part A: Nonlinearity under H_0 (single theta*, but y has nonlinear terms)
  - quadratic:  y = theta*x + beta*x^2 + eps
  - cubic:      y = theta*x + beta*x^3 + eps
  - sine:       y = theta*x + beta*sin(x) + eps

Part B: WLS-corrected heteroskedasticity
  - Estimate sigma(x) from OLS residuals, transform data, re-run SGD
  - Tests whether the WLS correction recovers the 4/pi calibration

Part C: Nonlinearity + heteroskedasticity combined

Writes:
  - results/misspecification_raw.csv
  - results/misspecification_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

if os.environ.get("NUMBA_DISABLE_JIT") == "1":
    def njit(*_args, **_kwargs):
        return lambda function: function
else:
    from numba import njit


@njit(cache=True)
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


def center_xy(x, y):
    y_c = y - y.mean()
    x_c = x - x.mean()
    return x_c, y_c


def run_scalar_trial(x, y, eta, t_sgd, rng_sgd, init_scale):
    """Run competitive SGD on (already centered) scalar data; return S_T and p-value."""
    beta_ols = float(np.dot(x, y) / np.dot(x, x))
    resid = y - beta_ols * x
    sigma_hat = float(np.sqrt(np.mean(resid**2)))
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
    sigma_t = float(np.sqrt(2.0 * eta * d_hat / sigma_hat**2))
    p_value = float(1.0 - norm.cdf((st - mu0) / sigma_t))
    return st, p_value, mu0, sigma_t, sigma_hat


# ── Data generation ──────────────────────────────────────────────────────

def gen_nonlinear(pattern, n, theta, beta_nl, sigma, rng):
    x = rng.standard_normal(n)
    eps = rng.standard_normal(n) * sigma
    if pattern == "linear":
        y = theta * x + eps
    elif pattern == "quadratic":
        y = theta * x + beta_nl * x**2 + eps
    elif pattern == "cubic":
        y = theta * x + beta_nl * x**3 + eps
    elif pattern == "sine":
        y = theta * x + beta_nl * np.sin(x) + eps
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    return x, y


def gen_heterosked(pattern, n, theta, gamma, sigma, rng):
    x = rng.standard_normal(n)
    if pattern == "homoskedastic":
        sd = np.full(n, sigma)
    elif pattern == "linear":
        sd = sigma * np.sqrt(1.0 + gamma * np.abs(x))
    elif pattern == "quadratic":
        sd = sigma * np.sqrt(1.0 + gamma * x**2)
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    eps = rng.standard_normal(n) * sd
    y = theta * x + eps
    return x, y, sd


def gen_combined(nl_pattern, het_pattern, n, theta, beta_nl, gamma, sigma, rng):
    x = rng.standard_normal(n)
    if het_pattern == "quadratic":
        sd = sigma * np.sqrt(1.0 + gamma * x**2)
    else:
        sd = np.full(n, sigma)
    eps = rng.standard_normal(n) * sd
    if nl_pattern == "quadratic":
        y = theta * x + beta_nl * x**2 + eps
    elif nl_pattern == "cubic":
        y = theta * x + beta_nl * x**3 + eps
    else:
        y = theta * x + eps
    return x, y, sd


# ── WLS correction ───────────────────────────────────────────────────────

def wls_transform(x, y):
    """Estimate sigma(x) from OLS residuals and return WLS-transformed data."""
    x_c, y_c = center_xy(x, y)
    beta_ols = np.dot(x_c, y_c) / np.dot(x_c, x_c)
    resid = y_c - beta_ols * x_c
    log_resid_sq = np.log(resid**2 + 1e-12)

    # Regress log(resid^2) on [1, x^2] to estimate log-variance
    n = len(x_c)
    X_var = np.column_stack([np.ones(n), x_c**2])
    gamma_hat = np.linalg.lstsq(X_var, log_resid_sq, rcond=None)[0]
    log_sigma2_hat = X_var @ gamma_hat
    sigma_hat_x = np.exp(0.5 * log_sigma2_hat)
    sigma_hat_x = np.maximum(sigma_hat_x, 1e-4)

    x_wls = x_c / sigma_hat_x
    y_wls = y_c / sigma_hat_x
    return x_wls, y_wls


# ── Experiment runners ───────────────────────────────────────────────────

def run_nonlinearity_experiments(args):
    rows = []
    settings = [
        ("linear (baseline)", "linear", 0.0, 1_000),
        ("quadratic β=0.2",  "quadratic", 0.2, 2_000),
        ("quadratic β=0.5",  "quadratic", 0.5, 3_000),
        ("quadratic β=1.0",  "quadratic", 1.0, 4_000),
        ("cubic β=0.1",      "cubic", 0.1, 5_000),
        ("cubic β=0.3",      "cubic", 0.3, 6_000),
        ("cubic β=0.5",      "cubic", 0.5, 7_000),
        ("sine β=0.5",       "sine", 0.5, 8_000),
        ("sine β=1.0",       "sine", 1.0, 9_000),
    ]
    for label, pattern, beta_nl, seed_base in settings:
        started = time.time()
        for rep in range(args.n_trials):
            rng_data = np.random.default_rng(seed_base + rep)
            rng_sgd = np.random.default_rng(seed_base + 50_000 + rep)
            x, y = gen_nonlinear(pattern, args.n_data, 1.0, beta_nl, args.sigma, rng_data)
            x_c, y_c = center_xy(x, y)
            st, pval, mu0, sigma_t, sigma_hat = run_scalar_trial(
                x_c, y_c, args.eta, args.t_sgd, rng_sgd, args.init_scale
            )
            rows.append({
                "category": "nonlinearity", "label": label, "pattern": pattern,
                "param": beta_nl, "correction": "none",
                "rep": rep, "S_T": st, "p_value": pval,
                "mu0": mu0, "sigma_hat": sigma_hat,
            })
        last_st = np.mean([r["S_T"] for r in rows if r["label"] == label])
        last_rej = np.mean([r["p_value"] < 0.05 for r in rows if r["label"] == label])
        print(f"  NL {label:22s}: mean_ST={last_st:.4f}, rej05={last_rej:.3f} [{time.time()-started:.1f}s]", flush=True)
    return rows


def run_wls_correction_experiments(args):
    rows = []
    settings = [
        ("homosked (baseline)",  "homoskedastic", 0.0, 20_000),
        ("quad γ=0.5 raw",       "quadratic", 0.5, 21_000),
        ("quad γ=1.0 raw",       "quadratic", 1.0, 22_000),
        ("quad γ=2.0 raw",       "quadratic", 2.0, 23_000),
        ("linear γ=1.0 raw",     "linear", 1.0, 24_000),
    ]
    for label, pattern, gamma, seed_base in settings:
        started = time.time()
        for rep in range(args.n_trials):
            rng_data = np.random.default_rng(seed_base + rep)

            if pattern == "homoskedastic":
                x, y, _ = gen_heterosked(pattern, args.n_data, 1.0, gamma, args.sigma, rng_data)
            else:
                x, y, _ = gen_heterosked(pattern, args.n_data, 1.0, gamma, args.sigma, rng_data)

            # Raw (uncorrected)
            rng_sgd = np.random.default_rng(seed_base + 50_000 + rep)
            x_c, y_c = center_xy(x, y)
            st_raw, pval_raw, mu0_raw, _, sh_raw = run_scalar_trial(
                x_c, y_c, args.eta, args.t_sgd, rng_sgd, args.init_scale
            )
            rows.append({
                "category": "wls_correction", "label": label, "pattern": pattern,
                "param": gamma, "correction": "none",
                "rep": rep, "S_T": st_raw, "p_value": pval_raw,
                "mu0": mu0_raw, "sigma_hat": sh_raw,
            })

            # WLS-corrected
            rng_sgd2 = np.random.default_rng(seed_base + 80_000 + rep)
            x_wls, y_wls = wls_transform(x, y)
            st_wls, pval_wls, mu0_wls, _, sh_wls = run_scalar_trial(
                x_wls, y_wls, args.eta, args.t_sgd, rng_sgd2, args.init_scale
            )
            rows.append({
                "category": "wls_correction",
                "label": label.replace(" raw", " WLS"),
                "pattern": pattern,
                "param": gamma, "correction": "wls",
                "rep": rep, "S_T": st_wls, "p_value": pval_wls,
                "mu0": mu0_wls, "sigma_hat": sh_wls,
            })
        raw_st = np.mean([r["S_T"] for r in rows if r["label"] == label])
        raw_rej = np.mean([r["p_value"] < 0.05 for r in rows if r["label"] == label])
        wls_label = label.replace(" raw", " WLS")
        wls_st = np.mean([r["S_T"] for r in rows if r["label"] == wls_label])
        wls_rej = np.mean([r["p_value"] < 0.05 for r in rows if r["label"] == wls_label])
        print(
            f"  WLS {label:22s}: raw ST={raw_st:.4f} rej={raw_rej:.3f} | "
            f"WLS ST={wls_st:.4f} rej={wls_rej:.3f} [{time.time()-started:.1f}s]",
            flush=True,
        )
    return rows


def run_combined_experiments(args):
    rows = []
    settings = [
        ("quad+het raw",  "quadratic", 0.5, "quadratic", 1.0, 30_000),
        ("cubic+het raw", "cubic",     0.3, "quadratic", 1.0, 31_000),
    ]
    for label, nl_pat, beta_nl, het_pat, gamma, seed_base in settings:
        started = time.time()
        for rep in range(args.n_trials):
            rng_data = np.random.default_rng(seed_base + rep)
            x, y, _ = gen_combined(nl_pat, het_pat, args.n_data, 1.0, beta_nl, gamma, args.sigma, rng_data)

            # Raw
            rng_sgd = np.random.default_rng(seed_base + 50_000 + rep)
            x_c, y_c = center_xy(x, y)
            st_raw, pval_raw, mu0_raw, _, sh_raw = run_scalar_trial(
                x_c, y_c, args.eta, args.t_sgd, rng_sgd, args.init_scale
            )
            rows.append({
                "category": "combined", "label": label, "pattern": f"{nl_pat}+{het_pat}",
                "param": beta_nl, "correction": "none",
                "rep": rep, "S_T": st_raw, "p_value": pval_raw,
                "mu0": mu0_raw, "sigma_hat": sh_raw,
            })

            # WLS-corrected
            rng_sgd2 = np.random.default_rng(seed_base + 80_000 + rep)
            x_wls, y_wls = wls_transform(x, y)
            st_wls, pval_wls, mu0_wls, _, sh_wls = run_scalar_trial(
                x_wls, y_wls, args.eta, args.t_sgd, rng_sgd2, args.init_scale
            )
            rows.append({
                "category": "combined",
                "label": label.replace(" raw", " WLS"),
                "pattern": f"{nl_pat}+{het_pat}",
                "param": beta_nl, "correction": "wls",
                "rep": rep, "S_T": st_wls, "p_value": pval_wls,
                "mu0": mu0_wls, "sigma_hat": sh_wls,
            })
        raw_st = np.mean([r["S_T"] for r in rows if r["label"] == label])
        raw_rej = np.mean([r["p_value"] < 0.05 for r in rows if r["label"] == label])
        wls_label = label.replace(" raw", " WLS")
        wls_st = np.mean([r["S_T"] for r in rows if r["label"] == wls_label])
        wls_rej = np.mean([r["p_value"] < 0.05 for r in rows if r["label"] == wls_label])
        print(
            f"  CMB {label:22s}: raw ST={raw_st:.4f} rej={raw_rej:.3f} | "
            f"WLS ST={wls_st:.4f} rej={wls_rej:.3f} [{time.time()-started:.1f}s]",
            flush=True,
        )
    return rows


def summarize(raw):
    summaries = []
    for keys, g in raw.groupby(["category", "label", "correction"], sort=False):
        cat, label, corr = keys
        n_t = len(g)
        mean_st = float(g["S_T"].mean())
        sd_st = float(g["S_T"].std(ddof=1))
        rej05 = float((g["p_value"] < 0.05).mean())
        rej01 = float((g["p_value"] < 0.01).mean())
        summaries.append({
            "category": cat, "label": label, "correction": corr,
            "n_trials": n_t, "mean_S_T": mean_st, "sd_S_T": sd_st,
            "4/pi": 4.0 / np.pi, "deviation": mean_st - 4.0 / np.pi,
            "rej_0.05": rej05, "rej_0.01": rej01,
        })
    return pd.DataFrame(summaries)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-data", type=int, default=200_000)
    parser.add_argument("--t-sgd", type=int, default=200_000)
    parser.add_argument("--eta", type=float, default=0.005)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.n_data = min(args.n_data, 20_000)
        args.t_sgd = min(args.t_sgd, 20_000)
        args.n_trials = min(args.n_trials, 20)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "misspecification_raw.csv"
    summary_path = args.out_dir / "misspecification_summary.csv"
    config_path = args.out_dir / "misspecification_config.json"

    config = vars(args).copy()
    config["out_dir"] = str(config["out_dir"])
    config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(
        f"Misspecification experiment: n={args.n_data}, T={args.t_sgd}, "
        f"eta={args.eta}, trials={args.n_trials}",
        flush=True,
    )
    t0 = time.time()

    print("\n=== Part A: Nonlinearity ===", flush=True)
    rows_nl = run_nonlinearity_experiments(args)

    print("\n=== Part B: WLS correction for heteroskedasticity ===", flush=True)
    rows_wls = run_wls_correction_experiments(args)

    print("\n=== Part C: Nonlinearity + heteroskedasticity ===", flush=True)
    rows_cmb = run_combined_experiments(args)

    all_rows = rows_nl + rows_wls + rows_cmb
    raw = pd.DataFrame(all_rows)
    raw.to_csv(raw_path, index=False)
    summary = summarize(raw)
    summary.to_csv(summary_path, index=False)

    print(f"\n{'='*70}")
    print(summary.to_string(index=False))
    print(f"\nSaved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
