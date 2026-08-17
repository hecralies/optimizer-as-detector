"""
Verify the heteroskedasticity-robust correction to the competitive SGD test.

The original test uses:
    mu0 = 2 * kappa_eps * kappa_x
        = 2 * (E[|resid|]/sigma_hat) * (E[|x|]/sqrt(E[x^2]))
    e_star = E[|resid|] * E[|x|] / E[x^2]

When |resid| and |x| are positively correlated, the
product-of-marginals UNDERESTIMATES the true ODE equilibrium, causing
inflated rejection rates (false positives).  The reverse covariance
deflates the true equilibrium relative to the plug-in reference.

The corrected test uses the joint moment:
    mu0_corrected = 2 * E[|resid| * |x|] / (sigma_hat * sqrt(E[x^2]))
    e_star_corrected = E[|resid| * |x|] / E[x^2]

Under homoskedasticity, the two are identical (independence).
Under heteroskedasticity, the joint moment correctly tracks the shifted
ODE equilibrium.

Under H1, with T=n (one pass), the gap S_T barely exceeds the (corrected)
H0 equilibrium, so the corrected test needs T >> n for power.  We test
with a T/n multiplier to verify power at higher T/n ratios.

Writes:
  results/heterosked_fix_raw.csv
  results/heterosked_fix_summary.csv
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


# ─────────────────────────────────────────────────────────────────────────────
# Numba-compiled SGD
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _sgd_scalar_centered(x, y, idx, eta, init_gap):
    """Competitive SGD for scalar (d=1) data with centered x, y."""
    n = x.shape[0]
    sxx = 0.0
    sxy = 0.0
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


# ─────────────────────────────────────────────────────────────────────────────
# Data generation
# ─────────────────────────────────────────────────────────────────────────────

def center_xy(x, y):
    return x - x.mean(), y - y.mean()


def gen_data_h0(pattern, n, gamma, sigma, rng):
    """Generate H0 data (single theta*=1) with various noise patterns."""
    theta_star = 1.0
    x = rng.standard_normal(n)

    if pattern == "homoskedastic":
        eps = rng.standard_normal(n) * sigma
    elif pattern == "linear":
        sd = sigma * np.sqrt(1.0 + gamma * np.abs(x))
        eps = rng.standard_normal(n) * sd
    elif pattern == "quadratic":
        sd = sigma * np.sqrt(1.0 + gamma * x**2)
        eps = rng.standard_normal(n) * sd
    elif pattern == "inverse_quadratic":
        sd = sigma * np.sqrt(1.0 + gamma / (1.0 + x**2))
        eps = rng.standard_normal(n) * sd
    elif pattern == "multiplicative":
        sd = sigma * np.abs(x)
        sd = np.maximum(sd, 1e-8)
        eps = rng.standard_normal(n) * sd
    else:
        raise ValueError(f"unknown pattern: {pattern}")

    y = theta_star * x + eps
    return x, y


def gen_data_h1(pattern, n, gamma, sigma, delta, rng):
    """
    Generate H1 data: mixture of two slopes theta*=1+delta/2 and 1-delta/2.
    Each observation is assigned to one component with probability 0.5.
    """
    theta_high = 1.0 + delta / 2.0
    theta_low = 1.0 - delta / 2.0
    x = rng.standard_normal(n)
    assignment = rng.random(n) < 0.5

    if pattern == "homoskedastic":
        eps = rng.standard_normal(n) * sigma
    elif pattern == "linear":
        sd = sigma * np.sqrt(1.0 + gamma * np.abs(x))
        eps = rng.standard_normal(n) * sd
    elif pattern == "quadratic":
        sd = sigma * np.sqrt(1.0 + gamma * x**2)
        eps = rng.standard_normal(n) * sd
    else:
        raise ValueError(f"unknown H1 pattern: {pattern}")

    y = np.where(assignment, theta_high * x, theta_low * x) + eps
    return x, y


# ─────────────────────────────────────────────────────────────────────────────
# Trial runner: computes both original and corrected p-values
# ─────────────────────────────────────────────────────────────────────────────

def run_trial(x_raw, y_raw, eta, t_sgd, rng_sgd, init_scale):
    """
    Run one trial.  Returns a dict with both original and corrected test results.

    The corrected test replaces:
      mu0 = 2 * kappa_eps * kappa_x  (product of marginals)
    with:
      mu0 = 2 * E[|resid|*|x|] / (sigma_hat * sqrt(E[x^2]))  (joint moment)

    and similarly for e_star in the variance D computation.
    """
    x, y = center_xy(x_raw, y_raw)
    beta_ols = float(np.dot(x, y) / np.dot(x, x))
    resid = y - beta_ols * x
    sigma_hat = float(np.sqrt(np.mean(resid**2)))
    sx = float(np.mean(x**2))

    # Run SGD
    idx = rng_sgd.integers(0, x.shape[0], size=t_sgd, dtype=np.int64)
    theta0, theta1 = _sgd_scalar_centered(
        x.astype(np.float64), y.astype(np.float64), idx, eta, init_scale * sigma_hat
    )
    st = abs(theta0 - theta1) * np.sqrt(sx) / sigma_hat

    # ── Original test (product of marginals) ──
    kappa_eps = float(np.mean(np.abs(resid)) / sigma_hat)
    kappa_x = float(np.mean(np.abs(x)) / np.sqrt(sx))
    mu0_orig = 2.0 * kappa_eps * kappa_x
    e_star_orig = float(np.mean(np.abs(resid)) * np.mean(np.abs(x)) / sx)
    d_hat_orig = float(np.mean(
        (resid - e_star_orig * x)**2 * x**2 * (resid * x >= 0)
    ))
    sigma_t_orig = float(np.sqrt(2.0 * eta * d_hat_orig / sigma_hat**2))
    if sigma_t_orig < 1e-12:
        p_orig = 1.0
    else:
        p_orig = float(1.0 - norm.cdf((st - mu0_orig) / sigma_t_orig))

    # ── Corrected test (joint moment) ──
    mu0_corr = 2.0 * float(np.mean(np.abs(resid) * np.abs(x))) / (
        sigma_hat * np.sqrt(sx)
    )
    e_star_corr = float(np.mean(np.abs(resid) * np.abs(x)) / sx)
    d_hat_corr = float(np.mean(
        (resid - e_star_corr * x)**2 * x**2 * (resid * x >= 0)
    ))
    sigma_t_corr = float(np.sqrt(2.0 * eta * d_hat_corr / sigma_hat**2))
    if sigma_t_corr < 1e-12:
        p_corr = 1.0
    else:
        p_corr = float(1.0 - norm.cdf((st - mu0_corr) / sigma_t_corr))

    return {
        "S_T": st,
        "mu0_orig": mu0_orig,
        "mu0_corr": mu0_corr,
        "sigma_T_orig": sigma_t_orig,
        "sigma_T_corr": sigma_t_corr,
        "e_star_orig": e_star_orig,
        "e_star_corr": e_star_corr,
        "p_orig": p_orig,
        "p_corr": p_corr,
        "sigma_hat": sigma_hat,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Experiment settings
# ─────────────────────────────────────────────────────────────────────────────

def get_settings():
    """Return list of (label, hypothesis, pattern, gamma, delta, seed_base)."""
    settings = []

    # ── H0 settings ──
    settings.append(("H0: homoskedastic",         "H0", "homoskedastic",  0.0, 0.0, 1_000))
    settings.append(("H0: linear g=0.5",          "H0", "linear",         0.5, 0.0, 2_000))
    settings.append(("H0: linear g=1.0",          "H0", "linear",         1.0, 0.0, 3_000))
    settings.append(("H0: linear g=2.0",          "H0", "linear",         2.0, 0.0, 4_000))
    settings.append(("H0: quad g=0.5",            "H0", "quadratic",      0.5, 0.0, 5_000))
    settings.append(("H0: quad g=1.0",            "H0", "quadratic",      1.0, 0.0, 6_000))
    settings.append(("H0: quad g=2.0",            "H0", "quadratic",      2.0, 0.0, 7_000))
    settings.append(("H0: multiplicative",        "H0", "multiplicative", 0.0, 0.0, 8_000))
    settings.append(("H0: inverse-quad g=2.0",    "H0", "inverse_quadratic", 2.0, 0.0, 9_000))

    # ── H1 settings (power check) with T/n > 1 ──
    settings.append(("H1: homo D=0.5",            "H1", "homoskedastic",  0.0, 0.5, 11_000))
    settings.append(("H1: homo D=1.0",            "H1", "homoskedastic",  0.0, 1.0, 12_000))
    settings.append(("H1: homo D=2.0",            "H1", "homoskedastic",  0.0, 2.0, 17_000))
    settings.append(("H1: linear g=1.0 D=1.0",   "H1", "linear",         1.0, 1.0, 14_000))
    settings.append(("H1: linear g=1.0 D=2.0",   "H1", "linear",         1.0, 2.0, 18_000))
    settings.append(("H1: quad g=1.0 D=1.0",     "H1", "quadratic",      1.0, 1.0, 16_000))
    settings.append(("H1: quad g=1.0 D=2.0",     "H1", "quadratic",      1.0, 2.0, 19_000))

    return settings


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def summarize(raw):
    summaries = []
    for keys, g in raw.groupby(["label", "hypothesis", "pattern", "gamma", "delta"],
                                sort=False):
        label, hyp, pattern, gamma, delta = keys
        n_t = len(g)
        mean_st = float(g["S_T"].mean())
        sd_st = float(g["S_T"].std(ddof=1))
        mean_mu0_orig = float(g["mu0_orig"].mean())
        mean_mu0_corr = float(g["mu0_corr"].mean())
        rej05_orig = float((g["p_orig"] < 0.05).mean())
        rej05_corr = float((g["p_corr"] < 0.05).mean())
        rej01_orig = float((g["p_orig"] < 0.01).mean())
        rej01_corr = float((g["p_corr"] < 0.01).mean())

        summaries.append({
            "label": label,
            "hypothesis": hyp,
            "pattern": pattern,
            "gamma": float(gamma),
            "delta": float(delta),
            "n_trials": n_t,
            "mean_S_T": mean_st,
            "sd_S_T": sd_st,
            "mean_mu0_orig": mean_mu0_orig,
            "mean_mu0_corr": mean_mu0_corr,
            "rej05_orig": rej05_orig,
            "rej05_corr": rej05_corr,
            "rej01_orig": rej01_orig,
            "rej01_corr": rej01_corr,
        })
    return pd.DataFrame(summaries)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify heteroskedasticity-robust correction to competitive SGD test"
    )
    parser.add_argument("--n-data", type=int, default=200_000)
    parser.add_argument("--t-sgd", type=int, default=200_000)
    parser.add_argument("--t-mult", type=int, default=1,
                        help="Multiplier for T/n ratio for H1 settings (default 1)")
    parser.add_argument("--eta", type=float, default=0.005)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true",
                        help="Fast mode: n=20000, T=20000, 20 trials")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.n_data = min(args.n_data, 20_000)
        args.t_sgd = min(args.t_sgd, 20_000)
        args.n_trials = min(args.n_trials, 20)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "heterosked_fix_raw.csv"
    summary_path = args.out_dir / "heterosked_fix_summary.csv"
    config_path = args.out_dir / "heterosked_fix_config.json"

    config = vars(args).copy()
    config["out_dir"] = str(config["out_dir"])
    config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    settings = get_settings()

    mode = "QUICK" if args.quick else "FULL"
    print(f"{'='*70}")
    print(f"Heteroskedasticity Fix Verification [{mode}]")
    print(f"{'='*70}")
    print(f"  n={args.n_data}, T={args.t_sgd}, eta={args.eta}, "
          f"trials={args.n_trials}, init_scale={args.init_scale}")
    print(f"  T/n multiplier for H1: {args.t_mult}")
    print(f"  Settings: {len(settings)} ({sum(1 for s in settings if s[1]=='H0')} H0, "
          f"{sum(1 for s in settings if s[1]=='H1')} H1)")
    print()

    # Header
    print(f"{'Setting':<30s} {'mean_ST':>8s} {'mu0_o':>7s} {'mu0_c':>7s} "
          f"{'rej_o':>7s} {'rej_c':>7s} {'time':>6s}")
    print("-" * 70)

    t0 = time.time()
    all_rows = []

    for label, hyp, pattern, gamma, delta, seed_base in settings:
        t_start = time.time()
        rows = []

        # For H1, use T*t_mult iterations to give SGD more time to separate
        t_sgd_eff = args.t_sgd * args.t_mult if hyp == "H1" else args.t_sgd

        for rep in range(args.n_trials):
            rng_data = np.random.default_rng(seed_base + rep)
            rng_sgd = np.random.default_rng(seed_base + 100_000 + rep)

            if hyp == "H0":
                x, y = gen_data_h0(pattern, args.n_data, gamma, args.sigma, rng_data)
            else:
                x, y = gen_data_h1(pattern, args.n_data, gamma, args.sigma, delta, rng_data)

            result = run_trial(x, y, args.eta, t_sgd_eff, rng_sgd, args.init_scale)
            result.update({
                "label": label,
                "hypothesis": hyp,
                "pattern": pattern,
                "gamma": gamma,
                "delta": delta,
                "rep": rep,
                "t_sgd_eff": t_sgd_eff,
            })
            rows.append(result)

        all_rows.extend(rows)

        # Incremental save
        raw = pd.DataFrame(all_rows)
        raw.to_csv(raw_path, index=False)
        summary = summarize(raw)
        summary.to_csv(summary_path, index=False)

        # Print progress
        last = summary[summary["label"] == label].iloc[0]
        elapsed = time.time() - t_start
        print(f"{label:<30s} {last['mean_S_T']:8.4f} {last['mean_mu0_orig']:7.4f} "
              f"{last['mean_mu0_corr']:7.4f} {last['rej05_orig']:7.3f} "
              f"{last['rej05_corr']:7.3f} {elapsed:5.1f}s")

    # ── Final summary ──
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")

    summary = summarize(pd.DataFrame(all_rows))

    print(f"\n{'--- H0 Settings (Target: corrected rej <= 0.10) ---':^70s}")
    print(f"{'Setting':<30s} {'rej_orig':>10s} {'rej_corr':>10s} {'PASS?':>8s}")
    print("-" * 60)
    h0_pass = True
    for _, r in summary[summary["hypothesis"] == "H0"].iterrows():
        ok = r["rej05_corr"] <= 0.10
        if not ok:
            h0_pass = False
        tag = "OK" if ok else "FAIL"
        print(f"{r['label']:<30s} {r['rej05_orig']:10.3f} {r['rej05_corr']:10.3f} {tag:>8s}")

    print(f"\n{'--- H1 Settings (Target: high power for corrected) ---':^70s}")
    print(f"{'Setting':<30s} {'rej_orig':>10s} {'rej_corr':>10s}")
    print("-" * 60)
    for _, r in summary[summary["hypothesis"] == "H1"].iterrows():
        print(f"{r['label']:<30s} {r['rej05_orig']:10.3f} {r['rej05_corr']:10.3f}")

    total_time = time.time() - t0
    print(f"\nSaved: {raw_path}")
    print(f"Saved: {summary_path}")
    print(f"Total time: {total_time:.1f}s")

    if h0_pass:
        print("\nVERDICT: ALL H0 settings PASS (corrected rej <= 0.10 at alpha=0.05)")
    else:
        print("\nVERDICT: SOME H0 settings FAIL (corrected rej > 0.10)")


if __name__ == "__main__":
    main()
