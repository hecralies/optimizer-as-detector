"""
Regenerate the SI Appendix 4/pi universality experiment.

Outputs:
  results/4pi_universality_raw.csv
  results/4pi_universality_summary.csv

The statistic is normalized by the two-fold cross-fitted quadratic-intercept
component-scale estimator.  Default parameters match the paper figure:
eta=0.005, T=200000, 30 seeds.
Use --quick for a short smoke test.
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from exp_synthetic_h1_robust import robust_component_scale

try:
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False


FOUR_PI = 4.0 / np.pi


if HAS_NUMBA:

    @njit(cache=False)
    def _sgd_scalar(x, y, theta, idx, eta):
        for t in range(idx.shape[0]):
            i = idx[t]
            xi = x[i]
            yi = y[i]
            r0 = yi - theta[0] * xi
            r1 = yi - theta[1] * xi
            if r0 * r0 <= r1 * r1:
                theta[0] += eta * r0 * xi
            else:
                theta[1] += eta * r1 * xi
        return theta

    @njit(cache=False)
    def _sgd_multi(x, y, theta, idx, eta):
        d = x.shape[1]
        for t in range(idx.shape[0]):
            i = idx[t]
            yi = y[i]
            r0 = yi
            r1 = yi
            for j in range(d):
                r0 -= theta[0, j] * x[i, j]
                r1 -= theta[1, j] * x[i, j]
            if r0 * r0 <= r1 * r1:
                for j in range(d):
                    theta[0, j] += eta * r0 * x[i, j]
            else:
                for j in range(d):
                    theta[1, j] += eta * r1 * x[i, j]
        return theta

else:
    _sgd_scalar = None
    _sgd_multi = None


def centered_ols_scalar(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    x_dm = x - x.mean()
    y_dm = y - y.mean()
    sx = float(np.mean(x_dm**2))
    beta = float(np.dot(x_dm, y_dm) / np.dot(x_dm, x_dm))
    resid = y_dm - beta * x_dm
    sigma = float(np.sqrt(np.mean(resid**2)))
    return beta, sx, sigma


def centered_ols_multi(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x_dm = x - x.mean(axis=0)
    y_dm = y - y.mean()
    n, d = x_dm.shape
    sigma_x = x_dm.T @ x_dm / n
    beta = np.linalg.solve(sigma_x + 1e-12 * np.eye(d), x_dm.T @ y_dm / n)
    resid = y_dm - x_dm @ beta
    sigma = float(np.sqrt(np.mean(resid**2)))
    return beta, sigma_x, sigma


def run_scalar(
    x: np.ndarray, y: np.ndarray, eta: float, n_steps: int, seed: int
) -> tuple[float, float, float, float]:
    x_dm = x - x.mean()
    y_dm = y - y.mean()
    beta, sx, _ = centered_ols_scalar(x, y)
    sigma_robust, sigma_pool = robust_component_scale(x_dm, y_dm)
    theta = np.array(
        [beta + 0.01 * sigma_robust, beta - 0.01 * sigma_robust],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=n_steps)
    if HAS_NUMBA:
        theta = _sgd_scalar(x_dm, y_dm, theta, idx, eta)
    else:  # pragma: no cover
        for i in idx:
            r0 = y_dm[i] - theta[0] * x_dm[i]
            r1 = y_dm[i] - theta[1] * x_dm[i]
            if r0 * r0 <= r1 * r1:
                theta[0] += eta * r0 * x_dm[i]
            else:
                theta[1] += eta * r1 * x_dm[i]
    gap = float(abs(theta[0] - theta[1]) * np.sqrt(sx))
    return gap / sigma_robust, gap / sigma_pool, sigma_robust, sigma_pool


def run_multi(
    x: np.ndarray, y: np.ndarray, eta: float, n_steps: int, seed: int
) -> tuple[float, float, float, float]:
    x_dm = x - x.mean(axis=0)
    y_dm = y - y.mean()
    beta, sigma_x, _ = centered_ols_multi(x, y)
    sigma_robust, sigma_pool = robust_component_scale(x_dm, y_dm)
    d = x.shape[1]
    direction = np.ones(d) / np.sqrt(d)
    theta = np.vstack(
        [
            beta + 0.01 * sigma_robust * direction,
            beta - 0.01 * sigma_robust * direction,
        ]
    )
    theta = np.asarray(theta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=n_steps)
    if HAS_NUMBA:
        theta = _sgd_multi(x_dm, y_dm, theta, idx, eta)
    else:  # pragma: no cover
        for i in idx:
            r0 = y_dm[i] - theta[0] @ x_dm[i]
            r1 = y_dm[i] - theta[1] @ x_dm[i]
            if r0 * r0 <= r1 * r1:
                theta[0] += eta * r0 * x_dm[i]
            else:
                theta[1] += eta * r1 * x_dm[i]
    diff = theta[0] - theta[1]
    gap = float(np.sqrt(diff @ sigma_x @ diff))
    return gap / sigma_robust, gap / sigma_pool, sigma_robust, sigma_pool


def gen_standard(setting: str, rng: np.random.Generator, n: int):
    if setting.startswith("d="):
        d = int(setting.split("=")[1])
        sigma = 1.0
        if d == 1:
            x = rng.standard_normal(n)
            y = x + sigma * rng.standard_normal(n)
        else:
            x = rng.standard_normal((n, d))
            y = x @ np.ones(d) + sigma * rng.standard_normal(n)
        return x, y
    if setting.startswith("sigma="):
        sigma = float(setting.split("=")[1])
        x = rng.standard_normal((n, 2))
        y = x @ np.ones(2) + sigma * rng.standard_normal(n)
        return x, y
    if setting == "Sigma=I":
        sigma_x = np.eye(3)
    elif setting == "Sigma=diag":
        sigma_x = np.diag([1.0, 2.0, 5.0])
    elif setting == "Sigma=rho0.5":
        sigma_x = np.full((3, 3), 0.5)
        np.fill_diagonal(sigma_x, 1.0)
    elif setting == "Sigma=rho0.8":
        sigma_x = np.full((3, 3), 0.8)
        np.fill_diagonal(sigma_x, 1.0)
    else:
        raise ValueError(setting)
    x = rng.standard_normal((n, 3)) @ np.linalg.cholesky(sigma_x).T
    y = x @ np.ones(3) + rng.standard_normal(n)
    return x, y


def gen_endogenous(d: int, rho: float, rng: np.random.Generator, n: int):
    sigma = 1.0
    if d == 1:
        cov = np.array([[1.0, rho * sigma], [rho * sigma, sigma**2]])
        z = rng.standard_normal((n, 2)) @ np.linalg.cholesky(cov).T
        x = z[:, 0]
        eps = z[:, 1]
        y = x + eps
        return x, y
    gamma = np.zeros(d)
    gamma[0] = rho * sigma
    cov = np.zeros((d + 1, d + 1))
    cov[:d, :d] = np.eye(d)
    cov[:d, d] = gamma
    cov[d, :d] = gamma
    cov[d, d] = sigma**2
    z = rng.standard_normal((n, d + 1)) @ np.linalg.cholesky(cov).T
    x = z[:, :d]
    eps = z[:, d]
    y = x @ np.ones(d) + eps
    return x, y


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in raw.groupby(["panel", "group", "setting_label", "d", "rho"], sort=False, dropna=False):
        panel, group_name, label, d, rho = keys
        vals = group["S_T"].to_numpy()
        sigma_robust = group["sigma_robust"].to_numpy()
        sigma_pool = group["sigma_pool"].to_numpy()
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        rows.append(
            {
                "panel": panel,
                "group": group_name,
                "setting_label": label,
                "d": d,
                "rho": rho,
                "n_seeds": len(vals),
                "mean_S_T": mean,
                "sd_S_T": sd,
                "rel_error": abs(mean - FOUR_PI) / FOUR_PI,
                "mean_sigma_robust": float(sigma_robust.mean()),
                "mean_sigma_pool": float(sigma_pool.mean()),
            }
        )
    return pd.DataFrame(rows)


def stable_offset(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) % 100_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eta", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.quick:
        args.steps = 20_000
        args.seeds = 5
    n = args.steps if args.n is None else args.n

    results_dir = args.out_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["out_dir"] = str(results_dir)
    config["n_effective"] = n
    config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (results_dir / "4pi_universality_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    standard = [
        ("Vary dimension", r"$d=1$", "d=1", 1, np.nan),
        ("Vary dimension", r"$d=2$", "d=2", 2, np.nan),
        ("Vary dimension", r"$d=5$", "d=5", 5, np.nan),
        ("Vary dimension", r"$d=10$", "d=10", 10, np.nan),
        ("Vary dimension", r"$d=20$", "d=20", 20, np.nan),
        ("Vary noise", r"$\sigma=0.5$", "sigma=0.5", 2, np.nan),
        ("Vary noise", r"$\sigma=1$", "sigma=1", 2, np.nan),
        ("Vary noise", r"$\sigma=2$", "sigma=2", 2, np.nan),
        ("Vary noise", r"$\sigma=5$", "sigma=5", 2, np.nan),
        ("Vary covariance", r"$I$", "Sigma=I", 3, np.nan),
        ("Vary covariance", "Diag", "Sigma=diag", 3, np.nan),
        ("Vary covariance", r"$\rho_x=0.5$", "Sigma=rho0.5", 3, np.nan),
        ("Vary covariance", r"$\rho_x=0.8$", "Sigma=rho0.8", 3, np.nan),
    ]
    endogenous = [
        (rf"$\rho={rho:.1f}$", d, rho)
        for d in [1, 2, 5]
        for rho in ([0.0, 0.2, 0.4, 0.6, 0.8, 0.9] if d == 1 else [0.0, 0.5, 0.9])
    ]

    raw_rows = []
    t0 = time.time()
    print(
        f"4/pi universality: eta={args.eta}, steps={args.steps}, n={n}, "
        f"seeds={args.seeds}, numba={HAS_NUMBA}",
        flush=True,
    )
    print(f"target 4/pi = {FOUR_PI:.6f}", flush=True)

    for group_name, label, code, d, rho in standard:
        vals = []
        offset = stable_offset(code)
        for seed in range(args.seeds):
            rng = np.random.default_rng(10_000 + 101 * seed + offset)
            x, y = gen_standard(code, rng, n)
            sgd_seed = 20_000 + 101 * seed + offset
            result = (
                run_scalar(x, y, args.eta, args.steps, sgd_seed)
                if d == 1
                else run_multi(x, y, args.eta, args.steps, sgd_seed)
            )
            st, st_pool, sigma_robust, sigma_pool = result
            vals.append(st)
            raw_rows.append(
                {
                    "panel": "standard",
                    "group": group_name,
                    "setting_label": label,
                    "setting_code": code,
                    "d": d,
                    "rho": rho,
                    "seed": seed,
                    "S_T": st,
                    "S_T_pool": st_pool,
                    "sigma_robust": sigma_robust,
                    "sigma_pool": sigma_pool,
                    "scale_estimator": "cross-fitted quadratic intercept",
                }
            )
        print(f"{label:14s} {np.mean(vals):.3f} +/- {np.std(vals, ddof=1):.3f}", flush=True)

    for label, d, rho in endogenous:
        vals = []
        group_name = rf"Endogeneity, $d={d}$"
        for seed in range(args.seeds):
            rng = np.random.default_rng(30_000 + 1_000 * d + int(100 * rho) * 17 + seed)
            x, y = gen_endogenous(d, rho, rng, n)
            sgd_seed = 40_000 + 1_000 * d + int(100 * rho) * 17 + seed
            result = (
                run_scalar(x, y, args.eta, args.steps, sgd_seed)
                if d == 1
                else run_multi(x, y, args.eta, args.steps, sgd_seed)
            )
            st, st_pool, sigma_robust, sigma_pool = result
            vals.append(st)
            raw_rows.append(
                {
                    "panel": "endogeneity",
                    "group": group_name,
                    "setting_label": label,
                    "setting_code": f"d={d},rho={rho:.1f}",
                    "d": d,
                    "rho": rho,
                    "seed": seed,
                    "S_T": st,
                    "S_T_pool": st_pool,
                    "sigma_robust": sigma_robust,
                    "sigma_pool": sigma_pool,
                    "scale_estimator": "cross-fitted quadratic intercept",
                }
            )
        print(f"d={d}, rho={rho:.1f} {np.mean(vals):.3f} +/- {np.std(vals, ddof=1):.3f}", flush=True)

    raw = pd.DataFrame(raw_rows)
    summary = summarize(raw)
    raw_path = results_dir / "4pi_universality_raw.csv"
    summary_path = results_dir / "4pi_universality_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"wrote {raw_path}")
    print(f"wrote {summary_path}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
