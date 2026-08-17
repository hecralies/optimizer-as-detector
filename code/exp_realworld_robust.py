"""Robust component-scale Stage 1 experiment for the nine JASA datasets.

The data definitions are imported from ``code/exp_realworld_K2_rigorous.py`` so
that this rerun uses exactly the observations and focal regressions used by the
paper.  The detection statistic is normalized by a randomized two-fold
cross-fitted quadratic-intercept estimate of the component noise scale.

Scalar datasets use the Gaussian-component analytic upper-tail calibration.
Multivariate datasets use a fitted Gaussian-component, empirical-covariate
bootstrap that reruns scale estimation and competitive SGD in every replicate.
In both cases, the decision rule rejects when S_T exceeds the fitted
(1-alpha) null critical value.

Outputs
-------
realworld_robust_raw.csv
realworld_robust_summary.csv
realworld_robust_null_draws.csv
realworld_robust_config.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

try:
    from numba import njit
    from numba import config as numba_config

    HAS_NUMBA = True
except Exception:  # pragma: no cover
    HAS_NUMBA = False
    numba_config = None


if HAS_NUMBA:

    @njit(cache=False)
    def _sgd_scalar(x, y, theta, indices, eta):
        for t in range(indices.shape[0]):
            i = indices[t]
            r0 = y[i] - theta[0] * x[i]
            r1 = y[i] - theta[1] * x[i]
            if r0 * r0 <= r1 * r1:
                theta[0] += eta * r0 * x[i]
            else:
                theta[1] += eta * r1 * x[i]
        return theta

    @njit(cache=False)
    def _sgd_multi(x, y, theta, indices, eta):
        d = x.shape[1]
        for t in range(indices.shape[0]):
            i = indices[t]
            r0 = y[i]
            r1 = y[i]
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

else:  # pragma: no cover
    _sgd_scalar = None
    _sgd_multi = None


def _load_dataset_module(project_root: Path):
    path = project_root / "code" / "exp_realworld_K2_rigorous.py"
    if not path.exists():
        raise FileNotFoundError(f"dataset loader not found: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("realworld_dataset_source", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # The legacy module imports an old cached SGD core even though its data
    # loaders do not need it.  Stub that dependency to avoid loading stale
    # compiled artifacts; this runner supplies its own kernels above.
    old_core = sys.modules.get("sgd_competitive_numba")
    stub = types.ModuleType("sgd_competitive_numba")
    stub.HAS_NUMBA = False
    stub._gen_perms = None
    stub._sgd_loop_multi = None
    stub.sgd_competitive = None
    sys.modules["sgd_competitive_numba"] = stub
    old_disable_jit = numba_config.DISABLE_JIT if numba_config is not None else None
    if numba_config is not None:
        numba_config.DISABLE_JIT = 1
    try:
        spec.loader.exec_module(module)
    finally:
        if numba_config is not None:
            numba_config.DISABLE_JIT = old_disable_jit
        if old_core is None:
            sys.modules.pop("sgd_competitive_numba", None)
        else:
            sys.modules["sgd_competitive_numba"] = old_core
    return module


def _as_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def _standardize(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = _as_matrix(x)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    x = x[mask]
    y = y[mask]
    x_sd = x.std(axis=0)
    if np.any(x_sd <= 1e-12) or y.std() <= 1e-12:
        raise RuntimeError("degenerate focal covariate or outcome")
    x = (x - x.mean(axis=0)) / x_sd
    y = (y - y.mean()) / y.std()
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _pooled_ols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    sigma_x = x.T @ x / x.shape[0]
    return np.linalg.solve(
        sigma_x + 1e-12 * np.eye(x.shape[1]),
        x.T @ y / x.shape[0],
    )


def _quadratic_design(x: np.ndarray) -> np.ndarray:
    n, d = x.shape
    terms = [np.ones(n, dtype=np.float64)]
    for i in range(d):
        for j in range(i, d):
            terms.append(x[:, i] * x[:, j])
    return np.column_stack(terms)


def robust_component_scale(
    x: np.ndarray,
    y: np.ndarray,
    fold_seed: int,
) -> tuple[float, float, dict[str, float]]:
    """Estimate component and pooled scales using cross-fitted second moments."""
    n, d = x.shape
    rng = np.random.default_rng(fold_seed)
    order = rng.permutation(n)
    fold = np.empty(n, dtype=np.int8)
    fold[order] = np.arange(n, dtype=np.int64) & 1
    residual = np.empty(n, dtype=np.float64)

    for held_out in (0, 1):
        train = fold != held_out
        test = ~train
        beta = _pooled_ols(x[train], y[train])
        residual[test] = y[test] - x[test] @ beta

    q_design = _quadratic_design(x)
    coef, _, rank, singular = np.linalg.lstsq(
        q_design, residual * residual, rcond=None
    )
    intercept = float(coef[0])
    n_terms = q_design.shape[1]
    if rank < n_terms:
        raise RuntimeError(
            f"quadratic scale is unidentified: rank {rank} < {n_terms}"
        )
    if not np.isfinite(intercept) or intercept <= 0.0:
        raise RuntimeError(
            f"nonpositive quadratic-regression intercept: {intercept}"
        )
    condition = (
        float(singular[0] / singular[-1])
        if singular.size and singular[-1] > 0
        else np.inf
    )
    sigma_component = float(np.sqrt(intercept))
    sigma_pool = float(np.sqrt(np.mean(residual * residual)))
    diagnostics = {
        "scale_intercept": intercept,
        "scale_rank": int(rank),
        "scale_terms": int(n_terms),
        "scale_condition": condition,
    }
    return sigma_component, sigma_pool, diagnostics


def competitive_gap(
    x: np.ndarray,
    y: np.ndarray,
    sigma_component: float,
    eta: float,
    passes: int,
    seed: int,
) -> float:
    n, d = x.shape
    beta = _pooled_ols(x, y)
    direction = np.ones(d, dtype=np.float64) / np.sqrt(d)
    theta = np.vstack(
        [
            beta + 0.01 * sigma_component * direction,
            beta - 0.01 * sigma_component * direction,
        ]
    ).astype(np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=n * passes, dtype=np.int64)

    if d == 1:
        theta_scalar = theta[:, 0].copy()
        if HAS_NUMBA:
            theta_scalar = _sgd_scalar(x[:, 0], y, theta_scalar, indices, eta)
        else:  # pragma: no cover
            for i in indices:
                r = y[i] - theta_scalar * x[i, 0]
                w = int(np.argmin(r * r))
                theta_scalar[w] += eta * r[w] * x[i, 0]
        gap = abs(theta_scalar[0] - theta_scalar[1]) * np.sqrt(
            np.mean(x[:, 0] ** 2)
        )
    else:
        if HAS_NUMBA:
            theta = _sgd_multi(x, y, theta, indices, eta)
        else:  # pragma: no cover
            for i in indices:
                residual = y[i] - theta @ x[i]
                w = int(np.argmin(residual * residual))
                theta[w] += eta * residual[w] * x[i]
        sigma_x = x.T @ x / n
        diff = theta[0] - theta[1]
        gap = float(np.sqrt(diff @ sigma_x @ diff))
    return float(gap / sigma_component)


def scalar_null_calibration(
    x: np.ndarray,
    sigma_component: float,
    eta: float,
    mc_size: int,
    seed: int,
) -> tuple[float, float, float]:
    """Gaussian-component scalar null center and diffusion scale."""
    x1 = x[:, 0]
    sx = float(np.mean(x1 * x1))
    mean_abs_x = float(np.mean(np.abs(x1)))
    kappa_x = mean_abs_x / np.sqrt(sx)
    mu0 = 2.0 * np.sqrt(2.0 / np.pi) * kappa_x
    e_star = sigma_component * np.sqrt(2.0 / np.pi) * mean_abs_x / sx

    rng = np.random.default_rng(seed)
    x_draw = x1[rng.integers(0, len(x1), size=mc_size)]
    eps = rng.normal(0.0, sigma_component, size=mc_size)
    event = eps * x_draw >= 0.0
    diffusion = float(
        np.mean((eps - e_star * x_draw) ** 2 * x_draw**2 * event)
    )
    sigma_t = float(
        np.sqrt(2.0 * eta * diffusion / sigma_component**2)
    )
    return mu0, sigma_t, diffusion


def multivariate_null_bootstrap(
    x: np.ndarray,
    sigma_component: float,
    eta: float,
    passes: int,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Run the complete fitted Gaussian-component null pipeline."""
    n = x.shape[0]
    beta = np.zeros(x.shape[1], dtype=np.float64)
    draws: list[float] = []
    diagnostics: list[dict[str, float]] = []
    master = np.random.default_rng(seed)

    for b in range(replicates):
        run_seed = int(master.integers(1, 2**31 - 1))
        rng = np.random.default_rng(run_seed)
        x_null = x[rng.integers(0, n, size=n)].copy()
        x_null -= x_null.mean(axis=0)
        eps = rng.normal(0.0, sigma_component, size=n)
        y_null = x_null @ beta + eps
        y_null -= y_null.mean()
        sigma_null, sigma_pool_null, diag = robust_component_scale(
            x_null, y_null, fold_seed=run_seed + 17
        )
        statistic = competitive_gap(
            x_null,
            y_null,
            sigma_null,
            eta=eta,
            passes=passes,
            seed=run_seed + 31,
        )
        draws.append(statistic)
        diagnostics.append(
            {
                "replicate": b,
                "seed": run_seed,
                "S_T": statistic,
                "sigma_component": sigma_null,
                "sigma_pool": sigma_pool_null,
                **diag,
            }
        )
    return np.asarray(draws), diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerun Stage 1 on the nine JASA datasets with robust scale."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument("--eta", type=float, default=0.0005)
    parser.add_argument("--passes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--multi-null-reps", type=int, default=499)
    parser.add_argument("--scalar-mc-size", type=int, default=2_000_000)
    parser.add_argument("--scale-seed", type=int, default=20260730)
    parser.add_argument("--null-seed", type=int, default=20260731)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.quick:
        args.passes = min(args.passes, 5)
        args.seeds = min(args.seeds, 2)
        args.multi_null_reps = min(args.multi_null_reps, 9)
        args.scalar_mc_size = min(args.scalar_mc_size, 100_000)

    source = _load_dataset_module(project_root)
    selected = (
        {name.lower() for name in args.datasets}
        if args.datasets
        else None
    )
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    start = time.time()

    print("=" * 88, flush=True)
    print("ROBUST COMPONENT-SCALE REAL-WORLD STAGE 1 RERUN", flush=True)
    print("=" * 88, flush=True)

    for loader in source.LOADERS:
        try:
            dataset = loader()
            name = dataset["name"]
            if selected and name.lower() not in selected:
                continue
            x, y = _standardize(dataset["x"], dataset["y"])
            n, d = x.shape
            sigma_component, sigma_pool, scale_diag = robust_component_scale(
                x, y, fold_seed=args.scale_seed
            )
            print(
                f"\n{name}: n={n}, d={d}, "
                f"sigma_component={sigma_component:.6g}, "
                f"sigma_pool={sigma_pool:.6g}",
                flush=True,
            )

            observed = []
            for seed in range(args.seeds):
                statistic = competitive_gap(
                    x,
                    y,
                    sigma_component,
                    eta=args.eta,
                    passes=args.passes,
                    seed=seed,
                )
                observed.append(statistic)
                raw_rows.append(
                    {
                        "Dataset": name,
                        "seed": seed,
                        "S_T": statistic,
                        "sigma_component": sigma_component,
                        "sigma_pool": sigma_pool,
                        "scale_ratio": sigma_component / sigma_pool,
                        **scale_diag,
                    }
                )

            observed_array = np.asarray(observed)
            mean_statistic = float(observed_array.mean())
            if d == 1:
                mu0, sigma_t, diffusion = scalar_null_calibration(
                    x,
                    sigma_component,
                    eta=args.eta,
                    mc_size=args.scalar_mc_size,
                    seed=args.null_seed,
                )
                critical_value = float(
                    mu0 + stats.norm.ppf(1.0 - args.alpha) * sigma_t
                )
                p_value = float(stats.norm.sf((mean_statistic - mu0) / sigma_t))
                null_method = "scalar Gaussian-component analytic"
                null_reps = args.scalar_mc_size
            else:
                draws, diagnostics = multivariate_null_bootstrap(
                    x,
                    sigma_component,
                    eta=args.eta,
                    passes=args.passes,
                    replicates=args.multi_null_reps,
                    seed=args.null_seed,
                )
                critical_value = float(
                    np.quantile(draws, 1.0 - args.alpha, method="higher")
                )
                mu0 = float(draws.mean())
                sigma_t = float(draws.std(ddof=1))
                p_value = float(
                    (1.0 + np.count_nonzero(draws >= mean_statistic))
                    / (len(draws) + 1.0)
                )
                diffusion = np.nan
                null_method = "empirical-covariate Gaussian-component bootstrap"
                null_reps = len(draws)
                for row in diagnostics:
                    null_rows.append({"Dataset": name, **row})

            summary = {
                "Dataset": name,
                "status": "ok",
                "n": n,
                "d": d,
                "eta": args.eta,
                "passes": args.passes,
                "n_seeds": args.seeds,
                "S_T_mean": mean_statistic,
                "S_T_sd": float(observed_array.std(ddof=1)),
                "sigma_component": sigma_component,
                "sigma_pool": sigma_pool,
                "scale_ratio": sigma_component / sigma_pool,
                "null_method": null_method,
                "null_reps": null_reps,
                "null_center": mu0,
                "null_sd": sigma_t,
                "alpha": args.alpha,
                "critical_value": critical_value,
                "reject": mean_statistic > critical_value,
                "p_value": p_value,
                "diffusion_D": diffusion,
                "source": dataset["source"],
                **scale_diag,
            }
            summary_rows.append(summary)
            print(
                f"  S_T={mean_statistic:.4f}, "
                f"q_{1.0 - args.alpha:.2f}={critical_value:.4f}, "
                f"reject={mean_statistic > critical_value}, p={p_value:.4g}",
                flush=True,
            )
        except Exception as exc:
            name = locals().get(
                "name",
                loader.__name__.replace("load_", "").replace("_", " ").title(),
            )
            print(f"\n{name}: FAILED: {exc}", flush=True)
            summary_rows.append(
                {
                    "Dataset": name,
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(limit=3),
                }
            )

    raw = pd.DataFrame(raw_rows)
    summary = pd.DataFrame(summary_rows)
    null = pd.DataFrame(null_rows)
    raw.to_csv(out_dir / "realworld_robust_raw.csv", index=False)
    summary.to_csv(out_dir / "realworld_robust_summary.csv", index=False)
    null.to_csv(out_dir / "realworld_robust_null_draws.csv", index=False)
    config = {
        "script": Path(__file__).name,
        "project_root": str(project_root),
        "eta": args.eta,
        "passes": args.passes,
        "seeds": args.seeds,
        "multi_null_reps": args.multi_null_reps,
        "scalar_mc_size": args.scalar_mc_size,
        "scale_seed": args.scale_seed,
        "null_seed": args.null_seed,
        "alpha": args.alpha,
        "quick": args.quick,
        "has_numba": HAS_NUMBA,
        "elapsed_seconds": time.time() - start,
        "scale_estimator": (
            "randomized two-fold cross-fitted pooled regression; "
            "OLS intercept in full distinct quadratic second-moment regression"
        ),
    }
    with (out_dir / "realworld_robust_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(config, stream, indent=2)

    print("\n" + "=" * 88, flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved outputs to {out_dir}", flush=True)
    print(f"Elapsed: {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
