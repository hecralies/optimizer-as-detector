"""Matched pooled-residual calibration for binary focal covariates.

This complements ``exp_realworld_robust.py`` when the focal covariate is
categorical and latent group membership may depend on it.  The statistic is
normalized by the ordinary pooled-OLS residual scale.  Its scalar fitted-null
center and diffusion scale use the empirical standardized absolute moments of
the pooled residuals and covariate, yielding a distribution-adaptive one-sided
upper-tail test.

Outputs
-------
results/categorical_pooled_remedy_raw.csv
results/categorical_pooled_remedy_summary.csv
results/categorical_pooled_remedy_config.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from exp_realworld_robust import (
    _load_dataset_module,
    _pooled_ols,
    _standardize,
    competitive_gap,
)


DEFAULT_DATASETS = ("Taylor", "Atlanta CES")


def pooled_null_calibration(
    x: np.ndarray,
    y: np.ndarray,
    eta: float,
    alpha: float,
) -> dict[str, float]:
    if x.shape[1] != 1:
        raise ValueError("the categorical pooled remedy is scalar")
    x1 = x[:, 0]
    beta = float(_pooled_ols(x, y)[0])
    residual = y - beta * x1
    sigma_pool = float(np.sqrt(np.mean(residual**2)))
    sx = float(np.mean(x1**2))
    mean_abs_x = float(np.mean(np.abs(x1)))
    mean_abs_residual = float(np.mean(np.abs(residual)))
    kappa_x = mean_abs_x / np.sqrt(sx)
    kappa_residual = mean_abs_residual / sigma_pool
    null_center = 2.0 * kappa_residual * kappa_x
    equilibrium_gap = mean_abs_residual * mean_abs_x / sx
    diffusion = float(
        np.mean(
            (residual - equilibrium_gap * x1) ** 2
            * x1**2
            * (residual * x1 >= 0.0)
        )
    )
    null_sd = float(np.sqrt(2.0 * eta * diffusion / sigma_pool**2))
    critical_value = float(
        null_center + stats.norm.ppf(1.0 - alpha) * null_sd
    )
    return {
        "sigma_pool": sigma_pool,
        "kappa_x": kappa_x,
        "kappa_residual": kappa_residual,
        "null_center": null_center,
        "null_sd": null_sd,
        "critical_value": critical_value,
        "diffusion_D": diffusion,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--eta", type=float, default=0.0005)
    parser.add_argument("--passes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = _load_dataset_module(project_root)
    selected = set(args.datasets)
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    started = time.time()

    for loader in source.LOADERS:
        dataset = loader()
        name = dataset["name"]
        if name not in selected:
            continue
        x, y = _standardize(dataset["x"], dataset["y"])
        calibration = pooled_null_calibration(x, y, args.eta, args.alpha)
        observed = []
        for seed in range(args.seeds):
            statistic = competitive_gap(
                x,
                y,
                calibration["sigma_pool"],
                eta=args.eta,
                passes=args.passes,
                seed=seed,
            )
            observed.append(statistic)
            raw_rows.append(
                {
                    "Dataset": name,
                    "seed": seed,
                    "S_T_pool": statistic,
                    **calibration,
                }
            )
        observed_array = np.asarray(observed)
        mean_statistic = float(observed_array.mean())
        p_value = float(
            stats.norm.sf(
                (mean_statistic - calibration["null_center"])
                / calibration["null_sd"]
            )
        )
        summary_rows.append(
            {
                "Dataset": name,
                "n": len(y),
                "d": 1,
                "eta": args.eta,
                "passes": args.passes,
                "n_seeds": args.seeds,
                "S_T_mean": mean_statistic,
                "S_T_sd": float(observed_array.std(ddof=1)),
                "alpha": args.alpha,
                "reject": mean_statistic > calibration["critical_value"],
                "p_value": p_value,
                "calibration_method": (
                    "pooled-residual distribution-adaptive analytic"
                ),
                "source": dataset["source"],
                **calibration,
            }
        )
        print(
            f"{name}: S_pool={mean_statistic:.4f}, "
            f"q_{1.0 - args.alpha:.2f}={calibration['critical_value']:.4f}, "
            f"p={p_value:.4g}",
            flush=True,
        )

    pd.DataFrame(raw_rows).to_csv(
        args.out_dir / "categorical_pooled_remedy_raw.csv", index=False
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        args.out_dir / "categorical_pooled_remedy_summary.csv", index=False
    )
    robust_path = args.out_dir / "realworld_robust_summary.csv"
    if robust_path.exists():
        combined = pd.read_csv(robust_path)
        combined["selected_method"] = (
            "robust component scale; Gaussian-component calibration"
        )
        for _, remedy in summary.iterrows():
            mask = combined["Dataset"] == remedy["Dataset"]
            combined.loc[mask, "S_T_mean"] = remedy["S_T_mean"]
            combined.loc[mask, "S_T_sd"] = remedy["S_T_sd"]
            combined.loc[mask, "sigma_component"] = np.nan
            combined.loc[mask, "sigma_pool"] = remedy["sigma_pool"]
            combined.loc[mask, "scale_ratio"] = np.nan
            combined.loc[mask, "null_method"] = remedy["calibration_method"]
            combined.loc[mask, "null_center"] = remedy["null_center"]
            combined.loc[mask, "null_sd"] = remedy["null_sd"]
            combined.loc[mask, "alpha"] = remedy["alpha"]
            combined.loc[mask, "critical_value"] = remedy["critical_value"]
            combined.loc[mask, "reject"] = remedy["reject"]
            combined.loc[mask, "p_value"] = remedy["p_value"]
            combined.loc[mask, "diffusion_D"] = remedy["diffusion_D"]
            combined.loc[mask, "scale_intercept"] = np.nan
            combined.loc[mask, "scale_rank"] = np.nan
            combined.loc[mask, "scale_terms"] = np.nan
            combined.loc[mask, "scale_condition"] = np.nan
            combined.loc[mask, "selected_method"] = (
                "pooled residual scale; distribution-adaptive calibration"
            )
        combined.to_csv(
            args.out_dir / "realworld_stage1_summary.csv", index=False
        )
    config = {
        "script": Path(__file__).name,
        "project_root": str(project_root),
        "datasets": args.datasets,
        "eta": args.eta,
        "passes": args.passes,
        "seeds": args.seeds,
        "alpha": args.alpha,
        "elapsed_seconds": time.time() - started,
        "scale_estimator": "ordinary pooled-OLS residual RMS",
        "null_calibration": (
            "empirical residual/covariate shape constants with scalar "
            "diffusion approximation"
        ),
        "combined_summary": "realworld_stage1_summary.csv",
    }
    (args.out_dir / "categorical_pooled_remedy_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
