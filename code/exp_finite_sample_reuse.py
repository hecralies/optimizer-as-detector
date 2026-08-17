"""Audit Type-I error under the repeated-data real-application protocol.

Each Monte Carlo replicate draws one finite null dataset, reuses it for the
specified number of SGD passes, and reruns nuisance estimation and calibration.
The default configuration reproduces Supplementary Table S1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from exp_categorical_pooled_remedy import pooled_null_calibration  # noqa: E402
from exp_realworld_robust import (  # noqa: E402
    _standardize,
    competitive_gap,
    robust_component_scale,
    scalar_null_calibration,
)


def run_one(
    *,
    n: int,
    replicate: int,
    eta: float,
    passes: int,
    seeds: int,
    mc_size: int,
    design: str,
    binary_probability: float,
) -> dict[str, object]:
    rng = np.random.default_rng(2_026_081_600 + 100_000 * n + replicate)
    if design == "binary":
        x = rng.binomial(1, binary_probability, size=n).astype(float)
    else:
        x = rng.normal(size=n)
    y = rng.normal(size=n)
    x, y = _standardize(x, y)

    if design == "binary":
        fitted_null = pooled_null_calibration(x, y, eta, 0.05)
        sigma = fitted_null["sigma_pool"]
        center = fitted_null["null_center"]
        null_sd = fitted_null["null_sd"]
        calibration = "pooled_distribution_adaptive"
    else:
        sigma, _, _ = robust_component_scale(
            x, y, fold_seed=20_260_730 + replicate
        )
        center, null_sd, _ = scalar_null_calibration(
            x,
            sigma,
            eta=eta,
            mc_size=mc_size,
            seed=20_260_731 + replicate,
        )
        calibration = "component_gaussian_analytic"

    statistics = [
        competitive_gap(
            x,
            y,
            sigma,
            eta=eta,
            passes=passes,
            seed=10_000 * replicate + seed,
        )
        for seed in range(seeds)
    ]
    mean_statistic = float(np.mean(statistics))
    cutoff = float(center + stats.norm.ppf(0.95) * null_sd)
    return {
        "design": "binary" if design == "binary" else "gaussian_scalar",
        "n": n,
        "calibration": calibration,
        "replicate": replicate,
        "mean_statistic": mean_statistic,
        "critical_value": cutoff,
        "reject": mean_statistic > cutoff,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--eta", type=float, default=0.0005)
    parser.add_argument("--passes", type=int, default=1000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--mc-size", type=int, default=20_000)
    parser.add_argument("--sizes", type=int, nargs="+", default=[150, 333, 777, 1840])
    parser.add_argument("--binary-n", type=int, default=777)
    parser.add_argument("--binary-probability", type=float, default=0.484)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.reps = min(args.reps, 3)
        args.passes = min(args.passes, 5)
        args.seeds = min(args.seeds, 2)
        args.mc_size = min(args.mc_size, 2_000)
        args.sizes = args.sizes[:1]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["out_dir"] = str(args.out_dir.resolve())
    config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (args.out_dir / "finite_n_reuse_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    settings = [("gaussian", n) for n in args.sizes]
    settings.append(("binary", args.binary_n))
    started = time.time()
    for design, n in settings:
        print(f"{design}: n={n}, reps={args.reps}", flush=True)
        for replicate in range(args.reps):
            rows.append(
                run_one(
                    n=n,
                    replicate=replicate,
                    eta=args.eta,
                    passes=args.passes,
                    seeds=args.seeds,
                    mc_size=args.mc_size,
                    design=design,
                    binary_probability=args.binary_probability,
                )
            )
            if (replicate + 1) % 25 == 0:
                print(f"  completed {replicate + 1}/{args.reps}", flush=True)

    raw = pd.DataFrame(rows)
    summary = (
        raw.groupby(["design", "n", "calibration"], sort=False)["reject"]
        .agg(reps="size", rejections="sum", type1_rate="mean")
        .reset_index()
    )
    summary["mc_se"] = np.sqrt(
        summary["type1_rate"] * (1.0 - summary["type1_rate"]) / summary["reps"]
    )
    raw.to_csv(args.out_dir / "finite_n_reuse_type1_raw.csv", index=False)
    summary.to_csv(args.out_dir / "finite_n_reuse_type1_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
