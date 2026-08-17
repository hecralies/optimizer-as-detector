"""
Regenerate the two H1 synthetic panels using the mixture-robust estimate of
the component noise scale from Remark 3 of jasa_main20260721.tex.

The estimator uses two-fold cross-fitted pooled residuals.  In the scalar
experiments it regresses squared residuals on (1, x^2).  In the multivariate
dimension experiment the data-generating covariance and mixture covariance
are diagonal, so the corresponding identified auxiliary regression uses
(1, x_1^2, ..., x_d^2).

Outputs
-------
results/synthetic_H1_robust_raw.csv
results/synthetic_H1_robust_summary.csv
fig/sim_ground_truth_paper_a.{pdf,png}
fig/sim_ground_truth_paper_b.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit


FOUR_PI = 4.0 / np.pi
BLUE = "#2176AE"
GREEN = "#57B894"
ORANGE = "#F0803C"
PURPLE = "#8B5CF6"
PINK = "#E74C8B"
BROWN = "#D97706"

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


@njit(cache=False)
def _sgd_scalar(x, y, idx, eta, beta_pool, init_gap):
    theta0 = beta_pool + init_gap
    theta1 = beta_pool - init_gap
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
def _sgd_multi(x, y, idx, eta, beta_pool, init_gap):
    d = x.shape[1]
    theta0 = beta_pool.copy()
    theta1 = beta_pool.copy()
    theta0[0] += init_gap
    theta1[0] -= init_gap
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


def _center(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y_centered = y - y.mean()
    if x.ndim == 1:
        return x - x.mean(), y_centered
    return x - x.mean(axis=0), y_centered


def _pooled_ols(x: np.ndarray, y: np.ndarray):
    if x.ndim == 1:
        return float(np.dot(x, y) / np.dot(x, x))
    sigma_x = x.T @ x / x.shape[0]
    return np.linalg.solve(sigma_x, x.T @ y / x.shape[0])


def robust_component_scale(
    x: np.ndarray, y: np.ndarray
) -> tuple[float, float]:
    """Return robust and pooled residual scales.

    Two deterministic folds make the pooled residuals cross-fitted.  The
    auxiliary design contains the intercept and the distinct diagonal
    quadratic terms.  This equals the full identified quadratic regression
    for the scalar panel and is the appropriate specialization for the
    diagonal multivariate designs used in panel (b).
    """
    n = y.shape[0]
    fold = np.arange(n) & 1
    residual = np.empty(n, dtype=np.float64)
    for held_out in (0, 1):
        train = fold != held_out
        test = ~train
        beta = _pooled_ols(x[train], y[train])
        if x.ndim == 1:
            residual[test] = y[test] - beta * x[test]
        else:
            residual[test] = y[test] - x[test] @ beta

    if x.ndim == 1:
        quadratic = (x * x)[:, None]
    else:
        quadratic = x * x
    design = np.column_stack((np.ones(n), quadratic))
    coef, *_ = np.linalg.lstsq(design, residual * residual, rcond=None)
    intercept = float(coef[0])
    if not np.isfinite(intercept) or intercept <= 0.0:
        raise RuntimeError(f"nonpositive quadratic-regression intercept: {intercept}")
    return float(np.sqrt(intercept)), float(np.sqrt(np.mean(residual * residual)))


def run_competitive_sgd(
    x_raw: np.ndarray,
    y_raw: np.ndarray,
    eta: float,
    steps: int,
    seed: int,
    init_scale: float,
) -> tuple[float, float, float]:
    x, y = _center(x_raw, y_raw)
    beta_pool = _pooled_ols(x, y)
    sigma_robust, sigma_pool = robust_component_scale(x, y)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, y.shape[0], size=steps, dtype=np.int64)
    init_gap = init_scale * sigma_robust
    if x.ndim == 1:
        theta0, theta1 = _sgd_scalar(
            x.astype(np.float64),
            y.astype(np.float64),
            idx,
            eta,
            beta_pool,
            init_gap,
        )
        gap = abs(theta0 - theta1) * np.sqrt(np.mean(x * x))
    else:
        theta0, theta1 = _sgd_multi(
            x.astype(np.float64),
            y.astype(np.float64),
            idx,
            eta,
            beta_pool.astype(np.float64),
            init_gap,
        )
        sigma_x = x.T @ x / x.shape[0]
        diff = theta0 - theta1
        gap = float(np.sqrt(diff @ sigma_x @ diff))
    return gap / sigma_robust, sigma_robust, sigma_pool


def generate_k_mixture(
    slopes: np.ndarray,
    d: int,
    n: int,
    sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if d == 1:
        x = rng.standard_normal(n)
        groups = rng.integers(0, slopes.shape[0], size=n)
        y = slopes[groups] * x + sigma * rng.standard_normal(n)
        return x, y
    x = rng.standard_normal((n, d))
    groups = rng.integers(0, slopes.shape[0], size=n)
    y = slopes[groups] * x[:, 0] + sigma * rng.standard_normal(n)
    return x, y


def generate_two_group(
    d: int, delta: float, n: int, sigma: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if d == 1:
        x = rng.standard_normal(n)
        groups = rng.integers(0, 2, size=n)
        slope = np.where(groups == 0, -delta, delta)
        y = slope * x + sigma * rng.standard_normal(n)
        return x, y
    x = rng.standard_normal((n, d))
    groups = rng.integers(0, 2, size=n)
    slope = np.where(groups == 0, -delta, delta)
    y = slope * x[:, 0] + sigma * rng.standard_normal(n)
    return x, y


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in raw.groupby(
        ["panel", "setting", "k_true", "d", "delta"], sort=False, dropna=False
    ):
        panel, setting, k_true, d, delta = keys
        rows.append(
            {
                "panel": panel,
                "setting": setting,
                "k_true": k_true,
                "d": d,
                "delta": delta,
                "n_seeds": len(group),
                "mean_S_T": group["S_T"].mean(),
                "sd_S_T": group["S_T"].std(ddof=1),
                "min_S_T": group["S_T"].min(),
                "fraction_above_null_center": group["above_null_center"].mean(),
                "mean_sigma_robust": group["sigma_robust"].mean(),
                "sd_sigma_robust": group["sigma_robust"].std(ddof=1),
                "mean_sigma_pool": group["sigma_pool"].mean(),
            }
        )
    return pd.DataFrame(rows)


def make_figures(raw: pd.DataFrame, fig_dir: Path) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)

    def add_boxplot(ax, data, positions, colors):
        box = ax.boxplot(
            data,
            positions=positions,
            widths=0.58,
            patch_artist=True,
            showfliers=True,
            whis=1.5,
            flierprops={
                "marker": "o",
                "markersize": 2.6,
                "markerfacecolor": "#555555",
                "markeredgecolor": "#555555",
                "alpha": 0.75,
            },
            medianprops={"color": "#222222", "linewidth": 1.4},
            boxprops={"linewidth": 1.0, "edgecolor": "#333333"},
            whiskerprops={"linewidth": 1.0, "color": "#555555"},
            capprops={"linewidth": 1.0, "color": "#555555"},
        )
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.58)

    def finish_axis(ax):
        ax.axhline(
            FOUR_PI,
            color="#D62728",
            linestyle="--",
            linewidth=1.7,
            zorder=0,
        )
        ax.set_ylabel(r"$\hat S_T$")

    panel_a = raw[raw["panel"] == "K"].copy()
    k_values = [2, 3, 4, 5, 6]
    dimensions = [1, 2, 5, 10]
    dimension_colors = [BLUE, GREEN, ORANGE, PURPLE]
    data = []
    box_colors = []
    positions = []
    tick_labels = []
    group_spans = []
    position = 1.0
    for k_true in k_values:
        start = position
        for d, color in zip(dimensions, dimension_colors):
            values = panel_a.loc[
                (panel_a["k_true"] == k_true) & (panel_a["d"] == d),
                "S_T",
            ]
            data.append(values.to_numpy())
            box_colors.append(color)
            positions.append(position)
            tick_labels.append(str(d))
            position += 1.0
        group_spans.append((k_true, start, position - 1.0))
        position += 0.65

    fig, ax = plt.subplots(figsize=(6.2, 3.15))
    add_boxplot(ax, data, positions, box_colors)
    finish_axis(ax)
    for (_, _, right), (_, next_left, _) in zip(group_spans[:-1], group_spans[1:]):
        ax.axvline((right + next_left) / 2, color="#B8B8B8", linewidth=0.7)
    for k_true, left, right in group_spans:
        ax.text(
            (left + right) / 2,
            1.015,
            rf"$K^*={k_true}$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=14,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(r"Dimension $d$", labelpad=2)
    ax.set_ylim(1.0, 6.9)
    ax.set_yticks(np.arange(1, 7, 1))
    ax.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)
    fig.tight_layout(pad=0.35)
    fig.savefig(fig_dir / "sim_ground_truth_paper_a.pdf")
    fig.savefig(fig_dir / "sim_ground_truth_paper_a.png", dpi=400)
    plt.close(fig)

    panel_b = raw[raw["panel"] == "dimension"].copy()
    dimensions = [1, 2, 5, 10]
    deltas = [0.3, 0.5, 1.0]
    colors = [BLUE, GREEN, ORANGE]
    data = []
    box_colors = []
    positions = []
    tick_labels = []
    group_spans = []
    position = 1.0
    for d in dimensions:
        start = position
        for delta, color in zip(deltas, colors):
            values = panel_b.loc[
                (panel_b["d"] == d) & np.isclose(panel_b["delta"], delta),
                "S_T",
            ]
            data.append(values.to_numpy())
            box_colors.append(color)
            positions.append(position)
            tick_labels.append(f"{delta:.1f}")
            position += 1.0
        group_spans.append((d, start, position - 1.0))
        position += 0.65

    fig, ax = plt.subplots(figsize=(6.2, 3.15))
    add_boxplot(ax, data, positions, box_colors)
    finish_axis(ax)
    for (_, _, right), (_, next_left, _) in zip(group_spans[:-1], group_spans[1:]):
        ax.axvline((right + next_left) / 2, color="#B8B8B8", linewidth=0.7)
    for d, left, right in group_spans:
        ax.text(
            (left + right) / 2,
            1.015,
            rf"$d={d}$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=14,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(r"Signal strength $\Delta$", labelpad=2)
    ax.set_ylim(1.1, 2.5)
    ax.set_yticks(np.arange(1.2, 2.51, 0.2))
    ax.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)
    fig.tight_layout(pad=0.35)
    fig.savefig(fig_dir / "sim_ground_truth_paper_b.pdf")
    fig.savefig(fig_dir / "sim_ground_truth_paper_b.png", dpi=400)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--eta", type=float, default=0.005)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--fig-dir", type=Path, default=Path("fig"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.n = min(args.n, 20_000)
        args.steps = min(args.steps, 20_000)
        args.seeds = min(args.seeds, 4)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["out_dir"] = str(config["out_dir"])
    config["fig_dir"] = str(config["fig_dir"])
    config["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (args.out_dir / "synthetic_H1_robust_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    k_scenarios = {
        2: np.array([2.0, -2.0]),
        3: np.array([3.0, 0.0, -3.0]),
        4: np.array([3.0, 1.0, -1.0, -3.0]),
        5: np.array([4.0, 2.0, 0.0, -2.0, -4.0]),
        6: np.array([5.0, 3.0, 1.0, -1.0, -3.0, -5.0]),
    }
    rows = []
    started = time.time()
    print(
        f"Robust H1 synthetic run: n={args.n}, T={args.steps}, eta={args.eta}, "
        f"seeds={args.seeds}",
        flush=True,
    )

    for k_true, slopes in k_scenarios.items():
        for d in [1, 2, 5, 10]:
            values = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(10_000 * k_true + 100 * d + seed)
                x, y = generate_k_mixture(slopes, d, args.n, args.sigma, rng)
                stat, sigma_robust, sigma_pool = run_competitive_sgd(
                    x,
                    y,
                    args.eta,
                    args.steps,
                    100_000 + 10_000 * k_true + 100 * d + seed,
                    args.init_scale,
                )
                values.append(stat)
                rows.append(
                    {
                        "panel": "K",
                        "setting": f"K*={k_true},d={d}",
                        "k_true": k_true,
                        "d": d,
                        "delta": np.nan,
                        "seed": seed,
                        "S_T": stat,
                        "above_null_center": stat > FOUR_PI,
                        "sigma_robust": sigma_robust,
                        "sigma_pool": sigma_pool,
                    }
                )
            print(
                f"K*={k_true}, d={d}: "
                f"S_T={np.mean(values):.3f} +/- {np.std(values, ddof=1):.3f}",
                flush=True,
            )

    for delta in [0.3, 0.5, 1.0]:
        for d in [1, 2, 5, 10]:
            values = []
            for seed in range(args.seeds):
                rng = np.random.default_rng(
                    200_000 + int(100 * delta) * 1_000 + d * 100 + seed
                )
                x, y = generate_two_group(d, delta, args.n, args.sigma, rng)
                stat, sigma_robust, sigma_pool = run_competitive_sgd(
                    x,
                    y,
                    args.eta,
                    args.steps,
                    300_000 + int(100 * delta) * 1_000 + d * 100 + seed,
                    args.init_scale,
                )
                values.append(stat)
                rows.append(
                    {
                        "panel": "dimension",
                        "setting": f"d={d},Delta={delta:.1f}",
                        "k_true": 2,
                        "d": d,
                        "delta": delta,
                        "seed": seed,
                        "S_T": stat,
                        "above_null_center": stat > FOUR_PI,
                        "sigma_robust": sigma_robust,
                        "sigma_pool": sigma_pool,
                    }
                )
            print(
                f"Delta={delta:.1f}, d={d}: "
                f"S_T={np.mean(values):.3f} +/- {np.std(values, ddof=1):.3f}",
                flush=True,
            )

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    raw.to_csv(args.out_dir / "synthetic_H1_robust_raw.csv", index=False)
    summary.to_csv(args.out_dir / "synthetic_H1_robust_summary.csv", index=False)
    make_figures(raw, args.fig_dir)
    print(summary.to_string(index=False), flush=True)
    print(f"Done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
