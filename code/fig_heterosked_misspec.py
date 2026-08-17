"""Plot the heteroskedasticity and nonlinear-misspecification experiments.

The figure uses the 200-trial summaries already produced by
``exp_heterosked_fix_verify.py`` and ``exp_misspecification.py``.  It writes
vector and high-resolution raster versions for the manuscript.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "fig"

ORANGE = "#D55E00"
BLUE = "#0072B2"
GRAY = "#5A5A5A"
BLACK = "#1A1A1A"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def wilson_interval(rate: np.ndarray, n: np.ndarray, alpha: float = 0.05):
    """Return lower and upper Wilson confidence limits for binomial rates."""
    z = norm.ppf(1.0 - alpha / 2.0)
    denominator = 1.0 + z**2 / n
    center = (rate + z**2 / (2.0 * n)) / denominator
    half_width = z * np.sqrt(rate * (1.0 - rate) / n + z**2 / (4.0 * n**2))
    half_width /= denominator
    return center - half_width, center + half_width


def add_error_bars(ax, x, y, n, color, fmt, label=None, zorder=3):
    lower, upper = wilson_interval(np.asarray(y), np.asarray(n))
    ax.errorbar(
        x,
        100.0 * np.asarray(y),
        yerr=np.vstack(
            [100.0 * (np.asarray(y) - lower), 100.0 * (upper - np.asarray(y))]
        ),
        color=color,
        fmt=fmt,
        capsize=2.2,
        linewidth=1.3,
        elinewidth=0.9,
        markersize=5.2,
        markeredgewidth=0.9,
        label=label,
        zorder=zorder,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--fig-dir", type=Path, default=FIGURES)
    args = parser.parse_args()

    hetero = pd.read_csv(args.results_dir / "heterosked_fix_summary.csv")
    misspec = pd.read_csv(args.results_dir / "misspecification_summary.csv")

    hetero = hetero.loc[hetero["hypothesis"].eq("H0")].copy()
    order = [
        ("homoskedastic", 0.0),
        ("linear", 0.5),
        ("linear", 1.0),
        ("linear", 2.0),
        ("quadratic", 0.5),
        ("quadratic", 1.0),
        ("quadratic", 2.0),
        ("multiplicative", 0.0),
        ("inverse_quadratic", 2.0),
    ]
    lookup = hetero.set_index(["pattern", "gamma"])
    hetero = lookup.loc[order].reset_index()
    hetero_labels = [
        "Homo.",
        ".5",
        "1",
        "2",
        ".5",
        "1",
        "2",
        "Mult.",
        "Inv.\n2",
    ]

    nonlinear = misspec.loc[
        misspec["category"].eq("nonlinearity")
        & misspec["label"].str.contains("linear \(baseline\)|quadratic", regex=True)
    ].copy()
    nonlinear["beta"] = [0.0 if "baseline" in label else value for label, value in zip(nonlinear["label"], [0.0, 0.2, 0.5, 1.0])]
    nonlinear = nonlinear.sort_values("beta")

    fig1, ax1 = plt.subplots(figsize=(3.65, 2.65))

    x = np.arange(len(hetero))
    raw = hetero["rej05_orig"].to_numpy()
    corrected = hetero["rej05_corr"].to_numpy()
    n = hetero["n_trials"].to_numpy()

    add_error_bars(
        ax1, x, raw, n, ORANGE, "o-", label="Homoskedastic", zorder=3
    )
    add_error_bars(
        ax1, x, corrected, n, BLUE, "s--", label="Joint-moment", zorder=4
    )
    ax1.axhline(5.0, color=GRAY, linestyle="--", linewidth=1.1, zorder=1)
    ax1.set_ylabel("Rejection rate (%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(hetero_labels)
    ax1.set_xlim(-0.25, len(x) - 0.75)
    ax1.text(
        2,
        -0.18,
        r"Linear ($\gamma$)",
        transform=ax1.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.5,
    )
    ax1.text(
        5,
        -0.18,
        r"Quadratic ($\gamma$)",
        transform=ax1.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=8.5,
    )
    ax1.set_ylim(0.0, 105.0)
    ax1.set_yticks(np.arange(0.0, 101.0, 20.0))
    ax1.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        borderaxespad=0.0,
        frameon=False,
        handlelength=2.0,
        ncol=2,
    )

    inset = ax1.inset_axes([0.42, 0.14, 0.42, 0.30])
    low, high = wilson_interval(corrected, n)
    inset.errorbar(
        x,
        100.0 * corrected,
        yerr=np.vstack(
            [100.0 * (corrected - low), 100.0 * (high - corrected)]
        ),
        color=BLUE,
        fmt="s--",
        capsize=1.5,
        linewidth=0.9,
        elinewidth=0.7,
        markersize=3.0,
        markeredgewidth=0.7,
        zorder=3,
    )
    inset.axhline(5.0, color=GRAY, linestyle="--", linewidth=0.8, zorder=1)
    inset.set_xlim(-0.25, len(x) - 0.75)
    inset.set_ylim(-0.25, 8.0)
    inset.set_yticks([0.0, 4.0, 8.0])
    inset.set_xticks([])
    inset.set_title("Joint-moment zoom", fontsize=7.0, pad=1.5)
    inset.tick_params(axis="both", labelsize=6.5, length=2.0, width=0.6)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)

    fig2, ax2 = plt.subplots(figsize=(3.65, 2.65))

    beta = nonlinear["beta"].to_numpy()
    rates = nonlinear["rej_0.05"].to_numpy()
    n_nonlin = nonlinear["n_trials"].to_numpy()
    add_error_bars(
        ax2,
        beta,
        rates,
        n_nonlin,
        BLACK,
        "o-",
    )
    ax2.axhline(5.0, color=GRAY, linestyle="--", linewidth=1.1, zorder=1)
    ax2.set_xlabel(r"Quadratic coefficient $\beta$")
    ax2.set_ylabel("Rejection rate (%)")
    ax2.set_xticks(beta)
    ax2.set_xticklabels(["0", "0.2", "0.5", "1.0"])
    ax2.set_xlim(-0.07, 1.07)
    ax2.set_ylim(0.0, 105.0)
    ax2.set_yticks(np.arange(0.0, 101.0, 20.0))

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", length=3.0, width=0.8)

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    fig1.savefig(args.fig_dir / "fig_heterosked_rejection.pdf")
    fig1.savefig(args.fig_dir / "fig_heterosked_rejection.png", dpi=400)
    fig2.savefig(args.fig_dir / "fig_misspec_rejection.pdf")
    fig2.savefig(args.fig_dir / "fig_misspec_rejection.png", dpi=400)
    plt.close(fig1)
    plt.close(fig2)
    print("Saved title-free, grid-free heteroskedasticity and misspecification panels")


if __name__ == "__main__":
    main()
