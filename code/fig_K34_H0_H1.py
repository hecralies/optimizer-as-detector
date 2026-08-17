"""Generate matched H0/H1 illustrations for K=3 and K=4 competitive SGD.

The K=3 panels support Proposition 8 in the main paper.  The K=4 panels are
exploratory and are placed in the Supplementary Materials.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "fig"
FIG_DIR.mkdir(exist_ok=True)

THETA0 = 2.0
SIGMA_X = 1.0
SIGMA_EPS = 1.0
M = 0.5
DELTA_STAR = 2 * M * SIGMA_X / SIGMA_EPS
ETA = 0.002
T = 300_000
RECORD_EVERY = 25
PLOT_EVERY = 4
SEED = 20_260_730
BURN_IN = 50_000
MIN_AVERAGING_SPAN = 20_000

C3 = 1.3201367134
C1_K4 = 0.4511909
C2_K4 = 2.1717553

PANEL_W = 4.15
PANEL_H = 2.75
DPI = 400

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.9,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


def g_cdf(u):
    return 0.5 + (np.arctan(u) + u / (1 + u * u)) / np.pi


def g_first_moment(u):
    return -1 / (np.pi * (1 + u * u))


def right_drift(a, shift):
    cutoff = a / 2 - shift
    return (
        (shift - a) * (1 - g_cdf(cutoff))
        - g_first_moment(cutoff)
    )


def k3_h1_radius():
    lam = DELTA_STAR / 2

    def equilibrium(a):
        return 0.5 * (right_drift(a, lam) + right_drift(a, -lam))

    return brentq(equilibrium, 0.8, 1.8)


def simulate(K, under_h1):
    rng = np.random.default_rng(SEED)
    x = rng.normal(0, SIGMA_X, T)
    eps = rng.normal(0, SIGMA_EPS, T)
    if under_h1:
        component = rng.choice(np.array([-1.0, 1.0]), size=T)
        slope = THETA0 + M * component
    else:
        slope = np.full(T, THETA0)
    y = slope * x + eps

    theta = THETA0 + np.linspace(-0.03, 0.03, K)
    n_records = T // RECORD_EVERY
    theta_hist = np.empty((n_records, K))
    iterations = np.empty(n_records, dtype=int)
    rec = 0

    for t in range(T):
        residuals = y[t] - theta * x[t]
        winner = int(np.argmin(residuals * residuals))
        theta[winner] += ETA * residuals[winner] * x[t]
        if (t + 1) % RECORD_EVERY == 0:
            theta_hist[rec] = np.sort(theta)
            iterations[rec] = t + 1
            rec += 1

    return iterations, theta_hist


def moving_average(values, window=200):
    if values.ndim == 1:
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")
    return np.column_stack(
        [moving_average(values[:, j], window) for j in range(values.shape[1])]
    )


def save_panel(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", dpi=DPI)
    plt.close(fig)


def add_reference_family(ax, levels, color="#d62728"):
    for value in levels:
        ax.axhline(value, color=color, linestyle="--", linewidth=1.5, alpha=0.8)


def plot_trajectories(iterations, theta_hist, K, under_h1, name):
    smooth = moving_average(theta_hist)
    smooth_t = iterations[len(iterations) - len(smooth):] / 1e5
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    for k in range(K):
        ax.plot(
            smooth_t[::PLOT_EVERY],
            smooth[::PLOT_EVERY, k],
            color=colors[k],
            label=rf"$\theta_{{{k+1},t}}$",
        )

    if K == 3:
        radius = k3_h1_radius() if under_h1 else C3
        add_reference_family(
            ax,
            THETA0 + np.array([-radius, 0.0, radius]),
        )
        if under_h1:
            for true_slope in (THETA0 - M, THETA0 + M):
                ax.axhline(
                    true_slope,
                    color="black",
                    linestyle=":",
                    linewidth=1.4,
                )
    elif under_h1:
        for true_slope in (THETA0 - M, THETA0 + M):
            ax.axhline(
                true_slope,
                color="black",
                linestyle=":",
                linewidth=1.4,
            )
    else:
        add_reference_family(
            ax,
            THETA0 + np.array([-C2_K4, -C1_K4, C1_K4, C2_K4]),
        )

    ax.set_xlabel(r"Iteration $t$ ($\times 10^5$)")
    ax.set_ylabel("Parameter value")
    ax.set_xlim(0, T / 1e5)
    fig.tight_layout()
    save_panel(fig, name)


def pairwise_gaps(theta_hist):
    K = theta_hist.shape[1]
    gaps = []
    for i in range(K):
        for j in range(i + 1, K):
            gaps.append(theta_hist[:, j] - theta_hist[:, i])
    return np.sort(np.column_stack(gaps), axis=1)


def plot_k3_gap(iterations, theta_hist, under_h1, name):
    gap = theta_hist[:, -1] - theta_hist[:, 0]
    burn = iterations >= BURN_IN
    tail_gap = gap[burn]
    tail_t = iterations[burn]
    gap_mean = np.cumsum(tail_gap) / np.arange(1, len(tail_gap) + 1)
    keep = tail_t >= BURN_IN + MIN_AVERAGING_SPAN
    plot_t = tail_t[keep] / 1e5
    gap_mean = gap_mean[keep]

    A = k3_h1_radius()
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    ax.plot(
        plot_t[::PLOT_EVERY],
        gap_mean[::PLOT_EVERY],
        color="#1f77b4",
        label="Running mean outer gap",
    )
    ax.axhline(
        2 * C3,
        color="#d62728",
        linestyle="--",
        linewidth=1.7,
        label=rf"$H_0$: $2c_3={2*C3:.3f}$",
    )
    if under_h1:
        ax.axhline(
            2 * A,
            color="#2ca02c",
            linestyle="-.",
            linewidth=1.7,
            label=rf"$H_1$: $2A={2*A:.3f}$",
        )
    ax.set_xlabel(r"Iteration $t$ ($\times 10^5$)")
    ax.set_ylabel("Normalized outer gap")
    ax.set_xlim((BURN_IN + MIN_AVERAGING_SPAN) / 1e5, T / 1e5)
    ax.set_ylim(2.53, 2.69)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_panel(fig, name)


def plot_k4_gaps(iterations, theta_hist, under_h1, name):
    gaps = pairwise_gaps(theta_hist)
    burn = iterations >= BURN_IN
    tail_gaps = gaps[burn]
    tail_t = iterations[burn]
    gaps_mean = np.cumsum(tail_gaps, axis=0)
    gaps_mean /= np.arange(1, len(tail_gaps) + 1)[:, None]
    keep = tail_t >= BURN_IN + MIN_AVERAGING_SPAN
    plot_t = tail_t[keep] / 1e5
    gaps_mean = gaps_mean[keep]
    colors = [
        "#1f77b4",
        "#6baed6",
        "#7f7f7f",
        "#a6a6a6",
        "#fdae6b",
        "#e6550d",
    ]
    labels = ["min", "2nd", "3rd", "4th", "5th", "max"]

    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    for k in range(gaps_mean.shape[1]):
        ax.plot(
            plot_t[::PLOT_EVERY],
            gaps_mean[::PLOT_EVERY, k],
            color=colors[k],
            label=labels[k],
            linewidth=1.5,
        )
    if not under_h1:
        null_levels = [
            2 * C1_K4,
            C2_K4 - C1_K4,
            C1_K4 + C2_K4,
            2 * C2_K4,
        ]
        add_reference_family(ax, null_levels)
    ax.set_xlabel(r"Iteration $t$ ($\times 10^5$)")
    ax.set_ylabel("Mean pairwise gaps")
    ax.set_xlim((BURN_IN + MIN_AVERAGING_SPAN) / 1e5, T / 1e5)
    ax.set_ylim(0.7, 4.6)
    ax.legend(loc="center right", ncol=2, fontsize=8.5)
    fig.tight_layout()
    save_panel(fig, name)


def main():
    global FIG_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()
    FIG_DIR = args.fig_dir.resolve()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for K in (3, 4):
        for under_h1 in (False, True):
            key = (K, under_h1)
            results[key] = simulate(K, under_h1)

    for under_h1, suffix in ((False, "H0"), (True, "H1")):
        iterations, theta_hist = results[(3, under_h1)]
        plot_trajectories(
            iterations,
            theta_hist,
            K=3,
            under_h1=under_h1,
            name=f"simK3_{suffix}_trajectories",
        )
        plot_k3_gap(
            iterations,
            theta_hist,
            under_h1=under_h1,
            name=f"simK3_{suffix}_gap",
        )

        iterations, theta_hist = results[(4, under_h1)]
        plot_trajectories(
            iterations,
            theta_hist,
            K=4,
            under_h1=under_h1,
            name=f"simK4_{suffix}_trajectories",
        )
        plot_k4_gaps(
            iterations,
            theta_hist,
            under_h1=under_h1,
            name=f"simK4_{suffix}_gaps",
        )

    print(f"K=3 H1 radius A(delta*=1): {k3_h1_radius():.9f}")
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
