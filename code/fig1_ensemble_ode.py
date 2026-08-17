"""Regenerate Figure 1 using ensemble means and across-path dispersion bands."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "fig"
FIG_DIR.mkdir(exist_ok=True)

THETA_TRUE = 2.0
SIGMA_X = 1.0
SIGMA_EPS = 1.0
ETA = 0.01
T = 50_000
N_PATHS = 200
RECORD_EVERY = 20
SEED = 20_260_730

C0 = 2 / np.pi
NULL_GAP = 2 * C0

C_BLUE = "#2176AE"
C_ORANGE = "#F0803C"
C_RED = "#D62728"

plt.rcParams.update(
    {
        "font.size": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 13,
        "lines.linewidth": 2.3,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)


def simulate_ensemble():
    rng = np.random.default_rng(SEED)
    theta1 = np.full(N_PATHS, THETA_TRUE + 0.01)
    theta2 = np.full(N_PATHS, THETA_TRUE - 0.01)

    n_records = T // RECORD_EVERY
    iterations = np.empty(n_records, dtype=int)
    lower_mean = np.empty(n_records)
    lower_sd = np.empty(n_records)
    upper_mean = np.empty(n_records)
    upper_sd = np.empty(n_records)
    gap_mean = np.empty(n_records)
    gap_sd = np.empty(n_records)
    rec = 0

    for t in range(T):
        x = rng.normal(0, SIGMA_X, N_PATHS)
        eps = rng.normal(0, SIGMA_EPS, N_PATHS)
        y = THETA_TRUE * x + eps
        r1 = y - theta1 * x
        r2 = y - theta2 * x
        winner1 = r1 * r1 <= r2 * r2
        theta1[winner1] += ETA * r1[winner1] * x[winner1]
        theta2[~winner1] += ETA * r2[~winner1] * x[~winner1]

        if (t + 1) % RECORD_EVERY == 0:
            lower = np.minimum(theta1, theta2)
            upper = np.maximum(theta1, theta2)
            gap = (upper - lower) * SIGMA_X / SIGMA_EPS
            iterations[rec] = t + 1
            lower_mean[rec] = lower.mean()
            lower_sd[rec] = lower.std(ddof=1)
            upper_mean[rec] = upper.mean()
            upper_sd[rec] = upper.std(ddof=1)
            gap_mean[rec] = gap.mean()
            gap_sd[rec] = gap.std(ddof=1)
            rec += 1

    return {
        "iterations": iterations,
        "lower_mean": lower_mean,
        "lower_sd": lower_sd,
        "upper_mean": upper_mean,
        "upper_sd": upper_sd,
        "gap_mean": gap_mean,
        "gap_sd": gap_sd,
    }


def save(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{suffix}", dpi=400)
    plt.close(fig)


def add_band(ax, x, mean, sd, color, label):
    ax.fill_between(
        x,
        mean - sd,
        mean + sd,
        color=color,
        alpha=0.18,
        linewidth=0,
    )
    ax.plot(x, mean, color=color, label=label)


def plot_figure(data):
    t = data["iterations"] / 1e4

    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    add_band(
        ax,
        t,
        data["lower_mean"],
        data["lower_sd"],
        C_BLUE,
        r"$\overline{\theta}_{L,t}$",
    )
    add_band(
        ax,
        t,
        data["upper_mean"],
        data["upper_sd"],
        C_ORANGE,
        r"$\overline{\theta}_{U,t}$",
    )
    ax.set_xlabel(r"Iteration $t$ ($\times 10^4$)")
    ax.set_ylabel("Parameter value")
    ax.set_xlim(0, T / 1e4)
    ax.legend(loc="center right")
    fig.tight_layout()
    save(fig, "sim0a_trajectories")

    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    add_band(
        ax,
        t,
        data["gap_mean"],
        data["gap_sd"],
        C_BLUE,
        r"$\overline{S}_t$",
    )
    ax.axhline(
        NULL_GAP,
        color=C_RED,
        linestyle="--",
        linewidth=2.0,
        label=r"$4/\pi\approx1.273$",
    )
    ax.set_xlabel(r"Iteration $t$ ($\times 10^4$)")
    ax.set_ylabel("Normalized gap")
    ax.set_xlim(0, T / 1e4)
    ax.set_ylim(0, 1.65)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "sim0b_Tn")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()
    FIG_DIR = args.fig_dir.resolve()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ensemble = simulate_ensemble()
    plot_figure(ensemble)
    tail = ensemble["gap_mean"][-500:].mean()
    print(f"Mean terminal ensemble gap: {tail:.6f}")
    print(f"Gaussian null benchmark: {NULL_GAP:.6f}")
