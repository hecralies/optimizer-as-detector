"""Plot compact boxplots for the 4/pi universality experiment.

Reads the 30 seed-level estimates in results/4pi_universality_raw.csv and
writes fig/fig_universality_a.{pdf,png} and
fig/fig_universality_b.{pdf,png}.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT_DIR, "fig")
RAW_PATH = os.path.join(ROOT_DIR, "results", "4pi_universality_raw.csv")
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


def draw_box_panel(
    ax,
    rows,
    group_order,
    group_colors,
    group_titles,
    tick_labels,
):
    data = []
    colors = []
    positions = []
    labels = []
    group_spans = []
    position = 1

    for group in group_order:
        group_rows = rows[rows["group"] == group]
        start = position
        for code in group_rows["setting_code"].drop_duplicates():
            values = group_rows.loc[group_rows["setting_code"] == code, "S_T"]
            data.append(values.to_numpy())
            colors.append(group_colors[group])
            positions.append(position)
            labels.append(tick_labels[code])
            position += 1
        group_spans.append((group, start, position - 1))
        position += 0.65

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

    ax.axhline(
        FOUR_PI,
        color="#D62728",
        linestyle="--",
        linewidth=1.7,
        zorder=0,
    )

    for (_, _, right), (_, next_left, _) in zip(group_spans[:-1], group_spans[1:]):
        ax.axvline((right + next_left) / 2, color="#B8B8B8", linewidth=0.7)

    for group, left, right in group_spans:
        ax.text(
            (left + right) / 2,
            1.015,
            group_titles[group],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=14,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"$\hat S_T$")
    ax.set_ylim(1.00, 1.50)
    ax.set_yticks(np.arange(1.0, 1.51, 0.1))
    ax.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)
    ax.tick_params(axis="x", length=2.5, pad=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path(ROOT_DIR) / "results")
    parser.add_argument("--fig-dir", type=Path, default=Path(FIG_DIR))
    args = parser.parse_args()
    raw = pd.read_csv(args.results_dir / Path(RAW_PATH).name)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    standard = raw[raw["panel"] == "standard"].copy()
    standard_groups = ["Vary dimension", "Vary noise", "Vary covariance"]
    standard_colors = {
        "Vary dimension": BLUE,
        "Vary noise": GREEN,
        "Vary covariance": ORANGE,
    }
    standard_titles = {
        "Vary dimension": r"Dimension $d$",
        "Vary noise": r"Noise $\sigma_\varepsilon$",
        "Vary covariance": r"Covariance $\Sigma$",
    }
    standard_ticks = {
        "d=1": "1",
        "d=2": "2",
        "d=5": "5",
        "d=10": "10",
        "d=20": "20",
        "sigma=0.5": "0.5",
        "sigma=1": "1",
        "sigma=2": "2",
        "sigma=5": "5",
        "Sigma=I": r"$I_3$",
        "Sigma=diag": "$\\operatorname{diag}$\n$(1,2,5)$",
        "Sigma=rho0.5": "0.5",
        "Sigma=rho0.8": "0.8",
    }

    fig, ax = plt.subplots(figsize=(6.2, 3.15))
    draw_box_panel(
        ax,
        standard,
        standard_groups,
        standard_colors,
        standard_titles,
        standard_ticks,
    )
    fig.tight_layout(pad=0.35)
    fig.savefig(args.fig_dir / "fig_universality_a.pdf")
    fig.savefig(args.fig_dir / "fig_universality_a.png", dpi=400)
    plt.close(fig)

    endogeneity = raw[raw["panel"] == "endogeneity"].copy()
    endogeneity_groups = [
        r"Endogeneity, $d=1$",
        r"Endogeneity, $d=2$",
        r"Endogeneity, $d=5$",
    ]
    endogeneity_colors = {
        r"Endogeneity, $d=1$": PURPLE,
        r"Endogeneity, $d=2$": PINK,
        r"Endogeneity, $d=5$": BROWN,
    }
    endogeneity_titles = {
        r"Endogeneity, $d=1$": r"$d=1$",
        r"Endogeneity, $d=2$": r"$d=2$",
        r"Endogeneity, $d=5$": r"$d=5$",
    }
    endogeneity_ticks = {
        code: rf"${rho:.1f}$"
        for code, rho in endogeneity[["setting_code", "rho"]]
        .drop_duplicates()
        .itertuples(index=False)
    }

    fig, ax = plt.subplots(figsize=(6.2, 3.15))
    draw_box_panel(
        ax,
        endogeneity,
        endogeneity_groups,
        endogeneity_colors,
        endogeneity_titles,
        endogeneity_ticks,
    )
    ax.set_xlabel(r"Endogeneity strength $\rho$", labelpad=2)
    fig.tight_layout(pad=0.35)
    fig.savefig(args.fig_dir / "fig_universality_b.pdf")
    fig.savefig(args.fig_dir / "fig_universality_b.png", dpi=400)
    plt.close(fig)

    print("Saved compact boxplots fig_universality_a/b pdf/png")


if __name__ == "__main__":
    main()
