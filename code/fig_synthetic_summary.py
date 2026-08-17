"""Create the synthetic panels used in the manuscript and supplement.

The main manuscript uses separate null-calibration and group-count panels in
a LaTeX 2-by-2 composite.  The signal-strength panel is shown in the
supplement.  The legacy vertically stacked summary is retained as a convenient
diagnostic output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT_DIR / "fig"
NULL_RAW = ROOT_DIR / "results" / "4pi_universality_raw.csv"
ALT_RAW = ROOT_DIR / "results" / "synthetic_H1_robust_raw.csv"
FOUR_PI = 4.0 / np.pi

BLUE = "#2176AE"
GREEN = "#57B894"
ORANGE = "#F0803C"
PURPLE = "#8B5CF6"

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.labelsize": 14,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def add_boxplot(ax, data, positions, colors) -> None:
    box = ax.boxplot(
        data,
        positions=positions,
        widths=0.72,
        patch_artist=True,
        showfliers=True,
        whis=1.5,
        flierprops={
            "marker": "o",
            "markersize": 2.4,
            "markerfacecolor": "#555555",
            "markeredgecolor": "#555555",
            "alpha": 0.75,
        },
        medianprops={"color": "#222222", "linewidth": 1.4},
        boxprops={"linewidth": 1.0, "edgecolor": "#444444"},
        whiskerprops={"linewidth": 1.0, "color": "#555555"},
        capprops={"linewidth": 1.0, "color": "#555555"},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)


def add_reference(ax) -> None:
    ax.axhline(
        FOUR_PI,
        color="#D62728",
        linestyle="--",
        linewidth=1.6,
        zorder=0,
    )
    ax.set_ylabel(r"$\hat S_T$")


def add_group_structure(
    ax, spans, titles, title_size=13, placement: str = "top"
) -> None:
    if placement == "bottom":
        label_y, vertical_alignment = -0.18, "top"
    else:
        label_y, vertical_alignment = 1.015, "bottom"

    for (_, _, right), (_, next_left, _) in zip(spans[:-1], spans[1:]):
        ax.axvline((right + next_left) / 2, color="#B8B8B8", linewidth=0.7)
    for key, left, right in spans:
        ax.text(
            (left + right) / 2,
            label_y,
            titles[key],
            transform=ax.get_xaxis_transform(),
            ha="center",
            va=vertical_alignment,
            fontsize=title_size,
        )


def add_panel_label(ax, label) -> None:
    ax.text(
        -0.075,
        1.015,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
    )


def draw_null_panel(ax, raw, panel_label: str | None = None) -> None:
    rows = raw[raw["panel"] == "standard"].copy()
    groups = ["Vary dimension", "Vary noise", "Vary covariance"]
    group_colors = {
        "Vary dimension": BLUE,
        "Vary noise": GREEN,
        "Vary covariance": ORANGE,
    }
    group_titles = {
        "Vary dimension": r"Dimension $d$",
        "Vary noise": r"Noise $\sigma_\varepsilon$",
        "Vary covariance": r"Covariance $\Sigma$",
    }
    tick_labels = {
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
        "Sigma=diag": "$D$",
        "Sigma=rho0.5": "0.5",
        "Sigma=rho0.8": "0.8",
    }

    data, colors, positions, labels, spans = [], [], [], [], []
    position = 1.0
    for group in groups:
        group_rows = rows[rows["group"] == group]
        start = position
        step = 1.25 if group == "Vary covariance" else 1.0
        for code in group_rows["setting_code"].drop_duplicates():
            values = group_rows.loc[group_rows["setting_code"] == code, "S_T"]
            data.append(values.to_numpy())
            colors.append(group_colors[group])
            positions.append(position)
            labels.append(tick_labels[code])
            position += step
        spans.append((group, start, positions[-1]))
        position += 0.75

    add_boxplot(ax, data, positions, colors)
    add_reference(ax)
    add_group_structure(
        ax, spans, group_titles, title_size=10.5, placement="bottom"
    )
    if panel_label is not None:
        add_panel_label(ax, panel_label)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(1.00, 1.50)
    ax.set_yticks(np.arange(1.0, 1.51, 0.1))
    ax.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)
    ax.tick_params(axis="x", length=2.5, pad=2)


def draw_group_count_panel(ax, raw, panel_label: str | None = None) -> None:
    rows = raw[raw["panel"] == "K"].copy()
    k_values = [2, 3, 4, 5, 6]
    dimensions = [1, 2, 5, 10]
    dimension_colors = [BLUE, GREEN, ORANGE, PURPLE]

    data, colors, positions, labels, spans = [], [], [], [], []
    position = 1.0
    for k_true in k_values:
        start = position
        for d, color in zip(dimensions, dimension_colors):
            values = rows.loc[
                (rows["k_true"] == k_true) & (rows["d"] == d), "S_T"
            ]
            data.append(values.to_numpy())
            colors.append(color)
            positions.append(position)
            labels.append(str(d))
            position += 1.0
        spans.append((k_true, start, position - 1.0))
        position += 0.65

    add_boxplot(ax, data, positions, colors)
    add_reference(ax)
    add_group_structure(
        ax,
        spans,
        {k_true: rf"$K^*={k_true}$" for k_true in k_values},
        title_size=10.5,
        placement="bottom",
    )
    if panel_label is not None:
        add_panel_label(ax, panel_label)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylim(1.0, 6.9)
    ax.set_yticks(np.arange(1, 7, 1))
    ax.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)


def draw_signal_panel(ax, raw, panel_label: str | None = None) -> None:
    rows = raw[raw["panel"] == "dimension"].copy()
    dimensions = [1, 2, 5, 10]
    deltas = [0.3, 0.5, 1.0]
    delta_colors = [BLUE, GREEN, ORANGE]

    data, colors, positions, labels, spans = [], [], [], [], []
    position = 1.0
    for d in dimensions:
        start = position
        for delta, color in zip(deltas, delta_colors):
            values = rows.loc[
                (rows["d"] == d) & np.isclose(rows["delta"], delta), "S_T"
            ]
            data.append(values.to_numpy())
            colors.append(color)
            positions.append(position)
            labels.append(f"{delta:.1f}")
            position += 1.0
        spans.append((d, start, position - 1.0))
        position += 0.65

    add_boxplot(ax, data, positions, colors)
    add_reference(ax)
    add_group_structure(ax, spans, {d: rf"$d={d}$" for d in dimensions})
    if panel_label is not None:
        add_panel_label(ax, panel_label)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_xlabel(r"Signal strength $\Delta$", labelpad=2)
    ax.set_ylim(1.1, 2.5)
    ax.set_yticks(np.arange(1.2, 2.51, 0.2))
    ax.set_xlim(positions[0] - 0.7, positions[-1] + 0.7)


def save_panel(
    draw, raw, stem: str, figsize: tuple[float, float], fig_dir: Path
) -> None:
    """Save one title-free panel for assembly with LaTeX subfigures."""
    fig, ax = plt.subplots(figsize=figsize)
    draw(ax, raw)
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.18, top=0.88)
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        fig_dir / f"{stem}.png",
        dpi=400,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=ROOT_DIR / "results")
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    null_raw = pd.read_csv(args.results_dir / NULL_RAW.name)
    alt_raw = pd.read_csv(args.results_dir / ALT_RAW.name)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    save_panel(
        draw_null_panel, null_raw, "fig_synthetic_null", (4.2, 3.25), args.fig_dir
    )
    save_panel(
        draw_group_count_panel,
        alt_raw,
        "fig_synthetic_groups",
        (4.2, 3.25),
        args.fig_dir,
    )
    save_panel(
        draw_signal_panel,
        alt_raw,
        "fig_synthetic_signal",
        (6.2, 3.0),
        args.fig_dir,
    )

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.6))
    draw_null_panel(axes[0], null_raw, "(a)")
    draw_group_count_panel(axes[1], alt_raw, "(b)")
    draw_signal_panel(axes[2], alt_raw, "(c)")
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.06, top=0.975, hspace=0.38)

    fig.savefig(
        args.fig_dir / "fig_synthetic_summary.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        args.fig_dir / "fig_synthetic_summary.png",
        dpi=350,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
