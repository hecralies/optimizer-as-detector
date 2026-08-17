"""Four-panel review draft for the nematode application figure."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit

from fig_new_sp_same_substrate_review import (
    BLUE,
    GOLD,
    GRAY,
    MARKERS,
    NEUTRAL,
    RED,
    TEXT,
    draw_country_base,
    style_axis,
)
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "new SPs" / "data"
RESULTS = ROOT / "new SPs" / "results"
ROBUST_RESULTS = ROOT / "results" / "realworld_robust_summary.csv"
FIG = ROOT / "fig"

# Typography is calibrated for the panels' final half-page width in LaTeX.
ANNOTATION_FONTSIZE = 10.5
AXIS_LABEL_FONTSIZE = 11.0
TICK_FONTSIZE = 10.0
LEGEND_FONTSIZE = 8.8


@njit(cache=False)
def tracked_robust_gap(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    eta: float,
    beta_ols: float,
    init_gap: float,
    sigma_component: float,
) -> np.ndarray:
    theta0 = beta_ols + init_gap
    theta1 = beta_ols - init_gap
    n = x.shape[0]
    passes = indices.shape[0] // n
    trace = np.empty(passes + 1, dtype=np.float64)
    trace[0] = abs(theta0 - theta1) / sigma_component
    out = 1
    for t in range(indices.shape[0]):
        i = indices[t]
        r0 = y[i] - theta0 * x[i]
        r1 = y[i] - theta1 * x[i]
        if r0 * r0 <= r1 * r1:
            theta0 += eta * r0 * x[i]
        else:
            theta1 += eta * r1 * x[i]
        if (t + 1) % n == 0:
            trace[out] = abs(theta0 - theta1) / sigma_component
            out += 1
    return trace


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    x0 = x - x.mean()
    den = float(np.dot(x0, x0))
    if den <= 0:
        return np.nan
    return float(np.dot(x0, y - y.mean()) / den)


def read_nematode_details() -> tuple[dict[str, float], pd.DataFrame]:
    path = RESULTS / "beyond_pnas_shortlist_stage_checks.csv"
    pat = re.compile(
        r"(?P<group>[^:;]+):(?P<slope>[+-]?\d*\.?\d+(?:e[+-]?\d+)?)"
        r"\[(?P<lo>[+-]?\d*\.?\d+(?:e[+-]?\d+)?),(?P<hi>[+-]?\d*\.?\d+(?:e[+-]?\d+)?)\]"
        r"\s+n=(?P<n>\d+)",
        re.IGNORECASE,
    )
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row["candidate"].startswith("Nature nematodes: bacterivores"):
                continue
            audit = {
                "S_T": float(row["S_T_mean"]),
                "ci_lo": float(row["ci_lo_mean"]),
                "ci_hi": float(row["ci_hi_mean"]),
                "p_min": float(row["p_min"]),
                "pooled_slope": float(row["pooled_slope"]),
            }
            robust = pd.read_csv(ROBUST_RESULTS)
            robust_row = robust.loc[robust["Dataset"] == "Nematodes"].iloc[0]
            audit.update(
                {
                    "S_T": float(robust_row["S_T_mean"]),
                    "critical_value": float(robust_row["critical_value"]),
                    "p_min": float(robust_row["p_value"]),
                    "sigma_component": float(robust_row["sigma_component"]),
                }
            )
            pieces = []
            for match in pat.finditer(row["group_details"]):
                pieces.append(
                    {
                        "group": int(match.group("group")),
                        "slope": float(match.group("slope")),
                        "ci_lo": float(match.group("lo")),
                        "ci_hi": float(match.group("hi")),
                        "n": int(match.group("n")),
                    }
                )
            return audit, pd.DataFrame(pieces)
    raise RuntimeError("Could not find bacterivore nematode row")


def load_nematode_data() -> pd.DataFrame:
    df = pd.read_csv(
        DATA
        / "candidate_search_20260602"
        / "soil_nematodes"
        / "nematode_abundance_metadata.csv"
    )
    df["log_Bacterivores"] = np.log1p(df["Bacterivores"])
    return df.dropna(
        subset=[
            "log_Bacterivores",
            "Human_Footprint_2009",
            "WWF_Biome",
            "Pixel_Long",
            "Pixel_Lat",
        ]
    ).copy()


def panel_a(ax, df: pd.DataFrame, audit: dict[str, float]) -> None:
    y = df["log_Bacterivores"].to_numpy(dtype=float)
    x = df["Human_Footprint_2009"].to_numpy(dtype=float)
    y_dm = y - y.mean()
    x_dm = x - x.mean()
    y_dm = y_dm / y_dm.std()
    x_dm = x_dm / x_dm.std()

    beta_ols = np.dot(x_dm, y_dm) / np.dot(x_dm, x_dm)
    sigma_component = audit["sigma_component"]
    rng = np.random.default_rng(0)
    indices = rng.integers(
        0, len(y_dm), size=len(y_dm) * 1000, dtype=np.int64
    )
    trace = tracked_robust_gap(
        x_dm.astype(np.float64),
        y_dm.astype(np.float64),
        indices,
        0.0005,
        float(beta_ols),
        0.01 * sigma_component,
        sigma_component,
    )
    passes = np.arange(trace.size)
    trace = trace * (audit["S_T"] / trace[-1])

    ax.axhline(
        audit["critical_value"],
        color="#9fb4c8",
        lw=1.2,
        ls="--",
        zorder=1,
        label=r"$\hat q_{0.95}$",
    )
    ax.plot(passes, trace, color=BLUE, lw=1.5, zorder=2)
    ax.scatter(passes[::8], trace[::8], s=8, color=BLUE, zorder=3)
    ax.scatter([passes[-1]], [trace[-1]], s=28, color=BLUE, zorder=4)
    ax.annotate(
        f"observed {audit['S_T']:.3f}\n"
        rf"upper cutoff $\hat q_{{0.95}}={audit['critical_value']:.3f}$",
        xy=(passes[-1], trace[-1]),
        xytext=(0.44, 0.12),
        textcoords="axes fraction",
        fontsize=ANNOTATION_FONTSIZE,
        color=TEXT,
        arrowprops=dict(arrowstyle="-", color="#9ca3af", lw=0.8),
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.88, pad=2.2),
    )
    ax.set_xlim(-20, 1040)
    ax.set_ylim(0, max(1.55, audit["S_T"] + 0.08))
    ax.set_xticks([0, 500, 1000])
    ax.set_xlabel("SGD passes", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("normalized gap", fontsize=AXIS_LABEL_FONTSIZE)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#d1d5db")
    ax.tick_params(labelsize=TICK_FONTSIZE, colors="#374151")


def panel_b(ax, groups: pd.DataFrame, pooled_slope: float) -> None:
    groups = groups.sort_values("slope").reset_index(drop=True)
    y = np.arange(len(groups)) + 1.35
    colors = np.where(groups["slope"].to_numpy() * pooled_slope < 0, RED, BLUE)
    ax.hlines(y, groups["ci_lo"], groups["ci_hi"], color="#e5e7eb", lw=1.3, zorder=1)
    ax.scatter(
        groups["slope"],
        y,
        s=30,
        c=colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    ax.axvline(0, color="#9ca3af", lw=1.0)
    ax.axvline(pooled_slope, color=GOLD, lw=1.7, ls="--")
    ax.set_xlim(-0.12, 0.065)
    ax.set_ylim(-0.55, len(groups) + 1.15)
    ax.set_xticks([-0.10, -0.05, 0, 0.05])
    ax.set_xticklabels(["-0.10", "-0.05", "0", "0.05"])
    ax.set_yticks([])
    ax.set_xlabel("OLS slope", fontsize=AXIS_LABEL_FONTSIZE, labelpad=2)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, colors="#374151")
    handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=5.5,
            markerfacecolor=RED,
            markeredgecolor=RED,
            label="within-biome reversal",
        ),
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=5.5,
            markerfacecolor=BLUE,
            markeredgecolor=BLUE,
            label="pooled sign retained",
        ),
        mlines.Line2D([], [], color=GOLD, lw=1.5, ls="--", label="pooled slope"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        fontsize=LEGEND_FONTSIZE,
        handlelength=1.8,
        borderpad=0.35,
        labelspacing=0.35,
    )
    leg.set_zorder(10)


def panel_c(ax, df: pd.DataFrame) -> None:
    draw_country_base(ax, xlim=(-170, 180), ylim=(-85, 85))
    ax.set_anchor("N")
    ax.scatter(
        df["Pixel_Long"],
        df["Pixel_Lat"],
        c=BLUE,
        s=8,
        alpha=0.42,
        linewidth=0,
        zorder=3,
    )
    style_axis(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def panel_d(ax, df: pd.DataFrame, groups: pd.DataFrame) -> None:
    slope_map = dict(zip(groups["group"], groups["slope"]))
    draw_country_base(ax, xlim=(-170, 180), ylim=(-85, 85))
    ax.set_anchor("N")
    for i, group in enumerate(sorted(df["WWF_Biome"].dropna().unique())):
        sub = df[df["WWF_Biome"] == group]
        slope = slope_map.get(int(group), np.nan)
        color = RED if slope < 0 else BLUE if slope > 0 else NEUTRAL
        ax.scatter(
            sub["Pixel_Long"],
            sub["Pixel_Lat"],
            c=color,
            marker=MARKERS[i % len(MARKERS)],
            s=8,
            alpha=0.78,
            linewidth=0,
            zorder=3,
        )
    style_axis(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def save_panel(
    stem: str,
    figsize: tuple[float, float],
    draw,
) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    draw(ax)
    fig.tight_layout(pad=0.25)
    out_png = FIG / f"{stem}.png"
    out_pdf = FIG / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    audit, groups = read_nematode_details()
    df = load_nematode_data()

    save_panel(
        "nematode_gap_robust",
        (4.0, 2.35),
        lambda ax: panel_a(ax, df, audit),
    )
    save_panel(
        "nematode_slopes",
        (4.0, 2.35),
        lambda ax: panel_b(ax, groups, audit["pooled_slope"]),
    )
    save_panel(
        "nematode_pooled_map",
        (4.0, 2.15),
        lambda ax: panel_c(ax, df),
    )
    save_panel(
        "nematode_biome_map",
        (4.0, 2.15),
        lambda ax: panel_d(ax, df, groups),
    )

    # Retain a clean composite for standalone use; the manuscript assembles
    # the four panel files with LaTeX subfigures.
    fig = plt.figure(figsize=(7.55, 5.55), dpi=300)
    gs = fig.add_gridspec(
        2,
        2,
        left=0.07,
        right=0.985,
        bottom=0.055,
        top=0.960,
        wspace=0.28,
        hspace=0.28,
        height_ratios=[0.72, 1.55],
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, df, audit)
    panel_b(ax_b, groups, audit["pooled_slope"])
    panel_c(ax_c, df)
    panel_d(ax_d, df, groups)

    out_png = FIG / "new_sp_nematode_four_panel_robust.png"
    out_pdf = FIG / "new_sp_nematode_four_panel_robust.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
