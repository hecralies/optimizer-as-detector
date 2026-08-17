"""Review-only same-substrate contrast figure for new SP applications.

Each row keeps the same visual substrate in the left and right panels.
Left: domain quantity close to the source-paper visual question.
Right: the same observations colored by the stratified slope sign revealed by
the Simpson's-paradox workflow.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "new SPs" / "data"
RESULTS = ROOT / "new SPs" / "results"
FIG = ROOT / "fig"

TEXT = "#111827"
GRAY = "#6b7280"
GRID = "#edf0f3"
GOLD = "#b88a22"
RED = "#b84a48"
BLUE = "#2f6f9f"
NEUTRAL = "#d1d5db"
POINT_GRAY = "#b9aa9a"
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">"]
MAP_FACE = "#f7f5f0"
MAP_EDGE = "#d8d2c8"
COUNTRIES_GEOJSON = ROOT / "tmp" / "countries.geojson"

COUNTRY_COORDS = {
    "Australia": (134, -25),
    "Austria": (14.5, 47.5),
    "Belgium": (4.5, 50.5),
    "Bosnia and herz": (17.8, 44.2),
    "Brazil": (-52, -10),
    "Bulgaria": (25, 43),
    "Chile": (-71, -30),
    "China": (104, 35),
    "Croatia": (16.5, 45.1),
    "Cyprus": (33, 35),
    "Czech republic": (15.5, 49.8),
    "Denmark": (10, 56),
    "EU27 & UK": (7, 51),
    "Estonia": (25.5, 58.7),
    "Finland": (26, 64),
    "France": (2, 46),
    "Germany": (10.5, 51),
    "Greece": (22, 39),
    "Hungary": (19, 47),
    "India": (78, 22),
    "Ireland": (-8, 53),
    "Italy": (12.5, 42.5),
    "Japan": (138, 37),
    "Kosovo": (20.9, 42.6),
    "Latvia": (25, 57),
    "Lithuania": (24, 55),
    "Luxembourg": (6.1, 49.8),
    "Mexico": (-102, 23),
    "Moldova": (28.4, 47.2),
    "Montenegro": (19.3, 42.7),
    "Netherlands": (5.3, 52.2),
    "North macedonia": (21.7, 41.6),
    "Norway": (8, 61),
    "Poland": (19, 52),
    "Portugal": (-8, 39.5),
    "Romania": (25, 46),
    "Russia": (90, 60),
    "Serbia": (21, 44),
    "Slovakia": (19.5, 48.7),
    "Slovenia": (14.8, 46.1),
    "South Africa": (24, -29),
    "Spain": (-3.5, 40),
    "Sweden": (15, 62),
    "Switzerland": (8.2, 46.8),
    "UK": (-2, 54),
    "US": (-98, 39),
    "Ukraine": (31, 49),
}

COUNTRY_NAME_ALIASES = {
    "Bosnia and herz": "Bosnia and Herzegovina",
    "Czech republic": "Czechia",
    "EU27 & UK": None,
    "North macedonia": "North Macedonia",
    "Russia": "Russian Federation",
    "UK": "United Kingdom",
    "US": "United States of America",
}


def load_countries():
    with COUNTRIES_GEOJSON.open(encoding="utf-8") as fh:
        return json.load(fh)["features"]


COUNTRY_FEATURES = load_countries() if COUNTRIES_GEOJSON.exists() else []


def iter_rings(geometry):
    if geometry["type"] == "Polygon":
        for polygon in [geometry["coordinates"]]:
            yield polygon[0]
    elif geometry["type"] == "MultiPolygon":
        for polygon in geometry["coordinates"]:
            yield polygon[0]


def draw_country_base(ax, xlim=(-180, 180), ylim=(-60, 85), highlight=None, alpha=1.0):
    highlight = highlight or {}
    ax.set_facecolor("#fbfbfa")
    for feature in COUNTRY_FEATURES:
        name = feature["properties"].get("name")
        face = highlight.get(name, MAP_FACE)
        edge = MAP_EDGE
        lw = 0.25
        z = 0 if name not in highlight else 1
        for ring in iter_rings(feature["geometry"]):
            arr = np.asarray(ring, dtype=float)
            if arr.ndim != 2 or arr.shape[0] < 3:
                continue
            ax.fill(
                arr[:, 0],
                arr[:, 1],
                facecolor=face,
                edgecolor=edge,
                linewidth=lw,
                alpha=alpha,
                zorder=z,
            )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")


def finite_quantiles(values, q=(0.02, 0.98)):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return np.quantile(values, q) if len(values) else (0, 1)


def ols_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    x0 = x - x.mean()
    den = np.dot(x0, x0)
    return np.nan if den <= 0 else float(np.dot(x0, y - y.mean()) / den)


def style_axis(ax):
    ax.grid(False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color("#d1d5db")
    ax.tick_params(labelsize=7.2, colors="#374151")


def add_letter(ax, letter, title):
    ax.text(
        0.015,
        0.965,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.4,
        fontweight="bold",
        color=TEXT,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.2),
        zorder=10,
    )


def read_power_points():
    raw = pd.read_csv(DATA / "CarbonMonitor_Global_PM_corT.csv")
    raw["date"] = pd.to_datetime(raw["date"], format="%d/%m/%Y", errors="coerce")
    fossil = {"Coal", "Gas", "Oil"}
    renewable = {"Hydroelectricity", "Solar", "Wind", "Other Renewables"}
    raw = raw[raw["sector"].isin(fossil | renewable)].copy()
    raw["kind"] = np.where(raw["sector"].isin(fossil), "fossil", "renewable")
    daily = raw.groupby(["country", "date", "kind"], as_index=False)["value"].sum()
    wide = daily.pivot(index=["country", "date"], columns="kind", values="value").reset_index()
    wide = wide.dropna(subset=["fossil", "renewable"])
    slopes = pd.read_csv(RESULTS / "carbonmonitor_fossil_renewables_country_slopes.csv")
    slopes = slopes[slopes["group"].str.lower() != "pooled"][["group", "slope"]]
    slope_map = dict(zip(slopes["group"], slopes["slope"]))
    wide["country_slope"] = wide["country"].map(slope_map)
    wide["sign"] = np.sign(wide["country_slope"])
    wide["total"] = wide["fossil"] + wide["renewable"]
    pooled = pd.read_csv(RESULTS / "carbonmonitor_fossil_renewables_country_slopes.csv")
    pooled_slope = float(pooled.loc[pooled["group"].str.lower() == "pooled", "slope"].iloc[0])
    return wide, pooled_slope


def power_row(ax_left, ax_right):
    df, pooled = read_power_points()
    slopes = pd.read_csv(RESULTS / "carbonmonitor_fossil_renewables_country_slopes.csv")
    slopes = slopes[slopes["group"].str.lower() != "pooled"].copy()
    country_names = {}
    for group in slopes["group"]:
        mapped = COUNTRY_NAME_ALIASES.get(group, group)
        if mapped:
            country_names[mapped] = BLUE
    draw_country_base(ax_left, highlight=country_names)
    add_letter(ax_left, "A", "Power: pooled positive view")
    ax_left.set_xticks([-150, -100, -50, 0, 50, 100, 150])
    ax_left.set_yticks([-40, 0, 40])
    style_axis(ax_left)

    slopes["lon"] = slopes["group"].map(lambda g: COUNTRY_COORDS.get(g, (np.nan, np.nan))[0])
    slopes["lat"] = slopes["group"].map(lambda g: COUNTRY_COORDS.get(g, (np.nan, np.nan))[1])
    fill_map = {}
    for _, row in slopes.iterrows():
        mapped = COUNTRY_NAME_ALIASES.get(row["group"], row["group"])
        if mapped:
            fill_map[mapped] = RED if row["slope"] < 0 else BLUE if row["slope"] > 0 else NEUTRAL
    draw_country_base(ax_right, highlight=fill_map)
    ax_right.set_xticks([-150, -100, -50, 0, 50, 100, 150])
    ax_right.set_yticks([-40, 0, 40])
    add_letter(ax_right, "A'", "Countries reverse the pooled sign")
    style_axis(ax_right)


def plant_row(ax_left, ax_right):
    df = pd.read_csv(DATA / "pnas_biodiversity" / "Simkin_et_al_2016_data_from_PNAS_Div_and_N_dep.csv")
    us_xlim = (-126, -66)
    us_ylim = (26, 50)
    draw_country_base(ax_left, xlim=us_xlim, ylim=us_ylim)
    ax_left.scatter(
        df["longitude"],
        df["latitude"],
        c=BLUE,
        s=6,
        alpha=0.38,
        linewidth=0,
        zorder=3,
    )
    add_letter(ax_left, "B", "Plant diversity: pooled positive view")
    style_axis(ax_left)

    slopes = pd.read_csv(RESULTS / "pnas_biodiv_log_spp_richness_vs_EX_kghayr_slopes_by_proj_orig.csv")
    slope_map = dict(zip(slopes["group"], slopes["slope"]))
    draw_country_base(ax_right, xlim=us_xlim, ylim=us_ylim)
    for i, group in enumerate(sorted(df["proj_orig"].dropna().unique())):
        sub = df[df["proj_orig"] == group]
        slope = slope_map.get(group, np.nan)
        color = RED if slope < 0 else BLUE if slope > 0 else NEUTRAL
        ax_right.scatter(
            sub["longitude"],
            sub["latitude"],
            c=color,
            marker=MARKERS[i % len(MARKERS)],
            s=6,
            alpha=0.70,
            linewidth=0,
            zorder=3,
        )
    add_letter(ax_right, "B'", "Projects reverse the pooled sign")
    neg = int(np.sum(slopes["slope"] < 0))
    total = len(slopes)
    style_axis(ax_right)


def read_nematode_slopes():
    path = RESULTS / "beyond_pnas_shortlist_stage_checks.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["candidate"].startswith("Nature nematodes: bacterivores"):
                pooled = float(row["pooled_slope"])
                pat = re.compile(
                    r"(?P<group>[^:;]+):(?P<slope>[+-]?\d*\.?\d+(?:e[+-]?\d+)?)"
                    r"\[(?P<lo>[+-]?\d*\.?\d+(?:e[+-]?\d+)?),(?P<hi>[+-]?\d*\.?\d+(?:e[+-]?\d+)?)\]"
                    r"\s+n=(?P<n>\d+)",
                    re.IGNORECASE,
                )
                slope_map = {}
                for match in pat.finditer(row["group_details"]):
                    slope_map[int(match.group("group"))] = float(match.group("slope"))
                return pooled, slope_map
    raise RuntimeError("Could not find nematode slope row")


def nematode_row(ax_left, ax_right):
    df = pd.read_csv(DATA / "candidate_search_20260602" / "soil_nematodes" / "nematode_abundance_metadata.csv")
    draw_country_base(ax_left)
    ax_left.scatter(
        df["Pixel_Long"],
        df["Pixel_Lat"],
        c=BLUE,
        s=8,
        alpha=0.40,
        linewidth=0,
        zorder=3,
    )
    add_letter(ax_left, "A", "Soil nematodes: pooled positive view")
    style_axis(ax_left)

    pooled, slope_map = read_nematode_slopes()
    draw_country_base(ax_right)
    for i, group in enumerate(sorted(df["WWF_Biome"].dropna().unique())):
        sub = df[df["WWF_Biome"] == group]
        slope = slope_map.get(int(group), np.nan)
        color = RED if slope < 0 else BLUE if slope > 0 else NEUTRAL
        ax_right.scatter(
            sub["Pixel_Long"],
            sub["Pixel_Lat"],
            c=color,
            marker=MARKERS[i % len(MARKERS)],
            s=8,
            alpha=0.72,
            linewidth=0,
            zorder=3,
        )
    add_letter(ax_right, "A'", "Biomes reverse the pooled sign")
    neg = sum(1 for s in slope_map.values() if s < 0)
    total = len(slope_map)
    style_axis(ax_right)


def main():
    FIG.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 3.2), dpi=260)
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.05, top=0.98, wspace=0.12, hspace=0.18)

    nematode_row(axes[0, 0], axes[0, 1])
    plant_row(axes[1, 0], axes[1, 1])

    out_png = FIG / "new_sp_same_substrate_review.png"
    out_pdf = FIG / "new_sp_same_substrate_review.pdf"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
