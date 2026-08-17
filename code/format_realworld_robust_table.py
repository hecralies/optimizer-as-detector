"""Format robust real-world Stage 1 results as LaTeX table cells."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


LABELS = {
    "Taylor": r"Taylor$^\dagger$",
    "Iris": r"Iris$^\dagger$",
    "Penguins": r"Penguins$^\dagger$",
    "Atlanta CES": r"Atlanta$^\dagger$",
    "Adult Census": r"Adult$^\dagger$",
    "Power": r"Power$^\star$",
    "CA Housing": r"CA Hous.$^\star$",
    "Nematodes": r"Nematodes$^\star$",
    "N-depos.": r"\shortstack[l]{Nitrogen\\deposition}$^\star$",
    "Nitrogen deposition": r"\shortstack[l]{Nitrogen\\deposition}$^\star$",
    "Auto MPG": r"Auto MPG$^\circ$",
}


def format_p(value: float, multivariate: bool) -> str:
    if multivariate:
        return f"${value:.3f}$"
    if value < 1e-15:
        return r"${<}10^{-15}$"
    if value < 1e-12:
        return r"${<}10^{-12}$"
    if value < 1e-10:
        return r"${<}10^{-10}$"
    if value < 1e-4:
        exponent = int(math.floor(math.log10(value)))
        coefficient = value / (10.0**exponent)
        return rf"${coefficient:.1f}\times10^{{{exponent}}}$"
    return f"${value:.4f}$"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "summary",
        type=Path,
        nargs="?",
        default=Path("results/realworld_stage1_summary.csv"),
    )
    args = parser.parse_args()
    data = pd.read_csv(args.summary)
    for name in LABELS:
        row = data.loc[data["Dataset"] == name].iloc[0]
        print(
            f"{LABELS[name]:<24} & "
            f"${row['S_T_mean']:.3f}$ & "
            f"${row['critical_value']:.3f}$ & "
            f"{format_p(float(row['p_value']), int(row['d']) > 1)} \\\\"
        )


if __name__ == "__main__":
    main()
