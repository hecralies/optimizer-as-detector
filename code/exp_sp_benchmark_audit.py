"""Audit SP-discovery benchmarks on the eight manuscript applications.

The script keeps three questions separate:

1. Does Xu-style enumeration output the nominated focal/reference triplet?
2. Does a CART implementation of the Shmueli--Yahav X-terminal ordering
   expose a reversed path containing the reference candidate above X?
3. What do the older, less specific heuristics report?

All methods receive the focal relationship and the complete candidate list.
The reference candidate is used only after each scan to score its output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# The loader module also contains optional numba-accelerated SGD routines.  This
# audit uses only its data loaders, so avoid importing/initializing numba.
_numba_stub = types.ModuleType("numba")


def _identity_njit(*args, **kwargs):
    if args and callable(args[0]):
        return args[0]
    return lambda function: function


_numba_stub.njit = _identity_njit
sys.modules.setdefault("numba", _numba_stub)

from exp_realworld_K2_rigorous import JASA_LOADERS


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "sp_benchmark_audit"


MIN_GROUP_N = {
    "Power": 200,
    "Nitrogen deposition": 100,
    "Nematodes": 30,
}


def focal_slope(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 3:
        return np.nan
    xd = x - x.mean()
    yd = y - y.mean()
    denom = float(np.dot(xd, xd))
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(xd, yd) / denom)


def make_strata(z, quartile_continuous=True):
    z = np.asarray(z)
    numeric = z.dtype.kind not in ("U", "S", "O", "b")
    if quartile_continuous and numeric and len(pd.unique(z[pd.notna(z)])) > 10:
        return np.asarray(pd.qcut(z.astype(float), 4, labels=False, duplicates="drop"))
    return z


def reversal_summary(y, x, z, min_group_n=5, quartile_continuous=True):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    z = make_strata(z, quartile_continuous=quartile_continuous)
    pooled = focal_slope(y, x)
    pooled_sign = np.sign(pooled)
    groups = []
    if not np.isfinite(pooled) or pooled_sign == 0:
        return {
            "pooled_slope": pooled,
            "eligible_groups": 0,
            "reversed_groups": 0,
            "strict_majority": False,
            "group_slopes": "",
        }
    for level in pd.unique(z[pd.notna(z)]):
        mask = np.asarray(z == level)
        if int(mask.sum()) < min_group_n:
            continue
        slope = focal_slope(y[mask], x[mask])
        if not np.isfinite(slope) or slope == 0:
            continue
        groups.append((str(level), int(mask.sum()), float(slope)))
    reversed_groups = sum(np.sign(slope) == -pooled_sign for _, _, slope in groups)
    eligible_groups = len(groups)
    return {
        "pooled_slope": pooled,
        "eligible_groups": eligible_groups,
        "reversed_groups": int(reversed_groups),
        "strict_majority": bool(eligible_groups >= 2 and 2 * reversed_groups > eligible_groups),
        "group_slopes": "; ".join(
            f"{level}:{slope:+.6g} n={n}" for level, n, slope in groups
        ),
    }


def xu_scan(dataset, min_group_n):
    y = np.asarray(dataset["y"], dtype=float)
    x = np.asarray(dataset.get("sp_x", dataset["x"]), dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    reference = dataset["true_confounder"]
    rows = []
    for candidate, z in dataset["confounders"].items():
        result = reversal_summary(y, x, z, min_group_n=min_group_n)
        rows.append(
            {
                "Dataset": dataset["name"],
                "method": "Xu-style enumeration",
                "candidate": candidate,
                "is_reference": candidate == reference,
                **result,
                "surfaced": result["strict_majority"],
            }
        )
    reference_row = next(row for row in rows if row["is_reference"])
    return reference_row, rows


def encode_tree_design(x, confounders):
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    blocks = [x[:, None]]
    owners = ["__focal_x__"]
    feature_labels = ["X"]
    for name, values in confounders.items():
        values = np.asarray(values)
        if values.dtype.kind in ("U", "S", "O", "b"):
            series = pd.Series(values, dtype="object").fillna("<missing>").astype(str)
            dummies = pd.get_dummies(series, prefix=name, dtype=float)
            blocks.append(dummies.to_numpy(dtype=float))
            owners.extend([name] * dummies.shape[1])
            feature_labels.extend(list(dummies.columns))
        else:
            numeric = pd.to_numeric(pd.Series(values), errors="coerce")
            fill = float(numeric.median()) if numeric.notna().any() else 0.0
            blocks.append(numeric.fillna(fill).to_numpy(dtype=float)[:, None])
            owners.append(name)
            feature_labels.append(name)
    return np.column_stack(blocks), owners, feature_labels


def x_terminal_scan(dataset, depths=(2, 3, 4, 5)):
    y = np.asarray(dataset["y"], dtype=float)
    x = np.asarray(dataset.get("sp_x", dataset["x"]), dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    design, owners, feature_labels = encode_tree_design(x, dataset["confounders"])
    ok = np.isfinite(y) & np.all(np.isfinite(design), axis=1)
    y = y[ok]
    x = x[ok]
    design = design[ok]
    reference = dataset["true_confounder"]
    pooled = focal_slope(y, x)
    pooled_sign = np.sign(pooled)
    min_leaf = max(10, len(y) // 100)
    paths = []

    for depth_limit in depths:
        model = DecisionTreeRegressor(
            max_depth=depth_limit,
            min_samples_leaf=min_leaf,
            random_state=depth_limit,
        )
        model.fit(design, y)
        tree = model.tree_

        def visit(node, mask, ancestors, depth):
            feature = int(tree.feature[node])
            if feature < 0:
                return
            owner = owners[feature]
            if owner == "__focal_x__":
                slope = focal_slope(y[mask], x[mask])
                reversed_path = bool(
                    np.isfinite(slope)
                    and slope != 0
                    and pooled_sign != 0
                    and np.sign(slope) == -pooled_sign
                )
                paths.append(
                    {
                        "Dataset": dataset["name"],
                        "method": "X-terminal CART",
                        "tree_depth": depth_limit,
                        "x_node_depth": depth,
                        "node_n": int(mask.sum()),
                        "ancestors": "; ".join(ancestors),
                        "reference_above_x": reference in ancestors,
                        "node_slope": slope,
                        "pooled_slope": pooled,
                        "reversed_path": reversed_path,
                        "surfaced_reference": bool(reference in ancestors and reversed_path),
                    }
                )
                return

            values = design[:, feature]
            threshold = float(tree.threshold[node])
            left = mask & (values <= threshold)
            right = mask & (values > threshold)
            label = f"{owner}[{feature_labels[feature]}]"
            visit(int(tree.children_left[node]), left, ancestors + [owner], depth + 1)
            visit(int(tree.children_right[node]), right, ancestors + [owner], depth + 1)

        visit(0, np.ones(len(y), dtype=bool), [], 0)

    surfaced = any(path["surfaced_reference"] for path in paths)
    reference_above = any(path["reference_above_x"] for path in paths)
    return {
        "Dataset": dataset["name"],
        "method": "X-terminal CART",
        "reference": reference,
        "pooled_slope": pooled,
        "min_samples_leaf": min_leaf,
        "x_terminal_paths": len(paths),
        "reference_above_x": reference_above,
        "surfaced": surfaced,
    }, paths


def legacy_tree_scan(dataset, min_group_n):
    y = np.asarray(dataset["y"], dtype=float)
    x = np.asarray(dataset.get("sp_x", dataset["x"]), dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    design, owners, _ = encode_tree_design(x, dataset["confounders"])
    n = len(y)
    selected = set()
    for depth in (2, 3, 4, 5):
        model = DecisionTreeRegressor(
            max_depth=depth,
            min_samples_leaf=max(10, n // 100),
            random_state=depth,
        )
        model.fit(design, y)
        selected.update(owners[int(f)] for f in model.tree_.feature if f >= 1)
    scan = {
        name: reversal_summary(y, x, z, min_group_n=min_group_n)
        for name, z in dataset["confounders"].items()
    }
    surfaced_candidates = sorted(
        name for name in selected if name in scan and scan[name]["strict_majority"]
    )
    reference = dataset["true_confounder"]
    return {
        "Dataset": dataset["name"],
        "method": "legacy tree heuristic",
        "reference": reference,
        "selected_candidates": "; ".join(sorted(selected - {"__focal_x__"})),
        "surfaced_candidates": "; ".join(surfaced_candidates),
        "legacy_any": bool(surfaced_candidates),
        "surfaced": bool(reference in surfaced_candidates),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    xu_rows = []
    tree_paths = []
    legacy_rows = []

    for loader in JASA_LOADERS:
        dataset = loader()
        min_group_n = MIN_GROUP_N.get(dataset["name"], 5)
        xu_reference, xu_detail = xu_scan(dataset, min_group_n)
        tree_summary, paths = x_terminal_scan(dataset)
        legacy = legacy_tree_scan(dataset, min_group_n)
        xu_rows.extend(xu_detail)
        tree_paths.extend(paths)
        legacy_rows.append(legacy)
        summary_rows.append(
            {
                "Dataset": dataset["name"],
                "reference": dataset["true_confounder"],
                "min_group_n": min_group_n,
                "reference_reversed_groups": xu_reference["reversed_groups"],
                "reference_eligible_groups": xu_reference["eligible_groups"],
                "Xu_style": bool(xu_reference["surfaced"]),
                "X_terminal_CART": bool(tree_summary["surfaced"]),
                "reference_above_X": bool(tree_summary["reference_above_x"]),
                "legacy_tree_any": bool(legacy["legacy_any"]),
                "legacy_tree_reference": bool(legacy["surfaced"]),
            }
        )
        print(
            f"{dataset['name']}: Xu={xu_reference['surfaced']} "
            f"({xu_reference['reversed_groups']}/{xu_reference['eligible_groups']}), "
            f"X-terminal={tree_summary['surfaced']}, "
            f"legacy-any={legacy['legacy_any']}, legacy-reference={legacy['surfaced']}",
            flush=True,
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "sp_benchmark_summary.csv", index=False)
    pd.DataFrame(xu_rows).to_csv(out_dir / "xu_style_candidate_scan.csv", index=False)
    pd.DataFrame(tree_paths).to_csv(out_dir / "x_terminal_tree_paths.csv", index=False)
    pd.DataFrame(legacy_rows).to_csv(out_dir / "legacy_tree_audit.csv", index=False)
    config = {
        "script": os.path.basename(__file__),
        "datasets": [row["Dataset"] for row in summary_rows],
        "xu_rule": "reference triplet is surfaced when a strict majority of eligible strata reverse the pooled slope; continuous candidates are quartile-discretized",
        "tree_rule": "CART depths 2--5; stop each path at its first X split; reference is surfaced when it occurs above X on a path whose local slope reverses the pooled slope",
        "reference_scoring": "reference identity is used only after each method scans all supplied candidates",
        "minimum_group_sizes": MIN_GROUP_N,
    }
    with open(out_dir / "sp_benchmark_config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    print("\n" + summary.to_string(index=False), flush=True)
    print("\nCounts", flush=True)
    print(summary[["Xu_style", "X_terminal_CART", "legacy_tree_any", "legacy_tree_reference"]].sum(), flush=True)
    print(f"\nSaved audit to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
