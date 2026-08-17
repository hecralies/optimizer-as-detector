"""
Real-world K=2 confounder-screening rerun.

This script is intentionally separate from exp_realworld_numba.py.  It focuses
on the manuscript's Stage 2 claim: with K=2, does the piecewise SGD assignment
rank the stated confounder first across random initializations?

Outputs:
  results/realworld_K2_raw.csv
  results/realworld_K2_summary.csv
  results/realworld_K2_config.json
"""

import argparse
import json
import os
import sys
import tarfile
import time
import traceback
import urllib.request
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
DATA_DIR = os.path.join(ROOT, "data", "benchmark")
EXPERIMENT_DIR = DATA_DIR
RESULTS_DIR = os.path.join(ROOT, "results")
EXTERNAL_DATA_DIR = DATA_DIR

sys.path.insert(0, THIS_DIR)
from sgd_competitive_numba import HAS_NUMBA as HAS_CORE_NUMBA
from sgd_competitive_numba import _gen_perms, _sgd_loop_multi, sgd_competitive


try:
    from numba import njit
    HAS_NUMBA = True

    @njit(cache=True)
    def _sgd_k_loop(y, x_aug, theta, perms, n_passes, eta, k_count, d_aug):
        n = len(y)
        for p in range(n_passes):
            for j in range(n):
                idx = perms[p, j]
                yi = y[idx]
                best_k = 0
                best_loss = 1e30
                for k in range(k_count):
                    r = yi
                    for dd in range(d_aug):
                        r -= theta[k, dd] * x_aug[idx, dd]
                    loss = r * r
                    if loss < best_loss:
                        best_loss = loss
                        best_k = k
                r_win = yi
                for dd in range(d_aug):
                    r_win -= theta[best_k, dd] * x_aug[idx, dd]
                for dd in range(d_aug):
                    theta[best_k, dd] += eta * r_win * x_aug[idx, dd]
        return theta

except ImportError:
    HAS_NUMBA = False


def _as_2d_x(x, n):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] != n and x.shape[1] == n:
        x = x.T
    if x.shape[0] != n:
        raise ValueError(f"x has shape {x.shape}, expected first dimension n={n}")
    return x


def _standardize_x(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        s = x.std()
        return (x - x.mean()) / (s + 1e-12)
    s = x.std(axis=0)
    return (x - x.mean(axis=0)) / (s + 1e-12)


def _standardize_y(y):
    y = np.asarray(y, dtype=float)
    return (y - y.mean()) / (y.std() + 1e-12)


def _clean_xy(y, x):
    y = np.asarray(y, dtype=float)
    x = _as_2d_x(x, len(y))
    mask = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    return y[mask], x[mask], mask


def diagnostic_tests(y, x):
    """Breusch-Pagan and Ramsey RESET diagnostics for the OLS null fit."""
    y, x, _ = _clean_xy(y, x)
    n = len(y)
    x = _standardize_x(x)
    y = _standardize_y(y)
    X = np.column_stack([np.ones(n), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    y_hat = X @ beta
    resid = y - y_hat

    z = resid ** 2
    z = z / (z.mean() + 1e-30)
    gamma = np.linalg.lstsq(X, z, rcond=None)[0]
    z_hat = X @ gamma
    ss_tot = np.sum((z - z.mean()) ** 2)
    r2 = 0.0 if ss_tot <= 1e-30 else 1.0 - np.sum((z - z_hat) ** 2) / ss_tot
    bp_lm = max(float(n * r2), 0.0)
    bp_df = x.shape[1]
    bp_p = float(1.0 - stats.chi2.cdf(bp_lm, bp_df))

    ss_r = np.sum(resid ** 2)
    Z = np.column_stack([X, y_hat ** 2, y_hat ** 3])
    beta_aug = np.linalg.lstsq(Z, y, rcond=None)[0]
    ss_ur = np.sum((y - Z @ beta_aug) ** 2)
    q = 2
    denom_df = max(n - Z.shape[1], 1)
    reset_f = float(((ss_r - ss_ur) / q) / (ss_ur / denom_df + 1e-30))
    reset_p = float(1.0 - stats.f.cdf(max(reset_f, 0.0), q, denom_df))
    return {
        "bp_lm": bp_lm,
        "bp_df": int(bp_df),
        "bp_p": bp_p,
        "reset_f": reset_f,
        "reset_df1": q,
        "reset_df2": int(denom_df),
        "reset_p": reset_p,
    }


def sgd_k_piece_assignments(y, x, k_count, eta, n_passes, seed):
    y = np.asarray(y, dtype=float)
    x = _as_2d_x(x, len(y))
    x_aug = np.ascontiguousarray(np.column_stack([np.ones(len(y)), x]))
    d_aug = x_aug.shape[1]

    rng = np.random.default_rng(seed)
    beta_ols = np.linalg.lstsq(x_aug, y, rcond=None)[0]
    theta = np.tile(beta_ols, (k_count, 1))
    init_scale = max(float(np.std(y)), 1e-12)
    for k in range(k_count):
        theta[k] += rng.standard_normal(d_aug) * init_scale * 0.5

    if HAS_NUMBA:
        perms = np.array([rng.permutation(len(y)) for _ in range(n_passes)])
        theta = _sgd_k_loop(
            y.astype(np.float64),
            x_aug.astype(np.float64),
            theta.astype(np.float64),
            perms,
            n_passes,
            eta,
            k_count,
            d_aug,
        )
    else:
        for _ in range(n_passes):
            for idx in rng.permutation(len(y)):
                xi = x_aug[idx]
                residuals = y[idx] - theta @ xi
                winner = int(np.argmin(residuals * residuals))
                theta[winner] += eta * residuals[winner] * xi

    residuals = y[:, None] - x_aug @ theta.T
    assignments = np.argmin(residuals * residuals, axis=1)
    return assignments, theta


def association_score(assignments, z):
    z = np.asarray(z)
    if z.dtype.kind in ("U", "S", "O", "b"):
        table = pd.crosstab(assignments, z)
        n_total = float(table.values.sum())
        if n_total == 0:
            return 0.0
        expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n_total
        observed = table.values.astype(float)
        chi2 = np.divide(
            (observed - expected) ** 2,
            expected,
            out=np.zeros_like(observed, dtype=float),
            where=expected > 0,
        ).sum()
        denom = n_total * max(min(table.shape) - 1, 1)
        return float(np.sqrt(chi2 / denom))

    z = z.astype(float)
    groups = np.unique(assignments)
    grand = float(np.mean(z))
    ss_between = 0.0
    for g in groups:
        vals = z[assignments == g]
        if len(vals):
            ss_between += len(vals) * (float(np.mean(vals)) - grand) ** 2
    ss_total = float(np.sum((z - grand) ** 2))
    return float(ss_between / (ss_total + 1e-30))


def rank_confounders(assignments, confounders):
    scores = [(name, association_score(assignments, values))
              for name, values in confounders.items()]
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def sign_reversal(y, x, z):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    z = np.asarray(z)

    x_dm = x - x.mean()
    y_dm = y - y.mean()
    pooled = np.sum(x_dm * y_dm) / (np.sum(x_dm * x_dm) + 1e-30)
    pooled_sign = np.sign(pooled)
    if pooled_sign == 0:
        return False, int(0), int(0), float(pooled)

    if z.dtype.kind not in ("U", "S", "O", "b") and len(np.unique(z)) > 10:
        z = pd.qcut(z.astype(float), 4, labels=False, duplicates="drop")

    reversed_count = 0
    group_count = 0
    for level in np.unique(z):
        mask = z == level
        if int(mask.sum()) < 5:
            continue
        xg = x[mask] - x[mask].mean()
        yg = y[mask] - y[mask].mean()
        denom = np.sum(xg * xg)
        if denom < 1e-12:
            continue
        slope = np.sum(xg * yg) / denom
        group_count += 1
        if np.sign(slope) != pooled_sign and slope != 0:
            reversed_count += 1
    return reversed_count > 0 and reversed_count >= group_count // 2, reversed_count, group_count, float(pooled)


def baseline_xu(y, x, confounders):
    """Exhaustive one-confounder sign-reversal screen."""
    for _, z in confounders.items():
        sp, _, _, _ = sign_reversal(y, x, z)
        if sp:
            return True
    return False


def baseline_tree(y, x, confounders):
    """Classification/regression-tree style confounder split followed by reversal check."""
    try:
        from sklearn.tree import DecisionTreeRegressor
    except Exception:
        return None

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    n = len(y)
    names = list(confounders.keys())
    if not names:
        return False

    columns = [x]
    for name in names:
        col = np.asarray(confounders[name])
        if col.dtype.kind in ("U", "S", "O", "b"):
            codes, _ = pd.factorize(col, sort=True)
            columns.append(codes.astype(float))
        else:
            columns.append(col.astype(float))
    design = np.column_stack(columns)

    for depth in [2, 3, 4, 5]:
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(10, n // 100), random_state=depth)
        tree.fit(design, y)
        split_features = set(int(f) for f in tree.tree_.feature if f >= 1)
        for feature in split_features:
            name = names[feature - 1]
            sp, _, _, _ = sign_reversal(y, x, confounders[name])
            if sp:
                return True
    return False


def baseline_lr(y, x, k_count=2, n_restarts=5, max_iter=200, tol=1e-6):
    """Likelihood-ratio mixture-regression baseline for scalar focal x."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        x = x[:, 0]
    n = len(y)
    x0 = x - x.mean()
    y0 = y - y.mean()
    beta0 = float(np.sum(x0 * y0) / (np.sum(x0 ** 2) + 1e-30))
    alpha0 = float(y.mean() - beta0 * x.mean())
    resid0 = y - alpha0 - beta0 * x
    sig0 = float(np.sqrt(np.mean(resid0 ** 2)) + 1e-12)
    ll_null = float(np.sum(-0.5 * np.log(2 * np.pi * sig0 ** 2) - 0.5 * resid0 ** 2 / sig0 ** 2))

    best_ll = -np.inf
    for restart in range(n_restarts):
        rng = np.random.default_rng(9900 + restart)
        pis = np.ones(k_count) / k_count
        alphas = rng.normal(y.mean(), y.std() * 0.1 + 1e-12, k_count)
        betas = rng.normal(beta0, abs(beta0) * 0.1 + 0.1, k_count)
        sigmas = np.full(k_count, y.std() + 1e-12)
        for _ in range(max_iter):
            log_r = np.empty((n, k_count))
            for k in range(k_count):
                r = y - alphas[k] - betas[k] * x
                log_r[:, k] = (
                    np.log(pis[k] + 1e-300)
                    - 0.5 * np.log(2 * np.pi * sigmas[k] ** 2 + 1e-300)
                    - 0.5 * r ** 2 / (sigmas[k] ** 2 + 1e-300)
                )
            mx = log_r.max(axis=1, keepdims=True)
            resp = np.exp(log_r - mx)
            resp /= resp.sum(axis=1, keepdims=True)
            Nk = resp.sum(axis=0) + 1e-30

            new_pis = Nk / n
            new_alphas = np.empty(k_count)
            new_betas = np.empty(k_count)
            new_sigmas = np.empty(k_count)
            for k in range(k_count):
                wk = resp[:, k]
                wx = float(np.sum(wk * x) / Nk[k])
                wy = float(np.sum(wk * y) / Nk[k])
                wxx = float(np.sum(wk * (x - wx) ** 2) / Nk[k])
                wxy = float(np.sum(wk * (x - wx) * (y - wy)) / Nk[k])
                new_betas[k] = wxy / wxx if wxx > 1e-12 else 0.0
                new_alphas[k] = wy - new_betas[k] * wx
                r = y - new_alphas[k] - new_betas[k] * x
                new_sigmas[k] = float(np.sqrt(np.sum(wk * r ** 2) / Nk[k]) + 1e-12)
            delta = float(np.max(np.abs(new_betas - betas)) + np.max(np.abs(new_alphas - alphas)))
            pis, alphas, betas, sigmas = new_pis, new_alphas, new_betas, new_sigmas
            if delta < tol:
                break

        log_c = np.empty((n, k_count))
        for k in range(k_count):
            r = y - alphas[k] - betas[k] * x
            log_c[:, k] = (
                np.log(pis[k] + 1e-300)
                - 0.5 * np.log(2 * np.pi * sigmas[k] ** 2 + 1e-300)
                - 0.5 * r ** 2 / (sigmas[k] ** 2 + 1e-300)
            )
        mx = log_c.max(axis=1)
        ll = float(np.sum(mx + np.log(np.sum(np.exp(log_c - mx[:, None]), axis=1))))
        best_ll = max(best_ll, ll)

    lr_stat = max(2.0 * (best_ll - ll_null), 0.0)
    p_lr = float(1.0 - stats.chi2.cdf(lr_stat, df=3))
    return p_lr < 0.05, p_lr, lr_stat


def _read_first_existing(paths, reader):
    for path in paths:
        if os.path.exists(path):
            return reader(path), path
    raise FileNotFoundError("none of these files exists: " + "; ".join(paths))


def _cached_download(filename, url):
    os.makedirs(EXTERNAL_DATA_DIR, exist_ok=True)
    path = os.path.join(EXTERNAL_DATA_DIR, filename)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return path


def load_taylor():
    df, source = _read_first_existing(
        [
            os.path.join(DATA_DIR, "simpson_paradox_data.csv"),
            os.path.join(EXPERIMENT_DIR, "simpson paradox data.xls"),
        ],
        lambda p: pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p),
    )
    df = df[df["Ethnicity"].isin(["White not Hispanic", "Hispanic"])].copy()
    x = (df["Ethnicity"] == "Hispanic").astype(float).to_numpy()
    y = df["Expenditures"].astype(float).to_numpy()
    return {
        "name": "Taylor",
        "source": source,
        "x": x,
        "y": y,
        "confounders": {"Age Cohort": df["Age Cohort"].to_numpy()},
        "true_confounder": "Age Cohort",
        "eta_stage1": 0.005,
        "passes_stage1": 100,
        "passes_stage2": 100,
    }


def load_iris():
    path = os.path.join(DATA_DIR, "iris.data")
    if os.path.exists(path):
        columns = [
            "sepal_length", "sepal_width", "petal_length", "petal_width",
            "species",
        ]
        df = pd.read_csv(path, names=columns).dropna().copy()
        data = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].to_numpy(dtype=float)
        species = df["species"].str.replace("Iris-", "", regex=False).to_numpy()
        source = path
    else:
        from sklearn.datasets import load_iris as sklearn_load_iris

        iris = sklearn_load_iris()
        species = np.asarray(iris.target_names)[iris.target]
        data = iris.data
        source = "sklearn.datasets.load_iris"
    return {
        "name": "Iris",
        "source": source,
        "x": data[:, 0],
        "y": data[:, 1],
        "confounders": {
            "petal_length": data[:, 2],
            "petal_width": data[:, 3],
            "species": species,
        },
        "true_confounder": "species",
        "eta_stage1": 0.01,
        "passes_stage1": 200,
        "passes_stage2": 100,
    }


def load_penguins():
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
    path = _cached_download("penguins.csv", url)
    df = pd.read_csv(path).dropna().copy()
    return {
        "name": "Penguins",
        "source": path,
        "x": df["bill_depth_mm"].astype(float).to_numpy(),
        "y": df["bill_length_mm"].astype(float).to_numpy(),
        "confounders": {
            "species": df["species"].astype(str).to_numpy(),
            "island": df["island"].astype(str).to_numpy(),
        },
        "true_confounder": "species",
        "eta_stage1": 0.005,
        "passes_stage1": 200,
        "passes_stage2": 100,
    }


def load_atlanta():
    df, source = _read_first_existing(
        [
            os.path.join(DATA_DIR, "atlanta_ces.csv"),
            os.path.join(EXPERIMENT_DIR, "atlanta_ces.csv"),
        ],
        pd.read_csv,
    )
    return {
        "name": "Atlanta CES",
        "source": source,
        "x": (df["sex"] == "Female").astype(float).to_numpy(),
        "y": df["annual.salary"].astype(float).to_numpy(),
        "confounders": {
            "age": df["age"].astype(float).to_numpy(),
            "ethnicity": df["ethnic.origin"].to_numpy(),
            "organization": df["organization"].to_numpy(),
        },
        "true_confounder": "organization",
        "eta_stage1": 0.001,
        "passes_stage1": 100,
        "passes_stage2": 100,
    }


def load_adult():
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income",
    ]
    train = os.path.join(EXPERIMENT_DIR, "adult.data")
    test = os.path.join(EXPERIMENT_DIR, "adult.test")
    if os.path.exists(train) and os.path.exists(test):
        df1 = pd.read_csv(train, names=columns, skipinitialspace=True)
        df2 = pd.read_csv(test, names=columns, skipinitialspace=True, skiprows=1)
        df2["income"] = df2["income"].str.replace(".", "", regex=False)
        df = pd.concat([df1, df2], ignore_index=True)
        source = train + "; " + test
    elif os.path.exists(train):
        df = pd.read_csv(train, names=columns, skipinitialspace=True)
        source = train
    else:
        raise FileNotFoundError(f"Adult files not found in {EXPERIMENT_DIR}")
    return {
        "name": "Adult Census",
        "source": source,
        "x": df["marital-status"].str.contains("Married").astype(float).to_numpy(),
        "y": df["hours-per-week"].astype(float).to_numpy(),
        "confounders": {
            "income": df["income"].str.contains(">50K").astype(int).to_numpy(),
            "age": df["age"].astype(float).to_numpy(),
            "occupation": df["occupation"].astype(str).to_numpy(),
        },
        "true_confounder": "income",
        "eta_stage1": 0.005,
        "passes_stage1": 150,
        "passes_stage2": 100,
    }


def load_ca_housing():
    path = os.path.join(DATA_DIR, "california_housing.tgz")
    if os.path.exists(path):
        with tarfile.open(path, "r:gz") as archive:
            member = next(m for m in archive.getmembers() if m.name.endswith("cal_housing.data"))
            with archive.extractfile(member) as fh:
                raw = pd.read_csv(fh, header=None)
        raw.columns = [
            "Longitude", "Latitude", "HouseAge", "TotalRooms", "TotalBedrooms",
            "Population", "Households", "MedInc", "MedHouseVal",
        ]
        households = raw["Households"].to_numpy(dtype=float)
        matrix = np.column_stack([
            raw["MedInc"].to_numpy(dtype=float),
            raw["HouseAge"].to_numpy(dtype=float),
            raw["TotalRooms"].to_numpy(dtype=float) / households,
            raw["TotalBedrooms"].to_numpy(dtype=float) / households,
            raw["Population"].to_numpy(dtype=float),
            raw["Population"].to_numpy(dtype=float) / households,
            raw["Latitude"].to_numpy(dtype=float),
            raw["Longitude"].to_numpy(dtype=float),
        ])
        target = raw["MedHouseVal"].to_numpy(dtype=float) / 100000.0
        feature_names = [
            "MedInc", "HouseAge", "AveRooms", "AveBedrms",
            "Population", "AveOccup", "Latitude", "Longitude",
        ]
        source = path
    else:
        from sklearn.datasets import fetch_california_housing

        data = fetch_california_housing()
        source = "sklearn.datasets.fetch_california_housing"
        matrix = data.data
        target = data.target.astype(float)
        feature_names = list(data.feature_names)

    feature_idx = [1, 2, 6, 7]
    x_raw = matrix[:, feature_idx]
    conf = {feature_names[i]: matrix[:, i]
            for i in range(len(feature_names)) if i not in feature_idx}
    return {
        "name": "CA Housing",
        "source": source,
        "x": _standardize_x(x_raw),
        "y": target,
        "confounders": conf,
        "true_confounder": "MedInc",
        "sp_x": matrix[:, 2],
        "sp_x_label": "AveRooms",
        "eta_stage1": 0.002,
        "passes_stage1": 150,
        "passes_stage2": 100,
    }


def load_power():
    path = os.path.join(ROOT, "data", "new_sp", "CarbonMonitor_Global_PM_corT.csv")
    raw = pd.read_csv(path)
    wide = raw.pivot_table(index=["country", "date"], columns="sector", values="value", aggfunc="sum").reset_index()
    wide.columns.name = None
    for col in ["Coal", "Gas", "Oil", "Hydroelectricity", "Solar", "Wind", "Other"]:
        if col not in wide:
            wide[col] = 0.0
    wide["Fossil"] = wide[["Coal", "Gas", "Oil"]].sum(axis=1)
    wide["Renewables"] = wide[["Hydroelectricity", "Solar", "Wind", "Other"]].sum(axis=1)
    wide["date_dt"] = pd.to_datetime(wide["date"], errors="coerce")
    wide["year"] = wide["date_dt"].dt.year.fillna(-1).astype(int)
    wide["month"] = wide["date_dt"].dt.month.fillna(-1).astype(int)
    wide["season"] = ((wide["month"] % 12) // 3 + 1).map({1: "winter", 2: "spring", 3: "summer", 4: "fall"}).fillna("unknown")
    df = wide[["Fossil", "Renewables", "country", "year", "month", "season"]].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "name": "Power",
        "source": path,
        "x": df["Renewables"].astype(float).to_numpy(),
        "y": df["Fossil"].astype(float).to_numpy(),
        "confounders": {
            "country": df["country"].astype(str).to_numpy(),
            "year": df["year"].to_numpy(),
            "month": df["month"].to_numpy(),
            "season": df["season"].astype(str).to_numpy(),
        },
        "true_confounder": "country",
        "passes_stage1": 100,
        "passes_stage2": 100,
    }


def load_n_deposition():
    path = os.path.join(ROOT, "data", "new_sp", "pnas_biodiversity", "Simkin_et_al_2016_data_from_PNAS_Div_and_N_dep.csv")
    df = pd.read_csv(path)
    df["log_spp_richness"] = np.log(df["spp_richness"].clip(lower=1))
    for c in ["N_dep_kghayr", "precip_mm", "temp_C_ave", "pH", "CL_kghayr", "EX_kghayr"]:
        df[f"{c}_q4"] = pd.qcut(df[c], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    df["lat_band"] = pd.qcut(df["latitude"], q=5, labels=False, duplicates="drop")
    df["lon_band"] = pd.qcut(df["longitude"], q=5, labels=False, duplicates="drop")
    cols = ["EX_kghayr", "log_spp_richness", "proj_orig", "two_class_veg", "NVC_1_name", "NVC_2_name",
            "pH_q4", "precip_mm_q4", "temp_C_ave_q4", "lat_band", "lon_band"]
    base = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["EX_kghayr", "log_spp_richness", "proj_orig"])
    return {
        "name": "Nitrogen deposition",
        "source": path,
        "x": base["EX_kghayr"].astype(float).to_numpy(),
        "y": base["log_spp_richness"].astype(float).to_numpy(),
        "confounders": {
            "proj_orig": base["proj_orig"].astype(str).to_numpy(),
            "two_class_veg": base["two_class_veg"].astype(str).to_numpy(),
            "NVC_1_name": base["NVC_1_name"].astype(str).to_numpy(),
            "NVC_2_name": base["NVC_2_name"].astype(str).to_numpy(),
            "pH_q4": base["pH_q4"].astype(str).to_numpy(),
            "precip_mm_q4": base["precip_mm_q4"].astype(str).to_numpy(),
            "temp_C_ave_q4": base["temp_C_ave_q4"].astype(str).to_numpy(),
            "lat_band": base["lat_band"].to_numpy(),
            "lon_band": base["lon_band"].to_numpy(),
        },
        "true_confounder": "proj_orig",
        "passes_stage1": 100,
        "passes_stage2": 100,
    }


def load_nematodes():
    path = os.path.join(
        ROOT,
        "data",
        "new_sp",
        "soil_nematodes",
        "nematode_abundance_metadata.csv",
    )
    df = pd.read_csv(path)
    df["log_Bacterivores"] = np.log1p(pd.to_numeric(df["Bacterivores"], errors="coerce"))
    df["WWF_Biome_num"] = pd.to_numeric(df["WWF_Biome"], errors="coerce")
    df["lat_band"] = pd.qcut(pd.to_numeric(df["Abs_Lat"], errors="coerce"), q=5, labels=False, duplicates="drop")
    cols = [
        "Human_Footprint_2009",
        "log_Bacterivores",
        "WWF_Biome_num",
        "lat_band",
        "Annual_Mean_Temperature",
        "Annual_Precipitation",
        "pHinHOX_15cm",
        "CContent_15cm",
        "Sand_Content_15cm",
        "Clay_Content_15cm",
    ]
    base = df[cols].replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Human_Footprint_2009", "log_Bacterivores", "WWF_Biome_num"]
    )
    base["WWF_Biome_cat"] = base["WWF_Biome_num"].astype(int).astype(str)
    return {
        "name": "Nematodes",
        "source": path,
        "x": base["Human_Footprint_2009"].astype(float).to_numpy(),
        "y": base["log_Bacterivores"].astype(float).to_numpy(),
        "confounders": {
            "WWF_Biome_cat": base["WWF_Biome_cat"].astype(str).to_numpy(),
            "lat_band": base["lat_band"].astype(str).to_numpy(),
            "Annual_Mean_Temperature": base["Annual_Mean_Temperature"].astype(float).to_numpy(),
            "Annual_Precipitation": base["Annual_Precipitation"].astype(float).to_numpy(),
            "pHinHOX_15cm": base["pHinHOX_15cm"].astype(float).to_numpy(),
            "CContent_15cm": base["CContent_15cm"].astype(float).to_numpy(),
            "Sand_Content_15cm": base["Sand_Content_15cm"].astype(float).to_numpy(),
            "Clay_Content_15cm": base["Clay_Content_15cm"].astype(float).to_numpy(),
        },
        "true_confounder": "WWF_Biome_cat",
        "sp_x_label": "Human_Footprint_2009",
        "passes_stage1": 1000,
        "passes_stage2": 100,
    }


def load_auto_mpg():
    local_paths = [
        os.path.join(EXPERIMENT_DIR, "auto-mpg.data"),
        os.path.join(DATA_DIR, "auto-mpg.data"),
    ]
    columns = [
        "mpg", "cylinders", "displacement", "horsepower", "weight",
        "acceleration", "model_year", "origin", "car_name",
    ]
    for path in local_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, sep=r"\s+", names=columns, na_values="?")
            source = path
            break
    else:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
        path = _cached_download("auto-mpg.data", url)
        df = pd.read_csv(path, sep=r"\s+", names=columns, na_values="?")
        source = path

    df = df.dropna(subset=["horsepower"]).copy()
    x = np.column_stack([
        df["weight"].astype(float).to_numpy(),
        df["horsepower"].astype(float).to_numpy(),
    ])
    return {
        "name": "Auto MPG",
        "source": source,
        "x": _standardize_x(x),
        "y": df["mpg"].astype(float).to_numpy(),
        "confounders": {
            "cylinders": df["cylinders"].to_numpy(),
            "displacement": df["displacement"].astype(float).to_numpy(),
            "acceleration": df["acceleration"].astype(float).to_numpy(),
            "model_year": df["model_year"].astype(float).to_numpy(),
        },
        "true_confounder": "cylinders",
        "eta_stage1": 0.005,
        "passes_stage1": 300,
        "passes_stage2": 100,
    }


JASA_LOADERS = [
    load_taylor,
    load_iris,
    load_penguins,
    load_atlanta,
    load_adult,
    load_power,
    load_n_deposition,
    load_nematodes,
    load_ca_housing,
]

LOADERS = JASA_LOADERS
ARCHIVED_LOADERS = [
    *JASA_LOADERS,
    load_auto_mpg,
]


def _multi_detection_null(y, x, eta, n_passes, null_seeds):
    n, d = x.shape
    y_dm = y - y.mean()
    x_dm = x - x.mean(axis=0)
    sigma_x = x_dm.T @ x_dm / n
    beta_ols = np.linalg.solve(sigma_x + 1e-10 * np.eye(d), x_dm.T @ y_dm / n)
    resid = y_dm - x_dm @ beta_ols
    sigma = float(np.sqrt(np.mean(resid ** 2)))
    null_s = []
    for ns in range(null_seeds):
        rng = np.random.default_rng(9999 + ns)
        x_null = x_dm[rng.integers(0, n, size=n)].copy()
        x_null -= x_null.mean(axis=0)
        eps_null = resid[rng.integers(0, n, size=n)].copy()
        eps_null *= rng.choice(np.array([-1.0, 1.0]), size=n)
        eps_null -= eps_null.mean()
        sigma_null = float(np.sqrt(np.mean(eps_null ** 2)))
        sigma_x_null = x_null.T @ x_null / n
        y_null = x_null @ beta_ols + eps_null
        y_null -= y_null.mean()
        direction = np.ones(d) / np.sqrt(d)
        b1 = beta_ols + 0.01 * sigma_null * direction
        b2 = beta_ols - 0.01 * sigma_null * direction
        if HAS_CORE_NUMBA:
            perms = _gen_perms(rng, n, n_passes)
            b1, b2 = _sgd_loop_multi(y_null, x_null, b1, b2, perms, n_passes, eta)
        else:
            for _ in range(n_passes):
                for idx in rng.permutation(n):
                    xi = x_null[idx]
                    yi = y_null[idx]
                    r1 = yi - b1 @ xi
                    r2 = yi - b2 @ xi
                    if r1 * r1 <= r2 * r2:
                        b1 += eta * r1 * xi
                    else:
                        b2 += eta * r2 * xi
        diff = b1 - b2
        null_s.append(float(np.sqrt(diff @ sigma_x_null @ diff) / sigma_null))
    mu0 = float(np.mean(null_s))
    sigma_t = float(np.std(null_s)) if float(np.std(null_s)) > 1e-8 else 0.01
    return y_dm, x_dm, sigma_x, beta_ols, sigma, mu0, sigma_t, np.asarray(null_s)


def _multi_detection_actual(y_dm, x_dm, sigma_x, beta_ols, sigma, eta, n_passes, seed):
    n, d = x_dm.shape
    rng = np.random.default_rng(seed)
    eps_init = 0.01 * sigma
    direction = np.ones(d) / np.sqrt(d)
    b1 = beta_ols + eps_init * direction
    b2 = beta_ols - eps_init * direction
    if HAS_CORE_NUMBA:
        perms = _gen_perms(rng, n, n_passes)
        b1, b2 = _sgd_loop_multi(y_dm, x_dm, b1, b2, perms, n_passes, eta)
    else:
        for _ in range(n_passes):
            for idx in rng.permutation(n):
                xi = x_dm[idx]
                yi = y_dm[idx]
                r1 = yi - b1 @ xi
                r2 = yi - b2 @ xi
                if r1 * r1 <= r2 * r2:
                    b1 += eta * r1 * xi
                else:
                    b2 += eta * r2 * xi
    diff = b1 - b2
    return float(np.sqrt(diff @ sigma_x @ diff) / sigma)


def run_detection(dataset, seeds, eta_stage1, passes_stage1, null_seeds):
    values = []
    y_std = _standardize_y(dataset["y"])
    x_std = _standardize_x(_as_2d_x(dataset["x"], len(y_std)))
    if x_std.shape[1] == 1:
        x_std = x_std[:, 0]
    else:
        y_dm, x_dm, sigma_x, beta_ols, sigma, mu0, sigma_t, null_s = _multi_detection_null(
            y_std, x_std, eta_stage1, passes_stage1, null_seeds
        )
        for seed in seeds:
            st = _multi_detection_actual(
                y_dm, x_dm, sigma_x, beta_ols, sigma, eta_stage1, passes_stage1, int(seed)
            )
            p_value = float((1.0 + np.count_nonzero(null_s >= st)) / (null_s.size + 1.0))
            values.append({
                "seed": int(seed),
                "S_T": st,
                "mu0": mu0,
                "sigma_T": sigma_t,
                "ci_lo": float(np.quantile(null_s, 0.025)),
                "ci_hi": float(np.quantile(null_s, 0.975)),
                "p_value": p_value,
            })
        return values
    for seed in seeds:
        result = sgd_competitive(
            y_std,
            x_std,
            eta=eta_stage1,
            n_passes=passes_stage1,
            seed=int(seed),
            null_seeds=null_seeds,
        )
        st = float(result["S_T"])
        mu0 = float(result["mu0"])
        sigma_t = float(result["sigma_T"])
        p_value = float(1.0 - stats.norm.cdf((st - mu0) / (sigma_t + 1e-30)))
        values.append({
            "seed": int(seed),
            "S_T": st,
            "mu0": mu0,
            "sigma_T": sigma_t,
            "ci_lo": mu0 - 1.96 * sigma_t,
            "ci_hi": mu0 + 1.96 * sigma_t,
            "p_value": p_value,
        })
    return values


def run_k2(dataset, seeds, eta_stage2, passes_stage2):
    rows = []
    true_name = dataset["true_confounder"]
    y_std = _standardize_y(dataset["y"])
    x_std = _standardize_x(_as_2d_x(dataset["x"], len(y_std)))
    for seed in seeds:
        assignments, _ = sgd_k_piece_assignments(
            y_std,
            x_std,
            k_count=2,
            eta=eta_stage2,
            n_passes=passes_stage2,
            seed=int(seed),
        )
        ranking = rank_confounders(assignments, dataset["confounders"])
        top_name, top_score = ranking[0]
        true_score = dict(ranking).get(true_name, np.nan)
        cluster_counts = np.bincount(assignments, minlength=2)
        tie_count = sum(np.isclose(score, top_score, rtol=1e-10, atol=1e-12)
                        for _, score in ranking)
        weak_first = bool(np.isclose(true_score, top_score, rtol=1e-10, atol=1e-12))
        strict_first = bool(top_name == true_name and tie_count == 1
                            and np.count_nonzero(cluster_counts) == 2)
        rows.append({
            "seed": int(seed),
            "top_confounder": top_name,
            "top_score": float(top_score),
            "true_confounder_score": float(true_score),
            "true_ranked_first": strict_first,
            "true_tied_for_first": weak_first,
            "top_tie_count": int(tie_count),
            "cluster0_n": int(cluster_counts[0]),
            "cluster1_n": int(cluster_counts[1]),
            "degenerate_split": bool(np.count_nonzero(cluster_counts) < 2),
            "ranking": "; ".join(f"{name}:{score:.6g}" for name, score in ranking),
        })
    return rows


def summarize_detection(rows):
    st = np.asarray([r["S_T"] for r in rows], dtype=float)
    mu0 = np.asarray([r["mu0"] for r in rows], dtype=float)
    sigma_t = np.asarray([r["sigma_T"] for r in rows], dtype=float)
    p_values = np.asarray([r["p_value"] for r in rows], dtype=float)
    mean_st = float(st.mean())
    mean_mu0 = float(mu0.mean())
    mean_sigma = float(sigma_t.mean())
    return {
        "S_T_mean": mean_st,
        "S_T_sd": float(st.std(ddof=1)) if len(st) > 1 else 0.0,
        "mu0_mean": mean_mu0,
        "sigma_T_mean": mean_sigma,
        "ci_lo": mean_mu0 - 1.96 * mean_sigma,
        "ci_hi": mean_mu0 + 1.96 * mean_sigma,
        "p_value_mean": float(p_values.mean()),
        "p_value_from_mean": float(1.0 - stats.norm.cdf((mean_st - mean_mu0) / (mean_sigma + 1e-30))),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Rerun real-world K=2 confounder screening.")
    parser.add_argument("--seeds", type=int, default=10, help="number of seeds for Stage 1 and K=2 screening")
    parser.add_argument("--eta-stage1", type=float, default=0.0005)
    parser.add_argument("--passes-stage1", type=int, default=1000)
    parser.add_argument("--eta-stage2", type=float, default=0.002)
    parser.add_argument("--passes-stage2", type=int, default=100)
    parser.add_argument("--null-seeds", type=int, default=20)
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="optional dataset names to run, e.g. Taylor 'Atlanta CES'")
    parser.add_argument("--skip-detection", action="store_true")
    parser.add_argument("--out-dir", default=RESULTS_DIR)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main():
    global RESULTS_DIR
    args = parse_args()
    if args.quick:
        args.seeds = min(args.seeds, 2)
        args.passes_stage1 = min(args.passes_stage1, 5)
        args.passes_stage2 = min(args.passes_stage2, 5)
        args.null_seeds = min(args.null_seeds, 2)
    RESULTS_DIR = os.path.abspath(args.out_dir)
    start = time.time()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    selected = None
    if args.datasets:
        selected = {name.lower() for name in args.datasets}

    config = {
        "script": os.path.basename(__file__),
        "K": 2,
        "n_seeds": args.seeds,
        "seeds": list(range(args.seeds)),
        "eta_stage2": args.eta_stage2,
        "eta_stage1": args.eta_stage1,
        "passes_stage1": args.passes_stage1,
        "passes_stage2": args.passes_stage2,
        "null_seeds": args.null_seeds,
        "standardize_stage1_y": True,
        "standardize_stage1_x": True,
        "standardize_stage2_y": True,
        "standardize_stage2_x": True,
        "has_numba": HAS_NUMBA,
        "data_dir": DATA_DIR,
        "experiment_dir": EXPERIMENT_DIR,
        "external_data_dir": EXTERNAL_DATA_DIR,
        "skip_detection": bool(args.skip_detection),
        "quick": bool(args.quick),
    }

    raw_rows = []
    summary_rows = []

    print("=" * 88, flush=True)
    print("REAL-WORLD K=2 CONFOUNDER SCREENING RERUN", flush=True)
    print("=" * 88, flush=True)

    for loader in LOADERS:
        dataset_name = loader.__name__.replace("load_", "").replace("_", " ").title()
        try:
            dataset = loader()
        except Exception as exc:
            name = dataset_name
            print(f"\n{name}: load failed: {exc}", flush=True)
            summary_rows.append({
                "Dataset": name,
                "status": "load_failed",
                "error": repr(exc),
            })
            raw_rows.append({
                "Dataset": name,
                "stage": "load",
                "status": "load_failed",
                "error": traceback.format_exc(limit=2),
            })
            continue

        if selected and dataset["name"].lower() not in selected:
            continue

        y = np.asarray(dataset["y"], dtype=float)
        x_original = np.asarray(dataset["x"], dtype=float)
        x_stage2 = _as_2d_x(x_original, len(y))
        dataset["y"] = y
        dataset["x"] = x_original
        n, d = x_stage2.shape

        print(f"\n{dataset['name']} (n={n}, d={d}, source={dataset['source']})", flush=True)

        diag = diagnostic_tests(y, x_original)
        print(
            "  Diagnostics: BP p={bp_p:.4g}, RESET p={reset_p:.4g}".format(**diag),
            flush=True,
        )
        raw_rows.append({
            "Dataset": dataset["name"],
            "stage": "diagnostics",
            "status": "ok",
            **diag,
        })

        detection_summary = {}
        if not args.skip_detection:
            print("  Stage 1 detection...", flush=True)
            detection_rows = run_detection(
                dataset,
                range(args.seeds),
                args.eta_stage1,
                args.passes_stage1,
                args.null_seeds,
            )
            detection_summary = summarize_detection(detection_rows)
            for row in detection_rows:
                raw_rows.append({
                    "Dataset": dataset["name"],
                    "stage": "detection",
                    "status": "ok",
                    **row,
                })
            print(
                "    S_T={S_T_mean:.3f}, CI=[{ci_lo:.3f},{ci_hi:.3f}], p={p_value_from_mean:.4g}".format(
                    **detection_summary
                ),
                flush=True,
            )

        print("  Stage 2 K=2 screening...", flush=True)
        k2_rows = run_k2(dataset, range(args.seeds), args.eta_stage2, args.passes_stage2)
        correct = int(sum(row["true_ranked_first"] for row in k2_rows))
        weak_correct = int(sum(row["true_tied_for_first"] for row in k2_rows))
        nondegenerate = int(sum(not row["degenerate_split"] for row in k2_rows))
        tied = int(sum(row["top_tie_count"] > 1 for row in k2_rows))
        for row in k2_rows:
            raw_rows.append({
                "Dataset": dataset["name"],
                "stage": "K2_screening",
                "status": "ok",
                **row,
            })
        print(f"    {dataset['true_confounder']} ranked #1: {correct}/{args.seeds}", flush=True)

        sp_x = dataset.get("sp_x", x_stage2[:, 0])
        sp, n_reversed, n_groups, pooled_slope = sign_reversal(
            y, sp_x, dataset["confounders"][dataset["true_confounder"]]
        )

        print("  Baselines...", flush=True)
        if dataset["name"] == "Auto MPG":
            xu_result = None
            tree_result = None
        else:
            xu_result = baseline_xu(y, sp_x, dataset["confounders"])
            tree_result = baseline_tree(y, sp_x, dataset["confounders"])
        lr_result, lr_p, lr_stat = baseline_lr(y, sp_x)
        print(f"    Xu={xu_result}, Tree={tree_result}, LR={lr_result} (p={lr_p:.4g})", flush=True)
        raw_rows.append({
            "Dataset": dataset["name"],
            "stage": "baselines",
            "status": "ok",
            "Xu": xu_result,
            "Tree": tree_result,
            "LR": lr_result,
            "LR_p": lr_p,
            "LR_stat": lr_stat,
        })

        row = {
            "Dataset": dataset["name"],
            "status": "ok",
            "n": n,
            "d": d,
            "source": dataset["source"],
            "true_confounder": dataset["true_confounder"],
            "K": 2,
            "K2_correct": correct,
            "K2_weak_top": weak_correct,
            "K2_total": args.seeds,
            "K2_rate": correct / args.seeds,
            "K2_weak_top_rate": weak_correct / args.seeds,
            "K2_nondegenerate": nondegenerate,
            "K2_tied_top": tied,
            "SP": bool(sp),
            "SP_x": dataset.get("sp_x_label", "x1"),
            "SP_reversed_groups": n_reversed,
            "SP_groups": n_groups,
            "pooled_slope": pooled_slope,
            "Xu": xu_result,
            "Tree": tree_result,
            "LR": lr_result,
            "LR_p": lr_p,
            "LR_stat": lr_stat,
            **diag,
            **detection_summary,
        }
        summary_rows.append(row)
        print(f"  Stage 3 sign reversal: {sp} ({n_reversed}/{n_groups} groups)", flush=True)
        print(f"  Elapsed: {time.time() - start:.1f}s", flush=True)

    raw_df = pd.DataFrame(raw_rows)
    summary_df = pd.DataFrame(summary_rows)

    raw_path = os.path.join(RESULTS_DIR, "realworld_K2_raw.csv")
    summary_path = os.path.join(RESULTS_DIR, "realworld_K2_summary.csv")
    config_path = os.path.join(RESULTS_DIR, "realworld_K2_config.json")
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(config_path, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    print("\n" + "=" * 88, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 88, flush=True)
    print(summary_df.to_string(index=False), flush=True)
    print(f"\nSaved {raw_path}", flush=True)
    print(f"Saved {summary_path}", flush=True)
    print(f"Saved {config_path}", flush=True)


if __name__ == "__main__":
    main()
