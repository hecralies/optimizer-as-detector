"""
sgd_competitive.py
===================
Core implementation of SGD with competitive loss (Algorithm 1 in the paper).

Supports:
  - Scalar (d=1): closed-form null distribution (distribution-free)
  - Multivariate (d>=2): empirical-covariate fitted-null bootstrap
"""

import numpy as np
from scipy import stats

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

# ── Numba-accelerated inner loops ──────────────────────────────────────
if HAS_NUMBA:
    @njit(cache=True)
    def _sgd_loop_scalar(y_dm, x_dm, betas, perms, n_passes, eta):
        """Inner SGD loop for d=1, called with pre-generated permutations."""
        n = len(y_dm)
        for p in range(n_passes):
            for j in range(n):
                idx = perms[p, j]
                xi = x_dm[idx]
                yi = y_dm[idx]
                r1 = yi - betas[0] * xi
                r2 = yi - betas[1] * xi
                if r1 * r1 <= r2 * r2:
                    betas[0] += eta * r1 * xi
                else:
                    betas[1] += eta * r2 * xi
        return betas

    @njit(cache=True)
    def _sgd_loop_multi(y_dm, X_dm, b1, b2, perms, n_passes, eta):
        """Inner SGD loop for d>=2, called with pre-generated permutations."""
        n = y_dm.shape[0]
        d = X_dm.shape[1]
        for p in range(n_passes):
            for j in range(n):
                idx = perms[p, j]
                yi = y_dm[idx]
                r1 = yi
                r2 = yi
                for k in range(d):
                    r1 -= b1[k] * X_dm[idx, k]
                    r2 -= b2[k] * X_dm[idx, k]
                if r1 * r1 <= r2 * r2:
                    for k in range(d):
                        b1[k] += eta * r1 * X_dm[idx, k]
                else:
                    for k in range(d):
                        b2[k] += eta * r2 * X_dm[idx, k]
        return b1, b2

    @njit(cache=True)
    def _sgd_loop_tracked(y_dm, x_dm, betas, perms, n_passes, eta,
                          record_every, S_x_sqrt_over_sigma):
        """Inner SGD loop with trajectory recording."""
        n = len(y_dm)
        max_records = (n * n_passes) // record_every + 1
        beta1_h = np.empty(max_records)
        beta2_h = np.empty(max_records)
        Tn_h = np.empty(max_records)
        iter_h = np.empty(max_records, dtype=np.int64)
        t_global = 0
        rec = 0
        for p in range(n_passes):
            for j in range(n):
                idx = perms[p, j]
                xi = x_dm[idx]
                yi = y_dm[idx]
                r1 = yi - betas[0] * xi
                r2 = yi - betas[1] * xi
                if r1 * r1 <= r2 * r2:
                    betas[0] += eta * r1 * xi
                else:
                    betas[1] += eta * r2 * xi
                t_global += 1
                if t_global % record_every == 0:
                    beta1_h[rec] = betas[0]
                    beta2_h[rec] = betas[1]
                    Tn_h[rec] = abs(betas[0] - betas[1]) * S_x_sqrt_over_sigma
                    iter_h[rec] = t_global
                    rec += 1
        return betas, beta1_h[:rec], beta2_h[:rec], Tn_h[:rec], iter_h[:rec]
else:
    _sgd_loop_scalar = None
    _sgd_loop_multi = None
    _sgd_loop_tracked = None


def _gen_perms(rng, n, n_passes):
    """Pre-generate all permutations for reproducibility."""
    return np.array([rng.permutation(n) for _ in range(n_passes)])


def sgd_competitive_scalar(y, x, eta=0.005, n_passes=100, seed=0):
    """Run competitive SGD for d=1.

    Parameters
    ----------
    y : array, shape (n,)
        Outcome variable (will be demeaned internally).
    x : array, shape (n,)
        Covariate (will be demeaned internally).
    eta : float
        Learning rate.
    n_passes : int
        Number of passes through the data.
    seed : int
        Random seed for SGD permutations.

    Returns
    -------
    dict with keys:
        S_T : float - gap statistic
        mu0 : float - null mean (2 * kappa_eps * kappa_x)
        sigma_T : float - null std
        ci_lo, ci_hi : float - 95% CI under H0
        p_value : float - one-sided p-value
        betas : array, shape (2,) - final estimates
    """
    n = len(y)
    y_dm = y - y.mean()
    x_dm = x - x.mean()
    S_x = np.mean(x_dm ** 2)
    if S_x < 1e-12:
        return {'S_T': np.nan, 'mu0': np.nan, 'sigma_T': np.nan,
                'ci_lo': np.nan, 'ci_hi': np.nan, 'p_value': 1.0,
                'betas': np.array([0.0, 0.0])}

    # OLS estimate and residuals
    beta_ols = np.sum(x_dm * y_dm) / np.sum(x_dm ** 2)
    resid = y_dm - beta_ols * x_dm
    sigma = np.sqrt(np.mean(resid ** 2))

    # Shape constants (Corollary 1)
    kappa_eps = np.mean(np.abs(resid)) / sigma
    kappa_x = np.mean(np.abs(x_dm)) / np.sqrt(S_x)
    mu0 = 2 * kappa_eps * kappa_x

    # Diffusion coefficient (Proposition 2)
    e_star = np.mean(np.abs(resid)) * np.mean(np.abs(x_dm)) / np.mean(x_dm ** 2)
    D = np.mean((resid - e_star * x_dm) ** 2 * x_dm ** 2
                * (resid * x_dm >= 0).astype(float))
    # Winner-only updating makes the half-gap innovation covariance D/2.
    sigma_T = np.sqrt(2 * eta * D / (sigma ** 2 + 1e-30))

    # Initialize symmetrically around OLS (Proposition 5)
    rng = np.random.default_rng(seed)
    eps_init = 0.01 * sigma
    betas = np.array([beta_ols + eps_init, beta_ols - eps_init])

    # SGD with competitive loss
    if HAS_NUMBA:
        perms = _gen_perms(rng, n, n_passes)
        betas = _sgd_loop_scalar(y_dm, x_dm, betas, perms, n_passes, eta)
    else:
        for _ in range(n_passes):
            perm = rng.permutation(n)
            for idx in perm:
                xi, yi = x_dm[idx], y_dm[idx]
                r1 = yi - betas[0] * xi
                r2 = yi - betas[1] * xi
                if r1 ** 2 <= r2 ** 2:
                    betas[0] += eta * r1 * xi
                else:
                    betas[1] += eta * r2 * xi

    # Gap statistic
    S_T = np.abs(betas[0] - betas[1]) * np.sqrt(S_x) / sigma
    ci_lo = mu0 - 1.96 * sigma_T
    ci_hi = mu0 + 1.96 * sigma_T
    p_value = 1 - stats.norm.cdf((S_T - mu0) / (sigma_T + 1e-30))

    return {'S_T': S_T, 'mu0': mu0, 'sigma_T': sigma_T,
            'ci_lo': ci_lo, 'ci_hi': ci_hi, 'p_value': p_value,
            'betas': betas}


def sgd_competitive_multi(y, x, eta=0.005, n_passes=100, seed=0,
                          null_seeds=20):
    """Run competitive SGD for d>=2.

    Parameters
    ----------
    y : array, shape (n,)
        Outcome variable.
    x : array, shape (n, d)
        Covariates.
    eta : float
        Learning rate.
    n_passes : int
        Number of passes through the data.
    seed : int
        Random seed for SGD permutations.
    null_seeds : int
        Number of null Monte Carlo replications for calibration.

    Returns
    -------
    dict with keys:
        S_T : float - gap statistic
        mu0 : float - null mean (simulated)
        sigma_T : float - null std (simulated)
        ci_lo, ci_hi : float - 95% CI under H0
        p_value : float - one-sided p-value
        betas : tuple of arrays - final (beta_1, beta_2)
    """
    n, d = x.shape
    y_dm = y - y.mean()
    X_dm = x - x.mean(axis=0)
    Sigma_x = X_dm.T @ X_dm / n
    beta_ols = np.linalg.solve(Sigma_x + 1e-10 * np.eye(d),
                               X_dm.T @ y_dm / n)
    resid = y_dm - X_dm @ beta_ols
    sigma = np.sqrt(np.mean(resid ** 2))

    # Empirical-covariate fitted-null bootstrap.  Resampling observed rows
    # preserves non-Gaussian projection ratios; random signs enforce the
    # symmetric-noise null without imposing a Gaussian residual law.
    null_S = []
    for ns in range(null_seeds):
        rng_n = np.random.default_rng(ns + 9999)
        X_null = X_dm[rng_n.integers(0, n, size=n)].copy()
        X_null -= X_null.mean(axis=0)
        eps_null = resid[rng_n.integers(0, n, size=n)].copy()
        eps_null *= rng_n.choice(np.array([-1.0, 1.0]), size=n)
        eps_null -= eps_null.mean()
        sigma_null = np.sqrt(np.mean(eps_null ** 2))
        Sigma_null = X_null.T @ X_null / n
        y_null = X_null @ beta_ols + eps_null
        y_null -= y_null.mean()
        direction_n = np.ones(d) / np.sqrt(d)
        b1n = beta_ols + 0.01 * sigma_null * direction_n
        b2n = beta_ols - 0.01 * sigma_null * direction_n
        if HAS_NUMBA:
            perms_n = _gen_perms(rng_n, n, n_passes)
            b1n, b2n = _sgd_loop_multi(y_null, X_null, b1n, b2n,
                                        perms_n, n_passes, eta)
        else:
            for _ in range(n_passes):
                perm = rng_n.permutation(n)
                for idx in perm:
                    xi = X_null[idx]
                    yi = y_null[idx]
                    r1 = yi - b1n @ xi
                    r2 = yi - b2n @ xi
                    if r1 ** 2 <= r2 ** 2:
                        b1n += eta * r1 * xi
                    else:
                        b2n += eta * r2 * xi
        dn = b1n - b2n
        null_S.append(np.sqrt(dn @ Sigma_null @ dn) / sigma_null)

    null_S = np.asarray(null_S)
    mu0 = np.mean(null_S)
    sigma_T = np.std(null_S) if np.std(null_S) > 1e-8 else 0.01

    # Run on actual data
    rng = np.random.default_rng(seed)
    eps_init = 0.01 * sigma
    direction = np.ones(d) / np.sqrt(d)
    b1 = beta_ols + eps_init * direction
    b2 = beta_ols - eps_init * direction

    if HAS_NUMBA:
        perms = _gen_perms(rng, n, n_passes)
        b1, b2 = _sgd_loop_multi(y_dm, X_dm, b1, b2, perms, n_passes, eta)
    else:
        for _ in range(n_passes):
            perm = rng.permutation(n)
            for idx in perm:
                xi = X_dm[idx]
                yi = y_dm[idx]
                r1 = yi - b1 @ xi
                r2 = yi - b2 @ xi
                if r1 ** 2 <= r2 ** 2:
                    b1 += eta * r1 * xi
                else:
                    b2 += eta * r2 * xi

    delta = b1 - b2
    S_T = np.sqrt(delta @ Sigma_x @ delta) / sigma
    ci_lo, ci_hi = np.quantile(null_S, [0.025, 0.975])
    p_value = (1.0 + np.count_nonzero(null_S >= S_T)) / (null_S.size + 1.0)

    return {'S_T': S_T, 'mu0': mu0, 'sigma_T': sigma_T,
            'ci_lo': ci_lo, 'ci_hi': ci_hi, 'p_value': p_value,
            'betas': (b1, b2)}


def sgd_competitive(y, x, eta=0.005, n_passes=100, seed=0, null_seeds=20):
    """Unified interface: dispatches to scalar or multivariate."""
    x = np.asarray(x)
    if x.ndim == 1:
        return sgd_competitive_scalar(y, x, eta, n_passes, seed)
    else:
        return sgd_competitive_multi(y, x, eta, n_passes, seed, null_seeds)


def sgd_competitive_tracked(y_dm, x_dm, eta, n_passes, seed=42,
                            beta_init=None, record_every=100):
    """Scalar competitive SGD with full trajectory tracking (for figures).

    Returns arrays of beta1, beta2, S_T, iteration counts, plus null stats.
    """
    rng = np.random.default_rng(seed)
    n = len(y_dm)
    S_x = np.mean(x_dm ** 2)

    if beta_init is not None:
        betas = np.array(beta_init, dtype=float)
    else:
        betas = rng.normal(0, 0.01, 2)

    beta_ols = np.sum(x_dm * y_dm) / np.sum(x_dm ** 2)
    resid = y_dm - beta_ols * x_dm
    sigma = np.sqrt(np.mean(resid ** 2))
    kappa_eps = np.mean(np.abs(resid)) / sigma
    kappa_x = np.mean(np.abs(x_dm)) / np.sqrt(S_x)
    mu0 = 2 * kappa_eps * kappa_x

    e_star = np.mean(np.abs(resid)) * np.mean(np.abs(x_dm)) / np.mean(x_dm ** 2)
    D = np.mean((resid - e_star * x_dm) ** 2 * x_dm ** 2
                * (resid * x_dm >= 0).astype(float))
    # Winner-only updating makes the half-gap innovation covariance D/2.
    sigma_T = np.sqrt(2 * eta * D / sigma ** 2)

    S_x_sqrt_over_sigma = np.sqrt(S_x) / sigma

    if HAS_NUMBA:
        perms = _gen_perms(rng, n, n_passes)
        betas, beta1_hist, beta2_hist, Tn_hist, iter_hist = \
            _sgd_loop_tracked(y_dm, x_dm, betas, perms, n_passes, eta,
                              record_every, S_x_sqrt_over_sigma)
    else:
        beta1_hist, beta2_hist, Tn_hist, iter_hist = [], [], [], []
        t_global = 0
        for _ in range(n_passes):
            perm = rng.permutation(n)
            for idx in perm:
                xi, yi = x_dm[idx], y_dm[idx]
                r1 = yi - betas[0] * xi
                r2 = yi - betas[1] * xi
                if r1 ** 2 <= r2 ** 2:
                    betas[0] += eta * r1 * xi
                else:
                    betas[1] += eta * r2 * xi
                t_global += 1
                if t_global % record_every == 0:
                    beta1_hist.append(betas[0])
                    beta2_hist.append(betas[1])
                    S_T = np.abs(betas[0] - betas[1]) * np.sqrt(S_x) / sigma
                    Tn_hist.append(S_T)
                    iter_hist.append(t_global)
        beta1_hist = np.array(beta1_hist)
        beta2_hist = np.array(beta2_hist)
        Tn_hist = np.array(Tn_hist)
        iter_hist = np.array(iter_hist)

    return (beta1_hist, beta2_hist, Tn_hist, iter_hist,
            mu0, sigma_T, beta_ols, sigma)
