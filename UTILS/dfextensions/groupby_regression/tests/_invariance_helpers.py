# -*- coding: utf-8 -*-
# _invariance_helpers.py
#
# Phase 13.8.GB — Shared Analytical Validation Utilities
#
# PURPOSE:
#   Provide regression-method-agnostic helper functions for invariance and
#   integration tests across all GroupBy regression implementations
#   (kernels, sliding window, v2, v4, v5).
#
# DESIGN PRINCIPLES:
#   1. Pure NumPy — no statsmodels, pandas, or heavy dependencies.
#   2. All tolerances derived from known physics (noise, sample size,
#      predictor distribution) — NO magic numbers.
#   3. Each function is self-contained and independently auditable.
#   4. Leading underscore in filename signals "not a test file" to pytest.
#
# THREE ANALYTICAL CHECKS:
#   Check 1: Value recovery — fitted ≈ truth within nsigma × SE
#   Check 2: Error estimator consistency — reported SE ≈ analytical SE
#   Check 3: Pull distribution — (fitted - truth) / SE has std ≈ 1.0
#
# Python 3.9.6 compatible.

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional


# =============================================================================
# Analytical Standard Errors (OLS)
# =============================================================================

def compute_ols_se(
    noise_std: float,
    n_eff: int,
    x_values: np.ndarray,
) -> Tuple[float, float]:
    """
    Analytical OLS standard errors for slope and intercept.

    For model:  Y = intercept + slope * X + ε,  ε ~ N(0, σ²)

    SE(slope)     = σ / sqrt(n × Var(X))
    SE(intercept) = σ × sqrt(E[X²] / (n × Var(X)))

    Parameters
    ----------
    noise_std : float
        Known standard deviation of the noise (σ).
    n_eff : int
        Effective number of observations in the fit.
    x_values : np.ndarray
        The predictor values (pooled across the window, if applicable).
        Used to compute Var(X) and E[X²].

    Returns
    -------
    se_slope : float
        Analytical standard error of the slope estimate.
    se_intercept : float
        Analytical standard error of the intercept estimate.

    Notes
    -----
    Returns (np.nan, np.nan) if n_eff < 2 or Var(X) ≈ 0.
    """
    if n_eff < 2:
        return (np.nan, np.nan)

    var_x = np.var(x_values, ddof=0)  # population variance of predictors
    if var_x < 1e-30:
        return (np.nan, np.nan)

    se_slope = noise_std / np.sqrt(n_eff * var_x)

    mean_x2 = np.mean(x_values ** 2)
    se_intercept = noise_std * np.sqrt(mean_x2 / (n_eff * var_x))

    return (se_slope, se_intercept)


def compute_ols_se_multi(
    noise_std: float,
    n_eff: int,
    X: np.ndarray,
) -> np.ndarray:
    """
    Analytical OLS standard errors for multiple predictors.

    For model: Y = β₀ + β₁X₁ + β₂X₂ + ... + ε

    SE(βⱼ) = σ × sqrt((X'X)⁻¹_jj)

    Parameters
    ----------
    noise_std : float
        Known noise standard deviation.
    n_eff : int
        Effective number of observations.
    X : np.ndarray, shape (n_eff, n_predictors)
        Predictor matrix (WITHOUT intercept column — added internally).

    Returns
    -------
    se : np.ndarray, shape (n_params,)
        Standard errors for [intercept, slope_1, slope_2, ...].
        Returns array of NaN if computation fails.
    """
    if n_eff < X.shape[1] + 2:
        return np.full(X.shape[1] + 1, np.nan)

    # Add intercept column
    X_design = np.column_stack([np.ones(n_eff), X])
    n_params = X_design.shape[1]

    try:
        XtX = X_design.T @ X_design
        XtX_inv = np.linalg.inv(XtX)
        se = noise_std * np.sqrt(np.diag(XtX_inv))
        return se
    except np.linalg.LinAlgError:
        return np.full(n_params, np.nan)


# =============================================================================
# Check 1: Value Recovery (nsigma cut)
# =============================================================================

def check_nsigma(
    fitted: float,
    true: float,
    se: float,
    nsigma: float = 4.0,
    label: str = "",
) -> float:
    """
    Assert that a fitted value is within nsigma standard errors of truth.

    z = |fitted - true| / SE
    Assert z < nsigma (default: 4σ → 99.99% confidence)

    Parameters
    ----------
    fitted : float
        The estimated coefficient.
    true : float
        The known true value.
    se : float
        The analytical standard error (from compute_ols_se).
    nsigma : float
        Number of standard deviations for the gate.
    label : str
        Descriptive label for error messages.

    Returns
    -------
    z : float
        The z-score |fitted - true| / SE.

    Raises
    ------
    AssertionError if z >= nsigma.
    """
    if np.isnan(se) or se <= 0:
        raise AssertionError(
            f"{label}: SE is invalid ({se}) — cannot perform nsigma check"
        )

    z = abs(fitted - true) / se
    assert z < nsigma, (
        f"{label}: z={z:.2f}σ ≥ {nsigma}σ "
        f"(fitted={fitted:.6f}, true={true:.6f}, SE={se:.6f})"
    )
    return z


# =============================================================================
# Check 2: Error Estimator Consistency (sigma check)
# =============================================================================

def adaptive_se_tolerance(
    n_eff: int,
    floor: float = 0.10,
    scale: float = 3.0,
) -> float:
    """
    Compute n_eff-dependent tolerance for error estimator consistency check.

    For large samples (n_eff >> 1), the tolerance approaches `floor`.
    For small samples, it widens to accommodate finite-sample effects
    (degrees-of-freedom corrections, chi² sampling variance of SE).

    Formula: tolerance = max(floor, scale / sqrt(n_eff))

    Examples:
        n_eff=1080 (interior, window=1, 3D) → tol ≈ 0.10
        n_eff=40   (boundary, single bin)    → tol ≈ 0.47

    Parameters
    ----------
    n_eff : int
        Effective number of observations in the fit.
    floor : float
        Minimum tolerance (for very large samples).
    scale : float
        Scaling factor for the 1/sqrt(n) term.

    Returns
    -------
    tolerance : float
    """
    if n_eff <= 0:
        return 1.0
    return max(floor, scale / np.sqrt(n_eff))


def check_error_ratio(
    reported_se: float,
    expected_se: float,
    n_eff: int = 0,
    tolerance: Optional[float] = None,
    label: str = "",
) -> float:
    """
    Assert that the reported standard error matches the analytical expectation.

    ratio = reported_SE / expected_SE
    Assert |ratio - 1.0| < tolerance

    If tolerance is None, uses adaptive_se_tolerance(n_eff).

    Parameters
    ----------
    reported_se : float
        The SE reported by the fitter (from res.bse / _err columns).
    expected_se : float
        The analytically computed SE (from compute_ols_se).
    n_eff : int
        Effective sample size (used for adaptive tolerance if tolerance=None).
    tolerance : float or None
        Fixed tolerance. If None, computed from n_eff.
    label : str
        Descriptive label for error messages.

    Returns
    -------
    ratio : float
        reported_se / expected_se.

    Raises
    ------
    AssertionError if |ratio - 1.0| >= tolerance.
    """
    if np.isnan(reported_se) or np.isnan(expected_se):
        raise AssertionError(
            f"{label}: SE is NaN (reported={reported_se}, expected={expected_se})"
        )
    if expected_se <= 0:
        raise AssertionError(
            f"{label}: expected_se must be positive (got {expected_se})"
        )

    if tolerance is None:
        tolerance = adaptive_se_tolerance(n_eff)

    ratio = reported_se / expected_se
    assert abs(ratio - 1.0) < tolerance, (
        f"{label}: SE ratio={ratio:.4f}, expected 1.0 ± {tolerance:.3f} "
        f"(reported={reported_se:.6f}, expected={expected_se:.6f}, n_eff={n_eff})"
    )
    return ratio


# =============================================================================
# Check 3: Pull Distribution (combined value + error check)
# =============================================================================

def compute_pulls(
    fitted_values: np.ndarray,
    true_value: float,
    reported_ses: np.ndarray,
) -> np.ndarray:
    """
    Compute standardised residuals (pulls).

    pull_i = (fitted_i - true) / reported_SE_i

    If the fitter is correct, pulls should be ~ N(0, 1).

    Parameters
    ----------
    fitted_values : np.ndarray
        Array of fitted coefficients (one per group/bin).
    true_value : float
        The known true value of the coefficient.
    reported_ses : np.ndarray
        Array of reported standard errors (one per group/bin).

    Returns
    -------
    pulls : np.ndarray
        Standardised residuals. NaN entries where SE is NaN or ≤ 0.
    """
    pulls = np.full_like(fitted_values, np.nan)
    valid = np.isfinite(reported_ses) & (reported_ses > 0)
    pulls[valid] = (fitted_values[valid] - true_value) / reported_ses[valid]
    return pulls


def check_pull_distribution(
    pulls: np.ndarray,
    nsigma: float = 4.0,
    df: int = 0,
    label: str = "",
) -> Tuple[float, float]:
    """
    Assert pull distribution has mean ≈ 0 and std ≈ expected_std.

    Mean gate: |mean(pulls)| < nsigma / sqrt(N)
    Std gate:  |std(pulls) - expected_std| < safety × SE(s)

    For large samples (df=0 or df > ~50), expected_std ≈ 1.0 (normal).
    For small samples with estimated σ², pulls follow t(df) and
    expected_std = sqrt(df/(df-2)).

    Parameters
    ----------
    pulls : np.ndarray
        Standardised residuals (NaN entries are excluded).
    nsigma : float
        Gate width in standard deviations.
    df : int
        Degrees of freedom for the t-distribution correction.
        If 0 (default), assumes normal distribution (expected_std=1.0).
        For OLS: df = n_rows_per_group - n_params.
    label : str
        Descriptive label for error messages.

    Returns
    -------
    (pull_mean, pull_std) : Tuple[float, float]

    Raises
    ------
    AssertionError if mean or std outside gates.
    """
    valid_pulls = pulls[np.isfinite(pulls)]
    n = len(valid_pulls)

    if n < 5:
        raise AssertionError(
            f"{label}: Too few valid pulls ({n}) for distribution check"
        )

    pull_mean = float(np.mean(valid_pulls))
    pull_std = float(np.std(valid_pulls, ddof=1))

    # Mean gate: should be consistent with zero
    mean_gate = nsigma / np.sqrt(n)
    assert abs(pull_mean) < mean_gate, (
        f"{label}: pull mean={pull_mean:.4f}, "
        f"expected 0.0 ± {mean_gate:.4f} ({nsigma}σ gate, N={n})"
    )

    # Expected std: normal (1.0) or t-distribution correction
    if df > 2:
        expected_std = np.sqrt(df / (df - 2.0))
    else:
        expected_std = 1.0

    # Std gate: SE of sample std ≈ expected_std / sqrt(2N)
    safety = 3.0
    std_gate = safety * expected_std / np.sqrt(2.0 * n)
    assert abs(pull_std - expected_std) < std_gate, (
        f"{label}: pull std={pull_std:.4f}, "
        f"expected {expected_std:.4f} ± {std_gate:.4f} "
        f"(safety={safety}, N={n}, df={df})"
    )

    return (pull_mean, pull_std)


# =============================================================================
# Bonus: RMSE check against known noise
# =============================================================================

def check_rmse_vs_noise(
    rmse_values: np.ndarray,
    noise_std: float,
    n_eff_values: np.ndarray,
    n_params: int = 2,
    nsigma: float = 4.0,
    label: str = "",
) -> float:
    """
    Assert that per-bin RMSE values are consistent with known noise level.

    For OLS with n observations and p parameters, the expected RMSE is:
        E[RMSE] = σ × sqrt((n - p) / n)    (bias-corrected)

    This corrects for the finite-sample downward bias of RMSE.

    Parameters
    ----------
    rmse_values : np.ndarray
        Per-bin RMSE values from the fit.
    noise_std : float
        Known noise standard deviation.
    n_eff_values : np.ndarray
        Per-bin effective sample sizes.
    n_params : int
        Number of fitted parameters (intercept + slopes).
    nsigma : float
        Gate width.
    label : str
        Descriptive label.

    Returns
    -------
    mean_ratio : float
        Mean of (RMSE / expected_RMSE) across valid bins.
    """
    valid = np.isfinite(rmse_values) & (n_eff_values > n_params)
    if np.sum(valid) < 3:
        raise AssertionError(
            f"{label}: Too few valid RMSE values ({np.sum(valid)})"
        )

    # Bias-corrected expected RMSE per bin
    expected = noise_std * np.sqrt(
        (n_eff_values[valid] - n_params) / n_eff_values[valid]
    )
    ratios = rmse_values[valid] / expected
    mean_ratio = float(np.mean(ratios))

    # Tolerance: RMSE ratio should be near 1.0
    # Use adaptive tolerance based on median n_eff
    median_n = float(np.median(n_eff_values[valid]))
    tol = adaptive_se_tolerance(int(median_n), floor=0.10, scale=3.0)

    assert abs(mean_ratio - 1.0) < tol, (
        f"{label}: mean RMSE ratio={mean_ratio:.4f}, "
        f"expected 1.0 ± {tol:.3f} (median n_eff={median_n:.0f})"
    )
    return mean_ratio
