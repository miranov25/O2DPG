# -*- coding: utf-8 -*-
# test_invariance_kernels.py
#
# Phase 13.8.GB — Kernel Invariance & Integration Tests
#
# PURPOSE:
#   Verify low-level Numba regression kernels against known ground truth
#   and cross-check against NumPy reference implementation.
#   Uses shared _invariance_helpers for nsigma-based validation —
#   NO magic numbers.
#
# TEST INVENTORY (4 tests):
#   1. test_kernel_single_nsigma_recovery    Check 1 (value, nsigma)
#   2. test_kernel_multi_nsigma_recovery     Check 1 (value, nsigma)
#   3. test_kernel_numba_equals_numpy_lstsq  A ≡ B parity (invariance)
#   4. test_kernel_single_equals_multi_t0    Consistency (invariance)
#
# WHAT THE KERNELS DO:
#   fit_groups_single_numba: OLS regression on N groups, one target.
#     Input: X_all (n_rows, n_feat), Y_all (n_rows,), offsets (n_groups+1,)
#     Output: beta (n_groups, n_params), errors (n_groups, n_params), ...
#     Each group is an independent OLS fit on rows[offsets[i]:offsets[i+1]].
#
#   fit_groups_multifit_numba: Same but multiple targets simultaneously,
#     sharing the XtX computation across targets.
#     Input: Y_all (n_rows, n_tgt), Output: beta (n_groups, n_tgt, n_params)
#
#   out_errors contains sqrt(s² × diag(XtX⁻¹)) — equivalent to res.bse
#   in statsmodels (OLS coefficient standard errors).
#
# Python 3.9.6 compatible.

from __future__ import annotations

import numpy as np
import pytest

# Import shared helpers
try:
    from ._invariance_helpers import (
        compute_ols_se,
        check_nsigma,
        check_error_ratio,
        compute_pulls,
        check_pull_distribution,
    )
except ImportError:
    from _invariance_helpers import (
        compute_ols_se,
        check_nsigma,
        check_error_ratio,
        compute_pulls,
        check_pull_distribution,
    )


# =============================================================================
# Imports — kernel layer
# =============================================================================

try:
    from groupby_regression_kernels import (
        fit_groups_single_numba,
        fit_groups_multifit_numba,
        INVALID_DETECT,
        STATUS_OK,
    )
    _KERNEL_AVAILABLE = True
except ImportError:
    try:
        from ..groupby_regression_kernels import (
            fit_groups_single_numba,
            fit_groups_multifit_numba,
            INVALID_DETECT,
            STATUS_OK,
        )
        _KERNEL_AVAILABLE = True
    except ImportError:
        _KERNEL_AVAILABLE = False


# =============================================================================
# Test Constants
# =============================================================================

NSIGMA = 4.0          # 99.99% confidence per group
NOISE_STD = 0.1       # Low noise for tight kernel validation
TRUE_INTERCEPT = 1.0
TRUE_SLOPES_BASE = 2.0  # slopes = [2.0, 3.0, 4.0, ...] for n_feat predictors


# =============================================================================
# Data Generators
# =============================================================================

def _generate_kernel_data(
    n_groups: int,
    rows_per_group: int,
    n_feat: int,
    n_targets: int = 1,
    seed: int = 42,
    noise_std: float = NOISE_STD,
):
    """
    Generate grouped regression data with known true coefficients.

    Y = intercept + X @ slopes + noise
    True coefficients: intercept=1.0, slopes=[2.0, 3.0, 4.0, ...]

    Returns: X_all, Y_all, W_all, offsets, true_coeffs
    """
    np.random.seed(seed)
    n_rows = n_groups * rows_per_group

    X_all = np.random.randn(n_rows, n_feat)

    true_slopes = np.arange(TRUE_SLOPES_BASE, TRUE_SLOPES_BASE + n_feat)
    true_coeffs = np.concatenate([[TRUE_INTERCEPT], true_slopes])

    Y_base = TRUE_INTERCEPT + X_all @ true_slopes

    if n_targets == 1:
        Y_all = Y_base + np.random.randn(n_rows) * noise_std
    else:
        Y_all = np.column_stack([
            Y_base * (1.0 + 0.1 * t) + np.random.randn(n_rows) * noise_std
            for t in range(n_targets)
        ])
        true_coeffs = np.stack([
            true_coeffs * (1.0 + 0.1 * t) for t in range(n_targets)
        ])

    W_all = np.empty(0, dtype=np.float64)
    offsets = np.arange(0, n_rows + 1, rows_per_group, dtype=np.int64)

    return X_all, Y_all, W_all, offsets, true_coeffs


def _allocate_single_outputs(n_groups: int, n_params: int):
    """Pre-allocate output arrays for single-fit kernel."""
    return (
        np.empty((n_groups, n_params), dtype=np.float64),       # beta
        np.empty((n_groups, n_params), dtype=np.float64),       # errors (bse)
        np.empty(n_groups, dtype=np.float64),                    # rms
        np.empty(n_groups, dtype=np.float64),                    # mad
        np.empty(n_groups, dtype=np.uint8),                      # status
        np.empty(n_groups, dtype=np.int64),                      # n_valid
        np.empty(n_groups, dtype=np.int64),                      # n_filtered
        np.empty(n_groups, dtype=np.float64),                    # cond
    )


def _allocate_multi_outputs(n_groups: int, n_targets: int, n_params: int):
    """Pre-allocate output arrays for multi-fit kernel."""
    return (
        np.empty((n_groups, n_targets, n_params), dtype=np.float64),  # beta
        np.empty((n_groups, n_targets, n_params), dtype=np.float64),  # errors
        np.empty((n_groups, n_targets), dtype=np.float64),            # rms
        np.empty((n_groups, n_targets), dtype=np.float64),            # mad
        np.empty((n_groups, n_targets), dtype=np.uint8),              # status
        np.empty(n_groups, dtype=np.int64),                           # n_valid
        np.empty(n_groups, dtype=np.int64),                           # n_filtered
        np.empty(n_groups, dtype=np.float64),                         # cond
    )


def _run_single_fit(n_groups, rows_per_group, n_feat, seed=42, noise_std=NOISE_STD):
    """Run single-fit kernel and return (out_tuple, true_coeffs, X_all)."""
    n_params = n_feat + 1
    X, Y, W, offsets, true_coeffs = _generate_kernel_data(
        n_groups, rows_per_group, n_feat, seed=seed, noise_std=noise_std,
    )
    out = _allocate_single_outputs(n_groups, n_params)
    fit_groups_single_numba(
        X, Y, W, offsets, n_groups, n_feat, n_params,
        True, 5, False, INVALID_DETECT, *out,
    )
    return out, true_coeffs, X, offsets


# #############################################################################
# Test 1: Kernel Single-Fit Value Recovery (Check 1)
# #############################################################################

@pytest.mark.skipif(not _KERNEL_AVAILABLE, reason="Kernel module not available")
class TestKernelSingleFitTruth:
    """
    Verify that Numba single-fit kernel recovers known coefficients
    within nsigma × analytical SE.

    Also validates error estimator (out_errors ≈ analytical SE) and
    pull distribution across groups.
    """

    @pytest.mark.parametrize("n_groups,rows_per_group,n_feat", [
        (500, 20, 2),    # Many small groups
        (200, 50, 4),    # Fewer groups, more features
    ])
    def test_kernel_single_nsigma_recovery(self, n_groups, rows_per_group, n_feat):
        """
        Test 1: Per-group coefficient recovery within nsigma.

        Three-level check:
          a) Value recovery: beta ≈ truth within nsigma × SE
          b) Error estimator: out_errors ≈ analytical SE
          c) Pull distribution: std(pull) ≈ 1.0
        """
        out, true_coeffs, X_all, offsets = _run_single_fit(
            n_groups, rows_per_group, n_feat,
        )
        out_beta, out_errors, out_rms, _, out_status, _, _, _ = out
        n_params = n_feat + 1

        # Select OK groups
        ok_mask = out_status == STATUS_OK
        assert np.sum(ok_mask) > n_groups * 0.9, (
            f"Too few OK groups: {np.sum(ok_mask)}/{n_groups}"
        )

        ok_indices = np.where(ok_mask)[0]

        # --- Check 1a: Value recovery per group (spot-check first 50) ---
        for gi in ok_indices[:50]:
            i0, i1 = offsets[gi], offsets[gi + 1]
            x_group = X_all[i0:i1]

            for p in range(n_params):
                if p == 0:
                    # Intercept
                    _, se = compute_ols_se(NOISE_STD, rows_per_group, x_group[:, 0])
                else:
                    # Slope for predictor p-1
                    se, _ = compute_ols_se(NOISE_STD, rows_per_group, x_group[:, p - 1])

                if np.isnan(se) or se <= 0:
                    continue

                check_nsigma(
                    out_beta[gi, p], true_coeffs[p], se,
                    nsigma=NSIGMA,
                    label=f"single group={gi} param={p}",
                )

        # --- Check 1b: Error estimator across all OK groups ---
        # For slope parameters, compare mean(out_errors) to analytical SE
        for p in range(1, n_params):
            # Use global X column to compute analytical SE
            reported_mean = np.mean(out_errors[ok_mask, p])
            # Analytical SE: for a typical group
            x_col = X_all[:rows_per_group, p - 1]
            se_analytical, _ = compute_ols_se(NOISE_STD, rows_per_group, x_col)
            if np.isnan(se_analytical):
                continue

            check_error_ratio(
                reported_mean, se_analytical,
                n_eff=rows_per_group,
                label=f"single slope_err param={p}",
            )

        # --- Check 1c: Pull distribution for slope[0] ---
        p = 1  # first slope
        slopes = out_beta[ok_mask, p]
        slope_errs = out_errors[ok_mask, p]
        pulls = compute_pulls(slopes, true_coeffs[p], slope_errs)
        # Kernels use estimated σ² with df = rows_per_group - n_params
        # so pulls follow t(df), not N(0,1)
        check_pull_distribution(
            pulls, nsigma=NSIGMA,
            df=rows_per_group - n_params,
            label=f"single slope pull (n_feat={n_feat})",
        )


# #############################################################################
# Test 2: Kernel Multi-Fit Value Recovery (Check 1)
# #############################################################################

@pytest.mark.skipif(not _KERNEL_AVAILABLE, reason="Kernel module not available")
class TestKernelMultiFitTruth:
    """
    Verify that Numba multi-fit kernel recovers known coefficients
    for multiple targets simultaneously.
    """

    @pytest.mark.parametrize("n_targets", [2, 6])
    def test_kernel_multi_nsigma_recovery(self, n_targets):
        """
        Test 2: Multi-target coefficient recovery within nsigma.
        """
        n_groups, rows_per_group, n_feat = 500, 20, 2
        n_params = n_feat + 1

        X, Y, W, offsets, true_coeffs = _generate_kernel_data(
            n_groups, rows_per_group, n_feat, n_targets=n_targets, seed=42,
        )
        out = _allocate_multi_outputs(n_groups, n_targets, n_params)
        out_beta, out_errors, _, _, out_status, _, _, _ = out

        fit_groups_multifit_numba(
            X, Y, W, offsets, n_groups, n_feat, n_targets, n_params,
            True, 5, False, *out,
        )

        # Check each target
        for t in range(n_targets):
            ok_mask = out_status[:, t] == STATUS_OK
            assert np.sum(ok_mask) > n_groups * 0.9, (
                f"Target {t}: too few OK groups: {np.sum(ok_mask)}/{n_groups}"
            )

            ok_indices = np.where(ok_mask)[0]

            # Value recovery: spot-check first 50 groups
            for gi in ok_indices[:50]:
                i0, i1 = offsets[gi], offsets[gi + 1]
                x_group = X[i0:i1]

                for p in range(1, n_params):  # slopes only (intercept SE formula differs per target scaling)
                    se, _ = compute_ols_se(NOISE_STD, rows_per_group, x_group[:, p - 1])
                    if np.isnan(se) or se <= 0:
                        continue

                    check_nsigma(
                        out_beta[gi, t, p], true_coeffs[t, p], se,
                        nsigma=NSIGMA,
                        label=f"multi target={t} group={gi} param={p}",
                    )

            # Pull distribution for first slope of this target
            slopes_t = out_beta[ok_mask, t, 1]
            errs_t = out_errors[ok_mask, t, 1]
            pulls = compute_pulls(slopes_t, true_coeffs[t, 1], errs_t)
            # Kernels use estimated σ² with df = rows_per_group - n_params
            check_pull_distribution(
                pulls, nsigma=NSIGMA,
                df=rows_per_group - n_params,
                label=f"multi target={t} slope pull",
            )


# #############################################################################
# Test 3: Numba ≡ NumPy lstsq Parity (Invariance)
# #############################################################################

@pytest.mark.skipif(not _KERNEL_AVAILABLE, reason="Kernel module not available")
class TestKernelNumbaNumpyParity:
    """
    Cross-check: Numba kernel output ≡ NumPy np.linalg.lstsq per group.
    This is an implementation-parity invariance test (A ≡ B).
    """

    def test_kernel_numba_equals_numpy_lstsq(self):
        """
        Test 3: For each group, Numba beta ≈ NumPy lstsq beta.
        """
        n_groups, rows_per_group, n_feat = 200, 30, 2
        n_params = n_feat + 1

        out, _, X_all, offsets = _run_single_fit(
            n_groups, rows_per_group, n_feat,
        )
        out_beta, out_errors, _, _, out_status, _, _, _ = out

        # Regenerate Y for numpy path (same seed)
        _, Y_all, _, _, _ = _generate_kernel_data(
            n_groups, rows_per_group, n_feat, seed=42,
        )

        n_checked = 0
        for gi in range(n_groups):
            if out_status[gi] != STATUS_OK:
                continue

            i0, i1 = offsets[gi], offsets[gi + 1]
            X_slice = X_all[i0:i1]
            Y_slice = Y_all[i0:i1]
            X_design = np.column_stack([np.ones(len(X_slice)), X_slice])

            # NumPy reference
            beta_np, _, _, _ = np.linalg.lstsq(X_design, Y_slice, rcond=None)

            # Coefficients: tight tolerance (same data, different code paths)
            np.testing.assert_allclose(
                out_beta[gi], beta_np,
                rtol=1e-8, atol=1e-10,
                err_msg=f"Numba ≠ NumPy for group {gi}",
            )

            # Also check error estimates against numpy computation
            resid = Y_slice - X_design @ beta_np
            s2 = np.sum(resid ** 2) / (len(Y_slice) - n_params)
            XtX_inv = np.linalg.inv(X_design.T @ X_design)
            se_np = np.sqrt(s2 * np.diag(XtX_inv))

            np.testing.assert_allclose(
                out_errors[gi], se_np,
                rtol=1e-6, atol=1e-10,
                err_msg=f"Numba errors ≠ NumPy errors for group {gi}",
            )

            n_checked += 1

        assert n_checked > n_groups * 0.9, (
            f"Too few groups checked: {n_checked}/{n_groups}"
        )


# #############################################################################
# Test 4: Single-Fit ≡ Multi-Fit[target=0] Parity (Invariance)
# #############################################################################

@pytest.mark.skipif(not _KERNEL_AVAILABLE, reason="Kernel module not available")
class TestKernelSingleMultiParity:
    """
    Invariance: single-fit on Y[:,0] must produce identical beta
    to multi-fit target 0. Same math, different code path.
    """

    def test_kernel_single_equals_multi_target0(self):
        """
        Test 4: single-fit beta ≡ multi-fit beta[target=0].
        """
        n_groups, rows_per_group, n_feat = 300, 20, 2
        n_targets = 3
        n_params = n_feat + 1

        X, Y_multi, W, offsets, _ = _generate_kernel_data(
            n_groups, rows_per_group, n_feat, n_targets=n_targets, seed=42,
        )
        Y_single = Y_multi[:, 0].copy()

        # Single-fit
        out_s = _allocate_single_outputs(n_groups, n_params)
        fit_groups_single_numba(
            X, Y_single, W, offsets, n_groups, n_feat, n_params,
            True, 5, False, INVALID_DETECT, *out_s,
        )
        beta_single, errors_single = out_s[0], out_s[1]
        status_single = out_s[4]

        # Multi-fit
        out_m = _allocate_multi_outputs(n_groups, n_targets, n_params)
        fit_groups_multifit_numba(
            X, Y_multi, W, offsets, n_groups, n_feat, n_targets, n_params,
            True, 5, False, *out_m,
        )
        beta_multi_t0, errors_multi_t0 = out_m[0][:, 0, :], out_m[1][:, 0, :]
        status_multi_t0 = out_m[4][:, 0]

        # Compare only groups where both report OK
        both_ok = (status_single == STATUS_OK) & (status_multi_t0 == STATUS_OK)
        assert np.sum(both_ok) > n_groups * 0.9, (
            f"Too few OK groups: {np.sum(both_ok)}/{n_groups}"
        )

        # Coefficients must be identical (same data, same math)
        np.testing.assert_allclose(
            beta_single[both_ok], beta_multi_t0[both_ok],
            rtol=1e-10, atol=1e-12,
            err_msg="Single-fit beta ≠ multi-fit beta[target=0]",
        )

        # Error estimates must also match
        np.testing.assert_allclose(
            errors_single[both_ok], errors_multi_t0[both_ok],
            rtol=1e-10, atol=1e-12,
            err_msg="Single-fit errors ≠ multi-fit errors[target=0]",
        )
