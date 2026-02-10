# -*- coding: utf-8 -*-
# test_invariance_sliding_window.py
#
# Phase 13.8.GB — Sliding Window Invariance & Integration Tests
#
# PURPOSE:
#   Lock correctness of make_sliding_window_fit BEFORE optimisation begins.
#   All tolerances derived from known physics (noise, sample size, predictor
#   distribution) — NO magic numbers.
#
# THREE ANALYTICAL CHECKS:
#   Check 1: Value recovery — fitted ≈ truth within nsigma × SE
#   Check 2: Error estimator consistency — reported SE ≈ analytical SE
#   Check 3: Pull distribution — (fitted - truth) / reported_SE has std ≈ 1
#
# TEST INVENTORY (13 tests):
#   1. test_sw_slope_nsigma_recovery          Check 1, parametrised window=0,1
#   2. test_sw_intercept_nsigma_recovery      Check 1, parametrised window=0,1
#   3. test_sw_error_estimator_consistency    Check 2, parametrised window=0,1
#   4. test_sw_pull_distribution              Check 3, window=1
#   5. test_sw_rmse_vs_known_noise            RMSE bonus, parametrised window=0,1
#   6. test_sw_window0_entries_equals_bin      Structural
#   7. test_sw_window0_neighbors_equals_one    Structural
#   8. test_sw_interior_entries_27x            Structural (3D)
#   9. test_sw_permutation_invariance          Metamorphic, window=1
#  10. test_sw_determinism                     Metamorphic
#  11. test_fast_sw_matches_current            Placeholder (skip)
#  12. test_sw_multi_predictor_nsigma_recovery Check 1, multi-predictor (P1-1)
#  13. test_sw_window0_equals_per_bin_ols      Oracle A≡B (P1-2)
#
# Python 3.9.6 compatible.

from __future__ import annotations

import itertools
import numpy as np
import pandas as pd
import pytest

# Import shared helpers (pure NumPy, no heavy deps)
try:
    from ._invariance_helpers import (
        compute_ols_se,
        compute_ols_se_multi,
        check_nsigma,
        adaptive_se_tolerance,
        check_error_ratio,
        compute_pulls,
        check_pull_distribution,
        check_rmse_vs_noise,
    )
except ImportError:
    from _invariance_helpers import (
        compute_ols_se,
        compute_ols_se_multi,
        check_nsigma,
        adaptive_se_tolerance,
        check_error_ratio,
        compute_pulls,
        check_pull_distribution,
        check_rmse_vs_noise,
    )

# =============================================================================
# Imports — sliding window
# =============================================================================

try:
    from groupby_regression_sliding_window import make_sliding_window_fit
    _SW_AVAILABLE = True
except ImportError:
    try:
        from ..groupby_regression_sliding_window import make_sliding_window_fit
        _SW_AVAILABLE = True
    except ImportError:
        _SW_AVAILABLE = False


# =============================================================================
# Test Constants — derived from physics, not magic
# =============================================================================

# Synthetic data parameters (known ground truth)
TRUE_INTERCEPT = 1.0
TRUE_SLOPE = 2.0
NOISE_STD = 0.5
SEED = 42

# Multi-predictor ground truth
TRUE_SLOPE_X1 = 2.0
TRUE_SLOPE_X2 = 3.0

# Grid parameters (small for speed)
N_BINS_PER_DIM = 4       # 4³ = 64 bins
ENTRIES_PER_BIN = 40      # 40 rows per bin → 2560 total

# Statistical gate
NSIGMA = 4.0             # 99.99% confidence per bin


# =============================================================================
# Data Generators
# =============================================================================

def _make_sw_grid_single(
    n_bins_per_dim: int = N_BINS_PER_DIM,
    entries_per_bin: int = ENTRIES_PER_BIN,
    seed: int = SEED,
    intercept: float = TRUE_INTERCEPT,
    slope: float = TRUE_SLOPE,
    noise_std: float = NOISE_STD,
) -> pd.DataFrame:
    """
    Build a dense 3D integer grid with known linear truth:
        value = intercept + slope × x + ε,  ε ~ N(0, σ²)

    Returns a DataFrame with columns: xBin, yBin, zBin, x, value, weight.
    """
    rng = np.random.default_rng(seed)
    bins = np.array(list(itertools.product(
        range(n_bins_per_dim),
        range(n_bins_per_dim),
        range(n_bins_per_dim),
    )))
    bins_expanded = np.repeat(bins, entries_per_bin, axis=0)
    df = pd.DataFrame(bins_expanded, columns=['xBin', 'yBin', 'zBin']).astype(np.int32)
    df['x'] = rng.normal(0.0, 1.0, len(df))
    df['value'] = intercept + slope * df['x'] + rng.normal(0.0, noise_std, len(df))
    df['weight'] = 1.0
    return df


def _make_sw_grid_multi(
    n_bins_per_dim: int = N_BINS_PER_DIM,
    entries_per_bin: int = ENTRIES_PER_BIN,
    seed: int = SEED,
    intercept: float = TRUE_INTERCEPT,
    slope_x1: float = TRUE_SLOPE_X1,
    slope_x2: float = TRUE_SLOPE_X2,
    noise_std: float = NOISE_STD,
) -> pd.DataFrame:
    """
    Build a dense 3D grid with known multi-predictor truth:
        value = intercept + slope_x1 × x1 + slope_x2 × x2 + ε

    Uses near-orthogonal predictors (independent normals) to avoid
    ill-conditioning artifacts (per Claude12/reviewer consensus).
    """
    rng = np.random.default_rng(seed)
    bins = np.array(list(itertools.product(
        range(n_bins_per_dim),
        range(n_bins_per_dim),
        range(n_bins_per_dim),
    )))
    bins_expanded = np.repeat(bins, entries_per_bin, axis=0)
    df = pd.DataFrame(bins_expanded, columns=['xBin', 'yBin', 'zBin']).astype(np.int32)
    df['x1'] = rng.normal(0.0, 1.0, len(df))
    df['x2'] = rng.normal(0.0, 1.0, len(df))
    df['value'] = (
        intercept
        + slope_x1 * df['x1']
        + slope_x2 * df['x2']
        + rng.normal(0.0, noise_std, len(df))
    )
    df['weight'] = 1.0
    return df


def _run_sw_fit(
    df: pd.DataFrame,
    window_size: int,
    linear_columns: list,
    min_stat: int = 5,
) -> pd.DataFrame:
    """Helper to run make_sliding_window_fit with standard parameters."""
    return make_sliding_window_fit(
        df=df,
        gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': window_size, 'yBin': window_size, 'zBin': window_size},
        fit_columns=['value'],
        linear_columns=linear_columns,
        min_stat=min_stat,
        suffix='',
    )


def _get_interior_mask(result: pd.DataFrame, n_bins: int) -> pd.Series:
    """Return boolean mask for interior bins (not on any boundary)."""
    return (
        (result['xBin'] > 0) & (result['xBin'] < n_bins - 1) &
        (result['yBin'] > 0) & (result['yBin'] < n_bins - 1) &
        (result['zBin'] > 0) & (result['zBin'] < n_bins - 1)
    )


# #############################################################################
# Tests 1–2: Slope and Intercept Value Recovery (Check 1)
# #############################################################################

@pytest.mark.skipif(not _SW_AVAILABLE, reason="Sliding window module not available")
class TestSWValueRecovery:
    """
    Check 1: Fitted coefficients match known truth within nsigma × SE.

    Per-bin verification: each bin has different n_eff from window aggregation.
    Interior bins have n_eff = (2w+1)^3 × entries_per_bin.
    Boundary bins have fewer neighbors → wider tolerance.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("statsmodels")

    @pytest.mark.parametrize("window_size", [0, 1])
    def test_sw_slope_nsigma_recovery(self, window_size):
        """Test 1: Mean slope across bins is within nsigma of truth."""
        df = _make_sw_grid_single()
        result = _run_sw_fit(df, window_size, ['x'])

        slopes = result['value_slope_x'].dropna()
        assert len(slopes) > 0, "No slopes recovered"

        # Per-bin nsigma check
        for _, row in result.dropna(subset=['value_slope_x']).iterrows():
            n_eff = int(row['n_rows_aggregated'])
            # Reconstruct pooled x values for this bin's window
            # Use analytical SE with known noise and n_eff
            # For uniform grid with N(0,1) predictors, var(x) ≈ 1.0
            # but we use a conservative estimate based on n_eff
            se_slope, _ = compute_ols_se(NOISE_STD, n_eff, df['x'].values[:n_eff])
            if np.isnan(se_slope):
                continue

            check_nsigma(
                row['value_slope_x'], TRUE_SLOPE, se_slope,
                nsigma=NSIGMA,
                label=f"slope bin ({row['xBin']},{row['yBin']},{row['zBin']}) w={window_size}",
            )

    @pytest.mark.parametrize("window_size", [0, 1])
    def test_sw_intercept_nsigma_recovery(self, window_size):
        """Test 2: Mean intercept across bins is within nsigma of truth."""
        df = _make_sw_grid_single()
        result = _run_sw_fit(df, window_size, ['x'])

        intercepts = result['value_intercept'].dropna()
        assert len(intercepts) > 0, "No intercepts recovered"

        for _, row in result.dropna(subset=['value_intercept']).iterrows():
            n_eff = int(row['n_rows_aggregated'])
            _, se_intercept = compute_ols_se(NOISE_STD, n_eff, df['x'].values[:n_eff])
            if np.isnan(se_intercept):
                continue

            check_nsigma(
                row['value_intercept'], TRUE_INTERCEPT, se_intercept,
                nsigma=NSIGMA,
                label=f"intercept bin ({row['xBin']},{row['yBin']},{row['zBin']}) w={window_size}",
            )


# #############################################################################
# Test 3: Error Estimator Consistency (Check 2)
# #############################################################################

@pytest.mark.skipif(not _SW_AVAILABLE, reason="Sliding window module not available")
class TestSWErrorEstimator:
    """
    Check 2: Reported standard errors (from res.bse / _err columns) match
    analytically expected standard errors.

    Uses adaptive n_eff-dependent tolerance (P1-3, Claude12 formula):
    - Interior bins (n_eff=1080): tolerance ≈ 0.10
    - Boundary bins (n_eff=40): tolerance ≈ 0.47
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("statsmodels")

    @pytest.mark.parametrize("window_size", [0, 1])
    def test_sw_error_estimator_consistency(self, window_size):
        """Test 3: Reported SE ≈ analytical SE within adaptive tolerance."""
        df = _make_sw_grid_single()
        result = _run_sw_fit(df, window_size, ['x'])

        # Only check bins with valid fit AND valid _err columns
        valid = result.dropna(subset=['value_slope_x', 'value_slope_x_err'])
        assert len(valid) > 0, "No valid bins with _err columns"

        # Guard: skip bins with var(x) ≈ 0 (P1-5)
        n_checked = 0
        for _, row in valid.iterrows():
            n_eff = int(row['n_rows_aggregated'])
            if n_eff < 5:
                continue

            # Analytical SE
            se_slope, se_intercept = compute_ols_se(
                NOISE_STD, n_eff, df['x'].values[:n_eff]
            )
            if np.isnan(se_slope) or se_slope <= 0:
                continue

            reported_se = row['value_slope_x_err']
            if np.isnan(reported_se) or reported_se <= 0:
                continue

            check_error_ratio(
                reported_se, se_slope,
                n_eff=n_eff,
                label=f"slope_err bin ({row['xBin']},{row['yBin']},{row['zBin']}) w={window_size}",
            )
            n_checked += 1

        assert n_checked > 0, "No bins passed the guard checks for error estimator"


# #############################################################################
# Test 4: Pull Distribution (Check 3)
# #############################################################################

@pytest.mark.skipif(not _SW_AVAILABLE, reason="Sliding window module not available")
class TestSWPullDistribution:
    """
    Check 3: The standardised residuals (pulls) across bins have
    mean ≈ 0 and std ≈ 1, confirming both value recovery AND error
    estimator are jointly correct.

    Uses window=1 for more bins and thus better pull statistics.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("statsmodels")

    def test_sw_pull_distribution(self):
        """Test 4: Pull distribution has mean≈0, std≈1."""
        df = _make_sw_grid_single()
        result = _run_sw_fit(df, window_size=1, linear_columns=['x'])

        valid = result.dropna(subset=['value_slope_x', 'value_slope_x_err'])
        assert len(valid) > 5, "Too few valid bins for pull distribution"

        slopes = valid['value_slope_x'].values
        slope_errs = valid['value_slope_x_err'].values

        # Guard: exclude bins with non-positive SE
        good = slope_errs > 0
        pulls = compute_pulls(slopes[good], TRUE_SLOPE, slope_errs[good])

        check_pull_distribution(
            pulls, nsigma=NSIGMA,
            # SW uses statsmodels OLS with estimated σ²; for window=1 interior
            # bins, df is large (~1078) so correction is negligible, but we
            # pass it for correctness. Use median n_eff as representative df.
            df=max(int(np.median(valid['n_rows_aggregated'].values)) - 2, 0),
            label="slope pull distribution (window=1)",
        )


# #############################################################################
# Test 5: RMSE vs Known Noise (Bonus)
# #############################################################################

@pytest.mark.skipif(not _SW_AVAILABLE, reason="Sliding window module not available")
class TestSWRmse:
    """
    Bonus: Per-bin RMSE should recover the input noise level,
    with bias correction for finite sample size.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("statsmodels")

    @pytest.mark.parametrize("window_size", [0, 1])
    def test_sw_rmse_vs_known_noise(self, window_size):
        """Test 5: RMSE ≈ σ_noise (bias-corrected)."""
        df = _make_sw_grid_single()
        result = _run_sw_fit(df, window_size, ['x'])

        valid = result.dropna(subset=['value_rmse'])
        rmse_vals = valid['value_rmse'].values
        n_eff_vals = valid['n_rows_aggregated'].values.astype(float)

        check_rmse_vs_noise(
            rmse_vals, NOISE_STD, n_eff_vals,
            n_params=2,  # intercept + slope
            nsigma=NSIGMA,
            label=f"RMSE vs noise (window={window_size})",
        )


# #############################################################################
# Tests 6–8: Structural Invariants
# #############################################################################

@pytest.mark.skipif(not _SW_AVAILABLE, reason="Sliding window module not available")
class TestSWStructuralInvariants:
    """
    Structural checks on combinatorial/integer outputs.
    These must be EXACT (P1-6: integer outputs use exact equality).
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("statsmodels")

    def test_sw_window0_entries_equals_bin(self):
        """Test 6: window=0 → n_rows_aggregated == entries_per_bin (exact)."""
        df = _make_sw_grid_single()
        result = _run_sw_fit(df, window_size=0, linear_columns=['x'], min_stat=1)

        for _, row in result.iterrows():
            assert row['n_rows_aggregated'] == ENTRIES_PER_BIN, (
                f"Bin ({row['xBin']},{row['yBin']},{row['zBin']}): "
                f"expected {ENTRIES_PER_BIN}, got {row['n_rows_aggregated']}"
            )
            assert row['value_entries'] == ENTRIES_PER_BIN

    def test_sw_window0_neighbors_equals_one(self):
        """Test 7: window=0 → n_neighbors_used == 1 (exact, self only)."""
        df = _make_sw_grid_single(n_bins_per_dim=3, entries_per_bin=20)
        result = _run_sw_fit(df, window_size=0, linear_columns=['x'], min_stat=1)

        assert (result['n_neighbors_used'] == 1).all(), (
            "window=0 should use exactly 1 neighbor (self)"
        )

    def test_sw_interior_entries_27x(self):
        """Test 8: window=1, 3D interior → n_rows_aggregated == 27 × entries_per_bin (exact)."""
        entries = 20
        n_bins = 4
        df = _make_sw_grid_single(n_bins_per_dim=n_bins, entries_per_bin=entries)
        result = _run_sw_fit(df, window_size=1, linear_columns=[], min_stat=1)

        interior = result[_get_interior_mask(result, n_bins)]
        expected = 27 * entries  # (2×1+1)³ = 27 neighbors × entries_per_bin
        for _, row in interior.iterrows():
            assert row['n_rows_aggregated'] == expected, (
                f"Interior bin ({row['xBin']},{row['yBin']},{row['zBin']}): "
                f"expected {expected}, got {row['n_rows_aggregated']}"
            )


# #############################################################################
# Tests 9–10: Metamorphic Tests
# #############################################################################

@pytest.mark.skipif(not _SW_AVAILABLE, reason="Sliding window module not available")
class TestSWMetamorphic:
    """
    Metamorphic properties: the result should be invariant to
    row permutation and deterministic across repeated runs.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("statsmodels")

    def test_sw_permutation_invariance(self):
        """Test 9: Shuffling input rows does not change fit results."""
        df = _make_sw_grid_single(n_bins_per_dim=3, entries_per_bin=30)

        result_orig = _run_sw_fit(df, window_size=1, linear_columns=['x'])

        # Shuffle rows
        df_shuffled = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
        result_shuf = _run_sw_fit(df_shuffled, window_size=1,
                                  linear_columns=['x'])

        # Sort both by bin coordinates for comparison
        sort_cols = ['xBin', 'yBin', 'zBin']
        r1 = result_orig.sort_values(sort_cols).reset_index(drop=True)
        r2 = result_shuf.sort_values(sort_cols).reset_index(drop=True)

        # Integer outputs must match exactly (P1-6)
        for col in ['n_rows_aggregated', 'n_neighbors_used', 'value_entries', 'value_n_fitted']:
            np.testing.assert_array_equal(
                r1[col].values, r2[col].values,
                err_msg=f"Permutation changed integer output: {col}",
            )

        # Float outputs within tight tolerance (same data, same code path)
        for col in ['value_slope_x', 'value_intercept', 'value_rmse',
                     'value_slope_x_err', 'value_intercept_err']:
            np.testing.assert_allclose(
                r1[col].values, r2[col].values,
                atol=1e-10, rtol=1e-10,
                err_msg=f"Permutation changed float output: {col}",
            )

    def test_sw_determinism(self):
        """Test 10: Same input → identical output on repeated runs."""
        df = _make_sw_grid_single(n_bins_per_dim=3, entries_per_bin=20)

        r1 = _run_sw_fit(df, window_size=0, linear_columns=['x'])
        r2 = _run_sw_fit(df, window_size=0, linear_columns=['x'])

        pd.testing.assert_frame_equal(r1, r2)


# #############################################################################
# Tests 11a-d: V1 (numpy) ≡ V2 (Numba) Parity + V2 Analytical Checks
# #############################################################################

# Check if Numba kernel is available
try:
    from groupby_regression_kernels import fit_groups_single_numba as _test_kernel
    _NUMBA_KERNEL_AVAILABLE = True
except ImportError:
    try:
        from ..groupby_regression_kernels import fit_groups_single_numba as _test_kernel
        _NUMBA_KERNEL_AVAILABLE = True
    except ImportError:
        _NUMBA_KERNEL_AVAILABLE = False

_skip_no_numba = pytest.mark.skipif(
    not (_SW_AVAILABLE and _NUMBA_KERNEL_AVAILABLE),
    reason="Sliding window or Numba kernel module not available",
)


def _run_sw_fit_backend(
    df: pd.DataFrame,
    window_size: int,
    linear_columns: list,
    backend: str,
    min_stat: int = 5,
) -> pd.DataFrame:
    """Run make_sliding_window_fit with explicit backend selection."""
    return make_sliding_window_fit(
        df=df,
        gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': window_size, 'yBin': window_size, 'zBin': window_size},
        fit_columns=['value'],
        linear_columns=linear_columns,
        min_stat=min_stat,
        backend=backend,
        suffix='',
    )


@_skip_no_numba
class TestSWNumba:
    """
    V2 (Numba kernel batch) parity and analytical verification.

    Step 4 tolerances (approved by 5/5 reviewers):
        COEFF_ATOL = 1e-6    (V1 vs V2 coefficient match)
        COEFF_RTOL = 1e-4
        ERR_ATOL   = 1e-5    (V1 vs V2 error estimate match)
        ERR_RTOL   = 1e-3
        N_FITTED   = exact   (integer outputs match exactly)
    """

    COEFF_ATOL = 1e-6
    COEFF_RTOL = 1e-4
    ERR_ATOL = 1e-5
    ERR_RTOL = 1e-3

    def test_sw_numba_backend_used(self):
        """Test 11a: backend='numba' actually dispatches to Numba kernel."""
        df = _make_sw_grid_single(n_bins_per_dim=3, entries_per_bin=20)
        result = _run_sw_fit_backend(df, window_size=0,
                                     linear_columns=['x'],
                                     backend='numba')
        assert result.attrs.get('backend_used') == 'numba', (
            f"Expected backend_used='numba', got '{result.attrs.get('backend_used')}'"
        )

    @pytest.mark.parametrize("window_size", [0, 1])
    def test_sw_numba_equals_numpy(self, window_size):
        """Test 11b: V1 (numpy) ≡ V2 (numba) coefficient and error parity."""
        df = _make_sw_grid_single(n_bins_per_dim=4, entries_per_bin=40)

        r_v1 = _run_sw_fit_backend(df, window_size, ['x'],
                                   backend='numpy')
        r_v2 = _run_sw_fit_backend(df, window_size, ['x'],
                                   backend='numba')

        sort_cols = ['xBin', 'yBin', 'zBin']
        r_v1 = r_v1.sort_values(sort_cols).reset_index(drop=True)
        r_v2 = r_v2.sort_values(sort_cols).reset_index(drop=True)

        # Integer outputs must match exactly (P1-6)
        for col in ['n_rows_aggregated', 'n_neighbors_used',
                     'value_entries', 'value_n_fitted']:
            np.testing.assert_array_equal(
                r_v1[col].values, r_v2[col].values,
                err_msg=f"V1 ≠ V2 integer output: {col} (window={window_size})",
            )

        # Coefficient parity
        for col in ['value_slope_x', 'value_intercept']:
            np.testing.assert_allclose(
                r_v1[col].values, r_v2[col].values,
                atol=self.COEFF_ATOL, rtol=self.COEFF_RTOL,
                err_msg=f"V1 ≠ V2 coefficient: {col} (window={window_size})",
            )

        # Error estimate parity
        for col in ['value_slope_x_err', 'value_intercept_err']:
            np.testing.assert_allclose(
                r_v1[col].values, r_v2[col].values,
                atol=self.ERR_ATOL, rtol=self.ERR_RTOL,
                err_msg=f"V1 ≠ V2 error: {col} (window={window_size})",
            )

        # RMSE parity
        np.testing.assert_allclose(
            r_v1['value_rmse'].values, r_v2['value_rmse'].values,
            atol=self.ERR_ATOL, rtol=self.ERR_RTOL,
            err_msg=f"V1 ≠ V2 RMSE (window={window_size})",
        )

    def test_sw_numba_nsigma_recovery(self):
        """Test 11c: V2 (numba) slope recovery within nsigma of truth."""
        df = _make_sw_grid_single()
        result = _run_sw_fit_backend(df, window_size=1,
                                     linear_columns=['x'],
                                     backend='numba')

        valid = result.dropna(subset=['value_slope_x', 'value_slope_x_err'])
        assert len(valid) > 5, "Too few valid V2 bins"

        slopes = valid['value_slope_x'].values
        slope_errs = valid['value_slope_x_err'].values

        good = slope_errs > 0
        pulls = compute_pulls(slopes[good], TRUE_SLOPE, slope_errs[good])

        # Use median n_eff for dof correction
        median_n = int(np.median(valid['n_rows_aggregated'].values))
        check_pull_distribution(
            pulls, nsigma=NSIGMA,
            df=max(median_n - 2, 0),
            label="V2 (numba) slope pull distribution",
        )

    def test_sw_numba_multi_predictor(self):
        """Test 11d: V2 (numba) multi-predictor parity with V1."""
        df = _make_sw_grid_multi()

        r_v1 = make_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin', 'zBin'],
            window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
            fit_columns=['value'], linear_columns=['x1', 'x2'],
            min_stat=5, backend='numpy', suffix='',
        )
        r_v2 = make_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin', 'zBin'],
            window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
            fit_columns=['value'], linear_columns=['x1', 'x2'],
            min_stat=5, backend='numba', suffix='',
        )

        sort_cols = ['xBin', 'yBin', 'zBin']
        r_v1 = r_v1.sort_values(sort_cols).reset_index(drop=True)
        r_v2 = r_v2.sort_values(sort_cols).reset_index(drop=True)

        for col in ['value_slope_x1', 'value_slope_x2', 'value_intercept']:
            np.testing.assert_allclose(
                r_v1[col].values, r_v2[col].values,
                atol=self.COEFF_ATOL, rtol=self.COEFF_RTOL,
                err_msg=f"V1 ≠ V2 multi-predictor: {col}",
            )

        for col in ['value_slope_x1_err', 'value_slope_x2_err',
                     'value_intercept_err']:
            np.testing.assert_allclose(
                r_v1[col].values, r_v2[col].values,
                atol=self.ERR_ATOL, rtol=self.ERR_RTOL,
                err_msg=f"V1 ≠ V2 multi-predictor error: {col}",
            )


# #############################################################################
# Test 12: Multi-Predictor Value Recovery (P1-1, 3 reviewers)
# #############################################################################

@pytest.mark.skipif(not _SW_AVAILABLE, reason="Sliding window module not available")
class TestSWMultiPredictor:
    """
    Check 1 extended to multi-predictor fits.
    value = intercept + slope_x1 × x1 + slope_x2 × x2 + ε

    Catches predictor indexing bugs that single-predictor tests miss.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("statsmodels")

    def test_sw_multi_predictor_nsigma_recovery(self):
        """Test 12: Multi-predictor slopes within nsigma of truth."""
        df = _make_sw_grid_multi()
        result = _run_sw_fit(
            df, window_size=1,
            linear_columns=['x1', 'x2'],
        )

        valid = result.dropna(subset=['value_slope_x1', 'value_slope_x2'])
        assert len(valid) > 0, "No valid multi-predictor fits"

        for _, row in valid.iterrows():
            n_eff = int(row['n_rows_aggregated'])
            if n_eff < 10:
                continue

            # Analytical SE for multi-predictor OLS
            X = df[['x1', 'x2']].values[:n_eff]
            se_multi = compute_ols_se_multi(NOISE_STD, n_eff, X)
            if np.any(np.isnan(se_multi)):
                continue

            # se_multi = [se_intercept, se_x1, se_x2]
            bin_label = f"({row['xBin']},{row['yBin']},{row['zBin']})"

            check_nsigma(
                row['value_intercept'], TRUE_INTERCEPT, se_multi[0],
                nsigma=NSIGMA,
                label=f"multi intercept {bin_label}",
            )
            check_nsigma(
                row['value_slope_x1'], TRUE_SLOPE_X1, se_multi[1],
                nsigma=NSIGMA,
                label=f"multi slope_x1 {bin_label}",
            )
            check_nsigma(
                row['value_slope_x2'], TRUE_SLOPE_X2, se_multi[2],
                nsigma=NSIGMA,
                label=f"multi slope_x2 {bin_label}",
            )


# #############################################################################
# Test 13: Window=0 ≡ Per-Bin OLS Oracle (P1-2, 2 reviewers)
# #############################################################################

@pytest.mark.skipif(not _SW_AVAILABLE, reason="Sliding window module not available")
class TestSWOracleParity:
    """
    Oracle test: SW(window=0) must produce identical results to direct
    per-bin OLS via numpy.linalg.lstsq.

    This bridges "SW is correct" to a well-understood reference
    implementation (numpy) and is extremely cheap to run.
    """

    @pytest.fixture(autouse=True)
    def _setup(self):
        pytest.importorskip("statsmodels")

    def test_sw_window0_equals_per_bin_ols(self):
        """Test 13: SW(window=0) ≡ numpy.lstsq per bin."""
        df = _make_sw_grid_single(n_bins_per_dim=3, entries_per_bin=50)

        result = _run_sw_fit(df, window_size=0, linear_columns=['x'], min_stat=1)

        # Run per-bin OLS via numpy
        for _, row in result.iterrows():
            bin_key = (int(row['xBin']), int(row['yBin']), int(row['zBin']))
            mask = (
                (df['xBin'] == bin_key[0]) &
                (df['yBin'] == bin_key[1]) &
                (df['zBin'] == bin_key[2])
            )
            bin_df = df[mask]

            x = bin_df['x'].values
            y = bin_df['value'].values
            X_design = np.column_stack([np.ones(len(x)), x])
            beta_np, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

            # Coefficients must match closely (same data, different code paths)
            np.testing.assert_allclose(
                row['value_intercept'], beta_np[0],
                atol=1e-6, rtol=1e-6,
                err_msg=f"Intercept mismatch at bin {bin_key}",
            )
            np.testing.assert_allclose(
                row['value_slope_x'], beta_np[1],
                atol=1e-6, rtol=1e-6,
                err_msg=f"Slope mismatch at bin {bin_key}",
            )

            # n_fitted must match exactly (P1-6)
            assert int(row['value_n_fitted']) == len(bin_df), (
                f"n_fitted mismatch at bin {bin_key}: "
                f"SW={int(row['value_n_fitted'])}, expected={len(bin_df)}"
            )


# #############################################################################
# Tests 14–20: V3 Incremental Algorithm Parity & Correctness
# #############################################################################

def _run_sw_fit_incremental(
    df: pd.DataFrame,
    window_size: int,
    linear_columns: list,
    min_stat: int = 5,
) -> pd.DataFrame:
    """Run make_sliding_window_fit with algorithm='incremental' (V3)."""
    return make_sliding_window_fit(
        df=df,
        gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': window_size, 'yBin': window_size, 'zBin': window_size},
        fit_columns=['value'],
        linear_columns=linear_columns,
        min_stat=min_stat,
        algorithm='incremental',
        suffix='',
    )


@pytest.mark.skipif(not _SW_AVAILABLE, reason="Sliding window module not available")
class TestSWV3Parity:
    """
    V3 (incremental, pre-computed XtX/XtY) must match V1 (recompute, lstsq)
    to machine precision for all regression outputs.

    V3 computes mean/std from sufficient statistics so we allow small
    differences in those columns (no median in V3 — set to NaN).

    Invariance checks (A ≡ B):
      14. V3 ≡ V1 coefficients (single predictor, window=0)
      15. V3 ≡ V1 coefficients (single predictor, window=1)
      16. V3 ≡ V1 coefficients (multi predictor, window=1)
      17. V3 ≡ V1 error estimators
      18. V3 ≡ V1 diagnostics (RMSE, R², n_fitted)
    """

    # Tolerances: V3 uses np.linalg.solve vs V1 uses np.linalg.lstsq
    # Both use double precision; differences are floating-point noise.
    COEFF_ATOL = 1e-10
    COEFF_RTOL = 1e-10
    ERR_ATOL = 1e-10
    ERR_RTOL = 1e-10
    DIAG_ATOL = 1e-10
    DIAG_RTOL = 1e-10
    # Stats from sufficient stats vs row-level: small differences
    STAT_ATOL = 1e-10
    STAT_RTOL = 1e-10

    @pytest.mark.parametrize("window_size", [0, 1])
    def test_v3_slope_matches_v1(self, window_size):
        """Test 14: V3 slope ≡ V1 slope (single predictor)."""
        df = _make_sw_grid_single()
        r_v1 = _run_sw_fit(df, window_size, ['x'])
        r_v3 = _run_sw_fit_incremental(df, window_size, ['x'])

        sort_cols = ['xBin', 'yBin', 'zBin']
        r_v1 = r_v1.sort_values(sort_cols).reset_index(drop=True)
        r_v3 = r_v3.sort_values(sort_cols).reset_index(drop=True)

        np.testing.assert_allclose(
            r_v1['value_slope_x'].values,
            r_v3['value_slope_x'].values,
            atol=self.COEFF_ATOL, rtol=self.COEFF_RTOL,
            err_msg=f"V3 ≠ V1 slope (window={window_size})",
        )
        np.testing.assert_allclose(
            r_v1['value_intercept'].values,
            r_v3['value_intercept'].values,
            atol=self.COEFF_ATOL, rtol=self.COEFF_RTOL,
            err_msg=f"V3 ≠ V1 intercept (window={window_size})",
        )

    def test_v3_multi_predictor_matches_v1(self):
        """Test 15: V3 ≡ V1 multi-predictor coefficients (window=1)."""
        df = _make_sw_grid_multi()
        r_v1 = make_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin', 'zBin'],
            window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
            fit_columns=['value'], linear_columns=['x1', 'x2'],
            min_stat=5, algorithm='recompute', suffix='',
        )
        r_v3 = make_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin', 'zBin'],
            window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
            fit_columns=['value'], linear_columns=['x1', 'x2'],
            min_stat=5, algorithm='incremental', suffix='',
        )

        sort_cols = ['xBin', 'yBin', 'zBin']
        r_v1 = r_v1.sort_values(sort_cols).reset_index(drop=True)
        r_v3 = r_v3.sort_values(sort_cols).reset_index(drop=True)

        for col in ['value_slope_x1', 'value_slope_x2', 'value_intercept']:
            np.testing.assert_allclose(
                r_v1[col].values, r_v3[col].values,
                atol=self.COEFF_ATOL, rtol=self.COEFF_RTOL,
                err_msg=f"V3 ≠ V1: {col}",
            )

    @pytest.mark.parametrize("window_size", [0, 1])
    def test_v3_errors_match_v1(self, window_size):
        """Test 16: V3 standard errors ≡ V1 standard errors."""
        df = _make_sw_grid_single()
        r_v1 = _run_sw_fit(df, window_size, ['x'])
        r_v3 = _run_sw_fit_incremental(df, window_size, ['x'])

        sort_cols = ['xBin', 'yBin', 'zBin']
        r_v1 = r_v1.sort_values(sort_cols).reset_index(drop=True)
        r_v3 = r_v3.sort_values(sort_cols).reset_index(drop=True)

        for col in ['value_slope_x_err', 'value_intercept_err']:
            np.testing.assert_allclose(
                r_v1[col].values, r_v3[col].values,
                atol=self.ERR_ATOL, rtol=self.ERR_RTOL,
                err_msg=f"V3 ≠ V1 error: {col} (window={window_size})",
            )

    @pytest.mark.parametrize("window_size", [0, 1])
    def test_v3_diagnostics_match_v1(self, window_size):
        """Test 17: V3 RMSE, R², n_fitted ≡ V1."""
        df = _make_sw_grid_single()
        r_v1 = _run_sw_fit(df, window_size, ['x'])
        r_v3 = _run_sw_fit_incremental(df, window_size, ['x'])

        sort_cols = ['xBin', 'yBin', 'zBin']
        r_v1 = r_v1.sort_values(sort_cols).reset_index(drop=True)
        r_v3 = r_v3.sort_values(sort_cols).reset_index(drop=True)

        np.testing.assert_allclose(
            r_v1['value_rmse'].values,
            r_v3['value_rmse'].values,
            atol=self.DIAG_ATOL, rtol=self.DIAG_RTOL,
            err_msg=f"V3 ≠ V1 RMSE (window={window_size})",
        )
        np.testing.assert_allclose(
            r_v1['value_r_squared'].values,
            r_v3['value_r_squared'].values,
            atol=self.DIAG_ATOL, rtol=self.DIAG_RTOL,
            err_msg=f"V3 ≠ V1 R² (window={window_size})",
        )
        np.testing.assert_array_equal(
            r_v1['value_n_fitted'].values,
            r_v3['value_n_fitted'].values,
            err_msg=f"V3 ≠ V1 n_fitted (window={window_size})",
        )

    def test_v3_stats_from_sufficient(self):
        """Test 18: V3 mean/std match V1 (from sufficient stats)."""
        df = _make_sw_grid_single()
        r_v1 = _run_sw_fit(df, window_size=1, linear_columns=['x'])
        r_v3 = _run_sw_fit_incremental(df, window_size=1, linear_columns=['x'])

        sort_cols = ['xBin', 'yBin', 'zBin']
        r_v1 = r_v1.sort_values(sort_cols).reset_index(drop=True)
        r_v3 = r_v3.sort_values(sort_cols).reset_index(drop=True)

        np.testing.assert_allclose(
            r_v1['value_mean'].values,
            r_v3['value_mean'].values,
            atol=self.STAT_ATOL, rtol=self.STAT_RTOL,
            err_msg="V3 ≠ V1 mean",
        )
        np.testing.assert_allclose(
            r_v1['value_std'].values,
            r_v3['value_std'].values,
            atol=self.STAT_ATOL, rtol=self.STAT_RTOL,
            err_msg="V3 ≠ V1 std",
        )
        np.testing.assert_array_equal(
            r_v1['value_entries'].values,
            r_v3['value_entries'].values,
            err_msg="V3 ≠ V1 entries",
        )

    def test_v3_metadata_algorithm(self):
        """Test 19: V3 metadata reports algorithm='incremental'."""
        df = _make_sw_grid_single(n_bins_per_dim=2, entries_per_bin=20)
        r_v3 = _run_sw_fit_incremental(df, window_size=1, linear_columns=['x'])
        assert r_v3.attrs.get('algorithm') == 'incremental'
        assert 'incremental' in r_v3.attrs.get('backend_used', '')

    def test_v3_nsigma_recovery(self):
        """Test 20: V3 independently recovers true slope within nsigma."""
        df = _make_sw_grid_single()
        r_v3 = _run_sw_fit_incremental(df, window_size=1, linear_columns=['x'])

        valid = r_v3.dropna(subset=['value_slope_x', 'value_slope_x_err'])
        slopes = valid['value_slope_x'].values
        slope_errs = valid['value_slope_x_err'].values

        good = slope_errs > 0
        pulls = compute_pulls(slopes[good], TRUE_SLOPE, slope_errs[good])

        median_n = int(np.median(valid['value_n_fitted'].values))
        check_pull_distribution(
            pulls, nsigma=NSIGMA,
            df=max(median_n - 2, 0),
            label="V3 incremental slope pull distribution",
        )
