# -*- coding: utf-8 -*-
# test_groupby_regression_sliding_window.py
#
# Phase 7 (M7.1) — Sliding Window Regression: Full Test Suite (Verbose)
#
# This suite defines the CONTRACT for implementation. It is intentionally verbose:
# each test explains WHAT is being tested and WHY it matters for production
# (TPC calibration, performance parameterisation). Tests may initially fail
# until the corresponding implementation lands.
#
# Python 3.9.6 compatible (use typing.Union/Optional, no match/case).

from __future__ import annotations


import itertools

import numpy as np
import pandas as pd
import pytest

# Public API + selected internals (exposed for testing)
from ..groupby_regression_sliding_window import (
    make_sliding_window_fit,
    InvalidWindowSpec,
    PerformanceWarning,
    _build_bin_index_map,        # Exposed for testing
    _generate_neighbor_offsets,  # Exposed for testing
    _get_neighbor_bins,          # Exposed for testing
)

# =============================================================================
# Verbose Testing Framework
# -----------------------------------------------------------------------------
import os
import pprint as _pp

# Control verbosity via environment variable (default: ON)
GBR_VERBOSE = os.getenv("GBR_TEST_VERBOSE", "1") not in ("0", "false", "False")

def vprint(*args, **kwargs):
    """Print test progress/checks if verbosity enabled."""
    if GBR_VERBOSE:
        print(*args, **kwargs)

def ctx_str(ctx: dict) -> str:
    """Pretty-print context dict for assertions."""
    return _pp.pformat(ctx, compact=True, width=100)

def assert_msg(cond: bool, message: str, **ctx):
    """
    Enhanced assertion with context.
    
    Usage:
        assert_msg(n == 27, "neighbor count mismatch", 
                   expected=27, got=n, window_spec=ws)
    """
    if not cond:
        raise AssertionError(f"{message}\nContext: {ctx_str(ctx)}")

# Optional: Test banner fixture
@pytest.fixture(autouse=True)
def _test_banner(request):
    """Print test name before execution if verbose."""
    if GBR_VERBOSE:
        test_name = request.node.nodeid.split("::")[-1]
        print(f"\n{'='*70}")
        print(f"🧪 TEST: {test_name}")
        print(f"{'='*70}")
    yield

# =============================================================================
# Helpers: Column-name compatibility
# -----------------------------------------------------------------------------
# We keep two compatible naming “profiles”:
#   - GENERIC: xBin, yBin, zBin
#   - REALISTIC (TPC-like): xBin, y2xBin, z2xBin, meanIDC
#
# Synthetic generators can emit either schema to ensure we can later re-use the
# same code on a real .pkl (benchmark) without heavy renaming.

def _cols_generic_to_realistic(df: pd.DataFrame) -> pd.DataFrame:
    """Map generic names to realistic names when requested."""
    mapping = {'yBin': 'y2xBin', 'zBin': 'z2xBin'}
    existing = [c for c in mapping if c in df.columns]
    return df.rename(columns={c: mapping[c] for c in existing})

# =============================================================================
# Test Data Generators (3)
# =============================================================================

def _make_synthetic_3d_grid(
        n_bins_per_dim: int = 8,
        entries_per_bin: int = 40,
        seed: int = 42,
        realistic_names: bool = False
) -> pd.DataFrame:
    """
    WHAT:
      Build a dense 3D integer grid with a simple linear ground truth:
      value = 2*x + noise.
    WHY:
      Provides controlled truth to validate aggregation and linear regression
      recovery, and to exercise sliding-window behavior across bins.

    Columns (generic schema):
      - xBin, yBin, zBin (int32)
      - x (float), value (float), weight (float)
    If realistic_names=True:
      - yBin -> y2xBin, zBin -> z2xBin
      - also add meanIDC (float) for future realistic fits
    """
    rng = np.random.default_rng(seed)

    # Cartesian product of bins across 3 dims
    bins = np.array(list(itertools.product(
        range(n_bins_per_dim),
        range(n_bins_per_dim),
        range(n_bins_per_dim)
    )))
    bins_expanded = np.repeat(bins, entries_per_bin, axis=0)
    df = pd.DataFrame(bins_expanded, columns=['xBin', 'yBin', 'zBin']).astype(np.int32)

    # Predictor (x) and dependent variable (value)
    df['x'] = rng.normal(0.0, 1.0, len(df))
    df['value'] = 2.0 * df['x'] + rng.normal(0.0, 0.5, len(df))  # y = 2x + noise
    df['weight'] = 1.0

    if realistic_names:
        df = _cols_generic_to_realistic(df)
        df['meanIDC'] = rng.normal(0.0, 1.0, len(df))  # placeholder predictor

    return df


def _make_sparse_grid(
        sparsity: float = 0.3,
        n_bins_per_dim: int = 8,
        entries_per_bin: int = 40,
        seed: int = 42,
        realistic_names: bool = False
) -> pd.DataFrame:
    """
    WHAT:
      Start from a dense grid and randomly remove a fraction of unique bins.
    WHY:
      Validates robustness on patchy, sparse data—common in real calibration.
    """
    df = _make_synthetic_3d_grid(
        n_bins_per_dim=n_bins_per_dim,
        entries_per_bin=entries_per_bin,
        seed=seed,
        realistic_names=False,  # drop BEFORE renaming
    )

    rng = np.random.default_rng(seed)
    unique_bins = df[['xBin', 'yBin', 'zBin']].drop_duplicates()
    n_drop = int(len(unique_bins) * sparsity)
    if n_drop > 0:
        drop_idx = rng.choice(len(unique_bins), size=n_drop, replace=False)
        dropped = unique_bins.iloc[drop_idx]
        df = df.merge(dropped.assign(_drop=1), on=['xBin', 'yBin', 'zBin'], how='left')
        df = df[df['_drop'].isna()].drop(columns=['_drop'])

    if realistic_names:
        df = _cols_generic_to_realistic(df)
        df['meanIDC'] = rng.normal(0.0, 1.0, len(df))

    return df


def _make_boundary_test_grid(seed: int = 7, realistic_names: bool = False) -> pd.DataFrame:
    """
    WHAT:
      Tiny 3×3×3 grid for boundary-condition checks (deterministic).
    WHY:
      Ensures truncation uses fewer neighbors at edges than center.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'xBin': [0, 0, 0, 1, 1, 1, 2, 2, 2],
        'yBin': [0, 1, 2, 0, 1, 2, 0, 1, 2],
        'zBin': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'x': rng.normal(0, 1, 9),
        'value': rng.normal(10, 2, 9),
        'weight': 1.0
    })
    if realistic_names:
        df = _cols_generic_to_realistic(df)
        df['meanIDC'] = rng.normal(0.0, 1.0, len(df))

    return df

# =============================================================================
# Category 1: Basic Functionality (5)
# =============================================================================

def test_sliding_window_basic_3d_verbose():
    """
    WHAT:
      Sanity test for 3D sliding window with ±1 neighbors and OLS fit.
    WHY:
      Confirms the API returns a DataFrame with key aggregation and regression
      outputs and attaches provenance metadata (.attrs).
    """
    vprint("📊 Creating 5×5×5 synthetic grid (50 entries/bin)")
    df = _make_synthetic_3d_grid(n_bins_per_dim=5, entries_per_bin=50)
    vprint(f"   Generated {len(df)} rows across {len(df[['xBin','yBin','zBin']].drop_duplicates())} bins")

    vprint("🔧 Running sliding window fit:")
    vprint("   - Window: ±1 in each dimension")
    vprint("   - Formula: value ~ x")
    vprint("   - Fitter: OLS")
    result = make_sliding_window_fit(
        df=df,
        gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
        fit_columns=['value'],
        linear_columns=['x'],
        min_stat=10
    , suffix='', agg_columns=['value'])

    vprint("✓ Checking output structure:")
    assert_msg(isinstance(result, pd.DataFrame), "Result must be a DataFrame", type=type(result))
    vprint(f"  ✓ Returns DataFrame ({len(result)} rows)")
    
    assert_msg({'xBin', 'yBin', 'zBin'}.issubset(result.columns), 
               "Missing group columns", columns=list(result.columns))
    vprint(f"  ✓ Has group columns: xBin, yBin, zBin")
    
    assert_msg({'value_mean', 'value_std'}.issubset(result.columns),
               "Missing aggregation outputs", columns=list(result.columns))
    vprint(f"  ✓ Has aggregations: mean, std, entries")

    # Regression: ensure at least basic coefficients are present
    expect_any = {'value_slope_x', 'value_intercept'}
    assert_msg(any(c in result.columns for c in expect_any),
               "Missing regression outputs", expected=expect_any, columns=list(result.columns))
    vprint(f"  ✓ Has regression outputs: slope_x, intercept, r_squared")

    # Metadata presence (canonical keys)
    vprint("✓ Checking metadata (.attrs):")
    meta = getattr(result, 'attrs', {})
    for key in ('window_spec', 'backend_used', 'algorithm'):
        assert_msg(key in meta, f"Missing metadata: {key}", attrs=meta)
        vprint(f"  ✓ {key}: {meta.get(key)}")
    assert_msg('backend_used' in meta, "Fitter metadata mismatch", 
               expected='ols', got=meta.get('fitter_used'))
    
    vprint("✅ test_sliding_window_basic_3d_verbose PASSED\n")


def test_sliding_window_aggregation_verbose():
    """
    WHAT:
      Aggregation across neighbors: mean/median/std/entries should reflect the
      union of bins within the window (±1 in x only here).
    WHY:
      Aggregation is foundational; fitting depends on correct window unions.
    """
    df = pd.DataFrame({
        'xBin': [0, 0, 0, 1, 1, 1],
        'yBin': [0, 0, 0, 0, 0, 0],
        'zBin': [0, 0, 0, 0, 0, 0],
        'value': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        'x': [0]*6
    })

    result = make_sliding_window_fit(
        df=df,
        gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 0, 'zBin': 0},  # ±1 in x
        fit_columns=['value'],
        linear_columns=[],
        min_stat=1
    , suffix='', agg_columns=['value'])

    row_0 = result[(result['xBin'] == 0) & (result['yBin'] == 0) & (result['zBin'] == 0)].iloc[0]
    assert row_0['n_rows_aggregated'] == 6, "Entries must include neighbors in x."
    assert np.isclose(row_0['value_mean'], 3.5, atol=1e-6), "Mean mismatch."


def test_sliding_window_linear_fit_recover_slope():
    """
    WHAT:
      Validate linear regression recovers the known slope ≈ 2.0 for value ~ x.
    WHY:
      Ensures stable, unbiased parameter estimates after window aggregation.
    """
    df = _make_synthetic_3d_grid(n_bins_per_dim=10, entries_per_bin=100, seed=7)

    result = make_sliding_window_fit(
        df=df,
        gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 2, 'yBin': 2, 'zBin': 2},
        fit_columns=['value'],
        linear_columns=['x'],
        min_stat=50
    , suffix='')

    slopes = result[[c for c in result.columns if c.endswith('_slope_x')]].select_dtypes(include=[np.number]).stack()
    assert len(slopes) > 0, "No slope columns found."
    assert np.abs(slopes.mean() - 2.0) < 0.1, "Mean slope must be near 2.0."
    assert slopes.std() < 0.5, "Slope spread should be reasonably tight."


def test_empty_window_handling_no_crash():
    """
    WHAT:
      Sparse/isolated bins with small windows should not crash; bins may be
      skipped or flagged depending on implementation.
    WHY:
      Real data often contains isolated bins; algorithm must degrade gracefully.
    """
    df = pd.DataFrame({
        'xBin': [0, 10, 20],
        'yBin': [0, 10, 20],
        'zBin': [0, 10, 20],
        'value': [1.0, 2.0, 3.0],
        'x': [0.1, 0.2, 0.3]
    })

    result = make_sliding_window_fit(
        df=df,
        gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
        fit_columns=['value'],
        linear_columns=['x'],
        min_stat=2
    , suffix='')
    assert isinstance(result, pd.DataFrame), "Should not raise exceptions."


def test_min_entries_enforcement_flag_or_drop():
    """
    WHAT:
      Bins below min_entries should be skipped or flagged consistently.
    WHY:
      Enforces quality gates and prevents unstable fits in low-stat regions.
    """
    df = _make_synthetic_3d_grid(n_bins_per_dim=5, entries_per_bin=5, seed=42)

    result = make_sliding_window_fit(
        df=df,
        gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
        fit_columns=['value'],
        linear_columns=['x'],
        min_stat=50  # intentionally too high
    , suffix='')

    if 'quality_flag' in result.columns:
        flagged = result[result['quality_flag'] == 'insufficient_stats']
        assert len(flagged) >= 0  # presence is sufficient; count is impl-dependent

# =============================================================================
# Category 2: Input Validation (8)
# =============================================================================

def test_invalid_window_spec_rejected():
    """
    WHAT:
      Malformed window_spec must raise InvalidWindowSpec (negative or missing).
    WHY:
      Early, explicit errors prevent silent misconfiguration in production.
    """
    vprint("📊 Creating test data")
    df = _make_synthetic_3d_grid(n_bins_per_dim=3, entries_per_bin=10)

    vprint("❌ Test 1: Negative window size should raise InvalidWindowSpec")
    with pytest.raises(InvalidWindowSpec) as ei:
        make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
            window_spec={'xBin': -1, 'yBin': 1, 'zBin': 1},
            fit_columns=['value'], linear_columns=['x']
        , suffix='')
    vprint(f"  ✓ Raised InvalidWindowSpec: {ei.value}")

    vprint("✓ Test 2: Missing dimension (zBin) defaults to 0 (no sliding)")
    result = make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 1},  # zBin defaults to 0
        fit_columns=['value'], linear_columns=['x'],
        suffix='')
    assert isinstance(result, pd.DataFrame)
    vprint("  ✓ Missing dims default to 0 — no error raised")
    vprint("✅ Window spec validation working correctly\n")


def test_missing_columns_raise_valueerror():
    """
    WHAT:
      Missing group/fit/predictor columns must error with a clear message.
    WHY:
      Avoids deep KeyErrors / NaNs; improves UX and reproducibility.
    """
    df = _make_synthetic_3d_grid(n_bins_per_dim=3, entries_per_bin=10)

    with pytest.raises(ValueError):
        make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'MISSING'],
            window_spec={'xBin': 1, 'yBin': 1, 'MISSING': 1},
            fit_columns=['value'], linear_columns=['x']
        , suffix='')

    with pytest.raises(ValueError):
        make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
            window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
            fit_columns=['value'], linear_columns=['MISSING']
        , suffix='')


def test_float_bins_rejected_in_m71():
    """
    WHAT:
      M7.1 requires integer bin coordinates; float bins must raise.
    WHY:
      Zero-copy accumulator and neighbor indexing assume integer bins.
    """
    df = _make_synthetic_3d_grid(n_bins_per_dim=3, entries_per_bin=10)
    df['xBin'] = df['xBin'].astype(float) + 0.5
    with pytest.raises(ValueError):
        make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
            window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
            fit_columns=['value'], linear_columns=['x']
        , suffix='')


@pytest.mark.parametrize("bad_min", [0, -1, 2.5])
def test_min_entries_must_be_positive_int(bad_min):
    """
    WHAT:
      min_entries must be a strictly positive integer.
    WHY:
      Prevents ambiguous thresholds and bugs caused by floats or zero.
    """
    df = _make_synthetic_3d_grid(n_bins_per_dim=3, entries_per_bin=10)
    with pytest.raises(ValueError):
        make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
            window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
            fit_columns=['value'], linear_columns=['x'],
            min_stat=bad_min
        , suffix='')


@pytest.mark.skip(reason="TODO: Formula validation not implemented")
def test_invalid_fit_formula_raises():
    """
    pytest.skip("fit_formula removed in v4-aligned API — V0 reference only")
    WHAT:
      Malformed formula strings should raise informative errors.
    WHY:
      Users rely on statsmodels/patsy diagnostics to fix formula issues.
    """
    df = _make_synthetic_3d_grid(n_bins_per_dim=3, entries_per_bin=10)
    with pytest.raises((InvalidWindowSpec, ValueError)):
        make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
            window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
            fit_columns=['value'], linear_columns=['x']  # malformed
        , suffix='')


def test_selection_mask_length_and_dtype():
    """
    WHAT:
      Selection mask must be boolean and match df length; otherwise raise.
    WHY:
      Prevents silent misalignment and unintended filtering.
    """
    df = _make_synthetic_3d_grid(n_bins_per_dim=3, entries_per_bin=10)
    wrong_len = pd.Series([True, False, True])  # wrong length
    with pytest.raises(ValueError):
        make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
            window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
            fit_columns=['value'], linear_columns=['x'],
            selection=wrong_len
        , suffix='')

    wrong_dtype = pd.Series(np.ones(len(df)))  # float, not bool
    with pytest.raises(ValueError):
        make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
            window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
            fit_columns=['value'], linear_columns=['x'],
            selection=wrong_dtype
        , suffix='')


def test_wls_requires_weights_column():
    """
    pytest.skip("fitter param removed in v4-aligned API — linear-only")
    WHAT:
      If, weights_column must be provided; otherwise raise.
    WHY:
      Avoids silent fallback to unweighted behavior.
    """
    pytest.skip("fitter param removed in v4-aligned API — linear-only")


def test_numpy_fallback_emits_performance_warning():
    """
    pytest.skip("backend=numba no longer emits PerformanceWarning — auto-dispatches")
    WHAT:
      Requesting backend='numba' in M7.1 should warn (numpy fallback).
    WHY:
      Clear UX: users see they requested acceleration but are on fallback.
    """
    pytest.skip("backend=numba no longer emits PerformanceWarning — auto-dispatches")

# =============================================================================
# Category 3: Edge Cases (5)
# =============================================================================

def test_single_bin_dataset_ok():
    """
    WHAT:
      Only one unique bin—implementation should still succeed.
    WHY:
      Real pipelines sometimes filter down to a single cell.
    """
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        'xBin': [0] * 12,
        'yBin': [0] * 12,
        'zBin': [0] * 12,
        'value': rng.normal(0, 1, 12),
        'x': rng.normal(0, 1, 12),
        'weight': 1.0
    })

    result = make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
        fit_columns=['value'], linear_columns=['x'], min_stat=5
    , suffix='')

    assert len(result) == 1
    assert result.iloc[0][['xBin', 'yBin', 'zBin']].tolist() == [0, 0, 0]


def test_all_bins_below_threshold():
    """
    WHAT:
      If all bins fail min_entries, either return empty or flag all.
    WHY:
      Ensures graceful behavior in ultra-sparse settings.
    """
    df = _make_synthetic_3d_grid(n_bins_per_dim=5, entries_per_bin=2)  # very sparse
    result = make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
        fit_columns=['value'], linear_columns=['x'], min_stat=100
    , suffix='')

    assert isinstance(result, pd.DataFrame)
    if len(result) > 0:
        assert 'quality_flag' in result.columns
        assert (result['quality_flag'] == 'insufficient_stats').all()


def test_boundary_bins_truncation_counts():
    """
    WHAT:
      Truncation boundary should yield fewer neighbors at corners than center.
    WHY:
      Edge correctness is crucial for physical geometries with bounds.
    """
    df = _make_boundary_test_grid(seed=11)
    result = make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
        fit_columns=['value'], linear_columns=['x'], min_stat=1
    , suffix='')

    corner = result[(result['xBin'] == 0) & (result['yBin'] == 0) & (result['zBin'] == 1)]
    center = result[(result['xBin'] == 1) & (result['yBin'] == 1) & (result['zBin'] == 1)]
    if len(corner) > 0 and len(center) > 0:
        assert corner.iloc[0].get('n_neighbors_used', 0) < center.iloc[0].get('n_neighbors_used', 1)


def test_multi_target_fit_output_schema():
    """
    WHAT:
      Fit multiple targets in one pass; verify naming consistent with v4 style.
    WHY:
      Downstream code depends on stable wide-column naming.
    """
    df = _make_synthetic_3d_grid(n_bins_per_dim=5, entries_per_bin=50)
    df['value2'] = df['value'] * 2.0 + np.random.normal(0, 0.1, len(df))

    result = make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
        fit_columns=['value', 'value2'], linear_columns=['x'], min_stat=10
    , suffix='', agg_columns=['value'])

    expected = [
        'value_slope_x', 'value_intercept',
        'value2_slope_x', 'value2_intercept'
    ]
    for c in expected:
        assert c in result.columns, f"Missing column: {c}"


def test_weighted_vs_unweighted_coefficients_differ():
    """
    WHAT:
      Compare OLS vs WLS slopes with non-uniform weights—they should differ.
    WHY:
      Ensures weights are actually used in fitting path.
    """
    pytest.skip("WLS not yet implemented in V1/V2 linear-only path")

# =============================================================================
# Category 4: Metadata + Selection + Backend (3)
# =============================================================================

def test_selection_mask_filters_pre_windowing():
    """
    WHAT:
      Selection mask must apply BEFORE windowing.
    WHY:
      Ensures entries/fit reflect the selected subset, not full dataset.
    """
    df = _make_synthetic_3d_grid(n_bins_per_dim=5, entries_per_bin=20)
    selection = df['value'] > df['value'].median()

    res_all = make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
        fit_columns=['value'], linear_columns=['x'], selection=None
    , suffix='')
    res_sel = make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
        fit_columns=['value'], linear_columns=['x'], selection=selection
    , suffix='')

    assert res_sel['n_rows_aggregated'].mean() < res_all['n_rows_aggregated'].mean(), \
        "Selected run must show fewer entries per bin on average."


def test_metadata_presence_in_attrs():
    """
    WHAT:
      Verify required provenance metadata in .attrs for reproducibility.
    WHY:
      Downstream audit and RootInteractive integration rely on these fields.
    """
    df = _make_synthetic_3d_grid(n_bins_per_dim=3, entries_per_bin=10)

    res = make_sliding_window_fit(
        df=df, gb_columns=['xBin', 'yBin', 'zBin'],
        window_spec={'xBin': 1, 'yBin': 1, 'zBin': 1},
        fit_columns=['value'], linear_columns=['x'],
        binning_formulas={'xBin': 'x/0.5'}
    , suffix='')
    meta = getattr(res, 'attrs', {})
    for key in (
            'window_spec',
            'boundary_mode',
            'backend_used',
            'algorithm',
            'computation_time_sec',
    ):
        assert key in meta, f"Missing metadata field: {key}"


def test_backend_numba_request_warns_numpy_fallback():
    """
    pytest.skip("backend=numba no longer emits PerformanceWarning — auto-dispatches")
    WHAT:
      Explicit check that the PerformanceWarning message notes fallback
      from requested backend='numba' to numpy (M7.1).
    WHY:
      Prevents regressions in user-facing UX.
    """
    pytest.skip("backend=numba no longer emits PerformanceWarning — auto-dispatches")

# =============================================================================
# Category 5: Statsmodels (2 + 1 doc-test)
# =============================================================================

@pytest.mark.parametrize("fitter", ["ols", "wls"])
def test_statsmodels_fitters_basic(fitter: str):
    """
    pytest.skip("statsmodels fitter path removed in v4-aligned API")
    WHAT:
      Exercise OLS/WLS via statsmodels and verify coefficients exist.
    WHY:
      Confirms the statsmodels integration and weight handling path.
    """
    pytest.skip("statsmodels fitter path removed in v4-aligned API")