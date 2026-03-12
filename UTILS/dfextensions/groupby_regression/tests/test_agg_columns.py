"""
Tests for agg_columns extension (Phase 13.9.GB Extension).

6 tests covering:
  1. Basic output columns
  2. Manual verification of mean/std
  3. Kernel-weighted vs unweighted
  4. Median optional flag
  5. Backward compatibility (agg_columns=None)
  6. V5 vs zerocopy path equivalence
"""
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from groupby_regression_sliding_window import make_sliding_window_fit


# ── Shared test data fixture ──

@pytest.fixture
def sample_df():
    """3D grid (xBin×yBin) with known values for validation."""
    rng = np.random.default_rng(42)
    n_per_bin = 50
    rows = []
    for xb in range(5):
        for yb in range(4):
            for _ in range(n_per_bin):
                rows.append({
                    'xBin': xb,
                    'yBin': yb,
                    'target': 1.0 + 0.5 * xb + rng.normal(0, 0.1),
                    'predictor': rng.normal(0, 1),
                    'extra_float': 10.0 * xb + yb + rng.normal(0, 0.5),
                    'coord_x': xb + rng.uniform(-0.3, 0.3),
                    'coord_y': yb + rng.uniform(-0.3, 0.3),
                })
    return pd.DataFrame(rows)


def _base_kwargs(sample_df, **overrides):
    """Common arguments for make_sliding_window_fit."""
    kw = dict(
        df=sample_df,
        gb_columns=['xBin', 'yBin'],
        fit_columns=['target'],
        linear_columns=['predictor'],
        window_spec={'xBin': 1},
        suffix='_sw',
        min_stat=5,
        algorithm='recompute',
        backend='numpy',
    )
    kw.update(overrides)
    return kw


# ── Test 1: Basic output columns ──

def test_agg_columns_basic(sample_df):
    """agg_columns=['extra_float'] produces extra_float_mean_sw, extra_float_std_sw."""
    result = make_sliding_window_fit(
        **_base_kwargs(sample_df, agg_columns=['extra_float', 'coord_x']))

    # Check columns exist
    assert 'extra_float_mean_sw' in result.columns
    assert 'extra_float_std_sw' in result.columns
    assert 'coord_x_mean_sw' in result.columns
    assert 'coord_x_std_sw' in result.columns

    # No median columns (agg_median=False by default)
    assert 'extra_float_median_sw' not in result.columns
    assert 'coord_x_median_sw' not in result.columns

    # Values are finite for all bins
    assert result['extra_float_mean_sw'].notna().all()
    assert result['extra_float_std_sw'].notna().all()
    assert result['coord_x_mean_sw'].notna().all()

    # Sanity: extra_float ~ 10*xBin + yBin, so mean should correlate with xBin
    merged = result.merge(
        result.groupby('xBin')['extra_float_mean_sw'].mean().reset_index(name='avg_ef'),
        on='xBin')
    assert merged['avg_ef'].is_monotonic_increasing or merged['avg_ef'].corr(merged['xBin']) > 0.9


# ── Test 2: Manual verification of mean/std ──

def test_agg_columns_matches_manual(sample_df):
    """agg mean/std matches manual groupby+window calculation for window=0 (no SW)."""
    # With window_spec={'xBin': 0, 'yBin': 0}, each bin aggregates only its own rows
    result = make_sliding_window_fit(
        **_base_kwargs(sample_df,
                       window_spec={'xBin': 0, 'yBin': 0},
                       agg_columns=['extra_float', 'coord_x']))

    # Manual calculation per bin
    manual = sample_df.groupby(['xBin', 'yBin']).agg(
        ef_mean=('extra_float', 'mean'),
        ef_std=('extra_float', lambda x: x.std(ddof=1)),
        cx_mean=('coord_x', 'mean'),
        cx_std=('coord_x', lambda x: x.std(ddof=1)),
    ).reset_index()

    merged = result.merge(manual, on=['xBin', 'yBin'])

    np.testing.assert_allclose(
        merged['extra_float_mean_sw'].values, merged['ef_mean'].values,
        rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        merged['extra_float_std_sw'].values, merged['ef_std'].values,
        rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        merged['coord_x_mean_sw'].values, merged['cx_mean'].values,
        rtol=1e-10, atol=1e-12)


# ── Test 3: Kernel-weighted vs unweighted ──

def test_agg_columns_with_kernel_weights(sample_df):
    """Gaussian kernel produces different agg mean than uniform kernel
    in the recompute (zerocopy) path where user weights affect aggregation.

    NOTE: The V3 incremental path computes agg_columns from sufficient stats
    which are always unweighted (per P1-6 review decision: COG is a data
    property, not a fit property). This test uses algorithm='recompute'
    to verify that user-provided weights do affect agg_columns stats.
    """
    # Add a weight column that varies with xBin to create a visible effect
    df = sample_df.copy()
    df['w_by_x'] = 1.0 / (1.0 + df['xBin'].values.astype(float))

    kw = _base_kwargs(df,
                      window_spec={'xBin': 1},
                      agg_columns=['extra_float'],
                      algorithm='recompute',
                      backend='numpy')

    result_no_w = make_sliding_window_fit(**kw, weights=None)
    result_w = make_sliding_window_fit(**kw, weights='w_by_x')

    # For interior bins (xBin in [1,3]), weighted mean should differ from unweighted
    interior = (result_no_w['xBin'] >= 1) & (result_no_w['xBin'] <= 3)
    mean_no_w = result_no_w.loc[interior, 'extra_float_mean_sw'].values
    mean_w = result_w.loc[interior, 'extra_float_mean_sw'].values

    # They should NOT be identical (weights give more weight to low xBin neighbors)
    assert not np.allclose(mean_no_w, mean_w, atol=1e-6), \
        "Weighted and unweighted agg means should differ for heterogeneous weights"

    # Both should be finite
    assert np.all(np.isfinite(mean_no_w))
    assert np.all(np.isfinite(mean_w))


# ── Test 4: Median optional flag ──

def test_agg_columns_median_optional(sample_df):
    """agg_median=False → no median; True → median present and correct."""
    result_no = make_sliding_window_fit(
        **_base_kwargs(sample_df, agg_columns=['extra_float'], agg_median=False))
    result_yes = make_sliding_window_fit(
        **_base_kwargs(sample_df, agg_columns=['extra_float'], agg_median=True))

    # No median when disabled
    assert 'extra_float_median_sw' not in result_no.columns

    # Median present when enabled
    assert 'extra_float_median_sw' in result_yes.columns
    assert result_yes['extra_float_median_sw'].notna().all()

    # For window=0, median should match manual
    result_w0 = make_sliding_window_fit(
        **_base_kwargs(sample_df,
                       window_spec={'xBin': 0, 'yBin': 0},
                       agg_columns=['extra_float'],
                       agg_median=True))

    manual_med = sample_df.groupby(['xBin', 'yBin'])['extra_float'].median().reset_index(
        name='ef_median')
    merged = result_w0.merge(manual_med, on=['xBin', 'yBin'])
    np.testing.assert_allclose(
        merged['extra_float_median_sw'].values, merged['ef_median'].values,
        rtol=1e-10, atol=1e-12)


# ── Test 5: Backward compatibility ──

def test_agg_columns_none_backward_compat(sample_df):
    """agg_columns=None gives bit-identical output to call without agg_columns."""
    kw = _base_kwargs(sample_df)

    result_without = make_sliding_window_fit(**kw)
    result_with_none = make_sliding_window_fit(**kw, agg_columns=None, agg_median=False)

    # Same columns
    assert list(result_without.columns) == list(result_with_none.columns)

    # Same values
    for col in result_without.columns:
        a = result_without[col].values
        b = result_with_none[col].values
        if a.dtype.kind == 'f':
            np.testing.assert_array_equal(a, b, err_msg=f"Column {col} differs")
        else:
            assert (a == b).all(), f"Column {col} differs"


# ── Test 6: V5 vs zerocopy path equivalence ──

def test_agg_columns_v5_matches_zerocopy(sample_df):
    """V5 (incremental+numba) and zerocopy (recompute) produce matching agg output."""
    common = dict(
        df=sample_df,
        gb_columns=['xBin', 'yBin'],
        fit_columns=['target'],
        linear_columns=['predictor'],
        window_spec={'xBin': 1},
        suffix='_sw',
        min_stat=5,
        agg_columns=['extra_float', 'coord_x'],
    )

    result_recompute = make_sliding_window_fit(
        **common, algorithm='recompute', backend='numpy')

    result_incremental = make_sliding_window_fit(
        **common, algorithm='incremental', backend='numpy')

    # Merge on bin coordinates
    merged = result_recompute.merge(
        result_incremental, on=['xBin', 'yBin'], suffixes=('_rc', '_inc'))

    # agg_columns mean/std should match between paths
    for col in ['extra_float', 'coord_x']:
        np.testing.assert_allclose(
            merged[f'{col}_mean_sw_rc'].values,
            merged[f'{col}_mean_sw_inc'].values,
            rtol=1e-10, atol=1e-12,
            err_msg=f"{col}_mean differs between recompute and incremental")
        np.testing.assert_allclose(
            merged[f'{col}_std_sw_rc'].values,
            merged[f'{col}_std_sw_inc'].values,
            rtol=1e-10, atol=1e-12,
            err_msg=f"{col}_std differs between recompute and incremental")


# ── Test 7: Default output has no fit_column stats ──

def test_default_no_fit_stats(sample_df):
    """Default output has no {t}_mean, {t}_std, {t}_median, {t}_entries, {t}_r_squared."""
    result = make_sliding_window_fit(**_base_kwargs(sample_df))

    for col in result.columns:
        assert not col.endswith('_mean_sw'), f"Unexpected fit stat column: {col}"
        assert not col.endswith('_std_sw'), f"Unexpected fit stat column: {col}"
        assert not col.endswith('_median_sw'), f"Unexpected fit stat column: {col}"
        assert not col.endswith('_entries_sw'), f"Unexpected fit stat column: {col}"
        assert '_r_squared_' not in col, f"Unexpected r_squared column: {col}"

    # Fit results should still be present
    assert 'target_intercept_sw' in result.columns
    assert 'target_slope_predictor_sw' in result.columns
    assert 'target_rmse_sw' in result.columns
    assert 'target_n_fitted_sw' in result.columns


# ── Test 8: agg_columns restores fit_column stats ──

def test_agg_columns_restores_fit_stats(sample_df):
    """Adding fit_column to agg_columns produces mean/std for that column."""
    result = make_sliding_window_fit(
        **_base_kwargs(sample_df, agg_columns=['target']))

    assert 'target_mean_sw' in result.columns
    assert 'target_std_sw' in result.columns
    assert result['target_mean_sw'].notna().all()
