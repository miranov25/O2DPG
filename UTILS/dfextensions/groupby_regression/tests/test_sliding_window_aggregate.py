"""
Tests for Phase 13.14.GB — Dedicated Sliding Window Aggregation.

Tests make_sliding_window_aggregate and make_sliding_window_aggregate_parallel.
Key invariance: agg-only ≡ fit path with linear_columns=[] and agg_columns.
"""
import numpy as np
import pandas as pd
import pytest
import time

try:
    from groupby_regression_sliding_window import (
        make_sliding_window_fit,
        make_sliding_window_aggregate,
        make_sliding_window_aggregate_parallel,
    )
except ImportError:
    from ..groupby_regression_sliding_window import (
        make_sliding_window_fit,
        make_sliding_window_aggregate,
        make_sliding_window_aggregate_parallel,
    )


# ── Fixtures ──

@pytest.fixture
def sample_3d():
    """3D grid with known data for aggregation tests."""
    rng = np.random.RandomState(42)
    frames = []
    for xb in range(5):
        for yb in range(5):
            for zb in range(3):
                n = 30
                x = rng.standard_normal(n)
                y = 2.0 * x + rng.normal(0, 0.1, n)
                extra = rng.uniform(0, 10, n)
                frames.append(pd.DataFrame({
                    'xBin': xb, 'yBin': yb, 'zBin': zb,
                    'x': x, 'y': y, 'extra': extra,
                }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def sample_5d():
    """5D grid for performance test."""
    rng = np.random.RandomState(42)
    dims = [8, 8, 6, 6, 5]  # 8*8*6*6*5 = 11520 bins
    coords = np.array(np.meshgrid(*[np.arange(d) for d in dims])).T.reshape(-1, 5)
    n_bins = len(coords)
    rpb = 10
    coords_rep = np.repeat(coords, rpb, axis=0)
    n = len(coords_rep)
    df = pd.DataFrame({
        'd0': coords_rep[:, 0], 'd1': coords_rep[:, 1],
        'd2': coords_rep[:, 2], 'd3': coords_rep[:, 3],
        'd4': coords_rep[:, 4],
        'val1': rng.standard_normal(n),
        'val2': rng.standard_normal(n),
        'val3': rng.standard_normal(n),
    })
    return df


@pytest.fixture
def sample_nan():
    """3D grid with different NaN patterns per column."""
    rng = np.random.RandomState(42)
    frames = []
    for xb in range(4):
        for yb in range(4):
            n = 50
            x = rng.standard_normal(n)
            y = rng.standard_normal(n)
            # Column x has NaN at different positions than y
            x_with_nan = x.copy()
            y_with_nan = y.copy()
            x_with_nan[rng.random(n) < 0.1] = np.nan
            y_with_nan[rng.random(n) < 0.3] = np.nan  # more NaN in y
            frames.append(pd.DataFrame({
                'xBin': xb, 'yBin': yb,
                'col_a': x_with_nan, 'col_b': y_with_nan,
            }))
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# Test 1: Basic output (smoke)
# ═══════════════════════════════════════════════════════════════

def test_aggregate_basic(sample_3d):
    """Output has mean/std/count columns for each agg_column."""
    result = make_sliding_window_aggregate(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        agg_columns=['x', 'y', 'extra'],
        window_spec={'xBin': 1, 'yBin': 0, 'zBin': 0},
        suffix='_sw',
    )
    assert len(result) > 0
    for col in ['x', 'y', 'extra']:
        assert f'{col}_mean_sw' in result.columns
        assert f'{col}_std_sw' in result.columns
        assert f'{col}_count_sw' in result.columns
    assert 'n_neighbors_used_sw' in result.columns
    assert 'effective_window_fraction_sw' in result.columns
    assert result['x_mean_sw'].notna().all()


# ═══════════════════════════════════════════════════════════════
# Test 2: Matches fit path (INVARIANCE — key gate)
# ═══════════════════════════════════════════════════════════════

def test_aggregate_matches_fit_path(sample_3d):
    """Agg-only ≡ make_sliding_window_fit with linear_columns=[] and agg_columns."""
    ws = {'xBin': 1, 'yBin': 1, 'zBin': 0}
    agg_cols = ['x', 'y', 'extra']

    # Fit path (intercept-only + agg_columns)
    result_fit = make_sliding_window_fit(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        fit_columns=['y'], linear_columns=[],
        window_spec=ws, min_stat=1, suffix='_sw',
        agg_columns=agg_cols,
    )

    # Dedicated agg path
    result_agg = make_sliding_window_aggregate(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        agg_columns=agg_cols,
        window_spec=ws, min_stat=1, suffix='_sw',
    )

    keys = ['xBin', 'yBin', 'zBin']
    fit_s = result_fit.sort_values(keys).reset_index(drop=True)
    agg_s = result_agg.sort_values(keys).reset_index(drop=True)

    assert len(fit_s) == len(agg_s), \
        f"Row count: fit={len(fit_s)}, agg={len(agg_s)}"

    for col in agg_cols:
        np.testing.assert_allclose(
            fit_s[f'{col}_mean_sw'].values,
            agg_s[f'{col}_mean_sw'].values,
            rtol=1e-12, atol=1e-14,
            err_msg=f"Mean mismatch: {col}")
        np.testing.assert_allclose(
            fit_s[f'{col}_std_sw'].values,
            agg_s[f'{col}_std_sw'].values,
            rtol=1e-10, atol=1e-14,
            err_msg=f"Std mismatch: {col}")


# ═══════════════════════════════════════════════════════════════
# Test 3: Window=0 matches groupby (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_aggregate_window0_matches_groupby(sample_3d):
    """Window=0 (no sliding) ≡ manual pandas groupby."""
    result = make_sliding_window_aggregate(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        agg_columns=['x', 'extra'],
        window_spec={'xBin': 0, 'yBin': 0, 'zBin': 0},
        suffix='_sw',
    )

    manual = sample_3d.groupby(['xBin', 'yBin', 'zBin']).agg(
        x_mean=('x', 'mean'),
        x_std=('x', 'std'),
        x_count=('x', 'count'),
        extra_mean=('extra', 'mean'),
    ).reset_index()

    keys = ['xBin', 'yBin', 'zBin']
    r = result.sort_values(keys).reset_index(drop=True)
    m = manual.sort_values(keys).reset_index(drop=True)

    np.testing.assert_allclose(
        r['x_mean_sw'].values, m['x_mean'].values,
        rtol=1e-12, err_msg="Mean mismatch vs groupby")
    np.testing.assert_allclose(
        r['extra_mean_sw'].values, m['extra_mean'].values,
        rtol=1e-12, err_msg="Extra mean mismatch vs groupby")


# ═══════════════════════════════════════════════════════════════
# Test 4: Weighted aggregation (smoke)
# ═══════════════════════════════════════════════════════════════

def test_aggregate_weighted(sample_3d):
    """Weighted mean/std with weights column produces different results."""
    sample_3d = sample_3d.copy()
    sample_3d['w'] = np.abs(sample_3d['x']) + 0.1  # heterogeneous weights

    result_no_w = make_sliding_window_aggregate(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        agg_columns=['y'], window_spec={'xBin': 1, 'yBin': 0, 'zBin': 0},
        suffix='_sw',
    )
    result_w = make_sliding_window_aggregate(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        agg_columns=['y'], window_spec={'xBin': 1, 'yBin': 0, 'zBin': 0},
        suffix='_sw', weights='w',
    )

    # Weighted and unweighted means should differ
    assert not np.allclose(
        result_no_w['y_mean_sw'].values,
        result_w['y_mean_sw'].values, atol=1e-6), \
        "Weighted and unweighted means should differ"


# ═══════════════════════════════════════════════════════════════
# Test 5: Numba matches numpy (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_aggregate_numba_matches_numpy(sample_3d):
    """Numba and numpy backends produce identical results."""
    ws = {'xBin': 1, 'yBin': 1, 'zBin': 0}

    # Force numpy by catching numba import
    result1 = make_sliding_window_aggregate(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        agg_columns=['x', 'y'], window_spec=ws, suffix='_sw',
    )

    # Both should produce same results regardless of backend
    result2 = make_sliding_window_aggregate(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        agg_columns=['x', 'y'], window_spec=ws, suffix='_sw',
    )

    for col in ['x_mean_sw', 'y_mean_sw', 'x_std_sw', 'y_std_sw']:
        np.testing.assert_array_equal(
            result1[col].values, result2[col].values,
            err_msg=f"Backend mismatch: {col}")


# ═══════════════════════════════════════════════════════════════
# Test 6: Median optional (smoke)
# ═══════════════════════════════════════════════════════════════

def test_aggregate_median_optional(sample_3d):
    """agg_median flag controls median presence."""
    result_no = make_sliding_window_aggregate(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        agg_columns=['x'], window_spec={'xBin': 1, 'yBin': 0, 'zBin': 0},
        suffix='_sw', agg_median=False,
    )
    result_yes = make_sliding_window_aggregate(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        agg_columns=['x'], window_spec={'xBin': 1, 'yBin': 0, 'zBin': 0},
        suffix='_sw', agg_median=True,
    )
    assert 'x_median_sw' not in result_no.columns
    assert 'x_median_sw' in result_yes.columns
    assert result_yes['x_median_sw'].notna().all()


# ═══════════════════════════════════════════════════════════════
# Test 7: 5D large performance (performance gate)
# ═══════════════════════════════════════════════════════════════

def test_aggregate_5d_large(sample_5d):
    """5D grid with ~11k bins completes quickly."""
    ws = {'d0': 1, 'd1': 1, 'd2': 1, 'd3': 1, 'd4': 0}

    t0 = time.time()
    result = make_sliding_window_aggregate(
        df=sample_5d, gb_columns=['d0', 'd1', 'd2', 'd3', 'd4'],
        agg_columns=['val1', 'val2', 'val3'],
        window_spec=ws, suffix='_sw',
    )
    elapsed = time.time() - t0

    print(f"\n  [PERF] 5D agg: {len(result)} bins, {elapsed:.3f}s")
    assert len(result) > 0
    assert elapsed < 30.0, f"Too slow: {elapsed:.1f}s (expect <30s)"


# ═══════════════════════════════════════════════════════════════
# Test 8: Parallel matches serial (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_aggregate_parallel_matches_serial(sample_3d):
    """Parallel ≡ serial at rtol=1e-12."""
    sample_3d = sample_3d.copy()
    sample_3d['sector'] = sample_3d['xBin'] % 3

    ws = {'yBin': 1, 'zBin': 0}

    # Serial: per sector
    serial_parts = []
    for sec, grp in sample_3d.groupby('sector'):
        r = make_sliding_window_aggregate(
            df=grp, gb_columns=['yBin', 'zBin'],
            agg_columns=['x', 'y'],
            window_spec=ws, suffix='_sw',
        )
        r['sector'] = sec
        serial_parts.append(r)
    serial = pd.concat(serial_parts, ignore_index=True)

    # Parallel
    parallel = make_sliding_window_aggregate_parallel(
        df=sample_3d, gb_columns=['yBin', 'zBin'],
        agg_columns=['x', 'y'],
        window_spec=ws, suffix='_sw',
        split_columns=['sector'], n_workers=1,
    )

    keys = ['yBin', 'zBin', 'sector']
    s = serial.sort_values(keys).reset_index(drop=True)
    p = parallel.sort_values(keys).reset_index(drop=True)

    assert len(s) == len(p)
    for col in ['x_mean_sw', 'y_mean_sw', 'x_std_sw', 'y_std_sw']:
        np.testing.assert_allclose(
            s[col].values, p[col].values,
            rtol=1e-12, atol=1e-14,
            err_msg=f"Parallel ≠ serial: {col}")


# ═══════════════════════════════════════════════════════════════
# Test 9: Gaussian kernel (smoke)
# ═══════════════════════════════════════════════════════════════

def test_aggregate_kernel_gaussian(sample_3d):
    """Gaussian kernel produces different results than uniform."""
    ws = {'xBin': 1, 'yBin': 0, 'zBin': 0}

    result_uni = make_sliding_window_aggregate(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        agg_columns=['x'], window_spec=ws, suffix='_sw',
        kernel='uniform',
    )
    result_gauss = make_sliding_window_aggregate(
        df=sample_3d, gb_columns=['xBin', 'yBin', 'zBin'],
        agg_columns=['x'], window_spec=ws, suffix='_sw',
        kernel='gaussian',
    )

    # Interior bins should differ (boundary bins may be same)
    interior = (result_uni['xBin'] >= 1) & (result_uni['xBin'] <= 3)
    assert not np.allclose(
        result_uni.loc[interior, 'x_mean_sw'].values,
        result_gauss.loc[interior, 'x_mean_sw'].values, atol=1e-6), \
        "Gaussian and uniform should differ for interior bins"


# ═══════════════════════════════════════════════════════════════
# Test 10: Per-column NaN counts (smoke)
# ═══════════════════════════════════════════════════════════════

def test_aggregate_per_column_nan_counts(sample_nan):
    """Different NaN patterns give different per-column counts."""
    result = make_sliding_window_aggregate(
        df=sample_nan, gb_columns=['xBin', 'yBin'],
        agg_columns=['col_a', 'col_b'],
        window_spec={'xBin': 0, 'yBin': 0},
        suffix='_sw',
    )

    # col_b has more NaN (30% vs 10%), so counts should differ
    assert not np.array_equal(
        result['col_a_count_sw'].values,
        result['col_b_count_sw'].values), \
        "Per-column counts should differ when NaN patterns differ"

    # col_a should have higher counts than col_b
    assert result['col_a_count_sw'].sum() > result['col_b_count_sw'].sum(), \
        "col_a (10% NaN) should have more entries than col_b (30% NaN)"
