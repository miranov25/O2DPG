"""
Tests for Phase 13.15.GB — N-Sigma Cut in Sliding Window Aggregation.

Key invariance tests:
  - n_sigma_cut=None ≡ no argument (bit-identical)
  - Clean Gaussian data: cut=5 ≡ no cut (no outliers to remove)
  - With injected outliers: cut recovers true mean
  - Parallel with cut ≡ serial with cut
"""
import numpy as np
import pandas as pd
import pytest

try:
    from groupby_regression_sliding_window import (
        make_sliding_window_aggregate,
        make_sliding_window_aggregate_parallel,
    )
except ImportError:
    from ..groupby_regression_sliding_window import (
        make_sliding_window_aggregate,
        make_sliding_window_aggregate_parallel,
    )


# ── Fixtures ──

@pytest.fixture
def clean_df():
    """3D grid with clean Gaussian data — no outliers."""
    rng = np.random.RandomState(42)
    frames = []
    for xb in range(5):
        for yb in range(5):
            n = 100
            x = rng.normal(0, 1, n)
            y = rng.normal(5, 0.5, n)
            frames.append(pd.DataFrame({
                'xBin': xb, 'yBin': yb,
                'x': x, 'y': y,
            }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def outlier_df():
    """3D grid with 2% outliers at +20σ.

    With 2% at +20: Pass 1 mean ≈ 0.4, std ≈ 2.9, cut at 3σ ≈ 8.7.
    Outliers at 20 are |20-0.4|/2.9 ≈ 6.8σ — well above 3σ cut.
    """
    rng = np.random.RandomState(42)
    frames = []
    for xb in range(5):
        for yb in range(5):
            n = 500
            x_clean = rng.normal(0, 1, n)
            y_clean = rng.normal(5, 0.5, n)
            # Inject 2% outliers at 20σ
            n_outlier = n // 50
            outlier_idx = rng.choice(n, n_outlier, replace=False)
            x_dirty = x_clean.copy()
            y_dirty = y_clean.copy()
            x_dirty[outlier_idx] = 20.0 + rng.normal(0, 0.5, n_outlier)
            y_dirty[outlier_idx] = 25.0 + rng.normal(0, 0.5, n_outlier)
            frames.append(pd.DataFrame({
                'xBin': xb, 'yBin': yb,
                'x': x_dirty, 'y': y_dirty,
                'x_true': x_clean, 'y_true': y_clean,
            }))
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def constant_bin_df():
    """Data with some bins having constant values (std=0)."""
    rng = np.random.RandomState(42)
    frames = []
    for xb in range(4):
        for yb in range(4):
            n = 50
            if xb == 0 and yb == 0:
                # Constant values — std = 0
                x = np.full(n, 3.0)
            else:
                x = rng.normal(0, 1, n)
            frames.append(pd.DataFrame({
                'xBin': xb, 'yBin': yb, 'x': x,
            }))
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# Test 1: n_sigma_cut=None ≡ no argument (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_sigma_cut_none_identical(clean_df):
    """n_sigma_cut=None produces bit-identical output to omitting the argument."""
    ws = {'xBin': 1, 'yBin': 0}

    result_default = make_sliding_window_aggregate(
        df=clean_df, gb_columns=['xBin', 'yBin'],
        agg_columns=['x', 'y'], window_spec=ws, suffix='_sw',
    )
    result_none = make_sliding_window_aggregate(
        df=clean_df, gb_columns=['xBin', 'yBin'],
        agg_columns=['x', 'y'], window_spec=ws, suffix='_sw',
        n_sigma_cut=None,
    )

    keys = ['xBin', 'yBin']
    r1 = result_default.sort_values(keys).reset_index(drop=True)
    r2 = result_none.sort_values(keys).reset_index(drop=True)

    assert list(r1.columns) == list(r2.columns)
    for col in r1.columns:
        if r1[col].dtype.kind == 'f':
            np.testing.assert_array_equal(
                r1[col].values, r2[col].values,
                err_msg=f"n_sigma_cut=None ≠ default: {col}")


# ═══════════════════════════════════════════════════════════════
# Test 2: Clean data unaffected by loose cut (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_sigma_cut_no_effect_clean_data(clean_df):
    """Pure Gaussian data with cut=5: virtually no outliers removed."""
    ws = {'xBin': 1, 'yBin': 0}

    result_nocut = make_sliding_window_aggregate(
        df=clean_df, gb_columns=['xBin', 'yBin'],
        agg_columns=['x', 'y'], window_spec=ws, suffix='_sw',
    )
    result_cut5 = make_sliding_window_aggregate(
        df=clean_df, gb_columns=['xBin', 'yBin'],
        agg_columns=['x', 'y'], window_spec=ws, suffix='_sw',
        n_sigma_cut=5.0,
    )

    keys = ['xBin', 'yBin']
    r1 = result_nocut.sort_values(keys).reset_index(drop=True)
    r2 = result_cut5.sort_values(keys).reset_index(drop=True)

    # With 5σ cut on Gaussian data, <0.003% of points removed
    # Results should be nearly identical
    for col in ['x_mean_sw', 'y_mean_sw']:
        np.testing.assert_allclose(
            r1[col].values, r2[col].values,
            rtol=1e-2, atol=1e-4,
            err_msg=f"5σ cut changed clean data: {col}")


# ═══════════════════════════════════════════════════════════════
# Test 3: Sigma cut recovers true mean (INVARIANCE — key gate)
# ═══════════════════════════════════════════════════════════════

def test_sigma_cut_recovers_true_mean(outlier_df):
    """With 10% outliers at 10σ, cut=3 recovers mean closer to true value."""
    ws = {'xBin': 1, 'yBin': 0}

    # Compute true mean from clean data
    result_true = make_sliding_window_aggregate(
        df=outlier_df.rename(columns={'x_true': 'x_t', 'y_true': 'y_t'}),
        gb_columns=['xBin', 'yBin'],
        agg_columns=['x_t', 'y_t'], window_spec=ws, suffix='_sw',
    )

    # Unclipped mean (biased by outliers)
    result_dirty = make_sliding_window_aggregate(
        df=outlier_df, gb_columns=['xBin', 'yBin'],
        agg_columns=['x', 'y'], window_spec=ws, suffix='_sw',
    )

    # Sigma-clipped mean (should be closer to true)
    result_clipped = make_sliding_window_aggregate(
        df=outlier_df, gb_columns=['xBin', 'yBin'],
        agg_columns=['x', 'y'], window_spec=ws, suffix='_sw',
        n_sigma_cut=3.0,
    )

    keys = ['xBin', 'yBin']
    true_s = result_true.sort_values(keys).reset_index(drop=True)
    dirty_s = result_dirty.sort_values(keys).reset_index(drop=True)
    clip_s = result_clipped.sort_values(keys).reset_index(drop=True)

    # Error of dirty vs true
    err_dirty = np.abs(dirty_s['x_mean_sw'].values - true_s['x_t_mean_sw'].values).mean()
    # Error of clipped vs true
    err_clipped = np.abs(clip_s['x_mean_sw'].values - true_s['x_t_mean_sw'].values).mean()

    assert err_clipped < err_dirty, \
        f"Sigma cut should reduce error: dirty={err_dirty:.4f}, clipped={err_clipped:.4f}"

    # Clipped should recover true mean much better than dirty
    np.testing.assert_allclose(
        clip_s['x_mean_sw'].values, true_s['x_t_mean_sw'].values,
        atol=0.5,
        err_msg="Sigma-clipped mean not closer to true mean")


# ═══════════════════════════════════════════════════════════════
# Test 4: Sigma cut reduces std (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_sigma_cut_reduces_std(outlier_df):
    """With outliers, clipped std < unclipped std."""
    ws = {'xBin': 1, 'yBin': 0}

    result_dirty = make_sliding_window_aggregate(
        df=outlier_df, gb_columns=['xBin', 'yBin'],
        agg_columns=['x'], window_spec=ws, suffix='_sw',
    )
    result_clipped = make_sliding_window_aggregate(
        df=outlier_df, gb_columns=['xBin', 'yBin'],
        agg_columns=['x'], window_spec=ws, suffix='_sw',
        n_sigma_cut=3.0,
    )

    keys = ['xBin', 'yBin']
    dirty_s = result_dirty.sort_values(keys).reset_index(drop=True)
    clip_s = result_clipped.sort_values(keys).reset_index(drop=True)

    # Clipped std should be smaller on average
    assert clip_s['x_std_sw'].mean() < dirty_s['x_std_sw'].mean(), \
        "Sigma cut should reduce std when outliers present"


# ═══════════════════════════════════════════════════════════════
# Test 5: std=0 bins survive sigma cut (P1-1 guard)
# ═══════════════════════════════════════════════════════════════

def test_sigma_cut_std_zero_safe(constant_bin_df):
    """Bins with constant values (std=0) are not wiped out by sigma cut."""
    ws = {'xBin': 0, 'yBin': 0}

    result = make_sliding_window_aggregate(
        df=constant_bin_df, gb_columns=['xBin', 'yBin'],
        agg_columns=['x'], window_spec=ws, suffix='_sw',
        n_sigma_cut=3.0,
    )

    # The constant bin (xBin=0, yBin=0) should still have valid mean
    const_bin = result[(result['xBin'] == 0) & (result['yBin'] == 0)]
    assert len(const_bin) == 1
    assert np.isfinite(const_bin['x_mean_sw'].values[0]), \
        "Constant bin (std=0) should survive sigma cut"
    np.testing.assert_allclose(
        const_bin['x_mean_sw'].values[0], 3.0, atol=1e-10,
        err_msg="Constant bin mean should be 3.0")
    assert const_bin['x_count_sw'].values[0] == 50, \
        "Constant bin should retain all 50 points (no outliers)"


# ═══════════════════════════════════════════════════════════════
# Test 6: Parallel with sigma cut ≡ serial (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_sigma_cut_parallel_matches_serial(outlier_df):
    """Parallel with n_sigma_cut ≡ serial with n_sigma_cut."""
    outlier_df = outlier_df.copy()
    outlier_df['sector'] = outlier_df['xBin'] % 3
    ws = {'yBin': 1}

    # Serial per sector
    serial_parts = []
    for sec, grp in outlier_df.groupby('sector'):
        r = make_sliding_window_aggregate(
            df=grp, gb_columns=['yBin'],
            agg_columns=['x', 'y'], window_spec=ws, suffix='_sw',
            n_sigma_cut=3.0,
        )
        r['sector'] = sec
        serial_parts.append(r)
    serial = pd.concat(serial_parts, ignore_index=True)

    # Parallel
    parallel = make_sliding_window_aggregate_parallel(
        df=outlier_df, gb_columns=['yBin'],
        agg_columns=['x', 'y'], window_spec=ws, suffix='_sw',
        split_columns=['sector'], n_workers=1,
        n_sigma_cut=3.0,
    )

    keys = ['yBin', 'sector']
    s = serial.sort_values(keys).reset_index(drop=True)
    p = parallel.sort_values(keys).reset_index(drop=True)

    assert len(s) == len(p)
    for col in ['x_mean_sw', 'y_mean_sw', 'x_std_sw']:
        np.testing.assert_allclose(
            s[col].values, p[col].values,
            rtol=1e-12, atol=1e-14,
            err_msg=f"Parallel ≠ serial with sigma cut: {col}")
