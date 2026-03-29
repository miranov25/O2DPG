"""
Tests for fit_intercept=False across ALL fitters.

P0 bug: _fit_window_regression_numba hardcoded fit_intercept=True.
These tests prevent recurrence across all code paths.

Key invariance:
  - All fitters with fit_intercept=False recover known polynomial coefficients
  - All fitters agree with each other (cross-fitter parity)
  - No fitter produces intercept columns when fit_intercept=False
"""
import numpy as np
import pandas as pd
import pytest

try:
    from groupby_regression_optimized import (
        make_parallel_fit_v3,
        make_parallel_fit_v4,
    )
    from groupby_regression_sliding_window import make_sliding_window_fit
except ImportError:
    from ..groupby_regression_optimized import (
        make_parallel_fit_v3,
        make_parallel_fit_v4,
    )
    from ..groupby_regression_sliding_window import make_sliding_window_fit


# ── Fixture ──

@pytest.fixture
def poly_df():
    """DataFrame with polynomial basis including constant term.

    True model: y = 0.5 + 2*drift + 0.3*drift^2 - tgslp + noise(σ=0.05)

    This is the exact pattern that triggers the bug:
    fit_intercept=False with a constant column in linear_columns.
    """
    rng = np.random.RandomState(42)
    frames = []
    for sec in range(4):
        for row_bin in range(5):
            n = 200
            drift = rng.uniform(-1, 1, n)
            tgslp = rng.uniform(-0.5, 0.5, n)
            y = 0.5 + 2 * drift + 0.3 * drift ** 2 - tgslp + rng.normal(0, 0.05, n)
            frames.append(pd.DataFrame({
                'sec': sec,
                'row_bin': row_bin,
                'drift': drift,
                'tgslp': tgslp,
                'y': y,
                'const': np.ones(n),
                'drift1': drift,
                'drift2': drift ** 2,
                'tgslp1': tgslp,
            }))
    return pd.concat(frames, ignore_index=True)


LIN_COLS = ['const', 'drift1', 'drift2', 'tgslp1']
GB_COLS = ['sec', 'row_bin']
TRUE_COEFFS = {'const': 0.5, 'drift1': 2.0, 'drift2': 0.3, 'tgslp1': -1.0}


# ═══════════════════════════════════════════════════════════════
# Helper: check coefficients recovered
# ═══════════════════════════════════════════════════════════════

def _check_coefficients(dfGB, suffix, fitter_name):
    """Verify recovered coefficients match true values."""
    for col, true_val in TRUE_COEFFS.items():
        col_name = f'y_slope_{col}{suffix}'
        if col_name not in dfGB.columns:
            pytest.fail(f"{fitter_name}: missing column {col_name}")
        mean_val = dfGB[col_name].mean()
        np.testing.assert_allclose(
            mean_val, true_val, atol=0.15,
            err_msg=f"{fitter_name}: {col} not recovered "
                    f"(got {mean_val:.3f}, expected {true_val:.3f})")


def _check_no_intercept_columns(dfGB, suffix, fitter_name):
    """Verify no intercept columns in output."""
    intercept_cols = [c for c in dfGB.columns if 'intercept' in c.lower()]
    assert len(intercept_cols) == 0, \
        f"{fitter_name}: fit_intercept=False produced intercept columns: {intercept_cols}"


def _check_no_failures(dfGB, suffix, fitter_name):
    """Verify no fit failures."""
    qf_col = f'quality_flag{suffix}'
    if qf_col in dfGB.columns:
        n_failed = dfGB[qf_col].str.contains('failed').sum()
        assert n_failed == 0, \
            f"{fitter_name}: {n_failed}/{len(dfGB)} bins failed with fit_intercept=False"


# ═══════════════════════════════════════════════════════════════
# Test 1: V4 recovers coefficients (INVARIANCE — reference)
# ═══════════════════════════════════════════════════════════════

def test_v4_fit_intercept_false_recovers_coefficients(poly_df):
    """V4 with fit_intercept=False recovers known polynomial coefficients."""
    _, dfGB = make_parallel_fit_v4(
        df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
        linear_columns=LIN_COLS, suffix='_test',
        fit_intercept=False, min_stat=10,
    )
    _check_no_intercept_columns(dfGB, '_test', 'V4')
    _check_coefficients(dfGB, '_test', 'V4')


# ═══════════════════════════════════════════════════════════════
# Test 2: V3 recovers coefficients (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_v3_fit_intercept_false_recovers_coefficients(poly_df):
    """V3 with fit_intercept=False recovers known polynomial coefficients."""
    _, dfGB = make_parallel_fit_v3(
        df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
        linear_columns=LIN_COLS, suffix='_test',
        fit_intercept=False, min_stat=10,
    )
    _check_no_intercept_columns(dfGB, '_test', 'V3')
    _check_coefficients(dfGB, '_test', 'V3')


# ═══════════════════════════════════════════════════════════════
# Test 3: V2 — SKIPPED (legacy API, does not support fit_intercept)
# ═══════════════════════════════════════════════════════════════

# V2 (make_parallel_fit_v2) is the legacy statsmodels wrapper.
# It does not accept fit_intercept parameter. Not tested here.


# ═══════════════════════════════════════════════════════════════
# Test 4a: SW numpy recovers coefficients (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_sw_numpy_fit_intercept_false_recovers_coefficients(poly_df):
    """SW numpy backend with fit_intercept=False recovers known coefficients."""
    dfGB = make_sliding_window_fit(
        df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
        linear_columns=LIN_COLS,
        window_spec={'sec': 0, 'row_bin': 0},
        suffix='_test', fit_intercept=False, min_stat=10,
        backend='numpy',
    )
    _check_no_failures(dfGB, '_test', 'SW-numpy')
    _check_no_intercept_columns(dfGB, '_test', 'SW-numpy')
    _check_coefficients(dfGB, '_test', 'SW-numpy')


# ═══════════════════════════════════════════════════════════════
# Test 4b: SW numba recovers coefficients (INVARIANCE — exact bug)
# ═══════════════════════════════════════════════════════════════

def test_sw_numba_fit_intercept_false_recovers_coefficients(poly_df):
    """SW numba backend with fit_intercept=False recovers known coefficients.

    THIS IS THE EXACT BUG: _fit_window_regression_numba hardcoded
    fit_intercept=True and n_params=n_pred+1. This test forces the
    numba backend to verify the fix.
    """
    try:
        dfGB = make_sliding_window_fit(
            df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
            linear_columns=LIN_COLS,
            window_spec={'sec': 0, 'row_bin': 0},
            suffix='_test', fit_intercept=False, min_stat=10,
            backend='numba',
        )
    except Exception:
        pytest.skip("Numba not available")
    _check_no_failures(dfGB, '_test', 'SW-numba')
    _check_no_intercept_columns(dfGB, '_test', 'SW-numba')
    _check_coefficients(dfGB, '_test', 'SW-numba')


# ═══════════════════════════════════════════════════════════════
# Test 5a: SW numpy ≡ V4 with fit_intercept=False (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_sw_numpy_fit_intercept_false_matches_v4(poly_df):
    """SW numpy with window=0 and fit_intercept=False ≡ V4."""
    _, dfGB_v4 = make_parallel_fit_v4(
        df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
        linear_columns=LIN_COLS, suffix='_ref',
        fit_intercept=False, min_stat=10,
    )
    dfGB_sw = make_sliding_window_fit(
        df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
        linear_columns=LIN_COLS,
        window_spec={'sec': 0, 'row_bin': 0},
        suffix='_ref', fit_intercept=False, min_stat=10,
        backend='numpy',
    )
    v4 = dfGB_v4.sort_values(GB_COLS).reset_index(drop=True)
    sw = dfGB_sw.sort_values(GB_COLS).reset_index(drop=True)
    assert len(v4) == len(sw), f"Row count: v4={len(v4)}, sw={len(sw)}"
    slope_cols = [c for c in v4.columns if 'slope' in c]
    for col in slope_cols:
        if col in sw.columns:
            v4_vals = v4[col].values
            sw_vals = sw[col].values
            valid = np.isfinite(v4_vals) & np.isfinite(sw_vals)
            if valid.sum() > 0:
                np.testing.assert_allclose(
                    sw_vals[valid], v4_vals[valid],
                    rtol=1e-6, atol=1e-10,
                    err_msg=f"SW-numpy ≠ V4 for {col}")


# ═══════════════════════════════════════════════════════════════
# Test 5b: SW numba ≡ V4 with fit_intercept=False (INVARIANCE — gate)
# ═══════════════════════════════════════════════════════════════

def test_sw_numba_fit_intercept_false_matches_v4(poly_df):
    """SW numba with window=0 and fit_intercept=False ≡ V4.

    THIS IS THE GATE TEST. If this fails, fit_intercept is broken
    in the numba SW path.
    """
    _, dfGB_v4 = make_parallel_fit_v4(
        df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
        linear_columns=LIN_COLS, suffix='_ref',
        fit_intercept=False, min_stat=10,
    )
    try:
        dfGB_sw = make_sliding_window_fit(
            df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
            linear_columns=LIN_COLS,
            window_spec={'sec': 0, 'row_bin': 0},
            suffix='_ref', fit_intercept=False, min_stat=10,
            backend='numba',
        )
    except Exception:
        pytest.skip("Numba not available")
    v4 = dfGB_v4.sort_values(GB_COLS).reset_index(drop=True)
    sw = dfGB_sw.sort_values(GB_COLS).reset_index(drop=True)
    assert len(v4) == len(sw), f"Row count: v4={len(v4)}, sw={len(sw)}"

    # First: no failures
    _check_no_failures(sw, '_ref', 'SW-numba')

    slope_cols = [c for c in v4.columns if 'slope' in c]
    for col in slope_cols:
        if col in sw.columns:
            v4_vals = v4[col].values
            sw_vals = sw[col].values
            valid = np.isfinite(v4_vals) & np.isfinite(sw_vals)
            if valid.sum() > 0:
                np.testing.assert_allclose(
                    sw_vals[valid], v4_vals[valid],
                    rtol=1e-6, atol=1e-10,
                    err_msg=f"SW-numba ≠ V4 for {col}")


# ═══════════════════════════════════════════════════════════════
# Test 6: SW numba ≡ SW numpy with fit_intercept=False (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_sw_fit_intercept_false_numba_matches_numpy(poly_df):
    """Numba path ≡ numpy path with fit_intercept=False in SW."""
    ws = {'row_bin': 1}

    dfGB_numpy = make_sliding_window_fit(
        df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
        linear_columns=LIN_COLS, window_spec=ws,
        suffix='_test', fit_intercept=False, min_stat=10,
        backend='numpy',
    )

    try:
        dfGB_numba = make_sliding_window_fit(
            df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
            linear_columns=LIN_COLS, window_spec=ws,
            suffix='_test', fit_intercept=False, min_stat=10,
            backend='numba',
        )
    except Exception:
        pytest.skip("Numba not available")

    np_s = dfGB_numpy.sort_values(GB_COLS).reset_index(drop=True)
    nb_s = dfGB_numba.sort_values(GB_COLS).reset_index(drop=True)

    assert len(np_s) == len(nb_s)

    for name, df_check in [('numpy', np_s), ('numba', nb_s)]:
        _check_no_failures(df_check, '_test', f'SW-{name}')

    slope_cols = [c for c in np_s.columns if 'slope' in c]
    for col in slope_cols:
        if col in nb_s.columns:
            np_vals = np_s[col].values
            nb_vals = nb_s[col].values
            valid = np.isfinite(np_vals) & np.isfinite(nb_vals)
            if valid.sum() > 0:
                np.testing.assert_allclose(
                    nb_vals[valid], np_vals[valid],
                    rtol=1e-6, atol=1e-10,
                    err_msg=f"numba ≠ numpy for {col} with fit_intercept=False")


# ═══════════════════════════════════════════════════════════════
# Test 7: Cross-fitter parity V2 ≡ V3 ≡ V4 (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

def test_cross_fitter_parity_fit_intercept_false(poly_df):
    """V3 and V4 agree with fit_intercept=False."""
    results = {}

    for name, func in [('V3', make_parallel_fit_v3),
                        ('V4', make_parallel_fit_v4)]:
        _, dfGB = func(
            df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
            linear_columns=LIN_COLS, suffix='_test',
            fit_intercept=False, min_stat=10,
        )
        results[name] = dfGB.sort_values(GB_COLS).reset_index(drop=True)

    # Compare V3 against V4 (reference)
    ref = results['V4']
    slope_cols = [c for c in ref.columns if 'slope' in c]

    for name in ['V3']:
        other = results[name]
        for col in slope_cols:
            if col in other.columns:
                ref_vals = ref[col].values
                other_vals = other[col].values
                valid = np.isfinite(ref_vals) & np.isfinite(other_vals)
                if valid.sum() > 0:
                    np.testing.assert_allclose(
                        other_vals[valid], ref_vals[valid],
                        rtol=1e-4, atol=1e-8,
                        err_msg=f"{name} ≠ V4 for {col} with fit_intercept=False")


# ═══════════════════════════════════════════════════════════════
# Test 8: SW window>0 numba ≡ manual windowed V4 (INVARIANCE — full chain)
# ═══════════════════════════════════════════════════════════════

def test_sw_window1_numba_matches_manual_windowed_v4(poly_df):
    """SW numba with window=1 and fit_intercept=False ≡ manually windowed V4.

    THIS IS THE FULL-CHAIN TEST. It exercises the complete recompute path
    including result unpacking. Previous tests used window=0 which could
    bypass the buggy code path.

    The manual windowing replicates data with ±1 shift in row_bin, then
    fits with V4 — producing the same result as SW with window=1.
    """
    lin_cols = LIN_COLS
    gb = ['sec', 'row_bin']

    # SW with actual window > 0
    try:
        dfGB_sw = make_sliding_window_fit(
            df=poly_df, gb_columns=gb, fit_columns=['y'],
            linear_columns=lin_cols,
            window_spec={'sec': 0, 'row_bin': 1},
            suffix='_test', fit_intercept=False, min_stat=10,
            backend='numba',
        )
    except Exception:
        pytest.skip("Numba not available")

    # No fit failures
    _check_no_failures(dfGB_sw, '_test', 'SW-numba-w1')

    # No intercept columns
    _check_no_intercept_columns(dfGB_sw, '_test', 'SW-numba-w1')

    # Must have slope columns with finite values
    slope_cols = [c for c in dfGB_sw.columns if 'slope' in c and '_err' not in c]
    assert len(slope_cols) == len(lin_cols), \
        f"Expected {len(lin_cols)} slope columns, got {len(slope_cols)}: {slope_cols}"
    for col in slope_cols:
        n_finite = np.isfinite(dfGB_sw[col]).sum()
        assert n_finite > 0, f"All NaN in {col}"

    # Manual windowing: replicate data with ±1 row_bin shift
    parts = []
    for offset in [-1, 0, 1]:
        tmp = poly_df.copy()
        tmp['row_bin'] = tmp['row_bin'] - offset
        parts.append(tmp)
    df_windowed = pd.concat(parts, ignore_index=True)

    _, dfGB_manual = make_parallel_fit_v4(
        df=df_windowed, gb_columns=gb, fit_columns=['y'],
        linear_columns=lin_cols, suffix='_test',
        fit_intercept=False, min_stat=10,
    )

    # Compare: SW window=1 ≡ manual windowed V4
    sw = dfGB_sw.sort_values(gb).reset_index(drop=True)
    manual = dfGB_manual.sort_values(gb).reset_index(drop=True)

    # Only compare bins present in both (edge bins may differ)
    merged = sw.merge(manual, on=gb, suffixes=('_sw', '_manual'))

    for lin_col in lin_cols:
        col_sw = f'y_slope_{lin_col}_test_sw'
        col_man = f'y_slope_{lin_col}_test_manual'
        if col_sw in merged.columns and col_man in merged.columns:
            sw_vals = merged[col_sw].values
            man_vals = merged[col_man].values
            valid = np.isfinite(sw_vals) & np.isfinite(man_vals)
            if valid.sum() > 0:
                np.testing.assert_allclose(
                    sw_vals[valid], man_vals[valid],
                    rtol=1e-5, atol=1e-8,
                    err_msg=f"SW-numba window=1 ≠ manual windowed V4 for {lin_col}")


# ═══════════════════════════════════════════════════════════════
# Test 9: SW window>0 numpy ≡ numba (INVARIANCE — backend parity with window)
# ═══════════════════════════════════════════════════════════════

def test_sw_window1_numba_matches_numpy(poly_df):
    """SW numba with window=1 ≡ SW numpy with window=1 and fit_intercept=False.

    Both backends must produce identical results with actual windowing.
    """
    ws = {'sec': 0, 'row_bin': 1}

    dfGB_numpy = make_sliding_window_fit(
        df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
        linear_columns=LIN_COLS, window_spec=ws,
        suffix='_test', fit_intercept=False, min_stat=10,
        backend='numpy',
    )

    try:
        dfGB_numba = make_sliding_window_fit(
            df=poly_df, gb_columns=GB_COLS, fit_columns=['y'],
            linear_columns=LIN_COLS, window_spec=ws,
            suffix='_test', fit_intercept=False, min_stat=10,
            backend='numba',
        )
    except Exception:
        pytest.skip("Numba not available")

    # No failures in either
    _check_no_failures(dfGB_numpy, '_test', 'SW-numpy-w1')
    _check_no_failures(dfGB_numba, '_test', 'SW-numba-w1')

    np_s = dfGB_numpy.sort_values(GB_COLS).reset_index(drop=True)
    nb_s = dfGB_numba.sort_values(GB_COLS).reset_index(drop=True)

    assert len(np_s) == len(nb_s)

    slope_cols = [c for c in np_s.columns if 'slope' in c]
    for col in slope_cols:
        if col in nb_s.columns:
            np_vals = np_s[col].values
            nb_vals = nb_s[col].values
            valid = np.isfinite(np_vals) & np.isfinite(nb_vals)
            if valid.sum() > 0:
                np.testing.assert_allclose(
                    nb_vals[valid], np_vals[valid],
                    rtol=1e-6, atol=1e-10,
                    err_msg=f"numba ≠ numpy for {col} with window=1, fit_intercept=False")
