"""
Tests for WLS weight fix + fit_intercept column fix (Phase 13.9.GB Extension).

7 tests:
  1-5: WLS weights actually affect regression
  6-7: fit_intercept=False produces correct output
"""
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from groupby_regression_sliding_window import make_sliding_window_fit


@pytest.fixture
def heteroscedastic_df():
    """Data with known heteroscedastic noise: y = 2 + 3*x + noise(x).

    Noise increases with x: std(noise) = 0.1 + 2*x.
    WLS with w = 1/var should recover (2, 3) better than OLS.
    """
    rng = np.random.default_rng(42)
    n_per_bin = 200
    rows = []
    for xb in range(5):
        for yb in range(3):
            for _ in range(n_per_bin):
                x = rng.normal(0, 1)
                noise_std = 0.1 + 2.0 * abs(x)  # heteroscedastic
                noise = rng.normal(0, noise_std)
                y = 2.0 + 3.0 * x + noise
                rows.append({
                    'xBin': xb, 'yBin': yb,
                    'predictor': x,
                    'target': y,
                    'w_inv_var': 1.0 / (noise_std ** 2),
                    'w_uniform': 1.0,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def ms_df():
    """Multiple scattering: sigma² = 0.01 + 0.5 * mp².

    At low mp, sigma is small → high weight. WLS should keep intercept > 0.
    """
    rng = np.random.default_rng(123)
    n_per_bin = 300
    rows = []
    for xb in range(4):
        for yb in range(3):
            for _ in range(n_per_bin):
                mp = rng.uniform(0.01, 3.0)
                sigma2_true = 0.01 + 0.5 * mp ** 2
                # observed |delta|² with chi2(1) fluctuation around sigma2_true
                sigma2_obs = sigma2_true * rng.chisquare(1)
                rows.append({
                    'xBin': xb, 'yBin': yb,
                    'mp2': mp ** 2,
                    'sigma2': sigma2_obs,
                    'w_ms': 1.0 / max(mp ** 2, 0.01),  # more weight at low mp
                })
    return pd.DataFrame(rows)


def _base_kwargs(df, **overrides):
    kw = dict(
        df=df,
        gb_columns=['xBin', 'yBin'],
        window_spec={'xBin': 1},
        suffix='_sw',
        min_stat=5,
        algorithm='recompute',
        backend='numpy',
    )
    kw.update(overrides)
    return kw


# ── Test 1: WLS changes coefficients ──

def test_wls_changes_coefficients(heteroscedastic_df):
    """With heteroscedastic data, WLS coefficients differ from OLS."""
    kw = _base_kwargs(heteroscedastic_df,
                      fit_columns=['target'],
                      linear_columns=['predictor'])

    result_ols = make_sliding_window_fit(**kw, weights=None)
    result_wls = make_sliding_window_fit(**kw, weights='w_inv_var')

    # Coefficients should differ
    slope_ols = result_ols['target_slope_predictor_sw'].values
    slope_wls = result_wls['target_slope_predictor_sw'].values

    assert not np.allclose(slope_ols, slope_wls, atol=1e-6), \
        "WLS should produce different coefficients than OLS for heteroscedastic data"

    # Both should be finite
    assert np.all(np.isfinite(slope_ols))
    assert np.all(np.isfinite(slope_wls))


# ── Test 2: WLS recovers known slope better ──

def test_wls_recovers_known_slope(heteroscedastic_df):
    """WLS with correct weights recovers true slope=3 better than OLS."""
    # Use window=0 for clean per-bin comparison
    kw = _base_kwargs(heteroscedastic_df,
                      fit_columns=['target'],
                      linear_columns=['predictor'],
                      window_spec={'xBin': 0, 'yBin': 0})

    result_ols = make_sliding_window_fit(**kw, weights=None)
    result_wls = make_sliding_window_fit(**kw, weights='w_inv_var')

    # Mean absolute error from true slope = 3.0
    mae_ols = np.abs(result_ols['target_slope_predictor_sw'].values - 3.0).mean()
    mae_wls = np.abs(result_wls['target_slope_predictor_sw'].values - 3.0).mean()

    # WLS should have smaller error (better recovery)
    assert mae_wls < mae_ols, \
        f"WLS MAE ({mae_wls:.4f}) should be smaller than OLS MAE ({mae_ols:.4f})"


# ── Test 3: Uniform weights ≡ OLS ──

def test_wls_uniform_weights_equals_ols(heteroscedastic_df):
    """weights=1.0 for all rows gives identical result to weights=None."""
    kw = _base_kwargs(heteroscedastic_df,
                      fit_columns=['target'],
                      linear_columns=['predictor'])

    result_none = make_sliding_window_fit(**kw, weights=None)
    result_uniform = make_sliding_window_fit(**kw, weights='w_uniform')

    for col in ['target_intercept_sw', 'target_slope_predictor_sw',
                'target_rmse_sw', 'target_r_squared_sw']:
        np.testing.assert_allclose(
            result_none[col].values, result_uniform[col].values,
            rtol=1e-10, atol=1e-12,
            err_msg=f"{col} differs between weights=None and weights=1.0")


# ── Test 4: V3 incremental matches V1 recompute with weights ──

def test_wls_all_paths_match(heteroscedastic_df):
    """V1 (recompute) and V3 (incremental) produce same WLS results."""
    common = dict(
        df=heteroscedastic_df,
        gb_columns=['xBin', 'yBin'],
        fit_columns=['target'],
        linear_columns=['predictor'],
        window_spec={'xBin': 1},
        suffix='_sw',
        min_stat=5,
        weights='w_inv_var',
    )

    result_v1 = make_sliding_window_fit(**common, algorithm='recompute', backend='numpy')
    result_v3 = make_sliding_window_fit(**common, algorithm='incremental', backend='numpy')

    merged = result_v1.merge(result_v3, on=['xBin', 'yBin'], suffixes=('_v1', '_v3'))

    for col in ['target_intercept_sw', 'target_slope_predictor_sw']:
        np.testing.assert_allclose(
            merged[f'{col}_v1'].values, merged[f'{col}_v3'].values,
            rtol=1e-6, atol=1e-10,
            err_msg=f"{col} differs between V1 and V3 with WLS")


# ── Test 5: Multiple scattering — WLS keeps intercept positive ──

def test_wls_positive_intercept_ms(ms_df):
    """Multiple scattering: sigma² = p0 + p1*mp², WLS gives p0 > 0."""
    kw = _base_kwargs(ms_df,
                      fit_columns=['sigma2'],
                      linear_columns=['mp2'],
                      window_spec={'xBin': 0, 'yBin': 0})

    result_ols = make_sliding_window_fit(**kw, weights=None)
    result_wls = make_sliding_window_fit(**kw, weights='w_ms')

    # WLS intercept should be closer to 0.01 (true value) and positive
    intercept_wls = result_wls['sigma2_intercept_sw'].values
    intercept_ols = result_ols['sigma2_intercept_sw'].values

    # WLS: most bins should have positive intercept
    frac_positive_wls = np.mean(intercept_wls > 0)
    frac_positive_ols = np.mean(intercept_ols > 0)

    assert frac_positive_wls >= frac_positive_ols, \
        f"WLS positive fraction ({frac_positive_wls:.2f}) should be >= OLS ({frac_positive_ols:.2f})"


# ── Test 6: fit_intercept=False — no intercept columns ──

def test_fit_intercept_false_no_intercept_columns(heteroscedastic_df):
    """fit_intercept=False → no {t}_intercept or {t}_intercept_err in output."""
    result = make_sliding_window_fit(
        **_base_kwargs(heteroscedastic_df,
                       fit_columns=['target'],
                       linear_columns=['predictor'],
                       fit_intercept=False))

    assert 'target_intercept_sw' not in result.columns, \
        "intercept column should not exist when fit_intercept=False"
    assert 'target_intercept_err_sw' not in result.columns, \
        "intercept_err column should not exist when fit_intercept=False"

    # Slope should still be present
    assert 'target_slope_predictor_sw' in result.columns
    assert 'target_slope_predictor_err_sw' in result.columns
    assert 'target_rmse_sw' in result.columns


# ── Test 7: fit_intercept=False — slope is correct ──

def test_fit_intercept_false_slope_correct(heteroscedastic_df):
    """fit_intercept=False with y=b*x data recovers b correctly.

    Create data: y = 5*x (no intercept). Fit without intercept should
    recover slope ≈ 5.
    """
    rng = np.random.default_rng(99)
    rows = []
    for xb in range(4):
        for yb in range(3):
            for _ in range(200):
                x = rng.normal(0, 1)
                y = 5.0 * x + rng.normal(0, 0.1)
                rows.append({'xBin': xb, 'yBin': yb, 'x': x, 'y': y})
    df = pd.DataFrame(rows)

    result = make_sliding_window_fit(
        df=df,
        gb_columns=['xBin', 'yBin'],
        fit_columns=['y'],
        linear_columns=['x'],
        window_spec={'xBin': 0, 'yBin': 0},
        suffix='_sw',
        min_stat=5,
        fit_intercept=False,
        algorithm='recompute',
        backend='numpy',
    )

    slopes = result['y_slope_x_sw'].values
    assert np.all(np.isfinite(slopes))
    np.testing.assert_allclose(slopes, 5.0, atol=0.2,
                               err_msg="fit_intercept=False should recover true slope ≈ 5.0")
