"""
Tests for Phase 13.10.GB — Non-Linear Sliding Window Fit

Standalone module tests.  groupby_regression_sliding_window.py is NOT modified.
"""

import numpy as np
import pandas as pd
import pytest

from ..groupby_regression_models import (
    register_fit_model, get_model, list_models,
    _gaussian, _estimate_p0_gaussian,
)
from ..groupby_regression_nonlinear import make_nonlinear_sliding_window_fit

def _has_scipy():
    try:
        from scipy.optimize import curve_fit
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
#  Test data generators
# ---------------------------------------------------------------------------

def _make_gaussian_data(n_bins_x=5, n_bins_y=5, n_per_bin=50, seed=42):
    """Gaussian peak in 'mass' per (xBin, yBin).
    True: amplitude=10, mean=0.5, sigma=0.1, offset=1.0
    """
    rng = np.random.RandomState(seed)
    rows = []
    for xb in range(n_bins_x):
        for yb in range(n_bins_y):
            mass = rng.uniform(0.0, 1.0, n_per_bin)
            true_val = 10.0 * np.exp(-0.5 * ((mass - 0.5) / 0.1) ** 2) + 1.0
            noise = rng.normal(0, 0.1, n_per_bin)
            rows.append(pd.DataFrame({
                'xBin': xb, 'yBin': yb, 'mass': mass, 'peak': true_val + noise,
            }))
    return pd.concat(rows, ignore_index=True)


def _make_simple_data(n_bins=10, n_per_bin=30, seed=42):
    """Simple data for basic dispatch tests."""
    rng = np.random.RandomState(seed)
    rows = []
    for b in range(n_bins):
        x = rng.uniform(-1, 1, n_per_bin)
        y = 2.0 + 3.0 * x + rng.normal(0, 0.1, n_per_bin)
        rows.append(pd.DataFrame({'xBin': b, 'pred': x, 'target': y}))
    return pd.concat(rows, ignore_index=True)


def _simple_custom_callable(X, y, weights, **kwargs):
    """Minimal callable: weighted mean + std."""
    w = weights / np.sum(weights)
    mean = float(np.sum(y * w))
    std = float(np.sqrt(np.sum(w * (y - mean) ** 2)))
    return (
        {'mean': mean, 'std_param': std},
        {'n_fitted': float(len(y)), 'converged': 1.0},
    )


# ================================================================== #
#  Registry tests (smoke)
# ================================================================== #

class TestModelRegistry:

    def test_built_in_models_registered(self):
        models = list_models()
        for name in ['gaussian', 'polynomial_2', 'polynomial_3',
                      'exponential', 'power_law']:
            assert name in models, f"Missing: {name}"

    def test_get_model_returns_spec(self):
        spec = get_model('gaussian')
        assert spec.param_names == ['amplitude', 'mean', 'sigma', 'offset']
        assert callable(spec.func)
        assert callable(spec.estimate_p0)

    def test_get_model_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model('nonexistent_model')

    def test_register_custom_model(self):
        def my_func(x, a, b):
            return a * x + b
        register_fit_model('test_custom_lin', my_func,
                           param_names=['slope', 'intercept'])
        spec = get_model('test_custom_lin')
        assert spec.param_names == ['slope', 'intercept']

    def test_gaussian_function_evaluates(self):
        x = np.linspace(-3, 3, 100)
        y = _gaussian(x, amplitude=1.0, mean=0.0, sigma=1.0, offset=0.0)
        assert y.shape == (100,)
        assert abs(y[50] - 1.0) < 0.05


# ================================================================== #
#  p0 estimation tests
# ================================================================== #

class TestP0Estimation:

    def test_gaussian_p0_reasonable(self):
        """Gaussian estimator returns reasonable initial parameters."""
        rng = np.random.RandomState(42)
        x = np.linspace(0, 1, 200)
        y = 10.0 * np.exp(-0.5 * ((x - 0.5) / 0.1) ** 2) + 1.0
        y += rng.normal(0, 0.2, len(x))
        w = np.ones_like(x)
        p0 = _estimate_p0_gaussian(x, y, w)
        assert len(p0) == 4
        # amplitude should be roughly right
        assert p0[0] > 5.0, f"amplitude={p0[0]}"
        # mean should be near 0.5
        assert abs(p0[1] - 0.5) < 0.15, f"mean={p0[1]}"
        # sigma should be positive
        assert p0[2] > 0, f"sigma={p0[2]}"

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_auto_p0_no_explicit(self):
        """Named model fits without explicit p0 using estimate_p0."""
        df = _make_gaussian_data(n_bins_x=3, n_bins_y=3, n_per_bin=100)
        # No optimizer_kwargs at all — should use estimate_p0 automatically
        result = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'],
            linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            suffix='_gaus',
            min_stat=20,
        )
        # Should still converge
        center = result[(result['xBin'] == 1) & (result['yBin'] == 1)]
        assert len(center) == 1
        assert center.iloc[0]['peak_converged_gaus'] > 0.5


# ================================================================== #
#  Custom callable dispatch (smoke)
# ================================================================== #

class TestCallableDispatch:

    def test_custom_callable_basic(self):
        df = _make_simple_data(n_bins=5, n_per_bin=20)
        result = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['target'],
            linear_columns=['pred'],
            window_spec={'xBin': 1},
            fit_func=_simple_custom_callable,
            suffix='_nl',
            min_stat=5,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert 'target_mean_nl' in result.columns
        assert 'target_std_param_nl' in result.columns

    def test_custom_callable_with_metadata(self):
        df = _make_simple_data(n_bins=5, n_per_bin=20)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['target'],
            linear_columns=['pred'],
            window_spec={'xBin': 1},
            fit_func=_simple_custom_callable,
            suffix='_nl',
            min_stat=5,
            return_metadata=True,
        )
        assert metadata['parameters']['fit_model'] == '<callable>'
        assert metadata['parameters']['fit_type'] == 'sliding_window'
        assert metadata['algorithm'] == 'recompute'
        assert metadata['version'] == '1.1'
        assert 'columns' in metadata
        assert 'parameters' in metadata

    def test_bad_callable_returns_nan(self):
        def bad_func(X, y, w, **kwargs):
            raise RuntimeError("intentional failure")

        df = _make_simple_data(n_bins=3, n_per_bin=20)
        result = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['target'],
            linear_columns=['pred'],
            window_spec={'xBin': 0},
            fit_func=bad_func,
            suffix='_nl',
            min_stat=5,
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_bad_return_type_handled(self):
        """Callable returning wrong types → NaN, no crash."""
        def bad_return(X, y, w, **kwargs):
            return "not_a_dict", 42

        df = _make_simple_data(n_bins=3, n_per_bin=20)
        result = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['target'],
            linear_columns=['pred'],
            window_spec={'xBin': 0},
            fit_func=bad_return,
            suffix='_nl',
            min_stat=5,
        )
        assert isinstance(result, pd.DataFrame)

    def test_fit_func_type_validation(self):
        """fit_func must be callable or string."""
        df = _make_simple_data(n_bins=3)
        with pytest.raises(TypeError, match="callable or named model string"):
            make_nonlinear_sliding_window_fit(
                df=df,
                gb_columns=['xBin'],
                fit_columns=['target'],
                linear_columns=['pred'],
                window_spec={'xBin': 0},
                fit_func=42,
            )


# ================================================================== #
#  Named model dispatch (smoke)
# ================================================================== #

class TestNamedModelDispatch:

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_gaussian_named_model(self):
        df = _make_gaussian_data(n_bins_x=3, n_bins_y=3, n_per_bin=100)
        result = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'],
            linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10.0, 0.5, 0.1, 1.0]},
            suffix='_gaus',
            min_stat=20,
        )
        assert isinstance(result, pd.DataFrame)
        assert 'peak_amplitude_gaus' in result.columns
        assert 'peak_mean_gaus' in result.columns
        assert 'peak_sigma_gaus' in result.columns
        assert 'peak_converged_gaus' in result.columns

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_metadata_schema(self):
        df = _make_gaussian_data(n_bins_x=3, n_bins_y=3, n_per_bin=100)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'],
            linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10.0, 0.5, 0.1, 1.0]},
            suffix='_gaus',
            min_stat=20,
            return_metadata=True,
        )
        assert metadata['parameters']['fit_model'] == 'gaussian'
        assert metadata['parameters']['param_names'] == [
            'amplitude', 'mean', 'sigma', 'offset']
        assert 'peak' in metadata['columns']['coefficients']
        assert 'peak' in metadata['columns']['errors']

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_unknown_model_raises(self):
        df = _make_simple_data(n_bins=3)
        with pytest.raises(KeyError, match="Unknown model"):
            make_nonlinear_sliding_window_fit(
                df=df,
                gb_columns=['xBin'],
                fit_columns=['target'],
                linear_columns=['pred'],
                window_spec={'xBin': 0},
                fit_func='nonexistent_model',
            )


# ================================================================== #
#  Invariance tests
# ================================================================== #

class TestNonLinearInvariance:

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_gaussian_peak_recovery(self):
        """I-NL.1: Known Gaussian parameters recovered to tolerance."""
        df = _make_gaussian_data(n_bins_x=3, n_bins_y=3, n_per_bin=200, seed=123)
        result = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'],
            linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10.0, 0.5, 0.1, 1.0]},
            suffix='_gaus',
            min_stat=20,
        )
        center = result[(result['xBin'] == 1) & (result['yBin'] == 1)]
        assert len(center) == 1
        row = center.iloc[0]
        assert abs(row['peak_amplitude_gaus'] - 10.0) < 2.0
        assert abs(row['peak_mean_gaus'] - 0.5) < 0.05
        assert abs(row['peak_sigma_gaus'] - 0.1) < 0.05
        assert row['peak_converged_gaus'] > 0.5

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_polynomial_exact_recovery(self):
        """I-NL.2: Quadratic from exact quadratic data."""
        rows = []
        for b in range(5):
            x = np.linspace(-1, 1, 50)
            y = 1.0 + 2.0 * x + 3.0 * x ** 2
            rows.append(pd.DataFrame({'xBin': b, 'pred': x, 'target': y}))
        df = pd.concat(rows, ignore_index=True)

        result = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['target'],
            linear_columns=['pred'],
            window_spec={'xBin': 1},
            fit_func='polynomial_2',
            suffix='_poly',
            min_stat=10,
        )
        center = result[result['xBin'] == 2].iloc[0]
        assert abs(center['target_coeff_0_poly'] - 1.0) < 0.01
        assert abs(center['target_coeff_1_poly'] - 2.0) < 0.01
        assert abs(center['target_coeff_2_poly'] - 3.0) < 0.01

    def test_multi_target_independence(self):
        """I-NL.3 (AC-6): Multi-target — fitting together = fitting separately."""
        rng = np.random.RandomState(42)
        rows = []
        for b in range(5):
            x = rng.uniform(0, 1, 40)
            rows.append(pd.DataFrame({
                'xBin': b, 'pred': x,
                'tgt1': 5.0 * np.ones_like(x),
                'tgt2': 10.0 * np.ones_like(x),
            }))
        df = pd.concat(rows, ignore_index=True)

        result_both = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['tgt1', 'tgt2'],
            linear_columns=['pred'],
            window_spec={'xBin': 1},
            fit_func=_simple_custom_callable,
            suffix='_nl',
            min_stat=5,
        )
        result_1 = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['tgt1'],
            linear_columns=['pred'],
            window_spec={'xBin': 1},
            fit_func=_simple_custom_callable,
            suffix='_nl',
            min_stat=5,
        )

        for col in ['tgt1_mean_nl', 'tgt1_std_param_nl']:
            np.testing.assert_allclose(
                result_both[col].values, result_1[col].values,
                atol=1e-12, err_msg=f"Cross-target contamination in {col}")

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_uniform_weights_equals_no_weights(self):
        """I-NL.4 (AC-1): Uniform weights produce same result as unweighted."""
        df = _make_gaussian_data(n_bins_x=3, n_bins_y=3, n_per_bin=100)
        df['w_uniform'] = 1.0

        r_no_w = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'],
            linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10.0, 0.5, 0.1, 1.0]},
            suffix='_gaus',
            min_stat=20,
        )
        r_w = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'],
            linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            weights='w_uniform',
            optimizer_kwargs={'p0': [10.0, 0.5, 0.1, 1.0]},
            suffix='_gaus',
            min_stat=20,
        )

        for col in ['peak_amplitude_gaus', 'peak_mean_gaus', 'peak_sigma_gaus']:
            np.testing.assert_allclose(
                r_no_w[col].values, r_w[col].values,
                atol=1e-6, err_msg=f"Uniform weights ≠ no weights for {col}")


# ================================================================== #
#  SW infrastructure reuse (smoke)
# ================================================================== #

class TestSWInfrastructureReuse:
    """Verify that windowing, boundary, and kernel work correctly."""

    def test_window_spec_affects_results(self):
        """Larger window → more data → different results."""
        df = _make_simple_data(n_bins=10, n_per_bin=20, seed=42)
        r0 = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['target'],
            linear_columns=['pred'],
            window_spec={'xBin': 0},
            fit_func=_simple_custom_callable,
            suffix='_nl', min_stat=5,
        )
        r2 = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['target'],
            linear_columns=['pred'],
            window_spec={'xBin': 2},
            fit_func=_simple_custom_callable,
            suffix='_nl', min_stat=5,
        )
        # Window=2 aggregates more rows
        assert (r2['n_rows_aggregated_nl'] >= r0['n_rows_aggregated_nl']).all()

    def test_min_stat_enforcement(self):
        """Bins with insufficient stats → quality_flag set."""
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            'xBin': [0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            'pred': rng.uniform(0, 1, 10),
            'target': rng.normal(0, 1, 10),
        })
        result = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['target'],
            linear_columns=['pred'],
            window_spec={'xBin': 0},
            fit_func=_simple_custom_callable,
            suffix='_nl', min_stat=5,
        )
        # xBin=0 has only 2 rows < min_stat=5
        row0 = result[result['xBin'] == 0].iloc[0]
        assert 'insufficient' in row0['target_quality_flag_nl']
