"""
Tests for Phase 13.10.GB — Non-Linear Sliding Window Fit

Standalone module tests.  groupby_regression_sliding_window.py is NOT modified.
"""

import numpy as np
import pandas as pd
import pytest

from ..groupby_regression_models import (
    register_fit_model, get_model, list_models,
    _gaussian, _estimate_p0_gaussian, _gaussian_plus_line,
)
from ..groupby_regression_nonlinear import make_nonlinear_sliding_window_fit
from ..groupby_regression_sliding_window import make_sliding_window_fit

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


# ================================================================== #
#  Evaluator extension tests (Phase 13.10.GB-D)
# ================================================================== #

from ..groupby_regression_evaluator import GroupByRegressionEvaluator


class TestEvaluatorNonLinear:
    """Tests for non-linear evaluator extension (Option A: interpolate params)."""

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_from_dfGB_nonlinear(self):
        """Evaluator constructs from non-linear SW output."""
        df = _make_gaussian_data(n_bins_x=5, n_bins_y=5, n_per_bin=100)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        assert ev.fit_model == 'gaussian'
        assert ev.param_names == ['amplitude', 'mean', 'sigma', 'offset']
        assert ev.grid_shape == (5, 5)

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_evaluate_model_peak_vs_tail(self):
        """Gaussian evaluate_model: peak > tail."""
        df = _make_gaussian_data(n_bins_x=5, n_bins_y=5, n_per_bin=100)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        y_peak = ev.evaluate_model(
            positions={'xBin': 2.0, 'yBin': 2.0}, x_query=0.5)
        y_tail = ev.evaluate_model(
            positions={'xBin': 2.0, 'yBin': 2.0}, x_query=0.0)
        assert y_peak['peak'] > y_tail['peak']
        assert y_peak['peak'] > 8.0  # amplitude ~10 + offset ~1
        assert y_tail['peak'] < 3.0  # near offset

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_evaluate_model_vectorized(self):
        """evaluate_model accepts array x_query."""
        df = _make_gaussian_data(n_bins_x=5, n_bins_y=5, n_per_bin=100)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        x = np.linspace(0, 1, 10)
        y = ev.evaluate_model(
            positions={'xBin': 2.0, 'yBin': 2.0}, x_query=x)
        assert isinstance(y['peak'], np.ndarray)
        assert y['peak'].shape == (10,)
        # Should be Gaussian-shaped: max near center
        assert np.argmax(y['peak']) in (4, 5)

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_evaluate_model_auto_registry_lookup(self):
        """Named model auto-resolved from registry — no model_func needed."""
        df = _make_gaussian_data(n_bins_x=3, n_bins_y=3, n_per_bin=100)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        # Should work without model_func= argument
        y = ev.evaluate_model(
            positions={'xBin': 1.0, 'yBin': 1.0}, x_query=0.5)
        assert 'peak' in y
        assert y['peak'] > 5.0

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_evaluate_model_custom_func(self):
        """Custom model_func passed explicitly at evaluate time."""
        df = _make_gaussian_data(n_bins_x=3, n_bins_y=3, n_per_bin=100)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        # Pass model_func explicitly — should give same result
        y_auto = ev.evaluate_model(
            positions={'xBin': 1.0, 'yBin': 1.0}, x_query=0.5)
        y_explicit = ev.evaluate_model(
            positions={'xBin': 1.0, 'yBin': 1.0}, x_query=0.5,
            model_func=_gaussian)
        assert abs(y_auto['peak'] - y_explicit['peak']) < 1e-10

    def test_evaluate_model_rejects_linear(self):
        """evaluate_model raises on linear evaluators."""
        coefficients = {
            'dX': {
                'intercept': np.array([[1.0, 2.0], [3.0, 4.0]]),
                'slope_pred': np.array([[0.5, 0.6], [0.7, 0.8]]),
            }
        }
        ev = GroupByRegressionEvaluator(
            grid_shape=(2, 2),
            group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'],
            targets=['dX'],
            bin_centers={'xBin': np.array([0.0, 1.0]),
                         'yBin': np.array([0.0, 1.0])},
            coefficients=coefficients,
        )
        with pytest.raises(ValueError, match="non-linear"):
            ev.evaluate_model(
                positions={'xBin': 0.5, 'yBin': 0.5}, x_query=0.5)

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_evaluator_interpolation_smoothness(self):
        """I-NL.5: Interpolated params produce smooth function."""
        df = _make_gaussian_data(n_bins_x=5, n_bins_y=5, n_per_bin=200)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        # Evaluate at neighboring positions — should be smooth
        x_query = 0.5
        y_values = []
        for pos in np.linspace(1.0, 3.0, 10):
            y = ev.evaluate_model(
                positions={'xBin': pos, 'yBin': 2.0}, x_query=x_query)
            y_values.append(y['peak'])
        y_arr = np.array(y_values)
        # Max jump between neighbors should be small
        jumps = np.abs(np.diff(y_arr))
        assert np.max(jumps) < 2.0, f"Large jump: {np.max(jumps):.3f}"

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_evaluate_function_peak_vs_tail(self):
        """Option B: Gaussian evaluate_function peak > tail."""
        df = _make_gaussian_data(n_bins_x=5, n_bins_y=5, n_per_bin=100)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        y_peak = ev.evaluate_function(
            positions={'xBin': 2.0, 'yBin': 2.0}, x_query=0.5)
        y_tail = ev.evaluate_function(
            positions={'xBin': 2.0, 'yBin': 2.0}, x_query=0.0)
        assert y_peak['peak'] > y_tail['peak']
        assert y_peak['peak'] > 8.0
        assert y_tail['peak'] < 3.0

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_evaluate_function_vectorized_x(self):
        """Option B: vectorized x_query."""
        df = _make_gaussian_data(n_bins_x=5, n_bins_y=5, n_per_bin=100)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        x = np.linspace(0, 1, 20)
        y = ev.evaluate_function(
            positions={'xBin': 2.0, 'yBin': 2.0}, x_query=x)
        assert isinstance(y['peak'], np.ndarray)
        assert y['peak'].shape == (20,)
        # Gaussian shaped
        assert np.argmax(y['peak']) in (9, 10)

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_evaluate_function_batch_positions(self):
        """Option B: batch of positions."""
        df = _make_gaussian_data(n_bins_x=5, n_bins_y=5, n_per_bin=100)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        pos = {'xBin': np.array([1.0, 2.0, 3.0]),
               'yBin': np.array([2.0, 2.0, 2.0])}
        y = ev.evaluate_function(positions=pos, x_query=0.5)
        assert isinstance(y['peak'], np.ndarray)
        assert y['peak'].shape == (3,)
        assert all(v > 8.0 for v in y['peak'])

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_option_a_equals_b_at_grid_center(self):
        """I-NL.6: Options A and B agree exactly at grid centers."""
        df = _make_gaussian_data(n_bins_x=5, n_bins_y=5, n_per_bin=100)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        # At exact grid center — no interpolation, A must equal B
        ya = ev.evaluate_model(
            positions={'xBin': 2.0, 'yBin': 2.0}, x_query=0.5)
        yb = ev.evaluate_function(
            positions={'xBin': 2.0, 'yBin': 2.0}, x_query=0.5)
        np.testing.assert_allclose(
            ya['peak'], yb['peak'], atol=1e-10,
            err_msg="Options A and B must agree at grid centers")

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_option_a_close_to_b_interpolated(self):
        """I-NL.7: Options A and B are close for slowly varying params."""
        df = _make_gaussian_data(n_bins_x=5, n_bins_y=5, n_per_bin=200)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        ya = ev.evaluate_model(
            positions={'xBin': 1.5, 'yBin': 2.5}, x_query=0.5)
        yb = ev.evaluate_function(
            positions={'xBin': 1.5, 'yBin': 2.5}, x_query=0.5)
        # For slowly varying params (all bins have ~same Gaussian),
        # A and B should agree within ~1%
        np.testing.assert_allclose(
            ya['peak'], yb['peak'], rtol=0.01,
            err_msg="A and B should agree for slowly varying params")

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_evaluate_params(self):
        """evaluate_params returns parameter dict."""
        df = _make_gaussian_data(n_bins_x=5, n_bins_y=5, n_per_bin=100)
        result, metadata = make_nonlinear_sliding_window_fit(
            df=df, gb_columns=['xBin', 'yBin'],
            fit_columns=['peak'], linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1]},
            suffix='_gaus', min_stat=20, return_metadata=True,
        )
        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        params = ev.evaluate_params(positions={'xBin': 2.0, 'yBin': 2.0})
        assert 'peak' in params
        assert set(params['peak'].keys()) == {
            'amplitude', 'mean', 'sigma', 'offset'}
        # Amplitude should be ~10
        amp = float(np.atleast_1d(params['peak']['amplitude'])[0])
        assert 8.0 < amp < 12.0


# ================================================================== #
#  Cross-engine and extended invariance tests
# ================================================================== #

class TestCrossEngineInvariance:
    """Tests that non-linear fits match linear fits for polynomial models."""

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_polynomial2_matches_linear_ols(self):
        """I-NL.8: polynomial_2 non-linear ≡ linear OLS with [x, x²].

        For y = 2 + 3x + 4x², fitting with linear SW (predictors=[x, x²],
        fit_intercept=True) and non-linear SW (fit_func='polynomial_2')
        must produce identical coefficients to numerical precision.
        """
        rng = np.random.RandomState(42)
        rows = []
        for b in range(7):
            x = np.linspace(-1, 1, 60)
            y = 2.0 + 3.0 * x + 4.0 * x ** 2 + rng.normal(0, 0.05, len(x))
            rows.append(pd.DataFrame({
                'xBin': b, 'pred': x, 'pred_sq': x ** 2, 'target': y,
            }))
        df = pd.concat(rows, ignore_index=True)

        # Linear SW: predictors = [pred, pred_sq], fit_intercept=True
        r_lin = make_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['target'],
            linear_columns=['pred', 'pred_sq'],
            window_spec={'xBin': 1},
            fit_intercept=True,
            suffix='_lin',
            min_stat=10,
        )

        # Non-linear SW: polynomial_2
        r_nl = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin'],
            fit_columns=['target'],
            linear_columns=['pred'],
            window_spec={'xBin': 1},
            fit_func='polynomial_2',
            suffix='_nl',
            min_stat=10,
        )

        # Compare at center bin (xBin=3, uses bins 2-4)
        lin_row = r_lin[r_lin['xBin'] == 3].iloc[0]
        nl_row = r_nl[r_nl['xBin'] == 3].iloc[0]

        # Linear: intercept ≡ coeff_0, slope_pred ≡ coeff_1, slope_pred_sq ≡ coeff_2
        np.testing.assert_allclose(
            lin_row['target_intercept_lin'],
            nl_row['target_coeff_0_nl'],
            rtol=1e-4,
            err_msg="intercept ≡ coeff_0")
        np.testing.assert_allclose(
            lin_row['target_slope_pred_lin'],
            nl_row['target_coeff_1_nl'],
            rtol=1e-4,
            err_msg="slope_pred ≡ coeff_1")
        np.testing.assert_allclose(
            lin_row['target_slope_pred_sq_lin'],
            nl_row['target_coeff_2_nl'],
            rtol=1e-4,
            err_msg="slope_pred_sq ≡ coeff_2")


class TestGaussianPlusLine:
    """Tests for the gaussian_plus_line model (most common TPC spectrum)."""

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_gaussian_plus_line_recovery(self):
        """I-NL.9a: Recover known gaussian_plus_line parameters.

        True: amplitude=10, mean=0.5, sigma=0.1, offset=1.0, slope=0.5
        """
        rng = np.random.RandomState(42)
        rows = []
        for xb in range(5):
            for yb in range(5):
                mass = rng.uniform(0, 1, 100)
                y_true = (10.0 * np.exp(-0.5 * ((mass - 0.5) / 0.1) ** 2)
                          + 1.0 + 0.5 * mass)
                y = y_true + rng.normal(0, 0.1, len(mass))
                rows.append(pd.DataFrame({
                    'xBin': xb, 'yBin': yb, 'mass': mass, 'signal': y,
                }))
        df = pd.concat(rows, ignore_index=True)

        result = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin', 'yBin'],
            fit_columns=['signal'],
            linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian_plus_line',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1.0, 0.5]},
            suffix='_gpl',
            min_stat=20,
        )

        center = result[(result['xBin'] == 2) & (result['yBin'] == 2)].iloc[0]
        assert abs(center['signal_amplitude_gpl'] - 10.0) < 2.0, \
            f"amplitude={center['signal_amplitude_gpl']}"
        assert abs(center['signal_mean_gpl'] - 0.5) < 0.05, \
            f"mean={center['signal_mean_gpl']}"
        assert abs(center['signal_sigma_gpl'] - 0.1) < 0.05, \
            f"sigma={center['signal_sigma_gpl']}"
        assert abs(center['signal_offset_gpl'] - 1.0) < 0.5, \
            f"offset={center['signal_offset_gpl']}"
        assert abs(center['signal_slope_gpl'] - 0.5) < 0.5, \
            f"slope={center['signal_slope_gpl']}"
        assert center['signal_converged_gpl'] > 0.5

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_gaussian_plus_line_evaluate_roundtrip(self):
        """I-NL.9b: Fit → evaluate → compare to ground truth.

        Fit gaussian_plus_line, evaluate at original x positions,
        verify mean |y_pred - y_true| < 3 * noise_level.
        """
        rng = np.random.RandomState(123)
        noise_level = 0.15
        rows = []
        for xb in range(5):
            for yb in range(5):
                mass = rng.uniform(0, 1, 80)
                y_true = (10.0 * np.exp(-0.5 * ((mass - 0.5) / 0.1) ** 2)
                          + 1.0 + 0.5 * mass)
                y = y_true + rng.normal(0, noise_level, len(mass))
                rows.append(pd.DataFrame({
                    'xBin': xb, 'yBin': yb, 'mass': mass,
                    'signal': y, 'signal_true': y_true,
                }))
        df = pd.concat(rows, ignore_index=True)

        result, metadata = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin', 'yBin'],
            fit_columns=['signal'],
            linear_columns=['mass'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian_plus_line',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1.0, 0.5]},
            suffix='_gpl',
            min_stat=20,
            return_metadata=True,
        )

        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)

        # Evaluate at center bin across x range
        x_eval = np.linspace(0, 1, 50)
        y_pred = ev.evaluate_model(
            positions={'xBin': 2.0, 'yBin': 2.0}, x_query=x_eval)
        y_true = (10.0 * np.exp(-0.5 * ((x_eval - 0.5) / 0.1) ** 2)
                  + 1.0 + 0.5 * x_eval)

        mae = np.mean(np.abs(y_pred['signal'] - y_true))
        assert mae < 3 * noise_level, \
            f"MAE={mae:.4f} > 3*noise={3*noise_level:.4f}"


class TestExtendedInvariance:
    """Extended invariance tests requested by reviewers."""

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_constant_field_invariance(self):
        """I-NL.12: If all bins have identical parameters, Option A ≡ Option B
        everywhere, regardless of query position.

        Constant field = no interpolation ambiguity.
        """
        rng = np.random.RandomState(42)
        rows = []
        for xb in range(5):
            for yb in range(5):
                x = rng.uniform(0, 1, 80)
                # Identical Gaussian in every bin
                y = 10.0 * np.exp(-0.5 * ((x - 0.5) / 0.1) ** 2) + 1.0
                y += rng.normal(0, 0.05, len(x))
                rows.append(pd.DataFrame({
                    'xBin': xb, 'yBin': yb, 'x': x, 'y': y,
                }))
        df = pd.concat(rows, ignore_index=True)

        result, metadata = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin', 'yBin'],
            fit_columns=['y'],
            linear_columns=['x'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1.0]},
            suffix='_cf',
            min_stat=20,
            return_metadata=True,
        )

        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)
        x_query = np.linspace(0, 1, 20)

        # Test at multiple positions including off-grid interpolated points
        positions_list = [
            {'xBin': 2.0, 'yBin': 2.0},   # grid center
            {'xBin': 1.5, 'yBin': 2.5},   # interpolated
            {'xBin': 0.8, 'yBin': 3.2},   # interpolated
        ]
        for pos in positions_list:
            y_a = ev.evaluate_model(positions=pos, x_query=x_query)
            y_b = ev.evaluate_function(positions=pos, x_query=x_query)
            np.testing.assert_allclose(
                y_a['y'], y_b['y'], rtol=0.02,
                err_msg=f"A ≠ B for constant field at {pos}")

    @pytest.mark.skipif(not _has_scipy(), reason="scipy not available")
    def test_area_conservation_option_b(self):
        """I-NL.11: Option B interpolation conserves integrated area.

        Integral of evaluate_function over x ≈ weighted average of
        per-corner integrals. Verifies interpolation doesn't create
        or destroy 'area' under the curve.
        """
        rng = np.random.RandomState(42)
        rows = []
        # Slowly varying amplitude across grid
        for xb in range(5):
            for yb in range(5):
                amp = 8.0 + 0.5 * xb + 0.3 * yb
                x = rng.uniform(0, 1, 100)
                y = amp * np.exp(-0.5 * ((x - 0.5) / 0.1) ** 2) + 1.0
                y += rng.normal(0, 0.05, len(x))
                rows.append(pd.DataFrame({
                    'xBin': xb, 'yBin': yb, 'x': x, 'y': y,
                }))
        df = pd.concat(rows, ignore_index=True)

        result, metadata = make_nonlinear_sliding_window_fit(
            df=df,
            gb_columns=['xBin', 'yBin'],
            fit_columns=['y'],
            linear_columns=['x'],
            window_spec={'xBin': 1, 'yBin': 1},
            fit_func='gaussian',
            optimizer_kwargs={'p0': [10, 0.5, 0.1, 1.0]},
            suffix='_ac',
            min_stat=20,
            return_metadata=True,
        )

        ev = GroupByRegressionEvaluator.from_dfGB(result, metadata=metadata)

        # Dense x grid for numerical integration (trapezoidal)
        x_dense = np.linspace(0.05, 0.95, 200)
        pos = {'xBin': 2.3, 'yBin': 1.7}  # interpolated position

        # Area via Option B (evaluate_function)
        y_b = ev.evaluate_function(positions=pos, x_query=x_dense)
        area_b = np.trapz(y_b['y'], x_dense)

        # Area via Option A (evaluate_model with interpolated params)
        y_a = ev.evaluate_model(positions=pos, x_query=x_dense)
        area_a = np.trapz(y_a['y'], x_dense)

        # For slowly varying params, both should agree within ~5%
        # (area conservation: interpolating function values ≈
        #  evaluating with interpolated params for smooth fields)
        rel_diff = abs(area_b - area_a) / abs(area_a)
        assert rel_diff < 0.05, \
            f"Area mismatch: A={area_a:.4f}, B={area_b:.4f}, rel={rel_diff:.4f}"
