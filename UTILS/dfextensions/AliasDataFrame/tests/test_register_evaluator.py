"""
Tests for Phase 13.10.ADF: register_evaluator

Comprehensive tests covering:
  - Basic registration and evaluation via alias
  - Overwrite / collision policy
  - Multi-predictor evaluators
  - Argument count validation
  - Type validation (no .evaluate method)
  - Composition with other aliases and polynomials
  - Invariance: alias result == direct evaluator result
  - Edge cases: NaN handling, empty arrays, large arrays
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


# =========================================================================
# Mock Evaluators
# =========================================================================

class MockLinearEvaluator:
    """Linear evaluator: result = sum(slope[col] * positions[col])."""

    def __init__(self, coefficients):
        self.coefficients = coefficients  # {'col_name': slope}

    def evaluate(self, positions):
        result = np.zeros(len(next(iter(positions.values()))))
        for col, slope in self.coefficients.items():
            result += slope * positions[col]
        return result


class MockMultiPredictorEvaluator:
    """Returns dict of multiple predictions."""

    def __init__(self, coefficients_dict):
        # {'dy': {'x': 0.1, 'y': 0.2}, 'dz': {'x': 0.3, 'y': 0.4}}
        self.coefficients_dict = coefficients_dict

    def evaluate(self, positions):
        result = {}
        for pred_name, coeffs in self.coefficients_dict.items():
            arr = np.zeros(len(next(iter(positions.values()))))
            for col, slope in coeffs.items():
                arr += slope * positions[col]
            result[pred_name] = arr
        return result


class MockSingleDictEvaluator:
    """Returns dict with single predictor."""

    def __init__(self, slope):
        self.slope = slope

    def evaluate(self, positions):
        x = next(iter(positions.values()))
        return {'prediction': self.slope * x}


class NotAnEvaluator:
    """Object without .evaluate() method — for type checking tests."""
    pass


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def adf_basic():
    """Simple ADF for evaluator tests."""
    df = pd.DataFrame({
        'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        'y': np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        'z': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
    })
    return AliasDataFrame(df)


@pytest.fixture
def adf_large():
    """Larger ADF for stress tests."""
    n = 10000
    df = pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n),
        'z': np.random.randn(n),
    })
    return AliasDataFrame(df)


# =========================================================================
# Basic Registration and Evaluation
# =========================================================================

class TestRegisterEvaluatorBasic:
    """Basic registration and alias evaluation."""

    def test_register_and_evaluate(self, adf_basic):
        """Register evaluator, add alias, materialize — correct values."""
        adf = adf_basic
        evaluator = MockLinearEvaluator({'x': 2.0, 'y': 0.1})
        adf.register_evaluator('myEval', evaluator, ['x', 'y'])
        adf.add_alias('result', 'myEval(x, y)')
        adf.materialize_alias('result')

        expected = 2.0 * adf.df['x'].values + 0.1 * adf.df['y'].values
        np.testing.assert_allclose(adf.df['result'].values, expected, atol=1e-10)

    def test_register_single_column(self, adf_basic):
        """Evaluator with single coordinate column."""
        adf = adf_basic
        evaluator = MockLinearEvaluator({'x': 3.0})
        adf.register_evaluator('scale', evaluator, ['x'])
        adf.add_alias('scaled', 'scale(x)')
        adf.materialize_alias('scaled')

        expected = 3.0 * adf.df['x'].values
        np.testing.assert_allclose(adf.df['scaled'].values, expected, atol=1e-10)

    def test_register_three_columns(self, adf_basic):
        """Evaluator with three coordinate columns."""
        adf = adf_basic
        evaluator = MockLinearEvaluator({'x': 1.0, 'y': 0.5, 'z': -2.0})
        adf.register_evaluator('f3d', evaluator, ['x', 'y', 'z'])
        adf.add_alias('result', 'f3d(x, y, z)')
        adf.materialize_alias('result')

        expected = 1.0 * adf.df['x'].values + 0.5 * adf.df['y'].values - 2.0 * adf.df['z'].values
        np.testing.assert_allclose(adf.df['result'].values, expected, atol=1e-10)

    def test_evaluator_in_expression(self, adf_basic):
        """Evaluator used in a compound expression."""
        adf = adf_basic
        evaluator = MockLinearEvaluator({'x': 1.0})
        adf.register_evaluator('f', evaluator, ['x'])
        adf.add_alias('result', 'x - f(x)')
        adf.materialize_alias('result')

        # f(x) = 1.0 * x, so x - f(x) = 0
        np.testing.assert_allclose(adf.df['result'].values, 0.0, atol=1e-10)


# =========================================================================
# Overwrite and Collision
# =========================================================================

class TestRegisterEvaluatorCollision:
    """Collision and overwrite handling."""

    def test_collision_raises(self, adf_basic):
        """Duplicate name without overwrite raises ValueError."""
        adf = adf_basic
        eval1 = MockLinearEvaluator({'x': 1.0})
        adf.register_evaluator('f', eval1, ['x'])
        with pytest.raises(ValueError, match="already registered"):
            adf.register_evaluator('f', eval1, ['x'])

    def test_overwrite_replaces(self, adf_basic):
        """overwrite=True replaces existing evaluator."""
        adf = adf_basic
        eval1 = MockLinearEvaluator({'x': 1.0})
        eval2 = MockLinearEvaluator({'x': 5.0})
        adf.register_evaluator('f', eval1, ['x'])
        adf.register_evaluator('f', eval2, ['x'], overwrite=True)
        adf.add_alias('result', 'f(x)')
        adf.materialize_alias('result')

        expected = 5.0 * adf.df['x'].values
        np.testing.assert_allclose(adf.df['result'].values, expected, atol=1e-10)

    def test_no_collision_with_columns(self, adf_basic):
        """Evaluator name doesn't collide with DataFrame column names."""
        adf = adf_basic
        # 'x' is both a column and now a function name — function takes precedence in eval
        evaluator = MockLinearEvaluator({'y': 1.0})
        adf.register_evaluator('x_func', evaluator, ['y'])
        adf.add_alias('result', 'x_func(y)')
        adf.materialize_alias('result')

        expected = 1.0 * adf.df['y'].values
        np.testing.assert_allclose(adf.df['result'].values, expected, atol=1e-10)


# =========================================================================
# Type Validation
# =========================================================================

class TestRegisterEvaluatorValidation:
    """Input validation."""

    def test_no_evaluate_method_raises(self, adf_basic):
        """Object without .evaluate() raises TypeError."""
        adf = adf_basic
        with pytest.raises(TypeError, match="must have .evaluate"):
            adf.register_evaluator('bad', NotAnEvaluator(), ['x'])

    def test_wrong_arg_count_raises(self, adf_basic):
        """Wrong number of arguments in alias raises ValueError."""
        adf = adf_basic
        evaluator = MockLinearEvaluator({'x': 1.0, 'y': 2.0})
        adf.register_evaluator('f', evaluator, ['x', 'y'])
        adf.add_alias('result', 'f(x)')  # only 1 arg, needs 2
        with pytest.raises(ValueError, match="expects 2 arguments"):
            adf.materialize_alias('result')


# =========================================================================
# Multi-Predictor
# =========================================================================

class TestRegisterEvaluatorMultiPredictor:
    """Multi-predictor evaluator handling."""

    def test_multi_predictor_with_selection(self, adf_basic):
        """predictor_columns selects correct output from multi-predictor."""
        adf = adf_basic
        evaluator = MockMultiPredictorEvaluator({
            'dy': {'x': 0.1, 'y': 0.2},
            'dz': {'x': 0.3, 'y': 0.4},
        })
        adf.register_evaluator('corr_dy', evaluator, ['x', 'y'],
                                predictor_columns=['dy'])
        adf.add_alias('result_dy', 'corr_dy(x, y)')
        adf.materialize_alias('result_dy')

        expected = 0.1 * adf.df['x'].values + 0.2 * adf.df['y'].values
        np.testing.assert_allclose(adf.df['result_dy'].values, expected, atol=1e-10)

    def test_multi_predictor_dz(self, adf_basic):
        """Second predictor selected correctly."""
        adf = adf_basic
        evaluator = MockMultiPredictorEvaluator({
            'dy': {'x': 0.1, 'y': 0.2},
            'dz': {'x': 0.3, 'y': 0.4},
        })
        adf.register_evaluator('corr_dz', evaluator, ['x', 'y'],
                                predictor_columns=['dz'])
        adf.add_alias('result_dz', 'corr_dz(x, y)')
        adf.materialize_alias('result_dz')

        expected = 0.3 * adf.df['x'].values + 0.4 * adf.df['y'].values
        np.testing.assert_allclose(adf.df['result_dz'].values, expected, atol=1e-10)

    def test_multi_predictor_no_selection_raises(self, adf_basic):
        """Multi-predictor without predictor_columns raises ValueError."""
        adf = adf_basic
        evaluator = MockMultiPredictorEvaluator({
            'dy': {'x': 0.1},
            'dz': {'x': 0.3},
        })
        adf.register_evaluator('corr', evaluator, ['x'])
        adf.add_alias('result', 'corr(x)')
        with pytest.raises(ValueError, match="multiple predictors"):
            adf.materialize_alias('result')

    def test_single_dict_predictor_no_selection_ok(self, adf_basic):
        """Single-predictor dict evaluator works without predictor_columns."""
        adf = adf_basic
        evaluator = MockSingleDictEvaluator(slope=2.5)
        adf.register_evaluator('f', evaluator, ['x'])
        adf.add_alias('result', 'f(x)')
        adf.materialize_alias('result')

        expected = 2.5 * adf.df['x'].values
        np.testing.assert_allclose(adf.df['result'].values, expected, atol=1e-10)


# =========================================================================
# Composition
# =========================================================================

class TestRegisterEvaluatorComposition:
    """Composition with other aliases and evaluators."""

    def test_two_evaluators_summed(self, adf_basic):
        """Two evaluators composed via alias addition."""
        adf = adf_basic
        eval1 = MockLinearEvaluator({'x': 1.0})
        eval2 = MockLinearEvaluator({'x': 0.5})
        adf.register_evaluator('f1', eval1, ['x'])
        adf.register_evaluator('f2', eval2, ['x'])
        adf.add_alias('total', 'f1(x) + f2(x)')
        adf.materialize_alias('total')

        expected = 1.5 * adf.df['x'].values
        np.testing.assert_allclose(adf.df['total'].values, expected, atol=1e-10)

    def test_evaluator_minus_column(self, adf_basic):
        """Evaluator subtracted from column."""
        adf = adf_basic
        evaluator = MockLinearEvaluator({'x': 1.0})
        adf.register_evaluator('correction', evaluator, ['x'])
        adf.add_alias('residual', 'y - correction(x)')
        adf.materialize_alias('residual')

        expected = adf.df['y'].values - 1.0 * adf.df['x'].values
        np.testing.assert_allclose(adf.df['residual'].values, expected, atol=1e-10)

    def test_evaluator_with_alias_dependency(self, adf_basic):
        """Evaluator using column that is itself an alias."""
        adf = adf_basic
        adf.add_alias('x_scaled', 'x * 2')
        adf.materialize_alias('x_scaled')
        evaluator = MockLinearEvaluator({'x_scaled': 0.5})
        adf.register_evaluator('f', evaluator, ['x_scaled'])
        adf.add_alias('result', 'f(x_scaled)')
        adf.materialize_alias('result')

        expected = 0.5 * (adf.df['x'].values * 2)
        np.testing.assert_allclose(adf.df['result'].values, expected, atol=1e-10)

    def test_chained_corrections(self, adf_basic):
        """Iterative correction pattern: y - corr0 - corr1."""
        adf = adf_basic
        eval0 = MockLinearEvaluator({'x': 0.3})
        eval1 = MockLinearEvaluator({'x': 0.1})
        adf.register_evaluator('corr0', eval0, ['x'])
        adf.register_evaluator('corr1', eval1, ['x'])
        adf.add_alias('y_I0', 'y - corr0(x)')
        adf.add_alias('y_I1', 'y_I0 - corr1(x)')
        adf.materialize_aliases(names=['y_I0', 'y_I1'])

        expected = adf.df['y'].values - 0.3 * adf.df['x'].values - 0.1 * adf.df['x'].values
        np.testing.assert_allclose(adf.df['y_I1'].values, expected, atol=1e-10)


# =========================================================================
# Invariance Tests
# =========================================================================

@pytest.mark.invariance
class TestRegisterEvaluatorInvariance:
    """Alias evaluation must match direct evaluator call."""

    def test_invariance_alias_vs_direct(self, adf_basic):
        """Alias result == direct evaluator.evaluate() call."""
        adf = adf_basic
        evaluator = MockLinearEvaluator({'x': 2.5, 'y': -0.3})
        adf.register_evaluator('f', evaluator, ['x', 'y'])
        adf.add_alias('via_alias', 'f(x, y)')
        adf.materialize_alias('via_alias')

        # Direct call
        direct = evaluator.evaluate({
            'x': adf.df['x'].values.astype(np.float64),
            'y': adf.df['y'].values.astype(np.float64),
        })

        np.testing.assert_allclose(adf.df['via_alias'].values, direct, atol=1e-10)

    def test_invariance_large_data(self, adf_large):
        """Invariance holds for larger dataset."""
        adf = adf_large
        evaluator = MockLinearEvaluator({'x': 1.5, 'y': -0.7, 'z': 0.3})
        adf.register_evaluator('f', evaluator, ['x', 'y', 'z'])
        adf.add_alias('via_alias', 'f(x, y, z)')
        adf.materialize_alias('via_alias')

        direct = evaluator.evaluate({
            'x': adf.df['x'].values.astype(np.float64),
            'y': adf.df['y'].values.astype(np.float64),
            'z': adf.df['z'].values.astype(np.float64),
        })

        np.testing.assert_allclose(adf.df['via_alias'].values, direct, atol=1e-10)

    def test_invariance_multi_predictor(self, adf_basic):
        """Multi-predictor: alias matches selected predictor from direct call."""
        adf = adf_basic
        evaluator = MockMultiPredictorEvaluator({
            'dy': {'x': 0.1, 'y': 0.2},
            'dz': {'x': 0.3, 'y': 0.4},
        })
        adf.register_evaluator('corr', evaluator, ['x', 'y'],
                                predictor_columns=['dz'])
        adf.add_alias('via_alias', 'corr(x, y)')
        adf.materialize_alias('via_alias')

        direct = evaluator.evaluate({
            'x': adf.df['x'].values.astype(np.float64),
            'y': adf.df['y'].values.astype(np.float64),
        })

        np.testing.assert_allclose(adf.df['via_alias'].values, direct['dz'], atol=1e-10)


# =========================================================================
# Edge Cases
# =========================================================================

class TestRegisterEvaluatorEdgeCases:
    """Edge cases and robustness."""

    def test_evaluator_with_nan_input(self, adf_basic):
        """Evaluator handles NaN in input gracefully."""
        adf = adf_basic
        adf.df.loc[2, 'x'] = np.nan
        evaluator = MockLinearEvaluator({'x': 2.0})
        adf.register_evaluator('f', evaluator, ['x'])
        adf.add_alias('result', 'f(x)')
        adf.materialize_alias('result')

        # NaN propagates (evaluator doesn't special-case it)
        assert np.isnan(adf.df['result'].values[2])
        assert not np.isnan(adf.df['result'].values[0])

    def test_schema_stores_contract(self, adf_basic):
        """Schema stores evaluator type and coord_columns."""
        adf = adf_basic
        evaluator = MockLinearEvaluator({'x': 1.0})
        adf.register_evaluator('myFunc', evaluator, ['x'])

        schema_entry = adf._schema['registered_functions']['myFunc']
        assert schema_entry['type'] == 'evaluator'
        assert schema_entry['coord_columns'] == ['x']
        assert schema_entry['predictor_columns'] is None

    def test_schema_with_predictor_columns(self, adf_basic):
        """Schema stores predictor_columns when specified."""
        adf = adf_basic
        evaluator = MockMultiPredictorEvaluator({'dy': {'x': 1.0}, 'dz': {'x': 2.0}})
        adf.register_evaluator('f', evaluator, ['x'], predictor_columns=['dy'])

        schema_entry = adf._schema['registered_functions']['f']
        assert schema_entry['predictor_columns'] == ['dy']

    def test_backward_compatibility(self, adf_basic):
        """Existing aliases (sqrt, etc.) still work after registering evaluator."""
        adf = adf_basic
        evaluator = MockLinearEvaluator({'x': 1.0})
        adf.register_evaluator('f', evaluator, ['x'])

        adf.add_alias('root_x', 'sqrt(x)')
        adf.materialize_alias('root_x')
        np.testing.assert_allclose(adf.df['root_x'].values,
                                    np.sqrt(adf.df['x'].values), atol=1e-10)
