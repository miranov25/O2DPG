"""
Tests for ArrowComputeMapper (Phase 9a).

This test suite validates the PyArrow expression compilation layer.
All tests compare Arrow results against NumPy for correctness.

Author: Claude (Coder)
Date: 2025-12-01
"""

import pytest
import numpy as np
import math

# Optional PyArrow import
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pc = None

# Skip all tests if PyArrow not available
pytestmark = pytest.mark.skipif(
    not PYARROW_AVAILABLE,
    reason="PyArrow not available"
)


class TestArrowComputeMapper:
    """Test suite for Arrow expression compilation."""
    
    @pytest.fixture
    def mapper(self):
        """Import the mapper (skip if not available)."""
        # Try different import paths depending on how tests are run
        try:
            from AliasDataFrame._arrow_compute import ArrowComputeMapper
        except ImportError:
            from _arrow_compute import ArrowComputeMapper
        # Clear cache between tests
        ArrowComputeMapper.clear_cache()
        return ArrowComputeMapper
    
    @pytest.fixture
    def sample_context(self):
        """Create sample Arrow arrays for testing."""
        return {
            'px': pa.array([3.0, 4.0, 5.0]),
            'py': pa.array([4.0, 3.0, 12.0]),
            'x': pa.array([1.0, 2.0, 3.0]),
            'y': pa.array([0.5, 1.0, 1.5]),
            'a': pa.array([1.0, 2.0, 4.0]),
            'b': pa.array([2.0, 2.0, 2.0]),
        }
    
    # =========================================================================
    # Arithmetic Operators
    # =========================================================================
    
    @pytest.mark.parametrize("expr,expected", [
        ("x + y", [1.5, 3.0, 4.5]),
        ("x - y", [0.5, 1.0, 1.5]),
        ("x * y", [0.5, 2.0, 4.5]),
        ("x / y", [2.0, 2.0, 2.0]),
        ("x ** 2", [1.0, 4.0, 9.0]),
        ("x ** y", [1.0, 2.0, 5.196152]),  # 1^0.5, 2^1, 3^1.5
    ])
    def test_arithmetic_operators(self, mapper, sample_context, expr, expected):
        """Test basic arithmetic operators."""
        compiled = mapper.compile(expr)
        result = compiled(sample_context).to_pylist()
        np.testing.assert_allclose(result, expected, rtol=1e-5)
    
    def test_floor_division(self, mapper):
        """Test floor division operator (//)."""
        ctx = {'a': pa.array([7.0, 8.0, 9.0]), 'b': pa.array([2.0, 3.0, 4.0])}
        compiled = mapper.compile("a // b")
        result = compiled(ctx).to_pylist()
        expected = [3.0, 2.0, 2.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_modulo(self, mapper):
        """Test modulo operator (%)."""
        # Note: PyArrow has pc.mod but AST uses ast.Mod
        # This test verifies it's properly mapped
        pass  # TODO: Add when mod is needed
    
    def test_unary_negation(self, mapper, sample_context):
        """Test unary minus (-x)."""
        compiled = mapper.compile("-x")
        result = compiled(sample_context).to_pylist()
        expected = [-1.0, -2.0, -3.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_unary_plus(self, mapper, sample_context):
        """Test unary plus (+x) - should be no-op."""
        compiled = mapper.compile("+x")
        result = compiled(sample_context).to_pylist()
        expected = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # =========================================================================
    # Trigonometric Functions
    # =========================================================================
    
    @pytest.mark.parametrize("func", ['sin', 'cos', 'tan'])
    def test_trig_functions(self, mapper, func):
        """Test basic trig functions."""
        ctx = {'x': pa.array([0.0, np.pi/4, np.pi/2])}
        compiled = mapper.compile(f"{func}(x)")
        result = compiled(ctx).to_numpy()
        expected = getattr(np, func)(ctx['x'].to_numpy())
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    @pytest.mark.parametrize("func_alias,numpy_func", [
        ('arcsin', 'arcsin'),
        ('arccos', 'arccos'),
        ('arctan', 'arctan'),
        ('asin', 'arcsin'),
        ('acos', 'arccos'),
        ('atan', 'arctan'),
    ])
    def test_inverse_trig(self, mapper, func_alias, numpy_func):
        """Test inverse trig functions with both naming conventions."""
        ctx = {'x': pa.array([0.0, 0.5, 0.9])}  # Valid domain for all
        compiled = mapper.compile(f"{func_alias}(x)")
        result = compiled(ctx).to_numpy()
        expected = getattr(np, numpy_func)(ctx['x'].to_numpy())
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_atan2(self, mapper):
        """Test two-argument arctangent."""
        ctx = {
            'y': pa.array([1.0, 1.0, -1.0, 0.0]),
            'x': pa.array([1.0, -1.0, 1.0, 1.0])
        }
        compiled = mapper.compile("atan2(y, x)")
        result = compiled(ctx).to_numpy()
        expected = np.arctan2([1.0, 1.0, -1.0, 0.0], [1.0, -1.0, 1.0, 1.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_arctan2_alias(self, mapper):
        """Test arctan2 alias."""
        ctx = {'y': pa.array([1.0]), 'x': pa.array([1.0])}
        compiled = mapper.compile("arctan2(y, x)")
        result = compiled(ctx).to_numpy()
        expected = [np.pi / 4]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # =========================================================================
    # Hyperbolic Functions
    # =========================================================================
    
    @pytest.mark.parametrize("func", ['sinh', 'cosh', 'tanh'])
    def test_hyperbolic_functions(self, mapper, func):
        """Test hyperbolic functions."""
        ctx = {'x': pa.array([0.0, 0.5, 1.0])}
        compiled = mapper.compile(f"{func}(x)")
        result = compiled(ctx).to_numpy()
        expected = getattr(np, func)(ctx['x'].to_numpy())
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    @pytest.mark.parametrize("func_alias,numpy_func,test_values", [
        ('asinh', 'arcsinh', [0.0, 0.5, 1.0]),
        ('arcsinh', 'arcsinh', [0.0, 0.5, 1.0]),
        ('acosh', 'arccosh', [1.0, 2.0, 3.0]),      # acosh requires x >= 1
        ('arccosh', 'arccosh', [1.0, 2.0, 3.0]),
        ('atanh', 'arctanh', [0.0, 0.5, 0.9]),      # atanh requires |x| < 1
        ('arctanh', 'arctanh', [0.0, 0.5, 0.9]),
    ])
    def test_inverse_hyperbolic(self, mapper, func_alias, numpy_func, test_values):
        """Test inverse hyperbolic functions."""
        ctx = {'x': pa.array(test_values)}
        compiled = mapper.compile(f"{func_alias}(x)")
        result = compiled(ctx).to_numpy()
        expected = getattr(np, numpy_func)(np.array(test_values))
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # =========================================================================
    # Exponential/Log Functions
    # =========================================================================
    
    def test_exp(self, mapper):
        """Test exponential function."""
        ctx = {'x': pa.array([0.0, 1.0, 2.0])}
        compiled = mapper.compile("exp(x)")
        result = compiled(ctx).to_numpy()
        expected = np.exp([0.0, 1.0, 2.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_log_maps_to_ln(self, mapper):
        """
        CRITICAL: log() must map to natural log (ln in PyArrow).
        
        This is one of the key differences between NumPy and PyArrow.
        NumPy: np.log() = natural log
        PyArrow: pc.ln() = natural log, pc.log10() = base-10
        """
        ctx = {'x': pa.array([1.0, np.e, np.e**2])}
        compiled = mapper.compile("log(x)")
        result = compiled(ctx).to_numpy()
        expected = np.log([1.0, np.e, np.e**2])  # Should be [0, 1, 2]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_log10(self, mapper):
        """Test base-10 logarithm."""
        ctx = {'x': pa.array([1.0, 10.0, 100.0])}
        compiled = mapper.compile("log10(x)")
        result = compiled(ctx).to_numpy()
        expected = [0.0, 1.0, 2.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_log2(self, mapper):
        """Test base-2 logarithm."""
        ctx = {'x': pa.array([1.0, 2.0, 4.0, 8.0])}
        compiled = mapper.compile("log2(x)")
        result = compiled(ctx).to_numpy()
        expected = [0.0, 1.0, 2.0, 3.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_log1p(self, mapper):
        """Test log(1+x) for small x (more accurate than log(1+x))."""
        ctx = {'x': pa.array([0.0, 1e-10, 1.0])}
        compiled = mapper.compile("log1p(x)")
        result = compiled(ctx).to_numpy()
        expected = np.log1p([0.0, 1e-10, 1.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_sqrt(self, mapper):
        """Test square root."""
        ctx = {'x': pa.array([0.0, 1.0, 4.0, 9.0])}
        compiled = mapper.compile("sqrt(x)")
        result = compiled(ctx).to_numpy()
        expected = [0.0, 1.0, 2.0, 3.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_power(self, mapper):
        """Test power function."""
        ctx = {'x': pa.array([2.0, 3.0, 4.0])}
        compiled = mapper.compile("power(x, 2)")
        result = compiled(ctx).to_numpy()
        expected = [4.0, 9.0, 16.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_pow_alias(self, mapper):
        """Test pow() as alias for power()."""
        ctx = {'x': pa.array([2.0, 3.0])}
        compiled = mapper.compile("pow(x, 3)")
        result = compiled(ctx).to_numpy()
        expected = [8.0, 27.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # =========================================================================
    # Rounding Functions
    # =========================================================================
    
    @pytest.mark.parametrize("func,input_vals,expected", [
        ('floor', [1.5, 2.3, -1.5], [1.0, 2.0, -2.0]),
        ('ceil', [1.5, 2.3, -1.5], [2.0, 3.0, -1.0]),
        ('trunc', [1.5, 2.7, -1.5], [1.0, 2.0, -1.0]),
        ('abs', [-1.0, 2.0, -3.0], [1.0, 2.0, 3.0]),
        ('sign', [-5.0, 0.0, 3.0], [-1.0, 0.0, 1.0]),
    ])
    def test_rounding_functions(self, mapper, func, input_vals, expected):
        """Test rounding and sign functions."""
        ctx = {'x': pa.array(input_vals)}
        compiled = mapper.compile(f"{func}(x)")
        result = compiled(ctx).to_numpy()
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # =========================================================================
    # Special Functions: clip()
    # =========================================================================
    
    def test_clip(self, mapper):
        """Test clip(x, min, max)."""
        ctx = {'x': pa.array([1.0, 5.0, 10.0, 15.0, 20.0])}
        compiled = mapper.compile("clip(x, 3, 12)")
        result = compiled(ctx).to_pylist()
        expected = [3.0, 5.0, 10.0, 12.0, 12.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_clip_with_variables(self, mapper):
        """Test clip() with variable bounds."""
        ctx = {
            'x': pa.array([1.0, 5.0, 10.0]),
            'lo': pa.array([2.0, 2.0, 2.0]),
            'hi': pa.array([8.0, 8.0, 8.0]),
        }
        compiled = mapper.compile("clip(x, lo, hi)")
        result = compiled(ctx).to_pylist()
        expected = [2.0, 5.0, 8.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_clip_wrong_args_raises(self, mapper):
        """Test that clip() with wrong number of args raises."""
        with pytest.raises(ValueError, match="requires exactly 3 arguments"):
            mapper.compile("clip(x, 1)")
    
    # =========================================================================
    # Special Functions: where()
    # =========================================================================
    
    def test_where(self, mapper):
        """Test where(condition, x, y)."""
        ctx = {
            'cond': pa.array([True, False, True]),
            'x': pa.array([1.0, 2.0, 3.0]),
            'y': pa.array([10.0, 20.0, 30.0]),
        }
        compiled = mapper.compile("where(cond, x, y)")
        result = compiled(ctx).to_pylist()
        expected = [1.0, 20.0, 3.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_where_with_comparison(self, mapper):
        """Test where() with inline comparison."""
        ctx = {'x': pa.array([-1.0, 0.0, 1.0, 2.0])}
        compiled = mapper.compile("where(x > 0, x, 0)")
        result = compiled(ctx).to_pylist()
        expected = [0.0, 0.0, 1.0, 2.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # =========================================================================
    # Comparison Operators
    # =========================================================================
    
    @pytest.mark.parametrize("op,expected", [
        ("x > y", [True, True, True]),
        ("x < y", [False, False, False]),
        ("x >= y", [True, True, True]),
        ("x <= y", [False, False, False]),
        ("x == y", [False, False, False]),
        ("x != y", [True, True, True]),
    ])
    def test_comparison_operators(self, mapper, sample_context, op, expected):
        """Test comparison operators."""
        compiled = mapper.compile(op)
        result = compiled(sample_context).to_pylist()
        assert result == expected
    
    def test_chained_comparison_raises(self, mapper):
        """Test that chained comparisons raise."""
        with pytest.raises(ValueError, match="Chained comparisons not supported"):
            mapper.compile("a < b < c")
    
    # =========================================================================
    # Nested Expressions
    # =========================================================================
    
    def test_nested_sqrt_sum_squares(self, mapper, sample_context):
        """Test sqrt(px**2 + py**2) - common physics expression."""
        compiled = mapper.compile("sqrt(px**2 + py**2)")
        result = compiled(sample_context).to_pylist()
        expected = [5.0, 5.0, 13.0]  # 3-4-5, 4-3-5, 5-12-13 triangles
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_complex_expression(self, mapper):
        """Test complex nested expression."""
        ctx = {'x': pa.array([1.0, 2.0, 3.0])}
        compiled = mapper.compile("sin(x) ** 2 + cos(x) ** 2")
        result = compiled(ctx).to_numpy()
        # sin^2 + cos^2 = 1 always
        expected = [1.0, 1.0, 1.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_real_tpc_expression(self, mapper):
        """Test actual TPC calibration expression: sinh(dy/40.)"""
        ctx = {'dy': pa.array([0.0, 40.0, -40.0, 80.0])}
        compiled = mapper.compile("sinh(dy/40.)")
        result = compiled(ctx).to_numpy()
        expected = np.sinh(np.array([0.0, 1.0, -1.0, 2.0]))
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_nested_trig_expression(self, mapper):
        """Test nested trig: atan2(sin(x), cos(x))"""
        ctx = {'x': pa.array([0.0, np.pi/4, np.pi/2])}
        compiled = mapper.compile("atan2(sin(x), cos(x))")
        result = compiled(ctx).to_numpy()
        # atan2(sin(x), cos(x)) = x for x in [-pi, pi]
        expected = [0.0, np.pi/4, np.pi/2]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # =========================================================================
    # Constants
    # =========================================================================
    
    def test_numeric_constant(self, mapper):
        """Test numeric constants in expressions."""
        ctx = {'x': pa.array([1.0, 2.0, 3.0])}
        compiled = mapper.compile("x + 10")
        result = compiled(ctx).to_pylist()
        expected = [11.0, 12.0, 13.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_float_constant(self, mapper):
        """Test float constants."""
        ctx = {'x': pa.array([1.0, 2.0, 3.0])}
        compiled = mapper.compile("x / 2.5")
        result = compiled(ctx).to_numpy()
        expected = [0.4, 0.8, 1.2]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_numpy_pi_constant(self, mapper):
        """Test np.pi constant."""
        ctx = {'x': pa.array([1.0, 2.0])}
        compiled = mapper.compile("x * np.pi")
        result = compiled(ctx).to_numpy()
        expected = [np.pi, 2 * np.pi]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_math_e_constant(self, mapper):
        """Test math.e constant."""
        ctx = {'x': pa.array([1.0, 2.0])}
        compiled = mapper.compile("log(math.e ** x)")
        result = compiled(ctx).to_numpy()
        expected = [1.0, 2.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # =========================================================================
    # Attribute Access (subframe.column)
    # =========================================================================
    
    def test_subframe_column_dotted_key(self, mapper):
        """Test subframe.column with dotted key in context."""
        ctx = {
            'subframe.dy': pa.array([40.0, 80.0, 120.0]),
        }
        compiled = mapper.compile("sinh(subframe.dy / 40.)")
        result = compiled(ctx).to_numpy()
        expected = np.sinh([1.0, 2.0, 3.0])
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    def test_subframe_column_nested_dict(self, mapper):
        """Test subframe.column with nested dict in context."""
        ctx = {
            'calibration': {
                'offset': pa.array([1.0, 2.0, 3.0]),
            }
        }
        compiled = mapper.compile("calibration.offset * 2")
        result = compiled(ctx).to_pylist()
        expected = [2.0, 4.0, 6.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # =========================================================================
    # Conditional Expressions (ternary)
    # =========================================================================
    
    def test_ternary_expression(self, mapper):
        """Test x if condition else y syntax."""
        ctx = {'x': pa.array([-1.0, 0.0, 1.0])}
        compiled = mapper.compile("x if x > 0 else 0")
        result = compiled(ctx).to_pylist()
        expected = [0.0, 0.0, 1.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    # =========================================================================
    # Error Handling
    # =========================================================================
    
    def test_unsupported_function_raises(self, mapper):
        """Test that unsupported functions raise with helpful message."""
        with pytest.raises(ValueError, match="No PyArrow equivalent for function 'unknown_func'"):
            mapper.compile("unknown_func(x)")
    
    def test_syntax_error_raises(self, mapper):
        """Test that invalid syntax raises."""
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            mapper.compile("sqrt(")
    
    def test_missing_variable_raises(self, mapper):
        """Test that missing variable raises KeyError."""
        ctx = {'x': pa.array([1.0, 2.0])}
        compiled = mapper.compile("y + 1")
        with pytest.raises(KeyError, match="Variable 'y' not found"):
            compiled(ctx)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def test_is_supported(self, mapper):
        """Test is_supported() method."""
        assert mapper.is_supported("sqrt(x**2 + y**2)")
        assert mapper.is_supported("sinh(dy/40.)")
        assert mapper.is_supported("clip(x, 0, 1)")
        assert not mapper.is_supported("unknown_func(x)")
        assert not mapper.is_supported("invalid syntax (")
    
    def test_get_supported_functions(self, mapper):
        """Test get_supported_functions() returns sorted list."""
        funcs = mapper.get_supported_functions()
        assert isinstance(funcs, list)
        assert funcs == sorted(funcs)  # Sorted
        assert 'sin' in funcs
        assert 'sqrt' in funcs
        assert 'log' in funcs
    
    def test_clear_cache(self, mapper):
        """Test cache clearing."""
        # Compile something to populate cache
        mapper.compile("x + 1")
        assert len(mapper._compile_cache) > 0
        
        # Clear and verify
        mapper.clear_cache()
        assert len(mapper._compile_cache) == 0
    
    def test_cache_hit(self, mapper):
        """Test that repeated compilations hit cache."""
        expr = "sqrt(x**2 + y**2)"
        compiled1 = mapper.compile(expr)
        compiled2 = mapper.compile(expr)
        assert compiled1 is compiled2  # Same object from cache
    
    # =========================================================================
    # Type Promotion Tests (per reviewer feedback)
    # =========================================================================
    
    @pytest.mark.parametrize("a_type,b_type", [
        (pa.int32(), pa.int32()),
        (pa.float32(), pa.int32()),
        (pa.int64(), pa.float64()),
        (pa.float32(), pa.float64()),
    ])
    def test_type_promotion_division(self, mapper, a_type, b_type):
        """Test that division works across different type combinations."""
        ctx = {
            'a': pa.array([1, 2, 4], type=a_type),
            'b': pa.array([2, 2, 2], type=b_type),
        }
        compiled = mapper.compile("a / b")
        result = compiled(ctx).to_numpy()
        expected = [0.5, 1.0, 2.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)
    
    @pytest.mark.parametrize("a_type,b_type", [
        (pa.int32(), pa.int32()),
        (pa.int64(), pa.float32()),
    ])
    def test_type_promotion_multiplication(self, mapper, a_type, b_type):
        """Test that multiplication works across different type combinations."""
        ctx = {
            'a': pa.array([1, 2, 3], type=a_type),
            'b': pa.array([2, 3, 4], type=b_type),
        }
        compiled = mapper.compile("a * b")
        result = compiled(ctx).to_numpy()
        expected = [2.0, 6.0, 12.0]
        np.testing.assert_allclose(result, expected, rtol=1e-6)


class TestEvaluateExpressionArrow:
    """Test the convenience function."""
    
    @pytest.fixture
    def evaluate_fn(self):
        """Import the evaluate function."""
        try:
            from AliasDataFrame._arrow_compute import evaluate_expression_arrow
        except ImportError:
            from _arrow_compute import evaluate_expression_arrow
        return evaluate_expression_arrow
    
    def test_basic_evaluation(self, evaluate_fn):
        """Test basic expression evaluation."""
        ctx = {'x': np.array([1.0, 4.0, 9.0])}
        result = evaluate_fn("sqrt(x)", ctx)
        expected = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6)
    
    def test_fallback_called(self, evaluate_fn):
        """Test that fallback is called for unsupported expressions."""
        fallback_called = []
        
        def fallback(expr, ctx):
            fallback_called.append(expr)
            return np.array([42.0])
        
        ctx = {'x': np.array([1.0])}
        result = evaluate_fn("unknown_func(x)", ctx, fallback_fn=fallback)
        
        assert len(fallback_called) == 1
        assert fallback_called[0] == "unknown_func(x)"
        np.testing.assert_array_equal(result, [42.0])
    
    def test_pandas_series_input(self, evaluate_fn):
        """Test that pandas Series input is handled."""
        import pandas as pd
        
        ctx = {'x': pd.Series([1.0, 4.0, 9.0])}
        result = evaluate_fn("sqrt(x)", ctx)
        expected = [1.0, 2.0, 3.0]
        np.testing.assert_allclose(result.to_numpy(), expected, rtol=1e-6)


class TestIsArrowAvailable:
    """Test module-level availability check."""
    
    def test_is_arrow_available(self):
        """Test is_arrow_available() function."""
        try:
            from AliasDataFrame._arrow_compute import is_arrow_available
        except ImportError:
            from _arrow_compute import is_arrow_available
        # If we got this far, PyArrow is available
        assert is_arrow_available() == True
