"""
Tests for PyArrow Expression Evaluation (Phase 9c).

This test suite validates the Arrow-accelerated expression evaluation
integrated into AliasDataFrame._eval_in_namespace().

Author: Claude (Coder)
Date: 2025-12-01
"""

import pytest
import numpy as np
import pandas as pd
import time

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


class TestArrowExpressionEvaluation:
    """Test Arrow-accelerated expression evaluation."""
    
    @pytest.fixture
    def large_adf(self):
        """Create AliasDataFrame with enough rows to trigger Arrow path."""
        # Import here to handle path issues
        try:
            from AliasDataFrame import AliasDataFrame
        except ImportError:
            from AliasDataFrame.AliasDataFrame import AliasDataFrame
        
        n = 50_000  # Above NUMBA_MIN_ROWS threshold
        df = pd.DataFrame({
            'x': np.random.randn(n).astype(np.float64),
            'y': np.random.randn(n).astype(np.float64),
            'z': np.random.randn(n).astype(np.float64),
            'a': np.random.randint(0, 100, n).astype(np.int32),
            'b': np.random.randint(1, 10, n).astype(np.int32),
        })
        return AliasDataFrame(df)
    
    @pytest.fixture
    def small_adf(self):
        """Create small AliasDataFrame that won't trigger Arrow path."""
        try:
            from AliasDataFrame import AliasDataFrame
        except ImportError:
            from AliasDataFrame.AliasDataFrame import AliasDataFrame
        
        n = 100  # Below threshold
        df = pd.DataFrame({
            'x': np.random.randn(n),
            'y': np.random.randn(n),
        })
        return AliasDataFrame(df)
    
    def test_arrow_info_includes_compute(self, large_adf):
        """Test that arrow_info includes compute_available flag."""
        info = large_adf.arrow_info
        assert 'compute_available' in info
        assert info['compute_available'] == True
        assert info['available'] == True
        assert info['enabled'] == True
    
    def test_simple_arithmetic(self, large_adf):
        """Test simple arithmetic expression with Arrow."""
        large_adf.add_alias('sum_xy', 'x + y')
        large_adf.materialize_alias('sum_xy')
        
        # Verify result
        expected = large_adf.df['x'] + large_adf.df['y']
        np.testing.assert_allclose(
            large_adf.df['sum_xy'].values, 
            expected.values, 
            rtol=1e-10
        )
    
    def test_complex_arithmetic(self, large_adf):
        """Test complex arithmetic expression."""
        large_adf.add_alias('complex', 'x * y + z / 2.0')
        large_adf.materialize_alias('complex')
        
        expected = large_adf.df['x'] * large_adf.df['y'] + large_adf.df['z'] / 2.0
        np.testing.assert_allclose(
            large_adf.df['complex'].values,
            expected.values,
            rtol=1e-10
        )
    
    def test_sqrt_expression(self, large_adf):
        """Test sqrt function."""
        # Use absolute values to avoid sqrt of negative
        large_adf.df['x_abs'] = np.abs(large_adf.df['x'])
        large_adf.add_alias('sqrt_x', 'sqrt(x_abs)')
        large_adf.materialize_alias('sqrt_x')
        
        expected = np.sqrt(np.abs(large_adf.df['x']))
        np.testing.assert_allclose(
            large_adf.df['sqrt_x'].values,
            expected.values,
            rtol=1e-6
        )
    
    def test_trig_expression(self, large_adf):
        """Test trigonometric functions."""
        large_adf.add_alias('sin_x', 'sin(x)')
        large_adf.add_alias('cos_y', 'cos(y)')
        large_adf.materialize_alias('sin_x')
        large_adf.materialize_alias('cos_y')
        
        np.testing.assert_allclose(
            large_adf.df['sin_x'].values,
            np.sin(large_adf.df['x']).values,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            large_adf.df['cos_y'].values,
            np.cos(large_adf.df['y']).values,
            rtol=1e-10
        )
    
    def test_exp_log_expression(self, large_adf):
        """Test exp and log functions."""
        # Use small values to avoid overflow
        large_adf.df['x_small'] = large_adf.df['x'] * 0.1
        large_adf.df['x_pos'] = np.abs(large_adf.df['x']) + 0.1
        
        large_adf.add_alias('exp_x', 'exp(x_small)')
        large_adf.add_alias('log_x', 'log(x_pos)')
        large_adf.materialize_alias('exp_x')
        large_adf.materialize_alias('log_x')
        
        np.testing.assert_allclose(
            large_adf.df['exp_x'].values,
            np.exp(large_adf.df['x_small']).values,
            rtol=1e-6
        )
        np.testing.assert_allclose(
            large_adf.df['log_x'].values,
            np.log(large_adf.df['x_pos']).values,
            rtol=1e-6
        )
    
    def test_power_expression(self, large_adf):
        """Test power operator."""
        large_adf.add_alias('x_squared', 'x ** 2')
        large_adf.add_alias('x_cubed', 'x ** 3')
        large_adf.materialize_alias('x_squared')
        large_adf.materialize_alias('x_cubed')
        
        np.testing.assert_allclose(
            large_adf.df['x_squared'].values,
            (large_adf.df['x'] ** 2).values,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            large_adf.df['x_cubed'].values,
            (large_adf.df['x'] ** 3).values,
            rtol=1e-10
        )
    
    def test_comparison_expression(self, large_adf):
        """Test comparison operators."""
        large_adf.add_alias('x_gt_y', 'x > y')
        large_adf.materialize_alias('x_gt_y')
        
        expected = (large_adf.df['x'] > large_adf.df['y']).astype(bool)
        result = large_adf.df['x_gt_y'].astype(bool)
        np.testing.assert_array_equal(result.values, expected.values)
    
    def test_division_type_promotion(self, large_adf):
        """Test that integer division produces float results."""
        large_adf.add_alias('a_div_b', 'a / b')
        large_adf.materialize_alias('a_div_b')
        
        # Result should be float
        assert np.issubdtype(large_adf.df['a_div_b'].dtype, np.floating)
        
        # Values should be correct
        expected = large_adf.df['a'].astype(float) / large_adf.df['b'].astype(float)
        np.testing.assert_allclose(
            large_adf.df['a_div_b'].values,
            expected.values,
            rtol=1e-10
        )
    
    def test_nested_expression(self, large_adf):
        """Test nested function calls."""
        large_adf.df['x_pos'] = np.abs(large_adf.df['x']) + 0.1
        large_adf.df['y_pos'] = np.abs(large_adf.df['y']) + 0.1
        
        large_adf.add_alias('nested', 'sqrt(x_pos**2 + y_pos**2)')
        large_adf.materialize_alias('nested')
        
        expected = np.sqrt(large_adf.df['x_pos']**2 + large_adf.df['y_pos']**2)
        np.testing.assert_allclose(
            large_adf.df['nested'].values,
            expected.values,
            rtol=1e-6
        )
    
    def test_fallback_for_unsupported(self, large_adf):
        """Test that unsupported expressions fall back to eval()."""
        # 'int' is a type cast function not in ArrowComputeMapper
        large_adf.add_alias('int_x', 'int(x * 10)')
        
        # Should still work via fallback
        large_adf.materialize_alias('int_x')
        assert 'int_x' in large_adf.df.columns
    
    def test_small_array_uses_eval(self, small_adf):
        """Test that small arrays use eval() path (below threshold)."""
        small_adf.add_alias('sum_xy', 'x + y')
        small_adf.materialize_alias('sum_xy')
        
        expected = small_adf.df['x'] + small_adf.df['y']
        np.testing.assert_allclose(
            small_adf.df['sum_xy'].values,
            expected.values,
            rtol=1e-10
        )
    
    def test_disabled_arrow(self, large_adf):
        """Test that Arrow can be disabled."""
        try:
            from AliasDataFrame import AliasDataFrame
        except ImportError:
            from AliasDataFrame.AliasDataFrame import AliasDataFrame
        
        # Create new ADF with Arrow disabled
        adf_no_arrow = AliasDataFrame(large_adf.df.copy(), use_arrow=False)
        assert adf_no_arrow.arrow_info['enabled'] == False
        
        # Should still work via eval()
        adf_no_arrow.add_alias('sum_xy', 'x + y')
        adf_no_arrow.materialize_alias('sum_xy')
        
        expected = adf_no_arrow.df['x'] + adf_no_arrow.df['y']
        np.testing.assert_allclose(
            adf_no_arrow.df['sum_xy'].values,
            expected.values,
            rtol=1e-10
        )
    
    def test_hyperbolic_functions(self, large_adf):
        """Test hyperbolic functions (require PyArrow >= 14)."""
        # Use small values to avoid overflow
        large_adf.df['x_small'] = large_adf.df['x'] * 0.1
        
        large_adf.add_alias('sinh_x', 'sinh(x_small)')
        large_adf.add_alias('cosh_x', 'cosh(x_small)')
        large_adf.materialize_alias('sinh_x')
        large_adf.materialize_alias('cosh_x')
        
        np.testing.assert_allclose(
            large_adf.df['sinh_x'].values,
            np.sinh(large_adf.df['x_small']).values,
            rtol=1e-6
        )
        np.testing.assert_allclose(
            large_adf.df['cosh_x'].values,
            np.cosh(large_adf.df['x_small']).values,
            rtol=1e-6
        )


class TestArrowExpressionPerformance:
    """Performance tests for Arrow expression evaluation."""
    
    @pytest.fixture
    def perf_adf(self):
        """Create large AliasDataFrame for performance testing."""
        try:
            from AliasDataFrame import AliasDataFrame
        except ImportError:
            from AliasDataFrame.AliasDataFrame import AliasDataFrame
        
        n = 2_000_000
        df = pd.DataFrame({
            'x': np.random.randn(n).astype(np.float64),
            'y': np.random.randn(n).astype(np.float64),
            'z': np.random.randn(n).astype(np.float64),
        })
        return AliasDataFrame(df)
    
    def test_expression_speed(self, perf_adf):
        """Test that expression evaluation completes in reasonable time."""
        perf_adf.add_alias('result', 'sqrt(x**2 + y**2 + z**2)')
        
        t0 = time.perf_counter()
        perf_adf.materialize_alias('result')
        elapsed = time.perf_counter() - t0
        
        # Should be reasonably fast (< 1 second for 2M rows)
        assert elapsed < 1.0, f"Expression eval took {elapsed:.3f}s, expected < 1.0s"
        print(f"\nExpression eval: 2M rows in {elapsed*1000:.1f}ms")
    
    def test_multiple_expressions_speed(self, perf_adf):
        """Test multiple expression evaluation speed."""
        perf_adf.add_alias('r', 'sqrt(x**2 + y**2)')
        perf_adf.add_alias('theta', 'arctan2(y, x)')
        perf_adf.add_alias('phi', 'arctan2(z, r)')
        
        t0 = time.perf_counter()
        perf_adf.materialize_alias('r')
        perf_adf.materialize_alias('theta')
        perf_adf.materialize_alias('phi')
        elapsed = time.perf_counter() - t0
        
        # 3 expressions should still be fast
        assert elapsed < 2.0, f"3 expressions took {elapsed:.3f}s, expected < 2.0s"
        print(f"\n3 expressions: 2M rows in {elapsed*1000:.1f}ms")


class TestArrowExpressionEdgeCases:
    """Edge case tests for Arrow expression evaluation."""
    
    @pytest.fixture
    def edge_adf(self):
        """Create AliasDataFrame with edge case data."""
        try:
            from AliasDataFrame import AliasDataFrame
        except ImportError:
            from AliasDataFrame.AliasDataFrame import AliasDataFrame
        
        n = 50_000
        df = pd.DataFrame({
            'x': np.concatenate([
                np.array([0.0, np.inf, -np.inf, np.nan]),
                np.random.randn(n - 4)
            ]),
            'y': np.concatenate([
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.random.randn(n - 4)
            ]),
        })
        return AliasDataFrame(df)
    
    def test_inf_handling(self, edge_adf):
        """Test that inf values are handled correctly."""
        edge_adf.add_alias('x_plus_one', 'x + 1')
        edge_adf.materialize_alias('x_plus_one')
        
        result = edge_adf.df['x_plus_one']
        assert np.isinf(result.iloc[1])  # inf + 1 = inf
        assert np.isinf(result.iloc[2])  # -inf + 1 = -inf
    
    def test_nan_propagation(self, edge_adf):
        """Test that NaN propagates correctly."""
        edge_adf.add_alias('x_times_y', 'x * y')
        edge_adf.materialize_alias('x_times_y')
        
        result = edge_adf.df['x_times_y']
        assert np.isnan(result.iloc[3])  # nan * anything = nan
    
    def test_zero_handling(self, edge_adf):
        """Test zero in expressions."""
        edge_adf.add_alias('y_div_x', 'y / x')
        edge_adf.materialize_alias('y_div_x')
        
        result = edge_adf.df['y_div_x']
        # 1.0 / 0.0 = inf
        assert np.isinf(result.iloc[0])


# =============================================================================
# If running standalone
# =============================================================================
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
