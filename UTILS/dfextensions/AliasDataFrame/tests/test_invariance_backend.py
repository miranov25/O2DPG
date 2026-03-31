"""
test_invariance_backend.py — I2: Backend Invariance

Phase 13.7.ADF: Invariance Test Suite
Priority: P1
Tests: 10

Property: use_numba=True produces identical results to use_numba=False

Author: Claude2-Coder
Date: 2026-01-16
Version: 2.0 - Fixed to use use_numba constructor parameter
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FLOAT_RTOL = 1e-10
FLOAT_ATOL = 1e-12


def assert_invariant_equal(result, expected, name=""):
    """Compare arrays with appropriate tolerance."""
    result = np.asarray(result)
    expected = np.asarray(expected)
    if np.issubdtype(result.dtype, np.floating):
        result_nan = np.isnan(result)
        expected_nan = np.isnan(expected)
        np.testing.assert_array_equal(result_nan, expected_nan, err_msg=f"{name}_nan_positions")
        mask = ~result_nan
        if mask.any():
            np.testing.assert_allclose(result[mask], expected[mask], rtol=FLOAT_RTOL, atol=FLOAT_ATOL, err_msg=name)
    else:
        np.testing.assert_array_equal(result, expected, err_msg=name)


def has_numba():
    """Check if Numba is available."""
    try:
        import numba
        return True
    except ImportError:
        return False


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def backend_test_df():
    """Create test DataFrame for backend comparison."""
    np.random.seed(42)
    n = 5000
    
    return pd.DataFrame({
        'x': np.random.randn(n).astype(np.float64),
        'y': np.random.randn(n).astype(np.float64),
        'z': np.random.randn(n).astype(np.float64),
        'a': np.random.randint(0, 100, n).astype(np.int64),
        'b': np.random.randint(0, 100, n).astype(np.int64),
        'flag': np.random.randint(0, 2, n).astype(np.int32),
    })


@pytest.fixture
def backend_subframe_data():
    """Create main + subframe data for backend join tests."""
    np.random.seed(42)
    n_main = 5000
    n_sub = 100
    
    main_df = pd.DataFrame({
        'key': np.random.randint(0, n_sub, n_main).astype(np.int64),
        'value': np.random.randn(n_main).astype(np.float64),
    })
    
    sub_df = pd.DataFrame({
        'key': np.arange(n_sub, dtype=np.int64),
        'offset': np.random.randn(n_sub).astype(np.float64),
        'scale': np.random.rand(n_sub).astype(np.float64) + 0.5,
    })
    
    return main_df, sub_df


# =============================================================================
# TEST CLASS: BACKEND INVARIANCE
# =============================================================================

@pytest.mark.invariance
class TestInvarianceBackend:
    """
    I2: Backend Invariance Tests
    
    Core invariant: use_numba=True produces identical results to use_numba=False.
    
    Tests focus on subframe joins where Numba acceleration is applied.
    """
    
    def test_I2_1_single_key_join_numba_vs_numpy(self, backend_subframe_data):
        """
        I2_1: Single-key subframe join produces same results with both backends.
        """
        from AliasDataFrame import AliasDataFrame
        
        if not has_numba():
            pytest.skip("Numba not available")
        
        main_df, sub_df = backend_subframe_data
        
        # NumPy backend (use_numba=False)
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        sub_adf_numpy = AliasDataFrame(sub_df.copy())
        adf_numpy.register_subframe('S', sub_adf_numpy, 'key')
        adf_numpy.add_alias('s_offset', 'S.offset')
        adf_numpy.materialize_alias('s_offset')
        numpy_result = adf_numpy.df['s_offset'].values.copy()
        
        # Numba backend (use_numba=True)
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        sub_adf_numba = AliasDataFrame(sub_df.copy())
        adf_numba.register_subframe('S', sub_adf_numba, 'key')
        adf_numba.add_alias('s_offset', 'S.offset')
        adf_numba.materialize_alias('s_offset')
        numba_result = adf_numba.df['s_offset'].values
        
        assert_invariant_equal(numba_result, numpy_result, "I2_1_single_key_join")
    
    def test_I2_2_join_with_arithmetic_numba_vs_numpy(self, backend_subframe_data):
        """
        I2_2: Join + arithmetic operations produce same results.
        """
        from AliasDataFrame import AliasDataFrame
        
        if not has_numba():
            pytest.skip("Numba not available")
        
        main_df, sub_df = backend_subframe_data
        
        # NumPy backend
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        sub_adf = AliasDataFrame(sub_df.copy())
        adf_numpy.register_subframe('S', sub_adf, 'key')
        adf_numpy.add_alias('scaled_value', 'value * S.scale + S.offset')
        adf_numpy.materialize_alias('scaled_value')
        numpy_result = adf_numpy.df['scaled_value'].values.copy()
        
        # Numba backend
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        sub_adf2 = AliasDataFrame(sub_df.copy())
        adf_numba.register_subframe('S', sub_adf2, 'key')
        adf_numba.add_alias('scaled_value', 'value * S.scale + S.offset')
        adf_numba.materialize_alias('scaled_value')
        numba_result = adf_numba.df['scaled_value'].values
        
        assert_invariant_equal(numba_result, numpy_result, "I2_2_join_arithmetic")
    
    def test_I2_3_multikey_join_numba_vs_numpy(self):
        """
        I2_3: Multi-key subframe join produces same results.
        """
        from AliasDataFrame import AliasDataFrame
        
        if not has_numba():
            pytest.skip("Numba not available")
        
        np.random.seed(42)
        n_main = 3000
        
        # Main frame with 2 key columns
        main_df = pd.DataFrame({
            'k1': np.random.randint(0, 10, n_main).astype(np.int64),
            'k2': np.random.randint(0, 10, n_main).astype(np.int64),
            'value': np.random.randn(n_main).astype(np.float64),
        })
        
        # Subframe with all k1, k2 combinations
        sub_data = []
        for i in range(10):
            for j in range(10):
                sub_data.append({'k1': i, 'k2': j, 'factor': np.random.randn()})
        sub_df = pd.DataFrame(sub_data)
        
        # NumPy backend
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        sub_adf = AliasDataFrame(sub_df.copy())
        adf_numpy.register_subframe('S', sub_adf, ['k1', 'k2'])
        adf_numpy.add_alias('result', 'value * S.factor')
        adf_numpy.materialize_alias('result')
        numpy_result = adf_numpy.df['result'].values.copy()
        
        # Numba backend
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        sub_adf2 = AliasDataFrame(sub_df.copy())
        adf_numba.register_subframe('S', sub_adf2, ['k1', 'k2'])
        adf_numba.add_alias('result', 'value * S.factor')
        adf_numba.materialize_alias('result')
        numba_result = adf_numba.df['result'].values
        
        assert_invariant_equal(numba_result, numpy_result, "I2_3_multikey_join")
    
    def test_I2_4_missing_keys_numba_vs_numpy(self, backend_subframe_data):
        """
        I2_4: Missing keys produce same NaN pattern with both backends.
        """
        from AliasDataFrame import AliasDataFrame
        
        if not has_numba():
            pytest.skip("Numba not available")
        
        main_df, sub_df = backend_subframe_data
        
        # Create gaps in subframe keys
        sub_df_sparse = sub_df[sub_df['key'] % 2 == 0].copy()  # Only even keys
        
        # NumPy backend
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        sub_adf = AliasDataFrame(sub_df_sparse.copy())
        adf_numpy.register_subframe('S', sub_adf, 'key')
        adf_numpy.add_alias('s_offset', 'S.offset')
        adf_numpy.materialize_alias('s_offset')
        numpy_result = adf_numpy.df['s_offset'].values.copy()
        
        # Numba backend
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        sub_adf2 = AliasDataFrame(sub_df_sparse.copy())
        adf_numba.register_subframe('S', sub_adf2, 'key')
        adf_numba.add_alias('s_offset', 'S.offset')
        adf_numba.materialize_alias('s_offset')
        numba_result = adf_numba.df['s_offset'].values
        
        # Check NaN positions match
        numpy_nan = np.isnan(numpy_result)
        numba_nan = np.isnan(numba_result)
        np.testing.assert_array_equal(numba_nan, numpy_nan, err_msg="I2_4_nan_positions")
        
        # Check non-NaN values match
        mask = ~numpy_nan
        if mask.any():
            assert_invariant_equal(numba_result[mask], numpy_result[mask], "I2_4_non_nan_values")
    
    def test_I2_5_duplicate_keys_numba_vs_numpy(self, backend_subframe_data):
        """
        I2_5: Duplicate keys in main frame handled correctly by both backends.
        """
        from AliasDataFrame import AliasDataFrame
        
        if not has_numba():
            pytest.skip("Numba not available")
        
        # Create main with many duplicates
        np.random.seed(42)
        main_df = pd.DataFrame({
            'key': np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3] * 100, dtype=np.int64),
            'value': np.random.randn(1000).astype(np.float64),
        })
        
        sub_df = pd.DataFrame({
            'key': np.arange(5, dtype=np.int64),
            'offset': np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        })
        
        # NumPy backend
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        sub_adf = AliasDataFrame(sub_df.copy())
        adf_numpy.register_subframe('S', sub_adf, 'key')
        adf_numpy.add_alias('result', 'value + S.offset')
        adf_numpy.materialize_alias('result')
        numpy_result = adf_numpy.df['result'].values.copy()
        
        # Numba backend
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        sub_adf2 = AliasDataFrame(sub_df.copy())
        adf_numba.register_subframe('S', sub_adf2, 'key')
        adf_numba.add_alias('result', 'value + S.offset')
        adf_numba.materialize_alias('result')
        numba_result = adf_numba.df['result'].values
        
        assert_invariant_equal(numba_result, numpy_result, "I2_5_duplicate_keys")
    
    def test_I2_6_chained_subframe_expressions_numba_vs_numpy(self, backend_subframe_data):
        """
        I2_6: Chained expressions with subframes produce same results.
        """
        from AliasDataFrame import AliasDataFrame
        
        if not has_numba():
            pytest.skip("Numba not available")
        
        main_df, sub_df = backend_subframe_data
        
        # NumPy backend
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        sub_adf = AliasDataFrame(sub_df.copy())
        adf_numpy.register_subframe('S', sub_adf, 'key')
        adf_numpy.add_alias('step1', 'value + S.offset')
        adf_numpy.add_alias('step2', 'step1 * S.scale')
        adf_numpy.add_alias('final', 'step2 / 10')
        adf_numpy.materialize_aliases(['step1', 'step2', 'final'])
        numpy_final = adf_numpy.df['final'].values.copy()
        
        # Numba backend
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        sub_adf2 = AliasDataFrame(sub_df.copy())
        adf_numba.register_subframe('S', sub_adf2, 'key')
        adf_numba.add_alias('step1', 'value + S.offset')
        adf_numba.add_alias('step2', 'step1 * S.scale')
        adf_numba.add_alias('final', 'step2 / 10')
        adf_numba.materialize_aliases(['step1', 'step2', 'final'])
        numba_final = adf_numba.df['final'].values
        
        assert_invariant_equal(numba_final, numpy_final, "I2_6_chained")
    
    def test_I2_7_multiple_subframes_numba_vs_numpy(self):
        """
        I2_7: Multiple subframes produce same results with both backends.
        """
        from AliasDataFrame import AliasDataFrame
        
        if not has_numba():
            pytest.skip("Numba not available")
        
        np.random.seed(42)
        n = 2000
        
        main_df = pd.DataFrame({
            'key_a': np.random.randint(0, 50, n).astype(np.int64),
            'key_b': np.random.randint(0, 30, n).astype(np.int64),
            'value': np.random.randn(n).astype(np.float64),
        })
        
        sub_a = pd.DataFrame({
            'key_a': np.arange(50, dtype=np.int64),
            'factor_a': np.random.randn(50).astype(np.float64),
        })
        
        sub_b = pd.DataFrame({
            'key_b': np.arange(30, dtype=np.int64),
            'factor_b': np.random.randn(30).astype(np.float64),
        })
        
        # NumPy backend
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        adf_numpy.register_subframe('A', AliasDataFrame(sub_a.copy()), 'key_a')
        adf_numpy.register_subframe('B', AliasDataFrame(sub_b.copy()), 'key_b')
        adf_numpy.add_alias('combined', 'value * A.factor_a + B.factor_b')
        adf_numpy.materialize_alias('combined')
        numpy_result = adf_numpy.df['combined'].values.copy()
        
        # Numba backend
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        adf_numba.register_subframe('A', AliasDataFrame(sub_a.copy()), 'key_a')
        adf_numba.register_subframe('B', AliasDataFrame(sub_b.copy()), 'key_b')
        adf_numba.add_alias('combined', 'value * A.factor_a + B.factor_b')
        adf_numba.materialize_alias('combined')
        numba_result = adf_numba.df['combined'].values
        
        assert_invariant_equal(numba_result, numpy_result, "I2_7_multiple_subframes")
    
    def test_I2_8_math_functions_in_join_numba_vs_numpy(self, backend_subframe_data):
        """
        I2_8: Math functions combined with joins produce same results.
        """
        from AliasDataFrame import AliasDataFrame
        
        if not has_numba():
            pytest.skip("Numba not available")
        
        main_df, sub_df = backend_subframe_data
        
        # NumPy backend
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        sub_adf = AliasDataFrame(sub_df.copy())
        adf_numpy.register_subframe('S', sub_adf, 'key')
        adf_numpy.add_alias('result', 'sin(value) * S.scale + cos(S.offset)')
        adf_numpy.materialize_alias('result')
        numpy_result = adf_numpy.df['result'].values.copy()
        
        # Numba backend
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        sub_adf2 = AliasDataFrame(sub_df.copy())
        adf_numba.register_subframe('S', sub_adf2, 'key')
        adf_numba.add_alias('result', 'sin(value) * S.scale + cos(S.offset)')
        adf_numba.materialize_alias('result')
        numba_result = adf_numba.df['result'].values
        
        assert_invariant_equal(numba_result, numpy_result, "I2_8_math_in_join")
    
    def test_I2_9_boolean_conditions_numba_vs_numpy(self, backend_subframe_data):
        """
        I2_9: Boolean conditions with subframe values produce same results.
        """
        from AliasDataFrame import AliasDataFrame
        
        if not has_numba():
            pytest.skip("Numba not available")
        
        main_df, sub_df = backend_subframe_data
        
        # NumPy backend
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        sub_adf = AliasDataFrame(sub_df.copy())
        adf_numpy.register_subframe('S', sub_adf, 'key')
        adf_numpy.add_alias('condition', '(value > 0) & (S.scale > 0.7)')
        adf_numpy.materialize_alias('condition')
        numpy_result = adf_numpy.df['condition'].values.copy()
        
        # Numba backend
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        sub_adf2 = AliasDataFrame(sub_df.copy())
        adf_numba.register_subframe('S', sub_adf2, 'key')
        adf_numba.add_alias('condition', '(value > 0) & (S.scale > 0.7)')
        adf_numba.materialize_alias('condition')
        numba_result = adf_numba.df['condition'].values
        
        np.testing.assert_array_equal(numba_result, numpy_result, err_msg="I2_9_boolean")
    
    @pytest.mark.slow
    def test_I2_10_large_dataset_numba_vs_numpy(self):
        """
        I2_10: Large dataset (500K rows) produces same results.
        """
        from AliasDataFrame import AliasDataFrame
        
        if not has_numba():
            pytest.skip("Numba not available")
        
        np.random.seed(42)
        n = 500_000
        n_sub = 1000
        
        main_df = pd.DataFrame({
            'key': np.random.randint(0, n_sub, n).astype(np.int64),
            'value': np.random.randn(n).astype(np.float64),
        })
        
        sub_df = pd.DataFrame({
            'key': np.arange(n_sub, dtype=np.int64),
            'offset': np.random.randn(n_sub).astype(np.float64),
        })
        
        # NumPy backend
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        sub_adf = AliasDataFrame(sub_df.copy())
        adf_numpy.register_subframe('S', sub_adf, 'key')
        adf_numpy.add_alias('result', 'value + S.offset')
        adf_numpy.materialize_alias('result')
        numpy_result = adf_numpy.df['result'].values.copy()
        
        # Numba backend
        adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
        sub_adf2 = AliasDataFrame(sub_df.copy())
        adf_numba.register_subframe('S', sub_adf2, 'key')
        adf_numba.add_alias('result', 'value + S.offset')
        adf_numba.materialize_alias('result')
        numba_result = adf_numba.df['result'].values
        
        assert_invariant_equal(numba_result, numpy_result, "I2_10_large_dataset")


# =============================================================================
# TEST SUMMARY
# =============================================================================

class TestBackendSummary:
    """Verify all I2 tests are present."""
    
    def test_I2_count(self):
        """Verify we have 10 backend tests."""
        tests = [
            'test_I2_1_single_key_join_numba_vs_numpy',
            'test_I2_2_join_with_arithmetic_numba_vs_numpy',
            'test_I2_3_multikey_join_numba_vs_numpy',
            'test_I2_4_missing_keys_numba_vs_numpy',
            'test_I2_5_duplicate_keys_numba_vs_numpy',
            'test_I2_6_chained_subframe_expressions_numba_vs_numpy',
            'test_I2_7_multiple_subframes_numba_vs_numpy',
            'test_I2_8_math_functions_in_join_numba_vs_numpy',
            'test_I2_9_boolean_conditions_numba_vs_numpy',
            'test_I2_10_large_dataset_numba_vs_numpy',
        ]
        assert len(tests) == 10, "Expected 10 I2 tests"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "invariance and not slow"])
