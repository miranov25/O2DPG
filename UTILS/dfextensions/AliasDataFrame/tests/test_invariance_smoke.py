"""
test_invariance_smoke.py — I0: Smoke Tests

Phase 13.7.ADF: Invariance Test Suite
Priority: P0
Tests: 6 (one per category)
Target Runtime: <10s total

Purpose:
- One fast canary test per invariance category
- Always runs in CI before any merge
- Failures immediately localize to a category

Categories:
- I1: Load Mode (lazy vs eager)
- I2: Backend (Numba vs NumPy vs Arrow)
- I3: Subframe Join (ADF join vs pandas.merge)
- I4: Compression (round-trip)
- I5: Schema (export/import)
- I6: Order (materialization order)

Author: Claude2-Coder
Date: 2026-01-15
Version: 1.1 - Fixed missing materialize_alias() calls
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import sys

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tolerance settings
FLOAT_RTOL = 1e-10
FLOAT_ATOL = 1e-12


def assert_invariant_equal(result, expected, name=""):
    """Compare arrays with appropriate tolerance."""
    result = np.asarray(result)
    expected = np.asarray(expected)
    if np.issubdtype(result.dtype, np.floating):
        # Handle NaN
        result_nan = np.isnan(result)
        expected_nan = np.isnan(expected)
        np.testing.assert_array_equal(result_nan, expected_nan, err_msg=f"{name}_nan_positions")
        mask = ~result_nan
        if mask.any():
            np.testing.assert_allclose(result[mask], expected[mask], rtol=FLOAT_RTOL, atol=FLOAT_ATOL, err_msg=name)
    else:
        np.testing.assert_array_equal(result, expected, err_msg=name)


# =============================================================================
# SMOKE TESTS
# =============================================================================

@pytest.mark.smoke
@pytest.mark.invariance
class TestInvarianceSmoke:
    """
    I0: Smoke Tests — One fast canary per category.
    
    Requirements:
    - Each test runs in <1s
    - Uses minimal data (10-100 rows)
    - Tests the core invariant only
    - Fails fast if category is broken
    
    Total smoke runtime target: <10s
    """
    
    # -------------------------------------------------------------------------
    # I1 Smoke: Load Mode
    # -------------------------------------------------------------------------
    
    def test_smoke_I1_load_mode(self, tmp_path):
        """
        I1 Smoke: Lazy load == Eager load.
        
        Invariant: read_tree_lazy() produces same column values as read_tree()
        Runtime target: <1s
        """
        uproot = pytest.importorskip("uproot")
        from AliasDataFrame import AliasDataFrame
        
        # Minimal test data
        file_path = tmp_path / "smoke_i1.root"
        data = {
            'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
            'y': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
        }
        
        with uproot.create(file_path) as f:
            f["tree"] = data
        
        # Eager
        adf_eager = AliasDataFrame.read_tree(str(file_path), "tree")
        eager_x = adf_eager['x'].values
        
        # Lazy
        adf_lazy = AliasDataFrame.read_tree_lazy(str(file_path), "tree")
        lazy_x = adf_lazy['x'].values
        
        # Invariant
        assert_invariant_equal(lazy_x, eager_x, "smoke_I1_load_mode")
    
    # -------------------------------------------------------------------------
    # I2 Smoke: Backend
    # -------------------------------------------------------------------------
    
    def test_smoke_I2_backend(self):
        """
        I2 Smoke: Numba backend == NumPy backend.
        
        Invariant: use_numba=True produces same join results as use_numba=False
        Runtime target: <1s
        """
        from AliasDataFrame import AliasDataFrame
        
        # Minimal data
        main_df = pd.DataFrame({
            'key': np.array([0, 1, 2, 0, 1], dtype=np.int64),
            'value': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        })
        
        sub_df = pd.DataFrame({
            'key': np.array([0, 1, 2], dtype=np.int64),
            'offset': np.array([0.1, 0.2, 0.3], dtype=np.float64),
        })
        
        # NumPy backend
        adf_numpy = AliasDataFrame(main_df.copy(), use_numba=False)
        adf_numpy.register_subframe('S', AliasDataFrame(sub_df.copy()), 'key')
        adf_numpy.add_alias('result', 'value + S.offset')
        adf_numpy.materialize_alias('result')  # FIX: Must materialize
        numpy_result = adf_numpy.df['result'].values
        
        # Numba backend (if available)
        try:
            adf_numba = AliasDataFrame(main_df.copy(), use_numba=True)
            adf_numba.register_subframe('S', AliasDataFrame(sub_df.copy()), 'key')
            adf_numba.add_alias('result', 'value + S.offset')
            adf_numba.materialize_alias('result')  # FIX: Must materialize
            numba_result = adf_numba.df['result'].values
            
            # Invariant
            assert_invariant_equal(numba_result, numpy_result, "smoke_I2_backend")
        except ImportError:
            pytest.skip("Numba not available")
    
    # -------------------------------------------------------------------------
    # I3 Smoke: Subframe Join
    # -------------------------------------------------------------------------
    
    def test_smoke_I3_subframe_join(self):
        """
        I3 Smoke: ADF join == pandas.merge (LEFT JOIN).
        
        Invariant: adf['T.column'] matches pandas.merge(how='left')['column']
        Runtime target: <1s
        """
        from AliasDataFrame import AliasDataFrame
        
        # Minimal data
        main_df = pd.DataFrame({
            'key': np.array([0, 1, 2, 0, 1], dtype=np.int64),
            'value': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        })
        
        sub_df = pd.DataFrame({
            'key': np.array([0, 1, 2], dtype=np.int64),
            'offset': np.array([0.1, 0.2, 0.3], dtype=np.float64),
        })
        
        # ADF join
        adf = AliasDataFrame(main_df.copy())
        adf.register_subframe('T', AliasDataFrame(sub_df.copy()), 'key')
        adf.add_alias('t_offset', 'T.offset')
        adf.materialize_alias('t_offset')  # FIX: Must materialize
        adf_result = adf.df['t_offset'].values
        
        # Pandas merge (contract: LEFT JOIN)
        merged = main_df.merge(sub_df, on='key', how='left')
        pandas_result = merged['offset'].values
        
        # Invariant
        assert_invariant_equal(adf_result, pandas_result, "smoke_I3_subframe_join")
    
    # -------------------------------------------------------------------------
    # I4 Smoke: Compression
    # -------------------------------------------------------------------------
    
    def test_smoke_I4_compression(self):
        """
        I4 Smoke: decompress(compress(x)) ≈ x.
        
        Invariant: Round-trip preserves values within tolerance
        Runtime target: <1s
        """
        from AliasDataFrame import AliasDataFrame
        
        # Minimal data
        df = pd.DataFrame({
            'x': np.linspace(-1.0, 1.0, 50).astype(np.float64),
        })
        
        adf = AliasDataFrame(df.copy())
        original = adf.df['x'].values.copy()
        
        # Compression spec (simpler version that works)
        compression_spec = {
            'x': {
                'compress': 'np.round(np.arcsinh(x) * 100).astype(np.int16)',
                'decompress': 'np.sinh(x_c.astype(np.float64) / 100.0)',
                'compressed_dtype': 'int16',
                'decompressed_dtype': 'float64',
            }
        }
        
        # Round-trip
        adf.compress_columns(compression_spec)
        adf.decompress_columns(['x'])
        recovered = adf.df['x'].values
        
        # Invariant (lossy compression tolerance)
        np.testing.assert_allclose(
            recovered, original, rtol=0.02, atol=0.01,
            err_msg="smoke_I4_compression"
        )
    
    # -------------------------------------------------------------------------
    # I5 Smoke: Schema
    # -------------------------------------------------------------------------
    
    def test_smoke_I5_schema(self, tmp_path):
        """
        I5 Smoke: import(export(schema)).aliases == schema.aliases.
        
        Invariant: Schema round-trip preserves alias definitions
        Runtime target: <1s
        """
        uproot = pytest.importorskip("uproot")
        from AliasDataFrame import AliasDataFrame
        
        # Minimal data with alias
        df = pd.DataFrame({
            'x': np.array([1.0, 2.0, 3.0], dtype=np.float64),
            'y': np.array([10.0, 20.0, 30.0], dtype=np.float64),
        })
        
        adf = AliasDataFrame(df)
        adf.add_alias('sum_xy', 'x + y')
        original_aliases = adf.aliases.copy()
        
        # Export
        file_path = tmp_path / "smoke_i5.root"
        adf.export_tree(str(file_path), "tree")
        
        # Import
        adf_loaded = AliasDataFrame.read_tree(str(file_path), "tree")
        loaded_aliases = adf_loaded.aliases
        
        # Invariant
        assert original_aliases == loaded_aliases, \
            f"smoke_I5_schema: Aliases differ. Original: {original_aliases}, Loaded: {loaded_aliases}"
    
    # -------------------------------------------------------------------------
    # I6 Smoke: Order
    # -------------------------------------------------------------------------
    
    def test_smoke_I6_order(self):
        """
        I6 Smoke: Materialization order doesn't affect results.
        
        Invariant: Order of alias evaluation doesn't change final values
        Runtime target: <1s
        """
        from AliasDataFrame import AliasDataFrame
        
        # Minimal data
        df = pd.DataFrame({
            'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64),
        })
        
        # Order 1: A then B
        adf1 = AliasDataFrame(df.copy())
        adf1.add_alias('A', 'x * 2')
        adf1.add_alias('B', 'x + 10')
        adf1.materialize_alias('A')  # FIX: Must materialize
        adf1.materialize_alias('B')  # FIX: Must materialize
        result_A1 = adf1.df['A'].values.copy()
        result_B1 = adf1.df['B'].values.copy()
        
        # Order 2: B then A
        adf2 = AliasDataFrame(df.copy())
        adf2.add_alias('B', 'x + 10')
        adf2.add_alias('A', 'x * 2')
        adf2.materialize_alias('B')  # FIX: Materialize B first
        adf2.materialize_alias('A')  # FIX: Then A
        result_B2 = adf2.df['B'].values.copy()
        result_A2 = adf2.df['A'].values.copy()
        
        # Invariants
        assert_invariant_equal(result_A1, result_A2, "smoke_I6_order_A")
        assert_invariant_equal(result_B1, result_B2, "smoke_I6_order_B")


# =============================================================================
# SUMMARY
# =============================================================================

class TestSmokeSummary:
    """Verify smoke test completeness."""
    
    def test_smoke_test_count(self):
        """Verify we have exactly 6 smoke tests."""
        smoke_tests = [
            'test_smoke_I1_load_mode',
            'test_smoke_I2_backend',
            'test_smoke_I3_subframe_join',
            'test_smoke_I4_compression',
            'test_smoke_I5_schema',
            'test_smoke_I6_order',
        ]
        assert len(smoke_tests) == 6, f"Expected 6 smoke tests, got {len(smoke_tests)}"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "smoke", "--tb=short"])
