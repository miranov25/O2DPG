"""
Tests for PyArrow Scatter (Phase 9b).

This test suite validates the PyArrow-based subframe value extraction
(gather/scatter operation using pc.take()).

Author: Claude (Coder)
Date: 2025-12-01
"""

import pytest
import numpy as np
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


class TestArrowScatterPrimitives:
    """Test PyArrow take() operation primitives."""
    
    def test_basic_take(self):
        """Test basic gather operation with pc.take()."""
        # Simulate subframe data
        subframe_col = pa.array([10.0, 20.0, 30.0, 40.0, 50.0])
        indices = pa.array([0, 2, 4, 1, 3])
        
        result = pc.take(subframe_col, indices)
        expected = [10.0, 30.0, 50.0, 20.0, 40.0]
        
        np.testing.assert_array_equal(result.to_numpy(), expected)
    
    def test_take_with_duplicates(self):
        """Test gather with duplicate indices (many-to-one join)."""
        subframe_col = pa.array([100.0, 200.0, 300.0])
        # Same subframe row used multiple times
        indices = pa.array([0, 0, 1, 1, 1, 2, 0])
        
        result = pc.take(subframe_col, indices)
        expected = [100.0, 100.0, 200.0, 200.0, 200.0, 300.0, 100.0]
        
        np.testing.assert_array_equal(result.to_numpy(), expected)
    
    def test_missing_key_handling(self):
        """Test handling of missing keys (-1 indices)."""
        subframe_col = pa.array([10.0, 20.0, 30.0])
        
        # -1 indices indicate missing keys
        # We need to replace them with valid indices, take, then mask
        indices_raw = np.array([0, -1, 2, -1, 1])
        missing_mask = indices_raw < 0
        
        # Replace -1 with 0 for safe take
        safe_indices = np.where(indices_raw >= 0, indices_raw, 0)
        indices = pa.array(safe_indices)
        
        # Take values
        taken = pc.take(subframe_col, indices)
        
        # Replace missing with null using if_else
        null_scalar = pa.scalar(None, type=taken.type)
        mask_arr = pa.array(~missing_mask)  # True = keep, False = null
        result = pc.if_else(mask_arr, taken, null_scalar)
        
        # Convert to numpy - nulls become NaN
        result_np = result.to_numpy(zero_copy_only=False)
        
        assert result_np[0] == 10.0
        assert np.isnan(result_np[1])
        assert result_np[2] == 30.0
        assert np.isnan(result_np[3])
        assert result_np[4] == 20.0
    
    def test_preserves_float32_dtype(self):
        """Test that float32 dtype is preserved."""
        subframe_col = pa.array(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        indices = pa.array([2, 0, 1])
        
        result = pc.take(subframe_col, indices)
        result_np = result.to_numpy()
        
        assert result_np.dtype == np.float32
        np.testing.assert_array_equal(result_np, [3.0, 1.0, 2.0])
    
    def test_preserves_float64_dtype(self):
        """Test that float64 dtype is preserved."""
        subframe_col = pa.array(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        indices = pa.array([2, 0, 1])
        
        result = pc.take(subframe_col, indices)
        result_np = result.to_numpy()
        
        assert result_np.dtype == np.float64
        np.testing.assert_array_equal(result_np, [3.0, 1.0, 2.0])
    
    def test_preserves_int32_dtype(self):
        """Test that int32 dtype is preserved (when no nulls)."""
        subframe_col = pa.array(np.array([1, 2, 3], dtype=np.int32))
        indices = pa.array([2, 0, 1])
        
        result = pc.take(subframe_col, indices)
        result_np = result.to_numpy()
        
        assert result_np.dtype == np.int32
        np.testing.assert_array_equal(result_np, [3, 1, 2])
    
    def test_int_with_nulls_becomes_float(self):
        """Test that integer arrays with nulls are handled properly."""
        subframe_col = pa.array(np.array([10, 20, 30], dtype=np.int64))
        
        indices_raw = np.array([0, -1, 2])
        missing_mask = indices_raw < 0
        
        safe_indices = np.where(indices_raw >= 0, indices_raw, 0)
        indices = pa.array(safe_indices)
        
        taken = pc.take(subframe_col, indices)
        
        # Apply null mask
        null_scalar = pa.scalar(None, type=taken.type)
        mask_arr = pa.array(~missing_mask)
        result = pc.if_else(mask_arr, taken, null_scalar)
        
        # For integer arrays with nulls, to_numpy returns object array
        # We need to convert to float to represent NaN
        result_np = result.to_numpy(zero_copy_only=False)
        
        # The result might be object dtype with None, or float with NaN
        # depending on PyArrow version
        if result_np.dtype == object:
            # Convert to float for NaN handling
            result_np = np.array([
                np.nan if x is None else float(x) 
                for x in result_np
            ])
        
        assert result_np[0] == 10.0
        assert np.isnan(result_np[1])
        assert result_np[2] == 30.0


class TestArrowScatterPerformance:
    """Performance tests for Arrow scatter operation."""
    
    def test_scatter_large_array_speed(self):
        """Test that Arrow scatter is fast for large arrays."""
        n_main = 2_000_000
        n_subframe = 1288  # Typical TPC pad count
        
        # Create test data
        subframe_col = pa.array(np.random.randn(n_subframe).astype(np.float32))
        indices = pa.array(np.random.randint(0, n_subframe, size=n_main))
        
        # Warmup
        _ = pc.take(subframe_col, indices)
        
        # Timed run
        t0 = time.perf_counter()
        result = pc.take(subframe_col, indices)
        _ = result.to_numpy(zero_copy_only=False)  # Force materialization
        elapsed = time.perf_counter() - t0
        
        # Should be fast (<100ms for 2M rows, allow 500ms for slower systems)
        assert elapsed < 0.5, f"Scatter took {elapsed:.3f}s, expected <0.5s"
        print(f"\nArrow scatter: {n_main:,} rows in {elapsed*1000:.1f}ms")
    
    def test_scatter_with_missing_keys_speed(self):
        """Test performance with missing keys."""
        n_main = 2_000_000
        n_subframe = 1288
        missing_rate = 0.1  # 10% missing keys
        
        # Create test data with some -1 indices
        indices_raw = np.random.randint(0, n_subframe, size=n_main)
        missing_mask = np.random.random(n_main) < missing_rate
        indices_raw[missing_mask] = -1
        
        subframe_col = pa.array(np.random.randn(n_subframe).astype(np.float32))
        safe_indices = np.where(indices_raw >= 0, indices_raw, 0)
        indices = pa.array(safe_indices)
        
        # Warmup
        taken = pc.take(subframe_col, indices)
        
        # Timed run
        t0 = time.perf_counter()
        taken = pc.take(subframe_col, indices)
        null_scalar = pa.scalar(None, type=taken.type)
        mask_arr = pa.array(~missing_mask)
        result = pc.if_else(mask_arr, taken, null_scalar)
        _ = result.to_numpy(zero_copy_only=False)
        elapsed = time.perf_counter() - t0
        
        # Should still be reasonably fast
        assert elapsed < 1.0, f"Scatter with missing took {elapsed:.3f}s, expected <1.0s"
        print(f"\nArrow scatter with {missing_rate*100:.0f}% missing: "
              f"{n_main:,} rows in {elapsed*1000:.1f}ms")
    
    def test_arrow_vs_numpy_performance(self):
        """Compare Arrow scatter vs NumPy fancy indexing."""
        n_main = 2_000_000
        n_subframe = 1288
        
        # Create test data
        subframe_values = np.random.randn(n_subframe).astype(np.float32)
        indices = np.random.randint(0, n_subframe, size=n_main)
        
        # NumPy approach (current implementation)
        t0 = time.perf_counter()
        result_numpy = subframe_values[indices]
        numpy_time = time.perf_counter() - t0
        
        # Arrow approach
        subframe_arr = pa.array(subframe_values)
        indices_arr = pa.array(indices)
        
        t0 = time.perf_counter()
        result_arrow = pc.take(subframe_arr, indices_arr)
        _ = result_arrow.to_numpy(zero_copy_only=False)
        arrow_time = time.perf_counter() - t0
        
        print(f"\nNumPy fancy indexing: {numpy_time*1000:.1f}ms")
        print(f"Arrow take(): {arrow_time*1000:.1f}ms")
        print(f"Speedup: {numpy_time/arrow_time:.2f}x")
        
        # Arrow should be competitive (within 3x of NumPy at worst)
        # Note: NumPy fancy indexing is already quite optimized
        assert arrow_time < numpy_time * 3, \
            f"Arrow unexpectedly slow: {arrow_time:.3f}s vs NumPy {numpy_time:.3f}s"


class TestArrowScatterCorrectness:
    """Correctness tests comparing Arrow to NumPy."""
    
    def test_matches_numpy_simple(self):
        """Test that Arrow results match NumPy for simple case."""
        subframe_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        indices = np.array([0, 2, 4, 1, 3, 0, 2])
        
        # NumPy reference
        expected = subframe_values[indices]
        
        # Arrow implementation
        subframe_arr = pa.array(subframe_values)
        indices_arr = pa.array(indices)
        result = pc.take(subframe_arr, indices_arr).to_numpy()
        
        np.testing.assert_array_equal(result, expected)
    
    def test_matches_numpy_with_missing(self):
        """Test that Arrow results match NumPy with missing keys."""
        subframe_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        indices_raw = np.array([0, -1, 2, -1, 4, 1, -1])
        missing_mask = indices_raw < 0
        
        # NumPy reference (current AliasDataFrame implementation)
        expected = np.full(len(indices_raw), np.nan, dtype=np.float64)
        valid = indices_raw >= 0
        expected[valid] = subframe_values[indices_raw[valid]]
        
        # Arrow implementation
        subframe_arr = pa.array(subframe_values)
        safe_indices = np.where(indices_raw >= 0, indices_raw, 0)
        indices_arr = pa.array(safe_indices)
        
        taken = pc.take(subframe_arr, indices_arr)
        null_scalar = pa.scalar(None, type=taken.type)
        mask_arr = pa.array(~missing_mask)
        result = pc.if_else(mask_arr, taken, null_scalar)
        result_np = result.to_numpy(zero_copy_only=False)
        
        # Compare - both should have NaN at same positions
        np.testing.assert_array_equal(np.isnan(result_np), np.isnan(expected))
        
        # Compare non-NaN values
        valid_mask = ~np.isnan(expected)
        np.testing.assert_array_equal(result_np[valid_mask], expected[valid_mask])
    
    def test_all_missing_keys(self):
        """Test edge case where all keys are missing."""
        subframe_values = np.array([10.0, 20.0, 30.0])
        indices_raw = np.array([-1, -1, -1, -1, -1])
        missing_mask = indices_raw < 0
        
        # Arrow implementation
        subframe_arr = pa.array(subframe_values)
        safe_indices = np.where(indices_raw >= 0, indices_raw, 0)
        indices_arr = pa.array(safe_indices)
        
        taken = pc.take(subframe_arr, indices_arr)
        null_scalar = pa.scalar(None, type=taken.type)
        mask_arr = pa.array(~missing_mask)
        result = pc.if_else(mask_arr, taken, null_scalar)
        result_np = result.to_numpy(zero_copy_only=False)
        
        # All should be NaN
        assert np.all(np.isnan(result_np))
    
    def test_no_missing_keys(self):
        """Test edge case where no keys are missing."""
        subframe_values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        indices = np.array([0, 1, 2, 3, 4, 0, 1])
        
        # Direct take without null handling
        subframe_arr = pa.array(subframe_values)
        indices_arr = pa.array(indices)
        result = pc.take(subframe_arr, indices_arr).to_numpy()
        
        expected = subframe_values[indices]
        np.testing.assert_array_equal(result, expected)
    
    def test_empty_arrays(self):
        """Test edge case with empty arrays."""
        subframe_values = np.array([10.0, 20.0, 30.0])
        indices = np.array([], dtype=np.int64)
        
        subframe_arr = pa.array(subframe_values)
        indices_arr = pa.array(indices)
        result = pc.take(subframe_arr, indices_arr).to_numpy()
        
        assert len(result) == 0
    
    def test_single_element(self):
        """Test edge case with single element."""
        subframe_values = np.array([42.0])
        indices = np.array([0, 0, 0])
        
        subframe_arr = pa.array(subframe_values)
        indices_arr = pa.array(indices)
        result = pc.take(subframe_arr, indices_arr).to_numpy()
        
        np.testing.assert_array_equal(result, [42.0, 42.0, 42.0])


class TestArrowScatterIntegration:
    """Integration tests simulating real AliasDataFrame usage."""
    
    def test_tpc_like_join(self):
        """Test pattern matching TPC calibration join.
        
        Main DataFrame: ~13.5M rows (track segments)
        Subframe: ~1288 rows (pad-by-pad calibration)
        """
        n_main = 100_000  # Reduced for test speed
        n_subframe = 1288
        
        # Simulate subframe with calibration values
        subframe_data = {
            'padrow': np.arange(n_subframe, dtype=np.int32),
            'gain': np.random.uniform(0.9, 1.1, n_subframe).astype(np.float32),
            'offset': np.random.uniform(-0.1, 0.1, n_subframe).astype(np.float32),
        }
        
        # Simulate main DataFrame join indices
        # Most tracks have valid padrow, ~5% missing
        indices = np.random.randint(0, n_subframe, size=n_main)
        missing_mask = np.random.random(n_main) < 0.05
        indices[missing_mask] = -1
        
        # Extract both columns using Arrow
        for col in ['gain', 'offset']:
            subframe_arr = pa.array(subframe_data[col])
            safe_indices = np.where(indices >= 0, indices, 0)
            indices_arr = pa.array(safe_indices)
            
            taken = pc.take(subframe_arr, indices_arr)
            null_scalar = pa.scalar(None, type=taken.type)
            mask_arr = pa.array(~missing_mask)
            result = pc.if_else(mask_arr, taken, null_scalar)
            result_np = result.to_numpy(zero_copy_only=False)
            
            # Verify NaN at missing positions
            assert np.all(np.isnan(result_np[missing_mask]))
            
            # Verify correct values at valid positions
            valid_indices = indices[~missing_mask]
            expected = subframe_data[col][valid_indices]
            np.testing.assert_array_almost_equal(
                result_np[~missing_mask], expected
            )
    
    def test_multi_column_extract(self):
        """Test extracting multiple columns from same subframe."""
        n_main = 50_000
        n_subframe = 500
        n_cols = 5
        
        # Create subframe with multiple columns
        subframe_data = {
            f'col_{i}': np.random.randn(n_subframe).astype(np.float32)
            for i in range(n_cols)
        }
        
        # Same indices for all columns
        indices = np.random.randint(0, n_subframe, size=n_main)
        indices_arr = pa.array(indices)
        
        # Extract all columns
        results = {}
        for col_name, col_data in subframe_data.items():
            subframe_arr = pa.array(col_data)
            taken = pc.take(subframe_arr, indices_arr)
            results[col_name] = taken.to_numpy()
        
        # Verify all extractions are correct
        for col_name, col_data in subframe_data.items():
            expected = col_data[indices]
            np.testing.assert_array_almost_equal(results[col_name], expected)


# =============================================================================
# If running standalone, show test discovery
# =============================================================================
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
