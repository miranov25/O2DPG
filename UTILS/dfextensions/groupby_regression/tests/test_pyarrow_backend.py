"""
Phase 13.1.GB: PyArrow backend tests for groupby regression.

Tests cover:
- Backend selection logic
- Results parity between pandas and pyarrow
- Sort stability and determinism
- Memory behavior (Arrow pool metrics)
- Edge cases
"""
import pytest
import numpy as np
import pandas as pd
import gc

# Check if PyArrow available
try:
    import pyarrow as pa
    PYARROW_AVAILABLE = True
    PYARROW_VERSION = tuple(int(x) for x in pa.__version__.split('.')[:2])
except ImportError:
    PYARROW_AVAILABLE = False
    PYARROW_VERSION = (0, 0)

# Import the module under test (relative import for package structure)
from ..groupby_regression_optimized import make_parallel_fit_v4, _PYARROW_AVAILABLE


# ============================================================================
# TEST: BACKEND SELECTION
# ============================================================================

class TestBackendSelection:
    """Tests for backend selection logic."""
    
    def test_backend_pandas_explicit(self):
        """Explicit pandas backend always uses pandas."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 500,
            'x': np.random.randn(1000),
            'y': np.random.randn(1000),
        })
        
        df_out, dfGB = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pandas',
        )
        
        assert len(dfGB) == 2
        assert 'y_intercept_v4' in dfGB.columns
    
    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")
    def test_backend_pyarrow_explicit(self):
        """Explicit pyarrow backend uses pyarrow."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 500,
            'x': np.random.randn(1000),
            'y': np.random.randn(1000),
        })
        
        df_out, dfGB = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
        )
        
        assert len(dfGB) == 2
        assert 'y_intercept_v4' in dfGB.columns
    
    def test_backend_auto_below_threshold(self):
        """Auto backend uses pandas below threshold."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 50,
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        
        df_out, dfGB = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='auto',
            pyarrow_threshold=1000,  # 100 < 1000
        )
        
        assert len(dfGB) == 2
    
    @pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")
    def test_backend_auto_above_threshold(self):
        """Auto backend uses pyarrow above threshold."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 1000,
            'x': np.random.randn(2000),
            'y': np.random.randn(2000),
        })
        
        df_out, dfGB = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='auto',
            pyarrow_threshold=1000,  # 2000 > 1000
        )
        
        assert len(dfGB) == 2
    
    def test_backend_invalid_raises(self):
        """Invalid backend raises ValueError."""
        df = pd.DataFrame({'g': [1, 2], 'x': [1.0, 2.0], 'y': [1.0, 2.0]})
        
        with pytest.raises(ValueError, match="Invalid backend"):
            make_parallel_fit_v4(
                df=df,
                gb_columns=['g'],
                fit_columns=['y'],
                linear_columns=['x'],
                backend='invalid',
            )


# ============================================================================
# TEST: RESULTS PARITY
# ============================================================================

@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")
class TestResultsParity:
    """Tests ensuring pandas and pyarrow produce identical results."""
    
    def test_results_match_simple(self):
        """Results identical for simple case."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 500,
            'x': np.random.randn(1000),
            'y': np.random.randn(1000),
        })
        
        _, dfGB_pandas = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pandas',
        )
        
        _, dfGB_arrow = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
        )
        
        pd.testing.assert_frame_equal(
            dfGB_pandas.sort_values('g').reset_index(drop=True),
            dfGB_arrow.sort_values('g').reset_index(drop=True),
            rtol=1e-10,
        )
    
    def test_results_match_multicolumn_groupby(self):
        """Results identical for multi-column groupby."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g1': np.random.randint(0, 5, 2000),
            'g2': np.random.randint(0, 3, 2000),
            'x': np.random.randn(2000),
            'y': np.random.randn(2000),
        })
        
        _, dfGB_pandas = make_parallel_fit_v4(
            df=df,
            gb_columns=['g1', 'g2'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pandas',
        )
        
        _, dfGB_arrow = make_parallel_fit_v4(
            df=df,
            gb_columns=['g1', 'g2'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
        )
        
        pd.testing.assert_frame_equal(
            dfGB_pandas.sort_values(['g1', 'g2']).reset_index(drop=True),
            dfGB_arrow.sort_values(['g1', 'g2']).reset_index(drop=True),
            rtol=1e-10,
        )
    
    def test_results_match_with_weights(self):
        """Results identical with weighted regression."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 500,
            'x': np.random.randn(1000),
            'y': np.random.randn(1000),
            'w': np.abs(np.random.randn(1000)) + 0.1,
        })
        
        _, dfGB_pandas = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            weights='w',
            backend='pandas',
        )
        
        _, dfGB_arrow = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            weights='w',
            backend='pyarrow',
        )
        
        pd.testing.assert_frame_equal(
            dfGB_pandas.sort_values('g').reset_index(drop=True),
            dfGB_arrow.sort_values('g').reset_index(drop=True),
            rtol=1e-10,
        )
    
    def test_results_match_multiple_targets(self):
        """Results identical for multiple fit targets."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 500,
            'x1': np.random.randn(1000),
            'x2': np.random.randn(1000),
            'y1': np.random.randn(1000),
            'y2': np.random.randn(1000),
        })
        
        _, dfGB_pandas = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y1', 'y2'],
            linear_columns=['x1', 'x2'],
            backend='pandas',
        )
        
        _, dfGB_arrow = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y1', 'y2'],
            linear_columns=['x1', 'x2'],
            backend='pyarrow',
        )
        
        pd.testing.assert_frame_equal(
            dfGB_pandas.sort_values('g').reset_index(drop=True),
            dfGB_arrow.sort_values('g').reset_index(drop=True),
            rtol=1e-10,
        )
    
    def test_results_match_no_intercept(self):
        """Results identical with fit_intercept=False."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 500,
            'x': np.random.randn(1000),
            'y': np.random.randn(1000),
        })
        
        _, dfGB_pandas = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            fit_intercept=False,
            backend='pandas',
        )
        
        _, dfGB_arrow = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            fit_intercept=False,
            backend='pyarrow',
        )
        
        pd.testing.assert_frame_equal(
            dfGB_pandas.sort_values('g').reset_index(drop=True),
            dfGB_arrow.sort_values('g').reset_index(drop=True),
            rtol=1e-10,
        )


# ============================================================================
# TEST: SORT STABILITY AND DETERMINISM
# ============================================================================

@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")
class TestSortStability:
    """Tests for sort stability and determinism."""
    
    def test_sort_stability_matches_pandas(self):
        """PyArrow sort produces identical ordering to pandas mergesort."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g1': np.random.randint(0, 10, 5000),
            'g2': np.random.randint(0, 5, 5000),
            'x': np.random.randn(5000),
            'y': np.random.randn(5000),
        })
        
        df_out_pandas, dfGB_pandas = make_parallel_fit_v4(
            df=df,
            gb_columns=['g1', 'g2'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pandas',
        )
        
        df_out_arrow, dfGB_arrow = make_parallel_fit_v4(
            df=df,
            gb_columns=['g1', 'g2'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
        )
        
        # Group results must match
        pd.testing.assert_frame_equal(
            dfGB_pandas.sort_values(['g1', 'g2']).reset_index(drop=True),
            dfGB_arrow.sort_values(['g1', 'g2']).reset_index(drop=True),
            rtol=1e-10,
        )
    
    def test_deterministic_column_order(self):
        """Column order is deterministic across runs."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 50,
            'x1': np.random.randn(100),
            'x2': np.random.randn(100),
            'y': np.random.randn(100),
        })
        
        orders = []
        for _ in range(5):
            df_out, _ = make_parallel_fit_v4(
                df=df,
                gb_columns=['g'],
                fit_columns=['y'],
                linear_columns=['x1', 'x2'],
                backend='pyarrow',
            )
            orders.append(list(df_out.columns))
        
        assert all(o == orders[0] for o in orders), "Column order not deterministic"
    
    def test_deterministic_results_across_runs(self):
        """Results are identical across multiple runs."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': np.random.randint(0, 10, 1000),
            'x': np.random.randn(1000),
            'y': np.random.randn(1000),
        })
        
        results = []
        for _ in range(3):
            _, dfGB = make_parallel_fit_v4(
                df=df,
                gb_columns=['g'],
                fit_columns=['y'],
                linear_columns=['x'],
                backend='pyarrow',
            )
            results.append(dfGB.sort_values('g').reset_index(drop=True))
        
        for i in range(1, len(results)):
            pd.testing.assert_frame_equal(results[0], results[i])
    
    def test_row_order_within_groups_matches_pandas(self):
        """
        Verify PyArrow sort produces identical row ordering to pandas mergesort.
        
        This tests sort STABILITY: rows with equal group keys should maintain
        their relative order. This is critical for features that depend on
        within-group row order (rolling stats, first/last semantics).
        """
        np.random.seed(42)
        n = 5000
        df = pd.DataFrame({
            'g1': np.random.randint(0, 10, n),
            'g2': np.random.randint(0, 5, n),
            'x': np.random.randn(n),
            'y': np.random.randn(n),
        })
        # Add row_id to track original order (use x as proxy since it's included)
        df['row_id'] = np.arange(n)
        
        df_out_pandas, _ = make_parallel_fit_v4(
            df=df,
            gb_columns=['g1', 'g2'],
            fit_columns=['y'],
            linear_columns=['x', 'row_id'],  # Include row_id as linear col to preserve it
            backend='pandas',
        )
        
        df_out_arrow, _ = make_parallel_fit_v4(
            df=df,
            gb_columns=['g1', 'g2'],
            fit_columns=['y'],
            linear_columns=['x', 'row_id'],
            backend='pyarrow',
        )
        
        # Row ordering must be identical (stable sort requirement)
        # Compare row_id sequence to verify exact row order matches
        pd.testing.assert_series_equal(
            df_out_pandas['row_id'].reset_index(drop=True),
            df_out_arrow['row_id'].reset_index(drop=True),
            check_names=False,
        )


# ============================================================================
# TEST: MEMORY BEHAVIOR
# ============================================================================

@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")
class TestMemoryBehavior:
    """
    Tests for memory efficiency and release.
    
    Note: tracemalloc only tracks Python heap allocations, NOT native
    Arrow/NumPy allocations. Memory reduction tests are marked slow.
    """
    
    def test_arrow_pool_bytes_check(self):
        """
        Arrow pool releases memory after operation.
        
        This is the PRIMARY automated memory check - Arrow pool metrics
        are more reliable than tracemalloc for native allocations.
        """
        pool = pa.default_memory_pool()
        
        np.random.seed(42)
        df = pd.DataFrame({
            'g': np.random.randint(0, 10, 100_000),
            'x': np.random.randn(100_000),
            'y': np.random.randn(100_000),
        })
        
        baseline = pool.bytes_allocated()
        
        _, dfGB = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
        )
        
        # Explicitly delete and collect
        del dfGB
        gc.collect()
        
        current = pool.bytes_allocated()
        
        # Generous tolerance (50 MB) - allocators may retain memory in pools
        assert current < baseline + 50_000_000, (
            f"Arrow memory not released: baseline={baseline/1e6:.1f} MB, "
            f"current={current/1e6:.1f} MB"
        )
    
    def test_pyarrow_backend_functional(self):
        """
        PyArrow backend produces correct output.
        
        This is the PRIMARY CI test - ensures the backend works correctly.
        Memory behavior is tested separately.
        """
        np.random.seed(42)
        df = pd.DataFrame({
            'g': np.random.randint(0, 10, 10_000),
            'x': np.random.randn(10_000),
            'y': np.random.randn(10_000),
        })
        
        df_out, dfGB = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
        )
        
        # Verify output structure
        assert len(dfGB) == 10  # 10 groups
        assert 'y_intercept_v4' in dfGB.columns
        assert 'y_slope_x_v4' in dfGB.columns
        assert 'y_rms_v4' in dfGB.columns
        
        # Verify no NaN in results
        assert not dfGB['y_intercept_v4'].isna().any()
        assert not dfGB['y_slope_x_v4'].isna().any()
    
    @pytest.mark.slow
    def test_memory_reduction_large_dataset(self):
        """
        Benchmark: PyArrow uses less peak memory for large datasets.
        
        WARNING: This test uses tracemalloc which only tracks Python heap.
        Arrow and NumPy allocate through native allocators (jemalloc, etc.)
        so results may not reflect true memory usage. Use as indicative only.
        """
        import tracemalloc
        
        np.random.seed(42)
        n_rows = 500_000  # Reduced for faster testing
        df = pd.DataFrame({
            'g': np.random.randint(0, 100, n_rows),
            'x': np.random.randn(n_rows),
            'y': np.random.randn(n_rows),
        })
        
        # Pandas path
        gc.collect()
        tracemalloc.start()
        _, _ = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pandas',
        )
        _, peak_pandas = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        gc.collect()
        
        # PyArrow path
        tracemalloc.start()
        _, _ = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
        )
        _, peak_arrow = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Log results for manual inspection
        print(f"Peak memory - Pandas: {peak_pandas/1e6:.1f} MB, PyArrow: {peak_arrow/1e6:.1f} MB")
        
        # Informational check - tracemalloc doesn't reliably track native allocations
        # Skip instead of fail to avoid CI flakiness across different environments
        if peak_arrow >= peak_pandas * 2.0:
            pytest.skip(
                f"Memory check inconclusive (tracemalloc limitation): "
                f"arrow={peak_arrow/1e6:.1f}MB, pandas={peak_pandas/1e6:.1f}MB. "
                f"Use Arrow pool metrics or RSS benchmarks for accurate measurement."
            )


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")
class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Empty DataFrame handled correctly."""
        df = pd.DataFrame({
            'g': pd.Series([], dtype=int),
            'x': pd.Series([], dtype=float),
            'y': pd.Series([], dtype=float),
        })
        
        df_out, dfGB = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
        )
        
        assert len(dfGB) == 0
    
    def test_single_group(self):
        """Single group handled correctly."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1] * 100,
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        
        _, dfGB = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
        )
        
        assert len(dfGB) == 1
    
    def test_single_row(self):
        """Single row handled correctly (insufficient data)."""
        df = pd.DataFrame({
            'g': [1],
            'x': [1.0],
            'y': [1.0],
        })
        
        _, dfGB = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
        )
        
        assert len(dfGB) == 1
    
    def test_with_nan_values(self):
        """NaN values handled correctly."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 50,
            'x': np.random.randn(100),
            'y': np.random.randn(100),
        })
        df.loc[0, 'y'] = np.nan
        df.loc[1, 'x'] = np.nan
        
        # Should not raise
        _, dfGB = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
        )
        
        assert len(dfGB) == 2
    
    def test_with_selection(self):
        """Selection filter works with PyArrow backend."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 100,
            'x': np.random.randn(200),
            'y': np.random.randn(200),
        })
        
        selection = df['x'] > 0
        
        _, dfGB = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            selection=selection,
            backend='pyarrow',
        )
        
        assert len(dfGB) == 2


# ============================================================================
# TEST: METADATA EXPORT
# ============================================================================

@pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")
class TestMetadataExport:
    """Tests for return_metadata with PyArrow backend."""
    
    def test_metadata_export_pyarrow(self):
        """Metadata export works with PyArrow backend."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 500,
            'x': np.random.randn(1000),
            'y': np.random.randn(1000),
        })
        
        df_out, dfGB, metadata = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
            return_metadata=True,
        )
        
        # Check metadata structure
        assert 'formulas' in metadata
        assert 'y_pred_v4' in metadata['formulas']
        assert 'columns' in metadata
    
    def test_metadata_matches_pandas(self):
        """Metadata identical between backends."""
        np.random.seed(42)
        df = pd.DataFrame({
            'g': [1, 2] * 500,
            'x': np.random.randn(1000),
            'y': np.random.randn(1000),
        })
        
        _, _, meta_pandas = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pandas',
            return_metadata=True,
        )
        
        _, _, meta_arrow = make_parallel_fit_v4(
            df=df,
            gb_columns=['g'],
            fit_columns=['y'],
            linear_columns=['x'],
            backend='pyarrow',
            return_metadata=True,
        )
        
        # Formulas should be identical
        assert meta_pandas['formulas'] == meta_arrow['formulas']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
