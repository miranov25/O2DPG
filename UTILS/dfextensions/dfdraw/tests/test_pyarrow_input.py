"""
Tests for Phase 13.1.DF: dfdraw PyArrow input support.

Tests that DFDraw accepts PyArrow Tables as input and produces
identical results to pandas DataFrames.

MINIMAL IMPLEMENTATION: PyArrow Tables are immediately converted to pandas.
This provides API compatibility with PyArrow-based pipelines but does not
reduce memory usage within dfdraw itself. Memory optimization occurs
upstream in groupby-regression (Phase 13.1.GB) and AliasDataFrame (Phase 13.3.ADF).
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Skip all tests if PyArrow not available
try:
    import pyarrow as pa
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYARROW_AVAILABLE, reason="PyArrow not installed")

from dfdraw import DFDraw, set_style


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n),
        'z': np.random.randn(n),
        'group': np.random.choice(['A', 'B', 'C'], n),
        'weight': np.abs(np.random.randn(n)),
    })


@pytest.fixture
def sample_table(sample_df):
    """Create PyArrow Table from sample DataFrame."""
    return pa.Table.from_pandas(sample_df)


@pytest.fixture(autouse=True)
def reset_style():
    """Reset style and close figures after each test."""
    set_style(None)
    yield
    plt.close('all')


class TestPyArrowInputAcceptance:
    """Test that DFDraw accepts PyArrow Tables."""
    
    def test_init_pandas_dataframe(self, sample_df):
        """pandas DataFrame works as before."""
        drawer = DFDraw(sample_df)
        assert drawer.backend == 'pandas'
        assert drawer._table is None
        assert len(drawer.df) == 1000
    
    def test_init_pyarrow_table(self, sample_table):
        """PyArrow Table is accepted and converted to pandas."""
        drawer = DFDraw(sample_table)
        assert drawer.backend == 'pyarrow'
        assert drawer._table is not None
        assert len(drawer.df) == 1000
        assert isinstance(drawer.df, pd.DataFrame)
    
    def test_init_dict(self):
        """Dict of arrays still works."""
        data = {'x': [1, 2, 3], 'y': [4, 5, 6]}
        drawer = DFDraw(data)
        assert drawer.backend == 'pandas'
        assert len(drawer.df) == 3
    
    def test_init_invalid_type(self):
        """Invalid input type raises TypeError."""
        with pytest.raises(TypeError, match="Cannot create DFDraw"):
            DFDraw("invalid")


class TestBackendProperty:
    """Test the backend property."""
    
    def test_backend_pandas(self, sample_df):
        """backend is 'pandas' for DataFrame input."""
        drawer = DFDraw(sample_df)
        assert drawer.backend == 'pandas'
    
    def test_backend_pyarrow(self, sample_table):
        """backend is 'pyarrow' for Table input."""
        drawer = DFDraw(sample_table)
        assert drawer.backend == 'pyarrow'


class TestMemoryInfo:
    """Test memory_info() method with unified keys."""
    
    def test_memory_info_pandas(self, sample_df):
        """memory_info returns correct info for pandas."""
        drawer = DFDraw(sample_df)
        info = drawer.memory_info()
        
        assert info['backend'] == 'pandas'
        assert info['num_rows'] == 1000
        assert info['num_columns'] == 5
        assert info['nbytes'] > 0
        # pandas input should NOT have original_nbytes
        assert 'original_nbytes' not in info
    
    def test_memory_info_pyarrow(self, sample_table):
        """memory_info returns correct info for PyArrow."""
        drawer = DFDraw(sample_table)
        info = drawer.memory_info()
        
        assert info['backend'] == 'pyarrow'
        assert info['num_rows'] == 1000
        assert info['num_columns'] == 5
        assert info['nbytes'] > 0  # Unified key: actual pandas memory
        assert info['original_nbytes'] > 0  # PyArrow original size
    
    def test_memory_info_keys_consistent(self, sample_df, sample_table):
        """Both backends have same base keys."""
        drawer_pd = DFDraw(sample_df)
        drawer_pa = DFDraw(sample_table)
        
        info_pd = drawer_pd.memory_info()
        info_pa = drawer_pa.memory_info()
        
        # Common keys present in both
        common_keys = {'backend', 'nbytes', 'num_rows', 'num_columns'}
        assert common_keys <= set(info_pd.keys())
        assert common_keys <= set(info_pa.keys())


class TestResultsParity:
    """Verify pandas and PyArrow produce identical plot results."""
    
    def test_hist_parity(self, sample_df, sample_table):
        """Histogram produces identical results."""
        drawer_pd = DFDraw(sample_df)
        drawer_pa = DFDraw(sample_table)
        
        fig_pd, ax_pd, stats_pd = drawer_pd.hist('x', bins=50)
        fig_pa, ax_pa, stats_pa = drawer_pa.hist('x', bins=50)
        
        assert stats_pd['n'] == stats_pa['n']
        np.testing.assert_allclose(stats_pd['mean'], stats_pa['mean'], rtol=1e-10)
        np.testing.assert_allclose(stats_pd['std'], stats_pa['std'], rtol=1e-10)
    
    def test_scatter_parity(self, sample_df, sample_table):
        """Scatter plot produces identical results."""
        drawer_pd = DFDraw(sample_df)
        drawer_pa = DFDraw(sample_table)
        
        fig_pd, ax_pd, stats_pd = drawer_pd.scatter('y:x')
        fig_pa, ax_pa, stats_pa = drawer_pa.scatter('y:x')
        
        assert stats_pd['n'] == stats_pa['n']
        np.testing.assert_allclose(stats_pd['mean_x'], stats_pa['mean_x'], rtol=1e-10)
        np.testing.assert_allclose(stats_pd['mean_y'], stats_pa['mean_y'], rtol=1e-10)
        np.testing.assert_allclose(stats_pd['corr'], stats_pa['corr'], rtol=1e-10)
    
    def test_profile_parity(self, sample_df, sample_table):
        """Profile plot produces identical results."""
        drawer_pd = DFDraw(sample_df)
        drawer_pa = DFDraw(sample_table)
        
        fig_pd, ax_pd, stats_pd = drawer_pd.profile('y:x', bins=20)
        fig_pa, ax_pa, stats_pa = drawer_pa.profile('y:x', bins=20)
        
        assert stats_pd['n'] == stats_pa['n']
        np.testing.assert_allclose(stats_pd['mean_x'], stats_pa['mean_x'], rtol=1e-10)
        np.testing.assert_allclose(stats_pd['mean_y'], stats_pa['mean_y'], rtol=1e-10)
    
    def test_hist2d_parity(self, sample_df, sample_table):
        """2D histogram produces identical results."""
        drawer_pd = DFDraw(sample_df)
        drawer_pa = DFDraw(sample_table)
        
        fig_pd, ax_pd, stats_pd = drawer_pd.hist2d('y:x', bins=30)
        fig_pa, ax_pa, stats_pa = drawer_pa.hist2d('y:x', bins=30)
        
        assert stats_pd['n'] == stats_pa['n']
        np.testing.assert_allclose(stats_pd['corr'], stats_pa['corr'], rtol=1e-10)
    
    def test_hexbin_parity(self, sample_df, sample_table):
        """Hexbin produces identical results."""
        drawer_pd = DFDraw(sample_df)
        drawer_pa = DFDraw(sample_table)
        
        fig_pd, ax_pd, stats_pd = drawer_pd.hexbin('y:x', gridsize=20)
        fig_pa, ax_pa, stats_pa = drawer_pa.hexbin('y:x', gridsize=20)
        
        assert stats_pd['n'] == stats_pa['n']


class TestComputedExpressions:
    """Test computed expressions work with PyArrow input."""
    
    def test_hist_computed_expression(self, sample_table):
        """Computed expression in hist works with PyArrow."""
        drawer = DFDraw(sample_table)
        fig, ax, stats = drawer.hist('x + y')
        assert stats['n'] == 1000
    
    def test_scatter_computed_expression(self, sample_table):
        """Computed expression in scatter works with PyArrow."""
        drawer = DFDraw(sample_table)
        fig, ax, stats = drawer.scatter('x + y : x * 2')
        assert stats['n'] == 1000
    
    def test_computed_expression_parity(self, sample_df, sample_table):
        """Computed expressions produce same results."""
        drawer_pd = DFDraw(sample_df)
        drawer_pa = DFDraw(sample_table)
        
        fig_pd, ax_pd, stats_pd = drawer_pd.hist('x + y', bins=50)
        fig_pa, ax_pa, stats_pa = drawer_pa.hist('x + y', bins=50)
        
        assert stats_pd['n'] == stats_pa['n']
        np.testing.assert_allclose(stats_pd['mean'], stats_pa['mean'], rtol=1e-10)


class TestSelectionWithPyArrow:
    """Test selection parameter with PyArrow input."""
    
    def test_selection_works(self, sample_table):
        """Selection works with PyArrow input."""
        drawer = DFDraw(sample_table)
        fig, ax, stats = drawer.hist('x', selection='x > 0')
        
        assert stats['n'] < 1000
        assert stats['mean'] > 0
    
    def test_selection_results_match(self, sample_df, sample_table):
        """Selection produces same results for both backends."""
        drawer_pd = DFDraw(sample_df)
        drawer_pa = DFDraw(sample_table)
        
        fig_pd, ax_pd, stats_pd = drawer_pd.hist('x', selection='x > 0')
        fig_pa, ax_pa, stats_pa = drawer_pa.hist('x', selection='x > 0')
        
        assert stats_pd['n'] == stats_pa['n']
        np.testing.assert_allclose(stats_pd['mean'], stats_pa['mean'], rtol=1e-10)


class TestGroupByWithPyArrow:
    """Test group_by parameter with PyArrow input."""
    
    def test_group_by_works(self, sample_table):
        """Group_by works with PyArrow input."""
        drawer = DFDraw(sample_table)
        fig, ax, stats = drawer.hist('x', group_by='group')
        
        assert stats.get('grouped', False)
        legend = ax.get_legend()
        assert legend is not None
    
    def test_group_by_results_match(self, sample_df, sample_table):
        """Group_by produces same grouped status for both backends."""
        drawer_pd = DFDraw(sample_df)
        drawer_pa = DFDraw(sample_table)
        
        fig_pd, ax_pd, stats_pd = drawer_pd.hist('x', group_by='group')
        fig_pa, ax_pa, stats_pa = drawer_pa.hist('x', group_by='group')
        
        assert stats_pd.get('grouped', False) == stats_pa.get('grouped', False)


class TestDrawDispatch:
    """Test draw() method dispatch with PyArrow input."""
    
    def test_draw_hist_pyarrow(self, sample_table):
        """draw() dispatches to hist for 1D expression."""
        drawer = DFDraw(sample_table)
        fig, ax, stats = drawer.draw('x')
        assert stats['n'] == 1000
        assert 'mean' in stats
    
    def test_draw_scatter_pyarrow(self, sample_table):
        """draw() dispatches to scatter for 2D expression."""
        drawer = DFDraw(sample_table)
        fig, ax, stats = drawer.draw('y:x')
        assert stats['n'] == 1000
        assert 'corr' in stats
    
    def test_draw_profile_pyarrow(self, sample_table):
        """draw() with type='profile' works."""
        drawer = DFDraw(sample_table)
        fig, ax, stats = drawer.draw('y:x', type='profile')
        assert stats['n'] == 1000


class TestBatchWithPyArrow:
    """Test batch processing with PyArrow input."""
    
    def test_batch_pyarrow(self, sample_table):
        """draw_batch works with PyArrow input."""
        drawer = DFDraw(sample_table)
        
        specs = {
            'hist_x': {'expr': 'x', 'bins': 50},
            'scatter_yx': {'expr': 'y:x'},
        }
        
        results = drawer.draw_batch(specs, verbose=False)
        
        assert results['_summary']['success'] == 2
        assert results['_summary']['failed'] == 0


class TestOriginalTablePreserved:
    """Test that original PyArrow Table is preserved."""
    
    def test_table_attribute(self, sample_table):
        """_table attribute stores original PyArrow Table."""
        drawer = DFDraw(sample_table)
        assert drawer._table is sample_table
    
    def test_table_none_for_pandas(self, sample_df):
        """_table is None for pandas input."""
        drawer = DFDraw(sample_df)
        assert drawer._table is None


class TestPandasBackwardCompatibility:
    """Test that all pandas functionality still works."""
    
    def test_df_attribute_pandas(self, sample_df):
        """df attribute works for pandas input."""
        drawer = DFDraw(sample_df)
        assert isinstance(drawer.df, pd.DataFrame)
        assert len(drawer.df) == 1000
    
    def test_df_attribute_pyarrow(self, sample_table):
        """df attribute works for PyArrow input (converted)."""
        drawer = DFDraw(sample_table)
        assert isinstance(drawer.df, pd.DataFrame)
        assert len(drawer.df) == 1000
    
    def test_data_source_preserved(self, sample_table):
        """_data_source preserves original input."""
        drawer = DFDraw(sample_table)
        assert drawer._data_source is sample_table


class TestChunkedArrays:
    """Test handling of chunked PyArrow arrays."""
    
    def test_chunked_array_handling(self, sample_df):
        """Chunked arrays are properly handled via pandas conversion."""
        # Create table with multiple chunks
        table1 = pa.Table.from_pandas(sample_df[:500])
        table2 = pa.Table.from_pandas(sample_df[500:])
        chunked_table = pa.concat_tables([table1, table2])
        
        # Verify it's chunked
        assert chunked_table.column('x').num_chunks == 2
        
        # Create drawer and plot - should work via pandas conversion
        drawer = DFDraw(chunked_table)
        fig, ax, stats = drawer.hist('x', bins=50)
        
        # Should work correctly
        assert stats['n'] == 1000


class TestPyArrowNotAvailable:
    """Test behavior when PyArrow is not available."""
    
    def test_pandas_works_without_pyarrow(self, sample_df):
        """pandas input works regardless of PyArrow availability."""
        drawer = DFDraw(sample_df)
        fig, ax, stats = drawer.hist('x')
        assert stats['n'] == 1000
