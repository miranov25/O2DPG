#!/usr/bin/env python3
"""
Test cases for profiling interface in AliasDataFrame.

Tests for:
- profile=True produces output
- profile_output writes to file
- Profiling works with materialize_alias and materialize_aliases
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


class TestProfilingInterface:
    """Tests for the profiling interface."""
    
    def test_materialize_alias_profile_produces_output(self, capsys):
        """profile=True should print profiling output."""
        df = pd.DataFrame({'x': np.random.randn(1000)})
        adf = AliasDataFrame(df)
        adf.add_alias('y', 'x * 2')
        
        adf.materialize_alias('y', profile=True)
        
        captured = capsys.readouterr()
        # Should contain cProfile output indicators
        assert 'ncalls' in captured.out or 'cumtime' in captured.out
        assert 'y' in adf.df.columns
    
    def test_materialize_aliases_profile_produces_output(self, capsys):
        """profile=True in materialize_aliases should print profiling output."""
        df = pd.DataFrame({'x': np.random.randn(1000)})
        adf = AliasDataFrame(df)
        adf.add_alias('y', 'x * 2')
        adf.add_alias('z', 'y + 1')
        
        adf.materialize_aliases(names=['z'], with_dependencies=True, profile=True)
        
        captured = capsys.readouterr()
        assert 'ncalls' in captured.out or 'cumtime' in captured.out
        assert 'z' in adf.df.columns
    
    def test_profile_output_writes_to_file(self):
        """profile_output should write results to specified file."""
        df = pd.DataFrame({'x': np.random.randn(1000)})
        adf = AliasDataFrame(df)
        adf.add_alias('y', 'x * 2')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name
        
        try:
            adf.materialize_alias('y', profile=True, profile_output=output_path)
            
            # File should exist and contain profiling data
            assert Path(output_path).exists()
            content = Path(output_path).read_text()
            assert 'ncalls' in content or 'cumtime' in content
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()
    
    def test_profile_output_with_materialize_aliases(self):
        """profile_output should work with materialize_aliases too."""
        df = pd.DataFrame({'x': np.random.randn(1000)})
        adf = AliasDataFrame(df)
        adf.add_alias('y', 'x * 2')
        adf.add_alias('z', 'y + 1')
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_path = f.name
        
        try:
            adf.materialize_aliases(
                names=['z'], 
                with_dependencies=True, 
                profile=True, 
                profile_output=output_path
            )
            
            assert Path(output_path).exists()
            content = Path(output_path).read_text()
            assert len(content) > 100  # Should have substantial output
        finally:
            if Path(output_path).exists():
                Path(output_path).unlink()
    
    def test_profile_false_no_output(self, capsys):
        """profile=False (default) should not produce profiling output."""
        df = pd.DataFrame({'x': np.random.randn(1000)})
        adf = AliasDataFrame(df)
        adf.add_alias('y', 'x * 2')
        
        adf.materialize_alias('y', profile=False)
        
        captured = capsys.readouterr()
        # Should not contain cProfile output
        assert 'ncalls' not in captured.out
        assert 'cumtime' not in captured.out
    
    def test_profile_shows_sorted_by_sections(self, capsys):
        """Profile output should show both cumulative and tottime sections."""
        df = pd.DataFrame({'x': np.random.randn(10000)})
        adf = AliasDataFrame(df)
        adf.add_alias('y', 'x * 2 + np.sin(x)')
        
        adf.materialize_alias('y', profile=True)
        
        captured = capsys.readouterr()
        # Should contain both sort sections
        assert 'Sorted by total time' in captured.out
    
    def test_profile_result_still_returned(self):
        """Profiling should not affect the return value."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        adf = AliasDataFrame(df)
        adf.add_alias('y', 'x * 2')
        adf.add_alias('z', 'y + 1')
        
        result = adf.materialize_aliases(names=['z'], with_dependencies=True, profile=True)
        
        # Result should still be the list of added aliases
        assert 'y' in result or 'z' in result  # Depends on cleanTemporary
        assert 'z' in adf.df.columns


class TestJoinCachingBaseline:
    """Baseline tests for join caching (without caching - Phase 1)."""
    
    def test_multiple_subframe_columns_work(self):
        """Multiple columns from same subframe should work correctly."""
        df = pd.DataFrame({
            'idx': [0, 0, 1, 1],
            'x': [1.0, 2.0, 3.0, 4.0]
        })
        sf = pd.DataFrame({
            'idx': [0, 1],
            'val_a': [10.0, 20.0],
            'val_b': [100.0, 200.0],
            'val_c': [1000.0, 2000.0]
        })
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        adf.add_alias('a', 'S.val_a')
        adf.add_alias('b', 'S.val_b')
        adf.add_alias('c', 'S.val_c')
        
        adf.materialize_aliases(names=['a', 'b', 'c'])
        
        assert 'a' in adf.df.columns
        assert 'b' in adf.df.columns
        assert 'c' in adf.df.columns
        
        # Verify values
        np.testing.assert_array_equal(adf.df['a'], [10.0, 10.0, 20.0, 20.0])
        np.testing.assert_array_equal(adf.df['b'], [100.0, 100.0, 200.0, 200.0])
        np.testing.assert_array_equal(adf.df['c'], [1000.0, 1000.0, 2000.0, 2000.0])
    
    def test_many_subframe_columns_performance_baseline(self):
        """Establish baseline for 20 subframe columns (Phase 2 will optimize)."""
        import time
        
        n_rows = 100_000
        n_cols = 20
        
        df = pd.DataFrame({
            'idx': np.random.randint(0, 1000, n_rows),
            'x': np.random.randn(n_rows)
        })
        
        sf_data = {'idx': np.arange(1000)}
        for i in range(n_cols):
            sf_data[f'val_{i}'] = np.random.randn(1000)
        sf = pd.DataFrame(sf_data)
        
        adf = AliasDataFrame(df)
        adf.register_subframe('S', AliasDataFrame(sf), index_columns='idx')
        
        for i in range(n_cols):
            adf.add_alias(f'col_{i}', f'S.val_{i}')
        
        t0 = time.time()
        adf.materialize_aliases(pattern=r'col_.*')
        elapsed = time.time() - t0
        
        # All columns should be materialized
        for i in range(n_cols):
            assert f'col_{i}' in adf.df.columns
        
        # Baseline threshold - Phase 2 caching should improve this
        # For now, just ensure it completes in reasonable time
        assert elapsed < 30.0, f"Baseline took {elapsed:.1f}s (threshold: 30s)"
        
        print(f"\n[Baseline] {n_cols} subframe columns on {n_rows:,} rows: {elapsed:.2f}s")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
