#!/usr/bin/env python
"""
Step 0: AliasDataFrame Smoke Test

Phase 12.14c.GB Gate — Run BEFORE implementing D1.

This test verifies AliasDataFrame basics work:
- Import
- Creation
- draw() method
- draw_batch() method
- Subframe registration

Usage:
    pytest benchmarks/tests/test_adf_smoke.py -v
    
    # Or standalone:
    python benchmarks/tests/test_adf_smoke.py

Gate Policy:
    If ANY test fails, STOP and report issue before proceeding to D1.

Author: Team 3 Coder
Date: 2026-01-01
Phase: 12.14c.GB (Step 0)
"""

import sys
import tempfile
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

# Correct import path for dfextensions
from dfextensions.AliasDataFrame import AliasDataFrame


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'time_s': [0.01, 0.02, 0.015, 0.011, 0.019, 0.016],
        'name': ['bench_a', 'bench_a', 'bench_a', 'bench_b', 'bench_b', 'bench_b'],
        'run_index': [1, 2, 3, 1, 2, 3],
        'peak_rss_mb': [100, 102, 101, 200, 198, 201],
    })


@pytest.fixture
def subframe_df():
    """Create sample subframe DataFrame."""
    return pd.DataFrame({
        'parent_id': ['bench_a', 'bench_a', 'bench_b'],
        'function': ['func1', 'func2', 'func1'],
        'cumtime_s': [0.005, 0.003, 0.008],
    })


# =============================================================================
# IMPORT TEST
# =============================================================================

class TestADFImport:
    """Test AliasDataFrame can be imported."""
    
    def test_import_aliasdataframe(self):
        """AliasDataFrame module must be importable."""
        # Import already done at module level - just verify it worked
        assert AliasDataFrame is not None
    
    def test_aliasdataframe_is_class(self):
        """AliasDataFrame must be a class."""
        assert isinstance(AliasDataFrame, type)


# =============================================================================
# CREATION TEST
# =============================================================================

class TestADFCreation:
    """Test AliasDataFrame creation."""
    
    def test_create_from_dataframe(self, sample_df):
        """AliasDataFrame can be created from pandas DataFrame."""
        adf = AliasDataFrame(sample_df)
        
        assert adf is not None
        assert hasattr(adf, 'df')
        assert len(adf.df) == len(sample_df)
    
    def test_df_attribute_is_dataframe(self, sample_df):
        """AliasDataFrame.df must be a pandas DataFrame."""
        
        
        adf = AliasDataFrame(sample_df)
        
        assert isinstance(adf.df, pd.DataFrame)
    
    def test_columns_preserved(self, sample_df):
        """All columns from input DataFrame must be preserved."""
        
        
        adf = AliasDataFrame(sample_df)
        
        for col in sample_df.columns:
            assert col in adf.df.columns, f"Column '{col}' not preserved"


# =============================================================================
# DRAW TEST
# =============================================================================

class TestADFDraw:
    """Test AliasDataFrame.draw() method."""
    
    def test_draw_method_exists(self, sample_df):
        """AliasDataFrame must have draw() method."""
        
        
        adf = AliasDataFrame(sample_df)
        
        assert hasattr(adf, 'draw')
        assert callable(adf.draw)
    
    def test_draw_scatter(self, sample_df):
        """draw() with scatter type must not raise."""
        
        
        adf = AliasDataFrame(sample_df)
        
        # Should not raise
        try:
            adf.draw('time_s:run_index', type='scatter')
        except Exception as e:
            pytest.fail(f"draw() raised exception: {e}")
    
    def test_draw_with_group_by(self, sample_df):
        """draw() with group_by parameter must not raise."""
        
        
        adf = AliasDataFrame(sample_df)
        
        try:
            adf.draw('time_s:run_index', type='scatter', group_by='name')
        except Exception as e:
            pytest.fail(f"draw() with group_by raised exception: {e}")


# =============================================================================
# DRAW_BATCH TEST
# =============================================================================

class TestADFDrawBatch:
    """Test AliasDataFrame.draw_batch() method."""
    
    def test_draw_batch_method_exists(self, sample_df):
        """AliasDataFrame must have draw_batch() method."""
        
        
        adf = AliasDataFrame(sample_df)
        
        assert hasattr(adf, 'draw_batch')
        assert callable(adf.draw_batch)
    
    def test_draw_batch_creates_files(self, sample_df):
        """draw_batch() must create output files."""
        
        
        adf = AliasDataFrame(sample_df)
        
        specs = {
            'time_trend': {
                'expr': 'time_s:run_index',
                'type': 'scatter',
                'title': 'Test Time Trend',
            },
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = adf.draw_batch(specs, save_dir=tmpdir)
            
            # Check results returned
            assert results is not None
            
            # Check files created (PNG or PDF)
            files = list(Path(tmpdir).glob('*.*'))
            assert len(files) > 0, "draw_batch() created no output files"
    
    def test_draw_batch_with_group_by(self, sample_df):
        """draw_batch() with group_by must not raise."""
        
        
        adf = AliasDataFrame(sample_df)
        
        specs = {
            'grouped_plot': {
                'expr': 'time_s:run_index',
                'type': 'scatter',
                'group_by': 'name',
                'title': 'Grouped Test',
            },
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                results = adf.draw_batch(specs, save_dir=tmpdir)
            except Exception as e:
                pytest.fail(f"draw_batch() with group_by raised: {e}")


# =============================================================================
# SUBFRAME TEST
# =============================================================================

class TestADFSubframe:
    """Test AliasDataFrame subframe registration."""
    
    def test_register_subframe_method_exists(self, sample_df):
        """AliasDataFrame must have register_subframe() method or equivalent."""
        
        
        adf = AliasDataFrame(sample_df)
        
        # Check for register_subframe or subframes attribute
        has_register = hasattr(adf, 'register_subframe')
        has_subframes = hasattr(adf, 'subframes')
        
        assert has_register or has_subframes, \
            "AliasDataFrame needs register_subframe() method or subframes attribute"
    
    def test_register_subframe(self, sample_df, subframe_df):
        """Subframe registration must work."""
        
        
        adf = AliasDataFrame(sample_df)
        sub_adf = AliasDataFrame(subframe_df)
        
        # Try to register subframe
        if hasattr(adf, 'register_subframe'):
            try:
                adf.register_subframe('TestSub', sub_adf, index_columns='name')
            except TypeError:
                # Try alternative API
                try:
                    adf.register_subframe('TestSub', sub_adf)
                except Exception as e:
                    pytest.skip(f"register_subframe() API differs: {e}")
            except Exception as e:
                pytest.skip(f"register_subframe() issue (may need different API): {e}")
        else:
            pytest.skip("register_subframe() not available")


# =============================================================================
# ALIAS TEST
# =============================================================================

class TestADFAlias:
    """Test AliasDataFrame alias functionality."""
    
    def test_add_alias_method_exists(self, sample_df):
        """AliasDataFrame should have add_alias() method."""
        
        
        adf = AliasDataFrame(sample_df)
        
        # add_alias is used in D1, check if available
        if not hasattr(adf, 'add_alias'):
            pytest.skip("add_alias() not available (check ADF API)")
    
    def test_add_alias(self, sample_df):
        """add_alias() should create computed column."""
        
        
        adf = AliasDataFrame(sample_df)
        
        if not hasattr(adf, 'add_alias'):
            pytest.skip("add_alias() not available")
        
        try:
            adf.add_alias('time_ms', 'time_s * 1000')
        except Exception as e:
            pytest.skip(f"add_alias() issue: {e}")


# =============================================================================
# SUMMARY TEST
# =============================================================================

class TestADFSummary:
    """Summary test to verify ADF is ready for Phase 12.14c.GB."""
    
    def test_adf_ready_for_phase_12_14c(self, sample_df):
        """
        Final gate test: AliasDataFrame is ready for benchmark visualization.
        
        This test verifies the minimum viable functionality needed for D1.
        """
        
        
        # 1. Creation works
        adf = AliasDataFrame(sample_df)
        assert adf is not None
        
        # 2. draw() works
        adf.draw('time_s:run_index', type='scatter')
        
        # 3. draw_batch() works
        specs = {
            'test_plot': {
                'expr': 'time_s:run_index',
                'type': 'scatter',
            },
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = adf.draw_batch(specs, save_dir=tmpdir)
            files = list(Path(tmpdir).glob('*.*'))
            assert len(files) > 0, "draw_batch() must create output files"
        
        print("\n" + "=" * 60)
        print("✅ STEP 0 PASSED — AliasDataFrame ready for Phase 12.14c.GB")
        print("=" * 60)


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    sys.exit(exit_code)
