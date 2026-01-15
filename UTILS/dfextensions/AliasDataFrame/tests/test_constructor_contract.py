"""
test_constructor_contract.py - P0 API Contract Tests

Phase 13.6.B - Prevents Team 2 (RDataFrameDSL) from using wrong
parameter names that will cause TypeError in production.

Created: 2026-01-15
Priority: P0 (Blocking for Phase 13.6.B)
Tests: 6 critical contract points

These tests lock the AliasDataFrame API contract based on Phase 13.6.B
review findings. Multiple reviewers gave conflicting answers about
parameter names - these tests prevent integration bugs by rejecting
incorrect parameter names that were suggested.

Run with: pytest test_constructor_contract.py -v
"""

import pytest
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


class TestAPIContract:
    """Verify API contract for Phase 13.6.B integration."""
    
    @pytest.fixture
    def sample_df(self):
        """Sample DataFrame for testing."""
        return pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    
    # =========================================================================
    # Constructor Tests
    # =========================================================================
    
    def test_constructor_accepts_schema_id(self, sample_df):
        """Verify schema_id parameter is supported.
        
        This is the correct way to attach provenance information
        at construction time. Team 2 should use:
            AliasDataFrame(df, schema_id='RDataFrameDSL_v13.6.B')
        """
        adf = AliasDataFrame(sample_df, schema_id='test_v1')
        assert adf.schema_id == 'test_v1'
    
    def test_constructor_rejects_schema_param(self, sample_df):
        """Verify schema= parameter does NOT exist.
        
        CRITICAL: Prevents Team 2 from trying to pass schema
        directly to constructor. If this test fails, it means
        the API changed and Phase 13.6.B review is invalid.
        
        Team 2 should use update_schema() instead:
            adf = AliasDataFrame(df, schema_id='...')
            adf.update_schema({'columns': {...}})
        """
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            AliasDataFrame(sample_df, schema={'columns': {}})
    
    def test_constructor_rejects_metadata_param(self, sample_df):
        """Verify metadata= parameter does NOT exist.
        
        CRITICAL: Prevents Team 2 from trying to pass metadata
        directly to constructor. If this test fails, it means
        the API changed and Phase 13.6.B review is invalid.
        
        Team 2 should use _schema['__meta__'] instead:
            adf = AliasDataFrame(df, schema_id='...')
            adf._schema['__meta__'].update({'source': 'RDataFrameDSL'})
        """
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            AliasDataFrame(sample_df, metadata={'source': 'test'})
    
    # =========================================================================
    # register_subframe Tests
    # =========================================================================
    
    def test_register_subframe_correct_params(self, sample_df):
        """Verify register_subframe accepts: name, adf, index_columns.
        
        This is the correct API signature verified from source code
        (AliasDataFrame.py:1461). Team 2 should use:
            adf.register_subframe('tracks', tracks_adf, ['event_id'])
        
        Or with keywords:
            adf.register_subframe(
                name='tracks',
                adf=tracks_adf,
                index_columns=['event_id']
            )
        """
        adf = AliasDataFrame(sample_df)
        sub = AliasDataFrame(sample_df.copy())
        
        # Positional form (most common)
        adf.register_subframe('T', sub, 'x')
        
        assert 'T' in adf._schema['subframes']
    
    def test_register_subframe_rejects_subframe_param(self, sample_df):
        """Verify subframe= parameter does NOT exist (use 'adf').
        
        CRITICAL: Prevents Team 2 from using wrong parameter name
        that GPT2 and Gemini3 incorrectly suggested in reviews.
        
        WRONG (will fail):
            adf.register_subframe('T', subframe=sub, index_columns='x')
        
        CORRECT:
            adf.register_subframe('T', adf=sub, index_columns='x')
        """
        adf = AliasDataFrame(sample_df)
        sub = AliasDataFrame(sample_df.copy())
        
        with pytest.raises(TypeError):
            adf.register_subframe('T', subframe=sub, index_columns='x')
    
    def test_register_subframe_rejects_on_param(self, sample_df):
        """Verify on= parameter does NOT exist (use 'index_columns').
        
        CRITICAL: Prevents Team 2 from using wrong parameter name
        that GPT2 incorrectly suggested in review (confusion with
        pandas merge semantics).
        
        WRONG (will fail):
            adf.register_subframe('T', sub, on='x')
        
        CORRECT:
            adf.register_subframe('T', sub, index_columns='x')
        """
        adf = AliasDataFrame(sample_df)
        sub = AliasDataFrame(sample_df.copy())
        
        with pytest.raises(TypeError):
            adf.register_subframe('T', sub, on='x')


if __name__ == '__main__':
    # Allow running as standalone script
    pytest.main([__file__, '-v'])
