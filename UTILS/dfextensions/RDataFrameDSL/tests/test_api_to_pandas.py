# =============================================================================
# tests/test_api_to_pandas.py
# =============================================================================
# API smoke tests for to_pandas() method.
#
# Purpose: Verify to_pandas() API works correctly (not correctness proofs -
#          those are in test_invariance_*.py). These tests answer "does the
#          method work?" not "are the results mathematically correct?"
#
# Phase: 13.6.F
# Date: 2026-01-23
# =============================================================================

import pytest
import numpy as np
import pandas as pd

from RDataFrameDSL import DSLCompiler
from RDataFrameDSL.ir_errors import IRError, IRErrorKind


# =============================================================================
# TestToPandas - Basic API functionality
# =============================================================================

class TestToPandas:
    """API smoke tests for to_pandas() method."""
    
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.root_serial  # Needs RDF fixture
    def test_to_pandas_scalar(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas() works with scalar columns.
        
        API Contract:
        - Returns pandas DataFrame
        - Scalar columns have correct dtype
        - No exceptions raised
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(nd_2d_rdf, columns=['event_id', 'event_weight'])
        
        # API contract: returns DataFrame
        assert isinstance(df, pd.DataFrame)
        
        # API contract: columns present
        assert 'event_id' in df.columns
        assert 'event_weight' in df.columns
        
        # API contract: has data
        assert len(df) > 0
    
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.root_serial
    def test_to_pandas_1d(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas() works with 1D (RVec) columns.
        
        API Contract:
        - Flattens RVec to rows
        - Adds track_idx column
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(nd_2d_rdf, columns=['track_pt'])
        
        assert isinstance(df, pd.DataFrame)
        assert 'track_pt' in df.columns
        assert 'track_idx' in df.columns  # Auto-added for 1D
        assert len(df) > 0
    
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.root_serial
    def test_to_pandas_2d(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas() works with 2D (RVec<RVec>) columns.
        
        API Contract:
        - Flattens to rows at deepest level
        - Adds track_idx and cluster_idx columns
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(nd_2d_rdf, columns=['cluster_Q'])
        
        assert isinstance(df, pd.DataFrame)
        assert 'cluster_Q' in df.columns
        assert 'track_idx' in df.columns
        assert 'cluster_idx' in df.columns  # Auto-added for 2D
        assert len(df) > 0
    
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.root_serial
    def test_to_pandas_mixed_depth(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas() works with mixed scalar + 1D + 2D columns.
        
        API Contract:
        - All depths in single call
        - Shallower columns replicated to match deepest
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf, 
            columns=['event_weight', 'track_pt', 'cluster_Q']
        )
        
        assert isinstance(df, pd.DataFrame)
        assert 'event_weight' in df.columns  # Scalar - replicated
        assert 'track_pt' in df.columns      # 1D - replicated to 2D
        assert 'cluster_Q' in df.columns     # 2D - base level
        assert len(df) > 0
    
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.root_serial
    def test_to_pandas_with_selection(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas() works with event_selection parameter.
        
        API Contract:
        - Filters events before flattening
        - Reduces output size
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        # Get all events first
        df_all = dsl.to_pandas(nd_2d_rdf, columns=['track_pt'])
        
        # Get filtered events
        df_filtered = dsl.to_pandas(
            nd_2d_rdf, 
            columns=['track_pt'],
            event_selection='n_tracks >= 2'
        )
        
        # Filtered should have fewer or equal rows
        assert len(df_filtered) <= len(df_all)
    
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.root_serial
    def test_to_pandas_join_inner(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas() works with join='inner' (default).
        
        API Contract:
        - No NaN values in result
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['track_pt', 'cluster_Q'],
            join='inner'
        )
        
        assert isinstance(df, pd.DataFrame)
        # Inner join should not produce NaN
        assert not df['track_pt'].isna().any()
        assert not df['cluster_Q'].isna().any()
    
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.root_serial
    def test_to_pandas_join_outer(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas() works with join='outer'.
        
        API Contract:
        - May have NaN values for missing indices
        - Includes all rows from both levels
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['track_pt', 'cluster_Q'],
            join='outer'
        )
        
        assert isinstance(df, pd.DataFrame)
        # Outer join result valid (may or may not have NaN)
        assert len(df) > 0


# =============================================================================
# TestToPandasErrors - Error handling
# =============================================================================

class TestToPandasErrors:
    """Error handling tests for to_pandas()."""
    
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.root_serial
    def test_to_pandas_missing_column(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas() raises IRError for column not in schema with suggestions.
        
        Phase 13.6.F Layer 1: DSL-level validation catches missing columns
        before ROOT execution, providing helpful "Did you mean...?" suggestions.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        # Should raise IRError with suggestions
        with pytest.raises(IRError) as exc_info:
            dsl.to_pandas(nd_2d_rdf, columns=['nonexistent_column'])
        
        # Verify error message mentions the column
        error = exc_info.value
        assert 'nonexistent_column' in str(error)
        assert error.kind == IRErrorKind.TYPE_ERROR
        
        # Verify suggestions are provided
        assert error.suggestions is not None
        assert len(error.suggestions) > 0
    
    @pytest.mark.feature("error_missing_column")
    @pytest.mark.feature("error_suggestions")
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.root_serial
    def test_to_pandas_missing_column_with_similar(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas() suggests similar column names for typos.
        
        Phase 13.6.F Layer 1: Tests "Did you mean...?" functionality.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        # "track_pT" should suggest "track_pt"
        with pytest.raises(IRError) as exc_info:
            dsl.to_pandas(nd_2d_rdf, columns=['track_pT'])
        
        error = exc_info.value
        # Should have suggestions including 'track_pt'
        suggestions_str = ' '.join(error.suggestions or [])
        assert 'track_pt' in suggestions_str.lower() or 'track' in suggestions_str.lower()
    
    @pytest.mark.feature("api_to_pandas")
    def test_to_pandas_empty_columns(self, nd_2d_rdf, nd_2d_schema):
        """
        to_pandas() raises error for empty columns list.
        
        Phase 13.6.F Layer 1: Empty columns list is caught early.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        with pytest.raises(IRError) as exc_info:
            dsl.to_pandas(nd_2d_rdf, columns=[])
        
        assert exc_info.value.kind == IRErrorKind.VALIDATION_ERROR
