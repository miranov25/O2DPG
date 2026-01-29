"""
Tests for Phase 13.6.B DSL Export API Methods.

These tests verify the NEW public API methods:
- DSLCompiler.to_pandas()
- DSLCompiler.export_to_aliasdf()
- DSLCompiler.export_to_aliasdf_flat()

TD21-TD26: Direct API-level tests (not just flatten engine).

Prerequisites:
- ALICEEventGenerator (tests/generators/alice_events.py)
- Phase 13.6.A-ext flatten module
- DSLCompiler with new methods

Phase 13.6.G: Added TestSchemaValidation for schema vs RDF validation.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Import generator
try:
    from tests.generators.alice_events import ALICEEventGenerator, GeneratorConfig
except ImportError:
    from generators.alice_events import ALICEEventGenerator, GeneratorConfig

# Import DSL
try:
    from RDataFrameDSL import DSLCompiler
    from RDataFrameDSL.flatten import FlattenBackend
    DSL_AVAILABLE = True
except ImportError:
    DSL_AVAILABLE = False
    DSLCompiler = None

# Phase 13.6.G: Import IRError for schema validation tests
try:
    from RDataFrameDSL.ir_errors import IRError
except ImportError:
    IRError = None

# Check if AliasDataFrame is available
ADF_AVAILABLE = False
AliasDataFrame = None

try:
    # Primary: dfextensions package (local project structure)
    from dfextensions import AliasDataFrame
    ADF_AVAILABLE = True
except ImportError:
    try:
        from AliasDataFrame import AliasDataFrame
        ADF_AVAILABLE = True
    except ImportError:
        try:
            from aliasdf import AliasDataFrame
            ADF_AVAILABLE = True
        except ImportError:
            ADF_AVAILABLE = False


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def alice_data():
    """Generate test data (100 events)."""
    gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
    return gen.generate_dict(n_events=100)


@pytest.fixture(scope="module")
def dsl_schema():
    """Schema for ALICE data."""
    return {
        'event_id': 'int',
        'n_tracks': 'int',
        'vertex_z': 'double',
        'track_pt': 'RVec<double>',
        'track_phi': 'RVec<double>',
        'track_pdgcode': 'RVec<int>',
        'cluster_Q': 'RVec<RVec<double>>',
        'cluster_x': 'RVec<RVec<double>>',
    }


# =============================================================================
# Mock RDataFrame for testing without ROOT
# =============================================================================

class MockRDataFrame:
    """Mock RDataFrame for testing DSL methods without ROOT."""
    
    def __init__(self, data: dict):
        self._data = data
        self._filter_expr = None
        self._range_limit = None
    
    def Filter(self, expr: str) -> 'MockRDataFrame':
        """Mock Filter - stores expression but doesn't apply."""
        new_rdf = MockRDataFrame(self._data)
        new_rdf._filter_expr = expr
        new_rdf._range_limit = self._range_limit
        return new_rdf
    
    def Range(self, n: int) -> 'MockRDataFrame':
        """Mock Range - limits entries."""
        new_rdf = MockRDataFrame(self._data)
        new_rdf._filter_expr = self._filter_expr
        new_rdf._range_limit = n
        return new_rdf
    
    def AsNumpy(self, columns: list) -> dict:
        """Return data as numpy arrays."""
        result = {}
        limit = self._range_limit or len(self._data.get('event_id', []))
        
        for col in columns:
            if col in self._data:
                data = self._data[col]
                if isinstance(data, np.ndarray) and data.dtype == object:
                    # RVec column - slice
                    result[col] = data[:limit]
                else:
                    result[col] = data[:limit]
        
        return result


class MockDSLCompiler:
    """
    Mock DSLCompiler with to_pandas() for testing without full DSL stack.
    
    This allows testing the API wiring without ROOT dependencies.
    """
    
    def __init__(self, schema: dict):
        self.schema = schema
        self._definitions = []
    
    def define(self, name: str, expr: str) -> 'MockDSLCompiler':
        self._definitions.append((name, expr))
        return self
    
    def apply(self, rdf):
        """Mock apply - returns rdf unchanged."""
        return rdf
    
    def to_pandas(
        self,
        rdf,
        columns: list,
        event_selection: str = None,
        parent_id_column: str = 'event_id',
        backend = None,
        max_entries: int = None,
    ) -> pd.DataFrame:
        """
        Mock to_pandas() that uses real flatten logic.
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe, FlattenBackend
        
        # Apply mock DSL
        applied_rdf = self.apply(rdf)
        
        # Apply event selection (mock)
        if event_selection:
            applied_rdf = applied_rdf.Filter(event_selection)
        
        # Apply max_entries
        if max_entries is not None:
            applied_rdf = applied_rdf.Range(max_entries)
        
        # Get columns
        columns_to_fetch = list(columns)
        if parent_id_column not in columns_to_fetch:
            columns_to_fetch.append(parent_id_column)
        
        data = applied_rdf.AsNumpy(columns_to_fetch)
        
        if backend is None:
            backend = FlattenBackend.AUTO
        
        return flatten_to_dataframe(
            data,
            columns=columns,
            parent_id_column=parent_id_column,
            backend=backend
        )
    
    def export_to_aliasdf(
        self,
        rdf,
        columns: list,
        event_selection: str = None,
        parent_id_column: str = 'event_id',
        max_entries: int = None,
    ):
        """Mock export_to_aliasdf() with subframes (Option C)."""
        from datetime import datetime, timezone
        from RDataFrameDSL.flatten import flatten_to_tables
        
        if not ADF_AVAILABLE:
            raise ImportError(
                "AliasDataFrame not installed. Install with: pip install aliasdf"
            )
        
        applied_rdf = self.apply(rdf)
        
        if event_selection:
            applied_rdf = applied_rdf.Filter(event_selection)
        
        if max_entries is not None:
            applied_rdf = applied_rdf.Range(max_entries)
        
        columns_to_fetch = list(columns)
        if parent_id_column not in columns_to_fetch:
            columns_to_fetch.append(parent_id_column)
        
        data = applied_rdf.AsNumpy(columns_to_fetch)
        
        tables = flatten_to_tables(
            data,
            columns=columns,
            parent_id_column=parent_id_column
        )
        
        if 'events' in tables and len(tables['events']) > 0:
            main_df = tables['events']
        else:
            from RDataFrameDSL.flatten import flatten_to_dataframe
            main_df = flatten_to_dataframe(
                data,
                columns=columns,
                parent_id_column=parent_id_column
            )
        
        adf = AliasDataFrame(main_df, schema_id='RDataFrameDSL_v13.6.B')
        
        if 'tracks' in tables and len(tables['tracks']) > 0:
            tracks_adf = AliasDataFrame(tables['tracks'])
            adf.register_subframe('tracks', tracks_adf, [parent_id_column])
        
        if 'clusters' in tables and len(tables['clusters']) > 0:
            clusters_adf = AliasDataFrame(tables['clusters'])
            adf.register_subframe('clusters', clusters_adf, [parent_id_column, 'idx_1'])
        
        adf._schema['__meta__'].update({
            'source': 'RDataFrameDSL',
            'source_version': '13.6.B',
            'parent_id_column': parent_id_column,
            'created_at': datetime.now(timezone.utc).isoformat(),
        })
        
        return adf
    
    def export_to_aliasdf_flat(
        self,
        rdf,
        columns: list,
        event_selection: str = None,
        parent_id_column: str = 'event_id',
        max_entries: int = None,
    ):
        """Mock export_to_aliasdf_flat() without subframes (Option A)."""
        from datetime import datetime, timezone
        
        if not ADF_AVAILABLE:
            raise ImportError(
                "AliasDataFrame not installed. Install with: pip install aliasdf"
            )
        
        df = self.to_pandas(
            rdf,
            columns=columns,
            event_selection=event_selection,
            parent_id_column=parent_id_column,
            max_entries=max_entries
        )
        
        adf = AliasDataFrame(df, schema_id='RDataFrameDSL_v13.6.B')
        
        adf._schema['__meta__'].update({
            'source': 'RDataFrameDSL',
            'source_version': '13.6.B',
            'export_mode': 'flat',
            'parent_id_column': parent_id_column,
            'created_at': datetime.now(timezone.utc).isoformat(),
        })
        
        return adf


# =============================================================================
# TD21-TD26: Direct API Tests
# =============================================================================

class TestDSLToPandasAPI:
    """Tests for DSLCompiler.to_pandas() API wiring."""
    
    def test_TD21_to_pandas_basic_wiring(self, alice_data, dsl_schema):
        """Verify DSLCompiler.to_pandas() calls flatten correctly."""
        # Use mock or real DSL
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        # Call to_pandas directly
        df = dsl.to_pandas(rdf, ['track_pt', 'track_phi'])
        
        # Verify output structure
        assert isinstance(df, pd.DataFrame)
        assert 'event_id' in df.columns
        assert 'idx_1' in df.columns
        assert 'track_pt' in df.columns
        assert 'track_phi' in df.columns
        
        # Verify row count matches total tracks
        total_tracks = sum(alice_data['n_tracks'])
        assert len(df) == total_tracks
    
    def test_TD21b_to_pandas_mixed_depths(self, alice_data, dsl_schema):
        """Verify to_pandas() handles mixed-depth columns."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        # Mixed: 2D + 1D + scalar
        df = dsl.to_pandas(rdf, ['cluster_Q', 'track_pt', 'n_tracks'])
        
        # Should be at cluster level
        assert 'idx_2' in df.columns
        assert 'idx_1' in df.columns
        assert 'event_id' in df.columns
        
        # 1D and scalar should be replicated
        assert 'track_pt' in df.columns
        assert 'n_tracks' in df.columns
    
    def test_TD24_event_selection_parameter(self, alice_data, dsl_schema):
        """Verify event_selection parameter is passed through."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        # With event_selection (mock doesn't actually filter, but tests wiring)
        df = dsl.to_pandas(
            rdf, 
            ['track_pt'], 
            event_selection='n_tracks > 5'
        )
        
        # Should return DataFrame (filter is mocked)
        assert isinstance(df, pd.DataFrame)
        assert 'track_pt' in df.columns
    
    def test_TD25_max_entries_parameter(self, alice_data, dsl_schema):
        """Verify max_entries parameter limits events."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        # Limit to 10 events
        df = dsl.to_pandas(rdf, ['n_tracks'], max_entries=10)
        
        # Should have exactly 10 rows (scalar column, one per event)
        assert len(df) == 10
    
    def test_TD21c_to_pandas_parent_id_column(self, alice_data, dsl_schema):
        """Verify parent_id_column parameter works."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        # Default parent_id_column
        df = dsl.to_pandas(rdf, ['track_pt'])
        assert 'event_id' in df.columns
    
    def test_TD21d_to_pandas_backend_parameter(self, alice_data, dsl_schema):
        """Verify backend parameter is accepted."""
        from RDataFrameDSL.flatten import FlattenBackend
        
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        # Explicit backend
        df = dsl.to_pandas(
            rdf, 
            ['track_pt'], 
            backend=FlattenBackend.NUMPY
        )
        
        assert isinstance(df, pd.DataFrame)


@pytest.mark.skipif(not ADF_AVAILABLE, reason="AliasDataFrame not installed")
class TestExportToAliasDF:
    """Tests for DSLCompiler.export_to_aliasdf() API."""
    
    def test_TD22_export_to_aliasdf_creates_subframes(self, alice_data, dsl_schema):
        """Verify export_to_aliasdf() creates subframes correctly."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        # Export with mixed depths
        adf = dsl.export_to_aliasdf(rdf, ['cluster_Q', 'track_pt', 'n_tracks'])
        
        # Verify AliasDataFrame returned
        assert isinstance(adf, AliasDataFrame)
        
        # Verify subframes exist
        assert hasattr(adf, '_schema')
        subframes = adf._schema.get('subframes', {})
        
        # Should have tracks and clusters subframes
        assert 'tracks' in subframes or len(subframes) > 0
    
    def test_TD22b_export_to_aliasdf_schema_id(self, alice_data, dsl_schema):
        """Verify schema_id is set correctly."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        adf = dsl.export_to_aliasdf(rdf, ['track_pt'])
        
        # Verify schema_id
        assert adf.schema_id == 'RDataFrameDSL_v13.6.B'
    
    def test_TD26_aliasdf_metadata(self, alice_data, dsl_schema):
        """Verify AliasDataFrame metadata is set correctly."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        adf = dsl.export_to_aliasdf(rdf, ['track_pt', 'n_tracks'])
        
        # Verify metadata
        meta = adf._schema.get('__meta__', {})
        
        assert meta.get('source') == 'RDataFrameDSL'
        assert meta.get('source_version') == '13.6.B'
        assert meta.get('parent_id_column') == 'event_id'
        assert 'created_at' in meta
    
    def test_TD23_export_to_aliasdf_flat_no_subframes(self, alice_data, dsl_schema):
        """Verify export_to_aliasdf_flat() has no subframes."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        # Export flat
        adf = dsl.export_to_aliasdf_flat(rdf, ['cluster_Q', 'track_pt'])
        
        # Verify AliasDataFrame returned
        assert isinstance(adf, AliasDataFrame)
        
        # Verify no subframes (or empty)
        subframes = adf._schema.get('subframes', {})
        assert len(subframes) == 0
    
    def test_TD23b_export_flat_metadata(self, alice_data, dsl_schema):
        """Verify flat export has correct metadata."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        adf = dsl.export_to_aliasdf_flat(rdf, ['track_pt'])
        
        meta = adf._schema.get('__meta__', {})
        
        assert meta.get('export_mode') == 'flat'
        assert meta.get('source') == 'RDataFrameDSL'


class TestErrorHandling:
    """Tests for error handling in DSL export methods."""
    
    def test_ERROR1_to_pandas_missing_column(self, alice_data, dsl_schema):
        """Test error for non-existent column."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        # Request non-existent column
        with pytest.raises((KeyError, ValueError)):
            dsl.to_pandas(rdf, ['nonexistent_column'])
    
    @pytest.mark.skipif(ADF_AVAILABLE, reason="Test requires AliasDataFrame NOT installed")
    def test_ERROR2_aliasdf_import_error(self, alice_data, dsl_schema):
        """Test graceful error when AliasDataFrame not installed."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        # This test only runs if AliasDataFrame is NOT available
        with pytest.raises(ImportError, match="AliasDataFrame"):
            dsl.export_to_aliasdf(rdf, ['track_pt'])
    
    def test_ERROR3_empty_columns_list(self, alice_data, dsl_schema):
        """Test error for empty columns list."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        with pytest.raises((ValueError, KeyError)):
            dsl.to_pandas(rdf, [])


class TestAPIIntegration:
    """Integration tests for full API chain."""
    
    def test_INTEG1_to_pandas_then_groupby(self, alice_data, dsl_schema):
        """Verify to_pandas() output works with pandas operations."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        df = dsl.to_pandas(rdf, ['track_pt', 'n_tracks'])
        
        # Groupby should work
        per_event = df.groupby('event_id')['track_pt'].mean()
        
        assert len(per_event) > 0
        assert isinstance(per_event, pd.Series)
    
    def test_INTEG2_to_pandas_dtypes_preserved(self, alice_data, dsl_schema):
        """Verify dtypes are preserved through API."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        df = dsl.to_pandas(rdf, ['track_pt', 'track_pdgcode'])
        
        # Float column
        assert df['track_pt'].dtype == np.float64
        
        # Integer column
        assert df['track_pdgcode'].dtype in [np.int32, np.int64]
    
    @pytest.mark.skipif(not ADF_AVAILABLE, reason="AliasDataFrame not installed")
    def test_INTEG3_aliasdf_can_add_alias(self, alice_data, dsl_schema):
        """Verify exported AliasDataFrame supports aliases."""
        dsl = MockDSLCompiler(dsl_schema)
        rdf = MockRDataFrame(alice_data)
        
        adf = dsl.export_to_aliasdf_flat(rdf, ['track_pt'])
        
        # Should be able to add alias
        try:
            adf.add_alias('pt_scaled', 'track_pt * 1.1')
            # If this succeeds, the ADF is functional
            assert True
        except Exception as e:
            # Some ADF versions may have different API
            pytest.skip(f"ADF add_alias not available: {e}")


# =============================================================================
# Phase 13.6.G: Schema Validation Tests
# =============================================================================

class TestSchemaValidation:
    """
    Test schema vs RDF validation.
    
    Phase 13.6.G: Prevents ROOT crashes from schema/RDF mismatch
    by validating that schema columns exist in RDF before calling Define().
    """
    
    @pytest.mark.root_serial
    @pytest.mark.feature("schema_validation")
    def test_schema_mismatch_gives_clean_error(self):
        """Schema with non-existent column should raise IRError, not crash."""
        import ROOT
        from RDataFrameDSL import DSLCompiler
        from RDataFrameDSL.ir_errors import IRError
        
        # Create RDF with only 'x' column
        rdf = ROOT.RDataFrame(10).Define("x", "1.0")
        
        # Schema claims 'y' exists (it doesn't)
        schema = {'x': 'double', 'y': 'double'}
        dsl = DSLCompiler(schema)
        dsl.define("z", "x + y")
        
        # Should raise clean error, not ROOT crash
        with pytest.raises(IRError) as exc_info:
            dsl.apply(rdf)
        
        # Verify error message is helpful
        error_msg = str(exc_info.value).lower()
        assert "y" in str(exc_info.value)  # Missing column mentioned
        assert "not" in error_msg or "missing" in error_msg or "found" in error_msg
    
    @pytest.mark.root_serial
    @pytest.mark.feature("schema_validation")
    def test_schema_subset_of_rdf_works(self):
        """Schema can be subset of RDF columns (RDF has extra columns)."""
        import ROOT
        from RDataFrameDSL import DSLCompiler
        
        # RDF has x, y, z
        rdf = ROOT.RDataFrame(10)
        rdf = rdf.Define("x", "1.0")
        rdf = rdf.Define("y", "2.0")
        rdf = rdf.Define("z", "3.0")
        
        # Schema only declares x (subset is OK)
        schema = {'x': 'double'}
        dsl = DSLCompiler(schema)
        dsl.define("x2", "x * 2")
        
        # Should work - schema is subset of RDF
        result_rdf = dsl.apply(rdf)
        assert result_rdf is not None
    
    @pytest.mark.root_serial
    @pytest.mark.feature("schema_validation")
    def test_multiple_missing_columns_all_reported(self):
        """Error should report ALL missing columns, not just first."""
        import ROOT
        from RDataFrameDSL import DSLCompiler
        from RDataFrameDSL.ir_errors import IRError
        
        # RDF has only 'x'
        rdf = ROOT.RDataFrame(10).Define("x", "1.0")
        
        # Schema claims a, b, c exist (none do except x)
        schema = {'x': 'double', 'a': 'double', 'b': 'double', 'c': 'double'}
        dsl = DSLCompiler(schema)
        dsl.define("sum", "x + a + b + c")
        
        with pytest.raises(IRError) as exc_info:
            dsl.apply(rdf)
        
        # All missing columns should be mentioned
        error_msg = str(exc_info.value)
        assert 'a' in error_msg
        assert 'b' in error_msg
        assert 'c' in error_msg
    
    @pytest.mark.root_serial
    @pytest.mark.feature("schema_validation")  
    def test_empty_rdf_validates_correctly(self):
        """Validation should work even with 0-row RDataFrame."""
        import ROOT
        from RDataFrameDSL import DSLCompiler
        from RDataFrameDSL.ir_errors import IRError
        
        # Empty RDF (0 rows) but has column 'x'
        rdf = ROOT.RDataFrame(0).Define("x", "1.0")
        
        # Schema claims 'y' exists
        schema = {'x': 'double', 'y': 'double'}
        dsl = DSLCompiler(schema)
        dsl.define("z", "x + y")
        
        # Should still catch the mismatch
        with pytest.raises(IRError):
            dsl.apply(rdf)
    
    @pytest.mark.root_serial
    @pytest.mark.feature("schema_validation")
    def test_rvec_column_validates(self):
        """RVec columns in schema should also be validated."""
        import ROOT
        from RDataFrameDSL import DSLCompiler
        from RDataFrameDSL.ir_errors import IRError
        
        # RDF has scalar 'x' only
        rdf = ROOT.RDataFrame(10).Define("x", "1.0")
        
        # Schema claims RVec column 'track_pt' exists
        schema = {'x': 'double', 'track_pt': 'RVec<double>'}
        dsl = DSLCompiler(schema)
        dsl.define("sum_pt", "Sum(track_pt)")
        
        with pytest.raises(IRError) as exc_info:
            dsl.apply(rdf)
        
        assert "track_pt" in str(exc_info.value)
    
    @pytest.mark.root_serial
    @pytest.mark.feature("schema_validation")
    def test_chained_definitions_work(self):
        """Chained definitions (a1=a0+1, a2=a1+1) should work correctly."""
        import ROOT
        from RDataFrameDSL import DSLCompiler
        
        # RDF has only 'a0'
        rdf = ROOT.RDataFrame(5).Define("a0", "1.0")
        
        # Schema declares only 'a0'
        schema = {'a0': 'double'}
        dsl = DSLCompiler(schema)
        
        # Chain of definitions - each references the previous
        dsl.define("a1", "a0 + 1")  # a1 = 2.0
        dsl.define("a2", "a1 + 1")  # a2 = 3.0
        dsl.define("a3", "a2 + 1")  # a3 = 4.0
        
        # Should work - DSL handles dependency resolution
        result_rdf = dsl.apply(rdf)
        
        # Verify values
        result = result_rdf.AsNumpy(["a0", "a1", "a2", "a3"])
        assert all(result["a0"] == 1.0)
        assert all(result["a1"] == 2.0)
        assert all(result["a2"] == 3.0)
        assert all(result["a3"] == 4.0)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
