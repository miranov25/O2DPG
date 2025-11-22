"""
test_schema_serialization.py - Phase 4b schema serialization tests

Tests for unified schema serialization in both ROOT and Parquet formats:
- Schema roundtrip (export → import → schema restored)
- Alias definitions preserved
- Compression metadata preserved
- Dtypes preserved (float16, int8, int16)
- Subframe metadata preserved
- Invariance tests (materialize before/after)
- Missing-key join behavior after roundtrip
- Recursive subframes (T → T2)
- Entry slicing with roundtrip
- Corrupted JSON handling

Run with: pytest test_schema_serialization.py -v
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import warnings

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import (
    AliasDataFrame, 
    SCHEMA_METADATA_KEY, 
    SCHEMA_VERSION,
    _serialize_schema,
    _deserialize_schema
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def simple_df():
    """Simple DataFrame for basic tests."""
    return pd.DataFrame({
        'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        'y': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float16),
        'z': np.array([100, 200, 300, 400, 500], dtype=np.int16),
        'key': np.array([0, 1, 2, 3, 4], dtype=np.int8),
    })


@pytest.fixture
def simple_adf(simple_df):
    """AliasDataFrame with aliases for testing."""
    adf = AliasDataFrame(simple_df.copy())
    adf.add_alias('sum_xy', 'x + y', dtype=np.float32)
    adf.add_alias('ratio', 'y / x', dtype=np.float64)
    adf.add_alias('const_pi', '3.14159', dtype=np.float32, is_constant=True)
    return adf


@pytest.fixture
def subframe_df():
    """DataFrame for subframe tests."""
    return pd.DataFrame({
        'key': np.array([0, 1, 2, 3, 4], dtype=np.int8),
        'offset': np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        'label': np.array([10, 20, 30, 40, 50], dtype=np.int16),
    })


@pytest.fixture
def adf_with_subframe(simple_df, subframe_df):
    """AliasDataFrame with a subframe registered."""
    adf = AliasDataFrame(simple_df.copy())
    adf.add_alias('adjusted', 'x + T.offset', dtype=np.float32)
    
    sub_adf = AliasDataFrame(subframe_df.copy())
    sub_adf.add_alias('double_offset', 'offset * 2', dtype=np.float32)
    
    adf.register_subframe('T', sub_adf, 'key')
    return adf


# =============================================================================
# Test: Serialization Helpers
# =============================================================================

class TestSerializationHelpers:
    """Tests for _serialize_schema and _deserialize_schema functions."""

    def test_serialize_schema_basic(self, simple_adf):
        """Verify _serialize_schema produces JSON-serializable output."""
        serialized = _serialize_schema(simple_adf._schema)
        
        # Should be JSON-serializable
        json_str = json.dumps(serialized)
        assert isinstance(json_str, str)
        
        # Should have required keys
        assert 'schema_version' in serialized
        assert 'columns' in serialized
        assert 'compression' in serialized
        assert 'subframes' in serialized

    def test_serialize_schema_converts_dtypes(self, simple_adf):
        """Verify dtypes are converted to strings."""
        serialized = _serialize_schema(simple_adf._schema)
        
        for name, spec in serialized['columns'].items():
            if 'dtype' in spec:
                assert isinstance(spec['dtype'], str)

    def test_deserialize_schema_restores_dtypes(self, simple_adf):
        """Verify deserialization restores dtype types."""
        serialized = _serialize_schema(simple_adf._schema)
        restored = _deserialize_schema(serialized)
        
        for name, spec in restored['columns'].items():
            if 'dtype' in spec:
                # Should be a numpy type, not a string
                assert hasattr(spec['dtype'], '__name__') or spec['dtype'] is None

    def test_serialize_deserialize_roundtrip(self, simple_adf):
        """Verify serialize → deserialize preserves data."""
        original = simple_adf._schema
        serialized = _serialize_schema(original)
        restored = _deserialize_schema(serialized)
        
        # Check columns
        assert set(original['columns'].keys()) == set(restored['columns'].keys())
        for name in original['columns']:
            assert original['columns'][name].get('expr') == restored['columns'][name].get('expr')

    def test_deserialize_warns_on_future_version(self):
        """Verify warning on schema version newer than supported."""
        future_schema = {
            'schema_version': 999,
            'columns': {},
            'compression': {},
            'subframes': {}
        }
        
        with pytest.warns(UserWarning, match="newer than supported"):
            _deserialize_schema(future_schema)


# =============================================================================
# Test: Property Setters Raise AttributeError (API Hardening)
# =============================================================================

class TestPropertySettersHardened:
    """Verify property setters raise AttributeError in Phase 4b."""

    def test_aliases_setter_raises(self, simple_adf):
        """Verify aliases setter raises AttributeError."""
        with pytest.raises(AttributeError, match="no longer supported"):
            simple_adf.aliases = {'new': 'x + 1'}

    def test_alias_dtypes_setter_raises(self, simple_adf):
        """Verify alias_dtypes setter raises AttributeError."""
        with pytest.raises(AttributeError, match="no longer supported"):
            simple_adf.alias_dtypes = {'sum_xy': np.int32}

    def test_constant_aliases_setter_raises(self, simple_adf):
        """Verify constant_aliases setter raises AttributeError."""
        with pytest.raises(AttributeError, match="no longer supported"):
            simple_adf.constant_aliases = {'new_const'}

    def test_compression_info_setter_raises(self, simple_adf):
        """Verify compression_info setter raises AttributeError."""
        with pytest.raises(AttributeError, match="no longer supported"):
            simple_adf.compression_info = {'col': {'state': 'test'}}


# =============================================================================
# Test: Parquet Schema Roundtrip
# =============================================================================

class TestParquetSchemaRoundtrip:
    """Tests for Parquet schema serialization roundtrip."""

    def test_schema_roundtrip_basic(self, simple_adf, temp_dir):
        """Verify schema survives Parquet roundtrip."""
        path = os.path.join(temp_dir, "test")
        simple_adf.save(path)
        
        loaded = AliasDataFrame.load(path)
        
        # Check aliases
        assert loaded.aliases == simple_adf.aliases
        
        # Check schema structure
        assert 'columns' in loaded._schema
        assert 'compression' in loaded._schema
        assert 'subframes' in loaded._schema

    def test_alias_definitions_preserved(self, simple_adf, temp_dir):
        """Verify alias definitions preserved after Parquet roundtrip."""
        path = os.path.join(temp_dir, "test")
        simple_adf.save(path)
        loaded = AliasDataFrame.load(path)
        
        assert loaded.aliases['sum_xy'] == 'x + y'
        assert loaded.aliases['ratio'] == 'y / x'
        assert loaded.aliases['const_pi'] == '3.14159'

    def test_alias_dtypes_preserved(self, simple_adf, temp_dir):
        """Verify alias dtypes preserved after Parquet roundtrip."""
        path = os.path.join(temp_dir, "test")
        simple_adf.save(path)
        loaded = AliasDataFrame.load(path)
        
        assert loaded.alias_dtypes.get('sum_xy') == np.float32
        assert loaded.alias_dtypes.get('ratio') == np.float64

    def test_constant_aliases_preserved(self, simple_adf, temp_dir):
        """Verify constant aliases preserved after Parquet roundtrip."""
        path = os.path.join(temp_dir, "test")
        simple_adf.save(path)
        loaded = AliasDataFrame.load(path)
        
        assert 'const_pi' in loaded.constant_aliases

    def test_column_dtypes_preserved(self, simple_adf, temp_dir):
        """Verify column dtypes (float16, int8, int16) preserved."""
        path = os.path.join(temp_dir, "test")
        simple_adf.save(path, dropAliasColumns=False)
        loaded = AliasDataFrame.load(path)
        
        # Check original dtypes are preserved
        assert loaded.df['x'].dtype == np.float32
        assert loaded.df['y'].dtype == np.float16
        assert loaded.df['z'].dtype == np.int16
        assert loaded.df['key'].dtype == np.int8

    def test_subframe_metadata_preserved(self, adf_with_subframe, temp_dir):
        """Verify subframe metadata preserved after Parquet roundtrip."""
        path = os.path.join(temp_dir, "test")
        adf_with_subframe.save(path)
        loaded = AliasDataFrame.load(path)
        
        # Check subframe metadata in schema
        assert 'T' in loaded._schema['subframes']
        assert loaded._schema['subframes']['T']['index'] == 'key'

    def test_subframe_data_preserved(self, adf_with_subframe, temp_dir):
        """Verify subframe data preserved after Parquet roundtrip."""
        path = os.path.join(temp_dir, "test")
        adf_with_subframe.save(path)
        loaded = AliasDataFrame.load(path)
        
        # Check subframe is actually loaded
        sub = loaded.get_subframe('T')
        assert sub is not None
        assert len(sub.df) == 5
        assert 'offset' in sub.df.columns

    def test_invariance_materialize_before_after(self, simple_adf, temp_dir):
        """Verify materialized alias values match after roundtrip."""
        # Materialize before export
        simple_adf.materialize_alias('sum_xy')
        expected_values = simple_adf.df['sum_xy'].values.copy()
        
        path = os.path.join(temp_dir, "test")
        simple_adf.save(path, dropAliasColumns=False)
        loaded = AliasDataFrame.load(path)
        
        # Materialize after load
        loaded.materialize_alias('sum_xy')
        actual_values = loaded.df['sum_xy'].values
        
        np.testing.assert_array_almost_equal(expected_values, actual_values)

    def test_corrupted_json_raises_clear_error(self, temp_dir):
        """Verify corrupted JSON raises clear error."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Create a parquet file with corrupted schema metadata
        df = pd.DataFrame({'x': [1, 2, 3]})
        table = pa.Table.from_pandas(df)
        
        corrupted_meta = {
            SCHEMA_METADATA_KEY.encode(): b'{"invalid json'
        }
        table = table.replace_schema_metadata(corrupted_meta)
        
        path = os.path.join(temp_dir, "corrupted.parquet")
        pq.write_table(table, path)
        
        with pytest.raises(ValueError, match="Corrupted schema metadata"):
            AliasDataFrame.load(os.path.join(temp_dir, "corrupted"))

    def test_legacy_format_backward_compat(self, temp_dir):
        """Verify loading files with legacy metadata format still works."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Create a parquet file with legacy metadata format
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        table = pa.Table.from_pandas(df)
        
        legacy_meta = {
            b'aliases': b'{"test_alias": "x * 2"}',
            b'dtypes': b'{"test_alias": "float32"}',
            b'constants': b'[]',
            b'compression_info': b'{"__meta__": {"schema_version": 1}}'
        }
        table = table.replace_schema_metadata(legacy_meta)
        
        path = os.path.join(temp_dir, "legacy.parquet")
        pq.write_table(table, path)
        
        loaded = AliasDataFrame.load(os.path.join(temp_dir, "legacy"))
        
        assert 'test_alias' in loaded.aliases
        assert loaded.aliases['test_alias'] == 'x * 2'


# =============================================================================
# Test: ROOT Schema Roundtrip (requires ROOT)
# =============================================================================

@pytest.fixture
def has_root():
    """Check if ROOT is available."""
    try:
        import ROOT
        return True
    except ImportError:
        return False


class TestROOTSchemaRoundtrip:
    """Tests for ROOT schema serialization roundtrip."""

    @pytest.fixture(autouse=True)
    def skip_without_root(self, has_root):
        if not has_root:
            pytest.skip("ROOT not available")

    def test_schema_roundtrip_root(self, simple_adf, temp_dir):
        """Verify schema survives ROOT roundtrip."""
        path = os.path.join(temp_dir, "test.root")
        simple_adf.export_tree(path, "tree")
        
        loaded = AliasDataFrame.read_tree(path, "tree")
        
        # Check aliases
        assert loaded.aliases == simple_adf.aliases

    def test_alias_definitions_preserved_root(self, simple_adf, temp_dir):
        """Verify alias definitions preserved after ROOT roundtrip."""
        path = os.path.join(temp_dir, "test.root")
        simple_adf.export_tree(path, "tree")
        loaded = AliasDataFrame.read_tree(path, "tree")
        
        assert loaded.aliases['sum_xy'] == 'x + y'
        assert loaded.aliases['ratio'] == 'y / x'

    def test_alias_dtypes_preserved_root(self, simple_adf, temp_dir):
        """Verify alias dtypes preserved after ROOT roundtrip."""
        path = os.path.join(temp_dir, "test.root")
        simple_adf.export_tree(path, "tree")
        loaded = AliasDataFrame.read_tree(path, "tree")
        
        assert loaded.alias_dtypes.get('sum_xy') == np.float32
        assert loaded.alias_dtypes.get('ratio') == np.float64

    def test_constant_aliases_preserved_root(self, simple_adf, temp_dir):
        """Verify constant aliases preserved after ROOT roundtrip."""
        path = os.path.join(temp_dir, "test.root")
        simple_adf.export_tree(path, "tree")
        loaded = AliasDataFrame.read_tree(path, "tree")
        
        assert 'const_pi' in loaded.constant_aliases

    def test_subframe_preserved_root(self, adf_with_subframe, temp_dir):
        """Verify subframes preserved after ROOT roundtrip."""
        path = os.path.join(temp_dir, "test.root")
        adf_with_subframe.export_tree(path, "tree")
        loaded = AliasDataFrame.read_tree(path, "tree")
        
        # Check subframe exists
        sub = loaded.get_subframe('T')
        assert sub is not None
        assert len(sub.df) == 5

    def test_entry_slice_roundtrip_root(self, simple_adf, temp_dir):
        """Verify entry slicing works with ROOT roundtrip."""
        path = os.path.join(temp_dir, "test.root")
        simple_adf.export_tree(path, "tree")
        
        # Read only first 3 entries
        loaded = AliasDataFrame.read_tree(path, "tree", entry_stop=3)
        
        assert len(loaded.df) == 3
        # Schema should still be complete
        assert loaded.aliases == simple_adf.aliases


# =============================================================================
# Test: Recursive Subframes
# =============================================================================

class TestRecursiveSubframes:
    """Tests for recursive subframes (T → T2)."""

    def test_recursive_subframe_parquet(self, simple_df, temp_dir):
        """Verify recursive subframes work with Parquet."""
        # Create nested structure: main → T → T2
        main_adf = AliasDataFrame(simple_df.copy())
        
        sub_df = pd.DataFrame({
            'key': np.array([0, 1, 2, 3, 4], dtype=np.int8),
            'sub_key': np.array([0, 0, 1, 1, 2], dtype=np.int8),
            'val1': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32),
        })
        sub_adf = AliasDataFrame(sub_df)
        
        sub_sub_df = pd.DataFrame({
            'sub_key': np.array([0, 1, 2], dtype=np.int8),
            'val2': np.array([10.0, 20.0, 30.0], dtype=np.float32),
        })
        sub_sub_adf = AliasDataFrame(sub_sub_df)
        sub_sub_adf.add_alias('val2_doubled', 'val2 * 2', dtype=np.float32)
        
        # Register nested subframes
        sub_adf.register_subframe('T2', sub_sub_adf, 'sub_key')
        main_adf.register_subframe('T', sub_adf, 'key')
        
        # Save and load
        path = os.path.join(temp_dir, "recursive")
        main_adf.save(path)
        loaded = AliasDataFrame.load(path)
        
        # Verify structure
        assert loaded.get_subframe('T') is not None
        t_sub = loaded.get_subframe('T')
        assert t_sub.get_subframe('T2') is not None
        
        # Verify T2 alias preserved
        t2_sub = t_sub.get_subframe('T2')
        assert 'val2_doubled' in t2_sub.aliases


# =============================================================================
# Test: Compression Metadata Preservation
# =============================================================================

class TestCompressionMetadataPreservation:
    """Tests for compression metadata preservation through roundtrip."""

    def test_compression_info_preserved_parquet(self, simple_df, temp_dir):
        """Verify compression_info preserved after Parquet roundtrip."""
        adf = AliasDataFrame(simple_df.copy())
        
        # Add compression info manually
        adf.compression_info['x'] = {
            'compressed_col': 'x_c',
            'compress_expr': 'round(x * 100)',
            'decompress_expr': 'x_c / 100.0',
            'compressed_dtype': 'int16',
            'decompressed_dtype': 'float32',
            'state': 'compressed'
        }
        
        path = os.path.join(temp_dir, "test")
        adf.save(path)
        loaded = AliasDataFrame.load(path)
        
        assert 'x' in loaded.compression_info
        assert loaded.compression_info['x']['state'] == 'compressed'
        assert loaded.compression_info['x']['compressed_col'] == 'x_c'


# =============================================================================
# Test: float16 Explicit Handling
# =============================================================================

class TestFloat16Handling:
    """Tests for float16 handling in Parquet (auto-cast to float32, restore on load)."""

    def test_float16_parquet_roundtrip(self, temp_dir):
        """Verify float16 columns survive Parquet roundtrip via auto-casting."""
        # Create DataFrame with float16 column
        df = pd.DataFrame({
            'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float16),
            'y': np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        })
        
        adf = AliasDataFrame(df)
        assert adf.df['x'].dtype == np.float16
        
        path = os.path.join(temp_dir, "float16_test")
        adf.save(path)
        loaded = AliasDataFrame.load(path)
        
        # float16 should be restored
        assert loaded.df['x'].dtype == np.float16
        # Values should match within float16 precision
        np.testing.assert_array_almost_equal(
            adf.df['x'].values, 
            loaded.df['x'].values,
            decimal=3
        )

    def test_float16_int8_int16_all_preserved(self, temp_dir):
        """Verify all narrow dtypes are preserved after Parquet roundtrip."""
        df = pd.DataFrame({
            'f16': np.array([1.0, 2.0, 3.0], dtype=np.float16),
            'f32': np.array([1.0, 2.0, 3.0], dtype=np.float32),
            'i8': np.array([1, 2, 3], dtype=np.int8),
            'i16': np.array([1, 2, 3], dtype=np.int16),
            'i32': np.array([1, 2, 3], dtype=np.int32),
        })
        
        adf = AliasDataFrame(df)
        
        path = os.path.join(temp_dir, "dtypes_test")
        adf.save(path)
        loaded = AliasDataFrame.load(path)
        
        assert loaded.df['f16'].dtype == np.float16
        assert loaded.df['f32'].dtype == np.float32
        assert loaded.df['i8'].dtype == np.int8
        assert loaded.df['i16'].dtype == np.int16
        assert loaded.df['i32'].dtype == np.int32

    def test_float16_with_aliases(self, temp_dir):
        """Verify float16 works with aliases after roundtrip."""
        df = pd.DataFrame({
            'x': np.array([1.0, 2.0, 3.0], dtype=np.float16),
        })
        
        adf = AliasDataFrame(df)
        adf.add_alias('x_doubled', 'x * 2', dtype=np.float32)
        
        # Materialize and check
        original_result = adf.get_alias_series('x_doubled').values.copy()
        
        path = os.path.join(temp_dir, "float16_alias_test")
        adf.save(path)
        loaded = AliasDataFrame.load(path)
        
        # Alias should work with restored float16
        loaded_result = loaded.get_alias_series('x_doubled').values
        
        np.testing.assert_array_almost_equal(original_result, loaded_result, decimal=3)


# =============================================================================
# Test: Restore Methods
# =============================================================================

class TestRestoreMethods:
    """Tests for internal restore methods."""

    def test_restore_aliases_from_dict(self, simple_df):
        """Verify _restore_aliases_from_dict works."""
        adf = AliasDataFrame(simple_df)
        adf._restore_aliases_from_dict({
            'test1': 'x + 1',
            'test2': 'y * 2'
        })
        
        assert 'test1' in adf.aliases
        assert adf.aliases['test1'] == 'x + 1'

    def test_restore_alias_dtypes_from_dict(self, simple_df):
        """Verify _restore_alias_dtypes_from_dict works."""
        adf = AliasDataFrame(simple_df)
        adf._restore_aliases_from_dict({'test': 'x + 1'})
        adf._restore_alias_dtypes_from_dict({'test': np.float32})
        
        assert adf.alias_dtypes.get('test') == np.float32

    def test_restore_constant_aliases(self, simple_df):
        """Verify _restore_constant_aliases works."""
        adf = AliasDataFrame(simple_df)
        adf._restore_aliases_from_dict({'const': '42'})
        adf._restore_constant_aliases(['const'])
        
        assert 'const' in adf.constant_aliases

    def test_restore_schema_full(self, simple_df):
        """Verify _restore_schema restores complete schema."""
        adf = AliasDataFrame(simple_df)
        
        schema = {
            'columns': {
                'test': {'expr': 'x + y', 'dtype': np.float32, 'constant': False}
            },
            'compression': {
                '__meta__': {'schema_version': 1, 'state_machine': 'CompressionState.v1'}
            },
            'subframes': {
                'T': {'index': 'key'}
            }
        }
        
        adf._restore_schema(schema)
        
        assert 'test' in adf.aliases
        assert 'T' in adf._schema['subframes']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
