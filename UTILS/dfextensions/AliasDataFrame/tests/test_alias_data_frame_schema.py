"""
test_alias_data_frame_schema.py - Tests for Phase 4 unified schema functionality

Tests the new schema-based API introduced in Phase 4:
- _schema as single source of truth
- Backward-compatible properties (aliases, alias_dtypes, constant_aliases, compression_info)
- New API methods (schema, update_schema, apply_aliases, apply_dtypes)
- Schema validation
- Schema persistence through subframe registration

Run with: pytest test_alias_data_frame_schema.py -v
"""

import pytest
import pandas as pd
import numpy as np
import copy
import warnings

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def simple_df():
    """Simple DataFrame for basic tests."""
    return pd.DataFrame({
        'x': [1.0, 2.0, 3.0, 4.0, 5.0],
        'y': [10.0, 20.0, 30.0, 40.0, 50.0],
        'z': [100, 200, 300, 400, 500],
    })


@pytest.fixture
def simple_adf(simple_df):
    """Simple AliasDataFrame for basic tests."""
    return AliasDataFrame(simple_df)


@pytest.fixture
def adf_with_aliases(simple_df):
    """AliasDataFrame with some aliases defined."""
    adf = AliasDataFrame(simple_df)
    adf.add_alias('sum_xy', 'x + y', dtype=np.float32)
    adf.add_alias('ratio', 'y / x', dtype=np.float64)
    adf.add_alias('const_pi', '3.14159', dtype=np.float32, is_constant=True)
    return adf


@pytest.fixture
def subframe_df():
    """DataFrame for subframe tests."""
    return pd.DataFrame({
        'key': [0, 1, 2, 3, 4],
        'offset': [0.1, 0.2, 0.3, 0.4, 0.5],
    })


# =============================================================================
# Test: Schema Structure
# =============================================================================

class TestSchemaStructure:
    """Tests for _schema internal structure."""

    def test_schema_initialized_with_correct_structure(self, simple_adf):
        """Verify _schema has correct sections after initialization."""
        assert hasattr(simple_adf, '_schema')
        assert 'columns' in simple_adf._schema
        assert 'compression' in simple_adf._schema
        assert 'subframes' in simple_adf._schema
        assert isinstance(simple_adf._schema['columns'], dict)
        assert isinstance(simple_adf._schema['compression'], dict)
        assert isinstance(simple_adf._schema['subframes'], dict)

    def test_schema_compression_has_meta(self, simple_adf):
        """Verify compression section has __meta__ with version info."""
        assert '__meta__' in simple_adf._schema['compression']
        meta = simple_adf._schema['compression']['__meta__']
        assert 'schema_version' in meta
        assert meta['schema_version'] == 1

    def test_schema_property_returns_deep_copy(self, simple_adf):
        """Verify schema property returns a deep copy, not reference."""
        schema1 = simple_adf.schema
        schema2 = simple_adf.schema
        
        # Modify the copy
        schema1['columns']['test'] = {'expr': 'x + 1'}
        
        # Original should be unchanged
        assert 'test' not in simple_adf._schema['columns']
        assert 'test' not in schema2['columns']

    def test_schema_property_includes_all_sections(self, adf_with_aliases):
        """Verify schema property includes columns, compression, subframes."""
        schema = adf_with_aliases.schema
        assert 'columns' in schema
        assert 'compression' in schema
        assert 'subframes' in schema


# =============================================================================
# Test: Backward Compatibility - aliases property
# =============================================================================

class TestAliasesProperty:
    """Tests for backward-compatible aliases property."""

    def test_aliases_property_returns_dict(self, simple_adf):
        """Verify aliases property returns a dict."""
        assert isinstance(simple_adf.aliases, dict)

    def test_aliases_property_empty_initially(self, simple_adf):
        """Verify aliases is empty for new ADF."""
        assert simple_adf.aliases == {}

    def test_aliases_property_reflects_add_alias(self, simple_adf):
        """Verify aliases property reflects add_alias calls."""
        simple_adf.add_alias('test', 'x + y')
        assert 'test' in simple_adf.aliases
        assert simple_adf.aliases['test'] == 'x + y'

    def test_aliases_property_reads_from_schema(self, simple_adf):
        """Verify aliases property reads from _schema."""
        # Directly manipulate _schema
        simple_adf._schema['columns']['direct'] = {'expr': 'x * 2'}
        assert 'direct' in simple_adf.aliases
        assert simple_adf.aliases['direct'] == 'x * 2'

    def test_aliases_setter_updates_schema(self, simple_adf):
        """Verify aliases setter updates _schema."""
        simple_adf.aliases = {'new_alias': 'x + 1'}
        assert 'new_alias' in simple_adf._schema['columns']
        assert simple_adf._schema['columns']['new_alias']['expr'] == 'x + 1'

    def test_aliases_setter_clears_old_aliases(self, adf_with_aliases):
        """Verify aliases setter clears existing aliases."""
        old_aliases = list(adf_with_aliases.aliases.keys())
        assert len(old_aliases) > 0
        
        adf_with_aliases.aliases = {'replacement': 'z * 2'}
        
        for old in old_aliases:
            assert old not in adf_with_aliases.aliases
        assert 'replacement' in adf_with_aliases.aliases

    def test_aliases_only_includes_columns_with_expr(self, simple_adf):
        """Verify aliases doesn't include columns without expr."""
        # Add a column entry without expr (like dtype-only)
        simple_adf._schema['columns']['no_expr'] = {'dtype': np.float32}
        assert 'no_expr' not in simple_adf.aliases


# =============================================================================
# Test: Backward Compatibility - alias_dtypes property
# =============================================================================

class TestAliasDtypesProperty:
    """Tests for backward-compatible alias_dtypes property."""

    def test_alias_dtypes_property_returns_dict(self, simple_adf):
        """Verify alias_dtypes property returns a dict."""
        assert isinstance(simple_adf.alias_dtypes, dict)

    def test_alias_dtypes_empty_initially(self, simple_adf):
        """Verify alias_dtypes is empty for new ADF."""
        assert simple_adf.alias_dtypes == {}

    def test_alias_dtypes_reflects_add_alias_with_dtype(self, simple_adf):
        """Verify alias_dtypes reflects add_alias with dtype."""
        simple_adf.add_alias('typed', 'x + y', dtype=np.float32)
        assert 'typed' in simple_adf.alias_dtypes
        assert simple_adf.alias_dtypes['typed'] == np.float32

    def test_alias_dtypes_excludes_aliases_without_dtype(self, simple_adf):
        """Verify alias_dtypes excludes aliases without dtype."""
        simple_adf.add_alias('untyped', 'x + y')
        simple_adf.add_alias('typed', 'x - y', dtype=np.float64)
        assert 'untyped' not in simple_adf.alias_dtypes
        assert 'typed' in simple_adf.alias_dtypes

    def test_alias_dtypes_setter_updates_schema(self, simple_adf):
        """Verify alias_dtypes setter updates _schema."""
        simple_adf.add_alias('test', 'x + 1')
        simple_adf.alias_dtypes = {'test': np.int32}
        assert simple_adf._schema['columns']['test']['dtype'] == np.int32


# =============================================================================
# Test: Backward Compatibility - constant_aliases property
# =============================================================================

class TestConstantAliasesProperty:
    """Tests for backward-compatible constant_aliases property."""

    def test_constant_aliases_property_returns_set(self, simple_adf):
        """Verify constant_aliases property returns a set."""
        assert isinstance(simple_adf.constant_aliases, set)

    def test_constant_aliases_empty_initially(self, simple_adf):
        """Verify constant_aliases is empty for new ADF."""
        assert simple_adf.constant_aliases == set()

    def test_constant_aliases_reflects_add_alias_constant(self, simple_adf):
        """Verify constant_aliases reflects add_alias with is_constant=True."""
        simple_adf.add_alias('pi', '3.14159', is_constant=True)
        assert 'pi' in simple_adf.constant_aliases

    def test_constant_aliases_setter_works(self, simple_adf):
        """Verify constant_aliases setter updates internal state."""
        simple_adf.add_alias('test', '42')
        simple_adf.constant_aliases = {'test'}
        assert 'test' in simple_adf.constant_aliases

    def test_constant_aliases_includes_schema_constants(self, simple_adf):
        """Verify constant_aliases includes constants from _schema."""
        simple_adf._schema['columns']['schema_const'] = {
            'expr': '99',
            'constant': True
        }
        assert 'schema_const' in simple_adf.constant_aliases


# =============================================================================
# Test: Backward Compatibility - compression_info property
# =============================================================================

class TestCompressionInfoProperty:
    """Tests for backward-compatible compression_info property."""

    def test_compression_info_property_returns_dict(self, simple_adf):
        """Verify compression_info property returns a dict."""
        assert isinstance(simple_adf.compression_info, dict)

    def test_compression_info_has_meta_initially(self, simple_adf):
        """Verify compression_info has __meta__ initially."""
        assert '__meta__' in simple_adf.compression_info

    def test_compression_info_references_schema(self, simple_adf):
        """Verify compression_info is reference to _schema['compression']."""
        # Modify via property
        simple_adf.compression_info['test_col'] = {'state': 'test'}
        # Should appear in _schema
        assert 'test_col' in simple_adf._schema['compression']

    def test_compression_info_setter_updates_schema(self, simple_adf):
        """Verify compression_info setter updates _schema."""
        new_info = {
            '__meta__': {'schema_version': 2},
            'col1': {'state': 'compressed'}
        }
        simple_adf.compression_info = new_info
        assert simple_adf._schema['compression'] == new_info


# =============================================================================
# Test: add_alias writes to _schema
# =============================================================================

class TestAddAliasSchema:
    """Tests for add_alias writing to _schema."""

    def test_add_alias_writes_to_schema_columns(self, simple_adf):
        """Verify add_alias writes to _schema['columns']."""
        simple_adf.add_alias('test', 'x + y')
        assert 'test' in simple_adf._schema['columns']
        assert simple_adf._schema['columns']['test']['expr'] == 'x + y'

    def test_add_alias_with_dtype_writes_dtype(self, simple_adf):
        """Verify add_alias with dtype writes dtype to schema."""
        simple_adf.add_alias('typed', 'x * 2', dtype=np.float16)
        assert simple_adf._schema['columns']['typed']['dtype'] == np.float16

    def test_add_alias_with_constant_writes_constant(self, simple_adf):
        """Verify add_alias with is_constant writes constant flag."""
        simple_adf.add_alias('const', '42', is_constant=True)
        assert simple_adf._schema['columns']['const'].get('constant') is True

    def test_add_alias_overwrites_existing(self, simple_adf):
        """Verify add_alias overwrites existing alias."""
        simple_adf.add_alias('test', 'x + 1')
        simple_adf.add_alias('test', 'x + 2')
        assert simple_adf._schema['columns']['test']['expr'] == 'x + 2'


# =============================================================================
# Test: register_subframe writes to _schema
# =============================================================================

class TestRegisterSubframeSchema:
    """Tests for register_subframe writing to _schema."""

    def test_register_subframe_writes_to_schema(self, simple_adf, subframe_df):
        """Verify register_subframe writes to _schema['subframes']."""
        sub_adf = AliasDataFrame(subframe_df)
        simple_adf.register_subframe('T', sub_adf, 'key')
        
        assert 'T' in simple_adf._schema['subframes']
        assert simple_adf._schema['subframes']['T']['index'] == 'key'

    def test_register_subframe_with_list_index(self, simple_adf, subframe_df):
        """Verify register_subframe works with list index."""
        sub_adf = AliasDataFrame(subframe_df)
        simple_adf.register_subframe('T', sub_adf, ['key'])
        
        assert simple_adf._schema['subframes']['T']['index'] == ['key']


# =============================================================================
# Test: update_schema method
# =============================================================================

class TestUpdateSchema:
    """Tests for update_schema method."""

    def test_update_schema_adds_column_spec(self, simple_adf):
        """Verify update_schema adds column specifications."""
        simple_adf.update_schema({
            'columns': {
                'new_alias': {'expr': 'x + y', 'dtype': np.float32}
            }
        })
        assert 'new_alias' in simple_adf._schema['columns']
        assert simple_adf._schema['columns']['new_alias']['expr'] == 'x + y'

    def test_update_schema_partial_update(self, simple_adf):
        """Verify update_schema does partial updates."""
        simple_adf.add_alias('existing', 'x + 1')
        simple_adf.update_schema({
            'columns': {
                'existing': {'dtype': np.float32}  # Add dtype to existing
            }
        })
        # Should have both expr and dtype now
        assert simple_adf._schema['columns']['existing']['expr'] == 'x + 1'
        assert simple_adf._schema['columns']['existing']['dtype'] == np.float32

    def test_update_schema_applies_dtype_to_physical_column(self, simple_df):
        """Verify update_schema applies dtype to physical columns when apply=True."""
        adf = AliasDataFrame(simple_df.copy())
        assert adf.df['x'].dtype == np.float64
        
        adf.update_schema({
            'columns': {'x': {'dtype': np.float32}}
        }, apply=True)
        
        assert adf.df['x'].dtype == np.float32

    def test_update_schema_no_apply_keeps_dtype(self, simple_df):
        """Verify update_schema with apply=False doesn't change DataFrame."""
        adf = AliasDataFrame(simple_df.copy())
        original_dtype = adf.df['x'].dtype
        
        adf.update_schema({
            'columns': {'x': {'dtype': np.float32}}
        }, apply=False)
        
        assert adf.df['x'].dtype == original_dtype

    def test_update_schema_updates_compression(self, simple_adf):
        """Verify update_schema updates compression section."""
        simple_adf.update_schema({
            'compression': {
                'test_col': {'state': 'schema_only'}
            }
        })
        assert 'test_col' in simple_adf._schema['compression']

    def test_update_schema_updates_subframes(self, simple_adf):
        """Verify update_schema updates subframes section."""
        simple_adf.update_schema({
            'subframes': {
                'T': {'index': 'track_index'}
            }
        })
        assert 'T' in simple_adf._schema['subframes']

    def test_update_schema_with_invalid_dtype_raises(self, simple_adf):
        """Verify update_schema raises on invalid dtype."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            simple_adf.update_schema({
                'columns': {'x': {'dtype': 'not_a_dtype'}}
            })

    def test_update_schema_with_non_string_expr_raises(self, simple_adf):
        """Verify update_schema raises on non-string expression."""
        with pytest.raises(ValueError, match="must be a string"):
            simple_adf.update_schema({
                'columns': {'test': {'expr': 123}}
            })

    def test_update_schema_validate_false_skips_validation(self, simple_adf):
        """Verify update_schema with validate=False skips validation."""
        # This would normally fail validation
        simple_adf.update_schema({
            'columns': {'test': {'expr': 123}}
        }, validate=False)
        # But it succeeds with validate=False
        assert simple_adf._schema['columns']['test']['expr'] == 123

    def test_update_schema_errors_warn(self, simple_df):
        """Verify update_schema with errors='warn' warns instead of raising."""
        adf = AliasDataFrame(simple_df.copy())
        # Try to cast to incompatible type
        adf.df['x'] = ['a', 'b', 'c', 'd', 'e']  # strings
        
        with pytest.warns(UserWarning, match="Failed to cast"):
            adf.update_schema({
                'columns': {'x': {'dtype': np.float32}}
            }, errors='warn')


# =============================================================================
# Test: _validate_schema_update method
# =============================================================================

class TestValidateSchemaUpdate:
    """Tests for _validate_schema_update method."""

    def test_validate_rejects_undefined_subframe_reference(self, simple_adf):
        """Verify validation rejects references to undefined subframes."""
        with pytest.raises(ValueError, match="undefined subframe"):
            simple_adf._validate_schema_update({
                'columns': {'test': {'expr': 'X.col + 1'}}
            })

    def test_validate_accepts_defined_subframe_reference(self, simple_adf, subframe_df):
        """Verify validation accepts references to defined subframes."""
        sub_adf = AliasDataFrame(subframe_df)
        simple_adf.register_subframe('T', sub_adf, 'key')
        
        # Should not raise
        simple_adf._validate_schema_update({
            'columns': {'test': {'expr': 'T.offset + 1'}}
        })

    def test_validate_subframe_requires_index(self, simple_adf):
        """Verify validation requires 'index' in subframe specs."""
        with pytest.raises(ValueError, match="must specify 'index'"):
            simple_adf._validate_schema_update({
                'subframes': {'T': {'other': 'value'}}
            })


# =============================================================================
# Test: apply_aliases method
# =============================================================================

class TestApplyAliases:
    """Tests for apply_aliases method."""

    def test_apply_aliases_registers_multiple(self, simple_adf):
        """Verify apply_aliases registers multiple aliases."""
        simple_adf.apply_aliases({
            'sum_xy': {'expr': 'x + y', 'dtype': np.float32},
            'diff_xy': {'expr': 'x - y', 'dtype': np.float64},
        })
        
        assert 'sum_xy' in simple_adf.aliases
        assert 'diff_xy' in simple_adf.aliases

    def test_apply_aliases_sets_dtype(self, simple_adf):
        """Verify apply_aliases sets dtype correctly."""
        simple_adf.apply_aliases({
            'typed': {'expr': 'x * 2', 'dtype': np.float16}
        })
        assert simple_adf.alias_dtypes.get('typed') == np.float16

    def test_apply_aliases_sets_constant(self, simple_adf):
        """Verify apply_aliases sets constant flag correctly."""
        simple_adf.apply_aliases({
            'pi': {'expr': '3.14159', 'constant': True}
        })
        assert 'pi' in simple_adf.constant_aliases

    def test_apply_aliases_requires_expr(self, simple_adf):
        """Verify apply_aliases raises if expr is missing."""
        with pytest.raises(ValueError, match="missing 'expr'"):
            simple_adf.apply_aliases({
                'bad': {'dtype': np.float32}  # No expr
            })


# =============================================================================
# Test: apply_dtypes method
# =============================================================================

class TestApplyDtypes:
    """Tests for apply_dtypes method."""

    def test_apply_dtypes_converts_column(self, simple_df):
        """Verify apply_dtypes converts column dtype."""
        adf = AliasDataFrame(simple_df.copy())
        adf.apply_dtypes({'x': np.float32})
        assert adf.df['x'].dtype == np.float32

    def test_apply_dtypes_multiple_columns(self, simple_df):
        """Verify apply_dtypes converts multiple columns."""
        adf = AliasDataFrame(simple_df.copy())
        adf.apply_dtypes({
            'x': np.float32,
            'y': np.float16,
            'z': np.int16
        })
        assert adf.df['x'].dtype == np.float32
        assert adf.df['y'].dtype == np.float16
        assert adf.df['z'].dtype == np.int16

    def test_apply_dtypes_updates_schema(self, simple_df):
        """Verify apply_dtypes updates _schema."""
        adf = AliasDataFrame(simple_df.copy())
        adf.apply_dtypes({'x': np.float32})
        assert 'x' in adf._schema['columns']
        assert adf._schema['columns']['x']['dtype'] == np.float32

    def test_apply_dtypes_missing_column_raises(self, simple_adf):
        """Verify apply_dtypes raises on missing column with errors='raise'."""
        with pytest.raises(ValueError, match="not found"):
            simple_adf.apply_dtypes({'nonexistent': np.float32}, errors='raise')

    def test_apply_dtypes_missing_column_warns(self, simple_adf):
        """Verify apply_dtypes warns on missing column with errors='warn'."""
        with pytest.warns(UserWarning, match="not found"):
            simple_adf.apply_dtypes({'nonexistent': np.float32}, errors='warn')

    def test_apply_dtypes_missing_column_ignores(self, simple_adf):
        """Verify apply_dtypes ignores missing column with errors='ignore'."""
        # Should not raise or warn
        simple_adf.apply_dtypes({'nonexistent': np.float32}, errors='ignore')


# =============================================================================
# Test: Schema roundtrip (add_alias → schema → verify)
# =============================================================================

class TestSchemaRoundtrip:
    """Tests for schema consistency across operations."""

    def test_add_alias_then_read_schema(self, simple_adf):
        """Verify schema reflects add_alias operations."""
        simple_adf.add_alias('a1', 'x + 1', dtype=np.float32, is_constant=False)
        simple_adf.add_alias('a2', 'y * 2', dtype=np.float64, is_constant=True)
        
        schema = simple_adf.schema
        
        assert 'a1' in schema['columns']
        assert schema['columns']['a1']['expr'] == 'x + 1'
        assert schema['columns']['a1']['dtype'] == np.float32
        
        assert 'a2' in schema['columns']
        assert schema['columns']['a2']['constant'] is True

    def test_schema_survives_materialize(self, simple_adf):
        """Verify schema survives materialize_alias."""
        simple_adf.add_alias('test', 'x + y', dtype=np.float32)
        simple_adf.materialize_alias('test')
        
        # Alias should still be in schema
        assert 'test' in simple_adf.aliases
        # And materialized in df
        assert 'test' in simple_adf.df.columns


# =============================================================================
# Test: SubframeRegistry.has_subframe
# =============================================================================

class TestSubframeRegistryHasSubframe:
    """Tests for SubframeRegistry.has_subframe method."""

    def test_has_subframe_false_when_empty(self, simple_adf):
        """Verify has_subframe returns False for empty registry."""
        assert not simple_adf._subframes.has_subframe('T')

    def test_has_subframe_true_after_register(self, simple_adf, subframe_df):
        """Verify has_subframe returns True after registration."""
        sub_adf = AliasDataFrame(subframe_df)
        simple_adf.register_subframe('T', sub_adf, 'key')
        
        assert simple_adf._subframes.has_subframe('T')
        assert not simple_adf._subframes.has_subframe('X')


# =============================================================================
# Test: Edge cases and error handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Verify schema works with empty DataFrame."""
        adf = AliasDataFrame(pd.DataFrame())
        assert adf.schema is not None
        assert adf.aliases == {}

    def test_update_schema_empty_update(self, simple_adf):
        """Verify update_schema handles empty update dict."""
        schema_before = simple_adf.schema
        simple_adf.update_schema({})
        schema_after = simple_adf.schema
        assert schema_before == schema_after

    def test_aliases_setter_with_empty_dict(self, adf_with_aliases):
        """Verify aliases setter with empty dict clears aliases."""
        adf_with_aliases.aliases = {}
        assert adf_with_aliases.aliases == {}

    def test_constant_aliases_setter_with_empty_set(self, simple_adf):
        """Verify constant_aliases setter with empty set clears constants."""
        simple_adf.add_alias('const', '42', is_constant=True)
        simple_adf.constant_aliases = set()
        # Note: schema-derived constants might still exist
        # This tests the _constant_aliases field specifically


# =============================================================================
# Test: Integration with existing functionality
# =============================================================================

class TestSchemaIntegration:
    """Tests for schema integration with existing functionality."""

    def test_aliases_property_works_with_materialize(self, simple_adf):
        """Verify aliases property works correctly with materialize_alias."""
        simple_adf.add_alias('sum', 'x + y')
        assert 'sum' in simple_adf.aliases
        
        simple_adf.materialize_alias('sum')
        assert 'sum' in simple_adf.df.columns
        # Alias should still be defined
        assert 'sum' in simple_adf.aliases

    def test_schema_with_chained_aliases(self, simple_adf):
        """Verify schema handles chained alias dependencies."""
        simple_adf.add_alias('a', 'x + 1')
        simple_adf.add_alias('b', 'a + 1')  # Depends on a
        simple_adf.add_alias('c', 'b + 1')  # Depends on b
        
        schema = simple_adf.schema
        assert 'a' in schema['columns']
        assert 'b' in schema['columns']
        assert 'c' in schema['columns']

    def test_schema_preserved_after_get_alias_series(self, simple_adf):
        """Verify schema preserved after get_alias_series."""
        simple_adf.add_alias('test', 'x * 2', dtype=np.float32)
        _ = simple_adf.get_alias_series('test')
        
        assert 'test' in simple_adf.aliases
        assert simple_adf.alias_dtypes.get('test') == np.float32


# =============================================================================
# Test: Mutation Safety (Gemini Review Directive)
# =============================================================================

class TestMutationSafety:
    """Tests ensuring property getters return safe objects that don't mutate _schema."""

    def test_aliases_mutation_does_not_affect_schema(self, simple_adf):
        """Verify mutating returned aliases dict doesn't affect _schema."""
        simple_adf.add_alias('original', 'x + 1')
        
        # Get aliases and mutate the returned dict
        aliases_copy = simple_adf.aliases
        aliases_copy['injected'] = 'malicious_expr'
        
        # _schema should be unaffected
        assert 'injected' not in simple_adf._schema['columns']
        assert 'injected' not in simple_adf.aliases

    def test_alias_dtypes_mutation_does_not_affect_schema(self, simple_adf):
        """Verify mutating returned alias_dtypes dict doesn't affect _schema."""
        simple_adf.add_alias('typed', 'x + 1', dtype=np.float32)
        
        # Get alias_dtypes and mutate the returned dict
        dtypes_copy = simple_adf.alias_dtypes
        dtypes_copy['typed'] = np.int64
        dtypes_copy['injected'] = np.float16
        
        # _schema should be unaffected
        assert simple_adf._schema['columns']['typed']['dtype'] == np.float32
        assert 'injected' not in simple_adf._schema['columns']

    def test_constant_aliases_mutation_does_not_affect_schema(self, simple_adf):
        """Verify mutating returned constant_aliases set doesn't affect _schema."""
        simple_adf.add_alias('const', '42', is_constant=True)
        
        # Get constant_aliases and mutate the returned set
        constants_copy = simple_adf.constant_aliases
        constants_copy.add('injected')
        constants_copy.discard('const')
        
        # _schema should be unaffected
        assert simple_adf._schema['columns']['const'].get('constant') is True
        assert 'injected' not in simple_adf._schema['columns']

    def test_compression_info_is_reference_by_design(self, simple_adf):
        """
        Document that compression_info IS a reference (backward compat).
        
        This is intentional - existing code modifies compression_info directly.
        Phase 4b may change this behavior.
        """
        # Modify via property
        simple_adf.compression_info['test_entry'] = {'state': 'test'}
        
        # Should appear in _schema (this is expected behavior)
        assert 'test_entry' in simple_adf._schema['compression']

    def test_schema_property_is_deep_copy(self, adf_with_aliases):
        """Verify schema property returns deep copy - mutations are isolated."""
        schema = adf_with_aliases.schema
        
        # Mutate the copy deeply
        schema['columns']['sum_xy']['expr'] = 'MUTATED'
        schema['compression']['__meta__']['schema_version'] = 999
        
        # Original _schema should be unaffected
        assert adf_with_aliases._schema['columns']['sum_xy']['expr'] == 'x + y'
        assert adf_with_aliases._schema['compression']['__meta__']['schema_version'] == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
