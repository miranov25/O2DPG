"""
test_schema_definition_vs_record.py - Tests for Schema Definition vs Record Schema

Tests the separation of Definition Schema (blueprint) from Record Schema (snapshot):
- export_definition_schema() vs export_record_schema() output structure
- validate_schema() with check_data and strict modes
- Cycle detection when loading definition schema onto compressed data
- Deprecation warnings for old-format schemas
- Schema change protection for compressed columns

Run with: pytest test_schema_definition_vs_record.py -v
"""

import pytest
import pandas as pd
import numpy as np
import warnings
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame, CompressionState


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'dy': np.random.randn(100).astype('float32'),
        'dz': np.random.randn(100).astype('float32'),
        'x': np.random.randn(100).astype('float32'),
    })


@pytest.fixture
def compression_spec():
    """Standard compression specification."""
    return {
        'dy': {
            'compress': 'round(asinh(dy)*40)',
            'decompress': 'sinh(dy_c/40.)',
            'compressed_dtype': np.int16,
            'decompressed_dtype': np.float16
        },
        'dz': {
            'compress': 'round(asinh(dz)*40)',
            'decompress': 'sinh(dz_c/40.)',
            'compressed_dtype': np.int16,
            'decompressed_dtype': np.float16
        }
    }


@pytest.fixture
def adf_with_compression(sample_df, compression_spec):
    """AliasDataFrame with compression schema defined."""
    adf = AliasDataFrame(sample_df)
    adf.define_compression_schema(compression_spec)
    return adf


@pytest.fixture
def adf_compressed(sample_df, compression_spec):
    """AliasDataFrame with columns already compressed."""
    adf = AliasDataFrame(sample_df)
    adf.compress_columns(compression_spec, columns=['dy', 'dz'])
    return adf


# =============================================================================
# Test: Export Definition Schema vs Record Schema
# =============================================================================

class TestDefinitionVsRecordExport:
    """Tests for export_definition_schema() vs export_record_schema()."""

    def test_definition_schema_compression_target_is_physical(self, adf_compressed):
        """Definition schema exports compression targets as physical columns (no expr)."""
        schema = adf_compressed.export_definition_schema()
        
        # dy should be physical column (no expr)
        assert 'dy' in schema['columns']
        assert 'expr' not in schema['columns']['dy']
        assert schema['columns']['dy'].get('dtype') == 'float16'

    def test_definition_schema_no_storage_columns(self, adf_compressed):
        """Definition schema does NOT export compressed storage columns."""
        schema = adf_compressed.export_definition_schema()
        
        # dy_c should NOT be in definition schema
        assert 'dy_c' not in schema['columns']
        assert 'dz_c' not in schema['columns']

    def test_definition_schema_no_state_fields(self, adf_compressed):
        """Definition schema has no state or original_removed fields."""
        schema = adf_compressed.export_definition_schema()
        
        # compression section should not have state
        dy_comp = schema.get('compression', {}).get('dy', {})
        assert 'state' not in dy_comp
        assert 'original_removed' not in dy_comp

    def test_record_schema_compression_target_is_alias(self, adf_compressed):
        """Record schema exports compression targets as aliases (with expr)."""
        schema = adf_compressed.export_record_schema()
        
        # dy should be alias with decompress expr
        assert 'dy' in schema['columns']
        assert 'expr' in schema['columns']['dy']
        assert 'sinh' in schema['columns']['dy']['expr']

    def test_record_schema_has_storage_columns(self, adf_compressed):
        """Record schema exports compressed storage columns."""
        schema = adf_compressed.export_record_schema()
        
        # dy_c should be in record schema
        assert 'dy_c' in schema['columns']
        assert 'dz_c' in schema['columns']

    def test_record_schema_has_state_fields(self, adf_compressed):
        """Record schema includes state and original_removed fields."""
        schema = adf_compressed.export_record_schema()
        
        # compression section should have state
        dy_comp = schema.get('compression', {}).get('dy', {})
        assert 'state' in dy_comp
        assert dy_comp['state'] == 'compressed'

    def test_definition_schema_before_compression(self, adf_with_compression):
        """Definition schema same structure before and after compression."""
        schema_before = adf_with_compression.export_definition_schema()
        
        # Compress
        adf_with_compression.compress_columns(columns=['dy'])
        schema_after = adf_with_compression.export_definition_schema()
        
        # Definition schema structure should be same
        assert 'dy' in schema_before['columns']
        assert 'dy' in schema_after['columns']
        assert 'expr' not in schema_before['columns']['dy']
        assert 'expr' not in schema_after['columns']['dy']

    def test_convenience_methods_match_explicit_params(self, adf_compressed):
        """Convenience methods produce same output as explicit parameters."""
        def_schema = adf_compressed.export_definition_schema()
        explicit_def = adf_compressed.export_schema_v2(include_state=False)
        
        rec_schema = adf_compressed.export_record_schema()
        explicit_rec = adf_compressed.export_schema_v2(include_state=True)
        
        # Should be equivalent (excluding timestamp)
        assert def_schema['columns'] == explicit_def['columns']
        assert rec_schema['columns'] == explicit_rec['columns']


# =============================================================================
# Test: validate_schema() Modes
# =============================================================================

class TestValidateSchemaCheckData:
    """Tests for validate_schema(check_data=...) parameter."""

    def test_check_data_true_detects_pending_alias(self, sample_df):
        """check_data=True detects pending aliases."""
        adf = AliasDataFrame(sample_df)
        adf.add_alias('missing_dep', 'x + nonexistent_column')
        
        result = adf.validate_schema(check_data=True, verbose=False)
        
        assert len(result['pending']) > 0
        assert any('nonexistent_column' in p for p in result['pending'])

    def test_check_data_false_skips_data_validation(self, sample_df):
        """check_data=False skips data consistency checks."""
        adf = AliasDataFrame(sample_df)
        adf.add_alias('missing_dep', 'x + nonexistent_column')
        
        result = adf.validate_schema(check_data=False, verbose=False)
        
        # Should not report pending (data not checked)
        assert len(result['pending']) == 0
        assert result['valid'] is True

    def test_check_data_false_still_validates_schema_structure(self, sample_df):
        """check_data=False still validates schema structure."""
        adf = AliasDataFrame(sample_df)
        adf.define_compression_schema({
            'x': {
                'compress': 'round(x*100)',
                'decompress': 'x_c/100.',
                'compressed_dtype': 'int16',
                'decompressed_dtype': 'float32'
            }
        })
        
        result = adf.validate_schema(check_data=False, verbose=False)
        
        # Structure should be valid
        assert result['valid'] is True
        assert len(result['info']['compression_targets']) == 1


class TestValidateSchemaStrict:
    """Tests for validate_schema(strict=True) parameter."""

    def test_strict_rejects_pending_alias(self, sample_df):
        """strict=True treats pending aliases as errors."""
        adf = AliasDataFrame(sample_df)
        adf.add_alias('missing_dep', 'x + nonexistent_column')
        
        result = adf.validate_schema(strict=True, verbose=False)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0

    def test_strict_rejects_missing_columns(self, sample_df):
        """strict=True treats missing columns as errors."""
        adf = AliasDataFrame(sample_df)
        # Add a column to schema that doesn't exist in DataFrame
        adf._schema['columns']['nonexistent'] = {'dtype': 'float32'}
        
        result = adf.validate_schema(strict=True, verbose=False)
        
        assert result['valid'] is False
        assert len(result['errors']) > 0

    def test_permissive_allows_pending_alias(self, sample_df):
        """Default (permissive) allows pending aliases."""
        adf = AliasDataFrame(sample_df)
        adf.add_alias('missing_dep', 'x + nonexistent_column')
        
        result = adf.validate_schema(verbose=False)  # Default: permissive
        
        assert result['valid'] is True
        assert len(result['pending']) > 0

    def test_strict_overrides_explicit_params(self, sample_df):
        """strict=True overrides allow_pending_aliases even if explicitly set."""
        adf = AliasDataFrame(sample_df)
        adf.add_alias('missing_dep', 'x + nonexistent_column')
        
        # Even with allow_pending_aliases=True, strict should override
        result = adf.validate_schema(
            strict=True, 
            allow_pending_aliases=True,  # This should be overridden
            verbose=False
        )
        
        assert result['valid'] is False

    def test_valid_schema_passes_strict(self, sample_df):
        """Valid schema with no issues passes strict validation."""
        adf = AliasDataFrame(sample_df)
        adf.add_alias('sum_xy', 'x + dy')  # Both columns exist
        
        result = adf.validate_schema(strict=True, verbose=False)
        
        assert result['valid'] is True
        assert len(result['errors']) == 0


# =============================================================================
# Test: Cycle Detection
# =============================================================================

class TestCycleDetection:
    """Tests for cycle detection when loading schemas."""

    def test_cycle_detected_when_compression_target_is_alias(self, sample_df):
        """Cycle detected when compression target registered as alias."""
        adf = AliasDataFrame(sample_df)
        
        # Define compression schema
        adf.define_compression_schema({
            'dy': {
                'compress': 'round(asinh(dy)*40)',
                'decompress': 'sinh(dy_c/40.)',
                'compressed_dtype': 'int16',
                'decompressed_dtype': 'float16'
            }
        })
        
        # Manually add alias (simulating bad schema load)
        adf._restore_aliases_from_dict({'dy': 'sinh(dy_c/40.)'})
        
        # Validation should detect cycle risk
        result = adf.validate_schema(verbose=False)
        
        assert result['valid'] is False
        assert any('CYCLE' in err for err in result['errors'])

    def test_no_cycle_after_proper_compression(self, sample_df, compression_spec):
        """No cycle detected after proper compression."""
        adf = AliasDataFrame(sample_df)
        adf.compress_columns(compression_spec, columns=['dy'])
        
        # After compression, dy is alias but dy_c exists - no cycle
        result = adf.validate_schema(verbose=False)
        
        # Should be valid (alias depends on existing column)
        # Cycle only when dy is BOTH physical AND has alias depending on dy_c
        assert result['valid'] is True

    def test_implicit_compression_detection(self, sample_df, compression_spec):
        """Detect when data is already compressed (idempotent behavior)."""
        # First compress normally
        adf = AliasDataFrame(sample_df.copy())
        adf.compress_columns(compression_spec, columns=['dy'])
        
        # Verify compressed
        assert 'dy_c' in adf.df.columns
        assert 'dy' not in adf.df.columns
        
        # Call compress again - should be idempotent
        adf.compress_columns(columns=['dy'])
        
        # Should still be compressed (no change)
        assert adf.get_compression_state('dy') == CompressionState.COMPRESSED
        assert 'dy_c' in adf.df.columns


# =============================================================================
# Test: Deprecation Warning
# =============================================================================

class TestDeprecationWarning:
    """Tests for deprecation warnings on old-format schemas."""

    def test_old_format_schema_emits_warning(self, sample_df):
        """Old-format schema (compression target with expr) emits DeprecationWarning."""
        old_schema = {
            '__meta__': {'schema_version': 2},
            'columns': {
                'dy': {'dtype': 'float16', 'expr': 'sinh(dy_c/40.)'},  # OLD FORMAT
                'dy_c': {'dtype': 'int16'},
                'x': {'dtype': 'float32'}
            },
            'compression': {
                'dy': {
                    'compressed_col': 'dy_c',
                    'compress_expr': 'round(asinh(dy)*40)',
                    'decompress_expr': 'sinh(dy_c/40.)',
                    'compressed_dtype': 'int16',
                    'decompressed_dtype': 'float16'
                }
            }
        }
        
        adf = AliasDataFrame(sample_df)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            adf.apply_schema(old_schema, warn_missing=False)
            
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) > 0
            assert 'compression target' in str(deprecation_warnings[0].message).lower()

    def test_new_format_schema_no_warning(self, sample_df):
        """New-format schema (compression target physical) emits no DeprecationWarning."""
        new_schema = {
            '__meta__': {'schema_version': 2},
            'columns': {
                'dy': {'dtype': 'float16'},  # NEW FORMAT: physical, no expr
                'x': {'dtype': 'float32'}
            },
            'compression': {
                'dy': {
                    'compressed_col': 'dy_c',
                    'compress_expr': 'round(asinh(dy)*40)',
                    'decompress_expr': 'sinh(dy_c/40.)',
                    'compressed_dtype': 'int16',
                    'decompressed_dtype': 'float16'
                }
            }
        }
        
        adf = AliasDataFrame(sample_df)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            adf.apply_schema(new_schema, warn_missing=False)
            
            deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(deprecation_warnings) == 0

    def test_old_schema_still_loads(self, sample_df):
        """Old-format schema still loads successfully (backward compatible)."""
        old_schema = {
            '__meta__': {'schema_version': 2},
            'columns': {
                'dy': {'dtype': 'float16', 'expr': 'sinh(dy_c/40.)'},
            },
            'compression': {
                'dy': {
                    'compressed_col': 'dy_c',
                    'compress_expr': 'round(asinh(dy)*40)',
                    'decompress_expr': 'sinh(dy_c/40.)',
                    'compressed_dtype': 'int16',
                    'decompressed_dtype': 'float16'
                }
            }
        }
        
        adf = AliasDataFrame(sample_df)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # Suppress warnings for this test
            adf.apply_schema(old_schema, warn_missing=False)
        
        # Schema should have loaded
        assert 'dy' in adf.aliases


# =============================================================================
# Test: Schema Change Protection
# =============================================================================

class TestSchemaChangeProtection:
    """Tests for schema change protection on compressed columns."""

    def test_different_schema_raises_error(self, sample_df, compression_spec):
        """Compressing with different schema raises ValueError."""
        adf = AliasDataFrame(sample_df)
        adf.compress_columns(compression_spec, columns=['dy'])
        
        # Different schema
        new_spec = {
            'dy': {
                'compress': 'round(dy*1000)',  # Different transform
                'decompress': 'dy_c/1000.',
                'compressed_dtype': np.int16,
                'decompressed_dtype': np.float32
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            adf.compress_columns(new_spec, columns=['dy'])
        
        assert 'different schema' in str(exc_info.value).lower()
        assert 'decompress first' in str(exc_info.value).lower()

    def test_same_schema_idempotent(self, sample_df, compression_spec):
        """Compressing with same schema is idempotent (no error)."""
        adf = AliasDataFrame(sample_df)
        adf.compress_columns(compression_spec, columns=['dy'])
        
        # Same schema again - should be idempotent
        adf.compress_columns(compression_spec, columns=['dy'])
        
        # Should still be compressed
        assert adf.get_compression_state('dy') == CompressionState.COMPRESSED
        assert 'dy_c' in adf.df.columns

    def test_reuse_mode_idempotent(self, sample_df, compression_spec):
        """Reuse mode (no spec) is idempotent."""
        adf = AliasDataFrame(sample_df)
        adf.compress_columns(compression_spec, columns=['dy'])
        
        # Reuse mode - no spec, just column name
        adf.compress_columns(columns=['dy'])
        
        # Should still be compressed
        assert adf.get_compression_state('dy') == CompressionState.COMPRESSED


# =============================================================================
# Test: Validation Output Structure
# =============================================================================

class TestValidationOutputStructure:
    """Tests for validation return structure."""

    def test_output_has_required_keys(self, sample_df):
        """Validation output has all required keys."""
        adf = AliasDataFrame(sample_df)
        result = adf.validate_schema(verbose=False)
        
        assert 'valid' in result
        assert 'errors' in result
        assert 'warnings' in result
        assert 'pending' in result
        assert 'info' in result

    def test_info_has_required_keys(self, sample_df, compression_spec):
        """Info dict has all required keys."""
        adf = AliasDataFrame(sample_df)
        adf.define_compression_schema(compression_spec)
        adf.add_alias('sum_xy', 'x + dy')
        
        result = adf.validate_schema(verbose=False)
        info = result['info']
        
        assert 'compression_targets' in info
        assert 'physical_columns' in info
        assert 'aliases' in info
        assert 'pending_aliases' in info
        assert 'pending_columns' in info

    def test_compression_targets_populated(self, sample_df, compression_spec):
        """Compression targets list is populated correctly."""
        adf = AliasDataFrame(sample_df)
        adf.define_compression_schema(compression_spec)
        
        result = adf.validate_schema(verbose=False)
        
        assert 'dy' in result['info']['compression_targets']
        assert 'dz' in result['info']['compression_targets']

    def test_pending_aliases_structure(self, sample_df):
        """Pending aliases have correct structure."""
        adf = AliasDataFrame(sample_df)
        adf.add_alias('missing_dep', 'x + nonexistent')
        
        result = adf.validate_schema(verbose=False)
        
        assert len(result['info']['pending_aliases']) == 1
        pending = result['info']['pending_aliases'][0]
        assert 'name' in pending
        assert 'missing' in pending
        assert pending['name'] == 'missing_dep'


# =============================================================================
# Test: Full Workflow
# =============================================================================

class TestFullWorkflow:
    """Integration tests for complete workflows."""

    def test_definition_schema_apply_compress_workflow(self, sample_df, compression_spec):
        """Full workflow: export definition → apply to fresh data → compress."""
        # Create and compress original
        adf1 = AliasDataFrame(sample_df.copy())
        adf1.compress_columns(compression_spec, columns=['dy', 'dz'])
        
        # Export definition schema
        def_schema = adf1.export_definition_schema()
        
        # Apply to fresh data
        fresh_df = pd.DataFrame({
            'dy': np.random.randn(50).astype('float32'),
            'dz': np.random.randn(50).astype('float32'),
            'x': np.random.randn(50).astype('float32'),
        })
        adf2 = AliasDataFrame(fresh_df)
        adf2.apply_schema(def_schema, warn_missing=False)
        
        # Validate before compression
        result = adf2.validate_schema(verbose=False)
        assert result['valid'] is True
        
        # Compress
        adf2.compress_columns(columns=['dy', 'dz'])
        
        # Verify compression
        assert adf2.get_compression_state('dy') == CompressionState.COMPRESSED
        assert adf2.get_compression_state('dz') == CompressionState.COMPRESSED

    def test_strict_validation_before_export(self, sample_df, compression_spec):
        """Strict validation passes before production export."""
        adf = AliasDataFrame(sample_df)
        adf.add_alias('ratio', 'dy / dz')
        adf.compress_columns(compression_spec, columns=['dy', 'dz'])
        
        # Strict validation should pass
        result = adf.validate_schema(strict=True, verbose=False)
        assert result['valid'] is True
        
        # Can export
        def_schema = adf.export_definition_schema()
        rec_schema = adf.export_record_schema()
        
        assert def_schema is not None
        assert rec_schema is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
