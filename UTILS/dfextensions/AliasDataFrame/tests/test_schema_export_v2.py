# =============================================================================
# Tests for Enhanced Schema Export v2
# =============================================================================

import pytest
import json
import tempfile
import os
import sys

import pandas as pd
import numpy as np

# Try to import from AliasDataFrame module (package structure)
try:
    # Direct import from module file within package
    from AliasDataFrame.AliasDataFrame import (
        _format_json_smart,
        _order_columns_by_groups,
        _export_column_spec_v2,
        _repair_index_columns,
        AliasDataFrame,
        SCHEMA_VERSION_V2
    )
    USE_ADF_MODULE = True
    def load_schema_v2(path):
        return AliasDataFrame.load_schema_v2(path)
except ImportError:
    try:
        # Fallback: direct module import (when AliasDataFrame.py is in same directory)
        from AliasDataFrame import (
            _format_json_smart,
            _order_columns_by_groups,
            _export_column_spec_v2,
            _repair_index_columns,
            AliasDataFrame,
            SCHEMA_VERSION_V2
        )
        USE_ADF_MODULE = True
        def load_schema_v2(path):
            return AliasDataFrame.load_schema_v2(path)
    except ImportError:
        # Fallback to standalone schema_export_v2 module for unit testing
        from schema_export_v2 import (
            _format_json_smart,
            _order_columns_by_groups,
            _export_column_spec_v2,
            _repair_index_columns,
            load_schema_v2,
            SCHEMA_VERSION_V2
        )
        USE_ADF_MODULE = False


# =============================================================================
# Test: Smart JSON Formatting
# =============================================================================

class TestSmartJsonFormatting:
    """Test _format_json_smart function."""
    
    def test_short_entries_stay_on_one_line(self):
        """Short column entries should remain on one line."""
        data = {
            "columns": {
                "x": {"dtype": "float32"},
                "y": {"dtype": "float32", "unit": "cm"}
            }
        }
        result = _format_json_smart(data, max_line_length=100)
        
        # Check that short entries are on one line
        assert '"x": {"dtype": "float32"}' in result
        assert '"y": {"dtype": "float32", "unit": "cm"}' in result
    
    def test_long_entries_are_expanded(self):
        """Long entries should be expanded across multiple lines."""
        long_expr = "dyC1_intercept_DITS0SideFit+dyC1_slope_rrel_DITS0SideFit*rrel+dyC1_slope_rrel2_DITS0SideFit*rrel2"
        data = {
            "columns": {
                "longAlias": {"dtype": "float16", "expr": long_expr}
            }
        }
        result = _format_json_smart(data, max_line_length=80)
        
        # Should be expanded
        lines = result.split('\n')
        # Find lines with "expr" - should be on separate line from key
        assert any('"expr":' in line for line in lines)
    
    def test_groups_format_correctly(self):
        """Groups section should format as expected."""
        data = {
            "groups": {
                "coord": ["x", "y", "z"],
                "cuts": ["isOK", "isNotEdge"]
            }
        }
        result = _format_json_smart(data, max_line_length=100)
        
        # Groups should be readable
        assert '"coord": ["x", "y", "z"]' in result
        assert '"cuts": ["isOK", "isNotEdge"]' in result
    
    def test_valid_json_output(self):
        """Output should be valid JSON."""
        data = {
            "__meta__": {"schema_version": 2},
            "columns": {
                "x": {"dtype": "float32"},
                "y": {"dtype": "float32", "expr": "x*2"}
            }
        }
        result = _format_json_smart(data)
        
        # Should parse without error
        parsed = json.loads(result)
        assert parsed["__meta__"]["schema_version"] == 2
        assert parsed["columns"]["x"]["dtype"] == "float32"


# =============================================================================
# Test: Column Ordering by Groups
# =============================================================================

class TestColumnOrderingByGroups:
    """Test _order_columns_by_groups function."""
    
    def test_groups_ordered_first(self):
        """Columns in groups should appear first in order."""
        columns = {
            "ungrouped": {"dtype": "float32"},
            "x": {"dtype": "float32"},
            "y": {"dtype": "float32"},
            "isOK": {"dtype": "bool"}
        }
        groups = {
            "coord": ["x", "y"],
            "cuts": ["isOK"]
        }
        
        result = _order_columns_by_groups(columns, groups)
        keys = list(result.keys())
        
        # Group columns should come first in group order
        assert keys.index("x") < keys.index("ungrouped")
        assert keys.index("y") < keys.index("ungrouped")
        assert keys.index("isOK") < keys.index("ungrouped")
        # Within groups, preserve group list order
        assert keys.index("x") < keys.index("y")
    
    def test_alphabetic_within_group_sort(self):
        """With alphabetic sort, columns within groups are sorted A-Z."""
        columns = {
            "z": {"dtype": "float32"},
            "a": {"dtype": "float32"},
            "m": {"dtype": "float32"}
        }
        groups = {"letters": ["z", "a", "m"]}
        
        result = _order_columns_by_groups(columns, groups, within_group_sort="alphabetic")
        keys = list(result.keys())
        
        # Should be alphabetically sorted within group
        assert keys == ["a", "m", "z"]
    
    def test_schema_order_preserves_group_list(self):
        """With schema sort (default), preserve order from groups dict."""
        columns = {
            "z": {"dtype": "float32"},
            "a": {"dtype": "float32"},
            "m": {"dtype": "float32"}
        }
        groups = {"letters": ["z", "a", "m"]}
        
        result = _order_columns_by_groups(columns, groups, within_group_sort="schema")
        keys = list(result.keys())
        
        # Should preserve group list order
        assert keys == ["z", "a", "m"]
    
    def test_missing_columns_in_group_ignored(self):
        """Columns listed in group but not in columns dict are ignored."""
        columns = {"x": {"dtype": "float32"}}
        groups = {"coord": ["x", "y", "z"]}  # y, z don't exist
        
        result = _order_columns_by_groups(columns, groups)
        
        assert list(result.keys()) == ["x"]
    
    def test_empty_groups_returns_original(self):
        """Empty groups dict returns columns unchanged."""
        columns = {"b": {}, "a": {}, "c": {}}
        
        result = _order_columns_by_groups(columns, {})
        
        assert list(result.keys()) == ["b", "a", "c"]


# =============================================================================
# Test: Column Spec Export
# =============================================================================

class TestColumnSpecExport:
    """Test _export_column_spec_v2 function."""
    
    def test_physical_column_no_expr_null(self):
        """Physical columns should not have 'expr': null."""
        col_info = {"dtype": np.float32, "expr": None}
        
        result = _export_column_spec_v2("x", col_info)
        
        assert result == {"dtype": "float32"}
        assert "expr" not in result
    
    def test_alias_column_has_expr(self):
        """Alias columns should have expr field."""
        col_info = {"dtype": np.float32, "expr": "sqrt(x**2+y**2)"}
        
        result = _export_column_spec_v2("r", col_info)
        
        assert result["dtype"] == "float32"
        assert result["expr"] == "sqrt(x**2+y**2)"
    
    def test_metadata_preserved(self):
        """Column metadata (unit, axisLabel, etc.) should be preserved."""
        col_info = {
            "dtype": np.float32,
            "unit": "cm",
            "axisLabel": "x (cm)",
            "description": "cluster position"
        }
        
        result = _export_column_spec_v2("x", col_info)
        
        assert result["unit"] == "cm"
        assert result["axisLabel"] == "x (cm)"
        assert result["description"] == "cluster position"
    
    def test_constant_only_if_true(self):
        """constant field only included if True."""
        col_info = {"dtype": np.float32, "constant": False}
        result1 = _export_column_spec_v2("x", col_info)
        
        col_info = {"dtype": np.float32, "constant": True}
        result2 = _export_column_spec_v2("x", col_info)
        
        assert "constant" not in result1
        assert result2["constant"] == True
    
    def test_dtype_from_dataframe(self):
        """If dtype not in col_info, get from DataFrame."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        
        result = _export_column_spec_v2("x", {}, df)
        
        assert result["dtype"] == "float64"
    
    def test_numpy_dtype_conversion(self):
        """Various numpy dtype formats should convert to string."""
        # numpy dtype object
        result1 = _export_column_spec_v2("x", {"dtype": np.dtype("float16")})
        assert result1["dtype"] == "float16"
        
        # numpy type
        result2 = _export_column_spec_v2("x", {"dtype": np.float32})
        assert result2["dtype"] == "float32"
        
        # string (passthrough)
        result3 = _export_column_spec_v2("x", {"dtype": "int8"})
        assert result3["dtype"] == "int8"


# =============================================================================
# Test: Index Column Repair
# =============================================================================

class TestIndexColumnRepair:
    """Test _repair_index_columns function."""
    
    def test_repairs_corrupted_index(self):
        """Detects and repairs character-iterated index."""
        corrupted = ['t', 'r', 'a', 'c', 'k', '_', 't', 'f', '_', 'u', 'i', 'd']
        
        with pytest.warns(UserWarning, match="Repaired corrupted index"):
            result = _repair_index_columns(corrupted, "T")
        
        assert result == ["track_tf_uid"]
    
    def test_does_not_repair_valid_multikey(self):
        """Valid multi-key indices are not changed."""
        valid = ["side", "row", "drift25"]
        
        result = _repair_index_columns(valid, "DTrack0")
        
        assert result == ["side", "row", "drift25"]
    
    def test_does_not_repair_single_element(self):
        """Single-element list is not changed."""
        single = ["track_tf_uid"]
        
        result = _repair_index_columns(single, "T")
        
        assert result == ["track_tf_uid"]
    
    def test_handles_string_input(self):
        """String input is returned as-is (caller handles conversion)."""
        result = _repair_index_columns("track_tf_uid", "T")
        
        assert result == "track_tf_uid"


# =============================================================================
# Test: Schema Loading (v1 → v2 normalization)
# =============================================================================

class TestSchemaLoading:
    """Test load_schema_v2 function."""
    
    def test_loads_v2_schema_unchanged(self):
        """V2 schema is loaded without modification."""
        v2_schema = {
            "__meta__": {"schema_version": 2, "schema_id": "test"},
            "columns": {
                "x": {"dtype": "float32"},
                "r": {"dtype": "float32", "expr": "sqrt(x**2)"}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(v2_schema, f)
            path = f.name
        
        try:
            result = load_schema_v2(path)
            assert result == v2_schema
        finally:
            os.unlink(path)
    
    def test_normalizes_v1_schema(self):
        """V1 schema is normalized to v2 format."""
        v1_schema = {
            "__meta__": {"schema_version": 1},
            "columns": {
                "x": {"dtype": "float32", "expr": None},  # Old format with expr: null
                "r": {"dtype": "float32", "expr": "sqrt(x**2)"}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(v1_schema, f)
            path = f.name
        
        try:
            result = load_schema_v2(path)
            
            # Should be normalized - no expr: null
            assert "expr" not in result["columns"]["x"]
            assert result["columns"]["r"]["expr"] == "sqrt(x**2)"
            assert result["__meta__"]["schema_version"] == 2
        finally:
            os.unlink(path)
    
    def test_repairs_corrupted_subframe_index(self):
        """Corrupted subframe indices are repaired on load."""
        schema_with_corruption = {
            "__meta__": {"schema_version": 1},
            "columns": {},
            "subframes": {
                "T": {"index": ['t', 'r', 'a', 'c', 'k']}  # Corrupted
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema_with_corruption, f)
            path = f.name
        
        try:
            with pytest.warns(UserWarning, match="Repaired corrupted index"):
                result = load_schema_v2(path)
            
            assert result["subframes"]["T"]["index"] == ["track"]
        finally:
            os.unlink(path)
    
    def test_converts_string_index_to_list(self):
        """String index is converted to list."""
        schema = {
            "__meta__": {"schema_version": 1},
            "columns": {},
            "subframes": {
                "T": {"index": "track_tf_uid"}  # String, not list
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(schema, f)
            path = f.name
        
        try:
            result = load_schema_v2(path)
            assert result["subframes"]["T"]["index"] == ["track_tf_uid"]
        finally:
            os.unlink(path)


# =============================================================================
# Test: Round-trip (export → save → load)
# =============================================================================

class TestSchemaRoundTrip:
    """Test full round-trip: export → save → load."""
    
    def test_round_trip_preserves_data(self):
        """Schema survives export → save → load cycle."""
        original = {
            "__meta__": {"schema_version": 2, "schema_id": "roundtrip_test"},
            "groups": {"coord": ["x", "y"]},
            "columns": {
                "x": {"dtype": "float32", "unit": "cm"},
                "y": {"dtype": "float32", "unit": "cm"},
                "r": {"dtype": "float32", "expr": "sqrt(x**2+y**2)"}
            },
            "subframes": {
                "T": {
                    "index": ["track_tf_uid"],
                    "columns": {
                        "track_tf_uid": {"dtype": "uint32"},
                        "pt": {"dtype": "float32"}
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Save with smart formatting
            json_str = _format_json_smart(original)
            f.write(json_str)
            path = f.name
        
        try:
            loaded = load_schema_v2(path)
            
            # Verify all data preserved
            assert loaded["__meta__"]["schema_id"] == "roundtrip_test"
            assert loaded["groups"]["coord"] == ["x", "y"]
            assert loaded["columns"]["x"]["unit"] == "cm"
            assert loaded["columns"]["r"]["expr"] == "sqrt(x**2+y**2)"
            assert loaded["subframes"]["T"]["index"] == ["track_tf_uid"]
            assert loaded["subframes"]["T"]["columns"]["pt"]["dtype"] == "float32"
        finally:
            os.unlink(path)


# =============================================================================
# Test: jq Queryability
# =============================================================================

class TestJqQueryability:
    """Test that exported schema is easily queryable with jq patterns."""
    
    def test_uniform_dtype_access(self):
        """All columns should have .dtype accessible uniformly."""
        schema = {
            "columns": {
                "x": {"dtype": "float32"},  # physical
                "r": {"dtype": "float32", "expr": "sqrt(x)"}  # alias
            }
        }
        
        # Simulate jq: .columns[].dtype
        dtypes = [spec["dtype"] for spec in schema["columns"].values()]
        assert dtypes == ["float32", "float32"]
    
    def test_filter_aliases_by_expr(self):
        """Can filter aliases by presence of expr field."""
        schema = {
            "columns": {
                "x": {"dtype": "float32"},
                "y": {"dtype": "float32"},
                "r": {"dtype": "float32", "expr": "sqrt(x**2+y**2)"},
                "phi": {"dtype": "float32", "expr": "atan2(y,x)"}
            }
        }
        
        # Simulate jq: .columns | to_entries[] | select(.value.expr) | .key
        aliases = [name for name, spec in schema["columns"].items() if "expr" in spec]
        assert set(aliases) == {"r", "phi"}
    
    def test_filter_columns_with_units(self):
        """Can filter columns that have unit metadata."""
        schema = {
            "columns": {
                "x": {"dtype": "float32", "unit": "cm"},
                "y": {"dtype": "float32", "unit": "cm"},
                "isOK": {"dtype": "bool"}
            }
        }
        
        # Simulate jq: .columns | to_entries[] | select(.value.unit) | .key
        with_units = [name for name, spec in schema["columns"].items() if "unit" in spec]
        assert set(with_units) == {"x", "y"}


# =============================================================================
# Test: Subframe Schema Population (Bug Fix)
# =============================================================================

class TestSubframeSchemaPopulation:
    """
    Test that subframes with empty schemas get their columns populated.
    
    Bug: When subframes are loaded from ROOT without embedded schema,
    their _schema['columns'] was empty, causing export_schema_v2() to
    produce empty column blocks for subframes.
    
    Fix: register_subframe() now auto-populates the subframe's schema
    from its DataFrame columns if empty.
    """
    
    def test_subframe_schema_populated_on_register(self):
        """Subframe with empty schema gets columns populated during registration."""
        # Create main frame
        main_df = pd.DataFrame({
            'x': np.array([1.0, 2.0], dtype=np.float32),
            'track_id': np.array([0, 1], dtype=np.int32)
        })
        adf = AliasDataFrame(main_df)
        
        # Create subframe with data but empty schema (simulates ROOT load without schema)
        sub_df = pd.DataFrame({
            'track_id': np.array([0, 1], dtype=np.int32),
            'pt': np.array([1.5, 2.5], dtype=np.float32),
            'eta': np.array([0.5, -0.5], dtype=np.float16)
        })
        sub_adf = AliasDataFrame(sub_df)
        
        # Verify schema is initially empty (no explicit column definitions)
        # Note: __init__ may create empty columns dict, but no column specs
        initial_columns = sub_adf._schema.get('columns', {})
        # Filter to only columns with actual dtype definitions
        defined_cols = {k: v for k, v in initial_columns.items() if v.get('dtype')}
        assert len(defined_cols) == 0 or len(defined_cols) == len(sub_df.columns)
        
        # Register subframe - this should populate schema if empty
        adf.register_subframe('T', sub_adf, index_columns='track_id')
        
        # Verify subframe schema now has columns
        assert 'columns' in sub_adf._schema
        assert 'track_id' in sub_adf._schema['columns']
        assert 'pt' in sub_adf._schema['columns']
        assert 'eta' in sub_adf._schema['columns']
        
        # Verify dtypes are correct
        assert sub_adf._schema['columns']['pt']['dtype'] == 'float32'
        assert sub_adf._schema['columns']['eta']['dtype'] == 'float16'
        assert sub_adf._schema['columns']['track_id']['dtype'] == 'int32'
    
    def test_subframe_empty_schema_simulates_root_load(self):
        """
        Simulate exact bug scenario: subframe loaded from ROOT without embedded schema.
        
        This test explicitly clears the schema to mimic what happens when
        read_tree() loads a subframe tree that has no TObjString metadata.
        """
        # Create main frame
        main_df = pd.DataFrame({
            'x': np.array([1.0, 2.0, 3.0], dtype=np.float32),
            'track_tf_uid': np.array([0, 1, 2], dtype=np.uint32)
        })
        adf = AliasDataFrame(main_df)
        
        # Create subframe - simulating what read_tree() produces for a tree without schema
        sub_df = pd.DataFrame({
            'track_tf_uid': np.array([0, 1, 2], dtype=np.uint32),
            'mX': np.array([39.0, 40.0, 41.0], dtype=np.float32),
            'mP4': np.array([0.5, -0.3, 0.1], dtype=np.float32),
            'nClsTPC': np.array([150, 145, 148], dtype=np.uint8)
        })
        sub_adf = AliasDataFrame(sub_df)
        
        # CRITICAL: Clear the schema to simulate ROOT load without embedded schema
        # This is exactly what happens when read_tree() loads a subframe tree
        # that doesn't have TObjString metadata with column info
        sub_adf._schema['columns'] = {}
        
        # Verify schema is empty (the bug condition)
        assert len(sub_adf._schema.get('columns', {})) == 0
        
        # But DataFrame has data
        assert len(sub_adf.df.columns) == 4
        
        # Register subframe - the fix should populate schema
        adf.register_subframe('T', sub_adf, index_columns='track_tf_uid')
        
        # Verify schema is now populated
        assert len(sub_adf._schema['columns']) == 4
        assert sub_adf._schema['columns']['mX']['dtype'] == 'float32'
        assert sub_adf._schema['columns']['nClsTPC']['dtype'] == 'uint8'
        
        # Verify export includes subframe columns
        schema = adf.export_schema_v2()
        assert 'T' in schema['subframes']
        assert len(schema['subframes']['T']['columns']) == 4
        assert schema['subframes']['T']['columns']['mX']['dtype'] == 'float32'
    
    def test_subframe_columns_in_exported_schema(self):
        """export_schema_v2() includes subframe columns."""
        # Create main frame
        main_df = pd.DataFrame({
            'x': np.array([1.0, 2.0], dtype=np.float32),
            'track_id': np.array([0, 1], dtype=np.int32)
        })
        adf = AliasDataFrame(main_df)
        
        # Create and register subframe
        sub_df = pd.DataFrame({
            'track_id': np.array([0, 1], dtype=np.int32),
            'pt': np.array([1.5, 2.5], dtype=np.float32),
            'phi': np.array([0.1, 0.2], dtype=np.float32)
        })
        sub_adf = AliasDataFrame(sub_df)
        adf.register_subframe('T', sub_adf, index_columns='track_id')
        
        # Export schema
        schema = adf.export_schema_v2()
        
        # Verify subframe has columns in exported schema
        assert 'subframes' in schema
        assert 'T' in schema['subframes']
        assert 'columns' in schema['subframes']['T']
        
        sf_columns = schema['subframes']['T']['columns']
        assert 'track_id' in sf_columns
        assert 'pt' in sf_columns
        assert 'phi' in sf_columns
        
        # Verify dtypes
        assert sf_columns['pt']['dtype'] == 'float32'
    
    def test_subframe_describe_schema_works(self):
        """Subframe's describe_schema() shows columns after registration."""
        # Create main frame
        main_df = pd.DataFrame({
            'x': np.array([1.0, 2.0], dtype=np.float32),
            'track_id': np.array([0, 1], dtype=np.int32)
        })
        adf = AliasDataFrame(main_df)
        
        # Create and register subframe
        sub_df = pd.DataFrame({
            'track_id': np.array([0, 1], dtype=np.int32),
            'pt': np.array([1.5, 2.5], dtype=np.float32)
        })
        sub_adf = AliasDataFrame(sub_df)
        adf.register_subframe('T', sub_adf, index_columns='track_id')
        
        # Get subframe and check its schema has columns
        sf = adf.subframe('T')
        assert len(sf._schema.get('columns', {})) > 0
        assert 'pt' in sf._schema['columns']
    
    def test_subframe_with_existing_schema_not_overwritten(self):
        """Subframe with existing schema is not overwritten."""
        # Create main frame
        main_df = pd.DataFrame({
            'x': np.array([1.0, 2.0], dtype=np.float32),
            'track_id': np.array([0, 1], dtype=np.int32)
        })
        adf = AliasDataFrame(main_df)
        
        # Create subframe with existing schema (includes an alias)
        sub_df = pd.DataFrame({
            'track_id': np.array([0, 1], dtype=np.int32),
            'px': np.array([1.0, 2.0], dtype=np.float32),
            'py': np.array([0.5, 1.0], dtype=np.float32)
        })
        sub_adf = AliasDataFrame(sub_df)
        sub_adf.add_alias('pt', 'sqrt(px**2+py**2)', dtype=np.float32)
        
        # Schema already has columns
        assert 'pt' in sub_adf._schema['columns']
        assert sub_adf._schema['columns']['pt'].get('expr') == 'sqrt(px**2+py**2)'
        
        # Register subframe - should NOT overwrite existing schema
        adf.register_subframe('T', sub_adf, index_columns='track_id')
        
        # Verify alias is preserved
        assert sub_adf._schema['columns']['pt'].get('expr') == 'sqrt(px**2+py**2)'
    
    def test_export_schema_v2_roundtrip_with_subframes(self):
        """Schema with subframe columns survives save/load cycle."""
        # Create main frame with subframe
        main_df = pd.DataFrame({
            'x': np.array([1.0, 2.0], dtype=np.float32),
            'track_id': np.array([0, 1], dtype=np.int32)
        })
        adf = AliasDataFrame(main_df)
        
        sub_df = pd.DataFrame({
            'track_id': np.array([0, 1], dtype=np.int32),
            'pt': np.array([1.5, 2.5], dtype=np.float32)
        })
        sub_adf = AliasDataFrame(sub_df)
        adf.register_subframe('T', sub_adf, index_columns='track_id')
        
        # Export schema
        schema = adf.export_schema_v2()
        
        # Save and reload
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_str = _format_json_smart(schema)
            f.write(json_str)
            path = f.name
        
        try:
            loaded = load_schema_v2(path)
            
            # Verify subframe columns survived
            assert 'T' in loaded['subframes']
            assert 'columns' in loaded['subframes']['T']
            assert 'pt' in loaded['subframes']['T']['columns']
            assert loaded['subframes']['T']['columns']['pt']['dtype'] == 'float32'
        finally:
            os.unlink(path)
    
    def test_multiple_subframes_all_populated(self):
        """
        Multiple subframes all get their schemas populated.
        
        Simulates real ALICE data with T, R, DTrack0, DITS0FitSide subframes.
        """
        # Create main frame
        main_df = pd.DataFrame({
            'x': np.array([1.0, 2.0], dtype=np.float32),
            'track_tf_uid': np.array([0, 1], dtype=np.uint32),
            'firstTForbit': np.array([30537824, 30537824], dtype=np.uint32),
            'row': np.array([50, 100], dtype=np.uint8),
            'drift25': np.array([10, 15], dtype=np.int8),
            'side': np.array([0, 1], dtype=np.int8)
        })
        adf = AliasDataFrame(main_df)
        
        # Create multiple subframes with empty schemas (simulating ROOT load)
        # Subframe T (tracks)
        sub_T = AliasDataFrame(pd.DataFrame({
            'track_tf_uid': np.array([0, 1], dtype=np.uint32),
            'mX': np.array([39.0, 40.0], dtype=np.float32),
            'mP4': np.array([0.5, -0.3], dtype=np.float32)
        }))
        sub_T._schema['columns'] = {}  # Clear to simulate ROOT load
        
        # Subframe R (run info)
        sub_R = AliasDataFrame(pd.DataFrame({
            'firstTForbit': np.array([30537824], dtype=np.uint32),
            'timestampMS': np.array([1700000000000], dtype=np.int64),
            'pressure': np.array([1013.25], dtype=np.float32)
        }))
        sub_R._schema['columns'] = {}  # Clear to simulate ROOT load
        
        # Subframe DTrack0 (calibration)
        sub_DTrack0 = AliasDataFrame(pd.DataFrame({
            'side': np.array([0, 1], dtype=np.int8),
            'row': np.array([50, 100], dtype=np.uint8),
            'drift25': np.array([10, 15], dtype=np.int8),
            'dyC0T_median': np.array([0.01, -0.02], dtype=np.float32)
        }))
        sub_DTrack0._schema['columns'] = {}  # Clear to simulate ROOT load
        
        # Register all subframes
        adf.register_subframe('T', sub_T, index_columns='track_tf_uid')
        adf.register_subframe('R', sub_R, index_columns='firstTForbit')
        adf.register_subframe('DTrack0', sub_DTrack0, index_columns=['side', 'row', 'drift25'])
        
        # Export schema
        schema = adf.export_schema_v2()
        
        # Verify all subframes have columns
        assert 'T' in schema['subframes']
        assert 'R' in schema['subframes']
        assert 'DTrack0' in schema['subframes']
        
        assert len(schema['subframes']['T']['columns']) == 3
        assert len(schema['subframes']['R']['columns']) == 3
        assert len(schema['subframes']['DTrack0']['columns']) == 4
        
        # Verify specific columns
        assert schema['subframes']['T']['columns']['mX']['dtype'] == 'float32'
        assert schema['subframes']['R']['columns']['timestampMS']['dtype'] == 'int64'
        assert schema['subframes']['DTrack0']['columns']['dyC0T_median']['dtype'] == 'float32'
        
        # Verify multi-key index preserved
        assert schema['subframes']['DTrack0']['index'] == ['side', 'row', 'drift25']


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
