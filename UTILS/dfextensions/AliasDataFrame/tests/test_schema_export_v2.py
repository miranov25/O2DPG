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
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
