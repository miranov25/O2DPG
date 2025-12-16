"""
Phase 13.2.DSL: Arrow export/import tests.

Tests ROOT ↔ Arrow bridge functionality.
"""

import pytest
import numpy as np


# Skip all tests if PyArrow not available
pa = pytest.importorskip("pyarrow")


class TestArrowExport:
    """Test to_arrow() functionality."""
    
    @pytest.fixture
    def dsl_with_data(self):
        """Create DSL with test RDataFrame."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        # Create test data
        numpy_dict = {
            'x': np.random.randn(1000).astype(np.float64),
            'y': np.random.randn(1000).astype(np.float64),
            'n': np.random.randint(0, 10, 1000).astype(np.int32),
        }
        rdf = ROOT.RDF.FromNumpy(numpy_dict)
        
        # Use simple schema format
        schema = {'x': 'double', 'y': 'double', 'n': 'int'}
        dsl = DSLCompiler(schema)
        dsl._rdf = rdf
        dsl.define('z', 'x + y')
        dsl.define('good', '(x > 0) & (y > 0)')  # Python syntax (& not &&)
        
        return dsl
    
    def test_to_arrow_basic(self, dsl_with_data):
        """Basic Arrow export works."""
        table = dsl_with_data.to_arrow()
        
        assert isinstance(table, pa.Table)
        assert 'x' in table.column_names
        assert 'y' in table.column_names
    
    def test_to_arrow_with_columns(self, dsl_with_data):
        """Export specific columns."""
        table = dsl_with_data.to_arrow(columns=['x', 'y'])
        
        assert table.num_columns == 2
        assert 'x' in table.column_names
        assert 'y' in table.column_names
        assert 'n' not in table.column_names
    
    def test_to_arrow_include_schema(self, dsl_with_data):
        """Schema included in metadata."""
        table = dsl_with_data.to_arrow(include_schema=True)
        
        assert table.schema.metadata is not None
        assert b'dsl_schema' in table.schema.metadata
    
    def test_to_arrow_exclude_schema(self, dsl_with_data):
        """Schema excluded when requested."""
        table = dsl_with_data.to_arrow(include_schema=False)
        
        if table.schema.metadata:
            assert b'dsl_schema' not in table.schema.metadata
    
    def test_to_arrow_no_rdf_error(self):
        """Clear error when no RDataFrame available."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({'x': 'double'})
        # No _rdf set
        
        with pytest.raises(ValueError, match='No RDataFrame available'):
            dsl.to_arrow()
    
    def test_to_arrow_without_pyarrow(self, monkeypatch):
        """Clear error when PyArrow not available."""
        from RDataFrameDSL import dsl_compiler
        monkeypatch.setattr(dsl_compiler, '_PYARROW_AVAILABLE', False)
        
        # Create minimal DSL
        dsl = dsl_compiler.DSLCompiler({'x': 'double'})
        
        with pytest.raises(ImportError, match='PyArrow required'):
            dsl.to_arrow()


class TestArrowImport:
    """Test from_arrow() functionality."""
    
    def test_from_arrow_basic(self):
        """Create DSL from Arrow table."""
        import pandas as pd
        from RDataFrameDSL import DSLCompiler
        ROOT = pytest.importorskip("ROOT")
        
        table = pa.Table.from_pandas(pd.DataFrame({
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0],
        }))
        
        dsl = DSLCompiler.from_arrow(table)
        
        assert dsl is not None
        assert 'x' in dsl.schema
        assert 'y' in dsl.schema
    
    def test_from_arrow_apply_schema(self):
        """Schema from metadata applied."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        # Create DSL with definitions
        numpy_dict = {
            'x': np.array([1.0, 2.0, 3.0]),
            'y': np.array([4.0, 5.0, 6.0]),
        }
        rdf = ROOT.RDF.FromNumpy(numpy_dict)
        
        schema = {'x': 'double', 'y': 'double'}
        original_dsl = DSLCompiler(schema)
        original_dsl._rdf = rdf
        original_dsl.define('z', 'x + y')
        
        # Export with schema
        table = original_dsl.to_arrow(include_schema=True)
        
        # Import
        with pytest.warns(UserWarning, match='Round-trip'):
            new_dsl = DSLCompiler.from_arrow(table, apply_schema=True)
        
        # Definitions should be recovered
        defs = new_dsl.get_definitions()
        assert 'z' in defs
    
    def test_from_arrow_no_schema(self):
        """Import without schema metadata."""
        import pandas as pd
        from RDataFrameDSL import DSLCompiler
        ROOT = pytest.importorskip("ROOT")
        
        table = pa.Table.from_pandas(pd.DataFrame({'x': [1.0, 2.0, 3.0]}))
        
        dsl = DSLCompiler.from_arrow(table, apply_schema=True)
        
        # Should work, just no definitions
        assert dsl.get_definitions() == {}
    
    def test_from_arrow_dtype_inference(self):
        """Dtype correctly inferred from Arrow types."""
        import pandas as pd
        from RDataFrameDSL import DSLCompiler
        ROOT = pytest.importorskip("ROOT")
        
        df = pd.DataFrame({
            'f64': np.array([1.0], dtype=np.float64),
            'f32': np.array([1.0], dtype=np.float32),
            'i32': np.array([1], dtype=np.int32),
            'i64': np.array([1], dtype=np.int64),
        })
        table = pa.Table.from_pandas(df)
        
        dsl = DSLCompiler.from_arrow(table)
        
        assert dsl.schema['f64'] == 'double'
        assert dsl.schema['f32'] == 'float'
        assert dsl.schema['i32'] == 'int'
        assert dsl.schema['i64'] == 'long'


class TestRoundTrip:
    """Test export → import round-trip."""
    
    @pytest.fixture
    def dsl_with_data(self):
        """Create DSL with test RDataFrame."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        numpy_dict = {
            'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'y': np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
        }
        rdf = ROOT.RDF.FromNumpy(numpy_dict)
        
        schema = {'x': 'double', 'y': 'double'}
        dsl = DSLCompiler(schema)
        dsl._rdf = rdf
        
        return dsl
    
    def test_data_preserved(self, dsl_with_data):
        """Data values preserved in round-trip."""
        from RDataFrameDSL import DSLCompiler
        
        # Export
        table = dsl_with_data.to_arrow(columns=['x', 'y'])
        original_x = table.column('x').to_numpy()
        
        # Import
        new_dsl = DSLCompiler.from_arrow(table)
        new_table = new_dsl.to_arrow(columns=['x', 'y'])
        roundtrip_x = new_table.column('x').to_numpy()
        
        np.testing.assert_array_equal(original_x, roundtrip_x)
    
    def test_schema_roundtrip(self, dsl_with_data):
        """Definitions preserved in round-trip."""
        from RDataFrameDSL import DSLCompiler
        
        # Add definition
        dsl_with_data.define('z', 'x + y')
        
        # Export
        table = dsl_with_data.to_arrow(include_schema=True)
        
        # Import
        with pytest.warns(UserWarning, match='Round-trip'):
            new_dsl = DSLCompiler.from_arrow(table, apply_schema=True)
        
        # Check definitions recovered
        defs = new_dsl.get_definitions()
        assert 'z' in defs


class TestBooleanConversion:
    """Test Python ↔ C++ expression conversion."""
    
    @pytest.fixture
    def dsl(self):
        from RDataFrameDSL import DSLCompiler
        return DSLCompiler({})
    
    def test_python_to_cpp_and(self, dsl):
        """& converted to &&."""
        result = dsl._python_to_cpp_expr('(a > 0) & (b > 0)')
        assert ' && ' in result
        assert ' & ' not in result
    
    def test_python_to_cpp_or(self, dsl):
        """| converted to ||."""
        result = dsl._python_to_cpp_expr('(a > 0) | (b > 0)')
        assert ' || ' in result
        assert ' | ' not in result.replace(' || ', '')
    
    def test_python_to_cpp_not_var(self, dsl):
        """~var converted to !var."""
        result = dsl._python_to_cpp_expr('~flag')
        assert result == '!flag'
    
    def test_python_to_cpp_not_expr(self, dsl):
        """~(expr) converted to !(expr)."""
        result = dsl._python_to_cpp_expr('~(a > 0)')
        assert result == '!(a > 0)'
    
    def test_roundtrip_and_or(self, dsl):
        """C++ → Python → C++ preserves semantics."""
        cpp_orig = '(a > 0) && (b > 0)'
        py_expr = dsl._cpp_to_python_expr(cpp_orig)
        cpp_back = dsl._python_to_cpp_expr(py_expr)
        
        assert '&&' in cpp_back
        assert '&' not in cpp_back.replace('&&', '')


class TestArrowMetadata:
    """Test Arrow metadata handling."""
    
    @pytest.fixture
    def dsl_with_data(self):
        """Create DSL with test RDataFrame."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        numpy_dict = {
            'x': np.array([1.0, 2.0, 3.0]),
            'y': np.array([4.0, 5.0, 6.0]),
        }
        rdf = ROOT.RDF.FromNumpy(numpy_dict)
        
        schema = {'x': 'double', 'y': 'double'}
        dsl = DSLCompiler(schema)
        dsl._rdf = rdf
        dsl.define('z', 'x + y')
        
        return dsl
    
    def test_metadata_integrity(self, dsl_with_data):
        """Schema survives Arrow operations."""
        import json
        
        table = dsl_with_data.to_arrow(include_schema=True)
        
        # Verify metadata structure
        meta = table.schema.metadata
        assert b'dsl_schema' in meta
        
        schema_dict = json.loads(meta[b'dsl_schema'].decode('utf-8'))
        assert 'columns' in schema_dict
        assert '__meta__' in schema_dict
    
    def test_rvec_columns_tracked(self, dsl_with_data):
        """RVec columns listed in metadata."""
        import json
        
        table = dsl_with_data.to_arrow(include_schema=True)
        
        meta = table.schema.metadata
        assert b'rvec_columns' in meta
        
        rvec_list = json.loads(meta[b'rvec_columns'].decode('utf-8'))
        assert isinstance(rvec_list, list)


class TestRVecHandling:
    """Test RVec → Arrow conversion."""
    
    @pytest.fixture
    def dsl_with_variable_rvec(self):
        """Create DSL with variable-length RVec columns."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        # Variable length: size ranges from 1 to 5
        rdf = ROOT.RDataFrame(100)
        rdf = rdf.Define("tracks", 
            "ROOT::RVecF((int)(rdfentry_ % 5) + 1, (float)rdfentry_)")
        
        schema = {'tracks': 'RVec<float>'}
        dsl = DSLCompiler(schema)
        dsl._rdf = rdf
        
        return dsl
    
    def test_rvec_to_listarray(self, dsl_with_variable_rvec):
        """RVec preserved as ListArray."""
        table = dsl_with_variable_rvec.to_arrow(flatten_rvec=False)
        
        col_type = table.column('tracks').type
        assert pa.types.is_list(col_type)
    
    def test_rvec_variable_length_listarray(self, dsl_with_variable_rvec):
        """Variable-length RVec preserved correctly."""
        table = dsl_with_variable_rvec.to_arrow(flatten_rvec=False)
        
        assert pa.types.is_list(table.column('tracks').type)
        
        # Verify offsets are not uniform
        list_col = table.column('tracks')
        lengths = [len(list_col[i].as_py()) for i in range(min(10, len(list_col)))]
        assert len(set(lengths)) > 1, "Expected variable-length arrays"
    
    def test_rvec_flatten(self, dsl_with_variable_rvec):
        """RVec flattened when requested."""
        table = dsl_with_variable_rvec.to_arrow(flatten_rvec=True)
        
        col_type = table.column('tracks').type
        # Should be flat, not list
        assert not pa.types.is_list(col_type)
    
    def test_rvec_flatten_length(self, dsl_with_variable_rvec):
        """Flattened RVec has correct total length."""
        table = dsl_with_variable_rvec.to_arrow(flatten_rvec=True)
        
        # 100 events with sizes 1-5 (repeating)
        # Sum of 1+2+3+4+5 = 15, times 20 full cycles = 300
        arr = table.column('tracks').to_numpy()
        assert len(arr) == 300


class TestArrowTypeInference:
    """Test _arrow_type_to_ctype helper."""
    
    def test_float64(self):
        from RDataFrameDSL.dsl_compiler import _arrow_type_to_ctype
        assert _arrow_type_to_ctype(pa.float64()) == 'double'
    
    def test_float32(self):
        from RDataFrameDSL.dsl_compiler import _arrow_type_to_ctype
        assert _arrow_type_to_ctype(pa.float32()) == 'float'
    
    def test_int32(self):
        from RDataFrameDSL.dsl_compiler import _arrow_type_to_ctype
        assert _arrow_type_to_ctype(pa.int32()) == 'int'
    
    def test_int64(self):
        from RDataFrameDSL.dsl_compiler import _arrow_type_to_ctype
        assert _arrow_type_to_ctype(pa.int64()) == 'long'
    
    def test_bool(self):
        from RDataFrameDSL.dsl_compiler import _arrow_type_to_ctype
        assert _arrow_type_to_ctype(pa.bool_()) == 'bool'
    
    def test_list_float(self):
        from RDataFrameDSL.dsl_compiler import _arrow_type_to_ctype
        assert _arrow_type_to_ctype(pa.list_(pa.float32())) == 'RVec<float>'
    
    def test_list_double(self):
        from RDataFrameDSL.dsl_compiler import _arrow_type_to_ctype
        assert _arrow_type_to_ctype(pa.list_(pa.float64())) == 'RVec<double>'
    
    def test_unknown_fallback(self):
        from RDataFrameDSL.dsl_compiler import _arrow_type_to_ctype
        # Unknown type should fallback to double
        assert _arrow_type_to_ctype(pa.string()) == 'double'
