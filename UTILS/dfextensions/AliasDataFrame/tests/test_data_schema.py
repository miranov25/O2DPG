"""
Test Phase 5 Week 1: Data description, schema management, dtype conversion.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from ..AliasDataFrame import AliasDataFrame


@pytest.fixture
def sample_adf():
    """Create sample AliasDataFrame with subframes for testing."""
    df = pd.DataFrame({
        'x': np.random.randn(100).astype(np.float32),
        'y': np.random.randn(100).astype(np.float16),
        'z': np.random.randn(100).astype(np.float64),
        'id': np.arange(100, dtype=np.int32),
    })
    
    adf = AliasDataFrame(df)
    adf.add_alias('r', 'sqrt(x**2 + y**2)', dtype=np.float32)
    adf.add_alias('theta', 'atan2(y, x)', dtype=np.float32)
    
    # Add subframe
    sub_df = pd.DataFrame({'id': [0, 1], 'calib': [0.1, 0.2]})
    sub_adf = AliasDataFrame(sub_df)
    adf.register_subframe('calibration', sub_adf, index_columns=['id'])
    adf.add_alias('cal', 'calibration.calib', dtype=np.float32)
    
    return adf


class TestSelectData:
    """Test select_data functionality."""
    
    def test_select_all(self, sample_adf):
        """Test selecting all columns."""
        result = sample_adf.select_data()
        assert len(result) == 7
        assert set(result) == {'x', 'y', 'z', 'id', 'r', 'theta', 'cal'}
    
    def test_select_by_dtype(self, sample_adf):
        """Test filtering by dtype."""
        result = sample_adf.select_data(dtype=np.float32)
        assert set(result) == {'x', 'r', 'theta', 'cal'}
    
    def test_select_multiple_dtypes(self, sample_adf):
        """Test filtering by multiple dtypes."""
        result = sample_adf.select_data(dtype=[np.float32, np.float16])
        assert set(result) == {'x', 'y', 'r', 'theta', 'cal'}
    
    def test_select_only_physical(self, sample_adf):
        """Test selecting only physical columns."""
        result = sample_adf.select_data(only_physical=True)
        assert set(result) == {'x', 'y', 'z', 'id'}
    
    def test_select_only_aliases(self, sample_adf):
        """Test selecting only alias columns."""
        result = sample_adf.select_data(only_aliases=True)
        assert set(result) == {'r', 'theta', 'cal'}
    
    def test_select_by_pattern(self, sample_adf):
        """Test filtering by regex pattern."""
        result = sample_adf.select_data(pattern=r'[xy]')
        assert set(result) == {'x', 'y'}
    
    def test_mutually_exclusive_filters(self, sample_adf):
        """Test that physical/aliases filters are mutually exclusive."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            sample_adf.select_data(only_physical=True, only_aliases=True)


class TestDescribeData:
    """Test describe_data functionality."""
    
    def test_describe_core_only(self, sample_adf):
        """Test describe with core info only."""
        result = sample_adf.describe_data(
            verbosity=sample_adf.DATA_SHOW_CORE,
            as_dict=True
        )
        assert len(result) == 7
        assert all('dtype' in info for info in result.values())
        assert all('memory_bytes' in info for info in result.values())
    
    def test_describe_with_source(self, sample_adf):
        """Test describe with source information."""
        result = sample_adf.describe_data(
            verbosity=sample_adf.DATA_SHOW_CORE | sample_adf.DATA_SHOW_SOURCE,
            as_dict=True
        )
        assert result['x']['source'] == 'physical'
        assert result['r']['source'] == 'alias'
        assert result['cal']['source'] == 'subframe'
    
    def test_describe_sort_by_memory(self, sample_adf):
        """Test sorting by memory."""
        result = sample_adf.describe_data(
            verbosity=sample_adf.DATA_SHOW_CORE,
            sort_by='memory',
            as_dict=True
        )
        # Just verify it doesn't crash
        assert len(result) == 7


class TestDescribeSchema:
    """Test describe_schema functionality."""
    
    def test_describe_schema_overview(self, sample_adf):
        """Test schema overview."""
        result = sample_adf.describe_schema(as_dict=True)
        assert 'columns' in result
        assert result['columns']['total'] == 3  # Only aliases in schema
        assert 'subframes' in result
    
    def test_describe_schema_with_verbosity(self, sample_adf):
        """Test schema description with different verbosity."""
        result = sample_adf.describe_schema(
            verbosity=sample_adf.SCHEMA_SHOW_CORE,
            as_dict=True
        )
        assert 'columns' in result


class TestSelectSchema:
    """Test select_schema functionality."""
    
    def test_select_by_dtype(self, sample_adf):
        """Test selecting schema entries by dtype."""
        result = sample_adf.select_schema(dtype=np.float32)
        assert set(result) == {'r', 'theta', 'cal'}
    
    def test_select_subframe_refs(self, sample_adf):
        """Test selecting subframe references."""
        result = sample_adf.select_schema(is_subframe_ref=True)
        assert result == ['cal']


class TestSchemaSerialization:
    """Test schema export/save/load."""
    
    def test_export_schema(self, sample_adf):
        """Test exporting schema."""
        schema = sample_adf.export_schema()
        assert 'columns' in schema
        assert 'compression' in schema
        assert 'subframes' in schema
        
        # Verify physical columns added
        assert 'x' in schema['columns']
        assert 'y' in schema['columns']
        
        # Verify dtypes are strings
        for name, info in schema['columns'].items():
            dtype = info.get('dtype')
            if dtype:
                assert isinstance(dtype, str), f"dtype for {name} not string: {type(dtype)}"
    
    def test_save_load_roundtrip(self, sample_adf):
        """Test save and load schema."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            schema_path = f.name
        
        try:
            sample_adf.save_schema(schema_path)
            loaded_schema = AliasDataFrame.load_schema(schema_path)
            
            # Verify structure preserved
            assert 'columns' in loaded_schema
            assert len(loaded_schema['columns']) > 0
        finally:
            if os.path.exists(schema_path):
                os.unlink(schema_path)


class TestApplySchema:
    """Test applying schema to new data."""
    
    def test_apply_schema_dtypes(self, sample_adf):
        """Test that dtypes are applied from schema."""
        schema = sample_adf.export_schema()
        
        # Create new DataFrame with different dtypes
        df2 = pd.DataFrame({
            'x': np.random.randn(50).astype(np.float64),
            'y': np.random.randn(50).astype(np.float32),
            'z': np.random.randn(50).astype(np.float32),
            'id': np.arange(50, dtype=np.int64),
        })
        
        adf2 = AliasDataFrame(df2)
        adf2.apply_schema(schema, warn_missing=False)
        
        # Verify dtypes applied
        assert adf2.df['x'].dtype == np.float32
        assert adf2.df['y'].dtype == np.float16
    
    def test_apply_schema_aliases(self, sample_adf):
        """Test that aliases are applied from schema."""
        schema = sample_adf.export_schema()
        
        df2 = pd.DataFrame({
            'x': np.random.randn(50).astype(np.float32),
            'y': np.random.randn(50).astype(np.float16),
            'z': np.random.randn(50).astype(np.float64),
            'id': np.arange(50, dtype=np.int32),
        })
        
        adf2 = AliasDataFrame(df2)
        adf2.apply_schema(schema, warn_missing=False)
        
        # Verify aliases applied
        assert 'r' in adf2.aliases
        assert 'theta' in adf2.aliases


class TestFromSchema:
    """Test creating AliasDataFrame from schema."""
    
    def test_from_schema_creates_empty_df(self, sample_adf):
        """Test from_schema creates empty DataFrame with correct structure."""
        schema = sample_adf.export_schema()
        adf_empty = AliasDataFrame.from_schema(schema)
        
        assert len(adf_empty.df) == 0
        assert 'x' in adf_empty.df.columns
        assert 'r' in adf_empty.aliases


class TestConvertDtypes:
    """Test dtype conversion."""
    
    def test_convert_dtypes_batch(self):
        """Test batch dtype conversion."""
        df = pd.DataFrame({
            'x': np.random.randn(100).astype(np.float32),
            'y': np.random.randn(100).astype(np.float32),
            'z': np.random.randn(100).astype(np.float32),
        })
        
        adf = AliasDataFrame(df)
        adf.convert_dtypes({'x': np.float16, 'y': np.float16})
        
        assert adf.df['x'].dtype == np.float16
        assert adf.df['y'].dtype == np.float16
        assert adf.df['z'].dtype == np.float32
    
    def test_convert_dtypes_pattern(self):
        """Test pattern-based dtype conversion."""
        df = pd.DataFrame({
            'dy_1': np.random.randn(100).astype(np.float32),
            'dy_2': np.random.randn(100).astype(np.float32),
            'dz_1': np.random.randn(100).astype(np.float32),
        })
        
        adf = AliasDataFrame(df)
        adf.convert_dtypes_pattern(r'dy.*', np.float16)
        
        assert adf.df['dy_1'].dtype == np.float16
        assert adf.df['dy_2'].dtype == np.float16
        assert adf.df['dz_1'].dtype == np.float32


class TestSubframeAliasVsMaterialized:
    """Test that subframe aliases produce same results as materialized joins."""
    
    def test_subframe_alias_matches_materialized(self):
        """Verify subframe alias gives same results as materialized join."""
        # Create main data
        main_df = pd.DataFrame({
            'id': [0, 1, 2, 0, 1, 2],
            'value': [10, 20, 30, 40, 50, 60]
        })
        
        # Create subframe (calibration data)
        sub_df = pd.DataFrame({
            'id': [0, 1, 2],
            'calib': [0.1, 0.2, 0.3]
        })
        
        # Method 1: Materialized join (traditional pandas)
        materialized = main_df.merge(sub_df, on='id', how='left')
        expected_result = materialized['calib'].values
        
        # Method 2: Subframe alias (AliasDataFrame)
        adf = AliasDataFrame(main_df.copy())
        sub_adf = AliasDataFrame(sub_df)
        adf.register_subframe('calibration', sub_adf, index_columns=['id'])
        adf.add_alias('calib', 'calibration.calib', dtype=np.float64)
        adf.materialize_alias('calib')
        alias_result = adf.df['calib'].values
        
        # Compare results
        np.testing.assert_array_equal(alias_result, expected_result)
    
    def test_subframe_alias_handles_missing_keys(self):
        """Verify subframe alias handles missing keys with NaN."""
        # Main data has id=3 which doesn't exist in subframe
        main_df = pd.DataFrame({
            'id': [0, 1, 2, 3],
            'value': [10, 20, 30, 40]
        })
        
        sub_df = pd.DataFrame({
            'id': [0, 1, 2],
            'calib': [0.1, 0.2, 0.3]
        })
        
        # Pandas join
        materialized = main_df.merge(sub_df, on='id', how='left')
        expected = materialized['calib'].values
        
        # AliasDataFrame
        adf = AliasDataFrame(main_df.copy())
        sub_adf = AliasDataFrame(sub_df)
        adf.register_subframe('calibration', sub_adf, index_columns=['id'])
        adf.add_alias('calib', 'calibration.calib', dtype=np.float64)
        adf.materialize_alias('calib')
        result = adf.df['calib'].values
        
        # Compare (including NaN)
        np.testing.assert_array_equal(result, expected)
        assert np.isnan(result[3])  # id=3 should be NaN
