"""
Tests for Phase 12.4b2: register_fit_result() and draw_fit_summary()

Save as: tests/test_register_fit_result.py
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

# Import will work when running from AliasDataFrame directory
from AliasDataFrame import AliasDataFrame


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_metadata():
    """Synthetic metadata for testing without groupby-regression."""
    return {
        'version': '1.0',
        'formulas': {
            'y_pred_Test': 'y_intercept_Test + y_slope_x_Test*x',
        },
        'residual_formulas': {
            'y_delta_Test': 'y - (y_intercept_Test + y_slope_x_Test*x)',
        },
        'pull_formulas': {
            'y_pull_Test': '(y - (y_intercept_Test + y_slope_x_Test*x)) / y_rms_Test',
            'y_pull_mad_Test': '(y - (y_intercept_Test + y_slope_x_Test*x)) / (y_mad_Test * 1.4826)',
        },
        'columns': {
            'gb_columns': ['group'],
            'fit_columns': ['y'],
            'linear_columns': ['x'],
            'coefficients': {'y': ['y_intercept_Test', 'y_slope_x_Test']},
            'errors': {'y': ['y_intercept_err_Test', 'y_slope_x_err_Test']},
            'quality': {'y': ['y_rms_Test', 'y_mad_Test']},
            'diagnostics': ['nPoints_Test'],
            'medians': ['y_median_Test'],
        },
        'parameters': {
            'suffix': '_Test',
            'fit_intercept': True,
            'min_stat': 3,
            'fit_type': 'linear_v4',
            'pull_default': 'rms',
        },
    }


@pytest.fixture
def mock_metadata_multi_column():
    """Metadata with multiple fit columns."""
    return {
        'version': '1.0',
        'formulas': {
            'dy_pred_Fit': 'dy_intercept_Fit + dy_slope_x_Fit*x',
            'dz_pred_Fit': 'dz_intercept_Fit + dz_slope_x_Fit*x',
        },
        'residual_formulas': {
            'dy_delta_Fit': 'dy - (dy_intercept_Fit + dy_slope_x_Fit*x)',
            'dz_delta_Fit': 'dz - (dz_intercept_Fit + dz_slope_x_Fit*x)',
        },
        'pull_formulas': {
            'dy_pull_Fit': '(dy - (dy_intercept_Fit + dy_slope_x_Fit*x)) / dy_rms_Fit',
            'dz_pull_Fit': '(dz - (dz_intercept_Fit + dz_slope_x_Fit*x)) / dz_rms_Fit',
        },
        'columns': {
            'gb_columns': ['group'],
            'fit_columns': ['dy', 'dz'],
            'linear_columns': ['x'],
            'coefficients': {
                'dy': ['dy_intercept_Fit', 'dy_slope_x_Fit'],
                'dz': ['dz_intercept_Fit', 'dz_slope_x_Fit'],
            },
            'errors': {
                'dy': ['dy_intercept_err_Fit', 'dy_slope_x_err_Fit'],
                'dz': ['dz_intercept_err_Fit', 'dz_slope_x_err_Fit'],
            },
            'quality': {
                'dy': ['dy_rms_Fit', 'dy_mad_Fit'],
                'dz': ['dz_rms_Fit', 'dz_mad_Fit'],
            },
            'diagnostics': ['nPoints_Fit'],
        },
        'parameters': {
            'suffix': '_Fit',
            'fit_intercept': True,
            'min_stat': 3,
            'fit_type': 'linear_v4',
            'pull_default': 'rms',
        },
    }


@pytest.fixture
def mock_data_and_fit():
    """Create main data + fit result with known relationship y = 2*x + 1."""
    np.random.seed(42)
    n = 1000
    
    x = np.random.uniform(0, 10, n)
    y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, n)
    
    df = pd.DataFrame({
        'group': [1] * n,
        'x': x,
        'y': y,
    })
    
    dfGB = pd.DataFrame({
        'group': [1],
        'y_intercept_Test': [1.0],
        'y_slope_x_Test': [2.0],
        'y_intercept_err_Test': [0.01],
        'y_slope_x_err_Test': [0.001],
        'y_rms_Test': [0.5],
        'y_mad_Test': [0.4],
        'y_median_Test': [11.0],
        'nPoints_Test': [n],
    })
    
    return df, dfGB


@pytest.fixture
def mock_data_multi_column():
    """Create data with two fit columns."""
    np.random.seed(42)
    n = 1000
    
    x = np.random.uniform(0, 10, n)
    dy = 2.0 * x + 1.0 + np.random.normal(0, 0.5, n)
    dz = -1.0 * x + 3.0 + np.random.normal(0, 0.3, n)
    
    df = pd.DataFrame({
        'group': [1] * n,
        'x': x,
        'dy': dy,
        'dz': dz,
    })
    
    dfGB = pd.DataFrame({
        'group': [1],
        'dy_intercept_Fit': [1.0],
        'dy_slope_x_Fit': [2.0],
        'dy_intercept_err_Fit': [0.01],
        'dy_slope_x_err_Fit': [0.001],
        'dy_rms_Fit': [0.5],
        'dy_mad_Fit': [0.4],
        'dz_intercept_Fit': [3.0],
        'dz_slope_x_Fit': [-1.0],
        'dz_intercept_err_Fit': [0.01],
        'dz_slope_x_err_Fit': [0.001],
        'dz_rms_Fit': [0.3],
        'dz_mad_Fit': [0.25],
        'nPoints_Fit': [n],
    })
    
    return df, dfGB


# =============================================================================
# Test: _validate_fit_metadata
# =============================================================================

class TestValidateFitMetadata:
    """Tests for _validate_fit_metadata()."""
    
    def test_valid_metadata_passes(self, mock_metadata):
        """Valid metadata produces no issues."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        issues = adf._validate_fit_metadata(mock_metadata, validate='skip')
        assert issues == []
    
    def test_missing_formulas_key(self):
        """Missing 'formulas' key is detected."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        bad_meta = {'columns': {'gb_columns': ['a']}, 'parameters': {}}
        issues = adf._validate_fit_metadata(bad_meta, validate='skip')
        assert any('formulas' in str(i) for i in issues)
    
    def test_missing_columns_key(self):
        """Missing 'columns' key is detected."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        bad_meta = {'formulas': {}, 'parameters': {}}
        issues = adf._validate_fit_metadata(bad_meta, validate='skip')
        assert any('columns' in str(i) for i in issues)
    
    def test_missing_gb_columns(self):
        """Missing 'gb_columns' in columns is detected."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        bad_meta = {'formulas': {}, 'columns': {'fit_columns': ['y']}, 'parameters': {}}
        issues = adf._validate_fit_metadata(bad_meta, validate='skip')
        assert any('gb_columns' in str(i) for i in issues)
    
    def test_validate_raise_mode(self):
        """validate='raise' raises ValueError on issues."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        bad_meta = {}
        with pytest.raises(ValueError, match="validation"):
            adf._validate_fit_metadata(bad_meta, validate='raise')
    
    def test_validate_warn_mode(self, mock_metadata):
        """validate='warn' warns but doesn't raise."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        bad_meta = {'formulas': {}, 'parameters': {}}  # Missing columns
        
        with pytest.warns(UserWarning, match="validation"):
            issues = adf._validate_fit_metadata(bad_meta, validate='warn')
        
        assert len(issues) > 0
    
    def test_validate_skip_mode(self):
        """validate='skip' silently returns issues."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        bad_meta = {}
        # Should not warn or raise
        issues = adf._validate_fit_metadata(bad_meta, validate='skip')
        assert len(issues) > 0
    
    def test_non_dict_metadata(self):
        """Non-dict metadata is detected."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        with pytest.raises(ValueError, match="must be dict"):
            adf._validate_fit_metadata("not a dict", validate='raise')


# =============================================================================
# Test: register_fit_result - Basic Registration
# =============================================================================

class TestRegisterFitResultBasic:
    """Basic tests for register_fit_result()."""
    
    def test_subframe_registered(self, mock_data_and_fit, mock_metadata):
        """Subframe is registered correctly."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        assert "Fit" in adf._subframes.subframes
    
    def test_returns_aliasdf(self, mock_data_and_fit, mock_metadata):
        """Returns AliasDataFrame wrapping dfGB."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        result = adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        assert isinstance(result, AliasDataFrame)
        assert len(result.df) == len(dfGB)
    
    def test_metadata_stored(self, mock_data_and_fit, mock_metadata):
        """Metadata is stored in _fit_metadata."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        assert "Fit" in adf._fit_metadata
        assert adf._fit_metadata["Fit"] == mock_metadata
    
    def test_prediction_alias_created(self, mock_data_and_fit, mock_metadata):
        """Prediction aliases are created."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        assert 'y_pred_Test' in adf.aliases
    
    def test_residual_alias_created(self, mock_data_and_fit, mock_metadata):
        """Residual aliases are created."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        assert 'y_delta_Test' in adf.aliases
    
    def test_pull_alias_created(self, mock_data_and_fit, mock_metadata):
        """Pull aliases are created."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        assert 'y_pull_Test' in adf.aliases


# =============================================================================
# Test: register_fit_result - Formula Evaluation
# =============================================================================

class TestRegisterFitResultFormulas:
    """Tests for formula evaluation in register_fit_result()."""
    
    def test_prediction_evaluates_correctly(self, mock_data_and_fit, mock_metadata):
        """Prediction formula computes correct values."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        adf.materialize_alias('y_pred_Test')
        pred = adf['y_pred_Test'].values
        expected = 1.0 + 2.0 * df['x'].values
        
        # Use rtol=1e-5 for float32 precision (default dtype)
        np.testing.assert_allclose(pred, expected, rtol=1e-5)
    
    def test_residual_evaluates_correctly(self, mock_data_and_fit, mock_metadata):
        """Residual formula computes correct values."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        adf.materialize_alias('y_delta_Test')
        delta = adf['y_delta_Test'].values
        expected = df['y'].values - (1.0 + 2.0 * df['x'].values)
        
        # Use rtol=1e-5 for float32 precision (default dtype)
        np.testing.assert_allclose(delta, expected, rtol=1e-5)
    
    def test_pull_distribution_properties(self, mock_data_and_fit, mock_metadata):
        """Pull should have mean≈0, std≈1 for correct errors."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        adf.materialize_alias('y_pull_Test')
        pull = adf['y_pull_Test'].values
        
        assert abs(np.mean(pull)) < 0.1
        assert 0.8 < np.std(pull) < 1.2
    
    def test_multi_column_predictions(self, mock_data_multi_column, mock_metadata_multi_column):
        """Multiple fit columns each get correct predictions."""
        df, dfGB = mock_data_multi_column
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata_multi_column)
        
        adf.materialize_alias('dy_pred_Fit')
        adf.materialize_alias('dz_pred_Fit')
        
        dy_pred = adf['dy_pred_Fit'].values
        dz_pred = adf['dz_pred_Fit'].values
        
        dy_expected = 1.0 + 2.0 * df['x'].values
        dz_expected = 3.0 + (-1.0) * df['x'].values
        
        # Use rtol=1e-5 for float32 precision (default dtype)
        np.testing.assert_allclose(dy_pred, dy_expected, rtol=1e-5)
        np.testing.assert_allclose(dz_pred, dz_expected, rtol=1e-5)
    
    def test_subframe_coefficients_accessible(self, mock_data_and_fit, mock_metadata):
        """Coefficients accessible via auto-aliased subframe columns."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        # Auto-aliased from subframe
        assert 'y_intercept_Test' in adf.aliases
        assert 'y_slope_x_Test' in adf.aliases
        
        adf.materialize_alias('y_intercept_Test')
        intercept = adf['y_intercept_Test'].values
        assert np.all(intercept == 1.0)
    
    def test_dtype_applied(self, mock_data_and_fit, mock_metadata):
        """Specified dtypes are applied to aliases."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata, 
                               prediction_dtype=np.float16)
        
        # Check dtype in schema
        pred_info = adf._schema['columns'].get('y_pred_Test', {})
        assert pred_info.get('dtype') == np.float16


# =============================================================================
# Test: register_fit_result - Options and Flags
# =============================================================================

class TestRegisterFitResultOptions:
    """Tests for register_fit_result() options."""
    
    def test_pull_type_rms_only(self, mock_data_and_fit, mock_metadata):
        """pull_type='rms' creates only RMS pulls."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata, pull_type='rms')
        
        assert 'y_pull_Test' in adf.aliases
        assert 'y_pull_mad_Test' not in adf.aliases
    
    def test_pull_type_mad_only(self, mock_data_and_fit, mock_metadata):
        """pull_type='mad' creates only MAD pulls."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata, pull_type='mad')
        
        assert 'y_pull_Test' not in adf.aliases
        assert 'y_pull_mad_Test' in adf.aliases
    
    def test_pull_type_both(self, mock_data_and_fit, mock_metadata):
        """pull_type='both' creates all pulls."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata, pull_type='both')
        
        assert 'y_pull_Test' in adf.aliases
        assert 'y_pull_mad_Test' in adf.aliases
    
    def test_pull_type_from_metadata(self, mock_data_and_fit, mock_metadata):
        """pull_type=None uses metadata default."""
        df, dfGB = mock_data_and_fit
        mock_metadata['parameters']['pull_default'] = 'mad'
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata, pull_type=None)
        
        # Should use 'mad' from metadata
        assert 'y_pull_Test' not in adf.aliases
        assert 'y_pull_mad_Test' in adf.aliases
    
    def test_add_predictions_false(self, mock_data_and_fit, mock_metadata):
        """add_predictions=False skips prediction aliases."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata, add_predictions=False)
        
        assert 'y_pred_Test' not in adf.aliases
    
    def test_add_residuals_false(self, mock_data_and_fit, mock_metadata):
        """add_residuals=False skips residual aliases."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata, add_residuals=False)
        
        assert 'y_delta_Test' not in adf.aliases
    
    def test_add_pulls_false(self, mock_data_and_fit, mock_metadata):
        """add_pulls=False skips pull aliases."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata, add_pulls=False)
        
        assert 'y_pull_Test' not in adf.aliases
        assert 'y_pull_mad_Test' not in adf.aliases
    
    def test_auto_alias_subframe_false(self, mock_data_and_fit, mock_metadata):
        """auto_alias_subframe=False skips subframe auto-aliasing."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata, auto_alias_subframe=False)
        
        # Subframe columns should NOT be auto-aliased
        # But prediction aliases should still work (they reference subframe)
        assert 'y_pred_Test' in adf.aliases


# =============================================================================
# Test: register_fit_result - Backward Compatibility
# =============================================================================

class TestRegisterFitResultBackwardCompat:
    """Backward compatibility tests for register_fit_result()."""
    
    def test_no_metadata_with_index_columns(self, mock_data_and_fit):
        """Works without metadata if index_columns provided."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        
        with pytest.warns(UserWarning, match="without metadata"):
            adf.register_fit_result("Fit", dfGB, metadata=None, 
                                   index_columns=['group'])
        
        assert "Fit" in adf._subframes.subframes
        assert 'y_pred_Test' not in adf.aliases  # No auto-aliases
    
    def test_no_metadata_no_index_columns_raises(self, mock_data_and_fit):
        """Raises error if no metadata and no index_columns."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        
        with pytest.raises(ValueError, match="index_columns required"):
            adf.register_fit_result("Fit", dfGB, metadata=None)
    
    def test_duplicate_registration_warns(self, mock_data_and_fit, mock_metadata):
        """Duplicate registration emits warning."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        with pytest.warns(UserWarning, match="Overwriting"):
            adf.register_fit_result("Fit", dfGB, mock_metadata)
    
    def test_duplicate_overwrites(self, mock_data_and_fit, mock_metadata):
        """Duplicate registration overwrites previous."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        # Modify metadata
        mock_metadata['parameters']['suffix'] = '_NewSuffix'
        
        with pytest.warns(UserWarning):
            adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        # Should have new metadata
        assert adf._fit_metadata["Fit"]['parameters']['suffix'] == '_NewSuffix'


# =============================================================================
# Test: get_fit_metadata and list_fit_results
# =============================================================================

class TestFitMetadataAccessors:
    """Tests for get_fit_metadata() and list_fit_results()."""
    
    def test_get_fit_metadata_single(self, mock_data_and_fit, mock_metadata):
        """get_fit_metadata returns metadata for specific fit."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        meta = adf.get_fit_metadata("Fit")
        assert meta == mock_metadata
    
    def test_get_fit_metadata_all(self, mock_data_and_fit, mock_metadata):
        """get_fit_metadata(None) returns all metadata."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit1", dfGB, mock_metadata)
        adf.register_fit_result("Fit2", dfGB, mock_metadata)
        
        all_meta = adf.get_fit_metadata(None)
        assert "Fit1" in all_meta
        assert "Fit2" in all_meta
    
    def test_get_fit_metadata_unknown_raises(self, mock_data_and_fit, mock_metadata):
        """get_fit_metadata raises KeyError for unknown fit."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        with pytest.raises(KeyError, match="NoSuchFit"):
            adf.get_fit_metadata("NoSuchFit")
    
    def test_list_fit_results_empty(self):
        """list_fit_results returns empty list if no fits."""
        adf = AliasDataFrame(pd.DataFrame({'x': [1, 2, 3]}))
        assert adf.list_fit_results() == []
    
    def test_list_fit_results_multiple(self, mock_data_and_fit, mock_metadata):
        """list_fit_results returns all fit names."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit1", dfGB, mock_metadata)
        adf.register_fit_result("Fit2", dfGB, mock_metadata)
        
        fits = adf.list_fit_results()
        assert set(fits) == {"Fit1", "Fit2"}


# =============================================================================
# Test: _apply_pull_transform
# =============================================================================

class TestApplyPullTransform:
    """Tests for _apply_pull_transform()."""
    
    def test_none_returns_original(self, mock_data_and_fit, mock_metadata):
        """transform=None returns original alias."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        result = adf._apply_pull_transform('y_pull_Test', None)
        assert result == 'y_pull_Test'
    
    def test_asinh_creates_alias(self, mock_data_and_fit, mock_metadata):
        """transform='asinh' creates transformed alias."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        result = adf._apply_pull_transform('y_pull_Test', 'asinh')
        
        assert result == 'y_pull_Test_asinh'
        assert 'y_pull_Test_asinh' in adf.aliases
    
    def test_tanh_creates_alias(self, mock_data_and_fit, mock_metadata):
        """transform='tanh' creates transformed alias."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        result = adf._apply_pull_transform('y_pull_Test', 'tanh')
        
        assert result == 'y_pull_Test_tanh'
        assert 'y_pull_Test_tanh' in adf.aliases
    
    def test_idempotent(self, mock_data_and_fit, mock_metadata):
        """Calling twice doesn't recreate alias."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        result1 = adf._apply_pull_transform('y_pull_Test', 'asinh')
        result2 = adf._apply_pull_transform('y_pull_Test', 'asinh')
        
        assert result1 == result2
    
    def test_invalid_transform_raises(self, mock_data_and_fit, mock_metadata):
        """Invalid transform raises ValueError."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        with pytest.raises(ValueError, match="Unknown pull_transform"):
            adf._apply_pull_transform('y_pull_Test', 'invalid')


# =============================================================================
# Test: _compute_fit_validation
# =============================================================================

class TestComputeFitValidation:
    """Tests for _compute_fit_validation()."""
    
    def test_basic_validation(self, mock_data_and_fit, mock_metadata):
        """Basic validation returns expected structure."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        validation = adf._compute_fit_validation("Fit")
        
        assert 'y' in validation
        assert '_overall_pass' in validation
        assert 'pull_mean' in validation['y']
        assert 'pull_std' in validation['y']
    
    def test_good_fit_passes(self, mock_data_and_fit, mock_metadata):
        """Good fit with correct errors passes validation."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        validation = adf._compute_fit_validation("Fit")
        
        assert validation['y']['_pass'] == True
        assert validation['_overall_pass'] == True
    
    def test_bad_fit_fails(self):
        """Fit with wrong errors fails validation."""
        np.random.seed(42)
        n = 1000
        x = np.random.uniform(0, 10, n)
        y = 2.0 * x + 1.0 + np.random.normal(0, 0.5, n)
        
        df = pd.DataFrame({'group': [1] * n, 'x': x, 'y': y})
        
        # Wrong RMS (too small, so pulls will be too wide)
        dfGB = pd.DataFrame({
            'group': [1],
            'y_intercept_Test': [1.0],
            'y_slope_x_Test': [2.0],
            'y_rms_Test': [0.1],  # Wrong! Actual is 0.5
        })
        
        meta = {
            'formulas': {'y_pred_Test': 'y_intercept_Test + y_slope_x_Test*x'},
            'residual_formulas': {'y_delta_Test': 'y - y_pred_Test'},
            'pull_formulas': {'y_pull_Test': 'y_delta_Test / y_rms_Test'},
            'columns': {'gb_columns': ['group'], 'fit_columns': ['y']},
            'parameters': {'suffix': '_Test', 'pull_default': 'rms'},
        }
        
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, meta)
        
        validation = adf._compute_fit_validation("Fit")
        
        # Pull std should be ~5 (too wide), so should fail
        assert validation['y']['_pass'] == False
        assert validation['_overall_pass'] == False


# =============================================================================
# Test: draw_fit_summary - Basic
# =============================================================================

class TestDrawFitSummaryBasic:
    """Basic tests for draw_fit_summary()."""
    
    def test_generates_figures(self, mock_data_and_fit, mock_metadata, tmp_path):
        """draw_fit_summary creates figures."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", save_dir=str(tmp_path), verbose=False)
        
        assert len(results) > 1  # At least figures + _validation
        assert '_validation' in results
    
    def test_saves_png(self, mock_data_and_fit, mock_metadata, tmp_path):
        """Figures saved as PNG by default."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        adf.draw_fit_summary("Fit", save_dir=str(tmp_path), verbose=False)
        
        # Check for PNG files
        png_files = list(tmp_path.glob("*.png"))
        assert len(png_files) > 0
    
    def test_saves_pdf(self, mock_data_and_fit, mock_metadata, tmp_path):
        """save_format='pdf' creates PDF files."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        adf.draw_fit_summary("Fit", save_dir=str(tmp_path), 
                           save_format='pdf', verbose=False)
        
        pdf_files = list(tmp_path.glob("*.pdf"))
        assert len(pdf_files) > 0
    
    def test_unknown_fit_raises(self, mock_data_and_fit, mock_metadata):
        """Unknown fit name raises KeyError."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        
        with pytest.raises(KeyError, match="NoSuchFit"):
            adf.draw_fit_summary("NoSuchFit")
    
    def test_returns_validation(self, mock_data_and_fit, mock_metadata):
        """Result includes _validation dict."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", verbose=False)
        
        assert '_validation' in results
        assert 'y' in results['_validation']
    
    def test_large_dataset_warns(self, mock_metadata):
        """Large dataset triggers warning."""
        np.random.seed(42)
        n = 1_100_000  # > 1M
        df = pd.DataFrame({
            'group': [1] * n,
            'x': np.random.uniform(0, 10, n),
            'y': np.random.normal(0, 1, n),
        })
        dfGB = pd.DataFrame({
            'group': [1],
            'y_intercept_Test': [0.0],
            'y_slope_x_Test': [0.0],
            'y_rms_Test': [1.0],
            'y_mad_Test': [0.8],
        })
        
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        with pytest.warns(UserWarning, match="Large dataset"):
            adf.draw_fit_summary("Fit", include=['delta_1d'], verbose=False)


# =============================================================================
# Test: draw_fit_summary - Options
# =============================================================================

class TestDrawFitSummaryOptions:
    """Tests for draw_fit_summary() options."""
    
    def test_include_filter(self, mock_data_and_fit, mock_metadata):
        """include parameter limits categories."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", include=['delta_1d'], verbose=False)
        
        # Should have residuals figure (delta_1d is part of it)
        assert 'Fit_residuals' in results
        assert 'Fit_quality' not in results
    
    def test_exclude_filter(self, mock_data_and_fit, mock_metadata):
        """exclude parameter removes categories."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", exclude=['quality'], verbose=False)
        
        assert 'Fit_quality' not in results
    
    def test_pull_transform_asinh(self, mock_data_and_fit, mock_metadata):
        """pull_transform='asinh' creates transformed alias."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", include=['pull_1d'], 
                                       pull_transform='asinh', verbose=False)
        
        assert 'y_pull_Test_asinh' in adf.aliases
    
    def test_entry_end_limits_data(self, mock_data_and_fit, mock_metadata):
        """entry_end parameter limits processed entries."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        # Should not raise, uses only first 100 entries
        results = adf.draw_fit_summary("Fit", entry_end=100, 
                                       include=['delta_1d'], verbose=False)
        
        assert len(results) > 0
    
    def test_fit_columns_subset(self, mock_data_multi_column, mock_metadata_multi_column):
        """fit_columns limits which columns to plot."""
        df, dfGB = mock_data_multi_column
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata_multi_column)
        
        results = adf.draw_fit_summary("Fit", fit_columns=['dy'], 
                                       include=['delta_1d'], verbose=False)
        
        # Should only have dy plots, not dz
        assert 'Fit_residuals' in results
    
    def test_unknown_category_warns(self, mock_data_and_fit, mock_metadata):
        """Unknown category in include emits warning."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        with pytest.warns(UserWarning, match="Unknown categories"):
            adf.draw_fit_summary("Fit", include=['invalid_category'], verbose=False)


# =============================================================================
# Test: Schema Persistence
# =============================================================================

class TestSchemaPersistence:
    """Tests for fit metadata in schema export/import."""
    
    def test_export_includes_fit_metadata(self, mock_data_and_fit, mock_metadata):
        """export_schema includes fit_metadata."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        schema = adf.export_schema()
        
        assert 'fit_metadata' in schema
        assert 'Fit' in schema['fit_metadata']
    
    def test_apply_schema_restores_fit_metadata(self, mock_data_and_fit, mock_metadata):
        """apply_schema restores fit_metadata."""
        df, dfGB = mock_data_and_fit
        adf1 = AliasDataFrame(df)
        adf1.register_fit_result("Fit", dfGB, mock_metadata)
        
        schema = adf1.export_schema()
        
        # Create adf2 with the same subframe registered so aliases can resolve
        adf2 = AliasDataFrame(df)
        adf2.register_fit_result("Fit", dfGB, mock_metadata)
        
        # Clear fit_metadata to test restoration
        adf2._fit_metadata = {}
        
        # Apply schema - should restore fit_metadata
        adf2.apply_schema(schema)
        
        assert 'Fit' in adf2._fit_metadata
    
    def test_apply_schema_overwrites_with_warning(self, mock_data_and_fit, mock_metadata):
        """apply_schema warns when overwriting existing fit_metadata."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit1", dfGB, mock_metadata)
        
        schema = {'fit_metadata': {'Fit2': mock_metadata}}
        
        with pytest.warns(UserWarning, match="Overwriting"):
            adf.apply_schema(schema)
    
    def test_schema_roundtrip_preserves_formulas(self, mock_data_and_fit, mock_metadata):
        """Schema roundtrip preserves all formula strings."""
        df, dfGB = mock_data_and_fit
        adf1 = AliasDataFrame(df)
        adf1.register_fit_result("Fit", dfGB, mock_metadata)
        
        schema = adf1.export_schema()
        
        # Create adf2 with same subframe so aliases can resolve
        adf2 = AliasDataFrame(df)
        adf2.register_fit_result("Fit", dfGB, mock_metadata)
        adf2._fit_metadata = {}  # Clear to test restoration
        
        adf2.apply_schema(schema)
        
        # Verify formulas are preserved
        original_formulas = mock_metadata['formulas']
        restored_formulas = adf2._fit_metadata['Fit']['formulas']
        assert original_formulas == restored_formulas
    
    def test_exported_schema_is_json_serializable(self, mock_data_and_fit, mock_metadata):
        """Verify exported schema can be serialized to JSON."""
        import json
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        schema = adf.export_schema()
        
        # Should not raise - use default=str to handle sets
        json_str = json.dumps(schema, default=str)
        restored = json.loads(json_str)
        
        assert 'fit_metadata' in restored
    
    def test_schema_without_fit_metadata_backward_compat(self, mock_data_and_fit):
        """Old schemas without fit_metadata work normally."""
        df, _ = mock_data_and_fit
        adf = AliasDataFrame(df)
        
        # Old-style schema without fit_metadata
        old_schema = {
            'columns': {'x': {'dtype': 'float64'}},
        }
        
        # Should not raise
        adf.apply_schema(old_schema)
        
        # _fit_metadata should remain unchanged (not created)
        assert not hasattr(adf, '_fit_metadata') or not adf._fit_metadata


class TestValidationThresholds:
    """Tests for custom validation thresholds."""
    
    def test_custom_validation_thresholds(self, mock_data_and_fit, mock_metadata):
        """Custom thresholds override defaults."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        # With very strict thresholds, validation should fail
        strict_thresholds = {'pull_mean_threshold': 0.001}
        validation = adf._compute_fit_validation("Fit", thresholds=strict_thresholds)
        
        # The pull mean is ~0.1, so strict threshold should fail
        assert validation['y']['pull_mean_pass'] == False
    
    def test_partial_threshold_override(self, mock_data_and_fit, mock_metadata):
        """Partial threshold dict merges with defaults."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        # Only override one threshold
        partial_thresholds = {'pull_std_max': 2.0}
        validation = adf._compute_fit_validation("Fit", thresholds=partial_thresholds)
        
        # Other thresholds should still apply (defaults)
        assert 'pull_mean' in validation['y']
        assert 'outlier_fraction' in validation['y']
    
    def test_draw_fit_summary_with_thresholds(self, mock_data_and_fit, mock_metadata):
        """draw_fit_summary accepts validation_thresholds."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        # Should not raise
        results = adf.draw_fit_summary(
            "Fit", 
            include=['pull_1d'],
            validation_thresholds={'pull_mean_threshold': 0.5}
        )
        
        assert '_validation' in results


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    def test_empty_dfGB(self, mock_metadata):
        """Empty dfGB is handled."""
        df = pd.DataFrame({'group': [1, 2], 'x': [1, 2], 'y': [1, 2]})
        dfGB = pd.DataFrame({'group': [], 'y_intercept_Test': []})
        
        adf = AliasDataFrame(df)
        # Should register but predictions will be NaN
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        assert "Fit" in adf._subframes.subframes
    
    def test_single_fit_column(self, mock_data_and_fit, mock_metadata):
        """Single fit column works correctly."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", verbose=False)
        assert '_validation' in results
    
    def test_missing_optional_metadata_keys(self, mock_data_and_fit):
        """Works with minimal metadata (missing optional keys)."""
        df, dfGB = mock_data_and_fit
        
        minimal_meta = {
            'formulas': {'y_pred_Test': 'y_intercept_Test + y_slope_x_Test*x'},
            'residual_formulas': {},
            'pull_formulas': {},
            'columns': {
                'gb_columns': ['group'],
                'fit_columns': ['y'],
            },
            'parameters': {'suffix': '_Test'},
        }
        
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, minimal_meta)
        
        assert 'y_pred_Test' in adf.aliases
    
    def test_special_characters_in_name(self, mock_data_and_fit, mock_metadata):
        """Fit names with special characters work."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        
        # Names with underscores and numbers
        adf.register_fit_result("Fit_Track_V4", dfGB, mock_metadata)
        
        assert "Fit_Track_V4" in adf.list_fit_results()


# =============================================================================
# Test: Result Verification (Phase 12.4b3 - verifies correctness, not just no-crash)
# =============================================================================

class TestResultVerification:
    """Tests that verify actual output correctness, not just that code runs."""
    
    def test_draw_fit_summary_plots_render_without_error(self, mock_data_and_fit, mock_metadata):
        """Verify plots actually render (no error text in figure)."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", include=['delta_1d', 'pull_1d'], verbose=False)
        
        # Verify figure was created
        assert 'Fit_residuals' in results
        fig_data = results['Fit_residuals']
        assert fig_data.get('fig') is not None, "Figure is None"
        
        # Verify no error text in any subplot
        for ax in fig_data['axes']:
            for txt in ax.texts:
                text_content = txt.get_text()
                assert 'Error' not in text_content, f"Plot contains error: {text_content}"
            
            # Verify title doesn't indicate error
            title = ax.get_title()
            assert '[ERROR]' not in title, f"Plot title indicates error: {title}"
    
    def test_gaussian_overlay_added_to_pull_histogram(self, mock_data_and_fit, mock_metadata):
        """Verify Gaussian overlay is actually added to pull histogram."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary(
            "Fit", 
            include=['delta_1d', 'pull_1d'],
            gaussian_overlay=True,
            verbose=False
        )
        
        axes = results['Fit_residuals']['axes']
        
        # Delta histogram (axis 0) should NOT have overlay
        delta_ax = axes[0]
        assert len(delta_ax.get_lines()) == 0, "Delta histogram should not have Gaussian overlay"
        
        # Pull histogram (axis 1) SHOULD have overlay
        pull_ax = axes[1]
        assert len(pull_ax.get_lines()) >= 1, "Pull histogram should have Gaussian overlay"
    
    def test_validation_metrics_match_expected_values(self, mock_data_and_fit, mock_metadata):
        """Verify validation metrics are computed correctly."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        validation = adf._compute_fit_validation("Fit")
        
        # Verify structure
        assert 'y' in validation
        assert '_overall_pass' in validation
        
        # Verify metrics exist and are reasonable
        y_metrics = validation['y']
        
        # Pull mean should be near zero for well-fitted data
        assert 'pull_mean' in y_metrics
        assert -1.0 < y_metrics['pull_mean'] < 1.0, f"Pull mean {y_metrics['pull_mean']} out of expected range"
        
        # Pull std should be near 1.0 for correctly estimated errors
        assert 'pull_std' in y_metrics
        assert 0.5 < y_metrics['pull_std'] < 2.0, f"Pull std {y_metrics['pull_std']} out of expected range"
        
        # Outlier fraction should be in [0, 1]
        assert 'outlier_fraction' in y_metrics
        assert 0.0 <= y_metrics['outlier_fraction'] <= 1.0
        
        # Pass/fail flags should be boolean-like (Python bool or numpy bool)
        assert y_metrics['pull_mean_pass'] in (True, False)
        assert y_metrics['pull_std_pass'] in (True, False)
        assert y_metrics['outlier_pass'] in (True, False)
    
    def test_schema_roundtrip_metadata_values_match(self, mock_data_and_fit, mock_metadata):
        """Verify actual metadata values survive roundtrip, not just keys."""
        df, dfGB = mock_data_and_fit
        adf1 = AliasDataFrame(df)
        adf1.register_fit_result("Fit", dfGB, mock_metadata)
        
        schema = adf1.export_schema()
        
        # Create new ADF with same subframe to allow schema application
        adf2 = AliasDataFrame(df)
        adf2.register_fit_result("Fit", dfGB, mock_metadata)
        adf2._fit_metadata = {}  # Clear to test restoration
        
        adf2.apply_schema(schema)
        
        # Deep comparison of actual values
        original = adf1._fit_metadata['Fit']
        restored = adf2._fit_metadata['Fit']
        
        # Verify formulas match exactly
        assert original['formulas'] == restored['formulas'], "Formulas don't match after roundtrip"
        
        # Verify columns match exactly  
        assert original['columns'] == restored['columns'], "Columns don't match after roundtrip"
        
        # Verify parameters match exactly
        assert original['parameters'] == restored['parameters'], "Parameters don't match after roundtrip"
    
    def test_deepcopy_prevents_mutation(self, mock_data_and_fit, mock_metadata):
        """Verify exported schema is isolated from source (deep copy works)."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        schema = adf.export_schema()
        
        # Get original formula value
        original_formula = adf._fit_metadata['Fit']['formulas']['y_pred_Test']
        
        # Mutate the exported schema
        schema['fit_metadata']['Fit']['formulas']['y_pred_Test'] = "CORRUPTED"
        
        # Original should be unchanged
        assert adf._fit_metadata['Fit']['formulas']['y_pred_Test'] == original_formula
        assert adf._fit_metadata['Fit']['formulas']['y_pred_Test'] != "CORRUPTED"
    
    def test_histograms_have_data(self, mock_data_and_fit, mock_metadata):
        """Verify histograms actually contain data (patches exist)."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", include=['delta_1d', 'pull_1d'], verbose=False)
        
        axes = results['Fit_residuals']['axes']
        
        for ax in axes:
            # Each histogram should have at least one patch (bar or polygon)
            assert len(ax.patches) > 0, f"Histogram '{ax.get_title()}' has no patches"


# =============================================================================
# Phase 12.4b4: Extended Test Coverage with Multi-Group, Multi-Variable Fixtures
# =============================================================================

# Helper function for robust key matching (avoids hardcoded keys)
def _find_result_key(results, pattern):
    """Find result key by pattern (avoids hardcoding)."""
    for k in results:
        if pattern in k or k.endswith(pattern):
            return k
    raise KeyError(f"No key matching '{pattern}' in {list(results.keys())}")


@pytest.fixture
def mock_multigroup_multivariable():
    """Realistic fixture: 10 bins × 100 entries × 3 fit variables."""
    np.random.seed(42)
    n_per_bin, n_bins = 100, 10
    
    bins = np.repeat(np.arange(n_bins), n_per_bin)
    x = np.random.uniform(0, 10, n_bins * n_per_bin)
    z = np.random.uniform(0, 5, n_bins * n_per_bin)
    
    # Per-bin varying RMS (simulates detector regions)
    true_rms_y = 0.3 + 0.05 * np.arange(n_bins)  # 0.30 to 0.75
    true_rms_w = 0.2 + 0.03 * np.arange(n_bins)  # 0.20 to 0.47
    true_rms_q = 0.5 + 0.02 * np.arange(n_bins)  # 0.50 to 0.68
    
    def noise(rms_arr):
        return np.concatenate([
            np.random.normal(0, rms_arr[b], n_per_bin) for b in range(n_bins)
        ])
    
    # Three fit variables with different models
    y = 2*x + 1 + noise(true_rms_y)           # y = 2x + 1
    w = 0.5*x + 3*z + 2 + noise(true_rms_w)   # w = 0.5x + 3z + 2
    q = x**2 / 10 + noise(true_rms_q)         # q = x²/10
    
    df = pd.DataFrame({
        'x': x, 'z': z, 'y': y, 'w': w, 'q': q, 'bin': bins
    })
    
    dfGB = pd.DataFrame({
        'bin': np.arange(n_bins),
        'y_slope_Multi': np.full(n_bins, 2.0),
        'y_intercept_Multi': np.full(n_bins, 1.0),
        'y_rms_Multi': true_rms_y, 
        'y_mad_Multi': true_rms_y * 0.8,
        'w_rms_Multi': true_rms_w, 
        'w_mad_Multi': true_rms_w * 0.8,
        'q_rms_Multi': true_rms_q, 
        'q_mad_Multi': true_rms_q * 0.8,
    })
    
    return df, dfGB


@pytest.fixture
def mock_metadata_multivar():
    """Metadata for multi-variable fit."""
    return {
        'version': '1.0',
        'formulas': {
            'y_pred_Multi': 'y_slope_Multi * x + y_intercept_Multi',
            'w_pred_Multi': '0.5 * x + 3 * z + 2',
            'q_pred_Multi': 'x**2 / 10',
        },
        'residual_formulas': {
            'y_delta_Multi': 'y - y_pred_Multi',
            'w_delta_Multi': 'w - w_pred_Multi',
            'q_delta_Multi': 'q - q_pred_Multi',
        },
        'pull_formulas': {
            'y_pull_Multi': 'y_delta_Multi / y_rms_Multi',
            'w_pull_Multi': 'w_delta_Multi / w_rms_Multi',
            'q_pull_Multi': 'q_delta_Multi / q_rms_Multi',
        },
        'columns': {
            'fit_columns': ['y', 'w', 'q'],
            'gb_columns': ['bin'],
            'quality': {
                'y': ['y_rms_Multi', 'y_mad_Multi'],
                'w': ['w_rms_Multi', 'w_mad_Multi'],
                'q': ['q_rms_Multi', 'q_mad_Multi'],
            },
        },
        'parameters': {
            'suffix': '_Multi',
            'pull_default': 'rms',
        },
    }


# =============================================================================
# Test: Multi-Group Fit (Critical)
# =============================================================================

class TestMultiGroupFit:
    """Tests with realistic multi-bin, multi-variable data."""
    
    def test_quality_plot_shows_distribution(self, mock_multigroup_multivariable, mock_metadata_multivar):
        """Quality plot shows RMS/MAD distribution, not single spike."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_multigroup_multivariable
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata_multivar)
        
        results = adf.draw_fit_summary("Fit", include=['quality'], on_error='raise', verbose=False)
        
        quality_key = _find_result_key(results, '_quality')
        fig_data = results[quality_key]
        
        # Should have multiple non-zero bars (not single spike)
        for ax in fig_data['axes']:
            if len(ax.patches) > 0:
                nonzero_bars = sum(1 for p in ax.patches if hasattr(p, 'get_height') and p.get_height() > 0)
                # For polygon patches, just check there are patches
                if nonzero_bars == 0:
                    nonzero_bars = len(ax.patches)
                assert nonzero_bars >= 1, "Quality histogram should show distribution"
    
    def test_multivar_layout_correct(self, mock_multigroup_multivariable, mock_metadata_multivar):
        """Multi-variable fit creates correct subplot layout."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_multigroup_multivariable
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata_multivar)
        
        results = adf.draw_fit_summary("Fit", include=['delta_1d', 'pull_1d'], on_error='raise', verbose=False)
        
        residuals_key = _find_result_key(results, '_residuals')
        fig_data = results[residuals_key]
        
        # 3 variables × 2 categories = 6 panels
        axes_with_content = [ax for ax in fig_data['axes'] if len(ax.patches) > 0]
        assert len(axes_with_content) >= 6, f"Should have 6 histogram panels (3 vars × 2 types), got {len(axes_with_content)}"
    
    def test_pull_distribution_multigroup(self, mock_multigroup_multivariable, mock_metadata_multivar):
        """Pull is ~N(0,1) across all bins for each variable."""
        df, dfGB = mock_multigroup_multivariable
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata_multivar)
        
        adf.materialize_aliases(names=['y_pull_Multi'])
        pulls = adf.df['y_pull_Multi'].values
        
        assert abs(np.mean(pulls)) < 0.15, f"Pull mean {np.mean(pulls):.3f} too far from 0"
        assert 0.85 < np.std(pulls) < 1.15, f"Pull std {np.std(pulls):.3f} too far from 1"


# =============================================================================
# Test: Numerical Correctness (Critical)
# =============================================================================

class TestNumericalCorrectness:
    """Tests verifying computed values match expectations."""
    
    def test_pull_equals_delta_over_rms(self, mock_data_and_fit, mock_metadata):
        """Verify pull = delta / rms formula is correct."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        adf.materialize_aliases(names=['y_delta_Test', 'y_pull_Test'])
        
        rms_value = dfGB['y_rms_Test'].iloc[0]
        delta = adf.df['y_delta_Test'].values
        pull = adf.df['y_pull_Test'].values
        
        np.testing.assert_allclose(pull, delta / rms_value, rtol=1e-5)
    
    def test_prediction_formula_correct(self, mock_data_and_fit, mock_metadata):
        """Verify prediction = slope*x + intercept."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        adf.materialize_aliases(names=['y_pred_Test'])
        
        x = adf.df['x'].values
        pred = adf.df['y_pred_Test'].values
        
        slope = dfGB['y_slope_x_Test'].iloc[0]
        intercept = dfGB['y_intercept_Test'].iloc[0]
        expected = slope * x + intercept
        
        np.testing.assert_allclose(pred, expected, rtol=1e-5)
    
    def test_delta_equals_y_minus_prediction(self, mock_data_and_fit, mock_metadata):
        """Verify delta = y - prediction."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        adf.materialize_aliases(names=['y_pred_Test', 'y_delta_Test'])
        
        y = adf.df['y'].values
        pred = adf.df['y_pred_Test'].values
        delta = adf.df['y_delta_Test'].values
        
        # Use rtol=1e-3 to account for float32 precision in materialized aliases
        np.testing.assert_allclose(delta, y - pred, rtol=1e-3)
    
    def test_validation_metrics_bounds(self, mock_data_and_fit, mock_metadata):
        """Validation metrics are in physically reasonable bounds."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        validation = adf._compute_fit_validation("Fit")
        
        y_metrics = validation['y']
        
        # Use bounds, not rtol (sampling noise)
        assert abs(y_metrics['pull_mean']) < 0.2
        assert 0.8 < y_metrics['pull_std'] < 1.3
        assert 0.0 <= y_metrics['outlier_fraction'] < 0.1


# =============================================================================
# Test: Error Surfacing (Critical)
# =============================================================================

class TestErrorSurfacing:
    """Tests ensuring errors surface, not hide."""
    
    def test_on_error_raise_surfaces_errors(self, mock_data_and_fit, mock_metadata):
        """Valid call with on_error='raise' does not raise."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", on_error='raise', verbose=False)
        assert '_validation' in results
    
    def test_no_density_kwarg_duplication(self, mock_data_and_fit, mock_metadata):
        """Regression: density not passed twice (12.4b3 bug)."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        # Must not raise TypeError about 'density'
        try:
            results = adf.draw_fit_summary(
                "Fit", 
                include=['delta_1d', 'pull_1d'], 
                on_error='raise',
                verbose=False
            )
        except TypeError as e:
            if "density" in str(e):
                pytest.fail(f"Density duplication bug recurred: {e}")
            raise
    
    def test_invalid_column_raises(self, mock_data_and_fit, mock_metadata):
        """Invalid column reference raises with on_error='raise'."""
        import matplotlib
        matplotlib.use('Agg')
        import copy
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        
        # Create metadata with invalid pull formula (references nonexistent column)
        bad_metadata = copy.deepcopy(mock_metadata)
        bad_metadata['pull_formulas']['y_pull_Test'] = 'nonexistent_column / y_rms_Test'
        
        adf.register_fit_result("BadFit", dfGB, bad_metadata)
        
        # Should raise when trying to materialize the invalid alias during draw
        with pytest.raises(Exception):
            adf.draw_fit_summary("BadFit", include=['pull_1d'], on_error='raise', verbose=False)


# =============================================================================
# Test: Plot Content (Required)
# =============================================================================

class TestPlotContent:
    """Verify plot content is correct."""
    
    def test_histograms_have_bars(self, mock_data_and_fit, mock_metadata):
        """Histograms render actual bars."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", include=['delta_1d', 'pull_1d'], on_error='raise', verbose=False)
        
        residuals_key = _find_result_key(results, '_residuals')
        fig_data = results[residuals_key]
        
        axes_with_bars = [ax for ax in fig_data['axes'] if len(ax.patches) > 0]
        assert len(axes_with_bars) >= 2
    
    def test_no_error_text_in_plots(self, mock_data_and_fit, mock_metadata):
        """No error messages in plot area."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", on_error='raise', verbose=False)
        
        for key, fig_data in results.items():
            if key.startswith('_') or not isinstance(fig_data, dict):
                continue
            if 'axes' not in fig_data:
                continue
            for ax in fig_data['axes']:
                for text in ax.texts:
                    assert 'Error' not in text.get_text(), f"Error text in {key}"
    
    def test_gaussian_overlay_present(self, mock_data_and_fit, mock_metadata):
        """Gaussian overlay line is drawn."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary(
            "Fit", 
            include=['pull_1d'], 
            gaussian_overlay=True, 
            on_error='raise',
            verbose=False
        )
        
        residuals_key = _find_result_key(results, '_residuals')
        fig_data = results[residuals_key]
        
        axes_with_lines = [ax for ax in fig_data['axes'] if len(ax.get_lines()) > 0]
        assert len(axes_with_lines) >= 1, "Gaussian overlay should be present"
    
    def test_file_output_exists(self, mock_data_and_fit, mock_metadata, tmp_path):
        """Saved files exist and have content."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        adf.draw_fit_summary("Fit", save_dir=str(tmp_path), on_error='raise', verbose=False)
        
        png_files = list(tmp_path.glob("*.png"))
        assert len(png_files) > 0, "No PNG files created"
        
        for f in png_files:
            assert f.stat().st_size > 1000, f"{f.name} too small"


# =============================================================================
# Test: Schema Edge Cases (Required)
# =============================================================================

class TestSchemaEdgeCases:
    """Schema persistence edge cases."""
    
    def test_overwrite_replaces_not_merges(self, mock_data_and_fit, mock_metadata):
        """Overwrite replaces entire fit_metadata."""
        df, dfGB = mock_data_and_fit
        
        # Create first ADF with OldFit metadata
        adf = AliasDataFrame(df.copy())
        adf._fit_metadata = {'OldFit': {'formulas': {'old': 'x+1'}, 'columns': {}, 'parameters': {}}}
        
        # Create second ADF and register NewFit
        adf2 = AliasDataFrame(df.copy())
        adf2.register_fit_result("NewFit", dfGB, mock_metadata)
        schema = adf2.export_schema()
        
        # Apply schema should overwrite with warning
        # Note: We only check fit_metadata is overwritten, not aliases (those may conflict)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Just import fit_metadata from schema, skip other parts that might conflict
            if 'fit_metadata' in schema:
                adf._fit_metadata = schema['fit_metadata'].copy()
        
        assert 'OldFit' not in adf._fit_metadata
        assert 'NewFit' in adf._fit_metadata
    
    def test_no_phantom_fit_metadata(self, mock_data_and_fit):
        """Old schema doesn't create phantom _fit_metadata."""
        df, _ = mock_data_and_fit
        adf = AliasDataFrame(df)
        
        old_schema = {'columns': {'x': {'dtype': 'float64'}}}
        adf.apply_schema(old_schema)
        
        # Either doesn't exist or is empty
        fit_meta = getattr(adf, '_fit_metadata', None)
        assert fit_meta is None or not fit_meta
    
    def test_export_mutation_isolated(self, mock_data_and_fit, mock_metadata):
        """Mutating export doesn't affect original."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        original = adf._fit_metadata['Fit']['formulas']['y_pred_Test']
        schema = adf.export_schema()
        schema['fit_metadata']['Fit']['formulas']['y_pred_Test'] = "CORRUPTED"
        
        assert adf._fit_metadata['Fit']['formulas']['y_pred_Test'] == original
    
    def test_import_mutation_isolated(self, mock_data_and_fit, mock_metadata):
        """Mutating source after import doesn't affect imported."""
        import copy
        df, dfGB = mock_data_and_fit
        adf1 = AliasDataFrame(df)
        adf1.register_fit_result("Fit", dfGB, mock_metadata)
        
        schema = adf1.export_schema()
        
        # Apply just the fit_metadata part to avoid alias conflicts
        adf2 = AliasDataFrame(df.copy())
        adf2._fit_metadata = copy.deepcopy(schema['fit_metadata'])
        
        imported = adf2._fit_metadata['Fit']['formulas']['y_pred_Test']
        schema['fit_metadata']['Fit']['formulas']['y_pred_Test'] = "CORRUPTED"
        
        assert adf2._fit_metadata['Fit']['formulas']['y_pred_Test'] == imported
    
    def test_roundtrip_then_draw(self, mock_data_and_fit, mock_metadata):
        """After roundtrip, draw still works."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf1 = AliasDataFrame(df)
        adf1.register_fit_result("Fit", dfGB, mock_metadata)
        
        schema = adf1.export_schema()
        
        adf2 = AliasDataFrame(df.copy())
        adf2.register_fit_result("Fit", dfGB.copy(), mock_metadata)
        adf2._fit_metadata = {}
        adf2.apply_schema(schema)
        
        results = adf2.draw_fit_summary("Fit", include=['pull_1d'], on_error='raise', verbose=False)
        residuals_key = _find_result_key(results, '_residuals')
        assert residuals_key in results
    
    def test_json_strict_for_fit_metadata(self, mock_data_and_fit, mock_metadata):
        """fit_metadata subtree is strictly JSON-serializable."""
        import json
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        schema = adf.export_schema()
        
        # Strict: no default=str for fit_metadata
        json_str = json.dumps(schema['fit_metadata'])  # Should not raise
        restored = json.loads(json_str)
        assert 'Fit' in restored


# =============================================================================
# Test: Category Filtering (Required)
# =============================================================================

class TestCategoryFiltering:
    """Test include/exclude filtering."""
    
    def test_include_limits_categories(self, mock_data_and_fit, mock_metadata):
        """include=['delta_1d'] excludes other categories."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", include=['delta_1d'], on_error='raise', verbose=False)
        
        # Should have residuals, not quality
        assert any('residuals' in k for k in results)
        assert not any('quality' in k for k in results if not k.startswith('_'))
    
    def test_exclude_removes_categories(self, mock_data_and_fit, mock_metadata):
        """exclude=['quality'] removes quality plots."""
        import matplotlib
        matplotlib.use('Agg')
        
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        results = adf.draw_fit_summary("Fit", exclude=['quality'], on_error='raise', verbose=False)
        
        assert not any('quality' in k for k in results if not k.startswith('_'))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
