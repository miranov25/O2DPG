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
    
    @pytest.mark.skip(reason="Requires schema export/import modifications")
    def test_export_includes_fit_metadata(self, mock_data_and_fit, mock_metadata):
        """export_schema includes fit_metadata."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, mock_metadata)
        
        schema = adf.export_schema()
        
        assert 'fit_metadata' in schema
        assert 'Fit' in schema['fit_metadata']
    
    @pytest.mark.skip(reason="Requires schema export/import modifications")
    def test_apply_schema_restores_fit_metadata(self, mock_data_and_fit, mock_metadata):
        """apply_schema restores fit_metadata."""
        df, dfGB = mock_data_and_fit
        adf1 = AliasDataFrame(df)
        adf1.register_fit_result("Fit", dfGB, mock_metadata)
        
        schema = adf1.export_schema()
        
        adf2 = AliasDataFrame(df)
        adf2.apply_schema(schema)
        
        assert 'Fit' in adf2._fit_metadata
    
    @pytest.mark.skip(reason="Requires schema export/import modifications")
    def test_apply_schema_overwrites_with_warning(self, mock_data_and_fit, mock_metadata):
        """apply_schema warns when overwriting existing fit_metadata."""
        df, dfGB = mock_data_and_fit
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit1", dfGB, mock_metadata)
        
        schema = {'fit_metadata': {'Fit2': mock_metadata}}
        
        with pytest.warns(UserWarning, match="Overwriting"):
            adf.apply_schema(schema)


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
