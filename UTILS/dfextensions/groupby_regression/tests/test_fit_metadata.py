"""
Test suite for Phase 12.4a: Fit Metadata Export

Tests for metadata generation functions in groupby_regression_optimized.py.
These are unit tests that do NOT require AliasDataFrame - they test
structural correctness, schema completeness, and internal consistency.

Integration tests (formula evaluation, prediction accuracy) belong in
AliasDataFrame test suite (Phase 12.4b).

Run with:
    pytest test_fit_metadata.py -v
"""

import pytest
import numpy as np
import pandas as pd
import re
import sys
from pathlib import Path

# Import the module under test (same pattern as other tests in this directory)
sys.path.insert(0, str(Path(__file__).parent))
from ..groupby_regression_optimized import (
    _validate_column_names,
    _build_prediction_formula,
    _build_residual_formula,
    _build_pull_formula,
    _build_fit_metadata,
)


# =============================================================================
# COLUMN NAME VALIDATION TESTS
# =============================================================================

class TestValidateColumnNames:
    """Tests for _validate_column_names()."""
    
    def test_valid_simple_names(self):
        """Simple alphanumeric names are valid."""
        _validate_column_names(['x', 'y', 'z'])  # Should not raise
    
    def test_valid_with_underscores(self):
        """Names with underscores are valid."""
        _validate_column_names(['dy_C2', 'dz_C2', 'rrel_2'])
    
    def test_valid_with_numbers(self):
        """Names with numbers (not leading) are valid."""
        _validate_column_names(['x1', 'x2', 'var123'])
    
    def test_valid_starting_with_underscore(self):
        """Names starting with underscore are valid."""
        _validate_column_names(['_private', '_x', '__dunder'])
    
    def test_valid_mixed_case(self):
        """Mixed case names are valid."""
        _validate_column_names(['DyC2', 'DzC2', 'MyVariable'])
    
    def test_valid_physics_names(self):
        """Typical physics column names are valid."""
        _validate_column_names([
            'dyC2', 'dzC2', 'rrel', 'rrel2', 'mP4', 
            'track_index', 'firstTForbit', 'weightTPCR'
        ])
    
    def test_invalid_starts_with_number(self):
        """Names starting with number raise ValueError."""
        with pytest.raises(ValueError, match="not safe for formula"):
            _validate_column_names(['2x', 'y'])
    
    def test_invalid_hyphen(self):
        """Names with hyphen raise ValueError."""
        with pytest.raises(ValueError, match="not safe for formula"):
            _validate_column_names(['dy-C2'])
    
    def test_invalid_space(self):
        """Names with space raise ValueError."""
        with pytest.raises(ValueError, match="not safe for formula"):
            _validate_column_names(['dy C2'])
    
    def test_invalid_special_chars(self):
        """Names with special characters raise ValueError."""
        invalid_names = ['dy.C2', 'dy@C2', 'dy$C2', 'dy#C2', 'dy!C2']
        for name in invalid_names:
            with pytest.raises(ValueError, match="not safe for formula"):
                _validate_column_names([name])
    
    def test_invalid_brackets(self):
        """Names with brackets raise ValueError."""
        with pytest.raises(ValueError, match="not safe for formula"):
            _validate_column_names(['dy[0]'])
    
    def test_context_in_error_message(self):
        """Context string appears in error message."""
        with pytest.raises(ValueError, match="in fit_columns"):
            _validate_column_names(['invalid-name'], " in fit_columns")
    
    def test_empty_list_valid(self):
        """Empty list is valid (no columns to validate)."""
        _validate_column_names([])


# =============================================================================
# PREDICTION FORMULA TESTS
# =============================================================================

class TestBuildPredictionFormula:
    """Tests for _build_prediction_formula()."""
    
    def test_single_predictor_with_intercept(self):
        """Single predictor with intercept."""
        formula = _build_prediction_formula('y', ['x'], '_Fit', True)
        assert formula == 'y_intercept_Fit + y_slope_x_Fit*x'
    
    def test_single_predictor_without_intercept(self):
        """Single predictor without intercept."""
        formula = _build_prediction_formula('y', ['x'], '_Fit', False)
        assert formula == 'y_slope_x_Fit*x'
    
    def test_two_predictors_with_intercept(self):
        """Two predictors with intercept."""
        formula = _build_prediction_formula('dyC2', ['rrel', 'rrel2'], '_FitAll', True)
        expected = 'dyC2_intercept_FitAll + dyC2_slope_rrel_FitAll*rrel + dyC2_slope_rrel2_FitAll*rrel2'
        assert formula == expected
    
    def test_two_predictors_without_intercept(self):
        """Two predictors without intercept."""
        formula = _build_prediction_formula('dyC2', ['rrel', 'rrel2'], '_FitAll', False)
        expected = 'dyC2_slope_rrel_FitAll*rrel + dyC2_slope_rrel2_FitAll*rrel2'
        assert formula == expected
    
    def test_many_predictors(self):
        """Many predictors produces correct formula."""
        predictors = ['x1', 'x2', 'x3', 'x4', 'x5']
        formula = _build_prediction_formula('y', predictors, '_Test', True)
        
        # Should have intercept + 5 slope terms
        assert formula.count('+') == 5
        assert formula.count('*') == 5
        
        # All predictors should appear
        for pred in predictors:
            assert f'y_slope_{pred}_Test*{pred}' in formula
    
    def test_preserves_predictor_order(self):
        """Formula terms appear in same order as linear_columns."""
        # Non-alphabetical order
        predictors = ['z', 'a', 'm', 'b']
        formula = _build_prediction_formula('y', predictors, '_Fit', True)
        
        # Find positions of slope terms
        positions = [formula.index(f'slope_{p}') for p in predictors]
        
        # Should be in z, a, m, b order
        assert positions == sorted(positions), "Terms must preserve linear_columns order"
    
    def test_empty_predictors_with_intercept(self):
        """Empty predictors with intercept gives intercept only."""
        formula = _build_prediction_formula('y', [], '_Fit', True)
        assert formula == 'y_intercept_Fit'
    
    def test_empty_predictors_without_intercept_raises(self):
        """Empty predictors without intercept raises ValueError."""
        with pytest.raises(ValueError, match="At least one predictor or intercept"):
            _build_prediction_formula('y', [], '_Fit', False)
    
    def test_different_suffixes(self):
        """Different suffixes produce correct column names."""
        for suffix in ['_v3', '_v4', '_TrackFit', '_SectorCorr', '']:
            formula = _build_prediction_formula('y', ['x'], suffix, True)
            assert f'y_intercept{suffix}' in formula
            assert f'y_slope_x{suffix}' in formula


# =============================================================================
# RESIDUAL FORMULA TESTS
# =============================================================================

class TestBuildResidualFormula:
    """Tests for _build_residual_formula()."""
    
    def test_simple_residual(self):
        """Simple residual formula."""
        pred = 'y_intercept_Fit + y_slope_x_Fit*x'
        formula = _build_residual_formula('y', pred)
        assert formula == 'y - (y_intercept_Fit + y_slope_x_Fit*x)'
    
    def test_parentheses_around_prediction(self):
        """Prediction is wrapped in parentheses for operator precedence."""
        pred = 'a + b*x'
        formula = _build_residual_formula('target', pred)
        assert formula.startswith('target - (')
        assert formula.endswith(')')
    
    def test_complex_prediction(self):
        """Complex multi-term prediction."""
        pred = 'y_intercept_Fit + y_slope_x1_Fit*x1 + y_slope_x2_Fit*x2 + y_slope_x3_Fit*x3'
        formula = _build_residual_formula('y', pred)
        expected = f'y - ({pred})'
        assert formula == expected


# =============================================================================
# PULL FORMULA TESTS
# =============================================================================

class TestBuildPullFormula:
    """Tests for _build_pull_formula()."""
    
    def test_simple_pull(self):
        """Simple pull formula with RMS."""
        resid = 'y - (y_intercept_Fit + y_slope_x_Fit*x)'
        formula = _build_pull_formula(resid, 'y_rms_Fit')
        expected = f'({resid}) / y_rms_Fit'
        assert formula == expected
    
    def test_parentheses_around_residual(self):
        """Residual is wrapped in parentheses for operator precedence."""
        formula = _build_pull_formula('a - b', 'error')
        assert formula.startswith('(')
        assert ') / error' in formula
    
    def test_mad_based_pull(self):
        """MAD-based pull with scale factor."""
        resid = 'y - pred'
        error_expr = '(y_mad_Fit * 1.4826)'
        formula = _build_pull_formula(resid, error_expr)
        assert formula == '(y - pred) / (y_mad_Fit * 1.4826)'


# =============================================================================
# FULL METADATA BUILDER TESTS
# =============================================================================

class TestBuildFitMetadata:
    """Tests for _build_fit_metadata()."""
    
    def test_schema_version_present(self):
        """Metadata includes schema version."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
        )
        assert 'version' in meta
        assert meta['version'] == '1.0'
    
    def test_all_top_level_keys_present(self):
        """All required top-level keys are present."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
        )
        required_keys = ['version', 'formulas', 'residual_formulas', 
                         'pull_formulas', 'columns', 'parameters']
        for key in required_keys:
            assert key in meta, f"Missing top-level key: {key}"
    
    def test_columns_subsections_present(self):
        """All columns subsections are present."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
        )
        required_subsections = ['gb_columns', 'fit_columns', 'linear_columns',
                                'coefficients', 'errors', 'quality', 
                                'diagnostics', 'medians']
        for key in required_subsections:
            assert key in meta['columns'], f"Missing columns subsection: {key}"
    
    def test_parameters_complete(self):
        """Parameters section captures all fit settings."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Custom',
            fit_intercept=False,
            gb_columns=['g1', 'g2'],
            weights_column='w',
            min_stat=10,
            fit_type='linear_v4',
        )
        params = meta['parameters']
        assert params['suffix'] == '_Custom'
        assert params['fit_intercept'] == False
        assert params['weights_column'] == 'w'
        assert params['min_stat'] == 10
        assert params['fit_type'] == 'linear_v4'
        assert params['pull_default'] == 'rms'
    
    def test_single_target_formulas(self):
        """Single target produces correct formulas."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
        )
        
        # Check prediction formula
        assert 'y_pred_Test' in meta['formulas']
        assert meta['formulas']['y_pred_Test'] == 'y_intercept_Test + y_slope_x_Test*x'
        
        # Check residual formula
        assert 'y_delta_Test' in meta['residual_formulas']
        assert 'y - (' in meta['residual_formulas']['y_delta_Test']
        
        # Check pull formulas (both RMS and MAD)
        assert 'y_pull_Test' in meta['pull_formulas']
        assert 'y_pull_mad_Test' in meta['pull_formulas']
        assert 'y_rms_Test' in meta['pull_formulas']['y_pull_Test']
        assert 'y_mad_Test' in meta['pull_formulas']['y_pull_mad_Test']
        assert '1.4826' in meta['pull_formulas']['y_pull_mad_Test']
    
    def test_multiple_targets(self):
        """Multiple targets each get their own formula set."""
        meta = _build_fit_metadata(
            fit_columns=['y1', 'y2', 'y3'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
        )
        
        # Each target should have formulas
        for target in ['y1', 'y2', 'y3']:
            assert f'{target}_pred_Test' in meta['formulas']
            assert f'{target}_delta_Test' in meta['residual_formulas']
            assert f'{target}_pull_Test' in meta['pull_formulas']
            assert f'{target}_pull_mad_Test' in meta['pull_formulas']
        
        # Should be exactly 3 prediction formulas
        assert len(meta['formulas']) == 3
    
    def test_fit_intercept_true_includes_intercept(self):
        """fit_intercept=True includes intercept in formula and columns."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
        )
        
        # Formula should have intercept
        assert 'y_intercept_Test' in meta['formulas']['y_pred_Test']
        
        # Coefficient columns should include intercept
        assert 'y_intercept_Test' in meta['columns']['coefficients']['y']
        assert 'y_intercept_err_Test' in meta['columns']['errors']['y']
    
    def test_fit_intercept_false_excludes_intercept(self):
        """fit_intercept=False excludes intercept from formula and columns."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=False,
            gb_columns=['group'],
        )
        
        # Formula should NOT have intercept
        assert 'intercept' not in meta['formulas']['y_pred_Test']
        
        # Coefficient columns should NOT include intercept
        assert 'y_intercept_Test' not in meta['columns']['coefficients']['y']
        assert 'y_intercept_err_Test' not in meta['columns']['errors']['y']
        
        # But should still have slope
        assert 'y_slope_x_Test' in meta['columns']['coefficients']['y']
    
    def test_columns_categorization(self):
        """Column names are correctly categorized."""
        meta = _build_fit_metadata(
            fit_columns=['dy', 'dz'],
            linear_columns=['x1', 'x2'],
            suffix='_Fit',
            fit_intercept=True,
            gb_columns=['sector', 'row'],
        )
        
        # Check gb_columns
        assert meta['columns']['gb_columns'] == ['sector', 'row']
        
        # Check fit_columns
        assert meta['columns']['fit_columns'] == ['dy', 'dz']
        
        # Check linear_columns  
        assert meta['columns']['linear_columns'] == ['x1', 'x2']
        
        # Check coefficients for each target
        for target in ['dy', 'dz']:
            coefs = meta['columns']['coefficients'][target]
            assert f'{target}_intercept_Fit' in coefs
            assert f'{target}_slope_x1_Fit' in coefs
            assert f'{target}_slope_x2_Fit' in coefs
            
            # Check errors
            errs = meta['columns']['errors'][target]
            assert f'{target}_intercept_err_Fit' in errs
            assert f'{target}_slope_x1_err_Fit' in errs
            
            # Check quality
            qual = meta['columns']['quality'][target]
            assert f'{target}_rms_Fit' in qual
            assert f'{target}_mad_Fit' in qual
    
    def test_diagnostics_empty_when_diag_false(self):
        """diagnostics list is empty when diag=False."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
            diag=False,
        )
        assert meta['columns']['diagnostics'] == []
    
    def test_diagnostics_populated_when_diag_true(self):
        """diagnostics list is populated when diag=True."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
            diag=True,
            diag_prefix='diag_',
        )
        diags = meta['columns']['diagnostics']
        assert 'diag_n_total_Test' in diags
        assert 'diag_n_valid_Test' in diags
        assert 'diag_n_filtered_Test' in diags
        assert 'diag_cond_xtx_Test' in diags
        assert 'diag_status_Test' in diags
    
    def test_custom_diag_prefix(self):
        """Custom diagnostic prefix is applied."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
            diag=True,
            diag_prefix='dbg_',
        )
        diags = meta['columns']['diagnostics']
        assert 'dbg_n_total_Test' in diags
        assert 'diag_n_total_Test' not in diags
    
    def test_medians_empty_when_not_provided(self):
        """medians list is empty when median_columns not provided."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
            median_columns=None,
        )
        assert meta['columns']['medians'] == []
    
    def test_medians_populated_when_provided(self):
        """medians list is populated when median_columns provided."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
            median_columns=['x', 'y', 'z'],
        )
        medians = meta['columns']['medians']
        assert 'x_Test' in medians
        assert 'y_Test' in medians
        assert 'z_Test' in medians
    
    def test_invalid_fit_column_raises(self):
        """Invalid fit column name raises ValueError."""
        with pytest.raises(ValueError, match="not safe for formula"):
            _build_fit_metadata(
                fit_columns=['y-invalid'],
                linear_columns=['x'],
                suffix='_Test',
                fit_intercept=True,
                gb_columns=['group'],
            )
    
    def test_invalid_linear_column_raises(self):
        """Invalid linear column name raises ValueError."""
        with pytest.raises(ValueError, match="not safe for formula"):
            _build_fit_metadata(
                fit_columns=['y'],
                linear_columns=['x.invalid'],
                suffix='_Test',
                fit_intercept=True,
                gb_columns=['group'],
            )
    
    def test_pull_both_rms_and_mad(self):
        """Both RMS and MAD pull formulas are generated."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
        )
        
        # Should have both pull types
        assert 'y_pull_Test' in meta['pull_formulas']
        assert 'y_pull_mad_Test' in meta['pull_formulas']
        
        # RMS pull uses rms column
        assert 'y_rms_Test' in meta['pull_formulas']['y_pull_Test']
        
        # MAD pull uses mad column with scale factor
        mad_pull = meta['pull_formulas']['y_pull_mad_Test']
        assert 'y_mad_Test' in mad_pull
        assert '1.4826' in mad_pull


# =============================================================================
# INTEGRATION-STYLE TESTS (Still without AliasDataFrame)
# =============================================================================

class TestMetadataConsistency:
    """Tests for internal consistency of generated metadata."""
    
    def test_formula_columns_match_coefficient_list(self):
        """Columns referenced in formula match coefficients list."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x1', 'x2'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
        )
        
        formula = meta['formulas']['y_pred_Test']
        coefs = meta['columns']['coefficients']['y']
        
        # Each coefficient should appear in formula (without the * suffix)
        for coef in coefs:
            # Intercept appears as-is, slopes appear with *predictor
            if 'intercept' in coef:
                assert coef in formula
            else:
                assert coef in formula
    
    def test_residual_references_original_target(self):
        """Residual formula references original target column."""
        meta = _build_fit_metadata(
            fit_columns=['dyC2'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
        )
        
        resid = meta['residual_formulas']['dyC2_delta_Test']
        assert resid.startswith('dyC2 - ')
    
    def test_pull_references_quality_column(self):
        """Pull formula references quality (rms/mad) column."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=['group'],
        )
        
        quality_cols = meta['columns']['quality']['y']
        pull_rms = meta['pull_formulas']['y_pull_Test']
        pull_mad = meta['pull_formulas']['y_pull_mad_Test']
        
        # RMS should be in quality and pull
        assert quality_cols[0] in pull_rms  # rms
        assert quality_cols[1] in pull_mad  # mad


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_suffix(self):
        """Empty suffix produces valid formulas."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='',
            fit_intercept=True,
            gb_columns=['group'],
        )
        
        assert meta['formulas']['y_pred'] == 'y_intercept + y_slope_x*x'
    
    def test_long_suffix(self):
        """Long suffix is handled correctly."""
        long_suffix = '_VeryLongSuffixForTesting123'
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix=long_suffix,
            fit_intercept=True,
            gb_columns=['group'],
        )
        
        assert f'y_pred{long_suffix}' in meta['formulas']
    
    def test_many_group_columns(self):
        """Many group-by columns are preserved."""
        gb_cols = ['sector', 'row', 'drift', 'chamber', 'layer']
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Test',
            fit_intercept=True,
            gb_columns=gb_cols,
        )
        
        assert meta['columns']['gb_columns'] == gb_cols
    
    def test_single_character_names(self):
        """Single character column names work."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_T',
            fit_intercept=True,
            gb_columns=['g'],
        )
        
        assert meta['formulas']['y_pred_T'] == 'y_intercept_T + y_slope_x_T*x'


# =============================================================================
# EVALUATION SMOKE TESTS (GPT Review Item 3)
# =============================================================================

class TestFormulaEvaluation:
    """
    Smoke tests to verify generated formulas are actually evaluatable.
    
    These tests use pd.eval() to ensure formulas don't have syntax errors
    and produce valid results when evaluated against real data.
    """
    
    def test_prediction_formula_evaluates(self):
        """Prediction formula can be evaluated with pd.eval()."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x1', 'x2'],
            suffix='_Fit',
            fit_intercept=True,
            gb_columns=['g'],
        )
        
        # Create a DataFrame with the required columns
        df = pd.DataFrame({
            'y': [1.0, 2.0, 3.0],
            'x1': [0.5, 1.0, 1.5],
            'x2': [0.1, 0.2, 0.3],
            'y_intercept_Fit': [0.1, 0.1, 0.1],
            'y_slope_x1_Fit': [2.0, 2.0, 2.0],
            'y_slope_x2_Fit': [3.0, 3.0, 3.0],
        })
        
        formula = meta['formulas']['y_pred_Fit']
        
        # Should not raise
        result = df.eval(formula)
        
        # Verify it produces numeric results
        assert len(result) == 3
        assert not result.isna().any()
        # Check calculation: 0.1 + 2.0*0.5 + 3.0*0.1 = 0.1 + 1.0 + 0.3 = 1.4
        assert abs(result.iloc[0] - 1.4) < 1e-10
    
    def test_residual_formula_evaluates(self):
        """Residual (delta) formula can be evaluated with pd.eval()."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Fit',
            fit_intercept=True,
            gb_columns=['g'],
        )
        
        df = pd.DataFrame({
            'y': [5.0, 6.0, 7.0],
            'x': [1.0, 2.0, 3.0],
            'y_intercept_Fit': [1.0, 1.0, 1.0],
            'y_slope_x_Fit': [2.0, 2.0, 2.0],
        })
        
        formula = meta['residual_formulas']['y_delta_Fit']
        
        # Should not raise
        result = df.eval(formula)
        
        # y - (intercept + slope*x)
        # 5 - (1 + 2*1) = 5 - 3 = 2
        assert abs(result.iloc[0] - 2.0) < 1e-10
    
    def test_pull_rms_formula_evaluates(self):
        """RMS-based pull formula can be evaluated with pd.eval()."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Fit',
            fit_intercept=True,
            gb_columns=['g'],
        )
        
        df = pd.DataFrame({
            'y': [5.0, 6.0, 7.0],
            'x': [1.0, 2.0, 3.0],
            'y_intercept_Fit': [1.0, 1.0, 1.0],
            'y_slope_x_Fit': [2.0, 2.0, 2.0],
            'y_rms_Fit': [0.5, 0.5, 0.5],
        })
        
        formula = meta['pull_formulas']['y_pull_Fit']
        
        # Should not raise
        result = df.eval(formula)
        
        # (y - (intercept + slope*x)) / rms
        # (5 - 3) / 0.5 = 4.0
        assert abs(result.iloc[0] - 4.0) < 1e-10
    
    def test_pull_mad_formula_evaluates(self):
        """MAD-based pull formula can be evaluated with pd.eval()."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Fit',
            fit_intercept=True,
            gb_columns=['g'],
        )
        
        df = pd.DataFrame({
            'y': [5.0, 6.0, 7.0],
            'x': [1.0, 2.0, 3.0],
            'y_intercept_Fit': [1.0, 1.0, 1.0],
            'y_slope_x_Fit': [2.0, 2.0, 2.0],
            'y_mad_Fit': [0.5, 0.5, 0.5],
        })
        
        formula = meta['pull_formulas']['y_pull_mad_Fit']
        
        # Should not raise
        result = df.eval(formula)
        
        # (y - (intercept + slope*x)) / (mad * 1.4826)
        # (5 - 3) / (0.5 * 1.4826) = 2 / 0.7413 ≈ 2.698
        expected = 2.0 / (0.5 * 1.4826)
        assert abs(result.iloc[0] - expected) < 1e-10
    
    def test_no_intercept_formula_evaluates(self):
        """Formula without intercept can be evaluated."""
        meta = _build_fit_metadata(
            fit_columns=['y'],
            linear_columns=['x'],
            suffix='_Fit',
            fit_intercept=False,
            gb_columns=['g'],
        )
        
        df = pd.DataFrame({
            'y': [2.0, 4.0, 6.0],
            'x': [1.0, 2.0, 3.0],
            'y_slope_x_Fit': [2.0, 2.0, 2.0],
            'y_rms_Fit': [0.1, 0.1, 0.1],
            'y_mad_Fit': [0.1, 0.1, 0.1],
        })
        
        pred_formula = meta['formulas']['y_pred_Fit']
        
        # Should not raise
        result = df.eval(pred_formula)
        
        # slope*x = 2*1 = 2
        assert abs(result.iloc[0] - 2.0) < 1e-10
    
    def test_multiple_targets_all_evaluate(self):
        """All formulas for multiple targets can be evaluated."""
        meta = _build_fit_metadata(
            fit_columns=['dy', 'dz'],
            linear_columns=['r'],
            suffix='_Fit',
            fit_intercept=True,
            gb_columns=['g'],
        )
        
        df = pd.DataFrame({
            'dy': [1.0, 2.0],
            'dz': [3.0, 4.0],
            'r': [0.5, 1.0],
            'dy_intercept_Fit': [0.1, 0.1],
            'dy_slope_r_Fit': [1.0, 1.0],
            'dy_rms_Fit': [0.1, 0.1],
            'dy_mad_Fit': [0.1, 0.1],
            'dz_intercept_Fit': [0.2, 0.2],
            'dz_slope_r_Fit': [2.0, 2.0],
            'dz_rms_Fit': [0.2, 0.2],
            'dz_mad_Fit': [0.2, 0.2],
        })
        
        # All prediction formulas should evaluate
        for key, formula in meta['formulas'].items():
            result = df.eval(formula)
            assert not result.isna().any(), f"Formula {key} produced NaN"
        
        # All residual formulas should evaluate
        for key, formula in meta['residual_formulas'].items():
            result = df.eval(formula)
            assert not result.isna().any(), f"Formula {key} produced NaN"
        
        # All pull formulas should evaluate
        for key, formula in meta['pull_formulas'].items():
            result = df.eval(formula)
            assert not result.isna().any(), f"Formula {key} produced NaN"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
