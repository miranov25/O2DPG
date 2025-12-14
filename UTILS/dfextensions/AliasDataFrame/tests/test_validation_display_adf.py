"""
Tests for Phase 12.4b5: AliasDataFrame Validation Display Methods

Tests ONLY the AliasDataFrame methods (no DFDraw imports).
DFDraw tests are in dfdraw/tests/test_validation_display.py
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from AliasDataFrame import AliasDataFrame


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fit_data_and_groupby():
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
    
    metadata = {
        'version': '1.0',
        'formulas': {
            'y_pred_Test': 'y_intercept_Test + y_slope_x_Test*x',
        },
        'residual_formulas': {
            'y_delta_Test': 'y - (y_intercept_Test + y_slope_x_Test*x)',
        },
        'pull_formulas': {
            'y_pull_Test': '(y - (y_intercept_Test + y_slope_x_Test*x)) / y_rms_Test',
        },
        'columns': {
            'gb_columns': ['group'],
            'fit_columns': ['y'],
            'quality': {'y': ['y_rms_Test', 'y_mad_Test']},
        },
        'parameters': {
            'suffix': '_Test',
            'pull_default': 'rms',
        },
    }
    
    return df, dfGB, metadata


def _find_result_key(results, pattern):
    """Find result key by pattern."""
    for k in results:
        if pattern in k or k.endswith(pattern):
            return k
    raise KeyError(f"No key matching '{pattern}' in {list(results.keys())}")


# =============================================================================
# Test: _compute_statistics()
# =============================================================================

class TestComputeStatistics:
    """Tests for _compute_statistics() method."""
    
    def test_returns_dict_structure(self, fit_data_and_groupby):
        """Verify _compute_statistics returns proper structure."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, metadata)
        
        stats = adf._compute_statistics("Fit")
        
        assert isinstance(stats, dict)
        assert 'y' in stats
        assert 'pull_mean' in stats['y']
        assert 'pull_std' in stats['y']
        assert 'pull_n' in stats['y']
    
    def test_pull_values_approximately_normal(self, fit_data_and_groupby):
        """Verify pull statistics are approximately N(0,1)."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, metadata)
        
        stats = adf._compute_statistics("Fit")
        
        # Pull should be approximately N(0,1)
        assert abs(stats['y']['pull_mean']) < 0.2, "Pull mean should be ~0"
        assert 0.8 < stats['y']['pull_std'] < 1.3, "Pull std should be ~1"
    
    def test_delta_statistics_included(self, fit_data_and_groupby):
        """Verify delta statistics are computed."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, metadata)
        
        stats = adf._compute_statistics("Fit")
        
        assert 'delta_mean' in stats['y']
        assert 'delta_std' in stats['y']
        assert 'delta_n' in stats['y']
    
    def test_unknown_fit_returns_empty(self):
        """Verify returns empty dict for unknown fit."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        
        stats = adf._compute_statistics("UnknownFit")
        
        assert stats == {}


# =============================================================================
# Test: _add_validation_indicator()
# =============================================================================

class TestAddValidationIndicator:
    """Tests for _add_validation_indicator() method."""
    
    def test_pass_indicator_green(self, fit_data_and_groupby):
        """Verify PASS indicator is green."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        
        fig, ax = plt.subplots()
        text_artist = adf._add_validation_indicator(ax, passed=True)
        
        assert 'PASS' in text_artist.get_text()
        assert text_artist.get_color() == 'green'
        
        plt.close(fig)
    
    def test_fail_indicator_red(self, fit_data_and_groupby):
        """Verify FAIL indicator is red."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        
        fig, ax = plt.subplots()
        text_artist = adf._add_validation_indicator(ax, passed=False)
        
        assert 'FAIL' in text_artist.get_text()
        assert text_artist.get_color() == 'red'
        
        plt.close(fig)


# =============================================================================
# Test: _add_validation_summary()
# =============================================================================

class TestAddValidationSummary:
    """Tests for _add_validation_summary() method."""
    
    def test_summary_contains_overall_status(self, fit_data_and_groupby):
        """Verify summary contains overall PASS/FAIL."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        
        fig, ax = plt.subplots()
        validation_results = {
            '_overall_pass': True,
            'y': {'pass': True, 'pull_mean': 0.05, 'pull_std': 0.98}
        }
        
        text_artist = adf._add_validation_summary(fig, validation_results)
        text_content = text_artist.get_text()
        
        assert 'Overall' in text_content
        assert 'PASS' in text_content
        
        plt.close(fig)
    
    def test_summary_contains_per_column_metrics(self, fit_data_and_groupby):
        """Verify summary contains per-column status."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        
        fig, ax = plt.subplots()
        validation_results = {
            '_overall_pass': False,
            'y': {'pass': True, 'pull_mean': 0.05, 'pull_std': 0.98},
            'w': {'pass': False, 'pull_mean': 0.50, 'pull_std': 1.50}
        }
        
        text_artist = adf._add_validation_summary(fig, validation_results)
        text_content = text_artist.get_text()
        
        assert 'y: PASS' in text_content
        assert 'w: FAIL' in text_content
        
        plt.close(fig)


# =============================================================================
# Test: draw_fit_summary() Integration
# =============================================================================

class TestDrawFitSummaryIntegration:
    """Tests for Phase 12.4b5 integration in draw_fit_summary()."""
    
    def test_statistics_returned_in_results(self, fit_data_and_groupby):
        """Verify _statistics dict is returned."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, metadata)
        
        results = adf.draw_fit_summary("Fit", on_error='raise', verbose=False)
        
        assert '_statistics' in results
        assert 'y' in results['_statistics']
        assert 'pull_mean' in results['_statistics']['y']
        plt.close('all')
    
    def test_show_statistics_adds_annotations(self, fit_data_and_groupby):
        """Verify show_statistics=True adds annotations."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, metadata)
        
        results = adf.draw_fit_summary(
            "Fit", 
            show_statistics=True,
            on_error='raise', 
            verbose=False
        )
        
        residuals_key = _find_result_key(results, '_residuals')
        fig_data = results[residuals_key]
        
        # Check for statistics text (mean or n =)
        stats_found = False
        for ax in fig_data['axes']:
            for text in ax.texts:
                text_content = text.get_text()
                if 'mean' in text_content or 'n =' in text_content:
                    stats_found = True
                    break
        
        assert stats_found, "Statistics annotation should be present"
        plt.close('all')
    
    def test_show_validation_adds_indicator(self, fit_data_and_groupby):
        """Verify show_validation=True adds PASS/FAIL indicator."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, metadata)
        
        results = adf.draw_fit_summary(
            "Fit", 
            show_validation=True,
            on_error='raise', 
            verbose=False
        )
        
        residuals_key = _find_result_key(results, '_residuals')
        fig_data = results[residuals_key]
        
        # Check for PASS or FAIL text
        indicator_found = False
        for ax in fig_data['axes']:
            for text in ax.texts:
                text_content = text.get_text()
                if text_content in ('PASS', 'FAIL'):
                    indicator_found = True
                    break
        
        assert indicator_found, "PASS/FAIL indicator should be present"
        plt.close('all')
    
    def test_show_summary_adds_panel(self, fit_data_and_groupby):
        """Verify show_summary=True adds summary panel."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, metadata)
        
        results = adf.draw_fit_summary(
            "Fit", 
            show_summary=True,
            on_error='raise', 
            verbose=False
        )
        
        residuals_key = _find_result_key(results, '_residuals')
        fig = results[residuals_key]['fig']
        
        # Check figure-level text
        summary_found = False
        for text in fig.texts:
            text_content = text.get_text()
            if 'Summary' in text_content or 'Overall' in text_content:
                summary_found = True
                break
        
        assert summary_found, "Summary panel should be present"
        plt.close('all')
    
    def test_default_no_annotations(self, fit_data_and_groupby):
        """Verify no annotations by default (backward compat)."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, metadata)
        
        results = adf.draw_fit_summary("Fit", on_error='raise', verbose=False)
        
        residuals_key = _find_result_key(results, '_residuals')
        fig_data = results[residuals_key]
        
        # Should NOT have PASS/FAIL indicators by default
        indicator_found = False
        for ax in fig_data['axes']:
            for text in ax.texts:
                if text.get_text() in ('PASS', 'FAIL'):
                    indicator_found = True
        
        assert not indicator_found, "No PASS/FAIL by default"
        plt.close('all')
    
    def test_all_parameters_combined(self, fit_data_and_groupby):
        """Verify all new parameters work together."""
        df, dfGB, metadata = fit_data_and_groupby
        adf = AliasDataFrame(df)
        adf.register_fit_result("Fit", dfGB, metadata)
        
        results = adf.draw_fit_summary(
            "Fit",
            show_statistics=True,
            show_validation=True,
            show_summary=True,
            on_error='raise',
            verbose=False
        )
        
        assert '_validation' in results
        assert '_statistics' in results
        plt.close('all')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
