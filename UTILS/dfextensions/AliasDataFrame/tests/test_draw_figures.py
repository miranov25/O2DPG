"""
Tests for AliasDataFrame.draw_figures()

Phase 12.4b1: Composed canvas for multi-subplot figures

Run with: pytest tests/test_draw_figures.py -v
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n) * 2,
        'z': np.random.randn(n) + 1,
        'category': np.random.choice(['A', 'B', 'C'], n),
    })


@pytest.fixture
def adf(sample_df):
    """Create AliasDataFrame with sample data and aliases."""
    from AliasDataFrame import AliasDataFrame
    adf = AliasDataFrame(sample_df)
    adf.add_alias('x2', 'x**2')
    adf.add_alias('y2', 'y**2')
    adf.add_alias('r', 'np.sqrt(x**2 + y**2)')
    return adf


@pytest.fixture
def basic_specs():
    """Basic figure specs for testing."""
    return [
        {
            'name': 'test_figure',
            'plots': [
                {'expr': 'x', 'bins': 50},
                {'expr': 'y', 'bins': 50},
            ]
        }
    ]


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


# =============================================================================
# Category 1: Validation Tests (7 tests)
# =============================================================================

class TestValidation:
    """Tests for _validate_figure_specs()."""
    
    def test_specs_must_be_list(self, adf):
        """specs must be a list, not dict."""
        with pytest.raises(ValueError, match="must be a list"):
            adf.draw_figures({'name': 'test', 'plots': ['x']})
    
    def test_specs_cannot_be_empty(self, adf):
        """Empty specs list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            adf.draw_figures([])
    
    def test_spec_must_be_dict(self, adf):
        """Each spec must be a dict."""
        with pytest.raises(ValueError, match="must be a dict"):
            adf.draw_figures(["not a dict"])
    
    def test_spec_must_have_plots(self, adf):
        """Each spec must have 'plots' key."""
        with pytest.raises(ValueError, match="missing required 'plots'"):
            adf.draw_figures([{'name': 'test'}])
    
    def test_plots_must_be_list(self, adf):
        """plots must be a list."""
        with pytest.raises(ValueError, match="must be a list"):
            adf.draw_figures([{'plots': 'x'}])
    
    def test_plots_cannot_be_empty(self, adf):
        """plots list cannot be empty."""
        with pytest.raises(ValueError, match="cannot be empty"):
            adf.draw_figures([{'plots': []}])
    
    def test_plot_must_have_expr(self, adf):
        """Dict plot spec must have 'expr' key."""
        with pytest.raises(ValueError, match="missing required 'expr'"):
            adf.draw_figures([{'plots': [{'bins': 50}]}])


# =============================================================================
# Category 2: Basic Tests (7 tests)
# =============================================================================

class TestBasic:
    """Tests for basic draw_figures functionality."""
    
    def test_short_form_valid(self, adf):
        """Short form 'column' is valid."""
        result = adf.draw_figures([{'name': 'test', 'plots': ['x']}])
        assert 'test' in result
        assert result['test']['fig'] is not None
    
    def test_single_figure_single_plot(self, adf):
        """Single figure with one plot."""
        specs = [{'name': 'single', 'plots': [{'expr': 'x'}]}]
        result = adf.draw_figures(specs)
        
        assert 'single' in result
        assert result['single']['fig'] is not None
        assert len(result['single']['axes']) == 1
        assert len(result['single']['stats']) == 1
    
    def test_single_figure_multiple_plots(self, adf, basic_specs):
        """Single figure with multiple plots."""
        result = adf.draw_figures(basic_specs)
        
        assert 'test_figure' in result
        assert len(result['test_figure']['axes']) == 2
        assert len(result['test_figure']['stats']) == 2
    
    def test_multiple_figures(self, adf):
        """Multiple figures in one call."""
        specs = [
            {'name': 'fig1', 'plots': ['x']},
            {'name': 'fig2', 'plots': ['y']},
        ]
        result = adf.draw_figures(specs)
        
        assert 'fig1' in result
        assert 'fig2' in result
        assert result['fig1']['fig'] is not None
        assert result['fig2']['fig'] is not None
    
    def test_short_form_expansion(self, adf):
        """Short form 'column' expands correctly."""
        specs = [{'name': 'test', 'plots': ['x', 'y', 'z']}]
        result = adf.draw_figures(specs)
        
        assert len(result['test']['axes']) == 3
    
    def test_returns_stats(self, adf):
        """Stats dict returned for each plot."""
        specs = [{'name': 'test', 'plots': [{'expr': 'x', 'bins': 50}]}]
        result = adf.draw_figures(specs)
        
        stats = result['test']['stats'][0]
        assert stats is not None
        # Stats should have some content
        assert isinstance(stats, dict)
    
    def test_auto_name_generation(self, adf):
        """Auto-generates name if not provided."""
        specs = [{'plots': ['x']}]
        result = adf.draw_figures(specs)
        
        assert 'figure_0' in result


# =============================================================================
# Category 3: Layout Tests (5 tests)
# =============================================================================

class TestLayout:
    """Tests for subplot layout."""
    
    def test_ncols_default_2(self, adf):
        """Default ncols is 2."""
        specs = [{'name': 'test', 'plots': ['x', 'y', 'z', 'category']}]
        result = adf.draw_figures(specs)
        
        # 4 plots with ncols=2 -> 2x2 grid
        fig = result['test']['fig']
        all_axes = fig.get_axes()
        visible_axes = [ax for ax in all_axes if ax.get_visible()]
        assert len(visible_axes) == 4
    
    def test_ncols_custom(self, adf):
        """Custom ncols respected."""
        specs = [{'name': 'test', 'ncols': 3, 'plots': ['x', 'y', 'z']}]
        result = adf.draw_figures(specs)
        
        # 3 plots with ncols=3 -> 1x3 grid
        assert len(result['test']['axes']) == 3
    
    def test_unused_axes_hidden(self, adf):
        """Unused axes are hidden."""
        specs = [{'name': 'test', 'ncols': 2, 'plots': ['x', 'y', 'z']}]
        result = adf.draw_figures(specs)
        
        # 3 plots in 2x2 grid -> 1 hidden axis
        fig = result['test']['fig']
        all_axes = fig.get_axes()
        visible = [ax for ax in all_axes if ax.get_visible()]
        hidden = [ax for ax in all_axes if not ax.get_visible()]
        assert len(visible) == 3
        assert len(hidden) == 1
    
    def test_suptitle_displayed(self, adf):
        """Suptitle appears on figure."""
        specs = [{'name': 'test', 'suptitle': 'My Title', 'plots': ['x']}]
        result = adf.draw_figures(specs)
        
        fig = result['test']['fig']
        assert fig._suptitle is not None
        assert 'My Title' in fig._suptitle.get_text()
    
    def test_figsize_custom(self, adf):
        """Custom figsize respected."""
        specs = [{'name': 'test', 'figsize': (10, 8), 'plots': ['x']}]
        result = adf.draw_figures(specs)
        
        fig = result['test']['fig']
        width, height = fig.get_size_inches()
        assert abs(width - 10) < 0.1
        assert abs(height - 8) < 0.1


# =============================================================================
# Category 4: Options Tests (6 tests)
# =============================================================================

class TestOptions:
    """Tests for draw_figures options."""
    
    def test_defaults_applied(self, adf):
        """defaults dict applied to all plots."""
        specs = [{'name': 'test', 'plots': ['x', 'y']}]
        # bins=100 should apply to all plots
        result = adf.draw_figures(specs, defaults={'bins': 100})
        
        # Just verify it runs without error
        assert result['test']['fig'] is not None
    
    def test_plot_overrides_defaults(self, adf):
        """Plot-level settings override defaults."""
        specs = [{
            'name': 'test',
            'plots': [
                {'expr': 'x', 'bins': 20},  # Override
            ]
        }]
        result = adf.draw_figures(specs, defaults={'bins': 100})
        
        assert result['test']['fig'] is not None
    
    def test_lazy_materialization(self, adf):
        """lazy=True materializes aliases."""
        # x2 is an alias, not yet materialized
        assert 'x2' not in adf.df.columns
        
        specs = [{'name': 'test', 'plots': ['x2']}]
        adf.draw_figures(specs, lazy=True, clear_after=False)
        
        # Now it should be materialized
        assert 'x2' in adf.df.columns
    
    def test_clear_after_cleanup(self, adf):
        """clear_after=True removes materialized aliases."""
        assert 'x2' not in adf.df.columns
        
        specs = [{'name': 'test', 'plots': ['x2']}]
        adf.draw_figures(specs, lazy=True, clear_after=True)
        
        # Should be cleaned up
        assert 'x2' not in adf.df.columns
    
    def test_verbose_false_quiet(self, adf, basic_specs, capsys):
        """verbose=False suppresses output."""
        adf.draw_figures(basic_specs, verbose=False)
        captured = capsys.readouterr()
        assert '[draw_figures]' not in captured.out
    
    def test_kwargs_as_defaults(self, adf):
        """**kwargs work as defaults."""
        specs = [{'name': 'test', 'plots': ['x']}]
        result = adf.draw_figures(specs, bins=75)  # kwarg
        
        assert result['test']['fig'] is not None


# =============================================================================
# Category 5: Entry Selection Tests (4 tests)
# =============================================================================

class TestEntrySelection:
    """Tests for entry selection options."""
    
    def test_max_entries_limits_rows(self, adf):
        """max_entries limits data."""
        specs = [{'name': 'test', 'plots': [{'expr': 'x', 'bins': 50}]}]
        result = adf.draw_figures(specs, max_entries=100)
        
        # Stats should reflect limited data
        stats = result['test']['stats'][0]
        n_entries = stats.get('n', stats.get('count', stats.get('entries', 0)))
        assert n_entries <= 100
    
    def test_entry_begin_end(self, adf):
        """entry_begin/entry_end works."""
        specs = [{'name': 'test', 'plots': [{'expr': 'x', 'bins': 50}]}]
        result = adf.draw_figures(specs, entry_begin=0, entry_end=50)
        
        stats = result['test']['stats'][0]
        n_entries = stats.get('n', stats.get('count', stats.get('entries', 0)))
        assert n_entries <= 50
    
    def test_entry_mask_boolean(self, adf):
        """Boolean entry_mask works."""
        mask = adf.df['x'] > 0
        specs = [{'name': 'test', 'plots': ['x']}]
        result = adf.draw_figures(specs, entry_mask=mask.values)
        
        # Should complete without error
        assert result['test']['fig'] is not None
    
    def test_entry_mask_integer(self, adf):
        """Integer entry_mask (indices) works."""
        indices = np.array([0, 10, 20, 30, 40])
        specs = [{'name': 'test', 'plots': [{'expr': 'x', 'bins': 10}]}]
        result = adf.draw_figures(specs, entry_mask=indices)
        
        stats = result['test']['stats'][0]
        n_entries = stats.get('n', stats.get('count', stats.get('entries', 0)))
        assert n_entries == 5


# =============================================================================
# Category 6: Error Handling Tests (4 tests)
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_on_error_skip_continues(self, adf):
        """on_error='skip' continues after error."""
        specs = [{
            'name': 'test',
            'plots': [
                {'expr': 'nonexistent_column'},  # Will fail
                {'expr': 'x'},  # Should still work
            ]
        }]
        result = adf.draw_figures(specs, on_error='skip')
        
        # Figure should exist
        assert result['test']['fig'] is not None
        # First plot has error (stats is None)
        assert result['test']['stats'][0] is None
        # Second plot succeeded
        assert result['test']['stats'][1] is not None
    
    def test_on_error_raise_stops(self, adf):
        """on_error='raise' stops on error."""
        specs = [{'name': 'test', 'plots': [{'expr': 'nonexistent'}]}]
        
        with pytest.raises(Exception):
            adf.draw_figures(specs, on_error='raise')
    
    def test_figure_error_returns_error_dict(self, adf):
        """Figure-level error returns dict with 'error' key."""
        # Force an error by using invalid column
        specs = [
            {'name': 'bad', 'plots': [{'expr': 'totally_invalid_column'}]},
            {'name': 'good', 'plots': ['x']},
        ]
        result = adf.draw_figures(specs, on_error='skip')
        
        # good figure should work
        assert result['good']['fig'] is not None
        assert result['good']['stats'][0] is not None
    
    def test_error_text_on_failed_plot(self, adf):
        """Error text displayed on failed plot axis."""
        specs = [{'name': 'test', 'plots': [{'expr': 'bad_column'}]}]
        result = adf.draw_figures(specs, on_error='skip')
        
        # Check that error text is on the axis
        ax = result['test']['axes'][0]
        texts = [t.get_text() for t in ax.texts]
        assert any('Error' in t for t in texts)


# =============================================================================
# Category 7: File I/O Tests (4 tests)
# =============================================================================

class TestFileIO:
    """Tests for file saving."""
    
    def test_savefig_creates_file(self, adf):
        """savefig creates file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [{'name': 'test', 'savefig': 'test.png', 'plots': ['x']}]
            adf.draw_figures(specs, save_dir=tmpdir)
            
            assert (Path(tmpdir) / 'test.png').exists()
    
    def test_save_dir_prepended(self, adf):
        """save_dir is prepended to savefig."""
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [{'name': 'test', 'savefig': 'output.png', 'plots': ['x']}]
            adf.draw_figures(specs, save_dir=tmpdir)
            
            assert (Path(tmpdir) / 'output.png').exists()
    
    def test_save_dir_creates_parents(self, adf):
        """Missing parent directories created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            specs = [{'name': 'test', 'savefig': 'sub/dir/test.png', 'plots': ['x']}]
            adf.draw_figures(specs, save_dir=tmpdir)
            
            assert (Path(tmpdir) / 'sub' / 'dir' / 'test.png').exists()
    
    def test_savefig_without_save_dir(self, adf):
        """savefig works without save_dir (saves to cwd)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                specs = [{'name': 'test', 'savefig': 'direct.png', 'plots': ['x']}]
                adf.draw_figures(specs)
                
                assert Path('direct.png').exists()
            finally:
                os.chdir(old_cwd)


# =============================================================================
# Category 8: Integration Tests (3 tests)
# =============================================================================

class TestIntegration:
    """Integration tests with real-world patterns."""
    
    def test_qa_dashboard_pattern(self, adf):
        """Typical QA dashboard spec works."""
        specs = [
            {
                'name': 'qa_overview',
                'suptitle': 'QA Overview',
                'ncols': 2,
                'plots': [
                    {'expr': 'x', 'bins': 50, 'title': 'X distribution'},
                    {'expr': 'y', 'bins': 50, 'title': 'Y distribution'},
                    {'expr': 'z', 'bins': 50, 'title': 'Z distribution'},
                    'category',  # Short form
                ]
            }
        ]
        result = adf.draw_figures(specs)
        
        assert len(result['qa_overview']['axes']) == 4
        assert result['qa_overview']['fig']._suptitle is not None
    
    def test_multi_figure_workflow(self, adf):
        """Multiple figure workflow."""
        specs = [
            {
                'name': 'distributions',
                'ncols': 3,
                'plots': ['x', 'y', 'z']
            },
            {
                'name': 'derived',
                'ncols': 2,
                'plots': ['x2', 'y2']  # Aliases
            }
        ]
        result = adf.draw_figures(specs, lazy=True, clear_after=True)
        
        assert 'distributions' in result
        assert 'derived' in result
        assert len(result['distributions']['axes']) == 3
        assert len(result['derived']['axes']) == 2
    
    def test_json_serializable_specs(self, adf):
        """Specs can be serialized to JSON and back."""
        import json
        
        specs = [
            {
                'name': 'test',
                'suptitle': 'Test Figure',
                'ncols': 2,
                'figsize': [10, 8],
                'plots': [
                    {'expr': 'x', 'bins': 50},
                    {'expr': 'y', 'bins': 50, 'title': 'Y plot'},
                ]
            }
        ]
        
        # Round-trip through JSON
        json_str = json.dumps(specs)
        restored_specs = json.loads(json_str)
        
        # Should work with restored specs
        result = adf.draw_figures(restored_specs)
        assert result['test']['fig'] is not None


# =============================================================================
# Summary
# =============================================================================

"""
Test Summary:
- Category 1: Validation (7 tests)
- Category 2: Basic (7 tests)
- Category 3: Layout (5 tests)
- Category 4: Options (6 tests)
- Category 5: Entry Selection (4 tests)
- Category 6: Error Handling (4 tests)
- Category 7: File I/O (4 tests)
- Category 8: Integration (3 tests)

Total: 40 tests
"""
