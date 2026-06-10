"""
Tests for batch processing in dfdraw.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tempfile
import os
import json

from dfdraw import DFDraw, set_style


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        'x': np.random.randn(n),
        'y': np.random.randn(n),
        'z': np.random.randn(n),
        'category': np.random.choice(['A', 'B', 'C'], n),
    })


@pytest.fixture(autouse=True)
def reset_style():
    """Reset style and close figures after each test."""
    set_style(None)
    yield
    plt.close('all')


class TestDrawBatchBasic:
    """Test basic draw_batch functionality."""
    
    def test_returns_dict_with_results(self, sample_df):
        """draw_batch returns dict with plot results."""
        plotter = DFDraw(sample_df)
        specs = {
            'hist_x': {'expr': 'x'},
            'scatter_yx': {'expr': 'y:x'},
        }
        results = plotter.draw_batch(specs, verbose=False)
        
        assert isinstance(results, dict)
        assert 'hist_x' in results
        assert 'scatter_yx' in results
    
    def test_summary_keys_present(self, sample_df):
        """Results contain _summary and _errors keys."""
        plotter = DFDraw(sample_df)
        specs = {'hist_x': {'expr': 'x'}}
        results = plotter.draw_batch(specs, verbose=False)
        
        assert '_summary' in results
        assert '_errors' in results
        assert 'total' in results['_summary']
        assert 'success' in results['_summary']
        assert 'failed' in results['_summary']
    
    def test_success_count_correct(self, sample_df):
        """Success count matches number of successful plots."""
        plotter = DFDraw(sample_df)
        specs = {
            'hist_x': {'expr': 'x'},
            'hist_y': {'expr': 'y'},
            'scatter_yx': {'expr': 'y:x'},
        }
        results = plotter.draw_batch(specs, verbose=False)
        
        assert results['_summary']['total'] == 3
        assert results['_summary']['success'] == 3
        assert results['_summary']['failed'] == 0
    
    def test_stats_populated(self, sample_df):
        """Stats dict is populated for each plot."""
        plotter = DFDraw(sample_df)
        specs = {'hist_x': {'expr': 'x', 'bins': 50}}
        results = plotter.draw_batch(specs, verbose=False)
        
        assert 'stats' in results['hist_x']
        assert 'n' in results['hist_x']['stats']
        assert 'mean' in results['hist_x']['stats']
    
    def test_fig_ax_returned_when_not_saving(self, sample_df):
        """Fig and ax returned when save_dir not specified."""
        plotter = DFDraw(sample_df)
        specs = {'hist_x': {'expr': 'x'}}
        results = plotter.draw_batch(specs, verbose=False)
        
        assert results['hist_x']['fig'] is not None
        assert results['hist_x']['ax'] is not None


class TestDrawBatchAutoDetect:
    """Test auto-detection of plot type."""
    
    def test_auto_detect_hist_for_1d(self, sample_df):
        """1D expression auto-detects as hist."""
        plotter = DFDraw(sample_df)
        specs = {'plot1': {'expr': 'x'}}
        results = plotter.draw_batch(specs, verbose=False)
        
        # Hist stats have 'mean' not 'corr'
        assert 'mean' in results['plot1']['stats']
        assert 'corr' not in results['plot1']['stats']
    
    def test_auto_detect_scatter_for_2d(self, sample_df):
        """2D expression auto-detects as scatter."""
        plotter = DFDraw(sample_df)
        specs = {'plot1': {'expr': 'y:x'}}
        results = plotter.draw_batch(specs, verbose=False)
        
        # Scatter stats have 'corr'
        assert 'corr' in results['plot1']['stats']
    
    def test_explicit_type_overrides(self, sample_df):
        """Explicit type overrides auto-detection."""
        plotter = DFDraw(sample_df)
        specs = {'plot1': {'expr': 'y:x', 'type': 'profile'}}
        results = plotter.draw_batch(specs, verbose=False)
        
        # Profile should have been called
        assert results['_summary']['success'] == 1


class TestDrawBatchDefaults:
    """Test defaults parameter."""
    
    def test_defaults_applied(self, sample_df):
        """Defaults are applied to all plots."""
        plotter = DFDraw(sample_df)
        specs = {
            'hist_x': {'expr': 'x'},
            'hist_y': {'expr': 'y'},
        }
        results = plotter.draw_batch(
            specs, 
            defaults={'bins': 30},
            verbose=False
        )
        
        assert results['_summary']['success'] == 2
    
    def test_spec_overrides_defaults(self, sample_df):
        """Per-spec values override defaults."""
        plotter = DFDraw(sample_df)
        specs = {
            'hist_x': {'expr': 'x', 'bins': 100},  # Override
        }
        results = plotter.draw_batch(
            specs,
            defaults={'bins': 30},
            verbose=False
        )
        
        assert results['_summary']['success'] == 1


class TestDrawBatchSaveDir:
    """Test saving to directory."""
    
    def test_creates_directory(self, sample_df):
        """save_dir creates directory if needed."""
        plotter = DFDraw(sample_df)
        specs = {'hist_x': {'expr': 'x'}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, 'newdir')
            results = plotter.draw_batch(specs, save_dir=save_path, verbose=False)
            
            assert os.path.exists(save_path)
    
    def test_saves_files(self, sample_df):
        """Files are saved to save_dir."""
        plotter = DFDraw(sample_df)
        specs = {
            'hist_x': {'expr': 'x'},
            'hist_y': {'expr': 'y'},
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = plotter.draw_batch(specs, save_dir=tmpdir, verbose=False)
            
            assert os.path.exists(os.path.join(tmpdir, 'hist_x.png'))
            assert os.path.exists(os.path.join(tmpdir, 'hist_y.png'))
    
    def test_correct_format(self, sample_df):
        """Files saved with correct format."""
        plotter = DFDraw(sample_df)
        specs = {'hist_x': {'expr': 'x'}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = plotter.draw_batch(
                specs, save_dir=tmpdir, save_format='pdf', verbose=False
            )
            
            assert os.path.exists(os.path.join(tmpdir, 'hist_x.pdf'))
    
    def test_path_in_result(self, sample_df):
        """Saved path is in result dict."""
        plotter = DFDraw(sample_df)
        specs = {'hist_x': {'expr': 'x'}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = plotter.draw_batch(specs, save_dir=tmpdir, verbose=False)
            
            expected_path = os.path.join(tmpdir, 'hist_x.png')
            assert results['hist_x']['path'] == expected_path


class TestDrawBatchErrorHandling:
    """Test error handling."""
    
    def test_skip_continues_on_error(self, sample_df):
        """on_error='skip' continues processing."""
        plotter = DFDraw(sample_df)
        specs = {
            'good_plot': {'expr': 'x'},
            'bad_plot': {'expr': 'nonexistent_column'},
            'another_good': {'expr': 'y'},
        }
        results = plotter.draw_batch(specs, on_error='skip', verbose=False)
        
        assert results['_summary']['success'] == 2
        assert results['_summary']['failed'] == 1
        assert 'bad_plot' in results['_errors']
    
    def test_raise_stops_on_error(self, sample_df):
        """on_error='raise' stops on first error."""
        plotter = DFDraw(sample_df)
        specs = {
            'bad_plot': {'expr': 'nonexistent'},
            'good_plot': {'expr': 'x'},
        }
        
        with pytest.raises((KeyError, ValueError)):
            plotter.draw_batch(specs, on_error='raise', verbose=False)
    
    def test_errors_collected(self, sample_df):
        """Errors are collected in _errors dict."""
        plotter = DFDraw(sample_df)
        specs = {
            'bad1': {'expr': 'missing1'},
            'bad2': {'expr': 'missing2'},
        }
        results = plotter.draw_batch(specs, on_error='skip', verbose=False)
        
        assert len(results['_errors']) == 2
        assert 'bad1' in results['_errors']
        assert 'bad2' in results['_errors']


class TestDrawBatchVerbose:
    """Test verbose output."""
    
    def test_verbose_true_prints(self, sample_df, capsys):
        """verbose=True prints progress."""
        plotter = DFDraw(sample_df)
        specs = {'hist_x': {'expr': 'x'}}
        results = plotter.draw_batch(specs, verbose=True)
        
        captured = capsys.readouterr()
        assert '[1/1]' in captured.out
        assert 'hist_x' in captured.out
        assert 'Completed' in captured.out
    
    def test_verbose_false_silent(self, sample_df, capsys):
        """verbose=False is silent."""
        plotter = DFDraw(sample_df)
        specs = {'hist_x': {'expr': 'x'}}
        results = plotter.draw_batch(specs, verbose=False)
        
        captured = capsys.readouterr()
        assert captured.out == ''


class TestDrawBatchCloseFigures:
    """Test close_figures parameter."""
    
    def test_close_figures_true_clears_fig(self, sample_df):
        """close_figures=True sets fig/ax to None."""
        plotter = DFDraw(sample_df)
        specs = {'hist_x': {'expr': 'x'}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = plotter.draw_batch(
                specs, save_dir=tmpdir, close_figures=True, verbose=False
            )
            
            assert results['hist_x']['fig'] is None
            assert results['hist_x']['ax'] is None
    
    def test_close_figures_false_keeps_fig(self, sample_df):
        """close_figures=False keeps fig/ax."""
        plotter = DFDraw(sample_df)
        specs = {'hist_x': {'expr': 'x'}}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            results = plotter.draw_batch(
                specs, save_dir=tmpdir, close_figures=False, verbose=False
            )
            
            assert results['hist_x']['fig'] is not None
            assert results['hist_x']['ax'] is not None


class TestDrawBatchFileLoad:
    """Test loading specs from files."""
    
    def test_json_file(self, sample_df):
        """JSON file loading works."""
        plotter = DFDraw(sample_df)
        
        specs_data = {
            'hist_x': {'expr': 'x', 'bins': 50},
            'scatter_yx': {'expr': 'y:x'},
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(specs_data, f)
            f.flush()
            
            try:
                results = plotter.draw_batch(f.name, verbose=False)
                
                assert results['_summary']['success'] == 2
                assert 'hist_x' in results
                assert 'scatter_yx' in results
            finally:
                os.unlink(f.name)
    
    def test_json_with_plots_key(self, sample_df):
        """JSON with 'plots' key works."""
        plotter = DFDraw(sample_df)
        
        specs_data = {
            'plots': {
                'hist_x': {'expr': 'x'},
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(specs_data, f)
            f.flush()
            
            try:
                results = plotter.draw_batch(f.name, verbose=False)
                assert 'hist_x' in results
            finally:
                os.unlink(f.name)
    
    def test_yaml_file(self, sample_df):
        """YAML file loading works (if pyyaml available)."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        
        plotter = DFDraw(sample_df)
        
        yaml_content = """
hist_x:
  expr: x
  bins: 50
scatter_yx:
  expr: "y:x"
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            
            try:
                results = plotter.draw_batch(f.name, verbose=False)
                
                assert results['_summary']['success'] == 2
            finally:
                os.unlink(f.name)
    
    def test_unsupported_format_raises(self, sample_df):
        """Unsupported file format raises ValueError."""
        plotter = DFDraw(sample_df)
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            plotter.draw_batch('specs.txt', verbose=False)


class TestDrawBatchPlotTypes:
    """Test different plot types in batch."""
    
    def test_all_plot_types(self, sample_df):
        """All plot types work in batch."""
        plotter = DFDraw(sample_df)
        specs = {
            'hist': {'expr': 'x', 'type': 'hist'},
            'scatter': {'expr': 'y:x', 'type': 'scatter'},
            'profile': {'expr': 'y:x', 'type': 'profile'},
            'hist2d': {'expr': 'y:x', 'type': 'hist2d'},
            'hexbin': {'expr': 'y:x', 'type': 'hexbin'},
        }
        results = plotter.draw_batch(specs, verbose=False)
        
        assert results['_summary']['success'] == 5
        assert results['_summary']['failed'] == 0
    
    def test_invalid_type_error(self, sample_df):
        """Invalid type raises error."""
        plotter = DFDraw(sample_df)
        specs = {'bad': {'expr': 'x', 'type': 'invalid_type'}}
        
        results = plotter.draw_batch(specs, on_error='skip', verbose=False)
        
        assert 'bad' in results['_errors']
        # Phase 13.55.DF: draw_batch now routes through draw(), which raises
        # "Unknown plot type 'invalid_type'..." instead of the prior
        # "Invalid type 'invalid_type'...". Behavior contract (invalid type
        # produces a visible error) preserved; only the wording changed.
        # Assert on the user-supplied type-name itself for source-independence.
        assert 'invalid_type' in results['_errors']['bad']


class TestDrawBatchMissingExpr:
    """Test handling of missing expr."""
    
    def test_missing_expr_error(self, sample_df):
        """Missing expr raises error."""
        plotter = DFDraw(sample_df)
        specs = {'bad': {'bins': 50}}  # No expr
        
        results = plotter.draw_batch(specs, on_error='skip', verbose=False)
        
        assert 'bad' in results['_errors']
        assert 'expr' in results['_errors']['bad'].lower()
