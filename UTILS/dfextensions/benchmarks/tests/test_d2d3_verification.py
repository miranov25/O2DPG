"""
Phase 12.14c.GB D2+D3 Verification Tests

Tests for benchmark visualization specs and CLI commands.

Run: pytest benchmarks/tests/test_d2d3_verification.py -v -s
"""

import argparse
import pytest
from pathlib import Path
from unittest.mock import MagicMock

# =============================================================================
# D2: SPECS TESTS
# =============================================================================

class TestD2Specs:
    """Tests for benchmark_specs.yaml loader."""
    
    def test_load_specs_returns_list(self):
        """load_benchmark_specs returns a list."""
        from dfextensions.benchmarks.specs import load_benchmark_specs
        specs = load_benchmark_specs()
        assert isinstance(specs, list)
        assert len(specs) > 0
        print(f"  ✓ Loaded {len(specs)} specs")
    
    def test_specs_have_required_keys(self):
        """Each spec has required keys for its kind."""
        from dfextensions.benchmarks.specs import load_benchmark_specs
        specs = load_benchmark_specs()
        
        for spec in specs:
            assert 'name' in spec, f"Spec missing 'name': {spec}"
            kind = spec.get('kind', 'scatter')
            
            if kind in ('scatter', 'line'):
                assert 'x' in spec, f"Spec '{spec['name']}' (scatter/line) missing 'x'"
                assert 'y' in spec, f"Spec '{spec['name']}' (scatter/line) missing 'y'"
            elif kind == 'hist':
                assert 'x' in spec, f"Spec '{spec['name']}' (hist) missing 'x'"
            elif kind == 'bar':
                assert 'y' in spec, f"Spec '{spec['name']}' (bar) missing 'y'"
        
        print(f"  ✓ All {len(specs)} specs have required keys")
    
    def test_invalid_spec_raises(self):
        """Invalid spec raises ValueError."""
        from dfextensions.benchmarks.specs import _validate_specs
        
        with pytest.raises(ValueError, match="missing required key"):
            _validate_specs([{"title": "no name"}])
        
        print("  ✓ Invalid spec correctly raises ValueError")
    
    def test_invalid_kind_raises(self):
        """Invalid kind raises ValueError."""
        from dfextensions.benchmarks.specs import _validate_specs
        
        with pytest.raises(ValueError, match="invalid kind"):
            _validate_specs([{"name": "test", "kind": "invalid_kind", "x": "a", "y": "b"}])
        
        print("  ✓ Invalid kind correctly raises ValueError")
    
    def test_schema_version_exists(self):
        """YAML has __meta__.schema_version."""
        try:
            import yaml
        except ImportError:
            pytest.skip("PyYAML not installed")
        
        specs_path = Path(__file__).parent.parent / "specs" / "benchmark_specs.yaml"
        if not specs_path.exists():
            pytest.skip(f"Specs file not found: {specs_path}")
        
        with open(specs_path) as f:
            data = yaml.safe_load(f)
        
        assert "__meta__" in data, "Missing __meta__ section"
        assert "schema_version" in data["__meta__"], "Missing schema_version"
        assert data["__meta__"]["schema_version"] == 1
        
        print(f"  ✓ Schema version: {data['__meta__']['schema_version']}")
    
    def test_get_enabled_specs(self):
        """get_enabled_specs filters disabled specs."""
        from dfextensions.benchmarks.specs import load_benchmark_specs, get_enabled_specs
        
        all_specs = load_benchmark_specs()
        enabled_specs = get_enabled_specs(all_specs)
        
        assert len(enabled_specs) <= len(all_specs)
        
        for spec in enabled_specs:
            assert spec.get('enabled', True) is True
        
        n_disabled = len(all_specs) - len(enabled_specs)
        print(f"  ✓ {len(enabled_specs)} enabled, {n_disabled} disabled")


# =============================================================================
# D3: CLI TESTS
# =============================================================================

class TestD3CLI:
    """Tests for CLI commands."""
    
    def test_history_empty_subproject(self):
        """--history returns 1 for nonexistent subproject."""
        from dfextensions.benchmarks.visualization_cli import cmd_history, MockArgs
        
        args = MockArgs(subproject='nonexistent_subproject_xyz')
        result = cmd_history(args)
        assert result == 1
        print("  ✓ --history returns 1 for empty subproject")
    
    def test_history_stats_returns_int(self):
        """--history-stats returns an int."""
        from dfextensions.benchmarks.visualization_cli import cmd_history_stats, MockArgs
        
        args = MockArgs(subproject='groupby_regression', baseline='7d')
        result = cmd_history_stats(args)
        assert isinstance(result, int)
        print(f"  ✓ --history-stats returns int: {result}")
    
    def test_cli_flags_mutually_exclusive(self):
        """--history and --history-stats are mutually exclusive."""
        from dfextensions.benchmarks.visualization_cli import add_visualization_args
        
        parser = argparse.ArgumentParser()
        add_visualization_args(parser)
        
        # Should raise on multiple flags
        with pytest.raises(SystemExit):
            parser.parse_args(['--history', '--history-stats'])
        
        print("  ✓ CLI flags are mutually exclusive")
    
    def test_add_visualization_args(self):
        """add_visualization_args adds expected arguments."""
        from dfextensions.benchmarks.visualization_cli import add_visualization_args
        
        parser = argparse.ArgumentParser()
        add_visualization_args(parser)
        
        # Parse with each flag individually
        args1 = parser.parse_args(['--history'])
        assert args1.history is True
        
        args2 = parser.parse_args(['--history-stats'])
        assert args2.history_stats is True
        
        args3 = parser.parse_args(['--plot', './test_dir'])
        assert args3.plot == './test_dir'
        
        args4 = parser.parse_args(['--baseline', '30d'])
        assert args4.baseline == '30d'
        
        print("  ✓ All visualization args added correctly")


# =============================================================================
# D3: INTEGRATION TESTS
# =============================================================================

class TestD3Integration:
    """Integration tests for CLI commands."""
    
    @pytest.fixture
    def sample_subproject(self):
        """Return subproject name with data."""
        return 'groupby_regression'
    
    def test_history_with_data(self, sample_subproject, capsys):
        """--history shows summary for real subproject."""
        from dfextensions.benchmarks.visualization_cli import cmd_history, MockArgs
        from dfextensions.benchmarks.benchmark_adf import load_benchmark_adf
        
        # Check if data exists
        adf = load_benchmark_adf(sample_subproject, max_runs=5)
        if adf.df.empty:
            pytest.skip(f"No data for {sample_subproject}")
        
        args = MockArgs(subproject=sample_subproject, max_runs=5)
        result = cmd_history(args)
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert "Benchmark History:" in captured.out
        assert "Total results:" in captured.out
        
        print(f"  ✓ --history shows summary (exit code: {result})")
    
    def test_history_stats_with_data(self, sample_subproject, capsys):
        """--history-stats shows statistics for real subproject."""
        from dfextensions.benchmarks.visualization_cli import cmd_history_stats, MockArgs
        from dfextensions.benchmarks.benchmark_adf import compute_benchmark_statistics
        
        # Check if data exists
        stats = compute_benchmark_statistics(sample_subproject)
        if stats.empty:
            pytest.skip(f"No stats for {sample_subproject}")
        
        args = MockArgs(subproject=sample_subproject, baseline='7d')
        result = cmd_history_stats(args)
        
        assert result == 0
        
        captured = capsys.readouterr()
        assert "Benchmark Statistics:" in captured.out
        assert "CV%" in captured.out
        
        print(f"  ✓ --history-stats shows statistics (exit code: {result})")
    
    def test_plot_creates_files(self, sample_subproject, tmp_path):
        """--plot creates PNG files."""
        pytest.importorskip("matplotlib")
        
        from dfextensions.benchmarks.visualization_cli import cmd_plot, MockArgs
        from dfextensions.benchmarks.benchmark_adf import load_benchmark_adf
        
        # Check if data exists
        adf = load_benchmark_adf(sample_subproject, max_runs=5)
        if adf.df.empty:
            pytest.skip(f"No data for {sample_subproject}")
        
        args = MockArgs(
            subproject=sample_subproject,
            plot=str(tmp_path),
            max_runs=5
        )
        result = cmd_plot(args)
        
        assert result in (0, 2)  # Success or partial
        
        png_files = list(tmp_path.glob('*.png'))
        assert len(png_files) > 0, "No PNG files generated"
        
        print(f"  ✓ Generated {len(png_files)} plots to {tmp_path}")
        for f in png_files:
            print(f"    - {f.name}")
    
    def test_plot_missing_matplotlib(self, monkeypatch):
        """--plot returns 3 when matplotlib not installed."""
        from dfextensions.benchmarks.visualization_cli import MockArgs
        
        # Simulate missing matplotlib
        import builtins
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'matplotlib':
                raise ImportError("matplotlib not installed")
            return original_import(name, *args, **kwargs)
        
        monkeypatch.setattr(builtins, '__import__', mock_import)
        
        # Re-import to get fresh module
        from dfextensions.benchmarks import visualization_cli
        import importlib
        importlib.reload(visualization_cli)
        
        args = MockArgs(subproject='test', plot='./test_plots')
        result = visualization_cli.cmd_plot(args)
        
        assert result == 3
        print("  ✓ --plot returns 3 when matplotlib missing")


# =============================================================================
# SUMMARY TEST
# =============================================================================

class TestD2D3Summary:
    """Summary verification."""
    
    def test_d2d3_complete(self, capsys):
        """Combined D2+D3 verification."""
        print("\n" + "=" * 60)
        print("D2+D3 VERIFICATION SUMMARY")
        print("=" * 60)
        
        # D2: Specs
        from dfextensions.benchmarks.specs import load_benchmark_specs, get_enabled_specs
        specs = load_benchmark_specs()
        enabled = get_enabled_specs(specs)
        print(f"\n  D2 Specs:")
        print(f"    Total specs:    {len(specs)}")
        print(f"    Enabled specs:  {len(enabled)}")
        
        # D3: CLI
        from dfextensions.benchmarks.visualization_cli import add_visualization_args
        import argparse
        parser = argparse.ArgumentParser()
        add_visualization_args(parser)
        print(f"\n  D3 CLI:")
        print(f"    --history:       available")
        print(f"    --history-stats: available")
        print(f"    --plot DIR:      available")
        
        print("\n" + "=" * 60)
        print("✅ D2+D3 COMPLETE")
        print("=" * 60)
        
        assert len(specs) > 0
        assert len(enabled) > 0
