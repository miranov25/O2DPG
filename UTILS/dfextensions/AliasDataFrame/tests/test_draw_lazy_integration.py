"""
Tests for Phase 7.3: Draw Integration with Lazy Loading.

Tests cover:
- draw() auto-loading branches in lazy mode
- draw_batch() pre-scanning and batch-loading
- Incremental loading (no reload of existing)
- Eager mode unchanged
- Error handling
"""

import pytest
import pandas as pd
import numpy as np
import os

from AliasDataFrame import AliasDataFrame


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_root_file(tmp_path):
    """Create a sample ROOT file for testing."""
    uproot = pytest.importorskip("uproot")
    
    file_path = tmp_path / "test_draw_data.root"
    
    n_entries = 500
    np.random.seed(42)
    data = {
        'x': np.random.randn(n_entries).astype(np.float32),
        'y': np.random.randn(n_entries).astype(np.float32),
        'pt': np.abs(np.random.randn(n_entries)).astype(np.float32),
        'eta': np.random.uniform(-2.5, 2.5, n_entries).astype(np.float32),
        'phi': np.random.uniform(-np.pi, np.pi, n_entries).astype(np.float32),
        'isOK': np.random.choice([True, False], n_entries),
        'charge': np.random.choice([-1, 1], n_entries).astype(np.int32),
        'signal': np.random.uniform(0, 100, n_entries).astype(np.float32),
        'trackLength': np.random.uniform(1, 10, n_entries).astype(np.float32),
    }
    
    with uproot.recreate(file_path) as f:
        f['tree'] = data
    
    return str(file_path)


@pytest.fixture
def sample_eager_adf():
    """Create sample eager ADF for testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'pt': np.abs(np.random.randn(100)),
    })
    return AliasDataFrame(df)


@pytest.fixture
def has_dfdraw():
    """Check if dfdraw is available."""
    try:
        import dfdraw
        return True
    except ImportError:
        return False


@pytest.fixture
def has_matplotlib():
    """Check if matplotlib is available."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for testing
        import matplotlib.pyplot as plt
        return True
    except ImportError:
        return False


# =============================================================================
# TestDrawLazyLoading - draw() with lazy loading
# =============================================================================

class TestDrawLazyLoading:
    """Tests for draw() with lazy loading."""
    
    @pytest.fixture(autouse=True)
    def setup(self, has_dfdraw, has_matplotlib):
        """Skip tests if dfdraw or matplotlib not available."""
        if not has_dfdraw:
            pytest.skip("dfdraw not available")
        if not has_matplotlib:
            pytest.skip("matplotlib not available")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt = plt
    
    def test_draw_auto_loads_branches(self, sample_root_file):
        """draw() auto-loads required branches."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # No branches loaded yet
        assert len(adf.loaded_branches) == 0
        
        # Draw should auto-load x, y
        fig, ax, stats = adf.draw('y:x')
        
        assert 'x' in adf.loaded_branches
        assert 'y' in adf.loaded_branches
        self.plt.close(fig)
    
    def test_draw_single_var_loads_branch(self, sample_root_file):
        """draw() with single variable loads that branch."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        fig, ax, stats = adf.draw('pt')
        
        assert 'pt' in adf.loaded_branches
        self.plt.close(fig)
    
    def test_draw_with_selection_loads_all(self, sample_root_file):
        """draw() with selection loads selection columns too."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Use Python-style operators (pandas query requirement)
        fig, ax, stats = adf.draw('y:x', selection='pt > 0.5 and isOK')
        
        assert {'x', 'y', 'pt', 'isOK'} <= adf.loaded_branches
        self.plt.close(fig)
    
    def test_draw_with_group_by_loads_column(self, sample_root_file):
        """draw() with group_by loads that column."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        fig, ax, stats = adf.draw('y:x', group_by='charge')
        
        assert 'charge' in adf.loaded_branches
        self.plt.close(fig)
    
    def test_draw_with_color_loads_column(self, sample_root_file):
        """draw() with color loads that column."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        fig, ax, stats = adf.draw('y:x', color='pt')
        
        assert 'pt' in adf.loaded_branches
        self.plt.close(fig)
    
    def test_draw_incremental_loading(self, sample_root_file):
        """Multiple draws incrementally load branches."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # First draw
        fig1, _, _ = adf.draw('x')
        assert adf.loaded_branches == {'x'}
        self.plt.close(fig1)
        
        # Second draw adds more
        fig2, _, _ = adf.draw('y:x')
        assert adf.loaded_branches == {'x', 'y'}
        self.plt.close(fig2)
        
        # Third draw adds more
        fig3, _, _ = adf.draw('pt:eta')
        assert {'x', 'y', 'pt', 'eta'} <= adf.loaded_branches
        self.plt.close(fig3)
    
    def test_draw_doesnt_reload_existing(self, sample_root_file):
        """draw() doesn't reload already-loaded branches."""
        adf = AliasDataFrame.read_tree_lazy(
            sample_root_file, 'tree',
            branches=['x', 'y']  # Pre-load
        )
        
        initial_branches = adf.loaded_branches.copy()
        
        # Draw using already-loaded branches
        fig, _, _ = adf.draw('y:x')
        
        # No new branches loaded
        assert adf.loaded_branches == initial_branches
        self.plt.close(fig)
    
    def test_draw_with_alias_loads_dependencies(self, sample_root_file):
        """draw() with alias loads alias dependencies."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        adf.add_alias('r', 'np.sqrt(x**2 + y**2)')
        
        # r depends on x, y - drawing r:pt should load x, y, pt
        # lazy=True enables alias materialization
        fig, _, _ = adf.draw('r:pt', lazy=True)
        
        assert {'x', 'y', 'pt'} <= adf.loaded_branches
        self.plt.close(fig)


# =============================================================================
# TestDrawBatchLazyLoading - draw_batch() with lazy loading
# =============================================================================

class TestDrawBatchLazyLoading:
    """Tests for draw_batch() with lazy loading."""
    
    @pytest.fixture(autouse=True)
    def setup(self, has_dfdraw, has_matplotlib):
        """Skip tests if dfdraw or matplotlib not available."""
        if not has_dfdraw:
            pytest.skip("dfdraw not available")
        if not has_matplotlib:
            pytest.skip("matplotlib not available")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt = plt
    
    def test_batch_pre_loads_all_branches(self, sample_root_file, tmp_path):
        """draw_batch() pre-loads all required branches."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        specs = {
            'hist_x': {'expr': 'x'},
            'hist_y': {'expr': 'y'},
            'scatter_pt_eta': {'expr': 'pt:eta'},
        }
        
        adf.draw_batch(specs, save_dir=str(tmp_path), verbose=False)
        
        # All branches should be loaded
        assert {'x', 'y', 'pt', 'eta'} <= adf.loaded_branches
        self.plt.close('all')
    
    def test_batch_with_selections(self, sample_root_file, tmp_path):
        """draw_batch() loads branches from selections too."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        specs = {
            'hist_x': {'expr': 'x', 'selection': 'isOK'},
            'scatter': {'expr': 'y:x', 'selection': 'pt > 0.5'},
        }
        
        adf.draw_batch(specs, save_dir=str(tmp_path), verbose=False)
        
        assert {'x', 'y', 'pt', 'isOK'} <= adf.loaded_branches
        self.plt.close('all')
    
    def test_batch_single_io_operation(self, sample_root_file, tmp_path):
        """draw_batch() loads all branches in one call."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Track ensure_branches calls
        call_count = [0]
        original_ensure = adf.ensure_branches
        def tracking_ensure(names):
            call_count[0] += 1
            return original_ensure(names)
        adf.ensure_branches = tracking_ensure
        
        specs = {
            'p1': {'expr': 'x'},
            'p2': {'expr': 'y'},
            'p3': {'expr': 'pt'},
        }
        
        adf.draw_batch(specs, save_dir=str(tmp_path), verbose=False)
        
        # Should be exactly 1 ensure_branches call (batched)
        assert call_count[0] == 1
        self.plt.close('all')
    
    def test_batch_verbose_output(self, sample_root_file, tmp_path, capsys):
        """draw_batch() verbose shows loaded branches."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        specs = {'hist': {'expr': 'x'}}
        adf.draw_batch(specs, save_dir=str(tmp_path), verbose=True)
        
        captured = capsys.readouterr()
        assert 'Loading' in captured.out and 'branch' in captured.out.lower()
        self.plt.close('all')
    
    def test_batch_with_defaults(self, sample_root_file, tmp_path):
        """draw_batch() handles defaults correctly."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        specs = {
            'hist_x': {},  # Uses default expr
            'hist_y': {'expr': 'y'},
        }
        defaults = {'expr': 'x', 'selection': 'pt > 0'}
        
        adf.draw_batch(specs, save_dir=str(tmp_path), defaults=defaults, verbose=False)
        
        # Should load x, y, pt (from defaults selection)
        assert {'x', 'y', 'pt'} <= adf.loaded_branches
        self.plt.close('all')
    
    def test_batch_no_reload_existing(self, sample_root_file, tmp_path):
        """draw_batch() doesn't reload existing branches."""
        adf = AliasDataFrame.read_tree_lazy(
            sample_root_file, 'tree',
            branches=['x', 'y']  # Pre-load
        )
        
        # Track ensure_branches calls
        call_count = [0]
        original_ensure = adf.ensure_branches
        def tracking_ensure(names):
            call_count[0] += 1
            return original_ensure(names)
        adf.ensure_branches = tracking_ensure
        
        specs = {
            'p1': {'expr': 'y:x'},  # Already loaded
        }
        
        adf.draw_batch(specs, save_dir=str(tmp_path), verbose=False)
        
        # Should NOT call ensure_branches (nothing new to load)
        assert call_count[0] == 0
        self.plt.close('all')


# =============================================================================
# TestDrawEagerUnchanged - Eager mode unchanged
# =============================================================================

class TestDrawEagerUnchanged:
    """Tests that eager mode draw() is unchanged."""
    
    @pytest.fixture(autouse=True)
    def setup(self, has_dfdraw, has_matplotlib):
        """Skip tests if dfdraw or matplotlib not available."""
        if not has_dfdraw:
            pytest.skip("dfdraw not available")
        if not has_matplotlib:
            pytest.skip("matplotlib not available")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt = plt
    
    def test_eager_draw_works(self, sample_eager_adf):
        """Eager ADF draw() works as before."""
        fig, ax, stats = sample_eager_adf.draw('y:x')
        assert fig is not None
        self.plt.close(fig)
    
    def test_eager_draw_batch_works(self, sample_eager_adf, tmp_path):
        """Eager ADF draw_batch() works as before."""
        specs = {'hist': {'expr': 'x'}}
        results = sample_eager_adf.draw_batch(specs, save_dir=str(tmp_path))
        assert '_summary' in results
        self.plt.close('all')
    
    def test_eager_no_lazy_reader(self, sample_eager_adf):
        """Eager ADF has no lazy_reader."""
        assert sample_eager_adf._lazy_reader is None
        assert not sample_eager_adf.is_lazy


# =============================================================================
# TestEdgeCases - Edge cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests."""
    
    @pytest.fixture(autouse=True)
    def setup(self, has_dfdraw, has_matplotlib):
        """Skip tests if dfdraw or matplotlib not available."""
        if not has_dfdraw:
            pytest.skip("dfdraw not available")
        if not has_matplotlib:
            pytest.skip("matplotlib not available")
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt = plt
    
    def test_draw_empty_lazy_then_load(self, sample_root_file):
        """Can draw after loading metadata-only ADF."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Metadata only - no columns
        assert len(adf.df.columns) == 0
        
        # Draw triggers load
        fig, ax, stats = adf.draw('x')
        assert 'x' in adf.loaded_branches
        self.plt.close(fig)
    
    def test_draw_complex_expression(self, sample_root_file):
        """draw() handles complex expressions."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Complex selection with multiple columns
        # Use Python-style operators (pandas query requirement)
        fig, _, _ = adf.draw(
            'y:x',
            selection='(pt > 0.5) and isOK and (abs(eta) < 2.0)',
            group_by='charge'
        )
        
        assert {'x', 'y', 'pt', 'isOK', 'eta', 'charge'} <= adf.loaded_branches
        self.plt.close(fig)


# =============================================================================
# TestLazyLoadingWithoutDraw - Test lazy loading mechanics without draw
# =============================================================================

class TestLazyLoadingWithoutDraw:
    """Test the lazy loading mechanics used by draw (without requiring dfdraw)."""
    
    def test_get_required_branches_for_draw(self, sample_root_file):
        """get_required_branches() works for draw-like parameters."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        required = adf.get_required_branches(
            expr='y:x',
            selection='pt > 0.5 && isOK',
            group_by='charge',
            color='eta'
        )
        
        assert required == {'x', 'y', 'pt', 'isOK', 'charge', 'eta'}
    
    def test_ensure_branches_loads_correctly(self, sample_root_file):
        """ensure_branches() loads the right columns."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        # Nothing loaded
        assert len(adf.loaded_branches) == 0
        
        # Load some branches
        adf.ensure_branches(['x', 'y', 'pt'])
        
        assert adf.loaded_branches == {'x', 'y', 'pt'}
        assert 'x' in adf.df.columns
        assert 'y' in adf.df.columns
        assert 'pt' in adf.df.columns
    
    def test_incremental_ensure(self, sample_root_file):
        """Multiple ensure_branches calls are incremental."""
        adf = AliasDataFrame.read_tree_lazy(sample_root_file, 'tree')
        
        adf.ensure_branches(['x'])
        assert adf.loaded_branches == {'x'}
        
        adf.ensure_branches(['y', 'pt'])
        assert adf.loaded_branches == {'x', 'y', 'pt'}
        
        # Re-ensuring existing doesn't break
        adf.ensure_branches(['x', 'y'])
        assert adf.loaded_branches == {'x', 'y', 'pt'}
