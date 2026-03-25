"""
Tests for Phase 13.14.DF v1.0: draw_batch defaults hierarchy + subplot grid.

Test plan from proposal Rev 2 — 17 tests:
- T1:  Old dict format unchanged (backward compat)
- T2:  List format single group
- T3:  Defaults cascade
- T4:  Defaults override
- T5:  ncols layout
- T6:  layout explicit
- T7:  layout overrides ncols
- T8:  Empty subplot hidden
- T9:  suptitle applied
- T10: savefig output
- T11: same=True within group
- T12: same=True first plot raises
- T13: Return structure
- T14: Multiple groups
- T15: figsize applied
- T16: Mixed old + new format in same session
- T17: Value correctness (profile values match standalone)
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dfdraw import DFDraw
from dfdraw.style import set_style


@pytest.fixture(autouse=True)
def cleanup():
    """Close all figures and reset style after each test."""
    yield
    plt.close('all')
    set_style(None)


@pytest.fixture
def sample_df():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    n = 500
    x = np.random.uniform(0, 10, n)
    y1 = 2 * x + np.random.normal(0, 1, n)
    y2 = -x + 5 + np.random.normal(0, 1, n)
    y3 = x ** 0.5 + np.random.normal(0, 0.5, n)
    cat = np.random.choice(['A', 'B', 'C'], n)
    return pd.DataFrame({'x': x, 'y1': y1, 'y2': y2, 'y3': y3, 'cat': cat})


@pytest.fixture
def plotter(sample_df):
    """Create DFDraw instance."""
    return DFDraw(sample_df)


# =========================================================================
# T1: Old dict format unchanged
# =========================================================================

def test_old_dict_format_unchanged(plotter):
    """T1: Old {'name': spec} format still works via isinstance routing."""
    specs = {
        'hist_y1': {'expr': 'y1', 'bins': 30},
        'prof_y1x': {'expr': 'y1:x', 'type': 'profile', 'bins': 20},
    }
    results = plotter.draw_batch(specs, verbose=False)
    assert results['_summary']['total'] == 2
    assert results['_summary']['success'] == 2
    assert 'hist_y1' in results
    assert 'prof_y1x' in results


# =========================================================================
# T2: List format single group
# =========================================================================

def test_list_format_single_group(plotter):
    """T2: List with one group creates figure with correct subplot count."""
    specs = [{
        'name': 'test_group',
        'plots': [
            {'expr': 'y1:x', 'type': 'profile', 'bins': 20},
            {'expr': 'y2:x', 'type': 'profile', 'bins': 20},
        ]
    }]
    results = plotter.draw_batch(specs, verbose=False)
    assert results['_summary']['success'] == 1
    assert 'test_group' in results
    assert len(results['test_group']['axes']) == 2
    assert len(results['test_group']['stats']) == 2


# =========================================================================
# T3: Defaults cascade
# =========================================================================

def test_defaults_cascade(plotter):
    """T3: kwargs < batch defaults < group defaults < plot spec."""
    specs = [{
        'name': 'cascade',
        'defaults': {'type': 'profile', 'bins': 20},
        'plots': [
            {'expr': 'y1:x'},  # inherits type=profile, bins=20
            {'expr': 'y2:x'},  # inherits type=profile, bins=20
        ]
    }]
    # Should not raise — type and bins inherited from group defaults
    results = plotter.draw_batch(specs, verbose=False)
    assert results['_summary']['success'] == 1
    assert len(results['cascade']['stats']) == 2


# =========================================================================
# T4: Defaults override
# =========================================================================

def test_defaults_override(plotter):
    """T4: Plot-level key overrides group default for same key."""
    specs = [{
        'name': 'override',
        'defaults': {'type': 'profile', 'bins': 20},
        'plots': [
            {'expr': 'y1:x'},                     # bins=20 from defaults
            {'expr': 'y2:x', 'bins': 5},           # bins=5 overrides
        ]
    }]
    results = plotter.draw_batch(specs, verbose=False)
    assert results['_summary']['success'] == 1
    # Both plots should work with their respective bin counts


# =========================================================================
# T5: ncols layout
# =========================================================================

def test_ncols_layout(plotter):
    """T5: 4 plots + ncols=2 → 2×2 grid."""
    specs = [{
        'name': 'grid',
        'ncols': 2,
        'plots': [
            {'expr': 'y1:x', 'type': 'profile', 'bins': 20},
            {'expr': 'y2:x', 'type': 'profile', 'bins': 20},
            {'expr': 'y3:x', 'type': 'profile', 'bins': 20},
            {'expr': 'y1', 'type': 'hist', 'bins': 30},
        ]
    }]
    results = plotter.draw_batch(specs, verbose=False)
    assert results['_summary']['success'] == 1
    assert len(results['grid']['axes']) == 4
    fig = results['grid']['fig']
    # 2x2 grid → 4 axes total in figure
    all_axes = fig.get_axes()
    assert len(all_axes) == 4


# =========================================================================
# T6: layout explicit
# =========================================================================

def test_layout_explicit(plotter):
    """T6: layout=(3,2) with 5 plots → 3×2 grid, 6th hidden."""
    specs = [{
        'name': 'explicit_layout',
        'layout': (3, 2),
        'plots': [
            {'expr': f'y{i}:x', 'type': 'profile', 'bins': 20}
            for i in [1, 2, 3, 1, 2]  # 5 plots
        ]
    }]
    results = plotter.draw_batch(specs, verbose=False)
    assert results['_summary']['success'] == 1
    assert len(results['explicit_layout']['axes']) == 5
    fig = results['explicit_layout']['fig']
    all_axes = fig.get_axes()
    # 3x2 = 6 axes, but 6th should be hidden
    assert len(all_axes) == 6
    assert not all_axes[5].get_visible()


# =========================================================================
# T7: layout overrides ncols
# =========================================================================

def test_layout_overrides_ncols(plotter):
    """T7: Both layout and ncols → layout wins."""
    specs = [{
        'name': 'precedence',
        'ncols': 1,        # would give 4×1
        'layout': (2, 2),  # should win → 2×2
        'plots': [
            {'expr': 'y1:x', 'type': 'profile', 'bins': 20},
            {'expr': 'y2:x', 'type': 'profile', 'bins': 20},
            {'expr': 'y3:x', 'type': 'profile', 'bins': 20},
            {'expr': 'y1', 'type': 'hist', 'bins': 30},
        ]
    }]
    results = plotter.draw_batch(specs, verbose=False)
    fig = results['precedence']['fig']
    all_axes = fig.get_axes()
    assert len(all_axes) == 4  # 2×2, not 4×1


# =========================================================================
# T8: Empty subplot hidden
# =========================================================================

def test_empty_subplot_hidden(plotter):
    """T8: 3 plots + ncols=2 → 4th axes not visible."""
    specs = [{
        'name': 'hidden',
        'ncols': 2,
        'plots': [
            {'expr': 'y1:x', 'type': 'profile', 'bins': 20},
            {'expr': 'y2:x', 'type': 'profile', 'bins': 20},
            {'expr': 'y3:x', 'type': 'profile', 'bins': 20},
        ]
    }]
    results = plotter.draw_batch(specs, verbose=False)
    fig = results['hidden']['fig']
    all_axes = fig.get_axes()
    assert len(all_axes) == 4   # 2×2 grid
    assert all_axes[0].get_visible()
    assert all_axes[1].get_visible()
    assert all_axes[2].get_visible()
    assert not all_axes[3].get_visible()  # hidden


# =========================================================================
# T9: suptitle applied
# =========================================================================

def test_suptitle_applied(plotter):
    """T9: suptitle in group spec shows on figure."""
    specs = [{
        'name': 'titled',
        'suptitle': 'My Dashboard Title',
        'plots': [
            {'expr': 'y1:x', 'type': 'profile', 'bins': 20},
        ]
    }]
    results = plotter.draw_batch(specs, verbose=False)
    fig = results['titled']['fig']
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == 'My Dashboard Title'


# =========================================================================
# T10: savefig output
# =========================================================================

def test_savefig_output(plotter):
    """T10: savefig path → file created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_output.png')
        specs = [{
            'name': 'saved',
            'savefig': save_path,
            'plots': [
                {'expr': 'y1:x', 'type': 'profile', 'bins': 20},
            ]
        }]
        results = plotter.draw_batch(specs, verbose=False)
        assert os.path.exists(save_path)
        assert results['saved']['path'] == save_path


# =========================================================================
# T11: same=True within group
# =========================================================================

def test_same_true_within_group(plotter):
    """T11: Second plot same=True overlays on first subplot."""
    specs = [{
        'name': 'overlay',
        'ncols': 1,
        'plots': [
            {'expr': 'y1:x', 'type': 'profile', 'bins': 20},
            {'expr': 'y2:x', 'type': 'profile', 'bins': 20, 'same': True},
        ]
    }]
    results = plotter.draw_batch(specs, verbose=False)
    # Only 1 subplot consumed (second overlays)
    assert len(results['overlay']['axes']) == 1
    # But 2 stats entries (one per plot)
    assert len(results['overlay']['stats']) == 2
    # The axes should have data from both plots
    ax = results['overlay']['axes'][0]
    assert ax.has_data()


# =========================================================================
# T12: same=True first plot raises
# =========================================================================

def test_same_true_first_plot_raises(plotter):
    """T12: same=True on first plot → ValueError."""
    specs = [{
        'name': 'bad',
        'plots': [
            {'expr': 'y1:x', 'type': 'profile', 'bins': 20, 'same': True},
            {'expr': 'y2:x', 'type': 'profile', 'bins': 20},
        ]
    }]
    with pytest.raises(ValueError, match="same=True on first plot"):
        plotter.draw_batch(specs, verbose=False, on_error='raise')


# =========================================================================
# T13: Return structure
# =========================================================================

def test_return_structure(plotter):
    """T13: Return dict has fig, axes, stats, path keys."""
    specs = [{
        'name': 'structure',
        'plots': [
            {'expr': 'y1:x', 'type': 'profile', 'bins': 20},
            {'expr': 'y2:x', 'type': 'profile', 'bins': 20},
        ]
    }]
    results = plotter.draw_batch(specs, verbose=False)
    entry = results['structure']
    assert 'fig' in entry
    assert 'axes' in entry
    assert 'stats' in entry
    assert 'path' in entry
    assert isinstance(entry['axes'], list)
    assert isinstance(entry['stats'], list)
    assert len(entry['axes']) == 2
    assert len(entry['stats']) == 2
    assert '_summary' in results


# =========================================================================
# T14: Multiple groups
# =========================================================================

def test_multiple_groups(plotter):
    """T14: List with 2 groups → 2 entries in results."""
    specs = [
        {
            'name': 'group_a',
            'plots': [{'expr': 'y1:x', 'type': 'profile', 'bins': 20}]
        },
        {
            'name': 'group_b',
            'plots': [{'expr': 'y2:x', 'type': 'profile', 'bins': 20}]
        },
    ]
    results = plotter.draw_batch(specs, verbose=False)
    assert results['_summary']['total'] == 2
    assert results['_summary']['success'] == 2
    assert 'group_a' in results
    assert 'group_b' in results


# =========================================================================
# T15: figsize applied
# =========================================================================

def test_figsize_applied(plotter):
    """T15: figsize=(16, 10) in group spec controls figure size."""
    specs = [{
        'name': 'sized',
        'figsize': (16, 10),
        'plots': [
            {'expr': 'y1:x', 'type': 'profile', 'bins': 20},
        ]
    }]
    results = plotter.draw_batch(specs, verbose=False)
    fig = results['sized']['fig']
    w, h = fig.get_size_inches()
    assert abs(w - 16) < 0.1
    assert abs(h - 10) < 0.1


# =========================================================================
# T16: Mixed old + new format in same session
# =========================================================================

def test_mixed_old_new_format(plotter):
    """T16: Old dict batch then new list batch in same session."""
    # Old format
    old_specs = {'hist_y1': {'expr': 'y1', 'bins': 30}}
    old_results = plotter.draw_batch(old_specs, verbose=False)
    assert old_results['_summary']['success'] == 1

    # New format
    new_specs = [{
        'name': 'new_group',
        'plots': [{'expr': 'y1:x', 'type': 'profile', 'bins': 20}]
    }]
    new_results = plotter.draw_batch(new_specs, verbose=False)
    assert new_results['_summary']['success'] == 1

    # Both should work independently
    assert 'hist_y1' in old_results
    assert 'new_group' in new_results


# =========================================================================
# T17: Value correctness
# =========================================================================

def test_value_correctness(plotter):
    """T17: Profile values in batch subplot match standalone draw."""
    # Standalone draw
    fig_standalone, ax_standalone, stats_standalone = plotter.profile(
        'y1:x', bins=20
    )

    # Batch draw
    specs = [{
        'name': 'verify',
        'plots': [{'expr': 'y1:x', 'type': 'profile', 'bins': 20}]
    }]
    results = plotter.draw_batch(specs, verbose=False)
    stats_batch = results['verify']['stats'][0]

    # Compare key statistics — must match, not just "no crash"
    assert stats_standalone['n'] == stats_batch['n'], \
        f"n mismatch: {stats_standalone['n']} vs {stats_batch['n']}"
    assert abs(stats_standalone['mean_x'] - stats_batch['mean_x']) < 1e-10, \
        "mean_x mismatch"
    assert abs(stats_standalone['mean_y'] - stats_batch['mean_y']) < 1e-10, \
        "mean_y mismatch"

    plt.close('all')
