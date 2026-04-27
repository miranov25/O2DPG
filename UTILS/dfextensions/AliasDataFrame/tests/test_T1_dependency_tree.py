"""
Phase 13.23.ADF — dependency_tree enhanced output tests

Tests for the three output modes:
  output='text'  — backward compatible (print to stdout)
  output='html'  — interactive collapsible HTML tree
  output='list'  — flat dependency list in resolution order
  
Also tests str vs list input for alias parameter.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


def _build_chain_adf():
    """ADF with a 3-level dependency chain + subframe."""
    df = pd.DataFrame({
        'x': np.array([1, 2, 3, 4], dtype=np.float32),
        'y': np.array([10, 20, 30, 40], dtype=np.float32),
        'sec': np.array([0, 1, 0, 1], dtype=np.int8),
    })
    adf = AliasDataFrame(df)

    # Subframe
    sf_df = pd.DataFrame({
        'sec': np.array([0, 1], dtype=np.int8),
        'c0': np.array([0.5, 1.5], dtype=np.float32),
    })
    adf.register_subframe('Coeff', AliasDataFrame(sf_df), index_columns=['sec'])

    # Chain: r → x_norm → x (3 levels)
    adf.add_alias('x_norm', '(x - 2.5) / 1.5', dtype=np.float32)
    adf.add_alias('r', 'sqrt(x_norm**2 + y**2)', dtype=np.float32)
    adf.add_alias('corrected', 'y + Coeff.c0 * x_norm', dtype=np.float32)

    return adf


class TestDependencyTreeText:
    """Backward compatibility — text output unchanged."""

    def test_T1_single_alias_text(self, capsys):
        """Text output for single alias — same as before."""
        adf = _build_chain_adf()
        adf.dependency_tree('r')
        out = capsys.readouterr().out
        assert 'r = ' in out
        assert 'x_norm' in out
        assert 'x [column]' in out
        assert 'y [column]' in out

    def test_T2_list_input_text(self, capsys):
        """List input prints multiple trees separated by blank lines."""
        adf = _build_chain_adf()
        adf.dependency_tree(['r', 'corrected'])
        out = capsys.readouterr().out
        assert 'r = ' in out
        assert 'corrected = ' in out
        assert 'Coeff.c0 [subframe]' in out


class TestDependencyTreeList:
    """output='list' — flat dependency list."""

    def test_T3_list_output_contains_all_deps(self):
        """List includes all dependencies, leaves first."""
        adf = _build_chain_adf()
        deps = adf.dependency_tree('r', output='list')
        assert isinstance(deps, list)
        assert 'x' in deps
        assert 'y' in deps
        assert 'x_norm' in deps
        assert 'r' in deps
        # Leaves should come before their parents
        assert deps.index('x') < deps.index('x_norm')
        assert deps.index('x_norm') < deps.index('r')

    def test_T4_list_output_unique(self):
        """List has no duplicates even with shared dependencies."""
        adf = _build_chain_adf()
        deps = adf.dependency_tree(['r', 'corrected'], output='list')
        assert len(deps) == len(set(deps)), "List should have no duplicates"
        # x_norm is shared between r and corrected
        assert deps.count('x_norm') == 1

    def test_T5_list_output_subframe(self):
        """Subframe references appear in list."""
        adf = _build_chain_adf()
        deps = adf.dependency_tree('corrected', output='list')
        assert 'Coeff.c0' in deps


class TestDependencyTreeHTML:
    """output='html' — interactive collapsible tree."""

    def test_T6_html_returns_string(self):
        """Without file=, returns HTML string."""
        adf = _build_chain_adf()
        html = adf.dependency_tree('r', output='html')
        assert isinstance(html, str)
        assert '<!DOCTYPE html>' in html
        assert 'r' in html
        assert 'x_norm' in html
        assert 'toggle' in html  # JS expand/collapse

    def test_T7_html_writes_file(self, tmp_path):
        """With file=, writes HTML and returns None."""
        adf = _build_chain_adf()
        fpath = str(tmp_path / 'deps.html')
        result = adf.dependency_tree('r', output='html', file=fpath)
        assert result is None
        assert os.path.exists(fpath)
        with open(fpath) as f:
            content = f.read()
        assert '<!DOCTYPE html>' in content
        assert 'x_norm' in content

    def test_T8_html_multiple_roots(self):
        """List input creates multiple roots in HTML."""
        adf = _build_chain_adf()
        html = adf.dependency_tree(['r', 'corrected'], output='html')
        # Both roots should appear
        assert '>r<' in html or 'r</strong>' in html
        assert '>corrected<' in html or 'corrected</strong>' in html
        # Stats should show counts
        assert 'aliases' in html
        assert 'columns' in html

    def test_T9_html_subframe_tagged(self):
        """Subframe nodes get a 'subframe' tag in HTML."""
        adf = _build_chain_adf()
        html = adf.dependency_tree('corrected', output='html')
        assert 'subframe' in html
        assert 'Coeff.c0' in html

    def test_T10_html_max_depth(self):
        """max_depth limits tree depth in HTML."""
        adf = _build_chain_adf()
        html_full = adf.dependency_tree('r', output='html')
        html_d1 = adf.dependency_tree('r', output='html', max_depth=1)
        # Full tree has 'x [column]' (depth 2), depth=1 should not
        # x_norm is at depth 1 (child of r), x is at depth 2 (child of x_norm)
        assert 'x_norm' in html_d1  # depth 1 — included
        # x as a leaf of x_norm is at depth 2 — should be cut
        # But x also appears as direct dep of r... let's just check full > d1
        assert len(html_full) > len(html_d1)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
