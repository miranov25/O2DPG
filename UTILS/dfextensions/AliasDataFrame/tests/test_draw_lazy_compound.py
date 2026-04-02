"""
Test for BUG_AliasDataFrame_20260401_draw_lazy_compound_expr

draw_lazy must resolve aliases inside compound expressions like
abs(alias), sqrt(alias**2), alias1 - alias2, etc.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


@pytest.fixture
def adf_with_aliases():
    """ADF with several aliases for draw testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'row': np.random.randint(0, 200, 100),
        'sec': np.random.randint(0, 36, 100),
    })
    adf = AliasDataFrame(df)
    adf.draw_lazy = True
    adf.add_alias('dx', 'x * 2')
    adf.add_alias('dy_corr', 'y - x * 0.1')
    adf.add_alias('r', 'sqrt(x**2 + y**2)')
    return adf


class TestDrawLazyCompoundExpr:
    """BUG_AliasDataFrame_20260401: draw_lazy compound expression resolution."""

    def test_parse_simple_alias(self, adf_with_aliases):
        """Simple alias in expression is parsed."""
        adf = adf_with_aliases
        result = adf._parse_expr_aliases('dx:x')
        assert 'dx' in result

    def test_parse_alias_in_abs(self, adf_with_aliases):
        """Alias inside abs() is parsed."""
        adf = adf_with_aliases
        result = adf._parse_expr_aliases('abs(dx):x')
        assert 'dx' in result

    def test_parse_alias_in_sqrt(self, adf_with_aliases):
        """Alias inside sqrt() is parsed."""
        adf = adf_with_aliases
        result = adf._parse_expr_aliases('sqrt(dx**2):x')
        assert 'dx' in result

    def test_parse_alias_arithmetic(self, adf_with_aliases):
        """Alias in arithmetic expression is parsed."""
        adf = adf_with_aliases
        result = adf._parse_expr_aliases('dx-dy_corr:x')
        assert 'dx' in result
        assert 'dy_corr' in result

    def test_parse_alias_in_selection(self, adf_with_aliases):
        """Alias in selection string is parsed."""
        adf = adf_with_aliases
        result = adf._parse_expr_aliases('y:x', selection='abs(dx)<5')
        assert 'dx' in result

    def test_parse_alias_in_group_by(self, adf_with_aliases):
        """Alias in group_by is parsed."""
        adf = adf_with_aliases
        result = adf._parse_expr_aliases('y:x', group_by='dx')
        assert 'dx' in result

    def test_parse_excludes_functions(self, adf_with_aliases):
        """Function names like abs, sqrt are NOT returned as aliases."""
        adf = adf_with_aliases
        result = adf._parse_expr_aliases('abs(dx):sqrt(y)')
        assert 'abs' not in result
        assert 'sqrt' not in result
        assert 'dx' in result

    def test_parse_excludes_physical_columns(self, adf_with_aliases):
        """Physical column names are NOT returned (only aliases)."""
        adf = adf_with_aliases
        result = adf._parse_expr_aliases('dx:x')
        assert 'x' not in result  # physical column
        assert 'dx' in result     # alias

    def test_draw_abs_alias_works(self, adf_with_aliases):
        """draw("abs(alias):x") works with lazy=True."""
        adf = adf_with_aliases
        # Should not raise — dx should be auto-materialized
        fig, ax, stats = adf.draw("abs(dx):x", type='profile', bins=10)
        assert stats['n'] > 0
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_draw_arithmetic_alias_works(self, adf_with_aliases):
        """draw("alias1-alias2:x") works with lazy=True."""
        adf = adf_with_aliases
        fig, ax, stats = adf.draw("dx-dy_corr:x", type='profile', bins=10)
        assert stats['n'] > 0
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_draw_selection_with_alias(self, adf_with_aliases):
        """draw with selection containing alias works."""
        adf = adf_with_aliases
        fig, ax, stats = adf.draw("y:x", selection="abs(dx)<5", type='profile', bins=10)
        assert stats['n'] > 0
        import matplotlib.pyplot as plt
        plt.close('all')

    @pytest.mark.invariance
    def test_invariance_lazy_vs_explicit(self, adf_with_aliases):
        """Invariance: lazy draw == explicit materialize + draw."""
        adf = adf_with_aliases

        # Path 1: explicit materialize
        adf.materialize_alias('dx')
        _, _, stats1 = adf.draw("abs(dx):x", type='profile', bins=10)
        adf.df.drop(columns=['dx'], inplace=True)

        # Path 2: lazy draw (auto-materialize)
        _, _, stats2 = adf.draw("abs(dx):x", type='profile', bins=10)

        assert stats1['n'] == stats2['n']
        np.testing.assert_allclose(stats1['mean_y'], stats2['mean_y'], atol=1e-10)

        import matplotlib.pyplot as plt
        plt.close('all')
