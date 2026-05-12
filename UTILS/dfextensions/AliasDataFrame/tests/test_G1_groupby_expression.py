"""
BUG_ADF_GroupBy_Expression_Materialization — regression tests.

dfdraw Phase 13.30 validates that group_by must be a real column.
ADF.draw() must materialize group_by expressions (e.g., "row%3")
as per-call temp columns before forwarding to dfdraw.

G1: arithmetic expression materializes through ADF
G2: existing column unchanged
G3: alias works (with explicit materialization)
G4: no alias pollution after expression group_by
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame

try:
    import matplotlib
    matplotlib.use('Agg')
    from dfdraw import DFDraw
    _HAS_DFDRAW = True
except ImportError:
    _HAS_DFDRAW = False


@pytest.fixture
def adf_for_groupby():
    """ADF with integer row column suitable for modular arithmetic."""
    rng = np.random.default_rng(42)
    n = 90
    df = pd.DataFrame({
        'x': np.linspace(0, 1, n, dtype=np.float32),
        'y': rng.standard_normal(n).astype(np.float32),
        'row': np.arange(n, dtype=np.int16),
        'sector': np.tile(np.arange(3, dtype=np.int8), n // 3),
    })
    return AliasDataFrame(df)


@pytest.mark.skipif(not _HAS_DFDRAW, reason="Requires dfdraw")
class TestGroupByExpressionMaterialization:

    @pytest.mark.invariance
    def test_G1_arithmetic_expression_materializes(self, adf_for_groupby):
        """
        adf.draw(group_by='row%3') must produce a grouped figure,
        NOT raise ValueError from dfdraw Phase 13.30 validator.
        Regression for BUG_ADF_GroupBy_Expression_Materialization.
        """
        adf = adf_for_groupby
        fig, ax, stats = adf.draw(
            'y:x', type='profile',
            group_by='row%3',
            bins=10, min_entries=2,
        )
        # Three distinct row%3 values (0, 1, 2) → at least 3 lines
        assert len(ax.get_lines()) >= 3
        assert ax.get_legend() is not None

    @pytest.mark.invariance
    def test_G2_existing_column_unchanged(self, adf_for_groupby):
        """group_by with existing column name works without materialization."""
        adf = adf_for_groupby
        fig, ax, stats = adf.draw(
            'y:x', type='profile',
            group_by='sector',
            bins=10, min_entries=2,
        )
        assert len(ax.get_lines()) >= 3

    @pytest.mark.invariance
    def test_G3_alias_works(self, adf_for_groupby):
        """group_by referencing an alias works when pre-materialized."""
        adf = adf_for_groupby
        adf.add_alias('row_mod3', 'row%3', dtype=np.int8)
        adf.materialize_aliases(names=['row_mod3'])
        fig, ax, stats = adf.draw(
            'y:x', type='profile',
            group_by='row_mod3',
            bins=10, min_entries=2,
        )
        assert len(ax.get_lines()) >= 3

    @pytest.mark.invariance
    def test_G4_no_alias_pollution(self, adf_for_groupby):
        """
        Per-call materialization MUST NOT leak as a persistent alias.
        After adf.draw(group_by='row%3'), 'row%3' should not be in
        adf.aliases — the temp column lived only for that call.
        """
        adf = adf_for_groupby
        assert 'row%3' not in adf.aliases
        adf.draw('y:x', type='profile', group_by='row%3',
                 bins=5, min_entries=2)
        assert 'row%3' not in adf.aliases, \
            "Expression group_by leaked as persistent alias"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
