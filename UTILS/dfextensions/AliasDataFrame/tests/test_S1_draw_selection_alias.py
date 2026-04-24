"""
BUG_AliasDataFrame_20260420_draw_selection_alias — Regression test

BUG: draw_batch() and draw_figures() do not pass selection/weights to
_parse_expr_aliases, so aliases used in selection expressions are not
auto-materialized. draw() handles this correctly.

REPRODUCER: adf.draw_figures([{..., selection='isNotEdge==1'}]) fails
with "name 'isNotEdge' is not defined" when isNotEdge is an alias.
Pre-materializing isNotEdge before the draw call works around the bug.

FIX: Pass selection= and weights= to _parse_expr_aliases in draw_batch
(line ~11135) and draw_figures (line ~11333).

ENTRY POINT: draw_batch(), draw_figures() — the production entry points
where the bug was discovered. Not _parse_expr_aliases directly.
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
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _build_adf_with_selection_alias():
    """ADF where 'isGood' is an alias used in selection, not a physical column."""
    rng = np.random.default_rng(2042)
    n = 200
    df = pd.DataFrame({
        'x': rng.uniform(0, 10, n).astype(np.float32),
        'y': rng.normal(0, 1, n).astype(np.float32),
        'flag': rng.integers(0, 3, n).astype(np.int8),
    })
    adf = AliasDataFrame(df)
    # isGood is an ALIAS, not a physical column
    adf.add_alias('isGood', '(flag > 0)', dtype=np.float32)
    # Also add a plot alias that depends on physical columns
    adf.add_alias('y_abs', 'abs(y)', dtype=np.float32)
    return adf


@pytest.mark.skipif(not _HAS_MPL, reason="Requires matplotlib")
class TestDrawSelectionAliasBug:
    """Tests for BUG_AliasDataFrame_20260420_draw_selection_alias."""

    @pytest.mark.xfail(reason=(
        "draw() defaults to lazy=False, which skips all alias materialization. "
        "Selection aliases are not auto-materialized. Workaround: draw(lazy=True) "
        "or pre-materialize. Separate issue from draw_batch/draw_figures fix."
    ))
    def test_S1_draw_materializes_selection_alias(self):
        """
        S1: draw() with default lazy=False does NOT auto-materialize
        selection aliases. This is a pre-existing limitation.
        
        draw_batch() and draw_figures() always materialize and are
        not affected (S2/S3 pass).
        """
        adf = _build_adf_with_selection_alias()
        assert 'isGood' not in adf.df.columns, "isGood should not be pre-materialized"

        # draw() with lazy=False (default) does not materialize isGood
        fig = adf.draw('y:x', selection='isGood==1')
        assert fig is not None, "S1: draw with selection alias should not fail"

    def test_S1b_draw_lazy_materializes_selection_alias(self):
        """
        S1b: draw(lazy=True) correctly materializes selection aliases.
        This is the workaround for S1.
        """
        adf = _build_adf_with_selection_alias()
        assert 'isGood' not in adf.df.columns

        fig = adf.draw('y:x', selection='isGood==1', lazy=True)
        assert fig is not None, "S1b: draw(lazy=True) with selection alias should work"

    def test_S2_draw_batch_materializes_selection_alias(self):
        """
        S2: draw_batch() must materialize aliases in selection.

        Before fix: fails with "name 'isGood' is not defined"
        After fix: auto-materializes isGood, draws successfully
        """
        adf = _build_adf_with_selection_alias()
        assert 'isGood' not in adf.df.columns

        specs = {
            'plot1': {'expr': 'y:x', 'selection': 'isGood==1'},
        }
        # Before fix: this raises "name 'isGood' is not defined"
        result = adf.draw_batch(specs)
        assert result is not None, "S2: draw_batch with selection alias should not fail"

    def test_S3_draw_figures_materializes_selection_alias(self):
        """
        S3: draw_figures() must materialize aliases in selection.

        Before fix: [ERROR] plot 0 'y:x': name 'isGood' is not defined
        After fix: auto-materializes isGood, draws successfully
        """
        adf = _build_adf_with_selection_alias()
        assert 'isGood' not in adf.df.columns

        specs = [{
            'name': 'test_fig',
            'plots': [
                {'expr': 'y:x', 'selection': 'isGood==1'},
            ],
        }]
        # Before fix: fails silently or raises
        result = adf.draw_figures(specs)
        assert 'test_fig' in result, "S3: draw_figures should return figure result"
        fig_result = result['test_fig']
        # Check no error
        assert fig_result.get('error') is None, (
            f"S3: draw_figures with selection alias failed: {fig_result.get('error')}"
        )

    def test_S4_draw_batch_materializes_weights_alias(self):
        """
        S4: draw_batch() must also materialize aliases used in weights.
        Same bug, same fix — weights= wasn't passed to _parse_expr_aliases either.
        """
        adf = _build_adf_with_selection_alias()
        adf.add_alias('w', 'abs(y) + 1', dtype=np.float32)
        assert 'w' not in adf.df.columns

        specs = {
            'plot1': {'expr': 'y:x', 'weights': 'w'},
        }
        result = adf.draw_batch(specs)
        assert result is not None, "S4: draw_batch with weights alias should not fail"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
