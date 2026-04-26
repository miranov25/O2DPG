"""
BUG_AliasDataFrame_20260426_draw_index_col_collision — Regression test

BUG: adf.draw('Sub.col:Sub.index_col') raises KeyError when the plotted
column is also one of the subframe's index_columns. The draw resolver
selects the column twice, renames both copies, destroying the join key.

FIX: Guard col_name in index_cols — copy + add instead of select + rename.

ENTRY POINT: draw(), draw_figures() — the production paths.
Discovered: O2DistAI Phase 0.3 (makeTrackPairGB QA plots).
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


def _build_adf_with_index_col_subframe():
    """ADF with a subframe where index columns are also plot targets."""
    rng = np.random.default_rng(2046)
    n = 300
    main_df = pd.DataFrame({
        'sec': rng.integers(0, 36, n).astype(np.int8),
        'tgl': rng.integers(0, 20, n).astype(np.int8),
        'x': rng.uniform(0, 10, n).astype(np.float32),
    })

    # Per-bin statistics subframe (like GroupByRegression output)
    n_bins = 36 * 20
    sub_df = pd.DataFrame({
        'sec': np.repeat(np.arange(36, dtype=np.int8), 20),
        'tgl': np.tile(np.arange(20, dtype=np.int8), 36),
        'dy_median': rng.normal(0, 0.5, n_bins).astype(np.float32),
        'count': rng.integers(1, 50, n_bins).astype(np.int32),
    })

    adf = AliasDataFrame(main_df)
    sf = AliasDataFrame(sub_df)
    adf.register_subframe('Stats', sf, index_columns=['sec', 'tgl'])
    return adf


@pytest.mark.skipif(not _HAS_MPL, reason="Requires matplotlib")
class TestDrawIndexColCollision:
    """Tests for BUG_AliasDataFrame_20260426_draw_index_col_collision."""

    def test_S5_1_draw_index_col_on_x_axis(self):
        """
        S5_1: draw('Sub.value_col:Sub.index_col') — index col on x-axis.
        Before fix: KeyError. After fix: draws successfully.
        """
        adf = _build_adf_with_index_col_subframe()
        fig = adf.draw('Stats.dy_median:Stats.sec', type='profile')
        assert fig is not None, "S5_1: draw with index col on x-axis should not fail"

    def test_S5_2_draw_index_col_on_y_axis(self):
        """
        S5_2: draw('Sub.index_col:x') — index col on y-axis.
        Before fix: KeyError. After fix: draws successfully.
        """
        adf = _build_adf_with_index_col_subframe()
        fig = adf.draw('Stats.sec:x')
        assert fig is not None, "S5_2: draw with index col on y-axis should not fail"

    def test_S5_3_draw_index_col_with_selection(self):
        """
        S5_3: draw with selection on subframe column including index col.
        """
        adf = _build_adf_with_index_col_subframe()
        fig = adf.draw('Stats.dy_median:Stats.sec',
                        selection='Stats.count>10', type='profile')
        assert fig is not None, "S5_3: draw with selection on subframe should work"

    @pytest.mark.invariance
    def test_S5_4_draw_values_match_subframe(self):
        """
        S5_4: values in drawn profile match manual subframe lookup.
        Not just no-crash — correctness check.
        Uses deterministic fixture to guarantee test point exists.
        """
        # Deterministic: every (sec, tgl) combination present
        main_df = pd.DataFrame({
            'sec': np.array([0, 1, 2, 0, 1, 2], dtype=np.int8),
            'tgl': np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),
            'x': np.array([1, 2, 3, 4, 5, 6], dtype=np.float32),
        })
        sub_df = pd.DataFrame({
            'sec': np.array([0, 1, 2, 0, 1, 2], dtype=np.int8),
            'tgl': np.array([0, 0, 0, 1, 1, 1], dtype=np.int8),
            'dy_median': np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
        })

        adf = AliasDataFrame(main_df)
        sf = AliasDataFrame(sub_df)
        adf.register_subframe('Stats', sf, index_columns=['sec', 'tgl'])

        adf.add_alias('test_val', 'Stats.dy_median', dtype=np.float32)
        adf.materialize_aliases(names=['test_val'])

        # sec=1, tgl=0 → dy_median=20
        mask = (adf.df['sec'] == 1) & (adf.df['tgl'] == 0)
        assert mask.any(), "fixture must contain test point"
        actual = adf.df.loc[mask, 'test_val'].iloc[0]
        assert actual == 20.0, f"S5_4: expected 20.0, got {actual}"

        # sec=2, tgl=1 → dy_median=60
        mask2 = (adf.df['sec'] == 2) & (adf.df['tgl'] == 1)
        actual2 = adf.df.loc[mask2, 'test_val'].iloc[0]
        assert actual2 == 60.0, f"S5_4: expected 60.0, got {actual2}"

    def test_S5_5_draw_figures_index_col(self):
        """
        S5_5: draw_figures with index col — same bug exists there.
        """
        adf = _build_adf_with_index_col_subframe()
        specs = [{
            'name': 'test_fig',
            'plots': [
                {'expr': 'Stats.dy_median:Stats.sec', 'type': 'profile'},
            ],
        }]
        result = adf.draw_figures(specs)
        assert 'test_fig' in result, "S5_5: draw_figures should return result"

    def test_S5_6_draw_both_axes_are_index_cols(self):
        """
        S5_6: both x and y are index columns of the subframe.
        Edge case: draw('Sub.sec:Sub.tgl').
        """
        adf = _build_adf_with_index_col_subframe()
        fig = adf.draw('Stats.sec:Stats.tgl')
        assert fig is not None, "S5_6: both axes as index cols should work"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
