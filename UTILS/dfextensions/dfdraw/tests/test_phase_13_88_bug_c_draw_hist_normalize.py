"""PHASE_13_88_DF Bug C: draw-door histogram normalize parity."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dfextensions.dfdraw import DFDraw


def test_C1_draw_hist_vector_normalize_matches_typed_owner():
    x = np.linspace(-2.0, 2.0, 1600)
    df = pd.DataFrame({'a': 20.0 + x, 'b': 20.0 + 0.5 * x})
    kwargs = dict(bins=20, range=(17.0, 23.0), normalize='ratio', vector_compose='outer')
    typed_fig, _, typed_stats = DFDraw(df).hist('[a,b]', **kwargs)
    draw_fig, _, draw_stats = DFDraw(df).draw('[a,b]', type='hist', **kwargs)
    try:
        assert 'normalize_data' in draw_stats
        assert len(draw_fig.axes) == len(typed_fig.axes) == 2
        pd.testing.assert_frame_equal(
            draw_stats['normalize_data'].reset_index(drop=True),
            typed_stats['normalize_data'].reset_index(drop=True), check_exact=True)
        for db, tb in zip(draw_stats['hist_branch_data'], typed_stats['hist_branch_data']):
            pd.testing.assert_frame_equal(db.reset_index(drop=True), tb.reset_index(drop=True), check_exact=True)
    finally:
        plt.close(typed_fig)
        plt.close(draw_fig)
