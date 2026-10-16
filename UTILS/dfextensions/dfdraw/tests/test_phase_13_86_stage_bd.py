import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfdraw.drawer import DFDraw


class _AxisTitleSource:
    def __init__(self):
        x = np.linspace(0.0, 3.0, 24)
        self.df = pd.DataFrame({
            'x': x,
            'y0': 1.0 + x,
            'y1': 10.0 + 2.0 * x,
        })
        self._titles = {
            'x': 'X metadata title',
            'y0': r'$Y_0$ metadata title',
            'y1': r'$Y_1$ metadata title',
        }

    def get_axis_title(self, name):
        return self._titles.get(name)


@pytest.mark.parametrize('door', ['typed', 'draw'])
def test_BD1_vector_facets_use_per_component_axis_title_metadata(door):
    src = _AxisTitleSource()
    d = DFDraw(src)
    kwargs = dict(bins=4, facet_by='vector', min_entries=1)
    if door == 'typed':
        fig, axes, _ = d.profile('[y0,y1]:x', **kwargs)
    else:
        fig, axes, _ = d.draw('[y0,y1]:x', type='profile', **kwargs)
    try:
        axes = np.asarray(axes, dtype=object).ravel()
        assert [ax.get_ylabel() for ax in axes[:2]] == [
            r'$Y_0$ metadata title',
            r'$Y_1$ metadata title',
        ]
        assert [ax.get_xlabel() for ax in axes[:2]] == [
            'X metadata title', 'X metadata title'
        ]
    finally:
        plt.close(fig)


def test_BD2_explicit_ylabel_still_wins_over_metadata():
    src = _AxisTitleSource()
    fig, axes, _ = DFDraw(src).profile(
        '[y0,y1]:x', bins=4, facet_by='vector', min_entries=1,
        ylabel='explicit y label',
    )
    try:
        axes = np.asarray(axes, dtype=object).ravel()
        assert [ax.get_ylabel() for ax in axes[:2]] == [
            'explicit y label', 'explicit y label'
        ]
    finally:
        plt.close(fig)


def test_BD3_scalar_reference_behavior_unchanged():
    src = _AxisTitleSource()
    fig, ax, _ = DFDraw(src).profile('y1:x', bins=4, min_entries=1)
    try:
        assert ax.get_ylabel() == r'$Y_1$ metadata title'
        assert ax.get_xlabel() == 'X metadata title'
    finally:
        plt.close(fig)
