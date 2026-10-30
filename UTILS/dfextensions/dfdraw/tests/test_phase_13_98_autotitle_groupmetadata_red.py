"""PHASE_13_98_DF independent RED/GREEN title ownership and provenance tests.

AT: N-D inner vector/profile calls must never fit titles which the Figure
owns. GM: the visible Figure must identify its physical group_by column,
not the private global-group column. Scientific profile state stays unchanged.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw
from dfextensions.dfdraw.plots import _auto_title


@pytest.fixture
def frame():
    rows = []
    for fr in range(2):
        for fc in range(2):
            for g in range(3):
                for xb in range(4):
                    for k in range(9):
                        rows.append(dict(
                            fr=fr, fc=fc, g=g + .01 * k,
                            x=xb + .02 * k,
                            y0=.2 * xb + .4 * g + k / 12,
                            y1=-.3 * xb + .2 * fr + .1 * fc - k / 17,
                            keep=True,
                        ))
    return pd.DataFrame(rows)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def draw_faceted(frame, door='profile', **options):
    kw = dict(
        facet_by=['fr', 'fc'], facet_by_bins=[2, 2],
        group_by='g', group_by_bins=3, group_by_scope='global',
        bins=4, min_entries=2, selection='keep',
        legend='shared',
        figure_margins=[.18, .96, .79, .12],
    )
    kw.update(options)
    obj = DFDraw(frame)
    if door == 'draw':
        return obj.draw('[y0,y1]:x', type='profile', **kw)
    return obj.profile('[y0,y1]:x', **kw)


@pytest.mark.parametrize('door', ['profile', 'draw'])
@pytest.mark.parametrize('auto_title', [False, True])
def test_AT01_AT02_no_discarded_inner_title_metrics(frame, monkeypatch,
                                                     door, auto_title):
    measured = []
    monkeypatch.setattr(_auto_title, 'fit_auto_title_text',
                        lambda text, *args, **kw: measured.append(str(text)) or text)
    fig, axes, stats = draw_faceted(frame, door=door, auto_title=auto_title)
    assert axes.shape == (2, 2)
    assert len([key for key in stats if isinstance(key, tuple)]) == 4
    # The Figure owns the one legitimate selection subtitle fit, if enabled;
    # per-cell title measurements must be zero in *both* settings.
    assert len(measured) == (1 if auto_title else 0), measured
    assert (fig._suptitle is not None) is auto_title
    if auto_title:
        assert 'keep' in fig._suptitle.get_text()


def test_AT04_nonfacet_vector_title_is_preserved(frame):
    fig, ax, _ = DFDraw(frame).profile(
        '[y0,y1]:x', selection='keep', bins=4)
    # The historical vector default remains an automatic title.
    assert ax.get_title() or any(t.get_text() for t in ax.texts)


@pytest.mark.parametrize('auto_title', [False, True])
def test_GM01_GM02_global_group_identity_once_at_figure_level(frame, auto_title):
    fig, axes, stats = draw_faceted(frame, auto_title=auto_title)
    text = '\n'.join(
        ([fig._suptitle.get_text()] if fig._suptitle else [])
        + [legend.get_title().get_text() for legend in fig.legends]
    )
    assert text.count('g') >= 1, text
    # A plain variable named g must not accidentally match 'group' or 'global'.
    # The original variable must appear as a group identifier, exactly once.
    import re
    labels = re.findall(r'(?<![A-Za-z_])g(?![A-Za-z_])', text)
    assert len(labels) == 1, text
    assert '__dfdraw_group_scope__' not in text
    assert (fig._suptitle is not None) is auto_title
    if auto_title:
        assert 'keep' in fig._suptitle.get_text()


def test_GM02_explicit_title_is_not_modified(frame):
    fig, axes, stats = draw_faceted(frame, auto_title=True, title='Custom Figure')
    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == 'Custom Figure'


def test_AT03_exact_3x5_singleton_std_has_no_discarded_titles(monkeypatch):
    """Real Stage F composition: two markers x five physical groups x 3x5."""
    rows = []
    for fr in range(3):
        for fc in range(5):
            for group in range(5):
                for xb in range(4):
                    for k in range(7):
                        rows.append(dict(
                            fr=fr + .2, fc=fc + .2,
                            physical_group=-2 + group + .01 * k,
                            x=xb + .5 + .001 * k,
                            y0=.1 * fr + xb * .2 + group * .3 + k * .04,
                            y1=.1 * fc - xb * .1 + group * .1 + k * .06,
                            keep=True,
                        ))
    data = pd.DataFrame(rows)
    calls = []
    monkeypatch.setattr(_auto_title, 'fit_auto_title_text',
                        lambda text, *args, **kw: calls.append(text) or text)
    fig, axes, stats = DFDraw(data).profile(
        '[y0,y1]:x',
        selection='keep', facet_by=['fr', 'fc'], facet_by_bins=[3, 5],
        group_by='physical_group', group_by_bins=5, group_by_scope='global',
        stat_list=['std'], group_style='color', vector_style='marker',
        linestyle='none', legend='shared', bins=4, range=(0, 4),
        min_entries=4, auto_title=True,
        figure_margins=[.15, .96, .80, .12])
    assert axes.shape == (3, 5)
    assert len([k for k in stats if isinstance(k, tuple)]) == 15
    assert len(calls) == 1, calls
    assert fig.legends[0].get_title().get_text() == (
        'Colors: physical_group (5 global bins)')
    assert '__dfdraw_group_scope__' not in fig.legends[0].get_title().get_text()


def test_AT_style_auto_title_does_not_reenable_inner_cell_titles(frame, monkeypatch):
    from dfextensions.dfdraw.style import get_style, set_style
    old = get_style()
    calls = []
    monkeypatch.setattr(_auto_title, 'fit_auto_title_text',
                        lambda text, *args, **kw: calls.append(text) or text)
    try:
        set_style({'auto_title': True})
        fig, axes, _ = draw_faceted(frame, auto_title=False)
        assert not calls, calls
        assert fig._suptitle is None
    finally:
        set_style(old)


def test_GM03_local_group_mode_does_not_claim_global_bins(frame):
    fig, axes, stats = draw_faceted(frame, group_by_scope='facet',
                                   group_by_bins=3, auto_title=False)
    assert all('global bins' not in legend.get_title().get_text()
               for legend in fig.legends)
    assert fig._suptitle is None


def test_GM03_categorical_global_groups_are_not_mislabeled_as_bins(frame):
    data = frame.copy()
    data['category'] = np.where(data['g'] < 1, 'A', 'B')
    obj = DFDraw(data)
    fig, axes, _ = obj.profile(
        '[y0,y1]:x', facet_by=['fr', 'fc'], facet_by_bins=[2, 2],
        group_by='category', group_by_scope='global',
        bins=4, min_entries=2, legend='shared', auto_title=False)
    assert fig.legends
    assert fig.legends[0].get_title().get_text() == (
        'Colors: category (2 global groups)')
