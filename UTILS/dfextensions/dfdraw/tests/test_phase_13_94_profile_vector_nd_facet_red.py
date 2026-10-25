"""PHASE_13_94_DF — independent RED-first profile vector × N-D facet oracles.

Source: PHASE_13_94_DF proposal v0.2, R1–R12, 2026-10-25.
Approval: GPT21.dfdraw consolidated v0.1 review authorizes RED-first tests;
product changes await explicit Architect group-edge Option A/B ratification.

The NEW N-D vector-profile surface is tested against independent row
membership and NumPy sample statistics. It must retain facet -> vector ->
physical group identities; legacy scalar facet semantics are not modified.

This file does not mark expected product failures as xfail: it MUST be RED
against 4cca8963 and 13d28b81 before product implementation starts.
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


X_EDGES = np.asarray([-0.5, 0.5, 1.5, 2.5, 3.5], dtype=float)
X_RANGE = (-0.5, 3.5)
VECTOR = "[y0,y1]:x"
RESIDUAL = "[y0-ref0,y1-ref1]:x"


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _categorical_data(*, missing_cell=False):
    """Injected truth: different means/variances in each logical branch."""
    records = []
    for ri, row in enumerate(("R0", "R1")):
        for ci, col in enumerate(("C0", "C1", "C2")):
            if missing_cell and (ri, ci) == (1, 2):
                continue
            for gi, grp in enumerate(("G0", "G1", "G2")):
                for xi in range(4):
                    for repeat in range(6):
                        y0 = 12 * ri + 3 * ci + 2 * gi + xi + repeat * 0.23
                        y1 = 100 + 23 * ri + 11 * ci + 5 * gi + 3 * xi + repeat * 0.71
                        ref0 = 1 + ci * 0.4 + 0.07 * repeat
                        ref1 = 2 + ri * 0.5 + 0.05 * repeat
                        records.append(dict(
                            row=row, col=col, g=grp, x=float(xi),
                            y0=y0, y1=y1, ref0=ref0, ref1=ref1,
                            residual0=y0 - ref0, residual1=y1 - ref1,
                            keep=repeat != 5,
                        ))
    return pd.DataFrame.from_records(records)


def _continuous_data(*, empty_cell=False):
    """3×5 N-D grid with deliberately unequal group ranges per cell."""
    records = []
    for ri in range(3):
        for ci in range(5):
            if empty_cell and (ri, ci) == (2, 4):
                continue
            for gi in range(5):
                # One intermediate physical group absent in a selected cell.
                if (ri, ci) == (1, 3) and gi == 2:
                    continue
                for xi in range(4):
                    for rep in range(6):
                        group_value = 3 * gi + ri * 1.4 + ci * .6 + .03 * rep
                        y0 = 2 * ri + 6 * ci + gi + 2 * xi + rep * .2
                        y1 = 40 + 7 * ri + 3 * ci - gi + 4 * xi + rep * .3
                        records.append(dict(
                            rate=float(ri) + .1, occupancy=float(ci * 10) + .2,
                            fTgl=group_value, x=float(xi), y0=y0, y1=y1,
                            ref0=1. + .08 * rep, ref1=2. + .04 * rep,
                            keep=(rep < 5),
                        ))
    return pd.DataFrame.from_records(records)


def _call(df, expr=VECTOR, *, door="typed", **kwargs):
    obj = DFDraw(df)
    kwargs.setdefault("bins", 4)
    kwargs.setdefault("range", X_RANGE)
    kwargs.setdefault("min_entries", 1)
    kwargs.setdefault("return_data", True)
    if door == "typed":
        return obj.profile(expr, **kwargs)
    return obj.draw(expr, type="profile", **kwargs)


def _grid(stats, axes, shape):
    """Assert the public N-D stats grid is keyed by facet coordinates."""
    assert isinstance(axes, np.ndarray) and axes.shape == shape
    assert isinstance(stats, dict), f"N-D facet coordinates lost: {type(stats).__name__}"
    cells = {key: value for key, value in stats.items() if isinstance(key, tuple)}
    assert len(cells) == shape[0] * shape[1], (len(cells), shape)
    return cells


def _branches(cell, n=2):
    """Facet -> ordered vector branches; never flatten into one profile."""
    assert isinstance(cell, (list, tuple)), (
        "Each populated facet must retain the ordered vector branches, "
        f"got {type(cell).__name__}: {str(cell)[:160]}"
    )
    assert len(cell) == n
    assert all(isinstance(b, dict) and "profile_data" in b for b in cell)
    return cell


def _compare_rows(table, truth, value_column, *, group=None):
    """Independent x-bin truth: row counts, mean, SEM and bin coordinates."""
    if group is not None:
        assert "group" in table.columns
        table = table[table["group"].astype(str) == str(group)]
    assert len(table) == 4
    for xi, center in enumerate((0., 1., 2., 3.)):
        t = table.iloc[xi]
        assert float(t["x_low"]) == pytest.approx(X_EDGES[xi], abs=1e-13)
        assert float(t["x_high"]) == pytest.approx(X_EDGES[xi + 1], abs=1e-13)
        assert float(t["x_center"]) == pytest.approx(center, abs=1e-13)
        vals = truth.loc[truth["x"] == center, value_column].to_numpy(dtype=float)
        assert int(t["count"]) == len(vals), (group, center)
        if len(vals) >= 2:
            assert float(t["y_mean"]) == pytest.approx(float(np.mean(vals)), abs=1e-12)
            assert float(t["y_sem"]) == pytest.approx(
                float(np.std(vals, ddof=1) / math.sqrt(len(vals))), abs=1e-12)


def _sorted_profile(table):
    return table.sort_values([col for col in ("group", "x_center")
                              if col in table], kind="stable").reset_index(drop=True)


@pytest.mark.parametrize("door", ["typed", "draw"])
def test_R1_two_branches_in_each_2d_facet_without_group(door):
    df = _categorical_data()
    fig, axes, stats = _call(df, door=door, facet_by=["row", "col"])
    cells = _grid(stats, axes, (2, 3))
    for (row, col), cell in cells.items():
        subset = df[(df.row == row) & (df.col == col)]
        for branch, field in zip(_branches(cell), ("y0", "y1")):
            _compare_rows(_sorted_profile(branch["profile_data"]), subset, field)
    assert len(fig.axes) == 6


def test_R2_computed_vector_residuals_equal_materialized_truth():
    df = _categorical_data()
    fig, axes, stats = _call(df, RESIDUAL, facet_by=["row", "col"])
    cells = _grid(stats, axes, (2, 3))
    for (row, col), cell in cells.items():
        part = df[(df.row == row) & (df.col == col)]
        for b, field in zip(_branches(cell), ("residual0", "residual1")):
            _compare_rows(_sorted_profile(b["profile_data"]), part, field)
    assert fig is axes[0, 0].figure


def test_R3_numeric_3x5_facet_classification_matches_global_pandas_membership():
    df = _continuous_data()
    fig, axes, stats = _call(df, facet_by=["rate", "occupancy"],
                             facet_by_bins=[3, 5], selection="keep")
    cells = _grid(stats, axes, (3, 5))
    selected = df[df.keep]
    rows = pd.cut(selected.rate, bins=3)
    cols = pd.cut(selected.occupancy, bins=5)
    for (row, col), cell in cells.items():
        mask = (rows == row) & (cols == col)
        part = selected.loc[mask]
        for branch, field in zip(_branches(cell), ("y0", "y1")):
            _compare_rows(_sorted_profile(branch["profile_data"]), part, field)
    assert fig.axes[:15] == list(axes.flat)


def test_R4_categorical_group_composes_2_vectors_times_3_groups():
    df = _categorical_data()
    _, axes, stats = _call(df, facet_by=["row", "col"], group_by="g")
    for (row, col), cell in _grid(stats, axes, (2, 3)).items():
        part = df[(df.row == row) & (df.col == col)]
        for branch, field in zip(_branches(cell), ("y0", "y1")):
            tab = _sorted_profile(branch["profile_data"])
            assert set(tab.group.astype(str)) == {"G0", "G1", "G2"}
            for grp in ("G0", "G1", "G2"):
                _compare_rows(tab, part[part.g == grp], field, group=grp)


def _interval_label(interval):
    """Independent label expectation for existing public profile interval UI.

    Membership is checked from independent pandas cut/qcut, not this label.
    """
    width = interval.right - interval.left
    fmt = ".0f" if width >= 10 else ".1f" if width >= 1 else (
        ".2f" if width >= .1 else ".3f")
    return f"{interval.left:{fmt}}-{interval.right:{fmt}}"


def _assert_global_group_tables(cells, selected, classified_groups):
    """Each physical interval has identical label and membership in all cells."""
    row_groups = pd.cut(selected.rate, bins=3)
    col_groups = pd.cut(selected.occupancy, bins=5)
    categories = classified_groups.cat.categories
    assert len(categories) >= 3
    for (r, c), cell in cells.items():
        facet_mask = (row_groups == r) & (col_groups == c)
        for branch, field in zip(_branches(cell), ("y0", "y1")):
            table = _sorted_profile(branch["profile_data"])
            actual_labels = set(table["group"].dropna().astype(str))
            expected_labels = set()
            for interval in categories:
                label = _interval_label(interval)
                truth_mask = facet_mask & (classified_groups == interval)
                truth = selected.loc[truth_mask]
                if truth.empty:
                    # Sparse cells may omit this group or retain zero-count
                    # placeholder rows, but may not produce wrong data.
                    present = table[table["group"].astype(str) == label]
                    assert present.empty or int(present["count"].sum()) == 0
                    continue
                expected_labels.add(label)
                assert label in actual_labels, (
                    f"Physical group {interval} incorrectly renamed/rebinned "
                    f"inside facet {(r,c)}; labels={sorted(actual_labels)}")
                _compare_rows(table, truth, field, group=label)
            assert expected_labels.issubset(actual_labels)
            assert int(table["count"].sum()) == int(facet_mask.sum())


def test_R5_global_group_bins_maintain_physical_identity_across_cells():
    """Option A proposed by Main Reviewer; requires Architect ratification."""
    df = _continuous_data()
    _, axes, stats = _call(df, facet_by=["rate", "occupancy"],
                           facet_by_bins=[3, 5], group_by="fTgl",
                           group_by_bins=5, selection="keep")
    selected = df[df.keep]
    global_groups = pd.cut(selected.fTgl, bins=5)
    _assert_global_group_tables(_grid(stats, axes, (3, 5)),
                                selected, global_groups)


def test_R6_group_quantile_boundaries_resolved_globally():
    df = _continuous_data()
    _, axes, stats = _call(df, facet_by=["rate", "occupancy"],
                           facet_by_bins=[3, 5], group_by="fTgl",
                           group_by_quantiles=3, selection="keep")
    selected = df[df.keep]
    global_groups = pd.qcut(selected.fTgl, q=3, duplicates="drop")
    assert len(global_groups.cat.categories) == 3
    _assert_global_group_tables(_grid(stats, axes, (3, 5)),
                                selected, global_groups)


def test_R7_vector_group_identity_and_single_figure_legend():
    df = _categorical_data()
    fig, axes, stats = _call(df, facet_by=["row", "col"], group_by="g",
                             auto_title=True)
    _grid(stats, axes, (2, 3))
    # Cell labels belong to N-D coordinator, not the inner vector renderer.
    assert all("[y0,y1]" not in ax.get_title() for ax in axes.flat)
    from matplotlib.legend import Legend
    assert all(ax.get_legend() is None for ax in axes.flat)
    assert all(not any(isinstance(a, Legend) for a in ax.get_children())
               for ax in axes.flat), 'an ax.add_artist secondary legend leaked'
    assert all(not any(t.get_position()[1] >= 1 for t in ax.texts
                       if t.get_transform() == ax.transAxes)
               for ax in axes.flat), 'an inner auto-title subtitle leaked'
    assert len(fig.legends) == 1
    texts = [t.get_text() for leg in fig.legends for t in leg.get_texts()]
    assert any("y0" in t for t in texts) and any("y1" in t for t in texts)
    assert all(any(grp in t for t in texts) for grp in ("G0", "G1", "G2"))
    # Identity remains recoverable even in the last Figure cell.
    assert len(axes[-1, -1].lines) >= 6


def test_R8_strong_numeric_decomposition_with_selection_and_all_coordinates():
    df = _categorical_data()
    _, axes, stats = _call(df, RESIDUAL, facet_by=["row", "col"],
                           group_by="g", selection="keep", error="sem")
    selected = df[df.keep]
    for (row, col), cell in _grid(stats, axes, (2, 3)).items():
        for branch, field in zip(_branches(cell), ("residual0", "residual1")):
            for g in ("G0", "G1", "G2"):
                truth = selected[(selected.row == row) & (selected.col == col)
                                 & (selected.g == g)]
                _compare_rows(_sorted_profile(branch["profile_data"]),
                              truth, field, group=g)


def test_R9_missing_facet_cell_preserves_empty_cell_schema():
    df = _categorical_data(missing_cell=True)
    fig, axes, stats = _call(df, facet_by=["row", "col"], group_by="g")
    cells = _grid(stats, axes, (2, 3))
    assert cells[("R1", "C2")].get("empty") is True
    assert cells[("R1", "C2")].get("n") == 0
    assert any("no data" in t.get_text().lower() for t in axes[1, 2].texts)
    assert len(_branches(cells[("R0", "C2")])) == 2
    assert fig.axes[5] is axes[1, 2]


def test_R10_forwarding_nondefault_min_entries_50_masks_short_profiles():
    df = _categorical_data()
    fig, axes, stats = _call(df, facet_by=["row", "col"],
                             group_by="g", bins=10, min_entries=50,
                             range=X_RANGE)
    _grid(stats, axes, (2, 3))
    # Every (cell, group, x-bin) contains at most six rows (< 50), so
    # no central profile may contain rendered data points.  Count remains
    # available in return_data; masking must not suppress the statistics.
    for ax in axes.flat:
        for line in ax.lines:
            assert len(line.get_xdata()) == 0 or len(line.get_ydata()) == 0
    assert fig is axes[0, 0].figure


def test_R11_typed_and_public_draw_nested_stats_equivalent():
    df = _categorical_data()
    kwargs = dict(facet_by=["row", "col"], group_by="g")
    ft, at, st = _call(df, door="typed", **kwargs)
    fd, ad, sd = _call(df, door="draw", **kwargs)
    tc = _grid(st, at, (2, 3))
    dc = _grid(sd, ad, (2, 3))
    assert tc.keys() == dc.keys()
    for key in tc:
        for lhs, rhs in zip(_branches(tc[key]), _branches(dc[key])):
            pd.testing.assert_frame_equal(
                _sorted_profile(lhs["profile_data"]),
                _sorted_profile(rhs["profile_data"]),
                check_exact=False, atol=1e-13, rtol=1e-13,
            )
    assert ft is at[0, 0].figure and fd is ad[0, 0].figure


def test_R12_existing_one_dimensional_profile_vector_facet_still_works():
    df = _categorical_data()
    fig, axes, stats = _call(df, facet_by="row", group_by="g")
    assert len(stats) == 2
    assert all(branch["faceted"] is True for branch in stats)
    assert all(set(branch["groups"]) == {"R0", "R1"} for branch in stats)
    assert fig is np.asarray(axes).flat[0].figure


@pytest.mark.parametrize("kind,expr", [("hist", "[y0,y1]"),
                                       ("scatter", VECTOR)])
def test_R13_neighbor_hist_scatter_nd_vector_still_refuse_loudly(kind, expr):
    df = _categorical_data()
    with pytest.raises(NotImplementedError, match="expression-vector.*facet"):
        DFDraw(df).draw(expr, type=kind, facet_by=["row", "col"])


def test_R14_existing_scalar_nd_facet_is_not_changed_by_new_global_policy():
    df = _categorical_data()
    fig, axes, stats = _call(df, expr="y0:x", facet_by=["row", "col"],
                             group_by="g")
    assert axes.shape == (2, 3)
    assert len([k for k in stats if isinstance(k, tuple)]) == 6
    assert fig is axes[0, 0].figure


def test_R15_missing_intermediate_group_keeps_global_color_marker_and_label():
    """The group's plotted marker as well as its colour cannot shift by cell."""
    df = _continuous_data()
    fig, axes, stats = _call(df, facet_by=['rate', 'occupancy'],
                             facet_by_bins=[3, 5], group_by='fTgl',
                             group_by_bins=5, selection='keep')
    ref = {}
    for ax in axes.flat:
        for container in ax.containers:
            handle = container.lines[0]
            if handle is None:
                continue
            label = container.get_label()
            ident = (tuple(np.asarray(handle.get_color(), dtype=float)),
                     handle.get_marker())
            if label in ref:
                assert ident == ref[label], f"physical group {label} changed style"
            else:
                ref[label] = ident
    assert len(ref) == 5
    assert len(fig.legends) == 1
    plt.close(fig)


@pytest.mark.parametrize('restricted', [
    {'normalize': 'delta'},
    {'selection_vector': ['keep', '~keep']},
    {'weights_vector': ['w0', 'w1']},
    {'same': True},
])
def test_R16_new_nd_vector_profile_neighbor_refuses_before_creating_figure(
        restricted, monkeypatch):
    """Refusal must not create a Figure, even if other tests left one open.

    In a full pytest-xdist worker, pyplot can retain Figures from unrelated
    earlier tests. Check *creation by this request*, not global emptiness.
    A pyplot.figure spy also catches a Figure that the rejected path might
    create and then close before the assertion.
    """
    df = _categorical_data()
    existing_figures = tuple(plt.get_fignums())
    original_figure = plt.figure
    figure_calls = []

    def record_figure(*args, **kwargs):
        figure_calls.append(True)
        return original_figure(*args, **kwargs)

    monkeypatch.setattr(plt, 'figure', record_figure)
    with pytest.raises((NotImplementedError, ValueError)):
        _call(df, facet_by=['row', 'col'], **restricted)
    assert not figure_calls, 'rejected composition attempted to create a Figure'
    assert tuple(plt.get_fignums()) == existing_figures, (
        'rejected composition changed the set of existing Figures'
    )


def test_R17_figure_channel_assignment_is_resolved_once(monkeypatch):
    from dfextensions.dfdraw import channels
    original = channels.assign_channels
    calls = []
    def counted(channels_arg):
        calls.append(tuple(c.name for c in channels_arg))
        return original(channels_arg)
    monkeypatch.setattr(channels, 'assign_channels', counted)
    df = _continuous_data()
    _call(df, facet_by=['rate', 'occupancy'], facet_by_bins=[3, 5],
          group_by='fTgl', group_by_bins=5)
    assert len(calls) == 1, f'channel identity re-resolved per panel: {calls}'


def test_R18_explicit_group_color_override_remains_authoritative():
    """The new global color plan must not overwrite an explicit user color."""
    df = _categorical_data()
    fig, axes, _ = _call(df, facet_by=['row', 'col'], group_by='g',
                         color='magenta')
    for ax in axes.flat:
        for container in ax.containers:
            central = container.lines[0]
            if central is not None:
                assert central.get_color() == 'magenta'
    assert len(fig.legends) == 1


def test_R19_group_bins_without_group_column_fail_before_figure():
    df = _categorical_data()
    with pytest.raises(ValueError, match='require group_by'):
        _call(df, facet_by=['row', 'col'], group_by_bins=5)
    assert not plt.get_fignums()
