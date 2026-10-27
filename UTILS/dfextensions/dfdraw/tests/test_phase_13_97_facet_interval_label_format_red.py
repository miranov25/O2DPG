"""PHASE_13_97_DF v0.2 — RED-first scientific facet interval label formatting.

These tests freeze the v0.2 contract:
- display-only 2-D numeric facet formatting;
- one public facet_label_format parameter and facet.label_format style key;
- omitted != explicit None;
- atomic mapping replacement, canonical full form and shortcut equivalence;
- collision-safe auto formatting;
- legacy AD-3/group identity and statistical payload invariance.

Do not xfail product REDs: this module must visibly fail before Stage A production code.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfdraw import DFDraw, DEFAULT_STYLE, get_style, set_style, save_style, load_style
from dfdraw.plots.profile import _format_interval_label


@pytest.fixture(autouse=True)
def _isolate_style_and_figures():
    set_style(None)
    plt.close("all")
    yield
    set_style(None)
    plt.close("all")


def _data(*, narrow=False):
    rows = []
    row_centres = ([0.0005, 0.0015, 0.0025] if narrow
                   else [3000.0, 9000.0, 15000.0])
    col_centres = [-0.75, -0.25, 0.25, 0.75, 1.25]
    for ri, rv in enumerate(row_centres):
        for ci, cv in enumerate(col_centres):
            for g in ("G0", "G1"):
                for xi in range(3):
                    for rep in range(4):
                        # Small deterministic spread keeps all bins populated.
                        r = rv + (rep - 1.5) * (1e-5 if narrow else 5.0)
                        c = cv + (rep - 1.5) * 0.002
                        rows.append(dict(
                            rowv=float(r), colv=float(c), third=float(ri % 2),
                            group=g, x=float(xi),
                            y=float(10 * ri + 3 * ci + xi + rep * .2 + (g == "G1")),
                        ))
    return pd.DataFrame(rows)


def _call(df, *, fmt="__OMIT__", row_bins=3, col_bins=5, door="typed", **extra):
    obj = DFDraw(df)
    kwargs = dict(
        facet_by=["rowv", "colv"], facet_by_bins=[row_bins, col_bins],
        facet_labels="outer", bins=3, min_entries=1, return_data=True,
    )
    kwargs.update(extra)
    if fmt != "__OMIT__":
        kwargs["facet_label_format"] = fmt
    if door == "draw":
        return obj.draw("y:x", type="profile", **kwargs)
    return obj.profile("y:x", **kwargs)


def _interval_headings(fig):
    return [t.get_text() for t in fig.texts
            if t.get_text().startswith(("(", "["))]


def _row_col_categories(df, row_bins=3, col_bins=5):
    rows = list(pd.cut(df["rowv"], bins=row_bins).cat.categories)
    cols = list(pd.cut(df["colv"], bins=col_bins).cat.categories)
    return rows, cols


def _expected_legacy(df, row_bins=3, col_bins=5):
    rows, cols = _row_col_categories(df, row_bins, col_bins)
    return [str(v) for v in cols] + [str(v) for v in rows]


def _expected_explicit(df, col_spec=None, row_spec=None, row_bins=3, col_bins=5):
    rows, cols = _row_col_categories(df, row_bins, col_bins)
    def one(interval, spec):
        if spec is None:
            return str(interval)
        left = format(0.0 if float(interval.left) == 0 else float(interval.left), spec)
        right = format(0.0 if float(interval.right) == 0 else float(interval.right), spec)
        lb = "[" if interval.closed in ("left", "both") else "("
        rb = "]" if interval.closed in ("right", "both") else ")"
        return f"{lb}{left}, {right}{rb}"
    return ([one(v, col_spec) for v in cols] +
            [one(v, row_spec) for v in rows])


def _positions(axes):
    return [(a.get_position().x0, a.get_position().y0,
             a.get_position().x1, a.get_position().y1)
            for a in axes.flat]


def _compare_stats(a, b):
    """Recursive equality for the public stats payload used by this test."""
    assert type(a) is type(b)
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                      check_dtype=True, check_exact=True)
    elif isinstance(a, dict):
        assert list(a.keys()) == list(b.keys())
        for key in a:
            _compare_stats(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _compare_stats(x, y)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, float) and np.isnan(a):
        assert isinstance(b, float) and np.isnan(b)
    else:
        assert a == b


# ---------------------------------------------------------------------------
# R01-R10 — scientific/display contract
# ---------------------------------------------------------------------------

def test_R01_omitted_baseline_reproduces_long_pandas_3x5_headings():
    df = _data()
    fig, axes, _ = _call(df)
    assert axes.shape == (3, 5)
    assert _interval_headings(fig) == _expected_legacy(df)
    assert any(len(s) >= 18 for s in _interval_headings(fig))


def test_R02_explicit_none_and_legacy_are_exact_historical_display():
    df = _data()
    expected = _expected_legacy(df)
    for fmt in (None, "legacy"):
        fig, _, _ = _call(df, fmt=fmt)
        assert _interval_headings(fig) == expected
        plt.close(fig)


def test_R03_explicit_numeric_formats_control_row_and_column_dimensions():
    df = _data()
    fig, _, _ = _call(
        df,
        fmt={"format": ".3g", "by": {"colv": ".2f"}},
    )
    labels = _interval_headings(fig)
    assert labels[:5] == _expected_explicit(df, col_spec=".2f", row_spec=".3g")[:5]
    # row headings use the general .3g policy, not raw pandas precision.
    assert labels == _expected_explicit(df, col_spec=".2f", row_spec=".3g")
    assert labels[5:] != _expected_legacy(df)[5:]


def test_R04_plain_per_dimension_mapping_leaves_unspecified_dimension_legacy():
    df = _data()
    fig, _, _ = _call(df, fmt={"colv": ".2f"})
    labels = _interval_headings(fig)
    assert labels[:5] == _expected_explicit(df, col_spec=".2f")[:5]
    assert labels[5:] == _expected_legacy(df)[5:]


def test_R05_auto_is_collision_safe_for_nearby_distinct_endpoints():
    df = _data(narrow=True)
    fig, _, _ = _call(df, fmt="auto", row_bins=3)
    labels = _interval_headings(fig)[5:]
    assert len(labels) == 3
    assert len(set(labels)) == 3
    assert all(s.startswith("(") and s.endswith("]") for s in labels)
    # Auto must not silently collapse these small distinct boundaries to 0.00.
    endpoints = []
    for label in labels:
        endpoints.extend(label[1:-1].split(", "))
    assert len(set(endpoints)) >= 4


def test_R06_underprecise_explicit_format_and_bad_dimension_fail_before_figure():
    df = _data(narrow=True)
    before = tuple(plt.get_fignums())
    with pytest.raises(ValueError, match="collision|distinct|precision"):
        _call(df, fmt={"rowv": ".2f"})
    assert tuple(plt.get_fignums()) == before
    with pytest.raises(ValueError, match="unknown|facet"):
        _call(df, fmt={"not_a_facet": ".3g"})
    assert tuple(plt.get_fignums()) == before


def test_R07_formatting_changes_no_stats_keys_payload_or_group_colors():
    df = _data()
    fig0, ax0, st0 = _call(df, group_by="group", group_by_scope="global")
    colors0 = [[line.get_color() for line in ax.lines] for ax in ax0.flat]
    fig1, ax1, st1 = _call(df, fmt="auto", group_by="group", group_by_scope="global")
    colors1 = [[line.get_color() for line in ax.lines] for ax in ax1.flat]
    _compare_stats(st0, st1)
    assert colors0 == colors1
    assert set(k for k in st0 if isinstance(k, tuple)) == set(
        k for k in st1 if isinstance(k, tuple))
    plt.close(fig0); plt.close(fig1)


def test_R08_auto_preserves_interval_brackets_signs_and_zero_crossing():
    df = _data()
    fig, _, _ = _call(df, fmt={"colv": "auto"})
    labels = _interval_headings(fig)[:5]
    assert all(s.startswith("(") and s.endswith("]") for s in labels)
    assert any("-" in s for s in labels)
    assert any("0" in s for s in labels)


def test_R09_formatting_keeps_user_axes_geometry_identical():
    df = _data()
    margins = [0.20, 0.97, 0.80, 0.11]
    spacing = [0.20, 0.25]
    fig0, ax0, _ = _call(df, figure_margins=margins, subplot_spacing=spacing)
    fig1, ax1, _ = _call(df, fmt="auto", figure_margins=margins, subplot_spacing=spacing)
    assert _positions(ax0) == pytest.approx(_positions(ax1), abs=1e-14)
    assert _interval_headings(fig1) != _interval_headings(fig0)
    plt.close(fig0); plt.close(fig1)


@pytest.mark.parametrize("door", ["typed", "draw"])
def test_R10_typed_profile_and_draw_forward_same_policy(door):
    df = _data()
    fig, axes, _ = _call(df, fmt={"colv": ".2f"}, door=door)
    assert axes.shape == (3, 5)
    assert _interval_headings(fig)[:5] == _expected_explicit(df, col_spec=".2f")[:5]


# ---------------------------------------------------------------------------
# C01-C09 — configuration / normalization / ownership contract
# ---------------------------------------------------------------------------

def test_C01_style_key_registered_and_save_load_roundtrips_structured_value():
    assert "facet.label_format" in DEFAULT_STYLE
    assert DEFAULT_STYLE["facet.label_format"] is None
    value = {"format": "auto", "by": {"colv": ".2f"}}
    set_style({"facet.label_format": value})
    assert get_style()["facet.label_format"] == value
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "style.json"
        save_style(path)
        set_style(None)
        load_style(path)
        assert get_style()["facet.label_format"] == value


def test_C02_omitted_inherits_style_but_explicit_none_forces_legacy():
    df = _data()
    set_style({"facet.label_format": "auto"})
    fig_style, _, _ = _call(df)
    fig_none, _, _ = _call(df, fmt=None)
    assert _interval_headings(fig_style) != _expected_legacy(df)
    assert _interval_headings(fig_none) == _expected_legacy(df)


def test_C03_explicit_scalar_wins_over_active_style_mapping():
    df = _data()
    set_style({"facet.label_format": {"format": "auto", "by": {"colv": ".1f"}}})
    fig, _, _ = _call(df, fmt=".2f")
    assert _interval_headings(fig) == _expected_explicit(
        df, col_spec=".2f", row_spec=".2f")


def test_C04_explicit_plain_mapping_replaces_style_atomically():
    df = _data()
    set_style({"facet.label_format": {"format": ".3g", "by": {"rowv": ".1f"}}})
    fig, _, _ = _call(df, fmt={"colv": ".2f"})
    labels = _interval_headings(fig)
    assert labels[:5][0] == _expected_explicit(df, col_spec=".2f")[0]
    assert labels[5:] == _expected_legacy(df)[5:]


def test_C05_shortcuts_and_canonical_full_form_are_equivalent():
    df = _data()
    fig1, _, _ = _call(df, fmt="auto")
    fig2, _, _ = _call(df, fmt={"format": "auto", "by": {}})
    assert _interval_headings(fig1) == _interval_headings(fig2)
    fig3, _, _ = _call(df, fmt={"colv": ".2f"})
    fig4, _, _ = _call(df, fmt={"format": "legacy", "by": {"colv": ".2f"}})
    assert _interval_headings(fig3) == _interval_headings(fig4)


@pytest.mark.parametrize("bad", [
    {"format": "auto", "colv": ".2f"},
    {"format": "auto", "by": {}, "extra": 1},
    {"format": object(), "by": {}},
    {"format": "auto", "by": {"colv": object()}},
])
def test_C06_ambiguous_or_invalid_structured_policies_refuse_before_figure(bad):
    df = _data()
    before = tuple(plt.get_fignums())
    with pytest.raises((TypeError, ValueError)):
        _call(df, fmt=bad)
    assert tuple(plt.get_fignums()) == before


def test_C07_existing_AD3_group_interval_formatter_is_byte_identical():
    interval = pd.Interval(0.0, 0.5, closed="right")
    assert _format_interval_label(interval) == "0.00-0.50"
    interval2 = pd.Interval(10.0, 20.0, closed="right")
    assert _format_interval_label(interval2) == "10-20"


def test_C08_style_state_resets_without_cross_test_leakage():
    assert get_style()["facet.label_format"] is None
    set_style({"facet.label_format": "auto"})
    assert get_style()["facet.label_format"] == "auto"
    set_style(None)
    assert get_style()["facet.label_format"] is None


def test_C09_nonlegacy_explicit_policy_rejects_unsupported_1d_and_3d_facets():
    df = _data()
    obj = DFDraw(df)
    before = tuple(plt.get_fignums())
    with pytest.raises((ValueError, NotImplementedError), match="two|2-D|dimension|facet"):
        obj.profile("y:x", facet_by=["rowv"], facet_by_bins=[3],
                    facet_label_format="auto", bins=3, min_entries=1)
    assert tuple(plt.get_fignums()) == before
    with pytest.raises((ValueError, NotImplementedError), match="two|2-D|dimension|facet"):
        obj.profile("y:x", facet_by=["rowv", "colv", "third"],
                    facet_by_bins=[3, 5, 2], facet_label_format="auto",
                    bins=3, min_entries=1)
    assert tuple(plt.get_fignums()) == before
    # Explicit legacy on 1-D remains the historical path rather than an error.
    fig, axes, _ = obj.profile("y:x", facet_by=["rowv"], facet_by_bins=[3],
                               facet_label_format=None, bins=3, min_entries=1)
    assert fig is not None
    assert len(np.atleast_1d(axes)) == 3
