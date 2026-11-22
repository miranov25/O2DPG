"""Regression tests for BUG_AliasDataFrame_20261116 (ADF-EXT-001 / POST76-010, and POST76-002).

The bug. A subframe can be joined on keys with different names (parent `posTrackIdx`, child
`rowId`, registered with `right_index_columns=`). Alias evaluation and draw_batch() honoured the
child key. draw() and draw_figures() had their own pandas merge on the PARENT key names, so a
direct reference like draw("Pos.pt") raised KeyError, or -- when the child happened to have a
column named like the parent key -- silently drew wrong values. The same merge ignored the
configured subframe fill and silently dropped unmatched parent rows (POST76-002).

The fix. draw() and draw_figures() use the gather draw_batch() already used
(_compute_join_indices + _extract_subframe_values_cached), through _gather_direct_subframe_ref.

How the tests check values. Every test draws through the public API, reads the result back from
the matplotlib figure, and compares it with numbers that can be checked by hand from the tiny
tables in this file. Three readers are used:
  _draw_points  scatter "<y>:<x>"  -> sorted list of ALL drawn (x, y) points; a repeated point
                                      stays in the list twice, so an extra or a missing point
                                      always changes the result (section 0 proves this)
  _draw_groups  scatter + group_by -> the points of every group
  _draw_hist    histogram          -> (left bin edge, height) of every bin

Three code states are named in the docstrings:
  "Unfixed"   AliasDataFrame.py before the fix, MD5 f617c13484f730668b6ad32d6b0a1b27
  "Fix v01"   the first candidate, MD5 ecad80353030fcf80f0ba5e752810187 (not committed); it broke
              draw_figures(max_entries=<negative>) for subframe references
  "Fix v02"   the candidate these tests ship with: every test passes
A docstring that says "Control" names a test that passes on all three states, except where it
states otherwise.
"""
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame  # noqa: E402

SURFACES = ["draw", "draw_figures", "draw_batch"]

# The track table of every V0 test: four tracks.
TRACK_ROW = [0, 1, 2, 3]
TRACK_PT = [10.0, 20.0, 30.0, 40.0]
TRACK_Q = [5, 6, 7, 8]                      # an integer column
TRACK_GOOD = [True, False, True, True]      # a boolean column
# Four V0s; the positive daughter of V0 i is track POS[i]; x is just the V0 number.
POS = [2, 0, 3, 2]
POS_PT = [30.0, 10.0, 40.0, 30.0]           # TRACK_PT[POS], by hand
POS_Q = [7, 5, 8, 7]                        # TRACK_Q[POS], by hand
V0_X = [0.0, 1.0, 2.0, 3.0]
HIST = {"bins": 4, "range": (-0.5, 3.5)}    # one bin per V0: bin i holds x == i


# ---------------------------------------------------------------------------------------------
# Builders and readers
# ---------------------------------------------------------------------------------------------

def _tracks(key="rowId"):
    return AliasDataFrame(pd.DataFrame({
        key: np.array(TRACK_ROW, np.int64), "pt": TRACK_PT, "q": np.array(TRACK_Q, np.int64),
        "good": np.array(TRACK_GOOD, bool)}))


def _v0(pos=POS, index=None):
    """V0 frame with subframe Pos joined on posTrackIdx (parent) -> rowId (child)."""
    v0 = AliasDataFrame(pd.DataFrame(
        {"posTrackIdx": np.array(pos, np.int64), "x": np.arange(len(pos), dtype=float)}, index=index))
    v0.register_subframe("Pos", _tracks(), index_columns="posTrackIdx", right_index_columns="rowId")
    return v0


def _same_name_v0(pos=POS, child_key=TRACK_ROW, child_pt=TRACK_PT, child_q=TRACK_Q):
    """V0 frame with subframe Pos joined on a key column called posTrackIdx on both sides."""
    v0 = AliasDataFrame(pd.DataFrame(
        {"posTrackIdx": np.array(pos, np.int64), "x": np.arange(len(pos), dtype=float)}))
    child = AliasDataFrame(pd.DataFrame({
        "posTrackIdx": np.array(child_key, np.int64), "pt": child_pt, "q": np.array(child_q, np.int64),
        "good": np.array((TRACK_GOOD + [True] * len(child_key))[:len(child_key)], bool)}))
    v0.register_subframe("Pos", child, index_columns="posTrackIdx")
    return v0


V0_BUILDERS = {"different_key_names": _v0, "same_key_name": _same_name_v0}


def _pts(xs, ys):
    """The expected points: sorted list of (x, y), every point kept, repeated or not."""
    return sorted((float(x), float(y)) for x, y in zip(xs, ys))


def _render(adf, surface, expr, plot_type, plot=None, **call):
    """Draw one plot through one surface and leave the figure current.

    plot -- options of the plot itself (selection, weights, group_by, bins, range)
    call -- options of the call (entry_begin, entry_end, entry_mask, max_entries, lazy)
    """
    plot = dict(plot or {})
    plt.close("all")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if surface == "draw":
            adf.draw(expr, type=plot_type, **plot, **call)
        elif surface == "draw_figures":
            adf.draw_figures([{"plots": [dict(expr=expr, type=plot_type, **plot)]}], verbose=False, **call)
        else:
            res = adf.draw_batch({"p": dict(expr=expr, type=plot_type, **plot)}, verbose=False, **call)
            assert res["_summary"]["failed"] == 0, res.get("_errors")


def _scatter_collections():
    """One sorted (x, y) list per scatter collection of the current figure."""
    return [sorted((float(x), float(y)) for x, y in np.asarray(c.get_offsets()))
            for ax in plt.gcf().axes for c in ax.collections if len(c.get_offsets())]


def _read_points():
    """ALL scatter points of the current figure, sorted. Nothing is merged: N drawn points give
    a list of length N."""
    return sorted(p for coll in _scatter_collections() for p in coll)


def _draw_points(adf, surface, expr, plot=None, **call):
    _render(adf, surface, expr, "scatter", plot, **call)
    points = _read_points()
    plt.close("all")
    return points


def _draw_groups(adf, surface, expr, group_by, **call):
    """The points of every group of a grouped scatter, as a sorted list of point lists."""
    _render(adf, surface, expr, "scatter", {"group_by": group_by}, **call)
    groups = sorted(_scatter_collections())
    plt.close("all")
    return groups


def _draw_hist(adf, surface, expr, plot=None, **call):
    """(left bin edge, height) of every bin of the one histogram drawn with HIST binning."""
    _render(adf, surface, expr, "hist", {**HIST, **(plot or {})}, **call)
    patches = [p for ax in plt.gcf().axes for p in ax.patches]
    plt.close("all")
    nbins = HIST["bins"]
    if len(patches) == 1 and hasattr(patches[0], "get_xy") and not hasattr(patches[0], "get_height"):
        # a step histogram is one polygon: (edge0, 0), then two top vertices per bin
        top = np.asarray(patches[0].get_xy())[1:2 * nbins + 1]
        assert len(top) == 2 * nbins, f"{surface}: unexpected histogram outline {top!r}"
        return [(float(top[2 * i][0]), float(top[2 * i][1])) for i in range(nbins)]
    assert len(patches) == nbins, f"{surface}: expected one histogram of {nbins} bins, got {patches!r}"
    return sorted((float(p.get_x()), float(p.get_height())) for p in patches)


# ---------------------------------------------------------------------------------------------
# 0. The readers themselves: a repeated or an extra point must be visible
#    (review finding F1: the first version of this file compared {x: y} dictionaries, in which
#    a second point with the same x silently replaced the first)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("surface", SURFACES)
def test_reader_keeps_a_repeated_point(surface):
    """Five V0s; V0 0 and V0 4 are the same point (x = 0, daughter track 2, pt 30). Five points
    must be read back, the repeated one twice. A reader that keys the points by x returns four
    and fails here. Control for the product: passes on Unfixed, Fix v01 and Fix v02."""
    v0 = _same_name_v0(pos=[2, 0, 3, 2, 2])
    v0.df["x"] = [0.0, 1.0, 2.0, 3.0, 0.0]
    points = _draw_points(v0, surface, "Pos.pt:x")
    assert len(points) == 5
    assert points == [(0.0, 30.0), (0.0, 30.0), (1.0, 10.0), (2.0, 40.0), (3.0, 30.0)]
    assert points != _pts(V0_X, POS_PT), "a repeated point must not compare equal to four points"


def test_reader_does_not_merge_points_with_the_same_x():
    """The reader on a hand-made figure: four expected points, five drawn. The old dictionary
    comparison called these equal."""
    plt.close("all")
    plt.figure().gca().scatter([0, 1, 2, 3, 0], [10, 10, 20, 20, 10])
    points = _read_points()
    plt.close("all")
    assert len(points) == 5
    assert points != _pts([0, 1, 2, 3], [10, 10, 20, 20])
    assert points == _pts([0, 1, 2, 3, 0], [10, 10, 20, 20, 10])


# ---------------------------------------------------------------------------------------------
# 1. The child join key is honoured (ADF-EXT-001 / POST76-010)
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("surface", SURFACES)
def test_child_key_with_a_different_name(surface):
    """The reported shape: draw("Pos.pt:x") with posTrackIdx -> rowId gives pt of the daughter track.
    Unfixed: draw and draw_figures raise KeyError "['posTrackIdx'] not in index"."""
    assert _draw_points(_v0(), surface, "Pos.pt:x") == _pts(V0_X, POS_PT)


def test_child_key_with_a_different_name_agrees_with_alias_evaluation():
    """The direct reference and the alias evaluator give the same values (unfixed: KeyError)."""
    v0 = _v0()
    v0.add_alias("posPt", "Pos.pt")
    v0.materialize_aliases(names=["posPt"])
    assert v0.df["posPt"].tolist() == POS_PT
    assert _draw_points(_v0(), "draw", "Pos.pt:x") == _pts(V0_X, v0.df["posPt"].tolist())


@pytest.mark.parametrize("surface", SURFACES)
def test_child_key_only_as_an_index_level(surface):
    """Same key name on both sides, but the child carries the key only as a DataFrame index level
    (a registered join key by AD-17). Unfixed: draw and draw_figures raise KeyError."""
    v0 = AliasDataFrame(pd.DataFrame({"posTrackIdx": np.array(POS, np.int64), "x": V0_X}))
    child = AliasDataFrame(pd.DataFrame({"pt": TRACK_PT},
                                        index=pd.Index(np.array(TRACK_ROW, np.int64), name="posTrackIdx")))
    v0.register_subframe("Pos", child, index_columns="posTrackIdx")
    assert _draw_points(v0, surface, "Pos.pt:x") == _pts(V0_X, POS_PT)


@pytest.mark.parametrize("surface", SURFACES)
def test_child_column_named_like_the_parent_key_is_not_the_join_key(surface):
    """A particle table linked to itself: motherIdx (parent) -> rowId (child), and the child also
    has a column motherIdx. rowId=[0,1,2,3] motherIdx=[0,0,1,1] pt=[10,20,30,40], so
    Mother.pt = pt[motherIdx] = [10,10,20,20].
    Unfixed: draw joined on the child's motherIdx and SILENTLY drew [10,10,30,30];
    draw_figures raised ValueError (length mismatch)."""
    cols = {"rowId": np.array([0, 1, 2, 3], np.int64), "motherIdx": np.array([0, 0, 1, 1], np.int64),
            "pt": [10.0, 20.0, 30.0, 40.0], "x": [0.0, 1.0, 2.0, 3.0]}
    par = AliasDataFrame(pd.DataFrame(cols))
    par.register_subframe("Mother", AliasDataFrame(pd.DataFrame(cols)),
                          index_columns="motherIdx", right_index_columns="rowId")
    assert _draw_points(par, surface, "Mother.pt:x") == [(0.0, 10.0), (1.0, 10.0), (2.0, 20.0), (3.0, 20.0)]


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("names", ["same_names", "different_names"])
@pytest.mark.parametrize("duplicate", [False, True], ids=["unique_child_keys", "duplicate_child_key"])
def test_two_column_key(surface, names, duplicate):
    """A key of two columns (run, sec). Parent rows (1,0) (1,1) (2,0) (2,1); the child rows are
    stored in another order with v = 10, 20, 30, 40 for those keys, so C.v is [10, 20, 30, 40].
    With `duplicate`, the child has a second row for key (1,0) with v = 99 AFTER the first one;
    the first row is used.
    Unfixed: with different names draw and draw_figures raise KeyError; with same names and a
    duplicate, draw_figures raises ValueError (length mismatch). Everything else passes."""
    child_run, child_sec, child_v = [2, 1, 1, 2], [1, 1, 0, 0], [40.0, 20.0, 10.0, 30.0]
    if duplicate:
        child_run, child_sec, child_v = child_run + [1], child_sec + [0], child_v + [99.0]
    run_name, sec_name = ("run", "sec") if names == "same_names" else ("childRun", "childSec")
    par = AliasDataFrame(pd.DataFrame({"run": np.array([1, 1, 2, 2], np.int64),
                                       "sec": np.array([0, 1, 0, 1], np.int64), "x": V0_X}))
    child = AliasDataFrame(pd.DataFrame({run_name: np.array(child_run, np.int64),
                                         sec_name: np.array(child_sec, np.int64), "v": child_v}))
    par.register_subframe("C", child, index_columns=["run", "sec"],
                          right_index_columns=None if names == "same_names" else [run_name, sec_name])
    assert _draw_points(par, surface, "C.v:x") == [(0.0, 10.0), (1.0, 20.0), (2.0, 30.0), (3.0, 40.0)]


# ---------------------------------------------------------------------------------------------
# 2. Every row selection and request slot goes through the same join
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("surface", SURFACES)
def test_entry_range_selects_rows_by_position(surface):
    """entry_begin=1, entry_end=3 keeps V0 rows 1 and 2. Unfixed: KeyError on draw/draw_figures."""
    assert _draw_points(_v0(), surface, "Pos.pt:x", entry_begin=1, entry_end=3) == [(1.0, 10.0), (2.0, 40.0)]


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("mask", [[True, False, False, True], [3, 0]], ids=["bool_mask", "int_mask"])
def test_entry_mask_with_repeated_parent_index_labels(surface, mask):
    """Rows are taken by POSITION: the parent index labels 7,7,8,9 repeat, and rows 0 and 3 are
    selected. Unfixed: KeyError on draw/draw_figures."""
    v0 = _v0(index=[7, 7, 8, 9])
    assert _draw_points(v0, surface, "Pos.pt:x", entry_mask=np.array(mask)) == [(0.0, 30.0), (3.0, 30.0)]


@pytest.mark.parametrize("surface", SURFACES)
def test_whole_frame_with_repeated_parent_index_labels(surface):
    """No row selection, parent index labels 7,7,8,9: still exactly four points, one per V0.
    Unfixed: KeyError on draw/draw_figures."""
    assert _draw_points(_v0(index=[7, 7, 8, 9]), surface, "Pos.pt:x") == _pts(V0_X, POS_PT)


# max_entries of draw_figures. The limit is applied as the slice [:max_entries] of the rows, and
# only when it is non-zero and smaller than the number of rows. So for the four V0s:
MAX_ENTRIES_ROWS = {
    1: [0], 2: [0, 1], 3: [0, 1, 2],        # the first rows
    4: [0, 1, 2, 3], 5: [0, 1, 2, 3],       # not smaller than the frame: no limit
    0: [0, 1, 2, 3],                        # zero: no limit
    -1: [0, 1, 2], -2: [0, 1], -3: [0],     # negative: all but the last rows (slice semantics)
    -4: [], -5: [],                         # negative, nothing left: nothing is drawn
}


@pytest.mark.parametrize("keys", sorted(V0_BUILDERS))
@pytest.mark.parametrize("max_entries", sorted(MAX_ENTRIES_ROWS))
def test_draw_figures_max_entries_keeps_the_rows_of_the_slice(keys, max_entries):
    """Every value of max_entries from -5 to 5 on four V0s; the kept rows are listed in
    MAX_ENTRIES_ROWS above. Review finding F2.
    Unfixed: different_key_names raises KeyError for every value; same_key_name passes.
    Fix v01: the six negative values with rows left (-1, -2, -3, both key kinds) raised
    ValueError "the join produced 0 values for a 3-row frame" -- a regression against Unfixed
    for same_key_name. -4 and -5 passed by accident (nothing drawn either way)."""
    rows = MAX_ENTRIES_ROWS[max_entries]
    points = _draw_points(V0_BUILDERS[keys](), "draw_figures", "Pos.pt:x", max_entries=max_entries)
    assert points == _pts([V0_X[i] for i in rows], [POS_PT[i] for i in rows])


@pytest.mark.parametrize("keys", sorted(V0_BUILDERS))
@pytest.mark.parametrize("call, rows", [
    (dict(entry_begin=1, max_entries=-1), [1, 2]),                      # rows 1,2,3 without the last
    (dict(entry_begin=1, max_entries=2), [1, 2]),                       # the first two of rows 1,2,3
    (dict(entry_end=3, max_entries=-2), [0]),                           # rows 0,1,2 without the last two
    (dict(entry_mask=np.array([3, 0, 1]), max_entries=2), [3, 0]),      # mask order 3,0,1: first two
    (dict(entry_mask=np.array([3, 0, 1]), max_entries=-1), [3, 0]),     # mask order 3,0,1: without the last
    (dict(entry_mask=np.array([False, True, True, True]), max_entries=-2), [1]),
], ids=["begin1_max-1", "begin1_max2", "end3_max-2", "intmask_max2", "intmask_max-1", "boolmask_max-2"])
def test_draw_figures_max_entries_after_a_row_selection(keys, call, rows):
    """max_entries is applied AFTER the row selection, to the selected rows in their order.
    Unfixed: different_key_names raises KeyError; same_key_name passes.
    Fix v01 and Fix v02: pass (with a row selection the positions were already sliced alike)."""
    points = _draw_points(V0_BUILDERS[keys](), "draw_figures", "Pos.pt:x", **call)
    assert points == _pts([V0_X[i] for i in rows], [POS_PT[i] for i in rows])


@pytest.mark.parametrize("surface", SURFACES)
def test_selection_slot_uses_the_child_key(surface):
    """selection="Pos.pt>15" keeps V0 rows 0, 2, 3 (pt 30, 40, 30).
    Unfixed: KeyError on draw/draw_figures."""
    points = _draw_points(_v0(), surface, "Pos.pt:x", plot={"selection": "Pos.pt>15"})
    assert points == [(0.0, 30.0), (2.0, 40.0), (3.0, 30.0)]


@pytest.mark.parametrize("surface", SURFACES)
def test_weights_slot_gives_the_weight_of_the_right_row(surface):
    """Histogram of x with one bin per V0 and weights="Pos.pt": the height of bin i is the pt of
    the daughter of V0 i, [30, 10, 40, 30]. A wrong row association changes a bin height.
    Unfixed: KeyError on draw/draw_figures."""
    assert _draw_hist(_v0(), surface, "x", plot={"weights": "Pos.pt"}) == [
        (-0.5, 30.0), (0.5, 10.0), (1.5, 40.0), (2.5, 30.0)]


@pytest.mark.parametrize("surface", SURFACES)
def test_weights_slot_with_an_entry_range(surface):
    """The same histogram for V0 rows 1 and 2 only: heights [0, 10, 40, 0].
    Unfixed: KeyError on draw/draw_figures."""
    heights = _draw_hist(_v0(), surface, "x", plot={"weights": "Pos.pt"}, entry_begin=1, entry_end=3)
    assert heights == [(-0.5, 0.0), (0.5, 10.0), (1.5, 40.0), (2.5, 0.0)]


@pytest.mark.parametrize("surface", SURFACES)
def test_group_by_slot_puts_each_row_in_the_right_group(surface):
    """Scatter Pos.pt:x grouped by the integer Pos.q = [7, 5, 8, 7]: three groups, and V0 0 and
    V0 3 (both daughter track 2) are together. Unfixed: KeyError on draw/draw_figures."""
    assert POS_Q == [7, 5, 8, 7]
    assert _draw_groups(_v0(), surface, "Pos.pt:x", "Pos.q") == [
        [(0.0, 30.0), (3.0, 30.0)],     # q = 7
        [(1.0, 10.0)],                  # q = 5
        [(2.0, 40.0)],                  # q = 8
    ]


# ---------------------------------------------------------------------------------------------
# 3. Unmatched parent rows follow the draw_batch contract (POST76-002)
#    V0 3 points to track 9, which does not exist. The key has the SAME name on both sides
#    here, so on the unfixed code these tests fail because of the fill defect alone.
# ---------------------------------------------------------------------------------------------

UNMATCHED_POS = [2, 0, 3, 9]


@pytest.mark.parametrize("surface", SURFACES)
def test_unmatched_row_gets_the_configured_fill(surface):
    """With set_subframe_fill("Pos", 0) the unmatched V0 is drawn at the fill value 0.
    Unfixed: draw and draw_figures ignored the fill and silently dropped the row
    (3 points drawn instead of 4)."""
    v0 = _same_name_v0(pos=UNMATCHED_POS)
    v0.set_subframe_fill("Pos", 0)
    assert _draw_points(v0, surface, "Pos.q:x") == [(0.0, 7.0), (1.0, 5.0), (2.0, 8.0), (3.0, 0.0)]


@pytest.mark.parametrize("surface", SURFACES)
@pytest.mark.parametrize("column", ["q", "good"], ids=["integer_column", "boolean_column"])
def test_unmatched_row_of_an_integer_or_boolean_column_without_fill_is_refused(surface, column):
    """An integer or a boolean column cannot hold "no value". Without a fill the request is
    refused with an error that names the subframe and the column (rule AD-19), on every surface.
    Unfixed: draw and draw_figures silently converted to float and dropped the row (no error)."""
    v0 = _same_name_v0(pos=UNMATCHED_POS)
    with pytest.raises(ValueError, match=rf"subframe 'Pos' column '{column}' with 1 missing join key"):
        _draw_points(v0, surface, f"Pos.{column}:x")


@pytest.mark.parametrize("surface", SURFACES)
def test_unmatched_row_outside_the_entry_range_is_not_refused(surface):
    """The refusal looks only at the rows of the request: entry_end=3 leaves the unmatched V0 3
    out, so the integer column is drawn for V0 0, 1, 2 without a fill.
    Control: passes before and after the fix."""
    v0 = _same_name_v0(pos=UNMATCHED_POS)
    assert _draw_points(v0, surface, "Pos.q:x", entry_end=3) == [(0.0, 7.0), (1.0, 5.0), (2.0, 8.0)]


@pytest.mark.parametrize("surface", SURFACES)
def test_unmatched_row_of_a_float_column_without_fill_is_not_drawn(surface):
    """A float column holds NaN for the unmatched V0, and that point is not drawn.
    Control: passes before and after the fix."""
    v0 = _same_name_v0(pos=UNMATCHED_POS)
    assert _draw_points(v0, surface, "Pos.pt:x") == [(0.0, 30.0), (1.0, 10.0), (2.0, 40.0)]


@pytest.mark.parametrize("surface", SURFACES)
def test_unmatched_row_with_a_different_key_name_gets_the_fill(surface):
    """Both parts together: posTrackIdx -> rowId, one unmatched V0, fill 0.
    Unfixed: draw and draw_figures raise KeyError."""
    v0 = _v0(pos=UNMATCHED_POS)
    v0.set_subframe_fill("Pos", 0)
    assert _draw_points(v0, surface, "Pos.q:x") == [(0.0, 7.0), (1.0, 5.0), (2.0, 8.0), (3.0, 0.0)]


# ---------------------------------------------------------------------------------------------
# 4. Controls: behaviour that must not change
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("surface", SURFACES)
def test_control_same_name_key_column(surface):
    """The ordinary case: same key name, ordinary column. Passes before and after the fix."""
    assert _draw_points(_same_name_v0(), surface, "Pos.pt:x") == _pts(V0_X, POS_PT)


@pytest.mark.parametrize("surface", SURFACES)
def test_control_duplicate_child_keys_keep_the_first_row(surface):
    """Child key 2 appears twice (pt 30, then 99): the first row is used, as draw and draw_batch
    did before, and exactly four points are drawn. Changed for draw_figures only: unfixed it
    raised a raw pandas ValueError (length mismatch); now it behaves like the other two surfaces."""
    v0 = _same_name_v0(child_key=[0, 1, 2, 3, 2], child_pt=[10.0, 20.0, 30.0, 40.0, 99.0],
                       child_q=[5, 6, 7, 8, 9])
    assert _draw_points(v0, surface, "Pos.pt:x") == _pts(V0_X, POS_PT)


@pytest.mark.parametrize("surface", SURFACES)
def test_control_two_level_reference(surface):
    """Cascade -> V0 -> track with a different key name at both levels; drawn as V0.Pos.pt from
    the cascade frame. Cascade i uses V0 v0Idx[i] = [3, 1]; V0 3 -> track 2 (30), V0 1 -> track 0 (10).
    Passes before and after the fix (two-level references use another code path)."""
    v0 = AliasDataFrame(pd.DataFrame({"rowId": np.array([0, 1, 2, 3], np.int64),
                                      "posTrackIdx": np.array(POS, np.int64)}))
    v0.register_subframe("Pos", _tracks(), index_columns="posTrackIdx", right_index_columns="rowId")
    casc = AliasDataFrame(pd.DataFrame({"v0Idx": np.array([3, 1], np.int64), "x": [0.0, 1.0]}))
    casc.register_subframe("V0", v0, index_columns="v0Idx", right_index_columns="rowId")
    assert _draw_points(casc, surface, "V0.Pos.pt:x") == [(0.0, 30.0), (1.0, 10.0)]


def test_control_workaround_through_an_alias_still_works():
    """The workaround people used before the fix: draw the alias with lazy=True."""
    v0 = _v0()
    v0.add_alias("posPt", "Pos.pt")
    assert _draw_points(v0, "draw", "posPt:x", lazy=True) == _pts(V0_X, POS_PT)


@pytest.mark.parametrize("max_entries", [-2, -1, 2])
def test_control_draw_figures_max_entries_without_a_subframe(max_entries):
    """max_entries on a plain column, no subframe reference: [:max_entries] of the rows.
    Passes on Unfixed, Fix v01 and Fix v02 (this path was never touched)."""
    rows = MAX_ENTRIES_ROWS[max_entries]
    points = _draw_points(_v0(), "draw_figures", "posTrackIdx:x", max_entries=max_entries)
    assert points == _pts([V0_X[i] for i in rows], [POS[i] for i in rows])
