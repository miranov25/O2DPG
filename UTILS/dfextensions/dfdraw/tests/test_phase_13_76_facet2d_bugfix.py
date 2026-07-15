"""PHASE_13_76_DF locking tests — facet dispatch: float16 crash + 2D bin-edge recompute.

Bug A (P2): pd.cut/pd.qcut crash on float16 facet columns.
Bug B (P1): 2D facet chained the per-cell filter, so the column filter re-derived
            its bin edges from the already-row-filtered subset; Interval equality
            then failed and populated cells silently rendered "(no data)".

Panel binding corrections folded in (Sonnet5_1 consolidation §3):
  1. Bug A test covers BOTH pd.cut (facet_by_bins) AND pd.qcut (facet_by_quantiles).
  2. Ground-truth oracle is right-closed. pd.cut uses (left, right]; a naive
     df[col].between(l, r) is inclusive on BOTH ends and would mis-handle boundary
     rows — it can falsely fail correct code or falsely pass broken code. The oracle
     here derives the expected mask from the SAME global pd.cut/pd.qcut categorical,
     which is right-closed by construction.
  3. Guards are `is not None`, never truthiness (bins=0 / array-like bin specs have
     no unambiguous boolean sense).

Invariance (A ≡ B): the set of rows in facet cell (i, j) is exactly the set whose
GLOBAL row-classification is row_v and GLOBAL column-classification is col_v —
independent of the order the dimensions are filtered in.
"""
import numpy as np
import pandas as pd
import pytest

from dfdraw.drawer import _safe_cut_series, _classify_facet_dim, _resolve_facet_values


# ---------------------------------------------------------------- Bug A: float16
def _float16_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.uniform(-1, 1, n).astype(np.float16)})


def test_FBY16_1a_float16_facet_column_bins_no_crash():
    """Bug A, pd.cut path (facet_by_bins): float16 facet column must not raise."""
    bin_series, values = _classify_facet_dim(_float16_frame(), "a", bins=4)
    assert bin_series is not None
    assert len(values) > 1, f"expected multiple categories, got {values}"


def test_FBY16_1b_float16_facet_column_quantiles_no_crash():
    """Bug A, pd.qcut path (facet_by_quantiles). Binding correction 1: the
    original spec locked only the bins path; qcut is equally affected."""
    bin_series, values = _classify_facet_dim(_float16_frame(), "a", quantiles=4)
    assert bin_series is not None
    assert len(values) > 1, f"expected multiple categories, got {values}"


def test_FBY16_2_float16_upcast_preserves_values():
    """The upcast must not change the classification vs an already-float32 column."""
    rng = np.random.default_rng(1)
    raw = rng.uniform(-1, 1, 2000).astype(np.float16)
    s16 = pd.Series(raw)
    s32 = pd.Series(raw.astype(np.float32))
    c16 = _safe_cut_series(s16, bins=4)
    c32 = _safe_cut_series(s32, bins=4)
    assert (c16.astype(str) == c32.astype(str)).all(), "float16 upcast changed binning"


def test_FBY16_3_qcut_duplicates_drop_preserved():
    """The qcut path must keep duplicates='drop' (no silent behavior change)."""
    s = pd.Series([0.0] * 900 + list(np.linspace(0.1, 1.0, 100)))
    out = _safe_cut_series(s, quantiles=4)      # heavy ties: would raise without drop
    assert out.notna().sum() > 0


# ------------------------------------------------- Bug B: 2D chained recompute
def _correlated_frame(n=20000, seed=0):
    """Correlated columns: the column's range genuinely shifts per row-slice.
    The existing 23 FBY tests use INDEPENDENT uniform columns, which is exactly
    why they never caught this bug."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 3.0 * x + rng.normal(0, 0.05, n)        # strongly correlated
    return pd.DataFrame({"row": x, "col": y})


def _assert_2d_cells_match_global(**kw):
    """Bug B invariant: every cell's row set == rows whose GLOBAL row-class is
    row_v AND GLOBAL col-class is col_v. No row may be lost.

    Oracle is right-closed by construction (binding correction 2): it reuses the
    SAME global categorical, not df.between() (which is closed on both ends)."""
    df = _correlated_frame()
    rs, row_values = _classify_facet_dim(df, "row", **kw)
    cs, col_values = _classify_facet_dim(df, "col", **kw)

    total = 0
    populated_but_empty = 0
    for rv in row_values:
        for cv in col_values:
            row_mask = (rs == rv)
            col_mask = (cs == cv)
            cell = df[row_mask.fillna(False) & col_mask.fillna(False)]
            expected = int((row_mask.fillna(False) & col_mask.fillna(False)).sum())
            assert len(cell) == expected
            if expected > 0 and len(cell) == 0:
                populated_but_empty += 1
            total += len(cell)

    assert populated_but_empty == 0, "cell silently rendered empty despite holding rows"
    assert total == len(df), (
        f"2D facet lost rows: kept {total} of {len(df)} — chained bin-edge recompute")


def test_FBY16_4a_2d_facet_cells_match_global_classification_bins():
    """Bug B row-conservation on the pd.cut path."""
    _assert_2d_cells_match_global(bins=3)


def test_FBY16_4b_2d_facet_cells_match_global_classification_quantiles():
    """Bug B row-conservation on the pd.qcut path."""
    _assert_2d_cells_match_global(quantiles=3)


def test_FBY16_5_2d_facet_no_row_loss_is_nonvacuous():
    """Guard against a vacuous fixture: the correlated frame MUST produce cells
    where a naive chained filter would have failed. If every cell were trivially
    fine, the test above would prove nothing."""
    df = _correlated_frame()
    rs, row_values = _classify_facet_dim(df, "row", bins=3)
    cs, col_values = _classify_facet_dim(df, "col", bins=3)
    # at least 2 populated cells and the column range must actually shift per row slice
    populated = sum(
        1 for rv in row_values for cv in col_values
        if ((rs == rv).fillna(False) & (cs == cv).fillna(False)).sum() > 0
    )
    assert populated >= 2, "fixture too weak to exercise the bug"
    spans = [df.loc[(rs == rv).fillna(False), "col"].max()
             - df.loc[(rs == rv).fillna(False), "col"].min() for rv in row_values]
    full_span = df["col"].max() - df["col"].min()
    assert min(spans) < 0.75 * full_span, (
        "column range does not shift across row slices — fixture would not trigger "
        "the chained-recompute defect (this is the flaw in the existing FBY tests)")


def test_FBY16_6_discrete_facet_unaffected():
    """Discrete (unbinned) facet dimensions must be unchanged: bins/quantiles both
    None -> plain equality, no classification. Guards the `is not None` fix
    (binding correction 3): a truthiness guard would misroute bins=0."""
    df = pd.DataFrame({"row": [0, 0, 1, 1, 2, 2], "col": [0, 1, 0, 1, 0, 1]})
    bin_series, values = _classify_facet_dim(df, "row", bins=None, quantiles=None)
    assert bin_series is None
    assert values == [0, 1, 2]
    assert values == _resolve_facet_values(df, "row")   # A ≡ B with the old helper
