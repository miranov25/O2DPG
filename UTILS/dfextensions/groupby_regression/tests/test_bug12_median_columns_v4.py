"""Regression tests — BUG_groupby_20260704_median_columns_silently_ignored.

Phase 13.23c.GB (PHASE_13_23C_GB_Audit fix set, step 1).
Architect ruling 2026-07-04: "It is bug and should be fixed."

Failure-mode #12 discipline: every test calls the affected entry point
`make_parallel_fit_v4` DIRECTLY (entry point exercised, not an internal
helper).

T1  oracle:            v4 medians == df.loc[sel].groupby(gb)[cols].median()
                       output name = f"{col}{suffix}" (architect naming
                       ruling "Option B", 2026-07-04 — suffix applied as for
                       every other output column, true legacy parity)
T2  cross-fitter:      v4 medians == legacy GroupByRegressor medians,
                       exercised for suffix="" AND a non-empty suffix
                       (legacy renames medians with the suffix too,
                       groupby_regression.py:538)
T3  collision guard:   ValueError naming the colliding column
T4  no-op safety:      median_columns absent / None / [] produce
                       bit-identical dfGB (tri-identity; the absent-kwarg
                       call is by construction the pre-fix baseline shape —
                       additionally verified once against the pristine
                       pre-fix module in the CRR run log)
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groupby_regression_optimized import make_parallel_fit_v4  # noqa: E402
from groupby_regression import GroupByRegressor  # noqa: E402


RNG = np.random.default_rng(42)


def _make_df(n_per_group=12, groups=((0, 0), (0, 1), (1, 0), (1, 1), (2, 1))):
    """Two group columns, one predictor, one target, two median candidates.

    Includes NaN values in a median column to pin the NaN-skipping
    semantics, and a float16 median column to mirror production dtypes.
    """
    rows = []
    for (g1, g2) in groups:
        x = RNG.normal(size=n_per_group)
        y = 1.5 * x + g1 - 0.5 * g2 + RNG.normal(scale=0.1, size=n_per_group)
        t = RNG.normal(loc=10 * g1 + g2, size=n_per_group)
        z = RNG.normal(loc=5, size=n_per_group).astype(np.float16)
        rows.append(pd.DataFrame({
            "g1": g1, "g2": g2, "x": x, "y": y, "t": t, "z": z}))
    df = pd.concat(rows, ignore_index=True)
    # NaNs in a median column (pandas .median() must skip them per group)
    df.loc[df.index[::7], "t"] = np.nan
    return df


def _selection(df):
    """Non-trivial boolean selection, same object passed to v4 and oracle."""
    return (df["x"] > -1.0) & df["y"].notna()


# ---------------------------------------------------------------------------
# T1 — oracle equality (exact: same pandas operation on the same rows)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("gb_columns", [["g1"], ["g1", "g2"]])
@pytest.mark.parametrize("suffix", ["", "_v4"])
def test_t1_v4_medians_match_groupby_oracle(gb_columns, suffix):
    df = _make_df()
    sel = _selection(df)
    median_columns = ["t", "z"]

    _, dfGB = make_parallel_fit_v4(
        df=df, gb_columns=gb_columns, fit_columns=["y"],
        linear_columns=["x"], median_columns=median_columns,
        suffix=suffix, selection=sel, min_stat=3,
    )

    # Option B contract: output under f"{col}{suffix}" for every entry
    out_names = [f"{col}{suffix}" for col in median_columns]
    for col, out in zip(median_columns, out_names):
        assert out in dfGB.columns, (
            f"median column '{out}' missing from dfGB (bug #12 regression)")
        if suffix:
            assert col not in dfGB.columns, (
                f"bare name '{col}' must not appear when suffix='{suffix}'")

    oracle = df.loc[sel].groupby(gb_columns, sort=True)[
        median_columns].median()
    oracle.columns = out_names
    got = dfGB.set_index(gb_columns)[out_names].sort_index()
    oracle = oracle.sort_index()

    # Exact — identical pandas op over identical rows (rtol=0 per work order)
    pd.testing.assert_frame_equal(
        got, oracle, check_exact=True, check_names=False)


def test_t1b_row_set_is_post_selection_not_full_df():
    """The median must be over the SELECTED rows, not the full frame."""
    df = _make_df()
    sel = _selection(df)
    _, dfGB = make_parallel_fit_v4(
        df=df, gb_columns=["g1", "g2"], fit_columns=["y"],
        linear_columns=["x"], median_columns=["t"], suffix="",
        selection=sel, min_stat=3,
    )
    full = df.groupby(["g1", "g2"])[["t"]].median()
    selected = df.loc[sel].groupby(["g1", "g2"])[["t"]].median()
    got = dfGB.set_index(["g1", "g2"])[["t"]].sort_index()
    # sanity: the two oracles genuinely differ on this data
    assert not full.sort_index().equals(selected.sort_index())
    pd.testing.assert_frame_equal(
        got, selected.sort_index(), check_exact=True, check_names=False)


# ---------------------------------------------------------------------------
# T2 — cross-fitter invariance vs legacy GroupByRegressor (suffix="")
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("suffix", ["", "_T2"])
def test_t2_v4_medians_match_legacy_groupbyregressor(suffix):
    df = _make_df(n_per_group=16)  # legacy filters len(group) >= min_stat/2
    df["w"] = 1.0  # legacy path requires a concrete weights column
    sel = _selection(df)
    gb_columns = ["g1", "g2"]
    median_columns = ["t"]
    out_names = [f"{c}{suffix}" for c in median_columns]

    _, dfGB_v4 = make_parallel_fit_v4(
        df=df, gb_columns=gb_columns, fit_columns=["y"],
        linear_columns=["x"], median_columns=median_columns,
        suffix=suffix, selection=sel, min_stat=3,
    )
    _, dfGB_legacy = GroupByRegressor.make_parallel_fit(
        df, gb_columns=gb_columns, fit_columns=["y"],
        linear_columns=["x"], median_columns=median_columns,
        weights="w", suffix=suffix, selection=sel,
        min_stat=[3, 3], n_jobs=1,
    )

    # Both fitters must emit the SAME suffixed median names (Option B)
    v4 = dfGB_v4.set_index(gb_columns)[out_names].sort_index()
    legacy = dfGB_legacy.set_index(gb_columns)[out_names].sort_index()
    # legacy may drop tiny groups (min_stat/2 pre-filter) — compare on the
    # intersection, require it non-empty and complete on the legacy side
    common = v4.index.intersection(legacy.index)
    assert len(common) == len(legacy), "legacy groups must all appear in v4"
    assert len(common) > 0
    pd.testing.assert_frame_equal(
        v4.loc[common].astype(float), legacy.loc[common].astype(float),
        check_exact=True, check_names=False)


# ---------------------------------------------------------------------------
# T3 — name-collision guard
# ---------------------------------------------------------------------------
def test_t3_collision_with_generated_output_raises():
    df = _make_df()
    # Manufacture a genuine collision: with suffix="", the suffixed median
    # name equals the bare name, which equals a generated coefficient
    # output column.
    df["y_intercept"] = RNG.normal(size=len(df))
    with pytest.raises(ValueError, match="y_intercept"):
        make_parallel_fit_v4(
            df=df, gb_columns=["g1", "g2"], fit_columns=["y"],
            linear_columns=["x"], median_columns=["y_intercept"],
            suffix="", min_stat=3,
        )


def test_t3b_collision_with_group_key_raises():
    df = _make_df()
    with pytest.raises(ValueError, match="g1"):
        make_parallel_fit_v4(
            df=df, gb_columns=["g1", "g2"], fit_columns=["y"],
            linear_columns=["x"], median_columns=["g1"],
            suffix="", min_stat=3,
        )


# ---------------------------------------------------------------------------
# T4 — no-op safety: absent / None / [] are bit-identical
# ---------------------------------------------------------------------------
def test_t4_noop_tri_identity():
    df = _make_df()
    sel = _selection(df)
    kw = dict(df=df, gb_columns=["g1", "g2"], fit_columns=["y"],
              linear_columns=["x"], suffix="_v4", selection=sel, min_stat=3)

    _, dfGB_absent = make_parallel_fit_v4(**kw)
    _, dfGB_none = make_parallel_fit_v4(median_columns=None, **kw)
    _, dfGB_empty = make_parallel_fit_v4(median_columns=[], **kw)

    pd.testing.assert_frame_equal(dfGB_absent, dfGB_none,
                                  check_exact=True, check_dtype=True)
    pd.testing.assert_frame_equal(dfGB_absent, dfGB_empty,
                                  check_exact=True, check_dtype=True)


def test_t4b_other_columns_unchanged_when_medians_requested():
    """Adding median_columns must not perturb any pre-existing output column
    (values, names, dtypes, row order)."""
    df = _make_df()
    sel = _selection(df)
    kw = dict(df=df, gb_columns=["g1", "g2"], fit_columns=["y"],
              linear_columns=["x"], suffix="_v4", selection=sel, min_stat=3)

    _, base = make_parallel_fit_v4(**kw)
    _, with_med = make_parallel_fit_v4(median_columns=["t"], **kw)

    assert list(with_med.columns) == list(base.columns) + ["t_v4"]
    pd.testing.assert_frame_equal(
        with_med[base.columns], base, check_exact=True, check_dtype=True)


# ---------------------------------------------------------------------------
# Metadata truth (audit failure-mode: requested-not-applied)
# ---------------------------------------------------------------------------
def test_metadata_medians_lists_actual_output_names():
    df = _make_df()
    _, dfGB, meta = make_parallel_fit_v4(
        df=df, gb_columns=["g1", "g2"], fit_columns=["y"],
        linear_columns=["x"], median_columns=["t"], suffix="_v4",
        min_stat=3, return_metadata=True,
    )
    assert meta["columns"]["medians"] == ["t_v4"]
    for name in meta["columns"]["medians"]:
        assert name in dfGB.columns, (
            "metadata must list names that actually exist in dfGB")
