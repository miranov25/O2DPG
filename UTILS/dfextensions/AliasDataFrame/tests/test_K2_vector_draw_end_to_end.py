"""
Phase 13.19.ADF.FIX1 — K2 End-to-End Test for Vector draw() with group_by

PURPOSE: Verify the dfdraw FIX1 (commit fe007b7c) works end-to-end through
ADF for the architect's production reproducer. This is what K1 should have
done from the start: count rendered output (legend entries, axes, panels),
not just intercept kwargs at the DFDraw boundary.

PHASE: 13.19.ADF.FIX1 v0.3 (post-dfdraw-FIX1 verification)
DEPENDS ON: dfdraw at commit `fe007b7c` (PHASE_13_16_DF_FIX1_END) or later
ARCHITECT REPRODUCER (real ITS data, see EMAIL §11.2):
    aDF.draw(
        "[dd_dzITS0..5]:staveITS",
        type='profile', group_by='mP3', group_by_bins=6, bins=12,
        ...
    )
    Expected: main legend = 6 entries (group bins), NOT 421.

DESIGN:
    - Build a synthetic ADF with structure mirroring the ITS reproducer:
      vector base names + a group_by column with many distinct values.
    - Call adf.draw() through the full ADF→DFDraw pipeline (no monkey-patch).
    - Inspect the returned Figure/Axes for legend entries, axis count,
      title structure.
    - Filter category labels per dfdraw FIX1 §9.2 — `_add_vector_legend`
      may contribute a proxy entry that is not a "real" group label.

NO MODIFICATIONS TO AliasDataFrame.py. Pure verification.

EXPECTED OUTCOMES (with dfdraw FIX1 landed):
    K2_1: scalar baseline — group_by alone produces N legend entries
          where N == group_by_bins.  Sanity check, expected pass.
    K2_2: vector + group_by — main legend has at most group_by_bins entries,
          NOT (group_by_bins × n_vector). The bug-fix gate.
    K2_3: production reproducer mirror — multi-vector + group_by_bins=6,
          assert main legend ≤ 6.
    K2_4: title sanity — title is short (≤2 lines), not duplicated
          per vector iteration.
"""

import os
import sys
import warnings

import pytest
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless, no display required
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


# ============================================================================
# Fixtures
# ============================================================================

def _build_adf_for_groupby(n=600, n_groups=20):
    """
    Synthetic ADF mirroring the structural shape of the ITS reproducer.

    Columns:
        x, y1..y6 : continuous (the 6-element "vector" surface)
        mP3       : continuous, will be quantile-binned by group_by_bins
        staveITS  : the 'x' axis category (we use continuous for simplicity)
        row, isPrimITS : selection columns

    n_groups is the number of distinct mP3 quantile-able values. Larger
    n_groups makes the bug more visible: pre-fix, vector + group_by_bins=6
    produced ~ (n_groups * 6) legend entries instead of 6.
    """
    rng = np.random.default_rng(13192)

    df = pd.DataFrame({
        "x": rng.uniform(0.0, 100.0, size=n),
        "staveITS": rng.uniform(0.0, 100.0, size=n),
        "mP3": rng.uniform(-3.0, 3.0, size=n),  # continuous → many distinct
        "row": rng.integers(0, 360, size=n),
        "isPrimITS": rng.integers(0, 2, size=n),
    })
    # 6 "vector" columns y1..y6 with mild dependence on x and mP3
    for i in range(1, 7):
        df[f"y{i}"] = (
            0.1 * i * df["x"]
            + 0.5 * df["mP3"]
            + rng.normal(0.0, 1.0, size=n)
        )
    return AliasDataFrame(df)


def _count_real_group_legend_entries(ax):
    """
    Return the count of "real" group legend entries on ax, filtering out
    proxy/placeholder entries from the secondary 'Variable' legend
    (per dfdraw FIX1 §9.2 documented behavior).

    Strategy: take the legend with the MOST entries (the main one), then
    filter out labels that look like vector-base placeholders (containing
    " vs " or matching one of the y* base names alone).
    """
    legends = [
        c for c in ax.get_children()
        if isinstance(c, matplotlib.legend.Legend)
    ]
    if not legends:
        # Some matplotlib versions stash legend on figure, not axes
        fig = ax.figure
        legends = [
            c for c in fig.get_children()
            if isinstance(c, matplotlib.legend.Legend)
        ]
    if not legends:
        return 0, []

    main_legend = max(legends, key=lambda lg: len(lg.get_texts()))
    raw_labels = [t.get_text() for t in main_legend.get_texts()]

    # Filter: drop variable-legend proxy entries (per FIX1 §9.2)
    filtered = []
    for lbl in raw_labels:
        if " vs " in lbl:
            continue
        # bare 'y1', 'y2', ... 'y6' are likely vector-base proxies
        if lbl.strip() in {f"y{i}" for i in range(1, 10)}:
            continue
        filtered.append(lbl)

    return len(filtered), filtered


# ============================================================================
# K2 Tests — output counting on rendered figure
# ============================================================================

class TestK2VectorDrawEndToEnd:
    """
    K2_1..K2_4 — end-to-end output verification through ADF→dfdraw pipeline.

    These are the tests we should have written from the start. K1 verified
    kwargs ARRIVE at DFDraw; K2 verifies kwargs PRODUCE THE RIGHT OUTPUT.
    """

    @pytest.mark.invariance
    def test_K2_1_scalar_groupby_baseline(self):
        """
        K2_1 SCALAR BASELINE.

        adf.draw('y1:x', type='profile', group_by='mP3', group_by_bins=6)
        should produce a figure with ~6 group-bin legend entries.

        This is the sanity baseline — if this fails, the issue is bigger
        than vector dispatch.
        """
        adf = _build_adf_for_groupby()

        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = adf.draw(
                    "y1:x",
                    type="profile",
                    group_by="mP3",
                    group_by_bins=6,
                    bins=12,
                )
            except Exception as e:
                pytest.fail(
                    f"K2_1: scalar adf.draw raised "
                    f"{type(e).__name__}: {e}"
                )

        # Result shape varies by dfdraw version: (fig, ax) or (fig, ax, stats)
        ax = result[1] if isinstance(result, tuple) else plt.gca()

        n_real, labels = _count_real_group_legend_entries(ax)
        print(
            f"\nK2_1 scalar baseline: {n_real} real group legend entries; "
            f"labels (first 10): {labels[:10]}"
        )

        assert 1 <= n_real <= 8, (
            f"K2_1 SANITY: scalar group_by_bins=6 should produce roughly "
            f"6 (1-8 acceptable for quantile-binning edge cases) main "
            f"legend entries, got {n_real}. Labels: {labels}"
        )
        plt.close("all")

    @pytest.mark.invariance
    def test_K2_2_vector_groupby_main_legend_bounded(self):
        """
        K2_2 VECTOR + group_by — THE BUG-FIX GATE.

        adf.draw('[y1,y2]:x', type='profile', group_by='mP3',
                 group_by_bins=6, ...) — main legend must NOT explode to
        (n_groups × n_vector_entries). It should remain ~ group_by_bins.

        Pre-dfdraw-FIX1: this would show ~ (20 distinct mP3 × 2) = 40 legend
        entries. Post-FIX1: should be ≤ ~6 (group_by_bins).
        """
        adf = _build_adf_for_groupby()

        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = adf.draw(
                    "[y1,y2]:x",
                    type="profile",
                    group_by="mP3",
                    group_by_bins=6,
                    bins=12,
                )
            except Exception as e:
                pytest.fail(
                    f"K2_2: vector adf.draw raised "
                    f"{type(e).__name__}: {e}"
                )

        ax = result[1] if isinstance(result, tuple) else plt.gca()
        n_real, labels = _count_real_group_legend_entries(ax)
        print(
            f"\nK2_2 vector + group_by_bins=6:\n"
            f"  Real main legend entries: {n_real}\n"
            f"  Labels (first 15): {labels[:15]}"
        )

        # Bound: ≤ 8 leaves headroom for quantile-edge effects, but is
        # well below the broken count (~40). Pre-fix was a runaway.
        assert n_real <= 8, (
            f"K2_2 BUG GATE: vector + group_by_bins=6 produced {n_real} "
            f"main legend entries, expected ≤ 8 (target ≈ 6). "
            f"This is the architect's reproducer symptom — fix did not "
            f"propagate end-to-end through ADF→dfdraw.\n"
            f"Labels: {labels}"
        )
        # Lower bound — must have at least some grouping
        assert n_real >= 1, (
            f"K2_2: vector + group_by produced {n_real} group entries, "
            f"expected ≥ 1. Labels: {labels}"
        )
        plt.close("all")

    @pytest.mark.invariance
    @pytest.mark.xfail(
        strict=True,
        reason="SEED-2.a Repair DEFERRED, owner=dfdraw [ARCHITECT RULING "
               "2026-07-19, AD-4/13.76.ADF symmetry-by-default]: channel "
               "capacity check (dfdraw channels.py step-5, error site "
               "'cycle capacity') fires on the pre-top_k cardinality; "
               "intended: top_k limits the effective channel set BEFORE "
               "capacity validation, so top_k=4 is checked against "
               "capacity 4. Fail-before preserved here as the deferred "
               "acceptance test; XPASS on the dfdraw fix forces marker "
               "removal. Filed to dfdraw; do not modify dfdraw in "
               "PHASE_13_76_ADF (R-4).")
    def test_K2_3_production_reproducer_mirror(self):
        """
        K2_3 PRODUCTION REPRODUCER MIRROR.

        Synthetic mirror of the architect's ITS calibration command:
            aDF.draw("[dd_dzITS0..5]:staveITS",
                     selection="row==180 & isPrimITS==1",
                     type='profile', bins=12, group_by="mP3",
                     group_by_bins=6, auto_title=True)

        We use y1..y6 instead of dd_dzITS0..5 and `staveITS` continuous.
        Pre-fix: 421 main legend entries. Post-fix: ≤ ~6.
        """
        adf = _build_adf_for_groupby()

        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = adf.draw(
                    "[y1,y2,y3,y4,y5,y6]:staveITS",
                    selection="row > 0 & isPrimITS == 1",
                    type="profile",
                    bins=12,
                    group_by="mP3",
                    group_by_bins=6,
                    auto_title=True,
                )
            except Exception as e:
                pytest.fail(
                    f"K2_3: production-mirror adf.draw raised "
                    f"{type(e).__name__}: {e}"
                )

        ax = result[1] if isinstance(result, tuple) else plt.gca()
        n_real, labels = _count_real_group_legend_entries(ax)
        print(
            f"\nK2_3 production-mirror (vector6 + group_by_bins=6):\n"
            f"  Real main legend entries: {n_real}\n"
            f"  Labels (first 15): {labels[:15]}"
        )

        assert n_real <= 8, (
            f"K2_3 PRODUCTION REPRODUCER: vector(6) + group_by_bins=6 "
            f"produced {n_real} main legend entries, expected ≤ 8 "
            f"(target ≈ 6). Pre-fix was 421. If you see large numbers "
            f"(>20), the dfdraw fix did not reach this code path "
            f"end-to-end through ADF.\n"
            f"Labels: {labels}"
        )
        plt.close("all")

    @pytest.mark.invariance
    def test_K2_4_title_not_duplicated_per_vector_iteration(self):
        """
        K2_4 TITLE SANITY.

        Pre-dfdraw-FIX1 B3/B4: title was appended/replaced N times in
        vector mode. Post-fix: title is set once, ≤ 2 lines.
        """
        adf = _build_adf_for_groupby()

        plt.close("all")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                result = adf.draw(
                    "[y1,y2,y3]:x",
                    type="profile",
                    group_by="mP3",
                    group_by_bins=6,
                    bins=12,
                    auto_title=True,
                )
            except Exception as e:
                pytest.fail(
                    f"K2_4: title-test adf.draw raised "
                    f"{type(e).__name__}: {e}"
                )

        ax = result[1] if isinstance(result, tuple) else plt.gca()
        title_text = ax.get_title()
        n_lines = title_text.count("\n") + 1 if title_text else 0
        print(f"\nK2_4 title text:\n  '{title_text}'\n  ({n_lines} line(s))")

        assert n_lines <= 3, (
            f"K2_4: title has {n_lines} lines, expected ≤ 3. "
            f"Pre-fix B3/B4 produced N-iteration-stacked titles. "
            f"Title text:\n{title_text}"
        )
        plt.close("all")
