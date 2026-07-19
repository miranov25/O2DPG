"""
PHASE_13_76_ADF Stage A — Draw-Path characterization tests (increment 1).

Scope of this increment:
  * SEED-3 (ax=/deepcopy phantom): mechanism + all four public input forms.
      - Forms that are CORRECT today are asserted as Preserve rows (must pass
        now and forever).
      - Forms that are DEFECTIVE today are asserted against INTENDED behavior
        and marked xfail(strict=True): they fail-before (xfail) on the current
        baseline and will XPASS loudly the moment the Stage-B Repair lands,
        forcing marker removal — fail-before/pass-after captured in history
        (FM#12: every test calls the public production surfaces).
  * C-1 caller non-mutation characterization on draw_batch / draw_figures
    (current deepcopy mechanism honors it; the Stage-B structural-copy
    replacement must keep these green — OBS-1 before-state).

Environment contract: runs in the coder sandbox (pandas + matplotlib +
dfextensions package layout) AND on TARGET. Skips cleanly, never fails, when
dfdraw/matplotlib are unavailable.

Matrix rows seeded by this file (DRAW_PATH_BEHAVIOR_MATRIX.md):
  SEED-3.a  draw / ax kwarg               -> Preserve   (test_seed3_1)
  SEED-3.b  draw_batch / top-level ax     -> Preserve   (test_seed3_2)
  SEED-3.c  draw_batch / per-spec ax      -> Repair     (test_seed3_3, xfail)
  SEED-3.d  draw_batch / defaults ax      -> Repair     (test_seed3_4, xfail)
  C-1.a     draw_batch caller dicts       -> Preserve   (test_c1_1)
  C-1.b     draw_figures caller dicts     -> Preserve   (test_c1_2)
Defect anchors (canonical post-13.75, AliasDataFrame.py MD5 c73f0c99...):
  per-spec phantom  : `specs = _copy.deepcopy(specs)` in draw_batch
  defaults phantom  : `defaults = _copy.deepcopy(defaults)` in draw_batch
"""

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import AliasDataFrame as A  # noqa: E402

# dfdraw must be importable through the production package path; otherwise the
# draw surfaces cannot delegate and every draw test here must SKIP (not fail).
try:  # pragma: no cover - environment probe
    from dfextensions.dfdraw import DFDraw  # noqa: F401
    _HAVE_DFDRAW = True
except Exception:  # pragma: no cover
    _HAVE_DFDRAW = False

needs_dfdraw = pytest.mark.skipif(
    not _HAVE_DFDRAW, reason="dfdraw not importable via dfextensions.dfdraw")


def _mini_adf(n=64):
    return A.AliasDataFrame(pd.DataFrame({
        "x": np.arange(float(n)),
        "y": np.arange(float(n)) ** 2,
    }))


def _artists(ax):
    """Rendered-evidence count on an Axes (hist -> patches; line plots -> lines)."""
    return len(ax.patches) + len(ax.lines) + len(ax.collections)


# ---------------------------------------------------------------------------
# SEED-3 stage 0: mechanism (no ADF surface; mirrors panel evidence)
# ---------------------------------------------------------------------------

class TestSeed3Mechanism:
    def test_seed3_0_deepcopy_disconnects_axes_silently(self):
        fig, ax = plt.subplots()
        try:
            phantom = copy.deepcopy({"ax": ax})["ax"]
            assert phantom is not ax
            assert phantom.figure is not fig
            phantom.plot([0, 1], [0, 1])
            assert _artists(ax) == 0  # caller axes untouched, no exception
        finally:
            plt.close("all")


# ---------------------------------------------------------------------------
# SEED-3 public-surface characterization (FM#12: production entry points)
# ---------------------------------------------------------------------------

@needs_dfdraw
class TestSeed3AxIdentity:
    def test_seed3_1_draw_ax_kwarg_renders_into_caller_axes(self):
        adf = _mini_adf()
        fig, ax = plt.subplots()
        try:
            adf.draw("x", type="hist", ax=ax)
            assert _artists(ax) > 0, "Preserve row SEED-3.a regressed"
        finally:
            plt.close("all")

    def test_seed3_2_draw_batch_top_level_ax_renders_into_caller_axes(self):
        adf = _mini_adf()
        fig, ax = plt.subplots()
        try:
            adf.draw_batch({"p": {"expr": "x", "type": "hist"}},
                           ax=ax, verbose=False)
            assert _artists(ax) > 0, "Preserve row SEED-3.b regressed"
        finally:
            plt.close("all")

    @pytest.mark.xfail(
        strict=True,
        reason="SEED-3.c Repair pending: draw_batch deep-copies specs, so a "
               "per-spec ax is replaced by a disconnected phantom Axes "
               "(silent; caller subplot stays empty). Fix lands in Stage B "
               "structural-copy normalizer (Rev2 §9); on fix this XPASSes "
               "strictly and the marker must be removed.")
    def test_seed3_3_draw_batch_per_spec_ax_renders_into_caller_axes(self):
        adf = _mini_adf()
        fig, ax = plt.subplots()
        try:
            adf.draw_batch({"p": {"expr": "x", "type": "hist", "ax": ax}},
                           verbose=False)
            assert _artists(ax) > 0
        finally:
            plt.close("all")

    @pytest.mark.xfail(
        strict=True,
        reason="SEED-3.d Repair pending: draw_batch deep-copies defaults, so "
               "defaults={'ax': ...} is replaced by a disconnected phantom "
               "Axes (silent). Fix lands in Stage B structural-copy "
               "normalizer (Rev2 §9); on fix this XPASSes strictly and the "
               "marker must be removed.")
    def test_seed3_4_draw_batch_defaults_ax_renders_into_caller_axes(self):
        adf = _mini_adf()
        fig, ax = plt.subplots()
        try:
            adf.draw_batch({"p": {"expr": "y", "type": "hist"}},
                           defaults={"ax": ax}, verbose=False)
            assert _artists(ax) > 0
        finally:
            plt.close("all")


# ---------------------------------------------------------------------------
# C-1 caller non-mutation — before-state of the current deepcopy mechanism
# (OBS-1). The Stage-B structural-copy replacement must keep these green.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SEED-1 current-behavior pin (architect ruling 2026-07-19: Repair DEFERRED,
# owner=dfdraw; ADF must only guard its own forwarding contract). The
# deferred ACCEPTANCE test is tests/test_K1_vector_draw_kwarg_diagnostic.py::
# test_K1_3_... (strict xfail).
# ---------------------------------------------------------------------------

@needs_dfdraw
class TestSeed1BinsScatter:
    def test_seed1_1_batch_bins_scatter_forwarded_then_warn_ignored(self):
        """CURRENT behavior, pinned end-to-end: ADF forwards batch-level
        bins verbatim; dfdraw warn-and-ignores it for type='scatter'
        (dfdraw Phase 13.57.DF K-3). If the warning disappears, either
        dfdraw fixed SEED-1 (then K1_3 XPASSes and both markers are
        removed together) or the forwarding broke (then this fails)."""
        adf = _mini_adf()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = adf.draw_batch({"p": {"expr": "y:x", "type": "scatter"}},
                                 bins=10, verbose=False)
        plt.close("all")
        assert res["_summary"]["failed"] == 0, "scatter batch must succeed today"
        ignore_warnings = [w for w in caught
                           if "is not used by type" in str(w.message)]
        assert ignore_warnings, (
            "expected the dfdraw 13.57.DF K-3 warn-and-ignore for 'bins' on "
            "scatter; its absence means the seam behavior changed - "
            "reconcile SEED-1 markers")

    @pytest.mark.xfail(
        strict=True,
        reason="AD-5/13.76.ADF acceptance (Repair DEFERRED, owner=dfdraw): "
               "an EXPLICIT bins= on a scatter spec must raise a clean, "
               "actionable error pointing to profile/hist2d/hexbin - not "
               "the current warn-and-ignore. XPASS on the dfdraw fix "
               "forces marker removal.")
    def test_seed1_2_explicit_bins_on_scatter_spec_raises_clean_error(self):
        adf = _mini_adf()
        try:
            # AD-5 intended: a REAL exception whose message names the binned
            # alternatives (profile/hist2d/hexbin). Matching on those names
            # (not on 'bins'/'scatter') guarantees the current K-3
            # warn-and-ignore text can never satisfy this, even escalated.
            with pytest.raises(Exception,
                               match="(?i)profile|hist2d|hexbin"):
                adf.draw_batch(
                    {"p": {"expr": "y:x", "type": "scatter", "bins": 7}},
                    verbose=False)
        finally:
            plt.close("all")


# ---------------------------------------------------------------------------
# SEED-3.e — draw_figures × caller ax (both forms). Executed 2026-07-19:
# both crash with TypeError "multiple values for keyword argument 'ax'" at
# _draw_single_figure (composer passes its own grid ax while the deep-copied
# caller ax rides **merged). Semantic ruling pending (Q-C): Refused-with-
# clear-error vs symmetric render-into-caller-ax. EITHER WAY the current
# bare TypeError is an error-quality Repair (ADF-owned): these tests pin the
# current crash so it cannot change silently; they are characterization,
# not endorsement.
# ---------------------------------------------------------------------------

@needs_dfdraw
class TestSeed3eDrawFiguresAx:
    def test_seed3_5_draw_figures_per_plot_ax_current_typeerror(self):
        adf = _mini_adf()
        fig, ax = plt.subplots()
        try:
            with pytest.raises(TypeError, match="multiple values.*'ax'"):
                adf.draw_figures(
                    [{"name": "f1", "ncols": 1,
                      "plots": [{"expr": "x", "type": "hist", "ax": ax}]}],
                    verbose=False)
        finally:
            plt.close("all")

    def test_seed3_6_draw_figures_defaults_ax_current_typeerror(self):
        adf = _mini_adf()
        fig, ax = plt.subplots()
        try:
            with pytest.raises(TypeError, match="multiple values.*'ax'"):
                adf.draw_figures(
                    [{"name": "f2", "ncols": 1,
                      "plots": [{"expr": "y", "type": "hist"}]}],
                    defaults={"ax": ax}, verbose=False)
        finally:
            plt.close("all")

    @pytest.mark.xfail(
        strict=True,
        reason="AD-6/13.76.ADF acceptance (Repair, owner=ADF, fix in Stage B "
               "spec validation): draw_figures must reject caller-supplied "
               "ax with a clean ValueError BEFORE any figure/axes creation, "
               "naming draw/draw_batch as the caller-owned-axes "
               "alternatives - replacing today's raw TypeError keyword "
               "collision. When the Stage-B fix lands this XPASSes; remove "
               "the marker and retire the two TypeError crash-pin tests "
               "above in the same commit.")
    def test_seed3_7_draw_figures_ax_rejected_with_clean_valueerror(self):
        adf = _mini_adf()
        fig, ax = plt.subplots()
        try:
            with pytest.raises(ValueError, match="draw_figures"):
                adf.draw_figures(
                    [{"name": "f1", "ncols": 1,
                      "plots": [{"expr": "x", "type": "hist", "ax": ax}]}],
                    verbose=False)
        finally:
            plt.close("all")


# ---------------------------------------------------------------------------
# C-1 caller non-mutation — before-state of the current deepcopy mechanism
# (OBS-1). The Stage-B structural-copy replacement must keep these green.
# ---------------------------------------------------------------------------

@needs_dfdraw
class TestC1CallerNonMutation:
    def test_c1_1_draw_batch_caller_specs_and_defaults_unchanged(self):
        adf = _mini_adf()
        specs = {"p1": {"expr": "x", "type": "hist"},
                 "p2": {"expr": "y", "type": "hist", "bins": 10}}
        defaults = {"stats": False}
        specs_snapshot = copy.deepcopy(specs)
        defaults_snapshot = copy.deepcopy(defaults)
        try:
            adf.draw_batch(specs, defaults=defaults, verbose=False)
        finally:
            plt.close("all")
        assert specs == specs_snapshot, "caller specs mutated (C-1 breach)"
        assert defaults == defaults_snapshot, "caller defaults mutated (C-1 breach)"

    def test_c1_2_draw_figures_caller_specs_and_defaults_unchanged(self):
        adf = _mini_adf()
        figures = [{
            "name": "f1", "ncols": 2,
            "plots": [{"expr": "x", "type": "hist"},
                      {"expr": "y", "type": "hist"}],
        }]
        defaults = {"stats": False}
        figures_snapshot = copy.deepcopy(figures)
        defaults_snapshot = copy.deepcopy(defaults)
        try:
            adf.draw_figures(figures, defaults=defaults, verbose=False)
        finally:
            plt.close("all")
        assert figures == figures_snapshot, "caller figure specs mutated (C-1 breach)"
        assert defaults == defaults_snapshot, "caller defaults mutated (C-1 breach)"


# ---------------------------------------------------------------------------
# O-1 — cross-surface statistics equivalence oracle (Rev2 §10).
# The same plot request through draw / draw_batch / draw_figures must yield
# identical statistics. These are Preserve rows: Stage B's consolidated
# pipeline must keep every one green. Data is seeded; equality is exact-float
# (same computation path) with a 1e-12 relative guard for future numeric
# reordering, n strictly exact.
# ---------------------------------------------------------------------------

def _oracle_adf(n=500):
    rng = np.random.default_rng(20260719)
    return A.AliasDataFrame(pd.DataFrame({
        "x": rng.normal(0.0, 1.0, n),
        "w": rng.uniform(0.5, 2.0, n),
        "cat": rng.integers(0, 3, n).astype(float),
    }))


def _stats_triple(adf, plot_kwargs):
    """Run the identical plot through all three surfaces; return 3 stats."""
    _f, _a, s_draw = adf.draw(plot_kwargs["expr"],
                              **{k: v for k, v in plot_kwargs.items()
                                 if k != "expr"})
    res = adf.draw_batch({"p": dict(plot_kwargs)}, verbose=False)
    s_batch = res["p"]["stats"]
    r3 = adf.draw_figures(
        [{"name": "f", "ncols": 1, "plots": [dict(plot_kwargs)]}],
        verbose=False)
    s_fig = r3["f"]["stats"][0]
    plt.close("all")
    return s_draw, s_batch, s_fig


@needs_dfdraw
class TestO1CrossSurfaceStatsEquivalence:
    @pytest.mark.parametrize("label,extra", [
        ("plain", {}),
        ("selection", {"selection": "x>0"}),
        ("weights", {"weights": "w"}),
    ])
    def test_o1_stats_identical_across_three_surfaces(self, label, extra):
        adf = _oracle_adf()
        kw = {"expr": "x", "type": "hist", "bins": 20, **extra}
        s1, s2, s3 = _stats_triple(adf, kw)
        for key in ("n", "mean", "std", "median"):
            v1, v2, v3 = s1.get(key), s2.get(key), s3.get(key)
            assert v1 is not None, f"[{label}] draw stats missing {key}"
            if key == "n":
                assert v1 == v2 == v3, f"[{label}] n diverges: {v1},{v2},{v3}"
            else:
                assert v2 == pytest.approx(v1, rel=1e-12), (
                    f"[{label}] draw_batch {key} diverges: {v1} vs {v2}")
                assert v3 == pytest.approx(v1, rel=1e-12), (
                    f"[{label}] draw_figures {key} diverges: {v1} vs {v3}")


# ---------------------------------------------------------------------------
# O-2 — policy-independence oracle (Rev2 §10). Identical user syntax under
# different instance policies must yield identical statistical results; the
# policies may only change lifecycle effects (what stays materialized),
# never the numbers. Uses an alias so materialization actually engages.
# ---------------------------------------------------------------------------

@needs_dfdraw
class TestO2PolicyIndependence:
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("keep,clear", [
        (True, True), (True, False), (False, True), (False, False),
    ])
    def test_o2_policies_do_not_change_results(self, lazy, keep, clear):
        adf = _oracle_adf()
        adf.add_alias("z", "x*2 + 1")
        adf.draw_lazy = lazy
        adf.draw_keep_materialized = keep
        adf.draw_clear_after = clear
        if not lazy:
            # draw_lazy=False policy REQUIRES explicit materialization for
            # alias draws (instance-policy contract, AliasDataFrame.py:1034)
            adf.materialize_aliases(names=["z"])
        _f, _a, s_draw = adf.draw("z", type="hist", bins=15)
        res = adf.draw_batch({"p": {"expr": "z", "type": "hist", "bins": 15}},
                             verbose=False)
        plt.close("all")
        s_batch = res["p"]["stats"]
        ref = _oracle_adf()
        ref.add_alias("z", "x*2 + 1")
        ref.materialize_aliases(names=["z"])
        _f2, _a2, s_ref = ref.draw("z", type="hist", bins=15)
        plt.close("all")
        for key in ("n", "mean", "std"):
            assert s_draw.get(key) == pytest.approx(s_ref.get(key), rel=1e-12), (
                f"policy (lazy={lazy},keep={keep},clear={clear}) changed draw {key}")
            assert s_batch.get(key) == pytest.approx(s_ref.get(key), rel=1e-12), (
                f"policy (lazy={lazy},keep={keep},clear={clear}) changed batch {key}")


# ---------------------------------------------------------------------------
# Slot × surface sweep — the historically missed slots (facet_by, weights)
# plus the guarded set, characterized on draw and draw_batch. Success +
# stats presence is asserted for every cell; numeric cross-surface equality
# additionally for the row-filtering/weighting slots where it is
# well-defined. group_by/facet_by/color produce surface-managed composite
# output; their numeric layout is characterized in a later increment.
# ---------------------------------------------------------------------------

@needs_dfdraw
class TestSlotSurfaceSweep:
    @pytest.mark.parametrize("slot,value,numeric_equiv", [
        ("selection", "x>0", True),
        ("weights", "w", True),
        ("group_by", "cat", False),
        ("facet_by", "cat", False),
        ("color", "cat", False),
    ])
    def test_slot_accepted_on_draw_and_batch(self, slot, value, numeric_equiv):
        adf = _oracle_adf()
        kw = {"expr": "x", "type": "hist", "bins": 10, slot: value}
        _f, _a, s_draw = adf.draw(kw["expr"],
                                  **{k: v for k, v in kw.items()
                                     if k != "expr"})
        assert isinstance(s_draw, dict) and s_draw, (
            f"draw with {slot} returned no stats")
        res = adf.draw_batch({"p": dict(kw)}, verbose=False)
        plt.close("all")
        assert res["_summary"]["failed"] == 0, (
            f"draw_batch with {slot} failed: {res['_errors']}")
        s_batch = res["p"]["stats"]
        assert isinstance(s_batch, dict) and s_batch, (
            f"draw_batch with {slot} returned no stats")
        if numeric_equiv:
            for key in ("n", "mean", "std"):
                assert s_batch.get(key) == pytest.approx(
                    s_draw.get(key), rel=1e-12), (
                    f"{slot}: batch {key} diverges from draw")


# ---------------------------------------------------------------------------
# SWEEP-2 — vector slots (selection_vector / weights_vector), both contexts.
# Vector context (bracket expr '[y1,y2]:x'): slots engage per channel —
# proven numerically. Scalar context: both slots are SILENTLY inert today
# (no warning, no filtering) — pinned as characterization; under the AD-5
# explicit-provenance precedent this is a Repair candidate (explicit
# inapplicable input should error, not silently no-op); final classification
# goes to the Gate-A ruling batch, matrix row SWEEP-2.c.
# ---------------------------------------------------------------------------

def _vector_adf(n=200):
    rng = np.random.default_rng(7)
    return A.AliasDataFrame(pd.DataFrame({
        "y1": rng.normal(0.0, 1.0, n),
        "y2": rng.normal(5.0, 1.0, n),
        "x": rng.uniform(0.0, 1.0, n),
        "w": rng.uniform(0.5, 2.0, n),
    })), rng


@needs_dfdraw
class TestSweep2VectorSlots:
    def test_sweep2_1_selection_vector_filters_per_channel_on_draw(self):
        adf, _ = _vector_adf()
        df = adf.df
        _f, _a, st = adf.draw("[y1,y2]:x", type="profile", bins=6,
                              selection_vector=["y1>0", "y2>4"])
        plt.close("all")
        assert isinstance(st, list) and len(st) == 2
        assert st[0]["n"] == int((df.y1 > 0).sum())
        assert st[1]["n"] == int((df.y2 > 4).sum())

    def test_sweep2_2_vector_slots_accepted_on_draw_batch(self):
        adf, _ = _vector_adf()
        res = adf.draw_batch(
            {"p": {"expr": "[y1,y2]:x", "type": "profile", "bins": 6,
                   "selection_vector": ["y1>0", "y2>4"],
                   "weights_vector": ["w", "w"]}},
            verbose=False)
        plt.close("all")
        assert res["_summary"]["failed"] == 0, res["_errors"]
        st = res["p"]["stats"]
        assert isinstance(st, list) and len(st) == 2

    def test_sweep2_3_scalar_context_vector_slots_silently_inert_today(self):
        """Characterization pin of CURRENT behavior (not endorsement):
        selection_vector on a plain scalar draw neither filters nor warns.
        Executed evidence: n stays at the unfiltered count. Matrix
        SWEEP-2.c; Repair candidate per AD-5 explicit-provenance
        precedent; awaiting Gate-A ruling batch."""
        adf, _ = _vector_adf()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _f, _a, st = adf.draw("y1", type="hist", bins=8,
                                  selection_vector=["y1>0"])
        plt.close("all")
        assert st["n"] == len(adf.df), (
            "scalar-context selection_vector started filtering - "
            "reclassify SWEEP-2.c before changing this pin")
        assert not [w for w in caught
                    if "selection_vector" in str(w.message)], (
            "a warning appeared - update the SWEEP-2.c matrix row")


# ---------------------------------------------------------------------------
# SWEEP-3 — draw_figures column of the scalar-slot sweep (completes the
# surface axis for SWEEP-1). Success + stats for every slot; numeric
# equality to draw for the row-transforming slots.
# ---------------------------------------------------------------------------

@needs_dfdraw
class TestSweep3FiguresColumn:
    @pytest.mark.parametrize("slot,value,numeric_equiv", [
        ("selection", "x>0", True),
        ("weights", "w", True),
        ("group_by", "cat", False),
        ("color", "cat", False),
    ])
    def test_slot_accepted_on_draw_figures(self, slot, value, numeric_equiv):
        adf = _oracle_adf()
        kw = {"expr": "x", "type": "hist", "bins": 10, slot: value}
        _f, _a, s_draw = adf.draw(kw["expr"],
                                  **{k: v for k, v in kw.items()
                                     if k != "expr"})
        r3 = adf.draw_figures(
            [{"name": "f", "ncols": 1, "plots": [dict(kw)]}], verbose=False)
        plt.close("all")
        st = r3["f"]["stats"]
        assert isinstance(st, list) and st, (
            f"draw_figures with {slot} returned no stats")
        s_fig = st[0]
        if numeric_equiv and isinstance(s_fig, dict):
            for key in ("n", "mean", "std"):
                assert s_fig.get(key) == pytest.approx(
                    s_draw.get(key), rel=1e-12), (
                    f"{slot}: draw_figures {key} diverges from draw")

    def test_sweep3_facet_by_on_figures_is_documented_interim_refusal(self):
        """facet_by in a draw_figures panel is DELIBERATELY refused today
        with a clean, actionable message naming the alternative and the
        tracked dfdraw work (BUG_dfdraw_20260611_facet_by_ax_ignored,
        nested sub-gridspec). Matrix SWEEP-3.f: Repair - deferred (already
        dfdraw-tracked), interim refusal Preserve-quality. This pin keeps
        the refusal loud and its message intact until the dfdraw mechanism
        lands."""
        adf = _oracle_adf()
        try:
            with pytest.raises(ValueError,
                               match="facet_by is not supported in "
                                     "draw_figures"):
                adf.draw_figures(
                    [{"name": "f", "ncols": 1,
                      "plots": [{"expr": "x", "type": "hist", "bins": 10,
                                 "facet_by": "cat"}]}],
                    verbose=False)
        finally:
            plt.close("all")
