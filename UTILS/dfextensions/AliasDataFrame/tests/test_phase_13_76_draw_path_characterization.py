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
import json
import warnings
import os
try:  # package-style (alma2: dfextensions on path)
    from AliasDataFrame.AliasDataFrame import (
        _EffectiveDrawSpec, _DrawExecutionPolicy)
except (ImportError, ModuleNotFoundError):  # module-style (sandbox)
    from AliasDataFrame import _EffectiveDrawSpec, _DrawExecutionPolicy
try:
    from AliasDataFrame.AliasDataFrame import (
        _DrawDependencyPlan, _DrawPreparationState)
except (ImportError, ModuleNotFoundError):
    from AliasDataFrame import _DrawDependencyPlan, _DrawPreparationState

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

    def test_seed3_3_draw_batch_per_spec_ax_renders_into_caller_axes(self):
        adf = _mini_adf()
        fig, ax = plt.subplots()
        try:
            adf.draw_batch({"p": {"expr": "x", "type": "hist", "ax": ax}},
                           verbose=False)
            assert _artists(ax) > 0
        finally:
            plt.close("all")

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


# ---------------------------------------------------------------------------
# STATE-1 — data-state equivalence (last sweep axis, Rev2 §8.8): the same
# logical data drawn from an eager frame, a lazy tree, and a lazy 2-file
# chain must yield identical statistics; and the O-1 cross-surface oracle
# holds on the lazy state too. Fixtures are written per-test via uproot
# mktree (dict assignment would write RNTuple, not TTree — make_fixtures.py
# note) with seeded data.
# ---------------------------------------------------------------------------

uproot = pytest.importorskip("uproot")


def _write_tree(path, x, w):
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": "float64", "w": "float64"})
        f["tree"].extend({"x": x, "w": w})


@needs_dfdraw
class TestState1DataStateEquivalence:
    @pytest.fixture()
    def state_fixture(self, tmp_path):
        rng = np.random.default_rng(20260720)
        x = rng.normal(0.0, 1.0, 400)
        w = rng.uniform(0.5, 2.0, 400)
        p1 = str(tmp_path / "part1.root")
        p2 = str(tmp_path / "part2.root")
        _write_tree(p1, x[:250], w[:250])
        _write_tree(p2, x[250:], w[250:])
        eager = A.AliasDataFrame(pd.DataFrame({"x": x, "w": w}))
        return eager, p1, p2

    def test_state1_1_lazy_tree_stats_equal_eager(self, state_fixture):
        eager, p1, _p2 = state_fixture
        sub = A.AliasDataFrame(eager.df.iloc[:250].reset_index(drop=True))
        lazy = A.AliasDataFrame.read_tree_lazy(p1, "tree")
        _f, _a, s_e = sub.draw("x", type="hist", bins=16)
        _f2, _a2, s_l = lazy.draw("x", type="hist", bins=16, lazy=True)
        plt.close("all")
        assert s_l["n"] == s_e["n"] == 250
        for key in ("mean", "std", "median"):
            assert s_l[key] == pytest.approx(s_e[key], rel=1e-12), (
                f"lazy tree {key} diverges from eager")

    def test_state1_2_lazy_chain_stats_equal_eager_concat(self, state_fixture):
        eager, p1, p2 = state_fixture
        chain = A.AliasDataFrame.read_chain_lazy([p1, p2], "tree")
        _f, _a, s_e = eager.draw("x", type="hist", bins=16)
        _f2, _a2, s_c = chain.draw("x", type="hist", bins=16, lazy=True)
        plt.close("all")
        assert s_c["n"] == s_e["n"] == 400
        for key in ("mean", "std", "median"):
            assert s_c[key] == pytest.approx(s_e[key], rel=1e-12), (
                f"lazy chain {key} diverges from eager concat")

    def test_state1_3_o1_cross_surface_holds_on_lazy_tree(self, state_fixture):
        _eager, p1, _p2 = state_fixture
        lazy = A.AliasDataFrame.read_tree_lazy(p1, "tree")
        _f, _a, s_draw = lazy.draw("x", type="hist", bins=16, lazy=True)
        res = lazy.draw_batch(
            {"p": {"expr": "x", "type": "hist", "bins": 16}},
            lazy=True, verbose=False)
        plt.close("all")
        assert res["_summary"]["failed"] == 0, res["_errors"]
        s_batch = res["p"]["stats"]
        assert s_batch["n"] == s_draw["n"]
        for key in ("mean", "std"):
            assert s_batch[key] == pytest.approx(s_draw[key], rel=1e-12), (
                f"O-1 on lazy tree: batch {key} diverges from draw")

    def test_state1_4_weighted_selection_equivalence_on_chain(self, state_fixture):
        eager, p1, p2 = state_fixture
        chain = A.AliasDataFrame.read_chain_lazy([p1, p2], "tree")
        kw = {"type": "hist", "bins": 12, "selection": "x>0", "weights": "w"}
        _f, _a, s_e = eager.draw("x", **kw)
        _f2, _a2, s_c = chain.draw("x", lazy=True, **kw)
        plt.close("all")
        assert s_c["n"] == s_e["n"]
        for key in ("mean", "std"):
            assert s_c[key] == pytest.approx(s_e[key], rel=1e-12), (
                f"chain weighted+selected {key} diverges from eager")


# ---------------------------------------------------------------------------
# ENTRY-1 — entry-selection layer (entry_begin/entry_end/entry_mask),
# Rev2 §8.4 specification layer. draw and draw_figures implement it
# (named params; _apply_entry_selection) with exact numerics — Preserve.
# draw_batch has NO entry layer: the kwargs fall through **kwargs into
# dfdraw defaults and die inside matplotlib with a raw
# "Polygon.set() got an unexpected keyword argument 'entry_begin'" —
# an AD-4 asymmetry => Repair (ADF-owned; natural fix = the Stage-B
# EffectiveDrawSpec entry layer). Matrix rows ENTRY-1.a-d.
# ---------------------------------------------------------------------------

@needs_dfdraw
class TestEntry1EntryLayer:
    def _adf300(self):
        rng = np.random.default_rng(3)
        return A.AliasDataFrame(
            pd.DataFrame({"x": rng.normal(0.0, 1.0, 300)}))

    def test_entry1_1_draw_window_exact(self):
        adf = self._adf300()
        x = adf.df["x"].to_numpy()
        _f, _a, s = adf.draw("x", type="hist", bins=10,
                             entry_begin=50, entry_end=150)
        plt.close("all")
        assert s["n"] == 100
        assert s["mean"] == pytest.approx(x[50:150].mean(), rel=1e-12)

    def test_entry1_2_draw_mask_exact(self):
        adf = self._adf300()
        x = adf.df["x"].to_numpy()
        mask = np.zeros(300, bool)
        mask[::3] = True
        _f, _a, s = adf.draw("x", type="hist", bins=10, entry_mask=mask)
        plt.close("all")
        assert s["n"] == int(mask.sum())
        assert s["mean"] == pytest.approx(x[mask].mean(), rel=1e-12)

    def test_entry1_3_figures_window_exact(self):
        adf = self._adf300()
        x = adf.df["x"].to_numpy()
        r = adf.draw_figures(
            [{"name": "f", "ncols": 1,
              "plots": [{"expr": "x", "type": "hist", "bins": 10}]}],
            entry_begin=50, entry_end=150, verbose=False)
        plt.close("all")
        st = r["f"]["stats"][0]
        assert st["n"] == 100
        assert st["mean"] == pytest.approx(x[50:150].mean(), rel=1e-12)

    def test_entry1_5_batch_window_acceptance(self):
        adf = self._adf300()
        x = adf.df["x"].to_numpy()
        res = adf.draw_batch({"p": {"expr": "x", "type": "hist",
                                    "bins": 10}},
                             entry_begin=50, entry_end=150, verbose=False)
        plt.close("all")
        assert res["_summary"]["failed"] == 0
        s = res["p"]["stats"]
        assert s["n"] == 100
        assert s["mean"] == pytest.approx(x[50:150].mean(), rel=1e-12)


# ---------------------------------------------------------------------------
# B2 — consolidation contract (proposal §14): within ONE user draw call, the
# struct-catalog check does its full work exactly once, and each plot-
# specification dictionary (plus the shared defaults dictionary) is rewritten
# exactly once. The drawing entry points leave the counts in
# _last_draw_prep_stats. Measured before B2 on this fixture: 7 catalog runs
# and 5 rewrites per 2-plot batch/figures call; after: 1 and 3 (two plots +
# one defaults dictionary).
# ---------------------------------------------------------------------------

@needs_dfdraw
class TestB2ConsolidationContract:
    def _lazy_struct_adf(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    @pytest.mark.xfail(
        strict=True,
        reason="B3.4 acceptance (re-anchored from B3.2, disclosed): the "
               "catalog INVOCATION count reaches 1 only when the demolition "
               "step removes the defensive re-checks inside shared helpers "
               "(get_required_branches, _dict_dispatch_columns) for "
               "executor-owned flows; the executor already performs the one "
               "owned ensure, and helper re-checks take the fingerprint "
               "fast path")
    def test_b2_1_draw_single_prep_pass(self):
        adf = self._lazy_struct_adf()
        adf.draw("dedxTPC.dEdxMaxTPC:mult", type="profile", bins=5,
                 lazy=True)
        plt.close("all")
        st = adf._last_draw_prep_stats
        assert st["catalog_full_runs"] == 1, st
        assert st["rewrite_full_runs"] == 1, st

    def test_b2_2_batch_rewrite_once_per_dict_by_construction(self):
        """B3.2 ACCEPTANCE, GREEN BY CONSTRUCTION: the single side-effect
        executor rewrites each dictionary exactly once for the whole batch
        call — two plot dictionaries plus the (empty) top-level kwargs
        dictionary here — and its preparation-state record says so."""
        adf = self._lazy_struct_adf()
        adf.draw_batch(
            {"a": {"expr": "dedxTPC.dEdxMaxTPC:mult", "type": "profile",
                   "bins": 5},
             "b": {"expr": "dedxTPC.dEdxTotTPC", "type": "hist", "bins": 5}},
            lazy=True, verbose=False)
        plt.close("all")
        assert adf._last_draw_prep_stats["rewrite_full_runs"] == 3
        state = adf._last_draw_prep_state
        assert state.catalog_ensured is True
        # Ruling 3 (2026-07-25): stated structurally, not as a fixture-shaped
        # literal. See _expected_rewrite_count for why the old '3' was an
        # artifact (two specs + the empty top-level kwargs dictionary).
        assert state.dicts_rewritten == _expected_rewrite_count(
            None, {},
            {"a": {"expr": "dedxTPC.dEdxMaxTPC:mult", "type": "profile",
                   "bins": 5},
             "b": {"expr": "dedxTPC.dEdxTotTPC", "type": "hist", "bins": 5}})
        assert "dedxTPC/dEdxMaxTPC" in state.branches_loaded

    @pytest.mark.xfail(
        strict=True,
        reason="B3.4 acceptance (catalog invocation count; see b2_1 reason) "
               "- batch surface")
    def test_b2_2b_batch_single_catalog_invocation(self):
        adf = self._lazy_struct_adf()
        adf.draw_batch(
            {"a": {"expr": "dedxTPC.dEdxMaxTPC:mult", "type": "profile",
                   "bins": 5},
             "b": {"expr": "dedxTPC.dEdxTotTPC", "type": "hist", "bins": 5}},
            lazy=True, verbose=False)
        plt.close("all")
        assert adf._last_draw_prep_stats["catalog_full_runs"] == 1

    @pytest.mark.xfail(
        strict=True,
        reason="B3.3 acceptance: draw_figures migrates onto the dependency "
               "plan and single executor in the next increment")
    def test_b2_3_figures_one_catalog_run_one_rewrite_per_dict(self):
        adf = self._lazy_struct_adf()
        adf.draw_figures(
            [{"name": "f", "ncols": 2,
              "plots": [{"expr": "dedxTPC.dEdxMaxTPC:mult",
                         "type": "profile", "bins": 5},
                        {"expr": "dedxTPC.dEdxTotTPC", "type": "hist",
                         "bins": 5}]}],
            lazy=True, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_stats
        assert st["catalog_full_runs"] == 1, st
        assert st["rewrite_full_runs"] == 3, st

    def test_b2_4_results_identical_with_consolidation(self):
        """The consolidation must not change a single number: same plot
        through draw and draw_batch on the struct fixture, statistics
        identical (this is the O-1 oracle applied to the consolidated
        preparation path on struct-bearing lazy data)."""
        adf1 = self._lazy_struct_adf()
        _f, _a, s_draw = adf1.draw("dedxTPC.dEdxMaxTPC:mult",
                                   type="profile", bins=5, lazy=True)
        adf2 = self._lazy_struct_adf()
        res = adf2.draw_batch(
            {"p": {"expr": "dedxTPC.dEdxMaxTPC:mult", "type": "profile",
                   "bins": 5}}, lazy=True, verbose=False)
        plt.close("all")
        s_batch = res["p"]["stats"]
        assert s_batch["n"] == s_draw["n"]
        for key in ("mean_x", "mean_y"):
            if key in s_draw:
                assert s_batch[key] == pytest.approx(s_draw[key], rel=1e-12)


# ---------------------------------------------------------------------------
# B3.1 — EffectiveDrawSpec + DrawExecutionPolicy on draw() (Rev 2 §11.1/2).
# The old inline head survives verbatim behind ADF_B3_OLD_DRAW_PATH=1 until
# step B3.4; these tests hold the two paths against each other. Counting is
# INDEPENDENT (monkeypatch of the real helpers), per the GPT24 review of B2:
# the oracle is not allowed to be the implementation's own counters.
# ---------------------------------------------------------------------------

class _CallCounter:
    """Wrap a real method; count invocations; delegate unchanged."""
    def __init__(self, obj, name):
        self.count = 0
        self._orig = getattr(obj, name)
        self._obj, self._name = obj, name
        def spy(*a, **k):
            self.count += 1
            return self._orig(*a, **k)
        setattr(obj, name, spy)
    def restore(self):
        setattr(self._obj, self._name, self._orig)


@needs_dfdraw
class TestB31EffectiveSpecOnDraw:
    def _fresh_eager(self):
        rng = np.random.default_rng(31)
        return A.AliasDataFrame(pd.DataFrame({
            "x": rng.normal(0, 1, 300), "w": rng.uniform(.5, 2, 300),
            "cat": rng.integers(0, 3, 300).astype(float)}))

    def _fresh_lazy(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    def _stats_under(self, monkeypatch, old_path, make_adf, draw_args,
                     draw_kwargs):
        if old_path:
            monkeypatch.setenv("ADF_B3_OLD_DRAW_PATH", "1")
        else:
            monkeypatch.delenv("ADF_B3_OLD_DRAW_PATH", raising=False)
        adf = make_adf()
        _f, _a, st = adf.draw(*draw_args, **draw_kwargs)
        plt.close("all")
        return st

    @pytest.mark.parametrize("label,args,kw", [
        ("plain",     ("x",), {"type": "hist", "bins": 12}),
        ("slots",     ("x",), {"type": "hist", "bins": 12,
                               "selection": "x>0", "weights": "w"}),
        ("entry",     ("x",), {"type": "hist", "bins": 12,
                               "entry_begin": 50, "entry_end": 200}),
        ("vector",    ("[x,w]:cat",), {"type": "profile", "bins": 4,
                                       "selection_vector": ["x>0", "w>1"]}),
    ])
    def test_b31_1_old_and_new_paths_produce_identical_stats(
            self, monkeypatch, label, args, kw):
        s_old = self._stats_under(monkeypatch, True, self._fresh_eager,
                                  args, kw)
        s_new = self._stats_under(monkeypatch, False, self._fresh_eager,
                                  args, kw)
        olds = s_old if isinstance(s_old, list) else [s_old]
        news = s_new if isinstance(s_new, list) else [s_new]
        assert len(olds) == len(news), (
            f"[{label}] channel count differs: {len(olds)} vs {len(news)}")
        for ch, (o, n) in enumerate(zip(olds, news)):
            for key in ("n", "mean", "std", "mean_x", "mean_y"):
                if key in o:
                    assert n[key] == pytest.approx(o[key], rel=1e-12), (
                        f"[{label}] ch{ch} {key}: old {o[key]} vs new {n[key]}")

    def test_b31_2_lazy_struct_identical_and_helpers_called_equally(
            self, monkeypatch):
        results = {}
        for tag, old in (("old", True), ("new", False)):
            if old:
                monkeypatch.setenv("ADF_B3_OLD_DRAW_PATH", "1")
            else:
                monkeypatch.delenv("ADF_B3_OLD_DRAW_PATH", raising=False)
            adf = self._fresh_lazy()
            spies = {nm: _CallCounter(adf, nm) for nm in (
                "_ensure_vector_kwargs_aliases",
                "_normalize_vector_compose_kwargs",
                "get_required_branches",
                "_lazy_ensure_subframe_refs")}
            try:
                _f, _a, st = adf.draw("dedxTPC.dEdxMaxTPC:mult",
                                      type="profile", bins=5, lazy=True)
            finally:
                counts = {nm: sp.count for nm, sp in spies.items()}
                for sp in spies.values():
                    sp.restore()
            plt.close("all")
            results[tag] = (st, counts)
        st_old, c_old = results["old"]
        st_new, c_new = results["new"]
        assert c_new == c_old, (
            f"helper invocation counts diverge: old {c_old} vs new {c_new}")
        for key in ("n", "mean_x", "mean_y"):
            if key in st_old:
                assert st_new[key] == pytest.approx(st_old[key], rel=1e-12)

    def test_b31_3_spec_slot_set_covers_scan_gap_slots(self):
        """The one-source slot list must contain the historically missed
        parameters (facet_by, weights, weights_vector, selection_vector —
        the Phase 13.58 scan-gap record), and required_branch_kwargs must
        map every slot plus the expression."""
        for missed in ("facet_by", "weights", "weights_vector",
                       "selection_vector"):
            assert missed in _EffectiveDrawSpec.SLOT_NAMES
        spec = _EffectiveDrawSpec.from_call(
            "x", "hist", {"selection": "x>0", "weights": "w"})
        rk = spec.required_branch_kwargs()
        assert rk["expr"] == "x" and rk["selection"] == "x>0"
        assert set(rk) == {"expr", *_EffectiveDrawSpec.SLOT_NAMES}

    def test_b31_4_policy_resolution_matches_instance_flags(self):
        adf = self._fresh_eager()
        adf.draw_lazy = True
        adf.draw_keep_materialized = False
        pol = _DrawExecutionPolicy.resolve(adf)
        assert pol.lazy is True and pol.keep_materialized is False
        pol2 = _DrawExecutionPolicy.resolve(adf, lazy=False,
                                             keep_materialized=True)
        assert pol2.lazy is False and pol2.keep_materialized is True


    def test_b31_5_effective_spec_construction_is_pure(self):
        """GPT25 blocking finding 1: building the specification record must
        not load branches, materialize aliases, or mutate anything."""
        adf = self._fresh_lazy()
        cols_before = list(adf.df.columns)
        loaded_before = set(adf._lazy_reader.loaded_branches)
        mat_before = set(adf._get_materialized_aliases())
        kwargs = {"selection": "mult>0",
                  "selection_vector": ["dedxTPC.dEdxMaxTPC>0"]}
        kwargs_before = dict(kwargs)
        spec = _EffectiveDrawSpec.from_call("dedxTPC.dEdxMaxTPC:mult",
                                            "profile", kwargs,
                                            entry_begin=1, entry_end=5)
        assert list(adf.df.columns) == cols_before
        assert set(adf._lazy_reader.loaded_branches) == loaded_before
        assert set(adf._get_materialized_aliases()) == mat_before
        assert kwargs == kwargs_before
        assert spec.has_entry_selection()

    def test_b31_6_reference_blob_covers_every_slot_from_one_source(self):
        """GPT25 subframe-coverage correction: the pre-scan text is derived
        from SLOT_NAMES, so every slot - scalar and vector - reaches it.
        Source-derived: markers are injected per slot name, no hand list."""
        style = {}
        for i, name in enumerate(_EffectiveDrawSpec.SLOT_NAMES):
            marker = f"SUBQ{i}.col{i}"
            style[name] = [marker] if name.endswith("_vector") else marker
        spec = _EffectiveDrawSpec.from_call("EXPRMARK:x", "hist", style)
        blob = spec.reference_text_blob(include_vector_slots=True)
        assert "EXPRMARK" in blob
        for i, name in enumerate(_EffectiveDrawSpec.SLOT_NAMES):
            assert f"SUBQ{i}.col{i}" in blob, f"slot {name} missing from blob"


    def test_b31_7_blob_never_crashes_on_array_valued_vector_slots(self):
        """GPT27 item 2: a numpy-array-valued vector slot previously hit
        "truth value of an array is ambiguous" in the blob join. Only real
        strings may reach the join; arrays are ignored, not stringified."""
        arr = np.linspace(0.5, 2.0, 7)
        spec = _EffectiveDrawSpec.from_call(
            "x", "hist", {"weights_vector": arr,
                          "selection_vector": ["SUBQ.col>0", arr],
                          "weights": arr})
        blob = spec.reference_text_blob(include_vector_slots=True)
        assert "SUBQ.col>0" in blob
        assert "linspace" not in blob and "[" not in blob

    def test_b31_8_record_isolated_from_later_caller_dict_rewrites(self):
        """GPT25 item 4: the record holds a structural copy; rewriting the
        caller's dictionary AFTER construction must not alter the record."""
        style = {"selection": "x>0", "weights": "w",
                 "selection_vector": ["a>1", "b>2"]}
        spec = _EffectiveDrawSpec.from_call("x", "hist", style)
        style["selection"] = "MUTATED"
        style["selection_vector"].append("MUTATED_ELEMENT")
        style.pop("weights")
        assert spec.slot("selection") == "x>0"
        assert spec.slot("weights") == "w"
        assert list(spec.slot("selection_vector")) == ["a>1", "b>2"]

    def test_b31_9_prescan_and_refusal_equivalence_old_vs_new_path(
            self, monkeypatch):
        """GPT27 item 3, answered by execution. (a) The subframe pre-scan
        receives IDENTICAL text under old and new paths for scalar slots —
        the widened vector-slot scan is deferred to B3.2, so B3.1 triggers
        no loading the old path did not. (b) A subframe-qualified reference
        inside a vector slot is refused by the EXISTING tracked guard
        (BUG_20260701_ADF_subframe_ref_slot_symmetry) with the same
        exception and message under BOTH paths, and that refusal fires
        before any pre-scan materialization could occur."""
        captured, refusals = {}, {}
        for tag, old in (("old", True), ("new", False)):
            if old:
                monkeypatch.setenv("ADF_B3_OLD_DRAW_PATH", "1")
            else:
                monkeypatch.delenv("ADF_B3_OLD_DRAW_PATH", raising=False)
            adf = self._fresh_lazy()
            calls = []
            orig = adf._lazy_ensure_subframe_refs
            def spy(text, _orig=orig, _calls=calls):
                _calls.append(text)
                return _orig(text)
            adf._lazy_ensure_subframe_refs = spy
            _f, _a, _st = adf.draw("dedxTPC.dEdxMaxTPC:mult",
                                   type="profile", bins=5, lazy=True,
                                   selection="mult>0")
            plt.close("all")
            # GPT26 round-3: the refused call must trigger NO further
            # pre-scan and NO state change — every call is recorded (not
            # just the last), and loaded branches / columns / materialized
            # aliases are snapshotted around the refusal.
            calls_before = list(calls)
            loaded_before = set(adf._lazy_reader.loaded_branches)
            cols_before = list(adf.df.columns)
            mat_before = set(adf._get_materialized_aliases())
            with pytest.raises(ValueError,
                               match="not yet supported") as ei:
                adf.draw("mult", type="hist", bins=5, lazy=True,
                         selection_vector=["dedxTPC.dEdxTotTPC>0"])
            plt.close("all")
            # The refused call MAY run its scalar pre-scan first (both
            # paths do, identically); it must NOT pre-scan the vector-slot
            # reference and must not change any state.
            extra = calls[len(calls_before):]
            assert all("dedxTPC.dEdxTotTPC" not in t for t in extra), (
                "the vector-slot subframe reference reached the pre-scan "
                "before the refusal fired")
            assert set(adf._lazy_reader.loaded_branches) == loaded_before
            assert list(adf.df.columns) == cols_before
            assert set(adf._get_materialized_aliases()) == mat_before
            captured[tag] = list(calls)
            refusals[tag] = str(ei.value)
        assert captured["new"] == captured["old"], (
            f"pre-scan call sequences diverge:\nold: {captured['old']}\n"
            f"new: {captured['new']}")
        assert refusals["new"] == refusals["old"], (
            "vector-slot subframe refusal message diverges between paths")


    @staticmethod
    def _deep_numeric_equal(a, b, path=""):
        """Recursive comparison for stats payloads: dicts, lists/tuples,
        numpy arrays and scalars, exact to 1e-12 relative."""
        import numbers
        if isinstance(a, dict) and isinstance(b, dict):
            assert set(a) == set(b), f"{path}: keys differ"
            for k in a:
                TestB31EffectiveSpecOnDraw._deep_numeric_equal(
                    a[k], b[k], f"{path}.{k}")
        elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            assert len(a) == len(b), f"{path}: length differs"
            for i, (x, y) in enumerate(zip(a, b)):
                TestB31EffectiveSpecOnDraw._deep_numeric_equal(
                    x, y, f"{path}[{i}]")
        elif isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
            assert list(a.columns) == list(b.columns), f"{path}: columns"
            for col in a.columns:
                np.testing.assert_allclose(
                    a[col].to_numpy(dtype=float),
                    b[col].to_numpy(dtype=float),
                    rtol=1e-12, err_msg=f"{path}.{col}")
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b),
                                       rtol=1e-12, err_msg=path)
        elif isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            assert b == pytest.approx(a, rel=1e-12, nan_ok=True), path
        else:
            assert a == b, path

    @pytest.mark.parametrize("label,expr,extra", [
        ("scalar", "x:cat", {"weights": "w"}),
        ("vector", "[x,w]:cat", {"selection_vector": ["x>0", "w>1"]}),
    ])
    def test_b31_10_profile_data_per_bin_identical_old_vs_new(
            self, monkeypatch, label, expr, extra):
        """GPT26/GPT27 round-3 (the P1-8 defect shape): summary statistics
        are not enough — the complete per-bin profile payload must be
        identical between the old and new paths, for EVERY channel of a
        vector request (GPT26 round-4 extension: the vector path is where
        B3.1's real defects lived)."""
        payloads = {}
        for tag, old in (("old", True), ("new", False)):
            if old:
                monkeypatch.setenv("ADF_B3_OLD_DRAW_PATH", "1")
            else:
                monkeypatch.delenv("ADF_B3_OLD_DRAW_PATH", raising=False)
            adf = self._fresh_eager()
            _f, _a, st = adf.draw(expr, type="profile", bins=3,
                                  return_data=True, **extra)
            plt.close("all")
            channels = st if isinstance(st, list) else [st]
            for ch, cst in enumerate(channels):
                assert "profile_data" in cst, (
                    f"[{label}] ch{ch}: profile_data missing despite "
                    "return_data=True (public contract, 13.75 P1-8)")
            payloads[tag] = channels
        assert len(payloads["old"]) == len(payloads["new"]), (
            f"[{label}] channel count differs")
        for ch, (o, n) in enumerate(zip(payloads["old"], payloads["new"])):
            self._deep_numeric_equal(o["profile_data"], n["profile_data"],
                                     f"[{label}] profile_data[{ch}]")

    @pytest.mark.parametrize("shape", ["eager_alias_selection",
                                       "lazy_struct_profile",
                                       "vector_selection"])
    def test_b31_11_delegated_frame_columns_identical_old_vs_new(
            self, monkeypatch, shape):
        """Bounded-plan requirement three of three: the FRAME handed to
        dfdraw is compared, not only the returned statistics — identical
        columns and row count under old and new paths, on the three shapes
        this refactor actually touched (GPT26 round-4: eager alias with
        selection; lazy struct profile; vector with selection_vector)."""
        import sys as _sys
        # warm-up so the dfdraw module ADF uses is loaded, then patch every
        # loaded candidate name (package-style on alma2 imports 'dfdraw';
        # the sandbox loads 'dfextensions.dfdraw')
        warm = self._fresh_eager()
        warm.draw("x", type="hist", bins=4)
        plt.close("all")
        mods = [_sys.modules[k] for k in ("dfdraw", "dfextensions.dfdraw")
                if k in _sys.modules]
        assert mods, "no dfdraw module loaded after a successful draw"
        delegated = {}
        real_cls = mods[0].DFDraw
        for tag, old in (("old", True), ("new", False)):
            if old:
                monkeypatch.setenv("ADF_B3_OLD_DRAW_PATH", "1")
            else:
                monkeypatch.delenv("ADF_B3_OLD_DRAW_PATH", raising=False)
            frames = []
            class _Spy(real_cls):
                def __init__(self, df, *a, **k):
                    frames.append((list(df.columns), len(df)))
                    super().__init__(df, *a, **k)
            for _m in mods:
                monkeypatch.setattr(_m, "DFDraw", _Spy)
            if shape == "eager_alias_selection":
                adf = self._fresh_eager()
                adf.add_alias("z", "x*2")
                adf.draw_lazy = True
                _f, _a, _st = adf.draw("z", type="hist", bins=8,
                                       selection="x>0")
            elif shape == "lazy_struct_profile":
                adf = self._fresh_lazy()
                _f, _a, _st = adf.draw("dedxTPC.dEdxMaxTPC:mult",
                                       type="profile", bins=5, lazy=True)
            else:  # vector_selection
                adf = self._fresh_eager()
                _f, _a, _st = adf.draw("[x,w]:cat", type="profile", bins=3,
                                       selection_vector=["x>0", "w>1"])
            plt.close("all")
            delegated[tag] = frames
        assert delegated["new"] == delegated["old"], (
            f"delegated frames diverge:\nold: {delegated['old']}\n"
            f"new: {delegated['new']}")


@needs_dfdraw
class TestB32PlanExecutor:
    def _lazy(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    def test_b32_1_plan_building_is_pure(self):
        """Rev 2 §11.3: building the dependency plan has NO effects."""
        adf = self._lazy()
        loaded = set(adf._lazy_reader.loaded_branches)
        cols = list(adf.df.columns)
        e = _EffectiveDrawSpec.from_call("dedxTPC.dEdxMaxTPC:mult",
                                        "profile", {"bins": 5})
        plan = _DrawDependencyPlan(especs=[e], rewrite_dicts=[{}],
                                   autoload_dicts=[])
        assert plan.prescan_text()  # computable without effects
        assert set(adf._lazy_reader.loaded_branches) == loaded
        assert list(adf.df.columns) == cols

    def test_b32_2_midcall_loading_still_normalized_on_batch(self):
        """GPT25/GPT26 adversarial requirement: a struct branch that only
        becomes needed (and loaded) DURING the call — here via a second
        spec whose branch was not touched before — is still normalized and
        drawn correctly; the executor's single pass covers the union up
        front, so nothing depends on a mid-call refresh."""
        adf = self._lazy()
        assert "dEdxTotTPC__dedxTPC" not in adf.df.columns
        res = adf.draw_batch(
            {"a": {"expr": "dedxTPC.dEdxMaxTPC:mult", "type": "profile",
                   "bins": 5},
             "b": {"expr": "dedxTPC.dEdxTotTPC", "type": "hist",
                   "bins": 5}},
            lazy=True, verbose=False)
        plt.close("all")
        assert res["_summary"]["failed"] == 0, res["_errors"]
        assert res["b"]["stats"]["n"] > 0

    def test_b32_3_executor_state_records_what_ran(self):
        adf = self._lazy()
        adf.draw_batch(
            {"a": {"expr": "dedxTPC.dEdxMaxTPC:mult", "type": "profile",
                   "bins": 5}},
            lazy=True, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert st.catalog_ensured is True
        # Ruling 3: structural, not a fixture-shaped literal (one spec
        # dictionary plus the empty top-level kwargs dictionary here).
        assert st.dicts_rewritten == _expected_rewrite_count(
            None, {}, {"only": {}})
        assert st.prescan_text  # scalar-slot text was scanned
        assert all("/" in b or b == "mult" for b in st.branches_loaded)

    def test_b32_4_defaults_supplied_expr_wins_over_name_fallback(self, tmp_path):
        """F-1 regression guard (B3.2 panel, unanimous P0), non-skipping:
        a spec of {} with a defaults-supplied expr must draw the defaults
        expression — the plot-name fallback may never poison the raw spec
        before the merge. Mirrors test_batch_with_defaults on a fixture
        that runs everywhere."""
        rng = np.random.default_rng(324)
        p = str(tmp_path / "defaults_fix.root")
        _write_tree(p, rng.normal(0.0, 1.0, 120), rng.uniform(0.5, 2, 120))
        adf = A.AliasDataFrame.read_tree_lazy(p, "tree")
        res = adf.draw_batch({"hist_x": {}},
                             defaults={"expr": "x", "type": "hist",
                                       "bins": 8},
                             lazy=True, verbose=False)
        plt.close("all")
        assert res["_summary"]["failed"] == 0, res["_errors"]
        assert res["hist_x"]["stats"]["n"] == 120
        assert "x" in adf._lazy_reader.loaded_branches


# ---------------------------------------------------------------------------
# PHASE_13_76_ADF B3.2 — architect Ruling 2 (2026-07-25).
#
# The ruling: "Do not defer the catalog requirement solely on assertion.
# Include the adversarial test. A reachable effect must move under the
# executor; a proven no-op may be physically removed in B3.4 only after an
# explicit recorded ruling."
#
# Executed answer (this file, these tests): a residual catalog re-check CAN
# perform a real effect. _ensure_struct_catalog has two legs. The detection
# leg (detect_structs) is guarded by a fingerprint over the reader's
# available_branches, which is static per file — that leg genuinely cannot
# re-fire mid-call. The D-3 full-structure-completion leg has NO such guard:
# it re-tests the frame's columns on every invocation and loads branches
# whenever a preceding load left a struct half-populated. Before this
# increment that made struct completion reachable from the defensive
# re-checks inside get_required_branches / _dict_dispatch_columns AFTER
# _execute_draw_plan returned — and it did so exactly when the struct
# reference lived in a per-spec dictionary rather than in defaults, because
# the plan's autoload work-list covered only [defaults, kwargs].
#
# The fix is positional, not list-shaped: the executor calls
# _complete_partial_structs() itself, immediately after its own load. Making
# it depend on which dictionary carried the reference is what produced the
# defect in the first place.
#
# Two tests below, deliberately paired:
#   * the BOUNDARY test states the guarantee (nothing loads after the
#     executor returns);
#   * the CAPABILITY test proves the boundary test is not vacuous — the
#     residual path still acts when the state it reacts to is constructed by
#     hand. Without it, the boundary test would keep passing if the D-3 leg
#     were silently deleted, and B3.4's removal would lose its safety
#     argument.
# ---------------------------------------------------------------------------


def _expected_rewrite_count(defaults, kwargs, specs):
    """Rewrite-pass expectation, stated structurally rather than as a
    fixture-shaped integer (architect Ruling 3, 2026-07-25: 'one
    authoritative rewrite pass per public call, traversing each effective
    specification dictionary exactly once').

    The literal '3' this replaced was an artifact: two spec dictionaries
    plus the EMPTY top-level kwargs dictionary, which passes the plan's
    isinstance(dict) filter. A call supplying defaults would give 4;
    defaults=None is filtered out. Counting the dictionaries that actually
    exist is fixture-independent and self-describing."""
    return len([d for d in (defaults, kwargs, *specs.values())
                if isinstance(d, dict)])


class _EffectTrace:
    """Independent oracle: wraps the REAL preparation helpers on the class and
    records every invocation, tagged by whether _execute_draw_plan was on the
    stack at the time. Deliberately not the implementation's own counters —
    self-reported counters were reviewer-rejected twice in this phase."""

    # B32P1-5 (GPT25/GPT27, P1): the tracer must observe EVERY effect the
    # executor contract names, otherwise the strict fail-before marker below
    # can XPASS while subframe joins, temporary columns or cleanup are still
    # running outside the executor. Three method names were not enough to
    # guard a six-effect boundary.
    METHODS = ("ensure_branches", "ensure_struct",
               "materialize_alias", "materialize_aliases",
               "_ensure_vector_kwargs_aliases", "_prepare_subframe_joins",
               "dematerialize")

    def __init__(self, cls):
        self.cls = cls
        self.inside = []
        self.after = []
        self._depth = 0
        self._orig = {}

    def __enter__(self):
        cls = self.cls
        # The executor is multi-phase by architect ruling (2026-07-25):
        # preparation before the draw, cleanup after the render. "Inside the
        # executor" means inside ANY of its phases, so both are counted.
        self._orig["_execute_draw_plan"] = cls._execute_draw_plan
        real_exec = cls._execute_draw_plan

        def exec_(inner_self, plan, verbose=False):
            self._depth += 1
            try:
                return real_exec(inner_self, plan, verbose=verbose)
            finally:
                self._depth -= 1

        cls._execute_draw_plan = exec_
        self._orig["_execute_draw_cleanup"] = cls._execute_draw_cleanup
        real_clean = cls._execute_draw_cleanup

        def clean_(inner_self, state, clear_after, verbose=False):
            self._depth += 1
            try:
                return real_clean(inner_self, state, clear_after,
                                  verbose=verbose)
            finally:
                self._depth -= 1

        cls._execute_draw_cleanup = clean_
        for name in self.METHODS:
            real = getattr(cls, name)
            self._orig[name] = real

            def make(nm, fn):
                def wrapper(inner_self, *a, **k):
                    label = f"{nm}({a[0]!r})" if a else nm
                    (self.inside if self._depth else self.after).append(label)
                    return fn(inner_self, *a, **k)
                return wrapper

            setattr(cls, name, make(name, real))
        return self

    def __exit__(self, *exc):
        for name, fn in self._orig.items():
            setattr(self.cls, name, fn)
        return False


@needs_dfdraw
class TestB32ExecutorBoundary:
    """Ruling 2 guarantee: struct/branch preparation does not escape the
    executor. Scope is stated honestly — this increment owns catalog,
    pre-scan, branch loading, full-structure completion and the rewrite
    pass. Alias materialization is NOT yet owned; the strict-xfail sibling
    below is its fail-before evidence and flips when that migration lands."""

    def _lazy(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    CASES = {
        "struct_ref_in_spec_only": (
            {"p1": {"expr": "dedxTPC.dEdxMaxTPC:mult", "type": "scatter"}},
            None),
        "struct_ref_in_defaults": (
            {"p1": {"type": "scatter"}},
            {"expr": "dedxTPC.dEdxMaxTPC:mult"}),
        "struct_ref_in_second_spec": (
            {"p0": {"expr": "tgl:mult", "type": "scatter"},
             "p1": {"expr": "dedxTPC.dEdxTotTPC:mult", "type": "scatter"}},
            None),
    }

    @pytest.mark.parametrize("case", sorted(CASES))
    def test_b32_5_no_branch_or_struct_effect_after_executor(self, case):
        """The regression this closes: with the struct reference in a
        per-spec dictionary the executor loaded one member and returned,
        and a downstream defensive catalog re-check then completed the
        struct — real branch I/O outside the single effect owner."""
        specs, defaults = self.CASES[case]
        adf = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _EffectTrace(A.AliasDataFrame) as tr:
                adf.draw_batch(specs, defaults=defaults, verbose=False)
        plt.close("all")
        escaped = [e for e in tr.after
                   if e.startswith(("ensure_branches", "ensure_struct"))]
        assert not escaped, (
            f"preparation effect(s) outside _execute_draw_plan: {escaped}; "
            f"inside was {tr.inside}")
        assert any(e.startswith("ensure_branches") for e in tr.inside), (
            "no load happened inside the executor either — the trace is "
            "not exercising the path it claims to")

    def test_b32_5b_executor_state_records_the_completion(self):
        """Rev 2 §11.5: the effect must be reported, not merely performed."""
        specs, defaults = self.CASES["struct_ref_in_spec_only"]
        adf = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch(specs, defaults=defaults, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert st.structs_completed == ("dedxTPC",), st.structs_completed
        assert "dedxTPC/dEdxMaxTPC" in st.branches_loaded
        # B32P1-4 (GPT25, P1): reconcile against EXTERNAL state, not against
        # the one field the executor happened to set. The previous version
        # passed while the record silently omitted the sibling branches that
        # completion itself read.
        actually = set(map(str, adf._lazy_reader.loaded_branches))
        assert set(st.branches_loaded) == actually, (
            f"record {sorted(st.branches_loaded)} != reader "
            f"{sorted(actually)}")
        assert st.reads_by_completion, (
            "completion read nothing? then this fixture no longer exercises "
            "the partial-struct path it was chosen for")

    def test_b32_5c_no_preparation_effect_of_any_kind_after_executor(self):
        """Marker removed 2026-07-25 by XPASS(strict), which is the mechanism
        working exactly as intended: this was written as fail-before evidence
        while alias materialization still ran in draw_batch after the executor
        returned, and it flipped the moment the migration landed rather than
        waiting for anyone to remember to update it.

        "Inside the executor" now means inside any of its named phases —
        preparation before the draw, cleanup after the render — per the
        architect's two-phase ruling of 2026-07-25. Cleanup genuinely cannot
        run before the draw, and pretending otherwise would have been the third
        false claim of this phase rather than an architecture."""
        adf = self._lazy()
        adf.add_alias("shift", "mult + 1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with _EffectTrace(A.AliasDataFrame) as tr:
                adf.draw_batch({"p": {"expr": "shift", "type": "hist",
                                      "bins": 5}},
                               lazy=True, verbose=False)
        plt.close("all")
        assert not tr.after, tr.after


@needs_dfdraw
class TestB32CatalogResidualPaths:
    """Ruling 2 evidence, both halves."""

    def _lazy(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    @staticmethod
    def _snapshot(adf):
        r = adf._lazy_reader
        return (sorted(map(str, adf.df.columns)),
                sorted(map(str, getattr(r, "loaded_branches", ()) or ())),
                sorted(adf._structs))

    RESIDUALS = {
        "ensure_struct_catalog":
            lambda adf: adf._ensure_struct_catalog(),
        "get_required_branches":
            lambda adf: adf.get_required_branches(
                expr="dedxTPC.dEdxMaxTPC:mult", validate=True),
        "dict_dispatch_columns":
            lambda adf: adf._dict_dispatch_columns(
                set(adf.df.columns), expr="dedxTPC.dEdxMaxTPC:mult"),
    }

    @pytest.mark.parametrize("path", sorted(RESIDUALS))
    def test_b32_6_residual_paths_are_no_ops_after_a_real_call(self, path):
        """Half one: after a real draw_batch the residual catalog re-checks
        change nothing. This is what makes their physical removal in B3.4
        safe — and it holds BECAUSE the executor completed the structs, not
        because the code path is inert."""
        adf = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch(
                {"p1": {"expr": "dedxTPC.dEdxMaxTPC:mult", "type": "scatter"}},
                verbose=False)
            plt.close("all")
            before = self._snapshot(adf)
            self.RESIDUALS[path](adf)
            after = self._snapshot(adf)
        assert before == after, (
            f"residual path {path!r} still performs an effect after the "
            f"executor owned preparation")

    @pytest.mark.parametrize("path", sorted(RESIDUALS))
    def test_b32_7_residual_paths_are_capable_when_state_is_partial(self, path):
        """Half two — the non-vacuity guard. Construct the partial-struct
        state by hand (load ONE member directly, bypassing the executor) and
        the same residual path DOES complete the struct. If this test ever
        starts passing-by-doing-nothing, test_b32_6 has stopped proving
        anything and B3.4's removal argument has silently expired."""
        adf = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf._ensure_struct_catalog()
            adf.ensure_branches(["dedxTPC/dEdxMaxTPC"])
            before = self._snapshot(adf)
            self.RESIDUALS[path](adf)
            after = self._snapshot(adf)
        assert "dEdxTotTPC__dedxTPC" not in before[0], (
            "fixture precondition broken: the sibling column must be ABSENT "
            "before the residual path runs, or this proves nothing")
        assert "dEdxTotTPC__dedxTPC" in after[0], (
            f"residual path {path!r} did NOT complete the partial struct; the "
            f"D-3 completion leg appears to have been removed or guarded — "
            f"re-derive the B3.4 removal argument before trusting "
            f"test_b32_6")
        gained_cols = set(after[0]) - set(before[0])
        gained_reads = set(after[1]) - set(before[1])
        assert gained_cols == {"dEdxTotTPC__dedxTPC", "dEdxMaxIROC__dedxTPC"}, \
            gained_cols
        assert gained_reads == {"dedxTPC/dEdxTotTPC", "dedxTPC/dEdxMaxIROC"}, \
            gained_reads


# ---------------------------------------------------------------------------
# PHASE_13_76_ADF B3.2 part-1 correction — panel [X] (GPT24 / GPT25 / GPT26,
# three independent EXECUTED reproductions; Main-Reviewer synthesis 2026-07-25).
#
# What the panel proved, and what it cost to learn: a preparation-state record
# built from what the executor INTENDED is wrong in three separate ways, and
# each way was found by running the case rather than reading it. The coder's
# own review request had named the eager path as "plausible, not covered by a
# test"; the honest description was "affirmatively false when executed".
#
#   B32P1-1  reads performed by full-structure completion were absent from
#            branches_loaded, because that field was written from the
#            up-front union-load intent and never revisited.
#   B32P1-2  a struct already partial on entry is completed by the INITIAL
#            catalog call, before the union-load line runs — leaving no trace
#            in either branches_loaded or structs_completed.
#   B32P1-3  on an eager frame ensure_struct() is a silent no-op (there is no
#            reader to load from), yet the struct name was appended to
#            structs_completed regardless: a field that PHASE_13_77_ADF is
#            specified to trust could say a struct was safe to use when its
#            columns did not exist.
#
# The corrections: every read/column field is now a MEASURED before/after
# delta taken at each stage boundary; completion is recorded only after
# re-reading the frame and confirming every member is present; and the
# initial catalog call is measured like any other stage.
#
# These three tests are the permanent form of the panel's probes. They
# reconcile the record against EXTERNAL observation — the reader's loaded
# branches and the frame's columns — never against another field the same
# code path set.
# ---------------------------------------------------------------------------


@needs_dfdraw
class TestB32StateReconciliation:
    """GPT25 required correction 4: three states, each reconciled at the
    public entry point against externally observed effects."""

    FIXTURE = "lazy_struct_fixture_clean.root"

    def _lazy(self):
        fixture = os.path.join(os.path.dirname(__file__), self.FIXTURE)
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    @staticmethod
    def _reader_branches(adf):
        rdr = getattr(adf, "_lazy_reader", None)
        return set(map(str, getattr(rdr, "loaded_branches", ()) or ()))

    def test_b32_8_fresh_lazy_partial_struct_reconciles(self):
        """State 1. The union load pulls one member; completion pulls the
        siblings. Every read must appear in the total, and the stage
        attribution must add up to it."""
        adf = self._lazy()
        before = self._reader_branches(adf)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch(
                {"p1": {"expr": "dedxTPC.dEdxMaxTPC:mult", "type": "scatter"}},
                verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        actually = self._reader_branches(adf) - before
        assert set(st.branches_loaded) == actually, (
            f"record {sorted(st.branches_loaded)} != observed "
            f"{sorted(actually)}")
        stages = {
            "catalog": set(st.reads_by_catalog),
            "prescan": set(st.reads_by_prescan),
            "union": set(st.reads_by_union_load),
            "completion": set(st.reads_by_completion),
            "autoload": set(st.reads_by_autoload),
        }
        staged = set().union(*stages.values())
        assert staged == set(st.branches_loaded), (
            f"stage attribution {sorted(staged)} does not add up to the "
            f"total {sorted(st.branches_loaded)} — an effect is unattributed")
        # R2-P2-2 (GPT25, P2): union equality alone hides double-counting.
        names = sorted(stages)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = stages[a] & stages[b]
                assert not overlap, (
                    f"stages {a!r} and {b!r} both claim {sorted(overlap)}; "
                    f"attribution must be pairwise disjoint")
        assert st.reads_by_completion, "completion path not exercised"
        # TODAY'S POLICY, not an eternal invariant: a partial struct on a lazy
        # frame is auto-completed to the full structure (13.75 D-3). The
        # architect has stated (2026-07-25) that working with a SUBSET of
        # branches will become a supported option; when that lands this
        # assertion is the thing to revisit, deliberately findable from here
        # rather than discovered as a test wall.
        assert st.structs_completed == ("dedxTPC",)
        for member in ("dEdxMaxTPC", "dEdxTotTPC", "dEdxMaxIROC"):
            assert f"{member}__dedxTPC" in adf.df.columns

    def test_b32_9_preexisting_lazy_partial_struct_reconciles(self):
        """State 2 (B32P1-2). The struct is ALREADY partial when the executor
        starts, so the initial catalog call completes it before the union
        load runs. Previously that produced an empty record beside a real
        effect."""
        adf = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf._ensure_struct_catalog()
            adf.ensure_branches(["dedxTPC/dEdxMaxTPC"])
            for c in ("dEdxTotTPC__dedxTPC", "dEdxMaxIROC__dedxTPC"):
                if c in adf.df.columns:
                    adf.df.drop(columns=[c], inplace=True)
            before = self._reader_branches(adf)
            cols_before = set(adf.df.columns)
            # the drawn expression deliberately does NOT mention the struct
            adf.draw_batch({"p1": {"expr": "tgl:mult", "type": "scatter"}},
                           verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        actually = self._reader_branches(adf) - before
        assert set(st.branches_loaded) == actually, (
            f"record {sorted(st.branches_loaded)} != observed "
            f"{sorted(actually)}")
        assert st.reads_by_catalog, (
            "the initial catalog call completed the struct but the record "
            "attributes no read to it (B32P1-2)")
        assert st.structs_completed == ("dedxTPC",), st.structs_completed
        assert set(st.columns_created) == set(adf.df.columns) - cols_before

    def test_b32_10_eager_partial_struct_is_never_reported_complete(self):
        """State 3 (B32P1-3), the falsifying case. An eager frame has no
        reader, so ensure_struct() cannot load anything; the record must say
        so. Eager partial structs are TOLERATED, not completed — pre-existing
        behavior, preserved here deliberately and pinned so a future change
        is visible. Whether they should instead be refused loudly is an open
        matrix cell for the architect."""
        adf = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf._ensure_struct_catalog()
            adf.ensure_struct("dedxTPC")
            adf.ensure_branches(["mult"])
            adf._lazy_reader = None                      # eager from here on
            adf.df = adf.df.drop(columns=["dEdxTotTPC__dedxTPC"])
            assert "dEdxMaxTPC__dedxTPC" in adf.df.columns  # genuinely partial
            adf.draw_batch({"p": {"expr": "mult", "type": "hist", "bins": 5}},
                           verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert "dEdxTotTPC__dedxTPC" not in adf.df.columns, (
            "fixture no longer partial — this test proves nothing")
        assert st.structs_completed == (), (
            "a struct was reported COMPLETED on an eager frame while its "
            "column was never created (B32P1-3)")
        assert st.columns_created == ()
        assert st.branches_loaded == ()
        # Architect ruling 2026-07-25: incompleteness is a NEUTRAL FACT, not a
        # fault — a subset of branches will be a supported way to work. So it
        # must be visible in the record (the reviewers' actual objection) while
        # raising no warning and refusing nothing.
        recorded = {n: (set(p), c) for n, p, c in st.struct_members_present}
        assert "dedxTPC" in recorded, recorded
        present, complete = recorded["dedxTPC"]
        assert complete is False
        assert "dEdxTotTPC" not in present


@needs_dfdraw
class TestB32DefaultsDoubleCompletion:
    """B32P1-6 / Disputed-DoubleComplete. GPT25 traced two ensure_struct calls
    and a repeated branch REQUEST; GPT26 argued from source that no duplicate
    branch I/O occurs. Both are right at different layers, and the synthesis
    left it unadjudicated pending one direct trace. This is that trace, kept
    permanently so the answer cannot drift.

    Measured: the second ensure_struct() DOES issue a second ensure_branches()
    request (GPT25's observation stands), but the reader performs exactly two
    real reads for the whole call (GPT26's conclusion stands) — the filtering
    happens inside ensure_branches, not, as argued, inside ensure_struct's
    missing-member check. So this is invocation overhead, not duplicated I/O,
    which is what makes it a B3.4 demolition item rather than part-2 scope.
    """

    def _lazy(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    def test_b32_11_defaults_path_costs_invocations_not_reads(self):
        adf = self._lazy()
        reads, calls = [], []
        real_load = adf._lazy_reader.load_branches
        adf._lazy_reader.load_branches = lambda names, *a, **k: (
            reads.append(sorted(names)), real_load(names, *a, **k))[1]
        cls = A.AliasDataFrame
        real_es = cls.ensure_struct
        cls.ensure_struct = lambda s, n, *a, **k: (
            calls.append(n), real_es(s, n, *a, **k))[1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf.draw_batch({"p1": {"type": "scatter"}},
                               defaults={"expr": "dedxTPC.dEdxMaxTPC:mult"},
                               verbose=False)
        finally:
            cls.ensure_struct = real_es
        plt.close("all")
        assert calls.count("dedxTPC") == 2, (
            f"expected the known double invocation, saw {calls}; if this is "
            f"now 1 the B3.4 demolition item is already closed — update the "
            f"matrix rather than deleting this test")
        assert len(reads) == 2, (
            f"duplicate branch I/O appeared: {reads}. The redundancy was "
            f"invocation-only; a second real read means the cost changed "
            f"and B32P1-6 must be re-scoped out of B3.4")
        # R2-P2-1 (GPT24/GPT25/GPT26, P2): equal call COUNTS do not prove
        # disjoint reads. Two overlapping batches would also give len == 2.
        flat = [b for batch in reads for b in batch]
        assert len(flat) == len(set(flat)), (
            f"a branch was read twice across batches: {reads}")
        assert set(flat) == {"dedxTPC/dEdxMaxTPC", "dedxTPC/dEdxTotTPC",
                             "dedxTPC/dEdxMaxIROC", "mult"}, sorted(flat)


@needs_dfdraw
class TestB32PhysicalFormCatalogCompletion:
    """Round-2 finding F1 (P0) — GPT24 and GPT26, two independent executed
    reproductions, reproduced a third time by the coder before acceptance.

    The state 'a struct partial on entry' has TWO representations, and the
    first correction only handled one. A struct can arrive as internal member
    columns (`dEdxMaxTPC__dedxTPC`) — covered by test_b32_9 — or in PHYSICAL
    form (`dedxTPC/dEdxMaxTPC`), not yet registered at all, which is the D4
    preloaded-column shape the executor's own comment cites. In the physical
    case the initial catalog call registers the struct, renames the column and
    loads the missing siblings, all in one stage. The membership snapshot taken
    before that call could not see a struct that did not yet exist, so the
    reads were recorded while the completion that caused them was not.

    Fixed by measuring the ENTRY column set with the definitions known AFTER
    registration — see _struct_membership_in. The control test below matters as
    much as the finding: measuring it the naive way would report a completion
    for a struct that merely got registered and renamed while already whole.
    """

    def _lazy_with_physical_member(self, member="dedxTPC/dEdxMaxTPC"):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        adf = A.AliasDataFrame.read_tree_lazy(fixture, "tree")
        # Undo auto-registration so the struct is genuinely unknown on entry,
        # then land ONE member under its physical name (bypassing the A-1
        # rename that ensure_branches would apply).
        adf._structs = {}
        adf._struct_catalog_fp = None
        adf.ensure_branches(["mult"])
        adf.df = adf._merge_loaded_data(
            adf.df, adf._lazy_reader.load_branches([member]))
        assert member in adf.df.columns
        return adf

    def test_b32_12_physical_form_completion_is_recorded(self):
        adf = self._lazy_with_physical_member()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "mult", "type": "hist", "bins": 5}},
                           verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert st.reads_by_catalog, (
            "the catalog stage read nothing — fixture no longer exercises the "
            "physical-form completion path")
        assert "dedxTPC" in st.structs_completed, (
            f"completion recorded reads {st.reads_by_catalog} but no struct "
            f"(F1); structs_completed={st.structs_completed}")
        for m in ("dEdxMaxTPC", "dEdxTotTPC", "dEdxMaxIROC"):
            assert f"{m}__dedxTPC" in adf.df.columns
        assert set(st.branches_loaded) >= set(st.reads_by_catalog)

    def test_b32_13_already_whole_struct_is_not_reported_completed(self):
        """Control for the fix, not for the defect. 'left' has a single
        member, so landing it physically makes the struct WHOLE before the
        catalog call; registering and renaming it is not a completion. A
        naive before/after membership diff would report it as one."""
        adf = self._lazy_with_physical_member("left/value")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "tgl", "type": "hist", "bins": 5}},
                           verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert "left" not in st.structs_completed, (
            f"a struct that was already whole was reported as completed: "
            f"{st.structs_completed}")
        assert "value__left" in adf.df.columns   # it was still normalized


@needs_dfdraw
class TestB32EagerCompletionNotInvoked:
    """Round-2 finding F2 (P1) — GPT27 caught it; GPT24 and GPT26 confirmed.

    The first correction called _complete_partial_structs() unconditionally,
    justified as preserving behaviour on eager frames. That justification was
    false: _ensure_struct_catalog() returns at its second statement when
    _lazy_reader is None, so the D-3 completion leg had never run on an eager
    frame. The unconditional call was a NEW eager invocation described as a
    preservation. The call is now gated to lazy frames, which is both the
    honest shape and the useful one — an eager frame has no reader to complete
    a struct from.
    """

    def test_b32_14_eager_frame_never_enters_completion(self):
        e = A.AliasDataFrame(pd.DataFrame({
            "mult": np.arange(100.0),
            "dEdxMaxTPC__dedxTPC": np.arange(100.0)}))
        e._restore_schema({"structs": {"dedxTPC": {
            "members": ["dEdxMaxTPC", "dEdxTotTPC", "dEdxMaxIROC"]}}})
        assert "dedxTPC" in e._structs, "schema restore did not register"
        cls = A.AliasDataFrame
        real = cls._complete_partial_structs
        calls = []
        cls._complete_partial_structs = lambda s: (calls.append(1), real(s))[1]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e.draw_batch({"p": {"expr": "mult", "type": "hist",
                                    "bins": 5}}, verbose=False)
        finally:
            cls._complete_partial_structs = real
        plt.close("all")
        assert calls == [], (
            "the executor invoked struct completion on an eager frame; that "
            "path never ran before this phase (F2)")

    def test_b32_15_eager_partial_struct_usable_and_absent_member_loud(self):
        """The architect ruling, pinned. A subset of branches will be a
        supported way to work, so a present member must stay usable; a genuinely
        absent member must still fail loudly through the 13.75 C3 guard, so
        nobody computes on data that is not there."""
        e = A.AliasDataFrame(pd.DataFrame({
            "mult": np.arange(100.0),
            "dEdxMaxTPC__dedxTPC": np.linspace(40.0, 60.0, 100)}))
        e._restore_schema({"structs": {"dedxTPC": {
            "members": ["dEdxMaxTPC", "dEdxTotTPC", "dEdxMaxIROC"]}}})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ok = e.draw_batch(
                {"p": {"expr": "dedxTPC.dEdxMaxTPC", "type": "hist",
                       "bins": 10}}, verbose=False)
        plt.close("all")
        assert ok["_summary"]["failed"] == 0, ok["_errors"]
        assert ok["p"]["stats"]["n"] == 100
        with pytest.raises(ValueError, match="projection inconsistency"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                e.draw_batch({"q": {"expr": "dedxTPC.dEdxTotTPC",
                                    "type": "hist", "bins": 10}},
                             verbose=False)
        plt.close("all")


def _write_tree_with_subframe(path, seed=7, n=200):
    """Main tree + a small calibration tree in one file, for lazy-subframe
    tests. mktree, not dict assignment — the latter writes an RNTuple."""
    rng = np.random.default_rng(seed)
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": np.float64, "y": np.float64, "sec": np.int32})
        f["tree"].extend({"x": rng.normal(0, 1, n), "y": rng.normal(0, 1, n),
                          "sec": rng.integers(0, 4, n).astype(np.int32)})
        f.mktree("SectorCalib", {"sec": np.int32, "corr": np.float64,
                                 "pad": np.float64})
        f["SectorCalib"].extend({"sec": np.arange(4, dtype=np.int32),
                                 "corr": np.array([1., 2., 3., 4.]),
                                 "pad": np.array([9., 9., 9., 9.])})
    return path


@needs_dfdraw
class TestB32PrescanAttribution:
    """Round-2 finding F3 (P1) — GPT26, executed.

    The subframe pre-scan loads branches of its own (the index columns a lazy
    subframe needs to join). Those reads used to fall inside the union-load
    observation window and were reported as union-load reads, while
    requested_reads — correctly — never mentioned them. Total accounting was
    unaffected, which is exactly why it went unnoticed: only the attribution
    was wrong, and no test looked at attribution on a frame where the pre-scan
    actually did anything.

    The fixture matters more than the assertion here. The first version of the
    stage-disjointness check passed with the boundary deliberately broken,
    because the struct fixture's pre-scan loads nothing. A test that cannot
    fail is not evidence.
    """

    def test_b32_16_prescan_reads_are_attributed_to_the_prescan(self, tmp_path):
        p = _write_tree_with_subframe(str(tmp_path / "sf.root"))
        adf = A.AliasDataFrame.read_tree_lazy(p, "tree")
        adf.register_subframe_lazy("SectorCalib", p,
                                   tree_name="SectorCalib",
                                   index_columns=["sec"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "SectorCalib.corr:x",
                                  "type": "scatter"}}, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert st.reads_by_prescan, (
            "the pre-scan loaded nothing — fixture no longer exercises F3")
        assert "sec" in st.reads_by_prescan, st.reads_by_prescan
        assert "sec" not in st.reads_by_union_load, (
            f"a pre-scan read is attributed to the union load (F3): "
            f"union={st.reads_by_union_load}")
        assert "sec" not in st.requested_reads, (
            "requested_reads is plan INTENT; the pre-scan's own reads are "
            "not part of it")
        assert set(st.branches_loaded) == set(st.reads_by_prescan) | set(
            st.reads_by_union_load) | set(st.reads_by_catalog) | set(
            st.reads_by_completion) | set(st.reads_by_autoload)


@needs_dfdraw
class TestB32CompletionVsPlainLoad:
    """Control for the F1 fix: a struct loaded from NOTHING during the call is
    a plain load, not a completion, and must not be reported as one.

    This is the branch of _structs_completed_between guarded by
    `not _present_before`. The first control test (b32_13) did not reach it —
    it exercised the already-whole path instead — and the guard survived a
    mutation undetected. Recording every fresh load as a "completion" would
    make the field meaningless precisely when PHASE_13_77_ADF starts reading it.
    """

    def _lazy(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    def test_b32_17_struct_loaded_from_nothing_is_not_a_completion(self):
        adf = self._lazy()
        assert not [c for c in adf.df.columns if "__mTOFLength" in str(c)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "mTOFLength.len:mult",
                                  "type": "scatter"}}, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert "len__mTOFLength" in adf.df.columns, "struct was not loaded"
        assert "mTOFLength" not in st.structs_completed, (
            f"a struct loaded from nothing was reported as COMPLETED: "
            f"{st.structs_completed}")


@needs_dfdraw
class TestB32ReaderGraphObservation:
    """Round-3 finding P0-ReaderGraph — GPT31, executed; the only seat across
    three rounds that built a subframe scenario rather than confirming the fix
    against the scenarios it was designed for.

    _observe_prep_effects() used to look at `self._lazy_reader` and
    `self.df.columns` and nothing else. A draw slot referencing a lazy subframe
    makes the executor's pre-scan materialize that subframe, which reads
    branches through the SUBFRAME'S OWN reader and builds the subframe's own
    frame. Those effects were invisible to the record by construction — so the
    state was complete for the main reader and silently blind to the rest of
    the graph, while calling itself the auditable answer to "which effects
    ran". Same class of error as the two rounds before it: the claim was wider
    than what the code observed.

    The record now walks the whole graph and qualifies names by owner
    (`SectorCalib::corr`), so a branch of the same name in two readers cannot
    collapse into one entry and under-report. Main-frame names stay
    unqualified, so every earlier reconciliation test keeps its meaning.
    """

    def _adf_with_lazy_subframe(self, tmp_path):
        p = _write_tree_with_subframe(str(tmp_path / "graph.root"))
        adf = A.AliasDataFrame.read_tree_lazy(p, "tree")
        adf.register_subframe_lazy("SectorCalib", p,
                                   tree_name="SectorCalib",
                                   index_columns=["sec"])
        return adf

    def test_b32_18_subframe_reader_effects_are_recorded(self, tmp_path):
        adf = self._adf_with_lazy_subframe(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "SectorCalib.corr:x",
                                  "type": "scatter"}}, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        # the subframe's own reader did real work; the record must say so
        assert any(b.startswith("SectorCalib::") for b in st.branches_loaded), (
            f"subframe reader effects are invisible to the record "
            f"(P0-ReaderGraph): branches_loaded={st.branches_loaded}")
        assert "SectorCalib::corr" in st.branches_loaded, st.branches_loaded
        assert any(c.startswith("SectorCalib::") for c in st.columns_created), (
            f"subframe frame columns are invisible: "
            f"columns_created={st.columns_created}")
        # main-frame names remain unqualified — earlier reconciliations hold
        assert "x" in st.branches_loaded
        assert not any(b.startswith("::") for b in st.branches_loaded)

    def test_b32_19_graph_reads_reconcile_and_stay_attributed(self, tmp_path):
        """The stage attribution must survive the widened observation: every
        graph read belongs to exactly one stage, and the stages still sum to
        the measured total."""
        adf = self._adf_with_lazy_subframe(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "SectorCalib.corr:x",
                                  "type": "scatter"}}, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        stages = {
            "catalog": set(st.reads_by_catalog),
            "prescan": set(st.reads_by_prescan),
            "union": set(st.reads_by_union_load),
            "completion": set(st.reads_by_completion),
            "autoload": set(st.reads_by_autoload),
        }
        staged = set().union(*stages.values())
        assert staged == set(st.branches_loaded), (
            f"graph reads {sorted(set(st.branches_loaded) - staged)} are "
            f"unattributed")
        names = sorted(stages)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                assert not (stages[a] & stages[b]), (
                    f"{a} and {b} both claim {sorted(stages[a] & stages[b])}")
        # the subframe reads belong to the pre-scan, which is what triggers them
        assert "SectorCalib::corr" in stages["prescan"], stages["prescan"]


@needs_dfdraw
class TestB32ChainSyntheticColumn:
    """Round-4 finding P0-ChainSyntheticRead — GPT30, executed; confirmed by
    the coder before acceptance.

    `LazyChainReader` adds a synthetic `__file_idx__` bookkeeping column to
    `loaded_branches` while deliberately excluding it from
    `available_branches` — its own docstring says so at two places. The graph
    walk copied loaded names unfiltered, so a name that was never read from a
    file entered `branches_loaded` and `reads_by_union_load`.

    This one blocked where the round's other finding did not, and the
    distinction is the standing bar for this record: `__file_idx__` in a read
    field is a FALSEHOOD — it tells a consumer that I/O happened which did not.
    A coverage gap merely omits; a falsehood misinforms. Reads are now filtered
    through each reader's own `available_branches`; the column still appears in
    `columns_created`, which is accurate, because it is a real column.
    """

    CHAIN = ("chain_part1.root", "chain_part2.root")

    def _chain(self, add_file_index=True):
        paths = [os.path.join(os.path.dirname(__file__), f) for f in self.CHAIN]
        for p in paths:
            if not os.path.exists(p):
                pytest.skip("chain fixtures not present (make_fixtures.py)")
        return A.AliasDataFrame.read_chain_lazy(
            paths, "tree", add_file_index=add_file_index)

    def test_b32_20_synthetic_column_never_enters_the_read_record(self):
        adf = self._chain(add_file_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "mult", "type": "hist", "bins": 5}},
                           verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        read_fields = (st.branches_loaded, st.requested_reads,
                       st.reads_by_catalog, st.reads_by_prescan,
                       st.reads_by_union_load, st.reads_by_completion,
                       st.reads_by_autoload)
        for field in read_fields:
            assert not any("__file_idx__" in str(b) for b in field), (
                f"a synthetic bookkeeping column is reported as a physical "
                f"read (P0-ChainSyntheticRead): {field}")
        # it IS a real column, so this half must stay true
        assert any("__file_idx__" in str(c) for c in st.columns_created), (
            "the synthetic column vanished from columns_created — the filter "
            "was applied to the wrong half of the record")
        assert "mult" in st.branches_loaded, "the real read was filtered away"

    def test_b32_21_real_chain_reads_survive_the_filter(self):
        """Guard against over-filtering: a chain WITHOUT the file index must
        record its ordinary branch reads unchanged."""
        adf = self._chain(add_file_index=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "mult", "type": "hist", "bins": 5}},
                           verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert "mult" in st.branches_loaded, st.branches_loaded
        assert not any("__file_idx__" in str(c) for c in st.columns_created)


@needs_dfdraw
class TestB32GraphScopeRulings:
    """The two round-4 omissions, closed by architect ruling — in opposite
    directions, which is worth reading before the tests below.

    Both were disclosed rather than fixed because each turned on a scope
    question, and settling a scope question inside a correction pass is the
    mistake this phase already paid for once with the eager path.

    Structs inside child frames: SUPPORTED (ruling 2026-07-25, "full
    functionality within the child tables").

    One child object under two subframe names: FORBIDDEN (ruling 2026-07-27,
    reversing the 2026-07-25 ruling that had allowed it). It was implemented
    as allowed, and the correction round showed what it cost — see
    test_b32_23. The use case behind the original request is served by two
    instances over the same source, which stays legal and is tested here too.
    """

    def _lazy(self, name="lazy_struct_fixture_clean.root"):
        fixture = os.path.join(os.path.dirname(__file__), name)
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    def test_b32_22_struct_inside_a_subframe_is_completed_d2(self):
        """D2 ruling (architect, 2026-07-25): "we should support full
        functionality within the child tables".

        Round 4 disclosed the opposite as a limit — completion was self-scoped
        while observation was graph-scoped, so a struct living inside a
        subframe stayed partial. The ruling closes it: completion walks the
        graph and is gated per node on that node's OWN lazy reader.

        The assertion is on the child frame's columns, not on the record. A
        record that says a struct was completed is worth exactly as much as
        the completion actually having happened, and part 1's P0 was precisely
        the case where those two came apart."""
        main = self._lazy()
        child = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            child._ensure_struct_catalog()
            child.ensure_branches(["dedxTPC/dEdxMaxTPC", "mult"])
            for c in ("dEdxTotTPC__dedxTPC", "dEdxMaxIROC__dedxTPC"):
                if c in child.df.columns:
                    child.df.drop(columns=[c], inplace=True)
            main.ensure_branches(["mult"])
            main.register_subframe("Child", child, index_columns=["mult"])
            before = set(map(str, child.df.columns))
            main.draw_batch({"p": {"expr": "mult", "type": "hist", "bins": 5}},
                            verbose=False)
        plt.close("all")
        after = set(map(str, child.df.columns))
        assert "dEdxTotTPC__dedxTPC" in after, (
            "the child struct was left partial — D2 says child tables get the "
            "full functionality")
        assert "dEdxMaxIROC__dedxTPC" in after, after - before
        st = main._last_draw_prep_state
        assert "Child::dedxTPC" in st.structs_completed, (
            "the completion happened but is recorded under the wrong scope; "
            "a child-frame effect must carry its owner path")
        assert "dedxTPC" not in st.structs_completed, (
            "the ROOT frame's struct was already whole — reporting it as "
            "completed would be the part-1 P0 all over again")

    def test_b32_22b_child_struct_membership_is_reported_under_its_owner(self):
        """The record has to describe the same scope it acts on. If completion
        reaches child frames but struct_members_present only ever describes
        this frame, the record answers one question about two scopes."""
        main = self._lazy()
        child = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            child._ensure_struct_catalog()
            child.ensure_branches(["dedxTPC/dEdxMaxTPC", "mult"])
            main.ensure_branches(["mult"])
            main.register_subframe("Child", child, index_columns=["mult"])
            main.draw_batch({"p": {"expr": "mult", "type": "hist", "bins": 5}},
                            verbose=False)
        plt.close("all")
        st = main._last_draw_prep_state
        names = {n for n, _p, _c in st.struct_members_present}
        assert "Child::dedxTPC" in names, sorted(names)
        for _n, _p, _c in st.struct_members_present:
            if _n == "Child::dedxTPC":
                assert _c is True, (
                    "reported incomplete after completion ran — the two "
                    "sides of the record disagree")

    def test_b32_23_same_object_under_two_names_is_refused_d2(self):
        """D2 ruling (architect, 2026-07-27) — this REVERSES the D1 ruling of
        2026-07-25, and the reversal is the point.

        D1 held the aliased registration legal and it was implemented. The
        correction round showed the bill: with one mutable object under two
        names, GPT27 found `structs_completed` naming one owner while
        `struct_members_present` named both — two record fields describing the
        same physical event and disagreeing, with no non-arbitrary answer to
        "how many completions happened". Aliases, materialization, caches and
        cleanup are shared across both paths for the same reason.

        The architect's actual use case — the same source data as two
        independent logical tables, e.g. nominal and varied parameterized
        aliases — is served by two INSTANCES, which is the next test."""
        main = self._lazy()
        child = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main.ensure_branches(["mult"])
            child.ensure_branches(["mult"])
            main.register_subframe("A", child, index_columns=["mult"])
            with pytest.raises(ValueError, match="already registered"):
                main.register_subframe("B", child, index_columns=["mult"])
        # the first registration must survive the refusal intact
        assert "A" in main._subframes.subframes
        assert "B" not in main._subframes.subframes

    def test_b32_23a_two_instances_over_one_source_stay_independent(self):
        """The other half of D2, and the half that matters to the user: the
        ruling forbids one OBJECT under two names, not one SOURCE behind two
        frames. Two instances over the same table, carrying different alias
        definitions, must register and must not leak into each other."""
        main = self._lazy()
        nominal = self._lazy()
        varied = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for f in (main, nominal, varied):
                f.ensure_branches(["mult"])
            nominal.add_alias("scaled", "mult * 2")
            varied.add_alias("scaled", "mult * 10")
            main.register_subframe("Nominal", nominal, index_columns=["mult"])
            main.register_subframe("Varied", varied, index_columns=["mult"])
            main.draw_batch({"p": {"expr": "mult", "type": "hist", "bins": 5}},
                            verbose=False)
        plt.close("all")
        assert nominal.aliases["scaled"] == "mult * 2"
        assert varied.aliases["scaled"] == "mult * 10"
        st = main._last_draw_prep_state
        assert "Nominal::scaled" in st.frame_aliases, st.frame_aliases
        assert "Varied::scaled" in st.frame_aliases
        nodes = dict(main._iter_frame_graph()[0])
        assert nodes["Nominal::"] is not nodes["Varied::"], (
            "two independent contexts collapsed onto one object")

    def test_b32_23b_a_real_cycle_still_terminates(self):
        """The ancestor-path guard buys D1 without buying infinite recursion.
        A frame reachable from itself must stop, or the walk that every phase
        of the executor depends on hangs the draw."""
        a = self._lazy()
        b = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a.ensure_branches(["mult"])
            b.ensure_branches(["mult"])
            a.register_subframe("B", b, index_columns=["mult"])
            b.register_subframe("A", a, index_columns=["mult"])
            nodes, aliases = a._iter_frame_graph()
        prefixes = [p for p, _ in nodes]
        assert "" in prefixes and "B::" in prefixes
        assert len(nodes) < 10, prefixes
        assert "B::A::" not in prefixes, (
            "the walk re-entered its own ancestor — that is a cycle, not a "
            "second owner path")


@needs_dfdraw
class TestB32PlanPurityByConstruction:
    """P1-PlanPurity, open since round 1 and flagged by four GPT seats across
    four rounds. `_DrawDependencyPlan` documented itself as PURE while carrying
    `required_branches(adf)`, which reached through `get_required_branches`
    into the struct catalog — an effect.

    The reviewers proposed a test that calls the effectful method and asserts
    no effect. That would have been a weaker fix: it leaves the effectful
    method reachable and relies on someone remembering the assertion. The
    method is gone instead, and resolution moved to the executor, so purity
    holds by construction. This test pins the *structural* property, which is
    the thing that cannot silently rot."""

    def test_b32_24_plan_exposes_no_effectful_method(self):
        plan_methods = {
            n for n in dir(_DrawDependencyPlan)
            if not n.startswith("__") and callable(getattr(_DrawDependencyPlan, n))
        }
        assert "required_branches" not in plan_methods, (
            "the plan has regained an effectful resolution method; resolution "
            "belongs to the executor (_resolve_required_branches)")
        # what remains must be describable without an ADF to act on
        assert plan_methods <= {"prescan_text"}, sorted(plan_methods)

    def test_b32_25_building_and_describing_a_plan_has_no_effects(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        adf = A.AliasDataFrame.read_tree_lazy(fixture, "tree")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            before = adf._observe_prep_effects()
            structs_before = dict(adf._structs)
            e = _EffectiveDrawSpec.from_call("dedxTPC.dEdxMaxTPC:mult",
                                             "profile", {"bins": 5})
            plan = _DrawDependencyPlan(especs=[e], rewrite_dicts=[{}],
                                       autoload_dicts=[],
                                       merged_specs=[{"expr": "mult"}])
            assert plan.prescan_text()          # describable
            after = adf._observe_prep_effects()
        assert after == before, (
            "building or describing the plan changed observable state")
        assert dict(adf._structs) == structs_before


@needs_dfdraw
class TestB32TwoPhaseExecutor:
    """PHASE_13_76_ADF B3.2 part 2 — the executor's named phases.

    Architect ruling 2026-07-25: two of the effects Rev 2 §11.4 assigns to the
    executor cannot live in a pre-draw phase. The single-level subframe join
    writes into the REDUCED frame, which does not exist until projection, and
    moving it earlier would mean growing self.df — which the D-ADF-DICT
    contract forbids in as many words. Cleanup runs after dfdraw has rendered;
    there is no "before the draw" that contains it.

    Rather than let physics make the sole-ownership claim false, the executor
    became multi-phase: preparation, projection, cleanup — three named phases
    of one owner, all reporting into one record. The alternative on offer was
    to narrow the claim to "three separate owners", which is what the phase
    was created to get rid of.
    """

    def _lazy(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    def test_b32_26_cleanup_is_a_phase_and_reports_what_it_dropped(self):
        adf = self._lazy()
        adf.add_alias("shift", "mult + 1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "shift", "type": "hist", "bins": 5}},
                           lazy=True, clear_after=True, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert "shift" in st.aliases_materialized, st.aliases_materialized
        assert st.cleanup_candidates == st.aliases_materialized, (
            "candidates must come from the executor's own measurement, not "
            "from a snapshot the calling surface kept — otherwise the bracket "
            "can drift from what phase one actually did")
        assert "shift" in st.aliases_dropped, st.aliases_dropped
        assert "shift" not in adf._get_materialized_aliases()

    def test_b32_27_clear_after_false_records_candidates_but_drops_nothing(self):
        adf = self._lazy()
        adf.add_alias("shift", "mult + 1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "shift", "type": "hist", "bins": 5}},
                           lazy=True, clear_after=False, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert "shift" in st.cleanup_candidates
        assert st.aliases_dropped == ()
        assert "shift" in adf._get_materialized_aliases()

    def test_b32_28_projection_phase_records_reduced_frame_temp_columns(
            self, tmp_path):
        """GPT24's R2-P1-2, closed. A tracer that wraps methods can never see
        `df_for_plot[flat_ref] = ...` because the write has no method to wrap.
        The boundary is therefore established by MEASURING the frame."""
        p = _write_tree_with_subframe(str(tmp_path / "proj.root"))
        adf = A.AliasDataFrame.read_tree_lazy(p, "tree")
        adf.register_subframe_lazy("SectorCalib", p,
                                   tree_name="SectorCalib",
                                   index_columns=["sec"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "SectorCalib.corr:x",
                                  "type": "scatter"}}, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert st.temporary_columns, (
            "the subframe join wrote a flattened column into the reduced "
            "frame and the projection phase did not record it")
        assert any("corr" in c for c in st.temporary_columns), \
            st.temporary_columns
        # temporary means temporary: it must NOT be reported as persistent
        for c in st.temporary_columns:
            assert c not in st.columns_created, (
                f"{c!r} is a reduced-frame temporary but is also reported as "
                f"a persistent column")
            assert c not in adf.df.columns, (
                f"{c!r} leaked onto the big frame — the D-ADF-DICT contract "
                f"says it is never grown")
        assert set(st.temporary_columns) <= set(st.projection_columns)

    def test_b32_29_cache_effects_record_a_measured_transition(self):
        """Rev 2 §11.5's last field group. Recorded as an observed transition,
        not as 'a cache was touched'."""
        adf = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf._structs = {}
            adf._struct_catalog_fp = None          # cache genuinely unset
            adf.draw_batch({"p": {"expr": "mult", "type": "hist", "bins": 5}},
                           verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert any(k == "struct_catalog_fingerprint" for k, _ in
                   st.cache_effects), st.cache_effects
        # a second call over a stable catalog must report NO cache effect
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "tgl", "type": "hist", "bins": 5}},
                           verbose=False)
        plt.close("all")
        assert adf._last_draw_prep_state.cache_effects == (), (
            "a stable catalog reported a cache effect — the field would then "
            "say 'something happened' on every call and mean nothing")


@needs_dfdraw
class TestB32ProjectionOwnership:
    """B3.2 part 2 correction, panel [X] — P0-ProjectionOwnership.

    Five GPT seats independently found the same thing: the first version of
    the projection phase did not OWN anything. The subframe-resolution block
    stayed inline in draw_batch and `_execute_draw_projection_effects` was
    called afterwards to diff two column lists. That is a good oracle, and an
    oracle is not an owner — the increment exists to make "one owner" a
    property of the code rather than a sentence in a docstring, and a method
    that only measures leaves the claim exactly as false as it was.

    These tests are written so they FAIL if the work moves back out. The
    measurement tests in TestB32TwoPhaseExecutor cannot do that: they would
    pass just as happily against the inline arrangement, which is how the
    arrangement survived a round of review.
    """

    def _sf(self, tmp_path, name="own.root"):
        p = _write_tree_with_subframe(str(tmp_path / name))
        adf = A.AliasDataFrame.read_tree_lazy(p, "tree")
        adf.register_subframe_lazy("SectorCalib", p, tree_name="SectorCalib",
                                   index_columns=["sec"])
        return adf

    def test_b32_30_neutralising_the_phase_neutralises_the_join(self, tmp_path):
        """The ownership test proper, by mutation. Replace the projection
        phase with a pass-through that does nothing and returns the frame
        unchanged: if the phase truly owns the join, the flattened column can
        no longer appear. If the work is still inline in draw_batch, the
        column appears anyway and this test fails — which is precisely the
        state the panel rejected."""
        adf = self._sf(tmp_path)
        cls = type(adf)
        real = cls._execute_draw_projection_effects
        cls._execute_draw_projection_effects = (
            lambda self, state, df_for_plot, specs, defaults, kwargs, **kw:
            (df_for_plot, {}))
        spec = {"p": {"expr": "SectorCalib.corr:x", "type": "scatter"}}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # GPT30 (correction round): asserting "some exception" is too
                # weak — a typo in the fixture would satisfy it. Assert the
                # SPECIFIC consequence of the join not happening: dfdraw is
                # handed a reference it cannot resolve.
                with pytest.raises(Exception, match="SectorCalib"):
                    adf.draw_batch(spec, verbose=False)
        finally:
            cls._execute_draw_projection_effects = real
            plt.close("all")
        assert "SectorCalib_corr" not in adf.df.columns
        assert spec["p"]["expr"] == "SectorCalib.corr:x", (
            "the spec was rewritten even though the phase did nothing — the "
            "rewrite has escaped the phase again")

    def test_b32_31_the_phase_returns_the_rewrite_map_it_produced(self,
                                                                 tmp_path):
        """The second half of ownership: the phase does not merely perform the
        join, it produces the dot->flat replacement map. Returning it is what
        makes draw_batch a caller rather than a co-owner holding half the
        state."""
        adf = self._sf(tmp_path)
        seen = {}
        cls = type(adf)
        real = cls._execute_draw_projection_effects

        def _spy(self, state, df_for_plot, specs, defaults, kwargs, **kw):
            out = real(self, state, df_for_plot, specs, defaults, kwargs, **kw)
            seen["frame"], seen["repl"] = out
            return out

        cls._execute_draw_projection_effects = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf.draw_batch({"p": {"expr": "SectorCalib.corr:x",
                                      "type": "scatter"}}, verbose=False)
        finally:
            cls._execute_draw_projection_effects = real
            plt.close("all")
        assert seen["repl"], "the phase produced no replacement map"
        assert any("SectorCalib.corr" in k for k in seen["repl"]), seen["repl"]
        assert any("corr" in str(c) for c in seen["frame"].columns), \
            list(seen["frame"].columns)

    def _eager_pair(self):
        """Eager main + eager child. Deliberately NOT the lazy fixture: a
        lazy subframe referenced only through an ALIAS never gets its join
        index columns pre-scanned, so the join fails before the projection
        phase can be reached. That is a pre-existing gap in the lazy pre-scan,
        disclosed rather than fixed here — repairing it inside a correction
        pass would mix two changes and this suite would then not be able to
        say which one the evidence covers."""
        rng = np.random.default_rng(3)
        n = 120
        main = A.AliasDataFrame(pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "sec": rng.integers(0, 4, n).astype(np.int32)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(4, dtype=np.int32),
            "corr": np.array([1., 2., 3., 4.])}))
        main.register_subframe("SectorCalib", child, index_columns=["sec"])
        return main, child

    def test_b32_32_child_alias_materialised_in_projection_is_cleaned_up(
            self):
        """P0-SubframeCleanupRegression. The projection phase materializes an
        alias on the CHILD frame to satisfy `Sub.alias`. Cleanup used to call
        self.dematerialize(), which can only reach THIS frame, so the alias
        was listed as a candidate and then quietly survived the call.

        A record that lists a candidate which is never dropped is not an
        incomplete record — it is a false one, and this record is specified to
        be trusted by PHASE_13_77_ADF."""
        adf, sf = self._eager_pair()
        sf.add_alias("corr2", "corr * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "SectorCalib.corr2:x",
                                  "type": "scatter"}},
                           clear_after=True, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert "SectorCalib::corr2" in st.aliases_materialized, \
            st.aliases_materialized
        assert "SectorCalib::corr2" in st.aliases_by_projection, (
            "materialized during projection but attributed elsewhere")
        assert "SectorCalib::corr2" in st.aliases_dropped, (
            "listed as a cleanup candidate and never dropped — the record "
            "would then state something untrue")
        assert "corr2" not in sf.df.columns, (
            "the child frame kept the column; cleanup reported a drop that "
            "did not happen")

    def test_b32_33_candidates_not_dropped_show_as_a_gap_not_a_claim(
            self):
        """The honest-failure shape. aliases_dropped is measured AFTER the
        drop attempt, so a candidate that cannot be dropped appears as the
        difference between the two fields rather than as a claim that it was
        dropped. Verified by making the drop impossible."""
        adf, sf = self._eager_pair()
        sf.add_alias("corr2", "corr * 2")
        cls = type(adf)
        real = cls._dematerialize_qualified
        cls._dematerialize_qualified = lambda self, names: ()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf.draw_batch({"p": {"expr": "SectorCalib.corr2:x",
                                      "type": "scatter"}},
                               clear_after=True, verbose=False)
        finally:
            cls._dematerialize_qualified = real
            plt.close("all")
        st = adf._last_draw_prep_state
        assert "SectorCalib::corr2" in st.cleanup_candidates
        assert "SectorCalib::corr2" not in st.aliases_dropped, (
            "nothing was dropped, so nothing may be reported as dropped")
        assert st.cleanup_outcome == "completed"


@needs_dfdraw
class TestB32ProjectionClassification:
    """P0-NestedPersistentMislabeled. Classification is by WHERE THE WRITE
    LANDED, not by which phase observed it.

    A multi-level reference (`A.B.col`) goes through _prepare_subframe_joins,
    which writes a PERSISTENT column onto self.df. The first version measured
    the reduced frame alone, saw the column appear there, and filed it under
    temporary_columns — whose documented meaning is "discarded when the call
    returns". A consumer trusting that would leak the big frame's growth.
    """

    def _nested(self, tmp_path):
        """main -> Mid -> Leaf, so `Mid.Leaf.val` is a multi-level ref."""
        p = str(tmp_path / "nested.root")
        rng = np.random.default_rng(3)
        n = 120
        with uproot.recreate(p) as f:
            f.mktree("tree", {"x": np.float64, "sec": np.int32})
            f["tree"].extend({"x": rng.normal(0, 1, n),
                              "sec": rng.integers(0, 4, n).astype(np.int32)})
        main = A.AliasDataFrame(pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "sec": rng.integers(0, 4, n).astype(np.int32)}))
        mid = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(4, dtype=np.int32),
            "grp": np.arange(4, dtype=np.int32)}))
        leaf = A.AliasDataFrame(pd.DataFrame({
            "grp": np.arange(4, dtype=np.int32),
            "val": np.array([10., 20., 30., 40.])}))
        mid.register_subframe("Leaf", leaf, index_columns=["grp"])
        main.register_subframe("Mid", mid, index_columns=["sec"])
        return main

    def test_b32_34_multilevel_join_column_is_persistent_not_temporary(
            self, tmp_path):
        main = self._nested(tmp_path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main.draw_batch({"p": {"expr": "Mid.Leaf.val:x",
                                   "type": "scatter"}}, verbose=False)
        plt.close("all")
        st = main._last_draw_prep_state
        flat = [c for c in main.df.columns if str(c).startswith("val__")]
        if not flat:
            pytest.skip("multi-level join produced no persistent column here")
        for c in flat:
            assert c not in st.temporary_columns, (
                f"{c!r} was written onto self.df and survives the call, but "
                f"is filed as a reduced-frame temporary")
            assert c in st.columns_created, (
                f"{c!r} is a persistent column and must be recorded as one")

    def test_b32_35_temporary_really_means_temporary(self, tmp_path):
        """The control. Everything reported temporary must be absent from the
        big frame afterwards — otherwise the classification is merely a
        different label for the same confusion."""
        p = _write_tree_with_subframe(str(tmp_path / "tmp.root"))
        adf = A.AliasDataFrame.read_tree_lazy(p, "tree")
        adf.register_subframe_lazy("SectorCalib", p, tree_name="SectorCalib",
                                   index_columns=["sec"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "SectorCalib.corr:x",
                                  "type": "scatter"}}, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert st.temporary_columns
        for c in st.temporary_columns:
            assert c not in adf.df.columns, c
            assert c not in st.columns_created, c


@needs_dfdraw
class TestB32CleanupOnRenderFailure:
    """D3 ruling (architect, 2026-07-25): what cleanup does when RENDERING
    raises is the caller's choice, not a fixed behaviour.

    The default preserves the pre-B3.2 behaviour exactly — a raised render
    leaves the materialized columns in place, which is what someone debugging
    a failed plot wants, since the columns are the evidence. The option buys
    the opposite trade for a long batch in a memory-tight session. What is NOT
    optional is that the record says which of the two happened.
    """

    def _lazy(self, name="lazy_struct_fixture_clean.root"):
        fixture = os.path.join(os.path.dirname(__file__), name)
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    def _boom(self, adf, **kw):
        import dfextensions.dfdraw as _dfd
        real = _dfd.DFDraw.draw_batch

        def _raise(self, *a, **k):
            raise RuntimeError("render failed (injected)")

        _dfd.DFDraw.draw_batch = _raise
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError, match="injected"):
                    adf.draw_batch(
                        {"p": {"expr": "shift", "type": "hist", "bins": 5}},
                        lazy=True, clear_after=True, verbose=False, **kw)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")

    def test_b32_36_default_leaves_the_evidence_in_place(self):
        adf = self._lazy()
        adf.add_alias("shift", "mult + 1")
        self._boom(adf)
        st = adf._last_draw_prep_state
        assert st is not None, (
            "a raised render left no record at all — the call did happen and "
            "did have effects")
        assert st.cleanup_outcome == "skipped_after_failure", \
            st.cleanup_outcome
        assert "shift" in st.cleanup_candidates
        assert st.aliases_dropped == ()
        assert "shift" in adf.df.columns, (
            "the default must not clean up on failure — that is the "
            "pre-B3.2 behaviour and changing it silently is a behaviour "
            "change dressed as a correction")

    def test_b32_37_opting_in_cleans_up_and_says_so(self):
        adf = self._lazy()
        adf.add_alias("shift", "mult + 1")
        self._boom(adf, clear_after_on_error=True)
        st = adf._last_draw_prep_state
        assert st.cleanup_outcome == "ran_after_failure", \
            st.cleanup_outcome
        assert "shift" in st.aliases_dropped, st.aliases_dropped
        assert "shift" not in adf.df.columns

    def test_b32_38_outcome_distinguishes_the_three_quiet_cases(self):
        """cleanup_candidates being empty is ambiguous between three different
        situations. The outcome field exists so a consumer can tell them
        apart; a field that collapses them would be no better than silence."""
        adf = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "mult", "type": "hist", "bins": 5}},
                           clear_after=False, verbose=False)
        plt.close("all")
        assert adf._last_draw_prep_state.cleanup_outcome == "not_requested"
        adf2 = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf2.draw_batch({"p": {"expr": "mult", "type": "hist", "bins": 5}},
                            clear_after=True, verbose=False)
        plt.close("all")
        assert adf2._last_draw_prep_state.cleanup_outcome == "nothing_to_clean"


@needs_dfdraw
class TestB32EntrySelectionProjection:
    """The correction round's blocking finding, found by five reviewers
    independently (GPT25, GPT26, GPT27, GPT30, GPT31), each by executing a
    call shape this suite did not build.

    `entry_begin`/`entry_end`/`entry_mask` became a `draw_batch` contract in
    B1. Subframe projection is what part 2 consolidated. Nothing had ever
    combined them. The join is computed over the WHOLE parent frame, so its
    result could not be assigned into the entry-selected reduced frame; the
    length mismatch was caught by the broad subframe `except`, downgraded to a
    warning, the dotted reference was never rewritten, and dfdraw then failed
    with a NameError that looked like something else entirely. Every entry
    form was affected.

    Every test here compares against an INDEPENDENTLY COMPUTED join restricted
    to the same rows. Asserting "no exception" would have passed against a
    projection that silently produced the wrong values, and getting the right
    rows is the entire content of the fix.
    """

    def _pair(self, n=24, seed=11):
        rng = np.random.default_rng(seed)
        main = A.AliasDataFrame(pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "sec": rng.integers(0, 4, n).astype(np.int32)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(4, dtype=np.int32),
            "corr": np.array([1., 2., 3., 4.])}))
        main.register_subframe("S", child, index_columns=["sec"])
        return main, child

    def _oracle(self, main, child, positions):
        """The join, computed by hand, for exactly those parent rows."""
        lookup = dict(zip(child.df["sec"].values, child.df["corr"].values))
        return np.array([lookup[s] for s in main.df["sec"].values[positions]])

    def _capture(self, adf, specs, **kw):
        """Run the real path, capturing the frame and specs dfdraw received."""
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["df"] = self.df.copy()
            seen["specs"] = k.get("specs") if "specs" in k else a[0]
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                adf.draw_batch(specs, verbose=False, **kw)
                seen["warnings"] = [str(w.message) for w in caught]
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        return seen

    def _check(self, main, child, positions, seen):
        assert len(seen["df"]) == len(positions), (
            f"delegated frame has {len(seen['df'])} rows for "
            f"{len(positions)} selected")
        assert "S_corr" in seen["df"].columns, (
            "the flattened subframe column was never created")
        np.testing.assert_allclose(
            seen["df"]["S_corr"].values, self._oracle(main, child, positions),
            err_msg="flattened values do not match a hand-computed join "
                    "restricted to the selected rows")
        assert seen["specs"]["p"]["expr"] == "S_corr:x", (
            f"the expression reaching dfdraw is still unresolved: "
            f"{seen['specs']['p']['expr']!r}")
        assert not [w for w in seen["warnings"] if "subframe" in w], \
            seen["warnings"]

    def test_b32_39_entry_window_x_subframe_column(self):
        main, child = self._pair()
        seen = self._capture(main, {"p": {"expr": "S.corr:x",
                                          "type": "scatter"}},
                             entry_begin=5, entry_end=17)
        self._check(main, child, np.arange(5, 17), seen)

    def test_b32_40_boolean_mask_x_subframe_column(self):
        main, child = self._pair()
        mask = np.zeros(len(main.df), dtype=bool)
        mask[::3] = True
        seen = self._capture(main, {"p": {"expr": "S.corr:x",
                                          "type": "scatter"}},
                             entry_mask=mask)
        self._check(main, child, np.flatnonzero(mask), seen)

    def test_b32_41_integer_mask_x_subframe_column(self):
        """Integer masks are positional and may be out of order — the case
        where "just truncate the array to the right length" would look like a
        fix and produce wrong numbers."""
        main, child = self._pair()
        idx = np.array([9, 2, 17, 4, 0])
        seen = self._capture(main, {"p": {"expr": "S.corr:x",
                                          "type": "scatter"}},
                             entry_mask=idx)
        self._check(main, child, idx, seen)

    def test_b32_42_entry_window_x_subframe_alias_with_cleanup(self):
        """The alias variant, plus cleanup: the child alias is materialized by
        projection, joined for the selected rows, and dropped afterwards."""
        main, child = self._pair()
        child.add_alias("corr2", "corr * 2")
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["df"] = self.df.copy()
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main.draw_batch({"p": {"expr": "S.corr2:x",
                                       "type": "scatter"}},
                                entry_begin=3, entry_end=11,
                                clear_after=True, verbose=False)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        pos = np.arange(3, 11)
        assert len(seen["df"]) == len(pos)
        expected = self._oracle(main, child, pos) * 2
        np.testing.assert_allclose(seen["df"]["S_corr2"].values, expected)
        st = main._last_draw_prep_state
        assert "S::corr2" in st.aliases_dropped, st.aliases_dropped
        assert "corr2" not in child.df.columns

    def test_b32_43_nested_subframe_x_entry_window(self):
        """The multi-level route writes a PERSISTENT column onto self.df and
        then copies it into the reduced frame. That copy used to be a Series
        assignment, which aligns by index LABEL — only accidentally right, and
        wrong outright when labels repeat. It is positional now."""
        rng = np.random.default_rng(5)
        n = 30
        main = A.AliasDataFrame(pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "sec": rng.integers(0, 4, n).astype(np.int32)}))
        mid = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(4, dtype=np.int32),
            "grp": np.arange(4, dtype=np.int32)}))
        leaf = A.AliasDataFrame(pd.DataFrame({
            "grp": np.arange(4, dtype=np.int32),
            "val": np.array([10., 20., 30., 40.])}))
        mid.register_subframe("Leaf", leaf, index_columns=["grp"])
        main.register_subframe("Mid", mid, index_columns=["sec"])
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["df"] = self.df.copy()
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main.draw_batch({"p": {"expr": "Mid.Leaf.val:x",
                                       "type": "scatter"}},
                                entry_begin=7, entry_end=19, verbose=False)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        pos = np.arange(7, 19)
        flat = [c for c in seen["df"].columns if str(c).startswith("val__")]
        assert flat, list(seen["df"].columns)
        expected = main.df[flat[0]].values[pos]
        np.testing.assert_allclose(seen["df"][flat[0]].values, expected)

    def test_b32_44_duplicate_index_labels_do_not_misalign(self):
        """Why positional and not reindex-by-label. Nothing forbids duplicate
        index labels; aligning by label on such a frame either fans out or
        silently picks the wrong row."""
        rng = np.random.default_rng(8)
        n = 20
        main = A.AliasDataFrame(pd.DataFrame(
            {"x": rng.normal(0, 1, n),
             "sec": rng.integers(0, 4, n).astype(np.int32)},
            index=np.zeros(n, dtype=int)))          # every label identical
        child = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(4, dtype=np.int32),
            "corr": np.array([1., 2., 3., 4.])}))
        main.register_subframe("S", child, index_columns=["sec"])
        seen = self._capture(main, {"p": {"expr": "S.corr:x",
                                          "type": "scatter"}},
                             entry_begin=4, entry_end=12)
        self._check(main, child, np.arange(4, 12), seen)


@needs_dfdraw
class TestB32RewriteScopeAndRepeatCalls:
    """Two more findings from the correction round, both GPT25's."""

    def _pair(self, n=16, seed=13):
        rng = np.random.default_rng(seed)
        main = A.AliasDataFrame(pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "sec": rng.integers(0, 4, n).astype(np.int32)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(4, dtype=np.int32),
            "corr": np.array([1., 2., 3., 4.])}))
        main.register_subframe("S", child, index_columns=["sec"])
        return main, child

    def test_b32_45_subframe_ref_in_defaults_is_rewritten(self):
        """A subframe reference supplied through `defaults` — a supported way
        to give one expression to a whole batch — was joined, got its
        flattened column, and then reached dfdraw still spelled with the dot,
        because the rewrite loop only walked the per-plot spec dicts."""
        main, child = self._pair()
        defaults = {"expr": "S.corr:x"}
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["defaults"] = dict(k.get("defaults") or {})
            seen["cols"] = list(self.df.columns)
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main.draw_batch({"p": {"type": "scatter"}}, defaults=defaults,
                                verbose=False)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        assert seen["defaults"].get("expr") == "S_corr:x", (
            f"dfdraw received {seen['defaults'].get('expr')!r} — the dotted "
            f"reference was joined but never rewritten")
        assert "S_corr" in seen["cols"], seen["cols"]
        # ...and the CALLER's dictionary is still untouched. draw_batch takes
        # a structural copy at entry (B1); rewriting the caller's object would
        # be the phantom-ax defect that increment removed.
        assert defaults["expr"] == "S.corr:x", (
            "the caller's defaults dict was mutated — B1's structural-copy "
            "contract is broken")

    def test_b32_46_subframe_ref_in_kwargs_is_rewritten(self):
        main, child = self._pair()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main.draw_batch({"p": {"type": "scatter"}}, expr="S.corr:x",
                            verbose=False)
        plt.close("all")

    def test_b32_47_repeat_call_does_not_relabel_a_persistent_column(self):
        """On a second call the persistent multi-level column already exists,
        so it is absent from this call's own write diff. Classifying on that
        diff reported it as a reduced-frame temporary — a column documented as
        "discarded when the call returns" that in fact lives on self.df."""
        rng = np.random.default_rng(21)
        n = 40
        main = A.AliasDataFrame(pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "sec": rng.integers(0, 4, n).astype(np.int32)}))
        mid = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(4, dtype=np.int32),
            "grp": np.arange(4, dtype=np.int32)}))
        leaf = A.AliasDataFrame(pd.DataFrame({
            "grp": np.arange(4, dtype=np.int32),
            "val": np.array([10., 20., 30., 40.])}))
        mid.register_subframe("Leaf", leaf, index_columns=["grp"])
        main.register_subframe("Mid", mid, index_columns=["sec"])
        spec = lambda: {"p": {"expr": "Mid.Leaf.val:x", "type": "scatter"}}
        for _ in range(2):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main.draw_batch(spec(), verbose=False)
            plt.close("all")
            st = main._last_draw_prep_state
            for c in st.temporary_columns:
                assert c not in main.df.columns, (
                    f"{c!r} is reported temporary but lives on the big frame")
        assert any(str(c).startswith("val__") for c in main.df.columns)


@needs_dfdraw
class TestB32ProjectionFailureIsLoud:
    """Architect ruling D1 Option 3 (2026-07-27).

    A failed subframe resolution used to warn and continue with an unrewritten
    dotted reference; dfdraw then failed with a NameError about an undefined
    subframe name, which reads like a user typo rather than an ADF failure.
    That is what hid the entry-selection P0 for the whole of part 2 — the
    warning was there, in every run, and nothing was watching for it.

    The ruling: raise at the phase that owns the effect, with an escape hatch.
    """

    def _broken(self):
        """A subframe reference that cannot resolve AT PROJECTION.

        ROUND 8: this fixture used to give the child a mismatched join key
        (`secX` against the parent's `sec`), so registration itself was the
        broken step. That worked only because `register_subframe` never
        validated symmetric keys — GPT25's B32F7-P1-2, now fixed: the key is
        checked before any state is written, so the old fixture is refused at
        registration and never reaches the projection phase this class is
        about.

        The fixture now models the case D1 was actually ruled for: the join
        resolves, and the referenced COLUMN does not exist on the child. That
        is the shape a user hits with a typo in `S.v`, and it is the one that
        used to warn and then die inside dfdraw with a NameError."""
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(6.), "sec": np.arange(6, dtype=np.int32)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(6, dtype=np.int32), "other": np.arange(6.)}))
        main.register_subframe("S", child, index_columns=["sec"])
        return main

    def test_b32_47b_mismatched_join_key_is_refused_at_registration(self):
        """The shape the old `_broken()` fixture used, kept as its own test so
        the coverage is not lost: a child whose join key does not exist is now
        refused before any state is written (GPT25 B32F7-P1-2)."""
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(6.), "sec": np.arange(6, dtype=np.int32)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "secX": np.arange(6, dtype=np.int32), "v": np.arange(6.)}))
        with pytest.raises(ValueError, match="right_index_columns not found"):
            main.register_subframe("S", child, index_columns=["sec"])
        assert "S" not in main._subframes.subframes
        assert "S" not in main._schema.get("subframes", {})
        assert not child._schema.get("columns")

    def test_b32_48_unresolvable_subframe_ref_raises_by_default(self):
        main = self._broken()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="S.v"):
                main.draw_batch({"p": {"expr": "S.v:x", "type": "scatter"}},
                                verbose=False)
        plt.close("all")
        st = main._last_draw_prep_state
        assert st.failure_phase == "projection", st.failure_phase

    def test_b32_49_warn_restores_the_previous_behaviour(self):
        main = self._broken()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(Exception):
                main.draw_batch({"p": {"expr": "S.v:x", "type": "scatter"}},
                                on_subframe_error="warn", verbose=False)
        plt.close("all")
        assert any("subframe" in str(w.message) for w in caught), \
            [str(w.message) for w in caught]

    def test_b32_50_failure_outcomes_name_the_phase_and_the_choice(self):
        """GPT27: one `else` branch was covering two different situations, so
        a render failure with clear_after=False claimed cleanup had been
        SKIPPED when it had never been requested."""
        import dfextensions.dfdraw as _dfd
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")

        def _run(clear_after, on_error, exc=RuntimeError):
            adf = A.AliasDataFrame.read_tree_lazy(fixture, "tree")
            adf.add_alias("shift", "mult + 1")
            real = _dfd.DFDraw.draw_batch

            def _boom(self, *a, **k):
                raise exc("injected")

            _dfd.DFDraw.draw_batch = _boom
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        adf.draw_batch(
                            {"p": {"expr": "shift", "type": "hist",
                                   "bins": 5}},
                            lazy=True, clear_after=clear_after,
                            clear_after_on_error=on_error, verbose=False)
                    except BaseException:
                        pass
            finally:
                _dfd.DFDraw.draw_batch = real
                plt.close("all")
            st = adf._last_draw_prep_state
            return st, ("shift" in adf.df.columns)

        st, present = _run(False, False)
        assert st.cleanup_outcome == "not_requested", st.cleanup_outcome
        assert st.failure_phase == "render"
        assert present

        st, present = _run(True, False)
        assert st.cleanup_outcome == "skipped_after_failure", st.cleanup_outcome
        assert present

        st, present = _run(True, True)
        assert st.cleanup_outcome == "ran_after_failure", st.cleanup_outcome
        assert not present

    def test_b32_51_keyboard_interrupt_is_not_a_render_failure(self):
        """The handler caught BaseException, so Ctrl-C during a long batch
        was treated as a failed plot and could delete the user's columns on
        the way out."""
        import dfextensions.dfdraw as _dfd
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        adf = A.AliasDataFrame.read_tree_lazy(fixture, "tree")
        adf.add_alias("shift", "mult + 1")
        real = _dfd.DFDraw.draw_batch

        def _interrupt(self, *a, **k):
            raise KeyboardInterrupt()

        _dfd.DFDraw.draw_batch = _interrupt
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(KeyboardInterrupt):
                    adf.draw_batch(
                        {"p": {"expr": "shift", "type": "hist", "bins": 5}},
                        lazy=True, clear_after=True,
                        clear_after_on_error=True, verbose=False)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        assert "shift" in adf.df.columns, (
            "an interrupted session had its materialized column deleted")


@needs_dfdraw
class TestB32DuplicateRegistrationScope:
    """The exact width of the D2 refusal, pinned as a matrix.

    The ruling forbids one mutable OBJECT reachable twice; it is not a ban on
    reusing a frame. Getting the width wrong in either direction breaks
    something real, and both directions were hit while implementing it: too
    wide broke `test_N1_7_cycle_detection`, and the first narrow version was
    ORDER-DEPENDENT — the same final structure was accepted or refused
    depending on which registration came last.

    The rule is a property of the RESULT: after this registration, would one
    object be reachable by two distinct paths in this graph?
    """

    def _f(self, n=4):
        return A.AliasDataFrame(pd.DataFrame(
            {"k": np.arange(n, dtype=np.int32)}))

    def test_b32_52_two_disjoint_parents_stay_legal(self):
        """`examples/time_series/time_series_TroubleShooting.py` registers one
        grouped frame into `adf` AND into `adfgbTPCDSec20`. Those two do not
        share a graph, so there is no single record describing both and no
        contradiction to prevent. Refusing it would break working analysis
        code to serve a rule aimed at something else."""
        a, b, child = self._f(), self._f(), self._f()
        a.register_subframe("C", child, index_columns=["k"])
        b.register_subframe("C", child, index_columns=["k"])
        assert a.get_subframe("C") is b.get_subframe("C")

    def test_b32_53_second_path_in_one_graph_is_refused(self):
        root, mid, child = self._f(), self._f(), self._f()
        root.register_subframe("Mid", mid, index_columns=["k"])
        mid.register_subframe("C", child, index_columns=["k"])
        with pytest.raises(ValueError, match="two paths"):
            root.register_subframe("C", child, index_columns=["k"])

    def test_b32_54_the_refusal_is_order_independent(self):
        """Same three registrations, different order, same final structure —
        so the same answer. The first implementation asked "have I already
        seen this object", which made the answer depend on the route."""
        root, mid, child = self._f(), self._f(), self._f()
        root.register_subframe("C", child, index_columns=["k"])
        mid.register_subframe("C", child, index_columns=["k"])
        with pytest.raises(ValueError, match="two paths"):
            root.register_subframe("Mid", mid, index_columns=["k"])

    def test_b32_55_two_names_one_object_is_refused(self):
        parent, child = self._f(), self._f()
        parent.register_subframe("A", child, index_columns=["k"])
        with pytest.raises(ValueError, match="two paths"):
            parent.register_subframe("B", child, index_columns=["k"])

    def test_b32_56_re_registering_the_same_name_is_an_update(self):
        parent, child = self._f(), self._f()
        parent.register_subframe("C", child, index_columns=["k"])
        parent.register_subframe("C", child, index_columns=["k"])
        assert parent.get_subframe("C") is child

    def test_b32_57_distinct_instances_under_distinct_names_are_legal(self):
        parent = self._f()
        parent.register_subframe("A", self._f(), index_columns=["k"])
        parent.register_subframe("B", self._f(), index_columns=["k"])
        assert {"A", "B"} <= set(parent._subframes.subframes)

    def test_b32_58_self_registration_is_out_of_scope(self):
        """A cycle, not an aliased child: one name, and materialize_aliases
        already refuses it where it does harm (test_N1_7_cycle_detection).
        Refusing it here too would be a second behaviour change riding along
        with the ruling."""
        frame = self._f()
        frame.register_subframe("Self", frame, index_columns=["k"])
        assert "Self" in frame._subframes.subframes


@needs_dfdraw
class TestB32MissingJoinKeys:
    """Correction round 2's blocking finding — all five GPT seats, one
    mechanism, and the class of bug this whole phase keeps producing.

    `_compute_join_indices` returns `-1` as its missing-key sentinel plus a
    `missing` mask. The projection phase captured `missing` and never read it,
    then did `values[join_idx]`. NumPy reads `-1` as "last row", so a parent
    key with no child match silently received the child's FINAL value. No
    exception, no warning, wrong numbers in a plot.

    Why 2,355 passing tests could coexist with it: every subframe fixture in
    this file generated parent keys inside the child's key range, so the
    missing branch could not fire. My own "compare against a hand-computed
    join" oracle was built on a fixture where every key matched — an oracle
    that could not fail in the way that mattered.

    **Standing rule from this round: no value oracle may use a fixture where
    every parent key is present in the child.** Every fixture below carries at
    least one unmatched key.
    """

    def _pair(self, keys=(0, 1, 9, 2, 9, 3)):
        """Parent keys 0..3 exist in the child; 9 deliberately does not."""
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(len(keys), dtype=float),
            "sec": np.asarray(keys, dtype=np.int32)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(4, dtype=np.int32),
            "corr": np.array([10., 20., 30., 40.])}))
        main.register_subframe("S", child, index_columns=["sec"])
        return main, child

    def _oracle(self, main, child, positions, fill=np.nan, col="corr"):
        """A LEFT join, computed by hand, for exactly those parent rows."""
        lookup = dict(zip(child.df["sec"].values, child.df[col].values))
        return np.array([lookup.get(k, fill)
                         for k in main.df["sec"].values[positions]])

    def _delegated(self, main, expr="S.corr:x", **kw):
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["df"] = self.df.copy()
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main.draw_batch({"p": {"expr": expr, "type": "scatter"}},
                                verbose=False, **kw)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        return seen["df"]

    def test_b32_59_missing_key_is_nan_not_the_last_child_row(self):
        main, child = self._pair()
        got = self._delegated(main)["S_corr"].values
        np.testing.assert_allclose(
            got, self._oracle(main, child, np.arange(len(main.df))),
            err_msg="missing keys did not come back as NaN")
        assert np.isnan(got[2]) and np.isnan(got[4]), (
            f"unmatched rows got {got[2]}, {got[4]} — the child's last value "
            f"is 40.0, which is what raw -1 indexing produces")

    def test_b32_60_configured_fill_missing_is_honoured(self):
        main, child = self._pair()
        main.set_subframe_fill("S", fill_missing=-999.0, fill_mode="direct")
        got = self._delegated(main)["S_corr"].values
        np.testing.assert_allclose(
            got, self._oracle(main, child, np.arange(len(main.df)),
                              fill=-999.0))

    def test_b32_61_missing_keys_survive_an_entry_window(self):
        main, child = self._pair()
        pos = np.arange(1, 5)          # includes both unmatched rows
        got = self._delegated(main, entry_begin=1, entry_end=5)["S_corr"].values
        np.testing.assert_allclose(got, self._oracle(main, child, pos))

    def test_b32_62_missing_keys_survive_an_unsorted_integer_mask(self):
        """The shape GPT25 named as the highest-value single test: unsorted
        positions mixing matched and unmatched keys. Truncating an array to
        the right length would pass a length check and fail this."""
        main, child = self._pair()
        pos = np.array([4, 0, 2, 5])
        got = self._delegated(main, entry_mask=pos)["S_corr"].values
        np.testing.assert_allclose(got, self._oracle(main, child, pos))

    def test_b32_63_missing_keys_survive_a_boolean_mask_with_fill(self):
        main, child = self._pair()
        main.set_subframe_fill("S", fill_missing=-1.0, fill_mode="direct")
        mask = np.array([True, False, True, True, True, False])
        got = self._delegated(main, entry_mask=mask)["S_corr"].values
        np.testing.assert_allclose(
            got, self._oracle(main, child, np.flatnonzero(mask), fill=-1.0))

    def test_b32_64_child_alias_with_a_missing_key(self):
        """The alias route goes through the same gather, so it must inherit
        the same missing-key contract rather than a second implementation."""
        main, child = self._pair()
        child.add_alias("corr2", "corr * 2")
        got = self._delegated(main, expr="S.corr2:x")["S_corr2"].values
        lookup = dict(zip(child.df["sec"].values,
                          child.df["corr"].values * 2))
        expected = np.array([lookup.get(k, np.nan)
                             for k in main.df["sec"].values])
        np.testing.assert_allclose(got, expected)

    def test_b32_65_every_key_missing(self):
        """The degenerate end of the range: a child that matches nothing must
        give an all-NaN column, not an all-last-row column."""
        main, child = self._pair(keys=(7, 8, 9))
        got = self._delegated(main)["S_corr"].values
        assert np.all(np.isnan(got)), got

    def test_b32_66_zero_selected_rows_with_missing_keys(self):
        main, _ = self._pair()
        got = self._delegated(main, entry_mask=np.zeros(6, dtype=bool))
        assert len(got) == 0
        assert "S_corr" in got.columns


@needs_dfdraw
class TestB32ProjectionRefusalCompleteness:
    """D1 says: raise at the phase that owns the effect. Two paths escaped it
    (GPT26, GPT27) because they warned and continued, so the call died later
    inside dfdraw and the record blamed `render` for a projection failure.

    Both are closed by the same move as the missing-key P0: the projection
    phase now borrows `_extract_subframe_values_cached`, which owns child-alias
    materialization and raises `KeyError` for an absent column. Borrowing a
    contract closes the paths that restating it had left open.
    """

    def _main(self):
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(6.), "sec": np.arange(6, dtype=np.int32)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(6, dtype=np.int32), "corr": np.arange(6.)}))
        main.register_subframe("S", child, index_columns=["sec"])
        return main, child

    def test_b32_67_missing_leaf_column_raises_at_projection(self):
        main, _ = self._main()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="S.nosuch"):
                main.draw_batch({"p": {"expr": "S.nosuch:x",
                                       "type": "scatter"}}, verbose=False)
        plt.close("all")
        assert main._last_draw_prep_state.failure_phase == "projection", (
            "the record blamed a later phase for a projection failure")

    def test_b32_68_unmaterializable_child_alias_raises_at_projection(self):
        main, child = self._main()
        child.add_alias("bad", "nonexistent_column + 1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="S.bad"):
                main.draw_batch({"p": {"expr": "S.bad:x",
                                       "type": "scatter"}}, verbose=False)
        plt.close("all")
        assert main._last_draw_prep_state.failure_phase == "projection"

    def test_b32_69_warn_mode_still_lets_both_through(self):
        main, child = self._main()
        child.add_alias("bad", "nonexistent_column + 1")
        for expr in ("S.nosuch:x", "S.bad:x"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with pytest.raises(Exception):
                    main.draw_batch({"p": {"expr": expr, "type": "scatter"}},
                                    on_subframe_error="warn", verbose=False)
            plt.close("all")
            assert any("subframe" in str(w.message) for w in caught), expr

    def test_b32_70_on_subframe_error_rejects_unknown_values(self):
        """Silently treating a typo as 'raise' would let a misspelling change
        failure behaviour without saying so."""
        main, _ = self._main()
        with pytest.raises(ValueError, match="on_subframe_error"):
            main.draw_batch({"p": {"expr": "x", "type": "hist", "bins": 5}},
                            on_subframe_error="Warn", verbose=False)
        plt.close("all")

    def test_b32_71_on_error_skip_does_not_rescue_a_bad_subframe_ref(self):
        """The two controls act at different phases and do not substitute for
        each other. With the default, projection raises before dfdraw sees any
        spec, so on_error='skip' cannot skip just the bad one — documented in
        the Args block, and pinned here so the documentation cannot drift."""
        main, _ = self._main()
        specs = {"good": {"expr": "x", "type": "hist", "bins": 5},
                 "bad": {"expr": "S.nosuch:x", "type": "scatter"}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError):
                main.draw_batch(specs, on_error="skip", verbose=False)
        plt.close("all")
        # ...and the documented escape hatch does rescue it
        main2, _ = self._main()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = main2.draw_batch(
                {"good": {"expr": "x", "type": "hist", "bins": 5},
                 "bad": {"expr": "S.nosuch:x", "type": "scatter"}},
                on_error="skip", on_subframe_error="warn", verbose=False)
        plt.close("all")
        assert res is not None


@needs_dfdraw
class TestB32FailureRecordCompleteness:
    """Every boundary a failure can cross must leave a record that is true.

    Three gaps (GPT26, GPT27, Sonet27): entry validation ran outside every
    bracket; the struct assertion and plotter construction sat between two
    brackets; and a phase that raised half-way left its alias fields empty, so
    the record said no alias had been materialized while one had — and
    `clear_after_on_error` could not clean what the record did not mention.
    """

    def _lazy(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    def test_b32_72_entry_validation_failure_is_recorded(self):
        adf = self._lazy()
        adf.add_alias("shift", "mult + 1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError):
                adf.draw_batch({"p": {"expr": "shift", "type": "hist",
                                      "bins": 5}},
                               lazy=True, entry_mask=np.ones(3, dtype=bool),
                               clear_after=True, verbose=False)
        plt.close("all")
        st = adf._last_draw_prep_state
        assert st is not None
        assert st.failure_phase == "entry_selection", st.failure_phase
        assert st.cleanup_outcome == "skipped_after_failure", \
            st.cleanup_outcome

    def test_b32_73_partial_preparation_failure_names_what_it_did(self):
        """One alias materializes, a later preparation step raises. The record
        must mention the alias that DID materialize — otherwise
        clear_after_on_error is asked to clean a list that omits it."""
        adf = self._lazy()
        adf.add_alias("good", "mult + 1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                adf.draw_batch(
                    {"p": {"expr": "good", "type": "hist", "bins": 5,
                           "selection_vector": ["S.nope > 0"]}},
                    lazy=True, clear_after=True, clear_after_on_error=True,
                    verbose=False)
            except Exception:
                pass
        plt.close("all")
        st = adf._last_draw_prep_state
        if st is None or not st.failure_phase:
            pytest.skip("this spec shape no longer fails during preparation")
        if "good" in adf.df.columns:
            assert "good" in st.cleanup_candidates, (
                "an alias was materialized and left behind, and the record "
                "does not list it as a cleanup candidate")

    def test_b32_74_cleanup_outcome_is_not_relabelled_when_empty(self):
        """Sonet27. With clear_after_on_error=True and nothing to clean, the
        cleanup phase wrote 'nothing_to_clean' over 'ran_after_failure', so the
        record no longer said a failure had occurred."""
        import dfextensions.dfdraw as _dfd
        adf = A.AliasDataFrame(pd.DataFrame({"x": np.arange(10.)}))
        real = _dfd.DFDraw.draw_batch

        def _boom(self, *a, **k):
            raise RuntimeError("injected")

        _dfd.DFDraw.draw_batch = _boom
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError):
                    adf.draw_batch({"p": {"expr": "x", "type": "hist",
                                          "bins": 5}},
                                   clear_after=True, clear_after_on_error=True,
                                   verbose=False)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        st = adf._last_draw_prep_state
        assert st.failure_phase == "render"
        assert st.cleanup_outcome == "ran_after_failure", st.cleanup_outcome


@needs_dfdraw
class TestB32LazySubframeJoinKeys:
    """Architect ruling (2026-07-27): fix this in the correction, do not defer.

    It was disclosed in two previous rounds as "a lazy subframe referenced only
    through an ALIAS". That description was wrong, and executing it says so: a
    plain physical column fails identically. The real condition is that once a
    lazy subframe has been materialized by ANYTHING — get_subframe(), an
    earlier draw, adding an alias to it — the PARENT's join index columns are
    never loaded, and every later draw_batch reference fails with
    "None of [Index(['sec'])] are in the [columns]".

    The cause was one guard: the parent's index load was nested inside "is the
    child still unloaded". Unrelated conditions.
    """

    def _file(self, tmp_path):
        return _write_tree_with_subframe(str(tmp_path / "lazyjoin.root"))

    def _adf(self, path):
        adf = A.AliasDataFrame.read_tree_lazy(path, "tree")
        adf.register_subframe_lazy("SectorCalib", path,
                                   tree_name="SectorCalib",
                                   index_columns=["sec"])
        return adf

    def _draw(self, adf, expr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": expr, "type": "scatter"}},
                           verbose=False)
        plt.close("all")

    def test_b32_75_physical_column_after_the_subframe_was_touched(self, tmp_path):
        """The case that proves it is not about aliases."""
        adf = self._adf(self._file(tmp_path))
        adf.get_subframe("SectorCalib")          # materialize it, nothing more
        self._draw(adf, "SectorCalib.corr:x")
        assert "sec" in adf.df.columns, (
            "the parent never loaded its join key")

    def test_b32_76_alias_on_an_already_materialized_lazy_subframe(self, tmp_path):
        adf = self._adf(self._file(tmp_path))
        sf = adf.get_subframe("SectorCalib")
        sf.add_alias("corr2", "corr * 2")
        self._draw(adf, "SectorCalib.corr2:x")

    def test_b32_77_two_draws_in_a_row(self, tmp_path):
        """The realistic shape: the first draw materializes the subframe, the
        second one used to fail because of it."""
        adf = self._adf(self._file(tmp_path))
        self._draw(adf, "SectorCalib.corr:x")
        self._draw(adf, "SectorCalib.pad:x")

    def test_b32_78_untouched_subframe_still_works(self, tmp_path):
        """Control — the path that already worked must keep working."""
        adf = self._adf(self._file(tmp_path))
        self._draw(adf, "SectorCalib.corr:x")


@needs_dfdraw
class TestB32DuplicateRegistrationLoopholes:
    """The two bypasses the panel found in the D2 refusal, plus the scope the
    architect ruled on.

    D2 is GRAPH-LOCAL (ruling 2026-07-27): within one reachable graph an
    object may not appear at two logical paths; across disconnected graphs it
    may, because no single record describes both. That is a deliberate choice
    with a cost, and test_b32_82 states the cost out loud rather than letting
    it read as isolation.
    """

    def _f(self, n=4):
        return A.AliasDataFrame(pd.DataFrame(
            {"k": np.arange(n, dtype=np.int32),
             "v": np.arange(n, dtype=float)}))

    def test_b32_79_self_under_two_different_names_is_refused(self):
        """GPT27's loophole. The graph walk skips the root — it has no
        subframe name — and the ancestor guard stops a self-referencing path
        from ever becoming a walked node, so neither self-registration was
        ever compared against the other."""
        frame = self._f()
        frame.register_subframe("Self1", frame, index_columns=["k"])
        with pytest.raises(ValueError, match="two paths"):
            frame.register_subframe("Self2", frame, index_columns=["k"])

    def test_b32_80_single_self_registration_remains_legal(self):
        """The cycle contract owns this one, and test_N1_7_cycle_detection
        has pinned it since long before this phase."""
        frame = self._f()
        frame.register_subframe("Self", frame, index_columns=["k"])
        assert "Self" in frame._subframes.subframes

    def test_b32_81_shared_descendant_of_two_subtrees_is_refused(self):
        """GPT25 and GPT26, independently. The refusal used to check only the
        object handed to it, never the descendants of the subtree it was
        attaching."""
        root, a, b, leaf = self._f(), self._f(), self._f(), self._f()
        a.register_subframe("L1", leaf, index_columns=["k"])
        b.register_subframe("L2", leaf, index_columns=["k"])
        root.register_subframe("A", a, index_columns=["k"])
        with pytest.raises(ValueError, match="two paths"):
            root.register_subframe("B", b, index_columns=["k"])

    def test_b32_82_disconnected_graphs_share_mutable_state_and_that_is_stated(self):
        """D2 is graph-local BY RULING, and the architect asked for the
        consequence to be documented rather than dressed up. This asserts the
        consequence so nobody reads the allowance as isolation: a column
        materialized through one parent IS visible through the other, because
        it is one object.

        Independent contexts need separate instances over the same source —
        which is what the ruling directs users to, and what
        test_b32_23a covers."""
        left, right, shared = self._f(), self._f(), self._f()
        left.register_subframe("S", shared, index_columns=["k"])
        right.register_subframe("S", shared, index_columns=["k"])
        left.get_subframe("S").df["side_effect"] = np.arange(4, dtype=float)
        assert "side_effect" in right.get_subframe("S").df.columns, (
            "if this ever fails, the frames became isolated and the "
            "documented hazard is stale — update the ruling text with it")
        assert left.get_subframe("S") is right.get_subframe("S")


@needs_dfdraw
class TestB32DtypePreservation:
    """AD-7/13.76.ADF (architect, 2026-07-28): the user's dtype is preserved.

    Correction round 3 fixed missing keys by borrowing
    `_extract_subframe_values_cached`, and that helper allocated
    `np.full(n, np.nan, dtype=float64)` for every non-floating source column.
    Four GPT seats executed it independently: categorical and string RAISED
    (`could not convert string to float`), while int, bool and datetime were
    silently coerced — datetime to raw epoch nanoseconds, which still plots.

    The lesson recorded in the code: borrowing a contract is right, but the
    borrowed contract has to cover what the replaced code covered. It did not.

    The ruling, measured rather than assumed:

        float / complex           NaN      dtype kept
        datetime64/timedelta64    NaT      dtype kept, 24 -> 24 bytes
        object                    None     dtype kept
        category                  native   dtype kept, 27 -> 27 bytes
        int                       NONE     NaN makes it float64
        bool                      NONE     NaN makes it object, 3 -> 24 bytes

    So int and bool are the only kinds with nowhere to put a missing value,
    and for those the EXISTING public fill contract decides — or it refuses.
    """

    COLUMNS = {
        "float64": np.array([1., 2., 3., 4.]),
        "float32": np.array([1., 2., 3., 4.], dtype=np.float32),
        "int64": np.array([1, 2, 3, 4], dtype=np.int64),
        "int16": np.array([1, 2, 3, 4], dtype=np.int16),
        "uint32": np.array([1, 2, 3, 4], dtype=np.uint32),
        "bool": np.array([True, False, True, False]),
        "datetime64": pd.to_datetime(
            ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
        "timedelta64": pd.to_timedelta([1, 2, 3, 4], unit="D"),
        "object": np.array(list("abcd"), dtype=object),
        "category": pd.Categorical(list("abcd")),
    }

    def _pair(self, col, keys, fill=None, fill_mode="direct"):
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(len(keys), dtype=float),
            "sec": np.asarray(keys, dtype=np.int32)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(4, dtype=np.int32), "v": col}))
        main.register_subframe("S", child, index_columns=["sec"])
        if fill is not None:
            main.set_subframe_fill("S", fill_missing=fill, fill_mode=fill_mode)
        return main, child

    def _delegated(self, main, slot="group_by", **kw):
        """Non-numeric columns cannot be a scatter axis, so the channel used
        here is `group_by` — which is GPT26's actual reproducer and a
        supported value-bearing subframe slot."""
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["df"] = self.df.copy()
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main.draw_batch({"p": {"expr": "x", "type": "hist", "bins": 3,
                                       slot: "S.v"}}, verbose=False, **kw)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        return seen["df"]["S_v"]

    @pytest.mark.parametrize("name", sorted(COLUMNS))
    def test_b32_83_matched_keys_preserve_the_exact_dtype(self, name):
        """Every key matches, so no missing value question arises at all —
        and therefore no reason for the dtype to change. The implementation
        allocates nothing on this path."""
        main, child = self._pair(self.COLUMNS[name], [0, 1, 2, 3])
        got = self._delegated(main)
        assert str(got.dtype) == str(child.df["v"].dtype), (
            f"{name}: {child.df['v'].dtype} became {got.dtype} with every "
            f"key matched")
        assert list(got.values) == list(child.df["v"].values)

    @pytest.mark.parametrize("name", ["float64", "float32", "datetime64",
                                      "timedelta64", "object", "category"])
    def test_b32_84_missing_uses_the_dtype_s_own_missing_value(self, name):
        """These dtypes each own a missing value that costs no dtype change
        and no memory. Using it is the ruling, not an exception to it."""
        main, child = self._pair(self.COLUMNS[name], [0, 1, 9, 3])
        got = self._delegated(main)
        assert str(got.dtype) == str(child.df["v"].dtype), (
            f"{name}: dtype changed to represent one missing key")
        assert pd.isna(got.values[2]), (
            f"{name}: the missing row is not missing: {got.values[2]!r}")
        assert not pd.isna(got.values[0])

    @pytest.mark.parametrize("name", ["int64", "int16", "uint32", "bool"])
    def test_b32_85_int_and_bool_refuse_rather_than_widen(self, name):
        """REVISED in round 10 under AD-19 (RATIFIED, architect 2026-07-29).

        This test has now been rewritten twice, in opposite directions, and
        the history is the point. In round 4 the coder made int/bool RAISE and
        broke three pre-phase tests, so it was inverted to pin the April
        NaN contract. AD-19 has now made refusal the ruling — but for a
        different and better reason than round 4's: not "an integer cannot
        hold NaN" but "the dtype is authoritative and only the user may choose
        the physical neutral value".

        The remedy is named in the error, and `test_b32_185` proves it works.
        """
        main, _ = self._pair(self.COLUMNS[name], [0, 1, 9, 3])
        with pytest.raises(ValueError, match="authoritative dtype"):
            self._delegated(main)

    def test_b32_86_configured_fill_membership_for_category(self):
        """GPT25 P2-2: the old `test_b32_86` was named for configured fill and
        configured none. GPT26 P1: a categorical fill that IS one of the
        column's own categories was refused by a generic `astype` round-trip
        that does not implement categorical semantics.

        Both are closed by the same move — the fill goes through pandas'
        own take, so pandas decides representability."""
        cat = pd.Categorical([1, 2, 3, 4])
        main, child = self._pair(cat, [0, 1, 9, 2], fill=1)
        got = self._delegated(main)
        assert str(got.dtype) == "category"
        assert got.values[2] == 1, list(got.values)
        # ...and a value that is NOT a category is refused, not silently added
        main2, _ = self._pair(cat, [0, 1, 9, 2], fill=99)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError):
                self._delegated(main2)

    def test_b32_87_alias_layer_honours_the_configured_fill(self):
        """REVISED in round 10 under AD-19. The alias layer no longer invents
        0/False, so what it restores is the CONFIGURED value, in the
        authoritative dtype, with every matched value exact."""
        main, _ = self._pair(np.array([1, 2, 3, 4], dtype=np.int8), [0, 1, 9, 3])
        main.add_alias("flagged", "S.v", dtype=np.int8, fill_value=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main.materialize_aliases(names=["flagged"])
        got = main.df["flagged"]
        assert got.dtype == np.int8
        assert list(got.values) == [1, 2, 0, 4]

    @pytest.mark.parametrize("slot", ["group_by", "facet_by", "color"])
    def test_b32_88_every_value_bearing_slot_behaves_the_same(self, slot):
        main, child = self._pair(self.COLUMNS["object"], [0, 1, 2, 3])
        got = self._delegated(main, slot=slot)
        assert str(got.dtype) == "object"

    def test_b32_89_entry_selection_does_not_change_the_dtype_rule(self):
        main, child = self._pair(self.COLUMNS["category"], [0, 1, 9, 3, 2, 0])
        got = self._delegated(main, entry_mask=np.array([4, 2, 0]))
        assert str(got.dtype) == "category"
        assert list(got.values[:1]) == ["c"]        # position 4 -> key 2
        assert pd.isna(got.values[1])               # position 2 -> key 9

    def test_b32_90_safe_fill_mode_reaches_the_projection(self):
        """`fill_mode='safe'` is the other documented mode and was in the
        untested-unknown list."""
        main, _ = self._pair(self.COLUMNS["float64"], [0, 1, 9, 3],
                             fill=-5.0, fill_mode="safe")
        got = self._delegated(main)
        assert str(got.dtype) == "float64"
        assert got.values[2] == -5.0


@needs_dfdraw
class TestB32GraphOwnershipValidator:
    """AD-8/13.76.ADF (architect, 2026-07-28).

    Registration-time refusal was defeated by ORDER three times. The fourth
    defeat, found by GPT31 and GPT25: attach two parents to a root, THEN give
    each of them the same child. Neither child registration can see the other,
    because a frame holds no back-reference to its parents.

    Rather than add back-references or a central registry, the graph is
    validated where it is CONSUMED — order-independent by construction,
    because it sees the graph that resulted rather than the sequence that
    built it.
    """

    def _f(self, n=4):
        return A.AliasDataFrame(pd.DataFrame(
            {"k": np.arange(n, dtype=np.int32),
             "x": np.arange(n, dtype=float)}))

    def _draw(self, adf):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "x", "type": "hist", "bins": 3}},
                           verbose=False)
        plt.close("all")

    def test_b32_91_delayed_connection_is_refused_at_draw(self):
        root, a, b, leaf = self._f(), self._f(), self._f(), self._f()
        root.register_subframe("A", a, index_columns=["k"])
        root.register_subframe("B", b, index_columns=["k"])
        a.register_subframe("L1", leaf, index_columns=["k"])
        b.register_subframe("L2", leaf, index_columns=["k"])
        with pytest.raises(ValueError, match="two different owner paths"):
            self._draw(root)

    def test_b32_92_the_error_names_both_paths(self):
        """A refusal that does not say WHERE the duplication is leaves the
        user to find it by hand in a graph they built incrementally."""
        root, a, b, leaf = self._f(), self._f(), self._f(), self._f()
        root.register_subframe("A", a, index_columns=["k"])
        root.register_subframe("B", b, index_columns=["k"])
        a.register_subframe("L1", leaf, index_columns=["k"])
        b.register_subframe("L2", leaf, index_columns=["k"])
        with pytest.raises(ValueError) as exc:
            self._draw(root)
        assert "A::L1" in str(exc.value) and "B::L2" in str(exc.value), \
            str(exc.value)

    def test_b32_93_validation_has_no_side_effects(self):
        """The architect's stated worry, made checkable: the validator must
        not load, materialize, join, mutate a cache or touch the record."""
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        adf = A.AliasDataFrame.read_tree_lazy(fixture, "tree")
        adf.add_alias("shift", "mult + 1")
        child = A.AliasDataFrame(pd.DataFrame(
            {"mult": np.arange(4, dtype=np.int32),
             "v": np.arange(4, dtype=float)}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.ensure_branches(["mult"])
            adf.register_subframe("Child", child, index_columns=["mult"])
            before = (
                list(adf.df.columns), list(child.df.columns),
                sorted(adf.aliases), sorted(child.aliases),
                sorted(getattr(adf._lazy_reader, "loaded_branches", []) or []),
                len(getattr(adf, "_join_index_cache", None) or {}),
                getattr(adf, "_struct_catalog_fp", None),
                getattr(adf, '_last_draw_prep_state', None),
            )
            adf._validate_frame_graph_ownership()
            after = (
                list(adf.df.columns), list(child.df.columns),
                sorted(adf.aliases), sorted(child.aliases),
                sorted(getattr(adf._lazy_reader, "loaded_branches", []) or []),
                len(getattr(adf, "_join_index_cache", None) or {}),
                getattr(adf, "_struct_catalog_fp", None),
                getattr(adf, '_last_draw_prep_state', None),
            )
        assert before == after, (
            "the ownership validator changed observable state; it is "
            "specified read-only and runs before every draw_batch")

    def test_b32_94_legal_graphs_are_untouched(self):
        root = self._f()
        root.register_subframe("A", self._f(), index_columns=["k"])
        root.register_subframe("B", self._f(), index_columns=["k"])
        self._draw(root)                       # distinct children
        z = self._f()
        z.register_subframe("Self", z, index_columns=["k"])
        self._draw(z)                          # cycle contract preserved
        left, right, shared = self._f(), self._f(), self._f()
        left.register_subframe("S", shared, index_columns=["k"])
        right.register_subframe("S", shared, index_columns=["k"])
        self._draw(left)                       # disconnected sharing is legal

    def test_b32_95_validation_runs_before_any_effect(self):
        """If it ran after preparation, an ambiguous graph would already have
        loaded branches and materialized aliases before being refused."""
        root, a, b, leaf = self._f(), self._f(), self._f(), self._f()
        root.add_alias("doubled", "x * 2")
        root.register_subframe("A", a, index_columns=["k"])
        root.register_subframe("B", b, index_columns=["k"])
        a.register_subframe("L1", leaf, index_columns=["k"])
        b.register_subframe("L2", leaf, index_columns=["k"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="two different owner paths"):
                root.draw_batch({"p": {"expr": "doubled", "type": "hist",
                                       "bins": 3}}, lazy=True, verbose=False)
        plt.close("all")
        assert "doubled" not in root.df.columns, (
            "an alias was materialized before the graph was refused")


@needs_dfdraw
class TestB32CleanupBracket:
    """The last unbracketed effect boundary (GPT25 P0-3, GPT31 P1-1)."""

    def _lazy(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        adf = A.AliasDataFrame.read_tree_lazy(fixture, "tree")
        adf.add_alias("shift", "mult + 1")
        return adf

    def test_b32_96_cleanup_failure_is_recorded_as_cleanup(self):
        adf = self._lazy()
        cls = type(adf)
        real = cls._dematerialize_qualified
        cls._dematerialize_qualified = (
            lambda self, names: (_ for _ in ()).throw(
                RuntimeError("cleanup injected")))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError, match="cleanup injected"):
                    adf.draw_batch({"p": {"expr": "shift", "type": "hist",
                                          "bins": 5}},
                                   lazy=True, clear_after=True, verbose=False)
        finally:
            cls._dematerialize_qualified = real
            plt.close("all")
        st = adf._last_draw_prep_state
        assert st.failure_phase == "cleanup", st.failure_phase
        assert st.cleanup_outcome == "failed", st.cleanup_outcome
        assert "shift" in st.cleanup_candidates
        assert "shift" in adf.df.columns, (
            "the drop failed, so the column must still be there — and the "
            "record must say so")

    def test_b32_97_the_failing_cleanup_is_not_re_entered(self):
        """_record_draw_failure runs cleanup when clear_after_on_error is set.
        Routing a cleanup failure through it would call the same failing
        cleanup a second time."""
        adf = self._lazy()
        calls = []
        cls = type(adf)
        real = cls._dematerialize_qualified

        def _boom(self, names):
            calls.append(1)
            raise RuntimeError("cleanup injected")

        cls._dematerialize_qualified = _boom
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError):
                    adf.draw_batch({"p": {"expr": "shift", "type": "hist",
                                          "bins": 5}},
                                   lazy=True, clear_after=True,
                                   clear_after_on_error=True, verbose=False)
        finally:
            cls._dematerialize_qualified = real
            plt.close("all")
        assert len(calls) == 1, f"cleanup was re-entered {len(calls)} times"


@needs_dfdraw
class TestB32PreviouslyUntestedCombinations:
    """Sonet29's "untested — status unknown" row, converted into a status.

    Four combinations were listed as neither confirmed working nor confirmed
    broken: a missing key on a LAZY subframe, a missing key on a NESTED path,
    an EMPTY child table, and `fill_mode='safe'`. Leaving a category called
    "unknown" open across rounds is how the missing-key P0 survived four of
    them, so each one is executed here and becomes either a passing assertion
    or a disclosed defect.
    """

    def _oracle(self, parent_keys, lookup, fill=np.nan):
        return np.array([lookup.get(k, fill) for k in parent_keys])

    def test_b32_98_missing_key_on_a_lazy_subframe(self, tmp_path):
        p = _write_tree_with_subframe(str(tmp_path / "lazymiss.root"))
        adf = A.AliasDataFrame.read_tree_lazy(p, "tree")
        adf.register_subframe_lazy("SectorCalib", p, tree_name="SectorCalib",
                                   index_columns=["sec"])
        # the fixture's parent has sectors 0..3 and the child covers 0..3, so
        # a key is REMOVED from the child to create the missing case rather
        # than assuming one exists (the standing no-all-matching-fixture rule)
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["df"] = self.df.copy()
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sf = adf.get_subframe("SectorCalib")
                sf.df.drop(sf.df.index[-1], inplace=True)   # drop sector 3
                adf.draw_batch({"p": {"expr": "SectorCalib.corr:x",
                                      "type": "scatter"}}, verbose=False)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        got = seen["df"]["SectorCalib_corr"]
        lookup = dict(zip(sf.df["sec"].values, sf.df["corr"].values))
        expected = self._oracle(adf.df["sec"].values[:len(got)], lookup)
        np.testing.assert_allclose(got.values, expected)
        assert got.isna().any(), (
            "no key was actually missing — the fixture stopped exercising "
            "the branch this test exists for")

    def test_b32_99_missing_key_on_a_nested_path(self):
        """The multi-level route uses _prepare_subframe_joins, a DIFFERENT
        implementation from the single-level gather. Whatever it does with a
        missing key, it must be recorded rather than unknown."""
        rng = np.random.default_rng(4)
        n = 20
        main = A.AliasDataFrame(pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "sec": np.array([0, 1, 2, 7] * 5, dtype=np.int32)}))
        mid = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(3, dtype=np.int32),
            "grp": np.arange(3, dtype=np.int32)}))          # no sector 7
        leaf = A.AliasDataFrame(pd.DataFrame({
            "grp": np.arange(3, dtype=np.int32),
            "val": np.array([10., 20., 30.])}))
        mid.register_subframe("Leaf", leaf, index_columns=["grp"])
        main.register_subframe("Mid", mid, index_columns=["sec"])
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["df"] = self.df.copy()
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main.draw_batch({"p": {"expr": "Mid.Leaf.val:x",
                                       "type": "scatter"}}, verbose=False)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        flat = [c for c in seen["df"].columns if str(c).startswith("val__")]
        assert flat, list(seen["df"].columns)
        got = seen["df"][flat[0]]
        lookup = {0: 10., 1: 20., 2: 30.}
        expected = self._oracle(main.df["sec"].values, lookup)
        np.testing.assert_allclose(got.values, expected, equal_nan=True)

    def test_b32_100_empty_child_table(self):
        """Nothing matches because there is nothing to match. Must be all
        missing, not an exception from an empty gather."""
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(5.), "sec": np.arange(5, dtype=np.int32)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "sec": np.array([], dtype=np.int32),
            "corr": np.array([], dtype=float)}))
        main.register_subframe("S", child, index_columns=["sec"])
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["df"] = self.df.copy()
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main.draw_batch({"p": {"expr": "S.corr:x",
                                       "type": "scatter"}}, verbose=False)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        assert seen["df"]["S_corr"].isna().all(), \
            list(seen["df"]["S_corr"].values)

    def test_b32_101_safe_fill_mode_on_a_numeric_column(self):
        """`fill_mode='safe'` additionally handles NaN and Inf already present
        in the child data, distinct from missing keys."""
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(4.), "sec": np.array([0, 1, 9, 2], dtype=np.int32)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(3, dtype=np.int32),
            "corr": np.array([1.0, np.nan, np.inf])}))
        main.register_subframe("S", child, index_columns=["sec"])
        main.set_subframe_fill("S", fill_missing=-1.0, fill_nan=-2.0,
                               fill_inf=-3.0, fill_mode="safe")
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["df"] = self.df.copy()
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main.draw_batch({"p": {"expr": "S.corr:x",
                                       "type": "scatter"}}, verbose=False)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        got = list(seen["df"]["S_corr"].values)
        assert got[0] == 1.0
        assert got[1] == -2.0, f"child NaN not filled by fill_nan: {got}"
        assert got[2] == -1.0, f"missing key not filled by fill_missing: {got}"
        assert got[3] == -3.0, f"child Inf not filled by fill_inf: {got}"


# ============================================================================
# STANDING DTYPE MATRIX — generated, not hand-written.
#
# Sonet29's structural finding, correction round 4: three of five B3.2 review
# rounds landed in the dtype domain, each time on a corner the previous
# round's hand-written adversarial variant did not reach — first non-float
# generally, then complex / timezone-aware / nullable-extension / empty
# non-numeric child. That is not a persistent lapse by any reviewer. It is a
# property of the domain: pandas' dtype surface is larger than any list a
# person maintains, so enumerating it by hand keeps missing a different piece.
#
# The answer is a matrix whose coverage is a property of the TABLE rather than
# of what anyone thought of on the day. Add a dtype to DTYPE_CASES and every
# combination below is exercised automatically.
#
# The implementation matches: one call to pandas' own `take(allow_fill=True)`
# instead of a branch per dtype, so the table and the code are general in the
# same way.
# ============================================================================

DTYPE_CASES = {
    "float64":     (np.array([1., 2., 3., 4.]),                       "float64"),
    "float32":     (np.array([1., 2., 3., 4.], dtype=np.float32),     "float32"),
    "complex64":   (np.array([1+2j, 2+3j, 3+4j, 4+5j], np.complex64), "complex64"),
    "complex128":  (np.array([1+2j, 2+3j, 3+4j, 4+5j], np.complex128), "complex128"),
    # int / uint / bool moved to DTYPE_REFUSED_ON_MISSING in round 10:
    # AD-19 makes every existing column dtype authoritative, so a gap they
    # cannot hold is refused rather than widened. See the block comment there.
    "datetime64":  (pd.date_range("2026-01-01", periods=4),           "datetime64[ns]"),
    "tz_aware":    (pd.date_range("2026-01-01", periods=4, tz="Europe/Berlin"),
                    "datetime64[ns, Europe/Berlin]"),
    "timedelta64": (pd.to_timedelta([1, 2, 3, 4], unit="D"),          "timedelta64[ns]"),
    "object":      (np.array(list("abcd"), dtype=object),             "object"),
    "category":    (pd.Categorical(list("abcd")),                     "category"),
    "Int64":       (pd.array([1, 2, 3, 4], dtype="Int64"),            "Int64"),
    "boolean":     (pd.array([True, False, True, False], dtype="boolean"), "boolean"),
    "string":      (pd.array(list("abcd"), dtype="string"),           "string"),
    "period":      (pd.period_range("2026-01", periods=4, freq="M"),  "period[M]"),
    # ---- correction round 6 (architect Decisions 2 and 4, 2026-07-28).
    # Every one of these answers `'f'` or `'i'` to `dtype.kind`, which is what
    # the routing predicate used to ask. `Float64` was therefore handed to the
    # `.to_numpy()` fast path and came back as **object** on a FULLY MATCHED
    # join, and both sparse dtypes were densified. Adding the rows is the
    # whole point of a generated matrix: four combinations each, for free.
    "Float64":     (pd.array([1., 2., 3., 4.], dtype="Float64"),      "Float64"),
    "Float32":     (pd.array([1., 2., 3., 4.], dtype="Float32"),      "Float32"),
    "sparse_float": (pd.arrays.SparseArray(np.array([1., 2., 3., 4.])),
                     "Sparse[float64, nan]"),
}
# The one dtype whose missing representation cannot be held without widening.
# The architect ruled it REFUSED rather than silently converted: a fully
# matched interval join still preserves its dtype.
DTYPE_REFUSED_ON_MISSING = {
    "interval": pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3, 4]),
    # Sparse[int64] cannot hold a gap without becoming Sparse[float64], and
    # before round 6 it did not even do that — `SparseDtype(np.int64).kind` is
    # `'i'`, so it took the plain-int contract branch and `.to_numpy()`
    # returned a DENSE float64 array: densified AND widened in one step, which
    # is the case Decision 4 names in as many words. Now refused, with the
    # error telling the user to configure a representable `fill_missing`.
    "sparse_int": pd.arrays.SparseArray(np.array([1, 2, 3, 4], dtype=np.int64)),
    # ---- ROUND 10, AD-19 RATIFIED (architect, 2026-07-29, Option 1 with an
    # operational definition). These four used to sit in DTYPE_CASES with an
    # expected WIDENED result (`int* -> float64`, `bool -> object`). The
    # ruling removed that:
    #
    #   "Every dtype observable from source metadata, an existing physical
    #    column, schema metadata, an explicit alias declaration, or the first
    #    successful creation/materialization is authoritative. ADF must
    #    preserve it thereafter. If a missing value cannot be represented in
    #    that dtype, ADF must use an explicitly configured compatible fill or
    #    refuse clearly."
    #
    # ADF does not need to know whether the user consciously chose `int8`;
    # the column has it, so it is authoritative. A missing key therefore
    # refuses unless a fill is configured — and the configured-fill path is
    # asserted separately (test_b32_185).
    "int64":  np.array([1, 2, 3, 4], dtype=np.int64),
    "int8":   np.array([1, 2, 3, 4], dtype=np.int8),
    "uint32": np.array([1, 2, 3, 4], dtype=np.uint32),
    "bool":   np.array([True, False, True, False]),
}


@needs_dfdraw
class TestB32StandingDtypeMatrix:
    """Every dtype × matched / missing / empty-child, through a real
    `draw_batch`. See the block comment above for why this is generated."""

    def _build(self, col, keys, fill=None, empty=False):
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(len(keys), dtype=float),
            "sec": np.asarray(keys, dtype=np.int64)}))
        if empty:
            child = A.AliasDataFrame(pd.DataFrame({
                "sec": pd.Series([], dtype="int64"),
                "v": pd.Series(col).iloc[:0]}))
        else:
            child = A.AliasDataFrame(pd.DataFrame({
                "sec": np.arange(4, dtype=np.int64), "v": col}))
        main.register_subframe("S", child, index_columns=["sec"])
        if fill is not None:
            main.set_subframe_fill("S", fill_missing=fill, fill_mode="direct")
        return main, child

    def _delegated(self, main, **kw):
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["df"] = self.df.copy()
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main.draw_batch({"p": {"expr": "x", "type": "hist", "bins": 3,
                                       "group_by": "S.v"}},
                                verbose=False, **kw)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        return seen["df"]["S_v"]

    @pytest.mark.parametrize("name", sorted(DTYPE_CASES))
    def test_b32_102_matched_preserves_the_exact_dtype(self, name):
        """Nothing is missing, so nothing may change. This is the case that
        regressed in round 3 and again in round 4 for a different set."""
        col, _ = DTYPE_CASES[name]
        main, child = self._build(col, [0, 1, 2, 3])
        got = self._delegated(main)
        assert str(got.dtype) == str(child.df["v"].dtype), name
        assert list(got.values) == list(child.df["v"].values), name

    @pytest.mark.parametrize("name", sorted(DTYPE_CASES))
    def test_b32_103_missing_uses_the_documented_representation(self, name):
        """`expected_missing_dtype` is the SECOND element of each case. For
        most dtypes it equals the source: the dtype owns a missing value. For
        int/uint/bool it does not, and the documented, ratified behaviour is
        that this layer yields NaN and the ALIAS layer restores the declared
        dtype through `_safe_dtype_cast`."""
        col, expected = DTYPE_CASES[name]
        main, _ = self._build(col, [0, 1, 9, 3])
        got = self._delegated(main)
        assert str(got.dtype) == expected, (
            f"{name}: expected {expected}, got {got.dtype}")
        assert pd.isna(got.values[2]), f"{name}: missing row is not missing"
        assert not pd.isna(got.values[0]), f"{name}: matched row became missing"

    @pytest.mark.parametrize("name", sorted(DTYPE_CASES))
    def test_b32_104_empty_child_is_all_missing_not_an_error(self, name):
        """An empty child is the extreme case of "every key missing". Round 4
        shipped a passing empty-child test that used a FLOAT column, which
        took a different code path; every other dtype raised an internal
        out-of-bounds error (GPT31)."""
        col, expected = DTYPE_CASES[name]
        main, _ = self._build(col, [0, 1], empty=True)
        got = self._delegated(main)
        assert len(got) == 2, name
        assert got.isna().all(), f"{name}: {list(got.values)}"

    @pytest.mark.parametrize("name", sorted(DTYPE_CASES))
    def test_b32_105_entry_selection_does_not_change_the_rule(self, name):
        col, expected = DTYPE_CASES[name]
        main, _ = self._build(col, [0, 1, 9, 3, 2, 0])
        got = self._delegated(main, entry_mask=np.array([4, 2, 0]))
        assert str(got.dtype) == expected, name
        assert pd.isna(got.values[1]), f"{name}: position 2 holds key 9"
        assert not pd.isna(got.values[0])

    @pytest.mark.parametrize("name", sorted(DTYPE_REFUSED_ON_MISSING))
    def test_b32_106_widening_dtypes_are_refused_not_converted(self, name):
        """Interval: a missing value changes the subtype
        (`interval[int64]` -> `interval[float64]`). The architect ruled this
        refused with a clean ADF error rather than silently widened, while a
        fully matched join still preserves the dtype."""
        col = DTYPE_REFUSED_ON_MISSING[name]
        main, child = self._build(col, [0, 1, 2, 3])
        got = self._delegated(main)
        assert str(got.dtype) == str(child.df["v"].dtype), (
            f"{name}: a fully matched join must preserve the dtype")
        main2, _ = self._build(col, [0, 1, 9, 3])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError):
                self._delegated(main2)


@needs_dfdraw
class TestB32PreIndexJoins:
    """GPT26-P0-3. `pre_index=True` sets the index with `drop=False`, so the
    join key is both an index level and a column and the merge fallback was
    ambiguous. A supported public registration option failed on every
    `draw_batch`."""

    def _pair(self, keys, multi=False, pre_index=True):
        n = len(keys)
        if multi:
            main = A.AliasDataFrame(pd.DataFrame({
                "k": np.asarray(keys, dtype=np.int64),
                "j": np.zeros(n, dtype=np.int64),
                "x": np.arange(n, dtype=float)}))
            child = A.AliasDataFrame(pd.DataFrame({
                "k": np.arange(4, dtype=np.int64),
                "j": np.zeros(4, dtype=np.int64),
                "v": np.arange(4, dtype=float) * 10}))
            cols = ["k", "j"]
        else:
            main = A.AliasDataFrame(pd.DataFrame({
                "k": np.asarray(keys, dtype=np.int64),
                "x": np.arange(n, dtype=float)}))
            child = A.AliasDataFrame(pd.DataFrame({
                "k": np.arange(4, dtype=np.int64),
                "v": np.arange(4, dtype=float) * 10}))
            cols = ["k"]
        main.register_subframe("S", child, index_columns=cols,
                               pre_index=pre_index)
        return main

    def _values(self, main, **kw):
        import dfextensions.dfdraw as _dfd
        seen = {}
        real = _dfd.DFDraw.draw_batch

        def _spy(self, *a, **k):
            seen["df"] = self.df.copy()
            return real(self, *a, **k)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                main.draw_batch({"p": {"expr": "S.v:x", "type": "scatter"}},
                                verbose=False, **kw)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")
        return list(seen["df"]["S_v"].values)

    def test_b32_107_single_key_all_matched(self):
        assert self._values(self._pair([0, 1, 2, 3])) == [0., 10., 20., 30.]

    def test_b32_108_single_key_one_missing(self):
        got = self._values(self._pair([0, 1, 9, 3]))
        assert got[:2] == [0., 10.] and np.isnan(got[2]) and got[3] == 30.

    def test_b32_109_multi_column_key(self):
        assert self._values(self._pair([0, 1, 2, 3], multi=True)) == \
            [0., 10., 20., 30.]

    def test_b32_110_with_entry_selection(self):
        assert self._values(self._pair([0, 1, 2, 3]),
                            entry_mask=np.array([3, 1])) == [30., 10.]

    def test_b32_111_pre_index_false_control(self):
        got = self._values(self._pair([0, 1, 9, 3], pre_index=False))
        assert got[:2] == [0., 10.] and np.isnan(got[2])


@needs_dfdraw
class TestB32GraphGuardSeparation:
    """GPT30-P0-2: one guard was doing two jobs.

    The ancestor-path check exists to TERMINATE recursion on a cycle. It was
    also deciding what the validator got to compare, so `root <- A(child)`
    then `child <- R(root)` passed: `root` is its own ancestor along `A::R`,
    the walk stopped, and the second owner path was never produced. The edges
    are now enumerated separately from the walk.
    """

    def _f(self, n=4):
        return A.AliasDataFrame(pd.DataFrame(
            {"k": np.arange(n, dtype=np.int32),
             "x": np.arange(n, dtype=float)}))

    def test_b32_112_back_edge_cycle_is_refused(self):
        root, child = self._f(), self._f()
        root.register_subframe("A", child, index_columns=["k"])
        child.register_subframe("R", root, index_columns=["k"])
        with pytest.raises(ValueError, match="two different owner paths"):
            root._validate_frame_graph_ownership()

    def test_b32_113_single_self_registration_still_legal(self):
        """The cycle contract still owns this one — the separation must not
        turn a legal shape into a refusal (test_N1_7_cycle_detection)."""
        frame = self._f()
        frame.register_subframe("Self", frame, index_columns=["k"])
        frame._validate_frame_graph_ownership()

    def test_b32_114_legal_shapes_all_still_pass(self):
        root = self._f()
        root.register_subframe("A", self._f(), index_columns=["k"])
        root.register_subframe("B", self._f(), index_columns=["k"])
        root._validate_frame_graph_ownership()
        r2, mid, leaf = self._f(), self._f(), self._f()
        mid.register_subframe("L", leaf, index_columns=["k"])
        r2.register_subframe("M", mid, index_columns=["k"])
        r2._validate_frame_graph_ownership()
        left, right, shared = self._f(), self._f(), self._f()
        left.register_subframe("S", shared, index_columns=["k"])
        right.register_subframe("S", shared, index_columns=["k"])
        left._validate_frame_graph_ownership()

    def test_b32_115_delayed_connection_still_refused(self):
        root, a, b, leaf = (self._f(), self._f(), self._f(), self._f())
        root.register_subframe("A", a, index_columns=["k"])
        root.register_subframe("B", b, index_columns=["k"])
        a.register_subframe("L1", leaf, index_columns=["k"])
        b.register_subframe("L2", leaf, index_columns=["k"])
        with pytest.raises(ValueError, match="two different owner paths"):
            root._validate_frame_graph_ownership()


@needs_dfdraw
class TestB32FailureRecordRound4:
    """The remaining boundaries, and the compound-failure ruling."""

    def _lazy(self):
        fixture = os.path.join(os.path.dirname(__file__),
                               "lazy_struct_fixture_clean.root")
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        adf = A.AliasDataFrame.read_tree_lazy(fixture, "tree")
        adf.add_alias("shift", "mult + 1")
        return adf

    def test_b32_116_dispatch_failure_is_bracketed(self):
        """The CRR for round 4 claimed this interval was bracketed. It was
        not — the normalization loop was bracketed and the claim was written
        as though that covered dispatch. A false statement in the record is
        worse than the gap it describes."""
        adf = self._lazy()
        cls = type(adf)
        real = cls._dict_dispatch_columns
        cls._dict_dispatch_columns = (
            lambda self, *a, **k: (_ for _ in ()).throw(
                RuntimeError("dispatch injected")))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError, match="dispatch injected"):
                    adf.draw_batch({"p": {"expr": "shift", "type": "hist",
                                          "bins": 3}},
                                   lazy=True, clear_after=True, verbose=False)
        finally:
            cls._dict_dispatch_columns = real
            plt.close("all")
        st = adf._last_draw_prep_state
        assert st.failure_phase == "dispatch", st.failure_phase
        assert "shift" in st.cleanup_candidates

    def test_b32_117_partial_cleanup_records_what_it_dropped(self):
        """GPT25: the drop delta was computed only after the drop RETURNED,
        so a cleanup that dropped one candidate and then raised recorded
        nothing — while the column was genuinely gone."""
        adf = self._lazy()
        adf.add_alias("second", "mult + 2")
        cls = type(adf)
        real = cls._dematerialize_qualified

        def _partial(self, names):
            ordered = sorted(names)
            real(self, [ordered[0]])
            raise RuntimeError("boom after partial")

        cls._dematerialize_qualified = _partial
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError, match="after partial"):
                    adf.draw_batch(
                        {"a": {"expr": "shift", "type": "hist", "bins": 3},
                         "b": {"expr": "second", "type": "hist", "bins": 3}},
                        lazy=True, clear_after=True, verbose=False)
        finally:
            cls._dematerialize_qualified = real
            plt.close("all")
        st = adf._last_draw_prep_state
        assert st.aliases_dropped, "a real drop happened and was not recorded"
        for dropped in st.aliases_dropped:
            assert dropped not in adf.df.columns, dropped

    def test_b32_118_cleanup_failure_does_not_replace_the_original(self):
        """Architect ruling, 2026-07-28: the original phase failure stays
        primary; the cleanup failure is secondary evidence, NOT chained as
        its cause — chaining would read as "cleanup caused the render
        failure", which is false."""
        import dfextensions.dfdraw as _dfd
        adf = self._lazy()
        cls = type(adf)
        real_dm = cls._dematerialize_qualified
        real_db = _dfd.DFDraw.draw_batch
        cls._dematerialize_qualified = (
            lambda self, n: (_ for _ in ()).throw(
                RuntimeError("cleanup during failure")))

        def _boom(self, *a, **k):
            raise ValueError("render original")

        _dfd.DFDraw.draw_batch = _boom
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(ValueError, match="render original"):
                    adf.draw_batch({"p": {"expr": "shift", "type": "hist",
                                          "bins": 3}},
                                   lazy=True, clear_after=True,
                                   clear_after_on_error=True, verbose=False)
        finally:
            cls._dematerialize_qualified = real_dm
            _dfd.DFDraw.draw_batch = real_db
            plt.close("all")
        st = adf._last_draw_prep_state
        assert st.failure_phase == "render", st.failure_phase
        assert st.cleanup_outcome == "failed", st.cleanup_outcome
        assert "cleanup during failure" in st.secondary_error, \
            st.secondary_error

    def test_b32_119_cleanup_only_failure_is_primary(self):
        """The inverse the ruling asks for separately: rendering succeeded, so
        the cleanup exception IS the failure and owns the phase."""
        adf = self._lazy()
        cls = type(adf)
        real = cls._dematerialize_qualified
        cls._dematerialize_qualified = (
            lambda self, n: (_ for _ in ()).throw(RuntimeError("cleanup only")))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(RuntimeError, match="cleanup only"):
                    adf.draw_batch({"p": {"expr": "shift", "type": "hist",
                                          "bins": 3}},
                                   lazy=True, clear_after=True, verbose=False)
        finally:
            cls._dematerialize_qualified = real
            plt.close("all")
        st = adf._last_draw_prep_state
        assert st.failure_phase == "cleanup", st.failure_phase
        assert st.cleanup_outcome == "failed", st.cleanup_outcome


# ============================================================================
# CORRECTION ROUND 6 — architect Decisions 1-4 and the symmetry requirement
# (2026-07-28, GPT31's decision set, approved by all GPT seats and Fable).
#
# Everything here was REPRODUCED on the round-5 bytes before it was fixed.
# ============================================================================


@needs_dfdraw
class TestB32Round6DtypeRouting:
    """Decision 2 / Decision 4 — the routing predicate.

    `dtype.kind` is defined on pandas ExtensionDtypes as well as on NumPy
    dtypes, and several of them answer `'f'` while behaving nothing like a
    NumPy float buffer:

        pd.Float64Dtype().kind           -> 'f'
        pd.SparseDtype(np.float64).kind  -> 'f'
        pd.SparseDtype(np.int64).kind    -> 'i'

    So the gather routed them to the `.to_numpy()` fast path or to the plain
    int contract. Measured on the round-5 bytes: `Float64` came back as
    **object** from a FULLY MATCHED join, `Sparse[float64]` was densified, and
    `Sparse[int64]` with one missing key came back as a dense `float64`.

    The generated matrix above now carries all three, which is the durable
    guard. These tests state the predicate itself, so the intent survives even
    if someone rewrites the matrix."""

    def test_b32_120_kind_is_not_a_numpy_float_test(self):
        """The premise, pinned. If a future pandas made these answer something
        else, the reason for `_is_plain_float_dtype` would have changed and the
        reader should be told here rather than guess."""
        assert pd.Float64Dtype().kind == "f"
        assert pd.SparseDtype(np.float64).kind == "f"
        assert pd.SparseDtype(np.int64).kind == "i"

    @pytest.mark.parametrize("dtype,expected", [
        (np.dtype("float64"), True),
        (np.dtype("float32"), True),
        (np.dtype("float16"), True),
        (np.dtype("int64"), False),
        (np.dtype("complex128"), False),
        (pd.Float64Dtype(), False),
        (pd.SparseDtype(np.float64), False),
        (pd.SparseDtype(np.int64), False),
        (pd.CategoricalDtype(list("ab")), False),
    ])
    def test_b32_121_plain_float_predicate(self, dtype, expected):
        assert A.AliasDataFrame._is_plain_float_dtype(dtype) is expected


@needs_dfdraw
class TestB32Round6FillRepresentability:
    """Decision 3 — a fill is accepted when it is compatible with the ACTUAL
    column dtype, a category is never silently added, and an incompatible fill
    raises a clear ADF error.

    The load-bearing case is `bool` + `fill_missing=0`: before round 6 it
    produced an **object** column holding `[True, False, 0, False]` — a literal
    integer mixed in among Booleans, which is exactly "silently change a
    Boolean column to object" (Decision 2). `fill_missing=False` on the same
    column worked. One knob, two dtypes of answer, decided by nothing the user
    could see."""

    def _project(self, col, keys, **fill):
        main = A.AliasDataFrame(pd.DataFrame({
            "kp": np.asarray(keys, dtype=np.int64),
            "x": np.arange(len(keys), dtype=float)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "kc": np.arange(4, dtype=np.int64)}))
        child.df["v"] = col
        main.register_subframe("C", child, index_columns=["kp"],
                               right_index_columns=["kc"])
        if fill:
            main.set_subframe_fill("C", **fill)
        idx, missing = main._compute_join_indices("C", ["kp"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.Series(
                main._extract_subframe_values_cached("C", "v", idx, missing))

    MATCHED = [0, 1, 2, 3]
    MISSING = [0, 1, 9, 3]

    def test_b32_122_bool_fill_zero_stays_boolean(self):
        got = self._project(np.array([True, False, True, False]),
                            self.MISSING, fill_missing=0)
        assert str(got.dtype) == "bool", got.dtype
        assert list(got.values) == [True, False, False, False]

    def test_b32_123_bool_fill_false_unchanged(self):
        got = self._project(np.array([True, False, True, False]),
                            self.MISSING, fill_missing=False)
        assert str(got.dtype) == "bool"
        assert list(got.values) == [True, False, False, False]

    @pytest.mark.parametrize("col,fill", [
        (np.array([True, False, True, False]), 2),
        (np.array([True, False, True, False]), "x"),
        (np.array([1, 2, 3, 4], dtype=np.int64), 1.5),
        (pd.Categorical([1, 2, 3, 4]), 9),
    ])
    def test_b32_124_unrepresentable_fill_is_refused(self, col, fill):
        """Refused, not stored. Each of these SUCCEEDS as a coercion and
        changes the value — `2 -> True`, `1.5 -> 1`, `9 -> NaN` — which is why
        a try/except around the cast would not have been enough. The rule is a
        round trip, not a conversion."""
        with pytest.raises(ValueError, match="does not survive"):
            self._project(col, self.MISSING, fill_missing=fill)

    def test_b32_125_category_fill_that_is_a_category_is_accepted(self):
        got = self._project(pd.Categorical([1, 2, 3, 4]), self.MISSING,
                            fill_missing=1)
        assert str(got.dtype) == "category"
        assert list(got.values) == [1, 2, 1, 4]

    def test_b32_126_string_fill_on_an_object_column(self):
        """Decision 3 removed the numeric-only restriction. This call raised
        `TypeError: fill_missing must be numeric, got str` before round 6."""
        got = self._project(np.array(list("abcd"), dtype=object),
                            self.MISSING, fill_missing="NA")
        assert list(got.values) == ["a", "b", "NA", "d"]

    def test_b32_127_sparse_int_missing_is_refused_not_densified(self):
        """Decision 4, stated as its own test because the matrix row only
        asserts that it raises — this one records WHAT it used to return."""
        with pytest.raises(ValueError, match="would change its dtype"):
            self._project(
                pd.arrays.SparseArray(np.array([1, 2, 3, 4], dtype=np.int64)),
                self.MISSING)

    def test_b32_128_sparse_int_with_a_representable_fill_is_preserved(self):
        """...and the escape hatch the error message names actually works."""
        got = self._project(
            pd.arrays.SparseArray(np.array([1, 2, 3, 4], dtype=np.int64)),
            self.MISSING, fill_missing=0)
        assert str(got.dtype) == "Sparse[int64, 0]", got.dtype
        assert list(got.values) == [1, 2, 0, 4]

    def test_b32_129_containers_are_still_rejected_at_configuration(self):
        main = A.AliasDataFrame(pd.DataFrame({"kp": [0, 1], "x": [1., 2.]}))
        child = A.AliasDataFrame(pd.DataFrame({"kp": [0, 1], "v": [1., 2.]}))
        main.register_subframe("C", child, index_columns=["kp"])
        for bad in ([1, 2], np.array([1, 2]), {"a": 1}):
            with pytest.raises(TypeError, match="scalar fill value"):
                main.set_subframe_fill("C", fill_missing=bad)


@needs_dfdraw
class TestB32Round6ComplexFillSymmetry:
    """GPT31 — `fill_nan` / `fill_inf` in `safe` mode were applied to real
    float columns and silently ignored for **complex** ones, although the
    public API accepts them identically. The control is the float call in the
    same test: whatever it does, complex must do."""

    def _project(self, col, **fill):
        main = A.AliasDataFrame(pd.DataFrame({
            "kp": np.arange(4, dtype=np.int64), "x": np.arange(4.)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "kc": np.arange(4, dtype=np.int64)}))
        child.df["v"] = col
        main.register_subframe("C", child, index_columns=["kp"],
                               right_index_columns=["kc"])
        main.set_subframe_fill("C", **fill)
        idx, missing = main._compute_join_indices("C", ["kp"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.Series(
                main._extract_subframe_values_cached("C", "v", idx, missing))

    def test_b32_130_complex_honours_fill_nan_and_fill_inf(self):
        got = self._project(np.array([1 + 1j, np.nan + 0j, np.inf + 0j, 4 + 0j]),
                            fill_mode="safe", fill_nan=99, fill_inf=77)
        assert str(got.dtype) == "complex128", got.dtype
        assert got.values[1] == 99 + 0j, got.values[1]
        assert got.values[2] == 77 + 0j, got.values[2]

    def test_b32_131_float_control_is_unchanged(self):
        got = self._project(np.array([1., np.nan, np.inf, 4.]),
                            fill_mode="safe", fill_nan=99, fill_inf=77)
        assert list(got.values) == [1., 99., 77., 4.]

    def test_b32_132_direct_mode_still_touches_only_missing_keys(self):
        """The guard moved into the shared helper, so this pins that `direct`
        did not quietly acquire NaN handling on the way."""
        got = self._project(np.array([1., np.nan, np.inf, 4.]),
                            fill_mode="direct", fill_nan=99, fill_inf=77)
        assert pd.isna(got.values[1])
        assert np.isinf(got.values[2])


@needs_dfdraw
class TestB32Round6IndexLevelJoinKeys:
    """A join key held as an INDEX LEVEL is the same key as one held in a
    column. `pre_index=True` keeps it as both (drop=False), which works; a
    child the USER indexed with the pandas default (`set_index('kc')`, which
    drops) died with a bare `KeyError: 'kc'` naming neither the subframe nor
    the side nor the remedy."""

    def _run(self, drop, pre_index, index_before):
        main = A.AliasDataFrame(pd.DataFrame({
            "kp": np.arange(4, dtype=np.int64), "x": np.arange(4.)}))
        cdf = pd.DataFrame({"kc": np.arange(4, dtype=np.int64),
                            "v": [10., 20., 30., 40.]})
        if index_before:
            cdf = cdf.set_index("kc", drop=drop)
        child = A.AliasDataFrame(cdf)
        main.register_subframe("C", child, index_columns=["kp"],
                               right_index_columns=["kc"],
                               pre_index=pre_index)
        if not index_before:
            child.df.set_index("kc", inplace=True, drop=drop)
        main.add_alias("j", "C.v")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main.materialize_aliases(names=["j"])
        return list(main.df["j"].values)

    @pytest.mark.parametrize("drop", [True, False])
    @pytest.mark.parametrize("pre_index", [True, False])
    @pytest.mark.parametrize("index_before", [True, False])
    def test_b32_133_all_index_shapes_join(self, drop, pre_index,
                                           index_before):
        assert self._run(drop, pre_index, index_before) == [10., 20., 30., 40.]

    def test_b32_134_a_key_that_is_truly_absent_still_names_itself(self):
        """Broadening the lookup must not broaden it into silence."""
        main = A.AliasDataFrame(pd.DataFrame({
            "kp": np.arange(4, dtype=np.int64), "x": np.arange(4.)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "other": np.arange(4, dtype=np.int64), "v": np.arange(4.)}))
        with pytest.raises(ValueError, match="right_index_columns not found"):
            main.register_subframe("C", child, index_columns=["kp"],
                                   right_index_columns=["kc"])


@needs_dfdraw
class TestB32Round6FacetBySlot:
    """Architect, 2026-07-28: "`facet_by` is an existing `draw_batch()`
    defect, not future work — fix it NOW."

    `_parse_expr_aliases` took five of the six scalar draw slots. A facet
    alias was therefore excluded from the ONE bulk `materialize_aliases()`
    call that `draw_batch` exists to make, and picked up afterwards, one at a
    time, by `_ensure_vector_kwargs_aliases`."""

    def test_b32_135_facet_alias_is_discovered_by_the_scanner(self):
        adf = A.AliasDataFrame(pd.DataFrame({
            "a": np.arange(20) % 4, "b": np.arange(20) * 1.0}))
        adf.add_alias("fa", "a*1")
        adf.add_alias("fb", "b*2")
        found = adf._parse_expr_aliases("fb:b", facet_by="fa")
        assert found == {"fa", "fb"}, found

    def test_b32_136_list_valued_facet_by(self):
        adf = A.AliasDataFrame(pd.DataFrame({
            "a": np.arange(8) % 4, "b": np.arange(8) * 1.0}))
        adf.add_alias("fa", "a*1")
        adf.add_alias("fc", "a+1")
        found = adf._parse_expr_aliases("b", facet_by=["fa", "fc"])
        assert found == {"fa", "fc"}, found

    @pytest.mark.parametrize("enum", ["group_by", "vector", "quantiles"])
    def test_b32_137_channel_enums_are_not_column_names(self, enum):
        """A dfdraw CHANNEL name is not a column reference. An alias that
        happens to be called `vector` must not be materialized because someone
        asked for the vector faceting channel."""
        adf = A.AliasDataFrame(pd.DataFrame({"b": np.arange(4) * 1.0}))
        adf.add_alias(enum, "b*3")
        assert adf._parse_expr_aliases("b", facet_by=enum) == set()

    def test_b32_138_facet_alias_reaches_the_single_bulk_call(self):
        """The behavioural half: the facet alias must be in the ONE batched
        materialization, not in a later per-alias pass."""
        adf = A.AliasDataFrame(pd.DataFrame({
            "a": np.arange(20) % 4, "b": np.arange(20) * 1.0}))
        adf.add_alias("fa", "a*1")
        adf.add_alias("fb", "b*2")
        calls = []
        cls = type(adf)
        real = cls.materialize_aliases

        def _spy(self, names=None, **k):
            calls.append(sorted(names) if names else [])
            return real(self, names=names, **k)

        cls.materialize_aliases = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf.draw_batch({"p": {"expr": "fb:b", "facet_by": "fa"}},
                               save_dir=None, lazy=True, verbose=False,
                               clear_after=False)
        finally:
            cls.materialize_aliases = real
            plt.close("all")
        assert calls, "materialize_aliases was never called"
        assert "fa" in calls[0], (
            f"facet alias missing from the bulk call: {calls}")


# ============================================================================
# CORRECTION ROUND 7 — the two acceptance matrices the round-6 panel demanded.
#
# GPT25, GPT27, GPT30 and GPT31 each EXECUTED and independently confirmed the
# same two defects; Sonet29's synthesis re-verified both against source and
# overturned its own [OK]. Four Claude-family seats read the round-6 diff
# carefully and missed both, because neither shape existed in any fixture.
#
# Both matrices below are GENERATED, for the same reason the dtype matrix is:
# the defects live in combinations nobody thought to write down, so coverage
# has to be a property of the table.
# ============================================================================

# Every dtype family that can hold NaN or Inf. Round 6 applied fill_nan /
# fill_inf to the first two and silently ignored them for the rest, because
# the gate asked `isinstance(dtype, np.dtype)` — the same storage-family
# assumption round 6 had just removed from the GATHER router, re-typed one
# helper over.
INVALID_FILL_CASES = {
    "float64":        (np.array([1., np.nan, np.inf, 4.]),                  "float64"),
    "float32":        (np.array([1., np.nan, np.inf, 4.], np.float32),      "float32"),
    "complex128":     (np.array([1+0j, np.nan+0j, np.inf+0j, 4+0j]),        "complex128"),
    "complex64":      (np.array([1+0j, np.nan+0j, np.inf+0j, 4+0j], np.complex64),
                                                                            "complex64"),
    "Float64":        (pd.array([1., None, np.inf, 4.], dtype="Float64"),   "Float64"),
    "Float32":        (pd.array([1., None, np.inf, 4.], dtype="Float32"),   "Float32"),
    "sparse_float64": (pd.arrays.SparseArray(np.array([1., np.nan, np.inf, 4.])),
                                                                            "Sparse[float64, nan]"),
    "sparse_float32": (pd.arrays.SparseArray(np.array([1., np.nan, np.inf, 4.], np.float32)),
                                                                            "Sparse[float32, nan]"),
}


@needs_dfdraw
class TestB32Round7InvalidValueFillMatrix:
    """`fill_nan` / `fill_inf` / `fill_invalid` × every NaN-capable dtype
    family × safe/direct × compatible/incompatible fill.

    The control is always the plain `float64` row: whatever it does, every
    other family must do. Round 6 had float64 applying both knobs and
    `Float64` / `Sparse[float64]` silently ignoring them — one public API,
    two answers, decided by storage internals the user never chose."""

    def _project(self, col, **fill):
        main = A.AliasDataFrame(pd.DataFrame({
            "k": np.arange(4, dtype=np.int64), "x": np.arange(4.)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "k": np.arange(4, dtype=np.int64)}))
        child.df["v"] = col
        main.register_subframe("S", child, index_columns=["k"])
        if fill:
            main.set_subframe_fill("S", **fill)
        idx, missing = main._compute_join_indices("S", ["k"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.Series(
                main._extract_subframe_values_cached("S", "v", idx, missing))

    @pytest.mark.parametrize("name", sorted(INVALID_FILL_CASES))
    def test_b32_139_safe_mode_applies_both_knobs(self, name):
        col, expected = INVALID_FILL_CASES[name]
        got = self._project(col, fill_mode="safe", fill_nan=99, fill_inf=77)
        assert str(got.dtype) == expected, f"{name}: dtype changed"
        vals = np.asarray(pd.Series(got).to_numpy(
            dtype="complex128" if "complex" in name else "float64",
            na_value=np.nan))
        assert vals[1] == 99, f"{name}: fill_nan ignored -> {vals[1]}"
        assert vals[2] == 77, f"{name}: fill_inf ignored -> {vals[2]}"

    @pytest.mark.parametrize("name", sorted(INVALID_FILL_CASES))
    def test_b32_140_fill_nan_alone_leaves_inf(self, name):
        col, expected = INVALID_FILL_CASES[name]
        got = self._project(col, fill_mode="safe", fill_nan=99)
        assert str(got.dtype) == expected, name
        vals = pd.Series(got).to_numpy(
            dtype="complex128" if "complex" in name else "float64",
            na_value=np.nan)
        assert vals[1] == 99, name
        assert np.isinf(vals[2]), f"{name}: fill_inf was not configured"

    @pytest.mark.parametrize("name", sorted(INVALID_FILL_CASES))
    def test_b32_141_fill_invalid_expands_to_both(self, name):
        """`fill_invalid` is documented as a shortcut for both knobs. If the
        expansion happened anywhere other than `_get_fill_config`, this row
        would diverge from `test_b32_139` for some family."""
        col, expected = INVALID_FILL_CASES[name]
        got = self._project(col, fill_mode="safe", fill_invalid=-5)
        assert str(got.dtype) == expected, name
        vals = pd.Series(got).to_numpy(
            dtype="complex128" if "complex" in name else "float64",
            na_value=np.nan)
        assert vals[1] == -5 and vals[2] == -5, f"{name}: {list(vals)}"

    @pytest.mark.parametrize("name", sorted(INVALID_FILL_CASES))
    def test_b32_142_direct_mode_touches_neither(self, name):
        """`direct` mode is documented to fill missing keys only. The round-7
        guard moved into the shared helper; this proves it did not quietly
        acquire NaN handling for one storage family on the way."""
        col, expected = INVALID_FILL_CASES[name]
        got = self._project(col, fill_mode="direct", fill_nan=99, fill_inf=77)
        assert str(got.dtype) == expected, name
        vals = pd.Series(got).to_numpy(
            dtype="complex128" if "complex" in name else "float64",
            na_value=np.nan)
        assert np.isnan(vals[1].real), f"{name}: direct mode filled a NaN"
        assert np.isinf(vals[2].real), f"{name}: direct mode filled an Inf"

    @pytest.mark.parametrize("name", sorted(INVALID_FILL_CASES))
    @pytest.mark.parametrize("knob", ["fill_nan", "fill_inf"])
    def test_b32_143_incompatible_invalid_fill_is_refused(self, name, knob):
        """Round 6: `float64` + `fill_nan="BAD"` returned an **object** column
        holding the literal string, with only a pandas FutureWarning. That is
        the silent dtype change AD-13/AD-14 forbid, produced by the round that
        introduced the rule — because `_coerce_fill_to_dtype` was called from
        exactly one site."""
        col, _ = INVALID_FILL_CASES[name]
        with pytest.raises(ValueError):
            self._project(col, fill_mode="safe", **{knob: "BAD"})


@needs_dfdraw
class TestB32Round7FillMissingOnTheFastPath:
    """`fill_missing` on the PLAIN-FLOAT fast path never reached
    `_coerce_fill_to_dtype` either — a separate bypass from the invalid-value
    one, found by GPT30."""

    def _project(self, col, **fill):
        main = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 1, 9, 3], dtype=np.int64), "x": np.arange(4.)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "k": np.arange(4, dtype=np.int64)}))
        child.df["v"] = col
        main.register_subframe("S", child, index_columns=["k"])
        if fill:
            main.set_subframe_fill("S", **fill)
        idx, missing = main._compute_join_indices("S", ["k"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pd.Series(
                main._extract_subframe_values_cached("S", "v", idx, missing))

    def test_b32_144_decimal_fill_preserves_float_dtype(self):
        """Round 6 returned an `object` column here, with a pandas
        incompatibility warning. A `Decimal` IS representable as float64 and
        survives the round trip, so it is accepted and the dtype is kept."""
        from decimal import Decimal
        got = self._project(np.array([1., 2., 3., 4.]),
                            fill_missing=Decimal("1.25"))
        assert str(got.dtype) == "float64", got.dtype
        assert got.values[2] == 1.25

    def test_b32_145_lossy_fill_on_the_fast_path_is_refused(self):
        with pytest.raises(ValueError, match="does not survive|cannot be stored"):
            self._project(np.array([1, 2, 3, 4], dtype=np.int64),
                          fill_missing=1.5)

    def test_b32_146_numpy_scalar_fill_is_accepted(self):
        """CRR round-6 §10.1 asked whether the new refusals reject something a
        physicist would reasonably configure. A NumPy scalar is the obvious
        candidate and must pass."""
        got = self._project(np.array([1., 2., 3., 4.], dtype=np.float32),
                            fill_missing=np.float32(7.5))
        assert str(got.dtype) == "float32", got.dtype
        assert got.values[2] == np.float32(7.5)

    def test_b32_147_zero_dim_array_is_a_scalar_fill(self):
        """GPT25 P2-3: `np.array(1.0)` is scalar-shaped and was refused by the
        blanket container check before the dimensionality test ran."""
        main = A.AliasDataFrame(pd.DataFrame({"k": [0, 1], "x": [0., 1.]}))
        child = A.AliasDataFrame(pd.DataFrame({"k": [0, 1], "v": [1., 2.]}))
        main.register_subframe("S", child, index_columns=["k"])
        main.set_subframe_fill("S", fill_missing=np.array(1.0))
        assert main._get_fill_config("S")["fill_missing"] == 1.0


@needs_dfdraw
class TestB32Round7JoinKeyRepresentationMatrix:
    """column-only / index-only / both-equal / both-DIFFERENT, on both sides.

    The last column of that table is the defect, and it is a REGRESSION
    introduced by this phase rather than a pre-existing gap. On the pre-phase
    baseline pandas itself refused the shape:

        ValueError: 'k' is both an index level and a column label,
                    which is ambiguous.

    The round-4 ambiguity normalization — added to make `pre_index=True` work,
    where the two spellings ALWAYS agree — rebuilt both key tables from column
    values and so removed that refusal for the case where they do not. Round 6
    then wrote AD-17 asserting the two spellings are one key, without ever
    checking it. Four reviewers executed the result independently; GPT27's
    reproduction returned silently REVERSED values."""

    def _child(self, shape):
        if shape == "column_only":
            return pd.DataFrame({"k": [0, 1, 2], "v": [10., 20., 30.]})
        if shape == "index_only":
            return pd.DataFrame({"k": [0, 1, 2],
                                 "v": [10., 20., 30.]}).set_index("k")
        df = pd.DataFrame({"k": [0, 1, 2], "v": [10., 20., 30.]})
        if shape == "both_equal":
            df.index = pd.Index([0, 1, 2], name="k")
            return df
        if shape == "both_reversed":
            df = pd.DataFrame({"k": [2, 1, 0], "v": [10., 20., 30.]})
            df.index = pd.Index([0, 1, 2], name="k")
            return df
        if shape == "both_disjoint":
            df.index = pd.Index([100, 101, 102], name="k")
            return df
        raise AssertionError(shape)

    def _join(self, cdf, pre_index=False):
        main = A.AliasDataFrame(pd.DataFrame({
            "k": [0, 1, 2], "x": [0., 1., 2.]}))
        main.register_subframe("S", A.AliasDataFrame(cdf),
                               index_columns=["k"], pre_index=pre_index)
        main.add_alias("j", "S.v")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main.materialize_aliases(names=["j"])
        return list(main.df["j"].values)

    @pytest.mark.parametrize("shape", ["column_only", "index_only",
                                       "both_equal"])
    @pytest.mark.parametrize("pre_index", [False, True])
    def test_b32_148_unambiguous_shapes_join(self, shape, pre_index):
        assert self._join(self._child(shape), pre_index) == [10., 20., 30.]

    @pytest.mark.parametrize("shape", ["both_reversed", "both_disjoint"])
    @pytest.mark.parametrize("pre_index", [False, True])
    def test_b32_149_conflicting_shapes_are_refused(self, shape, pre_index):
        with pytest.raises(ValueError, match="BOTH as a column and as an "
                                             "index level"):
            self._join(self._child(shape), pre_index)

    def test_b32_150_refusal_happens_at_registration(self):
        """Before any effect. GPT31: refuse at registration and again at graph
        consumption, so the error arrives at the line the user can see."""
        main = A.AliasDataFrame(pd.DataFrame({"k": [0, 1, 2],
                                              "x": [0., 1., 2.]}))
        with pytest.raises(ValueError, match="BOTH as a column"):
            main.register_subframe(
                "S", A.AliasDataFrame(self._child("both_reversed")),
                index_columns=["k"])
        assert "S" not in main._subframes.subframes

    def test_b32_151_parent_side_conflict_is_refused_too(self):
        """The check is symmetric. A conflicting key on the PARENT is exactly
        as wrong as one on the child, and round 6 read neither."""
        pdf = pd.DataFrame({"k": [2, 1, 0], "x": [0., 1., 2.]})
        pdf.index = pd.Index([0, 1, 2], name="k")
        main = A.AliasDataFrame(pdf)
        with pytest.raises(ValueError, match="parent side"):
            main.register_subframe(
                "S", A.AliasDataFrame(pd.DataFrame({"k": [0, 1, 2],
                                                    "v": [10., 20., 30.]})),
                index_columns=["k"])

    def test_b32_152_partial_multiindex_key(self):
        """GPT31 P1. A child carrying a MultiIndex `('a','b')` joined on `a`
        alone raised a bare `KeyError: "None of ['a'] are in the columns"`,
        because the old test compared the WHOLE index-name list against the
        requested keys instead of asking, per key, whether it is reachable."""
        cdf = pd.DataFrame({"a": [0, 0, 1, 1], "b": [10, 11, 10, 11],
                            "v": [1., 2., 3., 4.]}).set_index(["a", "b"],
                                                              drop=True)
        main = A.AliasDataFrame(pd.DataFrame({"ka": [0, 1], "x": [0., 1.]}))
        main.register_subframe("S", A.AliasDataFrame(cdf),
                               index_columns=["ka"],
                               right_index_columns=["a"], pre_index=True)
        main.add_alias("j", "S.v")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main.materialize_aliases(names=["j"])
        assert list(main.df["j"].values) == [1., 3.]

    def test_b32_153_absent_key_still_names_itself(self):
        """Broadening key lookup must not broaden it into silence."""
        main = A.AliasDataFrame(pd.DataFrame({"k": [0, 1], "x": [0., 1.]}))
        with pytest.raises(ValueError, match="right_index_columns not found"):
            main.register_subframe(
                "S", A.AliasDataFrame(pd.DataFrame({"other": [0, 1],
                                                    "v": [1., 2.]})),
                index_columns=["k"], right_index_columns=["kc"])


def _delegated_frame(main, spec_slot="group_by", ref="S.v", **kw):
    """Run a real `draw_batch()` and return the frame handed to dfdraw."""
    import dfextensions.dfdraw as _dfd
    seen, real = {}, _dfd.DFDraw.draw_batch

    def _spy(self, *a, **k):
        seen["df"] = self.df.copy()
        return real(self, *a, **k)

    _dfd.DFDraw.draw_batch = _spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main.draw_batch({"p": {"expr": "x", "type": "hist", "bins": 3,
                                   spec_slot: ref}}, verbose=False, **kw)
    finally:
        _dfd.DFDraw.draw_batch = real
        plt.close("all")
    return seen["df"][ref.replace(".", "_")]


@needs_dfdraw
class TestB32Round8DirectSlotWideningNoticeRemoved:
    """AD-13a is SUPERSEDED by AD-19 (architect, 2026-07-29): "widen now,
    emit a FutureWarning, make it an error later — was not my decision and
    contradicts AD-19."

    The notice mechanism is deleted, not disabled. GPT30 and the Main
    Reviewer both flagged that leaving a widen-and-warn function callable
    after a ruling saying "never widen-and-warn" is a contradiction inside
    the round's own record. This class replaces the five tests that pinned
    the notice, and asserts the absence rather than leaving a hole."""

    def test_b32_154_the_notice_function_no_longer_exists(self):
        assert not hasattr(A.AliasDataFrame, "_warn_direct_slot_widening")

    def test_b32_155_direct_slot_refuses_instead_of_warning(self):
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(4.), "sec": np.array([0, 1, 9, 3], dtype=np.int64)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "sec": np.arange(4, dtype=np.int64),
            "det": np.array([1, 2, 3, 4], dtype=np.int64)}))
        main.register_subframe("S", child, index_columns=["sec"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ValueError, match="authoritative dtype"):
                _delegated_frame(main, ref="S.det")
        assert not [w for w in caught if issubclass(w.category, FutureWarning)]


@needs_dfdraw
class TestB32Round8PublicFillMatrix:
    """The round-7 invalid-value fill matrix, re-proved through `draw_batch()`.

    Same eight dtype families, same knobs, same control row — but the
    assertion is now on the frame dfdraw receives, which is what the user
    actually gets."""

    def _build(self, col, keys, **fill):
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(len(keys), dtype=float),
            "k": np.asarray(keys, dtype=np.int64)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "k": np.arange(4, dtype=np.int64)}))
        child.df["v"] = col
        main.register_subframe("S", child, index_columns=["k"])
        if fill:
            main.set_subframe_fill("S", **fill)
        return main

    @staticmethod
    def _numeric(series, name):
        return series.to_numpy(
            dtype="complex128" if "complex" in name else "float64",
            na_value=np.nan)

    @pytest.mark.parametrize("name", sorted(INVALID_FILL_CASES))
    def test_b32_159_public_safe_mode_applies_both_knobs(self, name):
        col, expected = INVALID_FILL_CASES[name]
        got = _delegated_frame(self._build(col, [0, 1, 2, 3],
                                           fill_mode="safe",
                                           fill_nan=99, fill_inf=77))
        assert str(got.dtype) == expected, f"{name}: {got.dtype}"
        vals = self._numeric(got, name)
        assert vals[1] == 99 and vals[2] == 77, f"{name}: {list(vals)}"

    @pytest.mark.parametrize("name", sorted(INVALID_FILL_CASES))
    def test_b32_160_public_direct_mode_touches_neither(self, name):
        col, expected = INVALID_FILL_CASES[name]
        got = _delegated_frame(self._build(col, [0, 1, 2, 3],
                                           fill_mode="direct",
                                           fill_nan=99, fill_inf=77))
        assert str(got.dtype) == expected, name
        vals = self._numeric(got, name)
        assert np.isnan(vals[1].real) and np.isinf(vals[2].real), name

    @pytest.mark.parametrize("name", sorted(INVALID_FILL_CASES))
    def test_b32_161_public_incompatible_fill_is_refused(self, name):
        col, _ = INVALID_FILL_CASES[name]
        with pytest.raises(ValueError):
            _delegated_frame(self._build(col, [0, 1, 2, 3],
                                         fill_mode="safe", fill_nan="BAD"))

    def test_b32_162_public_decimal_fill_keeps_float_dtype(self):
        from decimal import Decimal
        got = _delegated_frame(self._build(np.array([1., 2., 3., 4.]),
                                           [0, 1, 9, 3],
                                           fill_missing=Decimal("1.25")))
        assert str(got.dtype) == "float64", got.dtype
        assert got.values[2] == 1.25


@needs_dfdraw
class TestB32Round8PublicKeyMatrix:
    """The round-7 join-key matrix, re-proved through `draw_batch()` rather
    than `materialize_aliases()`."""

    def _child(self, shape):
        df = pd.DataFrame({"k": [0, 1, 2], "v": [10., 20., 30.]})
        if shape == "column_only":
            return df
        if shape == "index_only":
            return df.set_index("k")
        if shape == "both_equal":
            df.index = pd.Index([0, 1, 2], name="k")
            return df
        if shape == "both_reversed":
            df = pd.DataFrame({"k": [2, 1, 0], "v": [10., 20., 30.]})
            df.index = pd.Index([0, 1, 2], name="k")
            return df
        raise AssertionError(shape)

    def _main(self, shape, pre_index=False):
        main = A.AliasDataFrame(pd.DataFrame({
            "x": [0., 1., 2.], "k": np.array([0, 1, 2], dtype=np.int64)}))
        main.register_subframe("S", A.AliasDataFrame(self._child(shape)),
                               index_columns=["k"], pre_index=pre_index)
        return main

    @pytest.mark.parametrize("shape", ["column_only", "index_only",
                                       "both_equal"])
    @pytest.mark.parametrize("pre_index", [False, True])
    def test_b32_163_public_unambiguous_shapes_project(self, shape,
                                                       pre_index):
        got = _delegated_frame(self._main(shape, pre_index))
        assert list(got.values) == [10., 20., 30.]

    @pytest.mark.parametrize("pre_index", [False, True])
    def test_b32_164_public_conflicting_shape_is_refused(self, pre_index):
        """Refused at registration, so `draw_batch` is never reached — which
        is the point: the user learns at the line they wrote."""
        with pytest.raises(ValueError, match="BOTH as a column"):
            self._main("both_reversed", pre_index)


@needs_dfdraw
class TestB32Round8NullableKeyEquality:
    """4/4 finding — GPT25 P1-1, GPT26 P0-2, GPT27 P1-1, GPT31 P1-1.

    `_key_arrays_equal` compared with `a == b` and then reduced the result to
    a bool. For any nullable or object array containing `pd.NA`, `a == b`
    yields `pd.NA` at those positions and the reduction raised

        TypeError: boolean value of NA is ambiguous

    out of a public `register_subframe()` — the raw pandas error AD-17
    promised not to produce. Reproduced for `object`, `string`, `boolean` AND
    `Int64`; only the plain float `NaN` control survived, which is why the
    round-7 matrix passed."""

    def _register(self, col_vals, idx_vals, dtype):
        cdf = pd.DataFrame({"k": pd.array(col_vals, dtype=dtype),
                            "v": [10., 20., 30.]})
        cdf.index = pd.Index(pd.array(idx_vals, dtype=dtype), name="k")
        main = A.AliasDataFrame(pd.DataFrame({
            "k": pd.array(col_vals, dtype=dtype), "x": [0., 1., 2.]}))
        main.register_subframe("S", A.AliasDataFrame(cdf),
                               index_columns=["k"])
        return main

    EQUAL_WITH_MISSING = {
        "object_NA":   (["a", pd.NA, "c"], "object"),
        "object_None": (["a", None, "c"], "object"),
        "string_NA":   (["a", pd.NA, "c"], "string"),
        "boolean_NA":  ([True, pd.NA, False], "boolean"),
        "Int64_NA":    ([1, pd.NA, 3], "Int64"),
        "float_NaN":   ([1., np.nan, 3.], "float64"),
        "datetime_NaT": ([pd.Timestamp("2026-01-01"), pd.NaT,
                          pd.Timestamp("2026-01-03")], "datetime64[ns]"),
    }

    @pytest.mark.parametrize("name", sorted(EQUAL_WITH_MISSING))
    def test_b32_165_equal_representations_with_missing_are_accepted(self,
                                                                     name):
        vals, dtype = self.EQUAL_WITH_MISSING[name]
        main = self._register(vals, vals, dtype)
        assert "S" in main._subframes.subframes

    def test_b32_166_missing_spelling_is_not_significant(self):
        """Decission KEY-MISSING-SPELLING, Option 1. `None` and `np.nan` in an
        object key are the same gap: a missing key matches no child row under
        either spelling, and pandas normalises between them on a `set_index()`
        round trip without the user asking."""
        main = self._register(["a", None, "c"], ["a", np.nan, "c"], "object")
        assert "S" in main._subframes.subframes

    @pytest.mark.parametrize("col,idx,dtype", [
        (["a", pd.NA, "c"], ["a", "b", "c"], "object"),
        (["a", "b", "c"], ["a", "x", "c"], "object"),
        ([1, pd.NA, 3], [1, 2, 3], "Int64"),
        ([1, 2, 3], [1, 9, 3], "Int64"),
    ])
    def test_b32_167_one_sided_missing_or_different_values_refuse(
            self, col, idx, dtype):
        """And the refusal is an ADF `ValueError`, never a raw pandas
        `TypeError` — the class of the error is part of the contract."""
        with pytest.raises(ValueError, match="BOTH as a column"):
            self._register(col, idx, dtype)


@needs_dfdraw
class TestB32Round8RegistrationLeavesNoPartialState:
    """GPT31 B32F7-P1-2 — and this one is a falsehood in the round-7 record,
    not a coverage gap.

    The CRR said refusal happens "before any effect" and that a refused
    registration "leaves no partial state". The round-7 test asserted only
    that the registry entry was absent. Measured: the child's `_schema` had
    already been auto-populated and the parent's join-index cache had already
    been invalidated before the raise.

    Every check now runs above the first mutation, and this test snapshots
    everything the round-7 one did not."""

    def _conflicting_child(self):
        cdf = pd.DataFrame({"k": [2, 1, 0], "v": [10., 20., 30.]})
        cdf.index = pd.Index([0, 1, 2], name="k")
        return A.AliasDataFrame(cdf)

    def _absent_key_child(self):
        return A.AliasDataFrame(pd.DataFrame({"other": [0, 1, 2],
                                              "v": [10., 20., 30.]}))

    @pytest.mark.parametrize("pre_index", [False, True])
    @pytest.mark.parametrize("kind", ["conflicting", "absent_key"])
    def test_b32_168_no_state_survives_a_refused_registration(self, kind,
                                                              pre_index):
        import copy as _copy
        main = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 1, 2], dtype=np.int64), "x": [0., 1., 2.]}))
        child = (self._conflicting_child() if kind == "conflicting"
                 else self._absent_key_child())
        # make the effects visible: a live cache entry and an empty schema
        main._join_index_cache["S"] = "sentinel"
        before = {
            "child_schema": _copy.deepcopy(child._schema),
            "parent_schema": _copy.deepcopy(main._schema),
            "cache": dict(main._join_index_cache),
            "child_index": list(child.df.index),
            "child_cols": list(child.df.columns),
        }
        with pytest.raises(ValueError):
            main.register_subframe("S", child, index_columns=["k"],
                                   pre_index=pre_index)
        assert "S" not in main._subframes.subframes
        assert "S" not in main._schema.get("subframes", {})
        assert child._schema == before["child_schema"], "child schema mutated"
        assert main._schema == before["parent_schema"], "parent schema mutated"
        assert dict(main._join_index_cache) == before["cache"], \
            "parent join cache was invalidated by a refused registration"
        assert list(child.df.index) == before["child_index"], \
            "pre_index mutated the child before the refusal"
        assert list(child.df.columns) == before["child_cols"]

    def test_b32_169_symmetric_absent_key_is_refused(self):
        """GPT25 B32F7-P1-2. The key-existence check lived inside the
        `right_index_columns is not None` branch since PHASE_13_65, so the
        ORDINARY symmetric call — the one every existing script makes — never
        ran it, and a child without the key registered successfully."""
        main = A.AliasDataFrame(pd.DataFrame({"k": [0, 1], "x": [0., 1.]}))
        with pytest.raises(ValueError, match="right_index_columns not found"):
            main.register_subframe("S", self._absent_key_child(),
                                   index_columns=["k"])


@needs_dfdraw
class TestB32Round8SparseDtypePortability:
    """GPT26 FIX7-P0-1, found by executing on pandas 2.2.3 — a runtime no seat
    had used before.

        pandas 1.5.3   Sparse[float32].take(...) -> Sparse[float32, nan]
        pandas 3.0.2   Sparse[float32].take(...) -> Sparse[float64, nan]

    and on the newer pandas it widens even for FULLY MATCHED positions. So
    round 7's exact-sparse-preservation claim held only on the coder's and the
    architect's pandas, and the generated matrix that "proved" it fails
    elsewhere. Trusting a pandas primitive to preserve a dtype is not
    portable; the result is normalized back to the verified source dtype."""

    @pytest.mark.parametrize("fill_value", [np.nan, 0.0])
    @pytest.mark.parametrize("subtype", [np.float32, np.float64])
    @pytest.mark.parametrize("keys", [[0, 1, 2, 3], [0, 1, 9, 3]])
    def test_b32_170_sparse_subtype_survives_the_gather(self, subtype,
                                                        fill_value, keys):
        col = pd.arrays.SparseArray(
            np.array([1., 2., 3., 4.], dtype=subtype), fill_value=fill_value)
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(4.), "k": np.asarray(keys, dtype=np.int64)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "k": np.arange(4, dtype=np.int64)}))
        child.df["v"] = col
        main.register_subframe("S", child, index_columns=["k"])
        if keys[2] == 9 and not np.isnan(fill_value):
            # a gap must be representable; NaN is not a Sparse[.., 0.0] value
            main.set_subframe_fill("S", fill_missing=fill_value)
        got = _delegated_frame(main)
        assert str(got.dtype) == str(col.dtype), (
            f"{col.dtype} -> {got.dtype} (pandas {pd.__version__})")

    def test_b32_171_restoration_refuses_a_lossy_cast(self):
        """The restoration must not become a silent downcast. If putting the
        gathered result back into the source dtype would change a value, the
        refusal path stands."""
        arr = pd.arrays.SparseArray(np.array([1., 2., 3., 4.]))
        s = pd.Series(arr)
        assert A.AliasDataFrame._restore_exact_dtype(
            s, pd.SparseDtype(np.float64)) is not None
        assert A.AliasDataFrame._restore_exact_dtype(
            pd.Series([1.5, 2.5]), np.dtype("int64")) is None


# ============================================================================
# CORRECTION ROUND 9 — AD-19 RATIFIED (architect, 2026-07-29).
#
#   "A missing-key operation may not change the explicitly supplied dtype and
#    may never change any non-missing value. If the dtype cannot represent the
#    gap, ADF must require an explicit compatible fill or refuse clearly."
#
# Round 7: GPT25/26/27/31 [X]. Round 8: GPT27/GPT30/GPT31 [X], Fabble5_7 [OK].
# ============================================================================


@needs_dfdraw
class TestB32Round9MatchedValuesNeverChange:
    """AD-19 clause 1, and the most serious defect of the phase — GPT31
    B32F8-P0-1.

    The April contract represents a gap by widening `int64` to `float64`.
    Above 2**53 that is lossy, so a SINGLE missing join key made three
    distinct matched measurements collapse into one:

        source          ...977, ...979, ...981, ...983    int64
        one key missing  1.152921504606847e+18 x3, NaN

    and a declared alias then cast them back to `int64` — type-correct,
    value-wrong, which is the worst failure mode in this system.

    Why eight rounds of dtype matrices never caught it: every integer in them
    is small and therefore exactly representable as `float64`. The table was
    structurally incapable of failing, exactly like the round-3 fixture where
    every key matched.

    These are not exotic values in this domain — a track/timeframe uid, a
    nanosecond timestamp, a bunch-crossing id. One missing key silently
    merged distinct tracks."""

    LOSSY = {
        "int64_2_53":  np.array([2**53 + 1, 2**53 + 3, 2**53 + 5, 2**53 + 7],
                                dtype=np.int64),
        "int64_2_60":  np.array([2**60 + 1, 2**60 + 3, 2**60 + 5, 2**60 + 7],
                                dtype=np.int64),
        "uint64_2_63": np.array([2**63 + 1, 2**63 + 3, 2**63 + 5, 2**63 + 7],
                                dtype=np.uint64),
    }
    EXACT = {
        "int32": np.array([1, 2, 3, 4], dtype=np.int32),
        "int64_small": np.array([1, 2, 3, 4], dtype=np.int64),
        "bool": np.array([True, False, True, False]),
    }

    def _main(self, col, keys, fill=None):
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(len(keys), dtype=float),
            "k": np.asarray(keys, dtype=np.int64)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "k": np.arange(4, dtype=np.int64)}))
        child.df["v"] = col
        main.register_subframe("S", child, index_columns=["k"])
        if fill is not None:
            main.set_subframe_fill("S", fill_missing=fill)
        return main

    @pytest.mark.parametrize("name", sorted(LOSSY))
    def test_b32_172_matched_join_is_exact_for_large_integers(self, name):
        """Nothing is missing, so nothing may change — including here."""
        col = self.LOSSY[name]
        got = _delegated_frame(self._main(col, [0, 1, 2, 3]))
        assert str(got.dtype) == str(col.dtype), name
        assert [int(v) for v in got.values] == [int(v) for v in col], name

    @pytest.mark.parametrize("name", sorted(LOSSY))
    def test_b32_173_missing_key_is_refused_not_widened(self, name):
        """Refused, NOT warned. A FutureWarning about a dtype does not make a
        wrong measurement acceptable."""
        col = self.LOSSY[name]
        with pytest.raises(ValueError, match="authoritative dtype"):
            _delegated_frame(self._main(col, [0, 1, 9, 3]))

    @pytest.mark.parametrize("name", sorted(LOSSY))
    def test_b32_174_configured_fill_preserves_dtype_and_values(self, name):
        """The remedy the error names must work, and must be EXACT — this is
        the architect's fill-plus-flag design: the value column keeps its
        dtype, the gap carries the physical neutral value, the flag column
        (the user's own) records that the measurement was absent."""
        col = self.LOSSY[name]
        got = _delegated_frame(self._main(col, [0, 1, 9, 3], fill=0))
        assert str(got.dtype) == str(col.dtype), name
        assert [int(v) for v in got.values] == [int(col[0]), int(col[1]), 0,
                                                int(col[3])], name

    @pytest.mark.parametrize("name", sorted(EXACT))
    def test_b32_175_small_values_are_refused_too_under_ad19(self, name):
        """SUPERSEDED AND INVERTED in round 10 (architect, AD-19 RATIFIED).

        Round 9 asserted that small integers "keep the April behaviour" —
        widening to float64 whenever the widening happened to be lossless. The
        architect removed that exception:

            "Every dtype observable from source metadata, an existing physical
             column, ... is authoritative. ADF must preserve it thereafter."

        Losslessness is no longer the test. The dtype changing at all is what
        is forbidden, because a downstream operation that expects `int8` gets
        `float64` regardless of whether the numbers survived. GPT27, GPT30 and
        GPT31 each filed the round-9 exception as blocking, independently.

        `test_b32_185` proves the configured-fill path this leaves as the
        remedy.
        """
        col = self.EXACT[name]
        with pytest.raises(ValueError, match="authoritative dtype"):
            _delegated_frame(self._main(col, [0, 1, 9, 3]))

    @pytest.mark.parametrize("name", sorted(LOSSY))
    def test_b32_176_alias_path_is_refused_too(self, name):
        """The alias path was WORSE than the direct one: `_safe_dtype_cast`
        cast the rounded floats back to `int64`, so the corruption arrived
        wearing the right dtype."""
        col = self.LOSSY[name]
        main = self._main(col, [0, 1, 9, 3])
        main.add_alias("d", "S.v", dtype=str(col.dtype))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="authoritative dtype|neutral value"):
                main.materialize_aliases(names=["d"])


@needs_dfdraw
class TestB32Round9ForcedWideningOracle:
    """GPT30 B32F8-P2-1. `test_b32_170` only catches the sparse defect on a
    pandas whose `take()` widens — it passes on 1.5.3 even with the
    restoration deleted, which is how round 8 shipped a matched-path gap that
    its own test could not see.

    Three adjacent versions behave three different ways:

        1.5.3  preserves matched AND missing
        2.2.3  widens    matched AND missing
        3.0.2  preserves matched, widens missing

    So correctness must not depend on which pandas the runner happens to
    have. These tests FORCE the widening, and therefore fail on every
    supported runtime if the restoration is removed."""

    def test_b32_177_restore_recovers_a_forced_widening(self):
        src = pd.SparseDtype(np.float32, np.nan)
        widened = pd.Series(pd.arrays.SparseArray(
            np.array([1., 2., 3., 4.]), fill_value=np.nan))
        assert str(widened.dtype) != str(src)
        out = A.AliasDataFrame._restore_exact_dtype(widened, src)
        assert out is not None and str(out.dtype) == str(src)

    def test_b32_178_forced_widening_through_the_public_path(self):
        """Monkeypatch the gather primitive to widen unconditionally, then
        require the public `draw_batch()` result to come back exact. Delete
        the restoration and this fails on pandas 1.5.3 too."""
        col = pd.arrays.SparseArray(
            np.array([1., 2., 3., 4.], dtype=np.float32), fill_value=np.nan)
        main = A.AliasDataFrame(pd.DataFrame({
            "x": np.arange(4.), "k": np.arange(4, dtype=np.int64)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "k": np.arange(4, dtype=np.int64)}))
        child.df["v"] = col
        main.register_subframe("S", child, index_columns=["k"])

        real_take = pd.arrays.SparseArray.take

        def _widening_take(self, indices, **kw):
            out = real_take(self, indices, **kw)
            return out.astype(pd.SparseDtype(np.float64,
                                             out.dtype.fill_value))

        pd.arrays.SparseArray.take = _widening_take
        try:
            got = _delegated_frame(main)
        finally:
            pd.arrays.SparseArray.take = real_take
        assert str(got.dtype) == "Sparse[float32, nan]", got.dtype

    def test_b32_179_forced_widening_that_is_lossy_is_refused(self):
        """And the restoration must never become a silent downcast."""
        widened = pd.Series([1.5, 2.5, 3.5])
        assert A.AliasDataFrame._restore_exact_dtype(
            widened, np.dtype("int64")) is None


@needs_dfdraw
class TestB32Round9LazyReaderKeys:
    """A branch a lazy reader ADVERTISES is present — it is physical data the
    frame owns and has not loaded yet.

    GPT27 FIX8-P0-2 (child side) is a regression I introduced in round 8:
    round 7 had no child-side check at all, so this shape registered fine.
    GPT30 B32F8-P1-1 and GPT31 B32F8-P1-1 report the parent-asymmetric side.

    The round-8 source comment said an unloaded lazy branch is a legitimate
    deferred key — and then the predicate did not check for one. Fourth
    iteration of the same scoping mistake in two rounds."""

    class _Reader:
        def __init__(self, branches):
            self.available_branches = list(branches)

    def test_b32_180_parent_key_from_lazy_reader_asymmetric(self):
        parent = A.AliasDataFrame(pd.DataFrame({"x": [0., 1.]}))
        parent._lazy_reader = self._Reader(["kp", "x"])
        child = A.AliasDataFrame(pd.DataFrame({"kc": [0, 1],
                                               "v": [10., 20.]}))
        parent.register_subframe("S", child, index_columns=["kp"],
                                 right_index_columns=["kc"])
        assert "S" in parent._subframes.subframes

    def test_b32_181_child_key_from_lazy_reader_symmetric(self):
        parent = A.AliasDataFrame(pd.DataFrame({"k": [0, 1], "x": [0., 1.]}))
        child = A.AliasDataFrame(pd.DataFrame({}))
        child._lazy_reader = self._Reader(["k", "v"])
        parent.register_subframe("S", child, index_columns=["k"])
        assert "S" in parent._subframes.subframes

    def test_b32_182_multi_column_mixed_loaded_and_unloaded(self):
        parent = A.AliasDataFrame(pd.DataFrame({"a": [0, 1], "x": [0., 1.]}))
        parent._lazy_reader = self._Reader(["a", "b"])
        child = A.AliasDataFrame(pd.DataFrame({"a": [0, 1], "b": [0, 1],
                                               "v": [1., 2.]}))
        parent.register_subframe("S", child, index_columns=["a", "b"],
                                 right_index_columns=["a", "b"])
        assert "S" in parent._subframes.subframes

    def test_b32_183_a_genuinely_unknown_key_is_still_refused(self):
        """Recognising advertised branches must not become recognising
        anything."""
        parent = A.AliasDataFrame(pd.DataFrame({"k": [0, 1], "x": [0., 1.]}))
        parent._lazy_reader = self._Reader(["k", "x"])
        child = A.AliasDataFrame(pd.DataFrame({"other": [0, 1],
                                               "v": [1., 2.]}))
        child._lazy_reader = self._Reader(["other", "v"])
        with pytest.raises(ValueError, match="right_index_columns not found"):
            parent.register_subframe("S", child, index_columns=["k"])
        assert "S" not in parent._subframes.subframes

    def test_b32_184_the_check_loads_nothing(self):
        """Registration must have no effects, least of all in the validator
        that exists to prevent them."""
        parent = A.AliasDataFrame(pd.DataFrame({"x": [0., 1.]}))
        reader = self._Reader(["kp", "x"])
        loads = []
        reader.load_branches = lambda *a, **k: loads.append(a)
        parent._lazy_reader = reader
        child = A.AliasDataFrame(pd.DataFrame({"kc": [0, 1],
                                               "v": [10., 20.]}))
        before = list(parent.df.columns)
        parent.register_subframe("S", child, index_columns=["kp"],
                                 right_index_columns=["kc"])
        assert not loads, "the validator loaded a branch"
        assert list(parent.df.columns) == before


# ============================================================================
# CORRECTION ROUND 10 — AD-19 RATIFIED, Option 1 with an operational
# definition (architect, 2026-07-29):
#
#   "Every dtype observable from source metadata, an existing physical column,
#    schema metadata, an explicit alias declaration, or the first successful
#    creation/materialization is authoritative. ADF must preserve it
#    thereafter. If a missing value cannot be represented in that dtype, ADF
#    must use an explicitly configured compatible fill or refuse clearly."
#
# and, on the fill itself:
#
#   "the fill is the physical neutral value used in the calculation; the flag
#    records applicability or provenance ... Do not automatically choose 0, 1,
#    False, or any other fill. Those are physical choices made by the user."
#
# The eight permanent tests below are the ones the architect enumerated, plus
# the fully-matched declared-alias case that caught the coder in round 9.
# ============================================================================


@needs_dfdraw
class TestB32Round10ConfiguredFillContract:
    """The three public mechanisms, their precedence, and the refusals.

    Round 9 shipped with `add_alias(fill_value=...)` unable to reach the
    gather: a user with a perfectly valid configured neutral value got a
    refusal, because the fill was applied one layer too late (GPT30 R9-P0-1,
    GPT31 B32F9-P1-1). It now participates, at the SAME precedence it always
    had — the architect asked that historical behaviour not change silently,
    and the measured historical order is subframe > global > alias."""

    BIG = np.array([2**60 + 1, 2**60 + 3, 2**60 + 5, 2**60 + 7], dtype=np.int64)
    SMALL = np.array([10, 20, 30, 40], dtype=np.int64)

    def _adf(self, col, keys=(0, 1, 9, 3), sub=None, glob=None):
        main = A.AliasDataFrame(pd.DataFrame({
            "k": np.asarray(keys, dtype=np.int64), "x": np.arange(4.)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "k": np.arange(4, dtype=np.int64)}))
        child.df["v"] = col
        main.register_subframe("S", child, index_columns=["k"])
        if glob is not None:
            main.set_global_fill(fill_missing=glob)
        if sub is not None:
            main.set_subframe_fill("S", fill_missing=sub)
        return main

    def _alias(self, main, **kw):
        main.add_alias("d", "S.v", **kw)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main.materialize_aliases(names=["d"])
        return main.df["d"]

    # ---- 1. additive alias fill 0 -------------------------------------
    def test_b32_185_additive_alias_fill_zero(self):
        got = self._alias(self._adf(self.BIG), dtype="int64", fill_value=0)
        assert str(got.dtype) == "int64"
        assert [int(v) for v in got.values] == [
            int(self.BIG[0]), int(self.BIG[1]), 0, int(self.BIG[3])]

    # ---- 2. multiplicative alias fill 1 --------------------------------
    def test_b32_186_multiplicative_alias_fill_one(self):
        """The whole reason ADF may not infer the neutral value: 0 is neutral
        for an additive correction and 1 for a multiplicative one, and the
        dtype is identical in both cases."""
        got = self._alias(self._adf(self.BIG), dtype="int64", fill_value=1)
        assert str(got.dtype) == "int64"
        assert int(got.values[2]) == 1
        assert int(got.values[0]) == int(self.BIG[0])

    # ---- 3. subframe-specific fill -------------------------------------
    def test_b32_187_subframe_fill(self):
        got = self._alias(self._adf(self.BIG, sub=7), dtype="int64")
        assert str(got.dtype) == "int64"
        assert int(got.values[2]) == 7
        assert int(got.values[0]) == int(self.BIG[0])

    # ---- 4. global fill -------------------------------------------------
    def test_b32_188_global_fill(self):
        got = self._alias(self._adf(self.BIG, glob=5), dtype="int64")
        assert str(got.dtype) == "int64"
        assert int(got.values[2]) == 5

    # ---- 5. precedence ---------------------------------------------------
    @pytest.mark.parametrize("sub,glob,alias_fill,expected", [
        (1,    0,    2, 1),   # subframe wins over both
        (None, 0,    2, 0),   # global wins over alias
        (None, None, 2, 2),   # alias used when nothing else is configured
        (1,    None, 2, 1),
    ])
    def test_b32_189_precedence_is_unchanged(self, sub, glob, alias_fill,
                                             expected):
        """MEASURED historical order, preserved deliberately: subframe >
        global > alias. The architect asked not to change it silently, so it
        is pinned rather than redesigned — the round-10 change is only that
        the alias value now reaches the gather instead of being applied after
        it, which is what makes it usable on values that cannot survive the
        intermediate."""
        got = self._alias(self._adf(self.SMALL, sub=sub, glob=glob),
                          dtype="int64", fill_value=alias_fill)
        assert int(got.values[2]) == expected

    # ---- 6. incompatible fill is refused ---------------------------------
    @pytest.mark.parametrize("bad", [1.5, "x"])
    def test_b32_190_incompatible_fill_is_refused(self, bad):
        with pytest.raises(ValueError):
            self._alias(self._adf(self.SMALL, sub=bad), dtype="int64")

    # ---- 7. no fill at all is refused ------------------------------------
    @pytest.mark.parametrize("col_name", ["BIG", "SMALL"])
    def test_b32_191_no_configured_fill_is_refused(self, col_name):
        """Both magnitudes. Round 9 refused only the lossy one; AD-19 removed
        that exception because the dtype changing at all is the violation."""
        col = getattr(self, col_name)
        with pytest.raises(ValueError,
                           match="authoritative dtype|neutral value"):
            self._alias(self._adf(col), dtype="int64")

    # ---- 8. dtype and matched values unchanged, every route ---------------
    @pytest.mark.parametrize("route", ["alias_fill", "subframe", "global"])
    @pytest.mark.parametrize("col_name", ["BIG", "SMALL"])
    def test_b32_192_dtype_and_matched_values_unchanged(self, route,
                                                        col_name):
        col = getattr(self, col_name)
        kw, sub, glob = {}, None, None
        if route == "alias_fill":
            kw = {"fill_value": 0}
        elif route == "subframe":
            sub = 0
        else:
            glob = 0
        got = self._alias(self._adf(col, sub=sub, glob=glob),
                          dtype="int64", **kw)
        assert str(got.dtype) == "int64", route
        for i in (0, 1, 3):
            assert int(got.values[i]) == int(col[[0, 1, 9, 3][i]]), route

    # ---- the case that caught the coder in round 9 -----------------------
    @pytest.mark.parametrize("col_name", ["BIG", "SMALL"])
    def test_b32_193_fully_matched_declared_alias_is_exact(self, col_name):
        """`_safe_dtype_cast` converted EVERY integer result through
        `float64`, unconditionally — so a fully matched alias with no missing
        key at all was corrupted by the function whose job is to preserve its
        dtype. Round 9's exactness guard sits in the gather and could not see
        it. The Main Reviewer asked for exactly this reproduction to settle a
        genuine disagreement between two reviewer traces; it settled here."""
        col = getattr(self, col_name)
        got = self._alias(self._adf(col, keys=(0, 1, 2, 3)), dtype="int64")
        assert str(got.dtype) == "int64"
        assert [int(v) for v in got.values] == [int(v) for v in col]

    def test_b32_194_no_automatic_neutral_value_anywhere(self):
        """The negative form, stated once: ADF must never produce 0/False for
        a gap it was not told about. If this ever passes by producing a value
        instead of raising, the ruling has been lost."""
        for dt, col in (("int64", self.SMALL),
                        ("bool", np.array([True, False, True, False]))):
            with pytest.raises(ValueError):
                self._alias(self._adf(col), dtype=dt)


# ============================================================================
# CORRECTION ROUND 11 — the ratified contract
# PHASE_13_76_ADF_AD19_DTYPE_AND_FILL_BRAINSTORM_v1_4_2_RATIFIED.md
# architect-ratified 2026-07-30, coding authorized §20/§21.
#
# TEST-FIRST SCAFFOLD. Every test in this class asserts the RATIFIED
# behaviour, so a subset of them MUST FAIL on the round-10 bytes
# (AliasDataFrame.py f9781761718e6d85f016f5f83fbbaf86). The baseline failure
# list is published in the round-11 CRR and each entry has to flip.
#
# Rounds 5-10 each shipped a matrix that was structurally incapable of
# failing on the defect the panel then found. This class is written before
# the fix, and its baseline result is recorded, so that cannot repeat.
# ============================================================================


class TestB32Round11RatifiedContract:
    """AR-1..AR-4 and AC_1..AC_8 of the ratified v1.4.2 contract."""

    BIG = np.array([2**60 + 1, 2**60 + 3], dtype=np.int64)

    # ---- builders --------------------------------------------------------
    def _pair(self, child_col=None, keys=(0, 9), x=(10, 20),
              sub=None, glob=None):
        """Parent with two rows; child holds key 0 only, so key 9 is an
        unmatched row-level join key (§7.3), never a structural absence."""
        main = A.AliasDataFrame(pd.DataFrame({
            "k": np.asarray(keys, dtype=np.int64),
            "x": np.asarray(x, dtype=np.int64)}))
        child = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        child.df["v"] = (np.array([3], dtype=np.int64)
                         if child_col is None else child_col)
        main.register_subframe("S", child, index_columns=["k"])
        if glob is not None:
            main.set_global_fill(fill_missing=glob)
        if sub is not None:
            main.set_subframe_fill("S", fill_missing=sub)
        return main

    @staticmethod
    def _mat(main, name="d", bulk=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bulk:
                main.materialize_aliases(names=[name])
            else:
                main.materialize_alias(name)
        return main.df[name]

    # ================= AR-3 : alias fill is FINAL-RESULT only =============

    # MARKER REMOVED in 11c — D_5 core landed and strict=True turned the
    # XPASS into a suite failure, which is what forced this edit.
    def test_b32_195_compound_additive_alias_fill_is_final_result(self):
        """AR-3 / §8.3. `S.v + x`, alias fill 1, no operand policy.

        The expression result at the unmatched row is UNDEFINED, so the
        configured alias value becomes the FINAL result: 1, not 1 + x.
        Round 10 published the alias fill into the gather and returned 21.
        Pre-phase bytes return 1; this pins the pre-phase meaning."""
        m = self._pair()
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        got = self._mat(m)
        assert str(got.dtype) == "int64"
        assert [int(v) for v in got.values] == [13, 1]

    # MARKER REMOVED in 11c — D_5 core.
    def test_b32_196_compound_multiplicative_alias_fill_is_final_result(self):
        """AR-3. `S.v * x`, alias fill 1 -> [30, 1], not [30, 20]."""
        m = self._pair()
        m.add_alias("d", "S.v * x", dtype="int64", fill_value=1)
        got = self._mat(m)
        assert [int(v) for v in got.values] == [30, 1]

    # MARKER REMOVED in 11c — D_5 core.
    def test_b32_197_compound_alias_fill_bulk_path(self):
        """AC_3. The bulk path must agree with the single path."""
        m = self._pair()
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        got = self._mat(m, bulk=True)
        assert [int(v) for v in got.values] == [13, 1]

    def test_b32_198_operand_fill_defines_the_input_alias_fill_stands_down(
            self):
        """AR-3 / §14.3 decisive control. subframe fill 0 DEFINES S.v, the
        expression evaluates 0 + 20 = 20, the result is valid, so the alias
        fill does NOT fire. This is the measured historical precedence and
        must not change."""
        m = self._pair(sub=0)
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        got = self._mat(m)
        assert [int(v) for v in got.values] == [13, 20]

    # ================= AC_3 : one evaluator, four wrappers ================

    @pytest.mark.parametrize("getter", ["get_alias_series", "get_alias_array"])
    # MARKER REMOVED in 11d — D_4b landed; strict=True forced this edit.
    def test_b32_199_alias_fill_reaches_the_non_materializing_getters(
            self, getter):
        """AC_3 / §6.3 (GPT30 F10-P0-1). A configured alias fill works
        through `materialize_alias` but the two getters refuse, because they
        never saw the fill. All four public surfaces are one evaluator."""
        m = self._pair(child_col=self.BIG[:1])
        m.add_alias("d", "S.v", dtype="int64", fill_value=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = getattr(m, getter)("d")
        arr = np.asarray(out)
        assert arr.dtype == np.int64
        assert [int(v) for v in arr] == [int(self.BIG[0]), 0]

    # MARKER REMOVED in 11d — D_4b landed.
    def test_b32_200_getters_do_not_commit_authority(self):
        """AC_5 / §5.4. A getter is not a stored in-frame publication."""
        m = self._pair()
        m.add_alias("q", "S.v * 2", fill_value=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.get_alias_series("q")
        assert not m.get_dtype_authority("q").known, (
            "a non-materializing getter must not record dtype authority")

    # ================= AC_5 : first stored in-frame authority =============

    def test_b32_201_first_materialization_records_and_enforces_dtype(self):
        """AD-19 source 4 / D_8 (GPT25, GPT26, GPT30, GPT31 F10). An alias
        with no declared dtype must record the dtype of its first successful
        nonempty stored in-frame materialization, and a later
        rematerialization must restore it or refuse -- never silently drift."""
        state = {"mode": "int"}

        def dyn(v):
            a = np.asarray(v)
            return (a.astype(np.int64) if state["mode"] == "int"
                    else a.astype(np.float64) + 0.5)

        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.register_function("dyn", dyn)
        m.add_alias("q", "dyn(x)")
        first = self._mat(m, "q")
        assert str(first.dtype) == "int64"

        auth = m.get_dtype_authority("q")
        assert auth.known and str(auth.dtype) == "int64", (
            "first stored in-frame materialization must record the authority")
        # the origin is a plain string BY DESIGN (it is serialized into the
        # schema and through ROOT metadata), so assert the wire value
        assert auth.origin == "first_stored_in_frame_materialization"

        m.dematerialize(["q"])
        state["mode"] = "float"
        try:
            again = self._mat(m, "q")
        except ValueError:
            return                              # refusal is a valid outcome
        assert str(again.dtype) == "int64", (
            "rematerialization must restore the authority or refuse, "
            "never silently adopt a newly inferred dtype")

    def test_b32_202_zero_row_materialization_does_not_commit_authority(self):
        """AC_5 / §5.4. A zero-row result establishes nothing; the observed
        dtype is a backend default, not an observation of the data."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([], np.int64)}))
        m.add_alias("q", "x * 2")
        self._mat(m, "q")
        assert not m.get_dtype_authority("q").known, (
            "zero-row materialization must leave the authority UNKNOWN")

    # ================= AC_6 : all rows undefined ==========================

    # MARKER REMOVED in 11e — AC_6 landed; strict=True forced this edit.
    def test_b32_203_all_undefined_fill_alone_does_not_establish_authority(
            self):
        """AC_6 / §5.4 (GPT29). No defined value exists to infer a
        provisional dtype from, and a fill constant is a policy choice, not
        an observation. Stored in-frame publication must refuse."""
        m = self._pair(keys=(7, 9))            # NO key matches
        m.add_alias("d", "S.v", fill_value=0)  # no explicit dtype
        with pytest.raises(ValueError):
            self._mat(m)

    def test_b32_204_all_undefined_with_explicit_dtype_succeeds(self):
        """AC_6 positive control. An explicit target supplies the contract
        the fill can be validated against, so publication succeeds."""
        m = self._pair(keys=(7, 9))
        m.add_alias("d", "S.v", dtype="int64", fill_value=0)
        got = self._mat(m)
        assert str(got.dtype) == "int64"
        assert [int(v) for v in got.values] == [0, 0]

    # ================= AC_2 : mask-only, no widening ======================

    def test_b32_205_large_int_survives_an_operand_filled_compound(self):
        """AC_2 / §15.3. Values above 2**53 with one undefined row. If
        undefinedness is carried as a sentinel in the value array the array
        must widen to float64 and these values collapse; carried in a mask,
        they are exact."""
        m = self._pair(child_col=self.BIG[:1], sub=0)
        m.add_alias("d", "S.v + x", dtype="int64")
        got = self._mat(m)
        assert str(got.dtype) == "int64"
        assert int(got.values[0]) == int(self.BIG[0]) + 10
        assert int(got.values[1]) == 20

    def test_b32_206_no_fill_no_widening_refusal_is_kept(self):
        """AR-4 Option B. Evaluation/materialization with an authoritative
        integer dtype, an unmatched row and no compatible fill refuses. This
        passes on the round-10 bytes and must keep passing."""
        m = self._pair(child_col=self.BIG[:1])
        m.add_alias("d", "S.v", dtype="int64")
        with pytest.raises(ValueError):
            self._mat(m)

    # ================= AC_1 : configured-fill compatibility ===============

    def test_b32_207_float_target_fill_uses_nearest_representable(self):
        """AC_1 / §4.2 floating clause. RELAXATION vs the round-10 bytes,
        which refuse 0.1 because it does not survive a float32 round trip.
        The ratified rule is nearest-representable: ordinary rounding is
        accepted, only destruction is refused."""
        child = np.array([1.0], dtype=np.float32)
        m = self._pair(child_col=child, sub=0.1)
        m.add_alias("d", "S.v")
        got = self._mat(m)
        assert str(got.dtype) == "float32"
        assert float(got.values[1]) == float(np.float32(0.1))

    @pytest.mark.parametrize("bad", [1e-50, 1e40])
    def test_b32_208_destructive_float_fill_is_refused(self, bad):
        """AC_1. Underflow to 0 and overflow to Inf destroy the configured
        value rather than round it, and stay refused.

        The scaffold first used 1e-45, which is WRONG as an oracle: 1e-45 is
        a float32 SUBNORMAL and survives (`np.float32(1e-45)` is 1e-45, not
        0). The v1.4.4 §4.2 boundary is destruction, and a subnormal is not
        destroyed. 1e-50 genuinely underflows to 0.0 and is the correct
        control. Measured, not assumed."""
        child = np.array([1.0], dtype=np.float32)
        with pytest.raises(ValueError):
            m = self._pair(child_col=child, sub=bad)
            m.add_alias("d", "S.v")
            self._mat(m)

    @pytest.mark.parametrize("dtype_name,bad", [
        ("int64", 1.5), ("uint8", -1), ("uint8", 300), ("bool", 2)])
    def test_b32_209_lossy_fill_constants_stay_refused(self, dtype_name, bad):
        """AC_1 / §4.2. Permissive ordinary conversion does not authorize a
        lossy fill constant. Passes today; pinned against regression."""
        col = np.array([1], dtype=dtype_name)
        with pytest.raises(ValueError):
            m = self._pair(child_col=col, sub=bad)
            m.add_alias("d", "S.v")
            self._mat(m)

    # ================= AC_7 : strict opt-in vs ordinary route =============

    def test_b32_210_strict_route_and_ordinary_route_differ(self):
        """AC_7 / AR-2 / §6.4. Both must be demonstrable, and they must give
        visibly different results on identical input -- otherwise the
        distinction the architect ratified has silently collapsed."""
        def strict_int8(v):
            a = np.asarray(v)
            lo, hi = np.iinfo(np.int8).min, np.iinfo(np.int8).max
            if a.dtype.kind == "f":
                if not np.all(np.isfinite(a)):
                    raise ValueError("strict_int8: non-finite value")
                if not np.all(a == np.floor(a)):
                    raise ValueError("strict_int8: fractional value")
            if a.size and (a.min() < lo or a.max() > hi):
                raise ValueError("strict_int8: value outside int8 range")
            return a.astype(np.int8)

        src = pd.DataFrame({"x": np.array([1.5, 300.0])})

        ordinary = A.AliasDataFrame(src.copy())
        ordinary.add_alias("o", "x", dtype="int8")
        ordinary.materialize_alias("o")
        assert str(ordinary.df["o"].dtype) == "int8"
        assert [int(v) for v in ordinary.df["o"].values] == [1, 44]

        strict = A.AliasDataFrame(src.copy())
        strict.register_function("strict_int8", strict_int8)
        strict.add_alias("s", "strict_int8(x)")
        with pytest.raises(ValueError):
            strict.materialize_alias("s")

    # ================= AC_4 : structural vs row-level =====================

    def test_b32_211_structural_absence_is_not_hidden_by_a_fill(self):
        """AC_4 / §7.2. A misspelled subframe column is a structural
        dependency error. A configured fill must never absorb it."""
        m = self._pair(sub=0)
        m.add_alias("d", "S.vv + x", dtype="int64", fill_value=0)
        with pytest.raises(Exception) as excinfo:
            self._mat(m)
        assert not isinstance(excinfo.value, AssertionError)


class TestB32Round11bAuthorityDefects:
    """The four defects GPT31 and GPT32 executed against the first D_8 draft.

    All four had the same root cause: D_8 was bolted onto ONE publication
    site instead of being a contract every stored in-frame publication owes.
    That is the "special if per case" shape the architect has flagged
    repeatedly, and the reason these are permanent tests rather than fixes.
    """

    @staticmethod
    def _dyn_frame():
        state = {"mode": "int"}

        def dyn(v):
            a = np.asarray(v)
            return (a.astype(np.int64) if state["mode"] == "int"
                    else a.astype(np.float64) + 0.5)

        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.register_function("dyn", dyn)
        m.add_alias("q", "dyn(x)")
        return m, state

    # ---- F11B-P0-1 : both public paths owe the same contract -------------
    @pytest.mark.parametrize("bulk", [False, True], ids=["single", "bulk"])
    def test_b32_212_both_materialization_paths_record_authority(self, bulk):
        m, _ = self._dyn_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_aliases(names=["q"]) if bulk else m.materialize_alias("q")
        auth = m.get_dtype_authority("q")
        assert auth.known and str(auth.dtype) == "int64", (
            "materialize_aliases() is a stored in-frame publication and owes "
            "the same source-4 contract as materialize_alias()")
        assert auth.origin == "first_stored_in_frame_materialization"

    @pytest.mark.parametrize("bulk", [False, True], ids=["single", "bulk"])
    def test_b32_213_both_paths_refuse_dtype_drift(self, bulk):
        m, state = self._dyn_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_aliases(names=["q"]) if bulk else m.materialize_alias("q")
            m.dematerialize(["q"])
            state["mode"] = "float"
            with pytest.raises(ValueError, match="AD-19 source 4"):
                m.materialize_aliases(names=["q"]) if bulk else m.materialize_alias("q")

    # ---- F11B-P0-2 : the record is storage-family neutral ----------------
    @pytest.mark.parametrize("spec", ["Int64", "boolean", "string", "category",
                                      "datetime64[ns, UTC]", "Float32"])
    def test_b32_214_extension_dtype_authority_is_readable(self, spec):
        """np.dtype() raises `data type 'Int64' not understood` for every one
        of these — the families AD-11 and AD-19 explicitly cover."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.add_alias("q", "x", dtype=spec)
        auth = m.get_dtype_authority("q")
        assert auth.known
        assert str(auth.dtype) == str(pd.api.types.pandas_dtype(spec))

    def test_b32_215_extension_dtype_survives_rematerialization(self):
        def ext(v):
            return pd.Series(np.asarray(v), dtype="Int64")
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.register_function("ext", ext)
        m.add_alias("q", "ext(x)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
            assert str(m.get_dtype_authority("q").dtype) == "Int64"
            m.dematerialize(["q"])
            m.materialize_alias("q")
        assert str(m.df["q"].dtype) == "Int64"

    # ---- F11B-P1-1 : the record must describe the STORED column ----------
    def test_b32_216_authority_records_the_stored_column_not_the_input(self):
        """`self.df[name] = result` realigns on the index. Recording the
        pre-assignment dtype made the registry state something false: stored
        float64/[NaN, NaN] while the record claimed int64."""
        def misaligned(v):
            return pd.Series([1, 2], index=[10, 11], dtype="int64")
        m = A.AliasDataFrame(pd.DataFrame({"x": [1, 2]}, index=[0, 1]))
        m.register_function("misaligned", misaligned)
        m.add_alias("q", "misaligned(x)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        assert str(m.get_dtype_authority("q").dtype) == str(m.df["q"].dtype), (
            "the recorded authority must describe the column pandas actually "
            "stored, never the object handed to the assignment")

    # ---- F11B-P1-3 : redefinition does not revoke an authority -----------
    def test_b32_217_authority_survives_expression_redefinition(self):
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.add_alias("q", "x * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        assert str(m.get_dtype_authority("q").dtype) == "int64"
        m.add_alias("q", "x / 2")            # expression changes, no dtype
        assert str(m.get_dtype_authority("q").dtype) == "int64", (
            "redefining the EXPRESSION does not revoke a source-4 authority")

    def test_b32_218_explicit_dtype_redefinition_supersedes(self):
        """An explicit dtype= IS a deliberate source-3 declaration and DOES
        replace an inference - the documented way to change the contract."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.add_alias("q", "x * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        m.add_alias("q", "x / 2", dtype="float64")
        auth = m.get_dtype_authority("q")
        assert str(auth.dtype) == "float64"
        assert auth.origin == "explicit_alias"


class TestB32Round11b2PublicationOwner:
    """F11B2-P1-1 and F11B2-P1-2 — the two defects GPT31/GPT32 executed
    against the FIRST correction. Both were in code written to close the
    previous round's findings, which is why they are pinned permanently."""

    # ---- F11B2-P1-1 : enforce the ALIGNED candidate, not the input -------
    @pytest.mark.parametrize("bulk", [False, True], ids=["single", "bulk"])
    def test_b32_219_post_alignment_never_violates_authority(self, bulk):
        """`self.df[name] = series` REALIGNS on the parent index. Enforcing
        the pre-assignment object let a misaligned int64 Series be stored as
        float64/[NaN, NaN] while the record still said int64 — the record was
        false about the physical column, which is the invariant this phase
        exists to protect."""
        state = {"misaligned": False}

        def dyn(v):
            if state["misaligned"]:
                return pd.Series([3, 4], index=[10, 11], dtype="int64")
            return pd.Series([1, 2], index=[0, 1], dtype="int64")

        m = A.AliasDataFrame(pd.DataFrame({"x": [1, 2]}, index=[0, 1]))
        m.register_function("dyn", dyn)
        m.add_alias("q", "dyn(x)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_aliases(names=["q"]) if bulk else m.materialize_alias("q")
            assert str(m.get_dtype_authority("q").dtype) == "int64"
            m.dematerialize(["q"])
            state["misaligned"] = True
            try:
                m.materialize_aliases(names=["q"]) if bulk else m.materialize_alias("q")
            except ValueError:
                return                       # refusal is the correct outcome
        # if it published, the record and the stored column MUST agree
        assert str(m.get_dtype_authority("q").dtype) == str(m.df["q"].dtype), (
            "the authority may never disagree with the column pandas stored")

    def test_b32_220_authority_always_matches_the_stored_column(self):
        """The invariant behind b32_219, stated directly."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.add_alias("q", "x * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        auth = m.get_dtype_authority("q")
        assert auth.known and str(auth.dtype) == str(m.df["q"].dtype)

    # ---- F11B2-P1-2 : never record a partial truth as exact --------------
    def test_b32_221_categorical_establishes_an_exact_inferred_authority(self):
        """SUPERSEDED BY ITS OWN NAMED SUCCESSOR, in B3.2b STEP 4.

        This test pinned a DEFERRAL, and said so: "disclosed deferral, B3.2b
        owns the exact structured codec". `str(dtype)` is `'category'` for
        every categorical, so recording it would have claimed an authority it
        did not have -- ['a','b'] unordered and ['b','a'] ORDERED stringify
        identically. Refusing was right while the codec did not exist.

        STEP 4 built it, so the deferral is over and the assertion inverts:
        a categorical now DOES establish authority, and the guard becomes the
        stronger one -- it must be EXACT, categories and order included.

        WHY THIS IS UPDATED WHILE `b32b_13b`'s CONFLICT WAS NOT, because the
        two look identical and are not. There, `test_deserialize_schema_
        restores_dtypes` pinned a representation and named NO successor, so my
        criterion was the intruder and I withdrew it. Here the pinned
        behaviour names B3.2b as the owner of the change that supersedes it.
        A test that says "X owns replacing me" is replaced by X; a test that
        simply disagrees with a new idea is not."""
        def cat(_):
            return pd.Series(pd.Categorical(["a", "b"],
                                            categories=["b", "a"],
                                            ordered=True))
        m = A.AliasDataFrame(pd.DataFrame({"x": [1, 2]}))
        m.register_function("cat", cat)
        m.add_alias("q", "cat(x)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        assert str(m.df["q"].dtype) == "category"
        auth = m.get_dtype_authority("q")
        assert auth.known, (
            "B3.2b STEP 4 built the structured codec, so a categorical must "
            f"now establish an authority: {auth!r}")
        # EXACT, not approximate — the property the deferral protected.
        assert list(auth.dtype.categories) == ["b", "a"], (
            f"category ORDER was not preserved: "
            f"{list(auth.dtype.categories)}")
        assert auth.dtype.ordered is True, "orderedness was not preserved"
        assert auth.dtype == m.df["q"].dtype, (
            "the recorded authority is not the column's actual dtype")

    @pytest.mark.parametrize("spec", ["int64", "float32", "Int64", "boolean",
                                      "datetime64[ns]"])
    def test_b32_222_exactly_representable_dtypes_still_record(self, spec):
        """The deferral is scoped by a ROUND TRIP, not by a dtype name list —
        the same discipline _coerce_fill_to_dtype uses. Everything whose
        string form reconstructs exactly must still establish authority."""
        assert A.AliasDataFrame._authority_is_exactly_representable(
            pd.api.types.pandas_dtype(spec)), spec

    def test_b32_223_categorical_fails_the_representability_round_trip(self):
        ct = pd.CategoricalDtype(categories=["a", "b"], ordered=True)
        assert not A.AliasDataFrame._authority_is_exactly_representable(ct)


class TestB32Round11b3ResultShapeParity:
    """F11B3-P1-1 — the publication owner must own EVERY supported result
    shape, not only the one the first test used.

    `_aligned_publication_candidate` normalized only `pd.Series`, so a scalar
    or list alias reached authority enforcement without a `.dtype` and raised
    `AttributeError` on rematerialization — while the bulk path worked,
    because `pd.DataFrame(results, index=...)` normalizes for free. A
    singular/bulk parity defect inside the helper written to end
    singular/bulk parity defects."""

    SHAPES = {
        "scalar":  ("5",        None),
        "list":    ("lst(x)",   lambda v: [7, 8]),
        "ndarray": ("arr(x)",   lambda v: np.array([9, 10], np.int64)),
        "series":  ("ser(x)",   lambda v: pd.Series([11, 12], dtype="int64")),
        "nullable": ("nul(x)",  lambda v: pd.Series([13, 14], dtype="Int64")),
    }

    def _frame(self, shape):
        expr, fn = self.SHAPES[shape]
        m = A.AliasDataFrame(pd.DataFrame({"x": [1, 2]}))
        if fn is not None:
            m.register_function(expr.split("(")[0], fn)
        m.add_alias("q", expr)
        return m

    @staticmethod
    def _mat(m, bulk):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_aliases(names=["q"]) if bulk else m.materialize_alias("q")

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    @pytest.mark.parametrize("bulk", [False, True], ids=["single", "bulk"])
    def test_b32_224_every_result_shape_publishes_and_rematerializes(
            self, shape, bulk):
        m = self._frame(shape)
        self._mat(m, bulk)
        first_vals, first_dtype = m.df["q"].tolist(), str(m.df["q"].dtype)
        auth = m.get_dtype_authority("q")
        assert auth.known and str(auth.dtype) == first_dtype
        m.dematerialize(["q"])
        self._mat(m, bulk)               # must NOT raise AttributeError
        assert m.df["q"].tolist() == first_vals
        assert str(m.df["q"].dtype) == first_dtype

    @pytest.mark.parametrize("shape", sorted(SHAPES))
    def test_b32_225_single_and_bulk_agree_on_dtype_and_values(self, shape):
        """The parity assertion itself, stated directly rather than implied
        by two separately-parameterized runs."""
        a, b = self._frame(shape), self._frame(shape)
        self._mat(a, False)
        self._mat(b, True)
        assert str(a.df["q"].dtype) == str(b.df["q"].dtype)
        assert a.df["q"].tolist() == b.df["q"].tolist()
        assert (str(a.get_dtype_authority("q").dtype)
                == str(b.get_dtype_authority("q").dtype))

    def test_b32_226_dataframe_result_is_refused_not_published(self):
        """A 2-D result is not a 1-D alias column; refuse before publication
        rather than let pandas invent a shape."""
        m = A.AliasDataFrame(pd.DataFrame({"x": [1, 2]}))
        m.register_function("two_d", lambda v: pd.DataFrame({"a": [1, 2],
                                                             "b": [3, 4]}))
        m.add_alias("q", "two_d(x)")
        with pytest.raises((TypeError, ValueError)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_alias("q")


# ============================================================================
# CORRECTION ROUND 11c — D_5 core (materialization path)
#
# Governed by the ratified design review
# PHASE_13_76_ADF_fix11c_D4_D5_OFFICIAL_DESIGN_REVIEW_SUMMARY_20260807.md
#   Decision A = Option 1, Decision B = Option 1
#   §5 fail-closed row-local gate is PRIMARY, A/B probe is defense-in-depth
#   §6 UNDEFINED * 0 -> UNDEFINED, never 0
#   §7 probe isolation must be proved on the real bytes
#
# Every test here was written against the design note, and the ones that
# could fail on the pre-11c bytes were checked to fail there first.
# ============================================================================


def _adf_module():
    """The module that DEFINES AliasDataFrame, resolved through the class.

    `import AliasDataFrame as A` gives the PACKAGE on alma2 and re-exports
    only public names, so module-level helpers (`_AliasEvalContext`,
    `_expression_is_row_local`) are not reachable as `A.<name>`. Going through
    `__module__` works identically on both layouts.
    """
    import sys as _sys
    return _sys.modules[A.AliasDataFrame.__module__]


class TestB32Round11cMaskCarriage:
    """D_5 core: placeholder + authoritative mask, refusal at the final stage."""

    @staticmethod
    def _pair(sub=None, child=None, keys=(0, 9), x=(10, 20)):
        main = A.AliasDataFrame(pd.DataFrame({
            "k": np.asarray(keys, dtype=np.int64),
            "x": np.asarray(x, dtype=np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = (np.array([3], dtype=np.int64) if child is None else child)
        main.register_subframe("S", ch, index_columns=["k"])
        if sub is not None:
            main.set_subframe_fill("S", fill_missing=sub)
        return main

    @staticmethod
    def _mat(m, name="d", bulk=False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bulk:
                m.materialize_aliases(names=[name])
            else:
                m.materialize_alias(name)
        return m.df[name]

    # ---- §7 probe isolation — the binding verification -------------------

    def test_b32_236_probe_b_is_isolated_from_probe_a(self):
        """Opus5_2's finding, turned into a permanent control.

        `_scatter_subframe_column` has an idempotent fast path that returns
        the joined column if it is already present in `self.df`. Measured on
        baseline ee9227e7, a second `_eval_in_namespace` consumed a poisoned
        `v__S` verbatim: [777,777] -> [787,797]. A probe built on re-evaluation
        through the normal path would therefore compare A against A, agree with
        itself on EVERY expression, and look like a working guard.

        This test pins the isolation directly: `_eval_prepared` must honour the
        overlay and must NOT read the frame column."""
        m = self._pair()
        m.df["v__S"] = np.array([3, 3], dtype=np.int64)
        got = m._eval_prepared("v__S + x", {"v__S": pd.Series(
            np.array([100, 100], dtype=np.int64), index=m.df.index)})
        assert [int(v) for v in np.asarray(got)] == [110, 120], (
            "probe B read the frame column instead of the overlay — the "
            "guard would be incapable of firing")
        assert [int(v) for v in m.df["v__S"].values] == [3, 3], (
            "probe B must not write back into self.df")

    def test_b32_237_probe_runs_exactly_two_top_level_evaluations(self):
        """Re-entrancy: `_eval_prepared` performs no scatter and carries no
        context, so the guard cannot recurse. Counted, not assumed."""
        m = self._pair()
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        seen = {}
        orig = A.AliasDataFrame._eval_in_namespace

        def _count(self, *a, **kw):
            ctx = kw.get("ctx")
            if ctx is not None:
                seen["ctx_evals"] = seen.get("ctx_evals", 0) + 1
            return orig(self, *a, **kw)

        A.AliasDataFrame._eval_in_namespace = _count
        try:
            self._mat(m)
        finally:
            A.AliasDataFrame._eval_in_namespace = orig
        assert seen.get("ctx_evals") == 1, (
            "the probe must not re-enter the join-preparing evaluator")

    # ---- §5 fail-closed row-local gate -----------------------------------

    @pytest.mark.parametrize("expr", [
        "(S.v + x).cumsum()",       # attribute/method — reduction
        "S.v + x - x.mean()",       # the shape no name-based deny-list catches
        "S.v / x.sum()",            # reduction through attribute
        "S.v + x if x[0] > 0 else x",   # conditional provenance
    ])
    def test_b32_238_non_row_local_is_refused_even_though_probes_may_agree(
            self, expr):
        """§5. The GATE is primary. None of these may publish, regardless of
        what the A/B probe would say — that is the whole point of fail-closed:
        an unlisted construct is refused, so an incomplete list costs a
        needless refusal, never a wrong number."""
        m = self._pair()
        m.add_alias("d", expr, fill_value=1)
        with pytest.raises(Exception) as ei:
            self._mat(m)
        assert "row-local" in str(ei.value) or "parsed" in str(ei.value)

    def test_b32_239_unclassified_registered_function_is_refused(self):
        """A user-registered function is NOT on the proven row-local list, so
        it is refused while a residual mask survives — even if it happens to
        be elementwise. Fail closed."""
        m = self._pair()
        m.register_function("my_scale", lambda v: v * 2)
        m.add_alias("d", "my_scale(S.v + x)", fill_value=1)
        with pytest.raises(Exception) as ei:
            self._mat(m)
        assert "row-local" in str(ei.value)

    def test_b32_240_gate_does_not_run_without_a_residual_mask(self):
        """The gate is scoped to residual undefinedness. With an operand fill
        configured the mask is cleared, and a reduction evaluates normally —
        proving 11c did not quietly ban reductions everywhere."""
        m = self._pair(sub=0)
        m.add_alias("d", "(S.v + x).cumsum()")
        got = self._mat(m)
        assert [int(v) for v in np.asarray(got)] == [13, 33]

    @pytest.mark.parametrize("expr,expected", [
        ("sqrt(S.v + x)", None),
        ("abs(S.v - x)", None),
        ("S.v + x * 2", None),
        ("(S.v > 1) & (x > 1)", None),
        ("-S.v + x", None),
    ])
    def test_b32_244_row_local_set_is_admitted(self, expr, expected):
        """The admitted operator families and ELEMENTWISE callables must
        actually evaluate, not merely be listed."""
        m = self._pair()
        m.add_alias("d", expr, fill_value=0)
        got = self._mat(m)
        assert len(got) == 2

    # ---- §6 no algebraic implicit fill -----------------------------------

    def test_b32_230_undefined_times_zero_stays_undefined(self):
        """§6, BINDING. The v1 design proposed accepting this because both
        probes agree. Not approved: ADF does not get to decide algebraically
        that an absent value became defined — that is the same 'choose a
        neutral value for the user' AD-19 forbids, arriving through arithmetic
        instead of through a default. The mask is union-propagated
        independently of operator semantics."""
        m = self._pair()
        m.add_alias("d", "S.v * 0", dtype="int64")
        with pytest.raises(ValueError) as ei:
            self._mat(m)
        assert "no defined value" in str(ei.value)

    def test_b32_230b_undefined_times_zero_is_resolved_by_the_alias_fill(self):
        """...and the final fill resolves it, at the final stage."""
        m = self._pair()
        m.add_alias("d", "S.v * 0", dtype="int64", fill_value=7)
        got = self._mat(m)
        assert [int(v) for v in got.values] == [0, 7]

    # ---- Decision B: transactional retraction ----------------------------

    def test_b32_232_placeholder_column_does_not_survive(self):
        """Decision B. On the pre-11c bytes `v__S` was left in the frame as a
        real int64 column holding the fabricated fill [3, 1]."""
        m = self._pair()
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        self._mat(m)
        assert "v__S" not in m.df.columns

    def test_b32_241_retraction_on_refusal(self):
        """Exit path 2 — no-fill refusal."""
        m = self._pair()
        m.add_alias("d", "S.v + x", dtype="int64")
        with pytest.raises(ValueError):
            self._mat(m)
        assert "v__S" not in m.df.columns

    def test_b32_242_retraction_on_provenance_refusal(self):
        """Exit path 3 — provenance/safety refusal."""
        m = self._pair()
        m.add_alias("d", "(S.v + x).cumsum()", fill_value=1)
        with pytest.raises(Exception):
            self._mat(m)
        assert "v__S" not in m.df.columns

    def test_b32_243_retraction_restores_a_preexisting_column(self):
        """Decision B: RESTORE, not merely remove, when the name pre-existed.

        Driven directly against `_retract_placeholder_columns`, DELIBERATELY
        and with the reason recorded. The first version of this test set
        `v__S` in the frame and materialized, which passed even with the
        retraction mutated away — because `_scatter_subframe_column`'s
        idempotent fast path returns the existing column, so the gather never
        runs and `_prior` is never non-None. That version asserted nothing.

        The restore branch is therefore currently UNREACHABLE through the
        integration path, and is retained because the panel required the
        semantics and because a future change to the fast path would make it
        live. Tested at the unit it belongs to, so the assertion is real."""
        m = self._pair()
        m.df["v__S"] = np.array([41, 42], dtype=np.int64)
        _M = _adf_module()
        ctx = _M._AliasEvalContext(alias_name="d", carry_mask=True)
        ctx.retracted.append(("v__S", m.df["v__S"].copy()))
        m.df["v__S"] = np.array([-1, -1], dtype=np.int64)
        m._retract_placeholder_columns(ctx)
        assert [int(v) for v in m.df["v__S"].values] == [41, 42]
        assert ctx.retracted == []

    def test_b32_243b_preexisting_joined_column_is_reused_by_the_fast_path(
            self):
        """Characterization, pre-existing and unchanged by 11c: if a column
        named like a joined temporary already exists, the scatter reuses it
        and no join is performed. Pinned here because b32_243's first version
        silently depended on it, and because the fast path's safety argument
        (see _scatter_subframe_column) rests on it."""
        m = self._pair()
        m.df["v__S"] = np.array([41, 42], dtype=np.int64)
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        got = self._mat(m)
        assert [int(v) for v in got.values] == [51, 62]
        assert [int(v) for v in m.df["v__S"].values] == [41, 42]

    def test_b32_247_probe_catches_a_reduction_when_the_gate_is_stubbed_open(
            self):
        """THE defense-in-depth assertion, and the honest one.

        With the fail-closed gate primary, the A/B probe can never fire for an
        expression the gate ADMITS — admitted expressions are row-local, and a
        row-local expression provably cannot propagate a placeholder into a
        defined row. So the probe's whole value is catching a GATE MISTAKE,
        and the only way to assert that value is to inject one.

        THE FIXTURE IS DELIBERATE, and the first version of this test got it
        wrong in an instructive way. With the UNDEFINED row LAST, a cumulative
        sum contaminates nothing that is defined, and the probe correctly sees
        no difference. Contamination is only observable when an undefined row
        precedes a defined one. That is direct evidence for the panel's ruling
        that the probe is not a proof: its sensitivity depends on WHERE the
        gaps fall, which no user controls.

        Also the mutation control for `_probe_b_values`: make probe B equal
        probe A and this test stops catching the reduction."""
        main = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([9, 0], np.int64), "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0, 1], np.int64)}))
        ch.df["v"] = np.array([5, 7], dtype=np.int64)
        main.register_subframe("S", ch, index_columns=["k"])
        main.add_alias("d", "(S.v + x).cumsum()", fill_value=1)
        _M = _adf_module()
        _orig = _M._expression_is_row_local
        _M._expression_is_row_local = lambda expr, *args, **kwargs: (True, None)
        try:
            with pytest.raises(Exception) as ei:
                self._mat(main)
        finally:
            _M._expression_is_row_local = _orig
        assert "placeholder" in str(ei.value), (
            "the gate was stubbed open, so the probe was the only thing left "
            "and it failed to catch a cumulative sum")

    def test_b32_245_real_data_joined_column_is_kept(self):
        """The retraction is scoped to PLACEHOLDER-bearing columns. An
        operand-fill-resolved column is real data and must survive exactly as
        it did before 11c — this is the control that stops the cleanup from
        becoming a behaviour regression."""
        m = self._pair(sub=0)
        m.add_alias("d", "S.v + x", dtype="int64")
        self._mat(m)
        assert "v__S" in m.df.columns
        assert [int(v) for v in m.df["v__S"].values] == [3, 0]

    # ---- placeholder policy ----------------------------------------------

    def test_b32_233_empty_child_frame_uses_the_dtype_default(self):
        """v1.4.4 P1-1. Nothing to borrow -> default-constructed value, and
        the decision still happens at the final stage, never in the gather."""
        main = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([7, 8], np.int64), "x": np.array([1, 2], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([], np.int64)}))
        ch.df["v"] = np.array([], dtype=np.int64)
        main.register_subframe("S", ch, index_columns=["k"])
        main.add_alias("d", "S.v + x", dtype="int64", fill_value=5)
        got = self._mat(main)
        assert [int(v) for v in got.values] == [5, 5]

    @pytest.mark.parametrize("child,dtype", [
        (np.array([127], dtype=np.int8), np.int8),
        (np.array([2 ** 63 - 1], dtype=np.int64), np.int64),
        (np.array([True], dtype=bool), bool),
    ])
    def test_b32_234_probe_b_never_overflows(self, child, dtype):
        """`A ^ 1`, deliberately not `A + 1`, which overflows at int8(127) and
        at 2**63-1. Both are live in this codebase."""
        vals = np.asarray([child[0], child[0]], dtype=dtype)
        out = A.AliasDataFrame._probe_b_values(
            pd.Series(vals), np.array([False, True]))
        assert str(out.dtype) == str(np.dtype(dtype))
        assert out.iloc[0] == child[0]
        assert out.iloc[1] != child[0]

    def test_b32_227_placeholder_is_not_observable(self):
        """Two different child values -> two different placeholders -> the
        SAME published result, because the placeholder is overwritten under
        the mask."""
        a = self._pair(child=np.array([3], np.int64))
        b = self._pair(child=np.array([987654321], np.int64))
        for m in (a, b):
            m.add_alias("d", "S.v * 0 + 4", dtype="int64", fill_value=1)
        assert ([int(v) for v in self._mat(a).values]
                == [int(v) for v in self._mat(b).values] == [4, 1])

    # ---- the moved refusal ------------------------------------------------

    def test_b32_235_moved_refusal_keeps_all_three_remedies(self):
        """The message moved five frames up; none of its user-facing guidance
        may be lost on the way."""
        m = self._pair()
        m.add_alias("d", "S.v + x", dtype="int64")
        with pytest.raises(ValueError) as ei:
            self._mat(m)
        msg = str(ei.value)
        assert "set_subframe_fill" in msg
        assert "set_global_fill" in msg
        assert "fill_value" in msg
        assert "'d'" in msg, "the refusal must name the alias"

    def test_b32_246_alias_fill_is_gone_from_the_operand_fill_config(self):
        """D_5 core deletes the round-10 shortcut. `_get_fill_config` must no
        longer see an alias-level fill under any circumstances — this is the
        structural assertion behind b32_195/196/197."""
        m = self._pair()
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        assert m._get_fill_config("S")["fill_missing"] is None
        assert not hasattr(m, "_active_alias_fill")


# ============================================================================
# CORRECTION ROUND 11d — D_4b: one evaluator, four public surfaces
#
# Round 11c gave the two MATERIALIZING entry points a shared contract. This
# increment brings the two NON-MATERIALIZING getters onto the same one, so
# the four public ways of evaluating an alias cannot drift apart again.
#   materialize_alias   materialize_aliases   get_alias_series   get_alias_array
# ============================================================================


class TestB32Round11dOneEvaluator:
    """AC_3: all four public surfaces share one evaluation contract."""

    SURFACES = ("materialize_alias", "materialize_aliases",
                "get_alias_series", "get_alias_array")

    @staticmethod
    def _pair(sub=None, child=None, keys=(0, 9), x=(10, 20)):
        main = A.AliasDataFrame(pd.DataFrame({
            "k": np.asarray(keys, dtype=np.int64),
            "x": np.asarray(x, dtype=np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = (np.array([3], dtype=np.int64) if child is None else child)
        main.register_subframe("S", ch, index_columns=["k"])
        if sub is not None:
            main.set_subframe_fill("S", fill_missing=sub)
        return main

    @classmethod
    def _evaluate(cls, m, surface, name="d"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if surface == "materialize_alias":
                m.materialize_alias(name);      return np.asarray(m.df[name])
            if surface == "materialize_aliases":
                m.materialize_aliases(names=[name]); return np.asarray(m.df[name])
            return np.asarray(getattr(m, surface)(name))

    @pytest.mark.parametrize("surface", SURFACES)
    def test_b32_248_compound_alias_fill_agrees_across_all_four_surfaces(
            self, surface):
        """The AR-3 result must be identical on every public surface. Before
        11c the materializing pair returned [13, 21]; before 11d the getters
        refused outright. One contract, one answer."""
        m = self._pair()
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        assert [int(v) for v in self._evaluate(m, surface)] == [13, 1]

    @pytest.mark.parametrize("surface", SURFACES)
    def test_b32_249_no_fill_refuses_on_all_four_surfaces(self, surface):
        """A refusal is part of the contract too. If one surface refuses and
        another quietly returns a number, the contract is not shared."""
        m = self._pair()
        m.add_alias("d", "S.v + x", dtype="int64")
        with pytest.raises(ValueError):
            self._evaluate(m, surface)

    @pytest.mark.parametrize("surface", SURFACES)
    def test_b32_250_provenance_gate_applies_on_all_four_surfaces(
            self, surface):
        """The fail-closed row-local gate must not be reachable-around by
        picking a different entry point."""
        m = self._pair()
        m.add_alias("d", "(S.v + x).cumsum()", fill_value=1)
        with pytest.raises(Exception) as ei:
            self._evaluate(m, surface)
        assert "row-local" in str(ei.value)

    @pytest.mark.parametrize("surface", SURFACES)
    def test_b32_251_operand_fill_precedence_holds_on_all_four_surfaces(
            self, surface):
        """AR-3 layering: an operand fill defines the operand and the alias
        fill stands down — on every surface."""
        m = self._pair(sub=0)
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        assert [int(v) for v in self._evaluate(m, surface)] == [13, 20]

    @pytest.mark.parametrize("surface", ["get_alias_series", "get_alias_array"])
    def test_b32_252_getters_retract_placeholder_columns_too(self, surface):
        """Decision B applies to the getters as well: no placeholder-bearing
        joined temporary may survive a getter call either."""
        m = self._pair()
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        self._evaluate(m, surface)
        assert "v__S" not in m.df.columns

    @pytest.mark.parametrize("surface", ["get_alias_series", "get_alias_array"])
    def test_b32_253_getters_still_do_not_publish_the_column(self, surface):
        """The ONE difference that must survive the unification: a getter
        evaluates without STORING. If sharing the evaluator had also made the
        getters publish, the whole point of a non-materializing surface would
        be gone.

        The alias deliberately carries NO declared dtype. The first version of
        this test used dtype="int64" and asserted no authority was recorded —
        which failed, correctly: a declared dtype is AD-19 SOURCE 3
        (explicit_alias) and is established by add_alias, long before any
        getter runs. What a getter must not create is SOURCE 4, first stored
        in-frame materialization. Conflating the two is the mistake this
        comment exists to stop the next reader repeating."""
        m = self._pair()
        m.add_alias("d", "S.v + x", fill_value=1)
        self._evaluate(m, surface)
        assert "d" not in m.df.columns, "a getter must not store the column"
        assert not m.get_dtype_authority("d").known, (
            "a getter must not commit source-4 authority")

    @pytest.mark.parametrize("surface", ["get_alias_series", "get_alias_array"])
    def test_b32_254_declared_dtype_authority_is_untouched_by_a_getter(
            self, surface):
        """The complement of b32_253: a DECLARED dtype (source 3) is
        established at add_alias time and a getter neither creates nor
        disturbs it."""
        m = self._pair()
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        before = m.get_dtype_authority("d")
        assert before.known and before.origin == "explicit_alias"
        self._evaluate(m, surface)
        after = m.get_dtype_authority("d")
        assert after.known and after.origin == "explicit_alias"
        assert str(after.dtype) == str(before.dtype)


# ============================================================================
# CORRECTION ROUND 11e — AC_6, and the publication/getter boundary
#
# Round-11d review adjudication (Main Reviewer GPT30, 4-5 split resolved on
# the merits): b32_203 asserts the STORED-PUBLICATION rule, which the
# ratified contract already specifies. The all-undefined GETTER question is a
# separate, genuinely open §5.4-vs-§9-step-10 ambiguity owned by B3.2b.
# The coder's re-anchor proposal answered the getter question with a test
# that never calls a getter, and was rejected.
#
# These tests pin the BOUNDARY, so a future increment cannot close the getter
# ambiguity by accident, and cannot reopen the publication rule by accident.
# ============================================================================


class TestB32Round11eAllUndefinedAuthority:

    @staticmethod
    def _pair(keys=(7, 9)):
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.asarray(keys, dtype=np.int64),
            "x": np.array([10, 20], dtype=np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        return m

    @pytest.mark.parametrize("bulk", [False, True])
    def test_b32_255_ac6_refuses_on_both_publishing_paths(self, bulk):
        """AC_6 must hold on the single AND the bulk publication path. b32_203
        exercises one of them; this is the parity control, because 'fixed one
        path and left the other' is the defect b32_197 already exists for."""
        m = self._pair()
        m.add_alias("d", "S.v", fill_value=0)
        with pytest.raises(ValueError) as ei:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if bulk:
                    m.materialize_aliases(names=["d"])
                else:
                    m.materialize_alias("d")
        assert "NO defined row" in str(ei.value)

    def test_b32_256_ac6_does_not_fire_when_one_row_is_defined(self):
        """The rule is ALL rows undefined. One observation is enough to infer
        from, so the ordinary residual-fill path applies and publication
        succeeds. Without this control the AC_6 branch could be written far
        too wide and every partial gap would start refusing."""
        m = self._pair(keys=(0, 9))          # row 0 matches
        m.add_alias("d", "S.v", fill_value=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("d")
        assert [int(v) for v in m.df["d"].values] == [3, 0]

    @pytest.mark.parametrize("getter", ["get_alias_series", "get_alias_array"])
    def test_b32_257_ac6_is_scoped_to_publication_not_to_the_getters(
            self, getter):
        """THE BOUNDARY. The all-undefined getter branch is an OPEN question
        assigned to B3.2b; AC_6 governs stored publication only. This test
        exists so that a future increment cannot silently answer the open
        question by widening the publication rule — if someone makes the
        getters refuse here, this fails and they have to go and get a ruling
        instead."""
        m = self._pair()
        m.add_alias("d", "S.v", fill_value=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.asarray(getattr(m, getter)("d"))
        assert [int(v) for v in out] == [0, 0]

    def test_b32_258_ac6_records_no_authority_after_refusing(self):
        """A refused publication must leave no trace: the point of AC_6 is
        that a policy constant never becomes source-4 authority."""
        m = self._pair()
        m.add_alias("d", "S.v", fill_value=0)
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_alias("d")
        assert "d" not in m.df.columns
        assert not m.get_dtype_authority("d").known
        assert "v__S" not in m.df.columns      # retraction still transactional


class TestB32Round11eAsymmetricKeyMaskInteraction:
    """ADF-EXT-001-T3 — the one characterization the round-11d panel required
    before the B3.2 tag (F11D-3).

    The AO2D/AI external report is a `draw()` / `draw_figures()` defect: two
    reviewer seats independently measured a loud KeyError there and correct
    values through `draw_batch`, so it is class C relative to B3.2 and is
    repaired in the B3.3 draw migration. What was NOT measured is whether
    asymmetric keys INTERACT with the round-11 residual-undefinedness
    machinery — a shape nobody had run. That is what this class pins.

    It lives here rather than in tests/test_phase_13_65_adf_asymmetric_keys.py
    because it characterizes ROUND-11 behaviour. The ADF-EXT-001 repair tests
    belong in the 13.65 file with the rest of the asymmetric-key contract.
    """

    @staticmethod
    def _asym():
        """GENUINELY asymmetric: the parent key column and the child key
        column have DIFFERENT NAMES, bound by `right_index_columns`
        (PHASE_13_65_ADF). `child_run=2` has no child row, so the join key is
        absent for exactly one parent row.

        THE FIRST VERSION OF THIS FIXTURE WAS NOT ASYMMETRIC AT ALL. It named
        both sides `run` and never passed `right_index_columns`, so the
        relation was symmetric by name and the extra `sector` column did
        nothing. The round-11e CRR nevertheless reported it as the executed
        ADF-EXT-001-T3 evidence. That was a false claim in the record — the
        one thing the standing round-4 bar rules out — and it is corrected
        here rather than quietly replaced (F11E-MR-P1-1; the fixture was
        inspected by GPT32 and GPT30, and three approving seats had accepted
        the description without checking the source)."""
        main = A.AliasDataFrame(pd.DataFrame({
            "parent_run": np.array([1, 1, 2], np.int64),
            "sector": np.array([0, 1, 0], np.int64),
            "x": np.array([10, 20, 30], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({
            "child_run": np.array([1], np.int64)}))
        ch.df["z"] = np.array([7], dtype=np.int64)
        main.register_subframe("C", ch,
                               index_columns=["parent_run"],
                               right_index_columns=["child_run"])
        return main

    @staticmethod
    def _mat(m, name="d"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias(name)
        return m.df[name]

    def test_b32_259_asymmetric_missing_key_refuses_without_a_fill(self):
        """The AD-19 refusal reaches the asymmetric shape unchanged."""
        m = self._asym()
        m.add_alias("d", "C.z + x", dtype="int64")
        with pytest.raises(ValueError):
            self._mat(m)

    def test_b32_260_asymmetric_alias_fill_is_still_final_result(self):
        """AR-3 across an asymmetric join. Oracle: C.z scatters to [7, 7, -],
        x = [10, 20, 30], so the two defined rows are 17 and 27 and the
        undefined row takes the alias fill — NOT fill + x."""
        m = self._asym()
        m.add_alias("d", "C.z + x", dtype="int64", fill_value=0)
        got = self._mat(m)
        assert str(got.dtype) == "int64"
        assert [int(v) for v in got.values] == [17, 27, 0]

    def test_b32_261_asymmetric_operand_fill_defines_the_operand(self):
        """The complementary layer: an operand fill of 0 makes the absent
        C.z a real 0, so the row is 0 + 30 = 30, not the alias fill."""
        m = self._asym()
        m.set_subframe_fill("C", fill_missing=0)
        m.add_alias("d", "C.z + x", dtype="int64", fill_value=99)
        assert [int(v) for v in self._mat(m).values] == [17, 27, 30]

    def test_b32_262_asymmetric_non_row_local_still_refuses(self):
        """The fail-closed provenance gate is not weakened by an asymmetric
        join — a surviving mask plus a reduction refuses here too."""
        m = self._asym()
        m.add_alias("d", "(C.z + x).cumsum()", fill_value=0)
        with pytest.raises(Exception) as ei:
            self._mat(m)
        assert "row-local" in str(ei.value)

    def test_b32_263_asymmetric_placeholder_does_not_survive(self):
        """Decision B holds on the asymmetric shape."""
        m = self._asym()
        m.add_alias("d", "C.z + x", dtype="int64", fill_value=0)
        self._mat(m)
        assert "z__C" not in m.df.columns


class TestB32Round11eRevisionAuthorityAndCoverage:
    """Round-11e revision, from the Main-Reviewer `[X]` summary.

    F11E-MR-P0-1  a RECORDED source-4 authority is a target contract too, so
                  an all-undefined rematerialization must publish in it, not
                  refuse. The first AC_6 predicate tested "no explicit dtype",
                  which is a different question.
    F11E-MR-P1-2  with no configured fill the ordinary no-fill refusal must
                  fire, not the fill-policy diagnostic.
    F11E-MR-P2-1  the bulk DEPENDENCY publication leg carries publishing=True
                  and had no direct AC_6 acceptance test.
    """

    @staticmethod
    def _frame(child_keys):
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 1], np.int64),
            "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.asarray(child_keys, np.int64)}))
        ch.df["v"] = np.arange(3, 3 + len(child_keys), dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        return m

    @staticmethod
    def _swap_to_all_missing(m):
        """Rebind the subframe to a child that matches nothing."""
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([77], np.int64)}))
        ch.df["v"] = np.array([9], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        m._join_index_cache.clear()
        if "v__S" in m.df.columns:
            del m.df["v__S"]

    @pytest.mark.parametrize("bulk", [False, True])
    def test_b32_264_existing_source4_authority_permits_all_undefined_remat(
            self, bulk):
        """F11E-MR-P0-1, the executed history GPT32 reported.

        No declared dtype, so the FIRST nonempty materialization establishes
        source-4 authority. After dematerializing, a relation that yields
        only undefined rows must still publish — the recorded dtype is the
        contract the fill is validated against. Refusing here would make a
        successful earlier measurement unusable."""
        m = self._frame([0, 1])
        m.add_alias("d", "S.v", fill_value=0)          # NO explicit dtype
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("d")
        auth = m.get_dtype_authority("d")
        assert auth.known and auth.origin == "first_stored_in_frame_materialization"
        assert str(auth.dtype) == "int64"

        m.dematerialize(drop=["d"])
        self._swap_to_all_missing(m)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bulk:
                m.materialize_aliases(names=["d"])
            else:
                m.materialize_alias("d")
        assert str(m.df["d"].dtype) == "int64"
        assert [int(v) for v in m.df["d"].values] == [0, 0]
        after = m.get_dtype_authority("d")
        assert after.known and str(after.dtype) == "int64", (
            "the recorded authority must survive the rematerialization")

    def test_b32_265_ac6_still_refuses_when_the_fill_is_the_only_basis(self):
        """The negative control for b32_264: without a prior successful
        materialization there is no recorded authority, so the fill really is
        the only possible basis and AC_6 refuses. If this ever passes, the
        P0-1 correction has been made too wide and AC_6 is dead."""
        m = self._frame([77])
        m.add_alias("d", "S.v", fill_value=0)
        with pytest.raises(ValueError) as ei:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_alias("d")
        assert "policy choice" in str(ei.value)

    def test_b32_266_no_fill_gets_the_ordinary_refusal_not_the_ac6_text(self):
        """F11E-MR-P1-2. All rows undefined and NO configured fill: the user
        must be told the value is undefined and how to configure one — not
        given a diagnostic about a fill they never set. Ordering fix: AC_6
        now runs after the no-fill refusal."""
        m = self._frame([77])
        m.add_alias("q", "S.v")                        # no dtype, no fill
        with pytest.raises(ValueError) as ei:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_alias("q")
        msg = str(ei.value)
        assert "no defined value" in msg
        assert "policy choice" not in msg, (
            "the fill-policy diagnostic must not fire when no fill is set")
        assert "set_subframe_fill" in msg and "set_global_fill" in msg

    def test_b32_267_ac6_covers_the_bulk_dependency_publication_leg(self):
        """F11E-MR-P2-1. `materialize_aliases` has a THIRD publishing path:
        a dependency alias carrying its own fill_value is evaluated and
        published inside the batch loop. It receives publishing=True, so AC_6
        must reach it — asserted directly rather than inferred from the two
        call sites that already had tests."""
        m = self._frame([77])
        m.add_alias("dep", "S.v", fill_value=0)        # no dtype -> AC_6 applies
        m.add_alias("top", "dep + x", dtype="int64")
        with pytest.raises(ValueError) as ei:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_aliases(names=["top"])
        assert "policy choice" in str(ei.value) or "no defined value" in str(ei.value)


class TestB32Round11eDrawBatchAsymmetricControl:
    """GPT27's recommended pre-tag control, added now so the Closure Report
    does not trigger another round.

    B3.2 claims the `draw_batch` surface. ADF-EXT-001 is a `draw()` /
    `draw_figures()` defect; the panel measured `draw_batch` as correct. This
    pins that end to end, on the genuinely asymmetric shape with a missing
    key, so the claim rests on a permanent test rather than on a one-off
    measurement in a review."""

    @staticmethod
    def _asym():
        main = A.AliasDataFrame(pd.DataFrame({
            "parent_run": np.array([1, 1, 2], np.int64),
            "x": np.array([10.0, 20.0, 30.0])}))
        ch = A.AliasDataFrame(pd.DataFrame({
            "child_run": np.array([1], np.int64)}))
        ch.df["z"] = np.array([7.0])
        main.register_subframe("C", ch,
                               index_columns=["parent_run"],
                               right_index_columns=["child_run"])
        return main

    def test_b32_268_draw_batch_asymmetric_missing_key_matches_the_oracle(
            self):
        """Decision 5 / AD-20 correction of the historical oracle.

        This is a draw_batch expression-value control, not a getter contract.
        The old test used get_alias_array("t") as truth and therefore pinned
        the superseded getter behavior.  The first Decision-5 rewrite then
        tried to route ``t`` through ``group_by``; that is also wrong because
        dfdraw Class-2 channels require a materialized column.

        Exercise the alias as the PLOTTED EXPRESSION with ``lazy=True`` so the
        ADF draw plan materializes ``t`` and delegates that real column to
        dfdraw.  Compare the delegated values against an independent fixture
        oracle: child z=7 joins parent_run=1, hence t=[17,27]; parent_run=2
        has no child key and remains NaN in the delegated floating frame.
        """
        import dfextensions.dfdraw as _dfd

        m = self._asym()
        m.add_alias("t", "C.z + x")
        seen, real = {}, _dfd.DFDraw.draw_batch

        def _spy(plotter, *args, **kwargs):
            seen["df"] = plotter.df.copy()
            return real(plotter, *args, **kwargs)

        _dfd.DFDraw.draw_batch = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = m.draw_batch(
                    {"p": {"expr": "t", "type": "hist", "bins": 3}},
                    lazy=True, verbose=False)
        finally:
            _dfd.DFDraw.draw_batch = real
            plt.close("all")

        assert res["_summary"]["failed"] == 0, res.get("_errors")
        assert "t" in seen["df"].columns, list(seen["df"].columns)
        vals = seen["df"]["t"].to_numpy(dtype=float)
        assert len(vals) == 3
        assert [float(v) for v in vals[:2]] == [17.0, 27.0], (
            "matched rows must equal the hand-computed C.z+x oracle")
        assert np.isnan(vals[2]), (
            "the unmatched parent key must remain undefined in the delegated "
            "draw_batch frame, not be fabricated by the getter policy")

    def test_b32_269_draw_batch_completes_on_the_asymmetric_shape(self):
        """The end-to-end control for the B3.2 claim: draw_batch does not
        raise on the shape that makes draw() raise (ADF-EXT-001)."""
        m = self._asym()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = m.draw_batch(
                {"a": {"expr": "C.z + x:x", "type": "scatter"}},
                lazy=False, verbose=False)
        plt.close("all")
        assert res["_summary"]["failed"] == 0, res.get("_errors")


class TestB32Round11fTargetDtypeResolver:
    """Round 11f — F11ER-MR-P1-1 (GPT29, upheld by the Main Reviewer).

    The final alias fill must be validated against the RESOLVED AUTHORITATIVE
    TARGET dtype, not against the provisional evaluated buffer the gather
    happened to produce. `b32_264` could not see this because it used int64
    on both sides, so authority and provisional agreed and the wrong
    validation target was invisible. Every test here varies that orthogonal
    dimension deliberately.
    """

    @staticmethod
    def _float_authority_then_int_provisional():
        """Establish source-4 float64 authority, then rebind to an int64
        child with no matching keys so the provisional buffer is int64."""
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 1], np.int64),
            "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0, 1], np.int64)}))
        ch.df["v"] = np.array([3.5, 4.5])                 # float64
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("d", "S.v", fill_value=0.5)           # NO declared dtype
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("d")
        assert str(m.get_dtype_authority("d").dtype) == "float64"
        m.dematerialize(drop=["d"])
        if "v__S" in m.df.columns:
            del m.df["v__S"]
        ch2 = A.AliasDataFrame(pd.DataFrame({"k": np.array([77], np.int64)}))
        ch2.df["v"] = np.array([9], dtype=np.int64)       # int64, no match
        m.register_subframe("S", ch2, index_columns=["k"])
        m._join_index_cache.clear()
        return m

    @pytest.mark.parametrize("bulk", [False, True])
    def test_b32_270_fill_is_validated_against_recorded_authority_not_buffer(
            self, bulk):
        """A and B of the required tests. Recorded authority float64,
        provisional int64, fill 0.5 -> publish float64 [0.5, 0.5]. Before 11f
        this refused, quoting int64 — a loud FALSE refusal."""
        m = self._float_authority_then_int_provisional()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bulk:
                m.materialize_aliases(names=["d"])
            else:
                m.materialize_alias("d")
        assert str(m.df["d"].dtype) == "float64"
        assert [float(v) for v in m.df["d"].values] == [0.5, 0.5]
        assert str(m.get_dtype_authority("d").dtype) == "float64", (
            "the recorded authority must be preserved, not replaced")

    def test_b32_271_explicit_target_governs_over_an_integer_buffer(self):
        """The source-3 stage-order control the review asked for. An explicit
        dtype='float64' with an integer provisional buffer must validate the
        fill against float64. This variant fails on the pre-11f bytes too, so
        the defect predates round 11e."""
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([7, 9], np.int64),
            "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("q", "S.v", dtype="float64", fill_value=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        assert str(m.df["q"].dtype) == "float64"
        assert [float(v) for v in m.df["q"].values] == [0.5, 0.5]

    def test_b32_272_incompatible_fill_still_refuses_against_the_target(self):
        """C of the required tests, and the negative control for the whole
        change: resolving the target must not become a way of ACCEPTING a
        fill the target cannot hold. 0.5 into an int64 target is destruction,
        and destruction is still refused."""
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([7, 9], np.int64),
            "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("r", "S.v", dtype="int64", fill_value=0.5)
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_alias("r")
        assert "r" not in m.df.columns
        assert "v__S" not in m.df.columns

    def test_b32_273_a_narrowing_target_still_works_when_values_fit(self):
        """NARROWING PRESERVATION CONTROL, and a correction to two of my own
        drafts — both recorded rather than quietly replaced.

        Draft one built a float64 child and asserted a refusal. It DID NOT
        RAISE, because a plain float child carries NaN natively, no mask
        survives, and step 10 is never reached: the test could not exercise
        the code it named. Same defect as the round-11e T3 fixture.

        Draft two assumed AR-1 permitted a value-changing int64 -> int8
        narrowing and asserted it published. Measured, `_safe_dtype_cast`
        already REFUSES it under AD-19, one layer above this code and long
        before 11f. So my first `_retarget_for_fill` guard was not stricter
        than the contract as I claimed — it was REDUNDANT with an existing
        guard. Removing it was still right (one owner per rule), but the
        stated reason was wrong and the comment now says so.

        What this test actually pins: when the values DO fit the narrower
        target, 11f changed nothing — the buffer is not promoted, the fill is
        validated against int8, and publication narrows as it always did."""
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64),
            "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([100], dtype=np.int64)      # fits int8
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("d", "S.v", dtype="int8", fill_value=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("d")
        assert str(m.df["d"].dtype) == "int8"
        assert [int(v) for v in m.df["d"].values] == [100, 0]

    def test_b32_273b_a_fill_the_target_cannot_hold_is_still_refused(self):
        """The fill contract is checked against the TARGET, not the
        provisional buffer. Decision 3 / AD-14 refuse a fill the target cannot
        represent; AR-1's permissive conversion governs COMPUTED data only."""
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64),
            "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("d", "S.v", dtype="int8", fill_value=300)
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_alias("d")
        assert "d" not in m.df.columns
        assert "v__S" not in m.df.columns

    def test_b32_273c_value_changing_narrowing_still_refused_by_its_owner(
            self):
        """The pre-existing AD-19 guard, pinned so 11f cannot be blamed for it
        and so a future refactor cannot delete it believing it duplicates the
        resolver. A defined value of 300 cannot become int8."""
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64),
            "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([300], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("d", "S.v", dtype="int8", fill_value=0)
        with pytest.raises(ValueError) as ei:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_alias("d")
        assert "dtype_cast" in str(ei.value) or "would change values" in str(ei.value)

    def test_b32_274_resolver_precedence_is_explicit_then_authority(self):
        """The resolver itself, unit-level. Explicit declaration outranks a
        recorded authority; authority outranks the provisional; provisional
        is the fallback."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.add_alias("a", "x * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("a")
        assert m.get_dtype_authority("a").known
        assert str(m._resolve_target_dtype("a", "float32", np.dtype("int8"))) \
            == "float32", "an explicit declaration must outrank authority"
        assert str(m._resolve_target_dtype("a", None, np.dtype("int8"))) \
            == "int64", "a recorded authority must outrank the provisional"
        assert str(m._resolve_target_dtype("nosuch", None, np.dtype("int8"))) \
            == "int8", "with neither, the provisional is the answer"


class TestB32Round11fDrawBatchDelegatedValues:
    """F11ER-MR-P2-1. The round-11e CRR called b32_268 a draw_batch value
    oracle; it is not — it evaluates through `get_alias_array`. Rather than
    only correcting the wording, this captures the values `draw_batch`
    ACTUALLY delegates to dfdraw and compares them to the oracle, which is
    what the claim should have rested on."""

    @staticmethod
    def _asym():
        main = A.AliasDataFrame(pd.DataFrame({
            "parent_run": np.array([1, 1, 2], np.int64),
            "x": np.array([10.0, 20.0, 30.0])}))
        ch = A.AliasDataFrame(pd.DataFrame({
            "child_run": np.array([1], np.int64)}))
        ch.df["z"] = np.array([7.0])
        main.register_subframe("C", ch,
                               index_columns=["parent_run"],
                               right_index_columns=["child_run"])
        return main

    def test_b32_275_draw_batch_delegates_the_oracle_values(self):
        """The real value oracle: run draw_batch, capture the frame handed to
        dfdraw, and require the delegated column to equal the evaluation
        truth row by row, NaN included."""
        m = self._asym()
        got = _delegated_frame(m, spec_slot="group_by", ref="C.z")
        vals = np.asarray(got, dtype=float)
        assert len(vals) == 3
        assert [float(v) for v in vals[:2]] == [7.0, 7.0], (
            "matched rows must carry the child value")
        assert np.isnan(vals[2]), (
            "the unmatched parent key must be delegated as undefined, "
            "not as a fabricated number")


class TestB32Round11fBufferPromotionUnit:
    """Direct unit coverage for `_buffer_dtype_for_fill`.

    WHY A UNIT TEST AND NOT ONLY AN INTEGRATION ONE — disclosed, because the
    mutation table would otherwise look complete when it is not. Disabling the
    promotion kills NO integration test on this sandbox, and that is a pandas
    VERSION artifact, not evidence that the promotion is dead code:

        pandas 1.5.3:  Series(int64)[mask] = 0.5  ->  upcasts to float64
                       so the promotion is redundant and the mutation is
                       invisible
        pandas 2 / 3:  the same setitem is deprecated / refuses, so the
                       promotion is load-bearing

    Reviewers run pandas 2.2.3 and 3.0.2; this sandbox runs 1.5.3. Version
    diversity has already found two P0s in this phase that seat diversity did
    not, so the behaviour is pinned at the unit here, where it holds on every
    version, rather than left to an integration test that only bites on some.
    """

    @staticmethod
    def _adf():
        return A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))

    def test_b32_276_widening_promotes_to_the_common_type(self):
        """int64 buffer, float64 target -> float64, so a fractional fill can
        be PLACED rather than truncated. This is F11ER-MR-P1-1's case."""
        got = self._adf()._buffer_dtype_for_fill(np.dtype("int64"),
                                                 np.dtype("float64"))
        assert str(got) == "float64"

    def test_b32_277_narrowing_leaves_the_buffer_alone(self):
        """int64 buffer, int8 target -> int64. The buffer is NOT narrowed
        here: `_safe_dtype_cast` owns the declared-dtype narrowing and already
        refuses a value-changing one under AD-19 (pinned by b32_273c). Two
        implementations of one rule is the divergence this phase keeps paying
        for."""
        got = self._adf()._buffer_dtype_for_fill(np.dtype("int64"),
                                                 np.dtype("int8"))
        assert str(got) == "int64"

    def test_b32_278_identical_dtypes_are_a_no_op(self):
        got = self._adf()._buffer_dtype_for_fill(np.dtype("int64"),
                                                 np.dtype("int64"))
        assert str(got) == "int64"

    def test_b32_279_uncombinable_dtypes_fall_back_to_the_provisional(self):
        """An extension dtype `np.result_type` cannot combine must return
        None, so `_retarget_for_fill` leaves the buffer as it was — the
        pre-11f behaviour — instead of raising from inside a helper."""
        got = self._adf()._buffer_dtype_for_fill(np.dtype("int64"),
                                                 pd.CategoricalDtype(["a"]))
        assert got is None


class TestB32Round11fValuePreservation:
    """F11F-P0-1 — the P0 that round 11f's first attempt INTRODUCED.

    Fixing a loud false refusal produced a silent one. `np.result_type` gives
    the COMMON type of two dtypes, not a type that can hold every value of
    both: `float64` is the common type of `int64` and `float64` and cannot
    represent an int64 above 2**53. Promoting the whole buffer therefore
    rounded a DEFINED measurement, and `_enforce_recorded_authority` could not
    see it because the loss had already happened inside the staging cast.

    Executed on the exact candidate by GPT32 and reproduced by the coder:

        2**60 + 1  published as  2**60

    Four reviewers found it independently while six approved the bytes. The
    approving seats proved that staging is NEEDED on pandas 2/3; none of them
    varied the defined value into the range where the staging is LOSSY. That
    distinction — need for staging versus losslessness of staging — is what
    this class pins.
    """

    BIG = 2 ** 60 + 1          # not representable in float64
    OK = 7                     # exactly representable

    @classmethod
    def _float_authority_then_int_data(cls, defined_value):
        """source-4 float64 authority, then a relation whose provisional
        buffer is int64 with ONE defined row and one missing row."""
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 1], np.int64),
            "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0, 1], np.int64)}))
        ch.df["v"] = np.array([3.5, 4.5])
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("d", "S.v", fill_value=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("d")
        assert str(m.get_dtype_authority("d").dtype) == "float64"
        m.dematerialize(drop=["d"])
        if "v__S" in m.df.columns:
            del m.df["v__S"]
        ch2 = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch2.df["v"] = np.array([defined_value], dtype=np.int64)
        m.register_subframe("S", ch2, index_columns=["k"])
        m.df["k"] = np.array([0, 9], np.int64)      # row 1 now unmatched
        m._join_index_cache.clear()
        return m

    @staticmethod
    def _run(m, bulk):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bulk:
                m.materialize_aliases(names=["d"])
            else:
                m.materialize_alias("d")

    @pytest.mark.parametrize("bulk", [False, True])
    def test_b32_280_large_defined_integer_is_never_silently_rounded(
            self, bulk):
        """11f-A and 11f-B. Fail closed: refuse rather than publish a rounded
        measurement. This is the direct regression test for the P0."""
        m = self._float_authority_then_int_data(self.BIG)
        with pytest.raises(ValueError) as ei:
            self._run(m, bulk)
        assert "defined" in str(ei.value)
        assert "d" not in m.df.columns, "nothing may be published"
        assert "v__S" not in m.df.columns, "retraction still transactional"

    @pytest.mark.parametrize("bulk", [False, True])
    def test_b32_281_exactly_representable_value_still_publishes(self, bulk):
        """11f-C, the positive control. The guard must refuse only what is
        actually lossy — otherwise it would make the 11f repair useless by
        refusing every widening."""
        m = self._float_authority_then_int_data(self.OK)
        self._run(m, bulk)
        assert str(m.df["d"].dtype) == "float64"
        vals = [float(v) for v in m.df["d"].values]
        assert vals == [7.0, 0.5]

    def test_b32_282_explicit_conversion_semantics_are_unchanged(self):
        """11f-D. The contrast case: with NO undefined row there is no mask,
        step 10 never runs, and an ordinary declared-dtype conversion keeps
        whatever semantics it always had. The guard is scoped to STAGING for
        a fill, and must not leak into ordinary conversion."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([self.BIG, 2],
                                                         np.int64)}))
        m.add_alias("q", "x", dtype="float64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        assert str(m.df["q"].dtype) == "float64"
        assert float(m.df["q"].values[0]) == float(np.float64(self.BIG))

    def test_b32_283_unit_the_guard_lives_in_retarget_not_in_the_helper(self):
        """11f-E's target, stated at the unit. `_buffer_dtype_for_fill` still
        answers `float64` for (int64, float64) — choosing the common type is
        correct and unchanged. The safety is `_retarget_for_fill`'s job, so a
        mutation that removes the round trip must break b32_280 while leaving
        the helper's own tests green."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        assert str(m._buffer_dtype_for_fill(np.dtype("int64"),
                                            np.dtype("float64"))) == "float64"

    def test_b32_284_extension_target_mismatch_fails_closed(self):
        """11f-F. `np.result_type` cannot combine a pandas extension dtype, so
        `_buffer_dtype_for_fill` returns None and `_retarget_for_fill` stages
        at the target itself under the same round-trip guard. A
        representation that cannot round-trip is REFUSED, never guessed."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        s = pd.Series(np.array([self.BIG, 5], dtype=np.int64))
        residual = np.array([False, True])
        with pytest.raises(ValueError):
            m._retarget_for_fill(s, pd.CategoricalDtype([1, 5]), residual, "q")


class TestB32Round11fSource3VsSource4Policy:
    """F11FC-MR-P1-1 — the decisive source-3 / source-4 contrast, on the SAME
    numerical shape.

    Round 11f's correction applied restoration exactness to every staging
    operation, including an EXPLICIT conversion the user asked for. The result
    was a conversion whose policy changed because a different row was missing:

        dtype="float64", defined int64 2**60+1
            no missing row      -> succeeded (ordinary backend conversion)
            one unrelated gap   -> REFUSED   (restoration exactness)

    GPT30 and GPT32 both flagged it; the Main Reviewer adjudicated that the
    ratified standards-first contract gives the two origins different rules.
    `b32_282` could not catch it — with no residual mask, step 10 never runs,
    so that test proves only that the guard does not affect calls that never
    reach it. Same test-construction defect I documented for the first
    `b32_273` draft, a third time.
    """

    BIG = 2 ** 60 + 1

    @classmethod
    def _partial_missing(cls, defined_value=None):
        """One matched row carrying a large int64, one unmatched row."""
        v = cls.BIG if defined_value is None else defined_value
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64),
            "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([v], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        return m

    @staticmethod
    def _mat(m, name, bulk):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bulk:
                m.materialize_aliases(names=[name])
            else:
                m.materialize_alias(name)

    # ---- T1 / T3 : EXPLICIT source-3 conversion --------------------------

    @pytest.mark.parametrize("bulk", [False, True])
    def test_b32_285_explicit_target_uses_ordinary_conversion(self, bulk):
        """T1 and T3. The user explicitly asked for float64. AR-1
        standards-first: the defined row follows the backend's ordinary
        conversion semantics, and the missing row takes the fill. This must
        give the SAME defined value as the no-missing-row case."""
        m = self._partial_missing()
        m.add_alias("q", "S.v", dtype="float64", fill_value=0.5)
        self._mat(m, "q", bulk)
        assert str(m.df["q"].dtype) == "float64"
        got = [float(v) for v in m.df["q"].values]
        assert got[0] == float(np.float64(self.BIG)), (
            "an explicit conversion must not change policy because another "
            "row is missing")
        assert got[1] == 0.5

    def test_b32_286_explicit_conversion_agrees_with_the_no_gap_case(self):
        """The invariant stated directly: the defined row's published value is
        identical whether or not some other row is undefined."""
        with_gap = self._partial_missing()
        with_gap.add_alias("q", "S.v", dtype="float64", fill_value=0.5)
        self._mat(with_gap, "q", False)

        no_gap = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0], np.int64), "x": np.array([10], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([self.BIG], dtype=np.int64)
        no_gap.register_subframe("S", ch, index_columns=["k"])
        no_gap.add_alias("q", "S.v", dtype="float64", fill_value=0.5)
        self._mat(no_gap, "q", False)

        assert float(with_gap.df["q"].values[0]) == float(no_gap.df["q"].values[0])

    # ---- T2 : RECORDED source-4 authority contrast ------------------------

    @pytest.mark.parametrize("bulk", [False, True])
    def test_b32_287_recorded_authority_still_refuses_on_the_same_shape(
            self, bulk):
        """T2 and T3. IDENTICAL numerical shape, but the target comes from a
        RECORDED source-4 authority rather than an explicit declaration.
        Nobody asked for this conversion — ADF is reconstructing a dtype it
        established earlier — so a defined value that cannot survive exactly
        REFUSES. This pair is the whole point: same numbers, different
        origin, different contract."""
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 1], np.int64),
            "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0, 1], np.int64)}))
        ch.df["v"] = np.array([3.5, 4.5])
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("q", "S.v", fill_value=0.5)          # NO explicit dtype
        self._mat(m, "q", False)
        assert str(m.get_dtype_authority("q").dtype) == "float64"
        m.dematerialize(drop=["q"])
        if "v__S" in m.df.columns:
            del m.df["v__S"]
        ch2 = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch2.df["v"] = np.array([self.BIG], dtype=np.int64)
        m.register_subframe("S", ch2, index_columns=["k"])
        m.df["k"] = np.array([0, 9], np.int64)
        m._join_index_cache.clear()
        with pytest.raises(ValueError) as ei:
            self._mat(m, "q", bulk)
        assert "defined" in str(ei.value)

    def test_b32_288_unit_the_origin_predicate(self):
        """The predicate itself. Explicit declaration -> not restoration;
        recorded authority with no declaration -> restoration; neither ->
        not restoration (AC_6 owns that case)."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.add_alias("a", "x * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("a")
        assert m._target_dtype_is_restoration("a", None) is True
        assert m._target_dtype_is_restoration("a", "float64") is False
        assert m._target_dtype_is_restoration("nosuch", None) is False


class TestB32Round11fNullableExtensionTarget:
    """F11FC-MR-P1-2 — the public nullable-floating control.

    `b32_284` proves the fail-closed fallback at the private helper with a
    categorical dtype. The Main Reviewer asked for the NUMERIC nullable family
    through a PUBLIC materialization path, because that is the storage family
    closest to the production promotion path and the one that motivated the
    pandas 2/3 concern. `np.result_type` cannot combine `int64` with
    `pd.Float64Dtype()`, so these exercise the stage-at-the-target branch for
    real rather than at the unit.
    """

    BIG = 2 ** 60 + 1

    @staticmethod
    def _frame(defined_value):
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64),
            "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([defined_value], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        return m

    @pytest.mark.parametrize("target", ["Float64", "Float32"])
    @pytest.mark.parametrize("bulk", [False, True])
    def test_b32_289_nullable_float_target_through_the_public_path(
            self, target, bulk):
        """T4. A pandas nullable float target, an exactly representable
        defined value, a residual missing row and a fractional fill — through
        `materialize_alias` / `materialize_aliases`, not through a helper.
        The extension dtype must be retained, the defined value correct, the
        fill applied, and no raw incompatible setitem relied upon."""
        m = self._frame(7)
        m.add_alias("q", "S.v", dtype=target, fill_value=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if bulk:
                m.materialize_aliases(names=["q"])
            else:
                m.materialize_alias("q")
        assert str(m.df["q"].dtype) == target, "the extension target is kept"
        assert float(m.df["q"].iloc[0]) == 7.0
        assert float(m.df["q"].iloc[1]) == 0.5

    def test_b32_290_nullable_float_explicit_target_follows_ar1(self):
        """The nullable family obeys the same source-3 rule as the NumPy one:
        an explicit target converts the defined row by ordinary semantics
        rather than refusing."""
        m = self._frame(self.BIG)
        m.add_alias("q", "S.v", dtype="Float64", fill_value=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        assert str(m.df["q"].dtype) == "Float64"
        assert float(m.df["q"].iloc[1]) == 0.5


class TestB32Round11fExtensionDtypeDeclaration:
    """A PRE-EXISTING defect this increment did not create, found by the
    round-11f review's own requirement.

    `_safe_dtype_cast` called `np.dtype(target_dtype)` unconditionally, and
    `np.dtype("Float64")` raises. So EVERY declared pandas extension dtype has
    been unusable on the alias-cast path since it was written. Measured on the
    simplest possible case — no subframe, no mask, no round-11 code:

        add_alias("q", "x * 2", dtype="Float64")  ->  TypeError

    It surfaced only because F11FC-MR-P1-2 demanded a PUBLIC nullable-float
    control and no such test existed. The fix is reached only for dtypes on
    which the old line RAISED, so nothing that worked before can change.
    """

    @pytest.mark.parametrize("dt", ["Float64", "Float32", "Int64", "boolean"])
    def test_b32_291_declared_extension_dtype_is_usable_at_all(self, dt):
        """The minimal reproduction, with no round-11 machinery on the path."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        expr = "x > 1" if dt == "boolean" else "x * 2"
        m.add_alias("q", expr, dtype=dt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        assert str(m.df["q"].dtype) == dt

    def test_b32_292_numpy_target_path_is_untouched(self):
        """The regression control for the guard: a NumPy target must not enter
        the new branch at all, because `np.dtype` succeeds for it."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.add_alias("q", "x * 2", dtype="int32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        assert str(m.df["q"].dtype) == "int32"
        assert [int(v) for v in m.df["q"].values] == [2, 4]

    def test_b32_293_an_impossible_extension_conversion_still_raises(self):
        """Fail closed: the new branch must not turn an impossible conversion
        into a silent success."""
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.register_function("as_text", lambda v: np.array(["a", "b"]))
        m.add_alias("q", "as_text(x)", dtype="Int64")
        with pytest.raises((TypeError, ValueError)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_alias("q")


# ============================================================================
# INCREMENT B3.2b — ACCEPTANCE SCAFFOLD  (revision 3)
#
# B3.2 closed with named exclusions under the architect's Rule-18 decision of
# 2026-08-12. B3.2b is MANDATORY BEFORE B3.3 (AD-12).
#
# WHY THIS EXISTS BEFORE ANY B3.2b CODE. B3.2's closure criterion was
# mechanical — "no strict xfail names B3.2" — and that is the only reason the
# stage closed by a command instead of an argument. At the B3.2 tag, B3.2b had
# ZERO acceptance tests.
#
# REVISION 3, after the Main Reviewer rejected revision 2 ([X], GPT26,
# 9 submissions / 7 seats, 4 [OK] and 3 [X]). The panel agreed revision 2
# fixed everything revision 1 got wrong; the rejection is that the resulting
# GATE was not yet sufficient to be the definition of done:
#
#   MR-P0-1  b32b_15b asserted `not re.search("densify-fill-resparsify")`
#            over the module source. RENAMING A COMMENT would have turned it
#            green with zero executable change — in the same file whose
#            header adopts the rule that a strict xfail must fail for the
#            defect its reason names. Replaced with an executable spy: the
#            dense conversion is forbidden at runtime and _place_fill is
#            driven directly. A comment cannot satisfy it.
#   MR-P0-2  named owners were one specimen per family, so a family could
#            close after a single case. Reader disagreement, the Arrow
#            matrix, the persistent-creator inventory and the non-`where`
#            fallback forms are now enumerated.
#   MR-P0-3  PLAN_GROUPS omitted normative Rev-2 §11.3 concepts. `logical
#            requirements`, `group materializations` and `slot/surface
#            provenance` are added; the map is concept -> implementation,
#            never identical spelling.
#   MR-P1-4  b32b_9's remaining failure belongs to ordinary D_3 conversion
#            semantics, not to the strict helper, which already works.
#            Split: the helper route is a PASSING characterization.
#   MR-P1-5  the all-undefined getter is ruled by AD-20 (below) and is now
#            family 10, with its already-true clauses pinned as passing
#            controls and its two open clauses as strict xfails.
#   MR-P2-1  the cast whitelist and the disposition marker carry recorded,
#            adjudicated reasons instead of bare names.
#   MR-P2-2  b32b_11's two production calls are separated so the intended
#            failure point cannot silently move.
#
# AUTHORING RULE, adopted permanently in revision 2 and applied again here: a
# strict xfail is not acceptance evidence until it has been run with xfail
# disabled and its failure shown to be the contract defect named in its
# reason string. Every `Measured baseline failure:` line below is the text of
# an actual --runxfail run on the tagged bytes, EXCEPT the two family-2 tests,
# which cannot execute in the coder sandbox and are disclosed as such.
#
# ---------------------------------------------------------------------------
# AD-20 — ALL-UNDEFINED NON-MATERIALIZING GETTER   [STEP-0 GATING RULING]
# Ruled by the architect 2026-08-12, resolving the §5.4-vs-§9-step-10
# ambiguity that AliasDataFrame.py:9611 records as owned by B3.2b:
#
#   A non-materializing getter MAY return an ephemeral filled result when
#   configured handling fully resolves the undefined rows. It MUST NOT
#   publish the requested alias and MUST NOT create AD-19 source-4
#   authority. Existing explicit or recorded authority governs the returned
#   dtype; with no authority, the backend-natural dtype may be returned
#   ephemerally. If residual undefinedness remains unresolved, refuse
#   clearly.
#
# Architect's note, recorded: real workflows may revisit this once a use case
# that needs the ephemeral-fill semantics appears.
#
# MEASURED against the tagged bytes before scaffolding (not assumed from the
# reviewers' text — the coder's first report that "current behaviour already
# satisfies it" was half wrong):
#
#   clause 1  no publication of the requested alias      ALREADY TRUE
#   clause 2  no source-4 authority created              ALREADY TRUE
#   clause 3  declared/recorded authority governs dtype  ALREADY TRUE
#   clause 4  ephemeral fill, else clear refusal         NOT TRUE — the
#             getter ignores the alias-level fill_value that
#             materialize_alias honours (nan vs 0.0 for the same alias),
#             and returns silent NaN where the ruling requires a refusal.
# ============================================================================

#: The ratified B3.2b work families. Every one must have an owner below;
#: `test_b32b_0_every_ratified_family_has_an_owner` enforces that from the
#: AST, so a family cannot be forgotten the way five of them were in
#: revision 1, and a docstring mentioning a name cannot satisfy it.
B32B_SCOPE = {
    "1-reader-metadata":         ["b32b_5", "b32b_5b", "b32b_5c",
                                  "b32b_5d", "b32b_5e"],
    "2-persisted-dtype-origin":  ["b32b_13", "b32b_13b", "b32b_13c",
                                  "b32b_13d", "b32b_13e"],
    "3-arrow-coverage":          ["b32b_14", "b32b_14b", "b32b_14c",
                                  "b32b_14d", "b32b_14e", "b32b_14f",
                                  "b32b_14g", "b32b_14h"],
    "4-sparse-no-dense-temp":    ["b32b_15", "b32b_15b"],
    "5-conditional-provenance":  ["b32b_16", "b32b_16b", "b32b_16c"],
    "6-cast-site-audit":         ["b32b_12"],
    "7-persistent-column-audit": ["b32b_4", "b32b_4b", "b32b_4c", "b32b_4d"],
    "8-conversion-api-audit":    ["b32b_17"],
    "9-rev2-dependency-plan":    ["b32b_1", "b32b_2", "b32b_2b"],
    "10-all-undefined-getter":   ["b32b_18", "b32b_18b", "b32b_18c",
                                  "b32b_18d", "b32b_18e"],
}


def _adf_source_text():
    import inspect
    return inspect.getsource(_adf_module())


def _write_tree_dtyped(path, dtype_x, n=4):
    """`_write_tree` fixes both branches at float64. The reader-metadata
    family needs branches that DISAGREE across a chain, which needs a
    per-branch dtype."""
    with uproot.recreate(path) as f:
        f.mktree("tree", {"x": dtype_x, "w": "float64"})
        f["tree"].extend({"x": np.arange(n).astype(dtype_x),
                          "w": np.ones(n, dtype=np.float64)})


def _arrow_env():
    """The environment, for every family-3 failure message.

    REVISION 3d. The architect reported the Arrow tests as intermittently
    failing. Without the versions and the string_storage option in the
    message, a divergence between his pandas 1.5.3 and a reviewer's 2.2.3 or
    3.0.2 is indistinguishable from a real regression, which is what made it
    look like flakiness.
    """
    try:
        import pyarrow as _pa
        pa_v = _pa.__version__
    except Exception:
        pa_v = "absent"
    return (f"pandas={pd.__version__} pyarrow={pa_v} "
            f"string_storage={pd.options.mode.string_storage}")


def _is_arrow_backed(dtype):
    """Structural, never textual — and it must recognise BOTH mechanisms.

    REVISION 3c/3d CORRECTION. Revision 3b tested `"pyarrow" in str(dtype)`
    and concluded that `string[pyarrow]` was 'silently downgraded'. FALSE:
    pandas prints `str(StringDtype)` as 'string' for every storage. Measured
    — the gathered dtype compares EQUAL to the child dtype.

    pandas has two separate Arrow mechanisms and they are different classes:
        pd.ArrowDtype                a real Arrow type   (int64[pyarrow])
        pd.StringDtype(storage=...)  a pandas string stored in Arrow
    `isinstance(dt, pd.ArrowDtype)` is False for the second. Treating them as
    one family is precisely the mistake that produced the false finding.
    """
    if isinstance(dtype, getattr(pd, "ArrowDtype", ())):
        return True
    return getattr(dtype, "storage", None) == "pyarrow"


def _arrow_build(dtype, values):
    """Capability probe. Returns the array, or None if THIS pandas/pyarrow
    cannot build the specimen — so an unsupported combination is reported as
    a named gap rather than crashing the test or vanishing silently.

    The sentinel "ARROW_STRING" asks for a genuinely Arrow-TYPED string,
    which must be constructed through `pd.ArrowDtype(pa.string())`;
    `pd.ArrowDtype.construct_from_string("string[pyarrow]")` raises, because
    that spelling belongs to StringDtype.
    """
    try:
        if dtype == "ARROW_STRING":
            import pyarrow as _pa
            return pd.array(values, dtype=pd.ArrowDtype(_pa.string()))
        return pd.array(values, dtype=dtype)
    except Exception:
        return None


def _arrow_specimens():
    """One specimen per Arrow MECHANISM, skipping whatever this environment
    cannot build. Keys are labels used in failure messages."""
    out = {}
    typed = _arrow_build("double[pyarrow]", [1.5])
    if typed is not None:
        out["arrow-typed double"] = typed
    arrow_str = _arrow_build("ARROW_STRING", ["a"])
    if arrow_str is not None:
        out["arrow-typed string"] = arrow_str
    storage = _arrow_build("string[pyarrow]", ["a"])
    if storage is not None:
        out["arrow-storage string"] = storage
    return out


def _arrow_neutral(dtype):
    """A fill value of the right shape for a specimen's dtype.

    B3.2b STEP 5 CORRECTION. This began with an arrow-backed branch that
    returned `""` for ANY dtype whose `.storage` is `"pyarrow"` — which is
    every Arrow specimen, `double[pyarrow]` included. So the double specimen
    was filled with a STRING, `_coerce_fill_to_dtype` correctly refused it,
    and `b32b_14f` swallowed that refusal through a `getattr(mod, "ADFError",
    ValueError)` fallback that resolved to `ValueError` while `ADFError` did
    not exist. The test passed without ever reaching its own assertion for
    that specimen. STEP 5 created `ADFError`, the fallback stopped resolving
    to `ValueError`, and the fixture defect surfaced — found by the identity
    diff, not by reading.

    The value type decides, and nothing else does.
    """
    kind = str(getattr(dtype, "kind", "")) or str(dtype)
    if "string" in str(dtype) or "str" in kind:
        return ""
    if "bool" in str(dtype):
        return False
    if "int" in str(dtype):
        return 0
    return 0.0


class TestB32bAcceptanceScaffold:
    """The machine-readable definition of 'B3.2b is finished'."""

    # ---- the scope guard --------------------------------------------------

    def test_b32b_0_every_ratified_family_has_an_owner(self):
        """Not an xfail — this passes today and must keep passing. It is the
        guard revision 1 lacked: five of the nine ratified families had no
        test at all and nothing detected that.

        AST, not source text (MR-P2, GPT27): revision 2 searched for the
        string `def test_<owner>_`, which a docstring, a comment or a
        commented-out definition satisfies. Only a real FunctionDef counts."""
        import ast as _ast
        tree = _ast.parse(_adf_scaffold_text())
        defined = {n.name for n in _ast.walk(tree)
                   if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
        for family, owners in B32B_SCOPE.items():
            for owner in owners:
                assert any(d.startswith(f"test_{owner}_") for d in defined), (
                    f"family {family} names owner {owner}, which is not a "
                    f"defined test function")

    # ---- family 9: the Rev-2 dependency plan (AD-12) ----------------------

    #: The normative Rev-2 §11.3 plan concepts. MR-P0-3: revision 2 carried
    #: nine and omitted `logical requirements`, explicit group materialization
    #: and `slot/surface provenance`, so b32b_1 could pass against an
    #: incomplete schema. Field names in the implementation may differ; the
    #: mapping below is the contract, the spelling is not.
    PLAN_GROUPS = ("logical_requirements",     # what the call REQUIRES, pre-physical
                   "branches",                 # physical branch reads
                   "aliases",                  # alias materializations
                   "group_materializations",   # vector-slot / group expansion
                   "structs",
                   "subframes",
                   "joins",
                   "temporary_columns",        # expected temporary writes
                   "persistent_columns",       # expected persistent writes
                   "cache_effects",            # cache effects / invalidations
                   "cleanup",                  # cleanup candidates
                   "slot_surface_provenance")  # which slot/surface asked for it

    #: The scaffold's OWN expected disposition per group — P0-STEP1-1.
    #:
    #: STEP 1 v01 carried a flat `PLAN_TO_STATE` that paired every group with
    #: some observed field, and three seats found that several pairings
    #: conflate different semantic axes: slot/surface provenance is not
    #: executor-stage attribution, a join is not a projection read, and a
    #: logical requirement is not a physical requested read. The contract is
    #: now a DISPOSITION, and a group that has no measured counterpart says so
    #: instead of borrowing an unrelated non-empty field.
    #:
    #: This is written INDEPENDENTLY of production and `b32b_2` checks
    #: production against it — v01 only validated the scaffold's own copy,
    #: so production could have been wrong with every test still green
    #: (P0-STEP1-2, anti-drift half).
    EXPECTED_DISPOSITION = {
        "logical_requirements":    ("PLAN_ONLY", ()),
        "branches":                ("STATE", ("branches_loaded",)),
        "aliases":                 ("STATE", ("aliases_materialized",)),
        "group_materializations":  ("STATE", ("aliases_by_projection",
                                              "projection_columns")),
        "structs":                 ("STATE", ("structs_completed",
                                              "struct_members_present")),
        "subframes":               ("PENDING", ()),
        "joins":                   ("PENDING", ()),
        "temporary_columns":       ("STATE", ("temporary_columns",)),
        "persistent_columns":      ("STATE", ("columns_created",)),
        "cache_effects":           ("STATE", ("cache_effects",)),
        "cleanup":                 ("STATE", ("cleanup_candidates",
                                              "aliases_dropped",
                                              "cleanup_outcome")),
        "slot_surface_provenance": ("PLAN_ONLY", ()),
    }

    #: Groups whose planned items must reconcile item-by-item at STEP 9.
    #: Only STATE groups can: PLAN_ONLY has no observation by construction and
    #: PENDING has none yet.
    RECONCILABLE_GROUPS = tuple(
        g for g, (kind, _) in EXPECTED_DISPOSITION.items() if kind == "STATE")

    def test_b32b_1_plan_carries_the_full_rev2_contract(self):
        """CLOSED BY B3.2b STEP 1 — was a strict xfail, now passing.

        The plan carried 5 slots at the B3.2 tag (especs, rewrite_dicts,
        autoload_dicts, merged_specs, lazy) against a 26-field observation
        record. STEP 1 adds the twelve normative Rev-2 §11.3 groups as
        SCHEMA; STEP 9 populates them, and `b32b_2b` is what refuses to let
        the schema alone count as reconciliation.

        The concept list below is this scaffold's OWN expectation and is
        deliberately not read from production, so the contract and the code
        cannot drift together — that drift is how three §11.3 concepts went
        missing for two revisions."""
        plan_cls = getattr(_adf_module(), "_DrawDependencyPlan")
        slots = set(getattr(plan_cls, "__slots__", ()))
        missing = [g for g in self.PLAN_GROUPS if g not in slots]
        assert not missing, f"plan is missing Rev-2 groups: {missing}"

    def test_b32b_2_plan_intent_reconciles_with_state_observation(self):
        """CLOSED BY B3.2b STEP 1 — was a strict xfail, now passing.

        P0-STEP1-2, anti-drift half. v01 validated only the scaffold's own
        map, so a semantically wrong production map passed every test. This
        checks PRODUCTION's `REV2_GROUP_DISPOSITION` against the scaffold's
        independent expectation, field by field.

        The contract it enforces:
          STATE      names measured field(s) that EXIST on the record
          PLAN_ONLY  names NO state field — reconciled inside the plan
          PENDING    names NO state field and names its owning STEP
        and every entry carries an adjudicated reason, so a disposition
        cannot be changed silently."""
        mod = _adf_module()
        plan_cls = mod._DrawDependencyPlan
        state = mod._DrawPreparationState()

        prod = getattr(plan_cls, "REV2_GROUP_DISPOSITION", None)
        assert prod is not None, (
            "production declares no Rev-2 group disposition")
        assert set(prod) == set(self.EXPECTED_DISPOSITION), (
            f"production groups {sorted(set(prod) ^ set(self.EXPECTED_DISPOSITION))} "
            f"differ from the scaffold's expectation")

        for group, (want_kind, want_fields) in self.EXPECTED_DISPOSITION.items():
            kind, fields, reason = prod[group]
            assert kind == want_kind, (
                f"{group}: production says {kind}, contract says {want_kind}")
            assert tuple(fields) == tuple(want_fields), (
                f"{group}: production counterparts {tuple(fields)} differ "
                f"from the contract's {tuple(want_fields)}")
            assert isinstance(reason, str) and len(reason) >= 30, (
                f"{group}: disposition carries no adjudicated reason")

            if kind == "STATE":
                assert fields, f"{group}: STATE with no counterpart"
                for name in fields:
                    assert hasattr(state, name), (
                        f"{group}: names {name!r}, which does not exist on "
                        f"_DrawPreparationState — a counterpart that is not "
                        f"there cannot be reconciled against")
            else:
                assert not fields, (
                    f"{group}: {kind} must name NO state field; naming one is "
                    f"the conflation P0-STEP1-1 removed")
            if kind == "PENDING":
                assert "STEP" in reason, (
                    f"{group}: PENDING must name the step that owes the "
                    f"measured counterpart")

        derived = getattr(plan_cls, "REV2_GROUP_TO_STATE", {})
        assert set(derived) == set(self.RECONCILABLE_GROUPS), (
            "the derived STATE-only map must expose exactly the groups that "
            "have a measured counterpart")

    @needs_dfdraw
    @pytest.mark.xfail(strict=True, reason=
        "B3.2b acceptance, family 9, the STEP-9 half (MR-P1-1, GPT32 R3D "
        "blocker A, and P0-STEP1-2 item-level half): a representative plan "
        "must be BUILT, EXECUTED through the real draw_batch path, and its "
        "planned items matched ITEM BY ITEM against the measured "
        "_DrawPreparationState. Measured baseline failure after STEP 1: "
        "'the executed plan carries no reconcilable requirement' — every "
        "Rev-2 group is empty because STEP 1 delivers the schema and STEP 9 "
        "populates it. This cannot XPASS from the schema alone, and it "
        "cannot XPASS from an unrelated observed field being non-empty.")
    def test_b32b_2b_plan_reconciles_against_executed_state(self):
        """GPT32 R3D blocker A, plus the item-level half of P0-STEP1-2.

        Two false-close mechanisms have been removed from this criterion:

        1. v00 read `__slots__` and probed an empty state for attribute
           existence, so STEP 1's schema alone would have made it XPASS.
        2. v01 executed a real draw but reconciled a group whenever its
           mapped observed field was NON-EMPTY. `planned branch x` against
           `observed branch y` would have passed. GPT29 and GPT32 both
           called that not-reconciliation, and they are right.

        The predicate is now membership: every planned item must be FOUND
        among the measured items of its counterpart fields. Only STATE
        groups are reconcilable — PLAN_ONLY has no observation by
        construction, PENDING has none yet — and asserting otherwise would
        reintroduce exactly the conflation P0-STEP1-1 removed."""
        adf = _mini_adf()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.draw_batch({"p": {"expr": "x", "type": "hist"}}, verbose=False)

        plan = getattr(adf, "_last_draw_plan", None)
        state = getattr(adf, "_last_draw_prep_state", None)
        assert plan is not None, "the executed plan was not retained"
        assert state is not None, "no preparation record was produced"

        def _items(value):
            if value is None:
                return []
            if isinstance(value, dict):
                return list(value)
            if isinstance(value, (list, tuple, set, frozenset)):
                return list(value)
            return [value]

        populated = [g for g in self.RECONCILABLE_GROUPS
                     if getattr(plan, g, None)]
        assert populated, (
            "the executed plan carries no reconcilable requirement, so there "
            "is nothing to match against the measured record (STEP 1 "
            "delivers the schema; STEP 9 populates it)")

        unreconciled = []
        for group in populated:
            planned = _items(getattr(plan, group))
            measured = []
            for name in self.EXPECTED_DISPOSITION[group][1]:
                measured.extend(_items(getattr(state, name, None)))
            missing = [item for item in planned if item not in measured]
            if missing:
                unreconciled.append(
                    f"{group}: planned {missing[:4]} not found among the "
                    f"measured {measured[:6]}")
        assert not unreconciled, (
            "planned items with no measured counterpart: "
            + "; ".join(unreconciled))

    # ---- family 7: ADF-created PERSISTENT columns (AD-19 source 5) --------

    def test_b32b_4_persistent_adf_created_column_is_an_authority_source(self):
        """P0-3 correction. Revision 1 used the joined temporary `v__S`, but
        the ratified text says a temporary working column is NOT an authority,
        and 11c retracts placeholder-bearing temporaries precisely so they
        cannot masquerade as data. Asserting source 5 on a temporary would
        have pushed B3.2b to undo that."""
        m = A.AliasDataFrame(pd.DataFrame({
            "dy": np.array([1.5, 2.5, 3.5])}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.compress_columns({"dy": {
                "compress": "round(dy*10)",
                "decompress": "dy_c/10.",
                "compressed_dtype": np.int16,
                "decompressed_dtype": np.float32}})
        assert "dy_c" in m.df.columns, "compress_columns creates a persistent column"
        auth = m.get_dtype_authority("dy_c")
        assert auth.known, "an ADF-created persistent column must be authoritative"
        assert auth.origin == _adf_module().DTypeOrigin.ADF_CREATED

    def test_b32b_4b_a_temporary_never_becomes_an_authority(self):
        """The negative control that must NEVER flip. It passes today and is
        the guard on b32b_4: whatever B3.2b does for persistent columns, a
        joined working temporary must not acquire public dtype authority.

        MR-P1-2: revision 2 asserted only that the temporary was physically
        gone, which a change that KEPT the column but exempted it from
        retraction would still satisfy while quietly minting an authority.
        Absence of the authority is now asserted directly, and separately from
        absence of the column."""
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64), "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("d", "S.v + x", dtype="int64", fill_value=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("d")
        assert "v__S" not in m.df.columns, "the placeholder temporary is retracted"
        auth = m.get_dtype_authority("v__S")
        assert not auth.known, (
            "no stale authority may survive for a retracted temporary — "
            f"got {auth!r}")

    def test_b32b_4c_every_persistent_creator_records_provenance(self):
        """MR-P1-2 / GPT27: revision 2 proved one path. The audit closes when
        every creator is enumerated and re-creation semantics are pinned —
        otherwise the family can close with compress_columns owned and
        decompress_columns silently unowned."""
        mod = _adf_module()
        created = []

        m = A.AliasDataFrame(pd.DataFrame({"dy": np.array([1.5, 2.5, 3.5])}))
        spec = {"dy": {"compress": "round(dy*10)",
                       "decompress": "dy_c/10.",
                       "compressed_dtype": np.int16,
                       "decompressed_dtype": np.float32}}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.compress_columns(spec)
        created.append(("compress_columns", m, "dy_c"))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.decompress_columns(["dy"])
        created.append(("decompress_columns", m, "dy"))

        declared = {n for n, (k, _) in
                    self.PERSISTENT_WRITER_DISPOSITION.items()
                    if k == "source-5"}
        exercised = {c for c, _, _ in created}
        assert declared <= exercised, (
            f"registry declares source-5 creators that no test exercises: "
            f"{sorted(declared - exercised)}")

        for creator, frame, col in created:
            assert col in frame.df.columns, f"{creator} did not create {col}"
            auth = frame.get_dtype_authority(col)
            assert auth.known, (
                f"{creator} created a PERSISTENT column {col!r} with no "
                f"authority — source 5 is not an inventory yet")
            assert auth.origin == mod.DTypeOrigin.ADF_CREATED, (
                f"{creator}: expected ADF_CREATED, got {auth.origin}")

        # ---- F4b (B3.2b STEP 3 correction). The claim "re-creation semantics
        # are pinned" was not exercised: the two creators above make DIFFERENT
        # names, so nothing here recreated the SAME persistent name. The
        # ratified source-5 table says recreation at the same dtype is
        # accepted and an incompatible recreation is refused, and until this
        # correction neither branch existed.
        m2 = A.AliasDataFrame(pd.DataFrame({"dy": np.array([1.5, 2.5, 3.5])}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2.compress_columns(spec)
        assert m2.get_dtype_authority("dy_c").origin == mod.DTypeOrigin.ADF_CREATED

        # same dtype -> accepted, authority intact
        m2["dy_c"] = np.array([7, 8, 9], dtype=np.int16)
        again = m2.get_dtype_authority("dy_c")
        assert again.known and again.origin == mod.DTypeOrigin.ADF_CREATED, (
            f"recreating at the SAME dtype must keep the source-5 authority: "
            f"{again!r}")

        # incompatible dtype -> refused, and the frame is left untouched
        _before = list(m2.df["dy_c"])
        with pytest.raises(ValueError) as _exc:
            m2["dy_c"] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert "dy_c" in str(_exc.value)
        assert list(m2.df["dy_c"]) == _before, (
            "a refused overwrite must not also corrupt the column")
        assert str(m2.df["dy_c"].dtype) == "int16", (
            "a refused overwrite must not change the dtype either")

        # and the authority must not survive the column being destroyed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m2.decompress_columns(["dy"], keep_compressed=False)
        assert "dy_c" not in m2.df.columns
        gone = m2.get_dtype_authority("dy_c")
        assert not gone.known, (
            f"the source-5 authority outlived the column it describes: "
            f"{gone!r}")

        # ---- F3 (v03). A same-name `add_alias(..., dtype=None)` must NOT
        # inherit the source-5 record. `_preserved_authority_on_redefinition`
        # was written to carry a SOURCE-4 inference across a redefinition;
        # STEP 3 put source-5 records under the same key and it returned them
        # unconditionally. Executed consequence before the fix:
        #
        #     compress                dy_c : int16, adf_created
        #     add_alias("dy_c","z*2", dtype=None) -> authority STILL int16
        #     materialize                          -> int16 [2, 4, 6]
        #
        # a brand-new alias over an unrelated expression constrained to a
        # compressed column's dtype.
        m3 = A.AliasDataFrame(pd.DataFrame({
            "dy": np.array([1.5, 2.5, 3.5]),
            "z": np.array([1.0, 2.0, 3.0])}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m3.compress_columns(spec)
        assert m3.get_dtype_authority("dy_c").origin == mod.DTypeOrigin.ADF_CREATED
        m3.add_alias("dy_c", "z * 2")            # same name, NO declared dtype
        after = m3.get_dtype_authority("dy_c")
        assert after.origin != mod.DTypeOrigin.ADF_CREATED, (
            f"the new alias inherited the destroyed column's source-5 "
            f"authority: {after!r}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m3.materialize_alias("dy_c")
        assert list(m3.df["dy_c"]) == [2.0, 4.0, 6.0], (
            f"the alias was constrained to the old column dtype: "
            f"{list(m3.df['dy_c'])} (z*2 is 2, 4, 6)")

    #: Every site that writes a column into `self.df`, with its adjudicated
    #: AD-19 disposition. R3B-P1-1 (GPT32): revision 3b enumerated two
    #: creators and called it a "complete persistent-column audit", so
    #: another creator could stay unclassified while the family closed.
    #: `b32b_4d` derives the write sites from the AST and requires every one
    #: to appear here, so the registry cannot silently fall behind the code.
    PERSISTENT_WRITER_DISPOSITION = {
        # --- source 5: ADF deliberately creates and PERSISTS a column ------
        "compress_columns":
            ("source-5",
             "creates <col>_c and keeps it; the compressed column is the "
             "stored representation, not a working temporary"),
        "decompress_columns":
            ("source-5",
             "restores <col> as a persistent column from the codec"),
        # --- explicitly NOT source 5, each with the reason ----------------
        "_publish_alias_column":
            ("not-source-5",
             "a materialized alias is AD-19 SOURCE 4 (first stored in-frame "
             "materialization), which is a different authority source"),
        "_publish_joined_column":
            ("not-source-5",
             "writes the join temporary v__S; the ratified text says a "
             "temporary working column is NOT an authority — b32b_4b is the "
             "control that must never flip"),
        "_retract_placeholder_columns":
            ("not-source-5",
             "removes columns rather than creating them; 11c added it so "
             "placeholder-bearing temporaries cannot masquerade as data"),
        "__setitem__":
            ("not-source-5",
             "user assignment; the result is an ordinary physical column, "
             "AD-19 source 2"),
        "apply_dtypes":
            ("not-source-5",
             "converts existing columns in place and creates none; family 8 "
             "owns its conversion disposition"),
        "convert_dtypes":
            ("not-source-5",
             "same as apply_dtypes — in-place conversion, family 8"),
        "apply_schema":
            ("not-source-5",
             "re-applies a declared schema; any authority follows the "
             "restored DECLARATION (source 3), not an ADF creation"),
        "update_schema":
            ("not-source-5",
             "schema bookkeeping; column writes here mirror a declaration"),
        "load":
            ("not-source-5",
             "reader ingestion; the authority is AD-19 source 1 (reader "
             "metadata) or source 2, never ADF creation"),
    }

    def test_b32b_4d_every_column_writer_has_an_adjudicated_disposition(self):
        """Completeness guard for ONE writer form — R3B-P1-1, narrowed in the
        B3.2b STEP 3 correction (F4).

        SCOPE, stated because the previous wording claimed more than the
        oracle can see. This walks the production AST for `self.df[...] = ...`
        subscript assignment ONLY. It does NOT recognise whole-frame
        publication:

            self.df = pd.concat(...)
            self.df = self._merge_loaded_data(...)
            self.df = self.df.drop(...)

        so "every column writer" was true of the assignment form and false of
        the family. No such path is a source-5 creator today — that was
        checked, not assumed — but the ORACLE cannot prove it, and a guard
        whose wording outruns its mechanism is the same defect as `b32b_5e`.
        STEP 6 owns the terminal persistent-column audit and is where
        whole-frame publication is enumerated; this guard is the assignment
        half of it.

        The site set is still derived from the AST rather than typed by hand,
        so adding a new subscript writer fails this test until somebody
        classifies it.

        Measured on the tagged bytes: ten functions assign into `self.df[...]`
        — apply_dtypes, apply_schema, convert_dtypes, decompress_columns,
        load, update_schema, __setitem__, _publish_alias_column,
        _publish_joined_column, _retract_placeholder_columns."""
        import ast as _ast
        tree = _ast.parse(_adf_source_text())
        writers = set()
        for fn in _ast.walk(tree):
            if not isinstance(fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            for node in _ast.walk(fn):
                if not isinstance(node, _ast.Assign):
                    continue
                for target in node.targets:
                    if (isinstance(target, _ast.Subscript)
                            and isinstance(target.value, _ast.Attribute)
                            and target.value.attr == "df"):
                        writers.add(fn.name)
        unclassified = sorted(w for w in writers
                              if w not in self.PERSISTENT_WRITER_DISPOSITION)
        assert not unclassified, (
            f"column writers with no adjudicated AD-19 disposition: "
            f"{unclassified}")
        for name, entry in self.PERSISTENT_WRITER_DISPOSITION.items():
            kind, reason = entry
            assert kind in ("source-5", "not-source-5"), name
            assert len(reason) >= 30, f"{name} has no stated reason"

    # ---- family 1: reader / branch metadata (AD-19 source 1) -------------

    def test_b32b_5_reader_metadata_is_an_authority_source(self, tmp_path):
        """P1-1 correction. Revision 1 counted a token in the module source,
        which a comment or dead branch could satisfy. The reviewers were right
        that this file already has uproot fixtures, so a behavioural test IS
        available — my 'no lazy reader in this sandbox' justification was
        wrong."""
        p = tmp_path / "b32b_src.root"
        _write_tree(p, np.arange(8, dtype=np.float64),
                    np.ones(8, dtype=np.float64))
        lazy = A.AliasDataFrame.read_tree_lazy(str(p), "tree")
        auth = lazy.get_dtype_authority("x")
        assert auth.known, "a lazy branch's declared dtype is authoritative"
        assert auth.origin == _adf_module().DTypeOrigin.READER_METADATA
        assert str(auth.dtype) == "float64"

    def test_b32b_5b_metadata_lookup_does_not_load_the_branch(self, tmp_path):
        p = tmp_path / "b32b_noload.root"
        _write_tree(p, np.arange(8, dtype=np.float64),
                    np.ones(8, dtype=np.float64))
        lazy = A.AliasDataFrame.read_tree_lazy(str(p), "tree")
        before = set(getattr(lazy._lazy_reader, "loaded_branches", ()) or ())
        auth = lazy.get_dtype_authority("x")
        after = set(getattr(lazy._lazy_reader, "loaded_branches", ()) or ())
        assert auth.known, (
            "no reader authority exists at all, so the no-load property is "
            f"vacuous: {auth!r}")
        assert before == after, "inspecting a dtype must not load data"

    def test_b32b_5c_chain_metadata_agreement_is_explicit(self, tmp_path):
        p1, p2 = tmp_path / "c1.root", tmp_path / "c2.root"
        for p in (p1, p2):
            _write_tree(p, np.arange(4, dtype=np.float64),
                        np.ones(4, dtype=np.float64))
        chain = A.AliasDataFrame.read_chain_lazy([str(p1), str(p2)], "tree")
        auth = chain.get_dtype_authority("x")
        assert auth.known, (
            f"an agreeing chain yields no reader authority: {auth!r}")
        assert str(auth.dtype) == "float64", f"wrong agreed dtype: {auth!r}"
        assert auth.origin == _adf_module().DTypeOrigin.READER_METADATA

    def test_b32b_5d_chain_metadata_disagreement_is_not_silently_resolved(
            self, tmp_path):
        """This is the coder's fifth reason-string-does-not-match-body defect
        in this phase and the reason the --runxfail rule exists. Revision 2
        wrote 'disagreeing metadata is an explicit conflict' into b32b_5c's
        reason and then built two AGREEING float64 files.

        The oracle is DISTINGUISHABILITY, not a guessed conflict field: the
        first draft of this test accepted `origin is not None`, which the
        default UNKNOWN satisfies, and it XPASSED on the tagged bytes. Caught
        by running it, not by reading it."""
        a1, a2 = tmp_path / "a1.root", tmp_path / "a2.root"
        _write_tree_dtyped(a1, "float64")
        _write_tree_dtyped(a2, "float64")
        agree = A.AliasDataFrame.read_chain_lazy([str(a1), str(a2)], "tree")

        d1, d2 = tmp_path / "d1.root", tmp_path / "d2.root"
        _write_tree_dtyped(d1, "float64")
        _write_tree_dtyped(d2, "float32")
        dis = A.AliasDataFrame.read_chain_lazy([str(d1), str(d2)], "tree")

        a_auth = agree.get_dtype_authority("x")
        d_auth = dis.get_dtype_authority("x")
        assert a_auth.known, (
            "precondition (AD-19 source 1): an AGREEING chain must yield a "
            f"known authority before disagreement can mean anything: {a_auth!r}")
        assert repr(d_auth) != repr(a_auth), (
            "an agreeing chain and a disagreeing chain are indistinguishable: "
            f"both report {a_auth!r}")
        if d_auth.known:
            assert getattr(d_auth, "conflict", False), (
                "a disagreeing chain reported one member's dtype as "
                f"authoritative without recording the conflict: {d_auth!r}")

    def test_b32b_5e_declared_and_loaded_dtypes_are_reconciled(self, tmp_path):
        """R3B-P0-1 (GPT32), and he is right. Revision 3b wrote a float32
        branch and compared the declared dtype with the loaded dtype OF THE
        SAME BRANCH — both float32, so no contradiction was ever built. Once
        source-1 reader authority lands, that version would have XPASSED
        while metadata-vs-loaded conflict handling stayed unimplemented:
        precisely the false closure this scaffold exists to prevent, and the
        same shape as the b32b_5c defect one revision earlier.

        Measured on the tagged bytes (pandas 1.5.3): the branch loads as
        float32 via `ensure_branches`, so the contradiction must be created
        deliberately by recasting the column. That makes the two dtypes
        really differ, which is what the finding requires."""
        p = tmp_path / "b32b_conflict.root"
        _write_tree_dtyped(p, "float32")
        lazy = A.AliasDataFrame.read_tree_lazy(str(p), "tree")
        declared = lazy.get_dtype_authority("x")
        assert declared.known, (
            f"no reader authority is declared, so metadata-vs-physical "
            f"cannot be contradicted: {declared!r}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lazy.ensure_branches(["x"])
        assert str(lazy.df["x"].dtype) == str(declared.dtype), (
            "precondition: the branch must load as its declared dtype")

        # The genuine contradiction: physical float64 against declared float32.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lazy.df["x"] = lazy.df["x"].astype(np.float64)
        after = lazy.get_dtype_authority("x")

        # TERMINAL ORACLE (B3.2b STEP 3 correction, F1). The previous form was
        #
        #     assert (str(after.dtype) == str(lazy.df["x"].dtype)
        #             or getattr(after, "conflict", False))
        #
        # and `DTypeAuthority` had no `conflict` attribute, so the second
        # clause was ALWAYS False and the first is exactly what source 2 does.
        # The assertion could not fail. A criterion written to forbid a silent
        # winner accepted the silent winner -- the precise false closure its
        # own docstring says it exists to prevent, and the third time this
        # `getattr`-a-field-that-does-not-exist shape has appeared in this
        # file. The field now exists, so the probe is load-bearing; asserting
        # it directly rather than through `getattr` means its removal breaks
        # the test instead of silently satisfying it.
        assert after.conflict, (
            f"reader metadata declared {declared.dtype} and the loaded column "
            f"is {lazy.df['x'].dtype}; the authority reports "
            f"{after.dtype} with NO recorded conflict, so one source silently "
            f"won: {after!r}")
        assert after.conflict_detail, (
            "a conflict must say WHAT disagrees, or a consumer cannot act on "
            f"it: {after!r}")
        assert str(declared.dtype) in after.conflict_detail, (
            f"the conflict detail does not name the contradicted reader "
            f"dtype {declared.dtype}: {after.conflict_detail!r}")

        # ...and the control: agreement must NOT be reported as a conflict,
        # or the check degenerates into "always contested" and proves nothing.
        agreeing = A.AliasDataFrame.read_tree_lazy(str(p), "tree")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agreeing.ensure_branches(["x"])
        assert not agreeing.get_dtype_authority("x").conflict, (
            "a branch loaded AS its declared dtype must not be reported as "
            "contested")

        # ---- F1 (v03): the conflict must be ENFORCED AT USE, not merely
        # recorded. v02 recorded it and consumed it nowhere -- `.conflict`
        # existed at __eq__, __hash__ and __repr__ and at no other production
        # site -- while the v02 CRR claimed refusal "at the point of use".
        # The contested subject is a physical OPERAND, so a check inside
        # `_resolve_target_dtype` could not see it; this pins the behaviour
        # rather than the location.
        lazy.add_alias("uses_x", "x * 2")
        for _getter in ("materialize_alias", "get_alias_series",
                        "get_alias_array"):
            with pytest.raises(ValueError) as _exc:
                getattr(lazy, _getter)("uses_x")
            assert "x" in str(_exc.value), (
                f"{_getter} refused without naming the contested subject: "
                f"{_exc.value}")

        # ...and INSPECTION still does not raise. That distinction is the
        # entire justification for recording rather than throwing.
        assert lazy.get_dtype_authority("x").conflict

        # CONTROL: the agreeing frame must still evaluate, or "refuses when
        # contested" degenerates into "refuses always".
        agreeing.add_alias("ok_x", "x * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            agreeing.materialize_alias("ok_x")
        assert "ok_x" in agreeing.df.columns

        # ---- v04 `F1`: the PUBLIC eval() surface. v03 guarded
        # `_evaluate_alias_expression` and the v03 CRR claimed that covered
        # "every alias-evaluating public path". `adf.eval()` goes straight to
        # `_eval_in_namespace` and bypassed it, returning values computed from
        # a contested column. The guard now lives at that single shared point,
        # so eval(), alias evaluation and the vector-group compute all reach
        # it.
        with pytest.raises(ValueError) as _e_eval:
            lazy.eval("x * 2")
        assert "x" in str(_e_eval.value)

        # CONTROL: the agreeing frame must still evaluate through eval().
        assert list(agreeing.eval("x * 2"))[:2] == [0.0, 2.0]

        # ---- v04 `F2`: SUBFRAME-QUALIFIED operands. `_analyze_expression`
        # returns `subframe_refs` alongside `column_refs`; v03 read only the
        # latter, so a parent alias over a contested CHILD column evaluated
        # happily. My own v03 CRR named this class and did not test it.
        def _pair():
            _p = A.AliasDataFrame(pd.DataFrame({
                "k": np.arange(3), "a": np.ones(3)}))
            _c = A.AliasDataFrame(pd.DataFrame({
                "k": np.arange(3), "q": np.arange(3, dtype=np.float32)}))
            _c.add_alias("q2", "q * 1", dtype="float32")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _c.materialize_alias("q2")
            _p.register_subframe("S", _c, index_columns=["k"])
            return _p, _c

        # CONTROL first: an agreeing child must still project.
        _par, _ch = _pair()
        _par.add_alias("p", "S.q2 * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _par.materialize_alias("p")
        assert list(_par.df["p"]) == [0.0, 2.0, 4.0]

        # ...then the contested child.
        _par, _ch = _pair()
        _ch.df["q2"] = _ch.df["q2"].astype(np.float64)
        assert _ch.get_dtype_authority("q2").conflict, (
            "fixture precondition: the CHILD column must be contested")
        _par.add_alias("p", "S.q2 * 2")
        with pytest.raises(ValueError) as _e_sub:
            _par.materialize_alias("p")
        assert "S.q2" in str(_e_sub.value), (
            f"the refusal must name the contested CHILD subject: "
            f"{_e_sub.value}")

        # ---- v05 `P0-LAZY`: source 1 must SURVIVE a lazy child's load.
        # `_load_lazy_subframe` built a plain `AliasDataFrame(df)`, so the
        # reader -- and with it AD-19 source 1 -- vanished at materialization.
        # A conflict needs TWO sources, so after the load an incompatible
        # recast could not even be represented and a parent expression
        # consumed it. The reader's DECLARED dtypes are now carried onto the
        # child (metadata only, not the reader itself, which would make the
        # child look lazy and invite further loading).
        _lz = tmp_path / "b32b_lazychild.root"
        with uproot.recreate(str(_lz)) as _f:
            _f.mktree("tree", {"k": "int64"})
            _f["tree"].extend({"k": np.arange(3)})
            _f.mktree("Ch", {"k": "int64", "q": "float32"})
            _f["Ch"].extend({"k": np.arange(3),
                             "q": np.arange(3, dtype=np.float32)})

        def _lazy_pair():
            _m = A.AliasDataFrame.read_tree_lazy(str(_lz), "tree")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _m.ensure_branches(["k"])
                _m.register_subframe_lazy("S", str(_lz), tree_name="Ch",
                                          index_columns=["k"])
                _m.ensure_subframe("S")
            return _m, _m._subframes.get("S")

        # CONTROL: matching reader/physical -> qualified use succeeds.
        _m, _c = _lazy_pair()
        _m.add_alias("u", "S.q * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _m.materialize_alias("u")
        assert list(_m.df["u"]) == [0.0, 2.0, 4.0]

        # loaded, then recast incompatibly -> conflict visible, use refuses.
        _m, _c = _lazy_pair()
        _c.df["q"] = _c.df["q"].astype(np.float64)
        assert _c.get_dtype_authority("q").conflict, (
            "the reader's declared dtype did not survive the child's load, "
            "so a physical contradiction cannot even be formed")
        _m.add_alias("u", "S.q * 2")
        with pytest.raises(ValueError) as _e_lazy:
            _m.materialize_alias("u")
        assert "S.q" in str(_e_lazy.value)

        # ---- v05 `P0-NESTED`: a MULTI-LEVEL chain. `_analyze_expression`
        # resolves one level -- `A.B.q` came back as `('A','B')` -- so the
        # deep leaf was never produced and went unchecked while the join
        # resolver reached it happily. Both now share one chain grammar.
        def _deep():
            _p = A.AliasDataFrame(pd.DataFrame({"k": np.arange(3)}))
            _mid = A.AliasDataFrame(pd.DataFrame({"k": np.arange(3)}))
            _leaf = A.AliasDataFrame(pd.DataFrame({
                "k": np.arange(3), "q": np.arange(3, dtype=np.float32)}))
            _leaf.add_alias("q2", "q * 1", dtype="float32")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _leaf.materialize_alias("q2")
            _mid.register_subframe("B", _leaf, index_columns=["k"])
            _p.register_subframe("A", _mid, index_columns=["k"])
            return _p, _leaf

        # CONTROL: nested and agreeing -> succeeds.
        _par, _leaf = _deep()
        _par.add_alias("w", "A.B.q2 * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _par.materialize_alias("w")
        assert list(_par.df["w"]) == [0.0, 2.0, 4.0]

        # contested DEEP leaf -> refuses, and names the qualified subject.
        _par, _leaf = _deep()
        _leaf.df["q2"] = _leaf.df["q2"].astype(np.float64)
        assert _leaf.get_dtype_authority("q2").conflict
        _par.add_alias("w", "A.B.q2 * 2")
        with pytest.raises(ValueError) as _e_deep:
            _par.materialize_alias("w")
        assert "A.B.q2" in str(_e_deep.value), (
            f"the refusal must name the full dotted subject the user wrote: "
            f"{_e_deep.value}")

    # ---- family 2: persisted dtype and origin ----------------------------
    #
    # ENVIRONMENT. `export_tree` requires xxhash and PyROOT is also absent;
    # the coder venv has no pip and cannot install either, so these three
    # tests SKIP in the coder sandbox and RUN on alma2.
    #
    # REVISION 3b. Revision 3 disclosed these as the only tests whose failure
    # mode the coder had not executed. The architect ran them on alma2 and
    # ALL THREE FAILED FOR THE WRONG REASON — two guessed fixtures and one
    # string comparison, none of them the contract defect the reason named:
    #
    #   b32b_13   got PAST `assert after.known` and failed on
    #             assert "<class 'numpy.float32'>" == 'float32'.
    #             The authority DOES survive. -> converted to PASSING.
    #   b32b_13b  cleared attributes (`_dtype_authority`, `_authority`,
    #             `dtype_authority`) that DO NOT EXIST, so the 'legacy'
    #             fixture was a no-op, and then checked a guessed origin
    #             allow-list. -> replaced by the canonical-representation
    #             defect the failure actually exposed.
    #   b32b_13c  raised KeyError 'q': read_tree restores `q` as an ALIAS,
    #             not a column. -> materialize first.
    #
    # Every assertion below is now written against measured alma2 output
    # (pandas 1.5.3 / numpy 1.24.2), not against the coder's assumption.

    def test_b32b_13_authority_survives_the_persistence_round_trip(
            self, tmp_path):
        """ALREADY SATISFIED ON ENTRY — converted from a strict xfail, on
        alma2 evidence.

        Revision 3 shipped this as an xfail whose reason said 'dtype AND
        origin survive the round trip'. Run on alma2 it got PAST
        `assert after.known` and failed on a SPELLING comparison:

            assert "<class 'numpy.float32'>" == 'float32'

        Semantically the authority survives intact — same dtype, same origin.
        My assertion compared `str()` of two representations of the same
        dtype. Measured on alma2 (pandas 1.5.3 / numpy 1.24.2):

            before  DTypeAuthority(float32, origin=explicit_alias, stored_in_frame)
                    .dtype = dtype('float32')
            after   DTypeAuthority(<class 'numpy.float32'>, origin=explicit_alias)
                    .dtype = <class 'numpy.float32'>
            origins equal      True
            np.dtype equal     True

        So survival is pinned here as PASSING, and the representation defect
        the spelling mismatch actually exposes is `b32b_13b`."""
        pytest.importorskip("xxhash")   # see the class note on family 2
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1.0, 2.0])}))
        m.add_alias("q", "x * 2", dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        before = m.get_dtype_authority("q")
        assert before.known
        out = str(tmp_path / "b32b_persist.root")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.export_tree(out, "t")
            back = A.AliasDataFrame.read_tree(out, "t")
        after = back.get_dtype_authority("q")
        assert after.known, "authority must survive persistence"
        assert np.dtype(after.dtype) == np.dtype(before.dtype), (
            f"dtype changed across persistence: {before.dtype!r} -> "
            f"{after.dtype!r}")
        assert after.origin == before.origin, (
            f"origin changed across persistence: {before.origin} -> "
            f"{after.origin}")

    def test_b32b_13b_restored_authority_is_canonically_represented(
            self, tmp_path):
        """ADJUDICATED in the B3.2b STEP 3 correction, per the Main Reviewer's
        instruction not to implement this one blindly.

        Revision 3b asserted `str(after.dtype) == str(before.dtype)`. That is
        REPRESENTATION identity, which is stronger than the ratified contract
        (§13 item 2: dtype AND origin survive). I implemented it — one line,
        `np.dtype(value)` instead of `np.dtype(value).type` in
        `_deserialize_schema` — and the full-suite failure-identity diff
        refused it:

            FAILED tests/test_schema_serialization.py::
                   test_deserialize_schema_restores_dtypes

        That test predates this phase and asserts
        `hasattr(spec['dtype'], '__name__')`, i.e. a numpy TYPE CLASS. Its own
        comment says "should be a numpy type, not a string", so its INTENT
        admits a dtype instance and its MECHANISM does not — the same
        intent-versus-mechanism gap that made `b32b_5e` a false close.

        So the current representation IS pinned by a ratified test. Editing
        that test so my criterion could pass would be accommodating in the
        wrong direction. This criterion is therefore adjudicated down to the
        SEMANTIC contract, which alma2 already measured as holding:

            before  DTypeAuthority(float32,               origin=explicit_alias)
            after   DTypeAuthority(<class numpy.float32>, origin=explicit_alias)
            np.dtype equal   True
            origins  equal   True

        Exact representation identity remains a genuine improvement. It
        belongs in a change that owns BOTH tests, not in a correction whose
        scope is authority sources."""
        pytest.importorskip("xxhash")   # see the class note on family 2
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1.0, 2.0])}))
        m.add_alias("q", "x * 2", dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        before = m.get_dtype_authority("q")
        out = str(tmp_path / "b32b_canonical.root")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.export_tree(out, "t")
            back = A.AliasDataFrame.read_tree(out, "t")
        after = back.get_dtype_authority("q")
        assert after.known
        assert np.dtype(after.dtype) == np.dtype(before.dtype), (
            f"the restored authority is not the same dtype: "
            f"{before.dtype!r} -> {after.dtype!r}")
        assert after.origin == before.origin, (
            f"the restored authority lost its origin: "
            f"{before.origin!r} -> {after.origin!r}")

    def test_b32b_13c_restored_metadata_conflict_is_explicit(self, tmp_path):
        """MR-P1-3, oracle strengthened in v03 (`F4`). The v02 body ended in

            assert np.dtype(auth.dtype) == back.df["q"].dtype or getattr(
                auth, "conflict", False)

        which accepts an END STATE and proves no TRANSITION: a build that
        reported `conflict=True` unconditionally would pass it, and so would
        one that never conflicts but happens to agree. v03 asserts the
        transition — agreement BEFORE, conflict AFTER, refusal at use — so
        both degenerate builds fail."""
        pytest.importorskip("xxhash")   # see the class note on family 2
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1.0, 2.0])}))
        m.add_alias("q", "x * 2", dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        out = str(tmp_path / "b32b_conflict.root")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.export_tree(out, "t")
            back = A.AliasDataFrame.read_tree(out, "t")
            back.materialize_alias("q")          # restored as an alias
        assert "q" in back.df.columns, "the fixture needs a physical column"

        # BEFORE: restored record and physical column AGREE. `F2` — a v02
        # build marked this contested because it compared str() of
        # `<class 'numpy.float32'>` against `dtype('float32')`.
        clean = back.get_dtype_authority("q")
        assert clean.known
        assert not clean.conflict, (
            f"a restored authority that MATCHES its column must not be "
            f"reported contested: {clean!r} vs {back.df['q'].dtype}")

        # AFTER: a real contradiction.
        back.df["q"] = back.df["q"].astype(np.float64)
        auth = back.get_dtype_authority("q")
        assert auth.known, "the restored authority must still be reportable"
        assert auth.conflict, (
            f"restored metadata and the physical column disagree and no "
            f"conflict is recorded: {auth!r} vs {back.df['q'].dtype}")
        assert auth.conflict_detail, "a conflict must say what disagrees"

        # AND USING IT REFUSES (`F1`) — inspection above never raised.
        back.add_alias("uses_q", "q * 2")
        with pytest.raises(ValueError) as exc:
            back.materialize_alias("uses_q")
        assert "q" in str(exc.value)

    def test_b32b_13d_persisted_record_carries_an_explicit_authority(self):
        """B32B-R3B-P0-1, oracle strengthened in v03 (`F4`). The v02 body
        asserted only that SOME key whose NAME contains "origin" or
        "authority" exists — a record holding `{"authority": None}` would have
        satisfied it. v03 asserts the contents: dtype, origin and
        subject_kind, each against what the authority actually reports."""
        mod = _adf_module()
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1.0, 2.0])}))
        m.add_alias("q", "x * 2", dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        auth = m.get_dtype_authority("q")
        assert auth.known, "precondition: the authority must be known"

        entry = m._schema["columns"]["q"]
        rec = entry.get("dtype_authority")
        assert isinstance(rec, dict), (
            f"the persisted record carries no authority entry, so its "
            f"absence in an older file is undetectable: {entry!r}")
        assert np.dtype(rec["dtype"]) == np.dtype(auth.dtype), (
            f"the persisted dtype {rec['dtype']!r} is not the authority's "
            f"{auth.dtype!r}")
        assert rec["origin"] == mod.DTypeOrigin.EXPLICIT_ALIAS, (
            f"a DECLARED dtype is AD-19 source 3; the record says "
            f"{rec['origin']!r}")
        assert rec["origin"] == auth.origin, "record and authority disagree"
        assert rec.get("subject_kind") == "alias", (
            f"the record must say WHAT it describes: {rec!r}")

    def test_b32b_13e_absent_metadata_does_not_fabricate_authority(self):
        """The other half of B32B-R3B-P0-1, and it already holds — pinned so
        that adding the authority record in STEP 3 cannot regress it.

        A column record with no dtype metadata is exactly the legacy shape
        (`{'expr': 'x * 2'}`, measured). Reading it must not invent an
        authority, must not crash, and must leave the result UNKNOWN until
        legitimate evidence exists."""
        mod = _adf_module()
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1.0, 2.0])}))
        m.add_alias("q", "x * 2")               # no declared dtype: legacy shape
        assert "dtype" not in m._schema["columns"]["q"], (
            "precondition: the record must genuinely lack dtype metadata")
        auth = m.get_dtype_authority("q")
        assert not auth.known, (
            f"authority was fabricated from absent metadata: {auth!r}")
        assert auth.origin == mod.DTypeOrigin.UNKNOWN

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        after = m.get_dtype_authority("q")
        assert after.origin == mod.DTypeOrigin.FIRST_MATERIALIZATION, (
            "once the column really is materialized in this frame, source 4 "
            "is legitimate evidence and must be recorded as such — the "
            f"legacy record must not block it: {after!r}")

    # ---- family 3: Arrow-backed dtypes ------------------------------------
    #
    # REVISION 3d. The architect reported the Arrow tests as unstable —
    # "sometimes OK, sometimes failing". They are not flaky. Three distinct
    # things were conflated, and measurement separates them:
    #
    #   1. pandas has TWO Arrow mechanisms, not one.
    #        int64[pyarrow]   -> pd.ArrowDtype           (a real Arrow type)
    #        string[pyarrow]  -> pd.StringDtype(storage) (a pandas string
    #                                                     stored in Arrow)
    #      `isinstance(dt, pd.ArrowDtype)` is True for the first and FALSE
    #      for the second. Testing them as one family is what produced
    #      revision 3's false "silent downgrade" finding.
    #
    #   2. str(dtype) is not stable across the two. ArrowDtype prints
    #      'int64[pyarrow]'; StringDtype prints 'string' whatever its
    #      storage. Every comparison here is therefore structural or
    #      relational — never textual. `_is_arrow_backed` classifies by
    #      `pd.ArrowDtype` / `.storage`, never by a substring.
    #
    #   3. THE REAL DEFECT, and the source of the reported instability:
    #      whether a string column gets a dtype authority depends on the
    #      PROCESS-WIDE option `pd.options.mode.string_storage`, not on the
    #      column. Measured on the tagged bytes, identical input:
    #
    #        string_storage='python'   -> DTypeAuthority(UNKNOWN)
    #        string_storage='pyarrow'  -> authority recorded
    #
    #      because `_authority_is_exactly_representable` round-trips through
    #      `pd.api.types.pandas_dtype(str(dtype))`, and for StringDtype that
    #      round trip is resolved by the global option:
    #
    #        pandas_dtype('string')  -> string[python]   (default here)
    #        pandas_dtype('string')  -> string[pyarrow]  (under the option)
    #
    #      The architect runs pandas 1.5.3, where the default is 'python';
    #      reviewers run 2.2.3 and 3.0.2, where pyarrow-backed strings become
    #      the default. Same file, same code, different answer — which is
    #      exactly what "sometimes OK, sometimes failing" looks like from the
    #      outside. `b32b_14g` owns it.
    #
    # Every test below therefore: pins the option it depends on rather than
    # inheriting it, probes whether this pandas+pyarrow can build a specimen
    # before using it, and prints the environment in its failure message.

    def test_b32b_14_arrow_backed_gather_already_preserves_the_backing(self):
        """ALREADY SATISFIED ON ENTRY, recorded rather than xfailed.

        I wrote this as a strict xfail and it XPASSED: the AD-7/AD-11
        symmetric gather already keeps a pyarrow backing through a subframe
        join with a missing key. Assuming a family was open because the
        closure report listed it would have been the same error as revision
        1 in the other direction — so the basic case is pinned as passing and
        the genuinely open parts are `b32b_14b`, `14c` and `14g`."""
        pytest.importorskip("pyarrow")
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64), "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = pd.array([3.5], dtype="double[pyarrow]")
        m.register_subframe("S", ch, index_columns=["k"])
        m.set_subframe_fill("S", fill_missing=0.0)
        m.add_alias("d", "S.v")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("d")
        assert m.df["d"].dtype == ch.df["v"].dtype, (
            f"the gather changed the dtype: {ch.df['v'].dtype!r} -> "
            f"{m.df['d'].dtype!r} [{_arrow_env()}]")

    def test_b32b_14b_arrow_backed_dtype_is_an_authority(self):
        pytest.importorskip("pyarrow")
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1.0, 2.0])}))
        m.df["v"] = pd.array([1.5, 2.5], dtype="double[pyarrow]")
        auth = m.get_dtype_authority("v")
        assert auth.known, (
            f"an Arrow-backed column must be authoritative [{_arrow_env()}]")
        assert auth.dtype == m.df["v"].dtype, (
            f"the authority lost the backing: {auth.dtype!r} vs "
            f"{m.df['v'].dtype!r} [{_arrow_env()}]")

    #: The ratified Arrow dimensions, each mapped to the test that owns it.
    #: R3B-P0-2 (GPT32 and GPT30, independently): revision 3b had four dtype
    #: SPECIMENS, which is not the same thing as the ratified "full
    #: Arrow-backed dtype coverage". `b32b_14d` enforces that every dimension
    #: keeps an owner, so fixing one case cannot close the family.
    ARROW_DIMENSIONS = {
        "matched":          "b32b_14e",   # every key present
        "missing":          "b32b_14",    # partial join, key absent
        "empty":            "b32b_14e",   # zero-row child
        "fill":             "b32b_14",    # configured fill through a gather
        "cast":             "b32b_14e",   # int64[pyarrow] preserved, not widened
        "metadata":         "b32b_14b",   # a physical Arrow column is an authority
        "native_parity":    "b32b_14e",   # numpy-backed twin behaves identically
        "dtype_breadth":    "b32b_14c",   # the ArrowDtype specimen matrix
        "storage_family":   "b32b_14g",   # ArrowDtype vs StringDtype(storage)
        "env_independence": "b32b_14g",   # no global option may decide authority
        "refusal":          "b32b_14f",   # carried exactly, or refused
    }

    #: ARROW-TYPED specimens — every one of these is a `pd.ArrowDtype`.
    #: Revision 3b called the first case "matched" while giving it parent
    #: keys [0, 9] against a one-row child, so it was a MISSING case wearing
    #: the wrong name (GPT32). Labels now name the dtype they test; matched,
    #: empty and cast have their own real fixtures in `b32b_14e`.
    ARROW_TYPED_CASES = (
        ("int64",  "int64[pyarrow]",  np.array([0, 9], np.int64), 0),
        ("double", "double[pyarrow]", np.array([7, 8], np.int64), 0.0),
        ("bool",   "bool[pyarrow]",   np.array([0, 9], np.int64), False),
        ("string", "ARROW_STRING",    np.array([0, 9], np.int64), ""),
    )

    def test_b32b_14d_every_arrow_dimension_has_an_owner(self):
        """Passing guard, the family-3 analogue of `b32b_0`. Without it the
        Arrow family can close on whichever dimensions happened to be
        sampled — R3B-P0-2. It fails if a dimension names a test that does
        not exist, so dimensions cannot be quietly dropped either."""
        import ast as _ast
        tree = _ast.parse(_adf_scaffold_text())
        defined = {n.name for n in _ast.walk(tree)
                   if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
        for dimension, owner in self.ARROW_DIMENSIONS.items():
            assert any(d.startswith(f"test_{owner}_") for d in defined), (
                f"Arrow dimension {dimension!r} names owner {owner!r}, "
                f"which is not a defined test function")

    def test_b32b_14e_arrow_matched_empty_cast_and_native_parity(self):
        """ALREADY SATISFIED ON ENTRY — measured, then pinned.

        R3B-P0-2 asked for the missing dimensions. I built them expecting
        xfails and all four already hold: a fully matched join, a zero-row
        child, an int64 backing that is not silently widened, and a
        numpy-backed twin that reaches the same authority origin. Pinning
        them as passing is the same call as `b32b_14` / `b32b_15`.

        Every assertion compares dtype OBJECTS against the fixture's own
        dtype, so nothing here depends on how pandas spells a dtype in this
        version."""
        pytest.importorskip("pyarrow")
        keys = np.array([0, 1], np.int64)

        def gather(child_keys, values, dtype, fill=None):
            m = A.AliasDataFrame(pd.DataFrame({
                "k": keys, "x": np.array([10, 20], np.int64)}))
            ch = A.AliasDataFrame(pd.DataFrame({"k": child_keys}))
            ch.df["v"] = pd.array(values, dtype=dtype)
            m.register_subframe("S", ch, index_columns=["k"])
            if fill is not None:
                m.set_subframe_fill("S", fill_missing=fill)
            m.add_alias("d", "S.v")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_alias("d")
            return m, ch

        matched, mch = gather(np.array([0, 1], np.int64), [1.5, 2.5],
                              "double[pyarrow]")
        assert matched.df["d"].dtype == mch.df["v"].dtype, (
            f"matched: {mch.df['v'].dtype!r} -> {matched.df['d'].dtype!r} "
            f"[{_arrow_env()}]")
        assert matched.get_dtype_authority("d").known

        empty, ech = gather(np.array([], np.int64), [], "double[pyarrow]",
                            fill=0.0)
        assert empty.df["d"].dtype == ech.df["v"].dtype, (
            f"empty child: {ech.df['v'].dtype!r} -> {empty.df['d'].dtype!r} "
            f"[{_arrow_env()}]")
        assert len(empty.df["d"]) == 2, "the parent row count is preserved"

        cast, cch = gather(np.array([0, 1], np.int64), [1, 2], "int64[pyarrow]")
        assert cast.df["d"].dtype == cch.df["v"].dtype, (
            f"an int64 Arrow backing was changed: {cch.df['v'].dtype!r} -> "
            f"{cast.df['d'].dtype!r} [{_arrow_env()}]")

        native, _ = gather(np.array([0, 1], np.int64), [1.5, 2.5], "float64")
        assert native.get_dtype_authority("d").origin == (
            matched.get_dtype_authority("d").origin), (
            f"native-vs-Arrow parity: the two backings must reach the same "
            f"authority origin [{_arrow_env()}]")

    def test_b32b_14f_no_arrow_form_is_silently_downgraded(self):
        """The refusal / no-downgrade dimension — ALREADY SATISFIED, and the
        test that CAUGHT MY OWN FALSE FINDING.

        I wrote this as a strict xfail whose reason said 'string[pyarrow] is
        silently downgraded to pandas string'. Run, it reported the dropped
        dtype as `string[pyarrow]` — contradicting itself. The cause: pandas
        prints `str(StringDtype)` as 'string' for EVERY storage, so revision
        3b's `"pyarrow" in str(dtype)` oracle read a preserved Arrow column
        as a downgraded one. Measured: the gathered dtype compares EQUAL to
        the child dtype for every specimen, typed and storage alike.

        So the §0.3 finding I published in the revision-3 CRR — and put in a
        draft commit message — was FALSE. The backing is preserved
        everywhere. What is genuinely missing for strings is the AUTHORITY
        (`b32b_14c`), and why it is missing is environmental (`b32b_14g`).

        This test pins the true invariant: an Arrow form is either carried
        through with its exact dtype or refused by an ADF-owned error.
        Silent downgrade is not an acceptable third option."""
        pytest.importorskip("pyarrow")
        mod = _adf_module()
        # NOT `getattr(mod, "ADFError", ValueError)`. That default silently
        # widened the acceptable-refusal branch to every ValueError in the
        # stack while `ADFError` was still unimplemented, and swallowed a
        # fixture bug for a whole increment. `D_6` delivered the root; the
        # test must now depend on it, and fail loudly if it ever disappears.
        root = mod.ADFError
        for label, source in _arrow_specimens().items():
            m = A.AliasDataFrame(pd.DataFrame({
                "k": np.array([0, 9], np.int64),
                "x": np.array([10, 20], np.int64)}))
            ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
            ch.df["v"] = source
            m.register_subframe("S", ch, index_columns=["k"])
            m.set_subframe_fill("S", fill_missing=_arrow_neutral(source.dtype))
            m.add_alias("d", "S.v")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.materialize_alias("d")
            except root:
                continue                # explicit refusal is acceptable
            assert m.df["d"].dtype == ch.df["v"].dtype, (
                f"{label}: the Arrow backing was changed silently from "
                f"{ch.df['v'].dtype!r} to {m.df['d'].dtype!r} — neither "
                f"preserved nor refused [{_arrow_env()}]")

    def test_b32b_14c_arrow_authority_holds_across_the_matrix(self):
        pytest.importorskip("pyarrow")
        failures = []
        for label, dtype, keys, fill in self.ARROW_TYPED_CASES:
            source = _arrow_build(dtype, [fill])
            if source is None:
                failures.append(f"{label}: this pandas/pyarrow cannot build "
                                f"the specimen [{_arrow_env()}]")
                continue
            m = A.AliasDataFrame(pd.DataFrame({
                "k": keys, "x": np.array([10, 20], np.int64)}))
            ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
            ch.df["v"] = source
            m.register_subframe("S", ch, index_columns=["k"])
            m.set_subframe_fill("S", fill_missing=fill)
            m.add_alias("d", "S.v")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.materialize_alias("d")
            except Exception as exc:
                failures.append(f"{label}: materialize raised "
                                f"{type(exc).__name__}")
                continue
            auth = m.get_dtype_authority("d")
            if not auth.known:
                failures.append(f"{label}: authority UNKNOWN after gather")
            elif auth.dtype != ch.df["v"].dtype:
                failures.append(f"{label}: authority is {auth.dtype!r}, not "
                                f"the backed {ch.df['v'].dtype!r}")
        assert not failures, ("Arrow matrix gaps: " + "; ".join(failures)
                              + f" [{_arrow_env()}]")

    def test_b32b_14g_authority_does_not_depend_on_a_global_option(self):
        """The environment-independence and storage-family dimensions.

        This is the test that turns 'the Arrow tests are unstable' into a
        defect with an address. It sets the global BOTH WAYS around
        otherwise identical work and requires the same answer. Because it
        pins the option itself, it gives the same result on pandas 1.5.3 and
        on 3.0 — the previous tests inherited whatever the machine's default
        was, which is what made them look flaky."""
        pytest.importorskip("pyarrow")

        def authority_under(storage):
            """The whole gather runs inside the option context, because the
            option is consulted at MATERIALIZATION time by
            `_authority_is_exactly_representable`, not when the column is
            built. A first draft of this test asked a bare physical column
            instead and XPASSED — AD-19 source 2 is unimplemented, so both
            branches answered UNKNOWN and the divergence was invisible.
            Caught by running it."""
            with pd.option_context("mode.string_storage", storage):
                m = A.AliasDataFrame(pd.DataFrame({
                    "k": np.array([0, 9], np.int64),
                    "x": np.array([10, 20], np.int64)}))
                ch = A.AliasDataFrame(pd.DataFrame({
                    "k": np.array([0], np.int64)}))
                ch.df["v"] = pd.array(["a"], dtype="string[pyarrow]")
                m.register_subframe("S", ch, index_columns=["k"])
                m.set_subframe_fill("S", fill_missing="")
                m.add_alias("d", "S.v")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.materialize_alias("d")
                return m.get_dtype_authority("d")

        as_python = authority_under("python")
        as_pyarrow = authority_under("pyarrow")

        assert as_python.known == as_pyarrow.known, (
            f"the same string[pyarrow] column is authoritative under "
            f"mode.string_storage='pyarrow' ({as_pyarrow!r}) and not under "
            f"'python' ({as_python!r}) — a process-wide option decided a "
            f"property of the column [{_arrow_env()}]")
        assert as_python.dtype == as_pyarrow.dtype, (
            f"the recorded dtype depends on the global option: "
            f"{as_python.dtype!r} vs {as_pyarrow.dtype!r} [{_arrow_env()}]")
        assert as_python.known, (
            f"and neither branch may be UNKNOWN once the column really is "
            f"materialized in this frame: {as_python!r} [{_arrow_env()}]")

    def test_b32b_14h_the_two_arrow_mechanisms_stay_distinguishable(self):
        """Passing control on the storage-family dimension, and the guard on
        the mistake that cost revision 3 a false finding.

        `int64[pyarrow]` is a `pd.ArrowDtype`. `string[pyarrow]` is a
        `pd.StringDtype` whose storage happens to be Arrow. They are
        different classes with different `str()` forms, and any future
        helper that treats them as one family reintroduces the defect."""
        pytest.importorskip("pyarrow")
        typed = pd.array([1], dtype="int64[pyarrow]").dtype
        storage = pd.array(["a"], dtype="string[pyarrow]").dtype

        assert isinstance(typed, pd.ArrowDtype), _arrow_env()
        assert not isinstance(storage, pd.ArrowDtype), (
            f"string[pyarrow] is a StringDtype, not an ArrowDtype "
            f"[{_arrow_env()}]")
        assert getattr(storage, "storage", None) == "pyarrow", _arrow_env()
        assert _is_arrow_backed(typed) and _is_arrow_backed(storage), (
            f"the classifier must recognise BOTH mechanisms [{_arrow_env()}]")
        assert str(typed) != str(storage).replace("string", "int64"), (
            "this assertion exists only to record that str() forms differ: "
            f"{str(typed)!r} vs {str(storage)!r} [{_arrow_env()}]")
    # ---- family 4: sparse fill without a dense temporary -----------------

    def test_b32b_15_sparse_fill_already_preserves_the_sparse_dtype(self):
        """ALREADY SATISFIED ON ENTRY. Also XPASSED when written as an xfail.
        AD-15 preserves sparsity through the gather; what B3.2b actually owns
        is the DENSE TEMPORARY, not the dtype — see `b32b_15b`."""
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64), "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = pd.arrays.SparseArray(np.array([3.0]), fill_value=0.0)
        m.register_subframe("S", ch, index_columns=["k"])
        m.set_subframe_fill("S", fill_missing=0.0)
        m.add_alias("d", "S.v")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("d")
        assert isinstance(m.df["d"].dtype, pd.SparseDtype)

    def test_b32b_15b_sparse_fill_never_densifies(self):
        """MR-P0-1, the blocking finding. Revision 2 asserted

            not re.search(r"densify-fill-resparsify", module_source)

        so DELETING OR RENAMING A COMMENT turned it green with zero
        executable change — in the same file whose header adopts the rule
        that an xfail must fail for the defect its reason names. Three seats
        found it independently.

        The replacement is executable and cannot be satisfied by any comment:
        `Series.to_numpy` is made to raise for the duration of one direct
        `_place_fill` call on a sparse series with a fill knob. If the
        implementation still densifies, the spy trips. If B3.2b lands the
        sparse-index reconstruction, nothing dense is ever built and the call
        returns normally.

        A peak-RSS assertion at the sizes where the 5.25x is visible was
        rejected: this project already has one timing-sensitive probe that
        flaps, and MR explicitly said no flaky memory threshold is required.

        SCOPE (STEP 4 v03, GPT29 `F3`): this test proves the property for
        ONE fixture. The name is a family-level claim, and it is literally
        true only because v03 made a declining reconstruction REFUSE rather
        than densify — so no sparse input can reach the dense route at all.
        `b32b_15c` owns the family: the natural matrix that reconstructs, and
        the forced decline that refuses without building anything dense.
        """
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1.0, 2.0, 3.0])}))
        sparse = pd.Series(pd.arrays.SparseArray(
            np.array([1.0, 0.0, 3.0]), fill_value=0.0))
        mask = np.array([False, True, False])

        tripped = []
        real_to_numpy = pd.Series.to_numpy

        def _no_densify(self, *a, **k):
            tripped.append(True)
            raise AssertionError("_place_fill densified a sparse column")

        pd.Series.to_numpy = _no_densify
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                out = m._place_fill(sparse, mask, 9.0, "fill_missing", "S", "v")
        finally:
            pd.Series.to_numpy = real_to_numpy

        assert not tripped, (
            "_place_fill built a dense temporary; AD-15 assigns the "
            "non-densifying sparse-index reconstruction to B3.2b")
        assert isinstance(out.dtype, pd.SparseDtype), (
            "the result must still be sparse")
        assert float(out.values[1]) == 9.0, "the fill must still be applied"

    def test_b32b_15c_sparse_reconstruction_refuses_instead_of_densifying(self):
        """STEP 4 v03 — GPT29 `F3`, his recommended branch, architect
        Decission 1.

        v01's criterion was NAMED "never densifies" while production
        densified on any reconstruction exception, so the criterion could
        close while the AD-15 cost quietly returned. This test owns the two
        halves that make the name literal:

        1. THE NATURAL MATRIX RECONSTRUCTS. GPT29 asked for `Sparse[int64]`
           and `Sparse[float64]` with `fill == fill_value` and
           `fill != fill_value`; this widens that to the shapes that could
           plausibly defeat an index-only rewrite — BlockIndex as well as
           IntIndex, the all-fill and no-fill extremes, object and bool
           storage, and masks at both ends and over the whole column. Every
           case must reconstruct, keep its exact sparse dtype, and produce
           the right values. That matrix is also the EVIDENCE that refusing
           costs nothing real: the dense route was already unreachable for
           sparse input in practice.
        2. A DECLINING RECONSTRUCTION REFUSES. Forced to fail, it must raise
           an ADF error and the dense-conversion spy must NOT run — the
           mutation control GPT29 specified, and the half no fixture-based
           test can supply on its own.
        """
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1.0])}))

        def _sp(vals, fv, kind="integer", dtype=None):
            return pd.Series(pd.arrays.SparseArray(
                np.array(vals) if dtype is None
                else np.array(vals, dtype=dtype),
                fill_value=fv, kind=kind))

        matrix = [
            ("float64 IntIndex", lambda: _sp([1.0, 0.0, 3.0, 0.0], 0.0),
             [9.0, 0.0, -1.5]),
            ("float64 BlockIndex",
             lambda: _sp([1.0, 1.0, 0.0, 0.0, 2.0, 2.0], 0.0, kind="block"),
             [9.0, 0.0]),
            ("float64 nan fill", lambda: _sp([1.0, np.nan, 3.0], np.nan),
             [9.0, np.nan]),
            ("int64 fill 0", lambda: _sp([1, 0, 3, 0], 0), [9, 0, -7]),
            ("int64 fill -1", lambda: _sp([1, -1, 3], -1), [9, -1]),
            ("float32", lambda: _sp([1.0, 0.0, 3.0], np.float32(0),
                                    dtype=np.float32), [9.0, 0.0]),
            ("bool", lambda: _sp([True, False, True], False), [True, False]),
            ("object", lambda: _sp(["a", "", "c"], "", dtype=object),
             ["z", ""]),
            ("all fill", lambda: _sp([0.0, 0.0, 0.0], 0.0), [9.0, 0.0]),
            ("no fill", lambda: _sp([1.0, 2.0, 3.0], 0.0), [9.0, 0.0]),
        ]

        checked = 0
        for label, make, fills in matrix:
            for fill in fills:
                n = len(make())
                for pattern in ("first", "last", "all", "interior"):
                    mask = np.zeros(n, dtype=bool)
                    if pattern == "first":
                        mask[0] = True
                    elif pattern == "last":
                        mask[-1] = True
                    elif pattern == "all":
                        mask[:] = True
                    else:
                        mask[min(1, n - 1)] = True

                    src = make()
                    want = np.asarray(src.values, dtype=object).copy()
                    want[mask] = fill
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        out = m._place_fill(src, mask, fill,
                                            "fill_missing", "S", "v")
                    checked += 1
                    where = f"{label} fill={fill!r} mask={pattern}"
                    assert isinstance(out.dtype, pd.SparseDtype), (
                        f"{where}: lost the sparse dtype -> {out.dtype}")
                    assert out.dtype == make().dtype, (
                        f"{where}: sparse dtype changed "
                        f"{make().dtype} -> {out.dtype}")
                    got = np.asarray(out.values, dtype=object)
                    for a, b in zip(want, got):
                        assert (pd.isna(a) and pd.isna(b)) or a == b, (
                            f"{where}: expected {list(want)} got {list(got)}")
        assert checked >= 80, (
            f"the matrix must actually be driven; only {checked} cases ran")

        # 2. forced decline -> REFUSE, and nothing dense is built.
        # The injection replaces the `IntIndex` module for the duration, which
        # `_place_fill` imports INSIDE its own try block — so only the
        # reconstruction is affected, not `_coerce_fill_to_dtype` and not the
        # dense route we are asserting never runs.
        import sys as _sys
        real_mod = _sys.modules["pandas._libs.sparse"]
        real_to_numpy = pd.Series.to_numpy
        densified = []

        class _Declines:
            def __getattr__(self, _n):
                raise RuntimeError("injected: sparse reconstruction declined")

        def _spy(self, *a, **k):
            densified.append(True)
            return real_to_numpy(self, *a, **k)

        src = _sp([1.0, 0.0, 3.0], 0.0)
        _sys.modules["pandas._libs.sparse"] = _Declines()
        pd.Series.to_numpy = _spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(Exception) as exc:
                    m._place_fill(src, np.array([False, True, False]), 9.0,
                                  "fill_missing", "S", "v")
        finally:
            pd.Series.to_numpy = real_to_numpy
            _sys.modules["pandas._libs.sparse"] = real_mod

        assert not densified, (
            "a declining reconstruction still built a dense temporary; that "
            "is the AD-15 cost returning silently")
        assert type(exc.value).__name__ != "AssertionError", (
            f"the refusal must be a real error, not a bare assert: {exc.value}")
        text = str(exc.value)
        assert "fill_missing" in text and "5.25" in text, (
            "the refusal must name the knob and the cost it is refusing to "
            f"pay, so the caller can decide: {text}")

    # ---- family 5: conditional / fallback row-wise provenance ------------

    def test_b32b_16_conditional_requiredness_is_row_wise(self):
        """`where(cond, a, S.v)` — the rows taking `a` never consult `S.v`,
        so their value is defined even though `S.v` is absent for some row.
        Round 11 deliberately treats requiredness as a syntactic union and
        B3.2b owns the row-wise refinement."""
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64),
            "a": np.array([100, 200], np.int64),
            "c": np.array([False, True])}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("d", "where(c, a, S.v)", dtype="int64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("d")
        assert [int(v) for v in m.df["d"].values] == [3, 200]

    def test_b32b_16b_conditional_still_refuses_a_selected_absent_operand(self):
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64),
            "a": np.array([100, 200], np.int64),
            "c": np.array([True, False])}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("d", "where(c, a, S.v)", dtype="int64")
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_alias("d")

    #: The ratified family-5 forms. `where` alone is one of three named in the
    #: scope text; MR-P1-5 requires the non-`where` forms too, or an explicit
    #: support/refusal matrix. This is the matrix.
    FALLBACK_FORMS = ("where(c, a, S.v)", "fillna(S.v, a)",
                      "coalesce(S.v, a)", "select(c, a, S.v)")

    def test_b32b_16c_every_fallback_form_is_supported_or_refused_cleanly(self):
        """MR-P1-5 / GPT27: revision 2 covered only `where`, so family 5 could
        have closed while `fillna` and `coalesce` still leaked a raw
        NameError to the physicist writing the alias."""
        mod = _adf_module()
        root = mod.ADFError
        undisposed = []
        for form in self.FALLBACK_FORMS:
            m = A.AliasDataFrame(pd.DataFrame({
                "k": np.array([0, 9], np.int64),
                "a": np.array([100, 200], np.int64),
                "c": np.array([False, True])}))
            ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
            ch.df["v"] = np.array([3], dtype=np.int64)
            m.register_subframe("S", ch, index_columns=["k"])
            m.add_alias("d", form, dtype="int64")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.materialize_alias("d")
            except NameError as exc:
                undisposed.append(f"{form}: raw NameError ({exc})")
            except root:
                pass          # explicitly refused — an acceptable disposition
            except Exception as exc:
                undisposed.append(f"{form}: {type(exc).__name__} ({exc})")
        assert not undisposed, (
            "fallback forms with no disposition: " + "; ".join(undisposed))

    def _fam5(self, cond, child_extra=None):
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64),
            "a": np.array([100, 200], np.int64),
            "c": np.array(cond)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        if child_extra:
            for _n, _v in child_extra.items():
                ch.df[_n] = np.array(_v)
        m.register_subframe("S", ch, index_columns=["k"])
        return m

    def _fam5_run(self, m, expr, dtype="int64"):
        m.add_alias("d", expr, dtype=dtype)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                m.materialize_alias("d")
            except Exception as exc:      # noqa: BLE001 - the subject
                return exc
        return [int(v) for v in m.df["d"].values]

    def test_b32b_16d_row_wise_requiredness_over_the_whole_condition_range(
            self):
        """STEP 5b — the boundary matrix, and it caught a real defect in my
        own first implementation.

        `b32b_16` drives ONE condition, `[False, True]`, where each branch is
        taken by exactly one row. That leaves the two ENDS of the range
        untested, and the all-true end was wrong: `where(c, a, S.v)` with `c`
        everywhere true never looks at `S.v` at all, and the first version
        still refused with "1 row(s) with no defined value". The cause was
        that a name the walk never reaches had NO entry, and no entry meant
        "consulted everywhere" rather than "consulted nowhere".

        One condition value is not a criterion for a row-wise property. The
        full range is.
        """
        cases = [
            ([False, True], [3, 200], "one branch each — the b32b_16 case"),
            ([True, True], [100, 200], "NEVER consults S.v — must succeed"),
            ([True, False], None, "row 1 selects the absent operand"),
            ([False, False], None, "row 1 selects it too"),
        ]
        for cond, expect, label in cases:
            got = self._fam5_run(self._fam5(cond), "where(c, a, S.v)")
            if expect is None:
                assert isinstance(got, Exception), (
                    f"{label}: expected a refusal, got {got}")
                assert isinstance(got, _adf_module().RowLevelMissingnessError), (
                    f"{label}: {type(got).__name__} is not row-level "
                    f"missingness")
            else:
                assert got == expect, f"{label}: {got}"

        # the refinement must not leak to expressions with NO conditional
        for expr in ("S.v", "S.v * 2", "S.v + a"):
            got = self._fam5_run(self._fam5([False, True]), expr)
            assert isinstance(got, _adf_module().RowLevelMissingnessError), (
                f"{expr!r} has no conditional, so every row consults S.v and "
                f"it must still refuse: {got}")

    def test_b32b_16e_the_condition_is_an_operand_too(self):
        """STEP 5b — the soundness question row-wise requiredness creates.

        If a row's CONDITION is itself undefined, ADF does not know which
        branch that row takes, so it cannot claim the row is defined — and
        picking a branch anyway would be a silent wrong result of exactly the
        kind AD-19 forbids, arriving through a conditional instead of through
        a default fill.

        The walk therefore visits the condition with the FULL reachability
        mask, never a narrowed one. This is the test that would fail first if
        anyone ever "optimised" that.
        """
        m = self._fam5([False, True], child_extra={"flag": [True]})
        got = self._fam5_run(m, "where(S.flag, a, a * 2)")
        assert isinstance(got, _adf_module().RowLevelMissingnessError), (
            "a row whose CONDITION is undefined must refuse; ADF cannot know "
            f"which branch it takes: {got}")

    def test_b32b_16f_the_forms_agree_and_compose(self):
        """STEP 5b. `where`, `select`, `fillna` and `coalesce` are one
        mechanism, not four: the fallback pair is rewritten into `where`
        against the operand's own undefinedness mask, so there is a single
        conditional to reason about and the row-wise walker needs no special
        case for them. If they ever disagree, that rewrite has drifted.

        Composition is asserted because the walker carries reachability DOWN
        a tree, so a conditional inside a conditional, and a conditional
        inside arithmetic, are the shapes where a per-node bug would show.
        """
        for form in ("where(c, a, S.v)", "select(c, a, S.v)",
                     "fillna(S.v, a)", "coalesce(S.v, a)"):
            got = self._fam5_run(self._fam5([False, True]), form)
            assert got == [3, 200], f"{form}: {got}"

        assert self._fam5_run(
            self._fam5([False, True]),
            "where(c, a, where(c, a, S.v))") == [3, 200], "nested"
        assert self._fam5_run(
            self._fam5([False, True]), "where(c, a, S.v) + 1") == [4, 201], (
            "a conditional inside arithmetic")

        # and with no subframe at all the forms are ordinary functions
        p = A.AliasDataFrame(pd.DataFrame({
            "x": np.array([1.0, 2.0, 3.0]),
            "c": np.array([True, False, True])}))
        p.add_alias("q", "where(c, x, -x)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p.materialize_alias("q")
        assert list(p.df["q"].values) == [1.0, -2.0, 3.0]
        assert list(np.asarray(p.eval("where(c, x, 0.0)"))) == [1.0, 0.0, 3.0]

    def test_b32b_16l_adf_owns_its_generated_call(self):
        """STEP 5b v03 — `P0-1` / `P2-1`. EXECUTED on the v02 bytes.

        v02's fallback emitted a PUBLIC `where(...)` call, so ADF's own
        generated operation went back through a namespace the user owns:

            register_function("where", lambda c, a, b: b)
            fillna(S.v, a)          -> published [3, 3]
            a COLUMN named `where`  -> TypeError

        And ADF-ness was a plain attribute, so a user callable carrying
        `_adf_conditional` inherited ADF's proof and published [3, 3] too.

        v03 emits the generated call against a freshly generated private name
        bound to the primitive itself, and decides ADF-ness by `is`. A public
        name the user has taken is then simply not ADF's function, so the
        proof declines and the ordinary provenance guard runs.
        """
        mod = _adf_module()

        def hijack(cond, a, b):
            return b

        for form in ("fillna", "coalesce"):
            m = self._fam5([False, True])
            m.register_function("where", hijack)
            got = self._fam5_run(m, f"{form}(S.v, a)")
            assert got == [3, 200], (
                f"a user-owned `where` captured ADF's generated {form} "
                f"rewrite: {got}")

        # a physical column of that name must not break the rewrite either
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64),
            "a": np.array([100, 200], np.int64),
            "c": np.array([False, True]),
            "where": np.array([1, 1], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        assert self._fam5_run(m, "fillna(S.v, a)") == [3, 200], (
            "a COLUMN named `where` broke ADF's generated rewrite")

        # `P2-1`: the marker is not the identity
        def spoof(cond, a, b):
            return b
        spoof._adf_conditional = "select"
        m = self._fam5([False, True])
        m.register_function("where", spoof)
        got = self._fam5_run(m, "where(c, a, S.v)")
        assert isinstance(got, mod.ADFError), (
            f"a callable carrying the marker inherited ADF's proof: {got}")

    def test_b32b_16m_nothing_unproven_is_hoisted(self):
        """STEP 5b v03 — `P0-2` / `P1-1`. Both EXECUTED on the v02 bytes.

        Binding evaluates a subexpression EARLY and removes it from the tree.
        Both consequences were defects:

            P0-2  `touch(a) + where(sel(a), a, S.v)` called `sel` BEFORE
                  `touch`. `register_function` promises neither purity nor
                  order-independence.
            P1-1  the row-local gate inspects the rewritten tree:
                      where(z - z.mean() > 0, a, v__S)   row-local FALSE
                      where(__adf_bound_0__, a, v__S)    row-local TRUE
                  Fix11c's fail-closed gate was disarmed by the rewrite that
                  was supposed to be safe.

        One rule closes both: hoist only what `_expression_is_row_local`
        already admits — which excludes reductions AND every registered
        callable — and let the gate inspect the UNBOUND expression. What
        binding removes from the tree was proven safe before it was removed.
        """
        mod = _adf_module()

        # P1-1: the gate must refuse the same thing before and after binding.
        # v04's proof is runtime-identity-aware, so supply a genuine default
        # namespace here; otherwise "no runtime binding supplied" would make
        # this assertion pass before it ever reached the reduction we intend
        # to test.
        _probe = self._fam5([False, True])
        _trusted = _probe._default_functions(include_registered=False)
        assert not _adf_module()._expression_is_row_local(
            "where(z - z.mean() > 0, a, v__S)",
            env=dict(_trusted), trusted_functions=_trusted)[0], (
            "fixture: the selector must be non-row-local because of mean(), "
            "not because runtime identity was omitted")
        m = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64),
            "a": np.array([100, 200], np.int64),
            "z": np.array([1.0, 5.0])}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        m.register_subframe("S", ch, index_columns=["k"])
        got = self._fam5_run(m, "where(z - z.mean() > 0, a, S.v)")
        assert isinstance(got, mod.ADFError), (
            f"a non-row-local selector was laundered behind a bound name and "
            f"the Fix11c gate never saw it: {got}")

        # P0-2: evaluation order is the source order, or nothing is hoisted
        order = []

        def touch(x):
            order.append("touch")
            return np.zeros(len(x), dtype=np.int64)

        def sel(x):
            order.append("sel")
            return np.array([False, True])

        m = self._fam5([False, True])
        m.register_function("touch", touch)
        m.register_function("sel", sel)
        got = self._fam5_run(m, "touch(a) + where(sel(a), a, S.v)")
        assert order[:2] == ["touch", "sel"], (
            f"the selector was hoisted ahead of an earlier registered call, "
            f"changing observable evaluation order: {order}")
        assert isinstance(got, mod.ADFError), (
            "an unhoisted selector cannot be refined, so the row stays "
            f"undefined and must refuse: {got}")

    def test_b32b_16n_unknown_na_fails_closed(self):
        """STEP 5b v03 — `P1-2` (GPT29, Fabble5_7).

        v02 collapsed "`pd.isna` raised" into the same `None` as "nothing is
        NA", so an operand ADF could not inspect was treated as fully
        defined — failing OPEN in the one place this increment is about
        failing closed. Three states now, and UNKNOWN declines the binding.
        """
        assert _adf_module()._ISNA_UNKNOWN is not None

        # A structured/record dtype: `pd.isna` raises at library level.
        # Fabble5_7 enumerated it; my first fixture here was a hand-rolled
        # object for which `pd.isna` quietly answers False, which proves
        # nothing — the test needs a value the LIBRARY cannot answer for.
        unanswerable = np.zeros(2, dtype=[("a", "i4"), ("b", "f4")])
        with pytest.raises(TypeError):
            pd.isna(unanswerable)

        got = _adf_module()._isna_mask(unanswerable)
        assert got is _adf_module()._ISNA_UNKNOWN, (
            f"an uninspectable operand must be UNKNOWN, not 'no NA': {got!r}")
        assert _adf_module()._isna_mask(np.array([1.0, 2.0])) is None, (
            "…and a genuinely NA-free operand must still be None")

    def test_b32b_16o_internal_primitive_identity_is_not_user_namespace(self):
        """STEP 5b v04 — private proof identity cannot be overwritten publicly.

        v03 stored the selector/fallback identities under
        ``__adf_*_primitive__`` keys and then merged ``register_function`` on
        top.  The spelling looked private, but the public API accepts arbitrary
        names.  Replacing the selector key made ADF ``fillna`` publish the
        unrepaired placeholder while the proof blessed the same user callable.

        v04 keeps the identities outside the evaluation namespace entirely.
        Registering either old key may create an ordinary user function binding,
        but it cannot change what ADF's own fallback implementation calls.
        """
        def wrong_select(cond, a, b):
            return b

        def wrong_fallback(x, f):
            return x

        for key, fn in (("__adf_select_primitive__", wrong_select),
                        ("__adf_fallback_primitive__", wrong_fallback)):
            for form in ("fillna", "coalesce"):
                m = self._fam5([False, True])
                m.register_function(key, fn)
                got = self._fam5_run(m, f"{form}(S.v, a)")
                assert got == [3, 200], (
                    f"registered {key!r} changed ADF's internal {form} "
                    f"semantics: {got}")

        base = self._fam5([False, True])._default_functions(
            include_registered=False)
        assert "__adf_select_primitive__" not in base
        assert "__adf_fallback_primitive__" not in base

    def test_b32b_16p_row_locality_follows_runtime_callable_identity(self):
        """STEP 5b v04 — a trusted FUNCTION NAME is not a trusted callable.

        v03 gated hoisting with ``_expression_is_row_local`` but that helper
        classified calls by spelling.  A registered stateful function named
        ``sin`` therefore inherited numpy.sin's proof and moved ahead of an
        earlier ``touch`` call.  The same name-only hole also let a registered
        ``where`` pass the primary Fix11c gate, leaving finite A/B probing as
        the only defence even though that probe is explicitly not proof.
        """
        mod = _adf_module()
        order = []
        state = {"touched": False}

        def touch(x):
            order.append("touch")
            state["touched"] = True
            return np.zeros(len(x), dtype=np.int64)

        def user_sin(x):
            order.append("sin")
            # Before touch row 1 selects clean ``a``; after touch it selects
            # absent S.v.  Hoisting therefore changes the semantic branch.
            return (np.array([1.0, 1.0]) if not state["touched"]
                    else np.array([1.0, 0.0]))

        m = self._fam5([False, True])
        m.register_function("touch", touch)
        m.register_function("sin", user_sin)
        got = self._fam5_run(
            m, "touch(a) + where(sin(a) > 0, a, S.v)")
        assert order[:2] == ["touch", "sin"], (
            f"registered sin inherited the default sin row-local proof and "
            f"was hoisted: {order}")
        assert isinstance(got, mod.ADFError), (
            f"an unproven registered callable must refuse, not publish: {got}")

        # Primary-gate check: a registered callable under a conditional spelling
        # must not become row-local merely because its NAME is on ADF's list.
        # Keep the user function deliberately cross-row; the primary gate must
        # reject before any finite placeholder coincidence could be trusted.
        def cross_row_where(cond, a, b):
            return np.full(len(a), int(np.max(np.asarray(b)) < 10), dtype=np.int64)

        m = self._fam5([False, True])
        m.register_function("where", cross_row_where)
        got = self._fam5_run(m, "where(c, a, S.v)")
        assert isinstance(got, mod.ADFProvenanceUnsupportedError), (
            f"registered where was certified row-local by spelling: {got}")
        assert "not bound to" in str(got) or "not proven row-local" in str(got)

    def test_b32b_16q_unknown_na_refuses_on_the_public_fallback_path(self):
        """STEP 5b v04 — UNKNOWN must stay fail-closed through its consumer.

        v03 made ``_isna_mask`` three-state, but the runtime fallback primitive
        still passed the UNKNOWN sentinel to ``np.where``.  Depending on dtype
        that either crashed in numpy or treated the opaque object as a truthy
        scalar and selected the fallback everywhere.  This test calls the public
        expression path, not only the helper that produces the sentinel.
        """
        mod = _adf_module()
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))

        def structured(x):
            out = np.zeros(len(x), dtype=[("a", "i4"), ("b", "f4")])
            out["a"] = np.asarray(x)
            return out

        m.register_function("structured", structured)
        for form in ("fillna", "coalesce"):
            with pytest.raises(mod.ADFProvenanceUnsupportedError,
                               match="cannot determine row-wise missingness"):
                m.eval(f"{form}(structured(x), x)")

    def test_b32b_16r_arithmetic_gating_refusal_names_the_conditional_remedy(self):
        """F6: a correct fail-closed refusal must tell the user how to
        express row-wise requiredness explicitly.  Arithmetic gating evaluates
        both operands, so ``c*a + (1-c)*S.v`` must still refuse when ``S.v``
        is structurally absent; the diagnostic should point to ``where``/
        ``select`` rather than leaving the supported form undiscoverable.
        """
        mod = _adf_module()
        m = self._fam5([True, True])
        m.add_alias("d", "c*a + (1-c)*S.v", dtype="int64")
        with pytest.raises(mod.RowLevelMissingnessError) as err:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.materialize_alias("d")
        msg = str(err.value)
        assert "where(condition" in msg
        assert "row-wise requiredness" in msg
        assert "arithmetic gating" in msg

    def test_b32b_16h_the_proof_follows_the_BINDING_not_the_name(self):
        """STEP 5b v02 — `F2` (GPT33, GPT29). EXECUTED on the v01 bytes.

        `register_function()` accepts arbitrary names and registered functions
        WIN the evaluation namespace. v01's analyzer keyed on the AST name
        `where`, so a user's callable produced the value while ADF's
        branch-selection semantics produced the proof. Measured on v01:

            register_function("where", lambda c, a, b: b)
            d = where(c, a, S.v)        -> published [3, 3]

        The user's function returns `b` on EVERY row, so row 1 truly reads the
        absent `S.v`. The analyzer believed row 1 took `a`, cleared the
        residual mask, and the Fix11c fail-closed gate never ran — the gate
        exists precisely for unclassified registered callables, and the bad
        refinement disarmed it before it could fire.

        The refinement is now conditional on the runtime object being ADF's
        own primitive, so a user override falls back to the pre-5b syntactic
        union and the ordinary provenance guard does its job.
        """
        mod = _adf_module()
        for name in ("where", "select"):
            m = self._fam5([False, True])
            m.register_function(name, lambda cond, a, b: b)
            got = self._fam5_run(m, f"{name}(c, a, S.v)")
            assert isinstance(got, mod.ADFError), (
                f"a registered {name!r} must not inherit ADF's row-wise "
                f"proof; row 1 really does read the absent operand: {got}")
        for name in ("fillna", "coalesce"):
            m = self._fam5([False, True])
            m.register_function(name, lambda x, f: x)
            got = self._fam5_run(m, f"{name}(S.v, a)")
            assert isinstance(got, mod.ADFError), (
                f"a registered {name!r} returns its first operand unrepaired, "
                f"so the absent row is still absent: {got}")

    def test_b32b_16i_the_selector_is_evaluated_exactly_once(self):
        """STEP 5b v02 — `F3` (GPT33; GPT34 independently). EXECUTED on v01.

        v01 evaluated the selector twice: once inside the expression that
        produced the value, once again to derive reachability. The public
        `register_function()` contract promises no purity or determinism, so
        the two evaluations could disagree. Measured on v01:

            call 1 -> row 1 selects S.v   (the value contains the placeholder)
            call 2 -> row 1 selects a     (the proof clears the dependency)
            published [3, 3]

        The selector is now evaluated ONCE and bound; the proof reads the
        bound array rather than re-running the source. The call count is
        asserted because "same value" and "same evaluation" are different
        claims, and only the second one is structural.
        """
        mod = _adf_module()
        calls = {"n": 0}

        def flip(x):
            calls["n"] += 1
            return np.array([False, calls["n"] > 1])

        m = self._fam5([False, True])
        m.register_function("flip", flip)
        got = self._fam5_run(m, "where(flip(a), a, S.v)")
        assert isinstance(got, mod.ADFError), (
            f"the FIRST evaluation selects the absent operand on row 1, so "
            f"the row is undefined however the second one votes: {got}")
        assert calls["n"] == 1, (
            f"the selector must be evaluated once and reused; a proof derived "
            f"from a second evaluation is a proof about a different "
            f"expression (called {calls['n']} times)")

    def test_b32b_16j_fallback_uses_RESULT_level_missingness(self):
        """STEP 5b v02 — `F1` (GPT34, GPT29; GPT33 on the NA half).
        EXECUTED on the v01 bytes.

        v01 derived a fallback's trigger from the SYNTACTIC UNION of masks for
        names appearing in its first operand. That is a different quantity
        from "is this operand's RESULT undefined on this row", and STEP 5b
        itself is what makes them differ — a conditional operand can already
        be defined on a row whose masked name is absent. Measured on v01:

            fillna(where(c, a, S.v), f)   c = [False, True]
            expected [3, 200]   ->   published [3, 900]

        Row 1 takes `a` inside the conditional, so nothing is missing and the
        fallback must not fire. The trigger is now the operand's result-level
        undefinedness — the reachability walk run on the operand itself — OR
        its own NA, so `fillna` means one thing regardless of dtype.
        """
        for form in ("fillna", "coalesce"):
            got = self._fam5_run(self._fam5([False, True]),
                                 f"{form}(where(c, a, S.v), f)")
            assert got == [3, 200], (
                f"{form} over a conditional operand: row 1 is already defined "
                f"by the conditional and must not be overwritten: {got}")
            assert self._fam5_run(self._fam5([False, True]),
                                  f"{form}(S.v, a)") == [3, 200], form

        # ONE meaning of `fillna`, whatever the operand's dtype. v01 guarded
        # on `dtype.kind in "fc"` and therefore did nothing at all for object
        # and for every nullable dtype — including the row it exists to fix.
        for label, vals in (("Int64", pd.array([pd.NA], dtype="Int64")),
                            ("object", np.array([np.nan], dtype=object)),
                            ("float64", np.array([np.nan]))):
            m = A.AliasDataFrame(pd.DataFrame({
                "k": np.array([0, 9], np.int64),
                "a": np.array([100.0, 200.0])}))
            ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
            ch.df["v"] = vals
            m.register_subframe("S", ch, index_columns=["k"])
            got = self._fam5_run(m, "fillna(S.v, a)")
            assert [float(v) for v in got] == [100.0, 200.0], (
                f"{label}: fillna silently did nothing — a matched NA on row "
                f"0 and an absent row 1 must BOTH take the fallback: {got}")

    def test_b32b_16k_generated_bindings_and_dependent_aliases_survive(self):
        """STEP 5b v02 — `F4` (GPT34, GPT29). Two separate collision surfaces,
        so two separate assertions.

        FIRST: the namespace applies functions AFTER columns AND after
        `context_override`, which is how a dependent alias reaches a batched
        evaluation before publication. v01 guarded only `self.df.columns`.
        Measured on v01: an alias named `where` through the batched path
        raised "unsupported operand type(s) for +: 'function' and 'int'".

        SECOND: the rewrite generates names for the arrays it binds. A fixed
        prefix is an assumption about the user's column names, not a
        guarantee, so the generator bumps until the name is actually free.
        """
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1.0, 2.0])}))
        m.add_alias("where", "x + 1")
        m.add_alias("q", "where + 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_aliases(pattern="q")
        assert list(m.df["q"].values) == [4.0, 5.0], (
            "an ALIAS named `where`, visible through context_override, was "
            "replaced by the new builtin")

        occupied = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64),
            "a": np.array([100, 200], np.int64),
            "c": np.array([False, True]),
            "__adf_bound_0__": np.array([7, 7], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        occupied.register_subframe("S", ch, index_columns=["k"])
        occupied.add_alias("d", "where(c, a, S.v)", dtype="int64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            occupied.materialize_alias("d")
        assert [int(v) for v in occupied.df["d"].values] == [3, 200]
        assert [int(v) for v in occupied.df["__adf_bound_0__"].values] == [7, 7], (
            "a generated binding name overwrote a real user column")

    def test_b32b_16g_a_column_named_where_is_not_shadowed(self):
        """STEP 5b. The eval namespace puts FUNCTIONS AFTER COLUMNS, so an
        unguarded injection of `where` would shadow a physicist's column of
        that name and silently change what their alias means. Four new names
        is four new collisions; the guard costs one `in`."""
        m = A.AliasDataFrame(pd.DataFrame({
            "where": np.array([1.0, 2.0]),
            "coalesce": np.array([10.0, 20.0])}))
        m.add_alias("q", "where + coalesce")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        assert list(m.df["q"].values) == [11.0, 22.0], (
            "a column named `where` must still be a column")

    # ---- family 8: bounded conversion-API audit --------------------------

    @pytest.mark.xfail(strict=True, reason=
        "B3.2b acceptance, family 8: apply_dtypes / convert_dtypes / "
        "convert_dtypes_pattern each route through the central conversion "
        "policy or carry a recorded disposition WITH A REASON. Measured "
        "baseline failure: none of the three references the resolver, and "
        "none carries a disposition marker.")
    def test_b32b_17_conversion_apis_have_a_recorded_disposition(self):
        """MR-P2-1: revision 2 accepted the bare token `B3.2b-DISPOSITION`,
        which a one-word comment satisfies — a rubber stamp. The marker must
        now be followed by a reason of real length on the same line, so
        'disposed' means somebody wrote down why."""
        import re as _re
        src = _adf_source_text()
        for api in ("apply_dtypes", "convert_dtypes", "convert_dtypes_pattern"):
            m = _re.search(r"def %s\(.*?(?=\n    def )" % api, src, _re.S)
            assert m, f"{api} not found"
            body = m.group(0)
            routed = ("_resolve_target_dtype" in body
                      or "_safe_dtype_cast" in body)
            reasoned = _re.search(r"B3\.2b-DISPOSITION[:\s]+(\S.{29,})", body)
            assert routed or reasoned, (
                f"{api} neither routes through central policy nor carries a "
                f"recorded B3.2b-DISPOSITION with a stated reason")

    # ---- family 6: the cast-site audit / one owner ------------------------

    #: Sites allowed to decide a dtype locally, each with the ADJUDICATED
    #: reason it is exempt (MR-P2-1: revision 2 listed bare names, so the
    #: whitelist could grow silently). B3.2b's audit is complete when every
    #: remaining site either appears here with a reason or routes through the
    #: resolver.
    CAST_SITE_WHITELIST = {
        "_resolve_target_dtype":
            "the owner itself — this IS the central decision",
        "_buffer_dtype_for_fill":
            "chooses a staging representation only; the target is resolved "
            "by _resolve_target_dtype before this is called",
        "_coerce_fill_to_dtype":
            "asks pandas whether a value is representable; does not choose "
            "a target",
        "_safe_dtype_cast":
            "applies an ALREADY-RESOLVED declared dtype; round 11f added the "
            "pandas-extension branch here",
        "_restore_exact_dtype":
            "AD-19 exact restoration of a recorded authority — by "
            "construction it may not consult anything else",
        "_canonical_dtype":
            "pure spelling normalisation, no policy",
        "_authority_is_exactly_representable":
            "a predicate over a recorded authority, no target chosen",
    }

    #: The classifications the scaffold expects production to use. Written
    #: independently of `DTYPE_SITE_DISPOSITION` so the two cannot agree by
    #: construction -- the same anti-drift shape as `EXPECTED_DISPOSITION`.
    DTYPE_SITE_KINDS = ("DECISION", "APPLICATION", "INSPECTION",
                        "EXTRACTION", "LOCAL")

    @staticmethod
    def _dtype_sites_from_ast(source, whitelist):
        """Enumerate dtype/array sites as (function, operation, ordinal).

        The ordinal is assigned in source order within the function, so every
        site is individually addressable. `values_call` is a mapping
        `.values()` -- distinguished STRUCTURALLY here, in the executable
        oracle, not merely asserted in prose (S2-P1-1). pandas `.values` is a
        property; `x.values()` that executes proves a mapping receiver.
        """
        import ast as _ast
        tree = _ast.parse(source)
        called = {id(n.func) for n in _ast.walk(tree)
                  if isinstance(n, _ast.Call)
                  and isinstance(n.func, _ast.Attribute)}
        sites = {}
        for fn in _ast.walk(tree):
            if not isinstance(fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            if fn.name in whitelist:
                continue
            hits = []
            for node in _ast.walk(fn):
                if not (isinstance(node, _ast.Attribute) and node.attr in (
                        "astype", "kind", "to_numpy", "values")):
                    continue
                op = ("values_call"
                      if node.attr == "values" and id(node) in called
                      else node.attr)
                hits.append((node.lineno, node.col_offset, op))
            per_op = {}
            for lineno, col, op in sorted(set(hits)):
                ordinal = per_op.get(op, 0)
                per_op[op] = ordinal + 1
                sites[f"{fn.name}:{op}:{ordinal}"] = (fn.name, op, lineno)
        return sites

    def test_b32b_12_every_dtype_site_is_classified_and_owned(self):
        """CLOSED BY B3.2b STEP 2 as a PER-SITE audit — S2-P1-1 corrected.

        The first version keyed by FUNCTION NAME and `break`-ed after the
        first matching operation, so a second unclassified cast inside an
        already-classified function passed. GPT32 demonstrated it by
        mutation; GPT31 found the same mechanism independently. Decision 6
        said "classify call sites, not merely syntax", and the CRR then
        claimed "every site classified" when it was every function. That
        overclaim is corrected here and in the CRR.

        The key is now `function:operation:ordinal`. Adding a site to
        AliasDataFrame.py produces a key the ledger does not carry and this
        test fails until somebody says what the site is.

        The count reduction from the old oracle (62 functions -> the real
        inventory) was BOOKKEEPING -- false positives removed, nothing fixed.
        The CRR says so in those words."""
        mod = _adf_module()
        ledger = getattr(mod, "DTYPE_SITE_DISPOSITION", None)
        assert ledger is not None, "production carries no dtype call-site ledger"

        sites = self._dtype_sites_from_ast(_adf_source_text(),
                                           self.CAST_SITE_WHITELIST)

        unclassified = sorted(set(sites) - set(ledger))
        assert not unclassified, (
            f"{len(unclassified)} dtype/array site(s) carry no adjudicated "
            f"disposition: {unclassified[:8]}")

        stale = sorted(set(ledger) - set(sites))
        assert not stale, (
            f"the ledger classifies sites that no longer exist, so it has "
            f"drifted from the code: {stale[:8]}")

        for key, entry in ledger.items():
            kind, reason = entry
            ok = kind in self.DTYPE_SITE_KINDS or kind.startswith("LATER:")
            assert ok, f"{key}: unknown classification {kind!r}"
            if kind.startswith("LATER:"):
                assert "STEP" in kind, (
                    f"{key}: a deferred site must name the owning step")
            assert isinstance(reason, str) and len(reason) >= 30, (
                f"{key}: classification {kind} carries no adjudicated reason")

    def test_b32b_12d_a_second_site_in_a_classified_function_is_caught(self):
        """The negative control GPT32 required — the mutation that used to
        pass.

        Take a function the ledger already classifies, give it ANOTHER site
        of the same operation, and the audit must fail. Under the old
        function-keyed oracle it passed, because the function name was
        already present and the walk stopped at the first hit.

        The mutation is applied to a COPY of the source text, so production
        is untouched."""
        mod = _adf_module()
        ledger = getattr(mod, "DTYPE_SITE_DISPOSITION", {})
        src = _adf_source_text()

        # find a classified function with an astype site, and inject a second
        victim = None
        for key in ledger:
            fn, op, _ = key.split(":")
            if op == "astype":
                victim = fn
                break
        assert victim, "no astype site to mutate"

        import ast as _ast
        tree = _ast.parse(src)
        target = next(n for n in _ast.walk(tree)
                      if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))
                      and n.name == victim)
        lines = src.split("\n")
        # Inject BEFORE the first body statement, at its own indentation.
        # A first draft used `body[-1].lineno`, which can land inside a
        # nested block and produce an IndentationError instead of a mutation.
        first = target.body[0]
        insert_at = first.lineno - 1                # 0-based, before it
        indent = " " * (len(lines[insert_at])
                        - len(lines[insert_at].lstrip()))
        mutated = lines[:insert_at] + [
            f"{indent}_b32b_12d_probe = __import__('numpy').array([1]).astype('int8')"
        ] + lines[insert_at:]
        mutated_src = "\n".join(mutated)

        sites = self._dtype_sites_from_ast(mutated_src, self.CAST_SITE_WHITELIST)
        unclassified = sorted(set(sites) - set(ledger))
        assert unclassified, (
            f"a second astype injected into {victim!r} produced no "
            f"unclassified site — the audit is not per-site and would let a "
            f"new cast hide behind a classified neighbour")
        assert any(u.startswith(f"{victim}:astype:") for u in unclassified), (
            f"the injected site was not attributed to {victim}: {unclassified[:4]}")

    def test_b32b_12c_no_duplicated_storage_family_test_remains(self):
        """The half of STEP 2 that is production progress rather than
        bookkeeping — and its contract, narrowed per `S2-P1-2`.

        "Does the AD-19 NumPy gap refusal apply?" was asked seven times as an
        inline `dtype.kind in "biu"`, and once as `"iub"` -- the same set
        spelled differently, which is how a storage-family rule ends up with
        seven independent answers.

        The helper was first called `_dtype_can_represent_gap`, which claims
        a general semantic property. GPT31, GPT29 and Fabble5_7 all pointed
        at `Sparse[int64]`: not an `np.dtype`, so it took the True branch,
        yet holding a gap would require widening to `Sparse[float64]`. The
        CODE was right at all seven sites -- each guards on
        `isinstance(np.dtype)` or receives an `np.asarray` result -- the
        CLAIM was wrong. Renamed to `_numpy_dtype_can_hold_gap`, where True
        for a non-NumPy dtype means only "not subject to the NumPy refusal;
        its own storage family owns the question", which is STEP 4."""
        import ast as _ast
        mod = _adf_module()
        pred = getattr(mod, "_numpy_dtype_can_hold_gap", None)
        assert pred is not None, "the storage-family predicate does not exist"

        # the NumPy domain -- the only thing this predicate adjudicates
        assert pred(np.dtype(np.float64)) is True
        assert pred(np.dtype(np.int64)) is False
        assert pred(np.dtype(np.uint32)) is False
        assert pred(np.dtype(bool)) is False
        # fixed-width NumPy strings are inside the NumPy domain and not biu;
        # their gap semantics are a storage-family question (STEP 4)
        assert pred(np.dtype("U8")) is True
        assert pred(np.dtype(object)) is True

        # outside the NumPy domain: True means "not the NumPy refusal", NOT
        # "can hold a gap in its own storage". Sparse[int64] is the case the
        # reviewers named, and it is recorded here as STEP 4's to answer.
        assert pred(pd.Int64Dtype()) is True, (
            "an extension dtype carries its own NA and is not subject to the "
            "NumPy refusal, even though its .kind is 'i'")
        assert pred(pd.SparseDtype(np.float64)) is True
        assert pred(pd.SparseDtype(np.int64)) is True, (
            "documented scope, not a semantic claim: Sparse[int64] is not "
            "subject to the NumPy refusal. Whether it can hold a gap while "
            "PRESERVING its storage dtype is a sparse-family question and "
            "STEP 4 owns it")

        # and no inline duplicate of the question survives
        offenders = []
        for fn in _ast.walk(_ast.parse(_adf_source_text())):
            if not isinstance(fn, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            if fn.name == "_numpy_dtype_can_hold_gap":
                continue
            for node in _ast.walk(fn):
                if not (isinstance(node, _ast.Compare)
                        and isinstance(node.left, _ast.Attribute)
                        and node.left.attr == "kind"):
                    continue
                for cmp_ in node.comparators:
                    text = _ast.unparse(cmp_).strip("'\"")
                    if text and set(text) <= set("biu"):
                        offenders.append(f"{fn.name}:{node.lineno}")
        assert not offenders, (
            "inline gap-refusal tests remain, so the question still has more "
            f"than one answer: {offenders}")

    # ---- S2-P0-1: the signed/unsigned join collision ----------------------

    def test_b32b_20_int64_key_normalization_is_value_preserving(self):
        """`S2-P0-1`, the predicate. CLOSED BY B3.2b STEP 2 CORRECTION.

        The Numba join fast path normalized both key sides with
        `.astype(np.int64)` and compared the results. `int64(-1)` and
        `uint64(2**64-1)` both become `-1`, so two distinct keys match.
        GPT32 executed it at `n >= NUMBA_MIN_ROWS` and saw false matches on
        every row where pandas, on the original typed keys, reports none.

        NOT A ROUND-TRIP TEST, and my first draft of the guard was exactly
        that and passed the collision case: two's complement gives
        `int64(-1)` and `uint64(2**64-1)` identical BITS, so
        `.astype(int64).astype(uint64)` returns the original. Bit
        preservation is what MAKES the collision; it cannot detect it. The
        question is whether the VALUE survives."""
        mod = _adf_module()
        f = getattr(mod, "_int64_key_cast_is_lossless", None)
        assert f is not None, "the key-cast guard does not exist"

        i64 = np.array([-1], np.int64)
        assert f(i64, np.array([5], np.int64)) is True
        assert f(i64, np.array([7], np.uint64)) is True, "small unsigned fits"
        assert f(i64, np.array([2**64 - 1], np.uint64)) is False, (
            "uint64(2**64-1) has a NEGATIVE int64 image and must fail closed")
        assert f(np.array([2**63], np.uint64)) is False, "the exact boundary"
        assert f(np.array([2**63 - 1], np.uint64)) is True, "just inside it"
        assert f(np.array([-1], np.int32),
                 np.array([4294967295], np.uint32)) is True, (
            "every 32-bit value fits int64; only uint64 can overflow it")
        assert f(np.array([], np.uint64)) is True, "an empty key set is safe"
        assert f(np.array([1.5])) is False, "a float key is not integer-castable"

    @needs_dfdraw
    def test_b32b_20b_mixed_sign_join_never_false_matches(self):
        """`S2-P0-1`, the behaviour, at Numba fast-path scale.

        Both orientations, plus an equal-key control so the guard is shown to
        gate the accelerator rather than disable it. The control matters as
        much as the collision cases: a guard that always refuses would pass
        the first two assertions and be useless."""
        mod = _adf_module()
        n = getattr(mod, "NUMBA_MIN_ROWS", 10000) + 10

        def matched(parent_keys, child_keys, use_numba):
            m = A.AliasDataFrame(pd.DataFrame({
                "k": parent_keys, "x": np.arange(len(parent_keys))}))
            m._use_numba = use_numba
            ch = A.AliasDataFrame(pd.DataFrame({"k": child_keys}))
            ch.df["v"] = np.arange(len(child_keys), dtype=np.int64)
            m.register_subframe("S", ch, index_columns=["k"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _idx, miss = m._compute_join_indices("S", ["k"])
            return int((~np.asarray(miss)).sum())

        collide_p = np.full(n, -1, dtype=np.int64)
        collide_c = np.full(n, 2**64 - 1, dtype=np.uint64)

        for use_numba in (False, True):
            assert matched(collide_p, collide_c, use_numba) == 0, (
                f"int64(-1) matched uint64(2**64-1) with use_numba="
                f"{use_numba} -- a silent wrong join")
            assert matched(collide_c, collide_p, use_numba) == 0, (
                f"reversed orientation false-matched with use_numba="
                f"{use_numba}")

        equal = np.arange(n, dtype=np.int64)
        for use_numba in (False, True):
            assert matched(equal, equal.copy(), use_numba) == n, (
                "the guard must gate the fast path, not disable joining")

    # ---- family 10: the all-undefined getter (AD-20) ---------------------

    @staticmethod
    def _all_undefined_frame():
        """Every join key absent: three rows, none matched."""
        m = A.AliasDataFrame(pd.DataFrame({
            "key": np.array([1, 2, 3], np.int64),
            "x": np.array([10.0, 20.0, 30.0])}))
        ch = A.AliasDataFrame(pd.DataFrame({"key": np.array([7, 8, 9], np.int64)}))
        ch.df["v"] = np.array([1.0, 2.0, 3.0])
        m.register_subframe("S", ch, index_columns=["key"])
        return m

    def test_b32b_18_getter_never_publishes_the_requested_alias(self):
        """AD-20 clause 1 — ALREADY TRUE, pinned so B3.2b cannot regress it
        while implementing clause 4. This is the clause that matters most:
        `materialize_alias` on the same all-undefined alias publishes a
        column AND mints float64 source-4 authority out of a fill policy
        nobody measured. The getter must never do that."""
        m = self._all_undefined_frame()
        m.set_subframe_fill("S", fill_missing=0.0)
        m.add_alias("scaled", "S.v * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = m.get_alias_series("scaled")
        assert len(out) == 3
        assert "scaled" not in m.df.columns, (
            "a non-materializing getter published the requested alias")

    def test_b32b_18b_getter_creates_no_source4_authority(self):
        """AD-20 clause 2 — ALREADY TRUE. The contrast is the point: the same
        alias through materialize_alias records
        DTypeOrigin.FIRST_MATERIALIZATION."""
        mod = _adf_module()
        m = self._all_undefined_frame()
        m.set_subframe_fill("S", fill_missing=0.0)
        m.add_alias("scaled", "S.v * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.get_alias_series("scaled")
        auth = m.get_dtype_authority("scaled")
        assert not auth.known, (
            f"the getter minted an authority from an ephemeral read: {auth!r}")

        publishing = self._all_undefined_frame()
        publishing.set_subframe_fill("S", fill_missing=0.0)
        publishing.add_alias("scaled", "S.v * 2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            publishing.materialize_alias("scaled")
        assert publishing.get_dtype_authority("scaled").origin == (
            mod.DTypeOrigin.FIRST_MATERIALIZATION), (
            "the contrast this test rests on has changed; re-derive AD-20")

    def test_b32b_18c_declared_authority_governs_the_ephemeral_dtype(self):
        """AD-20 clause 3 — ALREADY TRUE. An explicitly declared dtype
        (AD-19 source 3) governs what the getter hands back, and the
        declaration was made by add_alias, not by the getter."""
        mod = _adf_module()
        m = self._all_undefined_frame()
        m.set_subframe_fill("S", fill_missing=0.0)
        m.add_alias("scaled", "S.v * 2", dtype=np.int16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = m.get_alias_series("scaled")
        assert str(out.dtype) == "int16", (
            f"declared authority did not govern the getter: {out.dtype}")
        assert m.get_dtype_authority("scaled").origin == (
            mod.DTypeOrigin.EXPLICIT_ALIAS)
        assert "scaled" not in m.df.columns

    def test_b32b_18d_getter_honours_the_configured_alias_fill(self):
        """AD-20 clause 4: the ephemeral Series getter honours the same
        configured final alias fill as materialize_alias, without publishing.
        """
        m = self._all_undefined_frame()
        m.add_alias("scaled", "S.v * 2", fill_value=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = list(m.get_alias_series("scaled").values)

        publishing = self._all_undefined_frame()
        publishing.add_alias("scaled", "S.v * 2", fill_value=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            publishing.materialize_alias("scaled")
        expected = list(publishing.df["scaled"].values)

        assert got == expected, (
            f"getter returned {got}, materialize_alias returned {expected} "
            f"for the same alias and the same fill")
        assert "scaled" not in m.df.columns

    def test_b32b_18e_getter_refuses_unresolved_undefinedness(self):
        """AD-20 clause 4: unresolved structural absence is not silent NaN."""
        mod = _adf_module()
        m = self._all_undefined_frame()
        m.add_alias("scaled", "S.v * 2")
        with pytest.raises(mod.ADFError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.get_alias_series("scaled")

    def test_b32b_18f_array_getter_honours_the_configured_alias_fill(self):
        """AD-20 applies symmetrically to the public NumPy getter surface."""
        m = self._all_undefined_frame()
        m.add_alias("scaled", "S.v * 2", fill_value=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = list(m.get_alias_array("scaled"))
        assert got == [0.0, 0.0, 0.0]
        assert "scaled" not in m.df.columns

    def test_b32b_18g_array_getter_refuses_unresolved_undefinedness(self):
        """The public NumPy getter refuses the same unresolved absence."""
        mod = _adf_module()
        m = self._all_undefined_frame()
        m.add_alias("scaled", "S.v * 2")
        with pytest.raises(mod.ADFError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.get_alias_array("scaled")

    def test_b32b_18h_arithmetic_nan_is_not_misclassified_as_structural(self):
        """AR-7 negative control: an unconsulted absent operand does not turn
        an arithmetic NaN on the selected branch into structural absence.
        """
        m = self._all_undefined_frame()
        m.df["c"] = np.array([False, False, False])
        m.df["bad"] = np.array([-1.0, -1.0, -1.0])
        m.add_alias("q", "where(c, S.v, sqrt(bad))")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = m.get_alias_series("q")
        assert np.isnan(np.asarray(out, dtype=float)).all()
        assert "q" not in m.df.columns

    def test_b32b_18i_configured_fill_preserves_arithmetic_nan(self):
        """Decision 5 negative control: alias fill resolves structural
        absence only.  Pure arithmetic NaN remains NaN even when the alias has
        a configured fill, on both non-materializing getter surfaces.
        """
        m = A.AliasDataFrame(pd.DataFrame({
            "bad": np.array([-1.0, -1.0, -1.0], dtype=float)}))
        m.add_alias("q", "sqrt(bad)", fill_value=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            series = m.get_alias_series("q")
            array = m.get_alias_array("q")
        assert np.isnan(np.asarray(series, dtype=float)).all(), (
            f"configured fill changed arithmetic Series NaN: {series!r}")
        assert np.isnan(np.asarray(array, dtype=float)).all(), (
            f"configured fill changed arithmetic array NaN: {array!r}")
        assert "q" not in m.df.columns

    def test_b32b_18j_structural_fill_and_arithmetic_nan_stay_distinct(self):
        """Decision 5 mixed control in one getter call: structural absence
        is filled, while an independently produced arithmetic NaN remains NaN.
        Series and array surfaces must agree.
        """
        m = A.AliasDataFrame(pd.DataFrame({
            "key": np.array([1, 2], np.int64),
            "c": np.array([True, False]),
            "bad": np.array([-1.0, -1.0], dtype=float)}))
        child = A.AliasDataFrame(pd.DataFrame({
            "key": np.array([2], np.int64),
            "v": np.array([5.0], dtype=float)}))
        m.register_subframe("S", child, index_columns=["key"])
        m.add_alias("q", "where(c, S.v, sqrt(bad))", fill_value=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            series = np.asarray(m.get_alias_series("q"), dtype=float)
            array = np.asarray(m.get_alias_array("q"), dtype=float)
        for got, label in ((series, "Series"), (array, "array")):
            assert got[0] == 0.0, (
                f"{label} did not fill the structural row: {got!r}")
            assert np.isnan(got[1]), (
                f"{label} replaced arithmetic NaN with structural fill: {got!r}")
        assert "q" not in m.df.columns


    @staticmethod
    def _partially_undefined_getter_frame():
        """Two rows: key 1 absent from S, key 2 matched to S.v=5."""
        m = A.AliasDataFrame(pd.DataFrame({
            "key": np.array([1, 2], np.int64),
            "x": np.array([10.0, 20.0])}))
        ch = A.AliasDataFrame(pd.DataFrame({
            "key": np.array([2], np.int64),
            "v": np.array([5.0], dtype=float)}))
        m.register_subframe("S", ch, index_columns=["key"])
        return m

    def test_b32b_18k_failed_getter_retracts_structural_temp_before_fill_retry(self):
        """Decision-5 lifecycle: an expression exception after scattering S.v
        must not leave a representable-gap temporary that poisons the next
        getter.  The retry with configured fill must still see the absent key.
        """
        m = self._partially_undefined_getter_frame()

        def boom(v):
            raise RuntimeError("intentional getter failure after join")

        m.register_function("boom", boom)
        m.add_alias("q", "S.v + boom(x)", fill_value=0.0)
        with pytest.raises(RuntimeError, match="intentional getter failure"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.get_alias_series("q")
        assert "v__S" not in m.df.columns, (
            "failed getter leaked its structural join temporary")

        m.add_alias("r", "S.v * 2", fill_value=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            got = np.asarray(m.get_alias_series("r"), dtype=float)
        np.testing.assert_allclose(got, np.array([0.0, 10.0]))

    def test_b32b_18l_failed_getter_retracts_structural_temp_before_refusal_retry(self):
        """Same failed-getter lifecycle, but the retry has no handling and
        must therefore produce the ADF-owned Decision-5 refusal, not silent NaN.
        """
        mod = _adf_module()
        m = self._partially_undefined_getter_frame()

        def boom(v):
            raise RuntimeError("intentional getter failure after join")

        m.register_function("boom", boom)
        m.add_alias("q", "S.v + boom(x)", fill_value=0.0)
        with pytest.raises(RuntimeError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.get_alias_array("q")
        assert "v__S" not in m.df.columns

        m.add_alias("r", "S.v * 2")
        with pytest.raises(mod.ADFError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.get_alias_array("r")

    def test_b32b_18m_eval_preexisting_join_preserves_fill_getter_provenance(self):
        """A joined column left by public eval() remains caller-visible, but
        Decision-5 getter provenance must be reconstructed from the join rather
        than inferred from whether the flattened column is newly created.
        """
        m = self._partially_undefined_getter_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = np.asarray(m.eval("S.v"), dtype=float)
        assert np.isnan(raw[0]) and raw[1] == 5.0
        assert "v__S" in m.df.columns
        prior = m.df["v__S"].copy()

        m.add_alias("q", "S.v", fill_value=0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            series = np.asarray(m.get_alias_series("q"), dtype=float)
            array = np.asarray(m.get_alias_array("q"), dtype=float)
        np.testing.assert_allclose(series, np.array([0.0, 5.0]))
        np.testing.assert_allclose(array, np.array([0.0, 5.0]))
        assert "v__S" in m.df.columns, (
            "getter deleted a pre-existing joined column owned by eval()")
        pd.testing.assert_series_equal(m.df["v__S"], prior)

    def test_b32b_18n_eval_preexisting_join_preserves_no_fill_refusal(self):
        """Pre-existing joined representation must not turn unresolved
        structural absence into silent NaN on either getter surface.
        """
        mod = _adf_module()
        m = self._partially_undefined_getter_frame()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.eval("S.v")
        assert "v__S" in m.df.columns
        prior = m.df["v__S"].copy()

        m.add_alias("q", "S.v")
        for getter in (m.get_alias_series, m.get_alias_array):
            with pytest.raises(mod.ADFError):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    getter("q")
            assert "v__S" in m.df.columns
            pd.testing.assert_series_equal(m.df["v__S"], prior)

    def test_b32b_18o_nonrowlocal_structural_getter_with_fill_refuses(self):
        """Fix11c must consume AD-20's structural-mask channel too.

        A fill may resolve the structurally absent RESULT row only after ADF
        proves that the absent operand cannot contaminate defined rows.  A
        whole-column reduction over S.v is not row-local, so both getter
        surfaces must refuse before the alias fill can turn the missing row
        into an apparently valid result.
        """
        mod = _adf_module()
        for getter_name in ("get_alias_series", "get_alias_array"):
            m = self._partially_undefined_getter_frame()
            m.add_alias("q", "S.v - S.v.mean()", fill_value=0.0)
            with pytest.raises(mod.ADFProvenanceUnsupportedError,
                               match="row-local"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    getattr(m, getter_name)("q")

    def test_b32b_18p_nonrowlocal_structural_getter_without_fill_refuses_at_gate(self):
        """No-fill follows the same provenance gate, not the later resolver.

        The safety classification is independent of whether the user supplied
        a final-result fill.  In both cases the non-row-local expression must
        be rejected as ADFProvenanceUnsupportedError before Decision-5
        fill/refusal handling runs.
        """
        mod = _adf_module()
        for getter_name in ("get_alias_series", "get_alias_array"):
            m = self._partially_undefined_getter_frame()
            m.add_alias("q", "S.v - S.v.mean()")
            with pytest.raises(mod.ADFProvenanceUnsupportedError,
                               match="row-local"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    getattr(m, getter_name)("q")

    # ---- the remaining carried D_n items ---------------------------------

    def test_b32b_3_physical_column_is_an_authority_source(self):
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        auth = m.get_dtype_authority("x")
        assert auth.known, (
            f"a plain physical column is not an authority source: {auth!r}")
        assert auth.origin == _adf_module().DTypeOrigin.PHYSICAL_COLUMN

    @pytest.mark.xfail(strict=True, reason=
        "B3.2b acceptance (D_5 remainder): mask carriage reaches INNER levels "
        "of a multi-level subframe chain. Measured baseline failure: the "
        "inner gather raises the AD-19 'would change its authoritative dtype' "
        "refusal, because the inner scatter receives ctx=None by design.")
    def test_b32b_6_multilevel_chain_carries_the_mask(self):
        inner = A.AliasDataFrame(pd.DataFrame({"j": np.array([0], np.int64)}))
        inner.df["val"] = np.array([7], dtype=np.int64)
        mid = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 1], np.int64), "j": np.array([0, 9], np.int64)}))
        mid.register_subframe("I", inner, index_columns=["j"])
        main = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 1], np.int64), "x": np.array([10, 20], np.int64)}))
        main.register_subframe("M", mid, index_columns=["k"])
        main.add_alias("d", "M.I.val + x", dtype="int64", fill_value=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main.materialize_alias("d")
        assert [int(v) for v in main.df["d"].values] == [17, 1]

    def test_b32b_7_categorical_authority_is_recorded_exactly(self):
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        cats = pd.CategoricalDtype(["b", "a"], ordered=True)
        m.register_function("as_cat",
                            lambda v: pd.Series(["a", "b"]).astype(cats))
        m.add_alias("q", "as_cat(x)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        auth = m.get_dtype_authority("q")
        assert auth.known, (
            f"categorical authority cannot be recorded at all: {auth!r}")
        assert list(auth.dtype.categories) == ["b", "a"]
        assert auth.dtype.ordered is True

    def test_b32b_7b_codec_admission_is_derived_from_the_decoder(self):
        """STEP 4 v02 — GPT31 `F1` and GPT29 `F2`, one mechanism.

        v01's encoder judged its own exactness with a JSON-SAFETY check, and
        `json.dumps(x)` succeeding is a DIFFERENT proposition from
        `json.loads(json.dumps(x)) == x`. Both findings are that gap:

            CategoricalDtype([("a",1)])   dumps fine, decodes TypeError
            ArrowDtype(timestamp tz)      encodes fine, no `type_for_alias`

        Each wrote a record no reader could ever read — the exact outcome the
        "exact or no authority" rule exists to prevent. This test states the
        invariant itself rather than the two specimens: for EVERY dtype, an
        admitted encoding must survive the ACTUAL wire and decode back to the
        same dtype. A future lossy family is then refused by construction
        instead of waiting to be enumerated.
        """
        cls = A.AliasDataFrame
        specimens = [
            np.dtype("float64"), np.dtype("int64"), np.dtype("O"),
            pd.Int64Dtype(), pd.BooleanDtype(),
            pd.SparseDtype("int64", 0), pd.SparseDtype("float64", np.nan),
            pd.DatetimeTZDtype("ns", "Europe/Berlin"),
            pd.PeriodDtype("D"), pd.IntervalDtype("int64"),
            pd.CategoricalDtype(["a", "b"]),
            pd.CategoricalDtype(["b", "a"], ordered=True),
            pd.CategoricalDtype(pd.Index([1, 2], dtype="int64")),
            pd.CategoricalDtype([("a", 1), ("b", 2)]),      # F1
            pd.StringDtype("python"),
        ]
        try:
            import pyarrow as pa
            specimens += [
                pd.ArrowDtype(pa.string()), pd.ArrowDtype(pa.int64()),
                pd.ArrowDtype(pa.bool_()), pd.ArrowDtype(pa.float64()),
                pd.ArrowDtype(pa.timestamp("us", tz="UTC")),   # F2
                pd.ArrowDtype(pa.decimal128(5, 2)),            # F2
                pd.ArrowDtype(pa.list_(pa.int64())),           # F2
            ]
            specimens.append(pd.StringDtype("pyarrow"))
        except ImportError:
            pass

        admitted = 0
        for dt in specimens:
            enc = cls._encode_dtype(dt)
            if enc is None:
                continue                    # refused: no claim, nothing to check
            admitted += 1
            wire = json.loads(json.dumps(enc))     # the REAL wire, not the object
            back = cls._decode_dtype(wire)
            assert cls._dtype_exactly_equal(back, dt), (
                f"{dt!r} was admitted as an EXACT authority but the record "
                f"reads back as {back!r}")

        assert admitted >= 12, (
            "the self-check must not be satisfied by refusing everything; "
            f"only {admitted} specimens were admitted")

        # and the two executed findings specifically must now be REFUSED
        assert cls._encode_dtype(
            pd.CategoricalDtype([("a", 1), ("b", 2)])) is None, (
            "GPT31 F1: tuple categories become LISTS on the wire and the "
            "record then raises `unhashable type: 'list'` on decode")
        try:
            import pyarrow as pa
            assert cls._encode_dtype(
                pd.ArrowDtype(pa.timestamp("us", tz="UTC"))) is None, (
                "GPT29 F2: the encoder admitted a broader Arrow domain than "
                "`type_for_alias` can reconstruct")
        except ImportError:
            pass

    def test_b32b_7c_category_order_is_part_of_exactness(self):
        """The reviewers prescribed "semantic equality with the original
        dtype" as the admission invariant. Implemented with `==` it has a
        hole, and this test is why `_dtype_exactly_equal` exists:

            CategoricalDtype(["a","b"]) == CategoricalDtype(["b","a"]) -> True

        pandas compares UNORDERED categories as a set. The category ORDER is
        what the stored codes index, so a codec that permuted it would pass a
        `==`-based self-check while `b32b_7` requires categories AND order.
        """
        cls = A.AliasDataFrame
        a = pd.CategoricalDtype(["a", "b"])
        b = pd.CategoricalDtype(["b", "a"])
        assert a == b, (
            "pandas' own equality is set-like here; if this ever changes the "
            "stricter comparison below is redundant, not wrong")
        assert not cls._dtype_exactly_equal(a, b), (
            "admission equality must be stricter than the dtype's own")
        assert cls._dtype_exactly_equal(a, pd.CategoricalDtype(["a", "b"]))
        assert not cls._dtype_exactly_equal(
            a, pd.CategoricalDtype(["a", "b"], ordered=True))
        assert not cls._dtype_exactly_equal(
            pd.StringDtype("python"), pd.StringDtype("pyarrow"))

    def test_b32b_7e_every_recorder_admits_through_the_same_codec(self):
        """STEP 4 v02, free attack 5A — and it found a THIRD recorder.

        v01 changed the two materialization recorders and left the AD-19
        source-3 site (`add_alias(dtype=...)`, added by STEP 3 `13d`) on the
        OLD `str()`-based admission. A DECLARED categorical alias therefore
        stored no authority record at all, while the same dtype arriving by
        source 4 or 5 stored one exactly — and because the declaration still
        answers from memory, nothing looked wrong until the file was reread,
        which is precisely the persistence argument `13d` was written to make.

        The structural assertion is the point: no recorder may hold its own
        opinion about what is exactly representable.
        """
        cats = pd.CategoricalDtype(["b", "a"], ordered=True)
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
        m.add_alias("q", "x * 2", dtype=cats)
        rec = m._schema["columns"]["q"].get(
            A.AliasDataFrame._AUTHORITY_KEY)
        assert rec is not None, (
            "a DECLARED categorical dtype stored no authority record; the "
            "source-3 recorder is not using the STEP 4 codec")
        back = A.AliasDataFrame._canonical_dtype(
            json.loads(json.dumps(rec["dtype"])))
        assert A.AliasDataFrame._dtype_exactly_equal(back, cats), (
            f"the source-3 record does not read back exactly: {back!r}")

        # and the source-3 record must be reachable through the public getter
        auth = m.get_dtype_authority("q")
        assert auth.known and A.AliasDataFrame._dtype_exactly_equal(
            auth.dtype, cats), f"{auth!r}"

        # no recorder CALLS the superseded str()-based admission any more.
        # The definition survives only because `b32_222`/`b32_223` assert the
        # predicate directly, so the definition line is excluded and every
        # remaining CALL would be a recorder holding its own opinion.
        import re as _re
        src = open(__import__("AliasDataFrame").__file__).read()
        calls = _re.findall(
            r"(?<!def )_authority_is_exactly_representable\(", src)
        assert calls == [], (
            f"{len(calls)} recorder(s) still admit by the superseded "
            "str()-based rule")

    def test_b32b_7d_child_frame_categorical_authority_is_contestable(self):
        """The STEP 4 v01 CRR listed this as free attack `5B` and EXECUTED,
        but the diff carried no test to point at (Sonet28 `F4`). A probe that
        leaves no regression behind is not evidence a later change can be
        held to, so the probe is landed here as a named test.

        A categorical authority recorded on a CHILD frame, then contradicted
        by a physical recast, must refuse consumption through `S.c` — the
        authority must be a property of the column wherever the column lives,
        not of the top-level frame.
        """
        cats = pd.CategoricalDtype(["b", "a"], ordered=True)
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0, 1], np.int64)}))
        ch.register_function("as_cat",
                             lambda v: pd.Series(["a", "b"]).astype(cats))
        ch.add_alias("c", "as_cat(k)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ch.materialize_alias("c")
        auth = ch.get_dtype_authority("c")
        assert auth.known and list(auth.dtype.categories) == ["b", "a"], (
            f"the child frame must record the authority at all: {auth!r}")

        m = A.AliasDataFrame(pd.DataFrame({"k": np.array([0, 1], np.int64)}))
        m.register_subframe("S", ch, index_columns=["k"])
        m.add_alias("d", "S.c")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("d")        # CONTROL: uncontested, it works
        assert "d" in m.df.columns

        ch.df["c"] = ch.df["c"].astype(
            pd.CategoricalDtype(["a", "b"], ordered=True))   # order flipped
        assert ch.get_dtype_authority("c").conflict, (
            "a physical recast that changes the category ORDER must contest "
            "the recorded authority")
        m2 = A.AliasDataFrame(pd.DataFrame({"k": np.array([0, 1], np.int64)}))
        m2.register_subframe("S", ch, index_columns=["k"])
        m2.add_alias("d", "S.c")
        with pytest.raises(Exception) as exc:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m2.materialize_alias("d")
        assert "S.c" in str(exc.value), (
            f"the refusal must name the qualified child subject: {exc.value}")

    def test_b32b_8_publication_is_atomic_at_every_seam(self):
        """B32B-MR-P1-6: revision 1 injected one fault and checked two things.
        The publication transaction owns more than that, so both seams are
        driven and the whole snapshot is compared."""
        cls = A.AliasDataFrame
        seams = ("_commit_first_materialization_authority",
                 "_enforce_recorded_authority")
        for seam in seams:
            m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2], np.int64)}))
            m.add_alias("q", "x * 2")
            cols_before = list(m.df.columns)
            schema_before = copy.deepcopy(m._schema)
            orig = getattr(cls, seam)

            def boom(self, *a, **k):
                raise RuntimeError(f"injected fault at {seam}")

            setattr(cls, seam, boom)
            try:
                with pytest.raises(RuntimeError):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m.materialize_alias("q")
            finally:
                setattr(cls, seam, orig)
            assert list(m.df.columns) == cols_before, (
                f"seam {seam}: a failed publication left a column behind")
            assert m._schema == schema_before, (
                f"seam {seam}: a failed publication mutated the schema")


    def test_b32b_8a_bulk_publication_rolls_back_every_source4_alias(self):
        """STEP 6 source-4 bulk transaction.

        ``materialize_aliases`` publishes the aligned batch in one concat and
        then establishes source-4 authorities one by one.  Before STEP 6, a
        failure on authority #2 left BOTH physical columns plus authority #1.
        The public batch call must be all-or-none for the columns/authorities
        it owns in this publication.
        """
        cls = A.AliasDataFrame
        m = A.AliasDataFrame(pd.DataFrame({
            "x": np.array([1, 2], np.int64)}))
        m.add_alias("q1", "x * 2")
        m.add_alias("q2", "x * 3")
        cols_before = list(m.df.columns)
        schema_before = copy.deepcopy(m._schema)
        orig = cls._commit_first_materialization_authority
        calls = []

        def boom_second(self, name):
            calls.append(name)
            if name == "q2":
                raise RuntimeError("injected bulk authority fault on q2")
            return orig(self, name)

        cls._commit_first_materialization_authority = boom_second
        try:
            with pytest.raises(RuntimeError, match="q2"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.materialize_aliases(names=["q1", "q2"],
                                          with_dependencies=False,
                                          cleanTemporary=False)
        finally:
            cls._commit_first_materialization_authority = orig

        assert calls == ["q1", "q2"], (
            "fault injection must reach authority #2 after #1 committed")
        assert list(m.df.columns) == cols_before, (
            "failed bulk publication left one or more physical aliases")
        assert m._schema == schema_before, (
            "failed bulk publication left partial source-4 authority state")

    @staticmethod
    def _step6_compression_spec(*cols):
        """Small deterministic source-5 codec fixture for atomicity faults."""
        return {
            col: {
                "compress": f"round({col}*10)",
                "decompress": f"{col}_c/10.",
                "compressed_dtype": np.int16,
                "decompressed_dtype": np.float32,
            }
            for col in cols
        }

    def test_b32b_8b_compress_columns_rolls_back_source5_batch(self):
        """STEP 6 source-5 creation is all-or-none across a multi-column call.

        Before STEP 6, a source-5 authority failure on the second compressed
        column left the first transition committed and the second physical
        column/schema half-finished.  Compression owns one persistent state
        transition per public call, so every physical/schema/authority change
        from that call must roll back together.
        """
        cls = A.AliasDataFrame
        m = A.AliasDataFrame(pd.DataFrame({
            "dy": np.array([1.5, 2.5, 3.5]),
            "dz": np.array([4.5, 5.5, 6.5]),
        }))
        spec = self._step6_compression_spec("dy", "dz")
        df_before = m.df.copy(deep=True)
        schema_before = copy.deepcopy(m._schema)
        orig = cls._record_adf_created_authority
        calls = []

        def boom_second(self, name, dtype, reason):
            calls.append(name)
            if name == "dz_c":
                raise RuntimeError("injected source-5 compression fault on dz_c")
            return orig(self, name, dtype, reason)

        cls._record_adf_created_authority = boom_second
        try:
            with pytest.raises(RuntimeError, match="dz_c"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.compress_columns(spec)
        finally:
            cls._record_adf_created_authority = orig

        assert calls == ["dy_c", "dz_c"], (
            "fault must reach source-5 authority #2 after #1 succeeded")
        pd.testing.assert_frame_equal(m.df, df_before)
        assert m._schema == schema_before, (
            "failed source-5 compression left schema/authority state behind")

    def test_b32b_8c_decompress_creation_rolls_back_source5_state(self):
        """A failed source-5 record for the restored column restores COMPRESSED.

        Decompression materializes/casts the original column and removes its
        alias before recording the new source-5 authority.  A fault at that
        record used to leave a physical restored column with metadata still
        describing the compressed state.
        """
        cls = A.AliasDataFrame
        m = A.AliasDataFrame(pd.DataFrame({
            "dy": np.array([1.5, 2.5, 3.5])}))
        spec = self._step6_compression_spec("dy")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.compress_columns(spec)
        df_before = m.df.copy(deep=True)
        schema_before = copy.deepcopy(m._schema)
        orig = cls._record_adf_created_authority

        def boom_restore(self, name, dtype, reason):
            if name == "dy":
                raise RuntimeError("injected source-5 decompression fault")
            return orig(self, name, dtype, reason)

        cls._record_adf_created_authority = boom_restore
        try:
            with pytest.raises(RuntimeError, match="decompression fault"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.decompress_columns(["dy"])
        finally:
            cls._record_adf_created_authority = orig

        pd.testing.assert_frame_equal(m.df, df_before)
        assert m._schema == schema_before, (
            "failed decompression creation did not restore compressed metadata")

    def test_b32b_8d_decompress_destruction_rolls_back_column_and_authority(self):
        """Destroying compressed storage is atomic with clearing its authority.

        Before STEP 6, ``keep_compressed=False`` dropped ``dy_c`` and then
        cleared its source-5 record.  A fault in the clear left an authority
        that described a column that no longer existed.  The whole public
        decompression transition must restore its pre-call COMPRESSED state.
        """
        cls = A.AliasDataFrame
        m = A.AliasDataFrame(pd.DataFrame({
            "dy": np.array([1.5, 2.5, 3.5])}))
        spec = self._step6_compression_spec("dy")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.compress_columns(spec)
        df_before = m.df.copy(deep=True)
        schema_before = copy.deepcopy(m._schema)
        orig = cls._clear_adf_created_authority

        def boom_clear(self, name):
            if name == "dy_c":
                raise RuntimeError("injected source-5 destruction fault")
            return orig(self, name)

        cls._clear_adf_created_authority = boom_clear
        try:
            with pytest.raises(RuntimeError, match="destruction fault"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.decompress_columns(["dy"], keep_compressed=False)
        finally:
            cls._clear_adf_created_authority = orig

        pd.testing.assert_frame_equal(m.df, df_before)
        assert m._schema == schema_before, (
            "failed source-5 destruction left column/authority state split")

    def test_b32b_9_strict_route_is_a_registered_helper_not_a_framework_flag(
            self):
        """AR-2 / D_10 — ALREADY SATISFIED, converted from a strict xfail.

        MR-P1-4 (GPT29): revision 2 marked this xfail with a reason claiming
        'no strict route exists'. Run with --runxfail it fails on the OTHER
        assertion — the strict helper works exactly as AR-2 specifies, and
        what is still open is the ORDINARY route's conversion semantics,
        which belongs to D_3 and is now owned by `b32b_9b`. Leaving the two
        joined would have let family 10's real defect close under family 9's
        name.

        P0-4 correction, retained: revision 1 required
        `register_function(..., strict=True)`, inventing exactly the framework
        API that AR-2 was designed to avoid."""
        def strict_int8(v):
            a = np.asarray(v)
            if (a < -128).any() or (a > 127).any():
                raise ValueError("strict_int8: value out of range")
            return a.astype(np.int8)

        strict = A.AliasDataFrame(pd.DataFrame({"x": np.array([300, 2], np.int64)}))
        strict.register_function("strict_int8", strict_int8)
        strict.add_alias("q", "strict_int8(x)")
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                strict.materialize_alias("q")

    @pytest.mark.xfail(strict=True, reason=
        "B3.2b acceptance (D_3, §6.1 — split out of b32b_9 per MR-P1-4): the "
        "ORDINARY conversion route follows documented backend semantics and "
        "delivers the declared dtype, so the strict helper is a genuine "
        "opt-in rather than the only route that works. Measured baseline "
        "failure: materialize_alias raises before the dtype can be checked — "
        "the ordinary route refuses the out-of-range value instead of "
        "applying AR-1 standards-first conversion.")
    def test_b32b_9b_ordinary_route_follows_backend_semantics(self):
        ordinary = A.AliasDataFrame(
            pd.DataFrame({"x": np.array([300, 2], np.int64)}))
        ordinary.add_alias("q", "x", dtype="int8")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ordinary.materialize_alias("q")
        assert str(ordinary.df["q"].dtype) == "int8", (
            "the ORDINARY route must still follow documented backend "
            "semantics — the strict helper is opt-in, not a global switch")

    def test_b32b_10_casting_mode_is_pinned_and_consumed(self):
        """CLOSED BY B3.2b STEP 2 — was a strict xfail, now passing.

        `ADF_CASTING_MODE = "unsafe"` is declared at module level and consumed
        in `_safe_dtype_cast`, the single applier of a resolved target:

            _out = _arr.astype(target, casting=ADF_CASTING_MODE)

        AR-1 is standards-first — an ordinary conversion of COMPUTED data
        follows documented backend semantics rather than inventing strictness
        — and pinning the mode means a future NumPy default cannot move the
        ratified contract without the change appearing in a diff.

        B32B-MR-P1-2: revision 1 asserted only that a constant EXISTED; an
        unused constant would have flipped it. Declaration AND use.

        KNOWN LIMIT, still disclosed (GPT27, non-blocking): consumption is
        measured textually, so a constant used in a dead branch would satisfy
        it. The site it is consumed at is the exact-cast path of
        `_safe_dtype_cast`, which `b32b_9b` and the round-11f corruption
        tests both exercise, so it is not dead — but the test does not prove
        that, and saying so is cheaper than implying otherwise."""
        import re as _re
        mod = _adf_module()
        named = [n for n in dir(mod)
                 if "CASTING" in n.upper() and isinstance(getattr(mod, n), str)]
        assert named, "no named casting-mode constant exists"
        pinned = [n for n in named if getattr(mod, n) == "unsafe"]
        assert pinned, "AR-1 ratifies casting='unsafe'; pin it"
        src = _adf_source_text()
        used = any(len(_re.findall(r"casting=%s\b" % n, src)) > 0
                   for n in pinned)
        assert used, "the pinned constant is declared but never consumed"

    def test_b32b_11_structural_absence_raises_an_adf_type(self):
        """MR-P2-2 (Sonet28, Sonet31): revision 2 put `add_alias` and
        `materialize_alias` inside ONE `pytest.raises` block, so if
        `add_alias` ever starts raising first the test would silently narrow
        to a different production path and still pass. The two calls are now
        separated, and the row-level half is its own test so the two shapes
        cannot be conflated."""
        mod = _adf_module()
        root = getattr(mod, "ADFError", None)
        assert root is not None, "D_6 defines an ADFError root; it does not exist"

        structural = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2])}))
        structural.add_alias("d", "Nope.v")          # must NOT raise here
        with pytest.raises(root):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structural.materialize_alias("d")

    def test_b32b_11b_row_level_missingness_raises_a_distinct_adf_type(self):
        mod = _adf_module()
        root = getattr(mod, "ADFError", None)
        assert root is not None, "D_6 defines an ADFError root; it does not exist"

        structural = A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2])}))
        structural.add_alias("d", "Nope.v")
        with pytest.raises(root) as st:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                structural.materialize_alias("d")

        rowlevel = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 9], np.int64), "x": np.array([10, 20], np.int64)}))
        ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
        ch.df["v"] = np.array([3], dtype=np.int64)
        rowlevel.register_subframe("S", ch, index_columns=["k"])
        rowlevel.add_alias("d", "S.v", dtype="int64")
        with pytest.raises(root) as rl:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rowlevel.materialize_alias("d")

        assert type(st.value) is not type(rl.value), (
            "structural absence and row-level missingness must be "
            "distinguishable by type, not only by message")

    def test_b32b_11c_the_taxonomy_holds_on_every_alias_surface(self):
        """STEP 5 — the reason `b32b_11` alone does not close `D_6` §7.

        `b32b_11` drives ONE structural path (a missing subframe) through ONE
        public surface, and its own docstring warns that a single path
        silently narrows a criterion. Asked of the other surfaces, the
        narrowing was real: `S.nosuchcol` — a subframe that EXISTS without the
        referenced column, the commoner shape in practice — raised a bare
        `KeyError`, indistinguishable from a pandas lookup failure.

        So the criterion is stated as the taxonomy over the surfaces rather
        than as two specimens: every alias-evaluation absence is ADF-owned,
        and structural and row-level are never the same type.
        """
        mod = _adf_module()
        root = mod.ADFError
        structural = mod.StructuralAbsenceError
        rowlevel = mod.RowLevelMissingnessError

        def plain():
            return A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2])}))

        def joined():
            m = A.AliasDataFrame(pd.DataFrame({
                "k": np.array([0, 9], np.int64),
                "x": np.array([10, 20], np.int64)}))
            ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
            ch.df["v"] = np.array([3], dtype=np.int64)
            m.register_subframe("S", ch, index_columns=["k"])
            return m

        def run(build, expr, surface, dtype=None):
            m = build()
            if surface == "eval":
                op = lambda: m.eval(expr)
            else:
                m.add_alias("d", expr, dtype=dtype)
                _bound = getattr(m, surface)
                op = (lambda: _bound("d"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    op()
                except Exception as exc:      # noqa: BLE001 - the subject
                    return exc
            return None

        SURFACES = ("materialize_alias", "get_alias_array", "get_alias_series")

        # --- structural: absent by SCHEMA, no fill can repair it ----------
        cases = [(plain, "Nope.v", "missing subframe"),
                 (plain, "nope * 2", "missing column"),
                 (plain, "nosuchfunc(x)", "missing function"),
                 (joined, "S.nosuchcol", "subframe without the column")]
        for build, expr, label in cases:
            for surface in SURFACES:
                exc = run(build, expr, surface)
                assert exc is not None, f"{label} via {surface}: nothing raised"
                assert isinstance(exc, structural), (
                    f"{label} via {surface}: {type(exc).__name__} is not "
                    f"ADF-owned structural absence")
            exc = run(build, expr, "eval")
            assert isinstance(exc, structural), (
                f"{label} via eval(): {type(exc).__name__}")

        # the KeyError promise the production comment makes explicitly
        exc = run(joined, "S.nosuchcol", "materialize_alias")
        assert isinstance(exc, KeyError), (
            "the raise site promises KeyError for Sub.nonexistent references; "
            f"ADF ownership must be added to that, not replace it: {exc!r}")
        assert "does not contain" in str(exc), (
            f"KeyError.__str__ would repr() the message and break match=: "
            f"{str(exc)!r}")

        # --- row-level: EXISTS, some rows unmatched, a fill WOULD repair --
        for surface in SURFACES + ("eval",):
            expr = "d" if surface == "eval" else "S.v"
            m = joined()
            m.add_alias("d", "S.v", dtype="int64")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(rowlevel) as caught:
                    if surface == "eval":
                        m.eval("d")
                    else:
                        getattr(m, surface)("d")
            assert not isinstance(caught.value, structural), (
                f"{surface}: row-level missingness must NOT be structural — "
                "the caller may retry the first with a fill and must never "
                "retry the second")
            assert isinstance(caught.value, root)

        # --- the MULTI-LEVEL chain raises from its own site --------------
        # Found the same way as `S.nosuchcol`: the single-level path was
        # re-typed and the projection path two levels down was still raising a
        # bare ValueError. Same condition, different raise site — which is the
        # third time in this phase that "one path" turned out not to be the
        # criterion.
        inner = A.AliasDataFrame(pd.DataFrame({"j": np.array([0], np.int64)}))
        inner.df["val"] = np.array([7], dtype=np.int64)
        mid = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 1], np.int64),
            "j": np.array([0, 9], np.int64)}))
        mid.register_subframe("I", inner, index_columns=["j"])
        deep = A.AliasDataFrame(pd.DataFrame({
            "k": np.array([0, 1], np.int64),
            "x": np.array([10, 20], np.int64)}))
        deep.register_subframe("M", mid, index_columns=["k"])
        deep.add_alias("d", "M.I.val + x", dtype="int64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(rowlevel) as deep_exc:
                deep.materialize_alias("d")
        assert isinstance(deep_exc.value, ValueError), (
            "the projection sites raised ValueError before D_6; that must "
            "survive, or callers catching ValueError silently stop catching")

        # --- and the repair actually works, or the taxonomy means nothing -
        m = joined()
        m.add_alias("d", "S.v", dtype="int64", fill_value=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("d")
        assert [int(v) for v in m.df["d"].values] == [3, 0], (
            "a configured fill must resolve the row-level case; if it does "
            "not, calling it 'repairable' is a claim the code does not honour")

    def test_b32b_11d_step4_sparse_refusal_is_under_the_adf_root(self):
        """The debt STEP 4 opened and this step discharges.

        STEP 4's sparse reconstruction refuses rather than densifying, and it
        raised the bare `AliasDataFrameError` base ON PURPOSE, because `D_6`
        owned the hierarchy and inventing a type ahead of its owner is how an
        increment grows a defect. `D_6` has now placed a root above that base,
        so the refusal is ADF-rooted without a single line of STEP 4 changing.

        It is deliberately NOT structural and NOT row-level: nothing is
        absent. It is a representation refusal, and `D_6` §7 names exactly two
        absence kinds — so it stays under the root and outside the taxonomy.
        """
        mod = _adf_module()
        m = A.AliasDataFrame(pd.DataFrame({"x": np.array([1.0, 2.0, 3.0])}))
        src = pd.Series(pd.arrays.SparseArray(
            np.array([1.0, 0.0, 3.0]), fill_value=0.0))

        import sys as _sys
        real = _sys.modules["pandas._libs.sparse"]

        class _Declines:
            def __getattr__(self, _n):
                raise RuntimeError("injected: sparse reconstruction declined")

        _sys.modules["pandas._libs.sparse"] = _Declines()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(mod.ADFError) as exc:
                    m._place_fill(src, np.array([False, True, False]), 9.0,
                                  "fill_missing", "S", "v")
        finally:
            _sys.modules["pandas._libs.sparse"] = real

        assert not isinstance(exc.value, mod.StructuralAbsenceError)
        assert not isinstance(exc.value, mod.RowLevelMissingnessError)

    def test_b32b_11e_each_leaf_carries_only_its_own_builtin(self):
        """STEP 5a v02 — GPT29 `F2`, and the correction is entirely NEGATIVE.

        v01 made `StructuralAbsenceError` a `NameError` and hung the KeyError
        leaf beneath it, so `S.nosuchcol` became catchable by
        `except NameError` where it never had been. I disclosed exactly this
        as uncertainty #1 and did not execute it — the fourth time this phase
        that the defect was in a sentence I wrote rather than in the diff.

        Compatibility is not only about keeping what was caught. WIDENING what
        a handler catches is a change too: a caller with a broad
        `except NameError` around expression evaluation would silently begin
        swallowing subframe-column errors it used to let through, which is the
        harder failure to notice because nothing raises.

        So the semantic parent carries no builtin, each leaf carries exactly
        the one its own site raised, and this test asserts the ABSENCES.
        Positive assertions cannot catch a widening; only negative ones can.
        """
        mod = _adf_module()

        # the shape, stated on the classes so it cannot drift silently
        assert not issubclass(mod.StructuralAbsenceError, NameError), (
            "the SEMANTIC category must carry no builtin; the leaves do")
        assert not issubclass(mod.StructuralAbsenceError, KeyError)
        assert issubclass(mod.ExpressionNameAbsenceError, NameError)
        assert not issubclass(mod.ExpressionNameAbsenceError, KeyError)
        assert issubclass(mod.SubframeColumnAbsenceError, KeyError)
        assert not issubclass(mod.SubframeColumnAbsenceError, NameError), (
            "GPT29 F2: a KeyError path must not become catchable by "
            "`except NameError`")

        def joined():
            m = A.AliasDataFrame(pd.DataFrame({
                "k": np.array([0, 9], np.int64),
                "x": np.array([10, 20], np.int64)}))
            ch = A.AliasDataFrame(pd.DataFrame({"k": np.array([0], np.int64)}))
            ch.df["v"] = np.array([3], dtype=np.int64)
            m.register_subframe("S", ch, index_columns=["k"])
            return m

        def raised(build, expr):
            m = build()
            m.add_alias("d", expr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    m.materialize_alias("d")
                except Exception as exc:      # noqa: BLE001 - the subject
                    return exc
            raise AssertionError(f"{expr!r} did not raise")

        # and on the real production paths, both directions
        col = raised(joined, "S.nosuchcol")
        assert isinstance(col, mod.StructuralAbsenceError)
        assert isinstance(col, KeyError)
        assert not isinstance(col, NameError), (
            f"`except NameError` now catches the subframe-column path: {col!r}")

        name = raised(
            lambda: A.AliasDataFrame(pd.DataFrame({"x": np.array([1, 2])})),
            "Nope.v")
        assert isinstance(name, mod.StructuralAbsenceError)
        assert isinstance(name, NameError)
        assert not isinstance(name, KeyError), (
            f"`except KeyError` now catches the undefined-name path: {name!r}")

    def test_b32b_11f_the_root_covers_every_adf_owned_refusal(self):
        """STEP 5a — GPT27 `F1`, widened in v03 by `P1-1`.

        v02 asserted that every ADF-defined exception is under `ADFError` and
        swept TWO modules. Four seats found the counterexample in a third:
        `LazyChainReader.ChainShapeMismatchError(ValueError)`, public and
        ADF-owned, outside the root the CRR called "the single root of every
        ADF-owned error". I had disclosed the two-module limit as uncertainty
        #2 and had not executed the wider sweep — the same failure mode as
        `F2`, one revision later.

        The sweep is now over the PACKAGE. Every top-level module is parsed,
        every class whose bases name an exception is collected, and each is
        checked against the root. A module that declares such a class and
        cannot be imported FAILS rather than being skipped, so the search
        universe can never shrink quietly — which is exactly how v02's
        two-module census came to look complete.
        """
        mod = _adf_module()

        # --- the specific v01 finding stays pinned -------------------------
        assert issubclass(mod.ADFProvenanceUnsupportedError, mod.ADFError), (
            "GPT27 F1: a public ADF refusal outside the root")
        assert issubclass(mod.ADFProvenanceUnsupportedError, ValueError), (
            "…and its ValueError compatibility must survive the reparenting")
        assert not issubclass(
            mod.ADFProvenanceUnsupportedError, mod.StructuralAbsenceError)
        assert not issubclass(
            mod.ADFProvenanceUnsupportedError, mod.RowLevelMissingnessError)

        # --- and the v03 finding, which no two-module census could see -----
        import LazyChainReader as _lcr
        assert issubclass(_lcr.ChainShapeMismatchError, mod.ADFError), (
            "P1-1: a public ADF exception in a reader module, outside the "
            "root the CRR claims is complete")
        assert issubclass(_lcr.ChainShapeMismatchError, ValueError), (
            "…and its ValueError compatibility must survive")
        assert not issubclass(
            _lcr.ChainShapeMismatchError, mod.AliasDataFrameError), (
            "rooting it must not ALSO widen `except ChainValidationError` / "
            "`except AliasDataFrameError` — the F2 lesson applied to F1")

        # --- the census, over the whole package ----------------------------
        import ast as _ast, glob as _glob, importlib as _il
        import os as _os

        pkg = _os.path.dirname(_os.path.abspath(mod.__file__))
        BUILTIN_EXC = {
            "Exception", "BaseException", "ValueError", "KeyError",
            "NameError", "TypeError", "RuntimeError", "LookupError",
            "ArithmeticError", "OSError", "IOError", "AttributeError",
            "IndexError", "NotImplementedError"}

        orphans, unimportable, swept = [], [], 0
        for path in sorted(_glob.glob(_os.path.join(pkg, "*.py"))):
            modname = _os.path.splitext(_os.path.basename(path))[0]
            if modname == "__init__":
                continue
            try:
                with open(path) as fh:
                    tree = _ast.parse(fh.read())
            except (SyntaxError, UnicodeDecodeError):
                continue
            declared = []
            for node in _ast.walk(tree):
                if not isinstance(node, _ast.ClassDef):
                    continue
                bases = [b.id if isinstance(b, _ast.Name)
                         else getattr(b, "attr", "") for b in node.bases]
                if any(b in BUILTIN_EXC or b.endswith("Error") for b in bases):
                    declared.append(node.name)
            if not declared:
                continue
            try:
                module = _il.import_module(modname)
            except Exception as exc:      # noqa: BLE001 - reported, not hidden
                unimportable.append(f"{modname} ({type(exc).__name__}: {exc})")
                continue
            for nm in declared:
                obj = getattr(module, nm, None)
                if not (isinstance(obj, type)
                        and issubclass(obj, BaseException)):
                    continue
                swept += 1
                if not issubclass(obj, mod.ADFError):
                    orphans.append(f"{modname}.{nm}")

        assert not unimportable, (
            "a module declaring an exception could not be imported, so the "
            "root census is INCOMPLETE and this test cannot support the "
            f"claim it is used to support: {unimportable}")
        assert not orphans, (
            "every ADF-defined exception must be under the ADFError root; "
            f"outside it: {sorted(set(orphans))}")
        assert swept >= 13, (
            "the census must actually find the hierarchy; a sweep that "
            f"matches nothing passes vacuously. Found only {swept}")

    def test_b32b_11g_row_level_carries_only_value_error(self):
        """`P2-1` (GPT31, GPT33). Two lines, and they close a hole I listed as
        uncertainty #3 and left open: nothing asserted that
        `RowLevelMissingnessError` had not acquired a builtin it should not
        have. Every other leaf has its negative pin; this one did not, purely
        because no reviewer had asked yet."""
        mod = _adf_module()
        assert issubclass(mod.RowLevelMissingnessError, ValueError)
        assert not issubclass(mod.RowLevelMissingnessError, NameError)
        assert not issubclass(mod.RowLevelMissingnessError, KeyError)
        assert not issubclass(
            mod.RowLevelMissingnessError, mod.StructuralAbsenceError), (
            "row-level missingness is REPAIRABLE and structural absence is "
            "not; a caller must never retry the second")


def _adf_scaffold_text():
    """This test module's own source — for the family-owner guard."""
    import inspect, sys
    return inspect.getsource(sys.modules[__name__])
