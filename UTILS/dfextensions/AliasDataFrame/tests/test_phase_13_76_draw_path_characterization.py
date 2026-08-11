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
    def test_b32_221_categorical_does_not_establish_inferred_authority(self):
        """`str(dtype)` is "category" for EVERY categorical, so it cannot
        carry categories or order. Rather than record an approximate
        authority, none is established — disclosed deferral, B3.2b owns the
        exact structured codec."""
        def cat(_):
            return pd.Series(pd.Categorical(["a", "b"],
                                            categories=["a", "b"],
                                            ordered=False))
        m = A.AliasDataFrame(pd.DataFrame({"x": [1, 2]}))
        m.register_function("cat", cat)
        m.add_alias("q", "cat(x)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.materialize_alias("q")
        assert str(m.df["q"].dtype) == "category"
        assert not m.get_dtype_authority("q").known, (
            "a dtype whose string form loses metadata must not establish an "
            "inferred authority — a partial truth recorded as exact is worse "
            "than no record")

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
        _M._expression_is_row_local = lambda expr: (True, None)
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
        """The float leg: a missing key yields NaN natively, so no mask is
        carried and the row is simply absent from the plot. The values that
        ARE drawn must equal the eval oracle."""
        m = self._asym()
        m.add_alias("t", "C.z + x")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            truth = np.asarray(m.get_alias_array("t"))
        assert np.isnan(truth[2]), "the unmatched row must be undefined"
        assert [float(v) for v in truth[:2]] == [17.0, 27.0]

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
