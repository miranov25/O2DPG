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
        self._orig["_execute_draw_plan"] = cls._execute_draw_plan
        real_exec = cls._execute_draw_plan

        def exec_(inner_self, plan, verbose=False):
            self._depth += 1
            try:
                return real_exec(inner_self, plan, verbose=verbose)
            finally:
                self._depth -= 1

        cls._execute_draw_plan = exec_
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

    @pytest.mark.xfail(
        strict=True,
        reason="B3.2 remainder (fail-before evidence): alias materialization "
               "is still performed by draw_batch after the executor returns. "
               "XPASS when the materialization migration lands, which forces "
               "removal of this marker.")
    def test_b32_5c_no_preparation_effect_of_any_kind_after_executor(self):
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
class TestB32DisclosedScopeLimits:
    """The two round-4 omissions, pinned as DISCLOSED LIMITS rather than fixed.

    Both are coverage gaps, not falsehoods: nothing untrue is recorded, the
    scope is simply narrower than a casual reading of "the auditable answer to
    which effects ran" would suggest. Both also turn on questions that are the
    architect's to answer, and deciding them inside a correction pass is the
    mistake this phase already paid for once with the eager path.

    These tests exist so the limits are visible and so the day someone changes
    the answer, a test says so out loud instead of a docstring quietly going
    stale.
    """

    def _lazy(self, name="lazy_struct_fixture_clean.root"):
        fixture = os.path.join(os.path.dirname(__file__), name)
        if not os.path.exists(fixture):
            pytest.skip("struct fixture not present (make_fixtures.py)")
        return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

    def test_b32_22_struct_inside_a_subframe_is_left_partial_and_says_so(self):
        """Hypothesis raised independently by three seats (Sonet25, Sonet27,
        Fabble5_7), executed by the coder. Completion is self-scoped while
        observation is now graph-scoped, so a struct living inside a subframe
        is not completed. The record does NOT claim otherwise — which is why
        this is a disclosed limit and not a defect. Whether the executor should
        reach into subframe registries is an open scope ruling."""
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
        assert after == before, (
            "the executor now completes structs inside subframes — that is a "
            "scope change requiring an architect ruling; update this test and "
            "the documented scope together")
        st = main._last_draw_prep_state
        assert "dedxTPC" not in st.structs_completed, (
            "nothing was completed, so nothing may be reported as completed")

    def test_b32_23_aliased_subframe_object_is_omitted_not_misreported(self):
        """GPT26's finding, pinned as a limit. The same child object under two
        names is walked once, so the second owner path is OMITTED. Nothing
        false is recorded — the first path's effects are correct. Whether the
        registration should be permitted at all is an open architect ruling."""
        main = self._lazy()
        child = self._lazy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            main.ensure_branches(["mult"])
            child.ensure_branches(["mult"])
            main.register_subframe("A", child, index_columns=["mult"])
            main.register_subframe("B", child, index_columns=["mult"])
            main.draw_batch({"p": {"expr": "mult", "type": "hist", "bins": 5}},
                            verbose=False)
        plt.close("all")
        st = main._last_draw_prep_state
        paths = {c.split("::")[0] for c in st.columns_created if "::" in c}
        assert not ({"A", "B"} <= paths), (
            "both owner paths are now recorded — the aliasing limit has been "
            "closed; remove this test and the disclosure together")
        assert not any(str(c).startswith("B::") and str(c) not in
                       st.columns_created for c in st.columns_created)
