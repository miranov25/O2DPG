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
