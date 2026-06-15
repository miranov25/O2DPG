"""Phase 13.56.ADF — post-audit fixes tests.

Per PHASE_13_56_ADF_PostAuditFixes_Proposal_v1_2.md §2 (14 tests).
Guard/shim/error tests must fail pre-fix (FM#12); T-G1b/T-G2b use the
panel P1-1 three-assertion form VERBATIM (proposal C-6 clause); T-G2 uses
the delta-fignums form (P2-2); T-R1 is a regression LOCK (panel M-1 —
the 3-level facet return contract already works at HEAD).

Architect ratifications (proposal §0, 2026-06-11): D1=A, D2=B, D3=both
guards temporary, D4=A, D5=gallery+3, R-1 rescoped to lock, Help=full.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.AliasDataFrame import AliasDataFrame


def _mk(n=400, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "y": rng.normal(0, 1, n),
        "z": rng.normal(0, 1, n),
        "w": rng.uniform(0.5, 3.0, n),
        "side": rng.integers(0, 2, n).astype(np.int8),
        "flag": rng.integers(0, 3, n).astype(np.int8),
    })
    return AliasDataFrame(df)


def _figspec(plots, name="f"):
    return [{"name": name, "plots": plots}]


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


# =============================================================================
# T-G1 — profile2d-in-figures guard (E-3; D2=B semantics)
# =============================================================================

@pytest.mark.invariance
class TestG1Profile2dGuard:

    def test_TG1a_profile2d_spec_default_raises(self):
        """T-G1a: default on_error → clean ValueError naming the limitation,
        the bug report, and the remedy. Pre-fix: silent empty panel."""
        adf = _mk()
        with pytest.raises(ValueError, match="profile2d.*draw_figures"):
            adf.draw_figures(
                _figspec([{"expr": "z:y:x", "type": "profile2d", "bins": 6}]),
                verbose=False)

    def test_TG1b_profile2d_spec_skip_placeholder(self):
        """T-G1b: explicit skip — three-assertion form (panel P1-1, verbatim).
        Pre-fix all three fail: title normal, stats non-None, no message."""
        adf = _mk()
        res = adf.draw_figures(
            _figspec([{"expr": "z:y:x", "type": "profile2d", "bins": 6}]),
            on_error="skip", verbose=False)
        ax = res["f"]["axes"][0]
        stats = res["f"]["stats"]
        assert ax.get_title().startswith('[ERROR]')
        assert stats[0] is None
        assert any('profile2d' in t.get_text() for t in ax.texts)

    def test_TG1c_three_var_profile_promotion_reaches_guard(self):
        """T-G1c: 3-var type='profile' promotes to profile2d and then hits
        the guard (binding order §3.4) — closes the A4-caveat path."""
        adf = _mk()
        with pytest.raises(ValueError, match="profile2d.*draw_figures"):
            adf.draw_figures(
                _figspec([{"expr": "z:y:x", "type": "profile", "bins": 6}]),
                verbose=False)


# =============================================================================
# T-G2 — facet_by-in-figures guard (E-4; delta-fignums per P2-2)
# =============================================================================

@pytest.mark.invariance
class TestG2FacetByGuard:
    """adf.draw facet-path non-regression is locked by the 11 tests in
    test_bug_lazy_nd_facet_20260609.py (proposal C-5) — not duplicated here."""

    def test_TG2a_facet_by_spec_default_raises_no_leak(self):
        """T-G2a: default → clean ValueError with remedy; no leaked figure
        (delta-fignums: figure set unchanged across the call)."""
        adf = _mk()
        initial = set(plt.get_fignums())
        with pytest.raises(ValueError, match="facet_by.*draw_figures"):
            adf.draw_figures(
                _figspec([{"expr": "y:x", "type": "profile", "bins": 6,
                           "facet_by": "side"}]),
                verbose=False)
        # Delta form (P2-2): at most the dashboard figure itself is new
        # (it is created before the per-plot loop and stays open on raise —
        # pre-existing draw_figures behavior). Pre-fix the detached dfdraw
        # facet figure makes the delta 2; the guard fires before
        # plotter.draw(), so post-fix the delta is exactly the dashboard.
        assert len(set(plt.get_fignums()) - initial) <= 1

    def test_TG2b_facet_by_spec_skip_placeholder_no_leak(self):
        """T-G2b: skip — three-assertion form + delta-fignums.
        Pre-fix: detached faceted figure leaked, panel empty, no message."""
        adf = _mk()
        initial = set(plt.get_fignums())
        res = adf.draw_figures(
            _figspec([{"expr": "y:x", "type": "profile", "bins": 6,
                       "facet_by": "side"}]),
            on_error="skip", verbose=False)
        ax = res["f"]["axes"][0]
        assert ax.get_title().startswith('[ERROR]')
        assert res["f"]["stats"][0] is None
        assert any('facet_by' in t.get_text() for t in ax.texts)
        leaked = set(plt.get_fignums()) - initial
        assert leaked == {res["f"]["fig"].number}, (
            f"unexpected extra figures beyond the dashboard: {leaked}")


# =============================================================================
# T-G3 — F-A corrected behavior-matrix lock (fresh instance per sub-assert)
# =============================================================================

@pytest.mark.invariance
class TestG3SelectionAliasMatrix:

    def test_TG3_selection_alias_behavior_matrix(self):
        """T-G3: six sub-asserts locking the corrected C2/E-5 wording.
        Fresh ADF instance per sub-assert (C-5) so prior materialization
        cannot mask the lazy=False failures."""
        import pandas.errors as pde

        def fresh():
            a = _mk()
            a.add_alias("isGood", "(flag > 0)")
            return a

        # (1-3) plain selection= alias, lazy=False → UndefinedVariableError
        #       on ALL THREE surfaces (symmetric, loud post-13.55)
        with pytest.raises(pde.UndefinedVariableError):
            fresh().draw("y:x", selection="isGood==1")
        with pytest.raises(pde.UndefinedVariableError):
            fresh().draw_figures(
                _figspec([{"expr": "y:x", "selection": "isGood==1"}]),
                verbose=False)
        with pytest.raises(pde.UndefinedVariableError):
            fresh().draw_batch(
                {"k": {"expr": "y:x", "selection": "isGood==1"}},
                verbose=False)

        # (4-6) selection_vector= alias, lazy=False → works on ALL THREE
        #       (the 13.35 hook runs unconditionally on every surface)
        fig, ax, st = fresh().draw("x", type="hist", bins=10,
                                   selection_vector=["isGood>0", "isGood<1"])
        assert st is not None
        res = fresh().draw_figures(
            _figspec([{"expr": "x", "type": "hist", "bins": 10,
                       "selection_vector": ["isGood>0", "isGood<1"]}]),
            verbose=False)
        assert res["f"]["stats"][0] is not None
        resb = fresh().draw_batch(
            {"k": {"expr": "x", "type": "hist", "bins": 10,
                   "selection_vector": ["isGood>0", "isGood<1"]}},
            verbose=False)
        assert resb["k"]["stats"] is not None


# =============================================================================
# T-G4 — capability-matrix taxonomy registration
# =============================================================================

@pytest.mark.invariance
class TestG4Taxonomy:

    def test_TG4_dispatch_features_registered(self):
        """T-G4: DISPATCH.adf_routing + DISPATCH.error_visibility exist in
        the taxonomy and their patterns match this phase's + 13.55's tests."""
        import importlib.util, os
        here = os.path.dirname(__file__)
        spec = importlib.util.spec_from_file_location(
            "adf_feature_taxonomy", os.path.join(here, "feature_taxonomy.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        FEATURES = mod.FEATURES
        ids = {f["id"] for f in FEATURES}
        assert "DISPATCH.adf_routing" in ids
        assert "DISPATCH.error_visibility" in ids
        assert len(FEATURES) == 50  # 49 + LAZY.userinfo_backcompat (Phase 13.59.ADF)
        routing = next(f for f in FEATURES if f["id"] == "DISPATCH.adf_routing")
        vis = next(f for f in FEATURES if f["id"] == "DISPATCH.error_visibility")
        node_55 = ("test_phase_13_55_adf_dispatch_audit.py::"
                   "TestGroup1TypeCoverage::test_T1_draw_overlay_string")
        node_56 = ("test_phase_13_56_adf_post_audit.py::"
                   "TestG1Profile2dGuard::test_TG1a_profile2d_spec_default_raises")
        assert any(p in node_55 for p in routing["test_patterns"])
        assert any(p in node_56 for p in vis["test_patterns"])


# =============================================================================
# T-G5 — D4=A clean error (narrow intercept + negative control)
# =============================================================================

@pytest.mark.invariance
class TestG5AstypeTypeTokens:

    def test_TG5_astype_int_actionable_error(self):
        """T-G5: alias astype(int) → actionable error naming the quoted
        dtype form. Pre-fix: obscure 'Cannot interpret <function ...'."""
        adf = _mk()
        adf.add_alias("xi", "x.astype(int)")
        with pytest.raises(TypeError, match=r"astype\('int64'\)"):
            adf.materialize_aliases(names=["xi"])

    def test_TG5_negative_control_other_typeerrors_unchanged(self):
        """T-G5 negative control (§3.3): a genuine dtype error re-raises
        unchanged — the intercept is narrow."""
        adf = _mk()
        adf.add_alias("bad", "x.astype('not_a_dtype')")
        with pytest.raises(TypeError) as exc:
            adf.materialize_aliases(names=["bad"])
        assert "astype('int64')" not in str(exc.value)


# =============================================================================
# T-G6 — D1=A batch shims
# =============================================================================

@pytest.mark.invariance
class TestG6BatchShims:

    def test_TG6a_batch_literal_auto(self):
        """T-G6a: literal type='auto' in batch specs renders; includes one
        name-key spec (expr from spec name, C-2b). Pre-fix: ValueError."""
        adf = _mk()
        # NOTE (CRR finding F-13.56-1): the proposal C-2b name-key spec
        # cannot be locked — dfdraw draw_batch requires 'expr' in every
        # spec ("Missing 'expr' in spec"); the ADF-side
        # _merged_spec.get('expr', _name) fallback is defensive only.
        # Executed evidence in CRR; shim still reads via the fallback.
        res = adf.draw_batch(
            {"p1": {"expr": "x", "type": "auto", "bins": 10},
             "p2": {"expr": "y:x", "type": "auto", "bins": 10}},
            verbose=False)
        assert res["p1"]["stats"] is not None
        assert res["p2"]["stats"] is not None
        assert res.get("_errors", {}) == {}

    def test_TG6b_batch_three_var_profile_promotion(self):
        """T-G6b: 3-var type='profile' in a batch spec → profile2d renders
        (QuadMesh present). Pre-fix: ValueError 'Invalid expression'."""
        adf = _mk()
        res = adf.draw_batch(
            {"p2d": {"expr": "z:y:x", "type": "profile", "bins": 6}},
            verbose=False)
        assert res["p2d"]["stats"] is not None
        ax = res["p2d"]["ax"]
        from matplotlib.collections import QuadMesh
        assert any(isinstance(c, QuadMesh) for c in ax.collections)


# =============================================================================
# T-G7 — gallery-gap unit locks
# =============================================================================

@pytest.mark.invariance
class TestG7CoverageLocks:

    def test_TG7a_weights_alias_figures_and_batch(self):
        """T-G7a: weights= alias (lazy=True): weighted hist differs from
        unweighted on both batch surfaces."""
        adf = _mk()
        adf.add_alias("wbig", "w * w")

        def hmax(res_entry):
            ax = res_entry["axes"][0] if "axes" in res_entry else res_entry["ax"]
            ys = [float(p.get_xy()[:, 1].max()) if hasattr(p, "get_xy")
                  else float(p.get_height()) for p in ax.patches]
            return max(ys, default=0.0)

        ru = adf.draw_figures(
            _figspec([{"expr": "y", "type": "hist", "bins": 15}]),
            lazy=True, verbose=False)
        rw = adf.draw_figures(
            _figspec([{"expr": "y", "type": "hist", "bins": 15,
                       "weights": "wbig"}]),
            lazy=True, verbose=False)
        assert hmax(rw["f"]) != hmax(ru["f"])

        rbu = adf.draw_batch({"h": {"expr": "y", "type": "hist", "bins": 15}},
                             lazy=True, verbose=False)
        rbw = adf.draw_batch({"h": {"expr": "y", "type": "hist", "bins": 15,
                                    "weights": "wbig"}},
                             lazy=True, verbose=False)
        assert hmax(rbw["h"]) != hmax(rbu["h"])

    def test_TG7b_entry_window_via_figures(self):
        """T-G7b: entry_begin/entry_end through draw_figures: n == window."""
        adf = _mk(n=400)
        res = adf.draw_figures(
            _figspec([{"expr": "x", "type": "hist", "bins": 10}]),
            entry_begin=50, entry_end=150, verbose=False)
        assert res["f"]["stats"][0]["n"] == 100


# =============================================================================
# T-R1 — 3-level facet return contract (REGRESSION LOCK, panel M-1)
# =============================================================================

@pytest.mark.invariance
class TestR1ThreeLevelFacetLock:

    def test_TR1_three_level_facet_multi_figure_return(self):
        """T-R1 (lock — passes at HEAD): 3-level facet_by returns
        (list_of_figs, axes, stats); len == cardinality of the figID dim;
        each figure carries the row×col grid; lazy alias in the list works;
        4-level → clean NotImplementedError."""
        adf = _mk(n=600)
        adf.add_alias("qbin", "(x > 5)*1")          # lazy alias in facet list
        figs, axes, stats = adf.draw(
            "y:x", type="profile", bins=5,
            facet_by=["side", "qbin", "flag"], lazy=True)
        assert isinstance(figs, list)
        assert len(figs) == 3                        # card(flag) = 3
        for f in figs:
            assert len(f.axes) >= 4                  # 2×2 row×col grid
        with pytest.raises(NotImplementedError):
            adf.draw("y:x", type="profile", bins=5,
                     facet_by=["side", "qbin", "flag", "side"], lazy=True)


# =============================================================================
# T-H1 — draw_help full surface (+ negative control, P2-3)
# =============================================================================

@pytest.mark.invariance
class TestH1DrawHelp:

    def test_TH1_draw_help_full_surface(self, capsys):
        """T-H1: help lists overlay syntax, aliases, profile2d/scatter3d and
        the dfdraw-docs pointer — and (negative control) still lists the
        five core types."""
        adf = _mk()
        adf.draw_help()
        out = capsys.readouterr().out
        assert "hist2d+profile" in out
        assert "histo" in out
        assert "profile2d" in out and "scatter3d" in out
        assert "dfdraw" in out
        # negative control (panel P2-3)
        assert "hist" in out and "scatter" in out and "profile" in out
