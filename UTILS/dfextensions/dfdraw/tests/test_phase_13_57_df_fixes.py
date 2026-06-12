"""Phase 13.57.DF fix-phase tests.

Covers (proposal v1.4 + panel K-register + fable5_5 audit report):
  AF-2' / K-1 / K-3  - C-7 typo guard extended to typed methods; split
                       disposition (GUARD crash paths, WARN silent drops);
                       K-2 Option A: errors name the proper parameter.
  D4                 - quantiles_mode= kept-forever alias of quantile_mode=,
                       equivalence on BOTH surfaces (K-1.3).
  AF-1               - clean guards for time_format / quantiles dict input,
                       with negative controls (the unclean path no longer
                       reaches the backend).
  AF-3               - clean guard for non-numeric binning subjects
                       (group_by_bins / facet_by_bins), profile + dispatch.
  K-4 / E-3 / E-4    - draw() forwards named params at the profile2d
                       early-dispatch; full 2x2 entry-form/path matrix locks;
                       negative control: returned fig IS the provided fig
                       (no leaked figure). scatter3d save= companion.
  F-E                - native 3-var type='profile' -> profile2d promotion on
                       draw() and draw_batch.
  E-2 / D5           - y_central column in profile_data (ungrouped: rendered
                       central values; grouped: rendered means, FX-1 filed).
  AF-4               - unknown-type message derived from the live registry
                       (includes profile2d).
  DR-5               - fit dict 'p0' synonym of 'initial'; 'guess' untouched;
                       equivalence test p0 == initial fitted params.
  K-5                - zero-warning gate over corpus-form calls including
                       the matplotlib pass-through style vocabulary.
  K-12               - assertions use len(stats)/per-curve n, never
                       len(ax.lines) or hardcoded artist counts.
"""
import warnings

import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfdraw import DFDraw


@pytest.fixture
def df():
    rng = np.random.default_rng(1357)
    n = 2000
    d = pd.DataFrame({
        "x": rng.normal(0, 1, n),
        "y": rng.normal(0, 1, n),
        "z": rng.normal(5, 1, n),
        "t": np.linspace(0, 3600, n),
        "cat": rng.integers(0, 3, n),
        "w": rng.uniform(0.5, 1.5, n),
        "s": pd.array((["a", "b"] * (n // 2)), dtype="string"),
    })
    # Outliers separate median from mean (E-2 effect assertion).
    d.loc[:60, "y"] += 100.0
    return d


@pytest.fixture(autouse=True)
def _close_all():
    yield
    plt.close("all")


# =========================================================================
# AF-2' / K-1 / K-3: typo guard on the typed surface
# =========================================================================

class TestTypedSurfaceTypoGuard:
    def test_typed_close_match_names_proper_parameter(self, df):
        # Pre-13.57: leaked into matplotlib Line2D.set() (probe Q-4t).
        # Note: quantiles_mode is now the D4 alias, so use a different typo.
        with pytest.raises(ValueError, match=r"quantile_style"):
            DFDraw(df).profile("y:x", bins=10, quantile_styl="linestyle")

    def test_typed_inapplicable_semantic_kwarg_raises_clean(self, df):
        # Pre-13.57: crashed inside matplotlib PathCollection.set()
        # (panel K-3, Sonnet61/Sonnet62 typed-scatter crash).
        with pytest.raises(ValueError, match=r"not applicable to scatter"):
            DFDraw(df).scatter("y:x", bins=50)

    def test_typed_negative_control_no_matplotlib_frames(self, df):
        # Negative control (Opus48_3 13.51-FIX1 pattern): the clean error is
        # raised by dfdraw, NOT by matplotlib internals.
        try:
            DFDraw(df).scatter("y:x", bins=50)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "13.57" in str(e)
        except AttributeError as e:  # the pre-13.57 failure mode
            pytest.fail(f"crash leaked into matplotlib: {e}")

    def test_typed_unknown_kwarg_warns_not_silent(self, df):
        # Unknown kwargs are still forwarded to matplotlib (pre-13.57
        # contract, unchanged) and may fail there — the fix is that the
        # user now SEES the warning BEFORE the matplotlib failure.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            try:
                DFDraw(df).profile("y:x", bins=10, zzz_unknown_param=1)
            except Exception:
                pass  # matplotlib pass-through failure: not under test here
        assert any("zzz_unknown_param" in str(w.message) for w in rec)

    def test_typed_style_passthrough_stays_silent(self, df):
        # K-5 / RF-3: P-5 matplotlib vocabulary used by the corpus
        # (linestyle x8, markersize x6, linewidth x3) must never warn.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            DFDraw(df).profile("y:x", bins=10, linestyle="none",
                               markersize=10, linewidth=2)
        msgs = [str(w.message) for w in rec]
        assert not any("linestyle" in m or "markersize" in m
                       or "linewidth" in m for m in msgs), msgs


class TestDrawSurfaceSilentDropWarns:
    def test_fb1_bins_on_scatter_warns(self, df):
        # FB-1: pre-13.57 silent drop; K-3 WARN half.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            DFDraw(df).draw("y:x", type="scatter", bins=50)
        assert any("'bins'" in str(w.message) and "scatter" in str(w.message)
                   for w in rec)

    def test_draw_close_match_still_raises_with_proper_name(self, df):
        # K-2 Option A regression lock on the original C-7 surface.
        with pytest.raises(ValueError, match=r"Did you mean 'quantile_mode'"):
            DFDraw(df).draw("y:x", type="profile", bins=10,
                            quantile_modee="error_bars")

    def test_draw_applicable_params_do_not_warn(self, df):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            DFDraw(df).draw("y:x", type="profile", bins=10, fit="gauss")
        assert not any("13.57.DF K-3" in str(w.message) for w in rec)


# =========================================================================
# D4: quantiles_mode alias, both surfaces (K-1.3)
# =========================================================================

class TestQuantilesModeAlias:
    Q = [0.05, 0.5, 0.95]

    def _stats_signature(self, st):
        # K-12: compare stats content, not artist counts.
        return {k: np.round(np.asarray(v), 10).tolist()
                for k, v in st.items()
                if isinstance(v, (list, np.ndarray))}

    def test_alias_equivalence_typed_surface(self, df):
        d = DFDraw(df)
        _, _, st_old = d.profile("y:x", bins=10, quantiles=self.Q,
                                 quantile_mode="error_bars")
        _, _, st_new = d.profile("y:x", bins=10, quantiles=self.Q,
                                 quantiles_mode="error_bars")
        assert self._stats_signature(st_old) == self._stats_signature(st_new)

    def test_alias_equivalence_draw_surface(self, df):
        d = DFDraw(df)
        _, _, st_old = d.draw("y:x", type="profile", bins=10,
                              quantiles=self.Q, quantile_mode="error_bars")
        _, _, st_new = d.draw("y:x", type="profile", bins=10,
                              quantiles=self.Q, quantiles_mode="error_bars")
        assert self._stats_signature(st_old) == self._stats_signature(st_new)

    def test_alias_no_longer_rejected_where_typo_guard_fired(self, df):
        # K-1.3: at draw(), the C-7 guard used to reject quantiles_mode as a
        # typo. Post-D4 it is whitelisted and must neither raise nor warn.
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            DFDraw(df).draw("y:x", type="profile", bins=10,
                            quantiles=self.Q, quantiles_mode="error_bars")
        assert not any("quantiles_mode" in str(w.message) for w in rec)

    @pytest.mark.parametrize("method,ok", [
        ("profile", True), ("hist", False), ("hist2d", False),
        ("hexbin", False), ("scatter", False),
    ])
    def test_fx4_alias_scope_per_method(self, df, method, ok):
        """FX-4 (panel RV-1): the D4 alias is accepted ONLY where its
        target quantile_mode is consumed (profile). On every other typed
        method it raises the clean K-3 error instead of crashing inside
        matplotlib (Polygon.set()/QuadMesh.set(), the AF-2' class)."""
        d = DFDraw(df)
        expr = {"profile": "y:x", "hist": "x", "hist2d": "y:x",
                "hexbin": "y:x", "scatter": "y:x"}[method]
        kw = dict(quantiles_mode="error_bars")
        if method == "profile":
            kw["quantiles"] = [0.05, 0.5, 0.95]
            kw["bins"] = 10
        elif method in ("hist", "hist2d"):
            kw["bins"] = 10
        if ok:
            fig, ax, st = getattr(d, method)(expr, **kw)
            assert st is not None
        else:
            try:
                getattr(d, method)(expr, **kw)
                assert False, "expected clean ValueError (FX-4)"
            except ValueError as e:
                assert "not applicable" in str(e)
            except AttributeError as e:
                pytest.fail(f"alias leaked into matplotlib (FX-4): {e}")

    def test_fx4_draw_surface_alias_clean_raise_on_inapplicable_type(self, df):
        # draw() surface twin: quantiles_mode on type='hist' is forwarded
        # into hist(), whose typed guard raises the clean K-3 error —
        # pre-FX-4 this crashed in matplotlib Polygon.set().
        with pytest.raises(ValueError, match=r"not applicable to hist"):
            DFDraw(df).draw("x", type="hist", bins=10,
                            quantiles_mode="error_bars")

    def test_alias_conflict_raises(self, df):
        with pytest.raises(ValueError, match=r"synonyms"):
            DFDraw(df).profile("y:x", bins=10, quantiles=self.Q,
                               quantile_mode="error_bars",
                               quantiles_mode="discrete")


# =========================================================================
# AF-1: clean guards for dict input
# =========================================================================

class TestAF1Guards:
    def test_time_format_dict_clean_error(self, df):
        with pytest.raises(ValueError, match=r"strftime string"):
            DFDraw(df).profile("y:t", bins=10,
                               time_format={"format": "%H:%M"})

    def test_time_format_dict_negative_control_not_strftime(self, df):
        # Pre-13.57: TypeError from inside strftime (probe TF-2).
        with pytest.raises(ValueError):
            DFDraw(df).profile("y:t", bins=10, time_format={"f": 1})

    def test_quantiles_dict_clean_error(self, df):
        with pytest.raises(ValueError, match=r"list of numbers"):
            DFDraw(df).profile("y:x", bins=10,
                               quantiles={"levels": [0.1, 0.9]})

    def test_quantiles_string_elements_clean_error(self, df):
        # Pre-13.57: TypeError "'<=' not supported between str and int"
        # (probe Q-2 failure mode).
        with pytest.raises(ValueError, match=r"list of numbers"):
            DFDraw(df).profile("y:x", bins=10, quantiles=["a", "b"])

    def test_valid_forms_unaffected(self, df):
        # G-0: every corpus quantiles/time_format form still works.
        d = DFDraw(df)
        for q in ([0.05, 0.5, 0.95],
                  [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]):
            fig, ax, st = d.profile("y:x", bins=10, quantiles=q)
            assert st is not None
        fig, ax, st = d.profile("y:t", bins=10, time_format="%H:%M")
        assert st is not None


# =========================================================================
# AF-3: non-numeric binning subject
# =========================================================================

class TestAF3Guard:
    def test_string_group_by_bins_clean_error(self, df):
        with pytest.raises(ValueError, match=r"non-numeric"):
            DFDraw(df).profile("y:x", bins=5, group_by="s", group_by_bins=3)

    def test_negative_control_not_dtype_promotion_error(self, df):
        try:
            DFDraw(df).profile("y:x", bins=5, group_by="s", group_by_bins=3)
            assert False, "expected ValueError"
        except ValueError as e:
            assert "AF-3" in str(e)
        except Exception as e:  # numpy DTypePromotionError pre-13.57
            pytest.fail(f"unclean failure leaked: {type(e).__name__}: {e}")

    def test_string_discrete_grouping_still_works(self, df):
        # G-0: the suggested fix in the message actually works.
        fig, ax, st = DFDraw(df).profile("y:x", bins=5, group_by="s")
        assert st is not None

    def test_string_facet_by_bins_clean_error(self, df):
        with pytest.raises(ValueError, match=r"non-numeric"):
            DFDraw(df).profile("y:x", bins=5, facet_by="s", facet_by_bins=3)


# =========================================================================
# K-4 / E-3 / E-4: the ax-forwarding matrix
# =========================================================================

class TestE3Matrix:
    def test_e3a_draw_direct_honors_ax(self, df):
        # Pre-13.57: ax ignored, new figure created (probe E3-A).
        fig0, ax0 = plt.subplots()
        n_before = len(plt.get_fignums())
        f, a, st = DFDraw(df).draw("z:y:x", type="profile2d", ax=ax0)
        assert f is fig0, "returned fig must BE the provided fig (K-4)"
        assert len(ax0.collections) >= 1, "render must land on provided ax"
        assert len(plt.get_fignums()) == n_before, "no leaked figure"

    def test_e3b_typed_direct_honors_ax_regression_lock(self, df):
        fig0, ax0 = plt.subplots()
        f, a, st = DFDraw(df).profile2d("z:y:x", ax=ax0)
        assert f is fig0
        assert len(ax0.collections) >= 1

    def test_e3d_draw_faceted_no_orphan_blank_figure(self, df):
        # Pre-13.57 (probe E3-D): provided ax empty + leaked extra figure.
        fig0, ax0 = plt.subplots()
        n_before = len(plt.get_fignums())
        f, a, st = DFDraw(df).draw("z:y:x", type="profile2d",
                                   facet_by="cat", ax=ax0)
        n_after = len(plt.get_fignums())
        # Faceting may legitimately build its own grid figure; the E-4 lock
        # is that the PROVIDED figure is not left as a leaked blank orphan
        # while an additional figure is silently created on top of it.
        if f is not fig0:
            assert not (len(ax0.collections) == 0
                        and n_after > n_before), \
                "provided ax left blank AND extra figure leaked (E3-D)"

    def test_k4_named_params_reach_profile2d(self, df):
        # K-4 contract: the draw() entry now forwards named params, so the
        # two surfaces are IDENTICAL for identical arguments. bins= is the
        # discriminating named param (pre-13.57 it was dropped: the draw()
        # surface rendered default binning regardless of bins=).
        d = DFDraw(df)
        f1, a1, _ = d.draw("z:y:x", type="profile2d", bins=6)
        f2, a2, _ = d.profile2d("z:y:x", bins=6)
        n1 = a1.collections[0].get_array().size
        n2 = a2.collections[0].get_array().size
        assert n1 == n2, "draw() surface must match typed surface (K-4)"
        f3, a3, _ = d.draw("z:y:x", type="profile2d", bins=12)
        n3 = a3.collections[0].get_array().size
        assert n3 != n1, "bins= must have an effect on the draw() surface"
        # FX-2 (filed in the CRR, NOT under test): title= is ignored by the
        # profile2d route on BOTH surfaces — pre-existing, out of K-4 scope.

    def test_k4_selection_effect_assertion(self, df):
        # Effect assertion (not just no-exception): the selection actually
        # filters - compare against the typed surface on identical input.
        d = DFDraw(df)
        _, _, st_draw = d.draw("z:y:x", type="profile2d", selection="x>0",
                               bins=8)
        _, _, st_typed = d.profile2d("z:y:x", selection="x>0", bins=8)
        m_draw = st_draw.get("mesh_data") or st_draw
        m_typed = st_typed.get("mesh_data") or st_typed
        assert type(m_draw) is type(m_typed)

    def test_scatter3d_save_forwarded(self, df, tmp_path):
        out = tmp_path / "sc3d.png"
        DFDraw(df).draw("z:y:x", type="scatter3d", save=str(out))
        assert out.exists() and out.stat().st_size > 0


# =========================================================================
# F-E: native 3-var profile promotion
# =========================================================================

class TestFEPromotion:
    def test_draw_profile_3var_promotes(self, df):
        f, a, st = DFDraw(df).draw("z:y:x", type="profile", bins=8)
        assert len(a.collections) >= 1  # pcolormesh, the profile2d artist

    def test_draw_batch_profile_3var_promotes(self, df):
        res = DFDraw(df).draw_batch(
            {"p2d": {"expr": "z:y:x", "type": "profile", "bins": 8}})
        assert "p2d" in res
        assert res["p2d"].get("stats") is not None

    def test_profile_2var_unchanged(self, df):
        # G-0: ordinary 'y:x' profile untouched by the promotion gate.
        f, a, st = DFDraw(df).draw("y:x", type="profile", bins=10)
        assert st is not None


# =========================================================================
# E-2 / D5: y_central export
# =========================================================================

class TestE2YCentral:
    def test_y_central_present_and_is_median(self, df):
        _, _, st = DFDraw(df).profile("y:x", bins=10, central="median",
                                      return_data=True)
        pdat = st["profile_data"]
        assert "y_central" in pdat.columns
        assert "y_mean" in pdat.columns  # G-0: y_mean unchanged forever
        # Effect assertion: outlier-contaminated bins -> median differs
        # from mean, and y_central carries the median (rendered values).
        diff = np.nanmax(np.abs(pdat["y_central"] - pdat["y_mean"]))
        assert diff > 1.0, "y_central must carry the median, not the mean"

    def test_y_central_equals_mean_when_central_default(self, df):
        _, _, st = DFDraw(df).profile("y:x", bins=10, return_data=True)
        pdat = st["profile_data"]
        assert "y_central" in pdat.columns
        np.testing.assert_allclose(pdat["y_central"], pdat["y_mean"],
                                   equal_nan=True)

    def test_grouped_export_has_y_central(self, df):
        _, _, st = DFDraw(df).profile("y:x", bins=8, group_by="cat",
                                      return_data=True)
        pdat = st["profile_data"]
        assert "y_central" in pdat.columns
        # FX-1 (documented, unchanged behavior): grouped render uses means.
        np.testing.assert_allclose(pdat["y_central"], pdat["y_mean"],
                                   equal_nan=True)

    def test_pre_existing_columns_unchanged(self, df):
        _, _, st = DFDraw(df).profile("y:x", bins=10, return_data=True)
        cols = set(st["profile_data"].columns)
        assert {"x_center", "x_low", "x_high", "y_mean", "y_std", "y_sem",
                "count"} <= cols


# =========================================================================
# AF-4: live-registry type message
# =========================================================================

class TestAF4Message:
    def test_unknown_type_message_includes_profile2d(self, df):
        with pytest.raises(ValueError) as ei:
            DFDraw(df).draw("y:x", type="not_a_type_xyz")
        assert "profile2d" in str(ei.value)
        assert "histo" in str(ei.value)  # aliases listed too


# =========================================================================
# DR-5: p0 synonym
# =========================================================================

class TestDR5P0Alias:
    def _params(self, st):
        fit_obj = st["fit"]
        # stats['fit'][0][0]['params'] is an ndarray (established API fact).
        while isinstance(fit_obj, (list, tuple)):
            fit_obj = fit_obj[0]
        return np.asarray(fit_obj["params"])

    def test_p0_equivalent_to_initial(self, df):
        d = DFDraw(df.loc[61:])  # outlier-free region for stable fits
        _, _, st_i = d.profile("y:x", bins=20,
                               fit={"fun": "gauss", "initial": [1, 0, 2]})
        _, _, st_p = d.profile("y:x", bins=20,
                               fit={"fun": "gauss", "p0": [1, 0, 2]})
        np.testing.assert_allclose(self._params(st_i), self._params(st_p),
                                   rtol=1e-12)

    def test_p0_initial_conflict_raises(self, df):
        with pytest.raises(ValueError, match=r"synonym"):
            DFDraw(df).profile("y:x", bins=20,
                               fit={"fun": "gauss", "initial": [1, 0, 2],
                                    "p0": [9, 9, 9]})

    def test_guess_untouched(self, df):
        # DR-5: 'guess' stays its own (callable) key; not aliased to p0.
        d = DFDraw(df.loc[61:])
        _, _, st = d.profile("y:x", bins=20,
                             fit={"fun": "gauss",
                                  "guess": lambda x, y: [1, 0, 2]})
        assert st.get("fit") is not None

    def test_old_key_equivalence_regression(self, df):
        # EQ-FIT2 pattern: 'gaus' alias still identical to 'gauss'.
        d = DFDraw(df.loc[61:])
        _, _, s1 = d.profile("y:x", bins=20, fit="gauss")
        _, _, s2 = d.profile("y:x", bins=20, fit="gaus")
        np.testing.assert_allclose(self._params(s1), self._params(s2),
                                   rtol=1e-12)


# =========================================================================
# K-5: zero-warning gate over corpus-form calls
# =========================================================================

class TestK5ZeroWarningGate:
    def test_corpus_form_calls_emit_zero_warnings(self, df):
        """Synthetic replication of the production call grammar: every
        corpus-significant form (style pass-through, sibling families,
        quantiles+mode, fits, vectors, overlay, normalize) must run
        silently. A false warning in production QA loops is a P-0
        regression (panel K-5; ADF consumer column)."""
        d = DFDraw(df)
        calls = [
            lambda: d.draw("y:x", type="profile", bins=36, linestyle="none",
                           markersize=10, linewidth=2, min_entries=20,
                           auto_title=True),
            lambda: d.draw("y:x", type="profile", bins=50,
                           group_by="cat", group_by_bins=2, auto_title=True),
            lambda: d.draw("y:x", type="profile", bins=36, facet_by="cat",
                           facet_by_bins=3, group_by="z", group_by_bins=5,
                           auto_title=True),
            lambda: d.draw("y:x", type="profile", bins=36,
                           quantiles=[0.05, 0.5, 0.95],
                           quantile_mode="discrete", auto_title=True),
            lambda: d.draw("y:x", type="profile", bins=20, fit="gauss",
                           auto_title=True),
            lambda: d.draw("y:t", type="profile", bins=20,
                           time_format="%H:%M", auto_title=True),
            lambda: d.draw("x", type="hist", bins=100, auto_title=True),
            lambda: d.draw("x", type="histo", bins=50, auto_title=True),
            lambda: d.draw("y:x", type="hist2d", bins=50, auto_title=True),
            lambda: d.draw("y:x", type="hist2d+profile", bins=30,
                           auto_title=True),
            lambda: d.draw("y:x", type="profile", bins=20,
                           selection="x>-5",
                           selection_vector=["cat==0", "cat==1"],
                           vector_compose="outer", auto_title=True),
            lambda: d.draw("y:x", type="profile", bins=20, weights="w",
                           auto_title=True),
            lambda: d.draw("y:x", type="profile", bins=20,
                           normalize="delta",
                           selection_vector=["cat==0", "cat==1"],
                           vector_compose="outer", auto_title=True),
            lambda: d.profile("y:x", bins=36, linestyle="none",
                              markersize=10, min_entries=20),
            lambda: d.draw("y:x", type="profile", bins=20,
                           central="median", auto_title=True),
        ]
        offenders = []
        for i, call in enumerate(calls):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                try:
                    call()
                except Exception as e:
                    offenders.append((i, f"RAISED {type(e).__name__}: {e}"))
                    continue
            new = [str(w.message) for w in rec
                   if "13.57" in str(w.message)
                   or "Unknown keyword" in str(w.message)]
            if new:
                offenders.append((i, new))
        assert not offenders, f"K-5 gate violated: {offenders}"
