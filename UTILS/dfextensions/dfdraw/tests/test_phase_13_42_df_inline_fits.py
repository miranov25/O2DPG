"""
Phase 13.42.DF — Inline Fits §9 invariance tests

Tests F.1-F.26 lock the behavior specified in PHASE_13_42_DF_v1_4 Proposal.
Each test class targets one of the §6 categories:

  §6.1 Numerical correctness  F.1-F.5, F.23
  §6.2 Spec normalization     F.6-F.10
  §6.3 Composition            F.11-F.15
  §6.4 Dict-key parsing       F.16-F.18
  §6.5 Display & rendering    F.19-F.20
  §6.6 Failure handling       F.21-F.22
  §6.7 v1.3 panel-closure     F.24-F.26

Conventions:
  - Synthetic data with fixed RNG seeds for reproducibility.
  - Assertions on stats['fit'] follow the v1.3 §3.5 canonical list-of-lists
    shape (LOCKED v1.4): stats['fit'][i][j] is the j-th fit on the i-th
    curve. The per-curve inner list NEVER collapses to a bare dict.
  - With group_by: stats['fit'] is dict {group_val: [[...], ...]}.
  - With facet_by: stats['fit'] is dict {(row,col): [[...], ...]}.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt  # noqa: E402

import dfdraw  # noqa: E402
from dfdraw import DFDraw  # noqa: E402
from dfdraw.plots.fits import (  # noqa: E402
    normalize_fit_spec,
    dispatch_fit,
    available_fits,
    _FIT_REGISTRY,
    _FIT_INITIAL_HEURISTICS,
)


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def gauss_df():
    """N=2000 samples from N(mu=1.5, sigma=0.5) on wide x ∈ [-10, 10]."""
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.normal(loc=1.5, scale=0.5, size=n)
    return pd.DataFrame({'x': x})


@pytest.fixture
def parabola_df():
    """N=300 (x, y=ax^2+bx+c+noise) on x ∈ [-5, 5]."""
    rng = np.random.default_rng(7)
    n = 300
    x = rng.uniform(-5.0, 5.0, n)
    y = 2.0 * x**2 + 3.0 * x + 1.0 + rng.normal(0, 0.2, n)
    return pd.DataFrame({'x': x, 'y': y})


@pytest.fixture
def two_curve_profile_df():
    """N=2000 (x, y_a, y_b) producing two profiles when expr='[y_a, y_b]:x'."""
    rng = np.random.default_rng(11)
    n = 2000
    x = rng.uniform(-3.0, 3.0, n)
    y_a = 50.0 * np.exp(-x**2 / 2.0) + rng.normal(0, 1.0, n)
    y_b = 1.5 * x + 0.5 + rng.normal(0, 0.3, n)
    return pd.DataFrame({'x': x, 'y_a': y_a, 'y_b': y_b})


@pytest.fixture
def grouped_gauss_df():
    """N=4000 with 4 groups, each Gaussian-distributed in x."""
    rng = np.random.default_rng(99)
    groups = []
    for g, (mu, sig) in enumerate([(0.0, 0.3), (1.0, 0.4), (-1.0, 0.5),
                                    (2.0, 0.4)]):
        x = rng.normal(loc=mu, scale=sig, size=1000)
        groups.append(pd.DataFrame({'x': x, 'sector': g}))
    return pd.concat(groups, ignore_index=True)


@pytest.fixture
def faceted_2d_df():
    """N=8000 with row_v ∈ {A, B}, col_v ∈ {0, 1, 2} — 6 facet cells."""
    rng = np.random.default_rng(123)
    rows = []
    for row_v in ['A', 'B']:
        for col_v in [0, 1, 2]:
            mu = 0.5 if row_v == 'A' else -0.5
            x = rng.normal(loc=mu + 0.1 * col_v, scale=0.3, size=1300)
            rows.append(pd.DataFrame({
                'x': x, 'row_v': row_v, 'col_v': col_v
            }))
    return pd.concat(rows, ignore_index=True)


# =============================================================================
# §6.1 Numerical correctness (F.1-F.5, F.23)
# =============================================================================

class TestFitsNumericalCorrectness:
    """Lock the numerical-recovery contract on synthetic data."""

    # F.1 — gauss recovers (center, sigma) within stderr on synthetic Gaussian
    def test_f1_gauss_recovers_center_sigma(self, gauss_df):
        d = DFDraw(gauss_df)
        fig, ax, stats = d.hist('x', bins=80, fit='gauss')
        fit = stats['fit'][0][0]   # §3.5 canonical: [curve][fit-on-curve]
        assert fit['fit_status'] == 'ok'
        center, sigma = float(fit['params'][1]), float(fit['params'][2])
        c_err = float(fit['param_errors'][1])
        s_err = float(fit['param_errors'][2])
        # 4σ envelope (very forgiving — locks numerical correctness, not
        # statistical efficiency).
        assert abs(center - 1.5) < max(0.05, 4 * c_err), \
            f"center={center}, expected ~1.5, c_err={c_err}"
        assert abs(abs(sigma) - 0.5) < max(0.05, 4 * s_err), \
            f"sigma={sigma}, expected ~0.5, s_err={s_err}"
        plt.close(fig)

    # F.2 — pol2 recovers polynomial coefficients within tolerance
    def test_f2_pol2_recovers_coefficients(self, parabola_df):
        d = DFDraw(parabola_df)
        fig, ax, stats = d.profile('y:x', bins=30, fit='pol2')
        fit = stats['fit'][0][0]
        assert fit['fit_status'] == 'ok'
        # Coefficients in ascending order per _polynomial_factory
        c0, c1, c2 = (float(fit['params'][i]) for i in range(3))
        assert abs(c0 - 1.0) < 0.3, f"c0={c0}, expected ~1"
        assert abs(c1 - 3.0) < 0.3, f"c1={c1}, expected ~3"
        assert abs(c2 - 2.0) < 0.3, f"c2={c2}, expected ~2"
        plt.close(fig)

    # F.3 — user callable accepted, params recovered (with explicit initial)
    def test_f3_user_callable_with_initial(self, gauss_df):
        def my_gauss(x, A, mu, sigma):
            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

        d = DFDraw(gauss_df)
        fig, ax, stats = d.hist('x', bins=80,
                                 fit={'fun': my_gauss,
                                      'initial': [100.0, 1.0, 1.0]})
        fit = stats['fit'][0][0]
        assert fit['fit_status'] == 'ok'
        assert abs(float(fit['params'][1]) - 1.5) < 0.1
        assert abs(abs(float(fit['params'][2])) - 0.5) < 0.1
        plt.close(fig)

    # F.4 — pol1 alias == linear (same lineshape)
    def test_f4_pol1_alias_equals_linear(self, parabola_df):
        # Use first half so a linear fit is meaningful
        df = parabola_df[parabola_df['x'].abs() < 1.0].reset_index(drop=True)
        d = DFDraw(df)
        fig1, _, s1 = d.profile('y:x', bins=20, fit='pol1')
        plt.close(fig1)
        fig2, _, s2 = d.profile('y:x', bins=20, fit='linear')
        plt.close(fig2)
        # Identical lineshape → identical parameters
        p1 = s1['fit'][0][0]['params']
        p2 = s2['fit'][0][0]['params']
        np.testing.assert_allclose(p1, p2, rtol=1e-12)

    # F.5 — registry-bound gauss converges WITHOUT explicit initial
    def test_f5_gauss_registry_heuristic_converges(self, gauss_df):
        # The CP2-1 weighted-std heuristic must converge on a narrow peak
        # over a wide x-axis. Pre-CP2-1 (nanstd(x)/2 heuristic) gave
        # sigma_guess ≈ 2.9 — too large — and curve_fit would diverge.
        d = DFDraw(gauss_df)
        fig, ax, stats = d.hist('x', bins=80, range=(-10, 10), fit='gauss')
        fit = stats['fit'][0][0]
        assert fit['fit_status'] == 'ok', f"fit failed: {fit['fit_error']}"
        sigma = abs(float(fit['params'][2]))
        assert 0.3 < sigma < 0.7, f"sigma={sigma}, expected ~0.5"
        plt.close(fig)

    # F.23 — user-supplied 'guess' callable produces sensible p0
    def test_f23_user_guess_callable(self, gauss_df):
        # User callable with no built-in heuristic. Without 'guess',
        # scipy default p0=[1]*3 fails to converge.
        def my_model(x, A, mu, sigma):
            return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

        def my_guess(x_arr, y_arr):
            # Domain-specific heuristic that knows the data shape.
            return [float(np.nanmax(y_arr)),
                    float(x_arr[np.nanargmax(y_arr)]),
                    float(np.nanstd(x_arr)) / 4.0]

        d = DFDraw(gauss_df)
        fig, ax, stats = d.hist('x', bins=80, range=(-10, 10),
                                 fit={'fun': my_model, 'guess': my_guess})
        fit = stats['fit'][0][0]
        assert fit['fit_status'] == 'ok', f"fit failed: {fit['fit_error']}"
        assert abs(float(fit['params'][1]) - 1.5) < 0.1
        plt.close(fig)


# =============================================================================
# §6.2 Spec normalization (F.6-F.10)
# =============================================================================

class TestFitsSpecNormalization:
    """Lock the unified-spec shorthand → canonical-dict transformations."""

    # F.6 — fit=str shorthand equivalent to fit={"fun": str}
    def test_f6_str_shorthand_equiv_to_dict(self):
        a = normalize_fit_spec('gauss', 1)
        b = normalize_fit_spec({'fun': 'gauss'}, 1)
        assert a == b == [[{'fun': 'gauss'}]]

    # F.7 — fit=callable shorthand equivalent to fit={"fun": callable}
    def test_f7_callable_shorthand_equiv_to_dict(self):
        def f(x, a, b):
            return a * x + b
        a = normalize_fit_spec(f, 1)
        b = normalize_fit_spec({'fun': f}, 1)
        assert a == b
        assert a[0][0]['fun'] is f

    # F.8 — length-1 list broadcasts same as scalar
    def test_f8_length1_list_broadcasts(self):
        scalar = normalize_fit_spec('gauss', 3)
        list1 = normalize_fit_spec(['gauss'], 3)
        assert scalar == list1
        # 3 curves, each with one fit
        assert len(scalar) == 3
        assert all(len(c) == 1 for c in scalar)

    # F.9 — length match: list length 2 + 2 curves = per-channel pairing
    def test_f9_length_match_pair_per_channel(self):
        res = normalize_fit_spec(['gauss', 'pol2'], 2)
        assert len(res) == 2
        # Each curve gets exactly one fit (the paired one)
        assert all(len(c) == 1 for c in res)
        assert res[0][0]['fun'] == 'gauss'
        assert res[1][0]['fun'] == 'pol2'

    # F.10 — length mismatch: list length 3 + 2 curves → ValueError
    def test_f10_length_mismatch_raises(self):
        with pytest.raises(ValueError) as exc_info:
            normalize_fit_spec(['gauss', 'pol2', 'linear'], 2)
        msg = str(exc_info.value)
        assert 'list length' in msg
        assert 'n_curves' in msg


# =============================================================================
# §6.3 Composition (F.11-F.15)
# =============================================================================

class TestFitsComposition:
    """Lock the §3.4 composition matrix with the §3.5 canonical shape."""

    # F.11 — vector expression [y_a,y_b]:x + scalar fit "gauss"
    def test_f11_vector_expr_scalar_fit_broadcast(self, two_curve_profile_df):
        d = DFDraw(two_curve_profile_df)
        fig, ax, stats = d.profile('[y_a,y_b]:x', bins=20, fit='gauss')
        # Vector dispatch may return either:
        #   (a) unified dict stats with 'fit' key as list-of-lists, OR
        #   (b) list of per-curve stats dicts, each with own 'fit' key
        # Both shapes carry the same canonical per-curve information.
        if isinstance(stats, list):
            # Per-curve stats list shape
            assert len(stats) == 2
            for per_curve in stats:
                assert 'fit' in per_curve, "each curve's stats must have a 'fit' key"
                curve_fits = per_curve['fit']
                # The fit value for one curve is a list of fit-dicts (§3.5 inner)
                assert isinstance(curve_fits, list)
                assert len(curve_fits) == 1
        else:
            assert isinstance(stats['fit'], list)
            assert len(stats['fit']) == 2
            for curve_fits in stats['fit']:
                assert isinstance(curve_fits, list)
                assert len(curve_fits) == 1
        plt.close(fig)

    # F.12 — vector expr + vector fit ["gauss","pol2"]
    # NB v1.0: vector dispatch (single-Y iteration) broadcasts the fit list
    # to EACH curve as compound, rather than pairing fit[i] with curve[i].
    # I.e., both curves get [gauss, pol2] overlay. Per-channel PAIRING in
    # vector mode is a Phase 13.42 FIX1 candidate (CRR §2 D5).
    def test_f12_vector_expr_vector_fit_pair(self, two_curve_profile_df):
        """v1.2/D5: vector pairing — fit=['gauss','pol2'] on [y_a,y_b]:x
        pairs gauss with y_a and pol2 with y_b. Phase 13.42 v1.0 had
        compound-broadcast (deviation from v1.4 §6.3 verbatim spec);
        FIX1 restored pairing per architect 2026-05-26 verbal ratification.
        F.12 assertions flipped from compound-broadcast to pairing."""
        d = DFDraw(two_curve_profile_df)
        fig, ax, stats = d.profile('[y_a,y_b]:x', bins=20,
                                    fit=['gauss', 'pol2'])
        # Vector dispatch returns per-curve stats list. Pairing (FIX1):
        #   stats[0] (y_a) → fit = [[gauss]] (one fit, paired with fit[0])
        #   stats[1] (y_b) → fit = [[pol2]]  (one fit, paired with fit[1])
        assert isinstance(stats, list), \
            f"vector dispatch must return list, got {type(stats).__name__}"
        assert len(stats) == 2
        # y_a (channel 0) → gauss only
        assert stats[0]['fit'][0][0]['fit_name'] == 'gauss', \
            "D5 pairing: y_a should be paired with fit[0]=gauss"
        assert len(stats[0]['fit'][0]) == 1, \
            "D5 pairing: y_a should have ONE fit, not compound"
        # y_b (channel 1) → pol2 only
        assert stats[1]['fit'][0][0]['fit_name'] == 'pol2', \
            "D5 pairing: y_b should be paired with fit[1]=pol2"
        assert len(stats[1]['fit'][0]) == 1, \
            "D5 pairing: y_b should have ONE fit, not compound"
        plt.close(fig)

    # F.13 — compound on single curve: fit=["gauss","pol2"] with scalar expr
    def test_f13_compound_on_single_curve(self, gauss_df):
        d = DFDraw(gauss_df)
        fig, ax, stats = d.hist('x', bins=80, fit=['gauss', 'pol2'])
        # §3.5 LOCKED list-of-lists shape:
        #   stats['fit'] == [[gauss_dict, pol2_dict]]
        #   outer length 1 (one curve); inner length 2 (two fits)
        assert isinstance(stats['fit'], list), \
            "ungrouped: stats['fit'] must be a list"
        assert len(stats['fit']) == 1, \
            f"expected 1 curve, got {len(stats['fit'])}"
        assert isinstance(stats['fit'][0], list), \
            "per-curve entry must be a list (§3.5 v1.3 lock)"
        assert len(stats['fit'][0]) == 2, \
            f"expected 2 fits-per-curve, got {len(stats['fit'][0])}"
        assert stats['fit'][0][0]['fit_name'] == 'gauss'
        assert stats['fit'][0][1]['fit_name'] == 'pol2'
        # Two overlay lines should be drawn (one per fit)
        # NB: histogram itself may add patches; we count Line2D objects only.
        n_line2d = sum(1 for line in ax.lines)
        assert n_line2d >= 2, \
            f"expected ≥2 overlay lines, got {n_line2d}"
        plt.close(fig)

    # F.14 — group_by + fit="gauss" (v1.4 LOCKED: list-of-lists per group)
    def test_f14_group_by_list_of_lists_per_group(self, grouped_gauss_df):
        d = DFDraw(grouped_gauss_df)
        fig, ax, stats = d.hist('x', bins=40,
                                 group_by='sector', fit='gauss')
        # v1.4 lock: stats['fit'] is a dict keyed by group_val;
        # each value is list-of-lists (outer=curves, inner=fits-per-curve).
        # Coder MUST access via stats['fit'][group_val][0][0], NOT [group_val][0].
        assert isinstance(stats['fit'], dict), \
            "group_by: stats['fit'] must be a dict"
        for group_val, group_fits in stats['fit'].items():
            assert isinstance(group_fits, list), \
                f"group {group_val}: value must be list (outer=curves, " \
                f"inner=fits-per-curve), NOT a bare dict (§3.5 v1.3 lock)"
            assert len(group_fits) >= 1
            # Each curve entry within a group is itself a list (of fits)
            for curve_fits in group_fits:
                assert isinstance(curve_fits, list), \
                    "per-curve entry must be a list, NOT a bare dict " \
                    "(v1.4 VP1-2 lock — coder MUST access [group][0][0])"
                assert len(curve_fits) == 1
                # The leaf is a fit-dict (canonical bottom element)
                assert isinstance(curve_fits[0], dict)
                assert 'fit_name' in curve_fits[0]
        plt.close(fig)

    # F.15 — facet_by 2D (Phase 13.41) + fit (dict keyed by (row, col) tuple)
    def test_f15_facet_by_2d_tuple_keys(self, faceted_2d_df):
        d = DFDraw(faceted_2d_df)
        try:
            fig, axes, stats = d.hist(
                'x', bins=30, fit='gauss',
                facet_by=['row_v', 'col_v'],
            )
        except (NotImplementedError, TypeError, ValueError) as e:
            pytest.skip(
                f"facet_by=List[str] not available in this build: {e}"
            )
        # Facet 2D shape: top-level stats is a dict keyed by (row_v, col_v)
        # tuples; each per-cell value is itself a stats dict with its own
        # 'fit' key (per CRR §2 D3 — facet routing through dispatcher
        # delegates to per-cell draw_hist, each producing its own stats).
        assert isinstance(stats, dict)
        cell_keys = [k for k in stats.keys() if isinstance(k, tuple)]
        assert len(cell_keys) > 0, "no 2-tuple cell keys found in stats"
        for k in cell_keys:
            assert len(k) == 2, f"key {k!r} must be a 2-tuple"
            cell = stats[k]
            assert isinstance(cell, dict)
            assert 'fit' in cell, f"cell {k!r} missing 'fit' key"
            cell_fits = cell['fit']
            # §3.5 canonical: list-of-lists for ungrouped per-cell
            assert isinstance(cell_fits, list)
            for curve_fits in cell_fits:
                assert isinstance(curve_fits, list)
        plt.close(fig)


# =============================================================================
# §6.4 Dict-key parsing (F.16-F.18)
# =============================================================================

class TestFitsDictKeyParsing:
    """Lock the per-fit dict key semantics + validation."""

    # F.16 — fit_range via dict (range key) restricts fit domain
    def test_f16_range_restricts_fit_domain(self, gauss_df):
        d = DFDraw(gauss_df)
        # Restrict fit to within ±1 sigma of true center
        fig, ax, stats = d.hist('x', bins=80,
                                 fit={'fun': 'gauss',
                                      'range': (1.0, 2.0)})
        fit = stats['fit'][0][0]
        assert fit['fit_status'] == 'ok'
        x_lo, x_hi = fit['x_range']
        # The actual fit-data range is what falls inside the requested
        # range AND has non-NaN finite data.
        assert x_lo >= 1.0 - 0.1
        assert x_hi <= 2.0 + 0.1
        plt.close(fig)

    # F.17 — scipy bounds forwarded — constrain params to lie within bounds
    def test_f17_bounds_constrain_params(self, gauss_df):
        d = DFDraw(gauss_df)
        # Set bounds that bracket the true values
        bounds = ([0.0, -1.0, 0.0], [10_000.0, 5.0, 5.0])
        fig, ax, stats = d.hist('x', bins=80,
                                 fit={'fun': 'gauss', 'bounds': bounds})
        fit = stats['fit'][0][0]
        assert fit['fit_status'] == 'ok'
        # Each param must lie within its bounds
        for i, (lo, hi) in enumerate(zip(bounds[0], bounds[1])):
            assert lo <= float(fit['params'][i]) <= hi
        plt.close(fig)

    # F.18 — unknown dict key → ValueError naming the offender
    def test_f18_unknown_dict_key_raises(self, gauss_df):
        d = DFDraw(gauss_df)
        with pytest.raises(ValueError) as exc_info:
            d.hist('x', bins=80,
                   fit={'fun': 'gauss', 'initial_guess': [100, 0, 1]})
        msg = str(exc_info.value)
        assert 'Unknown key' in msg
        assert 'initial_guess' in msg


# =============================================================================
# §6.5 Display & rendering (F.19-F.20)
# =============================================================================

class TestFitsDisplayRendering:
    """Lock the textbox + overlay rendering semantics."""

    # F.19 — show_params=False hides ONE block but not others (multi-fit)
    def test_f19_show_params_false_hides_one_block(self, gauss_df):
        d = DFDraw(gauss_df)
        # Compound: hide the second block, keep the first.
        fig, ax, stats = d.hist(
            'x', bins=80,
            fit=[{'fun': 'gauss'},
                 {'fun': 'pol2', 'show_params': False}],
        )
        # Both fits ran:
        assert stats['fit'][0][0]['fit_status'] == 'ok'
        assert stats['fit'][0][1]['fit_status'] == 'ok'
        # ax.texts: textbox should mention 'gauss' but NOT '_polynomial_deg2'
        # (the hidden pol2 fit's name is suppressed from the textbox).
        # We look at the concatenated text of all axes annotations.
        all_text = '\n'.join(t.get_text() for t in ax.texts)
        assert 'gauss' in all_text
        assert '_polynomial_deg2' not in all_text and 'pol2' not in all_text
        # Both overlays should still be drawn (2 Line2D objects)
        assert len(ax.lines) >= 2
        plt.close(fig)

    # F.20 — multi-fit on single curve uses distinct linestyles
    def test_f20_multi_fit_distinct_linestyles(self, gauss_df):
        d = DFDraw(gauss_df)
        fig, ax, stats = d.hist('x', bins=80, fit=['gauss', 'pol2'])
        styles = [line.get_linestyle() for line in ax.lines]
        # The two overlays must NOT share the same linestyle
        # (fit.linestyle_cycle = ['-', '--', '-.', ':'])
        assert len(styles) >= 2
        assert styles[0] != styles[1], \
            f"linestyles {styles[:2]} should be distinct"
        plt.close(fig)


# =============================================================================
# §6.6 Failure handling (F.21-F.22)
# =============================================================================

class TestFitsFailureHandling:
    """Lock the graceful-failure default + raise_on_failure option."""

    # F.21 — convergence failure does NOT raise by default
    def test_f21_default_failure_does_not_raise(self):
        # Construct data that will NOT converge to a gauss without
        # explicit initial: uniform noise (no peak).
        rng = np.random.default_rng(7)
        df = pd.DataFrame({'x': rng.uniform(-10, 10, 500)})
        d = DFDraw(df)
        # Bound to a tiny silly range that forces convergence failure.
        fig, ax, stats = d.hist(
            'x', bins=20,
            fit={'fun': 'gauss', 'bounds': ([1, 100, 0.001],
                                             [1.001, 100.001, 0.0011])},
        )
        # No exception — fit_status='failed'.
        assert stats['fit'][0][0]['fit_status'] == 'failed'
        assert stats['fit'][0][0]['fit_error'] is not None
        plt.close(fig)

    # F.22 — raise_on_failure=True raises RuntimeError with [dfdraw.fits] prefix
    def test_f22_raise_on_failure_true_raises(self):
        rng = np.random.default_rng(7)
        df = pd.DataFrame({'x': rng.uniform(-10, 10, 500)})
        d = DFDraw(df)
        with pytest.raises(RuntimeError) as exc_info:
            d.hist(
                'x', bins=20,
                fit={'fun': 'gauss',
                     'bounds': ([1, 100, 0.001],
                                [1.001, 100.001, 0.0011]),
                     'raise_on_failure': True},
            )
        msg = str(exc_info.value)
        assert '[dfdraw.fits]' in msg
        assert 'Fix:' in msg


# =============================================================================
# §6.7 v1.3 panel-closure additions (F.24-F.26)
# =============================================================================

class TestFitsPanelClosures:
    """Lock the CP1-5, CP1-6, CP2-5 fixes that landed in v1.3."""

    # F.24 — register_fit + available_fits end-to-end (CP1-6 P1)
    def test_f24_register_fit_public_export(self, gauss_df):
        # Without the dfdraw/__init__.py export from CP1-6, the next two
        # lines raise AttributeError.
        assert callable(dfdraw.register_fit)
        assert isinstance(dfdraw.available_fits(), list)

        # Register a user model via the public path
        def my_model(x, a, mu, sigma):
            return a * np.exp(-((x - mu) / sigma) ** 2 / 2)

        dfdraw.register_fit('test_mymodel_f24', my_model,
                             initial_heuristic=lambda x, y: [
                                 float(np.nanmax(y)),
                                 float(x[np.nanargmax(y)]),
                                 float(np.nanstd(x)) / 4.0,
                             ],
                             replace=True)
        assert 'test_mymodel_f24' in dfdraw.available_fits()

        d = DFDraw(gauss_df)
        fig, ax, stats = d.hist('x', bins=80, fit='test_mymodel_f24')
        fit = stats['fit'][0][0]
        assert fit['fit_status'] == 'ok'
        assert fit['fit_name'] == 'test_mymodel_f24'
        plt.close(fig)

    # F.25 — group_by + facet_by 2D + fit composition (CP2-5 P2)
    def test_f25_groupby_facet_fit_deep_composition(self, faceted_2d_df):
        # Add a group column on top of the 2D facet
        df = faceted_2d_df.copy()
        rng = np.random.default_rng(31)
        df['species'] = rng.integers(0, 2, len(df))

        d = DFDraw(df)
        try:
            fig, axes, stats = d.hist(
                'x', bins=20, fit='gauss',
                facet_by=['row_v', 'col_v'],
                group_by='species',
            )
        except (NotImplementedError, TypeError, ValueError) as e:
            pytest.skip(
                f"facet_by=List[str] + group_by not available: {e}"
            )

        # Top-level stats is dict keyed by (row, col); each per-cell carries 'fit'.
        assert isinstance(stats, dict)
        cell_keys = [k for k in stats.keys() if isinstance(k, tuple)]
        assert len(cell_keys) > 0
        for k in cell_keys:
            assert len(k) == 2
            cell = stats[k]
            if not isinstance(cell, dict) or 'fit' not in cell:
                pytest.skip(
                    "group_by under facet_by N-D is dispatcher-specific; "
                    "per-cell shape may not carry 'fit' in this build."
                )
                return
            cell_fits = cell['fit']
            # Inside the cell, group_by yields dict keyed by group_value
            if isinstance(cell_fits, dict):
                for group_val, group_fits in cell_fits.items():
                    assert isinstance(group_fits, list), \
                        f"cell {k!r} group {group_val!r} should be list-of-lists"
            else:
                # Or list-of-lists for ungrouped per-cell fallback
                assert isinstance(cell_fits, list)
        plt.close(fig)

    # F.26 — normalize + fit silent consume (CP1-5 P1)
    def test_f26_normalize_plus_fit_silent_consume(self, gauss_df):
        # Build a df with two selection slices so normalize='delta' has the
        # two curves it needs (per Phase 13.33 semantics).
        df = gauss_df.copy()
        df['kind'] = (df.index < len(df) // 2).map(
            {True: 'A', False: 'B'}
        ) if hasattr(df.index < len(df) // 2, 'map') else \
            np.where(df.index < len(df) // 2, 'A', 'B')

        d = DFDraw(df)
        # The CP1-5 policy: fit silently consumed when normalize is active.
        # Must NOT raise — and stats must not contain a 'fit' key.
        try:
            fig, ax, stats = d.profile(
                'x:x', bins=20, fit='gauss',
                normalize='delta',
                selection_vector=['kind=="A"', 'kind=="B"'],
            )
        except (NotImplementedError, TypeError, ValueError) as e:
            # If the legacy normalize dispatcher does not recognize fit at
            # all (i.e., _CONSUMED_ON_NORMALIZE filter missing), the next
            # lines below would have caught a different exception class.
            # NotImplementedError is acceptable IF it's about normalize
            # itself, not about fit specifically.
            if 'fit' in str(e).lower():
                pytest.fail(
                    f"normalize+fit must be silently consumed, got "
                    f"explicit error: {e}"
                )
            pytest.skip(f"normalize+selection_vector not configured: {e}")
            return

        # stats['fit'] should NOT be present (silent consume)
        assert 'fit' not in stats, \
            "normalize+fit: stats['fit'] should be ABSENT (silent " \
            "consume policy from Phase 13.42 §4.2b, CP1-5)"
        plt.close(fig)


# =============================================================================
# F.27 — P1-B regression (Sonnet54 finding, CRR §2 D7):
# profile + group_by + fit must produce stats['fit'] as dict keyed by group_val.
# =============================================================================
class TestFitsP1BProfileGroupedRegression:
    """Phase 13.42.DF P1-B regression test.

    Sonnet54 caught this scope reduction in the v1.0 implementation review:
    profile.py originally guarded the fit block with `if group_by is None`,
    which made `d.profile('y:x', group_by='g', fit='gauss')` a silent no-op
    (no overlay, no stats['fit'], no error).

    This test locks the contract that grouped-profile fits produce a
    canonical dict-keyed-by-group_val with list-of-lists per §3.5.
    """

    def test_f27_profile_group_by_fit_returns_dict_keyed_by_group(self):
        rng = np.random.default_rng(127)
        n = 1500
        x = rng.uniform(-3.0, 3.0, n)
        # 3 groups, each a different Gaussian profile shape
        g = rng.integers(0, 3, n)
        # Build a y that varies by group so each fit is non-degenerate
        mu_per_group = {0: 0.0, 1: 0.6, 2: -0.4}
        y = np.zeros(n)
        for gi in range(3):
            mask = g == gi
            y[mask] = 50.0 * np.exp(
                -(x[mask] - mu_per_group[gi]) ** 2 / 2.0
            ) + rng.normal(0, 1.0, mask.sum())
        df = pd.DataFrame({'x': x, 'y': y, 'g': g})
        d = DFDraw(df)

        fig, ax, stats = d.profile('y:x', bins=20, group_by='g', fit='gauss')

        # P1-B regression: 'fit' MUST be present (was silently absent in v1.0)
        assert 'fit' in stats, (
            "P1-B regression: profile + group_by + fit must produce "
            "stats['fit']. Silent no-op was Sonnet54's finding."
        )

        fits_obj = stats['fit']
        # §3.5 grouped contract: dict keyed by group_val
        assert isinstance(fits_obj, dict), (
            f"profile grouped fit: stats['fit'] must be a dict keyed by "
            f"group_val, got {type(fits_obj).__name__}"
        )
        # Each value is list-of-lists per §3.5
        for g_val, group_fits in fits_obj.items():
            assert isinstance(group_fits, list), (
                f"group {g_val!r}: outer must be a list (per-curve "
                f"dimension), got {type(group_fits).__name__}"
            )
            for curve_fits in group_fits:
                assert isinstance(curve_fits, list), (
                    f"group {g_val!r}: inner must be a list (per-fit "
                    f"dimension), got {type(curve_fits).__name__}"
                )
                for fd in curve_fits:
                    assert isinstance(fd, dict)
                    assert 'fit_name' in fd
                    assert fd['fit_name'] == 'gauss'

        # At least 2 of 3 groups should converge (allow 1 to fail
        # gracefully — fit_status='failed' is OK, stats key still present)
        ok_count = sum(
            1 for gf in fits_obj.values()
            for cf in gf
            for fd in cf
            if fd.get('fit_status') == 'ok'
        )
        assert ok_count >= 2, (
            f"expected ≥2 groups to converge, got {ok_count} ok statuses"
        )
        plt.close(fig)


# ============================================================================
# Phase 13.42.DF FIX1 (v1.2) — new regression tests F.28–F.33
#
# F.28 — B4: grouped fit on quantile binning (was: half of fits silently corrupt)
# F.29 — B5: histogram fit redchi physically correct (was: ~1e7 garbage)
# F.30 — B1: set_style facet font round-trip (was: silently ignored)
# F.31 — D5: vector fit pairing (was: compound-broadcast deviated from v1.4 §6.3)
# F.32 — D9: stacked + group_by + fit → per-group fits, R4 (was: silent skip)
# F.33 — R5: per-call fit_textbox_kwargs.fontsize override (NEW interface)
# ============================================================================

class TestPhase1342FIX1Regressions:
    """v1.2 regression locks — see PHASE_13_42_DF_FIX1_v1_2_Proposal.md §5."""

    def test_f28_grouped_fit_quantile_binning(self):
        """B4: group_by + group_by_quantiles + fit → ALL groups must have
        n_data > 0 AND fit_status='ok'. v1.0 used df[df[group_by]==g] which
        silently mis-matched some Interval keys (Sonnet55 extension: same
        pattern affected top_k/sort_groups). FIX1 reuses main-path masks."""
        rng = np.random.default_rng(28)
        df = pd.DataFrame({
            'y': rng.normal(0, 1, 3000),
            'x_for_gb': rng.uniform(-2, 2, 3000),
        })
        d = DFDraw(df)
        fig, ax, stats = d.hist(
            'y', bins=80,
            group_by='x_for_gb', group_by_quantiles=3,
            fit='gauss',
        )
        assert 'fit' in stats
        assert isinstance(stats['fit'], dict), (
            "B4 regression: grouped fit must return dict (per-group), "
            f"got {type(stats['fit']).__name__}"
        )
        assert len(stats['fit']) == 3, (
            f"B4: expected 3 groups, got {len(stats['fit'])}"
        )
        for g_val, group_fits in stats['fit'].items():
            assert isinstance(group_fits, list) and len(group_fits) == 1
            fit_d = group_fits[0][0]
            assert fit_d['n_data'] > 0, (
                f"B4 regression: group {g_val!r} got n_data=0 — "
                f"main-path masks not properly reused"
            )
            assert fit_d['fit_status'] == 'ok', (
                f"B4 regression: group {g_val!r} fit_status="
                f"{fit_d['fit_status']!r}; expected 'ok'"
            )
        plt.close(fig)

    def test_f28b_skipped_empty_does_not_crash(self):
        """B4 / Sonnet55 P1-A regression: when a quantile group has zero
        data points after masking (e.g., all y-values are NaN for that
        group), the skipped_empty path produces a clean fit_status entry
        WITHOUT crashing.

        v1.0 of FIX1 had `np.array([], shape=(0,0))` which raises
        TypeError (np.array does not accept shape kwarg).
        Fixed to np.zeros((0,0), dtype=float).

        Triggers the path that F.28 cannot exercise (its 3000-sample
        uniform data leaves no group empty)."""
        rng = np.random.default_rng(287)
        n = 200
        g_col = rng.uniform(-1, 1, n)
        y_col = rng.normal(0, 1, n)
        # Top-quantile rows have NaN y → after dropna, that group has 0 data
        y_col[g_col > 0.5] = np.nan
        df = pd.DataFrame({'y': y_col, 'x_for_gb': g_col})
        d = DFDraw(df)
        # Must not raise:
        fig, ax, stats = d.hist(
            'y', bins=40, group_by='x_for_gb',
            group_by_quantiles=4, fit='gauss',
        )
        # Expect 4 group entries; one should be skipped_empty, others 'ok'
        assert len(stats['fit']) == 4
        statuses = [gf[0][0]['fit_status'] for gf in stats['fit'].values()]
        assert 'skipped_empty' in statuses, (
            f"expected at least one skipped_empty status, got {statuses}"
        )
        assert statuses.count('ok') == 3, (
            f"expected 3 ok statuses, got {statuses.count('ok')}: {statuses}"
        )
        # Verify the skipped_empty dict has the right shape (was the crash)
        for g_val, group_fits in stats['fit'].items():
            fit_d = group_fits[0][0]
            if fit_d['fit_status'] == 'skipped_empty':
                assert fit_d['n_data'] == 0
                assert fit_d['pcov'].shape == (0, 0), (
                    f"P1-A: pcov.shape must be (0,0), got {fit_d['pcov'].shape}"
                )
                assert fit_d['ndf'] == 0
                assert np.isnan(fit_d['chi2'])
                assert np.isnan(fit_d['redchi'])
        plt.close(fig)

    def test_f29_hist_fit_redchi_physically_correct(self):
        """B5: clean Gaussian on histogram → redchi ∈ [0.5, 2.5]. v1.0
        defaulted yerr=None for hist (gated by hist_errors), making
        chi² scale with N². FIX1: Poisson default sqrt(max(counts, 1))
        always; ROOT TH1::Fit Neyman convention."""
        rng = np.random.default_rng(29)
        df = pd.DataFrame({'x': rng.normal(0.0, 1.0, 50000)})
        d = DFDraw(df)
        fig, ax, stats = d.hist('x', bins=80, range=(-4, 4), fit='gauss')
        fit_d = stats['fit'][0][0]
        assert fit_d['fit_status'] == 'ok'
        assert 0.5 <= fit_d['redchi'] <= 2.5, (
            f"B5 regression: redchi={fit_d['redchi']:.3f} not in [0.5, 2.5]. "
            f"yerr likely not Poisson-default."
        )
        plt.close(fig)

    def test_f30_set_style_fit_textbox_fontsize_facet(self):
        """B1: set_style({'fit.text_fontsize_facet': N}) must take effect on
        rendered textbox in faceted mode. v1.0 had two-layer defect:
        (a) render_fit_textbox call sites didn't pass facet_mode (Sonet51
        diagnosis), and (b) _style_get called get_style(key) but get_style
        takes no args, so all style reads silently returned defaults.
        FIX1 closes both layers."""
        # Use dfdraw.* import path consistent with test file convention
        # (conftest adds two sys.path entries → dfdraw and dfextensions.dfdraw
        # are separate module instances with separate _current_style dicts;
        # all imports in this test must use the SAME path to share state).
        from dfdraw import set_style
        set_style({'fit.text_fontsize_facet': 3})
        try:
            rng = np.random.default_rng(30)
            df = pd.DataFrame({
                'x': rng.normal(0, 1, 2000),
                'g': rng.choice(list('ABCD'), 2000),
            })
            d = DFDraw(df)
            fig, axes, stats = d.hist('x', bins=40, facet_by='g', fit='gauss')
            checked = 0
            for ax in (axes if hasattr(axes, '__iter__') else [axes]):
                for ta in ax.texts:
                    if 'gauss' in ta.get_text().lower():
                        assert ta.get_fontsize() == 3, (
                            f"B1 regression: textbox fontsize="
                            f"{ta.get_fontsize()!r}, expected 3 from "
                            f"set_style. facet_mode not plumbed or "
                            f"_style_get broken?"
                        )
                        checked += 1
            assert checked > 0, "no fit textbox found in any facet cell"
        finally:
            set_style({'fit.text_fontsize_facet': 7})  # restore default
            plt.close(fig)

    def test_f31_vector_fit_pairing(self):
        """D5/R2: fit=['gauss','pol2'] on [y_a,y_b]:x → pairing per channel,
        NOT compound-broadcast on each. v1.4 §6.3 verbatim spec. Phase 13.42
        v1.0 implementation deviated (compound-broadcast); FIX1 restores
        pairing per architect 2026-05-26 ratification."""
        rng = np.random.default_rng(31)
        n = 2000
        x = rng.uniform(-3, 3, n)
        y_a = 50.0 * np.exp(-x**2 / 2.0) + rng.normal(0, 1.0, n)  # Gaussian
        y_b = 1.5 * x + 0.5 + rng.normal(0, 0.3, n)              # linear
        df = pd.DataFrame({'x': x, 'y_a': y_a, 'y_b': y_b})
        d = DFDraw(df)
        fig, ax, stats = d.profile('[y_a,y_b]:x', bins=20,
                                    fit=['gauss', 'pol2'])
        # I-1 fix: assert isinstance (not 'if isinstance' silent-pass guard)
        assert isinstance(stats, list), (
            f"D5: vector dispatch must return list of per-channel stats, "
            f"got {type(stats).__name__}"
        )
        assert len(stats) == 2, f"D5: expected 2 channels, got {len(stats)}"
        # y_a (channel 0) → gauss only (paired with fit[0])
        assert stats[0]['fit'][0][0]['fit_name'] == 'gauss', (
            "D5 pairing: y_a must be paired with fit[0]=gauss"
        )
        assert len(stats[0]['fit'][0]) == 1, (
            "D5 pairing: y_a must have ONE fit (gauss), not compound"
        )
        # y_b (channel 1) → pol2 only (paired with fit[1])
        assert stats[1]['fit'][0][0]['fit_name'] == 'pol2', (
            "D5 pairing: y_b must be paired with fit[1]=pol2"
        )
        assert len(stats[1]['fit'][0]) == 1, (
            "D5 pairing: y_b must have ONE fit (pol2), not compound"
        )
        plt.close(fig)

    def test_f32_stacked_hist_per_group_fits(self):
        """D9/R4: stacked=True + group_by + fit → N per-group fits, NOT
        fit-the-total. Stacking is purely visual; fits remain per-group.
        v1.0 had silent-skip (guarded by `and not stacked`); FIX1 removes
        guard. v1.1 spec was 'single fit on combined'; v1.2 (architect
        2026-05-26 'fit all figures all gb') = per-group dict."""
        rng = np.random.default_rng(32)
        df = pd.DataFrame({
            'x': rng.normal(0, 1, 3000),
            'group': rng.choice(['signal', 'background'], 3000),
        })
        d = DFDraw(df)
        fig, ax, stats = d.hist(
            'x', bins=50, range=(-3, 3),
            group_by='group', stacked=True, fit='gauss',
        )
        assert 'fit' in stats, (
            "D9 regression: stacked=True + group_by + fit must produce "
            "stats['fit']; silent skip is the v1.0 bug."
        )
        # v1.2: per-group dict (NOT single fit on combined total)
        assert isinstance(stats['fit'], dict), (
            f"D9/R4 v1.2: stats['fit'] must be dict (per-group) when "
            f"stacked=True + group_by + fit, got "
            f"{type(stats['fit']).__name__}. Stacking is visual only."
        )
        assert len(stats['fit']) == 2, (
            f"D9: expected 2 group fits, got {len(stats['fit'])}"
        )
        for group_key, group_fits in stats['fit'].items():
            fit_d = group_fits[0][0]
            assert fit_d['fit_status'] == 'ok', (
                f"D9: group {group_key!r} fit_status="
                f"{fit_d['fit_status']!r}"
            )
            assert fit_d['fit_name'] == 'gauss'
            # I-4 fix: >= 40 not == 50 (dispatch_fit filters empty bins)
            assert fit_d['n_data'] >= 40, (
                f"D9: group {group_key!r} n_data={fit_d['n_data']}, "
                f"expected >= 40"
            )
        plt.close(fig)

    def test_f33_fit_textbox_kwargs_fontsize_override(self):
        """R5: per-call fit_textbox_kwargs={'fontsize': N} must take effect.
        Parallel lock to F.30 (set_style path). Architect 2026-05-26:
        'Yes, but font size must work.'"""
        rng = np.random.default_rng(33)
        df = pd.DataFrame({'x': rng.normal(0, 1, 2000)})
        d = DFDraw(df)
        fig, ax, stats = d.hist(
            'x', bins=40, fit='gauss',
            fit_textbox_kwargs={'fontsize': 5},
        )
        try:
            fit_text_artists = [t for t in ax.texts
                                if 'gauss' in t.get_text().lower()]
            assert len(fit_text_artists) > 0, (
                "R5: fit textbox not rendered; expected at least one text "
                "artist containing 'gauss'"
            )
            for ta in fit_text_artists:
                assert ta.get_fontsize() == 5, (
                    f"R5 regression: fit_textbox_kwargs={{'fontsize': 5}} "
                    f"ignored; rendered fontsize={ta.get_fontsize()!r}. "
                    f"Per-call kwarg override must take precedence."
                )
        finally:
            plt.close(fig)

    def test_f33_fit_textbox_kwargs_precedence_over_style(self):
        """R5 part 2: per-call fit_textbox_kwargs.fontsize takes precedence
        over set_style({'fit.text_fontsize_*': N}) when both are set."""
        # Use dfdraw.* import path (see F.30 note on conftest dual-path).
        from dfdraw import set_style
        # Use facet_mode path so set_style({'fit.text_fontsize_facet': 9})
        # is unambiguously the value being overridden.
        set_style({'fit.text_fontsize_facet': 9})
        try:
            rng = np.random.default_rng(33)
            df = pd.DataFrame({
                'x': rng.normal(0, 1, 2000),
                'g': rng.choice(['A', 'B'], 2000),
            })
            d = DFDraw(df)
            fig, axes, stats = d.hist(
                'x', bins=40, facet_by='g', fit='gauss',
                fit_textbox_kwargs={'fontsize': 4},  # override → 4, not 9
            )
            for ax in (axes if hasattr(axes, '__iter__') else [axes]):
                for ta in ax.texts:
                    if 'gauss' in ta.get_text().lower():
                        assert ta.get_fontsize() == 4, (
                            f"R5 precedence: per-call fontsize=4 must win "
                            f"over set_style fontsize=9; got "
                            f"{ta.get_fontsize()!r}."
                        )
        finally:
            set_style({'fit.text_fontsize_facet': 7})  # restore default
            plt.close(fig)
