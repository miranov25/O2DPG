"""
Phase 13.50.DF — Fit-rendering visual primitives (Tier 1: primitive-only)

Sibling to ``test_phase_13_48_df_visual_testing.py`` for fit-specific render
checks. Same architecture: collect-all-then-assert, renderer-free,
backend-independent, asserts on ``fig.axes`` primitives via ``Text.get_text()``
(stored input strings — NOT rasterized glyphs; Tier 2 work for the latter).

This module starts (Phase 13.50 step 1 of §10) with three display-name checks
(F1/F2/F3). Steps 2-7 will extend ``FitVisualCheck`` with helpers for
precision, legend topology, summary_fit table semantics, etc.

References:
  - Proposal: PHASE_13_50_DF_v2_5_FitRenderingOverhaul_Proposal.md §3.7, §3.8
  - Source: plots/_fit_render.py:_DISPLAY_NAMES, plots/_fit_render.py:_resolve_display_name
  - Template: tests/test_phase_13_48_df_visual_testing.py:VisualCheck

All mechanics were source-verified against PHASE_13_49_DF_FIX1_END (a6ddc753)
before this suite was written (per Coder QRC v1.29 R16).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from dfextensions.dfdraw import DFDraw
# Phase 13.50 step 4 — direct import for normalizer idempotency invariance test
from dfextensions.dfdraw.plots._legend import _normalize_legend_spec


# --------------------------------------------------------------------------- #
# Primitive helpers — fit-textbox content via Text.get_text()
# --------------------------------------------------------------------------- #
# Renderer-free Tier 1 rule: Text.get_text() returns the STORED input string
# (e.g. r'$\mu$' literal), NOT the rasterized glyph 'μ'. Glyph-level checks
# require a renderer draw cycle (Tier 2, deferred to Phase 13.5X).

def textbox_text(ax):
    """Concatenate all ax.texts strings on a single axes.

    Returns the raw stored text (mathtext literals like ``$\\mu$`` appear
    verbatim — see module docstring on the Tier 1 / Tier 2 distinction).
    """
    return " ".join(t.get_text() for t in ax.texts)


def display_names_in_textbox(ax, name_set):
    """Return the subset of ``name_set`` that appears in this ax's textboxes.

    Substring match on the joined ``textbox_text``. Useful for asserting that
    a chosen short-name set IS or IS NOT present.
    """
    blob = textbox_text(ax)
    return {n for n in name_set if n in blob}


# Phase 13.50 step 2 — value/error precision parsing helpers.

import re

# Tolerant of mathtext names (e.g. '$\mu$=1.23±0.05'): allow $...$ + alnum +
# underscores in the name portion. ± may be the literal or fallback '+-'.
_VAL_ERR_PATTERN = re.compile(
    r'(?P<name>\S+?)\s*=\s*'
    r'(?P<val>[-+]?[0-9.eE+\-]+)'
    r'\s*[±]\s*'
    r'(?P<err>[-+]?[0-9.eE+\-]+)'
)


def parse_value_error(text):
    """Extract list of ``(name_str, value_str, error_str)`` tuples from textbox text.

    Returns the raw strings as rendered (not floats) — F4/F5 assertions are
    on the rendered format itself, not on numerical accuracy of the fit.
    """
    return [(m.group('name'), m.group('val'), m.group('err'))
            for m in _VAL_ERR_PATTERN.finditer(text)]


def _count_sig_figs(s):
    """Count significant figures in a number string like '1.23' or '0.045' or '1.2e-3'."""
    s = s.strip().lower()
    if 'e' in s:
        s = s.split('e')[0]
    # Strip sign, decimal point, and leading zeros
    digits = s.replace('.', '').replace('-', '').replace('+', '').lstrip('0')
    return len(digits)


def _decimal_places(s):
    """Count digits after the decimal point in a number string.

    For 'NeM' scientific notation, returns 0 (rare in dfdraw outputs since
    .2g/.1g default formats prefer plain decimals for moderate magnitudes).
    """
    s = s.strip().lower()
    if 'e' in s:
        return 0
    if '.' not in s:
        return 0
    return len(s.split('.')[1])


# Phase 13.50 step 4 — legend topology helper (extends 4-tuple per P2-NEW-2
# of v2.3 panel: count-tuple was necessary-but-not-sufficient; add label
# set + loc string to catch label drops and location moves).
def _matplotlib_legend_ncols(legend):
    """Compat: matplotlib renamed Legend._ncol → Legend._ncols in 3.7+."""
    return getattr(legend, '_ncols', getattr(legend, '_ncol', None))


def _matplotlib_legend_loc_str(legend):
    """Best-effort loc readback. ``Legend._loc_real`` is private (P3-2 risk
    in v2.5 §6) but stable in current matplotlib. Returns an int code (0-10)
    if available, else None."""
    return getattr(legend, '_loc_real', getattr(legend, '_loc', None))


def legend_topology(fig):
    """Return ``(n_fig_level, n_per_axes, frozenset(labels), loc_string)``.

    - ``n_fig_level``  : count of ``fig.legends`` (figure-level legends).
    - ``n_per_axes``   : count of ``ax.get_legend() is not None`` over fig.axes.
    - ``labels``       : frozenset union of all legend entry labels across
                         fig-level and per-axes legends. P3-3 caveat: if two
                         fits share an auto-label, frozenset deduplicates;
                         test fixtures must use distinct labels per fit.
    - ``loc_string``   : ``_loc_real`` int code of the FIRST fig-level legend
                         (if any), else None.

    Used by F18 (show_legend ↔ legend behavioral equivalence) and F10 (dict
    forwards loc + ncol).
    """
    n_fig_level = len(fig.legends)
    n_per_axes = sum(1 for ax in fig.axes if ax.get_legend() is not None)
    labels = []
    for leg in fig.legends:
        labels.extend(t.get_text() for t in leg.get_texts())
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            labels.extend(t.get_text() for t in leg.get_texts())
    loc_string = _matplotlib_legend_loc_str(fig.legends[0]) if fig.legends else None
    return (n_fig_level, n_per_axes, frozenset(labels), loc_string)


# --------------------------------------------------------------------------- #
# FitVisualCheck — sibling to Phase 13.48 VisualCheck
# --------------------------------------------------------------------------- #
# Same collect-all-then-assert idiom: methods append to self.defects; the
# final .assert_clean() raises with the full defect list.

class FitVisualCheck:
    def __init__(self, fig, stats=None, df=None):
        self.fig = fig
        self.stats = stats
        self.df = df
        self.defects = []

    def fail(self, name, detail):
        self.defects.append(f"{name}: {detail}")

    def check_textbox_uses_display_names_linear(self, ax):
        """F1 — linear fit: textbox shows p0/p1 (display names) and does NOT
        show slope/intercept (canonical Python identifiers)."""
        blob = textbox_text(ax)
        # Canonical names that MUST NOT appear in user-facing render
        leaked = {n for n in ('slope', 'intercept') if n in blob}
        if leaked:
            self.fail("F1.canonical_leak",
                      f"canonical names {sorted(leaked)} leaked into textbox: {blob!r}")
        # Display names that MUST appear (at least one — p0 or p1)
        if not ({'p0', 'p1'} & set(blob.split())):
            # Fall back to substring search (e.g. 'p1=1.02±0.03' has 'p1' as
            # substring not separate token); use 'in' for that case.
            if not ('p0' in blob or 'p1' in blob):
                self.fail("F1.display_missing",
                          f"display names p0/p1 missing from textbox: {blob!r}")
        return self

    def check_textbox_uses_mathtext_mu_for_gauss(self, ax):
        """F2 — gauss fit: textbox contains the '$\\mu$' mathtext literal
        for the canonical ``center`` parameter."""
        blob = textbox_text(ax)
        if r'$\mu$' not in blob:
            self.fail("F2.mu_missing",
                      f"'$\\\\mu$' literal missing from gauss textbox: {blob!r}")
        # Canonical 'center' must NOT leak
        if 'center' in blob:
            self.fail("F2.canonical_leak",
                      f"canonical 'center' leaked into gauss textbox: {blob!r}")
        return self

    def check_textbox_uses_mathtext_tau_for_exponential(self, ax):
        """F3 — exponential fit: textbox contains the '$\\tau$' mathtext
        literal for the canonical ``decay`` parameter."""
        blob = textbox_text(ax)
        if r'$\tau$' not in blob:
            self.fail("F3.tau_missing",
                      f"'$\\\\tau$' literal missing from expo textbox: {blob!r}")
        if 'decay' in blob:
            self.fail("F3.canonical_leak",
                      f"canonical 'decay' leaked into expo textbox: {blob!r}")
        return self

    # ------------------------------------------------------------------ #
    # Phase 13.50 step 2 — precision-key checks
    # ------------------------------------------------------------------ #

    def check_default_precision_value_2g_error_1g(self, ax):
        """F4 — default render: every parsed value has ≤2 sig figs, every
        error has ≤1 sig fig (matches ``fit.value_format='.2g'`` /
        ``fit.error_format='.1g'`` defaults). Locks the [BREACH] migration
        from single ``fit.text_format`` to the separate keys."""
        blob = textbox_text(ax)
        pairs = parse_value_error(blob)
        if not pairs:
            self.fail("F4.no_pairs",
                      f"no value±error pairs found in textbox: {blob!r}")
            return self
        for (name, val_s, err_s) in pairs:
            val_sf = _count_sig_figs(val_s)
            err_sf = _count_sig_figs(err_s)
            if val_sf > 2:
                self.fail("F4.value_sf",
                          f"{name}: value '{val_s}' has {val_sf} sf, expected ≤2 (.2g)")
            if err_sf > 1:
                self.fail("F4.error_sf",
                          f"{name}: error '{err_s}' has {err_sf} sf, expected ≤1 (.1g)")
        return self

    def check_precision_mode_physics_aligns_value_to_error(self, ax):
        """F5 — with ``fit.precision_mode='physics'``: every parsed pair has
        the same number of digits after the decimal point in value and error
        (error rounded to 1 sf, value matched to error's decimal place)."""
        blob = textbox_text(ax)
        pairs = parse_value_error(blob)
        if not pairs:
            self.fail("F5.no_pairs",
                      f"no value±error pairs found in physics-mode textbox: {blob!r}")
            return self
        for (name, val_s, err_s) in pairs:
            val_dec = _decimal_places(val_s)
            err_dec = _decimal_places(err_s)
            if val_dec != err_dec:
                self.fail("F5.misaligned",
                          f"{name}: value '{val_s}' has {val_dec} decimals, "
                          f"error '{err_s}' has {err_dec}; should match in physics mode")
        return self

    # ------------------------------------------------------------------ #
    # Phase 13.50 step 3 — fit_textbox_kwargs extension checks
    # ------------------------------------------------------------------ #

    def check_rename_params_overrides_display_map(self, ax,
                                                  expected_literal='sigma_x',
                                                  forbidden_default=r'$\sigma$'):
        """F6 — ``fit_textbox_kwargs={'rename_params': {canonical: display}}``
        beats the ``_DISPLAY_NAMES`` map for the listed canonical name(s)."""
        blob = textbox_text(ax)
        if expected_literal not in blob:
            self.fail("F6.override_missing",
                      f"'{expected_literal}' (rename_params override) missing "
                      f"from textbox: {blob!r}")
        if forbidden_default in blob:
            self.fail("F6.default_leaked",
                      f"'{forbidden_default}' (default display) leaked despite "
                      f"rename_params override: {blob!r}")
        return self

    def check_value_error_format_per_call_override(self, ax, min_sf_required=3):
        """F16 — ``fit_textbox_kwargs={'value_format': '.4g'}`` grants more
        precision than the ``.2g`` style default.

        Assertion: at least one parsed value has more than 2 sig figs
        (impossible under the default ``.2g``). If the fitted values happen
        to all round to ≤2 sf even under ``.4g``, the fixture isn't
        discriminating — test would fail loud and the fixture must be
        retuned to non-round true coefficients.
        """
        blob = textbox_text(ax)
        pairs = parse_value_error(blob)
        if not pairs:
            self.fail("F16.no_pairs",
                      f"no value±error pairs found in textbox: {blob!r}")
            return self
        max_sf = max(_count_sig_figs(val_s) for (_, val_s, _) in pairs)
        if max_sf < min_sf_required:
            self.fail("F16.no_override_effect",
                      f"max value sig figs = {max_sf}, expected ≥{min_sf_required} "
                      f"(per-call '.4g' override should grant more precision than "
                      f"default '.2g'). Pairs: {pairs!r}")
        return self

    # ------------------------------------------------------------------ #
    # Phase 13.50 step 4 — legend mode checks
    # ------------------------------------------------------------------ #

    def check_legend_shared_one_fig_level_zero_per_axes(self):
        """F7 — legend='shared': one fig-level legend, zero per-axes legends."""
        n_fig_level, n_per_axes, _, _ = legend_topology(self.fig)
        if n_fig_level != 1:
            self.fail("F7.fig_level_count",
                      f"expected exactly 1 fig-level legend, got {n_fig_level}")
        if n_per_axes != 0:
            self.fail("F7.per_axes_count",
                      f"expected 0 per-axes legends (consolidated into fig), got {n_per_axes}")
        return self

    def check_legend_first_only_axes_flat_zero(self):
        """F8 — legend='first': axes.flat[0] has its legend kept, others stripped."""
        if not self.fig.axes:
            self.fail("F8.no_axes", "figure has no axes")
            return self
        first_has = self.fig.axes[0].get_legend() is not None
        others_clean = all(ax.get_legend() is None for ax in self.fig.axes[1:])
        if not first_has:
            self.fail("F8.first_missing",
                      "axes.flat[0] should have its legend; it's None")
        if not others_clean:
            n_leaked = sum(1 for ax in self.fig.axes[1:]
                           if ax.get_legend() is not None)
            self.fail("F8.others_leaked",
                      f"axes.flat[1:] should have zero legends, got {n_leaked}")
        return self

    def check_legend_false_no_legends_anywhere(self):
        """F9 — legend=False: zero fig-level AND zero per-axes legends."""
        n_fig_level, n_per_axes, _, _ = legend_topology(self.fig)
        if n_fig_level != 0:
            self.fail("F9.fig_level_count",
                      f"expected 0 fig-level legends, got {n_fig_level}")
        if n_per_axes != 0:
            self.fail("F9.per_axes_count",
                      f"expected 0 per-axes legends, got {n_per_axes}")
        return self

    def check_legend_dict_forwards_loc_and_ncol(self, expected_ncol):
        """F10 — legend={'mode':'shared','loc':...,'ncol':N}: the dict kwargs
        reach fig.legend() — verified via fig.legends[0]._ncols (matplotlib
        3.7+; older: _ncol). Loc readback is left to legend_topology's
        loc_string field; F10 specifically asserts ncol propagates."""
        if not self.fig.legends:
            self.fail("F10.no_fig_legend",
                      f"expected fig-level legend from dict mode='shared', got none")
            return self
        fig_leg = self.fig.legends[0]
        actual_ncol = _matplotlib_legend_ncols(fig_leg)
        if actual_ncol != expected_ncol:
            self.fail("F10.ncol_not_forwarded",
                      f"expected ncol={expected_ncol}, fig.legends[0]._ncols={actual_ncol}")
        return self

    def check_show_legend_legend_behavioral_equivalence(self, topology_a, topology_b):
        """F18 — show_legend=X and legend=X produce identical legend_topology
        4-tuples. Caller captures topologies from two runs and passes them in.

        Per P2-NEW-2 of v2.3 panel: 2-tuple (counts only) was necessary but
        not sufficient — extended to 4-tuple (counts + label frozenset + loc)
        catches label drops and location moves that count-only equivalence
        misses.
        """
        if topology_a != topology_b:
            # Find which field differs for a clearer error
            fields = ('n_fig_level', 'n_per_axes', 'labels', 'loc_string')
            mismatches = [f"{name}: {a!r} vs {b!r}"
                          for name, a, b in zip(fields, topology_a, topology_b)
                          if a != b]
            self.fail("F18.equivalence_broken",
                      f"legend= vs show_legend= produce different topology: "
                      f"{'; '.join(mismatches)}")
        return self

    def assert_clean(self):
        if self.defects:
            raise AssertionError(
                "FitVisualCheck defects:\n  - " + "\n  - ".join(self.defects)
            )


# --------------------------------------------------------------------------- #
# Fixtures — synthetic data shaped for the canonical fit kinds
# --------------------------------------------------------------------------- #
# Deterministic seed=0. ~200 rows minimum per fit case — enough for stable
# fits without slowing the gate.

@pytest.fixture
def df_linear():
    """y = 2*x + 1 + Gaussian noise (σ=0.1). 200 rows. Exercises fit='linear'."""
    rs = np.random.RandomState(0)
    x = np.linspace(-3.0, 3.0, 200)
    y = 2.0 * x + 1.0 + rs.normal(0, 0.1, 200)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def df_gauss():
    """1000 samples ~ N(mu=0.5, sigma=0.7). Exercises fit='gauss' on a 1D hist."""
    rs = np.random.RandomState(0)
    samples = rs.normal(0.5, 0.7, 1000)
    return pd.DataFrame({"x": samples})


@pytest.fixture
def df_expo():
    """y = 3.0 * exp(-x / 1.5) + noise on x in [0, 6]. ~200 rows. Exercises fit='expo'."""
    rs = np.random.RandomState(0)
    x = np.linspace(0.0, 6.0, 200)
    y = 3.0 * np.exp(-x / 1.5) + rs.normal(0, 0.03, 200)
    return pd.DataFrame({"x": x, "y": y})


@pytest.fixture
def df_faceted():
    """600 rows in a 1×4 facet grid (sec=0..3) × 3 groups (g=0..2).

    Used for Phase 13.50 step 4 legend-mode tests (F7-F10, F18). Need
    multi-axes + multi-group so legend='shared' has something to consolidate
    and the topology helpers see >1 axes/labels.
    """
    rs = np.random.RandomState(0)
    n = 600
    return pd.DataFrame({
        "x":   rs.normal(0, 1, n),
        "y":   rs.normal(0, 1, n),
        "g":   rs.randint(0, 3, n),
        "sec": rs.randint(0, 4, n),
    })


# --------------------------------------------------------------------------- #
# Tests F1, F2, F3 — display-name map
# --------------------------------------------------------------------------- #

class TestPhase1350FitDisplayNames:
    """Phase 13.50 step 1: canonical names render as short/Greek display
    names. Locks _DISPLAY_NAMES + _resolve_display_name in
    plots/_fit_render.py.

    EXPECTED · WHY · APPROVE · FAIL_MODE per V-check (proposal §3.10 contract):

      F1: EXPECTED textbox shows 'p0'/'p1' for fit='linear';
          WHY long canonical names ('slope', 'intercept') consume horizontal
          space and conflict with polynomial naming convention (c0, c1);
          APPROVE 'p0' or 'p1' substring present AND no 'slope'/'intercept';
          FAIL_MODE canonical name leaked into render (display map not wired).

      F2: EXPECTED textbox contains '$\\mu$' literal for fit='gauss';
          WHY Greek mathtext matches physics convention; saves horizontal
          space vs 'center';
          APPROVE '$\\mu$' substring present via Text.get_text() AND no
          'center';
          FAIL_MODE _DISPLAY_NAMES missing the 'center' → '$\\mu$' entry, or
          _resolve_display_name not called at the rendering site.

      F3: EXPECTED textbox contains '$\\tau$' literal for fit='expo';
          WHY Greek mathtext matches physics convention for decay constant;
          APPROVE '$\\tau$' substring present AND no 'decay';
          FAIL_MODE same as F2 for the 'decay' → '$\\tau$' entry.
    """

    def test_F1_linear_short_names(self, df_linear):
        """F1 — fit='linear' textbox renders 'p0'/'p1', not 'slope'/'intercept'."""
        fig, ax, stats = DFDraw(df_linear).profile('y:x', bins=20, fit='linear')
        FitVisualCheck(fig, stats, df_linear) \
            .check_textbox_uses_display_names_linear(ax) \
            .assert_clean()
        plt.close(fig)

    def test_F2_gauss_mu_mathtext(self, df_gauss):
        """F2 — fit='gauss' textbox contains '$\\mu$' literal for canonical 'center'."""
        fig, ax, stats = DFDraw(df_gauss).hist('x', bins=40, fit='gauss')
        FitVisualCheck(fig, stats, df_gauss) \
            .check_textbox_uses_mathtext_mu_for_gauss(ax) \
            .assert_clean()
        plt.close(fig)

    def test_F3_exponential_tau_mathtext(self, df_expo):
        """F3 — fit='expo' textbox contains '$\\tau$' literal for canonical 'decay'."""
        fig, ax, stats = DFDraw(df_expo).profile('y:x', bins=20, fit='expo')
        FitVisualCheck(fig, stats, df_expo) \
            .check_textbox_uses_mathtext_tau_for_exponential(ax) \
            .assert_clean()
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Phase 13.50 step 2 — Tests F4, F5 (precision keys)
# --------------------------------------------------------------------------- #

class TestPhase1350FitPrecision:
    """Phase 13.50 step 2: precision keys replace single fit.text_format.

    EXPECTED · WHY · APPROVE · FAIL_MODE per V-check:

      F4: EXPECTED default-style render produces value with ≤2 sig figs
          and error with ≤1 sig fig;
          WHY single .4g is wasteful in faceted figures; physics convention
          rounds error to 1 sf and values to match;
          APPROVE every parsed (value, error) pair has val_sf ≤ 2 AND
          err_sf ≤ 1;
          FAIL_MODE _style_get('fit.value_format'/'fit.error_format') not
          wired, or format() calls in _fit_render.py still use old single
          text_format key.

      F5: EXPECTED with set_style({'fit.precision_mode': 'physics'}),
          every parsed pair has identical decimal places in value and
          error (error rounded to 1 sf, value matched to error's place);
          WHY physics convention for reporting fit uncertainties;
          APPROVE _decimal_places(val) == _decimal_places(err) for every
          pair;
          FAIL_MODE _format_value_error_pair's physics branch not reached,
          or precision_mode not read from style.
    """

    def test_F4_default_precision(self, df_linear):
        """F4 — default render: value '.2g', error '.1g' for fit='linear'."""
        fig, ax, stats = DFDraw(df_linear).profile('y:x', bins=20, fit='linear')
        FitVisualCheck(fig, stats, df_linear) \
            .check_default_precision_value_2g_error_1g(ax) \
            .assert_clean()
        plt.close(fig)

    def test_F5_precision_mode_physics(self, df_linear):
        """F5 — precision_mode='physics' aligns value's decimal place to error's.

        Uses set_style for the global key. The per-call override
        (fit_textbox_kwargs={'precision_mode': 'physics'}) is also valid
        in step 3 onwards; F5 stays on the global form to keep the
        global-key path exercised. Restores default in finally to keep the
        rest of the gate uncontaminated."""
        from dfextensions.dfdraw.style import set_style
        set_style({'fit.precision_mode': 'physics'})
        try:
            fig, ax, stats = DFDraw(df_linear).profile('y:x', bins=20, fit='linear')
            FitVisualCheck(fig, stats, df_linear) \
                .check_precision_mode_physics_aligns_value_to_error(ax) \
                .assert_clean()
            plt.close(fig)
        finally:
            set_style({'fit.precision_mode': None})


# --------------------------------------------------------------------------- #
# Phase 13.50 step 3 — Tests F6, F16 (fit_textbox_kwargs extensions)
# --------------------------------------------------------------------------- #

class TestPhase1350FitTextboxKwargsExtensions:
    """Phase 13.50 step 3: fit_textbox_kwargs gets four new sub-keys
    (rename_params, value_format, error_format, precision_mode). F6 and F16
    are the visual-primitive locks; F16 is the per-call override for
    value_format. precision_mode + error_format per-call overrides are
    smoke-covered by the validation block in _fit_render.py (raise on bad
    type/value) and the existing F4/F5 style-default coverage.

    EXPECTED · WHY · APPROVE · FAIL_MODE per V-check:

      F6: EXPECTED textbox shows 'sigma_x' (user override) for fit='gauss',
          with no '$\\sigma$' (default) appearing;
          WHY users sometimes need ad-hoc labels (e.g. 'σ_x' vs 'σ_y' for
          asymmetric Gaussians) without editing the global _DISPLAY_NAMES;
          APPROVE 'sigma_x' substring present AND '$\\sigma$' absent;
          FAIL_MODE rename_params not in _allowed_sub_keys, or not threaded
          into _resolve_display_name() at the render site.

      F16: EXPECTED textbox values have >2 sig figs for fit='linear' with
           fit_textbox_kwargs={'value_format': '.4g'};
           WHY per-call precision override lets a user request more digits
           for a specific figure without globally widening style defaults;
           APPROVE max(count_sig_figs(val)) ≥ 3 across parsed pairs;
           FAIL_MODE value_format not in _allowed_sub_keys, or override
           layer doesn't take precedence over the style default at the
           value_format = _style_get(...) site.
    """

    def test_F6_rename_params_overrides_display_map(self, df_gauss):
        """F6 — rename_params override beats _DISPLAY_NAMES default."""
        fig, ax, stats = DFDraw(df_gauss).hist(
            'x', bins=40, fit='gauss',
            fit_textbox_kwargs={'rename_params': {'sigma': 'sigma_x'}},
        )
        FitVisualCheck(fig, stats, df_gauss) \
            .check_rename_params_overrides_display_map(ax) \
            .assert_clean()
        plt.close(fig)

    def test_F16_value_format_per_call_override(self, df_linear):
        """F16 — per-call value_format='.4g' grants more precision than '.2g' default."""
        fig, ax, stats = DFDraw(df_linear).profile(
            'y:x', bins=20, fit='linear',
            fit_textbox_kwargs={'value_format': '.4g'},
        )
        FitVisualCheck(fig, stats, df_linear) \
            .check_value_error_format_per_call_override(ax) \
            .assert_clean()
        plt.close(fig)


# --------------------------------------------------------------------------- #
# Phase 13.50 step 4 — legend polymorphic kwarg + show_legend alias
# Tests F7, F8, F9, F10 (visual_primitive) + F18 (invariance) +
# normalizer idempotency (invariance). All under LEGEND.modes feature row.
# --------------------------------------------------------------------------- #

class TestPhase1350LegendModes:
    """Phase 13.50 step 4 visual-primitive checks for the legend= polymorphic
    kwarg and the parallel show_legend= bool kwarg.

    EXPECTED · WHY · APPROVE · FAIL_MODE per V-check:

      F7  (legend='shared'): EXPECTED 1 fig.legend + 0 per-axes legends after
          dispatch on a 1×4 faceted profile;
          WHY consolidating legends is the headline feature of the new
          legend= polymorphism — without it the rest of the modes are
          academic;
          APPROVE legend_topology[0] == 1 AND legend_topology[1] == 0;
          FAIL_MODE _apply_legend_mode never reached, or 'shared' branch
          doesn't strip per-axes before fig.legend, or handles aren't
          captured before strip.

      F8  (legend='first'): EXPECTED axes[0] keeps its legend, axes[1:] are
          stripped;
          WHY 'first' is a common pattern for grouped facets where the
          legend repeats identical entries per panel and only the first
          is needed for context;
          APPROVE axes[0].get_legend() is not None AND all(axes[1:] are None);
          FAIL_MODE applier branch off-by-one, or wrong axes index, or
          per-axes legends not actually drawn by default before strip.

      F9  (legend=False): EXPECTED zero legends anywhere (fig-level AND
          per-axes);
          WHY the bool shortcut must work — users will pass False as the
          simplest 'no legend' invocation;
          APPROVE legend_topology[0] == 0 AND legend_topology[1] == 0;
          FAIL_MODE bool→'none' mapping broken in normalizer, or 'none'
          branch in applier doesn't iterate all fig.axes.

      F10 (legend={'mode':'shared','loc':'lower center','ncol':7}):
          EXPECTED dict kwargs reach fig.legend() — verified via
          fig.legends[0]._ncols == 7;
          WHY users need to override matplotlib defaults (loc, ncol) without
          dropping to a bare ax.legend() call; the dict form is the escape
          hatch and must forward correctly;
          APPROVE fig.legends has 1 entry AND _ncols == 7;
          FAIL_MODE _build_canonical drops unknown keys, or applier doesn't
          pass them through to fig.legend(), or matplotlib version uses
          a different private attribute name (P3-2 risk).

      F18 (show_legend ↔ legend behavioral equivalence): EXPECTED
          show_legend=X and legend=bool(X) produce identical legend_topology
          4-tuples (per P2-NEW-2: counts + label frozenset + loc string);
          WHY F18 is the LOAD-BEARING test for the architect's "back-compat
          parallel" framing — if these aren't equivalent, then show_legend=
          and legend= are not parallel surfaces, just two confusingly named
          kwargs;
          APPROVE topology(show_legend=True) == topology(legend=True) AND
          topology(show_legend=False) == topology(legend=False);
          FAIL_MODE normalizer precedence wrong, OR show_legend mapping
          drops a field (e.g. forgets to set frameon to default).
    """

    def _profile_faceted(self, df, **legend_kwargs):
        """Shared dispatch helper — keeps tests focused on legend assertions
        rather than the profile call boilerplate. Returns (fig, axes, stats)."""
        return DFDraw(df).profile(
            'y:x', bins=10, group_by='g', facet_by='sec', **legend_kwargs,
        )

    def test_F7_shared_one_fig_zero_per_axes(self, df_faceted):
        """F7 — legend='shared' consolidates to one fig.legend + zero per-axes."""
        fig, axes, stats = self._profile_faceted(df_faceted, legend='shared')
        FitVisualCheck(fig, stats, df_faceted) \
            .check_legend_shared_one_fig_level_zero_per_axes() \
            .assert_clean()
        plt.close(fig)

    def test_F8_first_only_axes_0_kept(self, df_faceted):
        """F8 — legend='first' keeps axes.flat[0]'s legend, strips others."""
        fig, axes, stats = self._profile_faceted(df_faceted, legend='first')
        FitVisualCheck(fig, stats, df_faceted) \
            .check_legend_first_only_axes_flat_zero() \
            .assert_clean()
        plt.close(fig)

    def test_F9_false_no_legends(self, df_faceted):
        """F9 — legend=False produces zero legends anywhere."""
        fig, axes, stats = self._profile_faceted(df_faceted, legend=False)
        FitVisualCheck(fig, stats, df_faceted) \
            .check_legend_false_no_legends_anywhere() \
            .assert_clean()
        plt.close(fig)

    def test_F10_dict_forwards_loc_and_ncol(self, df_faceted):
        """F10 — dict form: loc + ncol reach fig.legend() via the canonical merge."""
        fig, axes, stats = self._profile_faceted(
            df_faceted,
            legend={'mode': 'shared', 'loc': 'lower center', 'ncol': 7},
        )
        FitVisualCheck(fig, stats, df_faceted) \
            .check_legend_dict_forwards_loc_and_ncol(expected_ncol=7) \
            .assert_clean()
        plt.close(fig)

    def test_F18_show_legend_legend_behavioral_equivalence(self, df_faceted):
        """F18 — show_legend=X and legend=bool(X) produce identical legend
        topology. Two value pairs tested: True/True and False/False."""
        # True path
        fig_a, _, _ = self._profile_faceted(df_faceted, legend=True)
        topo_a = legend_topology(fig_a)
        plt.close(fig_a)

        fig_b, _, _ = self._profile_faceted(df_faceted, show_legend=True)
        topo_b = legend_topology(fig_b)
        plt.close(fig_b)

        # False path
        fig_c, _, _ = self._profile_faceted(df_faceted, legend=False)
        topo_c = legend_topology(fig_c)
        plt.close(fig_c)

        fig_d, _, _ = self._profile_faceted(df_faceted, show_legend=False)
        topo_d = legend_topology(fig_d)
        plt.close(fig_d)

        # Use a throwaway fig for the FitVisualCheck (it just needs a fig
        # handle; the equivalence assertion is on the captured topologies).
        fig_dummy, _, _ = self._profile_faceted(df_faceted)
        FitVisualCheck(fig_dummy, None, df_faceted) \
            .check_show_legend_legend_behavioral_equivalence(topo_a, topo_b) \
            .check_show_legend_legend_behavioral_equivalence(topo_c, topo_d) \
            .assert_clean()
        plt.close(fig_dummy)


class TestPhase1350LegendNormalizerInvariance:
    """Phase 13.50 step 4 invariance layer: ``_normalize_legend_spec`` is a
    pure function and must be idempotent over all accepted input forms.

    Why invariance, not visual_primitive: this test exercises NO fig/axes —
    it's a pure function self-consistency check. The 4-tuple legend_topology
    helper is not invoked. Per Coder QRC layer taxonomy and v2.5 §3.6,
    pure-function tests go on the invariance layer regardless of which
    feature they support.

    Locks: f(f(x)) == f(x) for every accepted input form, including the
    show_legend → legend mapping path, the dict-merge path, and the
    legend-wins-when-both-set precedence rule.
    """

    def test_normalize_legend_spec_idempotent(self):
        """Pure normalizer self-consistency: re-normalizing a canonical dict
        produces the same canonical dict, for every input shape."""
        # Each row: (legend, show_legend) input. None pair excluded because
        # the function returns None for that, and re-feeding None yields
        # None — trivially idempotent but doesn't exercise the merge logic.
        accepted_inputs = [
            (True,  None),
            (False, None),
            ('all',    None),
            ('none',   None),
            ('shared', None),
            ('first',  None),
            ({'mode': 'all'},                                  None),
            ({'mode': 'shared', 'loc': 'lower center'},        None),
            ({'mode': 'shared', 'loc': 'upper right',
              'ncol': 3, 'frameon': False, 'fontsize': 8,
              'title': 'Groups', 'bbox_to_anchor': (0.5, 0.0)}, None),
            # show_legend → legend mapping path
            (None, True),
            (None, False),
            # precedence: legend= wins when both set
            (True,  False),
            (False, True),
            ('shared', True),
        ]
        for legend, show_legend in accepted_inputs:
            once  = _normalize_legend_spec(legend, show_legend)
            # Re-feed the canonical dict back through the legend= slot.
            twice = _normalize_legend_spec(once, None)
            assert once == twice, (
                f"non-idempotent for (legend={legend!r}, show_legend={show_legend!r}): "
                f"once={once!r}  twice={twice!r}"
            )
