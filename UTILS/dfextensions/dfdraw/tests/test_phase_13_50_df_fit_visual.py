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
