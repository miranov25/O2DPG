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
