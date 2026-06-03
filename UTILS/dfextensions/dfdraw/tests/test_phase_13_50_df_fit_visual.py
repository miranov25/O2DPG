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


# Phase 13.50 step 6 — table_shape helper for orientation tests.
# matplotlib's ax.table() produces a Table whose cells are addressable via
# table.get_celld(): a dict {(row_idx, col_idx): Cell}. Header row is at
# row_idx == 0 (when colLabels= is set) and data rows are 1..N. Cell
# columns are 0..M.
def table_shape(ax):
    """Return ``(n_rows, n_cols)`` of the first matplotlib Table on ``ax``,
    counting the header row when colLabels was set.

    Returns ``(0, 0)`` if no table is present (e.g. early-exit when no rows
    were produced by ``_flatten_to_rows``).
    """
    if not ax.tables:
        return (0, 0)
    cells = ax.tables[0].get_celld()
    if not cells:
        return (0, 0)
    n_rows = max(r for (r, c) in cells.keys()) + 1
    n_cols = max(c for (r, c) in cells.keys()) + 1
    return (n_rows, n_cols)


# Phase 13.50.DF step 7e — table_cells_keyed: keyed cell extraction helper.
# Per v2.4 P2-NEW-1 (Claude36 ADF panel finding), the F17 cross-variant
# equivalence check needs a key-based comparison so that a placement bug
# that permutes which fit's value goes where is detected. A flat set or
# count would be permutation-blind.
#
# Phase 13.50.DF step 7e FIX2 (post-step-7 F17 failure): returns a SET of
# frozensets — one frozenset per data row, holding (col_label, cell_text)
# pairs. Set-equality is collision-safe AND order-independent: identical
# row contents collapse to one (semantic dedupe, fine for placement
# equivalence); per-panel inset union is just ``set_a | set_b | ...``.
# The pre-fix dict-keyed-by-(col0_value, col_label) approach lost rows
# whose col-0 value collided (e.g., multiple fits with same 'group' value).
def table_cells_keyed(ax):
    """Return ``set[frozenset[tuple[col_label, cell_text]]]`` for the first
    matplotlib Table on ``ax``.

    Each frozenset is one data row's (column_label, cell_text) pairs.
    Set-based equality is the comparator: ``A == B`` iff both tables hold
    the same data rows (regardless of row order, and tolerant of single-row
    duplicates which are semantically a no-op for placement equivalence).

    For per-panel insets (placement='subfigure'), the cross-variant union
    is ``set.union(*per_panel_sets)``.

    Returns ``set()`` if no table is present or it has no data rows.
    """
    if not ax.tables:
        return set()
    cells = ax.tables[0].get_celld()
    if not cells:
        return set()
    # Build col_labels from row 0 (matplotlib convention: col headers at r=0,
    # data rows at r=1, r=2, ...).
    col_labels = {}
    for (r, c), cell in cells.items():
        if r == 0:
            col_labels[c] = cell.get_text().get_text().strip()
    if not col_labels:
        return set()
    max_row = max(r for (r, c) in cells.keys())
    result = set()
    for data_r in range(1, max_row + 1):
        row_pairs = []
        for c, label in col_labels.items():
            cell = cells.get((data_r, c))
            if cell is None:
                continue
            text = cell.get_text().get_text().strip()
            row_pairs.append((label, text))
        if row_pairs:
            result.add(frozenset(row_pairs))
    return result


# Phase 13.50 step 5 — placement_topology helper (per v2.4 P2-NEW-1: keyed
# dict catches permutation bugs that a count-tuple misses; if placement='pad'
# and placement='subfigure' get swapped, the topology dict's two values
# differ in shape).
def placement_topology(fig, stats):
    """Return a keyed dict describing how summary_fit was placed.

    Keys
    ----
    - 'placement'        : str — the placement label the renderer reports
                           ('figure' is inferred when no slot was used).
    - 'in_main_fig_axes' : bool — is the host axes one of ``fig.axes``?
                           True only for placement='pad'.
    - 'in_subfigure'     : bool — is the host axes inside the SubFigure
                           stashed on ``fig._dfdraw_summary_fit_subfigure``?
                           True only for placement='subfigure'.
    - 'separate_figure'  : bool — does ``stats['summary_fit']`` contain a
                           ``matplotlib.figure.Figure`` distinct from ``fig``?
                           True only for placement='figure'.
    - 'host_kind'        : str — what kind of object hosts the rendered
                           content: 'axes' | 'figure' | 'absent'.

    Permutation safety: for the three accepted placements (figure, subfigure,
    pad), the four-bool combinations are all distinct, so a placement→topology
    mapping bug (e.g. 'pad' renders to a SubFigure or 'figure' renders to the
    main fig) shows up as a topology mismatch.

    FIX1 of step 5: ``fig.subfigures`` is a CREATION METHOD on
    matplotlib.figure.Figure (used as ``fig.subfigures(nrows, ncols, ...)``),
    NOT an iterable property — iterating it raises ``TypeError: 'method'
    object is not iterable``. SubFigure detection here uses the renderer's
    stash on ``fig._dfdraw_summary_fit_subfigure``, which is set whenever
    ``render_summary_fit_into_slot`` materializes a SubFigure.
    """
    import matplotlib.figure as _mpl_fig
    import matplotlib.axes as _mpl_axes

    sf = stats.get('summary_fit') if isinstance(stats, dict) else None
    if not sf:
        return {
            'placement':        'absent',
            'in_main_fig_axes': False,
            'in_subfigure':     False,
            'separate_figure':  False,
            'host_kind':        'absent',
        }

    placement = sf.get('placement', 'figure')
    in_main_fig_axes = False
    in_subfigure = False
    separate_figure = False
    host_kind = 'absent'

    # The 'table' key holds the rendered content. In Phase 13.43 'figure'
    # placement it's a separate matplotlib Figure; in 'pad'/'subfigure' it's
    # the host Axes inside the main fig.
    host = sf.get('table')
    if isinstance(host, _mpl_fig.Figure):
        # 'figure' placement path: sf['table'] is a separate Figure.
        if host is not fig:
            separate_figure = True
            host_kind = 'figure'
    elif isinstance(host, _mpl_axes.Axes):
        # 'pad' or 'subfigure' placement: sf['table'] is the host axes.
        # IMPORTANT: matplotlib aggregates SubFigure-hosted axes into the
        # PARENT fig.axes list as well as the SubFigure's own axes list.
        # So we MUST check the stashed SubFigure first; otherwise the
        # subfigure path always misroutes to in_main_fig_axes=True and
        # placement='subfigure' looks indistinguishable from 'pad'.
        stashed_subfig = getattr(fig, '_dfdraw_summary_fit_subfigure', None)
        if stashed_subfig is not None and host in stashed_subfig.axes:
            in_subfigure = True
            host_kind = 'axes'
        elif host in fig.axes:
            in_main_fig_axes = True
            host_kind = 'axes'
    else:
        # Fallback: Phase 13.43 'figure' kind may sit under sf['figure']
        # instead of sf['table'] when only 'figure' kind was requested.
        for k in ('table', 'figure'):
            f = sf.get(k)
            if isinstance(f, _mpl_fig.Figure) and f is not fig:
                separate_figure = True
                host_kind = 'figure'
                break

    return {
        'placement':        placement,
        'in_main_fig_axes': in_main_fig_axes,
        'in_subfigure':     in_subfigure,
        'separate_figure':  separate_figure,
        'host_kind':        host_kind,
    }


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


# --------------------------------------------------------------------------- #
# Phase 13.50 step 5 — summary_fit.placement axis
# Tests F11, F12, F13 (visual_primitive) + F17 (invariance — placement_topology
# keyed dict catches mode-swap bugs per v2.4 P2-NEW-1).
# --------------------------------------------------------------------------- #

class TestPhase1350SummaryFitPlacement:
    """Phase 13.50 step 5 visual-primitive checks for ``summary_fit`` placement
    axis (where the summary content is rendered: separate figure / subfigure
    inside the main fig / pad-slot inside the main fig).

    EXPECTED · WHY · APPROVE · FAIL_MODE per V-check:

      F11 (placement='figure', default): EXPECTED ``stats['summary_fit']``
          contains a separate Figure object distinct from the main fig;
          ``fig._dfdraw_summary_fit_slot`` is absent;
          WHY default behavior MUST be preserved exactly — Phase 13.43
          callers rely on the standalone-Figure semantics;
          APPROVE topology['separate_figure'] True AND
          topology['in_main_fig_axes'] False AND
          getattr(fig, '_dfdraw_summary_fit_slot', None) is None;
          FAIL_MODE normalizer default drifted to 'pad'/'subfigure', or
          dispatcher unconditionally pre-plans the slot.

      F12 (placement='subfigure'): EXPECTED a SubFigure exists in
          ``fig.subfigures`` and the host axes is inside it;
          WHY subfigure placement is the integrated-in-main-fig path —
          must use matplotlib's SubFigure API, not just a regular axes;
          APPROVE len(fig.subfigures) >= 1 AND
          topology['in_subfigure'] True AND topology['host_kind']=='axes';
          FAIL_MODE renderer used add_subplot instead of add_subfigure;
          slot SubplotSpec not stashed; or placement got downgraded
          silently to 'figure' or 'pad'.

      F13 (placement='pad'): EXPECTED the host axes is an extra axes in
          ``fig.axes`` (beyond the facet count);
          WHY pad placement uses the simpler GridSpec extra-row strategy
          and the host must be a normal axes in the main fig;
          APPROVE topology['in_main_fig_axes'] True AND
          len(fig.axes) == n_facet_axes + 1;
          FAIL_MODE renderer accidentally used SubFigure (wrong API for
          'pad'), or pre-planning failed to add the extra row.

      F17 (placement_topology keyed dict invariance per v2.4 P2-NEW-1):
          EXPECTED running all 3 placements and collecting their topologies
          into a keyed dict yields three DISTINCT topology dicts;
          WHY a count-tuple alone (the early F17 draft) would have missed
          a swap bug where 'pad' and 'subfigure' renderers got crossed,
          since both produce one host axes — but the keyed dict makes
          ``in_main_fig_axes`` vs ``in_subfigure`` distinguish them
          unambiguously;
          APPROVE topo['pad'] != topo['subfigure'] != topo['figure'];
          FAIL_MODE any two placements collapse to the same topology dict
          (mode collapse / swap / drift).
    """

    def _profile_faceted(self, df, **profile_kwargs):
        """Shared dispatch helper. ``facet_by='sec'`` triggers
        ``_dispatch_faceted_render`` (the path that pre-plans the GridSpec
        slot for placement='pad'/'subfigure'); ``group_by='g'`` adds within-
        panel grouping for richer stats['fit'] rows. Both faceting params
        are required — group_by ALONE does not invoke the dispatcher (it
        produces a single-axes grouped plot), so the slot is never
        pre-planned and placement='pad'/'subfigure' raises."""
        return DFDraw(df).profile(
            'y:x', bins=10, group_by='g', facet_by='sec',
            fit='linear', **profile_kwargs,
        )

    def test_F11_placement_figure_default_unchanged(self, df_faceted):
        """F11 — placement='figure' (default) preserves Phase 13.43 standalone
        Figure semantics. No slot pre-planning on the main fig."""
        fig, axes, stats = self._profile_faceted(
            df_faceted,
            summary_fit={'kind': 'table'},  # placement defaults to 'figure'
        )
        topology = placement_topology(fig, stats)
        # Topology must reflect separate-figure semantics.
        assert topology['placement'] == 'figure', (
            f"expected placement='figure' (default), got {topology['placement']!r}")
        assert topology['separate_figure'] is True, (
            f"expected separate_figure=True for placement='figure', "
            f"got topology={topology!r}")
        assert topology['in_main_fig_axes'] is False, (
            f"placement='figure' must not host into main fig.axes; "
            f"topology={topology!r}")
        assert topology['in_subfigure'] is False, (
            f"placement='figure' must not host into a SubFigure; "
            f"topology={topology!r}")
        # Slot attribute must be absent for default placement.
        assert getattr(fig, '_dfdraw_summary_fit_slot', None) is None, (
            "placement='figure' must not pre-plan a slot")
        plt.close(fig)
        # Close the separate Figure too.
        for k in ('table', 'figure'):
            sub = stats.get('summary_fit', {}).get(k)
            if hasattr(sub, 'number'):
                plt.close(sub)

    def test_F12_placement_subfigure_per_panel_inset_table(self, df_faceted):
        """F12 — placement='subfigure' attaches per-panel inset_axes() to each
        visible facet panel, each hosting only that panel's fits.

        v2.5 spec §3.5 + P2-NEW-3 (Claude36 ADF panel finding folded into
        v2.4): subfigure insets are per-panel slices, NOT redundant full-table
        copies on every panel. The architect's stated rationale was to save
        space in 9-panel GB-fit — 9 small per-panel tables, each with just
        the 1-3 fits relevant to that panel.

        Phase 13.50.DF step 7d spec-conformance — previous (step-5) ship used a
        single fig.add_subfigure() with the full table, which contradicted the
        spec. This test locks the corrected per-panel semantics.

        EXPECTED · WHY · APPROVE · FAIL_MODE:
          EXPECTED ``stats['summary_fit']['placement'] == 'subfigure'`` with an
                   ``'insets'`` list whose length equals the number of visible
                   facet panels that produced fits, and a ``'per_panel_keyed'``
                   dict mapping facet keys to inset axes;
          WHY this is the cross-variant behavioural contract — every visible
              panel gets exactly one inset, no more, no less;
          APPROVE len(insets) > 0 AND insets ⊆ axes whose parent is a facet
                  axes (i.e., inset.get_axes_locator() / containing axes is
                  one of the visible facet axes);
          FAIL_MODE single-SubFigure regression (one inset for the whole
                    figure), or full-table copies in each inset (each inset
                    has the same row count as the figure-placement table).
        """
        fig, axes, stats = self._profile_faceted(
            df_faceted,
            summary_fit={'kind': 'table', 'placement': 'subfigure'},
        )
        sf = stats.get('summary_fit', {})
        assert sf.get('placement') == 'subfigure', (
            f"expected placement='subfigure' in stats; got {sf!r}")
        insets = sf.get('insets')
        assert isinstance(insets, list) and len(insets) > 0, (
            f"placement='subfigure' must produce a non-empty 'insets' list "
            f"(per-panel inset_axes); got {insets!r}")
        per_panel = sf.get('per_panel_keyed')
        assert isinstance(per_panel, dict) and len(per_panel) == len(insets), (
            f"per_panel_keyed dict length must equal insets list length "
            f"({len(insets)}); got per_panel_keyed={per_panel!r}")
        # Each inset must be hosted by ONE of the visible facet axes (the
        # axes that carry the dispatcher-stashed _dfdraw_facet_key marker).
        facet_axes = [ax for ax in fig.axes
                      if ax.get_visible() and hasattr(ax, '_dfdraw_facet_key')]
        facet_axes_set = set(id(ax) for ax in facet_axes)
        for inset in insets:
            # inset's host axes is its parent in matplotlib's container
            # hierarchy; identify by walking _axes_hosting OR by spatial
            # containment (any visible facet axes whose bbox contains the
            # inset's bbox).
            inset_pos = inset.get_position()
            host_id = None
            for fa in facet_axes:
                fa_pos = fa.get_position()
                if (fa_pos.x0 <= inset_pos.x0 and fa_pos.y0 <= inset_pos.y0
                        and fa_pos.x1 >= inset_pos.x1
                        and fa_pos.y1 >= inset_pos.y1):
                    host_id = id(fa)
                    break
            assert host_id is not None and host_id in facet_axes_set, (
                f"inset axes at position {inset_pos} is not contained by any "
                f"visible facet axes; per-panel slice semantic violated")
        # NO SubFigure should have been created — step 7d eliminates the
        # single-SubFigure approach. The step-5 stash attribute must be
        # absent (or None) on the fig.
        assert getattr(fig, '_dfdraw_summary_fit_subfigure', None) is None, (
            "placement='subfigure' must NOT create a fig.add_subfigure() "
            "container (step-5 single-SubFigure regression); use per-panel "
            "inset_axes() per v2.5 §3.5")
        plt.close(fig)

    def test_F13_placement_pad_uses_extra_row_axes(self, df_faceted):
        """F13 — placement='pad' adds an extra axes to the main fig (extra
        GridSpec row, height_ratio < 1)."""
        # Baseline: same dispatch without summary_fit → just facet axes.
        fig_baseline, axes_baseline, _ = self._profile_faceted(df_faceted)
        n_facet_axes = len(fig_baseline.axes)
        plt.close(fig_baseline)

        fig, axes, stats = self._profile_faceted(
            df_faceted,
            summary_fit={'kind': 'table', 'placement': 'pad'},
        )
        topology = placement_topology(fig, stats)
        assert topology['placement'] == 'pad', (
            f"expected placement='pad', got {topology['placement']!r}")
        assert topology['in_main_fig_axes'] is True, (
            f"host axes must be in fig.axes for 'pad'; topology={topology!r}")
        assert topology['in_subfigure'] is False, (
            f"'pad' must NOT use a SubFigure; topology={topology!r}")
        # The pad placement adds exactly one axes (the slot) to the facet count.
        assert len(fig.axes) == n_facet_axes + 1, (
            f"placement='pad' should add exactly 1 axes to the baseline facet "
            f"count ({n_facet_axes}); got len(fig.axes)={len(fig.axes)}")
        plt.close(fig)

    def test_F17_placement_variants_produce_equivalent_tables(self, df_faceted):
        """F17 — same data rendered via placement='figure', 'pad', 'subfigure'
        produces equivalent keyed cell dicts (cross-variant invariance).

        v2.5 §3.8 + v2.4 P2-NEW-1 (Claude36 ADF panel): the cross-variant
        equivalence test is the headline behavioural invariant — if the user
        switches placement, the SAME data must show up, just in a different
        container. Keyed comparison (dict[(row_label, col_label), cell_text])
        is permutation-safe; a placement bug that swaps which fit's value
        ends up where would show up as a key mismatch.

        Per P2-NEW-3 subfigure semantics: insets are per-panel slices, so
        union(table_cells_keyed(inset) for inset in subfigure_insets) must
        equal the figure/pad keyed dict (each inset contributes its panel's
        share; the union covers the full data set).

        Phase 13.50.DF step 7e spec-conformance — the step-5 ship implemented
        F17 as topology DISTINCTNESS (opposite invariant). This rewrite
        ships the v2.5-spec'd cross-variant EQUIVALENCE check.

        EXPECTED · WHY · APPROVE · FAIL_MODE:
          EXPECTED ``table_cells_keyed(figure_table) == table_cells_keyed(
                   pad_table) == union(table_cells_keyed(i) for i in
                   subfigure_insets)`` for the SAME (group_by, facet_by, fit)
                   call;
          WHY user-perceived equivalence is the spec promise — switching
              placement is a layout choice, not a data transform;
          APPROVE all three keyed dicts equal as Python dicts (header rows
                  excluded; cell text whitespace-stripped);
          FAIL_MODE renderer permutes rows / drops cells / changes formatting
                    between placements — typically a regression where one
                    placement uses a stale spec branch.
        """
        # 'figure' placement — table is a separate matplotlib Figure.
        fig_fig, _, stats_fig = self._profile_faceted(
            df_faceted,
            summary_fit={'kind': 'table', 'placement': 'figure'},
        )
        sep_fig = stats_fig['summary_fit'].get('table')
        assert hasattr(sep_fig, 'axes'), (
            f"placement='figure' should produce a separate Figure with a "
            f"table axes; got {sep_fig!r}")
        figure_keyed = table_cells_keyed(sep_fig.axes[0])
        plt.close(sep_fig)
        plt.close(fig_fig)

        # 'pad' placement — table is an axes inside the main fig.
        fig_pad, _, stats_pad = self._profile_faceted(
            df_faceted,
            summary_fit={'kind': 'table', 'placement': 'pad'},
        )
        pad_host = stats_pad['summary_fit']['table']
        pad_keyed = table_cells_keyed(pad_host)
        plt.close(fig_pad)

        # 'subfigure' placement — per-panel insets; union of their keyed
        # sets equals the figure/pad keyed set (P2-NEW-3 per-panel
        # slicing semantic). Each inset contributes the rows for its
        # panel; the union is the full data set.
        fig_sub, _, stats_sub = self._profile_faceted(
            df_faceted,
            summary_fit={'kind': 'table', 'placement': 'subfigure'},
        )
        insets = stats_sub['summary_fit'].get('insets', [])
        subfig_keyed_union = set()
        for inset in insets:
            subfig_keyed_union |= table_cells_keyed(inset)
        plt.close(fig_sub)

        # Invariance: all three keyed sets must agree.
        assert figure_keyed, (
            f"placement='figure' keyed set is empty — fixture didn't "
            f"produce a populated table")
        assert figure_keyed == pad_keyed, (
            f"placement='figure' vs 'pad' keyed sets diverge:\n"
            f"  |figure|={len(figure_keyed)}, |pad|={len(pad_keyed)}\n"
            f"  rows in figure not in pad: "
            f"{len(figure_keyed - pad_keyed)} (first: "
            f"{next(iter(figure_keyed - pad_keyed), None)})\n"
            f"  rows in pad not in figure: "
            f"{len(pad_keyed - figure_keyed)} (first: "
            f"{next(iter(pad_keyed - figure_keyed), None)})")
        assert figure_keyed == subfig_keyed_union, (
            f"placement='figure' vs 'subfigure' (union of per-panel insets) "
            f"keyed sets diverge:\n"
            f"  |figure|={len(figure_keyed)}, "
            f"|subfig_union|={len(subfig_keyed_union)}, "
            f"insets count: {len(insets)}\n"
            f"  rows in figure not in subfig: "
            f"{len(figure_keyed - subfig_keyed_union)} (first: "
            f"{next(iter(figure_keyed - subfig_keyed_union), None)})\n"
            f"  rows in subfig not in figure: "
            f"{len(subfig_keyed_union - figure_keyed)} (first: "
            f"{next(iter(subfig_keyed_union - figure_keyed), None)})")


# --------------------------------------------------------------------------- #
# Phase 13.50 step 6 — summary_fit.orientation axis
# Tests F14 (default 'row' locks shape), F15 ('column' transposes shape).
# Both visual_primitive layer.
# --------------------------------------------------------------------------- #

class TestPhase1350SummaryFitOrientation:
    """Phase 13.50 step 6 visual-primitive checks for the ``summary_fit``
    orientation axis (table layout direction).

    EXPECTED · WHY · APPROVE · FAIL_MODE per V-check:

      F14 (orientation default 'row'): EXPECTED the in-slot table
          renders with rows=fits, columns=id_keys+params (Phase 13.43
          layout shape preserved);
          WHY default MUST be backwards compatible — users have seen the
          row-oriented table since Phase 13.43; orientation='row'
          (default) must produce identical row/column counts to a call
          with NO orientation kwarg;
          APPROVE table_shape(no-kwarg) == table_shape(orientation='row');
          FAIL_MODE normalizer default drifted to 'column', or the
          row-branch logic accidentally transposes.

      F15 (orientation='column'): EXPECTED the transposed table shape
          where original DATA COLUMNS become DATA ROWS (each prefixed by
          a label cell containing the original column's header) and
          original DATA ROWS become DATA COLUMNS (prefixed by the new
          "fit_0", "fit_1", ... header row);
          WHY transposing is the headline feature: when params > fits the
          row layout becomes uncomfortably wide; column fixes
          this. Must verify a TRUE transpose, not e.g. a no-op or a
          partial rearrangement;
          APPROVE shape_col == (1 + shape_row[1], shape_row[0])
          — accounts for the label column inserted in column orientation
          (carrying the original column headers as row labels);
          FAIL_MODE transpose logic skipped (column falls through to
          row renderer), or wrong axis transposed (e.g. header
          rotated but data not), or off-by-one in row/col counts.
    """

    def _profile_with_orientation(self, df, orientation_kwarg):
        """Shared dispatch helper. placement='pad' is used so the in-slot
        renderer (which honors orientation in step 6) fires. group_by + facet_by
        are required for the dispatcher (same constraint as step 5)."""
        sf_spec = {'kind': 'table', 'placement': 'pad'}
        if orientation_kwarg is not None:
            sf_spec['orientation'] = orientation_kwarg
        return DFDraw(df).profile(
            'y:x', bins=10, group_by='g', facet_by='sec',
            fit='linear', summary_fit=sf_spec,
        )

    def test_F14_orientation_row_default_unchanged(self, df_faceted):
        """F14 — default and explicit 'row' produce identical table shape."""
        fig_default, _, stats_default = self._profile_with_orientation(
            df_faceted, orientation_kwarg=None)
        sf_default = stats_default.get('summary_fit', {})
        host_default = sf_default.get('table')
        assert host_default is not None, (
            f"placement='pad' should render a table host axes; "
            f"stats['summary_fit']={sf_default!r}")
        shape_default = table_shape(host_default)
        plt.close(fig_default)

        fig_explicit, _, stats_explicit = self._profile_with_orientation(
            df_faceted, orientation_kwarg='row')
        host_explicit = stats_explicit['summary_fit']['table']
        shape_explicit = table_shape(host_explicit)
        plt.close(fig_explicit)

        assert shape_default == shape_explicit, (
            f"orientation default and explicit 'row' must produce "
            f"identical table shapes; default={shape_default}, "
            f"explicit={shape_explicit}")
        # And: it must be a non-trivial table — at least 1 data row + 1 column
        # (otherwise we're not actually exercising the renderer).
        assert shape_default[0] >= 2 and shape_default[1] >= 2, (
            f"degenerate table shape {shape_default} — fixture didn't produce "
            f"enough fit rows for a meaningful orientation test")

    def test_F15_orientation_column_transposes_shape(self, df_faceted):
        """F15 — orientation='column' produces a TRUE transpose of the
        row-oriented table.

        Shape relationship (accounts for impl detail: original column headers
        become a leading label column in column mode):
            shape_col == (1 + shape_row[1], shape_row[0])
        Validated by /home/claude/p1350 smoke test with controlled rows:
            R=2 data rows × N=3 cols  →  shape_row=(3,3), shape_col=(4,3)
                                    →  1+3 == 4 ✓  AND  3 == 3 ✓
        """
        fig_h, _, stats_h = self._profile_with_orientation(
            df_faceted, orientation_kwarg='row')
        host_h = stats_h['summary_fit']['table']
        shape_h = table_shape(host_h)
        plt.close(fig_h)

        fig_v, _, stats_v = self._profile_with_orientation(
            df_faceted, orientation_kwarg='column')
        host_v = stats_v['summary_fit']['table']
        shape_v = table_shape(host_v)
        plt.close(fig_v)

        expected_v = (1 + shape_h[1], shape_h[0])
        assert shape_v == expected_v, (
            f"orientation='column' should produce shape {expected_v} "
            f"(transpose of row {shape_h} with +1 row for the "
            f"label-column header); got {shape_v}")
        # Sanity: shapes must differ — otherwise transpose was a no-op
        # (could happen on a degenerate square table; F14 already locks
        # min shape ≥ (2,2) so this is a defensive belt-check).
        assert shape_v != shape_h, (
            f"orientation='column' produced same shape as 'row' "
            f"({shape_h}); transpose may have been a no-op (fixture must "
            f"produce a non-square table for F15 to be discriminating)")
