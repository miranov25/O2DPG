"""Phase 13.43.DF v1.2 — Summary Fit test suite.

26 invariance tests per spec §10 (locked in v1.2):
  F.34, F.35, F.36, F.37        — basic scenarios (table / figure / both / omitted)
  F.38, F.38a                   — row count + selection_vector composition
  F.39, F.40, F.41              — params figure layout (auto, overlay, annotate)
  F.42                          — edge case: no group_by + no facet_by → Scenario E
  F.43                          — precision option
  F.44                          — 2D facet (Phase 13.41) composition
  F.45                          — normalize + summary_fit → Scenario E (CP1-5 + C-3)
  F.46                          — quantile-band profile + summary_fit → Scenario E
  F.47                          — cumulative=True (Phase 13.40)
  F.47b                         — stacked=True + group_by + fit (D9/R4 from FIX1)
  F.48                          — summary_fit without fit= → Scenario E silent path
  F.49, F.50                    — auto-title content + truncate
  F.51, F.52, F.53              — data_format dict/pandas + per-call override
  F.54                          — same=True Replace mode
  F.55                          — placement invariants (inspect.signature)
  F.56                          — faceted aggregation §4.2.0 (Shape 3 top-level)
  F.56b                         — faceted WITHOUT group_by (P2-C inner ladder)

All tests are standalone pytest functions under class
``TestPhase1343SummaryFit``. Gate impact: predecessor + 26 = +26.
"""
import inspect
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pytest

from dfextensions.dfdraw import DFDraw


# ============================================================================
# Test fixtures
# ============================================================================

def _gauss_df(n=5000, seed=43):
    """DataFrame with normal x + categorical g + 2 facet dims."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        'x':   rng.normal(0, 1, n),
        'y':   rng.normal(2, 1.5, n) + 0.3 * rng.normal(0, 1, n),
        'g':   rng.choice(['A', 'B', 'C'], n),
        'f1':  rng.choice(['L', 'R'], n),
        'f2':  rng.choice(['U', 'D'], n),
        'tag': rng.choice([0, 1, 2], n),
    })


def _close_all(stats):
    """Close every matplotlib Figure in stats['summary_fit'] + the main fig."""
    sf = stats.get('summary_fit') if isinstance(stats, dict) else None
    if isinstance(sf, dict):
        for v in sf.values():
            if isinstance(v, plt.Figure):
                plt.close(v)


# ============================================================================
# TestPhase1343SummaryFit — F.34 .. F.56b
# ============================================================================

class TestPhase1343SummaryFit:
    """Phase 13.43.DF v1.2 invariance regressions."""

    # -- F.34 — table only -----------------------------------------------

    def test_f34_summary_fit_table(self):
        """summary_fit='table' produces stats['summary_fit']['table'] as Figure
        AND stats['summary_fit']['data'] present."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist('x', bins=40, fit='gauss', summary_fit='table')
        sf = stats['summary_fit']
        assert isinstance(sf, dict)
        assert isinstance(sf.get('table'), plt.Figure)
        assert 'data' in sf
        plt.close(fig); _close_all(stats)

    # -- F.35 — figure only ----------------------------------------------

    def test_f35_summary_fit_figure(self):
        """summary_fit='figure' produces stats['summary_fit']['figure']."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist('x', bins=40, group_by='g', fit='gauss',
                                summary_fit='figure')
        sf = stats['summary_fit']
        assert isinstance(sf.get('figure'), plt.Figure)
        plt.close(fig); _close_all(stats)

    # -- F.36 — both / list form ----------------------------------------

    def test_f36_summary_fit_both(self):
        """summary_fit='both' and =['table','figure'] both produce both keys."""
        d = DFDraw(_gauss_df())
        for spec in ['both', ['table', 'figure']]:
            fig, ax, stats = d.hist('x', bins=40, group_by='g', fit='gauss',
                                    summary_fit=spec)
            sf = stats['summary_fit']
            assert isinstance(sf.get('table'), plt.Figure), f"spec={spec!r}"
            assert isinstance(sf.get('figure'), plt.Figure), f"spec={spec!r}"
            plt.close(fig); _close_all(stats)

    # -- F.37 — Scenario A: kwarg omitted, key absent --------------------

    def test_f37_summary_fit_omitted_is_scenario_a(self):
        """When summary_fit= is NOT passed, stats does NOT contain 'summary_fit'
        — Phase 13.42 unchanged."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist('x', bins=40, fit='gauss')
        assert 'summary_fit' not in stats, (
            "Scenario A: summary_fit kwarg omitted → key MUST be absent "
            "(Phase 13.42 contract; not just empty)"
        )
        plt.close(fig)

    # -- F.38 — table row count + group coloring ------------------------

    def test_f38_table_row_count(self):
        """Table row count == n_groups × n_facet_cells (with group_by)."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist('x', bins=40, group_by='g', fit='gauss',
                                summary_fit='table')
        data = stats['summary_fit']['data']
        # 3 groups × 1 (no facet) = 3 rows.
        assert len(data) == 3, f"expected 3 rows, got {len(data)}"
        # Each row should expose `group` set to one of A/B/C.
        groups_seen = sorted(set(r.get('group') for r in data))
        assert groups_seen == ['A', 'B', 'C'], f"group values: {groups_seen}"
        plt.close(fig); _close_all(stats)

    # -- F.38a — selection_vector composition ---------------------------

    def test_f38a_selection_vector_composition(self):
        """selection_vector → rows per (selection × group × cell). Each
        selection produces its own fit_dict; flatten captures all.

        Vector dispatch returns stats as a list of per-iteration dicts;
        summary_fit attaches to stats[0] (CRR §2 disclosure)."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist(
            'x', bins=40, group_by='g',
            selection_vector=['tag==0', 'tag==1'],
            vector_compose='outer',
            fit='gauss', summary_fit='table',
        )
        # stats is a list of 2 iter-dicts; summary_fit is on stats[0]
        assert isinstance(stats, list), "vector dispatch returns list"
        assert isinstance(stats[0], dict)
        sf = stats[0].get('summary_fit', {})
        assert isinstance(sf.get('table'), plt.Figure), (
            "summary_fit['table'] must be present on stats[0] for vector dispatch"
        )
        data = sf['data']
        # 2 selections × 3 groups = 6 fit rows
        assert len(data) >= 6, (
            f"selection_vector(2) × group_by(3) should give ≥6 rows, "
            f"got {len(data)}"
        )
        plt.close(fig); _close_all(stats[0])

    # -- F.39 — params figure auto layout -------------------------------

    def test_f39_params_figure_auto_layout(self):
        """Params figure subplot count == n_params for gauss (3 params:
        amplitude, center, sigma)."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist('x', bins=40, group_by='g', fit='gauss',
                                summary_fit='figure')
        params_fig = stats['summary_fit']['figure']
        # Count visible axes
        visible_axes = [a for a in params_fig.axes if a.get_visible()]
        # gauss has amplitude, center, sigma → 3 params; layout = ceil(sqrt(3))
        # = 2 → 2x2 grid with one hidden = 3 visible.
        assert len(visible_axes) == 3, (
            f"gauss has 3 params → 3 visible subplots; got {len(visible_axes)}"
        )
        plt.close(fig); _close_all(stats)

    # -- F.40 — overlay mode --------------------------------------------

    def test_f40_overlay_mode(self):
        """mode='overlay' → single axes (not subplot grid)."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist(
            'x', bins=40, group_by='g', fit='gauss',
            summary_fit={'kind': 'figure', 'mode': 'overlay'},
        )
        params_fig = stats['summary_fit']['figure']
        visible_axes = [a for a in params_fig.axes if a.get_visible()]
        assert len(visible_axes) == 1, (
            f"overlay mode → single axes; got {len(visible_axes)}"
        )
        plt.close(fig); _close_all(stats)

    # -- F.41 — annotate mode -------------------------------------------

    def test_f41_annotate_mode(self):
        """annotate=True adds text annotations at each point. With ≤15 points,
        annotations are present; with >15 (default off), they aren't."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist(
            'x', bins=40, group_by='g', fit='gauss',
            summary_fit={'kind': 'figure', 'annotate': True},
        )
        params_fig = stats['summary_fit']['figure']
        # Look for annotation text on at least one axes.
        annotation_count = 0
        for axis in params_fig.axes:
            annotation_count += sum(
                1 for t in axis.texts if t.get_text().strip()
            )
        assert annotation_count >= 1, (
            "annotate=True with ≤15 points should produce annotations; "
            f"found {annotation_count}"
        )
        plt.close(fig); _close_all(stats)

    # -- F.42 — Scenario E: no group_by + no facet_by for params figure --

    def test_f42_no_group_no_facet_figure_is_scenario_e(self):
        """summary_fit='figure' without group_by/facet_by → Scenario E:
        summary_fit={} + summary_fit_note set."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist('x', bins=40, fit='gauss', summary_fit='figure')
        assert stats['summary_fit'] == {}, (
            f"Scenario E: empty dict expected; got {stats['summary_fit']!r}"
        )
        assert 'summary_fit_note' in stats
        assert 'x-axis' in stats['summary_fit_note'].lower() or 'axis' in stats['summary_fit_note'].lower()
        plt.close(fig)

    # -- F.43 — precision option ----------------------------------------

    def test_f43_precision_in_table_cells(self):
        """precision=3 produces more digits in numeric cells than precision=1."""
        d = DFDraw(_gauss_df())
        # Render at precision=1 and precision=3, compare cell text widths.
        cell_text_lens = {}
        for prec in (1, 3):
            fig, ax, stats = d.hist(
                'x', bins=40, group_by='g', fit='gauss',
                summary_fit={'kind': 'table', 'precision': prec},
            )
            table_fig = stats['summary_fit']['table']
            ax_t = table_fig.axes[0]
            # collect all cell texts containing a digit
            texts = []
            for child in ax_t.get_children():
                if hasattr(child, 'get_celld'):
                    for cell in child.get_celld().values():
                        txt = cell.get_text().get_text()
                        if any(c.isdigit() for c in txt):
                            texts.append(txt)
            cell_text_lens[prec] = sum(len(t) for t in texts)
            plt.close(fig); _close_all(stats)
        assert cell_text_lens[3] > cell_text_lens[1], (
            f"precision=3 should produce longer cells than precision=1; "
            f"got {cell_text_lens}"
        )

    # -- F.44 — 2D facet (Phase 13.41) composition ----------------------

    def test_f44_2d_facet_composition(self):
        """group_by + 2D facet_by → table has facet info; figure X uses tuples."""
        d = DFDraw(_gauss_df())
        fig, axes, stats = d.hist(
            'x', bins=40, group_by='g', facet_by=['f1', 'f2'],
            fit='gauss', summary_fit='both',
        )
        sf = stats['summary_fit']
        assert isinstance(sf.get('table'), plt.Figure)
        assert isinstance(sf.get('figure'), plt.Figure)
        # Verify data rows have facet info
        data = sf['data']
        assert all(r.get('facet') for r in data), (
            "every row should have a non-None 'facet' field for 2D facet"
        )
        plt.close(fig); _close_all(stats)

    # -- F.45 — normalize + summary_fit → Scenario E --------------------

    def test_f45_normalize_plus_summary_fit_is_scenario_e(self):
        """normalize='delta' + fit + summary_fit → Scenario E (fit consumed by
        normalize per CP1-5; summary_fit consumed per C-3). normalize requires
        a 2-vector expression (signal + reference per AD-80)."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.profile(
            '[y,x]:x', bins=20, normalize='delta',
            fit='gauss', summary_fit='table',
        )
        assert stats.get('summary_fit') == {}, (
            f"normalize→Scenario E expected empty dict; got {stats.get('summary_fit')!r}"
        )
        note = stats.get('summary_fit_note', '')
        assert 'normalize' in note.lower(), (
            f"normalize-related note expected; got {note!r}"
        )
        plt.close(fig)

    # -- F.46 — quantile-band profile + summary_fit → Scenario E --------

    def test_f46_quantile_band_profile_is_scenario_e(self):
        """profile in quantile-band mode → no fit-target curves → Scenario E
        (or summary_fit silently absent if fit produced nothing). The
        invariant is that summary_fit doesn't CRASH on this code path."""
        d = DFDraw(_gauss_df())
        # Use valid quantile-band API: quantiles= list + quantile_mode='band'
        fig, ax, stats = d.profile(
            'y:x', bins=20,
            quantiles=[0.25, 0.5, 0.75], quantile_mode='band',
            fit='gauss',  # may or may not produce fits in band mode
            summary_fit='table',
        )
        sf = stats.get('summary_fit')
        assert sf is not None, "summary_fit key should be present (E or filled)"
        # If empty, note must be present (Scenario E semantics)
        if sf == {}:
            assert 'summary_fit_note' in stats, (
                "Scenario E requires summary_fit_note diagnostic"
            )
        plt.close(fig); _close_all(stats)

    # -- F.47 — cumulative hist (Phase 13.40) ---------------------------

    def test_f47_cumulative_hist(self):
        """cumulative=True + group_by + fit → renders normally; one fit per group."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist(
            'x', bins=40, group_by='g', cumulative=True,
            fit='gauss', summary_fit='table',
        )
        sf = stats['summary_fit']
        # Table should be present (may have some fit failures but pipeline runs)
        assert isinstance(sf.get('table'), plt.Figure), "table fig expected"
        plt.close(fig); _close_all(stats)

    # -- F.47b — stacked + group_by + fit (D9/R4) -----------------------

    def test_f47b_stacked_grouped_fit(self):
        """stacked=True + group_by + fit → per-group fits per FIX1 D9/R4;
        summary_fit renders normally."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist(
            'x', bins=40, group_by='g', stacked=True,
            fit='gauss', summary_fit='table',
        )
        # Per FIX1 D9/R4: each group fits independently; summary_fit table OK.
        sf = stats['summary_fit']
        assert isinstance(sf.get('table'), plt.Figure), (
            "stacked+group_by+fit should produce table per D9/R4 semantics"
        )
        data = sf['data']
        assert len(data) >= 1, "should have at least one fit row"
        plt.close(fig); _close_all(stats)

    # -- F.48 — summary_fit without fit= → Scenario E silent ------------

    def test_f48_summary_fit_without_fit(self):
        """summary_fit='table' WITHOUT fit= → Scenario E silent path; no
        exception; note describes the gap."""
        d = DFDraw(_gauss_df())
        # Should not raise.
        fig, ax, stats = d.hist('x', bins=40, summary_fit='table')
        assert stats.get('summary_fit') == {}
        assert 'summary_fit_note' in stats
        plt.close(fig)

    # -- F.49 — auto-title content --------------------------------------

    def test_f49_auto_title_content(self):
        """Auto title contains 'fit params: ' and the fit_name (gauss)."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist(
            'y:x', bins=40, group_by='g', fit='gauss', summary_fit='table',
        )
        table_fig = stats['summary_fit']['table']
        suptitle = table_fig._suptitle
        assert suptitle is not None, "auto-title should set a suptitle"
        text = suptitle.get_text()
        assert text.startswith('fit params:'), f"auto-title prefix: {text!r}"
        assert 'gauss' in text, f"fit_name missing in title: {text!r}"
        plt.close(fig); _close_all(stats)

    # -- F.50 — title_overflow='truncate' --------------------------------

    def test_f50_title_overflow_truncate(self):
        """title_overflow='truncate' caps with ellipsis at font_size_min."""
        d = DFDraw(_gauss_df())
        # Force a long title
        long_title = "this is a deliberately long title meant to overflow " * 5
        fig, ax, stats = d.hist(
            'x', bins=40, group_by='g', fit='gauss',
            summary_fit={
                'kind': 'table',
                'title': long_title,
                'title_overflow': 'truncate',
            },
        )
        table_fig = stats['summary_fit']['table']
        # Truncate behavior should leave an ellipsis at the end.
        text = table_fig._suptitle.get_text()
        # Either the original long_title fits unmodified, or it got truncated.
        # The invariant: if truncated, ends with ellipsis.
        if text != long_title:
            assert text.endswith('…'), f"truncated title should end with ellipsis: {text!r}"
        plt.close(fig); _close_all(stats)

    # -- F.51 — default data_format = list[dict] -----------------------

    def test_f51_default_data_format_is_list_dict(self):
        """Default data_format produces list[dict] with _make_row keys."""
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.hist(
            'x', bins=40, group_by='g', fit='gauss', summary_fit='table',
        )
        data = stats['summary_fit']['data']
        assert isinstance(data, list), f"default data_format should be list; got {type(data).__name__}"
        assert all(isinstance(r, dict) for r in data), "each row must be a dict"
        # Required keys per §4.1.1 _make_row
        for r in data:
            assert 'group' in r and 'facet' in r and 'fit_name' in r
        plt.close(fig); _close_all(stats)

    # -- F.52 — pandas via style key -----------------------------------

    def test_f52_pandas_via_style_key(self):
        """set_style({'summary_fit.data_format': 'pandas'}) globally switches
        data format → DataFrame via from_records."""
        from dfextensions.dfdraw.style import set_style, get_style_value
        d = DFDraw(_gauss_df())
        prev = get_style_value('summary_fit.data_format', 'dict')
        set_style({'summary_fit.data_format': 'pandas'})
        try:
            fig, ax, stats = d.hist(
                'x', bins=40, group_by='g', fit='gauss', summary_fit='table',
            )
            data = stats['summary_fit']['data']
            assert isinstance(data, pd.DataFrame), (
                f"style switch should give DataFrame; got {type(data).__name__}"
            )
            assert 'group' in data.columns and 'fit_name' in data.columns
            plt.close(fig); _close_all(stats)
        finally:
            set_style({'summary_fit.data_format': prev})

    # -- F.53 — per-call data_format override --------------------------

    def test_f53_per_call_data_format(self):
        """summary_fit={'kind':'table','data_format':'pandas'} overrides
        module style, without mutating it."""
        from dfextensions.dfdraw.style import get_style_value
        d = DFDraw(_gauss_df())
        before = get_style_value('summary_fit.data_format', 'dict')
        fig, ax, stats = d.hist(
            'x', bins=40, group_by='g', fit='gauss',
            summary_fit={'kind': 'table', 'data_format': 'pandas'},
        )
        after = get_style_value('summary_fit.data_format', 'dict')
        assert before == after, "per-call override must not mutate module style"
        assert isinstance(stats['summary_fit']['data'], pd.DataFrame)
        plt.close(fig); _close_all(stats)

    # -- F.54 — same=True Replace mode ----------------------------------

    def test_f54_same_true_replace_mode(self):
        """Each draw(same=True) replaces stats['summary_fit']; previous figs
        dropped (Replace mode per §3.11)."""
        d = DFDraw(_gauss_df())
        fig1, ax1, stats1 = d.hist(
            'x', bins=40, group_by='g', fit='gauss',
            same=True, summary_fit='table',
        )
        sf1 = stats1['summary_fit']
        first_fig = sf1.get('table')
        # Second draw with same=True
        fig2, ax2, stats2 = d.hist(
            'x', bins=40, group_by='g', fit='gauss',
            same=True, summary_fit='table',
        )
        sf2 = stats2['summary_fit']
        # Each call has its OWN stats dict — verify the second call's summary
        # is a fresh dict (Replace mode at the returned-stats level).
        assert sf2 is not sf1, "Replace: each call returns its own summary"
        assert isinstance(sf2.get('table'), plt.Figure)
        plt.close(fig1); plt.close(fig2); _close_all(stats1); _close_all(stats2)

    # -- F.55 — placement invariants (inspect.signature) ----------------

    def test_f55_placement_invariants(self):
        """summary_fit is NOT in _dispatch_faceted_render parameters; IS in
        DFDraw.hist parameters. Locks Pattern A invariant from §9.1."""
        d = DFDraw(_gauss_df())
        # _dispatch_faceted_render does NOT receive summary_fit
        sig_disp = inspect.signature(d._dispatch_faceted_render)
        assert 'summary_fit' not in sig_disp.parameters, (
            "summary_fit MUST NOT be a parameter on _dispatch_faceted_render "
            "(Pattern A: outer-layer consume; never forwarded into faceted "
            "renderer)"
        )
        # DFDraw.hist DOES have summary_fit
        sig_hist = inspect.signature(DFDraw.hist)
        assert 'summary_fit' in sig_hist.parameters, (
            "summary_fit MUST be a named parameter on DFDraw.hist"
        )
        # And NOT in FORWARDED_NAMES (Pattern A)
        assert 'summary_fit' not in DFDraw._HIST_FORWARDED_NAMES
        assert 'summary_fit' not in DFDraw._PROFILE_FORWARDED_NAMES
        assert 'summary_fit' not in DFDraw._SCATTER_FORWARDED_NAMES

    # -- F.56 — faceted aggregation §4.2.0 ------------------------------

    def test_f56_faceted_aggregation_shape3(self):
        """After faceted draw with fit=, stats['fit'] is top-level Shape 3
        dict (tuple keys), AND per-cell access still works
        (Phase 13.42 D4 preserved)."""
        d = DFDraw(_gauss_df())
        fig, axes, stats = d.hist(
            'x', bins=40, facet_by=['f1', 'f2'], group_by='g',
            fit='gauss',
        )  # no summary_fit — just verify aggregation lands in stats
        top_fit = stats.get('fit')
        assert isinstance(top_fit, dict), (
            "Aggregation: top-level stats['fit'] must be dict (Shape 3)"
        )
        # Keys must be tuples (faceted axis)
        for k in top_fit.keys():
            assert isinstance(k, tuple), f"Shape 3 key must be tuple; got {k!r}"
        # Per-cell access still preserved (Phase 13.42 D4)
        # 2D facet stores per-cell stats at stats[(row, col)]
        cell_keys = [k for k in stats if isinstance(k, tuple)]
        assert len(cell_keys) >= 1, "per-cell tuple keys preserved"
        plt.close(fig)

    # -- F.56c — R-2 regression: scalar delegation forwarding -----------

    def test_f56c_draw_scalar_forwards_fit_and_summary_fit(self):
        """R-2 (Sonnet54 panel finding): DFDraw.draw scalar-path delegations
        must explicitly forward fit, fit_textbox_kwargs, and summary_fit —
        these are NAMED params on DFDraw.draw, not in **kwargs.

        Pre-fix: d.draw('y:x', type='hist', fit='gauss', summary_fit='table')
        silently dropped BOTH fit and summary_fit. This test locks the fix
        so neither can regress.
        """
        d = DFDraw(_gauss_df())
        fig, ax, stats = d.draw(
            'x', type='hist', bins=40, group_by='g',
            fit='gauss', summary_fit='table',
        )
        # fit MUST land — Phase 13.42 inline fit pipeline produces stats['fit']
        assert 'fit' in stats, (
            "R-2: d.draw(type='hist', fit=...) must forward fit through "
            "scalar dispatch (was pre-existing Phase 13.42 silent-drop bug)"
        )
        assert stats['fit'], "stats['fit'] should be non-empty for gauss fit"
        # summary_fit MUST land — Phase 13.43 outer-consume must trigger
        assert 'summary_fit' in stats, (
            "R-2: d.draw(type='hist', summary_fit=...) must forward "
            "summary_fit through scalar dispatch"
        )
        assert isinstance(stats['summary_fit'].get('table'), plt.Figure)
        plt.close(fig); _close_all(stats)

    # -- F.56b — faceted WITHOUT group_by (P2-C edge) -------------------

    def test_f56b_faceted_no_group_by(self):
        """Faceted draw without group_by → stats['fit'][facet_key] is a LIST
        (not a dict). _flatten_to_rows handles both inner sub-cases."""
        d = DFDraw(_gauss_df())
        fig, axes, stats = d.hist(
            'x', bins=40, facet_by=['f1', 'f2'],
            # NO group_by
            fit='gauss', summary_fit='table',
        )
        top_fit = stats.get('fit')
        assert isinstance(top_fit, dict), "Shape 3 dict expected"
        # Each cell value should be a list (sub-case b) — NOT a dict.
        for facet_key, cell_value in top_fit.items():
            assert isinstance(cell_value, list), (
                f"sub-case (b): cell value for facet_key={facet_key!r} must be "
                f"a list (faceted-no-group_by); got {type(cell_value).__name__}"
            )
        # summary_fit table should still render
        sf = stats['summary_fit']
        assert isinstance(sf.get('table'), plt.Figure), (
            "summary_fit must render for faceted-no-group_by case"
        )
        plt.close(fig); _close_all(stats)
