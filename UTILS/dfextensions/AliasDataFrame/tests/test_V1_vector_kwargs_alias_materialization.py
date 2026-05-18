"""
Tests V1.1–V1.7 for PHASE_13_35_ADF — ADF alias pre-materialization for
selection_vector / weights_vector / facet_by kwargs.

Spec: PHASE_13_35_ADF_v1_2_Proposal_VectorParamsAliasMaterialization.md
Predecessor: 079b1467 (BUG_AliasDataFrame_20260518_PhaseA close)

Coverage:
  - V1.1: selection_vector in draw() with production-shape expression
          ('(abs(sector-13)<2)' from architect's 2026-05-18 session)
  - V1.2: selection_vector + weights_vector combined; both aliases asserted
  - V1.3: facet_by column-mode (Phase 13.31.DF) with alias column name
  - V1.4: idempotency — second draw() must be no-op for materialization
  - V1.5: draw_batch() with per-spec selection_vector containing alias
  - V1.6: draw_figures() with per-plot weights_vector containing alias
          (spec shape: list of fig_spec dicts, each with 'plots' list —
          source-verified at AliasDataFrame.py:12250)
  - V1.7: NEGATIVE BRANCH — facet_by channel-enum string must NOT
          trigger alias materialization (locks _FACET_BY_CHANNEL_ENUMS guard)

All tests follow these disciplines (per Phase A S10 precedent):
  - Failure Mode #12 entry-point rule: every test calls the exact public
    API the user hit (adf.draw, adf.draw_batch, adf.draw_figures).
  - Cold-draw discipline: NO test pre-accesses alias values before the
    first draw() call. Pre-access trivially "fixes" the bug as a side effect.
  - Fixture verification guards: fixture asserts preconditions (alias
    defined AND NOT yet materialized) before yielding ADF. Without these
    guards, a fixture-bug masquerades as a passing test.
  - Bug-fix-test entry-point rule: every test exercises the public API the
    original reproducer hit, and would fail without the fix.

Negative control: on unpatched source (HEAD 079b1467), V1.1–V1.6 fail with
UndefinedVariableError; V1.7 fails with AssertionError showing 'group_by'
materialized. Coder produces neg_control_before_patch.log per CRR §8.

Drafter: Claude36 (also Coder of v1.1 and v1.2 proposals).
COI: drafter ≠ implementer convention waived per architect direction
     (2026-05-18); routing for review requires non-Claude36 reviewers.
"""
import warnings

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest

from AliasDataFrame import AliasDataFrame


# =============================================================================
# Fixture
# =============================================================================

@pytest.fixture
def adf_with_alias_columns():
    """ADF where 'sector', 'weight_col', 'facet_col' are lazy aliases —
    not yet in df.columns. Mirrors the §1.4 production failure shape
    (time_series_tracks_0.root, 9.86M tracks, sector/time_s registered via
    apply_meta).

    Fixture preconditions (S10 precedent) — each must hold before yielding,
    otherwise a fixture-bug would masquerade as a passing test.
    """
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        'group_id':   np.repeat(np.arange(5), n // 5),
        'y_val':      rng.normal(0, 1, n).astype(np.float32),
        'x_val':      rng.uniform(0, 10, n).astype(np.float32),
        'raw_sector': np.tile(np.arange(36), n // 36 + 1)[:n].astype(np.int8),
        'raw_weight': rng.uniform(0.5, 1.5, n).astype(np.float32),
    })
    adf = AliasDataFrame(df)
    adf.draw_lazy = True
    adf.add_alias('sector', 'raw_sector')              # lazy alias
    adf.add_alias('weight_col', 'raw_weight')          # lazy alias
    adf.add_alias('facet_col', 'raw_sector % 2')       # lazy alias for facet_by

    # Fixture preconditions — preconditions for valid test (S10 precedent).
    # Aliases registered but NOT yet materialized: this is the EXACT state
    # production reproducer hits.
    assert 'sector'     not in adf.df.columns, \
        "FIXTURE BUG: 'sector' already materialized before test starts"
    assert 'weight_col' not in adf.df.columns, \
        "FIXTURE BUG: 'weight_col' already materialized before test starts"
    assert 'facet_col'  not in adf.df.columns, \
        "FIXTURE BUG: 'facet_col' already materialized before test starts"
    assert 'sector'     in adf.aliases, "FIXTURE BUG: 'sector' alias missing"
    assert 'weight_col' in adf.aliases, "FIXTURE BUG: 'weight_col' alias missing"
    assert 'facet_col'  in adf.aliases, "FIXTURE BUG: 'facet_col' alias missing"
    return adf


# =============================================================================
# Test class
# =============================================================================

class TestV1_VectorKwargsAliasMaterialization:
    """Phase 13.35.ADF — alias pre-materialization for vector draw kwargs."""

    # -------------------------------------------------------------------------
    # V1.1 — selection_vector with production-shape expression
    # -------------------------------------------------------------------------

    @pytest.mark.invariance
    def test_V1_1_selection_vector_alias_production_expression(self, adf_with_alias_columns):
        """selection_vector with the actual §1.4 production failure expression.

        Locks parenthesized + function-call expression form (abs(sector-13)<2)
        + alias resolution. Failure mode #12: calls exact public API the user
        hit (adf.draw with this kwarg shape in architect's 2026-05-18 session).

        Without the Phase 13.35.ADF fix this fails on UndefinedVariableError:
        name 'sector' is not defined.
        """
        adf = adf_with_alias_columns
        fig, ax, stats = adf.draw(
            "y_val:x_val",
            selection_vector=["(abs(sector-13)<2)", "(abs(sector-13)>=2)&(sector<36)"],
            type="profile", bins=5,
        )
        assert stats is not None, "Production-shape selection_vector returned no stats"
        # Alias materialized on adf.df as a side effect of draw()
        assert 'sector' in adf.df.columns, \
            "Alias 'sector' must be on adf.df after draw with selection_vector"
        plt.close('all')

    # -------------------------------------------------------------------------
    # V1.2 — combined selection_vector + weights_vector; both aliases
    # -------------------------------------------------------------------------

    @pytest.mark.invariance
    def test_V1_2_weights_vector_with_selection_vector(self, adf_with_alias_columns):
        """Both selection_vector and weights_vector aliases must be materialized.

        Catches regression where only one of the two kwargs is processed correctly.
        """
        adf = adf_with_alias_columns
        fig, ax, stats = adf.draw(
            "y_val:x_val",
            weights_vector=["weight_col", "weight_col"],
            selection_vector=["sector < 18", "sector >= 18"],
            type="profile", bins=5,
        )
        assert stats is not None
        # BOTH aliases must be materialized (P2-5 from v1.0 review)
        assert 'sector'     in adf.df.columns, \
            "Alias 'sector' (selection_vector) not materialized"
        assert 'weight_col' in adf.df.columns, \
            "Alias 'weight_col' (weights_vector) not materialized"
        plt.close('all')

    # -------------------------------------------------------------------------
    # V1.3 — facet_by column-mode (Phase 13.31.DF) with alias
    # -------------------------------------------------------------------------

    @pytest.mark.invariance
    def test_V1_3_facet_by_alias_column_materialized(self, adf_with_alias_columns):
        """facet_by='facet_col' where facet_col is a lazy alias works cold.

        Verifies column-name mode (Phase 13.31.DF AD-78 §2) with alias column.
        """
        adf = adf_with_alias_columns
        fig, ax, stats = adf.draw(
            "y_val:x_val",
            type="profile", bins=5,
            facet_by="facet_col",
        )
        assert stats is not None
        assert 'facet_col' in adf.df.columns, \
            "Alias 'facet_col' must be materialized when used as facet_by column"
        plt.close('all')

    # -------------------------------------------------------------------------
    # V1.4 — idempotency: second draw() is no-op for materialization
    # -------------------------------------------------------------------------

    @pytest.mark.invariance
    def test_V1_4_idempotent_repeated_draw(self, adf_with_alias_columns):
        """Second draw call must not re-materialize already-present aliases."""
        adf = adf_with_alias_columns
        adf.draw(
            "y_val:x_val",
            selection_vector=["sector < 18", "sector >= 18"],
            type="profile", bins=5,
        )
        cols_after_first = set(adf.df.columns)

        adf.draw(
            "y_val:x_val",
            selection_vector=["sector < 10", "sector >= 10"],
            type="profile", bins=5,
        )
        cols_after_second = set(adf.df.columns)

        assert cols_after_first == cols_after_second, (
            "Second draw must be a no-op for materialization. "
            "Added on 2nd call: {0}".format(cols_after_second - cols_after_first)
        )
        plt.close('all')

    # -------------------------------------------------------------------------
    # V1.5 — draw_batch() with per-spec selection_vector containing alias
    # -------------------------------------------------------------------------

    @pytest.mark.invariance
    def test_V1_5_draw_batch_selection_vector_alias(self, adf_with_alias_columns, tmp_path):
        """draw_batch() dispatches to dfdraw without self.draw() routing
        (source-verified at AliasDataFrame.py:11912). Per-spec selection_vector
        containing alias must be pre-materialized before dfdraw sees it.

        clear_after=False so the test can observe the materialization side
        effect; the default draw_batch behavior is to drop materialized aliases
        after the batch completes (see line 12144-12150). The fix is about
        the materialization HAPPENING during the call, not persistence after.
        Without the fix, the call raises UndefinedVariableError on 'sector'
        before reaching the cleanup block.
        """
        adf = adf_with_alias_columns
        specs = {
            'plot1': {
                'expr': 'y_val:x_val',
                'type': 'profile',
                'bins': 5,
                'selection_vector': ["sector < 18", "sector >= 18"],
            },
        }
        results = adf.draw_batch(
            specs=specs, save_dir=str(tmp_path), on_error='raise',
            verbose=False, clear_after=False,  # observe materialization
        )
        assert results is not None
        assert 'sector' in adf.df.columns, \
            "Alias 'sector' must be materialized when used in draw_batch per-spec selection_vector"
        plt.close('all')

    # -------------------------------------------------------------------------
    # V1.6 — draw_figures() with per-plot weights_vector containing alias
    # -------------------------------------------------------------------------
    # Spec dict shape source-verified at AliasDataFrame.py:12250:
    #   specs = [{'name': ..., 'plots': [{'expr': ..., 'selection_vector': ...}]}]
    # NOT the v1.2 proposal's draft shape ({'fig1': {'subplots': {...}}}).
    # Spec correction documented in CRR for v1.2.

    @pytest.mark.invariance
    def test_V1_6_draw_figures_weights_vector_alias(self, adf_with_alias_columns, tmp_path):
        """draw_figures() dispatches to dfdraw without self.draw() routing
        (source-verified at AliasDataFrame.py:12158, plot iteration at :12250).
        Per-plot weights_vector containing alias must be pre-materialized.

        clear_after=False so the test can observe the materialization side
        effect; see V1.5 docstring for the same rationale.
        """
        adf = adf_with_alias_columns
        specs = [
            {
                'name': 'fig1',
                'plots': [
                    {
                        'expr': 'y_val:x_val',
                        'type': 'profile',
                        'bins': 5,
                        'weights_vector': ["weight_col", "weight_col"],
                        'selection_vector': ["sector < 18", "sector >= 18"],
                    },
                ],
            },
        ]
        results = adf.draw_figures(
            specs=specs, save_dir=str(tmp_path), on_error='raise',
            verbose=False, clear_after=False,  # observe materialization
        )
        assert results is not None
        # BOTH aliases asserted
        assert 'sector'     in adf.df.columns, \
            "Alias 'sector' (selection_vector) not materialized in draw_figures"
        assert 'weight_col' in adf.df.columns, \
            "Alias 'weight_col' (weights_vector) not materialized in draw_figures"
        plt.close('all')

    # -------------------------------------------------------------------------
    # V1.8 — auto-force vector_compose='outer' (Phase 13.35.ADF ergonomic bridge)
    # -------------------------------------------------------------------------

    @pytest.mark.invariance
    def test_V1_8_auto_force_vector_compose_outer_for_single_Y(self, adf_with_alias_columns):
        """Single-Y + N-element selection_vector/weights_vector must trigger
        automatic vector_compose='outer' coercion.

        Locks the Phase 13.35.ADF ergonomic bridge: without this, users have to
        know dfdraw AD-67 and pass vector_compose='outer' explicitly, OR rely
        on normalize='delta' silently setting it inside dfdraw.

        Three sub-assertions in one test (helper-level, deterministic):
          (a) Positive: single-Y + N-sel forces 'outer'
          (b) Negative: multi-Y must NOT force (user is responsible)
          (c) Negative: user's explicit vector_compose='inner' must be respected
        """
        adf = adf_with_alias_columns

        # (a) Positive: single-Y + 2-element selection_vector → auto-force outer
        kwargs_a = {'selection_vector': ["sector < 18", "sector >= 18"]}
        adf._normalize_vector_compose_kwargs(kwargs_a, expr='y_val:x_val')
        assert kwargs_a.get('vector_compose') == 'outer', \
            "Auto-force failed: single-Y + 2-sel should set vector_compose='outer'"

        # (a') Positive: weights_vector also triggers
        kwargs_a2 = {'weights_vector': ["w1", "w2"]}
        adf._normalize_vector_compose_kwargs(kwargs_a2, expr='y_val:x_val')
        assert kwargs_a2.get('vector_compose') == 'outer', \
            "Auto-force failed: single-Y + 2-weights should set vector_compose='outer'"

        # (b) Negative: multi-Y must NOT auto-force (inner is correct for multi-Y)
        kwargs_b = {'selection_vector': ["sector < 18", "sector >= 18"]}
        adf._normalize_vector_compose_kwargs(kwargs_b, expr='y1,y2:x_val')
        assert 'vector_compose' not in kwargs_b, (
            "Multi-Y must NOT auto-force vector_compose — inner handles 2x2 natively. "
            "Found: {0}".format(kwargs_b.get('vector_compose'))
        )

        # (c) Negative: explicit user choice MUST be respected (don't overwrite)
        kwargs_c = {
            'selection_vector': ["sector < 18", "sector >= 18"],
            'vector_compose': 'inner',  # user explicitly requests inner
        }
        adf._normalize_vector_compose_kwargs(kwargs_c, expr='y_val:x_val')
        assert kwargs_c['vector_compose'] == 'inner', (
            "User explicit vector_compose='inner' must be respected (not overwritten). "
            "Found: {0}".format(kwargs_c['vector_compose'])
        )

        # (d) Edge: single-element selection_vector → no-op (N=1 is not vector mode)
        kwargs_d = {'selection_vector': ["sector < 18"]}
        adf._normalize_vector_compose_kwargs(kwargs_d, expr='y_val:x_val')
        assert 'vector_compose' not in kwargs_d, (
            "1-element selection_vector should not trigger auto-force. "
            "Found: {0}".format(kwargs_d.get('vector_compose'))
        )

    @pytest.mark.invariance
    def test_V1_7_facet_by_channel_enum_not_materialized(self, adf_with_alias_columns):
        """facet_by channel-enum string MUST NOT trigger alias materialization,
        even when an alias with the same name exists.

        Locks the _FACET_BY_CHANNEL_ENUMS guard ({'group_by', 'vector',
        'quantiles'}). Without this guard, a user who happens to define an
        alias named 'group_by' (or 'vector' / 'quantiles') would have it
        incorrectly materialized when they pass facet_by='group_by' in
        channel-enum mode — the Phase 13.31.DF AD-78 §2 contract would
        silently regress.
        """
        adf = adf_with_alias_columns

        # Pathological setup: define an alias whose name collides with channel
        # enum. This isolates the guard from "alias is missing" / "name doesn't
        # exist" effects — only the guard prevents materialization.
        adf.add_alias('group_by', 'raw_sector')
        assert 'group_by' in adf.aliases, \
            "FIXTURE BUG: pathological 'group_by' alias not registered"
        assert 'group_by' not in adf.df.columns, \
            "FIXTURE BUG: pathological 'group_by' alias pre-materialized"

        cols_before = set(adf.df.columns)

        # End-to-end via public API. If draw() raises for unrelated reasons
        # (e.g., dfdraw rejects the channel/expr combo), the assertion below
        # still holds — the test is about the materialization side effect,
        # not about draw() return success.
        try:
            adf.draw("y_val:x_val", type="profile", bins=5, facet_by="group_by")
        except Exception:
            pass  # acceptable — see docstring; we're testing side effect, not success

        # The guard must have prevented 'group_by' from being added to self.df
        assert 'group_by' not in adf.df.columns, (
            "Guard failed: facet_by='group_by' (channel enum) was treated as "
            "alias and materialized. _FACET_BY_CHANNEL_ENUMS guard regressed — "
            "Phase 13.31.DF AD-78 §2 channel-mode contract violated."
        )
        assert set(adf.df.columns) == cols_before, (
            "draw() with channel-enum facet_by changed adf.df.columns. "
            "Added: {0}".format(set(adf.df.columns) - cols_before)
        )
        plt.close('all')
