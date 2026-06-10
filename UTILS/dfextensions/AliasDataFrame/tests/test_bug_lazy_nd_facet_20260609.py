"""Regression tests for BUG_AliasDataFrame_20260609_lazy_nd_facet.

Origin: discovered during the time_series_draw.py real-data visual gallery
(fig13_profile_facet_nd — N-D facet over [side_type, qpt_bin10] where
qpt_bin10 is a lazy alias).

Bug class: same as BUG_20260518 (draw_subframe_alias_not_materialized) but
covers MODIFIER PARAMETERS instead of the main plot expression.

Pre-fix behavior:
    adf.draw_lazy = True
    adf.add_alias("qpt_bin10", "10*qpt")
    adf.draw("y:x", facet_by=["side_type", "qpt_bin10"], ...)
    # → KeyError: 'qpt_bin10'  (raised by dfdraw — column absent from adf.df)

Post-fix behavior:
    Same call succeeds; qpt_bin10 is materialized in adf.df before dfdraw
    accesses it, via _ensure_vector_kwargs_aliases extended to handle
    list-valued facet_by and group_by.

Fix location: AliasDataFrame.py:_ensure_vector_kwargs_aliases (Phase 13.35.ADF
mechanism), list-iteration branches added at lines 10980-11002.

Architectural pattern: same as the existing string-valued facet_by branch
(line 10975-10979) and the selection_vector/weights_vector loop (line 10965-
10969). No new function; the existing pre-materialization hook is extended.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Path setup — adjust if test layout differs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from AliasDataFrame import AliasDataFrame  # noqa: E402


def _build_test_adf(n=2000, seed=20260610):
    """Build a synthetic ADF mimicking the time_series_draw.py shape.

    Columns mirror the production gallery: side_type (categorical 0/1),
    qpt (continuous), sector (0-17), dcar_tpc_vertex (target y).
    Lazy alias qpt_bin10 = "10*qpt".int8 — replicates time_series.py:166.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'side_type':         rng.integers(0, 2, n).astype(np.int8),
        'qpt':               rng.uniform(-0.5, 0.5, n).astype(np.float32),
        'sector':            rng.integers(0, 18, n).astype(np.int32),
        'dcar_tpc_vertex':   rng.normal(0, 0.1, n).astype(np.float32),
        'tgl':               rng.uniform(-1.0, 1.0, n).astype(np.float32),
    })
    adf = AliasDataFrame(df)
    # Lazy alias — matches time_series.py:166 production pattern
    adf.add_alias("qpt_bin10", "10*qpt", dtype=np.int8)
    return adf


class TestBugLazyNDFacet(unittest.TestCase):
    """Regression tests for list-valued facet_by/group_by with lazy aliases."""

    # ─── Primary bug: list-valued facet_by with lazy alias ───────────────────

    def test_T1_list_facet_by_with_lazy_alias_no_crash(self):
        """[REGRESSION] adf.draw(facet_by=[col, lazy_alias]) does NOT raise KeyError.

        This is the canonical bug from the gallery (fig13_profile_facet_nd).
        Pre-fix: KeyError 'qpt_bin10' from dfdraw.
        Post-fix: succeeds; alias materialized before dispatch.
        """
        adf = _build_test_adf()
        adf.draw_lazy = True
        # Verify precondition: lazy alias is NOT yet in adf.df
        self.assertNotIn("qpt_bin10", adf.df.columns,
                         "precondition: lazy alias should not be pre-materialized")

        fig, ax, stats = adf.draw(
            "dcar_tpc_vertex:sector", type="profile", bins=18,
            facet_by=["side_type", "qpt_bin10"], facet_by_bins=[2, 3],
        )
        # Effect assertion: post-call, the alias IS now in adf.df (materialization happened)
        self.assertIn("qpt_bin10", adf.df.columns,
                      "lazy alias should be materialized by _ensure_vector_kwargs_aliases")
        # Sanity: a faceted figure was returned
        self.assertIsNotNone(fig)
        self.assertGreaterEqual(len(fig.axes), 2,
                                f"expected ≥2 facet panels, got {len(fig.axes)}")
        plt.close('all')

    def test_T2_list_facet_by_aliases_only(self):
        """[REGRESSION] All elements of facet_by are lazy aliases.

        Tests that the per-element loop materializes ALL list elements that
        are aliases (not just the first or last).
        """
        adf = _build_test_adf()
        adf.add_alias("qpt_bin5", "5*qpt", dtype=np.int8)
        adf.draw_lazy = True
        self.assertNotIn("qpt_bin10", adf.df.columns)
        self.assertNotIn("qpt_bin5", adf.df.columns)

        fig, ax, _ = adf.draw(
            "dcar_tpc_vertex:sector", type="profile", bins=18,
            facet_by=["qpt_bin5", "qpt_bin10"], facet_by_bins=[2, 3],
        )
        # Both aliases must be materialized
        self.assertIn("qpt_bin10", adf.df.columns)
        self.assertIn("qpt_bin5",  adf.df.columns)
        plt.close('all')

    def test_T3_list_facet_by_mixed_real_columns_and_alias(self):
        """[REGRESSION] facet_by mixing real columns and a lazy alias.

        side_type is a real column; qpt_bin10 is a lazy alias.
        The mixed list should work — only the alias triggers materialization.
        """
        adf = _build_test_adf()
        adf.draw_lazy = True

        fig, _, _ = adf.draw(
            "dcar_tpc_vertex:sector", type="profile", bins=18,
            facet_by=["side_type", "qpt_bin10"], facet_by_bins=[2, 3],
        )
        self.assertIn("qpt_bin10", adf.df.columns)
        # side_type was already a real column — no materialization needed for it
        plt.close('all')

    # ─── Companion gap NOT fixed (documented limitation) ─────────────────────
    # List-valued group_by was investigated as a companion gap but reverted
    # from the fix: dfdraw does not support list-valued group_by (raises
    # TypeError 'unhashable type: list' downstream). Materializing aliases
    # for an unsupported call shape would be dead code. If dfdraw adds
    # list-valued group_by support, a parallel branch in
    # _ensure_vector_kwargs_aliases should be added at that time.

    # ─── No-regression suite: existing paths must still work ─────────────────

    def test_T5_no_regression_string_facet_by(self):
        """[NO-REGRESSION] String-valued facet_by alias still resolved correctly.

        This was the pre-fix working path (line 10975-10979). Must not break.
        """
        adf = _build_test_adf()
        adf.draw_lazy = True

        fig, _, _ = adf.draw(
            "dcar_tpc_vertex:sector", type="profile", bins=18,
            facet_by="qpt_bin10",
        )
        self.assertIn("qpt_bin10", adf.df.columns)
        plt.close('all')

    def test_T6_no_regression_selection_vector_unchanged(self):
        """[NO-REGRESSION] selection_vector alias resolution still works.

        Sonnet59's scope-precision note: selection_vector and weights_vector
        were already handled at lines 10965-10969 — the fix MUST NOT touch
        them. This test ensures the new branches don't interact with the
        existing loop.
        """
        adf = _build_test_adf()
        adf.add_alias("is_pos_qpt", "qpt > 0")
        adf.draw_lazy = True
        self.assertNotIn("is_pos_qpt", adf.df.columns)

        fig, _, _ = adf.draw(
            "[dcar_tpc_vertex]:sector", type="profile", bins=18,
            selection_vector=["is_pos_qpt", "~is_pos_qpt"],
            normalize="delta",
        )
        self.assertIn("is_pos_qpt", adf.df.columns,
                      "selection_vector alias resolution must still work")
        plt.close('all')

    def test_T7_no_regression_idempotency_already_materialized(self):
        """[NO-REGRESSION] Pre-materialized aliases produce no extra work.

        If the alias is already a column, the fix path must be a silent no-op
        (no double-materialization, no exception).
        """
        adf = _build_test_adf()
        # Pre-materialize: alias now in df
        adf.materialize_aliases(names=["qpt_bin10"])
        self.assertIn("qpt_bin10", adf.df.columns)
        cols_before = list(adf.df.columns)

        # Call with the now-real column in list-valued facet_by
        adf.draw_lazy = True
        fig, _, _ = adf.draw(
            "dcar_tpc_vertex:sector", type="profile", bins=18,
            facet_by=["side_type", "qpt_bin10"], facet_by_bins=[2, 3],
        )
        # Columns unchanged (no duplicate added)
        cols_after = list(adf.df.columns)
        self.assertEqual(sorted(cols_before), sorted(cols_after),
                         "idempotent: pre-materialized alias must not be re-added")
        plt.close('all')

    def test_T8_empty_list_no_crash(self):
        """[EDGE] facet_by=[] (empty list) is a no-op."""
        adf = _build_test_adf()
        adf.draw_lazy = True
        # Empty list — must not raise and must not materialize anything
        try:
            fig, _, _ = adf.draw(
                "dcar_tpc_vertex:sector", type="profile", bins=18,
                facet_by=[],
            )
            plt.close('all')
        except (TypeError, ValueError):
            # Empty list may be rejected upstream by dfdraw — that's acceptable
            # (it's a dfdraw concern, not an alias-resolution concern). The point
            # of THIS test is that _ensure_vector_kwargs_aliases doesn't crash.
            pass
        # Specifically: ensure the alias-resolver did not add spurious columns
        self.assertNotIn("qpt_bin10", adf.df.columns,
                         "empty list must not trigger materialization of anything")


class TestEnsureVectorKwargsAliasesUnit(unittest.TestCase):
    """Unit tests targeting the _ensure_vector_kwargs_aliases helper directly,
    without invoking the full draw() pipeline. Closer to the fix site for
    precise diagnostic value when these tests fail."""

    def test_U1_list_facet_by_materializes_alias(self):
        """Unit: calling the helper directly materializes the list-valued alias."""
        adf = _build_test_adf()
        kwargs = {'facet_by': ['side_type', 'qpt_bin10'], 'facet_by_bins': [2, 3]}
        self.assertNotIn("qpt_bin10", adf.df.columns)
        adf._ensure_vector_kwargs_aliases(kwargs)
        self.assertIn("qpt_bin10", adf.df.columns)
        # kwargs itself should be unchanged
        self.assertEqual(kwargs['facet_by'], ['side_type', 'qpt_bin10'])

    def test_U2_string_facet_by_unchanged_behavior(self):
        """Unit: string-valued facet_by behavior unchanged by the fix."""
        adf = _build_test_adf()
        kwargs = {'facet_by': 'qpt_bin10'}
        adf._ensure_vector_kwargs_aliases(kwargs)
        self.assertIn("qpt_bin10", adf.df.columns)

    def test_U3_no_aliases_in_list_no_materialization(self):
        """Unit: list with only real-column names triggers no materialization."""
        adf = _build_test_adf()
        cols_before = set(adf.df.columns)
        kwargs = {'facet_by': ['side_type', 'sector']}  # both real columns
        adf._ensure_vector_kwargs_aliases(kwargs)
        self.assertEqual(set(adf.df.columns), cols_before,
                         "no aliases in list — column set unchanged")

    def test_U4_tuple_also_works(self):
        """Unit: tuple-valued facet_by (not just list) also triggers per-element."""
        adf = _build_test_adf()
        kwargs = {'facet_by': ('side_type', 'qpt_bin10')}  # tuple, not list
        adf._ensure_vector_kwargs_aliases(kwargs)
        self.assertIn("qpt_bin10", adf.df.columns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
