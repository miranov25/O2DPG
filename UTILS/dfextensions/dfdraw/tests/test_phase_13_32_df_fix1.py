"""Phase 13.32.DF FIX1 — Faceted Rendering Bug Fixes Regression Tests.

Real-data validation on time_series_tracks_0.root (TPC/ITS QA, 2026-05-16)
revealed three P1 bugs in ``_dispatch_faceted_render`` (drawer.py):

* BUG-001 — ``__dfdraw_facet_bin__`` internal temp column name leaked to
  subplot titles and ``stats['facet_by']``.
* BUG-002 — ``auto_title=True`` ignored in faceted mode (no figure-level
  suptitle generated).
* BUG-003 — facet bins sorted lexicographically (``"12" < "4"``) instead
  of numerically.

This file locks the invariance gates for those fixes. The §9.F002.2 test
also locks the v1.5-spec placement gap: ``_facet_display_name`` must be
defined at function entry so that channel-mode facets (``facet_by='group_by'``,
``'vector'``, ``'quantiles'``) do not raise ``NameError``.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


# =============================================================================
# Shared fixture
# =============================================================================

def make_facet_test_df():
    """Fixture for Phase 13.32 FIX1 tests.

    Uses uniform data in [0, 20] so that with ``facet_by_bins=5`` all five
    bins are populated:

        (0.0, 4.0], (4.0, 8.0], (8.0, 12.0], (12.0, 16.0], (16.0, 20.0]

    These labels expose the lexicographic-vs-numeric sort divergence:

    * Lexicographic order: ``(-0.0-4.0] < (12.0-16.0] < (16.0-20.0] <
      (4.0-8.0] < (8.0-12.0]``  ← BUG-003 buggy output
    * Numeric order:       ``(-0.0-4.0] < (4.0-8.0] < (8.0-12.0] <
      (12.0-16.0] < (16.0-20.0]``  ← correct after fix

    The previous fixture ``np.repeat([1.0, 4.0, 12.0, 20.0, 100.0])`` with
    ``bins=5`` populated only 2 bins (range 1–100, width ~20, values
    cluster in bins 0 and 4) — the assertion ``len(titles) == 5`` failed.
    """
    rng = np.random.default_rng(42)
    n = 500
    return pd.DataFrame({
        "x":         rng.uniform(0, 1, n),
        "y":         rng.normal(0, 1, n),
        "group_col": rng.uniform(0, 20, n),
    })


# =============================================================================
# §9.F001 — BUG-001 (subplot title leak)
# =============================================================================

class TestBUG001FacetBinNameLeak:
    """§9.F001 — locks that internal ``__dfdraw_facet_bin__`` column name
    does not appear in subplot titles or stats dict."""

    def test_BUG001_facet_bin_col_name_not_in_subplot_titles(self):
        """§9.F001.1: subplot titles must show the original ``facet_by``
        column name, never the internal ``__dfdraw_facet_bin__`` temp
        column. Stats dict ``facet_by`` must also expose the original."""
        d = DFDraw(make_facet_test_df())
        fig, ax, stats = d.profile(
            "y:x",
            facet_by="group_col",
            facet_by_bins=3,
        )
        try:
            for subplot_ax in fig.axes:
                title = subplot_ax.get_title()
                if not title:
                    continue
                assert "__dfdraw_facet_bin__" not in title, (
                    f"Internal column name leaked to subplot title: {title!r}"
                )
                assert "group_col" in title, (
                    f"Original facet_by name missing from subplot title: "
                    f"{title!r}"
                )
            # Lock the stats dict facet_by field too — same leak applied.
            assert stats["facet_by"] == "group_col", (
                f"stats['facet_by'] leaked internal name: "
                f"{stats['facet_by']!r}"
            )
        finally:
            plt.close(fig)


# =============================================================================
# §9.F002 — BUG-002 (auto_title=True in faceted mode)
# =============================================================================

class TestBUG002AutoTitleInFacetedMode:
    """§9.F002 — locks figure-level suptitle behaviour when ``auto_title=True``
    in faceted calls. F002.1 is the column-mode case; F002.2 is the
    channel-mode lock that closes the v1.5-spec scoping gap."""

    def test_BUG002_auto_title_sets_suptitle_in_faceted_mode(self):
        """§9.F002.1: ``auto_title=True`` must produce a non-empty
        ``fig.suptitle`` in faceted mode. Without the BUG-002 fix, the
        suptitle was never generated at all (per-subplot ``auto_title`` was
        correctly suppressed but the figure-level call was missing)."""
        d = DFDraw(make_facet_test_df())
        fig, ax, stats = d.profile(
            "y:x",
            facet_by="group_col",
            facet_by_bins=3,
            auto_title=True,
        )
        try:
            suptitle = fig._suptitle  # None if fig.suptitle() was never called
            assert suptitle is not None, (
                "fig.suptitle not set when auto_title=True in faceted mode"
            )
            assert len(suptitle.get_text()) > 0, (
                "fig.suptitle text is empty when auto_title=True in "
                "faceted mode"
            )
        finally:
            plt.close(fig)

    def test_BUG002_auto_title_channel_mode_facet_no_NameError(self):
        """§9.F002.2 (v1.5 spec-scoping gap lock — Sonet51 + Sonnet52_R1):

        Channel-mode facets (``facet_by='group_by'`` / ``'vector'`` /
        ``'quantiles'``) with ``auto_title=True`` must not raise
        ``NameError`` for ``_facet_display_name``. This is the test that
        would have caught the v1.5 spec bug where ``_facet_display_name``
        was initialised only inside the ``elif _facet_mode == 'column':``
        branch but used unconditionally at line 2703 + the BUG-002
        suptitle block.

        The fix places ``_facet_display_name = facet_by`` at function entry,
        so it's defined for ALL facet modes."""
        df_channel = pd.DataFrame({
            "x": np.arange(100, dtype=float),
            "y": np.random.default_rng(0).normal(size=100),
            "g": ["a"] * 50 + ["b"] * 50,
        })
        d = DFDraw(df_channel)
        # Should NOT raise NameError; suptitle text content unimportant here
        # — what's locked is that the channel-mode path doesn't crash.
        fig, ax, stats = d.profile(
            "y:x",
            group_by="g",
            facet_by="group_by",     # channel-mode sentinel
            auto_title=True,
        )
        try:
            # Channel mode should also produce a suptitle (failsafe or
            # proper). Asserting non-None catches both "code crashed" and
            # "failsafe never ran".
            assert fig._suptitle is not None, (
                "channel-mode + auto_title produced no suptitle "
                "(suggests _facet_display_name scoping regression)"
            )
        finally:
            plt.close(fig)


# =============================================================================
# §9.F003 — BUG-003 (lexicographic sort of bin intervals)
# =============================================================================

class TestBUG003FacetBinNumericSort:
    """§9.F003 — locks that auto-binned facets appear in numeric bin order,
    not lexicographic string order."""

    def test_BUG003_facet_bins_sorted_numerically_not_lexicographically(self):
        """§9.F003.1: ``facet_by_bins`` panels must appear in numeric bin
        order. With range [0, 20] and 5 bins the labels diverge:

        * Lexicographic: ``(-0.0-4.0] < (12.0-16.0] < (16.0-20.0] <
          (4.0-8.0] < (8.0-12.0]``  (wrong — what sorted() on strings gives)
        * Numeric:       ``(-0.0-4.0] < (4.0-8.0] < (8.0-12.0] <
          (12.0-16.0] < (16.0-20.0]``  (correct after _interval_sort_key)
        """
        from dfextensions.dfdraw.plots.profile import _interval_sort_key

        d = DFDraw(make_facet_test_df())
        fig, ax, stats = d.profile(
            "y:x",
            facet_by="group_col",
            facet_by_bins=5,
        )
        try:
            titles = [a.get_title() for a in fig.axes if a.get_title()]
            assert len(titles) == 5, (
                f"Expected 5 subplot titles, got {len(titles)}"
            )
            # Extract group_value (right of '=') from each title
            group_vals = [t.split("=", 1)[1] for t in titles if "=" in t]
            # Sort keys via the same function used by the production fix.
            # This locks "production order matches _interval_sort_key's
            # canonical order"; the _interval_sort_key function itself
            # has its own unit-test coverage in tests/test_profile.py.
            sort_keys = [_interval_sort_key(v) for v in group_vals]
            assert sort_keys == sorted(sort_keys), (
                f"Facet panels not in numeric bin order.\n"
                f"Group values (subplot order): {group_vals}\n"
                f"Sort keys: {sort_keys}"
            )
        finally:
            plt.close(fig)
