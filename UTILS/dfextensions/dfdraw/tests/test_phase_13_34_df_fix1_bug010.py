"""Phase 13.34.DF FIX1 — BUG-010 regression tests.

BUG-010: auto_title=True silently ignored in normalize= dispatchers.

Same bug class as Phase 13.32 FIX1 BUG-002 (faceted path), different code path:
  - BUG-002: _dispatch_faceted_render — fixed Phase 13.32 FIX1.
  - BUG-010: _dispatch_normalize_render, _dispatch_normalize_grouped_render,
             _dispatch_normalize_faceted_render — fixed Phase 13.34 FIX1.

§9 lock pattern matches §9.F002.1 from Phase 13.32 FIX1 (same invariant:
auto_title=True must produce a non-empty fig.suptitle).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


@pytest.fixture
def df_normalize_simple():
    """Two-group dataset suitable for normalize= testing."""
    rng = np.random.default_rng(42)
    n = 400
    return pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "y": rng.normal(0, 1, n),
        "group": ["a"] * (n // 2) + ["b"] * (n // 2),
        "fill":  list(range(4)) * (n // 4),
        "sector": rng.integers(0, 18, n),
    })


class TestBUG010_NormalizeAutoTitle:
    """§9.NB.* — auto_title=True must produce fig.suptitle in normalize= paths.

    Pattern matches §9.F002.1 from Phase 13.32 FIX1 (BUG-002, faceted path).
    Each test exercises one of the three normalize dispatchers separately,
    so a regression in one path fails one test (not all three).
    """

    def test_NB_1_auto_title_normalize_single_curve(self, df_normalize_simple):
        """§9.NB.1: auto_title=True in normalize= single-curve path must produce
        a non-empty fig.suptitle. Exercises _dispatch_normalize_render."""
        d = DFDraw(df_normalize_simple)
        fig, _ax, _stats = d.profile(
            "y:x",
            selection_vector=["group == 'a'", "group == 'b'"],
            normalize="delta",
            bins=10,
            auto_title=True,
        )
        try:
            suptitle = fig._suptitle
            assert suptitle is not None, (
                "fig.suptitle not set when auto_title=True in "
                "_dispatch_normalize_render (single-curve normalize path)"
            )
            text = suptitle.get_text()
            assert len(text) > 0, (
                f"fig.suptitle text is empty when auto_title=True; got {text!r}"
            )
        finally:
            plt.close(fig)

    def test_NB_2_auto_title_normalize_grouped(self, df_normalize_simple):
        """§9.NB.2: auto_title=True in normalize+group_by path must produce
        a non-empty fig.suptitle. Exercises _dispatch_normalize_grouped_render.

        group_by IS a parameter of this dispatcher, so build_auto_title
        receives group_by (the title may include "group:fill" depending on
        auto_title parts).
        """
        d = DFDraw(df_normalize_simple)
        fig, _ax, _stats = d.profile(
            "y:x",
            selection_vector=["group == 'a'", "group == 'b'"],
            normalize="delta",
            group_by="fill",
            bins=10,
            auto_title=True,
        )
        try:
            suptitle = fig._suptitle
            assert suptitle is not None, (
                "fig.suptitle not set when auto_title=True in "
                "_dispatch_normalize_grouped_render (normalize+group_by path)"
            )
            assert len(suptitle.get_text()) > 0, (
                "fig.suptitle text is empty in normalize+group_by path"
            )
        finally:
            plt.close(fig)

    def test_NB_3_auto_title_normalize_faceted(self, df_normalize_simple):
        """§9.NB.3: auto_title=True in normalize+facet_by path must produce
        a non-empty fig.suptitle. Exercises _dispatch_normalize_faceted_render.

        This dispatcher already had fig.suptitle(title) for explicit title;
        the BUG-010 fix adds the auto_title branch that runs when title is None.
        """
        d = DFDraw(df_normalize_simple)
        fig, _ax, _stats = d.profile(
            "y:x",
            selection_vector=["group == 'a'", "group == 'b'"],
            normalize="delta",
            facet_by="group",
            bins=10,
            auto_title=True,
        )
        try:
            suptitle = fig._suptitle
            assert suptitle is not None, (
                "fig.suptitle not set when auto_title=True in "
                "_dispatch_normalize_faceted_render (normalize+facet_by path)"
            )
            assert len(suptitle.get_text()) > 0, (
                "fig.suptitle text is empty in normalize+facet_by path"
            )
        finally:
            plt.close(fig)

    def test_NB_4_explicit_title_takes_precedence_over_auto_title(self, df_normalize_simple):
        """§9.NB.4: when both title= and auto_title=True are passed,
        explicit title wins (matches Phase 13.32 FIX1 BUG-002 contract).
        Locks the `if _auto_title_val and not title:` guard in all 3 dispatchers."""
        d = DFDraw(df_normalize_simple)
        fig, _ax, _stats = d.profile(
            "y:x",
            selection_vector=["group == 'a'", "group == 'b'"],
            normalize="delta",
            bins=10,
            auto_title=True,
            title="ExplicitTitleWins",
        )
        try:
            # Explicit title was given — set on the axes (existing behavior).
            # auto_title block must NOT also call fig.suptitle.
            suptitle = fig._suptitle
            # Either fig.suptitle is unset, OR it's the explicit title text
            # (some downstream paths may put title on figure too — accept either).
            if suptitle is not None:
                assert "ExplicitTitleWins" in suptitle.get_text() or suptitle.get_text() == "", (
                    f"auto_title overrode explicit title; suptitle={suptitle.get_text()!r}"
                )
        finally:
            plt.close(fig)

    def test_NB_5_auto_title_false_no_suptitle(self, df_normalize_simple):
        """§9.NB.5: auto_title=False (or omitted) must not set fig.suptitle
        in any normalize= path. Locks against the auto_title block running
        on the falsy default."""
        d = DFDraw(df_normalize_simple)
        # auto_title omitted (default False)
        fig, _ax, _stats = d.profile(
            "y:x",
            selection_vector=["group == 'a'", "group == 'b'"],
            normalize="delta",
            bins=10,
        )
        try:
            suptitle = fig._suptitle
            assert suptitle is None or suptitle.get_text() == "", (
                f"fig.suptitle was set without auto_title=True; "
                f"got {suptitle.get_text()!r}"
            )
        finally:
            plt.close(fig)
