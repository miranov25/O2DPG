"""
Phase 13.31.DF v1.0 — facet_by column-name support tests.

§9 load-bearing invariance markers per Coder QRC v1.30 Rule 14.

Test classes:
- TestPhase1330Safe                (3 — Sonnet P1 lock: channel-name facet_by must NOT
                                       raise from Phase 13.30 validate_column_references)
- TestColumnNameMode_Numeric       (3 — int / float column dtypes)
- TestColumnNameMode_String        (1 — string/object column dtype, regression for
                                       Sonnet implementation note on dtype handling)
- TestOrthogonalComposition        (1 — facet_by="side" + group_by="quantile_col" =
                                       N panels × M overlays per AD-78 §6)
- TestAmbiguityError               (2 — typo / non-existent: error names BOTH alternatives)
- TestCardinalityCap               (1 — too many unique values respects channels.cycles.facet_max)
- TestRegressionCommit1            (1 — existing facet_by="group_by" byte-identical
                                       to pre-Phase-13.31 behaviour, locks the orthogonal
                                       _inner_group_by branch logic)

Total: 12 invariance tests.

Refs:
- AD-78 (facet_by accepts column names in addition to channel names)
- PHASE_13_31_DF_v1_0_Proposal_FacetByColumnName.md
- BUG_ADF_GroupBy_Expression_Materialization (cross-reference for the
  selection-extension via mask filtering not string-concat)
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfdraw import DFDraw


# =============================================================================
# TestPhase1330Safe — Sonnet P1 lock
# =============================================================================
class TestPhase1330Safe:
    """Channel-name facet_by values must NOT be rejected by Phase 13.30's
    validate_column_references(). The Phase 13.30 _PROFILE_COLUMN_REFERENCES
    tuple deliberately does NOT include 'facet_by' (per AD-78 §6 + Sonnet P1
    cross-review). These tests lock that decision in place."""

    def _df(self, n=90):
        return pd.DataFrame({
            "x": np.linspace(0, 1, n),
            "y": np.random.RandomState(42).randn(n),
            "row": np.arange(n) % 3,
        })

    def test_channel_facet_by_group_by_does_not_raise_p1330(self):
        """§9.Phase1330Safe.1 — facet_by='group_by' must execute without raising.
        Pre-Phase-13.31 regression lock for Sonnet P1: if 'facet_by' were ever
        accidentally added to _PROFILE_COLUMN_REFERENCES, this test fails because
        the validator would raise on the channel-name string 'group_by'."""
        df = self._df()
        # §9.Phase1330Safe.1
        fig, axes, stats = DFDraw(df).profile(
            "y:x", group_by="row", facet_by="group_by",
            bins=10, min_entries=2,
        )
        assert stats["faceted"] is True
        assert stats["facet_by"] == "group_by"

    def test_channel_facet_by_quantiles_does_not_raise_p1330(self):
        """§9.Phase1330Safe.2 — same regression lock for 'quantiles' channel mode."""
        df = self._df()
        # §9.Phase1330Safe.2
        fig, axes, stats = DFDraw(df).profile(
            "y:x",
            facet_by="quantiles",
            quantiles=[0.25, 0.5, 0.75],
            quantile_mode="discrete",
            bins=10, min_entries=2,
        )
        assert stats["faceted"] is True

    def test_channel_facet_by_vector_does_not_raise_p1330(self):
        """§9.Phase1330Safe.3 — same regression lock for 'vector' channel mode."""
        df = pd.DataFrame({
            "x": np.linspace(0, 1, 30),
            "y1": np.random.RandomState(0).randn(30),
            "y2": np.random.RandomState(1).randn(30),
        })
        # §9.Phase1330Safe.3
        fig, axes, stats = DFDraw(df).profile(
            "[y1, y2]:x", facet_by="vector",
            bins=5, min_entries=2,
        )
        assert stats["faceted"] is True


# =============================================================================
# TestColumnNameMode_Numeric
# =============================================================================
class TestColumnNameMode_Numeric:
    """Column-name facet_by with numeric column dtypes."""

    def _df(self, n=120):
        return pd.DataFrame({
            "x": np.tile(np.linspace(0, 1, n // 3), 3),
            "y": np.random.RandomState(42).randn(n),
            "side_int": np.repeat([1, 2, 3], n // 3),                  # int dtype
            "quantile_col": np.tile(np.repeat([0, 1, 2, 3], n // 12), 3),
        })

    def test_int_column_facet_produces_n_subplots(self):
        """§9.NewCol.1 — facet_by='side_int' (int column with 3 unique values)
        produces 3 subplots."""
        df = self._df()
        # §9.NewCol.1
        fig, axes, stats = DFDraw(df).profile(
            "y:x", facet_by="side_int",
            bins=5, min_entries=2,
        )
        assert stats["faceted"] is True
        assert stats["facet_mode"] == "column"
        assert stats["n_groups"] == 3
        assert stats["groups"] == [1, 2, 3]  # sorted ascending per AD-78 §6

    def test_float_column_facet_works(self):
        """§9.NewCol.2 — facet_by works for float-dtype columns (e.g. binned
        floats from group_by_quantiles output that the user pre-materialized)."""
        df = pd.DataFrame({
            "x": np.tile(np.linspace(0, 1, 30), 2),
            "y": np.random.RandomState(0).randn(60),
            "quartile_pT": np.repeat([0.25, 0.75], 30).astype(float),
        })
        # §9.NewCol.2
        fig, axes, stats = DFDraw(df).profile(
            "y:x", facet_by="quartile_pT",
            bins=5, min_entries=2,
        )
        assert stats["n_groups"] == 2
        assert stats["facet_mode"] == "column"

    def test_subplot_titles_show_facet_value(self):
        """§9.NewCol.3 — each subplot title contains 'facet_by=group_value'."""
        df = self._df()
        # §9.NewCol.3
        fig, axes, stats = DFDraw(df).profile(
            "y:x", facet_by="side_int",
            bins=5, min_entries=2,
        )
        titles = [ax.get_title() for ax in axes]
        # Phase 13.27 Commit 1 convention: f"{facet_by}={group_value}"
        assert any("side_int" in t and "1" in t for t in titles), titles
        assert any("side_int" in t and "2" in t for t in titles), titles


# =============================================================================
# TestColumnNameMode_String — Sonnet implementation note coverage
# =============================================================================
class TestColumnNameMode_String:
    """String / object dtype columns work via mask filtering (no string-quoting
    concern because we filter via boolean mask, not selection-string concat —
    Sonnet implementation note, AD-78 §7.1)."""

    def test_string_column_facet_works_without_quoting_issue(self):
        """§9.NewCol.4 — facet_by on a string-dtype column produces correct subplots.
        Locks the Sonnet implementation-note regression: the existing Phase 13.27
        Commit 1 code uses df[df[col] == val] (mask filter), which handles strings
        and numerics uniformly — NO dtype dispatch needed in dfdraw."""
        n = 60
        df = pd.DataFrame({
            "x": np.tile(np.linspace(0, 1, n // 2), 2),
            "y": np.random.RandomState(0).randn(n),
            "region": np.repeat(["A_side", "C_side"], n // 2),  # string dtype
        })
        # §9.NewCol.4
        fig, axes, stats = DFDraw(df).profile(
            "y:x", facet_by="region",
            bins=5, min_entries=2,
        )
        assert stats["n_groups"] == 2
        assert stats["facet_mode"] == "column"
        assert set(stats["groups"]) == {"A_side", "C_side"}


# =============================================================================
# TestOrthogonalComposition — AD-78 §6 "what this enables"
# =============================================================================
class TestOrthogonalComposition:
    """facet_by (column-mode) composes orthogonally with group_by:
    facet_by selects spatial encoding (subplots), group_by selects color/style
    encoding (overlays). This is the production reproducer from the AD-78
    review note."""

    def test_facet_by_column_AND_group_by_compose(self):
        """§9.Orthogonal.1 — facet_by='side' (2 unique values) AND
        group_by='quartile' (3 unique values) produces 2 subplots,
        each with 3 overlaid lines (3 groups inside each panel).

        This is the production case: facet=spatial, group=color.
        """
        n = 240
        df = pd.DataFrame({
            "x":         np.tile(np.linspace(0, 1, 20), 12),
            "y":         np.random.RandomState(42).randn(n),
            "side":      np.repeat([1, -1], n // 2),                  # 2 panels
            "quartile":  np.tile(np.repeat([0, 1, 2], n // 6), 2),    # 3 overlays per panel
        })
        # §9.Orthogonal.1
        fig, axes, stats = DFDraw(df).profile(
            "y:x",
            facet_by="side",       # spatial: 2 panels
            group_by="quartile",   # color: 3 overlays per panel
            bins=5, min_entries=2,
        )
        # 2 panels
        assert stats["n_groups"] == 2
        assert stats["facet_mode"] == "column"
        # Each panel must have ≥3 lines (one per quartile value).
        # The 'group_by' was NOT consumed by the facet; it's still active
        # inside each subplot.
        line_counts = [len(ax.get_lines()) for ax in axes]
        assert all(n >= 3 for n in line_counts), (
            f"Each panel should have ≥3 group-overlay lines; got {line_counts}"
        )


# =============================================================================
# TestAmbiguityError
# =============================================================================
class TestAmbiguityError:
    """Typo / wholly-invalid facet_by value: error message must name BOTH
    interpretations (channel-name set + available columns)."""

    def test_typo_facet_by_raises_with_both_alternatives(self):
        """§9.AmbigErr.1 — facet_by='sied' (typo, neither channel nor column)
        must raise ValueError mentioning BOTH 'channel names' and 'columns'."""
        df = pd.DataFrame({
            "x": [1, 2, 3], "y": [4, 5, 6], "side": [0, 1, 0],
        })
        with pytest.raises(ValueError) as exc_info:
            DFDraw(df).profile("y:x", facet_by="sied", bins=2, min_entries=1)
        msg = str(exc_info.value)
        # §9.AmbigErr.1 — both alternatives present
        assert "channel name" in msg.lower(), f"Missing channel-name hint: {msg!r}"
        assert "column" in msg.lower(), f"Missing column hint: {msg!r}"
        assert "sied" in msg, f"Missing offending value: {msg!r}"

    def test_error_message_lists_available_columns(self):
        """§9.AmbigErr.2 — error message includes available columns for actionability."""
        df = pd.DataFrame({
            "x": [1, 2, 3], "y": [4, 5, 6], "side": [0, 1, 0],
        })
        with pytest.raises(ValueError) as exc_info:
            DFDraw(df).profile("y:x", facet_by="nope", bins=2, min_entries=1)
        msg = str(exc_info.value)
        # §9.AmbigErr.2
        assert "side" in msg, f"Available columns not surfaced: {msg!r}"


# =============================================================================
# TestCardinalityCap
# =============================================================================
class TestCardinalityCap:
    """Column-name facet honours channels.cycles.facet_max same as channel-name
    mode (AD-78 §6 'what this commits to')."""

    def test_column_facet_too_many_unique_values_triggers_cap(self):
        """§9.Cap.1 — column with >16 unique values triggers the cap policy.
        Default channels.overflow='error' (per Phase 13.27 Commit 1)."""
        n = 200
        df = pd.DataFrame({
            "x": np.linspace(0, 1, n),
            "y": np.random.RandomState(0).randn(n),
            "many_groups": np.arange(n) % 25,   # 25 unique values, >16 cap
        })
        # §9.Cap.1 — with default channels.overflow='error', should raise
        with pytest.raises(ValueError) as exc_info:
            DFDraw(df).profile(
                "y:x", facet_by="many_groups",
                bins=5, min_entries=1,
            )
        assert "facet_max" in str(exc_info.value).lower() or "exceed" in str(exc_info.value).lower()


# =============================================================================
# TestRegressionCommit1 — orthogonal group_by branch + invariance
# =============================================================================
class TestRegressionCommit1:
    """Phase 13.27 Commit 1's existing facet_by='group_by' channel-mode
    must remain byte-identical after Phase 13.31. Specifically: in 'group_by'
    channel mode, the inner group_by must be suppressed (the facet IS the
    group_by); in column-name mode, the inner group_by must be preserved
    (orthogonal dimension). The _inner_group_by branch in dispatch encodes
    this — regress if that logic is ever flattened."""

    def test_channel_group_by_mode_suppresses_inner_group_by(self):
        """§9.RegressCh.1 — facet_by='group_by' must produce subplots WITHOUT
        per-subplot group_by overlay (one line per subplot, not many).
        Locks the _inner_group_by=None branch."""
        n = 120
        df = pd.DataFrame({
            "x": np.tile(np.linspace(0, 1, 20), 6),
            "y": np.random.RandomState(0).randn(n),
            "side": np.repeat([1, 2, 3], n // 3),
        })
        # §9.RegressCh.1
        fig, axes, stats = DFDraw(df).profile(
            "y:x", group_by="side", facet_by="group_by",
            bins=5, min_entries=2,
        )
        assert stats["n_groups"] == 3
        # Each subplot has ONE line (the facet consumed group_by; no inner overlay)
        line_counts = [len(ax.get_lines()) for ax in axes]
        # Exactly one curve per panel (might also have error bars etc., so allow ≤ small)
        for lc, ax in zip(line_counts, axes):
            # No multi-group overlay — confirm by checking distinct legend entries
            legend = ax.get_legend()
            n_labels = len(legend.get_texts()) if legend is not None else 0
            assert n_labels <= 1, (
                f"Channel 'group_by' facet must suppress inner group overlay; "
                f"panel has {n_labels} legend entries"
            )
