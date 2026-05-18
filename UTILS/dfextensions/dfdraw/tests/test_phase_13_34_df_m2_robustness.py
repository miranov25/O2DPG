"""Phase 13.34.DF M2 — Robustness gap invariance tests.

Three new §9 areas surfaced during the Phase 13.32 FIX1 audit + 5-revision
spec cycle. Tests added here ARE the protection going forward.

* §9.MED.1 — central='median' should use MAD-sigma error bars, not mean σ/√n.
  Currently fails (known inconsistency from Phase 13.33 CRR §11), so marked
  xfail(strict=False) — locks the contract; converts to XPASS if/when fixed.

* §9.STATS.* — stats dict schema locks per plot kind. Catches silent key
  renames that would break downstream consumers (ADF, RootInteractive).

* §9.X.* — kwarg interaction tests. Recurrence prevention for the bug class
  the Phase 13.32 FIX1 spec cycle surfaced (BUG-001/002/003 all interaction-
  boundary bugs).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


# =============================================================================
# §9.MED.1 — central='median' must use MAD-sigma error bars
# =============================================================================

class TestMedianMADSigma:
    """§9.MED.1 — locks the pre-existing inconsistency flagged in Phase 13.33
    CRR §11. Currently `central='median'` uses mean-based σ/√n error bars,
    NOT MAD-sigma. The _compute_per_bin_mad_sigma helper exists (profile.py:1335)
    but is wired only into the normalize path."""

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "central='median' currently uses mean-based σ/√n errors, not "
            "MAD-sigma. Known inconsistency from Phase 13.33 CRR §11. "
            "_compute_per_bin_mad_sigma exists but is wired only into the "
            "normalize path. xfail(strict=False) allows test to silently "
            "convert to XPASS if/when fixed in a future phase, without "
            "breaking the suite. Source-of-truth lock for the contract."
        ),
    )
    def test_MED_1_median_uses_mad_sigma_for_errors(self):
        """§9.MED.1: With heavy-outlier data where MAD ≠ σ, central='median'
        error bars must match MAD-sigma to within 10%, not mean σ/√n."""
        rng = np.random.default_rng(42)
        n = 1000
        # Construct data where σ ≫ MAD:
        # - 95% Gaussian centered at 0 with σ=1
        # - 5% extreme outliers (factor 20)
        # MAD-sigma stays ~1; mean-sigma blows up to ~5+
        y_core = rng.normal(0, 1, int(n * 0.95))
        y_out = rng.normal(0, 20, n - len(y_core))
        y_all = np.concatenate([y_core, y_out])
        rng.shuffle(y_all)
        df = pd.DataFrame({"x": rng.uniform(0, 10, n), "y": y_all})

        d = DFDraw(df)
        fig, ax, stats = d.profile(
            "y:x", central="median", bins=5, error="sem",
            return_data=True,
        )
        try:
            # Extract rendered y-error magnitudes from the errorbar collection
            err_collections = [c for c in ax.collections if hasattr(c, 'get_segments')]
            assert err_collections, "no error bar segments rendered"
            segs = err_collections[0].get_segments()
            rendered_err = np.array([(s[1, 1] - s[0, 1]) / 2 for s in segs])

            # Independent MAD-sigma reference per bin
            from dfdraw.plots.profile import _compute_per_bin_mad_sigma
            mad_sigma = _compute_per_bin_mad_sigma(
                df["x"].to_numpy(), df["y"].to_numpy(),
                bins=5, x_range=(0, 10),
            )

            # The error bars should be MAD-sigma / sqrt(n), within 30% tolerance.
            # If mean-sigma is being used, error bars will be ~5x bigger (outliers).
            # Use a generous tolerance to lock the QUALITATIVE behaviour.
            n_per_bin = stats["profile_data"]["count"].to_numpy()
            expected_sem = mad_sigma / np.sqrt(np.maximum(n_per_bin, 1))
            finite = np.isfinite(rendered_err) & np.isfinite(expected_sem)
            ratio = rendered_err[finite] / expected_sem[finite]
            assert np.all(np.abs(ratio - 1.0) < 0.3), (
                f"central='median' error bars should match MAD-sigma SEM "
                f"within 30%. Got ratios: {ratio.tolist()}"
            )
        finally:
            plt.close(fig)


# =============================================================================
# §9.STATS.* — Stats dict schema locks per plot kind
# =============================================================================

class TestStatsDictSchema:
    """§9.STATS.* — locks required keys in stats dict per plot kind.
    Downstream consumers (ADF, RootInteractive integration) depend on these
    keys; silent renames during refactoring would break them.

    The lock is intentionally MINIMAL — only well-known stable keys that have
    been documented in CRRs or AD records. Extending these tests in future
    phases is a deliberate, reviewed action.
    """

    @pytest.fixture
    def df_simple(self):
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "x": rng.uniform(0, 10, 200),
            "y": rng.normal(0, 1, 200),
            "g": ["a"] * 100 + ["b"] * 100,
        })

    def test_STATS_profile_keys_present(self, df_simple):
        """§9.STATS.1: profile() stats dict must contain the keys downstream
        consumers rely on. Lock against silent renames."""
        d = DFDraw(df_simple)
        fig, _, stats = d.profile("y:x", bins=10, return_data=True)
        try:
            # Minimum required surface; not exhaustive but stable contract.
            required = ["profile_data", "n_input"]
            for key in required:
                assert key in stats, (
                    f"profile() stats dict missing key {key!r}; "
                    f"present: {sorted(stats.keys())}"
                )
        finally:
            plt.close(fig)

    def test_STATS_normalize_keys_single_curve(self, df_simple):
        """§9.STATS.2: normalize='delta' (single-curve mode) stats dict must
        contain ax_diff + normalize_mode + normalize_layout + normalize_data
        (AD-81 return-type contract from Phase 13.33 CRR)."""
        d = DFDraw(df_simple)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["g == 'a'", "g == 'b'"],
            normalize="delta",
            bins=10,
        )
        try:
            for key in ["ax_diff", "normalize_mode", "normalize_layout",
                       "normalize_data"]:
                assert key in stats, (
                    f"normalize single-curve stats missing {key!r}; "
                    f"present: {sorted(stats.keys())}"
                )
        finally:
            plt.close(fig)

    def test_STATS_normalize_keys_grouped(self, df_simple):
        """§9.STATS.3: normalize + group_by stats dict has different key
        contract (n_groups + normalize_data_grouped) — locked separately."""
        df = df_simple.copy()
        df["fill"] = ([1] * 50 + [2] * 50) * 2  # two fills per group
        d = DFDraw(df)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["g == 'a'", "g == 'b'"],
            normalize="delta",
            group_by="fill",
            bins=10,
        )
        try:
            for key in ["n_groups", "normalize_data_grouped", "ax_diff"]:
                assert key in stats, (
                    f"normalize+group_by stats missing {key!r}; "
                    f"present: {sorted(stats.keys())}"
                )
            # M1 single-curve key must NOT appear in grouped mode (different
            # contract); locks the v1.0 design decision.
            assert "normalize_data" not in stats, (
                "M1 'normalize_data' key must not appear in grouped contract"
            )
        finally:
            plt.close(fig)


# =============================================================================
# §9.X.* — Feature composition / interaction-boundary tests
# =============================================================================

class TestKwargComposition:
    """§9.X.* — interaction-boundary tests. Recurrence prevention for the bug
    class that the Phase 13.32 FIX1 spec cycle surfaced.

    BUG-001 (title leak), BUG-002 (auto_title ignored), BUG-003 (lex sort)
    are all interaction-boundary bugs that single-feature tests didn't catch.
    A single composition test per kwarg pair at Phase 13.32 delivery would
    have caught all three before the real-data session.
    """

    @pytest.fixture
    def df_facet_data(self):
        rng = np.random.default_rng(42)
        n = 500
        return pd.DataFrame({
            "x": rng.uniform(0, 1, n),
            "y": rng.normal(0, 1, n),
            "facet_col": rng.uniform(0, 20, n),
            "group_col": (["a"] * 250 + ["b"] * 250),
        })

    def test_X_facet_bins_with_auto_title(self, df_facet_data):
        """§9.X.1: facet_by_bins composed with auto_title=True must produce
        a non-empty suptitle (would have caught BUG-002)."""
        d = DFDraw(df_facet_data)
        fig, _, _ = d.profile(
            "y:x",
            facet_by="facet_col",
            facet_by_bins=3,
            auto_title=True,
        )
        try:
            assert fig._suptitle is not None, (
                "facet_by_bins + auto_title produced no suptitle"
            )
            assert len(fig._suptitle.get_text()) > 0
        finally:
            plt.close(fig)

    def test_X_facet_bins_with_subplot_titles(self, df_facet_data):
        """§9.X.2: facet_by_bins subplot titles show original column name,
        never the internal __dfdraw_facet_bin__ temp name (would have caught
        BUG-001)."""
        d = DFDraw(df_facet_data)
        fig, _, _ = d.profile(
            "y:x",
            facet_by="facet_col",
            facet_by_bins=3,
        )
        try:
            for ax in fig.axes:
                title = ax.get_title()
                if title:
                    assert "__dfdraw_facet_bin__" not in title
                    assert "facet_col" in title
        finally:
            plt.close(fig)

    def test_X_facet_bins_with_group_by_composition(self, df_facet_data):
        """§9.X.3: facet_by_bins + group_by orthogonal composition. Each
        facet shows both group_col groups overlaid. Locks the AD-78
        orthogonal-composition contract."""
        d = DFDraw(df_facet_data)
        fig, _, stats = d.profile(
            "y:x",
            facet_by="facet_col",
            facet_by_bins=3,
            group_by="group_col",
        )
        try:
            assert stats["n_groups"] == 3, (
                f"expected 3 facets, got {stats['n_groups']}"
            )
            # Each subplot should have ≥2 lines (one per group_col value)
            non_empty = [ax for ax in fig.axes if ax.get_lines()]
            assert len(non_empty) >= 3
            for ax in non_empty[:3]:
                assert len(ax.get_lines()) >= 2, (
                    f"facet subplot has only {len(ax.get_lines())} lines, "
                    f"expected ≥2 from group_by overlay"
                )
        finally:
            plt.close(fig)

    def test_X_normalize_with_facet_by_column_mode(self, df_facet_data):
        """§9.X.4: normalize + facet_by column-mode composition. Phase 13.33
        M2 delivered this; locks K×2 grid + per-facet stats."""
        d = DFDraw(df_facet_data)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["group_col == 'a'", "group_col == 'b'"],
            normalize="delta",
            facet_by="group_col",  # categorical column
        )
        try:
            assert stats["n_facets"] == 2
            assert "normalize_data_faceted" in stats
        finally:
            plt.close(fig)

    def test_X_channel_mode_facet_with_auto_title(self, df_facet_data):
        """§9.X.5: channel-mode facet (facet_by='group_by') + auto_title=True
        must not NameError. Locks the v1.5 scope-gap that Phase 13.32 FIX1
        Sonet51 + Sonnet52_R1 caught at spec review (would otherwise have
        shipped with NameError on every channel-mode call)."""
        d = DFDraw(df_facet_data)
        fig, _, _ = d.profile(
            "y:x",
            group_by="group_col",
            facet_by="group_by",  # channel-mode sentinel
            auto_title=True,
        )
        try:
            assert fig._suptitle is not None
        finally:
            plt.close(fig)
