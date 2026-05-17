"""
Phase 13.33.DF Milestone 1 — Normalized Differential Profiles.

Test file scope per PHASE_13_33_DF_v1_1_Proposal_NormalizedDifferentialProfiles.md
section 8 (M1 of the 2-milestone split: M1 = single-curve modes, M2 = group_by /
facet_by composition).

§9-marker convention: each test docstring begins with §9.<class_prefix>.<index>
to identify the invariant locked, mirroring the FIX1.FIX1 file's structure.

22 M1 tests, in 10 classes:
    TestNormalizeDelta             (3) — ND.1..ND.3
    TestNormalizeRatio             (3) — NR.1..NR.3
    TestNormalizeLogRatio          (2) — NL.1..NL.2
    TestNormalizePull              (2) — NP.1..NP.2
    TestNormalizeCallable          (2) — NC.1..NC.2
    TestNormalizeLayout            (3) — NLY.1..NLY.3
    TestNormalizeBackwardCompat    (1) — NBC.1
    TestNormalizeValidation        (3) — NV.1..NV.3
    TestNormalizeSignConvention    (1) — NSC.1 (AD-80 lock)
    TestNormalizeSingleYConvention (2) — NSY.1..NSY.2 (§6 directive lock —
                                          closes Phase 13.27 FIX1.FIX1 deferral)
"""
import math
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfdraw import DFDraw


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def df_two_sectors():
    """Synthetic two-sector dataset.

    Both sectors: y = 2x + N(0, 1) per-event; sector 1 adds a constant +0.5.
    Each sector has 2000 independent samples drawn from Uniform(0, 10) on x.

    Therefore the per-bin delta = sector0_mean − sector1_mean ≈ −0.5 EXACTLY
    across all x bins (the 2x trend cancels per-bin between the two curves).
    Ratio mean ≈ (2x_bin + 0) / (2x_bin + 0.5) ≈ 1.0 averaged over x ∈ [0, 10].
    Log_ratio mean ≈ 0 (small negative since signal slightly below reference).
    Pull values: large-σ deviation since the 0.5 offset is meaningful at this n.
    """
    np.random.seed(42)
    n_per = 2000
    x_s0 = np.random.uniform(0, 10, n_per)
    x_s1 = np.random.uniform(0, 10, n_per)
    y_s0 = 2.0 * x_s0 + np.random.normal(0, 1, n_per)
    y_s1 = 2.0 * x_s1 + 0.5 + np.random.normal(0, 1, n_per)
    df = pd.DataFrame({
        "x": np.concatenate([x_s0, x_s1]),
        "y": np.concatenate([y_s0, y_s1]),
        "sector": np.concatenate([np.zeros(n_per), np.ones(n_per)]).astype(int),
    })
    return df


@pytest.fixture
def df_deterministic():
    """Deterministic data for AD-80 sign-convention lock.

    Per-bin means: signal_y - reference_y = +1.0 exactly, no noise.
    Used for lock tests where we need exact equality, not a tolerance match.
    """
    n_per = 500
    np.random.seed(1)
    x = np.random.uniform(0, 10, n_per)
    df = pd.DataFrame({
        "x": np.concatenate([x, x]),
        "y": np.concatenate([
            5.0 + 0.0 * x,   # constant signal at y=5
            4.0 + 0.0 * x,   # constant reference at y=4 → delta = +1.0
        ]),
        "tag": np.concatenate([
            np.full(n_per, "A"),
            np.full(n_per, "B"),
        ]),
    })
    return df


# =============================================================================
# TestNormalizeDelta — basic delta mode (3 tests)
# =============================================================================

class TestNormalizeDelta:
    """§9.ND.* — delta mode (signal − reference) and SEM error propagation."""

    def test_ND_1_basic_delta(self, df_two_sectors):
        """§9.ND.1: normalize='delta' produces a bottom panel whose values
        recover the known per-bin offset between two sector populations.

        Synthetic data: sector1 shifted +0.5 vs sector0 → delta mean ≈ −0.5.
        """
        d = DFDraw(df_two_sectors)
        fig, ax_top, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            bins=20,
        )
        try:
            nd = stats["normalize_data"]
            delta_mean = nd["value"].mean()
            # Synthetic offset is 0.5, but each sector's draw_uniform pulls
            # were independent so per-bin x distributions differ slightly.
            # Use a 0.2 tolerance — broad enough for stochastic variation,
            # tight enough to lock the SIGN and order of magnitude.
            assert -0.7 < delta_mean < -0.3, (
                f"delta mean {delta_mean:.4f} outside expected [-0.7, -0.3] "
                f"for 0.5-offset synthetic"
            )
            assert stats["normalize_mode"] == "delta"
        finally:
            plt.close(fig)

    def test_ND_2_delta_error_formula(self, df_two_sectors):
        """§9.ND.2: per-bin delta errors equal sqrt(SEM_0² + SEM_1²)
        within numerical tolerance. Locks the SEM-based propagation in
        _compute_normalize_transform.
        """
        d = DFDraw(df_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            bins=20,
        )
        try:
            nd = stats["normalize_data"]
            # Reconstruct expected error from per-curve stats stored in the
            # normalize_data DataFrame.
            sig_sigma = nd["signal_sigma"].to_numpy()
            sig_count = nd["signal_count"].to_numpy()
            ref_sigma = nd["reference_sigma"].to_numpy()
            ref_count = nd["reference_count"].to_numpy()
            # SEM² = σ² / n; combined error² = sem_s² + sem_r²
            with np.errstate(divide="ignore", invalid="ignore"):
                sem2_s = (sig_sigma ** 2) / np.where(sig_count > 0, sig_count, 1)
                sem2_r = (ref_sigma ** 2) / np.where(ref_count > 0, ref_count, 1)
                expected_err = np.sqrt(sem2_s + sem2_r)
            actual_err = nd["error"].to_numpy()
            # Compare only on bins where both have entries.
            valid = (sig_count > 0) & (ref_count > 0)
            assert valid.sum() > 0, "no overlapping bins to verify"
            np.testing.assert_allclose(
                actual_err[valid], expected_err[valid],
                rtol=1e-10,
                err_msg="delta error must match sqrt(SEM_0^2 + SEM_1^2)",
            )
        finally:
            plt.close(fig)

    def test_ND_3_stats_dict_structure(self, df_two_sectors):
        """§9.ND.3: stats dict structure per AD-81 contract.

        Required keys: ax_diff, normalize_data (DataFrame), normalize_mode,
        normalize_layout, n_masked_bins, n_total_bins.
        normalize_data must have columns: x_center, value, error,
        mask_undefined, signal_central, signal_sigma, signal_count,
        reference_central, reference_sigma, reference_count.
        """
        d = DFDraw(df_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            bins=20,
        )
        try:
            for key in (
                "ax_diff", "normalize_data", "normalize_mode",
                "normalize_layout", "n_masked_bins", "n_total_bins",
            ):
                assert key in stats, f"stats dict missing required key {key!r}"
            assert isinstance(stats["normalize_data"], pd.DataFrame)
            for col in (
                "x_center", "value", "error", "mask_undefined",
                "signal_central", "signal_sigma", "signal_count",
                "reference_central", "reference_sigma", "reference_count",
            ):
                assert col in stats["normalize_data"].columns, (
                    f"normalize_data missing column {col!r}"
                )
            # profile_data must NOT leak — it is internal-only per Q3
            # (architect+reviewer panel; stripped before return).
            assert "profile_data" not in stats, (
                "profile_data must be stripped from user-facing stats dict"
            )
        finally:
            plt.close(fig)


# =============================================================================
# TestNormalizeRatio — basic ratio mode (3 tests)
# =============================================================================

class TestNormalizeRatio:
    """§9.NR.* — ratio mode (signal / reference) + masking + error propagation."""

    def test_NR_1_basic_ratio(self, df_two_sectors):
        """§9.NR.1: normalize='ratio' produces values close to 1.0 when
        signal and reference have similar means.
        """
        d = DFDraw(df_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="ratio",
            bins=20,
        )
        try:
            ratio_mean = stats["normalize_data"]["value"].mean()
            # Sectors differ only by a 0.5 shift on means in the [0, 20] range
            # → ratio close to but slightly below 1.0.
            assert 0.85 < ratio_mean < 1.05, (
                f"ratio mean {ratio_mean:.4f} outside expected [0.85, 1.05]"
            )
            assert stats["normalize_mode"] == "ratio"
        finally:
            plt.close(fig)

    def test_NR_2_zero_denominator_masked(self):
        """§9.NR.2: bins where the reference mean is zero produce
        mask_undefined=True (the ratio is mathematically undefined there).
        """
        # Construct data where reference mean is exactly zero in some bin.
        np.random.seed(3)
        n = 500
        x = np.random.uniform(0, 10, n)
        df = pd.DataFrame({
            "x": np.concatenate([x, x]),
            "y": np.concatenate([
                np.full(n, 5.0),         # signal: constant 5
                np.zeros(n),             # reference: ALL zeros → mean = 0
            ]),
            "tag": np.concatenate([np.full(n, "A"), np.full(n, "B")]),
        })
        d = DFDraw(df)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["tag == 'A'", "tag == 'B'"],
            normalize="ratio",
            bins=10,
        )
        try:
            mask = stats["normalize_data"]["mask_undefined"].to_numpy()
            # Every populated bin should be masked (ref mean = 0 → ratio undef).
            counts = stats["normalize_data"]["reference_count"].to_numpy()
            populated = counts > 0
            assert mask[populated].all(), (
                "all bins with mu_ref=0 must be flagged mask_undefined"
            )
            assert stats["n_masked_bins"] >= populated.sum()
        finally:
            plt.close(fig)

    def test_NR_3_ratio_error_formula(self, df_two_sectors):
        """§9.NR.3: ratio errors match |μ₀/μ₁|·sqrt(σ₀²/(n₀·μ₀²) + σ₁²/(n₁·μ₁²))
        within numerical tolerance."""
        d = DFDraw(df_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="ratio",
            bins=20,
        )
        try:
            nd = stats["normalize_data"]
            mu_s = nd["signal_central"].to_numpy()
            mu_r = nd["reference_central"].to_numpy()
            sig_s = nd["signal_sigma"].to_numpy()
            n_s = nd["signal_count"].to_numpy()
            sig_r = nd["reference_sigma"].to_numpy()
            n_r = nd["reference_count"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_var_s = (sig_s ** 2) / np.where(
                    (mu_s != 0) & (n_s > 0), n_s * mu_s ** 2, 1
                )
                rel_var_r = (sig_r ** 2) / np.where(
                    (mu_r != 0) & (n_r > 0), n_r * mu_r ** 2, 1
                )
                expected = np.abs(mu_s / mu_r) * np.sqrt(rel_var_s + rel_var_r)
            actual = nd["error"].to_numpy()
            valid = (
                (n_s > 0) & (n_r > 0) & (mu_s != 0) & (mu_r != 0)
                & np.isfinite(expected) & np.isfinite(actual)
            )
            assert valid.sum() > 0, "no overlapping non-zero-mean bins"
            np.testing.assert_allclose(
                actual[valid], expected[valid],
                rtol=1e-10,
                err_msg="ratio error formula mismatch vs analytical",
            )
        finally:
            plt.close(fig)


# =============================================================================
# TestNormalizeLogRatio — log_ratio mode (2 tests)
# =============================================================================

class TestNormalizeLogRatio:
    """§9.NL.* — log_ratio mode (ln(signal/reference)) and non-positive masking."""

    def test_NL_1_basic_log_ratio(self, df_two_sectors):
        """§9.NL.1: normalize='log_ratio' on positive-mean data returns
        values near zero (since both sectors have similar means)."""
        d = DFDraw(df_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="log_ratio",
            bins=20,
        )
        try:
            lr_mean = stats["normalize_data"]["value"].mean()
            # ln(0.95) ≈ −0.05, ln(1.05) ≈ +0.05 — close to zero
            assert -0.3 < lr_mean < 0.3, (
                f"log_ratio mean {lr_mean:.4f} outside expected [-0.3, 0.3]"
            )
            assert stats["normalize_mode"] == "log_ratio"
        finally:
            plt.close(fig)

    def test_NL_2_non_positive_mean_masked(self):
        """§9.NL.2: bins where signal or reference mean is ≤ 0 produce
        mask_undefined=True (log is undefined for non-positive arguments)."""
        np.random.seed(4)
        n = 500
        x = np.random.uniform(0, 10, n)
        df = pd.DataFrame({
            "x": np.concatenate([x, x]),
            "y": np.concatenate([
                np.full(n, 5.0),        # positive signal
                np.full(n, -3.0),       # negative reference → log undef
            ]),
            "tag": np.concatenate([np.full(n, "A"), np.full(n, "B")]),
        })
        d = DFDraw(df)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["tag == 'A'", "tag == 'B'"],
            normalize="log_ratio",
            bins=10,
        )
        try:
            mask = stats["normalize_data"]["mask_undefined"].to_numpy()
            counts = stats["normalize_data"]["reference_count"].to_numpy()
            populated = counts > 0
            assert mask[populated].all(), (
                "all bins with mu_ref<0 must be flagged mask_undefined for log_ratio"
            )
        finally:
            plt.close(fig)


# =============================================================================
# TestNormalizePull — pull mode + bands (2 tests)
# =============================================================================

class TestNormalizePull:
    """§9.NP.* — pull mode and the ±1σ/±2σ band rendering."""

    def test_NP_1_basic_pull_dimensionless(self, df_two_sectors):
        """§9.NP.1: pull values are dimensionless (in units of σ) and errors
        are 1.0 by construction per AD-82."""
        d = DFDraw(df_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="pull",
            bins=20,
        )
        try:
            nd = stats["normalize_data"]
            # Pull errors must be exactly 1.0 (AD-82 — pull is already in σ).
            valid = (nd["signal_count"] > 0) & (nd["reference_count"] > 0)
            errs = nd["error"][valid].to_numpy()
            assert np.allclose(errs, 1.0), (
                f"pull errors must all be 1.0, got {errs[:5]}"
            )
            assert stats["normalize_mode"] == "pull"
        finally:
            plt.close(fig)

    def test_NP_2_pull_bands_rendered(self, df_two_sectors):
        """§9.NP.2: pull mode renders ±1σ and ±2σ horizontal bands on the
        diff panel via axhspan calls. We verify at least 3 patches are added
        to the diff axes (one per axhspan call: ±1σ inner, ±2σ above, ±2σ below).
        We compare counts vs a non-pull baseline to account for any baseline
        patches the panel may include."""
        d = DFDraw(df_two_sectors)
        # Baseline: delta mode renders NO bands (only the differential curve
        # + the reference line, neither of which adds a patch).
        fig_base, _, stats_base = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            bins=20,
        )
        baseline_patch_count = len(stats_base["ax_diff"].patches)
        plt.close(fig_base)

        # Pull mode must add at least 3 more patches (the bands).
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="pull",
            bins=20,
        )
        try:
            pull_patch_count = len(stats["ax_diff"].patches)
            assert pull_patch_count >= baseline_patch_count + 3, (
                f"pull mode must add >=3 axhspan patches over baseline; "
                f"baseline={baseline_patch_count}, pull={pull_patch_count}"
            )
        finally:
            plt.close(fig)


# =============================================================================
# TestNormalizeCallable — callable mode (2 tests)
# =============================================================================

class TestNormalizeCallable:
    """§9.NC.* — callable normalize= accepting a 2-arg function."""

    def test_NC_1_callable_with_errors(self, df_two_sectors):
        """§9.NC.1: callable returning (values, errors) tuple — both arrays
        flow through to the rendered normalize_data."""

        def my_transform(s0, s1):
            values = s0["central"] - s1["central"]
            errors = np.full_like(values, 0.123)  # constant marker for verification
            return values, errors

        d = DFDraw(df_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize=my_transform,
            bins=20,
        )
        try:
            nd = stats["normalize_data"]
            valid = (nd["signal_count"] > 0) & (nd["reference_count"] > 0)
            # Where both curves have entries the error should be exactly 0.123.
            errs = nd.loc[valid, "error"].to_numpy()
            assert np.allclose(errs, 0.123), (
                f"callable errors must propagate; got {errs[:5]}"
            )
            assert stats["normalize_mode"] == "callable"
        finally:
            plt.close(fig)

    def test_NC_2_callable_values_only(self, df_two_sectors):
        """§9.NC.2: callable returning values array alone — errors propagate
        as NaN array (rendered without error bars)."""

        def values_only(s0, s1):
            return s0["central"] - s1["central"]  # no tuple, no errors

        d = DFDraw(df_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize=values_only,
            bins=20,
        )
        try:
            nd = stats["normalize_data"]
            # Errors should be all-NaN when callable returned values only.
            assert nd["error"].isna().all(), (
                "callable values-only mode must yield NaN error column"
            )
        finally:
            plt.close(fig)


# =============================================================================
# TestNormalizeLayout — overlay+diff vs diff_only (3 tests)
# =============================================================================

class TestNormalizeLayout:
    """§9.NLY.* — figure layout per normalize_layout kwarg."""

    def test_NLY_1_overlay_diff_creates_two_panels(self, df_two_sectors):
        """§9.NLY.1: default 'overlay+diff' produces a figure with exactly
        2 axes (top overlay + bottom diff)."""
        d = DFDraw(df_two_sectors)
        fig, ax_top, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            bins=20,
            normalize_layout="overlay+diff",
        )
        try:
            assert len(fig.axes) == 2, (
                f"overlay+diff must yield 2 axes, got {len(fig.axes)}"
            )
            assert stats["ax_diff"] is not None
            assert ax_top is not stats["ax_diff"], (
                "ax_top and ax_diff must be distinct axes"
            )
        finally:
            plt.close(fig)

    def test_NLY_2_diff_only_single_panel(self, df_two_sectors):
        """§9.NLY.2: 'diff_only' produces a figure with exactly 1 axes."""
        d = DFDraw(df_two_sectors)
        fig, ax, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            bins=20,
            normalize_layout="diff_only",
        )
        try:
            assert len(fig.axes) == 1, (
                f"diff_only must yield 1 axes, got {len(fig.axes)}"
            )
            # Per AD-81: in diff_only the returned 'ax_top' slot IS the
            # diff axes (the only panel the user reads).
            assert stats["ax_diff"] is ax, (
                "in diff_only, returned ax_top must equal stats['ax_diff']"
            )
        finally:
            plt.close(fig)

    def test_NLY_3_height_ratio_style_respected(self, df_two_sectors):
        """§9.NLY.3: 'normalize.panel.height_ratio' style key controls the
        gridspec height ratios. Setting a custom ratio is reflected in the
        rendered figure."""
        from dfdraw.style import set_style, get_style_value
        # Save original to restore after test.
        original = get_style_value("normalize.panel.height_ratio")
        try:
            set_style({"normalize.panel.height_ratio": [5, 1]})
            d = DFDraw(df_two_sectors)
            fig, _, _ = d.profile(
                "y:x",
                selection_vector=["sector == 0", "sector == 1"],
                normalize="delta",
                bins=20,
            )
            try:
                # The first axes should be ~5x taller than the second.
                bbox_top = fig.axes[0].get_position()
                bbox_diff = fig.axes[1].get_position()
                ratio = bbox_top.height / bbox_diff.height
                # Allow slack — gridspec adds some shared whitespace etc.
                assert 3.5 < ratio < 7.0, (
                    f"height ratio {ratio:.2f} not in [3.5, 7.0] for [5,1] setting"
                )
            finally:
                plt.close(fig)
        finally:
            set_style({"normalize.panel.height_ratio": original})


# =============================================================================
# TestNormalizeBackwardCompat — normalize=None preserves existing behavior (1)
# =============================================================================

class TestNormalizeBackwardCompat:
    """§9.NBC.* — Phase 13.33 must be strictly additive to existing API."""

    def test_NBC_1_default_normalize_none_unchanged(self, df_two_sectors):
        """§9.NBC.1: profile() with normalize=None (default) returns the
        usual (fig, ax, stats) with the SAME stats keys as pre-Phase-13.33.
        No 'ax_diff', 'normalize_data', 'normalize_mode', etc."""
        d = DFDraw(df_two_sectors)
        fig, ax, stats = d.profile("y:x", selection="sector == 0", bins=20)
        try:
            for forbidden in (
                "ax_diff", "normalize_data", "normalize_mode",
                "normalize_layout", "n_masked_bins",
            ):
                assert forbidden not in stats, (
                    f"normalize-related key {forbidden!r} must NOT appear in "
                    f"the stats dict when normalize=None (additive contract)"
                )
        finally:
            plt.close(fig)


# =============================================================================
# TestNormalizeValidation — 3 raise paths (3 tests)
# =============================================================================

class TestNormalizeValidation:
    """§9.NV.* — validation guards at the public entry."""

    def test_NV_1_wrong_vector_count_raises(self, df_two_sectors):
        """§9.NV.1: normalize= requires exactly 2 vector elements
        (signal + reference). 3 or more raises ValueError."""
        d = DFDraw(df_two_sectors)
        with pytest.raises(ValueError, match=r"exactly 2 vector elements"):
            d.profile(
                "y:x",
                selection_vector=["sector == 0", "sector == 1", "sector == 2"],
                normalize="delta",
            )

    def test_NV_2_invalid_mode_string_raises(self, df_two_sectors):
        """§9.NV.2: normalize=string not in NORMALIZE_MODES raises ValueError
        listing the valid modes."""
        d = DFDraw(df_two_sectors)
        with pytest.raises(ValueError, match=r"normalize must be one of"):
            d.profile(
                "y:x",
                selection_vector=["sector == 0", "sector == 1"],
                normalize="bogus_mode",
            )

    def test_NV_3_same_true_with_normalize_raises(self, df_two_sectors):
        """§9.NV.3: normalize is mutually exclusive with same=True
        — the diff panel needs its own figure."""
        d = DFDraw(df_two_sectors)
        with pytest.raises(ValueError, match=r"mutually exclusive with same=True"):
            d.profile(
                "y:x",
                selection_vector=["sector == 0", "sector == 1"],
                normalize="delta",
                same=True,
            )


# =============================================================================
# TestNormalizeSignConvention — AD-80 lock (1 test)
# =============================================================================

class TestNormalizeSignConvention:
    """§9.NSC.* — AD-80: vector[0] = signal, vector[1] = reference;
    delta = vector[0] − vector[1] (positive when signal > reference)."""

    def test_NSC_1_signal_above_reference_delta_positive(self, df_deterministic):
        """§9.NSC.1 (AD-80 LOCK): when vector[0] mean > vector[1] mean,
        delta values are positive. Locks the sign convention against
        accidental swap during refactor."""
        d = DFDraw(df_deterministic)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["tag == 'A'", "tag == 'B'"],
            normalize="delta",
            bins=10,
        )
        try:
            nd = stats["normalize_data"]
            valid = (nd["signal_count"] > 0) & (nd["reference_count"] > 0)
            values = nd.loc[valid, "value"].to_numpy()
            # Signal=5, reference=4 → delta=+1.0 exactly per bin
            assert (values > 0).all(), (
                f"AD-80 SIGN VIOLATION: signal > reference must yield positive "
                f"delta. Got values[:5]={values[:5]}"
            )
            # Tighter check: should be approximately +1.0
            np.testing.assert_allclose(
                values, 1.0, atol=1e-10,
                err_msg="deterministic delta must be exactly +1.0",
            )
        finally:
            plt.close(fig)


# =============================================================================
# TestNormalizeSingleYConvention — §6 directive lock (2 tests)
# =============================================================================
# These tests lock the Phase 13.27 FIX1.FIX1 §6 closure: when normalize is set
# on a single-Y expression with a 2-element selection_vector, vector_compose
# is forced to 'outer' transparently. User writes normalize='delta' without
# any vector_compose= boilerplate. This is convention application of the
# scalar→vector mapping from Phase 13.27 FIX1 (architect framing).

class TestNormalizeSingleYConvention:
    """§9.NSY.* — Phase 13.27 FIX1.FIX1 §6 deferred closure (option c)."""

    def test_NSY_1_single_y_default_inner_works(self, df_two_sectors):
        """§9.NSY.1 (§6 DIRECTIVE LOCK): single-Y + selection_vector + default
        vector_compose (inner) + normalize= must NOT raise. Pre-FIX1.FIX1 this
        raised 'inner requires equal lengths: vector=1, selection_vector=2';
        the §6 directive transparently forces vector_compose='outer' when
        normalize is set, so the canonical Phase 13.33 call works without
        any compose boilerplate."""
        d = DFDraw(df_two_sectors)
        # Note: NO vector_compose= passed; default is 'inner'.
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            bins=20,
        )
        try:
            # The call succeeded — that alone is the locked behavior.
            # Additionally verify the result is sensible (delta in expected range).
            delta_mean = stats["normalize_data"]["value"].mean()
            assert -0.7 < delta_mean < -0.3, (
                f"§6 directive should produce same result as explicit outer; "
                f"got delta mean {delta_mean:.4f}"
            )
        finally:
            plt.close(fig)

    def test_NSY_2_single_y_explicit_outer_equivalent(self, df_two_sectors):
        """§9.NSY.2 (§6 DIRECTIVE IDEMPOTENCY): the §6 directive must produce
        the same result whether the user defaults to inner OR explicitly
        passes vector_compose='outer'. Tests idempotency of the directive."""
        d = DFDraw(df_two_sectors)
        # Implicit (directive applies): default inner.
        fig_a, _, stats_a = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            bins=20,
        )
        # Explicit outer (directive is a no-op since user picked outer).
        fig_b, _, stats_b = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            vector_compose="outer",
            bins=20,
        )
        try:
            va = stats_a["normalize_data"]["value"].to_numpy()
            vb = stats_b["normalize_data"]["value"].to_numpy()
            np.testing.assert_allclose(
                va, vb, equal_nan=True, rtol=1e-12,
                err_msg=(
                    "§6 directive (default inner) must produce byte-identical "
                    "differential values to explicit vector_compose='outer'"
                ),
            )
        finally:
            plt.close(fig_a)
            plt.close(fig_b)


# =============================================================================
# Phase 13.33.DF M2 — group_by + normalize composition
# =============================================================================

@pytest.fixture
def df_three_fills_two_sectors():
    """Synthetic data: 3 fills × 2 sectors. Same per-sector shift across fills.

    Within each fill_id, sector 1 mean is +0.5 above sector 0.
    Therefore delta = sector0 − sector1 ≈ −0.5 for EVERY fill.
    Different from df_two_sectors fixture in that it has the group_by column.
    """
    np.random.seed(7)
    n_per_cell = 1000
    records = []
    for fill_id in (41, 42, 43):
        for sector in (0, 1):
            x_arr = np.random.uniform(0, 10, n_per_cell)
            y_arr = (
                2.0 * x_arr
                + (0.5 if sector == 1 else 0.0)
                + np.random.normal(0, 1, n_per_cell)
            )
            for x_v, y_v in zip(x_arr, y_arr):
                records.append({
                    "x": x_v, "y": y_v,
                    "sector": sector,
                    "fill_id": fill_id,
                })
    return pd.DataFrame(records)


class TestNormalizeGroupBy:
    """§9.NG.* — group_by + normalize composition (Phase 13.33.DF M2)."""

    def test_NG_1_basic_group_by_3_fills(self, df_three_fills_two_sectors):
        """§9.NG.1: group_by='fill_id' on 3-fill dataset produces 3 differential
        curves (one per group) and 3-entry per-group stats dict."""
        d = DFDraw(df_three_fills_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            group_by="fill_id",
            bins=20,
        )
        try:
            assert stats["n_groups"] == 3, (
                f"expected 3 groups, got {stats['n_groups']}"
            )
            grouped = stats["normalize_data_grouped"]
            assert isinstance(grouped, dict)
            assert len(grouped) == 3
            # Each group must have per-bin values + errors arrays
            for g_key, g_stats in grouped.items():
                for arr_key in (
                    "values", "errors", "mask_undefined", "bin_centers",
                    "signal_central", "signal_sigma", "signal_count",
                    "reference_central", "reference_sigma", "reference_count",
                ):
                    assert arr_key in g_stats, (
                        f"group {g_key!r} missing key {arr_key!r}"
                    )
        finally:
            plt.close(fig)

    def test_NG_2_per_group_delta_recovers_offset(
        self, df_three_fills_two_sectors
    ):
        """§9.NG.2: each group's delta mean should recover the same per-sector
        offset (≈ −0.5) — the differential machinery applied per-group must
        produce per-group physics consistent with the data construction."""
        d = DFDraw(df_three_fills_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            group_by="fill_id",
            bins=20,
        )
        try:
            for g_key, g_stats in stats["normalize_data_grouped"].items():
                v = g_stats["values"]
                v_mean = np.nanmean(v)
                # Per-fill delta should track the −0.5 sector offset within
                # statistical noise (n=1000/sector → SEM ≈ 0.03 per fill).
                assert -0.8 < v_mean < -0.2, (
                    f"group {g_key!r}: per-group delta mean {v_mean:.4f} "
                    f"not in [-0.8, -0.2] (expected ≈ -0.5)"
                )
        finally:
            plt.close(fig)

    def test_NG_3_stats_dict_grouped_structure(
        self, df_three_fills_two_sectors
    ):
        """§9.NG.3: M2 grouped stats dict structure.

        Required keys differ from M1: 'group_by', 'n_groups',
        'normalize_data_grouped' instead of 'normalize_data'.
        Per-group entries are themselves dicts containing the same per-bin
        arrays as M1's normalize_data DataFrame (but as numpy arrays, not
        DataFrame — caller can construct DataFrame from any group's entry)."""
        d = DFDraw(df_three_fills_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            group_by="fill_id",
            bins=20,
        )
        try:
            for key in (
                "group_by", "n_groups", "normalize_data_grouped",
                "normalize_mode", "normalize_layout", "ax_diff",
            ):
                assert key in stats, f"M2 grouped stats missing {key!r}"
            assert stats["group_by"] == "fill_id"
            # M1's flat 'normalize_data' key must NOT appear in grouped mode
            # (different contract).
            assert "normalize_data" not in stats, (
                "M1 flat normalize_data must not appear in grouped contract"
            )
        finally:
            plt.close(fig)


# =============================================================================
# Phase 13.33.DF M2 — facet_by + normalize composition
# =============================================================================

class TestNormalizeFacetBy:
    """§9.NF.* — facet_by + normalize composition (Phase 13.33.DF M2)."""

    def test_NF_1_k_by_2_grid(self, df_three_fills_two_sectors):
        """§9.NF.1: facet_by='fill_id' (3 fills) produces a K×2 grid →
        2*K = 6 axes total in the figure. Each facet column has a top panel
        (signal+reference overlay) and a diff panel."""
        d = DFDraw(df_three_fills_two_sectors)
        fig, returned_top, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            facet_by="fill_id",
            bins=20,
        )
        try:
            assert stats["n_facets"] == 3
            assert len(fig.axes) == 6, (
                f"K=3 facets × 2 panels each = 6 axes; got {len(fig.axes)}"
            )
            # Returned top is a list of top-panel axes, one per facet.
            assert isinstance(returned_top, list), (
                "facet+normalize must return a LIST of top axes"
            )
            assert len(returned_top) == 3
            # The stats dict's ax_diffs is a parallel list.
            assert isinstance(stats["ax_diffs"], list)
            assert len(stats["ax_diffs"]) == 3
        finally:
            plt.close(fig)

    def test_NF_2_per_facet_independent_computation(
        self, df_three_fills_two_sectors
    ):
        """§9.NF.2: each facet's differential is computed from its OWN
        subset, not from the union. The construction guarantees ≈ −0.5
        within every fill; verifying the per-facet delta means cluster
        around −0.5 confirms that the facet partitioning happens BEFORE
        the normalize computation (not after)."""
        d = DFDraw(df_three_fills_two_sectors)
        fig, _, stats = d.profile(
            "y:x",
            selection_vector=["sector == 0", "sector == 1"],
            normalize="delta",
            facet_by="fill_id",
            bins=20,
        )
        try:
            faceted = stats["normalize_data_faceted"]
            assert len(faceted) == 3
            for f_key, f_stats in faceted.items():
                v_mean = np.nanmean(f_stats["values"])
                assert -0.8 < v_mean < -0.2, (
                    f"facet {f_key!r}: per-facet delta mean {v_mean:.4f} "
                    f"not in expected range [-0.8, -0.2]"
                )
        finally:
            plt.close(fig)

    def test_NF_3_facet_by_bins_with_normalize_raises_with_workaround_hint(
        self, df_three_fills_two_sectors
    ):
        """§9.NF.3 (M2 v1.0 deferred-behavior lock — closes Coder QRC Rule 14
        gap caught by Sonnet52_R1 + Sonnet53_R2 panel review).

        Combining ``normalize=`` with ``facet_by_bins`` or
        ``facet_by_quantiles`` raises NotImplementedError rather than
        silently falling back to categorical facets or producing incorrect
        output. M2 v1.0 supports categorical-column facets only; auto-binning
        of the facet variable composing with normalize is reserved for a
        future fix-up phase.

        The error message must direct the user to the available workaround
        (pre-bin into a categorical column) so they aren't left guessing —
        Phase 13.16.DF actionable-error convention. This second assertion
        protects against silent message regression in future refactors.
        """
        d = DFDraw(df_three_fills_two_sectors)
        with pytest.raises(NotImplementedError, match=r"facet_by_bins") as exc:
            d.profile(
                "y:x",
                selection_vector=["sector == 0", "sector == 1"],
                normalize="delta",
                facet_by="fill_id",
                facet_by_bins=3,
            )
        msg = str(exc.value)
        # Workaround-hint lock: message must name the categorical workaround
        # so users discover the path forward from the error alone.
        assert "categorical" in msg.lower(), (
            f"NotImplementedError must direct user to the categorical-column "
            f"workaround; got message: {msg!r}"
        )

        # Same lock for facet_by_quantiles (the other auto-binning path).
        with pytest.raises(NotImplementedError, match=r"facet_by_quantiles") as exc:
            d.profile(
                "y:x",
                selection_vector=["sector == 0", "sector == 1"],
                normalize="delta",
                facet_by="fill_id",
                facet_by_quantiles=3,
            )
        assert "categorical" in str(exc.value).lower()

