"""
test_smoke_pdf_estimator.py — Phase 13.11.DF v2.1

Smoke tests for:
  - 3-layer PDF estimator (kernel + Poisson + local polynomial)
  - Threshold-based efficiency sampling (architect's algorithm)

Self-contained: generates test data via pytest fixture.
N=10,000, 50 iterations, Δx=0.1. Runs in <10 seconds.

Tests:
  S1:  Pipeline sanity (no NaN/Inf, weight_raw = 1/max(pdf, threshold))
  S2:  Integral (Σcw → N_orig)
  S3:  Positivity (all PDF ≥ 0)
  S4:  Empty bins → PDF=0 (AD-11)
  S5:  Correction reduces bias (key test)
  S6:  Poisson correction bounded [0, 1]
  S7:  Kernel counts non-negative
  S8:  Polynomial fallback at edges
  S9:  Reconstruction in core (threshold sampling)
  S10: Parameter scalar and list
  S11: frac/threshold mutual exclusion
  S12: Threshold ↔ frac roundtrip
  S13: pdf_func replaces empirical PDF
"""

import numpy as np
import pandas as pd
import pytest

from dfextensions.sampling.downsample import (
    downsampleDFSmoothFactorized,
    downsampleDFSmooth,
    _estimate_pdf_smooth_1d,
    _estimate_binned_pdf,
    _normalize_per_dim_param,
    _threshold_from_frac,
    _frac_from_threshold,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def gauss_data():
    """Generate Gaussian test data: N=10000, 50 iterations."""
    rng = np.random.RandomState(42)
    N = 10_000
    n_iter = 50
    sigma = 1.0

    all_x = []
    all_iter = []
    for it in range(n_iter):
        x = rng.normal(0, sigma, N)
        x = x[(x >= -6) & (x <= 6)][:N]
        while len(x) < N:
            extra = rng.normal(0, sigma, N - len(x))
            extra = extra[(extra >= -6) & (extra <= 6)]
            x = np.concatenate([x, extra])[:N]
        all_x.append(x)
        all_iter.append(np.full(N, it))

    df = pd.DataFrame({
        "x": np.concatenate(all_x),
        "iteration": np.concatenate(all_iter).astype(int),
    })
    return df, N, n_iter, sigma


@pytest.fixture(scope="module")
def bin_edges():
    return np.linspace(-6, 6, 121)  # 120 bins, Δx=0.1


@pytest.fixture(scope="module")
def pdf_true_func():
    def f(x, sigma=1.0):
        return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    return f


# =============================================================================
# S1: Pipeline sanity
# =============================================================================

class TestS1PipelineSanity:
    """S1: Columns exist, no NaN/Inf, debug columns consistent."""

    def test_factorized_no_nan(self, gauss_data):
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        variables = {"x": (120, -6, 6)}
        result = downsampleDFSmoothFactorized(
            sub, variables, 42, frac=0.1,
            debug=True, pdf_params={"poly_order": 2, "poly_half_range": 0.5},
        )
        assert not result["_debug_pdf"].isna().any()
        assert not result["_debug_weight_raw"].isna().any()
        assert np.all(np.isfinite(result["_debug_pdf"].values))
        assert np.all(np.isfinite(result["_debug_weight_raw"].values))

    def test_weight_raw_bounded(self, gauss_data):
        """weight_raw = 1/max(pdf, threshold) → product pdf*wr ≤ 1."""
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        variables = {"x": (120, -6, 6)}
        result = downsampleDFSmoothFactorized(
            sub, variables, 42, frac=0.1,
            debug=True, pdf_params={"poly_order": 2, "poly_half_range": 0.5},
        )
        pdf = result["_debug_pdf"].values
        wr = result["_debug_weight_raw"].values
        # For threshold sampling: wr = 1/max(pdf, threshold)
        # So pdf * wr ≤ 1 (equality when pdf ≥ threshold)
        product = pdf * wr
        assert np.all(product <= 1.0 + 1e-10)
        assert np.all(product > 0)


# =============================================================================
# S2: Integral
# =============================================================================

class TestS2Integral:
    """S2: Σcw approaches N_orig (statistical, not exact for threshold sampling)."""

    def test_integral_approximate(self, gauss_data):
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        variables = {"x": (120, -6, 6)}
        result = downsampleDFSmoothFactorized(
            sub, variables, 42, frac=0.1,
            debug=True, pdf_params={"poly_order": 2, "poly_half_range": 0.5},
        )
        wr = result["_debug_weight_raw"].values
        ratio = wr.sum() / len(sub)
        # For threshold sampling, Σcw ≈ N_orig (in expectation, not exact)
        assert 0.5 < ratio < 2.0, f"Σcw/N_orig = {ratio:.4f}, expected ~1.0"


# =============================================================================
# S3: Positivity
# =============================================================================

class TestS3Positivity:
    def test_smooth_1d_positive(self, gauss_data, bin_edges):
        df, N, n_iter, sigma = gauss_data
        x = df[df["iteration"] == 0]["x"].values.astype(np.float64)
        centers, pdf = _estimate_pdf_smooth_1d(
            x, bin_edges, kernel_sigma_bins=0.5, poly_order=2, poly_half_range=0.5
        )
        assert np.all(pdf >= 0)

    def test_binned_positive(self, gauss_data, bin_edges):
        df, N, n_iter, sigma = gauss_data
        x = df[df["iteration"] == 0]["x"].values.astype(np.float64)
        centers, pdf, counts = _estimate_binned_pdf(x, bin_edges, bias_correction=True)
        assert np.all(pdf >= 0)


# =============================================================================
# S4: Empty bins → PDF=0 (AD-11)
# =============================================================================

class TestS4EmptyBins:
    def test_empty_bins_zero(self, gauss_data, bin_edges):
        df, N, n_iter, sigma = gauss_data
        x = df[df["iteration"] == 0]["x"].values.astype(np.float64)
        centers, pdf_uncorr, counts = _estimate_binned_pdf(x, bin_edges, bias_correction=False)
        centers, pdf_corr, _ = _estimate_binned_pdf(x, bin_edges, bias_correction=True)

        empty = counts == 0
        assert empty.sum() > 0, "No empty bins"
        assert np.all(pdf_uncorr[empty] == 0)
        assert np.all(pdf_corr[empty] == 0)


# =============================================================================
# S5: Correction reduces bias (KEY TEST)
# =============================================================================

class TestS5CorrectionReducesBias:
    """S5: |mean_bias_v5| < |mean_bias_raw| for n_bin > 5."""

    def test_v5_better_than_raw(self, gauss_data, bin_edges, pdf_true_func):
        df, N, n_iter, sigma = gauss_data

        bias_raw_all = []
        bias_v5_all = []

        for it in range(min(50, df["iteration"].nunique())):
            x = df[df["iteration"] == it]["x"].values.astype(np.float64)
            f_true = pdf_true_func(x, sigma)

            centers_r, pdf_r, counts_r = _estimate_binned_pdf(x, bin_edges, bias_correction=False)
            bin_idx = np.clip(np.digitize(x, bin_edges) - 1, 0, len(counts_r) - 1)
            pdf_raw_at_x = pdf_r[bin_idx]
            nbin_at_x = counts_r[bin_idx]

            centers_v5, pdf_v5_grid = _estimate_pdf_smooth_1d(
                x, bin_edges, kernel_sigma_bins=0.5, poly_order=2, poly_half_range=0.5
            )
            pdf_v5_at_x = np.interp(x, centers_v5, pdf_v5_grid)

            mask = (f_true > 0.001) & (pdf_raw_at_x > 0) & (pdf_v5_at_x > 0) & (nbin_at_x > 5)
            if mask.sum() > 0:
                bias_raw_all.extend((pdf_raw_at_x[mask] / f_true[mask] - 1).tolist())
                bias_v5_all.extend((pdf_v5_at_x[mask] / f_true[mask] - 1).tolist())

        mean_bias_raw = abs(np.mean(bias_raw_all))
        mean_bias_v5 = abs(np.mean(bias_v5_all))

        assert mean_bias_v5 < mean_bias_raw, (
            f"v5 bias ({mean_bias_v5:.6f}) not less than raw ({mean_bias_raw:.6f})"
        )


# =============================================================================
# S6: Correction bounded
# =============================================================================

class TestS6CorrectionBounded:
    def test_poisson_correction_range(self, gauss_data, bin_edges):
        df, N, n_iter, sigma = gauss_data
        x = df[df["iteration"] == 0]["x"].values.astype(np.float64)
        counts, _ = np.histogram(x, bins=bin_edges)
        correction = 1.0 - np.exp(-counts.astype(np.float64))
        assert np.all(correction >= 0.0)
        assert np.all(correction <= 1.0)


# =============================================================================
# S7: Kernel non-negative
# =============================================================================

class TestS7KernelNonneg:
    def test_kernel_counts_nonneg(self, gauss_data, bin_edges):
        from scipy.ndimage import gaussian_filter1d
        df, N, n_iter, sigma = gauss_data
        x = df[df["iteration"] == 0]["x"].values.astype(np.float64)
        counts, _ = np.histogram(x, bins=bin_edges)
        smoothed = gaussian_filter1d(counts.astype(np.float64), sigma=0.5)
        assert np.all(smoothed >= 0)


# =============================================================================
# S8: Polynomial fallback at edges
# =============================================================================

class TestS8PolynomialFallback:
    def test_edge_bins_no_crash(self):
        rng = np.random.RandomState(99)
        x = rng.normal(5.0, 0.1, 500)
        edges = np.linspace(0, 6, 61)
        centers, pdf = _estimate_pdf_smooth_1d(
            x, edges, kernel_sigma_bins=0.5, poly_order=2, poly_half_range=0.5
        )
        # Returns 60 bin centers + up to 2 edge extrapolation points
        assert len(pdf) >= 60
        assert len(pdf) <= 62
        assert len(centers) == len(pdf)
        assert np.all(pdf >= 0)
        assert np.all(np.isfinite(pdf))
        # Edge points extend to bin edges
        assert centers[0] <= edges[0] + 1e-10
        assert centers[-1] >= edges[-1] - 1e-10


# =============================================================================
# S9: Reconstruction in core (threshold sampling — should now pass)
# =============================================================================

class TestS9ReconstructionCore:
    """S9: Reweighted/original ratio in core |x|<2σ.
    Uses threshold sampling with cw = 1/max(pdf, threshold).
    Tolerance ±15% due to PDF estimation bias at N=10k.
    Tighter validation deferred to invariance tests I1-I4."""

    def test_reconstruction_ratio(self, gauss_data):
        df, N, n_iter, sigma = gauss_data

        bins_check = np.linspace(-4, 4, 41)
        centers_check = 0.5 * (bins_check[:-1] + bins_check[1:])
        core = np.abs(centers_check) < 2.0

        h_orig_total = np.zeros(len(bins_check) - 1)
        h_rw_total = np.zeros(len(bins_check) - 1)

        for it in range(min(50, df["iteration"].nunique())):
            sub = df[df["iteration"] == it].copy().reset_index(drop=True)
            n_orig = len(sub)

            variables = {"x": (120, -6, 6)}
            result = downsampleDFSmoothFactorized(
                sub, variables, it, frac=0.1,
                debug=True, pdf_params={"poly_order": 2, "poly_half_range": 0.5},
            )

            h_orig, _ = np.histogram(sub["x"], bins=bins_check)
            h_orig_total += h_orig

            # Reconstruction: cw = 1/max(pdf, threshold) = _debug_weight_raw
            cw = result["_debug_weight_raw"].values
            h_rw, _ = np.histogram(result["x"], bins=bins_check, weights=cw)
            h_rw_total += h_rw

        good = core & (h_orig_total > 10)
        ratio = h_rw_total[good] / h_orig_total[good]
        mean_ratio = ratio.mean()

        assert 0.85 < mean_ratio < 1.15, (
            f"Reconstruction ratio {mean_ratio:.4f} outside [0.85, 1.15]"
        )


# =============================================================================
# S10: Parameter scalar and list
# =============================================================================

class TestS10ParameterFormats:
    def test_scalar_params(self, gauss_data):
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        variables = {"x": (120, -6, 6)}
        result = downsampleDFSmoothFactorized(
            sub, variables, 42, frac=0.1,
            pdf_params={"poly_order": 2, "poly_half_range": 0.5, "kernel_sigma_bins": 0.5},
        )
        assert len(result) > 0

    def test_list_params(self, gauss_data):
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        variables = {"x": (120, -6, 6)}
        result = downsampleDFSmoothFactorized(
            sub, variables, 42, frac=0.1,
            pdf_params={"poly_order": [2], "poly_half_range": [0.5], "kernel_sigma_bins": [0.5]},
        )
        assert len(result) > 0

    def test_none_params_legacy(self, gauss_data):
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        variables = {"x": (120, -6, 6)}
        result = downsampleDFSmoothFactorized(
            sub, variables, 42, frac=0.1, pdf_params=None,
        )
        assert len(result) > 0

    def test_2d_list_params(self, gauss_data):
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        sub["y"] = np.random.RandomState(99).normal(0, 1, len(sub))
        variables = {"x": (60, -6, 6), "y": (60, -6, 6)}
        result = downsampleDFSmoothFactorized(
            sub, variables, 42, frac=0.1,
            pdf_params={"poly_order": [2, 1], "poly_half_range": [0.5, 0.3]},
        )
        assert len(result) > 0

    def test_normalize_per_dim_param(self):
        assert _normalize_per_dim_param(2, 3, "test") == [2, 2, 2]
        assert _normalize_per_dim_param([1, 2, 3], 3, "test") == [1, 2, 3]
        with pytest.raises(ValueError):
            _normalize_per_dim_param([1, 2], 3, "test")


# =============================================================================
# S11: frac/threshold mutual exclusion
# =============================================================================

class TestS11MutualExclusion:
    """Exactly one of frac or threshold must be specified."""

    def test_neither_raises(self, gauss_data):
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        variables = {"x": (120, -6, 6)}
        with pytest.raises(ValueError, match="exactly one"):
            downsampleDFSmoothFactorized(sub, variables, 42)

    def test_both_raises(self, gauss_data):
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        variables = {"x": (120, -6, 6)}
        with pytest.raises(ValueError, match="exactly one"):
            downsampleDFSmoothFactorized(sub, variables, 42, frac=0.1, threshold=0.01)

    def test_threshold_works(self, gauss_data):
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        variables = {"x": (120, -6, 6)}
        result = downsampleDFSmoothFactorized(sub, variables, 42, threshold=0.1)
        assert len(result) > 0

    def test_frac_works(self, gauss_data):
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        variables = {"x": (120, -6, 6)}
        result = downsampleDFSmoothFactorized(sub, variables, 42, frac=0.1)
        assert len(result) > 0


# =============================================================================
# S12: Threshold ↔ frac roundtrip
# =============================================================================

class TestS12ThresholdFracRoundtrip:
    """threshold_from_frac and frac_from_threshold are consistent."""

    def test_roundtrip(self, gauss_data, pdf_true_func):
        df, N, n_iter, sigma = gauss_data
        x = df[df["iteration"] == 0]["x"].values.astype(np.float64)
        pdf = pdf_true_func(x, sigma)

        target_frac = 0.15
        threshold = _threshold_from_frac(pdf, target_frac)
        recovered_frac = _frac_from_threshold(pdf, threshold)

        assert abs(recovered_frac - target_frac) < 0.001, (
            f"Roundtrip: target={target_frac}, recovered={recovered_frac:.6f}"
        )

    def test_monotonic(self, gauss_data, pdf_true_func):
        """Higher threshold → higher fraction."""
        df, N, n_iter, sigma = gauss_data
        x = df[df["iteration"] == 0]["x"].values.astype(np.float64)
        pdf = pdf_true_func(x, sigma)

        fracs = [_frac_from_threshold(pdf, t) for t in [0.01, 0.05, 0.1, 0.2]]
        for i in range(len(fracs) - 1):
            assert fracs[i] < fracs[i + 1], f"Not monotonic: {fracs}"


# =============================================================================
# S13: pdf_func replaces empirical PDF
# =============================================================================

class TestS13PdfFunc:
    """Optional pdf_func replaces empirical estimator."""

    def test_pdf_func_used(self, gauss_data, pdf_true_func):
        df, N, n_iter, sigma = gauss_data
        sub = df[df["iteration"] == 0].copy().reset_index(drop=True)
        variables = {"x": (120, -6, 6)}

        def my_pdf(df_in):
            return pdf_true_func(df_in["x"].values, sigma)

        result = downsampleDFSmoothFactorized(
            sub, variables, 42, frac=0.1,
            debug=True, pdf_func=my_pdf,
        )
        # With exact PDF, debug_pdf should match the analytical values
        pdf_result = result["_debug_pdf"].values
        pdf_expected = pdf_true_func(result["x"].values, sigma)
        assert np.allclose(pdf_result, pdf_expected, rtol=1e-10)
