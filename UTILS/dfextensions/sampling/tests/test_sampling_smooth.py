"""
test_sampling_smooth.py

Unit tests for dfextensions/sampling/downsample_smooth.py
Phase 13.10.DF v3.1 — Smooth downsampling

Run:  python -m pytest test_sampling_smooth.py -v
"""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from downsample import downsampleDFSmoothFactorized, downsampleDFSmooth


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def gaussian_1d():
    """Single Gaussian, 100k points."""
    np.random.seed(0)
    return pd.DataFrame({"x": np.random.normal(0, 2, 100_000)})


@pytest.fixture
def gaussian_2d():
    """2D Gaussian, 100k points."""
    np.random.seed(0)
    n = 100_000
    return pd.DataFrame({
        "x": np.random.normal(0, 2, n),
        "y": np.random.normal(0, 1.5, n),
    })


@pytest.fixture
def exponential_1d():
    """Exponential distribution, 100k points."""
    np.random.seed(1)
    return pd.DataFrame({"x": np.random.exponential(2.0, 100_000)})


# ===========================================================================
# downsampleDFSmoothFactorized tests  (10 tests)
# ===========================================================================
class TestDownsampleDFSmoothFactorized:

    def test_output_size(self, gaussian_1d):
        """Sampled rows = int(N * frac)."""
        out = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        assert len(out) == int(len(gaussian_1d) * 0.1)

    def test_weight_column_present(self, gaussian_1d):
        """Weight column present by default."""
        out = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        assert "weight" in out.columns

    def test_weight_column_absent(self, gaussian_1d):
        """Weight column absent when keep_weights=False."""
        out = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1, variables="x", random_state=42,
            keep_weights=False)
        assert "weight" not in out.columns

    def test_weight_dtype(self, gaussian_1d):
        """Default weight dtype is float32."""
        out = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        assert out["weight"].dtype == np.float32

    def test_reproducibility(self, gaussian_1d):
        """Same random_state → identical output."""
        a = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        b = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True))

    def test_input_not_mutated(self, gaussian_1d):
        """Original DataFrame not modified."""
        cols_before = list(gaussian_1d.columns)
        _ = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        assert list(gaussian_1d.columns) == cols_before

    def test_invalid_frac_raises(self, gaussian_1d):
        """frac outside (0,1] raises ValueError."""
        with pytest.raises(ValueError, match="frac"):
            downsampleDFSmoothFactorized(
                gaussian_1d, frac=0.0, variables="x", random_state=42)

    def test_weight_column_collision(self, gaussian_1d):
        """Pre-existing weight column raises ValueError."""
        gaussian_1d["weight"] = 1.0
        with pytest.raises(ValueError, match="weight_column"):
            downsampleDFSmoothFactorized(
                gaussian_1d, frac=0.1, variables="x", random_state=42)

    def test_pdf_flattening_1d(self, gaussian_1d):
        """Sampled PDF should be flatter than original (1D Gaussian)."""
        out = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1, variables="x", random_state=42, n_bins=30)
        bins = np.linspace(-5, 5, 20)
        h_orig, _ = np.histogram(gaussian_1d["x"], bins=bins, density=True)
        h_samp, _ = np.histogram(out["x"], bins=bins, density=True)
        # Sampled should have lower std of bin heights (flatter)
        assert h_samp[h_samp > 0].std() < h_orig[h_orig > 0].std()

    def test_2d_factorized(self, gaussian_2d):
        """Factorized 2D downsampling produces correct output size."""
        out = downsampleDFSmoothFactorized(
            gaussian_2d, frac=0.1, variables=["x", "y"],
            random_state=42, n_bins=[30, 30])
        assert len(out) == int(len(gaussian_2d) * 0.1)
        assert "weight" in out.columns

    def test_n_bins_per_variable(self, gaussian_2d):
        """Different n_bins per variable works."""
        out = downsampleDFSmoothFactorized(
            gaussian_2d, frac=0.1, variables=["x", "y"],
            random_state=42, n_bins=[20, 40])
        assert len(out) == int(len(gaussian_2d) * 0.1)

    def test_n_bins_mismatch_raises(self, gaussian_2d):
        """n_bins length mismatch raises ValueError."""
        with pytest.raises(ValueError, match="n_bins length"):
            downsampleDFSmoothFactorized(
                gaussian_2d, frac=0.1, variables=["x", "y"],
                random_state=42, n_bins=[20, 40, 60])


# ===========================================================================
# downsampleDFSmooth tests  (10 tests)
# ===========================================================================
class TestDownsampleDFSmooth:

    def test_output_size(self, gaussian_1d):
        """Sampled rows = int(N * frac)."""
        out = downsampleDFSmooth(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        assert len(out) == int(len(gaussian_1d) * 0.1)

    def test_weight_column_present(self, gaussian_1d):
        """Weight column present by default."""
        out = downsampleDFSmooth(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        assert "weight" in out.columns

    def test_weight_dtype(self, gaussian_1d):
        """Default weight dtype is float32."""
        out = downsampleDFSmooth(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        assert out["weight"].dtype == np.float32

    def test_reproducibility(self, gaussian_1d):
        """Same random_state → identical output."""
        a = downsampleDFSmooth(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        b = downsampleDFSmooth(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True))

    def test_input_not_mutated(self, gaussian_1d):
        """Original DataFrame not modified."""
        cols_before = list(gaussian_1d.columns)
        _ = downsampleDFSmooth(
            gaussian_1d, frac=0.1, variables="x", random_state=42)
        assert list(gaussian_1d.columns) == cols_before

    def test_invalid_frac_raises(self, gaussian_1d):
        """frac outside (0,1] raises ValueError."""
        with pytest.raises(ValueError, match="frac"):
            downsampleDFSmooth(
                gaussian_1d, frac=1.5, variables="x", random_state=42)

    def test_max_dimensions_raises(self, gaussian_1d):
        """More than 5 dimensions raises ValueError."""
        df = pd.DataFrame({f"x{i}": np.random.randn(1000) for i in range(6)})
        with pytest.raises(ValueError, match="5 dimensions"):
            downsampleDFSmooth(
                df, frac=0.1, variables=[f"x{i}" for i in range(6)],
                random_state=42)

    def test_pdf_flattening_1d(self, gaussian_1d):
        """Sampled PDF should be flatter than original (1D)."""
        out = downsampleDFSmooth(
            gaussian_1d, frac=0.1, variables="x", random_state=42, n_bins=30)
        bins = np.linspace(-5, 5, 20)
        h_orig, _ = np.histogram(gaussian_1d["x"], bins=bins, density=True)
        h_samp, _ = np.histogram(out["x"], bins=bins, density=True)
        assert h_samp[h_samp > 0].std() < h_orig[h_orig > 0].std()

    def test_2d_smooth(self, gaussian_2d):
        """Full 2D smooth downsampling produces correct output."""
        out = downsampleDFSmooth(
            gaussian_2d, frac=0.1, variables=["x", "y"],
            random_state=42, n_bins=20)
        assert len(out) == int(len(gaussian_2d) * 0.1)
        assert "weight" in out.columns

    def test_1d_smooth_vs_factorized_similar(self, gaussian_1d):
        """In 1D, smooth and factorized should produce similar distributions."""
        a = downsampleDFSmooth(
            gaussian_1d, frac=0.1, variables="x", random_state=42, n_bins=30)
        b = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1, variables="x", random_state=42, n_bins=30)
        # Same size
        assert len(a) == len(b)
        # Similar distributional statistics (not identical due to edge handling)
        assert abs(a["x"].mean() - b["x"].mean()) < 0.2
        assert abs(a["x"].std() - b["x"].std()) < 0.2
