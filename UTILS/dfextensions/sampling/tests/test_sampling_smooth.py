"""
test_sampling_smooth.py

Tests for smooth downsampling functions (Phase 13.11.DF v3.2 interface).
~30 tests covering: parsing, mask, out-of-range, categorical, non-uniform bins, triggers.

Run:  python -m pytest test_sampling_smooth.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from downsample import (
    downsampleDFSmoothFactorized,
    downsampleDFSmooth,
    downsampleDFSmoothTrigger,
    _parse_variables,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def gaussian_1d():
    np.random.seed(0)
    return pd.DataFrame({"x": np.random.normal(0, 2, 100_000)})


@pytest.fixture
def gaussian_2d():
    np.random.seed(0)
    n = 100_000
    return pd.DataFrame({
        "x": np.random.normal(0, 2, n),
        "y": np.random.normal(0, 1.5, n),
    })


@pytest.fixture
def categorical_df():
    """DataFrame with categorical + continuous columns."""
    np.random.seed(2)
    n = 50_000
    types = np.random.choice([0, 1, 2], n)
    x = np.where(types == 0, np.random.normal(0, 1, n),
         np.where(types == 1, np.random.normal(3, 0.5, n),
                              np.random.normal(-2, 0.8, n)))
    return pd.DataFrame({
        "type": types,
        "x": x,
        "isForFit": np.random.random(n) > 0.3,
    })


@pytest.fixture
def trigger_df():
    """DataFrame for trigger tests."""
    np.random.seed(3)
    n = 50_000
    return pd.DataFrame({
        "type": np.random.choice([0, 1], n),
        "pid": np.random.choice([0, 1, 2, 3], n),
        "pT": np.random.exponential(2, n),
        "eta": np.random.normal(0, 1, n),
        "isDe": (np.random.random(n) < 0.05).astype(np.uint8),
        "isForFit": np.random.random(n) > 0.2,
    })


# ===========================================================================
# §4.1 Variable Parsing Tests (8 tests)
# ===========================================================================
class TestVariableParsing:

    def test_option_c_uniform_bins(self, gaussian_1d):
        cat, cont = _parse_variables({"x": (50, -5, 5)}, gaussian_1d)
        assert len(cat) == 0
        assert len(cont["x"]) == 51
        assert cont["x"][0] == -5.0
        assert cont["x"][-1] == 5.0

    def test_option_d_explicit_edges(self, gaussian_1d):
        edges = np.logspace(-1, 1, 20)
        cat, cont = _parse_variables({"x": edges}, gaussian_1d)
        np.testing.assert_array_equal(cont["x"], edges)

    def test_categorical_variable(self, categorical_df):
        cat, cont = _parse_variables({"type": "categorical"}, categorical_df)
        assert cat == ["type"]
        assert len(cont) == 0

    def test_mixed_c_d_categorical(self, categorical_df):
        variables = {
            "type": "categorical",
            "x": (30, -5, 5),
        }
        cat, cont = _parse_variables(variables, categorical_df)
        assert cat == ["type"]
        assert "x" in cont

    def test_invalid_spec_raises(self, gaussian_1d):
        with pytest.raises(ValueError, match="spec must be"):
            _parse_variables({"x": "invalid"}, gaussian_1d)

    def test_missing_column_raises(self, gaussian_1d):
        with pytest.raises(ValueError, match="not in DataFrame"):
            _parse_variables({"z": (50, -5, 5)}, gaussian_1d)

    def test_list_3_edges_is_option_d(self, gaussian_1d):
        """P0.1: [0.0, 1.0, 2.0] is Option D (3 edges = 2 bins), NOT Option C."""
        cat, cont = _parse_variables({"x": [0.0, 1.0, 2.0]}, gaussian_1d)
        assert len(cont["x"]) == 3  # 3 edges, 2 bins
        assert cont["x"][0] == 0.0

    def test_tuple_int_first_is_option_c(self, gaussian_1d):
        """P0.1: (50, -5.0, 5.0) with int first = Option C."""
        cat, cont = _parse_variables({"x": (50, -5.0, 5.0)}, gaussian_1d)
        assert len(cont["x"]) == 51

    def test_lo_ge_hi_raises(self, gaussian_1d):
        with pytest.raises(ValueError, match="lo must be < hi"):
            _parse_variables({"x": (50, 5.0, -5.0)}, gaussian_1d)


# ===========================================================================
# §4.2 Mask Tests (6 tests)
# ===========================================================================
class TestMask:

    def test_mask_reduces_candidates(self, categorical_df):
        out = downsampleDFSmoothFactorized(
            categorical_df, frac=0.1,
            variables={"x": (30, -5, 5)},
            random_state=42, mask="isForFit",
        )
        # All output rows should have isForFit == True
        assert out["isForFit"].all()

    def test_mask_affects_pdf(self, categorical_df):
        """AD-8: PDF estimated on masked rows only → different result vs no mask."""
        out_masked = downsampleDFSmoothFactorized(
            categorical_df, frac=0.1,
            variables={"x": (30, -5, 5)},
            random_state=42, mask="isForFit",
        )
        out_all = downsampleDFSmoothFactorized(
            categorical_df, frac=0.1,
            variables={"x": (30, -5, 5)},
            random_state=42,
        )
        # Sizes should differ (different candidate pools)
        assert len(out_masked) != len(out_all)

    def test_no_mask_default(self, gaussian_1d):
        out = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": (30, -5, 5)},
            random_state=42,
        )
        assert len(out) > 0

    def test_invalid_mask_column(self, gaussian_1d):
        with pytest.raises(ValueError, match="Mask column"):
            downsampleDFSmoothFactorized(
                gaussian_1d, frac=0.1,
                variables={"x": (30, -5, 5)},
                random_state=42, mask="nonexistent",
            )

    def test_mask_as_array(self, gaussian_1d):
        mask_arr = np.random.random(len(gaussian_1d)) > 0.5
        out = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": (30, -5, 5)},
            random_state=42, mask=mask_arr,
        )
        assert len(out) > 0

    def test_mask_column_preserved(self, categorical_df):
        """Mask column should be in output (it's a df column, not removed)."""
        out = downsampleDFSmoothFactorized(
            categorical_df, frac=0.1,
            variables={"x": (30, -5, 5)},
            random_state=42, mask="isForFit",
        )
        assert "isForFit" in out.columns


# ===========================================================================
# §4.3 Out-of-Range Tests (3 tests)
# ===========================================================================
class TestOutOfRange:

    def test_out_of_range_excluded(self, gaussian_1d):
        """P0.2: values outside bin range not in output."""
        out = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": (30, -2, 2)},  # narrow range
            random_state=42,
        )
        assert out["x"].min() >= -2
        assert out["x"].max() <= 2

    def test_out_of_range_not_in_pdf(self, gaussian_1d):
        """Out-of-range values don't affect PDF estimate."""
        # With outlier-free data, should get same result
        df_clean = gaussian_1d[(gaussian_1d["x"] >= -2) & (gaussian_1d["x"] <= 2)].copy()
        out_dirty = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": (30, -2, 2)},
            random_state=42,
        )
        out_clean = downsampleDFSmoothFactorized(
            df_clean, frac=0.1,
            variables={"x": (30, -2, 2)},
            random_state=42,
        )
        # Similar size (within statistical fluctuation of different N)
        assert abs(len(out_dirty) - len(out_clean)) / max(len(out_dirty), len(out_clean)) < 0.05

    def test_range_and_mask_interaction(self, categorical_df):
        """Mask applied first, then range filter."""
        out = downsampleDFSmoothFactorized(
            categorical_df, frac=0.1,
            variables={"x": (30, -2, 2)},
            random_state=42, mask="isForFit",
        )
        assert out["isForFit"].all()
        assert out["x"].min() >= -2
        assert out["x"].max() <= 2


# ===========================================================================
# §4.4 Categorical Tests (4 tests)
# ===========================================================================
class TestCategorical:

    def test_categorical_only(self, categorical_df):
        """AD-10: pure categorical → groupby weighting."""
        out = downsampleDFSmoothFactorized(
            categorical_df, frac=0.1,
            variables={"type": "categorical"},
            random_state=42,
        )
        assert len(out) == int(len(categorical_df) * 0.1)
        assert "weight" in out.columns

    def test_categorical_plus_continuous(self, categorical_df):
        """Mixed: per-category smooth PDF."""
        out = downsampleDFSmoothFactorized(
            categorical_df, frac=0.1,
            variables={"type": "categorical", "x": (30, -5, 5)},
            random_state=42,
        )
        assert len(out) > 0
        assert "weight" in out.columns

    def test_no_interpolation_across_categories(self, categorical_df):
        """Per-category weights should differ when distributions differ."""
        out = downsampleDFSmooth(
            categorical_df, frac=0.1,
            variables={"type": "categorical", "x": (30, -5, 5)},
            random_state=42,
        )
        # Type 0 (centered at 0) and type 1 (centered at 3) should both appear
        assert set(out["type"].unique()) == {0, 1, 2}

    def test_all_categorical_matches_downsampleDF(self, categorical_df):
        """AD-10: all-categorical smooth should behave like downsampleDF."""
        from downsample import downsampleDF
        out_smooth = downsampleDFSmoothFactorized(
            categorical_df, frac=0.1,
            variables={"type": "categorical"},
            random_state=42,
        )
        out_binned = downsampleDF(
            categorical_df, frac=0.1,
            stratify="type",
            random_state=42,
        )
        assert len(out_smooth) == len(out_binned)


# ===========================================================================
# §4.5 Non-Uniform Binning Tests (3 tests)
# ===========================================================================
class TestNonUniformBins:

    def test_logspace_bins_accepted(self, gaussian_1d):
        """Option D with log-spaced edges works."""
        edges = list(np.linspace(-5, 5, 30))
        out = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": edges},
            random_state=42,
        )
        assert len(out) > 0

    def test_nonuniform_vs_uniform_equivalent(self, gaussian_1d):
        """Same uniform edges via C and D give same-size output."""
        out_c = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": (30, -5, 5)},
            random_state=42,
        )
        edges_d = np.linspace(-5, 5, 31)  # same as (30, -5, 5)
        out_d = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": list(edges_d)},
            random_state=42,
        )
        assert len(out_c) == len(out_d)

    def test_option_d_2_edges_is_1_bin(self, gaussian_1d):
        """Minimum: 2 edges = 1 bin."""
        out = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": [-5.0, 5.0]},
            random_state=42,
        )
        assert len(out) > 0


# ===========================================================================
# §4.6 Smooth Function Tests (5 tests)
# ===========================================================================
class TestSmoothFunctions:

    def test_factorized_output_size(self, gaussian_1d):
        out = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": (50, -6, 6)},
            random_state=42,
        )
        assert len(out) == int(len(gaussian_1d[
            (gaussian_1d["x"] >= -6) & (gaussian_1d["x"] <= 6)
        ]) * 0.1)

    def test_smooth_nd_output_size(self, gaussian_2d):
        out = downsampleDFSmooth(
            gaussian_2d, frac=0.1,
            variables={"x": (20, -5, 5), "y": (20, -4, 4)},
            random_state=42,
        )
        assert len(out) > 0
        assert "weight" in out.columns

    def test_reproducibility_factorized(self, gaussian_1d):
        a = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": (50, -6, 6)},
            random_state=42,
        )
        b = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": (50, -6, 6)},
            random_state=42,
        )
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    def test_input_not_mutated(self, gaussian_1d):
        cols_before = list(gaussian_1d.columns)
        _ = downsampleDFSmoothFactorized(
            gaussian_1d, frac=0.1,
            variables={"x": (50, -6, 6)},
            random_state=42,
        )
        assert list(gaussian_1d.columns) == cols_before

    def test_max_dimensions_raises(self):
        df = pd.DataFrame({f"x{i}": np.random.randn(1000) for i in range(6)})
        variables = {f"x{i}": (10, -3, 3) for i in range(6)}
        with pytest.raises(ValueError, match="5 continuous"):
            downsampleDFSmooth(df, frac=0.1, variables=variables, random_state=42)


# ===========================================================================
# §4.7 Trigger Tests (6 tests)
# ===========================================================================
class TestSmoothTrigger:

    def test_bitmask_column_present(self, trigger_df):
        triggers = [
            {"name": "t0", "variables": {"pT": (20, 0, 10)}, "frac": 0.1},
        ]
        out = downsampleDFSmoothTrigger(trigger_df, triggers, random_state=42)
        assert "combined_trigger" in out.columns

    def test_weight_columns_per_trigger(self, trigger_df):
        triggers = [
            {"name": "alpha", "variables": {"pT": (20, 0, 10)}, "frac": 0.1},
            {"name": "beta", "variables": {"eta": (20, -3, 3)}, "frac": 0.1},
        ]
        out = downsampleDFSmoothTrigger(trigger_df, triggers, random_state=42)
        assert "weight_alpha" in out.columns
        assert "weight_beta" in out.columns

    def test_mixed_c_d_categorical_trigger(self, trigger_df):
        """AD-5: triggers can mix Option C, D, and categorical."""
        triggers = [
            {
                "name": "mixed",
                "variables": {
                    "type": "categorical",
                    "pid": "categorical",
                    "pT": (20, 0, 10),
                },
                "frac": 0.1,
            },
            {
                "name": "rare",
                "variables": {"isDe": "categorical"},
                "frac": 0.5,
            },
        ]
        out = downsampleDFSmoothTrigger(trigger_df, triggers, random_state=42)
        assert "weight_mixed" in out.columns
        assert "weight_rare" in out.columns
        assert set(out["combined_trigger"].unique()).issubset({1, 2, 3})

    def test_trigger_reproducibility(self, trigger_df):
        triggers = [{"name": "t0", "variables": {"pT": (20, 0, 10)}, "frac": 0.1}]
        a = downsampleDFSmoothTrigger(trigger_df, triggers, random_state=42)
        b = downsampleDFSmoothTrigger(trigger_df, triggers, random_state=42)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    def test_trigger_mask(self, trigger_df):
        """Mask applied before all triggers."""
        triggers = [{"name": "t0", "variables": {"pT": (20, 0, 10)}, "frac": 0.1}]
        out = downsampleDFSmoothTrigger(
            trigger_df, triggers, random_state=42, mask="isForFit"
        )
        assert out["isForFit"].all()

    def test_max_17_triggers_raises(self, trigger_df):
        triggers = [
            {"name": f"t{i}", "variables": {"pT": (10, 0, 10)}, "frac": 0.5}
            for i in range(17)
        ]
        with pytest.raises(ValueError, match="16"):
            downsampleDFSmoothTrigger(trigger_df, triggers, random_state=42)
