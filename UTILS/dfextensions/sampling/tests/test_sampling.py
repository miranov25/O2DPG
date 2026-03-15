"""
test_sampling.py

Unit tests for dfextensions/sampling/downsample.py
Phase 13.10.DF v3.0 — 15 tests total

Run:  python -m pytest test_sampling.py -v
"""

import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from downsample import downsampleDF, downsampleDFTrigger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def imbalanced_df():
    """Highly imbalanced 2-group DataFrame (90/10 split)."""
    np.random.seed(0)
    return pd.DataFrame(
        {
            "group": ["A"] * 9000 + ["B"] * 1000,
            "x": np.concatenate(
                [np.random.normal(0, 1, 9000), np.random.normal(3, 0.5, 1000)]
            ),
        }
    )


@pytest.fixture
def multi_group_df():
    """DataFrame with 3 stratify columns for trigger tests."""
    np.random.seed(1)
    n = 10000
    return pd.DataFrame(
        {
            "type": np.random.choice([0, 1], n),
            "pt_bin": np.random.choice(["low", "mid", "high"], n),
            "eta_bin": np.random.choice(["fwd", "mid", "bwd"], n),
            "x": np.random.randn(n),
        }
    )


# ===========================================================================
# downsampleDF tests  (8 tests)
# ===========================================================================
class TestDownsampleDF:

    def test_output_size(self, imbalanced_df):
        """Sampled rows = int(N * frac)."""
        out = downsampleDF(imbalanced_df, frac=0.1, stratify="group", random_state=42)
        assert len(out) == int(len(imbalanced_df) * 0.1)

    def test_weight_column_present(self, imbalanced_df):
        """Weight column exists when keep_weights=True."""
        out = downsampleDF(imbalanced_df, frac=0.1, stratify="group", random_state=42)
        assert "weight" in out.columns

    def test_weight_column_absent(self, imbalanced_df):
        """Weight column absent when keep_weights=False."""
        out = downsampleDF(
            imbalanced_df, frac=0.1, stratify="group",
            random_state=42, keep_weights=False,
        )
        assert "weight" not in out.columns

    def test_weight_dtype_float32(self, imbalanced_df):
        """Default weight dtype is float32."""
        out = downsampleDF(imbalanced_df, frac=0.1, stratify="group", random_state=42)
        assert out["weight"].dtype == np.float32

    def test_reproducibility(self, imbalanced_df):
        """Same random_state → identical output."""
        a = downsampleDF(imbalanced_df, frac=0.1, stratify="group", random_state=42)
        b = downsampleDF(imbalanced_df, frac=0.1, stratify="group", random_state=42)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    def test_input_not_mutated(self, imbalanced_df):
        """Original DataFrame must not be modified."""
        cols_before = list(imbalanced_df.columns)
        n_before = len(imbalanced_df)
        _ = downsampleDF(imbalanced_df, frac=0.1, stratify="group", random_state=42)
        assert list(imbalanced_df.columns) == cols_before
        assert len(imbalanced_df) == n_before

    def test_invalid_frac_raises(self, imbalanced_df):
        """frac outside (0,1] raises ValueError."""
        with pytest.raises(ValueError, match="frac"):
            downsampleDF(imbalanced_df, frac=0.0, stratify="group", random_state=42)
        with pytest.raises(ValueError, match="frac"):
            downsampleDF(imbalanced_df, frac=1.5, stratify="group", random_state=42)

    def test_weight_column_collision(self, imbalanced_df):
        """Pre-existing weight column raises ValueError."""
        imbalanced_df["weight"] = 1.0
        with pytest.raises(ValueError, match="weight_column"):
            downsampleDF(imbalanced_df, frac=0.1, stratify="group", random_state=42)

    def test_rare_group_boosted(self, imbalanced_df):
        """Rare group B (10%) should get >30% of the sample."""
        out = downsampleDF(imbalanced_df, frac=0.1, stratify="group", random_state=42)
        b_frac = (out["group"] == "B").mean()
        assert b_frac > 0.3, f"Rare group B only {b_frac:.1%} of sample"


# ===========================================================================
# downsampleDFTrigger tests  (6 tests)
# ===========================================================================
class TestDownsampleDFTrigger:

    def _make_triggers(self, names, stratifies, fracs):
        return [
            {"name": n, "stratify": s, "frac": f}
            for n, s, f in zip(names, stratifies, fracs)
        ]

    def test_bitmask_column_present(self, multi_group_df):
        """Output must contain combined_trigger column."""
        triggers = self._make_triggers(
            ["t0"], [["type"]], [0.1]
        )
        out = downsampleDFTrigger(multi_group_df, triggers, random_state=42)
        assert "combined_trigger" in out.columns

    def test_bitmask_values(self, multi_group_df):
        """With 2 triggers, bitmask values should be in {1, 2, 3}."""
        triggers = self._make_triggers(
            ["t0", "t1"], [["type"], ["pt_bin"]], [0.1, 0.1]
        )
        out = downsampleDFTrigger(multi_group_df, triggers, random_state=42)
        assert set(out["combined_trigger"].unique()).issubset({1, 2, 3})

    def test_weight_columns_per_trigger(self, multi_group_df):
        """Each trigger gets its own weight_<name> column."""
        triggers = self._make_triggers(
            ["alpha", "beta"], [["type"], ["pt_bin"]], [0.1, 0.1]
        )
        out = downsampleDFTrigger(multi_group_df, triggers, random_state=42)
        assert "weight_alpha" in out.columns
        assert "weight_beta" in out.columns

    def test_reproducibility(self, multi_group_df):
        """Same random_state → identical output."""
        triggers = self._make_triggers(["t0"], [["type"]], [0.2])
        a = downsampleDFTrigger(multi_group_df, triggers, random_state=99)
        b = downsampleDFTrigger(multi_group_df, triggers, random_state=99)
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))

    def test_input_not_mutated(self, multi_group_df):
        """Original DataFrame must not be modified."""
        cols_before = list(multi_group_df.columns)
        triggers = self._make_triggers(["t0"], [["type"]], [0.1])
        _ = downsampleDFTrigger(multi_group_df, triggers, random_state=42)
        assert list(multi_group_df.columns) == cols_before

    def test_max_17_triggers_raises(self, multi_group_df):
        """More than 16 triggers must raise ValueError."""
        triggers = [
            {"name": f"t{i}", "stratify": ["type"], "frac": 0.5}
            for i in range(17)
        ]
        with pytest.raises(ValueError, match="16"):
            downsampleDFTrigger(multi_group_df, triggers, random_state=42)

    def test_missing_key_raises(self, multi_group_df):
        """Trigger dict without 'name' key raises ValueError."""
        triggers = [{"stratify": ["type"], "frac": 0.1}]  # no 'name'
        with pytest.raises(ValueError, match="missing"):
            downsampleDFTrigger(multi_group_df, triggers, random_state=42)
