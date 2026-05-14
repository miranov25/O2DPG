"""
Phase 13.26.ADF — dtype_overrides on read_tree

D1: regex pattern matching converts float64→float16
D2: first-match-wins precedence
D3: existing columns (no override) unchanged
D4: overflow detection warns on finite→inf
D5: NaN preserved across float downcast
D6: schema dtype_hints not overridden when no override matches
D7: round-trip: write float64, read with override, values within tolerance
"""

import os
import sys
import tempfile
import warnings
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame

try:
    import ROOT
    import uproot
    _HAS_ROOT = ROOT is not None
except ImportError:
    _HAS_ROOT = False


@pytest.fixture
def tmp_root_file():
    """Create a temp ROOT file with float64 branches for testing dtype_overrides."""
    if not _HAS_ROOT:
        pytest.skip("Requires ROOT + uproot")

    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, "test_dtype_overrides.root")

    rng = np.random.default_rng(42)
    n = 1000
    df = pd.DataFrame({
        'x': rng.uniform(0, 300, n).astype(np.float64),
        'dy_intercept_PIter1': rng.normal(0, 0.1, n).astype(np.float64),
        'dy_slope_PIter1': rng.normal(0, 0.01, n).astype(np.float64),
        'dy_err_PIter1': rng.uniform(0.001, 0.1, n).astype(np.float64),
        'dz_intercept_PIter2': rng.normal(0, 0.2, n).astype(np.float64),
        'row': np.arange(n, dtype=np.int64),
        'firstTForbit': np.full(n, 29893280, dtype=np.int64),
    })

    adf = AliasDataFrame(df)
    adf.export_tree(filepath, treename="tree")
    return filepath


@pytest.mark.skipif(not _HAS_ROOT, reason="Requires ROOT + uproot")
class TestDtypeOverrides:

    @pytest.mark.invariance
    def test_D1_regex_converts_float64_to_float16(self, tmp_root_file):
        """Branches matching regex pattern are converted to target dtype."""
        adf = AliasDataFrame.read_tree(tmp_root_file, "tree", dtype_overrides={
            r'dy_.*_PIter\d+': np.float16,
        })
        assert adf.df['dy_intercept_PIter1'].dtype == np.float16
        assert adf.df['dy_slope_PIter1'].dtype == np.float16
        assert adf.df['dy_err_PIter1'].dtype == np.float16
        # Non-matching columns unchanged
        assert adf.df['dz_intercept_PIter2'].dtype != np.float16

    @pytest.mark.invariance
    def test_D2_first_match_wins(self, tmp_root_file):
        """First matching pattern takes priority."""
        adf = AliasDataFrame.read_tree(tmp_root_file, "tree", dtype_overrides={
            r'dy_err_.*': np.float32,       # errors → float32 (matches first)
            r'dy_.*_PIter\d+': np.float16,  # everything else → float16
        })
        assert adf.df['dy_err_PIter1'].dtype == np.float32  # first match
        assert adf.df['dy_intercept_PIter1'].dtype == np.float16  # second match
        assert adf.df['dy_slope_PIter1'].dtype == np.float16

    @pytest.mark.invariance
    def test_D3_no_override_columns_unchanged(self, tmp_root_file):
        """Columns not matching any pattern keep their original dtype."""
        adf = AliasDataFrame.read_tree(tmp_root_file, "tree", dtype_overrides={
            r'dy_.*': np.float16,
        })
        # 'x' and 'dz_*' should not be affected
        assert adf.df['x'].dtype != np.float16
        assert adf.df['dz_intercept_PIter2'].dtype != np.float16

    @pytest.mark.invariance
    def test_D4_overflow_warns(self, tmp_root_file):
        """Overflow on downcast produces a UserWarning."""
        # Write a file with large values that overflow float16
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "overflow_test.root")
        df = pd.DataFrame({
            'big_values': np.array([1e10, 1e20, 0.5, -0.5], dtype=np.float64),
            'row': np.arange(4, dtype=np.int64),
        })
        AliasDataFrame(df).export_tree(filepath, "tree")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf = AliasDataFrame.read_tree(filepath, "tree", dtype_overrides={
                r'big_values': np.float16,
            })
            overflow_warnings = [x for x in w if "overflowed" in str(x.message)]
            assert len(overflow_warnings) >= 1

    @pytest.mark.invariance
    def test_D5_nan_preserved(self, tmp_root_file):
        """NaN values survive float64→float16 downcast."""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "nan_test.root")
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        df = pd.DataFrame({'val': arr, 'row': np.arange(5, dtype=np.int64)})
        AliasDataFrame(df).export_tree(filepath, "tree")

        adf = AliasDataFrame.read_tree(filepath, "tree", dtype_overrides={
            r'val': np.float16,
        })
        assert adf.df['val'].dtype == np.float16
        assert np.isnan(adf.df['val'].values[1])
        assert np.isnan(adf.df['val'].values[3])
        assert not np.isnan(adf.df['val'].values[0])

    @pytest.mark.invariance
    def test_D6_no_overrides_matches_baseline(self, tmp_root_file):
        """read_tree with dtype_overrides=None produces same result as without."""
        adf_base = AliasDataFrame.read_tree(tmp_root_file, "tree")
        adf_none = AliasDataFrame.read_tree(tmp_root_file, "tree", dtype_overrides=None)
        adf_empty = AliasDataFrame.read_tree(tmp_root_file, "tree", dtype_overrides={})

        for col in adf_base.df.columns:
            assert adf_base.df[col].dtype == adf_none.df[col].dtype
            assert adf_base.df[col].dtype == adf_empty.df[col].dtype

    @pytest.mark.invariance
    def test_D7_roundtrip_values_within_tolerance(self, tmp_root_file):
        """Values survive write→read-with-override within float16 tolerance."""
        adf_orig = AliasDataFrame.read_tree(tmp_root_file, "tree")
        adf_f16 = AliasDataFrame.read_tree(tmp_root_file, "tree", dtype_overrides={
            r'dy_intercept_PIter1': np.float16,
        })

        orig_vals = adf_orig.df['dy_intercept_PIter1'].values
        f16_vals = adf_f16.df['dy_intercept_PIter1'].values.astype(np.float64)

        # float16 has ~3 decimal digits of precision
        np.testing.assert_allclose(orig_vals, f16_vals, rtol=1e-2, atol=1e-3,
                                   err_msg="D7: round-trip values diverged beyond float16 tolerance")

    @pytest.mark.invariance
    def test_D8_schema_roundtrip_preserves_overridden_dtype(self, tmp_root_file):
        """Export after read-with-override preserves the overridden dtype in schema.
        
        Sequence: read(override float16) → export → re-read(no override)
        Expected: re-read gets float16 via schema column_dtypes, not float64.
        """
        tmpdir = tempfile.mkdtemp()
        reexport_path = os.path.join(tmpdir, "reexported.root")

        # Read with override
        adf = AliasDataFrame.read_tree(tmp_root_file, "tree", dtype_overrides={
            r'dy_intercept_PIter1': np.float16,
        })
        assert adf.df['dy_intercept_PIter1'].dtype == np.float16

        # Export (schema records actual dtype)
        adf.export_tree(reexport_path, "tree")

        # Re-read WITHOUT override — schema should preserve float16
        adf2 = AliasDataFrame.read_tree(reexport_path, "tree")
        assert adf2.df['dy_intercept_PIter1'].dtype == np.float16, \
            "D8: schema round-trip lost overridden dtype"

    @pytest.mark.invariance
    def test_D9_entry_range_with_overrides(self, tmp_root_file):
        """dtype_overrides applied consistently with entry_start/entry_stop."""
        # Read full file with override
        adf_full = AliasDataFrame.read_tree(tmp_root_file, "tree", dtype_overrides={
            r'dy_.*_PIter\d+': np.float16,
        })

        # Read subset with same override
        adf_sub = AliasDataFrame.read_tree(tmp_root_file, "tree",
            entry_start=100, entry_stop=500,
            dtype_overrides={r'dy_.*_PIter\d+': np.float16})

        # Dtypes must match
        assert adf_sub.df['dy_intercept_PIter1'].dtype == np.float16
        assert adf_sub.df['dy_slope_PIter1'].dtype == np.float16

        # Values must match the corresponding slice of full read
        np.testing.assert_array_equal(
            adf_sub.df['dy_intercept_PIter1'].values,
            adf_full.df['dy_intercept_PIter1'].values[100:500],
            err_msg="D9: entry_range + override produced different values"
        )

    @pytest.mark.invariance
    def test_D10_override_warning_shows_correct_dtypes(self, tmp_root_file):
        """Overflow warning message shows original→target dtype, not target→target."""
        tmpdir = tempfile.mkdtemp()
        filepath = os.path.join(tmpdir, "dtype_msg_test.root")
        df = pd.DataFrame({
            'big': np.array([1e10, 1e20], dtype=np.float64),
            'row': np.arange(2, dtype=np.int64),
        })
        AliasDataFrame(df).export_tree(filepath, "tree")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AliasDataFrame.read_tree(filepath, "tree", dtype_overrides={
                r'big': np.float16,
            })
            overflow_msgs = [str(x.message) for x in w if "overflowed" in str(x.message)]
            assert len(overflow_msgs) >= 1
            # Must show float64 → float16, NOT float16 → float16
            assert "float64" in overflow_msgs[0], \
                f"D10: warning should show original dtype float64, got: {overflow_msgs[0]}"
            assert "float16" in overflow_msgs[0]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
