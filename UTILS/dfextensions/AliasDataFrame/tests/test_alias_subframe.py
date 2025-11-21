"""
test_alias_subframe.py - Dedicated tests for subframe functionality

Phase 3A: Subframe Normalisation & Alias Resolution

Test Classes:
    - TestSubframeBasicJoin: Single and multi-key joins
    - TestSubframeMissingKeys: NaN + warning behavior (NEW)
    - TestSubframeLazyEvaluation: Lazy and idempotent behavior
    - TestSubframeRoundtrip: ROOT/Parquet persistence
    - TestSubframeEdgeCases: Error handling, invalid references
"""

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import warnings

# Adjust import path as needed
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfextensions.AliasDataFrame import AliasDataFrame

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = ROOT is not None
except ImportError:
    HAS_ROOT = False


class TestSubframeBasicJoin(unittest.TestCase):
    """Test basic subframe join operations with 1-key and N-key indices"""

    def setUp(self):
        """Create standard test data: clusters referencing tracks"""
        np.random.seed(42)
        n_tracks = 100
        n_clusters_per_track = 10

        # Track data (subframe)
        self.df_tracks = pd.DataFrame({
            "track_index": np.arange(n_tracks),
            "mX": np.random.normal(0, 10, n_tracks).astype(np.float32),
            "mY": np.random.normal(0, 10, n_tracks).astype(np.float32),
            "mZ": np.random.normal(0, 10, n_tracks).astype(np.float32),
            "mPt": np.random.exponential(1.0, n_tracks).astype(np.float32),
        })

        # Cluster data (main frame) - each track has n_clusters_per_track clusters
        cluster_idx = np.repeat(self.df_tracks["track_index"].values, n_clusters_per_track)
        n_clusters = len(cluster_idx)
        self.df_clusters = pd.DataFrame({
            "track_index": cluster_idx,
            "mX": np.random.normal(0, 10, n_clusters).astype(np.float32),
            "mY": np.random.normal(0, 10, n_clusters).astype(np.float32),
            "mZ": np.random.normal(0, 10, n_clusters).astype(np.float32),
        })

    def test_single_key_join(self):
        """Test basic single-key join (track_index)"""
        adf_clusters = AliasDataFrame(self.df_clusters.copy())
        adf_tracks = AliasDataFrame(self.df_tracks.copy())

        adf_clusters.register_subframe("T", adf_tracks, index_columns="track_index")
        adf_clusters.add_alias("mDX", "mX - T.mX")
        adf_clusters.materialize_alias("mDX")

        # Verify result matches manual merge
        merged = self.df_clusters.merge(self.df_tracks, on="track_index", suffixes=("", "_trk"))
        expected = merged["mX"] - merged["mX_trk"]

        pd.testing.assert_series_equal(
            adf_clusters.df["mDX"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            rtol=1e-5
        )

    def test_multi_key_join_2keys(self):
        """Test two-key join (track_index, firstTFOrbit)"""
        df_main = pd.DataFrame({
            'track_index': [0, 0, 1, 1],
            'firstTFOrbit': [100, 200, 100, 200],
            'x': [1.0, 2.0, 3.0, 4.0]
        })
        df_sub = pd.DataFrame({
            'track_index': [0, 0, 1, 1],
            'firstTFOrbit': [100, 200, 100, 200],
            'y': [10.0, 20.0, 30.0, 40.0]
        })

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("T", adf_sub, index_columns=["track_index", "firstTFOrbit"])
        adf_main.add_alias("sum_xy", "x + T.y")
        adf_main.materialize_alias("sum_xy")

        expected = np.array([11.0, 22.0, 33.0, 44.0])
        np.testing.assert_array_almost_equal(adf_main.df['sum_xy'].values, expected)

    def test_multi_key_join_3keys(self):
        """Test three-key join (side, row, drift25) - real use case"""
        df_main = pd.DataFrame({
            'side': [0, 0, 1, 1],
            'row': [0, 1, 0, 1],
            'drift25': [100, 100, 200, 200],
            'value': [1.0, 2.0, 3.0, 4.0]
        })
        df_sub = pd.DataFrame({
            'side': [0, 0, 1, 1],
            'row': [0, 1, 0, 1],
            'drift25': [100, 100, 200, 200],
            'correction': [0.1, 0.2, 0.3, 0.4]
        })

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("DTrack0", adf_sub, index_columns=["side", "row", "drift25"])
        adf_main.add_alias("corrected", "value - DTrack0.correction")
        adf_main.materialize_alias("corrected")

        expected = np.array([0.9, 1.8, 2.7, 3.6])
        np.testing.assert_array_almost_equal(adf_main.df['corrected'].values, expected)

    def test_subframe_alias_in_expression(self):
        """Test that subframe's own aliases can be used in main frame expressions"""
        df_main = pd.DataFrame({"track_id": [0, 1, 2], "x": [10.0, 20.0, 30.0]})
        df_sub = pd.DataFrame({"track_id": [0, 1, 2], "residual": [1.1, 2.2, 3.3]})

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)

        # Define alias in subframe
        adf_sub.add_alias("residual_scaled", "residual * 100", dtype=np.float64)

        # Register and use subframe alias
        adf_main.register_subframe("track", adf_sub, index_columns="track_id")
        adf_main.add_alias("resid100", "track.residual_scaled", dtype=np.float64)
        adf_main.materialize_alias("resid100")

        expected = np.array([110.0, 220.0, 330.0])
        np.testing.assert_array_almost_equal(adf_main.df['resid100'].values, expected)


class TestSubframeMissingKeys(unittest.TestCase):
    """Test missing-key behavior: NaN + warning (Phase 3A requirement)"""

    def test_missing_keys_produce_nan_not_dropped(self):
        """Critical: Missing keys must produce NaN, not drop rows"""
        # Main frame has keys 0, 1, 2, 3, 4
        df_main = pd.DataFrame({
            'key': [0, 1, 2, 3, 4],
            'value': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        # Subframe only has keys 0, 2, 4 (missing 1, 3)
        df_sub = pd.DataFrame({
            'key': [0, 2, 4],
            'factor': [1.0, 2.0, 3.0]
        })

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("S", adf_sub, index_columns="key")
        adf_main.add_alias("result", "value * S.factor")

        # Should warn about missing keys
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf_main.materialize_alias("result")

            # Check warning was emitted
            self.assertEqual(len(w), 1)
            self.assertIn("not found", str(w[0].message).lower())
            self.assertIn("S", str(w[0].message))

        # Critical: All 5 rows must be preserved
        self.assertEqual(len(adf_main.df), 5)

        # Keys 0, 2, 4 should have valid results
        self.assertAlmostEqual(adf_main.df.loc[0, 'result'], 10.0)
        self.assertAlmostEqual(adf_main.df.loc[2, 'result'], 60.0)
        self.assertAlmostEqual(adf_main.df.loc[4, 'result'], 150.0)

        # Keys 1, 3 should be NaN
        self.assertTrue(np.isnan(adf_main.df.loc[1, 'result']))
        self.assertTrue(np.isnan(adf_main.df.loc[3, 'result']))

    def test_warning_shows_count(self):
        """Warning should show count of missing keys"""
        df_main = pd.DataFrame({
            'key': np.arange(100),
            'value': np.ones(100)
        })
        # Subframe only has even keys (50 missing)
        df_sub = pd.DataFrame({
            'key': np.arange(0, 100, 2),
            'factor': np.ones(50)
        })

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("S", adf_sub, index_columns="key")
        adf_main.add_alias("result", "S.factor")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf_main.materialize_alias("result")

            self.assertEqual(len(w), 1)
            # Warning should mention the count
            self.assertIn("50", str(w[0].message))

    def test_warning_suppression(self):
        """Test that warnings can be suppressed"""
        df_main = pd.DataFrame({'key': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})
        df_sub = pd.DataFrame({'key': [0, 2], 'factor': [1.0, 2.0]})

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("S", adf_sub, index_columns="key")
        adf_main.add_alias("result", "S.factor")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf_main.materialize_alias("result", warn_missing_keys=False)

            # No warning should be emitted
            self.assertEqual(len(w), 0)

        # But NaN should still be produced
        self.assertTrue(np.isnan(adf_main.df.loc[1, 'result']))

    def test_no_warning_when_all_keys_present(self):
        """No warning when all keys are found"""
        df_main = pd.DataFrame({'key': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})
        df_sub = pd.DataFrame({'key': [0, 1, 2], 'factor': [1.0, 2.0, 3.0]})

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("S", adf_sub, index_columns="key")
        adf_main.add_alias("result", "value * S.factor")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf_main.materialize_alias("result")

            # No warning
            self.assertEqual(len(w), 0)

        # All values valid
        expected = np.array([1.0, 4.0, 9.0])
        np.testing.assert_array_almost_equal(adf_main.df['result'].values, expected)

    def test_missing_keys_multi_key_join(self):
        """Test missing key behavior with multi-key joins"""
        df_main = pd.DataFrame({
            'k1': [0, 0, 1, 1],
            'k2': [0, 1, 0, 1],
            'value': [1.0, 2.0, 3.0, 4.0]
        })
        # Missing (0, 1) and (1, 0)
        df_sub = pd.DataFrame({
            'k1': [0, 1],
            'k2': [0, 1],
            'factor': [10.0, 20.0]
        })

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("S", adf_sub, index_columns=["k1", "k2"])
        adf_main.add_alias("result", "value * S.factor")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf_main.materialize_alias("result")

            self.assertEqual(len(w), 1)
            self.assertIn("2", str(w[0].message))  # 2 missing keys

        # All rows preserved
        self.assertEqual(len(adf_main.df), 4)

        # (0,0) and (1,1) have values
        self.assertAlmostEqual(adf_main.df.loc[0, 'result'], 10.0)
        self.assertAlmostEqual(adf_main.df.loc[3, 'result'], 80.0)

        # (0,1) and (1,0) are NaN
        self.assertTrue(np.isnan(adf_main.df.loc[1, 'result']))
        self.assertTrue(np.isnan(adf_main.df.loc[2, 'result']))


class TestSubframeLazyEvaluation(unittest.TestCase):
    """Test lazy evaluation and idempotent behavior"""

    def setUp(self):
        self.df_main = pd.DataFrame({
            'key': [0, 1, 2],
            'x': [1.0, 2.0, 3.0]
        })
        self.df_sub = pd.DataFrame({
            'key': [0, 1, 2],
            'y': [10.0, 20.0, 30.0]
        })

    def test_alias_not_materialized_until_requested(self):
        """Subframe alias should not create columns until explicitly requested"""
        adf_main = AliasDataFrame(self.df_main.copy())
        adf_sub = AliasDataFrame(self.df_sub.copy())

        adf_main.register_subframe("S", adf_sub, index_columns="key")
        adf_main.add_alias("result", "x + S.y")

        # Alias defined but not materialized
        self.assertIn("result", adf_main.aliases)
        self.assertNotIn("result", adf_main.df.columns)

        # Subframe column not yet joined
        self.assertNotIn("y__S", adf_main.df.columns)
        self.assertNotIn("S__y", adf_main.df.columns)

    def test_materialization_idempotent(self):
        """Multiple materialize calls should be idempotent"""
        adf_main = AliasDataFrame(self.df_main.copy())
        adf_sub = AliasDataFrame(self.df_sub.copy())

        adf_main.register_subframe("S", adf_sub, index_columns="key")
        adf_main.add_alias("result", "x + S.y")

        # First materialization
        adf_main.materialize_alias("result")
        first_result = adf_main.df["result"].copy()

        # Second materialization (should be no-op or same result)
        adf_main.materialize_alias("result")
        second_result = adf_main.df["result"]

        pd.testing.assert_series_equal(first_result, second_result)

    def test_no_accidental_flattening(self):
        """Subframe columns should not appear in main df unless alias is materialized"""
        adf_main = AliasDataFrame(self.df_main.copy())
        adf_sub = AliasDataFrame(self.df_sub.copy())

        original_columns = set(adf_main.df.columns)

        adf_main.register_subframe("S", adf_sub, index_columns="key")

        # Just registering should not add columns
        self.assertEqual(set(adf_main.df.columns), original_columns)

        # Adding alias should not add columns
        adf_main.add_alias("result", "x + S.y")
        self.assertEqual(set(adf_main.df.columns), original_columns)

    def test_getattr_triggers_materialization(self):
        """Accessing alias via __getattr__ should trigger materialization"""
        adf_main = AliasDataFrame(self.df_main.copy())
        adf_sub = AliasDataFrame(self.df_sub.copy())

        adf_main.register_subframe("S", adf_sub, index_columns="key")
        adf_main.add_alias("result", "x + S.y")

        self.assertNotIn("result", adf_main.df.columns)

        # Access via attribute
        _ = adf_main.result

        # Now it should be materialized
        self.assertIn("result", adf_main.df.columns)


class TestSubframeRoundtrip(unittest.TestCase):
    """Test ROOT and Parquet persistence of subframes"""

    def setUp(self):
        np.random.seed(42)
        self.df_main = pd.DataFrame({
            'track_index': [0, 0, 1, 1, 2, 2],
            'firstTFOrbit': [100, 100, 100, 200, 200, 200],
            'x': np.random.randn(6).astype(np.float32),
        })
        self.df_sub = pd.DataFrame({
            'track_index': [0, 1, 2],
            'firstTFOrbit': [100, 100, 200],
            'y': np.random.randn(3).astype(np.float32),
        })

    @unittest.skipUnless(HAS_ROOT, "ROOT not available")
    def test_root_roundtrip_basic(self):
        """Test basic ROOT export/import preserves subframe functionality"""
        adf_main = AliasDataFrame(self.df_main.copy())
        adf_sub = AliasDataFrame(self.df_sub.copy())

        adf_main.register_subframe("T", adf_sub, index_columns=["track_index", "firstTFOrbit"])
        adf_main.add_alias("sum_xy", "x + T.y", dtype=np.float32)

        with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Export
            adf_main.export_tree(tmp_path, treename="tree")

            # Import
            adf_loaded = AliasDataFrame.read_tree(tmp_path, treename="tree")

            # Check subframe was loaded
            self.assertIn("T", adf_loaded._subframes.subframes)

            # Check alias exists and works
            self.assertIn("sum_xy", adf_loaded.aliases)
            adf_loaded.materialize_alias("sum_xy")

            # Results should match
            adf_main.materialize_alias("sum_xy")
            np.testing.assert_array_almost_equal(
                adf_loaded.df['sum_xy'].values,
                adf_main.df['sum_xy'].values,
                decimal=5
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @unittest.skipUnless(HAS_ROOT, "ROOT not available")
    def test_root_roundtrip_with_entry_range(self):
        """Test that entry_range applies only to main tree, not subframes"""
        adf_main = AliasDataFrame(self.df_main.copy())
        adf_sub = AliasDataFrame(self.df_sub.copy())

        adf_main.register_subframe("T", adf_sub, index_columns=["track_index", "firstTFOrbit"])

        with tempfile.NamedTemporaryFile(suffix=".root", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            adf_main.export_tree(tmp_path, treename="tree")

            # Read with entry_range
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                adf_loaded = AliasDataFrame.read_tree(tmp_path, treename="tree", entry_stop=3)

                # Should warn about subframes being fully loaded
                self.assertTrue(any("entry_start/entry_stop" in str(warning.message) for warning in w))

            # Main tree should be sliced
            self.assertEqual(len(adf_loaded.df), 3)

            # Subframe should be fully loaded
            sf = adf_loaded.get_subframe("T")
            self.assertEqual(len(sf.df), len(self.df_sub))
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_parquet_roundtrip(self):
        """Test Parquet save/load preserves subframe metadata"""
        adf_main = AliasDataFrame(self.df_main.copy())
        adf_sub = AliasDataFrame(self.df_sub.copy())

        adf_main.register_subframe("T", adf_sub, index_columns=["track_index", "firstTFOrbit"])
        adf_main.add_alias("sum_xy", "x + T.y")

        with tempfile.TemporaryDirectory() as tmpdir:
            path_main = os.path.join(tmpdir, "main.parquet")
            path_sub = os.path.join(tmpdir, "sub.parquet")

            adf_main.save(path_main)
            adf_sub.save(path_sub)

            # Load and reconnect
            adf_main_loaded = AliasDataFrame.load(path_main)
            adf_sub_loaded = AliasDataFrame.load(path_sub)
            adf_main_loaded.register_subframe("T", adf_sub_loaded, index_columns=["track_index", "firstTFOrbit"])

            # Alias should still work
            self.assertIn("sum_xy", adf_main_loaded.aliases)
            adf_main_loaded.materialize_alias("sum_xy")

            # Compare with original
            adf_main.materialize_alias("sum_xy")
            np.testing.assert_array_almost_equal(
                adf_main_loaded.df['sum_xy'].values,
                adf_main.df['sum_xy'].values,
                decimal=5
            )


class TestSubframeEdgeCases(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_undefined_subframe_raises(self):
        """Referencing undefined subframe should raise clear error"""
        df_main = pd.DataFrame({'x': [1, 2, 3]})
        adf_main = AliasDataFrame(df_main)

        adf_main.add_alias("bad", "x + NONEXISTENT.y")

        with self.assertRaises(Exception) as cm:
            adf_main.materialize_alias("bad")

        # Error should mention the undefined reference
        self.assertTrue(
            "NONEXISTENT" in str(cm.exception) or
            "undefined" in str(cm.exception).lower()
        )

    def test_missing_column_in_subframe_raises(self):
        """Referencing non-existent column in subframe should raise clear error"""
        df_main = pd.DataFrame({'key': [0, 1, 2], 'x': [1, 2, 3]})
        df_sub = pd.DataFrame({'key': [0, 1, 2], 'y': [10, 20, 30]})

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("S", adf_sub, index_columns="key")
        adf_main.add_alias("bad", "x + S.nonexistent")

        with self.assertRaises(KeyError) as cm:
            adf_main.materialize_alias("bad")

        self.assertIn("S", str(cm.exception))
        self.assertIn("nonexistent", str(cm.exception))

    def test_missing_join_key_in_main_raises(self):
        """If main frame doesn't have join key, should raise clear error"""
        df_main = pd.DataFrame({'x': [1, 2, 3]})  # No 'key' column
        df_sub = pd.DataFrame({'key': [0, 1, 2], 'y': [10, 20, 30]})

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("S", adf_sub, index_columns="key")
        adf_main.add_alias("result", "S.y")

        with self.assertRaises(Exception):
            adf_main.materialize_alias("result")

    def test_empty_subframe(self):
        """Empty subframe should produce all NaN"""
        df_main = pd.DataFrame({'key': [0, 1, 2], 'x': [1.0, 2.0, 3.0]})
        df_sub = pd.DataFrame({'key': [], 'y': []})

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("S", adf_sub, index_columns="key")
        adf_main.add_alias("result", "S.y")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            adf_main.materialize_alias("result")

        # All rows preserved, all NaN
        self.assertEqual(len(adf_main.df), 3)
        self.assertTrue(adf_main.df['result'].isna().all())

    def test_duplicate_keys_in_subframe(self):
        """Duplicate keys in subframe should work (takes first match by default)"""
        df_main = pd.DataFrame({'key': [0, 1], 'x': [1.0, 2.0]})
        df_sub = pd.DataFrame({
            'key': [0, 0, 1, 1],  # Duplicates
            'y': [10.0, 11.0, 20.0, 21.0]
        })

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("S", adf_sub, index_columns="key")
        adf_main.add_alias("result", "S.y")

        # This may produce multiple rows or take first - behavior should be defined
        # For now, just ensure it doesn't crash
        adf_main.materialize_alias("result")
        self.assertIn("result", adf_main.df.columns)


class TestSubframeVsFlattened(unittest.TestCase):
    """Test that subframe access matches direct column access (unit test for T.mP3 == mP3)"""

    def test_subframe_column_equals_direct_column(self):
        """T.mP3 should equal mP3 when both are present"""
        # Simulate scenario where main frame has mP3 and subframe T also has mP3
        df_main = pd.DataFrame({
            'key': [0, 1, 2],
            'mP3': [0.1, 0.2, 0.3]  # Direct column
        })
        df_sub = pd.DataFrame({
            'key': [0, 1, 2],
            'mP3': [0.1, 0.2, 0.3]  # Same values in subframe
        })

        adf_main = AliasDataFrame(df_main)
        adf_sub = AliasDataFrame(df_sub)
        adf_main.register_subframe("T", adf_sub, index_columns="key")
        adf_main.add_alias("mP3_from_T", "T.mP3")
        adf_main.materialize_alias("mP3_from_T")

        # T.mP3 should equal mP3
        np.testing.assert_array_almost_equal(
            adf_main.df['mP3_from_T'].values,
            adf_main.df['mP3'].values
        )


if __name__ == "__main__":
    unittest.main()
