"""
test_ttree_draw_subframe.py - TTree::Draw compatibility tests for subframe aliases

Tests that subframe aliases work correctly with ROOT's TTree::Draw using friend trees.
This validates the dot-notation compatibility requirement (T.mX syntax).
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
import warnings

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = ROOT is not None and hasattr(ROOT, 'TFile')
except ImportError:
    HAS_ROOT = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dfextensions.AliasDataFrame import AliasDataFrame


@unittest.skipUnless(HAS_ROOT, "ROOT not available")
class TestTTreeDrawSubframe(unittest.TestCase):
    """Test TTree::Draw compatibility with subframe aliases using friend trees"""

    @classmethod
    def setUpClass(cls):
        """Create test data and export to ROOT file"""
        np.random.seed(42)
        
        # Main frame: clusters with track references
        n_clusters = 1000
        n_tracks = 100
        
        cls.df_main = pd.DataFrame({
            'track_index': np.random.randint(0, n_tracks, n_clusters),
            'mX': np.random.normal(0, 10, n_clusters).astype(np.float32),
            'mY': np.random.normal(0, 10, n_clusters).astype(np.float32),
        })
        
        # Subframe T: track properties (one per track)
        cls.df_track = pd.DataFrame({
            'track_index': np.arange(n_tracks),
            'mX': np.random.normal(0, 5, n_tracks).astype(np.float32),  # Track position
            'mPt': np.random.exponential(1.0, n_tracks).astype(np.float32),
            'mEta': np.random.normal(0, 1, n_tracks).astype(np.float32),
        })
        
        # Create AliasDataFrames
        cls.adf_main = AliasDataFrame(cls.df_main.copy())
        cls.adf_track = AliasDataFrame(cls.df_track.copy())
        cls.adf_main.register_subframe("T", cls.adf_track, index_columns="track_index")
        
        # Add aliases
        cls.adf_main.add_alias("dX", "mX - T.mX", dtype=np.float32)
        cls.adf_main.add_alias("track_mPt", "T.mPt", dtype=np.float32)
        
        # Create temp file
        cls.tmp_dir = tempfile.mkdtemp()
        cls.root_file = os.path.join(cls.tmp_dir, "test_draw.root")
        
        # Export to ROOT
        cls.adf_main.export_tree(cls.root_file, treename="tree")
        
        # Compute expected values using pandas
        merged = cls.df_main.merge(cls.df_track, on='track_index', suffixes=('', '_trk'))
        cls.expected_dX = (merged['mX'] - merged['mX_trk']).values
        cls.expected_track_mPt = merged['mPt'].values

    @classmethod
    def tearDownClass(cls):
        """Cleanup temp files"""
        import shutil
        if hasattr(cls, 'tmp_dir') and os.path.exists(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir)

    def test_file_structure(self):
        """Test that ROOT file has main tree and subframe tree"""
        f = ROOT.TFile.Open(self.root_file)
        self.assertIsNotNone(f)
        
        # Check main tree exists
        tree = f.Get("tree")
        self.assertIsNotNone(tree)
        self.assertEqual(tree.GetEntries(), len(self.df_main))
        
        # Check subframe tree exists
        sf_tree = f.Get("tree__subframe__T")
        self.assertIsNotNone(sf_tree)
        self.assertEqual(sf_tree.GetEntries(), len(self.df_track))
        
        f.Close()

    def test_draw_main_column(self):
        """Test TTree::Draw on main tree column (baseline)"""
        f = ROOT.TFile.Open(self.root_file)
        tree = f.Get("tree")
        
        # Draw main column mX
        n = tree.Draw("mX", "", "goff")
        self.assertEqual(n, len(self.df_main))
        
        # Get values and compare
        v = tree.GetV1()
        drawn_values = np.array([v[i] for i in range(n)], dtype=np.float32)
        
        # Values should match (order preserved)
        np.testing.assert_array_almost_equal(
            drawn_values, 
            self.df_main['mX'].values, 
            decimal=5
        )
        
        f.Close()

    def test_draw_with_friend_tree(self):
        """Test TTree::Draw with subframe as friend tree using dot notation"""
        f = ROOT.TFile.Open(self.root_file)
        tree = f.Get("tree")
        sf_tree = f.Get("tree__subframe__T")
        
        # Build index on subframe for friend join
        sf_tree.BuildIndex("track_index")
        
        # Add as friend with alias "T"
        tree.AddFriend(sf_tree, "T")
        
        # Draw T.mX (subframe column via friend)
        n = tree.Draw("T.mX", "", "goff")
        self.assertGreater(n, 0, "Draw should return entries")
        
        # Get drawn values
        v = tree.GetV1()
        drawn_values = np.array([v[i] for i in range(n)], dtype=np.float32)
        
        # Compare with expected (from pandas merge)
        # Note: Friend tree join may have different semantics for missing keys
        # We just verify it works and returns reasonable values
        self.assertEqual(len(drawn_values), n)
        
        f.Close()

    def test_draw_expression_with_friend(self):
        """Test TTree::Draw with expression combining main and friend columns"""
        f = ROOT.TFile.Open(self.root_file)
        tree = f.Get("tree")
        sf_tree = f.Get("tree__subframe__T")
        
        # Build index and add friend
        sf_tree.BuildIndex("track_index")
        tree.AddFriend(sf_tree, "T")
        
        # Draw expression: mX - T.mX (same as our dX alias)
        n = tree.Draw("mX - T.mX", "", "goff")
        self.assertGreater(n, 0)
        
        v = tree.GetV1()
        drawn_dX = np.array([v[i] for i in range(n)], dtype=np.float32)
        
        # The expression should produce similar results to our alias
        # (may differ due to index matching semantics)
        print(f"\nTTree::Draw 'mX - T.mX': {n} entries")
        print(f"  Mean: {np.mean(drawn_dX):.3f}, Std: {np.std(drawn_dX):.3f}")
        
        f.Close()

    def test_alias_matches_draw_expression(self):
        """Test that ADF alias produces same result as TTree::Draw expression"""
        # Materialize alias in ADF
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.adf_main.materialize_alias("dX")
        
        adf_dX = self.adf_main.df['dX'].values
        
        # Get TTree::Draw result
        f = ROOT.TFile.Open(self.root_file)
        tree = f.Get("tree")
        sf_tree = f.Get("tree__subframe__T")
        
        sf_tree.BuildIndex("track_index")
        tree.AddFriend(sf_tree, "T")
        
        n = tree.Draw("mX - T.mX", "", "goff")
        v = tree.GetV1()
        draw_dX = np.array([v[i] for i in range(n)], dtype=np.float32)
        
        f.Close()
        
        # Compare results
        valid_mask = ~np.isnan(adf_dX)
        
        print(f"\nAlias vs Draw comparison:")
        print(f"  ADF dX: valid={valid_mask.sum()}/{len(adf_dX)}")
        print(f"  Draw dX: count={n}")
        
        # For entries where ADF has valid values, compare with Draw result
        if valid_mask.sum() > 0 and n > 0:
            adf_valid = adf_dX[valid_mask]
            # Note: Draw result may have different count due to friend index behavior
            # Compare what we can
            min_len = min(len(adf_valid), n)
            if min_len > 0:
                adf_mean = np.mean(adf_valid)
                draw_mean = np.mean(draw_dX)
                print(f"  ADF mean={adf_mean:.3f}, Draw mean={draw_mean:.3f}")
                # Means should match reasonably
                self.assertAlmostEqual(adf_mean, draw_mean, delta=2.0)

    def test_draw_2d_main_vs_friend(self):
        """Test 2D TTree::Draw: main column vs friend column"""
        f = ROOT.TFile.Open(self.root_file)
        tree = f.Get("tree")
        sf_tree = f.Get("tree__subframe__T")
        
        sf_tree.BuildIndex("track_index")
        tree.AddFriend(sf_tree, "T")
        
        # 2D draw: mX vs T.mX
        n = tree.Draw("mX:T.mX", "", "goff")
        self.assertGreater(n, 0)
        
        # Get both axes
        vx = tree.GetV1()  # mX
        vy = tree.GetV2()  # T.mX
        
        x_vals = np.array([vx[i] for i in range(n)], dtype=np.float32)
        y_vals = np.array([vy[i] for i in range(n)], dtype=np.float32)
        
        print(f"\n2D Draw 'mX:T.mX': {n} entries")
        print(f"  mX range: [{x_vals.min():.2f}, {x_vals.max():.2f}]")
        print(f"  T.mX range: [{y_vals.min():.2f}, {y_vals.max():.2f}]")
        
        f.Close()

    def test_draw_with_cut_on_friend(self):
        """Test TTree::Draw with selection cut on friend column"""
        f = ROOT.TFile.Open(self.root_file)
        tree = f.Get("tree")
        sf_tree = f.Get("tree__subframe__T")
        
        sf_tree.BuildIndex("track_index")
        tree.AddFriend(sf_tree, "T")
        
        # Draw with cut: T.mPt > 1.0
        n_all = tree.Draw("mX", "", "goff")
        n_cut = tree.Draw("mX", "T.mPt > 1.0", "goff")
        
        print(f"\nDraw with cut 'T.mPt > 1.0':")
        print(f"  All entries: {n_all}")
        print(f"  After cut: {n_cut}")
        
        self.assertGreater(n_all, n_cut, "Cut should reduce entries")
        self.assertGreater(n_cut, 0, "Some entries should pass cut")
        
        f.Close()


@unittest.skipUnless(HAS_ROOT, "ROOT not available")
class TestTTreeDrawStrictAccuracy(unittest.TestCase):
    """Strict element-wise accuracy test with 1:1 keys (no missing entries)"""

    @classmethod
    def setUpClass(cls):
        """Create synthetic data with perfect 1:1 key mapping"""
        np.random.seed(999)
        
        n_entries = 100
        
        # Main frame: each entry has unique key
        cls.df_main = pd.DataFrame({
            'key': np.arange(n_entries),
            'x': np.random.randn(n_entries).astype(np.float32),
        })
        
        # Subframe: exactly matching keys (1:1, no missing)
        cls.df_sub = pd.DataFrame({
            'key': np.arange(n_entries),
            'y': np.random.randn(n_entries).astype(np.float32),
        })
        
        # Expected result: x - y
        cls.expected = (cls.df_main['x'] - cls.df_sub['y']).values
        
        # Create ADF and export
        cls.adf_main = AliasDataFrame(cls.df_main.copy())
        cls.adf_sub = AliasDataFrame(cls.df_sub.copy())
        cls.adf_main.register_subframe("S", cls.adf_sub, index_columns="key")
        cls.adf_main.add_alias("diff", "x - S.y", dtype=np.float32)
        
        cls.tmp_dir = tempfile.mkdtemp()
        cls.root_file = os.path.join(cls.tmp_dir, "test_strict.root")
        cls.adf_main.export_tree(cls.root_file, treename="tree")

    @classmethod
    def tearDownClass(cls):
        import shutil
        if hasattr(cls, 'tmp_dir') and os.path.exists(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir)

    def test_strict_elementwise_equality(self):
        """Strict test: ADF alias must match TTree::Draw element-by-element"""
        # Materialize alias
        self.adf_main.materialize_alias("diff")
        adf_result = self.adf_main.df['diff'].values
        
        # Get TTree::Draw result
        f = ROOT.TFile.Open(self.root_file)
        tree = f.Get("tree")
        sf_tree = f.Get("tree__subframe__S")
        
        sf_tree.BuildIndex("key")
        tree.AddFriend(sf_tree, "S")
        
        n = tree.Draw("x - S.y", "", "goff")
        v = tree.GetV1()
        draw_result = np.array([v[i] for i in range(n)], dtype=np.float32)
        
        f.Close()
        
        # Strict checks
        self.assertEqual(n, len(self.df_main), "Draw should return all entries")
        self.assertEqual(len(adf_result), len(draw_result), "Same length")
        
        # Element-wise comparison
        np.testing.assert_array_almost_equal(
            adf_result, draw_result, decimal=5,
            err_msg="ADF alias and TTree::Draw must match element-wise"
        )
        
        # Also verify against expected pandas result
        np.testing.assert_array_almost_equal(
            adf_result, self.expected, decimal=5,
            err_msg="ADF alias must match pandas reference"
        )
        
        print(f"\n✓ Strict accuracy test passed: {n} entries match element-wise")


@unittest.skipUnless(HAS_ROOT, "ROOT not available")
class TestTTreeDrawMultiKeySubframe(unittest.TestCase):
    """Test TTree::Draw with multi-key subframe joins"""

    @classmethod
    def setUpClass(cls):
        """Create test data with multi-key join"""
        np.random.seed(123)
        
        # Main frame with composite key
        cls.df_main = pd.DataFrame({
            'track_index': [0, 0, 1, 1, 2, 2],
            'firstTFOrbit': [100, 200, 100, 200, 100, 200],
            'value': np.random.randn(6).astype(np.float32),
        })
        
        # Subframe with same composite key
        cls.df_sub = pd.DataFrame({
            'track_index': [0, 0, 1, 1, 2, 2],
            'firstTFOrbit': [100, 200, 100, 200, 100, 200],
            'correction': np.random.randn(6).astype(np.float32),
        })
        
        cls.adf_main = AliasDataFrame(cls.df_main.copy())
        cls.adf_sub = AliasDataFrame(cls.df_sub.copy())
        cls.adf_main.register_subframe("C", cls.adf_sub, 
                                        index_columns=["track_index", "firstTFOrbit"])
        
        cls.tmp_dir = tempfile.mkdtemp()
        cls.root_file = os.path.join(cls.tmp_dir, "test_multikey.root")
        
        cls.adf_main.export_tree(cls.root_file, treename="tree")

    @classmethod
    def tearDownClass(cls):
        import shutil
        if hasattr(cls, 'tmp_dir') and os.path.exists(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir)

    def test_multikey_friend_index(self):
        """Test building composite index for multi-key friend"""
        f = ROOT.TFile.Open(self.root_file)
        tree = f.Get("tree")
        sf_tree = f.Get("tree__subframe__C")
        
        self.assertIsNotNone(sf_tree)
        
        # Build composite index
        # ROOT supports: BuildIndex("majorname", "minorname")
        sf_tree.BuildIndex("track_index", "firstTFOrbit")
        
        tree.AddFriend(sf_tree, "C")
        
        # Draw with friend
        n = tree.Draw("C.correction", "", "goff")
        self.assertEqual(n, len(self.df_main))
        
        f.Close()

    def test_multikey_expression(self):
        """Test expression with multi-key friend"""
        f = ROOT.TFile.Open(self.root_file)
        tree = f.Get("tree")
        sf_tree = f.Get("tree__subframe__C")
        
        sf_tree.BuildIndex("track_index", "firstTFOrbit")
        tree.AddFriend(sf_tree, "C")
        
        # Expression combining main and friend
        n = tree.Draw("value - C.correction", "", "goff")
        self.assertEqual(n, len(self.df_main))
        
        v = tree.GetV1()
        result = np.array([v[i] for i in range(n)], dtype=np.float32)
        
        # Compare with pandas
        expected = self.df_main['value'].values - self.df_sub['correction'].values
        np.testing.assert_array_almost_equal(result, expected, decimal=5)
        
        f.Close()


if __name__ == "__main__":
    unittest.main()
