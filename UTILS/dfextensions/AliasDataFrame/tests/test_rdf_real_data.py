"""
Real data validation tests for AliasDataFrame RDataFrame integration.

These tests run against actual calibration data files to validate
that RDataFrame produces results matching AliasDataFrame.

Usage:
    # Set environment variable to calibration file
    export ADF_TEST_CALIB_FILE=/path/to/calibration.root
    
    # Run tests
    pytest tests/test_rdf_real_data.py -v -s

Tests are skipped if calibration file is not available.
"""

import pytest
import numpy as np
import os
import sys

# Add parent directory to path for imports
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Get calibration file path from environment
CALIB_FILE = os.environ.get('ADF_TEST_CALIB_FILE', None)

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False

# Check if file exists
HAS_CALIB_FILE = CALIB_FILE is not None and os.path.exists(CALIB_FILE)


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
@pytest.mark.skipif(not HAS_CALIB_FILE, reason="Set ADF_TEST_CALIB_FILE environment variable")
class TestRDFRealData:
    """
    Real data validation tests.
    
    These tests verify that RDataFrame produces results matching
    AliasDataFrame's pandas-based evaluation.
    """
    
    @pytest.fixture(scope="class")
    def loaded_data(self):
        """Load calibration data once for all tests."""
        from AliasDataFrame import AliasDataFrame
        from AliasDataFrameRDF import setup_tree_with_friends
        
        aDF = AliasDataFrame.from_root(CALIB_FILE, "tree")
        tree, f = setup_tree_with_friends(CALIB_FILE, "tree", aDF.schema)
        
        return {'aDF': aDF, 'tree': tree, 'file': f}
    
    def test_dyC2_histogram(self, loaded_data):
        """
        Generate dyC2 histogram with RDataFrame.
        Validates against AliasDataFrame result.
        """
        from AliasDataFrameRDF import get_ordered_defines
        
        aDF = loaded_data['aDF']
        tree = loaded_data['tree']
        
        # RDataFrame approach
        df = ROOT.RDataFrame(tree)
        defines = get_ordered_defines(['dyC2'], aDF=aDF)
        
        for d in defines:
            print(f"  Define: {d['name']} = {d['cpp_expr']}")
            df = df.Define(d['name'], d['cpp_expr'])
        
        h = df.Histo1D(("h_dyC2", "dyC2 RDataFrame", 100, -2, 2), "dyC2")
        
        # Force evaluation
        entries_rdf = h.GetEntries()
        mean_rdf = h.GetMean()
        stddev_rdf = h.GetStdDev()
        
        print(f"\n[RDataFrame] dyC2: entries={entries_rdf}, mean={mean_rdf:.6f}, std={stddev_rdf:.6f}")
        
        # AliasDataFrame approach
        aDF.materialize_alias('dyC2')
        dyC2_adf = aDF['dyC2'].dropna()
        
        mean_adf = dyC2_adf.mean()
        stddev_adf = dyC2_adf.std()
        
        print(f"[AliasDataFrame] dyC2: entries={len(dyC2_adf)}, mean={mean_adf:.6f}, std={stddev_adf:.6f}")
        
        # Validate
        assert entries_rdf > 0, "RDataFrame histogram is empty"
        np.testing.assert_allclose(mean_rdf, mean_adf, rtol=0.01, 
                                   err_msg="Mean mismatch between RDF and ADF")
        np.testing.assert_allclose(stddev_rdf, stddev_adf, rtol=0.01,
                                   err_msg="StdDev mismatch between RDF and ADF")
        
        print("[PASS] dyC2 histogram matches between RDataFrame and AliasDataFrame")
    
    def test_dzC2_histogram(self, loaded_data):
        """Generate dzC2 histogram with RDataFrame."""
        from AliasDataFrameRDF import get_ordered_defines
        
        aDF = loaded_data['aDF']
        tree = loaded_data['tree']
        
        df = ROOT.RDataFrame(tree)
        defines = get_ordered_defines(['dzC2'], aDF=aDF)
        
        for d in defines:
            df = df.Define(d['name'], d['cpp_expr'])
        
        h = df.Histo1D(("h_dzC2", "dzC2 RDataFrame", 100, -2, 2), "dzC2")
        
        entries_rdf = h.GetEntries()
        mean_rdf = h.GetMean()
        
        # AliasDataFrame
        aDF.materialize_alias('dzC2')
        dzC2_adf = aDF['dzC2'].dropna()
        mean_adf = dzC2_adf.mean()
        
        print(f"\n[RDataFrame] dzC2: entries={entries_rdf}, mean={mean_rdf:.6f}")
        print(f"[AliasDataFrame] dzC2: entries={len(dzC2_adf)}, mean={mean_adf:.6f}")
        
        assert entries_rdf > 0
        np.testing.assert_allclose(mean_rdf, mean_adf, rtol=0.01)
        
        print("[PASS] dzC2 histogram matches")
    
    def test_2d_histogram(self, loaded_data):
        """Generate 2D histogram dyC2 vs dzC2."""
        from AliasDataFrameRDF import get_ordered_defines
        
        aDF = loaded_data['aDF']
        tree = loaded_data['tree']
        
        df = ROOT.RDataFrame(tree)
        
        # Need both aliases
        defines = get_ordered_defines(['dyC2', 'dzC2'], aDF=aDF)
        
        for d in defines:
            df = df.Define(d['name'], d['cpp_expr'])
        
        h2 = df.Histo2D(
            ("h2_dydzC2", "dyC2 vs dzC2", 50, -2, 2, 50, -2, 2),
            "dyC2", "dzC2"
        )
        
        entries = h2.GetEntries()
        
        print(f"\n[RDataFrame] 2D histogram entries: {entries}")
        
        assert entries > 0, "2D histogram is empty"
        print("[PASS] 2D histogram generated successfully")
    
    def test_correlation_coefficient(self, loaded_data):
        """Compare correlation coefficients between RDF and ADF."""
        from AliasDataFrameRDF import get_ordered_defines
        
        aDF = loaded_data['aDF']
        tree = loaded_data['tree']
        
        df = ROOT.RDataFrame(tree)
        defines = get_ordered_defines(['dyC2', 'dzC2'], aDF=aDF)
        
        for d in defines:
            df = df.Define(d['name'], d['cpp_expr'])
        
        # Compute correlation via 2D histogram
        h2 = df.Histo2D(
            ("h2_corr", "", 100, -5, 5, 100, -5, 5),
            "dyC2", "dzC2"
        )
        
        corr_rdf = h2.GetCorrelationFactor()
        
        # AliasDataFrame correlation
        aDF.materialize_aliases(names=['dyC2', 'dzC2'], with_dependencies=True)
        corr_adf = aDF[['dyC2', 'dzC2']].corr().iloc[0, 1]
        
        print(f"\n[RDataFrame] Correlation(dyC2, dzC2): {corr_rdf:.4f}")
        print(f"[AliasDataFrame] Correlation(dyC2, dzC2): {corr_adf:.4f}")
        
        np.testing.assert_allclose(corr_rdf, corr_adf, atol=0.05,
                                   err_msg="Correlation coefficient mismatch")
        
        print("[PASS] Correlation coefficients match")


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
@pytest.mark.skipif(not HAS_CALIB_FILE, reason="Set ADF_TEST_CALIB_FILE environment variable")
class TestRDFMultiThread:
    """Test RDataFrame with multi-threading enabled."""
    
    def test_mt_histogram(self):
        """Test histogram generation with multi-threading."""
        from AliasDataFrame import AliasDataFrame
        from AliasDataFrameRDF import get_ordered_defines, setup_tree_with_friends
        
        # Enable MT
        ROOT.EnableImplicitMT()
        n_threads = ROOT.GetThreadPoolSize()
        print(f"\n[MT] Running with {n_threads} threads")
        
        aDF = AliasDataFrame.from_root(CALIB_FILE, "tree")
        tree, f = setup_tree_with_friends(CALIB_FILE, "tree", aDF.schema)
        
        df = ROOT.RDataFrame(tree)
        defines = get_ordered_defines(['dyC2'], aDF=aDF)
        
        for d in defines:
            df = df.Define(d['name'], d['cpp_expr'])
        
        h = df.Histo1D(("h_mt", "dyC2 MT", 100, -2, 2), "dyC2")
        
        entries = h.GetEntries()
        print(f"[MT] Histogram entries: {entries}")
        
        assert entries > 0
        
        # Disable MT for other tests
        ROOT.DisableImplicitMT()
        
        print("[PASS] Multi-threaded histogram generation works")


# =============================================================================
# Summary Report
# =============================================================================

if __name__ == '__main__':
    print("="*60)
    print("Real Data Validation Tests")
    print("="*60)
    
    if not HAS_ROOT:
        print("ERROR: ROOT not available")
        sys.exit(1)
    
    if not HAS_CALIB_FILE:
        print("ERROR: Set ADF_TEST_CALIB_FILE environment variable")
        print("  export ADF_TEST_CALIB_FILE=/path/to/calibration.root")
        sys.exit(1)
    
    print(f"Calibration file: {CALIB_FILE}")
    print(f"ROOT version: {ROOT.gROOT.GetVersion()}")
    
    pytest.main([__file__, '-v', '-s', '--tb=short'])
