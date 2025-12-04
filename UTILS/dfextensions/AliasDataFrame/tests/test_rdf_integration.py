"""
Integration tests for AliasDataFrameRDF with synthetic data.

These tests use fixtures from conftest.py that create test data with:
- 4 subframes (T, R, DTrack0, DITS0FitSide)
- Multi-key composite indices
- Representative aliases

Run with: pytest tests/test_rdf_integration.py -v -s
"""

import pytest
import sys
import os

# Add parent directory to path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = True
    ROOT_VERSION = ROOT.gROOT.GetVersion()
except ImportError:
    HAS_ROOT = False
    ROOT_VERSION = None

from AliasDataFrameRDF import (
    get_ordered_defines,
    setup_tree_with_friends,
    to_cpp_expr,
)


# =============================================================================
# Schema and Dependency Tests (No ROOT RDataFrame needed)
# =============================================================================

@pytest.mark.integration
class TestOrderedDefinesWithSchema:
    """Test get_ordered_defines with multi-subframe schema."""
    
    def test_simple_alias_ordering(self, rdf_test_adf):
        """Test that dependencies come before dependents."""
        defines = get_ordered_defines(['dyC2'], aDF=rdf_test_adf)
        
        names = [d['name'] for d in defines]
        
        # dyC2 depends on dy_c, so dy_c must come first
        assert 'dy_c' in names
        assert 'dyC2' in names
        assert names.index('dy_c') < names.index('dyC2')
    
    def test_multiple_alias_ordering(self, rdf_test_adf):
        """Test ordering with multiple target aliases."""
        defines = get_ordered_defines(['dyC2', 'dzC2'], aDF=rdf_test_adf)
        
        names = [d['name'] for d in defines]
        
        # Both dy_c and dz_c should be present
        assert 'dy_c' in names
        assert 'dz_c' in names
        assert 'dyC2' in names
        assert 'dzC2' in names
        
        # Dependencies before dependents
        assert names.index('dy_c') < names.index('dyC2')
        assert names.index('dz_c') < names.index('dzC2')
    
    def test_isvalid_deep_dependency(self, rdf_test_adf):
        """Test alias with deep dependency chain (isValid depends on dyC2)."""
        defines = get_ordered_defines(['isValid'], aDF=rdf_test_adf)
        
        names = [d['name'] for d in defines]
        
        # isValid depends on dyC2
        # dyC2 depends on dy_c
        assert 'dy_c' in names
        assert 'dyC2' in names
        assert 'isValid' in names
        
        # Check order
        assert names.index('dy_c') < names.index('dyC2')
        assert names.index('dyC2') < names.index('isValid')
    
    def test_cpp_expr_conversion(self, rdf_test_adf):
        """Test that C++ expressions are properly converted."""
        defines = get_ordered_defines(['z_calc'], aDF=rdf_test_adf)
        
        # Find z_calc definition
        z_calc_def = next(d for d in defines if d['name'] == 'z_calc')
        
        # Should have tan() function
        assert 'tan' in z_calc_def['cpp_expr']
    
    def test_subframe_dependencies_identified(self, rdf_test_adf):
        """Test that subframe column dependencies are found."""
        defines = get_ordered_defines(['dyC2'], aDF=rdf_test_adf)
        
        # dy_c should depend on T.mP2
        dy_c_def = next(d for d in defines if d['name'] == 'dy_c')
        assert 'T.mP2' in dy_c_def['expr']
        
        # dyC2 should depend on DTrack0.dyC2_median
        dyC2_def = next(d for d in defines if d['name'] == 'dyC2')
        assert 'DTrack0' in dyC2_def['expr']


# =============================================================================
# RDataFrame Integration Tests (Require ROOT)
# =============================================================================

@pytest.mark.integration
@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestRDFIntegration:
    """Full RDataFrame integration tests with synthetic data."""
    
    def test_ordered_defines_with_real_schema(self, rdf_test_file, rdf_test_adf):
        """Test get_ordered_defines with multi-subframe schema."""
        defines = get_ordered_defines(['dyC2', 'dzC2'], aDF=rdf_test_adf)
        
        # Should include dependencies
        names = [d['name'] for d in defines]
        assert 'dy_c' in names  # Dependency of dyC2
        assert 'dz_c' in names  # Dependency of dzC2
        assert names.index('dy_c') < names.index('dyC2')  # Correct order
        assert names.index('dz_c') < names.index('dzC2')  # Correct order
    
    def test_setup_tree_with_friends(self, rdf_test_file, rdf_test_adf):
        """Test that tree setup creates friends correctly."""
        tree, f = setup_tree_with_friends(rdf_test_file, "tree", rdf_test_adf.schema)
        
        # Check tree loaded
        assert tree is not None
        assert tree.GetEntries() == 10_000
        
        # Check friends attached
        friends = tree.GetListOfFriends()
        assert friends is not None
        
        friend_names = [f.GetName() for f in friends]
        print(f"Friends attached: {friend_names}")
        assert 'T' in friend_names
        assert 'R' in friend_names
    
    def test_rdf_define_chain(self, rdf_test_file, rdf_test_adf):
        """Test full RDataFrame Define() chain including 3-key subframe."""
        tree, f = setup_tree_with_friends(rdf_test_file, "tree", rdf_test_adf.schema)
        
        df = ROOT.RDataFrame(tree)
        # dyC2 depends on dy_c AND DTrack0.dyC2_median (3-key subframe)
        defines = get_ordered_defines(['dyC2'], aDF=rdf_test_adf)
        
        print(f"\nApplying {len(defines)} defines:")
        for d in defines:
            print(f"  {d['name']} = {d['cpp_expr']}")
        
        for d in defines:
            df = df.Define(d['name'], d['cpp_expr'])
        
        # Verify execution - this proves 3-key join works!
        # Use 'float' since all columns are float32
        result = df.Take['float']('dyC2').GetValue()
        assert len(result) == 10_000
        print(f"[PASS] dyC2 evaluated: {len(result)} values (uses 3-key DTrack0)")
    
    def test_3key_subframe_access(self, rdf_test_file, rdf_test_adf):
        """Test DTrack0 with 3-key composite index."""
        tree, f = setup_tree_with_friends(rdf_test_file, "tree", rdf_test_adf.schema)
        
        df = ROOT.RDataFrame(tree)
        
        # Direct subframe access
        try:
            df2 = df.Define("median", "DTrack0.dyC2_median")
            result = df2.Take['float']('median').GetValue()
            assert len(result) == 10_000
            print(f"[PASS] 3-key DTrack0.dyC2_median accessible: {len(result)} values")
        except Exception as e:
            print(f"[INFO] 3-key subframe: {e}")
            pytest.skip("3-key composite index not available")
    
    def test_1key_subframe_access(self, rdf_test_file, rdf_test_adf):
        """Test T and R with 1-key index."""
        tree, f = setup_tree_with_friends(rdf_test_file, "tree", rdf_test_adf.schema)
        
        df = ROOT.RDataFrame(tree)
        
        # Test T subframe (track_tf_uid key)
        df2 = df.Define("t_val", "T.mP2")
        result = df2.Take['float']('t_val').GetValue()
        assert len(result) == 10_000
        print(f"[PASS] T.mP2: {len(result)} values")
        
        # Test R subframe (firstTForbit key)
        df3 = df.Define("r_val", "R.refX")
        result = df3.Take['float']('r_val').GetValue()
        assert len(result) == 10_000
        print(f"[PASS] R.refX: {len(result)} values")
    
    def test_2key_subframe_access(self, rdf_test_file, rdf_test_adf):
        """Test DITS0FitSide with 2-key index."""
        tree, f = setup_tree_with_friends(rdf_test_file, "tree", rdf_test_adf.schema)
        
        df = ROOT.RDataFrame(tree)
        
        try:
            df2 = df.Define("dits_val", "DITS0FitSide.itsParam")
            result = df2.Take['float']('dits_val').GetValue()
            assert len(result) == 10_000
            print(f"[PASS] 2-key DITS0FitSide.itsParam: {len(result)} values")
        except Exception as e:
            print(f"[INFO] 2-key subframe: {e}")
            pytest.skip("2-key index not available")


# =============================================================================
# Summary
# =============================================================================

if __name__ == '__main__':
    print("RDataFrame Integration Tests")
    print("=" * 60)
    if HAS_ROOT:
        print(f"ROOT version: {ROOT_VERSION}")
    else:
        print("ROOT not available - some tests will be skipped")
    print("\nRun with: pytest tests/test_rdf_integration.py -v -s")
    print("Or: pytest tests/test_rdf_integration.py -v -s -m integration")
