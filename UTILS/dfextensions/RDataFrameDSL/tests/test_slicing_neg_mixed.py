"""
Tests for mixed negative indices slicing (SLICING-1D-NEG-MIXED).

Phase 13.6.G: Support track_pt[1:-1] and similar patterns.

Limitation ID: SLICING-1D-NEG-MIXED
"""
import pytest
import numpy as np


class TestMixedNegativeSlicing:
    """Test mixed negative index slicing [a:b] where a or b can be negative."""
    
    @pytest.fixture
    def compiler_and_data(self):
        """Set up DSL compiler and test data."""
        import ROOT
        from tests.generators.toy_nd import generate_nd_2d_root
        from RDataFrameDSL import DSLCompiler
        
        filename = generate_nd_2d_root(size='S', seed=42)
        rdf = ROOT.RDataFrame("Events", filename)
        
        schema = {
            'track_pt': 'RVec<double>',
            'track_eta': 'RVec<double>',
        }
        
        return DSLCompiler(schema), rdf
    
    def test_middle_slice_1_to_minus1(self, compiler_and_data):
        """Test [1:-1] - elements from index 1 to second-to-last."""
        dsl, rdf = compiler_and_data
        
        dsl.define("middle", "track_pt[1:-1]")
        
        applied = dsl.apply(rdf)
        result = applied.Range(10).AsNumpy(['track_pt', 'middle'])
        
        for i in range(10):
            pt = np.array(result['track_pt'][i])  # Convert RVec to numpy
            expected = pt[1:-1] if len(pt) >= 2 else np.array([])
            actual = np.array(result['middle'][i])
            assert np.allclose(expected, actual), f"Event {i}: [1:-1] mismatch"
    
    def test_slice_0_to_minus1(self, compiler_and_data):
        """Test [:-1] - all but last element."""
        dsl, rdf = compiler_and_data
        
        dsl.define("all_but_last", "track_pt[0:-1]")
        
        applied = dsl.apply(rdf)
        result = applied.Range(10).AsNumpy(['track_pt', 'all_but_last'])
        
        for i in range(10):
            pt = np.array(result['track_pt'][i])  # Convert RVec to numpy
            expected = pt[:-1] if len(pt) > 0 else np.array([])
            actual = np.array(result['all_but_last'][i])
            assert np.allclose(expected, actual), f"Event {i}: [0:-1] mismatch"
    
    def test_slice_minus3_to_minus1(self, compiler_and_data):
        """Test [-3:-1] - third from last to second from last."""
        dsl, rdf = compiler_and_data
        
        dsl.define("tail_middle", "track_pt[-3:-1]")
        
        applied = dsl.apply(rdf)
        result = applied.Range(10).AsNumpy(['track_pt', 'tail_middle'])
        
        for i in range(10):
            pt = np.array(result['track_pt'][i])  # Convert RVec to numpy
            expected = pt[-3:-1] if len(pt) >= 2 else np.array([])
            actual = np.array(result['tail_middle'][i])
            # NumPy handles edge cases, DSL should match
            if len(expected) == 0:
                assert len(actual) == 0, f"Event {i}: expected empty"
            else:
                assert np.allclose(expected, actual), f"Event {i}: [-3:-1] mismatch"
    
    def test_slice_2_to_minus2(self, compiler_and_data):
        """Test [2:-2] - from third to third-from-last."""
        dsl, rdf = compiler_and_data
        
        dsl.define("inner", "track_pt[2:-2]")
        
        applied = dsl.apply(rdf)
        result = applied.Range(10).AsNumpy(['track_pt', 'inner'])
        
        for i in range(10):
            pt = np.array(result['track_pt'][i])  # Convert RVec to numpy
            expected = pt[2:-2]
            actual = np.array(result['inner'][i])
            if len(expected) == 0:
                assert len(actual) == 0, f"Event {i}: expected empty"
            else:
                assert np.allclose(expected, actual), f"Event {i}: [2:-2] mismatch"
    
    def test_empty_result_small_arrays(self, compiler_and_data):
        """Test that small arrays return empty correctly."""
        dsl, rdf = compiler_and_data
        
        # [1:-1] on array of length 0, 1, 2 should return empty
        dsl.define("middle", "track_pt[1:-1]")
        
        applied = dsl.apply(rdf)
        result = applied.AsNumpy(['track_pt', 'middle'])
        
        for i in range(len(result['track_pt'])):
            pt = np.array(result['track_pt'][i])  # Convert RVec to numpy
            expected = pt[1:-1]
            actual = np.array(result['middle'][i])
            
            assert len(expected) == len(actual), f"Event {i}: length mismatch"
            if len(expected) > 0:
                assert np.allclose(expected, actual), f"Event {i}: value mismatch"
    
    def test_out_of_bounds_clamp(self, compiler_and_data):
        """Test that large negative indices are clamped properly."""
        dsl, rdf = compiler_and_data
        
        # [-100:5] should start from 0 for small arrays
        dsl.define("clamped", "track_pt[-100:5]")
        
        applied = dsl.apply(rdf)
        result = applied.Range(10).AsNumpy(['track_pt', 'clamped'])
        
        for i in range(10):
            pt = np.array(result['track_pt'][i])  # Convert RVec to numpy
            expected = pt[-100:5]  # NumPy clamps automatically
            actual = np.array(result['clamped'][i])
            assert np.allclose(expected, actual), f"Event {i}: clamped mismatch"


class TestEdgeCases:
    """Test edge cases for slicing."""
    
    @pytest.fixture
    def compiler_and_data(self):
        """Set up DSL compiler and test data."""
        import ROOT
        from tests.generators.toy_nd import generate_nd_2d_root
        from RDataFrameDSL import DSLCompiler
        
        filename = generate_nd_2d_root(size='S', seed=123)
        rdf = ROOT.RDataFrame("Events", filename)
        
        schema = {'track_pt': 'RVec<double>'}
        return DSLCompiler(schema), rdf
    
    def test_reverse_still_works(self, compiler_and_data):
        """Verify [::-1] still works correctly after changes."""
        dsl, rdf = compiler_and_data
        
        dsl.define("rev", "track_pt[::-1]")
        
        applied = dsl.apply(rdf)
        result = applied.Range(5).AsNumpy(['track_pt', 'rev'])
        
        for i in range(5):
            pt = np.array(result['track_pt'][i])  # Convert RVec to numpy
            expected = pt[::-1]
            actual = np.array(result['rev'][i])
            assert np.allclose(expected, actual), f"Event {i}: reverse mismatch"
    
    def test_step_still_works(self, compiler_and_data):
        """Verify [::2] still works correctly after changes."""
        dsl, rdf = compiler_and_data
        
        dsl.define("stepped", "track_pt[::2]")
        
        applied = dsl.apply(rdf)
        result = applied.Range(5).AsNumpy(['track_pt', 'stepped'])
        
        for i in range(5):
            pt = np.array(result['track_pt'][i])  # Convert RVec to numpy
            expected = pt[::2]
            actual = np.array(result['stepped'][i])
            assert np.allclose(expected, actual), f"Event {i}: step mismatch"
    
    def test_positive_range_still_works(self, compiler_and_data):
        """Verify [1:3] still works correctly after changes."""
        dsl, rdf = compiler_and_data
        
        dsl.define("ranged", "track_pt[1:3]")
        
        applied = dsl.apply(rdf)
        result = applied.Range(5).AsNumpy(['track_pt', 'ranged'])
        
        for i in range(5):
            pt = np.array(result['track_pt'][i])  # Convert RVec to numpy
            expected = pt[1:3]
            actual = np.array(result['ranged'][i])
            if len(expected) == 0:
                assert len(actual) == 0
            else:
                assert np.allclose(expected, actual), f"Event {i}: range mismatch"
