"""
Phase 8 ROOT Integration Tests: Method Broadcasting

Tests that verify broadcast code compiles and executes correctly with ROOT.
These tests are skipped if ROOT is not available.
"""

import pytest
from typing import List

# Skip all tests if ROOT is not available
ROOT = pytest.importorskip("ROOT")

from RDataFrameDSL.ir_builder import IRBuilder
from RDataFrameDSL.type_inferrer import TypeInferrer
from RDataFrameDSL.backend_cpp import CppCodeGenerator, FunctionLibrary
from RDataFrameDSL.dsl_compiler import DSLCompiler


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def setup_root():
    """Set up ROOT environment once per module."""
    # Suppress ROOT startup messages
    ROOT.gROOT.SetBatch(True)
    # Declare TLorentzVector header
    ROOT.gInterpreter.Declare('#include <TLorentzVector.h>')
    ROOT.gInterpreter.Declare('#include <TVector3.h>')
    ROOT.gInterpreter.Declare('#include <TParticle.h>')
    yield ROOT


@pytest.fixture
def lorentz_vectors(setup_root):
    """Create test RVec<TLorentzVector>."""
    import uuid
    unique_id = uuid.uuid4().hex[:8]
    func_name = f"create_test_tracks_{unique_id}"
    
    # Create some TLorentzVector objects with known values
    code = f"""
    ROOT::RVec<TLorentzVector> {func_name}() {{
        ROOT::RVec<TLorentzVector> tracks;
        tracks.push_back(TLorentzVector(10, 0, 0, 15));   // Pt=10, Px=10
        tracks.push_back(TLorentzVector(0, 20, 0, 25));   // Pt=20, Py=20
        tracks.push_back(TLorentzVector(30, 40, 0, 60));  // Pt=50, hypot(30,40)
        return tracks;
    }}
    """
    ROOT.gInterpreter.Declare(code)
    return getattr(ROOT, func_name)()


@pytest.fixture
def track_compiler():
    """DSLCompiler with RVec<TLorentzVector> schema."""
    schema = {'tracks': 'RVec<TLorentzVector>'}
    return DSLCompiler(schema)


# =============================================================================
# Test F01: Basic Method Broadcast
# =============================================================================

class TestF01BasicMethodBroadcast:
    """F01: tracks.Pt() executes correctly."""
    
    def test_method_broadcast_compiles(self, setup_root, track_compiler):
        """Generated code compiles without errors."""
        track_compiler.define("track_pts", "tracks.Pt()")
        # Compilation happens on compile()
        track_compiler.compile_all()
        # If we get here, compilation succeeded
        assert True
    
    def test_method_broadcast_returns_rvec(self, setup_root, lorentz_vectors, track_compiler):
        """tracks.Pt() returns RVec<double>."""
        track_compiler.define("pts", "tracks.Pt()")
        track_compiler.compile_all()
        
        func_name = track_compiler._functions["pts"].name
        func = getattr(ROOT, func_name)
        result = func(lorentz_vectors)
        
        # Check it's an RVec
        assert hasattr(result, 'size')
        assert result.size() == 3
    
    def test_method_broadcast_correct_values(self, setup_root, lorentz_vectors, track_compiler):
        """tracks.Pt() returns correct Pt values."""
        track_compiler.define("pts", "tracks.Pt()")
        track_compiler.compile_all()
        
        func_name = track_compiler._functions["pts"].name
        func = getattr(ROOT, func_name)
        result = func(lorentz_vectors)
        
        # Check values: Pt = sqrt(Px^2 + Py^2)
        # Track 0: Pt = sqrt(10^2 + 0^2) = 10
        # Track 1: Pt = sqrt(0^2 + 20^2) = 20  
        # Track 2: Pt = sqrt(30^2 + 40^2) = 50
        assert abs(result[0] - 10.0) < 0.01
        assert abs(result[1] - 20.0) < 0.01
        assert abs(result[2] - 50.0) < 0.01


# =============================================================================
# Test F02: Multiple Methods
# =============================================================================

class TestF02MultipleMethods:
    """F02: Multiple methods (Pt, Eta, Phi) on same source."""
    
    def test_multiple_methods_compile(self, setup_root, track_compiler):
        """Multiple broadcast expressions compile."""
        track_compiler.define("pts", "tracks.Pt()")
        track_compiler.define("etas", "tracks.Eta()")
        track_compiler.define("phis", "tracks.Phi()")
        track_compiler.compile_all()
        assert True
    
    def test_px_py_pz_values(self, setup_root, lorentz_vectors, track_compiler):
        """Px, Py, Pz return correct component values."""
        track_compiler.define("px_vals", "tracks.Px()")
        track_compiler.define("py_vals", "tracks.Py()")
        track_compiler.define("pz_vals", "tracks.Pz()")
        track_compiler.compile_all()
        
        px_func = getattr(ROOT, track_compiler._functions["px_vals"].name)
        py_func = getattr(ROOT, track_compiler._functions["py_vals"].name)
        pz_func = getattr(ROOT, track_compiler._functions["pz_vals"].name)
        
        px = px_func(lorentz_vectors)
        py = py_func(lorentz_vectors)
        pz = pz_func(lorentz_vectors)
        
        # Track 0: (10, 0, 0, 15)
        assert abs(px[0] - 10.0) < 0.01
        assert abs(py[0] - 0.0) < 0.01
        assert abs(pz[0] - 0.0) < 0.01
        
        # Track 1: (0, 20, 0, 25)
        assert abs(px[1] - 0.0) < 0.01
        assert abs(py[1] - 20.0) < 0.01
        
        # Track 2: (30, 40, 0, 60)
        assert abs(px[2] - 30.0) < 0.01
        assert abs(py[2] - 40.0) < 0.01


# =============================================================================
# Test F03: Slice Then Broadcast
# =============================================================================

class TestF03SliceThenBroadcast:
    """F03: tracks[:2].Pt() - slice then broadcast."""
    
    def test_slice_then_broadcast_compiles(self, setup_root, track_compiler):
        """Slice-then-broadcast compiles."""
        track_compiler.define("lead2_pt", "tracks[:2].Pt()")
        track_compiler.compile_all()
        assert True
    
    def test_slice_then_broadcast_result_size(self, setup_root, lorentz_vectors, track_compiler):
        """Slice-then-broadcast returns correct size."""
        track_compiler.define("lead2_pt", "tracks[:2].Pt()")
        track_compiler.compile_all()
        
        func = getattr(ROOT, track_compiler._functions["lead2_pt"].name)
        result = func(lorentz_vectors)
        
        # Should only have first 2 tracks
        assert result.size() == 2
    
    def test_slice_then_broadcast_correct_values(self, setup_root, lorentz_vectors, track_compiler):
        """Slice-then-broadcast returns correct values."""
        track_compiler.define("lead2_pt", "tracks[:2].Pt()")
        track_compiler.compile_all()
        
        func = getattr(ROOT, track_compiler._functions["lead2_pt"].name)
        result = func(lorentz_vectors)
        
        assert abs(result[0] - 10.0) < 0.01  # First track
        assert abs(result[1] - 20.0) < 0.01  # Second track


# =============================================================================
# Test F04: Filter Then Broadcast
# =============================================================================

class TestF04FilterThenBroadcast:
    """F04: tracks[tracks.Pt() > 15].Eta() - filter then broadcast."""
    
    def test_filter_broadcast_compiles(self, setup_root, track_compiler):
        """Filter-then-broadcast compiles."""
        track_compiler.define("high_pt_eta", "tracks[tracks.Pt() > 15.0].Eta()")
        track_compiler.compile_all()
        assert True
    
    def test_filter_broadcast_correct_count(self, setup_root, lorentz_vectors, track_compiler):
        """Filter-then-broadcast filters correctly."""
        track_compiler.define("high_pt_eta", "tracks[tracks.Pt() > 15.0].Eta()")
        track_compiler.compile_all()
        
        func = getattr(ROOT, track_compiler._functions["high_pt_eta"].name)
        result = func(lorentz_vectors)
        
        # Tracks with Pt > 15: Track 1 (Pt=20) and Track 2 (Pt=50)
        assert result.size() == 2


# =============================================================================
# Test F05: Arithmetic on Broadcast Results
# =============================================================================

# Phase 9: Now enabled - RVec arithmetic type propagation implemented
class TestF05ArithmeticOnBroadcast:
    """F05: sqrt(tracks.Px()**2 + tracks.Py()**2).
    
    This requires type propagation for arithmetic on RVec results.
    When tracks.Px() returns RVec<double>, then tracks.Px()**2 should
    also be RVec<double>. This is Phase 9 functionality.
    """
    
    def test_arithmetic_on_broadcast_compiles(self, setup_root, track_compiler):
        """Arithmetic on broadcast results compiles."""
        track_compiler.define("manual_pt", "sqrt(tracks.Px()**2 + tracks.Py()**2)")
        track_compiler.compile_all()
        assert True
    
    def test_arithmetic_matches_method(self, setup_root, lorentz_vectors, track_compiler):
        """Manual Pt calculation matches Pt() method."""
        track_compiler.define("method_pt", "tracks.Pt()")
        track_compiler.define("manual_pt", "sqrt(tracks.Px()**2 + tracks.Py()**2)")
        track_compiler.compile_all()
        
        method_func = getattr(ROOT, track_compiler._functions["method_pt"].name)
        manual_func = getattr(ROOT, track_compiler._functions["manual_pt"].name)
        
        method_result = method_func(lorentz_vectors)
        manual_result = manual_func(lorentz_vectors)
        
        # Should match within floating point tolerance
        for i in range(3):
            assert abs(method_result[i] - manual_result[i]) < 0.01


# =============================================================================
# Test F06: Method Returning Object
# =============================================================================

class TestF06MethodReturningObject:
    """F06: tracks.Vect() returns RVec<TVector3>."""
    
    def test_vect_broadcast_compiles(self, setup_root, track_compiler):
        """Vect() broadcast compiles."""
        track_compiler.define("track_vects", "tracks.Vect()")
        track_compiler.compile_all()
        assert True
    
    def test_vect_returns_vector3(self, setup_root, lorentz_vectors, track_compiler):
        """Vect() returns RVec of TVector3."""
        track_compiler.define("track_vects", "tracks.Vect()")
        track_compiler.compile_all()
        
        func = getattr(ROOT, track_compiler._functions["track_vects"].name)
        result = func(lorentz_vectors)
        
        # Check size
        assert result.size() == 3
        
        # Check first element is TVector3-like (has X(), Y(), Z())
        first_vec = result[0]
        assert abs(first_vec.X() - 10.0) < 0.01  # Track 0 Px = 10


# =============================================================================
# Test F07: Empty Vector Handling
# =============================================================================

class TestF07EmptyVector:
    """F07: Empty RVec<TLorentzVector> doesn't crash."""
    
    def test_broadcast_on_empty_vector(self, setup_root, track_compiler):
        """Broadcasting on empty vector returns empty result."""
        track_compiler.define("pts", "tracks.Pt()")
        track_compiler.compile_all()
        
        # Create empty vector
        empty = ROOT.ROOT.RVec["TLorentzVector"]()
        
        func = getattr(ROOT, track_compiler._functions["pts"].name)
        result = func(empty)
        
        assert result.size() == 0


# =============================================================================
# Test F08: RDataFrame Integration
# =============================================================================

class TestF08RDataFrameIntegration:
    """F08: Broadcasting works in RDataFrame pipeline."""
    
    def test_broadcast_in_rdf(self, setup_root, track_compiler):
        """Broadcasting works in RDataFrame Define()."""
        import uuid
        unique_id = uuid.uuid4().hex[:8]
        func_name_helper = f"make_tracks_row_{unique_id}"
        
        # Create a simple RDataFrame with tracks column
        ROOT.gInterpreter.Declare(f"""
        ROOT::RVec<TLorentzVector> {func_name_helper}() {{
            ROOT::RVec<TLorentzVector> tracks;
            tracks.push_back(TLorentzVector(10, 0, 0, 15));
            tracks.push_back(TLorentzVector(0, 20, 0, 25));
            return tracks;
        }}
        """)
        
        # Create RDF with 5 rows
        rdf = ROOT.RDataFrame(5)
        rdf = rdf.Define("tracks", f"{func_name_helper}()")
        
        # Use our broadcast function
        track_compiler.define("track_pts", "tracks.Pt()")
        track_compiler.compile_all()
        
        func_name = track_compiler._functions["track_pts"].name
        rdf = rdf.Define("pts", f"{func_name}(tracks)")
        
        # Verify it works
        result = rdf.Take["ROOT::RVec<double>"]("pts").GetValue()
        
        assert len(result) == 5  # 5 rows
        assert result[0].size() == 2  # 2 tracks per row


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
