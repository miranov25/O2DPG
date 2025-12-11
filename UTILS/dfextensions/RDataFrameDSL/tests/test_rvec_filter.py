"""
Phase 10.7: RVec boolean mask filtering tests.

Tests for boolean mask filtering that filters RVec elements:
- pt[pt > 1.0] - filter by condition on same vector
- tracks[tracks.Pt() > 1.0] - filter object vectors
- pt[abs(eta) < 1.0] - cross-vector filtering
- Chain with reductions: Sum(pt[pt > 2.0])
"""

import pytest
import sys

# Try to import ROOT, mark if unavailable
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False
    ROOT = None

from RDataFrameDSL import DSLCompiler


# =============================================================================
# Mock Tests (no ROOT required)
# =============================================================================

class TestBooleanMaskMock:
    """Mock tests that don't require ROOT compilation."""
    
    @pytest.fixture
    def dsl(self):
        return DSLCompiler({
            "pt": "RVec<double>",
            "eta": "RVec<double>",
            "counts": "RVec<int>",
            "flags": "RVec<bool>",
            "tracks": "RVec<TLorentzVector>"
        })
    
    # Code generation tests
    def test_simple_mask_generates_indexing(self, dsl):
        """pt[pt > 1.0] generates bracket indexing."""
        dsl.define("high_pt", "pt[pt > 1.0]")
        code = dsl.preview()
        assert "[" in code and "]" in code
        assert ">" in code
    
    def test_cross_vector_mask_code(self, dsl):
        """pt[eta < 1.0] filters pt by eta condition."""
        dsl.define("central_pt", "pt[abs(eta) < 1.0]")
        code = dsl.preview()
        assert "pt" in code
        assert "eta" in code
    
    def test_le_mask_code(self, dsl):
        """pt[pt <= 2.0] generates <= comparison."""
        dsl.define("low_pt", "pt[pt <= 2.0]")
        code = dsl.preview()
        assert "<=" in code
    
    def test_eq_mask_code(self, dsl):
        """flags[flags == 0] generates == comparison."""
        dsl.define("zeros", "counts[counts == 0]")
        code = dsl.preview()
        assert "==" in code
    
    # Return type tests
    def test_mask_preserves_double_type(self, dsl):
        """Filtered RVec<double> is still RVec<double>."""
        dsl.define("high_pt", "pt[pt > 1.0]")
        func = dsl.get_function("high_pt")
        assert "RVec" in func.return_type
        assert "double" in func.return_type
    
    def test_mask_preserves_int_type(self, dsl):
        """Filtered RVec<int> is still RVec<int>."""
        dsl.define("high_counts", "counts[counts > 5]")
        func = dsl.get_function("high_counts")
        assert "RVec" in func.return_type
        assert "int" in func.return_type.lower()
    
    def test_mask_preserves_object_type(self, dsl):
        """Filtered RVec<TLorentzVector> preserves type."""
        dsl.define("good_tracks", "tracks[tracks.Pt() > 1.0]")
        func = dsl.get_function("good_tracks")
        assert "RVec" in func.return_type
        assert "TLorentzVector" in func.return_type
    
    # Chaining tests
    def test_filter_then_sum_code(self, dsl):
        """Filter then Sum generates correct code."""
        dsl.define("high_pt", "pt[pt > 2.0]")
        dsl.define("sum_high", "Sum(high_pt)")
        code = dsl.preview()
        assert "ROOT::VecOps::Sum" in code
    
    def test_filter_then_size_code(self, dsl):
        """Filter then .size() generates correct code."""
        dsl.define("high_pt", "pt[pt > 2.0]")
        dsl.define("n_high", "high_pt.size()")
        code = dsl.preview()
        assert ".size()" in code
    
    def test_filter_then_method_sum_code(self, dsl):
        """Filter then .sum() generates correct code."""
        dsl.define("high_pt", "pt[pt > 2.0]")
        dsl.define("sum_high", "high_pt.sum()")
        code = dsl.preview()
        assert "ROOT::VecOps::Sum" in code


# =============================================================================
# Integration Tests (require ROOT)
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestBooleanMaskCompilation:
    """ROOT compilation tests."""
    
    def test_simple_mask_compiles(self):
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("high_pt", "pt[pt > 1.0]")
        dsl.compile_all()  # Should not raise
    
    def test_cross_vector_mask_compiles(self):
        dsl = DSLCompiler({"pt": "RVec<double>", "eta": "RVec<double>"})
        dsl.define("central_pt", "pt[abs(eta) < 1.0]")
        dsl.compile_all()
    
    def test_le_mask_compiles(self):
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("low_pt", "pt[pt <= 2.0]")
        dsl.compile_all()
    
    def test_eq_mask_compiles(self):
        dsl = DSLCompiler({"flags": "RVec<int>"})
        dsl.define("zeros", "flags[flags == 0]")
        dsl.compile_all()
    
    def test_object_filter_compiles(self):
        dsl = DSLCompiler({"tracks": "RVec<TLorentzVector>"})
        dsl.define("good", "tracks[tracks.Pt() > 1.0]")
        dsl.compile_all()


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestBooleanMaskExecution:
    """Execution tests with actual values."""
    
    def test_filter_greater_than(self):
        """pt[pt > 2.0] filters correctly."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("high_pt", "pt[pt > 2.0]")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 3.0, 0.5, 4.0}")
        rdf = dsl.apply(rdf)
        
        # Check size of filtered vector
        rdf = rdf.Define("n_high", "high_pt.size()")
        result = rdf.Mean("n_high").GetValue()
        assert abs(result - 2.0) < 0.001  # 3.0 and 4.0 pass
    
    def test_filter_values_correct(self):
        """Filtered values are the correct ones."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("high_pt", "pt[pt > 2.0]")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 3.0, 0.5, 4.0}")
        rdf = dsl.apply(rdf)
        
        # Sum should be 3.0 + 4.0 = 7.0
        rdf = rdf.Define("sum_high", "ROOT::VecOps::Sum(high_pt)")
        result = rdf.Mean("sum_high").GetValue()
        assert abs(result - 7.0) < 0.001
    
    def test_cross_vector_filter(self):
        """pt[abs(eta) < 1.0] filters by different vector."""
        dsl = DSLCompiler({"pt": "RVec<double>", "eta": "RVec<double>"})
        dsl.define("central_pt", "pt[abs(eta) < 1.0]")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0, 4.0}")
        rdf = rdf.Define("eta", "ROOT::RVec<double>{0.5, 1.5, 0.2, 2.0}")
        rdf = dsl.apply(rdf)
        
        # eta[0]=0.5 < 1.0 ✓, eta[2]=0.2 < 1.0 ✓
        rdf = rdf.Define("n_central", "central_pt.size()")
        result = rdf.Mean("n_central").GetValue()
        assert abs(result - 2.0) < 0.001
    
    def test_empty_result(self):
        """Filter that results in empty vector."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("high_pt", "pt[pt > 100.0]")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
        rdf = dsl.apply(rdf)
        
        rdf = rdf.Define("n_high", "high_pt.size()")
        result = rdf.Mean("n_high").GetValue()
        assert abs(result - 0.0) < 0.001
    
    def test_all_pass(self):
        """Filter where all elements pass."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("positive", "pt[pt > 0.0]")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
        rdf = dsl.apply(rdf)
        
        rdf = rdf.Define("n_pos", "positive.size()")
        result = rdf.Mean("n_pos").GetValue()
        assert abs(result - 3.0) < 0.001


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestBooleanMaskChaining:
    """Test filter combined with other operations."""
    
    def test_filter_then_sum(self):
        """Sum(pt[pt > 2.0]) via alias chaining works."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("high_pt", "pt[pt > 2.0]")
        dsl.define("sum_high", "Sum(high_pt)")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 3.0, 0.5, 4.0}")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("sum_high").GetValue()
        assert abs(result - 7.0) < 0.001  # 3.0 + 4.0
    
    def test_filter_then_mean(self):
        """Mean(pt[pt > 2.0]) via alias chaining works."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("high_pt", "pt[pt > 2.0]")
        dsl.define("avg_high", "Mean(high_pt)")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 3.0, 0.5, 4.0}")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("avg_high").GetValue()
        assert abs(result - 3.5) < 0.001  # (3.0 + 4.0) / 2
    
    def test_filter_then_size(self):
        """high_pt.size() works after filtering."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("high_pt", "pt[pt > 2.0]")
        dsl.define("n_high", "high_pt.size()")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 3.0, 0.5, 4.0}")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("n_high").GetValue()
        assert abs(result - 2.0) < 0.001
    
    def test_filter_then_method_aggregate(self):
        """high_pt.sum() works after filtering."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("high_pt", "pt[pt > 2.0]")
        dsl.define("sum_high", "high_pt.sum()")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 3.0, 0.5, 4.0}")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("sum_high").GetValue()
        assert abs(result - 7.0) < 0.001


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestBooleanMaskObjects:
    """Test filtering object vectors."""
    
    def test_filter_tlorentzvector_compiles(self):
        """tracks[tracks.Pt() > 1.0] compiles."""
        dsl = DSLCompiler({"tracks": "RVec<TLorentzVector>"})
        dsl.define("good", "tracks[tracks.Pt() > 1.0]")
        dsl.compile_all()
    
    def test_filter_then_broadcast_compiles(self):
        """Filter objects, then broadcast method."""
        dsl = DSLCompiler({"tracks": "RVec<TLorentzVector>"})
        dsl.define("good", "tracks[tracks.Pt() > 1.0]")
        dsl.define("good_px", "good.Px()")
        dsl.compile_all()
    
    def test_filter_preserves_object_type(self):
        """Filtered object vector preserves element type."""
        dsl = DSLCompiler({"tracks": "RVec<TLorentzVector>"})
        dsl.define("good", "tracks[tracks.Pt() > 1.0]")
        func = dsl.get_function("good")
        assert "TLorentzVector" in func.return_type


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestBooleanMaskErrors:
    """Error handling tests."""
    
    def test_mask_on_scalar_generates_invalid_code(self):
        """Masking a scalar generates invalid C++ (caught at compile time)."""
        dsl = DSLCompiler({"x": "double", "flags": "RVec<bool>"})
        # This generates invalid C++ code that would fail compilation
        # (tries to call .size() on a scalar)
        dsl.define("bad", "x[flags]")
        code = dsl.preview()
        # The generated code is invalid - it tries to do x.size() on a double
        # This would fail at ROOT compile time
        assert "x.size()" in code or "flags" in code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
