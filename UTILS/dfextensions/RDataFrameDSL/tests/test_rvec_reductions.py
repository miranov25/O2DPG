"""
Phase 10.5: RVec Reduction tests.

Tests for reduction functions that operate on RVec and return scalars:
- Function style: Sum(pt), Mean(pt), Max(pt), Min(pt), Any(flags), All(flags), StdDev(pt), Var(pt)
- Method style: pt.sum(), pt.mean(), pt.max(), pt.min(), flags.any(), flags.all(), pt.std(), pt.var()
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

class TestReductionMock:
    """Mock tests that don't require ROOT compilation."""
    
    @pytest.fixture
    def dsl(self):
        return DSLCompiler({"pt": "RVec<double>", "flags": "RVec<bool>"})
    
    # Function-style code generation
    def test_sum_generates_vecops(self, dsl):
        dsl.define("total", "Sum(pt)")
        assert "ROOT::VecOps::Sum" in dsl.preview()
    
    def test_mean_generates_vecops(self, dsl):
        dsl.define("avg", "Mean(pt)")
        assert "ROOT::VecOps::Mean" in dsl.preview()
    
    def test_max_generates_vecops(self, dsl):
        dsl.define("highest", "Max(pt)")
        assert "ROOT::VecOps::Max" in dsl.preview()
    
    def test_min_generates_vecops(self, dsl):
        dsl.define("lowest", "Min(pt)")
        assert "ROOT::VecOps::Min" in dsl.preview()
    
    def test_any_generates_vecops(self, dsl):
        dsl.define("has_true", "Any(flags)")
        assert "ROOT::VecOps::Any" in dsl.preview()
    
    def test_all_generates_vecops(self, dsl):
        dsl.define("all_true", "All(flags)")
        assert "ROOT::VecOps::All" in dsl.preview()
    
    def test_stddev_generates_vecops(self, dsl):
        dsl.define("sigma", "StdDev(pt)")
        assert "ROOT::VecOps::StdDev" in dsl.preview()
    
    def test_var_generates_vecops(self, dsl):
        dsl.define("variance", "Var(pt)")
        assert "ROOT::VecOps::Var" in dsl.preview()
    
    # Method-style code generation
    def test_sum_method(self, dsl):
        dsl.define("total", "pt.sum()")
        assert "ROOT::VecOps::Sum" in dsl.preview()
    
    def test_mean_method(self, dsl):
        dsl.define("avg", "pt.mean()")
        assert "ROOT::VecOps::Mean" in dsl.preview()
    
    def test_max_method(self, dsl):
        dsl.define("highest", "pt.max()")
        assert "ROOT::VecOps::Max" in dsl.preview()
    
    def test_min_method(self, dsl):
        dsl.define("lowest", "pt.min()")
        assert "ROOT::VecOps::Min" in dsl.preview()
    
    def test_any_method(self, dsl):
        dsl.define("has_true", "flags.any()")
        assert "ROOT::VecOps::Any" in dsl.preview()
    
    def test_all_method(self, dsl):
        dsl.define("all_true", "flags.all()")
        assert "ROOT::VecOps::All" in dsl.preview()
    
    def test_std_method(self, dsl):
        dsl.define("sigma", "pt.std()")
        assert "ROOT::VecOps::StdDev" in dsl.preview()
    
    def test_var_method(self, dsl):
        dsl.define("variance", "pt.var()")
        assert "ROOT::VecOps::Var" in dsl.preview()
    
    # Return type tests (check that RVec is NOT in return type)
    def test_sum_returns_scalar(self, dsl):
        dsl.define("total", "Sum(pt)")
        func = dsl.get_function("total")
        assert "RVec" not in func.return_type
    
    def test_sum_method_returns_scalar(self, dsl):
        dsl.define("total", "pt.sum()")
        func = dsl.get_function("total")
        assert "RVec" not in func.return_type
    
    def test_mean_returns_double(self, dsl):
        dsl.define("avg", "Mean(pt)")
        func = dsl.get_function("avg")
        assert "double" in func.return_type.lower()
    
    def test_any_returns_bool(self, dsl):
        dsl.define("has", "Any(flags)")
        func = dsl.get_function("has")
        assert "bool" in func.return_type.lower()
    
    def test_all_returns_bool(self, dsl):
        dsl.define("every", "All(flags)")
        func = dsl.get_function("every")
        assert "bool" in func.return_type.lower()


# =============================================================================
# Integration Tests (require ROOT)
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestReductionCodeGeneration:
    """Code generation tests for function-style reductions."""
    
    @pytest.fixture
    def dsl(self):
        return DSLCompiler({"pt": "RVec<double>", "flags": "RVec<bool>"})
    
    def test_sum_generates_vecops(self, dsl):
        dsl.define("total", "Sum(pt)")
        assert "ROOT::VecOps::Sum" in dsl.preview()
    
    def test_mean_generates_vecops(self, dsl):
        dsl.define("avg", "Mean(pt)")
        assert "ROOT::VecOps::Mean" in dsl.preview()
    
    def test_max_generates_vecops(self, dsl):
        dsl.define("highest", "Max(pt)")
        assert "ROOT::VecOps::Max" in dsl.preview()
    
    def test_min_generates_vecops(self, dsl):
        dsl.define("lowest", "Min(pt)")
        assert "ROOT::VecOps::Min" in dsl.preview()
    
    def test_any_generates_vecops(self, dsl):
        dsl.define("has_true", "Any(flags)")
        assert "ROOT::VecOps::Any" in dsl.preview()
    
    def test_all_generates_vecops(self, dsl):
        dsl.define("all_true", "All(flags)")
        assert "ROOT::VecOps::All" in dsl.preview()
    
    def test_stddev_generates_vecops(self, dsl):
        dsl.define("sigma", "StdDev(pt)")
        assert "ROOT::VecOps::StdDev" in dsl.preview()
    
    def test_var_generates_vecops(self, dsl):
        dsl.define("variance", "Var(pt)")
        assert "ROOT::VecOps::Var" in dsl.preview()


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestReductionReturnTypes:
    """Return type inference tests."""
    
    @pytest.fixture
    def dsl(self):
        return DSLCompiler({
            "pt": "RVec<double>",
            "counts": "RVec<int>",
            "flags": "RVec<bool>"
        })
    
    def test_sum_returns_scalar(self, dsl):
        """Sum returns scalar, not RVec."""
        dsl.define("total", "Sum(pt)")
        func = dsl.get_function("total")
        assert "RVec" not in func.return_type
    
    def test_sum_returns_element_type_double(self, dsl):
        dsl.define("total", "Sum(pt)")
        func = dsl.get_function("total")
        assert "double" in func.return_type.lower()
    
    def test_mean_returns_double(self, dsl):
        dsl.define("avg", "Mean(pt)")
        func = dsl.get_function("avg")
        assert "double" in func.return_type.lower()
    
    def test_max_returns_element_type(self, dsl):
        dsl.define("highest", "Max(pt)")
        func = dsl.get_function("highest")
        assert "double" in func.return_type.lower()
    
    def test_any_returns_bool(self, dsl):
        dsl.define("has", "Any(flags)")
        func = dsl.get_function("has")
        assert "bool" in func.return_type.lower()
    
    def test_all_returns_bool(self, dsl):
        dsl.define("every", "All(flags)")
        func = dsl.get_function("every")
        assert "bool" in func.return_type.lower()
    
    def test_stddev_returns_double(self, dsl):
        dsl.define("sigma", "StdDev(pt)")
        func = dsl.get_function("sigma")
        assert "double" in func.return_type.lower()
    
    def test_var_returns_double(self, dsl):
        dsl.define("variance", "Var(pt)")
        func = dsl.get_function("variance")
        assert "double" in func.return_type.lower()


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestReductionCompilation:
    """ROOT compilation tests."""
    
    @pytest.fixture
    def dsl(self):
        return DSLCompiler({"pt": "RVec<double>", "flags": "RVec<bool>"})
    
    def test_sum_compiles(self, dsl):
        dsl.define("total", "Sum(pt)")
        dsl.compile_all()  # Should not raise
    
    def test_mean_compiles(self, dsl):
        dsl.define("avg", "Mean(pt)")
        dsl.compile_all()
    
    def test_max_compiles(self, dsl):
        dsl.define("highest", "Max(pt)")
        dsl.compile_all()
    
    def test_min_compiles(self, dsl):
        dsl.define("lowest", "Min(pt)")
        dsl.compile_all()
    
    def test_any_compiles(self, dsl):
        dsl.define("has", "Any(flags)")
        dsl.compile_all()
    
    def test_all_compiles(self, dsl):
        dsl.define("every", "All(flags)")
        dsl.compile_all()
    
    def test_stddev_compiles(self, dsl):
        dsl.define("sigma", "StdDev(pt)")
        dsl.compile_all()
    
    def test_var_compiles(self, dsl):
        dsl.define("variance", "Var(pt)")
        dsl.compile_all()


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestReductionExecution:
    """ROOT execution tests with actual values."""
    
    def test_sum_correct_value(self):
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("total", "Sum(pt)")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("total").GetValue()
        assert abs(result - 6.0) < 0.001
    
    def test_mean_correct_value(self):
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("avg", "Mean(pt)")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("avg").GetValue()
        assert abs(result - 2.0) < 0.001
    
    def test_max_correct_value(self):
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("highest", "Max(pt)")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 5.0, 3.0}")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("highest").GetValue()
        assert abs(result - 5.0) < 0.001
    
    def test_min_correct_value(self):
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("lowest", "Min(pt)")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 5.0, 3.0}")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("lowest").GetValue()
        assert abs(result - 1.0) < 0.001
    
    def test_sum_bool_counts_true(self):
        """Sum of bool RVec counts true values."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("n_high", "Sum(pt > 2.0)")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 3.0, 4.0, 0.5}")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("n_high").GetValue()
        assert abs(result - 2.0) < 0.001  # 3.0 and 4.0 are > 2.0


# =============================================================================
# Method-Style Tests
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestMethodStyleCodeGeneration:
    """Code generation tests for method-style reductions."""
    
    @pytest.fixture
    def dsl(self):
        return DSLCompiler({"pt": "RVec<double>", "flags": "RVec<bool>"})
    
    def test_sum_method(self, dsl):
        dsl.define("total", "pt.sum()")
        assert "ROOT::VecOps::Sum" in dsl.preview()
    
    def test_mean_method(self, dsl):
        dsl.define("avg", "pt.mean()")
        assert "ROOT::VecOps::Mean" in dsl.preview()
    
    def test_max_method(self, dsl):
        dsl.define("highest", "pt.max()")
        assert "ROOT::VecOps::Max" in dsl.preview()
    
    def test_min_method(self, dsl):
        dsl.define("lowest", "pt.min()")
        assert "ROOT::VecOps::Min" in dsl.preview()
    
    def test_any_method(self, dsl):
        dsl.define("has_true", "flags.any()")
        assert "ROOT::VecOps::Any" in dsl.preview()
    
    def test_all_method(self, dsl):
        dsl.define("all_true", "flags.all()")
        assert "ROOT::VecOps::All" in dsl.preview()
    
    def test_std_method(self, dsl):
        dsl.define("sigma", "pt.std()")
        assert "ROOT::VecOps::StdDev" in dsl.preview()
    
    def test_var_method(self, dsl):
        dsl.define("variance", "pt.var()")
        assert "ROOT::VecOps::Var" in dsl.preview()


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestMethodStyleReturnTypes:
    """Return type tests for method-style reductions."""
    
    @pytest.fixture
    def dsl(self):
        return DSLCompiler({"pt": "RVec<double>", "flags": "RVec<bool>"})
    
    def test_sum_method_returns_scalar(self, dsl):
        dsl.define("total", "pt.sum()")
        func = dsl.get_function("total")
        assert "double" in func.return_type.lower()
        # Should NOT be RVec
        assert "RVec" not in func.return_type
    
    def test_mean_method_returns_double(self, dsl):
        dsl.define("avg", "pt.mean()")
        func = dsl.get_function("avg")
        assert "double" in func.return_type.lower()
    
    def test_any_method_returns_bool(self, dsl):
        dsl.define("has", "flags.any()")
        func = dsl.get_function("has")
        assert "bool" in func.return_type.lower()


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestMethodStyleCompilation:
    """Compilation tests for method-style reductions."""
    
    @pytest.fixture
    def dsl(self):
        return DSLCompiler({"pt": "RVec<double>", "flags": "RVec<bool>"})
    
    def test_sum_method_compiles(self, dsl):
        dsl.define("total", "pt.sum()")
        dsl.compile_all()
    
    def test_mean_method_compiles(self, dsl):
        dsl.define("avg", "pt.mean()")
        dsl.compile_all()
    
    def test_any_method_compiles(self, dsl):
        dsl.define("has", "flags.any()")
        dsl.compile_all()


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestMethodStyleExecution:
    """Execution tests for method-style reductions."""
    
    def test_sum_method_execution(self):
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("total", "pt.sum()")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("total").GetValue()
        assert abs(result - 6.0) < 0.001
    
    def test_mean_method_execution(self):
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("avg", "pt.mean()")
        dsl.compile_all()
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
        rdf = dsl.apply(rdf)
        
        result = rdf.Mean("avg").GetValue()
        assert abs(result - 2.0) < 0.001


# =============================================================================
# Combined Usage Tests
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestReductionWithBroadcast:
    """Test reductions combined with method broadcasting."""
    
    def test_sum_of_alias(self):
        """Sum(alias) where alias is RVec."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("doubled", "pt * 2")  # RVec<double>
        dsl.define("total", "Sum(doubled)")
        
        assert "ROOT::VecOps::Sum" in dsl.preview()
    
    def test_mean_of_alias(self):
        """Mean(alias) where alias is RVec."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("doubled", "pt * 2")
        dsl.define("avg", "Mean(doubled)")
        
        assert "ROOT::VecOps::Mean" in dsl.preview()
    
    def test_method_on_alias(self):
        """alias.sum() where alias is RVec."""
        dsl = DSLCompiler({"pt": "RVec<double>"})
        dsl.define("doubled", "pt * 2")
        dsl.define("total", "doubled.sum()")
        
        assert "ROOT::VecOps::Sum" in dsl.preview()


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestReductionEquivalence:
    """Test that function and method styles are equivalent."""
    
    def test_sum_equivalence(self):
        """Sum(pt) and pt.sum() produce same result."""
        dsl1 = DSLCompiler({"pt": "RVec<double>"})
        dsl1.define("result", "Sum(pt)")
        
        dsl2 = DSLCompiler({"pt": "RVec<double>"})
        dsl2.define("result", "pt.sum()")
        
        dsl1.compile_all()
        dsl2.compile_all()
        
        rdf1 = ROOT.RDataFrame(1).Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
        rdf2 = ROOT.RDataFrame(1).Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
        
        rdf1 = dsl1.apply(rdf1)
        rdf2 = dsl2.apply(rdf2)
        
        r1 = rdf1.Mean("result").GetValue()
        r2 = rdf2.Mean("result").GetValue()
        
        assert abs(r1 - r2) < 0.001
    
    def test_mean_equivalence(self):
        """Mean(pt) and pt.mean() produce same result."""
        dsl1 = DSLCompiler({"pt": "RVec<double>"})
        dsl1.define("result", "Mean(pt)")
        
        dsl2 = DSLCompiler({"pt": "RVec<double>"})
        dsl2.define("result", "pt.mean()")
        
        dsl1.compile_all()
        dsl2.compile_all()
        
        rdf1 = ROOT.RDataFrame(1).Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
        rdf2 = ROOT.RDataFrame(1).Define("pt", "ROOT::RVec<double>{1.0, 2.0, 3.0}")
        
        rdf1 = dsl1.apply(rdf1)
        rdf2 = dsl2.apply(rdf2)
        
        r1 = rdf1.Mean("result").GetValue()
        r2 = rdf2.Mean("result").GetValue()
        
        assert abs(r1 - r2) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
