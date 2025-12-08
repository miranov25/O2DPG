"""
test_root_integration.py - ROOT Behavior and DSL Pipeline Integration Tests

This file validates:
1. ROOT C++ behavior we depend on (Category A)
2. DSL code generation correctness (Category B)
3. Full RDataFrame pipeline (Category C)
4. Techniques for future phases (Category D)

Requirements:
- ROOT >= 6.26
- Run after unit tests in CI

Run with:
    pytest tests/test_root_integration.py -v
    pytest tests/test_root_integration.py -v -m future  # Only future tests
    pytest tests/test_root_integration.py -v -m "not future"  # Skip future

These tests require ROOT. Auto-skipped if ROOT unavailable.

Phase 6.9 - Created to validate assumptions before Phase 7 (Slicing).
"""

import pytest
import math
import tempfile
import os

# Skip entire file if ROOT not available
ROOT = pytest.importorskip("ROOT")

# Custom marker for future phase techniques
future = pytest.mark.future

# Document minimum ROOT version
MIN_ROOT_VERSION = (6, 26)


# =============================================================================
# Category A: ROOT C++ Behavior Assumptions
# =============================================================================

class TestROOTCppAssumptions:
    """
    Category A: ROOT C++ Behavior Assumptions
    
    These tests verify ROOT behaves as we expect.
    If any fail, our design assumptions are invalid.
    """
    
    def test_A01_std_sqrt_available(self):
        """std::sqrt works in gInterpreter."""
        ROOT.gInterpreter.Declare('''
            double test_a01_sqrt(double x) { return std::sqrt(x); }
        ''')
        result = ROOT.test_a01_sqrt(4.0)
        assert abs(result - 2.0) < 0.001
    
    def test_A02_std_pow_available(self):
        """std::pow works in gInterpreter."""
        ROOT.gInterpreter.Declare('''
            double test_a02_pow(double x, double y) { return std::pow(x, y); }
        ''')
        result = ROOT.test_a02_pow(2.0, 3.0)
        assert abs(result - 8.0) < 0.001
    
    def test_A03_rvec_arithmetic(self):
        """RVec element-wise arithmetic works."""
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<double> test_a03_mul(const ROOT::RVec<double>& v) {
                return v * 2.0;
            }
        ''')
        v = ROOT.RVec('double')([1.0, 2.0, 3.0])
        result = ROOT.test_a03_mul(v)
        assert list(result) == [2.0, 4.0, 6.0]
    
    def test_A04_rvec_indexing(self):
        """RVec indexing returns correct element."""
        ROOT.gInterpreter.Declare('''
            double test_a04_idx(const ROOT::RVec<double>& v, int i) {
                return v[i];
            }
        ''')
        v = ROOT.RVec('double')([10.0, 20.0, 30.0])
        assert ROOT.test_a04_idx(v, 1) == 20.0
    
    def test_A05_rvec_size(self):
        """RVec.size() returns correct count."""
        ROOT.gInterpreter.Declare('''
            size_t test_a05_size(const ROOT::RVec<double>& v) {
                return v.size();
            }
        ''')
        v = ROOT.RVec('double')([1.0, 2.0, 3.0, 4.0])
        assert ROOT.test_a05_size(v) == 4
    
    def test_A06_adl_finds_vecops_sqrt(self):
        """ADL finds ROOT::VecOps::sqrt for RVec."""
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<double> test_a06_sqrt(const ROOT::RVec<double>& v) {
                return sqrt(v);  // ADL should find ROOT::VecOps::sqrt
            }
        ''')
        v = ROOT.RVec('double')([4.0, 9.0, 16.0])
        result = ROOT.test_a06_sqrt(v)
        assert list(result) == [2.0, 3.0, 4.0]
    
    def test_A07_nan_propagation(self):
        """NaN propagates correctly in comparisons."""
        ROOT.gInterpreter.Declare('''
            bool test_a07_nan_cmp(double x) {
                return x > 5.0;
            }
        ''')
        nan = float('nan')
        # NaN comparisons should return false
        assert ROOT.test_a07_nan_cmp(nan) == False
    
    def test_A08_quiet_nan_generation(self):
        """std::numeric_limits<double>::quiet_NaN() works."""
        ROOT.gInterpreter.Declare('''
            double test_a08_nan() {
                return std::numeric_limits<double>::quiet_NaN();
            }
        ''')
        result = ROOT.test_a08_nan()
        assert math.isnan(result)
    
    def test_A09_tclass_get_data_member(self):
        """TClass::GetDataMember finds protected members."""
        tclass = ROOT.TClass.GetClass("TVector3")
        dm = tclass.GetDataMember("fX")
        assert dm is not None
        assert dm.GetOffset() >= 0
    
    def test_A10_static_in_lambda(self):
        """Static variables in lambda work (C++11 magic statics)."""
        ROOT.gInterpreter.Declare('''
            double test_a10_static_lambda(double x) {
                return [&]() -> double {
                    static double cached = x * 2;
                    return cached;
                }();
            }
        ''')
        # First call caches 10*2=20
        assert ROOT.test_a10_static_lambda(10.0) == 20.0
        # Second call returns cached value (should still be 20)
        assert ROOT.test_a10_static_lambda(5.0) == 20.0
    
    def test_A11_const_rvec_ref_in_rdf(self):
        """const RVec<T>& works as RDataFrame Define parameter."""
        ROOT.gInterpreter.Declare('''
            double test_a11_sum(const ROOT::RVec<double>& v) {
                double s = 0;
                for (auto x : v) s += x;
                return s;
            }
        ''')
        # Just verify it compiles - RDataFrame uses this signature
        v = ROOT.RVec('double')([1.0, 2.0, 3.0])
        assert ROOT.test_a11_sum(v) == 6.0
    
    def test_A12_reinterpret_cast_offset(self):
        """reinterpret_cast with offset reads correct value."""
        ROOT.gInterpreter.Declare('''
            double test_a12_offset_read(const TVector3& v) {
                TClass* cls = TClass::GetClass("TVector3");
                TDataMember* dm = cls->GetDataMember("fX");
                Long_t offset = dm->GetOffset();
                return *reinterpret_cast<const double*>(
                    reinterpret_cast<const char*>(&v) + offset);
            }
        ''')
        v = ROOT.TVector3(1.5, 2.5, 3.5)
        assert abs(ROOT.test_a12_offset_read(v) - 1.5) < 0.001


# =============================================================================
# Category B: DSL Code Generation Correctness
# =============================================================================

class TestCodeGenerationCorrectness:
    """
    Category B: DSL Code Generation Correctness
    
    Tests that generated C++ code compiles and executes correctly.
    """
    
    @pytest.fixture
    def dsl_tools(self):
        """Set up DSL pipeline tools."""
        from RDataFrameDSL import (
            TypeInferrer, IRBuilder, CppCodeGenerator, 
            FunctionLibrary, ReflectionCache
        )
        return {
            'TypeInferrer': TypeInferrer,
            'IRBuilder': IRBuilder,
            'CppCodeGenerator': CppCodeGenerator,
            'FunctionLibrary': FunctionLibrary,
            'ReflectionCache': ReflectionCache,
        }
    
    def _create_pipeline(self, dsl_tools, schema_simple):
        """Helper to create DSL pipeline from simple schema."""
        # Convert simple schema to full format
        columns = {}
        for name, type_str in schema_simple.items():
            if type_str.startswith("RVec<") and type_str.endswith(">"):
                inner = type_str[5:-1]
                columns[name] = {"dtype": inner, "rank": 1}
            else:
                columns[name] = {"dtype": type_str, "rank": 0}
        
        full_schema = {"columns": columns}
        inferrer = dsl_tools['TypeInferrer'].from_schema(full_schema)
        generator = dsl_tools['CppCodeGenerator'](
            type_inferrer=inferrer,
            safe_indexing=True
        )
        library = dsl_tools['FunctionLibrary']()
        builder = dsl_tools['IRBuilder'](inferrer)
        
        return inferrer, builder, generator, library
    
    def test_B01_scalar_sqrt(self, dsl_tools):
        """sqrt(x**2 + y**2) compiles and executes correctly."""
        schema = {"px": "double", "py": "double"}
        _, builder, generator, library = self._create_pipeline(dsl_tools, schema)
        
        ir = builder.build("sqrt(px**2 + py**2)")
        func = generator.generate(ir, "b01_pt")
        library.add(func)
        library.compile(func.name)
        
        # Verify: sqrt(3^2 + 4^2) = 5
        result = getattr(ROOT, func.name)(3.0, 4.0)
        assert abs(result - 5.0) < 0.001
    
    def test_B02_rvec_index_first(self, dsl_tools):
        """pt[0] returns first element."""
        schema = {"pt": "RVec<double>"}
        _, builder, generator, library = self._create_pipeline(dsl_tools, schema)
        
        ir = builder.build("pt[0]")
        func = generator.generate(ir, "b02_first")
        library.add(func)
        library.compile(func.name)
        
        v = ROOT.RVec('double')([1.5, 2.5, 3.5])
        result = getattr(ROOT, func.name)(v)
        assert abs(result - 1.5) < 0.001
    
    def test_B03_rvec_index_negative(self, dsl_tools):
        """pt[-1] returns last element (regression test for Bug 2.2)."""
        schema = {"pt": "RVec<double>"}
        _, builder, generator, library = self._create_pipeline(dsl_tools, schema)
        
        ir = builder.build("pt[-1]")
        func = generator.generate(ir, "b03_last")
        library.add(func)
        library.compile(func.name)
        
        v = ROOT.RVec('double')([1.0, 2.0, 3.0])
        result = getattr(ROOT, func.name)(v)
        assert abs(result - 3.0) < 0.001
    
    def test_B04_rvec_size(self, dsl_tools):
        """pt.size() returns count."""
        schema = {"pt": "RVec<double>"}
        _, builder, generator, library = self._create_pipeline(dsl_tools, schema)
        
        ir = builder.build("pt.size()")
        func = generator.generate(ir, "b04_size")
        library.add(func)
        library.compile(func.name)
        
        v = ROOT.RVec('double')([1.0, 2.0, 3.0, 4.0])
        result = getattr(ROOT, func.name)(v)
        assert result == 4
    
    def test_B05_rvec_arithmetic(self, dsl_tools):
        """pt * 2.0 scales all elements."""
        schema = {"pt": "RVec<double>"}
        _, builder, generator, library = self._create_pipeline(dsl_tools, schema)
        
        ir = builder.build("pt * 2.0")
        func = generator.generate(ir, "b05_scaled")
        library.add(func)
        library.compile(func.name)
        
        v = ROOT.RVec('double')([1.0, 2.0, 3.0])
        result = getattr(ROOT, func.name)(v)
        assert list(result) == [2.0, 4.0, 6.0]
    
    def test_B06_rvec_empty_index_returns_nan(self, dsl_tools):
        """Indexing empty vector returns NaN."""
        schema = {"pt": "RVec<double>"}
        _, builder, generator, library = self._create_pipeline(dsl_tools, schema)
        
        ir = builder.build("pt[0]")
        func = generator.generate(ir, "b06_empty")
        library.add(func)
        library.compile(func.name)
        
        v = ROOT.RVec('double')()  # Empty
        result = getattr(ROOT, func.name)(v)
        assert math.isnan(result)
    
    def test_B07_rvec_oob_returns_nan(self, dsl_tools):
        """Out-of-bounds index returns NaN."""
        schema = {"pt": "RVec<double>"}
        _, builder, generator, library = self._create_pipeline(dsl_tools, schema)
        
        ir = builder.build("pt[999]")
        func = generator.generate(ir, "b07_oob")
        library.add(func)
        library.compile(func.name)
        
        v = ROOT.RVec('double')([1.0, 2.0, 3.0])
        result = getattr(ROOT, func.name)(v)
        assert math.isnan(result)
    
    def test_B08_rvec_add_rvec(self, dsl_tools):
        """a + b adds element-wise."""
        schema = {"a": "RVec<double>", "b": "RVec<double>"}
        _, builder, generator, library = self._create_pipeline(dsl_tools, schema)
        
        ir = builder.build("a + b")
        func = generator.generate(ir, "b08_add")
        library.add(func)
        library.compile(func.name)
        
        va = ROOT.RVec('double')([1.0, 2.0, 3.0])
        vb = ROOT.RVec('double')([10.0, 20.0, 30.0])
        result = getattr(ROOT, func.name)(va, vb)
        assert list(result) == [11.0, 22.0, 33.0]
    
    def test_B09_no_double_std_namespace(self, dsl_tools):
        """Generated code has no std::std:: double namespace (regression test for Bug 2.1)."""
        schema = {"x": "double"}
        _, builder, generator, library = self._create_pipeline(dsl_tools, schema)
        
        ir = builder.build("sqrt(x)")
        func = generator.generate(ir, "b09_sqrt")
        
        # Check generated code
        assert "std::std::" not in func.code
        assert "std::sqrt" in func.code or "sqrt" in func.code
    
    def test_B10_negative_index_not_always_nan(self, dsl_tools):
        """pt[-1] should NOT always return NaN (regression test for Bug 2.2)."""
        schema = {"pt": "RVec<double>"}
        _, builder, generator, library = self._create_pipeline(dsl_tools, schema)
        
        ir = builder.build("pt[-1]")
        func = generator.generate(ir, "b10_last")
        
        # Verify generated code doesn't have the buggy pattern
        assert "(-1) >= 0" not in func.code  # This would always be false
        
        library.add(func)
        library.compile(func.name)
        
        v = ROOT.RVec('double')([1.0, 2.0, 3.0])
        result = getattr(ROOT, func.name)(v)
        assert abs(result - 3.0) < 0.001  # Should be 3.0, not NaN


# =============================================================================
# Category C: RDataFrame Integration
# =============================================================================

class TestRDataFrameIntegration:
    """
    Category C: Full RDataFrame Pipeline Integration
    
    End-to-end tests with actual TTree and RDataFrame.
    """
    
    @pytest.fixture
    def test_tree(self, tmp_path):
        """Create test TTree with known values."""
        filename = str(tmp_path / "test_rdf.root")
        
        ROOT.gInterpreter.ProcessLine(f'''
            void create_test_tree_c(const char* fname) {{
                TFile f(fname, "RECREATE");
                TTree tree("Events", "Test");
                
                double px, py, pz;
                std::vector<double> track_pt;
                
                tree.Branch("px", &px);
                tree.Branch("py", &py);
                tree.Branch("pz", &pz);
                tree.Branch("track_pt", &track_pt);
                
                // Event 1: pt=5, 4 tracks
                px = 3.0; py = 4.0; pz = 0.0;
                track_pt = {{1.5, 2.5, 3.5, 4.5}};
                tree.Fill();
                
                // Event 2: pt=1, 2 tracks
                px = 1.0; py = 0.0; pz = 0.0;
                track_pt = {{10.0, 20.0}};
                tree.Fill();
                
                // Event 3: pt=5, p=13, 1 track
                px = 0.0; py = 5.0; pz = 12.0;
                track_pt = {{0.5}};
                tree.Fill();
                
                tree.Write();
                f.Close();
            }}
        ''')
        ROOT.create_test_tree_c(filename)
        return filename
    
    @pytest.fixture
    def dsl_compiler(self):
        """Create DSL compiler helper."""
        from RDataFrameDSL import (
            TypeInferrer, IRBuilder, CppCodeGenerator, FunctionLibrary
        )
        
        class SimpleCompiler:
            def __init__(self, schema):
                columns = {}
                for name, type_str in schema.items():
                    if type_str.startswith("RVec<"):
                        inner = type_str[5:-1]
                        columns[name] = {"dtype": inner, "rank": 1}
                    else:
                        columns[name] = {"dtype": type_str, "rank": 0}
                
                self.inferrer = TypeInferrer.from_schema({"columns": columns})
                self.generator = CppCodeGenerator(
                    type_inferrer=self.inferrer,
                    safe_indexing=True
                )
                self.library = FunctionLibrary()
                self.funcs = {}
            
            def define(self, name, expr):
                builder = IRBuilder(self.inferrer)
                ir = builder.build(expr)
                func = self.generator.generate(ir, name)
                self.library.add(func)
                self.funcs[name] = func
                return self
            
            def compile_all(self):
                for name, func in self.funcs.items():
                    self.library.compile(func.name)  # Use func.name (e.g., 'alias_pt'), not user name
            
            def apply(self, rdf):
                self.compile_all()
                for name, func in self.funcs.items():
                    rdf = rdf.Define(name, func.get_call_expression())
                return rdf
        
        return SimpleCompiler
    
    def test_C01_scalar_define(self, test_tree, dsl_compiler):
        """Scalar expression works in RDataFrame.Define()."""
        schema = {"px": "double", "py": "double"}
        compiler = dsl_compiler(schema)
        compiler.define("c01_pt", "sqrt(px**2 + py**2)")
        
        rdf = ROOT.RDataFrame("Events", test_tree)
        rdf = compiler.apply(rdf)
        
        results = rdf.AsNumpy(["c01_pt"])
        assert abs(results["c01_pt"][0] - 5.0) < 0.001  # sqrt(9+16)
        assert abs(results["c01_pt"][1] - 1.0) < 0.001  # sqrt(1+0)
    
    def test_C02_rvec_size_define(self, test_tree, dsl_compiler):
        """RVec.size() works in RDataFrame.Define()."""
        schema = {"track_pt": "RVec<double>"}
        compiler = dsl_compiler(schema)
        compiler.define("c02_n_tracks", "track_pt.size()")
        
        rdf = ROOT.RDataFrame("Events", test_tree)
        rdf = compiler.apply(rdf)
        
        results = rdf.AsNumpy(["c02_n_tracks"])
        assert results["c02_n_tracks"][0] == 4
        assert results["c02_n_tracks"][1] == 2
        assert results["c02_n_tracks"][2] == 1
    
    def test_C03_rvec_index_define(self, test_tree, dsl_compiler):
        """RVec indexing works in RDataFrame.Define()."""
        schema = {"track_pt": "RVec<double>"}
        compiler = dsl_compiler(schema)
        compiler.define("c03_lead_pt", "track_pt[0]")
        
        rdf = ROOT.RDataFrame("Events", test_tree)
        rdf = compiler.apply(rdf)
        
        results = rdf.AsNumpy(["c03_lead_pt"])
        assert abs(results["c03_lead_pt"][0] - 1.5) < 0.001
        assert abs(results["c03_lead_pt"][1] - 10.0) < 0.001
    
    def test_C04_multiple_defines(self, test_tree, dsl_compiler):
        """Multiple Define() calls work correctly."""
        schema = {
            "px": "double", 
            "py": "double",
            "track_pt": "RVec<double>"
        }
        compiler = dsl_compiler(schema)
        compiler.define("c04_pt", "sqrt(px**2 + py**2)")
        compiler.define("c04_n_tracks", "track_pt.size()")
        compiler.define("c04_lead_pt", "track_pt[0]")
        compiler.define("c04_scaled_pt", "track_pt * 2.0")
        
        rdf = ROOT.RDataFrame("Events", test_tree)
        rdf = compiler.apply(rdf)
        
        results = rdf.AsNumpy(["c04_pt", "c04_n_tracks", "c04_lead_pt"])
        assert abs(results["c04_pt"][0] - 5.0) < 0.001
        assert results["c04_n_tracks"][0] == 4
        assert abs(results["c04_lead_pt"][0] - 1.5) < 0.001
    
    def test_C05_negative_index_in_rdf(self, test_tree, dsl_compiler):
        """Negative index works in RDataFrame (regression test)."""
        schema = {"track_pt": "RVec<double>"}
        compiler = dsl_compiler(schema)
        compiler.define("c05_last_pt", "track_pt[-1]")
        
        rdf = ROOT.RDataFrame("Events", test_tree)
        rdf = compiler.apply(rdf)
        
        results = rdf.AsNumpy(["c05_last_pt"])
        assert abs(results["c05_last_pt"][0] - 4.5) < 0.001  # Last of [1.5, 2.5, 3.5, 4.5]
        assert abs(results["c05_last_pt"][1] - 20.0) < 0.001  # Last of [10.0, 20.0]
        assert abs(results["c05_last_pt"][2] - 0.5) < 0.001  # Last of [0.5]


# =============================================================================
# Category D: Future Phase Techniques
# =============================================================================

@future
class TestFuturePhaseTechniques:
    """
    Category D: Future Phase Techniques
    
    Tests that ROOT supports techniques we plan to use.
    These are marked @future and validate Phase 7+ approaches.
    """
    
    def test_D01_vecops_take_indices(self):
        """VecOps::Take with index list works (Phase 7 slicing)."""
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<double> test_d01_take(const ROOT::RVec<double>& v) {
                return ROOT::VecOps::Take(v, {1, 2});
            }
        ''')
        v = ROOT.RVec('double')([10.0, 20.0, 30.0, 40.0])
        result = ROOT.test_d01_take(v)
        assert list(result) == [20.0, 30.0]
    
    def test_D02_vecops_take_negative(self):
        """VecOps::Take with negative n gets last n elements."""
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<double> test_d02_take_neg(const ROOT::RVec<double>& v) {
                return ROOT::VecOps::Take(v, -2);
            }
        ''')
        v = ROOT.RVec('double')([10.0, 20.0, 30.0, 40.0])
        result = ROOT.test_d02_take_neg(v)
        assert list(result) == [30.0, 40.0]
    
    def test_D03_boolean_indexing(self):
        """Boolean indexing v[mask] works (Phase 7)."""
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<double> test_d03_bool(const ROOT::RVec<double>& v) {
                return v[v > 2.0];
            }
        ''')
        v = ROOT.RVec('double')([1.0, 2.0, 3.0, 4.0])
        result = ROOT.test_d03_bool(v)
        assert list(result) == [3.0, 4.0]
    
    def test_D04_vecops_filter(self):
        """VecOps::Filter with lambda predicate works (Phase 7 alternative)."""
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<double> test_d04_filter(const ROOT::RVec<double>& v) {
                return ROOT::VecOps::Filter(v, [](double x) { return x > 2.0; });
            }
        ''')
        v = ROOT.RVec('double')([1.0, 2.0, 3.0, 4.0])
        result = ROOT.test_d04_filter(v)
        assert list(result) == [3.0, 4.0]
    
    def test_D05_vecops_map(self):
        """VecOps::Map works (Phase 8 method broadcasting)."""
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<double> test_d05_map(const ROOT::RVec<TVector3>& vecs) {
                return ROOT::VecOps::Map(vecs, [](const TVector3& v) { 
                    return v.Mag(); 
                });
            }
        ''')
        vecs = ROOT.RVec('TVector3')()
        vecs.push_back(ROOT.TVector3(3.0, 4.0, 0.0))
        vecs.push_back(ROOT.TVector3(0.0, 0.0, 5.0))
        result = ROOT.test_d05_map(vecs)
        assert abs(result[0] - 5.0) < 0.001
        assert abs(result[1] - 5.0) < 0.001
    
    def test_D06_vecops_range(self):
        """VecOps::Range generates index sequence."""
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<int> test_d06_range() {
                return ROOT::VecOps::Range(1, 5);
            }
        ''')
        result = ROOT.test_d06_range()
        assert list(result) == [1, 2, 3, 4]
    
    def test_D07_step_indexing_loop(self):
        """Loop-based step indexing works (Phase 7 pt[::2])."""
        ROOT.gInterpreter.Declare('''
            ROOT::RVec<double> test_d07_step(const ROOT::RVec<double>& v) {
                ROOT::RVec<double> result;
                for (size_t i = 0; i < v.size(); i += 2) {
                    result.push_back(v[i]);
                }
                return result;
            }
        ''')
        v = ROOT.RVec('double')([1.0, 2.0, 3.0, 4.0, 5.0])
        result = ROOT.test_d07_step(v)
        assert list(result) == [1.0, 3.0, 5.0]
