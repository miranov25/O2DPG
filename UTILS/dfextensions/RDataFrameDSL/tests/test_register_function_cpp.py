#!/usr/bin/env python3
"""
Unit Tests for register_function_cpp() — Phase 13.5.B

Tests the C++ function registration API added to DSLCompiler.

Author: Claude Opus 4.5 (Architecture Assistant)
Date: 2026-01-11
"""

import pytest
import threading
import re


class TestRegisterFunctionCppBasic:
    """Basic registration tests."""
    
    @pytest.mark.feature("api_register_function_cpp")
    def test_simple_function_registration(self):
        """Test registering a simple C++ function."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp('''
            double square(double x) {
                return x * x;
            }
        ''')
        
        func = dsl.get_registered_function("square")
        assert func is not None
        assert func.name == "square"
        assert func.return_type == "double"
        assert len(func.params) == 1
        assert func.params[0] == ("x", "double")
    
    @pytest.mark.feature("api_register_function_cpp")
    def test_function_with_multiple_params(self):
        """Test function with multiple parameters."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"px": "double", "py": "double"})
        dsl.register_function_cpp('''
            double pt(double px, double py) {
                return sqrt(px*px + py*py);
            }
        ''')
        
        func = dsl.get_registered_function("pt")
        assert func is not None
        assert len(func.params) == 2
        assert func.params[0] == ("px", "double")
        assert func.params[1] == ("py", "double")
    
    def test_function_no_params(self):
        """Test function with no parameters."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp('''
            double pi() {
                return 3.14159265359;
            }
        ''')
        
        func = dsl.get_registered_function("pi")
        assert func is not None
        assert len(func.params) == 0
    
    def test_method_chaining(self):
        """Test that register_function_cpp returns self for chaining."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        result = dsl.register_function_cpp("double f(double x) { return x; }")
        
        assert result is dsl
    
    def test_multiple_registrations(self):
        """Test registering multiple functions."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp("double f1(double x) { return x * 2; }")
        dsl.register_function_cpp("double f2(double x) { return x * 3; }")
        dsl.register_function_cpp("double f3(double x) { return x * 4; }")
        
        names = dsl.list_registered_functions()
        assert "f1" in names
        assert "f2" in names
        assert "f3" in names
        assert len(names) == 3


class TestRegisterFunctionCppNaming:
    """Test naming convention: dsl_{name}_{hash16}."""
    
    def test_cpp_name_format(self):
        """Test C++ name follows dsl_{name}_{hash16} format."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp("double pt(double x) { return x; }")
        
        func = dsl.get_registered_function("pt")
        assert func.cpp_name.startswith("dsl_pt_")
        # Hash is 16 chars
        hash_part = func.cpp_name[len("dsl_pt_"):]
        assert len(hash_part) == 16
        assert all(c in "0123456789abcdef" for c in hash_part)
    
    def test_hash_deterministic(self):
        """Test hash is deterministic for same content."""
        from RDataFrameDSL import DSLCompiler
        
        code = "double f(double x) { return x * 2; }"
        
        dsl1 = DSLCompiler({"x": "double"})
        dsl1.register_function_cpp(code)
        
        dsl2 = DSLCompiler({"x": "double"})
        dsl2.register_function_cpp(code)
        
        assert dsl1.get_registered_function("f").hash == dsl2.get_registered_function("f").hash
    
    def test_hash_excludes_function_name(self):
        """Test hash excludes function name (allows renaming)."""
        from RDataFrameDSL import DSLCompiler
        
        dsl1 = DSLCompiler({"x": "double"})
        dsl1.register_function_cpp("double foo(double x) { return x * 2; }")
        
        dsl2 = DSLCompiler({"x": "double"})
        dsl2.register_function_cpp("double bar(double x) { return x * 2; }")
        
        # Same body, different name → same hash
        assert dsl1.get_registered_function("foo").hash == dsl2.get_registered_function("bar").hash
    
    def test_hash_param_order_preserved_p0_1(self):
        """
        P0-1 FIX: Parameter order must affect hash.
        f(int, double) != f(double, int)
        """
        from RDataFrameDSL import DSLCompiler
        
        dsl1 = DSLCompiler({"a": "int", "b": "double"})
        dsl1.register_function_cpp("double f1(int a, double b) { return a + b; }")
        
        dsl2 = DSLCompiler({"a": "double", "b": "int"})
        dsl2.register_function_cpp("double f2(double a, int b) { return a + b; }")
        
        # Different param order → different hash
        assert dsl1.get_registered_function("f1").hash != dsl2.get_registered_function("f2").hash
    
    def test_hash_different_body(self):
        """Test different body → different hash."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp("double f(double x) { return x * 2; }")
        dsl.register_function_cpp("double f(double x) { return x * 3; }")
        
        funcs = dsl.get_all_registered_functions("f")
        assert len(funcs) == 2
        assert funcs[0].hash != funcs[1].hash


class TestRegisterFunctionCppLambdaRejection:
    """Test FROZEN RULE #1: No lambda expressions."""
    
    @pytest.mark.feature("error_lambda_rejected")
    def test_simple_lambda_rejected(self):
        """Test rejection of simple lambda."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        with pytest.raises(ValueError, match="Lambda|lambda|FROZEN RULE"):
            dsl.register_function_cpp("[](double x) { return x * 2; }")
    
    @pytest.mark.feature("error_lambda_rejected")
    def test_capture_lambda_rejected(self):
        """Test rejection of lambda with capture."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        with pytest.raises(ValueError, match="Lambda|lambda|FROZEN RULE"):
            dsl.register_function_cpp("[&](double x) { return x * 2; }")
    
    @pytest.mark.feature("error_lambda_rejected")
    def test_auto_lambda_rejected(self):
        """Test rejection of auto lambda assignment."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        with pytest.raises(ValueError, match="Lambda|lambda|FROZEN RULE"):
            dsl.register_function_cpp("auto f = [](double x) { return x; };")
    
    def test_named_function_accepted(self):
        """Test that named functions are accepted."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        # Should NOT raise
        dsl.register_function_cpp("double my_func(double x) { return x * 2; }")
        
        func = dsl.get_registered_function("my_func")
        assert func is not None


class TestRegisterFunctionCppHeaders:
    """Test header handling."""
    
    def test_default_headers_included(self):
        """Test default headers are always included."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp("double f(double x) { return x; }")
        
        func = dsl.get_registered_function("f")
        assert "<cmath>" in func.headers
        assert "<ROOT/RVec.hxx>" in func.headers
    
    def test_cmath_auto_detected(self):
        """Test cmath functions trigger header detection."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp("double f(double x) { return sqrt(x); }")
        
        func = dsl.get_registered_function("f")
        assert "<cmath>" in func.headers
    
    def test_user_headers_added(self):
        """Test user-provided headers are added."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp(
            "double f(double x) { return x; }",
            headers=["<custom.h>"]
        )
        
        func = dsl.get_registered_function("f")
        assert "<custom.h>" in func.headers
        # Defaults still present
        assert "<cmath>" in func.headers


class TestRegisterFunctionCppPragmas:
    """Test pragma handling (P0-4: raw pragma lines)."""
    
    def test_raw_pragma_stored(self):
        """Test raw pragma lines are stored."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp(
            "double f(double x) { return x; }",
            pragmas=["#pragma link C++ class MyStruct+;"]
        )
        
        func = dsl.get_registered_function("f")
        assert "#pragma link C++ class MyStruct+;" in func.pragmas
    
    def test_pragma_in_full_cpp(self):
        """Test pragmas appear in generated C++ code."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp(
            "double f(double x) { return x; }",
            pragmas=["#pragma link C++ class MyStruct+;"]
        )
        
        func = dsl.get_registered_function("f")
        assert "#pragma link C++ class MyStruct+;" in func.full_cpp


class TestRegisterFunctionCppComplexTypes:
    """Test complex C++ type handling."""
    
    def test_rvec_parameter(self):
        """Test RVec parameter type."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"v": "RVec<double>"})
        dsl.register_function_cpp('''
            double sum_vec(RVec<double> v) {
                return Sum(v);
            }
        ''')
        
        func = dsl.get_registered_function("sum_vec")
        assert func is not None
        assert "RVec<double>" in func.params[0][1]
    
    def test_const_reference_parameter(self):
        """Test const reference parameter type."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"v": "RVec<double>"})
        dsl.register_function_cpp('''
            double sum_vec(const RVec<double>& v) {
                return Sum(v);
            }
        ''')
        
        func = dsl.get_registered_function("sum_vec")
        assert func is not None
        assert "const" in func.params[0][1]
        assert "&" in func.params[0][1]
    
    def test_rvec_return_type(self):
        """Test RVec return type."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"v": "RVec<double>"})
        dsl.register_function_cpp('''
            RVec<double> scale(RVec<double> v, double factor) {
                return v * factor;
            }
        ''')
        
        func = dsl.get_registered_function("scale")
        assert func is not None
        assert "RVec<double>" in func.return_type


class TestRegisterFunctionCppQuery:
    """Test query methods."""
    
    def test_get_registered_function_returns_latest(self):
        """Test get_registered_function returns most recent version."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp("double f(double x) { return x * 1; }")
        dsl.register_function_cpp("double f(double x) { return x * 2; }")
        dsl.register_function_cpp("double f(double x) { return x * 3; }")
        
        func = dsl.get_registered_function("f")
        assert "x * 3" in func.body
    
    def test_get_all_registered_functions(self):
        """Test get_all_registered_functions returns all versions."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp("double f(double x) { return x * 1; }")
        dsl.register_function_cpp("double f(double x) { return x * 2; }")
        dsl.register_function_cpp("double f(double x) { return x * 3; }")
        
        funcs = dsl.get_all_registered_functions("f")
        assert len(funcs) == 3
    
    def test_list_registered_functions(self):
        """Test list_registered_functions."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp("double f1(double x) { return x; }")
        dsl.register_function_cpp("double f2(double x) { return x; }")
        
        names = dsl.list_registered_functions()
        assert set(names) == {"f1", "f2"}
    
    def test_get_registered_cpp_name(self):
        """Test get_registered_cpp_name."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp("double pt(double x) { return x; }")
        
        cpp_name = dsl.get_registered_cpp_name("pt")
        assert cpp_name is not None
        assert cpp_name.startswith("dsl_pt_")
    
    def test_get_nonexistent_function(self):
        """Test querying non-existent function returns None."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        
        assert dsl.get_registered_function("nonexistent") is None
        assert dsl.get_registered_cpp_name("nonexistent") is None


class TestRegisterFunctionCppNameOverride:
    """Test name override parameter."""
    
    def test_name_override(self):
        """Test overriding parsed function name."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp(
            "double internal_name(double x) { return x; }",
            name="public_name"
        )
        
        # Registered under override name
        assert dsl.get_registered_function("public_name") is not None
        assert dsl.get_registered_function("internal_name") is None


class TestRegisterFunctionCppErrors:
    """Test error handling."""
    
    def test_parse_error_invalid_code(self):
        """Test parse error on invalid code."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        with pytest.raises(ValueError, match="parse|Parse"):
            dsl.register_function_cpp("not valid c++ code at all")
    
    def test_parse_error_missing_body(self):
        """Test parse error on missing function body."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        with pytest.raises(ValueError, match="parse|Parse"):
            dsl.register_function_cpp("double f(double x);")  # Declaration, not definition


class TestRegisterFunctionCppThreadSafety:
    """Test thread safety (P0-3: class-level lock)."""
    
    def test_class_level_lock_exists(self):
        """Test class-level lock is defined."""
        from RDataFrameDSL import DSLCompiler
        
        assert hasattr(DSLCompiler, '_global_cpp_compile_lock')
    
    def test_class_level_declared_names_exists(self):
        """Test class-level declared names set exists."""
        from RDataFrameDSL import DSLCompiler
        
        assert hasattr(DSLCompiler, '_global_cpp_declared_names')
        assert isinstance(DSLCompiler._global_cpp_declared_names, set)
    
    def test_concurrent_registration(self):
        """Test concurrent registration from multiple threads."""
        from RDataFrameDSL import DSLCompiler
        
        results = []
        errors = []
        
        def register_func(idx):
            try:
                dsl = DSLCompiler({"x": "double"})
                dsl.register_function_cpp(
                    f"double thread_func_{idx}(double x) {{ return x * {idx}; }}"
                )
                func = dsl.get_registered_function(f"thread_func_{idx}")
                results.append((idx, func.hash))
            except Exception as e:
                errors.append((idx, str(e)))
        
        threads = []
        for i in range(10):
            t = threading.Thread(target=register_func, args=(i,))
            threads.append(t)
        
        for t in threads:
            t.start()
        
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 10


class TestRegisterFunctionCppIntegration:
    """Integration tests with RDataFrame."""
    
    @pytest.fixture
    def simple_rdf(self):
        """Create simple RDataFrame for testing."""
        import ROOT
        return ROOT.RDataFrame(10).Define("x", "1.0 * rdfentry_")
    
    def test_apply_returns_rdf(self, simple_rdf):
        """Test apply() returns the RDataFrame."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp("double f(double x) { return x * 2; }")
        
        result = dsl.apply(simple_rdf)
        assert result is simple_rdf  # Same object returned
    
    def test_registered_function_callable(self, simple_rdf):
        """Test registered function can be used in Define."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp('''
            double double_it(double x) {
                return x * 2;
            }
        ''')
        
        # Get the internal name
        func = dsl.get_registered_function("double_it")
        cpp_name = func.cpp_name
        
        # Apply and use
        rdf = dsl.apply(simple_rdf)
        rdf = rdf.Define("y", f"{cpp_name}(x)")
        
        # Verify it works
        result = rdf.Sum("y").GetValue()
        expected = sum(i * 2 for i in range(10))  # x = 0,1,2,...,9; y = x*2
        assert abs(result - expected) < 0.001


class TestRegisterFunctionCppROOTDeclaration:
    """Test ROOT declaration behavior."""
    
    def test_function_declared_in_root(self):
        """Test function is declared in ROOT's interpreter."""
        from RDataFrameDSL import DSLCompiler
        import ROOT
        
        dsl = DSLCompiler({"x": "double"})
        dsl.register_function_cpp('''
            double test_root_decl(double x) {
                return x + 100;
            }
        ''')
        
        func = dsl.get_registered_function("test_root_decl")
        
        # Function should be declared (check via ProcessLine)
        # If function exists, calling it should work
        result = ROOT.gInterpreter.Calc(f"{func.cpp_name}(5.0)")
        assert abs(result - 105.0) < 0.001
    
    def test_idempotent_declaration(self):
        """Test re-registering same function doesn't cause errors."""
        from RDataFrameDSL import DSLCompiler
        
        code = "double idempotent_func(double x) { return x; }"
        
        dsl1 = DSLCompiler({"x": "double"})
        dsl1.register_function_cpp(code)
        
        dsl2 = DSLCompiler({"x": "double"})
        # Should not raise even though same code is declared again
        dsl2.register_function_cpp(code)
        
        # Both should have same hash
        assert dsl1.get_registered_function("idempotent_func").hash == \
               dsl2.get_registered_function("idempotent_func").hash
