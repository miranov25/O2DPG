"""
Phase 13.4 D9: FROZEN RULE #1 Enforcement Test

Validates that NO lambda expressions appear in generated C++ code.
ROOT JIT crashes on lambdas in RDataFrame.Define() strings.

This test ensures the rule is mechanically enforced, preventing regression.
"""

import pytest
from RDataFrameDSL.ir_nodes_linalg import SliceParams
from RDataFrameDSL.ir_nodes_carray import (
    make_carray_element_access,
    make_carray_slice_access,
    make_carray_row_access,
    make_carray_column_access,
    make_carray_subarray_access,
    make_carray_plane_access,
)
from RDataFrameDSL.backend_carray import generate_carray_code


class TestNoLambdaInGeneratedCode:
    """
    FROZEN RULE #1: No lambda expressions in RDataFrame.Define() strings.
    
    ROOT's JIT compiler crashes on lambdas with capture scope issues.
    All generated code must use named JIT functions with macro guards.
    """
    
    # Lambda patterns that must NOT appear in generated code
    FORBIDDEN_PATTERNS = [
        "[&](",      # Capture-by-reference lambda
        "[](",       # No-capture lambda  
        "[=](",      # Capture-by-value lambda
        "[&,",       # Mixed capture lambda
        "[=,",       # Mixed capture lambda
        "->",        # Lambda return type (in isolation) - but this can appear in other contexts
    ]
    
    # Patterns that indicate lambda IIFE (Immediately Invoked Function Expression)
    LAMBDA_IIFE_PATTERNS = [
        "[&]()",     # Lambda IIFE start
        "[]()",      # Lambda IIFE start
        "}()",       # Lambda IIFE end (closing brace followed by invocation)
    ]
    
    def _check_no_lambdas(self, code: str, jit_declarations: str, context: str):
        """Check that neither code nor JIT declarations contain lambda patterns."""
        
        # Check the call-site code (should be a function call, not lambda)
        for pattern in self.LAMBDA_IIFE_PATTERNS:
            assert pattern not in code, \
                f"FROZEN RULE VIOLATION: Lambda IIFE pattern '{pattern}' found in code for {context}: {code[:100]}"
        
        # Check JIT declarations (should be named functions, not lambdas)
        for pattern in self.FORBIDDEN_PATTERNS:
            assert pattern not in jit_declarations, \
                f"FROZEN RULE VIOLATION: Lambda pattern '{pattern}' found in JIT declarations for {context}"
        
        # Additional check: code should be a function call, not a lambda expression
        assert not (code.strip().startswith("[") and "(" in code and ")" in code), \
            f"FROZEN RULE VIOLATION: Code appears to be lambda expression for {context}: {code[:100]}"
    
    # =========================================================================
    # 1D Operations
    # =========================================================================
    
    def test_1d_element_no_lambda(self):
        """1D element access generates no lambdas."""
        node = make_carray_element_access("arr", "float", [(10, True)], [5])
        result = generate_carray_code(node)
        self._check_no_lambdas(result.code, result.jit_declarations, "1D element access")
    
    def test_1d_slice_no_lambda(self):
        """1D slice generates no lambdas."""
        node = make_carray_slice_access("arr", "float", [(10, True)], SliceParams(stop=5))
        result = generate_carray_code(node)
        self._check_no_lambdas(result.code, result.jit_declarations, "1D slice")
    
    def test_1d_negative_index_no_lambda(self):
        """1D negative index generates no lambdas."""
        node = make_carray_element_access("arr", "float", [(10, True)], [-1])
        result = generate_carray_code(node)
        self._check_no_lambdas(result.code, result.jit_declarations, "1D negative index")
    
    def test_1d_reverse_slice_no_lambda(self):
        """1D reverse slice generates no lambdas."""
        node = make_carray_slice_access("arr", "float", [(10, True)], 
                                         SliceParams(start=9, stop=-1, step=-1))
        result = generate_carray_code(node)
        self._check_no_lambdas(result.code, result.jit_declarations, "1D reverse slice")
    
    # =========================================================================
    # 2D Operations
    # =========================================================================
    
    def test_2d_element_no_lambda(self):
        """2D element access generates no lambdas."""
        node = make_carray_element_access("mat", "float", [(3, True), (4, True)], [1, 2])
        result = generate_carray_code(node)
        self._check_no_lambdas(result.code, result.jit_declarations, "2D element access")
    
    def test_2d_row_no_lambda(self):
        """2D row access generates no lambdas."""
        node = make_carray_row_access("mat", "float", [(3, True), (4, True)], 1)
        result = generate_carray_code(node)
        self._check_no_lambdas(result.code, result.jit_declarations, "2D row access")
    
    def test_2d_column_no_lambda(self):
        """2D column access generates no lambdas."""
        node = make_carray_column_access("mat", "float", [(3, True), (4, True)], 2)
        result = generate_carray_code(node)
        self._check_no_lambdas(result.code, result.jit_declarations, "2D column access")
    
    def test_2d_subarray_no_lambda(self):
        """2D subarray access generates no lambdas."""
        node = make_carray_subarray_access("mat", "float", [(3, True), (4, True)],
                                            SliceParams(stop=2), SliceParams(stop=3))
        result = generate_carray_code(node)
        self._check_no_lambdas(result.code, result.jit_declarations, "2D subarray access")
    
    # =========================================================================
    # 3D Operations
    # =========================================================================
    
    def test_3d_element_no_lambda(self):
        """3D element access generates no lambdas."""
        node = make_carray_element_access("tensor", "float", 
                                           [(2, True), (3, True), (4, True)], [0, 1, 2])
        result = generate_carray_code(node)
        self._check_no_lambdas(result.code, result.jit_declarations, "3D element access")
    
    def test_3d_plane_no_lambda(self):
        """3D plane access generates no lambdas."""
        node = make_carray_plane_access("tensor", "float",
                                         [(2, True), (3, True), (4, True)], 0)
        result = generate_carray_code(node)
        self._check_no_lambdas(result.code, result.jit_declarations, "3D plane access")
    
    # =========================================================================
    # Hybrid Arrays (variable dimension)
    # =========================================================================
    
    def test_hybrid_2d_no_lambda(self):
        """Hybrid 2D array (float[n][3]) generates no lambdas."""
        node = make_carray_column_access("hits", "float", [("n", False), (3, True)], 0)
        result = generate_carray_code(node)
        self._check_no_lambdas(result.code, result.jit_declarations, "hybrid 2D array")


class TestNamedFunctionPattern:
    """Verify generated code uses named functions with macro guards."""
    
    def test_has_ifndef_guard(self):
        """Generated JIT declarations use #ifndef guards."""
        node = make_carray_element_access("arr", "float", [(10, True)], [5])
        result = generate_carray_code(node)
        
        assert "#ifndef" in result.jit_declarations, \
            "JIT declarations must use #ifndef guards"
        assert "#define" in result.jit_declarations, \
            "JIT declarations must use #define"
        assert "#endif" in result.jit_declarations, \
            "JIT declarations must use #endif"
    
    def test_has_named_function(self):
        """Generated code calls a named function (not lambda)."""
        node = make_carray_element_access("arr", "float", [(10, True)], [5])
        result = generate_carray_code(node)
        
        # Code should be a function call like: carray_elem1d_abc12345(arr, 10, 5)
        assert result.code.startswith("carray_"), \
            f"Generated code should call named function: {result.code}"
        assert "(" in result.code and ")" in result.code, \
            f"Generated code should be a function call: {result.code}"
    
    def test_function_uses_const_rvec_ref(self):
        """Generated functions use const RVec<T>& parameters."""
        node = make_carray_element_access("arr", "float", [(10, True)], [5])
        result = generate_carray_code(node)
        
        assert "const ROOT::RVec<float>&" in result.jit_declarations, \
            "Functions must use const RVec<T>& parameters"
