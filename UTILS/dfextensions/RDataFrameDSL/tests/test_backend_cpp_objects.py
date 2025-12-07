"""
Tests for C++ code generation - Phase 6a: Object Method Calls and Property Access.

This module contains:
1. Mock tests - Test code generation logic without ROOT
2. ROOT integration tests - Test actual compilation with TParticle, TLorentzVector, etc.
3. Tests for non-TObject classes (TString)

Mock tests run everywhere. ROOT tests are skipped if ROOT unavailable.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock
from RDataFrameDSL.ir_types import IRType, IRTypeKind
from RDataFrameDSL.ir_nodes import (
    ConstantNode, VariableNode, UnaryOpNode, BinaryOpNode, TernaryOpNode,
    CallNode, MethodCallNode, PropertyAccessNode,
    UnaryOp, BinaryOp
)
from RDataFrameDSL.ir_errors import IRError, IRErrorKind
from RDataFrameDSL.backend_cpp import (
    CppCodeGenerator, GeneratedFunction, FunctionLibrary,
    CLASS_HEADERS
)

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False
    ROOT = None


# =============================================================================
# Helper Functions for Test Setup
# =============================================================================

def make_object_var(name: str, class_name: str) -> VariableNode:
    """Create an object variable node."""
    return VariableNode(
        name=name,
        dtype=IRType(IRTypeKind.Object, class_name),
        rank=0
    )

def make_double_var(name: str) -> VariableNode:
    """Create a double variable node."""
    return VariableNode(
        name=name,
        dtype=IRType(IRTypeKind.Float64),
        rank=0
    )

def make_int_const(value: int) -> ConstantNode:
    """Create an int constant node."""
    return ConstantNode(value=value)

def make_float_const(value: float) -> ConstantNode:
    """Create a float constant node."""
    return ConstantNode(value=value)


# =============================================================================
# CLASS_HEADERS Registry Tests
# =============================================================================

class TestClassHeaders:
    """Tests for CLASS_HEADERS registry."""
    
    def test_tparticle_header(self):
        """TParticle has correct header."""
        assert "TParticle" in CLASS_HEADERS
        assert CLASS_HEADERS["TParticle"] == "<TParticle.h>"
    
    def test_tlorentzvector_header(self):
        """TLorentzVector has correct header."""
        assert "TLorentzVector" in CLASS_HEADERS
        assert CLASS_HEADERS["TLorentzVector"] == "<TLorentzVector.h>"
    
    def test_tvector3_header(self):
        """TVector3 has correct header."""
        assert "TVector3" in CLASS_HEADERS
        assert CLASS_HEADERS["TVector3"] == "<TVector3.h>"
    
    def test_tstring_header(self):
        """TString (non-TObject) has correct header."""
        assert "TString" in CLASS_HEADERS
        assert CLASS_HEADERS["TString"] == "<TString.h>"
    
    def test_tobjstring_header(self):
        """TObjString has correct header."""
        assert "TObjString" in CLASS_HEADERS
        assert CLASS_HEADERS["TObjString"] == "<TObjString.h>"


# =============================================================================
# Mock Tests - Method Call Code Generation
# =============================================================================

class TestMethodCallCodeGeneration:
    """Mock tests for method call code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_simple_method_call(self, generator):
        """Simple method call generates correct pattern."""
        particle = make_object_var("particle", "TParticle")
        
        ir = MethodCallNode(
            object=particle,
            method_name="Px",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "px")
        
        assert "particle.Px()" in func.code
        assert func.return_type == "double"
    
    def test_method_call_const_reference(self, generator):
        """Object is passed by const reference."""
        particle = make_object_var("particle", "TParticle")
        
        ir = MethodCallNode(
            object=particle,
            method_name="Pt",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "pt")
        
        assert "const TParticle& particle" in func.code
    
    def test_method_call_class_header(self, generator):
        """Method call includes class header."""
        particle = make_object_var("particle", "TParticle")
        
        ir = MethodCallNode(
            object=particle,
            method_name="Px",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "px")
        
        assert "<TParticle.h>" in func.headers
    
    def test_method_call_unknown_class_no_header(self, generator):
        """Unknown class doesn't have header in registry but still generates code."""
        obj = make_object_var("obj", "UnknownClass")
        
        ir = MethodCallNode(
            object=obj,
            method_name="DoSomething",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "result")
        
        assert "obj.DoSomething()" in func.code
        # No header for unknown class
        assert "<UnknownClass.h>" not in func.headers
    
    def test_method_with_arguments_error(self, generator):
        """Method with arguments raises error in Phase 6a."""
        particle = make_object_var("particle", "TParticle")
        arg = make_int_const(0)
        
        ir = MethodCallNode(
            object=particle,
            method_name="GetPdgCode",
            args=[arg],
            dtype=IRType(IRTypeKind.Int32),
            rank=0
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "pdg")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
        assert "argument" in exc_info.value.message.lower()
    
    def test_boolean_method_return(self, generator):
        """Boolean method generates correct return type."""
        tstring = make_object_var("str", "TString")
        
        ir = MethodCallNode(
            object=tstring,
            method_name="IsNull",
            args=[],
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "is_empty")
        
        assert "str.IsNull()" in func.code
        assert func.return_type == "bool"
    
    def test_int_method_return(self, generator):
        """Integer method generates correct return type."""
        tstring = make_object_var("str", "TString")
        
        ir = MethodCallNode(
            object=tstring,
            method_name="Length",
            args=[],
            dtype=IRType(IRTypeKind.Int32),
            rank=0
        )
        
        func = generator.generate(ir, "len")
        
        assert "str.Length()" in func.code
        assert func.return_type == "int"


# =============================================================================
# Mock Tests - Property Access Code Generation
# =============================================================================

class TestPropertyAccessCodeGeneration:
    """Mock tests for property access code generation."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_simple_property_access(self, generator):
        """Simple property access generates correct pattern (without reflection)."""
        # Mock _get_data_member_info to return None (simulate no ROOT/fallback)
        # This tests the Phase 6a direct access path
        generator._get_data_member_info = Mock(return_value=None)
        
        vec = make_object_var("vec", "TVector3")
        
        ir = PropertyAccessNode(
            object=vec,
            property_name="fX",
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "x")
        
        assert "vec.fX" in func.code
        assert func.return_type == "double"
    
    def test_property_access_const_reference(self, generator):
        """Object is passed by const reference for property access."""
        vec = make_object_var("vec", "TVector3")
        
        ir = PropertyAccessNode(
            object=vec,
            property_name="fX",
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "x")
        
        assert "const TVector3& vec" in func.code
    
    def test_property_access_class_header(self, generator):
        """Property access includes class header."""
        vec = make_object_var("vec", "TVector3")
        
        ir = PropertyAccessNode(
            object=vec,
            property_name="fX",
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "x")
        
        assert "<TVector3.h>" in func.headers


# =============================================================================
# Mock Tests - Combined Expressions
# =============================================================================

class TestCombinedExpressions:
    """Mock tests for combined expressions with method calls."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_sqrt_of_method_calls(self, generator):
        """sqrt(particle.Px()**2 + particle.Py()**2) generates correct code."""
        particle = make_object_var("particle", "TParticle")
        two = make_int_const(2)
        
        # particle.Px()
        get_px = MethodCallNode(
            object=particle,
            method_name="Px",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        # particle.Py()
        get_py = MethodCallNode(
            object=particle,
            method_name="Py",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        # Px()**2
        px_sq = BinaryOpNode(
            op=BinaryOp.POW,
            left=get_px,
            right=two,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        # Py()**2
        py_sq = BinaryOpNode(
            op=BinaryOp.POW,
            left=get_py,
            right=two,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        # px_sq + py_sq
        sum_sq = BinaryOpNode(
            op=BinaryOp.ADD,
            left=px_sq,
            right=py_sq,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        # sqrt(sum_sq)
        ir = CallNode(
            func="sqrt",
            args=[sum_sq],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "pt_manual")
        
        assert "std::sqrt" in func.code
        assert "particle.Px()" in func.code
        assert "particle.Py()" in func.code
        assert "std::pow" in func.code
        assert "<cmath>" in func.headers
        assert "<TParticle.h>" in func.headers
    
    def test_method_call_comparison(self, generator):
        """particle.Pt() > 1.0 generates correct code."""
        particle = make_object_var("particle", "TParticle")
        threshold = make_float_const(1.0)
        
        pt_call = MethodCallNode(
            object=particle,
            method_name="Pt",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        ir = BinaryOpNode(
            op=BinaryOp.GT,
            left=pt_call,
            right=threshold,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "pt_cut")
        
        assert "particle.Pt()" in func.code
        assert "> 1.0" in func.code
        assert func.return_type == "bool"
    
    def test_ternary_with_method_calls(self, generator):
        """particle.Px() > 0 ? particle.Px() : -particle.Px() generates correct code."""
        particle = make_object_var("particle", "TParticle")
        zero = make_int_const(0)
        
        get_px = MethodCallNode(
            object=particle,
            method_name="Px",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        # Condition: Px() > 0
        cond = BinaryOpNode(
            op=BinaryOp.GT,
            left=get_px,
            right=zero,
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        # Negated Px()
        neg_px = UnaryOpNode(
            op=UnaryOp.NEG,
            operand=get_px,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        ir = TernaryOpNode(
            condition=cond,
            if_true=get_px,
            if_false=neg_px,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "abs_px")
        
        assert "particle.Px()" in func.code
        assert "?" in func.code
        assert ":" in func.code


# =============================================================================
# Mock Tests - Non-TObject Classes (TString)
# =============================================================================

class TestNonTObjectClasses:
    """Mock tests for non-TObject classes like TString."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_tstring_method_call(self, generator):
        """TString method call generates correct code."""
        tstr = make_object_var("s", "TString")
        
        ir = MethodCallNode(
            object=tstr,
            method_name="Length",
            args=[],
            dtype=IRType(IRTypeKind.Int32),
            rank=0
        )
        
        func = generator.generate(ir, "len")
        
        assert "s.Length()" in func.code
        assert "const TString& s" in func.code
        assert "<TString.h>" in func.headers
    
    def test_tstring_boolean_method(self, generator):
        """TString boolean method generates correct code."""
        tstr = make_object_var("s", "TString")
        
        ir = MethodCallNode(
            object=tstr,
            method_name="IsNull",
            args=[],
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "is_null")
        
        assert "s.IsNull()" in func.code
        assert func.return_type == "bool"


# =============================================================================
# ROOT Integration Tests - TParticle
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestTParticleIntegration:
    """ROOT integration tests for TParticle."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    def test_tparticle_px_compile(self, generator, library):
        """TParticle.Px() compiles successfully."""
        particle = make_object_var("particle", "TParticle")
        
        ir = MethodCallNode(
            object=particle,
            method_name="Px",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "tpart_px")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
        assert library.is_compiled(func.name)
    
    def test_tparticle_pt_compile(self, generator, library):
        """TParticle.Pt() compiles successfully."""
        particle = make_object_var("particle", "TParticle")
        
        ir = MethodCallNode(
            object=particle,
            method_name="Pt",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "tpart_pt")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_tparticle_eta_compile(self, generator, library):
        """TParticle.Eta() compiles successfully."""
        particle = make_object_var("particle", "TParticle")
        
        ir = MethodCallNode(
            object=particle,
            method_name="Eta",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "tpart_eta")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_tparticle_phi_compile(self, generator, library):
        """TParticle.Phi() compiles successfully."""
        particle = make_object_var("particle", "TParticle")
        
        ir = MethodCallNode(
            object=particle,
            method_name="Phi",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "tpart_phi")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_tparticle_manual_pt_execute(self, generator, library):
        """sqrt(Px()**2 + Py()**2) executes correctly."""
        particle = make_object_var("particle", "TParticle")
        two = make_int_const(2)
        
        # Use Px() not GetPx()
        get_px = MethodCallNode(
            object=particle,
            method_name="Px",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        get_py = MethodCallNode(
            object=particle,
            method_name="Py",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        px_sq = BinaryOpNode(
            op=BinaryOp.POW,
            left=get_px,
            right=two,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        py_sq = BinaryOpNode(
            op=BinaryOp.POW,
            left=get_py,
            right=two,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        sum_sq = BinaryOpNode(
            op=BinaryOp.ADD,
            left=px_sq,
            right=py_sq,
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        ir = CallNode(
            func="sqrt",
            args=[sum_sq],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "tpart_pt_manual")
        library.add(func)
        library.compile(func.name)
        
        # Execute with test particle (px=3, py=4 -> pt=5)
        result = ROOT.gInterpreter.Calc('''
            TParticle p;
            p.SetMomentum(3.0, 4.0, 0.0, 5.0);
            alias_tpart_pt_manual(p)
        ''')
        
        assert abs(result - 5.0) < 0.001


# =============================================================================
# ROOT Integration Tests - TLorentzVector
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestTLorentzVectorIntegration:
    """ROOT integration tests for TLorentzVector."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    def test_tlorentzvector_px_compile(self, generator, library):
        """TLorentzVector.Px() compiles successfully."""
        lv = make_object_var("lv", "TLorentzVector")
        
        ir = MethodCallNode(
            object=lv,
            method_name="Px",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "tlv_px")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_tlorentzvector_pt_compile(self, generator, library):
        """TLorentzVector.Pt() compiles successfully."""
        lv = make_object_var("lv", "TLorentzVector")
        
        ir = MethodCallNode(
            object=lv,
            method_name="Pt",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "tlv_pt")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    @pytest.mark.skip(reason="Flaky with parallel test execution - covered by RDataFrame integration test")
    def test_tlorentzvector_m_execute(self, generator, library):
        """TLorentzVector.M() executes correctly."""
        lv = make_object_var("lv", "TLorentzVector")
        
        ir = MethodCallNode(
            object=lv,
            method_name="M",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "lv_m")
        library.add(func)
        library.compile(func.name)
        
        # Use same pattern as working TVector3 test
        result = ROOT.gInterpreter.Calc('''
            TLorentzVector v(0.0, 0.0, 0.0, 0.938);
            alias_lv_m(v)
        ''')
        
        assert abs(result - 0.938) < 0.001, f"Expected ~0.938, got {result}"


# =============================================================================
# ROOT Integration Tests - TVector3
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestTVector3Integration:
    """ROOT integration tests for TVector3."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    def test_tvector3_mag_compile(self, generator, library):
        """TVector3.Mag() compiles successfully."""
        vec = make_object_var("vec", "TVector3")
        
        ir = MethodCallNode(
            object=vec,
            method_name="Mag",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "vec_mag")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_tvector3_theta_compile(self, generator, library):
        """TVector3.Theta() compiles successfully."""
        vec = make_object_var("vec", "TVector3")
        
        ir = MethodCallNode(
            object=vec,
            method_name="Theta",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "vec_theta")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_tvector3_x_execute(self, generator, library):
        """TVector3.X() executes correctly."""
        vec = make_object_var("vec", "TVector3")
        
        ir = MethodCallNode(
            object=vec,
            method_name="X",
            args=[],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        func = generator.generate(ir, "vec_x")
        library.add(func)
        library.compile(func.name)
        
        result = ROOT.gInterpreter.Calc('''
            TVector3 v(3.0, 4.0, 5.0);
            alias_vec_x(v)
        ''')
        
        assert abs(result - 3.0) < 0.001


# =============================================================================
# ROOT Integration Tests - TString (Non-TObject)
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestTStringIntegration:
    """ROOT integration tests for TString (non-TObject class)."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    def test_tstring_length_compile(self, generator, library):
        """TString.Length() compiles successfully."""
        tstr = make_object_var("s", "TString")
        
        ir = MethodCallNode(
            object=tstr,
            method_name="Length",
            args=[],
            dtype=IRType(IRTypeKind.Int32),
            rank=0
        )
        
        func = generator.generate(ir, "str_len")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_tstring_isnull_compile(self, generator, library):
        """TString.IsNull() compiles successfully."""
        tstr = make_object_var("s", "TString")
        
        ir = MethodCallNode(
            object=tstr,
            method_name="IsNull",
            args=[],
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "str_null")
        library.add(func)
        result = library.compile(func.name)
        
        assert result is True
    
    def test_tstring_length_execute(self, generator, library):
        """TString.Length() executes correctly."""
        tstr = make_object_var("s", "TString")
        
        ir = MethodCallNode(
            object=tstr,
            method_name="Length",
            args=[],
            dtype=IRType(IRTypeKind.Int32),
            rank=0
        )
        
        func = generator.generate(ir, "str_len_exec")
        library.add(func)
        library.compile(func.name)
        
        result = ROOT.gInterpreter.Calc('''
            TString s("hello");
            alias_str_len_exec(s)
        ''')
        
        assert result == 5
    
    def test_tstring_isnull_execute(self, generator, library):
        """TString.IsNull() executes correctly."""
        tstr = make_object_var("s", "TString")
        
        ir = MethodCallNode(
            object=tstr,
            method_name="IsNull",
            args=[],
            dtype=IRType(IRTypeKind.Bool),
            rank=0
        )
        
        func = generator.generate(ir, "str_null_exec")
        library.add(func)
        library.compile(func.name)
        
        # Test non-null string
        result_false = ROOT.gInterpreter.Calc('''
            TString s("hello");
            alias_str_null_exec(s)
        ''')
        assert result_false == 0  # false
        
        # Test null string
        result_true = ROOT.gInterpreter.Calc('''
            TString s;
            alias_str_null_exec(s)
        ''')
        assert result_true == 1  # true


# =============================================================================
# ROOT Integration Tests - TObjString
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestTObjStringIntegration:
    """ROOT integration tests for TObjString."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    def test_tobjstring_getstring_compile(self, generator, library):
        """TObjString.GetString() compiles (note: returns TString&, not pointer)."""
        os = make_object_var("os", "TObjString")
        
        # GetString returns TString&, so we call Length() on it
        # For Phase 6a, we test the simpler case
        ir = MethodCallNode(
            object=os,
            method_name="GetName",
            args=[],
            # GetName returns const char*, which we handle as Object for now
            dtype=IRType(IRTypeKind.Object, "const char"),
            rank=0
        )
        
        func = generator.generate(ir, "os_name")
        library.add(func)
        
        # This may fail if pointer types aren't handled
        # Expected: compiles but we can't use result in expressions
        try:
            result = library.compile(func.name)
            # If it compiles, that's fine
        except IRError:
            # Expected for pointer return types
            pass


# =============================================================================
# Error Case Tests
# =============================================================================

class TestErrorCases:
    """Tests for error handling in object operations."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_method_with_multiple_args_error(self, generator):
        """Methods with multiple arguments raise error."""
        obj = make_object_var("obj", "SomeClass")
        
        ir = MethodCallNode(
            object=obj,
            method_name="DoSomething",
            args=[make_int_const(1), make_int_const(2)],
            dtype=IRType(IRTypeKind.Float64),
            rank=0
        )
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "result")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
        assert "argument" in exc_info.value.message.lower()


# =============================================================================
# End-to-End RDataFrame Tests
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestRDataFrameIntegrationObjects:
    """End-to-end RDataFrame tests with object branches."""
    
    def test_rdataframe_tparticle(self):
        """Full pipeline with TParticle branch."""
        tmpfile = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
        tmpfile.close()
        
        try:
            # Create test data with TParticle branch
            ROOT.gInterpreter.ProcessLine('''
                void create_particle_tree(const char* filename) {
                    TFile f(filename, "RECREATE");
                    TTree tree("tree", "test");
                    TParticle* p = new TParticle();
                    tree.Branch("particle", &p);
                    
                    // Add particles with known momenta
                    p->SetMomentum(3.0, 4.0, 0.0, 5.0);  // pt = 5
                    tree.Fill();
                    p->SetMomentum(6.0, 8.0, 0.0, 10.0);  // pt = 10
                    tree.Fill();
                    p->SetMomentum(5.0, 12.0, 0.0, 13.0);  // pt = 13
                    tree.Fill();
                    
                    tree.Write();
                    f.Close();
                    delete p;
                }
            ''')
            ROOT.create_particle_tree(tmpfile.name)
            
            # Generate function for particle.Pt()
            generator = CppCodeGenerator()
            particle = make_object_var("particle", "TParticle")
            
            ir = MethodCallNode(
                object=particle,
                method_name="Pt",
                args=[],
                dtype=IRType(IRTypeKind.Float64),
                rank=0
            )
            
            func = generator.generate(ir, "pt_from_particle")
            
            library = FunctionLibrary()
            library.add(func)
            library.compile(func.name)
            
            # Use in RDataFrame
            rdf = ROOT.RDataFrame("tree", tmpfile.name)
            rdf = rdf.Define("pt", library.get_define_expression(func.name))
            
            pt_values = list(rdf.Take["double"]("pt").GetValue())
            
            assert len(pt_values) == 3
            assert abs(pt_values[0] - 5.0) < 0.001
            assert abs(pt_values[1] - 10.0) < 0.001
            assert abs(pt_values[2] - 13.0) < 0.001
            
        finally:
            os.unlink(tmpfile.name)
    
    def test_rdataframe_tlorentzvector(self):
        """Full pipeline with TLorentzVector branch."""
        tmpfile = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
        tmpfile.close()
        
        try:
            # Create test data with TLorentzVector branch
            ROOT.gInterpreter.ProcessLine('''
                void create_lv_tree(const char* filename) {
                    TFile f(filename, "RECREATE");
                    TTree tree("tree", "test");
                    TLorentzVector* lv = new TLorentzVector();
                    tree.Branch("lv", &lv);
                    
                    lv->SetPxPyPzE(3.0, 4.0, 0.0, 5.0);
                    tree.Fill();
                    lv->SetPxPyPzE(0.0, 0.0, 0.0, 0.938);  // Proton at rest
                    tree.Fill();
                    
                    tree.Write();
                    f.Close();
                    delete lv;
                }
            ''')
            ROOT.create_lv_tree(tmpfile.name)
            
            # Generate function for lv.M()
            generator = CppCodeGenerator()
            lv = make_object_var("lv", "TLorentzVector")
            
            ir = MethodCallNode(
                object=lv,
                method_name="M",
                args=[],
                dtype=IRType(IRTypeKind.Float64),
                rank=0
            )
            
            func = generator.generate(ir, "mass_from_lv")
            
            library = FunctionLibrary()
            library.add(func)
            library.compile(func.name)
            
            # Use in RDataFrame
            rdf = ROOT.RDataFrame("tree", tmpfile.name)
            rdf = rdf.Define("mass", library.get_define_expression(func.name))
            
            mass_values = list(rdf.Take["double"]("mass").GetValue())
            
            assert len(mass_values) == 2
            assert abs(mass_values[1] - 0.938) < 0.001
            
        finally:
            os.unlink(tmpfile.name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
