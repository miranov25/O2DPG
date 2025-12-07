"""
Tests for Phase 6c: Private/Protected Member Access via Reflection.

This module tests reflection-based access to protected/private data members
using TClass::GetDataMember() and GetOffset().

Test Categories:
1. Mock Tests (no ROOT required):
   - Public member generates direct access
   - Protected/private member generates reflection lambda
   - Non-basic member rejection
   - Pointer member rejection
   - Header collection

2. ROOT Integration Tests (require ROOT):
   - TVector3 protected member access (fX, fY, fZ)
   - Custom test class with all access levels
   - Compilation tests
   - Execution tests

3. RDataFrame End-to-End Tests:
   - Define column using protected member
"""

import pytest
from unittest.mock import Mock, patch

from RDataFrameDSL.ir_nodes import (
    VariableNode, PropertyAccessNode, ConstantNode, BinaryOpNode
)
from RDataFrameDSL.ir_types import IRType, IRTypeKind
from RDataFrameDSL.ir_errors import IRError, IRErrorKind
from RDataFrameDSL.backend_cpp import (
    CppCodeGenerator, FunctionLibrary, REFLECTION_HEADERS
)

# Check if ROOT is available
try:
    import ROOT
    ROOT_AVAILABLE = True
except ImportError:
    ROOT_AVAILABLE = False


# =============================================================================
# Test Helpers
# =============================================================================

def make_object_var(name: str, class_name: str) -> VariableNode:
    """Create an object variable node."""
    return VariableNode(
        name=name,
        dtype=IRType(IRTypeKind.Object, cpp_type=class_name),
        rank=0
    )


def make_property_access(obj: VariableNode, prop_name: str, 
                          result_type: IRType = None) -> PropertyAccessNode:
    """Create a property access node."""
    if result_type is None:
        result_type = IRType(IRTypeKind.Float64)
    return PropertyAccessNode(
        object=obj,
        property_name=prop_name,
        dtype=result_type,
        rank=0
    )


# =============================================================================
# Mock Tests - No ROOT Required
# =============================================================================

class TestPublicMemberAccess:
    """Test direct access for public members."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_public_member_direct_access(self, generator):
        """Public member generates direct access code."""
        # Mock _get_data_member_info to return public member
        generator._get_data_member_info = Mock(return_value={
            'is_public': True,
            'is_basic': True,
            'is_pointer': False,
            'type_name': 'double',
            'access_level': 'public'
        })
        
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "get_x")
        
        # Should use direct access
        assert "vec.fX" in func.code
        assert "TClass" not in func.code
        assert "GetOffset" not in func.code
    
    def test_public_member_no_reflection_headers(self, generator):
        """Public member access does not include reflection headers."""
        generator._get_data_member_info = Mock(return_value={
            'is_public': True,
            'is_basic': True,
            'is_pointer': False,
            'type_name': 'double',
            'access_level': 'public'
        })
        
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "get_x")
        
        # Should NOT include reflection headers
        assert "<TClass.h>" not in func.headers
        assert "<TDataMember.h>" not in func.headers


class TestProtectedMemberAccess:
    """Test reflection access for protected members."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_protected_member_reflection_access(self, generator):
        """Protected member generates reflection-based access."""
        generator._get_data_member_info = Mock(return_value={
            'is_public': False,
            'is_basic': True,
            'is_pointer': False,
            'type_name': 'double',
            'access_level': 'protected'
        })
        
        obj = make_object_var("particle", "TParticle")
        ir = make_property_access(obj, "fPx")
        
        func = generator.generate(ir, "get_px")
        
        # Should use reflection access
        assert "TClass::GetClass" in func.code
        assert 'GetDataMember("fPx")' in func.code
        assert "GetOffset" in func.code
        assert "reinterpret_cast" in func.code
    
    def test_protected_member_lambda_pattern(self, generator):
        """Protected member uses lambda with static caching."""
        generator._get_data_member_info = Mock(return_value={
            'is_public': False,
            'is_basic': True,
            'is_pointer': False,
            'type_name': 'double',
            'access_level': 'protected'
        })
        
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "get_x")
        
        # Should use lambda pattern with static variables
        assert "[&]()" in func.code
        assert "static TClass*" in func.code
        assert "static TDataMember*" in func.code
        assert "static Long_t offset" in func.code
    
    def test_protected_member_reflection_headers(self, generator):
        """Protected member access includes reflection headers."""
        generator._get_data_member_info = Mock(return_value={
            'is_public': False,
            'is_basic': True,
            'is_pointer': False,
            'type_name': 'double',
            'access_level': 'protected'
        })
        
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "get_x")
        
        # Should include reflection headers
        assert "<TClass.h>" in func.headers
        assert "<TDataMember.h>" in func.headers


class TestPrivateMemberAccess:
    """Test reflection access for private members."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_private_member_reflection_access(self, generator):
        """Private member generates reflection-based access."""
        generator._get_data_member_info = Mock(return_value={
            'is_public': False,
            'is_basic': True,
            'is_pointer': False,
            'type_name': 'int',
            'access_level': 'private'
        })
        
        obj = make_object_var("obj", "MyClass")
        ir = make_property_access(obj, "fPrivate", IRType(IRTypeKind.Int32))
        
        func = generator.generate(ir, "get_private")
        
        # Should use reflection access
        assert "TClass::GetClass" in func.code
        assert 'GetDataMember("fPrivate")' in func.code


class TestNonBasicMemberRejection:
    """Test that non-basic members are rejected."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_non_basic_member_error(self, generator):
        """Non-basic member raises UNSUPPORTED_OP error."""
        generator._get_data_member_info = Mock(return_value={
            'is_public': False,
            'is_basic': False,  # Not a basic type
            'is_pointer': False,
            'type_name': 'TString',
            'access_level': 'protected'
        })
        
        obj = make_object_var("obj", "MyClass")
        ir = make_property_access(obj, "fName", IRType(IRTypeKind.Object, cpp_type="TString"))
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "get_name")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
        assert "non-basic" in exc_info.value.message.lower()
        assert "fName" in exc_info.value.message
    
    def test_non_basic_error_includes_access_level(self, generator):
        """Non-basic error message includes access level."""
        generator._get_data_member_info = Mock(return_value={
            'is_public': False,
            'is_basic': False,
            'is_pointer': False,
            'type_name': 'TString',
            'access_level': 'protected'
        })
        
        obj = make_object_var("obj", "MyClass")
        ir = make_property_access(obj, "fComplex", IRType(IRTypeKind.Object))
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "test")
        
        assert "protected" in exc_info.value.message


class TestPointerMemberRejection:
    """Test that pointer members are rejected."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_pointer_member_error(self, generator):
        """Pointer member raises UNSUPPORTED_OP error."""
        generator._get_data_member_info = Mock(return_value={
            'is_public': False,
            'is_basic': True,
            'is_pointer': True,  # Is a pointer
            'type_name': 'TObject*',
            'access_level': 'private'
        })
        
        obj = make_object_var("obj", "MyClass")
        ir = make_property_access(obj, "fParent", IRType(IRTypeKind.Object))
        
        with pytest.raises(IRError) as exc_info:
            generator.generate(ir, "get_parent")
        
        assert exc_info.value.kind == IRErrorKind.UNSUPPORTED_OP
        assert "pointer" in exc_info.value.message.lower()


class TestUseReflectionFlag:
    """Test use_reflection flag behavior."""
    
    def test_reflection_disabled_direct_access(self):
        """With use_reflection=False, always use direct access."""
        generator = CppCodeGenerator(use_reflection=False)
        
        # Even though we mock it as protected, should use direct access
        generator._get_data_member_info = Mock(return_value={
            'is_public': False,
            'is_basic': True,
            'is_pointer': False,
            'type_name': 'double',
            'access_level': 'protected'
        })
        
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "get_x")
        
        # Should use direct access (let C++ compiler enforce)
        assert "vec.fX" in func.code
        assert "TClass" not in func.code
    
    def test_reflection_disabled_no_headers(self):
        """With use_reflection=False, no reflection headers."""
        generator = CppCodeGenerator(use_reflection=False)
        
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "get_x")
        
        assert "<TClass.h>" not in func.headers
        assert "<TDataMember.h>" not in func.headers


class TestFallbackBehavior:
    """Test fallback when reflection info unavailable."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_member_not_found_fallback(self, generator):
        """When member info not found, fall back to direct access."""
        generator._get_data_member_info = Mock(return_value=None)
        
        obj = make_object_var("obj", "UnknownClass")
        ir = make_property_access(obj, "fUnknown")
        
        func = generator.generate(ir, "test")
        
        # Should fall back to direct access
        assert "obj.fUnknown" in func.code
        assert "TClass" not in func.code


class TestReflectionHeadersConstant:
    """Test REFLECTION_HEADERS constant."""
    
    def test_reflection_headers_content(self):
        """REFLECTION_HEADERS contains expected headers."""
        assert "<TClass.h>" in REFLECTION_HEADERS
        assert "<TDataMember.h>" in REFLECTION_HEADERS


# =============================================================================
# ROOT Integration Tests
# =============================================================================

@pytest.mark.skipif(not ROOT_AVAILABLE, reason="ROOT not available")
class TestTVector3ReflectionIntegration:
    """Integration tests with TVector3 (has protected fX, fY, fZ)."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    def test_tvector3_fx_is_protected(self):
        """Verify TVector3.fX is actually protected."""
        tclass = ROOT.TClass.GetClass("TVector3")
        dm = tclass.GetDataMember("fX")
        
        # Should NOT be public
        assert not (dm.Property() & ROOT.kIsPublic)
    
    def test_tvector3_fx_reflection_codegen(self, generator):
        """TVector3.fX generates reflection code."""
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "get_x")
        
        # Should use reflection (fX is protected)
        assert "TClass::GetClass" in func.code
        assert 'GetDataMember("fX")' in func.code
    
    def test_tvector3_fx_compiles(self, generator, library):
        """TVector3.fX reflection code compiles."""
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "get_x")
        library.add(func)
        
        # Should compile without error
        library.compile(func.name)
    
    def test_tvector3_fx_executes(self, generator, library):
        """TVector3.fX returns correct value."""
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "vec_fx")
        library.add(func)
        library.compile(func.name)
        
        # Create test vector
        v = ROOT.TVector3(1.5, 2.5, 3.5)
        
        # Call generated function
        result = getattr(ROOT, func.name)(v)
        
        assert abs(result - 1.5) < 0.001
    
    def test_tvector3_all_components(self, generator, library):
        """TVector3 fX, fY, fZ all work."""
        results = {}
        
        for comp in ['fX', 'fY', 'fZ']:
            obj = make_object_var("vec", "TVector3")
            ir = make_property_access(obj, comp)
            
            func = generator.generate(ir, f"get_{comp}")
            library.add(func)
            library.compile(func.name)
            
            v = ROOT.TVector3(1.0, 2.0, 3.0)
            results[comp] = getattr(ROOT, func.name)(v)
        
        assert abs(results['fX'] - 1.0) < 0.001
        assert abs(results['fY'] - 2.0) < 0.001
        assert abs(results['fZ'] - 3.0) < 0.001


@pytest.mark.skipif(not ROOT_AVAILABLE, reason="ROOT not available")
class TestCustomClassReflection:
    """Integration tests with custom test class."""
    
    @pytest.fixture(scope="class")
    def setup_test_class(self):
        """Create test class with known access levels."""
        ROOT.gInterpreter.Declare('''
            #ifndef TEST_ACCESS_CLASS_DEFINED
            #define TEST_ACCESS_CLASS_DEFINED
            class TestAccessClass {
            public:
                double fPublic = 1.0;
            protected:
                double fProtected = 2.0;
            private:
                double fPrivate = 3.0;
            public:
                TestAccessClass() {}
                TestAccessClass(double pub, double prot, double priv) 
                    : fPublic(pub), fProtected(prot), fPrivate(priv) {}
            };
            #endif
        ''')
        return True
    
    @pytest.fixture
    def generator(self, setup_test_class):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self, setup_test_class):
        return FunctionLibrary()
    
    def test_public_member_direct(self, generator, library):
        """Public member uses direct access."""
        obj = make_object_var("obj", "TestAccessClass")
        ir = make_property_access(obj, "fPublic")
        
        func = generator.generate(ir, "get_public")
        
        # Should use direct access (not reflection)
        assert "obj.fPublic" in func.code
        assert "TClass::GetClass" not in func.code
    
    def test_protected_member_reflection(self, generator, library):
        """Protected member uses reflection."""
        obj = make_object_var("obj", "TestAccessClass")
        ir = make_property_access(obj, "fProtected")
        
        func = generator.generate(ir, "get_protected")
        
        # Should use reflection
        assert "TClass::GetClass" in func.code
    
    def test_private_member_reflection(self, generator, library):
        """Private member uses reflection."""
        obj = make_object_var("obj", "TestAccessClass")
        ir = make_property_access(obj, "fPrivate")
        
        func = generator.generate(ir, "get_private")
        
        # Should use reflection
        assert "TClass::GetClass" in func.code
    
    def test_all_access_levels_execute(self, generator, library):
        """All access levels return correct values."""
        # Create test object
        ROOT.gInterpreter.Declare('''
            TestAccessClass testObj(10.0, 20.0, 30.0);
        ''')
        
        # Test public
        obj_pub = make_object_var("obj", "TestAccessClass")
        ir_pub = make_property_access(obj_pub, "fPublic")
        func_pub = generator.generate(ir_pub, "exec_public")
        library.add(func_pub)
        library.compile(func_pub.name)
        
        # Test protected
        obj_prot = make_object_var("obj", "TestAccessClass")
        ir_prot = make_property_access(obj_prot, "fProtected")
        func_prot = generator.generate(ir_prot, "exec_protected")
        library.add(func_prot)
        library.compile(func_prot.name)
        
        # Test private
        obj_priv = make_object_var("obj", "TestAccessClass")
        ir_priv = make_property_access(obj_priv, "fPrivate")
        func_priv = generator.generate(ir_priv, "exec_private")
        library.add(func_priv)
        library.compile(func_priv.name)
        
        # Create object and test
        obj = ROOT.TestAccessClass(10.0, 20.0, 30.0)
        
        result_pub = getattr(ROOT, func_pub.name)(obj)
        result_prot = getattr(ROOT, func_prot.name)(obj)
        result_priv = getattr(ROOT, func_priv.name)(obj)
        
        assert abs(result_pub - 10.0) < 0.001
        assert abs(result_prot - 20.0) < 0.001
        assert abs(result_priv - 30.0) < 0.001


@pytest.mark.skipif(not ROOT_AVAILABLE, reason="ROOT not available")
class TestTParticleReflection:
    """Integration tests with TParticle (physics class)."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    def test_tparticle_fpx_protected(self):
        """Verify TParticle.fPx is protected."""
        tclass = ROOT.TClass.GetClass("TParticle")
        dm = tclass.GetDataMember("fPx")
        
        assert dm is not None
        assert not (dm.Property() & ROOT.kIsPublic)
    
    def test_tparticle_fpx_compiles(self, generator, library):
        """TParticle.fPx reflection code compiles."""
        obj = make_object_var("p", "TParticle")
        ir = make_property_access(obj, "fPx")
        
        func = generator.generate(ir, "particle_px")
        library.add(func)
        library.compile(func.name)
    
    def test_tparticle_momentum_components(self, generator, library):
        """TParticle momentum components accessible."""
        # Test fPx, fPy, fPz
        for comp in ['fPx', 'fPy', 'fPz']:
            obj = make_object_var("p", "TParticle")
            ir = make_property_access(obj, comp)
            
            func = generator.generate(ir, f"get_{comp}")
            library.add(func)
            library.compile(func.name)


# =============================================================================
# RDataFrame Integration Tests
# =============================================================================

@pytest.mark.skipif(not ROOT_AVAILABLE, reason="ROOT not available")
class TestRDataFrameReflection:
    """End-to-end RDataFrame integration tests."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    @pytest.fixture
    def library(self):
        return FunctionLibrary()
    
    def test_rdataframe_tvector3_column(self, generator, library):
        """Define RDataFrame column using TVector3.fX."""
        # Create temporary file with TVector3 branch
        import tempfile
        import os
        
        tmpfile = tempfile.NamedTemporaryFile(suffix=".root", delete=False)
        tmpfile.close()
        
        try:
            # Create tree with TVector3 branch
            f = ROOT.TFile(tmpfile.name, "RECREATE")
            tree = ROOT.TTree("tree", "Test tree")
            
            vec = ROOT.TVector3()
            tree.Branch("vec", vec)
            
            for i in range(10):
                vec.SetXYZ(float(i), float(i*2), float(i*3))
                tree.Fill()
            
            tree.Write()
            f.Close()
            
            # Generate helper function for fX access
            obj = make_object_var("vec", "TVector3")
            ir = make_property_access(obj, "fX")
            
            func = generator.generate(ir, "rdf_vec_x")
            library.add(func)
            library.compile(func.name)
            
            # Use in RDataFrame
            rdf = ROOT.RDataFrame("tree", tmpfile.name)
            rdf_with_x = rdf.Define("x", f"{func.name}(vec)")
            
            # Get results
            x_vals = rdf_with_x.Take['double']("x").GetValue()
            
            # Verify
            expected = [float(i) for i in range(10)]
            for got, exp in zip(x_vals, expected):
                assert abs(got - exp) < 0.001
                
        finally:
            os.unlink(tmpfile.name)


# =============================================================================
# Access Level Detection Tests
# =============================================================================

@pytest.mark.skipif(not ROOT_AVAILABLE, reason="ROOT not available")
class TestAccessLevelDetection:
    """Test _get_data_member_info access level detection."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_detect_public(self, generator):
        """Detect public member correctly."""
        # TObject has public fUniqueID
        info = generator._get_data_member_info("TObject", "fUniqueID")
        
        if info:  # May be None if TObject has no fUniqueID
            assert 'is_public' in info
    
    def test_detect_protected(self, generator):
        """Detect protected member correctly."""
        info = generator._get_data_member_info("TVector3", "fX")
        
        assert info is not None
        assert info['is_public'] == False
        assert info['access_level'] in ('protected', 'private')
    
    def test_detect_is_basic(self, generator):
        """Detect basic type correctly."""
        info = generator._get_data_member_info("TVector3", "fX")
        
        assert info is not None
        assert info['is_basic'] == True
    
    def test_nonexistent_member(self, generator):
        """Non-existent member returns None."""
        info = generator._get_data_member_info("TVector3", "fNoSuchMember")
        
        assert info is None
    
    def test_nonexistent_class(self, generator):
        """Non-existent class returns None."""
        info = generator._get_data_member_info("NoSuchClass", "fX")
        
        assert info is None


# =============================================================================
# Header Collection Tests
# =============================================================================

class TestReflectionHeaderCollection:
    """Test header collection for reflection access."""
    
    @pytest.fixture
    def generator(self):
        return CppCodeGenerator()
    
    def test_reflection_headers_when_needed(self, generator):
        """Reflection headers included when accessing protected member."""
        generator._get_data_member_info = Mock(return_value={
            'is_public': False,
            'is_basic': True,
            'is_pointer': False,
            'type_name': 'double',
            'access_level': 'protected'
        })
        
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "test")
        
        assert "<TClass.h>" in func.headers
        assert "<TDataMember.h>" in func.headers
    
    def test_no_reflection_headers_for_public(self, generator):
        """No reflection headers for public member access."""
        generator._get_data_member_info = Mock(return_value={
            'is_public': True,
            'is_basic': True,
            'is_pointer': False,
            'type_name': 'double',
            'access_level': 'public'
        })
        
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "test")
        
        assert "<TClass.h>" not in func.headers
        assert "<TDataMember.h>" not in func.headers
    
    def test_class_header_still_included(self, generator):
        """Class header (e.g., TVector3.h) still included with reflection."""
        generator._get_data_member_info = Mock(return_value={
            'is_public': False,
            'is_basic': True,
            'is_pointer': False,
            'type_name': 'double',
            'access_level': 'protected'
        })
        
        obj = make_object_var("vec", "TVector3")
        ir = make_property_access(obj, "fX")
        
        func = generator.generate(ir, "test")
        
        # TVector3.h should still be included
        assert "<TVector3.h>" in func.headers
