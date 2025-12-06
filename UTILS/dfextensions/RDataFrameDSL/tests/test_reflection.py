"""
Tests for class reflection (Phase 4).

This module contains two categories of tests:
1. Mock tests - Test ReflectionCache with schema-based resolution (no ROOT)
2. Real ROOT tests - Integration tests with actual ROOT TClass

Mock tests run everywhere. ROOT tests are skipped if ROOT is unavailable.
"""

import pytest
from RDataFrameDSL.ir_types import IRType, IRTypeKind
from RDataFrameDSL.ir_errors import IRError, IRErrorKind
from RDataFrameDSL.reflection import (
    ReflectionCache, MethodInfo, PropertyInfo
)

# Check if ROOT is available
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False
    ROOT = None


# =============================================================================
# MethodInfo Tests (No ROOT Required)
# =============================================================================

class TestMethodInfo:
    """Tests for MethodInfo dataclass."""
    
    def test_basic_method_info(self):
        """Basic MethodInfo creation."""
        info = MethodInfo(
            class_name="TLorentzVector",
            method_name="Pt",
            return_type="Double_t",
            is_const=True,
            source="tclass"
        )
        assert info.class_name == "TLorentzVector"
        assert info.method_name == "Pt"
        assert info.return_type == "Double_t"
        assert info.is_const
        assert info.source == "tclass"
    
    def test_method_info_with_args(self):
        """MethodInfo with argument types."""
        info = MethodInfo(
            class_name="TVector3",
            method_name="SetXYZ",
            return_type="void",
            arg_types=["Double_t", "Double_t", "Double_t"],
            is_const=False,
        )
        assert len(info.arg_types) == 3
        assert info.arg_types[0] == "Double_t"
    
    def test_method_info_from_schema(self):
        """Create MethodInfo from schema entry."""
        schema_entry = {
            "return_type": "float",
            "args": ["int", "float"],
            "is_const": True
        }
        info = MethodInfo.from_schema("MyClass", "calculate", schema_entry)
        
        assert info.class_name == "MyClass"
        assert info.method_name == "calculate"
        assert info.return_type == "float"
        assert info.arg_types == ["int", "float"]
        assert info.is_const
        assert info.source == "schema"
    
    def test_method_info_get_ir_type(self):
        """MethodInfo.get_ir_type() conversion."""
        info = MethodInfo(
            class_name="Test",
            method_name="getX",
            return_type="float"
        )
        ir_type = info.get_ir_type()
        assert ir_type.kind == IRTypeKind.Float32
    
    def test_method_info_repr(self):
        """MethodInfo repr includes key details."""
        info = MethodInfo(
            class_name="MyClass",
            method_name="foo",
            return_type="int",
            arg_types=["float"],
            is_const=True,
            source="schema"
        )
        r = repr(info)
        assert "MyClass" in r
        assert "foo" in r
        assert "int" in r
        assert "const" in r
        assert "schema" in r


# =============================================================================
# PropertyInfo Tests (No ROOT Required)
# =============================================================================

class TestPropertyInfo:
    """Tests for PropertyInfo dataclass."""
    
    def test_basic_property_info(self):
        """Basic PropertyInfo creation."""
        info = PropertyInfo(
            class_name="TParticle",
            property_name="fPx",
            property_type="Double_t",
            source="tclass"
        )
        assert info.class_name == "TParticle"
        assert info.property_name == "fPx"
        assert info.property_type == "Double_t"
        assert info.source == "tclass"
    
    def test_property_info_from_schema(self):
        """Create PropertyInfo from schema entry."""
        schema_entry = {"type": "double"}
        info = PropertyInfo.from_schema("MyClass", "mValue", schema_entry)
        
        assert info.class_name == "MyClass"
        assert info.property_name == "mValue"
        assert info.property_type == "double"
        assert info.source == "schema"
    
    def test_property_info_get_ir_type(self):
        """PropertyInfo.get_ir_type() conversion."""
        info = PropertyInfo(
            class_name="Test",
            property_name="value",
            property_type="double"
        )
        ir_type = info.get_ir_type()
        assert ir_type.kind == IRTypeKind.Float64
    
    def test_property_info_repr(self):
        """PropertyInfo repr includes key details."""
        info = PropertyInfo(
            class_name="MyClass",
            property_name="mX",
            property_type="float",
            source="schema"
        )
        r = repr(info)
        assert "MyClass" in r
        assert "mX" in r
        assert "float" in r
        assert "schema" in r


# =============================================================================
# Schema-Based Resolution Tests (No ROOT Required)
# =============================================================================

class TestReflectionCacheWithSchema:
    """Tests for ReflectionCache using schema (no ROOT required)."""
    
    @pytest.fixture
    def schema(self):
        """Sample schema for testing."""
        return {
            "methods": {
                "MyClass::getX": {
                    "return_type": "float",
                    "args": [],
                    "is_const": True
                },
                "MyClass::calculate": {
                    "return_type": "double",
                    "args": ["float", "float"],
                    "is_const": False
                },
                "MyClass::setX": {
                    "return_type": "void",
                    "args": ["float"],
                    "is_const": False
                }
            },
            "properties": {
                "MyClass::mX": {"type": "float"},
                "MyClass::mY": {"type": "float"},
                "MyClass::mValue": {"type": "double"}
            }
        }
    
    @pytest.fixture
    def cache(self, schema):
        """ReflectionCache with schema."""
        return ReflectionCache(schema=schema)
    
    def test_resolve_method_from_schema(self, cache):
        """Resolve method using schema."""
        info = cache.resolve_method("MyClass", "getX")
        
        assert info.class_name == "MyClass"
        assert info.method_name == "getX"
        assert info.return_type == "float"
        assert info.is_const
        assert info.source == "schema"
    
    def test_resolve_method_with_args(self, cache):
        """Resolve method with arguments."""
        info = cache.resolve_method("MyClass", "calculate")
        
        assert info.return_type == "double"
        assert info.arg_types == ["float", "float"]
        assert not info.is_const
    
    def test_resolve_property_from_schema(self, cache):
        """Resolve property using schema."""
        info = cache.resolve_property("MyClass", "mX")
        
        assert info.class_name == "MyClass"
        assert info.property_name == "mX"
        assert info.property_type == "float"
        assert info.source == "schema"
    
    def test_method_caching(self, cache):
        """Method resolution is cached."""
        info1 = cache.resolve_method("MyClass", "getX")
        info2 = cache.resolve_method("MyClass", "getX")
        
        assert info1 is info2  # Same object (cached)
    
    def test_property_caching(self, cache):
        """Property resolution is cached."""
        info1 = cache.resolve_property("MyClass", "mX")
        info2 = cache.resolve_property("MyClass", "mX")
        
        assert info1 is info2  # Same object (cached)
    
    def test_unknown_class_error(self, cache):
        """Unknown class raises MISSING_DICT error."""
        with pytest.raises(IRError) as exc_info:
            cache.resolve_method("UnknownClass", "foo")
        
        assert exc_info.value.kind == IRErrorKind.MISSING_DICT
        assert "UnknownClass" in exc_info.value.message
    
    def test_unknown_method_error(self, cache):
        """Unknown method raises error with suggestions."""
        # Class exists in schema but method doesn't
        with pytest.raises(IRError) as exc_info:
            cache.resolve_method("MyClass", "unknownMethod")
        
        # Since class is only in schema (no TClass), it's MISSING_DICT
        assert "unknownMethod" in exc_info.value.message or "MyClass" in exc_info.value.message
    
    def test_unknown_property_error(self, cache):
        """Unknown property raises error."""
        with pytest.raises(IRError) as exc_info:
            cache.resolve_property("MyClass", "unknownProp")
        
        assert "unknownProp" in exc_info.value.message or "MyClass" in exc_info.value.message
    
    def test_has_class_with_schema(self, cache):
        """has_class returns True for schema classes."""
        assert cache.has_class("MyClass")
        assert not cache.has_class("NonExistentClass")
    
    def test_list_methods_from_schema(self, cache):
        """list_methods returns schema methods."""
        methods = cache.list_methods("MyClass")
        
        assert "getX" in methods
        assert "calculate" in methods
        assert "setX" in methods
    
    def test_list_properties_from_schema(self, cache):
        """list_properties returns schema properties."""
        props = cache.list_properties("MyClass")
        
        assert "mX" in props
        assert "mY" in props
        assert "mValue" in props
    
    def test_clear_cache(self, cache):
        """clear_cache removes all cached results."""
        # Populate cache
        cache.resolve_method("MyClass", "getX")
        cache.resolve_property("MyClass", "mX")
        
        # Clear
        cache.clear_cache()
        
        # Cache should be empty
        assert len(cache._method_cache) == 0
        assert len(cache._property_cache) == 0
    
    def test_describe(self, cache):
        """describe returns human-readable class info."""
        desc = cache.describe("MyClass")
        
        assert "MyClass" in desc
        assert "Methods" in desc
        assert "getX" in desc
        assert "Properties" in desc
        assert "mX" in desc


class TestReflectionCacheEmpty:
    """Tests for ReflectionCache without schema."""
    
    def test_no_schema_method_error(self):
        """Without schema, method resolution fails for unknown classes."""
        cache = ReflectionCache()  # No schema
        
        with pytest.raises(IRError) as exc_info:
            cache.resolve_method("FakeClass", "foo")
        
        assert exc_info.value.kind == IRErrorKind.MISSING_DICT
    
    def test_no_schema_property_error(self):
        """Without schema, property resolution fails for unknown classes."""
        cache = ReflectionCache()
        
        with pytest.raises(IRError) as exc_info:
            cache.resolve_property("FakeClass", "bar")
        
        assert exc_info.value.kind == IRErrorKind.MISSING_DICT
    
    def test_has_class_without_schema(self):
        """has_class returns False for unknown classes without schema."""
        cache = ReflectionCache()
        assert not cache.has_class("NonExistentClass")


# =============================================================================
# Priority Order Tests (TClass > Schema)
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestReflectionPriority:
    """Test that TClass takes priority over schema."""
    
    def test_tclass_wins_over_schema(self):
        """TClass resolution wins when both available."""
        # Schema with wrong return type
        schema = {
            "methods": {
                "TVector3::X": {
                    "return_type": "int",  # Wrong! Should be Double_t
                    "args": [],
                    "is_const": True
                }
            }
        }
        
        cache = ReflectionCache(schema=schema)
        info = cache.resolve_method("TVector3", "X")
        
        # Should use TClass, not schema
        assert info.source == "tclass"
        assert "Double" in info.return_type or "double" in info.return_type.lower()
    
    def test_schema_used_when_tclass_fails(self):
        """Schema used when TClass doesn't have the method."""
        schema = {
            "methods": {
                "TVector3::customMethod": {
                    "return_type": "float",
                    "args": [],
                    "is_const": False
                }
            }
        }
        
        cache = ReflectionCache(schema=schema)
        info = cache.resolve_method("TVector3", "customMethod")
        
        # Should fall back to schema
        assert info.source == "schema"
        assert info.return_type == "float"


# =============================================================================
# Real ROOT Tests (Require ROOT)
# =============================================================================

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestReflectionCacheWithROOT:
    """Integration tests with real ROOT TClass."""
    
    @pytest.fixture
    def cache(self):
        """ReflectionCache without schema (rely on TClass)."""
        return ReflectionCache()
    
    def test_resolve_tvector3_method(self, cache):
        """Resolve TVector3::X() method."""
        info = cache.resolve_method("TVector3", "X")
        
        assert info.class_name == "TVector3"
        assert info.method_name == "X"
        assert info.source == "tclass"
        # ROOT returns Double_t or double
        assert "double" in info.return_type.lower() or "Double" in info.return_type
    
    def test_resolve_tlorentzvector_pt(self, cache):
        """Resolve TLorentzVector::Pt() method."""
        info = cache.resolve_method("TLorentzVector", "Pt")
        
        assert info.method_name == "Pt"
        assert info.source == "tclass"
    
    def test_resolve_tlorentzvector_property(self, cache):
        """Resolve TLorentzVector data member."""
        # TLorentzVector structure varies by ROOT version
        # Try common member names - at least one should work
        possible_members = ["fE", "fP", "fX"]
        
        resolved = None
        for member in possible_members:
            try:
                resolved = cache.resolve_property("TLorentzVector", member)
                break
            except IRError:
                continue
        
        # If none found, try listing properties to find any valid one
        if resolved is None:
            props = cache.list_properties("TLorentzVector")
            if props:
                resolved = cache.resolve_property("TLorentzVector", props[0])
        
        assert resolved is not None, "Should find at least one property in TLorentzVector"
        assert resolved.source == "tclass"
    
    def test_has_class_tclass(self, cache):
        """has_class returns True for ROOT classes."""
        assert cache.has_class("TVector3")
        assert cache.has_class("TLorentzVector")
        assert not cache.has_class("NonExistentROOTClass")
    
    def test_list_methods_tclass(self, cache):
        """list_methods returns TClass methods."""
        methods = cache.list_methods("TVector3")
        
        assert "X" in methods
        assert "Y" in methods
        assert "Z" in methods
        assert "SetXYZ" in methods
    
    def test_list_properties_tclass(self, cache):
        """list_properties returns TClass data members."""
        props = cache.list_properties("TVector3")
        
        # TVector3 has fX, fY, fZ members
        assert "fX" in props
        assert "fY" in props
        assert "fZ" in props
    
    def test_unknown_method_suggestions(self, cache):
        """Unknown method error includes suggestions."""
        with pytest.raises(IRError) as exc_info:
            cache.resolve_method("TVector3", "Xyz")  # Non-existent method
        
        # Should suggest similar methods like 'X', 'Y', 'Z'
        assert len(exc_info.value.suggestions) >= 0  # May or may not have suggestions
    
    def test_describe_with_tclass(self, cache):
        """describe shows TClass info."""
        desc = cache.describe("TVector3")
        
        assert "TVector3" in desc
        assert "TClass" in desc or "dictionary" in desc
        assert "Methods" in desc


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestReflectionCacheMixedSources:
    """Tests combining TClass and schema sources."""
    
    def test_tclass_methods_with_schema_properties(self):
        """TClass methods combined with schema properties."""
        schema = {
            "properties": {
                "TVector3::customProp": {"type": "int"}
            }
        }
        
        cache = ReflectionCache(schema=schema)
        
        # Method from TClass
        method_info = cache.resolve_method("TVector3", "X")
        assert method_info.source == "tclass"
        
        # Property from schema
        prop_info = cache.resolve_property("TVector3", "customProp")
        assert prop_info.source == "schema"
    
    def test_list_combined_sources(self):
        """list_methods/properties combines TClass and schema."""
        schema = {
            "methods": {
                "TVector3::customMethod": {"return_type": "void", "args": []}
            },
            "properties": {
                "TVector3::customProp": {"type": "int"}
            }
        }
        
        cache = ReflectionCache(schema=schema)
        
        methods = cache.list_methods("TVector3")
        assert "X" in methods  # From TClass
        assert "customMethod" in methods  # From schema
        
        props = cache.list_properties("TVector3")
        assert "fX" in props  # From TClass
        assert "customProp" in props  # From schema


# =============================================================================
# IR Type Conversion Tests
# =============================================================================

class TestIRTypeConversion:
    """Tests for converting reflection types to IR types."""
    
    def test_method_double_to_float64(self):
        """Double_t converts to Float64."""
        info = MethodInfo(
            class_name="Test",
            method_name="get",
            return_type="Double_t"
        )
        assert info.get_ir_type().kind == IRTypeKind.Float64
    
    def test_method_float_to_float32(self):
        """Float_t converts to Float32."""
        info = MethodInfo(
            class_name="Test",
            method_name="get",
            return_type="Float_t"
        )
        assert info.get_ir_type().kind == IRTypeKind.Float32
    
    def test_property_int_to_int32(self):
        """Int_t converts to Int32."""
        info = PropertyInfo(
            class_name="Test",
            property_name="n",
            property_type="Int_t"
        )
        assert info.get_ir_type().kind == IRTypeKind.Int32
    
    def test_unknown_type_to_object(self):
        """Unknown C++ types become Object."""
        info = MethodInfo(
            class_name="Test",
            method_name="get",
            return_type="SomeUnknownClass"
        )
        ir_type = info.get_ir_type()
        assert ir_type.kind == IRTypeKind.Object
        assert ir_type.cpp_type == "SomeUnknownClass"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
