"""
Phase 8 Tests: Method and Property Broadcasting

Tests for element-wise method/property calls on RVec<Object>.
"""

import pytest
from typing import Dict

from RDataFrameDSL.ir_builder import IRBuilder
from RDataFrameDSL.type_inferrer import TypeInferrer
from RDataFrameDSL.backend_cpp import CppCodeGenerator
from RDataFrameDSL.ir_nodes import (
    MethodBroadcastNode, PropertyBroadcastNode, RVecSliceNode
)
from RDataFrameDSL.ir_errors import IRError


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def track_schema():
    """Schema with RVec<TLorentzVector> tracks."""
    return {'columns': {
        'tracks': {'dtype': 'RVec<TLorentzVector>', 'rank': 1},
        'jets': {'dtype': 'RVec<TLorentzVector>', 'rank': 1},
    }}


@pytest.fixture
def particle_schema():
    """Schema with RVec<TParticle> particles."""
    return {'columns': {
        'particles': {'dtype': 'RVec<TParticle>', 'rank': 1},
    }}


@pytest.fixture
def vector_schema():
    """Schema with RVec<TVector3>."""
    return {'columns': {
        'vectors': {'dtype': 'RVec<TVector3>', 'rank': 1},
    }}


@pytest.fixture
def track_builder(track_schema):
    """IRBuilder for track schema."""
    inferrer = TypeInferrer.from_schema(track_schema)
    return IRBuilder(inferrer)


@pytest.fixture
def particle_builder(particle_schema):
    """IRBuilder for particle schema."""
    inferrer = TypeInferrer.from_schema(particle_schema)
    return IRBuilder(inferrer)


@pytest.fixture
def track_generator(track_schema):
    """CppCodeGenerator for track schema."""
    inferrer = TypeInferrer.from_schema(track_schema)
    return CppCodeGenerator(inferrer)


@pytest.fixture
def particle_generator(particle_schema):
    """CppCodeGenerator for particle schema."""
    inferrer = TypeInferrer.from_schema(particle_schema)
    return CppCodeGenerator(inferrer)


# =============================================================================
# IR Builder Tests - Method Broadcast Detection
# =============================================================================

class TestMethodBroadcastDetection:
    """Test that IRBuilder detects method broadcast patterns."""
    
    def test_method_broadcast_creates_correct_node(self, track_builder):
        """tracks.Pt() creates MethodBroadcastNode."""
        node = track_builder.build("tracks.Pt()")
        
        assert isinstance(node, MethodBroadcastNode)
        assert node.method_name == "Pt"
        assert node.element_type == "TLorentzVector"
        # Accept both 'double' and 'Double_t' (ROOT typedef)
        assert node.result_element_type in ("double", "Double_t")
        assert node.rank == 1
    
    def test_method_broadcast_various_methods(self, track_builder):
        """Various TLorentzVector methods work."""
        # Accept both C++ and ROOT typedefs
        double_types = ("double", "Double_t")
        
        methods = [
            ("tracks.Eta()", "Eta"),
            ("tracks.Phi()", "Phi"),
            ("tracks.M()", "M"),
            ("tracks.Px()", "Px"),
            ("tracks.Py()", "Py"),
            ("tracks.Pz()", "Pz"),
            ("tracks.E()", "E"),
        ]
        
        for expr, method_name in methods:
            node = track_builder.build(expr)
            assert isinstance(node, MethodBroadcastNode), f"Failed for {expr}"
            assert node.method_name == method_name
            assert node.result_element_type in double_types, f"Unexpected type {node.result_element_type} for {expr}"
    
    def test_method_returning_object(self, track_builder):
        """tracks.Vect() returns RVec<TVector3>."""
        node = track_builder.build("tracks.Vect()")
        
        assert isinstance(node, MethodBroadcastNode)
        assert node.method_name == "Vect"
        assert node.result_element_type == "TVector3"
        assert "TVector3" in str(node.dtype)
    
    def test_unknown_method_error(self, track_builder):
        """Unknown method gives helpful error."""
        with pytest.raises(IRError) as exc:
            track_builder.build("tracks.NonExistent()")
        
        assert "not found" in str(exc.value).lower()
        assert "TLorentzVector" in str(exc.value)
        # Should have suggestions
        assert exc.value.suggestions


# =============================================================================
# IR Builder Tests - Property Broadcast Detection
# =============================================================================

class TestPropertyBroadcastDetection:
    """Test that IRBuilder detects property broadcast patterns."""
    
    def test_property_broadcast_creates_correct_node(self, particle_builder):
        """particles.fPx creates PropertyBroadcastNode."""
        node = particle_builder.build("particles.fPx")
        
        assert isinstance(node, PropertyBroadcastNode)
        assert node.property_name == "fPx"
        assert node.element_type == "TParticle"
        # Accept both C++ and ROOT typedefs
        assert node.result_element_type in ("double", "Double_t")
        assert node.rank == 1
    
    def test_property_broadcast_various_properties(self, particle_builder):
        """Various TParticle properties work."""
        # Accept both C++ and ROOT typedefs
        double_types = ("double", "Double_t")
        int_types = ("int", "Int_t")
        
        properties = [
            ("particles.fPx", "fPx", double_types),
            ("particles.fPy", "fPy", double_types),
            ("particles.fPz", "fPz", double_types),
            ("particles.fE", "fE", double_types),
            ("particles.fPdgCode", "fPdgCode", int_types),
        ]
        
        for expr, prop_name, valid_types in properties:
            node = particle_builder.build(expr)
            assert isinstance(node, PropertyBroadcastNode), f"Failed for {expr}"
            assert node.property_name == prop_name
            assert node.result_element_type in valid_types, f"Unexpected type {node.result_element_type} for {expr}"


# =============================================================================
# IR Builder Tests - Slice + Broadcast Patterns
# =============================================================================

class TestSliceBroadcastPatterns:
    """Test slice-then-broadcast and error for broadcast-then-slice."""
    
    def test_slice_then_broadcast_works(self, track_builder):
        """tracks[:3].Pt() creates slice → broadcast chain."""
        node = track_builder.build("tracks[:3].Pt()")
        
        assert isinstance(node, MethodBroadcastNode)
        assert isinstance(node.target, RVecSliceNode)
        assert node.method_name == "Pt"
    
    def test_filter_then_broadcast_pattern(self, track_builder):
        """tracks[mask].Eta() pattern builds correctly."""
        # Add mask to schema
        schema = {'columns': {
            'tracks': {'dtype': 'RVec<TLorentzVector>', 'rank': 1},
            'mask': {'dtype': 'RVec<bool>', 'rank': 1},
        }}
        inferrer = TypeInferrer.from_schema(schema)
        builder = IRBuilder(inferrer)
        
        # This tests boolean masking followed by broadcast
        node = builder.build("tracks[mask].Eta()")
        
        assert isinstance(node, MethodBroadcastNode)
        assert node.method_name == "Eta"
    
    def test_broadcast_then_slice_error(self, track_builder):
        """tracks.Pt()[:3] gives helpful error."""
        with pytest.raises(IRError) as exc:
            track_builder.build("tracks.Pt()[:3]")
        
        assert "not supported" in str(exc.value).lower()
        # Should suggest correct pattern
        suggestions = " ".join(exc.value.suggestions)
        assert "tracks[:n].Pt()" in suggestions or "tracks[:3].Pt()" in suggestions


# =============================================================================
# Backend Tests - Method Broadcast Code Generation
# =============================================================================

class TestMethodBroadcastCodeGen:
    """Test C++ code generation for method broadcasting."""
    
    def test_gen_method_broadcast_loop(self, track_builder, track_generator):
        """tracks.Pt() generates correct loop code."""
        ir = track_builder.build("tracks.Pt()")
        func = track_generator.generate(ir, "track_pts")
        
        # Check code structure
        assert "for (const auto& elem" in func.code
        assert "elem.Pt()" in func.code
        assert "result.reserve(" in func.code
        assert "result.push_back(" in func.code
    
    def test_gen_return_type_is_rvec(self, track_builder, track_generator):
        """Return type is RVec<T>."""
        ir = track_builder.build("tracks.Pt()")
        func = track_generator.generate(ir, "test")
        
        # Accept both C++ and ROOT typedefs
        assert "ROOT::RVec<double>" in func.return_type or "ROOT::RVec<Double_t>" in func.return_type
    
    def test_gen_input_type_has_prefix(self, track_builder, track_generator):
        """Input parameter has ROOT:: prefix."""
        ir = track_builder.build("tracks.Pt()")
        func = track_generator.generate(ir, "test")
        
        # Check inputs tuple
        assert len(func.inputs) == 1
        assert func.inputs[0][0] == "tracks"
        assert "ROOT::RVec<TLorentzVector>" in func.inputs[0][1]
    
    def test_gen_headers_include_element_type(self, track_builder, track_generator):
        """Headers include element type header."""
        ir = track_builder.build("tracks.Pt()")
        func = track_generator.generate(ir, "test")
        
        # Should include TLorentzVector header
        assert any("TLorentzVector" in h for h in func.headers)
        # Should include RVec header
        assert any("RVec" in h for h in func.headers)
    
    def test_gen_method_returning_object(self, track_builder, track_generator):
        """tracks.Vect() generates RVec<TVector3>."""
        ir = track_builder.build("tracks.Vect()")
        func = track_generator.generate(ir, "track_vects")
        
        assert "ROOT::RVec<TVector3>" in func.return_type
        assert "elem.Vect()" in func.code


# =============================================================================
# Backend Tests - Property Broadcast Code Generation
# =============================================================================

class TestPropertyBroadcastCodeGen:
    """Test C++ code generation for property broadcasting."""
    
    def test_gen_property_broadcast_loop(self, particle_builder, particle_generator):
        """particles.fPx generates correct loop code."""
        ir = particle_builder.build("particles.fPx")
        func = particle_generator.generate(ir, "all_px")
        
        assert "for (const auto& elem" in func.code
        assert "elem.fPx" in func.code
        assert "result.reserve(" in func.code
    
    def test_gen_property_return_type(self, particle_builder, particle_generator):
        """Property return type is RVec<T>."""
        ir = particle_builder.build("particles.fPx")
        func = particle_generator.generate(ir, "test")
        
        # Accept both C++ and ROOT typedefs
        assert "ROOT::RVec<double>" in func.return_type or "ROOT::RVec<Double_t>" in func.return_type


# =============================================================================
# Backend Tests - Slice Then Broadcast Code Generation
# =============================================================================

class TestSliceBroadcastCodeGen:
    """Test code generation for slice-then-broadcast patterns."""
    
    def test_gen_slice_then_broadcast(self, track_builder, track_generator):
        """tracks[:3].Pt() generates nested code."""
        ir = track_builder.build("tracks[:3].Pt()")
        func = track_generator.generate(ir, "lead3_pt")
        
        # Should have slice code (Take)
        assert "Take" in func.code or "take" in func.code.lower()
        # Should have broadcast code
        assert "elem.Pt()" in func.code
    
    def test_gen_negative_slice_then_broadcast(self, track_builder, track_generator):
        """tracks[-3:].Eta() generates correct code."""
        ir = track_builder.build("tracks[-3:].Eta()")
        func = track_generator.generate(ir, "last3_eta")
        
        # Should have broadcast code
        assert "elem.Eta()" in func.code


# =============================================================================
# Backend Tests - Multiple Broadcasts
# =============================================================================

class TestMultipleBroadcasts:
    """Test multiple broadcast expressions."""
    
    def test_multiple_methods_same_source(self, track_builder, track_generator):
        """Multiple broadcasts from same source generate separate functions."""
        ir1 = track_builder.build("tracks.Pt()")
        ir2 = track_builder.build("tracks.Eta()")
        
        func1 = track_generator.generate(ir1, "pts")
        func2 = track_generator.generate(ir2, "etas")
        
        assert "elem.Pt()" in func1.code
        assert "elem.Eta()" in func2.code


# =============================================================================
# Edge Cases
# =============================================================================

class TestBroadcastEdgeCases:
    """Test edge cases for broadcasting."""
    
    def test_rvec_scalar_no_broadcast(self):
        """RVec<double>.size() doesn't broadcast - it's RVec method."""
        schema = {'columns': {
            'pt': {'dtype': 'RVec<double>', 'rank': 1},
        }}
        inferrer = TypeInferrer.from_schema(schema)
        builder = IRBuilder(inferrer)
        
        # size() on RVec<double> should NOT be a broadcast
        node = builder.build("pt.size()")
        
        assert not isinstance(node, MethodBroadcastNode)
    
    def test_scalar_object_no_broadcast(self):
        """Scalar object method call doesn't broadcast."""
        schema = {'columns': {
            'track': {'dtype': 'TLorentzVector', 'rank': 0},
        }}
        inferrer = TypeInferrer.from_schema(schema)
        builder = IRBuilder(inferrer)
        
        # Single track.Pt() should NOT be a broadcast
        node = builder.build("track.Pt()")
        
        assert not isinstance(node, MethodBroadcastNode)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
