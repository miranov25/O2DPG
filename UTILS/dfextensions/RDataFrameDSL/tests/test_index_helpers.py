"""
Tests for Phase 13.7.B: Parent-Child Indexing.

Unit tests for index_helpers.py — registry, classification, expansion.
These tests use mocks and do NOT require ROOT.

Spec reference: PHASE_13_7_B_v1.2_Specification.md §7.1
"""

import pytest
from unittest.mock import MagicMock, call
from RDataFrameDSL.index_helpers import (
    ParentChildConfig,
    ParentChildRegistry,
    expand_parent_columns,
    _resolve_cpp_type,
)


# =============================================================================
# T1: register_parent_child() stores config correctly
# =============================================================================

class TestRegistration:
    
    def test_T1_stores_config(self):
        """T1: register stores config with correct fields."""
        reg = ParentChildRegistry()
        reg.register(parent="td.trk", child="res",
                     offset_column="trackInfo.idxFirstResidual")
        
        assert reg.has_registrations()
        result = reg.classify_column("td.trk.dEdxTPC")
        assert result is not None
        level, config = result
        assert level == 'parent'
        assert config.parent_prefix == "td.trk"
        assert config.child_prefix == "res"
        assert config.offset_column == "trackInfo.idxFirstResidual"
    
    def test_T2_missing_offset_column(self):
        """T2: Missing offset_column raises ValueError."""
        reg = ParentChildRegistry()
        with pytest.raises(ValueError, match="Must provide offset_column"):
            reg.register(parent="td.trk", child="res", offset_column="")
    
    def test_T2b_missing_parent(self):
        """T2b: Missing parent raises ValueError."""
        reg = ParentChildRegistry()
        with pytest.raises(ValueError, match="parent prefix must be non-empty"):
            reg.register(parent="", child="res",
                         offset_column="trackInfo.idxFirstResidual")
    
    def test_T2c_missing_child(self):
        """T2c: Missing child raises ValueError."""
        reg = ParentChildRegistry()
        with pytest.raises(ValueError, match="child prefix must be non-empty"):
            reg.register(parent="td.trk", child="",
                         offset_column="trackInfo.idxFirstResidual")
    
    def test_T3_duplicate_child(self):
        """T3: Duplicate child prefix raises ValueError."""
        reg = ParentChildRegistry()
        reg.register(parent="td.trk", child="res",
                     offset_column="trackInfo.idxFirstResidual")
        
        with pytest.raises(ValueError, match="Child 'res' already registered"):
            reg.register(parent="other", child="res",
                         offset_column="otherInfo.offset")
    
    def test_T4_offset_prefix_overlaps_parent(self):
        """T4: offset_column sharing parent prefix raises ValueError."""
        reg = ParentChildRegistry()
        with pytest.raises(ValueError, match="shares prefix with parent"):
            reg.register(parent="td.trk", child="res",
                         offset_column="td.trk.someOffset")
    
    def test_T4b_offset_prefix_overlaps_child(self):
        """T4b: offset_column sharing child prefix raises ValueError."""
        reg = ParentChildRegistry()
        with pytest.raises(ValueError, match="shares prefix with child"):
            reg.register(parent="td.trk", child="res",
                         offset_column="res.someOffset")


# =============================================================================
# T5-T7: Column classification
# =============================================================================

class TestClassification:
    
    @pytest.fixture
    def registry(self):
        reg = ParentChildRegistry()
        reg.register(parent="td.trk", child="res",
                     offset_column="trackInfo.idxFirstResidual")
        return reg
    
    def test_T5_classify_parent(self, registry):
        """T5: classify_column returns ('parent', config) for parent column."""
        result = registry.classify_column("td.trk.dEdxTPC")
        assert result is not None
        level, config = result
        assert level == 'parent'
        assert config.parent_prefix == "td.trk"
    
    def test_T5b_classify_parent_deep(self, registry):
        """T5b: Classify parent with deeper nesting."""
        result = registry.classify_column("td.trk.chi2TPC")
        assert result is not None
        assert result[0] == 'parent'
    
    def test_T6_classify_child(self, registry):
        """T6: classify_column returns ('child', config) for child column."""
        result = registry.classify_column("res.dy")
        assert result is not None
        level, config = result
        assert level == 'child'
        assert config.child_prefix == "res"
    
    def test_T7_classify_unregistered(self, registry):
        """T7: classify_column returns None for unregistered column."""
        assert registry.classify_column("event.timestamp") is None
    
    def test_T7b_classify_offset_column(self, registry):
        """T7b: Offset column is unregistered (different prefix)."""
        assert registry.classify_column("trackInfo.idxFirstResidual") is None
    
    def test_T12_no_dot_column(self, registry):
        """T12: Column without dot → unregistered."""
        assert registry.classify_column("multiplicity") is None
    
    def test_T12b_exact_prefix_not_matched(self, registry):
        """Exact prefix without trailing content doesn't match."""
        # "td.trk" alone (no dot + suffix) should not match
        assert registry.classify_column("td.trk") is None
    
    def test_prefix_substring_not_matched(self, registry):
        """Prefix must be followed by dot, not just substring."""
        # "td.trkExtra.foo" should NOT match "td.trk." prefix
        assert registry.classify_column("td.trkExtra.foo") is None


# =============================================================================
# T8-T9: Expansion trigger logic
# =============================================================================

class TestExpansionTrigger:
    
    @pytest.fixture
    def registry(self):
        reg = ParentChildRegistry()
        reg.register(parent="td.trk", child="res",
                     offset_column="trackInfo.idxFirstResidual")
        return reg
    
    def test_T8_no_expansion_child_only(self, registry):
        """T8: No expansion when only child columns requested."""
        info = registry.get_expansion_info(["res.dy", "res.dz"])
        assert info is None
    
    def test_T9_no_expansion_parent_only(self, registry):
        """T9: No expansion when only parent columns requested."""
        info = registry.get_expansion_info(["td.trk.dEdxTPC", "td.trk.chi2TPC"])
        assert info is None
    
    def test_expansion_mixed_levels(self, registry):
        """Expansion triggers when parent + child columns present."""
        info = registry.get_expansion_info(["res.dy", "td.trk.dEdxTPC"])
        assert info is not None
        assert info['parent_columns'] == ["td.trk.dEdxTPC"]
        assert info['child_columns'] == ["res.dy"]
        assert info['child_size_column'] == "res.dy"
    
    def test_expansion_multiple_parents(self, registry):
        """Multiple parent columns all detected."""
        info = registry.get_expansion_info(
            ["res.dy", "td.trk.dEdxTPC", "td.trk.chi2TPC"]
        )
        assert info is not None
        assert set(info['parent_columns']) == {"td.trk.dEdxTPC", "td.trk.chi2TPC"}
    
    def test_expansion_child_size_is_first_child(self, registry):
        """child_size_column is the first child column in the list."""
        info = registry.get_expansion_info(
            ["res.dz", "res.dy", "td.trk.dEdxTPC"]
        )
        assert info is not None
        assert info['child_size_column'] == "res.dz"  # first in list
    
    def test_no_expansion_unregistered(self, registry):
        """No expansion for unregistered columns."""
        info = registry.get_expansion_info(["event.timestamp", "other.col"])
        assert info is None
    
    def test_no_expansion_empty_registry(self):
        """No expansion when registry is empty."""
        reg = ParentChildRegistry()
        info = reg.get_expansion_info(["res.dy", "td.trk.dEdxTPC"])
        assert info is None


# =============================================================================
# T10: Expansion generates correct Define expression
# =============================================================================

class TestExpandParentColumns:
    
    def test_T10_correct_define(self):
        """T10: expand_parent_columns injects Define with correct expression."""
        reg = ParentChildRegistry()
        reg.register(parent="td.trk", child="res",
                     offset_column="trackInfo.idxFirstResidual")
        
        mock_rdf = MagicMock()
        mock_rdf.Define.return_value = mock_rdf
        
        schema = {"td.trk.dEdxTPC": "RVec<float>"}
        
        rdf_out, rename_map = expand_parent_columns(
            mock_rdf, reg,
            ["res.dy", "td.trk.dEdxTPC"],
            schema
        )
        
        # Verify Define was called with correct arguments
        mock_rdf.Define.assert_called_once()
        call_args = mock_rdf.Define.call_args
        expanded_name = call_args[0][0]
        cpp_expr = call_args[0][1]
        
        assert expanded_name == "__pc_expanded_td_trk_dEdxTPC"
        assert "ExpandToChildrenFromOffsets<float>" in cpp_expr
        assert "td.trk.dEdxTPC" in cpp_expr
        assert "trackInfo.idxFirstResidual" in cpp_expr
        assert "(int)res.dy.size()" in cpp_expr
    
    def test_T10b_rename_map(self):
        """T10b: rename_map maps expanded name → original name."""
        reg = ParentChildRegistry()
        reg.register(parent="td.trk", child="res",
                     offset_column="trackInfo.idxFirstResidual")
        
        mock_rdf = MagicMock()
        mock_rdf.Define.return_value = mock_rdf
        
        _, rename_map = expand_parent_columns(
            mock_rdf, reg,
            ["res.dy", "td.trk.dEdxTPC"],
            {"td.trk.dEdxTPC": "RVec<float>"}
        )
        
        assert rename_map == {
            "__pc_expanded_td_trk_dEdxTPC": "td.trk.dEdxTPC"
        }
    
    def test_T10c_multiple_parent_columns(self):
        """Multiple parent columns each get their own Define."""
        reg = ParentChildRegistry()
        reg.register(parent="td.trk", child="res",
                     offset_column="trackInfo.idxFirstResidual")
        
        mock_rdf = MagicMock()
        mock_rdf.Define.return_value = mock_rdf
        
        schema = {
            "td.trk.dEdxTPC": "RVec<float>",
            "td.trk.chi2TPC": "RVec<double>",
        }
        
        _, rename_map = expand_parent_columns(
            mock_rdf, reg,
            ["res.dy", "td.trk.dEdxTPC", "td.trk.chi2TPC"],
            schema
        )
        
        assert mock_rdf.Define.call_count == 2
        assert len(rename_map) == 2
        assert "__pc_expanded_td_trk_dEdxTPC" in rename_map
        assert "__pc_expanded_td_trk_chi2TPC" in rename_map
    
    def test_no_expansion_returns_same_rdf(self):
        """No expansion returns original rdf and empty rename_map."""
        reg = ParentChildRegistry()
        reg.register(parent="td.trk", child="res",
                     offset_column="trackInfo.idxFirstResidual")
        
        mock_rdf = MagicMock()
        
        rdf_out, rename_map = expand_parent_columns(
            mock_rdf, reg,
            ["res.dy", "res.dz"],  # child only
            {}
        )
        
        assert rdf_out is mock_rdf
        assert rename_map == {}
        mock_rdf.Define.assert_not_called()


# =============================================================================
# T11: Multiple relationships
# =============================================================================

class TestMultipleRelationships:
    
    def test_T11_same_parent_different_children(self):
        """T11: Same parent, different children — both register."""
        reg = ParentChildRegistry()
        reg.register(parent="td.trk", child="res",
                     offset_column="trackInfo.idxFirstResidual")
        reg.register(parent="td.trk", child="cluster",
                     offset_column="trackInfo.idxFirstCluster")
        
        assert len(reg._relations) == 2
        
        # res column classified correctly
        result = reg.classify_column("res.dy")
        assert result[0] == 'child'
        assert result[1].child_prefix == "res"
        
        # cluster column classified correctly
        result = reg.classify_column("cluster.x")
        assert result[0] == 'child'
        assert result[1].child_prefix == "cluster"
    
    def test_expansion_selects_correct_relationship(self):
        """Expansion uses the relationship where both parent+child match."""
        reg = ParentChildRegistry()
        reg.register(parent="td.trk", child="res",
                     offset_column="trackInfo.idxFirstResidual")
        reg.register(parent="td.trk", child="cluster",
                     offset_column="trackInfo.idxFirstCluster")
        
        # Request res + parent → uses res relationship
        info = reg.get_expansion_info(["res.dy", "td.trk.dEdxTPC"])
        assert info is not None
        assert info['config'].child_prefix == "res"
        assert info['config'].offset_column == "trackInfo.idxFirstResidual"
        
        # Request cluster + parent → uses cluster relationship
        info = reg.get_expansion_info(["cluster.x", "td.trk.dEdxTPC"])
        assert info is not None
        assert info['config'].child_prefix == "cluster"
        assert info['config'].offset_column == "trackInfo.idxFirstCluster"


# =============================================================================
# Type Resolution
# =============================================================================

class TestTypeResolution:
    
    def test_rvec_float(self):
        assert _resolve_cpp_type("col", {"col": "RVec<float>"}) == "float"
    
    def test_rvec_double(self):
        assert _resolve_cpp_type("col", {"col": "RVec<double>"}) == "double"
    
    def test_rvec_short(self):
        assert _resolve_cpp_type("col", {"col": "RVec<short>"}) == "short"
    
    def test_rvec_int(self):
        assert _resolve_cpp_type("col", {"col": "RVec<int>"}) == "int"
    
    def test_full_rvec_path(self):
        assert _resolve_cpp_type(
            "col", {"col": "ROOT::VecOps::RVec<float>"}
        ) == "float"
    
    def test_bare_type(self):
        assert _resolve_cpp_type("col", {"col": "float"}) == "float"
    
    def test_not_in_schema_defaults_float(self):
        assert _resolve_cpp_type("missing", {}) == "float"
    
    def test_Float_t(self):
        assert _resolve_cpp_type("col", {"col": "RVec<Float_t>"}) == "float"


# =============================================================================
# ParentChildConfig dataclass
# =============================================================================

class TestParentChildConfig:
    
    def test_creation(self):
        config = ParentChildConfig(
            parent_prefix="td.trk",
            child_prefix="res",
            offset_column="trackInfo.idxFirstResidual"
        )
        assert config.parent_prefix == "td.trk"
        assert config.child_prefix == "res"
        assert config.offset_column == "trackInfo.idxFirstResidual"
    
    def test_equality(self):
        c1 = ParentChildConfig("td.trk", "res", "trackInfo.idxFirstResidual")
        c2 = ParentChildConfig("td.trk", "res", "trackInfo.idxFirstResidual")
        assert c1 == c2
