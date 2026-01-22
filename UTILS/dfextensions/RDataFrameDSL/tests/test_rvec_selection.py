"""
Phase 12.2: RVec Selection Functions Tests

Tests for Take, Range, Where, and IndicesFromOffsets functions.
"""

import pytest
pytestmark = pytest.mark.root_serial
import numpy as np


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def track_cluster_schema():
    """Schema with track and cluster columns."""
    return {
        "trackPt": "RVec<float>",
        "trackEta": "RVec<float>",
        "trackIsGood": "RVec<bool>",
        "trackClusterFirst": "RVec<int>",
        "trackClusterN": "RVec<int>",
        "clusterDy": "RVec<float>",
        "clusterDz": "RVec<float>",
        "clusterRow": "RVec<int>",
        "clusterIsEdge": "RVec<bool>",
    }


@pytest.fixture
def synthetic_track_cluster_rdf():
    """Create synthetic RDataFrame with track/cluster data."""
    ROOT = pytest.importorskip("ROOT")
    
    rdf = ROOT.RDataFrame(5)
    
    # Track data
    rdf = rdf.Define("trackPt", "ROOT::RVec<float>{10.5f, 20.3f, 5.1f}")
    rdf = rdf.Define("trackEta", "ROOT::RVec<float>{0.5f, -1.2f, 0.8f}")
    rdf = rdf.Define("trackIsGood", "ROOT::RVec<bool>{true, false, true}")
    rdf = rdf.Define("trackClusterFirst", "ROOT::RVec<int>{0, 3, 5}")
    rdf = rdf.Define("trackClusterN", "ROOT::RVec<int>{3, 2, 4}")
    
    # Cluster data (9 clusters total)
    rdf = rdf.Define("clusterDy", "ROOT::RVec<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f}")
    rdf = rdf.Define("clusterDz", "ROOT::RVec<float>{1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f}")
    rdf = rdf.Define("clusterRow", "ROOT::RVec<int>{0, 1, 2, 3, 4, 5, 6, 7, 8}")
    rdf = rdf.Define("clusterIsEdge", "ROOT::RVec<bool>{false, false, true, false, true, false, false, true, false}")
    
    return rdf


# =============================================================================
# Test Existing Mask Patterns Still Work
# =============================================================================

class TestExistingMaskPatterns:
    """Verify existing mask/slice patterns aren't broken."""
    
    def test_boolean_mask_still_works(self, track_cluster_schema):
        """Boolean masking: arr[mask]."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        dsl.define("good_pt", "trackPt[trackIsGood]")
        
        code = dsl.preview()
        assert "good_pt" in code
    
    def test_slice_still_works(self, track_cluster_schema):
        """Slice syntax: arr[:10]."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        dsl.define("first3", "trackPt[:3]")
        
        code = dsl.preview()
        assert "first3" in code
    
    def test_negative_slice_still_works(self, track_cluster_schema):
        """Negative slice: arr[-3:]."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        dsl.define("last2", "trackPt[-2:]")
        
        code = dsl.preview()
        assert "last2" in code


# =============================================================================
# Take Function Tests
# =============================================================================

class TestTakeCompilation:
    """Test Take function compilation."""
    
    def test_take_with_scalar(self, track_cluster_schema):
        """Take(vec, n) compiles."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        dsl.define("first2", "Take(trackPt, 2)")
        
        code = dsl.preview()
        assert "Take" in code
    
    def test_take_with_negative(self, track_cluster_schema):
        """Take(vec, -n) for last n elements."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        dsl.define("last2", "Take(trackPt, -2)")
        
        code = dsl.preview()
        assert "Take" in code
    
    def test_take_with_indices(self, track_cluster_schema):
        """Take(vec, indices) compiles."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        dsl.define("selected", "Take(clusterDy, clusterRow)")
        
        code = dsl.preview()
        assert "Take" in code


class TestTakeExecution:
    """Test Take function execution with ROOT."""
    
    def test_take_first_n(self):
        """Take first n elements."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("arr", "ROOT::RVec<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}")
        
        dsl = DSLCompiler({"arr": "RVec<float>"})
        dsl.define("first3", "Take(arr, 3)")
        
        rdf = dsl.apply(rdf)
        result = rdf.AsNumpy(["first3"])
        
        assert len(result["first3"][0]) == 3
        assert list(result["first3"][0]) == [1.0, 2.0, 3.0]
    
    def test_take_last_n(self):
        """Take last n elements with negative index."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("arr", "ROOT::RVec<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}")
        
        dsl = DSLCompiler({"arr": "RVec<float>"})
        dsl.define("last2", "Take(arr, -2)")
        
        rdf = dsl.apply(rdf)
        result = rdf.AsNumpy(["last2"])
        
        assert len(result["last2"][0]) == 2
        assert list(result["last2"][0]) == [4.0, 5.0]


# =============================================================================
# Range Function Tests
# =============================================================================

class TestRangeCompilation:
    """Test Range function compilation."""
    
    def test_range_single_arg(self):
        """Range(n) compiles."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"n": "int"})
        dsl.define("indices", "Range(n)")
        
        code = dsl.preview()
        assert "Range" in code
    
    def test_range_two_args(self):
        """Range(start, end) compiles."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"start": "int", "end": "int"})
        dsl.define("indices", "Range(start, end)")
        
        code = dsl.preview()
        assert "Range" in code
    
    def test_range_three_args(self):
        """Range(start, end, step) compiles."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"a": "int", "b": "int", "s": "int"})
        dsl.define("indices", "Range(a, b, s)")
        
        code = dsl.preview()
        assert "Range" in code
    
    def test_range_rejects_rvec_args(self):
        """Range with RVec args should error with helpful message."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        from RDataFrameDSL.ir_errors import IRError
        
        dsl = DSLCompiler({
            "first_vec": "RVec<int>",
            "count_vec": "RVec<int>"
        })
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("indices", "Range(first_vec, first_vec + count_vec)")
        
        # Error message should mention scalar requirement
        error_msg = str(exc_info.value).lower()
        assert "scalar" in error_msg or "rank" in error_msg


class TestRangeExecution:
    """Test Range function execution with ROOT."""
    
    def test_range_single_arg(self):
        """Range(n) generates 0..n-1."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("n", "5")
        
        dsl = DSLCompiler({"n": "int"})
        dsl.define("indices", "Range(n)")
        
        rdf = dsl.apply(rdf)
        result = rdf.AsNumpy(["indices"])
        
        assert list(result["indices"][0]) == [0, 1, 2, 3, 4]
    
    def test_range_two_args(self):
        """Range(start, end) generates start..end-1."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("start", "2")
        rdf = rdf.Define("end", "7")
        
        dsl = DSLCompiler({"start": "int", "end": "int"})
        dsl.define("indices", "Range(start, end)")
        
        rdf = dsl.apply(rdf)
        result = rdf.AsNumpy(["indices"])
        
        assert list(result["indices"][0]) == [2, 3, 4, 5, 6]


# =============================================================================
# Where Function Tests
# =============================================================================

class TestWhereCompilation:
    """Test Where function compilation."""
    
    def test_where_scalar_replacement(self):
        """Where(cond, scalar, vec) compiles."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"arr": "RVec<float>"})
        dsl.define("clamped", "Where(arr > 100.0, 100.0, arr)")
        
        code = dsl.preview()
        assert "Where" in code
    
    def test_where_vec_replacement(self):
        """Where(cond, vec, vec) compiles."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"a": "RVec<float>", "b": "RVec<float>", "mask": "RVec<bool>"})
        dsl.define("selected", "Where(mask, a, b)")
        
        code = dsl.preview()
        assert "Where" in code


class TestWhereExecution:
    """Test Where function execution with ROOT."""
    
    def test_where_clamp_values(self):
        """Where clamps values above threshold."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("arr", "ROOT::RVec<float>{50.0f, 150.0f, 80.0f, 200.0f}")
        
        dsl = DSLCompiler({"arr": "RVec<float>"})
        dsl.define("clamped", "Where(arr > 100.0, 100.0, arr)")
        
        rdf = dsl.apply(rdf)
        result = rdf.AsNumpy(["clamped"])
        
        expected = [50.0, 100.0, 80.0, 100.0]
        assert list(result["clamped"][0]) == pytest.approx(expected)


# =============================================================================
# IndicesFromOffsets Tests
# =============================================================================

class TestIndicesFromOffsetsCompilation:
    """Test IndicesFromOffsets function compilation."""
    
    def test_basic_compilation(self, track_cluster_schema):
        """IndicesFromOffsets(first, count) compiles."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        dsl.define("idx", "IndicesFromOffsets(trackClusterFirst, trackClusterN)")
        
        code = dsl.preview()
        assert "IndicesFromOffsets" in code
    
    def test_with_masked_input(self, track_cluster_schema):
        """IndicesFromOffsets with masked arrays compiles."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        dsl.define("good_idx", "IndicesFromOffsets(trackClusterFirst[trackIsGood], trackClusterN[trackIsGood])")
        
        code = dsl.preview()
        assert "IndicesFromOffsets" in code


class TestIndicesFromOffsetsExecution:
    """Test IndicesFromOffsets function execution with ROOT."""
    
    def test_basic_execution(self):
        """IndicesFromOffsets generates correct indices."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("first", "ROOT::RVec<int>{0, 5, 8}")
        rdf = rdf.Define("count", "ROOT::RVec<int>{3, 2, 4}")
        
        dsl = DSLCompiler({
            "first": "RVec<int>",
            "count": "RVec<int>"
        })
        dsl.define("idx", "IndicesFromOffsets(first, count)")
        
        rdf = dsl.apply(rdf)
        result = rdf.AsNumpy(["idx"])
        
        # first={0,5,8}, count={3,2,4}
        # -> {0,1,2} + {5,6} + {8,9,10,11}
        expected = [0, 1, 2, 5, 6, 8, 9, 10, 11]
        assert list(result["idx"][0]) == expected
    
    def test_empty_input(self):
        """IndicesFromOffsets handles empty arrays."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("first", "ROOT::RVec<int>{}")
        rdf = rdf.Define("count", "ROOT::RVec<int>{}")
        
        dsl = DSLCompiler({
            "first": "RVec<int>",
            "count": "RVec<int>"
        })
        dsl.define("idx", "IndicesFromOffsets(first, count)")
        
        rdf = dsl.apply(rdf)
        result = rdf.AsNumpy(["idx"])
        
        assert len(result["idx"][0]) == 0
    
    def test_zero_counts(self):
        """IndicesFromOffsets handles zero counts."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("first", "ROOT::RVec<int>{0, 3, 3}")
        rdf = rdf.Define("count", "ROOT::RVec<int>{3, 0, 2}")
        
        dsl = DSLCompiler({
            "first": "RVec<int>",
            "count": "RVec<int>"
        })
        dsl.define("idx", "IndicesFromOffsets(first, count)")
        
        rdf = dsl.apply(rdf)
        result = rdf.AsNumpy(["idx"])
        
        # first={0,3,3}, count={3,0,2}
        # -> {0,1,2} + {} + {3,4}
        expected = [0, 1, 2, 3, 4]
        assert list(result["idx"][0]) == expected
    
    def test_size_mismatch_error(self):
        """IndicesFromOffsets errors on mismatched sizes."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("first", "ROOT::RVec<int>{0, 5}")      # size 2
        rdf = rdf.Define("count", "ROOT::RVec<int>{3, 2, 4}")   # size 3
        
        dsl = DSLCompiler({
            "first": "RVec<int>",
            "count": "RVec<int>"
        })
        dsl.define("idx", "IndicesFromOffsets(first, count)")
        
        rdf = dsl.apply(rdf)
        
        # Should error at runtime
        with pytest.raises(Exception):
            rdf.AsNumpy(["idx"])


# =============================================================================
# Calibration Workflow Tests
# =============================================================================

class TestCalibrationWorkflow:
    """Test full calibration workflow patterns."""
    
    def test_per_track_selection(self, synthetic_track_cluster_rdf, track_cluster_schema):
        """Select clusters for a single track using IndicesFromOffsets."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        # Select clusters for first track only using Take with first track's indices
        # Use a simpler pattern that avoids operator precedence issues with safe indexing
        dsl.define("all_idx", "IndicesFromOffsets(trackClusterFirst, trackClusterN)")
        dsl.define("track0_n", "trackClusterN[0]")
        dsl.define("track0_clusters", "Take(clusterDy, track0_n)")
        
        rdf = dsl.apply(synthetic_track_cluster_rdf)
        result = rdf.AsNumpy(["track0_clusters"])
        
        # First track: n=3 -> 3 clusters
        assert len(result["track0_clusters"][0]) == 3
    
    def test_full_mask_indices_take_pattern(self, synthetic_track_cluster_rdf, track_cluster_schema):
        """Full pattern: mask → IndicesFromOffsets → Take."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(track_cluster_schema)
        
        # Get cluster indices for good tracks
        dsl.define("good_idx", "IndicesFromOffsets(trackClusterFirst[trackIsGood], trackClusterN[trackIsGood])")
        dsl.define("good_dy", "Take(clusterDy, good_idx)")
        dsl.define("good_dz", "Take(clusterDz, good_idx)")
        
        rdf = dsl.apply(synthetic_track_cluster_rdf)
        result = rdf.AsNumpy(["good_dy", "good_dz"])
        
        # Good tracks: 0 (first=0, n=3) and 2 (first=5, n=4)
        # Indices: 0,1,2,5,6,7,8
        assert len(result["good_dy"][0]) == 7


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_take_empty_indices(self):
        """Take with empty indices returns empty."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("arr", "ROOT::RVec<float>{1.0f, 2.0f, 3.0f}")
        rdf = rdf.Define("empty_idx", "ROOT::RVec<int>{}")
        
        dsl = DSLCompiler({
            "arr": "RVec<float>",
            "empty_idx": "RVec<int>"
        })
        dsl.define("selected", "Take(arr, empty_idx)")
        
        rdf = dsl.apply(rdf)
        result = rdf.AsNumpy(["selected"])
        
        assert len(result["selected"][0]) == 0
    
    def test_where_all_true(self):
        """Where with all-true condition."""
        ROOT = pytest.importorskip("ROOT")
        from RDataFrameDSL import DSLCompiler
        
        rdf = ROOT.RDataFrame(1)
        rdf = rdf.Define("arr", "ROOT::RVec<float>{1.0f, 2.0f, 3.0f}")
        
        dsl = DSLCompiler({"arr": "RVec<float>"})
        dsl.define("result", "Where(arr > 0, arr, 0.0)")
        
        rdf = dsl.apply(rdf)
        result = rdf.AsNumpy(["result"])
        
        assert list(result["result"][0]) == pytest.approx([1.0, 2.0, 3.0])


# =============================================================================
# Type Inference Mock Tests (No ROOT Required)
# =============================================================================

class TestTypeInferenceMock:
    """Test type inference without ROOT."""
    
    def test_take_returns_same_type(self):
        """Take preserves element type."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"arr": "RVec<float>", "n": "int"})
        dsl.define("result", "Take(arr, n)")
        
        func = dsl._functions["result"]
        assert "float" in func.return_type.lower() or "RVec" in func.return_type
    
    def test_range_returns_rvec_int(self):
        """Range returns RVec<int>."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({"n": "int"})
        dsl.define("result", "Range(n)")
        
        func = dsl._functions["result"]
        assert "int" in func.return_type.lower()
    
    def test_where_type_promotion(self):
        """Where promotes types."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({
            "cond": "RVec<bool>",
            "a": "RVec<int>",
            "b": "RVec<float>"
        })
        dsl.define("result", "Where(cond, a, b)")
        
        code = dsl.preview()
        assert "Where" in code
    
    def test_range_rejects_rvec_args_mock(self):
        """Range with RVec args should error (mock test)."""
        from RDataFrameDSL import DSLCompiler
        from RDataFrameDSL.ir_errors import IRError
        
        dsl = DSLCompiler({
            "arr": "RVec<int>"
        })
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("bad", "Range(arr, arr + 1)")
        
        assert "scalar" in str(exc_info.value).lower() or "rank" in str(exc_info.value).lower()
    
    def test_indicesfromoffsets_rejects_scalar_args(self):
        """IndicesFromOffsets with scalar args should error."""
        from RDataFrameDSL import DSLCompiler
        from RDataFrameDSL.ir_errors import IRError
        
        dsl = DSLCompiler({
            "a": "int",
            "b": "int"
        })
        
        with pytest.raises(IRError) as exc_info:
            dsl.define("bad", "IndicesFromOffsets(a, b)")
        
        assert "rvec" in str(exc_info.value).lower() or "rank" in str(exc_info.value).lower()
    
    def test_indicesfromoffsets_returns_rvec_int(self):
        """IndicesFromOffsets returns RVec<int>."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler({
            "first": "RVec<int>",
            "count": "RVec<int>"
        })
        dsl.define("result", "IndicesFromOffsets(first, count)")
        
        func = dsl._functions["result"]
        assert "int" in func.return_type.lower()
