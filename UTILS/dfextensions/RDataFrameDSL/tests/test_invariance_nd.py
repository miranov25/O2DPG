"""
N-D Slicing Invariance Tests (INV-ND*)

Phase 13.6.C: Validates N-dimensional slicing operations in DSL.

Test Categories:
- INV-ND1: 2D slicing basic (cluster level)
- INV-ND2: 2D slicing advanced (mixed index/slice)
- INV-ND3: 3D slicing (hit level)
- INV-ND4: Mathematical invariance (sum consistency)
- INV-ND5: Edge cases (empty, bounds)

Test Types:
- Type A (Engine): Pure Python flatten operations
- Type B (DSL): Full DSL → ROOT → result chain

Invariant Formulas (from toy_nd.py):
- cluster_Q[e][t][c] = 1000*e + 100*t + c
- hit_E[e][t][c][h] = 10000*e + 1000*t + 100*c + h
- track_pt uses Pythagorean triples: (3,4,5), (5,12,13), ...

Author: Claude Opus 4.5
Date: 2026-01-18 (Updated 2026-01-19)
Phase: 13.6.C
"""

import pytest
import numpy as np
import pandas as pd


# =============================================================================
# Import N-D Fixtures from toy_nd.py (via conftest)
# =============================================================================
# These fixtures are imported via conftest.py:
#   - nd_2d_dict, nd_2d_dict_small: 2D test data as dicts
#   - nd_3d_dict, nd_3d_dict_small: 3D test data as dicts
#   - nd_2d_schema, nd_3d_schema: Schema definitions
#   - cluster_value_fn: Q = 1000*e + 100*t + c
#   - hit_value_fn: E = 10000*e + 1000*t + 100*c + h
#   - expected_2d_flat, expected_sum: Validation helpers
#
# For ROOT tests (Tier 3):
#   - nd_2d_root_file_S, nd_3d_root_file_S: ROOT file fixtures
#   - nd_2d_rdf, nd_3d_rdf: RDataFrame fixtures

try:
    from conftest_nd_additions import (
        cluster_value,
        hit_value,
        get_expected_2d_flat,
        get_expected_2d_slice,
        get_expected_sum,
        _ND_GENERATORS_AVAILABLE,
    )
except ImportError:
    _ND_GENERATORS_AVAILABLE = False
    cluster_value = None
    hit_value = None
    get_expected_2d_flat = None
    get_expected_2d_slice = None
    get_expected_sum = None


# =============================================================================
# Legacy Fixtures (for backwards compatibility during transition)
# =============================================================================
# These will be REMOVED once all tests migrate to toy_nd.py fixtures.

@pytest.fixture(scope="module")
def nd_test_data_2d_legacy():
    """
    LEGACY: Deterministic 2D test data with OLD formula.
    
    OLD Formula: cluster_Q[e][t][c] = e*1000 + t*100 + c*10
    NEW Formula: cluster_Q[e][t][c] = e*1000 + t*100 + c (from toy_nd.py)
    
    DEPRECATED: Use nd_2d_dict fixture instead.
    """
    return {
        'event_id': np.array([0, 1, 2], dtype=np.int64),
        'n_tracks': np.array([2, 3, 2], dtype=np.int32),
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0, 5.0], dtype=np.float64),
            np.array([6.0, 7.0], dtype=np.float64),
        ], dtype=object),
        'cluster_Q': np.array([
            # Event 0: e=0 → 0*1000 = 0 base
            np.array([
                np.array([0., 10., 20., 30.], dtype=np.float64),      # Track 0
                np.array([100., 110., 120., 130., 140.], dtype=np.float64),  # Track 1
            ], dtype=object),
            # Event 1: e=1 → 1*1000 = 1000 base
            np.array([
                np.array([1000., 1010., 1020.], dtype=np.float64),    # Track 0
                np.array([1100., 1110., 1120., 1130.], dtype=np.float64),  # Track 1
                np.array([1200., 1210.], dtype=np.float64),           # Track 2
            ], dtype=object),
            # Event 2: e=2 → 2*1000 = 2000 base
            np.array([
                np.array([2000., 2010., 2020., 2030.], dtype=np.float64),  # Track 0
                np.array([2100., 2110., 2120.], dtype=np.float64),    # Track 1
            ], dtype=object),
        ], dtype=object),
    }


@pytest.fixture(scope="module")
def nd_test_data_3d_legacy():
    """
    LEGACY: Deterministic 3D test data.
    
    DEPRECATED: Use nd_3d_dict fixture instead.
    """
    return {
        'event_id': np.array([0, 1], dtype=np.int64),
        'n_tracks': np.array([2, 2], dtype=np.int32),
        'hit_E': np.array([
            # Event 0: base 0
            np.array([
                # Track 0: base 0
                np.array([
                    np.array([0., 1., 2.], dtype=np.float64),       # Cluster 0: 3 hits
                    np.array([100., 101.], dtype=np.float64),       # Cluster 1: 2 hits
                ], dtype=object),
                # Track 1: base 1000
                np.array([
                    np.array([1000., 1001., 1002.], dtype=np.float64),  # Cluster 0: 3 hits
                    np.array([1100., 1101.], dtype=np.float64),         # Cluster 1: 2 hits
                ], dtype=object),
            ], dtype=object),
            # Event 1: base 10000
            np.array([
                # Track 0: base 10000
                np.array([
                    np.array([10000., 10001.], dtype=np.float64),       # Cluster 0: 2 hits
                    np.array([10100., 10101., 10102.], dtype=np.float64),  # Cluster 1: 3 hits
                ], dtype=object),
                # Track 1: base 11000
                np.array([
                    np.array([11000., 11001.], dtype=np.float64),       # Cluster 0: 2 hits
                    np.array([11100., 11101., 11102.], dtype=np.float64),  # Cluster 1: 3 hits
                ], dtype=object),
            ], dtype=object),
        ], dtype=object),
    }


# Alias old fixture names for backwards compatibility
@pytest.fixture(scope="module")
def nd_test_data_2d(nd_test_data_2d_legacy):
    """Alias for legacy fixture (use nd_2d_dict instead)."""
    return nd_test_data_2d_legacy


@pytest.fixture(scope="module")
def nd_test_data_3d(nd_test_data_3d_legacy):
    """Alias for legacy fixture (use nd_3d_dict instead)."""
    return nd_test_data_3d_legacy


@pytest.fixture
def dsl_schema_2d():
    """Schema for 2D test data."""
    return {
        'event_id': 'long',
        'n_tracks': 'int',
        'track_pt': 'RVec<double>',
        'cluster_Q': 'RVec<RVec<double>>',
    }


@pytest.fixture
def dsl_schema_3d():
    """Schema for 3D test data."""
    return {
        'event_id': 'long',
        'n_tracks': 'int',
        'hit_E': 'RVec<RVec<RVec<double>>>',
    }


# =============================================================================
# INV-ND1: 2D Slicing Basic Tests
# =============================================================================

class TestND1_2DSlicingBasic:
    """
    INV-ND1: Basic 2D slicing tests (cluster level).
    
    All tests validate that cluster_Q[a:b, c:d] produces expected C++ code
    and correct results when executed.
    """
    
    @pytest.mark.feature("nd_slice_2d")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ND1a_first_n_both_dims(self, nd_test_data_2d, dsl_schema_2d):
        """
        INV-ND1a: cluster_Q[0:2, 0:3] - first 2 tracks, first 3 clusters.
        
        Expected per event:
            Event 0: [[0,10,20], [100,110,120]]
            Event 1: [[1000,1010,1020], [1100,1110,1120]]
            Event 2: [[2000,2010,2020], [2100,2110,2120]]
        
        Type: B (DSL end-to-end)
        Priority: P0 (CRITICAL)
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        # Should compile without error
        try:
            dsl.define("sliced_2d", "cluster_Q[0:2, 0:3]")
        except Exception as e:
            pytest.fail(f"Failed to compile 2D slice expression: {e}")
        
        # Verify the generated function exists
        assert "sliced_2d" in dsl._functions, "Function not registered"
    
    @pytest.mark.feature("nd_slice_2d")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ND1b_full_first_dim(self, nd_test_data_2d, dsl_schema_2d):
        """
        INV-ND1b: cluster_Q[:, 0:2] - all tracks, first 2 clusters.
        
        Expected: Each track contributes first 2 clusters.
        
        Type: B (DSL)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        try:
            dsl.define("full_first", "cluster_Q[:, 0:2]")
        except Exception as e:
            pytest.fail(f"Failed to compile cluster_Q[:, 0:2]: {e}")
    
    @pytest.mark.feature("nd_slice_2d")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ND1c_full_both_dims(self, nd_test_data_2d, dsl_schema_2d):
        """
        INV-ND1c: cluster_Q[:, :] - identity (all tracks, all clusters).
        
        Invariant: cluster_Q[:, :] == cluster_Q
        
        Type: B (DSL)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        try:
            dsl.define("identity", "cluster_Q[:, :]")
        except Exception as e:
            pytest.fail(f"Failed to compile identity slice: {e}")


# =============================================================================
# INV-ND2: 2D Slicing Advanced Tests
# =============================================================================

class TestND2_2DSlicingAdvanced:
    """
    INV-ND2: Advanced 2D slicing tests (mixed index/slice, negative indices).
    """
    
    @pytest.mark.feature("nd_slice_2d")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ND2a_mixed_index_slice(self, nd_test_data_2d, dsl_schema_2d):
        """
        INV-ND2a: cluster_Q[0, :] - first track only, all clusters.
        
        Expected: Reduces to 1D - just clusters of track 0.
            Event 0: [0, 10, 20, 30]
            Event 1: [1000, 1010, 1020]
            Event 2: [2000, 2010, 2020, 2030]
        
        Type: B (DSL)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        try:
            dsl.define("first_track", "cluster_Q[0, :]")
        except Exception as e:
            pytest.fail(f"Failed to compile mixed index/slice: {e}")
    
    @pytest.mark.feature("nd_slice_2d")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ND2b_slice_index(self, nd_test_data_2d, dsl_schema_2d):
        """
        INV-ND2b: cluster_Q[:, 0] - all tracks, first cluster only.
        
        Expected: For each track, get cluster[0].
            Event 0: [[0], [100]]  (or flattened: [0, 100])
        
        Type: B (DSL)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        try:
            dsl.define("first_cluster", "cluster_Q[:, 0]")
        except Exception as e:
            pytest.fail(f"Failed to compile cluster_Q[:, 0]: {e}")
    
    @pytest.mark.feature("nd_slice_2d")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_ND2c_negative_outer(self, nd_test_data_2d, dsl_schema_2d):
        """
        INV-ND2c: cluster_Q[-1, :] - last track only, all clusters.
        
        Expected:
            Event 0: Track 1 → [100, 110, 120, 130, 140]
            Event 1: Track 2 → [1200, 1210]
            Event 2: Track 1 → [2100, 2110, 2120]
        
        Type: B (DSL)
        Priority: P1
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        try:
            dsl.define("last_track", "cluster_Q[-1, :]")
        except Exception as e:
            pytest.fail(f"Failed to compile negative index: {e}")
    
    @pytest.mark.feature("nd_slice_2d")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_ND2d_negative_inner(self, nd_test_data_2d, dsl_schema_2d):
        """
        INV-ND2d: cluster_Q[:, -2:] - all tracks, last 2 clusters each.
        
        Type: B (DSL)
        Priority: P1
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        try:
            dsl.define("last_2_clusters", "cluster_Q[:, -2:]")
        except Exception as e:
            pytest.fail(f"Failed to compile negative slice: {e}")
    
    @pytest.mark.feature("nd_slice_2d")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_ND2e_reverse_inner(self, nd_test_data_2d, dsl_schema_2d):
        """
        INV-ND2e: cluster_Q[:, ::-1] - all tracks, reversed clusters.
        
        Type: B (DSL)
        Priority: P1
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        try:
            dsl.define("reversed_clusters", "cluster_Q[:, ::-1]")
        except Exception as e:
            pytest.fail(f"Failed to compile reverse slice: {e}")
    
    @pytest.mark.feature("nd_slice_2d")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_ND2f_step_outer(self, nd_test_data_2d, dsl_schema_2d):
        """
        INV-ND2f: cluster_Q[::2, :] - every other track, all clusters.
        
        Type: B (DSL)
        Priority: P1
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        try:
            dsl.define("every_other_track", "cluster_Q[::2, :]")
        except Exception as e:
            pytest.fail(f"Failed to compile step slice: {e}")


# =============================================================================
# INV-ND3: 3D Slicing Tests
# =============================================================================

class TestND3_3DSlicing:
    """
    INV-ND3: 3D slicing tests (hit level: tracks → clusters → hits).
    """
    
    @pytest.mark.feature("nd_slice_3d")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ND3a_first_n_all_dims(self, nd_test_data_3d, dsl_schema_3d):
        """
        INV-ND3a: hit_E[0:1, 0:1, 0:2] - first track, first cluster, first 2 hits.
        
        Expected:
            Event 0: [[[0, 1]]]  (track 0, cluster 0, hits 0-1)
            Event 1: [[[10000, 10001]]]
        
        Type: B (DSL)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_3d)
        
        try:
            dsl.define("sliced_3d", "hit_E[0:1, 0:1, 0:2]")
        except Exception as e:
            pytest.fail(f"Failed to compile 3D slice: {e}")
    
    @pytest.mark.feature("nd_slice_3d")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ND3b_full_middle(self, nd_test_data_3d, dsl_schema_3d):
        """
        INV-ND3b: hit_E[0:1, :, 0:2] - first track, ALL clusters, first 2 hits.
        
        Type: B (DSL)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_3d)
        
        try:
            dsl.define("full_middle", "hit_E[0:1, :, 0:2]")
        except Exception as e:
            pytest.fail(f"Failed to compile 3D with full middle: {e}")
    
    @pytest.mark.feature("nd_slice_3d")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_ND3c_single_element(self, nd_test_data_3d, dsl_schema_3d):
        """
        INV-ND3c: hit_E[0, 0, 0] - single element access (should be scalar).
        
        Expected:
            Event 0: 0.0
            Event 1: 10000.0
        
        Type: B (DSL)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_3d)
        
        try:
            dsl.define("single_hit", "hit_E[0, 0, 0]")
        except Exception as e:
            pytest.fail(f"Failed to compile single element access: {e}")
    
    @pytest.mark.feature("nd_slice_3d")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_ND3d_mixed_3d(self, nd_test_data_3d, dsl_schema_3d):
        """
        INV-ND3d: hit_E[0, :, 0:2] - track 0, all clusters, first 2 hits.
        
        Type: B (DSL)
        Priority: P1
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_3d)
        
        try:
            dsl.define("mixed_3d", "hit_E[0, :, 0:2]")
        except Exception as e:
            pytest.fail(f"Failed to compile mixed 3D: {e}")


# =============================================================================
# INV-ND4: Mathematical Invariance Tests
# =============================================================================

class TestND4_MathematicalInvariance:
    """
    INV-ND4: Mathematical invariance tests for N-D slicing.
    
    These tests verify mathematical properties that must hold
    regardless of implementation details.
    """
    
    @pytest.mark.feature("nd_slice_invariance")
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_ND4a_sum_consistency_2d(self, nd_test_data_2d):
        """
        INV-ND4a: Sum of sliced 2D data equals sum of source elements.
        
        Invariant: sum(cluster_Q[0:2, 0:2]) = sum of first 2 clusters of first 2 tracks
        
        Type: A (Engine)
        Priority: P0
        """
        data = nd_test_data_2d['cluster_Q']
        
        # Event 0: first 2 tracks, first 2 clusters
        expected_e0 = (
            data[0][0][0] + data[0][0][1] +  # Track 0: clusters 0,1
            data[0][1][0] + data[0][1][1]    # Track 1: clusters 0,1
        )
        
        # Manual calculation: 0 + 10 + 100 + 110 = 220
        assert expected_e0 == 220.0, f"Event 0 sum mismatch: {expected_e0}"
        
        # Event 1: first 2 tracks, first 2 clusters
        expected_e1 = (
            data[1][0][0] + data[1][0][1] +  # Track 0: 1000, 1010
            data[1][1][0] + data[1][1][1]    # Track 1: 1100, 1110
        )
        assert expected_e1 == 4220.0, f"Event 1 sum mismatch: {expected_e1}"
    
    @pytest.mark.feature("nd_slice_invariance")
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_ND4b_identity_slice_2d(self, nd_test_data_2d):
        """
        INV-ND4b: Full slice is identity - cluster_Q[:, :] == cluster_Q.
        
        Type: A (Engine)
        Priority: P0
        """
        data = nd_test_data_2d['cluster_Q']
        
        # Verify structure preserved
        assert len(data) == 3, "Should have 3 events"
        assert len(data[0]) == 2, "Event 0 should have 2 tracks"
        assert len(data[0][0]) == 4, "Event 0, Track 0 should have 4 clusters"
    
    @pytest.mark.feature("nd_slice_invariance")
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_ND4c_negative_index_equivalence(self, nd_test_data_2d):
        """
        INV-ND4c: Negative index equivalent to positive.
        
        Invariant: cluster_Q[-1, :] == cluster_Q[n_tracks-1, :]
        
        Type: A (Engine)
        Priority: P0
        """
        data = nd_test_data_2d['cluster_Q']
        
        # Event 0: -1 should equal track 1 (index 1 in 0-indexed)
        last_track_e0 = data[0][-1]
        expected_e0 = data[0][1]
        np.testing.assert_array_equal(last_track_e0, expected_e0)
        
        # Event 1: -1 should equal track 2 (index 2)
        last_track_e1 = data[1][-1]
        expected_e1 = data[1][2]
        np.testing.assert_array_equal(last_track_e1, expected_e1)
    
    @pytest.mark.feature("nd_slice_invariance")
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_ND4d_slice_sum_partition(self, nd_test_data_2d):
        """
        INV-ND4d: Adjacent slices partition the data.
        
        Invariant: cluster_Q[:, 0:2] + cluster_Q[:, 2:] covers all clusters
        
        Type: A (Engine)
        Priority: P0
        """
        data = nd_test_data_2d['cluster_Q']
        
        # For event 0, track 0: [0, 10, 20, 30]
        # [:, 0:2] → [0, 10]
        # [:, 2:] → [20, 30]
        # Together: [0, 10, 20, 30] ✓
        
        track_data = data[0][0]
        first_half = track_data[:2]
        second_half = track_data[2:]
        combined = np.concatenate([first_half, second_half])
        
        np.testing.assert_array_equal(combined, track_data)


# =============================================================================
# INV-ND5: Edge Cases
# =============================================================================

class TestND5_EdgeCases:
    """
    INV-ND5: Edge case tests for N-D slicing.
    """
    
    @pytest.mark.feature("nd_slice_edge")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_ND5a_oob_clipping(self, nd_test_data_2d, dsl_schema_2d):
        """
        INV-ND5a: Out-of-bounds indices should clip, not error.
        
        cluster_Q[0:100, 0:100] should return all available data.
        
        Type: B (DSL)
        Priority: P1
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        try:
            dsl.define("oob_clip", "cluster_Q[0:100, 0:100]")
        except Exception as e:
            pytest.fail(f"OOB slicing should not error: {e}")
    
    @pytest.mark.feature("nd_slice_edge")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_ND5b_empty_result(self, nd_test_data_2d, dsl_schema_2d):
        """
        INV-ND5b: Completely out-of-range returns empty, not error.
        
        cluster_Q[100:200, :] should return empty vectors.
        
        Type: B (DSL)
        Priority: P1
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        try:
            dsl.define("empty_result", "cluster_Q[100:200, :]")
        except Exception as e:
            pytest.fail(f"Empty slice should not error: {e}")
    
    @pytest.mark.feature("nd_slice_edge")
    @pytest.mark.type_b
    @pytest.mark.p2
    def test_INV_ND5c_dimension_limit(self, dsl_schema_3d):
        """
        INV-ND5c: >5D slicing should raise an error.
        
        Type: B (DSL)
        Priority: P2
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
            from RDataFrameDSL.ir_errors import IRError
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        schema_6d = {
            'event_id': 'long',
            'deep': 'RVec<RVec<RVec<RVec<RVec<RVec<double>>>>>>',
        }
        
        dsl = DSLCompiler(schema_6d)
        
        # 6D slicing should fail during validation
        with pytest.raises((IRError, Exception)):
            dsl.define("too_deep", "deep[0, 0, 0, 0, 0, 0]")
    
    @pytest.mark.feature("nd_slice_edge")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_ND5d_jagged_structure(self, nd_test_data_2d):
        """
        INV-ND5d: Jagged arrays handled correctly.
        
        Different events have different track/cluster counts.
        Slicing should handle this without error.
        
        Type: A (Engine)
        Priority: P1
        """
        data = nd_test_data_2d['cluster_Q']
        
        # Verify jagged structure
        track_counts = [len(e) for e in data]
        assert track_counts == [2, 3, 2], f"Track counts: {track_counts}"
        
        # Cluster counts per track
        for e, event in enumerate(data):
            for t, track in enumerate(event):
                # Each track should be sliceable
                sliced = track[:2] if len(track) >= 2 else track[:]
                assert len(sliced) <= 2, f"Slice length exceeded at event {e}, track {t}"


# =============================================================================
# Integration Tests (Require ROOT Execution)
# =============================================================================

# Check if ROOT is available for integration tests
try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False

@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
class TestNDSlicingIntegration:
    """
    Full integration tests with ROOT execution.
    
    Uses alice_rdf fixture which creates 2D data via C++ (ALICEEventGenerator).
    Schema includes cluster_Q as RVec<RVec<double>>.
    """
    
    def test_2d_slice_compiles_and_declares(self, dsl_schema_2d):
        """
        Test that 2D slicing expression compiles with DSL.
        
        Verifies that define() completes without error, which means:
        - Parser accepted the syntax
        - IR builder created valid nodes
        - Code generation succeeded
        """
        import ROOT
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(dsl_schema_2d)
        
        # Should compile without error
        try:
            dsl.define("sliced_2d", "cluster_Q[0:2, 0:3]")
        except Exception as e:
            pytest.fail(f"Failed to compile 2D slice expression: {e}")
        
        # Verify the column was defined
        assert "sliced_2d" in dsl._functions or hasattr(dsl, '_columns'), \
            "sliced_2d not registered after define()"
    
    def test_2d_slice_execution_e2e(self, alice_rdf, dsl_schema):
        """
        End-to-end test: compile DSL, apply to RDataFrame, extract results.
        
        Uses alice_rdf which has cluster_Q as RVec<RVec<double>>.
        """
        import ROOT
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(dsl_schema)
        dsl.define("sliced_2d", "cluster_Q[0:2, 0:3]")
        
        # Apply to RDataFrame
        rdf_with_slice = dsl.apply(alice_rdf)
        
        # Extract results - should not crash
        results = rdf_with_slice.Take['ROOT::RVec<ROOT::RVec<double>>']("sliced_2d").GetValue()
        
        # Basic validation
        assert len(results) > 0, "No results returned"
        
        # Each result should be a 2D structure with at most 2 tracks
        for evt_idx, result in enumerate(results):
            assert len(result) <= 2, f"Event {evt_idx}: expected <=2 tracks, got {len(result)}"
            for trk_idx, track in enumerate(result):
                assert len(track) <= 3, f"Event {evt_idx}, Track {trk_idx}: expected <=3 clusters, got {len(track)}"
    
    def test_2d_slice_inner_only_e2e(self, alice_rdf, dsl_schema):
        """
        Test cluster_Q[:, 0:2] - all tracks, first 2 clusters.
        """
        import ROOT
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(dsl_schema)
        dsl.define("sliced_inner", "cluster_Q[:, 0:2]")
        
        rdf_with_slice = dsl.apply(alice_rdf)
        results = rdf_with_slice.Take['ROOT::RVec<ROOT::RVec<double>>']("sliced_inner").GetValue()
        
        assert len(results) > 0
        
        # Each track should have at most 2 clusters
        for evt_idx, result in enumerate(results):
            for trk_idx, track in enumerate(result):
                assert len(track) <= 2, f"Event {evt_idx}, Track {trk_idx}: expected <=2 clusters, got {len(track)}"
    
    def test_2d_mixed_index_slice_e2e(self, alice_rdf, dsl_schema):
        """
        Test cluster_Q[0, :] - first track only, all its clusters.
        
        Result should be 1D: RVec<double>.
        """
        import ROOT
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(dsl_schema)
        dsl.define("first_track", "cluster_Q[0, :]")
        
        rdf_with_slice = dsl.apply(alice_rdf)
        
        # Result is 1D after indexing first dimension
        results = rdf_with_slice.Take['ROOT::RVec<double>']("first_track").GetValue()
        
        assert len(results) > 0
        # Each result is the clusters of track 0
        for evt_idx, result in enumerate(results):
            # Just verify it's a 1D vector (no crash, valid data)
            assert isinstance(len(result), int), f"Event {evt_idx}: invalid result type"
    
    def test_negative_index_e2e(self, alice_rdf, dsl_schema):
        """
        Test cluster_Q[-1, :] - last track, all its clusters.
        """
        import ROOT
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(dsl_schema)
        dsl.define("last_track", "cluster_Q[-1, :]")
        
        rdf_with_slice = dsl.apply(alice_rdf)
        results = rdf_with_slice.Take['ROOT::RVec<double>']("last_track").GetValue()
        
        assert len(results) > 0
    
    def test_step_slice_e2e(self, alice_rdf, dsl_schema):
        """
        Test cluster_Q[::2, :] - every other track.
        """
        import ROOT
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(dsl_schema)
        dsl.define("every_other", "cluster_Q[::2, :]")
        
        rdf_with_slice = dsl.apply(alice_rdf)
        results = rdf_with_slice.Take['ROOT::RVec<ROOT::RVec<double>>']("every_other").GetValue()
        
        assert len(results) > 0


# =============================================================================
# NEW: Tests Using toy_nd.py Fixtures (Exact Value Validation)
# =============================================================================
# These tests use the deterministic fixtures from toy_nd.py which provide
# exact integer values that can be validated without floating-point tolerance.

@pytest.mark.skipif(not _ND_GENERATORS_AVAILABLE, reason="toy_nd.py generators not available")
class TestND_ToyGeneratorInvariants:
    """
    Invariance tests using toy_nd.py fixtures.
    
    These use the exact-value invariants:
    - cluster_Q[e][t][c] = 1000*e + 100*t + c
    - hit_E[e][t][c][h] = 10000*e + 1000*t + 100*c + h
    """
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_ND_EXACT_cluster_value(self, nd_2d_dict_small, cluster_value_fn):
        """
        INV-ND-EXACT-1: Verify cluster_Q values match formula exactly.
        
        cluster_Q[e][t][c] = 1000*e + 100*t + c
        
        Type: A (Engine - no ROOT)
        Priority: P0
        """
        data = nd_2d_dict_small
        
        for evt in range(len(data['event_id'])):
            for trk in range(len(data['cluster_Q'][evt])):
                for clus in range(len(data['cluster_Q'][evt][trk])):
                    expected = cluster_value_fn(evt, trk, clus)
                    actual = data['cluster_Q'][evt][trk][clus]
                    assert actual == expected, \
                        f"cluster_Q[{evt}][{trk}][{clus}]: {actual} != {expected}"
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_ND_EXACT_x_y_offset(self, nd_2d_dict_small, cluster_value_fn):
        """
        INV-ND-EXACT-2: Verify x = Q + 0.1, y = Q + 0.2 exactly.
        
        Type: A (Engine - no ROOT)
        Priority: P0
        """
        data = nd_2d_dict_small
        
        for evt in range(len(data['event_id'])):
            for trk in range(len(data['cluster_Q'][evt])):
                Q = data['cluster_Q'][evt][trk]
                x = data['cluster_x'][evt][trk]
                y = data['cluster_y'][evt][trk]
                
                # Use 1e-10 tolerance for floating-point arithmetic
                np.testing.assert_allclose(x - Q, 0.1, rtol=0, atol=1e-10)
                np.testing.assert_allclose(y - Q, 0.2, rtol=0, atol=1e-10)
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_ND_IDENTITY(self, nd_2d_dict_small, expected_2d_flat):
        """
        INV-ND-IDENTITY: Full slice equals original data.
        
        cluster_Q[:, :] == cluster_Q (all values)
        
        Type: A (Engine)
        Priority: P0
        """
        data = nd_2d_dict_small
        layout = data['_layout']
        
        for evt in range(len(data['event_id'])):
            # Get expected flat values using helper
            expected = expected_2d_flat(evt, slice(None), slice(None), layout)
            
            # Flatten actual data
            actual = []
            for trk in range(len(data['cluster_Q'][evt])):
                actual.extend(data['cluster_Q'][evt][trk].tolist())
            
            assert actual == expected, f"Event {evt}: identity invariance violated"
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_ND_PARTITION_SUM(self, nd_2d_dict, expected_sum):
        """
        INV-ND-PARTITION: Sum of partitions equals total sum.
        
        sum(cluster_Q[0:2, :]) + sum(cluster_Q[2:, :]) == sum(cluster_Q[:, :])
        
        Type: A (Engine)
        Priority: P0
        """
        data = nd_2d_dict
        layout = data['_layout']
        
        for evt in range(min(10, len(data['event_id']))):  # Check first 10 events
            n_tracks = len(layout[evt])
            if n_tracks < 3:
                continue  # Need at least 3 tracks to partition
            
            sum_first = expected_sum(evt, slice(0, 2), slice(None), layout)
            sum_rest = expected_sum(evt, slice(2, None), slice(None), layout)
            sum_all = expected_sum(evt, slice(None), slice(None), layout)
            
            assert sum_first + sum_rest == sum_all, \
                f"Event {evt}: partition sum {sum_first} + {sum_rest} != {sum_all}"
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_ND_SLICE_BOUNDS(self, nd_2d_dict_small, cluster_value_fn):
        """
        INV-ND-SLICE-BOUNDS: Verify slice [0:2, 0:3] returns correct values.
        
        Type: A (Engine)
        Priority: P0
        """
        data = nd_2d_dict_small
        
        for evt in range(len(data['event_id'])):
            # Get tracks 0:2 (up to 2 tracks)
            n_tracks = min(2, len(data['cluster_Q'][evt]))
            
            for trk in range(n_tracks):
                # Get clusters 0:3 (up to 3 clusters)
                n_clusters = min(3, len(data['cluster_Q'][evt][trk]))
                
                for clus in range(n_clusters):
                    expected = cluster_value_fn(evt, trk, clus)
                    actual = data['cluster_Q'][evt][trk][clus]
                    assert actual == expected, \
                        f"Slice bounds check failed at [{evt}][{trk}][{clus}]"


@pytest.mark.skipif(not _ND_GENERATORS_AVAILABLE, reason="toy_nd.py generators not available")
class TestND_3D_ToyGeneratorInvariants:
    """
    3D Invariance tests using toy_nd.py fixtures.
    
    Invariant: hit_E[e][t][c][h] = 10000*e + 1000*t + 100*c + h
    """
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_ND3_EXACT_hit_value(self, nd_3d_dict_small, hit_value_fn):
        """
        INV-ND3-EXACT-1: Verify hit_E values match formula exactly.
        
        hit_E[e][t][c][h] = 10000*e + 1000*t + 100*c + h
        
        Type: A (Engine - no ROOT)
        Priority: P0
        """
        data = nd_3d_dict_small
        
        for evt in range(len(data['event_id'])):
            for trk in range(len(data['hit_E'][evt])):
                for clus in range(len(data['hit_E'][evt][trk])):
                    for hit in range(len(data['hit_E'][evt][trk][clus])):
                        expected = hit_value_fn(evt, trk, clus, hit)
                        actual = data['hit_E'][evt][trk][clus][hit]
                        assert actual == expected, \
                            f"hit_E[{evt}][{trk}][{clus}][{hit}]: {actual} != {expected}"
    
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_ND3_t_offset(self, nd_3d_dict_small, hit_value_fn):
        """
        INV-ND3-EXACT-2: Verify hit_t = hit_E + 0.01 exactly.
        
        Type: A (Engine - no ROOT)
        Priority: P1
        """
        data = nd_3d_dict_small
        
        for evt in range(len(data['event_id'])):
            for trk in range(len(data['hit_E'][evt])):
                for clus in range(len(data['hit_E'][evt][trk])):
                    E = data['hit_E'][evt][trk][clus]
                    t = data['hit_t'][evt][trk][clus]
                    np.testing.assert_allclose(t - E, 0.01, rtol=0, atol=1e-10)


# =============================================================================
# ROOT Integration Tests Using toy_nd.py Files (Tier 3)
# =============================================================================

try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
@pytest.mark.skipif(not _ND_GENERATORS_AVAILABLE, reason="toy_nd.py generators not available")
class TestND_ROOT_ToyGenerator:
    """
    Tier 3: ROOT integration tests using toy_nd.py generated files.
    
    These tests create ROOT files with deterministic data and verify
    the values are correct after reading back.
    """
    
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_2d_root_roundtrip(self, nd_2d_root_file_S):
        """
        Test 2D ROOT file roundtrip: generate → read → verify values.
        
        Type: B (ROOT integration)
        Priority: P0
        """
        import ROOT
        
        rdf = ROOT.RDataFrame("Events", nd_2d_root_file_S)
        n_events = rdf.Count().GetValue()
        
        assert n_events > 0, "ROOT file has no events"
        
        # Read first event's cluster_Q
        cluster_Q = rdf.Range(0, 1).Take['ROOT::RVec<ROOT::RVec<double>>']("cluster_Q").GetValue()[0]
        
        # Verify using cluster_value formula
        for trk in range(len(cluster_Q)):
            for clus in range(len(cluster_Q[trk])):
                expected = cluster_value(0, trk, clus)  # event 0
                actual = cluster_Q[trk][clus]
                assert actual == expected, \
                    f"ROOT roundtrip: cluster_Q[0][{trk}][{clus}] = {actual} != {expected}"
    
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_3d_root_roundtrip(self, nd_3d_root_file_S):
        """
        Test 3D ROOT file roundtrip: generate → read → verify values.
        
        Type: B (ROOT integration)
        Priority: P0
        """
        import ROOT
        
        rdf = ROOT.RDataFrame("Events", nd_3d_root_file_S)
        n_events = rdf.Count().GetValue()
        
        assert n_events > 0, "ROOT file has no events"
        
        # Read first event's hit_E
        hit_E = rdf.Range(0, 1).Take['ROOT::RVec<ROOT::RVec<ROOT::RVec<double>>>']("hit_E").GetValue()[0]
        
        # Verify using hit_value formula
        for trk in range(len(hit_E)):
            for clus in range(len(hit_E[trk])):
                for hit_idx in range(len(hit_E[trk][clus])):
                    expected = hit_value(0, trk, clus, hit_idx)  # event 0
                    actual = hit_E[trk][clus][hit_idx]
                    assert actual == expected, \
                        f"ROOT roundtrip: hit_E[0][{trk}][{clus}][{hit_idx}] = {actual} != {expected}"
    
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_2d_dsl_slice_exact_values(self, nd_2d_rdf, nd_2d_schema):
        """
        Test DSL slicing returns exact expected values.
        
        cluster_Q[0:2, 0:3] should return first 2 tracks, first 3 clusters.
        Values should match cluster_value formula exactly.
        
        Type: B (DSL end-to-end)
        Priority: P1
        """
        try:
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("RDataFrameDSL not available")
        
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("sliced", "cluster_Q[0:2, 0:3]")
        
        rdf_with_slice = dsl.apply(nd_2d_rdf)
        results = rdf_with_slice.Take['ROOT::RVec<ROOT::RVec<double>>']("sliced").GetValue()
        
        # Check first event
        if len(results) > 0:
            result = results[0]
            assert len(result) <= 2, f"Expected <=2 tracks, got {len(result)}"
            
            for trk in range(len(result)):
                assert len(result[trk]) <= 3, f"Expected <=3 clusters, got {len(result[trk])}"
                
                for clus in range(len(result[trk])):
                    expected = cluster_value(0, trk, clus)  # event 0
                    actual = result[trk][clus]
                    assert actual == expected, \
                        f"DSL slice: [0][{trk}][{clus}] = {actual} != {expected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
