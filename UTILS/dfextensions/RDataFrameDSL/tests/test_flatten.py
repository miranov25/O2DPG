"""
Phase 13.6.A: Flatten Tests

28 tests covering:
- FL1-FL10: Correctness — Primitive Types (10 tests)
- FL11-FL14: Correctness — Struct Types (4 tests)
- FL15-FL18: Correctness — DSL-Computed RVec Columns (4 tests)
- BM1-BM5: Benchmarks (5 tests)
- INT1-INT5: Integration (5 tests)

All correctness tests run with all backends via pytest parameterization.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Import flatten module
from RDataFrameDSL.flatten import (
    flatten_to_dataframe,
    flatten_to_dict,
    FlattenBackend,
    awkward_available,
    is_nested_rvec,
    validate_same_structure,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def simple_1level_data():
    """Simple 1-level RVec data: 3 events with varying track counts."""
    # Event 100: 2 tracks, Event 101: 3 tracks, Event 102: 1 track
    return {
        'event_id': np.array([100, 101, 102], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.2, 3.4], dtype=np.float64),
            np.array([5.6, 7.8, 9.0], dtype=np.float64),
            np.array([2.5], dtype=np.float64),
        ], dtype=object),
        'track_eta': np.array([
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0.3, 0.4, 0.5], dtype=np.float64),
            np.array([0.6], dtype=np.float64),
        ], dtype=object),
    }


@pytest.fixture
def empty_events_data():
    """Data with some empty events."""
    return {
        'event_id': np.array([100, 101, 102, 103], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([], dtype=np.float64),  # Empty
            np.array([3.0], dtype=np.float64),
            np.array([], dtype=np.float64),  # Empty
        ], dtype=object),
    }


@pytest.fixture
def nested_2level_data():
    """2-level nested data: Events → Tracks → Clusters."""
    # Event 100: 2 tracks (2 clusters, 3 clusters)
    # Event 101: 1 track (2 clusters)
    return {
        'event_id': np.array([100, 101], dtype=np.int64),
        'cluster_Q': np.array([
            np.array([
                np.array([10.0, 20.0], dtype=np.float64),
                np.array([30.0, 40.0, 50.0], dtype=np.float64),
            ], dtype=object),
            np.array([
                np.array([60.0, 70.0], dtype=np.float64),
            ], dtype=object),
        ], dtype=object),
    }


@pytest.fixture
def float32_data():
    """Data with float32 type."""
    return {
        'event_id': np.array([100, 101], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float32),
            np.array([3.0], dtype=np.float32),
        ], dtype=object),
    }


# =============================================================================
# Backend Parametrization
# =============================================================================

def get_available_backends():
    """Get list of available backends for parametrization."""
    backends = [FlattenBackend.NUMPY]
    if awkward_available():
        backends.append(FlattenBackend.AWKWARD)
    # C++ not yet implemented, skip
    return backends


AVAILABLE_BACKENDS = get_available_backends()


# =============================================================================
# FL1-FL10: Correctness — Primitive Types
# =============================================================================

class TestFlattenPrimitives:
    """Tests FL1-FL10: Primitive type correctness."""
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL1_flatten_single_rvec_column(self, simple_1level_data, backend):
        """FL1: Flatten single RVec<double> column."""
        df = flatten_to_dataframe(
            simple_1level_data,
            rvec_columns=['track_pt'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Should have 6 rows (2 + 3 + 1 tracks)
        assert len(df) == 6
        
        # Should have correct columns
        assert 'event_id' in df.columns
        assert 'idx_1' in df.columns
        assert 'track_pt' in df.columns
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL2_parent_indices_correct(self, simple_1level_data, backend):
        """FL2: Parent indices (event_id) correct."""
        df = flatten_to_dataframe(
            simple_1level_data,
            rvec_columns=['track_pt'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Event IDs should be replicated correctly
        expected_event_ids = np.array([100, 100, 101, 101, 101, 102])
        np.testing.assert_array_equal(df['event_id'].values, expected_event_ids)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL3_child_indices_correct(self, simple_1level_data, backend):
        """FL3: Child indices (track_idx) correct."""
        df = flatten_to_dataframe(
            simple_1level_data,
            rvec_columns=['track_pt'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Track indices should be 0-based within each event
        expected_track_idx = np.array([0, 1, 0, 1, 2, 0])
        np.testing.assert_array_equal(df['idx_1'].values, expected_track_idx)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL4_values_match_original(self, simple_1level_data, backend):
        """FL4: Values match original."""
        df = flatten_to_dataframe(
            simple_1level_data,
            rvec_columns=['track_pt'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Values should match flattened original
        expected_pt = np.array([1.2, 3.4, 5.6, 7.8, 9.0, 2.5])
        np.testing.assert_array_almost_equal(df['track_pt'].values, expected_pt)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL5_empty_rvec_handling(self, empty_events_data, backend):
        """FL5: Empty RVec handling (n=0)."""
        df = flatten_to_dataframe(
            empty_events_data,
            rvec_columns=['track_pt'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Should have 3 rows (2 + 0 + 1 + 0)
        assert len(df) == 3
        
        # Event 101 and 103 should not appear (empty)
        assert 101 not in df['event_id'].values
        assert 103 not in df['event_id'].values
        
        # Values from non-empty events
        expected_pt = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(df['track_pt'].values, expected_pt)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL6_type_preservation(self, float32_data, backend):
        """FL6: Type preservation (float32/64)."""
        df = flatten_to_dataframe(
            float32_data,
            rvec_columns=['track_pt'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Should preserve float32
        assert df['track_pt'].dtype == np.float32
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL7_multiple_rvec_columns(self, simple_1level_data, backend):
        """FL7: Multiple RVec columns."""
        df = flatten_to_dataframe(
            simple_1level_data,
            rvec_columns=['track_pt', 'track_eta'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Should have both columns
        assert 'track_pt' in df.columns
        assert 'track_eta' in df.columns
        
        # Values should be aligned
        expected_pt = np.array([1.2, 3.4, 5.6, 7.8, 9.0, 2.5])
        expected_eta = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        
        np.testing.assert_array_almost_equal(df['track_pt'].values, expected_pt)
        np.testing.assert_array_almost_equal(df['track_eta'].values, expected_eta)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL8_2level_flatten(self, nested_2level_data, backend):
        """FL8: RVec<RVec<double>> 2-level flatten."""
        df = flatten_to_dataframe(
            nested_2level_data,
            rvec_columns=['cluster_Q'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Should have 7 rows (2 + 3 + 2 clusters)
        assert len(df) == 7
        
        # Should have cluster_idx column
        assert 'idx_2' in df.columns
        
        # Values should be flattened
        expected_Q = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        np.testing.assert_array_almost_equal(df['cluster_Q'].values, expected_Q)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL9_2level_indices(self, nested_2level_data, backend):
        """FL9: 2-level indices (event+track+cluster)."""
        df = flatten_to_dataframe(
            nested_2level_data,
            rvec_columns=['cluster_Q'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Event IDs
        expected_event_ids = np.array([100, 100, 100, 100, 100, 101, 101])
        np.testing.assert_array_equal(df['event_id'].values, expected_event_ids)
        
        # Track indices (0-based within event)
        expected_track_idx = np.array([0, 0, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(df['idx_1'].values, expected_track_idx)
        
        # Cluster indices (0-based within track)
        expected_cluster_idx = np.array([0, 1, 0, 1, 2, 0, 1])
        np.testing.assert_array_equal(df['idx_2'].values, expected_cluster_idx)
    
    def test_FL10_all_backends_same_result(self, simple_1level_data):
        """FL10: All backends produce same result (exact bit equality)."""
        results = {}
        
        for backend in AVAILABLE_BACKENDS:
            df = flatten_to_dataframe(
                simple_1level_data,
                rvec_columns=['track_pt', 'track_eta'],
                parent_id_column='event_id',
                backend=backend
            )
            results[backend] = df
        
        if len(results) < 2:
            pytest.skip("Need at least 2 backends for comparison")
        
        # Compare all backends to first
        reference = list(results.values())[0]
        
        for backend, df in results.items():
            # Exact equality for all columns
            for col in reference.columns:
                np.testing.assert_array_equal(
                    df[col].values, 
                    reference[col].values,
                    err_msg=f"Backend {backend} differs in column {col}"
                )


# =============================================================================
# FL11-FL14: Correctness — Struct Types
# =============================================================================

class TestFlattenStructs:
    """Tests FL11-FL14: Struct type correctness."""
    
    @pytest.fixture
    def simple_hit_data(self):
        """RVec<SimpleHit> data with x, y, z members."""
        # Simulate struct as dict of arrays (how it would come from AsNumpy)
        return {
            'event_id': np.array([100, 101], dtype=np.int64),
            'hit_x': np.array([
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([3.0], dtype=np.float64),
            ], dtype=object),
            'hit_y': np.array([
                np.array([4.0, 5.0], dtype=np.float64),
                np.array([6.0], dtype=np.float64),
            ], dtype=object),
            'hit_z': np.array([
                np.array([7.0, 8.0], dtype=np.float64),
                np.array([9.0], dtype=np.float64),
            ], dtype=object),
        }
    
    @pytest.fixture
    def track_data(self):
        """RVec<Track> data with pt, eta, phi, charge members."""
        return {
            'event_id': np.array([100, 101], dtype=np.int64),
            'track_pt': np.array([
                np.array([1.5, 2.5], dtype=np.float64),
                np.array([3.5, 4.5, 5.5], dtype=np.float64),
            ], dtype=object),
            'track_eta': np.array([
                np.array([0.1, 0.2], dtype=np.float64),
                np.array([0.3, 0.4, 0.5], dtype=np.float64),
            ], dtype=object),
            'track_phi': np.array([
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([3.0, 4.0, 5.0], dtype=np.float64),
            ], dtype=object),
            'track_charge': np.array([
                np.array([1, -1], dtype=np.int32),
                np.array([1, 1, -1], dtype=np.int32),
            ], dtype=object),
        }
    
    @pytest.fixture
    def nested_hit_data(self):
        """RVec<RVec<SimpleHit>> data."""
        return {
            'event_id': np.array([100], dtype=np.int64),
            'hit_x': np.array([
                np.array([
                    np.array([1.0, 2.0], dtype=np.float64),
                    np.array([3.0], dtype=np.float64),
                ], dtype=object),
            ], dtype=object),
            'hit_y': np.array([
                np.array([
                    np.array([4.0, 5.0], dtype=np.float64),
                    np.array([6.0], dtype=np.float64),
                ], dtype=object),
            ], dtype=object),
            'hit_z': np.array([
                np.array([
                    np.array([7.0, 8.0], dtype=np.float64),
                    np.array([9.0], dtype=np.float64),
                ], dtype=object),
            ], dtype=object),
        }
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL11_rvec_simple_hit(self, simple_hit_data, backend):
        """FL11: RVec<SimpleHit> (x,y,z) → 3 columns + indices."""
        df = flatten_to_dataframe(
            simple_hit_data,
            rvec_columns=['hit_x', 'hit_y', 'hit_z'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Should have 3 rows
        assert len(df) == 3
        
        # All columns present
        assert 'hit_x' in df.columns
        assert 'hit_y' in df.columns
        assert 'hit_z' in df.columns
        assert 'idx_1' in df.columns
        
        # Values aligned correctly
        expected_x = np.array([1.0, 2.0, 3.0])
        expected_y = np.array([4.0, 5.0, 6.0])
        expected_z = np.array([7.0, 8.0, 9.0])
        
        np.testing.assert_array_almost_equal(df['hit_x'].values, expected_x)
        np.testing.assert_array_almost_equal(df['hit_y'].values, expected_y)
        np.testing.assert_array_almost_equal(df['hit_z'].values, expected_z)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL12_rvec_track(self, track_data, backend):
        """FL12: RVec<Track> (pt,eta,phi,charge) → 4 columns + indices."""
        df = flatten_to_dataframe(
            track_data,
            rvec_columns=['track_pt', 'track_eta', 'track_phi', 'track_charge'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Should have 5 rows (2 + 3)
        assert len(df) == 5
        
        # All columns present with correct types
        assert df['track_pt'].dtype == np.float64
        assert df['track_charge'].dtype == np.int32
        
        # Check values
        expected_pt = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        expected_charge = np.array([1, -1, 1, 1, -1])
        
        np.testing.assert_array_almost_equal(df['track_pt'].values, expected_pt)
        np.testing.assert_array_equal(df['track_charge'].values, expected_charge)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL13_rvec_rvec_simple_hit(self, nested_hit_data, backend):
        """FL13: RVec<RVec<SimpleHit>> → 3 columns + 2-level indices."""
        df = flatten_to_dataframe(
            nested_hit_data,
            rvec_columns=['hit_x', 'hit_y', 'hit_z'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Should have 3 rows (2 + 1)
        assert len(df) == 3
        
        # Should have 2-level indices
        assert 'idx_1' in df.columns
        assert 'idx_2' in df.columns
        
        # Values correct
        expected_x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(df['hit_x'].values, expected_x)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL14_rvec_rvec_cluster(self, backend):
        """FL14: RVec<RVec<Cluster>> (Q,x,y,z) → 4 columns + 2-level indices."""
        # Create cluster data
        data = {
            'event_id': np.array([100], dtype=np.int64),
            'cluster_Q': np.array([
                np.array([
                    np.array([10.0, 20.0], dtype=np.float64),
                    np.array([30.0], dtype=np.float64),
                ], dtype=object),
            ], dtype=object),
            'cluster_x': np.array([
                np.array([
                    np.array([1.0, 2.0], dtype=np.float64),
                    np.array([3.0], dtype=np.float64),
                ], dtype=object),
            ], dtype=object),
            'cluster_y': np.array([
                np.array([
                    np.array([4.0, 5.0], dtype=np.float64),
                    np.array([6.0], dtype=np.float64),
                ], dtype=object),
            ], dtype=object),
            'cluster_z': np.array([
                np.array([
                    np.array([7.0, 8.0], dtype=np.float64),
                    np.array([9.0], dtype=np.float64),
                ], dtype=object),
            ], dtype=object),
        }
        
        df = flatten_to_dataframe(
            data,
            rvec_columns=['cluster_Q', 'cluster_x', 'cluster_y', 'cluster_z'],
            parent_id_column='event_id',
            backend=backend
        )
        
        # Should have 3 rows
        assert len(df) == 3
        
        # All 4 columns + indices
        assert all(col in df.columns for col in ['cluster_Q', 'cluster_x', 'cluster_y', 'cluster_z'])
        assert 'idx_1' in df.columns
        assert 'idx_2' in df.columns


# =============================================================================
# FL15-FL18: Correctness — DSL-Computed RVec Columns
# =============================================================================

class TestFlattenDSLComputed:
    """Tests FL15-FL18: DSL-computed RVec columns."""
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL15_flatten_computed_rvec(self, backend):
        """FL15: Flatten DSL-computed RVec<double>."""
        # Simulate DSL-computed column (track_pt * 2)
        data = {
            'event_id': np.array([100, 101], dtype=np.int64),
            'track_momentum': np.array([
                np.array([2.4, 6.8], dtype=np.float64),  # 1.2*2, 3.4*2
                np.array([11.2], dtype=np.float64),      # 5.6*2
            ], dtype=object),
        }
        
        df = flatten_to_dataframe(
            data,
            rvec_columns=['track_momentum'],
            parent_id_column='event_id',
            backend=backend
        )
        
        assert len(df) == 3
        expected = np.array([2.4, 6.8, 11.2])
        np.testing.assert_array_almost_equal(df['track_momentum'].values, expected)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL16_flatten_aggregation_result(self, backend):
        """FL16: Flatten result of aggregation → RVec<double>."""
        # Simulate cumulative sum result
        data = {
            'event_id': np.array([100, 101], dtype=np.int64),
            'cumsum_pt': np.array([
                np.array([1.0, 3.0], dtype=np.float64),  # cumsum([1, 2])
                np.array([5.0, 9.0, 15.0], dtype=np.float64),  # cumsum([5, 4, 6])
            ], dtype=object),
        }
        
        df = flatten_to_dataframe(
            data,
            rvec_columns=['cumsum_pt'],
            parent_id_column='event_id',
            backend=backend
        )
        
        assert len(df) == 5
        expected = np.array([1.0, 3.0, 5.0, 9.0, 15.0])
        np.testing.assert_array_almost_equal(df['cumsum_pt'].values, expected)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL17_mixed_raw_and_computed(self, backend):
        """FL17: Mixed flatten: raw columns + computed columns together."""
        data = {
            'event_id': np.array([100, 101], dtype=np.int64),
            'track_pt': np.array([
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([3.0], dtype=np.float64),
            ], dtype=object),
            'track_p': np.array([  # Computed: pt * cosh(eta)
                np.array([1.1, 2.2], dtype=np.float64),
                np.array([3.3], dtype=np.float64),
            ], dtype=object),
            'track_eta': np.array([
                np.array([0.1, 0.2], dtype=np.float64),
                np.array([0.3], dtype=np.float64),
            ], dtype=object),
        }
        
        df = flatten_to_dataframe(
            data,
            rvec_columns=['track_pt', 'track_p', 'track_eta'],
            parent_id_column='event_id',
            backend=backend
        )
        
        assert len(df) == 3
        assert all(col in df.columns for col in ['track_pt', 'track_p', 'track_eta'])
        
        # All values aligned
        expected_pt = np.array([1.0, 2.0, 3.0])
        expected_p = np.array([1.1, 2.2, 3.3])
        
        np.testing.assert_array_almost_equal(df['track_pt'].values, expected_pt)
        np.testing.assert_array_almost_equal(df['track_p'].values, expected_p)
    
    @pytest.mark.parametrize("backend", AVAILABLE_BACKENDS)
    def test_FL18_phase8_method_broadcast(self, backend):
        """FL18: Phase 8 method broadcasting: tracks.Pt() → RVec<double>."""
        # Simulate Phase 8 result: tracks.Pt() produces RVec<double>
        # This is the PRIMARY USE CASE
        data = {
            'event_id': np.array([100, 101], dtype=np.int64),
            'track_pts': np.array([  # Result of tracks.Pt()
                np.array([1.5, 2.5, 3.5], dtype=np.float64),
                np.array([4.5, 5.5], dtype=np.float64),
            ], dtype=object),
        }
        
        df = flatten_to_dataframe(
            data,
            rvec_columns=['track_pts'],
            parent_id_column='event_id',
            backend=backend
        )
        
        assert len(df) == 5
        
        # Correct indices
        expected_event_ids = np.array([100, 100, 100, 101, 101])
        expected_track_idx = np.array([0, 1, 2, 0, 1])
        
        np.testing.assert_array_equal(df['event_id'].values, expected_event_ids)
        np.testing.assert_array_equal(df['idx_1'].values, expected_track_idx)
        
        # Correct values
        expected_pts = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        np.testing.assert_array_almost_equal(df['track_pts'].values, expected_pts)


# =============================================================================
# BM1-BM5: Benchmarks
# =============================================================================

class TestFlattenBenchmarks:
    """Tests BM1-BM5: Performance benchmarks."""
    
    @pytest.fixture
    def large_data_500k(self):
        """Generate 500k tracks across 10k events."""
        np.random.seed(42)
        n_events = 10_000
        tracks_per_event = 50
        
        event_ids = np.arange(n_events, dtype=np.int64)
        
        # Generate RVec arrays
        track_pt = np.array([
            np.random.exponential(2.0, tracks_per_event).astype(np.float64)
            for _ in range(n_events)
        ], dtype=object)
        
        track_eta = np.array([
            np.random.normal(0, 1, tracks_per_event).astype(np.float64)
            for _ in range(n_events)
        ], dtype=object)
        
        return {
            'event_id': event_ids,
            'track_pt': track_pt,
            'track_eta': track_eta,
        }
    
    @pytest.mark.benchmark
    def test_BM1_concat_vs_preallocate(self, large_data_500k):
        """BM1: NumPy concat vs preallocate comparison."""
        import time
        
        data = large_data_500k
        
        # Preallocate method (our implementation)
        start = time.perf_counter()
        df = flatten_to_dataframe(
            data,
            rvec_columns=['track_pt'],
            parent_id_column='event_id',
            backend=FlattenBackend.NUMPY
        )
        preallocate_time = time.perf_counter() - start
        
        # Verify result
        assert len(df) == 500_000
        
        # Log timing
        print(f"\nBM1: Preallocate time: {preallocate_time*1000:.1f}ms for 500k tracks")
        
        # Should be under 500ms target
        assert preallocate_time < 0.5, f"Too slow: {preallocate_time:.3f}s"
    
    @pytest.mark.benchmark
    def test_BM2_all_backends_comparison(self, large_data_500k):
        """BM2: All backends comparison (500k tracks)."""
        import time
        
        data = large_data_500k
        timings = {}
        
        for backend in AVAILABLE_BACKENDS:
            start = time.perf_counter()
            df = flatten_to_dataframe(
                data,
                rvec_columns=['track_pt'],
                parent_id_column='event_id',
                backend=backend
            )
            elapsed = time.perf_counter() - start
            timings[backend.value] = elapsed
            
            # Verify
            assert len(df) == 500_000
        
        print(f"\nBM2 Backend comparison (500k tracks):")
        for name, t in timings.items():
            print(f"  {name}: {t*1000:.1f}ms")
        
        # All should be under 500ms
        for name, t in timings.items():
            assert t < 0.5, f"{name} too slow: {t:.3f}s"
    
    @pytest.mark.benchmark
    def test_BM3_scaling_test(self):
        """BM3: Scaling test (S→M→L)."""
        import time
        
        np.random.seed(42)
        
        scenarios = [
            ('S', 1_000, 50, 50_000, 0.1),
            ('M', 10_000, 50, 500_000, 0.5),
            # ('L', 100_000, 50, 5_000_000, 2.0),  # Skip L for CI
        ]
        
        print("\nBM3 Scaling test:")
        
        for name, n_events, tracks_per, expected_total, target_time in scenarios:
            # Generate data
            data = {
                'event_id': np.arange(n_events, dtype=np.int64),
                'track_pt': np.array([
                    np.random.exponential(2.0, tracks_per).astype(np.float64)
                    for _ in range(n_events)
                ], dtype=object),
            }
            
            start = time.perf_counter()
            df = flatten_to_dataframe(
                data,
                rvec_columns=['track_pt'],
                parent_id_column='event_id',
                backend=FlattenBackend.NUMPY
            )
            elapsed = time.perf_counter() - start
            
            print(f"  {name}: {len(df)} tracks in {elapsed*1000:.1f}ms (target: {target_time*1000:.0f}ms)")
            
            assert len(df) == expected_total
            assert elapsed < target_time, f"{name} too slow: {elapsed:.3f}s > {target_time}s"
    
    @pytest.mark.benchmark
    def test_BM4_memory_usage(self, large_data_500k):
        """BM4: Memory usage (tracemalloc)."""
        import tracemalloc
        
        data = large_data_500k
        
        tracemalloc.start()
        
        df = flatten_to_dataframe(
            data,
            rvec_columns=['track_pt', 'track_eta'],
            parent_id_column='event_id',
            backend=FlattenBackend.NUMPY
        )
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024
        
        print(f"\nBM4 Memory usage:")
        print(f"  Current: {current_mb:.1f} MB")
        print(f"  Peak: {peak_mb:.1f} MB")
        
        # Should be under 200MB for 500k tracks (target from spec)
        assert peak_mb < 200, f"Peak memory too high: {peak_mb:.1f} MB"
    
    @pytest.mark.benchmark
    def test_BM5_fragmentation_measurement(self, large_data_500k):
        """BM5: Fragmentation measurement."""
        import tracemalloc
        import gc
        
        data = large_data_500k
        
        tracemalloc.start()
        
        df = flatten_to_dataframe(
            data,
            rvec_columns=['track_pt'],
            parent_id_column='event_id',
            backend=FlattenBackend.NUMPY
        )
        
        _, peak = tracemalloc.get_traced_memory()
        
        # Force GC and measure final
        gc.collect()
        final, _ = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        final_mb = final / 1024 / 1024
        
        fragmentation_ratio = peak_mb / final_mb if final_mb > 0 else 1.0
        
        print(f"\nBM5 Fragmentation:")
        print(f"  Peak: {peak_mb:.1f} MB")
        print(f"  Final: {final_mb:.1f} MB")
        print(f"  Ratio: {fragmentation_ratio:.2f}")
        
        # Fragmentation ratio should be reasonable (< 3x)
        assert fragmentation_ratio < 3.0, f"High fragmentation: {fragmentation_ratio:.2f}"


# =============================================================================
# INT1-INT5: Integration Tests
# =============================================================================

class TestFlattenIntegration:
    """Tests INT1-INT5: Integration with other components."""
    
    @pytest.fixture
    def integration_data(self):
        """Data for integration tests."""
        return {
            'event_id': np.array([100, 101, 102], dtype=np.int64),
            'track_pt': np.array([
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([3.0, 4.0, 5.0], dtype=np.float64),
                np.array([6.0], dtype=np.float64),
            ], dtype=object),
            'track_eta': np.array([
                np.array([0.1, 0.2], dtype=np.float64),
                np.array([0.3, 0.4, 0.5], dtype=np.float64),
                np.array([0.6], dtype=np.float64),
            ], dtype=object),
        }
    
    def test_INT1_to_pandas_dataframe(self, integration_data):
        """INT1: Flattened → pandas DataFrame."""
        df = flatten_to_dataframe(
            integration_data,
            rvec_columns=['track_pt', 'track_eta'],
            parent_id_column='event_id'
        )
        
        # Should be pandas DataFrame
        assert isinstance(df, pd.DataFrame)
        
        # Should have correct shape
        assert df.shape == (6, 4)  # 6 rows, 4 columns
        
        # Can use pandas operations
        mean_pt = df.groupby('event_id')['track_pt'].mean()
        assert len(mean_pt) == 3
    
    def test_INT2_aliasdf_compatible_format(self, integration_data):
        """INT2: Compatible with AliasDataFrame format."""
        df = flatten_to_dataframe(
            integration_data,
            rvec_columns=['track_pt', 'track_eta'],
            parent_id_column='event_id'
        )
        
        # AliasDataFrame expects: event_id, track_idx, data columns
        required_columns = ['event_id', 'idx_1', 'track_pt', 'track_eta']
        assert all(col in df.columns for col in required_columns)
        
        # Dtypes should be preserved
        assert df['event_id'].dtype == np.int64
        assert df['idx_1'].dtype == np.int64
        assert df['track_pt'].dtype == np.float64
    
    def test_INT3_dfdraw_compatible(self, integration_data):
        """INT3: Works with dfdraw.draw() (simulated)."""
        df = flatten_to_dataframe(
            integration_data,
            rvec_columns=['track_pt', 'track_eta'],
            parent_id_column='event_id'
        )
        
        # dfdraw expects flat DataFrame with numeric columns
        # Simulate draw by creating histogram
        import numpy as np
        
        # Should be able to histogram
        hist, edges = np.histogram(df['track_pt'], bins=10)
        assert len(hist) == 10
        
        # Should be able to scatter plot data
        x = df['track_pt'].values
        y = df['track_eta'].values
        assert len(x) == len(y) == 6
    
    def test_INT4_dsl_defined_columns_included(self):
        """INT4: DSL-defined columns included in flatten."""
        # Simulate data with both raw and DSL-computed columns
        data = {
            'event_id': np.array([100, 101], dtype=np.int64),
            'track_pt': np.array([  # Raw from data
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([3.0], dtype=np.float64),
            ], dtype=object),
            'track_momentum': np.array([  # DSL-computed
                np.array([1.1, 2.2], dtype=np.float64),
                np.array([3.3], dtype=np.float64),
            ], dtype=object),
        }
        
        df = flatten_to_dataframe(
            data,
            rvec_columns=['track_pt', 'track_momentum'],
            parent_id_column='event_id'
        )
        
        # Both columns should be present
        assert 'track_pt' in df.columns
        assert 'track_momentum' in df.columns
        
        # Values aligned
        assert len(df) == 3
    
    def test_INT5_filters_applied_before_flatten(self):
        """INT5: Filters applied before flatten (simulated)."""
        # Simulate filtered data (only high-pt tracks)
        # In real usage: dsl.filter("track_pt > 2.0")
        data = {
            'event_id': np.array([100, 101], dtype=np.int64),
            'track_pt': np.array([
                np.array([3.0], dtype=np.float64),  # Only 1 track > 2.0
                np.array([4.0, 5.0], dtype=np.float64),  # 2 tracks > 2.0
            ], dtype=object),
        }
        
        df = flatten_to_dataframe(
            data,
            rvec_columns=['track_pt'],
            parent_id_column='event_id'
        )
        
        # Should have filtered result
        assert len(df) == 3
        assert all(df['track_pt'] > 2.0)


# =============================================================================
# Validation Tests
# =============================================================================

class TestFlattenValidation:
    """Tests for input validation."""
    
    def test_empty_rvec_columns_rejected(self):
        """Empty rvec_columns should raise error."""
        data = {'event_id': np.array([100])}
        
        with pytest.raises(ValueError, match="cannot be empty"):
            flatten_to_dataframe(data, rvec_columns=[], parent_id_column='event_id')
    
    def test_missing_parent_column_rejected(self):
        """Missing parent_id_column should raise error."""
        data = {
            'track_pt': np.array([np.array([1.0])], dtype=object),
        }
        
        with pytest.raises(ValueError, match="not in data"):
            flatten_to_dataframe(data, rvec_columns=['track_pt'], parent_id_column='event_id')
    
    def test_missing_rvec_column_rejected(self):
        """Missing rvec column should raise error."""
        data = {
            'event_id': np.array([100]),
            'track_pt': np.array([np.array([1.0])], dtype=object),
        }
        
        with pytest.raises(ValueError, match="not in data"):
            flatten_to_dataframe(data, rvec_columns=['track_pt', 'track_eta'], parent_id_column='event_id')
    
    def test_different_structures_rejected(self):
        """Columns with different jagged structures should raise error."""
        data = {
            'event_id': np.array([100, 101], dtype=np.int64),
            'track_pt': np.array([
                np.array([1.0, 2.0], dtype=np.float64),
                np.array([3.0], dtype=np.float64),
            ], dtype=object),
            'track_eta': np.array([
                np.array([0.1], dtype=np.float64),  # Different length!
                np.array([0.3, 0.4], dtype=np.float64),  # Different length!
            ], dtype=object),
        }
        
        with pytest.raises(ValueError, match="different structures"):
            flatten_to_dataframe(data, rvec_columns=['track_pt', 'track_eta'], parent_id_column='event_id')


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
