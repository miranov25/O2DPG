"""
Integration Tests: Flatten + Join (Phase 13.6.C)

Tests that flatten_to_dataframe properly accepts join parameter
and handles mixed-depth data.

These are Tier 1 tests (pure Python, no ROOT).
"""

import pytest
import numpy as np
import pandas as pd

from RDataFrameDSL.flatten import flatten_to_dataframe, FlattenBackend


# =============================================================================
# Fixtures: Synthetic Data (mimics toy_nd.py structure)
# =============================================================================

@pytest.fixture
def mixed_depth_data():
    """
    Synthetic data with event (0D), track (1D), and cluster (2D) columns.
    
    Structure:
    - 3 events
    - Event 0: 2 tracks, 2 clusters each
    - Event 1: 3 tracks, 2 clusters each
    - Event 2: 2 tracks, 2 clusters each
    """
    # Event-level (depth 0)
    event_id = np.array([0, 1, 2], dtype=np.int64)
    event_weight = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    
    # Track-level (depth 1) - RVec<double>
    track_pt = np.empty(3, dtype=object)
    track_pt[0] = np.array([5.0, 13.0], dtype=np.float64)      # Event 0: 2 tracks
    track_pt[1] = np.array([5.0, 13.0, 25.0], dtype=np.float64)  # Event 1: 3 tracks
    track_pt[2] = np.array([5.0, 13.0], dtype=np.float64)      # Event 2: 2 tracks
    
    # Cluster-level (depth 2) - RVec<RVec<double>>
    cluster_Q = np.empty(3, dtype=object)
    
    # Event 0: 2 tracks, 2 clusters each
    cluster_Q[0] = np.empty(2, dtype=object)
    cluster_Q[0][0] = np.array([0.0, 1.0], dtype=np.float64)    # Track 0
    cluster_Q[0][1] = np.array([100.0, 101.0], dtype=np.float64)  # Track 1
    
    # Event 1: 3 tracks, 2 clusters each
    cluster_Q[1] = np.empty(3, dtype=object)
    cluster_Q[1][0] = np.array([1000.0, 1001.0], dtype=np.float64)
    cluster_Q[1][1] = np.array([1100.0, 1101.0], dtype=np.float64)
    cluster_Q[1][2] = np.array([1200.0, 1201.0], dtype=np.float64)
    
    # Event 2: 2 tracks, 2 clusters each
    cluster_Q[2] = np.empty(2, dtype=object)
    cluster_Q[2][0] = np.array([2000.0, 2001.0], dtype=np.float64)
    cluster_Q[2][1] = np.array([2100.0, 2101.0], dtype=np.float64)
    
    return {
        'event_id': event_id,
        'event_weight': event_weight,
        'track_pt': track_pt,
        'cluster_Q': cluster_Q,
    }


# =============================================================================
# Test: Join Parameter Accepted
# =============================================================================

class TestJoinParameterAccepted:
    """Tests that join parameter is properly accepted."""
    
    def test_join_inner_default(self, mixed_depth_data):
        """Inner join (default) should work."""
        df = flatten_to_dataframe(
            mixed_depth_data,
            columns=['cluster_Q', 'track_pt'],
            join='inner'  # explicit default
        )
        assert len(df) > 0
        assert 'cluster_Q' in df.columns
        assert 'track_pt' in df.columns
    
    def test_join_outer(self, mixed_depth_data):
        """Outer join should work."""
        df = flatten_to_dataframe(
            mixed_depth_data,
            columns=['cluster_Q', 'track_pt'],
            join='outer'
        )
        assert len(df) > 0
    
    def test_join_left(self, mixed_depth_data):
        """Left join should work."""
        df = flatten_to_dataframe(
            mixed_depth_data,
            columns=['cluster_Q', 'track_pt'],
            join='left'
        )
        assert len(df) > 0
    
    def test_join_right(self, mixed_depth_data):
        """Right join should work."""
        df = flatten_to_dataframe(
            mixed_depth_data,
            columns=['cluster_Q', 'track_pt'],
            join='right'
        )
        assert len(df) > 0
    
    def test_invalid_join_rejected(self, mixed_depth_data):
        """Invalid join type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid join type"):
            flatten_to_dataframe(
                mixed_depth_data,
                columns=['cluster_Q'],
                join='invalid'
            )


# =============================================================================
# Test: Mixed-Depth Flatten with Join
# =============================================================================

class TestMixedDepthFlatten:
    """Tests for mixed-depth flattening behavior."""
    
    def test_depth_2_1_join(self, mixed_depth_data):
        """
        cluster_Q (depth 2) + track_pt (depth 1) → flatten to depth 2.
        
        track_pt should be broadcast to every cluster.
        """
        df = flatten_to_dataframe(
            mixed_depth_data,
            columns=['cluster_Q', 'track_pt'],
            join='inner'
        )
        
        # Total clusters: (2+2) + (2+2+2) + (2+2) = 4 + 6 + 4 = 14
        assert len(df) == 14
        
        # Should have index columns
        assert 'event_id' in df.columns
        assert 'idx_1' in df.columns
        assert 'idx_2' in df.columns
        
        # Verify broadcast: each cluster has correct track_pt
        for _, row in df.iterrows():
            evt = int(row['event_id'])
            trk = int(row['idx_1'])
            expected_pt = mixed_depth_data['track_pt'][evt][trk]
            assert row['track_pt'] == expected_pt
    
    def test_depth_2_0_join(self, mixed_depth_data):
        """
        cluster_Q (depth 2) + event_weight (depth 0) → flatten to depth 2.
        
        event_weight should be broadcast to every cluster.
        """
        df = flatten_to_dataframe(
            mixed_depth_data,
            columns=['cluster_Q', 'event_weight'],
            join='inner'
        )
        
        assert len(df) == 14  # Same as above
        
        # Verify broadcast: each cluster has correct event_weight
        for _, row in df.iterrows():
            evt = int(row['event_id'])
            expected_weight = mixed_depth_data['event_weight'][evt]
            assert row['event_weight'] == expected_weight
    
    def test_depth_1_0_join(self, mixed_depth_data):
        """
        track_pt (depth 1) + event_weight (depth 0) → flatten to depth 1.
        """
        df = flatten_to_dataframe(
            mixed_depth_data,
            columns=['track_pt', 'event_weight'],
            join='inner'
        )
        
        # Total tracks: 2 + 3 + 2 = 7
        assert len(df) == 7
        
        # Should have track_idx
        assert 'idx_1' in df.columns
        
        # Verify broadcast
        for _, row in df.iterrows():
            evt = int(row['event_id'])
            expected_weight = mixed_depth_data['event_weight'][evt]
            assert row['event_weight'] == expected_weight


# =============================================================================
# Test: Three-Level Join
# =============================================================================

class TestThreeLevelJoin:
    """Tests for depth 2 + depth 1 + depth 0 together."""
    
    def test_all_depths_together(self, mixed_depth_data):
        """
        cluster_Q (2) + track_pt (1) + event_weight (0) → depth 2.
        
        Both track_pt and event_weight should be broadcast.
        """
        df = flatten_to_dataframe(
            mixed_depth_data,
            columns=['cluster_Q', 'track_pt', 'event_weight'],
            join='inner'
        )
        
        assert len(df) == 14
        assert 'cluster_Q' in df.columns
        assert 'track_pt' in df.columns
        assert 'event_weight' in df.columns
        
        # Verify all broadcasts
        for _, row in df.iterrows():
            evt = int(row['event_id'])
            trk = int(row['idx_1'])
            clus = int(row['idx_2'])
            
            # Check cluster_Q value
            expected_Q = mixed_depth_data['cluster_Q'][evt][trk][clus]
            assert row['cluster_Q'] == expected_Q
            
            # Check track_pt broadcast
            expected_pt = mixed_depth_data['track_pt'][evt][trk]
            assert row['track_pt'] == expected_pt
            
            # Check event_weight broadcast
            expected_weight = mixed_depth_data['event_weight'][evt]
            assert row['event_weight'] == expected_weight


# =============================================================================
# Test: Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Tests that existing behavior still works."""
    
    def test_single_column_no_join(self, mixed_depth_data):
        """Single column should work without join logic."""
        df = flatten_to_dataframe(
            mixed_depth_data,
            columns=['cluster_Q'],
        )
        assert len(df) == 14
        assert 'cluster_Q' in df.columns
    
    def test_same_depth_columns(self, mixed_depth_data):
        """Same-depth columns should work (no join needed)."""
        # Create cluster_x with same structure as cluster_Q
        mixed_depth_data['cluster_x'] = np.empty(3, dtype=object)
        for evt in range(3):
            n_tracks = len(mixed_depth_data['cluster_Q'][evt])
            mixed_depth_data['cluster_x'][evt] = np.empty(n_tracks, dtype=object)
            for trk in range(n_tracks):
                n_clus = len(mixed_depth_data['cluster_Q'][evt][trk])
                mixed_depth_data['cluster_x'][evt][trk] = mixed_depth_data['cluster_Q'][evt][trk] + 0.1
        
        df = flatten_to_dataframe(
            mixed_depth_data,
            columns=['cluster_Q', 'cluster_x'],
        )
        assert len(df) == 14
        assert 'cluster_Q' in df.columns
        assert 'cluster_x' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
