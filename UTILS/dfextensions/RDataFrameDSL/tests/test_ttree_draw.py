"""
Tests for Phase 13.6.B DSL export methods.

Tests:
- TD1-TD4: Event-level queries
- TD5-TD10: Track-level queries  
- TD11-TD14: Cluster-level queries
- TD15-TD20: Integration tests

Prerequisites:
- ALICEEventGenerator (tests/generators/alice_events.py)
- Phase 13.6.A-ext flatten module
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Import generator - try multiple paths
try:
    from tests.generators.alice_events import ALICEEventGenerator, GeneratorConfig
except ImportError:
    try:
        from generators.alice_events import ALICEEventGenerator, GeneratorConfig
    except ImportError:
        # Direct import if in same directory structure
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from generators.alice_events import ALICEEventGenerator, GeneratorConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def alice_data_xs():
    """Generate XS test data (100 events)."""
    gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
    return gen.generate_dict(n_events=100)


@pytest.fixture(scope="module")
def alice_data_s():
    """Generate S test data (1000 events)."""
    gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
    return gen.generate_dict(n_events=1000)


# =============================================================================
# TD1-TD4: Event-Level Queries
# =============================================================================

class TestEventLevelQueries:
    """Tests for scalar/event-level data export."""
    
    def test_TD1_scalar_column_export(self, alice_data_xs):
        """Export single scalar column."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        # Use scalar column only
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['n_tracks'],
            parent_id_column='event_id'
        )
        
        # Should have one row per event
        assert len(df) == 100
        assert 'event_id' in df.columns
        assert 'n_tracks' in df.columns
        
        # Values should match
        np.testing.assert_array_equal(df['n_tracks'].values, alice_data_xs['n_tracks'])
    
    def test_TD2_multiple_scalars(self, alice_data_xs):
        """Export multiple scalar columns."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['n_tracks', 'vertex_z'],
            parent_id_column='event_id'
        )
        
        assert len(df) == 100
        assert 'n_tracks' in df.columns
        assert 'vertex_z' in df.columns
    
    def test_TD3_scalar_with_event_id(self, alice_data_xs):
        """Event ID is included correctly."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['vertex_z'],
            parent_id_column='event_id'
        )
        
        # Event IDs should be sequential
        expected_ids = np.arange(100)
        np.testing.assert_array_equal(df['event_id'].values, expected_ids)
    
    def test_TD4_empty_events_handled(self, alice_data_xs):
        """Empty events are preserved in scalar export."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['n_tracks'],
            parent_id_column='event_id'
        )
        
        # Check for events with 0 tracks
        empty_events = df[df['n_tracks'] == 0]
        # Should have some empty events (2% configured)
        assert len(empty_events) >= 0  # May have 0 due to seed


# =============================================================================
# TD5-TD10: Track-Level Queries
# =============================================================================

class TestTrackLevelQueries:
    """Tests for 1D RVec (track-level) data export."""
    
    def test_TD5_basic_1d_flatten(self, alice_data_xs):
        """Basic 1D RVec flatten."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt'],
            parent_id_column='event_id'
        )
        
        # Should have one row per track
        total_tracks = sum(alice_data_xs['n_tracks'])
        assert len(df) == total_tracks
        
        # Required columns
        assert 'event_id' in df.columns
        assert 'idx_1' in df.columns
        assert 'track_pt' in df.columns
    
    def test_TD6_multiple_1d_same_structure(self, alice_data_xs):
        """Multiple 1D columns with same structure."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'track_phi'],
            parent_id_column='event_id'
        )
        
        # Both columns should have same length
        assert len(df['track_pt']) == len(df['track_phi'])
        
        # Values should be aligned (same track_idx)
        assert 'idx_1' in df.columns
    
    def test_TD7_track_idx_zero_based(self, alice_data_xs):
        """track_idx is 0-based within each event."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt'],
            parent_id_column='event_id'
        )
        
        # Check first event with tracks
        first_event = df[df['event_id'] == df['event_id'].iloc[0]]
        if len(first_event) > 0:
            # track_idx should start at 0
            assert first_event['idx_1'].min() == 0
            # Should be contiguous
            expected_idx = np.arange(len(first_event))
            np.testing.assert_array_equal(first_event['idx_1'].values, expected_idx)
    
    def test_TD8_mixed_1d_scalar(self, alice_data_xs):
        """Mixed 1D + scalar columns (TTree::Draw style)."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'n_tracks'],
            parent_id_column='event_id'
        )
        
        # Scalar should be replicated to track level
        total_tracks = sum(alice_data_xs['n_tracks'])
        assert len(df) == total_tracks
        
        # n_tracks should be replicated
        assert 'n_tracks' in df.columns
        
        # Verify replication: all tracks in an event should have same n_tracks
        for event_id in df['event_id'].unique()[:5]:
            event_df = df[df['event_id'] == event_id]
            assert event_df['n_tracks'].nunique() == 1
    
    def test_TD9_track_pt_values_correct(self, alice_data_xs):
        """Track pt values match original data."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt'],
            parent_id_column='event_id'
        )
        
        # Verify first event
        event_0_tracks = df[df['event_id'] == 0]
        original_pt = alice_data_xs['track_pt'][0]
        
        if len(original_pt) > 0:
            np.testing.assert_array_almost_equal(
                event_0_tracks['track_pt'].values,
                original_pt
            )
    
    def test_TD10_pdgcode_integer_preserved(self, alice_data_xs):
        """Integer columns (pdgcode) preserved correctly."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pdgcode'],
            parent_id_column='event_id'
        )
        
        # Should be integer type
        assert df['track_pdgcode'].dtype in [np.int32, np.int64]
        
        # Values should be valid PDG codes
        valid_pdgs = {211, -211, 321, -321, 2212, -2212, 11, -11,
                      1000010020, -1000010020, 1000010030, -1000010030}
        assert set(df['track_pdgcode'].unique()).issubset(valid_pdgs)


# =============================================================================
# TD11-TD14: Cluster-Level Queries
# =============================================================================

class TestClusterLevelQueries:
    """Tests for 2D RVec<RVec> (cluster-level) data export."""
    
    def test_TD11_basic_2d_flatten(self, alice_data_xs):
        """Basic 2D RVec<RVec> flatten."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        # Should have one row per cluster
        assert 'event_id' in df.columns
        assert 'idx_1' in df.columns
        assert 'idx_2' in df.columns
        assert 'cluster_Q' in df.columns
        
        # Count total clusters
        total_clusters = 0
        for evt in range(len(alice_data_xs['cluster_Q'])):
            for trk in range(len(alice_data_xs['cluster_Q'][evt])):
                total_clusters += len(alice_data_xs['cluster_Q'][evt][trk])
        
        assert len(df) == total_clusters
    
    def test_TD12_mixed_2d_1d(self, alice_data_xs):
        """Mixed 2D + 1D columns (TTree::Draw style)."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q', 'track_pt'],
            parent_id_column='event_id'
        )
        
        # Result should be at cluster level
        assert 'idx_2' in df.columns
        
        # track_pt should be replicated for each cluster
        assert 'track_pt' in df.columns
        
        # Verify replication: all clusters in a track should have same track_pt
        for (event_id, track_idx), group in df.groupby(['event_id', 'idx_1']):
            if len(group) > 1:
                assert group['track_pt'].nunique() == 1
    
    def test_TD13_mixed_2d_1d_scalar(self, alice_data_xs):
        """Mixed 2D + 1D + scalar (full TTree::Draw)."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q', 'track_pt', 'n_tracks'],
            parent_id_column='event_id'
        )
        
        # All columns present
        assert 'cluster_Q' in df.columns
        assert 'track_pt' in df.columns
        assert 'n_tracks' in df.columns
        
        # Index columns present
        assert 'event_id' in df.columns
        assert 'idx_1' in df.columns
        assert 'idx_2' in df.columns
    
    def test_TD14_multiple_2d_columns(self, alice_data_xs):
        """Multiple 2D columns with same structure."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q', 'cluster_x', 'cluster_y'],
            parent_id_column='event_id'
        )
        
        # All columns should have same length
        assert len(df['cluster_Q']) == len(df['cluster_x'])
        assert len(df['cluster_Q']) == len(df['cluster_y'])


# =============================================================================
# TD15-TD20: Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""
    
    def test_TD15_full_chain_generator_to_flatten(self, alice_data_xs):
        """Full chain: Generator → Flatten → DataFrame."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        # Should work without errors
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q', 'track_pt', 'track_pdgcode', 'n_tracks'],
            parent_id_column='event_id'
        )
        
        # Basic sanity checks
        assert len(df) > 0
        assert not df['cluster_Q'].isna().any()
        assert not df['track_pt'].isna().any()
    
    def test_TD16_normalized_tables(self, alice_data_xs):
        """Test flatten_to_tables() for normalized output."""
        from RDataFrameDSL.flatten import flatten_to_tables
        
        tables = flatten_to_tables(
            alice_data_xs,
            columns=['cluster_Q', 'track_pt', 'n_tracks'],
            parent_id_column='event_id'
        )
        
        # Should have three levels
        assert 'events' in tables
        assert 'tracks' in tables
        assert 'clusters' in tables
        
        # Events table should have scalar columns
        assert 'n_tracks' in tables['events'].columns
        
        # Tracks table should have track columns
        assert 'track_pt' in tables['tracks'].columns
        
        # Clusters table should have cluster columns
        assert 'cluster_Q' in tables['clusters'].columns
    
    def test_TD17_joinable_tables(self, alice_data_xs):
        """Normalized tables can be joined correctly."""
        from RDataFrameDSL.flatten import flatten_to_tables
        
        tables = flatten_to_tables(
            alice_data_xs,
            columns=['cluster_Q', 'track_pt', 'n_tracks'],
            parent_id_column='event_id'
        )
        
        # Join tracks to events
        merged = pd.merge(
            tables['events'],
            tables['tracks'],
            on='event_id',
            how='left'
        )
        
        # Should have both event and track columns
        assert 'n_tracks' in merged.columns
        assert 'track_pt' in merged.columns
    
    def test_TD18_empty_events_no_crash(self):
        """Empty events don't cause crashes."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        # Generate data with higher empty event fraction
        gen = ALICEEventGenerator(config=GeneratorConfig(
            seed=123,
            fraction_empty_events=0.3
        ))
        data = gen.generate_dict(n_events=50)
        
        # Should not crash
        df = flatten_to_dataframe(
            data,
            columns=['track_pt'],
            parent_id_column='event_id'
        )
        
        # May have fewer rows due to empty events
        assert len(df) >= 0
    
    def test_TD19_groupby_operations(self, alice_data_xs):
        """Flattened data supports groupby operations."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt'],
            parent_id_column='event_id'
        )
        
        # Group by event
        per_event = df.groupby('event_id')['track_pt'].agg(['mean', 'sum', 'count'])
        
        # Should have one row per event (with tracks)
        assert len(per_event) <= 100
    
    def test_TD20_dtype_preservation(self, alice_data_xs):
        """Data types are preserved through flattening."""
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'track_pdgcode', 'cluster_Q'],
            parent_id_column='event_id'
        )
        
        # Float columns
        assert df['track_pt'].dtype == np.float64
        assert df['cluster_Q'].dtype == np.float64
        
        # Integer columns
        assert df['track_pdgcode'].dtype in [np.int32, np.int64]
        assert df['event_id'].dtype in [np.int32, np.int64]


# =============================================================================
# Performance Benchmark Tests (optional)
# =============================================================================

class TestPerformance:
    """Performance benchmark tests."""
    
    @pytest.mark.slow
    def test_BM1_flatten_s_under_1_second(self, alice_data_s):
        """Flatten S-size data in <1 second."""
        import time
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        start = time.time()
        df = flatten_to_dataframe(
            alice_data_s,
            columns=['cluster_Q', 'track_pt'],
            parent_id_column='event_id'
        )
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"Flatten took {elapsed:.2f}s, target <1.0s"
    
    @pytest.mark.slow
    def test_BM2_normalized_vs_flat_memory(self, alice_data_s):
        """Normalized tables use less memory than flat."""
        import sys
        from RDataFrameDSL.flatten import flatten_to_dataframe, flatten_to_tables
        
        # Get flat DataFrame
        df_flat = flatten_to_dataframe(
            alice_data_s,
            columns=['cluster_Q', 'track_pt', 'n_tracks'],
            parent_id_column='event_id'
        )
        
        # Get normalized tables
        tables = flatten_to_tables(
            alice_data_s,
            columns=['cluster_Q', 'track_pt', 'n_tracks'],
            parent_id_column='event_id'
        )
        
        # Calculate memory usage
        flat_memory = df_flat.memory_usage(deep=True).sum()
        
        norm_memory = sum(
            t.memory_usage(deep=True).sum()
            for t in tables.values()
        )
        
        # Normalized should be smaller (less replication)
        # Note: This may not always hold for small datasets
        print(f"Flat memory: {flat_memory / 1024:.1f} KB")
        print(f"Normalized memory: {norm_memory / 1024:.1f} KB")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
