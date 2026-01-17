"""
Structural Invariance Tests (INV-X*)

Phase 13.6.B.fix: Validates flatten engine correctness.

All tests are Type A (Engine):
- INV-X1-SUM: Sum preservation
- INV-X2-COUNT: Count consistency
- INV-X3-UNIQUE: Index uniqueness
- INV-X4-GROUPBY: Groupby reconstruction
- INV-X5-DTYPE: Dtype preservation

These tests validate that flatten_to_dataframe() produces correct output
without using the DSL chain.

Author: Claude Opus 4.5
Date: 2026-01-15
Phase: 13.6.B.fix
"""

import pytest
import numpy as np
import pandas as pd


class TestStructuralInvariance:
    """Structural invariance tests - Type A (Engine)."""
    
    # =========================================================================
    # INV-X1-SUM: Sum Preservation (P0)
    # =========================================================================
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_X1_SUM_preservation(self, alice_data_xs):
        """
        Sum of flattened values equals sum of nested values.
        
        Type: A (Engine)
        Priority: P0
        
        Tests: No data loss during flatten.
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        # Sum from nested structure
        nested_sum = 0.0
        for event in alice_data_xs['cluster_Q']:
            for track in event:
                nested_sum += track.sum()
        
        # Sum from flattened
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        flat_sum = df['cluster_Q'].sum()
        
        assert np.isclose(nested_sum, flat_sum, rtol=1e-10), \
            f"Sum mismatch: nested={nested_sum}, flat={flat_sum}"
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_X1_SUM_preservation_1d(self, alice_data_xs):
        """
        Sum preservation for 1D columns (track-level).
        
        Type: A (Engine)
        Priority: P0
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        # Sum from nested
        nested_sum = sum(event.sum() for event in alice_data_xs['track_pt'])
        
        # Sum from flattened (1D only)
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt'],
            parent_id_column='event_id'
        )
        flat_sum = df['track_pt'].sum()
        
        assert np.isclose(nested_sum, flat_sum, rtol=1e-10), \
            f"1D sum mismatch: nested={nested_sum}, flat={flat_sum}"
    
    # =========================================================================
    # INV-X2-COUNT: Count Consistency (P0)
    # =========================================================================
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_X2_COUNT_consistency_2d(self, alice_data_xs):
        """
        Flattened row count equals sum of nested lengths (2D).
        
        Type: A (Engine)
        Priority: P0
        
        Tests: No rows lost or duplicated.
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        # Count from nested
        expected_rows = sum(
            sum(len(track) for track in event)
            for event in alice_data_xs['cluster_Q']
        )
        
        # Count from flattened
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        assert len(df) == expected_rows, \
            f"Row count mismatch: expected={expected_rows}, got={len(df)}"
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_X2_COUNT_consistency_1d(self, alice_data_xs):
        """
        Flattened row count equals sum of nested lengths (1D).
        
        Type: A (Engine)
        Priority: P0
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        # Count from nested
        expected_rows = sum(len(event) for event in alice_data_xs['track_pt'])
        
        # Count from flattened
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt'],
            parent_id_column='event_id'
        )
        
        assert len(df) == expected_rows, \
            f"1D row count mismatch: expected={expected_rows}, got={len(df)}"
    
    # =========================================================================
    # INV-X3-UNIQUE: Index Uniqueness (P0)
    # =========================================================================
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_X3_UNIQUE_index_2d(self, alice_data_xs):
        """
        Composite index (event_id, track_idx, cluster_idx) is unique.
        
        Type: A (Engine)
        Priority: P0
        
        Tests: No accidental row duplication.
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        key_cols = ['event_id', 'track_idx', 'cluster_idx']
        duplicates = df.duplicated(subset=key_cols).sum()
        
        assert duplicates == 0, \
            f"Found {duplicates} duplicate index combinations"
    
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_X3_UNIQUE_index_1d(self, alice_data_xs):
        """
        Composite index (event_id, track_idx) is unique for 1D columns.
        
        Type: A (Engine)
        Priority: P0
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt'],
            parent_id_column='event_id'
        )
        
        key_cols = ['event_id', 'track_idx']
        duplicates = df.duplicated(subset=key_cols).sum()
        
        assert duplicates == 0, \
            f"Found {duplicates} duplicate index combinations in 1D"
    
    # =========================================================================
    # INV-X4-GROUPBY: Groupby Reconstruction (P1)
    # =========================================================================
    
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_X4_GROUPBY_per_event_sum(self, alice_data_xs):
        """
        Per-event sum from groupby matches nested structure.
        
        Type: A (Engine)
        Priority: P1
        
        Tests: Index correctness enables aggregation.
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        # Per-event sum from groupby
        gb_sums = df.groupby('event_id')['cluster_Q'].sum()
        
        # Verify against original nested structure
        for event_id in range(len(alice_data_xs['cluster_Q'])):
            event_data = alice_data_xs['cluster_Q'][event_id]
            if len(event_data) > 0:
                expected = sum(track.sum() for track in event_data)
                actual = gb_sums.get(event_id, 0)
                assert np.isclose(expected, actual, rtol=1e-10), \
                    f"Event {event_id}: groupby sum mismatch"
    
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_X4_GROUPBY_per_track_sum(self, alice_data_xs):
        """
        Per-track sum from groupby matches nested structure.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        # Per-track sum from groupby
        gb_sums = df.groupby(['event_id', 'track_idx'])['cluster_Q'].sum()
        
        # Verify against original nested structure
        for event_id in range(len(alice_data_xs['cluster_Q'])):
            event_data = alice_data_xs['cluster_Q'][event_id]
            for track_idx, track_data in enumerate(event_data):
                expected = track_data.sum()
                actual = gb_sums.get((event_id, track_idx), 0)
                assert np.isclose(expected, actual, rtol=1e-10), \
                    f"Event {event_id}, Track {track_idx}: groupby sum mismatch"
    
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_X4_GROUPBY_count(self, alice_data_xs):
        """
        Per-track count from groupby matches nested lengths.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        # Per-track count from groupby
        gb_counts = df.groupby(['event_id', 'track_idx'])['cluster_Q'].count()
        
        # Verify against original nested structure
        for event_id in range(len(alice_data_xs['cluster_Q'])):
            event_data = alice_data_xs['cluster_Q'][event_id]
            for track_idx, track_data in enumerate(event_data):
                expected = len(track_data)
                actual = gb_counts.get((event_id, track_idx), 0)
                assert expected == actual, \
                    f"Event {event_id}, Track {track_idx}: count mismatch"
    
    # =========================================================================
    # INV-X5-DTYPE: Dtype Preservation (P1)
    # =========================================================================
    
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_X5_DTYPE_float64_preserved(self, alice_data_xs):
        """
        Float64 dtype is preserved through flatten.
        
        Type: A (Engine)
        Priority: P1
        
        Tests: No dtype corruption during flatten.
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q', 'track_pt'],
            parent_id_column='event_id'
        )
        
        # cluster_Q should be float64
        assert df['cluster_Q'].dtype == np.float64, \
            f"cluster_Q dtype: expected float64, got {df['cluster_Q'].dtype}"
        
        # track_pt should be float64
        assert df['track_pt'].dtype == np.float64, \
            f"track_pt dtype: expected float64, got {df['track_pt'].dtype}"
    
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_X5_DTYPE_index_int(self, alice_data_xs):
        """
        Index columns are integer types.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        # event_id should be integer
        assert np.issubdtype(df['event_id'].dtype, np.integer), \
            f"event_id dtype: expected integer, got {df['event_id'].dtype}"
        
        # track_idx should be integer
        assert np.issubdtype(df['track_idx'].dtype, np.integer), \
            f"track_idx dtype: expected integer, got {df['track_idx'].dtype}"
        
        # cluster_idx should be integer
        assert np.issubdtype(df['cluster_idx'].dtype, np.integer), \
            f"cluster_idx dtype: expected integer, got {df['cluster_idx'].dtype}"
    
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_X5_DTYPE_no_nan(self, alice_data_xs):
        """
        No NaN values introduced during flatten.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q', 'track_pt'],
            parent_id_column='event_id'
        )
        
        # Check for NaN in value columns
        assert not df['cluster_Q'].isna().any(), \
            f"Found NaN values in cluster_Q"
        
        assert not df['track_pt'].isna().any(), \
            f"Found NaN values in track_pt"
        
        # Check for NaN in index columns
        assert not df['event_id'].isna().any(), \
            f"Found NaN values in event_id"
        
        assert not df['track_idx'].isna().any(), \
            f"Found NaN values in track_idx"


class TestStructuralInvarianceSimple:
    """Structural tests with simple toy data."""
    
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_X_simple_range_sum(self, simple_range_data):
        """
        Sum preservation with simple deterministic data.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        # Expected sums (from simple_range_data fixture)
        # Event 0: 10+11+12+13 + 20+21+22+23+24 = 46 + 110 = 156
        # Event 1: 30+31+32 + 40+41+42+43 + 50+51 = 93 + 166 + 101 = 360
        # Event 2: 60+61+62+63 + 70+71+72 = 246 + 213 = 459
        
        nested_sum = 0.0
        for event in simple_range_data['cluster_Q']:
            for track in event:
                nested_sum += track.sum()
        
        df = flatten_to_dataframe(
            simple_range_data,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        flat_sum = df['cluster_Q'].sum()
        
        assert nested_sum == flat_sum, \
            f"Simple range sum mismatch: nested={nested_sum}, flat={flat_sum}"
    
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_X_first_last_elements(self, simple_range_data):
        """
        First and last elements accessible correctly.
        
        Tests: No off-by-one errors in indexing.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            simple_range_data,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        for (event_id, track_idx), group in df.groupby(['event_id', 'track_idx']):
            # Get original array
            original = simple_range_data['cluster_Q'][event_id][track_idx]
            
            # First element
            assert group['cluster_Q'].iloc[0] == original[0], \
                f"Event {event_id}, Track {track_idx}: first element mismatch"
            
            # Last element
            assert group['cluster_Q'].iloc[-1] == original[-1], \
                f"Event {event_id}, Track {track_idx}: last element mismatch"
            
            # Length
            assert len(group) == len(original), \
                f"Event {event_id}, Track {track_idx}: length mismatch"
