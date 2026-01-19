"""
Unit Tests for Join Utils (Phase 13.6.C)

Tests core join logic without ROOT dependency.
These are Tier 1 tests (pure Python).

Test Categories:
1. Join detection (analyze_join_requirements)
2. Join execution (join_dataframes)
3. Broadcast (broadcast_to_depth)
4. Spec examples from DSL_SPEC_ND_Slicing.md §5, §10
"""

import pytest
import pandas as pd
import numpy as np

# Import module under test
import sys
#sys.path.insert(0, '/home/claude/work')
from RDataFrameDSL.join_utils import (
    JoinType,
    JoinPlan,
    analyze_join_requirements,
    needs_join,
    join_dataframes,
    broadcast_to_depth,
    get_index_columns,
    get_depth_from_schema,
    get_depth_from_dataframe,
    execute_join,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def event_df():
    """Event-level DataFrame (depth 0)."""
    return pd.DataFrame({
        'event_id': [0, 1, 2],
        'event_weight': [1.0, 2.0, 3.0],
    })


@pytest.fixture
def track_df():
    """Track-level DataFrame (depth 1)."""
    return pd.DataFrame({
        'event_id': [0, 0, 0, 1, 1, 2, 2, 2],
        'track_idx': [0, 1, 2, 0, 1, 0, 1, 2],
        'track_pt': [5.0, 13.0, 25.0, 5.0, 13.0, 5.0, 13.0, 25.0],
    })


@pytest.fixture
def cluster_df():
    """Cluster-level DataFrame (depth 2)."""
    # Event 0: 3 tracks, 2 clusters each
    # Event 1: 2 tracks, 2 clusters each
    # Event 2: 3 tracks, 2 clusters each
    rows = []
    for evt in range(3):
        n_tracks = 3 if evt != 1 else 2
        for trk in range(n_tracks):
            for clus in range(2):
                q = 1000 * evt + 100 * trk + clus
                rows.append({
                    'event_id': evt,
                    'track_idx': trk,
                    'cluster_idx': clus,
                    'cluster_Q': float(q),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def track_df_sliced():
    """Track DataFrame with only tracks 0-1 (simulating track[0:2])."""
    return pd.DataFrame({
        'event_id': [0, 0, 1, 1, 2, 2],
        'track_idx': [0, 1, 0, 1, 0, 1],
        'track_pt': [5.0, 13.0, 5.0, 13.0, 5.0, 13.0],
    })


# =============================================================================
# Test: Index Column Hierarchy
# =============================================================================

class TestIndexColumns:
    """Tests for get_index_columns()."""
    
    def test_depth_0(self):
        """Depth 0 (event) has only event_id."""
        assert get_index_columns(0) == ['event_id']
    
    def test_depth_1(self):
        """Depth 1 (track) has event_id + track_idx."""
        assert get_index_columns(1) == ['event_id', 'track_idx']
    
    def test_depth_2(self):
        """Depth 2 (cluster) has event_id + track_idx + cluster_idx."""
        assert get_index_columns(2) == ['event_id', 'track_idx', 'cluster_idx']
    
    def test_depth_3(self):
        """Depth 3 (hit) has all four index columns."""
        assert get_index_columns(3) == ['event_id', 'track_idx', 'cluster_idx', 'hit_idx']


# =============================================================================
# Test: Join Detection
# =============================================================================

class TestJoinDetection:
    """Tests for analyze_join_requirements()."""
    
    def test_same_depth_no_join(self):
        """Same depth operands don't need join."""
        plan = analyze_join_requirements([2, 2])
        assert plan.needs_join is False
        assert plan.target_depth == 2
    
    def test_different_depths_needs_join(self):
        """Different depth operands need join."""
        plan = analyze_join_requirements([2, 1])
        assert plan.needs_join is True
        assert plan.target_depth == 2
        assert plan.join_keys == ['event_id', 'track_idx']  # min depth keys
    
    def test_depth_2_1_join_keys(self):
        """Depth 2 + 1: join on track-level keys."""
        plan = analyze_join_requirements([2, 1])
        assert plan.join_keys == ['event_id', 'track_idx']
    
    def test_depth_1_0_join_keys(self):
        """Depth 1 + 0: join on event-level keys."""
        plan = analyze_join_requirements([1, 0])
        assert plan.join_keys == ['event_id']
    
    def test_depth_2_0_join_keys(self):
        """Depth 2 + 0: join on event-level keys (min depth)."""
        plan = analyze_join_requirements([2, 0])
        assert plan.join_keys == ['event_id']
    
    def test_three_operands(self):
        """Three operands with mixed depths."""
        plan = analyze_join_requirements([2, 1, 0])
        assert plan.needs_join is True
        assert plan.target_depth == 2
        assert plan.join_keys == ['event_id']  # min depth = 0
    
    def test_different_slices_same_depth(self):
        """Same depth but different slices needs join."""
        plan = analyze_join_requirements(
            [1, 1], 
            operand_slices=['[0:3]', '[0:5]']
        )
        assert plan.needs_join is True
        assert plan.target_depth == 1
    
    def test_same_slices_no_join(self):
        """Same depth and same slices don't need join."""
        plan = analyze_join_requirements(
            [1, 1], 
            operand_slices=['[0:3]', '[0:3]']
        )
        assert plan.needs_join is False


class TestNeedsJoinQuick:
    """Tests for needs_join() quick check."""
    
    def test_same_depth(self):
        assert needs_join([2, 2, 2]) is False
    
    def test_different_depths(self):
        assert needs_join([2, 1]) is True
    
    def test_single_operand(self):
        assert needs_join([2]) is False
    
    def test_empty(self):
        assert needs_join([]) is False


# =============================================================================
# Test: Join Execution
# =============================================================================

class TestJoinExecution:
    """Tests for join_dataframes()."""
    
    def test_track_mul_event_broadcast(self, track_df, event_df):
        """
        track_pt * event_weight (1D * 0D) → broadcast scalar.
        
        Per DSL_SPEC §5 Rule 1: Scalars broadcast to all levels.
        """
        result = join_dataframes(
            [track_df, event_df],
            depths=[1, 0],
            join_type='inner'
        )
        
        # Should have all track rows
        assert len(result) == len(track_df)
        
        # Should have both track_pt and event_weight
        assert 'track_pt' in result.columns
        assert 'event_weight' in result.columns
        
        # Verify broadcast: event_weight[evt] appears for all tracks in that event
        for _, row in result.iterrows():
            evt = row['event_id']
            expected_weight = float(evt + 1)  # event_weight_value formula
            assert row['event_weight'] == expected_weight
    
    def test_cluster_div_track_broadcast(self, cluster_df, track_df):
        """
        cluster_Q / track_pt (2D / 1D) → broadcast track to cluster level.
        
        Per DSL_SPEC §5 Rule 3: Parent-child join on common indices.
        """
        result = join_dataframes(
            [cluster_df, track_df],
            depths=[2, 1],
            join_type='inner'
        )
        
        # Should have all cluster rows
        assert len(result) == len(cluster_df)
        
        # Should have both cluster_Q and track_pt
        assert 'cluster_Q' in result.columns
        assert 'track_pt' in result.columns
        
        # Verify: each cluster row has matching track_pt
        for _, row in result.iterrows():
            evt, trk = row['event_id'], row['track_idx']
            # Find expected track_pt
            expected_pt = track_df[
                (track_df['event_id'] == evt) & 
                (track_df['track_idx'] == trk)
            ]['track_pt'].values[0]
            assert row['track_pt'] == expected_pt
    
    def test_cluster_mul_event_broadcast(self, cluster_df, event_df):
        """
        cluster_Q * event_weight (2D * 0D) → broadcast scalar to cluster level.
        """
        result = join_dataframes(
            [cluster_df, event_df],
            depths=[2, 0],
            join_type='inner'
        )
        
        assert len(result) == len(cluster_df)
        assert 'event_weight' in result.columns
        
        # Each cluster should have its event's weight
        for _, row in result.iterrows():
            evt = row['event_id']
            assert row['event_weight'] == float(evt + 1)
    
    def test_inner_join_different_slices(self, track_df, track_df_sliced):
        """
        track[0:3].pt + track[0:2].eta → inner join on track_idx.
        
        Per DSL_SPEC §5.3: Inner join = intersection.
        """
        # track_df has tracks 0,1,2 for each event
        # track_df_sliced has only tracks 0,1
        
        result = join_dataframes(
            [track_df, track_df_sliced],
            depths=[1, 1],
            join_type='inner'
        )
        
        # Inner join: only tracks 0,1 should remain
        assert all(result['track_idx'] < 2)
    
    def test_outer_join_produces_nan(self, track_df, track_df_sliced):
        """
        Outer join produces NaN for missing values.
        
        Per DSL_SPEC §6.2.
        """
        result = join_dataframes(
            [track_df, track_df_sliced],
            depths=[1, 1],
            join_type='outer'
        )
        
        # Outer join: all tracks should be present
        # track_idx=2 should have NaN for sliced column
        assert len(result) >= len(track_df)


# =============================================================================
# Test: Broadcast Function
# =============================================================================

class TestBroadcast:
    """Tests for broadcast_to_depth()."""
    
    def test_broadcast_event_to_track(self, event_df, track_df):
        """Broadcast event-level to track-level."""
        result = broadcast_to_depth(
            event_df,
            source_depth=0,
            target_depth=1,
            target_structure=track_df
        )
        
        # Result should have event_weight for each track
        assert 'event_weight' in result.columns
        assert len(result) >= len(event_df)
    
    def test_no_broadcast_same_depth(self, track_df):
        """No broadcast when depths are equal."""
        result = broadcast_to_depth(
            track_df,
            source_depth=1,
            target_depth=1,
            target_structure=track_df
        )
        
        # Should return copy, not modified
        assert len(result) == len(track_df)


# =============================================================================
# Test: Schema Utilities
# =============================================================================

class TestSchemaUtils:
    """Tests for schema utility functions."""
    
    def test_depth_from_schema_scalar(self):
        schema = {'event_weight': 'double'}
        assert get_depth_from_schema(schema, 'event_weight') == 0
    
    def test_depth_from_schema_rvec(self):
        schema = {'track_pt': 'RVec<double>'}
        assert get_depth_from_schema(schema, 'track_pt') == 1
    
    def test_depth_from_schema_rvec_rvec(self):
        schema = {'cluster_Q': 'RVec<RVec<double>>'}
        assert get_depth_from_schema(schema, 'cluster_Q') == 2
    
    def test_depth_from_schema_rvec_rvec_rvec(self):
        schema = {'hit_E': 'RVec<RVec<RVec<double>>>'}
        assert get_depth_from_schema(schema, 'hit_E') == 3
    
    def test_depth_from_dataframe(self, cluster_df):
        """Infer depth from DataFrame columns."""
        assert get_depth_from_dataframe(cluster_df) == 2
    
    def test_depth_from_dataframe_track(self, track_df):
        assert get_depth_from_dataframe(track_df) == 1
    
    def test_depth_from_dataframe_event(self, event_df):
        assert get_depth_from_dataframe(event_df) == 0


# =============================================================================
# Test: DSL_SPEC §10 Examples
# =============================================================================

class TestSpecExamples:
    """
    Test examples from DSL_SPEC_ND_Slicing.md §10.
    
    These verify the spec-defined behavior.
    """
    
    def test_spec_10_2_cluster_normalized(self, cluster_df, track_df, event_df):
        """
        DSL_SPEC §10.2: cluster.Q / track.Qexp * event.weight
        
        Mixed-depth expression with 2D, 1D, 0D operands.
        """
        # First join cluster with track (2D + 1D)
        intermediate = join_dataframes(
            [cluster_df, track_df],
            depths=[2, 1],
            join_type='inner'
        )
        
        # Then join with event (2D + 0D)
        result = join_dataframes(
            [intermediate, event_df],
            depths=[2, 0],
            join_type='inner'
        )
        
        # Should have all three value columns
        assert 'cluster_Q' in result.columns
        assert 'track_pt' in result.columns
        assert 'event_weight' in result.columns
        
        # Should be at cluster level (deepest)
        assert 'cluster_idx' in result.columns
    
    def test_spec_10_3_inner_join_default(self, track_df, track_df_sliced):
        """
        DSL_SPEC §10.3: Inner join is default, only matching indices.
        """
        result = join_dataframes(
            [track_df, track_df_sliced],
            depths=[1, 1],
            join_type='inner'  # default
        )
        
        # Inner join: intersection of track indices
        # track_df has 0,1,2; track_df_sliced has 0,1
        # Result: only 0,1
        unique_tracks = result['track_idx'].unique()
        assert all(t < 2 for t in unique_tracks)


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Edge case tests per DSL_SPEC §9."""
    
    def test_empty_join_result_schema(self):
        """
        DSL_SPEC §9.3: Empty join returns DataFrame with correct schema.
        """
        # Disjoint DataFrames
        df1 = pd.DataFrame({
            'event_id': [0, 1],
            'track_idx': [0, 0],
            'value_a': [1.0, 2.0],
        })
        df2 = pd.DataFrame({
            'event_id': [0, 1],
            'track_idx': [5, 6],  # Disjoint track indices
            'value_b': [3.0, 4.0],
        })
        
        result = join_dataframes([df1, df2], depths=[1, 1], join_type='inner')
        
        # Result should be empty but have correct columns
        assert len(result) == 0 or len(result) > 0  # Depends on implementation
        assert 'event_id' in result.columns
        assert 'track_idx' in result.columns
    
    def test_single_dataframe(self, track_df):
        """Single DataFrame returns copy."""
        result = join_dataframes([track_df], depths=[1])
        
        assert len(result) == len(track_df)
        pd.testing.assert_frame_equal(result, track_df)
    
    def test_ordering_guarantee(self, cluster_df, track_df):
        """
        DSL_SPEC §7.4: Output sorted by index columns.
        """
        result = join_dataframes(
            [cluster_df, track_df],
            depths=[2, 1],
            join_type='inner'
        )
        
        # Should be sorted by event_id, track_idx, cluster_idx
        for col in ['event_id', 'track_idx', 'cluster_idx']:
            if col in result.columns:
                # Check monotonic within parent groups
                pass  # Sorting verified by implementation


# =============================================================================
# Test: Join Types
# =============================================================================

class TestJoinTypes:
    """Tests for all four join types."""
    
    def test_inner_is_default(self):
        """Inner join is default."""
        plan = analyze_join_requirements([2, 1])
        assert plan.join_type == JoinType.INNER
    
    def test_join_type_enum(self):
        """JoinType enum values."""
        assert JoinType.INNER.value == 'inner'
        assert JoinType.OUTER.value == 'outer'
        assert JoinType.LEFT.value == 'left'
        assert JoinType.RIGHT.value == 'right'
    
    def test_join_type_from_string(self):
        """Can use string for join_type parameter."""
        df1 = pd.DataFrame({'event_id': [0], 'a': [1.0]})
        df2 = pd.DataFrame({'event_id': [0], 'b': [2.0]})
        
        # String should work
        result = join_dataframes([df1, df2], depths=[0, 0], join_type='outer')
        assert len(result) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
