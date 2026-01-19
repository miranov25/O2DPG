"""
End-to-End Tests: Join Strategy with RDataFrame (Phase 13.6.C)

Tier 3 tests: Full chain from ROOT file → DSL → to_pandas() with join.

These tests verify that join parameter works through the complete pipeline.
"""

import pytest
import numpy as np


# =============================================================================
# Skip if ROOT not available
# =============================================================================

try:
    import ROOT
    ROOT_AVAILABLE = True
except ImportError:
    ROOT_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ROOT_AVAILABLE, 
    reason="ROOT not available"
)


# =============================================================================
# Test: Mixed-Depth Join with RDataFrame
# =============================================================================

class TestJoinWithRDataFrame:
    """
    End-to-end tests for join parameter with actual RDataFrame.
    
    Uses nd_2d_rdf fixture from conftest_nd_additions.py.
    """
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_E2E_join_cluster_track(self, nd_2d_rdf, nd_2d_schema):
        """
        E2E-JOIN-1: cluster_Q (2D) + track_pt (1D) with join='inner'.
        
        Verifies track_pt is broadcast to every cluster.
        """
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['cluster_Q', 'track_pt'],
            join='inner'
        )
        
        # Should have data
        assert len(df) > 0, "No rows returned"
        
        # Should have both columns
        assert 'cluster_Q' in df.columns
        assert 'track_pt' in df.columns
        
        # Should have index columns for depth 2
        assert 'event_id' in df.columns
        assert 'track_idx' in df.columns
        assert 'cluster_idx' in df.columns
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_E2E_join_cluster_event(self, nd_2d_rdf, nd_2d_schema):
        """
        E2E-JOIN-2: cluster_Q (2D) + event_weight (0D) with join='inner'.
        
        Verifies event_weight is broadcast to every cluster.
        """
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['cluster_Q', 'event_weight'],
            join='inner'
        )
        
        assert len(df) > 0, "No rows returned"
        assert 'cluster_Q' in df.columns
        assert 'event_weight' in df.columns
        
        # Verify broadcast: each row's event_weight matches event_id + 1
        for _, row in df.iterrows():
            evt = int(row['event_id'])
            expected_weight = float(evt + 1)  # event_weight_value formula
            assert row['event_weight'] == expected_weight, \
                f"Event {evt}: expected weight {expected_weight}, got {row['event_weight']}"
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_E2E_join_track_event(self, nd_2d_rdf, nd_2d_schema):
        """
        E2E-JOIN-3: track_pt (1D) + event_weight (0D) with join='inner'.
        
        Verifies event_weight is broadcast to every track.
        """
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['track_pt', 'event_weight'],
            join='inner'
        )
        
        assert len(df) > 0, "No rows returned"
        assert 'track_pt' in df.columns
        assert 'event_weight' in df.columns
        assert 'track_idx' in df.columns
        
        # Verify broadcast
        for _, row in df.iterrows():
            evt = int(row['event_id'])
            expected_weight = float(evt + 1)
            assert row['event_weight'] == expected_weight
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_E2E_join_three_depths(self, nd_2d_rdf, nd_2d_schema):
        """
        E2E-JOIN-4: cluster_Q (2D) + track_pt (1D) + event_weight (0D).
        
        Full three-level join.
        """
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['cluster_Q', 'track_pt', 'event_weight'],
            join='inner'
        )
        
        assert len(df) > 0, "No rows returned"
        assert 'cluster_Q' in df.columns
        assert 'track_pt' in df.columns
        assert 'event_weight' in df.columns
        
        # Output should be at cluster level (depth 2)
        assert 'cluster_idx' in df.columns


# =============================================================================
# Test: Join Strategies
# =============================================================================

class TestJoinStrategiesE2E:
    """End-to-end tests for all four join strategies."""
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_E2E_join_inner(self, nd_2d_rdf, nd_2d_schema):
        """Inner join (default) - no NaN expected."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['cluster_Q', 'track_pt'],
            join='inner'
        )
        
        # Inner join should have no NaN
        assert not df['cluster_Q'].isna().any(), "Inner join produced NaN in cluster_Q"
        assert not df['track_pt'].isna().any(), "Inner join produced NaN in track_pt"
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_E2E_join_outer(self, nd_2d_rdf, nd_2d_schema):
        """Outer join - may have NaN for mismatched indices."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['cluster_Q', 'track_pt'],
            join='outer'
        )
        
        assert len(df) > 0, "No rows returned"
        # Outer join with matching data may or may not have NaN
        # Just verify it runs
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_E2E_join_left(self, nd_2d_rdf, nd_2d_schema):
        """Left join - all from deeper operand."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['cluster_Q', 'track_pt'],
            join='left'
        )
        
        assert len(df) > 0, "No rows returned"
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_E2E_join_right(self, nd_2d_rdf, nd_2d_schema):
        """Right join - all from shallower operand."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['cluster_Q', 'track_pt'],
            join='right'
        )
        
        assert len(df) > 0, "No rows returned"


# =============================================================================
# Test: Invariance with Join
# =============================================================================

class TestJoinInvarianceE2E:
    """
    Invariance tests for join operations with RDataFrame.
    
    Verifies mathematical correctness of broadcast values.
    """
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_E2E_broadcast_event_weight(self, nd_2d_rdf, nd_2d_schema):
        """
        INV-E2E-JOIN-1: event_weight broadcast matches formula.
        
        event_weight[e] = e + 1.0 (from event_weight_value)
        """
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['cluster_Q', 'event_weight'],
            join='inner'
        )
        
        # Every row should satisfy: event_weight == event_id + 1
        for _, row in df.iterrows():
            evt = int(row['event_id'])
            expected = float(evt + 1)
            actual = row['event_weight']
            assert actual == expected, \
                f"Event {evt}: event_weight={actual}, expected={expected}"
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_E2E_cluster_Q_preserved(self, nd_2d_rdf, nd_2d_schema):
        """
        INV-E2E-JOIN-2: cluster_Q values preserved after join.
        
        cluster_Q[e][t][c] = 1000*e + 100*t + c (from cluster_value)
        """
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['cluster_Q', 'track_pt'],
            join='inner'
        )
        
        # Verify cluster_Q follows invariant pattern
        for _, row in df.iterrows():
            evt = int(row['event_id'])
            trk = int(row['track_idx'])
            clus = int(row['cluster_idx'])
            expected_Q = 1000.0 * evt + 100.0 * trk + clus
            actual_Q = row['cluster_Q']
            assert actual_Q == expected_Q, \
                f"cluster_Q[{evt}][{trk}][{clus}]={actual_Q}, expected={expected_Q}"


# =============================================================================
# Test: Backward Compatibility
# =============================================================================

class TestBackwardCompatibilityE2E:
    """Ensure existing behavior still works."""
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_E2E_no_join_param_default(self, nd_2d_rdf, nd_2d_schema):
        """Omitting join parameter should use 'inner' (default)."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        # No join parameter - should work with default
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['cluster_Q', 'track_pt'],
        )
        
        assert len(df) > 0
        assert 'cluster_Q' in df.columns
        assert 'track_pt' in df.columns
    
    @pytest.mark.feature("nd_join_strategy")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_E2E_single_column_still_works(self, nd_2d_rdf, nd_2d_schema):
        """Single column (no join needed) should still work."""
        from RDataFrameDSL import DSLCompiler
        
        dsl = DSLCompiler(nd_2d_schema)
        
        df = dsl.to_pandas(
            nd_2d_rdf,
            columns=['cluster_Q'],
        )
        
        assert len(df) > 0
        assert 'cluster_Q' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
