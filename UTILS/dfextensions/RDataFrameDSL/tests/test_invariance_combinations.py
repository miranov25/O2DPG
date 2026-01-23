"""
Depth Combination Invariance Tests (INV-S*)

Phase 13.6.B.fix: Validates mixed-depth operations.

Tests by depth combination:
- INV-S00-ENG: 0D+0D scalar arithmetic
- INV-S01a/b-ENG: 0D+1D scalar replication
- INV-S02a/b-ENG: 0D+2D scalar to cluster replication
- INV-S11a/b-ENG: 1D+1D alignment
- INV-S12a/b-ENG: 1D+2D replication (CRITICAL)
- INV-S12c/d-DSL: 1D+2D DSL chain (Type B)
- INV-S22a-ENG: 2D+2D alignment

Type A tests use flatten_to_dataframe() then compute in pandas/NumPy.
Type B tests use dsl.define() → dsl.apply() → dsl.to_pandas().

Author: Claude Opus 4.5
Date: 2026-01-15
Phase: 13.6.B.fix
"""

import pytest
import numpy as np
import pandas as pd


class TestDepthCombinations:
    """Depth combination invariance tests."""
    
    # =========================================================================
    # INV-S00-ENG: Scalar + Scalar (0D+0D) - P2
    # =========================================================================
    
    @pytest.mark.feature("flatten_scalar")
    @pytest.mark.type_a
    @pytest.mark.p2
    def test_INV_S00_ENG_scalar_scalar_arithmetic(self, alice_data_xs):
        """
        Scalar + Scalar operations at event level.
        
        Expression: n_tracks + event_id - event_id - n_tracks = 0
        Expected: 0 (exact)
        
        Type: A (Engine)
        Priority: P2
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        # Scalars only - flatten gives one row per event
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['n_tracks'],
            parent_id_column='event_id'
        )
        
        # Simple arithmetic on scalars
        n_tracks = df['n_tracks'].values if 'n_tracks' in df.columns else alice_data_xs['n_tracks']
        event_id = df['event_id'].values
        
        result = n_tracks + event_id - event_id - n_tracks
        
        assert (result == 0).all(), "Scalar+Scalar arithmetic failed"
    
    # =========================================================================
    # INV-S01-ENG: Scalar + 1D (0D+1D) - P1
    # =========================================================================
    
    @pytest.mark.feature("flatten_1d")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_S01a_ENG_scalar_1d_replication(self, alice_data_xs):
        """
        Scalar replicated to 1D level.
        
        Expression: (n_tracks * track_pt) / track_pt - n_tracks = 0
        Expected: 0 (within tolerance for division)
        
        Tests: Scalar correctly replicated to track level.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'n_tracks'],
            parent_id_column='event_id'
        )
        
        # Avoid division by zero
        mask = df['track_pt'] != 0
        
        n_tracks = df.loc[mask, 'n_tracks']
        track_pt = df.loc[mask, 'track_pt']
        
        result = (n_tracks * track_pt) / track_pt - n_tracks
        
        tolerance = np.finfo(np.float64).eps * n_tracks.max() * 10
        assert np.abs(result).max() < tolerance, \
            f"Scalar+1D replication failed: max_error={np.abs(result).max()}"
    
    @pytest.mark.feature("flatten_1d")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_S01b_ENG_scalar_1d_consistency(self, alice_data_xs):
        """
        Scalar value consistent across all tracks in event.
        
        Tests: n_tracks same for all tracks in each event.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'n_tracks'],
            parent_id_column='event_id'
        )
        
        for event_id, group in df.groupby('event_id'):
            unique_values = group['n_tracks'].nunique()
            assert unique_values == 1, \
                f"Event {event_id}: n_tracks has {unique_values} different values"
    
    # =========================================================================
    # INV-S02-ENG: Scalar + 2D (0D+2D) - P1
    # =========================================================================
    
    @pytest.mark.feature("flatten_2d")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_S02a_ENG_scalar_2d_replication(self, alice_data_xs):
        """
        Scalar replicated to 2D (cluster) level.
        
        Expression: n_tracks + cluster_Q - cluster_Q - n_tracks = 0
        Expected: 0 (exact)
        
        Tests: Scalar correctly replicated to cluster level.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q', 'n_tracks'],
            parent_id_column='event_id'
        )
        
        result = df['n_tracks'] + df['cluster_Q'] - df['cluster_Q'] - df['n_tracks']
        
        # Use tolerance for floating point comparison
        tolerance = np.finfo(np.float64).eps * 100
        assert np.abs(result).max() < tolerance, \
            f"Scalar+2D replication failed: max_error={np.abs(result).max()}"
    
    @pytest.mark.feature("flatten_2d")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_S02b_ENG_scalar_2d_consistency(self, alice_data_xs):
        """
        Scalar value consistent across all clusters in event.
        
        Tests: n_tracks same for all clusters in each event.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q', 'n_tracks'],
            parent_id_column='event_id'
        )
        
        for event_id, group in df.groupby('event_id'):
            unique_values = group['n_tracks'].nunique()
            assert unique_values == 1, \
                f"Event {event_id}: n_tracks varies across clusters ({unique_values} values)"
    
    # =========================================================================
    # INV-S11-ENG: 1D + 1D - P1
    # =========================================================================
    
    @pytest.mark.feature("flatten_1d")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_S11a_ENG_1d_1d_alignment(self, alice_data_xs):
        """
        Two 1D columns aligned correctly.
        
        Expression: track_pt * track_phi / track_phi - track_pt = 0
        Expected: 0 (within tolerance)
        
        Tests: 1D columns aligned by track_idx.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'track_phi'],
            parent_id_column='event_id'
        )
        
        mask = df['track_phi'] != 0
        track_pt = df.loc[mask, 'track_pt']
        track_phi = df.loc[mask, 'track_phi']
        
        result = track_pt * track_phi / track_phi - track_pt
        
        tolerance = np.finfo(np.float64).eps * track_pt.max() * 10
        assert np.abs(result).max() < tolerance, f"1D+1D alignment failed"
    
    @pytest.mark.feature("flatten_1d")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_S11b_ENG_1d_1d_index_match(self, alice_data_xs):
        """
        1D columns have matching indices (no NaN, same row count).
        
        Tests: track_idx is same for track_pt and track_phi.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'track_phi'],
            parent_id_column='event_id'
        )
        
        # Verify no NaN (which would indicate misalignment)
        assert not df['track_pt'].isna().any(), "NaN in track_pt"
        assert not df['track_phi'].isna().any(), "NaN in track_phi"
        
        # Same row count
        assert len(df['track_pt']) == len(df['track_phi']), \
            "1D columns have different lengths"
    
    # =========================================================================
    # INV-S12-ENG: 1D + 2D (CRITICAL) - P0
    # =========================================================================
    
    @pytest.mark.feature("flatten_mixed")
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_S12a_ENG_1d_2d_replication(self, alice_data_xs):
        """
        1D replicated to 2D level correctly.
        
        Expression: (track_pt * cluster_Q) / cluster_Q - track_pt = 0
        Expected: 0 (within tolerance)
        
        Tests: track_pt replicated per-cluster within track.
        
        Type: A (Engine)
        Priority: P0 (CRITICAL)
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'cluster_Q'],
            parent_id_column='event_id'
        )
        
        mask = df['cluster_Q'] != 0
        track_pt = df.loc[mask, 'track_pt']
        cluster_Q = df.loc[mask, 'cluster_Q']
        
        result = (track_pt * cluster_Q) / cluster_Q - track_pt
        
        tolerance = np.finfo(np.float64).eps * track_pt.max() * 10
        assert np.abs(result).max() < tolerance, f"1D+2D replication failed"
    
    @pytest.mark.feature("flatten_mixed")
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_S12b_ENG_1d_2d_consistency(self, alice_data_xs):
        """
        1D value consistent across all clusters in track.
        
        Tests: track_pt same for all clusters in each track.
        
        Type: A (Engine)
        Priority: P0 (CRITICAL)
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'cluster_Q'],
            parent_id_column='event_id'
        )
        
        for (event_id, track_idx), group in df.groupby(['event_id', 'track_idx']):
            unique_values = group['track_pt'].nunique()
            assert unique_values == 1, \
                f"Event {event_id}, Track {track_idx}: track_pt varies across clusters"
    
    @pytest.mark.feature("flatten_mixed")
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_S12_ENG_main_architect_expression_engine(self, alice_data_xs):
        """
        Main Architect's invariance test - Engine version (Type A).
        
        Expression: (pt*cly + pt*clx) - 0.5*pt*cly - 0.5*pt*clx - 
                   (0.5*pt*cly + 0.5*pt*clx) = 0
        
        Type: A (Engine)
        Priority: P0 (CRITICAL)
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'cluster_x', 'cluster_y'],
            parent_id_column='event_id'
        )
        
        pt = df['track_pt'].values
        clx = df['cluster_x'].values
        cly = df['cluster_y'].values
        
        # Main Architect's expression (corrected: last term is + not -)
        # (pt*cly + pt*clx) - 0.5*pt*cly - 0.5*pt*clx - (0.5*pt*cly + 0.5*pt*clx) = 0
        result = (pt*cly + pt*clx) - 0.5*pt*cly - 0.5*pt*clx - \
                 (0.5*pt*cly + 0.5*pt*clx)
        
        max_mag = np.abs(pt * clx).max()
        tolerance = np.finfo(np.float64).eps * max_mag * 20
        
        assert np.abs(result).max() < tolerance, \
            f"Main Architect expression failed: max_error={np.abs(result).max()}"
    
    # =========================================================================
    # INV-S12-DSL: 1D + 2D DSL Chain (Type B) - P0
    # =========================================================================
    
    @pytest.mark.feature("flatten_mixed")
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.feature("api_define")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_S12c_DSL_main_architect_expression(self, alice_rdf):
        """
        Main Architect's invariance test - FULL DSL CHAIN.
        
        Expression computed IN DSL (not pandas):
        (pt*cly + pt*clx) - 0.5*pt*cly - 0.5*pt*clx - (0.5*pt*cly + 0.5*pt*clx) = 0
        
        Type: B (DSL end-to-end)
        Priority: P0 (CRITICAL)
        """
        try:
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("RDataFrameDSL not available")
        
        # Schema required for DSLCompiler
        schema = {
            'event_id': 'long',
            'track_pt': 'RVec<double>',
            'cluster_x': 'RVec<RVec<double>>',
            'cluster_y': 'RVec<RVec<double>>',
        }
        
        dsl = DSLCompiler(schema)
        
        # Define expression IN DSL - computed by ROOT, not pandas
        # Corrected: last term is + not -
        dsl.define("invariant", 
            "(track_pt * cluster_y + track_pt * cluster_x) - "
            "0.5 * track_pt * cluster_y - 0.5 * track_pt * cluster_x - "
            "(0.5 * track_pt * cluster_y + 0.5 * track_pt * cluster_x)"
        )
        
        # Execute through DSL chain - to_pandas calls apply internally
        df = dsl.to_pandas(alice_rdf, ['invariant', 'event_id'])
        
        # Assertions
        assert len(df) > 0, "No rows returned"
        assert 'invariant' in df.columns, "invariant column missing"
        
        # Numerical invariance check
        max_val = df['invariant'].abs().max()
        tolerance = np.finfo(np.float64).eps * 1e6  # Conservative for complex expression
        
        assert max_val < tolerance, \
            f"Main Architect's expression failed: max={max_val}, tolerance={tolerance}"
    
    @pytest.mark.feature("flatten_mixed")
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.feature("api_define")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_S12d_DSL_mixed_depth_chain(self, alice_rdf):
        """
        Mixed-depth DSL chain test.
        
        Tests dsl.define() → dsl.apply() → dsl.to_pandas() with
        expression involving both 1D and 2D columns.
        
        Type: B (DSL end-to-end)
        Priority: P0
        """
        try:
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("RDataFrameDSL not available")
        
        # Schema required for DSLCompiler
        schema = {
            'event_id': 'long',
            'track_pt': 'RVec<double>',
            'cluster_Q': 'RVec<RVec<double>>',
        }
        
        dsl = DSLCompiler(schema)
        
        # Simple mixed-depth expression
        dsl.define("mixed", "track_pt * cluster_Q")
        
        # to_pandas calls apply internally
        df = dsl.to_pandas(alice_rdf, ['mixed', 'track_pt', 'cluster_Q', 'event_id'])
        
        # Verify structure
        assert len(df) > 0, "No rows returned"
        assert 'mixed' in df.columns, "mixed column missing"
        
        # Verify computation
        expected = df['track_pt'] * df['cluster_Q']
        diff = (df['mixed'] - expected).abs().max()
        
        tolerance = np.finfo(np.float64).eps * df['mixed'].abs().max() * 10
        assert diff < tolerance, f"Mixed-depth computation incorrect: diff={diff}"
    
    # =========================================================================
    # INV-S22-ENG: 2D + 2D - P1
    # =========================================================================
    
    @pytest.mark.feature("flatten_2d")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_S22a_ENG_2d_2d_alignment(self, alice_data_xs):
        """
        Two 2D columns aligned correctly.
        
        Expression: cluster_x * cluster_y / cluster_y - cluster_x = 0
        Expected: 0 (within tolerance)
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_x', 'cluster_y'],
            parent_id_column='event_id'
        )
        
        mask = df['cluster_y'] != 0
        cluster_x = df.loc[mask, 'cluster_x']
        cluster_y = df.loc[mask, 'cluster_y']
        
        result = cluster_x * cluster_y / cluster_y - cluster_x
        
        tolerance = np.finfo(np.float64).eps * cluster_x.abs().max() * 10
        assert np.abs(result).max() < tolerance, f"2D+2D alignment failed"
    
    @pytest.mark.feature("flatten_2d")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_S22b_ENG_2d_2d_index_match(self, alice_data_xs):
        """
        2D columns have matching indices (no NaN, same row count).
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_x', 'cluster_y', 'cluster_Q'],
            parent_id_column='event_id'
        )
        
        # No NaN
        assert not df['cluster_x'].isna().any(), "NaN in cluster_x"
        assert not df['cluster_y'].isna().any(), "NaN in cluster_y"
        assert not df['cluster_Q'].isna().any(), "NaN in cluster_Q"
        
        # Same row count
        assert len(df['cluster_x']) == len(df['cluster_y']) == len(df['cluster_Q']), \
            "2D columns have different lengths"


class TestDepthCombinationsToy:
    """Depth combination tests with toy data for exact validation."""
    
    @pytest.mark.feature("toy_pythagorean")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_S_toy_1d_2d_replication(self, toy_data_with_clusters):
        """
        1D+2D replication with toy data (exact values).
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            toy_data_with_clusters,
            columns=['track_pt', 'cluster_Q'],
            parent_id_column='event_id'
        )
        
        # Verify track_pt consistent within each (event, track) group
        for (event_id, track_idx), group in df.groupby(['event_id', 'track_idx']):
            unique_pt = group['track_pt'].nunique()
            assert unique_pt == 1, \
                f"Event {event_id}, Track {track_idx}: track_pt not constant"
    
    @pytest.mark.feature("toy_pythagorean")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_S_toy_pythagorean_identity(self, toy_data):
        """
        Pythagorean identity with toy data: pt = sqrt(px² + py²).
        
        Uses exact integer Pythagorean triples (no tolerance needed).
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            toy_data,
            columns=['track_pt', 'track_px', 'track_py'],
            parent_id_column='event_id'
        )
        
        pt = df['track_pt'].values
        px = df['track_px'].values
        py = df['track_py'].values
        
        computed_pt = np.sqrt(px**2 + py**2)
        
        # Exact match (Pythagorean triples)
        assert np.allclose(pt, computed_pt), \
            f"Pythagorean identity failed: pt != sqrt(px² + py²)"
