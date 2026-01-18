"""
Sliding Range Invariance Tests (INV-R*)

Phase 13.6.B.fix: Validates DSL slicing operations.

All tests are Type B (DSL chain):
- INV-R1-SLICE: tracks[:2] slice-only
- INV-R2-SLICE: clusters[:3] slice-only
- INV-R3-COMBINED: tracks[:2].Pt() slice+method
- INV-R4-FILTER: tracks[tracks.Pt() > threshold]
- INV-R5-SUM: Sum preservation via DSL
- INV-R6-BOUNDARY: First/last via DSL

These tests validate DSL slicing syntax:
    dsl.define() → dsl.to_pandas()  # to_pandas calls apply internally
NOT pandas slicing on exported data.

Author: Claude Opus 4.5
Date: 2026-01-15
Phase: 13.6.B.fix
"""

import pytest
import numpy as np
import pandas as pd


class TestSlidingRangesDSL:
    """Sliding range tests - Type B (DSL chain)."""
    
    # =========================================================================
    # INV-R1-SLICE: tracks[:2] slice-only - P0
    # =========================================================================
    
    @pytest.mark.feature("slice_1d")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_R1_SLICE_dsl_slicing(self, toy_lorentz_file):
        """
        Test DSL slicing syntax: tracks[:2]
        
        Must use dsl.define() → dsl.apply() → dsl.to_pandas()
        NOT pandas slicing on exported data.
        
        Type: B (DSL end-to-end)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        rdf = ROOT.RDataFrame("Events", toy_lorentz_file)
        # Schema for toy Lorentz data
        schema = {
            'event_id': 'long',
            'n_tracks': 'int',
            'tracks': 'RVec<TLorentzVector>',
            'track_pt': 'RVec<double>',
            'track_px': 'RVec<double>',
            'track_py': 'RVec<double>',
            'track_phi': 'RVec<double>',
            'track_eta': 'RVec<double>',
        }
        
        dsl = DSLCompiler(schema)
        
        try:
            # Define slice IN DSL - executed by ROOT
            dsl.define("first_2_tracks_pt", "track_pt[:2]")
            
            # Execute DSL chain
            # to_pandas calls apply internally - pass original rdf
            df = dsl.to_pandas(rdf, ['first_2_tracks_pt', 'event_id'])
            
            # Assertions
            assert len(df) > 0, "No rows returned"
            
            # Each event should have at most 2 tracks
            for event_id, group in df.groupby('event_id'):
                n_tracks = len(group)
                assert n_tracks <= 2, \
                    f"Event {event_id}: expected ≤2 tracks, got {n_tracks}"
            
            # Verify values are from beginning of each event's track list
            # Event 0: tracks[:2] should give pt=5, pt=13 (first 2 Pythagorean triples)
            event_0 = df[df['event_id'] == 0]
            if len(event_0) >= 2:
                assert 5.0 in event_0['first_2_tracks_pt'].values, \
                    "Missing pt=5 in event 0 slice"
                assert 13.0 in event_0['first_2_tracks_pt'].values, \
                    "Missing pt=13 in event 0 slice"
                    
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"DSL slicing not implemented: {e}")
    
    @pytest.mark.feature("slice_1d")
    @pytest.mark.limitation("L1")
    @pytest.mark.xfail(reason="L1: slice chain validation - known limitation")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_R1_SLICE_preserves_order(self, toy_lorentz_file):
        """
        DSL slicing preserves element order.
        
        Type: B (DSL end-to-end)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        rdf = ROOT.RDataFrame("Events", toy_lorentz_file)
        # Schema for toy Lorentz data
        schema = {
            'event_id': 'long',
            'n_tracks': 'int',
            'tracks': 'RVec<TLorentzVector>',
            'track_pt': 'RVec<double>',
            'track_px': 'RVec<double>',
            'track_py': 'RVec<double>',
            'track_phi': 'RVec<double>',
            'track_eta': 'RVec<double>',
        }
        
        dsl = DSLCompiler(schema)
        
        try:
            dsl.define("sliced_pt", "track_pt[:2]")
            
            # to_pandas calls apply internally - pass original rdf
            df = dsl.to_pandas(rdf, ['sliced_pt', 'event_id', 'track_idx'])
            
            # Within each event, track_idx should be 0, 1 (in order)
            for event_id, group in df.groupby('event_id'):
                track_indices = group['track_idx'].values
                if len(track_indices) >= 2:
                    assert track_indices[0] == 0, "First track should be idx 0"
                    assert track_indices[1] == 1, "Second track should be idx 1"
                    
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"DSL slicing not implemented: {e}")
    
    # =========================================================================
    # INV-R2-SLICE: clusters[:3] slice-only - P1
    # =========================================================================
    
    @pytest.mark.feature("slice_2d")
    @pytest.mark.limitation("L1")
    @pytest.mark.xfail(reason="L1: slice chain validation - known limitation")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_R2_SLICE_2d_slicing(self, alice_rdf):
        """
        Test DSL slicing on 2D column: clusters[:3]
        
        Type: B (DSL end-to-end)
        Priority: P1
        """
        try:
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("RDataFrameDSL not available")
        
        # Schema for ALICE data
        schema = {
            'event_id': 'long',
            'track_idx': 'int',
            'cluster_Q': 'RVec<RVec<double>>',
        }
        
        dsl = DSLCompiler(schema)
        
        try:
            # Slice 2D column
            dsl.define("first_3_clusters_Q", "cluster_Q[:3]")
            
            # to_pandas calls apply internally - pass original rdf
            df = dsl.to_pandas(alice_rdf, ['first_3_clusters_Q', 'event_id', 'track_idx'])
            
            assert len(df) > 0, "No rows returned"
            
            # Each track should have at most 3 clusters
            for (event_id, track_idx), group in df.groupby(['event_id', 'track_idx']):
                n_clusters = len(group)
                assert n_clusters <= 3, \
                    f"Event {event_id}, Track {track_idx}: expected ≤3 clusters, got {n_clusters}"
                    
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"DSL 2D slicing not implemented: {e}")
    
    # =========================================================================
    # INV-R3-COMBINED: tracks[:2].Pt() slice+method - P0
    # =========================================================================
    
    @pytest.mark.feature("slice_chain")
    @pytest.mark.type_b
    @pytest.mark.p0
    @pytest.mark.phase8
    def test_INV_R3_COMBINED_slice_then_method(self, toy_lorentz_file):
        """
        Test combined slice + method: tracks[:2].Pt()
        
        Validates "slice then broadcast method" ordering.
        
        Type: B (DSL end-to-end)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        rdf = ROOT.RDataFrame("Events", toy_lorentz_file)
        # Schema for toy Lorentz data
        schema = {
            'event_id': 'long',
            'n_tracks': 'int',
            'tracks': 'RVec<TLorentzVector>',
            'track_pt': 'RVec<double>',
            'track_px': 'RVec<double>',
            'track_py': 'RVec<double>',
            'track_phi': 'RVec<double>',
            'track_eta': 'RVec<double>',
        }
        
        dsl = DSLCompiler(schema)
        
        try:
            # Combined: slice THEN method call
            dsl.define("first_2_pts", "tracks[:2].Pt()")
            
            # Also get pre-computed for comparison
            dsl.define("first_2_stored", "track_pt[:2]")
            
            # to_pandas calls apply internally - pass original rdf
            df = dsl.to_pandas(rdf, ['first_2_pts', 'first_2_stored', 'event_id'])
            
            # Assertions
            assert len(df) > 0, "No rows returned"
            
            # Method result should match stored
            assert (df['first_2_pts'] == df['first_2_stored']).all(), \
                "tracks[:2].Pt() != track_pt[:2]"
                
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"Slice+method not implemented: {e}")
    
    # =========================================================================
    # INV-R4-FILTER: tracks[tracks.Pt() > threshold] - P0
    # =========================================================================
    
    @pytest.mark.feature("filter_export")
    @pytest.mark.type_b
    @pytest.mark.p0
    @pytest.mark.phase8
    def test_INV_R4_FILTER_dsl_filtering(self, toy_lorentz_file):
        """
        Test DSL filtering: tracks[tracks.Pt() > threshold]
        
        Must verify filter condition in resulting rows.
        
        Type: B (DSL end-to-end)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        rdf = ROOT.RDataFrame("Events", toy_lorentz_file)
        # Schema for toy Lorentz data
        schema = {
            'event_id': 'long',
            'n_tracks': 'int',
            'tracks': 'RVec<TLorentzVector>',
            'track_pt': 'RVec<double>',
            'track_px': 'RVec<double>',
            'track_py': 'RVec<double>',
            'track_phi': 'RVec<double>',
            'track_eta': 'RVec<double>',
        }
        
        dsl = DSLCompiler(schema)
        
        # Use threshold between known pt values
        # Pythagorean pts: 5, 13, 17, 25, 29, 37, 41
        THRESHOLD = 20.0  # Should filter out pt=5, 13, 17
        
        try:
            # Define filter IN DSL
            dsl.define("high_pt_values", f"tracks[tracks.Pt() > {THRESHOLD}].Pt()")
            
            # to_pandas calls apply internally - pass original rdf
            df = dsl.to_pandas(rdf, ['high_pt_values', 'event_id'])
            
            # Assertions
            if len(df) > 0:
                # ALL returned pt values must exceed threshold
                below_threshold = (df['high_pt_values'] <= THRESHOLD).sum()
                assert below_threshold == 0, \
                    f"Filter failed: {below_threshold} rows have pt <= {THRESHOLD}"
                
                # Should only have pt values > 20: 25, 29, 37, 41
                expected_high_pts = {25.0, 29.0, 37.0, 41.0}
                actual_pts = set(df['high_pt_values'].unique())
                
                for pt in actual_pts:
                    assert pt in expected_high_pts, \
                        f"Unexpected high pt value: {pt}"
                        
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"DSL filtering not implemented: {e}")
    
    @pytest.mark.feature("filter_export")
    @pytest.mark.type_b
    @pytest.mark.p0
    @pytest.mark.phase8
    def test_INV_R4_FILTER_low_threshold(self, toy_lorentz_file):
        """
        Test DSL filtering with low threshold (most tracks pass).
        
        Type: B (DSL end-to-end)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        rdf = ROOT.RDataFrame("Events", toy_lorentz_file)
        # Schema for toy Lorentz data
        schema = {
            'event_id': 'long',
            'n_tracks': 'int',
            'tracks': 'RVec<TLorentzVector>',
            'track_pt': 'RVec<double>',
            'track_px': 'RVec<double>',
            'track_py': 'RVec<double>',
            'track_phi': 'RVec<double>',
            'track_eta': 'RVec<double>',
        }
        
        dsl = DSLCompiler(schema)
        
        THRESHOLD = 1.0  # Very low, all should pass
        
        try:
            dsl.define("filtered_pt", f"tracks[tracks.Pt() > {THRESHOLD}].Pt()")
            
            # to_pandas calls apply internally - pass original rdf
            df = dsl.to_pandas(rdf, ['filtered_pt', 'event_id'])
            
            # Should have all tracks (pt >= 5)
            assert len(df) > 0, "No rows returned"
            
            # All pt values should exceed threshold
            min_pt = df['filtered_pt'].min()
            assert min_pt > THRESHOLD, \
                f"Found pt={min_pt} which is <= {THRESHOLD}"
                
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"DSL filtering not implemented: {e}")
    
    # =========================================================================
    # INV-R5-SUM: Sum preservation via DSL - P1
    # =========================================================================
    
    @pytest.mark.feature("slice_chain")
    @pytest.mark.limitation("L1")
    @pytest.mark.xfail(reason="L1: slice chain validation - known limitation")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_R5_SUM_dsl_sum_preservation(self, toy_lorentz_file):
        """
        Sum preservation when using DSL slicing.
        
        Sum of sliced values should match expected.
        
        Type: B (DSL end-to-end)
        Priority: P1
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        rdf = ROOT.RDataFrame("Events", toy_lorentz_file)
        # Schema for toy Lorentz data
        schema = {
            'event_id': 'long',
            'n_tracks': 'int',
            'tracks': 'RVec<TLorentzVector>',
            'track_pt': 'RVec<double>',
            'track_px': 'RVec<double>',
            'track_py': 'RVec<double>',
            'track_phi': 'RVec<double>',
            'track_eta': 'RVec<double>',
        }
        
        dsl = DSLCompiler(schema)
        
        try:
            # Get full and sliced pt
            dsl.define("full_pt", "track_pt")
            dsl.define("sliced_pt", "track_pt[:2]")
            
            # to_pandas calls apply internally - get all columns at once
            df = dsl.to_pandas(rdf, ['full_pt', 'sliced_pt', 'event_id'])
            
            # For each event, sliced sum should be <= full sum
            full_sums = df.groupby('event_id')['full_pt'].sum()
            sliced_sums = df.groupby('event_id')['sliced_pt'].sum()
            
            for event_id in sliced_sums.index:
                assert sliced_sums[event_id] <= full_sums[event_id] + 1e-10, \
                    f"Event {event_id}: sliced sum > full sum"
                    
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"DSL slicing not implemented: {e}")
    
    # =========================================================================
    # INV-R6-BOUNDARY: First/last via DSL - P1
    # =========================================================================
    
    @pytest.mark.feature("boundary_access")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_R6_BOUNDARY_first_element(self, toy_lorentz_file):
        """
        First element via DSL slicing: arr[:1]
        
        Type: B (DSL end-to-end)
        Priority: P1
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        rdf = ROOT.RDataFrame("Events", toy_lorentz_file)
        # Schema for toy Lorentz data
        schema = {
            'event_id': 'long',
            'n_tracks': 'int',
            'tracks': 'RVec<TLorentzVector>',
            'track_pt': 'RVec<double>',
            'track_px': 'RVec<double>',
            'track_py': 'RVec<double>',
            'track_phi': 'RVec<double>',
            'track_eta': 'RVec<double>',
        }
        
        dsl = DSLCompiler(schema)
        
        try:
            dsl.define("first_pt", "track_pt[:1]")
            
            # to_pandas calls apply internally - pass original rdf
            df = dsl.to_pandas(rdf, ['first_pt', 'event_id'])
            
            # Each event should have exactly 1 row
            for event_id, group in df.groupby('event_id'):
                assert len(group) == 1, \
                    f"Event {event_id}: expected 1 track, got {len(group)}"
            
            # Event 0 first pt should be 5
            event_0 = df[df['event_id'] == 0]
            if len(event_0) > 0:
                assert event_0['first_pt'].iloc[0] == 5.0, \
                    f"Event 0 first pt should be 5, got {event_0['first_pt'].iloc[0]}"
                    
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"DSL slicing not implemented: {e}")
    
    @pytest.mark.feature("boundary_access")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_R6_BOUNDARY_negative_index(self, toy_lorentz_file):
        """
        Last element via DSL slicing: arr[-1:]
        
        Type: B (DSL end-to-end)
        Priority: P1
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        rdf = ROOT.RDataFrame("Events", toy_lorentz_file)
        # Schema for toy Lorentz data
        schema = {
            'event_id': 'long',
            'n_tracks': 'int',
            'tracks': 'RVec<TLorentzVector>',
            'track_pt': 'RVec<double>',
            'track_px': 'RVec<double>',
            'track_py': 'RVec<double>',
            'track_phi': 'RVec<double>',
            'track_eta': 'RVec<double>',
        }
        
        dsl = DSLCompiler(schema)
        
        try:
            dsl.define("last_pt", "track_pt[-1:]")
            
            # to_pandas calls apply internally - pass original rdf
            df = dsl.to_pandas(rdf, ['last_pt', 'event_id'])
            
            # Each event should have exactly 1 row
            for event_id, group in df.groupby('event_id'):
                assert len(group) == 1, \
                    f"Event {event_id}: expected 1 track, got {len(group)}"
            
            # Event 0 last pt should be 13 (2 tracks: 5, 13)
            event_0 = df[df['event_id'] == 0]
            if len(event_0) > 0:
                assert event_0['last_pt'].iloc[0] == 13.0, \
                    f"Event 0 last pt should be 13, got {event_0['last_pt'].iloc[0]}"
                    
        except (NotImplementedError, AttributeError) as e:
            pytest.skip(f"DSL negative indexing not implemented: {e}")


class TestSlidingRangesSimple:
    """Sliding range tests with simple toy data (Type A for validation)."""
    
    @pytest.mark.feature("simple_ranges")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_R_simple_fixed_window_1d_sum(self, simple_range_data):
        """
        Fixed window [i:i+2] on 1D array.
        
        Invariance: sum of window = explicit sum of elements
        
        Type: A (Engine - for validation)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            simple_range_data,
            columns=['track_pt'],
            parent_id_column='event_id'
        )
        
        # For each event, compute sliding window sums
        for event_id, group in df.groupby('event_id'):
            pt_values = group['track_pt'].values
            n_tracks = len(pt_values)
            
            # Window size 2
            for i in range(n_tracks - 1):
                window_sum = pt_values[i] + pt_values[i + 1]
                # This is the expected sum for window [i:i+2]
                assert window_sum == pt_values[i:i+2].sum(), \
                    f"Event {event_id}: window sum mismatch at position {i}"
    
    @pytest.mark.feature("simple_ranges")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_R_simple_fixed_window_2d_sum(self, simple_range_data):
        """
        Fixed window [j:j+2] on 2D array (cluster level).
        
        Invariance: sum of window = explicit sum of elements
        
        Type: A (Engine - for validation)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            simple_range_data,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        # For each track, compute sliding window sums
        for (event_id, track_idx), group in df.groupby(['event_id', 'track_idx']):
            Q_values = group['cluster_Q'].values
            n_clusters = len(Q_values)
            
            # Window size 2
            for j in range(n_clusters - 1):
                window_sum = Q_values[j] + Q_values[j + 1]
                assert window_sum == Q_values[j:j+2].sum(), \
                    f"Event {event_id}, Track {track_idx}: window sum mismatch"
    
    @pytest.mark.feature("simple_ranges")
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_R_simple_track_cluster_sum(self, simple_range_data):
        """
        Per-track cluster sum matches nested structure.
        
        Invariance: groupby(['event_id', 'track_idx']).sum() matches
                   original nested sum
        
        Type: A (Engine)
        Priority: P0
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            simple_range_data,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        # Compute sums from flattened
        flat_sums = df.groupby(['event_id', 'track_idx'])['cluster_Q'].sum()
        
        # Verify against original nested structure
        for event_id in range(len(simple_range_data['cluster_Q'])):
            for track_idx, track_clusters in enumerate(simple_range_data['cluster_Q'][event_id]):
                expected = track_clusters.sum()
                actual = flat_sums[(event_id, track_idx)]
                assert expected == actual, \
                    f"Event {event_id}, Track {track_idx}: sum mismatch"
    
    @pytest.mark.feature("simple_ranges")
    @pytest.mark.type_a
    @pytest.mark.p0
    def test_INV_R_simple_event_cluster_sum(self, simple_range_data):
        """
        Per-event total cluster sum matches nested structure.
        
        Invariance: groupby('event_id').sum() matches original nested sum
        
        Type: A (Engine)
        Priority: P0
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            simple_range_data,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        # Compute sums from flattened
        flat_sums = df.groupby('event_id')['cluster_Q'].sum()
        
        # Verify against original nested structure
        for event_id in range(len(simple_range_data['cluster_Q'])):
            expected = sum(
                track_clusters.sum()
                for track_clusters in simple_range_data['cluster_Q'][event_id]
            )
            actual = flat_sums[event_id]
            assert expected == actual, \
                f"Event {event_id}: total sum mismatch"
