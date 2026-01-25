"""
Invariance tests for BROADCAST-RVEC-SCALAR: RVec × scalar column operations.

Phase 13.6.G: Invariance tests for track_pt * event_weight etc.

Feature: broadcast_rvec_scalar
Test IDs: INV-BROADCAST-1 through INV-BROADCAST-5

These are INVARIANCE tests, not just smoke tests:
- We verify the DSL result matches manual NumPy computation
- We use deterministic toy data with known formulas
"""
import pytest
import numpy as np


class TestInvarianceBroadcast:
    """Invariance tests for RVec × scalar column operations (INV-BROADCAST-*)."""
    
    @pytest.fixture
    def compiler_and_data(self):
        """Set up DSL compiler and test data."""
        import ROOT
        from tests.generators.toy_nd import generate_nd_2d_root
        from RDataFrameDSL import DSLCompiler
        
        filename = generate_nd_2d_root(size='S', seed=42)
        rdf = ROOT.RDataFrame("Events", filename)
        
        schema = {
            'track_pt': 'RVec<double>',
            'track_eta': 'RVec<double>',
            'event_weight': 'double',
            'n_tracks': 'int',
        }
        
        return DSLCompiler(schema), rdf
    
    @pytest.mark.feature("broadcast_rvec_scalar")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_BROADCAST_1_rvec_times_scalar(self, compiler_and_data):
        """
        INV-BROADCAST-1: track_pt * event_weight
        
        Invariant: weighted_pt[i] = track_pt[i] * event_weight for all i
        
        Verifies element-wise multiplication of RVec by scalar column.
        """
        dsl, rdf = compiler_and_data
        
        dsl.define("weighted_pt", "track_pt * event_weight")
        
        applied = dsl.apply(rdf)
        result = applied.AsNumpy(['track_pt', 'event_weight', 'weighted_pt'])
        
        n_events = len(result['track_pt'])
        for i in range(n_events):
            pt = np.array(result['track_pt'][i])
            w = result['event_weight'][i]
            expected = pt * w
            actual = np.array(result['weighted_pt'][i])
            
            assert len(expected) == len(actual), \
                f"Event {i}: length mismatch {len(expected)} vs {len(actual)}"
            assert np.allclose(expected, actual, rtol=1e-10), \
                f"Event {i}: weighted_pt invariant violated\n  expected: {expected}\n  actual: {actual}"
    
    @pytest.mark.feature("broadcast_rvec_scalar")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_BROADCAST_2_scalar_times_rvec(self, compiler_and_data):
        """
        INV-BROADCAST-2: event_weight * track_pt (commutativity)
        
        Invariant: result[i] = event_weight * track_pt[i] for all i
        
        Verifies commutativity: scalar * RVec == RVec * scalar
        """
        dsl, rdf = compiler_and_data
        
        dsl.define("weighted_pt", "event_weight * track_pt")
        
        applied = dsl.apply(rdf)
        result = applied.AsNumpy(['track_pt', 'event_weight', 'weighted_pt'])
        
        n_events = len(result['track_pt'])
        for i in range(n_events):
            pt = np.array(result['track_pt'][i])
            w = result['event_weight'][i]
            expected = w * pt  # Scalar first
            actual = np.array(result['weighted_pt'][i])
            
            assert np.allclose(expected, actual, rtol=1e-10), \
                f"Event {i}: commutativity invariant violated"
    
    @pytest.mark.feature("broadcast_rvec_scalar")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_BROADCAST_3_rvec_plus_scalar(self, compiler_and_data):
        """
        INV-BROADCAST-3: track_pt + event_weight
        
        Invariant: shifted_pt[i] = track_pt[i] + event_weight for all i
        
        Verifies element-wise addition of RVec and scalar column.
        """
        dsl, rdf = compiler_and_data
        
        dsl.define("shifted_pt", "track_pt + event_weight")
        
        applied = dsl.apply(rdf)
        result = applied.AsNumpy(['track_pt', 'event_weight', 'shifted_pt'])
        
        n_events = len(result['track_pt'])
        for i in range(n_events):
            pt = np.array(result['track_pt'][i])
            w = result['event_weight'][i]
            expected = pt + w
            actual = np.array(result['shifted_pt'][i])
            
            assert np.allclose(expected, actual, rtol=1e-10), \
                f"Event {i}: addition invariant violated"
    
    @pytest.mark.feature("broadcast_rvec_scalar")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_BROADCAST_4_rvec_div_scalar(self, compiler_and_data):
        """
        INV-BROADCAST-4: track_pt / event_weight
        
        Invariant: normalized_pt[i] = track_pt[i] / event_weight for all i
        
        Verifies element-wise division of RVec by scalar column.
        """
        dsl, rdf = compiler_and_data
        
        dsl.define("normalized_pt", "track_pt / event_weight")
        
        applied = dsl.apply(rdf)
        result = applied.AsNumpy(['track_pt', 'event_weight', 'normalized_pt'])
        
        n_events = len(result['track_pt'])
        for i in range(n_events):
            pt = np.array(result['track_pt'][i])
            w = result['event_weight'][i]
            if abs(w) > 1e-10:  # Avoid division by zero
                expected = pt / w
                actual = np.array(result['normalized_pt'][i])
                
                assert np.allclose(expected, actual, rtol=1e-10), \
                    f"Event {i}: division invariant violated"
    
    @pytest.mark.feature("broadcast_rvec_scalar")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_BROADCAST_5_int_times_rvec(self, compiler_and_data):
        """
        INV-BROADCAST-5: n_tracks * track_pt (int × RVec<double>)
        
        Invariant: scaled_pt[i] = n_tracks * track_pt[i] for all i
        
        Verifies type promotion: int × double → double
        """
        dsl, rdf = compiler_and_data
        
        dsl.define("scaled_pt", "n_tracks * track_pt")
        
        applied = dsl.apply(rdf)
        result = applied.AsNumpy(['track_pt', 'n_tracks', 'scaled_pt'])
        
        n_events = len(result['track_pt'])
        for i in range(n_events):
            pt = np.array(result['track_pt'][i])
            n = result['n_tracks'][i]
            expected = n * pt  # int × double
            actual = np.array(result['scaled_pt'][i])
            
            assert np.allclose(expected, actual, rtol=1e-10), \
                f"Event {i}: int×double invariant violated"


class TestInvarianceBroadcast2D:
    """Invariance tests for RVec<RVec> × scalar column operations (INV-BROADCAST-2D-*)."""
    
    @pytest.fixture
    def compiler_and_data(self):
        """Set up DSL compiler and test data with 2D arrays."""
        import ROOT
        from tests.generators.toy_nd import generate_nd_2d_root
        from RDataFrameDSL import DSLCompiler
        
        filename = generate_nd_2d_root(size='S', seed=42)
        rdf = ROOT.RDataFrame("Events", filename)
        
        schema = {
            'cluster_Q': 'RVec<RVec<double>>',
            'track_pt': 'RVec<double>',
            'event_weight': 'double',
            'n_tracks': 'int',
        }
        
        return DSLCompiler(schema), rdf
    
    @pytest.mark.feature("broadcast_rvec_scalar")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_BROADCAST_2D_1_rvec2d_times_scalar(self, compiler_and_data):
        """
        INV-BROADCAST-2D-1: cluster_Q * event_weight
        
        Invariant: weighted_Q[t][c] = cluster_Q[t][c] * event_weight for all t, c
        
        Verifies element-wise multiplication of RVec<RVec> by scalar column.
        """
        dsl, rdf = compiler_and_data
        
        dsl.define("weighted_Q", "cluster_Q * event_weight")
        
        applied = dsl.apply(rdf)
        result = applied.AsNumpy(['cluster_Q', 'event_weight', 'weighted_Q'])
        
        n_events = len(result['cluster_Q'])
        for i in range(n_events):
            Q = result['cluster_Q'][i]  # RVec<RVec<double>>
            w = result['event_weight'][i]
            actual = result['weighted_Q'][i]
            
            assert len(Q) == len(actual), \
                f"Event {i}: outer length mismatch {len(Q)} vs {len(actual)}"
            
            for t in range(len(Q)):
                expected_track = np.array(Q[t]) * w
                actual_track = np.array(actual[t])
                assert np.allclose(expected_track, actual_track, rtol=1e-10), \
                    f"Event {i}, Track {t}: weighted_Q invariant violated"
    
    @pytest.mark.feature("broadcast_rvec_scalar")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_BROADCAST_2D_2_scalar_times_rvec2d(self, compiler_and_data):
        """
        INV-BROADCAST-2D-2: event_weight * cluster_Q (commutativity)
        
        Invariant: result[t][c] = event_weight * cluster_Q[t][c] for all t, c
        
        Verifies commutativity: scalar * RVec<RVec> == RVec<RVec> * scalar
        """
        dsl, rdf = compiler_and_data
        
        dsl.define("weighted_Q", "event_weight * cluster_Q")
        
        applied = dsl.apply(rdf)
        result = applied.AsNumpy(['cluster_Q', 'event_weight', 'weighted_Q'])
        
        n_events = len(result['cluster_Q'])
        for i in range(n_events):
            Q = result['cluster_Q'][i]
            w = result['event_weight'][i]
            actual = result['weighted_Q'][i]
            
            for t in range(len(Q)):
                expected_track = w * np.array(Q[t])  # Scalar first
                actual_track = np.array(actual[t])
                assert np.allclose(expected_track, actual_track, rtol=1e-10), \
                    f"Event {i}, Track {t}: commutativity invariant violated"
    
    @pytest.mark.feature("broadcast_rvec_scalar")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_BROADCAST_2D_3_rvec2d_plus_scalar(self, compiler_and_data):
        """
        INV-BROADCAST-2D-3: cluster_Q + event_weight
        
        Invariant: shifted_Q[t][c] = cluster_Q[t][c] + event_weight for all t, c
        
        Verifies element-wise addition of RVec<RVec> and scalar column.
        """
        dsl, rdf = compiler_and_data
        
        dsl.define("shifted_Q", "cluster_Q + event_weight")
        
        applied = dsl.apply(rdf)
        result = applied.AsNumpy(['cluster_Q', 'event_weight', 'shifted_Q'])
        
        n_events = len(result['cluster_Q'])
        for i in range(n_events):
            Q = result['cluster_Q'][i]
            w = result['event_weight'][i]
            actual = result['shifted_Q'][i]
            
            for t in range(len(Q)):
                expected_track = np.array(Q[t]) + w
                actual_track = np.array(actual[t])
                assert np.allclose(expected_track, actual_track, rtol=1e-10), \
                    f"Event {i}, Track {t}: addition invariant violated"
    
    @pytest.mark.feature("broadcast_rvec_scalar")
    @pytest.mark.type_b
    @pytest.mark.p0
    def test_INV_BROADCAST_2D_4_rvec2d_div_scalar(self, compiler_and_data):
        """
        INV-BROADCAST-2D-4: cluster_Q / event_weight
        
        Invariant: normalized_Q[t][c] = cluster_Q[t][c] / event_weight for all t, c
        
        Verifies element-wise division of RVec<RVec> by scalar column.
        """
        dsl, rdf = compiler_and_data
        
        dsl.define("normalized_Q", "cluster_Q / event_weight")
        
        applied = dsl.apply(rdf)
        result = applied.AsNumpy(['cluster_Q', 'event_weight', 'normalized_Q'])
        
        n_events = len(result['cluster_Q'])
        for i in range(n_events):
            Q = result['cluster_Q'][i]
            w = result['event_weight'][i]
            actual = result['normalized_Q'][i]
            
            if abs(w) > 1e-10:  # Avoid division by zero
                for t in range(len(Q)):
                    expected_track = np.array(Q[t]) / w
                    actual_track = np.array(actual[t])
                    assert np.allclose(expected_track, actual_track, rtol=1e-10), \
                        f"Event {i}, Track {t}: division invariant violated"
    
    @pytest.mark.feature("broadcast_rvec_scalar")
    @pytest.mark.type_b
    @pytest.mark.p1
    def test_INV_BROADCAST_2D_5_int_times_rvec2d(self, compiler_and_data):
        """
        INV-BROADCAST-2D-5: n_tracks * cluster_Q (int × RVec<RVec<double>>)
        
        Invariant: scaled_Q[t][c] = n_tracks * cluster_Q[t][c] for all t, c
        
        Verifies type promotion: int × double → double for 2D arrays.
        """
        dsl, rdf = compiler_and_data
        
        dsl.define("scaled_Q", "n_tracks * cluster_Q")
        
        applied = dsl.apply(rdf)
        result = applied.AsNumpy(['cluster_Q', 'n_tracks', 'scaled_Q'])
        
        n_events = len(result['cluster_Q'])
        for i in range(n_events):
            Q = result['cluster_Q'][i]
            n = result['n_tracks'][i]
            actual = result['scaled_Q'][i]
            
            for t in range(len(Q)):
                expected_track = n * np.array(Q[t])  # int × double
                actual_track = np.array(actual[t])
                assert np.allclose(expected_track, actual_track, rtol=1e-10), \
                    f"Event {i}, Track {t}: int×double invariant violated"
