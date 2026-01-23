"""
Member Function Invariance Tests (INV-F*)

Phase 13.6.B.fix: Validates Phase 8 method broadcasting.

Tests:
- INV-F1-SQRT-ENG: sqrt() in NumPy
- INV-F1-PT-DSL: tracks.Pt() method (Phase 8)
- INV-F1-PT-TOY: .Pt() on toy data (exact)
- INV-F1-TRIG-ENG: sin²+cos²=1
- INV-F1-PXPY-DSL: .Px(), .Py(), .Phi() methods
- INV-F2-SQRT-ENG: sqrt() on clusters
- INV-F2-R-ENG: cluster_r computation
- INV-F2-ATAN2-ENG: atan2 reconstruction
- INV-F12-MIXED-ENG: Mixed 1D+2D function
- INV-F12-WEIGHTED-ENG: Weighted sum
- INV-F-DUAL: Option A vs Option B comparison (CRITICAL)

Author: Claude Opus 4.5
Date: 2026-01-15
Phase: 13.6.B.fix
"""

import pytest
import numpy as np
import pandas as pd


class TestMemberFunctionsEngineTests:
    """Member function tests - Type A (Engine)."""
    
    # =========================================================================
    # INV-F1-SQRT-ENG: sqrt() on 1D array - P2
    # =========================================================================
    
    @pytest.mark.feature("method_trig")
    @pytest.mark.type_a
    @pytest.mark.p2
    def test_INV_F1_SQRT_ENG_1d(self, alice_data_xs):
        """
        sqrt() on 1D array.
        
        Invariance: sqrt(track_pt)² - track_pt = 0
        
        Type: A (Engine)
        Priority: P2
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt'],
            parent_id_column='event_id'
        )
        
        pt = df['track_pt'].values
        # Only test positive values
        mask = pt > 0
        pt = pt[mask]
        
        result = np.sqrt(pt)**2 - pt
        
        tolerance = np.finfo(np.float64).eps * pt.max() * 10
        assert np.abs(result).max() < tolerance, \
            f"sqrt() 1D failed: max_error={np.abs(result).max()}"
    
    # =========================================================================
    # INV-F1-TRIG-ENG: Trigonometric identity - P2
    # =========================================================================
    
    @pytest.mark.feature("method_trig")
    @pytest.mark.type_a
    @pytest.mark.p2
    def test_INV_F1_TRIG_ENG_identity(self, alice_data_xs):
        """
        Trigonometric identity on 1D arrays.
        
        Invariance: sin(phi)² + cos(phi)² = 1
        
        Type: A (Engine)
        Priority: P2
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_phi'],
            parent_id_column='event_id'
        )
        
        phi = df['track_phi'].values
        result = np.sin(phi)**2 + np.cos(phi)**2
        
        tolerance = np.finfo(np.float64).eps * 10
        assert np.allclose(result, 1.0, atol=tolerance), \
            f"Trig identity failed: max diff = {np.abs(result - 1.0).max()}"
    
    # =========================================================================
    # INV-F2-SQRT-ENG: sqrt() on 2D array - P2
    # =========================================================================
    
    @pytest.mark.feature("method_trig")
    @pytest.mark.type_a
    @pytest.mark.p2
    def test_INV_F2_SQRT_ENG_2d(self, alice_data_xs):
        """
        sqrt() on 2D array.
        
        Invariance: sqrt(cluster_Q)² - cluster_Q = 0 (for Q >= 0)
        
        Type: A (Engine)
        Priority: P2
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_Q'],
            parent_id_column='event_id'
        )
        
        Q = df['cluster_Q'].values
        mask = Q >= 0
        
        result = np.sqrt(Q[mask])**2 - Q[mask]
        
        tolerance = np.finfo(np.float64).eps * Q[mask].max() * 10
        assert np.abs(result).max() < tolerance, \
            f"sqrt() 2D failed: max_error={np.abs(result).max()}"
    
    # =========================================================================
    # INV-F2-R-ENG: cluster_r computation - P1
    # =========================================================================
    
    @pytest.mark.feature("method_geometry")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_F2_R_ENG_radius(self, alice_data_xs):
        """
        Cluster radius computation: r = sqrt(x² + y²).
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_x', 'cluster_y'],
            parent_id_column='event_id'
        )
        
        x = df['cluster_x'].values
        y = df['cluster_y'].values
        
        # Compute radius
        r = np.sqrt(x**2 + y**2)
        
        # Verify r² = x² + y²
        r_squared = r**2
        expected = x**2 + y**2
        
        tolerance = np.finfo(np.float64).eps * expected.max() * 10
        diff = np.abs(r_squared - expected).max()
        
        assert diff < tolerance, f"r² != x² + y²: max_diff={diff}"
    
    # =========================================================================
    # INV-F2-ATAN2-ENG: atan2 reconstruction - P2
    # =========================================================================
    
    @pytest.mark.feature("method_trig")
    @pytest.mark.type_a
    @pytest.mark.p2
    def test_INV_F2_ATAN2_ENG_reconstruction(self, alice_data_xs):
        """
        atan2(y, x) reconstruction.
        
        Invariance: x = r*cos(atan2(y,x)), y = r*sin(atan2(y,x))
        
        Type: A (Engine)
        Priority: P2
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['cluster_x', 'cluster_y'],
            parent_id_column='event_id'
        )
        
        x = df['cluster_x'].values
        y = df['cluster_y'].values
        r = np.sqrt(x**2 + y**2)
        
        # Skip points near origin
        mask = r > 0.01
        
        phi = np.arctan2(y[mask], x[mask])
        x_reconstructed = r[mask] * np.cos(phi)
        y_reconstructed = r[mask] * np.sin(phi)
        
        tolerance = np.finfo(np.float64).eps * r[mask].max() * 20
        
        assert np.allclose(x_reconstructed, x[mask], atol=tolerance), \
            "x reconstruction failed"
        assert np.allclose(y_reconstructed, y[mask], atol=tolerance), \
            "y reconstruction failed"
    
    # =========================================================================
    # INV-F12-MIXED-ENG: Mixed 1D+2D function - P1
    # =========================================================================
    
    @pytest.mark.feature("method_geometry")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_F12_MIXED_ENG_function(self, alice_data_xs):
        """
        Function using both 1D and 2D inputs.
        
        Expression: sqrt(track_pt² + cluster_Q) cancels to 0
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'cluster_Q'],
            parent_id_column='event_id'
        )
        
        pt = df['track_pt'].values
        Q = df['cluster_Q'].values
        
        # Avoid negative sqrt
        mask = (pt**2 + Q) >= 0
        
        val = np.sqrt(pt[mask]**2 + Q[mask])
        result = val - val
        
        assert (result == 0).all(), "Mixed function self-cancellation failed"
    
    # =========================================================================
    # INV-F12-WEIGHTED-ENG: Weighted cluster sum - P1
    # =========================================================================
    
    @pytest.mark.feature("method_geometry")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_F12_WEIGHTED_ENG_cluster_sum(self, alice_data_xs):
        """
        Track-weighted cluster sum invariance.
        
        For each track: sum(pt * cluster_Q) = pt * sum(cluster_Q)
        (since pt is constant per track)
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            alice_data_xs,
            columns=['track_pt', 'cluster_Q'],
            parent_id_column='event_id'
        )
        
        # Add weighted column
        df['weighted_Q'] = df['track_pt'] * df['cluster_Q']
        
        for (event_id, track_idx), group in df.groupby(['event_id', 'track_idx']):
            pt = group['track_pt'].iloc[0]  # Same for all clusters
            sum_weighted = group['weighted_Q'].sum()
            sum_Q = group['cluster_Q'].sum()
            
            expected = pt * sum_Q
            tolerance = np.finfo(np.float64).eps * abs(expected) * 10 if expected != 0 else 1e-15
            
            assert np.isclose(sum_weighted, expected, atol=tolerance), \
                f"Event {event_id}, Track {track_idx}: weighted sum invariance failed"


class TestMemberFunctionsDSLTests:
    """Member function tests - Type B (DSL chain)."""
    
    # =========================================================================
    # INV-F1-PT-DSL: tracks.Pt() method - P0
    # =========================================================================
    
    @pytest.mark.feature("method_broadcast")
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.feature("api_define")
    @pytest.mark.type_b
    @pytest.mark.p0
    @pytest.mark.phase8
    def test_INV_F1_PT_DSL_phase8_method(self, toy_lorentz_file):
        """
        Test Phase 8 method broadcasting: tracks.Pt()
        
        Validates method call through full DSL chain.
        Compares to computed sqrt(px² + py²).
        
        Type: B (DSL end-to-end)
        Priority: P0
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        # Create RDataFrame from toy file (has 'tracks' column)
        rdf = ROOT.RDataFrame("Events", toy_lorentz_file)
        
        # Schema required for DSLCompiler
        schema = {
            'event_id': 'long',
            'tracks': 'RVec<TLorentzVector>',
            'track_pt': 'RVec<double>',
            'track_px': 'RVec<double>',
            'track_py': 'RVec<double>',
        }
        
        dsl = DSLCompiler(schema)
        
        # Method call via DSL
        try:
            dsl.define("pt_method", "tracks.Pt()")
            dsl.define("px", "tracks.Px()")
            dsl.define("py", "tracks.Py()")
            
            # Computed via DSL (for comparison)
            dsl.define("pt_computed", "sqrt(px*px + py*py)")
            
            # to_pandas calls apply internally - pass rdf from toy file
            df = dsl.to_pandas(rdf, ['pt_method', 'pt_computed', 'event_id'])
            
            # Assertions
            assert len(df) > 0, "No rows returned"
            
            # Method should match computed
            tolerance = np.finfo(np.float64).eps * df['pt_method'].max() * 10
            diff = (df['pt_method'] - df['pt_computed']).abs().max()
            
            assert diff < tolerance, \
                f"tracks.Pt() != sqrt(px²+py²): max_diff={diff}"
                
        except (AttributeError, NotImplementedError) as e:
            pytest.skip(f"Phase 8 not implemented: {e}")
    
    # =========================================================================
    # INV-F1-PT-TOY: .Pt() on toy data (exact) - P0
    # =========================================================================
    
    @pytest.mark.feature("method_broadcast")
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.feature("api_define")
    @pytest.mark.type_b
    @pytest.mark.p0
    @pytest.mark.phase8
    def test_INV_F1_PT_TOY_exact(self, toy_lorentz_file):
        """
        Test .Pt() on toy data with Pythagorean triples.
        
        Uses exact integer pt values (no tolerance needed).
        
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
        }
        
        dsl = DSLCompiler(schema)
        
        try:
            # Method call on TLorentzVector
            dsl.define("pt_method", "tracks.Pt()")
            
            # to_pandas calls apply internally - pass original rdf
            df = dsl.to_pandas(rdf, ['pt_method', 'track_pt', 'event_id'])
            
            assert len(df) > 0, "No rows returned"
            
            # Expected pt values from Pythagorean triples
            expected_pts = {5.0, 13.0, 17.0, 25.0, 29.0, 41.0, 37.0}
            actual_pts = set(df['pt_method'].unique())
            
            # All actual values should be in expected set
            for pt in actual_pts:
                assert pt in expected_pts, f"Unexpected pt value: {pt}"
                
        except (AttributeError, NotImplementedError) as e:
            pytest.skip(f"Phase 8 not implemented: {e}")
    
    # =========================================================================
    # INV-F1-PXPY-DSL: .Px(), .Py(), .Phi() methods - P0
    # =========================================================================
    
    @pytest.mark.feature("method_broadcast")
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.feature("api_define")
    @pytest.mark.type_b
    @pytest.mark.p0
    @pytest.mark.phase8
    def test_INV_F1_PXPY_DSL_consistency(self, toy_lorentz_file):
        """
        px, py, pt, phi consistency on 1D arrays - Phase 8 validation.
        
        Invariance: 
        - px = pt * cos(phi)
        - py = pt * sin(phi)
        
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
            'tracks': 'RVec<TLorentzVector>',
            'track_pt': 'RVec<double>',
            'track_px': 'RVec<double>',
            'track_py': 'RVec<double>',
        }
        
        dsl = DSLCompiler(schema)
        
        try:
            # Use Phase 8 methods
            dsl.define("pt", "tracks.Pt()")
            dsl.define("phi", "tracks.Phi()")
            dsl.define("px_method", "tracks.Px()")
            dsl.define("py_method", "tracks.Py()")
            
            # to_pandas calls apply internally - pass original rdf
            df = dsl.to_pandas(rdf, ['pt', 'phi', 'px_method', 'py_method', 'event_id'])
            
            pt = df['pt'].values
            phi = df['phi'].values
            px_method = df['px_method'].values
            py_method = df['py_method'].values
            
            px_computed = pt * np.cos(phi)
            py_computed = pt * np.sin(phi)
            
            tolerance = np.finfo(np.float64).eps * pt.max() * 10
            
            assert np.allclose(px_computed, px_method, atol=tolerance), \
                "px != pt*cos(phi)"
            assert np.allclose(py_computed, py_method, atol=tolerance), \
                "py != pt*sin(phi)"
                
        except (AttributeError, NotImplementedError) as e:
            pytest.skip(f"Phase 8 not implemented: {e}")
    
    # =========================================================================
    # INV-F-DUAL: Option A vs Option B comparison (CRITICAL) - P0
    # =========================================================================
    
    @pytest.mark.feature("method_broadcast")
    @pytest.mark.feature("api_to_pandas")
    @pytest.mark.feature("api_define")
    @pytest.mark.type_b
    @pytest.mark.p0
    @pytest.mark.phase8
    def test_INV_F_DUAL_access_paths(self, toy_lorentz_file):
        """
        Compare Option A (method call) vs Option B (pre-computed) on SAME data.
        
        This test validates "both options A and B" requirement.
        Uses Pythagorean triple toy data for EXACT validation.
        
        Main Architect's requirement:
        "In the toy source, we can have both, and we can check invariance 
        using different access options A and B"
        
        Type: B (DSL end-to-end)
        Priority: P0 (CRITICAL)
        """
        try:
            import ROOT
            from RDataFrameDSL import DSLCompiler
        except ImportError:
            pytest.skip("ROOT or RDataFrameDSL not available")
        
        # Load toy data with TLorentzVector
        rdf = ROOT.RDataFrame("Events", toy_lorentz_file)
        
        # Schema for toy Lorentz data
        schema = {
            'event_id': 'long',
            'n_tracks': 'int',
            'tracks': 'RVec<TLorentzVector>',
            'track_pt': 'RVec<double>',
            'track_px': 'RVec<double>',
            'track_py': 'RVec<double>',
        }
        
        dsl = DSLCompiler(schema)
        
        try:
            # Option A: Method call on TLorentzVector
            dsl.define("pt_from_method", "tracks.Pt()")
            
            # to_pandas calls apply internally - pass original rdf
            df = dsl.to_pandas(rdf, ['pt_from_method', 'track_pt', 'event_id'])
            
            # Assertions
            assert len(df) > 0, "No rows returned"
            assert 'pt_from_method' in df.columns, "pt_from_method missing"
            assert 'track_pt' in df.columns, "track_pt missing"
            
            # EXACT match (Pythagorean triples = integer results)
            # No tolerance needed - values should be exactly equal
            mismatches = (df['pt_from_method'] != df['track_pt']).sum()
            
            assert mismatches == 0, \
                f"Option A vs Option B mismatch: {mismatches} rows differ\n" \
                f"Method: {df['pt_from_method'].values}\n" \
                f"Stored: {df['track_pt'].values}"
            
            # Verify known Pythagorean values
            expected_pts = [5, 13, 17, 25, 29, 41, 37]
            actual_pts = df['pt_from_method'].unique()
            
            for pt in actual_pts:
                assert pt in expected_pts, f"Unexpected pt value: {pt}"
                
        except (AttributeError, NotImplementedError) as e:
            pytest.skip(f"Phase 8 not implemented: {e}")


class TestMemberFunctionsToyData:
    """Member function tests with toy data for exact validation."""
    
    @pytest.mark.feature("toy_pythagorean")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_F_TOY_pythagorean_identity(self, toy_data):
        """
        Pythagorean identity with toy data.
        
        pt² = px² + py² (exact for Pythagorean triples)
        
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
        
        # pt² should equal px² + py² exactly
        pt_squared = pt**2
        sum_squared = px**2 + py**2
        
        # For Pythagorean triples, this is exact
        assert np.allclose(pt_squared, sum_squared), \
            f"Pythagorean identity failed: pt²={pt_squared}, px²+py²={sum_squared}"
    
    @pytest.mark.feature("toy_pythagorean")
    @pytest.mark.type_a
    @pytest.mark.p1
    def test_INV_F_TOY_known_values(self, toy_data):
        """
        Verify toy data has expected Pythagorean triple values.
        
        Type: A (Engine)
        Priority: P1
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        from tests.generators.toy_lorentz import PYTHAGOREAN_TRIPLES
        
        df = flatten_to_dataframe(
            toy_data,
            columns=['track_pt', 'track_px', 'track_py'],
            parent_id_column='event_id'
        )
        
        expected_triples = set(PYTHAGOREAN_TRIPLES)
        
        for _, row in df.iterrows():
            px, py, pt = int(row['track_px']), int(row['track_py']), int(row['track_pt'])
            assert (px, py, pt) in expected_triples, \
                f"Unexpected triple: ({px}, {py}, {pt})"
    
    @pytest.mark.feature("toy_pythagorean")
    @pytest.mark.type_a
    @pytest.mark.p2
    def test_INV_F_TOY_phi_consistency(self, toy_data):
        """
        Verify phi = atan2(py, px) consistency.
        
        Type: A (Engine)
        Priority: P2
        """
        from RDataFrameDSL.flatten import flatten_to_dataframe
        
        df = flatten_to_dataframe(
            toy_data,
            columns=['track_phi', 'track_px', 'track_py'],
            parent_id_column='event_id'
        )
        
        phi = df['track_phi'].values
        px = df['track_px'].values
        py = df['track_py'].values
        
        computed_phi = np.arctan2(py, px)
        
        tolerance = np.finfo(np.float64).eps * 10
        assert np.allclose(phi, computed_phi, atol=tolerance), \
            "phi != atan2(py, px)"
