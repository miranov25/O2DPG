"""
Tests for ALICEEventGenerator.

Phase 13.6.B: Validates physics-motivated test data generation.

Tests:
- GEN1-GEN5: Generation correctness
- PERF1-PERF3: Performance targets
- PHYS1-PHYS5: Physics validation
"""

import pytest
import numpy as np
import time
import os
import tempfile

from tests.generators.alice_events import ALICEEventGenerator, GeneratorConfig


# =============================================================================
# Generation Correctness Tests (GEN1-GEN5)
# =============================================================================

class TestGenerationCorrectness:
    """Tests for basic generation correctness."""
    
    def test_GEN1_generate_dict_returns_all_columns(self):
        """generate_dict() returns all required columns."""
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
        data = gen.generate_dict(n_events=10)
        
        # Event-level columns
        assert 'event_id' in data
        assert 'vertex_x' in data
        assert 'vertex_y' in data
        assert 'vertex_z' in data
        assert 'n_tracks' in data
        
        # Track-level columns (1D RVec)
        assert 'track_pt' in data
        assert 'track_phi' in data
        assert 'track_pz_over_pt' in data
        assert 'track_pdgcode' in data
        assert 'track_charge' in data
        assert 'track_px' in data
        assert 'track_py' in data
        assert 'track_pz' in data
        
        # Cluster-level columns (2D RVec<RVec>)
        assert 'cluster_x' in data
        assert 'cluster_y' in data
        assert 'cluster_z' in data
        assert 'cluster_r' in data
        assert 'cluster_Q' in data
    
    def test_GEN2_event_count_correct(self):
        """Correct number of events generated."""
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
        
        for n in [1, 10, 100]:
            data = gen.generate_dict(n_events=n)
            assert len(data['event_id']) == n
            assert len(data['vertex_z']) == n
            assert len(data['track_pt']) == n
            assert len(data['cluster_x']) == n
    
    def test_GEN3_reproducible_with_seed(self):
        """Same seed produces identical results."""
        gen1 = ALICEEventGenerator(config=GeneratorConfig(seed=123))
        gen2 = ALICEEventGenerator(config=GeneratorConfig(seed=123))
        
        data1 = gen1.generate_dict(n_events=10)
        data2 = gen2.generate_dict(n_events=10)
        
        # Check event-level
        np.testing.assert_array_equal(data1['event_id'], data2['event_id'])
        np.testing.assert_array_equal(data1['vertex_z'], data2['vertex_z'])
        np.testing.assert_array_equal(data1['n_tracks'], data2['n_tracks'])
        
        # Check track-level (first event)
        if len(data1['track_pt'][0]) > 0:
            np.testing.assert_array_equal(data1['track_pt'][0], data2['track_pt'][0])
            np.testing.assert_array_equal(data1['track_pdgcode'][0], data2['track_pdgcode'][0])
    
    def test_GEN4_different_seeds_different_results(self):
        """Different seeds produce different results."""
        gen1 = ALICEEventGenerator(config=GeneratorConfig(seed=1))
        gen2 = ALICEEventGenerator(config=GeneratorConfig(seed=2))
        
        data1 = gen1.generate_dict(n_events=100)
        data2 = gen2.generate_dict(n_events=100)
        
        # Should have different vertex_z values
        assert not np.allclose(data1['vertex_z'], data2['vertex_z'])
    
    def test_GEN5_dtype_consistency(self):
        """Output dtypes match configuration."""
        config = GeneratorConfig(seed=42, dtype_float='float64', dtype_int='int64')
        gen = ALICEEventGenerator(config=config)
        data = gen.generate_dict(n_events=10)
        
        # Event-level
        assert data['event_id'].dtype == np.int64
        assert data['vertex_z'].dtype == np.float64
        
        # Track-level (check first non-empty event)
        for evt in range(10):
            if len(data['track_pt'][evt]) > 0:
                assert data['track_pt'][evt].dtype == np.float64
                assert data['track_pdgcode'][evt].dtype == np.int64
                break


# =============================================================================
# Performance Tests (PERF1-PERF3)
# =============================================================================

class TestPerformance:
    """Performance target validation."""
    
    def test_PERF1_XS_under_1_second(self):
        """XS (100 events) generates in <1 second."""
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=0))
        
        start = time.time()
        gen.generate_dict(n_events=100)
        elapsed = time.time() - start
        
        assert elapsed < 1.0, f"XS took {elapsed:.2f}s, target <1.0s"
    
    def test_PERF2_S_under_5_seconds(self):
        """S (1000 events) generates in <5 seconds."""
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=0))
        
        start = time.time()
        gen.generate_dict(n_events=1000)
        elapsed = time.time() - start
        
        assert elapsed < 5.0, f"S took {elapsed:.2f}s, target <5.0s"
    
    @pytest.mark.slow
    def test_PERF3_M_under_30_seconds(self):
        """M (10000 events) generates in <30 seconds."""
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=0))
        
        start = time.time()
        gen.generate_dict(n_events=10000)
        elapsed = time.time() - start
        
        assert elapsed < 30.0, f"M took {elapsed:.2f}s, target <30.0s"


# =============================================================================
# Physics Validation Tests (PHYS1-PHYS5)
# =============================================================================

class TestPhysicsValidation:
    """Physics parameter validation."""
    
    def test_PHYS1_vertex_z_distribution(self):
        """Vertex z follows Gaussian distribution."""
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=42, vertex_z_sigma=5.0))
        data = gen.generate_dict(n_events=1000)
        
        vz = data['vertex_z']
        
        # Mean should be ~0
        assert abs(np.mean(vz)) < 0.5, f"Mean vertex_z = {np.mean(vz)}, expected ~0"
        
        # Std should be ~5
        assert 4.0 < np.std(vz) < 6.0, f"Std vertex_z = {np.std(vz)}, expected ~5"
    
    def test_PHYS2_pt_exponential_distribution(self):
        """Track pt follows exponential distribution."""
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
        data = gen.generate_dict(n_events=1000)
        
        # Collect all pt values
        all_pt = np.concatenate([data['track_pt'][i] for i in range(len(data['track_pt']))
                                 if len(data['track_pt'][i]) > 0])
        
        # Should have exponential-like shape (most tracks at low pt)
        low_pt = np.sum(all_pt < 1.0)
        high_pt = np.sum(all_pt > 2.0)
        
        assert low_pt > high_pt, "Expected more low-pt tracks than high-pt"
        
        # All should be >= pt_min
        assert np.all(all_pt >= gen.config.pt_min)
    
    def test_PHYS3_pdg_codes_valid(self):
        """PDG codes are from valid set."""
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
        data = gen.generate_dict(n_events=100)
        
        valid_pdgs = set(gen.config.pdg_probabilities.keys())
        
        for evt in range(len(data['track_pdgcode'])):
            for pdg in data['track_pdgcode'][evt]:
                assert pdg in valid_pdgs, f"Invalid PDG code: {pdg}"
    
    def test_PHYS4_charge_matches_pdg(self):
        """Track charge matches PDG code."""
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
        data = gen.generate_dict(n_events=100)
        
        for evt in range(len(data['track_pdgcode'])):
            for i, pdg in enumerate(data['track_pdgcode'][evt]):
                charge = data['track_charge'][evt][i]
                expected = gen.PDG_CHARGE.get(pdg, 1)
                assert charge == expected, \
                    f"PDG {pdg}: charge={charge}, expected={expected}"
    
    def test_PHYS5_cluster_radii_in_range(self):
        """Cluster radii are within TPC range."""
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
        data = gen.generate_dict(n_events=100)
        
        for evt in range(len(data['cluster_r'])):
            for trk in range(len(data['cluster_r'][evt])):
                r = data['cluster_r'][evt][trk]
                if len(r) > 0:
                    # Allow some tolerance for helix geometry
                    assert np.all(r >= -10), f"Negative radius found"
                    # Outer radius can exceed 250 due to helix curvature
                    assert np.all(r < 500), f"Radius too large: {np.max(r)}"


# =============================================================================
# Edge Case Tests (EDGE1-EDGE3)
# =============================================================================

class TestEdgeCases:
    """Edge case handling."""
    
    def test_EDGE1_empty_events_exist(self):
        """Some events have zero tracks (edge case)."""
        config = GeneratorConfig(seed=42, fraction_empty_events=0.1)
        gen = ALICEEventGenerator(config=config)
        data = gen.generate_dict(n_events=100)
        
        empty_count = np.sum(data['n_tracks'] == 0)
        
        # With 10% empty fraction, expect some empty events
        assert empty_count > 0, "Expected some empty events"
        assert empty_count < 30, "Too many empty events"
    
    def test_EDGE2_empty_event_structure(self):
        """Empty events have correct structure."""
        config = GeneratorConfig(seed=42, fraction_empty_events=0.5)
        gen = ALICEEventGenerator(config=config)
        data = gen.generate_dict(n_events=100)
        
        for evt in range(len(data['n_tracks'])):
            if data['n_tracks'][evt] == 0:
                # Track arrays should be empty
                assert len(data['track_pt'][evt]) == 0
                assert len(data['track_pdgcode'][evt]) == 0
                
                # Cluster arrays should be empty
                assert len(data['cluster_x'][evt]) == 0
    
    def test_EDGE3_short_tracks_exist(self):
        """Some tracks have fewer clusters (edge case)."""
        config = GeneratorConfig(seed=42, fraction_short_tracks=0.2)
        gen = ALICEEventGenerator(config=config)
        data = gen.generate_dict(n_events=100)
        
        short_count = 0
        total_tracks = 0
        
        for evt in range(len(data['cluster_x'])):
            for trk in range(len(data['cluster_x'][evt])):
                n_cls = len(data['cluster_x'][evt][trk])
                total_tracks += 1
                if n_cls < 10:
                    short_count += 1
        
        # With 20% short fraction, expect some short tracks
        if total_tracks > 0:
            fraction = short_count / total_tracks
            assert fraction > 0.05, f"Expected some short tracks, got {fraction:.1%}"


# =============================================================================
# ROOT I/O Tests (ROOT1-ROOT2)
# =============================================================================

@pytest.mark.skipif(
    not os.environ.get('ROOTSYS'),
    reason="ROOT not available"
)
class TestROOTIO:
    """ROOT file I/O tests."""
    
    def test_ROOT1_generate_tree_creates_file(self):
        """generate_tree() creates valid ROOT file."""
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
        
        with tempfile.NamedTemporaryFile(suffix='.root', delete=False) as f:
            filename = f.name
        
        try:
            result = gen.generate_tree(filename, n_events=10)
            
            assert result == filename
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
        finally:
            if os.path.exists(filename):
                os.unlink(filename)
    
    def test_ROOT2_tree_readable(self):
        """Generated ROOT file is readable."""
        import ROOT
        
        gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
        
        with tempfile.NamedTemporaryFile(suffix='.root', delete=False) as f:
            filename = f.name
        
        try:
            gen.generate_tree(filename, n_events=10)
            
            # Read back
            f = ROOT.TFile(filename, "READ")
            tree = f.Get("Events")
            
            assert tree is not None
            assert tree.GetEntries() == 10
            
            # Check branches exist
            assert tree.GetBranch("event_id") is not None
            assert tree.GetBranch("track_pt") is not None
            assert tree.GetBranch("cluster_x") is not None
            
            f.Close()
        finally:
            if os.path.exists(filename):
                os.unlink(filename)


# =============================================================================
# Configuration Validation Tests (CONFIG1-CONFIG3)
# =============================================================================

class TestConfigValidation:
    """Configuration validation tests."""
    
    def test_CONFIG1_invalid_pt_slope_rejected(self):
        """Invalid pt_exp_slope raises error."""
        with pytest.raises(AssertionError):
            GeneratorConfig(pt_exp_slope=0)
        
        with pytest.raises(AssertionError):
            GeneratorConfig(pt_exp_slope=-1)
    
    def test_CONFIG2_invalid_pdg_probs_rejected(self):
        """PDG probabilities must sum to ~1."""
        with pytest.raises(AssertionError, match="sum to"):
            GeneratorConfig(pdg_probabilities={211: 0.5})  # Only 50%
    
    def test_CONFIG3_custom_config_works(self):
        """Custom configuration is respected."""
        config = GeneratorConfig(
            seed=999,
            mean_tracks_per_event=5,
            clusters_per_track=20,
        )
        gen = ALICEEventGenerator(config=config)
        data = gen.generate_dict(n_events=100)
        
        # Check tracks per event is around 5
        avg_tracks = np.mean(data['n_tracks'][data['n_tracks'] > 0])
        assert 3 < avg_tracks < 8, f"Expected ~5 tracks, got {avg_tracks:.1f}"
        
        # Check clusters per track is 20
        for evt in range(len(data['cluster_x'])):
            for trk in range(len(data['cluster_x'][evt])):
                n_cls = len(data['cluster_x'][evt][trk])
                # Most tracks should have 20 clusters (unless short)
                if n_cls >= 10:  # Not a short track
                    assert n_cls == 20, f"Expected 20 clusters, got {n_cls}"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
