"""
Tests for Phase 13.6.G helix trajectory generator.

Tests:
- Helix position calculation (single track)
- Vectorized helix computation (performance)
- ROOT file generation
- Visual validation (curved tracks)
"""

import pytest
import numpy as np
import time


@pytest.fixture(scope="module")
def register_classes():
    """Register ToyTrack/ToyCluster once per module (expensive)."""
    from tests.generators.toy_nd import register_custom_classes
    
    print("\n[fixture] Registering ToyTrack/ToyCluster classes...")
    start = time.perf_counter()
    register_custom_classes()
    elapsed = time.perf_counter() - start
    print(f"[fixture] Class registration completed in {elapsed:.2f}s (ROOT dictionary generation)")
    
    return True


class TestHelixPhysics:
    """Test helix position calculations."""
    
    def test_helix_position_basic(self):
        """helix_position returns (x, y, z) tuple."""
        from tests.generators.toy_nd import helix_position
        
        x, y, z = helix_position(
            pt=1.0,      # 1 GeV
            eta=0.0,     # perpendicular
            phi=0.0,     # along x
            charge=1,    # positive
            r=10.0,      # 10 cm layer
        )
        
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)
    
    def test_helix_position_on_layer(self):
        """Cluster position has correct radius."""
        from tests.generators.toy_nd import helix_position
        
        r_layer = 10.0  # 10 cm
        
        x, y, z = helix_position(
            pt=2.0, eta=0.5, phi=0.7, charge=1, r=r_layer
        )
        
        # Position should be at radius r_layer
        r_actual = np.sqrt(x**2 + y**2)
        assert abs(r_actual - r_layer) < 1e-10, f"Expected r={r_layer}, got {r_actual}"
    
    def test_helix_charge_direction(self):
        """Positive and negative charges curve opposite directions."""
        from tests.generators.toy_nd import helix_position
        
        params = dict(pt=1.0, eta=0.0, phi=0.0, r=10.0)  # 10 cm
        
        x_pos, y_pos, _ = helix_position(charge=+1, **params)
        x_neg, y_neg, _ = helix_position(charge=-1, **params)
        
        # Should curve in opposite directions
        # For phi=0, positive charge curves to negative y, negative to positive y
        assert y_pos != y_neg, "Charges should curve opposite directions"
    
    def test_helix_high_pt_straight(self):
        """High pT tracks are nearly straight."""
        from tests.generators.toy_nd import helix_position
        
        r_layer = 10.0  # 10 cm
        
        # Very high pT = large helix radius = nearly straight
        x, y, z = helix_position(
            pt=100.0,  # 100 GeV - very high
            eta=0.0,
            phi=0.0,   # along x-axis
            charge=1,
            r=r_layer,
        )
        
        # For phi=0, straight track would have y ≈ 0
        assert abs(y) < 1.0, f"High pT track should be nearly straight, got y={y} cm"


class TestVectorizedHelix:
    """Test vectorized helix computation."""
    
    def test_vectorized_shape(self):
        """Vectorized output has correct shape."""
        from tests.generators.toy_nd import compute_helix_positions_vectorized
        
        n_tracks = 5
        n_layers = 7
        
        pt = np.array([1.0, 2.0, 1.5, 3.0, 0.8])
        eta = np.array([0.0, 0.5, -0.3, 0.8, -0.5])
        phi = np.array([0.0, 0.5, 1.0, -0.5, 2.0])
        charge = np.array([1, -1, 1, 1, -1])
        layers = np.array([2.3, 3.1, 3.9, 7.6, 15.0, 24.0, 40.0])  # ITS layers [cm]
        
        x, y, z = compute_helix_positions_vectorized(pt, eta, phi, charge, layers)
        
        assert x.shape == (n_tracks, n_layers)
        assert y.shape == (n_tracks, n_layers)
        assert z.shape == (n_tracks, n_layers)
    
    def test_vectorized_matches_scalar(self):
        """Vectorized result matches scalar computation."""
        from tests.generators.toy_nd import helix_position, compute_helix_positions_vectorized
        
        pt = np.array([1.5])
        eta = np.array([0.3])
        phi = np.array([0.7])
        charge = np.array([1])
        layers = np.array([10.0, 20.0])  # 10 cm, 20 cm
        
        # Vectorized
        x_vec, y_vec, z_vec = compute_helix_positions_vectorized(pt, eta, phi, charge, layers)
        
        # Scalar
        x0, y0, z0 = helix_position(pt[0], eta[0], phi[0], charge[0], layers[0])
        x1, y1, z1 = helix_position(pt[0], eta[0], phi[0], charge[0], layers[1])
        
        assert abs(x_vec[0, 0] - x0) < 1e-10
        assert abs(y_vec[0, 0] - y0) < 1e-10
        assert abs(x_vec[0, 1] - x1) < 1e-10
        assert abs(y_vec[0, 1] - y1) < 1e-10
    
    def test_vectorized_performance(self):
        """Vectorized computation is fast enough."""
        from tests.generators.toy_nd import compute_helix_positions_vectorized, LAYERS_ALL
        
        # Simulate 1000 events × 5 tracks = 5000 tracks
        n_tracks = 5000
        rng = np.random.default_rng(42)
        
        pt = rng.uniform(0.5, 5.0, n_tracks)
        eta = rng.uniform(-1.0, 1.0, n_tracks)
        phi = rng.uniform(-np.pi, np.pi, n_tracks)
        charge = rng.choice([-1, 1], n_tracks)
        
        start = time.perf_counter()
        x, y, z = compute_helix_positions_vectorized(pt, eta, phi, charge, LAYERS_ALL)
        elapsed = time.perf_counter() - start
        
        # Should be very fast (< 0.1s for 5000 tracks × 57 layers)
        assert elapsed < 0.5, f"Vectorized computation too slow: {elapsed:.3f}s"


class TestHelixGenerator:
    """Test ROOT file generation with helix trajectories."""
    
    @pytest.mark.root_serial
    def test_generate_helix_root_creates_file(self, register_classes):
        """generate_helix_root creates valid ROOT file."""
        from tests.generators.toy_nd import generate_helix_root
        import ROOT
        import os
        
        filename = generate_helix_root(n_events=3, seed=42)
        
        assert os.path.exists(filename)
        
        # Open and verify
        f = ROOT.TFile(filename)
        tree = f.Get("Events")
        assert tree is not None
        assert tree.GetEntries() == 3
        
        f.Close()
        os.unlink(filename)
    
    @pytest.mark.root_serial
    def test_generate_helix_root_schema(self, register_classes):
        """Generated file has correct schema."""
        from tests.generators.toy_nd import generate_helix_root
        import ROOT
        import os
        
        filename = generate_helix_root(n_events=2, seed=42)
        
        f = ROOT.TFile(filename)
        tree = f.Get("Events")
        
        # Check branches exist
        branch_names = [b.GetName() for b in tree.GetListOfBranches()]
        assert 'event_id' in branch_names
        assert 'event_weight' in branch_names
        assert 'vertex_x' in branch_names
        assert 'tracks' in branch_names
        
        f.Close()
        os.unlink(filename)
    
    @pytest.mark.root_serial
    def test_generate_helix_root_tracks_have_clusters(self, register_classes):
        """Tracks have clusters with helix positions."""
        from tests.generators.toy_nd import generate_helix_root
        import ROOT
        import os
        
        filename = generate_helix_root(n_events=2, seed=42)
        
        rdf = ROOT.RDataFrame("Events", filename)
        data = rdf.Range(1).AsNumpy(['tracks'])
        
        tracks = data['tracks'][0]
        assert len(tracks) > 0, "Should have tracks"
        
        track = tracks[0]
        assert track.nClusters() > 0, "Track should have clusters"
        
        # Check cluster has position
        cluster = track.clusters()[0]
        x = cluster.getX()
        y = cluster.getY()
        z = cluster.getZ()
        
        # Position should be on first detector layer (~2.3 cm)
        r = np.sqrt(x**2 + y**2)
        assert 2.0 < r < 3.0, f"First cluster should be at ITS layer 1 (~2.3 cm), got r={r:.2f} cm"
        
        os.unlink(filename)
    
    @pytest.mark.root_serial
    def test_generate_helix_root_curved_tracks(self, register_classes):
        """Clusters follow curved trajectories (visual validation)."""
        from tests.generators.toy_nd import generate_helix_root
        import ROOT
        import os
        
        # Use full detector for more visible curvature
        filename = generate_helix_root(
            n_events=1,
            tracks_per_event=(1, 1),  # Single track
            full_detector=True,       # All 57 layers
            pt_range=(1.0, 1.0),      # Fixed pT for predictable curvature
            eta_range=(0.0, 0.0),     # Perpendicular track
            seed=42,
        )
        
        rdf = ROOT.RDataFrame("Events", filename)
        data = rdf.Range(1).AsNumpy(['tracks'])
        
        track = data['tracks'][0][0]
        clusters = track.clusters()
        
        # Extract positions
        x_vals = [c.getX() for c in clusters]
        y_vals = [c.getY() for c in clusters]
        
        # For a curved track, y should not be constant
        # (straight track would have constant y if phi=0)
        y_range = max(y_vals) - min(y_vals)
        assert y_range > 1.0, f"Track should be curved, got y_range={y_range:.2f} cm"
        
        os.unlink(filename)
    
    @pytest.mark.root_serial
    def test_generate_helix_root_performance(self, register_classes):
        """Generator meets performance target (< 2s for 100 events)."""
        from tests.generators.toy_nd import generate_helix_root
        import os
        
        start = time.perf_counter()
        filename = generate_helix_root(n_events=100, seed=42)
        elapsed = time.perf_counter() - start
        
        # Target: should be fast after class registration
        assert elapsed < 2.0, f"Generator too slow: {elapsed:.2f}s"
        
        os.unlink(filename)


# Pytest markers
pytest.mark.feature("helix_generator")
pytest.mark.phase("13.6.G")
