"""
ALICEEventGenerator — Physics-motivated ALICE TPC event generator.

Phase 13.6.B: Test data generation for TTree::Draw stress tests.

Generates helix trajectories with:
- Proper curvature based on pt and charge
- PDG codes for particle identification
- Fast NumPy vectorization (no loops in cluster generation)

Usage:
    from tests.generators.alice_events import ALICEEventGenerator, GeneratorConfig
    
    # Generate test data
    gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
    data = gen.generate_dict(n_events=100)
    
    # Generate ROOT file
    gen.generate_tree('alice_events.root', n_events=1000)

Author: Claude Opus 4.5 (Team2 Coder)
Date: 2026-01-15
Approved by: GPT4, GPT7, GPT8, Gemini2, Claude Opus 4.5, Marian (Main Architect)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GeneratorConfig:
    """
    ALICE TPC event generator configuration.
    
    Physics-motivated but simple generator for:
    - Trajectory visualization
    - PDG-based queries
    - DSL testing and benchmarks
    
    NOT a full detector simulation.
    
    Performance Note:
        Event/track loops are acceptable (~10 tracks/event).
        Cluster loops are NOT acceptable (50-150 points/track × many tracks).
        All cluster generation must be vectorized NumPy.
    """
    
    # Event-level
    mean_tracks_per_event: int = 10
    vertex_x: float = 0.0
    vertex_y: float = 0.0
    vertex_z_sigma: float = 5.0
    
    # Track kinematics
    pt_exp_slope: float = 0.5       # GeV/c (exponential scale)
    pt_min: float = 0.1             # GeV/c (minimum cutoff)
    phi_range: Tuple[float, float] = (0.0, 2 * np.pi)
    pz_over_pt_range: Tuple[float, float] = (-2.0, 2.0)
    b_field: float = 0.5            # Tesla
    
    # Particle species (PDG codes and probabilities)
    # Total must sum to 1.0
    pdg_probabilities: Dict[int, float] = field(default_factory=lambda: {
        # Pions (~68%)
        211: 0.34,      # π+
        -211: 0.34,     # π-
        # Kaons (~16%)
        321: 0.08,      # K+
        -321: 0.08,     # K-
        # Protons (~10%)
        2212: 0.05,     # p
        -2212: 0.05,    # p̄
        # Electrons (~3%)
        11: 0.015,      # e-
        -11: 0.015,     # e+
        # Light nuclei (~1%)
        1000010020: 0.01,    # deuteron
        -1000010020: 0.01,   # anti-deuteron
        1000010030: 0.005,   # tritium
        -1000010030: 0.005,  # anti-tritium
    })  # Sum = 1.00
    
    # Clusters
    clusters_radial_min: float = 0.0        # cm
    clusters_radial_max: float = 250.0      # cm (TPC outer radius)
    clusters_per_track: int = 50            # Fast mode (tests/CI)
    clusters_per_track_dense: int = 150     # Gallery mode (visualization)
    cluster_q_mean: float = 100.0           # ADC units
    cluster_q_sigma: float = 20.0           # ADC spread
    
    # Edge cases
    fraction_empty_events: float = 0.02     # 2% events with 0 tracks
    fraction_short_tracks: float = 0.05     # 5% tracks with <10 clusters
    
    # Reproducibility
    seed: int = 0
    dtype_float: str = "float64"
    dtype_int: str = "int64"
    
    def __post_init__(self):
        """Validate configuration parameters."""
        # Validate physics
        assert self.mean_tracks_per_event >= 0, "mean_tracks_per_event must be >= 0"
        assert self.pt_exp_slope > 0, "pt_exp_slope must be > 0"
        assert self.pt_min >= 0, "pt_min must be >= 0"
        assert self.vertex_z_sigma > 0, "vertex_z_sigma must be > 0"
        assert 0 <= self.fraction_empty_events <= 1
        assert 0 <= self.fraction_short_tracks <= 1
        
        # Validate ranges
        assert self.phi_range[0] < self.phi_range[1]
        assert self.pz_over_pt_range[0] < self.pz_over_pt_range[1]
        assert self.clusters_radial_min >= 0
        assert self.clusters_radial_max > self.clusters_radial_min
        assert self.clusters_per_track > 0
        assert self.clusters_per_track_dense >= self.clusters_per_track
        assert self.b_field > 0
        
        # Validate PDG probabilities
        prob_sum = sum(self.pdg_probabilities.values())
        assert 0.99 <= prob_sum <= 1.01, \
            f"PDG probabilities sum to {prob_sum}, must be ~1.0"
        
        # Validate dtypes
        assert self.dtype_float in ["float32", "float64"]
        assert self.dtype_int in ["int32", "int64"]


# =============================================================================
# Generator
# =============================================================================

class ALICEEventGenerator:
    """
    Physics-motivated ALICE TPC event generator.
    
    Generates helix trajectories with:
    - Proper curvature based on pt and charge
    - PDG codes for particle identification
    - Fast NumPy vectorization (no loops in cluster generation)
    
    Performance:
        - XS (100 events): <1 second
        - S (1k events): <5 seconds
        - M (10k events): <30 seconds
    
    Example:
        >>> gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
        >>> data = gen.generate_dict(n_events=100)
        >>> gen.generate_tree('output.root', n_events=1000)
    """
    
    # PDG mass table [GeV/c²] (for reference, not used in generation)
    PDG_MASS = {
        11: 0.000511,       # electron
        211: 0.13957,       # pion
        321: 0.49368,       # kaon
        2212: 0.93827,      # proton
        1000010020: 1.8756, # deuteron
        1000010030: 2.8089, # tritium
    }
    
    # PDG charge table (unit charges only for toy model)
    PDG_CHARGE = {
        11: -1,      # e-
        -11: +1,     # e+
        211: +1,     # π+
        -211: -1,    # π-
        321: +1,     # K+
        -321: -1,    # K-
        2212: +1,    # p
        -2212: -1,   # p̄
        1000010020: +1,   # d (treated as +1 for curvature)
        -1000010020: -1,  # d̄
        1000010030: +1,   # t (treated as +1 for curvature)
        -1000010030: -1,  # t̄
    }
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize generator with configuration.
        
        Args:
            config: Generator configuration (uses defaults if None)
        """
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Cache PDG arrays for vectorized sampling
        self._pdg_codes = np.array(list(self.config.pdg_probabilities.keys()))
        self._pdg_probs = np.array(list(self.config.pdg_probabilities.values()))
    
    def generate_dict(self, n_events: int, dense: bool = False) -> Dict[str, np.ndarray]:
        """
        Generate events as dictionary (for tests without ROOT I/O).
        
        Args:
            n_events: Number of events to generate
            dense: If True, use clusters_per_track_dense (for visualization)
        
        Returns:
            Dict with event, track, and cluster columns matching
            ROOT.RDataFrame.AsNumpy() format.
        
        Output Schema:
            Event-level (scalars):
                event_id, vertex_x, vertex_y, vertex_z, n_tracks
            Track-level (1D RVec):
                track_pt, track_phi, track_pz_over_pt, track_pdgcode,
                track_charge, track_px, track_py, track_pz
            Cluster-level (2D RVec<RVec>):
                cluster_x, cluster_y, cluster_z, cluster_r, cluster_Q
        """
        n_clusters = (self.config.clusters_per_track_dense if dense 
                      else self.config.clusters_per_track)
        
        dtype_float = np.dtype(self.config.dtype_float)
        dtype_int = np.dtype(self.config.dtype_int)
        
        # Initialize output containers
        # Event-level (scalars)
        event_ids = np.empty(n_events, dtype=dtype_int)
        vertex_x = np.empty(n_events, dtype=dtype_float)
        vertex_y = np.empty(n_events, dtype=dtype_float)
        vertex_z = np.empty(n_events, dtype=dtype_float)
        n_tracks_per_event = np.empty(n_events, dtype=dtype_int)
        
        # Track-level (1D RVec) - object arrays of arrays
        track_pt = np.empty(n_events, dtype=object)
        track_phi = np.empty(n_events, dtype=object)
        track_pz_over_pt = np.empty(n_events, dtype=object)
        track_pdgcode = np.empty(n_events, dtype=object)
        track_charge = np.empty(n_events, dtype=object)
        track_px = np.empty(n_events, dtype=object)
        track_py = np.empty(n_events, dtype=object)
        track_pz = np.empty(n_events, dtype=object)
        
        # Cluster-level (2D RVec<RVec>) - object arrays of object arrays
        cluster_x = np.empty(n_events, dtype=object)
        cluster_y = np.empty(n_events, dtype=object)
        cluster_z = np.empty(n_events, dtype=object)
        cluster_r = np.empty(n_events, dtype=object)
        cluster_Q = np.empty(n_events, dtype=object)
        
        # Generate events (loop over events is acceptable)
        for evt in range(n_events):
            event_ids[evt] = evt
            
            # Vertex position
            vx = self.config.vertex_x
            vy = self.config.vertex_y
            vz = self.rng.normal(0.0, self.config.vertex_z_sigma)
            
            vertex_x[evt] = vx
            vertex_y[evt] = vy
            vertex_z[evt] = vz
            
            # Number of tracks (Poisson, with edge case for empty events)
            if self.rng.random() < self.config.fraction_empty_events:
                n_trk = 0
            else:
                n_trk = self.rng.poisson(self.config.mean_tracks_per_event)
            
            n_tracks_per_event[evt] = n_trk
            
            # Generate tracks for this event
            if n_trk == 0:
                # Empty event
                track_pt[evt] = np.array([], dtype=dtype_float)
                track_phi[evt] = np.array([], dtype=dtype_float)
                track_pz_over_pt[evt] = np.array([], dtype=dtype_float)
                track_pdgcode[evt] = np.array([], dtype=dtype_int)
                track_charge[evt] = np.array([], dtype=dtype_int)
                track_px[evt] = np.array([], dtype=dtype_float)
                track_py[evt] = np.array([], dtype=dtype_float)
                track_pz[evt] = np.array([], dtype=dtype_float)
                
                cluster_x[evt] = np.array([], dtype=object)
                cluster_y[evt] = np.array([], dtype=object)
                cluster_z[evt] = np.array([], dtype=object)
                cluster_r[evt] = np.array([], dtype=object)
                cluster_Q[evt] = np.array([], dtype=object)
            else:
                # Generate track kinematics (vectorized for all tracks in event)
                trk_pt, trk_phi, trk_pz_pt, trk_pdg = self._generate_tracks_vectorized(n_trk)
                
                # Get charges from PDG codes
                trk_charge = np.array([self.PDG_CHARGE.get(p, 1) for p in trk_pdg], 
                                       dtype=dtype_int)
                
                # Compute momentum components
                trk_px = trk_pt * np.cos(trk_phi)
                trk_py = trk_pt * np.sin(trk_phi)
                trk_pz = trk_pt * trk_pz_pt
                
                # Store track arrays
                track_pt[evt] = trk_pt.astype(dtype_float)
                track_phi[evt] = trk_phi.astype(dtype_float)
                track_pz_over_pt[evt] = trk_pz_pt.astype(dtype_float)
                track_pdgcode[evt] = trk_pdg.astype(dtype_int)
                track_charge[evt] = trk_charge
                track_px[evt] = trk_px.astype(dtype_float)
                track_py[evt] = trk_py.astype(dtype_float)
                track_pz[evt] = trk_pz.astype(dtype_float)
                
                # Generate clusters for each track (2D structure)
                evt_cluster_x = np.empty(n_trk, dtype=object)
                evt_cluster_y = np.empty(n_trk, dtype=object)
                evt_cluster_z = np.empty(n_trk, dtype=object)
                evt_cluster_r = np.empty(n_trk, dtype=object)
                evt_cluster_Q = np.empty(n_trk, dtype=object)
                
                for trk in range(n_trk):
                    # Determine number of clusters (edge case: short tracks)
                    if self.rng.random() < self.config.fraction_short_tracks:
                        n_cls = self.rng.integers(1, 10)
                    else:
                        n_cls = n_clusters
                    
                    # VECTORIZED cluster generation (CRITICAL - no loops here)
                    cx, cy, cz = self._generate_helix_clusters(
                        vx, vy, vz,
                        trk_pt[trk], trk_phi[trk], trk_pz_pt[trk],
                        trk_charge[trk], n_cls
                    )
                    
                    # Compute radial distance (vectorized)
                    cr = np.sqrt(cx**2 + cy**2)
                    
                    # Generate cluster charge (vectorized)
                    cQ = self.rng.normal(
                        self.config.cluster_q_mean,
                        self.config.cluster_q_sigma,
                        n_cls
                    )
                    cQ = np.maximum(cQ, 1.0)  # Ensure positive
                    
                    evt_cluster_x[trk] = cx.astype(dtype_float)
                    evt_cluster_y[trk] = cy.astype(dtype_float)
                    evt_cluster_z[trk] = cz.astype(dtype_float)
                    evt_cluster_r[trk] = cr.astype(dtype_float)
                    evt_cluster_Q[trk] = cQ.astype(dtype_float)
                
                cluster_x[evt] = evt_cluster_x
                cluster_y[evt] = evt_cluster_y
                cluster_z[evt] = evt_cluster_z
                cluster_r[evt] = evt_cluster_r
                cluster_Q[evt] = evt_cluster_Q
        
        return {
            # Event-level
            'event_id': event_ids,
            'vertex_x': vertex_x,
            'vertex_y': vertex_y,
            'vertex_z': vertex_z,
            'n_tracks': n_tracks_per_event,
            # Track-level (1D RVec)
            'track_pt': track_pt,
            'track_phi': track_phi,
            'track_pz_over_pt': track_pz_over_pt,
            'track_pdgcode': track_pdgcode,
            'track_charge': track_charge,
            'track_px': track_px,
            'track_py': track_py,
            'track_pz': track_pz,
            # Cluster-level (2D RVec<RVec>)
            'cluster_x': cluster_x,
            'cluster_y': cluster_y,
            'cluster_z': cluster_z,
            'cluster_r': cluster_r,
            'cluster_Q': cluster_Q,
        }
    
    def generate_tree(self, filename: str, n_events: int, dense: bool = False) -> str:
        """
        Generate ROOT TTree file.
        
        Args:
            filename: Output ROOT file path
            n_events: Number of events to generate
            dense: If True, use more clusters per track (for visualization)
        
        Returns:
            Path to created file (for verification)
        """
        import ROOT
        
        # Generate data first
        data = self.generate_dict(n_events, dense=dense)
        
        # Create ROOT file and tree
        f = ROOT.TFile(filename, "RECREATE")
        tree = ROOT.TTree("Events", "ALICE TPC Events (generated)")
        
        # Setup branches
        # Event-level branches
        event_id = np.zeros(1, dtype=np.int64)
        vertex_x_arr = np.zeros(1, dtype=np.float64)
        vertex_y_arr = np.zeros(1, dtype=np.float64)
        vertex_z_arr = np.zeros(1, dtype=np.float64)
        n_tracks_arr = np.zeros(1, dtype=np.int64)
        
        tree.Branch("event_id", event_id, "event_id/L")
        tree.Branch("vertex_x", vertex_x_arr, "vertex_x/D")
        tree.Branch("vertex_y", vertex_y_arr, "vertex_y/D")
        tree.Branch("vertex_z", vertex_z_arr, "vertex_z/D")
        tree.Branch("n_tracks", n_tracks_arr, "n_tracks/L")
        
        # Track-level branches (RVec)
        track_pt_vec = ROOT.std.vector('double')()
        track_phi_vec = ROOT.std.vector('double')()
        track_pz_over_pt_vec = ROOT.std.vector('double')()
        track_pdgcode_vec = ROOT.std.vector('long')()
        track_charge_vec = ROOT.std.vector('long')()
        track_px_vec = ROOT.std.vector('double')()
        track_py_vec = ROOT.std.vector('double')()
        track_pz_vec = ROOT.std.vector('double')()
        
        tree.Branch("track_pt", track_pt_vec)
        tree.Branch("track_phi", track_phi_vec)
        tree.Branch("track_pz_over_pt", track_pz_over_pt_vec)
        tree.Branch("track_pdgcode", track_pdgcode_vec)
        tree.Branch("track_charge", track_charge_vec)
        tree.Branch("track_px", track_px_vec)
        tree.Branch("track_py", track_py_vec)
        tree.Branch("track_pz", track_pz_vec)
        
        # Cluster-level branches (RVec<RVec>)
        cluster_x_vec = ROOT.std.vector('ROOT::VecOps::RVec<double>')()
        cluster_y_vec = ROOT.std.vector('ROOT::VecOps::RVec<double>')()
        cluster_z_vec = ROOT.std.vector('ROOT::VecOps::RVec<double>')()
        cluster_r_vec = ROOT.std.vector('ROOT::VecOps::RVec<double>')()
        cluster_Q_vec = ROOT.std.vector('ROOT::VecOps::RVec<double>')()
        
        tree.Branch("cluster_x", cluster_x_vec)
        tree.Branch("cluster_y", cluster_y_vec)
        tree.Branch("cluster_z", cluster_z_vec)
        tree.Branch("cluster_r", cluster_r_vec)
        tree.Branch("cluster_Q", cluster_Q_vec)
        
        # Fill tree
        for evt in range(n_events):
            # Event-level
            event_id[0] = data['event_id'][evt]
            vertex_x_arr[0] = data['vertex_x'][evt]
            vertex_y_arr[0] = data['vertex_y'][evt]
            vertex_z_arr[0] = data['vertex_z'][evt]
            n_tracks_arr[0] = data['n_tracks'][evt]
            
            # Track-level
            track_pt_vec.clear()
            track_phi_vec.clear()
            track_pz_over_pt_vec.clear()
            track_pdgcode_vec.clear()
            track_charge_vec.clear()
            track_px_vec.clear()
            track_py_vec.clear()
            track_pz_vec.clear()
            
            for val in data['track_pt'][evt]:
                track_pt_vec.push_back(float(val))
            for val in data['track_phi'][evt]:
                track_phi_vec.push_back(float(val))
            for val in data['track_pz_over_pt'][evt]:
                track_pz_over_pt_vec.push_back(float(val))
            for val in data['track_pdgcode'][evt]:
                track_pdgcode_vec.push_back(int(val))
            for val in data['track_charge'][evt]:
                track_charge_vec.push_back(int(val))
            for val in data['track_px'][evt]:
                track_px_vec.push_back(float(val))
            for val in data['track_py'][evt]:
                track_py_vec.push_back(float(val))
            for val in data['track_pz'][evt]:
                track_pz_vec.push_back(float(val))
            
            # Cluster-level (2D)
            cluster_x_vec.clear()
            cluster_y_vec.clear()
            cluster_z_vec.clear()
            cluster_r_vec.clear()
            cluster_Q_vec.clear()
            
            n_trk = data['n_tracks'][evt]
            for trk in range(n_trk):
                # Create inner RVec for each track
                inner_x = ROOT.ROOT.VecOps.RVec('double')()
                inner_y = ROOT.ROOT.VecOps.RVec('double')()
                inner_z = ROOT.ROOT.VecOps.RVec('double')()
                inner_r = ROOT.ROOT.VecOps.RVec('double')()
                inner_Q = ROOT.ROOT.VecOps.RVec('double')()
                
                for val in data['cluster_x'][evt][trk]:
                    inner_x.push_back(float(val))
                for val in data['cluster_y'][evt][trk]:
                    inner_y.push_back(float(val))
                for val in data['cluster_z'][evt][trk]:
                    inner_z.push_back(float(val))
                for val in data['cluster_r'][evt][trk]:
                    inner_r.push_back(float(val))
                for val in data['cluster_Q'][evt][trk]:
                    inner_Q.push_back(float(val))
                
                cluster_x_vec.push_back(inner_x)
                cluster_y_vec.push_back(inner_y)
                cluster_z_vec.push_back(inner_z)
                cluster_r_vec.push_back(inner_r)
                cluster_Q_vec.push_back(inner_Q)
            
            tree.Fill()
        
        # Write and close
        tree.Write()
        f.Close()
        
        return filename
    
    # =========================================================================
    # Private helper methods
    # =========================================================================
    
    def _generate_tracks_vectorized(self, n_tracks: int) -> Tuple[np.ndarray, ...]:
        """
        Generate track kinematics for all tracks in an event (vectorized).
        
        Args:
            n_tracks: Number of tracks to generate
        
        Returns:
            (pt, phi, pz_over_pt, pdgcode) arrays
        """
        # Sample pt from exponential with minimum cutoff
        # Using shifted exponential: pt = pt_min + exponential(scale)
        pt = self.config.pt_min + self.rng.exponential(
            self.config.pt_exp_slope, size=n_tracks
        )
        
        # Sample phi uniformly
        phi = self.rng.uniform(
            self.config.phi_range[0], 
            self.config.phi_range[1], 
            size=n_tracks
        )
        
        # Sample pz/pt uniformly
        pz_over_pt = self.rng.uniform(
            self.config.pz_over_pt_range[0],
            self.config.pz_over_pt_range[1],
            size=n_tracks
        )
        
        # Sample PDG codes
        pdgcode = self.rng.choice(
            self._pdg_codes, 
            size=n_tracks, 
            p=self._pdg_probs
        )
        
        return pt, phi, pz_over_pt, pdgcode
    
    def _generate_helix_clusters(
        self,
        vertex_x: float,
        vertex_y: float,
        vertex_z: float,
        pt: float,
        phi: float,
        pz_over_pt: float,
        charge: int,
        n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate cluster positions using VECTORIZED NumPy operations.
        
        CRITICAL: No Python loops. All operations on full arrays.
        
        Physics:
            - Helix radius R = pt / (0.3 * q * B) [in cm when pt in GeV, B in T]
            - Charge determines curvature direction
            - z-position depends on pseudorapidity
        
        Note:
            This uses simplified arc-length approximation (arc ≈ radius).
            Acceptable for visualization purposes.
        
        Args:
            vertex_x, vertex_y, vertex_z: Collision vertex position [cm]
            pt: Transverse momentum [GeV/c]
            phi: Azimuthal angle [rad]
            pz_over_pt: pz/pt ratio (related to pseudorapidity)
            charge: Particle charge (±1, used for curvature direction)
            n_clusters: Number of clusters to generate
        
        Returns:
            (cluster_x, cluster_y, cluster_z) arrays [cm]
        """
        # Helix radius [cm]
        # R = pt / (0.3 * |q| * B) where pt in GeV, B in Tesla gives R in meters
        # Multiply by 100 to convert to cm
        R = pt / (0.3 * abs(charge) * self.config.b_field) * 100.0
        
        # Helix center (perpendicular to initial momentum direction)
        sign = -charge
        cx = vertex_x + sign * R * np.sin(phi)
        cy = vertex_y - sign * R * np.cos(phi)
        
        # ✅ VECTORIZED: Generate all radii at once (no loop)
        radii = np.linspace(
            self.config.clusters_radial_min,
            self.config.clusters_radial_max,
            n_clusters
        )
        
        # ✅ VECTORIZED: Compute all arc lengths at once
        # Simplified: arc length ≈ radial distance (acceptable for visualization)
        arc_lengths = radii
        
        # ✅ VECTORIZED: Compute all angles at once (array operation)
        theta_helix = arc_lengths / R * sign
        
        # ✅ VECTORIZED: Compute all x,y positions at once (broadcasting)
        cluster_x = cx - sign * R * np.sin(phi + theta_helix)
        cluster_y = cy + sign * R * np.cos(phi + theta_helix)
        
        # ✅ VECTORIZED: Compute all z positions at once
        eta = np.arcsinh(pz_over_pt)
        cluster_z = vertex_z + arc_lengths * np.tanh(eta)
        
        return cluster_x, cluster_y, cluster_z


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for generating test data."""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(
        description='Generate ALICE TPC test data'
    )
    parser.add_argument(
        '--size', choices=['XS', 'S', 'M', 'L'], default='S',
        help='Data size preset (default: S)'
    )
    parser.add_argument(
        '--n-events', type=int, default=None,
        help='Number of events (overrides --size)'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed for reproducibility (default: 0)'
    )
    parser.add_argument(
        '--clusters', type=int, default=50,
        help='Clusters per track (default: 50, use 150 for gallery)'
    )
    parser.add_argument(
        '--output', type=str, default='alice_events.root',
        help='Output ROOT file path'
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help='Run performance benchmark'
    )
    
    args = parser.parse_args()
    
    # Size presets
    size_map = {
        'XS': 100,
        'S': 1000,
        'M': 10000,
        'L': 100000,
    }
    
    n_events = args.n_events or size_map[args.size]
    dense = args.clusters >= 150
    
    # Create generator
    config = GeneratorConfig(
        seed=args.seed,
        clusters_per_track=args.clusters if not dense else 50,
        clusters_per_track_dense=args.clusters if dense else 150,
    )
    gen = ALICEEventGenerator(config=config)
    
    print(f"Generating {n_events} events...")
    print(f"  Clusters per track: {args.clusters}")
    print(f"  Seed: {args.seed}")
    print(f"  Output: {args.output}")
    
    start = time.time()
    gen.generate_tree(args.output, n_events, dense=dense)
    elapsed = time.time() - start
    
    print(f"Done in {elapsed:.2f} seconds")
    
    # Performance targets
    targets = {'XS': 1.0, 'S': 5.0, 'M': 30.0, 'L': 300.0}
    target = targets.get(args.size, 999)
    
    if elapsed > target:
        print(f"⚠️ WARNING: Exceeded target time ({target}s)")
    else:
        print(f"✅ Within target time ({target}s)")
    
    if args.benchmark:
        # Run benchmark for all sizes
        print("\n=== Benchmark ===")
        for size, n in size_map.items():
            if n > n_events * 2:  # Skip sizes much larger than requested
                continue
            start = time.time()
            gen.generate_dict(n)
            elapsed = time.time() - start
            status = "✅" if elapsed < targets[size] else "❌"
            print(f"{size} ({n:,} events): {elapsed:.2f}s {status}")


if __name__ == "__main__":
    main()
