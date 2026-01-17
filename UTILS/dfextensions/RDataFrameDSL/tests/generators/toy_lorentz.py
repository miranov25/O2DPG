"""
Toy Event Generator with TLorentzVector

Phase 13.6.B.fix: Invariance Tests - Dual Access Path Validation

Generates deterministic data with BOTH access paths:
- Option A: tracks (RVec<TLorentzVector>) with .Pt(), .Px(), .Py() methods
- Option B: track_pt, track_px, track_py (RVec<double>) pre-computed arrays

Uses Pythagorean triples for EXACT integer pt values (no tolerance needed).

Pythagorean Triples Used:
    Track 0: px=3,  py=4  → pt=5   (3-4-5 triangle)
    Track 1: px=5,  py=12 → pt=13  (5-12-13 triangle)
    Track 2: px=8,  py=15 → pt=17  (8-15-17 triangle)
    Track 3: px=7,  py=24 → pt=25  (7-24-25 triangle)
    Track 4: px=20, py=21 → pt=29  (20-21-29 triangle)
    Track 5: px=9,  py=40 → pt=41  (9-40-41 triangle)
    Track 6: px=12, py=35 → pt=37  (12-35-37 triangle)

Usage:
    from tests.generators.toy_lorentz import generate_toy_lorentz_root, generate_toy_lorentz_dict
    
    # Generate ROOT file with TLorentzVector (Option A + B)
    generate_toy_lorentz_root("toy.root", n_events=3)
    
    # Generate dict (Option B only)
    data = generate_toy_lorentz_dict(n_events=3)

Author: Claude Opus 4.5
Date: 2026-01-15
Phase: 13.6.B.fix
"""

import numpy as np
from typing import List, Optional, Dict, Any

# Pythagorean triples: (px, py, pt)
# Integer values ensure EXACT comparison (no floating point tolerance)
PYTHAGOREAN_TRIPLES = [
    (3, 4, 5),
    (5, 12, 13),
    (8, 15, 17),
    (7, 24, 25),
    (20, 21, 29),
    (9, 40, 41),
    (12, 35, 37),
]


def generate_toy_lorentz_root(
    filename: str = "toy_lorentz.root",
    n_events: int = 3,
    tracks_per_event: Optional[List[int]] = None,
) -> str:
    """
    Generate toy ROOT file with TLorentzVector tracks.
    
    Creates BOTH access paths:
    - tracks: RVec<TLorentzVector> (Option A - method calls)
    - track_pt: RVec<double> (Option B - pre-computed arrays)
    - track_px, track_py: RVec<double> (for verification)
    
    Parameters
    ----------
    filename : str
        Output ROOT file path
    n_events : int
        Number of events (default: 3)
    tracks_per_event : List[int], optional
        Tracks per event (default: [2, 3, 2])
    
    Returns
    -------
    str
        Path to generated file
    
    Data Layout
    -----------
    Event 0 (2 tracks):
      Track 0: px=3, py=4 → pt=5
      Track 1: px=5, py=12 → pt=13
    Event 1 (3 tracks):
      Track 0: px=8, py=15 → pt=17
      Track 1: px=7, py=24 → pt=25
      Track 2: px=20, py=21 → pt=29
    Event 2 (2 tracks):
      Track 0: px=9, py=40 → pt=41
      Track 1: px=12, py=35 → pt=37
    
    Example
    -------
    >>> generate_toy_lorentz_root("test.root", n_events=3)
    'test.root'
    """
    import ROOT
    
    if tracks_per_event is None:
        tracks_per_event = [2, 3, 2]
    
    # Extend tracks_per_event if n_events > len(tracks_per_event)
    while len(tracks_per_event) < n_events:
        tracks_per_event.append(2)
    
    # Create ROOT file
    f = ROOT.TFile(filename, "RECREATE")
    tree = ROOT.TTree("Events", "Toy Events with TLorentzVector")
    
    # Branches
    event_id = np.zeros(1, dtype=np.int64)
    n_tracks = np.zeros(1, dtype=np.int32)
    
    # Option A: TLorentzVector (for method calls)
    tracks = ROOT.std.vector('TLorentzVector')()
    
    # Option B: Pre-computed arrays (for direct comparison)
    track_pt = ROOT.std.vector('double')()
    track_px = ROOT.std.vector('double')()
    track_py = ROOT.std.vector('double')()
    track_phi = ROOT.std.vector('double')()
    track_eta = ROOT.std.vector('double')()
    
    tree.Branch("event_id", event_id, "event_id/L")
    tree.Branch("n_tracks", n_tracks, "n_tracks/I")
    tree.Branch("tracks", tracks)
    tree.Branch("track_pt", track_pt)
    tree.Branch("track_px", track_px)
    tree.Branch("track_py", track_py)
    tree.Branch("track_phi", track_phi)
    tree.Branch("track_eta", track_eta)
    
    # Fill events
    triple_idx = 0
    for evt in range(n_events):
        event_id[0] = evt
        n_tracks[0] = tracks_per_event[evt]
        
        tracks.clear()
        track_pt.clear()
        track_px.clear()
        track_py.clear()
        track_phi.clear()
        track_eta.clear()
        
        for _ in range(tracks_per_event[evt]):
            px, py, pt = PYTHAGOREAN_TRIPLES[triple_idx % len(PYTHAGOREAN_TRIPLES)]
            triple_idx += 1
            
            # Option A: TLorentzVector
            # Use pz=0, E=pt for simplicity (massless, central rapidity)
            tlv = ROOT.TLorentzVector()
            tlv.SetPxPyPzE(float(px), float(py), 0.0, float(pt))
            tracks.push_back(tlv)
            
            # Option B: Pre-computed
            track_pt.push_back(float(pt))
            track_px.push_back(float(px))
            track_py.push_back(float(py))
            track_phi.push_back(float(np.arctan2(py, px)))
            track_eta.push_back(0.0)  # pz=0 → eta=0
        
        tree.Fill()
    
    f.Write()
    f.Close()
    
    return filename


def generate_toy_lorentz_dict(
    n_events: int = 3,
    tracks_per_event: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Generate toy data as dict (for flatten tests without ROOT file).
    
    Returns dict with pre-computed arrays only (Option B).
    For Option A (TLorentzVector), use generate_toy_lorentz_root().
    
    Parameters
    ----------
    n_events : int
        Number of events (default: 3)
    tracks_per_event : List[int], optional
        Tracks per event (default: [2, 3, 2])
    
    Returns
    -------
    dict
        Dictionary with event_id, n_tracks, track_pt, track_px, track_py
    
    Example
    -------
    >>> data = generate_toy_lorentz_dict(n_events=3)
    >>> data['track_pt'][0]  # [5.0, 13.0] - Event 0 has 2 tracks
    """
    if tracks_per_event is None:
        tracks_per_event = [2, 3, 2]
    
    # Extend if needed
    while len(tracks_per_event) < n_events:
        tracks_per_event.append(2)
    
    event_ids = np.arange(n_events, dtype=np.int64)
    n_tracks_arr = np.array(tracks_per_event[:n_events], dtype=np.int32)
    
    track_pt = np.empty(n_events, dtype=object)
    track_px = np.empty(n_events, dtype=object)
    track_py = np.empty(n_events, dtype=object)
    track_phi = np.empty(n_events, dtype=object)
    track_eta = np.empty(n_events, dtype=object)
    
    triple_idx = 0
    for evt in range(n_events):
        pts, pxs, pys, phis, etas = [], [], [], [], []
        for _ in range(tracks_per_event[evt]):
            px, py, pt = PYTHAGOREAN_TRIPLES[triple_idx % len(PYTHAGOREAN_TRIPLES)]
            triple_idx += 1
            pts.append(float(pt))
            pxs.append(float(px))
            pys.append(float(py))
            phis.append(float(np.arctan2(py, px)))
            etas.append(0.0)
        
        track_pt[evt] = np.array(pts, dtype=np.float64)
        track_px[evt] = np.array(pxs, dtype=np.float64)
        track_py[evt] = np.array(pys, dtype=np.float64)
        track_phi[evt] = np.array(phis, dtype=np.float64)
        track_eta[evt] = np.array(etas, dtype=np.float64)
    
    return {
        'event_id': event_ids,
        'n_tracks': n_tracks_arr,
        'track_pt': track_pt,
        'track_px': track_px,
        'track_py': track_py,
        'track_phi': track_phi,
        'track_eta': track_eta,
    }


def generate_toy_with_clusters(
    n_events: int = 3,
    tracks_per_event: Optional[List[int]] = None,
    clusters_per_track: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """
    Generate toy data with 2D cluster structure.
    
    For testing mixed 1D+2D depth operations.
    
    Parameters
    ----------
    n_events : int
        Number of events
    tracks_per_event : List[int], optional
        Tracks per event (default: [2, 3, 2])
    clusters_per_track : List[List[int]], optional
        Clusters per track per event (default: deterministic pattern)
    
    Returns
    -------
    dict
        Dictionary with track and cluster data
    """
    if tracks_per_event is None:
        tracks_per_event = [2, 3, 2]
    
    if clusters_per_track is None:
        clusters_per_track = [
            [4, 5],           # Event 0: track0=4 clusters, track1=5 clusters
            [3, 4, 2],        # Event 1: track0=3, track1=4, track2=2
            [4, 3],           # Event 2: track0=4, track1=3
        ]
    
    # Extend if needed
    while len(tracks_per_event) < n_events:
        tracks_per_event.append(2)
    while len(clusters_per_track) < n_events:
        clusters_per_track.append([3, 3])
    
    # Get base track data
    data = generate_toy_lorentz_dict(n_events, tracks_per_event)
    
    # Add cluster data (2D: RVec<RVec>)
    cluster_Q = np.empty(n_events, dtype=object)
    cluster_x = np.empty(n_events, dtype=object)
    cluster_y = np.empty(n_events, dtype=object)
    
    for evt in range(n_events):
        n_trk = tracks_per_event[evt]
        event_Q = np.empty(n_trk, dtype=object)
        event_x = np.empty(n_trk, dtype=object)
        event_y = np.empty(n_trk, dtype=object)
        
        for trk in range(n_trk):
            n_clus = clusters_per_track[evt][trk] if trk < len(clusters_per_track[evt]) else 3
            
            # Deterministic values: 10*trk + 100*evt + cluster_idx
            base = 10.0 * (trk + 1) + 100.0 * evt
            event_Q[trk] = np.arange(base, base + n_clus, dtype=np.float64)
            event_x[trk] = np.arange(base + 0.1, base + n_clus + 0.1, dtype=np.float64)
            event_y[trk] = np.arange(base + 0.2, base + n_clus + 0.2, dtype=np.float64)
        
        cluster_Q[evt] = event_Q
        cluster_x[evt] = event_x
        cluster_y[evt] = event_y
    
    data['cluster_Q'] = cluster_Q
    data['cluster_x'] = cluster_x
    data['cluster_y'] = cluster_y
    
    return data


def validate_toy_data(data: Dict[str, Any]) -> bool:
    """
    Validate toy data structure and properties.
    
    Returns True if data is valid, raises AssertionError otherwise.
    """
    assert 'event_id' in data
    assert 'n_tracks' in data
    assert 'track_pt' in data
    
    n_events = len(data['event_id'])
    assert len(data['n_tracks']) == n_events
    assert len(data['track_pt']) == n_events
    
    # Validate track counts
    for event_id in range(n_events):
        expected_tracks = data['n_tracks'][event_id]
        actual_tracks = len(data['track_pt'][event_id])
        assert actual_tracks == expected_tracks, \
            f"Event {event_id}: n_tracks={expected_tracks} but track_pt has {actual_tracks}"
    
    # Validate Pythagorean identity if px, py present
    if 'track_px' in data and 'track_py' in data:
        for event_id in range(n_events):
            px = data['track_px'][event_id]
            py = data['track_py'][event_id]
            pt = data['track_pt'][event_id]
            
            computed_pt = np.sqrt(px**2 + py**2)
            assert np.allclose(pt, computed_pt), \
                f"Event {event_id}: pt != sqrt(px² + py²)"
    
    return True


def get_expected_pt_values(n_events: int = 3, tracks_per_event: Optional[List[int]] = None) -> List[float]:
    """
    Get expected pt values for toy data (for test assertions).
    
    Returns flat list of all pt values in order.
    """
    if tracks_per_event is None:
        tracks_per_event = [2, 3, 2]
    
    pts = []
    triple_idx = 0
    for evt in range(n_events):
        for _ in range(tracks_per_event[evt]):
            _, _, pt = PYTHAGOREAN_TRIPLES[triple_idx % len(PYTHAGOREAN_TRIPLES)]
            pts.append(float(pt))
            triple_idx += 1
    
    return pts


# Convenience function for pytest fixture
def get_toy_fixture() -> Dict[str, Any]:
    """Get default toy data for pytest fixture."""
    return generate_toy_lorentz_dict(n_events=3)
