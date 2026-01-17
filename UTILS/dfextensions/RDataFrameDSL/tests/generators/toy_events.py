"""
Toy Event Generator (Arrays-Only)

Phase 13.6.B.fix: Simple deterministic data for flatten tests.

This generator creates dict-based data WITHOUT ROOT dependencies.
For TLorentzVector tests, use toy_lorentz.py instead.

Usage:
    from tests.generators.toy_events import generate_toy_dict, generate_toy_with_clusters
    
    # Simple track data
    data = generate_toy_dict(n_events=3)
    
    # Track + cluster data (2D)
    data = generate_toy_with_clusters(n_events=3)

Author: Claude Opus 4.5
Date: 2026-01-15
Phase: 13.6.B.fix
"""

import numpy as np
from typing import List, Optional, Dict, Any


def generate_toy_dict(
    n_events: int = 3,
    tracks_per_event: Optional[List[int]] = None,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Generate deterministic toy data as dict.
    
    Parameters
    ----------
    n_events : int
        Number of events to generate
    tracks_per_event : list, optional
        Number of tracks per event. If None, uses [2, 3, 2]
    seed : int, optional
        Random seed (for future extensions)
    
    Returns
    -------
    dict
        Dictionary with event_id, n_tracks, track_pt, track_phi, track_eta
    
    Example
    -------
    >>> data = generate_toy_dict(n_events=3)
    >>> data['event_id']  # [0, 1, 2]
    >>> data['track_pt'][0]  # [1.0, 2.0] (Event 0 has 2 tracks)
    """
    if seed is not None:
        np.random.seed(seed)
    
    if tracks_per_event is None:
        tracks_per_event = [2, 3, 2]  # 3 events: 2, 3, 2 tracks
    
    # Extend if needed
    while len(tracks_per_event) < n_events:
        tracks_per_event.append(2)
    
    # Generate event IDs
    event_ids = np.arange(n_events, dtype=np.int64)
    n_tracks = np.array(tracks_per_event[:n_events], dtype=np.int32)
    
    # Generate track-level data (1D RVec)
    track_pt = np.empty(n_events, dtype=object)
    track_phi = np.empty(n_events, dtype=object)
    track_eta = np.empty(n_events, dtype=object)
    
    for event_id in range(n_events):
        n_trk = n_tracks[event_id]
        # Arithmetic sequence: 1, 2, 3, ... or 3, 4, 5, ... or 6, 7, ...
        start_pt = 1.0 + event_id * 3.0
        track_pt[event_id] = np.arange(start_pt, start_pt + n_trk, dtype=np.float64)
        # Phi evenly distributed
        track_phi[event_id] = np.linspace(0, np.pi, n_trk, dtype=np.float64)
        # Eta around zero
        track_eta[event_id] = np.linspace(-1, 1, n_trk, dtype=np.float64)
    
    return {
        'event_id': event_ids,
        'n_tracks': n_tracks,
        'track_pt': track_pt,
        'track_phi': track_phi,
        'track_eta': track_eta,
    }


def generate_toy_with_clusters(
    n_events: int = 3,
    tracks_per_event: Optional[List[int]] = None,
    clusters_per_track: Optional[List[List[int]]] = None,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Generate toy data with 2D cluster structure.
    
    For testing mixed 1D+2D depth operations.
    
    Parameters
    ----------
    n_events : int
        Number of events
    tracks_per_event : list, optional
        Tracks per event (default: [2, 3, 2])
    clusters_per_track : list of lists, optional
        Clusters per track per event (default: deterministic pattern)
    seed : int, optional
        Random seed
    
    Returns
    -------
    dict
        Dictionary with track and cluster data
    
    Data Layout
    -----------
    Event 0: 2 tracks (4, 5 clusters)
    Event 1: 3 tracks (3, 4, 2 clusters)
    Event 2: 2 tracks (4, 3 clusters)
    
    Example
    -------
    >>> data = generate_toy_with_clusters(n_events=3)
    >>> data['cluster_Q'][0][0]  # [10, 11, 12, 13] - Event 0, Track 0
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
    data = generate_toy_dict(n_events, tracks_per_event, seed)
    
    # Add cluster data (2D: RVec<RVec>)
    cluster_Q = np.empty(n_events, dtype=object)
    cluster_x = np.empty(n_events, dtype=object)
    cluster_y = np.empty(n_events, dtype=object)
    cluster_z = np.empty(n_events, dtype=object)
    
    for evt in range(n_events):
        n_trk = tracks_per_event[evt]
        event_Q = np.empty(n_trk, dtype=object)
        event_x = np.empty(n_trk, dtype=object)
        event_y = np.empty(n_trk, dtype=object)
        event_z = np.empty(n_trk, dtype=object)
        
        for trk in range(n_trk):
            n_clus = clusters_per_track[evt][trk] if trk < len(clusters_per_track[evt]) else 3
            
            # Deterministic values: 10*trk + 100*evt + cluster_idx
            base = 10.0 * (trk + 1) + 100.0 * evt
            event_Q[trk] = np.arange(base, base + n_clus, dtype=np.float64)
            event_x[trk] = np.arange(base + 0.1, base + n_clus + 0.1, dtype=np.float64)
            event_y[trk] = np.arange(base + 0.2, base + n_clus + 0.2, dtype=np.float64)
            event_z[trk] = np.arange(base + 0.3, base + n_clus + 0.3, dtype=np.float64)
        
        cluster_Q[evt] = event_Q
        cluster_x[evt] = event_x
        cluster_y[evt] = event_y
        cluster_z[evt] = event_z
    
    data['cluster_Q'] = cluster_Q
    data['cluster_x'] = cluster_x
    data['cluster_y'] = cluster_y
    data['cluster_z'] = cluster_z
    
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
        
        # Validate cluster counts if present
        if 'cluster_Q' in data:
            assert len(data['cluster_Q'][event_id]) == expected_tracks, \
                f"Event {event_id}: cluster_Q has wrong track count"
    
    return True


def get_expected_sums(
    n_events: int = 3,
    tracks_per_event: Optional[List[int]] = None,
    clusters_per_track: Optional[List[List[int]]] = None,
) -> Dict[str, float]:
    """
    Get expected sums for validation (known analytical values).
    
    Returns
    -------
    dict
        Dictionary with expected sums for each column
    """
    data = generate_toy_with_clusters(n_events, tracks_per_event, clusters_per_track)
    
    track_pt_sum = sum(arr.sum() for arr in data['track_pt'])
    
    cluster_Q_sum = 0.0
    for event in data['cluster_Q']:
        for track in event:
            cluster_Q_sum += track.sum()
    
    return {
        'track_pt_sum': track_pt_sum,
        'cluster_Q_sum': cluster_Q_sum,
        'n_events': n_events,
        'total_tracks': sum(data['n_tracks']),
        'total_clusters': sum(
            sum(len(track) for track in event)
            for event in data['cluster_Q']
        ),
    }


# Convenience function for pytest fixture
def get_toy_fixture() -> Dict[str, Any]:
    """Get default toy data for pytest fixture."""
    return generate_toy_with_clusters(n_events=3)
