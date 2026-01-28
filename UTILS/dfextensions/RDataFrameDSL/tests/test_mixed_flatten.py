"""
Phase 13.6.A-ext: Mixed Flatten Tests

Tests T1-T11 (positive) and N1-N5 (negative) from spec.
"""

import pytest
import numpy as np
import pandas as pd
import warnings

import sys
sys.path.insert(0, '/home/claude')

from RDataFrameDSL.flatten import (
    flatten_to_dataframe,
    flatten_to_dict,
    flatten_to_tables,
    FlattenBackend,
)


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def scalar_1d_data():
    """Data with scalars and 1D columns."""
    return {
        'event_id': np.array([100, 101], dtype=np.int64),
        'multiplicity': np.array([2, 3], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0, 5.0], dtype=np.float64),
        ], dtype=object),
    }


@pytest.fixture
def full_mixed_data():
    """Data with scalars, 1D, and 2D columns."""
    return {
        'event_id': np.array([100], dtype=np.int64),
        'multiplicity': np.array([3], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ], dtype=object),
        'cluster_Q': np.array([
            np.array([
                np.array([10.0, 20.0], dtype=np.float64),
                np.array([30.0], dtype=np.float64),
                np.array([40.0, 50.0, 60.0], dtype=np.float64),
            ], dtype=object),
        ], dtype=object),
    }


@pytest.fixture
def empty_event_data():
    """Data with empty events."""
    return {
        'event_id': np.array([100, 101, 102], dtype=np.int64),
        'multiplicity': np.array([2, 0, 1], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([], dtype=np.float64),  # Empty
            np.array([3.0], dtype=np.float64),
        ], dtype=object),
    }


@pytest.fixture
def empty_track_data():
    """Data with empty tracks (0 clusters)."""
    return {
        'event_id': np.array([100], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ], dtype=object),
        'cluster_Q': np.array([
            np.array([
                np.array([], dtype=np.float64),  # Track 0: 0 clusters
                np.array([10.0, 20.0], dtype=np.float64),
                np.array([30.0], dtype=np.float64),
            ], dtype=object),
        ], dtype=object),
    }


# =============================================================================
# T1: Scalar + 1D (Basic Mixed)
# =============================================================================

def test_T1_scalar_plus_1d(scalar_1d_data):
    """TD8 equivalent: 1D + scalar in single query."""
    df = flatten_to_dataframe(scalar_1d_data, columns=['track_pt', 'multiplicity'])
    
    # 5 rows (2 + 3 tracks)
    assert len(df) == 5
    
    # Columns present
    assert list(df.columns) == ['event_id', 'multiplicity', 'track_idx', 'track_pt']
    
    # Scalar replicated correctly
    expected_mult = np.array([2, 2, 3, 3, 3])
    np.testing.assert_array_equal(df['multiplicity'].values, expected_mult)
    
    # Track values correct
    expected_pt = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    np.testing.assert_array_almost_equal(df['track_pt'].values, expected_pt)


# =============================================================================
# T2: 2D + 1D (Track Replication)
# =============================================================================

def test_T2_2d_plus_1d():
    """TD12 equivalent: 2D + 1D mixed."""
    data = {
        'event_id': np.array([100], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ], dtype=object),
        'cluster_Q': np.array([
            np.array([
                np.array([10.0, 20.0], dtype=np.float64),
                np.array([30.0], dtype=np.float64),
                np.array([40.0, 50.0, 60.0], dtype=np.float64),
            ], dtype=object),
        ], dtype=object),
    }
    
    df = flatten_to_dataframe(data, columns=['cluster_Q', 'track_pt'])
    
    # 6 rows (2 + 1 + 3 clusters)
    assert len(df) == 6
    
    # Track values replicated to cluster level
    expected_pt = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    np.testing.assert_array_almost_equal(df['track_pt'].values, expected_pt)
    
    # Cluster values correct
    expected_Q = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    np.testing.assert_array_almost_equal(df['cluster_Q'].values, expected_Q)
    
    # Track indices correct (replicated per cluster)
    expected_track_idx = np.array([0, 0, 1, 2, 2, 2])
    np.testing.assert_array_equal(df['track_idx'].values, expected_track_idx)


# =============================================================================
# T3: 2D + 1D + Scalar (Full Mixed)
# =============================================================================

def test_T3_2d_plus_1d_plus_scalar(full_mixed_data):
    """TD13 equivalent: Full 2D + 1D + scalar."""
    df = flatten_to_dataframe(
        full_mixed_data, 
        columns=['cluster_Q', 'track_pt', 'multiplicity']
    )
    
    # 6 rows
    assert len(df) == 6
    
    # All columns present in correct order
    assert list(df.columns) == [
        'event_id', 'multiplicity', 'track_idx', 'track_pt', 'cluster_idx', 'cluster_Q'
    ]
    
    # Scalar replicated to all 6 rows
    assert all(df['multiplicity'] == 3)
    
    # Track values replicated per cluster
    expected_pt = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    np.testing.assert_array_almost_equal(df['track_pt'].values, expected_pt)


# =============================================================================
# T4: Empty Event Handling
# =============================================================================

def test_T4_empty_event(empty_event_data):
    """Empty event in middle."""
    df = flatten_to_dataframe(
        empty_event_data, 
        columns=['track_pt', 'multiplicity']
    )
    
    # 3 rows (event 101 contributes 0)
    assert len(df) == 3
    
    # Event 101 not present
    assert 101 not in df['event_id'].values


# =============================================================================
# T5: Empty Track Handling (Disappearing Rows)
# =============================================================================

def test_T5_empty_track(empty_track_data):
    """Track with zero clusters."""
    df = flatten_to_dataframe(
        empty_track_data, 
        columns=['cluster_Q', 'track_pt']
    )
    
    # 3 rows (0 + 2 + 1 clusters)
    assert len(df) == 3
    
    # Track 0 not present (no clusters)
    assert 0 not in df['track_idx'].values
    
    # Track 1 and 2 present
    assert 1 in df['track_idx'].values
    assert 2 in df['track_idx'].values


# =============================================================================
# T6: Scalars Only
# =============================================================================

def test_T6_scalars_only():
    """Only scalar columns - no flattening."""
    data = {
        'event_id': np.array([100, 101, 102], dtype=np.int64),
        'multiplicity': np.array([50, 30, 45], dtype=np.int64),
        'vertex_z': np.array([1.5, -2.3, 0.8], dtype=np.float64),
    }
    
    df = flatten_to_dataframe(data, columns=['multiplicity', 'vertex_z'])
    
    # 3 rows (event-level)
    assert len(df) == 3
    
    # No index columns
    assert 'track_idx' not in df.columns
    assert 'cluster_idx' not in df.columns


# =============================================================================
# T7: Backward Compatibility
# =============================================================================

def test_T7_backward_compat_rvec_columns():
    """Phase 13.6.A API still works with deprecation warning."""
    data = {
        'event_id': np.array([100, 101], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
        ], dtype=object),
    }
    
    with pytest.warns(DeprecationWarning, match="rvec_columns.*deprecated"):
        df = flatten_to_dataframe(data, rvec_columns=['track_pt'])
    
    # Still works
    assert len(df) == 3
    assert list(df.columns) == ['event_id', 'track_idx', 'track_pt']


# =============================================================================
# T8-T11: Normalized Mode Tests
# =============================================================================

def test_T8_normalized_full_mixed():
    """flatten_to_tables with all three levels."""
    data = {
        'event_id': np.array([100, 101], dtype=np.int64),
        'multiplicity': np.array([2, 1], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
        ], dtype=object),
        'cluster_Q': np.array([
            np.array([
                np.array([10.0, 20.0], dtype=np.float64),
                np.array([30.0], dtype=np.float64),
            ], dtype=object),
            np.array([
                np.array([40.0, 50.0], dtype=np.float64),
            ], dtype=object),
        ], dtype=object),
    }
    
    tables = flatten_to_tables(data, columns=['multiplicity', 'track_pt', 'cluster_Q'])
    
    # Three tables returned
    assert set(tables.keys()) == {'events', 'tracks', 'clusters'}
    
    # Events table: 2 rows, no duplication
    assert len(tables['events']) == 2
    assert list(tables['events'].columns) == ['event_id', 'multiplicity']
    
    # Tracks table: 3 rows (2 + 1)
    assert len(tables['tracks']) == 3
    assert list(tables['tracks'].columns) == ['event_id', 'track_idx', 'track_pt']
    
    # Clusters table: 5 rows (2 + 1 + 2)
    assert len(tables['clusters']) == 5
    assert list(tables['clusters'].columns) == ['event_id', 'track_idx', 'cluster_idx', 'cluster_Q']
    
    # No track_pt in clusters (normalized, not replicated)
    assert 'track_pt' not in tables['clusters'].columns
    
    # No multiplicity in tracks (normalized)
    assert 'multiplicity' not in tables['tracks'].columns


def test_T9_normalized_joinable():
    """Verify tables can be joined correctly."""
    data = {
        'event_id': np.array([100], dtype=np.int64),
        'multiplicity': np.array([3], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ], dtype=object),
        'cluster_Q': np.array([
            np.array([
                np.array([10.0, 20.0], dtype=np.float64),
                np.array([30.0], dtype=np.float64),
                np.array([40.0, 50.0, 60.0], dtype=np.float64),
            ], dtype=object),
        ], dtype=object),
    }
    
    tables = flatten_to_tables(data, columns=['multiplicity', 'track_pt', 'cluster_Q'])
    
    # Join tracks with events
    tracks_with_event = tables['tracks'].merge(tables['events'], on='event_id')
    assert len(tracks_with_event) == 3
    assert 'multiplicity' in tracks_with_event.columns
    assert all(tracks_with_event['multiplicity'] == 3)
    
    # Join clusters with tracks
    clusters_with_track = tables['clusters'].merge(
        tables['tracks'], on=['event_id', 'track_idx']
    )
    assert len(clusters_with_track) == 6
    assert 'track_pt' in clusters_with_track.columns


def test_T10_normalized_1d_only():
    """flatten_to_tables with only 1D columns."""
    data = {
        'event_id': np.array([100, 101], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
        ], dtype=object),
        'track_eta': np.array([
            np.array([0.1, 0.2], dtype=np.float64),
            np.array([0.3], dtype=np.float64),
        ], dtype=object),
    }
    
    tables = flatten_to_tables(data, columns=['track_pt', 'track_eta'])
    
    # Only tracks table (no events, no clusters)
    assert set(tables.keys()) == {'tracks'}
    assert len(tables['tracks']) == 3


def test_T11_normalized_no_duplication():
    """Verify normalized mode doesn't duplicate data."""
    data = {
        'event_id': np.array([100], dtype=np.int64),
        'big_scalar': np.array([999999], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ], dtype=object),
    }
    
    # Flat mode: scalar replicated 3 times
    df_flat = flatten_to_dataframe(data, columns=['track_pt', 'big_scalar'])
    assert len(df_flat) == 3
    assert list(df_flat['big_scalar']) == [999999, 999999, 999999]  # Replicated
    
    # Normalized mode: scalar stored once
    tables = flatten_to_tables(data, columns=['track_pt', 'big_scalar'])
    assert len(tables['events']) == 1
    assert tables['events']['big_scalar'].iloc[0] == 999999  # Not replicated


# =============================================================================
# N1-N5: Negative Tests
# =============================================================================

def test_N1_1d_structure_mismatch():
    """Different 1D column lengths - should error."""
    data = {
        'event_id': np.array([100], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        ], dtype=object),
        'filtered': np.array([
            np.array([2.0, 3.0], dtype=np.float64),  # Different length!
        ], dtype=object),
    }
    
    with pytest.raises(ValueError, match="different structures"):
        flatten_to_dataframe(data, columns=['track_pt', 'filtered'])


def test_N2_track_axis_mismatch():
    """1D and 2D have different track counts - should error."""
    data = {
        'event_id': np.array([100], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0, 3.0], dtype=np.float64),  # 3 tracks
        ], dtype=object),
        'cluster_Q': np.array([
            np.array([
                np.array([10.0], dtype=np.float64),
                np.array([20.0], dtype=np.float64),
                # Only 2 track-groups!
            ], dtype=object),
        ], dtype=object),
    }
    
    with pytest.raises(ValueError, match="Track axis mismatch"):
        flatten_to_dataframe(data, columns=['cluster_Q', 'track_pt'])


def test_N3_missing_parent_id():
    """parent_id_column not in data - should error."""
    data = {
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float64),
        ], dtype=object),
    }
    
    with pytest.raises(ValueError, match="not in data"):
        flatten_to_dataframe(data, columns=['track_pt'], parent_id_column='event_id')


def test_N4_missing_column():
    """Requested column not in data - should error."""
    data = {
        'event_id': np.array([100], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float64),
        ], dtype=object),
    }
    
    with pytest.raises(ValueError, match="not in data"):
        flatten_to_dataframe(data, columns=['track_pt', 'track_eta'])


def test_N5_both_columns_and_rvec_columns():
    """Cannot specify both - should error."""
    data = {
        'event_id': np.array([100], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0], dtype=np.float64),
        ], dtype=object),
    }
    
    with pytest.raises(ValueError, match="Cannot specify both"):
        flatten_to_dataframe(data, columns=['track_pt'], rvec_columns=['track_pt'])


# =============================================================================
# Additional Edge Case Tests
# =============================================================================

def test_duplicate_parent_id_warning():
    """Duplicate parent IDs should warn (not error)."""
    data = {
        'event_id': np.array([100, 100, 101], dtype=np.int64),  # Duplicate!
        'track_pt': np.array([
            np.array([1.0], dtype=np.float64),
            np.array([2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
        ], dtype=object),
    }
    
    with pytest.warns(UserWarning, match="duplicate values"):
        df = flatten_to_dataframe(data, columns=['track_pt'])
    
    # Still works
    assert len(df) == 3


def test_empty_columns_error():
    """Empty columns list should error."""
    data = {
        'event_id': np.array([100], dtype=np.int64),
    }
    
    with pytest.raises(ValueError, match="cannot be empty"):
        flatten_to_dataframe(data, columns=[])


def test_none_columns_error():
    """None columns should error."""
    data = {
        'event_id': np.array([100], dtype=np.int64),
    }
    
    with pytest.raises(ValueError, match="Must specify"):
        flatten_to_dataframe(data, columns=None)


# =============================================================================
# AUTO Backend Selection Tests (GPT8 Regression Tests)
# =============================================================================

def test_auto_selects_awkward_for_2d_same_depth():
    """
    GPT8 REGRESSION TEST: AUTO should select AWKWARD for same-depth 2D when available.
    
    This test verifies the Phase 13.6.A AUTO selection behavior is preserved:
    - 2D columns + Awkward available → AWKWARD backend selected
    - This prevents silent API contract changes
    """
    from RDataFrameDSL.flatten import (
        _select_backend_same_depth, 
        FlattenBackend, 
        awkward_available,
        is_nested_rvec
    )
    
    # 2D test data (RVec<RVec<double>>)
    data_2d = {
        'event_id': np.array([100, 101], dtype=np.int64),
        'cluster_Q': np.array([
            np.array([
                np.array([10.0, 20.0], dtype=np.float64),
                np.array([30.0], dtype=np.float64),
            ], dtype=object),
            np.array([
                np.array([40.0, 50.0], dtype=np.float64),
            ], dtype=object),
        ], dtype=object),
    }
    
    # Verify it's actually 2D
    assert is_nested_rvec(data_2d, 'cluster_Q'), "Test data should be 2D"
    
    # Test AUTO selection
    selected = _select_backend_same_depth(data_2d, ['cluster_Q'])
    
    if awkward_available():
        # AUTO should select AWKWARD for 2D when available
        assert selected == FlattenBackend.AWKWARD, \
            f"AUTO should select AWKWARD for 2D when available, got {selected}"
    else:
        # Without Awkward, AUTO falls back to NUMPY
        assert selected == FlattenBackend.NUMPY, \
            f"AUTO should select NUMPY when Awkward unavailable, got {selected}"


def test_auto_selects_numpy_for_1d_same_depth():
    """AUTO should select NUMPY for same-depth 1D columns."""
    from RDataFrameDSL.flatten import (
        _select_backend_same_depth, 
        FlattenBackend,
        is_nested_rvec
    )
    
    # 1D test data (RVec<double>)
    data_1d = {
        'event_id': np.array([100, 101], dtype=np.int64),
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0], dtype=np.float64),
        ], dtype=object),
    }
    
    # Verify it's actually 1D
    assert not is_nested_rvec(data_1d, 'track_pt'), "Test data should be 1D"
    
    # AUTO should always select NUMPY for 1D
    selected = _select_backend_same_depth(data_1d, ['track_pt'])
    assert selected == FlattenBackend.NUMPY, \
        f"AUTO should select NUMPY for 1D, got {selected}"


def test_depth_detection_accepts_scalar_in_object_array():
    """
    Phase 13.6.G+ TEST: Depth detection accepts scalar items in object arrays.
    
    ROOT's AsNumpy() returns bool columns as object dtype with Python bool values.
    This test verifies we correctly detect these as depth 0 (scalar).
    
    Previously (P1-2): This raised ValueError - but that breaks draw() with bool columns.
    Now: We accept scalars in object arrays and treat them as depth 0.
    
    APPROVAL REQUEST:
    - Behavior change: _get_depth_from_data() now returns 0 for scalar-in-object-array
    - Reason: ROOT returns bool columns as dtype=object with Python bool values
    - Impact: Fixes draw() with bool selection columns (e.g., isOK)
    - Risk: Low - scalars in object array are now handled, not rejected
    """
    from RDataFrameDSL.flatten import _get_depth_from_data
    
    # Case 1: bool scalars in object array (ROOT's bool column representation)
    bool_data = {
        'event_id': np.array([100, 101], dtype=np.int64),
        'isOK': np.array([True, False], dtype=object),  # ROOT's bool representation
    }
    depth = _get_depth_from_data(bool_data, 'isOK')
    assert depth == 0, f"bool in object array should be depth 0, got {depth}"
    
    # Case 2: int scalars in object array
    int_data = {
        'event_id': np.array([100, 101], dtype=np.int64),
        'flag': np.array([42, 99], dtype=object),  # int scalars
    }
    depth = _get_depth_from_data(int_data, 'flag')
    assert depth == 0, f"int in object array should be depth 0, got {depth}"
    
    # Case 3: float scalars in object array
    float_data = {
        'event_id': np.array([100, 101], dtype=np.int64),
        'value': np.array([1.5, 2.5], dtype=object),  # float scalars
    }
    depth = _get_depth_from_data(float_data, 'value')
    assert depth == 0, f"float in object array should be depth 0, got {depth}"


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
