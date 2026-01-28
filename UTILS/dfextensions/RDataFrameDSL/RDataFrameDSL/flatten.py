"""
RDataFrameDSL Flatten Module

Phase 13.6.A: Basic flatten functionality (same-depth columns)
Phase 13.6.A-ext: Mixed-depth flatten (scalar + 1D + 2D)

This module provides functions to flatten hierarchical RVec data structures
into flat pandas DataFrames for analysis and visualization.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import numpy as np
import pandas as pd


# =============================================================================
# Backend Selection
# =============================================================================

class FlattenBackend(Enum):
    """Backend selection for flatten operations."""
    AUTO = "auto"
    NUMPY = "numpy"
    AWKWARD = "awkward"
    CPP = "cpp"  # Future


def awkward_available() -> bool:
    """Check if Awkward Array is available."""
    try:
        import awkward
        return True
    except ImportError:
        return False


def _select_backend_same_depth(
    data: Dict[str, np.ndarray], 
    rvec_columns: List[str]
) -> 'FlattenBackend':
    """
    AUTO backend selection for same-depth flattening (Phase 13.6.A behavior).
    
    This restores the original AUTO selection logic:
    - 2D columns + Awkward available → AWKWARD
    - Otherwise → NUMPY
    
    Note: Mixed-depth flattening always uses NumPy (per Phase 13.6.A-ext spec).
    
    Args:
        data: Dict from rdf.AsNumpy()
        rvec_columns: List of RVec column names
    
    Returns:
        Selected FlattenBackend
    """
    if not rvec_columns:
        return FlattenBackend.NUMPY
    
    # Check if 2D (nested RVec)
    is_2d = is_nested_rvec(data, rvec_columns[0])
    
    # 2D + Awkward available → use Awkward
    if is_2d and awkward_available():
        return FlattenBackend.AWKWARD
    
    return FlattenBackend.NUMPY


# =============================================================================
# Type Detection
# =============================================================================

def is_nested_rvec(data: Dict[str, np.ndarray], column: str) -> bool:
    """
    Check if column contains nested RVec (RVec<RVec<T>>).
    
    Returns True if column is 2-level nested (depth 2).
    """
    return _get_depth_from_data(data, column) == 2


def _get_depth_from_data(data: Dict[str, np.ndarray], col: str) -> int:
    """
    Determine nesting depth by inspecting data.
    
    Returns:
        0: Scalar (e.g., int64 array)
        1: RVec (e.g., object array of float64 arrays)
        2: RVec<RVec> (e.g., object array of object arrays)
    
    Phase 13.6.G+: Handle bool columns stored as object dtype by ROOT.
    """
    col_data = data[col]
    
    # Check if it's a simple numpy array (scalar per event)
    if col_data.dtype != object:
        return 0
    
    # It's object array - check first non-empty element
    for item in col_data:
        if item is None:
            continue
        
        # Phase 13.6.G+: Handle scalar types stored as object array
        # ROOT's AsNumpy() returns bool columns as object dtype with Python bool values
        if isinstance(item, (bool, np.bool_, int, np.integer, float, np.floating, str)):
            return 0  # Scalar column stored as object array
        
        # Check if it has length (is array-like)
        if not hasattr(item, '__len__'):
            return 0  # Treat unknown scalars as depth 0
        
        if len(item) > 0:
            first_elem = item[0]
            # Is first element itself an array? → depth 2
            if hasattr(first_elem, '__len__') and not isinstance(first_elem, (str, bytes)):
                return 2
            return 1
    
    # All empty - assume depth 1 (RVec)
    return 1


def _infer_dtype(rvec_column: np.ndarray) -> np.dtype:
    """Infer dtype from first non-empty RVec element."""
    for item in rvec_column:
        if item is not None and len(item) > 0:
            return np.asarray(item).dtype
    return np.float64  # Default


def _infer_dtype_2d(rvec_column: np.ndarray) -> np.dtype:
    """Infer dtype from first non-empty 2D RVec element."""
    for event in rvec_column:
        if event is not None:
            for track in event:
                if track is not None and len(track) > 0:
                    return np.asarray(track).dtype
    return np.float64  # Default


# =============================================================================
# Validation (Phase 13.6.A + 13.6.A-ext)
# =============================================================================

def validate_same_structure(data: Dict[str, np.ndarray], rvec_columns: List[str]) -> bool:
    """
    Validate that all RVec columns have identical jagged structure.
    
    Phase 13.6.A: Used for same-depth validation.
    
    Args:
        data: Dict from rdf.AsNumpy()
        rvec_columns: List of column names to validate
    
    Returns:
        True if valid
    
    Raises:
        ValueError if structures don't match
    """
    if len(rvec_columns) < 2:
        return True
    
    ref_col = rvec_columns[0]
    ref_lengths = [len(rv) for rv in data[ref_col]]
    
    for col in rvec_columns[1:]:
        col_lengths = [len(rv) for rv in data[col]]
        
        for event_idx, (ref_len, col_len) in enumerate(zip(ref_lengths, col_lengths)):
            if ref_len != col_len:
                raise ValueError(
                    f"Columns have different structures in event {event_idx}: "
                    f"'{ref_col}' has {ref_len} elements, "
                    f"'{col}' has {col_len} elements. "
                    f"All RVec columns must have identical per-event lengths."
                )
    
    return True


def _validate_parent_id(data: Dict[str, np.ndarray], parent_id_column: str) -> None:
    """
    Validate parent ID column exists and check for duplicates.
    
    Phase 13.6.A-ext: NEW validation.
    
    Raises:
        ValueError if parent_id_column not found
    
    Warns:
        UserWarning if duplicate parent IDs detected
    """
    if parent_id_column not in data:
        raise ValueError(
            f"parent_id_column '{parent_id_column}' not in data. "
            f"Available columns: {list(data.keys())}"
        )
    
    parent_ids = data[parent_id_column]
    unique_ids = np.unique(parent_ids)
    
    if len(unique_ids) != len(parent_ids):
        n_duplicates = len(parent_ids) - len(unique_ids)
        warnings.warn(
            f"parent_id_column '{parent_id_column}' contains {n_duplicates} duplicate values. "
            f"This is allowed but may cause unexpected results in normalized mode joins. "
            f"If this is intentional (e.g., filtered data), you can ignore this warning.",
            UserWarning,
            stacklevel=3
        )


def _validate_columns_exist(data: Dict[str, np.ndarray], columns: List[str]) -> None:
    """Validate all requested columns exist in data."""
    for col in columns:
        if col not in data:
            raise ValueError(
                f"Column '{col}' not in data. "
                f"Available columns: {list(data.keys())}"
            )


def _validate_1d_structure(data: Dict[str, np.ndarray], rvec_1d_cols: List[str]) -> None:
    """
    Validate all 1D columns have identical per-event lengths.
    
    Phase 13.6.A-ext: Used for mixed-depth validation.
    """
    if len(rvec_1d_cols) < 2:
        return
    
    ref_col = rvec_1d_cols[0]
    ref_lengths = [len(rv) for rv in data[ref_col]]
    
    for col in rvec_1d_cols[1:]:
        col_lengths = [len(rv) for rv in data[col]]
        
        for event_idx, (ref_len, col_len) in enumerate(zip(ref_lengths, col_lengths)):
            if ref_len != col_len:
                raise ValueError(
                    f"Columns have different structures in event {event_idx}: "
                    f"'{ref_col}' has {ref_len} elements, "
                    f"'{col}' has {col_len} elements. "
                    f"All 1D columns must have identical per-event lengths."
                )


def _validate_2d_structure(data: Dict[str, np.ndarray], rvec_2d_cols: List[str]) -> None:
    """
    Validate all 2D columns have identical nested structure.
    
    Phase 13.6.A-ext: Used for mixed-depth validation.
    """
    if len(rvec_2d_cols) < 2:
        return
    
    ref_col = rvec_2d_cols[0]
    
    for col in rvec_2d_cols[1:]:
        for event_idx, (ref_event, col_event) in enumerate(zip(data[ref_col], data[col])):
            # Check track count
            if len(ref_event) != len(col_event):
                raise ValueError(
                    f"2D structure mismatch in event {event_idx}: "
                    f"'{ref_col}' has {len(ref_event)} tracks, "
                    f"'{col}' has {len(col_event)} tracks."
                )
            
            # Check cluster count per track
            for track_idx, (ref_track, col_track) in enumerate(zip(ref_event, col_event)):
                if len(ref_track) != len(col_track):
                    raise ValueError(
                        f"2D structure mismatch in event {event_idx}, track {track_idx}: "
                        f"'{ref_col}' has {len(ref_track)} clusters, "
                        f"'{col}' has {len(col_track)} clusters."
                    )


def _validate_track_axis_alignment(
    data: Dict[str, np.ndarray], 
    col_1d: str, 
    col_2d: str
) -> None:
    """
    Validate that 1D and 2D columns share the same track axis.
    
    Phase 13.6.A-ext: Critical for mixed 2D+1D flattening.
    """
    for event_idx, (rvec_1d, rvec_2d) in enumerate(zip(data[col_1d], data[col_2d])):
        n_tracks_1d = len(rvec_1d)
        n_tracks_2d = len(rvec_2d)
        
        if n_tracks_1d != n_tracks_2d:
            raise ValueError(
                f"Track axis mismatch in event {event_idx}: "
                f"1D column '{col_1d}' has {n_tracks_1d} tracks, "
                f"2D column '{col_2d}' has {n_tracks_2d} track-groups. "
                f"Cannot align 1D and 2D columns with different track counts."
            )


def _validate_mixed_structure(
    data: Dict[str, np.ndarray],
    scalar_cols: List[str],
    rvec_1d_cols: List[str],
    rvec_2d_cols: List[str],
    parent_id_column: str
) -> None:
    """
    Validate that columns can be flattened together.
    
    Phase 13.6.A-ext: Full mixed-depth validation.
    
    Rules:
        R0: All columns must have same number of events
        R1: All 1D columns must have identical per-event lengths
        R2: All 2D columns must have identical nested structure
        R3: If both 1D and 2D present, track axis must align
        R4: Parent ID must exist (with duplicate warning)
    """
    all_cols = scalar_cols + rvec_1d_cols + rvec_2d_cols
    if not all_cols:
        return
    
    # R0: Event count consistency
    n_events = None
    for col in all_cols:
        col_len = len(data[col])
        if n_events is None:
            n_events = col_len
        elif col_len != n_events:
            raise ValueError(
                f"Event count mismatch: '{col}' has {col_len} events, "
                f"expected {n_events}"
            )
    
    # R1: 1D structure consistency
    if len(rvec_1d_cols) > 1:
        _validate_1d_structure(data, rvec_1d_cols)
    
    # R2: 2D structure consistency
    if len(rvec_2d_cols) > 1:
        _validate_2d_structure(data, rvec_2d_cols)
    
    # R3: Track axis alignment (1D vs 2D)
    if rvec_1d_cols and rvec_2d_cols:
        _validate_track_axis_alignment(data, rvec_1d_cols[0], rvec_2d_cols[0])
    
    # R4: Parent ID validation
    _validate_parent_id(data, parent_id_column)


# =============================================================================
# Column Classification (Phase 13.6.A-ext)
# =============================================================================

def _classify_columns_by_depth(
    data: Dict[str, np.ndarray], 
    columns: List[str]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Classify columns into depth buckets.
    
    Phase 13.6.A-ext: NEW function.
    
    Args:
        data: Dict from rdf.AsNumpy()
        columns: List of column names
    
    Returns:
        Tuple of (scalar_cols, rvec_1d_cols, rvec_2d_cols)
    """
    scalar_cols = []
    rvec_1d_cols = []
    rvec_2d_cols = []
    
    for col in columns:
        depth = _get_depth_from_data(data, col)
        if depth == 0:
            scalar_cols.append(col)
        elif depth == 1:
            rvec_1d_cols.append(col)
        elif depth == 2:
            rvec_2d_cols.append(col)
    
    return scalar_cols, rvec_1d_cols, rvec_2d_cols


def _determine_target_depth(
    scalar_cols: List[str], 
    rvec_1d_cols: List[str], 
    rvec_2d_cols: List[str]
) -> int:
    """
    Determine deepest nesting level.
    
    Phase 13.6.A-ext: NEW function.
    
    Returns:
        0 if only scalars
        1 if any 1D columns (and no 2D)
        2 if any 2D columns
    """
    if rvec_2d_cols:
        return 2
    if rvec_1d_cols:
        return 1
    return 0


def _is_mixed_depth(
    scalar_cols: List[str], 
    rvec_1d_cols: List[str], 
    rvec_2d_cols: List[str]
) -> bool:
    """Check if columns have mixed nesting depths."""
    has_scalar = len(scalar_cols) > 0
    has_1d = len(rvec_1d_cols) > 0
    has_2d = len(rvec_2d_cols) > 0
    
    # Mixed if: (scalar + 1D) or (scalar + 2D) or (1D + 2D) or all three
    return (has_scalar and (has_1d or has_2d)) or (has_1d and has_2d)


# =============================================================================
# Phase 13.6.A: Same-Depth Flatten (NumPy Backend)
# =============================================================================

def _flatten_numpy_1level(
    data: Dict[str, np.ndarray],
    rvec_columns: List[str],
    parent_id_column: Optional[str]
) -> Dict[str, np.ndarray]:
    """
    NumPy backend for 1-level flatten.
    
    Phase 13.6.A: Original implementation.
    Phase 13.6.G+: parent_id_column now optional.
    """
    # Get parent_ids if available
    if parent_id_column is not None and parent_id_column in data:
        parent_ids = data[parent_id_column]
        n_events = len(parent_ids)
    else:
        parent_ids = None
        # Infer n_events from first rvec column
        ref_col = rvec_columns[0]
        n_events = len(data[ref_col])
    
    # Get structure from first rvec column
    ref_col = rvec_columns[0]
    sizes = np.array([len(rv) for rv in data[ref_col]], dtype=np.int64)
    total = sizes.sum()
    
    if total == 0:
        # Empty result
        result = {
            'track_idx': np.array([], dtype=np.int64),
        }
        if parent_ids is not None:
            result[parent_id_column] = np.array([], dtype=parent_ids.dtype)
        for col in rvec_columns:
            result[col] = np.array([], dtype=_infer_dtype(data[col]))
        return result
    
    # Preallocate
    result = {
        'track_idx': np.empty(total, dtype=np.int64),
    }
    if parent_ids is not None:
        result[parent_id_column] = np.empty(total, dtype=parent_ids.dtype)
    for col in rvec_columns:
        dtype = _infer_dtype(data[col])
        result[col] = np.empty(total, dtype=dtype)
    
    # Fill
    offset = 0
    for e in range(n_events):
        n = sizes[e]
        if n == 0:
            continue
        end = offset + n
        
        if parent_ids is not None:
            result[parent_id_column][offset:end] = parent_ids[e]
        result['track_idx'][offset:end] = np.arange(n)
        
        for col in rvec_columns:
            result[col][offset:end] = np.asarray(data[col][e])
        
        offset = end
    
    return result


def _flatten_numpy_2level(
    data: Dict[str, np.ndarray],
    rvec_columns: List[str],
    parent_id_column: Optional[str]
) -> Dict[str, np.ndarray]:
    """
    NumPy backend for 2-level flatten.
    
    Phase 13.6.A: Original implementation.
    Phase 13.6.G+: parent_id_column now optional.
    """
    # Get parent_ids if available
    if parent_id_column is not None and parent_id_column in data:
        parent_ids = data[parent_id_column]
        n_events = len(parent_ids)
    else:
        parent_ids = None
        # Infer n_events from first rvec column
        ref_col = rvec_columns[0]
        n_events = len(data[ref_col])
    
    # Get structure from first rvec column
    ref_col = rvec_columns[0]
    
    # Count total clusters
    total = 0
    for event in data[ref_col]:
        for track in event:
            total += len(track)
    
    if total == 0:
        # Empty result
        result = {
            'track_idx': np.array([], dtype=np.int64),
            'cluster_idx': np.array([], dtype=np.int64),
        }
        if parent_ids is not None:
            result[parent_id_column] = np.array([], dtype=parent_ids.dtype)
        for col in rvec_columns:
            result[col] = np.array([], dtype=_infer_dtype_2d(data[col]))
        return result
    
    # Preallocate
    result = {
        'track_idx': np.empty(total, dtype=np.int64),
        'cluster_idx': np.empty(total, dtype=np.int64),
    }
    if parent_ids is not None:
        result[parent_id_column] = np.empty(total, dtype=parent_ids.dtype)
    for col in rvec_columns:
        dtype = _infer_dtype_2d(data[col])
        result[col] = np.empty(total, dtype=dtype)
    
    # Fill
    offset = 0
    for e in range(n_events):
        event_data = data[ref_col][e]
        n_tracks = len(event_data)
        
        for t in range(n_tracks):
            track_data = event_data[t]
            n_clusters = len(track_data)
            
            if n_clusters == 0:
                continue
            
            end = offset + n_clusters
            
            if parent_ids is not None:
                result[parent_id_column][offset:end] = parent_ids[e]
            result['track_idx'][offset:end] = t
            result['cluster_idx'][offset:end] = np.arange(n_clusters)
            
            for col in rvec_columns:
                result[col][offset:end] = np.asarray(data[col][e][t])
            
            offset = end
    
    return result


# =============================================================================
# Phase 13.6.A: Same-Depth Flatten (Awkward Backend)
# =============================================================================

def _flatten_awkward_1level(
    data: Dict[str, np.ndarray],
    rvec_columns: List[str],
    parent_id_column: str
) -> Dict[str, np.ndarray]:
    """
    Awkward Array backend for 1-level flatten.
    
    Phase 13.6.A: Delegates to NumPy (Awkward provides no advantage here).
    """
    return _flatten_numpy_1level(data, rvec_columns, parent_id_column)


def _flatten_awkward_2level(
    data: Dict[str, np.ndarray],
    rvec_columns: List[str],
    parent_id_column: str
) -> Dict[str, np.ndarray]:
    """
    Awkward Array backend for 2-level flatten.
    
    Phase 13.6.A: Originally intended for Awkward performance benefits.
    Phase 13.6.C: Delegates to NumPy (Awkward conversion overhead eliminated).
    
    The previous implementation converted to Awkward Array just to count
    elements, then fell back to NumPy anyway. This was wasteful - the
    Awkward conversion has O(n) overhead with high constant factor due to
    Python list comprehension + ak.Array construction.
    
    Now directly delegates to NumPy backend (same as _flatten_awkward_1level).
    """
    return _flatten_numpy_2level(data, rvec_columns, parent_id_column)


# =============================================================================
# Phase 13.6.A-ext: Mixed-Depth Flatten
# =============================================================================

def _flatten_depth_0(
    data: Dict[str, np.ndarray],
    scalar_cols: List[str],
    parent_id_column: Optional[str]
) -> Dict[str, np.ndarray]:
    """
    No flattening needed - return scalars as-is.
    
    Phase 13.6.A-ext: NEW function.
    Phase 13.6.G+: parent_id_column now optional.
    """
    result = {}
    if parent_id_column is not None and parent_id_column in data:
        result[parent_id_column] = data[parent_id_column]
    for col in scalar_cols:
        if col != parent_id_column:
            result[col] = data[col]
    return result


def _flatten_depth_1_mixed(
    data: Dict[str, np.ndarray],
    scalar_cols: List[str],
    rvec_1d_cols: List[str],
    parent_id_column: Optional[str]
) -> Dict[str, np.ndarray]:
    """
    Flatten to track level, replicating scalars.
    
    Phase 13.6.A-ext: NEW function.
    Phase 13.6.G+: parent_id_column now optional.
    
    Input (1 event):
        event_id: 100           # scalar
        multiplicity: 3         # scalar
        track_pt: [1.0, 2.0, 3.0]  # 1D
    
    Output (3 rows):
        event_id  multiplicity  track_idx  track_pt
        100       3             0          1.0
        100       3             1          2.0
        100       3             2          3.0
    """
    # Get parent_ids if available
    if parent_id_column is not None and parent_id_column in data:
        parent_ids = data[parent_id_column]
        n_events = len(parent_ids)
    else:
        parent_ids = None
        # Infer n_events from first 1D column
        ref_col = rvec_1d_cols[0] if rvec_1d_cols else scalar_cols[0]
        n_events = len(data[ref_col])
    
    # Get reference 1D column for structure
    ref_1d_col = rvec_1d_cols[0]
    sizes = np.array([len(rv) for rv in data[ref_1d_col]], dtype=np.int64)
    total_rows = sizes.sum()
    
    # Handle empty case
    if total_rows == 0:
        result = {
            'track_idx': np.array([], dtype=np.int64),
        }
        if parent_ids is not None:
            result[parent_id_column] = np.array([], dtype=parent_ids.dtype)
        for col in scalar_cols:
            if col != parent_id_column:
                result[col] = np.array([], dtype=data[col].dtype)
        for col in rvec_1d_cols:
            result[col] = np.array([], dtype=_infer_dtype(data[col]))
        return result
    
    # Preallocate output arrays
    result = {}
    
    # Parent ID column (optional)
    if parent_ids is not None:
        result[parent_id_column] = np.empty(total_rows, dtype=parent_ids.dtype)
    
    # Scalar columns (will be replicated)
    for col in scalar_cols:
        if col != parent_id_column:
            result[col] = np.empty(total_rows, dtype=data[col].dtype)
    
    # Track index
    result['track_idx'] = np.empty(total_rows, dtype=np.int64)
    
    # 1D columns
    for col in rvec_1d_cols:
        dtype = _infer_dtype(data[col])
        result[col] = np.empty(total_rows, dtype=dtype)
    
    # Fill arrays
    offset = 0
    for e in range(n_events):
        n_tracks = sizes[e]
        if n_tracks == 0:
            continue
        
        end = offset + n_tracks
        
        # Replicate parent ID (if available)
        if parent_ids is not None:
            result[parent_id_column][offset:end] = parent_ids[e]
        
        # Replicate scalars
        for col in scalar_cols:
            if col != parent_id_column:
                result[col][offset:end] = data[col][e]
        
        # Generate track index
        result['track_idx'][offset:end] = np.arange(n_tracks)
        
        # Copy 1D values
        for col in rvec_1d_cols:
            result[col][offset:end] = np.asarray(data[col][e])
        
        offset = end
    
    return result


def _flatten_depth_2_mixed(
    data: Dict[str, np.ndarray],
    scalar_cols: List[str],
    rvec_1d_cols: List[str],
    rvec_2d_cols: List[str],
    parent_id_column: Optional[str]
) -> Dict[str, np.ndarray]:
    """
    Flatten to cluster level, replicating scalars and track values.
    
    Phase 13.6.A-ext: NEW function.
    Phase 13.6.G+: parent_id_column now optional.
    
    WARNING: Tracks with zero clusters will "disappear" from the output.
    Their 1D values are NOT preserved. Use normalized mode if you need all tracks.
    
    Input (1 event):
        event_id: 100                          # scalar
        multiplicity: 3                        # scalar
        track_pt: [1.0, 2.0, 3.0]             # 1D (3 tracks)
        cluster_Q: [[10,20], [30], [40,50,60]] # 2D (2+1+3 = 6 clusters)
    
    Output (6 rows):
        event_id  multiplicity  track_idx  track_pt  cluster_idx  cluster_Q
        100       3             0          1.0       0            10
        100       3             0          1.0       1            20
        100       3             1          2.0       0            30
        100       3             2          3.0       0            40
        100       3             2          3.0       1            50
        100       3             2          3.0       2            60
    """
    # Get parent_ids if available
    if parent_id_column is not None and parent_id_column in data:
        parent_ids = data[parent_id_column]
        n_events = len(parent_ids)
    else:
        parent_ids = None
        # Infer n_events from first 2D column
        ref_col = rvec_2d_cols[0]
        n_events = len(data[ref_col])
    
    # Get reference 2D column for structure
    ref_2d_col = rvec_2d_cols[0]
    
    # Count total clusters
    total_rows = 0
    for event_data in data[ref_2d_col]:
        for track_data in event_data:
            total_rows += len(track_data)
    
    # Handle empty case
    if total_rows == 0:
        result = {
            'track_idx': np.array([], dtype=np.int64),
            'cluster_idx': np.array([], dtype=np.int64),
        }
        if parent_ids is not None:
            result[parent_id_column] = np.array([], dtype=parent_ids.dtype)
        for col in scalar_cols:
            if col != parent_id_column:
                result[col] = np.array([], dtype=data[col].dtype)
        for col in rvec_1d_cols:
            result[col] = np.array([], dtype=_infer_dtype(data[col]))
        for col in rvec_2d_cols:
            result[col] = np.array([], dtype=_infer_dtype_2d(data[col]))
        return result
    
    # Preallocate output arrays
    result = {}
    
    # Parent ID column (optional)
    if parent_ids is not None:
        result[parent_id_column] = np.empty(total_rows, dtype=parent_ids.dtype)
    
    # Scalar columns
    for col in scalar_cols:
        if col != parent_id_column:
            result[col] = np.empty(total_rows, dtype=data[col].dtype)
    
    # Track index
    result['track_idx'] = np.empty(total_rows, dtype=np.int64)
    
    # 1D columns (will be replicated to cluster level)
    for col in rvec_1d_cols:
        dtype = _infer_dtype(data[col])
        result[col] = np.empty(total_rows, dtype=dtype)
    
    # Cluster index
    result['cluster_idx'] = np.empty(total_rows, dtype=np.int64)
    
    # 2D columns
    for col in rvec_2d_cols:
        dtype = _infer_dtype_2d(data[col])
        result[col] = np.empty(total_rows, dtype=dtype)
    
    # Fill arrays
    offset = 0
    for e in range(n_events):
        event_2d = data[ref_2d_col][e]
        n_tracks = len(event_2d)
        
        for t in range(n_tracks):
            track_2d = event_2d[t]
            n_clusters = len(track_2d)
            
            if n_clusters == 0:
                continue
            
            end = offset + n_clusters
            
            # Replicate parent ID (if available)
            if parent_ids is not None:
                result[parent_id_column][offset:end] = parent_ids[e]
            
            # Replicate scalars
            for col in scalar_cols:
                if col != parent_id_column:
                    result[col][offset:end] = data[col][e]
            
            # Replicate track index
            result['track_idx'][offset:end] = t
            
            # Replicate 1D values (track-level → cluster-level)
            for col in rvec_1d_cols:
                track_value = data[col][e][t]  # Single value for this track
                result[col][offset:end] = track_value
            
            # Generate cluster index
            result['cluster_idx'][offset:end] = np.arange(n_clusters)
            
            # Copy 2D values
            for col in rvec_2d_cols:
                result[col][offset:end] = np.asarray(data[col][e][t])
            
            offset = end
    
    return result


def _build_output_dataframe(
    result: Dict[str, np.ndarray],
    parent_id_column: Optional[str],
    scalar_cols: List[str],
    rvec_1d_cols: List[str],
    rvec_2d_cols: List[str],
    target_depth: int
) -> pd.DataFrame:
    """
    Build DataFrame with deterministic column order.
    
    Phase 13.6.A-ext: NEW function (fixed from v0.1).
    Phase 13.6.G+: parent_id_column now optional.
    
    Order:
        1. parent_id_column (e.g., 'event_id') - if provided
        2. Scalar columns (input order preserved)
        3. 'track_idx' (if depth >= 1)
        4. 1D columns (input order preserved)
        5. 'cluster_idx' (if depth == 2)
        6. 2D columns (input order preserved)
    """
    ordered_columns = []
    
    # Parent ID column (optional)
    if parent_id_column is not None and parent_id_column in result:
        ordered_columns.append(parent_id_column)
    
    # Scalar columns (input order, excluding parent_id)
    for col in scalar_cols:
        if col != parent_id_column and col in result:
            ordered_columns.append(col)
    
    # Track index and 1D columns
    if target_depth >= 1:
        if 'track_idx' in result:
            ordered_columns.append('track_idx')
        for col in rvec_1d_cols:
            if col in result:
                ordered_columns.append(col)
    
    # Cluster index and 2D columns
    if target_depth == 2:
        if 'cluster_idx' in result:
            ordered_columns.append('cluster_idx')
        for col in rvec_2d_cols:
            if col in result:
                ordered_columns.append(col)
    
    return pd.DataFrame({col: result[col] for col in ordered_columns})


# =============================================================================
# Main API Functions
# =============================================================================

def flatten_to_dataframe(
    data: Dict[str, np.ndarray],
    columns: Optional[List[str]] = None,
    rvec_columns: Optional[List[str]] = None,  # DEPRECATED (Phase 13.6.A compat)
    parent_id_column: Optional[str] = 'event_id',
    backend: FlattenBackend = FlattenBackend.AUTO,
    join: str = 'inner',
) -> pd.DataFrame:
    """
    Flatten RVec columns to pandas DataFrame.
    
    Phase 13.6.A: Same-depth columns (backward compatible via rvec_columns)
    Phase 13.6.A-ext: Mixed-depth columns (scalar + 1D + 2D via columns)
    Phase 13.6.C: Join strategy parameter
    Phase 13.6.G+: parent_id_column can be None for simple operations
    
    Args:
        data: Dict from rdf.AsNumpy() containing columns
        columns: Column names to flatten (supports mixed depths)
                 NEW in 13.6.A-ext
        rvec_columns: DEPRECATED - use 'columns' instead
                      Kept for Phase 13.6.A backward compatibility
        parent_id_column: Parent ID column name (default: 'event_id')
                         Set to None if no parent tracking needed.
        backend: Flatten backend (default: AUTO)
        join: Join strategy for mixed-depth columns (Phase 13.6.C)
              - 'inner': Intersection of indices (default, no NaN)
              - 'outer': Union of indices (NaN for missing)
              - 'left': All from deeper operand
              - 'right': All from shallower operand
    
    Returns:
        Flat pandas DataFrame with appropriate index columns
    
    Backward Compatibility:
        - If 'rvec_columns' is provided, it's treated as 'columns'
        - Deprecation warning is issued
        - Cannot specify both 'columns' and 'rvec_columns'
    
    Examples:
        # Phase 13.6.A style (still works):
        >>> df = flatten_to_dataframe(data, rvec_columns=['track_pt'])
        
        # Phase 13.6.A-ext style (recommended):
        >>> df = flatten_to_dataframe(data, columns=['track_pt'])
        
        # Mixed depths (NEW):
        >>> df = flatten_to_dataframe(data, 
        ...     columns=['cluster_Q', 'track_pt', 'multiplicity'])
        
        # With join strategy (Phase 13.6.C):
        >>> df = flatten_to_dataframe(data,
        ...     columns=['cluster_Q', 'track_pt'],
        ...     join='outer')  # NaN for missing
        
        # Without parent tracking (Phase 13.6.G+):
        >>> df = flatten_to_dataframe(data, columns=['track_pt'], parent_id_column=None)
    """
    # Handle backward compatibility
    if rvec_columns is not None:
        if columns is not None:
            raise ValueError(
                "Cannot specify both 'columns' and 'rvec_columns'. "
                "Use 'columns' (rvec_columns is deprecated)."
            )
        warnings.warn(
            "'rvec_columns' parameter is deprecated. Use 'columns' instead. "
            "The new 'columns' parameter supports mixed nesting depths "
            "(scalars + RVec + RVec<RVec>).",
            DeprecationWarning,
            stacklevel=2
        )
        columns = rvec_columns
    
    if columns is None:
        raise ValueError("Must specify 'columns' parameter")
    
    if not columns:
        raise ValueError("'columns' list cannot be empty")
    
    # Validate join parameter (Phase 13.6.C)
    valid_joins = ('inner', 'outer', 'left', 'right')
    if join not in valid_joins:
        raise ValueError(
            f"Invalid join type '{join}'. Must be one of: {valid_joins}"
        )
    
    # Validate columns exist
    # Phase 13.6.G+: parent_id_column is optional
    cols_to_validate = list(columns)
    if parent_id_column is not None:
        cols_to_validate.append(parent_id_column)
    _validate_columns_exist(data, cols_to_validate)
    
    # Classify columns by depth
    scalar_cols, rvec_1d_cols, rvec_2d_cols = _classify_columns_by_depth(data, columns)
    target_depth = _determine_target_depth(scalar_cols, rvec_1d_cols, rvec_2d_cols)
    is_mixed = _is_mixed_depth(scalar_cols, rvec_1d_cols, rvec_2d_cols)
    
    # Validate structure (only if parent_id_column provided)
    if parent_id_column is not None:
        _validate_mixed_structure(data, scalar_cols, rvec_1d_cols, rvec_2d_cols, parent_id_column)
    
    # Backend dispatch
    if is_mixed:
        # Mixed-depth: NumPy only (Phase 13.6.A-ext)
        if backend == FlattenBackend.AWKWARD:
            raise NotImplementedError(
                "Awkward Array backend does not yet support mixed-depth flattening. "
                "Use backend=FlattenBackend.NUMPY or AUTO."
            )
        if backend == FlattenBackend.CPP:
            raise NotImplementedError(
                "C++ backend does not yet support mixed-depth flattening."
            )
        
        # Dispatch by target depth
        if target_depth == 0:
            result = _flatten_depth_0(data, scalar_cols, parent_id_column)
        elif target_depth == 1:
            result = _flatten_depth_1_mixed(data, scalar_cols, rvec_1d_cols, parent_id_column)
        else:  # target_depth == 2
            result = _flatten_depth_2_mixed(
                data, scalar_cols, rvec_1d_cols, rvec_2d_cols, parent_id_column
            )
        
        return _build_output_dataframe(
            result, parent_id_column, scalar_cols, rvec_1d_cols, rvec_2d_cols, target_depth
        )
    
    # Same-depth: Use Phase 13.6.A backends
    all_rvec_cols = rvec_1d_cols + rvec_2d_cols
    
    if not all_rvec_cols:
        # Only scalars requested (treated as depth 0 mixed)
        result = _flatten_depth_0(data, scalar_cols, parent_id_column)
        return _build_output_dataframe(
            result, parent_id_column, scalar_cols, [], [], 0
        )
    
    # Validate same structure
    validate_same_structure(data, all_rvec_cols)
    
    # Determine if 1D or 2D
    is_2d = is_nested_rvec(data, all_rvec_cols[0])
    
    # Backend selection - restore Phase 13.6.A AUTO behavior
    use_backend = backend
    if use_backend == FlattenBackend.AUTO:
        # GPT8 FIX: Restore original AUTO selection for same-depth
        # - 2D + Awkward available → AWKWARD
        # - Otherwise → NUMPY
        use_backend = _select_backend_same_depth(data, all_rvec_cols)
    
    if use_backend == FlattenBackend.CPP:
        raise NotImplementedError("C++ backend not yet implemented")
    
    # Dispatch
    if is_2d:
        if use_backend == FlattenBackend.AWKWARD and awkward_available():
            result = _flatten_awkward_2level(data, all_rvec_cols, parent_id_column)
        else:
            result = _flatten_numpy_2level(data, all_rvec_cols, parent_id_column)
        return _build_output_dataframe(
            result, parent_id_column, [], [], all_rvec_cols, 2
        )
    else:
        if use_backend == FlattenBackend.AWKWARD and awkward_available():
            result = _flatten_awkward_1level(data, all_rvec_cols, parent_id_column)
        else:
            result = _flatten_numpy_1level(data, all_rvec_cols, parent_id_column)
        return _build_output_dataframe(
            result, parent_id_column, [], all_rvec_cols, [], 1
        )


def flatten_to_dict(
    data: Dict[str, np.ndarray],
    columns: Optional[List[str]] = None,
    rvec_columns: Optional[List[str]] = None,
    parent_id_column: str = "event_id",
    backend: FlattenBackend = FlattenBackend.AUTO
) -> Dict[str, np.ndarray]:
    """
    Flatten RVec columns to dict of arrays (without DataFrame conversion).
    
    Phase 13.6.A: Original API.
    
    Same arguments as flatten_to_dataframe().
    Returns dict of numpy arrays instead of DataFrame.
    """
    # Handle backward compatibility
    if rvec_columns is not None:
        if columns is not None:
            raise ValueError(
                "Cannot specify both 'columns' and 'rvec_columns'. "
                "Use 'columns' (rvec_columns is deprecated)."
            )
        warnings.warn(
            "'rvec_columns' parameter is deprecated. Use 'columns' instead.",
            DeprecationWarning,
            stacklevel=2
        )
        columns = rvec_columns
    
    if columns is None:
        raise ValueError("Must specify 'columns' parameter")
    
    _validate_columns_exist(data, columns + [parent_id_column])
    
    scalar_cols, rvec_1d_cols, rvec_2d_cols = _classify_columns_by_depth(data, columns)
    target_depth = _determine_target_depth(scalar_cols, rvec_1d_cols, rvec_2d_cols)
    is_mixed = _is_mixed_depth(scalar_cols, rvec_1d_cols, rvec_2d_cols)
    
    _validate_mixed_structure(data, scalar_cols, rvec_1d_cols, rvec_2d_cols, parent_id_column)
    
    if is_mixed:
        if target_depth == 0:
            return _flatten_depth_0(data, scalar_cols, parent_id_column)
        elif target_depth == 1:
            return _flatten_depth_1_mixed(data, scalar_cols, rvec_1d_cols, parent_id_column)
        else:
            return _flatten_depth_2_mixed(
                data, scalar_cols, rvec_1d_cols, rvec_2d_cols, parent_id_column
            )
    
    all_rvec_cols = rvec_1d_cols + rvec_2d_cols
    
    if not all_rvec_cols:
        return _flatten_depth_0(data, scalar_cols, parent_id_column)
    
    validate_same_structure(data, all_rvec_cols)
    is_2d = is_nested_rvec(data, all_rvec_cols[0])
    
    if is_2d:
        return _flatten_numpy_2level(data, all_rvec_cols, parent_id_column)
    else:
        return _flatten_numpy_1level(data, all_rvec_cols, parent_id_column)


# =============================================================================
# Phase 13.6.A-ext: Normalized Tables Output
# =============================================================================

def flatten_to_tables(
    data: Dict[str, np.ndarray],
    columns: List[str],
    parent_id_column: str = "event_id"
) -> Dict[str, pd.DataFrame]:
    """
    Flatten to normalized tables (no replication).
    
    Phase 13.6.A-ext: NEW function.
    
    Returns separate DataFrames for each depth level, suitable for
    joins and AliasDataFrame integration.
    
    Args:
        data: Dict from rdf.AsNumpy()
        columns: Column names to include
        parent_id_column: Parent ID column (default: 'event_id')
    
    Returns:
        Dict with keys 'events', 'tracks', 'clusters' (as applicable).
        Only levels with requested columns are included.
    
    Join Keys:
        - events ↔ tracks: parent_id_column
        - tracks ↔ clusters: (parent_id_column, track_idx)
    
    Example:
        >>> tables = flatten_to_tables(data, 
        ...     columns=['multiplicity', 'track_pt', 'cluster_Q'])
        >>> tables.keys()
        dict_keys(['events', 'tracks', 'clusters'])
        >>> 
        >>> # Join for analysis
        >>> merged = tables['tracks'].merge(tables['events'], on='event_id')
    """
    _validate_columns_exist(data, columns + [parent_id_column])
    
    scalar_cols, rvec_1d_cols, rvec_2d_cols = _classify_columns_by_depth(data, columns)
    
    _validate_mixed_structure(data, scalar_cols, rvec_1d_cols, rvec_2d_cols, parent_id_column)
    
    result = {}
    
    # Events table (scalars)
    if scalar_cols:
        result['events'] = _build_events_table(data, scalar_cols, parent_id_column)
    
    # Tracks table (1D columns)
    if rvec_1d_cols:
        result['tracks'] = _build_tracks_table(data, rvec_1d_cols, parent_id_column)
    
    # Clusters table (2D columns)
    if rvec_2d_cols:
        result['clusters'] = _build_clusters_table(data, rvec_2d_cols, parent_id_column)
    
    return result


def _build_events_table(
    data: Dict[str, np.ndarray],
    scalar_cols: List[str],
    parent_id_column: str
) -> pd.DataFrame:
    """Build events table (no flattening, just scalars)."""
    result = {parent_id_column: data[parent_id_column]}
    for col in scalar_cols:
        if col != parent_id_column:
            result[col] = data[col]
    
    # Preserve column order
    ordered_cols = [parent_id_column] + [c for c in scalar_cols if c != parent_id_column]
    return pd.DataFrame({col: result[col] for col in ordered_cols if col in result})


def _build_tracks_table(
    data: Dict[str, np.ndarray],
    rvec_1d_cols: List[str],
    parent_id_column: str
) -> pd.DataFrame:
    """Build tracks table (flatten 1D only, no scalar replication)."""
    parent_ids = data[parent_id_column]
    n_events = len(parent_ids)
    
    ref_col = rvec_1d_cols[0]
    sizes = np.array([len(rv) for rv in data[ref_col]], dtype=np.int64)
    total = sizes.sum()
    
    if total == 0:
        result = {
            parent_id_column: np.array([], dtype=parent_ids.dtype),
            'track_idx': np.array([], dtype=np.int64),
        }
        for col in rvec_1d_cols:
            result[col] = np.array([], dtype=_infer_dtype(data[col]))
        return pd.DataFrame(result)
    
    # Preallocate
    result = {
        parent_id_column: np.empty(total, dtype=parent_ids.dtype),
        'track_idx': np.empty(total, dtype=np.int64),
    }
    for col in rvec_1d_cols:
        result[col] = np.empty(total, dtype=_infer_dtype(data[col]))
    
    # Fill
    offset = 0
    for e in range(n_events):
        n = sizes[e]
        if n == 0:
            continue
        end = offset + n
        
        result[parent_id_column][offset:end] = parent_ids[e]
        result['track_idx'][offset:end] = np.arange(n)
        for col in rvec_1d_cols:
            result[col][offset:end] = np.asarray(data[col][e])
        
        offset = end
    
    # Column order
    ordered_cols = [parent_id_column, 'track_idx'] + list(rvec_1d_cols)
    return pd.DataFrame({col: result[col] for col in ordered_cols})


def _build_clusters_table(
    data: Dict[str, np.ndarray],
    rvec_2d_cols: List[str],
    parent_id_column: str
) -> pd.DataFrame:
    """Build clusters table (flatten 2D only, no track replication)."""
    parent_ids = data[parent_id_column]
    n_events = len(parent_ids)
    
    ref_col = rvec_2d_cols[0]
    
    # Count total clusters
    total = sum(
        len(track)
        for event in data[ref_col]
        for track in event
    )
    
    if total == 0:
        result = {
            parent_id_column: np.array([], dtype=parent_ids.dtype),
            'track_idx': np.array([], dtype=np.int64),
            'cluster_idx': np.array([], dtype=np.int64),
        }
        for col in rvec_2d_cols:
            result[col] = np.array([], dtype=_infer_dtype_2d(data[col]))
        return pd.DataFrame(result)
    
    # Preallocate
    result = {
        parent_id_column: np.empty(total, dtype=parent_ids.dtype),
        'track_idx': np.empty(total, dtype=np.int64),
        'cluster_idx': np.empty(total, dtype=np.int64),
    }
    for col in rvec_2d_cols:
        result[col] = np.empty(total, dtype=_infer_dtype_2d(data[col]))
    
    # Fill
    offset = 0
    for e in range(n_events):
        for t, track_data in enumerate(data[ref_col][e]):
            n = len(track_data)
            if n == 0:
                continue
            end = offset + n
            
            result[parent_id_column][offset:end] = parent_ids[e]
            result['track_idx'][offset:end] = t
            result['cluster_idx'][offset:end] = np.arange(n)
            for col in rvec_2d_cols:
                result[col][offset:end] = np.asarray(data[col][e][t])
            
            offset = end
    
    # Column order
    ordered_cols = [parent_id_column, 'track_idx', 'cluster_idx'] + list(rvec_2d_cols)
    return pd.DataFrame({col: result[col] for col in ordered_cols})
