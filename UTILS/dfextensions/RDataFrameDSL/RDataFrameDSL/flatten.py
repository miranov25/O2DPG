"""
Phase 13.6.A: RDataFrame Flattening

Flatten hierarchical RVec data to flat arrays for TTree::Draw-like functionality.
Multiple backends: NumPy (baseline), Awkward (2-level), C++ (production).

Usage:
    from RDataFrameDSL.flatten import flatten_to_dataframe, FlattenBackend
    
    # Get data from RDataFrame
    data = rdf.AsNumpy(['event_id', 'track_pt', 'track_eta'])
    
    # Flatten RVec columns
    df = flatten_to_dataframe(
        data, 
        rvec_columns=['track_pt', 'track_eta'],
        parent_id_column='event_id'
    )
    
    # Result: DataFrame with event_id, track_idx, track_pt, track_eta

Index Semantics Contract:
    - event_id: Replicated verbatim from input (physics identifier)
    - track_idx: Generated 0-based index within parent event
    - cluster_idx: Generated 0-based index within parent track (2-level)
    - All backends produce identical output (exact bit equality)
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import warnings
import logging
import os

logger = logging.getLogger(__name__)


# =============================================================================
# Backend Enum
# =============================================================================

class FlattenBackend(Enum):
    """Available flattening backends."""
    NUMPY = "numpy"
    AWKWARD = "awkward"
    CPP = "cpp"
    AUTO = "auto"


# =============================================================================
# Backend Availability Checks
# =============================================================================

def awkward_available() -> bool:
    """Check if Awkward Array is available."""
    try:
        import awkward
        return True
    except ImportError:
        return False


def cpp_helper_available() -> bool:
    """Check if C++ flatten helper is registered."""
    # TODO: Implement when C++ helper is added
    return False


# =============================================================================
# Type Detection
# =============================================================================

def is_rvec(obj: Any) -> bool:
    """Check if object is an RVec or RVec-like."""
    type_name = type(obj).__name__
    if 'RVec' in type_name:
        return True
    # Also handle numpy arrays that came from RVec
    if hasattr(obj, '__len__') and not isinstance(obj, (str, bytes)):
        return True
    return False


def is_nested_rvec(data: Dict[str, np.ndarray], column: str) -> bool:
    """
    Check if column contains nested RVec (RVec<RVec<T>>).
    
    Returns True if the first non-empty element is itself iterable.
    """
    col_data = data[column]
    for item in col_data:
        if len(item) > 0:
            first_elem = item[0]
            # Check if first element is itself iterable (nested)
            if hasattr(first_elem, '__len__') and not isinstance(first_elem, (str, bytes)):
                return True
            return False
    return False


def get_nesting_depth(data: Dict[str, np.ndarray], column: str) -> int:
    """
    Get nesting depth of RVec column.
    
    Returns:
        1 for RVec<T>
        2 for RVec<RVec<T>>
    """
    if is_nested_rvec(data, column):
        return 2
    return 1


def infer_dtype(rvec_sample: Any) -> np.dtype:
    """Infer numpy dtype from RVec sample."""
    if len(rvec_sample) > 0:
        arr = np.asarray(rvec_sample)
        return arr.dtype
    # Default to float64 for empty
    return np.dtype('float64')


# =============================================================================
# Backend Selection
# =============================================================================

def select_backend(
    data: Dict[str, np.ndarray], 
    rvec_columns: List[str],
    verbose: bool = False
) -> FlattenBackend:
    """
    AUTO backend selection heuristic.
    
    Rules:
    1. If C++ helper unavailable → NumPy
    2. If nested depth >= 2 (RVec<RVec>) → Awkward (if available) or C++
    3. If total tracks > 1M → C++ (performance)
    4. If total tracks < 100k → NumPy (simplicity)
    5. Else → C++ (default for production)
    
    Args:
        data: Output from rdf.AsNumpy()
        rvec_columns: Columns to flatten
        verbose: Log selection decision
    
    Returns:
        Selected backend
    """
    # Estimate total items (use first column)
    first_col = rvec_columns[0]
    total_items = sum(len(rv) for rv in data[first_col])
    
    # Check for nested RVec
    is_nested = any(is_nested_rvec(data, col) for col in rvec_columns)
    
    reason = ""
    
    # Rule 1: No C++ → NumPy
    if not cpp_helper_available():
        # Rule 2: Nested → Awkward if available
        if is_nested:
            if awkward_available():
                backend = FlattenBackend.AWKWARD
                reason = "nested RVec, Awkward available"
            else:
                backend = FlattenBackend.NUMPY
                reason = "nested RVec, fallback to NumPy (no Awkward)"
        # Rule 4: Small → NumPy
        elif total_items < 100_000:
            backend = FlattenBackend.NUMPY
            reason = f"small dataset ({total_items} items)"
        else:
            backend = FlattenBackend.NUMPY
            reason = "no C++ helper available"
    else:
        # C++ available
        if is_nested:
            if awkward_available():
                backend = FlattenBackend.AWKWARD
                reason = "nested RVec, Awkward preferred"
            else:
                backend = FlattenBackend.CPP
                reason = "nested RVec, C++ fallback"
        elif total_items > 1_000_000:
            backend = FlattenBackend.CPP
            reason = f"large dataset ({total_items} items)"
        elif total_items < 100_000:
            backend = FlattenBackend.NUMPY
            reason = f"small dataset ({total_items} items)"
        else:
            backend = FlattenBackend.CPP
            reason = "default production backend"
    
    if verbose or os.getenv("DFEXT_DEBUG"):
        logger.info(f"AUTO selected {backend.value}: {reason}")
    
    return backend


# =============================================================================
# Validation
# =============================================================================

def validate_same_structure(
    data: Dict[str, np.ndarray], 
    rvec_columns: List[str]
) -> None:
    """
    Validate all RVec columns have same jagged structure.
    
    All columns must have same number of elements per parent.
    
    Raises:
        ValueError: If columns have different structures
    """
    if len(rvec_columns) < 2:
        return
    
    ref_col = rvec_columns[0]
    ref_lengths = [len(rv) for rv in data[ref_col]]
    
    for col in rvec_columns[1:]:
        col_lengths = [len(rv) for rv in data[col]]
        if col_lengths != ref_lengths:
            raise ValueError(
                f"RVec columns have different structures. "
                f"'{ref_col}' lengths: {ref_lengths[:5]}..., "
                f"'{col}' lengths: {col_lengths[:5]}..."
                f"\nAll columns in one flatten call must have identical jagged structure."
            )


# =============================================================================
# NumPy Backend (Baseline)
# =============================================================================

def _flatten_numpy_1level(
    data: Dict[str, np.ndarray],
    rvec_columns: List[str],
    parent_id_column: str
) -> Dict[str, np.ndarray]:
    """
    Flatten 1-level RVec columns using NumPy preallocate.
    
    This is the proven baseline (0.056s for 500k tracks).
    """
    # Get parent IDs and compute sizes
    parent_ids = data[parent_id_column]
    n_events = len(parent_ids)
    
    # Get sizes from first RVec column
    first_col = rvec_columns[0]
    sizes = np.array([len(rv) for rv in data[first_col]], dtype=np.int64)
    total = sizes.sum()
    
    if total == 0:
        # Handle empty case
        result = {
            parent_id_column: np.array([], dtype=parent_ids.dtype),
            'track_idx': np.array([], dtype=np.int64),
        }
        for col in rvec_columns:
            dtype = infer_dtype(data[col][0]) if n_events > 0 else np.float64
            result[col] = np.array([], dtype=dtype)
        return result
    
    # Preallocate output arrays
    # event_id: replicated parent IDs
    flat_parent_ids = np.empty(total, dtype=parent_ids.dtype)
    
    # track_idx: 0-based within each event
    flat_track_idx = np.empty(total, dtype=np.int64)
    
    # Flatten each RVec column
    flat_columns = {}
    for col in rvec_columns:
        # Infer dtype from first non-empty RVec
        dtype = infer_dtype(data[col][0]) if sizes[0] > 0 else np.float64
        for i, rv in enumerate(data[col]):
            if len(rv) > 0:
                dtype = np.asarray(rv).dtype
                break
        flat_columns[col] = np.empty(total, dtype=dtype)
    
    # Fill arrays
    offset = 0
    for i in range(n_events):
        n = sizes[i]
        if n == 0:
            continue
        
        # Parent ID: replicate
        flat_parent_ids[offset:offset+n] = parent_ids[i]
        
        # Track index: 0 to n-1
        flat_track_idx[offset:offset+n] = np.arange(n)
        
        # Data columns
        for col in rvec_columns:
            arr = np.asarray(data[col][i])
            flat_columns[col][offset:offset+n] = arr
        
        offset += n
    
    # Build result
    result = {
        parent_id_column: flat_parent_ids,
        'track_idx': flat_track_idx,
    }
    result.update(flat_columns)
    
    return result


def _flatten_numpy_2level(
    data: Dict[str, np.ndarray],
    rvec_columns: List[str],
    parent_id_column: str
) -> Dict[str, np.ndarray]:
    """
    Flatten 2-level RVec<RVec> columns using NumPy.
    
    Produces: event_id, track_idx, cluster_idx, data columns
    """
    parent_ids = data[parent_id_column]
    n_events = len(parent_ids)
    
    # Count total elements (iterate through 2 levels)
    first_col = rvec_columns[0]
    total = 0
    for event_data in data[first_col]:
        for track_data in event_data:
            total += len(track_data)
    
    if total == 0:
        result = {
            parent_id_column: np.array([], dtype=parent_ids.dtype),
            'track_idx': np.array([], dtype=np.int64),
            'cluster_idx': np.array([], dtype=np.int64),
        }
        for col in rvec_columns:
            result[col] = np.array([], dtype=np.float64)
        return result
    
    # Preallocate
    flat_parent_ids = np.empty(total, dtype=parent_ids.dtype)
    flat_track_idx = np.empty(total, dtype=np.int64)
    flat_cluster_idx = np.empty(total, dtype=np.int64)
    
    # Infer dtype for each column
    flat_columns = {}
    for col in rvec_columns:
        # Find first non-empty element
        dtype = np.float64
        for event_data in data[col]:
            for track_data in event_data:
                if len(track_data) > 0:
                    dtype = np.asarray(track_data).dtype
                    break
            else:
                continue
            break
        flat_columns[col] = np.empty(total, dtype=dtype)
    
    # Fill arrays
    offset = 0
    for e in range(n_events):
        event_data = data[first_col][e]
        n_tracks = len(event_data)
        
        for t in range(n_tracks):
            track_data = event_data[t]
            n_clusters = len(track_data)
            
            if n_clusters == 0:
                continue
            
            # Fill indices
            flat_parent_ids[offset:offset+n_clusters] = parent_ids[e]
            flat_track_idx[offset:offset+n_clusters] = t
            flat_cluster_idx[offset:offset+n_clusters] = np.arange(n_clusters)
            
            # Fill data columns
            for col in rvec_columns:
                arr = np.asarray(data[col][e][t])
                flat_columns[col][offset:offset+n_clusters] = arr
            
            offset += n_clusters
    
    # Build result
    result = {
        parent_id_column: flat_parent_ids,
        'track_idx': flat_track_idx,
        'cluster_idx': flat_cluster_idx,
    }
    result.update(flat_columns)
    
    return result


def flatten_numpy(
    data: Dict[str, np.ndarray],
    rvec_columns: List[str],
    parent_id_column: str
) -> Dict[str, np.ndarray]:
    """
    NumPy backend for flattening.
    
    Automatically detects 1-level vs 2-level nesting.
    """
    # Check nesting depth
    if is_nested_rvec(data, rvec_columns[0]):
        return _flatten_numpy_2level(data, rvec_columns, parent_id_column)
    else:
        return _flatten_numpy_1level(data, rvec_columns, parent_id_column)


# =============================================================================
# Awkward Backend (2-Level)
# =============================================================================

def flatten_awkward(
    data: Dict[str, np.ndarray],
    rvec_columns: List[str],
    parent_id_column: str
) -> Dict[str, np.ndarray]:
    """
    Awkward Array backend for flattening.
    
    For 1-level nesting: Delegates to NumPy backend (simpler, faster).
    For 2-level nesting: Uses Awkward Array (structural clarity).
    
    Roofline Analysis:
    - 1-level: NumPy preallocate is optimal, Awkward adds overhead
    - 2-level: Awkward's nested structure handling is cleaner
    """
    try:
        import awkward as ak
    except ImportError:
        raise ImportError(
            "Awkward Array required for this backend. "
            "Install with: pip install awkward"
        )
    
    parent_ids = data[parent_id_column]
    n_events = len(parent_ids)
    first_col = rvec_columns[0]
    
    # Check nesting depth
    is_2level = is_nested_rvec(data, first_col)
    
    if not is_2level:
        # 1-level: Delegate to NumPy backend (faster, simpler, no dtype issues)
        return flatten_numpy(data, rvec_columns, parent_id_column)
    
    # 2-level: Event → Track → Cluster
    # Infer original dtypes
    original_dtypes = {}
    for col in rvec_columns:
        dtype_found = False
        for event in data[col]:
            for track in event:
                if len(track) > 0:
                    arr = np.asarray(track)
                    if arr.dtype != np.dtype('O'):
                        original_dtypes[col] = arr.dtype
                        dtype_found = True
                        break
            if dtype_found:
                break
        if not dtype_found:
            original_dtypes[col] = np.float64
    
    # Build awkward arrays for 2-level
    jagged_data = {}
    for col in rvec_columns:
        nested_list = []
        for event in data[col]:
            event_list = []
            for track in event:
                event_list.append(np.asarray(track))
            nested_list.append(event_list)
        jagged_data[col] = ak.Array(nested_list)
    
    # Get structure from first column
    first_jagged = jagged_data[first_col]
    
    # Flatten values and restore original dtype
    flat_columns = {}
    for col in rvec_columns:
        flat_arr = ak.flatten(jagged_data[col], axis=None).to_numpy()
        flat_columns[col] = flat_arr.astype(original_dtypes[col])
    
    total = len(flat_columns[first_col])
    
    if total == 0:
        result = {
            parent_id_column: np.array([], dtype=parent_ids.dtype),
            'track_idx': np.array([], dtype=np.int64),
            'cluster_idx': np.array([], dtype=np.int64),
        }
        result.update(flat_columns)
        return result
    
    # Build indices
    flat_parent_ids = np.empty(total, dtype=parent_ids.dtype)
    flat_track_idx = np.empty(total, dtype=np.int64)
    flat_cluster_idx = np.empty(total, dtype=np.int64)
    
    offset = 0
    for e in range(n_events):
        n_tracks = len(first_jagged[e])
        for t in range(n_tracks):
            n_clusters = len(first_jagged[e][t])
            if n_clusters == 0:
                continue
            flat_parent_ids[offset:offset+n_clusters] = parent_ids[e]
            flat_track_idx[offset:offset+n_clusters] = t
            flat_cluster_idx[offset:offset+n_clusters] = np.arange(n_clusters)
            offset += n_clusters
    
    result = {
        parent_id_column: flat_parent_ids,
        'track_idx': flat_track_idx,
        'cluster_idx': flat_cluster_idx,
    }
    result.update(flat_columns)
    return result


# =============================================================================
# C++ Backend (Production)
# =============================================================================

def flatten_cpp(
    data: Dict[str, np.ndarray],
    rvec_columns: List[str],
    parent_id_column: str
) -> Dict[str, np.ndarray]:
    """
    C++ backend for flattening.
    
    Uses registered C++ helper functions via Phase 13.5.C.
    Falls back to NumPy if C++ helper not available.
    """
    if not cpp_helper_available():
        warnings.warn(
            "C++ flatten helper not available, falling back to NumPy backend.",
            UserWarning
        )
        return flatten_numpy(data, rvec_columns, parent_id_column)
    
    # TODO: Implement C++ helper integration
    # For now, fall back to NumPy
    return flatten_numpy(data, rvec_columns, parent_id_column)


# =============================================================================
# Main API
# =============================================================================

def flatten_to_dataframe(
    data: Dict[str, np.ndarray],
    rvec_columns: List[str],
    parent_id_column: str = "event_id",
    backend: FlattenBackend = FlattenBackend.AUTO
) -> 'pd.DataFrame':
    """
    Flatten RVec columns to pandas DataFrame.
    
    Converts hierarchical RDataFrame data (Events × Tracks × Clusters)
    to flat DataFrame suitable for TTree::Draw-like operations.
    
    Args:
        data: Output from rdf.AsNumpy() containing RVec columns
        rvec_columns: List of RVec column names to flatten
        parent_id_column: Column with parent IDs to replicate (default: 'event_id')
        backend: Implementation backend (default: AUTO)
    
    Returns:
        pandas DataFrame with:
        - parent_id_column: Replicated parent IDs
        - track_idx: 0-based index within parent event
        - cluster_idx: 0-based index within parent track (if 2-level)
        - [rvec_columns]: Flattened data columns
    
    Index Semantics:
        - event_id: Replicated verbatim from input data (not generated)
        - track_idx: Always local to parent event (0 to n_tracks-1)
        - cluster_idx: Always local to parent track (0 to n_clusters-1)
        - Indices are deterministic and stable across all backends
    
    Example:
        >>> data = rdf.AsNumpy(['event_id', 'track_pt', 'track_eta'])
        >>> df = flatten_to_dataframe(data, ['track_pt', 'track_eta'])
        >>> # Result: event_id, track_idx, track_pt, track_eta
    
    Raises:
        ValueError: If rvec_columns have different jagged structures
        ImportError: If Awkward backend requested but not installed
    """
    import pandas as pd
    
    # Validate inputs
    if not rvec_columns:
        raise ValueError("rvec_columns cannot be empty")
    
    if parent_id_column not in data:
        raise ValueError(f"parent_id_column '{parent_id_column}' not in data")
    
    for col in rvec_columns:
        if col not in data:
            raise ValueError(f"Column '{col}' not in data")
    
    # Validate same structure
    validate_same_structure(data, rvec_columns)
    
    # Select backend
    if backend == FlattenBackend.AUTO:
        backend = select_backend(data, rvec_columns)
    
    # Dispatch to backend
    if backend == FlattenBackend.NUMPY:
        result = flatten_numpy(data, rvec_columns, parent_id_column)
    elif backend == FlattenBackend.AWKWARD:
        result = flatten_awkward(data, rvec_columns, parent_id_column)
    elif backend == FlattenBackend.CPP:
        result = flatten_cpp(data, rvec_columns, parent_id_column)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Convert to DataFrame
    return pd.DataFrame(result)


def flatten_to_dict(
    data: Dict[str, np.ndarray],
    rvec_columns: List[str],
    parent_id_column: str = "event_id",
    backend: FlattenBackend = FlattenBackend.AUTO
) -> Dict[str, np.ndarray]:
    """
    Flatten RVec columns to dictionary of numpy arrays.
    
    Same as flatten_to_dataframe but returns dict instead of DataFrame.
    Useful when pandas overhead is not desired.
    
    Args:
        data: Output from rdf.AsNumpy()
        rvec_columns: Columns to flatten
        parent_id_column: Column with parent IDs
        backend: Implementation backend
    
    Returns:
        Dictionary with flattened arrays
    """
    # Validate inputs
    if not rvec_columns:
        raise ValueError("rvec_columns cannot be empty")
    
    if parent_id_column not in data:
        raise ValueError(f"parent_id_column '{parent_id_column}' not in data")
    
    for col in rvec_columns:
        if col not in data:
            raise ValueError(f"Column '{col}' not in data")
    
    # Validate same structure
    validate_same_structure(data, rvec_columns)
    
    # Select backend
    if backend == FlattenBackend.AUTO:
        backend = select_backend(data, rvec_columns)
    
    # Dispatch to backend
    if backend == FlattenBackend.NUMPY:
        return flatten_numpy(data, rvec_columns, parent_id_column)
    elif backend == FlattenBackend.AWKWARD:
        return flatten_awkward(data, rvec_columns, parent_id_column)
    elif backend == FlattenBackend.CPP:
        return flatten_cpp(data, rvec_columns, parent_id_column)
    else:
        raise ValueError(f"Unknown backend: {backend}")
