"""
Numba-accelerated functions for AliasDataFrame.

This module provides JIT-compiled functions for performance-critical operations.
All functions gracefully degrade to NumPy when Numba is not available.

Phase 8a: Value extraction (scatter)
Phase 8b: Index lookup (replace pd.merge)
"""

import numpy as np

# Numba availability detection
try:
    import numba
    from numba import njit, prange
    from numba.typed import Dict as NumbaDict
    from numba import types as numba_types
    NUMBA_AVAILABLE = True
    NUMBA_VERSION = numba.__version__
except ImportError:
    NUMBA_AVAILABLE = False
    NUMBA_VERSION = None
    NumbaDict = None
    numba_types = None


# =============================================================================
# Phase 8a: Value Extraction (Scatter)
# =============================================================================
#
# These functions scatter values from a subframe into a result array using
# precomputed indices. They replace NumPy advanced indexing with explicit loops
# that Numba can parallelize.
#

if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _numba_scatter_f64(sub_values, indices, result):
        """
        Scatter float64 values from subframe into result array.
        
        Parameters
        ----------
        sub_values : np.ndarray[float64]
            Source values from subframe column
        indices : np.ndarray[int64]
            Row indices into sub_values (-1 means missing/skip)
        result : np.ndarray[float64]
            Pre-allocated result array (pre-filled with NaN)
            Modified in-place.
        """
        n = len(indices)
        for i in prange(n):
            idx = indices[i]
            if idx >= 0:
                result[i] = sub_values[idx]
    
    @njit(cache=True, parallel=True)
    def _numba_scatter_f32(sub_values, indices, result):
        """
        Scatter float32 values from subframe into result array.
        
        Parameters
        ----------
        sub_values : np.ndarray[float32]
            Source values from subframe column
        indices : np.ndarray[int64]
            Row indices into sub_values (-1 means missing/skip)
        result : np.ndarray[float32]
            Pre-allocated result array (pre-filled with NaN)
            Modified in-place.
        """
        n = len(indices)
        for i in prange(n):
            idx = indices[i]
            if idx >= 0:
                result[i] = sub_values[idx]
    
    @njit(cache=True, parallel=True)
    def _numba_scatter_i64(sub_values, indices, result, fill_value):
        """
        Scatter int64 values from subframe into result array.
        
        Parameters
        ----------
        sub_values : np.ndarray[int64]
            Source values from subframe column
        indices : np.ndarray[int64]
            Row indices into sub_values (-1 means missing/skip)
        result : np.ndarray[int64]
            Pre-allocated result array
            Modified in-place.
        fill_value : int64
            Value to use for missing indices
        """
        n = len(indices)
        for i in prange(n):
            idx = indices[i]
            if idx >= 0:
                result[i] = sub_values[idx]
            else:
                result[i] = fill_value


# =============================================================================
# Phase 8b: Index Lookup (Replace pd.merge)
# =============================================================================
#
# These functions build an index map from subframe keys and perform lookups
# to find matching row indices. They replace pd.merge() for integer keys.
#

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _numba_build_index_map_direct(subframe_keys, max_key):
        """
        Build direct-addressing index map for integer keys.
        
        Uses O(max_key) memory but provides O(1) lookup.
        Best for dense key ranges (e.g., row indices 0..N).
        
        Parameters
        ----------
        subframe_keys : np.ndarray[int64]
            Unique keys from subframe (e.g., index column values)
        max_key : int64
            Maximum key value (determines map size)
            
        Returns
        -------
        np.ndarray[int64]
            Index map where map[key] = subframe row index, -1 if not present
        """
        # Allocate map with -1 (not found)
        index_map = np.full(max_key + 1, -1, dtype=np.int64)
        
        # Populate map - keep first occurrence for duplicates
        n = len(subframe_keys)
        for i in range(n):
            key = subframe_keys[i]
            if 0 <= key <= max_key and index_map[key] == -1:
                index_map[key] = i
        
        return index_map
    
    @njit(cache=True, parallel=True)
    def _numba_lookup_indices_direct(main_keys, index_map):
        """
        Lookup indices using direct-addressing map.
        
        Parameters
        ----------
        main_keys : np.ndarray[int64]
            Keys from main DataFrame to look up
        index_map : np.ndarray[int64]
            Index map from _numba_build_index_map_direct
            
        Returns
        -------
        indices : np.ndarray[int64]
            Subframe row indices (-1 for missing keys)
        missing_mask : np.ndarray[bool]
            True where key was not found
        """
        n = len(main_keys)
        map_size = len(index_map)
        
        indices = np.empty(n, dtype=np.int64)
        missing_mask = np.empty(n, dtype=np.bool_)
        
        for i in prange(n):
            key = main_keys[i]
            if 0 <= key < map_size:
                idx = index_map[key]
                indices[i] = idx
                missing_mask[i] = (idx == -1)
            else:
                indices[i] = -1
                missing_mask[i] = True
        
        return indices, missing_mask
    
    # Note: This function is NOT JIT-compiled because Dict.empty() cannot be
    # called inside nopython mode. The dict creation is O(n) and fast enough
    # in Python - it's the lookup that benefits from JIT compilation.
    def _numba_build_index_map_hash(subframe_keys):
        """
        Build hash-based index map for integer keys.
        
        Uses O(n) memory regardless of key range.
        Slower than direct addressing but handles sparse/large key ranges.
        
        Note: This function is NOT JIT-compiled because Numba typed dict
        creation must happen in Python. The lookup function IS JIT-compiled.
        
        Parameters
        ----------
        subframe_keys : np.ndarray[int64]
            Keys from subframe
            
        Returns
        -------
        dict
            Numba typed dict: key -> subframe row index
        """
        index_map = NumbaDict.empty(
            key_type=numba_types.int64,
            value_type=numba_types.int64
        )
        
        n = len(subframe_keys)
        for i in range(n):
            key = subframe_keys[i]
            if key not in index_map:
                index_map[key] = i
        
        return index_map
    
    @njit(cache=True, parallel=True)
    def _numba_lookup_indices_hash(main_keys, index_map):
        """
        Lookup indices using hash-based map.
        
        Parameters
        ----------
        main_keys : np.ndarray[int64]
            Keys from main DataFrame to look up
        index_map : numba.typed.Dict
            Index map from _numba_build_index_map_hash
            
        Returns
        -------
        indices : np.ndarray[int64]
            Subframe row indices (-1 for missing keys)
        missing_mask : np.ndarray[bool]
            True where key was not found
        """
        n = len(main_keys)
        indices = np.empty(n, dtype=np.int64)
        missing_mask = np.empty(n, dtype=np.bool_)
        
        for i in prange(n):
            key = main_keys[i]
            if key in index_map:
                indices[i] = index_map[key]
                missing_mask[i] = False
            else:
                indices[i] = -1
                missing_mask[i] = True
        
        return indices, missing_mask


# =============================================================================
# Public API: Dispatch Functions
# =============================================================================
#
# These functions provide a clean API and handle dtype dispatch.
#

def numba_scatter(sub_values, indices, result):
    """
    Scatter values from subframe into result array using Numba.
    
    Dispatches to appropriate dtype-specific function.
    Falls back to NumPy if Numba unavailable or unsupported dtype.
    
    Parameters
    ----------
    sub_values : np.ndarray
        Source values from subframe column
    indices : np.ndarray[int64]
        Row indices into sub_values (-1 means missing/skip)
    result : np.ndarray
        Pre-allocated result array (modified in-place)
        
    Returns
    -------
    bool
        True if Numba was used, False if fell back to NumPy
    """
    if not NUMBA_AVAILABLE:
        # NumPy fallback
        valid = indices >= 0
        result[valid] = sub_values[indices[valid]]
        return False
    
    dtype = sub_values.dtype
    
    if dtype == np.float64:
        _numba_scatter_f64(sub_values, indices, result)
        return True
    elif dtype == np.float32:
        _numba_scatter_f32(sub_values, indices, result)
        return True
    else:
        # Unsupported dtype - use NumPy
        valid = indices >= 0
        result[valid] = sub_values[indices[valid]]
        return False


def numba_compute_join_indices(main_keys, subframe_keys, use_hash=None):
    """
    Compute join indices using Numba (replaces pd.merge).
    
    Parameters
    ----------
    main_keys : np.ndarray[int64]
        Keys from main DataFrame
    subframe_keys : np.ndarray[int64]
        Keys from subframe (should be unique or first match is used)
    use_hash : bool, optional
        If True, use hash-based lookup (O(n) memory).
        If False, use direct addressing (O(max_key) memory).
        If None, auto-select based on key range.
        
    Returns
    -------
    indices : np.ndarray[int64]
        Subframe row indices (-1 for missing keys)
    missing_mask : np.ndarray[bool]
        True where key was not found
    used_numba : bool
        True if Numba was used, False if fell back to NumPy/Pandas
    """
    if not NUMBA_AVAILABLE:
        return None, None, False
    
    # Ensure int64
    if main_keys.dtype != np.int64:
        main_keys = main_keys.astype(np.int64)
    if subframe_keys.dtype != np.int64:
        subframe_keys = subframe_keys.astype(np.int64)
    
    # Auto-select method based on key range
    if use_hash is None:
        max_key = max(main_keys.max(), subframe_keys.max()) if len(subframe_keys) > 0 else 0
        min_key = min(main_keys.min(), subframe_keys.min()) if len(subframe_keys) > 0 else 0
        
        # Use direct addressing if range is reasonable
        # Rule: direct if max_key < 10 * n_subframe_rows and min_key >= 0
        use_hash = (min_key < 0 or max_key > 10 * len(subframe_keys))
    
    if use_hash:
        index_map = _numba_build_index_map_hash(subframe_keys)
        indices, missing_mask = _numba_lookup_indices_hash(main_keys, index_map)
    else:
        max_key = int(max(main_keys.max(), subframe_keys.max()))
        index_map = _numba_build_index_map_direct(subframe_keys, max_key)
        indices, missing_mask = _numba_lookup_indices_direct(main_keys, index_map)
    
    return indices, missing_mask, True


# =============================================================================
# Phase 8c: Multi-Column Key Linearization
# =============================================================================
#
# These functions pack multi-column integer keys into single int64 values,
# enabling use of Phase 8b lookup for composite keys.
#
# Key insight: (col1, col2, col3) can be linearized as:
#   linear_key = col1 * stride1 + col2 * stride2 + col3
# where strides are computed from GLOBAL max values across both DataFrames.
#

if NUMBA_AVAILABLE:
    @njit(cache=True, parallel=True)
    def _numba_linearize_keys(keys_2d, strides):
        """
        Pack multi-column keys into single int64 values.
        
        Parameters
        ----------
        keys_2d : np.ndarray[int64] of shape (n_rows, n_cols)
            Key columns stacked horizontally
        strides : np.ndarray[int64] of shape (n_cols,)
            Stride multipliers for each column (rightmost = 1)
            
        Returns
        -------
        np.ndarray[int64] of shape (n_rows,)
            Linearized keys
        """
        n_rows = keys_2d.shape[0]
        n_cols = keys_2d.shape[1]
        result = np.zeros(n_rows, dtype=np.int64)
        
        for i in prange(n_rows):
            val = 0
            for j in range(n_cols):
                val += keys_2d[i, j] * strides[j]
            result[i] = val
        
        return result


def linearize_multi_column_keys_pair(main_df, sub_df, key_cols):
    """
    Linearize keys from BOTH DataFrames using GLOBAL strides.
    
    Critical: Both DataFrames must use the same strides computed from
    global max values, otherwise the same key tuple would map to different
    linear values and the join would silently fail.
    
    Parameters
    ----------
    main_df : pd.DataFrame
        Main DataFrame
    sub_df : pd.DataFrame  
        Subframe DataFrame
    key_cols : list of str
        Column names to use as keys
        
    Returns
    -------
    linear_main : np.ndarray[int64] or None
        Linearized keys for main DataFrame
    linear_sub : np.ndarray[int64] or None
        Linearized keys for subframe
    success : bool
        False if linearization not possible (overflow, negative, non-integer)
    """
    if not NUMBA_AVAILABLE:
        return None, None, False
    
    # Stack key columns into 2D arrays
    try:
        keys_main = np.column_stack([main_df[c].to_numpy() for c in key_cols])
        keys_sub = np.column_stack([sub_df[c].to_numpy() for c in key_cols])
    except (KeyError, ValueError):
        return None, None, False
    
    # Handle empty subframe
    if len(keys_sub) == 0:
        return None, None, False
    
    # Check for integer dtype
    if not (np.issubdtype(keys_main.dtype, np.integer) and 
            np.issubdtype(keys_sub.dtype, np.integer)):
        return None, None, False
    
    # Check for negative keys (fallback to pandas)
    if np.any(keys_main < 0) or np.any(keys_sub < 0):
        return None, None, False
    
    # Compute GLOBAL maxes from BOTH DataFrames
    # This is critical for correctness!
    max_main = keys_main.max(axis=0) if len(keys_main) > 0 else np.zeros(len(key_cols))
    max_sub = keys_sub.max(axis=0)
    global_maxes = np.maximum(max_main, max_sub)
    
    # Check for overflow using Python ints (avoid NumPy wraparound)
    product = 1
    for m in global_maxes:
        product *= (int(m) + 1)
        if product > 2**62:
            return None, None, False
    
    # Compute strides (rightmost = 1, C-order / row-major)
    n_cols = len(key_cols)
    strides = np.ones(n_cols, dtype=np.int64)
    for i in range(n_cols - 2, -1, -1):
        strides[i] = strides[i + 1] * (int(global_maxes[i + 1]) + 1)
    
    # Linearize both using SAME strides
    linear_main = _numba_linearize_keys(keys_main.astype(np.int64), strides)
    linear_sub = _numba_linearize_keys(keys_sub.astype(np.int64), strides)
    
    return linear_main, linear_sub, True


# =============================================================================
# Utility Functions
# =============================================================================

def get_numba_info():
    """
    Get information about Numba availability and configuration.
    
    Returns
    -------
    dict
        Information about Numba status
    """
    return {
        'available': NUMBA_AVAILABLE,
        'version': NUMBA_VERSION,
        'parallel_enabled': NUMBA_AVAILABLE,  # prange is used
    }


# Minimum row count to use Numba (JIT overhead not worth it for small arrays)
NUMBA_MIN_ROWS = 10000
