"""
Composite key generation utilities for multi-column index linearization.

This module provides stateless, DataFrame-based functions for generating
composite keys used in RDataFrame friend tree joins with >2 index columns.

Key concepts:
- Dense linearization: key = k0 + k1*max0 + k2*max0*max1 + ...
  Efficient when key ranges are compact (small gaps between values)
- Sparse mapping: Uses np.unique to assign sequential integers
  Required when dense would overflow int64 or be >10x wasteful

Used by:
- AliasDataFrameRDF.py (RDataFrame friend joins)
- AliasDataFrame.py (future: Python-based materialization)

Examples
--------
>>> from _composite_keys import compute_composite_key_auto
>>> main_keys, sub_keys, method = compute_composite_key_auto(main_df, sub_df, ['k1', 'k2', 'k3'])
>>> print(f"Used {method} method")

>>> from _composite_keys import get_composite_key_column_name
>>> col_name = get_composite_key_column_name('DTrack0')
>>> print(col_name)  # '__adf_key_DTrack0__'
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union

__all__ = [
    'get_composite_key_column_name',
    'check_dense_overflow',
    'should_use_sparse',
    'generate_dense_cpp_expression',
    'compute_composite_key_dense',
    'compute_composite_key_sparse',
    'compute_composite_key_auto',
]

# Constants
MAX_INT64 = np.iinfo(np.int64).max
MAX_INT32 = 2**31


def get_composite_key_column_name(subframe_name: str) -> str:
    """
    Get the standard column name for a composite key.
    
    Parameters
    ----------
    subframe_name : str
        Name of the subframe
        
    Returns
    -------
    str
        Column name like '__adf_key_DTrack0__'
        
    Examples
    --------
    >>> get_composite_key_column_name('DTrack0')
    '__adf_key_DTrack0__'
    >>> get_composite_key_column_name('S')
    '__adf_key_S__'
    """
    return f"__adf_key_{subframe_name}__"


def check_dense_overflow(max_values: List[int]) -> Tuple[bool, Union[int, float]]:
    """
    Check if dense linearization would overflow int64.
    
    Dense linearization computes: key = k0 + k1*max0 + k2*max0*max1 + ...
    This requires that the total key space (product of max values) fits in int64.
    
    Parameters
    ----------
    max_values : list of int
        Maximum values for each key column (typically max + 1 for range)
        
    Returns
    -------
    tuple
        (is_safe, compact_range) where:
        - is_safe: True if dense linearization won't overflow int64
        - compact_range: Total key space size (or inf if overflow)
        
    Examples
    --------
    >>> check_dense_overflow([10, 20, 30])
    (True, 6000)
    >>> check_dense_overflow([1000000, 1000000, 1000000])
    (True, 1000000000000000000)
    >>> check_dense_overflow([10**10, 10**10])
    (False, inf)
    """
    # Calculate product carefully to detect overflow
    compact_range = 1
    for mv in max_values:
        # Check if multiplication would overflow int64
        if compact_range > 0 and mv > (2**63 - 1) // compact_range:
            return False, float('inf')
        compact_range *= mv
    
    return compact_range <= 2**63 - 1, compact_range


def should_use_sparse(df: pd.DataFrame, key_columns: List[str]) -> bool:
    """
    Determine if sparse key mapping should be used instead of dense linearization.
    
    Use sparse mapping when:
    1. Compact range exceeds int32 (2^31), OR
    2. Compact range is >10x wasteful compared to actual unique combinations
    
    The int32 threshold is conservative to ensure good BuildIndex performance.
    The 10x wasteful threshold catches cases like sparse IDs (e.g., timestamps).
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with key columns
    key_columns : list of str
        Column names forming the composite key
        
    Returns
    -------
    bool
        True if sparse mapping should be used
        
    Examples
    --------
    >>> df = pd.DataFrame({'a': [0, 1, 2], 'b': [0, 1, 2]})
    >>> should_use_sparse(df, ['a', 'b'])
    False
    >>> df_sparse = pd.DataFrame({'a': [0, 1000000], 'b': [0, 1000000]})
    >>> should_use_sparse(df_sparse, ['a', 'b'])
    True
    """
    max_vals = [int(df[k].max()) + 1 for k in key_columns]
    compact_range = np.prod(max_vals, dtype=np.int64)
    n_unique = np.prod([df[k].nunique() for k in key_columns])
    
    return compact_range > MAX_INT32 or compact_range > 10 * n_unique


def generate_dense_cpp_expression(key_columns: List[str], max_values: List[int]) -> str:
    """
    Generate C++ expression for dense composite key computation.
    
    Used for runtime generation via rdf.Define() when composite key
    needs to be computed on-the-fly in RDataFrame.
    
    The expression computes: k0 + k1*max0 + k2*max0*max1 + ...
    
    Parameters
    ----------
    key_columns : list of str
        Column names forming the composite key
    max_values : list of int
        Maximum values for each key column (max + 1 for range)
        
    Returns
    -------
    str
        C++ expression like "k0 + k1 * 10 + k2 * 10 * 5"
        
    Examples
    --------
    >>> generate_dense_cpp_expression(['side', 'row'], [2, 152])
    'side + row * 2'
    >>> generate_dense_cpp_expression(['a', 'b', 'c'], [10, 20, 30])
    'a + b * 10 + c * 10 * 20'
    >>> generate_dense_cpp_expression(['x'], [100])
    'x'
    """
    if len(key_columns) == 1:
        return key_columns[0]
    
    # First term: just the first column
    parts = [key_columns[0]]
    
    # Subsequent terms: column * product of previous max values
    multiplier_parts = []
    for i in range(1, len(key_columns)):
        multiplier_parts.append(str(max_values[i-1]))
        multiplier = " * ".join(multiplier_parts)
        parts.append(f"{key_columns[i]} * {multiplier}")
    
    return " + ".join(parts)


def compute_composite_key_dense(
    df: pd.DataFrame, 
    key_columns: List[str], 
    max_values: Optional[List[int]] = None
) -> np.ndarray:
    """
    Compute composite key using dense (compact) linearization.
    
    Computes: __adf_key__ = k0 + k1*max0 + k2*max0*max1 + ...
    
    This is efficient when key ranges are compact (few gaps between values).
    Use should_use_sparse() to check if this method is appropriate.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with key columns
    key_columns : list of str
        Column names forming the composite key
    max_values : list of int, optional
        Maximum values for each key column. If None, computed from data.
        When computing keys for multiple DataFrames that will be joined,
        pass the same max_values to ensure consistent keys.
        
    Returns
    -------
    np.ndarray
        Int64 composite keys, one per row
        
    Examples
    --------
    >>> df = pd.DataFrame({'a': [0, 1, 2], 'b': [0, 1, 0]})
    >>> compute_composite_key_dense(df, ['a', 'b'])
    array([0, 4, 2])  # With max_values=[3, 2]: 0+0*3, 1+1*3, 2+0*3
    """
    if max_values is None:
        max_values = [int(df[k].max()) + 1 for k in key_columns]
    
    key = df[key_columns[0]].values.astype(np.int64)
    multiplier = max_values[0]
    
    for i, col in enumerate(key_columns[1:], 1):
        key = key + df[col].values.astype(np.int64) * multiplier
        multiplier *= max_values[i]
    
    return key


def compute_composite_key_sparse(
    main_df: pd.DataFrame, 
    sub_df: pd.DataFrame, 
    key_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute composite key using vectorized sparse mapping.
    
    Works for any key distribution (dense or sparse).
    Uses np.unique(axis=0) for efficient vectorized computation.
    
    This method assigns sequential integers to unique key combinations,
    avoiding the wasteful key space of dense linearization for sparse data.
    
    Parameters
    ----------
    main_df : DataFrame
        Main DataFrame with key columns
    sub_df : DataFrame
        Subframe DataFrame with key columns
    key_columns : list of str
        Column names forming the composite key
        
    Returns
    -------
    main_keys : np.ndarray
        Int64 composite keys for main DataFrame
    sub_keys : np.ndarray
        Int64 composite keys for subframe DataFrame
        
    Notes
    -----
    Both DataFrames use the same mapping, ensuring keys match for joins.
    Complexity: O(n log n) via np.unique, fully vectorized.
    
    Examples
    --------
    >>> main = pd.DataFrame({'a': [100, 200], 'b': [1000, 2000]})
    >>> sub = pd.DataFrame({'a': [200, 100], 'b': [2000, 1000]})
    >>> main_keys, sub_keys = compute_composite_key_sparse(main, sub, ['a', 'b'])
    >>> # Keys are sequential integers based on sorted unique combinations
    """
    # Combine main and sub to build shared mapping
    main_vals = main_df[key_columns].to_numpy()
    sub_vals = sub_df[key_columns].to_numpy()
    all_vals = np.vstack([main_vals, sub_vals])
    
    # Get unique rows and inverse mapping
    _, inverse = np.unique(all_vals, axis=0, return_inverse=True)
    
    # Split back into main and sub
    n_main = len(main_df)
    main_keys = inverse[:n_main].astype(np.int64)
    sub_keys = inverse[n_main:].astype(np.int64)
    
    return main_keys, sub_keys


def compute_composite_key_auto(
    main_df: pd.DataFrame, 
    sub_df: pd.DataFrame, 
    key_columns: List[str],
    method: str = 'auto'
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Compute composite keys with automatic method selection.
    
    Uses dense linearization when key ranges are compact,
    sparse mapping when ranges are too large or wasteful.
    
    Parameters
    ----------
    main_df : DataFrame
        Main DataFrame with key columns
    sub_df : DataFrame
        Subframe DataFrame with key columns
    key_columns : list of str
        Column names forming the composite key
    method : str, default 'auto'
        Method selection:
        - 'auto': Automatically choose based on data characteristics
        - 'dense': Force dense linearization (may overflow)
        - 'sparse': Force sparse mapping
        
    Returns
    -------
    main_keys : np.ndarray
        Int64 composite keys for main DataFrame
    sub_keys : np.ndarray
        Int64 composite keys for subframe DataFrame
    method_used : str
        'dense' or 'sparse' indicating which method was used
        
    Examples
    --------
    >>> main = pd.DataFrame({'a': [0, 1, 2], 'b': [0, 1, 2]})
    >>> sub = pd.DataFrame({'a': [0, 1], 'b': [0, 1]})
    >>> main_keys, sub_keys, method = compute_composite_key_auto(main, sub, ['a', 'b'])
    >>> print(f"Used {method} method")
    Used dense method
    """
    if method == 'sparse':
        main_keys, sub_keys = compute_composite_key_sparse(main_df, sub_df, key_columns)
        return main_keys, sub_keys, 'sparse'
    
    if method == 'dense':
        combined = pd.concat([main_df[key_columns], sub_df[key_columns]], ignore_index=True)
        max_values = [int(combined[k].max()) + 1 for k in key_columns]
        main_keys = compute_composite_key_dense(main_df, key_columns, max_values)
        sub_keys = compute_composite_key_dense(sub_df, key_columns, max_values)
        return main_keys, sub_keys, 'dense'
    
    # Auto: Check if sparse is needed using combined data
    combined = pd.concat([main_df[key_columns], sub_df[key_columns]], ignore_index=True)
    
    if should_use_sparse(combined, key_columns):
        main_keys, sub_keys = compute_composite_key_sparse(main_df, sub_df, key_columns)
        return main_keys, sub_keys, 'sparse'
    else:
        # Compute shared max values from union
        max_values = [int(combined[k].max()) + 1 for k in key_columns]
        main_keys = compute_composite_key_dense(main_df, key_columns, max_values)
        sub_keys = compute_composite_key_dense(sub_df, key_columns, max_values)
        return main_keys, sub_keys, 'dense'
