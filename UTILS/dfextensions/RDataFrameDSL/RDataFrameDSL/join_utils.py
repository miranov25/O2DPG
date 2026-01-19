"""
RDataFrameDSL Join Utilities

Phase 13.6.C: Join strategy implementation for mixed-depth and different-slice operations.

This module provides the core join logic for combining DataFrames with different
nesting depths or different slice ranges, following DSL_SPEC_ND_Slicing.md §5.

Key concepts:
- Join detection: Determine when join is needed vs element-wise operation
- Join keys: Determine common index columns based on depth hierarchy
- Join execution: Use pandas.merge() with specified strategy
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np


# =============================================================================
# Join Types (DSL_SPEC §5.1)
# =============================================================================

class JoinType(Enum):
    """
    Join strategy types per DSL_SPEC_ND_Slicing.md §5.1.
    
    - INNER: Intersection of indices (default, no NaN)
    - OUTER: Union of indices (NaN for missing)
    - LEFT: All from deeper/first operand
    - RIGHT: All from shallower/second operand
    """
    INNER = "inner"
    OUTER = "outer"
    LEFT = "left"
    RIGHT = "right"


# =============================================================================
# Index Column Hierarchy (DSL_SPEC §2.1)
# =============================================================================

# Standard index columns by depth
INDEX_COLUMNS_BY_DEPTH = {
    0: ['event_id'],
    1: ['event_id', 'track_idx'],
    2: ['event_id', 'track_idx', 'cluster_idx'],
    3: ['event_id', 'track_idx', 'cluster_idx', 'hit_idx'],
}

def get_index_columns(depth: int) -> List[str]:
    """
    Get index column names for given depth.
    
    Args:
        depth: Nesting depth (0=event, 1=track, 2=cluster, 3=hit)
        
    Returns:
        List of index column names
    """
    if depth in INDEX_COLUMNS_BY_DEPTH:
        return INDEX_COLUMNS_BY_DEPTH[depth]
    
    # Extend for depth > 3
    cols = list(INDEX_COLUMNS_BY_DEPTH[3])
    for d in range(4, depth + 1):
        cols.append(f'idx_{d}')
    return cols


# =============================================================================
# Join Plan (analysis result)
# =============================================================================

@dataclass
class JoinPlan:
    """
    Plan for joining multiple DataFrames.
    
    Attributes:
        needs_join: Whether a join is required (vs element-wise)
        join_keys: Common index columns to join on
        target_depth: Output depth (max of all operands)
        operand_depths: List of depths for each operand
        join_type: Join strategy to use
    """
    needs_join: bool
    join_keys: List[str]
    target_depth: int
    operand_depths: List[int]
    join_type: JoinType = JoinType.INNER
    
    def __repr__(self) -> str:
        if not self.needs_join:
            return "JoinPlan(element-wise, no join needed)"
        return (f"JoinPlan(join_type={self.join_type.value}, "
                f"keys={self.join_keys}, target_depth={self.target_depth})")


# =============================================================================
# Join Detection (DSL_SPEC §5.2, §5.3)
# =============================================================================

def analyze_join_requirements(
    operand_depths: List[int],
    operand_slices: Optional[List[Any]] = None,
) -> JoinPlan:
    """
    Analyze whether join is needed and determine join keys.
    
    Per DSL_SPEC_ND_Slicing.md §5:
    - Same depth, same slice → element-wise (no join)
    - Different depths → join on common indices (broadcast shallower)
    - Same depth, different slices → join on indices
    
    Args:
        operand_depths: List of nesting depths for each operand
        operand_slices: Optional list of slice specs (for same-depth comparison)
        
    Returns:
        JoinPlan with analysis results
    """
    if not operand_depths:
        return JoinPlan(
            needs_join=False,
            join_keys=['event_id'],
            target_depth=0,
            operand_depths=[],
        )
    
    target_depth = max(operand_depths)
    min_depth = min(operand_depths)
    
    # Different depths → join needed (broadcast shallower to deeper)
    if max(operand_depths) != min(operand_depths):
        # Join keys = index columns at minimum depth (common to all)
        join_keys = get_index_columns(min_depth)
        return JoinPlan(
            needs_join=True,
            join_keys=join_keys,
            target_depth=target_depth,
            operand_depths=operand_depths,
        )
    
    # Same depth - check if slices differ
    if operand_slices is not None:
        unique_slices = set(str(s) for s in operand_slices)
        if len(unique_slices) > 1:
            # Different slices at same depth → join
            join_keys = get_index_columns(target_depth)
            return JoinPlan(
                needs_join=True,
                join_keys=join_keys,
                target_depth=target_depth,
                operand_depths=operand_depths,
            )
    
    # Same depth, same slice (or no slice info) → element-wise
    return JoinPlan(
        needs_join=False,
        join_keys=get_index_columns(target_depth),
        target_depth=target_depth,
        operand_depths=operand_depths,
    )


def needs_join(operand_depths: List[int]) -> bool:
    """
    Quick check if join is needed based on depths only.
    
    Args:
        operand_depths: List of nesting depths
        
    Returns:
        True if join is needed (different depths)
    """
    if len(operand_depths) < 2:
        return False
    return max(operand_depths) != min(operand_depths)


# =============================================================================
# Join Execution (DSL_SPEC §5)
# =============================================================================

def execute_join(
    dataframes: List[pd.DataFrame],
    plan: JoinPlan,
    value_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Execute join according to plan.
    
    Args:
        dataframes: List of DataFrames to join
        plan: JoinPlan from analyze_join_requirements()
        value_columns: Optional list of value column names (one per DataFrame)
        
    Returns:
        Joined DataFrame at target depth
    """
    if not dataframes:
        raise ValueError("No DataFrames to join")
    
    if len(dataframes) == 1:
        return dataframes[0].copy()
    
    if not plan.needs_join:
        # Element-wise: assume same structure, just concat columns
        return _merge_element_wise(dataframes, value_columns)
    
    # Perform join
    return _merge_with_join(dataframes, plan, value_columns)


def _merge_element_wise(
    dataframes: List[pd.DataFrame],
    value_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Merge DataFrames that have identical structure (no join needed).
    
    Assumes all DataFrames have the same index columns and row count.
    """
    if len(dataframes) == 1:
        return dataframes[0].copy()
    
    # Start with first DataFrame
    result = dataframes[0].copy()
    
    # Identify index columns (present in all DataFrames)
    index_cols = [c for c in result.columns 
                  if c in ['event_id', 'track_idx', 'cluster_idx', 'hit_idx']
                  or c.startswith('idx_')]
    
    # Add value columns from other DataFrames
    for i, df in enumerate(dataframes[1:], start=1):
        for col in df.columns:
            if col not in index_cols and col not in result.columns:
                if value_columns and i < len(value_columns):
                    # Use specified name
                    result[value_columns[i]] = df[col].values
                else:
                    result[col] = df[col].values
    
    return result


def _merge_with_join(
    dataframes: List[pd.DataFrame],
    plan: JoinPlan,
    value_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Merge DataFrames using pandas join.
    
    Per DSL_SPEC §5:
    - Inner join: intersection of indices
    - Outer join: union of indices (NaN for missing)
    - Left/Right: preserve one side
    """
    if len(dataframes) < 2:
        return dataframes[0].copy() if dataframes else pd.DataFrame()
    
    # Map JoinType to pandas how parameter
    how_map = {
        JoinType.INNER: 'inner',
        JoinType.OUTER: 'outer',
        JoinType.LEFT: 'left',
        JoinType.RIGHT: 'right',
    }
    how = how_map[plan.join_type]
    
    # Start with first DataFrame
    result = dataframes[0].copy()
    
    # Identify which columns are join keys
    join_keys = plan.join_keys
    
    # Ensure join keys exist in result
    available_keys = [k for k in join_keys if k in result.columns]
    if not available_keys:
        available_keys = ['event_id']  # Fallback
    
    # Merge remaining DataFrames
    for i, df in enumerate(dataframes[1:], start=1):
        # Find available join keys in this DataFrame
        df_keys = [k for k in available_keys if k in df.columns]
        if not df_keys:
            df_keys = ['event_id']  # Fallback
        
        # Get value columns (non-index columns)
        df_value_cols = [c for c in df.columns if c not in join_keys]
        
        # Rename value columns if needed to avoid conflicts
        rename_map = {}
        for col in df_value_cols:
            if col in result.columns:
                rename_map[col] = f"{col}_{i}"
        
        df_to_merge = df.rename(columns=rename_map) if rename_map else df
        
        # Perform merge
        result = pd.merge(
            result,
            df_to_merge,
            on=df_keys,
            how=how,
            suffixes=('', f'_{i}'),
        )
    
    # Sort by index columns (per DSL_SPEC §7.4)
    sort_cols = [c for c in join_keys if c in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)
    
    return result


# =============================================================================
# Broadcast (DSL_SPEC §5 Rule 1)
# =============================================================================

def broadcast_to_depth(
    df: pd.DataFrame,
    source_depth: int,
    target_depth: int,
    target_structure: pd.DataFrame,
) -> pd.DataFrame:
    """
    Broadcast shallower DataFrame to deeper level.
    
    Per DSL_SPEC §5 Rule 1: Scalars broadcast to all levels.
    Per DSL_SPEC §7.3: Shallower values replicate to match deeper level.
    
    Args:
        df: Source DataFrame at shallower level
        source_depth: Depth of source (e.g., 0 for event, 1 for track)
        target_depth: Target depth to broadcast to
        target_structure: DataFrame with target index structure
        
    Returns:
        DataFrame with source values replicated to target level
    """
    if source_depth >= target_depth:
        return df.copy()
    
    # Get join keys at source depth (these are shared with target)
    source_keys = get_index_columns(source_depth)
    available_source_keys = [k for k in source_keys if k in df.columns]
    
    # Get target index columns
    target_keys = get_index_columns(target_depth)
    available_target_keys = [k for k in target_keys if k in target_structure.columns]
    
    if not available_source_keys:
        # No common keys - can't broadcast properly
        # This shouldn't happen if data is well-formed
        return df.copy()
    
    # Get the target structure with just index columns
    target_indices = target_structure[available_target_keys].drop_duplicates()
    
    # Merge: target structure LEFT JOIN source data on source keys
    # This replicates source values for each target row with matching source keys
    result = pd.merge(
        target_indices,
        df,
        on=available_source_keys,
        how='left'
    )
    
    return result


# =============================================================================
# High-Level Join Function
# =============================================================================

def join_dataframes(
    dataframes: List[pd.DataFrame],
    depths: List[int],
    join_type: Union[str, JoinType] = 'inner',
    value_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Join multiple DataFrames with different depths.
    
    This is the main entry point for join operations.
    
    Args:
        dataframes: List of DataFrames to join
        depths: List of nesting depths (one per DataFrame)
        join_type: Join strategy ('inner', 'outer', 'left', 'right')
        value_columns: Optional names for value columns in result
        
    Returns:
        Joined DataFrame at deepest level
        
    Example:
        >>> # cluster_Q (depth 2) / track_pt (depth 1)
        >>> result = join_dataframes(
        ...     [cluster_df, track_df],
        ...     depths=[2, 1],
        ...     join_type='inner'
        ... )
    """
    if isinstance(join_type, str):
        join_type = JoinType(join_type)
    
    if not dataframes:
        raise ValueError("No DataFrames to join")
    
    if len(dataframes) == 1:
        return dataframes[0].copy()
    
    # Analyze join requirements
    plan = analyze_join_requirements(depths)
    plan.join_type = join_type
    target_depth = plan.target_depth
    
    # Map JoinType to pandas how parameter
    how_map = {
        JoinType.INNER: 'inner',
        JoinType.OUTER: 'outer',
        JoinType.LEFT: 'left',
        JoinType.RIGHT: 'right',
    }
    how = how_map[join_type]
    
    # Index columns set for filtering
    index_col_set = {'event_id', 'track_idx', 'cluster_idx', 'hit_idx'}
    for i in range(4, 10):
        index_col_set.add(f'idx_{i}')
    
    # Find the first DataFrame at target depth to use as base
    target_idx = None
    for i, d in enumerate(depths):
        if d == target_depth:
            target_idx = i
            break
    if target_idx is None:
        target_idx = 0
    
    # Start with the deepest DataFrame as base
    result = dataframes[target_idx].copy()
    
    # Merge other DataFrames into result
    for i, (df, depth) in enumerate(zip(dataframes, depths)):
        if i == target_idx:
            continue  # Already in result
        
        # Determine join keys
        if depth < target_depth:
            # Shallower: join on keys at shallower level (broadcast)
            join_keys = get_index_columns(depth)
        else:
            # Same or deeper depth: join on all index keys present in both
            join_keys = get_index_columns(min(depth, target_depth))
        
        # Filter to available keys in both DataFrames
        available_keys = [k for k in join_keys 
                         if k in result.columns and k in df.columns]
        
        if not available_keys:
            available_keys = ['event_id']
        
        # Get value columns from df (non-index columns)
        df_value_cols = [c for c in df.columns if c not in index_col_set]
        
        # Handle column name conflicts
        rename_map = {}
        for col in df_value_cols:
            if col in result.columns:
                rename_map[col] = f"{col}_{i}"
        
        df_to_merge = df.rename(columns=rename_map) if rename_map else df
        
        # For shallower depth, we need to deduplicate on join keys
        # For same depth, we merge as-is (the join handles matching)
        if depth < target_depth:
            # Select columns for merge and deduplicate
            cols_to_merge = list(set(available_keys + 
                                    [c for c in df_to_merge.columns if c not in index_col_set]))
            df_to_merge = df_to_merge[cols_to_merge].drop_duplicates(subset=available_keys)
        
        # Perform merge
        result = pd.merge(
            result,
            df_to_merge,
            on=available_keys,
            how=how,
        )
    
    # Sort by index columns (per DSL_SPEC §7.4)
    sort_cols = [c for c in get_index_columns(target_depth) if c in result.columns]
    if sort_cols:
        result = result.sort_values(sort_cols).reset_index(drop=True)
    
    return result


# =============================================================================
# Utility Functions
# =============================================================================

def get_depth_from_schema(schema: Dict[str, str], column: str) -> int:
    """
    Determine column depth from schema type string.
    
    Args:
        schema: Column name → type string mapping
        column: Column name to check
        
    Returns:
        Depth (0=scalar, 1=RVec, 2=RVec<RVec>, etc.)
    """
    if column not in schema:
        return 0
    
    type_str = schema[column]
    depth = 0
    while 'RVec<' in type_str:
        depth += 1
        type_str = type_str.replace('RVec<', '', 1)
        if type_str.endswith('>'):
            type_str = type_str[:-1]
    
    return depth


def get_depth_from_dataframe(df: pd.DataFrame) -> int:
    """
    Infer depth from DataFrame index columns.
    
    Args:
        df: DataFrame with index columns
        
    Returns:
        Depth based on present index columns
    """
    if 'hit_idx' in df.columns:
        return 3
    if 'cluster_idx' in df.columns:
        return 2
    if 'track_idx' in df.columns:
        return 1
    return 0


def validate_join_inputs(
    dataframes: List[pd.DataFrame],
    depths: List[int],
) -> None:
    """
    Validate inputs for join operation.
    
    Raises:
        ValueError: If inputs are invalid
    """
    if len(dataframes) != len(depths):
        raise ValueError(
            f"Number of DataFrames ({len(dataframes)}) must match "
            f"number of depths ({len(depths)})"
        )
    
    if not dataframes:
        raise ValueError("At least one DataFrame required")
    
    # Check that all DataFrames have event_id (required per spec)
    for i, df in enumerate(dataframes):
        if 'event_id' not in df.columns:
            raise ValueError(
                f"DataFrame {i} missing required 'event_id' column"
            )
