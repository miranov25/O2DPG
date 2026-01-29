"""
Statistics computation for dfdraw.

Phase 13.6.G.DF: Statistics Enhancements
- Issue 1: format_stats_box() auto-detects plot type for default fields
- Issue 2: Range-aware stats computation
- Issue 3: Robust statistics (median, quartiles, MAD)
- Breaking change: std now uses ddof=0 (population) to match ROOT
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


def compute_stats(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    group_by: Optional[str] = None,
    range_x: Optional[Tuple[float, float]] = None,
    range_y: Optional[Tuple[float, float]] = None,
    robust: bool = False,
) -> pd.DataFrame:
    """
    Compute statistics for plotting.
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    y_col : str
        Primary column (or only column for 1D).
    x_col : str, optional
        Secondary column for 2D stats.
    group_by : str, optional
        Compute stats per group.
    range_x : tuple, optional
        (min, max) inclusive range for x values.
        For 1D plots, this filters the primary variable.
        For 2D plots, this filters the x-axis variable.
    range_y : tuple, optional
        (min, max) inclusive range for y values (2D only).
    robust : bool, default False
        If True, include robust statistics (median, q25, q75, mad).
    
    Returns
    -------
    DataFrame
        Statistics table.
    
    Notes
    -----
    Phase 13.6.G.DF Breaking Change:
        std, std_x, std_y now use population standard deviation (ddof=0)
        to match ROOT's TTree::Draw behavior. Values will be slightly
        smaller than previous versions which used sample std (ddof=1).
    """
    if group_by is not None:
        groups = df.groupby(group_by)
        results = []
        for name, group in groups:
            stats = _compute_single_stats(
                group, y_col, x_col, 
                range_x=range_x, range_y=range_y,
                robust=robust
            )
            stats['group'] = name
            results.append(stats)
        return pd.DataFrame(results)
    else:
        stats = _compute_single_stats(
            df, y_col, x_col,
            range_x=range_x, range_y=range_y,
            robust=robust
        )
        return pd.DataFrame([stats])


def _compute_single_stats(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    range_x: Optional[Tuple[float, float]] = None,
    range_y: Optional[Tuple[float, float]] = None,
    robust: bool = False,
) -> Dict[str, Any]:
    """
    Compute statistics for a single group, optionally within range.
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    y_col : str
        Primary column expression.
    x_col : str, optional
        Secondary column for 2D stats.
    range_x : tuple, optional
        (min, max) inclusive range for x values.
    range_y : tuple, optional
        (min, max) inclusive range for y values (2D only).
    robust : bool, default False
        If True, include median, q25, q75, mad.
    
    Returns
    -------
    dict
        Statistics dictionary.
    
    Notes
    -----
    Phase 13.6.G.DF:
    - std uses population std (ddof=0) to match ROOT
    - For 2D, n counts both-valid pairs only
    - Range filtering is inclusive [min, max]
    """
    stats = {}
    
    # Get y values (primary variable)
    try:
        y = pd.to_numeric(df.eval(y_col), errors='coerce')
    except:
        y = pd.to_numeric(df[y_col], errors='coerce')
    
    # 1D case: apply range filter to the single variable
    if x_col is None:
        if range_x is not None:
            # For 1D hist, range_x filters the primary variable
            mask = (y >= range_x[0]) & (y <= range_x[1])
            y = y[mask]
        
        y_valid = y.dropna()
        
        stats['n'] = len(y_valid)
        
        if len(y_valid) == 0:
            # Return all 1D fields as NaN for empty range
            stats['mean'] = np.nan
            stats['std'] = np.nan
            stats['min'] = np.nan
            stats['max'] = np.nan
            if robust:
                stats['median'] = np.nan
                stats['q25'] = np.nan
                stats['q75'] = np.nan
                stats['mad'] = np.nan
            return stats
        
        # Phase 13.6.G.DF: Use population std (ddof=0) to match ROOT
        stats['mean'] = float(y_valid.mean())
        stats['std'] = float(y_valid.std(ddof=0))
        stats['min'] = float(y_valid.min())
        stats['max'] = float(y_valid.max())
        
        # Issue 3: Robust statistics
        if robust:
            y_arr = y_valid.values
            stats['median'] = float(np.median(y_arr))
            stats['q25'] = float(np.percentile(y_arr, 25))
            stats['q75'] = float(np.percentile(y_arr, 75))
            stats['mad'] = float(np.median(np.abs(y_arr - stats['median'])))
        
        return stats
    
    # 2D case
    try:
        x = pd.to_numeric(df.eval(x_col), errors='coerce')
    except:
        x = pd.to_numeric(df[x_col], errors='coerce')
    
    # Apply 2D range filter
    mask = np.ones(len(df), dtype=bool)
    if range_x is not None:
        mask &= (x >= range_x[0]) & (x <= range_x[1])
    if range_y is not None:
        mask &= (y >= range_y[0]) & (y <= range_y[1])
    
    x = x[mask]
    y = y[mask]
    
    # For 2D plots, n must count both-valid pairs
    # (rows where both x and y are finite)
    both_valid = x.notna() & y.notna()
    stats['n'] = int(both_valid.sum())
    
    if stats['n'] == 0:
        # Return all 2D fields as NaN for empty range
        stats['mean_x'] = np.nan
        stats['mean_y'] = np.nan
        stats['std_x'] = np.nan
        stats['std_y'] = np.nan
        stats['corr'] = np.nan
        if robust:
            stats['median'] = np.nan
            stats['q25'] = np.nan
            stats['q75'] = np.nan
            stats['mad'] = np.nan
        return stats
    
    # Extract valid pairs
    x_valid = x[both_valid].values
    y_valid = y[both_valid].values
    
    # Phase 13.6.G.DF: Use population std (ddof=0) to match ROOT
    stats['mean_x'] = float(np.mean(x_valid))
    stats['std_x'] = float(np.std(x_valid, ddof=0))
    stats['mean_y'] = float(np.mean(y_valid))
    stats['std_y'] = float(np.std(y_valid, ddof=0))
    
    # Correlation requires at least 2 points
    if stats['n'] > 1:
        stats['corr'] = float(np.corrcoef(x_valid, y_valid)[0, 1])
    else:
        stats['corr'] = np.nan
    
    # Issue 3: Robust statistics for 2D (apply to y-axis only per spec)
    if robust:
        stats['median'] = float(np.median(y_valid))
        stats['q25'] = float(np.percentile(y_valid, 25))
        stats['q75'] = float(np.percentile(y_valid, 75))
        stats['mad'] = float(np.median(np.abs(y_valid - stats['median'])))
    
    return stats


def format_stats_box(
    stats: Dict[str, Any],
    fields: Optional[List[str]] = None,
    plot_type: Optional[str] = None,
) -> str:
    """
    Format statistics for display in plot.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary.
    fields : list, optional
        Fields to include. If None, auto-detected from plot_type.
    plot_type : str, optional
        Plot type hint: 'hist', 'hist2d', 'scatter', 'profile', 'hexbin'
        Used to select appropriate default fields.
        
        Default fields by plot_type:
        - 'hist': ['n', 'mean', 'std']
        - 'hist2d': ['n', 'mean_x', 'mean_y', 'std_x', 'std_y', 'corr']
        - 'scatter', 'profile', 'hexbin': ['n', 'mean_x', 'mean_y']
    
    Returns
    -------
    str
        Formatted text for stats box.
    
    Notes
    -----
    Phase 13.6.G.DF Issue 1:
        Default fields now auto-detected based on plot_type parameter.
        If plot_type is None, falls back to detecting from stats dict keys.
    """
    if fields is None:
        # Issue 1: Select defaults based on plot_type
        # Use get_default_stats_fields() to avoid logic duplication
        if plot_type is not None:
            fields = get_default_stats_fields(plot_type, robust=False)
        elif 'mean_x' in stats:
            # Fallback: 2D defaults if no plot_type but 2D stats present
            fields = get_default_stats_fields('hist2d', robust=False)
        else:
            # Fallback: 1D defaults
            fields = get_default_stats_fields('hist', robust=False)
    
    lines = []
    for field in fields:
        if field in stats:
            value = stats[field]
            if field == 'n':
                lines.append(f"N = {value:,}")
            elif isinstance(value, float):
                if np.isnan(value):
                    lines.append(f"{field} = N/A")
                else:
                    lines.append(f"{field} = {value:.4g}")
            else:
                lines.append(f"{field} = {value}")
    
    return "\n".join(lines)


def get_default_stats_fields(
    plot_type: str,
    robust: bool = False,
) -> List[str]:
    """
    Get default statistics fields for a plot type.
    
    Parameters
    ----------
    plot_type : str
        Plot type: 'hist', 'hist2d', 'scatter', 'profile', 'hexbin'
    robust : bool, default False
        If True, use robust defaults for 1D plots.
    
    Returns
    -------
    list
        List of field names.
    
    Notes
    -----
    Phase 13.6.G.DF Issue 3:
        When robust=True, 1D defaults change to ['n', 'median', 'mad'].
        2D defaults are unchanged (robust mode does not affect 2D).
    """
    if plot_type == 'hist':
        if robust:
            return ["n", "median", "mad"]
        return ["n", "mean", "std"]
    elif plot_type == 'hist2d':
        return ["n", "mean_x", "mean_y", "std_x", "std_y", "corr"]
    elif plot_type in ['scatter', 'profile', 'hexbin']:
        return ["n", "mean_x", "mean_y"]
    else:
        # Unknown plot type, use 1D defaults
        if robust:
            return ["n", "median", "mad"]
        return ["n", "mean", "std"]
