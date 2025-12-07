"""
Statistics computation for dfdraw.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional


def compute_stats(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    group_by: Optional[str] = None,
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
    
    Returns
    -------
    DataFrame
        Statistics table.
    """
    if group_by is not None:
        groups = df.groupby(group_by)
        results = []
        for name, group in groups:
            stats = _compute_single_stats(group, y_col, x_col)
            stats['group'] = name
            results.append(stats)
        return pd.DataFrame(results)
    else:
        stats = _compute_single_stats(df, y_col, x_col)
        return pd.DataFrame([stats])


def _compute_single_stats(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute statistics for a single group.
    """
    stats = {}
    
    # Get y values
    try:
        y = pd.to_numeric(df.eval(y_col), errors='coerce')
    except:
        y = pd.to_numeric(df[y_col], errors='coerce')
    
    y_valid = y.dropna()
    
    stats['n'] = len(y_valid)
    stats['mean'] = y_valid.mean() if len(y_valid) > 0 else np.nan
    stats['std'] = y_valid.std() if len(y_valid) > 0 else np.nan
    stats['min'] = y_valid.min() if len(y_valid) > 0 else np.nan
    stats['max'] = y_valid.max() if len(y_valid) > 0 else np.nan
    
    if x_col is not None:
        # 2D stats
        try:
            x = pd.to_numeric(df.eval(x_col), errors='coerce')
        except:
            x = pd.to_numeric(df[x_col], errors='coerce')
        
        x_valid = x.dropna()
        
        stats['mean_x'] = x_valid.mean() if len(x_valid) > 0 else np.nan
        stats['std_x'] = x_valid.std() if len(x_valid) > 0 else np.nan
        stats['mean_y'] = stats.pop('mean')
        stats['std_y'] = stats.pop('std')
        
        # Correlation
        mask = y.notna() & x.notna()
        if mask.sum() > 1:
            stats['corr'] = np.corrcoef(x[mask], y[mask])[0, 1]
        else:
            stats['corr'] = np.nan
    
    return stats


def format_stats_box(
    stats: Dict[str, Any],
    fields: Optional[List[str]] = None,
) -> str:
    """
    Format statistics for display in plot.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary.
    fields : list, optional
        Fields to include. Default: ["n", "mean", "std"]
    
    Returns
    -------
    str
        Formatted text for stats box.
    """
    if fields is None:
        fields = ["n", "mean", "std"]
    
    lines = []
    for field in fields:
        if field in stats:
            value = stats[field]
            if field == 'n':
                lines.append(f"N = {value:,}")
            elif isinstance(value, float):
                lines.append(f"{field} = {value:.4g}")
            else:
                lines.append(f"{field} = {value}")
    
    return "\n".join(lines)
