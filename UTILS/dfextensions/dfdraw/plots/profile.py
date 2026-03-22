"""
Profile plot implementation for dfdraw.

A profile shows the mean (and error) of y values in bins of x.
Similar to ROOT's TProfile.

Supports:
- Configurable bins and range
- Error types: sem (standard error), std, none
- Statistics box
- Group-by overlay
- Style integration

Phase 13.12.DF additions:
- F1: return_data=True → export profile statistics as DataFrame
- F2: min_entries=3 → suppress low-statistics bins (AD-1)
- F3: group_by_bins/group_by_quantiles → auto-bin float columns
- F4: sort_groups=True → sorted legend order
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union

from ..style import get_style_value
from ..stats import format_stats_box


# =============================================================================
# Phase 13.12.DF: Interval label formatting (AD-3)
# =============================================================================

def _format_interval_label(interval) -> str:
    """
    Format pandas Interval as 'low-high' string.
    
    AD-3: Custom format instead of pandas default '(0.0, 1.5]'.
    Uses consistent decimal places based on interval width.
    
    Parameters
    ----------
    interval : pandas.Interval
        Interval object from pd.cut or pd.qcut.
    
    Returns
    -------
    str
        Formatted label like '0.5-1.2'
    """
    # Determine precision based on interval width (P2 fix from Reviewer 30)
    width = interval.right - interval.left
    if width >= 10:
        fmt = ".0f"
    elif width >= 1:
        fmt = ".1f"
    elif width >= 0.1:
        fmt = ".2f"
    else:
        fmt = ".3f"
    
    return f"{interval.left:{fmt}}-{interval.right:{fmt}}"


def draw_profile(
    df: pd.DataFrame,
    x: Union[str, pd.Series, np.ndarray],
    y: Union[str, pd.Series, np.ndarray],
    ax: Optional[plt.Axes] = None,
    bins: Optional[int] = None,
    x_range: Optional[Tuple[float, float]] = None,
    error: str = "sem",
    stats: Optional[Union[bool, List[str]]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color: Optional[str] = None,
    marker: Optional[str] = None,
    markersize: Optional[float] = None,
    capsize: Optional[float] = None,
    linestyle: Optional[str] = None,
    linewidth: Optional[float] = None,
    label: Optional[str] = None,
    group_by: Optional[str] = None,
    top_k: Optional[int] = None,
    # Phase 13.12.DF: New parameters
    return_data: bool = False,
    min_entries: int = 3,
    group_by_bins: Optional[int] = None,
    group_by_quantiles: Optional[int] = None,
    sort_groups: bool = True,
    # Phase 13.12.DF v1.1: Weights support
    weights: Optional[str] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Draw profile plot (mean of y vs binned x).
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    x : str, Series, or array
        X-axis column (will be binned).
    y : str, Series, or array
        Y-axis column (mean computed per bin).
    ax : Axes, optional
        Existing axes to plot on.
    bins : int, optional
        Number of bins for x. Default from style.
    x_range : tuple, optional
        (min, max) range for x binning.
    error : str, default "sem"
        Error bar type: "sem" (standard error of mean), "std", "none".
    stats : bool or list, optional
        Show statistics box.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    color : str, optional
        Line/marker color.
    marker : str, optional
        Marker style.
    markersize : float, optional
        Marker size.
    capsize : float, optional
        Error bar cap size.
    linestyle : str, optional
        Line style connecting points.
    linewidth : float, optional
        Line width.
    label : str, optional
        Legend label.
    group_by : str, optional
        Column for grouping (creates overlaid profiles).
    top_k : int, optional
        Show only top K categories.
    return_data : bool, default False
        If True, include 'profile_data' DataFrame in stats_dict.
        Phase 13.12.DF F1.
    min_entries : int, default 3
        Minimum entries per bin to be plotted. Bins with fewer entries
        are excluded from the plot but included in profile_data if
        return_data=True. AD-1: default=3 for stable error bars.
        Phase 13.12.DF F2.
    group_by_bins : int, optional
        Number of equal-width bins for float group_by column.
        Uses pd.cut internally. Mutually exclusive with group_by_quantiles.
        Phase 13.12.DF F3.
    group_by_quantiles : int, optional
        Number of equal-count quantile bins for float group_by column.
        Uses pd.qcut internally. Mutually exclusive with group_by_bins.
        Phase 13.12.DF F3.
    sort_groups : bool, default True
        If True, sort groups numerically/alphabetically in legend.
        If False, use DataFrame occurrence order.
        Phase 13.12.DF F4.
    weights : str, optional
        Column name for weights. If provided, computes weighted mean/std/sem.
        Useful for reconstructing distributions from importance sampling.
        Phase 13.12.DF v1.1.
    **kwargs
        Additional arguments passed to plt.errorbar().
    
    Returns
    -------
    tuple
        (fig, ax, stats_dict)
        
        If return_data=True, stats_dict['profile_data'] contains a DataFrame
        with columns: x_center, x_low, x_high, y_mean, y_std, y_sem, count,
        sum_weights (if weights used), and 'group' if group_by is used.
    """
    # Phase 13.12.DF F3: Validate mutual exclusion
    if group_by_bins is not None and group_by_quantiles is not None:
        raise ValueError("Cannot specify both group_by_bins and group_by_quantiles")
    
    # Get style defaults
    if bins is None:
        bins = get_style_value("hist.bins", 50)
    if marker is None:
        marker = get_style_value("profile.marker", "o")
    if markersize is None:
        markersize = get_style_value("profile.markersize", 6)
    if capsize is None:
        capsize = get_style_value("profile.capsize", 3)
    if linestyle is None:
        linestyle = "-"
    if linewidth is None:
        linewidth = 1.5
    
    # Create figure if needed
    if ax is None:
        figsize = get_style_value("figure.figsize", (8, 6))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get data
    if isinstance(x, str):
        x_name = x
        x_data = df[x].values.astype(float)
    else:
        x_name = "x"
        x_data = np.asarray(x, dtype=float)
    
    if isinstance(y, str):
        y_name = y
        y_data = df[y].values.astype(float)
    else:
        y_name = "y"
        y_data = np.asarray(y, dtype=float)
    
    # Phase 13.12.DF v1.1: Get weights if specified
    w_data = None
    if weights is not None and weights in df.columns:
        w_data = df[weights].values.astype(float)
    
    # Remove NaN (include weights in mask if present)
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    if w_data is not None:
        mask &= ~np.isnan(w_data)
        w_data = w_data[mask]
    x_data = x_data[mask]
    y_data = y_data[mask]
    df_filtered = df[mask].copy() if len(df) == len(mask) else df.copy()
    
    # Compute profile statistics
    stats_dict = _compute_profile_stats(x_data, y_data)
    
    # Phase 13.12.DF F3: Auto-bin float group_by column
    group_col = group_by
    if group_by is not None and group_by in df_filtered.columns:
        if group_by_bins is not None:
            intervals = pd.cut(df_filtered[group_by], bins=group_by_bins)
            df_filtered['_group'] = intervals.map(_format_interval_label)
            group_col = '_group'
        elif group_by_quantiles is not None:
            intervals = pd.qcut(df_filtered[group_by], q=group_by_quantiles, duplicates='drop')
            df_filtered['_group'] = intervals.map(_format_interval_label)
            group_col = '_group'
    
    # Group-by handling
    if group_col is not None and group_col in df_filtered.columns:
        profile_data_list = _draw_profile_grouped(
            df_filtered, x, y, ax, group_col, top_k,
            bins=bins, x_range=x_range, error=error,
            marker=marker, markersize=markersize, capsize=capsize,
            linestyle=linestyle, linewidth=linewidth,
            min_entries=min_entries,
            sort_groups=sort_groups,
            return_data=return_data,
            weights=weights,  # Phase 13.12.DF v1.1
            **kwargs
        )
        stats_dict["grouped"] = True
        
        # Phase 13.12.DF F1: Combine profile data from all groups
        if return_data and profile_data_list:
            stats_dict['profile_data'] = pd.concat(profile_data_list, ignore_index=True)
    else:
        # Single profile
        bin_centers, bin_means, bin_errors, bin_counts, profile_df = _compute_profile(
            x_data, y_data, bins, x_range, error, return_data=return_data,
            w_data=w_data  # Phase 13.12.DF v1.1
        )
        
        # Phase 13.12.DF F2: Apply min_entries filter for plotting
        plot_mask = bin_counts >= min_entries
        
        ax.errorbar(
            bin_centers[plot_mask], bin_means[plot_mask], yerr=bin_errors[plot_mask],
            fmt=marker, color=color, markersize=markersize,
            capsize=capsize, linestyle=linestyle, linewidth=linewidth,
            label=label, **kwargs
        )
        
        # Phase 13.12.DF F1: Add profile data to stats
        if return_data and profile_df is not None:
            stats_dict['profile_data'] = profile_df
    
    # Labels
    ax.set_xlabel(xlabel or x_name)
    ax.set_ylabel(ylabel or f"<{y_name}>")
    
    if title:
        ax.set_title(title)
    
    # Statistics box
    if stats is True or (stats is None and get_style_value("stats.show", False)):
        _add_stats_box(ax, stats_dict, stats if isinstance(stats, list) else None)
    elif isinstance(stats, list):
        _add_stats_box(ax, stats_dict, stats)
    
    # Legend for grouped
    if group_col is not None:
        ax.legend(loc=get_style_value("legend.loc", "best"))
    
    plt.tight_layout()
    return fig, ax, stats_dict


def _compute_profile(
    x_data: np.ndarray,
    y_data: np.ndarray,
    bins: int,
    x_range: Optional[Tuple[float, float]],
    error: str,
    return_data: bool = False,
    w_data: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[pd.DataFrame]]:
    """
    Compute profile (mean of y in bins of x).
    
    Parameters
    ----------
    x_data : array
        X values.
    y_data : array
        Y values.
    bins : int
        Number of bins.
    x_range : tuple or None
        (min, max) range for binning.
    error : str
        Error type: "sem", "std", "none".
    return_data : bool
        If True, return profile DataFrame.
    w_data : array, optional
        Weights for weighted statistics. Phase 13.12.DF v1.1.
    
    Returns
    -------
    tuple
        (bin_centers, bin_means, bin_errors, bin_counts, profile_df or None)
        
        Phase 13.12.DF: Extended return to include bin_counts and profile_df.
        Phase 13.12.DF v1.1: Supports weighted statistics.
    """
    if x_range is None:
        x_range = (np.nanmin(x_data), np.nanmax(x_data))
    
    # Create bins
    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Digitize x values
    bin_indices = np.digitize(x_data, bin_edges) - 1
    # Clip to valid range (handle edge cases)
    bin_indices = np.clip(bin_indices, 0, bins - 1)
    
    # Compute mean, std, sem, count for each bin
    bin_means = np.full(bins, np.nan)
    bin_stds = np.full(bins, np.nan)
    bin_sems = np.full(bins, np.nan)
    bin_counts = np.zeros(bins, dtype=int)
    bin_sum_weights = np.full(bins, np.nan) if w_data is not None else None
    
    for i in range(bins):
        mask = bin_indices == i
        y_bin = y_data[mask]
        n = len(y_bin)
        bin_counts[i] = n
        
        if n > 0:
            if w_data is not None:
                # Phase 13.12.DF v1.1: Weighted statistics
                w_bin = w_data[mask]
                sum_w = np.sum(w_bin)
                bin_sum_weights[i] = sum_w
                
                if sum_w > 0:
                    # Weighted mean: Σ(w × y) / Σw
                    bin_means[i] = np.sum(w_bin * y_bin) / sum_w
                    
                    if n > 1:
                        # Weighted variance: Σ(w × (y - mean)²) / Σw
                        weighted_var = np.sum(w_bin * (y_bin - bin_means[i])**2) / sum_w
                        bin_stds[i] = np.sqrt(weighted_var)
                        
                        # Effective sample size: (Σw)² / Σ(w²)
                        sum_w2 = np.sum(w_bin**2)
                        n_eff = (sum_w**2) / sum_w2 if sum_w2 > 0 else 1
                        
                        # Weighted SEM: std / sqrt(n_eff)
                        bin_sems[i] = bin_stds[i] / np.sqrt(n_eff) if n_eff > 0 else np.nan
            else:
                # Unweighted statistics (original behavior)
                bin_means[i] = np.mean(y_bin)
                if n > 1:
                    bin_stds[i] = np.std(y_bin, ddof=1)
                    bin_sems[i] = bin_stds[i] / np.sqrt(n)
    
    # Select error type
    if error == "std":
        bin_errors = bin_stds.copy()
    elif error == "none":
        bin_errors = np.zeros(bins)
    else:  # "sem" (default)
        bin_errors = bin_sems.copy()
    
    # Phase 13.12.DF F1: Build DataFrame if requested
    profile_df = None
    if return_data:
        profile_df = pd.DataFrame({
            'x_center': bin_centers,
            'x_low': bin_edges[:-1],
            'x_high': bin_edges[1:],
            'y_mean': bin_means,
            'y_std': bin_stds,
            'y_sem': bin_sems,
            'count': bin_counts,
        })
        # Phase 13.12.DF v1.1: Add sum_weights column if weighted
        if bin_sum_weights is not None:
            profile_df['sum_weights'] = bin_sum_weights
    
    return bin_centers, bin_means, bin_errors, bin_counts, profile_df


def _compute_profile_stats(x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
    """Compute overall profile statistics."""
    n = len(x_data)
    return {
        "n": n,
        "mean_x": float(np.mean(x_data)) if n > 0 else np.nan,
        "mean_y": float(np.mean(y_data)) if n > 0 else np.nan,
        "std_x": float(np.std(x_data)) if n > 0 else np.nan,
        "std_y": float(np.std(y_data)) if n > 0 else np.nan,
        "corr": float(np.corrcoef(x_data, y_data)[0, 1]) if n > 1 else np.nan,
    }


def _draw_profile_grouped(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax: plt.Axes,
    group_by: str,
    top_k: Optional[int],
    min_entries: int = 3,
    sort_groups: bool = True,
    return_data: bool = False,
    weights: Optional[str] = None,  # Phase 13.12.DF v1.1
    **profile_kwargs
) -> Optional[List[pd.DataFrame]]:
    """
    Draw grouped profile plots.
    
    Phase 13.12.DF: Added min_entries, sort_groups, return_data parameters.
    Phase 13.12.DF v1.1: Added weights parameter.
    
    Returns
    -------
    list of DataFrame or None
        If return_data=True, returns list of profile DataFrames (one per group).
    """
    # Get groups
    groups = df[group_by].unique()
    
    # Phase 13.12.DF F4: Sort groups
    if sort_groups:
        try:
            # Try numeric sort first
            groups = sorted(groups, key=lambda x: float(x) if not pd.isna(x) else float('inf'))
        except (ValueError, TypeError):
            # Fall back to string sort
            groups = sorted(groups, key=str)
    
    # Top-K filtering
    if top_k is not None and len(groups) > top_k:
        counts = df[group_by].value_counts()
        top_groups = counts.head(top_k).index.tolist()
        # Preserve sort order
        if sort_groups:
            try:
                top_groups = sorted(top_groups, key=lambda x: float(x) if not pd.isna(x) else float('inf'))
            except (ValueError, TypeError):
                top_groups = sorted(top_groups, key=str)
        groups = top_groups
    
    # Color palette
    palette_name = get_style_value("colors.palette", "tab10")
    palette = plt.colormaps.get_cmap(palette_name)
    
    # Marker cycle
    markers = get_style_value("markers.cycle", ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"])
    
    # Extract common kwargs
    bins = profile_kwargs.pop('bins', 50)
    x_range = profile_kwargs.pop('x_range', None)
    error = profile_kwargs.pop('error', 'sem')
    # Remove marker/markersize from kwargs - we use fmt and dedicated markers
    profile_kwargs.pop('marker', None)
    profile_kwargs.pop('markersize', None)
    
    # Phase 13.12.DF F1: Collect profile data
    profile_data_list = [] if return_data else None
    
    for i, group in enumerate(groups):
        group_df = df[df[group_by] == group]
        x_data = group_df[x].values.astype(float)
        y_data = group_df[y].values.astype(float)
        
        # Phase 13.12.DF v1.1: Get weights for this group
        w_data = None
        if weights is not None and weights in group_df.columns:
            w_data = group_df[weights].values.astype(float)
        
        # Remove NaN (include weights in mask if present)
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        if w_data is not None:
            mask &= ~np.isnan(w_data)
            w_data = w_data[mask]
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        if len(x_data) == 0:
            continue
        
        bin_centers, bin_means, bin_errors, bin_counts, profile_df = _compute_profile(
            x_data, y_data, bins, x_range, error, return_data=return_data,
            w_data=w_data  # Phase 13.12.DF v1.1
        )
        
        # Phase 13.12.DF F1: Add group column and collect
        if return_data and profile_df is not None:
            profile_df['group'] = group
            profile_data_list.append(profile_df)
        
        # Phase 13.12.DF F2: Apply min_entries filter for plotting
        plot_mask = bin_counts >= min_entries
        
        ax.errorbar(
            bin_centers[plot_mask], bin_means[plot_mask], yerr=bin_errors[plot_mask],
            fmt=markers[i % len(markers)],
            color=palette(i % 10),
            label=str(group),
            **profile_kwargs
        )
    
    return profile_data_list


def _add_stats_box(
    ax: plt.Axes,
    stats: Dict[str, Any],
    fields: Optional[List[str]] = None
) -> None:
    """Add statistics box to plot."""
    if fields is None:
        fields = get_style_value("stats.fields", ["n", "mean_y", "corr"])
    
    text = format_stats_box(stats, fields)
    
    position = get_style_value("stats.position", "upper right")
    fontsize = get_style_value("stats.fontsize", 10)
    alpha = get_style_value("stats.alpha", 0.8)
    boxstyle = get_style_value("stats.boxstyle", "round")
    
    pos_map = {
        "upper right": (0.95, 0.95, "right", "top"),
        "upper left": (0.05, 0.95, "left", "top"),
        "lower right": (0.95, 0.05, "right", "bottom"),
        "lower left": (0.05, 0.05, "left", "bottom"),
    }
    x, y, ha, va = pos_map.get(position, (0.95, 0.95, "right", "top"))
    
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=dict(boxstyle=boxstyle, facecolor="white", alpha=alpha)
    )
