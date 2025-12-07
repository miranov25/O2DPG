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
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union

from ..style import get_style_value
from ..stats import format_stats_box


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
    **kwargs
        Additional arguments passed to plt.errorbar().
    
    Returns
    -------
    tuple
        (fig, ax, stats_dict)
    """
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
    
    # Remove NaN
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[mask]
    y_data = y_data[mask]
    df_filtered = df[mask] if len(df) == len(mask) else df
    
    # Compute profile statistics
    stats_dict = _compute_profile_stats(x_data, y_data)
    
    # Group-by handling
    if group_by is not None and group_by in df.columns:
        _draw_profile_grouped(
            df_filtered, x, y, ax, group_by, top_k,
            bins=bins, x_range=x_range, error=error,
            marker=marker, markersize=markersize, capsize=capsize,
            linestyle=linestyle, linewidth=linewidth, **kwargs
        )
        stats_dict["grouped"] = True
    else:
        # Single profile
        bin_centers, bin_means, bin_errors = _compute_profile(
            x_data, y_data, bins, x_range, error
        )
        
        ax.errorbar(
            bin_centers, bin_means, yerr=bin_errors,
            fmt=marker, color=color, markersize=markersize,
            capsize=capsize, linestyle=linestyle, linewidth=linewidth,
            label=label, **kwargs
        )
    
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
    if group_by is not None:
        ax.legend(loc=get_style_value("legend.loc", "best"))
    
    plt.tight_layout()
    return fig, ax, stats_dict


def _compute_profile(
    x_data: np.ndarray,
    y_data: np.ndarray,
    bins: int,
    x_range: Optional[Tuple[float, float]],
    error: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute profile (mean of y in bins of x).
    
    Returns
    -------
    tuple
        (bin_centers, bin_means, bin_errors)
    """
    if x_range is None:
        x_range = (np.nanmin(x_data), np.nanmax(x_data))
    
    # Create bins
    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Digitize x values
    bin_indices = np.digitize(x_data, bin_edges) - 1
    
    # Compute mean and error for each bin
    bin_means = np.full(bins, np.nan)
    bin_errors = np.full(bins, np.nan)
    
    for i in range(bins):
        mask = bin_indices == i
        y_bin = y_data[mask]
        
        if len(y_bin) > 0:
            bin_means[i] = np.mean(y_bin)
            
            if error == "sem" and len(y_bin) > 1:
                bin_errors[i] = np.std(y_bin, ddof=1) / np.sqrt(len(y_bin))
            elif error == "std":
                bin_errors[i] = np.std(y_bin)
            elif error == "none":
                bin_errors[i] = 0
            else:  # default to sem
                if len(y_bin) > 1:
                    bin_errors[i] = np.std(y_bin, ddof=1) / np.sqrt(len(y_bin))
                else:
                    bin_errors[i] = 0
    
    return bin_centers, bin_means, bin_errors


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
    **profile_kwargs
) -> None:
    """Draw grouped profile plots."""
    # Get groups
    groups = df[group_by].unique()
    
    # Top-K filtering
    if top_k is not None and len(groups) > top_k:
        counts = df[group_by].value_counts()
        top_groups = counts.head(top_k).index.tolist()
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
    
    for i, group in enumerate(groups):
        group_df = df[df[group_by] == group]
        x_data = group_df[x].values.astype(float)
        y_data = group_df[y].values.astype(float)
        
        # Remove NaN
        mask = ~(np.isnan(x_data) | np.isnan(y_data))
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        if len(x_data) == 0:
            continue
        
        bin_centers, bin_means, bin_errors = _compute_profile(
            x_data, y_data, bins, x_range, error
        )
        
        ax.errorbar(
            bin_centers, bin_means, yerr=bin_errors,
            fmt=markers[i % len(markers)],
            color=palette(i % 10),
            label=str(group),
            **profile_kwargs
        )


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
