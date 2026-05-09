"""
Scatter plot implementation for dfdraw.

Supports:
- Color mapping (continuous and categorical)
- Size mapping
- Marker mapping
- Statistics box
- Group-by overlay
- Jitter for quantized data
- Style integration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Any, Dict, List, Optional, Tuple, Union

from ..style import get_style_value
from ..stats import format_stats_box
# Phase 13.28.DF: Robust data handling
from ._data_sanitize import sanitize_for_plot


def draw_scatter(
    df: pd.DataFrame,
    x: Union[str, pd.Series, np.ndarray],
    y: Union[str, pd.Series, np.ndarray],
    ax: Optional[plt.Axes] = None,
    color: Optional[Union[str, np.ndarray]] = None,
    size: Optional[Union[str, float, np.ndarray]] = None,
    marker: Optional[Union[str, List[str]]] = None,
    stats: Optional[Union[bool, List[str]]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    alpha: Optional[float] = None,
    edgecolors: Optional[str] = None,
    linewidths: Optional[float] = None,
    cmap: Optional[str] = None,
    colorbar: bool = True,
    clabel: Optional[str] = None,
    group_by: Optional[str] = None,
    top_k: Optional[int] = None,
    jitter: Optional[Union[bool, float, Tuple[float, float]]] = None,
    # Phase 13.28.DF: NaN/inf filter policy (AD-70)
    nan_policy: str = "filter",
    # Phase 13.16.DF FIX1: vector dispatch suppression flags (private).
    _suppress_legend: bool = False,
    _suppress_title: bool = False,
    _suppress_layout: bool = False,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Draw scatter plot.
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    x : str, Series, or array
        X-axis column name or data.
    y : str, Series, or array
        Y-axis column name or data.
    ax : Axes, optional
        Existing axes to plot on.
    color : str, array, optional
        Column for color mapping, fixed color, or array.
        If column: continuous → colormap, categorical → discrete colors.
    size : str, float, or array, optional
        Column for size mapping or fixed size.
    marker : str or list, optional
        Marker style or column for marker mapping.
    stats : bool or list, optional
        Show statistics box.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    alpha : float, optional
        Transparency.
    edgecolors : str, optional
        Marker edge color.
    linewidths : float, optional
        Marker edge width.
    cmap : str, optional
        Colormap for continuous color mapping.
    colorbar : bool, default True
        Show colorbar for continuous color mapping.
    clabel : str, optional
        Colorbar label.
    group_by : str, optional
        Column for grouping (creates overlaid scatter plots).
    top_k : int, optional
        Show only top K categories.
    jitter : bool, float, or tuple, optional
        Add jitter to points. True for auto, float for amount,
        tuple for (x_jitter, y_jitter).
    **kwargs
        Additional arguments passed to plt.scatter().
    
    Returns
    -------
    tuple
        (fig, ax, stats_dict)
    """
    # Get style defaults
    if alpha is None:
        alpha = get_style_value("scatter.alpha", 0.7)
    if edgecolors is None:
        edgecolors = get_style_value("scatter.edgecolors", "black")
    if linewidths is None:
        linewidths = get_style_value("scatter.linewidths", 0.5)
    default_size = get_style_value("scatter.size", 50)
    
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
    
    # Phase 13.28.DF: NaN/inf sanitization (AD-69, AD-70).
    # Compute counters then build joint mask (so df_filtered + mask align).
    _, _, _sanitize_stats = sanitize_for_plot(
        x_data, y_data, nan_policy=nan_policy, column_names=(x_name, y_name)
    )
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    x_data = x_data[mask]
    y_data = y_data[mask]
    df_filtered = df[mask] if len(df) == len(mask) else df
    
    # Statistics
    stats_dict = _compute_scatter_stats(x_data, y_data)
    # Phase 13.28.DF: Sanitize counters (AD-71)
    stats_dict.update(_sanitize_stats)
    # Scatter doesn't auto-range; record explicit / matplotlib-default.
    stats_dict["autorange_used"] = (
        ((float(x_data.min()), float(x_data.max())),
         (float(y_data.min()), float(y_data.max())))
        if len(x_data) > 0 else ((0.0, 1.0), (0.0, 1.0))
    )
    stats_dict["autorange_strategy"] = "minmax"
    
    # Apply jitter
    if jitter:
        x_data, y_data = _apply_jitter(x_data, y_data, jitter)
    
    # Group-by handling
    if group_by is not None and group_by in df.columns:
        _draw_scatter_grouped(
            df_filtered, x, y, ax, group_by, top_k,
            alpha=alpha, edgecolors=edgecolors, linewidths=linewidths,
            s=size if size is not None else default_size, **kwargs
        )
        stats_dict["grouped"] = True
    else:
        # Process color
        c, cmap_used, is_categorical = _process_color(
            df_filtered, color, cmap, mask, len(x_data)
        )
        
        # Process size
        s = _process_size(df_filtered, size, default_size, mask, len(x_data))
        
        # Draw scatter
        scatter = ax.scatter(
            x_data, y_data,
            c=c, s=s, alpha=alpha,
            edgecolors=edgecolors, linewidths=linewidths,
            cmap=cmap_used if not is_categorical else None,
            marker=marker if isinstance(marker, str) else 'o',
            **kwargs
        )
        
        # Colorbar for continuous color
        if c is not None and not is_categorical and colorbar:
            cbar = plt.colorbar(scatter, ax=ax)
            if clabel:
                cbar.set_label(clabel)
            elif isinstance(color, str) and color in df.columns:
                cbar.set_label(color)
        
        # Legend for categorical color (Phase 13.16.DF FIX1: skip when suppressed)
        if is_categorical and isinstance(color, str) and not _suppress_legend:
            ax.legend(loc=get_style_value("legend.loc", "best"))
    
    # Labels
    ax.set_xlabel(xlabel or x_name)
    ax.set_ylabel(ylabel or y_name)
    
    if title and not _suppress_title:
        ax.set_title(title)
    
    # Statistics box
    if stats is True or (stats is None and get_style_value("stats.show", False)):
        _add_stats_box(ax, stats_dict, stats if isinstance(stats, list) else None)
    elif isinstance(stats, list):
        _add_stats_box(ax, stats_dict, stats)
    
    # Legend for grouped (Phase 13.16.DF FIX1: skip when suppressed)
    if group_by is not None and not _suppress_legend:
        ax.legend(loc=get_style_value("legend.loc", "best"))
    
    # Phase 13.16.DF FIX1: skip tight_layout when suppressed
    if not _suppress_layout:
        plt.tight_layout()
    return fig, ax, stats_dict


def _compute_scatter_stats(x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, Any]:
    """Compute scatter plot statistics."""
    n = len(x_data)
    stats = {
        "n": n,
        "mean_x": float(np.mean(x_data)) if n > 0 else np.nan,
        "mean_y": float(np.mean(y_data)) if n > 0 else np.nan,
        "std_x": float(np.std(x_data)) if n > 0 else np.nan,
        "std_y": float(np.std(y_data)) if n > 0 else np.nan,
    }
    
    # Correlation
    if n > 1:
        stats["corr"] = float(np.corrcoef(x_data, y_data)[0, 1])
    else:
        stats["corr"] = np.nan
    
    return stats


def _process_color(
    df: pd.DataFrame,
    color: Optional[Union[str, np.ndarray]],
    cmap: Optional[str],
    mask: np.ndarray,
    n_points: int
) -> Tuple[Optional[np.ndarray], Optional[str], bool]:
    """
    Process color specification.
    
    Returns
    -------
    tuple
        (color_array, colormap_name, is_categorical)
    """
    if color is None:
        return None, None, False
    
    # Fixed color string (e.g., "blue", "#FF0000")
    if isinstance(color, str) and color not in df.columns:
        return color, None, False
    
    # Array provided directly
    if isinstance(color, np.ndarray):
        return color, cmap or "viridis", False
    
    # Column name
    if isinstance(color, str) and color in df.columns:
        color_data = df[color].values
        if len(mask) == len(color_data):
            color_data = color_data[mask]
        
        # Check if categorical
        if color_data.dtype == object or isinstance(color_data.dtype, pd.CategoricalDtype):
            # Categorical - will be handled separately
            return None, None, True
        else:
            # Continuous
            return color_data.astype(float), cmap or "viridis", False
    
    return None, None, False


def _process_size(
    df: pd.DataFrame,
    size: Optional[Union[str, float, np.ndarray]],
    default_size: float,
    mask: np.ndarray,
    n_points: int
) -> np.ndarray:
    """Process size specification."""
    if size is None:
        return default_size
    
    if isinstance(size, (int, float)):
        return size
    
    if isinstance(size, np.ndarray):
        return size
    
    if isinstance(size, str) and size in df.columns:
        size_data = df[size].values.astype(float)
        if len(mask) == len(size_data):
            size_data = size_data[mask]
        # Normalize to reasonable range
        size_min, size_max = 20, 200
        s_min, s_max = np.nanmin(size_data), np.nanmax(size_data)
        if s_max > s_min:
            return size_min + (size_data - s_min) / (s_max - s_min) * (size_max - size_min)
        return default_size
    
    return default_size


def _apply_jitter(
    x_data: np.ndarray,
    y_data: np.ndarray,
    jitter: Union[bool, float, Tuple[float, float]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply jitter to data points."""
    if jitter is True:
        # Auto jitter based on data range
        x_range = np.ptp(x_data) if len(x_data) > 0 else 1
        y_range = np.ptp(y_data) if len(y_data) > 0 else 1
        jitter_x = x_range * 0.01
        jitter_y = y_range * 0.01
    elif isinstance(jitter, (int, float)):
        jitter_x = jitter_y = jitter
    else:
        jitter_x, jitter_y = jitter
    
    x_data = x_data + np.random.uniform(-jitter_x, jitter_x, len(x_data))
    y_data = y_data + np.random.uniform(-jitter_y, jitter_y, len(y_data))
    
    return x_data, y_data


def _draw_scatter_grouped(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax: plt.Axes,
    group_by: str,
    top_k: Optional[int],
    **scatter_kwargs
) -> None:
    """Draw grouped scatter plots."""
    import matplotlib.pyplot as plt
    
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
    
    for i, group in enumerate(groups):
        group_df = df[df[group_by] == group]
        x_data = group_df[x].values.astype(float)
        y_data = group_df[y].values.astype(float)
        
        # Phase 13.28.DF: catch NaN AND inf via isfinite
        mask = np.isfinite(x_data) & np.isfinite(y_data)
        x_data = x_data[mask]
        y_data = y_data[mask]
        
        ax.scatter(
            x_data, y_data,
            c=[palette(i % 10)],
            marker=markers[i % len(markers)],
            label=str(group),
            **scatter_kwargs
        )


def _add_stats_box(
    ax: plt.Axes,
    stats: Dict[str, Any],
    fields: Optional[List[str]] = None
) -> None:
    """Add statistics box to plot."""
    if fields is None:
        fields = get_style_value("stats.fields", ["n", "mean_x", "mean_y", "corr"])
    
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
