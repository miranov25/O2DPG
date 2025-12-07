"""
Histogram plot implementation for dfdraw.

Supports:
- 1D histograms with configurable bins
- Normalization: count, density, probability
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


def draw_hist(
    df: pd.DataFrame,
    x: Union[str, pd.Series, np.ndarray],
    ax: Optional[plt.Axes] = None,
    bins: Optional[int] = None,
    range: Optional[Tuple[float, float]] = None,
    norm: Optional[str] = None,
    stats: Optional[Union[bool, List[str]]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    color: Optional[str] = None,
    alpha: Optional[float] = None,
    histtype: Optional[str] = None,
    edgecolor: Optional[str] = None,
    linewidth: Optional[float] = None,
    label: Optional[str] = None,
    group_by: Optional[str] = None,
    top_k: Optional[int] = None,
    stacked: bool = False,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Draw 1D histogram.
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    x : str, Series, or array
        Column name or data to histogram.
    ax : Axes, optional
        Existing axes to plot on. Creates new figure if None.
    bins : int, optional
        Number of bins. Default from style.
    range : tuple, optional
        (min, max) range for histogram.
    norm : str, optional
        Normalization: "count" (default), "density", "probability".
    stats : bool or list, optional
        Show statistics box. True for defaults, or list of stat names.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label. Defaults to column name.
    ylabel : str, optional
        Y-axis label.
    color : str, optional
        Histogram color.
    alpha : float, optional
        Transparency.
    histtype : str, optional
        Histogram type: "bar", "step", "stepfilled".
    edgecolor : str, optional
        Edge color.
    linewidth : float, optional
        Edge line width.
    label : str, optional
        Legend label.
    group_by : str, optional
        Column for grouping (creates overlaid histograms).
    top_k : int, optional
        Show only top K categories when using group_by.
    stacked : bool, default False
        Stack histograms when using group_by.
    **kwargs
        Additional arguments passed to plt.hist().
    
    Returns
    -------
    tuple
        (fig, ax, stats_dict)
    """
    # Get style defaults
    if bins is None:
        bins = get_style_value("hist.bins", 50)
    if alpha is None:
        alpha = get_style_value("hist.alpha", 0.7)
    if histtype is None:
        histtype = get_style_value("hist.histtype", "stepfilled")
    if edgecolor is None:
        edgecolor = get_style_value("hist.edgecolor", "black")
    if linewidth is None:
        linewidth = get_style_value("hist.linewidth", 1.0)
    
    # Create figure if needed
    if ax is None:
        figsize = get_style_value("figure.figsize", (8, 6))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get data
    if isinstance(x, str):
        x_name = x
        x_data = df[x].values
    else:
        x_name = "x"
        x_data = np.asarray(x)
    
    # Remove NaN
    mask = ~np.isnan(x_data.astype(float))
    x_data = x_data[mask]
    
    # Statistics dict
    stats_dict = {
        "n": len(x_data),
        "mean": float(np.mean(x_data)) if len(x_data) > 0 else np.nan,
        "std": float(np.std(x_data)) if len(x_data) > 0 else np.nan,
        "min": float(np.min(x_data)) if len(x_data) > 0 else np.nan,
        "max": float(np.max(x_data)) if len(x_data) > 0 else np.nan,
    }
    
    # Normalization
    density = False
    weights = None
    if norm == "density":
        density = True
    elif norm == "probability":
        weights = np.ones_like(x_data) / len(x_data) if len(x_data) > 0 else None
    
    # Group-by handling
    if group_by is not None and group_by in df.columns:
        _draw_hist_grouped(
            df, x, ax, group_by, top_k, stacked,
            bins=bins, range=range, density=density, weights=weights,
            alpha=alpha, histtype=histtype, edgecolor=edgecolor,
            linewidth=linewidth, **kwargs
        )
        stats_dict["grouped"] = True
    else:
        # Single histogram
        ax.hist(
            x_data, bins=bins, range=range, density=density, weights=weights,
            color=color, alpha=alpha, histtype=histtype, edgecolor=edgecolor,
            linewidth=linewidth, label=label, **kwargs
        )
    
    # Labels
    ax.set_xlabel(xlabel or x_name)
    if ylabel:
        ax.set_ylabel(ylabel)
    elif norm == "density":
        ax.set_ylabel("Density")
    elif norm == "probability":
        ax.set_ylabel("Probability")
    else:
        ax.set_ylabel("Count")
    
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


def _draw_hist_grouped(
    df: pd.DataFrame,
    x: str,
    ax: plt.Axes,
    group_by: str,
    top_k: Optional[int],
    stacked: bool,
    **hist_kwargs
) -> None:
    """Draw grouped/overlaid histograms."""
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
    colors = [palette(i % 10) for i in range(len(groups))]
    
    if stacked:
        # Stacked histogram
        data_list = [df[df[group_by] == g][x].dropna().values for g in groups]
        ax.hist(data_list, label=[str(g) for g in groups], color=colors,
                stacked=True, **hist_kwargs)
    else:
        # Overlaid histograms
        for i, group in enumerate(groups):
            group_data = df[df[group_by] == group][x].dropna().values
            ax.hist(group_data, label=str(group), color=colors[i], **hist_kwargs)


def _add_stats_box(
    ax: plt.Axes,
    stats: Dict[str, Any],
    fields: Optional[List[str]] = None
) -> None:
    """Add statistics box to plot."""
    if fields is None:
        fields = get_style_value("stats.fields", ["n", "mean", "std"])
    
    text = format_stats_box(stats, fields)
    
    position = get_style_value("stats.position", "upper right")
    fontsize = get_style_value("stats.fontsize", 10)
    alpha = get_style_value("stats.alpha", 0.8)
    boxstyle = get_style_value("stats.boxstyle", "round")
    
    # Position mapping
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
