"""
Facet plot utilities for dfdraw.

Creates subplot grids where each subplot shows data for one group.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from .style import get_style_value


def create_facet_grid(
    df: pd.DataFrame,
    group_by: str,
    top_k: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = True,
    sharey: bool = True,
) -> Tuple[plt.Figure, np.ndarray, List[Any]]:
    """
    Create a grid of subplots for faceted plotting.
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    group_by : str
        Column name for grouping.
    top_k : int, optional
        Limit to top K groups by count.
    ncols : int, optional
        Number of columns. Auto-calculated if None.
    figsize : tuple, optional
        Figure size. Auto-calculated if None.
    sharex : bool, default True
        Share x-axis across subplots.
    sharey : bool, default True
        Share y-axis across subplots.
    
    Returns
    -------
    tuple
        (fig, axes_array, groups_list)
    """
    # Get groups
    groups = df[group_by].unique()
    
    # Top-K filtering
    if top_k is not None and len(groups) > top_k:
        counts = df[group_by].value_counts()
        groups = counts.head(top_k).index.tolist()
    else:
        groups = list(groups)
    
    n_groups = len(groups)
    
    if n_groups == 0:
        raise ValueError(f"No groups found in column '{group_by}'")
    
    # Calculate grid dimensions
    if ncols is None:
        ncols = min(3, n_groups)  # Default max 3 columns
    nrows = int(np.ceil(n_groups / ncols))
    
    # Calculate figure size
    if figsize is None:
        base_size = get_style_value("figure.figsize", (8, 6))
        subplot_width = base_size[0] / 1.5
        subplot_height = base_size[1] / 1.5
        figsize = (subplot_width * ncols, subplot_height * nrows)
    
    # Create figure and axes
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=False  # Always return 2D array
    )
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    # Hide unused subplots
    for idx in range(n_groups, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    return fig, axes_flat[:n_groups], groups


def draw_facet(
    df: pd.DataFrame,
    group_by: str,
    plot_func: Callable,
    top_k: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = True,
    sharey: bool = True,
    title_template: str = "{group}",
    suptitle: Optional[str] = None,
    **plot_kwargs
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Any]]:
    """
    Create faceted plot with subplots for each group.
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    group_by : str
        Column for grouping.
    plot_func : callable
        Function to call for each subplot: plot_func(df_subset, ax, **kwargs)
    top_k : int, optional
        Limit to top K groups.
    ncols : int, optional
        Number of columns.
    figsize : tuple, optional
        Figure size.
    sharex : bool, default True
        Share x-axis.
    sharey : bool, default True
        Share y-axis.
    title_template : str, default "{group}"
        Template for subplot titles. {group} is replaced with group name.
    suptitle : str, optional
        Overall figure title.
    **plot_kwargs
        Additional arguments passed to plot_func.
    
    Returns
    -------
    tuple
        (fig, axes_array, combined_stats_dict)
    """
    # Create grid
    fig, axes, groups = create_facet_grid(
        df, group_by, top_k=top_k, ncols=ncols, figsize=figsize,
        sharex=sharex, sharey=sharey
    )
    
    # Collect stats from all subplots
    all_stats = {}
    
    # Plot each group
    for ax, group in zip(axes, groups):
        # Filter data for this group
        group_df = df[df[group_by] == group]
        
        # Call plot function
        stats = plot_func(group_df, ax, **plot_kwargs)
        
        # Set subplot title
        ax.set_title(title_template.format(group=group))
        
        # Store stats
        all_stats[str(group)] = stats
    
    # Add suptitle
    if suptitle:
        fig.suptitle(suptitle, fontsize=get_style_value("axes.titlesize", 14) + 2)
    
    # Adjust layout
    plt.tight_layout()
    if suptitle:
        plt.subplots_adjust(top=0.92)
    
    # Combined stats
    combined_stats = {
        "n_groups": len(groups),
        "groups": groups,
        "per_group": all_stats,
        "faceted": True,
    }
    
    # Add total n
    total_n = sum(s.get("n", 0) for s in all_stats.values())
    combined_stats["n_total"] = total_n
    
    return fig, axes, combined_stats


def facet_hist(
    df: pd.DataFrame,
    x: str,
    group_by: str,
    top_k: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = True,
    sharey: bool = True,
    suptitle: Optional[str] = None,
    **hist_kwargs
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Any]]:
    """
    Create faceted histogram with subplots for each group.
    """
    from .plots.histogram import draw_hist
    
    def plot_func(group_df, ax, **kwargs):
        _, _, stats = draw_hist(group_df, x, ax=ax, **kwargs)
        return stats
    
    return draw_facet(
        df, group_by, plot_func,
        top_k=top_k, ncols=ncols, figsize=figsize,
        sharex=sharex, sharey=sharey, suptitle=suptitle,
        **hist_kwargs
    )


def facet_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    group_by: str,
    top_k: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = True,
    sharey: bool = True,
    suptitle: Optional[str] = None,
    **scatter_kwargs
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Any]]:
    """
    Create faceted scatter plot with subplots for each group.
    """
    from .plots.scatter import draw_scatter
    
    def plot_func(group_df, ax, **kwargs):
        _, _, stats = draw_scatter(group_df, x, y, ax=ax, **kwargs)
        return stats
    
    return draw_facet(
        df, group_by, plot_func,
        top_k=top_k, ncols=ncols, figsize=figsize,
        sharex=sharex, sharey=sharey, suptitle=suptitle,
        **scatter_kwargs
    )


def facet_profile(
    df: pd.DataFrame,
    x: str,
    y: str,
    group_by: str,
    top_k: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = True,
    sharey: bool = True,
    suptitle: Optional[str] = None,
    **profile_kwargs
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Any]]:
    """
    Create faceted profile plot with subplots for each group.
    """
    from .plots.profile import draw_profile
    
    def plot_func(group_df, ax, **kwargs):
        _, _, stats = draw_profile(group_df, x, y, ax=ax, **kwargs)
        return stats
    
    return draw_facet(
        df, group_by, plot_func,
        top_k=top_k, ncols=ncols, figsize=figsize,
        sharex=sharex, sharey=sharey, suptitle=suptitle,
        **profile_kwargs
    )


def facet_hist2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    group_by: str,
    top_k: Optional[int] = None,
    ncols: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = True,
    sharey: bool = True,
    suptitle: Optional[str] = None,
    **hist2d_kwargs
) -> Tuple[plt.Figure, np.ndarray, Dict[str, Any]]:
    """
    Create faceted 2D histogram with subplots for each group.
    """
    from .plots.histogram import draw_hist2d
    
    # For hist2d, disable colorbar per subplot (too cluttered)
    hist2d_kwargs.setdefault('colorbar', False)
    
    def plot_func(group_df, ax, **kwargs):
        _, _, stats = draw_hist2d(group_df, x, y, ax=ax, **kwargs)
        return stats
    
    return draw_facet(
        df, group_by, plot_func,
        top_k=top_k, ncols=ncols, figsize=figsize,
        sharex=sharex, sharey=sharey, suptitle=suptitle,
        **hist2d_kwargs
    )
