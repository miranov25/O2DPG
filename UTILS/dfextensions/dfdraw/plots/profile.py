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
from ._auto_title import build_auto_title, apply_auto_title, parse_auto_title_parts, resolve_auto_title


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


def _interval_sort_key(label):
    """
    Sort key for group labels: plain numbers, interval labels, or strings.
    
    Returns a tuple (priority, value) so numbers sort before strings:
    - (0, float) for numeric labels and interval left boundaries
    - (1, str) for non-numeric string labels
    
    Handles interval labels like '0.01-0.40' or '-0.39--0.00' by extracting
    the left boundary. NaN sorts to end.
    
    Parameters
    ----------
    label : str or any
        Group label to sort.
    
    Returns
    -------
    tuple
        (priority, sort_value) for consistent ordering.
    """
    if pd.isna(label):
        return (2, 0)
    s = str(label)
    # Try plain number first
    try:
        return (0, float(s))
    except ValueError:
        pass
    # Interval label: left boundary is before the separator dash.
    # The separator dash is the first '-' that follows a digit.
    # For "-0.39--0.00": skip leading '-' (negative sign), find next '-' after digit.
    try:
        for i in range(1, len(s)):
            if s[i] == '-' and s[i-1].isdigit():
                return (0, float(s[:i]))
    except (ValueError, IndexError):
        pass
    # String fallback — alphabetical
    return (1, s)


def _eval_weights(df: pd.DataFrame, weights: str) -> np.ndarray:
    """
    Evaluate weight column or expression.
    
    Supports both column names and computed expressions.
    Raises ValueError if evaluation fails (not silent).
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    weights : str
        Column name or expression (e.g., "(1+mP4**2)").
    
    Returns
    -------
    ndarray
        Weight values as float array.
    
    Raises
    ------
    ValueError
        If weight expression cannot be evaluated.
    """
    # Direct column access
    if weights in df.columns:
        return df[weights].values.astype(float)
    # Computed expression via df.eval()
    try:
        return df.eval(weights).values.astype(float)
    except Exception as e:
        raise ValueError(
            f"Cannot evaluate weight expression '{weights}': {e}. "
            "Weights must be a column name or a valid pandas expression."
        )


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
    # Phase 13.12.DF v1.2: Auto-title
    auto_title: Union[bool, str] = False,
    selection: Optional[Union[str, np.ndarray, callable]] = None,
    # Phase 13.18.DF: Robust statistics extension
    stat_fields=None,
    # Phase 13.25.DF (Phase A): Quantile rendering
    quantiles: Optional[List[float]] = None,
    central: Optional[str] = None,       # None → get_style_value("quantile.central_default", "mean")
    quantile_mode: str = "auto",
    # Phase 13.16.DF FIX1: suppress flags for vector dispatch
    _suppress_legend: bool = False,
    _suppress_title: bool = False,
    _suppress_layout: bool = False,
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
    auto_title : bool or str, default False
        Automatic title from plot parameters. Phase 13.12.DF v1.2.
        - False: no auto-title (default, or from style)
        - True / "all": "y vs x  group:group_by  weights:w\\nselection"
        - "expr": "y vs x" only
        - "expr+group": "y vs x  group:group_by"
        - "expr+sel": "y vs x\\nselection"
        Explicit title= always overrides auto_title.
    selection : str, array, or None
        Selection string (used for auto-title display only).
        Non-string selections are silently skipped in the title.
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
    # Phase 13.18.DF (AD-40): guard against boolean True (must be integer)
    if isinstance(group_by_quantiles, bool) and group_by_quantiles:
        raise ValueError(
            "group_by_quantiles must be an integer (number of quantile bins), "
            "not True. Example: group_by_quantiles=4"
        )
    
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
    # Phase 13.12.DF v1.2: auto_title from style if not set per-call
    auto_title = resolve_auto_title(auto_title)
    
    # Phase 13.25.DF: Resolve central= and validate quantile parameters
    if central is None:
        central = get_style_value("quantile.central_default", "mean")
    if central not in ('mean', 'median', 'both', 'none'):
        raise ValueError(
            f"central must be 'mean', 'median', 'both', or 'none', got {central!r}"
        )
    
    _resolved_quantile_mode = None
    _quantile_pair = None
    if quantiles is not None:
        # Validate and auto-detect mode
        if quantile_mode == 'auto':
            _resolved_quantile_mode = _detect_quantile_mode(quantiles)
        elif quantile_mode in ('error_bars', 'band'):
            _resolved_quantile_mode = quantile_mode
            # Still validate the list
            for q in quantiles:
                if q <= 0 or q >= 1:
                    raise ValueError(f"quantiles must be in (0, 1), got {q}")
        elif quantile_mode in ('discrete', 'nested_band'):
            raise NotImplementedError(
                f"Phase B: {quantile_mode} mode is not yet implemented."
            )
        else:
            raise ValueError(
                f"quantile_mode must be 'auto', 'error_bars', or 'band', "
                f"got {quantile_mode!r}"
            )
        
        # AD-51: central='none' + error_bars is invalid
        if central == 'none' and _resolved_quantile_mode == 'error_bars':
            raise ValueError(
                "central='none' is invalid with quantile_mode='error_bars' — "
                "error bars require a central line to ride on. "
                "Use central='mean' or central='median', or switch to "
                "quantile_mode='band' which supports central='none'."
            )
        
        # Extract the quantile pair (lower, upper) for computation
        qs = sorted(quantiles)
        if _resolved_quantile_mode == 'error_bars':
            _quantile_pair = (qs[0], qs[1])
        elif _resolved_quantile_mode == 'band':
            _quantile_pair = (qs[0], qs[2])  # skip 0.5 in the middle
        
        # AD-52: default coupling — rebind error="quantile" for error_bars mode
        # when user did not explicitly set error=
        if _resolved_quantile_mode == 'error_bars' and error == "sem":
            error = "quantile"
        
        # Weighted quantiles not supported in Phase A
        if weights is not None:
            raise NotImplementedError("weighted quantiles deferred to Phase B")
    
    # Validate error="quantile" requires quantiles=
    if error == "quantile" and quantiles is None:
        raise ValueError(
            "error='quantile' requires quantiles= parameter. "
            "Example: profile('y:x', quantiles=[0.16, 0.84], error='quantile')"
        )
    
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
    # Bugfix: support weight expressions (e.g., "(1+mP4**2)"), not just column names
    w_data = None
    if weights is not None:
        w_data = _eval_weights(df, weights)
    
    # Remove NaN (include weights in mask if present)
    mask = ~(np.isnan(x_data) | np.isnan(y_data))
    if w_data is not None:
        mask &= ~np.isnan(w_data)
        w_data = w_data[mask]
    x_data = x_data[mask]
    y_data = y_data[mask]
    df_filtered = df[mask].copy() if len(df) == len(mask) else df.copy()
    
    # Compute profile statistics
    stats_dict = _compute_profile_stats(x_data, y_data, stat_fields=stat_fields)
    
    # Phase 13.12.DF F3: Auto-bin float group_by column
    group_col = group_by
    if group_by is not None and group_by in df_filtered.columns:
        # float16 not supported by pd.cut/pd.qcut (pandas Index limitation)
        if df_filtered[group_by].dtype == np.float16:
            df_filtered[group_by] = df_filtered[group_by].astype(np.float32)
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
        # Phase A: compute standard profile (needed for mean line + SEM/STD bars)
        _error_for_compute = error if error != "quantile" else "sem"
        bin_centers, bin_means, bin_errors, bin_counts, profile_df = _compute_profile(
            x_data, y_data, bins, x_range, _error_for_compute, return_data=return_data,
            w_data=w_data  # Phase 13.12.DF v1.1
        )
        
        # Phase 13.12.DF F2: Apply min_entries filter for plotting
        plot_mask = bin_counts >= min_entries
        
        # Phase 13.25.DF: Compute per-bin quantiles if requested
        _q_lower = _q_upper = None
        if quantiles is not None and _quantile_pair is not None:
            _, _q_lower, _q_upper, _ = _compute_per_bin_quantiles(
                x_data, y_data, bins, x_range, _quantile_pair,
            )
            # Add to stats dict
            stats_dict['q_lower_per_bin'] = _q_lower
            stats_dict['q_upper_per_bin'] = _q_upper
        
        # Phase 13.25.DF: Compute per-bin median if needed
        _bin_medians = None
        if central in ('median', 'both'):
            _bin_medians = _compute_per_bin_median(x_data, y_data, bins, x_range)
        
        # Phase 13.25.DF: Determine central values for plotting
        if central == 'median':
            _central_values = _bin_medians
        else:
            _central_values = bin_means  # 'mean', 'both', 'none' all use mean as primary
        
        # Phase 13.25.DF: Render based on quantile_mode
        if _resolved_quantile_mode == 'error_bars' and error == "quantile":
            # Quantile-derived asymmetric error bars (zero visual channel cost)
            _render_quantile_error_bars(
                ax, bin_centers, _central_values, _q_lower, _q_upper,
                plot_mask, color, marker, markersize, linestyle, linewidth, label,
            )
        elif _resolved_quantile_mode == 'error_bars' and error in ("sem", "std"):
            # Both: quantile bars AND SEM/STD bars (user explicitly requested both)
            # Draw SEM/STD bars first (symmetric)
            ax.errorbar(
                bin_centers[plot_mask], _central_values[plot_mask],
                yerr=bin_errors[plot_mask],
                fmt=marker, color=color, markersize=markersize,
                capsize=capsize, linestyle=linestyle, linewidth=linewidth,
                label=label, **kwargs
            )
            # Overlay quantile bars (asymmetric, no marker to avoid double-plotting)
            c = _central_values[plot_mask]
            lower_delta = c - _q_lower[plot_mask]
            upper_delta = _q_upper[plot_mask] - c
            ax.errorbar(
                bin_centers[plot_mask], c,
                yerr=np.array([lower_delta, upper_delta]),
                fmt='none', color=color,
                capsize=get_style_value("quantile.error_bars.capsize", 3.0),
                linestyle='none',
            )
        elif _resolved_quantile_mode == 'band':
            # Band mode: render band first (behind), then central line on top
            if _q_lower is not None and _q_upper is not None:
                _render_quantile_band(
                    ax, bin_centers, _q_lower, _q_upper, plot_mask, color,
                )
            # Central line (unless central='none')
            if central != 'none':
                ax.errorbar(
                    bin_centers[plot_mask], _central_values[plot_mask],
                    yerr=bin_errors[plot_mask],
                    fmt=marker, color=color, markersize=markersize,
                    capsize=capsize, linestyle=linestyle, linewidth=linewidth,
                    label=label, **kwargs
                )
        else:
            # No quantiles — standard profile rendering (existing behavior)
            ax.errorbar(
                bin_centers[plot_mask], bin_means[plot_mask],
                yerr=bin_errors[plot_mask],
                fmt=marker, color=color, markersize=markersize,
                capsize=capsize, linestyle=linestyle, linewidth=linewidth,
                label=label, **kwargs
            )
        
        # Phase 13.25.DF: Render second central line for central='both'
        if central == 'both' and _bin_medians is not None:
            ax.plot(
                bin_centers[plot_mask], _bin_medians[plot_mask],
                color=color, linestyle='--', linewidth=linewidth,
                marker=marker, markersize=markersize * 0.7,
                label=f"{label or ''} (median)".strip(),
            )
        
        # Phase 13.12.DF F1: Add profile data to stats
        if return_data and profile_df is not None:
            stats_dict['profile_data'] = profile_df
    
    # Labels
    ax.set_xlabel(xlabel or x_name)
    ax.set_ylabel(ylabel or f"<{y_name}>")
    
    # Title: explicit > auto > none
    if not _suppress_title:
        if title:
            ax.set_title(title)
        elif auto_title:
            parts = parse_auto_title_parts(auto_title)
            td = build_auto_title(x_name, y_name, group_by=group_by,
                                  selection=selection, weights=weights, parts=parts)
            apply_auto_title(ax, td)
    
    # Statistics box
    if stats is True or (stats is None and get_style_value("stats.show", False)):
        _add_stats_box(ax, stats_dict, stats if isinstance(stats, list) else None)
    elif isinstance(stats, list):
        _add_stats_box(ax, stats_dict, stats)
    
    # Legend for grouped
    if not _suppress_legend:
        if group_col is not None:
            ax.legend(loc=get_style_value("legend.loc", "best"))
    
    if not _suppress_layout:
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


def _compute_profile_stats(x_data: np.ndarray, y_data: np.ndarray,
                           stat_fields=None) -> Dict[str, Any]:
    """Compute overall profile statistics with robust extensions (Phase 13.18.DF)."""
    from .histogram import _parse_stat_fields, _compute_robust_stats_1d
    n = len(x_data)
    result = {
        "n": n,
        "mean_x": float(np.mean(x_data)) if n > 0 else np.nan,
        "mean_y": float(np.mean(y_data)) if n > 0 else np.nan,
        "std_x": float(np.std(x_data)) if n > 0 else np.nan,
        "std_y": float(np.std(y_data)) if n > 0 else np.nan,
        "corr": float(np.corrcoef(x_data, y_data)[0, 1]) if n > 1 else np.nan,
    }
    # Phase 13.18.DF: per-axis robust + optional stats (same code path as hist2d)
    _groups = _parse_stat_fields(stat_fields)
    result.update(_compute_robust_stats_1d(x_data, _groups, suffix='_x'))
    result.update(_compute_robust_stats_1d(y_data, _groups, suffix='_y'))
    return result


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
    # Phase 13.14.DF: Use _interval_sort_key for correct negative interval sorting
    if sort_groups:
        groups = sorted(groups, key=_interval_sort_key)
    
    # Top-K filtering
    if top_k is not None and len(groups) > top_k:
        counts = df[group_by].value_counts()
        top_groups = counts.head(top_k).index.tolist()
        # Preserve sort order
        if sort_groups:
            top_groups = sorted(top_groups, key=_interval_sort_key)
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
        # Bugfix: support weight expressions, not just column names
        w_data = None
        if weights is not None:
            w_data = _eval_weights(group_df, weights)
        
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


# =============================================================================
# Phase 13.25.DF (Phase A): Quantile rendering helpers
# =============================================================================

def _detect_quantile_mode(quantiles: list) -> str:
    """
    Auto-detect quantile rendering mode from the shape of the quantiles list.
    
    Phase A supports error_bars and band only; other shapes raise
    NotImplementedError (Phase B) or ValueError.
    
    Parameters
    ----------
    quantiles : list of float
        Quantile fractions in (0, 1).
    
    Returns
    -------
    str
        'error_bars' or 'band' (Phase A).
    
    Raises
    ------
    ValueError
        If quantiles is empty, single-valued, or contains out-of-range values.
    NotImplementedError
        If quantiles shape requires Phase B modes (discrete-line, nested-band).
    """
    if not quantiles:
        raise ValueError("quantiles must be non-empty list of fractions in (0, 1)")
    
    # Range check
    for q in quantiles:
        if q <= 0 or q >= 1:
            raise ValueError(
                f"quantiles must be in (0, 1), got {q}. "
                f"Use fractions like [0.16, 0.84], not percentages."
            )
    
    qs = sorted(quantiles)
    n = len(qs)
    
    if n == 1:
        raise ValueError(
            f"quantiles requires at least a symmetric pair (e.g., [0.16, 0.84]), "
            f"got single value [{qs[0]}]"
        )
    
    # Check if symmetric pair (no 0.5)
    if n == 2:
        is_symmetric = abs(qs[0] + qs[1] - 1.0) < 1e-9
        has_05 = any(abs(q - 0.5) < 1e-9 for q in qs)
        if is_symmetric and not has_05:
            return 'error_bars'
    
    # Check if symmetric triple with 0.5
    if n == 3:
        has_05 = abs(qs[1] - 0.5) < 1e-9
        is_symmetric = abs(qs[0] + qs[2] - 1.0) < 1e-9
        if has_05 and is_symmetric:
            return 'band'
    
    # Multi-pair symmetric (Phase B: nested-band)
    # Check if all non-0.5 entries form symmetric pairs
    non_05 = [q for q in qs if abs(q - 0.5) > 1e-9]
    if len(non_05) >= 4:
        pairs_symmetric = all(
            abs(non_05[i] + non_05[-(i+1)] - 1.0) < 1e-9
            for i in range(len(non_05) // 2)
        )
        if pairs_symmetric:
            raise NotImplementedError(
                "Phase B: nested-band mode is not yet implemented. "
                "Use a single symmetric pair (Phase A error_bars, e.g., [0.16, 0.84]) "
                "or a symmetric triple including 0.5 (Phase A band, e.g., [0.16, 0.5, 0.84]) "
                "until Phase B ships. See brainstorm §8.1."
            )
    
    # Asymmetric / arbitrary → Phase B discrete-line
    raise NotImplementedError(
        "Phase B: discrete-line mode is not yet implemented. "
        "Use a single symmetric pair (Phase A error_bars, e.g., [0.16, 0.84]) "
        "or a symmetric triple including 0.5 (Phase A band, e.g., [0.16, 0.5, 0.84]) "
        "until Phase B ships. See brainstorm §8.1."
    )


def _compute_per_bin_quantiles(
    x_data: np.ndarray,
    y_data: np.ndarray,
    bins: int,
    x_range,
    quantile_pair: tuple,
    w_data=None,
) -> tuple:
    """
    Compute per-bin quantiles of y in bins of x.
    
    Single source of truth for quantile computation per AD-48.
    Reuses the same binning algorithm as _compute_profile().
    
    Parameters
    ----------
    x_data, y_data : arrays
    bins : int
    x_range : tuple (min, max) or None
    quantile_pair : tuple of 2 floats, e.g., (0.16, 0.84)
    w_data : array, optional — weighted quantiles NOT supported in Phase A.
    
    Returns
    -------
    (bin_centers, q_lower, q_upper, bin_counts)
        Arrays of length `bins`.
    """
    if w_data is not None:
        raise NotImplementedError("weighted quantiles deferred to Phase B")
    
    if x_range is None:
        x_range = (np.nanmin(x_data), np.nanmax(x_data))
    
    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.clip(np.digitize(x_data, bin_edges) - 1, 0, bins - 1)
    
    q_lower_frac, q_upper_frac = quantile_pair
    q_lower = np.full(bins, np.nan)
    q_upper = np.full(bins, np.nan)
    bin_counts = np.zeros(bins, dtype=int)
    
    for i in range(bins):
        mask = bin_indices == i
        y_bin = y_data[mask]
        y_bin = y_bin[~np.isnan(y_bin)]
        n = len(y_bin)
        bin_counts[i] = n
        if n > 0:
            q_lower[i] = float(np.nanpercentile(y_bin, q_lower_frac * 100))
            q_upper[i] = float(np.nanpercentile(y_bin, q_upper_frac * 100))
    
    return bin_centers, q_lower, q_upper, bin_counts


def _compute_per_bin_median(
    x_data: np.ndarray,
    y_data: np.ndarray,
    bins: int,
    x_range,
) -> np.ndarray:
    """
    Compute per-bin median of y in bins of x.
    
    Used when central='median' or central='both' — the median line
    is computed even if 0.5 is not in the quantiles list.
    
    Returns
    -------
    bin_medians : array of length `bins`
    """
    if x_range is None:
        x_range = (np.nanmin(x_data), np.nanmax(x_data))
    
    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_indices = np.clip(np.digitize(x_data, bin_edges) - 1, 0, bins - 1)
    
    bin_medians = np.full(bins, np.nan)
    for i in range(bins):
        mask = bin_indices == i
        y_bin = y_data[mask]
        y_bin = y_bin[~np.isnan(y_bin)]
        if len(y_bin) > 0:
            bin_medians[i] = float(np.nanmedian(y_bin))
    
    return bin_medians


def _render_quantile_error_bars(
    ax, bin_centers, central_values, q_lower, q_upper,
    plot_mask, color, marker, markersize, linestyle, linewidth, label,
    capsize=None, **kwargs
):
    """
    Render quantile-derived asymmetric error bars on the central line.
    
    Uses asymmetric yerr=[[lower_deltas], [upper_deltas]] per R3.
    Capsize read from quantile.error_bars.capsize style key (AD-53).
    """
    if capsize is None:
        capsize = get_style_value("quantile.error_bars.capsize", 3.0)
    
    # Asymmetric error bars: yerr = [[lower_deltas], [upper_deltas]]
    # lower_delta = central - q_lower (positive value = bar extends downward)
    # upper_delta = q_upper - central (positive value = bar extends upward)
    c = central_values[plot_mask]
    lower_delta = c - q_lower[plot_mask]
    upper_delta = q_upper[plot_mask] - c
    yerr = np.array([lower_delta, upper_delta])
    
    ax.errorbar(
        bin_centers[plot_mask], c, yerr=yerr,
        fmt=marker, color=color, markersize=markersize,
        capsize=capsize, linestyle=linestyle, linewidth=linewidth,
        label=label, **kwargs
    )


def _render_quantile_band(
    ax, bin_centers, q_lower, q_upper, plot_mask, color,
):
    """
    Render quantile band via fill_between.
    
    Alpha and hatch read from quantile.band.alpha and quantile.band.hatch
    style keys (AD-53). Band color matches the central line's color.
    """
    alpha = get_style_value("quantile.band.alpha", 0.25)
    hatch = get_style_value("quantile.band.hatch", None)
    
    fill_kwargs = dict(alpha=alpha, color=color)
    if hatch is not None:
        fill_kwargs['hatch'] = hatch
    
    ax.fill_between(
        bin_centers[plot_mask],
        q_lower[plot_mask],
        q_upper[plot_mask],
        **fill_kwargs
    )
