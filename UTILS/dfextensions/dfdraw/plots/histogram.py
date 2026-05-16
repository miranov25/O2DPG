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
from ._auto_title import build_auto_title, apply_auto_title, parse_auto_title_parts, resolve_auto_title
# Phase 13.28.DF: Robust data handling
from ._data_sanitize import sanitize_for_plot
from ._autorange import compute_autorange, VALID_STRATEGIES
# Phase 13.30.DF: Class-2 column-reference parameter validation
from ._validation import validate_column_references


# =============================================================================
# Phase 13.18.DF: Robust statistics helpers
# =============================================================================

# Valid group names for stat_fields parameter
_VALID_STAT_GROUPS = frozenset({'quantiles', 'shape', 'all'})


def _parse_stat_fields(stat_fields):
    """
    Parse stat_fields parameter into a set of group names to compute.
    
    Always includes 'base' and 'robust'. Optional groups added on request.
    Raises ValueError on unrecognized input.
    
    Parameters
    ----------
    stat_fields : None, str, or list of str
    
    Returns
    -------
    set of str — group names to compute
    """
    groups = {'base', 'robust'}  # always-on
    if stat_fields is None:
        return groups
    if isinstance(stat_fields, str):
        stat_fields = [stat_fields]
    for name in stat_fields:
        if name == 'all':
            groups.update({'quantiles', 'shape'})
        elif name in _VALID_STAT_GROUPS:
            groups.add(name)
        else:
            raise ValueError(
                f"stat_fields: unrecognized group '{name}'. "
                f"Valid values: {sorted(_VALID_STAT_GROUPS)}, or a list of them."
            )
    return groups


def _compute_robust_stats_1d(data, groups, suffix=''):
    """
    Compute robust statistics for a 1D array.
    
    Parameters
    ----------
    data : np.ndarray — cleaned (no NaN) data
    groups : set of str — which groups to compute
    suffix : str — key suffix ('_x', '_y', or '' for 1D hist)
    
    Returns
    -------
    dict — stats keyed as e.g. 'median', 'mad', 'mad_sigma' (1D)
           or 'median_x', 'mad_x', 'mad_sigma_x' (2D with suffix)
    
    Note
    ----
    Phase 13.6.G.DF added compute_stats() in stats.py with median/MAD
    support for stats-box display. This helper computes for the returned
    dict (always-on, decoupled from display). If stats.py::compute_stats()
    is refactored to a shared path, this can delegate to it.
    """
    result = {}
    n = len(data)
    
    # --- robust (always) ---
    if n > 0:
        median = float(np.nanmedian(data))
        mad = float(np.nanmedian(np.abs(data - median)))
        mad_sigma = mad * 1.4826
    else:
        median = np.nan
        mad = np.nan
        mad_sigma = np.nan
    
    result[f'median{suffix}'] = median
    result[f'mad{suffix}'] = mad
    result[f'mad_sigma{suffix}'] = mad_sigma
    
    # --- quantiles (on request) ---
    if 'quantiles' in groups:
        if n > 0:
            pcts = np.nanpercentile(data, [5, 15.87, 25, 50, 75, 84.13, 95])
            result[f'q05{suffix}'] = float(pcts[0])
            result[f'q16{suffix}'] = float(pcts[1])
            result[f'q50{suffix}'] = float(pcts[3])
            result[f'q84{suffix}'] = float(pcts[5])
            result[f'q95{suffix}'] = float(pcts[6])
            result[f'iqr{suffix}'] = float(pcts[4] - pcts[2])  # Q75 - Q25
        else:
            for key in ('q05', 'q16', 'q50', 'q84', 'q95', 'iqr'):
                result[f'{key}{suffix}'] = np.nan
    
    # --- shape (on request) ---
    if 'shape' in groups:
        if n >= 10:
            s = float(np.nanstd(data))
            if s > 0:
                m = data - np.nanmean(data)
                skewness = float(np.nanmean(m**3) / s**3)
                kurtosis = float(np.nanmean(m**4) / s**4 - 3.0)
                robust_to_std = s / mad_sigma if mad_sigma > 0 else np.nan
            else:
                skewness = np.nan
                kurtosis = np.nan
                robust_to_std = np.nan
        else:
            skewness = np.nan
            kurtosis = np.nan
            robust_to_std = np.nan
        
        result[f'skewness{suffix}'] = skewness
        result[f'kurtosis{suffix}'] = kurtosis
        result[f'robust_to_std{suffix}'] = robust_to_std
    
    return result


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
    # Phase 13.12.DF v1.2: Auto-title
    auto_title: Union[bool, str] = False,
    selection: Optional[Union[str, np.ndarray, callable]] = None,
    # Phase 13.18.DF: Robust statistics extension
    stat_fields: Optional[Union[str, List[str]]] = None,
    # Phase 13.28.DF: NaN/inf filter policy (AD-70)
    nan_policy: str = "filter",
    # Phase 13.27.DF Commit 2 FIX1 (§7b): per-row weights as column name or
    # df.eval-able expression. When set, evaluated to an array, sanitized
    # in lockstep with x_data (same NaN/inf mask), and passed to ax.hist
    # via the weights= kwarg. Precedence with norm="probability": explicit
    # per-row weights win and are additionally scaled by 1/n_clean.
    weights: Optional[str] = None,
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
    # Phase 13.12.DF v1.2: auto_title from style if not set per-call
    auto_title = resolve_auto_title(auto_title)
    
    # Create figure if needed
    if ax is None:
        figsize = get_style_value("figure.figsize", (8, 6))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Get data
    # BUG_dfdraw_20260505: cast to float — boolean expressions (==, !=, >, <, &, |, ~)
    # produce np.bool_ columns; np.histogram cannot subtract boolean edges.
    # Profile, hist2d, hexbin all already cast to float here; hist was the outlier.
    if isinstance(x, str):
        x_name = x
        x_data = df[x].values.astype(float)
    else:
        x_name = "x"
        x_data = np.asarray(x, dtype=float)

    # Phase 13.27.DF Commit 2 FIX1 (§7b): evaluate per-row weights column / expression.
    # Mirrors profile.py's _eval_weights pattern. Sanitized jointly with x_data below.
    w_data = None
    if weights is not None:
        if isinstance(weights, str):
            if weights in df.columns:
                w_data = df[weights].values.astype(float)
            else:
                try:
                    w_data = df.eval(weights).values.astype(float)
                except Exception as e:
                    raise ValueError(
                        f"Cannot evaluate weight expression '{weights}': {e}. "
                        "Weights must be a column name or a valid pandas expression."
                    )
        else:
            # Pre-existing call sites may pass an array directly (e.g. via _draw_vector
            # plumbing). Accept and pass through.
            w_data = np.asarray(weights, dtype=float)

    # Phase 13.28.DF: NaN/inf sanitization (AD-69, AD-70).
    # Use sanitize_for_plot for counters + policy enforcement (raise/warn);
    # apply joint mask manually to align weights with x_data (mirrors profile.py).
    _, _, _sanitize_stats = sanitize_for_plot(
        x_data, y_data=None, nan_policy=nan_policy, column_names=(x_name, "")
    )
    if w_data is not None:
        # Joint mask: x finite AND w finite. Realigns w_data to surviving rows.
        # Phase 13.27.DF Commit 2 FIX1 (§7b): group_by + column-name weights is
        # NotImplementedError — the grouped path would need per-group w slicing.
        if group_by is not None and group_by in df.columns:
            raise NotImplementedError(
                "weights= (column-name / expression) combined with group_by is "
                "not yet supported. Use group_by alone, or apply your weighting "
                "filter via selection= and call hist without weights="
            )
        _mask = np.isfinite(x_data) & np.isfinite(w_data)
        x_data = x_data[_mask]
        w_data = w_data[_mask]
    else:
        # No weights: keep pre-FIX1 behavior — sanitize_for_plot already
        # filtered x_data via its returned x_clean.
        x_data, _, _ = sanitize_for_plot(
            x_data, y_data=None, nan_policy=nan_policy, column_names=(x_name, "")
        )

    # Phase 13.28.DF: Resolve autorange (AD-73, AD-77)
    from ._autorange import resolve_range_1d
    if len(x_data) > 0:
        _used_range, _autorange_strategy = resolve_range_1d(
            range,
            x_data,
            style_strategy=get_style_value("autorange.strategy", "hybrid"),
            style_k_robust=get_style_value("autorange.k_robust", 4.0),
            style_k_outlier=get_style_value("autorange.k_outlier", 1.5),
            style_percentile=get_style_value("autorange.percentile", (1.0, 99.0)),
        )
    else:
        _used_range, _autorange_strategy = (0.0, 1.0), (range if isinstance(range, str) else "explicit" if range is not None else "hybrid")

    # Statistics dict
    stats_dict = {
        "n": len(x_data),
        "mean": float(np.mean(x_data)) if len(x_data) > 0 else np.nan,
        "std": float(np.std(x_data)) if len(x_data) > 0 else np.nan,
        "min": float(np.min(x_data)) if len(x_data) > 0 else np.nan,
        "max": float(np.max(x_data)) if len(x_data) > 0 else np.nan,
    }
    # Phase 13.28.DF: Sanitize counters (AD-71) + autorange diagnostics (AD-77)
    stats_dict.update(_sanitize_stats)
    stats_dict["autorange_used"] = _used_range
    stats_dict["autorange_strategy"] = _autorange_strategy
    
    # Phase 13.18.DF: robust + optional stats groups
    _groups = _parse_stat_fields(stat_fields)
    stats_dict.update(_compute_robust_stats_1d(x_data, _groups))
    
    # Phase 13.16.DF FIX1: strip private _suppress_* kwargs before they
    # reach matplotlib (injected by _draw_vector for layout/legend/title control).
    _suppress_legend = kwargs.pop('_suppress_legend', False)
    _suppress_title = kwargs.pop('_suppress_title', False)
    _suppress_layout = kwargs.pop('_suppress_layout', False)
    
    # Normalization
    # Normalization + weights resolution
    # Phase 13.27.DF Commit 2 FIX1 (§7b): use _hist_weights as the matplotlib
    # weights= array. When the user passed `weights=` (now in w_data after
    # sanitize), use it. With norm="probability", scale by 1/n_clean.
    density = False
    _hist_weights = w_data  # may be None
    if norm == "density":
        density = True
    elif norm == "probability":
        if _hist_weights is not None:
            # Explicit user weights × probability normalization: per-row weight
            # multiplied by 1/n. Matches the "probability per row" semantic.
            _hist_weights = (_hist_weights / len(x_data)) if len(x_data) > 0 else _hist_weights
        else:
            # Pre-FIX1 behavior: synthesize uniform 1/n weights.
            _hist_weights = np.ones_like(x_data) / len(x_data) if len(x_data) > 0 else None

    # Phase 13.30.DF: Validate Class-2 column-reference parameters.
    # Catches BUG_ADF_GroupBy_Expression_Materialization (silent fallthrough below).
    from ..drawer import DFDraw as _DFDraw
    validate_column_references(
        df, locals(),
        names=_DFDraw._HIST_COLUMN_REFERENCES,
        context="hist",
    )

    # Group-by handling
    if group_by is not None and group_by in df.columns:
        # w_data + group_by raises above; here _hist_weights is either None or
        # the probability-synthesized 1/n array (pre-FIX1 behavior).
        _draw_hist_grouped(
            df, x, ax, group_by, top_k, stacked,
            bins=bins, range=_used_range, density=density, weights=_hist_weights,
            alpha=alpha, histtype=histtype, edgecolor=edgecolor,
            linewidth=linewidth, **kwargs
        )
        stats_dict["grouped"] = True
    else:
        # Single histogram
        ax.hist(
            x_data, bins=bins, range=_used_range, density=density, weights=_hist_weights,
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
    
    if not _suppress_title:
        if title:
            ax.set_title(title)
        elif auto_title:
            parts = parse_auto_title_parts(auto_title)
            td = build_auto_title(x_name, y=None, group_by=group_by,
                                  selection=selection, parts=parts)
            apply_auto_title(ax, td)
    
    # Statistics box
    if stats is True or (stats is None and get_style_value("stats.show", False)):
        _add_stats_box(ax, stats_dict, stats if isinstance(stats, list) else None)
    elif isinstance(stats, list):
        _add_stats_box(ax, stats_dict, stats)
    
    # Legend for grouped
    if not _suppress_legend:
        if group_by is not None:
            ax.legend(loc=get_style_value("legend.loc", "best"))
    
    if not _suppress_layout:
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
        # BUG_dfdraw_20260505: cast to float for boolean expressions
        data_list = [df[df[group_by] == g][x].dropna().values.astype(float) for g in groups]
        ax.hist(data_list, label=[str(g) for g in groups], color=colors,
                stacked=True, **hist_kwargs)
    else:
        # Overlaid histograms
        for i, group in enumerate(groups):
            # BUG_dfdraw_20260505: cast to float for boolean expressions
            group_data = df[df[group_by] == group][x].dropna().values.astype(float)
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


def draw_hist2d(
    df: pd.DataFrame,
    x: Union[str, pd.Series, np.ndarray],
    y: Union[str, pd.Series, np.ndarray],
    ax: Optional[plt.Axes] = None,
    bins: Optional[Union[int, List[int], Tuple[int, int]]] = None,
    range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    norm: Optional[str] = None,
    stats: Optional[Union[bool, List[str]]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: Optional[str] = None,
    colorbar: bool = True,
    clabel: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    # Phase 13.12.DF v1.2: Auto-title
    auto_title: Union[bool, str] = False,
    selection: Optional[Union[str, np.ndarray, callable]] = None,
    # Phase 13.18.DF: Robust statistics extension
    stat_fields: Optional[Union[str, List[str]]] = None,
    # Phase 13.28.DF: NaN/inf filter policy (AD-70)
    nan_policy: str = "filter",
    **kwargs
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Draw 2D histogram (density plot).
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    x : str, Series, or array
        X-axis column.
    y : str, Series, or array
        Y-axis column.
    ax : Axes, optional
        Existing axes to plot on.
    bins : int, list, or tuple, optional
        Number of bins. Can be:
        - int: same bins for x and y
        - [nx, ny]: different bins for x and y
    range : tuple, optional
        ((xmin, xmax), (ymin, ymax)) range.
    norm : str, optional
        Normalization: "count" (default), "density", "log".
    stats : bool or list, optional
        Show statistics box.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    cmap : str, optional
        Colormap name.
    colorbar : bool, default True
        Show colorbar.
    clabel : str, optional
        Colorbar label.
    vmin, vmax : float, optional
        Color scale limits.
    **kwargs
        Additional arguments passed to plt.hist2d().
    
    Returns
    -------
    tuple
        (fig, ax, stats_dict)
    """
    # Get style defaults
    if bins is None:
        bins = get_style_value("hist.bins", 50)
    if cmap is None:
        cmap = "viridis"
    # Phase 13.12.DF v1.2: auto_title from style if not set per-call
    auto_title = resolve_auto_title(auto_title)
    
    # Handle bins parameter
    if isinstance(bins, int):
        bins = [bins, bins]
    elif isinstance(bins, (list, tuple)) and len(bins) == 2:
        bins = list(bins)
    
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
    
    # Phase 13.28.DF: NaN/inf sanitization (AD-69, AD-70)
    x_data, y_data, _sanitize_stats = sanitize_for_plot(
        x_data, y_data, nan_policy=nan_policy, column_names=(x_name, y_name)
    )

    # Phase 13.28.DF: Resolve autorange per-axis (AD-73, AD-74, AD-77)
    from ._autorange import resolve_range_2d
    if len(x_data) > 0:
        _used_range, _autorange_strategy = resolve_range_2d(
            range,
            x_data, y_data,
            style_strategy=get_style_value("autorange.strategy", "hybrid"),
            style_k_robust=get_style_value("autorange.k_robust", 4.0),
            style_k_outlier=get_style_value("autorange.k_outlier", 1.5),
            style_percentile=get_style_value("autorange.percentile", (1.0, 99.0)),
        )
    else:
        _used_range = ((0.0, 1.0), (0.0, 1.0))
        _autorange_strategy = (range if isinstance(range, str)
                                else "explicit" if range is not None else "hybrid")

    # Statistics
    stats_dict = _compute_hist2d_stats(x_data, y_data, stat_fields=stat_fields)
    # Phase 13.28.DF: Sanitize counters (AD-71) + autorange diagnostics (AD-77)
    stats_dict.update(_sanitize_stats)
    stats_dict["autorange_used"] = _used_range
    stats_dict["autorange_strategy"] = _autorange_strategy
    
    # Strip private _suppress_* kwargs (injected by draw_batch/vector dispatch)
    kwargs.pop('_suppress_legend', None)
    kwargs.pop('_suppress_title', None)
    _suppress_layout = kwargs.pop('_suppress_layout', False)
    
    # Normalization
    norm_obj = None
    if norm == "log":
        from matplotlib.colors import LogNorm
        norm_obj = LogNorm(vmin=vmin if vmin and vmin > 0 else 1, vmax=vmax)
    elif norm == "density":
        # Will use density=True in hist2d
        pass
    
    # Draw 2D histogram
    h, xedges, yedges, im = ax.hist2d(
        x_data, y_data,
        bins=bins,
        range=_used_range,
        density=(norm == "density"),
        cmap=cmap,
        vmin=vmin if norm != "log" else None,
        vmax=vmax if norm != "log" else None,
        norm=norm_obj,
        **kwargs
    )
    
    # Colorbar
    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        if clabel:
            cbar.set_label(clabel)
        elif norm == "density":
            cbar.set_label("Density")
        else:
            cbar.set_label("Count")
    
    # Labels
    ax.set_xlabel(xlabel or x_name)
    ax.set_ylabel(ylabel or y_name)
    
    # Title: explicit > auto > none
    if title:
        ax.set_title(title)
    elif auto_title:
        parts = parse_auto_title_parts(auto_title)
        td = build_auto_title(x_name, y_name, selection=selection, parts=parts)
        apply_auto_title(ax, td)
    
    # Statistics box
    if stats is True or (stats is None and get_style_value("stats.show", False)):
        _add_stats_box_2d(ax, stats_dict, stats if isinstance(stats, list) else None)
    elif isinstance(stats, list):
        _add_stats_box_2d(ax, stats_dict, stats)
    
    if not _suppress_layout:
        plt.tight_layout()
    return fig, ax, stats_dict


def _compute_hist2d_stats(x_data: np.ndarray, y_data: np.ndarray,
                          stat_fields=None) -> Dict[str, Any]:
    """Compute 2D histogram statistics with robust extensions (Phase 13.18.DF)."""
    n = len(x_data)
    result = {
        "n": n,
        "mean_x": float(np.mean(x_data)) if n > 0 else np.nan,
        "mean_y": float(np.mean(y_data)) if n > 0 else np.nan,
        "std_x": float(np.std(x_data)) if n > 0 else np.nan,
        "std_y": float(np.std(y_data)) if n > 0 else np.nan,
        "corr": float(np.corrcoef(x_data, y_data)[0, 1]) if n > 1 else np.nan,
    }
    # Phase 13.18.DF: per-axis robust + optional stats
    _groups = _parse_stat_fields(stat_fields)
    result.update(_compute_robust_stats_1d(x_data, _groups, suffix='_x'))
    result.update(_compute_robust_stats_1d(y_data, _groups, suffix='_y'))
    return result


def _add_stats_box_2d(
    ax: plt.Axes,
    stats: Dict[str, Any],
    fields: Optional[List[str]] = None
) -> None:
    """Add statistics box to 2D histogram."""
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


def draw_hexbin(
    df: pd.DataFrame,
    x: Union[str, pd.Series, np.ndarray],
    y: Union[str, pd.Series, np.ndarray],
    ax: Optional[plt.Axes] = None,
    gridsize: int = 50,
    extent: Optional[Tuple[float, float, float, float]] = None,
    norm: Optional[str] = None,
    stats: Optional[Union[bool, List[str]]] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cmap: Optional[str] = None,
    colorbar: bool = True,
    clabel: Optional[str] = None,
    mincnt: Optional[int] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    # Phase 13.12.DF v1.2: Auto-title
    auto_title: Union[bool, str] = False,
    selection: Optional[Union[str, np.ndarray, callable]] = None,
    # Phase 13.18.DF: Robust statistics (inherits from _compute_hist2d_stats)
    stat_fields: Optional[Union[str, List[str]]] = None,
    # Phase 13.28.DF: NaN/inf filter policy (AD-70)
    nan_policy: str = "filter",
    **kwargs
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Draw hexbin plot (2D density with hexagonal bins).
    
    Better than hist2d for large datasets - hexagons tile more efficiently
    and avoid alignment artifacts.
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    x : str, Series, or array
        X-axis column.
    y : str, Series, or array
        Y-axis column.
    ax : Axes, optional
        Existing axes to plot on.
    gridsize : int, default 50
        Number of hexagons in x-direction.
    extent : tuple, optional
        (xmin, xmax, ymin, ymax) extent.
    norm : str, optional
        Normalization: None (count), "log".
    stats : bool or list, optional
        Show statistics box.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    cmap : str, optional
        Colormap name.
    colorbar : bool, default True
        Show colorbar.
    clabel : str, optional
        Colorbar label.
    mincnt : int, optional
        Minimum count to display a hexagon.
    vmin, vmax : float, optional
        Color scale limits.
    **kwargs
        Additional arguments passed to plt.hexbin().
    
    Returns
    -------
    tuple
        (fig, ax, stats_dict)
    """
    # Get style defaults
    if cmap is None:
        cmap = "viridis"
    # Phase 13.12.DF v1.2: auto_title from style if not set per-call
    auto_title = resolve_auto_title(auto_title)
    
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
    
    # Phase 13.28.DF: NaN/inf sanitization (AD-69, AD-70)
    # Note: hexbin uses 'extent' parameter (matplotlib-controlled), not 'range',
    # so autorange wiring does not apply here. Only sanitize counters added.
    x_data, y_data, _sanitize_stats = sanitize_for_plot(
        x_data, y_data, nan_policy=nan_policy, column_names=(x_name, y_name)
    )

    # Statistics
    stats_dict = _compute_hist2d_stats(x_data, y_data, stat_fields=stat_fields)
    # Phase 13.28.DF: Sanitize counters (AD-71)
    stats_dict.update(_sanitize_stats)
    stats_dict["autorange_used"] = extent if extent is not None else (
        (float(x_data.min()), float(x_data.max()),
         float(y_data.min()), float(y_data.max())) if len(x_data) > 0 else (0.0, 1.0, 0.0, 1.0)
    )
    stats_dict["autorange_strategy"] = "explicit" if extent is not None else "minmax"
    
    # Strip private _suppress_* kwargs (injected by draw_batch/vector dispatch)
    kwargs.pop('_suppress_legend', None)
    kwargs.pop('_suppress_title', None)
    _suppress_layout = kwargs.pop('_suppress_layout', False)
    
    # Normalization
    bins_arg = None
    if norm == "log":
        bins_arg = "log"
    
    # Draw hexbin
    hb = ax.hexbin(
        x_data, y_data,
        gridsize=gridsize,
        extent=extent,
        cmap=cmap,
        mincnt=mincnt,
        vmin=vmin,
        vmax=vmax,
        bins=bins_arg,
        **kwargs
    )
    
    # Colorbar
    if colorbar:
        cbar = plt.colorbar(hb, ax=ax)
        if clabel:
            cbar.set_label(clabel)
        elif norm == "log":
            cbar.set_label("log10(Count)")
        else:
            cbar.set_label("Count")
    
    # Labels
    ax.set_xlabel(xlabel or x_name)
    ax.set_ylabel(ylabel or y_name)
    
    # Title: explicit > auto > none
    if title:
        ax.set_title(title)
    elif auto_title:
        parts = parse_auto_title_parts(auto_title)
        td = build_auto_title(x_name, y_name, selection=selection, parts=parts)
        apply_auto_title(ax, td)
    
    # Statistics box
    if stats is True or (stats is None and get_style_value("stats.show", False)):
        _add_stats_box_2d(ax, stats_dict, stats if isinstance(stats, list) else None)
    elif isinstance(stats, list):
        _add_stats_box_2d(ax, stats_dict, stats)
    
    if not _suppress_layout:
        plt.tight_layout()
    return fig, ax, stats_dict
