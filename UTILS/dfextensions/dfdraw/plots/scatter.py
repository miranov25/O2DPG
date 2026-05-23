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
import warnings
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Any, Dict, List, Optional, Tuple, Union

from ..style import get_style_value
from ..stats import format_stats_box
# Phase 13.28.DF: Robust data handling
from ._data_sanitize import sanitize_for_plot
# Phase 13.30.DF: Class-2 column-reference parameter validation
from ._validation import validate_column_references


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
    # Phase 13.38.DF: scatter error bars (column name or df.eval() expression).
    # When either is set, render via ax.errorbar() instead of ax.scatter().
    # NaN/inf policy in _eval_error(): raise on 100% non-finite, warn at >50%,
    # silent zeroing at ≤50%. Locked by §9.SE.6.
    xerr: Optional[str] = None,
    yerr: Optional[str] = None,
    # Phase 13.39.DF: time-axis formatting (pre-conversion approach, CP1-4 auto-detect).
    time_format: Optional[str] = None,
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
    # Phase 13.39.DF CP1-4: detect datetime64 column BEFORE astype(float)
    _x_is_datetime = (
        isinstance(x, str) and x in df.columns
        and np.issubdtype(df[x].dtype, np.datetime64)
    )
    if isinstance(x, str):
        x_name = x
        if _x_is_datetime:
            x_data = df[x].values   # keep datetime64
        else:
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

    # Phase 13.39.DF (CP1-4): time_format pre-conversion with dtype auto-detect.
    # Convert x_data to matplotlib date numbers BEFORE ax.scatter/ax.errorbar
    # so plotted x positions are date floats, not raw Unix ints.
    if time_format is not None:
        import matplotlib.dates as mdates
        _x_arr = np.asarray(x_data)
        if np.issubdtype(_x_arr.dtype, np.datetime64):
            x_data = mdates.date2num(_x_arr)
        else:
            x_data = mdates.date2num(
                pd.to_datetime(_x_arr, unit='s').to_pydatetime()
            )
    
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
    
    # Phase 13.30.DF: Validate Class-2 column-reference parameters.
    # Catches BUG_ADF_GroupBy_Expression_Materialization (silent fallthrough below).
    from ..drawer import DFDraw as _DFDraw
    validate_column_references(
        df, locals(),
        names=_DFDraw._SCATTER_COLUMN_REFERENCES,
        context="scatter",
    )

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

        # ============================================================== #
        # Phase 13.38.DF: scatter error bars + per-point marker resolution #
        # ============================================================== #
        xerr_arr, xerr_nanfrac = _eval_error(df_filtered, xerr, mask, 'xerr')
        yerr_arr, yerr_nanfrac = _eval_error(df_filtered, yerr, mask, 'yerr')
        if xerr is not None:
            stats_dict['xerr_nanfrac'] = xerr_nanfrac
        if yerr is not None:
            stats_dict['yerr_nanfrac'] = yerr_nanfrac

        resolved_marker, marker_is_scalar = _resolve_marker_per_point(
            df_filtered, marker, mask
        )

        # Branch A: error bars present → ax.errorbar() dispatch
        if xerr_arr is not None or yerr_arr is not None:
            # fmt determines point marker; default to first marker if per-point
            fmt = resolved_marker if marker_is_scalar else (
                resolved_marker[0] if len(resolved_marker) > 0 else 'o'
            )
            # Resolve color for errorbar (scalar only — c=array goes via
            # post-render path; eval-color + xerr/yerr is documented composition)
            resolved_color = c if isinstance(c, str) else None
            scatter = ax.errorbar(
                x_data, y_data,
                xerr=xerr_arr, yerr=yerr_arr,
                fmt=fmt if isinstance(fmt, str) else 'o',
                color=resolved_color,
                alpha=alpha,
                elinewidth=get_style_value("scatter.error_elinewidth", 1.0),
                capsize=get_style_value("scatter.error_capsize", 2),
                ecolor=get_style_value("scatter.error_ecolor", None),
                **kwargs
            )
            # No colorbar in errorbar branch (errorbar doesn't return ScalarMappable)
        # Branch B: per-point marker → np.unique loop with _nolegend_
        elif not marker_is_scalar:
            scatter = None
            for m in np.unique(resolved_marker):
                idx = resolved_marker == m
                if not np.any(idx):
                    continue
                # Subset color array if continuous (np.ndarray)
                if isinstance(c, np.ndarray):
                    c_sub = c[idx]
                elif c is not None:
                    c_sub = c  # fixed color string
                else:
                    c_sub = None
                # Subset size if it's an array
                s_sub = s[idx] if isinstance(s, np.ndarray) else s
                _scat = ax.scatter(
                    x_data[idx], y_data[idx],
                    c=c_sub, s=s_sub, alpha=alpha,
                    edgecolors=edgecolors, linewidths=linewidths,
                    cmap=cmap_used if not is_categorical else None,
                    marker=m,
                    label='_nolegend_',  # §9.ECM.8: prevent duplicate legend entries
                    **kwargs
                )
                # Keep the last collection for colorbar attachment
                scatter = _scat
            # Colorbar (continuous expression color + per-point markers compose)
            if scatter is not None and isinstance(c, np.ndarray) and not is_categorical and colorbar:
                cbar = plt.colorbar(scatter, ax=ax)
                if clabel:
                    cbar.set_label(clabel)
                elif isinstance(color, str) and color not in df.columns:
                    # Expression color — label with the expression itself
                    cbar.set_label(color)
        # Branch C: existing scalar-marker ax.scatter (dispatch invariance §9.SE.5)
        else:
            scatter = ax.scatter(
                x_data, y_data,
                c=c, s=s, alpha=alpha,
                edgecolors=edgecolors, linewidths=linewidths,
                cmap=cmap_used if not is_categorical else None,
                marker=marker if isinstance(marker, str) else 'o',
                **kwargs
            )
            # Colorbar for continuous color
            if c is not None and not is_categorical and colorbar and not isinstance(c, str):
                cbar = plt.colorbar(scatter, ax=ax)
                if clabel:
                    cbar.set_label(clabel)
                elif isinstance(color, str) and color in df.columns:
                    cbar.set_label(color)
                elif isinstance(color, str) and color not in df.columns:
                    # Phase 13.38.DF: expression color label
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

    # Phase 13.39.DF: apply time_format formatter AFTER render
    if time_format is not None:
        import matplotlib.dates as mdates
        if time_format == "auto":
            ax.xaxis.set_major_formatter(
                mdates.AutoDateFormatter(mdates.AutoDateLocator()))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
        fig.autofmt_xdate()

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

    Dispatch order (Phase 13.38.DF CP0-1):
        None → array → column-name → fixed-color (to_rgba) → df.eval() → terminal

    Column-name check precedes the matplotlib `to_rgba()` heuristic to preserve
    the backward-compat property that column names win over named-color
    collisions. E.g. a column named 'b' (also matplotlib blue) renders as a
    colormap from the column values, NOT as fixed blue. Locked by §9.ECM.6.

    The new df.eval() branch (added Phase 13.38.DF) supports expressions like
    color='abs(tgl)' — evaluated against df, mapped via colormap.

    Returns
    -------
    tuple
        (color_array, colormap_name, is_categorical)
    """
    if color is None:
        return None, None, False

    # (1) Array provided directly (no string ambiguity)
    if isinstance(color, np.ndarray):
        return color, cmap or "viridis", False

    # (2) Column name — wins over named-color collisions (BACKWARD-COMPAT LOCK §9.ECM.6)
    if isinstance(color, str) and color in df.columns:
        color_data = df[color].values
        if len(mask) == len(color_data):
            color_data = color_data[mask]
        # Phase 13.41.DF FIX1 (Sonnet54 P2 carry-forward): broaden categorical
        # detection to cover pandas StringDtype + ArrowStringDtype + any
        # ExtensionDtype that doesn't convert to float. The original check
        # (object dtype + CategoricalDtype) missed pd.StringDtype which is
        # common in newer pandas (Py3.12 default) — caused Linux CI fail in
        # test_vector.py::test_vector_draw_kwarg_surface_enumeration since
        # Phase 13.38.
        if color_data.dtype == object or isinstance(color_data.dtype, pd.CategoricalDtype):
            return None, None, True
        # Try numeric conversion; if it fails (e.g. StringDtype 'A','B','C'),
        # fall back to categorical mode for safe rendering with discrete cmap.
        try:
            return color_data.astype(float), cmap or "viridis", False
        except (ValueError, TypeError):
            return None, None, True

    # (3) Fixed color string via matplotlib heuristic
    if isinstance(color, str):
        try:
            import matplotlib.colors as mc
            mc.to_rgba(color)
            return color, None, False
        except (ValueError, TypeError):
            pass

        # (4) Phase 13.38.DF: df.eval() expression — last string fallback
        try:
            color_data = df.eval(color).values
        except Exception as e:
            raise ValueError(
                f"color={color!r} is not a column name, valid color string, "
                f"or valid df.eval() expression. Error: {e}"
            ) from e
        if len(mask) == len(color_data):
            color_data = color_data[mask]
        if color_data.dtype == object or isinstance(color_data.dtype, pd.CategoricalDtype):
            return None, None, True
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


# ====================================================================== #
# Phase 13.38.DF — error bar evaluation + per-point marker resolution      #
# ====================================================================== #

def _eval_error(
    df: pd.DataFrame,
    expr: Optional[str],
    mask: np.ndarray,
    name: str = 'yerr',
) -> Tuple[Optional[np.ndarray], float]:
    """Evaluate xerr/yerr column name or df.eval() expression.

    NaN/inf policy (Phase 13.38.DF CP1-4, three-tier):
      - nanfrac == 1.0  → ValueError (programming error; all errors invalid)
      - nanfrac >  0.5  → UserWarning (likely data quality issue)
      - nanfrac >  0    → silent zeroing (per-point safety; matplotlib needs finite)
      - nanfrac == 0    → no-op

    Parameters
    ----------
    df : pd.DataFrame
        Source DataFrame.
    expr : str or None
        Column name or df.eval() expression. Returns (None, 0.0) for None.
    mask : np.ndarray
        Boolean mask applied to evaluated values (Phase 13.28 sanitize mask).
    name : str
        Parameter name ('xerr' or 'yerr') for error messages.

    Returns
    -------
    (arr, nanfrac) : tuple of (np.ndarray or None, float)
        arr has non-finite values zeroed to 0.0. nanfrac is the fraction of
        non-finite values BEFORE zeroing — write into stats dict for
        observability.

    Raises
    ------
    ValueError
        If expr is not None, not a column name, and not a valid df.eval()
        expression. Also raised if ALL values are non-finite.
    """
    if expr is None:
        return None, 0.0

    if expr in df.columns:
        arr = df[expr].values
    else:
        try:
            arr = df.eval(expr).values
        except Exception as e:
            raise ValueError(
                f"{name}={expr!r} is neither a column name nor a "
                f"valid df.eval() expression. Error: {e}"
            ) from e

    arr = arr[mask].astype(float)

    # Compute nanfrac BEFORE zeroing (for stats + policy enforcement)
    n_total = len(arr)
    if n_total == 0:
        return arr, 0.0
    finite_mask = np.isfinite(arr)
    n_nonfinite = int((~finite_mask).sum())
    nanfrac = n_nonfinite / n_total

    # CP1-4 three-tier policy
    if nanfrac == 1.0:
        raise ValueError(
            f"{name}={expr!r}: ALL {n_total} values are non-finite "
            f"(NaN/inf). This is a programming error — error bars cannot "
            f"be rendered with no finite values."
        )
    if nanfrac > 0.5:
        warnings.warn(
            f"{name}={expr!r}: {nanfrac:.1%} of values ({n_nonfinite}/"
            f"{n_total}) are non-finite. Zeroed to 0.0 for rendering "
            f"(error bars will be invisible for those points).",
            UserWarning,
            stacklevel=2,
        )

    # Sanitize: NaN/inf → 0.0 (silent at nanfrac ≤ 0.5)
    arr = np.where(finite_mask, arr, 0.0)
    return arr, nanfrac


def _resolve_marker_per_point(
    df: pd.DataFrame,
    marker: Optional[Union[str, List[str]]],
    mask: np.ndarray,
) -> Tuple[Any, bool]:
    """Resolve marker to per-point array if it's a column name or boolean expression.

    Phase 13.38.DF: extends marker= to support boolean df.eval() expressions
    (True → 's', False → 'o'), in addition to existing column-name and
    fixed-string support.

    Returns
    -------
    (resolved, is_scalar) : tuple
        - If is_scalar=True: `resolved` is a fixed string (or None) for the
          scalar fast path.
        - If is_scalar=False: `resolved` is a numpy array of per-point markers
          for the per-point rendering loop.
    """
    if marker is None or not isinstance(marker, str):
        return marker, True

    # (1) Column name → per-point marker array
    if marker in df.columns:
        markers = df[marker].values
        if len(mask) == len(markers):
            markers = markers[mask]
        return markers, False

    # (2) df.eval() boolean expression → two-marker encoding
    try:
        result = df.eval(marker).values
        if len(mask) == len(result):
            result = result[mask]
        if result.dtype == bool or result.dtype == np.bool_:
            markers = np.where(result, 's', 'o')
            return markers, False
    except Exception:
        pass

    # (3) Fixed marker string → scalar fast path
    return marker, True


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


# ====================================================================== #
# Phase 13.39.DF — draw_scatter3d (3D point cloud via z:y:x expression)   #
# ====================================================================== #

def draw_scatter3d(
    df: pd.DataFrame,
    z_expr: str,
    y_expr: str,
    x_expr: str,
    ax=None,
    selection: Optional[Union[str, np.ndarray, callable]] = None,
    sample: Optional[int] = None,
    color: Optional[Union[str, np.ndarray]] = None,
    size: Optional[Union[str, float, np.ndarray]] = None,
    cmap: str = 'viridis',
    alpha: Optional[float] = None,
    nan_policy: str = 'filter',
    elev: Optional[float] = None,
    azim: Optional[float] = None,
    auto_title: Union[bool, str] = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
    same: bool = False,
    **kwargs,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Draw 3D scatter plot (z:y:x expression).

    Phase 13.39.DF — invoked when DFDraw.draw() detects type='scatter3d'
    AND colon_count == 2. Reuses Phase 13.38 _process_color() and
    _process_size() unchanged.
    """
    # Function-local import (mpl_toolkits.mplot3d is standard matplotlib —
    # N6: not a soft-dep; placement is for readability only)
    from mpl_toolkits.mplot3d import Axes3D

    # same=True dimensionality guard (CP2-2 / §9.SC3D.8)
    if same and ax is not None and not isinstance(ax, Axes3D):
        raise ValueError(
            "type='scatter3d' with same=True requires existing axes to be "
            "3D projection (Axes3D). Got 2D axes — cannot overlay 3D on 2D."
        )

    # Evaluate expressions
    def _eval(expr):
        if expr in df.columns:
            return df[expr].values
        return df.eval(expr).values

    x_data = _eval(x_expr)
    y_data = _eval(y_expr)
    z_data = _eval(z_expr)

    # Joint NaN/inf mask across all 3 axes
    mask = (np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(z_data))
    n_filtered = int((~mask).sum())
    x_data = x_data[mask].astype(float)
    y_data = y_data[mask].astype(float)
    z_data = z_data[mask].astype(float)

    # Reuse Phase 13.38 helpers — pass the mask that recovers original df indices
    # For Phase 13.39, the simpler contract: mask aligns with df rows; we
    # construct a boolean array matching df length.
    df_mask = np.zeros(len(df), dtype=bool)
    df_mask[np.where(mask)[0]] = True   # not strictly needed but documents intent
    color_values, color_cmap, is_categorical = _process_color(
        df, color, cmap, mask, len(x_data),
    )
    size_values = _process_size(df, size, get_style_value("scatter.size", 50),
                                mask, len(x_data))

    # Create axes if not provided
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()

    # Render
    scatter = ax.scatter(
        x_data, y_data, z_data,
        c=color_values if color_values is not None else None,
        cmap=color_cmap if (color_values is not None and not is_categorical) else None,
        s=size_values,
        alpha=alpha,
        **kwargs,
    )

    if elev is not None or azim is not None:
        ax.view_init(
            elev=elev if elev is not None else ax.elev,
            azim=azim if azim is not None else ax.azim,
        )

    ax.set_xlabel(xlabel or x_expr)
    ax.set_ylabel(ylabel or y_expr)
    ax.set_zlabel(zlabel or z_expr)
    if title:
        ax.set_title(title)
    elif auto_title:
        ax.set_title(f"{z_expr} vs ({y_expr}, {x_expr})")

    # Stats dict (CP1-3: SC3D.6 locks all 3 means)
    stats: Dict[str, Any] = {
        'n': int(mask.sum()),
        'n_filtered': n_filtered,
        'mean_x': float(np.mean(x_data)) if len(x_data) > 0 else float('nan'),
        'mean_y': float(np.mean(y_data)) if len(y_data) > 0 else float('nan'),
        'mean_z': float(np.mean(z_data)) if len(z_data) > 0 else float('nan'),
        'std_x': float(np.std(x_data)) if len(x_data) > 0 else float('nan'),
        'std_y': float(np.std(y_data)) if len(y_data) > 0 else float('nan'),
        'std_z': float(np.std(z_data)) if len(z_data) > 0 else float('nan'),
    }
    return fig, ax, stats
