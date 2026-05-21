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
# Phase 13.28.DF: Robust data handling
from ._data_sanitize import sanitize_for_plot
from ._autorange import compute_autorange, resolve_range_1d, VALID_STRATEGIES
# Phase 13.30.DF: Class-2 column-reference parameter validation
from ._validation import validate_column_references


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
    # Phase 13.37.DF (BUG-016): pd.Interval objects from histogram grouped path.
    # _draw_hist_grouped() receives raw pd.Interval values from pd.cut() in the
    # routing block (post-Phase 13.35). str(Interval(10.0, 12.0)) is "(10.0, 12.0]"
    # — the leading '(' defeats both float() parse and the digit-dash detector
    # below, falling through to (1, s) lexicographic sort. This guard short-
    # circuits with the numeric .left boundary BEFORE the string-based logic.
    # Profile path (string labels from _format_interval_label) is unaffected
    # (strings don't have a .left attribute).
    if hasattr(label, 'left'):
        return (0, float(label.left))
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
    error: Optional[str] = None,   # Phase 13.25.DF FIX1: None → resolve per context
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
    # Phase 13.26.DF (Phase B): Channel-aware quantile rendering
    quantile_style: Optional[str] = None,
    # Phase 13.28.DF: NaN/inf filter policy (AD-70)
    nan_policy: str = "filter",
    # Phase 13.16.DF FIX1: suppress flags for vector dispatch
    _suppress_legend: bool = False,
    _suppress_title: bool = False,
    _suppress_layout: bool = False,
    # Phase 13.37.DF: per-group linestyle cycling mode flag. When True and
    # user did not pass linestyle= explicitly, cycle through
    # channels.cycles.linestyle per group. User-explicit linestyle= wins
    # (Phase 13.36 sentinel pattern via _ud_user_linestyle capture above).
    linestyle_cycle: bool = False,
    # Phase 13.39.DF: time-axis formatting (pre-conversion approach, CP1-4 auto-detect).
    time_format: Optional[str] = None,
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

    # Phase 13.30.DF: Validate Class-2 column-reference parameters.
    # Catches BUG_ADF_GroupBy_Expression_Materialization: caller passed a
    # computed expression (e.g. "row%3") where a column name is required.
    # Import tuple from drawer at call time to avoid circular import.
    from ..drawer import DFDraw as _DFDraw
    validate_column_references(
        df, locals(),
        names=_DFDraw._PROFILE_COLUMN_REFERENCES,
        context="profile",
    )

    # Phase 13.36.DF: capture user-explicit marker/markersize BEFORE the style
    # fill-in below (which sets marker='o', markersize=6 when user passed None).
    # We need to distinguish "user explicitly passed marker='s'" from "user
    # passed nothing; style filled in default 'o'". The grouped path uses
    # these to decide between uniform-override vs per-group cycle.
    # color is NOT filled in by style (matplotlib handles None directly).
    _ud_user_marker = marker          # None or user's value
    _ud_user_markersize = markersize  # None or user's value
    _ud_user_color = color            # None or user's value
    # Phase 13.37.DF: capture user-explicit linestyle BEFORE the style fill-in
    # at line 340 (which sets linestyle = "-" when None). Same Edit 17 pattern.
    # _ud_user_linestyle = None means "use cycle if linestyle_cycle=True, else
    # style default". Non-None means "user explicitly passed; uniform override
    # wins over cycle".
    _ud_user_linestyle = linestyle    # None or user's value

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
    
    # Phase 13.25.DF FIX1 (I-1): resolve error=None sentinel.
    # None means "user did not explicitly set error=".
    # Mirror R2's central=None pattern (Claude49 P1-1 + Claude37 P1-2 convergent).
    _error_was_explicit = (error is not None)
    if error is None:
        error = "sem"  # backward-compat default

    # Phase 13.25.DF: Resolve central= and validate quantile parameters
    if central is None:
        central = get_style_value("quantile.central_default", "mean")
    if central not in ('mean', 'median', 'both', 'none'):
        raise ValueError(
            f"central must be 'mean', 'median', 'both', or 'none', got {central!r}"
        )
    
    _resolved_quantile_mode = None
    _quantile_pair = None
    _quantile_list = None
    if quantiles is not None:
        # Validate and auto-detect mode
        if quantile_mode == 'auto':
            _resolved_quantile_mode = _detect_quantile_mode(quantiles)
        elif quantile_mode in ('error_bars', 'band', 'discrete', 'nested_band'):
            _resolved_quantile_mode = quantile_mode
            # Still validate the list
            for q in quantiles:
                if q <= 0 or q >= 1:
                    raise ValueError(f"quantiles must be in (0, 1), got {q}")
        else:
            raise ValueError(
                f"quantile_mode must be 'auto', 'error_bars', 'band', "
                f"'discrete', or 'nested_band', got {quantile_mode!r}"
            )
        
        # AD-51: central='none' + error_bars is invalid
        if central == 'none' and _resolved_quantile_mode == 'error_bars':
            raise ValueError(
                "central='none' is invalid with quantile_mode='error_bars' — "
                "error bars require a central line to ride on. "
                "Use central='mean' or central='median', or switch to "
                "quantile_mode='band' which supports central='none'."
            )
        
        # Extract quantile values for computation
        qs = sorted(quantiles)
        _quantile_pair = None
        _quantile_list = None
        if _resolved_quantile_mode == 'error_bars':
            _quantile_pair = (qs[0], qs[1])
        elif _resolved_quantile_mode == 'band':
            _quantile_pair = (qs[0], qs[2])  # skip 0.5 in the middle
        elif _resolved_quantile_mode == 'discrete':
            _quantile_list = qs  # all quantiles, one line each
        elif _resolved_quantile_mode == 'nested_band':
            _quantile_list = qs  # all quantiles, paired into nested bands
        
        # AD-52 FIX1: rebind error only when user did NOT explicitly set it.
        # None (signature default) → rebind to "quantile" for error_bars mode.
        # Explicit "sem" → keep "sem" → both SEM + quantile bars rendered.
        # Explicit "none" → keep "none" → quantile bars only (I-2 dispatch).
        if _resolved_quantile_mode == 'error_bars' and not _error_was_explicit:
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
    # Phase 13.39.DF: detect datetime64 column BEFORE astype(float).
    # If x is a column name and df[x] is datetime64[ns], we must
    # convert to date numbers via mdates.date2num() directly, NOT via
    # pd.to_datetime(..., unit='s') (which would re-interpret the int64
    # nanosecond representation as seconds → year out of range crash).
    _x_is_datetime = (
        isinstance(x, str) and x in df.columns
        and np.issubdtype(df[x].dtype, np.datetime64)
    )
    if isinstance(x, str):
        x_name = x
        if _x_is_datetime:
            # Keep datetime64 — astype(float) would convert to ns count
            x_data = df[x].values
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
    
    # Phase 13.12.DF v1.1: Get weights if specified
    # Bugfix: support weight expressions (e.g., "(1+mP4**2)"), not just column names
    w_data = None
    if weights is not None:
        w_data = _eval_weights(df, weights)
    
    # Phase 13.28.DF: NaN/inf sanitization (AD-69, AD-70).
    # Sanitize counters use x/y only (public contract); apply joint mask
    # including weights for actual filtering (matches pre-Phase-13.28 semantics).
    _, _, _sanitize_stats = sanitize_for_plot(
        x_data, y_data, nan_policy=nan_policy, column_names=(x_name, y_name)
    )
    mask = np.isfinite(x_data) & np.isfinite(y_data)
    if w_data is not None:
        mask &= np.isfinite(w_data)
        w_data = w_data[mask]
    x_data = x_data[mask]
    y_data = y_data[mask]
    df_filtered = df[mask].copy() if len(df) == len(mask) else df.copy()

    # Phase 13.39.DF (CP1-4): time_format pre-conversion with dtype auto-detect.
    # Convert x_data to matplotlib date numbers BEFORE binning so bin centers
    # come out as date numbers automatically (same axis units throughout).
    # Auto-detect: datetime64 dtype → direct convert; else assume Unix epoch
    # seconds (the standard ROOT/legacy convention).
    if time_format is not None:
        import matplotlib.dates as mdates
        _x_arr = np.asarray(x_data)
        if np.issubdtype(_x_arr.dtype, np.datetime64):
            x_data = mdates.date2num(_x_arr)
        else:
            x_data = mdates.date2num(
                pd.to_datetime(_x_arr, unit='s').to_pydatetime()
            )

    # Phase 13.28.DF: Resolve x_range autorange (AD-73, AD-77)
    if len(x_data) > 0:
        _used_xrange, _autorange_strategy = resolve_range_1d(
            x_range,
            x_data,
            style_strategy=get_style_value("autorange.strategy", "hybrid"),
            style_k_robust=get_style_value("autorange.k_robust", 4.0),
            style_k_outlier=get_style_value("autorange.k_outlier", 1.5),
            style_percentile=get_style_value("autorange.percentile", (1.0, 99.0)),
        )
    else:
        _used_xrange = (0.0, 1.0)
        _autorange_strategy = (x_range if isinstance(x_range, str)
                                else "explicit" if x_range is not None else "hybrid")

    # Compute profile statistics
    stats_dict = _compute_profile_stats(x_data, y_data, stat_fields=stat_fields)
    # Phase 13.28.DF: Sanitize counters (AD-71) + autorange diagnostics (AD-77)
    stats_dict.update(_sanitize_stats)
    stats_dict["autorange_used"] = _used_xrange
    stats_dict["autorange_strategy"] = _autorange_strategy
    
    # Phase 13.12.DF F3: Auto-bin float group_by column
    group_col = group_by
    if group_by is not None and group_by in df_filtered.columns:
        _col = df_filtered[group_by]
        # Phase 13.37.DF (BUG-015): float column + no bins + high cardinality
        # → memory hang/OOM. Mirrors Phase 13.35 hist() BUG-012 guard. Fires
        # before the float16 upcast so users get a clear error instead of an
        # unhelpful crash. Categorical/int columns skip the guard (any value
        # of nunique() is acceptable). Limitation: expression-string group_by
        # like "abs(tgl)" is NOT in df.columns → guard is skipped (same gap as
        # Phase 13.35; deferred to df.eval() path phase — see §8 Out of scope).
        if (_col.dtype.kind == 'f'
                and group_by_bins is None
                and group_by_quantiles is None
                and _col.nunique() > 20):
            raise ValueError(
                f"group_by='{group_by}' is a float column with "
                f"{_col.nunique()} unique values. "
                f"Add group_by_bins=N or group_by_quantiles=N to bin it. "
                f"Example: group_by_bins=5 or group_by_quantiles=5."
            )
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
        # Phase 13.32.DF Sub-fix 2: forward Phase 13.25 resolved quantile state
        # (resolved earlier in this function: _resolved_quantile_mode,
        # _quantile_pair, _quantile_list). The grouped path renders bands or
        # discrete lines per group color; see _draw_profile_grouped docstring.
        profile_data_list, _per_group_stats = _draw_profile_grouped(
            df_filtered, x, y, ax, group_col, top_k,
            # Phase 13.36.DF: forward user style overrides as named params
            # (BUG-013 fix). The user's marker/markersize/color values live in
            # draw_profile()'s LOCAL VARIABLES — but the style fill-in above
            # at lines 314-317 replaces None with style defaults BEFORE this
            # call. Use the _ud_user_* captures (None when user passed nothing)
            # to correctly distinguish explicit-pass from style-fill.
            _user_marker=_ud_user_marker,
            _user_markersize=_ud_user_markersize,
            _user_color=_ud_user_color,
            # Phase 13.37.DF: user linestyle sentinel + cycle mode flag.
            _user_linestyle=_ud_user_linestyle,
            linestyle_cycle=linestyle_cycle,
            bins=bins, x_range=_used_xrange, error=error,
            # Phase 13.36.DF: marker= and markersize= REMOVED from this call.
            # They are now forwarded via _user_marker / _user_markersize above.
            # The previous pops at lines 984-985 are also deleted (nothing to pop).
            capsize=capsize,
            linestyle=linestyle, linewidth=linewidth,
            min_entries=min_entries,
            sort_groups=sort_groups,
            return_data=return_data,
            weights=weights,  # Phase 13.12.DF v1.1
            # Phase 13.32.DF Sub-fix 2: quantile rendering state
            quantiles=quantiles,
            quantile_mode=_resolved_quantile_mode,
            central=central,
            quantile_pair=_quantile_pair,
            quantile_list=_quantile_list,
            quantile_style=quantile_style,
            **kwargs
        )
        stats_dict["grouped"] = True
        # Phase 13.32.DF Sub-fix 2: surface quantile metadata + per-group stats
        # in the returned stats_dict per v1.2 §3.2 schema.
        if quantiles is not None:
            stats_dict["quantile_mode"] = _resolved_quantile_mode
            stats_dict["quantile_pair"] = _quantile_pair
            stats_dict["quantile_list"] = _quantile_list
            if _per_group_stats is not None:
                stats_dict["per_group"] = _per_group_stats
        
        # Phase 13.12.DF F1: Combine profile data from all groups
        if return_data and profile_data_list:
            stats_dict['profile_data'] = pd.concat(profile_data_list, ignore_index=True)
    else:
        # Single profile
        # Phase A: compute standard profile (needed for mean line + SEM/STD bars)
        _error_for_compute = error if error != "quantile" else "sem"
        bin_centers, bin_means, bin_errors, bin_counts, profile_df = _compute_profile(
            x_data, y_data, bins, _used_xrange, _error_for_compute, return_data=return_data,
            w_data=w_data  # Phase 13.12.DF v1.1
        )
        
        # Phase 13.12.DF F2: Apply min_entries filter for plotting
        plot_mask = bin_counts >= min_entries
        
        # Phase 13.25.DF: Compute per-bin quantiles if requested
        _q_lower = _q_upper = None
        _q_all = None  # dict {q_value: per_bin_array} for discrete mode
        if quantiles is not None and _quantile_pair is not None:
            _, _q_lower, _q_upper, _ = _compute_per_bin_quantiles(
                x_data, y_data, bins, _used_xrange, _quantile_pair,
            )
            # Add to stats dict
            stats_dict['q_lower_per_bin'] = _q_lower
            stats_dict['q_upper_per_bin'] = _q_upper
        elif quantiles is not None and _quantile_list is not None:
            # Discrete mode OR nested_band: compute per-bin value for EACH quantile
            _q_all = _compute_per_bin_all_quantiles(
                x_data, y_data, bins, _used_xrange, _quantile_list,
            )
            # Add to stats dict
            stats_dict['quantiles_per_bin'] = _q_all
        
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
        elif _resolved_quantile_mode == 'error_bars' and error == "none":
            # FIX1 I-2 (Claude37 P1-1): error="none" + quantile error_bars
            # → render quantile bars only on the central line (no SEM/STD)
            _render_quantile_error_bars(
                ax, bin_centers, _central_values, _q_lower, _q_upper,
                plot_mask, color, marker, markersize, linestyle, linewidth, label,
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
        elif _resolved_quantile_mode == 'nested_band' and _q_all is not None:
            # Phase 13.26.DF Phase B (Option A, AD-57): nested alpha-stacked
            # bands. >=2 symmetric pairs (with or without 0.5). Central line
            # rendered when central != 'none'. Zero-cost mode (no channel
            # consumed). Max 3 nested bands (silent truncation, outermost 3).
            if central != 'none':
                ax.errorbar(
                    bin_centers[plot_mask], _central_values[plot_mask],
                    yerr=bin_errors[plot_mask],
                    fmt=marker, color=color, markersize=markersize,
                    capsize=capsize, linestyle=linestyle, linewidth=linewidth,
                    label=label, **kwargs
                )
            _render_quantile_nested_band(
                ax, bin_centers, _q_all, plot_mask, color,
            )
        elif _resolved_quantile_mode == 'discrete' and _q_all is not None:
            # Discrete mode: one line per quantile value — the general case.
            # Central line first (unless central='none')
            if central != 'none':
                ax.errorbar(
                    bin_centers[plot_mask], _central_values[plot_mask],
                    yerr=bin_errors[plot_mask],
                    fmt=marker, color=color, markersize=markersize,
                    capsize=capsize, linestyle=linestyle, linewidth=linewidth,
                    label=label, **kwargs
                )
            # Phase 13.26.DF Phase B: channel-aware rendering. quantile_style
            # picked by Algorithm A (default 'linestyle' per AD-56). The
            # cycle is read from style keys (channels.cycles.<channel>),
            # replacing the FIX2 hardcoded local _ls_cycle. Per v1.2 §11.3:
            # for quantile_style='linestyle', skip solid (index 0) so the
            # central line's solid stays distinct (FIX2 invariant).
            _qs_kind = (quantile_style
                        or get_style_value("channels.default.quantiles")
                        or 'linestyle')
            if _qs_kind == 'linestyle':
                _full_cycle = get_style_value(
                    "channels.cycles.linestyle", ["-", "--", "-.", ":"])
                # FIX2 invariant: reserve solid for central line. Slice [1:]
                # so quantile lines never coincide with central style.
                _qs_cycle = list(_full_cycle[1:]) if len(_full_cycle) > 1 else list(_full_cycle)
                if not _qs_cycle:  # degenerate case
                    _qs_cycle = ["--"]
            elif _qs_kind == 'marker':
                _qs_cycle = list(get_style_value(
                    "channels.cycles.marker",
                    ["o", "s", "^", "D", "v", "<", ">", "p"]))
            elif _qs_kind == 'color':
                # Color cycle managed by matplotlib's color_cycle; we delegate
                # by passing color=None per quantile line and let the axes
                # cycle pick the next colour.
                _qs_cycle = None
            else:
                # Unknown channel — fallback to linestyle behaviour.
                _qs_cycle = ['--', '-.', ':']

            for j, (q_val, q_per_bin) in enumerate(_q_all.items()):
                q_label = f'q={q_val:.0%}' if abs(q_val - 0.5) > 1e-9 else 'median'
                if _qs_kind == 'linestyle':
                    q_ls = _qs_cycle[j % len(_qs_cycle)]
                    line, = ax.plot(
                        bin_centers[plot_mask], q_per_bin[plot_mask],
                        color=color, linestyle=q_ls,
                        linewidth=linewidth * 0.8, label=q_label,
                    )
                elif _qs_kind == 'marker':
                    q_mk = _qs_cycle[j % len(_qs_cycle)]
                    line, = ax.plot(
                        bin_centers[plot_mask], q_per_bin[plot_mask],
                        color=color, marker=q_mk,
                        linestyle='-', linewidth=linewidth * 0.8,
                        markersize=markersize * 0.7, label=q_label,
                    )
                else:  # color or fallback — let axes color cycle pick
                    line, = ax.plot(
                        bin_centers[plot_mask], q_per_bin[plot_mask],
                        linestyle='-', linewidth=linewidth * 0.8,
                        label=q_label,
                    )
                # On-line annotation (FIX2 behaviour preserved as default UX
                # for quantile_style='linestyle'; suppressed for other channels
                # since marker/color don't need disambiguating labels).
                if _qs_kind == 'linestyle':
                    valid = plot_mask & ~np.isnan(q_per_bin)
                    n_valid = np.sum(valid)
                    if n_valid > 2:
                        idx = np.where(valid)[0][int(0.4 * n_valid)]
                        ax.annotate(
                            f'{q_val:.0%}',
                            xy=(bin_centers[idx], q_per_bin[idx]),
                            fontsize=7, fontweight='bold',
                            color=line.get_color(),
                            backgroundcolor='white',
                            ha='center', va='bottom',
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
    
    # Legend for grouped or discrete quantiles
    if not _suppress_legend:
        if group_col is not None:
            ax.legend(loc=get_style_value("legend.loc", "best"))
        elif _resolved_quantile_mode == 'discrete' and quantiles is not None:
            ax.legend(loc=get_style_value("legend.loc", "best"),
                      fontsize=get_style_value("legend.fontsize", 9))
    
    if not _suppress_layout:
        plt.tight_layout()

    # Phase 13.39.DF: apply time_format formatter AFTER render. The
    # date-number conversion already happened upstream (line ~485 area);
    # this just installs the DateFormatter on the x-axis.
    if time_format is not None:
        import matplotlib.dates as mdates
        if time_format == "auto":
            ax.xaxis.set_major_formatter(
                mdates.AutoDateFormatter(mdates.AutoDateLocator()))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
        fig.autofmt_xdate()

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
    # Phase 13.32.DF Sub-fix 2: per-group quantile rendering parameters
    quantiles: Optional[list] = None,
    quantile_mode: Optional[str] = None,
    central: Optional[str] = None,
    quantile_pair: Optional[tuple] = None,
    quantile_list: Optional[list] = None,
    quantile_style: Optional[str] = None,
    # Phase 13.36.DF: user style overrides (BUG-013 fix).
    # None = "user did not pass" (matches matplotlib default-color semantics).
    # Non-None = apply uniformly to all groups, override the auto-cycle.
    # See module-level docstring for design rationale.
    _user_marker: Optional[str] = None,
    _user_markersize: Optional[float] = None,
    _user_color: Optional[str] = None,
    # Phase 13.37.DF: user linestyle sentinel + cycle mode flag.
    # _user_linestyle is captured BEFORE style fill-in at profile.py:340
    # (same Phase 13.36 Edit 17 pattern). linestyle_cycle=True with user
    # passing nothing → cycle through channels.cycles.linestyle. User
    # explicit always wins (CP1-3 lock).
    _user_linestyle: Optional[str] = None,
    linestyle_cycle: bool = False,
    **profile_kwargs
) -> tuple:
    """
    Draw grouped profile plots.

    Phase 13.12.DF: Added min_entries, sort_groups, return_data parameters.
    Phase 13.12.DF v1.1: Added weights parameter.
    Phase 13.32.DF Sub-fix 2 (v1.2 C11): Added quantile rendering parameters.
        Per-group band-mode rendering uses _render_quantile_band with the
        Sub-fix 2 alpha override. Per-group discrete-mode rendering is inline
        (option B): group_color + linestyle='--' for quantile lines, deliberately
        omitting the FIX2 channel-cycle machinery from the single-profile path
        (cycling linestyle PER QUANTILE on top of group color creates unreadable
        plots with N groups × M quantiles).

        nested_band mode raises NotImplementedError explicitly — it is the exact
        silent-drop bug class Phase 13.32 exists to fix.

    Returns
    -------
    (profile_data_list, per_group_stats)
        profile_data_list : list of DataFrame or None
            One per-group profile DataFrame, if return_data=True; else None.
        per_group_stats : dict[group_label, dict] or None
            Per-group quantile diagnostics, populated only when quantiles is
            not None. Schema per PHASE_13_32_DF_v1_2_Proposal §3.2.
    """
    # Phase 13.32.DF Sub-fix 2: nested_band + group_by raises early.
    # Silent fallthrough would mean the quantile_list is computed but never
    # rendered, reproducing the v1.0-incident A bug class.
    if quantiles is not None and quantile_mode == 'nested_band':
        raise NotImplementedError(
            "quantile_mode='nested_band' is not yet supported with group_by. "
            "Use quantile_mode='band' or quantile_mode='discrete' instead. "
            "Nested-band rendering with per-group color coordination is "
            "scheduled for a future phase."
        )

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
    linewidth = profile_kwargs.get('linewidth', 1.5)
    # Phase 13.36.DF: pops for 'marker' and 'markersize' DELETED. These values
    # are now forwarded as _user_marker / _user_markersize named params from
    # draw_profile() (the local variables where the user's values live after
    # consumption by draw_profile()'s explicit signature). The call site at
    # line 489 no longer passes marker=marker / markersize=markersize.
    # Both vector and non-vector paths route the values through _user_* params.
    # → Nothing reaches profile_kwargs to pop.

    # Phase 13.12.DF F1: Collect profile data
    profile_data_list = [] if return_data else None

    # Phase 13.32.DF Sub-fix 2: per-group stats (only populated when quantiles is set)
    per_group_stats = {} if quantiles is not None else None

    # Phase 13.32.DF Sub-fix 2: resolve grouped band alpha once
    _band_alpha_grouped = get_style_value("quantile.band.alpha_grouped", 0.15)

    # Phase 13.37.DF: linestyle_cycle setup. When active AND user didn't pass
    # linestyle explicitly, pop linestyle from profile_kwargs ONCE (out-of-loop)
    # and set per-group linestyle inside the loop. If linestyle_cycle=False OR
    # user passed linestyle, leave profile_kwargs untouched (existing flow).
    # Priority: user explicit > cycle > style default (CP1-3 contract).
    _ls_cycle = None
    if linestyle_cycle and _user_linestyle is None:
        # Pop linestyle so per-group cycle linestyle can be passed explicitly
        # without **profile_kwargs double-keying it. Style fill-in in
        # draw_profile already filled linestyle to "-"; popping the post-fill
        # value is fine (we don't reuse it).
        profile_kwargs.pop('linestyle', None)
        _ls_cycle = get_style_value(
            "channels.cycles.linestyle", ['-', '--', '-.', ':']
        )

    for i, group in enumerate(groups):
        group_df = df[df[group_by] == group]
        x_data = group_df[x].values.astype(float)
        y_data = group_df[y].values.astype(float)

        # Phase 13.12.DF v1.1: Get weights for this group
        # Bugfix: support weight expressions, not just column names
        w_data = None
        if weights is not None:
            w_data = _eval_weights(group_df, weights)

        # Phase 13.28.DF: catch NaN AND inf via isfinite (consistent with sanitize_for_plot)
        mask = np.isfinite(x_data) & np.isfinite(y_data)
        if w_data is not None:
            mask &= np.isfinite(w_data)
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

        # Phase 13.36.DF: user color override > palette cycle.
        # None = "user did not pass" (matches matplotlib default-color semantics).
        group_color = (palette(i % 10)
                       if _user_color is None
                       else _user_color)

        # Phase 13.36.DF: warn ONCE per call (i == 0) when color= makes all
        # groups indistinguishable. Architect decision 2026-05-20.
        if i == 0 and _user_color is not None and len(groups) > 1:
            import warnings
            warnings.warn(
                f"color={_user_color!r} applied uniformly to all {len(groups)} groups. "
                f"Groups will be indistinguishable by color. "
                f"Omit color= to use per-group colors (default behavior).",
                UserWarning, stacklevel=4
            )

        # Phase 13.32.DF Sub-fix 2: per-group quantile rendering BEFORE the
        # central line so the central line stays visually on top.
        if quantiles is not None:
            per_group_stats[group] = {'n': int(len(x_data))}

            if quantile_pair is not None and quantile_mode in ('band', 'error_bars'):
                # Band mode (error_bars in grouped path also renders as a band
                # because per-group asymmetric error bars would visually clash
                # with the per-group central marker).
                _, _q_lower, _q_upper, _ = _compute_per_bin_quantiles(
                    x_data, y_data, bins, x_range, quantile_pair,
                )
                _render_quantile_band(
                    ax, bin_centers, _q_lower, _q_upper, plot_mask,
                    color=group_color,
                    alpha=_band_alpha_grouped,  # Sub-fix 2 override
                )
                per_group_stats[group]['q_lower_per_bin'] = _q_lower
                per_group_stats[group]['q_upper_per_bin'] = _q_upper

            elif quantile_list is not None and quantile_mode == 'discrete':
                # Phase 13.32 Sub-fix 2 (v1.2 §3.2 option B): inline simplified
                # discrete rendering. No FIX2 channel cycle (groups already use
                # color); consistent dashed linestyle for quantile lines under
                # group_color keeps central line (solid) visually distinct.
                _q_all = _compute_per_bin_all_quantiles(
                    x_data, y_data, bins, x_range, quantile_list,
                )
                for j, (q_val, q_per_bin) in enumerate(_q_all.items()):
                    # Per Sonet50 v1.2 P2-1: legend entry shows just the group
                    # label (not "Group A q=10%") to avoid implying only one
                    # quantile is shown. One legend entry per group via j==0 guard.
                    legend_label = str(group) if j == 0 else None
                    ax.plot(
                        bin_centers[plot_mask], q_per_bin[plot_mask],
                        color=group_color,
                        linestyle='--',
                        linewidth=linewidth * 0.7,
                        label=legend_label,
                    )
                per_group_stats[group]['quantiles_per_bin'] = _q_all

        # Central line (mean or median) — always rendered last so it sits on top
        # Phase 13.36.DF: user marker/markersize override > cycle.
        # None on either passes through to matplotlib defaults (correct).
        group_marker = (markers[i % len(markers)]
                        if _user_marker is None
                        else _user_marker)
        # Phase 13.37.DF: per-group linestyle from cycle when active.
        # _ls_cycle is None except when (linestyle_cycle=True AND user didn't
        # pass linestyle). When None, profile_kwargs still has linestyle from
        # the existing flow (user explicit OR style default "-").
        if _ls_cycle is not None:
            _per_group_ls = _ls_cycle[i % len(_ls_cycle)]
            ax.errorbar(
                bin_centers[plot_mask], bin_means[plot_mask], yerr=bin_errors[plot_mask],
                fmt=group_marker,
                color=group_color,
                markersize=_user_markersize,
                linestyle=_per_group_ls,   # Phase 13.37.DF cycle
                label=(None if quantiles is not None and quantile_list is not None
                       and quantile_mode == 'discrete'
                       else str(group)),
                **profile_kwargs
            )
        else:
            ax.errorbar(
                bin_centers[plot_mask], bin_means[plot_mask], yerr=bin_errors[plot_mask],
                fmt=group_marker,
                color=group_color,
                markersize=_user_markersize,   # None → matplotlib default
                # If quantiles already added a legend entry for this group, suppress
                # the duplicate central-line legend entry by setting label=None.
                label=(None if quantiles is not None and quantile_list is not None
                       and quantile_mode == 'discrete'
                       else str(group)),
                **profile_kwargs
            )

    return profile_data_list, per_group_stats


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
    
    The GENERAL case is discrete (one line per quantile). Error_bars, band,
    and nested_band are OPTIMIZATIONS for specific symmetric shapes.
    
    Returns
    -------
    str
        'error_bars', 'band', 'nested_band', or 'discrete'.
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
    
    # Single value → discrete (just one quantile line)
    if n == 1:
        return 'discrete'
    
    # Check if symmetric pair (no 0.5) → error_bars optimization
    if n == 2:
        is_symmetric = abs(qs[0] + qs[1] - 1.0) < 1e-9
        has_05 = any(abs(q - 0.5) < 1e-9 for q in qs)
        if is_symmetric and not has_05:
            return 'error_bars'
    
    # Check if symmetric triple with 0.5 → band optimization
    if n == 3:
        has_05 = abs(qs[1] - 0.5) < 1e-9
        is_symmetric = abs(qs[0] + qs[2] - 1.0) < 1e-9
        if has_05 and is_symmetric:
            return 'band'
    
    # Phase 13.26.DF Phase B (Option A, AD-57): multiple symmetric pairs
    # form a nested-band, with or without a central 0.5. Per architect F-5
    # 2026-05-05: central=0.5 is NOT required — the central line is handled
    # by the central= parameter independently from mode detection.
    # Detection rule: at least 2 symmetric pairs (non_05_count >= 4).
    non_05 = [q for q in qs if abs(q - 0.5) > 1e-9]
    if len(non_05) >= 4:
        # Check that all non_05 entries form symmetric pairs
        # (sorted, so qs[i] + qs[-(i+1)] should equal 1.0 for each pair).
        n_non05 = len(non_05)
        pairs_symmetric = all(
            abs(non_05[i] + non_05[n_non05 - 1 - i] - 1.0) < 1e-9
            for i in range(n_non05 // 2)
        )
        if pairs_symmetric:
            return 'nested_band'
    
    # Everything else → discrete (one line per quantile)
    return 'discrete'


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


def _compute_per_bin_all_quantiles(
    x_data: np.ndarray,
    y_data: np.ndarray,
    bins: int,
    x_range,
    quantile_list: list,
    w_data=None,
) -> dict:
    """
    Compute per-bin values for ALL quantiles in the list.
    
    The general case: any list of quantile fractions.
    Returns a dict {q_value: per_bin_array} for discrete-mode rendering.
    """
    if w_data is not None:
        raise NotImplementedError("weighted quantiles deferred to Phase B")
    
    if x_range is None:
        x_range = (np.nanmin(x_data), np.nanmax(x_data))
    
    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_indices = np.clip(np.digitize(x_data, bin_edges) - 1, 0, bins - 1)
    
    result = {}
    for q in quantile_list:
        q_per_bin = np.full(bins, np.nan)
        for i in range(bins):
            mask = bin_indices == i
            y_bin = y_data[mask]
            y_bin = y_bin[~np.isnan(y_bin)]
            if len(y_bin) > 0:
                q_per_bin[i] = float(np.nanpercentile(y_bin, q * 100))
        result[q] = q_per_bin
    
    return result


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


# =============================================================================
# Phase 13.33.DF: Normalized differential profiles — helpers (AD-80/81/82)
# =============================================================================
# These three helpers are used by drawer.DFDraw._dispatch_normalize_render()
# to power the normalize= kwarg on profile(). They are pure functions —
# fully testable in isolation, no matplotlib state dependency beyond the
# Axes passed to _render_normalize_panel.

# Median Absolute Deviation scale factor: 1.4826 makes MAD a consistent
# estimator of σ for normally-distributed data. Same constant used in
# plots/_autorange.py for the robust window strategy (single source of truth
# kept here as a module-local since the dependency direction is
# profile.py → _autorange.py, never the reverse).
_MAD_TO_SIGMA = 1.4826


def _compute_per_bin_mad_sigma(
    x_data: np.ndarray,
    y_data: np.ndarray,
    bins: int,
    x_range,
) -> np.ndarray:
    """
    Compute per-bin MAD-sigma of y in bins of x (Phase 13.33.DF AD-80).

    MAD-sigma = 1.4826 × median(|y - median(y)|) per bin — a robust
    dispersion estimate that pairs naturally with central='median' for
    normalize-mode error propagation.

    Mirrors the structure of _compute_per_bin_median above (same binning,
    same NaN handling) so the two arrays index identically by bin.

    For bins with fewer than 2 entries, MAD is undefined (a single point
    has zero deviation by construction); we return NaN in those bins so
    downstream code can mask them consistently.

    Parameters
    ----------
    x_data, y_data : array
        Sanitized 1D arrays (NaN/inf already removed by upstream sanitize).
    bins : int
        Number of x bins.
    x_range : tuple or None
        (min, max) binning range; auto-detected from x_data if None.

    Returns
    -------
    bin_mad_sigma : array of length `bins`
        Per-bin MAD-sigma estimate. NaN for empty or singleton bins.
    """
    if x_range is None:
        x_range = (np.nanmin(x_data), np.nanmax(x_data))

    bin_edges = np.linspace(x_range[0], x_range[1], bins + 1)
    bin_indices = np.clip(np.digitize(x_data, bin_edges) - 1, 0, bins - 1)

    bin_mad_sigma = np.full(bins, np.nan)
    for i in range(bins):
        mask = bin_indices == i
        y_bin = y_data[mask]
        y_bin = y_bin[~np.isnan(y_bin)]
        if len(y_bin) >= 2:
            med = np.nanmedian(y_bin)
            mad = np.nanmedian(np.abs(y_bin - med))
            bin_mad_sigma[i] = float(_MAD_TO_SIGMA * mad)

    return bin_mad_sigma


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
    # Clamp to >= 0: for highly skewed data (e.g., boolean), quantile can
    # exceed the central value, producing negative deltas. matplotlib
    # requires yerr >= 0.
    c = central_values[plot_mask]
    lower_delta = np.maximum(0, c - q_lower[plot_mask])
    upper_delta = np.maximum(0, q_upper[plot_mask] - c)
    yerr = np.array([lower_delta, upper_delta])
    
    ax.errorbar(
        bin_centers[plot_mask], c, yerr=yerr,
        fmt=marker, color=color, markersize=markersize,
        capsize=capsize, linestyle=linestyle, linewidth=linewidth,
        label=label, **kwargs
    )


def _render_quantile_band(
    ax, bin_centers, q_lower, q_upper, plot_mask, color,
    alpha=None,  # Phase 13.32.DF Sub-fix 2: optional override of style-key alpha
):
    """
    Render quantile band via fill_between.
    
    Alpha and hatch read from quantile.band.alpha and quantile.band.hatch
    style keys (AD-53). Band color matches the central line's color.

    Phase 13.32.DF Sub-fix 2: ``alpha`` parameter allows callers (the grouped
    path) to override the style-key default with a lower value
    (``quantile.band.alpha_grouped``) so stacked group-colored bands stay
    readable.
    """
    if alpha is None:
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


def _render_quantile_nested_band(
    ax, bin_centers, q_all, plot_mask, color,
):
    """
    Render nested alpha-stacked bands from outermost pair inward.

    Phase 13.26.DF Phase B (Option A, AD-57). Multiple symmetric quantile
    pairs rendered as fill_between regions with decreasing alpha from outer
    to inner. Maximum 3 nested bands (silent truncation, outermost 3
    selected).

    Channel cost: 0 (zero-cost mode — no visual channel slot consumed by
    Algorithm A; band fills overlay the central line).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    bin_centers : np.ndarray
        Bin centre coordinates.
    q_all : dict[float, np.ndarray]
        Mapping ``{q_value: per_bin_quantile_values}`` from
        ``_compute_per_bin_all_quantiles``.
    plot_mask : np.ndarray of bool
        Bins to render (min_entries filter).
    color : str or RGBA tuple
        Fill color (matches central line).
    """
    qs = sorted(q_all.keys())
    # Find symmetric pairs (outermost first). Walk from both ends; pair
    # entries whose sum is 1.0 (within tolerance). Skip 0.5 implicitly
    # since 0.5 + 0.5 = 1.0 only matches itself.
    pairs = []
    left, right = 0, len(qs) - 1
    while left < right:
        if abs(qs[left] + qs[right] - 1.0) < 1e-9:
            pairs.append((qs[left], qs[right]))
            left += 1
            right -= 1
        else:
            # Asymmetric — skip the entry that overshoots; should not happen
            # given _detect_quantile_mode validates symmetry, but be safe.
            if qs[left] + qs[right] < 1.0:
                left += 1
            else:
                right -= 1

    # Max 3 nested bands per AD-57 / v1.2 §7.2 (silent truncation, outermost 3).
    pairs = pairs[:3]
    n_pairs = len(pairs)
    if n_pairs == 0:
        return

    # Outer band: lowest alpha; inner band: highest alpha.
    # Alpha range: 0.10 (outermost) to 0.30 (innermost) for n_pairs=3.
    base_alpha = get_style_value("quantile.band.alpha", 0.25)
    for i, (q_lo, q_hi) in enumerate(pairs):
        # Outer (i=0) gets the lowest alpha; inner (i=n_pairs-1) gets base_alpha.
        # Linear interpolation: alpha = base_alpha * (i + 1) / n_pairs.
        alpha = base_alpha * (i + 1) / n_pairs
        ax.fill_between(
            bin_centers[plot_mask],
            q_all[q_lo][plot_mask],
            q_all[q_hi][plot_mask],
            alpha=alpha, color=color,
        )


# =============================================================================
# Phase 13.33.DF: Normalize transform — math kernel (AD-80, all 5 modes)
# =============================================================================
# Pure function. Called by drawer.DFDraw._dispatch_normalize_render after both
# top-panel profiles have been computed. Returns the bottom-panel arrays
# (values + errors per bin) plus a mask of bins where the transform is undefined.

# Allowed string modes for normalize=. Anything outside this set + non-callable
# raises ValueError at the public API entry (drawer.profile validation block).
NORMALIZE_MODES = ("delta", "ratio", "log_ratio", "pull")


def _compute_normalize_transform(
    stats_0: Dict[str, np.ndarray],
    stats_1: Dict[str, np.ndarray],
    mode,
    central: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the normalize transform to two per-bin profile statistics.

    AD-80 sign convention: stats_0 is the signal (vector[0]), stats_1 is the
    reference (vector[1]). `delta = signal − reference` (signed; positive
    means signal is above reference).

    Parameters
    ----------
    stats_0, stats_1 : dict
        Per-bin arrays from the two profile curves. Required keys:
          - 'bin_centers': array of bin midpoints (must be identical between
            the two — caller is responsible for ensuring matched binning).
          - 'central':     per-bin central estimator (mean if central='mean',
                           median if central='median').
          - 'sigma':       per-bin dispersion (std if central='mean',
                           MAD-sigma if central='median').
          - 'counts':      per-bin n (used in SEM-based error propagation).
    mode : str or callable
        Transform mode. If callable, receives (stats_0, stats_1) and must
        return either (values, errors) or values alone (errors → None,
        bottom-panel rendered without error bars).
        String modes: "delta", "ratio", "log_ratio", "pull".
    central : str
        "mean" or "median". Determines which dispersion enters the error
        formula — std/sem for mean mode, MAD-sigma for median mode. Phase
        13.33 §3.4 implements all 8 cells (mean/median × delta/ratio/
        log_ratio/pull); no NotImplementedError.

    Returns
    -------
    values : array
        Per-bin transform output (delta, ratio, log_ratio, pull, or callable
        return). Length matches stats_0['bin_centers'].
    errors : array or None
        Per-bin propagated error (or None for callable returning values only).
    mask_undefined : array of bool
        True for bins where the transform is undefined (e.g. ratio with zero
        denominator, log_ratio with non-positive input, callable returned NaN).
        Caller masks both values and errors to NaN at these positions.

    Notes
    -----
    Error propagation uses standard error of mean (SEM-based):
        SEM = σ / √n   per bin

    Per v1.1 §3.3:
      delta_err   = √(σ₀²/n₀ + σ₁²/n₁)
      ratio_err   = |μ₀/μ₁| · √(σ₀²/(n₀·μ₀²) + σ₁²/(n₁·μ₁²))
      log_ratio_err = √(σ₀²/(n₀·μ₀²) + σ₁²/(n₁·μ₁²))     (derived via log)
      pull        = (μ₀ − μ₁) / √(σ₀²/n₀ + σ₁²/n₁)        (errors = 1 by def.)

    For median mode, σ in the formulas above is MAD-sigma. Pre-existing
    inconsistency: profile(central='median') currently renders mean-based
    errors on the central line itself; this is NOT touched in Phase 13.33
    (flagged in CRR §11 as a candidate for a separate fix-up phase).
    """
    mu_0 = stats_0['central']
    mu_1 = stats_1['central']
    sig_0 = stats_0['sigma']
    sig_1 = stats_1['sigma']
    n_0 = stats_0['counts']
    n_1 = stats_1['counts']

    # Bins where either side is empty → undefined for every mode.
    base_undefined = (n_0 < 1) | (n_1 < 1)

    # Suppress numpy warnings inside the math — we explicitly mask afterwards.
    with np.errstate(divide='ignore', invalid='ignore'):
        # SEM² per side (used by every mode that propagates error).
        sem2_0 = (sig_0 ** 2) / np.where(n_0 > 0, n_0, 1)
        sem2_1 = (sig_1 ** 2) / np.where(n_1 > 0, n_1, 1)

        if callable(mode):
            result = mode(stats_0, stats_1)
            # Accept (values, errors) tuple OR values alone (errors=None).
            if isinstance(result, tuple) and len(result) == 2:
                values, errors = result
                values = np.asarray(values, dtype=float)
                errors = np.asarray(errors, dtype=float) if errors is not None else None
            else:
                values = np.asarray(result, dtype=float)
                errors = None
            mask_undefined = base_undefined | ~np.isfinite(values)

        elif mode == "delta":
            values = mu_0 - mu_1
            errors = np.sqrt(sem2_0 + sem2_1)
            mask_undefined = base_undefined

        elif mode == "ratio":
            # Undefined where reference (mu_1) is zero.
            mask_zero_ref = (mu_1 == 0) | ~np.isfinite(mu_1)
            mask_undefined = base_undefined | mask_zero_ref
            values = mu_0 / mu_1
            # |μ₀/μ₁| · √(σ₀²/(n₀·μ₀²) + σ₁²/(n₁·μ₁²))
            # The factor μ_i² in the denominator makes ratio_err undefined
            # where μ_i=0; we mask the result there.
            mask_zero_either = mask_zero_ref | (mu_0 == 0)
            rel_var_0 = np.where(
                (mu_0 != 0) & np.isfinite(mu_0),
                sem2_0 / (mu_0 ** 2),
                0.0,
            )
            rel_var_1 = np.where(
                (mu_1 != 0) & np.isfinite(mu_1),
                sem2_1 / (mu_1 ** 2),
                0.0,
            )
            errors = np.abs(values) * np.sqrt(rel_var_0 + rel_var_1)
            # If either side is zero we still report value (∞ or 0) but error
            # cannot be propagated → mask both at undefined positions.
            mask_undefined = mask_undefined | mask_zero_either

        elif mode == "log_ratio":
            # ln(μ₀ / μ₁) is defined only where both means are strictly positive.
            mask_nonpos = (mu_0 <= 0) | (mu_1 <= 0) | ~np.isfinite(mu_0) | ~np.isfinite(mu_1)
            mask_undefined = base_undefined | mask_nonpos
            values = np.log(np.where(mask_nonpos, np.nan, mu_0 / mu_1))
            # d(ln(μ₀/μ₁)) = dμ₀/μ₀ − dμ₁/μ₁ → variance = σ₀²/(n₀ μ₀²) + σ₁²/(n₁ μ₁²)
            rel_var_0 = np.where(
                (mu_0 > 0) & np.isfinite(mu_0),
                sem2_0 / (mu_0 ** 2),
                0.0,
            )
            rel_var_1 = np.where(
                (mu_1 > 0) & np.isfinite(mu_1),
                sem2_1 / (mu_1 ** 2),
                0.0,
            )
            errors = np.sqrt(rel_var_0 + rel_var_1)

        elif mode == "pull":
            # pull = (μ₀ − μ₁) / √(σ₀²/n₀ + σ₁²/n₁)
            denom_var = sem2_0 + sem2_1
            mask_zero_denom = (denom_var == 0) | ~np.isfinite(denom_var)
            mask_undefined = base_undefined | mask_zero_denom
            denom = np.sqrt(np.where(mask_zero_denom, np.nan, denom_var))
            values = (mu_0 - mu_1) / denom
            # Pull is by construction in units of σ → "error" is 1.0
            # in those units. We return np.ones() so error bars can show
            # the statistical 1σ on the pull plot (matches HEP convention
            # for residual / pull plots — vertical bar = 1σ uncertainty
            # on the pull value itself).
            errors = np.ones_like(values, dtype=float)

        else:
            raise ValueError(
                f"Unknown normalize mode {mode!r}. Expected one of "
                f"{NORMALIZE_MODES} or a callable."
            )

    # Apply mask: NaN out undefined bins so downstream rendering skips them.
    values = np.where(mask_undefined, np.nan, values)
    if errors is not None:
        errors = np.where(mask_undefined, np.nan, errors)

    return values, errors, mask_undefined


# =============================================================================
# Phase 13.33.DF: Normalize-panel rendering (AD-82 pull bands; AD-80 ref line)
# =============================================================================

def _render_normalize_panel(
    ax,
    bin_centers: np.ndarray,
    values: np.ndarray,
    errors,
    mode,
    *,
    color: Optional[str] = None,
    marker: Optional[str] = None,
    markersize: Optional[float] = None,
    capsize: Optional[float] = None,
    label: Optional[str] = None,
) -> None:
    """
    Render the bottom panel of a normalize plot.

    Draws (in order, per v1.1 §6):
      1. Pull bands (±1σ, ±2σ) if mode == 'pull'
      2. Reference line (y=0 for delta/log_ratio/pull; y=1 for ratio)
         — gated on style key 'normalize.panel.reference_line'
      3. The differential curve with error bars
      4. Pull-anomaly highlighting (|pull| > threshold) for mode='pull'

    Caller (drawer._dispatch_normalize_render) supplies ax pre-built from
    gridspec with sharex=True against the top panel — formatter and limits
    inherit automatically (sidesteps BUG-004 per Claude40 awareness note).

    NaN values in `values` are skipped by errorbar (matplotlib treats them
    as missing — no markers drawn at those bin centers). This matches the
    upstream convention for empty bins in profile().
    """
    # ---- 1. Pull bands ------------------------------------------------------
    if mode == "pull":
        a1 = get_style_value("normalize.pull.band_1sigma_alpha", 0.15)
        a2 = get_style_value("normalize.pull.band_2sigma_alpha", 0.08)
        # Bands span the full x range of the panel — use axhspan-style fill
        # across the visible bin_centers range (sharex copies the actual limits).
        # We use ax.axhspan for infinite-width bands so they stay flush against
        # the panel edges regardless of bin layout.
        ax.axhspan(-1.0,  1.0, alpha=a1, color="gray", zorder=0)
        ax.axhspan(-2.0, -1.0, alpha=a2, color="gray", zorder=0)
        ax.axhspan( 1.0,  2.0, alpha=a2, color="gray", zorder=0)

    # ---- 2. Reference line --------------------------------------------------
    if get_style_value("normalize.panel.reference_line", True):
        ref_y = 1.0 if mode == "ratio" else 0.0
        ax.axhline(
            ref_y,
            color=get_style_value("normalize.panel.ref_line_color", "gray"),
            linestyle=get_style_value("normalize.panel.ref_line_style", "--"),
            linewidth=1.0,
            zorder=1,
        )

    # ---- 3. The differential curve -----------------------------------------
    if marker is None:
        marker = get_style_value("profile.marker", "o")
    if markersize is None:
        markersize = get_style_value("profile.markersize", 6)
    if capsize is None:
        capsize = get_style_value("profile.capsize", 3)

    # errorbar with yerr=None when errors is None (callable mode returning
    # values only).
    yerr = errors if errors is not None else None
    ax.errorbar(
        bin_centers, values,
        yerr=yerr,
        fmt=marker,
        markersize=markersize,
        capsize=capsize,
        color=color,
        label=label,
        zorder=3,
    )

    # ---- 4. Pull anomaly highlighting --------------------------------------
    if mode == "pull":
        threshold = get_style_value("normalize.pull.highlight_threshold", 3.0)
        anomaly_mask = np.abs(values) > threshold
        # Only draw highlights where defined (mask out NaN to avoid the
        # all-NaN-comparison warning that np.abs raises on masked arrays).
        anomaly_mask = anomaly_mask & np.isfinite(values)
        if anomaly_mask.any():
            ax.scatter(
                bin_centers[anomaly_mask],
                values[anomaly_mask],
                s=(markersize ** 2) * 2.5,  # larger than errorbar marker
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                zorder=4,
                label=None,  # don't pollute legend
            )


# ====================================================================== #
# Phase 13.39.DF — draw_profile2d (2D mean heatmap via z:y:x expression)  #
# ====================================================================== #

def draw_profile2d(
    df: pd.DataFrame,
    z_expr: str,
    y_expr: str,
    x_expr: str,
    ax: Optional[plt.Axes] = None,
    bins: Union[int, List[int]] = 50,
    bins2: Optional[int] = None,
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
    central: str = 'mean',
    cmap: str = 'viridis',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    clabel: Optional[str] = None,
    colorbar: bool = True,
    min_entries: int = 0,
    norm: Optional[str] = None,
    time_format: Optional[str] = None,
    auto_title: Union[bool, str] = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    nan_policy: str = 'filter',
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """Draw 2D profile (mean of z in bins of x and y).

    Phase 13.39.DF — invoked when DFDraw.profile() detects a 'z:y:x'
    expression (colon_count == 2). Uses scipy.stats.binned_statistic_2d
    for per-cell statistics, renders via ax.pcolormesh.

    Parameters
    ----------
    df : DataFrame
        Pre-filtered (selection + sample applied by caller).
    z_expr, y_expr, x_expr : str
        Column names or df.eval() expressions.
    bins : int or [int, int]
        x-axis bin count (scalar) or [n_x, n_y].
    bins2 : int, optional
        y-axis bin count (overrides bins[1] if both passed).
    central : str
        'mean' (default) or 'median' (scipy statistic).
    min_entries : int
        Cells with count < min_entries are masked (set to NaN).
    norm : str, optional
        'log' for LogNorm; None for linear.
    time_format : str, optional
        x-axis time format. 'auto' → AutoDateFormatter; else DateFormatter
        with the given strftime string. Phase 13.39 CP1-4: auto-detects
        datetime64 dtype to avoid pd.to_datetime unit='s' crash.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic_2d  # Phase 13.39 CP1-6: required dep

    # Evaluate expressions (column name OR df.eval)
    def _eval(expr):
        if expr in df.columns:
            return df[expr].values
        return df.eval(expr).values

    x_data = _eval(x_expr)
    y_data = _eval(y_expr)
    z_data = _eval(z_expr)

    # Drop rows where any of x/y/z is non-finite (joint mask)
    mask = (np.isfinite(x_data) & np.isfinite(y_data) & np.isfinite(z_data))
    x_data = x_data[mask].astype(float)
    y_data = y_data[mask].astype(float)
    z_data = z_data[mask].astype(float)

    # Resolve bin counts
    if isinstance(bins, (list, tuple)):
        n_x = bins[0]
        n_y_default = bins[1] if len(bins) > 1 else bins[0]
    else:
        n_x = bins
        n_y_default = bins
    n_y = bins2 if bins2 is not None else n_y_default

    # CP1-2: range is single-level outer list (NOT 3-level nested)
    if len(x_data) == 0:
        raise ValueError("draw_profile2d: no data points after sanitization")
    _range = [
        list(x_range) if x_range else [float(np.min(x_data)), float(np.max(x_data))],
        list(y_range) if y_range else [float(np.min(y_data)), float(np.max(y_data))],
    ]

    # Compute per-cell mean (or median) via scipy
    z_mean, x_edges, y_edges, _ = binned_statistic_2d(
        x_data, y_data, z_data,
        statistic=central, bins=[n_x, n_y], range=_range,
    )
    z_count, _, _, _ = binned_statistic_2d(
        x_data, y_data, z_data,
        statistic='count', bins=[n_x, n_y], range=_range,
    )
    # Mask low-count cells
    z_mean[z_count < min_entries] = np.nan
    n_masked = int((z_count < min_entries).sum())

    # Log norm (mirrors draw_hist2d pattern)
    norm_obj = None
    if norm == 'log':
        from matplotlib.colors import LogNorm
        positive_vmin = vmin if (vmin is not None and vmin > 0) else None
        norm_obj = LogNorm(vmin=positive_vmin, vmax=vmax)

    # time_format pre-conversion (CP1-4 auto-detect)
    if time_format is not None:
        import matplotlib.dates as mdates
        x_edges_arr = np.asarray(x_edges)
        if np.issubdtype(x_edges_arr.dtype, np.datetime64):
            x_edges = mdates.date2num(x_edges_arr)
        else:
            x_edges = mdates.date2num(
                pd.to_datetime(x_edges_arr, unit='s').to_pydatetime()
            )

    # Set up axes
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # Render via pcolormesh — z_mean.T because scipy returns shape (n_x, n_y)
    # but pcolormesh expects shape (n_y, n_x) for proper orientation
    im = ax.pcolormesh(
        x_edges, y_edges, z_mean.T,
        cmap=cmap, vmin=vmin, vmax=vmax, norm=norm_obj,
    )

    if colorbar:
        cbar = plt.colorbar(im, ax=ax)
        if clabel:
            cbar.set_label(clabel)

    # time_format formatter applied after render
    if time_format is not None:
        import matplotlib.dates as mdates
        if time_format == "auto":
            ax.xaxis.set_major_formatter(
                mdates.AutoDateFormatter(mdates.AutoDateLocator()))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
        fig.autofmt_xdate()

    # Labels
    ax.set_xlabel(xlabel or x_expr)
    ax.set_ylabel(ylabel or y_expr)
    if title:
        ax.set_title(title)
    elif auto_title:
        ax.set_title(f"{z_expr} vs ({y_expr}, {x_expr})")

    stats: Dict[str, Any] = {
        'n': int(mask.sum()),
        'n_cells': n_x * n_y,
        'n_masked_cells': n_masked,
        'z_mean': z_mean,
        'z_count': z_count,
        'clabel': clabel,
        'x_edges': x_edges,
        'y_edges': y_edges,
    }
    return fig, ax, stats
