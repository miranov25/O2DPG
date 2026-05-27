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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..style import get_style_value
from ..stats import format_stats_box
from ._auto_title import build_auto_title, apply_auto_title, parse_auto_title_parts, resolve_auto_title
# Phase 13.28.DF: Robust data handling
from ._data_sanitize import sanitize_for_plot
from ._autorange import compute_autorange, VALID_STRATEGIES
# Phase 13.30.DF: Class-2 column-reference parameter validation
from ._validation import validate_column_references
# Phase 13.37.DF (BUG-016): pd.Interval-aware sort key for legend ordering.
# _interval_sort_key was extended in Phase 13.37 to handle pd.Interval objects
# via hasattr(label, 'left'). Imported here so _draw_hist_grouped can sort raw
# pd.Interval groups from pd.cut() in numeric order, not lexicographic.
from .profile import _interval_sort_key
# Phase 13.42.DF: Inline fits
from .fits import normalize_fit_spec, dispatch_fit
from ._fit_render import render_fit_overlays, render_fit_textbox


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


def _group_weights(
    x_group: np.ndarray,
    bin_edges: Optional[np.ndarray],
    hist_norm: Optional[str],
) -> Optional[np.ndarray]:
    """Per-group histogram weights for hist_norm normalization (Phase 13.35.DF).

    Returns
    -------
    None  if hist_norm is None (raw counts — pass-through to ax.hist)
    ndarray of length len(x_group) otherwise:
      'probability'  : weights = 1/n        → sum(heights) = 1.0
      'density'      : weights = 1/(n × Δx) → ∫ heights dx = 1.0

    Raises
    ------
    ValueError if hist_norm is not None / 'probability' / 'density'.
    """
    if hist_norm is None:
        return None
    n = len(x_group)
    if n == 0:
        return None
    if hist_norm == "probability":
        return np.ones(n) / n
    if hist_norm == "density":
        if bin_edges is not None:
            bin_width = float(np.diff(bin_edges).mean())
        else:
            bin_width = 1.0
        return np.ones(n) / (n * bin_width)
    raise ValueError(
        f"hist_norm must be None, 'probability', or 'density'; "
        f"got {hist_norm!r}"
    )


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
    # Phase 13.35.DF: float group_by binning + per-group normalization (BUG-013 fix).
    # group_by_bins / group_by_quantiles: bin a float group_by column via pd.cut/qcut
    # before the per-group rendering loop. min_entries: skip groups below threshold.
    # hist_norm: per-group normalization (None=raw counts, "probability"=sum=1,
    # "density"=area=1). Independent of single-histogram norm= parameter.
    group_by_bins: Optional[int] = None,
    group_by_quantiles: Optional[int] = None,
    hist_norm: Optional[str] = None,
    min_entries: int = 0,
    # Phase 13.37.DF: Poisson error bar overlay on histogram bars.
    # When True, computes per-bin √n / N error bars and overlays via
    # ax.errorbar. Composes with hist_norm (probability, density). Composes
    # with weights= column via weighted Poisson (yerr = sqrt(Σw²)). Zero-count
    # bins are skipped (mask = counts > 0).
    hist_errors: bool = False,
    # Phase 13.37.DF: per-group linestyle cycling mode flag. When True and
    # user did not pass linestyle= explicitly, cycle through
    # channels.cycles.linestyle per group. User-explicit linestyle= takes
    # precedence (Phase 13.36 sentinel pattern).
    linestyle_cycle: bool = False,
    # Phase 13.39.DF: time-axis formatting (pre-conversion approach, CP1-4 auto-detect).
    time_format: Optional[str] = None,
    # Phase 13.40.DF: cumulative histogram (CDF/ECDF/survival).
    # False (default) → regular histogram (byte-identical backward compat)
    # True → ascending cumulative (each bin = count ≤ right edge)
    # -1 → descending / survival (ROOT convention)
    cumulative: Union[bool, int] = False,
    # Phase 13.42.DF: Inline fit specification (architect 2026-05-22).
    fit: Optional[Union[str, Dict, Callable, List]] = None,
    # Phase 13.42.DF FIX1 (B2/R5): per-call fit textbox formatting overrides.
    # Sub-keys: 'fontsize' (int), 'format' ('multiline'|'compact'|'auto'),
    # 'show_fields' (list of str). None → use style defaults. Sub-key names +
    # accepted enum values are LOCKED at FIX1 close.
    fit_textbox_kwargs: Optional[Dict[str, Any]] = None,
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
    # Phase 13.37.DF (BUG-014): capture user-explicit edgecolor BEFORE the
    # style fill-in below. Pattern mirrors Phase 13.36 Edit 17 (the lesson:
    # style fill-in replaces None with default; sentinels must capture pre-
    # fill-in to distinguish "user passed nothing" from "user explicitly
    # passed 'red'"). Used in _draw_hist_grouped step-mode branch.
    _ud_user_edgecolor = edgecolor
    # Phase 13.37.DF: capture user-explicit linestyle for linestyle_cycle
    # priority logic. linestyle is NOT in draw_hist's explicit signature
    # (lives in **kwargs); peek without popping so any non-cycle path still
    # receives it via **kwargs → ax.hist.
    _ud_user_linestyle = kwargs.get('linestyle', None)
    if edgecolor is None:
        edgecolor = get_style_value("hist.edgecolor", "black")
    if linewidth is None:
        linewidth = get_style_value("hist.linewidth", 1.0)
    # Phase 13.12.DF v1.2: auto_title from style if not set per-call
    auto_title = resolve_auto_title(auto_title)

    # Phase 13.36.DF: pop 'marker' from kwargs BEFORE any rendering path.
    # 'marker' flows in here via _HIST_FORWARDED_NAMES (Phase 13.36 added it
    # to the tuple). matplotlib's ax.hist() does NOT accept marker — would
    # raise AttributeError on every hist render (vector path, non-grouped
    # path, grouped path). The pop must happen at draw_hist top, not deeper
    # inside _draw_hist_grouped (which is only reached for the grouped path).
    # markersize is NOT in _HIST_FORWARDED_NAMES (Sonet51 P1 from v1.2 review)
    # so it cannot arrive here via the vector path. No pop needed for it.
    _user_marker = kwargs.pop('marker', None)
    if _user_marker is not None:
        import warnings
        _ht_hint = histtype if histtype is not None else 'bar'
        warnings.warn(
            f"marker={_user_marker!r} has no effect on histograms "
            f"(histtype={_ht_hint!r} — ax.hist does not render markers). "
            f"Use color= to distinguish groups instead.",
            UserWarning, stacklevel=3
        )

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
    # Phase 13.39.DF CP1-4: detect datetime64 column BEFORE astype(float)
    _x_is_datetime = (
        isinstance(x, str) and x in df.columns
        and np.issubdtype(df[x].dtype, np.datetime64)
    )

    # Phase 13.40.DF M5 correctness guard: hist_errors + cumulative is
    # statistically wrong. Poisson per-bin errors assume independent counts;
    # cumulative counts have correlated uncertainty. Locked by §9.CH.6.
    if hist_errors and cumulative:
        raise NotImplementedError(
            "hist_errors=True and cumulative=True cannot be composed: "
            "Poisson per-bin errors assume independent counts; cumulative "
            "counts have correlated uncertainty (each bin's error depends "
            "on all prior bins). Use cumulative=True without "
            "hist_errors=True, or use a single-bin approach for the "
            "threshold of interest."
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
    # FIX1.FIX1 (Sonnet53_R2): single call — capture x_clean here, reuse below
    # to avoid the previous double-call when w_data is None.
    _x_clean, _, _sanitize_stats = sanitize_for_plot(
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
        # No weights: use the sanitized output directly (no double-call).
        x_data = _x_clean

    # Phase 13.39.DF (CP1-4): time_format pre-conversion with dtype auto-detect.
    # Convert x_data to matplotlib date numbers BEFORE ax.hist binning so bin
    # edges come out as date numbers. CRITICAL: post-hoc rewrite of ax.get_lines()
    # is a NO-OP for hist (creates Patch objects, not Line2D) — pre-conversion
    # is the only correct approach. Locked by §9.TA.5.
    if time_format is not None:
        import matplotlib.dates as mdates
        _x_arr = np.asarray(x_data)
        if np.issubdtype(_x_arr.dtype, np.datetime64):
            x_data = mdates.date2num(_x_arr)
        else:
            x_data = mdates.date2num(
                pd.to_datetime(_x_arr, unit='s').to_pydatetime()
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
    # Phase 13.42.DF FIX1 (B1/Sonet51): _facet_mode is set by the facet
    # dispatcher in drawer.py per-cell calls. Plumbed to render_fit_textbox so
    # it picks fit.text_fontsize_facet (default 7) instead of
    # fit.text_fontsize_default (default 9). Without this, the facet branch in
    # render_fit_textbox was permanently unreachable (v1.0 silent gap).
    _facet_mode = kwargs.pop('_facet_mode', False)
    
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
        #
        # Phase 13.35.DF (BUG-013 fix): float group_by + binning + per-group norm.

        col = df[group_by]

        # BUG-012 protection: float column with high cardinality and no bins.
        # Threshold 20 is a heuristic — see BUG-012 report for rationale.
        # Catches the silent-explosion case where group_by='z' on a continuous
        # float column produces hundreds of one-entry groups.
        if (col.dtype.kind == 'f'
                and group_by_bins is None
                and group_by_quantiles is None
                and col.nunique() > 20):
            raise ValueError(
                f"group_by='{group_by}' is a float column with "
                f"{col.nunique()} unique values. "
                f"Add group_by_bins=N or group_by_quantiles=N to bin it. "
                f"Example: group_by_bins=5 or group_by_quantiles=5."
            )

        # Float binning: replace group_by column values with pd.Interval labels.
        # Use df.copy() to avoid modifying the caller's DataFrame.
        if group_by_bins is not None or group_by_quantiles is not None:
            df = df.copy()
            if col.dtype == np.float16:
                # Match profile path pattern (drawer.py:2585): pd.cut/qcut
                # don't accept float16 directly; upcast.
                df[group_by] = df[group_by].astype(np.float32)
            if group_by_bins is not None:
                df[group_by] = pd.cut(df[group_by], bins=group_by_bins)
            else:
                df[group_by] = pd.qcut(
                    df[group_by], q=group_by_quantiles, duplicates='drop'
                )

        # Shared bin edges computed from sanitized x_data.
        # x_data is fully sanitized (nan_policy applied) at lines 259-307.
        # Do NOT use df[x].dropna() here — that would bypass the nan_policy
        # sanitization already applied. (v1.1 P1-B from review panel.)
        _bins_for_edges = bins if bins is not None else 100
        _, shared_edges = np.histogram(
            x_data, bins=_bins_for_edges, range=_used_range
        )

        n_rendered = _draw_hist_grouped(
            df, x, ax, group_by, top_k, stacked,
            bin_edges=shared_edges,
            hist_norm=hist_norm,
            min_entries=min_entries,
            # Phase 13.36.DF: forward user color from draw_hist's local variable
            # (consumed by explicit signature at line ~157, NOT in **kwargs).
            # Without this, the user's color= silently no-ops in the grouped path
            # (v1.0 P1-A pattern). marker= still flows through **kwargs to
            # _draw_hist_grouped where it's popped + warned.
            _user_color=color,
            # Phase 13.37.DF: forward user-explicit edgecolor sentinel (captured
            # BEFORE the style fill-in above per Phase 13.36 Edit 17 pattern),
            # hist_errors flag, linestyle_cycle flag + user linestyle sentinel,
            # and pre-computed weights for weighted-Poisson errors.
            _user_edgecolor=_ud_user_edgecolor,
            hist_errors=hist_errors,
            linestyle_cycle=linestyle_cycle,
            _user_linestyle=_ud_user_linestyle,
            _hist_weights_arr=_hist_weights,
            density=density, weights=_hist_weights,
            alpha=alpha, histtype=histtype, edgecolor=edgecolor,
            linewidth=linewidth,
            # Phase 13.40.DF CP1-2: cumulative forwarded explicitly to the
            # grouped path (covers BOTH stacked branch at ~line 803 AND the
            # 2 overlaid branches at ~847/853). Without this, all 3 grouped
            # call sites silently drop cumulative.
            cumulative=cumulative,
            **kwargs   # group_by_bins/hist_norm/min_entries already consumed
        )
        stats_dict["grouped"] = True
        stats_dict["n_groups"] = n_rendered   # was missing — T1 observation
    else:
        # Single histogram
        ax.hist(
            x_data, bins=bins, range=_used_range, density=density, weights=_hist_weights,
            color=color, alpha=alpha, histtype=histtype, edgecolor=edgecolor,
            linewidth=linewidth, label=label,
            # Phase 13.40.DF: cumulative histogram (explicit forward — call site 1/4)
            cumulative=cumulative,
            **kwargs
        )
        # Phase 13.37.DF: Poisson error bar overlay for ungrouped path.
        # CP1-7: use edges from np.histogram() return (bins= may be int).
        # CP1-6: weighted Poisson via Σw² when weights= column is set.
        # CP1-8: per-bin density formula (vectorized np.diff(edges)).
        if hist_errors:
            if _hist_weights is not None:
                _w = np.asarray(_hist_weights)
                sum_w, _edges = np.histogram(x_data, bins=bins, range=_used_range,
                                             weights=_w)
                sum_w2, _ = np.histogram(x_data, bins=bins, range=_used_range,
                                         weights=_w**2)
                total_w = float(_w.sum()) if len(_w) > 0 else 1.0
                if norm == "probability":
                    heights = sum_w / total_w
                    errs = np.sqrt(sum_w2) / total_w
                elif norm == "density" or density:
                    bws = np.diff(_edges)
                    heights = sum_w / (total_w * bws)
                    errs = np.sqrt(sum_w2) / (total_w * bws)
                else:
                    heights = sum_w.astype(float)
                    errs = np.sqrt(sum_w2)
                counts_for_mask = sum_w
            else:
                counts, _edges = np.histogram(x_data, bins=bins, range=_used_range)
                n_total = len(x_data)
                if norm == "probability":
                    heights = counts / n_total
                    errs = np.sqrt(counts) / n_total
                elif norm == "density" or density:
                    bws = np.diff(_edges)
                    heights = counts / (n_total * bws)
                    errs = np.sqrt(counts) / (n_total * bws)
                else:
                    heights = counts.astype(float)
                    errs = np.sqrt(counts)
                counts_for_mask = counts
            bin_centers = 0.5 * (_edges[:-1] + _edges[1:])
            mask = counts_for_mask > 0
            # Phase 13.37.DF: error bar color matches bars (color is the local
            # variable; for single-hist ungrouped path, it's either the user's
            # explicit color or matplotlib's default — both fine).
            ax.errorbar(bin_centers[mask], heights[mask], yerr=errs[mask],
                        fmt='none', color=color,
                        elinewidth=get_style_value("hist.error_elinewidth", 1.0),
                        capsize=get_style_value("hist.error_capsize", 2),
                        zorder=3)

    # ========================================================================
    # Phase 13.42.DF: Inline fits — apply AFTER ax.hist. curves_list uses dict
    # form per CRR §2 D2 (panel consensus on N1 from v1.4 review).
    #
    # FIX1 (B4/Sonnet55 + D9/R4 + B5/R1):
    #   - B4: previous block used `df[df[group_by] == g][x_name].dropna()` which
    #     silently returned empty for some group keys (Sonnet55 generalized:
    #     re-filtering is structurally fragile; also affected top_k/sort_groups).
    #     Fix: reuse the same df + group_by column the main path operates on
    #     (already pd.Interval-categorized by the upstream pd.qcut/pd.cut at
    #     line ~547). Use Series.eq for Interval-safe comparison and dropna()
    #     to skip NaN-binned rows.
    #   - D9/R4: previous block had `if group_by is not None and not stacked`.
    #     Per architect 2026-05-26: stacked + group_by + fit → per-group fits
    #     (same as unstacked). Stacking is purely visual. Removed `not stacked`
    #     guard.
    #   - B5/R1: yerr = sqrt(max(counts, 1)) ALWAYS (ROOT TH1::Fit Neyman
    #     convention). hist_errors flag controls display errorbars only.
    # ========================================================================
    if fit is not None:
        if group_by is not None:
            # Per-group fits → stats['fit'] dict keyed by group_val per §3.5.
            # D9/R4: applies regardless of stacked (was guarded).
            fits_dict = {}
            x_name_for_groupby = x if isinstance(x, str) else x_name
            # B4: read groups from the SAME df that the main path used. By the
            # time we reach here, pd.qcut/pd.cut (line ~547) has already
            # replaced df[group_by] with Interval categorical values. dropna()
            # skips NaN-binned rows (quantile edge effects).
            _gb_series = df[group_by].dropna()
            groups_for_fit = _gb_series.unique().tolist()
            if top_k is not None and len(groups_for_fit) > top_k:
                vc = _gb_series.value_counts()
                groups_for_fit = vc.head(top_k).index.tolist()
            for g in groups_for_fit:
                # B4: Series.eq is Interval-safe; df[df[group_by]==g] was the
                # v1.0 form which silently mis-matched some Intervals.
                # We use .eq() then explicitly select rows with True, matching
                # main-path masking semantics.
                _group_mask = df[group_by].eq(g)
                gdata_raw = df.loc[_group_mask, x_name_for_groupby].dropna()
                gdata = gdata_raw.values.astype(float) if hasattr(gdata_raw, 'values') else np.asarray(gdata_raw, dtype=float)
                if gdata.size == 0:
                    # NEW status value per CRR §2 disclosure (I-7):
                    # 'skipped_empty' surfaces empty-after-mask cases cleanly
                    # instead of silently failing inside curve_fit.
                    fits_dict[g] = [[{
                        'fit_name': fit if isinstance(fit, str) else 'unknown',
                        'params': np.array([], dtype=float),
                        'param_names': [],
                        'param_errors': np.array([], dtype=float),
                        'pcov': np.zeros((0, 0), dtype=float),
                        'chi2': float('nan'),
                        'ndf': 0,
                        'redchi': float('nan'),
                        'function': None,
                        'x_range': (float('nan'), float('nan')),
                        'n_data': 0,
                        'fit_status': 'skipped_empty',
                        'fit_error': 'group has no data points after masking',
                        'fit_spec': fit if isinstance(fit, dict) else {'fun': fit},
                    }]]
                    continue
                counts_g, edges_g = np.histogram(gdata, bins=bins, range=_used_range, density=False)
                centers_g = 0.5 * (edges_g[:-1] + edges_g[1:])
                # B5/R1: Poisson always (max(counts, 1) per ROOT)
                yerr_g = np.sqrt(np.maximum(counts_g, 1))
                curve_g = {
                    'x_data': centers_g,
                    'y_data': counts_g.astype(float),
                    'yerr_data': yerr_g,
                    'color': None,
                    'label': str(g),
                }
                normalized_g = normalize_fit_spec(fit, 1)
                curve_fits_g = [
                    dispatch_fit(curve_g['x_data'], curve_g['y_data'], fd,
                                 yerr=curve_g['yerr_data'], plot_kind='hist')
                    for fd in normalized_g[0]
                ]
                fits_dict[g] = [curve_fits_g]
                render_fit_overlays(ax, [curve_g], [curve_fits_g])
            # Single combined textbox covering all groups
            if fits_dict:
                all_curves_combined = []
                all_fits_combined = []
                for g, group_fits in fits_dict.items():
                    for curve_fits in group_fits:
                        all_curves_combined.append({'label': str(g), 'color': None,
                                                    'x_data': np.array([]), 'y_data': np.array([])})
                        all_fits_combined.append(curve_fits)
                render_fit_textbox(ax, all_curves_combined, all_fits_combined, facet_mode=_facet_mode, textbox_kwargs=fit_textbox_kwargs)
            stats_dict['fit'] = fits_dict
        else:
            # Ungrouped path — single curve, fit per §3.5 list-of-lists
            counts_fit, edges_fit = np.histogram(x_data, bins=bins, range=_used_range,
                                                  weights=_hist_weights, density=False)
            bin_centers_fit = 0.5 * (edges_fit[:-1] + edges_fit[1:])
            # Phase 13.42.DF FIX1 (B5/R1): Poisson default per ROOT TH1::Fit Neyman
            # convention. v1.0's `hist_errors` gating made χ² meaningless (yerr=None
            # → curve_fit minimizes Σresid² as if yerr=1; reported χ² scales with N²).
            # `hist_errors` now controls *display* errorbars only, NOT fit yerr.
            # max(counts, 1) matches ROOT precisely for empty bins.
            #
            # Phase 13.42.DF FIX2 (I-8, Sonnet53_R2 v1.0 panel finding, v1.2 §8
            # B5(c) commitment): when user has weights= AND fit=, χ² uses
            # sqrt(counts) (UNWEIGHTED Neyman) not sqrt(Σw²) (weighted Poisson).
            # The errorbar display path (Phase 13.37 CP1-6) DOES use Σw², but
            # the fit path does not — that's a known limitation. Emit a
            # one-time UserWarning so analysts choosing weighted hist for QA
            # know the reported chi² treats weights as if they were counts.
            if _hist_weights is not None:
                import warnings as _warnings
                _warnings.warn(
                    "Phase 13.42.DF FIX2 (I-8): fit= combined with weights= "
                    "uses sqrt(counts) Neyman errors, NOT sqrt(Σw²) weighted "
                    "Poisson. The reported χ²/ndf treats weighted bin contents "
                    "as if they were raw counts; if you need the weighted "
                    "Poisson formula, compute it manually from stats['fit'] "
                    "params and a separate np.histogram(weights=w**2, ...) "
                    "call. (FIX2 limitation; full sum-of-weights fit deferred "
                    "to a later phase.)",
                    UserWarning,
                    stacklevel=2,
                )
            yerr_hist = np.sqrt(np.maximum(counts_fit, 1))
            curve = {
                'x_data': bin_centers_fit,
                'y_data': counts_fit.astype(float),
                'yerr_data': yerr_hist,
                'color': color,
                'label': label,
            }
            curves_list_fit = [curve]
            normalized = normalize_fit_spec(fit, 1)
            curve_fits = [
                dispatch_fit(curve['x_data'], curve['y_data'], fd,
                             yerr=curve['yerr_data'], plot_kind='hist')
                for fd in normalized[0]
            ]
            fits_per_curve = [curve_fits]
            render_fit_overlays(ax, curves_list_fit, fits_per_curve)
            render_fit_textbox(ax, curves_list_fit, fits_per_curve, facet_mode=_facet_mode, textbox_kwargs=fit_textbox_kwargs)
            stats_dict['fit'] = fits_per_curve

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

    # Phase 13.39.DF: apply time_format formatter AFTER render (pre-conversion
    # already happened upstream, line ~404 area). Locked by §9.TA.5 (realistic
    # timestamps) and §9.TA.7 (datetime64[ns] no-crash).
    if time_format is not None:
        import matplotlib.dates as mdates
        if time_format == "auto":
            ax.xaxis.set_major_formatter(
                mdates.AutoDateFormatter(mdates.AutoDateLocator()))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(time_format))
        fig.autofmt_xdate()

    return fig, ax, stats_dict


def _draw_hist_grouped(
    df: pd.DataFrame,
    x: str,
    ax: plt.Axes,
    group_by: str,
    top_k: Optional[int],
    stacked: bool,
    bin_edges: Optional[np.ndarray] = None,   # Phase 13.35.DF: shared edges from full dataset
    hist_norm: Optional[str] = None,          # Phase 13.35.DF: None | "probability" | "density"
    min_entries: int = 0,                     # Phase 13.35.DF: skip groups below threshold
    # Phase 13.36.DF: user color override (None = use per-group palette cycle).
    # 'color' is consumed by draw_hist()'s explicit signature — must be passed
    # as a named param from there (not via **kwargs which doesn't contain it).
    _user_color: Optional[str] = None,
    # Phase 13.37.DF (BUG-014): user edgecolor override sentinel. None = "user
    # did not pass edgecolor; use group_color as step-line color when
    # histtype='step'". Non-None = "user explicitly passed; uniform override".
    # Same Edit 17 architectural pattern as _user_color.
    _user_edgecolor: Optional[str] = None,
    # Phase 13.37.DF: Poisson error bar overlay flag.
    hist_errors: bool = False,
    # Phase 13.37.DF: per-group linestyle cycling mode flag (composes with
    # _user_linestyle sentinel: user explicit > cycle > style default).
    linestyle_cycle: bool = False,
    _user_linestyle: Optional[str] = None,
    # Phase 13.37.DF: pre-computed per-row weights (forwarded from draw_hist
    # for hist_errors weighted-Poisson computation in the grouped path).
    _hist_weights_arr: Optional[np.ndarray] = None,
    # Phase 13.40.DF CP1-2: cumulative histogram. Recursive named-param
    # forwarding per QRC v1.32 #6 — DFDraw.hist → draw_hist → _draw_hist_grouped
    # → ax.hist. NEVER access via kwargs.get/**hist_kwargs.
    cumulative: Union[bool, int] = False,
    **hist_kwargs
) -> int:
    """Draw grouped/overlaid histograms.

    Phase 13.35.DF: extended for float group_by binning + per-group normalization.
    Phase 13.36.DF: extended for user color override + marker UserWarning.
    Phase 13.37.DF: extended for step edgecolor sentinel, hist_errors,
                    linestyle_cycle.
    Returns the number of groups actually rendered (post min_entries filter).
    """
    import matplotlib.pyplot as plt

    # Phase 13.36.DF: 'marker' is popped earlier in draw_hist() body, so it
    # cannot arrive here in hist_kwargs. The UserWarning fires once at the
    # draw_hist level regardless of grouped/non-grouped routing.

    # Phase 13.35.DF: pop 'weights' from hist_kwargs — the grouped path uses
    # per-group hist_norm weights (from _group_weights), not the routing
    # block's _hist_weights (which is None on this path anyway because
    # group_by + column-name weights raises NotImplementedError earlier).
    # Without this pop, ax.hist sees 'weights' twice (TypeError).
    hist_kwargs.pop('weights', None)

    # Get groups (pd.Interval objects when group_by_bins/_quantiles was used;
    # scalar values otherwise).
    # Phase 13.37.DF (BUG-016): sort by _interval_sort_key so pd.Interval groups
    # appear in numeric order in the legend (not lexicographic). Without this,
    # bins crossing 10 render legend as "(0.04, 0.83], (10.0, 12.0], (2.0, 4.0]"
    # because '(1' < '(2' lexicographically. Categorical string groups are
    # unaffected — _interval_sort_key falls through to string order for those.
    groups = sorted(df[group_by].unique(), key=_interval_sort_key)

    # Top-K filtering (existing behavior)
    if top_k is not None and len(groups) > top_k:
        counts = df[group_by].value_counts()
        top_groups = counts.head(top_k).index.tolist()
        groups = top_groups

    # Color palette (existing behavior)
    palette_name = get_style_value("colors.palette", "tab10")
    palette = plt.colormaps.get_cmap(palette_name)
    colors = [palette(i % 10) for i in range(len(groups))]

    # Use shared edges if provided; fall back to matplotlib auto-bin (or kwarg)
    bins_arg = bin_edges if bin_edges is not None else hist_kwargs.pop('bins', 100)

    # Phase 13.36.DF: warn ONCE per call when user color= makes all groups
    # indistinguishable. Fires before either render branch.
    if _user_color is not None and len(groups) > 1:
        import warnings
        warnings.warn(
            f"color={_user_color!r} applied uniformly to all {len(groups)} groups. "
            f"Groups will be indistinguishable by color. "
            f"Omit color= to use per-group colors (default behavior).",
            UserWarning, stacklevel=3
        )

    # Phase 13.37.DF: resolve linestyle cycle list (Phase 13.26 style key).
    # Only used when linestyle_cycle=True AND user did not pass linestyle=.
    _ls_cycle = get_style_value("channels.cycles.linestyle", ['-', '--', '-.', ':'])

    # Phase 13.37.DF: peek at histtype for step-mode edgecolor logic.
    # 'histtype' is in hist_kwargs (passed explicitly from draw_hist body),
    # so use get() not pop() — ax.hist still needs it.
    _histtype = hist_kwargs.get('histtype', 'bar')

    if stacked:
        # One-pass loop: build data_list, labels, AND surviving_colors in lockstep
        # so all three stay aligned when min_entries filters drop groups.
        # v1.2 used two-pass list comprehensions (data_list filtered, then labels
        # zipped against UNFILTERED groups) → misaligned labels.
        # That was v1.2 P1-D — Hard Constraint §3 silent wrong result.
        # P3 (color-shift) also fixed here: pre-Phase 13.35.DF, colors[:M] gave
        # sequential tab10 colors, not the original-index color per surviving
        # group; surviving_colors[i] preserves the original colors[i] mapping.
        # Phase 13.36.DF: when _user_color is set, surviving_colors becomes
        # uniform (user wants all groups same color — already warned above).
        # Phase 13.37.DF: hist_errors not currently supported in stacked mode
        # (semantically unclear — error of which stack layer?). Linestyle_cycle
        # also skipped in stacked mode (one ax.hist call shared across stacks).
        data_list, labels, surviving_colors = [], [], []
        for i, g in enumerate(groups):
            d = df[df[group_by] == g][x].dropna().values.astype(float)
            if len(d) >= min_entries:
                data_list.append(d)
                # Label: str(g) handles both scalars and pd.Interval objects.
                # _format_interval_label is defined in profile.py and NOT
                # imported here. (v1.1 P1-A from review panel.)
                labels.append(str(g))
                # Phase 13.36.DF: user override > palette.
                surviving_colors.append(
                    colors[i] if _user_color is None else _user_color
                )
        if not data_list:
            return 0
        # Phase 13.37.DF (BUG-014): for stacked + step mode, mirror the
        # edgecolor sentinel. User explicit > group_color (matches the bars).
        if _histtype == 'step' and _user_edgecolor is None:
            # Step lines use surviving_colors (parallel to the fill colors).
            # ax.hist with stacked=True uses 'edgecolor' kwarg as list-of-N.
            _stacked_ec = surviving_colors
        else:
            # User-explicit edgecolor (uniform) OR bar/stepfilled (use style default)
            _stacked_ec = _user_edgecolor if _user_edgecolor is not None else hist_kwargs.pop('edgecolor', None)
        # Strip 'edgecolor' from hist_kwargs to avoid passing twice
        hist_kwargs.pop('edgecolor', None)
        ax.hist(data_list, bins=bins_arg, label=labels,
                color=surviving_colors, edgecolor=_stacked_ec,
                stacked=True,
                # Phase 13.40.DF CP1-1: cumulative forwarded explicitly (call
                # site 2/4 — stacked branch). v1.1 spec missed this site;
                # added in v1.2 per Sonet50 panel. Locked by §9.CH.10.
                cumulative=cumulative,
                **hist_kwargs)
        return len(data_list)
    else:
        # Overlaid histograms — one ax.hist call per surviving group.
        # colors[i] preserves original-index color when groups are skipped
        # (overlaid branch was already correct in baseline; documented for parity).
        # Phase 13.36.DF: user override > palette per group.
        # Phase 13.37.DF: per-group edgecolor (BUG-014), linestyle_cycle,
        # and hist_errors Poisson overlay.

        # Pop edgecolor from hist_kwargs so we can override per group.
        # Original value preserved in _style_edgecolor for non-step paths.
        _style_edgecolor = hist_kwargs.pop('edgecolor', None)

        n_rendered = 0
        for i, group in enumerate(groups):
            # BUG_dfdraw_20260505: cast to float for boolean expressions
            group_mask = df[group_by] == group
            group_data = df[group_mask][x].dropna().values.astype(float)
            if len(group_data) < min_entries:
                continue
            weights = _group_weights(group_data, bin_edges, hist_norm)
            # Phase 13.36.DF: user override > palette
            group_color = colors[i] if _user_color is None else _user_color

            # Phase 13.37.DF (BUG-014): step-mode edgecolor sentinel.
            # User explicit > group_color (when step) > style default (when bar).
            if _histtype == 'step' and _user_edgecolor is None:
                _ec = group_color
            else:
                # User explicit OR non-step mode → use captured/style value.
                _ec = _user_edgecolor if _user_edgecolor is not None else _style_edgecolor

            # Phase 13.37.DF: linestyle_cycle priority — user explicit > cycle.
            # _ud_user_linestyle was peeked from kwargs in draw_hist (not popped),
            # so it's still in hist_kwargs and ax.hist receives it via **hist_kwargs.
            # When cycle mode is active AND user did not pass linestyle, override
            # by popping from kwargs and passing explicitly.
            if linestyle_cycle and _user_linestyle is None:
                # User did NOT pass linestyle → safe to override
                hist_kwargs.pop('linestyle', None)   # idempotent
                _per_group_ls = _ls_cycle[i % len(_ls_cycle)]
                ax.hist(group_data, bins=bins_arg,
                        label=str(group), color=group_color,
                        edgecolor=_ec, linestyle=_per_group_ls,
                        weights=weights,
                        # Phase 13.40.DF: cumulative (call site 3/4 — overlaid
                        # linestyle_cycle path. Phase 13.37 split overlaid into
                        # 2 branches; both need explicit forward.)
                        cumulative=cumulative,
                        **hist_kwargs)
            else:
                # No cycle OR user explicit → existing flow (linestyle in **kwargs)
                ax.hist(group_data, bins=bins_arg,
                        label=str(group), color=group_color,
                        edgecolor=_ec,
                        weights=weights,
                        # Phase 13.40.DF: cumulative (call site 4/4 — overlaid
                        # default path).
                        cumulative=cumulative,
                        **hist_kwargs)

            # Phase 13.37.DF: Poisson error bar overlay (CP1-1/CP1-2 fix:
            # color=group_color follows Phase 13.36 sentinel, NOT colors[i]).
            if hist_errors:
                if _hist_weights_arr is not None:
                    # Weighted Poisson: variance per bin = Σw² (CP1-6 fix)
                    _w = np.asarray(_hist_weights_arr)[group_mask.values]
                    # Sanitize to match group_data
                    _w = _w[~np.isnan(_w)][:len(group_data)] if len(_w) >= len(group_data) else _w
                    sum_w, _edges = np.histogram(group_data, bins=bins_arg, weights=_w)
                    sum_w2, _ = np.histogram(group_data, bins=bins_arg, weights=_w**2)
                    total_w = float(_w.sum()) if len(_w) > 0 else 1.0
                    if hist_norm == "probability":
                        heights = sum_w / total_w
                        errs = np.sqrt(sum_w2) / total_w
                    elif hist_norm == "density":
                        bws = np.diff(_edges)
                        heights = sum_w / (total_w * bws)
                        errs = np.sqrt(sum_w2) / (total_w * bws)
                    else:
                        heights = sum_w.astype(float)
                        errs = np.sqrt(sum_w2)
                    counts_for_mask = sum_w
                else:
                    counts, _edges = np.histogram(group_data, bins=bins_arg)
                    n_total = len(group_data)
                    if hist_norm == "probability":
                        heights = counts / n_total
                        errs = np.sqrt(counts) / n_total
                    elif hist_norm == "density":
                        # Per-bin width (CP1-8 fix — was mean width in v1.0)
                        bws = np.diff(_edges)
                        heights = counts / (n_total * bws)
                        errs = np.sqrt(counts) / (n_total * bws)
                    else:
                        heights = counts.astype(float)
                        errs = np.sqrt(counts)
                    counts_for_mask = counts
                bin_centers = 0.5 * (_edges[:-1] + _edges[1:])
                mask = counts_for_mask > 0
                ax.errorbar(bin_centers[mask], heights[mask], yerr=errs[mask],
                            fmt='none',
                            color=group_color,   # Phase 13.36 sentinel — CP1-2 fix
                            elinewidth=get_style_value("hist.error_elinewidth", 1.0),
                            capsize=get_style_value("hist.error_capsize", 2),
                            zorder=3)

            n_rendered += 1   # count post-skip (v1.1 P1-C from review panel)
        if n_rendered > 0:
            ax.legend()
        return n_rendered


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
