"""Phase 13.43.DF — Summary fit (standalone figures for fit results).

Produces new matplotlib Figure objects from stats['fit'] (Phase 13.42 canonical
list-of-lists / dict-of-list-of-lists / faceted-tuple-keyed dict). Never
modifies the caller's main figure. Returns figures that the caller attaches
to stats['summary_fit'] dict.

Key entry points
----------------
- ``render_summary_fit(stats_fit, *, kinds, ...) -> (figs, data, note)``:
  the public entry called from DFDraw.{hist,profile,scatter,draw} outer
  layer AFTER ``_dispatch_faceted_render`` returns.
- ``_normalize_summary_fit_spec(value) -> dict``: validates and normalizes
  the user-supplied ``summary_fit=`` kwarg per v1.2 §3.8 error grammar.

See `PHASE_13_43_DF_v1_2_SummaryFit_Proposal.md` §4.1 / §4.1.1 / §4.2.
"""

from typing import Optional, Union, Dict, List, Tuple, Any
import warnings

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# Public API — render_summary_fit
# ============================================================================

def render_summary_fit(
    stats_fit: Any,
    *,
    kinds: List[str],
    group_by_col: Optional[str] = None,
    facet_by_cols: Optional[List[str]] = None,
    params: Optional[List[str]] = None,
    mode: str = 'subplots',
    annotate: Union[bool, str] = False,
    precision: int = 2,
    columns: Optional[List[str]] = None,
    data_format: str = 'dict',
    title: Union[str, None] = 'auto',
    title_overflow: str = 'shrink',
    style: Optional[Any] = None,
    expr_for_auto_title: Optional[str] = None,
) -> Tuple[Dict[str, plt.Figure], Any, Optional[str]]:
    """Render the requested summary fits as standalone figures.

    Parameters
    ----------
    stats_fit : list | dict
        Phase 13.42 canonical fit container. See §4.1.1 for the 3 shapes:
          - Shape 1: ungrouped, list-of-lists ``[[fit_dict]]``
          - Shape 2: grouped unfaceted, ``{scalar_key: [[fit_dict, ...]]}``
          - Shape 3: faceted, ``{tuple_key: ...}`` where inner is either
                     a Shape-2-like dict (faceted + group_by) or a list
                     (faceted, no group_by).
    kinds : list[str]
        Subset of {'table', 'figure'}.
    group_by_col, facet_by_cols : optional
        Echo of the call-site group/facet column names for display.
    params : optional list[str]
        Subset of param names to render in the params figure (default: all).
    mode : 'subplots' (default) | 'overlay'
        Params figure layout mode (§3.5).
    annotate : bool | 'always'
        Annotate-numeric-value-at-each-point flag (§3.6). Auto-suppressed
        above 15 points unless 'always'.
    precision : 1 | 2 | 3
        Significant digits in numeric formatting (§3.7).
    columns : optional list[str]
        Subset of fit-param + metric columns to show in the table.
    data_format : 'dict' (default) | 'pandas'
        Data-key format inside ``stats['summary_fit']['data']`` (§3.10).
    title : str | None | 'auto'
        - 'auto'  : auto-title ``f"fit params: {fit_name} — {expr}"`` (§3.9)
        - None    : no title
        - other str: literal title.
    title_overflow : 'shrink' | 'truncate' | 'wrap'
        How to handle title at font_size_min (§3.9).
    style : optional dict-like
        Access object for ``style['summary_fit.*']`` keys. None → use
        built-in defaults.
    expr_for_auto_title : optional
        The original expression string (e.g., "y:x"), used to build the
        auto-title. None → caller didn't pass it; auto-title becomes
        just ``f"fit params: {fit_name}"``.

    Returns
    -------
    figs : dict
        Maps kind name to Figure object. Possible keys: 'table', 'figure'.
        Either / both / neither may be present depending on what
        renderable content was available.
    data : list[dict] | pandas.DataFrame
        Flattened row representation suitable for further processing.
    note : str | None
        Scenario-E diagnostic when nothing was rendered. None when at
        least one figure was produced.
    """
    figs: Dict[str, plt.Figure] = {}
    note: Optional[str] = None

    # Flatten first; the row list is shared by table and figure renderers.
    rows = _flatten_to_rows(stats_fit, group_by_col, facet_by_cols)

    if not rows:
        # Scenario E: kwarg requested but no fits to render.
        return ({}, _format_data([], data_format), 
                "summary_fit requested but no fit rows to render "
                "(stats['fit'] empty or all cells produced no fit_dicts).")

    if 'table' in kinds:
        figs['table'] = _render_table_figure(
            rows, columns=columns, precision=precision,
            title=title, title_overflow=title_overflow,
            expr_for_auto_title=expr_for_auto_title, style=style,
        )

    if 'figure' in kinds:
        params_fig = _render_params_figure(
            rows, params=params, mode=mode, annotate=annotate,
            precision=precision, group_by_col=group_by_col,
            facet_by_cols=facet_by_cols,
            title=title, title_overflow=title_overflow,
            expr_for_auto_title=expr_for_auto_title, style=style,
        )
        if params_fig is not None:
            figs['figure'] = params_fig
        else:
            note = ("summary_fit='figure' requires either group_by or "
                    "facet_by to define an x-axis; neither was given. "
                    "Params figure silently omitted.")

    data = _format_data(rows, data_format)
    return figs, data, note


# ============================================================================
# §3.8 — Spec normalization and error grammar
# ============================================================================

_ALLOWED_KIND_STRINGS = {'table', 'figure', 'both'}
_ALLOWED_DICT_KEYS = {
    'kind', 'params', 'mode', 'annotate', 'precision', 'columns',
    'data_format', 'title', 'title_overflow',
}
_ALLOWED_MODES = {'subplots', 'overlay'}
_ALLOWED_PRECISIONS = {1, 2, 3}
_ALLOWED_DATA_FORMATS = {'dict', 'pandas'}
_ALLOWED_TITLE_OVERFLOWS = {'shrink', 'truncate', 'wrap'}


def _normalize_summary_fit_spec(value: Any) -> Dict[str, Any]:
    """Validate user's ``summary_fit=`` input; return canonical dict spec.

    Per v1.2 §3.1 the user may pass:
      - str: 'table' | 'figure' | 'both'
      - list[str]: subset of {'table', 'figure'}
      - dict: with required 'kind' key + optional config
      - None: caller should not even invoke this (filtered upstream)

    Returns canonical form:
        {'kinds': List[str], 'params': ..., 'mode': ..., 'annotate': ...,
         'precision': ..., 'columns': ..., 'data_format': ...,
         'title': ..., 'title_overflow': ...}

    Raises ValueError with §3.8 error grammar on unknown values.
    """
    canonical: Dict[str, Any] = {
        'kinds': None,
        'params': None,
        'mode': 'subplots',
        'annotate': False,
        'precision': 2,
        'columns': None,
        'data_format': None,
        'title': 'auto',
        'title_overflow': 'shrink',
    }

    if isinstance(value, str):
        if value not in _ALLOWED_KIND_STRINGS:
            raise ValueError(
                f"Unknown summary_fit kind {value!r}. "
                f"Allowed: 'table', 'figure', 'both', or list "
                f"['table','figure']. Fix: summary_fit='table'."
            )
        canonical['kinds'] = (['table', 'figure'] if value == 'both'
                              else [value])
        return canonical

    if isinstance(value, list):
        bad = [v for v in value if v not in ('table', 'figure')]
        if bad:
            raise ValueError(
                f"Unknown summary_fit kind(s) in list: {bad!r}. "
                f"Allowed entries: 'table', 'figure'. "
                f"Fix: summary_fit=['table','figure']."
            )
        canonical['kinds'] = list(dict.fromkeys(value))  # dedup + preserve order
        if not canonical['kinds']:
            raise ValueError(
                "summary_fit list is empty. "
                "Fix: summary_fit='table' or summary_fit=['table','figure']."
            )
        return canonical

    if isinstance(value, dict):
        if 'kind' not in value:
            raise ValueError(
                "summary_fit dict requires 'kind' key. "
                "Fix: summary_fit={'kind': 'table', 'precision': 3}."
            )
        unknown = set(value) - _ALLOWED_DICT_KEYS
        if unknown:
            raise ValueError(
                f"Unknown summary_fit dict key(s) {sorted(unknown)!r}. "
                f"Allowed: {sorted(_ALLOWED_DICT_KEYS)!r}. "
                f"Fix: remove unknown keys or check spelling."
            )
        kind = value['kind']
        if isinstance(kind, str):
            if kind not in _ALLOWED_KIND_STRINGS:
                raise ValueError(
                    f"Unknown summary_fit kind {kind!r}. "
                    f"Allowed: 'table', 'figure', 'both'. "
                    f"Fix: summary_fit={{'kind': 'table'}}."
                )
            canonical['kinds'] = (['table', 'figure'] if kind == 'both'
                                  else [kind])
        elif isinstance(kind, list):
            bad = [v for v in kind if v not in ('table', 'figure')]
            if bad:
                raise ValueError(
                    f"Unknown summary_fit kind(s) in list: {bad!r}. "
                    f"Allowed: 'table', 'figure'. "
                    f"Fix: 'kind': ['table','figure']."
                )
            canonical['kinds'] = list(dict.fromkeys(kind))
        else:
            raise ValueError(
                f"summary_fit dict 'kind' must be a string or list "
                f"(got {type(kind).__name__}). "
                f"Fix: 'kind': 'table' or 'kind': ['table','figure']."
            )

        # Optional fields
        if 'params' in value:
            p = value['params']
            if p is not None and not isinstance(p, list):
                raise ValueError(
                    f"summary_fit 'params' must be None or list[str] "
                    f"(got {type(p).__name__}). "
                    f"Fix: 'params': ['sigma','center']."
                )
            canonical['params'] = p
        if 'mode' in value:
            if value['mode'] not in _ALLOWED_MODES:
                raise ValueError(
                    f"Unknown summary_fit mode {value['mode']!r}. "
                    f"Allowed: 'subplots' (default), 'overlay'. "
                    f"Fix: 'mode': 'subplots'."
                )
            canonical['mode'] = value['mode']
        if 'annotate' in value:
            a = value['annotate']
            if a not in (True, False, 'always'):
                raise ValueError(
                    f"summary_fit 'annotate' must be True, False, or "
                    f"'always' (got {a!r}). "
                    f"Fix: 'annotate': True."
                )
            canonical['annotate'] = a
        if 'precision' in value:
            if value['precision'] not in _ALLOWED_PRECISIONS:
                raise ValueError(
                    f"summary_fit precision must be 1, 2, or 3 "
                    f"(got {value['precision']!r}). "
                    f"Fix: 'precision': 2."
                )
            canonical['precision'] = value['precision']
        if 'columns' in value:
            c = value['columns']
            if c is not None and not isinstance(c, list):
                raise ValueError(
                    f"summary_fit 'columns' must be None or list[str] "
                    f"(got {type(c).__name__}). "
                    f"Fix: 'columns': ['amplitude','sigma']."
                )
            canonical['columns'] = c
        if 'data_format' in value:
            if value['data_format'] not in _ALLOWED_DATA_FORMATS:
                raise ValueError(
                    f"summary_fit data_format must be 'dict' or 'pandas' "
                    f"(got {value['data_format']!r}). "
                    f"Fix: 'data_format': 'dict'."
                )
            canonical['data_format'] = value['data_format']
        if 'title' in value:
            t = value['title']
            if t is not None and t != 'auto' and not isinstance(t, str):
                raise ValueError(
                    f"summary_fit 'title' must be None, 'auto', or str "
                    f"(got {type(t).__name__}). "
                    f"Fix: 'title': 'My QA plot'."
                )
            canonical['title'] = t
        if 'title_overflow' in value:
            if value['title_overflow'] not in _ALLOWED_TITLE_OVERFLOWS:
                raise ValueError(
                    f"summary_fit title_overflow must be one of "
                    f"{sorted(_ALLOWED_TITLE_OVERFLOWS)!r} "
                    f"(got {value['title_overflow']!r}). "
                    f"Fix: 'title_overflow': 'shrink'."
                )
            canonical['title_overflow'] = value['title_overflow']
        return canonical

    raise ValueError(
        f"summary_fit must be None, str, list, or dict "
        f"(got {type(value).__name__}). "
        f"Fix: summary_fit='table' or summary_fit={{'kind': 'table'}}."
    )


# ============================================================================
# §4.1.1 — _flatten_to_rows: shape-aware row construction
# ============================================================================

def _flatten_to_rows(stats_fit, group_by_col, facet_by_cols):
    """Flatten Phase 13.42 stats['fit'] into a uniform row list.

    Shape distinguished by:
      - list (Shape 1):                ungrouped, unfaceted
      - dict with scalar keys (Shape 2): grouped, unfaceted
      - dict with tuple  keys (Shape 3): faceted (Phase 13.41)
            sub-case (a): cell_value is dict → faceted + group_by
            sub-case (b): cell_value is list → faceted, no group_by

    Returns: list[dict[str, Any]] — row dicts with uniform keys.
    """
    rows: List[Dict[str, Any]] = []

    if isinstance(stats_fit, list):
        # Shape 1: ungrouped, unfaceted.
        for outer in stats_fit:
            if not _is_iterable_of_fits(outer):
                continue
            for fit_idx, fit_dict in enumerate(outer):
                rows.append(_make_row(
                    group=None, facet=None,
                    fit_name=fit_dict.get('fit_name', f'fit_{fit_idx}'),
                    fit_dict=fit_dict,
                ))

    elif isinstance(stats_fit, dict):
        sample_key = next(iter(stats_fit), None)
        if sample_key is None:
            return []

        if isinstance(sample_key, tuple):
            # Shape 3: faceted.
            for facet_key, cell_value in stats_fit.items():
                facet_display = _format_facet_key(facet_key, facet_by_cols)
                if isinstance(cell_value, dict):
                    # Sub-case (a): faceted + group_by.
                    for group_val, inner_list in cell_value.items():
                        for outer in inner_list:
                            if not _is_iterable_of_fits(outer):
                                continue
                            for fit_idx, fit_dict in enumerate(outer):
                                rows.append(_make_row(
                                    group=group_val, facet=facet_display,
                                    fit_name=fit_dict.get(
                                        'fit_name', f'fit_{fit_idx}'),
                                    fit_dict=fit_dict,
                                ))
                elif isinstance(cell_value, list):
                    # Sub-case (b): faceted, no group_by. P2-C lock.
                    for outer in cell_value:
                        if not _is_iterable_of_fits(outer):
                            continue
                        for fit_idx, fit_dict in enumerate(outer):
                            rows.append(_make_row(
                                group=None, facet=facet_display,
                                fit_name=fit_dict.get(
                                    'fit_name', f'fit_{fit_idx}'),
                                fit_dict=fit_dict,
                            ))
                else:
                    raise ValueError(
                        f"Shape 3 cell value has unexpected type "
                        f"{type(cell_value).__name__} at "
                        f"facet_key={facet_key!r}. "
                        f"Expected dict (faceted + group_by) or list "
                        f"(faceted, no group_by). "
                        f"Fix: check Phase 13.42 fit-pipeline output for "
                        f"this facet cell."
                    )
        else:
            # Shape 2: grouped, unfaceted.
            for group_val, inner_list in stats_fit.items():
                if not isinstance(inner_list, list):
                    continue
                for outer in inner_list:
                    if not _is_iterable_of_fits(outer):
                        continue
                    for fit_idx, fit_dict in enumerate(outer):
                        rows.append(_make_row(
                            group=group_val, facet=None,
                            fit_name=fit_dict.get(
                                'fit_name', f'fit_{fit_idx}'),
                            fit_dict=fit_dict,
                        ))
    else:
        raise ValueError(
            f"Unrecognized stats['fit'] shape: "
            f"{type(stats_fit).__name__}. "
            f"Expected: list (Shape 1, ungrouped), dict with scalar keys "
            f"(Shape 2, grouped unfaceted), or dict with tuple keys "
            f"(Shape 3, faceted). "
            f"Fix: check Phase 13.42 fit-pipeline output for this plot "
            f"type and group_by/facet_by configuration."
        )

    return rows


def _is_iterable_of_fits(outer):
    """True if `outer` looks like a sequence of fit_dicts (each a dict)."""
    if not isinstance(outer, (list, tuple)):
        return False
    return all(isinstance(d, dict) for d in outer)


def _format_facet_key(key, facet_by_cols):
    """Render a Shape-3 tuple key as a display string."""
    if not isinstance(key, tuple):
        return str(key)
    if facet_by_cols is None or len(facet_by_cols) != len(key):
        return '(' + ', '.join(str(v) for v in key) + ')'
    return ' '.join(f"{c}={v}" for c, v in zip(facet_by_cols, key))


def _make_row(*, group, facet, fit_name, fit_dict):
    """Build one row dict from a fit_dict, with consistent keys.

    Phase 13.42 inline-fit fit_dict layout (verified against fits.py):
      ``{'fit_name', 'params', 'param_names', 'param_errors', 'pcov',
         'chi2', 'ndf', 'redchi', 'function', 'x_range', 'n_data',
         'fit_status', 'fit_error', 'fit_spec', ...}``

    We flatten the parallel (params, param_names, param_errors) arrays into
    per-param columns ``<name>`` + ``<name>_err``. Metrics (chi2, redchi,
    ndf, fit_status, n_data) carry through. Internal-only keys (pcov,
    function, x_range, fit_spec, fit_error) are dropped — they're not
    useful in a table/figure summary view.

    Note (CRR §2.1 disclosure): adds an optional 'channel' key when the
    fit_dict carries one (Phase 13.42 FIX1 D5 per-channel pairing).
    """
    row: Dict[str, Any] = {
        'group':    group,
        'facet':    facet,
        'fit_name': fit_name,
    }
    if 'channel' in fit_dict:
        row['channel'] = fit_dict['channel']

    pnames = fit_dict.get('param_names')
    pvals  = fit_dict.get('params')
    perrs  = fit_dict.get('param_errors')
    pnames = list(pnames) if pnames is not None else []
    pvals  = list(pvals)  if pvals  is not None else []
    perrs  = list(perrs)  if perrs  is not None else []
    for i, name in enumerate(pnames):
        if i < len(pvals):
            row[name] = float(pvals[i]) if hasattr(pvals[i], '__float__') else pvals[i]
        if i < len(perrs):
            row[f"{name}_err"] = float(perrs[i]) if hasattr(perrs[i], '__float__') else perrs[i]

    for k in ('chi2', 'ndf', 'redchi', 'fit_status', 'n_data'):
        if k in fit_dict:
            row[k] = fit_dict[k]
    return row


def _format_data(rows, data_format):
    """Convert flat rows to user-facing data per §3.10."""
    if not data_format or data_format == 'dict':
        return list(rows)
    if data_format == 'pandas':
        try:
            import pandas as pd
        except ImportError:
            warnings.warn(
                "summary_fit data_format='pandas' requested but pandas is "
                "not installed; falling back to list[dict].",
                UserWarning, stacklevel=3,
            )
            return list(rows)
        return pd.DataFrame.from_records(rows)
    return list(rows)


# ============================================================================
# Table renderer
# ============================================================================

def _render_table_figure(
    rows, *, columns, precision, title, title_overflow,
    expr_for_auto_title, style,
) -> plt.Figure:
    """Render a pure-table figure: each row = one fit; cells = formatted values."""
    columns_to_show = columns or _default_columns(rows)

    # Figure size scales with row/col count.
    figsize_per_row = _style_get(style, 'summary_fit.table.figsize_per_row', 0.35)
    figsize_per_col = _style_get(style, 'summary_fit.table.figsize_per_col', 1.20)
    fig_w = max(6.0, len(columns_to_show) * figsize_per_col)
    fig_h = max(2.0, 0.7 + len(rows) * figsize_per_row + 0.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    # Format cells.
    cell_text: List[List[str]] = []
    for r in rows:
        row_cells = []
        for col in columns_to_show:
            row_cells.append(_format_cell(r, col, precision))
        cell_text.append(row_cells)

    table = ax.table(
        cellText=cell_text,
        colLabels=columns_to_show,
        cellLoc='center',
        loc='upper center',
    )
    table.auto_set_font_size(False)
    font_max = _style_get(style, 'summary_fit.table.font_size_max', 10)
    table.set_fontsize(font_max)
    table.scale(1.0, 1.2)

    _apply_title(fig, rows, title, title_overflow, expr_for_auto_title, style)
    fig.tight_layout()
    return fig


def _default_columns(rows):
    """Pick a sensible default set of columns covering all rows."""
    if not rows:
        return []
    # Preserve insertion order across all rows; skip None-valued meta cols
    # that aren't relevant (e.g., no group_by → drop 'group').
    seen: Dict[str, None] = {}
    for r in rows:
        for k in r:
            seen.setdefault(k, None)
    cols = list(seen.keys())
    # Drop meta cols that are uniformly None.
    def all_none(name):
        return all(r.get(name) is None for r in rows)
    return [c for c in cols if not all_none(c)]


def _format_cell(row, col, precision):
    """Format one cell — paired value/err combined when both present."""
    val = row.get(col)
    if val is None:
        return ''
    err_key = f"{col}_err"
    err = row.get(err_key) if not col.endswith('_err') else None
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        if err is not None and isinstance(err, (float, int, np.number)):
            return f"{_fmt_num(val, precision)} ± {_fmt_num(err, precision)}"
        return _fmt_num(val, precision)
    return str(val)


def _fmt_num(x, precision):
    """Sig-digit numeric format. precision ∈ {1, 2, 3}."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x == 0 or not np.isfinite(x):
        return f"{x:.{precision}g}" if np.isfinite(x) else str(x)
    return f"{x:.{precision}g}"


# ============================================================================
# Params figure renderer
# ============================================================================

def _render_params_figure(
    rows, *, params, mode, annotate, precision,
    group_by_col, facet_by_cols,
    title, title_overflow, expr_for_auto_title, style,
) -> Optional[plt.Figure]:
    """Render the parameter-trend figure per §3.4 conventions.

    Returns None when neither group_by nor facet_by is present (no x-axis).
    """
    if group_by_col is None and not facet_by_cols:
        return None

    param_names = params or _all_param_names(rows)
    if not param_names:
        return None
    n_params = len(param_names)

    subplot_w = _style_get(style, 'summary_fit.figure.subplot_width', 4.0)
    subplot_h = _style_get(style, 'summary_fit.figure.subplot_height', 3.0)

    if mode == 'overlay':
        fig, ax = plt.subplots(figsize=(max(6.0, subplot_w * 1.5),
                                          max(4.0, subplot_h * 1.2)))
        for p in param_names:
            _plot_param_overlay(ax, rows, p, group_by_col, facet_by_cols,
                                annotate, precision, style)
        ax.legend(loc='best', fontsize=8)
    else:
        nrows = int(np.ceil(np.sqrt(n_params)))
        ncols = int(np.ceil(n_params / nrows))
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(subplot_w * ncols, subplot_h * nrows),
            squeeze=False,
        )
        for idx, p in enumerate(param_names):
            ax = axes.flat[idx]
            _plot_param_subplot(ax, rows, p, group_by_col, facet_by_cols,
                                annotate, precision, style)
        for idx in range(n_params, nrows * ncols):
            axes.flat[idx].set_visible(False)

    _apply_title(fig, rows, title, title_overflow, expr_for_auto_title, style)
    fig.tight_layout()
    return fig


def _all_param_names(rows):
    """Discover all numeric param columns across rows (skip metadata + errors)."""
    meta = {'group', 'facet', 'fit_name', 'channel',
            'chi2', 'redchi', 'fit_status', 'ndf', 'n_data'}
    seen: Dict[str, None] = {}
    for r in rows:
        for k, v in r.items():
            if k in meta or k.endswith('_err'):
                continue
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                seen.setdefault(k, None)
    return list(seen.keys())


def _plot_param_subplot(ax, rows, param, group_by_col, facet_by_cols,
                        annotate, precision, style):
    """One subplot per param: X = facet (or group), Y = param value."""
    use_facet_x = bool(facet_by_cols)
    color_by_group = use_facet_x and group_by_col is not None

    if color_by_group:
        # X = facet, color = group.
        groups = _uniq([r.get('group') for r in rows])
        cmap_colors = plt.get_cmap('tab10').colors
        for gi, gv in enumerate(groups):
            xs, ys, errs = _series_for(rows, param, x_field='facet',
                                       group_filter=gv)
            if not xs:
                continue
            ax.errorbar(range(len(xs)), ys, yerr=errs, fmt='o-',
                        color=cmap_colors[gi % len(cmap_colors)],
                        label=str(gv), markersize=4)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([str(v) for v in xs], rotation=30, ha='right',
                               fontsize=8)
            if _should_annotate(annotate, len(xs)):
                for i, (y, e) in enumerate(zip(ys, errs)):
                    ax.annotate(_fmt_num(y, precision), (i, y),
                                fontsize=_style_get(style, 'summary_fit.figure.annotate_fontsize', 8),
                                xytext=_style_get(style, 'summary_fit.figure.annotate_offset', (5, 5)),
                                textcoords='offset points')
        ax.legend(fontsize=7, loc='best')
    elif use_facet_x:
        xs, ys, errs = _series_for(rows, param, x_field='facet')
        if xs:
            ax.errorbar(range(len(xs)), ys, yerr=errs, fmt='o-',
                        markersize=4)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([str(v) for v in xs], rotation=30, ha='right',
                               fontsize=8)
            if _should_annotate(annotate, len(xs)):
                for i, y in enumerate(ys):
                    ax.annotate(_fmt_num(y, precision), (i, y), fontsize=8,
                                xytext=(5, 5), textcoords='offset points')
    else:
        # group only (no facet): X = group.
        xs, ys, errs = _series_for(rows, param, x_field='group')
        if xs:
            ax.errorbar(range(len(xs)), ys, yerr=errs, fmt='o-',
                        markersize=4)
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels([str(v) for v in xs], rotation=30, ha='right',
                               fontsize=8)
            if _should_annotate(annotate, len(xs)):
                for i, y in enumerate(ys):
                    ax.annotate(_fmt_num(y, precision), (i, y), fontsize=8,
                                xytext=(5, 5), textcoords='offset points')

    ax.set_title(param, fontsize=10)
    ax.set_ylabel(param, fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_param_overlay(ax, rows, param, group_by_col, facet_by_cols,
                        annotate, precision, style):
    """Overlay mode: all params share one axes (single y-scale)."""
    use_facet_x = bool(facet_by_cols)
    xs, ys, errs = _series_for(rows, param,
                               x_field='facet' if use_facet_x else 'group')
    if xs:
        ax.errorbar(range(len(xs)), ys, yerr=errs, fmt='o-',
                    label=param, markersize=4)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(v) for v in xs], rotation=30, ha='right',
                           fontsize=8)


def _series_for(rows, param, *, x_field, group_filter=None):
    """Extract (x, y, yerr) lists from rows for a given param, filtered by group."""
    xs, ys, errs = [], [], []
    seen_x: Dict[Any, None] = {}
    for r in rows:
        if group_filter is not None and r.get('group') != group_filter:
            continue
        x = r.get(x_field)
        y = r.get(param)
        if y is None:
            continue
        if x in seen_x:
            continue   # one point per x per group; first wins
        seen_x[x] = None
        xs.append(x)
        ys.append(float(y))
        e = r.get(f"{param}_err")
        errs.append(float(e) if isinstance(e, (int, float, np.number)) else 0.0)
    return xs, ys, errs


def _uniq(seq):
    """Order-preserving unique."""
    out, seen = [], set()
    for v in seq:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _should_annotate(annotate, n_points):
    """Annotate-or-not decision per §3.6."""
    if annotate == 'always':
        return True
    if not annotate:
        return False
    return n_points <= 15


# ============================================================================
# §3.9 — Auto-title with auto-fit
# ============================================================================

def _apply_title(fig, rows, title, title_overflow, expr_for_auto_title, style):
    """Apply a (possibly auto-generated, possibly auto-shrunk) figure title."""
    if title is None:
        return
    text = (_auto_title(rows, expr_for_auto_title)
            if title == 'auto' else title)
    if not text:
        return

    font_max = _style_get(style, 'summary_fit.title.font_size_max', 12)
    font_min = _style_get(style, 'summary_fit.title.font_size_min', 8)

    suptitle = fig.suptitle(text, fontsize=font_max)
    # Step-shrink loop. matplotlib measures via renderer; we approximate by
    # char count vs figure width (good enough for QA-grade auto-fit; precise
    # measurement requires a draw cycle).
    fig_width_in = fig.get_size_inches()[0]
    target_in = fig_width_in * 0.95
    for size in range(font_max, font_min - 1, -1):
        approx_width = len(text) * size * 0.012  # 1pt ≈ 0.012 in for narrow font
        if approx_width <= target_in:
            suptitle.set_fontsize(size)
            return
    # Hit font_min, still too wide → apply overflow strategy.
    if title_overflow == 'truncate':
        cap = max(8, int(target_in / (font_min * 0.012)))
        suptitle.set_text(text[:cap - 1] + '…')
        suptitle.set_fontsize(font_min)
    elif title_overflow == 'wrap':
        # Insert a newline near the middle on a space.
        mid = len(text) // 2
        for off in range(0, mid):
            for i in (mid - off, mid + off):
                if 0 <= i < len(text) and text[i] == ' ':
                    suptitle.set_text(text[:i] + '\n' + text[i + 1:])
                    suptitle.set_fontsize(font_min)
                    return
        suptitle.set_fontsize(font_min)
    else:  # 'shrink' default — keep at font_min
        suptitle.set_fontsize(font_min)


def _auto_title(rows, expr):
    """Build the auto-title string per §3.9."""
    if not rows:
        return ''
    fit_names = _uniq([r.get('fit_name') for r in rows if r.get('fit_name')])
    if len(fit_names) == 1:
        fit_part = fit_names[0]
    elif fit_names:
        fit_part = ','.join(fit_names)
    else:
        fit_part = '?'
    if expr:
        return f"fit params: {fit_part} — {expr}"
    return f"fit params: {fit_part}"


# ============================================================================
# Style helper
# ============================================================================

def _style_get(style, key, default):
    """Read ``style[key]`` if a style-like dict/object is given; else default."""
    if style is None:
        return default
    # Try mapping access first.
    try:
        if key in style:
            v = style[key]
            return default if v is None else v
    except TypeError:
        pass
    # Try get() method.
    try:
        v = style.get(key, default)
        return default if v is None else v
    except (AttributeError, TypeError):
        return default
