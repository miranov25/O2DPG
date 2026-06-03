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
    # Phase 13.50.DF step 7a [BREACH] — precision int REMOVED; replaced by
    # value_format / error_format strings (architect v2.5 §3.3 + §9: clean
    # removal, no alias). Defaults mirror the style.py step-2 keys.
    value_format: str = '.2g',
    error_format: str = '.1g',
    columns: Optional[List[str]] = None,
    data_format: str = 'dict',
    title: Union[str, None] = 'auto',
    title_overflow: str = 'shrink',
    style: Optional[Any] = None,
    expr_for_auto_title: Optional[str] = None,
    # Phase 13.50.DF step 7b R1 — orientation axis for Phase 13.43 'figure'
    # placement renderer (was an undisclosed partial in step 6: orientation
    # was honored only by the slot renderer). 'row' default preserves layout.
    orientation: str = 'row',
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
    value_format : str, default '.2g'
        Python format-spec string for numeric values in table cells and
        figure annotations (Phase 13.50.DF step 7a). Mirrors the
        style.py 'fit.value_format' key.
    error_format : str, default '.1g'
        Python format-spec string for numeric ERRORS in table cells.
        Default '.1g' matches the physics convention (errors at 1 sig
        fig, values matched to error's decimal place).
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
            rows, columns=columns,
            value_format=value_format, error_format=error_format,
            title=title, title_overflow=title_overflow,
            expr_for_auto_title=expr_for_auto_title, style=style,
            # Phase 13.50.DF step 7b R1 — propagate orientation kwarg to the
            # Phase 13.43 'figure' placement renderer (was an undisclosed
            # partial in step 6: orientation was honored only by the slot
            # renderer, not by the standalone-Figure path).
            orientation=orientation,
        )

    if 'figure' in kinds:
        params_fig = _render_params_figure(
            rows, params=params, mode=mode, annotate=annotate,
            value_format=value_format,
            group_by_col=group_by_col,
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
    'kind', 'params', 'mode', 'annotate', 'columns',
    'data_format', 'title', 'title_overflow',
    # Phase 13.50.DF step 5 — placement axis (where the summary_fit content
    # is rendered: separate figure, or inside the main fig as a SubFigure /
    # GridSpec pad slot). Default 'figure' preserves Phase 13.43 behavior.
    'placement',
    # Phase 13.50.DF step 6 — orientation axis (table layout direction).
    # Post step 7b rename: 'row' (default) = one row per fit, columns =
    # id keys + params; 'column' transposes (one row per id key + param,
    # columns = fits). Honored by BOTH the slot renderer
    # (_render_table_in_axes — placements 'pad' / 'subfigure') AND the
    # Phase 13.43 'figure' placement renderer (_render_table_figure, per
    # step 7b R1: v2.5 §3.4(b) put no placement restriction on orientation).
    'orientation',
    # Phase 13.50.DF step 7c spec-conformance — placement sub-keys per
    # v2.5 §3.5 (missing in step 5 ship). 'pad_location' selects which
    # edge of the main fig the GridSpec pad attaches to ('bottom'|'right'
    # |'top'|'left'); 'pad_size' is the fraction of the figure dim the
    # pad consumes (uses GridSpec width_ratios/height_ratios when the
    # fraction doesn't match equal-cell allocation per v2.4 P3-NEW-1);
    # 'inset_bbox' is the (x, y, w, h) axes-coords bbox for the per-panel
    # inset_axes used by placement='subfigure' (consumed in step 7d).
    'pad_location',
    'pad_size',
    'inset_bbox',
    # Phase 13.50.DF step 7a [BREACH] — precision int field REMOVED;
    # replaced by separate value_format / error_format string keys to mirror
    # the fit.value_format / fit.error_format style keys introduced in
    # step 2. Per v2.5 §3.3 + §9 architect direction: clean removal, no
    # deprecation alias. value_format defaults to '.2g' (2 sig figs for
    # values); error_format defaults to '.1g' (1 sig fig for errors, the
    # physics convention).
    'value_format',
    'error_format',
}
_ALLOWED_MODES = {'subplots', 'overlay'}
_ALLOWED_DATA_FORMATS = {'dict', 'pandas'}
# Phase 13.50.DF step 5 — accepted placement values.
# 'figure'    → render to a separate matplotlib Figure (Phase 13.43 default).
# 'subfigure' → render inside the main fig in a reserved SubFigure region;
#               requires GridSpec pre-planning at the dispatcher entry.
# 'pad'       → render inside the main fig in a single reserved axes (extra
#               GridSpec row, height_ratio < 1.0); same pre-planning constraint.
_ALLOWED_PLACEMENTS = {'figure', 'subfigure', 'pad'}
# Phase 13.50.DF step 6 — accepted orientation values.
# 'row'    → one row per fit, columns = identifying keys + params (default;
#            preserves the table layout users have seen since Phase 13.43;
#            v2.5 §3.4(b) canonical naming, aligns with pandas .melt(),
#            seaborn 'orient', and matplotlib table mental model)
# 'column' → transposed: one row per identifying key + param, columns = fits
#            (useful when there are many params and few fits — column
#            orientation avoids the wide-table scroll problem in that regime)
# Phase 13.50.DF step 7b spec-conformance: renamed from {'horizontal',
# 'vertical'} to match v2.5 §3.4(b) exact naming. The shipped step-6 names
# were equivalent in behavior but deviated from spec; corrected here.
_ALLOWED_ORIENTATIONS = {'row', 'column'}
# Phase 13.50.DF step 7c spec-conformance — placement sub-key value sets.
# 'pad_location' picks the edge the pad attaches to. The four cardinal
# edges are the v2.5 §3.5 sub-key values; the dispatcher in drawer.py
# translates these to GridSpec (extra row vs extra column, slice side).
_ALLOWED_PAD_LOCATIONS = {'bottom', 'right', 'top', 'left'}
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
         'value_format': '.2g', 'error_format': '.1g', 'columns': ...,
         'data_format': ..., 'title': ..., 'title_overflow': ...,
         'placement': ..., 'orientation': ...}

    Raises ValueError with §3.8 error grammar on unknown values.
    """
    canonical: Dict[str, Any] = {
        'kinds': None,
        'params': None,
        'mode': 'subplots',
        'annotate': False,
        # Phase 13.50.DF step 7a [BREACH] — precision int REMOVED; replaced
        # by separate value_format / error_format strings (architect direction
        # v2.5 §3.3 + §9: clean removal, no deprecation alias). Defaults
        # match the style.py step-2 keys: '.2g' values (2 sig figs),
        # '.1g' errors (1 sig fig, physics convention).
        'value_format': '.2g',
        'error_format': '.1g',
        'columns': None,
        'data_format': None,
        'title': 'auto',
        'title_overflow': 'shrink',
        # Phase 13.50.DF step 5 — placement axis: where the summary_fit content
        # is rendered. Default 'figure' = separate matplotlib Figure (Phase
        # 13.43 behavior preserved exactly). 'subfigure'/'pad' require the
        # caller to have pre-planned a GridSpec slot before plt.subplots()
        # (immutable post-creation per v2.3 panel P1-A).
        'placement': 'figure',
        # Phase 13.50.DF step 6 — orientation axis: table layout direction.
        # 'row' (default) preserves Phase 13.43 table shape (rows = fits,
        # columns = id keys + params). 'column' transposes — rows become
        # param names, columns become fit instances. Honored by BOTH the
        # in-slot renderer (placement='pad'/'subfigure') AND the Phase 13.43
        # 'figure' placement renderer (step 7b R1 spec-conformance fix:
        # v2.5 §3.4(b) put no placement restriction on orientation; the
        # original step-6 limitation was an undisclosed partial impl).
        'orientation': 'row',
        # Phase 13.50.DF step 7c spec-conformance — placement sub-keys per
        # v2.5 §3.5. Defaults preserve the step-5 baseline (pad at bottom,
        # ~25% of fig height, no per-panel inset bbox until placement=
        # 'subfigure' is asked for). Consumed in drawer.py's
        # _compute_pad_allocation (pad_location, pad_size) and in the
        # 'subfigure' slot renderer (inset_bbox, step 7d).
        'pad_location': 'bottom',
        'pad_size': 0.25,
        'inset_bbox': (0.55, 0.02, 0.43, 0.30),
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
                "Fix: summary_fit={'kind': 'table', 'value_format': '.2g'}."
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
        # Phase 13.50.DF step 7a [BREACH] — precision int validation REMOVED;
        # replaced by value_format / error_format string validation. The
        # validation tries the format on a sample float to catch malformed
        # spec strings at normalization time, not at render time.
        if 'value_format' in value:
            vf = value['value_format']
            if not isinstance(vf, str):
                raise ValueError(
                    f"summary_fit 'value_format' must be a format-spec str "
                    f"(got {type(vf).__name__}). "
                    f"Fix: 'value_format': '.2g'."
                )
            try:
                format(1.23, vf)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"summary_fit 'value_format' {vf!r} is not a valid "
                    f"format-spec ({exc}). "
                    f"Fix: 'value_format': '.2g' (or '.3f', '.4e', etc.)."
                ) from None
            canonical['value_format'] = vf
        if 'error_format' in value:
            ef = value['error_format']
            if not isinstance(ef, str):
                raise ValueError(
                    f"summary_fit 'error_format' must be a format-spec str "
                    f"(got {type(ef).__name__}). "
                    f"Fix: 'error_format': '.1g'."
                )
            try:
                format(1.23, ef)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"summary_fit 'error_format' {ef!r} is not a valid "
                    f"format-spec ({exc}). "
                    f"Fix: 'error_format': '.1g'."
                ) from None
            canonical['error_format'] = ef
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
        # Phase 13.50.DF step 5 — placement axis.
        if 'placement' in value:
            p = value['placement']
            if p not in _ALLOWED_PLACEMENTS:
                raise ValueError(
                    f"summary_fit placement must be one of "
                    f"{sorted(_ALLOWED_PLACEMENTS)!r} "
                    f"(got {p!r}). "
                    f"Fix: 'placement': 'figure' (default), 'subfigure', or 'pad'."
                )
            canonical['placement'] = p
        # Phase 13.50.DF step 6 — orientation axis.
        if 'orientation' in value:
            o = value['orientation']
            if o not in _ALLOWED_ORIENTATIONS:
                raise ValueError(
                    f"summary_fit orientation must be one of "
                    f"{sorted(_ALLOWED_ORIENTATIONS)!r} "
                    f"(got {o!r}). "
                    f"Fix: 'orientation': 'row' (default) or 'column'."
                )
            canonical['orientation'] = o
        # Phase 13.50.DF step 7c spec-conformance — placement sub-keys.
        # Validated even when placement is the default 'figure' (per
        # v2.5 §3.5 they're inputs to the dispatcher's pad allocator, not
        # implicit to a particular placement). Out-of-range values raise
        # at normalization time rather than mid-render.
        if 'pad_location' in value:
            pl = value['pad_location']
            if pl not in _ALLOWED_PAD_LOCATIONS:
                raise ValueError(
                    f"summary_fit pad_location must be one of "
                    f"{sorted(_ALLOWED_PAD_LOCATIONS)!r} "
                    f"(got {pl!r}). "
                    f"Fix: 'pad_location': 'bottom' (default), 'right', "
                    f"'top', or 'left'."
                )
            canonical['pad_location'] = pl
        if 'pad_size' in value:
            ps = value['pad_size']
            if not isinstance(ps, (int, float)):
                raise ValueError(
                    f"summary_fit pad_size must be a number in (0, 1) "
                    f"(got {type(ps).__name__}). "
                    f"Fix: 'pad_size': 0.25."
                )
            ps_f = float(ps)
            if not (0.0 < ps_f < 1.0):
                raise ValueError(
                    f"summary_fit pad_size must be in (0, 1) "
                    f"(got {ps!r}). "
                    f"Fix: 'pad_size': 0.25."
                )
            canonical['pad_size'] = ps_f
        if 'inset_bbox' in value:
            ib = value['inset_bbox']
            if (not isinstance(ib, (tuple, list))
                    or len(ib) != 4
                    or not all(isinstance(v, (int, float)) for v in ib)):
                raise ValueError(
                    f"summary_fit inset_bbox must be a 4-tuple "
                    f"(x, y, w, h) of numbers in [0, 1] (got {ib!r}). "
                    f"Fix: 'inset_bbox': (0.55, 0.02, 0.43, 0.30)."
                )
            x, y, w, h = (float(v) for v in ib)
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
                    and 0.0 < w <= 1.0 and 0.0 < h <= 1.0
                    and x + w <= 1.0 and y + h <= 1.0):
                raise ValueError(
                    f"summary_fit inset_bbox out of axes-coords range "
                    f"(got {ib!r}); each value in [0, 1] and x+w, y+h "
                    f"≤ 1.0. "
                    f"Fix: 'inset_bbox': (0.55, 0.02, 0.43, 0.30)."
                )
            canonical['inset_bbox'] = (x, y, w, h)
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
    rows, *, columns, value_format, error_format, title, title_overflow,
    expr_for_auto_title, style, orientation='row',
) -> plt.Figure:
    """Render a pure-table figure: each row = one fit; cells = formatted values.

    Phase 13.50.DF step 7a [BREACH]: precision int parameter removed.
    Replaced by value_format / error_format format-spec strings (e.g., '.2g',
    '.1g'). Cells with a paired error (col + col+'_err') render as
    "value ± err" using the two formats; cells without an error use
    value_format only.

    Phase 13.50.DF step 7b R1 spec-conformance: ``orientation`` parameter
    added (was previously ignored on the 'figure' placement path; v2.5
    §3.4(b) put no placement restriction on orientation). Default 'row'
    preserves Phase 13.43 layout. 'column' transposes — original column
    headers become the first column (acting as row labels); each original
    data row becomes a column of the transposed table.
    """
    columns_to_show = columns or _default_columns(rows)

    # Figure size scales with row/col count. Compute pre-transpose first
    # so 'row' (default) keeps the existing dimensions exactly.
    figsize_per_row = _style_get(style, 'summary_fit.table.figsize_per_row', 0.35)
    figsize_per_col = _style_get(style, 'summary_fit.table.figsize_per_col', 1.20)

    # Format cells in row-orientation first; transpose later if requested.
    cell_text: List[List[str]] = []
    for r in rows:
        row_cells = []
        for col in columns_to_show:
            row_cells.append(_format_cell(r, col, value_format, error_format))
        cell_text.append(row_cells)

    col_labels = list(columns_to_show)

    # Phase 13.50.DF step 7b R1 — orientation transpose for Phase 13.43 path.
    # Mirrors the slot renderer's transpose logic so 'figure' and 'pad'
    # placements with orientation='column' produce equivalent table content
    # (the F17 cross-variant equivalence invariant from v2.5 §3.8).
    if orientation == 'column':
        transposed: List[List[str]] = []
        for col_idx, header in enumerate(col_labels):
            new_row = [str(header)]
            for orig_row in cell_text:
                new_row.append(orig_row[col_idx] if col_idx < len(orig_row) else "")
            transposed.append(new_row)
        col_labels = [""] + [f"fit_{i}" for i in range(len(cell_text))]
        cell_text = transposed

    # Re-compute figsize using post-transpose dimensions so the figure
    # actually fits the visible table.
    fig_w = max(6.0, len(col_labels) * figsize_per_col)
    fig_h = max(2.0, 0.7 + len(cell_text) * figsize_per_row + 0.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
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


def _format_cell(row, col, value_format, error_format):
    """Format one cell — paired value/err combined when both present.

    Phase 13.50.DF step 7a [BREACH]: signature changed from
    (row, col, precision) to (row, col, value_format, error_format).
    """
    val = row.get(col)
    if val is None:
        return ''
    err_key = f"{col}_err"
    err = row.get(err_key) if not col.endswith('_err') else None
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        if err is not None and isinstance(err, (float, int, np.number)):
            return f"{_fmt_num(val, value_format)} ± {_fmt_num(err, error_format)}"
        return _fmt_num(val, value_format)
    return str(val)


def _fmt_num(x, fmt):
    """Format a numeric value using a Python format-spec string.

    Phase 13.50.DF step 7a [BREACH]: signature changed from (x, precision: int)
    to (x, fmt: str). Caller supplies format-spec like '.2g', '.3f', '.4e'.
    Non-finite values fall back to str(x) (unchanged behavior).
    """
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return str(x)
    return f"{x:{fmt}}"


# ============================================================================
# Params figure renderer
# ============================================================================

def _render_params_figure(
    rows, *, params, mode, annotate, value_format,
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
                                annotate, value_format, style)
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
                                annotate, value_format, style)
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
                        annotate, value_format, style):
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
                    ax.annotate(_fmt_num(y, value_format), (i, y),
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
                    ax.annotate(_fmt_num(y, value_format), (i, y), fontsize=8,
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
                    ax.annotate(_fmt_num(y, value_format), (i, y), fontsize=8,
                                xytext=(5, 5), textcoords='offset points')

    ax.set_title(param, fontsize=10)
    ax.set_ylabel(param, fontsize=9)
    ax.grid(True, alpha=0.3)


def _plot_param_overlay(ax, rows, param, group_by_col, facet_by_cols,
                        annotate, value_format, style):
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


# ============================================================================
# Phase 13.50.DF step 5 — Slot-based renderer for placement='pad'/'subfigure'
# ============================================================================
# When the dispatcher pre-plans a GridSpec slot (via fig._dfdraw_summary_fit_slot),
# the attach path skips the new-Figure rendering of render_summary_fit() and
# routes through render_summary_fit_into_slot() below. The implementation is
# intentionally lightweight: it produces a single axes (for 'pad') or a single
# SubFigure containing one axes (for 'subfigure'), and renders the FIT TABLE
# into it. The params-trend FIGURE kind, when requested under 'pad'/'subfigure',
# is rendered in the SAME slot beneath/beside the table — single axes per slot.
# Full multi-axes layouts inside the subfigure are deferred to Phase 13.50 FIX1.

def render_summary_fit_into_slot(slot_spec, fig, stats_fit, spec, *,
                                 group_by_col=None, facet_by_cols=None,
                                 expr_for_auto_title=None, style=None):
    """Phase 13.50.DF step 5 — render summary_fit content into a pre-reserved
    GridSpec slot instead of creating a new Figure.

    Parameters
    ----------
    slot_spec : tuple (SubplotSpec, placement_mode_str)
        The slot reserved by ``_dispatch_faceted_render`` and stashed on
        ``fig._dfdraw_summary_fit_slot``. The placement_mode_str is one of
        {'pad', 'subfigure'} and selects the slot fill strategy.
    fig : matplotlib.figure.Figure
        The main figure with the reserved slot.
    stats_fit : any
        Phase 13.42 stats['fit'] structure (Shape 1/2/3 per
        ``_flatten_to_rows``).
    spec : dict
        Canonical spec from ``_normalize_summary_fit_spec``.
    group_by_col, facet_by_cols : optional
        Passed through for table row keying (same semantics as in
        ``render_summary_fit``).
    expr_for_auto_title : optional
        Original expression for auto-title rendering.
    style : optional
        ``_ModuleStyleProxy`` instance (or any get/__contains__/__getitem__
        protocol).

    Returns
    -------
    dict
        ``{'table': ax, 'placement': 'pad' | 'subfigure'}`` when the slot
        was populated; ``{}`` if no rows produced. ``ax`` is the axes
        actually used to host the table (either a direct ``fig.add_subplot``
        on the slot for 'pad', or a single subplot inside the SubFigure
        for 'subfigure'). Tests use this for placement_topology inspection.
    """
    if slot_spec is None:
        return {}
    subplot_spec, placement_mode = slot_spec

    rows = _flatten_to_rows(stats_fit, group_by_col, facet_by_cols)
    if not rows:
        return {}

    # Materialize the slot per placement_mode.
    if placement_mode == 'pad':
        host_axes = fig.add_subplot(subplot_spec)
    elif placement_mode == 'subfigure':
        # fig.add_subfigure(subplotspec) reserves a SubFigure region; one
        # subplot inside it hosts the table. Multi-axes sub-layouts (table
        # + trend in separate axes within the same SubFigure) are FIX1 work.
        subfig = fig.add_subfigure(subplot_spec)
        host_axes = subfig.subplots(1, 1)
        # FIX1 of step 5: stash the SubFigure on fig so placement_topology
        # can detect "host_axes lives inside a SubFigure" without relying on
        # fig.subfigures (which is a CREATION METHOD on matplotlib Figure,
        # NOT an iterable property — iterating it raises TypeError).
        fig._dfdraw_summary_fit_subfigure = subfig
    else:
        raise ValueError(
            f"render_summary_fit_into_slot: unknown placement_mode "
            f"{placement_mode!r}; expected 'pad' or 'subfigure'."
        )

    # Render table into host_axes.
    # Table cells: one row per stats row; columns = group + facet keys + params.
    kinds = spec.get('kinds') or []
    if 'table' in kinds or 'both' in kinds:
        _render_table_in_axes(host_axes, rows, spec)
    else:
        # 'figure'-only kind requested in 'pad'/'subfigure' slot: fall back to
        # a single text annotation since the trend-figure layout doesn't fit
        # in a single axes well. FIX1 will broaden this.
        host_axes.axis('off')
        host_axes.text(0.5, 0.5,
                       "summary_fit 'figure' kind in placement='pad'/'subfigure': "
                       "rendered as table fallback (FIX1 work — full multi-axes "
                       "layout in the subfigure region).",
                       ha='center', va='center',
                       transform=host_axes.transAxes,
                       fontsize=7, wrap=True)

    return {'table': host_axes, 'placement': placement_mode}


def _render_table_in_axes(ax, rows, spec):
    """Render the summary_fit table into a host axes using ax.table().

    Single axes inside a GridSpec slot — table fills the axes. The
    `params`, `value_format`, and `error_format` settings from spec control
    which columns appear and their formatting (Phase 13.50.DF step 7a
    [BREACH]: precision int parameter replaced by format-spec strings).

    Phase 13.50.DF step 7e FIX1 (post-step-7 F17 failure): column
    discovery + cell formatting now mirror ``_render_table_figure``
    exactly — uses ``_default_columns(rows)`` for the column list and
    ``_format_cell`` for paired ``value ± err`` cells. The pre-fix logic
    looked for a nested ``row['params']`` sub-dict that ``_make_row``
    never produces (params are flattened to top-level row keys), so only
    id_keys (group, facet, fit_name) survived to the rendered table. This
    made the 'pad'-placement table look like a stub vs the 'figure'-
    placement table's full column set, breaking the F17 cross-variant
    equivalence invariant (v2.5 §3.8). Cross-variant content equivalence
    is the v2.5 headline guarantee for the placement axis — switching
    placement must not change which cells render.
    """
    if not rows:
        ax.axis('off')
        return

    # Mirror the figure renderer's column discovery.
    columns_to_show = _default_columns(rows)

    # Apply user's params= subset filter, preserving id-keys + per-param
    # error columns (`<name>_err`) for any param the user kept.
    requested = spec.get('params')
    if requested is not None:
        id_key_set = {'group', 'facet', 'fit_name'}
        metric_keys = {'chi2', 'ndf', 'redchi', 'fit_status', 'n_data'}
        keep = set(requested)
        # Include the *_err sibling for any kept param.
        keep_err = {f"{p}_err" for p in requested}
        columns_to_show = [
            c for c in columns_to_show
            if c in id_key_set or c in metric_keys
            or c in keep or c in keep_err
        ]

    # Mirror the figure renderer's cell formatting (value ± err pairing).
    value_format = spec.get('value_format', '.2g')
    error_format = spec.get('error_format', '.1g')

    headers = list(columns_to_show)
    cell_rows = []
    for row in rows:
        row_cells = []
        for col in columns_to_show:
            row_cells.append(_format_cell(row, col, value_format, error_format))
        cell_rows.append(row_cells)

    ax.axis('off')
    if not cell_rows or not headers:
        return

    # Phase 13.50.DF step 6 — orientation axis (step 7b naming alignment).
    # 'row'    (default): one row per fit, columns = id keys + params
    #                     (cell_rows[i] = row i, headers as colLabels)
    # 'column' (step 7b rename, was 'vertical'): transpose — one row per
    #          (id key OR param), columns = fits. The original `headers`
    #          list becomes the first column of the transposed table
    #          (acting as row labels); each cell_rows[i] becomes a column.
    #          New colLabels = generic "fit_0", "fit_1", ... since the
    #          original fits don't have a single short identifier (they're
    #          characterized by the id_keys composition).
    orientation = spec.get('orientation', 'row')
    if orientation == 'column':
        # Build transposed: each ORIGINAL column becomes a row whose first
        # cell is the original header label and remaining cells are the
        # values from each original row at that column index.
        transposed_rows = []
        for col_idx, header in enumerate(headers):
            new_row = [str(header)]
            for orig_row in cell_rows:
                new_row.append(orig_row[col_idx] if col_idx < len(orig_row) else "")
            transposed_rows.append(new_row)
        # First column header is empty (row label position); remaining are
        # generic indices for the original fits.
        new_headers = [""] + [f"fit_{i}" for i in range(len(cell_rows))]
        cell_rows = transposed_rows
        headers = new_headers

    table = ax.table(cellText=cell_rows, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.0)


# ============================================================================
# Phase 13.50.DF step 7d — Per-panel inset renderer for placement='subfigure'
# ============================================================================
# Per v2.5 §3.5 + P2-NEW-3 (Claude36 ADF panel finding folded into v2.4):
# placement='subfigure' means each facet panel hosts its OWN inset_axes()
# showing ONLY that panel's fits — per-panel slices, not a redundant
# full-table copy on every panel. The v1.0 step-5 ship was wrong (single
# SubFigure with the entire table); this step ships the spec'd semantic.

def render_summary_fit_per_panel_insets(
    fig, stats_fit, spec, *,
    group_by_col=None, facet_by_cols=None, style=None,
):
    """Render per-panel summary_fit insets (placement='subfigure').

    For each visible facet axes carrying a ``_dfdraw_facet_key`` marker,
    add an ``Axes.inset_axes(inset_bbox)`` inset and render that panel's
    fit subset into the inset table. Panels with no fit rows get no inset.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Main figure with facet axes laid out by ``_dispatch_faceted_render``.
        Each candidate axes is identified via the ``_dfdraw_facet_key``
        attribute stashed during the dispatch loop (step 7d wiring).
    stats_fit : any
        Phase 13.42 stats['fit'] container. Shape 3 (faceted) is the
        normal case; shape 1/2 produce one inset on the first visible
        axes only.
    spec : dict
        Canonical spec from ``_normalize_summary_fit_spec``. Reads
        ``inset_bbox`` for the per-panel inset placement (defaulted to
        ``(0.55, 0.02, 0.43, 0.30)`` per v2.5 §3.5).
    group_by_col, facet_by_cols : optional
        Echo of the call-site faceting axis names — passed to
        ``_flatten_to_rows`` per-panel.
    style : optional
        Style proxy (unused in this renderer; the table fontsize and
        scale are fixed for in-axes insets).

    Returns
    -------
    dict
        ``{'insets': [list of inset axes],
           'per_panel_keyed': {facet_key: inset_axes, ...},
           'placement': 'subfigure'}``
        when at least one panel got an inset. Empty dict if no panels
        produced fits (caller treats this as "rows were produced=False").
    """
    inset_bbox = spec.get('inset_bbox') or (0.55, 0.02, 0.43, 0.30)

    # Iterate over candidate axes. Visible-axes filter keeps us off the
    # hidden-spare cells; the facet-key marker keeps us off non-dispatch
    # axes (legends, colorbars, etc.). Shape-3 stats_fit is the normal
    # case here; shapes 1/2 are handled by the single-axes fallback below.
    candidates = [ax for ax in fig.axes
                  if ax.get_visible()
                  and hasattr(ax, '_dfdraw_facet_key')]
    if not candidates:
        return {}

    insets: list = []
    per_panel_keyed: Dict[Any, plt.Axes] = {}

    # If stats_fit is a dict, treat each top-level key as a panel.
    # Otherwise (list-of-lists), fall back to attaching one inset to the
    # first visible axes only (degenerate "1 fit, 1 panel" case).
    if isinstance(stats_fit, dict):
        for ax in candidates:
            panel_key = ax._dfdraw_facet_key
            panel_stats = _select_panel_stats(stats_fit, panel_key)
            if panel_stats is None:
                continue
            panel_rows = _flatten_to_rows(
                panel_stats, group_by_col, facet_by_cols,
            )
            if not panel_rows:
                continue
            inset = ax.inset_axes(list(inset_bbox))
            _render_table_in_axes(inset, panel_rows, spec)
            insets.append(inset)
            per_panel_keyed[panel_key] = inset
    else:
        # Non-dict stats_fit: attach to first visible facet axes only.
        rows = _flatten_to_rows(stats_fit, group_by_col, facet_by_cols)
        if rows:
            ax = candidates[0]
            inset = ax.inset_axes(list(inset_bbox))
            _render_table_in_axes(inset, rows, spec)
            insets.append(inset)
            per_panel_keyed[ax._dfdraw_facet_key] = inset

    if not insets:
        return {}
    return {
        'insets': insets,
        'per_panel_keyed': per_panel_keyed,
        'placement': 'subfigure',
    }


def _select_panel_stats(stats_fit_dict, panel_key):
    """Return the subset of stats_fit_dict matching panel_key, or None.

    Phase 13.50.DF step 7d FIX2 (post-step-7 F12 failure): the dispatcher
    keys per-facet stats with ``str(group_value)`` at drawer.py:3784, and
    aggregates faceted fits as ``{(str(_gval),): _cell_fit}`` at drawer.py:
    3863. The dispatch loop, however, stashes the RAW ``group_value`` on
    ``ax._dfdraw_facet_key``. So a panel_key like int ``3`` will not match
    a stats_fit key like ``('3',)`` via the raw membership check. This
    function now tries both raw and stringified forms:

      1. ``panel_key in stats_fit_dict``         (raw scalar key)
      2. ``str(panel_key) in stats_fit_dict``    (stringified scalar key)
      3. tuple-key containing raw panel_key      (faceted_by='quantiles' /
         vector y modes — raw types preserved)
      4. tuple-key containing stringified key    (faceted_by='column' /
         'group_by' modes — dispatcher stringifies before keying)

    Wraps the matched entry in a single-key dict so ``_flatten_to_rows``
    sees the canonical Shape-2-like container.
    """
    panel_key_str = str(panel_key)
    if panel_key in stats_fit_dict:
        return {panel_key: stats_fit_dict[panel_key]}
    if panel_key_str in stats_fit_dict:
        return {panel_key_str: stats_fit_dict[panel_key_str]}
    for k in stats_fit_dict:
        if not isinstance(k, tuple):
            continue
        if panel_key in k:
            return {k: stats_fit_dict[k]}
        if any(str(x) == panel_key_str for x in k):
            return {k: stats_fit_dict[k]}
    return None
