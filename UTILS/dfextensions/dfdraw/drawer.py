"""
DFDraw - Main drawing class with TTree::Draw-like interface.

Phase 13.1.DF: Added PyArrow Table input support.
Phase 13.16.DF FIX1 (2026-04-14): Vector path kwarg propagation fix.
"""

import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .style import get_style, get_style_value
# Phase 13.32.DF Sub-fix 1+3: dispatch-level binning needs the same interval-label
# formatter that plots.profile uses, to keep subplot-key strings identical to the
# format the per-subplot recursion would have produced. Safe top-level import:
# plots/profile.py has no top-level dependency on drawer.py (it imports DFDraw
# inside draw_profile() only).
from .plots.profile import _format_interval_label, _interval_sort_key

# =============================================================================
# Phase 13.16.DF FIX1: Sentinel for "parameter was not passed by caller".
# Used by vector dispatch tuple-driven forwarding so that we can distinguish
# "caller passed value=None explicitly" from "caller didn't pass it at all".
# =============================================================================

_MISSING = object()

# =============================================================================
# Phase 13.1.DF: PyArrow Detection
# =============================================================================

try:
    import pyarrow as pa
    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False
    pa = None


def _is_pyarrow_table(obj) -> bool:
    """Check if object is a PyArrow Table (safe when PyArrow not installed)."""
    return _PYARROW_AVAILABLE and isinstance(obj, pa.Table)


# Type alias for return value
DrawResult = Tuple[Any, Any, Dict[str, Any]]  # (fig, ax, stats)


# =============================================================================
# Phase 13.41.DF v1.6 — Multi-dimensional faceting helpers
# =============================================================================
# Convention LOCKED (matches numpy/pandas (n_rows, n_cols, ...) shape):
#   facet_by[0] = ROW dimension     (vertical within each figure)
#   facet_by[1] = COLUMN dimension  (horizontal within each figure)
#   facet_by[2] = FIGID dimension   (separate figures, one per value)
#   facet_by[3+] → NotImplementedError
#
# Locked by §9.FBY.6 (2D row/col convention) + §9.FBY.11 (3D figID).
# =============================================================================

def _validate_share_axis_value(share, name):
    """Validate share_x / share_y values per Phase 13.41 v1.2 CP0-2.
    
    Valid values: 'all', 'row', 'col', 'none'.
    """
    if share not in ('all', 'row', 'col', 'none'):
        raise ValueError(
            f"{name} must be one of 'all'/'row'/'col'/'none', got {share!r}"
        )


def _to_mpl_share(share):
    """Map dfdraw share_x/share_y value → matplotlib plt.subplots(sharex/sharey).
    
    Phase 13.41.DF v1.2 CP0-1 — symmetric for both axes (was BUGGY in v1.1):
      'all'  → True   (all cells share)
      'row'  → 'row'  (cells in same row share)
      'col'  → 'col'  (cells in same column share)
      'none' → False  (no sharing)
    
    Verified by execution: plt.subplots(sharex='row') correctly links cells
    in the same row; sharex=False links nothing. The v1.1 bug ({'row': False})
    caused share_x='row' to silently disable sharing — Hard Constraint #3.
    Locked by §9.FBY.12 (share_x='row') + §9.FBY.19 (share_x='col' symmetry).
    """
    return {'all': True, 'row': 'row', 'col': 'col', 'none': False}[share]


def _normalize_facet_args(facet_by, facet_by_bins, facet_by_quantiles):
    """Convert all facet_by forms to canonical list-of-N representation.
    
    Phase 13.41.DF v1.6 (matches §4 spec).
    
    Returns: (facet_list, bins_list, quantiles_list) — all lists of length N.
    
    Raises:
      NotImplementedError for N > 3 (visualization deferred)
      ValueError on length mismatch between facet_by and bins/quantiles lists
    """
    # Coerce facet_by to list
    if isinstance(facet_by, str):
        facet_list = [facet_by]
    elif isinstance(facet_by, list):
        facet_list = list(facet_by)
    else:
        raise ValueError(
            f"facet_by must be str or list of str, got {type(facet_by).__name__}: "
            f"{facet_by!r}"
        )
    n = len(facet_list)
    
    if n == 0:
        raise ValueError("facet_by list cannot be empty")
    if n > 3:
        raise NotImplementedError(
            f"facet_by length {n} > 3. Up to 3 dimensions supported: "
            f"[row, col, figID]. For more dimensions, use group_by overlay "
            f"inside cells. Got: facet_by={facet_by!r}"
        )
    
    # Coerce bins (CP2-A v1.4 note: int broadcasts to first dim only)
    if facet_by_bins is None:
        bins_list = [None] * n
    elif isinstance(facet_by_bins, (int, np.integer)) and not isinstance(facet_by_bins, bool):
        # int form: apply to first dim, others None
        bins_list = [int(facet_by_bins)] + [None] * (n - 1)
    elif isinstance(facet_by_bins, list):
        bins_list = list(facet_by_bins)
        if len(bins_list) != n:
            raise ValueError(
                f"facet_by has {n} dimensions; facet_by_bins must have {n} "
                f"elements (got {len(bins_list)}). "
                f"Got: facet_by={facet_by!r}, facet_by_bins={facet_by_bins!r}"
            )
    else:
        raise ValueError(
            f"facet_by_bins must be int, list, or None; got "
            f"{type(facet_by_bins).__name__}: {facet_by_bins!r}"
        )
    
    # Coerce quantiles
    if facet_by_quantiles is None:
        quantiles_list = [None] * n
    elif isinstance(facet_by_quantiles, list):
        # Could be List[float] (1D form) or List[List[float]] (per-dim form)
        if n == 1:
            # 1D — accept either form for backward compat
            if all(isinstance(q, (int, float, np.number)) for q in facet_by_quantiles):
                quantiles_list = [facet_by_quantiles]
            else:
                quantiles_list = list(facet_by_quantiles)
        else:
            # N-D — must be List[List[float]] or List[None]
            quantiles_list = list(facet_by_quantiles)
            if len(quantiles_list) != n:
                raise ValueError(
                    f"facet_by has {n} dimensions; facet_by_quantiles must "
                    f"have {n} elements (got {len(quantiles_list)}). "
                    f"Got: facet_by={facet_by!r}, "
                    f"facet_by_quantiles={facet_by_quantiles!r}"
                )
    else:
        raise ValueError(
            f"facet_by_quantiles must be list or None; got "
            f"{type(facet_by_quantiles).__name__}: {facet_by_quantiles!r}"
        )
    
    return facet_list, bins_list, quantiles_list


def _resolve_facet_values(df, col, bins=None, quantiles=None):
    """Resolve a facet dimension's column + binning into a list of value-groups.
    
    Phase 13.41.DF v1.6 §4 (CP2-A: NEW helper, not pre-existing).
    
    Returns: list of values (scalars for discrete; pd.Interval objects for binned).
    """
    if bins is None and quantiles is None:
        # Discrete column — sorted unique values
        return sorted(df[col].dropna().unique().tolist())
    
    if quantiles is not None:
        bin_series = pd.qcut(df[col], q=quantiles, duplicates='drop')
    else:
        bin_series = pd.cut(df[col], bins=bins)
    
    # Return the Categorical's categories (Intervals) in sorted order
    return list(bin_series.cat.categories)


def _filter_facet_value(df, col, value, bins=None, quantiles=None):
    """Filter df to rows matching a single facet value.
    
    Phase 13.41.DF v1.2 CP1-3 — discrete vs binned distinction:
    - Discrete (bins+quantiles both None): df[df[col] == value]
    - Binned (one of them set): reconstruct cut series, filter by Interval
    """
    if bins is None and quantiles is None:
        return df[df[col] == value]
    
    if quantiles is not None:
        bin_series = pd.qcut(df[col], q=quantiles, duplicates='drop')
    else:
        bin_series = pd.cut(df[col], bins=bins)
    
    mask = (bin_series == value)
    return df[mask.fillna(False) if mask.dtype == object else mask]


# Phase 13.46.DF C-2: ROOT-convention plot-type aliases. Applied before the
# type-dispatch ladder in DFDraw.draw so e.g. ROOT's "histo" maps to "hist".
# Kept as a module-level dict so it is greppable and trivially extensible.
_TYPE_ALIASES = {'histo': 'hist'}


def _get_suptitle(fig):
    """Phase 13.46.DF C-4: public-API suptitle text (matplotlib >= 3.8) with a
    private-attribute fallback for older matplotlib.

    Returns the suptitle string, or '' when the figure has no suptitle.
    Source-level counterpart to the test helper that already used the public
    API; replaces 9 inline ``fig._suptitle.get_text()`` expressions so the
    source no longer depends on the private ``_suptitle`` attribute.
    """
    try:
        t = fig.get_suptitle()          # public API, matplotlib >= 3.8
        return t if t else ''
    except AttributeError:
        st = getattr(fig, '_suptitle', None)
        return st.get_text() if st else ''


def _suptitle_top_for_title(title_text):
    """Phase 13.42.DF FIX2 (B6): compute a reasonable subplots_adjust top= value
    based on suptitle line count, so multi-line titles or dense facet grids
    don't overlap subplot titles.

    Production-gate B6 finding (Phase 13.42 close): single-line auto_title +
    long-facet-tuple subtitle frequently overlapped the row-0 subplot titles
    when facet_by was 2D. The fixed top=0.92 reserved 8% for the suptitle
    block, which is insufficient for 2-3 line titles.

    Returns top fraction in [0.84, 0.94] depending on the number of newlines
    in title_text. None → falls back to 0.92 (existing behavior).
    """
    if not title_text:
        return 0.92
    n_lines = str(title_text).count('\n') + 1
    # 1 line → 0.93; 2 lines → 0.89; 3 lines → 0.86; 4+ → 0.84
    return max(0.84, 0.93 - 0.035 * max(0, n_lines - 1))


def _compute_global_ranges(df, x_expr, y_expr, plot_kind):
    """For 3D share_across_figures=True, compute global x/y ranges.
    
    Phase 13.41.DF v1.2 CP1-2 — per-plot-kind logic:
      scatter:  lock both x AND y (raw data on both axes)
      hist:     lock x only; x-axis data is in y_expr (hist convention has
                x_expr=None and the histogrammed column in y_expr)
      profile:  lock x only; y is aggregate (auto-scales per figure)
      hist2d / profile2d: no global lock (per-cell auto-scale)
    
    Rationale: locking y for hist/profile would crush sparse figID values
    to invisibility when N-per-figID varies by ≥10× (common in ALICE
    cross-run comparisons).
    """
    def _eval_expr(expr):
        """Evaluate expression — column name or df.eval expression."""
        if expr is None:
            return None
        if isinstance(expr, str) and expr in df.columns:
            return df[expr].values
        if isinstance(expr, str):
            try:
                return df.eval(expr).values
            except Exception:
                return None
        return None
    
    if plot_kind == 'hist':
        # Hist convention: x_expr=None, y_expr=column being histogrammed.
        # The histogrammed column appears on the x-axis of the plot.
        data = _eval_expr(y_expr)
        if data is None or len(data) == 0:
            return None, None
        try:
            x_range = (float(np.nanmin(data)), float(np.nanmax(data)))
        except (ValueError, TypeError):
            x_range = None
        return x_range, None    # y is bin count → auto-scale per figure
    
    x_data = _eval_expr(x_expr)
    if x_data is None or len(x_data) == 0:
        return None, None
    
    try:
        x_range = (float(np.nanmin(x_data)), float(np.nanmax(x_data)))
    except (ValueError, TypeError):
        x_range = None
    
    if plot_kind == 'scatter':
        y_data = _eval_expr(y_expr)
        if y_data is None or len(y_data) == 0:
            return x_range, None
        try:
            y_range = (float(np.nanmin(y_data)), float(np.nanmax(y_data)))
        except (ValueError, TypeError):
            y_range = None
        return x_range, y_range
    elif plot_kind == 'profile':
        return x_range, None   # x only; y aggregate auto-scales (CP1-2 design)
    else:
        return None, None       # hist2d / profile2d / unknown


class DFDraw:
    """
    DataFrame drawing class with TTree::Draw-like interface.
    
    Parameters
    ----------
    data : DataFrame-like or PyArrow Table
        Input data. Accepts:
        - pandas.DataFrame
        - pyarrow.Table (Phase 13.1.DF - converted to pandas internally)
        - AliasDataFrame (uses .df attribute)
        - dict of arrays (converted to DataFrame)
    
    Examples
    --------
    >>> plotter = DFDraw(df)
    >>> fig, ax, stats = plotter.draw("y:x", color="category")
    >>> fig, ax, stats = plotter.hist("x", bins=100)
    
    # Phase 13.1.DF: PyArrow input
    >>> import pyarrow as pa
    >>> table = pa.Table.from_pandas(df)
    >>> plotter = DFDraw(table)
    >>> fig, ax, stats = plotter.hist("x", bins=100)
    
    Notes
    -----
    Phase 13.1.DF: PyArrow Tables are accepted for API compatibility with
    PyArrow-based pipelines (e.g., groupby-regression output), but are
    converted to pandas internally. This provides seamless integration
    but does not reduce memory usage within dfdraw itself. Memory
    optimization occurs upstream in groupby-regression (Phase 13.1.GB)
    and AliasDataFrame (Phase 13.3.ADF).
    """
    
    def __init__(self, data):
        self._data_source = data  # Keep reference for duck typing (axis titles)
        self._table = None  # Phase 13.1.DF: Store original PyArrow Table if provided
        self.df = self._normalize_data(data)
        # Phase 13.13.DF: Track last axes for same=True (AD-15)
        self._last_ax = None
        self._color_cycle_index = 1  # Start at 1: first plot uses index 0 (AD-16, A2)
        self._last_plot_expr = None  # Phase 13.13.DF fix: track first plot expression for retroactive label
    
    def _normalize_data(self, data) -> pd.DataFrame:
        """
        Convert input to pandas DataFrame.
        
        Supports duck typing:
        - PyArrow Table: convert to DataFrame (Phase 13.1.DF)
        - DataFrame: use as-is
        - Has .df attribute: extract DataFrame (AliasDataFrame)
        - dict-like: convert to DataFrame
        """
        # Phase 13.1.DF: PyArrow Table - convert to pandas immediately
        # Note: Immediate conversion for API compatibility. dfdraw requires pandas
        # for expression evaluation (df.eval), selection, and group_by operations.
        # Memory optimization happens upstream (groupby-regression, AliasDataFrame),
        # not in dfdraw which is an end-of-pipeline visualization tool.
        if _is_pyarrow_table(data):
            self._table = data
            return data.to_pandas()
        
        # Already a DataFrame
        if isinstance(data, pd.DataFrame):
            return data
        
        # AliasDataFrame or similar (has .df attribute)
        if hasattr(data, 'df') and isinstance(data.df, pd.DataFrame):
            return data.df
        
        # Dict of arrays
        if isinstance(data, dict):
            return pd.DataFrame(data)
        
        # Has __getitem__ and keys() - dict-like
        if hasattr(data, '__getitem__') and hasattr(data, 'keys'):
            return pd.DataFrame({k: data[k] for k in data.keys()})
        
        raise TypeError(
            f"Cannot create DFDraw from {type(data).__name__}. "
            "Expected DataFrame, PyArrow Table, AliasDataFrame, or dict of arrays."
        )
    
    # =========================================================================
    # Phase 13.1.DF: Backend Detection
    # =========================================================================
    
    @property
    def backend(self) -> str:
        """
        Return storage backend type.
        
        Returns 'pyarrow' if input was PyArrow Table, 'pandas' otherwise.
        Note: Data is always converted to pandas internally for processing.
        """
        return 'pyarrow' if self._table is not None else 'pandas'
    
    def memory_info(self) -> Dict[str, Any]:
        """
        Return memory usage information.
        
        Returns
        -------
        dict
            Memory statistics including backend type and byte sizes.
            - backend: 'pyarrow' or 'pandas' (original input type)
            - nbytes: Current memory usage (always pandas internally)
            - original_nbytes: Original PyArrow size (only if PyArrow input)
            - num_rows, num_columns: Shape information
        """
        pandas_bytes = int(self.df.memory_usage(deep=True).sum())
        
        info = {
            'backend': 'pyarrow' if self._table is not None else 'pandas',
            'nbytes': pandas_bytes,  # Actual memory used (always pandas internally)
            'num_rows': len(self.df),
            'num_columns': len(self.df.columns),
        }
        
        if self._table is not None:
            info['original_nbytes'] = self._table.nbytes  # Input size before conversion
        
        return info
    
    # =========================================================================
    # Phase 13.43.DF v1.2 — Summary Fit (outer-layer consume)
    # =========================================================================

    def _maybe_attach_summary_fit(self, stats, summary_fit_spec, *,
                                  group_by=None, facet_by=None,
                                  expr_for_auto_title=None,
                                  consumed_by_normalize=False):
        """Phase 13.43.DF v1.2 §4.2: attach summary_fit results to stats.

        Called AT EACH return site of DFDraw.{hist,profile,scatter,draw}
        AFTER the dispatch returns. Modifies ``stats`` in place — adds
        ``stats['summary_fit']`` (dict of Figures + 'data') OR Scenario E
        empty dict + ``stats['summary_fit_note']`` diagnostic.

        Parameters
        ----------
        stats : dict
            The stats dict the outer method is about to return.
        summary_fit_spec : str | list | dict | None
            User's summary_fit= kwarg. None → no-op (Scenario A unchanged).
        group_by, facet_by : optional
            Column name(s) the call used; passed through for axis logic
            and table column-key construction.
        expr_for_auto_title : optional
            Original expression string for §3.9 auto-title.
        consumed_by_normalize : bool
            True when normalize= was active and consumed fit/summary_fit.
            Forces Scenario E with a normalize-specific note even if a
            stale stats['fit'] would otherwise look renderable.
        """
        if summary_fit_spec is None:
            return  # Scenario A: kwarg absent, no key added.

        # ------------------------------------------------------------------
        # Vector dispatch path: _draw_vector returns stats as a list of
        # per-iteration dicts (one per channel × selection × weights iter).
        # Aggregate the per-iteration stats['fit'] entries into a synthetic
        # Shape 3 dict (keyed by iteration tuple) so _flatten_to_rows
        # produces the right per-(selection × group × cell) rows, then
        # write the rendered summary_fit back into the FIRST iter dict so
        # callers can access it via stats[0]['summary_fit'].
        # CRR §2 disclosure: vector-dispatch summary_fit lives on the first
        # iter dict, not at top level — there is no top level when stats
        # is a list. This is documented in the proposal as the natural
        # location given the per-iteration list contract.
        # ------------------------------------------------------------------
        if isinstance(stats, list):
            if not stats:
                return
            aggregated_fits: Dict[Tuple[Any, ...], Any] = {}
            for i, iter_stats in enumerate(stats):
                if not isinstance(iter_stats, dict):
                    continue
                cell_fit = iter_stats.get('fit')
                if cell_fit:
                    aggregated_fits[(i,)] = cell_fit
            # Render into a wrapper, then deposit on stats[0].
            wrapper: Dict[str, Any] = {'fit': aggregated_fits}
            self._maybe_attach_summary_fit(
                wrapper, summary_fit_spec,
                group_by=group_by, facet_by=facet_by,
                expr_for_auto_title=expr_for_auto_title,
                consumed_by_normalize=consumed_by_normalize,
            )
            target = next((d for d in stats if isinstance(d, dict)), None)
            if target is not None:
                if 'summary_fit' in wrapper:
                    target['summary_fit'] = wrapper['summary_fit']
                if 'summary_fit_note' in wrapper:
                    target['summary_fit_note'] = wrapper['summary_fit_note']
            return

        if not isinstance(stats, dict):
            return  # Defensive: unexpected return shape.

        from .plots._summary_fit import (
            _normalize_summary_fit_spec,
            render_summary_fit,
        )

        try:
            spec = _normalize_summary_fit_spec(summary_fit_spec)
        except ValueError:
            raise  # §3.8 error grammar — surface to user.

        if consumed_by_normalize:
            stats['summary_fit'] = {}
            stats['summary_fit_note'] = (
                "summary_fit consumed: normalize= active. The legacy normalize "
                "dispatcher predates inline fits; fit= and summary_fit= are "
                "silently consumed. Workaround: compute normalized residuals "
                "into an alias column with adf.add_alias(), then call draw() "
                "on the alias with fit= and summary_fit=."
            )
            return

        stats_fit = stats.get('fit')
        if not stats_fit:
            stats['summary_fit'] = {}
            stats['summary_fit_note'] = (
                "summary_fit requested but stats['fit'] is empty. Causes: "
                "fit= kwarg was not provided; or quantile-band profile mode "
                "(no fit-target curves); or the combination consumed fit "
                "upstream. No figures rendered."
            )
            return

        # Normalize facet_by to a list-of-strings for display.
        if isinstance(facet_by, str):
            facet_cols = [facet_by]
        elif isinstance(facet_by, (list, tuple)):
            facet_cols = list(facet_by)
        else:
            facet_cols = None

        # Resolve data_format precedence: per-call spec > module style > default.
        from .style import get_style_value as _get_style_value
        _data_format = spec.get('data_format')
        if _data_format is None:
            _data_format = _get_style_value('summary_fit.data_format', 'dict')

        # Build a style accessor closure for render_summary_fit. The renderer
        # treats this as a mapping; get_style_value semantics give us per-key
        # default fallback.
        class _ModuleStyleProxy:
            def get(self, key, default=None):
                return _get_style_value(key, default)
            def __contains__(self, key):
                return _get_style_value(key, None) is not None
            def __getitem__(self, key):
                v = _get_style_value(key, None)
                if v is None:
                    raise KeyError(key)
                return v
        _style = _ModuleStyleProxy()

        figs, data, note = render_summary_fit(
            stats_fit,
            kinds=spec['kinds'],
            group_by_col=group_by,
            facet_by_cols=facet_cols,
            params=spec.get('params'),
            mode=spec.get('mode', 'subplots'),
            annotate=spec.get('annotate', False),
            precision=spec.get('precision', 2),
            columns=spec.get('columns'),
            data_format=_data_format,
            title=spec.get('title', 'auto'),
            title_overflow=spec.get('title_overflow', 'shrink'),
            style=_style,
            expr_for_auto_title=expr_for_auto_title,
        )

        if figs:
            stats['summary_fit'] = {**figs, 'data': data}
        else:
            stats['summary_fit'] = {}
            stats['summary_fit_note'] = note or (
                "summary_fit requested but no figures rendered."
            )

    # =========================================================================
    # Expression Parsing
    # =========================================================================
    
    def _parse_expr(self, expr: str):
        """
        Parse TTree::Draw-style expression. Supports scalar and vector forms.
        
        Scalar (backward compatible):
            "y:x"            -> ("y", "x")
            "x"              -> ("x", None)
            "max(a,b):x"     -> ("max(a,b)", "x")  (comma inside parens is safe)
        
        Vector (Phase 13.16.DF):
            "[y1,y2,y3]:x"       -> (["y1","y2","y3"], ["x","x","x"])    (N:1)
            "y:[x1,x2,x3]"       -> (["y","y","y"], ["x1","x2","x3"])    (1:N)
            "[y1,y2]:[x1,x2]"    -> (["y1","y2"], ["x1","x2"])           (N:N)
            "[y1,y2,y3]"         -> (["y1","y2","y3"], [None,None,None]) (1D)
        
        Returns
        -------
        tuple
            Scalar: (str, str|None)
            Vector: (list, list) — same length after broadcasting
        
        Raises
        ------
        ValueError
            Too many top-level ':' separators, broadcast mismatch, empty vector.
        """
        # P0-1 / P1-3: preserve existing validation — reject >1 top-level colon
        colon_count = self._count_colons_outside_brackets(expr)
        if colon_count > 1:
            raise ValueError(
                f"Invalid expression '{expr}'. "
                "Expected 'y:x' or 'x' format."
            )
        
        if colon_count == 0:
            # 1D form (may still be vector)
            return self._parse_expr_1d(expr)
        
        # colon_count == 1: 2D form, may be scalar or vector on either side
        y_part, x_part = self._split_top_level_colon(expr)
        y_list = self._parse_vector_part(y_part)
        x_list = self._parse_vector_part(x_part)
        
        if len(y_list) == 1 and len(x_list) == 1:
            # Scalar path — backward compatible
            return (y_list[0], x_list[0])
        
        # Vector path — broadcast
        ny, nx = len(y_list), len(x_list)
        if ny == 0 or nx == 0:
            raise ValueError(f"Empty vector in expression '{expr}'")
        if ny == nx:
            pass  # N:N
        elif ny == 1:
            y_list = y_list * nx  # 1:N broadcast
        elif nx == 1:
            x_list = x_list * ny  # N:1 broadcast
        else:
            raise ValueError(
                f"Cannot broadcast {ny} y-expressions with {nx} x-expressions "
                f"in '{expr}'. Need N:N, N:1, or 1:N."
            )
        return (y_list, x_list)
    
    def _parse_expr_1d(self, expr: str):
        """Parse 1D expression — scalar or vector."""
        y_list = self._parse_vector_part(expr)
        if len(y_list) == 0:
            raise ValueError(f"Empty expression: '{expr}'")
        if len(y_list) == 1:
            return (y_list[0], None)  # scalar 1D, unchanged
        return (y_list, [None] * len(y_list))  # vector 1D
    
    def _parse_vector_part(self, part: str):
        """
        Parse '[a,b,c]' -> ['a','b','c'] or 'scalar' -> ['scalar'].
        
        Only strips brackets if the ENTIRE part (after stripping whitespace)
        is wrapped in brackets. Uses paren-aware split to respect function calls.
        """
        stripped = part.strip()
        if len(stripped) >= 2 and stripped[0] == '[' and stripped[-1] == ']':
            # Ensure brackets are balanced at the outer level
            inner = stripped[1:-1]
            # Paren-aware split of inner content
            items = self._split_paren_aware(inner)
            return [item.strip() for item in items]
        return [stripped]
    
    def _split_paren_aware(self, s: str):
        """
        Split on top-level commas, respecting parenthesis and bracket depth.
        
        'max(a,b),max(c,d)' -> ['max(a,b)', 'max(c,d)']
        'y1,y2,y3'          -> ['y1', 'y2', 'y3']
        """
        parts = []
        depth = 0
        current = []
        for ch in s:
            if ch in '([{':
                depth += 1
                current.append(ch)
            elif ch in ')]}':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                parts.append(''.join(current))
                current = []
            else:
                current.append(ch)
        if current or (s.endswith(',')):
            parts.append(''.join(current))
        return parts
    
    def _count_colons_outside_brackets(self, expr: str) -> int:
        """Count ':' characters that are NOT inside [...] or (...)."""
        count = 0
        depth = 0
        for ch in expr:
            if ch in '([':
                depth += 1
            elif ch in ')]':
                depth -= 1
            elif ch == ':' and depth == 0:
                count += 1
        return count
    
    def _split_top_level_colon(self, expr: str):
        """Split on the single top-level ':' — assumes exactly one exists."""
        depth = 0
        for i, ch in enumerate(expr):
            if ch in '([':
                depth += 1
            elif ch in ')]':
                depth -= 1
            elif ch == ':' and depth == 0:
                return expr[:i], expr[i+1:]
        raise ValueError(f"No top-level ':' found in '{expr}'")

    def _split_top_level_colons_3(self, expr: str):
        """Split on exactly TWO top-level ':' — returns (z, y, x).

        Phase 13.39.DF: supports 'z:y:x' expressions for draw_profile2d
        (2D profile heatmap) and draw_scatter3d (3D scatter).

        Caller must verify colon_count == 2 before calling — this method
        assumes the contract.
        """
        depth = 0
        positions = []
        for i, ch in enumerate(expr):
            if ch in '([':
                depth += 1
            elif ch in ')]':
                depth -= 1
            elif ch == ':' and depth == 0:
                positions.append(i)
        if len(positions) != 2:
            raise ValueError(
                f"Expected exactly 2 top-level ':' in '{expr}', "
                f"got {len(positions)}."
            )
        p0, p1 = positions
        return expr[:p0], expr[p0+1:p1], expr[p1+1:]
    
    def _eval_column(self, expr: str) -> pd.Series:
        """
        Evaluate column expression.
        
        Supports:
        - Direct column names: "x"
        - Computed expressions: "x + y", "x * 2"
        """
        expr = expr.strip()
        
        # Direct column access
        if expr in self.df.columns:
            return self.df[expr]
        
        # Computed expression via df.eval()
        try:
            return self.df.eval(expr)
        except Exception as e:
            raise ValueError(
                f"Cannot evaluate expression '{expr}': {e}"
            )
    
    def _apply_selection(
        self, 
        df: pd.DataFrame, 
        selection: Optional[Union[str, np.ndarray, callable]]
    ) -> pd.DataFrame:
        """
        Apply selection/cut to DataFrame.
        
        Parameters
        ----------
        df : DataFrame
            Input data.
        selection : str, array, callable, or None
            - str: pandas query string
            - array: boolean mask
            - callable: function(df) -> mask
            - None: no selection
        
        Returns
        -------
        DataFrame
            Filtered data.
        """
        if selection is None:
            return df
        
        if isinstance(selection, str):
            return df.query(selection, engine="python")
        
        if callable(selection):
            mask = selection(df)
            return df[mask]
        
        # Assume boolean mask
        return df[selection]
    
    def _apply_sampling(
        self, 
        df: pd.DataFrame, 
        sample: Optional[int]
    ) -> pd.DataFrame:
        """
        Apply random sampling if requested.
        
        Parameters
        ----------
        df : DataFrame
            Input data.
        sample : int or None
            Maximum number of points. None = no limit.
        
        Returns
        -------
        DataFrame
            Possibly sampled data.
        """
        if sample is None:
            sample = get_style_value("sample.max_points")
        
        if sample is None or len(df) <= sample:
            return df
        
        random_state = get_style_value("sample.random_state", 42)
        return df.sample(n=sample, random_state=random_state)
    
    # =========================================================================
    # Label Resolution (Duck Typing for AliasDataFrame)
    # =========================================================================
    
    def _get_label(self, varname: str):
        """
        Get display label for a variable via duck typing.
        
        Checks if data source has get_axis_title() method (duck typing).
        Returns None if no title is set, allowing underlying plot methods
        to use their own default formatting.
        
        Parameters
        ----------
        varname : str
            Column/variable name
        
        Returns
        -------
        str or None
            Display label string if set, None to use default
        """
        # Duck typing: check if data source provides axis titles
        if hasattr(self, '_data_source') and hasattr(self._data_source, 'get_axis_title'):
            title = self._data_source.get_axis_title(varname)
            if title:
                return title
        return None  # Let plot methods use their default behavior
    
    # =========================================================================
    # Phase 13.13.DF: same=True Support (AD-15 through AD-18)
    # =========================================================================
    
    def _resolve_axes(self, same: bool, ax):
        """
        Resolve axes for drawing.
        
        AD-15: self._last_ax with plt.gca() fallback.
        
        Parameters
        ----------
        same : bool
            If True, reuse last axes.
        ax : Axes or None
            Explicitly provided axes.
        
        Returns
        -------
        tuple
            (resolved_axes_or_None, is_new_figure)
            If resolved_axes is None, caller creates new figure.
        
        Notes
        -----
        For AliasDataFrame users: if AliasDataFrame creates new DFDraw on each
        draw() call, same=True will use plt.gca() fallback instead of instance-
        tracked _last_ax. AD-37 requires AliasDataFrame to cache DFDraw instance
        for safe same=True behavior.
        """
        import matplotlib.pyplot as plt
        
        if ax is not None:
            # Explicit ax= always wins (§5.4 Precedence Rule 2)
            return ax, False
        
        if same:
            if self._last_ax is not None:
                return self._last_ax, False
            else:
                # Fallback to plt.gca() (AD-15)
                current = plt.gca()
                if current.has_data() or len(current.get_children()) > 5:
                    return current, False
                # No valid axes found, create new
                return None, True
        
        # Not same=True: create new figure
        return None, True
    
    def _get_next_color(self):
        """
        Get next color from palette for same=True overlay.
        
        AD-16: Auto-increment colors from palette.
        Color cycle starts at index 1 (first plot uses index 0 via default).
        """
        import matplotlib.pyplot as plt
        palette = plt.colormaps.get_cmap(get_style_value("colors.palette", "tab10"))
        color = palette(self._color_cycle_index % 10)
        self._color_cycle_index += 1
        return color
    
    def _reset_color_cycle(self):
        """
        Reset color cycle when creating new figure.
        
        Starts at 1: first plot implicitly uses palette index 0 (matplotlib default).
        """
        self._color_cycle_index = 1
    
    def _auto_label(self, y_expr, x_expr=None):
        """
        Generate label from expression for legend.
        
        AD-17: Auto-generate label from expression (user can override with label=).
        
        Parameters
        ----------
        y_expr : str
            Y-axis expression name.
        x_expr : str or None
            X-axis expression name.
        
        Returns
        -------
        str
            Label string like "y vs x" or "y".
        """
        if x_expr:
            return f"{y_expr} vs {x_expr}"
        return y_expr
    
    # =========================================================================
    # Vector Expression Support (Phase 13.16.DF)
    # =========================================================================
    
    # Style channel cycles
    _LINESTYLE_CYCLE = ['-', '--', '-.', ':']
    _MARKER_CYCLE = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    _VALID_STYLE_CHANNELS = ('color', 'linestyle', 'marker')

    # =========================================================================
    # Phase 13.16.DF FIX1: Vector dispatch forwarded-name tuples.
    # 
    # Each tuple enumerates the named parameters of one method that must be
    # propagated through vector dispatch via locals().get(name, _MISSING).
    # 
    # Excluded from each tuple:
    #   - self, expr, **kwargs (handled separately)
    #   - group_by (passed as explicit named arg to _draw_vector)
    #   - facet, ncols, sharex, sharey, top_k for profile/hist/scatter
    #     (R4 fail-fast guard catches facet=True; the others are facet-only)
    #   - draw(): also excludes 'type' (consumed for routing) and 'figsize'
    #     (figure created once by draw() before dispatch; per-iteration
    #     forwarding is semantic noise per R3)
    # 
    # All entries are validated against signatures at module import via
    # _validate_forwarded_names() (see end of this file).
    # =========================================================================

    _PROFILE_FORWARDED_NAMES = (
        'selection', 'sample', 'bins', 'range', 'error', 'stats',
        'title', 'xlabel', 'ylabel', 'ax', 'save',
        'top_k',  # included: profile.draw_profile accepts top_k as scalar-mode group filter
        'return_data', 'min_entries',
        'group_by_bins', 'group_by_quantiles', 'sort_groups',
        'weights',
        'auto_title',
        'same',
        'stat_fields',  # Phase 13.18.DF: robust statistics groups
        'quantiles', 'central', 'quantile_mode',  # Phase 13.25.DF: quantile rendering
        'quantile_style',  # Phase 13.26.DF: channel-aware quantile rendering
        'nan_policy',  # Phase 13.28.DF: NaN/inf filter policy (AD-70)
        'facet_by',  # Phase 13.27.DF: facet routing through channel framework (AD-67)
        'facet_by_bins', 'facet_by_quantiles',  # Phase 13.32.DF Sub-fix 3 (AD-79)
        # Phase 13.41.DF v1.6 + FIX1: N-D faceting axis-sharing controls
        'share_x', 'share_y', 'share_across_figures',
        # Phase 13.27.DF Commit 2 (Phase D): selection/weights vectors + per-curve label management
        'selection_vector', 'weights_vector',
        'selection_labels', 'weights_labels',
        'selection_categorical', 'weights_categorical',
        'vector_compose', 'delta_facet',
        # Phase 13.33.DF: Normalized differential profiles (AD-80/81/82)
        'normalize', 'normalize_layout',
        # Phase 13.36.DF: user style override kwargs (BUG-013 fix).
        # draw_profile() has all 3 as explicit params at lines 162-164;
        # adding to FORWARDED_NAMES lets vector-dispatch path forward them.
        'marker', 'color', 'markersize',
        # Phase 13.37.DF: per-group linestyle cycling mode flag. Explicit
        # param of draw_profile() — R6 validator passes.
        'linestyle_cycle',
        # Phase 13.39.DF: time-axis formatting (pre-conversion approach)
        'time_format',
        # Phase 13.42.DF: Inline fits
        'fit',
        # Phase 13.42.DF FIX2 (ADV-3, Sonnet55 P2-2 carry-forward): per-call
        # fit textbox formatting overrides. Pattern B (forwarded inward):
        # draw_profile() has fit_textbox_kwargs as an explicit parameter
        # since FIX1 and consumes it via render_fit_textbox.
        'fit_textbox_kwargs',
    )

    _HIST_FORWARDED_NAMES = (
        'selection', 'sample', 'bins', 'range', 'norm', 'stats',
        'title', 'xlabel', 'ylabel', 'ax', 'save',
        'top_k',  # included: histogram.draw_hist accepts top_k as scalar-mode group filter
        'auto_title',
        'same',
        'stat_fields',  # Phase 13.18.DF: robust statistics groups
        'nan_policy',  # Phase 13.28.DF: NaN/inf filter policy (AD-70)
        'weights',  # Phase 13.27.DF Commit 2 FIX1 (§7b): column-name / expression weighting on hist
        'facet_by',  # Phase 13.32.DF Sub-fix 3: extend AD-78 column-mode facet_by to hist
        'facet_by_bins', 'facet_by_quantiles',  # Phase 13.32.DF Sub-fix 3 (AD-79)
        # Phase 13.41.DF v1.6 + FIX1: N-D faceting axis-sharing controls
        'share_x', 'share_y', 'share_across_figures',
        # Phase 13.27.DF Commit 2 (Phase D): selection/weights vectors + per-curve label management
        'selection_vector', 'weights_vector',
        'selection_labels', 'weights_labels',
        'selection_categorical', 'weights_categorical',
        'vector_compose', 'delta_facet',
        # Phase 13.35.DF: float group_by binning + per-group normalization (BUG-013 fix).
        # Without these, group_by_bins/quantiles/hist_norm/min_entries fall into
        # **kwargs → forwarded to _draw_hist_grouped() **hist_kwargs → reach
        # ax.hist() which raises AttributeError (T2/T3/T4 from v1.3 §3.2).
        'group_by_bins', 'group_by_quantiles',
        'hist_norm',
        'min_entries',
        # Phase 13.36.DF: user style override kwargs (BUG-013 fix).
        # draw_hist() has 'color' as explicit param (consumed by signature);
        # 'marker' is NOT explicit — flows through **kwargs to _draw_hist_grouped()
        # where it gets popped + a UserWarning is issued (markers are meaningless
        # for histograms).
        # NOTE: 'markersize' deliberately NOT added — draw_hist() has no
        # markersize explicit param, so adding it would let it flow to ax.hist()
        # via vector dispatch and crash (matplotlib rejects markersize). Sonet51
        # P1 from v1.2 review.
        'marker', 'color',
        # Phase 13.37.DF: hist_errors (Poisson overlay flag) + linestyle_cycle
        # (per-group linestyle mode flag). Both are explicit params of
        # draw_hist() — R6 validator passes.
        'hist_errors', 'linestyle_cycle',
        # Phase 13.39.DF: time-axis formatting (pre-conversion approach)
        'time_format',
        # Phase 13.40.DF: cumulative histogram (CDF/ECDF/survival)
        'cumulative',
        # Phase 13.42.DF: Inline fits
        'fit',
        # Phase 13.42.DF FIX2 (ADV-3, Sonnet55 P2-2 carry-forward): per-call
        # fit textbox formatting overrides. See _PROFILE_FORWARDED_NAMES note.
        'fit_textbox_kwargs',
    )

    _SCATTER_FORWARDED_NAMES = (
        'selection', 'sample', 'color', 'size', 'marker', 'stats',
        'title', 'xlabel', 'ylabel', 'ax', 'save',
        'top_k',  # included: scatter.draw_scatter accepts top_k as scalar-mode group filter
        'cmap', 'colorbar', 'clabel', 'jitter',
        'same',
        'nan_policy',  # Phase 13.28.DF: NaN/inf filter policy (AD-70)
        'facet_by',  # Phase 13.32.DF Sub-fix 3: extend AD-78 column-mode facet_by to scatter
        'facet_by_bins', 'facet_by_quantiles',  # Phase 13.32.DF Sub-fix 3 (AD-79)
        # Phase 13.41.DF v1.6 + FIX1: N-D faceting axis-sharing controls
        'share_x', 'share_y', 'share_across_figures',
        'xerr', 'yerr',  # Phase 13.38.DF: scatter error bars (column name or df.eval())
        'time_format',  # Phase 13.39.DF: time-axis formatting (pre-conversion)
        # Phase 13.27.DF Commit 2 (Phase D): selection/weights vectors + per-curve label management
        # NB: weights_vector accepted by scatter() but the per-curve weights have no
        # effect on scatter rendering — proposal §5.5; one-time UserWarning emitted.
        'selection_vector', 'weights_vector',
        'selection_labels', 'weights_labels',
        'selection_categorical', 'weights_categorical',
        'vector_compose', 'delta_facet',
        # Phase 13.42.DF: Inline fits
        'fit',
        # Phase 13.42.DF FIX2 (ADV-3, Sonnet55 P2-2 carry-forward): per-call
        # fit textbox formatting overrides. See _PROFILE_FORWARDED_NAMES note.
        'fit_textbox_kwargs',
    )
    # (previously absent — hist2d used inline kwargs handling).
    _HIST2D_FORWARDED_NAMES = (
        'selection', 'sample', 'bins', 'range', 'norm', 'stats',
        'title', 'xlabel', 'ylabel', 'ax', 'save',
        'top_k',
        'cmap', 'colorbar', 'clabel', 'vmin', 'vmax',
        'auto_title',
        'same',
        'stat_fields',
        'nan_policy',
        'facet_by', 'facet_by_bins', 'facet_by_quantiles',
        # Phase 13.41.DF v1.6 + FIX1: N-D faceting axis-sharing controls
        'share_x', 'share_y', 'share_across_figures',
    )

    _DRAW_FORWARDED_NAMES = (
        'selection', 'color', 'size', 'marker',
        'bins', 'stats', 'norm', 'title', 'ax', 'sample', 'save',
        'same',
        'nan_policy',  # Phase 13.28.DF: NaN/inf filter policy (AD-70)
        'facet_by',  # Phase 13.32.DF Sub-fix 3: facet_by reachable from draw() dispatcher
        'facet_by_bins', 'facet_by_quantiles',  # Phase 13.32.DF Sub-fix 3 (AD-79)
        # Phase 13.41.DF v1.6 + FIX1: N-D faceting axis-sharing controls
        'share_x', 'share_y', 'share_across_figures',
        # Phase 13.27.DF Commit 2 (Phase D): selection/weights vectors + per-curve label management
        'selection_vector', 'weights_vector',
        'selection_labels', 'weights_labels',
        'selection_categorical', 'weights_categorical',
        'vector_compose', 'delta_facet',
        # Phase 13.42.DF: Inline fits
        'fit',
        # Note: 'type' consumed for routing; 'figsize' deliberately excluded
        # (figure already created); 'facet' caught by R4 guard; 'group_by'
        # passed as explicit named arg to _draw_vector.
        # Phase 13.33.DF (AD-80/81/82): normalize / normalize_layout are
        # profile-only and intentionally NOT in this tuple — auto-forwarding
        # them would leak the kwargs to hist/scatter dispatch paths (and
        # bomb on matplotlib's `ax.hist(**kwargs)` since Polygon doesn't
        # accept normalize_layout). They remain in the draw() signature
        # for users routing via d.draw(type='profile', normalize='delta'),
        # and reach profile() via **kwargs in the routing dispatch.
    )

    # Private kwargs that _draw_vector injects into iter_kwargs to suppress
    # per-iteration legend/title/tight_layout in the underlying plot modules.
    _VECTOR_SUPPRESS_KWARGS = ('_suppress_legend', '_suppress_title', '_suppress_layout')

    # =========================================================================
    # Phase 13.32.DF Sub-fix 3 (AD-79, v1.2 §3.3): shared validation helper
    # for facet_by_bins/facet_by_quantiles. Used by profile()/hist()/hist2d()/
    # scatter() at entry. Ensures the binning kwargs are consistent with the
    # facet_by tagged union (must be column-mode, must not collide).
    # =========================================================================

    @staticmethod
    def _validate_facet_by_binning(facet_by, facet_by_bins, facet_by_quantiles, df):
        """Validate facet_by_bins/_quantiles inputs at plot-method entry.

        Raises ValueError on:
          - facet_by_bins/_quantiles set without facet_by= (orphan binning)
          - both facet_by_bins AND facet_by_quantiles set (mutex)
          - boolean True passed (must be integer, per AD-40 pattern)
          - facet_by is not a column name (e.g. channel enum 'group_by')
        
        Phase 13.41.DF v1.3 CP1-1 (Sonnet54 P1-A): list-form facet_by is
        validated by _normalize_facet_args in _dispatch_faceted_render.
        This str-only validator must early-return on list input to avoid
        `facet_by not in df.columns` raising TypeError: unhashable type 'list'.
        """
        # Phase 13.41.DF v1.3 CP1-1: list-form deferred to _normalize_facet_args
        if isinstance(facet_by, list):
            return
        
        if facet_by_bins is None and facet_by_quantiles is None:
            return
        if facet_by is None:
            raise ValueError(
                "facet_by_bins/facet_by_quantiles requires facet_by= to be set"
            )
        if facet_by_bins is not None and facet_by_quantiles is not None:
            raise ValueError(
                "Cannot specify both facet_by_bins and facet_by_quantiles"
            )
        if isinstance(facet_by_bins, bool) and facet_by_bins:
            raise ValueError(
                "facet_by_bins must be an integer (number of bins), not True. "
                "Example: facet_by_bins=5"
            )
        if isinstance(facet_by_quantiles, bool) and facet_by_quantiles:
            raise ValueError(
                "facet_by_quantiles must be an integer (number of quantile "
                "bins), not True. Example: facet_by_quantiles=5"
            )
        if df is None or facet_by not in df.columns:
            raise ValueError(
                f"facet_by_bins/facet_by_quantiles requires facet_by to be a "
                f"DataFrame column name (got facet_by={facet_by!r} which is a "
                f"channel name or not in df.columns)"
            )

    # =========================================================================
    # Phase 13.27.DF Commit 2 (Phase D, v1.2 §5.2 + §4.4): helpers for
    # selection_vector / weights_vector composition. Used by _draw_vector to
    # build per-curve iter_kwargs.
    # =========================================================================

    @staticmethod
    def _compute_vector_iteration_indices(n_y, selection_vector, weights_vector,
                                          vector_compose):
        """Phase 13.27.DF Commit 2 (v1.2 §4.2 + §5.2): compute the per-curve
        index triples (y_idx, sel_idx, w_idx) for the _draw_vector loop.

        Returns a list of length n_curves. Each entry is a 3-tuple of indices:
            y_idx   — index into y_list / x_list (always set)
            sel_idx — index into selection_vector (None if not active)
            w_idx   — index into weights_vector (None if not active)

        When both selection_vector and weights_vector are None or have length
        <= 1 (per AD-67: 1-element silently degrades to scalar), the result is
        bit-identical to the pre-Commit-2 zip(y_list, x_list) iteration:
            [(0, None, None), (1, None, None), ..., (n_y-1, None, None)]

        Raises ValueError per proposal §4.2 edge cases:
          - empty list anywhere
          - inner with mismatched lengths
          - invalid vector_compose value
        """
        # AD-67: 1-element silently degrades to scalar (cost-0 channel).
        # Treat as if the vector were None for iteration-count purposes.
        n_s = 0 if not selection_vector else len(selection_vector)
        n_w = 0 if not weights_vector else len(weights_vector)

        # §4.2.2 edge case — empty list raises
        if selection_vector is not None and n_s == 0:
            raise ValueError("selection_vector must be non-empty (use None to omit)")
        if weights_vector is not None and n_w == 0:
            raise ValueError("weights_vector must be non-empty (use None to omit)")

        # 1-element degrades — treat as "not active" for cardinality purposes
        sel_active = n_s >= 2
        w_active = n_w >= 2

        # No active list-valued channels → backward-compat zip behavior
        if not sel_active and not w_active:
            return [(i, None, None) for i in range(n_y)]

        if vector_compose == "inner":
            # All active axes must match length (n_y is always active)
            active_lengths = {n_y}
            if sel_active:
                active_lengths.add(n_s)
            if w_active:
                active_lengths.add(n_w)
            if len(active_lengths) > 1:
                raise ValueError(
                    f"3-axis inner requires equal lengths: vector={n_y}, "
                    f"selection_vector={n_s or 'unused'}, "
                    f"weights_vector={n_w or 'unused'}"
                )
            n_curves = n_y
            return [
                (i,
                 i if sel_active else None,
                 i if w_active else None)
                for i in range(n_curves)
            ]
        elif vector_compose == "outer":
            # Cross-product over active axes; 1-element degrades to a single
            # "None" index for that dimension.
            sel_range = range(n_s) if sel_active else [None]
            w_range = range(n_w) if w_active else [None]
            out = []
            for y_i in range(n_y):
                for s_i in sel_range:
                    for w_i in w_range:
                        out.append((y_i, s_i, w_i))
            return out
        else:
            raise ValueError(
                f"vector_compose must be 'inner' or 'outer', got "
                f"{vector_compose!r}"
            )

    @staticmethod
    def _combine_selections(global_sel, per_curve_sel):
        """Phase 13.27.DF Commit 2 (v1.2 §4.4): logical-AND composition of
        global `selection` with per-curve `selection_vector[i]`.

        Returns a string expression suitable for df.eval (per Class-1 contract).
        None propagates: combine(None, None) → None; combine(s, None) → s.
        """
        if global_sel and per_curve_sel:
            return f"({global_sel}) & ({per_curve_sel})"
        return global_sel or per_curve_sel  # None-safe

    @staticmethod
    def _combine_weights(global_w, per_curve_w):
        """Phase 13.27.DF Commit 2 (v1.2 §4.4): multiplicative composition of
        global `weights` with per-curve `weights_vector[i]`.

        Returns a string expression suitable for df.eval. None propagates.
        """
        if global_w and per_curve_w:
            return f"({global_w}) * ({per_curve_w})"
        return global_w or per_curve_w

    # =========================================================================
    # Phase 13.30.DF v1.0 — Class-2 column-reference parameter tuples.
    # (RESTORED in Phase 13.31 after the initial Phase 13.31 patch was
    # generated from a pre-Phase-13.30 source snapshot and accidentally
    # clobbered this block. See Phase 13.31 lesson note in the commit msg.)
    #
    # Mirror of _*_FORWARDED_NAMES discipline: single source of truth + one
    # validation loop per plot type. plots/_validation.py iterates these to
    # enforce that string-valued column-reference parameters name real columns,
    # raising ValueError instead of silently falling back to ungrouped mode.
    #
    # Parameter class taxonomy (Phase 13.30 proposal §3):
    #   Class 2 — column reference (strict): listed here.
    #     Examples: group_by (today); selection_vector, weights_vector
    #     (Phase 13.27 Commit 2 — anticipated by *_COLUMN_REFERENCE_LISTS).
    #   Class 4 — expression-or-column (permissive): NOT listed here.
    #     Examples: weights (handled by _eval_weights with df.eval fallback).
    #
    # Phase 13.31.DF (AD-78) NOTE: 'facet_by' is a tagged union (channel-name
    # enum OR DataFrame column name) and is therefore NOT a Class-2 parameter.
    # 'facet_by' must NOT be added to these tuples — Sonnet P1 catch in AD-78
    # cross-review. facet_by validation lives in _dispatch_faceted_render()
    # via the AD-78 §2 disambiguation algorithm.
    #
    # Adding a Class-4 parameter here would break the existing
    # expression-accepting contract.
    #
    # Validated at module import by _validate_forwarded_names() (extended for
    # this phase): every entry must be a real parameter of the target method.
    # =========================================================================

    _PROFILE_COLUMN_REFERENCES = ('group_by',)
    _HIST_COLUMN_REFERENCES    = ('group_by',)
    _SCATTER_COLUMN_REFERENCES = ('group_by',)
    _DRAW_COLUMN_REFERENCES    = ('group_by',)
    # hist2d / hexbin have no group_by parameter — empty tuples document that
    # explicitly. Adding group_by to those plot types in a future phase requires
    # also appending here.
    _HIST2D_COLUMN_REFERENCES  = ()
    _HEXBIN_COLUMN_REFERENCES  = ()

    # Reserved for Phase 13.27.DF Commit 2 (selection_vector / weights_vector).
    # Each entry names a parameter that is List[str] of column references.
    # Empty for v1.0 of Phase 13.30; Commit 2 fills them and adds a list-aware
    # validator overload.
    _PROFILE_COLUMN_REFERENCE_LISTS = ()
    _HIST_COLUMN_REFERENCE_LISTS    = ()
    _SCATTER_COLUMN_REFERENCE_LISTS = ()

    def _draw_vector(self, y_list, x_list, draw_method,
                     vector_style=None, group_style='color',
                     group_by=None, **kwargs):
        """
        Draw multiple (y, x) pairs overlaid on one axes (Phase 13.16.DF).
        
        Parameters
        ----------
        y_list, x_list : list
            Parallel lists from _parse_expr (already broadcast to same length).
        draw_method : bound method
            One of self.profile, self.hist, self.scatter.
        vector_style : str, optional
            Channel distinguishing vector curves: 'color', 'linestyle', 'marker'.
            Context-dependent default:
                - Without group_by: 'color' (existing same=True cycle)
                - With group_by:    'linestyle' (color reserved for groups)
        group_style : str, default 'color'
            Channel distinguishing groups (when group_by is set).
        group_by : str, optional
            Grouping column, passed through to each iteration.
        **kwargs
            All other parameters passed to draw_method.
        
        Returns
        -------
        (fig, ax, stats_list)
            stats_list is list[dict] — one entry per (y_i, x_i) pair.
        """
        # P0-4: extract 'same' from outer kwargs to avoid collision
        outer_same = kwargs.pop('same', False)

        # =====================================================================
        # Phase 13.27.DF Commit 2 (Phase D, v1.2 §5.2): extract list-valued
        # selection/weights kwargs and per-curve label management kwargs.
        # These never propagate as scalar kwargs to draw_method — they govern
        # the per-iteration composition (see §4.2 + §4.4).
        # =====================================================================
        _selection_vector  = kwargs.pop('selection_vector',  None)
        _weights_vector    = kwargs.pop('weights_vector',    None)
        _selection_labels  = kwargs.pop('selection_labels',  None)
        _weights_labels    = kwargs.pop('weights_labels',    None)
        _selection_categorical = kwargs.pop('selection_categorical', False)
        _weights_categorical   = kwargs.pop('weights_categorical',   False)
        _vector_compose    = kwargs.pop('vector_compose',    'inner')
        _delta_facet       = kwargs.pop('delta_facet',       None)

        # NB: scatter() emits UserWarning + drops weights_vector at its method
        # entry (proposal §5.5). By the time we get here in scatter's vector
        # path, weights_vector has already been forced to None. No additional
        # check needed in _draw_vector.

        # P1-1 + GPT5 fix: only reset color cycle when NOT chaining onto existing overlay
        if not outer_same:
            self._reset_color_cycle()
        # else: continue existing cycle (preserves SAME.axes_reuse contract)
        
        # P1-10 / architect Q4: auto_title defaults True for vector mode
        # BUT only for methods that actually accept auto_title (profile, hist)
        # Scatter does not have auto_title in its signature.
        import inspect
        try:
            sig_params = inspect.signature(draw_method).parameters
            supports_auto_title = 'auto_title' in sig_params
        except (TypeError, ValueError):
            supports_auto_title = False
        if supports_auto_title and 'auto_title' not in kwargs:
            kwargs['auto_title'] = True
        # If the caller passed auto_title but the method doesn't support it,
        # strip it to avoid matplotlib TypeError.
        if not supports_auto_title:
            kwargs.pop('auto_title', None)
        
        # Phase 13.16.DF FIX1 + 13.26.DF Phase B contract:
        # Reject obvious user-vs-user channel collisions before Algorithm A,
        # preserving the existing error message ("cannot both use ..."). This
        # short-circuit only fires when the user explicitly set
        # vector_style == group_style; downstream Algorithm A still handles
        # subtler collisions (e.g., per-call kwarg vs style.default).
        if (group_by is not None
            and vector_style is not None
            and vector_style == group_style):
            raise ValueError(
                f"vector_style and group_style cannot both use {vector_style!r}. "
                f"Choose different channels from {self._VALID_STYLE_CHANNELS}."
            )

        # Context-dependent default for vector_style.
        # Phase 13.26.DF Phase B: Algorithm A determines the assignment via
        # assign_channels() (channels.py). The function takes the active data
        # channels (vector + optional group_by + optional quantile-discrete)
        # and returns {channel_name: visual_channel}. Per-call kwargs
        # (vector_style, group_style) act as DataChannel.requested_style
        # overrides — Algorithm A's resolution chain handles them.
        from .channels import DataChannel, assign_channels

        # Determine quantile channel cost (Step 0 — zero-cost modes consume
        # no visual channel slot). The actual quantile_mode resolution lives
        # in plots/profile.py:_detect_quantile_mode; we mirror its decision
        # here only to know whether quantiles add a channel.
        _quantiles = kwargs.get('quantiles')
        _quantile_mode = kwargs.get('quantile_mode', 'auto')
        _quantile_style_kwarg = kwargs.get('quantile_style')
        _has_quantile_channel = False
        _q_card = 0
        if _quantiles is not None and len(_quantiles) > 0:
            _resolved_mode = _quantile_mode
            if _resolved_mode == 'auto':
                from .plots.profile import _detect_quantile_mode
                _resolved_mode = _detect_quantile_mode(list(_quantiles))
            if _resolved_mode == 'discrete':
                _has_quantile_channel = True
                _q_card = len(_quantiles)

        # Build DataChannel list (vector is always present in _draw_vector;
        # group_by and quantile are optional cost-bearing channels).
        # group_style defaults to 'color' in this method's signature; pass
        # it faithfully so Algorithm A can detect per-call collisions.
        _channels = [
            DataChannel(
                'vector',
                is_categorical=True,
                cardinality=len(y_list),
                requested_style=vector_style,
                cost=1,
            ),
        ]
        if group_by is not None:
            _g_card = get_style_value("channels.cycles.color_count", 10)
            _channels.append(DataChannel(
                'group_by',
                is_categorical=True,
                cardinality=_g_card,
                # group_style default 'color' aligns with EXPLICIT_RULES
                # entry for {vector, group_by}; passing it as requested_style
                # is consistent with the precedence chain.
                requested_style=group_style,
                cost=1,
            ))
        if _has_quantile_channel:
            _channels.append(DataChannel(
                'quantiles',
                is_categorical=False,
                cardinality=_q_card,
                requested_style=_quantile_style_kwarg,
                cost=1,
            ))

        # Phase 13.27.DF Commit 2 (v1.2 §3.1): selection_delta / weights_delta
        # channels. AD-67: 1-element list silently degrades to scalar (cost 0,
        # not added to _channels). AD-61: is_categorical controlled by
        # _selection_categorical / _weights_categorical kwarg (default False
        # = ordinal, linestyle-preferred greedy fallback).
        _sel_active = _selection_vector is not None and len(_selection_vector) >= 2
        _w_active   = _weights_vector   is not None and len(_weights_vector)   >= 2
        if _sel_active:
            _channels.append(DataChannel(
                'selection_delta',
                is_categorical=_selection_categorical,
                cardinality=len(_selection_vector),
                requested_style=None,  # resolved via Algorithm A (EXPLICIT_RULES)
                cost=1,
            ))
        if _w_active:
            _channels.append(DataChannel(
                'weights_delta',
                is_categorical=_weights_categorical,
                cardinality=len(_weights_vector),
                requested_style=None,
                cost=1,
            ))

        _assignment = assign_channels(_channels)

        # Resolve scalar styles from the assignment.
        vector_style = _assignment.get('vector', vector_style or 'color')
        if group_by is not None:
            group_style = _assignment.get('group_by', group_style)
        if _has_quantile_channel:
            kwargs['quantile_style'] = _assignment.get('quantiles')
        
        # Validate channel names
        if vector_style not in self._VALID_STYLE_CHANNELS:
            raise ValueError(
                f"vector_style must be one of {self._VALID_STYLE_CHANNELS}, "
                f"got {vector_style!r}"
            )
        if group_style not in self._VALID_STYLE_CHANNELS:
            raise ValueError(
                f"group_style must be one of {self._VALID_STYLE_CHANNELS}, "
                f"got {group_style!r}"
            )
        
        # Channel collision check (only meaningful when group_by is set)
        if group_by is not None and vector_style == group_style:
            raise ValueError(
                f"vector_style and group_style cannot both use {vector_style!r}. "
                f"Choose different channels from {self._VALID_STYLE_CHANNELS}."
            )
        
        stats_list = []
        per_curve_sanitize = []  # Phase 13.27.DF Commit 2 v1.2 §5.2.1
        fig, ax = None, None

        # Phase 13.27.DF Commit 2 (v1.2 §5.2): compute iteration indices.
        # When selection_vector/weights_vector are None or 1-element, this is
        # bit-identical to the pre-Commit-2 zip(y_list, x_list) iteration:
        # [(0, None, None), (1, None, None), ..., (n_y-1, None, None)].
        iteration_indices = self._compute_vector_iteration_indices(
            n_y=len(y_list),
            selection_vector=_selection_vector,
            weights_vector=_weights_vector,
            vector_compose=_vector_compose,
        )
        n_iter = len(iteration_indices)

        # Phase 13.27.DF Commit 2 (v1.2 §3.1): assigned visual channels for
        # the per-curve delta channels. None when the channel is not active.
        _selection_visual = _assignment.get('selection_delta')
        _weights_visual   = _assignment.get('weights_delta')

        for i, (y_idx, sel_idx, w_idx) in enumerate(iteration_indices):
            y = y_list[y_idx]
            x = x_list[y_idx]
            expr = f"{y}:{x}" if x is not None else y
            iter_kwargs = dict(kwargs)

            # Phase 13.42.DF FIX1 (D5/R2): vector fit pairing. v1.0 passed the
            # full fit list to each Y-channel iteration ⇒ compound-broadcast
            # on every curve, deviating from v1.4 §6.3 verbatim spec. Per
            # architect 2026-05-26 ("Pairs yes or vector-scalar"): slice the
            # fit list per channel (pairing); scalar broadcasts to all
            # channels (v1.0 behavior preserved for scalar input).
            _user_fit = iter_kwargs.get('fit', None)
            if _user_fit is not None and isinstance(_user_fit, list):
                n_channels = len(y_list)
                if len(_user_fit) != n_channels:
                    raise ValueError(
                        f"[vector_fit] length mismatch: {len(_user_fit)} fits "
                        f"vs {n_channels} channels. Fix: provide list of length "
                        f"{n_channels} (per-channel pairing) or a single fit "
                        f"(broadcast)."
                    )
                # D5/I-2 (v1.1): nested-list form e.g. fit=[[a],[b,c]] is
                # explicitly NOT in FIX1 scope; defer to FIX2. Each pairing
                # slot is one fit spec.
                _slot = _user_fit[y_idx]
                if isinstance(_slot, list):
                    raise ValueError(
                        "[vector_fit] nested list fit spec (e.g. "
                        "[[fit_a],[fit_b,fit_c]]) is not supported in FIX1. "
                        "Fix: use scalar fit (broadcast) or flat list of "
                        "length N (pairing). FIX2 may add nested-list compound."
                    )
                iter_kwargs['fit'] = _slot
            # Scalar (str/dict/callable) → keep as-is; downstream broadcasts.

            # Phase 13.27.DF Commit 2 (v1.2 §4.4): logical AND composition of
            # global selection with per-curve selection_vector[sel_idx];
            # multiplicative composition of global weights with per-curve
            # weights_vector[w_idx]. Done at string level — draw_method sees
            # a single composed selection/weights string per iteration.
            if _sel_active and sel_idx is not None:
                iter_kwargs['selection'] = self._combine_selections(
                    kwargs.get('selection'), _selection_vector[sel_idx]
                )
            if _w_active and w_idx is not None:
                iter_kwargs['weights'] = self._combine_weights(
                    kwargs.get('weights'), _weights_vector[w_idx]
                )

            # Apply vector style channel for this iteration.
            # Phase 13.16.DF FIX1 B1b: use setdefault so user-supplied
            # linestyle/marker survives instead of being clobbered.
            # Phase 13.26.DF Phase B: cycles read from style keys
            # (channels.cycles.linestyle / channels.cycles.marker), replacing
            # the _LINESTYLE_CYCLE / _MARKER_CYCLE class constants.
            # Phase 13.27.DF Commit 2: use y_idx (NOT loop counter i) for the
            # cycle position — preserves bit-identical behavior in the no-vec
            # case (y_idx == i then) and produces stable styling per y in the
            # outer-compose case.
            if vector_style == 'linestyle':
                _ls_cycle = get_style_value(
                    "channels.cycles.linestyle",
                    list(self._LINESTYLE_CYCLE),
                )
                iter_kwargs.setdefault(
                    'linestyle',
                    _ls_cycle[y_idx % len(_ls_cycle)],
                )
                # P1-2: suppress same=True color cycle so group_by colors are preserved
                if group_by is not None:
                    iter_kwargs['_suppress_color_cycle'] = True
            elif vector_style == 'marker':
                _mk_cycle = get_style_value(
                    "channels.cycles.marker",
                    list(self._MARKER_CYCLE),
                )
                iter_kwargs.setdefault(
                    'marker',
                    _mk_cycle[y_idx % len(_mk_cycle)],
                )
                if group_by is not None:
                    iter_kwargs['_suppress_color_cycle'] = True
            # vector_style == 'color': rely on existing same=True color cycle

            # Phase 13.27.DF Commit 2 (v1.2 §3.1): apply selection_delta /
            # weights_delta visual channels independently. Algorithm A
            # guarantees disjoint visual assignments — these branches never
            # clobber the vector_style branch above.
            if _sel_active and sel_idx is not None:
                if _selection_visual == 'linestyle':
                    _ls_cycle = get_style_value(
                        "channels.cycles.linestyle",
                        list(self._LINESTYLE_CYCLE),
                    )
                    iter_kwargs.setdefault(
                        'linestyle',
                        _ls_cycle[sel_idx % len(_ls_cycle)],
                    )
                elif _selection_visual == 'marker':
                    _mk_cycle = get_style_value(
                        "channels.cycles.marker",
                        list(self._MARKER_CYCLE),
                    )
                    iter_kwargs.setdefault(
                        'marker',
                        _mk_cycle[sel_idx % len(_mk_cycle)],
                    )
                # 'color' relies on same=True color cycle (advances per-iteration)
            if _w_active and w_idx is not None:
                if _weights_visual == 'linestyle':
                    _ls_cycle = get_style_value(
                        "channels.cycles.linestyle",
                        list(self._LINESTYLE_CYCLE),
                    )
                    iter_kwargs.setdefault(
                        'linestyle',
                        _ls_cycle[w_idx % len(_ls_cycle)],
                    )
                elif _weights_visual == 'marker':
                    _mk_cycle = get_style_value(
                        "channels.cycles.marker",
                        list(self._MARKER_CYCLE),
                    )
                    iter_kwargs.setdefault(
                        'marker',
                        _mk_cycle[w_idx % len(_mk_cycle)],
                    )

            if group_by is not None:
                iter_kwargs['group_by'] = group_by

            # First iteration uses outer same; subsequent always same=True
            iter_kwargs['same'] = outer_same if i == 0 else True

            # Phase 13.16.DF FIX1 (B2-B5): suppress per-iteration legend / title /
            # tight_layout in the underlying plot modules. We perform a single
            # post-loop pass below for legend and layout.
            # Title: suppress on iterations 0..N-2; let the LAST iteration's
            # title through (the plot module's auto_title logic handles it
            # correctly for the final y-var). Post-loop helper may override
            # with a better common-prefix title if the auto_title import works.
            iter_kwargs['_suppress_legend'] = True
            iter_kwargs['_suppress_title'] = (i < n_iter - 1)
            iter_kwargs['_suppress_layout'] = True

            fig, ax, stats = draw_method(expr, **iter_kwargs)
            stats_list.append(stats)

            # Phase 13.27.DF Commit 2 v1.2 §5.2.1: aggregate per-curve sanitize stats
            if isinstance(stats, dict) and 'sanitize_stats' in stats:
                per_curve_sanitize.append(stats['sanitize_stats'])
        
        # Phase 13.16.DF FIX1 (B2): post-loop main-group legend dedup.
        # Underlying plot modules collected handles via ax.scatter/plot label= but
        # we suppressed their legend calls. Now build a deduplicated legend.
        if group_by is not None and ax is not None:
            self._add_vector_main_legend_dedup(ax)
        
        # P1-6: secondary legend for vector + group_by (existing behavior)
        if group_by is not None and ax is not None and vector_style != 'color':
            self._add_vector_legend(ax, y_list, x_list, vector_style)
        
        # Phase 13.16.DF FIX1 (B3+B4): post-loop title.
        # Only needed when group_by is set — each iteration produces a group-specific
        # title that needs dedup. Without group_by, the last iteration's title
        # (produced by _suppress_title=False on the final iteration) is correct.
        if group_by is not None and ax is not None:
            self._set_vector_title_with_groupby(
                ax, y_list, x_list,
                explicit_title=kwargs.get('title'),
                auto_title=kwargs.get('auto_title', False),
                group_by=group_by,
                selection=kwargs.get('selection'),
                weights=kwargs.get('weights'),
            )
        
        # Phase 13.16.DF FIX1: auto_title failsafe for no-group_by case.
        # If auto_title was requested and the axes still have no title after
        # the loop (can happen if apply_auto_title from the plot module didn't
        # persist across same=True iterations), build a minimal title here.
        if (ax is not None and kwargs.get('auto_title')
                and not kwargs.get('title') and not ax.get_title()):
            # Build a simple title from y expressions + x
            unique_ys = list(dict.fromkeys(y_list))
            # Common-prefix y name like "y*" if common prefix ≥ 2 chars, else list
            import os.path as _ospath
            prefix = _ospath.commonprefix(unique_ys) if len(unique_ys) > 1 else unique_ys[0]
            if len(unique_ys) > 1 and len(prefix) >= 2:
                y_label = f"{prefix}*"
            elif len(unique_ys) == 1:
                y_label = unique_ys[0]
            else:
                y_label = "[" + ",".join(unique_ys) + "]"
            x_label = x_list[0] if x_list and x_list[0] is not None else None
            if x_label:
                ax.set_title(f"{y_label} vs {x_label}")
            else:
                ax.set_title(y_label)
        
        # P1-8: deterministic y-axis label for vector (existing behavior)
        if ax is not None:
            self._set_vector_ylabel(ax, y_list, x_list)
        
        # Phase 13.16.DF FIX1 (B5): single tight_layout call at end.
        # Phase 13.16.DF FIX1 (B5): single tight_layout call at end.
        # Suppressed when called from draw_batch (constrained_layout=True).
        if fig is not None and not kwargs.get('_suppress_layout', False):
            try:
                import matplotlib.pyplot as _plt
                _plt.tight_layout()
            except Exception:
                pass  # tight_layout warnings are non-fatal
        
        return fig, ax, stats_list
    
    # =========================================================================
    # Phase 13.16.DF FIX1 helpers (B2, B3+B4)
    # =========================================================================
    
    def _add_vector_main_legend_dedup(self, ax):
        """
        Phase 13.16.DF FIX1 (B2): Build deduplicated main group legend.
        
        After the vector loop, the axes hold N×N_groups labeled artists (one
        labeled artist per group, per vector iteration). This collapses them
        to N_groups by keeping only the first occurrence of each label.
        
        Called only when group_by is set; ignored otherwise (the secondary
        Variable legend is still added by _add_vector_legend below).
        """
        handles, labels = ax.get_legend_handles_labels()
        # Deduplicate while preserving order
        seen = set()
        unique = [(h, l) for h, l in zip(handles, labels)
                  if not (l in seen or seen.add(l))]
        if unique:
            uh, ul = zip(*unique)
            ax.legend(uh, ul, loc=get_style_value("legend.loc", "best"))
    
    def _set_vector_title_with_groupby(self, ax, y_list, x_list,
                                       explicit_title=None,
                                       auto_title=False,
                                       group_by=None,
                                       selection=None,
                                       weights=None):
        """
        Phase 13.16.DF FIX1 (B3+B4): Build a single title for the vector plot.
        
        Resolution order:
          1. explicit_title (user passed title='...')                 → use as-is
          2. auto_title truthy + axes already has title from suppress → leave it
          3. auto_title truthy                                        → build via build_auto_title
          4. otherwise                                                → no title
        
        The reason (3) re-builds rather than letting plot modules build it
        per-iteration is that per-iteration titles use a single y-name; the
        vector title should reflect the common prefix (handled by callers
        of build_auto_title via parts/group_by).
        """
        if explicit_title:
            ax.set_title(explicit_title)
            return
        if not auto_title:
            return
        # Use common-prefix y_name for vector
        # The auto_title helpers live in plots._auto_title (sibling to drawer.py).
        try:
            from .plots._auto_title import (
                parse_auto_title_parts, build_auto_title, apply_auto_title,
            )
        except ImportError:
            return  # auto_title module not available; leave title blank
        # Pick representative names
        x_name = x_list[0] if x_list and x_list[0] is not None else None
        # Common-prefix y name (mirror _set_vector_ylabel)
        common = self._common_prefix(y_list) if y_list else None
        y_name = common if common else (y_list[0] if y_list else None)
        try:
            parts = parse_auto_title_parts(auto_title)
            td = build_auto_title(
                x_name, y_name, group_by=group_by,
                selection=selection, weights=weights, parts=parts,
            )
            apply_auto_title(ax, td)
        except Exception:
            # Title building failed (e.g. unusual signature); silent fallback
            pass
    
    @staticmethod
    def _common_prefix(strings):
        """Return longest common prefix of a list of strings (or None if empty)."""
        if not strings:
            return None
        shortest = min(strings, key=len)
        for i, ch in enumerate(shortest):
            if any(s[i] != ch for s in strings):
                return shortest[:i] if i > 0 else None
        return shortest
    
    def _add_vector_legend(self, ax, y_list, x_list, vector_style):
        """
        Add a secondary legend identifying vector channel meaning.
        
        When group_by is set, the main legend shows groups (colors).
        This adds a small secondary legend showing which linestyle/marker
        corresponds to which y expression. P1-6.
        """
        from matplotlib.lines import Line2D
        
        unique_pairs = []
        seen = set()
        for y, x in zip(y_list, x_list):
            key = (y, x)
            if key not in seen:
                seen.add(key)
                unique_pairs.append(key)
        
        proxies = []
        labels = []
        for i, (y, x) in enumerate(unique_pairs):
            label = f"{y} vs {x}" if x is not None else str(y)
            if vector_style == 'linestyle':
                # Phase 13.26.DF Phase B: read cycle from style
                _ls_cycle = get_style_value(
                    "channels.cycles.linestyle",
                    list(self._LINESTYLE_CYCLE),
                )
                ls = _ls_cycle[i % len(_ls_cycle)]
                proxies.append(Line2D([0], [0], color='black', linestyle=ls))
            elif vector_style == 'marker':
                _mk_cycle = get_style_value(
                    "channels.cycles.marker",
                    list(self._MARKER_CYCLE),
                )
                mk = _mk_cycle[i % len(_mk_cycle)]
                proxies.append(Line2D([0], [0], color='black', marker=mk,
                                     linestyle='', markerfacecolor='black'))
            else:
                continue
            labels.append(label)
        
        if not proxies:
            return
        
        # Preserve existing group-legend by re-adding it as a secondary artist
        first_legend = ax.get_legend()
        second_legend = ax.legend(proxies, labels, loc='lower right',
                                  title='Variable', fontsize='small',
                                  framealpha=0.8)
        if first_legend is not None:
            ax.add_artist(first_legend)
    
    def _set_vector_ylabel(self, ax, y_list, x_list):
        """
        Deterministic y-axis label for vector plots. P1-8.
        
        Rule:
          1. Dedupe y expressions (preserve order)
          2. If all share a common prefix of length >= 2: use "{prefix}*"
          3. Otherwise: bracket-list notation, truncated if >40 chars
        """
        import os.path
        unique_ys = list(dict.fromkeys(y_list))  # preserve order, dedupe
        if len(unique_ys) <= 1:
            return  # nothing to override
        
        prefix = os.path.commonprefix(unique_ys)
        if len(prefix) >= 2:
            ax.set_ylabel(f"{prefix}*")
        else:
            joined = ", ".join(unique_ys)
            if len(joined) > 40:
                joined = ", ".join(unique_ys[:3]) + f", ... (+{len(unique_ys)-3} more)"
            ax.set_ylabel(f"[{joined}]")
    
    def _handle_same_post(self, ax, same, auto_title, selection, y_name,
                          x_name=None, group_by=None, weights=None):
        """
        Post-draw handling for same=True: title append and legend.
        
        Called after the underlying draw_* function returns.
        
        Parameters
        ----------
        ax : Axes
            The axes that was drawn on.
        same : bool
            Whether same=True was used.
        auto_title : bool or str
            Auto-title setting from caller.
        selection : str or None
            Selection string for auto-title.
        y_name, x_name : str
            Expression names for auto-title.
        group_by : str or None
            Group-by column name.
        weights : str or None
            Weights column name.
        """
        from .plots._auto_title import (
            build_auto_title, append_auto_title, apply_auto_title,
            parse_auto_title_parts, resolve_auto_title
        )
        
        # Store axes for next same=True call
        self._last_ax = ax
        
        if not same:
            # Store expression for retroactive labeling when same=True arrives
            self._last_plot_expr = (y_name, x_name)
            return
        
        # Phase 13.13.DF fix: retroactively label first plot when first same=True arrives
        if self._last_plot_expr is not None:
            first_y, first_x = self._last_plot_expr
            first_label = self._auto_label(first_y, first_x)
            # Find unlabeled artists (matplotlib default labels start with '_' or are None)
            # Check lines (profiles), containers (bar charts), patches (histograms), collections (scatter)
            for artist in ax.lines + list(ax.containers) + list(ax.patches) + ax.collections:
                label = artist.get_label()
                if label is None or (isinstance(label, str) and label.startswith('_')):
                    artist.set_label(first_label)
                    break  # Only label the first unlabeled artist
            self._last_plot_expr = None  # Only do this once
        
        # AD-18: Append to title when same=True + auto_title=True
        auto_title = resolve_auto_title(auto_title)
        if auto_title:
            parts = parse_auto_title_parts(auto_title)
            td = build_auto_title(
                x_name or y_name, y_name if x_name else None,
                group_by=group_by, selection=selection,
                weights=weights, parts=parts
            )
            if ax.get_title():
                append_auto_title(ax, td)
            else:
                apply_auto_title(ax, td)
        
        # Show legend when overlaying
        ax.legend(loc=get_style_value("legend.loc", "best"))
    
    # =========================================================================
    # Phase 13.33.DF: Normalized differential profile dispatch (AD-80/81/82)
    # =========================================================================
    # Two-pass orchestrator — fundamentally different from _draw_vector's
    # iterate-and-render: we must complete BOTH curves before computing the
    # differential transform that drives the bottom panel.
    #
    # M1 scope: exactly 2 curves (validated at entry in profile()); no group_by
    # or facet_by composition (M2). Supports overlay+diff and diff_only layouts.

    def _dispatch_normalize_render(
        self,
        y_list,
        x_list,
        *,
        normalize,
        normalize_layout,
        # Per-curve composition inputs (already validated to yield exactly 2)
        selection,
        sample,
        selection_vector,
        weights_vector,
        vector_compose,
        # Profile-specific inputs (forwarded to draw_profile per curve)
        bins,
        x_range,
        error,
        central,
        title,
        xlabel,
        ylabel,
        weights,
        nan_policy,
        # Pass-through bag for less-common kwargs
        **passthrough,
    ):
        """Phase 13.33.DF — two-pass orchestrator for normalize-mode profile().

        Renders signal and reference profiles on the top panel, computes the
        differential transform per v1.1 §3.3, then renders the bottom panel.

        Returns ``(fig, ax_top, stats_dict)``. ``stats_dict['ax_diff']`` is the
        bottom panel Axes (or None for ``normalize_layout='diff_only'`` where
        the diff IS the only panel and the user reads it as the returned
        ``ax_top``-equivalent slot — see AD-81 stats contract).
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from .plots.profile import (
            draw_profile, _compute_normalize_transform, _render_normalize_panel,
            _compute_per_bin_mad_sigma, NORMALIZE_MODES,
        )
        from .style import get_style_value

        # --- 1. Resolve the 2-curve iteration plan -------------------------------
        n_y = len(y_list)
        indices = self._compute_vector_iteration_indices(
            n_y, selection_vector, weights_vector, vector_compose
        )
        if len(indices) != 2:
            # Defensive — entry validation should have caught this.
            raise ValueError(
                f"normalize= requires exactly 2 curves; got {len(indices)}. "
                f"(This is a logic bug — entry validation should have raised first.)"
            )

        # --- 2. Apply the outer selection + sampling (same as scalar path) ------
        df_full = self._apply_selection(self.df, selection)
        df_full = self._apply_sampling(df_full, sample)

        # --- 3. Determine common x_range so both curves share binning -----------
        # If user supplied x_range, both curves use it (already exact). If not,
        # we must derive it BEFORE the per-curve filtering so the two
        # selection-filtered subsets share bin edges. Use the full (post-outer-
        # selection) sample's x range — both per-curve subsets are subsets of it.
        if x_range is None:
            # Use the union across both per-curve subsets; equivalent to using
            # the full df range since each subset's x ⊆ df_full's x.
            x0 = y_list[0]  # placeholder — actual x is x_list[0]
            try:
                x_full = (df_full[x_list[0]]
                          if x_list[0] in df_full.columns
                          else self._eval_column(x_list[0], df=df_full))
            except Exception:
                # Fall through; draw_profile will compute its own per-curve
                # range and bin centers may differ slightly. Logged in CRR §11.
                x_full = None
            if x_full is not None:
                x_arr = pd.Series(x_full).to_numpy()
                x_arr = x_arr[np.isfinite(x_arr)]
                if x_arr.size > 0:
                    x_range = (float(np.min(x_arr)), float(np.max(x_arr)))

        # --- 4. Build figure with gridspec layout -------------------------------
        figsize = passthrough.pop('figsize', None) or get_style_value("figure.figsize", (8, 6))
        if normalize_layout == "overlay+diff":
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(
                2, 1,
                height_ratios=get_style_value("normalize.panel.height_ratio", [3, 1]),
                hspace=get_style_value("normalize.panel.hspace", 0.05),
            )
            ax_top = fig.add_subplot(gs[0])
            ax_diff = fig.add_subplot(gs[1], sharex=ax_top)
            # Hide x-tick-labels on the top panel — they belong to ax_diff
            # (which inherits the formatter via sharex; sidesteps BUG-004).
            plt.setp(ax_top.get_xticklabels(), visible=False)
        else:  # 'diff_only' (validated at entry)
            fig, ax_diff = plt.subplots(figsize=figsize)
            ax_top = None

        # --- 5. Loop the 2 curves: render top panel + capture per-bin stats ----
        # Reset color cycle so signal and reference get curve-cycle colors 0/1.
        self._reset_color_cycle()
        per_curve_stats = []

        for curve_idx, (y_idx, sel_idx, w_idx) in enumerate(indices):
            y_expr = y_list[y_idx]
            x_expr = x_list[y_idx]

            # Resolve per-curve selection (compose outer selection ∧ vector entry).
            curve_sel = None
            if sel_idx is not None and selection_vector is not None:
                curve_sel = selection_vector[sel_idx]

            # Resolve per-curve weights expression.
            curve_weights = weights
            if w_idx is not None and weights_vector is not None:
                curve_weights = weights_vector[w_idx]

            # Filter for this curve.
            curve_df = df_full
            if curve_sel is not None:
                curve_df = self._apply_selection(curve_df, curve_sel)

            # Per-curve label: AD-80 — signal first, reference second.
            role = "signal" if curve_idx == 0 else "reference"
            curve_label = curve_sel if curve_sel is not None else f"{y_expr} ({role})"

            # Render on top panel (skipped for diff_only layout).
            target_ax = ax_top  # may be None
            f_local, ax_returned, sd = draw_profile(
                curve_df, x_expr, y_expr,
                ax=target_ax,
                bins=bins, x_range=x_range, error=error,
                title=None,  # title applied to fig at end
                xlabel=xlabel, ylabel=ylabel,
                label=curve_label,
                return_data=True,  # M1 §10.2 directive #5: force capture
                central=central,
                weights=curve_weights,
                nan_policy=nan_policy,
                _suppress_legend=True,  # legend drawn once at fig level
                _suppress_title=True,
                _suppress_layout=True,
            )

            # If target_ax was None (diff_only), close the figure draw_profile
            # opened so it doesn't leak. Per-bin stats are still in sd.
            if target_ax is None and f_local is not None:
                plt.close(f_local)

            # Extract per-bin arrays from profile_data DataFrame.
            df_bin = sd.get('profile_data')
            if df_bin is None:
                raise RuntimeError(
                    "_dispatch_normalize_render: profile_data missing despite "
                    "return_data=True — implementation bug in draw_profile?"
                )

            # For central='median', y_mean column carries the medians (existing
            # profile.py contract); std/sem are mean-based and would be wrong
            # for the normalize error formula. Compute MAD-sigma separately.
            if central == 'median':
                # Need raw arrays to compute MAD per bin. Pull from curve_df.
                x_raw = (curve_df[x_expr].to_numpy()
                         if x_expr in curve_df.columns
                         else self._eval_column(x_expr, df=curve_df).to_numpy())
                y_raw = (curve_df[y_expr].to_numpy()
                         if y_expr in curve_df.columns
                         else self._eval_column(y_expr, df=curve_df).to_numpy())
                # Drop NaN/inf jointly (mirrors draw_profile's sanitize step).
                finite_mask = np.isfinite(x_raw) & np.isfinite(y_raw)
                x_raw = x_raw[finite_mask]
                y_raw = y_raw[finite_mask]
                mad_sig = _compute_per_bin_mad_sigma(
                    x_raw, y_raw, bins=len(df_bin), x_range=x_range
                )
                sigma_arr = mad_sig
            else:
                sigma_arr = df_bin['y_std'].to_numpy()

            per_curve_stats.append({
                'bin_centers': df_bin['x_center'].to_numpy(),
                'central':     df_bin['y_mean'].to_numpy(),
                'sigma':       sigma_arr,
                'counts':      df_bin['count'].to_numpy(),
            })

        # --- 6. Compute the normalize transform --------------------------------
        values, errors, mask_undef = _compute_normalize_transform(
            per_curve_stats[0], per_curve_stats[1],
            mode=normalize, central=(central or 'mean'),
        )

        # --- 7. Render bottom panel --------------------------------------------
        _render_normalize_panel(
            ax_diff,
            bin_centers=per_curve_stats[0]['bin_centers'],
            values=values, errors=errors,
            mode=normalize,
            label=None,
        )

        # Bottom panel labelling: y-axis name derived from mode.
        diff_ylabel_map = {
            'delta':     'Δ (signal − reference)',
            'ratio':     'signal / reference',
            'log_ratio': 'ln(signal / reference)',
            'pull':      '(s − r) / σ',
        }
        ax_diff.set_ylabel(
            diff_ylabel_map.get(normalize, 'normalize')
            if isinstance(normalize, str) else 'normalize(s, r)'
        )
        # The x-axis label goes on the bottom panel (it's the visible row).
        if xlabel is not None:
            ax_diff.set_xlabel(xlabel)

        # --- 8. Figure-level adornments -----------------------------------------
        # Top panel: legend (signal vs reference), ylabel.
        if ax_top is not None:
            if ylabel is not None:
                ax_top.set_ylabel(ylabel)
            # Show legend on top panel only.
            if ax_top.get_legend_handles_labels()[1]:
                ax_top.legend(loc='best', fontsize=get_style_value('legend.fontsize', 10))
        if title is not None:
            (ax_top if ax_top is not None else ax_diff).set_title(title)

        # Phase 13.34.DF FIX1 BUG-010: figure-level suptitle when auto_title=True
        # for normalize= single-curve path. Same bug class as Phase 13.32 FIX1
        # BUG-002 (faceted path); different dispatcher.
        # Pattern copied from BUG-002 fix in _dispatch_faceted_render
        # (drawer.py:2736-2774). Verifications applied:
        #   - dispatcher local vars: y_list/x_list (not y_expr/x_expr),
        #     passthrough (not plot_kwargs), selection is explicit arg.
        #   - group_by=None (single-curve normalize has no group_by param).
        #   - top=0.92 (matches BUG-002, not 0.88 from initial bug report).
        #   - _auto_title.py:97 return dict keys: 'main'/'sub' (no 'title').
        _auto_title_val = passthrough.get('auto_title', False)
        if _auto_title_val and not title:
            try:
                from .plots._auto_title import (
                    parse_auto_title_parts, build_auto_title, resolve_auto_title
                )
                _at = resolve_auto_title(_auto_title_val)
                if _at:
                    parts = parse_auto_title_parts(_at)
                    _y_str = (y_list if isinstance(y_list, str)
                              else (str(y_list[0]) if y_list else ''))
                    _x_str = (x_list if isinstance(x_list, str)
                              else (str(x_list[0]) if x_list else ''))
                    td = build_auto_title(
                        _x_str, _y_str,
                        group_by=None,  # no group_by in single-curve normalize
                        selection=selection,
                        weights=None,
                        parts=parts,
                    )
                    _main = td.get('main', '')
                    _sub = td.get('sub')
                    fig.suptitle(
                        f"{_main}\n{_sub}" if _sub else _main,
                        fontsize=get_style_value("axes.titlesize", 14),
                    )
                    plt.subplots_adjust(top=_suptitle_top_for_title(_get_suptitle(fig) or None))
            except Exception:
                # Failsafe — same defensive pattern as BUG-002 fix.
                _y_str = (y_list if isinstance(y_list, str)
                          else f"[{','.join(map(str, y_list))}]")
                _x_str = (x_list if isinstance(x_list, str)
                          else str(x_list[0] if x_list else ''))
                fig.suptitle(f"{_y_str} vs {_x_str}",
                             fontsize=get_style_value("axes.titlesize", 14))
                plt.subplots_adjust(top=_suptitle_top_for_title(_get_suptitle(fig) or None))

        # --- 9. Build stats dict (M1 scope per v1.1 §7) ------------------------
        # Drop profile_data from user-facing stats — internal-only.
        # Provide normalize-specific keys per the AD-81 stats contract.
        stats_dict: Dict[str, Any] = {
            'normalize_mode': normalize if isinstance(normalize, str) else 'callable',
            'normalize_layout': normalize_layout,
            'n_masked_bins': int(mask_undef.sum()),
            'n_total_bins': int(len(values)),
            'ax_diff': ax_diff,
            # Per-bin differential data as DataFrame (parallel to profile_data).
            'normalize_data': pd.DataFrame({
                'x_center':     per_curve_stats[0]['bin_centers'],
                'value':        values,
                'error':        errors if errors is not None else np.full_like(values, np.nan),
                'mask_undefined': mask_undef.astype(bool),
                'signal_central':    per_curve_stats[0]['central'],
                'signal_sigma':      per_curve_stats[0]['sigma'],
                'signal_count':      per_curve_stats[0]['counts'],
                'reference_central': per_curve_stats[1]['central'],
                'reference_sigma':   per_curve_stats[1]['sigma'],
                'reference_count':   per_curve_stats[1]['counts'],
            }),
        }

        # M1 closing — return signal panel as the "top" ax (for diff_only this
        # is ax_diff itself, matching the AD-81 contract that the returned
        # ax is the panel the user reads as primary).
        return fig, (ax_top if ax_top is not None else ax_diff), stats_dict

    # =========================================================================
    # Phase 13.33.DF M2: group_by composition (AD-81)
    # =========================================================================
    # Per-group differential — each group gets its own 2-curve render and its
    # own differential curve. All groups share one ax_top + one ax_diff, with
    # colors distinguishing groups and linestyles distinguishing signal/ref
    # within group.
    #
    # M2 design choice: separate method instead of refactoring the M1 path.
    # The two paths share ~40 lines of inner curve-loop logic — this is
    # accepted duplication for M2 in exchange for not perturbing the
    # panel-approved M1 implementation. A unifying refactor is a candidate
    # for a later structural fix-up phase (flagged in CRR §11).

    def _dispatch_normalize_grouped_render(
        self,
        y_list,
        x_list,
        *,
        normalize,
        normalize_layout,
        group_by,
        selection,
        sample,
        selection_vector,
        weights_vector,
        vector_compose,
        bins,
        x_range,
        error,
        central,
        title,
        xlabel,
        ylabel,
        weights,
        nan_policy,
        **passthrough,
    ):
        """Phase 13.33.DF M2 — group_by + normalize composition."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        from .plots.profile import (
            draw_profile, _compute_normalize_transform,
            _render_normalize_panel, _compute_per_bin_mad_sigma,
        )
        from .style import get_style_value

        # --- 1. Resolve 2-curve plan (same as M1) ------------------------------
        n_y = len(y_list)
        indices = self._compute_vector_iteration_indices(
            n_y, selection_vector, weights_vector, vector_compose
        )
        if len(indices) != 2:
            raise ValueError(
                f"normalize= requires exactly 2 curves; got {len(indices)}."
            )

        # --- 2. Outer selection + sampling -------------------------------------
        df_outer = self._apply_selection(self.df, selection)
        df_outer = self._apply_sampling(df_outer, sample)

        # --- 3. Resolve x_range from full sample (shared across groups+curves)-
        if x_range is None:
            try:
                x_full = (df_outer[x_list[0]]
                          if x_list[0] in df_outer.columns
                          else self._eval_column(x_list[0], df=df_outer))
                x_arr = pd.Series(x_full).to_numpy()
                x_arr = x_arr[np.isfinite(x_arr)]
                if x_arr.size > 0:
                    x_range = (float(np.min(x_arr)), float(np.max(x_arr)))
            except Exception:
                pass  # fall through; per-curve auto-detect

        # --- 4. Determine groups (preserved order) -----------------------------
        if group_by not in df_outer.columns:
            raise ValueError(
                f"group_by={group_by!r} not in DataFrame columns "
                f"(got {list(df_outer.columns)[:10]}...)"
            )
        # P2 perf fix (Sonnet53_R2): pd.unique() is O(N) in C and preserves
        # first-appearance order — semantically equivalent to the previous
        # Python-level seen-list loop, ~100x faster on ITS-scale DataFrames
        # (4M rows × K groups → 12M Python comparisons -> single C pass).
        group_values = list(pd.unique(df_outer[group_by]))
        if len(group_values) == 0:
            raise ValueError(
                f"group_by={group_by!r}: no groups found in DataFrame"
            )

        # --- 5. Build figure ---------------------------------------------------
        figsize = passthrough.pop('figsize', None) or get_style_value("figure.figsize", (8, 6))
        if normalize_layout == "overlay+diff":
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(
                2, 1,
                height_ratios=get_style_value("normalize.panel.height_ratio", [3, 1]),
                hspace=get_style_value("normalize.panel.hspace", 0.05),
            )
            ax_top = fig.add_subplot(gs[0])
            ax_diff = fig.add_subplot(gs[1], sharex=ax_top)
            plt.setp(ax_top.get_xticklabels(), visible=False)
        else:  # diff_only
            fig, ax_diff = plt.subplots(figsize=figsize)
            ax_top = None

        # --- 6. Resolve color cycle for groups ---------------------------------
        # One color per group; signal/ref distinguished by linestyle within group.
        import itertools
        cycle = plt.rcParams['axes.prop_cycle'].by_key().get(
            'color', ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        )
        group_colors = list(itertools.islice(itertools.cycle(cycle), len(group_values)))

        per_group_stats = {}  # group_value → stats_dict

        # --- 7. Loop groups ----------------------------------------------------
        for g_idx, g_val in enumerate(group_values):
            g_color = group_colors[g_idx]
            df_group = df_outer[df_outer[group_by] == g_val]

            # Inner 2-curve loop (mirrors M1 _dispatch_normalize_render step 5)
            per_curve_stats = []
            for curve_idx, (y_idx, sel_idx, w_idx) in enumerate(indices):
                y_expr = y_list[y_idx]
                x_expr = x_list[y_idx]

                curve_sel = None
                if sel_idx is not None and selection_vector is not None:
                    curve_sel = selection_vector[sel_idx]
                curve_weights = weights
                if w_idx is not None and weights_vector is not None:
                    curve_weights = weights_vector[w_idx]

                curve_df = df_group
                if curve_sel is not None:
                    curve_df = self._apply_selection(curve_df, curve_sel)

                # Per-curve label: "{group_value} signal/ref"
                role = "signal" if curve_idx == 0 else "ref"
                curve_label = f"{g_val} {role}"
                # Signal solid, reference dashed; group color shared.
                curve_linestyle = "-" if curve_idx == 0 else "--"

                target_ax = ax_top
                # Use ax.errorbar directly for fine color/linestyle control
                # rather than draw_profile, since draw_profile is more rigid
                # about color cycle. We still call draw_profile via ax= to
                # capture per-bin stats, then re-style the rendered artists.
                f_local, ax_ret, sd = draw_profile(
                    curve_df, x_expr, y_expr,
                    ax=target_ax,
                    bins=bins, x_range=x_range, error=error,
                    title=None,
                    xlabel=None, ylabel=None,
                    label=curve_label,
                    return_data=True,
                    central=central,
                    weights=curve_weights,
                    nan_policy=nan_policy,
                    color=g_color,
                    linestyle=curve_linestyle,
                    _suppress_legend=True,
                    _suppress_title=True,
                    _suppress_layout=True,
                )
                if target_ax is None and f_local is not None:
                    plt.close(f_local)

                df_bin = sd.get('profile_data')
                if df_bin is None:
                    raise RuntimeError(
                        "_dispatch_normalize_grouped_render: profile_data "
                        "missing despite return_data=True"
                    )

                if central == 'median':
                    x_raw = (curve_df[x_expr].to_numpy()
                             if x_expr in curve_df.columns
                             else self._eval_column(x_expr, df=curve_df).to_numpy())
                    y_raw = (curve_df[y_expr].to_numpy()
                             if y_expr in curve_df.columns
                             else self._eval_column(y_expr, df=curve_df).to_numpy())
                    finite_mask = np.isfinite(x_raw) & np.isfinite(y_raw)
                    x_raw = x_raw[finite_mask]
                    y_raw = y_raw[finite_mask]
                    sigma_arr = _compute_per_bin_mad_sigma(
                        x_raw, y_raw, bins=len(df_bin), x_range=x_range
                    )
                else:
                    sigma_arr = df_bin['y_std'].to_numpy()

                per_curve_stats.append({
                    'bin_centers': df_bin['x_center'].to_numpy(),
                    'central':     df_bin['y_mean'].to_numpy(),
                    'sigma':       sigma_arr,
                    'counts':      df_bin['count'].to_numpy(),
                })

            # Compute transform for this group
            values, errors, mask_undef = _compute_normalize_transform(
                per_curve_stats[0], per_curve_stats[1],
                mode=normalize, central=(central or 'mean'),
            )

            # Render this group's diff curve on shared ax_diff with group color
            ax_diff.errorbar(
                per_curve_stats[0]['bin_centers'], values,
                yerr=errors if errors is not None else None,
                fmt='o', markersize=get_style_value("profile.markersize", 6),
                capsize=get_style_value("profile.capsize", 3),
                color=g_color, label=str(g_val),
                zorder=3,
            )

            per_group_stats[str(g_val)] = {
                'values': values,
                'errors': errors if errors is not None else np.full_like(values, np.nan),
                'mask_undefined': mask_undef,
                'n_masked_bins': int(mask_undef.sum()),
                'bin_centers': per_curve_stats[0]['bin_centers'],
                'signal_central':    per_curve_stats[0]['central'],
                'signal_sigma':      per_curve_stats[0]['sigma'],
                'signal_count':      per_curve_stats[0]['counts'],
                'reference_central': per_curve_stats[1]['central'],
                'reference_sigma':   per_curve_stats[1]['sigma'],
                'reference_count':   per_curve_stats[1]['counts'],
            }

        # --- 8. Render bands + reference line (ONCE across all groups) ---------
        # Pull mode: bands. All modes: reference line.
        if normalize == "pull":
            a1 = get_style_value("normalize.pull.band_1sigma_alpha", 0.15)
            a2 = get_style_value("normalize.pull.band_2sigma_alpha", 0.08)
            ax_diff.axhspan(-1.0,  1.0, alpha=a1, color="gray", zorder=0)
            ax_diff.axhspan(-2.0, -1.0, alpha=a2, color="gray", zorder=0)
            ax_diff.axhspan( 1.0,  2.0, alpha=a2, color="gray", zorder=0)
        if get_style_value("normalize.panel.reference_line", True):
            ref_y = 1.0 if normalize == "ratio" else 0.0
            ax_diff.axhline(
                ref_y,
                color=get_style_value("normalize.panel.ref_line_color", "gray"),
                linestyle=get_style_value("normalize.panel.ref_line_style", "--"),
                linewidth=1.0, zorder=1,
            )

        # --- 9. Bottom panel labelling ----------------------------------------
        diff_ylabel_map = {
            'delta':     'Δ (signal − reference)',
            'ratio':     'signal / reference',
            'log_ratio': 'ln(signal / reference)',
            'pull':      '(s − r) / σ',
        }
        ax_diff.set_ylabel(
            diff_ylabel_map.get(normalize, 'normalize')
            if isinstance(normalize, str) else 'normalize(s, r)'
        )
        if xlabel is not None:
            ax_diff.set_xlabel(xlabel)

        # --- 10. Figure-level adornments --------------------------------------
        if ax_top is not None:
            if ylabel is not None:
                ax_top.set_ylabel(ylabel)
            if ax_top.get_legend_handles_labels()[1]:
                ax_top.legend(loc='best', fontsize=get_style_value('legend.fontsize', 10))
        # Bottom panel: small legend for the differential per group
        if len(group_values) > 1:
            ax_diff.legend(loc='best', fontsize=get_style_value('legend.fontsize', 9),
                           title=group_by)
        if title is not None:
            (ax_top if ax_top is not None else ax_diff).set_title(title)

        # Phase 13.34.DF FIX1 BUG-010: figure-level suptitle for normalize+group_by.
        # Same pattern as fix in _dispatch_normalize_render (single-curve) and
        # in _dispatch_faceted_render (BUG-002). Difference: group_by IS a
        # parameter of this dispatcher, so pass it to build_auto_title.
        _auto_title_val = passthrough.get('auto_title', False)
        if _auto_title_val and not title:
            try:
                from .plots._auto_title import (
                    parse_auto_title_parts, build_auto_title, resolve_auto_title
                )
                _at = resolve_auto_title(_auto_title_val)
                if _at:
                    parts = parse_auto_title_parts(_at)
                    _y_str = (y_list if isinstance(y_list, str)
                              else (str(y_list[0]) if y_list else ''))
                    _x_str = (x_list if isinstance(x_list, str)
                              else (str(x_list[0]) if x_list else ''))
                    td = build_auto_title(
                        _x_str, _y_str,
                        group_by=group_by,
                        selection=selection,
                        weights=None,
                        parts=parts,
                    )
                    _main = td.get('main', '')
                    _sub = td.get('sub')
                    fig.suptitle(
                        f"{_main}\n{_sub}" if _sub else _main,
                        fontsize=get_style_value("axes.titlesize", 14),
                    )
                    plt.subplots_adjust(top=_suptitle_top_for_title(_get_suptitle(fig) or None))
            except Exception:
                _y_str = (y_list if isinstance(y_list, str)
                          else f"[{','.join(map(str, y_list))}]")
                _x_str = (x_list if isinstance(x_list, str)
                          else str(x_list[0] if x_list else ''))
                fig.suptitle(f"{_y_str} vs {_x_str}",
                             fontsize=get_style_value("axes.titlesize", 14))
                plt.subplots_adjust(top=_suptitle_top_for_title(_get_suptitle(fig) or None))

        # --- 11. Build stats dict (M2 grouped contract) -----------------------
        stats_dict: Dict[str, Any] = {
            'normalize_mode': normalize if isinstance(normalize, str) else 'callable',
            'normalize_layout': normalize_layout,
            'group_by': group_by,
            'n_groups': len(group_values),
            'n_masked_bins': int(sum(s['n_masked_bins'] for s in per_group_stats.values())),
            'ax_diff': ax_diff,
            'normalize_data_grouped': per_group_stats,  # dict by group value
        }
        return fig, (ax_top if ax_top is not None else ax_diff), stats_dict

    # =========================================================================
    # Phase 13.33.DF M2: facet_by composition — K×2 grid (AD-81)
    # =========================================================================
    # Each facet gets its own (top, diff) panel pair. Facets share x-axis
    # within their column; the diff panels share y-axis across columns for
    # easier cross-facet comparison.

    def _dispatch_normalize_faceted_render(
        self,
        y_list,
        x_list,
        *,
        normalize,
        normalize_layout,
        facet_by,
        facet_by_bins,
        facet_by_quantiles,
        selection,
        sample,
        selection_vector,
        weights_vector,
        vector_compose,
        bins,
        x_range,
        error,
        central,
        title,
        xlabel,
        ylabel,
        weights,
        nan_policy,
        ncols=None,
        **passthrough,
    ):
        """Phase 13.33.DF M2 — facet_by + normalize composition (K×2 grid)."""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
        from .plots.profile import (
            draw_profile, _compute_normalize_transform,
            _render_normalize_panel, _compute_per_bin_mad_sigma,
        )
        from .style import get_style_value

        # --- 1. Resolve 2-curve plan ------------------------------------------
        n_y = len(y_list)
        indices = self._compute_vector_iteration_indices(
            n_y, selection_vector, weights_vector, vector_compose
        )
        if len(indices) != 2:
            raise ValueError(
                f"normalize= requires exactly 2 curves; got {len(indices)}."
            )

        # --- 2. Outer selection + sampling ------------------------------------
        df_outer = self._apply_selection(self.df, selection)
        df_outer = self._apply_sampling(df_outer, sample)

        # --- 3. Resolve x_range -----------------------------------------------
        if x_range is None:
            try:
                x_full = (df_outer[x_list[0]]
                          if x_list[0] in df_outer.columns
                          else self._eval_column(x_list[0], df=df_outer))
                x_arr = pd.Series(x_full).to_numpy()
                x_arr = x_arr[np.isfinite(x_arr)]
                if x_arr.size > 0:
                    x_range = (float(np.min(x_arr)), float(np.max(x_arr)))
            except Exception:
                pass

        # --- 4. Resolve facet bin edges + labels via the existing helper -----
        if facet_by not in df_outer.columns:
            raise ValueError(
                f"facet_by={facet_by!r} not in DataFrame columns"
            )
        # For M2 simplicity, use a 'column' facet — facet_by is a column name
        # whose unique values define the K facets. (Other facet_by modes —
        # 'group_by' / 'vector' / 'quantiles' — composing with normalize are
        # NOT supported in M2 v1.0 and raise at the public entry. v1.1 §3.7
        # specifies these as Phase 13.33.DF + future-phase scope.)
        if facet_by_bins is not None or facet_by_quantiles is not None:
            raise NotImplementedError(
                "facet_by_bins / facet_by_quantiles composing with normalize= "
                "is not yet supported in Phase 13.33.DF v1.0 (M2). "
                "Workaround: use a categorical facet_by column directly "
                "(pre-bin the facet variable into a discrete column on the "
                "DataFrame, then pass facet_by='<that column>' without "
                "facet_by_bins / facet_by_quantiles). The auto-binning "
                "composition is reserved for a future fix-up phase."
            )
        # P2 perf fix (Sonnet53_R2): pd.unique() — see _dispatch_normalize_grouped_render
        # for rationale. Same O(N) C-level path, same first-appearance order.
        facet_values = list(pd.unique(df_outer[facet_by]))
        K = len(facet_values)
        if K == 0:
            raise ValueError(f"facet_by={facet_by!r}: no facet values found")

        # --- 5. Build figure with K columns, 2 rows (top + diff) --------------
        # Figsize scales with K.
        default_fig = get_style_value("figure.figsize", (8, 6))
        figsize = (default_fig[0] * K, default_fig[1])
        fig = plt.figure(figsize=figsize)
        if normalize_layout == "diff_only":
            gs = GridSpec(1, K, hspace=0.05, wspace=0.2)
            ax_tops = [None] * K
            # P2 cleanup (Sonnet52_R1, Sonnet53_R2): sharey is applied
            # explicitly in the loop below across all K diff panels (not
            # just the first). The previous ternary was dead code (both
            # branches yielded None).
            ax_diffs = [fig.add_subplot(gs[0, i]) for i in range(K)]
        else:  # overlay+diff
            gs = GridSpec(
                2, K,
                height_ratios=get_style_value("normalize.panel.height_ratio", [3, 1]),
                hspace=get_style_value("normalize.panel.hspace", 0.05),
                wspace=0.2,
            )
            ax_tops, ax_diffs = [], []
            for i in range(K):
                ax_top_i = fig.add_subplot(gs[0, i])
                ax_diff_i = fig.add_subplot(gs[1, i], sharex=ax_top_i)
                plt.setp(ax_top_i.get_xticklabels(), visible=False)
                ax_tops.append(ax_top_i)
                ax_diffs.append(ax_diff_i)

        # Share y-axis across all diff panels for easier cross-facet comparison.
        for j in range(1, K):
            ax_diffs[j].sharey(ax_diffs[0])

        # --- 6. Loop facets ----------------------------------------------------
        per_facet_stats = {}
        for f_idx, f_val in enumerate(facet_values):
            ax_top_i = ax_tops[f_idx]
            ax_diff_i = ax_diffs[f_idx]
            df_facet = df_outer[df_outer[facet_by] == f_val]

            # Inner 2-curve loop (M1 pattern, scoped to this facet)
            per_curve_stats = []
            for curve_idx, (y_idx, sel_idx, w_idx) in enumerate(indices):
                y_expr = y_list[y_idx]
                x_expr = x_list[y_idx]

                curve_sel = None
                if sel_idx is not None and selection_vector is not None:
                    curve_sel = selection_vector[sel_idx]
                curve_weights = weights
                if w_idx is not None and weights_vector is not None:
                    curve_weights = weights_vector[w_idx]

                curve_df = df_facet
                if curve_sel is not None:
                    curve_df = self._apply_selection(curve_df, curve_sel)

                role = "signal" if curve_idx == 0 else "reference"
                curve_label = curve_sel if curve_sel is not None else f"{y_expr} ({role})"

                target_ax = ax_top_i
                f_local, _, sd = draw_profile(
                    curve_df, x_expr, y_expr,
                    ax=target_ax,
                    bins=bins, x_range=x_range, error=error,
                    title=None,
                    xlabel=None, ylabel=None,
                    label=curve_label,
                    return_data=True,
                    central=central,
                    weights=curve_weights,
                    nan_policy=nan_policy,
                    _suppress_legend=True,
                    _suppress_title=True,
                    _suppress_layout=True,
                )
                if target_ax is None and f_local is not None:
                    plt.close(f_local)

                df_bin = sd.get('profile_data')
                if df_bin is None:
                    raise RuntimeError(
                        "_dispatch_normalize_faceted_render: profile_data "
                        "missing despite return_data=True"
                    )

                if central == 'median':
                    x_raw = (curve_df[x_expr].to_numpy()
                             if x_expr in curve_df.columns
                             else self._eval_column(x_expr, df=curve_df).to_numpy())
                    y_raw = (curve_df[y_expr].to_numpy()
                             if y_expr in curve_df.columns
                             else self._eval_column(y_expr, df=curve_df).to_numpy())
                    finite_mask = np.isfinite(x_raw) & np.isfinite(y_raw)
                    sigma_arr = _compute_per_bin_mad_sigma(
                        x_raw[finite_mask], y_raw[finite_mask],
                        bins=len(df_bin), x_range=x_range
                    )
                else:
                    sigma_arr = df_bin['y_std'].to_numpy()

                per_curve_stats.append({
                    'bin_centers': df_bin['x_center'].to_numpy(),
                    'central':     df_bin['y_mean'].to_numpy(),
                    'sigma':       sigma_arr,
                    'counts':      df_bin['count'].to_numpy(),
                })

            # Compute transform for this facet
            values, errors, mask_undef = _compute_normalize_transform(
                per_curve_stats[0], per_curve_stats[1],
                mode=normalize, central=(central or 'mean'),
            )

            # Render facet's diff panel
            _render_normalize_panel(
                ax_diff_i,
                bin_centers=per_curve_stats[0]['bin_centers'],
                values=values, errors=errors,
                mode=normalize, label=None,
            )

            # Per-facet title (small) — only on top panel
            if ax_top_i is not None:
                ax_top_i.set_title(f"{facet_by}={f_val}",
                                   fontsize=get_style_value('axes.titlesize', 10))
            # x-label on diff panel
            if xlabel is not None:
                ax_diff_i.set_xlabel(xlabel)

            per_facet_stats[str(f_val)] = {
                'values': values,
                'errors': errors if errors is not None else np.full_like(values, np.nan),
                'mask_undefined': mask_undef,
                'bin_centers': per_curve_stats[0]['bin_centers'],
            }

        # --- 7. Y-labels: only on leftmost column to avoid clutter ------------
        diff_ylabel_map = {
            'delta':     'Δ',
            'ratio':     'ratio',
            'log_ratio': 'ln-ratio',
            'pull':      'pull',
        }
        ax_diffs[0].set_ylabel(
            diff_ylabel_map.get(normalize, 'normalize')
            if isinstance(normalize, str) else 'normalize(s, r)'
        )
        if ylabel is not None and ax_tops[0] is not None:
            ax_tops[0].set_ylabel(ylabel)

        # Figure-level title spans all facets
        if title is not None:
            fig.suptitle(title)

        # Phase 13.34.DF FIX1 BUG-010: figure-level suptitle for normalize+facet_by.
        # Same pattern as fixes in _dispatch_normalize_render and
        # _dispatch_normalize_grouped_render. Note: this dispatcher ALREADY
        # had fig.suptitle(title) for explicit title (line above) — auto_title
        # block runs only when title is None. group_by=None: each facet is its
        # own panel, facet identity is in subplot titles via subplot_titles,
        # not folded into the figure-level suptitle.
        _auto_title_val = passthrough.get('auto_title', False)
        if _auto_title_val and title is None:
            try:
                from .plots._auto_title import (
                    parse_auto_title_parts, build_auto_title, resolve_auto_title
                )
                _at = resolve_auto_title(_auto_title_val)
                if _at:
                    parts = parse_auto_title_parts(_at)
                    _y_str = (y_list if isinstance(y_list, str)
                              else (str(y_list[0]) if y_list else ''))
                    _x_str = (x_list if isinstance(x_list, str)
                              else (str(x_list[0]) if x_list else ''))
                    td = build_auto_title(
                        _x_str, _y_str,
                        group_by=None,  # facet_by is in subplot titles, not figure title
                        selection=selection,
                        weights=None,
                        parts=parts,
                    )
                    _main = td.get('main', '')
                    _sub = td.get('sub')
                    fig.suptitle(
                        f"{_main}\n{_sub}" if _sub else _main,
                        fontsize=get_style_value("axes.titlesize", 14),
                    )
                    plt.subplots_adjust(top=_suptitle_top_for_title(_get_suptitle(fig) or None))
            except Exception:
                _y_str = (y_list if isinstance(y_list, str)
                          else f"[{','.join(map(str, y_list))}]")
                _x_str = (x_list if isinstance(x_list, str)
                          else str(x_list[0] if x_list else ''))
                fig.suptitle(f"{_y_str} vs {_x_str}",
                             fontsize=get_style_value("axes.titlesize", 14))
                plt.subplots_adjust(top=_suptitle_top_for_title(_get_suptitle(fig) or None))

        # --- 8. Stats dict ----------------------------------------------------
        stats_dict: Dict[str, Any] = {
            'normalize_mode': normalize if isinstance(normalize, str) else 'callable',
            'normalize_layout': normalize_layout,
            'facet_by': facet_by,
            'n_facets': K,
            'ax_diffs': ax_diffs,  # list, one per facet
            'normalize_data_faceted': per_facet_stats,
        }
        # M2 contract: when faceted, the returned ax_top is a LIST (one per
        # facet) instead of a single axes. Caller can iterate.
        returned_top = ax_tops if normalize_layout == "overlay+diff" else ax_diffs
        return fig, returned_top, stats_dict

    # =========================================================================
    # Phase 13.27.DF (Phase D): Faceted rendering dispatch
    # =========================================================================
    
    # Valid facet_by values for Phase 13.27 Commit 1.
    # Commit 2 will add 'selection_delta' and 'weights_delta'.
    _VALID_FACET_BY_VALUES_COMMIT1 = ('group_by', 'vector', 'quantiles')

    def _dispatch_faceted_render(
        self,
        df: pd.DataFrame,
        x_expr: str,
        y_expr: Union[str, list],
        facet_by,                            # Phase 13.41.DF v1.6: Union[str, List[str]]
        plot_kind: str,
        ncols: Optional[int] = None,
        sharex: bool = True,
        sharey: bool = True,
        title: Optional[str] = None,
        group_by: Optional[str] = None,
        top_k: Optional[int] = None,
        quantiles: Optional[list] = None,
        quantile_mode: str = "auto",
        # Phase 13.41.DF v1.6: N-D faceting (list-form facet_by) params
        share_x: str = 'all',                # 'all' | 'row' | 'col' | 'none'
        share_y: str = 'all',                # same
        share_across_figures: bool = True,   # 3D only
        **plot_kwargs
    ) -> Tuple[plt.Figure, np.ndarray, Dict[str, Any]]:
        """
        Coordinate faceted rendering: validate facet_by, build subplot grid,
        per-subplot recursion into the appropriate plot module.

        Phase 13.27.DF (Phase D), §5.4 of v1.1 proposal.

        Parameters
        ----------
        df : DataFrame
            Pre-filtered, pre-sampled DataFrame (selection/sample applied
            by caller).
        x_expr, y_expr : str or list
            Column expressions. y_expr may be list (vector y).
        facet_by : str
            One of {'group_by', 'vector', 'quantiles'} for Commit 1.
            Commit 2 extends to {'selection_delta', 'weights_delta'}.
        plot_kind : str
            Plot module to dispatch ('profile' for Commit 1; 'hist'/'scatter'
            in Commit 2).
        ncols, sharex, sharey, title : standard facet kwargs
        group_by, top_k, quantiles, quantile_mode : forwarded to per-subplot calls
        **plot_kwargs : forwarded to per-subplot draw_* call

        Returns
        -------
        (fig, axes_flat, combined_stats_dict)
            stats_dict format:
                {'n_groups': int, 'groups': list, 'per_group': {...},
                 'n_total': int, 'faceted': True, 'facet_by': str}

        Raises
        ------
        ValueError
            - On invalid facet_by name
            - On capacity overflow (when channels.overflow='error')
            - On 'group_by' facet without group_by parameter set
            - On 'vector' facet without list-valued y_expr
            - On 'quantiles' facet without quantile_mode='discrete'
        """
        # =====================================================================
        # Phase 13.41.DF v1.6 — N-D faceting entry point
        # =====================================================================
        # Convention LOCKED: facet_by[0]=row, [1]=col, [2]=figID (numpy shape)
        # 
        # List-form facet_by routing:
        #   length 1 (or str) → unwrap to scalar, fall through to 1D path
        #                       (Phase 13.32 backward compat, byte-identical)
        #   length 2          → 2D row × col grid (NEW)
        #   length 3          → 3D via figID multi-figure (NEW)
        #   length 4+         → NotImplementedError via _normalize_facet_args
        # =====================================================================
        if isinstance(facet_by, list):
            _facet_by_bins_raw = plot_kwargs.get('facet_by_bins')
            _facet_by_quantiles_raw = plot_kwargs.get('facet_by_quantiles')
            facet_list, bins_list, quantiles_list = _normalize_facet_args(
                facet_by, _facet_by_bins_raw, _facet_by_quantiles_raw)
            
            # Validate share_x / share_y
            _validate_share_axis_value(share_x, 'share_x')
            _validate_share_axis_value(share_y, 'share_y')
            
            # Phase 13.41.DF v1.6: pop outer-only kwargs that conflict with
            # inner call signatures or N-D dispatch semantics. User-supplied
            # range/x_range stay in plot_kwargs and are reconciled inside
            # _dispatch_inner_per_cell via user_range / user_x_range pop.
            plot_kwargs.pop('ncols', None)
            plot_kwargs.pop('sharex', None)
            plot_kwargs.pop('sharey', None)
            plot_kwargs.pop('facet', None)
            # auto_title: scatter doesn't accept it (drawer.py:1198), so always
            # pop and re-add per plot_kind inside _dispatch_inner_per_cell.
            # Phase 13.41 FIX1 (Sonnet54 P2): capture user's choice; default
            # False matches the DFDraw.hist/profile defaults. v1.6 hardcoded
            # False unconditionally — user's auto_title=True was silently dropped.
            _user_auto_title = plot_kwargs.pop('auto_title', False)
            
            n_dims = len(facet_list)
            if n_dims == 1:
                # List of length 1 — unwrap and fall through to 1D path below
                facet_by = facet_list[0]
                # Update plot_kwargs to also unwrap bins/quantiles to scalar/list
                if bins_list[0] is not None:
                    plot_kwargs['facet_by_bins'] = bins_list[0]
                if quantiles_list[0] is not None:
                    plot_kwargs['facet_by_quantiles'] = quantiles_list[0]
                # Restore auto_title for 1D fall-through (1D path expects it in kwargs)
                if _user_auto_title:
                    plot_kwargs['auto_title'] = _user_auto_title
                # Fall through to existing 1D logic
            elif n_dims == 2:
                # Pop bins/quantiles from plot_kwargs (now in lists)
                plot_kwargs.pop('facet_by_bins', None)
                plot_kwargs.pop('facet_by_quantiles', None)
                return self._dispatch_2d_facet(
                    df, x_expr, y_expr, facet_list, bins_list, quantiles_list,
                    plot_kind, share_x=share_x, share_y=share_y,
                    title=title, group_by=group_by, top_k=top_k,
                    quantiles=quantiles, quantile_mode=quantile_mode,
                    _lock_x_range=None, _lock_y_range=None,
                    _user_auto_title=_user_auto_title,
                    **plot_kwargs)
            elif n_dims == 3:
                # 3D: loop over figID dimension, dispatch 2D per figID
                plot_kwargs.pop('facet_by_bins', None)
                plot_kwargs.pop('facet_by_quantiles', None)
                return self._dispatch_3d_facet(
                    df, x_expr, y_expr, facet_list, bins_list, quantiles_list,
                    plot_kind, share_x=share_x, share_y=share_y,
                    share_across_figures=share_across_figures,
                    title=title, group_by=group_by, top_k=top_k,
                    quantiles=quantiles, quantile_mode=quantile_mode,
                    _user_auto_title=_user_auto_title,
                    **plot_kwargs)
        # =====================================================================
        # End Phase 13.41 N-D branch — existing 1D code follows unchanged
        # =====================================================================
        
        # ---- Validate facet_by ---------------------------------------------
        # Phase 13.31.DF v1.0 (AD-78): facet_by is a tagged union — either a
        # channel-name enum value (Phase 13.27 Commit 1 semantics, preserved)
        # OR a DataFrame column name (new direct facet-by-column path).
        #
        # Disambiguation (deterministic, channel-enum and df.columns are
        # disjoint by construction):
        #   1. None              → no faceting (handled by caller, not here)
        #   2. channel-name enum → existing path
        #   3. df column name    → NEW column-name path
        #   4. else              → ValueError citing BOTH interpretations
        #
        # The Phase 13.30 _PROFILE_COLUMN_REFERENCES tuple deliberately does
        # NOT include facet_by (Sonnet P1, AD-78 cross-review): listing it
        # there would cause validate_column_references() to reject every
        # valid channel-name value (e.g. facet_by="group_by") because the
        # string "group_by" is not a column. Validation lives here instead.
        _facet_mode = None  # 'channel' | 'column'
        # Phase 13.32.DF FIX1 BUG-001: save the original facet_by string at
        # function entry, BEFORE any rebinding can happen at line 2524
        # (column-mode binning replaces facet_by with '__dfdraw_facet_bin__').
        # _facet_display_name is used in subplot titles (line 2703), stats
        # dict (line 2720), and the auto_title suptitle block (BUG-002 fix).
        # It must be defined for ALL facet modes (channel + column) since
        # those code paths run unconditionally. Placement at function entry
        # rather than inside the column-mode branch closes the v1.5 scoping
        # gap (Sonet51 + Sonnet52_R1 P1 catch).
        _facet_display_name = facet_by
        if facet_by in self._VALID_FACET_BY_VALUES_COMMIT1:
            _facet_mode = 'channel'
        elif (df is not None
              and isinstance(facet_by, str)
              and facet_by
              and facet_by in df.columns):
            _facet_mode = 'column'
            # Phase 13.38.DF (BUG-017): float facet_by column + no bins +
            # high cardinality → memory hang/OOM. Mirrors BUG-012 (hist,
            # Phase 13.35) and BUG-015 (profile, Phase 13.37). Fires BEFORE
            # the unique() enumeration below so users get a clear error
            # instead of an unhelpful crash. facet_by_bins/_quantiles are
            # passed via **plot_kwargs (not direct named params on this
            # method). Limitation: expression-string facet_by="z/250." is
            # NOT in df.columns → this branch is not entered → guard is
            # skipped (same gap as Phase 13.35/13.37; deferred to df.eval()
            # path phase).
            _facet_col = df[facet_by]
            _fbb = plot_kwargs.get('facet_by_bins', None)
            _fbq = plot_kwargs.get('facet_by_quantiles', None)
            if (_facet_col.dtype.kind == 'f'
                    and _fbb is None
                    and _fbq is None
                    and _facet_col.nunique() > 20):
                raise ValueError(
                    f"facet_by={facet_by!r} is a float column with "
                    f"{_facet_col.nunique()} unique values. "
                    f"Add facet_by_bins=N or facet_by_quantiles=N to bin it. "
                    f"Example: facet_by_bins=9 or facet_by_quantiles=5."
                )
        else:
            # Defer 'selection_delta' / 'weights_delta' to Commit 2
            if facet_by in ('selection_delta', 'weights_delta'):
                raise NotImplementedError(
                    f"facet_by={facet_by!r} is reserved for Phase 13.27 Commit 2 "
                    f"(selection_vector + weights_vector). Currently supported: "
                    f"channel names {self._VALID_FACET_BY_VALUES_COMMIT1} "
                    f"or DataFrame column names."
                )
            # Build a clear error message mentioning BOTH interpretations
            cols = list(df.columns) if df is not None else []
            if len(cols) > 10:
                cols_preview = ", ".join(repr(c) for c in cols[:10]) + f", ... ({len(cols)} total)"
            else:
                cols_preview = ", ".join(repr(c) for c in cols)
            raise ValueError(
                f"facet_by={facet_by!r} is neither a recognized channel name "
                f"nor a DataFrame column. "
                f"Valid channel names: {self._VALID_FACET_BY_VALUES_COMMIT1}. "
                f"Available columns: [{cols_preview}]."
            )

        # ---- Determine groups (subplot keys) -------------------------------
        if facet_by == 'group_by':
            if group_by is None:
                raise ValueError(
                    "facet_by='group_by' requires group_by= parameter to be set"
                )
            # Phase 13.32.DF Sub-fix 1 (v1.2 C1, FIX 1 from v1.0 panel): apply
            # group_by_bins/_quantiles AT DISPATCH LEVEL before group enumeration.
            # Without this, dispatch saw the raw column's unique values (e.g.
            # 60 driftM_bin25 levels) and hit the 16-cap before binning had a
            # chance to collapse them to 5.
            #
            # CRITICAL: also pop group_by/group_by_bins/group_by_quantiles from
            # plot_kwargs to prevent per-subplot draw_profile recursion from
            # re-binning the already-filtered slice (which would create N×N
            # spurious sub-groups inside each subplot — the FIX 1 P1 from
            # Sonet51/Claude48/GPT1 v1.0 review).
            _gby_bins = plot_kwargs.pop('group_by_bins', None)
            _gby_quantiles = plot_kwargs.pop('group_by_quantiles', None)
            _effective_df = df
            _effective_group_col = group_by
            if _gby_bins is not None or _gby_quantiles is not None:
                _effective_df = df.copy()
                if df[group_by].dtype == np.float16:
                    _effective_df[group_by] = _effective_df[group_by].astype(np.float32)
                if _gby_bins is not None:
                    intervals = pd.cut(_effective_df[group_by], bins=_gby_bins)
                else:
                    intervals = pd.qcut(
                        _effective_df[group_by], q=_gby_quantiles, duplicates='drop'
                    )
                # Phase 13.32.DF v1.2 P2 (collision-safe): __dfdraw_*__ sentinel
                # name avoids collision with any user column.
                _effective_df['__dfdraw_group_bin__'] = intervals.map(_format_interval_label)
                _effective_group_col = '__dfdraw_group_bin__'

            # Pop group_by too — facet_by='group_by' means the facet IS the
            # group_by; per-subplot calls must not re-introduce an inner overlay.
            plot_kwargs.pop('group_by', None)
            # Replace df + group_by for the rest of dispatch:
            df = _effective_df
            group_by = _effective_group_col

            groups = list(df[group_by].unique())
            if top_k is not None and len(groups) > top_k:
                counts = df[group_by].value_counts()
                groups = counts.head(top_k).index.tolist()

        elif facet_by == 'vector':
            if not isinstance(y_expr, list):
                raise ValueError(
                    "facet_by='vector' requires a list-valued y expression "
                    f"(got scalar y_expr={y_expr!r})"
                )
            groups = list(y_expr)

        elif facet_by == 'quantiles':
            if quantile_mode != 'discrete':
                raise ValueError(
                    "facet_by='quantiles' requires quantile_mode='discrete' "
                    f"(got quantile_mode={quantile_mode!r})"
                )
            if quantiles is None or len(quantiles) == 0:
                raise ValueError(
                    "facet_by='quantiles' requires quantiles= parameter "
                    "with at least one value"
                )
            groups = list(quantiles)

        elif _facet_mode == 'column':
            # Phase 13.31.DF (AD-78): column-name facet — one subplot per
            # unique value of df[facet_by]. NaN values are dropped (they
            # represent unobserved combinations of the faceting dimension).
            # Sorted ascending for deterministic subplot ordering; user can
            # override behaviour via sort_groups= (existing Phase 13.12 kwarg
            # on profile() — not consumed at this dispatch level, applies
            # inside each subplot's group_by).
            #
            # Phase 13.32.DF Sub-fix 3 (AD-79, v1.2 §3.3): apply facet_by_bins
            # / facet_by_quantiles AT DISPATCH LEVEL when the facet column is
            # a float-typed column. Pop them from plot_kwargs so per-subplot
            # recursion does not re-bin. Validation that they are mutually
            # exclusive + require column-mode facet_by happens at the plot
            # method entry point (_validate_facet_by_binning helper).
            _fby_bins = plot_kwargs.pop('facet_by_bins', None)
            _fby_quantiles = plot_kwargs.pop('facet_by_quantiles', None)
            _effective_df = df
            _effective_facet_col = facet_by
            if _fby_bins is not None or _fby_quantiles is not None:
                _effective_df = df.copy()
                if df[facet_by].dtype == np.float16:
                    _effective_df[facet_by] = _effective_df[facet_by].astype(np.float32)
                if _fby_bins is not None:
                    intervals = pd.cut(_effective_df[facet_by], bins=_fby_bins)
                else:
                    intervals = pd.qcut(
                        _effective_df[facet_by], q=_fby_quantiles, duplicates='drop'
                    )
                _effective_df['__dfdraw_facet_bin__'] = intervals.map(_format_interval_label)
                _effective_facet_col = '__dfdraw_facet_bin__'
            # Replace df + facet_by for the rest of dispatch:
            df = _effective_df
            facet_by = _effective_facet_col

            # Phase 13.32.DF FIX1 BUG-003: numeric sort using _interval_sort_key
            # (already in profile.py). Without this, string labels produced by
            # _format_interval_label sort lexicographically: "12.0-16.0" <
            # "4.0-8.0" because "1" < "4". _interval_sort_key extracts the
            # numeric left boundary for correct ordering. _fby_bins and
            # _fby_quantiles are set at lines 2506-2507 (same scope, in range).
            if _fby_bins is not None or _fby_quantiles is not None:
                groups = sorted(
                    df[facet_by].dropna().unique().tolist(),
                    key=_interval_sort_key
                )
            else:
                try:
                    groups = sorted(df[facet_by].dropna().unique().tolist())
                except TypeError:
                    # Mixed/unsortable dtype — fall back to unsorted order
                    groups = df[facet_by].dropna().unique().tolist()
            if top_k is not None and len(groups) > top_k:
                counts = df[facet_by].value_counts()
                groups = counts.head(top_k).index.tolist()

        else:
            # Unreachable due to validation above
            raise ValueError(f"Unhandled facet_by={facet_by!r}")

        n_groups = len(groups)
        if n_groups == 0:
            raise ValueError(f"No groups found for facet_by={facet_by!r}")

        # ---- Capacity check (channels.cycles.facet_max) --------------------
        facet_max = get_style_value("channels.cycles.facet_max", 16)
        if n_groups > facet_max:
            overflow_mode = get_style_value("channels.overflow", "error")
            msg = (
                f"facet_by={facet_by!r} produces {n_groups} subplots, exceeding "
                f"channels.cycles.facet_max={facet_max}. Reduce cardinality, "
                f"set top_k=, or increase channels.cycles.facet_max."
            )
            if overflow_mode == "warn":
                import warnings
                warnings.warn(msg, UserWarning, stacklevel=2)
                # Truncate to facet_max
                groups = groups[:facet_max]
                n_groups = facet_max
            else:
                raise ValueError(msg)

        # ---- Create subplot grid -------------------------------------------
        if ncols is None:
            ncols = min(3, n_groups)
        nrows = int(np.ceil(n_groups / ncols))
        base_size = get_style_value("figure.figsize", (8, 6))
        figsize = (base_size[0] / 1.5 * ncols, base_size[1] / 1.5 * nrows)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize,
            sharex=sharex, sharey=sharey, squeeze=False
        )
        axes_flat = axes.flatten()
        # Hide unused subplots
        for idx in range(n_groups, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        # ---- Per-subplot recursion -----------------------------------------
        # Dispatch based on plot_kind. Commit 1: profile only.
        # Phase 13.32.DF Sub-fix 3 (AD-79, v1.2 §3.3): full multi-kind dispatch.
        # Phase 13.27 Commit 1 only routed 'profile'; v1.2 scope extends to
        # all plot kinds. Each plot has a different signature, so the
        # per-subplot call must be branched. Deferred imports (function-level)
        # avoid any circular-import risk from drawer ↔ plots.*.
        if plot_kind == 'profile':
            from .plots.profile import draw_profile
            plot_fn = draw_profile
        elif plot_kind == 'hist':
            from .plots.histogram import draw_hist
            plot_fn = draw_hist
        elif plot_kind == 'hist2d':
            from .plots.histogram import draw_hist2d
            plot_fn = draw_hist2d
        elif plot_kind == 'scatter':
            from .plots.scatter import draw_scatter
            plot_fn = draw_scatter
        else:
            raise NotImplementedError(
                f"plot_kind={plot_kind!r} not supported. "
                f"Valid: 'profile', 'hist', 'hist2d', 'scatter'."
            )

        all_stats: Dict[str, Any] = {}
        for ax_i, group_value in zip(axes_flat[:n_groups], groups):
            # Filter / specialize per facet_by mode
            if facet_by == 'group_by':
                subplot_df = df[df[group_by] == group_value]
                subplot_y = y_expr
                subplot_quantiles = quantiles
            elif facet_by == 'vector':
                subplot_df = df
                subplot_y = group_value  # one element of y_expr list
                subplot_quantiles = quantiles
            elif facet_by == 'quantiles':
                subplot_df = df
                subplot_y = y_expr
                subplot_quantiles = [group_value]  # one quantile per subplot
            elif _facet_mode == 'column':
                # Phase 13.31.DF (AD-78): column-name facet — filter via
                # boolean mask, same pattern as 'group_by' channel mode. NB:
                # mask comparison handles numeric AND string dtypes uniformly
                # (no string-quoting concern that would arise if we extended
                # the selection string instead — Sonnet implementation note).
                subplot_df = df[df[facet_by] == group_value]
                subplot_y = y_expr
                subplot_quantiles = quantiles

            # Strip kwargs that are facet-coordinator-only (don't forward)
            # AD-68: ax provided per-subplot; selection already applied at top
            forwarded = dict(plot_kwargs)
            forwarded.pop('ax', None)
            forwarded.pop('selection', None)
            forwarded.pop('save', None)
            # auto_title: per §5.4, suptitle on figure level; subplot title is
            # the facet value. Suppress per-subplot auto_title — but ONLY on
            # plot kinds whose draw_* signature accepts auto_title. draw_scatter
            # does NOT have auto_title; setting it would leak through **kwargs
            # to ax.scatter() and raise AttributeError (Phase 13.32 FIX1 from
            # the column-mode scatter test).
            if plot_kind != 'scatter':
                forwarded['auto_title'] = False

            # Phase 13.42.DF FIX1 (B1/Sonet51): facet_mode sentinel — informs
            # per-cell draw call that it is rendering inside a facet grid, so
            # render_fit_textbox uses fit.text_fontsize_facet instead of
            # fit.text_fontsize_default. v1.0 missed this plumbing: render_fit_textbox
            # signature had facet_mode but all 3 call sites passed nothing, so the
            # facet font branch was permanently unreachable.
            # Only forward to plot kinds that support fit= (hist/profile/scatter);
            # other kinds (hist2d/scatter3d/hexbin/profile2d) would leak the
            # sentinel into matplotlib via **kwargs.
            if plot_kind in ('hist', 'profile', 'scatter'):
                forwarded['_facet_mode'] = True

            # Phase 13.31.DF (AD-78): for column-mode facet, KEEP group_by
            # for inner overlay (orthogonal dimension — that's the whole
            # point of dual-mode facet_by). The 'group_by' channel mode
            # still suppresses group_by because the facet IS the group_by.
            if facet_by == 'group_by':
                _inner_group_by = None
            else:
                _inner_group_by = group_by

            # Phase 13.32.DF Sub-fix 3 (AD-79, v1.2 §3.3): plot-kind-specific
            # call. Each plot's signature differs in positional shape and
            # accepted kwargs; passing profile-only kwargs (quantiles,
            # quantile_mode) to hist/scatter/hist2d would raise TypeError,
            # and hist2d does not accept group_by/top_k. Build call args
            # explicitly per plot_kind.
            try:
                if plot_kind == 'profile':
                    _, _, stats = plot_fn(
                        subplot_df, x_expr, subplot_y,
                        ax=ax_i,
                        quantiles=subplot_quantiles,
                        quantile_mode=quantile_mode,
                        group_by=_inner_group_by,
                        top_k=None,
                        **forwarded
                    )
                elif plot_kind == 'hist':
                    # hist is 1D: takes only x (the variable to histogram).
                    # Calling method passed col_expr as y_expr, so subplot_y
                    # is the variable. hist accepts group_by + top_k.
                    _, _, stats = plot_fn(
                        subplot_df, subplot_y,
                        ax=ax_i,
                        group_by=_inner_group_by,
                        top_k=None,
                        **forwarded
                    )
                elif plot_kind == 'scatter':
                    # scatter accepts (x, y), group_by, top_k. No quantiles.
                    _, _, stats = plot_fn(
                        subplot_df, x_expr, subplot_y,
                        ax=ax_i,
                        group_by=_inner_group_by,
                        top_k=None,
                        **forwarded
                    )
                elif plot_kind == 'hist2d':
                    # hist2d accepts (x, y). Does NOT accept group_by or
                    # top_k — 2D density doesn't overlay groups.
                    _, _, stats = plot_fn(
                        subplot_df, x_expr, subplot_y,
                        ax=ax_i,
                        **forwarded
                    )
            except Exception as e:
                # Re-raise with facet context
                raise type(e)(
                    f"Error in faceted subplot (facet_by={facet_by!r}, "
                    f"group={group_value!r}): {e}"
                ) from e

            # Subplot title from facet value
            # Phase 13.32.DF FIX1 BUG-001: use _facet_display_name (the
            # original facet_by string saved at function entry) rather than
            # the possibly-rebinded facet_by. For column-mode + binning, the
            # rebinded value is the internal '__dfdraw_facet_bin__' name.
            ax_i.set_title(f"{_facet_display_name}={group_value}")
            all_stats[str(group_value)] = stats

        # ---- Figure-level title (suptitle) ---------------------------------
        if title:
            fig.suptitle(title, fontsize=get_style_value("axes.titlesize", 14) + 2)

        # Phase 13.32.DF FIX1 BUG-002: figure-level suptitle when auto_title=True.
        # Per-subplot calls run with auto_title=False (line 2639 — correct,
        # prevents per-subplot title collision). Figure level was missing.
        # build_auto_title signature verified against drawer.py:1329-1333.
        # Return dict keys verified against _auto_title.py:97:
        #     return {"main": main, "sub": sub}   (no 'title' key)
        _auto_title_val = plot_kwargs.get('auto_title', False)
        if _auto_title_val and not title:
            try:
                from .plots._auto_title import (
                    parse_auto_title_parts, build_auto_title, resolve_auto_title
                )
                _at = resolve_auto_title(_auto_title_val)
                if _at:
                    parts = parse_auto_title_parts(_at)
                    td = build_auto_title(
                        x_expr or '',
                        str(y_expr) if isinstance(y_expr, str) else str(y_expr[0]),
                        group_by=group_by if _facet_display_name != 'group_by' else None,
                        selection=plot_kwargs.get('selection'),
                        weights=None,
                        parts=parts,
                    )
                    _main = td.get('main', '')
                    _sub = td.get('sub')
                    fig.suptitle(
                        f"{_main}\n{_sub}" if _sub else _main,
                        fontsize=get_style_value("axes.titlesize", 14),
                    )
                    plt.subplots_adjust(top=_suptitle_top_for_title(_get_suptitle(fig) or None))
            except Exception:
                # Failsafe: minimal title — prevents auto_title import/build
                # errors from crashing the plot. Same defensive pattern as
                # _draw_vector failsafe (line ~1230).
                y_str = (y_expr if isinstance(y_expr, str)
                         else f"[{','.join(y_expr)}]")
                fig.suptitle(f"{y_str} vs {x_expr}",
                             fontsize=get_style_value("axes.titlesize", 14))
                plt.subplots_adjust(top=_suptitle_top_for_title(_get_suptitle(fig) or None))

        plt.tight_layout()
        if title:
            plt.subplots_adjust(top=_suptitle_top_for_title(_get_suptitle(fig) or None))

        # ---- Combined stats ------------------------------------------------
        combined_stats = {
            "n_groups": n_groups,
            "groups": groups,
            "per_group": all_stats,
            "faceted": True,
            # Phase 13.32.DF FIX1 BUG-001: expose ORIGINAL facet_by name to
            # consumers, not the possibly-rebinded internal temp column name.
            "facet_by": _facet_display_name,
            # Phase 13.31.DF (AD-78): expose mode for consumers to discriminate
            # between channel-name and column-name semantics.
            "facet_mode": _facet_mode,
            "n_total": sum(s.get("n", 0) for s in all_stats.values()),
        }

        # ====================================================================
        # Phase 13.43.DF v1.2 §4.2.0 — Faceted fit aggregation (Option A).
        # ────────────────────────────────────────────────────────────────────
        # 1D facet case: per-cell stats live at combined_stats['per_group'][k];
        # aggregate per-cell 'fit' into a top-level combined_stats['fit'] as
        # Shape 3 with 1-tuple keys. Phase 13.42 D4 per-cell contract is
        # PRESERVED — combined_stats['per_group'][k] still has 'fit' under it.
        # ====================================================================
        _aggregated_fits = {}
        for _gval, _cell_stats in all_stats.items():
            if not isinstance(_cell_stats, dict):
                continue
            _cell_fit = _cell_stats.get('fit')
            if _cell_fit:
                _aggregated_fits[(_gval,)] = _cell_fit
        if _aggregated_fits:
            combined_stats['fit'] = _aggregated_fits

        return fig, axes_flat[:n_groups], combined_stats

    # =========================================================================
    # Phase 13.41.DF v1.6 — N-D faceting dispatch (2D row × col, 3D figID)
    # =========================================================================
    
    def _dispatch_inner_per_cell(self, sub_df, x_expr, y_expr, plot_kind,
                                  ax_ij, _lock_x_range, _lock_y_range,
                                  **plot_kwargs):
        """Per-plot-kind inner dispatch within a 2D facet cell.
        
        Phase 13.41.DF v1.4 CP1-1 — corrected per-function range params:
          hist:     range=x_range  (matplotlib convention — NOT x_range=)
          profile:  x_range=x_range  (per profile.py:166)
          scatter:  ax.set_xlim/set_ylim post-draw (no native range params)
          hist2d/profile2d: no range params (auto-scale per cell)
        
        _lock_x_range/_lock_y_range: internal cross-figure ranges; user-supplied
        range (hist) / x_range (profile) come via plot_kwargs and take precedence.
        
        Per-cell `auto_title` is ALWAYS False — cell titles are reserved for facet
        labels (row_col=val, col_col=val). User's auto_title=True is honored at
        the figure level (suptitle) in _dispatch_2d_facet (Phase 13.41 FIX1 +
        FIX2 item 5: dead `_user_auto_title` param removed from this signature).
        """
        # Build the y:x or just x expression for the inner call
        if isinstance(y_expr, str) and y_expr:
            inner_expr = f"{y_expr}:{x_expr}"
        else:
            inner_expr = x_expr
        
        # Empty cell handling (CP2-2)
        if len(sub_df) == 0:
            ax_ij.text(0.5, 0.5, "(no data)",
                       ha='center', va='center',
                       transform=ax_ij.transAxes,
                       color='gray', fontsize=8)
            return {'n': 0, 'empty': True}
        
        # Create per-cell DFDraw view; matplotlib will use its defaults
        sub_adf = DFDraw(sub_df)
        
        # Phase 13.41.DF v1.6: reconcile user-supplied range kwargs.
        # NOTE: BOTH DFDraw.hist AND DFDraw.profile use `range:` at the
        # DFDraw layer (DFDraw.profile internally remaps to x_range when
        # calling draw_profile per drawer.py:329/404/420/436).
        # User-supplied range wins; otherwise apply our lock range.
        user_range = plot_kwargs.pop('range', None)
        # Defensive: also pop x_range (some call paths use it; we map back below)
        user_x_range = plot_kwargs.pop('x_range', None)
        effective_range = user_range if user_range is not None else (
            user_x_range if user_x_range is not None else _lock_x_range)
        
        if plot_kind == 'hist':
            # DFDraw.hist uses range= (matches matplotlib convention)
            # auto_title is handled at figure level (suptitle) in 2D facet mode,
            # not per-cell (cell titles are reserved for facet labels).
            _, _, stats_ij = sub_adf.hist(
                inner_expr, ax=ax_ij,
                range=effective_range,
                auto_title=False,
                **plot_kwargs)
            # range= sets bin range; also lock xlim for share_across_figures
            if _lock_x_range is not None and user_range is None and user_x_range is None:
                ax_ij.set_xlim(_lock_x_range)
        elif plot_kind == 'profile':
            # DFDraw.profile also uses range= (it remaps to x_range internally
            # when calling draw_profile — see drawer.py:329/404/420/436)
            _, _, stats_ij = sub_adf.profile(
                inner_expr, ax=ax_ij,
                range=effective_range,
                auto_title=False,
                **plot_kwargs)
            # range= sets bin range; also lock xlim for share_across_figures
            if _lock_x_range is not None and user_range is None and user_x_range is None:
                ax_ij.set_xlim(_lock_x_range)
        elif plot_kind == 'scatter':
            # scatter has no range params; apply ranges via ax post-draw
            _, _, stats_ij = sub_adf.scatter(
                inner_expr, ax=ax_ij, **plot_kwargs)
            if _lock_x_range is not None and user_range is None and user_x_range is None:
                ax_ij.set_xlim(_lock_x_range)
            if _lock_y_range is not None:
                ax_ij.set_ylim(_lock_y_range)
        elif plot_kind == 'hist2d':
            _, _, stats_ij = sub_adf.hist2d(
                inner_expr, ax=ax_ij, **plot_kwargs)
        elif plot_kind == 'profile2d':
            _, _, stats_ij = sub_adf.profile(
                inner_expr, ax=ax_ij, **plot_kwargs)
        else:
            raise ValueError(
                f"facet_by composition with plot_kind={plot_kind!r} not supported"
            )
        
        return stats_ij
    
    def _dispatch_2d_facet(self, df, x_expr, y_expr, facet_list, bins_list,
                            quantiles_list, plot_kind,
                            share_x='all', share_y='all',
                            _lock_x_range=None, _lock_y_range=None,
                            _user_auto_title=False,
                            title=None, group_by=None, top_k=None,
                            quantiles=None, quantile_mode='auto',
                            **plot_kwargs):
        """2D row × col grid faceting. Phase 13.41.DF v1.6 §4.
        
        _lock_x_range/_lock_y_range: internal lock from 3D share_across_figures
        (not user-facing; user's hist range / profile x_range come via plot_kwargs).
        Returns: (fig, axes_2d, stats_grid_dict)
        """
        row_col = facet_list[0]
        col_col = facet_list[1]
        row_values = _resolve_facet_values(df, row_col, bins_list[0], quantiles_list[0])
        col_values = _resolve_facet_values(df, col_col, bins_list[1], quantiles_list[1])
        
        n_rows = len(row_values)
        n_cols = len(col_values)
        
        # Build figure with axis sharing
        figsize = (4 * n_cols, 3 * n_rows)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=figsize,
            sharex=_to_mpl_share(share_x),
            sharey=_to_mpl_share(share_y),
            squeeze=False,        # always 2D for consistent indexing
        )
        
        stats_grid = {}
        for i, row_v in enumerate(row_values):
            for j, col_v in enumerate(col_values):
                # CP1-3: discrete vs binned filtering
                sub_df = _filter_facet_value(df, row_col, row_v,
                                              bins_list[0], quantiles_list[0])
                sub_df = _filter_facet_value(sub_df, col_col, col_v,
                                              bins_list[1], quantiles_list[1])
                
                ax_ij = axes[i, j]
                
                # Forward group_by, top_k, quantiles into inner if set
                inner_kwargs = dict(plot_kwargs)
                if group_by is not None:
                    inner_kwargs['group_by'] = group_by
                if top_k is not None:
                    inner_kwargs['top_k'] = top_k
                if quantiles is not None:
                    inner_kwargs['quantiles'] = quantiles
                
                stats_ij = self._dispatch_inner_per_cell(
                    sub_df, x_expr, y_expr, plot_kind, ax_ij,
                    _lock_x_range, _lock_y_range,
                    **inner_kwargs)
                stats_grid[(row_v, col_v)] = stats_ij
                
                # Edge labels (top row = col headers; left col = row labels)
                if i == 0:
                    ax_ij.set_title(f"{col_col}={col_v}", fontsize=10)
                if j == 0:
                    cur_ylabel = ax_ij.get_ylabel()
                    ax_ij.set_ylabel(f"{row_col}={row_v}\n{cur_ylabel}".rstrip())
        
        if title:
            fig.suptitle(title, fontsize=12)
        elif _user_auto_title:
            # Phase 13.41 FIX1 (Sonnet54 P2): in 2D facet mode, auto_title=True
            # sets a figure-level suptitle (cell titles are reserved for facet
            # labels). Title summarizes the plot expression + facet dimensions.
            if isinstance(y_expr, str) and y_expr and x_expr:
                _auto_expr = f"{y_expr} vs {x_expr}"
            elif isinstance(y_expr, str) and y_expr:
                _auto_expr = y_expr
            else:
                _auto_expr = x_expr or 'data'
            fig.suptitle(
                f"{_auto_expr}  [faceted by {facet_list[0]} × {facet_list[1]}]",
                fontsize=11)
        fig.tight_layout()

        # ====================================================================
        # Phase 13.43.DF v1.2 §4.2.0 — Faceted fit aggregation (Option A).
        # ────────────────────────────────────────────────────────────────────
        # Aggregate per-cell stats[(row,col)]['fit'] into a top-level
        # stats['fit'] as Shape 3 (tuple keys), so Phase 13.43 summary_fit and
        # any other downstream consumer can access fits without iterating the
        # per-cell dict structure. Phase 13.42 D4 per-cell contract is
        # PRESERVED — stats_grid[(row,col)] still has 'fit' under it; this
        # only ADDS a convenience top-level key.
        # See PHASE_13_43_DF_v1_2_SummaryFit_Proposal.md §4.2.0 and F.56.
        # ====================================================================
        if any(isinstance(k, tuple) for k in stats_grid):
            aggregated_fits = {}
            for cell_key, cell_stats in stats_grid.items():
                if not isinstance(cell_key, tuple):
                    continue
                if not isinstance(cell_stats, dict):
                    continue
                _cell_fit = cell_stats.get('fit')
                if _cell_fit:
                    aggregated_fits[cell_key] = _cell_fit
            if aggregated_fits:
                stats_grid['fit'] = aggregated_fits

        return fig, axes, stats_grid
    
    def _dispatch_3d_facet(self, df, x_expr, y_expr, facet_list, bins_list,
                            quantiles_list, plot_kind,
                            share_x='all', share_y='all',
                            share_across_figures=True,
                            _user_auto_title=False,
                            title=None, group_by=None, top_k=None,
                            quantiles=None, quantile_mode='auto',
                            **plot_kwargs):
        """3D faceting via multi-figure output. Phase 13.41.DF v1.6 §4.
        
        facet_list = [row_col, col_col, figid_col]; bins/quantiles same shape.
        Returns: (List[Figure], List[axes_2d], List[stats_grid])
        """
        figid_col = facet_list[2]
        figid_values = _resolve_facet_values(df, figid_col,
                                              bins_list[2], quantiles_list[2])
        
        # CP1-2 v1.2 + CP2-C v1.4: compute global x/y ranges
        # using existing x_expr / y_expr named params (no plot_kwargs extraction)
        global_x_range = global_y_range = None
        if share_across_figures:
            global_x_range, global_y_range = _compute_global_ranges(
                df, x_expr, y_expr, plot_kind)
        
        figures, all_axes, all_stats = [], [], []
        for figid_v in figid_values:
            sub_df = _filter_facet_value(df, figid_col, figid_v,
                                          bins_list[2], quantiles_list[2])
            
            # Phase 13.41 FIX2 item 2 (Sonnet54 P2): in 3D mode, the 2D dispatch's
            # auto_title suptitle would be overwritten by the figID label below.
            # Don't forward _user_auto_title to 2D — handle the combined suptitle
            # here so auto_title=True isn't a silent no-op in 3D.
            fig, axes, stats = self._dispatch_2d_facet(
                sub_df, x_expr, y_expr, facet_list[:2], bins_list[:2],
                quantiles_list[:2], plot_kind,
                share_x=share_x, share_y=share_y,
                _lock_x_range=global_x_range, _lock_y_range=global_y_range,
                _user_auto_title=False,    # 3D handles auto_title at fig-level (see below)
                title=None, group_by=group_by, top_k=top_k,
                quantiles=quantiles, quantile_mode=quantile_mode,
                **plot_kwargs)
            
            # Set per-figure title — 3 cases per Phase 13.41 FIX2 item 2 design:
            #   1. user title=    → "{user_title} ({figid_col} = {figid_v})"
            #   2. auto_title=True → "{expr} [faceted by {r}×{c}×{f} = {figid_v}]"
            #   3. default         → "{figid_col} = {figid_v}"  (Phase 13.41 v1.6)
            if title:
                fig.suptitle(f"{title} ({figid_col} = {figid_v})", fontsize=12)
            elif _user_auto_title:
                if isinstance(y_expr, str) and y_expr and x_expr:
                    _auto_expr = f"{y_expr} vs {x_expr}"
                elif isinstance(y_expr, str) and y_expr:
                    _auto_expr = y_expr
                else:
                    _auto_expr = x_expr or 'data'
                fig.suptitle(
                    f"{_auto_expr}  [faceted by {facet_list[0]} × {facet_list[1]} "
                    f"× {facet_list[2]} = {figid_v}]",
                    fontsize=11)
            else:
                fig.suptitle(f"{figid_col} = {figid_v}", fontsize=12)
            
            figures.append(fig)
            all_axes.append(axes)
            all_stats.append({'figid_value': figid_v, 'cells': stats})
        
        return figures, all_axes, all_stats
    
    # =========================================================================
    # Main Draw Method
    # =========================================================================
    
    def draw(
        self,
        expr: str,
        type: Optional[str] = None,
        selection: Optional[Union[str, np.ndarray, callable]] = None,
        color: Optional[str] = None,
        size: Optional[Union[str, float]] = None,
        marker: Optional[str] = None,
        group_by: Optional[str] = None,
        facet: bool = False,
        bins: Optional[Union[int, List[int]]] = None,
        stats: Optional[Union[bool, List[str]]] = None,
        norm: Optional[str] = None,
        title: Optional[str] = None,
        ax=None,
        sample: Optional[int] = None,
        save: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        # Phase 13.13.DF: same=True for superposition (AD-15)
        same: bool = False,
        # Phase 13.28.DF: NaN/inf filter policy (AD-70)
        nan_policy: str = "filter",
        # Phase 13.32.DF Sub-fix 3 (AD-79): reachable from top-level draw() too
        facet_by: Optional[Union[str, List[str]]] = None,
        facet_by_bins: Optional[Union[int, List[Optional[int]]]] = None,
        facet_by_quantiles: Optional[Union[int, List]] = None,
        # Phase 13.41.DF v1.6 + FIX1: N-D faceting axis-sharing controls
        share_x: str = 'all',
        share_y: str = 'all',
        share_across_figures: bool = True,
        # Phase 13.27.DF Commit 2 (Phase D): selection/weights vectors + per-curve label management
        # AD-61, AD-62, AD-65, AD-66, AD-67. Method body wiring lands in Turn 3.
        selection_vector: Optional[List[str]] = None,
        weights_vector: Optional[List[str]] = None,
        selection_labels: Optional[List[str]] = None,
        weights_labels: Optional[List[str]] = None,
        selection_categorical: bool = False,
        weights_categorical: bool = False,
        vector_compose: str = "inner",
        delta_facet: Optional[str] = None,
        # Phase 13.33.DF: Normalized differential profiles (AD-80/81/82) —
        # forwarded to profile() when type='profile'; other type values
        # (hist/scatter/hist2d) silently ignore these per the universal-
        # dispatch contract on _DRAW_FORWARDED_NAMES.
        normalize: Optional[Union[str, "callable"]] = None,
        normalize_layout: str = "overlay+diff",
        # Phase 13.42.DF: Inline fit specification (architect 2026-05-22).
        fit: Optional[Union[str, Dict, Callable, List]] = None,
        # Phase 13.42.DF FIX2 (ADV-3, Sonnet55 P2-2 carry-forward):
        # per-call fit textbox formatting overrides. Pattern B —
        # forwarded inward; consumed by inner draw_hist/profile/scatter
        # via render_fit_textbox.
        fit_textbox_kwargs: Optional[Dict] = None,
        # Phase 13.43.DF v1.2 (architect OQ-A1): standalone summary fit
        # figures (table + params trend). Pattern A — outer-layer
        # consume: popped from kwargs at THIS layer and rendered AFTER
        # _dispatch_faceted_render returns, NEVER forwarded into the
        # faceted renderer. NOT in _*_FORWARDED_NAMES (would fail R6
        # validator). See §4.2 / §9.1 of the v1.2 proposal.
        summary_fit: Optional[Union[str, List[str], Dict]] = None,
        **kwargs
    ) -> DrawResult:
        """
        Draw plot using TTree::Draw-like syntax.
        
        Parameters
        ----------
        expr : str
            Expression in "y:x" (2D) or "x" (1D) format.
            Supports computed expressions like "x+y:z*2".
        type : str, optional
            Plot type: "scatter", "hist", "hist2d", "profile".
            Auto-detected if None (1 var → hist, 2 vars → scatter).
        selection : str, array, or callable, optional
            Data selection/cut.
        color : str, optional
            Column for color mapping.
        size : str or float, optional
            Column for marker size or fixed size.
        marker : str, optional
            Column for marker style.
        group_by : str, optional
            Column for grouping (overlay or facet).
        facet : bool, default False
            If True with group_by, create subplots instead of overlay.
        bins : int or list, optional
            Bin count for histograms.
        stats : bool or list, optional
            Show statistics box. True for defaults, or list of stat names.
        norm : str, optional
            Histogram normalization: "count", "density", "probability".
            For cumulative distributions, use the `cumulative=True` parameter
            (Phase 13.40.DF) — NOT norm="cumulative" (raises ValueError).
        title : str, optional
            Plot title.
        ax : matplotlib Axes, optional
            Existing axes to plot on.
        sample : int, optional
            Maximum points to plot (random sampling).
        save : str, optional
            Save figure to this path.
        figsize : tuple, optional
            Figure size as (width, height) in inches.
        same : bool, default False
            If True, overlay on last axes. Phase 13.13.DF (AD-15).
            Uses self._last_ax with plt.gca() fallback.
        **kwargs
            Additional style overrides.
        
        Returns
        -------
        tuple
            (fig, ax, stats_dict)
        """
        # Phase 13.46.DF C-7: kwarg-typo guard (difflib "did you mean",
        # industry standard — argparse/click/plotly). A kwarg that is a near
        # miss for a known parameter raises with a suggestion (catches the
        # silent facet_by_bin -> facet_by_bins class); a genuinely-unknown
        # kwarg only warns (matplotlib passthrough still works, but the user
        # now SEES it). The known set K is built from the signatures of draw()
        # and every typed plot method PLUS the _*_FORWARDED_NAMES tuples
        # (N-1, Claude48): inner-method-specific kwargs forwarded via **kwargs
        # are legitimate and must not warn.
        if kwargs:
            import difflib
            import warnings
            _known_kwargs = set()
            for _m in (self.draw, self.hist, self.scatter, self.profile,
                       self.hist2d, self.hexbin):
                for _pname, _p in inspect.signature(_m).parameters.items():
                    if _pname == 'self':
                        continue
                    if _p.kind in (_p.VAR_KEYWORD, _p.VAR_POSITIONAL):
                        continue
                    _known_kwargs.add(_pname)
            for _tup in (self._DRAW_FORWARDED_NAMES, self._HIST_FORWARDED_NAMES,
                         self._SCATTER_FORWARDED_NAMES,
                         self._PROFILE_FORWARDED_NAMES,
                         self._HIST2D_FORWARDED_NAMES):
                _known_kwargs.update(_tup)
            for _name in list(kwargs):
                if _name in _known_kwargs:
                    continue
                _near = difflib.get_close_matches(
                    _name, _known_kwargs, n=1, cutoff=0.8)
                if _near:
                    raise ValueError(
                        f"Unknown keyword argument {_name!r}. "
                        f"Did you mean {_near[0]!r}? "
                        f"(dfdraw Phase 13.46.DF C-7 typo guard.)"
                    )
                else:
                    warnings.warn(
                        f"Unknown keyword argument {_name!r} — forwarded to "
                        f"matplotlib or ignored. If this is a dfdraw typo, "
                        f"check the API.",
                        UserWarning, stacklevel=2,
                    )

        # Phase 13.42.DF: AD-42 guard removed; 'fit=' is now the inline-fit
        # specification per the unified str/dict/callable/list grammar
        # (see plots/fits.py and PHASE_13_42_DF v1.4 §3).

        # Handle figsize: create axes if not provided
        if figsize is not None and ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)

        # Phase 13.39.DF Item 3: scatter3d dispatch BEFORE _parse_expr,
        # because _parse_expr rejects colon_count > 1 (z:y:x has 2).
        if type == "scatter3d":
            colon_count = self._count_colons_outside_brackets(expr)
            if colon_count != 2:
                raise ValueError(
                    f"type='scatter3d' requires a 3-variable expression "
                    f"'z:y:x', got {expr!r} with {colon_count} top-level "
                    f"colon(s)."
                )
            if group_by is not None:
                # CP2-1 (§9.SC3D.7) scope boundary
                raise ValueError(
                    "group_by is not supported with type='scatter3d' "
                    "(deferred to future phase). Got: group_by={!r}".format(
                        group_by)
                )
            z_part, y_part, x_part = self._split_top_level_colons_3(expr)
            _df_3d = self._apply_selection(self.df, selection)
            _df_3d = self._apply_sampling(_df_3d, sample)
            from .plots.scatter import draw_scatter3d
            _allowed = {
                'cmap', 'alpha', 'auto_title', 'title',
                'xlabel', 'ylabel', 'zlabel',
                'nan_policy', 'elev', 'azim',
            }
            _sc3d_kwargs = {k: v for k, v in kwargs.items() if k in _allowed}
            return draw_scatter3d(
                _df_3d, z_part, y_part, x_part,
                ax=ax, color=color, size=size,
                same=same, **_sc3d_kwargs,
            )

        # Parse expression to determine dimensionality
        y_expr, x_expr = self._parse_expr(expr)
        
        # Phase 13.16.DF: Vector expression dispatch
        # Phase 13.16.DF FIX1: tuple-driven forwarding via _DRAW_FORWARDED_NAMES.
        # Note: facet check is performed inside the routed method (profile/hist/
        # scatter), not here, because draw() forwards type/method-specific kwargs
        # via _draw_vector which calls the routed method.
        if isinstance(y_expr, list):
            # P1-4: Auto-detect type for vectors
            if type is None:
                # x_expr is a list with None entries for 1D vectors
                type = "hist" if x_expr[0] is None else "scatter"
            
            # Map type to bound method
            method_map = {
                'hist': self.hist,
                'scatter': self.scatter,
                'profile': self.profile,
            }
            if type not in method_map:
                raise ValueError(
                    f"Vector expressions not supported for type={type!r}. "
                    f"Supported types: {sorted(method_map.keys())}"
                )
            
            # R4 mirror: facet=True + vector is undefined at draw() level too.
            if facet:
                raise ValueError(
                    "facet=True is not supported with vector expression. "
                    "Use a scalar expression with facet=True for subplot grids, "
                    "or a vector expression without facet for overlay."
                )
            
            # FIX1 B1a: forward every named param via tuple.
            # type/figsize/facet/group_by/expr deliberately excluded.
            vector_kwargs = dict(kwargs)
            _local = locals()
            for name in self._DRAW_FORWARDED_NAMES:
                val = _local.get(name, _MISSING)
                if val is not _MISSING and val is not None:
                    vector_kwargs.setdefault(name, val)
            
            _fig_sf, _axes_sf, _stats_sf = self._draw_vector(
                y_expr, x_expr, method_map[type],
                group_by=group_by, **vector_kwargs
            )
            self._maybe_attach_summary_fit(
                _stats_sf, summary_fit,
                group_by=group_by, facet_by=facet_by,
                expr_for_auto_title=expr)
            return _fig_sf, _axes_sf, _stats_sf
        
        # Auto-detect type (scalar path)
        if type is None:
            if x_expr is None:
                type = "hist"
            else:
                type = "scatter"
        
        # Dispatch to specific plot method
        # Phase 13.46.DF C-2: normalize ROOT-convention type aliases (e.g.
        # "histo" -> "hist") before the dispatch ladder.
        type = _TYPE_ALIASES.get(type, type)
        # Phase 13.43.DF v1.0 R-2 (Sonnet54 panel finding): explicitly
        # forward fit / fit_textbox_kwargs / summary_fit at scalar dispatch.
        # These are NAMED params on DFDraw.draw (Phase 13.42 + 13.43), so
        # they are NOT in **kwargs after Python signature binding. The bug
        # was pre-existing for fit (Phase 13.42) and fit_textbox_kwargs
        # (Phase 13.42 FIX2 ADV-3); summary_fit (Phase 13.43) would have
        # silently dropped via the same route. Fixed all 3 together.
        if type == "hist":
            return self.hist(
                expr, selection=selection, bins=bins, stats=stats,
                norm=norm, title=title, ax=ax, sample=sample, 
                save=save, group_by=group_by, facet=facet,
                same=same,
                fit=fit, fit_textbox_kwargs=fit_textbox_kwargs,
                summary_fit=summary_fit,
                **kwargs
            )
        elif type == "scatter":
            return self.scatter(
                expr, selection=selection, color=color, size=size,
                marker=marker, stats=stats, title=title, ax=ax,
                sample=sample, save=save, group_by=group_by, 
                facet=facet, same=same,
                fit=fit, fit_textbox_kwargs=fit_textbox_kwargs,
                summary_fit=summary_fit,
                **kwargs
            )
        elif type == "hist2d":
            return self.hist2d(
                expr, selection=selection, bins=bins, stats=stats,
                title=title, ax=ax, sample=sample, save=save,
                same=same, **kwargs
            )
        elif type == "profile":
            return self.profile(
                expr, selection=selection, bins=bins, stats=stats,
                title=title, ax=ax, sample=sample, save=save,
                group_by=group_by, same=same,
                fit=fit, fit_textbox_kwargs=fit_textbox_kwargs,
                summary_fit=summary_fit,
                **kwargs
            )
        else:
            raise ValueError(
                f"Unknown plot type '{type}'. "
                "Expected: scatter, hist, hist2d, profile, scatter3d"
            )
    
    # =========================================================================
    # Plot Type Methods (Stubs - to be implemented in Phase 6.2-6.5)
    # =========================================================================
    
    def hist(
        self, 
        expr: str, 
        selection: Optional[Union[str, np.ndarray, callable]] = None,
        sample: Optional[int] = None,
        bins: Optional[int] = None,
        range: Optional[Tuple[float, float]] = None,
        norm: Optional[str] = None,
        stats: Optional[Union[bool, List[str]]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ax=None,
        save: Optional[str] = None,
        group_by: Optional[str] = None,
        facet: bool = False,
        ncols: Optional[int] = None,
        sharex: bool = True,
        sharey: bool = True,
        top_k: Optional[int] = None,
        # Phase 13.12.DF v1.2: Auto-title
        auto_title: Union[bool, str] = False,
        # Phase 13.13.DF: same=True (AD-15)
        same: bool = False,
        # Phase 13.18.DF: Robust statistics
        stat_fields: Optional[Union[str, List[str]]] = None,
        # Phase 13.28.DF: NaN/inf filter policy (AD-70)
        nan_policy: str = "filter",
        # Phase 13.39.DF: time-axis formatting (pre-conversion approach)
        time_format: Optional[str] = None,
        # Phase 13.40.DF: cumulative histogram (CDF/ECDF/survival)
        cumulative: Union[bool, int] = False,
        # Phase 13.27.DF Commit 2 FIX1 (§7b): weights as column name or
        # df.eval-able expression. Mirrors profile()'s weights= semantics.
        # If both `weights=` and `norm="probability"` are passed, the
        # explicit per-row weights win and are additionally scaled by
        # 1/n_clean for probability normalization.
        weights: Optional[str] = None,
        # Phase 13.32.DF Sub-fix 3 (AD-79): extend AD-78 facet_by column-mode to hist
        facet_by: Optional[Union[str, List[str]]] = None,
        facet_by_bins: Optional[Union[int, List[Optional[int]]]] = None,
        facet_by_quantiles: Optional[Union[int, List]] = None,
        # Phase 13.41.DF v1.6 + FIX1: N-D faceting axis-sharing controls
        share_x: str = 'all',
        share_y: str = 'all',
        share_across_figures: bool = True,
        # Phase 13.27.DF Commit 2 (Phase D): selection/weights vectors + per-curve label management
        # AD-61, AD-62, AD-65, AD-66, AD-67. Method body wiring lands in Turn 3.
        selection_vector: Optional[List[str]] = None,
        weights_vector: Optional[List[str]] = None,
        selection_labels: Optional[List[str]] = None,
        weights_labels: Optional[List[str]] = None,
        selection_categorical: bool = False,
        weights_categorical: bool = False,
        vector_compose: str = "inner",
        delta_facet: Optional[str] = None,
        # Phase 13.35.DF: float group_by binning + per-group normalization
        # (BUG-013 fix). group_by_bins / group_by_quantiles bin a float
        # group_by column via pd.cut/qcut. min_entries skips groups below
        # threshold. hist_norm: None | "probability" | "density" (per-group).
        group_by_bins: Optional[int] = None,
        group_by_quantiles: Optional[int] = None,
        hist_norm: Optional[str] = None,
        min_entries: int = 0,
        # Phase 13.36.DF: user style overrides for group_by path (BUG-013 fix).
        # color applies uniformly to all groups. marker is consumed inside
        # _draw_hist_grouped() with a UserWarning (matplotlib's ax.hist does
        # not render markers — markers are meaningless for histograms).
        # NOTE: 'markersize' deliberately absent — draw_hist() has no
        # corresponding draw_hist() explicit param; adding it to
        # _HIST_FORWARDED_NAMES would crash ax.hist via vector dispatch.
        # See PHASE_13_36_DF_v1_2_Proposal §3.3 + Sonet51 P1.
        color: Optional[str] = None,
        marker: Optional[str] = None,
        # Phase 13.37.DF: hist_errors (Poisson overlay) + linestyle_cycle
        # (per-group linestyle mode). Both forwarded explicitly to draw_hist().
        hist_errors: bool = False,
        linestyle_cycle: bool = False,
        # Phase 13.42.DF: Inline fit specification
        fit: Optional[Union[str, Dict, Callable, List]] = None,
        # Phase 13.42.DF FIX2 (ADV-3, Sonnet55 P2-2 carry-forward):
        # per-call fit textbox formatting overrides. Pattern B —
        # forwarded inward; consumed by inner draw_hist/profile/scatter
        # via render_fit_textbox.
        fit_textbox_kwargs: Optional[Dict] = None,
        # Phase 13.43.DF v1.2 (architect OQ-A1): standalone summary fit
        # figures (table + params trend). Pattern A — outer-layer
        # consume: popped from kwargs at THIS layer and rendered AFTER
        # _dispatch_faceted_render returns, NEVER forwarded into the
        # faceted renderer. NOT in _*_FORWARDED_NAMES (would fail R6
        # validator). See §4.2 / §9.1 of the v1.2 proposal.
        summary_fit: Optional[Union[str, List[str], Dict]] = None,
        **kwargs
    ) -> DrawResult:
        """
        Draw 1D histogram.
        
        Parameters
        ----------
        expr : str
            Column expression (single variable).
        selection : optional
            Data selection/cut.
        sample : int, optional
            Max points to use.
        bins : int, optional
            Number of bins.
        range : tuple, optional
            (min, max) range.
        norm : str, optional
            Normalization: "count", "density", "probability".
        stats : bool or list, optional
            Show stats box.
        title : str, optional
            Plot title.
        xlabel, ylabel : str, optional
            Axis labels.
        ax : Axes, optional
            Existing axes.
        save : str, optional
            Save path.
        group_by : str, optional
            Column for grouping.
        facet : bool, default False
            Create subplots for groups (requires group_by).
        ncols : int, optional
            Number of columns for facet grid.
        sharex : bool, default True
            Share x-axis in facet mode.
        sharey : bool, default True
            Share y-axis in facet mode.
        top_k : int, optional
            Show only top K groups.
        same : bool, default False
            If True, overlay on last axes. Phase 13.13.DF (AD-15).
        **kwargs
            Additional arguments to plt.hist().
        
        Returns
        -------
        tuple
            (fig, ax, stats_dict)
        """
        from .plots.histogram import draw_hist

        # Parse expression (take first part only for 1D)
        y_expr, x_expr = self._parse_expr(expr)

        # Phase 13.42.DF FIX2 (ADV-1, Sonnet55 v1.2 panel finding, v1.2 §8
        # D9(d) commitment): stacked=True + selection_vector (>1 selection)
        # + fit= has ambiguous semantics in the current architecture (does
        # each selection get its own stack? overlay without sum? fit per
        # selection or per-stack-component or on the stacked total?).
        # Architect did not ratify any of these in v1.2. Until a concrete
        # use case drives the design, raise NotImplementedError with a clear
        # actionable suggestion so users don't get silently wrong output.
        # Placed BEFORE vector dispatch so the 3-axis-inner length-equality
        # validator inside _compute_vector_iteration_indices does not pre-empt
        # this more-specific architectural error.
        # Note: `stacked` is forwarded via **kwargs on DFDraw.hist (not a
        # named outer param), so we read it from kwargs.
        if (kwargs.get('stacked') is True
                and selection_vector is not None and len(selection_vector) > 1
                and fit is not None):
            raise NotImplementedError(
                "Phase 13.42.DF FIX2 (ADV-1): the combination "
                "stacked=True + selection_vector (>1 selection) + fit= is not "
                "supported. Selection-vector semantics for stacked-hist fits "
                "were not ratified in Phase 13.42 (v1.2 §8 D9(d) — coder did "
                "not specify which of {per-selection fit, per-stack-component "
                "fit, fit on the stacked total} applies). "
                "Fix options: "
                "(a) drop stacked=True (per-group fits work via D9/R4); or "
                "(b) loop over selections in user code with separate hist() "
                "calls; or "
                "(c) drop fit= and use Phase 13.43 summary_fit= when "
                "available."
            )

        # Phase 13.16.DF: Vector dispatch
        # Phase 13.16.DF FIX1: tuple-driven forwarding via _HIST_FORWARDED_NAMES
        # + R4 fail-fast guard on facet=True + vector.
        # Phase 13.27.DF Commit 2 FIX1 (§7a): single-X + selection_vector /
        # weights_vector also engages vector mode (single-X is wrapped as a
        # 1-element list). The previous silent-ignore + UserWarning guard
        # has been removed — list-valued selection/weights now compose with
        # single-X correctly via vector_compose='outer'. With default
        # vector_compose='inner' on n_y=1 + n_s>=2, the iteration helper
        # raises an actionable "3-axis inner requires equal lengths" error.
        _y_is_vector = isinstance(y_expr, list)
        # Phase 13.27.DF Commit 2 FIX1 (§7a): single-Y vector dispatch is
        # gated on facet_by being absent or channel-mode 'vector' (which
        # requires _y_is_vector). When column-mode facet_by is set on
        # single-Y, defer to the facet path (Phase 13.32 dispatcher);
        # composition of column-mode facet_by + single-Y vector channels
        # is a Phase 13.33 concern (spec v1.2 §4.2.4: facet partitions
        # first, vector composes within each subplot).
        _column_mode_facet = (facet_by is not None and facet_by != 'vector')
        _need_vector_dispatch = (
            _y_is_vector
            or (
                not _column_mode_facet
                and (
                    (selection_vector is not None and len(selection_vector) >= 2)
                    or (weights_vector is not None and len(weights_vector) >= 2)
                )
            )
        )
        if _need_vector_dispatch:
            if not _y_is_vector:
                # Wrap single-X as 1-element list so _draw_vector path is uniform.
                y_expr = [y_expr]
                if not isinstance(x_expr, list):
                    x_expr = [x_expr]
            # R4: facet=True + vector is undefined; fail-fast.
            if facet:
                raise ValueError(
                    "facet=True is not supported with vector expression. "
                    "Use a scalar expression with facet=True for subplot grids, "
                    "or a vector expression without facet for overlay."
                )
            # FIX1 B1a: forward every named param via tuple.
            vector_kwargs = dict(kwargs)
            _local = locals()
            for name in self._HIST_FORWARDED_NAMES:
                val = _local.get(name, _MISSING)
                if val is not _MISSING and val is not None:
                    # FIX1: auto_title=False is signature default, not user choice.
                    if name == 'auto_title' and val is False:
                        continue
                    vector_kwargs.setdefault(name, val)
            _fig_sf, _axes_sf, _stats_sf = self._draw_vector(
                y_expr, x_expr, self.hist,
                group_by=group_by, **vector_kwargs
            )
            self._maybe_attach_summary_fit(
                _stats_sf, summary_fit,
                group_by=group_by, facet_by=facet_by,
                expr_for_auto_title=expr)
            return _fig_sf, _axes_sf, _stats_sf
        
        col_expr = y_expr  # Use y (first part) as the variable
        
        # Phase 13.13.DF: Resolve axes for same=True
        resolved_ax, is_new = self._resolve_axes(same, ax)
        if is_new:
            self._reset_color_cycle()
            ax = None
        else:
            ax = resolved_ax
        
        # Phase 13.13.DF: Inject color and label for same=True
        # Phase 13.16.DF: _suppress_color_cycle flag (P1-2)
        # Phase 13.36.DF: color is now explicit param. Two changes:
        # (1) Assign to local `color`, not `kwargs['color']` (collision with
        #     explicit forwarding to draw_hist below).
        # (2) Skip auto-color when group_by is active — same rationale as
        #     DFDraw.profile() (preserves effective pre-13.36 behavior where
        #     auto-color was silently dropped in the grouped path).
        _suppress_color_cycle = kwargs.pop('_suppress_color_cycle', False)
        save_auto_title = auto_title
        if same:
            if (color is None and not _suppress_color_cycle
                    and group_by is None):
                color = self._get_next_color()
            if 'label' not in kwargs and group_by is None:
                kwargs['label'] = self._auto_label(col_expr)
            # Suppress title handling — we do it in _handle_same_post
            if ax is not None and ax.get_title():
                title = None
                auto_title = False
        
        # Apply selection and sampling
        df = self._apply_selection(self.df, selection)
        df = self._apply_sampling(df, sample)

        # Phase 13.32.DF Sub-fix 3: validate facet_by_bins/_quantiles at entry
        self._validate_facet_by_binning(
            facet_by, facet_by_bins, facet_by_quantiles, df
        )
        
        # Evaluate expression if needed
        if col_expr not in df.columns:
            df = df.assign(**{col_expr: self._eval_column(col_expr)})
        
        # Apply duck-typed label lookup if not explicitly set
        if xlabel is None:
            duck_label = self._get_label(col_expr)
            if duck_label is not None:
                xlabel = duck_label

        # Phase 13.32.DF Sub-fix 3 (AD-79): facet_by-driven dispatch via the
        # channel framework. Takes precedence over legacy facet=True path.
        # x_expr for hist is None (1D), so we pass y_expr=col_expr.
        _effective_facet_by = facet_by
        if not _effective_facet_by and facet and group_by is not None:
            _effective_facet_by = 'group_by'  # AD-67 backward-compat
        if _effective_facet_by is not None:
            fig, axes, stats_dict = self._dispatch_faceted_render(
                df=df, x_expr=None, y_expr=col_expr,
                facet_by=_effective_facet_by, plot_kind='hist',
                bins=bins, range=range, norm=norm,
                stats=stats, title=title, xlabel=xlabel, ylabel=ylabel,
                group_by=group_by, top_k=top_k, ncols=ncols,
                sharex=sharex, sharey=sharey,
                auto_title=auto_title, selection=selection,
                stat_fields=stat_fields,
                nan_policy=nan_policy,
                # Phase 13.27.DF Commit 2 FIX1 (§7b): forward column-name weights
                weights=weights,
                facet_by_bins=facet_by_bins,
                facet_by_quantiles=facet_by_quantiles,
                # Phase 13.41.DF v1.6 + FIX1: forward N-D axis-sharing controls
                share_x=share_x,
                share_y=share_y,
                share_across_figures=share_across_figures,
                # Phase 13.35.DF: forward float group_by binning + per-group
                # normalization to per-subplot draw_hist (BUG-013 fix T3).
                # Without these, the architect's call
                #   d.hist('x', group_by='z', group_by_bins=5, facet_by='sec')
                # loses group_by_bins between method-level explicit-param
                # consumption and per-subplot draw_hist invocation.
                group_by_bins=group_by_bins,
                group_by_quantiles=group_by_quantiles,
                hist_norm=hist_norm,
                min_entries=min_entries,
                # Phase 13.40.DF: cumulative must be explicit through faceted
                # dispatch (recursive QRC v1.32 #6 — every forwarding layer
                # must pass the named param explicitly). Locked by §9.CH.8.
                cumulative=cumulative,
                # Phase 13.42.DF: inline fits — recursive forwarding through facet
                fit=fit,
                # Phase 13.42.DF FIX2 (ADV-3): fit_textbox_kwargs paired
                # with fit= (R6 validator now requires explicit forwarding).
                fit_textbox_kwargs=fit_textbox_kwargs,
                **kwargs
            )
        # Facet mode (legacy path, same=True ignored in facet mode)
        elif facet and group_by is not None:
            from .facet import facet_hist
            fig, axes, stats_dict = facet_hist(
                df, col_expr, group_by,
                top_k=top_k, ncols=ncols, sharex=sharex, sharey=sharey,
                suptitle=title, bins=bins, range=range, norm=norm,
                stats=stats, xlabel=xlabel, ylabel=ylabel, **kwargs
            )
        else:
            # Standard mode (single plot or overlay)
            fig, ax, stats_dict = draw_hist(
                df, col_expr,
                ax=ax, bins=bins, range=range, norm=norm, stats=stats,
                title=title, xlabel=xlabel, ylabel=ylabel,
                group_by=group_by, top_k=top_k,
                # Phase 13.12.DF v1.2: auto-title
                auto_title=auto_title, selection=selection,
                # Phase 13.18.DF: robust statistics
                stat_fields=stat_fields,
                # Phase 13.28.DF: NaN/inf filter policy
                nan_policy=nan_policy,
                # Phase 13.27.DF Commit 2 FIX1 (§7b): column-name weights
                weights=weights,
                # Phase 13.35.DF: float group_by binning + per-group normalization
                group_by_bins=group_by_bins,
                group_by_quantiles=group_by_quantiles,
                hist_norm=hist_norm,
                min_entries=min_entries,
                # Phase 13.36.DF: user style overrides (color → draw_hist;
                # marker flows via **kwargs after FORWARDED_NAMES, popped at
                # _draw_hist_grouped first line). Without explicit forward,
                # consumed by DFDraw.hist signature and lost.
                color=color,
                marker=marker,
                # Phase 13.37.DF: Poisson error bars + linestyle cycle mode.
                hist_errors=hist_errors,
                linestyle_cycle=linestyle_cycle,
                # Phase 13.39.DF: time-axis formatting
                time_format=time_format,
                # Phase 13.40.DF: cumulative histogram (explicit forward)
                cumulative=cumulative,
                # Phase 13.42.DF: inline fits (QRC v1.32 #6 recursive forwarding)
                fit=fit,
                fit_textbox_kwargs=fit_textbox_kwargs,
                **kwargs
            )
            axes = ax
            
            # Phase 13.13.DF: Post-draw handling for same=True
            self._handle_same_post(
                ax, same, save_auto_title, selection,
                y_name=col_expr
            )
        
        # Save if requested
        if save:
            fig.savefig(save, dpi=get_style_value("figure.dpi", 100), 
                       bbox_inches="tight")
        
        self._maybe_attach_summary_fit(
            stats_dict, summary_fit,
            group_by=group_by, facet_by=facet_by,
            expr_for_auto_title=expr)
        return fig, axes, stats_dict
    
    def scatter(
        self,
        expr: str,
        selection: Optional[Union[str, np.ndarray, callable]] = None,
        sample: Optional[int] = None,
        color: Optional[Union[str, np.ndarray]] = None,
        size: Optional[Union[str, float, np.ndarray]] = None,
        marker: Optional[str] = None,
        stats: Optional[Union[bool, List[str]]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ax=None,
        save: Optional[str] = None,
        group_by: Optional[str] = None,
        facet: bool = False,
        ncols: Optional[int] = None,
        sharex: bool = True,
        sharey: bool = True,
        top_k: Optional[int] = None,
        cmap: Optional[str] = None,
        colorbar: bool = True,
        clabel: Optional[str] = None,
        jitter: Optional[Union[bool, float, Tuple[float, float]]] = None,
        # Phase 13.13.DF: same=True (AD-15)
        same: bool = False,
        # Phase 13.28.DF: NaN/inf filter policy (AD-70)
        nan_policy: str = "filter",
        # Phase 13.32.DF Sub-fix 3 (AD-79): extend AD-78 facet_by column-mode to scatter
        facet_by: Optional[Union[str, List[str]]] = None,
        facet_by_bins: Optional[Union[int, List[Optional[int]]]] = None,
        facet_by_quantiles: Optional[Union[int, List]] = None,
        # Phase 13.41.DF v1.6 + FIX1: N-D faceting axis-sharing controls
        share_x: str = 'all',
        share_y: str = 'all',
        share_across_figures: bool = True,
        # Phase 13.38.DF: scatter error bars (column name or df.eval() expression).
        # When either is set, render via ax.errorbar() instead of ax.scatter().
        # NaN/inf policy in _eval_error(): raise on 100% non-finite, warn at >50%,
        # silent zeroing at <=50%. Locked by §9.SE.6 (3-part).
        xerr: Optional[str] = None,
        yerr: Optional[str] = None,
        time_format: Optional[str] = None,  # Phase 13.39.DF: time-axis formatting
        # Phase 13.27.DF Commit 2 (Phase D): selection/weights vectors + per-curve label management
        # AD-61, AD-62, AD-65, AD-66, AD-67. Method body wiring lands in Turn 3.
        # NB: weights_vector is accepted to honor uniform-API contract (A-1), but
        # has no rendering effect on scatter — proposal §5.5; one-time UserWarning
        # emitted in Turn 3 implementation.
        selection_vector: Optional[List[str]] = None,
        weights_vector: Optional[List[str]] = None,
        selection_labels: Optional[List[str]] = None,
        weights_labels: Optional[List[str]] = None,
        selection_categorical: bool = False,
        weights_categorical: bool = False,
        vector_compose: str = "inner",
        delta_facet: Optional[str] = None,
        # Phase 13.42.DF: Inline fit specification
        fit: Optional[Union[str, Dict, Callable, List]] = None,
        # Phase 13.42.DF FIX2 (ADV-3, Sonnet55 P2-2 carry-forward):
        # per-call fit textbox formatting overrides. Pattern B —
        # forwarded inward; consumed by inner draw_hist/profile/scatter
        # via render_fit_textbox.
        fit_textbox_kwargs: Optional[Dict] = None,
        # Phase 13.43.DF v1.2 (architect OQ-A1): standalone summary fit
        # figures (table + params trend). Pattern A — outer-layer
        # consume: popped from kwargs at THIS layer and rendered AFTER
        # _dispatch_faceted_render returns, NEVER forwarded into the
        # faceted renderer. NOT in _*_FORWARDED_NAMES (would fail R6
        # validator). See §4.2 / §9.1 of the v1.2 proposal.
        summary_fit: Optional[Union[str, List[str], Dict]] = None,
        **kwargs
    ) -> DrawResult:
        """
        Draw scatter plot.
        
        Parameters
        ----------
        expr : str
            Expression in "y:x" format.
        selection : optional
            Data selection/cut.
        sample : int, optional
            Max points to plot.
        color : str, array, optional
            Color mapping column or fixed color.
        size : str, float, optional
            Size mapping column or fixed size.
        marker : str, optional
            Marker style.
        stats : bool or list, optional
            Show stats box.
        title : str, optional
            Plot title.
        xlabel, ylabel : str, optional
            Axis labels.
        ax : Axes, optional
            Existing axes.
        save : str, optional
            Save path.
        group_by : str, optional
            Column for grouping.
        facet : bool, default False
            Create subplots for groups (requires group_by).
        ncols : int, optional
            Number of columns for facet grid.
        sharex : bool, default True
            Share x-axis in facet mode.
        sharey : bool, default True
            Share y-axis in facet mode.
        top_k : int, optional
            Show only top K groups.
        cmap : str, optional
            Colormap name.
        colorbar : bool, default True
            Show colorbar.
        clabel : str, optional
            Colorbar label.
        jitter : bool, float, or tuple, optional
            Add jitter to points.
        same : bool, default False
            If True, overlay on last axes. Phase 13.13.DF (AD-15).
        **kwargs
            Additional arguments to plt.scatter().
        
        Returns
        -------
        tuple
            (fig, ax, stats_dict)
        """
        from .plots.scatter import draw_scatter

        # Phase 13.27.DF Commit 2 (v1.2 §5.5, §9.WDS.1): weights_vector has no
        # effect on scatter (point-size weighting deferred to Phase E). Emit
        # one-time UserWarning at method entry — fires for both single-Y scalar
        # path AND vector path, ensuring users see the message regardless of
        # whether _draw_vector is engaged. The kwarg is silently dropped after.
        if weights_vector is not None:
            import warnings as _warnings
            _warnings.warn(
                "weights_vector has no effect on scatter (point-size weighting "
                "deferred to Phase E)",
                UserWarning,
                stacklevel=2,
            )
            weights_vector = None

        # Phase 13.27 Commit 2 FIX1-pending guard (Sonnet52_R1 P1-2 / Hard
        # Constraint §3) REMOVED in FIX1 §7a: single-Y + selection_vector now
        # engages vector mode below. The earlier `weights_vector` warning
        # block (Phase E deferral — point-size weighting) is preserved.

        # Parse expression
        y_expr, x_expr = self._parse_expr(expr)

        # Phase 13.16.DF: Vector dispatch
        # Phase 13.16.DF FIX1: tuple-driven forwarding via _SCATTER_FORWARDED_NAMES
        # + R4 fail-fast guard on facet=True + vector.
        # Phase 13.27.DF Commit 2 FIX1 (§7a): single-Y + selection_vector
        # also engages vector mode.
        _y_is_vector = isinstance(y_expr, list)
        # Phase 13.27.DF Commit 2 FIX1 (§7a): see note in hist()/profile().
        _column_mode_facet = (facet_by is not None and facet_by != 'vector')
        _need_vector_dispatch = (
            _y_is_vector
            or (
                not _column_mode_facet
                and (selection_vector is not None and len(selection_vector) >= 2)
            )
            # weights_vector is already None by here (warn-then-drop above)
        )
        if _need_vector_dispatch:
            if not _y_is_vector:
                y_expr = [y_expr]
                if not isinstance(x_expr, list):
                    x_expr = [x_expr]
            if x_expr[0] is None:
                raise ValueError(
                    f"Scatter plot requires 'y:x' format, got vector 1D '{expr}'"
                )
            # R4: facet=True + vector is undefined; fail-fast.
            if facet:
                raise ValueError(
                    "facet=True is not supported with vector expression. "
                    "Use a scalar expression with facet=True for subplot grids, "
                    "or a vector expression without facet for overlay."
                )
            # FIX1 B1a: forward every named param via tuple.
            vector_kwargs = dict(kwargs)
            _local = locals()
            for name in self._SCATTER_FORWARDED_NAMES:
                val = _local.get(name, _MISSING)
                if val is not _MISSING and val is not None:
                    vector_kwargs.setdefault(name, val)
            _fig_sf, _axes_sf, _stats_sf = self._draw_vector(
                y_expr, x_expr, self.scatter,
                group_by=group_by, **vector_kwargs
            )
            self._maybe_attach_summary_fit(
                _stats_sf, summary_fit,
                group_by=group_by, facet_by=facet_by,
                expr_for_auto_title=expr)
            return _fig_sf, _axes_sf, _stats_sf
        
        if x_expr is None:
            raise ValueError(
                f"Scatter plot requires 'y:x' format, got '{expr}'"
            )
        
        # Phase 13.13.DF: Resolve axes for same=True
        resolved_ax, is_new = self._resolve_axes(same, ax)
        if is_new:
            self._reset_color_cycle()
            ax = None
        else:
            ax = resolved_ax
        
        # Phase 13.13.DF: Inject color and label for same=True
        # Phase 13.16.DF: _suppress_color_cycle flag (P1-2)
        _suppress_color_cycle = kwargs.pop('_suppress_color_cycle', False)
        if same:
            if color is None and not _suppress_color_cycle:
                color = self._get_next_color()
            if 'label' not in kwargs and group_by is None:
                kwargs['label'] = self._auto_label(y_expr, x_expr)
            # Suppress title — handled in _handle_same_post
            if ax is not None and ax.get_title():
                title = None
        
        # Apply selection and sampling
        df = self._apply_selection(self.df, selection)
        df = self._apply_sampling(df, sample)

        # Phase 13.32.DF Sub-fix 3: validate facet_by_bins/_quantiles at entry
        self._validate_facet_by_binning(
            facet_by, facet_by_bins, facet_by_quantiles, df
        )
        
        # Evaluate expressions if needed
        if y_expr not in df.columns:
            df = df.assign(**{y_expr: self._eval_column(y_expr)})
        if x_expr not in df.columns:
            df = df.assign(**{x_expr: self._eval_column(x_expr)})
        
        # Apply duck-typed label lookup if not explicitly set
        if xlabel is None:
            duck_label = self._get_label(x_expr)
            if duck_label is not None:
                xlabel = duck_label
        if ylabel is None:
            duck_label = self._get_label(y_expr)
            if duck_label is not None:
                ylabel = duck_label

        # Phase 13.32.DF Sub-fix 3 (AD-79): facet_by-driven dispatch via the
        # channel framework. Takes precedence over legacy facet=True path.
        _effective_facet_by = facet_by
        if not _effective_facet_by and facet and group_by is not None:
            _effective_facet_by = 'group_by'  # AD-67 backward-compat
        if _effective_facet_by is not None:
            fig, axes, stats_dict = self._dispatch_faceted_render(
                df=df, x_expr=x_expr, y_expr=y_expr,
                facet_by=_effective_facet_by, plot_kind='scatter',
                color=color, size=size, marker=marker,
                stats=stats, title=title, xlabel=xlabel, ylabel=ylabel,
                group_by=group_by, top_k=top_k, ncols=ncols,
                sharex=sharex, sharey=sharey,
                cmap=cmap, colorbar=colorbar, clabel=clabel, jitter=jitter,
                selection=selection,
                nan_policy=nan_policy,
                facet_by_bins=facet_by_bins,
                facet_by_quantiles=facet_by_quantiles,
                # Phase 13.41.DF v1.6 + FIX1: forward N-D axis-sharing controls
                share_x=share_x,
                share_y=share_y,
                share_across_figures=share_across_figures,
                # Phase 13.42.DF: inline fits
                fit=fit,
                fit_textbox_kwargs=fit_textbox_kwargs,
                **kwargs
            )
        # Facet mode (legacy path, same=True ignored in facet mode)
        elif facet and group_by is not None:
            from .facet import facet_scatter
            fig, axes, stats_dict = facet_scatter(
                df, x_expr, y_expr, group_by,
                top_k=top_k, ncols=ncols, sharex=sharex, sharey=sharey,
                suptitle=title, color=color, size=size, marker=marker,
                stats=stats, xlabel=xlabel, ylabel=ylabel,
                cmap=cmap, colorbar=colorbar, clabel=clabel,
                jitter=jitter, **kwargs
            )
        else:
            # Standard mode (single plot or overlay)
            fig, ax, stats_dict = draw_scatter(
                df, x_expr, y_expr,
                ax=ax, color=color, size=size, marker=marker,
                stats=stats, title=title, xlabel=xlabel, ylabel=ylabel,
                group_by=group_by, top_k=top_k, cmap=cmap, colorbar=colorbar,
                clabel=clabel, jitter=jitter,
                # Phase 13.28.DF: NaN/inf filter policy
                nan_policy=nan_policy,
                # Phase 13.38.DF: scatter error bars
                xerr=xerr, yerr=yerr,
                # Phase 13.39.DF: time-axis formatting
                time_format=time_format,
                # Phase 13.42.DF: inline fits
                fit=fit,
                fit_textbox_kwargs=fit_textbox_kwargs,
                **kwargs
            )
            axes = ax
            
            # Phase 13.13.DF: Post-draw handling for same=True
            self._handle_same_post(
                ax, same, False, selection,
                y_name=y_expr, x_name=x_expr
            )
        
        # Save if requested
        if save:
            fig.savefig(save, dpi=get_style_value("figure.dpi", 100),
                       bbox_inches="tight")
        
        self._maybe_attach_summary_fit(
            stats_dict, summary_fit,
            group_by=group_by, facet_by=facet_by,
            expr_for_auto_title=expr)
        return fig, axes, stats_dict
    
    def profile(
        self,
        expr: str,
        selection: Optional[Union[str, np.ndarray, callable]] = None,
        sample: Optional[int] = None,
        bins: Optional[Union[int, List[int]]] = None,
        bins2: Optional[int] = None,  # Phase 13.39.DF: y-axis bin count for 2D profile
        time_format: Optional[str] = None,  # Phase 13.39.DF: time-axis formatting
        range: Optional[Tuple[float, float]] = None,
        error: Optional[str] = None,   # FIX1: None → resolve per context
        stats: Optional[Union[bool, List[str]]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ax=None,
        save: Optional[str] = None,
        group_by: Optional[str] = None,
        facet: bool = False,
        ncols: Optional[int] = None,
        sharex: bool = True,
        sharey: bool = True,
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
        # Phase 13.13.DF: same=True (AD-15)
        same: bool = False,
        # Phase 13.18.DF: Robust statistics
        stat_fields: Optional[Union[str, List[str]]] = None,
        # Phase 13.25.DF (Phase A): Quantile rendering
        quantiles: Optional[List[float]] = None,
        central: Optional[str] = None,
        quantile_mode: str = "auto",
        # Phase 13.26.DF (Phase B): Channel-aware quantile rendering
        quantile_style: Optional[str] = None,
        # Phase 13.28.DF: NaN/inf filter policy (AD-70)
        nan_policy: str = "filter",
        # Phase 13.27.DF (Phase D): Facet routing through channel framework (AD-61, AD-67)
        facet_by: Optional[Union[str, List[str]]] = None,
        # Phase 13.32.DF Sub-fix 3 (AD-79): symmetric binning on facet_by axis
        facet_by_bins: Optional[Union[int, List[Optional[int]]]] = None,
        facet_by_quantiles: Optional[Union[int, List]] = None,
        # Phase 13.41.DF v1.6 + FIX1: N-D faceting axis-sharing controls
        share_x: str = 'all',
        share_y: str = 'all',
        share_across_figures: bool = True,
        # Phase 13.27.DF Commit 2 (Phase D): selection/weights vectors + per-curve label management
        # AD-61, AD-62, AD-65, AD-66, AD-67. Method body wiring lands in Turn 3.
        selection_vector: Optional[List[str]] = None,
        weights_vector: Optional[List[str]] = None,
        selection_labels: Optional[List[str]] = None,
        weights_labels: Optional[List[str]] = None,
        selection_categorical: bool = False,
        weights_categorical: bool = False,
        vector_compose: str = "inner",
        delta_facet: Optional[str] = None,
        # Phase 13.33.DF: Normalized differential profiles (AD-80/81/82)
        # When set, profile() requires exactly 2 vector elements (vector[0]=signal,
        # vector[1]=reference per AD-80) and renders a bottom panel showing the
        # differential transform of the two. Mutually exclusive with same=True
        # (the diff panel can't share an existing figure). See
        # PHASE_13_33_DF_v1_1_Proposal_NormalizedDifferentialProfiles.md.
        normalize: Optional[Union[str, "callable"]] = None,
        normalize_layout: str = "overlay+diff",
        # Phase 13.36.DF: user style overrides for group_by path (BUG-013 fix).
        # When passed, applies uniformly to ALL groups in the call (overrides
        # the per-group auto-cycle). None = use auto-cycle (default).
        # See PHASE_13_36_DF_v1_2_Proposal_UserStyleOverride.md.
        color: Optional[str] = None,
        marker: Optional[str] = None,
        markersize: Optional[float] = None,
        # Phase 13.37.DF: per-group linestyle cycling mode flag. Composes with
        # Phase 13.36 sentinel (user explicit linestyle= wins via
        # _ud_user_linestyle capture).
        linestyle_cycle: bool = False,
        # Phase 13.42.DF: Inline fit specification
        fit: Optional[Union[str, Dict, Callable, List]] = None,
        # Phase 13.42.DF FIX2 (ADV-3, Sonnet55 P2-2 carry-forward):
        # per-call fit textbox formatting overrides. Pattern B —
        # forwarded inward; consumed by inner draw_hist/profile/scatter
        # via render_fit_textbox.
        fit_textbox_kwargs: Optional[Dict] = None,
        # Phase 13.43.DF v1.2 (architect OQ-A1): standalone summary fit
        # figures (table + params trend). Pattern A — outer-layer
        # consume: popped from kwargs at THIS layer and rendered AFTER
        # _dispatch_faceted_render returns, NEVER forwarded into the
        # faceted renderer. NOT in _*_FORWARDED_NAMES (would fail R6
        # validator). See §4.2 / §9.1 of the v1.2 proposal.
        summary_fit: Optional[Union[str, List[str], Dict]] = None,
        **kwargs
    ) -> DrawResult:
        """
        Draw profile plot (mean of y vs binned x).
        
        Parameters
        ----------
        expr : str
            Expression in "y:x" format.
        selection : optional
            Data selection/cut.
        sample : int, optional
            Max points to use.
        bins : int, optional
            Number of bins for x.
        range : tuple, optional
            (min, max) range for x.
        error : str, default "sem"
            Error type: "sem", "std", "none".
        stats : bool or list, optional
            Show stats box.
        title : str, optional
            Plot title.
        xlabel, ylabel : str, optional
            Axis labels.
        ax : Axes, optional
            Existing axes.
        save : str, optional
            Save path.
        group_by : str, optional
            Column for grouping.
        facet : bool, default False
            Create subplots for groups (requires group_by).
        ncols : int, optional
            Number of columns for facet grid.
        sharex : bool, default True
            Share x-axis in facet mode.
        sharey : bool, default True
            Share y-axis in facet mode.
        top_k : int, optional
            Show only top K groups.
        return_data : bool, default False
            If True, include 'profile_data' DataFrame in stats_dict.
            Phase 13.12.DF F1.
        min_entries : int, default 3
            Minimum entries per bin to be plotted. Bins with fewer entries
            are excluded from the plot but included in profile_data.
            AD-1: default=3 for stable error bars. Phase 13.12.DF F2.
        group_by_bins : int, optional
            Number of equal-width bins for float group_by column.
            Mutually exclusive with group_by_quantiles. Phase 13.12.DF F3.
        group_by_quantiles : int, optional
            Number of equal-count quantile bins for float group_by column.
            Mutually exclusive with group_by_bins. Phase 13.12.DF F3.
        sort_groups : bool, default True
            If True, sort groups numerically/alphabetically in legend.
            Phase 13.12.DF F4.
        weights : str, optional
            Column name for weights. If provided, computes weighted mean/std/sem.
            Useful for reconstructing distributions from importance sampling.
            Phase 13.12.DF v1.1.
        auto_title : bool or str, default False
            Automatic title. True/"all", "expr", "expr+group", "expr+sel".
            Phase 13.12.DF v1.2.
        same : bool, default False
            If True, overlay on last axes. Phase 13.13.DF (AD-15).
        **kwargs
            Additional arguments.
        
        Returns
        -------
        tuple
            (fig, ax, stats_dict)
            
            If return_data=True, stats_dict['profile_data'] contains DataFrame
            with columns: x_center, x_low, x_high, y_mean, y_std, y_sem, count,
            and 'group' if group_by is used.
        """
        from .plots.profile import draw_profile

        # Phase 13.27 Commit 2 FIX1-pending guard (Sonnet52_R1 P1-2 / Hard
        # Constraint §3) REMOVED in FIX1 §7a: single-Y + selection_vector /
        # weights_vector now engages vector mode below.

        # =====================================================================
        # Phase 13.39.DF CP1-5: 2D Profile dispatch (z:y:x expression).
        # Intercept BEFORE _parse_expr() rejects colon_count > 1 at line ~210.
        # =====================================================================
        colon_count = self._count_colons_outside_brackets(expr)
        if colon_count == 2:
            # CP2-1 (§9.P2D.10) scope boundary: group_by + profile2d raises
            if group_by is not None:
                raise ValueError(
                    "group_by is not supported with 2D profile (z:y:x) "
                    "expression. Use 1D profile (y:x) with group_by, or wait "
                    "for a future phase. Got: expr={!r}, group_by={!r}".format(
                        expr, group_by)
                )
            # Apply selection + sampling before dispatching (mirrors how
            # draw_profile would do internally; needed because draw_profile2d
            # expects pre-filtered df).
            _df_2d = self._apply_selection(self.df, selection)
            _df_2d = self._apply_sampling(_df_2d, sample)
            z_part, y_part, x_part = self._split_top_level_colons_3(expr)
            from .plots.profile import draw_profile2d
            # Surface kwargs accepted by draw_profile2d; ignore unknown.
            _allowed = {
                'cmap', 'vmin', 'vmax', 'clabel', 'colorbar', 'norm',
                'auto_title', 'title', 'xlabel', 'ylabel',
                'nan_policy', 'x_range', 'y_range', 'central',
            }
            _profile2d_kwargs = {k: v for k, v in kwargs.items() if k in _allowed}
            # min_entries_2d is via kwargs (separate from 1D min_entries semantics)
            _min_entries_2d = kwargs.get('min_entries_2d', 0)
            return draw_profile2d(
                _df_2d, z_part, y_part, x_part,
                ax=ax, bins=bins if bins is not None else 50,
                bins2=bins2, min_entries=_min_entries_2d,
                time_format=time_format,
                **_profile2d_kwargs,
            )
        # =====================================================================

        # Parse expression
        y_expr, x_expr = self._parse_expr(expr)

        # Phase 13.16.DF: Vector dispatch
        # Phase 13.16.DF FIX1: tuple-driven forwarding via _PROFILE_FORWARDED_NAMES
        # + R4 fail-fast guard on facet=True + vector.
        # Phase 13.27.DF Commit 2 FIX1 (§7a): single-Y + selection_vector /
        # weights_vector also engages vector mode. The facet_by='vector'
        # short-circuit still requires _y_is_vector (it splits subplots per
        # vector y element — undefined for single-Y).
        _y_is_vector = isinstance(y_expr, list)

        # =====================================================================
        # Phase 13.33.DF (AD-80/81/82): normalize= validation + §6 directive.
        # =====================================================================
        # Validation block runs BEFORE vector-dispatch decision so error
        # messages are actionable at the public API entry. The §6 directive
        # (Phase 13.27 FIX1.FIX1 deferral, option c) follows.
        if normalize is not None:
            # (a) Mode validation — string in NORMALIZE_MODES or callable.
            from .plots.profile import NORMALIZE_MODES
            if not callable(normalize) and normalize not in NORMALIZE_MODES:
                raise ValueError(
                    f"normalize must be one of {NORMALIZE_MODES} or a callable, "
                    f"got {normalize!r}."
                )
            # (b) Layout validation.
            if normalize_layout not in ("overlay+diff", "diff_only"):
                raise ValueError(
                    f"normalize_layout must be 'overlay+diff' or 'diff_only', "
                    f"got {normalize_layout!r}."
                )
            # (c) same=True is incompatible — the diff panel needs its own
            # figure and gridspec; cannot overlay on existing axes.
            if same:
                raise ValueError(
                    "normalize= is mutually exclusive with same=True — the "
                    "differential bottom panel requires a new figure. Either "
                    "drop same=True or call normalize-mode profile() in its "
                    "own figure."
                )
            # (d) Exactly 2 vector elements required (AD-80: vector[0]=signal,
            # vector[1]=reference). The vector source is either y (multi-Y),
            # selection_vector, or weights_vector.
            n_y_eff = len(y_expr) if _y_is_vector else 1
            n_sel = len(selection_vector) if selection_vector is not None else 0
            n_w = len(weights_vector) if weights_vector is not None else 0
            # The "effective vector length" is the max of these three.
            n_vec = max(n_y_eff, n_sel, n_w)
            if n_vec != 2:
                raise ValueError(
                    f"normalize= requires exactly 2 vector elements (signal + "
                    f"reference per AD-80). Got n_y={n_y_eff}, "
                    f"n_selection_vector={n_sel}, n_weights_vector={n_w} "
                    f"(effective vector length {n_vec})."
                )
            # (e) §6 directive — Phase 13.27 FIX1.FIX1 deferred closure
            # (option c, panel vote 3/5 incl. Main Reviewer). When normalize
            # is set with single-Y + 2-element selection_vector, force
            # vector_compose='outer' transparently. The normalize= API hides
            # vector_compose mechanics — user writes normalize='delta' and
            # gets the right result without touching compose semantics. This
            # is convention application, not a workaround.
            if (not _y_is_vector
                    and selection_vector is not None
                    and len(selection_vector) == 2):
                vector_compose = "outer"
        # =====================================================================

        # Phase 13.27.DF Commit 2 FIX1 (§7a): single-Y vector dispatch is
        # gated on facet_by being absent or channel-mode 'vector' (which
        # requires _y_is_vector). When column-mode facet_by is set on
        # single-Y, defer to the facet path (Phase 13.32 dispatcher);
        # composition of column-mode facet_by + single-Y vector channels
        # is a Phase 13.33 concern (spec v1.2 §4.2.4: facet partitions
        # first, vector composes within each subplot).
        _column_mode_facet = (facet_by is not None and facet_by != 'vector')
        # Phase 13.33.DF M2: normalize= always engages vector dispatch
        # (regardless of facet_by). The normalize routing fork below then
        # picks the appropriate dispatcher (faceted / grouped / single).
        # This pre-empts the _column_mode_facet short-circuit that would
        # otherwise route to the regular faceted path which knows nothing
        # about normalize.
        _need_vector_dispatch = (
            _y_is_vector
            or (normalize is not None)
            or (
                not _column_mode_facet
                and (
                    (selection_vector is not None and len(selection_vector) >= 2)
                    or (weights_vector is not None and len(weights_vector) >= 2)
                )
            )
        )
        if _need_vector_dispatch:
            if not _y_is_vector:
                y_expr = [y_expr]
                if not isinstance(x_expr, list):
                    x_expr = [x_expr]
            # Confirm 2D — profile requires x
            if x_expr[0] is None:
                raise ValueError(
                    f"Profile plot requires 'y:x' format, got vector 1D '{expr}'"
                )
            # R4: facet=True + vector is undefined; fail-fast with actionable message.
            if facet:
                raise ValueError(
                    "facet=True is not supported with vector expression. "
                    "Use a scalar expression with facet=True for subplot grids, "
                    "or a vector expression without facet for overlay."
                )
            # Phase 13.27.DF: facet_by='vector' on a vector expression splits
            # into one subplot per vector element (intercept BEFORE _draw_vector
            # which would otherwise overlay all curves). All other facet_by
            # values are not yet defined for vector y; route through _draw_vector.
            # FIX1 §7a: gate on _y_is_vector — single-Y wrapped as 1-element
            # list has no vector channel to split.
            if facet_by == 'vector' and _y_is_vector:
                # Apply selection / sampling at the dispatch level (same as
                # the scalar branch below) so each subplot sees clean data.
                df_for_facet = self._apply_selection(self.df, selection)
                df_for_facet = self._apply_sampling(df_for_facet, sample)
                # Use scalar x for facet (vector x is not supported for profile)
                x_for_facet = x_expr[0] if isinstance(x_expr, list) else x_expr
                _fig_sf, _axes_sf, _stats_sf = self._dispatch_faceted_render(
                    df=df_for_facet, x_expr=x_for_facet, y_expr=y_expr,
                    facet_by='vector', plot_kind='profile',
                    bins=bins, x_range=range, error=error,
                    stats=stats, title=title, xlabel=xlabel, ylabel=ylabel,
                    group_by=group_by, top_k=top_k, ncols=ncols,
                    sharex=sharex, sharey=sharey,
                    return_data=return_data, min_entries=min_entries,
                    group_by_bins=group_by_bins, group_by_quantiles=group_by_quantiles,
                    sort_groups=sort_groups, weights=weights,
                    auto_title=auto_title, selection=selection,
                    stat_fields=stat_fields,
                    quantiles=quantiles, central=central, quantile_mode=quantile_mode,
                    quantile_style=quantile_style,
                    nan_policy=nan_policy,
                    **kwargs
                )
                self._maybe_attach_summary_fit(
                    _stats_sf, summary_fit,
                    group_by=group_by, facet_by=facet_by,
                    expr_for_auto_title=expr)
                return _fig_sf, _axes_sf, _stats_sf
            # FIX1 B1a: forward every named param via tuple + locals().get(name, _MISSING).
            # _MISSING distinguishes "caller didn't pass" from "caller passed None".
            vector_kwargs = dict(kwargs)
            _local = locals()
            for name in self._PROFILE_FORWARDED_NAMES:
                val = _local.get(name, _MISSING)
                if val is not _MISSING and val is not None:
                    # FIX1: auto_title=False is signature default, not user choice.
                    if name == 'auto_title' and val is False:
                        continue
                    vector_kwargs.setdefault(name, val)
            # Phase 13.33.DF: Normalize-mode dispatch routes to two-pass
            # orchestrator instead of the standard _draw_vector overlay path.
            # The orchestrator handles signal/reference roles, gridspec layout,
            # transform computation, and bottom-panel rendering atomically.
            #
            # M2 (group_by / facet_by composition): when group_by is set,
            # route to _dispatch_normalize_grouped_render (per-group differential).
            # When facet_by is set, route to _dispatch_normalize_faceted_render
            # (K×2 grid). When both are set, the panel approved facet_by as
            # the outer dimension — facet wins.
            if normalize is not None:
                _consumed = {
                    'normalize', 'normalize_layout',
                    'selection', 'sample',
                    'selection_vector', 'weights_vector', 'vector_compose',
                    'bins', 'range', 'error', 'central',
                    'title', 'xlabel', 'ylabel',
                    'weights', 'nan_policy',
                    'group_by', 'facet_by', 'facet_by_bins', 'facet_by_quantiles',
                    # Phase 13.34.DF FIX1 BUG-010: 'auto_title' REMOVED from _consumed.
                    # The 3 normalize dispatchers (_dispatch_normalize_render,
                    # _grouped_render, _faceted_render) now read auto_title from
                    # **passthrough to handle the figure-level suptitle, matching
                    # the Phase 13.32 FIX1 BUG-002 pattern in _dispatch_faceted_render.
                    'same', 'return_data', 'min_entries',
                    'stats', 'stat_fields', 'top_k', 'group_by_bins',
                    'group_by_quantiles', 'sort_groups', 'ax', 'save',
                    'quantiles', 'quantile_mode', 'quantile_style',
                    'selection_labels', 'weights_labels',
                    'selection_categorical', 'weights_categorical',
                    'delta_facet',
                    # ────────────────────────────────────────────────────────
                    # Phase 13.41.DF FIX1.1 + FIX2 item 3 (Sonnet51/52/54 P2):
                    # When normalize= AND facet_by= are BOTH set, routing goes
                    # to the legacy K×2 normalize+facet dispatcher (predates
                    # Phase 13.41 N-D faceting). That dispatcher uses its own
                    # GridSpec layout and does NOT honor share_x/share_y/
                    # share_across_figures. We filter these out of _passthrough
                    # so they don't reach _dispatch_normalize_faceted_render
                    # via **passthrough (which would silently swallow them
                    # anyway — but explicit filtering documents intent and
                    # avoids surprise in future maintenance).
                    #
                    # BEHAVIOR: d.profile(normalize=..., facet_by=['a','b'],
                    # share_x='row')  →  K×2 grid, share_x SILENTLY IGNORED.
                    # WORKAROUND: use normalize= or facet_by=List[str], not both.
                    # FIX2 BACKLOG: add N-D faceting support to normalize
                    # dispatcher (or raise NotImplementedError with hint).
                    # ────────────────────────────────────────────────────────
                    'share_x', 'share_y', 'share_across_figures',
                    # Phase 13.42.DF §4.2b (CP1-5): fit silently consumed when
                    # normalize is active. The legacy normalize dispatcher
                    # predates inline fits and doesn't honor fit=. Workaround:
                    # compute the normalized values into an alias column with
                    # adf.add_alias(), then call draw() on the alias with fit=.
                    'fit',
                    # Phase 13.43.DF v1.2 §4.6 (C-3 lock): summary_fit also
                    # consumed when normalize= is active — there's no fit to
                    # summarize. The outer dispatch detects "summary_fit
                    # requested but stats['fit'] empty" and produces
                    # Scenario E: stats['summary_fit'] = {} +
                    # stats['summary_fit_note'].
                    # Also consume fit_textbox_kwargs (Phase 13.42 FIX2 ADV-3
                    # promoted to outer-named param): no fit textbox to format
                    # when fit is consumed.
                    'summary_fit',
                    'fit_textbox_kwargs',
                }
                _passthrough = {k: v for k, v in vector_kwargs.items()
                                if k not in _consumed}

                # Route hierarchy (panel-decided priority):
                #   facet_by → faceted dispatcher (K×2 grid, outer dimension)
                #   group_by → grouped dispatcher (per-group diff curves)
                #   else    → M1 single-render dispatcher
                if facet_by is not None:
                    _fig_sf, _ax_sf, _stats_sf = self._dispatch_normalize_faceted_render(
                        y_expr, x_expr,
                        normalize=normalize, normalize_layout=normalize_layout,
                        facet_by=facet_by,
                        facet_by_bins=facet_by_bins,
                        facet_by_quantiles=facet_by_quantiles,
                        selection=selection, sample=sample,
                        selection_vector=selection_vector,
                        weights_vector=weights_vector,
                        vector_compose=vector_compose,
                        bins=bins, x_range=range, error=error,
                        central=central, title=title,
                        xlabel=xlabel, ylabel=ylabel,
                        weights=weights, nan_policy=nan_policy,
                        ncols=ncols,
                        **_passthrough,
                    )
                    self._maybe_attach_summary_fit(
                        _stats_sf, summary_fit,
                        group_by=group_by, facet_by=facet_by,
                        expr_for_auto_title=expr,
                        consumed_by_normalize=True)
                    return _fig_sf, _ax_sf, _stats_sf
                if group_by is not None:
                    _fig_sf, _ax_sf, _stats_sf = self._dispatch_normalize_grouped_render(
                        y_expr, x_expr,
                        normalize=normalize, normalize_layout=normalize_layout,
                        group_by=group_by,
                        selection=selection, sample=sample,
                        selection_vector=selection_vector,
                        weights_vector=weights_vector,
                        vector_compose=vector_compose,
                        bins=bins, x_range=range, error=error,
                        central=central, title=title,
                        xlabel=xlabel, ylabel=ylabel,
                        weights=weights, nan_policy=nan_policy,
                        **_passthrough,
                    )
                    self._maybe_attach_summary_fit(
                        _stats_sf, summary_fit,
                        group_by=group_by, facet_by=facet_by,
                        expr_for_auto_title=expr,
                        consumed_by_normalize=True)
                    return _fig_sf, _ax_sf, _stats_sf
                _fig_sf, _ax_sf, _stats_sf = self._dispatch_normalize_render(
                    y_expr, x_expr,
                    normalize=normalize,
                    normalize_layout=normalize_layout,
                    selection=selection,
                    sample=sample,
                    selection_vector=selection_vector,
                    weights_vector=weights_vector,
                    vector_compose=vector_compose,
                    bins=bins,
                    x_range=range,
                    error=error,
                    central=central,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    weights=weights,
                    nan_policy=nan_policy,
                    **_passthrough,
                )
                self._maybe_attach_summary_fit(
                    _stats_sf, summary_fit,
                    group_by=group_by, facet_by=facet_by,
                    expr_for_auto_title=expr,
                    consumed_by_normalize=True)
                return _fig_sf, _ax_sf, _stats_sf
            _fig_sf, _axes_sf, _stats_sf = self._draw_vector(
                y_expr, x_expr, self.profile,
                group_by=group_by, **vector_kwargs
            )
            self._maybe_attach_summary_fit(
                _stats_sf, summary_fit,
                group_by=group_by, facet_by=facet_by,
                expr_for_auto_title=expr)
            return _fig_sf, _axes_sf, _stats_sf
        
        if x_expr is None:
            raise ValueError(
                f"Profile plot requires 'y:x' format, got '{expr}'"
            )
        
        # Phase 13.13.DF: Resolve axes for same=True
        resolved_ax, is_new = self._resolve_axes(same, ax)
        if is_new:
            self._reset_color_cycle()
            ax = None
        else:
            ax = resolved_ax
        
        # Phase 13.13.DF: Inject color and label for same=True
        # Phase 13.16.DF: _suppress_color_cycle flag (P1-2)
        # Phase 13.36.DF: color is now explicit param of DFDraw.profile().
        # Two changes vs pre-13.36:
        # (1) Assign auto-color to local `color`, not `kwargs['color']`
        #     (would collide with explicit forwarding at line ~4262).
        # (2) Skip auto-color injection when group_by is active. Pre-13.36
        #     dropped auto-color silently in the grouped path; post-13.36
        #     it would flow through and force all groups uniform, defeating
        #     the per-group color cycle. The cycle already distinguishes
        #     groups; same=True overlay distinguishes via marker (cycle or
        #     user-passed). This preserves the effective pre-13.36 behavior.
        _suppress_color_cycle = kwargs.pop('_suppress_color_cycle', False)
        save_auto_title = auto_title
        if same:
            if (color is None and not _suppress_color_cycle
                    and group_by is None):
                color = self._get_next_color()
            if 'label' not in kwargs and group_by is None:
                kwargs['label'] = self._auto_label(y_expr, x_expr)
            # Suppress title — handled in _handle_same_post
            if ax is not None and ax.get_title():
                title = None
                auto_title = False
        
        # Apply selection and sampling
        df = self._apply_selection(self.df, selection)
        df = self._apply_sampling(df, sample)

        # Phase 13.32.DF Sub-fix 3 (v1.2 §3.3): validate facet_by_bins/_quantiles
        # against facet_by + df at plot-method entry, BEFORE dispatch.
        self._validate_facet_by_binning(
            facet_by, facet_by_bins, facet_by_quantiles, df
        )
        
        # Evaluate expressions if needed
        if y_expr not in df.columns:
            df = df.assign(**{y_expr: self._eval_column(y_expr)})
        if x_expr not in df.columns:
            df = df.assign(**{x_expr: self._eval_column(x_expr)})
        
        # Apply duck-typed label lookup if not explicitly set
        if xlabel is None:
            duck_label = self._get_label(x_expr)
            if duck_label is not None:
                xlabel = duck_label
        if ylabel is None:
            duck_label = self._get_label(y_expr)
            if duck_label is not None:
                ylabel = duck_label
        
        # Phase 13.27.DF (Phase D): Facet routing through channel framework.
        # Backward compat: facet=True is normalized to facet_by='group_by' (AD-67).
        # Mutual exclusion: facet_by + same=True is invalid (architect rule §5.4).
        _effective_facet_by = facet_by
        if _effective_facet_by is None and facet and group_by is not None:
            _effective_facet_by = 'group_by'

        if _effective_facet_by is not None and same:
            raise ValueError(
                "facet_by and same=True are mutually exclusive — facet creates "
                "a new figure; same=True overlays on existing axes"
            )

        # Facet mode dispatch
        if _effective_facet_by is not None:
            fig, axes, stats_dict = self._dispatch_faceted_render(
                df=df, x_expr=x_expr, y_expr=y_expr,
                facet_by=_effective_facet_by, plot_kind='profile',
                # Forwarded to per-subplot draw_profile call
                bins=bins, x_range=range, error=error,
                stats=stats, title=title, xlabel=xlabel, ylabel=ylabel,
                group_by=group_by, top_k=top_k, ncols=ncols,
                sharex=sharex, sharey=sharey,
                # Phase 13.12.DF
                return_data=return_data, min_entries=min_entries,
                group_by_bins=group_by_bins, group_by_quantiles=group_by_quantiles,
                sort_groups=sort_groups, weights=weights,
                # Phase 13.12.DF v1.2: auto-title
                auto_title=auto_title, selection=selection,
                # Phase 13.18.DF: robust statistics
                stat_fields=stat_fields,
                # Phase 13.25.DF: quantile rendering
                quantiles=quantiles, central=central, quantile_mode=quantile_mode,
                # Phase 13.26.DF: channel quantile_style
                quantile_style=quantile_style,
                # Phase 13.28.DF: nan_policy
                nan_policy=nan_policy,
                # Phase 13.32.DF Sub-fix 3 (AD-79): symmetric binning on facet axis
                facet_by_bins=facet_by_bins,
                facet_by_quantiles=facet_by_quantiles,
                # Phase 13.41.DF v1.6 + FIX1: forward N-D axis-sharing controls
                share_x=share_x,
                share_y=share_y,
                share_across_figures=share_across_figures,
                # Phase 13.42.DF: inline fits
                fit=fit,
                fit_textbox_kwargs=fit_textbox_kwargs,
                **kwargs
            )
        else:
            # Standard mode (single plot or overlay)
            fig, ax, stats_dict = draw_profile(
                df, x_expr, y_expr,
                ax=ax, bins=bins, x_range=range, error=error,
                stats=stats, title=title, xlabel=xlabel, ylabel=ylabel,
                group_by=group_by, top_k=top_k,
                # Phase 13.12.DF: pass new parameters
                return_data=return_data, min_entries=min_entries,
                group_by_bins=group_by_bins, group_by_quantiles=group_by_quantiles,
                sort_groups=sort_groups, weights=weights,
                # Phase 13.12.DF v1.2: auto-title
                auto_title=auto_title, selection=selection,
                # Phase 13.18.DF: robust statistics
                stat_fields=stat_fields,
                # Phase 13.25.DF: quantile rendering
                quantiles=quantiles, central=central, quantile_mode=quantile_mode,
                # Phase 13.28.DF: NaN/inf filter policy
                nan_policy=nan_policy,
                # Phase 13.36.DF: user style overrides — all three forwarded
                # from DFDraw.profile() explicit signature. draw_profile() has
                # all three as explicit params at lines 162-164. Without these,
                # consumed by DFDraw.profile() signature and lost.
                color=color, marker=marker, markersize=markersize,
                # Phase 13.37.DF: per-group linestyle cycle mode flag.
                linestyle_cycle=linestyle_cycle,
                # Phase 13.39.DF: time-axis formatting
                time_format=time_format,
                # Phase 13.42.DF: inline fits
                fit=fit,
                fit_textbox_kwargs=fit_textbox_kwargs,
                **kwargs
            )
            axes = ax
            
            # Phase 13.13.DF: Post-draw handling for same=True
            self._handle_same_post(
                ax, same, save_auto_title, selection,
                y_name=y_expr, x_name=x_expr,
                group_by=group_by, weights=weights
            )
        
        # Save if requested
        if save:
            fig.savefig(save, dpi=get_style_value("figure.dpi", 100),
                       bbox_inches="tight")
        
        self._maybe_attach_summary_fit(
            stats_dict, summary_fit,
            group_by=group_by, facet_by=facet_by,
            expr_for_auto_title=expr)
        return fig, axes, stats_dict
    
    def hist2d(
        self,
        expr: str,
        selection: Optional[Union[str, np.ndarray, callable]] = None,
        sample: Optional[int] = None,
        bins: Optional[Union[int, List[int], Tuple[int, int]]] = None,
        range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        norm: Optional[str] = None,
        stats: Optional[Union[bool, List[str]]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ax=None,
        save: Optional[str] = None,
        group_by: Optional[str] = None,
        facet: bool = False,
        ncols: Optional[int] = None,
        sharex: bool = True,
        sharey: bool = True,
        top_k: Optional[int] = None,
        cmap: Optional[str] = None,
        colorbar: bool = True,
        clabel: Optional[str] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        # Phase 13.12.DF v1.2: Auto-title
        auto_title: Union[bool, str] = False,
        # Phase 13.13.DF: same=True (AD-15)
        same: bool = False,
        # Phase 13.18.DF: Robust statistics
        stat_fields: Optional[Union[str, List[str]]] = None,
        # Phase 13.28.DF: NaN/inf filter policy (AD-70)
        nan_policy: str = "filter",
        # Phase 13.32.DF Sub-fix 3 (AD-79): extend AD-78 facet_by column-mode to hist2d
        facet_by: Optional[Union[str, List[str]]] = None,
        facet_by_bins: Optional[Union[int, List[Optional[int]]]] = None,
        facet_by_quantiles: Optional[Union[int, List]] = None,
        # Phase 13.41.DF v1.6 + FIX1: N-D faceting axis-sharing controls
        share_x: str = 'all',
        share_y: str = 'all',
        share_across_figures: bool = True,
        **kwargs
    ) -> DrawResult:
        """
        Draw 2D histogram (density plot).
        
        Parameters
        ----------
        expr : str
            Expression in "y:x" format.
        selection : optional
            Data selection/cut.
        sample : int, optional
            Max points to use.
        bins : int, list, or tuple, optional
            Number of bins ([nx, ny] or single int).
        range : tuple, optional
            ((xmin, xmax), (ymin, ymax)) range.
        norm : str, optional
            Normalization: "count", "density", "log".
        stats : bool or list, optional
            Show stats box.
        title : str, optional
            Plot title.
        xlabel, ylabel : str, optional
            Axis labels.
        ax : Axes, optional
            Existing axes.
        save : str, optional
            Save path.
        group_by : str, optional
            Column for grouping.
        facet : bool, default False
            Create subplots for groups (requires group_by).
        ncols : int, optional
            Number of columns for facet grid.
        sharex : bool, default True
            Share x-axis in facet mode.
        sharey : bool, default True
            Share y-axis in facet mode.
        top_k : int, optional
            Show only top K groups.
        cmap : str, optional
            Colormap name.
        colorbar : bool, default True
            Show colorbar.
        clabel : str, optional
            Colorbar label.
        vmin, vmax : float, optional
            Color scale limits.
        same : bool, default False
            If True, overlay on last axes. Phase 13.13.DF (AD-15).
            Useful for overlaying profile on hist2d.
        **kwargs
            Additional arguments.
        
        Returns
        -------
        tuple
            (fig, ax, stats_dict)
        """
        from .plots.histogram import draw_hist2d
        
        # Parse expression
        y_expr, x_expr = self._parse_expr(expr)
        
        # Phase 13.27.DF Commit 2 (v1.2 §4.1.1, §5.5): hist2d does not support
        # per-curve selection/weights vectors (2D density is single-surface).
        # The _HIST2D_FORWARDED_NAMES tuple already excludes them, but **kwargs
        # catch-all permits direct calls — surface a clear TypeError before
        # the kwargs leak through matplotlib's QuadMesh.set().
        for _bad in ('selection_vector', 'weights_vector'):
            if _bad in kwargs:
                raise TypeError(
                    f"hist2d() got an unexpected keyword argument {_bad!r} "
                    f"(Phase 13.27 Commit 2: hist2d does not support per-curve "
                    f"selection/weights vectors; use hist or profile instead)"
                )

        # Phase 13.16.DF: vector not supported in hist2d (2D density is single-surface)
        if isinstance(y_expr, list):
            raise ValueError(
                "Vector expressions are not supported by hist2d(). "
                "2D density plots are fundamentally single-surface. "
                "Use profile(), hist(), or scatter() for vector overlays."
            )
        
        if x_expr is None:
            raise ValueError(
                f"hist2d requires 'y:x' format, got '{expr}'"
            )
        
        # Phase 13.13.DF: Resolve axes for same=True
        resolved_ax, is_new = self._resolve_axes(same, ax)
        if is_new:
            self._reset_color_cycle()
            ax = None
        else:
            ax = resolved_ax
        
        # Apply selection and sampling
        df = self._apply_selection(self.df, selection)
        df = self._apply_sampling(df, sample)

        # Phase 13.32.DF Sub-fix 3: validate facet_by_bins/_quantiles at entry
        self._validate_facet_by_binning(
            facet_by, facet_by_bins, facet_by_quantiles, df
        )
        
        # Evaluate expressions if needed
        if y_expr not in df.columns:
            df = df.assign(**{y_expr: self._eval_column(y_expr)})
        if x_expr not in df.columns:
            df = df.assign(**{x_expr: self._eval_column(x_expr)})
        
        # Apply duck-typed label lookup if not explicitly set
        if xlabel is None:
            duck_label = self._get_label(x_expr)
            if duck_label is not None:
                xlabel = duck_label
        if ylabel is None:
            duck_label = self._get_label(y_expr)
            if duck_label is not None:
                ylabel = duck_label

        # Phase 13.32.DF Sub-fix 3 (AD-79): facet_by-driven dispatch via the
        # channel framework. Takes precedence over legacy facet=True path.
        _effective_facet_by = facet_by
        if not _effective_facet_by and facet and group_by is not None:
            _effective_facet_by = 'group_by'  # AD-67 backward-compat
        if _effective_facet_by is not None:
            fig, axes, stats_dict = self._dispatch_faceted_render(
                df=df, x_expr=x_expr, y_expr=y_expr,
                facet_by=_effective_facet_by, plot_kind='hist2d',
                bins=bins, range=range, norm=norm,
                stats=stats, title=title, xlabel=xlabel, ylabel=ylabel,
                group_by=group_by, top_k=top_k, ncols=ncols,
                sharex=sharex, sharey=sharey,
                cmap=cmap, colorbar=colorbar, clabel=clabel,
                vmin=vmin, vmax=vmax,
                auto_title=auto_title, selection=selection,
                stat_fields=stat_fields,
                nan_policy=nan_policy,
                facet_by_bins=facet_by_bins,
                facet_by_quantiles=facet_by_quantiles,
                # Phase 13.41.DF v1.6 + FIX1: forward N-D axis-sharing controls
                share_x=share_x,
                share_y=share_y,
                share_across_figures=share_across_figures,
                **kwargs
            )
        # Facet mode (legacy path, same=True ignored in facet mode)
        elif facet and group_by is not None:
            from .facet import facet_hist2d
            fig, axes, stats_dict = facet_hist2d(
                df, x_expr, y_expr, group_by,
                top_k=top_k, ncols=ncols, sharex=sharex, sharey=sharey,
                suptitle=title, bins=bins, range=range, norm=norm,
                stats=stats, xlabel=xlabel, ylabel=ylabel,
                cmap=cmap, colorbar=colorbar, clabel=clabel,
                vmin=vmin, vmax=vmax, **kwargs
            )
        else:
            # Standard mode
            fig, ax, stats_dict = draw_hist2d(
                df, x_expr, y_expr,
                ax=ax, bins=bins, range=range, norm=norm,
                stats=stats, title=title, xlabel=xlabel, ylabel=ylabel,
                cmap=cmap, colorbar=colorbar, clabel=clabel,
                vmin=vmin, vmax=vmax,
                # Phase 13.12.DF v1.2: auto-title
                auto_title=auto_title, selection=selection,
                # Phase 13.18.DF: robust statistics
                stat_fields=stat_fields,
                # Phase 13.28.DF: NaN/inf filter policy
                nan_policy=nan_policy,
                **kwargs
            )
            axes = ax
            
            # Phase 13.13.DF: Store axes for same=True chain
            self._last_ax = ax
        
        # Save if requested
        if save:
            fig.savefig(save, dpi=get_style_value("figure.dpi", 100),
                       bbox_inches="tight")
        
        return fig, axes, stats_dict
    
    def hexbin(
        self,
        expr: str,
        selection: Optional[Union[str, np.ndarray, callable]] = None,
        sample: Optional[int] = None,
        gridsize: int = 50,
        extent: Optional[Tuple[float, float, float, float]] = None,
        norm: Optional[str] = None,
        stats: Optional[Union[bool, List[str]]] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        ax=None,
        save: Optional[str] = None,
        group_by: Optional[str] = None,
        facet: bool = False,
        ncols: Optional[int] = None,
        sharex: bool = True,
        sharey: bool = True,
        top_k: Optional[int] = None,
        cmap: Optional[str] = None,
        colorbar: bool = True,
        clabel: Optional[str] = None,
        mincnt: Optional[int] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        # Phase 13.12.DF v1.2: Auto-title
        auto_title: Union[bool, str] = False,
        # Phase 13.13.DF: same=True (AD-15)
        same: bool = False,
        **kwargs
    ) -> DrawResult:
        """
        Draw hexbin plot (2D density with hexagonal bins).
        
        Better than hist2d for large datasets - hexagons tile more 
        efficiently and avoid alignment artifacts.
        
        Parameters
        ----------
        expr : str
            Expression in "y:x" format.
        selection : optional
            Data selection/cut.
        sample : int, optional
            Max points to use.
        gridsize : int, default 50
            Number of hexagons in x-direction.
        extent : tuple, optional
            (xmin, xmax, ymin, ymax) extent.
        norm : str, optional
            Normalization: None (count), "log".
        stats : bool or list, optional
            Show stats box.
        title : str, optional
            Plot title.
        xlabel, ylabel : str, optional
            Axis labels.
        ax : Axes, optional
            Existing axes.
        save : str, optional
            Save path.
        group_by : str, optional
            Column for grouping.
        facet : bool, default False
            Create subplots for groups (requires group_by).
        ncols : int, optional
            Number of columns for facet grid.
        sharex : bool, default True
            Share x-axis in facet mode.
        sharey : bool, default True
            Share y-axis in facet mode.
        top_k : int, optional
            Show only top K groups.
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
        same : bool, default False
            If True, overlay on last axes. Phase 13.13.DF (AD-15).
            Useful for overlaying profile on hexbin.
        **kwargs
            Additional arguments to plt.hexbin().
        
        Returns
        -------
        tuple
            (fig, ax, stats_dict)
        """
        from .plots.histogram import draw_hexbin
        
        # Parse expression
        y_expr, x_expr = self._parse_expr(expr)
        
        # Phase 13.16.DF: vector not supported in hexbin (2D density is single-surface)
        if isinstance(y_expr, list):
            raise ValueError(
                "Vector expressions are not supported by hexbin(). "
                "2D density plots are fundamentally single-surface. "
                "Use profile(), hist(), or scatter() for vector overlays."
            )
        
        if x_expr is None:
            raise ValueError(
                f"hexbin requires 'y:x' format, got '{expr}'"
            )
        
        # Phase 13.13.DF: Resolve axes for same=True
        resolved_ax, is_new = self._resolve_axes(same, ax)
        if is_new:
            self._reset_color_cycle()
            ax = None
        else:
            ax = resolved_ax
        
        # Apply selection and sampling
        df = self._apply_selection(self.df, selection)
        df = self._apply_sampling(df, sample)
        
        # Evaluate expressions if needed
        if y_expr not in df.columns:
            df = df.assign(**{y_expr: self._eval_column(y_expr)})
        if x_expr not in df.columns:
            df = df.assign(**{x_expr: self._eval_column(x_expr)})
        
        # Apply duck-typed label lookup if not explicitly set
        if xlabel is None:
            duck_label = self._get_label(x_expr)
            if duck_label is not None:
                xlabel = duck_label
        if ylabel is None:
            duck_label = self._get_label(y_expr)
            if duck_label is not None:
                ylabel = duck_label
        
        # Facet mode (same=True ignored in facet mode)
        if facet and group_by is not None:
            from .facet import facet_hexbin
            fig, axes, stats_dict = facet_hexbin(
                df, x_expr, y_expr, group_by,
                top_k=top_k, ncols=ncols, sharex=sharex, sharey=sharey,
                suptitle=title, gridsize=gridsize, extent=extent, norm=norm,
                stats=stats, xlabel=xlabel, ylabel=ylabel,
                cmap=cmap, colorbar=colorbar, clabel=clabel,
                mincnt=mincnt, vmin=vmin, vmax=vmax, **kwargs
            )
        else:
            # Standard mode
            fig, ax, stats_dict = draw_hexbin(
                df, x_expr, y_expr,
                ax=ax, gridsize=gridsize, extent=extent, norm=norm,
                stats=stats, title=title, xlabel=xlabel, ylabel=ylabel,
                cmap=cmap, colorbar=colorbar, clabel=clabel,
                mincnt=mincnt, vmin=vmin, vmax=vmax,
                # Phase 13.12.DF v1.2: auto-title
                auto_title=auto_title, selection=selection,
                **kwargs
            )
            axes = ax
            
            # Phase 13.13.DF: Store axes for same=True chain
            self._last_ax = ax
        
        # Save if requested
        if save:
            fig.savefig(save, dpi=get_style_value("figure.dpi", 100),
                       bbox_inches="tight")
        
        return fig, axes, stats_dict
    
    # =========================================================================
    # Statistics Method
    # =========================================================================
    
    def stats(
        self,
        expr: str,
        selection: Optional[Union[str, np.ndarray, callable]] = None,
        group_by: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute statistics without plotting.
        
        Parameters
        ----------
        expr : str
            Expression in "y:x" or "x" format.
        selection : optional
            Data selection.
        group_by : str, optional
            Compute stats per group.
        
        Returns
        -------
        DataFrame
            Statistics table.
        """
        from .stats import compute_stats
        
        df = self._apply_selection(self.df, selection)
        y_expr, x_expr = self._parse_expr(expr)
        
        # Phase 13.16.DF: vector support - per-pair stats
        if isinstance(y_expr, list):
            stats_list = []
            for y, x in zip(y_expr, x_expr):
                stats_list.append(compute_stats(df, y, x, group_by=group_by))
            return stats_list
        
        return compute_stats(df, y_expr, x_expr, group_by=group_by)
    
    # =========================================================================
    # Annotation Methods (Phase 12.4b5)
    # =========================================================================
    
    def add_statistics_box(
        self, 
        ax, 
        values, 
        position: str = 'upper right',
        expected_mean: Optional[float] = None, 
        expected_std: Optional[float] = None,
        precision: int = 3, 
        fontsize: int = 8, 
        alpha: float = 0.5
    ):
        """
        Add statistics annotation box to axis.
        
        Generic method for any histogram.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axis to annotate.
        values : array-like
            Array of values for statistics computation.
        position : str, default 'upper right'
            Box position: 'upper right', 'upper left', 'lower right', 'lower left'.
        expected_mean : float, optional
            If provided, show Δμ = mean - expected.
        expected_std : float, optional
            If provided, show Δσ = std - expected.
        precision : int, default 3
            Decimal places for values.
        fontsize : int, default 8
            Font size for text.
        alpha : float, default 0.5
            Background transparency.
            
        Returns
        -------
        matplotlib.text.Text
            The created text artist.
        """
        values = np.asarray(values)
        values = values[~np.isnan(values)]
        
        if len(values) == 0:
            return None
        
        mean = np.mean(values)
        std = np.std(values)
        n = len(values)
        
        # Build annotation text
        lines = [
            f"n = {n:,}",
            f"μ = {mean:.{precision}f}",
            f"σ = {std:.{precision}f}",
        ]
        
        if expected_mean is not None:
            delta_mean = mean - expected_mean
            lines.append(f"Δμ = {delta_mean:+.{precision}f}")
        
        if expected_std is not None:
            delta_std = std - expected_std
            lines.append(f"Δσ = {delta_std:+.{precision}f}")
        
        text = "\n".join(lines)
        
        # Position mapping
        positions = {
            'upper right': (0.95, 0.95, 'top', 'right'),
            'upper left': (0.05, 0.95, 'top', 'left'),
            'lower right': (0.95, 0.05, 'bottom', 'right'),
            'lower left': (0.05, 0.05, 'bottom', 'left'),
        }
        x, y, va, ha = positions.get(position, positions['upper right'])
        
        text_artist = ax.text(
            x, y, text, transform=ax.transAxes,
            verticalalignment=va, horizontalalignment=ha,
            fontsize=fontsize, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=alpha)
        )
        
        return text_artist
    
    def add_reference_overlay(
        self, 
        ax, 
        func: str = 'gaussian', 
        mu: float = 0, 
        sigma: float = 1,
        label: Optional[str] = None, 
        color: str = 'red', 
        linestyle: str = '--',
        linewidth: float = 1.5, 
        show_legend: bool = True, 
        n_points: int = 100
    ):
        """
        Add reference function overlay scaled to histogram.
        
        Generic method for any histogram.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axis containing a histogram.
        func : str or callable, default 'gaussian'
            'gaussian' or callable f(x) -> y.
        mu : float, default 0
            Mean parameter for gaussian.
        sigma : float, default 1
            Std parameter for gaussian.
        label : str, optional
            Legend label (default: 'N(μ,σ)' for gaussian).
        color : str, default 'red'
            Line color.
        linestyle : str, default '--'
            Line style.
        linewidth : float, default 1.5
            Line width.
        show_legend : bool, default True
            Whether to add legend.
        n_points : int, default 100
            Number of points for curve.
            
        Returns
        -------
        matplotlib.lines.Line2D or None
            The created line artist, or None if no histogram found.
        """
        # Get histogram data for scaling
        patches = ax.patches
        if not patches:
            return None
        
        # Handle both Rectangle and Polygon patches
        heights = []
        widths = []
        lefts = []
        
        for p in patches:
            if hasattr(p, 'get_height') and hasattr(p, 'get_width'):
                # Rectangle patch (standard bar histogram)
                heights.append(p.get_height())
                widths.append(p.get_width())
                lefts.append(p.get_x())
            elif hasattr(p, 'get_xy'):
                # Polygon patch (stepfilled histogram from DFDraw)
                # Extract bounds from polygon vertices
                xy = p.get_xy()
                if len(xy) > 0:
                    x_coords = xy[:, 0]
                    y_coords = xy[:, 1]
                    lefts.append(np.min(x_coords))
                    # Approximate height and width from polygon bounds
                    heights.append(np.max(y_coords))
                    widths.append(np.max(x_coords) - np.min(x_coords))
        
        if not heights or not widths:
            return None
        
        # Calculate total area for scaling
        if len(heights) > 1:
            # Multiple patches - sum individual areas
            total_area = sum(h * w for h, w in zip(heights, widths))
            x_min = min(lefts)
            x_max = max(lefts) + widths[-1] if widths else max(lefts)
        else:
            # Single polygon - use axis limits
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            x_min, x_max = xlim
            # Estimate area from visible histogram
            total_area = heights[0] * (x_max - x_min) / 2  # Rough estimate
        
        # Extend range slightly
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, n_points)
        
        # Generate reference curve
        if func == 'gaussian':
            y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma)**2)
            if label is None:
                label = f'N({mu},{sigma})' if (mu != 0 or sigma != 1) else 'N(0,1)'
        elif callable(func):
            y = func(x)
            if label is None:
                label = 'Reference'
        else:
            raise ValueError(f"func must be 'gaussian' or callable, got {func}")
        
        # Scale to histogram
        y_scaled = y * total_area
        
        line_artist, = ax.plot(x, y_scaled, color=color, linestyle=linestyle,
                               linewidth=linewidth, label=label)
        
        if show_legend:
            ax.legend(loc='upper left', fontsize=8)
        
        return line_artist
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def draw_batch(
        self,
        specs: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]], str],
        save_dir: Optional[str] = None,
        defaults: Optional[Dict[str, Any]] = None,
        on_error: str = 'skip',
        verbose: Union[bool, int] = True,
        save_format: str = 'png',
        dpi: int = 150,
        close_figures: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Draw multiple figures from specification.
        
        Supports two formats:
        
        **Dict format (original):** Each key is a plot name, value is a spec dict.
        Each spec produces one independent figure.
        
        **List format (Phase 13.14.DF):** List of group dicts. Each group produces
        one figure with subplots. Group-level defaults cascade to individual plots
        (option hierarchy: kwargs < batch defaults < group defaults < plot spec).
        
        Parameters
        ----------
        specs : dict, list, or str
            Dict format: ``{'name': {'expr': 'y:x', ...}, ...}``
            List format: ``[{'name': 'fig1', 'defaults': {...}, 'plots': [...]}, ...]``
            String: path to YAML/JSON file (dict format only).
        save_dir : str, optional
            Directory to save figures. Created if doesn't exist.
        defaults : dict, optional
            Default parameters applied to all plots (overridden by per-plot specs).
        on_error : str, default 'skip'
            'skip': Continue on errors, collect in results['_errors']
            'raise': Stop on first error
        verbose : bool or int, default True
            Verbosity level:
            - ``False`` / ``0``: Silent — no output.
            - ``True`` / ``1``: Progress — group names, save paths, summary.
            - ``2``: Debug — also prints merged parameters per plot.
        save_format : str, default 'png'
            Output format: 'png', 'pdf', 'svg', etc.
        dpi : int, default 150
            Figure resolution for saving.
        close_figures : bool, default True
            Close figures after saving (recommended for large batches).
        **kwargs
            Additional parameters passed to all plot methods.
        
        Returns
        -------
        dict
            Results dictionary with plot results, errors, and summary.
            
            Dict format: ``{'name': {'stats': dict, 'fig': Figure, 'ax': Axes, 'path': str}, ...}``
            List format: ``{'name': {'fig': Figure, 'axes': list, 'stats': list, 'path': str}, ...}``
        
        Examples
        --------
        Dict format (original):
        
        >>> specs = {
        ...     'hist_x': {'expr': 'x', 'bins': 50},
        ...     'profile_yx': {'expr': 'y:x', 'type': 'profile', 'bins': 100},
        ... }
        >>> results = plotter.draw_batch(specs, save_dir='plots/')
        
        List format (Phase 13.14.DF — defaults hierarchy + subplot grid):
        
        >>> specs = [{
        ...     'name': 'residuals_qa',
        ...     'suptitle': 'ITS-TPC Residuals QA',
        ...     'ncols': 2,
        ...     'figsize': (16, 12),
        ...     'savefig': 'residuals_qa.png',
        ...     'defaults': {
        ...         'type': 'profile', 'bins': 152, 'min_entries': 250,
        ...         'selection': 'group_count>100',
        ...     },
        ...     'plots': [
        ...         {'expr': 'dystd:row', 'group_by': 'mP4_bin'},
        ...         {'expr': 'Side.dy:xM', 'group_by': 'mP4', 'group_by_quantiles': 10},
        ...         {'expr': 'Side.dy:row'},
        ...     ]
        ... }]
        >>> results = plotter.draw_batch(specs, dpi=150)
        """
        import os
        import matplotlib.pyplot as plt
        
        # Phase 13.14.DF: Detect list format → group mode
        if isinstance(specs, list):
            return self._draw_batch_groups(
                specs, save_dir=save_dir, defaults=defaults,
                on_error=on_error, verbose=verbose, save_format=save_format,
                dpi=dpi, close_figures=close_figures, **kwargs
            )
        
        # Load from file if string path provided
        if isinstance(specs, str):
            specs = self._load_specs_file(specs)
        
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        results = {}
        errors = {}
        n = len(specs)
        
        for i, (name, spec) in enumerate(specs.items()):
            if verbose:
                print(f"[{i+1}/{n}] {name}", end="", flush=True)
            
            try:
                # Merge: kwargs < defaults < spec
                merged = {**kwargs, **(defaults or {}), **spec}
                
                # Extract required 'expr'
                if 'expr' not in merged:
                    raise ValueError(f"Missing 'expr' in spec for '{name}'")
                expr = merged.pop('expr')
                
                # Determine plot type
                plot_type = merged.pop('type', None)
                if plot_type is None:
                    plot_type = 'hist' if ':' not in expr else 'scatter'
                
                # Validate plot type
                valid_types = ('hist', 'scatter', 'profile', 'hist2d', 'hexbin')
                if plot_type not in valid_types:
                    raise ValueError(f"Invalid type '{plot_type}'. Must be one of {valid_types}")
                
                # Call appropriate method
                method = getattr(self, plot_type)
                fig, ax, stats = method(expr, **merged)
                
                # Build result entry
                result_entry = {'stats': stats, 'fig': fig, 'ax': ax, 'path': None}
                
                # Save if directory specified
                if save_dir:
                    path = os.path.join(save_dir, f"{name}.{save_format}")
                    fig.savefig(path, dpi=dpi, bbox_inches='tight')
                    result_entry['path'] = path
                    
                    if close_figures:
                        plt.close(fig)
                        result_entry['fig'] = None
                        result_entry['ax'] = None
                    
                    if verbose:
                        print(f" → {path}")
                elif verbose:
                    print()
                
                results[name] = result_entry
                
            except Exception as e:
                errors[name] = str(e)
                if verbose:
                    print(f" ✗ {e}")
                if on_error == 'raise':
                    raise
        
        # Summary
        results['_errors'] = errors
        results['_summary'] = {
            'total': n,
            'success': n - len(errors),
            'failed': len(errors)
        }
        
        if verbose:
            print(f"Completed: {results['_summary']['success']}/{n} ({len(errors)} errors)")
        
        return results
    
    # =========================================================================
    # Phase 13.14.DF: Group-based Batch Processing
    # =========================================================================
    
    # Group-level keys that are not draw parameters — stripped before dispatch
    _GROUP_KEYS = frozenset({
        'name', 'defaults', 'ncols', 'layout', 'figsize',
        'suptitle', 'savefig', 'sharex', 'sharey', 'plots',
    })
    
    def _draw_batch_groups(
        self,
        groups: List[Dict[str, Any]],
        save_dir: Optional[str] = None,
        defaults: Optional[Dict[str, Any]] = None,
        on_error: str = 'skip',
        verbose: Union[bool, int] = True,
        save_format: str = 'png',
        dpi: int = 150,
        close_figures: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Draw batch from list-of-groups format.
        
        Each group produces one figure with subplots. Group-level defaults
        cascade to individual plots via option hierarchy:
        ``kwargs < batch defaults < group defaults < plot spec``
        
        Phase 13.14.DF v1.0.
        
        Parameters
        ----------
        groups : list of dict
            Each dict has 'name' (required), 'plots' (required),
            and optional 'defaults', 'ncols', 'layout', 'figsize',
            'suptitle', 'savefig', 'sharex', 'sharey'.
        save_dir : str, optional
            Directory for saving (used when 'savefig' not in group).
        defaults : dict, optional
            Batch-level defaults (below group defaults in hierarchy).
        on_error : str, default 'skip'
            'skip' or 'raise'.
        verbose : bool or int, default True
            Verbosity level: False/0=silent, True/1=progress, 2=debug.
        save_format : str, default 'png'
            Format when using save_dir.
        dpi : int, default 150
            Save resolution.
        close_figures : bool, default True
            Close figures after saving.
        **kwargs
            Lowest-priority defaults passed to all draw methods.
        
        Returns
        -------
        dict
            ``{'group_name': {'fig': Figure, 'axes': list, 'stats': list, 'path': str}, ...}``
        """
        import os
        import math
        import matplotlib.pyplot as plt
        
        results = {}
        errors = {}
        total_groups = len(groups)
        
        for g_idx, group in enumerate(groups):
            name = group.get('name', f'group_{g_idx}')
            group_defaults = group.get('defaults', {})
            plots = group.get('plots', [])
            suptitle = group.get('suptitle', None)
            savefig = group.get('savefig', None)
            sharex = group.get('sharex', False)
            sharey = group.get('sharey', False)
            
            if verbose:
                print(f"[{g_idx+1}/{total_groups}] {name} ({len(plots)} plots)",
                      end="", flush=True)
            
            try:
                if not plots:
                    raise ValueError(f"Group '{name}' has no plots")
                
                # Count subplots (same=True doesn't consume a new subplot)
                n_subplots = sum(1 for p in plots if not p.get('same', False))
                if n_subplots == 0:
                    raise ValueError(f"Group '{name}': all plots have same=True")
                
                # Layout resolution: layout > ncols > auto (AD-29)
                if 'layout' in group:
                    nrows, ncols = group['layout']
                elif 'ncols' in group:
                    ncols = group['ncols']
                    nrows = math.ceil(n_subplots / ncols)
                else:
                    ncols = min(3, n_subplots)
                    nrows = math.ceil(n_subplots / ncols) if ncols > 0 else 1
                
                # Figure size: explicit > auto-scaled
                if 'figsize' in group:
                    figsize = group['figsize']
                else:
                    base = get_style_value("figure.figsize", (8, 6))
                    figsize = (base[0] * ncols / 1.5, base[1] * nrows / 1.5)
                
                # Create figure — squeeze=False ensures axes is always 2D array (P1-2)
                # constrained_layout handles suptitle + subplot title spacing automatically
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=figsize,
                    squeeze=False, sharex=sharex, sharey=sharey,
                    constrained_layout=True
                )
                
                # Hide empty subplots (P1-3)
                for j in range(n_subplots, nrows * ncols):
                    axes.flat[j].set_visible(False)
                
                subplot_idx = -1
                group_stats = []
                
                for p_idx, plot_spec in enumerate(plots):
                    # Merge: kwargs < batch defaults < group defaults < plot spec
                    merged = {**kwargs, **(defaults or {}), **group_defaults, **plot_spec}
                    
                    expr = merged.pop('expr')
                    is_same = merged.pop('same', False)
                    
                    # Verbose>=2: show merged params per plot (debug mode)
                    if verbose and int(verbose) >= 2:
                        print(f"\n  plot[{p_idx}] expr='{expr}' merged:")
                        for k, v in sorted(merged.items()):
                            if k != 'ax':
                                print(f"    {k}: {v!r}")
                    
                    # Guard: same=True on first plot (P1-1)
                    if is_same and subplot_idx < 0:
                        raise ValueError(
                            f"same=True on first plot in group '{name}' "
                            "has no previous subplot"
                        )
                    
                    if not is_same:
                        subplot_idx += 1
                    merged['ax'] = axes.flat[subplot_idx]
                    
                    # Remove group-level keys that aren't draw parameters
                    for key in self._GROUP_KEYS:
                        merged.pop(key, None)
                    
                    # Dispatch
                    plot_type = merged.pop('type', None)
                    if plot_type is None:
                        plot_type = 'hist' if ':' not in expr else 'scatter'
                    
                    valid_types = ('hist', 'scatter', 'profile', 'hist2d', 'hexbin')
                    if plot_type not in valid_types:
                        raise ValueError(
                            f"Invalid type '{plot_type}' in group '{name}' "
                            f"plot {p_idx}. Must be one of {valid_types}"
                        )
                    
                    method = getattr(self, plot_type)
                    # Suppress per-subplot tight_layout — constrained_layout
                    # handles the full figure layout automatically.
                    merged['_suppress_layout'] = True
                    _, _, stats = method(expr, **merged)
                    group_stats.append(stats)
                
                # Suptitle — constrained_layout automatically reserves space
                if suptitle:
                    fig.suptitle(
                        suptitle,
                        fontsize=get_style_value("axes.titlesize", 14) + 2
                    )
                
                # No tight_layout() call — constrained_layout=True (set at
                # figure creation) handles all spacing including suptitle,
                # subplot titles, axis labels, and legends automatically.
                
                # Save
                save_path = None
                if savefig:
                    save_path = savefig
                elif save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(save_dir, f"{name}.{save_format}")
                
                if save_path:
                    fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
                    if verbose:
                        print(f" → {save_path}")
                elif verbose:
                    print()
                
                results[name] = {
                    'fig': fig,
                    'axes': list(axes.flat[:n_subplots]),
                    'stats': group_stats,
                    'path': save_path,
                }
                
                if close_figures and save_path:
                    plt.close(fig)
                    results[name]['fig'] = None
                
            except Exception as e:
                errors[name] = str(e)
                if verbose:
                    print(f" ✗ {e}")
                if on_error == 'raise':
                    raise
        
        results['_errors'] = errors
        results['_summary'] = {
            'total': total_groups,
            'success': total_groups - len(errors),
            'failed': len(errors)
        }
        if verbose:
            print(f"Completed: {results['_summary']['success']}/{total_groups} "
                  f"({len(errors)} errors)")
        
        return results
    
    def _load_specs_file(self, path: str) -> Dict[str, Dict[str, Any]]:
        """Load specs from YAML or JSON file."""
        import json
        
        if path.endswith('.json'):
            with open(path, 'r') as f:
                data = json.load(f)
        elif path.endswith(('.yaml', '.yml')):
            try:
                import yaml
                with open(path, 'r') as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML required for YAML files: pip install pyyaml")
        else:
            raise ValueError(f"Unsupported file format: {path}. Use .json, .yaml, or .yml")
        
        # Handle 'plots' key if present (YAML style)
        if 'plots' in data:
            return data['plots']
        return data


# =============================================================================
# Phase 13.16.DF FIX1 (R6): Class-load validation of forwarded-name tuples.
# 
# Runs at module import. Verifies every entry in each _*_FORWARDED_NAMES tuple
# is a real parameter of the corresponding signature. If a future phase renames
# a parameter and forgets to update the tuple, the import fails loudly here
# instead of producing silent kwarg drops at user-trigger time.
# =============================================================================

def _validate_forwarded_names():
    """Validate at module-import that all _*_FORWARDED_NAMES and
    _*_COLUMN_REFERENCES entries match their target method signatures."""
    pairs = [
        (DFDraw._PROFILE_FORWARDED_NAMES, DFDraw.profile, 'profile'),
        (DFDraw._HIST_FORWARDED_NAMES,    DFDraw.hist,    'hist'),
        (DFDraw._SCATTER_FORWARDED_NAMES, DFDraw.scatter, 'scatter'),
        (DFDraw._DRAW_FORWARDED_NAMES,    DFDraw.draw,    'draw'),
        # Phase 13.32.DF Sub-fix 3: hist2d's new FORWARDED_NAMES tuple
        (DFDraw._HIST2D_FORWARDED_NAMES,  DFDraw.hist2d,  'hist2d'),
        # Phase 13.30.DF v1.0 — Class-2 column-reference tuples.
        # Same validation: every entry must be a real parameter of the target.
        # (Restored in Phase 13.31 after initial Phase 13.31 patch clobbered.)
        (DFDraw._PROFILE_COLUMN_REFERENCES, DFDraw.profile, 'profile (col-refs)'),
        (DFDraw._HIST_COLUMN_REFERENCES,    DFDraw.hist,    'hist (col-refs)'),
        (DFDraw._SCATTER_COLUMN_REFERENCES, DFDraw.scatter, 'scatter (col-refs)'),
        (DFDraw._DRAW_COLUMN_REFERENCES,    DFDraw.draw,    'draw (col-refs)'),
    ]
    errors = []
    for tup, method, name in pairs:
        try:
            sig_params = set(inspect.signature(method).parameters)
        except (TypeError, ValueError):
            continue  # introspection failed; skip silently
        missing = set(tup) - sig_params
        if missing:
            errors.append(
                f"_{name.upper().replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')} "
                f"contains non-signature parameters: {sorted(missing)}"
            )
    if errors:
        raise RuntimeError(
            "Phase 13.16.DF FIX1 R6 / Phase 13.30.DF Class-2 validation failed "
            "at module import:\n"
            + "\n".join("  - " + e for e in errors)
            + "\n\nUpdate the relevant tuple in DFDraw."
        )


_validate_forwarded_names()
