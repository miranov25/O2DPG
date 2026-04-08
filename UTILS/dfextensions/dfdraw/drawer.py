"""
DFDraw - Main drawing class with TTree::Draw-like interface.

Phase 13.1.DF: Added PyArrow Table input support.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from .style import get_style, get_style_value

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
        
        # Context-dependent default for vector_style
        if vector_style is None:
            vector_style = 'linestyle' if group_by is not None else 'color'
        
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
        fig, ax = None, None
        
        for i, (y, x) in enumerate(zip(y_list, x_list)):
            expr = f"{y}:{x}" if x is not None else y
            iter_kwargs = dict(kwargs)
            
            # Apply vector style channel for this iteration
            if vector_style == 'linestyle':
                iter_kwargs['linestyle'] = self._LINESTYLE_CYCLE[i % len(self._LINESTYLE_CYCLE)]
                # P1-2: suppress same=True color cycle so group_by colors are preserved
                if group_by is not None:
                    iter_kwargs['_suppress_color_cycle'] = True
            elif vector_style == 'marker':
                iter_kwargs['marker'] = self._MARKER_CYCLE[i % len(self._MARKER_CYCLE)]
                if group_by is not None:
                    iter_kwargs['_suppress_color_cycle'] = True
            # vector_style == 'color': rely on existing same=True color cycle
            
            if group_by is not None:
                iter_kwargs['group_by'] = group_by
            
            # First iteration uses outer same; subsequent always same=True
            iter_kwargs['same'] = outer_same if i == 0 else True
            
            fig, ax, stats = draw_method(expr, **iter_kwargs)
            stats_list.append(stats)
        
        # P1-6: secondary legend for vector + group_by
        if group_by is not None and ax is not None and vector_style != 'color':
            self._add_vector_legend(ax, y_list, x_list, vector_style)
        
        # P1-8: deterministic y-axis label for vector
        if ax is not None:
            self._set_vector_ylabel(ax, y_list, x_list)
        
        return fig, ax, stats_list
    
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
                ls = self._LINESTYLE_CYCLE[i % len(self._LINESTYLE_CYCLE)]
                proxies.append(Line2D([0], [0], color='black', linestyle=ls))
            elif vector_style == 'marker':
                mk = self._MARKER_CYCLE[i % len(self._MARKER_CYCLE)]
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
            Histogram normalization: "count", "density", "probability", "cumulative".
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
        # Handle figsize: create axes if not provided
        if figsize is not None and ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=figsize)
        
        # Parse expression to determine dimensionality
        y_expr, x_expr = self._parse_expr(expr)
        
        # Phase 13.16.DF: Vector expression dispatch
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
            
            # P0-5: Build kwargs for _draw_vector, dropping 'type'
            # (already consumed) and passing through all others.
            vector_kwargs = dict(kwargs)
            # Forward known named params from draw() signature
            if selection is not None:
                vector_kwargs.setdefault('selection', selection)
            if color is not None:
                vector_kwargs.setdefault('color', color)
            if size is not None:
                vector_kwargs.setdefault('size', size)
            if marker is not None:
                vector_kwargs.setdefault('marker', marker)
            if bins is not None:
                vector_kwargs.setdefault('bins', bins)
            if stats is not None:
                vector_kwargs.setdefault('stats', stats)
            if norm is not None:
                vector_kwargs.setdefault('norm', norm)
            if title is not None:
                vector_kwargs.setdefault('title', title)
            if ax is not None:
                vector_kwargs.setdefault('ax', ax)
            if sample is not None:
                vector_kwargs.setdefault('sample', sample)
            if save is not None:
                vector_kwargs.setdefault('save', save)
            if same:
                vector_kwargs.setdefault('same', same)
            # Note: 'type', 'facet', 'group_by' deliberately not put in vector_kwargs.
            # group_by is passed as named param to _draw_vector (see below).
            # facet with vector is undefined.
            
            return self._draw_vector(
                y_expr, x_expr, method_map[type],
                group_by=group_by, **vector_kwargs
            )
        
        # Auto-detect type (scalar path)
        if type is None:
            if x_expr is None:
                type = "hist"
            else:
                type = "scatter"
        
        # Dispatch to specific plot method
        if type == "hist":
            return self.hist(
                expr, selection=selection, bins=bins, stats=stats,
                norm=norm, title=title, ax=ax, sample=sample, 
                save=save, group_by=group_by, facet=facet,
                same=same, **kwargs
            )
        elif type == "scatter":
            return self.scatter(
                expr, selection=selection, color=color, size=size,
                marker=marker, stats=stats, title=title, ax=ax,
                sample=sample, save=save, group_by=group_by, 
                facet=facet, same=same, **kwargs
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
                group_by=group_by, same=same, **kwargs
            )
        else:
            raise ValueError(
                f"Unknown plot type '{type}'. "
                "Expected: scatter, hist, hist2d, profile"
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
        
        # Phase 13.16.DF: Vector dispatch
        if isinstance(y_expr, list):
            # For hist, we accept both 1D vector (x is list of Nones) and 2D
            # (x is present, but hist only uses the first part — treat as 1D per element).
            vector_kwargs = dict(kwargs)
            if selection is not None:
                vector_kwargs.setdefault('selection', selection)
            if sample is not None:
                vector_kwargs.setdefault('sample', sample)
            if bins is not None:
                vector_kwargs.setdefault('bins', bins)
            if range is not None:
                vector_kwargs.setdefault('range', range)
            if norm is not None:
                vector_kwargs.setdefault('norm', norm)
            if stats is not None:
                vector_kwargs.setdefault('stats', stats)
            if title is not None:
                vector_kwargs.setdefault('title', title)
            if xlabel is not None:
                vector_kwargs.setdefault('xlabel', xlabel)
            if ylabel is not None:
                vector_kwargs.setdefault('ylabel', ylabel)
            if ax is not None:
                vector_kwargs.setdefault('ax', ax)
            if save is not None:
                vector_kwargs.setdefault('save', save)
            if auto_title:
                vector_kwargs.setdefault('auto_title', auto_title)
            if same:
                vector_kwargs.setdefault('same', same)
            return self._draw_vector(
                y_expr, x_expr, self.hist,
                group_by=group_by, **vector_kwargs
            )
        
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
        _suppress_color_cycle = kwargs.pop('_suppress_color_cycle', False)
        save_auto_title = auto_title
        if same:
            if 'color' not in kwargs and not _suppress_color_cycle:
                kwargs['color'] = self._get_next_color()
            if 'label' not in kwargs and group_by is None:
                kwargs['label'] = self._auto_label(col_expr)
            # Suppress title handling — we do it in _handle_same_post
            if ax is not None and ax.get_title():
                title = None
                auto_title = False
        
        # Apply selection and sampling
        df = self._apply_selection(self.df, selection)
        df = self._apply_sampling(df, sample)
        
        # Evaluate expression if needed
        if col_expr not in df.columns:
            df = df.assign(**{col_expr: self._eval_column(col_expr)})
        
        # Apply duck-typed label lookup if not explicitly set
        if xlabel is None:
            duck_label = self._get_label(col_expr)
            if duck_label is not None:
                xlabel = duck_label
        
        # Facet mode (same=True ignored in facet mode)
        if facet and group_by is not None:
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
        
        # Parse expression
        y_expr, x_expr = self._parse_expr(expr)
        
        # Phase 13.16.DF: Vector dispatch
        if isinstance(y_expr, list):
            if x_expr[0] is None:
                raise ValueError(
                    f"Scatter plot requires 'y:x' format, got vector 1D '{expr}'"
                )
            vector_kwargs = dict(kwargs)
            if selection is not None:
                vector_kwargs.setdefault('selection', selection)
            if sample is not None:
                vector_kwargs.setdefault('sample', sample)
            if color is not None:
                vector_kwargs.setdefault('color', color)
            if size is not None:
                vector_kwargs.setdefault('size', size)
            if marker is not None:
                vector_kwargs.setdefault('marker', marker)
            if stats is not None:
                vector_kwargs.setdefault('stats', stats)
            if title is not None:
                vector_kwargs.setdefault('title', title)
            if xlabel is not None:
                vector_kwargs.setdefault('xlabel', xlabel)
            if ylabel is not None:
                vector_kwargs.setdefault('ylabel', ylabel)
            if ax is not None:
                vector_kwargs.setdefault('ax', ax)
            if save is not None:
                vector_kwargs.setdefault('save', save)
            if same:
                vector_kwargs.setdefault('same', same)
            return self._draw_vector(
                y_expr, x_expr, self.scatter,
                group_by=group_by, **vector_kwargs
            )
        
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
                clabel=clabel, jitter=jitter, **kwargs
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
        
        return fig, axes, stats_dict
    
    def profile(
        self,
        expr: str,
        selection: Optional[Union[str, np.ndarray, callable]] = None,
        sample: Optional[int] = None,
        bins: Optional[int] = None,
        range: Optional[Tuple[float, float]] = None,
        error: str = "sem",
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
        
        # Parse expression
        y_expr, x_expr = self._parse_expr(expr)
        
        # Phase 13.16.DF: Vector dispatch
        if isinstance(y_expr, list):
            # Confirm 2D — profile requires x
            if x_expr[0] is None:
                raise ValueError(
                    f"Profile plot requires 'y:x' format, got vector 1D '{expr}'"
                )
            # Collect all current locals that are relevant for delegation.
            vector_kwargs = dict(kwargs)
            if selection is not None:
                vector_kwargs.setdefault('selection', selection)
            if bins is not None:
                vector_kwargs.setdefault('bins', bins)
            if stats is not None:
                vector_kwargs.setdefault('stats', stats)
            if title is not None:
                vector_kwargs.setdefault('title', title)
            if ax is not None:
                vector_kwargs.setdefault('ax', ax)
            if sample is not None:
                vector_kwargs.setdefault('sample', sample)
            if save is not None:
                vector_kwargs.setdefault('save', save)
            if xlabel is not None:
                vector_kwargs.setdefault('xlabel', xlabel)
            if ylabel is not None:
                vector_kwargs.setdefault('ylabel', ylabel)
            if weights is not None:
                vector_kwargs.setdefault('weights', weights)
            if same:
                vector_kwargs.setdefault('same', same)
            if auto_title:
                vector_kwargs.setdefault('auto_title', auto_title)
            return self._draw_vector(
                y_expr, x_expr, self.profile,
                group_by=group_by, **vector_kwargs
            )
        
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
        _suppress_color_cycle = kwargs.pop('_suppress_color_cycle', False)
        save_auto_title = auto_title
        if same:
            if 'color' not in kwargs and not _suppress_color_cycle:
                kwargs['color'] = self._get_next_color()
            if 'label' not in kwargs and group_by is None:
                kwargs['label'] = self._auto_label(y_expr, x_expr)
            # Suppress title — handled in _handle_same_post
            if ax is not None and ax.get_title():
                title = None
                auto_title = False
        
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
            from .facet import facet_profile
            fig, axes, stats_dict = facet_profile(
                df, x_expr, y_expr, group_by,
                top_k=top_k, ncols=ncols, sharex=sharex, sharey=sharey,
                suptitle=title, bins=bins, x_range=range, error=error,
                stats=stats, xlabel=xlabel, ylabel=ylabel,
                # Phase 13.12.DF: pass new parameters
                return_data=return_data, min_entries=min_entries,
                group_by_bins=group_by_bins, group_by_quantiles=group_by_quantiles,
                sort_groups=sort_groups, weights=weights,
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
                fig, axes = plt.subplots(
                    nrows, ncols, figsize=figsize,
                    squeeze=False, sharex=sharex, sharey=sharey
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
                    _, _, stats = method(expr, **merged)
                    group_stats.append(stats)
                
                # Suptitle
                if suptitle:
                    fig.suptitle(
                        suptitle,
                        fontsize=get_style_value("axes.titlesize", 14) + 2
                    )
                
                # Layout
                plt.tight_layout()
                if suptitle:
                    plt.subplots_adjust(top=0.92)
                
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
