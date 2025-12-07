"""
DFDraw - Main drawing class with TTree::Draw-like interface.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from .style import get_style, get_style_value

# Type alias for return value
DrawResult = Tuple[Any, Any, Dict[str, Any]]  # (fig, ax, stats)


class DFDraw:
    """
    DataFrame drawing class with TTree::Draw-like interface.
    
    Parameters
    ----------
    data : DataFrame-like
        Input data. Accepts:
        - pandas.DataFrame
        - AliasDataFrame (uses .df attribute)
        - dict of arrays (converted to DataFrame)
    
    Examples
    --------
    >>> plotter = DFDraw(df)
    >>> fig, ax, stats = plotter.draw("y:x", color="category")
    >>> fig, ax, stats = plotter.hist("x", bins=100)
    """
    
    def __init__(self, data):
        self.df = self._normalize_data(data)
    
    def _normalize_data(self, data) -> pd.DataFrame:
        """
        Convert input to pandas DataFrame.
        
        Supports duck typing:
        - DataFrame: use as-is
        - Has .df attribute: extract DataFrame (AliasDataFrame)
        - dict-like: convert to DataFrame
        """
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
            f"Cannot create DFDraw from {type(data)}. "
            "Expected DataFrame, AliasDataFrame, or dict of arrays."
        )
    
    # =========================================================================
    # Expression Parsing
    # =========================================================================
    
    def _parse_expr(self, expr: str) -> Tuple[str, Optional[str]]:
        """
        Parse TTree::Draw-style expression.
        
        Parameters
        ----------
        expr : str
            Expression like "y:x" or "x"
        
        Returns
        -------
        tuple
            (y_expr, x_expr) or (x_expr, None) for 1D
        """
        parts = expr.split(":")
        if len(parts) == 1:
            return (parts[0].strip(), None)
        elif len(parts) == 2:
            return (parts[0].strip(), parts[1].strip())
        else:
            raise ValueError(
                f"Invalid expression '{expr}'. "
                "Expected 'y:x' or 'x' format."
            )
    
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
        **kwargs
            Additional style overrides.
        
        Returns
        -------
        tuple
            (fig, ax, stats_dict)
        """
        # Parse expression to determine dimensionality
        y_expr, x_expr = self._parse_expr(expr)
        
        # Auto-detect type
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
                save=save, group_by=group_by, facet=facet, **kwargs
            )
        elif type == "scatter":
            return self.scatter(
                expr, selection=selection, color=color, size=size,
                marker=marker, stats=stats, title=title, ax=ax,
                sample=sample, save=save, group_by=group_by, 
                facet=facet, **kwargs
            )
        elif type == "hist2d":
            return self.hist2d(
                expr, selection=selection, bins=bins, stats=stats,
                title=title, ax=ax, sample=sample, save=save, **kwargs
            )
        elif type == "profile":
            return self.profile(
                expr, selection=selection, bins=bins, stats=stats,
                title=title, ax=ax, sample=sample, save=save,
                group_by=group_by, **kwargs
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
        col_expr = y_expr  # Use y (first part) as the variable
        
        # Apply selection and sampling
        df = self._apply_selection(self.df, selection)
        df = self._apply_sampling(df, sample)
        
        # Evaluate expression if needed
        if col_expr not in df.columns:
            df = df.assign(**{col_expr: self._eval_column(col_expr)})
        
        # Facet mode
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
                group_by=group_by, top_k=top_k, **kwargs
            )
            axes = ax
        
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
        
        if x_expr is None:
            raise ValueError(
                f"Scatter plot requires 'y:x' format, got '{expr}'"
            )
        
        # Apply selection and sampling
        df = self._apply_selection(self.df, selection)
        df = self._apply_sampling(df, sample)
        
        # Evaluate expressions if needed
        if y_expr not in df.columns:
            df = df.assign(**{y_expr: self._eval_column(y_expr)})
        if x_expr not in df.columns:
            df = df.assign(**{x_expr: self._eval_column(x_expr)})
        
        # Facet mode
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
        **kwargs
            Additional arguments.
        
        Returns
        -------
        tuple
            (fig, ax, stats_dict)
        """
        from .plots.profile import draw_profile
        
        # Parse expression
        y_expr, x_expr = self._parse_expr(expr)
        
        if x_expr is None:
            raise ValueError(
                f"Profile plot requires 'y:x' format, got '{expr}'"
            )
        
        # Apply selection and sampling
        df = self._apply_selection(self.df, selection)
        df = self._apply_sampling(df, sample)
        
        # Evaluate expressions if needed
        if y_expr not in df.columns:
            df = df.assign(**{y_expr: self._eval_column(y_expr)})
        if x_expr not in df.columns:
            df = df.assign(**{x_expr: self._eval_column(x_expr)})
        
        # Facet mode
        if facet and group_by is not None:
            from .facet import facet_profile
            fig, axes, stats_dict = facet_profile(
                df, x_expr, y_expr, group_by,
                top_k=top_k, ncols=ncols, sharex=sharex, sharey=sharey,
                suptitle=title, bins=bins, x_range=range, error=error,
                stats=stats, xlabel=xlabel, ylabel=ylabel, **kwargs
            )
        else:
            # Standard mode (single plot or overlay)
            fig, ax, stats_dict = draw_profile(
                df, x_expr, y_expr,
                ax=ax, bins=bins, x_range=range, error=error,
                stats=stats, title=title, xlabel=xlabel, ylabel=ylabel,
                group_by=group_by, top_k=top_k, **kwargs
            )
            axes = ax
        
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
        
        if x_expr is None:
            raise ValueError(
                f"hist2d requires 'y:x' format, got '{expr}'"
            )
        
        # Apply selection and sampling
        df = self._apply_selection(self.df, selection)
        df = self._apply_sampling(df, sample)
        
        # Evaluate expressions if needed
        if y_expr not in df.columns:
            df = df.assign(**{y_expr: self._eval_column(y_expr)})
        if x_expr not in df.columns:
            df = df.assign(**{x_expr: self._eval_column(x_expr)})
        
        # Facet mode
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
                vmin=vmin, vmax=vmax, **kwargs
            )
            axes = ax
        
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
        
        if x_expr is None:
            raise ValueError(
                f"hexbin requires 'y:x' format, got '{expr}'"
            )
        
        # Apply selection and sampling
        df = self._apply_selection(self.df, selection)
        df = self._apply_sampling(df, sample)
        
        # Evaluate expressions if needed
        if y_expr not in df.columns:
            df = df.assign(**{y_expr: self._eval_column(y_expr)})
        if x_expr not in df.columns:
            df = df.assign(**{x_expr: self._eval_column(x_expr)})
        
        # Facet mode
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
                mincnt=mincnt, vmin=vmin, vmax=vmax, **kwargs
            )
            axes = ax
        
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
        
        return compute_stats(df, y_expr, x_expr, group_by=group_by)
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def draw_batch(
        self,
        specs: Union[Dict[str, Dict[str, Any]], str],
        save_dir: Optional[str] = None,
        defaults: Optional[Dict[str, Any]] = None,
        on_error: str = 'skip',
        verbose: bool = True,
        save_format: str = 'png',
        dpi: int = 150,
        close_figures: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Draw multiple figures from specification dictionary.
        
        Parameters
        ----------
        specs : dict or str
            Dictionary of plot specifications, or path to YAML/JSON file.
            Each key is the plot name, value is dict with 'expr' and optional parameters.
        save_dir : str, optional
            Directory to save figures. Created if doesn't exist.
        defaults : dict, optional
            Default parameters applied to all plots (overridden by per-plot specs).
        on_error : str, default 'skip'
            'skip': Continue on errors, collect in results['_errors']
            'raise': Stop on first error
        verbose : bool, default True
            Print progress messages.
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
            Each plot entry has: {'stats': dict, 'fig': Figure, 'ax': Axes, 'path': str}
            If close_figures=True and save_dir set, fig/ax will be None.
            '_errors': dict of {name: error_message}
            '_summary': {'total': int, 'success': int, 'failed': int}
        
        Examples
        --------
        >>> specs = {
        ...     'hist_x': {'expr': 'x', 'bins': 50},
        ...     'scatter_yx': {'expr': 'y:x', 'sample': 10000},
        ...     'profile_dEdx': {'expr': 'dEdx:p', 'type': 'profile', 'bins': 100},
        ... }
        >>> results = plotter.draw_batch(specs, save_dir='plots/', verbose=True)
        [1/3] hist_x → plots/hist_x.png
        [2/3] scatter_yx → plots/scatter_yx.png
        [3/3] profile_dEdx → plots/profile_dEdx.png
        Completed: 3/3 (0 errors)
        """
        import os
        import matplotlib.pyplot as plt
        
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
