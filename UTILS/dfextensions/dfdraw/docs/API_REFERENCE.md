# dfdraw Documentation Review & API Summary

## Module Docstring Status

| File | Module Docstring | Status |
|------|------------------|--------|
| `drawer.py` | "DFDraw - Main drawing class with TTree::Draw-like interface." | [OK] Present |
| `histogram.py` | "Histogram plot implementation for dfdraw. Supports: 1D histograms..." | [OK] Present, comprehensive |
| `scatter.py` | "Scatter plot implementation for dfdraw. Supports: Color mapping..." | [OK] Present, comprehensive |
| `profile.py` | "Profile plot implementation for dfdraw. A profile shows the mean..." | [OK] Present, comprehensive |
| `facet.py` | "Facet plot utilities for dfdraw. Creates subplot grids..." | [OK] Present |
| `style.py` | "Style management for dfdraw. Supports: Predefined styles..." | [OK] Present, comprehensive |
| `stats.py` | "Statistics computation for dfdraw. Phase 13.6.G.DF: Statistics Enhancements..." | [OK] Present, comprehensive |

**Assessment:** All files have module-level docstrings. [OK]

---

## ⚠️ Breaking Change (Phase 13.6.G.DF)

**Standard deviation calculation changed from sample (ddof=1) to population (ddof=0) to match ROOT.**

| Field | Change |
|-------|--------|
| `std` | Now uses ddof=0 (population std) |
| `std_x` | Now uses ddof=0 (population std) |
| `std_y` | Now uses ddof=0 (population std) |

Values are approximately 6% smaller than previous versions.

---

## Public API Summary

### drawer.py - `DFDraw` Class

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `__init__(self, data)` | Create drawer from DataFrame, AliasDataFrame, PyArrow Table, or dict |
| `draw` | `draw(expr, type=None, selection=None, color=None, size=None, marker=None, group_by=None, facet=False, bins=None, stats=None, norm=None, title=None, ax=None, sample=None, save=None, **kwargs)` | Universal draw method with auto type detection |
| `hist` | `hist(expr, bins=None, range=None, norm=None, stats=None, title=None, xlabel=None, ylabel=None, color=None, alpha=None, histtype=None, edgecolor=None, linewidth=None, label=None, group_by=None, top_k=None, stacked=False, **kwargs)` | Draw 1D histogram |
| `scatter` | `scatter(expr, color=None, size=None, marker=None, stats=None, title=None, xlabel=None, ylabel=None, alpha=None, edgecolors=None, linewidths=None, cmap=None, colorbar=True, clabel=None, group_by=None, top_k=None, jitter=None, **kwargs)` | Draw scatter plot |
| `profile` | `profile(expr, bins=None, x_range=None, error="sem", stats=None, title=None, xlabel=None, ylabel=None, color=None, marker=None, markersize=None, capsize=None, linestyle=None, linewidth=None, label=None, group_by=None, top_k=None, **kwargs)` | Draw profile plot (mean of y vs binned x) |
| `hist2d` | `hist2d(expr, bins=None, range=None, norm=None, stats=None, title=None, xlabel=None, ylabel=None, cmap=None, colorbar=True, clabel=None, vmin=None, vmax=None, **kwargs)` | Draw 2D histogram |
| `hexbin` | `hexbin(expr, gridsize=50, extent=None, norm=None, stats=None, title=None, xlabel=None, ylabel=None, cmap=None, colorbar=True, clabel=None, mincnt=None, vmin=None, vmax=None, **kwargs)` | Draw hexbin plot |
| `stats` | `stats(expr, selection=None, group_by=None)` | Compute statistics without plotting |
| `draw_batch` | `draw_batch(specs, save_dir=None, defaults=None, on_error='skip', verbose=True, save_format='png', dpi=150, close_figures=True, **kwargs)` | Batch plot generation from specification dict or YAML/JSON |
| `add_statistics_box` | `add_statistics_box(ax, values, position='upper right', expected_mean=None, expected_std=None, precision=3, fontsize=8, alpha=0.5)` | Add statistics annotation box to axis |
| `add_reference_overlay` | `add_reference_overlay(ax, func='gaussian', mu=0, sigma=1, label=None, color='red', linestyle='--', linewidth=1.5, show_legend=True, n_points=100)` | Add reference function overlay scaled to histogram |
| `backend` | `@property` | Return storage backend type ('pyarrow' or 'pandas') |
| `memory_info` | `memory_info()` | Return memory usage information |

### Annotation Methods (Phase 12.4b5)

#### `add_statistics_box()`

```python
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
) -> matplotlib.text.Text:
    """
    Add statistics annotation box to axis.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matplotlib axis to annotate.
    values : array-like
        Array of values for statistics computation.
    position : str, default 'upper right'
        Box position: 'upper right', 'upper left', 'lower right', 'lower left'.
    expected_mean : float, optional
        If provided, show delta_mu = mean - expected.
    expected_std : float, optional
        If provided, show delta_sigma = std - expected.
    precision : int, default 3
        Decimal places for values.
    fontsize : int, default 8
        Font size for text.
    alpha : float, default 0.5
        Background transparency.
        
    Returns
    -------
    matplotlib.text.Text or None
        The created text artist, or None if values is empty.
    """
```

#### `add_reference_overlay()`

```python
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
) -> matplotlib.lines.Line2D:
    """
    Add reference function overlay scaled to histogram.
    
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
        Legend label (default: 'N(mu,sigma)' for gaussian).
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
```

### histogram.py - Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `draw_hist` | `draw_hist(df, x, ax=None, bins=None, range=None, norm=None, stats=None, title=None, xlabel=None, ylabel=None, color=None, alpha=None, histtype=None, edgecolor=None, linewidth=None, label=None, group_by=None, top_k=None, stacked=False, **kwargs)` | Draw 1D histogram |
| `draw_hist2d` | `draw_hist2d(df, x, y, ax=None, bins=None, range=None, norm=None, stats=None, title=None, xlabel=None, ylabel=None, cmap=None, colorbar=True, clabel=None, vmin=None, vmax=None, **kwargs)` | Draw 2D histogram |
| `draw_hexbin` | `draw_hexbin(df, x, y, ax=None, gridsize=50, extent=None, norm=None, stats=None, title=None, xlabel=None, ylabel=None, cmap=None, colorbar=True, clabel=None, mincnt=None, vmin=None, vmax=None, **kwargs)` | Draw hexbin plot |

### scatter.py - Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `draw_scatter` | `draw_scatter(df, x, y, ax=None, color=None, size=None, marker=None, stats=None, title=None, xlabel=None, ylabel=None, alpha=None, edgecolors=None, linewidths=None, cmap=None, colorbar=True, clabel=None, group_by=None, top_k=None, jitter=None, **kwargs)` | Draw scatter plot |

### profile.py - Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `draw_profile` | `draw_profile(df, x, y, ax=None, bins=None, x_range=None, error="sem", stats=None, title=None, xlabel=None, ylabel=None, color=None, marker=None, markersize=None, capsize=None, linestyle=None, linewidth=None, label=None, group_by=None, top_k=None, **kwargs)` | Draw profile plot |

### facet.py - Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `create_facet_grid` | `create_facet_grid(df, group_by, top_k=None, ncols=None, figsize=None, sharex=True, sharey=True)` | Create grid of subplots for faceted plotting |
| `draw_facet` | `draw_facet(df, group_by, plot_func, top_k=None, ncols=None, figsize=None, sharex=True, sharey=True, title_template="{group}", suptitle=None, **plot_kwargs)` | Create faceted plot with custom plot function |
| `facet_hist` | `facet_hist(df, x, group_by, top_k=None, ncols=None, figsize=None, sharex=True, sharey=True, suptitle=None, **hist_kwargs)` | Create faceted 1D histogram |
| `facet_scatter` | `facet_scatter(df, x, y, group_by, top_k=None, ncols=None, figsize=None, sharex=True, sharey=True, suptitle=None, **scatter_kwargs)` | Create faceted scatter plot |
| `facet_profile` | `facet_profile(df, x, y, group_by, top_k=None, ncols=None, figsize=None, sharex=True, sharey=True, suptitle=None, **profile_kwargs)` | Create faceted profile plot |
| `facet_hist2d` | `facet_hist2d(df, x, y, group_by, top_k=None, ncols=None, figsize=None, sharex=True, sharey=True, suptitle=None, **hist2d_kwargs)` | Create faceted 2D histogram |
| `facet_hexbin` | `facet_hexbin(df, x, y, group_by, top_k=None, ncols=None, figsize=None, sharex=True, sharey=True, suptitle=None, **hexbin_kwargs)` | Create faceted hexbin plot |

### style.py - Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `get_style` | `get_style()` | Get copy of current style dictionary |
| `set_style` | `set_style(style)` | Set current style (str name, dict, or None to reset) |
| `save_style` | `save_style(path)` | Save current style to JSON file |
| `load_style` | `load_style(path)` | Load and apply style from JSON file |
| `list_styles` | `list_styles()` | List available predefined style names |
| `get_style_value` | `get_style_value(key, default=None)` | Get single style value |

### stats.py - Functions (Phase 13.6.G.DF Updated)

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_stats` | `compute_stats(df, y_col, x_col=None, group_by=None, range_x=None, range_y=None, robust=False)` | Compute statistics for plotting |
| `format_stats_box` | `format_stats_box(stats, fields=None, plot_type=None)` | Format statistics dict for display |
| `get_default_stats_fields` | `get_default_stats_fields(plot_type, robust=False)` | Get default fields for plot type |

#### `compute_stats()` (Phase 13.6.G.DF)

```python
def compute_stats(
    df: pd.DataFrame,
    y_col: str,
    x_col: Optional[str] = None,
    group_by: Optional[str] = None,
    range_x: Optional[Tuple[float, float]] = None,
    range_y: Optional[Tuple[float, float]] = None,
    robust: bool = False,
) -> pd.DataFrame:
    """
    Compute statistics for plotting.
    
    Parameters
    ----------
    df : DataFrame
        Input data.
    y_col : str
        Primary column (or only column for 1D).
    x_col : str, optional
        Secondary column for 2D stats.
    group_by : str, optional
        Compute stats per group.
    range_x : tuple, optional
        (min, max) inclusive range for x values.
        For 1D plots, this filters the primary variable.
        For 2D plots, this filters the x-axis variable.
    range_y : tuple, optional
        (min, max) inclusive range for y values (2D only).
    robust : bool, default False
        If True, include robust statistics (median, q25, q75, mad).
    
    Returns
    -------
    DataFrame
        Statistics table.
    
    Notes
    -----
    Phase 13.6.G.DF Breaking Change:
        std, std_x, std_y now use population standard deviation (ddof=0)
        to match ROOT's TTree::Draw behavior.
    """
```

#### `format_stats_box()` (Phase 13.6.G.DF)

```python
def format_stats_box(
    stats: Dict[str, Any],
    fields: Optional[List[str]] = None,
    plot_type: Optional[str] = None,
) -> str:
    """
    Format statistics for display in plot.
    
    Parameters
    ----------
    stats : dict
        Statistics dictionary.
    fields : list, optional
        Fields to include. If None, auto-detected from plot_type.
    plot_type : str, optional
        Plot type hint: 'hist', 'hist2d', 'scatter', 'profile', 'hexbin'
        Used to select appropriate default fields.
        
        Default fields by plot_type:
        - 'hist': ['n', 'mean', 'std']
        - 'hist2d': ['n', 'mean_x', 'mean_y', 'std_x', 'std_y', 'corr']
        - 'scatter', 'profile', 'hexbin': ['n', 'mean_x', 'mean_y']
    
    Returns
    -------
    str
        Formatted text for stats box.
    """
```

#### `get_default_stats_fields()` (Phase 13.6.G.DF - NEW)

```python
def get_default_stats_fields(
    plot_type: str,
    robust: bool = False,
) -> List[str]:
    """
    Get default statistics fields for a plot type.
    
    Parameters
    ----------
    plot_type : str
        Plot type: 'hist', 'hist2d', 'scatter', 'profile', 'hexbin'
    robust : bool, default False
        If True, use robust defaults for 1D plots.
    
    Returns
    -------
    list
        List of field names.
    
    Notes
    -----
    When robust=True:
    - 1D defaults change to ['n', 'median', 'mad']
    - 2D defaults are unchanged
    """
```

---

## Return Value Structure

All draw functions return `(fig, ax, stats_dict)`:

```python
stats_dict = {
    # 1D stats
    "n": int,           # Number of entries (within range)
    "mean": float,      # Mean value
    "std": float,       # Standard deviation (population, ddof=0)
    "min": float,       # Minimum
    "max": float,       # Maximum
    
    # 2D stats (additional)
    "mean_x": float,    # Mean of x
    "mean_y": float,    # Mean of y
    "std_x": float,     # Std of x (population, ddof=0)
    "std_y": float,     # Std of y (population, ddof=0)
    "corr": float,      # Correlation coefficient
    
    # Robust stats (when robust=True)
    "median": float,    # 50th percentile
    "q25": float,       # 25th percentile
    "q75": float,       # 75th percentile
    "mad": float,       # Median absolute deviation
    
    # Group stats (additional)
    "grouped": bool,    # Was group_by used?
}
```

---

## Default Stats Fields by Plot Type (Phase 13.6.G.DF)

| Plot Type | Default Fields |
|-----------|----------------|
| `hist` | `['n', 'mean', 'std']` |
| `hist` (robust=True) | `['n', 'median', 'mad']` |
| `hist2d` | `['n', 'mean_x', 'mean_y', 'std_x', 'std_y', 'corr']` |
| `scatter` | `['n', 'mean_x', 'mean_y']` |
| `profile` | `['n', 'mean_x', 'mean_y']` |
| `hexbin` | `['n', 'mean_x', 'mean_y']` |

---

## Predefined Styles

| Style | Key Differences |
|-------|-----------------|
| `"default"` | Standard settings |
| `"publication"` | Smaller (6x4.5), no grid, step histograms, no legend frame |
| `"presentation"` | Larger (10x7), big fonts, 80pt markers |
| `"minimal"` | No grid, no edges, no legend frame |

---

## Style Keys (Phase 13.6.G.DF Updated)

| Key | Default | Description |
|-----|---------|-------------|
| `stats.show` | False | Show stats box by default |
| `stats.position` | "upper right" | Stats box position |
| `stats.fields` | ["n", "mean", "std"] | Default stats fields |
| `stats.fontsize` | 10 | Stats box font size |
| `stats.alpha` | 0.8 | Stats box transparency |
| `stats.boxstyle` | "round" | Stats box style |
| `stats.robust` | False | **NEW** Use robust defaults for 1D (median, MAD) |

---

## Normalization Options

| Value | Description | Histogram | Profile |
|-------|-------------|-----------|---------|
| `None` / `"count"` | Raw counts | [OK] | N/A |
| `"density"` | Probability density (area = 1) | [OK] | N/A |
| `"probability"` | Probability (sum = 1) | [OK] | N/A |
| `"log"` | Logarithmic color scale | [OK] (hist2d/hexbin) | N/A |

---

## Error Bar Options (Profile)

| Value | Description |
|-------|-------------|
| `"sem"` | Standard error of mean: sigma/sqrt(n) (default) |
| `"std"` | Standard deviation: sigma |
| `"none"` | No error bars |

---

## Range Semantics (Phase 13.6.G.DF)

| Plot Type | Range Format | Stats Application |
|-----------|--------------|-------------------|
| `hist` | `range=(min, max)` | Filter values, inclusive [min, max] |
| `hist2d` | `range=((xmin, xmax), (ymin, ymax))` | Filter both axes, inclusive |
| `scatter` | No range parameter | Stats on all plotted data |
| `profile` | `range=(min, max)` | Filter x values only |
| `hexbin` | `extent=(xmin, xmax, ymin, ymax)` | Stats filter uses `extent` |

**Note:** For hexbin, if both `extent` and `range` are provided, `extent` takes precedence.

---

## Comparison to ROOT TTree::Draw

| ROOT | dfdraw | Notes |
|------|--------|-------|
| `tree->Draw("x")` | `drawer.draw("x")` | 1D histogram |
| `tree->Draw("y:x")` | `drawer.draw("y:x")` | 2D scatter |
| `tree->Draw("y:x", "x>0")` | `drawer.draw("y:x", selection="x>0")` | With cut |
| `tree->Draw("y:x", "", "prof")` | `drawer.draw("y:x", type="profile")` | Profile |
| `tree->Draw("y:x", "", "colz")` | `drawer.draw("y:x", type="hist2d")` | 2D histogram |
| `tree->Draw("y:x>>h(100,0,1)")` | `drawer.draw("y:x", bins=100, range=(0,1))` | Custom binning |

**Phase 13.6.G.DF:** Stats now computed within range, matching ROOT behavior.

---

## Test Coverage

| Test File | Coverage |
|-----------|----------|
| `test_drawer.py` | DFDraw class init, expression parsing, selection, sampling, dispatch |
| `test_histogram.py` | 1D histogram: bins, range, norm, labels, selection, grouping, stats |
| `test_scatter.py` | Scatter: color/size mapping, jitter, grouping, stats |
| `test_profile.py` | Profile: bins, error types, grouping, labels |
| `test_hexbin.py` | Hexbin: gridsize, colormap, normalization, faceting |
| `test_hist2d.py` | 2D histogram: bins, colormap, normalization |
| `test_facet.py` | Faceted layouts for all plot types |
| `test_style.py` | Style get/set/save/load, predefined styles |
| `test_batch.py` | Batch processing, YAML/JSON loading, error handling |
| `test_adf_integration.py` | AliasDataFrame duck-typing, axis titles |
| `test_validation_display.py` | Statistics box, reference overlay |
| `test_pyarrow_input.py` | PyArrow Table input support |
| `test_stats_enhancements.py` | **NEW** Statistics enhancements (Phase 13.6.G.DF) |

**Total tests:** 310 passing

---

## Phase 13.6.G.DF Summary

### New Features

1. **Auto-detect default stats fields** by plot type
2. **Range-aware stats** — computed within specified range only
3. **Robust statistics** — median, q25, q75, MAD
4. **`stats.robust` style key** — global robust mode for 1D

### Breaking Change

- `std`, `std_x`, `std_y` now use population std (ddof=0) to match ROOT

### New Functions

- `get_default_stats_fields(plot_type, robust)` — helper for default fields

### Updated Signatures

- `compute_stats(..., range_x=None, range_y=None, robust=False)`
- `format_stats_box(..., plot_type=None)`
