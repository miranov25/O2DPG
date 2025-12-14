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
| `stats.py` | "Statistics computation for dfdraw." | [OK] Present (brief) |

**Assessment:** All files have module-level docstrings. [OK]

---

## Public API Summary

### drawer.py - `DFDraw` Class

| Method | Signature | Description |
|--------|-----------|-------------|
| `__init__` | `__init__(self, data)` | Create drawer from DataFrame, AliasDataFrame, or dict |
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

### stats.py - Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `compute_stats` | `compute_stats(df, y_col, x_col=None, group_by=None)` | Compute statistics for plotting |
| `format_stats_box` | `format_stats_box(stats, fields=None)` | Format statistics dict for display |

---

## Return Value Structure

All draw functions return `(fig, ax, stats_dict)`:

```python
stats_dict = {
    # 1D stats
    "n": int,           # Number of entries
    "mean": float,      # Mean value
    "std": float,       # Standard deviation
    "min": float,       # Minimum
    "max": float,       # Maximum
    
    # 2D stats (additional)
    "mean_x": float,    # Mean of x
    "mean_y": float,    # Mean of y
    "std_x": float,     # Std of x
    "std_y": float,     # Std of y
    "corr": float,      # Correlation coefficient
    
    # Group stats (additional)
    "grouped": bool,    # Was group_by used?
}
```

---

## Predefined Styles

| Style | Key Differences |
|-------|-----------------|
| `"default"` | Standard settings |
| `"publication"` | Smaller (6x4.5), no grid, step histograms, no legend frame |
| `"presentation"` | Larger (10x7), big fonts, 80pt markers |
| `"minimal"` | No grid, no edges, no legend frame |

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

## Comparison to ROOT TTree::Draw

| ROOT | dfdraw | Notes |
|------|--------|-------|
| `tree->Draw("x")` | `drawer.draw("x")` | 1D histogram |
| `tree->Draw("y:x")` | `drawer.draw("y:x")` | 2D scatter |
| `tree->Draw("y:x", "x>0")` | `drawer.draw("y:x", selection="x>0")` | With cut |
| `tree->Draw("y:x", "", "prof")` | `drawer.draw("y:x", type="profile")` | Profile |
| `tree->Draw("y:x", "", "colz")` | `drawer.draw("y:x", type="hist2d")` | 2D histogram |
| `tree->Draw("y:x>>h(100,0,1)")` | `drawer.draw("y:x", bins=100, range=(0,1))` | Custom binning |

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

**Total tests:** 232 passing
