# dfdraw Documentation Review & API Summary

**Version:** Phase 13.16.DF v1.0
**Last Updated:** 2026-04-09

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
| `_auto_title.py` | "Auto-title helpers for dfdraw plot functions. Phase 13.12.DF v1.2..." | [OK] Present |

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
| `draw` | `draw(expr, type=None, selection=None, color=None, size=None, marker=None, group_by=None, facet=False, bins=None, stats=None, norm=None, title=None, ax=None, sample=None, save=None, figsize=None, same=False, vector_style=None, group_style=None, **kwargs)` | Universal draw method with auto type detection; supports vector expressions |
| `hist` | `hist(expr, ..., auto_title=False, same=False, vector_style=None, group_style=None, **kwargs)` | Draw 1D histogram (supports vector `[y1,y2,y3]`) |
| `scatter` | `scatter(expr, ..., same=False, vector_style=None, group_style=None, **kwargs)` | Draw scatter plot (supports vector `[y1,y2]:[x1,x2]` etc.) |
| `profile` | `profile(expr, ..., return_data=False, min_entries=3, group_by_bins=None, group_by_quantiles=None, sort_groups=True, weights=None, auto_title=False, same=False, vector_style=None, group_style=None, **kwargs)` | Draw profile plot (supports vector expressions) |
| `hist2d` | `hist2d(expr, ..., auto_title=False, same=False, **kwargs)` | Draw 2D histogram (vector input rejected — fail-fast) |
| `hexbin` | `hexbin(expr, ..., auto_title=False, same=False, **kwargs)` | Draw hexbin plot (vector input rejected — fail-fast) |
| `stats` | `stats(expr, selection=None, group_by=None)` | Compute statistics without plotting; returns `list[dict]` for vector input |
| `draw_batch` | `draw_batch(specs, save_dir=None, defaults=None, on_error='skip', verbose=True, save_format='png', dpi=150, close_figures=True, **kwargs)` | Batch plot generation — dict format or list-of-groups format; supports vector expressions in plot specs |
| `add_statistics_box` | `add_statistics_box(ax, values, position='upper right', expected_mean=None, expected_std=None, precision=3, fontsize=8, alpha=0.5)` | Add statistics annotation box to axis |
| `add_reference_overlay` | `add_reference_overlay(ax, func='gaussian', mu=0, sigma=1, label=None, color='red', linestyle='--', linewidth=1.5, show_legend=True, n_points=100)` | Add reference function overlay scaled to histogram |
| `backend` | `@property` | Return storage backend type ('pyarrow' or 'pandas') |
| `memory_info` | `memory_info()` | Return memory usage information |

### New Parameters (Phase 13.12–13.16.DF)

#### Vector Expression Syntax (Phase 13.16.DF v1.0)

Bracket-vector syntax draws multiple curves/series in a single call. Available on
`draw()`, `profile()`, `hist()`, `scatter()`. Fail-fast on `hist2d()` and `hexbin()`.

**Syntax patterns:**

```python
# N:1 — 3 y-columns vs shared x
drawer.profile("[y1,y2,y3]:x")

# 1:N — shared y vs 3 x-columns
drawer.profile("y1:[x1,x2,x3]")

# N:N — 2 paired series (element-wise)
drawer.profile("[y1,y2]:[x1,x2]")

# 1D vector — 3 overlaid histograms
drawer.hist("[y1,y2,y3]")

# Paren-aware expressions inside brackets
drawer.profile("[max(a,b),max(c,d)]:x")

# N:M mismatch raises ValueError
drawer.profile("[y1,y2]:[x1,x2,x3]")  # ValueError
```

**New parameters:**

```python
vector_style: Optional[str] = None    # 'color' | 'linestyle' — channel for vector dimension
group_style: Optional[str] = None     # 'color' | 'linestyle' — channel for group_by dimension
```

**Style channel defaults:**
- Without `group_by`: `vector_style='color'` (each vector series gets a different color)
- With `group_by`: `vector_style='linestyle'`, `group_style='color'` (groups get colors, vector series get linestyles)
- Both channels must be distinct when used together (otherwise `ValueError`)

**Return value for vector input:**
- `fig, ax, stats_list` where `stats_list` is `list[dict]` of length N
- `auto_title` defaults to `True` in vector mode (can be overridden)
- Y-axis label uses common-prefix rule when ≥2 characters shared, else bracket-list
- Secondary legend for vector dimension added via `ax.add_artist()` (preserves group legend)

**Architectural note — AD-37 fix:** The vector interface was introduced to fix a bug
in `AliasDataFrame.draw()` where each call creates a fresh `DFDraw` instance, resetting
the color cycle. A scalar `same=True` loop through ADF therefore showed only 1–2 colors.
The vector path uses a single `DFDraw` instance internally, preserving color/label
continuity across the full series. Invariance test
`test_vector_through_adf_equivalent_to_direct` confirms the fix.

**Example — ITS layer residuals (real use case):**

```python
# 6 ITS layer residuals vs stave, single call
aDF.draw("[dd_dyITS0,dd_dyITS1,dd_dyITS2,dd_dyITS3,dd_dyITS4,dd_dyITS5]:staveITS",
         selection="row==180 & isPrimITS==1",
         type='profile', bins=12)
# Result: 6 distinct colors, per-layer legend, auto-title, stats_list of length 6
```

**Invariance contract:** Vector path produces byte-identical axes state (line count,
colors, linestyles, labels, xdata/ydata to 10 decimals) and per-pair stats (each
numeric field within 1e-9) compared to a manual scalar `same=True` loop. Verified
by 7 A≡B tests in `tests/test_vector.py::TestVectorInvariance`.

#### same=True — Plot Superposition (Phase 13.13.DF, AD-15)

Available on all draw methods: `draw()`, `hist()`, `scatter()`, `profile()`, `hist2d()`, `hexbin()`.

```python
same: bool = False
```

When `True`:
- Reuses last axes (`self._last_ax`) or falls back to `plt.gca()`
- Auto-increments colors from palette (AD-16)
- Auto-generates label from expression (AD-17)
- Appends to title when `auto_title=True` (AD-18)
- Shows legend automatically

Precedence: `ax=` wins over `same=True`. Explicit `color=`, `label=`, `title=` override auto-features.

#### auto_title — Automatic Title (Phase 13.12.DF v1.2)

Available on: `hist()`, `profile()`, `hist2d()`, `hexbin()`.

```python
auto_title: Union[bool, str] = False
```

Values:
- `False`: no auto-title (default, or from style)
- `True` / `"all"`: "y vs x  group:group_by  weights:w\nselection"
- `"expr"`: "y vs x" only
- `"expr+group"`: "y vs x  group:group_by"
- `"expr+sel"`: "y vs x\nselection"

Explicit `title=` always overrides `auto_title`.

#### Profile-Specific Parameters (Phase 13.12.DF)

```python
return_data: bool = False       # F1: Include profile DataFrame in stats_dict
min_entries: int = 3            # F2: Min entries per bin to plot (AD-1)
group_by_bins: int = None       # F3: Equal-width bins for float group_by
group_by_quantiles: int = None  # F3: Equal-count bins for float group_by
sort_groups: bool = True        # F4: Sort groups in legend
weights: str = None             # v1.1: Column name for weighted statistics
```

#### draw_batch — List Format (Phase 13.14.DF)

`specs` parameter now accepts `list` (group format) in addition to `dict` (original format):

```python
specs: Union[Dict[str, Dict[str, Any]], List[Dict[str, Any]], str]
```

Group spec keys:
- `name` (required), `plots` (required)
- `defaults`, `ncols`, `layout`, `figsize`, `suptitle`, `savefig`, `sharex`, `sharey`

Option hierarchy: `kwargs < draw_batch defaults= < group['defaults'] < plot_spec`

#### verbose — Verbosity Levels (Phase 13.14.DF)

```python
verbose: Union[bool, int] = True
```

| Value | Behavior |
|-------|----------|
| `False` / `0` | Silent |
| `True` / `1` | Progress — group names, save paths, summary |
| `2` | Debug — also prints merged parameters per plot |

---

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
| `draw_hist` | `draw_hist(df, x, ax=None, bins=None, range=None, norm=None, stats=None, title=None, xlabel=None, ylabel=None, color=None, alpha=None, histtype=None, edgecolor=None, linewidth=None, label=None, group_by=None, top_k=None, stacked=False, auto_title=False, selection=None, **kwargs)` | Draw 1D histogram |
| `draw_hist2d` | `draw_hist2d(df, x, y, ax=None, bins=None, range=None, norm=None, stats=None, title=None, xlabel=None, ylabel=None, cmap=None, colorbar=True, clabel=None, vmin=None, vmax=None, auto_title=False, selection=None, **kwargs)` | Draw 2D histogram |
| `draw_hexbin` | `draw_hexbin(df, x, y, ax=None, gridsize=50, extent=None, norm=None, stats=None, title=None, xlabel=None, ylabel=None, cmap=None, colorbar=True, clabel=None, mincnt=None, vmin=None, vmax=None, auto_title=False, selection=None, **kwargs)` | Draw hexbin plot |

### scatter.py - Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `draw_scatter` | `draw_scatter(df, x, y, ax=None, color=None, size=None, marker=None, stats=None, title=None, xlabel=None, ylabel=None, alpha=None, edgecolors=None, linewidths=None, cmap=None, colorbar=True, clabel=None, group_by=None, top_k=None, jitter=None, **kwargs)` | Draw scatter plot |

### profile.py - Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `draw_profile` | `draw_profile(df, x, y, ax=None, bins=None, x_range=None, error="sem", stats=None, title=None, xlabel=None, ylabel=None, color=None, marker=None, markersize=None, capsize=None, linestyle=None, linewidth=None, label=None, group_by=None, top_k=None, return_data=False, min_entries=3, group_by_bins=None, group_by_quantiles=None, sort_groups=True, weights=None, auto_title=False, selection=None, **kwargs)` | Draw profile plot |
| `_format_interval_label` | `_format_interval_label(interval) -> str` | Format pandas Interval as 'low-high' string (AD-3) |
| `_interval_sort_key` | `_interval_sort_key(label) -> tuple` | Sort key for interval labels including negative ranges |

### _auto_title.py - Functions (Phase 13.12.DF v1.2)

| Function | Signature | Description |
|----------|-----------|-------------|
| `parse_auto_title_parts` | `parse_auto_title_parts(auto_title) -> set` | Parse auto_title parameter into set of parts |
| `build_auto_title` | `build_auto_title(x, y=None, group_by=None, selection=None, weights=None, parts=...) -> dict` | Build auto-title dict with 'main' and 'sub' keys |
| `apply_auto_title` | `apply_auto_title(ax, title_dict, fontsize=None, sub_fontsize=None)` | Apply auto-title to axes (first plot) |
| `append_auto_title` | `append_auto_title(ax, title_dict, fontsize=None, sub_fontsize=None)` | Append to existing title for same=True overlay (Phase 13.13.DF) |
| `resolve_auto_title` | `resolve_auto_title(auto_title) -> bool\|str` | Resolve auto_title: per-call value > style default |

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
    
    # Profile data export (when return_data=True, Phase 13.12.DF)
    "profile_data": DataFrame,  # x_center, x_low, x_high, y_mean, y_std, y_sem, count
}
```

**draw_batch return (list format, Phase 13.14.DF):**
```python
results = {
    'group_name': {
        'fig': Figure,          # None if close_figures + save
        'axes': [ax1, ax2, ...],  # List of subplot axes
        'stats': [stats1, stats2, ...],  # Stats per plot
        'path': 'saved/path.png',  # Save path or None
    },
    '_errors': {'name': 'error message'},
    '_summary': {'total': int, 'success': int, 'failed': int},
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

## Style Keys (Phase 13.14.DF Updated)

| Key | Default | Description |
|-----|---------|-------------|
| `stats.show` | False | Show stats box by default |
| `stats.position` | "upper right" | Stats box position |
| `stats.fields` | ["n", "mean", "std"] | Default stats fields |
| `stats.fontsize` | 10 | Stats box font size |
| `stats.alpha` | 0.8 | Stats box transparency |
| `stats.boxstyle` | "round" | Stats box style |
| `stats.robust` | False | Use robust defaults for 1D (median, MAD) |
| `auto_title` | False | Enable auto-title globally (Phase 13.12.DF v1.2) |
| `auto_title.fontsize` | 10 | Auto-title main font size |
| `auto_title.sel_fontsize` | 8 | Auto-title selection subtitle font size |

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
| `hist2->Draw("same")` | `drawer.draw("y:x", same=True)` | Superposition (Phase 13.13.DF) |

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
| `test_stats_enhancements.py` | Statistics enhancements (Phase 13.6.G.DF) |
| `test_auto_title.py` | Auto-title system (Phase 13.12.DF v1.2) |
| `test_profile_phase13_12.py` | Profile enhancements: return_data, min_entries, group_by_bins, weights |
| `test_same.py` | same=True superposition: axes reuse, colors, labels, titles (Phase 13.13.DF) |
| `test_batch_groups.py` | Batch groups: defaults cascade, layouts, same=True, verbose levels (Phase 13.14.DF) |

**Total tests:** 399 passing

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

---

## Phase 13.12–13.16.DF Summary

### Phase 13.12.DF — Profile Enhancements

- `return_data=True`: export profile stats as DataFrame
- `min_entries=3`: suppress low-statistics bins
- `group_by_bins`/`group_by_quantiles`: auto-bin float columns
- `sort_groups=True`: sorted legend order
- `weights='col'`: weighted mean/std/sem
- `auto_title`: automatic title system

### Phase 13.13.DF — same=True Superposition

- `same=True` on all draw methods
- Auto-increment colors, auto-labels, title append
- `self._last_ax` tracking with `plt.gca()` fallback
- `append_auto_title()` for title merging

### Phase 13.14.DF — draw_batch Defaults Hierarchy

- List-of-groups format with option hierarchy
- Subplot grid: `ncols`, `layout=(r,c)`, `figsize`, `suptitle`
- `same=True` within groups
- `verbose=2` debug mode (merged params per plot)
- Interval sort fix for negative ranges

### Phase 13.15.DF — Test Infrastructure

- `tests/feature_taxonomy.py` — 35 feature enumeration (expanded to 43 in 13.16.DF)
- `tests/test_layer_classification.py` — smoke vs invariance test markers
- `scripts/generate_capability_matrix.py` → auto-generated `docs/CAPABILITY_MATRIX.md`
- `run_tests.sh` — full/quick/matrix modes + `reviewer.zip` packaging
- `scripts/phase_tag.sh` — phase boundary tag helper
- 401/401 tests passing, 5 Verified features

### Phase 13.16.DF — Vector Expression Interface

- Bracket-vector syntax: `[y1,y2,y3]:x`, `y:[x1,x2]`, `[y1,y2]:[x1,x2]`, `[y1,y2,y3]`
- Paren-aware comma split: `[max(a,b),max(c,d)]:x`
- New parameters: `vector_style`, `group_style` (style channel decomposition)
- Per-pair `stats()` returns `list[dict]`
- Fail-fast on `hist2d()`/`hexbin()` vector input
- Architectural fix for AD-37 AliasDataFrame color-cycling bug
- Conditional color cycle reset (preserves `SAME.axes_reuse` contract)
- 7 strong A≡B invariance tests (`TestVectorInvariance`)
- 451/451 tests passing, 43 features, 21 invariance, 6 Verified

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-14 | Main Reviewer | Initial API_REFERENCE.md |
| 1.1 | 2026-01-29 | Claude-Main | Added Phase 13.6.G.DF stats API |
| 1.2 | 2026-03-28 | Claude41 | Added Phase 13.12-13.14 APIs: same=, auto_title=, profile params, list format, verbose=int, _auto_title.py functions, _interval_sort_key, updated test coverage to 399 |
| 1.3 | 2026-04-09 | Claude41 | Added Phase 13.15.DF (test infrastructure) and Phase 13.16.DF (vector expressions): bracket syntax, `vector_style`/`group_style` parameters, `stats()` list return for vector, fail-fast on hist2d/hexbin, AD-37 fix documentation. Updated test coverage to 451 (43 features, 21 invariance, 6 Verified). |
