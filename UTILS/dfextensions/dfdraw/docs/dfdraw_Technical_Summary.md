# dfdraw Technical Summary

**Version:** Phase 13.14.DF v1.0 (399 tests passing)  
**Date:** 2026-03-28  
**Audience:** Integration teams (Team2, architects, developers)  
**Purpose:** Technical reference for dfdraw capabilities, limitations, and integration

---

## Executive Summary

**dfdraw** is a declarative plotting library for pandas DataFrames with ROOT-like syntax.

**Key characteristics:**
- Expression-based (like TTree::Draw)
- Returns figures + statistics
- Batch-oriented for QA workflows
- Minimal styling (intentionally)
- No magic aggregation

**Design principle:** Visualization backend, not analysis framework.

---

## 1. Supported Plot Types

### 1.1 Available Plot Types

| Type | Method | Expression Syntax | Use Case |
|------|--------|------------------|----------|
| **Histogram (1D)** | `hist()` | `'x'` | Distribution of single variable |
| **Scatter (2D)** | `scatter()` | `'y:x'` | Individual points, correlations |
| **Profile (2D)** | `profile()` | `'y:x'` | Mean y per x bin with error bars |
| **Hist2D** | `hist2d()` | `'y:x'` | Density heatmap |
| **Hexbin** | `hexbin()` | `'y:x'` | Hexagonal binning (large datasets) |

### 1.2 Dispatch via draw()

```python
# Unified entry point - auto-dispatches
drawer.draw('x')           # → hist
drawer.draw('y:x')         # → scatter
drawer.draw('y:x', type='profile')  # → profile

# Or explicit methods
drawer.hist('x')
drawer.scatter('y:x')
drawer.profile('y:x')
```

**Expression syntax:**
- 1D: `'column'` or `'expression'`
- 2D: `'y:x'` (y vs x, y on vertical axis)

---

## 2. Size Control

### 2.1 Figure Size

**Default:** Controlled by matplotlib style (typically 8×6 inches)

**Control via:**

#### Option 1: Set style globally
```python
from dfdraw import set_style

set_style({'figure.figsize': (10, 8)})
drawer = DFDraw(df)
drawer.draw('x')  # Uses 10×8
```

#### Option 2: Pass to individual plot
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
drawer = DFDraw(df)
drawer.draw('x', ax=ax)
```

#### Option 3: Modify returned figure
```python
fig, ax, stats = drawer.draw('x')
fig.set_size_inches(10, 8)
```

**Available in:** All plot types

**DPI Control:**
```python
fig.savefig('plot.png', dpi=150)  # Control at save time
```

### 2.2 Subplot Layout (Faceting)

```python
# Automatic grid layout
drawer.draw('x', group_by='category')  # Creates subplots

# Manual control
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    drawer.draw('x', selection=f'category=={i}', ax=ax)
```

---

## 3. Styling Options

### 3.1 Style System

**Architecture:**
- Global style dictionary
- Per-plot override via kwargs
- Matplotlib pass-through

**Available styles:** Default only (extensible)

```python
from dfdraw import set_style, get_style, list_styles

# View current style
style = get_style()
print(style.keys())  # All matplotlib rcParams

# Modify globally
set_style({
    'figure.figsize': (10, 8),
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'font.size': 12
})

# Reset to default
set_style(None)
```

### 3.2 Color Options

#### Histogram colors
```python
drawer.hist('x', color='blue', alpha=0.7, edgecolor='black')
```

#### Scatter colors
```python
# Fixed color
drawer.scatter('y:x', color='red', alpha=0.5)

# Color by third variable
drawer.scatter('y:x', color='z')  # Automatic colormap

# Custom colormap
drawer.scatter('y:x', color='z', cmap='viridis')
```

#### Group colors (overlays)
```python
# Automatic colors per group
drawer.hist('x', group_by='category')
# Each category gets different color from cycle
```

### 3.3 Labels and Titles

```python
# Automatic from column names
drawer.hist('x')  # xlabel='x'

# Manual override
drawer.hist('x', xlabel='Custom X', ylabel='Events', title='My Plot')

# Modify after creation
fig, ax, stats = drawer.hist('x')
ax.set_xlabel('Custom Label', fontsize=14)
ax.set_title('Title', fontsize=16)
```

### 3.4 Auto-Title (Phase 13.12.DF v1.2)

```python
# Automatic title from plot parameters
drawer.profile('y:x', auto_title=True)
# → Title: "y vs x"

# With group and selection
drawer.profile('y:x', group_by='sector', selection='pt>0.5',
               auto_title=True)
# → Title: "y vs x  group:sector"
# → Subtitle: "pt>0.5" (italic, smaller font)

# Partial auto-title
drawer.profile('y:x', auto_title='expr')       # Expression only
drawer.profile('y:x', auto_title='expr+group')  # Expression + group
drawer.profile('y:x', auto_title='expr+sel')    # Expression + selection

# Enable globally via style
set_style({'auto_title': True})
```

### 3.5 Legends

```python
# Automatic for group_by
drawer.hist('x', group_by='charge')  # Legend added

# Manual control
fig, ax, stats = drawer.hist('x', group_by='charge')
ax.legend(loc='upper right', fontsize=12)
```

### 3.6 Grid

```python
fig, ax, stats = drawer.hist('x')
ax.grid(True, alpha=0.3)
```

---

## 4. Statistics Display

### 4.1 Statistics Dictionary

**All plots return stats dict:**

```python
fig, ax, stats = drawer.hist('x')

# 1D histogram stats
stats = {
    'n': 10000,           # int: entry count
    'mean': 0.023,        # float: mean value
    'std': 1.015,         # float: standard deviation
    'min': -3.456,        # float: minimum
    'max': 3.789          # float: maximum
}
```

**2D plot stats (scatter, profile, hist2d):**
```python
stats = {
    'n': 10000,
    'mean_x': 0.01,
    'mean_y': 0.02,
    'std_x': 1.0,
    'std_y': 1.0,
    'corr': 0.85,         # Correlation coefficient
    # ... other fields
}
```

**Warning:** Different plot types have different stats fields. Always use `stats.get('key')` or check `'key' in stats`.

### 4.2 Statistics Box (Phase 12.4b5)

**Built-in statistics box:**

```python
from dfdraw.stats import add_statistics_box

fig, ax, stats = drawer.hist('x')

# Add stats box to plot
add_statistics_box(
    ax, 
    stats,
    position='upper right',  # Or 'upper left', etc.
    fontsize=10
)
```

**Displays:** n, mean, std (configurable)

**Customization:**
```python
# Custom stats selection
add_statistics_box(ax, stats, 
                   fields=['n', 'mean', 'std'],
                   position='upper right')

# With expected values comparison
add_statistics_box(ax, stats,
                   expected={'mean': 0.0, 'std': 1.0})
```

---

## 5. Overlay and Multi-Plot Support

### 5.1 Overlays (Same Axes)

#### Via group_by
```python
# Automatic overlay - separate histograms, same axes
drawer.hist('x', group_by='category', bins=50)
```

**What happens:**
- Computes separate histogram per category
- Overlays on same axes
- Different colors per category
- Legend added automatically

**Supported for:**
- Histograms ✅
- Profiles ✅  
- Scatter (colors by group) ✅

**Not supported for:**
- hist2d (use faceting instead)
- hexbin (use faceting instead)

#### Via same=True (Phase 13.13.DF v1.0)

ROOT-like superposition:
```python
# Simple overlay — no need to capture ax
drawer.profile('y1:x')
drawer.profile('y2:x', same=True)
drawer.profile('y3:x', same=True)
# → Three curves, different colors, auto-labeled, legend shown
```

**Automatic features with same=True:**
- Auto-increment colors from palette (AD-16)
- Auto-generate labels from expression (AD-17)
- Append to title when `auto_title=True` (AD-18)
- Legend shown automatically

**Precedence rules:**
- `ax=` always wins over `same=True`
- Explicit `color=`, `label=`, `title=` override auto-features
- `title=` replaces entire title (no append)

**Mixed plot types:**
```python
# Profile overlay on 2D histogram
drawer.hist2d('y:x', norm='log')
drawer.profile('y:x', same=True)
# → Profile line on top of 2D histogram
```

**Instance tracking (AD-15):**
- Uses `self._last_ax` to track last axes
- Falls back to `plt.gca()` when no previous draw
- For safe AliasDataFrame integration, AD-37 requires caching DFDraw instance

#### Manual overlay
```python
fig, ax = plt.subplots()

# Draw multiple times on same axes
drawer.draw('x', selection='charge==1', ax=ax, label='Positive', alpha=0.5)
drawer.draw('x', selection='charge==-1', ax=ax, label='Negative', alpha=0.5)

ax.legend()
```

### 5.2 Faceting (Separate Subplots)

**Automatic faceting:**
```python
# Creates subplot grid automatically
drawer.draw('x', group_by='sector', facet=True)
```

**Manual subplot layout:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

drawer.draw('x', selection='charge==1', ax=axes[0,0])
drawer.draw('y', selection='charge==1', ax=axes[0,1])
drawer.draw('x', selection='charge==-1', ax=axes[1,0])
drawer.draw('y', selection='charge==-1', ax=axes[1,1])

plt.tight_layout()
```

### 5.3 Reference Overlays (Phase 12.4b5)

**Add reference distributions:**

```python
from dfdraw.stats import add_reference_overlay

fig, ax, stats = drawer.hist('x', bins=50)

# Add Gaussian overlay
add_reference_overlay(
    ax,
    reference_type='gaussian',
    params={'mean': 0, 'sigma': 1},
    color='red',
    linestyle='--',
    label='Expected'
)
```

**Available reference types:**
- `'gaussian'` - Normal distribution
- `'callable'` - Custom function

---

## 6. Current Defaults

### 6.1 Figure Settings

```python
# From matplotlib defaults (can be overridden via set_style)
figure.figsize: (8, 6)      # inches
figure.dpi: 100             # screen display
savefig.dpi: 150            # saved figures
```

### 6.2 Histogram Defaults

```python
bins: 50                    # Number of bins
range: None                 # Auto from data
histtype: 'stepfilled'      # Bar style
alpha: 0.7                  # Transparency
edgecolor: 'black'          # Bin edge color
```

### 6.3 Scatter Defaults

```python
s: 20                       # Point size (marker size)
alpha: 0.6                  # Transparency
marker: 'o'                 # Circle markers
```

### 6.4 Profile Defaults

```python
bins: 50                    # X-axis bins
marker: 'o'                 # Point marker
linestyle: '-'              # Connecting line
capsize: 3                  # Error bar caps
```

### 6.5 Hexbin Defaults

```python
gridsize: 50                # Hexagon grid size
cmap: 'viridis'             # Colormap
mincnt: 1                   # Min count to show
```

---

## 7. Selection and Filtering

### 7.1 Selection Syntax

**String expression (pandas query syntax):**
```python
drawer.draw('x', selection='pt > 0.5')
drawer.draw('x', selection='(pt > 0.5) & (eta < 1.0)')
drawer.draw('x', selection='(charge == 1) | (charge == -1)')
```

**Important:** Use Python syntax (`&`, `|`), NOT C++ syntax (`&&`, `||`)

### 7.2 Entry Range

```python
# Quick iteration - first 10k entries
drawer.draw('x', entry_begin=0, entry_end=10000)

# Skip first 1000
drawer.draw('x', entry_begin=1000)
```

### 7.3 Boolean Mask

```python
import numpy as np

mask = (df['pt'] > 1.0) & (df['charge'] != 0)
drawer.draw('x', entry_mask=mask.values)
```

---

## 8. Batch Processing

### 8.1 Dict Format (Original)

```python
specs = {
    'hist_x': {
        'expr': 'x',
        'bins': 50,
        'title': 'X Distribution'
    },
    'scatter_yx': {
        'expr': 'y:x',
        'type': 'scatter'
    },
    'profile_pt': {
        'expr': 'pt:eta',
        'type': 'profile',
        'bins': 100
    }
}

results = drawer.draw_batch(specs, verbose=True)

# Access individual results
fig, ax, stats = results['hist_x']
```

### 8.2 List Format — Group-Based with Defaults Hierarchy (Phase 13.14.DF v1.0)

```python
specs = [{
    'name': 'residuals_qa',
    'suptitle': 'ITS-TPC Residuals QA',
    'ncols': 2,
    'figsize': (16, 12),
    'savefig': 'residuals_qa.png',
    'defaults': {
        'type': 'profile',
        'bins': 152,
        'min_entries': 250,
        'auto_title': True,
        'linestyle': 'none',
        'selection': 'group_count>100&row<152',
    },
    'plots': [
        {'expr': 'dy:row', 'group_by': 'mP4_bin'},
        {'expr': 'dy:row', 'group_by': 'mP4', 'group_by_quantiles': 10},
        {'expr': 'dz:row', 'group_by': 'mP4', 'group_by_quantiles': 10},
        {'expr': 'dz:row'},  # inherits all defaults
    ]
}]

results = drawer.draw_batch(specs, verbose=2)
# verbose=2 prints merged parameters per plot (debug mode)
```

**Option hierarchy (more local wins):**
```
kwargs < draw_batch defaults= < group defaults < plot spec
```

**Group-level keys** (figure structure only):
`name`, `suptitle`, `layout`, `ncols`, `figsize`, `savefig`, `sharex`, `sharey`, `plots`, `defaults`

**All draw parameters** go in `defaults` or in individual plot specs.

**Layout control:**
```python
# Auto-compute rows from ncols
'ncols': 2  # 4 plots → 2×2

# Explicit grid
'layout': (3, 2)  # overrides ncols

# Figure size
'figsize': (16, 10)  # whole figure, not per-subplot
```

**Overlay within group:**
```python
'plots': [
    {'expr': 'y:x', 'type': 'hist2d', 'norm': 'log'},
    {'expr': 'y:x', 'type': 'profile', 'same': True},  # overlays on previous
    {'expr': 'z:x', 'type': 'profile'},                  # new subplot
]
```

### 8.3 Batch Saving

```python
results = drawer.draw_batch(specs, save_dir='plots/')
# Automatically saves all plots to plots/ directory
```

### 8.4 Error Handling

```python
results = drawer.draw_batch(specs, verbose=True)

# Check summary
print(results['_summary'])
# {'success': 2, 'failed': 1, 'errors': {...}}

# Access failed plots
if results['_summary']['failed'] > 0:
    print(results['_summary']['errors'])
```

### 8.5 Verbose Levels (Phase 13.14.DF)

| Value | Behavior |
|-------|----------|
| `False` / `0` | Silent |
| `True` / `1` | Progress — group names, save paths, summary |
| `2` | Debug — also prints merged parameters per plot |

---

## 9. Expression Evaluation

### 9.1 Supported Expressions

**Column references:**
```python
drawer.draw('x')            # Direct column
drawer.draw('y:x')          # Two columns
```

**NumPy operations:**
```python
drawer.draw('x**2')                    # Power
drawer.draw('np.sqrt(x**2 + y**2)')    # Functions
drawer.draw('np.log(pt)')              # Logarithm
```

**Pandas operations:**
```python
# Uses pandas.eval() under the hood
drawer.draw('(x - mean_x) / std_x')    # Standardization
```

### 9.2 Limitations

**Cannot do:**
- ❌ Column creation in expression: `drawer.draw('pt = sqrt(px**2 + py**2)')` 
- ❌ Aggregations: `drawer.draw('sum(x)')`
- ❌ Group operations: `drawer.draw('x.groupby(y).mean()')`

**Workaround:** Add column to DataFrame first:
```python
df['pt'] = np.sqrt(df['px']**2 + df['py']**2)
drawer.draw('pt')
```

---

## 10. Profile Enhancements (Phase 13.12.DF)

### 10.1 Return Profile Data (F1)

```python
fig, ax, stats = drawer.profile('y:x', return_data=True)
profile_df = stats['profile_data']
# DataFrame with: x_center, x_low, x_high, y_mean, y_std, y_sem, count
```

### 10.2 Minimum Entries Filter (F2)

```python
# Suppress bins with < 3 entries (default)
drawer.profile('y:x', min_entries=3)

# More restrictive
drawer.profile('y:x', min_entries=20)
```

### 10.3 Auto-Bin Float Group-By (F3)

```python
# Equal-width bins for float grouping variable
drawer.profile('y:x', group_by='mP3', group_by_bins=8)

# Equal-count quantile bins
drawer.profile('y:x', group_by='mP3', group_by_quantiles=5)
```

### 10.4 Sorted Groups (F4)

```python
# Sorted legend order (default)
drawer.profile('y:x', group_by='sector', sort_groups=True)
```

### 10.5 Weighted Statistics (Phase 13.12.DF v1.1)

```python
# Weighted mean/std/sem
drawer.profile('y:x', weights='w')
```

---

## 11. Known Limitations

### 11.1 Design Limitations (Intentional)

| Feature | Status | Reason |
|---------|--------|--------|
| **Schema validation** | ❌ Not supported | Flexibility - accepts any DataFrame |
| **Automatic fitting** | ❌ Not supported | Keep visualization separate from analysis |
| **Smart aggregation** | ❌ Not supported | Explicit is better than implicit |
| **Table joins** | ❌ Not supported | Upstream responsibility (AliasDataFrame) |
| **Lazy evaluation** | ❌ Not supported | Works on materialized DataFrames |

### 11.2 Technical Limitations

**Memory:**
- Large scatter plots (>1M points) may be slow
- Recommendation: Use hexbin for >100k points

**Performance:**
- Expression evaluation in Python (not compiled)
- For performance-critical workflows, materialize columns first

**Stats computation:**
- Stats are diagnostic, not analytical
- For advanced statistics, use scipy/statsmodels

### 11.3 Current Bugs / Issues

**None identified in Phase 13.14.DF** (399/399 tests passing)

**Known integration gap:**
- AliasDataFrame's `draw_figures()` reimplements the drawing loop independently.
  Group-level `defaults` cascade, `same=True`, `layout`, and `figsize` do not
  work through `aDF.draw_figures()`. Workaround: use `DFDraw(aDF.df).draw_batch(specs)`
  directly. Fix planned in Phase 13.13.ADF v1.0 (ADF delegates to dfdraw).

---

## 12. Integration Notes

### 12.1 With AliasDataFrame

**Handoff point:** `adf.draw()` → `DFDraw(df)`

**AliasDataFrame handles:**
- Table joins (subframes)
- Alias resolution
- Lazy branch loading

**dfdraw handles:**
- Visualization
- Statistics computation

**See:** [Integration Patterns](dfdraw_README.md#integration-patterns) for detailed examples

### 12.2 With PyArrow (Phase 13.1.DF)

**Supported:**
```python
import pyarrow as pa

table = pa.Table.from_pandas(df)
drawer = DFDraw(table)  # Automatically converts to pandas
drawer.draw('x')
```

**Backend detection:**
```python
drawer.backend  # 'pyarrow' or 'pandas'
drawer.memory_info()  # Diagnostic information
```

**Design:** Immediate conversion to pandas for compatibility with pandas.eval(), query(), groupby()

### 12.3 With RDataFrameDSL (Future)

**Not yet integrated.** Planned for future phase.

**Expected pattern:**
```python
# DSL compiles expressions
dsl = RDataFrameDSL()
dsl.define("pt", "sqrt(px*px + py*py)")

# Execute → DataFrame
df = dsl.execute(rdf)

# Visualize
drawer = DFDraw(df)
drawer.draw('pt')
```

---

## 13. API Surface (Quick Reference)

### 13.1 Core Methods

```python
from dfdraw import DFDraw

drawer = DFDraw(df)

# Primary methods — all support same=True (Phase 13.13.DF)
drawer.draw(expr, **kwargs) -> (fig, ax, stats)
drawer.hist(expr, **kwargs) -> (fig, ax, stats)
drawer.scatter(expr, **kwargs) -> (fig, ax, stats)
drawer.profile(expr, **kwargs) -> (fig, ax, stats)
drawer.hist2d(expr, **kwargs) -> (fig, ax, stats)
drawer.hexbin(expr, **kwargs) -> (fig, ax, stats)

# Batch processing — dict or list format (Phase 13.14.DF)
drawer.draw_batch(specs, **kwargs) -> dict

# Properties
drawer.backend -> str  # 'pandas' or 'pyarrow'
drawer.memory_info() -> dict
```

### 13.2 Style Functions

```python
from dfdraw import set_style, get_style, list_styles

set_style(style_dict)    # Set global style
get_style() -> dict      # Get current style
list_styles() -> list    # List available styles
```

### 13.3 Statistics Functions

```python
from dfdraw.stats import add_statistics_box, add_reference_overlay

add_statistics_box(ax, stats, position='upper right')
add_reference_overlay(ax, reference_type='gaussian', params={...})
```

---

## 14. Common Parameters

### 14.1 All Plot Types

```python
expr: str               # Expression to plot
ax: Axes               # Matplotlib axes (optional)
selection: str         # Data filter
entry_begin: int       # Start entry (0-indexed)
entry_end: int         # End entry (exclusive)
entry_mask: ndarray    # Boolean mask
group_by: str          # Column for grouping/overlay
same: bool             # Overlay on last axes (Phase 13.13.DF)
auto_title: bool|str   # Automatic title (Phase 13.12.DF v1.2)
```

### 14.2 Histogram-Specific

```python
bins: int or array     # Number of bins or edges
range: tuple           # (min, max) range
color: str             # Bar color
alpha: float           # Transparency [0,1]
edgecolor: str         # Bin edge color
```

### 14.3 Scatter-Specific

```python
color: str or array    # Point color or third variable
s: float or array      # Point size
marker: str            # Marker style ('o', 's', '^', etc.)
alpha: float           # Transparency
cmap: str              # Colormap (when color is array)
```

### 14.4 Profile-Specific

```python
bins: int              # Number of x bins
marker: str            # Point marker
linestyle: str         # Line style
capsize: float         # Error bar cap size
return_data: bool      # Export profile DataFrame (Phase 13.12.DF)
min_entries: int       # Min entries per bin (Phase 13.12.DF)
group_by_bins: int     # Equal-width bins for group_by (Phase 13.12.DF)
group_by_quantiles: int # Quantile bins for group_by (Phase 13.12.DF)
sort_groups: bool      # Sort legend order (Phase 13.12.DF)
weights: str           # Weight column (Phase 13.12.DF v1.1)
```

---

## 15. Answers to Team 2 Questions

### Q1: Does dfdraw support figsize?

**Yes, three ways:**

1. **Global style:**
   ```python
   set_style({'figure.figsize': (10, 6)})
   ```

2. **Per-plot axes:**
   ```python
   fig, ax = plt.subplots(figsize=(12, 8))
   drawer.draw('x', ax=ax)
   ```

3. **Modify after creation:**
   ```python
   fig, ax, stats = drawer.draw('x')
   fig.set_size_inches(8, 6)
   ```

**For debug notebooks:** Recommend (6, 4) or (8, 5)

### Q2: What plot types are available?

**Five plot types:**
- hist (1D histogram)
- scatter (2D scatter)
- profile (2D mean with error bars)
- hist2d (2D density heatmap)
- hexbin (hexagonal binning)

**All accessible via:** `draw()` or explicit methods

### Q3: Does it support overlays?

**Yes, three ways:**
- Automatic via `group_by='column'`
- Via `same=True` for ROOT-like superposition (Phase 13.13.DF)
- Manual via repeated draws on same axes

**Works for:** hist, scatter, profile

**Not for:** hist2d, hexbin (use faceting instead, or overlay profile on top with `same=True`)

### Q4: Does it support statistics box?

**Yes (Phase 12.4b5):**
```python
from dfdraw.stats import add_statistics_box

fig, ax, stats = drawer.hist('x')
add_statistics_box(ax, stats, position='upper right')
```

**Shows:** n, mean, std (customizable)

### Q5: Color/style options?

**Yes:**
- Per-plot colors: `color='blue'`
- Third variable coloring: `color='z'`
- Colormaps: `cmap='viridis'`
- Transparency: `alpha=0.7`
- Global style: `set_style({...})`

**Matplotlib pass-through:** All matplotlib kwargs work

### Q6: Current default sizes?

**Figures:** (8, 6) inches  
**DPI:** 100 (screen), 150 (saved)  
**Markers:** 20 (scatter point size)  
**Bins:** 50 (histogram default)

**Override globally:**
```python
set_style({
    'figure.figsize': (6, 4),  # Smaller for notebooks
    'savefig.dpi': 100         # Lower resolution
})
```

### Q7: Known limitations?

**For Phase 13.6.G (debug notebooks):**

- ✅ All plot types supported
- ✅ Size control available
- ✅ Statistics display available
- ⚠️ Default size may be large for notebooks (easy to fix)
- ❌ No interactive widgets (matplotlib backend dependent)
- ❌ No automatic fitting overlays (design choice)

**Recommendation:** Use `set_style({'figure.figsize': (6, 4)})` at notebook start

---

## 16. Testing Coverage

**Test suite:** 399 tests (100% passing)

**Coverage areas:**
- All plot types: hist, scatter, profile, hist2d, hexbin ✅
- Expression evaluation ✅
- Selection and filtering ✅
- Group-by overlays ✅
- Batch processing (dict and list formats) ✅
- PyArrow integration ✅
- Stats computation ✅
- Error handling ✅
- same=True superposition (Phase 13.13.DF) ✅
- Defaults cascade and verbose levels (Phase 13.14.DF) ✅
- Profile enhancements: return_data, min_entries, group_by_bins, weights (Phase 13.12.DF) ✅

**Integration tests:**
- AliasDataFrame integration ✅
- Lazy loading ✅
- Subframe joins ✅

**Performance tests:**
- Large datasets (100k+ rows) ✅
- Memory overhead (PyArrow conversion) ✅

---

## 17. Documentation References

**Primary documentation:**
- `dfdraw_README.md` - User guide
- `dfdraw_api_summary.md` - Complete API reference
- `PHASE_HISTORY.md` - Development history

**Integration examples:**
- `AliasDataFrame/tutorials/drawing/` - Working examples
- `AliasDataFrame/tutorials/cheatsheets/draw_cheatsheet.md` - Quick reference

**Tests as documentation:**
- `tests/test_histogram.py` - Histogram examples
- `tests/test_scatter.py` - Scatter examples
- `tests/test_pyarrow_input.py` - PyArrow integration
- `tests/test_same.py` - same=True superposition examples
- `tests/test_batch_groups.py` - Batch group format examples

---

## 18. Support and Questions

**For integration questions:**
- Contact: Team3-dfdraw
- Documentation: This file + dfdraw_README.md
- Examples: AliasDataFrame tutorials

**For bug reports:**
- Check: test suite (399 tests as regression suite)
- Verify: Minimal reproduction case
- Report: With test.log output

**For feature requests:**
- Check: Design limitations (Section 11.1)
- Propose: Via specification document
- Discuss: Architecture team review

---

## Appendix A: Quick Decision Matrix

**"Should dfdraw do X?"**

| Question | Answer | Reason |
|----------|--------|--------|
| Plot types beyond 5 core types? | Maybe | Propose with use case |
| Automatic fitting? | No | Separate concern (analysis) |
| Schema validation? | No | Flexibility by design |
| Custom aggregations? | No | Use pandas first, then plot |
| Interactive widgets? | No | Matplotlib backend concern |
| 3D plots? | No | Out of scope |
| Animation? | No | Out of scope |
| Size control? | Yes | Already supported |
| Style control? | Yes | Already supported |
| Statistics display? | Yes | Already supported |

---

## Appendix B: Version History

**Phase 13.14.DF v1.0 (Current):**
- draw_batch group format with defaults hierarchy
- Subplot grid: ncols, layout, figsize, suptitle
- same=True within groups
- verbose=2 debug mode
- 399 tests passing

**Phase 13.13.DF v1.0:**
- same=True superposition for all draw methods
- Auto-increment colors, auto-labels, title append
- self._last_ax with plt.gca() fallback

**Phase 13.12.DF v1.0–v1.2:**
- Profile: return_data, min_entries, group_by_bins/quantiles, sort_groups
- Weighted profile statistics
- Auto-title system (auto_title parameter + style key)
- Interval label sorting fix for negative ranges

**Phase 13.6.G.DF:**
- Statistics enhancements: range-aware, robust, plot-type-aware defaults
- Breaking change: std uses ddof=0 (population) to match ROOT
- 310 tests passing

**Phase 13.1.DF:**
- PyArrow Table input support
- 264 tests passing
- Memory diagnostic methods
- Complete backward compatibility

**Phase 12.4b5:**
- Statistics box support
- Reference overlay support
- Enhanced stats computation

**Phase 6.8:**
- AliasDataFrame integration
- Duck-typed axis titles
- Batch plotting support

**See:** PHASE_HISTORY.md for complete history

---

## Appendix C: Performance Characteristics

**Time complexity:**
- Histogram: O(n) where n = data rows
- Scatter: O(n)
- Profile: O(n)
- Expression eval: O(n) per expression

**Memory:**
- PyArrow conversion: ~1.2-1.5× data size
- matplotlib figures: ~1-5 MB per plot
- Batch mode: Parallelizable (future enhancement)

**Scalability:**
- Tested up to 1M rows (scatter)
- Recommended: hexbin for >100k points
- Batch mode: Tested with 50+ plots

---

**Document Status:** Ready for GitLab  
**Maintainer:** Team3-dfdraw  
**Last Updated:** 2026-03-28  
**Version:** Phase 13.14.DF v1.0
