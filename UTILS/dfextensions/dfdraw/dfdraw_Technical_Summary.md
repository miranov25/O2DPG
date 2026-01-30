# dfdraw Technical Summary

**Version:** Phase 13.6.G.DF (310 tests passing)  
**Date:** 2026-01-29  
**Audience:** Integration teams (Team2, architects, developers)  
**Purpose:** Technical reference for dfdraw capabilities, limitations, and integration

---

## ⚠️ Breaking Change Notice (Phase 13.6.G.DF)

**Standard deviation now uses population std (ddof=0) to match ROOT.**

| Affected Fields | Old Behavior | New Behavior |
|-----------------|--------------|--------------|
| `std` | ddof=1 (sample) | ddof=0 (population) |
| `std_x` | ddof=1 (sample) | ddof=0 (population) |
| `std_y` | ddof=1 (sample) | ddof=0 (population) |

**Impact:** Values ~6% smaller. Matches ROOT TTree::Draw exactly.

---

## Executive Summary

**dfdraw** is a declarative plotting library for pandas DataFrames with ROOT-like syntax.

**Key characteristics:**
- Expression-based (like TTree::Draw)
- Returns figures + statistics
- Batch-oriented for QA workflows
- Minimal styling (intentionally)
- No magic aggregation
- **ROOT-compatible statistics** (Phase 13.6.G.DF)

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
    'font.size': 12,
    'stats.robust': True,  # NEW: Phase 13.6.G.DF
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

### 3.4 Legends

```python
# Automatic for group_by
drawer.hist('x', group_by='charge')  # Legend added

# Manual control
fig, ax, stats = drawer.hist('x', group_by='charge')
ax.legend(loc='upper right', fontsize=12)
```

### 3.5 Grid

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
    'std': 1.015,         # float: standard deviation (population, ddof=0)
    'min': -3.456,        # float: minimum
    'max': 3.789          # float: maximum
}
```

**2D plot stats (scatter, profile, hist2d):**
```python
stats = {
    'n': 10000,           # Counts both-valid pairs only (Phase 13.6.G.DF)
    'mean_x': 0.01,
    'mean_y': 0.02,
    'std_x': 1.0,         # Population std (ddof=0)
    'std_y': 1.0,         # Population std (ddof=0)
    'corr': 0.85,         # Correlation coefficient
}
```

**Robust stats (Phase 13.6.G.DF, when `robust=True`):**
```python
stats = {
    'n': 10000,
    'mean': 0.023,
    'std': 1.015,
    'median': 0.018,      # 50th percentile
    'q25': -0.67,         # 25th percentile
    'q75': 0.69,          # 75th percentile
    'mad': 0.68,          # Median absolute deviation
}
```

**Warning:** Different plot types have different stats fields. Always use `stats.get('key')` or check `'key' in stats`.

### 4.2 Statistics Box (Phase 12.4b5 + 13.6.G.DF)

**Built-in statistics box:**

```python
from dfdraw.stats import format_stats_box

fig, ax, stats = drawer.hist('x')

# Add stats box to plot - auto-detects appropriate fields (Phase 13.6.G.DF)
text = format_stats_box(stats, plot_type='hist')
ax.text(0.95, 0.95, text, transform=ax.transAxes, 
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
```

**Default fields by plot type (Phase 13.6.G.DF):**

| Plot Type | Default Fields |
|-----------|---------------|
| `hist` | n, mean, std |
| `hist` (robust) | n, median, mad |
| `hist2d` | n, mean_x, mean_y, std_x, std_y, corr |
| `scatter` | n, mean_x, mean_y |
| `profile` | n, mean_x, mean_y |
| `hexbin` | n, mean_x, mean_y |

**Customization:**
```python
# Custom stats selection
format_stats_box(stats, fields=['n', 'mean', 'std', 'median'])

# With expected values comparison
drawer.add_statistics_box(ax, values,
                         expected_mean=0.0, expected_std=1.0)
```

### 4.3 Range-Aware Statistics (Phase 13.6.G.DF)

**Stats computed only within specified range:**

```python
from dfdraw.stats import compute_stats

# 1D: range_x filters the variable
stats = compute_stats(df, 'x', range_x=(0, 100))

# 2D: both ranges applied
stats = compute_stats(df, 'y', 'x', range_x=(0, 10), range_y=(-5, 5))
```

**Semantics:**
- Inclusive boundaries: `[min, max]`
- Matches ROOT TTree::Draw behavior
- Empty range returns all fields as NaN

### 4.4 Robust Statistics (Phase 13.6.G.DF)

**For non-Gaussian distributions:**

```python
from dfdraw.stats import compute_stats

# Request robust stats
stats = compute_stats(df, 'x', robust=True)
# Returns: n, mean, std, min, max, median, q25, q75, mad

# Global setting
set_style({'stats.robust': True})
```

**Available robust fields:**
- `median`: 50th percentile
- `q25`: 25th percentile  
- `q75`: 75th percentile
- `mad`: Median Absolute Deviation = median(|x - median(x)|)

**Note:** For 2D plots, robust stats apply to y-axis only.

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
fig, ax, stats = drawer.hist('x', bins=50)

# Add Gaussian overlay
drawer.add_reference_overlay(
    ax,
    func='gaussian',
    mu=0, sigma=1,
    color='red',
    linestyle='--',
    label='Expected'
)
```

**Available reference types:**
- `'gaussian'` - Normal distribution
- Callable - Custom function `f(x) -> y`

---

## 6. Current Defaults

### 6.1 Figure Settings

```python
# From matplotlib defaults (can be overridden via set_style)
DEFAULT_STYLE = {
    'figure.figsize': (8, 6),
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'hist.bins': 50,
    'hist.alpha': 0.7,
    'scatter.alpha': 0.7,
    'scatter.size': 50,
    'stats.show': False,
    'stats.position': 'upper right',
    'stats.robust': False,       # NEW: Phase 13.6.G.DF
    'colors.palette': 'tab10',
}
```

### 6.2 Statistics Defaults (Phase 13.6.G.DF)

| Setting | Default | Description |
|---------|---------|-------------|
| `stats.show` | False | Show stats box automatically |
| `stats.position` | 'upper right' | Stats box position |
| `stats.robust` | False | Use robust stats (median, MAD) for 1D |

---

## 7. Design Philosophy

### 7.1 What dfdraw IS

- **Visualization backend:** Converts DataFrames to plots
- **ROOT-style API:** Expression-based, one-liner syntax
- **Statistics provider:** Returns computed stats with every plot
- **Batch processor:** Generates multiple plots from specs
- **Style system:** Global + per-plot customization

### 7.2 What dfdraw is NOT

- **Analysis framework:** Use pandas/numpy for computation
- **Fitting library:** Use scipy, iminuit, etc.
- **Interactive dashboard:** Use RootInteractive for that
- **Schema validator:** No automatic validation
- **Aggregation engine:** Pre-aggregate data, then plot

### 7.3 Design Decisions

| Decision | Rationale |
|----------|-----------|
| No automatic fitting | Separation of concerns |
| No schema validation | Maximum flexibility |
| matplotlib backend | Publication-ready, universal |
| Immediate pandas conversion | Compatibility with eval/query |
| Population std (ddof=0) | ROOT compatibility |

---

## 8. Error Handling

### 8.1 Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'column'` | Column not in DataFrame | Check column names |
| `SyntaxError` in expression | Invalid pandas eval syntax | Simplify expression |
| Empty plot | All data filtered out | Check selection |
| NaN in stats | No valid data points | Check range/selection |

### 8.2 Batch Mode Errors

```python
results = drawer.draw_batch(specs, on_error='skip')

# Check for errors
for name, result in results.items():
    if 'error' in result:
        print(f"{name}: {result['error']}")
```

---

## 9. Limitations

### 9.1 By Design (Won't Fix)

| Limitation | Reason |
|------------|--------|
| No automatic fitting | Separate concern |
| No 3D plots | Out of scope |
| No animation | Out of scope |
| No interactive widgets | matplotlib backend |
| No schema validation | Flexibility |

### 9.2 Current Limitations (May Address)

| Limitation | Status | Notes |
|------------|--------|-------|
| ~~2D stats don't adapt to plot type~~ | ✅ Fixed (13.6.G.DF) | Auto-detects defaults |
| ~~Stats ignore range parameter~~ | ✅ Fixed (13.6.G.DF) | Range-aware stats |
| ~~No robust statistics~~ | ✅ Fixed (13.6.G.DF) | median, q25, q75, mad |
| No RootInteractive backend | Planned | Future phase |
| No Plotly backend | Maybe | Propose with use case |

---

## 10. Performance

### 10.1 Complexity

| Operation | Complexity |
|-----------|------------|
| Histogram | O(n) |
| Scatter | O(n) |
| Profile | O(n) |
| Expression eval | O(n) per expression |
| Statistics | O(n) |

### 10.2 Memory

| Component | Memory |
|-----------|--------|
| PyArrow conversion | 1.2-1.5× data size |
| matplotlib figure | 1-5 MB per plot |
| Statistics dict | Negligible |

### 10.3 Recommendations

| Dataset Size | Recommendation |
|--------------|----------------|
| < 10k rows | Any plot type |
| 10k-100k rows | Prefer hist2d over scatter |
| > 100k rows | Use hexbin |
| > 1M rows | Pre-aggregate or sample |

---

## 11. Integration

### 11.1 With AliasDataFrame

**Duck-typed integration:**
```python
from AliasDataFrame import AliasDataFrame
from dfdraw import DFDraw

aDF = AliasDataFrame(df)
aDF.set_axis_title('pt', 'p_{T} [GeV/c]')

drawer = DFDraw(aDF)
drawer.draw('pt')  # Axis title applied automatically
```

**Features:**
- Automatic axis titles
- Alias resolution
- Lazy evaluation support
- Statistics computation

### 11.2 With PyArrow (Phase 13.1.DF)

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

### 11.3 With RDataFrameDSL (Future)

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

## 12. API Surface (Quick Reference)

### 12.1 Core Methods

```python
from dfdraw import DFDraw

drawer = DFDraw(df)

# Primary methods
drawer.draw(expr, **kwargs) -> (fig, ax, stats)
drawer.hist(expr, **kwargs) -> (fig, ax, stats)
drawer.scatter(expr, **kwargs) -> (fig, ax, stats)
drawer.profile(expr, **kwargs) -> (fig, ax, stats)
drawer.hist2d(expr, **kwargs) -> (fig, ax, stats)
drawer.hexbin(expr, **kwargs) -> (fig, ax, stats)

# Batch processing
drawer.draw_batch(specs, **kwargs) -> dict

# Properties
drawer.backend -> str  # 'pandas' or 'pyarrow'
drawer.memory_info() -> dict
```

### 12.2 Style Functions

```python
from dfdraw import set_style, get_style, list_styles

set_style(style_dict)    # Set global style
get_style() -> dict      # Get current style
list_styles() -> list    # List available styles
```

### 12.3 Statistics Functions (Phase 13.6.G.DF Updated)

```python
from dfdraw.stats import compute_stats, format_stats_box, get_default_stats_fields

# Compute stats with range filtering and robust options
compute_stats(df, y_col, x_col=None, group_by=None,
              range_x=None, range_y=None, robust=False) -> DataFrame

# Format stats for display
format_stats_box(stats, fields=None, plot_type=None) -> str

# Get default fields for plot type
get_default_stats_fields(plot_type, robust=False) -> List[str]
```

### 12.4 Annotation Methods

```python
drawer.add_statistics_box(ax, values, position='upper right',
                         expected_mean=None, expected_std=None)
drawer.add_reference_overlay(ax, func='gaussian', mu=0, sigma=1)
```

---

## 13. Common Parameters

### 13.1 All Plot Types

```python
expr: str               # Expression to plot
ax: Axes               # Matplotlib axes (optional)
selection: str         # Data filter
entry_begin: int       # Start entry (0-indexed)
entry_end: int         # End entry (exclusive)
entry_mask: ndarray    # Boolean mask
group_by: str          # Column for grouping/overlay
stats: bool or list    # Show statistics (NEW: auto-detects fields)
```

### 13.2 Histogram-Specific

```python
bins: int or array     # Number of bins or edges
range: tuple           # (min, max) range - stats respect this (Phase 13.6.G.DF)
color: str             # Bar color
alpha: float           # Transparency [0,1]
edgecolor: str         # Bin edge color
```

### 13.3 Scatter-Specific

```python
color: str or array    # Point color or third variable
s: float or array      # Point size
marker: str            # Marker style ('o', 's', '^', etc.)
alpha: float           # Transparency
cmap: str              # Colormap (when color is array)
```

### 13.4 Profile-Specific

```python
bins: int              # Number of x bins
marker: str            # Point marker
linestyle: str         # Line style
capsize: float         # Error bar cap size
```

---

## 14. Answers to Team 2 Questions

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

**Yes:**
- Automatic via `group_by='column'`
- Manual via repeated draws on same axes
- Works for: hist, scatter, profile

**Not for:** hist2d, hexbin (use faceting instead)

### Q4: Does it support statistics box?

**Yes (Enhanced in Phase 13.6.G.DF):**
```python
# Auto-detects appropriate fields for plot type
drawer.hist('x', stats=True)
drawer.hist2d('y:x', stats=True)  # Shows 2D-appropriate fields

# Robust stats
set_style({'stats.robust': True})
drawer.hist('x', stats=True)  # Shows n, median, mad
```

**Shows:** Auto-detected by plot type, or customizable

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

**For Phase 13.6.H (debug notebooks):**

- ✅ All plot types supported
- ✅ Size control available
- ✅ Statistics display available
- ✅ Range-aware stats (Phase 13.6.G.DF)
- ✅ Robust statistics (Phase 13.6.G.DF)
- ⚠️ Default size may be large for notebooks (easy to fix)
- ❌ No interactive widgets (matplotlib backend dependent)
- ❌ No automatic fitting overlays (design choice)

**Recommendation:** Use `set_style({'figure.figsize': (6, 4)})` at notebook start

### Q8: Does std match ROOT? (NEW)

**Yes (Phase 13.6.G.DF):**
- `std`, `std_x`, `std_y` now use population std (ddof=0)
- Matches ROOT TTree::Draw exactly
- **Breaking change:** Values ~6% smaller than before

---

## 15. Testing Coverage

**Test suite:** 310 tests (100% passing)

**Coverage areas:**
- All plot types: hist, scatter, profile, hist2d, hexbin ✅
- Expression evaluation ✅
- Selection and filtering ✅
- Group-by overlays ✅
- Batch processing ✅
- PyArrow integration ✅
- Stats computation ✅
- Range-aware stats ✅ (Phase 13.6.G.DF)
- Robust statistics ✅ (Phase 13.6.G.DF)
- Error handling ✅

**Integration tests:**
- AliasDataFrame integration ✅
- Lazy loading ✅
- Subframe joins ✅

**Performance tests:**
- Large datasets (100k+ rows) ✅
- Memory overhead (PyArrow conversion) ✅

---

## 16. Documentation References

**Primary documentation:**
- `docs/README.md` - User guide
- `docs/API_REFERENCE.md` - Complete API reference
- `docs/PHASE_HISTORY.md` - Development history

**Integration examples:**
- `AliasDataFrame/tutorials/drawing/` - Working examples
- `AliasDataFrame/tutorials/cheatsheets/draw_cheatsheet.md` - Quick reference

**Tests as documentation:**
- `tests/test_histogram.py` - Histogram examples
- `tests/test_scatter.py` - Scatter examples
- `tests/test_pyarrow_input.py` - PyArrow integration
- `tests/test_stats_enhancements.py` - Statistics examples (Phase 13.6.G.DF)

---

## 17. Support and Questions

**For integration questions:**
- Contact: Team3-dfdraw
- Documentation: This file + docs/README.md
- Examples: AliasDataFrame tutorials

**For bug reports:**
- Check: test suite (310 tests as regression suite)
- Verify: Minimal reproduction case
- Report: With test.log output

**For feature requests:**
- Check: Design limitations (Section 9.1)
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
| Range-aware stats? | Yes | Phase 13.6.G.DF |
| Robust statistics? | Yes | Phase 13.6.G.DF |

---

## Appendix B: Version History

**Phase 13.6.G.DF (Current):**
- ⚠️ Breaking: std uses population std (ddof=0)
- Range-aware statistics
- Robust statistics (median, q25, q75, mad)
- Auto-detect default stats fields by plot type
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
- Statistics: O(n)

**Memory:**
- PyArrow conversion: ~1.2-1.5× data size
- matplotlib figures: ~1-5 MB per plot
- Batch mode: Parallelizable (future enhancement)

**Scalability:**
- Tested up to 1M rows (scatter)
- Recommended: hexbin for >100k points
- Batch mode: Tested with 50+ plots

---

## Appendix D: ROOT Compatibility (Phase 13.6.G.DF)

| Feature | ROOT | dfdraw | Match |
|---------|------|--------|-------|
| Expression syntax | `"y:x"` | `"y:x"` | ✅ |
| Selection | `"x>0"` | `selection="x>0"` | ✅ |
| Statistics box | `gStyle->SetOptStat()` | `stats=True` | ✅ |
| Profile plot | `"prof"` option | `type="profile"` | ✅ |
| Population std | ddof=0 | ddof=0 | ✅ |
| Range-filtered stats | Yes | Yes (13.6.G.DF) | ✅ |
| 2D n (both-valid) | Yes | Yes (13.6.G.DF) | ✅ |

---

**Document Status:** Ready for GitLab  
**Maintainer:** Team3-dfdraw  
**Last Updated:** 2026-01-29  
**Version:** Phase 13.6.G.DF
