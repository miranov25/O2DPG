# dfdraw Technical Summary

**Version:** Phase 13.27.DF Commit 1 (663 tests passing, 62 features, 28+ invariance tests, 7 Verified)
**Date:** 2026-05-09
**Audience:** Integration teams (Team2, architects, developers)
**Purpose:** Technical reference for dfdraw capabilities, limitations, and integration

> **What's new since Phase 13.16.DF v1.0** — robust statistics extension (13.18); quantile rendering on profile (13.25, MultiGraph Phase A); N-channel framework with Algorithm A automatic visual-encoding assignment (13.26, MultiGraph Phase B); facet refactor through the channel framework (13.27 Commit 1, MultiGraph Phase D, profile-only); robust data handling with NaN/inf filter and hybrid autorange (13.28). Two FIX1 follow-ups pending (Phase 13.26 + 13.28; same bug class). See Appendix B for full version history.

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
- Vector (Phase 13.16.DF): `'[y1,y2,y3]:x'`, `'y:[x1,x2]'`, `'[y1,y2]:[x1,x2]'`

### 1.3 Vector Expression Syntax (Phase 13.16.DF v1.0)

Bracket-vector syntax draws multiple related curves/series in a single call,
overlaid on one axes. This is the architecturally-correct solution for the
AD-37 AliasDataFrame color-cycling bug: a single `DFDraw` instance handles
the full loop internally, so color/label continuity is preserved across the
full series.

**Broadcasting rules:**

| Pattern | Syntax | Result |
|---------|--------|--------|
| N:1 | `[y1,y2,y3]:x` | 3 series sharing x |
| 1:N | `y:[x1,x2,x3]` | 3 series sharing y |
| N:N | `[y1,y2]:[x1,x2]` | 2 paired series |
| N:M (N≠M) | `[y1,y2]:[x1,x2,x3]` | `ValueError` |
| 1D vector | `[y1,y2,y3]` | 3 overlaid histograms |

**Features:**
- Paren-aware split: `[max(a,b),max(c,d)]:x` parses correctly
- `vector_style` channel: `'color'` (default without group_by) or `'linestyle'` (default with group_by)
- `group_style` channel for second dimension when combining vector with `group_by`
- Per-pair stats: `stats()` returns `list[dict]` of length N for vector input
- `auto_title` defaults to `True` in vector mode
- Y-axis label: common-prefix rule (≥2 chars) or truncated bracket-list
- Fail-fast on `hist2d()` and `hexbin()` (single-density surfaces have no meaningful overlay)

**Supported on:** `draw()`, `profile()`, `hist()`, `scatter()`, `draw_batch()`

**Example — ITS layer residuals:**
```python
# Before (broken AD-37): only 2 colors appear instead of 6
aDF.draw("dd_dyITS0:staveITS", type='profile', bins=12)
aDF.draw("dd_dyITS1:staveITS", type='profile', bins=12, same=True)
# ... 6 times — each call creates fresh DFDraw, resetting color cycle

# After (Phase 13.16.DF): single call, correct 6 colors
aDF.draw("[dd_dyITS0,dd_dyITS1,dd_dyITS2,dd_dyITS3,dd_dyITS4,dd_dyITS5]:staveITS",
         selection="row==180 & isPrimITS==1",
         type='profile', bins=12)
```

**Invariance contract:** Vector path produces byte-identical axes state
(line count, colors, linestyles, labels, xdata/ydata) and per-pair stats
to a scalar `same=True` loop. Verified by 7 A≡B tests in
`tests/test_vector.py::TestVectorInvariance`.

### 1.4 Quantile Rendering on Profile (Phase 13.25.DF v1.3 — MultiGraph Phase A)

Per-bin quantile rendering on `profile()` for distribution-shape comparison
beyond mean ± std.

```python
# error_bars mode: symmetric pair without 0.5 → asymmetric error bars on central line
d.profile("y:x", quantiles=[0.16, 0.84])

# band mode: symmetric triple with 0.5 → fill_between
d.profile("y:x", quantiles=[0.16, 0.5, 0.84])

# nested-band auto-detection (Phase 13.26 AD-57): ≥4 non-0.5 entries
d.profile("y:x", quantiles=[0.05, 0.25, 0.5, 0.75, 0.95])  # 2 alpha-stacked regions

# central= overrides the line plotted under bars/band
d.profile("y:x", quantiles=[0.16, 0.84], central='median')
```

**New parameters on `profile()`:**
- `quantiles: Optional[List[float]]` — list of fractions in (0, 1)
- `central: Optional[str]` — `'mean'` (default), `'median'`, `'both'`, `'none'`
- `quantile_mode: str = "auto"` — `'auto'` dispatches via `_detect_quantile_mode()`; explicit override available with `'discrete'`, `'band'`, `'error_bars'`, `'nested_band'`

**New stats dict keys when quantiles set:** `q_lower_per_bin`, `q_upper_per_bin`, `quantiles_per_bin` (per mode).

**New style keys (4):** `quantile.band.alpha`, `quantile.band.hatch`, `quantile.error_bars.capsize`, `quantile.central_default`.

### 1.5 Channel-Aware Visual Encoding (Phase 13.26.DF v1.2 — MultiGraph Phase B)

Algorithm A automatically resolves which visual encoding (color/linestyle/marker)
each active data channel gets, eliminating channel-collision bugs (e.g., both
vector and group_by silently landing on `color`).

```python
# 3-channel call: vector × group_by × quantiles
d.profile("[y1,y2,y3]:x", group_by="sector",
          quantiles=[0.16, 0.5, 0.84])
# Algorithm A assigns: group_by → color, vector → marker, quantiles → linestyle
# Factored legend: section headers per channel; entries = sum of cardinalities (not product)

# Override per-call
d.profile("y:x", quantiles=[0.16, 0.84], quantile_style='color')

# Override default via style
set_style({"channels.priority.categorical": ["color", "marker", "linestyle"]})
```

**New parameter on `profile()`:**
- `quantile_style: Optional[str]` — channel for quantile dimension when combined with group_by

**New style keys (10):** `channels.priority.{categorical,ordinal}`, `channels.cycles.{linestyle,marker,color_count}`, `channels.default.{vector,group_by,quantiles}`, `channels.overflow`, `channels.legend.factored`.

**Capacity:** Each channel has a cycle limit (`channels.cycles.color_count=10`, `linestyle` has 4 entries by default, `marker` has 8). Overflow mode is `'error'` by default with actionable suggestions (`top_k=`, `facet=True`, `group_by_bins=`); `'warn'` permits truncation with warning.

**Factored legend** (default `True`): entry count is sum of cardinalities, not product. 3-channel call with `|group|=5, |vector|=3, |quantiles|=5` produces 13 entries factored vs 75 unfactored.

**Nested-band auto-detection (AD-57):** Symmetric quantile lists with ≥4 non-0.5 entries auto-route to `'nested_band'` mode (alpha-stacked filled regions). Central line handled independently via `central=` parameter.

### 1.6 Facet Refactor (Phase 13.27.DF Commit 1 — MultiGraph Phase D, profile-only)

Replaces inline `facet=True` path on `profile()` with a unified
`_dispatch_faceted_render()` method routing through the channel framework.
`facet=True` remains as backward-compat alias.

```python
# Backward compat — facet=True still works (normalized to facet_by='group_by')
d.profile("y:x", group_by="sector", facet=True)

# New API — facet_by= explicit channel name
d.profile("y:x", group_by="sector", facet_by="group_by")

# Facet by vector (one subplot per vector y element)
d.profile("[y1,y2,y3]:x", facet_by="vector")

# Facet by quantile (with quantile_mode='discrete')
d.profile("y:x", quantiles=[0.25, 0.5, 0.75],
          quantile_mode="discrete", facet_by="quantiles")
```

**New parameter on `profile()`:**
- `facet_by: Optional[str]` — facet channel; valid in Commit 1: `{'group_by', 'vector', 'quantiles'}`. `'selection_delta'` / `'weights_delta'` reserved for Commit 2 (raise `NotImplementedError`).

**Mutual exclusion (architect rule):** `facet_by` and `same=True` raise `ValueError` (`"facet_by and same=True are mutually exclusive"`). Facet creates a new figure; `same=True` overlays on existing axes — semantics conflict.

**Capacity:** Subplot count bounded by `channels.cycles.facet_max=16`. Overflow `'warn'` truncates and warns; `'error'` raises with actionable message.

**Backward-compat invariance:** `facet=True` ≡ `facet_by='group_by'` byte-identical figure output, locked by invariance test `test_facet_true_eqivalent_to_facet_by_groupby`.

**New style keys (6):** `channels.cycles.facet_max`, `channels.legend.facet_position`, plus 4 reserved for Commit 2 (`channels.label.{selection,weights}_truncate`, `channels.default.{selection,weights}_delta`).

**Out of scope for Commit 1 (Commit 2 pending):** `selection_vector` + `weights_vector` parameters; `delta_facet` plot type; hist + scatter facet integration; 50 invariance tests for selection/weights.

### 1.7 Robust Data Handling (Phase 13.28.DF v1.1)

Centralized NaN/inf filter and outlier-aware autorange across all 5 plot types.
Closes a class of robustness bugs surfaced on real TPC data: `y/x` expressions
with `x=0` producing `inf` no longer crash matplotlib; NaN-filtered selections
no longer silently return `n=0` with no diagnostic.

```python
# Default behaviour (nan_policy='filter'): silent drop + counters
fig, ax, stats = d.hist2d("y/x:x", selection="detType==0")
# stats includes: n_input=4, n=3, n_inf_y=1, autorange_used=((1.0, 4.0), (0.5, 1.5)),
#                 autorange_strategy='hybrid'

# Explicit warn or raise modes
d.profile("y:x", nan_policy='warn')   # UserWarning when invalid present
d.profile("y:x", nan_policy='raise')  # ValueError on any NaN/inf

# Strategy override
d.hist("y", range='minmax')          # backward-compat (matplotlib-equivalent)
d.hist("y", range='percentile_99')   # 1st-99th percentile clip
d.hist("y", range='hybrid')          # default — outlier-aware
```

**New parameter on all 5 plot methods (`draw / hist / hist2d / profile / scatter`):**
- `nan_policy: str = "filter"` — `'filter'` (silent drop + counters, default), `'warn'` (drop + UserWarning), `'raise'` (ValueError)

**`range=` extended:** Now accepts strategy strings `'hybrid'` (default), `'minmax'`, `'percentile_99'`, `'percentile_95'`, `'robust_3mad'`, `'robust_4mad'` in addition to numeric tuples.

**New stats dict keys (always populated):**
- Sanitize counters: `n_input`, `n_filtered`, `n_inf_x`, `n_nan_x`, `n_inf_y`, `n_nan_y`
- Autorange diagnostics: `autorange_used` (the actual numeric range used), `autorange_strategy` (`'hybrid'`, `'minmax'`, etc., or `'explicit'` when user passed numeric range)

**Critical semantics lock (AD-77):** `stats['n']` is the post-sanitize finite count, NOT range-clipped. Range filtering is **VISUAL ONLY** — `range=` does not reduce `stats['n']`.

**New style keys (5):** `data.nan_policy`, `autorange.strategy`, `autorange.k_robust`, `autorange.k_outlier`, `autorange.percentile`.

**Hybrid autorange algorithm:** Computes robust window `(median ± k_robust·sigma_MAD)`. Per-side, declares outlier on side S if data extreme exceeds median by more than `(k_outlier · k_robust · sigma_MAD)`. Uses robust bound when outlier present, else uses data extreme. Clean Gaussians get full `(min, max)`; outlier-bearing data gets clipped only on the affected side. 2D autorange is per-axis independent.

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

**Vector expressions (Phase 13.16.DF):**
- Nested brackets not supported: `"[arr[0],arr[1]]:x"` — use explicit column names
- Broadcasting is strict N:1, 1:N, N:N — mismatched shapes raise `ValueError`
- `hist2d`/`hexbin` do not accept vector input (fail-fast with clear message)
- Typical N ≤ 20; complexity is O(N × draw_cost), not optimized for N ≥ 100
- `stats()` with vector returns `list[dict]` instead of a single `dict`

### 11.3 Current Bugs / Issues

**No tests failing as of Phase 13.27.DF Commit 1** (663/663 tests passing on commit `PHASE_13_27_DF_Commit1_END`).

**Resolved in Phase 13.16.DF:**
- **AD-37** — AliasDataFrame per-call `DFDraw` instantiation reset the color
  cycle, causing `aDF.draw(...same=True)` loops to show only 1–2 colors instead
  of the expected N. Fixed via vector expression interface: a single `DFDraw`
  instance now handles the full series internally. Invariance test
  `test_vector_through_adf_equivalent_to_direct` confirms the fix.

**Resolved in Phase 13.28.DF:**
- **`y/x:row` hist2d crash on `x=0` rows** — matplotlib raised
  `autodetected range of [-inf, inf] is not finite` whenever `inf` reached
  the histogram bin-edge calculation. Fixed via centralized
  `sanitize_for_plot()` in `plots/_data_sanitize.py` called by all 5 plot
  types. Invariance test
  `TestPlotIntegration::test_hist2d_with_inf_does_not_crash_and_reports_counters`
  locks the architect's exact reproducer.
- **Silent `n=0` from NaN selections** — `hist2d` and friends silently
  returned `n=0` when all rows had NaN in the chosen expression columns. Now
  `stats['n_input']`, `stats['n_filtered']`, `stats['n_inf_*']`, `stats['n_nan_*']`
  are always populated; `nan_policy='warn'` or `'raise'` available for
  per-call escalation.

**Open follow-ups (FIX1 — non-blocking, same bug class):**
- **Phase 13.26 FIX1** (Claude40-approved, not started) — `group_style='color'`
  signature default blocks `set_style({"channels.default.group_by": "linestyle"})`
  from taking effect. Mechanical fix: change default to `None`, resolve from
  style at runtime.
- **Phase 13.28 FIX1** (GPT4 #2 finding from closure review) — `nan_policy="filter"`
  signature default blocks `set_style({"data.nan_policy": "raise"})` for the
  same structural reason. Recommended bundle with Phase 13.26 FIX1.

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

### 14.5 Quantile Rendering (Phase 13.25.DF v1.3)

```python
quantiles: List[float]            # Fractions in (0, 1) — symmetric pair, triple, or 4+
central: str                      # 'mean' (default), 'median', 'both', 'none'
quantile_mode: str                # 'auto' (default), 'discrete', 'band',
                                  #     'error_bars', 'nested_band'
```

### 14.6 Channel-Aware Encoding (Phase 13.26.DF v1.2)

```python
quantile_style: str               # Channel for quantile dimension
                                  #     ('color', 'linestyle', 'marker')
                                  # Used when combined with group_by
                                  # Algorithm A auto-resolves if not set
```

### 14.7 Facet Routing (Phase 13.27.DF Commit 1, profile-only)

```python
facet_by: str                     # Facet channel — Commit 1 valid:
                                  #     'group_by', 'vector', 'quantiles'
                                  # Reserved for Commit 2 (NotImplementedError):
                                  #     'selection_delta', 'weights_delta'
```

### 14.8 Robust Data Handling (Phase 13.28.DF v1.1)

```python
nan_policy: str                   # 'filter' (default), 'warn', 'raise'
range: tuple | str                # Numeric (lo, hi) or strategy name:
                                  #     'hybrid' (default), 'minmax',
                                  #     'percentile_99', 'percentile_95',
                                  #     'robust_3mad', 'robust_4mad'
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

**Test suite:** 663 tests (100% passing as of Phase 13.27.DF Commit 1)
**Features:** 62
**Invariance tests:** 28+ (A≡B semantic contracts; §9 markers introduced in Phase 13.27)
**Verified features:** 7 (features with at least one invariance test)

**Coverage areas:**
- All plot types: hist, scatter, profile, hist2d, hexbin ✅
- Expression evaluation ✅
- Vector expression interface (Phase 13.16.DF) ✅ **Verified**
- Selection and filtering ✅
- Group-by overlays ✅
- Batch processing (dict and list formats) ✅
- PyArrow integration ✅ **Verified**
- Stats computation ✅
- Error handling ✅
- same=True superposition (Phase 13.13.DF) ✅ **Verified**
- Defaults cascade and verbose levels (Phase 13.14.DF) ✅ **Verified**
- Profile enhancements: return_data, min_entries, group_by_bins, weights (Phase 13.12.DF) ✅
- Test infrastructure: feature taxonomy, capability matrix (Phase 13.15.DF) ✅
- Robust statistics groups: stat_fields parameter (Phase 13.18.DF) ✅
- Quantile rendering: error_bars + band + nested-band auto-detect (Phase 13.25.DF) ✅
- N-channel framework: Algorithm A automatic visual-encoding assignment (Phase 13.26.DF) ✅
- Facet refactor: `_dispatch_faceted_render()`, `facet_by=`, capacity, mutual exclusion (Phase 13.27.DF Commit 1) ✅ **§9-marked invariance tests**
- Robust data handling: NaN/inf filter, hybrid autorange, sanitize counters, autorange diagnostics (Phase 13.28.DF) ✅

**Verified (A≡B invariance) features:**
- `SAME.axes_reuse` — `same=True` byte-identical axes reuse
- `SAME.override` — `ax=` / explicit overrides precedence
- `SAME.cross_method` — profile-on-hist2d cross-type superposition
- `BATCH.group_format` — defaults cascade byte-identical to manual merge
- `PYARROW.input` — PyArrow ≡ pandas parity for all plot types
- `VECTOR.invariance` — vector path ≡ scalar `same=True` loop (7 strong tests)
- `VECTOR.kwarg_propagation` — vector path forwards all scalar-mode kwargs (Phase 13.16.DF FIX1, 7 strong tests)

**Phase 13.27.DF Commit 1 invariance tests** (10, with `# §9.<class>.<id>` markers per Coder QRC v1.30 Rule 14):
- `TestFacetLegacyEquivalence` (3) — `facet=True` ≡ `facet_by='group_by'` byte-identical
- `TestFacetByChannel` (4) — vector / group_by / quantiles routing + invalid raises
- `TestFacetCapacity` (2) — capacity error mode + warn mode
- `TestFacetSameTrueExclusion` (1) — `facet_by` × `same=True` mutual exclusion

**Integration tests:**
- AliasDataFrame integration ✅
- Lazy loading ✅
- Subframe joins ✅
- ADF vector dispatch (Phase 13.16.DF) ✅
- Plot-module integration of sanitize_for_plot + autorange (Phase 13.28.DF) ✅

**Performance tests:**
- Large datasets (100k+ rows) ✅
- Memory overhead (PyArrow conversion) ✅

**Test infrastructure (Phase 13.15.DF, extended through Phase 13.28.DF):**
- `tests/feature_taxonomy.py` — 62 features enumerated
- `tests/test_layer_classification.py` — smoke vs invariance classification
- `scripts/generate_capability_matrix.py` → `docs/CAPABILITY_MATRIX.md`
- `run_tests.sh` — full/quick/matrix modes + `reviewer.zip` packaging
- `_validate_forwarded_names()` — class-load R6 check on forwarder/signature parity (Phase 13.16.DF FIX1; caught Phase 13.28 FIX1 missing-`nan_policy` mid-integration)

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

## 19. MultiGraph Framework — Phases 13.25 / 13.26 / 13.27

The MultiGraph Framework is a coordinated three-phase effort that elevates
`dfdraw` from a per-call plotter to a **systematic visualization grammar**
for multi-dimensional differential analysis. Each phase locked one piece
of the grammar; together they let an analyst describe an N-channel plot
declaratively and let the framework decide how each channel is rendered.

### 19.1 Phase Hierarchy

| Phase | Role | Status |
|---|---|---|
| **13.25 Phase A** | Quantile rendering primitive on `profile()` | ✅ Closed (v1.3 + FIX1 + FIX2) |
| **13.26 Phase B** | N-channel framework, Algorithm A, channel-aware rendering | ✅ Closed (v1.2 implementation; FIX1 pending) |
| **13.27 Phase D** | Selection / weights vectors + facet refactor + delta facet | 🟡 Commit 1 closed (facet refactor, profile-only); Commit 2 pending |

### 19.2 The Visualization Grammar

A user-facing call now decomposes into three independent dimensions:

```python
d.profile("[y1,y2,y3]:x",            # vector channel:    3 series
          group_by="sector",          # group_by channel:  N groups
          quantiles=[0.16,0.5,0.84])  # quantile channel:  3 quantiles
```

The framework asks three questions per call:

1. **Which data channels are active?** (vector, group_by, quantiles, selection_delta, weights_delta)
2. **What is each channel's cardinality?** (counts the unique values)
3. **Which visual encoding (color / linestyle / marker) does each channel get?** (Algorithm A)

The answer is computed by `assign_channels()` in `dfdraw/channels.py` using
`EXPLICIT_RULES` (architect-approved overrides per channel-set, append-only
across phases — see GP-2 in `STYLING_FRAMEWORK_DECISIONS.md` §3) plus a
priority-list fallback (`channels.priority.categorical` defaults to
`["color", "linestyle", "marker"]`).

### 19.3 Architectural Properties Locked Across the Three Phases

- **Append-only `EXPLICIT_RULES`** — once a phase locks how channels assign
  for a particular set, that mapping is preserved. New channels add new
  list entries, never modify existing.
- **Style keys land at interface introduction (GP-1)** — every new
  channel/feature lands its style keys in the same phase as its rendering
  logic. No deferred Phase-C+ retrofits.
- **Capacity gates are explicit and actionable** — overflows raise (default)
  or warn with the suggested mitigation (`top_k=`, `facet=True`, `group_by_bins=`).
  Silent auto-facet was rejected as hiding intent.
- **Factored legend** — entry count is sum of cardinalities, not product.
  3-channel call with `|g|=5, |v|=3, |q|=5` → 11 entries, not 75.
- **Mutual exclusion at dispatch level** — `same=True` and `facet_by=`
  are mutually exclusive (Phase 13.27); vector overlay (`_draw_vector`) and
  vector facet (`_dispatch_faceted_render`) are mutually exclusive (Phase 13.27 vector-entry intercept).

### 19.4 Out of Scope (Phase 13.27 Commit 2)

- `selection_vector` and `weights_vector` parameters
- `delta_facet` plot type (`facet_by='selection_delta'`, `facet_by='weights_delta'`)
- hist + scatter facet integration (currently profile-only)
- ~50 invariance tests for the selection/weights paths

---

## 20. dfdraw as Multidimensional Differential Analysis (MDA) Substrate

Per `Multidimensional_Differential_Analysis-v0_5.md` §"Tooling — dfextensions
and Companions as MDA Substrate", `dfdraw` is the visualization substrate
for the MDA methodology. Each phase delivered enables a specific MDA
operation:

| Phase | dfdraw capability | MDA enabler |
|---|---|---|
| 13.16 | Vector dispatch + R6 forwarder validator | Shadow projections across observables in one call |
| 13.18 | Robust statistics groups (`stat_fields='robust'`) | Detection of outlier-driven mismeasurement (median + MAD beside mean + std) |
| 13.25 | Quantiles on `profile()` | Per-bin distribution-shape comparison; goes beyond mean ± std |
| 13.26 | N-Channel framework (Algorithm A) | Visual encoding budget for high-D analysis without channel collisions |
| 13.27 (Commit 1) | Facet through channel framework | Multi-dimensional differential breakdown into subplot grid |
| 13.27 (Commit 2 — pending) | `selection_vector` + `weights_vector` + `delta_facet` | The **core MDA operation** — declare M selections × N weight schemes and let the framework render the cross-product as a facet grid |
| 13.28 | Robust autorange + NaN/inf safety | Reliable plots on raw production data without per-call data hygiene |

The framework's contract — **per-pair invariance, byte-identical axes
state, factored legends, channel-aware encoding** — is precisely what an
MDA workflow needs to compare distributions across slicing dimensions
without rendering artifacts masking real differences.

---

## 21. Governance and Review Discipline

dfdraw development follows a formal phase-lifecycle (proposal → multi-reviewer
panel → consolidated review → implementation → tag) with versioned governance
documents. This section summarises the discipline; full text in
`docs/STYLING_FRAMEWORK_DECISIONS.md` (§3 Governance Principles) and
`docs/Organization-structure.md`.

### 21.1 Governance Principles (GP-1 through GP-5)

- **GP-1** — Style configurability lands at interface introduction, not deferred.
- **GP-2** — Internal APIs accepting new data-channel types must be list-based from day one (`EXPLICIT_RULES` is append-only).
- **GP-3** — Architect signals preserved verbatim with typos. Reformulating quotes has caused production bugs.
- **GP-4** — Backward-compat scope must be justified by production-usage verification (`grep`/AST against production scripts).
- **GP-5** — Drafter rotation across phases is healthy (different drafters bring different lenses; precedent: Phase 13.26 v1.0/v1.1 → v1.2 drafter handoff caught a structural improvement).

### 21.2 Coder Quick Reference Card v1.30 — Rule 14 (new in Phase 13.27)

**Binding §9 assertion-marker rule.** Each test class in a phase's invariance
test file must contain at least one assertion drawn from the proposal's §9
load-bearing list, marked inline as `# §9.<class>.<id>` (e.g.
`# §9.LegacyEquiv.1`). This pins each test body to a specific architect-
locked invariant; reviewers verify against the proposal §9 list.

### 21.3 Cross-Group Review Standard

Phases that touch the user-facing API surface (`DFDraw.draw / .hist / .hist2d /
.profile / .scatter`) follow a 3+3 cross-group review standard: 3 dfdraw
reviewers + 3 ADF reviewers (or equivalent cross-subproject). Phase 13.28
closure (5-0: Claude40 + Claude48 + GPT4×2 + Claude32 ADF) is the most
recent example.

### 21.4 Multi-Reviewer Model — Empirical Evidence

The multi-reviewer model has caught real defects that single-reviewer
discipline would have missed:

- **Phase 13.16.DF Rev2 → Rev3** — 6 P0 defects caught by external panel
  that internal approvers missed (3-colon parsing, paren-inside-bracket,
  ADF entry test missing, `same=`/`type=` keyword collisions, 7 call sites
  not 4).
- **Phase 13.26.DF v1.0 → v1.2** — Drafter handoff (Claude48 → Claude49Coder)
  caught the G-7 forward-extensibility upgrade that the original drafter
  did not surface.
- **Phase 13.28.DF closure** — GPT4 #2 caught the `nan_policy` style-key
  signature-default blocker that 3 other reviewers (including Claude40
  and Claude48) missed.
- **PHASE_HISTORY.md v1.5 review** — Sonet51 caught the stale Overview
  header (`Phase 13.16.DF FIX1 / 469 tests`) that Claude40 (full git access)
  and Sonet50 (sources.zip access) both missed. Catch came from the
  reviewer with the *least* source access.

### 21.5 Two-Commit Patterns

Two patterns now codified:

- **Phase 13.16 FIX1 pattern** — When fixing a regression, ship as two
  commits: Commit 1 (red baseline reproducing the bug) + Commit 2 (green
  fix). Preserves diagnostic state in git history; pre-fix `reviewer.zip`
  becomes a permanent regression-detection artifact.
- **Phase 13.28 pattern** — When delivering a multi-part feature, each
  commit ships **working code with passing tests** (no scaffolding-only
  commits). Phase 13.28 shipped as Part A → Part B → Integration, each
  with its own passing tests. Phase 13.27 Commit 1 followed the same
  discipline (10 invariance tests pass at commit time, no skip stubs).

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

**Phase 13.27.DF Commit 1 (Current — MultiGraph Phase D, profile-only):**
- Facet refactor: replaces inline `facet=True` path with unified `_dispatch_faceted_render()` method routing through channel framework
- New parameter: `facet_by` ∈ `{'group_by', 'vector', 'quantiles'}` (Commit 1 scope; `'selection_delta'` / `'weights_delta'` reserved for Commit 2)
- Backward compat: `facet=True` normalizes to `facet_by='group_by'` byte-identical (locked by invariance test)
- Vector-entry intercept: `facet_by='vector'` routes BEFORE `_draw_vector` (vector overlay × vector facet mutually exclusive at dispatch level)
- Mutual exclusion: `facet_by` × `same=True` raises ValueError (architect rule v1.1 §5.4)
- Capacity check via `channels.cycles.facet_max=16`; overflow `'warn'` truncates+warns, `'error'` raises
- 6 new style keys (2 used in Commit 1, 4 reserved for Commit 2)
- `'facet_by'` added to `_PROFILE_FORWARDED_NAMES` (Phase 13.28 FIX1 lesson applied prospectively)
- Stale `test_default_style_has_all_10_keys` updated to `*_all_channels_keys` (16 keys grouped by phase with comments)
- 10 §9-marked invariance tests (TestFacetLegacyEquivalence + TestFacetByChannel + TestFacetCapacity + TestFacetSameTrueExclusion); Coder QRC v1.30 Rule 14 adopted
- AD-61..AD-68
- 663 tests passing, 62 features, 28+ invariance tests, 7 Verified

**Phase 13.28.DF v1.1 (closed at commit `8b02d241`, tags `PHASE_13_28_DF_v1_0_END` and `PHASE_13_28_DF_Integration_END`):**
- Centralized sanitization: `plots/_data_sanitize.py::sanitize_for_plot()` — uniform NaN/inf handling across hist / hist2d / hexbin / profile / scatter
- New parameter: `nan_policy` ∈ `{'filter', 'warn', 'raise'}`, default `'filter'` (silent drop with counters)
- New stats keys: `n_input`, `n_filtered`, `n_inf_x`, `n_nan_x`, `n_inf_y`, `n_nan_y` (always populated)
- Hybrid autorange: `compute_autorange()` with 6 strategies (`'hybrid'` default, `'minmax'`, `'percentile_99'`, `'percentile_95'`, `'robust_3mad'`, `'robust_4mad'`)
- `range=` now accepts strategy strings in addition to numeric tuples
- New stats keys: `autorange_used`, `autorange_strategy` (always populated; `'explicit'` when user passed numeric range)
- `stats['n']` semantics locked: post-sanitize finite count, NOT range-clipped (range is VISUAL ONLY)
- 5 new style keys: `data.nan_policy`, `autorange.{strategy,k_robust,k_outlier,percentile}`
- Bug fix: `adf.draw("y/x:row", type='hist2d')` no longer crashes matplotlib on `inf` from `x=0` rows
- Bug fix: NaN-filtered selections no longer silently return `n=0`
- 26 new tests (12 sanitize + 9 autorange + 5 integration); 5-0 closure verdict (Claude40 + Claude48 + GPT4×2 + Claude32 ADF)
- AD-69..AD-77
- 653 tests passing at closure; +10 in Phase 13.27.DF Commit 1 → current 663
- FIX1 pending (GPT4 #2): `nan_policy` style-key default blocker; same bug class as Phase 13.26 FIX1

**Phase 13.26.DF v1.2 (MultiGraph Phase B, closed at `PHASE_13_26_DF_v1_0_END`):**
- N-channel framework: `dfdraw/channels.py` with `assign_channels()` resolving color/linestyle/marker assignment for active data channels (vector / group_by / quantiles)
- Algorithm A: `EXPLICIT_RULES` (architect-approved per channel-set, append-only) + priority-list fallback (`channels.priority.categorical`)
- New parameter: `quantile_style` for explicit channel override
- 10 new `channels.*` style keys: `priority.categorical`, `priority.ordinal`, `cycles.linestyle`, `cycles.marker`, `cycles.color_count`, `default.vector`, `default.group_by`, `default.quantiles`, `overflow`, `legend.factored`
- Factored legend (default `True`): entry count = sum of cardinalities, not product
- Nested-band auto-detection (AD-57): symmetric quantile lists with ≥4 non-0.5 entries → `'nested_band'` mode
- FIX2 visual elements channel-aware preservation: linestyle cycle from style key (`[1:]` slice — solid reserved for central line); on-line annotations preserved for `quantile_style='linestyle'`, suppressed for `'marker'`/`'color'`
- 50 new tests (TestChannelAssignment{1,2,3}Active, TestChannelCapacity, TestChannelCollision, TestChannelUserOverride, TestChannelStyleOverride, TestFactoredLegend, TestNestedBand, TestIdempotency)
- AD-55..AD-60; GP-2 / GP-4 / GP-5 governance principles recorded
- 627 tests passing
- FIX1 pending (Claude40-approved, not started): `group_style='color'` default + 10 docstrings + CAPABILITY_MATRIX amend

**Phase 13.25.DF v1.3 + FIX1 + FIX2 (MultiGraph Phase A):**
- Quantile rendering on `profile()`: error_bars (asymmetric bars from symmetric pair without 0.5), band (fill_between from symmetric triple with 0.5), auto-detect via `_detect_quantile_mode()`
- New parameters: `quantiles` (list of fractions), `central` (`'mean'`/`'median'`/`'both'`/`'none'`), `quantile_mode` (`'auto'`/`'discrete'`/`'band'`/`'error_bars'`/`'nested_band'`)
- New stats keys: `q_lower_per_bin`, `q_upper_per_bin`, `quantiles_per_bin`
- 4 new style keys: `quantile.band.alpha`, `quantile.band.hatch`, `quantile.error_bars.capsize`, `quantile.central_default`
- FIX1: empty quantile dict pruning (caught by 6-reviewer panel as P1)
- FIX2: linestyle cycle + on-line percentage annotations for discrete quantile rendering
- 28+ new tests; AD-44..AD-54; GP-1 / GP-3 governance principles recorded

**Phase 13.18.DF (Robust statistics extension, tag `PHASE_13_18_DF_v1_0_END`):**
- New parameter: `stat_fields` ∈ `{'base', 'robust', 'all', list[str]}` on all 5 plot methods + `DFDraw.draw()`
- New stats keys when `'robust'` active: `median`, `mad`, `q25`, `q75`, `iqr` (MAD scaled by 1.4826 for Gaussian-equivalent sigma)
- 2 new STATS.* features in feature_taxonomy: `STATS.robust`, `STATS.range_aware`

**Phase 13.16.DF v1.0:**
- Vector expression interface: `[y1,y2,y3]:x`, `y:[x1,x2]`, `[y1,y2]:[x1,x2]`
- Paren-aware comma split: `[max(a,b),max(c,d)]:x`
- Architectural fix for AD-37 AliasDataFrame color-cycling bug
- `vector_style` and `group_style` channel decomposition
- Per-pair stats return (`list[dict]`)
- Fail-fast on `hist2d`/`hexbin` vector input
- 7 strong A≡B invariance tests (`TestVectorInvariance`)
- Conditional color cycle reset preserves `SAME.axes_reuse` contract
- 451 tests passing, 43 features, 21 invariance tests, 6 Verified

**Phase 13.15.DF v1.0:**
- Test infrastructure: `feature_taxonomy.py` (35→43 features)
- `test_layer_classification.py` — smoke vs invariance markers
- `scripts/generate_capability_matrix.py` → `CAPABILITY_MATRIX.md`
- `run_tests.sh` with full/quick/matrix modes + `reviewer.zip`
- `scripts/phase_tag.sh` helper
- 401 tests passing, 5 Verified features

**Phase 13.14.DF v1.0:**
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

**Document Status:** Updated through Phase 13.27.DF Commit 1 (663 tests, 62 features, 28+ invariance tests, 7 Verified)
**Maintainer:** Team3-dfdraw
**Last Updated:** 2026-05-09
**Version:** Phase 13.27.DF Commit 1 + Phase 13.28.DF v1.1 (closed in parallel)
**Next Update:** After Phase 13.27.DF Commit 2 (selection_vector + weights_vector + delta_facet + hist/scatter integration) closure, or after Phase 13.26/13.28 FIX1 bundle
