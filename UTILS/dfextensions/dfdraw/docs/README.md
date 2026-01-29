# dfdraw - DataFrame Drawing Utilities

**ROOT-style one-liner plotting for Pandas DataFrames**

`dfdraw` brings the simplicity of ROOT's `TTree::Draw()` to Python DataFrames. Draw histograms, scatter plots, and profiles with a single expression string, while retaining full matplotlib customization.

Part of the `dfextensions` toolkit for ALICE experiment calibration and QA at CERN.

---

## ⚠️ Breaking Change (Phase 13.6.G.DF)

**Standard deviation now uses population std (ddof=0) to match ROOT.**

As of Phase 13.6.G.DF, `std`, `std_x`, and `std_y` use population standard deviation (ddof=0) instead of sample standard deviation (ddof=1). This matches ROOT's TTree::Draw behavior exactly.

**Impact:** Values are ~6% smaller than previous versions.

```python
# Example: data = [1, 2, 3, 4, 5]
# Before (ddof=1): std = 1.5811
# After (ddof=0):  std = 1.4142  ← Matches ROOT
```

---

## Toolkit Architecture

`dfdraw` is part of the loosely-coupled dfextensions toolkit:

| Tool | Purpose | Integration |
|------|---------|-------------|
| **AliasDataFrame** | Lazy evaluation, schema-driven analysis | Provides data + axis titles |
| **GroupByRegressor** | Fast statistical fits | Results visualized via dfdraw |
| **RDataFrameDSL** | Python->C++ code generation | Uses dfdraw for QA plots |
| **dfdraw** | ROOT-style plotting | Standalone or integrated |

Each tool works independently. Integration is optional.

---

## Installation

```bash
# From the dfextensions package
pip install dfextensions

# Or standalone
cd UTILS/dfextensions/dfdraw
pip install -e .
```

**Dependencies:** `pandas`, `numpy`, `matplotlib`  
**Optional:** `pyarrow` (for PyArrow Table input)

---

## Quick Start

```python
from dfdraw import DFDraw
import pandas as pd
import numpy as np

# Create drawer from DataFrame
df = pd.DataFrame({'x': np.random.randn(10000), 'y': np.random.randn(10000)})
drawer = DFDraw(df)

# 1D histogram (like TTree::Draw("x"))
fig, ax, stats = drawer.draw("x")

# 2D scatter (like TTree::Draw("y:x"))
fig, ax, stats = drawer.draw("y:x")

# Profile plot (mean of y vs x)
fig, ax, stats = drawer.draw("y:x", type="profile")

# With selection (like TTree::Draw("x", "x > 0"))
fig, ax, stats = drawer.draw("x", selection="x > 0")
```

---

## Core Concepts

### Expression Syntax

| Expression | Plot Type | ROOT Equivalent |
|------------|-----------|-----------------|
| `"x"` | 1D histogram | `tree->Draw("x")` |
| `"y:x"` | 2D scatter/hist2d | `tree->Draw("y:x")` |
| `"y:x"` + `type="profile"` | Profile (mean y vs x) | `tree->Draw("y:x", "", "prof")` |

### Automatic Type Detection

- **1D expression** (`"x"`) -> histogram
- **2D expression** (`"y:x"`) -> scatter plot (or hist2d/hexbin/profile with `type=`)

### Return Values

All draw methods return `(fig, ax, stats)`:
- `fig`: matplotlib Figure
- `ax`: matplotlib Axes (or array for faceted plots)
- `stats`: dict with computed statistics (`n`, `mean`, `std`, `corr`, etc.)

---

## Usage Modes

`dfdraw` integrates with the dfextensions toolkit at three levels:

| Mode | Data Source | Use Case |
|------|-------------|----------|
| **Standalone** | `pd.DataFrame` | Quick exploration, any pandas data |
| **With AliasDataFrame** | `AliasDataFrame` | Calibration workflows, lazy evaluation |
| **With RDataFrameDSL** | ROOT TTree via DSL | C++ performance, ROOT integration |

### Mode 1: Standalone (Pandas only)

```python
import pandas as pd
from dfdraw import DFDraw

df = pd.read_parquet("data.parquet")
drawer = DFDraw(df)
drawer.draw("pt:eta", selection="quality > 0.5")
```

### Mode 2: With AliasDataFrame

```python
from AliasDataFrame import AliasDataFrame
from dfdraw import DFDraw

aDF = AliasDataFrame(df)
aDF.add_alias("pt_gev", "pt / 1000")
aDF.set_axis_title("pt_gev", "p_{T} [GeV/c]")

drawer = DFDraw(aDF)
drawer.draw("pt_gev")  # Alias resolved, title applied automatically
```

### Mode 3: With RDataFrameDSL

```python
from RDataFrameDSL import DSLCompiler

dsl = DSLCompiler(schema)
# ... define expressions ...

# draw_figures() uses dfdraw internally
dsl.draw_figures(qa_specs, rdf, save_dir="plots/")
```

### Architecture Diagram

```
+-----------------------------------------------------------+
|                    User Code                              |
+-----------------------------+-----------------------------+
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
+---------------+     +---------------+     +---------------+
|  pd.DataFrame |     |AliasDataFrame |     | RDataFrameDSL |
|  (standalone) |     |(schema-driven)|     |  (C++ perf)   |
+-------+-------+     +-------+-------+     +-------+-------+
        |                     |                     |
        +---------------------+---------------------+
                              |
                              v
                    +-------------------+
                    |      dfdraw       |
                    |   (ROOT-style     |
                    |    plotting)      |
                    +-------------------+
```

---

## API Reference

### Main Class: `DFDraw`

```python
class DFDraw:
    def __init__(self, data):
        """
        Create drawer from data source.
        
        Parameters
        ----------
        data : DataFrame, AliasDataFrame, PyArrow Table, or dict
            Input data. AliasDataFrame is auto-detected via duck typing.
            PyArrow Tables are converted to pandas internally.
        """
```

### Primary Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `draw()` | `draw(expr, type=None, selection=None, **kwargs)` | Universal draw method with auto type detection |
| `hist()` | `hist(expr, bins=50, range=None, norm=None, **kwargs)` | 1D histogram |
| `scatter()` | `scatter(expr, color=None, size=None, **kwargs)` | Scatter plot |
| `profile()` | `profile(expr, bins=50, error="sem", **kwargs)` | Profile plot (mean +/- error vs x) |
| `hist2d()` | `hist2d(expr, bins=50, norm=None, **kwargs)` | 2D histogram |
| `hexbin()` | `hexbin(expr, gridsize=50, **kwargs)` | Hexagonal binning (better for large data) |
| `stats()` | `stats(expr, selection=None, group_by=None)` | Compute stats without plotting |
| `draw_batch()` | `draw_batch(specs, save_dir=None, **kwargs)` | Batch plot generation from spec dict |

### Annotation Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_statistics_box()` | `add_statistics_box(ax, values, position='upper right', expected_mean=None, expected_std=None, **kwargs)` | Add n/mu/sigma annotation box |
| `add_reference_overlay()` | `add_reference_overlay(ax, func='gaussian', mu=0, sigma=1, **kwargs)` | Add scaled reference curve |

### Common Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `selection` | str, array, callable | Filter data (query string, bool mask, or function) |
| `group_by` | str | Column for overlaid groups |
| `top_k` | int | Limit to top K groups by count |
| `facet` | bool | Create subplot grid instead of overlay |
| `stats` | bool or list | Show statistics box |
| `save` | str | Save figure to path |
| `ax` | Axes | Plot on existing axes |

---

## Plot Types

### 1D Histogram

```python
# Basic
drawer.hist("pt", bins=100)

# With normalization
drawer.hist("pt", norm="density")      # Probability density
drawer.hist("pt", norm="probability")  # Normalized to 1

# Grouped overlay
drawer.hist("pt", group_by="particle_type", top_k=5)

# Stacked
drawer.hist("pt", group_by="particle_type", stacked=True)
```

### Scatter Plot

```python
# Basic
drawer.scatter("y:x")

# Color by column (continuous or categorical)
drawer.scatter("y:x", color="energy")
drawer.scatter("y:x", color="category", colorbar=True)

# Size mapping
drawer.scatter("y:x", size="weight")

# With jitter (for quantized data)
drawer.scatter("y:x", jitter=True)
drawer.scatter("y:x", jitter=0.1)  # Custom amount
```

### Profile Plot

```python
# Mean of y in bins of x
drawer.profile("dEdx:momentum", bins=100)

# Error types
drawer.profile("y:x", error="sem")   # Standard error of mean (default)
drawer.profile("y:x", error="std")   # Standard deviation
drawer.profile("y:x", error="none")  # No error bars
```

### 2D Histogram / Hexbin

```python
# 2D histogram
drawer.hist2d("y:x", bins=[100, 50])

# Log scale
drawer.hist2d("y:x", norm="log")

# Hexbin (better for large datasets)
drawer.hexbin("y:x", gridsize=30, mincnt=1)
```

---

## Grouping and Faceting

### Overlay Mode (default)

```python
# Multiple histograms on same axes
drawer.hist("pt", group_by="particle_type")
```

### Facet Mode (subplot grid)

```python
# Separate subplot per group
drawer.hist("pt", group_by="sector", facet=True, ncols=4)
```

---

## Statistics Box

### Basic Usage

```python
# Enable stats box
drawer.hist("x", stats=True)

# Custom fields
drawer.hist("x", stats=["n", "mean", "std", "min", "max"])

# For 2D plots
drawer.scatter("y:x", stats=["n", "mean_x", "mean_y", "corr"])
```

### Default Fields by Plot Type (Phase 13.6.G.DF)

Stats box now auto-detects appropriate fields based on plot type:

| Plot Type | Default Fields |
|-----------|---------------|
| `hist` | n, mean, std |
| `hist2d` | n, mean_x, mean_y, std_x, std_y, corr |
| `scatter` | n, mean_x, mean_y |
| `profile` | n, mean_x, mean_y |
| `hexbin` | n, mean_x, mean_y |

### Available Statistics Fields

| Field | Description | Plot Types |
|-------|-------------|------------|
| `n` | Entry count | All |
| `mean` | Arithmetic mean | 1D |
| `std` | Standard deviation (population, ddof=0) | 1D |
| `min` | Minimum value | 1D |
| `max` | Maximum value | 1D |
| `mean_x` | Mean of x | 2D |
| `mean_y` | Mean of y | 2D |
| `std_x` | Std dev of x (population, ddof=0) | 2D |
| `std_y` | Std dev of y (population, ddof=0) | 2D |
| `corr` | Pearson correlation | 2D |
| `median` | 50th percentile | All (robust) |
| `q25` | 25th percentile | All (robust) |
| `q75` | 75th percentile | All (robust) |
| `mad` | Median absolute deviation | All (robust) |

### Range-Aware Statistics (Phase 13.6.G.DF)

Statistics are now computed **only for data within the specified range**, matching ROOT's TTree::Draw behavior:

```python
# Stats computed on data in [0, 100] only
drawer.hist("pt", range=(0, 100), stats=True)

# 2D range: stats computed within both axis ranges
drawer.hist2d("y:x", range=((0, 10), (-5, 5)), stats=True)
```

### Robust Statistics (Phase 13.6.G.DF)

For non-Gaussian distributions, use robust statistics:

```python
# Per-plot: request specific robust fields
drawer.hist("dEdx", stats=["n", "median", "q25", "q75", "mad"])

# Global setting: change 1D defaults to robust
from dfdraw import set_style
set_style({"stats.robust": True})
drawer.hist("dEdx", stats=True)  # Shows: n, median, MAD
```

**Note:** Robust mode only affects 1D defaults. 2D plots retain standard defaults.

---

## Annotation Methods

### Statistics Box with Expected Values

Add a statistics annotation box showing n, mean (mu), std (sigma), and optionally delta values comparing to expected:

```python
fig, ax, stats = drawer.hist("pull", bins=50)

# Basic statistics box
drawer.add_statistics_box(ax, df['pull'].values)

# With expected values (shows delta)
drawer.add_statistics_box(
    ax, df['pull'].values,
    expected_mean=0.0,    # Shows delta_mu = mu - 0
    expected_std=1.0,     # Shows delta_sigma = sigma - 1
    position='upper right',
    precision=3,
    fontsize=8
)
```

**Parameters:**
- `ax`: matplotlib Axes to annotate
- `values`: array-like data for statistics computation
- `position`: 'upper right', 'upper left', 'lower right', 'lower left'
- `expected_mean`: if provided, shows delta_mu
- `expected_std`: if provided, shows delta_sigma
- `precision`: decimal places (default: 3)
- `fontsize`: text size (default: 8)
- `alpha`: box transparency (default: 0.5)

### Reference Overlay

Add a reference function (Gaussian or custom) scaled to match the histogram:

```python
fig, ax, stats = drawer.hist("pull", bins=50)

# Gaussian N(0,1) overlay
drawer.add_reference_overlay(ax, func='gaussian', mu=0, sigma=1)

# Custom function overlay
def laplace_pdf(x):
    return 0.5 * np.exp(-np.abs(x))

drawer.add_reference_overlay(ax, func=laplace_pdf, label='Laplace')

# Styling options
drawer.add_reference_overlay(
    ax, func='gaussian', mu=0, sigma=1,
    color='blue', linestyle='-', linewidth=2,
    show_legend=True
)
```

**Parameters:**
- `ax`: matplotlib Axes containing histogram
- `func`: 'gaussian' or callable f(x) -> y
- `mu`, `sigma`: parameters for Gaussian
- `label`: legend label (auto-generated for Gaussian)
- `color`, `linestyle`, `linewidth`: line styling
- `show_legend`: whether to display legend (default: True)
- `n_points`: number of points for curve (default: 100)

### Pull Distribution QA Workflow

Typical usage for validating normalized residuals:

```python
# Create pull distribution histogram
fig, ax, stats = drawer.hist('pull', bins=50, title='Pull Distribution')

# Add N(0,1) reference curve
drawer.add_reference_overlay(ax, func='gaussian', mu=0, sigma=1)

# Add statistics box with expected values
drawer.add_statistics_box(
    ax, df['pull'].values,
    expected_mean=0.0,
    expected_std=1.0,
    position='upper right'
)

# Result shows:
#   n = 10,000
#   mu = 0.012
#   sigma = 1.023
#   delta_mu = +0.012
#   delta_sigma = +0.023
```

---

## Styling

### Predefined Styles

```python
from dfdraw import set_style

set_style("default")       # Standard style
set_style("publication")   # Clean, minimal for papers
set_style("presentation")  # Large fonts for slides
set_style("minimal")       # No grid, minimal chrome
```

### Custom Style

```python
set_style({
    "figure.figsize": (10, 8),
    "hist.bins": 100,
    "scatter.alpha": 0.5,
    "stats.show": True,
    "stats.robust": True,  # Use robust defaults (Phase 13.6.G.DF)
})
```

### Save/Load Style

```python
from dfdraw import save_style, load_style

save_style("my_style.json")
load_style("my_style.json")
```

---

## Batch Processing

Generate multiple plots from a specification dictionary:

```python
specs = {
    'pt_dist': {'expr': 'pt', 'bins': 100, 'title': 'pT Distribution'},
    'eta_phi': {'expr': 'phi:eta', 'type': 'hist2d'},
    'dEdx_profile': {'expr': 'dEdx:p', 'type': 'profile', 'bins': 50},
}

results = drawer.draw_batch(specs, save_dir='plots/', verbose=True)
# [1/3] pt_dist -> plots/pt_dist.png
# [2/3] eta_phi -> plots/eta_phi.png
# [3/3] dEdx_profile -> plots/dEdx_profile.png
# Completed: 3/3 (0 errors)
```

### From YAML/JSON

```python
# plots.yaml
# plots:
#   pt_dist:
#     expr: pt
#     bins: 100
#   eta_phi:
#     expr: phi:eta
#     type: hist2d

results = drawer.draw_batch("plots.yaml", save_dir="output/")
```

---

## Integration with AliasDataFrame

`dfdraw` integrates seamlessly with `AliasDataFrame` via duck typing:

```python
from AliasDataFrame import AliasDataFrame
from dfdraw import DFDraw

# AliasDataFrame with axis titles
aDF = AliasDataFrame(df)
aDF.set_axis_title("pt", "p_{T} [GeV/c]")
aDF.set_axis_title("eta", "#eta")

# DFDraw auto-detects AliasDataFrame
drawer = DFDraw(aDF)

# Axis labels automatically use titles
drawer.hist("pt")  # X-axis label: "p_{T} [GeV/c]"
```

---

## Integration with RDataFrameDSL

Use with `draw_figures()` from Phase 12.3:

```python
from RDataFrameDSL import DSLCompiler

dsl = DSLCompiler(schema)
dsl.define("good_pt", "trackPt[isGoodTrack]")

# draw_figures uses dfdraw internally
qa_report = [
    {
        'name': 'track_qa',
        'ncols': 2,
        'plots': [
            {'expr': 'good_pt', 'bins': 50},
            {'expr': 'eta:phi', 'type': 'hist2d'},
        ]
    }
]
dsl.draw_figures(qa_report, rdf)
```

---

## Examples

### QA Dashboard

```python
import pandas as pd
from dfdraw import DFDraw, set_style

# Load data
df = pd.read_parquet("track_data.parquet")
drawer = DFDraw(df)

# Set presentation style
set_style("presentation")

# Batch generate QA plots
qa_specs = {
    'pt': {'expr': 'pt', 'bins': 100, 'selection': 'pt > 0.1'},
    'eta': {'expr': 'eta', 'bins': 50},
    'phi': {'expr': 'phi', 'bins': 50},
    'nHits': {'expr': 'nHits', 'bins': 30},
    'chi2': {'expr': 'chi2', 'bins': 100, 'norm': 'density'},
    'eta_phi': {'expr': 'eta:phi', 'type': 'hist2d', 'bins': [50, 50]},
    'dEdx_p': {'expr': 'dEdx:p', 'type': 'profile', 'bins': 100},
}

results = drawer.draw_batch(qa_specs, save_dir='qa_plots/', dpi=150)
```

### Calibration Residuals

```python
# Profile of residuals vs position
drawer.profile("dy:row", bins=152, error="sem", 
               title="Y Residual vs Pad Row",
               xlabel="Pad Row", ylabel="<dy> [cm]")

# 2D correlation
drawer.hexbin("dy:dz", gridsize=50, norm="log",
              title="Y vs Z Residuals")
```

### Generate Example Gallery

```bash
# Generate example figures for documentation
python examples/generate_gallery.py --output-dir docs/examples
```

---

## Tests

Run the test suite:

```bash
cd UTILS/dfextensions/dfdraw
pytest tests/ -v
```

Test files:
- `test_drawer.py` - Main DFDraw class
- `test_histogram.py` - 1D/2D histogram functions
- `test_scatter.py` - Scatter plot functions
- `test_profile.py` - Profile plot functions
- `test_hexbin.py` - Hexbin plot functions
- `test_hist2d.py` - 2D histogram functions
- `test_facet.py` - Faceted plot layouts
- `test_style.py` - Style management
- `test_batch.py` - Batch processing
- `test_adf_integration.py` - AliasDataFrame integration
- `test_validation_display.py` - Statistics box and reference overlay
- `test_pyarrow_input.py` - PyArrow Table input support
- `test_stats_enhancements.py` - Statistics enhancements (Phase 13.6.G.DF)

**Total tests:** 310 passing

---

## Style Reference

### Default Style Keys

| Key | Default | Description |
|-----|---------|-------------|
| `figure.figsize` | (8, 6) | Figure size in inches |
| `figure.dpi` | 100 | Resolution |
| `hist.bins` | 50 | Default histogram bins |
| `hist.alpha` | 0.7 | Histogram transparency |
| `hist.histtype` | "stepfilled" | Histogram style |
| `scatter.alpha` | 0.7 | Scatter transparency |
| `scatter.size` | 50 | Default marker size |
| `profile.marker` | "o" | Profile marker style |
| `profile.capsize` | 3 | Error bar cap size |
| `stats.show` | False | Show stats by default |
| `stats.position` | "upper right" | Stats box position |
| `stats.robust` | False | Use robust defaults for 1D (Phase 13.6.G.DF) |
| `colors.palette` | "tab10" | Color palette name |

---

## License

Apache 2.0 - See repository root for details.
