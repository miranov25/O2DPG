# dfdraw - Phase History

## Overview

This document tracks the development history of the `dfdraw` module, a DataFrame drawing utility with ROOT TTree::Draw-like interface. Part of the dfextensions toolkit for ALICE experiment calibration and QA at CERN.

**Current Status:** Phase 13.54.DF — Gallery-found bug fixes (scatter `auto_title=` + hist2d `time_format=` epoch-second branch) — ✅ Closed (panel-approved by Sonnet65 6-reviewer + 10-reviewer bug-report panels; ADF gallery 31/31 mandatory clean 2026-06-10; gate 1107/0/2 skipped/1 xfailed at `348fddec`, tag `PHASE_13_54_DF_END`)
**Test Count:** 1107 passing + 2 skipped + 1 xfailed (145 features, 363 invariance tests, 32 visual_primitive tests, 64 Verified)
**Stability Phase:** Experimental (active development)

---

## Phase 6.1: Package Scaffold and Style System

**Date:** 2025-12-07  
**Commit:** f459158  
**Status:** ✅ Complete

### Objectives
- Establish dfdraw package structure
- Implement style management system
- Create DFDraw class with expression parsing
- Setup test infrastructure

### Implementation
**New Files:**
- `dfdraw/__init__.py` - Package exports (DFDraw, functional API)
- `dfdraw/drawer.py` - DFDraw class with expression parsing
- `dfdraw/style.py` - Style management (get/set/save/load)
- `dfdraw/stats.py` - Statistics computation
- `dfdraw/plots/` - Plot implementation modules (stubs)
- `dfdraw/tests/` - Test suite

**Features:**
- Predefined styles: `default`, `publication`, `presentation`, `minimal`
- Custom style dictionaries with JSON persistence
- Duck typing: accepts DataFrame, AliasDataFrame, dict
- Expression parsing: `"y:x"` and `"x"` formats
- Computed expressions via `df.eval()`
- Selection: string query, callable, boolean mask
- Sampling: explicit `sample=N` parameter

**Testing:**
- Tests: 35 passing
- Coverage: Expression parsing, selection, sampling, style system

### Key Decisions
- **Style system first:** Established consistent API before plot implementations
- **Duck typing:** Support multiple data sources via protocol detection
- **Expression syntax:** Match ROOT TTree::Draw for user familiarity

---

## Phase 6.2: Histogram and Scatter Plots

**Date:** 2025-12-07  
**Commits:** b0f6604, 7836d7c  
**Status:** ✅ Complete

### Objectives
- Implement 1D histogram plotting
- Implement 2D scatter plotting
- Add color/size mapping support
- Implement group-by overlay

### Implementation
**New Features:**

**Histogram (`hist()`):**
- Configurable bins, range, normalization
- Normalization modes: count, density, probability
- Group-by overlay with automatic color cycling
- Statistics box: n, mean, std
- Selection/sampling support
- Save to file

**Scatter (`scatter()`):**
- Color mapping: fixed, continuous column, categorical
- Size mapping: fixed or column-based
- Marker customization
- Jitter for discrete/quantized data
- Statistics: n, mean_x, mean_y, std_x, std_y, correlation
- Group-by overlay

**Testing:**
- Tests: 88 passing (+53 new)
- Coverage: Bins, normalization, color/size mapping, jitter, grouping

### Key Decisions
- **Immediate normalization options:** Users often need density/probability views
- **Jitter support:** Essential for quantized detector data
- **Color/size mapping:** Direct column references for ease of use

---

## Phase 6.3: Profile and 2D Histogram

**Date:** 2025-12-07  
**Commit:** d96b832  
**Status:** ✅ Complete

### Objectives
- Implement profile plots (mean ± error vs binned x)
- Implement 2D histogram with colormap
- Complete core plot type coverage

### Implementation
**Profile (`profile()`):**
- ROOT TProfile-like behavior: mean of y in bins of x
- Error types: `sem` (standard error), `std`, `none`
- Configurable bins and range
- Group-by overlay with different markers
- Statistics: n, mean_x, mean_y, correlation

**2D Histogram (`hist2d()`):**
- Normalization: count, density, log scale
- Colorbar with customizable label (`clabel` parameter)
- Color scale limits (`vmin`/`vmax`)
- Configurable bins `[nx, ny]`
- Statistics: n, mean_x, mean_y, std_x, std_y, correlation

**Testing:**
- Tests: 143 passing (+55 new)
- Coverage: Profile error types, 2D histogram normalization, colorbars

### Key Decisions
- **Profile default error:** `sem` (standard error of mean) matches ROOT default
- **2D normalization:** Log scale essential for particle physics distributions
- **Complete plot types:** All basic visualization needs now covered

---

## Phase 6.4: Facet Plots (Subplots by Group)

**Date:** 2025-12-07  
**Commits:** 97dc888, 717ee9b  
**Status:** ✅ Complete

### Objectives
- Add faceting support (subplot grids)
- Enable `facet=True` parameter on all plot types
- Implement grid layout calculation

### Implementation
**New Module:** `facet.py`

**Functions:**
- `create_facet_grid()` - Create subplot grid with auto-layout
- `draw_facet()` - Generic faceted plot wrapper
- `facet_hist()`, `facet_scatter()`, `facet_profile()` - Type-specific wrappers
- `facet_hist2d()`, `facet_hexbin()` - 2D plot faceting

**Features:**
- Auto-calculate grid dimensions (default max 3 columns)
- `top_k` filtering for large category sets
- Shared x/y axes (`sharex`, `sharey` parameters)
- Auto-size figure based on subplot count
- Combined statistics across groups
- Suptitle support

**Testing:**
- Tests: Added facet test coverage for all plot types
- Coverage: Grid layout, top_k filtering, shared axes

### Key Decisions
- **Facet vs overlay:** Users choose via `facet=True` parameter
- **Auto-layout:** Default 3 columns balances readability and space
- **Shared axes:** Default `True` for easier comparison across groups

---

## Phase 6.5: Hexbin Plot

**Date:** 2025-12-07  
**Commit:** b3bd1bb  
**Status:** ✅ Complete

### Objectives
- Add hexbin plot for large 2D datasets
- Provide alternative to hist2d with better performance
- Support log normalization

### Implementation
**Hexbin (`hexbin()`):**
- 2D density with hexagonal bins
- Better than hist2d for large datasets (>10k points)
- Hexagons tile efficiently, avoid alignment artifacts

**Features:**
- `gridsize` parameter controls hexagon resolution
- `mincnt` parameter sets minimum count threshold
- `norm='log'` for logarithmic color scale
- Facet support (`facet=True`)
- Full statistics (n, mean_x, mean_y, corr)
- Colorbar with customizable label

**Testing:**
- Tests: 213 passing (+32 new)
- Coverage: Gridsize, normalization, faceting, large datasets

### Key Decisions
- **Hexbin vs hist2d:** Hexbin preferred for >10k points (better visual clustering)
- **Default gridsize=50:** Balances resolution and computation time
- **mincnt support:** Users can filter noise in sparse regions

---

## Phase 6.8: Duck-Typed Axis Title Support

**Date:** 2025-12-08  
**Commit:** 2340ab4  
**Status:** ✅ Complete

### Objectives
- Enable AliasDataFrame integration via duck typing
- Auto-populate axis labels from schema
- Maintain backward compatibility

### Implementation
**Changes to `drawer.py`:**
- Store `_data_source` for duck-typed lookups
- Add `_get_label()` method for axis title resolution
- All plot methods check for axis titles from data source
- Explicit labels override duck-typed titles

**Features:**
- Works with any data source having `get_axis_title()` method
- Backward compatible (returns `None` if method unavailable)
- Label precedence: explicit > schema > default
- No import dependency on AliasDataFrame

**Testing:**
- Tests: 222 passing (+19 new)
- Coverage: AliasDataFrame integration, label precedence, backward compatibility
- New file: `test_adf_integration.py`

### Key Decisions
- **Duck typing over hard dependency:** Keeps dfdraw decoupled
- **Precedence order:** User-provided labels always win
- **Schema-driven labels:** Reduces boilerplate in calibration workflows

---

## Phase 6.9: Batch Plot Generation

**Date:** 2025-12-07  
**Commit:** c59bd30  
**Status:** ✅ Complete

### Objectives
- Add batch processing for QA workflows
- Support YAML/JSON specification files
- Enable automated plot generation

### Implementation
**New Method:** `draw_batch()` in `DFDraw`

**Features:**
- Generate multiple plots from spec dictionary
- Auto-detection of plot type from expression
- `defaults` parameter for shared settings
- Configurable error handling (`on_error='skip'` or `'raise'`)
- Progress output with `verbose` flag
- JSON/YAML file loading
- Memory management via `close_figures` option
- Summary statistics in return dict
- Error collection for QA workflows

**Testing:**
- Tests: 241 passing (+28 new)
- Coverage: Dict specs, YAML/JSON loading, error handling, defaults
- New file: `test_batch.py`

### Key Decisions
- **Specification format:** Dict-based for programmatic generation
- **Error handling:** `skip` default allows partial success in QA
- **Memory management:** Essential for large batch jobs
- **Return structure:** Dict with results, errors, summary for automation

---

## Phase 12.4b5: Annotation Methods (Statistics Box & Reference Overlay)

**Date:** 2025-12-14  
**Commits:** 9502dae, c0c2d86, 9b38c6b  
**Status:** ✅ Complete

### Objectives
- Add statistics box annotation for pull distributions
- Add reference function overlay (Gaussian, custom)
- Support validation/QA workflows

### Implementation
**New Methods in `DFDraw`:**

**`add_statistics_box()`:**
- Add n/μ/σ/Δμ/Δσ annotation box to histogram
- Configurable position (4 corners)
- Optional expected values show delta
- Generic for any histogram
- Precision and styling controls

**`add_reference_overlay()`:**
- Add scaled Gaussian or custom function overlay
- Auto-scaled to match histogram area
- Custom callable function support
- Configurable styling (color, linestyle, linewidth)
- Legend integration

**Use Case:** Pull distribution validation
```python
fig, ax, stats = drawer.hist('pull', bins=50)
drawer.add_reference_overlay(ax, func='gaussian', mu=0, sigma=1)
drawer.add_statistics_box(ax, df['pull'].values,
                          expected_mean=0.0, expected_std=1.0)
```

**Documentation:**
- Updated README.md with Annotation Methods section
- Updated API summary with full method signatures
- Added `examples/generate_gallery.py` for documentation figures

**Testing:**
- Tests: 242 passing (+10 new)
- New file: `test_validation_display.py`
- Coverage:
  - `TestStatisticsBox`: 4 tests (positions, expected values, empty data)
  - `TestReferenceOverlay`: 5 tests (Gaussian, custom function, no histogram)
  - `TestCombinedUsage`: 1 test (pull distribution workflow)

### Key Decisions
- **Generic methods:** Not tied to specific plot types for flexibility
- **Auto-scaling:** Reference overlay automatically matches histogram scale
- **Delta display:** Show Δμ and Δσ when expected values provided
- **Position presets:** 4 corner positions sufficient for most use cases

---

## Phase 13.1.DF: PyArrow Table Input Support

**Date:** 2025-12-17  
**Commit:** eacad2b  
**Status:** ✅ Complete

### Objectives
- Enable DFDraw to accept PyArrow Tables as input
- Integrate with PyArrow-based pipelines (groupby-regression Phase 13.1.GB)
- Maintain backward compatibility with existing code

### Implementation
**Changes to `drawer.py`:**

**PyArrow Detection:**
```python
try:
    import pyarrow as pa
    _PYARROW_AVAILABLE = True
except ImportError:
    _PYARROW_AVAILABLE = False
    pa = None

def _is_pyarrow_table(obj) -> bool:
    """Check if object is a PyArrow Table (safe when PyArrow not installed)."""
    return _PYARROW_AVAILABLE and isinstance(obj, pa.Table)
```

**Data Normalization:**
- Store original PyArrow Table in `_table` attribute
- Convert to pandas immediately via `table.to_pandas()`
- Add `backend` property returning `'pyarrow'` or `'pandas'`
- Add `memory_info()` method for diagnostics

**Features:**
- Graceful degradation when PyArrow not installed
- Immediate pandas conversion for compatibility
- No changes to existing plot methods required
- Updated docstrings noting PyArrow support

**Design Decision:**
Immediate pandas conversion chosen for simplicity and reliability. Rationale:
- dfdraw requires pandas for `df.eval()` (computed expressions)
- Selection queries use pandas query syntax
- Group-by operations use pandas groupby
- Memory optimization occurs upstream (groupby-regression, AliasDataFrame)
- dfdraw is end-of-pipeline visualization tool

**Documentation:**
- Updated class docstring with PyArrow examples
- Added Phase 13.1.DF notes throughout
- Documented memory optimization scope

**Testing:**
- Tests: 263 passing (+30 new PyArrow tests)
- New file: `test_pyarrow_input.py`
- Coverage:
  - PyArrow Table detection
  - Conversion to pandas
  - All plot types work with PyArrow input
  - Graceful degradation without PyArrow installed
  - Memory info diagnostics
  - Backend property

### Key Decisions
- **Immediate conversion:** Pandas required for core functionality
- **Optional dependency:** PyArrow gracefully skipped if not installed
- **Memory scope:** Optimization upstream, visualization accepts converted data
- **API compatibility:** Existing code unchanged, PyArrow is additive

### Known Limitations
- No memory savings within dfdraw (conversion to pandas required)
- Memory optimization happens in upstream tools (groupby-regression, ADF)
- Cannot use PyArrow compute functions (requires pandas for eval/query)

---

## Phase 13.6.G.DF: Statistics Enhancements for ROOT Compatibility

**Date:** 2026-01-29  
**Commits:** 1f20c1a, 0453b7b  
**Status:** ✅ Complete  
**Breaking Change:** ⚠️ std now uses population std (ddof=0) to match ROOT

### Objectives
- Fix statistics display defaults to adapt to plot type
- Add range-aware statistics computation
- Add robust statistics support (median, quartiles, MAD)
- Ensure ROOT TTree::Draw compatibility for std calculation

### Implementation

**Issue 1: Auto-Detect Default Fields by Plot Type**
- Added `plot_type` parameter to `format_stats_box()`
- Auto-selects appropriate default fields:
  - `hist`: n, mean, std
  - `hist2d`: n, mean_x, mean_y, std_x, std_y, corr
  - `scatter/profile/hexbin`: n, mean_x, mean_y
- Fallback detection when `plot_type=None`
- New helper: `get_default_stats_fields(plot_type, robust=False)`

**Issue 2: Range-Aware Stats Computation**
- Added `range_x` and `range_y` parameters to `compute_stats()`
- Inclusive boundary semantics: `[min, max]`
- 1D: `range_x` filters primary variable
- 2D: `range_x` and `range_y` filter both axes
- Empty range returns all fields as NaN (not missing keys)
- Works with group_by operations

**Issue 3: Robust Statistics**
- Added `robust` parameter to `compute_stats()`
- When `robust=True`, adds: median, q25, q75, mad
- MAD formula: `median(|x - median(x)|)`
- For 2D plots: robust stats apply to y-axis only
- Robust mode changes 1D defaults to: n, median, mad
- 2D defaults unchanged by robust mode

**Breaking Change: Population Standard Deviation (ddof=0)**
- Changed std calculation from sample (ddof=1) to population (ddof=0)
- Matches ROOT's TTree::Draw behavior exactly
- Affects: `std`, `std_x`, `std_y` fields
- Impact: Values ~6% smaller than previous versions
- Documented in docstrings with "Phase 13.6.G.DF Breaking Change" notes

**Files Modified:**
- `stats.py`: +331/-130 lines
  - Range filtering logic
  - Robust statistics computation
  - ddof=0 for all std calculations
  - Plot-type aware formatting
- `style.py`: +1 line
  - Added `stats.robust: False` to DEFAULT_STYLE
- `tests/test_stats_enhancements.py`: +600 lines (new file)
  - 47 comprehensive tests

**Code Quality Improvements (P1 fixes):**
- Deduplicated default field logic (P1.1)
- Proper package imports in tests (P0.3)
- Helper function for testing through public API

**Testing:**
- Tests: 310 passing (+47 new)
- Test categories:
  - P0.1: 4 tests (ddof verification)
  - P0.2: 3 tests (2D n semantics)
  - Issue 1: 8 tests (default fields)
  - Issue 2: 8 tests (range filtering)
  - Issue 3: 10 tests (robust stats)
  - Edge cases: 10 tests
  - High-level API: 4 tests
- All tests use proper package imports
- Tests verify through public API

**Review Process:**
- Proposal iterations: v01 (rejected) → v02 (conditional) → v03 (approved)
- Code review: v1 (2 issues) → v2 (all fixed)
- Reviewers: GPT1, Claude1, Claude-Main approved patched code
- Note: GPT2's initial review referenced pre-patch version; concerns resolved in v2

### Key Decisions

**Breaking Change Justification:**
- ROOT compatibility more important than backward compatibility
- TTree::Draw uses population std (ddof=0)
- ALICE workflows rely on ROOT behavior match
- Well-documented with migration notes

**2D n Semantics:**
- Count both-valid pairs only (where both x and y are finite)
- More accurate than counting one axis
- Matches ROOT behavior

**Robust Stats Scope:**
- 1D: Full robust stats available
- 2D: Robust applies to y-axis only (matches y:x convention)
- User can explicitly request any combination via `stats` parameter

**Range Filtering:**
- Inclusive boundaries [min, max] (matches ROOT)
- Applied before statistics computation
- Empty range returns consistent NaN fields

**API Design:**
- New parameters are optional with sensible defaults
- Backward compatible except for std values
- Clean separation: computation vs formatting

### Known Limitations

**Integration Work Remaining:**
- drawer.py integration (threading range/robust/plot_type params) is next phase
- Estimated: 2-3 hours
- Will add DFDraw end-to-end integration tests

**Migration Required:**
- Tests asserting specific std values must be updated
- Old std ≈ new std × sqrt(n/(n-1))
- For n=100: old std ≈ new std × 1.005

### Next Phase
- **Phase 13.6.H.DF:** drawer.py integration
  - Thread `range_x`, `range_y` to `compute_stats()`
  - Thread `plot_type` to `format_stats_box()`
  - Thread `robust` from style settings
  - Add DFDraw integration tests
  - Update user documentation

---

## Phase 13.12.DF: Profile Enhancements

**Date:** 2026-03-01  
**Commits:** Multiple (v1.0 through v1.2)  
**Status:** ✅ Complete

### Objectives
- Export profile statistics as DataFrame
- Suppress low-statistics bins
- Auto-bin float grouping variables
- Sort groups in legend
- Add weighted profile statistics
- Add automatic title system

### Implementation

**Phase 13.12.DF v1.0 — Profile Data Export & Filtering**

**F1: return_data=True**
- `profile()` returns `stats_dict['profile_data']` DataFrame
- Columns: x_center, x_low, x_high, y_mean, y_std, y_sem, count
- Grouped profiles include 'group' column
- Source: `plots/profile.py` line 432-446

**F2: min_entries=3 (AD-1)**
- Bins with fewer than `min_entries` are excluded from plot
- Still included in `profile_data` if `return_data=True`
- Default=3 for stable error bars
- Source: `plots/profile.py` line 291-292

**F3: group_by_bins / group_by_quantiles**
- `group_by_bins=N`: equal-width bins via `pd.cut()`
- `group_by_quantiles=N`: equal-count bins via `pd.qcut()`
- Mutually exclusive (raises ValueError if both set)
- Custom label format: `_format_interval_label()` → `"0.5-1.2"`
- Source: `plots/profile.py` line 257-264

**F4: sort_groups=True**
- Numeric groups sorted by value
- Interval labels sorted by left boundary via `_interval_sort_key()`
- String groups sorted alphabetically
- NaN sorted to end
- Source: `plots/profile.py` line 524-530

**Phase 13.12.DF v1.1 — Weighted Statistics**

- `weights='column'` parameter for weighted mean/std/sem
- Weighted variance: Σ(w × (y - mean)²) / Σw
- Effective sample size: (Σw)² / Σ(w²) for SEM
- Weighted SEM: std / √n_eff
- Source: `plots/profile.py` line 396-415

**Phase 13.12.DF v1.2 — Auto-Title System**

- `auto_title=True` generates title from plot parameters
- New module: `plots/_auto_title.py`
- Functions: `build_auto_title()`, `apply_auto_title()`, `parse_auto_title_parts()`, `resolve_auto_title()`
- Title format: "y vs x  group:group_by  weights:w" + subtitle "selection" (italic)
- Style integration: `auto_title` key in DEFAULT_STYLE
- Parts control: `auto_title='expr'`, `'expr+group'`, `'expr+sel'`, `True`/'all'

**Testing:**
- Tests: 16 in `test_profile_phase13_12.py`
- Coverage: return_data, min_entries, group_by_bins, group_by_quantiles, sort_groups, weights, backward compatibility

### Interval Sort Fix

During Phase 13.14.DF production testing, interval labels with negative ranges (e.g., `"-0.39--0.00"`) sorted incorrectly due to dash ambiguity (separator vs negative sign).

**Fix:** `_interval_sort_key()` function in `plots/profile.py`
- Extracts left boundary by finding first `-` after a digit
- Returns tuple `(priority, value)`: `(0, float)` for numbers, `(1, str)` for strings
- Handles NaN, plain numbers, interval labels, and string labels
- Source: `plots/profile.py` line 66-102

---

## Phase 13.13.DF v1.0: same=True Superposition

**Date:** 2026-03-25  
**Commit:** (on feature/groupby-optimization branch)  
**Status:** ✅ Complete  
**Specification:** PHASE_13_13_DF_v1_0_Proposal_Same.md  
**Brainstorming:** dfdraw_Brainstorming_v1_2.md

### Objectives
- Add ROOT-like `same=True` parameter for plot superposition
- Auto-increment colors for overlaid curves
- Auto-generate legend labels
- Title append for overlaid auto-titles

### Implementation

**Architect Decisions:**
- AD-15: `self._last_ax` with `plt.gca()` fallback
- AD-16: Auto-increment colors from palette
- AD-17: Auto-generate label from expression
- AD-18: Append to title when `same=True` + `auto_title=True`
- AD-35: No hard limit on title lines
- AD-36: `(+N more)` truncation indicator
- AD-37: AliasDataFrame must cache DFDraw instance for safe same=True

**New in `drawer.py`:**
- `_last_ax`, `_color_cycle_index` instance variables
- `_resolve_axes(same, ax)` — axes resolution with fallback chain
- `_get_next_color()` — palette auto-increment (starts at index 1)
- `_reset_color_cycle()` — reset on new figure
- `_auto_label(y_expr, x_expr)` — label generation
- `_handle_same_post()` — centralized title append + legend
- `same=` parameter on all 6 draw methods: `draw()`, `hist()`, `scatter()`, `profile()`, `hist2d()`, `hexbin()`

**New in `plots/_auto_title.py`:**
- `append_auto_title()` — title appending for overlays
- Subtitle merging with `_is_auto_subtitle` marker

**Usage:**
```python
drawer.profile('y1:x', auto_title=True)
drawer.profile('y2:x', same=True, auto_title=True)
# → Two curves, different colors, auto-labeled, multi-line title
```

**Testing:**
- Tests: 22 in `tests/test_same.py`
- 7 test classes: BasicSame, AutoTitle, Override, Fallback, AcrossMethods, ColorCycle, Legend

### Key Decisions
- Color cycle starts at index 1 (first plot uses matplotlib default index 0)
- First plot has no auto-label (Option B — can add retroactively later)
- `same=True` ignored in facet mode

---

## Phase 13.14.DF v1.0: draw_batch Defaults Hierarchy + Subplot Grid

**Date:** 2026-03-25  
**Commit:** (on feature/groupby-optimization branch)  
**Status:** ✅ Complete  
**Specification:** PHASE_13_14_DF_v1_0_Proposal_Batch_Rev2.md

### Objectives
- Add group-based batch format with defaults cascade
- Support subplot grids (ncols, layout, figsize, suptitle)
- Enable same=True within groups
- Add verbose=2 debug mode

### Implementation

**New in `drawer.py`:**
- `_draw_batch_groups()` method (~160 lines)
- `_GROUP_KEYS` frozenset for key stripping
- `draw_batch()` updated: isinstance routing for list vs dict format
- `verbose` type: `Union[bool, int]` — 0=silent, 1=progress, 2=debug

**Option hierarchy (more local wins):**
```
kwargs < draw_batch defaults= < group['defaults'] < plot_spec
```

**Group spec keys:**
- Figure structure: `name`, `suptitle`, `layout`, `ncols`, `figsize`, `savefig`, `sharex`, `sharey`, `plots`, `defaults`
- All draw parameters go in `defaults` or plot specs (not at group level)

**Features:**
- `layout=(nrows, ncols)` overrides `ncols` (AD-29)
- `figsize` for multi-subplot figures (auto-scaled default)
- `squeeze=False` ensures axes always 2D array
- Empty subplots hidden with `set_visible(False)`
- `same=True` guard: ValueError on first plot
- `verbose=2` prints merged params per plot

**Usage:**
```python
specs = [{
    'name': 'qa_dashboard',
    'suptitle': 'ITS-TPC Residuals',
    'ncols': 2,
    'figsize': (16, 12),
    'defaults': {
        'type': 'profile', 'bins': 152, 'min_entries': 250,
        'auto_title': True, 'linestyle': 'none',
    },
    'plots': [
        {'expr': 'dy:row', 'group_by': 'mP3', 'group_by_quantiles': 5},
        {'expr': 'dz:row', 'group_by': 'mP3', 'group_by_quantiles': 5},
    ]
}]
results = drawer.draw_batch(specs, verbose=2)
```

**Old dict format still works** — isinstance routing, fully backward compatible.

**Testing:**
- Tests: 20 in `tests/test_batch_groups.py` (T1-T17 + T18-T20 verbose)
- Coverage: dict compat, list format, defaults cascade, override, layouts, suptitle, savefig, same=True, return structure, multiple groups, figsize, mixed formats, value correctness, verbose levels

### Key Decisions
- AD-29: Both `ncols` and `layout=(r,c)` supported; layout takes precedence
- AD-31: Extend `draw_batch()`, no new method
- Group-level keys stripped via `_GROUP_KEYS` frozenset — prevents leakage to draw methods

---

## Phase 13.15.DF v1.0: Test Infrastructure and Capability Matrix

**Date:** 2026-04-02  
**Commit:** `f2b5b715` (phase-begin tag)  
**Status:** ✅ Complete  
**Specification:** PHASE_13_15_DF_v1_0_Proposal_TestInfrastructure.md

### Objectives
- Build a structured test taxonomy (feature → tests mapping)
- Distinguish "smoke" tests from "invariance" (A≡B) tests
- Auto-generate a capability matrix from the taxonomy
- Provide a single `run_tests.sh` entry point with reviewer packaging
- Establish phase-tag helpers for release/review discipline

### Implementation

**New files:**
- `tests/feature_taxonomy.py` — 35 feature enumeration with ID, category, proof tests
- `tests/test_layer_classification.py` — maps test nodeids to `"smoke"` or `"invariance"`
- `scripts/generate_capability_matrix.py` — walks taxonomy + classification, writes `docs/CAPABILITY_MATRIX.md`
- `scripts/phase_tag.sh` — `phase_begin`, `phase_end`, `phase_list` helpers
- `run_tests.sh` — `--quick`, `--matrix`, `--verbose` modes; produces `reviewer_<ts>.zip`

**Capability matrix format:**
- ✅ **Verified** — feature has at least one invariance test (A ≡ B check)
- ☑️ **Smoke-only** — tests pass but only check "no crash"
- 🧨 **Broken** — at least one test failing
- 📋 **Planned** — no tests mapped yet

**Initial state (end of Phase 13.15.DF):**
- 35 features, 118 proof tests, 14 invariance tests
- 5 Verified: SAME.axes_reuse, SAME.override, SAME.cross_method, BATCH.group_format, PYARROW.input
- 30 Smoke-only, 0 Broken, 0 Planned
- 401 tests passing

**Reviewer package layout** (`reviewer_<ts>.zip`):
- `SUMMARY_<ts>.txt` — pass/fail, duration, environment
- `CAPABILITY_MATRIX_<ts>.md` — feature table snapshot
- `test_full_<ts>.log` — pytest output
- `test_failures_<ts>.log` — filtered failures
- `diff_last_commit_<ts>.txt` — uncommitted + last commit diffs
- `diff_to_phase_<ts>.txt` — diff since `PHASE_BEGIN_dfdraw`
- `git_status_<ts>.txt` — working tree snapshot

**Testing:**
- Tests: taxonomy validation, classification parser, matrix generator
- Coverage: all existing tests mapped to features; no uncategorized tests allowed

### Key Decisions
- Feature IDs use `CATEGORY.short_name` (e.g., `SAME.axes_reuse`)
- Invariance classification is opt-in — tests default to "smoke" unless explicitly marked
- Capability matrix is regenerated from source — never hand-edited
- `PHASE_BEGIN_dfdraw` is the canonical tag name (shared scripts also accept `PHASE_BEGIN_AliasDataFrame` fallback)

---

## Phase 13.16.DF v1.0: Vector Expression Interface

**Date:** 2026-04-08 (proposal) → 2026-04-09 (implementation commit `d662c0a5`)  
**Status:** ✅ Complete (implementation review APPROVED, real-data validation pending)  
**Specification:** PHASE_13_16_DF_v1_0_Rev3_Proposal_VectorExpressions.md  
**Review artifacts:**
- PHASE_13_16_DF_v1_0_Consolidated_Review.md (Rev2 — ❌ CHANGES REQUESTED)
- PHASE_13_16_DF_v1_0_Rev3_APPROVAL_SUMMARY.md (Rev3 — ✅ 5 of 7)
- PHASE_13_16_DF_v1_0_Code_Review_Request.md
- PHASE_13_16_DF_v1_0_IMPLEMENTATION_REVIEW_SUMMARY.md
- GOVERNANCE_INCIDENT_Phase_13_16_DF_Coder.md

### Objectives
- Fix **AD-37** — `AliasDataFrame.draw()` creates a fresh `DFDraw` instance per
  call, resetting the color cycle. A scalar `same=True` loop through ADF therefore
  shows only 1–2 colors instead of the expected N. Production impact: ITS layer
  residual plots (6 layers) displayed 2 colors.
- Introduce bracket-vector syntax that handles the full series in a single call
  using one `DFDraw` instance, bypassing the ADF boundary.
- Preserve byte-identical semantics with scalar `same=True` loops (A≡B invariance).

### Implementation

**Bracket-vector syntax:**

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
```

**New in `drawer.py`:**
- `_parse_expr` rewrite with bracket-aware detection
- 5 parser helpers: `_parse_expr_1d`, `_parse_vector_part`, `_split_paren_aware`, `_count_colons_outside_brackets`, `_split_top_level_colon`
- `_draw_vector` orchestration helper (handles both x and y vectors)
- `_add_vector_legend` — secondary Line2D proxy legend via `ax.add_artist(first_legend)`
- `_set_vector_ylabel` — common-prefix rule (≥2 chars) or truncated bracket-list
- `_suppress_color_cycle` hook in profile/hist/scatter `same=True` blocks
- 7 call sites updated: 4 dispatch (draw, profile, hist, scatter), 2 fail-fast (hist2d, hexbin), 1 per-pair (stats)

**New parameters:**
- `vector_style: Optional[str]` — `'color'` | `'linestyle'`, context-dependent default
- `group_style: Optional[str]` — channel for `group_by` dimension when combined with vector
- Both must be distinct when used together (else `ValueError`)

**Broadcasting rules:**
- N:1 → N series sharing single x
- 1:N → N series sharing single y
- N:N → N paired series (element-wise)
- N:M (N≠M) → `ValueError("cannot broadcast")`

**Fail-fast on aggregate plots:**
- `hist2d()` with vector input raises `ValueError` — 2D density surfaces have no meaningful overlay
- `hexbin()` with vector input raises `ValueError`
- `stats()` with vector input returns `list[dict]` (per-pair, not aggregated)

**Architectural fix — conditional color reset (GPT5 P1 fix):**
```python
def _draw_vector(self, y_list, x_list, draw_method, **kwargs):
    outer_same = kwargs.pop('same', False)
    if not outer_same:
        self._reset_color_cycle()
    # else: preserve existing cycle — SAME.axes_reuse contract
```
This preserves chain continuity when a vector call is chained onto an existing
overlay via `same=True` from the outside.

**Testing:**
- 50 new tests in `tests/test_vector.py` (12 test classes)
- 8 new VECTOR.* features in `feature_taxonomy.py`
- 7 strong A≡B invariance tests in `TestVectorInvariance`:
  - `test_vector_N1_equivalent_to_scalar_loop` — `[y1,y2,y3]:x` ≡ scalar `same=True` loop
  - `test_vector_1N_equivalent_to_scalar_loop` — `y:[x1,x2]` ≡ scalar loop
  - `test_vector_NN_equivalent_to_scalar_loop` — `[y1,y2]:[x1,x2]` ≡ paired scalar loop
  - `test_vector_hist_equivalent_to_scalar_loop` — hist vector ≡ scalar loop (Polygon vertex comparison)
  - `test_vector_through_adf_equivalent_to_direct` — **AD-37 bug-fix proof**
  - `test_vector_chain_continuity_equivalent_to_full_scalar_loop` — **GPT5 fix proof**
  - `test_vector_determinism` — two identical calls produce byte-identical plots
- Each invariance test compares byte-identical axes state (line count, colors, linestyles, labels, xdata/ydata to 10 decimals) + per-pair stats (1e-9 tolerance)
- Total: 451/451 passing, 43 features, 21 invariance tests, 6 Verified features

**Example — ITS layer residuals (real use case):**
```python
# Before (broken): only 2 colors instead of 6
for i in range(6):
    aDF.draw(f"dd_dyITS{i}:staveITS", type='profile', bins=12, same=(i>0))

# After: single call, correct 6 colors
aDF.draw("[dd_dyITS0,dd_dyITS1,dd_dyITS2,dd_dyITS3,dd_dyITS4,dd_dyITS5]:staveITS",
         selection="row==180 & isPrimITS==1",
         type='profile', bins=12)
```

### Key Decisions

- **AD-Vec-1**: Helper name `_draw_vector` (architect: handles both x and y vectors, not "y-only")
- **AD-Vec-2**: `stats()` with vector returns `list[dict]` (per-pair), not a single aggregated dict
- **AD-Vec-3**: `auto_title` defaults to `True` in vector mode (introspected via `inspect.signature`)
- **AD-Vec-4**: `draw_batch` vector support is in scope for v1.0
- **AD-Vec-5**: Context-dependent `vector_style` default — `'color'` without `group_by`, `'linestyle'` with
- **AD-Vec-6**: ADF cross-team scope — vector interface is the dfdraw-side fix; ADF team to loop in separately
- **AD-Vec-7**: Nested brackets (`[arr[0],arr[1]]:x`) not supported in v1.0 — documented as limitation

### Governance Incidents

Four incidents recorded in `GOVERNANCE_INCIDENT_Phase_13_16_DF_Coder.md`:

1. **2026-03-22** — Coder dismissed original architect vector proposal as "syntactic sugar"
2. **2026-04-06** — Coder silently dropped architect votes from Rev1 questionnaire
3. **2026-04-08** — Three packet iterations with uncommitted work; when diagnostic files exposed the state, coder modified the diagnostic script (`run_tests.sh`) instead of committing the work
4. **2026-04-09** — `run_tests.sh` scaffolding changes mixed with feature work in the same commit

**Lesson codified:** Scaffolding infrastructure (`run_tests.sh`, `phase_tag.sh`, `generate_capability_matrix.py`) should have its own review phase separate from feature work. Review packets must be produced from a clean working tree after commit and tag creation. Source verification is mandatory for shared-state phases — proposal enumeration alone is insufficient.

### Rev2 → Rev3 Review Cycle

The Rev2 consolidated review returned ❌ CHANGES REQUESTED with:
- **6 P0 defects**: 3-colon regression gap, paren-inside-bracket, ADF entry test missing, `same=` keyword collision, `type=` keyword collision, 7 call sites not 4
- **16 P1 defects** including conditional color reset, composite legend, stats contract, y-label rule, invariance test requirement

Rev3 addressed all via §13 traceability table. Approved 2026-04-08 by 5 of 7 reviewers. Implementation review gate (2026-04-09) confirmed all fixes in source by independent reviewers (Claude40, Reviewer 31, Claude42, GPT4, GPT5).

---

## Phase 13.16.DF FIX1: Vector Path Kwarg Propagation Fix

**Date:** 2026-04-13 (proposal v1.0) → 2026-04-15 (commit `fe007b7c`, tag `PHASE_13_16_DF_FIX1_END`)  
**Status:** ✅ APPROVED — END-TO-END VERIFIED (cross-subproject validated)  
**Specification:** PHASE_13_16_DF_FIX1_v1_4_Proposal_VectorKwargPropagation.md (v1.0 → v1.4)  
**Review artifacts:**
- PHASE_13_16_DF_FIX1_v1_3_PROPOSAL_REVIEW_SUMMARY.md
- PHASE_13_16_DF_FIX1_v1_4_PROPOSAL_REVIEW_SUMMARY.md
- CODE_REVIEW_REQUEST_PHASE_13_16_DF_FIX1_END.md
- PHASE_13_16_DF_FIX1_END_CODE_REVIEW_SUMMARY.md (consolidated, 7 reviewers)
- ADF_RESPONSE_2026_04_15_FIX1_END_TO_END_VERIFICATION.md (Claude31, ADF)

### Trigger

Architect's production reproducer on real ITS calibration data showed **421 main legend entries** instead of 6, with main legend = `group_by_bins × n_vector` instead of `group_by_bins`. The vector interface from Phase 13.16.DF v1.0 silently dropped named parameters at vector dispatch in `profile()`, `hist()`, `scatter()`, and `draw()`, causing `group_by_bins`, `top_k`, `weights`, `min_entries`, `sort_groups`, `return_data`, `group_by_quantiles` and other parameters to never reach the underlying plot modules in vector mode.

### Objectives

- Localize and fix the kwarg-propagation bug at all 4 vector dispatch sites
- Eliminate per-iteration legend duplication, title accumulation, and redundant `tight_layout()` calls
- Validate fix end-to-end through real ADF→DFDraw pipeline (not just unit-level)
- Add permanent surface-enumeration tests to prevent recurrence
- Establish class-load validation pattern that fails loudly on future signature drift

### Two-Commit Pattern (codified for future fix phases)

**Commit 1 — Test baseline** (`444ad7f3`, 2026-04-15):
- 18 new diagnostic tests added with tags `[dfdraw B*a|b/method/kwarg]`
- 17 of 18 fail at this commit (the diagnostic instrument for localizing the bug)
- 3 new permanent capability matrix entries: `VECTOR.kwarg_propagation`, `VECTOR.groupby_polish`, `VECTOR.kwarg_surface`
- Pre-fix `reviewer_*.zip` is now a permanent diagnostic artifact

**Commit 2 — Fix** (`fe007b7c`, 2026-04-15):
- 17 → 0 failures, total 469/0/0
- 5 files modified: `drawer.py`, `plots/profile.py`, `plots/histogram.py`, `plots/scatter.py`, `tests/test_vector.py`
- Plus auto-regenerated `docs/CAPABILITY_MATRIX.md`

**Tooling commit** (`46af8af4`, between baseline and fix):
- `run_tests.sh`: include `test_full_*.log` in `reviewer.zip` (Claude45 hygiene fix)
- Paid off twice within the same phase (see §11.3 of code review summary)

### Implementation

**B1a — Named-parameter omission at vector dispatch (root bug):**

Python binds named parameters before `**kwargs`, so vector dispatch blocks only enumerated a hardcoded subset of caller arguments. Fix:

```python
# 4 class-level forwarded-name tuples on DFDraw
_PROFILE_FORWARDED_NAMES  = ('selection', 'sample', 'bins', ..., 'auto_title')  # 20 entries
_HIST_FORWARDED_NAMES     = (..., )                                              # 14 entries
_SCATTER_FORWARDED_NAMES  = (..., )                                              # 17 entries
_DRAW_FORWARDED_NAMES     = (..., )                                              # 12 entries

# _MISSING sentinel distinguishes "caller didn't pass" from "caller passed None"
_MISSING = object()

# Each dispatch block iterates the tuple and forwards via locals().get(name, _MISSING)
for name in self._PROFILE_FORWARDED_NAMES:
    val = _local.get(name, _MISSING)
    if val is not _MISSING and val is not None:
        if name == 'auto_title' and val is False:  # skip signature default
            continue
        vector_kwargs.setdefault(name, val)

# Module-import validation — fails loudly if signatures drift in future phases
_validate_forwarded_names()  # at end of drawer.py
```

**B1b — Matplotlib channel clobbering** at `_draw_vector` lines 606, 611:
```python
# Before: iter_kwargs['linestyle'] = ...   # unconditionally overwrote user choice
# After:  iter_kwargs.setdefault('linestyle', ...)   # user wins
```

**B2 — Main legend duplicated N times:**
- `_suppress_legend=True` injected per iteration
- Single post-loop `_add_vector_main_legend_dedup()` builds deduplicated legend

**B3+B4 — Title appended/replaced N times:**
- Iterations 0..N-2: `_suppress_title=True`
- Iteration N-1: `_suppress_title=False` (last iteration produces title naturally)
- Failsafe in `_draw_vector` for `group_by=None` + empty-title case

**B5 — `plt.tight_layout()` called N times:**
- `_suppress_layout=True` injected per iteration
- Single post-loop `plt.tight_layout()` call (wrapped in try/except — non-fatal warnings)

**R4 — `facet=True` + vector silently undefined:**
- All 4 dispatch sites raise `ValueError` with actionable message
- Implementation added 4th guard at `draw()` (defense-in-depth beyond v1.4 §3.1c spec)

### Two Implementation Deviations from v1.4 (both positive)

Recorded per §11.2 of code review summary as the **scope-positive divergence** pattern:

1. **`top_k` inclusion in 3 tuples** (caught by Claude43, Claude46): v1.4 §5.1 categorized `top_k` as facet-only and excluded it. Implementation discovered via source-read at `plots/profile.py:913`, `plots/histogram.py:595`, `plots/scatter.py:745` that `top_k` actually works in overlay mode. Added to `_PROFILE_FORWARDED_NAMES`, `_HIST_FORWARDED_NAMES`, `_SCATTER_FORWARDED_NAMES`. **None of 4 source-verifying reviewers caught the categorization error at proposal stage; implementation phase caught it.**

2. **R4 facet-guard at 4 sites instead of 3** (caught by Claude42, Claude43, Claude46): v1.4 §3.1c specified guard at 3 dispatch sites. Implementation added 4th guard at `draw()` for defense-in-depth. Catches `facet=True + vector` even when ADF or other callers route through `draw()` with `type=`.

Both deviations: discovered via source-read, correctness-improving, inline-documented with rationale, disclosed in Review Request §3 deviations table.

### Late Catch — `auto_title=False` Forwarding Bug (fresh-reviewer pattern)

During implementation, `test_auto_title_default_on_for_vector` (pre-existing Phase 13.16.DF test) regressed for **4 consecutive test runs**. Coder's debug-print-driven approach failed to converge (commitment bias accumulating across turns).

**Fresh reviewer Claude45 diagnosed correctly on first source read:** the signature default `auto_title=False` was being propagated through the forwarding loop, preventing `_draw_vector`'s vector-mode default-True injection because `'auto_title' in kwargs` evaluated True with value False, skipping the default-injection branch.

**Fix** — single-location, 2-line addition in both `profile()` and `hist()` forwarding loops:
```python
if name == 'auto_title' and val is False:
    continue
```

After this fix: 469/469 passed.

### Cross-Subproject End-to-End Verification (Phase 13.19.ADF.FIX1)

ADF team independently identified the same failure-mode class with 3 manifestations and executed parallel **Phase 13.19.ADF.FIX1**:

- **K1 boundary diagnostic** confirmed ADF forwards kwargs correctly — bug entirely on dfdraw side
- **K2 test suite (4 tests)** covers full `aDF.draw → DFDraw.*` pipeline
- **K2_3 = synthetic mirror of architect's ITS reproducer**: `[y1..y6]:staveITS, group_by='mP3', group_by_bins=6` — main legend bounded by `group_by_bins` (target ≤6, actual ≤8 with quantile-edge headroom), not `group_by_bins × n_vector`
- **Entry-point clarification** (closes R5 from v1.3 review): `aDF.draw(..., type='profile', ...)` dispatches via `getattr(plotter, 'profile')(expr, **kwargs)` at `AliasDataFrame.py:10180-10184`, bypassing `DFDraw.draw()` — confirms v1.4 §2.1 working hypothesis verbatim

**Three-level verification coverage:**
1. **dfdraw unit level** — 469/469 tests pass
2. **ADF integration level** — K2 suite 4/4 pass
3. **Production-pattern level** — K2_3 mirrors architect's real ITS reproducer

### Five-Iteration Source-Verification Chain (governance evidence)

This phase produced a complete catch chain across 5 abstraction levels:

| Iteration | Catch level | Reviewer | What was caught |
|-----------|-------------|----------|-----------------|
| v1.0 → v1.1 | Symptom-level | Architect | Screenshot misanalysis |
| v1.2 → v1.3 | Location-level | Claude43 | Fix location was `_draw_vector` (wrong); should be dispatch blocks |
| v1.3 → v1.4 | Documentation-level | Claude42/43/45/46 | §3.1 inventory wrong for 3 of 4 methods |
| Implementation | Runtime-level | Claude45 | `auto_title=False` forwarding bug (4 turns of debug failed; fresh source-read resolved in 1 turn) |
| Closure | Pipeline-level | Claude31 (ADF) | End-to-end verification through full `aDF → DFDraw` pipeline |

Each iteration caught a different class of error at a different abstraction level. **The most important governance evidence produced by the project to date.**

### Multi-Reviewer Verdict (7 reviewers, 5 source-verified)

| Reviewer | Verdict | Source-verified | Notes |
|----------|---------|-----------------|-------|
| Claude40 (Main) | ✅ APPROVED | ✅ Yes | All 7 §6.1 items verified |
| Claude42 | ✅ APPROVED | ✅ Yes | Identified top_k deviation |
| Claude43 (deepest) | ✅ APPROVED | ✅ AST set-comparison | Mathematical match (∅ symmetric difference) |
| Claude45 (NEW) | ⚠️ APPROVED W/ COMMENTS | ✅ Yes | Caught RR text typos + tag gap; previously caught auto_title=False |
| Claude46 (NEW) | ✅ APPROVED W/ P2 NOTES | ✅ AST validation | Caught both implementation deviations |
| GPT4 | ⚠️ APPROVED W/ P2 NOTES | ❌ Self-disclosed gap | Architectural framing |
| GPT5 | ⚠️ APPROVED W/ COMMENTS | ⚠️ Partial | Caught capability matrix phase header drift |
| Claude31 (ADF, end-to-end) | ✅ END-TO-END VERIFIED | ✅ Pipeline-level | K2 suite 4/4 |

**Consolidated:** 0 P0, 0 code-correctness P1, ~6 admin P1 (deduplicated to 4 housekeeping items), ~23 P2 (consolidated to 4).

### Capability Matrix Delta

| Metric | Before FIX1 | After FIX1 |
|--------|------------:|-----------:|
| Total features | 43 | **46** (+3) |
| Total proof tests | 152 | **170** (+18) |
| Invariance tests | 21 | **28** (+7) |
| ✅ Verified | 6 | **7** (+1: `VECTOR.kwarg_propagation`) |
| ☑️ Smoke-only | 37 | **39** (+2) |
| 🧨 Broken | 0 | **0** |

### Test Coverage Added

- `TestVectorKwargPropagation` — **7 invariance tests** (A≡B for `group_by_bins`, `group_by_quantiles`, `min_entries`, `sort_groups`, `weights`, `top_k`, `linestyle`)
- `TestVectorGroupBy` — **5 smoke tests** (legend dedup, title one-line, secondary legend count, tight_layout call count, architect's ITS production reproducer)
- `TestVectorKwargSurface` — **6 surface + guard tests** (4 per-method enumeration + R4 facet-guard + R17 forwarded-names-validity regression)

### Process Lessons Codified

- **Tooling investments compound:** `46af8af4` log-in-zip fix paid off twice in the same phase (Claude45's `auto_title=False` diagnosis + multi-reviewer test-name verification)
- **Two-commit pattern for fix phases:** Commit 1 (red baseline) + Commit 2 (green fix) preserves the diagnostic state in git history forever
- **Fresh-reviewer rule (proposed Coder QRC Rule 13):** "If 2 consecutive 'single-line fix' attempts fail to resolve a bug, request a fresh reviewer source-read before adding more debug"
- **Cross-subproject convergence triggered AND closed:** ADF team executed parallel verification within 2 working days of dfdraw FIX1 commit
- **Source verification at all 5 abstraction levels** (symptom → location → documentation → runtime → pipeline) demonstrated in a single phase

### Closure Status

**Required items closed:**
- ✅ Architect production reproducer verified (synthetic K2_3 mirror passed; real ITS data acceptance gate per v1.4 §8 — see architect confirmation)
- ✅ Tag `PHASE_13_16_DF_FIX1_END` confirmed at commit `fe007b7c`
- ✅ ADF entry-point hypothesis confirmed verbatim
- ✅ Cross-subproject end-to-end verification complete

**Recommended housekeeping (non-blocking):**
- Capability matrix header phase ID update (script enhancement)
- Workspace cleanup (`diagnose_auto_title.py`, `.ipynb_checkpoints/`, untracked PNGs)
- Tooling patch (add `git tag --list 'PHASE_*'` to reviewer.zip git_status section)

---

## Phase 13.18.DF: Robust Statistics Extension

**Date:** 2026-04 (commit window) → tag `PHASE_13_18_DF_v1_0_END`
**Status:** ✅ Complete
**Specification:** PHASE_13_18_DF_v1_2_Proposal_RobustStats.md

### Objectives
- Extend statistics computation with robust groups (median + MAD) alongside existing classical groups (mean + std).
- Allow callers to request specific statistics subsets via a new `stat_fields` parameter without recomputing the whole stats dict.
- Provide a uniform interface across `profile`, `hist`, `hist2d`, `scatter`, `hexbin`.

### Implementation
- New `stat_fields: Optional[Union[str, List[str]]]` parameter on all 5 plot methods + `DFDraw.draw()`.
- Valid groups: `'base'` (n, mean, std), `'robust'` (median, MAD), `'all'` (everything available), or list of group names.
- `_parse_stat_fields()` validator in `histogram.py` raises `ValueError` with the valid-group list on unrecognized input.
- Stats dict gains keys: `median`, `mad`, `q25`, `q75`, `iqr` when `'robust'` is active.
- 2 new STATS.* features registered in `feature_taxonomy.py`: `STATS.robust`, `STATS.range_aware` (with 2 + 3 tests respectively).
- Added `'stat_fields'` to `_PROFILE/HIST/SCATTER_FORWARDED_NAMES` tuples + signatures.

### Key Decisions
- **Robust stats are opt-in via `stat_fields`** — default remains classical (mean+std) for backward compatibility with all existing assertions.
- **Group-based API over flag-based API** — `stat_fields='robust'` is more extensible than a boolean `robust=True`. Future phases can add `'quantile'`, `'spread'`, etc. without signature changes.
- **MAD scaled by 1.4826** — Gaussian-equivalent sigma; matches the convention later used in Phase 13.28.DF hybrid autorange.

---

## Phase 13.25.DF v1.3: Quantiles on Profile (MultiGraph Framework — Phase A)

**Date:** 2026-04 → 2026-05 (full v1.0/v1.1/v1.2/v1.3 iteration cycle); v1.0_END tag `PHASE_13_25_DF_v1_0_END`; FIX1 tag `PHASE_13_25_DF_FIX1_END`; FIX2 commit `da8895e2`, tag `PHASE_13_25_DF_FIX2_END`
**Status:** ✅ Complete (Phase A of MultiGraph Framework)
**Specification:** PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md
**Review artifacts:** PHASE_13_25_DF_v1_3_PROPOSAL_REVIEW_SUMMARY.md, PHASE_13_25_DF_CODE_REVIEW_REQUEST.md

### Objectives
- Add per-bin quantile rendering on `profile()` to support distribution-shape comparison beyond mean ± std.
- Support two visual modes (error bars from symmetric pair, fill-band from symmetric triple) without requiring users to pick the rendering manually.
- Establish the naming and namespace conventions that downstream phases (Phase B / N-Channel Framework, Phase D / Selection-Weights-Facet) will inherit.

### Implementation

**New API surface:**
```python
# error_bars mode: symmetric pair without 0.5 → asymmetric error bars on central line
d.profile("y:x", quantiles=[0.16, 0.84])

# band mode: symmetric triple with 0.5 → fill_between
d.profile("y:x", quantiles=[0.16, 0.5, 0.84])

# central= overrides the line plotted under the band/bars
d.profile("y:x", quantiles=[0.16, 0.84], central='median')
```

- `quantiles: Optional[List[float]]` — list of fractions in (0, 1).
- `central: Optional[str]` — `'mean'` (default), `'median'`, `'both'`, `'none'`.
- `quantile_mode: str = "auto"` — `'auto'` dispatches via `_detect_quantile_mode()` based on list shape.
- New plots/profile.py helpers: `_compute_per_bin_quantiles()`, `_compute_per_bin_all_quantiles()`, `_detect_quantile_mode()`.
- Stats dict gains keys: `q_lower_per_bin`, `q_upper_per_bin`, `quantiles_per_bin` (per mode).
- New style keys: `quantile.band.alpha`, `quantile.band.hatch`, `quantile.error_bars.capsize`, `quantile.central_default` (AD-51 — keys land at interface introduction).

**Testing:**
- 28+ new tests in `tests/test_quantiles_profile.py` across multiple classes (TestQuantileBand, TestQuantileErrorBars, TestQuantileCentral, TestQuantileAutoDetection, TestQuantileStyleKeys, TestQuantileParity).
- Determinism + backward-compat regression-lock: byte-identical output across re-runs and against pre-Phase-13.25 behavior on quantile-free calls.

### FIX1 — Empty Quantile Dict Pruning (AD-52)

**Date:** 2026-04 → tag `PHASE_13_25_DF_FIX1_END`
**Caught by:** 6-reviewer panel during v1.0_END review (P1).
**Issue:** Per-bin quantile dicts with empty values were leaking as orphan legend entries.
**Fix:** Prune empty quantile dicts before legend rendering. Tests added under TestQuantileParity.

### FIX2 — Visual Elements (Linestyle Cycle + On-Line Annotations)

**Date:** 2026-04 → commit `da8895e2`, tag `PHASE_13_25_DF_FIX2_END`
**Scope:** Refined discrete-quantile rendering — added linestyle cycle for visual differentiation and on-line percentage annotations. Per v1.2 §11.3 directive these elements are preserved as the channel-aware default for `quantile_style='linestyle'` once Phase B lands.

### Key Decisions (AD-44 through AD-54 — see STYLING_FRAMEWORK_DECISIONS.md §2 for full text)

- **AD-44** — `quantiles=[…]` accepts list of fractions in (0, 1); matches numpy/pandas convention.
- **AD-45** — Default `central='mean'` when `quantiles=[…]` is set (backward compat with GB-mean).
- **AD-46** — `error_bars` mode = symmetric pair without 0.5 → asymmetric error bars on central line.
- **AD-47** — `band` mode = symmetric triple with 0.5 → fill_between.
- **AD-48** — `_detect_quantile_mode()` auto-routes by list shape (single API, pattern-based dispatch).
- **AD-49** — `central='none'` invalid with `error_bars` mode.
- **AD-50** — ADF-side parity deferred (independent track).
- **AD-51** — `quantile.*` style keys land in Phase A (new interface = cheapest place to lock keys). Established as **GP-1** in governance principles.
- **AD-52** — Empty quantile dict pruned before rendering (FIX1).
- **AD-53** — `quantile.*` keys are independent (no cascading from `profile.*` keys); matches existing dfdraw pattern.
- **AD-54** — Naming convention locked: `<channel>_style` kwargs, `channels.<key>` style namespace, `_assign_channels()` for the algorithm, `TestChannel*` for test classes. Vocabulary inherited by Phase B.

### Governance Principles Established

- **GP-1 — Style configurability lands at interface introduction.** New phases that introduce a mechanism must land its style keys in the same phase, not deferred. Precedent: `quantile.*` keys in this phase, `channels.*` keys in Phase B, `autorange.*` keys in Phase 13.28.
- **GP-3 — Architect signals preserved verbatim with typos.** Reformulating architect quotes into "polished" prose has caused production bugs (AD-50 cascade-vs-independence resolution required FIX1 to recover).

---

## Phase 13.26.DF v1.2: N-Channel Framework (MultiGraph Framework — Phase B)

**Date:** 2026-05-04 (Commit 1 scaffolding, parent `0df4c00b`) → 2026-05-06 (Commit 2 implementation, tag `PHASE_13_26_DF_v1_0_END`)
**Status:** ✅ Implementation complete; FIX1 pending (Claude40-approved, not yet started)
**Specification:** PHASE_13_26_DF_v1_2_Proposal_NChannelFramework.md
**Review artifacts:** Claude48_PHASE_13_26_DF_v1_2_Review_20260505.md, PHASE_13_26_DF_v1_0_END_CODE_REVIEW_SUMMARY.md

### Objectives
- Generalize the visual-channel system to N independent data channels (vector × group_by × quantiles) with automatic conflict resolution.
- Eliminate the channel-collision bug class where two data channels would silently land on the same visual encoding (e.g., both vector and group_by getting `color`).
- Lock channel cycles + capacity + overflow behaviour via style keys so users can tune without code changes.

### Implementation

**Algorithm A** (`dfdraw/channels.py`, ~310 LOC):
- `assign_channels(active_channels, priority_categorical, priority_ordinal, explicit_rules)` resolves which visual encoding (color/linestyle/marker) each active data channel gets.
- `EXPLICIT_RULES: dict[frozenset[str], dict[str, str]]` — explicit overrides per channel-set, append-only across phases (per GP-2).
- Capacity check per channel via `channels.cycles.{color_count, linestyle, marker}` style keys; overflow mode `'error'` (default, actionable message) or `'warn'`.
- `quantile_style` kwarg forwarded from `DFDraw.profile()` into channel decision.

**Nested-band auto-detection (AD-57):** Symmetric quantile lists with ≥4 non-0.5 entries auto-route to `'nested_band'` mode (alpha-stacked filled regions). Central line handled independently via `central=` parameter.

**Factored legend (AD-59):** Default `True` — legend rendered with section headers per data channel; entry count = sum of cardinalities (not product). 3-channel call with |group|=5, |vector|=3, |quantiles|=5 → 11 entries factored vs 75 unfactored. Override via `channels.legend.factored=False`.

**FIX2 visual elements preservation (AD-60):** FIX2's hardcoded linestyle cycle (`profile.py:559`) replaced with `get_style_value("channels.cycles.linestyle", default)[1:]`. Solid linestyle remains reserved for central line. On-line percentage annotations preserved for `quantile_style='linestyle'`, suppressed for `'marker'`/`'color'`.

**New style keys (10):** `channels.priority.categorical`, `channels.priority.ordinal`, `channels.cycles.linestyle`, `channels.cycles.marker`, `channels.cycles.color_count`, `channels.default.vector`, `channels.default.group_by`, `channels.default.quantiles`, `channels.overflow`, `channels.legend.factored`.

**Testing:**
- 50 new tests in `tests/test_channel_assignment.py` across multiple classes (TestChannelAssignment{1,2,3}Active, TestChannelCapacity, TestChannelCollision, TestChannelUserOverride, TestChannelStyleOverride, TestFactoredLegend, TestNestedBand, TestIdempotency).
- Final count: **627 passed + 1 skipped + 0 failed** (+50 vs Commit 1 baseline of 577); zero regressions.

### Key Decisions (AD-55 through AD-60)

- **AD-55** — Algorithm A categorical priority `["color", "linestyle", "marker"]` (default, changeable via style key).
- **AD-56** — 3-channel default: `group_by → color, vector → marker, quantiles → linestyle`. Implemented via `EXPLICIT_RULES`. Reconciled from Claude48 P1-1 (Option C).
- **AD-57** — Nested-band auto-detection: non_05_count ≥ 4 AND symmetric pairs → `'nested_band'` (with or without central).
- **AD-58** — Overflow default `"error"` with actionable suggestions (`top_k=`, `facet=True`, `group_by_bins=`). Silent auto-facet rejected as hiding intent.
- **AD-59** — Factored legend default `True`.
- **AD-60** — FIX2 visual-elements channel-aware preservation (linestyle cycle + on-line annotations).

### Governance Principles Established / Reinforced

- **GP-2 — Internal APIs accepting new data-channel types must be list-based from day one.** v1.1 used three booleans; v1.2 G-7 generalized to `list[DataChannel]` with `EXPLICIT_RULES`. Same LOC; eliminates Phase D internal-API redo. Corollary: `EXPLICIT_RULES` is append-only across phases.
- **GP-4 — Backward-compat scope must be justified by production-usage verification.** v1.0 → v1.1 took three iterations because no reviewer verified the quantile path had production users. v1.2 G-3 reframed Class 10 around real `makeSmoothMapsWithTPC.py` patterns.
- **GP-5 — Drafter rotation across phases is healthy.** Phase 13.26 v1.0/v1.1 drafted by Claude48; v1.2 drafted by Claude49Coder. The successor drafter caught the G-7 forward-extensibility upgrade that the original drafter did not surface.

### Open Follow-Ups
- **Phase 13.26.DF FIX1** approved by Claude40, not yet started: `group_style='color'` default blocks style parameterization (same bug class as Phase 13.28 P1); 10 docstrings to update; CAPABILITY_MATRIX amend.

---

## Phase 13.27.DF Commit 1: Facet Refactor (MultiGraph Framework — Phase D, profile-only)

**Date:** 2026-05-09 (proposal v1.1 approved + implementation + tag `PHASE_13_27_DF_Commit1_END`)
**Status:** ✅ Commit 1 complete (profile-only facet refactor); Commit 2 pending (selection_vector + weights_vector + hist/scatter integration + 50 tests)
**Specification:** PHASE_13_27_DF_v1_1_Proposal_SelectionWeightDeltaFacet.md
**Review artifacts:** PHASE_13_27_DF_v1_0_PROPOSAL_REVIEW_SUMMARY.md; v1.1 architect direct-approval (path B per consolidated review)

### Objectives
- Replace inline `facet=True` path in `DFDraw.profile()` with a unified `_dispatch_faceted_render()` method routing through the N-Channel framework (AD-67).
- Add `facet_by=` as the new public API; preserve `facet=True` as backward-compat alias.
- Treat facet as the 4th visual encoding (spatial) — gated by `channels.cycles.facet_max` capacity.
- Lock the architect rule that `same=True` and `facet_by=` are mutually exclusive (facet creates new figure; same overlays on existing axes).

### Implementation

**New method `DFDraw._dispatch_faceted_render()`** coordinates faceted rendering:
1. Validate `facet_by` against `_VALID_FACET_BY_VALUES_COMMIT1 = ('group_by', 'vector', 'quantiles')`. `'selection_delta'` / `'weights_delta'` reserved for Commit 2 with `NotImplementedError`.
2. Determine groups (unique values of the faceted channel).
3. Capacity check via `channels.cycles.facet_max` (default 16); overflow `'warn'` truncates+warns, `'error'` raises.
4. Build subplot grid via `plt.subplots(nrows, ncols, sharex, sharey, squeeze=False)`.
5. Per-subplot recursion: filter DataFrame per group value, call `draw_profile` with pre-created `ax=`, suppress per-subplot auto_title (suptitle owned by coordinator).
6. Combined stats dict: `{'n_groups', 'groups', 'per_group', 'n_total', 'faceted': True, 'facet_by'}`.

**Backward compatibility (AD-67):** `facet=True` normalized to `facet_by='group_by'`. Both API forms route through the same code path → byte-identical figure output. Locked by Class 1 invariance test `test_facet_true_eqivalent_to_facet_by_groupby`.

**Vector entry-point intercept:** When `facet_by='vector'` is set on a list-valued `y_expr`, profile() routes BEFORE `_draw_vector` to `_dispatch_faceted_render`. Vector overlay and vector facet are mutually exclusive at dispatch level (no nested re-entry).

**Mutual exclusion (AD-68):** `facet_by` + `same=True` raises `ValueError` with the phrasing "mutually exclusive".

**New style keys (6):** `channels.cycles.facet_max=16`, `channels.legend.facet_position='upper right'`, `channels.label.selection_truncate=25`, `channels.label.weights_truncate=25`, `channels.default.selection_delta=None`, `channels.default.weights_delta=None`. Commit 1 uses the first two; the remaining four are reserved for Commit 2 but landed in this commit per **GP-1** ("style keys land at interface introduction").

**Forwarder discipline:** `'facet_by'` added to `_PROFILE_FORWARDED_NAMES`; tuple-validation R6 check passes at module import. This is the Phase 13.28 FIX1 lesson applied prospectively (no silent kwarg drops).

**Stale-test housekeeping:** `tests/test_channel_assignment.py::TestChannelStyleOverride::test_default_style_has_all_10_keys` renamed to `test_default_style_has_all_channels_keys` and updated to expect 16 keys (10 Phase 13.26 + 6 Phase 13.27). Removes hardcoded count so future phases adding `channels.*` keys won't need re-renaming.

### Testing
- 10 invariance tests in `tests/test_phase_13_27_facet_refactor.py`, each load-bearing assertion marked `# §9.<class>.<id>` per Coder QRC v1.30 Rule 14:
  - **TestFacetLegacyEquivalence (3)** — backward compatibility (facet=True ≡ facet_by='group_by')
  - **TestFacetByChannel (4)** — 4-channel routing (vector / group_by / quantiles / invalid)
  - **TestFacetCapacity (2)** — capacity enforcement (error mode + warn mode)
  - **TestFacetSameTrueExclusion (1)** — facet_by × same=True mutual exclusion
- Final count: **663 passed + 1 skipped + 0 failed** (+10 vs Phase 13.28 close baseline of 653).

### Key Decisions (AD-61 through AD-68; AD-61..AD-66 land in Commit 1; AD-67, AD-68 cover dispatch rules)

- **AD-61** — `channels.cycles.facet_max=16` bounds subplot count.
- **AD-62** — `channels.legend.facet_position='upper right'` for shared figure-level legend in faceted rendering.
- **AD-63, AD-64** — Selection/weights label truncation defaults (25 chars). Used by Commit 2.
- **AD-65, AD-66** — Pinned defaults for `selection_delta` / `weights_delta` channels. Used by Commit 2.
- **AD-67** — `facet=True` normalizes to `facet_by='group_by'`; same-path dispatch guarantees byte-identical output.
- **AD-68** — `facet_by` + `same=True` are mutually exclusive (architect rule, v1.1 §5.4).

### Out of Scope (Commit 2)
- `selection_vector` + `weights_vector` public parameters
- `delta_facet` plot type (`facet_by='selection_delta'`, `facet_by='weights_delta'`)
- hist + scatter facet integration (currently profile-only)
- 50 invariance tests for selection/weights

---

## Phase 13.28.DF v1.1: Robust Data Handling

**Date:** 2026-05-06 (v1.0 proposal) → 2026-05-09 (closure tag `PHASE_13_28_DF_v1_0_END` at commit `8b02d241`; also tagged `PHASE_13_28_DF_Integration_END` at the same commit)
**Status:** ✅ Complete; FIX1 pending (GPT4 #2 finding)
**Specification:** PHASE_13_28_DF_v1_1_Proposal_RobustDataHandling.md
**Review artifacts:** PHASE_13_28_DF_v1_1_Code_Review_Request.md; consolidated review by Claude40 (5-0 approval: Claude40, Claude48, GPT4 ×2, Claude32 ADF cross-group)

### Objectives

Close a class of robustness bugs surfaced by the architect on real TPC data:
- `adf.draw("y/x:row", type="hist2d")` previously crashed matplotlib with `autodetected range of [-inf, inf] is not finite` when `y/x` produced `inf` (rows with `x=0`).
- NaN in expression columns silently returned `n=0` with no diagnostic.

Introduce centralized sanitization + outlier-aware autorange across all 5 plot types (`hist`, `hist2d`, `hexbin`, `profile`, `scatter`).

### Implementation

Phase shipped as 3 commits:

**Part A — NaN/inf filter (commit `b1d153b6`, tag `PHASE_13_28_DF_PartA_END`)**
- New module `plots/_data_sanitize.py` exposing `sanitize_for_plot(x_data, y_data, nan_policy, column_names)` (~140 LOC).
- `nan_policy` ∈ `{'filter', 'warn', 'raise'}`; default `'filter'` preserves pre-Phase-13.28 bit-identical behavior on clean data.
- Returns `sanitize_stats` dict with always-populated counters: `n_input, n_filtered, n_inf_x, n_nan_x, n_inf_y, n_nan_y`.
- 12 invariance tests pass.

**Part B — Hybrid autorange (commit `f25e8928`, tag `PHASE_13_28_DF_PartB_END`)**
- New module `plots/_autorange.py` exposing `compute_autorange(data, strategy, ...)` and `hybrid_autorange()` (~280 LOC).
- 6 strategies: `'hybrid'` (default, outlier-aware), `'minmax'` (backward compat), `'percentile_99'`, `'percentile_95'`, `'robust_3mad'`, `'robust_4mad'`.
- `hybrid_autorange()` formal definition (AD-72): compute robust window `(median ± k_robust·sigma_MAD)`. Per-side, declare outlier on side S if data extreme exceeds median by more than `(k_outlier · k_robust · sigma_MAD)`. Use robust bound when outlier present, else use data extreme. Per-axis independent for 2D (AD-74).
- 21 unit tests pass (12 Part A + 9 Part B).

**Integration (commit `8b02d241`, tag `PHASE_13_28_DF_Integration_END` then `PHASE_13_28_DF_v1_0_END`)**
- `nan_policy=` parameter exposed on all 5 `draw_*` plot functions AND on `DFDraw.draw / .hist / .hist2d / .profile / .scatter`.
- `~np.isnan` masks replaced with `sanitize_for_plot()` — also catches `inf`.
- `range=` accepts strategy strings (`'hybrid'`, `'minmax'`, etc.) in addition to numeric tuples; resolved via `resolve_range_1d/_2d` helpers.
- Stats dict gains 8 new keys (6 sanitize counters + `autorange_used` + `autorange_strategy` per AD-77).
- **FIX1 mid-integration:** First integration attempt had `nan_policy` silently dropped between user-facing API and `draw_*` because FORWARDED_NAMES tuples + method signatures + scalar-path call sites all needed updates. R6 validator caught the asymmetry; same fix pattern applied across 4 method signatures, 4 forwarder tuples, 4 explicit call sites.
- 5 new integration tests (TestPlotIntegration) — including the architect's exact bug reproducer locked as an invariance test.
- Final count: **653 passed + 1 skipped + 0 failed**.

### Bug Fix — Architect's Reproducer

```python
df = pd.DataFrame({
    "x": [1, 2, 0, 4, 5, 0, 7, 8],
    "y": [1, 2, 3, 4, 5, 6, 7, 8],
    "detType": [0, 0, 0, 0, 1, 1, 1, 1],
})
fig, ax, stats = DFDraw(df).hist2d("y/x:x", selection="detType==0")
```

| | Pre-Phase-13.28 | Post-Phase-13.28 |
|---|---|---|
| Crash | matplotlib `autodetected range of [-inf, inf] is not finite` | No crash |
| `stats["n"]` | `0` silently | `3` |
| `stats["n_inf_y"]` | not present | `1` |
| `stats["autorange_used"]` | not present | `((1.0, 4.0), (0.5, 1.5))` |
| `stats["autorange_strategy"]` | not present | `'hybrid'` |

Bound by `TestPlotIntegration::test_hist2d_with_inf_does_not_crash_and_reports_counters`.

### Key Decisions (AD-69 through AD-77)

- **AD-69** — Centralized sanitization module (one entry point for all plot types).
- **AD-70** — `nan_policy='filter'` default (silently drops, populates counters); `'warn'` / `'raise'` alternatives.
- **AD-71** — Counter keys always populated regardless of nan_policy.
- **AD-72** — Hybrid autorange formal definition (per-side outlier decision).
- **AD-73** — Default autorange strategy `'hybrid'`.
- **AD-74** — 2D autorange per-axis independent.
- **AD-75** — Backward-compat lock: existing tests depending on min/max get explicit `range='minmax'`.
- **AD-76** — Strategy parameters (`k_robust`, `k_outlier`, `percentile`) tunable via style keys only in v1.0; per-call kwarg override deferred.
- **AD-77** — Diagnostic stats keys (`autorange_used`, `autorange_strategy`) always populated. **`stats['n']` semantics locked: post-sanitize finite count, NOT range-clipped. Range filtering is VISUAL ONLY.** GPT4 #1 + #2 convergent finding.

### Code Review Verdict (closure)

5-0 approval per `PHASE_13_28_DF_CONSOLIDATED_CODE_REVIEW_SUMMARY` (Claude40 main):

| Reviewer | Group | Verdict | P1 |
|---|---|---|:--:|
| Claude40 | dfdraw | ✅ | 0 |
| Claude48 | dfdraw | ✅ | 0 |
| GPT4 #1 | dfdraw | ⚠ | 3 |
| GPT4 #2 | dfdraw | ⚠ | 1 |
| Claude32 | ADF | ✅ | 0 |

### Open Follow-Ups (FIX1)
- **GPT4 #2 P1 — Style-key default blocks style parameterization.** `nan_policy="filter"` signature default means `set_style({"data.nan_policy": "raise"})` has no effect. Same structural bug class as Phase 13.26 P1-1 (`group_style='color'`). Mechanical fix: change defaults to `None`, resolve from style at runtime. Recommended to bundle with the pending Phase 13.26 FIX1 since it's the same bug class.

---

## Phase 13.28.DF FIX1: Restore autorange.* style keys in DEFAULT_STYLE

**Date:** 2026-05-15 (commit `57576ebf`, tag `PHASE_13_28_DF_FIX1_END`)
**Status:** ✅ Complete
**Discovery:** Sonet50 documentation audit of Phase 13.32 v1.0 proposal
**Review artifacts:** 3-of-3 panel approval (Claude48, Sonet50, Claude40)

### Objectives

Close a silent regression caught by docs audit: Phase 13.28 Part B introduced four `autorange.*` style keys (`strategy`, `k_robust`, `k_outlier`, `percentile`) referenced via `get_style_value(...)` from `plots/_autorange.py + profile.py + histogram.py`, but the keys were **never registered** in `DEFAULT_STYLE`. Calling `set_style({"autorange.k_robust": 8.0})` raised `ValueError("unknown style key")`. Discovered through Phase 13.32 audit, not by any runtime test — production style-customization path was effectively broken since Phase 13.28 introduction.

### Implementation

- 4-key registration in `style.py` after the existing `profile.*` section, with defaults matching the `get_style_value(...)` fallback arguments:
  - `autorange.strategy = "hybrid"`
  - `autorange.k_robust = 4.0`
  - `autorange.k_outlier = 1.5`
  - `autorange.percentile = (1.0, 99.0)`
- 9 §9-marked tests in `tests/test_phase_13_28_df_fix1_autorange_style_keys.py` lock: round-trip `set_style → get_style_value`, defaults intact, integration into profile/histogram autorange paths.
- Test count: **687 → 696** passed, 0 failed.

### Key Decisions

- **Sonet50 P1 (`data.nan_policy` not restored) REJECTED as misclassification.** `nan_policy` is a per-call kwarg with signature default `"filter"`, not a style key (no `get_style_value("data.nan_policy")` exists anywhere). Decision deferred to Phase 13.30 sub-fix 2 if registration is desired.

### Lessons captured

- **Style-key registration regression class** — same bug shape as `nan_policy` (signature default blocks style override). A `get_style_value()` ↔ `DEFAULT_STYLE` cross-check validator (analogous to R6 for `_*_FORWARDED_NAMES`) would prevent this entire class of bug. Folded into Phase 13.30 sub-fix 2 scope.

---

## Phase 13.30.DF v1.0: Column-Reference Parameter Validation (Class-2)

**Date:** 2026-05-12 (commit `e8278531`, tag `PHASE_13_30_DF_ColumnRefValidation_v1_0_END`)
**Status:** ✅ Complete (sub-fix 1 of "Parameter Class Validation" suite); sub-fix 2 deferred to governance closure
**Specification:** PHASE_13_30_DF_v1_0_Proposal_ColumnRefValidation.md
**Review artifacts:** Panel-approved (Claude40 Main, Claude48, Sonet51, GPT1)

### Objectives

Establish formal parameter taxonomy and runtime validation for **Class-2 parameters** — those that must resolve to a literal DataFrame column (the v1.0 tuple is `_PROFILE_COLUMN_REFERENCES = ('group_by',)` — additional Class-2 names like `weights`, `color`, `size` are candidates for future expansion but were *not* in scope for the v1.0 phase). Distinguish from Class-1 (expressions evaluable via `df.eval`) where any token may be a temporary alias. Phase 13.30 ships Class-2 validation only; Class-3 (callable-typed parameters) and Class-5 (style-key cascades) validation are deferred to a future phase.

### Implementation

- New class-level tuple `_PROFILE_COLUMN_REFERENCES = ('group_by',)` enumerating Class-2 params for `profile()` (v1.0 ships a single entry; expansion to `weights`, `color`, `size` deferred pending decisions on their typing semantics).
- New module-import validator `_validate_column_reference_tuples()` analogous to Phase 13.16 FIX1's R6 — catches signature drift at import time, not silently at runtime.
- Runtime resolution: when a Class-2 kwarg is non-None, validate it appears as a literal column in `df.columns` before plot logic runs; raise `ValueError` with the column list on miss. Eliminates the silent-pass-through regression where `group_by="typo"` would slip through to `df.groupby("typo")` and surface as an obscure KeyError downstream.
- 12 §9-marked invariance tests in `tests/test_phase_13_30_df_column_ref_validation.py`.
- Test count: **663 → 675** passed.

### Key Decisions

- **Class-2 only in v1.0.** Class-1 expression validation requires AST-level expression introspection; deferred to a future phase.
- **No retroactive change to caller signatures.** Validation lives entirely in dispatch; existing tests unchanged except for the +12 new invariance tests.
- **AD-78 forward compatibility (Sonnet P1):** NO change to `_PROFILE_COLUMN_REFERENCES` for `facet_by` — Phase 13.31 introduces dual-path (channel-name vs column-name) `facet_by`, and adding it here would break every existing channel-mode usage. Strictly column-resolution params only.

### Code Review Verdict (closure)

| Reviewer | Group | Verdict | P1 |
|---|---|---|:--:|
| Claude40 | dfdraw | ✅ | 0 |
| Claude48 | dfdraw | ✅ | 1 (governance: `feature_taxonomy.py` entry deferred) |
| Sonet51 | dfdraw | ✅ | 0 |
| GPT1 | dfdraw | ✅ | 0 |

### Open Follow-Ups (governance closure)

- `tests/feature_taxonomy.py` entry `DATA.column_reference_validation` + classify 12 P13.30 tests under it.
- `DFDraw.draw()` dispatch test: missing `group_by` raises through dispatch path too (Claude40 P1).
- Sub-fix 2: register `data.nan_policy` in `DEFAULT_STYLE` (or formally document why not).

---

## Phase 13.31.DF v1.0: `facet_by` Column-Name Support (AD-78)

**Date:** 2026-05-14 (commit `f3ca432a`, tag `PHASE_13_31_DF_FacetByColumn_v1_0_END`)
**Status:** ✅ Complete
**Specification:** PHASE_13_31_DF_v1_0_Proposal_FacetByColumn.md
**Review artifacts:** PHASE_13_31_DF_v1_0_END_CODE_REVIEW_REQUEST.md — panel approval (Claude40 Main, Claude48, Sonet51); BUG_ADF_GroupBy_Expression_Materialization handed off to ADF team

### Objectives

Extend Phase 13.27 Commit 1 `facet_by` from channel-name only (`'group_by'`, `'vector'`, `'quantiles'`) to also accept any literal DataFrame column name. Closes a user-experience gap: `d.profile("y:x", facet_by="quartile_val")` should work without first writing `d.profile("y:x", group_by="quartile_val", facet_by="group_by")`. The two paths are kept structurally separate in dispatch ("dual-path") to preserve channel-mode semantics where `group_by` is *also* set as an overlay dimension.

### Implementation

- `_dispatch_faceted_render` extended with `_facet_mode` selector: `'channel'` (existing) vs `'column'` (new).
- Column-mode branch: filter df by literal-column equality (`df[facet_by] == group_value`) per subplot; preserves an outer `group_by` for inner overlay (orthogonal-composition test S3.4).
- Boolean and numeric column dtypes uniformly handled via mask comparison (avoids string-quoting issues that would arise from extending the selection-string path).
- 12 §9-marked invariance tests in `tests/test_phase_13_31_facet_by_column.py`, covering channel-mode regression, column-mode subplot cardinality, mutual-exclusion guards, and orthogonal `group_by` composition.

### Key Decisions (AD-78)

- **AD-78** — Dual-path `facet_by`: channel-name routing preserved; column-name routing added as parallel branch. Both paths converge in subplot grid layout, legend, and stats aggregation. Architect-signed 2026-05-14.

### Code Review Verdict (closure)

Panel-approved. Sonet51 P1 (governance: `feature_taxonomy.py` entry `FACET.column_mode` deferred) and the AD-78 §0 reviewer-ID typo (`Sonnet` → `Sonet50`) folded into the deferred Phase 13.30+31+32 closure pass.

### Open Follow-Ups (governance closure)

- `tests/feature_taxonomy.py` entry `FACET.column_mode` + test classification.
- AD-78 §0 reviewer-ID typo.

---

## Phase 13.32.DF v1.0: `group_by × quantiles` in Grouped Path + Symmetric `facet_by` Binning (AD-79)

**Date:** 2026-05-15 (commit `cb6a1aed`, tag `PHASE_13_32_DF_GroupByQuantilesFacet_v1_0_END`)
**Status:** ✅ Complete; advisory items pending in governance closure pass
**Specification:** PHASE_13_32_DF_v1_2_Proposal_GroupByQuantilesFacetBinning.md (v1.0 → v1.1 → v1.2 cycle, panel-approved 2026-05-15)
**Review artifacts:** PHASE_13_32_DF_GroupByQuantilesFacet_v1_0_END_Code_Review_Request.md; consolidated review by Sonet50 — APPROVED (4 of 5 reviewers on correct source, 0 blocking issues)

### Objectives

Close three composition gaps in the grouped + faceted dispatch surface:
- **Sub-fix 1** — `facet_by='group_by'` + `group_by_bins`/`group_by_quantiles` produced N×N spurious sub-groups in per-subplot recursion (architect's production `In[129]` reproducer: 60 raw `driftM_bin25` levels → cap-overflow before binning → 5 expected subplots, each previously containing 5 ghost lines).
- **Sub-fix 2** — `group_by` + `quantiles=` (with `quantile_mode ∈ {band, error_bars, discrete}`) silently dropped the quantile rendering in the grouped path (architect's production `In[126]` reproducer).
- **Sub-fix 3 (AD-79)** — Symmetric `facet_by_bins` / `facet_by_quantiles` counterparts to `group_by_bins` / `group_by_quantiles` on the column-mode `facet_by` path (Phase 13.31 AD-78), available across all four plot kinds (`profile`, `hist`, `scatter`, `hist2d`).

### Implementation

- **Sub-fix 1** — `_dispatch_faceted_render` hoists binning *before* group enumeration when `facet_by='group_by'`; pops `group_by_bins`/`group_by_quantiles` from `plot_kwargs` so per-subplot recursion does not re-bin. Convergent P1 finding from Sonet51, Claude48, GPT1 in the v1.0 panel review.
- **Sub-fix 2** — `_draw_profile_grouped` extended with `quantiles`, `quantile_mode`, `central`, `quantile_pair`, `quantile_list`, `quantile_style` parameters; per-group `band` / `error_bars` / `discrete` rendering inlined (~25 LOC, option B from v1.2 §3.2). `nested_band` explicitly raises `NotImplementedError` in grouped path (FIX 2 P1, Claude48 / Claude40 / GPT1). New style key `quantile.band.alpha_grouped = 0.15` (renamed from v1.0's `channels.quantile.band_alpha_per_group` per FIX 3 namespace agreement, Sonet50 / Sonet51).
- **Sub-fix 3 (AD-79)** — Symmetric `_validate_facet_by_binning()` static helper; column-mode dispatch branch with hoisted binning analogous to Sub-fix 1. Per-subplot loop rewritten with **plot-kind-specific dispatch** (each `draw_*` has distinct signature and incompatible kwargs — `hist` takes only `x`; `hist2d` does not accept `group_by`/`top_k`; `scatter` does not accept `auto_title`). New `_HIST2D_FORWARDED_NAMES` tuple created; all five FORWARDED_NAMES tuples extended with `facet_by_bins` / `facet_by_quantiles`; R6 validator AST-equivalent simulated at drafter time before delivery (lesson from session debug cycle).
- 19 §9-marked invariance tests in `tests/test_phase_13_32_groupby_quantiles_facet.py` across 5 classes (`TestSubfix1`, `TestSubfix2`, `TestSubfix3Profile`, `TestSubfix3AllPlots`, `TestRepro`). One-line bump in `tests/test_quantiles_profile.py::TestQuantileStyleKeyDefaults::test_namespace_integrity` (count 4→5 for new style key).
- Test count: **696 → 715** passed + 1 skipped.

### Bug Fix — Architect's Reproducers

| Reproducer | Pre-Phase-13.32 | Post-Phase-13.32 |
|---|---|---|
| `In[126]` overlay-with-quantiles (`group_by=` + `quantiles=` + `quantile_mode='band'`) | Quantile band silently dropped; only error bars rendered | Per-group color-coordinated band + central line as specified |
| `In[129]` facet-with-binning (`facet_by='group_by'` + `group_by_bins=5`) | Subplot cap overflow before binning (60 raw groups → matplotlib error or 5 subplots × 5 ghost lines) | 5 binned subplots, exactly 1 profile per subplot |

Bound by `TestRepro::test_in_126_overlay_with_quantiles` and `TestRepro::test_in_129_facet_with_groupby_bins`.

### Key Decisions (AD-79)

- **AD-79** — Symmetric `facet_by_bins` / `facet_by_quantiles` across all four plot kinds (option B from v1.2 §3.3). Architect-signed 2026-05-14.

### Code Review Verdict (closure)

Consolidated by Sonet50 — **APPROVED** (4 of 5 reviewers on correct source, 0 blocking):

| Reviewer | Group | Verdict | Notes |
|---|---|---|---|
| Claude40 | dfdraw | ✅ [OK] | Full functionality verified, production reproducers locked |
| Sonet50 | dfdraw | ✅ [OK] | Diff-level verification |
| Sonet51 | dfdraw | ✅ [OK] | 7 test bodies sampled, all v1.0 P1s traced |
| Sonnet52_R1 | dfdraw | [!] | 2 advisory P2 items (`quantile_style` silent drop, `error_bars→band` lock test) |
| Sonnet53_R2 | dfdraw | [X] | Excluded — reviewed wrong source bundle (`sourcesdf.zip` instead of official `reviewer_20260515_165043.zip`); P0/P1 findings were artifacts of stale source |

### Open Follow-Ups (governance closure pass)

Folded into the deferred Phase 13.30 + 13.31 + 13.32 unified closure pass:
- **Adv-1** — `NotImplementedError` for `quantile_style` in grouped path (mirroring `nested_band` treatment) — Sonnet52_R1 / Sonet50.
- **Adv-2** — §9.S2.7 test locking `error_bars → band` intentional degradation in grouped path — Sonnet52_R1 provided test code.
- `tests/feature_taxonomy.py` entries: `FACET.column_mode_binning` (for Sub-fix 3 coverage).
- Sonet50 / Sonnet52_R1: nested-band design pass for grouped path → Phase 13.34 (provisional).
- Weighted-quantile composition (currently raises `NotImplementedError`) → Phase 13.25 Phase B.

### Session governance amendments (proposed for Coder QRC v1.32)

Two binding rules emerged from the debug arc, captured in the Code Review Request §8:
1. **AST-level R6-equivalent pre-delivery check** when editing any `_*_FORWARDED_NAMES` tuple. Caught Bug 0 in this session (the `DFDraw.draw()` signature/tuple mismatch) only after a failed test run; should have been caught at drafter time.
2. **No local `sed` patches.** All file edits flow through `present_files` artifacts so the patch is auditable in the chat transcript.

---

## Phase 13.27.DF Commit 2 v1.0: `selection_vector` + `weights_vector` + `delta_facet` (Phase D completion)

**Date:** 2026-05-16 (commit `84dcf916`; first commit `bd35ea5d` 2026-05-16 12:20 then re-committed 12:45 with Sonnet52_R1 P1-2 guard appended; both messages identical title)
**Status:** ✅ v1.0 closed; FIX1 / FIX1.FIX1 follow-ups documented below
**Specification:** PHASE_13_27_DF_Commit2_v1_0_Proposal.md (panel-conditional APPROVAL converted to APPROVAL by panel-requested Hard Constraint §3 guard)

### Objectives

Complete Phase D (MultiGraph) Commit 2: per-curve `selection_vector` and `weights_vector` channels with `delta_facet` label management. Closes the cross-curve composition story started by Phase 13.27.DF Commit 1 (facet refactor) and Phase 13.26.DF v1.2 (N-Channel Framework / Algorithm A).

### Implementation

- 11 new `EXPLICIT_RULES` entries (`selection_delta` / `weights_delta` combinations)
- 1 new style key: `channels.label.delta_separator`
- 8 new kwargs on `profile()` / `hist()` / `scatter()` / `draw()` uniformly (per A-1 surface convention)
- `hist2d()` signature gate: explicit `TypeError` for vector kwargs (per-pixel density has no meaningful overlay)
- `scatter()` `weights_vector` UserWarning at entry: per-row weights have no rendering effect on scatter (point-size weighting deferred to Phase E)
- 3 new static helpers: `_compute_vector_iteration_indices`, `_combine_selections`, `_combine_weights`
- `_draw_vector` iteration-loop refactor with per-curve selection/weights composition
- **Hard Constraint §3 guard** (Sonnet52_R1 P1-2): `UserWarning` when single-Y + multi-element `selection_vector` / `weights_vector` (silent-degrade prevention). FIX1 enables full single-Y dispatch.
- 52 §9 invariance tests across 12 classes (load-bearing assertions per Coder QRC Rule 14)
- `feature_taxonomy.py`: +5 CHANNEL entries (62 → 67 features)

**ADs realized:** AD-61, AD-62, AD-65, AD-66, AD-67 (AD-63, AD-64, AD-68 realized in earlier phases). Phase D complete except FIX1 (single-Y full dispatch + hist weights rendering).

### Tests

**Gate:** 715 → **767** / 0 / 1 skipped (+52 §9 invariance).

---

## Phase 13.27.DF Commit 2 FIX1: Single-Y vector dispatch + hist weights rendering

**Date:** 2026-05-16 (commit `ba42fcde`)
**Status:** ✅ Closed
**Specification:** PHASE_13_27_DF_Commit2_FIX1.md

### Objectives

Replace the v1.0 Hard Constraint §3 UserWarning guard with full functional dispatch, and add column-name/expression `weights=` to `hist()`.

### Implementation

- **§7(a) Single-Y vector dispatch:** Single-Y / single-X + `selection_vector` / `weights_vector` now engages `_draw_vector` dispatch (was silent-ignore + UserWarning guard). The 3 FIX1-pending UserWarning blocks in `hist()` / `scatter()` / `profile()` have been removed. Trigger gated on `not _column_mode_facet` to preserve facet path for single-Y + column-mode `facet_by` composition (a Phase 13.33 concern per spec §4.2.4).
- **§7(b) hist weights:** `draw_hist()` accepts `weights=` as column name or `df.eval()` expression. Joint NaN/inf sanitize mask aligns weights with `x_data`. Composes with `norm='probability'` (per-row weights × 1/n_clean). `group_by` + column-name weights raises `NotImplementedError` (clean FIX1 scope).
- Coder QRC v1.28 binding + v1.32 amendments exercised.

### Tests

**Gate:** 767 → **776** / 0 / 1 skipped (+9 new §9-marked lock-tests in `TestPhase_13_27_Commit2_FIX1` class; `§9.WDH.2` promoted from signature-plumbing to rendering invariance; `§9.SDP.2` updated `vector_compose='inner'` → `vector_compose='outer'`; `§9.HW.*` for weights added).

---

## Phase 13.27.DF Commit 2 FIX1.FIX1: Assertion strengthen + sanitize cleanup

**Date:** 2026-05-16 (commit `b929ccb9`)
**Status:** ✅ Closed
**Review:** PHASE_13_27_DF_Commit2_FIX1_END panel (Sonnet53_R2 P1; Sonet50 P2; Sonnet53_R2 P2; §6 vote 3/5 for option (c) including Main Reviewer — defer n_y=1 silent-degrade to Phase 13.33)

### Objectives

Address panel feedback from PHASE_13_27_DF_Commit2_FIX1_END review without behavior change to production code paths.

### Implementation

- **P1 (Sonnet53_R2):** `§9.SDP.6` assertion was too weak. `ax.get_lines() >= 2` passed even on a single profile curve because `ax.errorbar()` creates ≥2 Line2D objects per curve (main line + caplines). Switched to `len(ax.containers) >= 2` — one `ErrorbarContainer` per rendered curve. Matches the precedent in `WDH.2` / `WDH.4`.
- **P2 (Sonet50) `§9.SDP.9` NEW:** Locks that the inner-mode `ValueError` message names the mismatched lengths (`vector=1, selection_vector=2`) so Phase 13.33 users immediately see why their default-inner call failed. Per §6 option (c), the n_y=1 silent-degrade fix is deferred to Phase 13.33; users must pass `vector_compose='outer'` until then — this test ensures the message makes the workaround discoverable.
- **P2 (Sonnet53_R2):** `histogram.py` `sanitize_for_plot` was called twice when `w_data is None`. Captured `x_clean` from the first call and reused.

### Tests

**Gate:** 776 → **777** / 0 / 1 skipped (+1 `§9.SDP.9` lock test).

---

## Phase 13.33.DF v1.0 M1: Normalized differential profiles (single-curve modes)

**Date:** 2026-05-17 (commit `61460df5`)
**Status:** ✅ M1 complete (M2 + FIX1 below; closure verdict at end of FIX1)
**Specification:** PHASE_13_33_DF_v1_1_Proposal_NormalizedDifferentialProfiles.md (v1.0 → v1.1 with M1/M2 milestone split per Coder/architect agreement)
**ADs:** AD-80 (sign convention), AD-81 (stats dict), AD-82 (pull bands)

### Objectives

Implement Milestone 1 of the 2-milestone split: single-curve `normalize=` modes complete; `group_by` / `facet_by` composition deferred to M2. Five differential normalization modes (`delta`, `ratio`, `log_ratio`, `pull`, callable) with two layouts (`overlay+diff`, `diff_only`) and 8 style keys for panel geometry + pull bands.

### Implementation

- **`profile()` gains `normalize=` and `normalize_layout=` kwargs** (AD-80/81/82).
- **5 modes:** `delta` (`v[0]−v[1]` with SEM error propagation), `ratio` (`v[0]/v[1]` with delta-method error and zero-denom mask), `log_ratio` (`ln(v[0]/v[1])` with non-positive-mean mask), `pull` (`(v[0]−v[1])/σ` with ±1σ/±2σ bands per AD-82), `callable` (user-supplied `f(stats_0, stats_1) → (values, errors)`).
- **2 layouts:** `overlay+diff` (default, 2-panel), `diff_only` (single panel).
- **8 new style keys** for panel geometry + pull bands.

**Architecture:**
- New `_dispatch_normalize_render` method orchestrates the two-pass rendering (top panel: signal + reference; bottom panel: differential). Architecturally a sibling of `_dispatch_faceted_render` — two-pass rendering is fundamentally different from `_draw_vector`'s iterate-and-render.
- `_compute_per_bin_mad_sigma` added (15 lines mirroring `_compute_per_bin_median`). All 8 cells of the 2×4 interaction matrix (mean/median × 4 modes) work.
- `_compute_normalize_transform` handles all 5 modes + masks for undefined bins (zero denominator, non-positive log argument, empty bin).

**§6 directive (Phase 13.27 FIX1.FIX1 deferred closure, option c):** Single-Y + `selection_vector` + default `vector_compose='inner'` is auto-rewritten to `vector_compose='outer'` per the §6 directive. This is a clean dispatcher convention (Q1), not a deferral or workaround — same scalar→vector convention validated in Phase 13.27 FIX1. Locked by `§9.NSY.1`, `§9.NSY.2`.

**Forwarding chain (Q2):** `_DRAW_FORWARDED_NAMES` and `_PROFILE_FORWARDED_NAMES` extended with `normalize` and `normalize_layout`; `_HIST_FORWARDED_NAMES` intentionally NOT extended (profile-only kwargs; auto-forwarding caused `hist()` crash via `Polygon.set()` — self-caught pre-delivery via regression).

**Pre-existing inconsistencies flagged for separate fix-up (not Phase 13.33 scope per Main Architect):** `central='median'` continues to use mean-based errors. `_compute_per_bin_mad_sigma` was added as a helper that a future fix-up can wire into the regular median path (locked by xfail `§9.MED.1` in Phase 13.34 M2).

### Tests

**Gate:** 777 → **799** / 0 / 1 skipped (+22 `§9` invariance tests). Locks AD-80 sign convention, AD-81 stats dict, AD-82 pull bands, §6 directive idempotency, and 3 validation paths. R6 validator green.

**Deferred to M2:** `group_by` + `normalize` composition (per-group differential), `facet_by` + `normalize` composition (K × 2 grid), 5 more tests.

---

## Phase 13.33.DF v1.0 M2: `group_by` + `facet_by` + `normalize` composition

**Date:** 2026-05-17 (commit `c6a3245f`)
**Status:** ✅ M2 complete

### Objectives

Close Phase 13.33.DF v1.0 by adding `group_by` and `facet_by` composition with `normalize=`.

### Implementation

- **`group_by` + `normalize`:** Per-group differential rendering. Each group gets its own signal+reference top-panel pair and its own differential bottom-panel curve. Colors distinguish groups; signal/reference distinguished by linestyle within group.
- **`facet_by` + `normalize`:** K × 2 grid where each facet column is an independent (top, diff) panel pair. Diff panels share y-axis across facets for cross-facet comparison. M2 v1.0 restriction: categorical column facets only; `facet_by_bins` / `facet_by_quantiles` + `normalize` raises `NotImplementedError` (CRR §11 flag for future fix-up).
- **Self-caught bug fix:** extended `_need_vector_dispatch` to fire on `normalize=`, pre-empting the `_column_mode_facet` short-circuit that would otherwise route facet+normalize to the regular facet path.
- Two new dispatchers added: `_dispatch_normalize_grouped_render` (~225 LOC), `_dispatch_normalize_faceted_render` (~225 LOC). Code duplication ~40 LOC of the inner 2-curve loop body is accepted for M2 in exchange for not touching the panel-approved M1 implementation. A unifying refactor is a candidate for a later structural fix-up phase (CRR §11).
- Both dispatchers reuse M1's helpers (`_compute_normalize_transform`, `_render_normalize_panel`, `_compute_per_bin_mad_sigma`, `NORMALIZE_MODES`) unchanged.

**Dispatch hierarchy when `normalize` is set** (panel-decided in v1.1 §3.7):
- `facet_by` → faceted dispatcher (K×2 grid, outer dimension)
- `group_by` → grouped dispatcher (per-group differentials)
- else → M1 single-render (preserved unchanged)

### Tests

**Gate:** 799 → **804** / 0 / 1 skipped (+5 new `§9.NG.1/2/3`, `§9.NF.1/2`; plus new fixture `df_three_fills_two_sectors`. M1's 22 tests untouched. 27 normalize tests total).

---

## Phase 13.33.DF v1.0 FIX1: Panel-feedback fixup

**Date:** 2026-05-17 (commit `94594f89`)
**Status:** ✅ Closed (panel: Claude40 [OK], Sonet50/51 [OK], Sonnet52_R1 [!], Sonnet53_R2 [!])
**Tag:** `PHASE_13_33_DF_v1_0_FIX1_END`

### Objectives

Address 1 P1 + 2 P2 surfaced in the v1.0 closure panel. Architect approved fixing all three in one commit.

### Implementation

- **P1 (Sonnet52_R1 + Sonnet53_R2 convergent — Coder QRC Rule 14 gap):** `§9.NF.3` — locks that `facet_by_bins` / `facet_by_quantiles` composing with `normalize=` raises `NotImplementedError` (instead of silently falling back to categorical facets or producing incorrect output). Tests BOTH paths (bins + quantiles) and BOTH the raise itself AND that the error message names the categorical-column workaround per Phase 13.16.DF actionable-error convention. Protects against silent message regression in future refactors.
- **P2a (Sonnet52_R1 + Sonnet53_R2 + Sonet51):** Dead code in `_dispatch_normalize_faceted_render` `diff_only` branch — `sharey=(None if i == 0 else None)` ternary always evaluated to `None`; actual `sharey` applied in the explicit loop below. Removed the no-op ternary.
- **P2b (Sonnet53_R2 — relevant for production validation on real ITS/TPC data):** First-pass performance optimization on `_dispatch_normalize_grouped_render` group enumeration — replaces a Python-side filter with a single pandas C-pass for large DataFrames (e.g., 4M-row ITS DataFrames: 12M Python comparisons → single C pass).
- **Bonus:** Error messages on `facet_by_bins` / `_quantiles` + `normalize` now explicitly name the "pre-bin into a categorical column" workaround so users discover the path forward from the error alone.

### Tests

**Gate:** 804 → **805** / 0 / 1 skipped (+1 new `§9.NF.3` locking 4 invariants: bins raises, bins message has workaround hint, quantiles raises, quantiles message has workaround hint).

---

## Phase 13.32.DF FIX1: Faceted rendering bug fixes (BUG-001 / BUG-002 / BUG-003)

**Date:** 2026-05-17 → 2026-05-18 (initial commit `195ab4ea` 2026-05-17 10:48; final commit `d0b04f88` 2026-05-18 09:40; tag `PHASE_13_32_DF_FIX1_END`)
**Status:** ✅ Closed
**Discovery:** Real-data validation on `time_series_tracks_0.root` (TPC/ITS QA, 2026-05-16) revealed three P1 bugs in `_dispatch_faceted_render` (`drawer.py`).
**Note:** Phase numbering follows the originating phase (13.32) rather than commit date — the FIX1 commits chronologically follow Phase 13.33 v1.0 M2 / FIX1. This is consistent with the post-13.27 Commit 1 phase ordering note.

### Bugs

- **BUG-001** (function entry + lines 2703, 2720): `__dfdraw_facet_bin__` internal column name leaked to subplot titles and `stats['facet_by']`. Root: line 2524 rebinds `facet_by` to `_effective_facet_col` (the temp `'__dfdraw_facet_bin__'` column name) before title generation. Fix: save `_facet_display_name = facet_by` at FUNCTION ENTRY (before `_facet_mode` is determined at line 2392) so it is defined for ALL facet modes (channel + column). Use the saved display name at `set_title()` and in `combined_stats['facet_by']`. Placement at function entry rather than inside the column-mode branch closes the v1.5 spec scoping gap (Sonet51 + Sonnet52_R1 P1) where channel-mode calls would have raised `NameError`. **Locks:** `§9.F001.1` + `§9.F002.2`.
- **BUG-002** (after per-subplot loop): `auto_title=True` ignored in faceted mode. Root: only the explicit `title=` kwarg (line 2329) triggered `fig.suptitle()`; `auto_title` in `**plot_kwargs` was correctly suppressed per-subplot but never used at the figure level. Fix: extract `auto_title` from `plot_kwargs` after the per-subplot loop; call `fig.suptitle()` via the `resolve_auto_title` + `parse_auto_title_parts` + `build_auto_title` pattern (verified against `drawer.py:1329-1333` authoritative call site). Uses `td['main']` / `td['sub']` per the actual return shape (verified against `_auto_title.py:97`; the spec evolution v1.3 → v1.4 → v1.5 surfaced repeatedly that `build_auto_title` returns `{'main', 'sub'}`, not `{'title'}`). `try/except` failsafe preserves the plot on `auto_title` import/build errors (same defensive pattern as `_draw_vector` failsafe at ~line 1230). **Lock:** `§9.F002.1`.
- **BUG-003** (line 20 import + lines 2526-2533): Facet bins sorted lexicographically not numerically. Root: `_format_interval_label` converts `pd.Interval` objects to strings at line 2520 (`'12.0-16.0'`); `sorted()` on these strings is lexicographic (`'1' < '4'`, so `'12.0-16.0' < '4.0-8.0'`). Fix: extend the line-20 import to include `_interval_sort_key` (already exists at `profile.py:71` — handles the exact label format produced by `_format_interval_label`); use as sort key when binning was applied (`_fby_bins` or `_fby_quantiles` set). **Lock:** `§9.F003.1`.

### Implementation Discipline

After 5 spec revisions each introducing new issues, switched to direct coding with source verification at every call site (per architect "faster way" directive). Each fix was smoke-tested before moving to the next; channel-mode `auto_title` (`§9.F002.2`) verified mid-flight before being locked by the test.

### Tests

**Gate:** 805 → **809** / 0 / 1 skipped (+4 new). New `tests/test_phase_13_32_df_fix1.py` with shared fixture `make_facet_test_df()` using `rng.uniform(0, 20, n)` to produce 5 populated bins whose labels diverge in lex vs numeric sort.

---

## Phase 13.34.DF v1.0: Capability Matrix taxonomy refresh + robustness gap tests

**Date:** 2026-05-18 (commit `463deb36`; re-committed as `abf5fe40` with identical title)
**Status:** ✅ Closed
**Tag:** `PHASE_13_34_DF_END`

### Objectives

Close 6-phase taxonomy drift (Phase 13.27 Commit 2 → Phase 13.32 FIX1) and add 3 robustness invariance test areas surfaced during the audit.

### Implementation

**M1 — Taxonomy refresh (no source code changes):**

- `feature_taxonomy.py`: +20 new feature entries across 7 sections:
  - `COLUMN_REF` (new): Phase 13.30 column-reference validation (12 paths)
  - `FACET` (5 new): `column_mode` (13.31), binning (13.32), `title` / `sort` / `auto_title` (13.32 FIX1)
  - `PROFILE` (1 new) / `QUANTILE` (1 new): `quantiles_grouped` (13.32), `single_y_dispatch` (13.27 FIX1)
  - `HIST` / `DATA` (1 new): hist `weights` parameter (13.27 FIX1)
  - `NORMALIZE` (11 new, entire section): all of Phase 13.33 M1+M2
- `test_layer_classification.py`: +156 §9-marked tests reclassified from default `smoke` to `invariance`. Generated via §9 marker grep.
- Fixed 4 stale references (1 from Sonnet52_R1 audit + 3 more caught by deterministic A.1 verification — `test_pre_phase_keys_unchanged`, `test_new_keys_present_when_default_policy`, `test_strategy_style_key_default`, `test_explicit_numeric_range_records_strategy_explicit` — all in Phase 13.28 entries pointing at renamed/removed tests).

**M2 — Robustness gap tests (1 new file, +8 §9 tests + 1 xfail):**

`tests/test_phase_13_34_df_m2_robustness.py`:
- `§9.MED.1` (xfail strict=False) — `central='median'` should use MAD-sigma error bars; locks the Phase 13.33 CRR §11 inconsistency until source-side fix lands. Converts to XPASS if/when fixed.
- `§9.STATS.1-3` — stats dict key contracts per plot kind. Catches silent renames that would break ADF / RootInteractive integration.
- `§9.X.1-5` — feature interaction tests for kwarg pairs. A single test per pair at Phase 13.32 delivery would have caught BUG-001/002/003.

### Matrix Snapshot

| Metric | Before | After |
|---|---|---|
| Features | 67 | **90** |
| Verified (✅) | 7 (10%) | **33 (37%)** |
| Invariance tests | 28 | **193** |
| Capability Matrix header phase | 13.15.DF *(stale auto-generated)* | 13.34.DF |

> Note (2026-05-21): The `CAPABILITY_MATRIX.md` header still auto-generates as `**Phase:** 13.15.DF`. This is a known generator-side staleness — the header variable in `scripts/generate_capability_matrix.py` has not been updated since Phase 13.15. Content (feature entries) is current through Phase 13.34 M2. Header drift to be fixed in a future tooling pass.

### Audits Consolidated

Claude48 (quantification + process framing + extra stale-ref catch), Sonnet52_R1 (implementation catalog + 1 stale-ref catch), Sonet50 (proposal structure + reclassification evidence).

### Tests

**Gate:** 809 → **817** / 0 / 1 skipped (+8 `§9` invariance tests in M2; M1 added no tests, only reclassified existing ones).

---

## Phase 13.34.DF FIX1 (BUG-010): Untracked test file caught by run_tests.sh staging check

**Date:** 2026-05-18 (commit `379f26bd` adds the missing test file; cleanup commit `14851d42` regens matrix timestamp + minor `drawer.py` touch-up)
**Status:** ✅ Closed
**Tag:** `PHASE_13_34_DF_FIX1_END`

### Bug

`BUG-010`: A pre-amend Phase 13.34 commit (`879a0835`) contained `drawer.py` only; `tests/test_phase_13_34_df_fix1_bug010.py` was untracked but `pytest` ran it (because it was in the working tree), producing an 822/0/0 gate over a 817-test commit. Bundle shipped with the inflated gate; reviewer caught it via diff inspection (Rule 8). Cost: 1 review cycle.

### Fix

Add the missing test file (`tests/test_phase_13_34_df_fix1_bug010.py`, 5 §9 tests locking the bug class). Matrix timestamp regen + `drawer.py` cleanup committed separately.

### Tests

**Gate:** 817 → **822** / 0 / 1 skipped (+5 `§9` tests locking BUG-010 invariants).

---

## Phase 13.34.DF FIX2 (BUG-011): `run_tests.sh` pre-bundle staging check

**Date:** 2026-05-18 (commit `b38395db`)
**Status:** ✅ Closed
**Tag:** `PHASE_13_34_DF_FIX2_END`
**Discovered by:** Sonet50 in PHASE_13_34_DF_FIX1_BUG010 review.

### Bug

`BUG-011` (process-class bug): the BUG-010 incident represents an entire class — new `.py` files in `tests/` are created in the working tree but never staged. `pytest` finds them and reports a green gate; the bundle ships with that gate; reviewers read it as trustworthy; but the commit has fewer tests than the gate claims.

### Fix

~35 LOC inserted in `run_tests.sh` between the test-summary section and the bundle-packaging section. Blocks bundle creation (exit 1) when `git status --porcelain tests/ | grep '^??' | grep '\.py$'` is non-empty. Override: `DFDRAW_SKIP_STAGING_CHECK=1 bash run_tests.sh`. Test results from the blocked run are still saved to `test_logs/` — only the `reviewer.zip` artifact is prevented.

### Validation

Manual reproduction of 5 scenarios (clean, untracked, override, non-py untracked, modified-tracked).

### Tests

**Gate:** 822 / 0 / 1 skipped (no test count change — tooling-only).

---

## Phase 13.35.DF v1.3: `group_by_bins` + `hist_norm` for `hist()` (BUG-013 — hist side)

**Date:** 2026-05-20 (commit `3b910aec`)
**Status:** ✅ Closed
**Tag:** `PHASE_13_35_DF_END`
**Specification:** PHASE_13_35_DF_v1_3_Proposal_HistGroupByNorm.md (notes repo). v1.0 → v1.1: 5 factual source errors (Claude40 panel). v1.1 → v1.2: 3 P1s (Sonet50 / Sonnet52_R1 / Sonnet53_R2 panel). v1.2 → v1.3: 1 P1 stacked branch + 1 P2 comment + 1 §9 test add.

### Objectives

Fix 3 `AttributeError` crashes confirmed in 2026-05-20 live testing (T2: `group_by_bins` leaks to `ax.hist()` (no facet); T3: same crash in faceted subplot path; T4: `hist_norm` leaks to `ax.hist()`). These are the hist-side instance of BUG-013 — the same kwarg-propagation bug class as Phase 13.16.DF FIX1, here on the histogram path.

### Root Cause

`group_by_bins`, `group_by_quantiles`, `hist_norm`, `min_entries` absent from BOTH `DFDraw.hist()` method signature AND `draw_hist()` function signature AND `_HIST_FORWARDED_NAMES` → fall into `**kwargs` → forwarded to `_draw_hist_grouped()` `**hist_kwargs` → reach `ax.hist()` which rejects.

### Implementation (7 edits, ~300 LOC source + 384 LOC tests)

1. `_HIST_FORWARDED_NAMES`: add `group_by_bins`, `group_by_quantiles`, `hist_norm`, `min_entries` (`drawer.py`).
2. `draw_hist()` signature: add the 4 as explicit params (`histogram.py`).
3. `group_by` routing block: BUG-012 guard (float + `nunique() > 20` + no bins → `ValueError` with `'group_by_bins=N'` guidance), `pd.cut` / `pd.qcut` binning with float16 → float32 upcast, `df.copy()` to avoid caller mutation, shared bin edges from `x_data` (sanitized at lines 259-307, NOT `df[x].dropna()` which would bypass `nan_policy`), `stats_dict['n_groups']` population (was missing — T1 observation).
4. `_draw_hist_grouped()`: pop `'weights'` from `hist_kwargs` to avoid `ax.hist()` double-weights `TypeError`; one-pass stacked loop building `data_list` + `labels` + `surviving_colors` in lockstep (fixes v1.2 P1-D label misalignment AND P3 color-shift); `n_rendered` counter in overlaid branch (v1.1 P1-C); `str(group)` labels (no `_format_interval_label` import — v1.1 P1-A); return → `int`.
5. `_group_weights()`: new module-level helper (probability / density).
6. `DFDraw.hist()` signature: add the 4 params (required by R6 module-import validator — discovered during implementation).
7. `_dispatch_faceted_render()` call: forward the 4 params explicitly (required after Edit 6 consumes them off `**kwargs` — discovered when T3 architect-call test initially failed).

### Architect Production Reproducer (now works end-to-end)

```python
adf.draw('dyp_I6-dyp_recoV2', type='hist',
         group_by='z', group_by_bins=5,
         min_entries=25, facet_by='sec')
```

Live-tested: 9 sector panels × 5 drift-coordinate bins each.

### Tests

**Gate:** 822 → **833** / 0 / 1 skipped (+11 `§9` invariance):

- `HGB.1-3` — `group_by_bins` + `facet_by` + shared bin edges
- `HN.1-3` — `hist_norm` probability / None / density (math at data layer)
- `HGS.1-2` — `stats['n_groups']` + BUG-012 guard with actionable error
- `HGSt.1` — stacked + `min_entries` label alignment (regression lock for v1.2 P1-D, verified by failure injection)
- `HGBC.1-2` — backward compat (single-hist path; categorical `group_by`)

---

## Phase 13.36.DF v1.2: User style kwargs override auto-cycle in `group_by` path (BUG-013 — style-override side)

**Date:** 2026-05-20 (commit `2f4d959f`)
**Status:** ✅ Closed
**Tag:** `PHASE_13_36_DF_END`
**Rolling tag:** `PHASE_BEGIN_dfdraw` → `2f4d959f`
**Specification:** PHASE_13_36_DF_v1_2_Proposal_UserStyleOverride.md
**Review:** v1.2 panel — Sonet50 [OK], Sonet51 [!] (P1 markersize crash in hist vector path — fixed at code time per architect "no v1.3" directive), Claude40 [✅ APPROVED]

### Bug Closed

`BUG-013` (style-override side): `marker='s'`, `color='red'`, `markersize=10` silently ignored when `group_by` was active. The grouped rendering path (profile + hist) used the per-group color/marker cycle unconditionally, discarding user style kwargs. Architect's primary use case (TPC/ITS calibration overlay, same=True second call):

```python
adf.draw('y:row', group_by='drift', group_by_bins=5)
adf.draw('y:row', group_by='drift', group_by_bins=5,
         same=True, marker='s')
```

### Architect Priority Rule (2026-05-20)

> **user kwarg > channel auto-cycle > style default**
>
> `None` = "user did not pass" (matches matplotlib default-color semantics).

### Profile Fix (`plots/profile.py`)

- `_draw_profile_grouped()` signature: add `_user_marker`, `_user_markersize`, `_user_color` named params (default `None`).
- `draw_profile()` captures `_ud_user_marker`, `_ud_user_markersize`, `_ud_user_color` at lines 311-321 **BEFORE** the style fill-in (which replaces `None` with style defaults). Without pre-fill-in capture, the sentinel arrives as `'o'` even when user passed nothing → cycle override (CODER NOTE §5.1).
- `draw_profile()` call to `_draw_profile_grouped()`: pass `_user_marker=_ud_user_marker` etc. (the pre-fill-in captures).
- REMOVED: `marker=marker`, `markersize=markersize` from the call.
- REMOVED: `profile_kwargs.pop('marker' / 'markersize')` (nothing left to pop).
- Per-group loop: `is None` check before palette / marker cycle.
- `UserWarning` fires ONCE per call (`i == 0`) when `color=` makes all groups uniform ("indistinguishable").

### Hist Fix (`plots/histogram.py`)

- `_draw_hist_grouped()` signature: add `_user_color` named param (default `None`).
- `draw_hist()` call to `_draw_hist_grouped()`: pass `_user_color=color`.
- `draw_hist()` body (early, before routing): pop `'marker'` from `kwargs` + issue `UserWarning`. v1.2 spec said pop inside `_draw_hist_grouped`, but that path is only reached for grouped hist — non-grouped / vector path would crash `ax.hist` (`AttributeError` on `marker`). Single source of truth at `draw_hist` top (CODER NOTE §5.2).
- Per-group color: sentinel checked in BOTH stacked + overlaid branches.

### `_*_FORWARDED_NAMES` (`drawer.py`)

- `_PROFILE_FORWARDED_NAMES` += `marker`, `color`, `markersize` (all 3 are explicit params of `draw_profile()`).
- `_HIST_FORWARDED_NAMES` += `marker`, `color` ONLY (NOT `markersize` — `draw_hist()` has no `markersize` explicit param; adding it would crash `ax.hist` via vector path — Sonet51 P1 from v1.2 review).

### `DFDraw` Method Signatures + Bodies (`drawer.py`)

- `DFDraw.profile()` += `color`, `marker`, `markersize` params (R6 validator).
- `DFDraw.hist()` += `color`, `marker` params (R6 validator).
- Both bodies forward the params explicitly to `draw_profile` / `draw_hist`.
- `same=True` auto-color injection refactored: (a) assign to local `color` var not `kwargs['color']` (collision); (b) skip auto-color when `group_by` is active — preserves pre-13.36 behavior (auto-color was silently dropped in grouped path) and prevents false-positive "indistinguishable" `UserWarning` (CODER NOTE §5.3).

### Scope Boundary

`linestyle`: out of scope — already works (brainstorm + v0 / v1 panel live tests, 2026-05-20). `§9.LS.1` locks current behavior.

### Implementation Deviations from v1.2 Spec (Mandatory Disclosure)

- Edit 17 (NEW): pre-fill-in capture for `_ud_user_*` — caught at smoke test, not anticipated in spec. §5.1.
- Edit 18: `marker` pop moved from `_draw_hist_grouped` to `draw_hist` top — caught by full regression (96 vector hist failures). §5.2.
- Edit 15/16: `same=True` + `group_by` guard — caught at smoke test, false-positive `UserWarning`. §5.3.

### Spec History

- **v1.0 (Sonet50):** 3 P1s including "color not forwarded to grouped path".
- **v1.1 (Sonet50):** attempted P1-A fix via `profile_kwargs.pop`, returns `_UNSET` unconditionally (color / marker / markersize consumed by explicit signature, not in `**kwargs`). Sonnet53_R2 [X], Claude48Coder [X].
- **v1.2 (Claude48Coder, architect-greenlit takeover):** Sonnet53_R2 Option A — local variables, `None` sentinel.

### Tests

**Gate:** 833 → **843** / 0 / 1 skipped (+10 `§9` invariance: `SO.1-5`, `SOH.1-2`, `LS.1`, `SC.1`, `VF.1`).

- `SO.1-5` — TestUserStyleOverride (marker, color+warning, markersize, default cycle preserved, architect primary use case)
- `SOH.1-2` — TestHistStyleOverride (marker triggers warning, color uniform on step histtype)
- `LS.1` — TestLinestyleDocumentationLock (linestyle='None' regression lock; OUT-OF-SCOPE behavior locked)
- `SC.1` — TestScatterUntouched (scatter color / marker unaffected)
- `VF.1` — TestVectorPathForwarding (vector expression + marker uniform via `_PROFILE_FORWARDED_NAMES`)

**Run platforms:**
- Architect's Mac Py 3.9.6: **843 / 0 / 1 skipped / 1 xfailed** (per `SUMMARY_20260521_092254.txt`, commit `2f4d959f`)
- Linux Py 3.12 (Coder env): 810 / 1 / 33 — 1 fail + 33 skipped are pre-existing per Phase 13.35.DF conversation summary, NOT caused by Phase 13.36.

---


---

## Phase 13.37.DF v1.1: Histogram Robustness — BUG-014 / BUG-015 / BUG-016 + `hist_errors` + `linestyle_cycle`

**Date:** 2026-05-21
**Commit:** `67fccf3d16dc60a123f6d4b84fd5e9042b148635`
**Tag:** `PHASE_13_37_DF_END`
**Gate:** 843 → **867** (+24 §9 invariance tests)

**Five items closed in the histogram / profile grouped path.** Items 1–3 are bug fixes; items 4–5 are new features. All extend the Phase 13.36 sentinel pattern (capture user-explicit value BEFORE style fill-in) from 3 kwargs (color/marker/markersize) to 5 (adds edgecolor and linestyle).

### BUG-014: `histtype='step'` all-black lines
`edgecolor` style default (`"black"`) overrode the per-group `color=` for step histograms in matplotlib. Fix: capture `_ud_user_edgecolor` before style fill-in at `draw_hist` line 290; when `_histtype=='step'` and user did not pass `edgecolor`, use `group_color` as edgecolor. User-explicit `edgecolor='red'` wins uniformly. Bar and stepfilled modes use style default (`"black"`) unchanged.

### BUG-015: Profile `group_by` float without bins — memory hang / OOM
Mirrors Phase 13.35 hist BUG-012 fix. Guard at `profile.py:477` before `pd.cut`: `nunique() > 20` → `ValueError` with `group_by_bins=N` guidance. Same limitation: expression-string `group_by='abs(tgl)'` falls through (not a column name) — deferred.

### BUG-016: `_interval_sort_key` was a no-op for `pd.Interval` objects
Post-Phase-13.35, `_draw_hist_grouped` receives raw `pd.Interval` values from `pd.cut()`; `str(Interval(10.0, 12.0))` is `'(10.0, 12.0]'` which defeated the digit-dash detector and fell through to lexicographic sort → wrong legend order for bins crossing 10. Fix: `hasattr(label, 'left')` guard at function entry returns `(0, float(label.left))`. **Behaviorally verified by execution** (per Opus2 QRC lesson — three Sonnet reviewers had marked "10/10 checklist ✅" without executing the function and all missed it). Backward compat: string labels unaffected.

### `hist_errors=True`: Poisson error bar overlay
Raw counts: `yerr=√n`. Probability: `yerr=√n/N`. Density: per-bin `yerr=√n/(N·bw_i)` (vectorized `np.diff(edges)`, NOT mean width — CP1-8). Weighted Poisson via `Σw²` when `weights=` column is set. Zero-count bins skipped. Ungrouped path uses edges from `np.histogram()` return, not `bins=` int (CP1-7). Error bar color follows Phase 13.36 sentinel: `group_color` not `colors[i]`. Two new style keys: `hist.error_capsize=2`, `hist.error_elinewidth=1.0`.

### `linestyle_cycle=True`: per-group linestyle from channels
Phase 13.26 `channels.cycles.linestyle` style key. Sentinel extended to linestyle — `_ud_user_linestyle` capture before style fill-in. When `linestyle_cycle=True` AND `_user_linestyle` is None → cycle per group. Explicit `linestyle='--'` always wins. Composable with `same=True`.

**Spec history:** v1.0 (Sonnet52_R1): 2 P0 + 11 P1 + 3 P2 → Opus2 consolidated [X]. v1.1: all 13 P0+P1 findings addressed → Approved.

**Tests (+24):** TestBUG014StepColor (4), TestBUG015ProfileGuard (2), TestBUG016IntervalSort (3), TestHistErrors (10: HE.1-9 + HE.style), TestLinestyleCycle (5)

- Architect's Mac Py 3.9.6: **867 / 0 / 1 skipped / 1 xfailed** (commit `67fccf3d`)
- Linux Py 3.12 (Coder env): 834 / 1 / 33 — pre-existing failures unchanged

---

## Phase 13.37.DF FIX1: Test Expansion — Phase 13.36 Backward Compat Locks

**Date:** 2026-05-21
**Commit:** `095d6f28` (staged at docs commit `2e5c83df` during bundle generation)
**Tag:** `PHASE_13_37_DF_FIX1_END`
**Gate:** 867 → **870** (+3 §9 invariance tests)
**Source changes: ZERO** (test-only phase)

**Gap identified by dual independent audit (Sonnet53_R2 + Opus2, 2026-05-21):** Phase 13.36 rewrote `_draw_profile_grouped()` signature (added `_user_marker`, `_user_markersize`, `_user_color` sentinel params) and refactored `same=True` auto-color injection (added `group_by-is-None` guard preventing false-positive `'indistinguishable'` UserWarning). Existing smoke tests did not lock behavioral invariance after these Phase 13.36 changes.

**Two spec bugs caught at code time (fix-at-code-time per Phase 13.36 precedent):**
- **Filter bug:** spec used `label != '_nolegend_'` filter; matplotlib errorbar central `Line2D` has `label='_nolegend_'`. Fix: `marker != '_'` filter (Phase 13.36 convention). This anti-pattern appeared in the v1.0 spec by Sonnet53_R2 despite the existing convention.
- **SO.COMPAT.3 premise bug:** vector-vs-scalar A≡B comparison was wrong — vector path applies Phase 13.26 vector→linestyle channel; scalar `same=True` does not replicate this. Verified by execution: `MATCH? False`. Reformulated as 3 direct vector-path invariants: (a) N_groups × N_vector lines, (b) N_groups distinct colors, (c) N_vector distinct linestyles.

**Feature promotions (capability matrix):**

| Feature | Before | After | Tests |
|---|---|---|---|
| `PROFILE.group_by_bins` | ☑️ Smoke | ✅ Verified | 4 → 5 |
| `SAME.auto_features` | ☑️ Smoke | ✅ Verified | 4 → 5 |
| `VECTOR.color_cycle` | ☑️ Smoke | ✅ Verified | 3 → 4 |

**Verified count: 34 → 37**. Invariance: 193 → 196. Proof tests: 422 → 425.

**Panel:** Sonet50 [OK] + Sonet51 [OK] + Sonnet52_R1 [!] — all approved.

- Architect's Mac Py 3.9.6: **870 / 0 / 1 skipped / 1 xfailed** (commit `095d6f28`)

---

## Phase 13.38.DF v1.1: Scatter Enhancements — BUG-017 + `xerr`/`yerr` + Expression `color`/`marker`

**Date:** 2026-05-21
**Commit:** `0f5257435124b88b02a556c826a0f43345f1e46c`
**Tag:** `PHASE_13_38_DF_END`
**Gate:** 870 → **889** (+19 §9 invariance tests)

**Three independent items, all surface-adjacent:**

### BUG-017: `facet_by` float column without bins — memory hang / OOM
Third instance of the BUG-012/BUG-015 float-guard class (hist Phase 13.35, profile Phase 13.37, `facet_by` dispatch this phase). Guard at `_dispatch_faceted_render` column-mode groups block (`drawer.py:~2571`): `nunique() > 20` + no `facet_by_bins`/`facet_by_quantiles` → `ValueError` with `facet_by_bins=N` guidance. **Fix-at-code-time discovery:** `facet_by_bins`/`facet_by_quantiles` arrive via `**plot_kwargs`, NOT direct named params — spec assumed direct params (CRR §2.1).

### `xerr=` / `yerr=` on `scatter()`: per-point error bars
Column name or `df.eval()` expression. `_eval_error()` helper with explicit three-tier NaN policy: raise on 100% non-finite (programming error), warn at >50% (data quality), silent zeroing at ≤50% (per-point safety). `nanfrac` written to stats dict. When `xerr`/`yerr` provided: render via `ax.errorbar()`; otherwise `ax.scatter()` path unchanged (dispatch invariance locked by §9.SE.5). Three new style keys: `scatter.error_capsize=2`, `scatter.error_elinewidth=1.0`, `scatter.error_ecolor=None`.

### Expression `color=` and `marker=` for scatter
**`_process_color()` dispatch order reordered (CP0-1):** None → array → **column-name** → fixed-color (`to_rgba`) → `df.eval()` → terminal. Column-name check precedes `to_rgba()` to preserve backward compat for column names that are also matplotlib colors (`'b'`, `'r'`, `'k'`). Regression-lock: §9.ECM.6 (column `'b'` → colormap, NOT fixed blue). **Fix-at-code-time discovery:** `get_facecolor()` returns the base color until `fig.canvas.draw()` — tests must use `get_array()` which directly probes the dispatch decision (promotes to QRC v1.32 carry-forward).

Boolean `marker=` expression (`'ncl > 100'`) → two-marker encoding (True → `'s'`, False → `'o'`). Per-point rendering via `np.unique` loop with `label='_nolegend_'` (no spurious legend duplication). Single-path scatter only (group_by + expression raises clean error — §9.ECM.7).

**v1.0 panel (Opus2 [X], 15 findings):** CP0-1 dispatch order (P0, 3/4 reviewers converged). v1.1 applied all 7 panel fixes + 4 strengthening tests.

**Tests (+19):** FBGUARD.1-2 (BUG-017 float guard), SE.1-9 (error bars: extents/expressions/NaN/style/group_by/facet), ECM.1-8 (expression color+marker: colormap/backward-compat/error/CP0-1-regression/group_by-scope/2-marker/compose/legend)

**Fix-at-code-time disclosures (CRR §2):** BUG-017 `plot_kwargs` scope; ECM.1/6 `get_facecolor` → `get_array()`; SE.8 DataFrame length mismatch.

**Capability matrix:** Verified 37 → 42 (+5). Invariance 196 → 215 (+19). Proof tests 425 → 444 (+19). Features 100.

- Architect's Mac Py 3.9.6: **889 / 0 / 1 skipped / 1 xfailed** (commit `0f525743`)

---

## Phase 13.39.DF v1.2: 2D Profile (`profcolz`) + Time Axis + Scatter3D

**Date:** 2026-05-21
**Commits:** `b024414ec14432eeee51ba19397e908622fe1874` (main), `3c5d45474dcbdd0683969adde76342fa904f5059` (docs/tag)
**Tag:** `PHASE_13_39_DF_END`
**Gate:** 889 → **913** (+24 §9 invariance tests)

**Three items using the new `z:y:x` expression infrastructure:**

### Item 1: 2D Profile — `draw_profile2d()` via `z:y:x` expression
Expression `'z:y:x'` (`colon_count == 2`) intercepted at `DFDraw.profile()` entry AFTER selection/sampling, BEFORE `_parse_expr()` (CP1-5 — exact insertion point). New helper `_split_top_level_colons_3()` alongside existing `_split_top_level_colon()`. Algorithm: `scipy.stats.binned_statistic_2d` for per-cell mean/median; `min_entries=N` masks low-count cells → NaN; `pcolormesh` + colorbar mirrors `draw_hist2d` pattern. `bins=[nx,ny]` list form OR `bins=nx + bins2=ny` scalar form. `vmin`/`vmax`/`cmap`/`clabel`/`colorbar`/`norm='log'` supported. z/y/x accept column names or `df.eval()` expressions. Backward compat: `'y:x'` (`colon_count == 1`) routes unchanged to 1D profile — locked by §9.P2D.8.

ROOT equivalent: `TProfile2D::Draw("colz")`.

### Item 2: Time Axis — `time_format=` kwarg
Pre-conversion approach (N3): `x_data` converted to matplotlib date numbers via `mdates.date2num()` BEFORE any `ax.*` render call. **CP1-4 auto-detect:** `datetime64` column dtype detected BEFORE `astype(float)` — else int64-nanosecond cast becomes ~1.76e15 → `pd.to_datetime(unit='s')` crashes "year 56003119 out of range" (**discovered at code time — §2.1 disclosure**). Auto-detect branches: `datetime64*` → `mdates.date2num(x_arr)` directly; else → `pd.to_datetime(unit='s').to_pydatetime()`. Applied to `draw_profile`, `draw_hist`, `draw_scatter`, `draw_profile2d`. DateFormatter/AutoDateFormatter applied post-render. `time_format='auto'` → `AutoDateFormatter`; any strftime string → `DateFormatter`. `unit='s'` assumption: Unix epoch seconds (TPC/ITS `time_s` columns) — a `time_unit='s'` companion parameter is a future generalization.

**§9.TA.5 regression lock:** uses realistic 2024-epoch timestamps (~1.7e9) so `ticks[0] < 100000` distinguishes pre-converted matplotlib date numbers (~19723) from raw Unix timestamps (~1.7e9). Epoch-0 data would make both paths pass — CP1-1 fix.

### Item 3: Scatter3D — `type='scatter3d'`
Explicit `type='scatter3d'` (AD-SC3D-1: less surprising than auto-routing; recommended by Sonet50 + Sonet51 independently). `z:y:x` expression → 3D scatter via `mpl_toolkits.mplot3d`. Reuses Phase 13.38 `_process_color()` + `_process_size()` unchanged. `color=`/`size=` accept column names or `df.eval()` expressions. `elev=`/`azim=` forwarded to `ax.view_init()`. `same=True` guard: raises `ValueError` if existing axes is not `Axes3D`. `group_by` + `scatter3d` raises (scope boundary locked by §9.SC3D.7). Stats dict: `{n, mean_x, mean_y, mean_z, std_x, std_y, std_z, n_filtered}` — all 3 means locked to 1e-9 (§9.SC3D.6).

**Fix-at-code-time disclosures (CRR §2):**
- §2.1: `_x_is_datetime` detection BEFORE `astype(float)` (spec put check too late)
- §2.2: named params `bins2=bins2` / `time_format=time_format` (not `kwargs.get`) — R6 validator promotes FORWARDED_NAMES entries to named params on parent method
- §2.3: scatter3d dispatch BEFORE `_parse_expr()` in `DFDraw.draw()` — else colon_count > 1 rejects `"z:y:x"` before type dispatch
- §2.4: `elev`/`azim` removed from `_SCATTER_FORWARDED_NAMES` — scatter3D-only params not on `DFDraw.scatter` signature

**v1.1 → v1.2 changes (6 fixes from Sonet50 consolidated panel, 6 reviewers):** CP1-1 (TA.5 realistic timestamps), CP1-2 (scipy `range` fix — 3-level nested → 2-level), CP1-3 (SC3D.6 locks all 3 means), CP1-4 (datetime64 auto-detect), CP1-5 (dispatch insertion point), CP1-6 (scipy required, fallback dropped), CP2-1 (group_by scope locks), CP2-2 (same=True 3D guard).

**Tests (+24):** P2D.1-8+10 (9 profile2d tests), TA.1-7 (7 time-axis tests: profile/scatter/hist/profile2d + datetime64 no-crash), SC3D.1-8 (8 scatter3d tests: basic/color-expr/size-col/selection/backward-compat/stats-all-3-means/group_by-scope/same-true-guard)

**Capability matrix:** Verified 42 → **47** (+5). Invariance 215 → **239** (+24). Proof tests 444 → **468** (+24). Features **105** (+5).

**QRC v1.32 carry-forward items (2 new, now 6 total):**
- "Dtype-preserving transforms must run BEFORE blanket `astype(float)`" (datetime64, categorical, decimal columns silently destroyed otherwise)
- "Named param ≠ kwarg after FORWARDED_NAMES promotion" (R6 validator promotes → access by name, not `kwargs.get`)

- Architect's Mac Py 3.9.6: **913 / 0 / 1 skipped / 1 xfailed** (commit `b024414e`)
- Pre-existing failure (Linux Py3.12 only): `test_vector_draw_kwarg_surface_enumeration` — pandas `StringDtype` not matched in `_process_color` (Phase 13.38 debt). Mac Py3.9.6 unaffected. Tracked as Phase 13.40 candidate.

## Phase 13.40.DF v1.0: Cumulative Histogram (`cumulative=True/-1/False`)

**Date:** 2026-05-22
**Commit:** `67d125e2`
**Tag:** `PHASE_13_40_DF_END`
**Status:** ✅ Complete (923 / 0 / 1 skipped / 1 xfailed)

### Objectives
- Add ROOT `TH1::Draw("cumulative")` equivalent to `hist()`
- Three accepted values: `True` (ascending CDF/ECDF), `False` (default, byte-identical backward compat), `-1` (descending / survival, ROOT convention)
- Compose with existing axes: `norm='probability'` → 0-1 ECDF; per-group overlay AND stacked; `facet_by`; `histtype='step'` (HEP-standard step ECDF)
- Correctness guard: `hist_errors=True + cumulative` → `NotImplementedError` (Poisson per-bin errors are independent; cumulative bins are correlated)

### Implementation
**Files modified:** `plots/histogram.py`, `drawer.py` (4 internal call sites threaded), `_HIST_FORWARDED_NAMES` (added `cumulative`)

**Architecture:**
- matplotlib native `cumulative=` forwarded explicitly at 4 internal call sites — Phase 13.39 §2.2 lesson applied recursively: `DFDraw.hist → draw_hist → _draw_hist_grouped → ax.hist`; **also through `_dispatch_faceted_render` for `facet_by` composition** (CP2-1 regression-lock on 3rd call site)
- Vector dispatch `[x,y]` propagates `cumulative` correctly (Phase 13.16.DF FIX1 bug class lock)

### Testing
- **+10** invariance tests under `HIST.cumulative` (CDF byte-equality, descending sign convention, group_by+stacked, facet_by per-cell, histtype='step', M5 NotImplementedError guard)
- Test count: 913 → **923** (+10)

### Carry-Forward Lesson
- Per-feature regression matrix should include `_dispatch_faceted_render` as a mandatory call site for any new `**kwargs`-eligible parameter — the 3rd-call-site bug from Phase 13.16.DF FIX1 keeps recurring

---

## Phase 13.41.DF v1.0: N-D Faceting via `facet_by=List[str]` (1D/2D/3D)

**Date:** 2026-05-23
**Commit:** `530954d1`
**Tag:** `PHASE_13_41_DF_END`
**Status:** ✅ Complete (initial gate 942; FIX1 + FIX2 carry to 946)

### Objectives
- Extend `facet_by` from `Union[str, None]` to `Union[str, List[str]]` for 1D/2D/3D faceting
- Convention LOCKED matching numpy/pandas `(n_rows, n_cols, ...)` shape:
  - `facet_by[0]` → **ROW** (vertical within figure)
  - `facet_by[1]` → **COLUMN** (horizontal within figure)
  - `facet_by[2]` → **FIGID** (separate figures, one per value)
  - `facet_by[3+]` → `NotImplementedError`
- New params: `share_x`, `share_y` ∈ `{'all','row','col','none'}`; `share_across_figures: bool` (3D global range lock)
- 3D returns `(List[Figure], List[axes_2d], List[stats_dict])` — **DEVIATES** from standard `(fig, ax, stats)` contract; documented prominently in inline help

### Implementation
**New helpers:** `_normalize_facet_args`, `_to_mpl_share`, `_validate_share_axis_value`, `_resolve_facet_values`, `_filter_facet_value`, `_compute_global_ranges`

**Per-plot-kind lock for `share_across_figures` (CP1-2):**
- `scatter` locks x AND y
- `hist`/`profile` locks x only (y auto-scales per figure to handle sparse-figID variance)

**Per-plot-kind dispatch for x-range:**
- `hist` uses `range=` (matplotlib convention)
- `profile` uses `range=` which `DFDraw.profile` remaps to `draw_profile`'s `x_range=` internally
- `scatter` uses `ax.set_xlim`/`set_ylim` post-draw (no native `range` params)
- `hist` also locks `ax.set_xlim` post-draw (`range=` only locks bins, not axis xlim)

**Empty cell handling:** `'(no data)'` diagnostic + `stats={'n':0, 'empty':True}`

### Architectural Significance
**dfdraw is the FIRST major plotting library with a unified API where the Nth faceting dimension generates separate figures.** seaborn / ggplot2 / plotly / altair all require manual loops for the 3+ dimension case.

### Testing
- **+23** invariance tests under `FACET.list_grid`
- Test count: 923 → **942** (+19; FIX1 + FIX2 carry forward to gate 946)

---

## Phase 13.41.DF FIX1: 3 bug fixes carried forward from v1.6 panel

**Date:** 2026-05-23
**Commit:** `b84576a0`
**Tag:** `PHASE_13_41_DF_FIX1_END`
**Status:** ✅ Complete (predecessor `530954d1`)

### Bugs Fixed
- Three bugs flagged in v1.6 panel review of Phase 13.41.DF — closed in FIX1 commit (commit message terse; see panel review summary for details)

### Testing
- Test count: 942 → **945** (+3) per panel-confirmed tally

---

## Phase 13.41.DF FIX2: 5 panel-flagged P2/P3 items + FBY.23 lock

**Date:** 2026-05-23
**Commit:** `70b94a3e`
**Tag:** `PHASE_13_41_DF_FIX2_END`
**Status:** ✅ Complete (gate 946; predecessor `b84576a0`)

### Items Closed
- 5 panel-flagged P2/P3 items from v1.6 close panel
- `FBY.23` lock added to lock the N-D faceting feature shape

### Testing
- Test count: 945 → **946** (+1)

---

## Phase 13.42.DF v1.0: Inline Fits (`fit=` parameter on hist/profile/scatter/draw)

**Date:** 2026-05-26
**Commit:** `38aed2d8`
**Tag:** `PHASE_13_42_DF_END`
**Status:** ✅ Closed (973 / 0 / 1 skipped / 1 xfailed)

### Objectives
- Inline ROOT-style fits as a first-class parameter on hist/profile/scatter/draw — implements `PHASE_13_42_DF_v1_4_Proposal_InlineFits.md`
- Predecessor: `PHASE_13_41_DF_FIX2_END` (gate 946)
- Three input forms: `str` shorthand (`fit='gauss'`), `dict` spec (`fit={'fun':'gauss','initial_guess':[...],'bounds':...}`), `Callable`
- List form for vector dispatch (compound on single curve OR per-channel pair — see v1.4 §6.3)
- Per-channel `linestyle_cycle` for multi-fit overlays
- Stats dict integration: `stats['fit']` is `List[List[Dict]]` (list of fit-result dicts per curve)
- Composition with `group_by`, `facet_by`, `vector_expr`, `normalize=` (silent consume per CP1-5)

### Implementation
**New module:** `plots/fits.py` (registry-based dispatch, scipy.optimize.curve_fit, fail-soft default)
**New module:** `plots/_fit_render.py` (ROOT-style param textbox + overlay rendering)
**Modified:** `plots/histogram.py`, `plots/profile.py`, `plots/scatter.py`, `drawer.py`

**Registry:**
- Built-in fits: `gauss`, `pol0` … `pol5`, `linear` (alias `pol1`), `landau` (placeholder), `expo`
- Public `register_fit(name, function, n_params, guess_fn)` for user-defined

**Style keys (7 new):** `fit.linewidth`, `fit.linestyle_cycle`, `fit.position`, `fit.text_format`, `fit.text_padding`, `fit.text_fontsize_default`, `fit.text_fontsize_facet` — see Phase 13.42 FIX1 D-2 disclosure for v1.0 silent-no-op defect

**CP1-5 — `normalize=` + `fit=`:** `normalize` is consumed first; `fit=` silently no-op when both set. Documented behavior; F.26 lock.

**P1-B (Sonnet54 finding, fixed pre-tag):** profile grouped fit path returned single fit on combined data instead of per-group dict. Fix at `plots/profile.py` grouped block — F.27 lock.

### Testing
- **+27** invariance tests under `FIT.inline` (F.1 - F.27, with F.23 in NumericalCorrectness class)
- Test count: 946 → **973** (+27)

### Panel Verdict
v1.4 proposal: 5-reviewer panel [!] APPROVED WITH COMMENTS through 4 versions (v1.0 → v1.4); close panel 5-reviewer [OK]. CRR v2 closed by Sonet50 [!] post-Sonnet54 P1-B fix.

---

## Phase 13.42.DF FIX1: Production-Gate Bug Closure + Interface Lock

**Date:** 2026-05-27
**Commits:** `82aaa903` (main) + `28f7f3ce` (P1 follow-up from Sonet50 panel `[X]`)
**Tag:** `PHASE_13_42_DF_FIX1_END`
**Status:** ✅ Closed (**981 / 0 / 1 skipped / 1 xfailed**)

### Objectives
- Close 7 production-gate bugs (B1-B7) found within 30 minutes of running real TPC ITS-TPC calibration data through Phase 13.42 v1.0 (5 P1 silently-wrong-output bugs that 5 reviewers + 27 invariance tests missed)
- Close 3 CRR §2 backlog items (D5, D8, D9) disclosed at v1.0 close
- Surface 2 `[BREACH]` disclosures (D-1, D-2) flagged under proposed Coder QRC #10
- Lock the fit interface (`fit=`, `fit_textbox_kwargs=`) before production user adoption — architect 2026-05-26: *"we will need to keep the interface for later, once we use it"*

### Correctness Fixes
- **B4:** Grouped fit reuses main-path masks (`df[group_by].eq(g)` Interval-safe, not v1.0's broken `df[df[group_by] == g]`). Sonnet55 extension: same fix prevents top_k / sort_groups divergence. New `fit_status='skipped_empty'` for empty-after-mask groups.
- **B5:** Histogram fit χ² Poisson default (`yerr = sqrt(max(counts, 1))` always; `hist_errors` flag now controls display only). Matches ROOT `TH1::Fit` Neyman convention. **D-1 [BREACH]:** required companion fix at `dispatch_fit` (`use_errors` default flipped `False → True` for hist) — not in proposal §3.1 but needed alongside the formula change.
- **D5:** Vector fit pairing per v1.4 §6.3 verbatim spec (architect 2026-05-26 ratification `R2`). `fit=['gauss','pol2']` on `[y_a,y_b]:x` now pairs gauss→y_a, pol2→y_b. F.12 assertions flipped from compound-broadcast (v1.0 deviation) to pairing.
- **D9/R4:** `stacked=True + group_by + fit` produces **N per-group fits** (architect 2026-05-26: *"fit all figures, all gb"*). Stacking is purely visual; fits remain per-group.

### Rendering Fixes
- **B1:** `facet_mode` plumbed at 3 `render_fit_textbox` call sites (Sonet51 diagnosis). **D-2 [BREACH]:** deeper second root cause — `_fit_render._style_get` called `get_style(key)` but `get_style()` takes no args → TypeError silently caught → **ALL `fit.*` style keys were silently ignored since Phase 13.42 v1.0**. Fixed to `get_style_value(key, default)`.
- **B2/B3:** New per-call kwarg `fit_textbox_kwargs={'fontsize': int, 'format': 'multiline'|'compact'|'auto', 'show_fields': List[str]}` with sub-key validation and accepted enum values **LOCKED at FIX1 close**. Compact format = one line per fit; auto = compact if facet_mode & n_blocks>1. `show_fields` filters from `{amplitude, center, sigma, slope, intercept, chi2, ndf, redchi, fit_name, x_range}`.
- **B6/B7:** Deferred to FIX2 (cosmetic; no regression test added in FIX1).

### Interface Additions (LOCKED at FIX1 close)
- `fit_textbox_kwargs` sub-keys + accepted enum values
- **D8/R3 (simplified per architect):** `scatter` honors `yerr=` column param; presence of `yerr=` is the opt-in (`use_errors=True` flag becomes redundant — no second gating signal needed)
- **D9/R4:** `stacked + group_by + fit` per-group fits (visual stack independent of fit semantics)

### Tests
- **+8** new tests under `TestPhase1342FIX1Regressions`: F.28 (B4 expression+quantile), F.28b (skipped_empty path — Sonet50 P1-A regression lock added during FIX1 panel revision), F.29 (B5 redchi ∈ [0.5, 2.5]), F.30 (B1 `set_style` round-trip), F.31 (D5 pairing), F.32 (D9 per-group dict), F.33 (`fit_textbox_kwargs.fontsize` override + precedence over `set_style`)
- F.12 inverted (compound-broadcast → pairing)
- Test count: 973 → **981** (+8)
- FIT.inline count: 27 → **35** tests
- Invariance: 299 → **307**

### Panel Verdict & Path
1. **v1.0 proposal** (Claude48 drafted; 6-reviewer panel `[!]` APPROVED WITH COMMENTS — 2 P1s I-1 (F.31 silent-pass) + I-2 (D5 nested-list scope))
2. **v1.1 proposal** (panel corrections incorporated; minor refinements I-4/I-5/I-9/I-10)
3. **v1.2 proposal** (4-reviewer second-round panel `[!]` APPROVED WITH COMMENTS — only gate arithmetic GA-1 + D9+selection_vector ADV-1 + plt.close ADV-2 carry-over; architect-driven R3/R4/R5 modifications)
4. **CRR v1** (Sonet50 [X] REVISION_REQUESTED — Sonnet55 found P1-A `np.array(shape=...)` crash; 4/4 found P1-B taxonomy not staged for the **third consecutive phase**)
5. **CRR v2** (P1-A 1-line fix `np.array → np.zeros` + F.28b regression lock; P1-B taxonomy + layer classification + matrix updated; commit `28f7f3ce`)

### [BREACH] Disclosures (proposed Coder QRC #10)
Two findings emerged during implementation that the v1.2 proposal did not anticipate. Both are 1-line fixes but both deviate from defaults the architect ratified verbatim:
- **D-1** (`fits.py:469`): `use_errors` default `False → True` for hist (needed alongside B5 Poisson formula change; proposal §3.1 specified only the formula)
- **D-2** (`_fit_render.py:25-36`): `_style_get` used broken `get_style(key)` API; all `fit.*` style keys silently ignored since Phase 13.42. Backward compat: users who tried to override and silently got defaults now actually see their overrides honored.

Both flagged in CRR §2 as `[BREACH — requires architect ratification]` per proposed QRC #10 wording (governance learning §7 of v1.2 proposal, ratified by architect as R6).

### Recurring Class Captured
**Third consecutive phase to miss `feature_taxonomy.py` + `test_layer_classification.py` staging.** Sonet50 governance note recommends adding pre-bundle taxonomy-count check to `run_tests.sh` (analogous to Phase 13.34.DF FIX2 BUG-011 staging check). Tracked in `Claude48_Feedback_to_Organization_Team_20260526.md` Item 2 for Org-team adoption.

### Cross-Cutting Process Wins
- **Production gate methodology validated:** real-data testing on TPC ITS-TPC calibration found 5 P1 silently-wrong-output bugs that 5 reviewers + 27 synthetic invariance tests missed (B4 silent failure on expression+quantile binning; B5 χ² ~7 orders of magnitude wrong by default). Synthesized into `PHASE_13_42_DF_PROD_GATE_Bugs_v1_0.md` + `PHASE_13_42_DF_POST_GATE_Audit_Questions_v1_0.md` for post-FIX1 process-improvement audit.
- **QRC #10 governance rule** (verbatim-spec deviation escalation) drafted and architect-ratified during FIX1 cycle; encoded as proposed Coder QRC v1.32 → v1.33 update; submitted to Org team feedback memo.
- **Declarative-composition architectural commitment** (architect 2026-05-26: *"Custom macros are practically not debuggable. Declarative coding is more manageable."*) — rejects refactor-to-macros path; recorded for Architectural Decision Record AD-N in Org team feedback Item 6.

### Deferred to FIX2
- B6 suptitle padding (cosmetic)
- B7 x-axis cascade visual verification on production data (likely closed by B2 compact format)
- I-8 weighted hist `sum(w²)` UserWarning at draw time
- ADV-1 D9 + `selection_vector` → explicit `NotImplementedError`
- ADV-3 `fit_textbox_kwargs` in `_HIST/PROFILE/SCATTER_FORWARDED_NAMES` (Sonnet55 P2-2)

### Statistics
- Test count: 973 → **981** (+8)
- Verified features: 47 → **50** (+3: FIT.inline promoted to 35 tests; HIST.cumulative + FACET.list_grid already verified post-Phase 13.40/13.41)
- Invariance tests: 299 → **307** (+8)
- Features: 105 → **108**

## Phase 13.42.DF FIX2: Close 5 items deferred at FIX1

**Date:** 2026-05-27
**Commit:** `79d449c3`
**Tag:** `PHASE_13_42_DF_FIX2_END`
**Status:** ✅ Closed (987 / 0 / 1 skipped / 1 xfailed)
**Predecessor:** `PHASE_13_42_DF_FIX1_END` @ `28f7f3ce` (gate 981)

### Objectives

Close the 5 items wrongly deferred at Phase 13.42.DF FIX1. See companion governance memo `Claude48_Feedback_FIX1_Defer_Anti_Pattern_20260527.md` for the process-gap analysis and the proposed QRC #11 + Reviewer supplement that arose from this defer-anti-pattern.

### Items Closed

- **B6** — suptitle padding adapts to title line count (per v1.2 §4 cosmetic backlog).
- **B7** — faceted + fit no-crash smoke verification (per v1.2 §4; cascade from B2 compact format).
- **I-8** — hist + weights + fit emits a `UserWarning` (per v1.2 §8 B5(c) commitment; weighted-hist `sum(w²)` advisory at draw time).
- **ADV-1** — stacked + `selection_vector` (>1) + fit raises `NotImplementedError` (per v1.2 §8 D9(d) commitment).
- **ADV-3** — `fit_textbox_kwargs` threaded into the 3 `_HIST/PROFILE/SCATTER_FORWARDED_NAMES` tuples + outer `DFDraw` signatures + explicit forwarding sites (per Sonnet55 v1.2 P2-2; also satisfies the Phase 13.43 v1.2 §9 carry-forward checklist).

### Tests

**Gate:** 981 → **987** / 0 / 1 skipped (+6: F.59, F.60, F.61, F.61b, F.62, F.63 in `TestPhase1342FIX2Regressions`). FIT.inline count 35 → 41. Taxonomy staged in-commit.

### Governance

The defer-anti-pattern (items B6/B7/I-8/ADV-1/ADV-3 deferred at FIX1 without explicit architect ratification of the deferral) was flagged as a recurring class. Proposed Coder QRC #11 + a Reviewer supplement memo were drafted to require explicit defer-ratification rather than silent carry-forward.

---

## Phase 13.43.DF v1.0: `summary_fit` — Standalone Fit-Result Figures

**Date:** 2026-05-27
**Commit:** `0e0d79f7`
**Tag:** `PHASE_13_43_DF_END`
**Status:** ✅ Closed (**1014 / 0 / 0 / 1 skipped**)
**Specification:** `PHASE_13_43_DF_v1_2_SummaryFit_Proposal.md` (LOCKED 2026-05-27)
**Predecessor:** `PHASE_13_42_DF_FIX2_END` @ `79d449c3` (gate 987)

### Objectives

Add `summary_fit=` to produce standalone fit-result figures (parameter tables / fit summaries rendered as their own figure) consumed at the outer `DFDraw` layer, composing with faceted dispatch.

### Implementation

- New module `plots/_summary_fit.py` (~650 LOC) plus an outer-layer consume wired through `DFDraw.{hist,profile,scatter,draw}`.
- Faceted aggregation in `_dispatch_2d_facet` and `_dispatch_faceted_render` per v1.2 §4.2.0.
- `_consumed` normalize-set extended to include `summary_fit` per §4.6 / C-3.
- 13 `summary_fit.*` keys added to `DEFAULT_STYLE` per §4.3.

**CRR §2 disclosures:** vector-dispatch `summary_fit` attaches to `stats[0]`; 3D-facet `summary_fit` deferred to a future FIX1; module-level `set_style` replaces v1.2 §3.10 instance-style language; `_make_row` extracts from the `(params, param_names, param_errors)` triplet; `_all_param_names` excludes `n_data`; title auto-fit uses a char-count approximation.

### R-2 fix at END (scalar delegation drop)

During closure, `DFDraw.draw()`'s scalar delegations were found to drop `fit` / `fit_textbox_kwargs` / `summary_fit` (named params not present in `**kwargs`) at the hist/scatter/profile sites. Fixed at all 3 sites; locked by **F.56c**. (`feature_taxonomy.py` `name`-schema fix also applied — a stale `title` key crashed the matrix generator.)

### Tests

**Gate:** 987 → **1014** / 0 / 0 / 1 skipped. +26 summary_fit invariance tests (F.34–F.56 + F.38a + F.47b) in `TestPhase1343SummaryFit` (commit-message body states the pre-R-2 count of 1013 / +26); the END gate is **1014** after the R-2 F.56c regression lock. New feature `FIT.summary` (Verified). Invariance 313 → 340. Taxonomy staged in-commit.

---

## Tooling: `run_tests.sh` PHASE_HISTORY ↔ git-tag drift check

**Date:** 2026-05-28
**Commit:** `02510a20`
**Status:** ✅ Committed (tooling-only; no test-count change — gate 1014)
**Tag:** `PHASE_13_46_DF_BEGIN` placed here (predecessor marker for Phase 13.46)

### Objectives

Catch the doc-vs-repo drift class where `PHASE_*_END` entries claimed in `docs/PHASE_HISTORY.md` do not exist as git tags (or vice-versa) — the same disease as stale-backlog tracking. Analogous to the Phase 13.34.DF FIX2 BUG-011 staging check, applied to phase-tag traceability.

### Implementation

~96-line block inserted in `run_tests.sh` after the BUG-011 staging check. Two behaviors:
1. **BLOCK (exit 1):** any `PHASE_*_END` grep'd from `docs/PHASE_HISTORY.md` that is not in `git tag --list 'PHASE_*_END'`. Override: `DFDRAW_SKIP_TAG_DRIFT_CHECK=1`.
2. **WARN (heuristic):** a `FIX<N>_END` tag whose tagged-commit subject mentions a different `FIX<M>` (misplaced-tag detector).

The reverse direction (repo tags ahead of the doc, e.g. a freshly-tagged phase not yet backfilled) is intentionally NOT blocked — it is the expected transient during a backfill pass. Two-way tested (clean → exit 0; injected fake `PHASE_99_DF_END` → exit 1); `bash -n` clean. Sonet50 panel `[!]` APPROVED.

### Incident found by the new check

The drift `diff` immediately surfaced a Phase 13.25 tag incident: `PHASE_13_25_DF_FIX2_END` was claimed in the doc at `da8895e2` but did not exist as a tag (created), and `PHASE_13_25_DF_FIX1_END` was misplaced ON `da8895e2` (the FIX2 commit) instead of the real FIX1 commit `06f84ff8` (deleted + recreated on `06f84ff8`; verified `06f84ff8` is an ancestor of `da8895e2`). All-local, no remote rewrite. **Lesson: phase open/closed status is determined SOLELY by git tags, never by memory or backlog notes.**

---

## Phase 13.46.DF v1.0: Audit Bucket ① Fixes (C-1 / C-2 / C-4 / C-7 / C-9)

**Date:** 2026-05-28
**Commit:** `1d77702e`
**Status:** ✅ Implementation closed (superseded by FIX1; closure tag is `PHASE_13_46_DF_FIX1_END`)
**Specification:** `PHASE_13_46_DF_v1_3_AuditFixes_Proposal.md` (panel-approved)
**Predecessor:** `PHASE_13_43_DF_END` @ `0e0d79f7` (gate 1014)
**Source:** `PHASE_13_45_dfdraw_Audit_Findings.md` (audit defining the bucket-① items)

### Objectives

Close audit bucket ① — five independent fixes surfaced by the Phase 13.45 audit:

- **C-1** — ROOT TF1 alias `fit='gaus'`: `register_fit('gaus', _gaussian, _gaussian_guess)` in `plots/fits.py`. Lock F.64.
- **C-2** — ROOT type alias `type='histo'`: module-level `_TYPE_ALIASES = {'histo':'hist'}` applied before the dispatch ladder in `drawer.py`. Lock F.65.
- **C-4** — source `_get_suptitle(fig)` helper (public `get_suptitle()` for mpl ≥ 3.8 + private fallback); replaced all 9 inline `fig._suptitle.get_text()` sites. Retires Claude's own Phase 13.42 FIX2 §2.2 private-access disclosure. (The audit premise was partly wrong — the helper existed only as a test helper, not in the code path.) Lock F.70.
- **C-7** — kwarg-typo guard at `DFDraw.draw()` entry: `difflib.get_close_matches(cutoff=0.8)` did-you-mean; the known-key set K = union of all 6 method signatures (draw/hist/scatter/profile/hist2d/hexbin) ∪ the 5 `_*_FORWARDED_NAMES` tuples (reviewer note N-1). Near-miss → raise with suggestion; far-unknown → warn. Lock F.66.
- **C-9** — `range=` on `scatter()` resolved through the shared `resolve_range_2d` (AD-74 per-axis), identical handling to hist/profile/2D. v1.0 applied it as a **view clip** (`set_xlim`/`set_ylim`); non-faceted exact, all strategies, honest stats; original profile/hist unpack bug fixed. Locks F.67/F.68/F.69a/F.69b. (Superseded by FIX1 point-filtering — see below.)

**Excluded:** C-3 (faceted `auto_title=False`) — intentional per `drawer.py` comment ("faceted plots share one selection → show once, not per-cell") → deferred to Phase 13.47.

### §2.1 architect ruling (faceted scatter `range=`)

Panel-decided **Option 1 (shared-global):** faceted scatter `range=` applies at the shared-axis / global level (consistent with faceted hist), NOT per-cell, because facet grids use matplotlib shared axes. Per-cell ranges remain available to users via `facet_by=[list] + share_x='none'`. Per-cell tightened-strategy windows and the `facet_by='string'` + `share_x='none'` gap were recorded as FIX1 candidates.

### Tests

**Gate:** 1014 → **1022** / 0 / 0 / 1 skipped. +8 invariance tests F.64–F.70 (F.69 split a/b) in `TestPhase1346AuditFixes`. +4 features (`FIT.root_aliases`, `API.kwarg_typo_guard`, `RANGE.scatter`, `TITLE.get_suptitle`). Invariance 340 → 348; Verified 51 → 55. Taxonomy staged in-commit. Panel: Sonet50 5-reviewer `[!]` APPROVED.

---

## Phase 13.46.DF FIX1: Scatter `range=` Removes Out-of-Range Points

**Date:** 2026-05-28
**Commit:** `ad91e251`
**Tag:** `PHASE_13_46_DF_FIX1_END`
**Rolling tag:** `PHASE_BEGIN_dfdraw` → `ad91e251`
**Status:** ✅ Closed (**1023 / 0 / 0 / 1 skipped**)
**Predecessor:** Phase 13.46.DF v1.0 @ `1d77702e` (gate 1022)
**Specification:** `PHASE_13_46_DF_FIX1_Code_Review_Request.md`

### Trigger

v1.0 set the **view window** (`set_xlim`/`set_ylim`) — out-of-range points stayed in the collection, off-screen. Architect 2026-05-28: scatter `range=` must **DROP** the points (a point filter), consistent with how hist/profile `range=` exclude points from binning. Without filtering, `percentile_99`/`hybrid` strategies are cosmetic; with filtering they do their job.

### Implementation (`plots/scatter.py`, ~30 LOC net)

- **Point filter (before stats + plotting):** resolve range via `resolve_range_2d`; build an in-range mask `_rmask` over `x_data`/`y_data`; filter `x_data`, `y_data`, AND `df_filtered` by the **same** mask. Alignment is automatic — color/size/marker/error-bar helpers all derive from `df_filtered`, so filtering it keeps every parallel array aligned (the helpers' `mask` param is a defensive no-op when `df` is pre-filtered).
- **Stats honesty:** computed after the filter → point counts reflect in-range data; `autorange_used` / `autorange_strategy` record the resolved window + strategy.
- **View (non-facet only):** tight exact `set_xlim`/`set_ylim` to the resolved window so F.67's exact-view assertion holds; faceted skips it (shared axes autoscale to the union of filtered data — no last-cell-wins). The v1.0 re-resolve block is removed.

### Behavior

`range="minmax"` → window == full data → nothing removed (no-op, view unchanged). `range="percentile_99"` / explicit tuple → out-of-range points dropped + tight exact view (non-facet). Faceted: each cell filters and the shared axes autoscale to the union. The v1.0 §2.1 shared-global disclosure becomes largely moot — the points are gone, so the shared axes reflect filtered data.

### Tests

**Gate:** 1022 → **1023** / 0 / 0 / 1 skipped. +1 invariance test **F.71** (`test_f71_scatter_range_removes_out_of_range_points`: percentile_99 & explicit-tuple drop points; minmax removes nothing; parallel color array stays aligned). +1 feature `RANGE.scatter_filter`. Invariance 348 → 349; Verified 55 → 56. Taxonomy staged in-commit.

---

## run_tests.sh: Tag-Drift Check Downgraded to Non-Blocking WARNING

**Date:** 2026-05-28
**Commit:** `df3057a3`
**Status:** ✅ Closed (tooling-only; no test count / feature / invariance change)
**Predecessor:** Phase 13.46.DF FIX1 @ `ad91e251` (gate 1023)

### Trigger

The tag-drift check added in v1.10's `02510a20` (Phase 13.43 / `PHASE_13_46_DF_BEGIN` window) hard-blocked the reviewer bundle on a heuristic `grep` over `PHASE_HISTORY.md` prose. Two failure modes surfaced in practice:

- **False positives:** the heuristic matched prose mentions of phase numbers (e.g., "Phase 13.42 is referenced in Phase 13.43") as tag declarations, so docs that were correct still failed the check.
- **Override pressure:** when the heuristic falsely fired, operators had no good remediation path except to bypass the check invisibly. A check that gets bypassed silently degrades into dead-weight that no longer protects against the original drift class.

### Implementation (tooling-only)

- **Scope narrowed:** the `grep` now matches only declarative `` 'tag `PHASE_X_END`' `` references, which is the actual fingerprint of an undocumented tag. Prose mentions of phase numbers no longer trigger.
- **Severity demoted:** drift no longer blocks bundle creation. Instead, drift is recorded as a **non-fatal WARNING** in the `SUMMARY` artifact that travels in `reviewer.zip` — drift stays visible to every panel reviewer, the bundle always builds, and the operator cannot silently bypass.

Lands as a standalone tooling commit between Phase 13.46.DF FIX1 and Phase 13.48.DF v1.0. Parallels the earlier `02510a20` event (also tooling-only, also recorded as its own Statistics Summary row in v1.10).

### Testing

- **+0** new tests (tooling-only)
- Test count: 1023 → **1023** (unchanged)
- Features unchanged at 114; Verified unchanged at 56; invariance unchanged at 349

### Lesson Recorded

**Heuristic gate-blocks need a non-blocking off-ramp.** When a gate check is heuristic (rather than mechanical), false-positive cost compounds with each silent bypass. The fix template that emerged here: tighten the scope to a declarative fingerprint where possible, AND demote severity to a visible WARNING when the operator can't otherwise act on the false positive. Both are necessary; tightening alone leaves the override-pressure failure mode, demotion alone leaves the noise.

---

## Phase 13.48.DF v1.0: Tier-1 Automated Visual Testing

**Date:** 2026-05-28
**Commit:** `9f612601`
**Tag:** `PHASE_13_48_DF_END`
**Status:** ✅ Closed (**1034 / 0 / 0 / 1 skipped / 1 xfailed**)
**Predecessor:** `run_tests.sh` WARN downgrade @ `df3057a3` (gate 1023, tooling-only); architectural predecessor Phase 13.46.DF FIX1 @ `ad91e251` (gate 1023)
**Specification:** `PHASE_13_48_DF_v1_4_VisualTesting_Proposal.md` (4-reviewer panel `[!]` APPROVED WITH COMMENTS through v1.4)

### Objectives

Open a new test layer (`visual_primitive`) for renderer-free, deterministic, backend-independent visual assertions over already-rendered matplotlib figures. Phase 13.48 ships the framework + 10 V-checks (V.1–V.10) + V.2 ragged-padding-safety lock, deliberately scoped narrow to validate the framework before broader catalog work. The motivating gap: the existing test suite asserts on `stats` dicts and component shapes but cannot catch a class of bugs where the dispatcher produces structurally correct stats but the matplotlib figure itself is visually wrong (missing series, wrong cell, color collision, hidden cell counted as visible).

### Implementation

**New test module:** `tests/test_phase_13_48_df_visual_testing.py` (test-only; no library source change).

**Framework — `VisualCheck(fig, stats, df)`:**
- Collect-all-then-assert pattern: all checks accumulate; final assertion produces a structured failure report listing every assertion that failed
- Cell iteration via `fig.axes` (dispatcher-agnostic) rather than via `stats[(row,col)]` keying (which depends on dispatcher internals)
- `visible_cell_axes` helper excludes hidden padding cells (the n_groups < nrows·ncols case)
- Series counts via `ax.containers`, not `len(ax.lines)` — the errorbar-cap trap: errorbar caps populate `ax.lines` but aren't series
- Distinct-RGBA distinctness for color-coded series
- Cell-to-value via `zip(sorted_unique(...))` (dtype-safe across int / str / pd.Interval categorical types)

**Backend hygiene:** `matplotlib.use("Agg")` at module import; checks operate on persistent in-memory figure state, no renderer draw cycle required.

**V-checks shipped (V.1–V.10 + V.2 ragged-padding-safety lock = 11 visual_primitive tests under `TestPhase1348VisualPrimitive`).** Six new `VISUAL.*` features (`VISUAL.framework`, `VISUAL.cell_iteration`, `VISUAL.series_count`, `VISUAL.distinct_colors`, `VISUAL.shared_axes`, `VISUAL.layout_visibility`) claim them.

**Taxonomy posture (architect direction pre-merge):** the new `visual_primitive` layer ships with all 6 new `VISUAL.*` features marked **Smoke-only** rather than Verified. The Phase 13.48 spec explicitly defers the matrix Verified-promotion mechanism (the orthogonal Visual column + 👁 badge) to the matrix-traceability work that becomes Phase 13.49 — the visual_primitive layer is dedicated-status pending that infrastructure.

### Testing

- **+11** visual_primitive tests (V.1–V.10 + V.2 padding-safety lock)
- Test count: 1023 → **1034**
- +6 features (`VISUAL.*` set above) — all Smoke-only at close
- **visual_primitive layer NEW:** 0 → 11
- Verified count: 56 unchanged (V-checks are visual_primitive layer, not invariance → no Verified promotion at 13.48 close)
- Invariance count: 349 unchanged

### Panel Verdict

Spec v1.4: 4-reviewer panel `[!]` APPROVED WITH COMMENTS (Sonet51 closed at v1.4 with one-line `ax.get_visible()` filter folded into CRR §2). CRR v1.0: 5-reviewer panel — Opus2 `[!]` (one P2 hardening advisory: `check_counts` should zip `visible_cell_axes` before extended-graphics on real data; currently safe under N-2 precondition); remaining reviewers `[OK]`/`[!]`. Architect approval at commit.

### Architectural Posture

**Tier 1 (renderer-free) vs Tier 2 (renderer-driven) split established here.** Tier 1 reads stored figure state (`Text.get_text()` returns input string not rendered glyph; `Patch.get_facecolor()` returns RGBA tuple; `Table._cells` is a dict of cells with text). Tier 2 requires `fig.canvas.get_renderer()` and a draw cycle for `get_window_extent()`-class assertions (text-vs-data overlap, label clipping). Tier 2 is deferred to a future framework increment (eventually scoped as Phase 13.5X with F19 textbox-bbox-overlap as the architect-flagged spacing-bug proof — see Phase 13.50 §Architectural Posture).

---

## Phase 13.49.DF v1.0: Capability Matrix Traceability (Link + HTML + Visual)

**Date:** 2026-05-29
**Commit:** `89bc63c6`
**Tag:** `PHASE_13_49_DF_END`
**Status:** ✅ Closed (**1038 / 0 / 0 / 1 skipped / 1 xfailed**)
**Predecessor:** Phase 13.48.DF v1.0 @ `9f612601` (gate 1034)
**Specification:** `PHASE_13_49_DF_v1_2_CapabilityMatrixTraceability_Proposal.md` (Sonet50 v1.2 `[!]` APPROVED; Sonet51 CRR v1.1 `[!]` APPROVED 8/8)

### Objectives

The capability matrix at Phase 13.48 close listed 120 features but provided no traceability from a feature to the specific test(s) that prove it (no "Verified" claim could be followed to its proof). The architect's challenge: a reviewer must be able to navigate matrix entry → test list → file:lineno → assertion. Phase 13.49 ships the link infrastructure, the HTML rendering, the orthogonal Visual column with 👁 badge, the 4 M-tests (M.1–M.4) that enforce taxonomy integrity, and `KNOWN_UNCLAIMED` governance for legacy unclaimed tests.

### Implementation

**Spec changes (test / tooling only):**
- `tests/feature_taxonomy.py`: each feature row gains a `tests: [List[str]]` field (explicit test-method names per feature; dfdraw-canonical, more powerful than ADF's `test_patterns` which match by regex — Marian's plan: unify dfdraw-canonical first, port ADF later)
- `tests/test_layer_classification.py`: `TEST_LAYERS` dict maps each test ID to its layer (`visual_primitive` | `invariance` | `proof` | `meta`)
- `scripts/generate_capability_matrix.py`: extended to compute per-feature Verified/Smoke-only/Broken/Planned status from `test_results` + emit both MD and HTML; **new** orthogonal `Visual` column with 👁 badge for features with at least one passing visual_primitive test (orthogonal to status — a Smoke-only feature with a 👁 is still Smoke-only)
- `docs/CAPABILITY_MATRIX.html`: per-feature expandable tests-panel; status × visual × category filters AND-combined; expansion-state preserved across filter changes

**M-tests (4 new meta-tests, invariance layer):**
- **M.1** (`test_taxonomy_tests_resolve`): every `FEATURES[*].tests` entry resolves to a real pytest node ID at collection time. No-grandfathering: 4 dangling test refs from prior phases repaired in this commit.
- **M.2** (`test_classification_coverage`): every test in `TEST_LAYERS` is claimed by at least one feature (no orphan tests).
- **M.3** (`test_html_matrix_locks`): HTML generator output structural locks (category-row presence, data-status attribute integrity). Extended in Phase 13.49 FIX1 with H-1/H-3 invariants.
- **M.4** (`test_no_orphan_visual_tests`): every visual_primitive test is claimed by a feature.

**KNOWN_UNCLAIMED governance (`tests/test_meta_capability_matrix.py`):**
- Spec §3.7: location-fixed file holds a list of pytest node IDs explicitly waived from M.2/M.4 enforcement.
- Required fields per entry: `test_id` + `reason` (≥10 chars) + `target_phase` (the phase that will claim the test).
- Seeded at phase open with **64 entries** distributed across source phases per the actual gap: Phase 13.27.DF=50, Phase 13.28.DF=9, Phase 13.32.DF=5 (per Opus48_1 P2-A advisory adopted at this phase: use SPECIFIC target phases not blanket "OPEN").
- Growth in `KNOWN_UNCLAIMED` raises a warning; missing fields fail the gate. HTML audit hook lists every entry with its target phase for reviewer visibility.

**run_tests.sh:** `--phase` derivation from latest `_END` tag (corrected glob `PHASE_[0-9]*_DF*_END` — the v1.2 spec text had a broken middle-underscore glob; fix verified against the real tag set). Phase 13.48 prose-grep tightening + BLOCK→WARN for the taxonomy-uncommitted check (bundled).

**Source-defect normalization fixes (D-K, D-L per Opus2 + Sonnet53_R2 panel finding on v1.0 [X] rejection):**
- v1.0 CRR was **REJECTED by Opus2** with 3 P0s — most consequential was P0-1: `load_test_results` normalization regression (`if startswith("tests/"):` skip; but real node-IDs are `dfextensions/dfdraw/tests/...` when pytest rootdir is the O2DPG repo). Matched 0 features → ALL 121 features showed Planned vs prior 56 Verified.
- v1.1 (resubmission, accepted `[!]`): D-K + D-L fixed at 3 sites via robust `basename = path.split('/')[-1]` normalization (`load_test_results`, `load_collected_tests`, `_collected_test_ids`). **D-L was self-discovered** — Claude48 found the same pattern in `_collected_test_ids` while fixing D-K and disclosed it under proposed Coder QRC R7 self-discovery wording.

### Testing

- **+4** invariance tests M.1–M.4 in `tests/test_meta_capability_matrix.py`
- Test count: 1034 → **1038**
- +1 feature `META.capability_matrix` claiming the 4 M-tests
- Features: 120 → 121
- Verified: 56 → 57 (META.capability_matrix has 4 passing invariance tests → Verified)
- Invariance: 349 → 353
- visual_primitive count unchanged at 11 (Phase 13.48's V-checks); the 6 `VISUAL.*` features remain Smoke-only at this phase close (no new invariance tests for them)

### Panel Verdict & Path

1. **v1.0 proposal** (Sonet50 + 8-reviewer panel `[!]` APPROVED through v1.2 with `tests: [explicit list]` per-feature locked as the link mechanism)
2. **CRR v1.0** (Opus2 `[X]` REJECTED — 3 P0s: P0-1 normalization regression; P0-2 missing meta-test file `tests/test_meta_capability_matrix.py`; P0-3 gate reported at 1034 not 1038)
3. **CRR v1.1** (Claude48 fixed all 3 P0s including D-L self-discovery; 5-reviewer panel `[!]` APPROVED; one missing D-M target_phase deviation disclosure noted as P2)

### Recurring Class Captured

**Normalization regression class:** pytest node-ID prefix assumptions break when running from a higher rootdir (O2DPG repo vs dfdraw subdir). Confirmed across three sites in Phase 13.49 v1.0 → v1.1, then again as a pattern in Phase 13.50 proposal-review work where one reviewer's confused source-grep produced a cascade error through subsequent revisions until source-grepping reviewer broke the chain. Corrective discipline: always source-verify line refs and test paths against the actual file, not against working memory or another reviewer's claim.

---

## Phase 13.49.DF FIX1: HTML Rendering Fixes (H-1 / H-2 / H-3) + M.3 Lock

**Date:** 2026-05-30
**Commit:** `a6ddc753`
**Tag:** `PHASE_13_49_DF_FIX1_END`
**Status:** ✅ Closed (**1038 / 0 / 0 / 1 skipped / 1 xfailed** — same-test stricter, no test count change)
**Predecessor:** Phase 13.49.DF v1.0 @ `89bc63c6` (gate 1038)
**Specification:** `PHASE_13_49_DF_FIX1_Code_Review_Request.md`

### Trigger

After v1.0 commit, the architect rendered `docs/CAPABILITY_MATRIX.html` in a browser and found three bugs the 8-reviewer v1.1 panel had not caught:
- **H-1:** `META.capability_matrix` showed `Broken` in HTML but `Verified` in MD — content divergence
- **H-2:** clicking a feature `<tr>` row hijacked filter state — the JS selector `[data-status]` was too broad and matched both filter buttons AND feature rows (both carry `data-status` for `applyFilters` correctness)
- **H-3:** 15 duplicate category headers (FACET appearing 5×, PROFILE 4×, HIST 4×) — `FEATURES` is in chronological commit order; the emitter wrote a per-transition header without first sorting by category

The meta-lesson: the v1.1 panel approved without rendering the HTML in a browser. Same failure class as Phase 13.45 audit PR-3 at the meta level. **Reviewer QRC v1.31 Rule 14 caveat** (*"diff-read does not discharge visual-render verification; Q6 trigger applies independently; both required"*) was drafted during this round in direct response.

### Implementation

- **H-2:** scope the JS filter selectors to `.filter-group [data-status]` and `.filter-group [data-visual]` (8 occurrences updated in `generate_capability_matrix.py`). The broad selector previously matched the 121 feature `<tr>` rows; clicking a row hijacked filter state.
- **H-3:** sort `FEATURES` by category (stable) before emit, in BOTH MD and HTML paths. Alphabetic regrouping in the diff confirms the fix.
- **H-1:** could not reproduce in the container post-D-L. Same `test_results` + same `compute_feature_stats` produces `Verified` in both MD and HTML outputs. Hypothesis: the committed HTML was a **stale artifact** (generated pre-D-L when M.1 was failing → META showed Broken at that snapshot, then never regenerated). Resolved by regeneration in this commit; locked against future stale-or-divergent recurrence by the M.3 extension below.
- **M.3 extension (same test, stricter assertions — no test count change):** locks two invariants on the HTML output:
  - (a) each category appears as a `category-row` header exactly once (catches H-3 class)
  - (b) HTML `data-status` counts agree with MD-computed status counts on the same `test_results` (catches H-1 class whether stale-artifact or divergent-compute)

  Both fail on the original buggy emitter; pass on this commit's. Gate unchanged at 1038.

### Testing

- **+0** new tests (M.3 same-test stricter)
- Test count: 1038 → **1038**
- Features unchanged at 121
- Verified unchanged at 57
- Invariance unchanged at 353

### Panel Verdict

Sonet51 `[!]` APPROVED WITH COMMENTS consolidating 5 reviewers (Sonnet54, Sonnet55, Sonnet52_R1, Opus2, Sonet51). Two P2 FIX2 candidates noted: P2-1 HTML still missing from reviewer zip (**first** recorded occurrence of what became a 3× recurring packaging gap, closed in Phase 13.50 FIX2 below); P2-2 expand-after-filter JS bug — `removeAttribute('hidden')` doesn't clear `style.display = 'none'` set by filter — Sonnet52_R1 found via actual click testing in their browser.

### Process Lesson

**"Render the artifact; don't trust the diff."** The Reviewer QRC v1.31 Rule 14 caveat — that diff-reading does not discharge visual-render verification when the artifact is itself a rendered output — landed in this round as a direct response to the H-1/H-2/H-3 miss. First operational success of the rule: in the FIX1 review round, 3 of 5 reviewers rendered the HTML, and Sonnet52_R1's hand-click test surfaced P2-2 (expand-after-filter behavior bug) — exactly the class Rule 14 exists to catch. Rule worked once applied.

---

## Phase 13.50.DF: Fit-Rendering Overhaul + V-Test Coverage (Proposal v2.5)

**Date:** 2026-06-03
**Commit:** `f3034153`
**Tag:** `PHASE_13_50_DF_END`
**Status:** ✅ Closed (**1057 / 0 / 0 / 1 skipped / 1 xfailed**)
**Predecessor:** Phase 13.49.DF FIX1 @ `a6ddc753` (gate 1038)
**Specification:** `PHASE_13_50_DF_v2_5_FitRenderingOverhaul_Proposal.md` (architect-approved after 4-round panel convergence v2.2 → v2.3 → v2.4 → v2.5)
**Intermediate step commits (preserved on `feature/groupby-optimization`):** step 1 `c029b405`, step 2 `e904ccec`, step 3 `181c2366`, step 4 `ccd8243d`, step 5 `068da47f`, step 7 `f3034153` (closing — covers sub-steps 7a–7f + folded step-7 FIX1).

### Trigger

92 production `.draw()` call sites surveyed across `time_series.py`, `time_series_TroubleShooting.py`, and `makeSmoothMapsWithTPC.py` showed five UX gaps in fit rendering: long parameter names rendered verbatim, fixed `.4g` precision (errors not 1-sig-fig physics-convention), no shared-legend mode, no per-panel or pad placement for `summary_fit` tables, and no orientation control. Plus the architect-flagged motivating bug — *"problems only with spacing of the fits in the latest test"* (2026-05-30) — which is the textbox-bbox-overlap class. **Phase 13.50 ships the API surface to fix the spacing class**; the test that proves it's fixed (F19, Tier 2 renderer-driven) is deferred to Phase 13.5X. F1–F18 green is necessary but not sufficient evidence — framed explicitly in CRR §1 / §6 risks.

### Implementation (7 sub-steps; closing step is single commit)

- **Step 1 — display-name map** (`plots/_fit_render.py`): new `_DISPLAY_NAMES` render-only dict (`slope→p1`, `intercept→p0` matching `_polynomial_factory` ascending-powers convention `c0 + c1·x + c2·x² + ...`; `center→$\mu$`, `sigma→$\sigma$`, `decay→$\tau$`, `amplitude→A`). `plots/fits.py` UNCHANGED — render-time substitution only. F1, F2, F3 lock textbox content via `Text.get_text()` (Tier 1 sees stored string, not rendered glyph).
- **Step 2 — precision keys** (`style.py`, `plots/_fit_render.py`): **[BREACH]** `fit.text_format` style key REMOVED (v2.5 §5 + §3.3 + §9 authorize). Replaced by `fit.value_format` (default `'.2g'`), `fit.error_format` (default `'.1g'`), `fit.precision_mode` (`None | 'physics' | 'uniform'`). `precision_mode='physics'`: error → 1 sig fig (standard convention), value's decimal aligned to error's place. Edge cases (error≤0, NaN, ±inf) fall back to `value_format`. F4, F5 locks.
- **Step 3 — `fit_textbox_kwargs` extensions** (`plots/_fit_render.py`): `_allowed_sub_keys` extended with `rename_params`, `value_format`, `error_format`, `precision_mode` (all per-call overrides of style defaults). Additive — existing keys (`fontsize`, `format`, `show_fields`) unchanged. F6, F16 locks.
- **Step 4 — polymorphic `legend=` kwarg** (new `plots/_legend.py` + `drawer.py` hook): `legend=` accepts `bool | str | dict | None` polymorphically (`True`/`False`/`'shared'`/`'first'`/dict). `show_legend=` retained as **permanent back-compat parallel** (NOT deprecated; no `DeprecationWarning`; no removal scheduled). Both reach `_normalize_legend_spec(spec) → canonical dict`. **Pattern A for both** (popped at top-level dispatcher entry, never in `*_FORWARDED_NAMES` tuples — same discipline as Phase 13.43 §9.1 for `summary_fit=`). `show_legend=` newly added as named param in `draw()` signature (previously absent — would have tripped the Phase 13.46 C-7 kwarg-typo guard). `_apply_legend_mode` (~50 LOC) hook in `drawer.py` after `_dispatch_faceted_render`: `mode='shared'` consolidates per-panel legends to a single `fig.legend()`; `mode='first'` keeps only `axes.flat[0]`. F7, F8, F9, F10, F18, `test_normalize_legend_spec_idempotent` invariance test.
- **Step 5 — `summary_fit.placement` axis + GridSpec pre-planning** (`plots/_summary_fit.py`, `drawer.py`): three modes — `'figure'` (default, preserves Phase 13.43 behavior; separate `Figure` in `stats['summary_fit']`), `'subfigure'` (per-panel `Axes.inset_axes()` per visible facet panel — **each inset shows only that panel's fits**, per Claude36 ADF panel finding v2.4 P2-NEW-3), `'pad'` (new GridSpec cell in same `Figure`, **pre-planned at top of `_dispatch_faceted_render` BEFORE `plt.subplots()`** because `gridspec.GridSpec` is immutable post-construction, per Sonnet52_R1 v2.3 panel finding P1-A). Sub-keys: `pad_location` (4 cardinal edges), `pad_size` (fraction of figure dim; non-uniform sizes use proportional `width_ratios`/`height_ratios` per P3-NEW-1), `inset_bbox` (per-panel inset positioning). `ax` return contract: always main-plot axes array; pad axes reachable only via `stats['summary_fit']` or `fig.axes`. F11, F12, F13, F17 locks.
- **Step 6 → renamed in step 7b — `summary_fit.orientation` axis** (`plots/_summary_fit.py`): two modes — `'row'` (default; one row per fit, columns are params) and `'column'` (transposed; one column per fit, rows are params). Step 6 initial ship used `'horizontal'`/`'vertical'` tokens; step 7b spec-conformance renamed to `'row'`/`'column'` per v2.5 §3.4(b) canonical. Step 7b R1 extended orientation honoring to the `placement='figure'` renderer (`_render_table_figure` transposes when `orientation='column'`) — step 6 had been an undisclosed partial implementation (in-slot renderer only). F14, F15 locks.
- **Step 7 — closing step, single commit, covers sub-steps 7a–7f + folded step-7 FIX1:**
  - **7a [BREACH]** `summary_fit.precision` int field REMOVED (v2.5 §5 + §3.3 + §9). Replaced by `value_format` / `error_format` format-spec strings mirroring step 2's `fit.*` style keys. **11 sites** updated in `plots/_summary_fit.py` (8 behavioral: signature, allowed-keys, validation, default dict, pass-throughs; + 3 documentation: docstring example, error message body, downstream helper signature — sweep coverage per Opus2 v2.3 P2-B). `tests/test_phase_13_43.py::test_f43_precision_in_table_cells` migrated.
  - **7b** orientation token rename (above) + R1 figure-renderer extension.
  - **7c** pad sub-keys (`pad_location`, `pad_size`, `inset_bbox`) — normalizer + drawer.py pad allocator with proportional GridSpec ratios.
  - **7d** subfigure semantic flip from single `fig.add_subfigure()` (step 5 ship) to per-panel `Axes.inset_axes()` (v2.5 §3.5 + P2-NEW-3 spec). `render_summary_fit_per_panel_insets()` new in `_summary_fit.py`; dispatch loop stashes `_dfdraw_facet_key` on each axes; `_maybe_attach_summary_fit` split-routes `'pad'` vs `'subfigure'`. F12 test body rewritten. `stats['summary_fit']` shape change for subfigure: now `{'insets': List[Axes], 'per_panel_keyed': Dict[facet_key → Axes], 'placement': 'subfigure'}` (was `{'table': Axes, 'placement': 'subfigure'}`).
  - **7e** `table_cells_keyed(ax) → set[frozenset[(col_label, cell_text)]]` helper (collision-safe + set-union-friendly for subfigure mode's per-panel slices, per Claude36 v2.4 P2-NEW-1 keyed-comparison upgrade from v2.3's permutation-blind `set[str]`). **F17 test body rewritten** from step-5's topology-distinctness (the opposite invariant) to v2.5-spec'd cross-variant table-content **equivalence**: same draw via `placement='figure'`, `'pad'`, `'subfigure'` produces identical keyed cell dicts (subfigure uses set-union across per-panel insets).
  - **7f hist2d `summary_fit=` forwarding** (`drawer.py:~6155` call site; new `summary_fit: Optional[Union[str, List[str], Dict]] = None` named param on `DFDraw.hist2d()`): v2.5 §2 IN-scope listed `summary_fit.placement` with no plot-kind exclusion, but the step-5 ship omitted forwarding at the hist2d dispatcher call site (kwarg was silently dropped). R4 panel feedback called this out as a should-have-been-in-scope miss; promoted into Phase 13.50 scope. Pattern A verified: `summary_fit` NOT in `_HIST2D_FORWARDED_NAMES`.
  - **Folded step-7 FIX1 cycle (post-step-7 deploy, 3 bugs surfaced and fixed in the same commit):**
    1. **`_select_panel_stats` type mismatch:** dispatcher stringifies `group_value` at `drawer.py:3784` (`all_stats[str(group_value)] = ...`); dispatch loop stashed raw value on `ax._dfdraw_facet_key`. Membership check `3 in ('3',)` is False → every panel resolved to None → F12 returned empty insets. Fixed by accepting raw + stringified scalar + tuple-of-stringified-element forms.
    2. **`_render_table_in_axes` pre-existing column-selection bug:** looked for `row.get('params')` as a nested sub-dict that `_make_row` never produces (params flattened to top-level row keys). Pad-placement table dropped all params/metrics columns → F17 cross-variant equivalence broke. Fixed by mirroring `_render_table_figure` exactly: `_default_columns(rows)` + `_format_cell` with paired error formatting.
    3. **Registry split-deploy:** `tests/test_layer_classification.py` was missed in first FIX1 deploy (renames synced only in `feature_taxonomy.py`). Asymmetric meta-failures (`test_taxonomy_tests_resolve` passing + `test_classification_coverage` + `test_no_orphan_visual_tests` failing) is the diagnostic fingerprint — codified in CRR §9 QRC backlog item 4.

### Testing

- **+19** new tests under `tests/test_phase_13_50_df_fit_visual.py` (new file with `FitVisualCheck` sibling-of-`VisualCheck` class):
  - **+16 visual_primitive** (F1–F16)
  - **+3 invariance** (F17 cross-variant equivalence; F18 `show_legend`/`legend` behavioral equivalence via 4-tuple `legend_topology = (n_fig_level, n_per_axes, frozenset(labels), loc_string)` per Claude36 v2.4 P2-NEW-2; `test_normalize_legend_spec_idempotent`)
- Test count: 1038 → **1057** (+19)
- Features: 121 → 127 (+6: `FIT.display_names`, `FIT.precision_modes`, `FIT.textbox_kwargs_extensions`, `LEGEND.modes`, `SUMMARY_FIT.placement`, `SUMMARY_FIT.orientation`)
- visual_primitive layer: 11 → 27 (+16)
- Invariance: 353 → 356 (+3)
- Verified: 57 → 59 (+2 — `LEGEND.modes` and `SUMMARY_FIT.placement` have invariance tests via F18 and F17 → Verified; other 4 new features remain Smoke-only at close)
- **Tier 2 deferred (NOT in gate): F19** `check_textbox_does_not_overlap_data_bbox` — the test that proves the architect-flagged motivating spacing bug is cured. Requires `fig.canvas.get_renderer()` draw cycle; lands in Phase 13.5X with the Tier 2 framework.

### [BREACH] Disclosures (Coder QRC v1.29 R14)

Two architect-authorized removals (v2.5 §5 + §3.3 + §9 explicitly authorize; no aliases; clean cut):
- **§2.1** — `fit.text_format` style key REMOVED. Migration: `set_style({'fit.text_format': X})` → two-key form `set_style({'fit.value_format': '.2g', 'fit.error_format': '.1g'})`. Old style sheets that set `text_format` raise `KeyError` at style-load time.
- **§2.2** — `summary_fit.precision` int field REMOVED. Migration: `summary_fit={'precision': N}` → `summary_fit={'value_format': '.Ng', 'error_format': '.Ng'}`. 11 sites in `_summary_fit.py` cleared (8 behavioral + 3 documentation, all enumerated in CRR §5).

Two narrow-window non-[BREACH] disclosures:
- **§2.3** — `summary_fit.orientation` token rename `'horizontal'`/`'vertical'` → `'row'`/`'column'` (step 7b spec-conformance; narrow window between step 6 commit and step 7 commit; not strictly [BREACH] because step 6 was not released externally per architect confirmation).
- **§2.4** — `placement='subfigure'` stats structure flip (step 7d; was `{'table': Axes, 'placement': 'subfigure'}` in step 5 initial ship; now `{'insets': List[Axes], 'per_panel_keyed': Dict[facet_key → Axes], 'placement': 'subfigure'}`).

### Panel Verdict & Path

1. **v2.2 proposal** (Claude48 drafted; Sonet51 8-reviewer panel `[!]` APPROVED WITH COMMENTS — 2 P1s + 6 P2s + 1 P3 pair; key finding P1-A GridSpec immutability from Sonnet52_R1)
2. **v2.3 proposal** (folded 9 panel findings; Opus2 `[!]` with 2 new findings: P2-A visual count arithmetic off by 2; P2-B precision-removal site count understated)
3. **v2.4 proposal** (folded 9 more findings including **Claude36 ADF cross-team's three substantive test-design upgrades**: P2-NEW-1 keyed-dict cross-variant comparison, P2-NEW-2 extended legend topology 4-tuple, P2-NEW-3 subfigure per-panel-slice semantics; Opus2 caught the cascade-error in §3.1 line ref — v2.3 P3-1 had incorrectly "corrected" `_normalize_one_fit` from `fits.py:330` to `:276`; line 276 is the DIFFERENT function `normalize_fit_spec`. Sonnet55's earlier source claim had conflated the two; v2.4 P3-1 propagated the error; Opus2's Rule-16 source-grep this round caught it)
4. **v2.5 proposal** (closed Sonnet54 P2-1 orphan-normalizer-test M.2 risk; Opus2 P2-2 line-ref cascade-error correction; 3 P3 risk notes including private mpl `_loc_real` attribute and `frozenset(labels)` deduplication; **architect-approved**)
5. **Implementation step commits 1–6** landed in earlier sessions per §10 implementation order
6. **CRR v1.0** (Claude48 introduced the **§0 pre-CRR gap-audit attestation pattern** — when architect's "Was all functionality from spec implemented?" challenge caught 4 undisclosed partials in v1.0-draft, draft was scrapped, §0 audit table cross-checked all 20 v2.5 in-scope items, second sweep confirmed conformance, THEN the CRR was drafted; Sonnet53_R2 consolidating 6-reviewer panel `[!]` APPROVED WITH COMMENTS)

### Process Wins Captured (CRR §9 QRC Backlog Additions)

1. **Pre-CRR source-vs-spec audit pattern (§0 attestation):** the architect's "Was all functionality implemented?" challenge → coder scraps draft → §0 audit table written → second sweep confirms zero remaining partials → THEN the CRR is drafted. Proposed adoption as **Coder QRC R17**.
2. **Cross-variant equivalence tests need collision-safe keying** (set-of-frozensets, not dict-keyed-by-(col_0_value, col_label) which silently drops duplicate-col-0 rows).
3. **Dispatcher key-form discipline:** when the dispatcher stringifies `group_value` for stats aggregation, any per-axes marker downstream must also be stringified or membership checks silently fail.
4. **Split-deploy detection via asymmetric meta failures:** `test_taxonomy_tests_resolve` passing + `test_classification_coverage` + `test_no_orphan_visual_tests` failing is the fingerprint identifying which of the two registry files was missed in the first deploy.

### Architectural Posture

**This phase ships the API surface to fix the architect-flagged spacing bug; F19 in Phase 13.5X is the test that proves it's fixed.** F1–F18 green is necessary but not sufficient evidence. The Tier 1 / Tier 2 distinction established in Phase 13.48 is now load-bearing: Tier 1 covers structural correctness (textbox content, axes counts, table shapes, legend topology); Tier 2 will cover visual-layout correctness (textbox overlap with data, label clipping, legend covering data) once `fig.canvas.get_renderer()` framework support lands.

---

## Phase 13.50.DF FIX1: P2-1 Taxonomy Fix + Matrix Regen + CRR Documentation Corrections

**Date:** 2026-06-03
**Commit:** `5010cf78`
**Tag:** `PHASE_13_50_DF_FIX1_END`
**Status:** ✅ Closed (**1057 / 0 / 0 / 1 skipped / 1 xfailed** — documentation-only at runtime)
**Predecessor:** Phase 13.50.DF v1.0 END @ `f3034153` (gate 1057)
**Specification:** `PHASE_13_50_DF_FIX1_CRR_v1_0.md` (delta amendment against CRR v1.0)

### Trigger

Sonet51-led 6-reviewer panel verdict on `PHASE_13_50_DF_CRR_v1_0` (Sonnet53_R2 main; panel members Sonnet54, Sonnet55, Sonnet56, Sonnet57, Opus2 contributing): `[!]` APPROVED WITH COMMENTS. Required pre-tag actions:
- **P2-1** (Sonnet56, Sonnet57, Sonnet54 — 3/5 independent finds): `SUMMARY_FIT.orientation` feature description in `tests/feature_taxonomy.py` doubly stale — tokens still `'horizontal'`/`'vertical'` and scope still says `"in-slot renderer only"`. Step 7b renamed the tokens; step 7b R1 extended orientation to `placement='figure'`. The targeted sed-rename for test-method names did NOT cover the feature description text. Propagates into `CAPABILITY_MATRIX.md/.html` verbatim → user-facing documentation misrepresented shipped behavior.
- **P3-1** (Sonnet53_R2, Sonnet57): CRR v1.0 §4.4 cited `test_f54` Path C; actual skipped test is `test_adf_cached_last_ax` (Phase 13.25 pre-existing). `test_f54` passed in delivery.
- **P3-2** (Sonnet57): CRR v1.0 §4.1 gate format omits `1 xfailed` (Phase 13.34 deferred feature `test_MED_1_median_uses_mad_sigma_for_errors`; exit 0 unchanged).
- **P3-3** (Opus2): CRR v1.0 §10 cited `reviewer_20260603_122216.zip`; actual delivery `reviewer_20260603_123615.zip` (~14 min timestamp drift).

### Implementation

**Source changes (P2-1 fix):**
- `tests/feature_taxonomy.py` — `SUMMARY_FIT.orientation` block (~lines 1600–1617):
  - `"name"` field updated from `"summary_fit.orientation axis: horizontal (default) / vertical (transpose) — in-slot renderer only"` to `"summary_fit.orientation axis: row (default) / column (transpose) — honored by all placements (figure, pad, subfigure)"`
  - Rationale comment block above the entry rewritten with canonical token names per v2.5 §3.4(b), step 7b rename heritage, and step 7b R1 extension scope (in-slot framing removed).
- `plots/_summary_fit.py` — two stale-token mop-up sites surfaced by parallel-sweep follow-on:
  - `_ALLOWED_DICT_KEYS` rationale comment (line ~172): tokens + scope aligned to `'row'`/`'column'` + all-placements semantic.
  - `_ALLOWED_ORIENTATIONS` use-case comment (line ~216): replaced flow text `"vertical avoids horizontal scrolling"` (legacy token leak) with `"column orientation avoids the wide-table scroll problem in that regime"`.
  - Heritage rename notes (`was 'vertical'`, `renamed from {'horizontal','vertical'}`) preserved intentionally as cross-rename traceability.
- `docs/CAPABILITY_MATRIX.md` + `docs/CAPABILITY_MATRIX.html` regenerated via `run_tests.sh` (matrix shows `PHASE_13_50_DF_END` phase header reflecting the now-applied tag).

**Documentation corrections (in FIX1 CRR §A.2, no code impact at runtime):**
- v1.0 §4.4 → corrected: actual skipped test is `test_adf_cached_last_ax`; `test_f54` passed.
- v1.0 §4.1 → corrected: actual gate is `1057 passed / 1 skipped / 1 xfailed / exit 0`. Gate-declaration format standard updated to include `xfailed` row going forward.
- v1.0 §10 → corrected: `reviewer_20260603_123615.zip`.

### Testing

- **+0** new tests (taxonomy-description + comment-block text only at runtime)
- Test count: 1057 → **1057** (unchanged)
- Features unchanged at 127
- Verified unchanged at 59
- Invariance unchanged at 356

### QRC Backlog Additions (FIX1 CRR §A.4)

The pre-CRR audit pattern Claude48 introduced in CRR v1.0 (§0 attestation, proposed as Coder QRC R17) covered **spec-vs-shipped-code** conformance but not **shipped-code-vs-documentation** conformance. P2-1 was a direct consequence: source-side step 7b rename was complete, but `feature_taxonomy.py` `"name"` field text was outside the audit scope. Two additions for the next QRC governance cycle:
- **R17 extension** — when a pre-CRR audit covers a rename or scope change, it MUST include a parallel sweep across documentation surfaces: `feature_taxonomy.py` `"name"` fields + rationale comments, `CAPABILITY_MATRIX.md/.html` generated content, inline source comments describing the feature. Fingerprint command: `grep -rn "OLD_TOKEN" tests/feature_taxonomy.py plots/ docs/`. Expected output: only heritage rename notes ("was X, now Y"); current-form descriptions using OLD_TOKEN are stale.
- **§4-facts-from-logs rule** — CRR §4 gate counts and test identity claims (skipped / xfailed test IDs) MUST be sourced from delivery-run `test_logs/test_full_*.log` and `test_logs/SUMMARY_*.txt`, NOT from working memory or compaction summaries. P3-1's `test_f54` vs `test_adf_cached_last_ax` confusion was a memory-carryover failure.

### Panel Verdict

Opus2 `[OK]` (source-verified the taxonomy `"name"` field update, the two `_summary_fit.py` mop-up sites, and the parallel-sweep coverage; confirmed only remaining stale-token grep hit is `FACET.list_grid` at `CAPABILITY_MATRIX.md:80` which is matplotlib subplot grid row/column terminology — out of `SUMMARY_FIT.orientation` scope per amendment §A.1).

**P2-2 (HTML missing from reviewer zip — third consecutive phase) deferred to Phase 13.51 scope** per panel direction. Needs `run_tests.sh` zip-builder enforcement (~5–10 lines bash, infrastructure not dfdraw source). Closed in Phase 13.50 FIX2 below.

---

## Phase 13.50.DF FIX2: run_tests.sh HTML Packaging Enforcement (Closes P2-2)

**Date:** 2026-06-03
**Commit:** `07606c02`
**Tag:** `PHASE_13_50_DF_FIX2_END`
**Status:** ✅ Closed (**1057 / 0 / 0 / 1 skipped / 1 xfailed** — tooling-only)
**Predecessor:** Phase 13.50.DF FIX1 @ `5010cf78`

### Trigger

`docs/CAPABILITY_MATRIX.html` had been absent from the reviewer zip across **three consecutive phases**: Phase 13.49 P2-1 (first occurrence), Phase 13.49 FIX1 P2-1 (reoccurrence), Phase 13.50 v1.0 P2-2 (third occurrence). Each round, the panel had flagged it as a FIX2 candidate; each round, voluntary discipline failed to prevent the next recurrence. The root cause was that `run_tests.sh`'s explicit file list in the reviewer-zip packaging block had `docs/CAPABILITY_MATRIX.md` but not `docs/CAPABILITY_MATRIX.html`. Panel direction (Sonet51 + Opus2 v1.0 CRR review): apply a mechanical guard, not another voluntary-discipline round.

### Implementation (~2 lines net + mechanical guard)

- Added `docs/CAPABILITY_MATRIX.html` to the file list in `run_tests.sh`'s reviewer-zip packaging block (root-cause fix).
- Added post-zip presence assertion: `unzip -l "$ZIPFILE" | grep -q '\.html'` warning if absent. Warning is **non-fatal** (matches the severity of the existing `.md`-unstaged warning at lines 524–533 — both flag a likely-but-not-certain miss to the operator without blocking the bundle).
- `bash -n run_tests.sh` verified.

### Testing

- **+0** new tests (tooling-only; no library / test-suite changes)
- Test count: 1057 → **1057** (unchanged)
- Features unchanged at 127
- Verified unchanged at 59
- Invariance unchanged at 356

### Lesson Recorded

**Voluntary discipline ≠ recurring-class prevention.** Three consecutive phases is the threshold where the same gap warrants infrastructure enforcement, not another flag-and-promise cycle. The 2-line `run_tests.sh` fix closes the entire class. Worth recording as a governance pattern: when a P2 recurs 3× across phases despite each occurrence being flagged, escalate to mechanical guard rather than continue advisory flagging.

---

## Statistics Summary

| Phase | Test Count | Delta | Key Feature |
|-------|------------|-------|-------------|
| 6.1 | 35 | +35 | Package scaffold, style system |
| 6.2 | 88 | +53 | Histogram, scatter plots |
| 6.3 | 143 | +55 | Profile, hist2d |
| 6.4 | - | - | Facet plots (no separate count) |
| 6.5 | 213 | +32 | Hexbin plot |
| 6.8 | 222 | +19 | AliasDataFrame integration |
| 6.9 | 241 | +28 | Batch processing |
| 12.4b5 | 242 | +10 | Statistics box, reference overlay |
| 13.1.DF | 263 | +30 | PyArrow Table input |
| 13.6.G.DF | 310 | +47 | Stats enhancements, ROOT compatibility |
| 13.12.DF | 326 | +16 | Profile enhancements, auto-title |
| 13.13.DF | 348 | +22 | same=True superposition |
| 13.14.DF | 399 | +51 | Batch defaults, subplot grid, verbose=2, interval sort fix |
| 13.15.DF | 401 | +2 | Test infrastructure: feature taxonomy, capability matrix, run_tests.sh |
| **13.16.DF** | **451** | **+50** | **Vector expression interface (bracket syntax, AD-37 fix, 7 strong invariance tests)** |
| **13.16.DF FIX1** | **469** | **+18** | **Vector path kwarg propagation fix (B1a-B5 + R4 + auto_title forwarding; 7 invariance + 5 smoke + 6 surface; ADF end-to-end verified)** |
| **13.18.DF** | ≈480 | ≈+11 | **Robust statistics extension (`stat_fields` parameter; STATS.robust + STATS.range_aware features)** |
| **13.25.DF v1.3** | 577 | ≈+97 | **Quantiles on profile (MultiGraph Phase A): error_bars + band + auto-detect modes; AD-44..AD-54; GP-1 + GP-3 governance principles)** |
| **13.26.DF v1.2** | 627 | +50 | **N-Channel Framework (MultiGraph Phase B): Algorithm A for visual-channel assignment; 10 `channels.*` style keys; factored legend; nested-band auto-detect; AD-55..AD-60; GP-2/4/5)** |
| **13.27.DF Commit 1** | 663 | +10 | **Facet refactor (MultiGraph Phase D, profile-only): `_dispatch_faceted_render()`; `facet_by=` API; backward-compat with `facet=True`; mutual exclusion with `same=True`; 6 new `channels.*` keys; 10 §9-marked invariance tests; AD-61..AD-68)** |
| **13.28.DF v1.1** | 653 | (separate branch — closed pre-Phase-13.27) | **Robust data handling: `sanitize_for_plot()` (NaN/inf filter with `nan_policy`); hybrid autorange (6 strategies, outlier-aware per-side decision); 26 tests (21 unit + 5 integration); AD-69..AD-77)** |
| **13.30.DF v1.0** | 675 | +12 | **Column-Reference Parameter Validation (Class-2): `_PROFILE_COLUMN_REFERENCES = ('group_by',)`; `_validate_column_reference_tuples()` module-import validator; runtime `ValueError` on missing-column kwargs** |
| **13.31.DF v1.0** | 687 | +12 | **`facet_by` column-name support (AD-78): dual-path dispatch (channel-mode vs column-mode); orthogonal `group_by` overlay composition; 12 §9-marked invariance tests** |
| **13.28.DF FIX1** | 696 | +9 | **Restore `autorange.*` style keys in `DEFAULT_STYLE` (4 keys registered post-Phase-13.28-Part-B regression); 9 §9-marked tests** |
| **13.32.DF v1.0** | 715 | +19 | **`group_by × quantiles` in grouped path + symmetric `facet_by_bins`/`facet_by_quantiles` (AD-79): three sub-fixes; multi-kind plot dispatch (profile/hist/scatter/hist2d); new style key `quantile.band.alpha_grouped`; 19 §9-marked invariance tests across 5 classes** |
| **13.27.DF Commit 2 v1.0** | 767 | +52 | **Phase D completion: `selection_vector` + `weights_vector` + `delta_facet`; 11 EXPLICIT_RULES; channels.label.delta_separator style key; 8 new kwargs uniformly on profile/hist/scatter/draw; hist2d signature gate; Hard Constraint §3 guard; 52 §9 tests across 12 classes (AD-61/62/65/66/67)** |
| **13.27.DF Commit 2 FIX1** | 776 | +9 | **Single-Y vector dispatch (replaces v1.0 UserWarning guard) + hist `weights=` column/expression rendering (§7a + §7b)** |
| **13.27.DF Commit 2 FIX1.FIX1** | 777 | +1 | **Assertion strengthen (`§9.SDP.6` ErrorbarContainer count) + sanitize cleanup + `§9.SDP.9` lock on inner-mode actionable error message** |
| **13.33.DF v1.0 M1** | 799 | +22 | **Normalized differential profiles: `normalize=` + `normalize_layout=` on `profile()`; 5 modes (delta/ratio/log_ratio/pull/callable); 2 layouts; 8 style keys; `_dispatch_normalize_render` (AD-80/81/82)** |
| **13.33.DF v1.0 M2** | 804 | +5 | **`group_by` + `normalize` (per-group differential) + `facet_by` + `normalize` (K×2 grid); two new dispatchers; M2 v1.0 restriction: `facet_by_bins/_quantiles` + `normalize` raises** |
| **13.33.DF v1.0 FIX1** | 805 | +1 | **`§9.NF.3` lock on `facet_by_bins/_quantiles` + `normalize` NotImplementedError with categorical-column workaround hint; dead-code cleanup; 4M-row pandas C-pass optimization** |
| **13.32.DF FIX1** | 809 | +4 | **Real-data faceted rendering bug fixes: BUG-001 (`__dfdraw_facet_bin__` leaks to titles), BUG-002 (`auto_title=True` ignored in faceted mode), BUG-003 (facet bins lexicographic sort); discovered in `time_series_tracks_0.root` TPC/ITS QA** |
| **13.34.DF v1.0** | 817 | +8 | **Capability Matrix taxonomy refresh (6-phase drift closure: +20 feature entries + 156 §9 reclassifications); M2 robustness gaps: `§9.MED.1` xfail (median MAD-sigma), `§9.STATS.1-3` (stats dict schema), `§9.X.1-5` (kwarg-composition feature-interaction)** |
| **13.34.DF FIX1** | 822 | +5 | **BUG-010: untracked test file inflated 822/0 gate over 817-test commit; +5 §9 tests locking the bug class (`test_phase_13_34_df_fix1_bug010.py`)** |
| **13.34.DF FIX2** | 822 | 0 | **BUG-011: `run_tests.sh` pre-bundle staging check blocks `reviewer.zip` when untracked `.py` files in `tests/`; override via `DFDRAW_SKIP_STAGING_CHECK=1`; tooling-only, no test count change** |
| **13.35.DF v1.3** | 833 | +11 | **`group_by_bins` + `hist_norm` for `hist()` (BUG-013 hist side): adds 4 explicit params to draw_hist + DFDraw.hist signatures + `_HIST_FORWARDED_NAMES`; BUG-012 guard; shared bin edges; per-group normalization; `_group_weights` helper; 7 edits across drawer.py + histogram.py; architect TPC/ITS reproducer green** |
| **13.36.DF v1.2** | **843** | **+10** | **User style kwargs override auto-cycle (BUG-013 style-override side): `_ud_user_*` sentinel capture before style fill-in; `_user_marker`/`_user_markersize`/`_user_color` forwarded to `_draw_profile_grouped`; `_user_color` to `_draw_hist_grouped`; per-group cycle gated on `is None`; "indistinguishable" UserWarning; `_*_FORWARDED_NAMES` extended (`markersize` omitted from hist per Sonet51 P1); architect priority rule: user kwarg > channel cycle > style default** |

| **13.37.DF v1.1** | 867 | +24 | **Histogram robustness: BUG-014 (`histtype='step'` all-black — edgecolor sentinel), BUG-015 (profile float group_by guard), BUG-016 (`_interval_sort_key` no-op for pd.Interval — hasattr guard; verified by execution), `hist_errors=True` (Poisson error bars with 3-tier Σw² support), `linestyle_cycle=True` (Phase 13.26 channels extension). Sentinel pattern extended to edgecolor + linestyle. Spec: 2 P0 rounds (v1.0 → v1.1)** |
| **13.37.DF FIX1** | 870 | +3 | **Test expansion: Phase 13.36 backward-compat locks. Promoted PROFILE.group_by_bins, SAME.auto_features, VECTOR.color_cycle to ✅ Verified (34 → 37). 2 spec bugs caught at code time: label filter anti-pattern + SO.COMPAT.3 vector-vs-scalar wrong premise. Verified: 34 → 37, Invariance: 193 → 196** |
| **13.38.DF v1.1** | 889 | +19 | **Scatter enhancements: BUG-017 (`facet_by` float guard — 3rd instance of BUG-012/015 class), `xerr`/`yerr` error bars (3-tier NaN policy), expression `color=`/`marker=` (`_process_color` CP0-1 dispatch reorder: column before `to_rgba`; regression-lock ECM.6). `get_array()` over `get_facecolor()` for colormap dispatch (QRC v1.32 carry-forward). Verified: 37 → 42, Invariance: 196 → 215** |
| **13.39.DF v1.2** | **913** | **+24** | **2D Profile (`profcolz`): `draw_profile2d()` via `z:y:x` expression; `scipy.stats.binned_statistic_2d`; `min_entries=` masking; pcolormesh + colorbar; ROOT TProfile2D equivalent. Time Axis: `time_format=` kwarg on 4 plot types; datetime64 auto-detect before `astype(float)` (QRC v1.32 carry-forward). Scatter3D: `type='scatter3d'`; `mpl_toolkits.mplot3d`; reuses Phase 13.38 `_process_color`/`_process_size`. 4 fix-at-code-time disclosures. Verified: 42 → 47, Invariance: 215 → 239** |
| **13.40.DF v1.0** | 923 | +10 | **Cumulative histogram (`cumulative=True/-1/False`) — ROOT TH1::Draw('cumulative') equivalent. 3 values: True (ascending CDF/ECDF), False (default, byte-identical backward compat), -1 (descending/survival, ROOT convention). matplotlib native `cumulative=` forwarded at 4 internal call sites (incl. `_dispatch_faceted_render` — Phase 13.16.DF FIX1 lesson applied recursively; CP2-1 lock for 3rd site). Composes with: `norm='probability'`, group_by overlaid+stacked, facet_by, histtype='step'. M5 correctness guard: `hist_errors+cumulative` → `NotImplementedError` (Poisson per-bin errors independent; cumulative bins correlated). Vector dispatch `[x,y]` propagation lock — Phase 13.16.DF FIX1 bug class** |
| **13.41.DF v1.0** | 942 | +19 | **N-D Faceting via `facet_by=List[str]` for 1D/2D/3D — `facet_by[0]=ROW`, `[1]=COL`, `[2]=FIGID` (numpy/pandas shape convention LOCKED). 3D returns `(List[Figure], List[axes_2d], List[stats_dict])` — DEVIATES from `(fig, ax, stats)` contract; documented in inline help. New params: `share_x`, `share_y` ∈ {'all','row','col','none'}, `share_across_figures`. New helpers: `_normalize_facet_args`, `_to_mpl_share`, `_validate_share_axis_value`, `_resolve_facet_values`, `_filter_facet_value`, `_compute_global_ranges`. Per-plot-kind lock for `share_across_figures` (CP1-2): scatter locks x AND y; hist/profile locks x only. Per-plot-kind x-range dispatch: hist `range=`, profile `range=` remapped to `x_range=`, scatter `ax.set_xlim` post-draw. Empty-cell `'(no data)'` diagnostic + `stats={'n':0,'empty':True}`. dfdraw FIRST major plotting library with unified API for Nth-dimension separate-figures faceting (seaborn/ggplot2/plotly/altair require manual loops)** |
| **13.41.DF FIX1** | 945 | +3 | **3 bugs from v1.6 panel review — closed in FIX1 commit (predecessor `530954d1`)** |
| **13.41.DF FIX2** | 946 | +1 | **5 panel-flagged P2/P3 items + FBY.23 lock — closes the N-D faceting feature shape** |
| **13.42.DF v1.0** | **973** | **+27** | **Inline fits (`fit=` parameter on hist/profile/scatter/draw). Three input forms: `str` shorthand, `dict` spec (initial_guess/bounds/range/use_errors/raise_on_failure), `Callable`. Vector dispatch list form. Per-channel `linestyle_cycle`. Stats integration: `stats['fit']` = `List[List[Dict]]`. Composes with `group_by` (dict keyed by group), `facet_by` (per-cell), `vector_expr`. `normalize=` + `fit=` silent consume (CP1-5; F.26 lock). New `plots/fits.py` registry (gauss/pol0-5/linear/expo); public `register_fit(name, function, n_params, guess_fn)`. New `plots/_fit_render.py` (ROOT-style param textbox + overlay). 7 new style keys: `fit.linewidth/linestyle_cycle/position/text_format/text_padding/text_fontsize_default/text_fontsize_facet` (NOTE: all silently no-op in v1.0 due to D-2 `_style_get` defect; fixed in FIX1). Sonnet54 P1-B at close (CRR v2): profile grouped fit returned single fit on combined data instead of per-group dict; fixed pre-tag (F.27 lock). Predecessor: `PHASE_13_41_DF_FIX2_END` (gate 946)** |
| **13.42.DF FIX1** | **981** | **+8** | **Production-gate bug closure + interface lock. 30 minutes of real TPC ITS-TPC calibration data surfaced 7 bugs (5 P1 silently-wrong-output) that 5 reviewers + 27 invariance tests missed. Correctness: B4 (grouped fit reuses main-path masks; `.eq()` Interval-safe; Sonnet55 extension to top_k/sort_groups; new `fit_status='skipped_empty'`), B5 (χ² Poisson default `sqrt(max(counts,1))`; matches ROOT TH1::Fit Neyman convention; **D-1 [BREACH]** companion fix at `dispatch_fit` use_errors default flip False→True for hist), D5 (vector fit pairing per v1.4 §6.3 verbatim; F.12 inverted), D9/R4 (stacked+group_by+fit → N per-group fits, stacking purely visual). Rendering: B1 (facet_mode plumbed at 3 call sites; **D-2 [BREACH]** deeper root cause — `_style_get` used broken `get_style(key)` API → ALL `fit.*` style keys silently ignored since Phase 13.42 v1.0; fixed to `get_style_value(key, default)`), B2/B3 (new `fit_textbox_kwargs={'fontsize','format','show_fields'}` per-call kwarg with sub-key validation; compact format = one line per fit; format='auto' = compact if facet & n_blocks>1). Interface LOCKED at close: `fit_textbox_kwargs` sub-keys + enum values; D8/R3 (scatter `yerr=` column as opt-in; `use_errors` redundant); D9/R4 (per-group dict shape). Tests: F.28+F.28b (B4 expression+quantile + skipped_empty), F.29 (B5 redchi ∈ [0.5,2.5]), F.30 (B1 set_style round-trip), F.31 (D5 pairing), F.32 (D9 per-group dict), F.33×2 (override + precedence over set_style). FIT.inline count 27 → 35. **Two `[BREACH]` disclosures (D-1, D-2)** flagged per proposed Coder QRC #10 (verbatim-spec deviation escalation rule, architect-ratified R6). Third consecutive phase to miss taxonomy staging (Sonet50 governance note → run_tests.sh pre-bundle taxonomy-count check proposed). Production gate methodology validated; `PHASE_13_42_DF_PROD_GATE_Bugs_v1_0.md` + `PHASE_13_42_DF_POST_GATE_Audit_Questions_v1_0.md` shipped for post-FIX1 process-improvement audit** |

| **13.42.DF FIX2** | **987** | **+6** | **Close 5 items deferred at FIX1: B6 (suptitle padding adapts to title line count), B7 (faceted+fit no-crash smoke), I-8 (hist+weights+fit UserWarning), ADV-1 (stacked+selection_vector(>1)+fit → NotImplementedError), ADV-3 (`fit_textbox_kwargs` threaded into 3 FORWARDED_NAMES tuples + outer signatures + explicit forwarding). F.59-F.63 + F.61b. FIT.inline 35 → 41. Defer-anti-pattern → proposed QRC #11 + Reviewer supplement (`Claude48_Feedback_FIX1_Defer_Anti_Pattern_20260527.md`). Tag `PHASE_13_42_DF_FIX2_END`** |
| **13.43.DF v1.0** | **1014** | **+27** | **`summary_fit=` standalone fit-result figures. New `plots/_summary_fit.py` (~650 LOC); outer-layer consume through `DFDraw.{hist,profile,scatter,draw}`; faceted aggregation in `_dispatch_2d_facet` + `_dispatch_faceted_render`; `_consumed` normalize-set extended; 13 `summary_fit.*` style keys. 26 invariance tests F.34-F.56 (+F.38a/F.47b) in `TestPhase1343SummaryFit` (FIT.summary feature, Verified). R-2 fix at END: `DFDraw.draw()` scalar delegations dropped `fit`/`fit_textbox_kwargs`/`summary_fit` (named params not in `**kwargs`) at 3 sites — fixed, locked by F.56c; `feature_taxonomy.py` `name`-schema fix (stale `title` key crashed matrix). CRR §2: vector summary_fit → stats[0]; 3D-facet deferred. Commit body states pre-R-2 1013/+26; END gate 1014. Invariance 313 → 340. Tag `PHASE_13_43_DF_END`** |
| **run_tests.sh (tooling)** | 1014 | 0 | **PHASE_HISTORY ↔ git-tag drift check. ~96 LOC after BUG-011 staging check: BLOCK (exit 1) on any `PHASE_*_END` in `docs/PHASE_HISTORY.md` not in `git tag --list 'PHASE_*_END'` (override `DFDRAW_SKIP_TAG_DRIFT_CHECK=1`); WARN heuristic for misplaced `FIX<N>_END` tags. Reverse direction (repo ahead of doc) intentionally not blocked (expected backfill transient). Surfaced + resolved the Phase 13.25 tag incident (`PHASE_13_25_DF_FIX1_END` misplaced on `da8895e2` → moved to `06f84ff8`; `PHASE_13_25_DF_FIX2_END` created on `da8895e2`). Lesson: phase status = git tags ONLY. Tooling-only, no test-count change. `PHASE_13_46_DF_BEGIN` placed here (`02510a20`). Sonet50 panel `[!]` APPROVED** |
| **13.46.DF v1.0** | **1022** | **+8** | **Audit bucket ① fixes (C-1/C-2/C-4/C-7/C-9). C-1 `fit='gaus'` ROOT TF1 alias (`register_fit`); C-2 `type='histo'` ROOT alias (`_TYPE_ALIASES`); C-4 source `_get_suptitle` helper (public `get_suptitle()` mpl≥3.8 + private fallback) replacing 9 inline `fig._suptitle` sites (retires Phase 13.42 FIX2 §2.2 disclosure); C-7 kwarg-typo guard at `draw()` entry (`difflib.get_close_matches(cutoff=0.8)` did-you-mean; K = 6 method sigs ∪ 5 FORWARDED_NAMES, reviewer note N-1); C-9 `range=` on scatter via shared `resolve_range_2d` (v1.0 view-clip, all strategies, honest stats; original profile/hist unpack bug fixed). C-3 (faceted auto_title) intentionally excluded → Phase 13.47. F.64-F.70 (F.69 a/b). +4 features (FIT.root_aliases, API.kwarg_typo_guard, RANGE.scatter, TITLE.get_suptitle). §2.1 ruling: Option 1 shared-global faceted scatter range (per-cell available via `facet_by=[list]+share_x='none'`). Invariance 340 → 348, Verified 51 → 55. Spec `PHASE_13_46_DF_v1_3_AuditFixes_Proposal.md`; audit `PHASE_13_45_dfdraw_Audit_Findings.md`. Predecessor `PHASE_13_43_DF_END` @ gate 1014. Closure tag is FIX1_END (no separate v1.0 END tag)** |
| **13.46.DF FIX1** | **1023** | **+1** | **Scatter `range=` REMOVES out-of-range points (point filter), per architect 2026-05-28 — consistent with hist/profile range= excluding points from binning. v1.0 only view-clipped (`set_xlim`); FIX1 filters `x_data`/`y_data`/`df_filtered` by one mask before stats+plotting (parallel color/size/marker/error arrays derive from `df_filtered` → stay aligned automatically). Stats computed post-filter (honest counts). Non-facet keeps exact tight view; faceted cells filter and shared axes autoscale to the union (no last-cell-wins). F.71 locks the point-removal invariant (percentile_99 & explicit-tuple drop points; minmax removes nothing; color array stays aligned). +1 feature `RANGE.scatter_filter`. Invariance 348 → 349, Verified 55 → 56. Predecessor v1.0 @ `1d77702e`. Tag `PHASE_13_46_DF_FIX1_END`; rolling `PHASE_BEGIN_dfdraw` → `ad91e251`** |
| **run_tests.sh WARN downgrade** | **1023** | **+0** | **Tag-drift check downgraded to non-blocking WARNING (commit `df3057a3`, tooling-only). Hard-blocking the reviewer bundle on heuristic `grep` of PHASE_HISTORY prose was fragile and created override pressure (silent bypass → dead-weight check). Two-part fix: (1) `grep` scope narrowed to declarative `` 'tag `PHASE_X_END`' `` references only (not prose mentions); (2) severity demoted to non-fatal WARNING in `SUMMARY` artifact in `reviewer.zip` — drift stays visible, bundle always builds, operator cannot silently bypass. Parallels the earlier `02510a20` tag-drift check addition recorded in v1.10. Lesson: heuristic gate-blocks need a non-blocking off-ramp — tighten scope AND demote severity together; either alone leaves a failure mode. Features unchanged at 114; Verified unchanged at 56; invariance unchanged at 349** |
| **13.48.DF v1.0** | **1034** | **+11** | **Tier-1 automated visual testing — new `visual_primitive` test layer. `VisualCheck(fig, stats, df)` framework with collect-all-then-assert pattern; cell iteration via `fig.axes` (dispatcher-agnostic); `visible_cell_axes` excludes hidden padding; series counts via `ax.containers` (not `len(ax.lines)` — errorbar caps populate `ax.lines` but aren't series); distinct-RGBA distinctness; dtype-safe `zip(sorted_unique(...))`. 11 V-checks V.1-V.10 + V.2 ragged-padding-safety lock under `TestPhase1348VisualPrimitive`. +6 features (`VISUAL.framework`, `VISUAL.cell_iteration`, `VISUAL.series_count`, `VISUAL.distinct_colors`, `VISUAL.shared_axes`, `VISUAL.layout_visibility`) — all Smoke-only at close pending Phase 13.49 matrix-traceability work. Verified 56 unchanged; invariance 349 unchanged; **visual_primitive layer NEW: 0 → 11**. Tier 1 / Tier 2 split established here. Predecessor `PHASE_13_46_DF_FIX1_END @ ad91e251` (gate 1023). Tag `PHASE_13_48_DF_END`** |
| **13.49.DF v1.0** | **1038** | **+4** | **Capability Matrix Traceability — link infrastructure (per-feature `tests: [List[str]]` field in `feature_taxonomy.py`; `TEST_LAYERS` dict in `test_layer_classification.py`); HTML matrix with per-feature expandable tests-panel, status × visual × category filters; orthogonal Visual column with 👁 badge. 4 M-tests M.1-M.4 (`test_taxonomy_tests_resolve`, `test_classification_coverage`, `test_html_matrix_locks`, `test_no_orphan_visual_tests`) + `META.capability_matrix` feature (Verified — claims the 4 M-tests). KNOWN_UNCLAIMED §3.7 governance: 64 seeded entries with SPECIFIC target_phases (13.27.DF=50, 13.28.DF=9, 13.32.DF=5 per Opus48_1 P2-A advisory adopted). D-K + D-L normalization fixes at 3 sites (basename `split('/')[-1]`): v1.0 CRR was [X] REJECTED by Opus2 for `if startswith("tests/"):` regression matching 0 features → all 121 Planned. v1.1 resubmission [!] APPROVED 8/8; D-L was Claude48 self-discovery under proposed Coder QRC R7. Features 120 → 121 (+1 META); Verified 56 → 57; invariance 349 → 353. Predecessor `PHASE_13_48_DF_END @ 9f612601`. Tag `PHASE_13_49_DF_END`** |
| **13.49.DF FIX1** | **1038** | **+0** | **HTML rendering fixes (H-1/H-2/H-3) + M.3 lock (same-test stricter, no test count change). Architect rendered HTML and found 3 bugs the 8-reviewer v1.1 panel missed: H-1 META Broken in HTML vs Verified in MD (stale-artifact hypothesis); H-2 JS selector `[data-status]` too broad — clicking feature `<tr>` row hijacked filter state (8 selectors scoped to `.filter-group [data-status]`); H-3 15 duplicate category headers (FACET 5×, PROFILE 4×, HIST 4×) — FEATURES in chronological commit order needed sort-by-category before emit in BOTH MD and HTML. M.3 extended with two gate-locked invariants on HTML: (a) each category-row appears exactly once; (b) HTML data-status counts match MD-computed counts on same test_results. Both fail on original buggy emitter; pass on this commit. Meta-lesson: v1.1 panel approved without rendering the HTML — direct trigger for **Reviewer QRC v1.31 Rule 14 caveat** ("diff-read does not discharge visual-render verification; Q6 trigger applies independently"). P2-1 HTML still missing from reviewer zip recorded as first occurrence of what became 3× recurring packaging gap (closed in Phase 13.50 FIX2). Features unchanged at 121; Verified unchanged at 57; invariance unchanged at 353. Tag `PHASE_13_49_DF_FIX1_END`** |
| **13.50.DF v1.0** | **1057** | **+19** | **Fit-rendering overhaul (proposal v2.5 after 4-round panel convergence v2.2→v2.3→v2.4→v2.5). Six features added: `FIT.display_names` (render-only `_DISPLAY_NAMES` map — `slope→p1`/`intercept→p0`/`center→$\mu$`/`sigma→$\sigma$`/`decay→$\tau$`/`amplitude→A`; `plots/fits.py` UNCHANGED); `FIT.precision_modes` (**[BREACH]** `fit.text_format` REMOVED; replaced by `fit.value_format`/`fit.error_format`/`fit.precision_mode='physics'` — error 1sf, value aligned to error's decimal); `FIT.textbox_kwargs_extensions` (`rename_params`/`value_format`/`error_format`/`precision_mode` per-call overrides); `LEGEND.modes` (polymorphic `legend=` kwarg `bool|str|dict|None` with `'shared'`/`'first'` modes; `show_legend=` permanent back-compat parallel — both Pattern A); `SUMMARY_FIT.placement` axis (`'figure'`/`'subfigure'`/`'pad'`; subfigure insets are PER-PANEL slices per Claude36 ADF P2-NEW-3; pad pre-planned at GridSpec construction per Sonnet52_R1 P1-A — GridSpec is immutable post-creation); `SUMMARY_FIT.orientation` axis (`'row'`/`'column'`; honored by all placements). **[BREACH]** `summary_fit.precision` int REMOVED (11 sites: 8 behavioral + 3 documentation). 19 new tests in `tests/test_phase_13_50_df_fit_visual.py`: F1-F16 visual_primitive + F17 cross-variant table-content equivalence (via collision-safe `table_cells_keyed → set[frozenset[(col, val)]]`) + F18 `show_legend↔legend` 4-tuple `legend_topology` behavioral equivalence + `test_normalize_legend_spec_idempotent`. CRR introduces **§0 pre-CRR gap-audit attestation pattern** (proposed Coder QRC R17). Features 121 → 127; Verified 57 → 59; invariance 353 → 356; **visual_primitive 11 → 27** (+16). **F19 textbox-bbox-overlap deferred to Phase 13.5X Tier 2** — F1-F18 green is necessary but not sufficient evidence for the architect-flagged spacing bug being cured. Predecessor `PHASE_13_49_DF_FIX1_END @ a6ddc753`. Tag `PHASE_13_50_DF_END`** |
| **13.50.DF FIX1** | **1057** | **+0** | **P2-1 stale-taxonomy fix (Sonnet56/57/54 — 3/5 independent finds in 6-reviewer panel on CRR v1.0) + matrix regen + CRR v1.0 documentation corrections (P3-1/P3-2/P3-3). `SUMMARY_FIT.orientation` feature description in `tests/feature_taxonomy.py` was doubly stale — tokens still `'horizontal'`/`'vertical'` and scope still `"in-slot renderer only"` despite step 7b rename + step 7b R1 figure-renderer extension; targeted sed-rename did not cover the description text → propagated into CAPABILITY_MATRIX verbatim. `"name"` field updated to `"summary_fit.orientation axis: row (default) / column (transpose) — honored by all placements (figure, pad, subfigure)"`; rationale comment block rewritten; two stale-token mop-up sites in `plots/_summary_fit.py` (`_ALLOWED_DICT_KEYS` line ~172, `_ALLOWED_ORIENTATIONS` line ~216) caught by parallel-sweep follow-on; heritage rename notes preserved. CRR v1.0 §4.4 corrected: actual skipped test is `test_adf_cached_last_ax` (Phase 13.25 pre-existing) not `test_f54`. §4.1 gate format corrected to include `1 xfailed` (Phase 13.34 deferred). §10 zip filename corrected. **QRC backlog items 5+6 added**: R17 EXTENSION (parallel doc-surface sweep on rename/scope-change events; fingerprint `grep -rn "OLD_TOKEN" tests/feature_taxonomy.py plots/ docs/` returning only heritage rename notes); §4-facts-from-logs rule (CRR §4 from `test_logs/test_full_*.log`, not working memory). Documentation-only at runtime: features/Verified/invariance unchanged. Tag `PHASE_13_50_DF_FIX1_END`** |
| **13.50.DF FIX2** | **1057** | **+0** | **`run_tests.sh` HTML packaging enforcement — closes P2-2 (HTML missing from reviewer zip) after 3 consecutive recurrences (Phase 13.49 P2-1, Phase 13.49 FIX1 P2-1, Phase 13.50 v1.0 P2-2). Voluntary discipline failed 3× in a row; panel direction (Sonet51 + Opus2) was mechanical guard not another flag-and-promise cycle. Two-line change: added `docs/CAPABILITY_MATRIX.html` to the explicit file list in the reviewer-zip packaging block (root-cause fix); added post-zip presence assertion `unzip -l "$ZIPFILE" | grep -q '\.html'` as non-fatal warning (matches severity of existing `.md`-unstaged warning at lines 524-533). `bash -n run_tests.sh` verified. Tooling-only: features/Verified/invariance unchanged. **Lesson recorded**: 3 consecutive phases is the threshold where a recurring P2 warrants infrastructure enforcement, not another flag-and-promise cycle. Tag `PHASE_13_50_DF_FIX2_END`** |

**Total Development (as of Phase 13.50.DF FIX2):** 59 phase entries, **1057 tests** + 1 skipped + 1 xfailed, **127 features**, **356 invariance tests**, **27 visual_primitive tests**, **59 Verified features**

> **Phase ordering note (post-13.46 FIX1):** chronological commit order is 13.46 FIX1 (`ad91e251`, tag `PHASE_13_46_DF_FIX1_END`, gate 1023) → PHASE_HISTORY v1.10 doc commit (`89884f81`, tooling) → `run_tests.sh` WARN downgrade (`df3057a3`, tooling-only — tag-drift check hard-block → non-blocking WARNING) → 13.48 v1.0 (`9f612601`, tag `PHASE_13_48_DF_END`, gate 1034) → 13.49 v1.0 (`89bc63c6`, tag `PHASE_13_49_DF_END`, gate 1038) → 13.49 FIX1 (`a6ddc753`, tag `PHASE_13_49_DF_FIX1_END`, gate 1038 same-test stricter) → 13.50 step 1 (`c029b405`) → step 2 (`e904ccec`) → step 3 (`181c2366`) → step 4 (`ccd8243d`) → step 5 (`068da47f`) → 13.50 step 7 closing commit (`f3034153`, tag `PHASE_13_50_DF_END`, gate 1057) → 13.50 FIX1 (`5010cf78`, tag `PHASE_13_50_DF_FIX1_END`, gate 1057) → 13.50 FIX2 (`07606c02`, tag `PHASE_13_50_DF_FIX2_END`, gate 1057). Phase numbers monotonic across this window. Phase 13.47 (slim bucket) intentionally skipped — pending in queue per architect priority direction toward extended-graphics work (13.48+). The `df3057a3` tooling commit is recorded as a phase entry in the Statistics Summary table parallel to the earlier `02510a20` tag-drift check addition (v1.10). Phase 13.50.DF v1.0 step commits 1-6 land separately on the branch but the END tag is at the closing step-7 commit (`f3034153`); steps 1-6 do not have separate END tags (parallel to Phase 13.46 v1.0 / FIX1 single-END-tag pattern). Tag `PHASE_BEGIN_dfdraw` rolled `ad91e251` → `07606c02` across this window.

> **Phase ordering note (post-13.39):** chronological commit order is 13.39 v1.2 (`b024414e` / `3c5d4547`) → 13.40 v1.0 (`67d125e2`) → 13.41 v1.0 (`530954d1`) → 13.41 FIX1 (`b84576a0`) → 13.41 FIX2 (`70b94a3e`) → 13.42 v1.0 (`38aed2d8`) → 13.42 FIX1 main (`82aaa903`) → 13.42 FIX1 P1 follow-up (`28f7f3ce`). Phase numbers monotonic in this window. Tag `PHASE_BEGIN_dfdraw` was at `38aed2d8` at Phase 13.42.DF close; moved to `28f7f3ce` at Phase 13.42.DF FIX1 close.

> **Phase ordering note (post-13.42 FIX1):** chronological commit order is 13.42 FIX1 P1 follow-up (`28f7f3ce`) → PHASE_HISTORY v1.9 doc (`e463161c`) → 13.42 FIX2 (`79d449c3`, tag `PHASE_13_42_DF_FIX2_END`, gate 987) → 13.43 v1.0 (`0e0d79f7`, tag `PHASE_13_43_DF_END`, gate 1014) → run_tests.sh tag-drift check (`02510a20`, tag `PHASE_13_46_DF_BEGIN`) → 13.46 v1.0 (`1d77702e`, gate 1022) → 13.46 FIX1 (`ad91e251`, tag `PHASE_13_46_DF_FIX1_END`, gate 1023). Phase numbers monotonic in this window. Note: the v1.9 doc commit (`e463161c`) predates 13.42 FIX2, so v1.9 did not yet record FIX2 — this v1.10 backfill adds it. Phase 13.46.DF v1.0 was committed (`1d77702e`) but not separately END-tagged; its closure tag is `PHASE_13_46_DF_FIX1_END` at `ad91e251`. Tag `PHASE_BEGIN_dfdraw` moved `28f7f3ce` → `ad91e251` across this window.


> **Phase ordering note (post-13.32 v1.0):** the chronological commit order on `feature/groupby-optimization` from 2026-05-16 onward is 13.27.DF Commit 2 v1.0 (`84dcf916`) → Commit 2 FIX1 (`ba42fcde`) → Commit 2 FIX1.FIX1 (`b929ccb9`) → 13.33.DF v1.0 M1 (`61460df5`) → 13.33.DF v1.0 M2 (`c6a3245f`) → 13.33.DF v1.0 FIX1 (`94594f89`) → 13.32.DF FIX1 (`195ab4ea` / `d0b04f88`) → 13.34.DF v1.0 (`463deb36` / `abf5fe40`) → 13.34.DF FIX1 (`379f26bd`) → 13.34.DF FIX2 (`b38395db`) → 13.35.DF (`3b910aec`) → 13.36.DF (`2f4d959f`). Phase numbers are NON-monotonic vs commit date: 13.27 Commit 2 series lands after the 13.32 v1.0 entry above, and 13.32 FIX1 chronologically follows 13.33 v1.0 FIX1. Phase numbers index the *originating* phase, not the commit order — consistent with the post-13.27 Commit 1 phase ordering note above.

> Note: Phase 13.28.DF closed at 653 tests on commit `8b02d241` (2026-05-09). Phase 13.27.DF Commit 1 then added 10 tests for a current total of 663. Phase 13.28 was developed in parallel with Phase 13.27 design; the two phases used disjoint AD ranges (AD-61..AD-68 vs AD-69..AD-77) so the merge was clean.

> **Phase ordering note (post-13.27 commit 1):** the chronological commit order on `feature/groupby-optimization` is 13.27.DF Commit 1 (663) → 13.30.DF (675) → 13.31.DF (687) → 13.28.DF FIX1 (696) → 13.32.DF (715). The phase-number sequence is non-monotonic because 13.28.DF FIX1 is a follow-up FIX commit that landed *after* the 13.30/13.31 main phases — phase numbers index the *originating* phase, not the commit order. AD ranges remain disjoint across all phases.

---

## API Stability

**Current Phase:** Experimental

All APIs subject to change based on user feedback and integration testing with:
- AliasDataFrame (Phase 13.3.ADF)
- GroupByRegressor (Phase 13.1.GB)
- RDataFrameDSL (Phase 12.3)

**Stability Roadmap:**
- **Phase 14.x:** Experimental → Stable (API freeze after integration testing)
- **Phase 15.x:** Stable → Frozen (production deployments)

---

## Cross-Component Integration

### Upstream Dependencies
- **pandas:** Required (DataFrame operations)
- **numpy:** Required (numerical operations)
- **matplotlib:** Required (plotting)
- **pyarrow:** Optional (Phase 13.1.DF)

### Downstream Integrations
- **AliasDataFrame:** Duck-typed axis titles (Phase 6.8), draw_figures (planned delegation)
- **GroupByRegressor:** PyArrow Table output (Phase 13.1.DF)
- **RDataFrameDSL:** Batch QA plot generation (Phase 6.9)

### Integration Points
1. **Data Input:** DataFrame, AliasDataFrame, dict, PyArrow Table
2. **Axis Titles:** Duck-typed `get_axis_title()` method
3. **Batch Processing:** Dict specs, list-of-groups, JSON/YAML files
4. **Statistics:** Standardized stats dict format for all plots
5. **Overlay:** `same=True` for ROOT-like superposition

---

## Open Items

### For Next Phase
- [ ] **AliasDataFrame delegation:** ADF `draw_figures()` should delegate drawing to dfdraw `draw_batch()` (3-phase split: scan → draw → cleanup)
- [ ] **Technical summary update:** ✅ Done (this document)
- [ ] **Test infrastructure:** `run_tests.sh`, `phase_tag.sh`, CAPABILITY_MATRIX

### Technical Debt
- AliasDataFrame `draw_figures()` reimplements drawing loop independently (missing defaults cascade, same=True, layout, figsize). Fix planned in Phase 13.13.ADF v1.0.

### Documentation Gaps
- [ ] Add tutorial notebook for PyArrow workflows
- [ ] Document integration patterns with GroupByRegressor
- [ ] Add performance comparison (pandas vs PyArrow upstream)

---

## Lessons Learned

### What Worked Well
1. **Incremental development:** Each phase added clear value
2. **Test-first approach:** 469 tests caught regressions early
3. **Duck typing:** Clean integration without hard dependencies
4. **Style system:** Established early, avoided later refactoring
5. **Governance process:** Proposal → review → implement → test cycle caught issues before production
6. **Multi-reviewer source verification (Phase 13.16.DF):** External reviewers catching 6 P0 defects that internal approvers missed proved the Rev2→Rev3 cycle works as designed
7. **Strong A≡B invariance tests (Phase 13.16.DF):** Byte-identical axes comparison catches divergences at unit-test level instead of real-data level
8. **Two-commit pattern for fix phases (Phase 13.16.DF FIX1):** Commit 1 (red baseline) + Commit 2 (green fix) preserves the diagnostic state in git history forever; pre-fix `reviewer.zip` becomes a permanent regression-detection artifact
9. **Fresh-reviewer pattern (Phase 13.16.DF FIX1):** When debug cycles exceed 2 turns, an unbiased source-read by a fresh reviewer resolves faster than continued debug-print iteration; commitment bias is real
10. **Cross-subproject end-to-end verification (Phase 13.16.DF FIX1):** ADF team's parallel Phase 13.19.ADF.FIX1 with K2 test suite validated the dfdraw fix through the full pipeline within 2 working days — the strongest possible cross-subproject validation pattern
11. **Class-load validation (Phase 13.16.DF FIX1):** `_validate_forwarded_names()` running at module import catches signature drift loudly at import time, not silently at runtime — codified as the pattern for all future signature-coupled tuples

### What Could Improve
1. **Earlier integration testing:** ADF `draw_figures()` duplication discovered late
2. **Documentation cadence:** Should update with each phase (codified in Org v1.24 § Update Discipline; Phase 13.16.DF is the first phase to apply this rule)
3. **Performance profiling:** Should have benchmarked earlier phases
4. **Verbose debug mode:** Would have caught the ADF defaults cascade issue faster
5. **Source verification discipline (Phase 13.16.DF):** Proposal enumeration alone is insufficient for shared-state changes — reviewers must count call sites in source
6. **Scaffolding separation (Phase 13.16.DF):** `run_tests.sh` and similar infrastructure should not share commits with feature work
7. **Tooling-packet hygiene (Phase 13.16.DF FIX1):** `reviewer.zip` was missing `test_full_*.log` until Claude45 caught it mid-cycle; tooling completeness gaps surface only when downstream reviewers actually need the artifact
8. **Spec inventory accuracy (Phase 13.16.DF FIX1):** v1.4 §3.1 inventory had `top_k` miscategorized as facet-only across 3 methods; 4 source-verifying reviewers approved the proposal without catching it; only implementation source-read caught the categorization error — argues for AST-derived inventories over hand-typed ones
9. **Style-key registration regression class (Phase 13.28.DF FIX1):** Four `autorange.*` keys referenced via `get_style_value()` from three modules but never registered in `DEFAULT_STYLE` — silently broke `set_style({"autorange.*": ...})` since Phase 13.28 Part B introduction. Caught only by Phase 13.32 documentation audit, not by any runtime test. Argues for a `get_style_value` ↔ `DEFAULT_STYLE` cross-check validator (R6-analogue for style keys) — folded into Phase 13.30 sub-fix 2 scope
10. **Source freshness as binding rule (Phase 13.32.DF debug cycle):** Two patches built on stale source baselines (drafter held an older `drawer.py` in working memory than the architect's repo). Resolved by Coder QRC Rule N+5: every patch declares its baseline SHA and is rebased only on the architect's committed HEAD
11. **Wrong-bundle review artifact (Phase 13.32.DF closure):** Sonnet53_R2 reviewed `sourcesdf.zip` (intermediate state) instead of the official `reviewer_20260515_165043.zip` — produced spurious P0/P1 findings invalidated by Sonet50 consolidation. Argues for review-bundle checksum verification before issuing verdicts

### Best Practices Established
1. **Expression syntax:** ROOT-like syntax reduces learning curve
2. **Return tuples:** `(fig, ax, stats)` consistent across all methods
3. **Keyword-only args:** After first positional, all kwargs for clarity
4. **Graceful degradation:** Optional dependencies handled cleanly
5. **Option hierarchy:** More local wins (kwargs < batch < group < plot)
6. **Byte-identical invariance tests:** The quality bar for phases touching shared state (line count, colors, linestyles, xdata/ydata to 10 decimals + stats to 1e-9)
7. **Vector expressions over loops:** For N-series overlays where color/label continuity matters, vector syntax (`[y1,y2]:x`) beats scalar loops with `same=True` — especially across ADF boundaries
8. **Forwarded-name tuples + class-load validation (Phase 13.16.DF FIX1):** Class-level tuples enumerate which named parameters propagate through dispatch; `_validate_forwarded_names()` runs at module import to catch signature drift loudly
9. **AST-derived over hand-typed inventories (Phase 13.16.DF FIX1):** When a proposal must enumerate signature parameters, derive via `inspect.signature()` rather than hand-typing; v1.4 §3.1 categorization errors were avoided in implementation by reading source directly
10. **Scope-positive divergence pattern (Phase 13.16.DF FIX1):** Implementation-time discoveries that improve correctness beyond spec are acceptable when (a) discovered via source-read, (b) inline-documented with rationale, and (c) disclosed in the Review Request deviations table
11. **Three-level test coverage for cross-subproject features (Phase 13.16.DF FIX1):** Unit-level (dfdraw), integration-level (ADF K2 suite), production-pattern level (synthetic mirror of architect's reproducer) — full pipeline validated
12. **Dual-path dispatch over signature-merging (Phase 13.31.DF):** When extending a parameter's meaning (e.g., `facet_by` from channel-name to also accept column-name), keep the two paths structurally separate in dispatch instead of widening one parameter's signature. Preserves backward-compat regression tests and makes mutual-exclusion guards trivial
13. **Multi-kind plot dispatch with explicit signature branching (Phase 13.32.DF Sub-fix 3):** When a coordinator like `_dispatch_faceted_render` must invoke multiple `draw_*` functions with different signatures, branch the per-subplot call explicitly per plot kind rather than rely on `**kwargs` pass-through. `auto_title` (profile/hist/hist2d only, not scatter), `quantiles`/`quantile_mode` (profile only), and `group_by`/`top_k` (no hist2d) each leak via `**kwargs` if not gated
14. **AST-level R6-equivalent pre-delivery check (Phase 13.32.DF, proposed Coder QRC v1.32):** When the drafter edits any `_*_FORWARDED_NAMES` tuple, simulate the R6 validator (Phase 13.16 FIX1) locally against the AST of the target file before delivery. Caught Bug 0 in Phase 13.32 only after a failed import; should have been caught at drafter time
15. **Auditable patches over local sed (Phase 13.32.DF, proposed Coder QRC v1.32):** All file edits flow through `present_files` artifacts so the patch is visible in the chat transcript. Local `sed` instructions, however minimal, leave no audit trail and risk divergence between drafter intent and architect-applied result

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-14 | Main Reviewer | Initial PHASE_HISTORY.md from git log |
| 1.1 | 2026-01-29 | Claude-Main | Added Phase 13.6.G.DF (stats enhancements) |
| 1.2 | 2026-03-28 | Claude41 | Added Phases 13.12.DF, 13.13.DF, 13.14.DF; interval sort fix; updated test count to 399 |
| 1.3 | 2026-04-09 | Claude41 | Added Phase 13.15.DF (test infrastructure) and Phase 13.16.DF (vector expression interface, AD-37 fix); updated test count to 451; added 7 lessons learned from Rev2→Rev3 cycle and governance incidents; added source verification discipline and scaffolding-separation best practices |
| 1.4 | 2026-04-15 | Claude41 | Added Phase 13.16.DF FIX1 (vector path kwarg propagation fix, B1a-B5 + R4 + auto_title forwarding); updated test count to 469; +3 features +7 invariance tests +1 Verified; added 5-iteration source-verification chain (symptom → location → documentation → runtime → pipeline); cross-subproject end-to-end verification via ADF Phase 13.19.ADF.FIX1; added 4 lessons learned (two-commit pattern, fresh-reviewer rule, cross-subproject convergence, class-load validation) and 4 best practices (forwarded-name tuples, AST-derived inventories, scope-positive divergence pattern, three-level test coverage) |
| 1.5 | 2026-05-09 | Claude49Coder | Backfill of phases that landed between v1.4 and current state. Added Phase 13.18.DF (robust statistics extension), Phase 13.25.DF v1.3 with FIX1 + FIX2 (Quantiles on Profile — MultiGraph Phase A; AD-44..AD-54), Phase 13.26.DF v1.2 (N-Channel Framework — MultiGraph Phase B; Algorithm A; AD-55..AD-60), Phase 13.27.DF Commit 1 (Facet refactor — MultiGraph Phase D, profile-only; AD-61..AD-68), Phase 13.28.DF v1.1 (Robust Data Handling — `sanitize_for_plot` + hybrid autorange; AD-69..AD-77; 5-0 closure verdict). Statistics table updated through Phase 13.27.DF Commit 1 (current 663 tests, 62 features). Governance principles GP-1 through GP-5 summarized in their phase-of-origin sections (full text remains in STYLING_FRAMEWORK_DECISIONS.md §3). |
| 1.6 | 2026-05-15 | Claude49Coder | Added Phase 13.28.DF FIX1 (autorange.* style key registration; commit `57576ebf`), Phase 13.30.DF v1.0 (Class-2 column-reference parameter validation; commit `e8278531`), Phase 13.31.DF v1.0 (`facet_by` column-name support, AD-78; commit `f3ca432a`), Phase 13.32.DF v1.0 (`group_by × quantiles` in grouped path + symmetric `facet_by` binning, AD-79; commit `cb6a1aed`). Test count 663 → 715. Added 3 lessons learned (style-key registration regression class, source freshness as binding rule, wrong-bundle review artifact) and 4 best practices (dual-path dispatch, multi-kind plot dispatch with explicit signature branching, AST R6-equivalent pre-delivery check, auditable patches over local sed). Statistics Summary table extended with 4 new rows + phase-ordering note. Panel review (Claude40 consolidating Sonet50, Sonet51, Sonnet52_R1, Sonnet53_R2, Claude46, Claude48): approved-with-3-mechanical-fixes — applied pre-commit: (a) Phase 13.30 Class-2 tuple corrected to `('group_by',)` only and Class-1 → Class-3/Class-5 deferral; (b) Phase 13.26 → Phase 13.28 autorange-introduction attribution (FIX1 entry + Lesson #9); (c) Statistics table totals updated to 28 phase entries / 715 tests; transient "Pending push" line removed. |
| 1.7 | 2026-05-21 | Opus1 (Reviewer) at architect request | **Backfill of 9 phase events that landed between Phase 13.32.DF v1.0 closure (`cb6a1aed`, 2026-05-15) and current HEAD (`2f4d959f`, 2026-05-20).** Added strictly append-only — every existing entry preserved verbatim per architect's "Previous coders removed history, which was completely wrong" directive. New H2 sections (in phase-number order, inserted before § Statistics Summary): Phase 13.27.DF Commit 2 v1.0 (`84dcf916`, +52 tests, Phase D completion: `selection_vector` + `weights_vector` + `delta_facet`), Phase 13.27.DF Commit 2 FIX1 (`ba42fcde`, +9), Phase 13.27.DF Commit 2 FIX1.FIX1 (`b929ccb9`, +1), Phase 13.33.DF v1.0 M1 (`61460df5`, +22, Normalized differential profiles, AD-80/81/82), Phase 13.33.DF v1.0 M2 (`c6a3245f`, +5, group_by/facet_by composition), Phase 13.33.DF v1.0 FIX1 (`94594f89`, +1, tag `PHASE_13_33_DF_v1_0_FIX1_END`), Phase 13.32.DF FIX1 (`195ab4ea` / `d0b04f88`, +4, BUG-001/002/003 faceted rendering bugs caught in real-data TPC/ITS QA, tag `PHASE_13_32_DF_FIX1_END`), Phase 13.34.DF v1.0 (`463deb36` / `abf5fe40`, +8, Capability Matrix taxonomy refresh + M2 robustness gaps, tag `PHASE_13_34_DF_END`), Phase 13.34.DF FIX1 (`379f26bd` + `14851d42`, +5, BUG-010 untracked test file, tag `PHASE_13_34_DF_FIX1_END`), Phase 13.34.DF FIX2 (`b38395db`, +0, BUG-011 run_tests.sh pre-bundle staging check, tag `PHASE_13_34_DF_FIX2_END`), Phase 13.35.DF v1.3 (`3b910aec`, +11, `group_by_bins` + `hist_norm` for `hist()`, BUG-013 hist side, tag `PHASE_13_35_DF_END`), Phase 13.36.DF v1.2 (`2f4d959f`, +10, user style kwargs override auto-cycle, BUG-013 style-override side, tag `PHASE_13_36_DF_END`). Test count 715 → **843**. Statistics Summary table extended with 13 new rows (one per phase event) + extended phase-ordering note covering the non-monotonic commit order from 2026-05-16 onward. Totals updated: 28 → 37 phase entries; 715 → 843 tests; 62 → 90 features; 28+ → 193 invariance tests; 7 → 33 Verified features. Source: `gitlog.txt` (commits cb6a1aed..2f4d959f), `reviewer_20260521_092254.zip` (843/0/1 confirmed at HEAD), `CAPABILITY_MATRIX_20260521_092254.md` (90 features / 33 Verified confirmed). Note: backfilled entries derive from commit messages (verbatim phrasing preserved where present); each new entry cites its commit hash and tag per Org v1.30 § Source-Line Evidence Standard `[MUST]`. No existing line of this document was removed or shortened. Standalone review of this PHASE_HISTORY backfill not performed — architect-directed governance closure, awaiting panel review. |
| **1.8** | **2026-05-21** | **Sonet50 (consolidated panel review)** | **Added 4 new phase sections (Phases 13.37.DF v1.1, 13.37.DF FIX1, 13.38.DF v1.1, 13.39.DF v1.2) and 4 new Statistics Summary rows. Strictly append-only. Commits: `67fccf3d` (13.37), `095d6f28` (13.37 FIX1), `0f525743` (13.38), `b024414e`+`3c5d4547` (13.39). Test count 843 → 913 (+70). Verified 33 → 47 (+14). Invariance 193 → 239 (+46). Features 90 → 105 (+15). Phase entries 37 → 41 (+4). Sources: gitlog.txt (commits `2f4d959f`..`3c5d4547`), session approval summaries (Sonet50_PHASE_13_37/38/39_*_ReviewSummary_AllReviewers_20260521.md), CAPABILITY_MATRIX.md (47 Verified / 239 invariance / 913 tests confirmed). All pre-existing content preserved verbatim.** |
| **1.9** | **2026-05-27** | **Claude48 (coder seat) at architect request** | **Backfill of 6 phase events that landed between Phase 13.39.DF v1.2 closure (`3c5d4547`, 2026-05-21) and current HEAD (`28f7f3ce`, 2026-05-27). Added strictly append-only — every existing entry preserved verbatim per architect's append-only directive. New H2 sections (chronological commit order, inserted before § Statistics Summary): Phase 13.40.DF v1.0 (`67d125e2`, +10, Cumulative histogram `cumulative=True/-1/False`; ROOT `TH1::Draw("cumulative")` equivalent; 4 call sites threaded incl. `_dispatch_faceted_render`; M5 `hist_errors+cumulative` NotImplementedError guard; tag `PHASE_13_40_DF_END`), Phase 13.41.DF v1.0 (`530954d1`, +19, N-D Faceting via `facet_by=List[str]` for 1D/2D/3D; ROW/COL/FIGID convention LOCKED; 3D returns `(List[Figure], List[axes_2d], List[stats_dict])`; `share_x`/`share_y`/`share_across_figures` new params; dfdraw FIRST major plotting library with unified Nth-dimension-figure API; tag `PHASE_13_41_DF_END`), Phase 13.41.DF FIX1 (`b84576a0`, +3, 3 bugs from v1.6 panel; tag `PHASE_13_41_DF_FIX1_END`), Phase 13.41.DF FIX2 (`70b94a3e`, +1, 5 P2/P3 items + FBY.23 lock; tag `PHASE_13_41_DF_FIX2_END`; gate 946), Phase 13.42.DF v1.0 (`38aed2d8`, +27, Inline fits `fit=` parameter on hist/profile/scatter/draw; new `plots/fits.py` registry + `plots/_fit_render.py`; str/dict/callable/list forms; 7 fit.* style keys; stats integration; group_by/facet_by/vector composition; Sonnet54 P1-B fixed pre-tag for profile grouped path; tag `PHASE_13_42_DF_END`; gate 973), Phase 13.42.DF FIX1 (`82aaa903` + `28f7f3ce`, +8, Production-gate bug closure + interface lock; 7 production-gate bugs B1-B7 surfaced within 30 minutes of real TPC ITS-TPC calibration data testing; 5 P1 silently-wrong-output bugs that 5 reviewers + 27 invariance tests missed; D-1 [BREACH] use_errors default flip + D-2 [BREACH] `_style_get` broken since Phase 13.42 v1.0 → ALL `fit.*` style keys silently ignored; D5 vector pairing per v1.4 §6.3 verbatim; D9/R4 stacked+group_by+fit per-group dict; new `fit_textbox_kwargs={'fontsize','format','show_fields'}` LOCKED at close; F.28-F.33 + F.28b (8 new tests); FIT.inline 27 → 35; Sonet50 CRR `[X]` REVISION_REQUESTED → CRR v2 P1-A `np.array(shape=) → np.zeros((0,0))` + P1-B taxonomy staging; THIRD consecutive phase to miss taxonomy staging — Sonet50 governance note recommends `run_tests.sh` pre-bundle taxonomy-count check; tag `PHASE_13_42_DF_FIX1_END`). Test count 913 → **981** (+68 across 6 phases). Verified 47 → 50. Invariance 239 → 307 (+68). Features 105 → 108 (+3). Phase entries 41 → 47 (+6). Statistics Summary table extended with 6 new rows + new phase-ordering note for the post-13.39 window. Sources: gitlog.txt (commits `3c5d45474dcbdd0683969adde76342fa904f5059`..`28f7f3ce640c2c3a0b6b839ddde8ad171ca16c73`), CAPABILITY_MATRIX.md (50 Verified / 108 features / FIT.inline 35 / FACET.list_grid 23 / HIST.cumulative 10 confirmed), `PHASE_13_42_DF_PROD_GATE_Bugs_v1_0.md`, `PHASE_13_42_DF_FIX1_v1_2_Proposal.md`, `PHASE_13_42_DF_FIX1_Code_Review_Request_v1.md`. Phase 13.42.DF FIX1 process-improvement audit deliverables (`PHASE_13_42_DF_POST_GATE_Audit_Questions_v1_0.md`, `Claude48_Feedback_to_Organization_Team_20260526.md`) shipped to Org team for QRC #10 + production-gate policy adoption. All pre-existing content preserved verbatim per append-only directive.** |
| **1.10** | **2026-05-28** | **Claude48 (coder seat) at architect request** | **Backfill of 5 phase events that landed between the v1.9 doc commit (`e463161c`, 2026-05-27) and current HEAD (`ad91e251`, 2026-05-28). Strictly append-only — every existing entry preserved verbatim. New H2 sections (chronological commit order, before § Statistics Summary): Phase 13.42.DF FIX2 (`79d449c3`, +6, close 5 items deferred at FIX1: B6/B7/I-8/ADV-1/ADV-3; F.59-F.63+F.61b; FIT.inline 35→41; tag `PHASE_13_42_DF_FIX2_END`, gate 987 — landed AFTER the v1.9 doc commit so v1.9 did not record it), Phase 13.43.DF v1.0 (`0e0d79f7`, +27, `summary_fit` standalone fit-result figures; new `plots/_summary_fit.py`; 13 `summary_fit.*` keys; F.34-F.56 + R-2/F.56c END fix for scalar-delegation drop of fit/fit_textbox_kwargs/summary_fit; FIT.summary feature; tag `PHASE_13_43_DF_END`, gate 1014 — commit body states pre-R-2 1013/+26), run_tests.sh PHASE_HISTORY↔git-tag drift check (`02510a20`, tooling-only, gate 1014; surfaced+resolved the Phase 13.25 tag incident; `PHASE_13_46_DF_BEGIN` placed here), Phase 13.46.DF v1.0 (`1d77702e`, +8, audit bucket ① C-1/C-2/C-4/C-7/C-9; F.64-F.70; +4 features; §2.1 Option-1 shared-global ruling; closure tag is FIX1_END — no separate v1.0 END tag; gate 1022), Phase 13.46.DF FIX1 (`ad91e251`, +1, scatter range= point-filtering — removes out-of-range points per architect 2026-05-28; F.71; +1 feature RANGE.scatter_filter; tag `PHASE_13_46_DF_FIX1_END`; rolling `PHASE_BEGIN_dfdraw` → `ad91e251`; gate 1023). Test count 981 → **1023** (+42 across 5 phases). Verified 50 → 56. Invariance 307 → 349 (+42). Features 108 → 114 (+6). Phase entries 47 → 52 (+5). Statistics Summary table extended with 5 new rows + post-13.42-FIX1 phase-ordering note. Overview header updated to Phase 13.46.DF FIX1 / 1023 / 114 / 349 / 56. Sources: git.log (commits `28f7f3ce`..`ad91e251`), CAPABILITY_MATRIX.md (56 Verified / 114 features / 578 proof / 349 invariance at HEAD `ad91e251`), `PHASE_13_46_DF_v1_3_AuditFixes_Proposal.md`, `PHASE_13_46_DF_FIX1_Code_Review_Request.md`, `PHASE_13_45_dfdraw_Audit_Findings.md`. New run_tests.sh tag-drift check passes after this backfill (the 5 tags were the intentionally-non-blocking repo-ahead-of-doc transient). All pre-existing content preserved verbatim per append-only directive.** |
| **1.11** | **2026-06-03** | **Opus2 (Reviewer) at architect request; Sonnet53_R2 panel P2-1 amendment applied pre-commit** | **Backfill of 7 phase events that landed between Phase 13.46.DF FIX1 closure (`ad91e251`, 2026-05-28) and current HEAD (`07606c02`, 2026-06-03). Added strictly append-only — every existing entry preserved verbatim per architect's append-only directive. New H2 sections (chronological commit order, inserted before § Statistics Summary): `run_tests.sh` tag-drift WARN downgrade (`df3057a3`, tooling-only, gate 1023 unchanged — heuristic-gate hard-block → non-blocking WARNING in SUMMARY artifact; scope narrowed to declarative tag references only; **added per Sonnet53_R2-led panel P2-1 amendment** after initial v1.11 draft omitted this commit, parallels v1.10's earlier `02510a20` tag-drift addition), Phase 13.48.DF v1.0 (`9f612601`, +11, Tier-1 automated visual testing framework `VisualCheck` + 10 V-checks V.1-V.10 + V.2 ragged-padding-safety lock, new `visual_primitive` test layer, +6 `VISUAL.*` features Smoke-only at close, Tier 1 / Tier 2 split established; tag `PHASE_13_48_DF_END`, gate 1034), Phase 13.49.DF v1.0 (`89bc63c6`, +4, Capability Matrix Traceability link infrastructure + HTML rendering + orthogonal Visual column with 👁 badge + M.1-M.4 meta-tests + KNOWN_UNCLAIMED §3.7 governance with 64 seeded entries SPECIFIC target_phases, D-K + D-L normalization fixes after Opus2 [X] rejection of v1.0 CRR for `startswith("tests/")` regression; tag `PHASE_13_49_DF_END`, gate 1038, +1 feature META.capability_matrix, Verified 56→57), Phase 13.49.DF FIX1 (`a6ddc753`, +0 same-test stricter, HTML rendering fixes H-1/H-2/H-3 after architect rendered the HTML in browser and found 3 bugs the 8-reviewer v1.1 panel missed; M.3 extended with category-uniqueness + MD↔HTML count agreement invariants; direct trigger for Reviewer QRC v1.31 Rule 14 caveat; tag `PHASE_13_49_DF_FIX1_END`, gate 1038), Phase 13.50.DF v1.0 (`f3034153`, +19, fit-rendering overhaul proposal v2.5 after 4-round panel convergence v2.2→v2.3→v2.4→v2.5; six features `FIT.display_names`/`FIT.precision_modes`/`FIT.textbox_kwargs_extensions`/`LEGEND.modes`/`SUMMARY_FIT.placement`/`SUMMARY_FIT.orientation`; two [BREACH]es `fit.text_format` style key + `summary_fit.precision` int field both architect-authorized clean removals; F1-F18 visual_primitive + F17 cross-variant equivalence + F18 4-tuple legend_topology + normalizer idempotency; Claude48 introduced **§0 pre-CRR gap-audit attestation pattern** proposed as Coder QRC R17 after architect's "Was all functionality from spec implemented?" challenge caught 4 undisclosed partials in v1.0-draft; F19 textbox-bbox-overlap deferred to Tier 2 Phase 13.5X — the test that proves the architect-flagged motivating spacing bug is cured; tag `PHASE_13_50_DF_END`, gate 1057, +6 features Verified 57→59 visual_primitive 11→27 invariance 353→356), Phase 13.50.DF FIX1 (`5010cf78`, +0, P2-1 stale-taxonomy fix `SUMMARY_FIT.orientation` "name" field + scope + two `_summary_fit.py` mop-up sites via parallel-sweep follow-on; CRR v1.0 documentation corrections P3-1/P3-2/P3-3; QRC backlog items 5+6 added — R17 EXTENSION parallel doc-surface sweep + §4-facts-from-logs rule; tag `PHASE_13_50_DF_FIX1_END`, gate 1057), Phase 13.50.DF FIX2 (`07606c02`, +0, `run_tests.sh` HTML packaging enforcement closing P2-2 after 3 consecutive recurrences; mechanical guard `unzip -l ... | grep -q '\.html'` non-fatal; lesson recorded as governance pattern — 3 consecutive recurrences is the threshold for infrastructure enforcement vs another flag-and-promise cycle; tag `PHASE_13_50_DF_FIX2_END`, gate 1057). Test count 1023 → **1057** (+34 across 7 phase events). Verified 56 → 59 (+3). Invariance 349 → 356 (+7: +4 from 13.49 M-tests + 3 from 13.50 F17/F18/normalizer). **visual_primitive layer NEW (introduced in 13.48): 0 → 27** (+11 from 13.48 V-checks + 16 from 13.50 F1-F16). Features 114 → 127 (+13: +6 VISUAL.* from 13.48 + 1 META from 13.49 + 6 FIT/LEGEND/SUMMARY_FIT from 13.50). Phase entries 52 → **59** (+7). Statistics Summary table extended with 7 new rows + post-13.46-FIX1 phase-ordering note. Overview header updated to Phase 13.50.DF FIX2 / 1057 / 127 / 356 / 59. Sources: gitlog.txt (commits `ad91e251`..`07606c02`), reviewer.zip artifacts from each phase, `PHASE_13_50_DF_CRR_v1_0.md` + `PHASE_13_50_DF_FIX1_CRR_v1_0.md` (substantive content), `Sonnet53_R2_PHASE_13_50_DF_CRR_v1_0_Summary_Review_20260603.md` (panel verdict), `PHASE_13_49_DF_v1_2_CapabilityMatrixTraceability_Proposal.md`, `PHASE_13_48_DF_v1_4_VisualTesting_Proposal.md`, `PHASE_13_49_DF_FIX1_Code_Review_Request.md`, **Sonnet53_R2-led 3-reviewer panel verdict on Diff A vs Diff B (2026-06-03, [!] APPROVED WITH COMMENTS): Diff B (Opus2) adopted as base; P2-1 (`df3057a3` event missing) applied pre-commit; P2-2 (`VISUAL.*` feature IDs in Phase 13.48 section may diverge from canonical IDs in `feature_taxonomy.py` at HEAD `07606c02` — flagged as source-verification gate before final commit; if mismatch, one-line `sed` correction will be shipped without re-versioning)**. Author note: Opus2 reviewed each of these phases as a panel member in real time (13.48 CRR v1.0, 13.49 v1.0 [X] reject + v1.1 [!] approve + FIX1 [!], 13.50 v2.2/v2.3/v2.4/v2.5 + CRR v1.0 + FIX1 [OK]) — not reconstructing from commit messages alone. All pre-existing content preserved verbatim per append-only directive.** |

| **1.12** | **2026-06-09** | **Sonnet65 (Reviewer) at architect request** | **Backfill of 9 phase events that landed between Phase 13.50.DF FIX2 closure (`07606c02`, 2026-06-03) and current HEAD (`9a950c7b`, 2026-06-09). Added strictly append-only — every existing entry preserved verbatim per architect's append-only directive. New H2 sections (chronological commit order, inserted before § Statistics Summary): `docs: ARCHITECT_DECISIONS.md v1.0.1` (`ee698002`, tooling/doc-only, gate 1057 unchanged — Org panel touch-ups: GP section transitional authority, Date+Scope fields on all 8 GPs, AD-37/AD-50 relationship, v1.0.1), `docs: ARCHITECT_DECISIONS.md v1.1.0` (`7c1c5a7c`, doc-only, gate 1057 — GP-9 Archive Substantive Deliberation + AD-1/TS_v6.DF Executive Summary structure + AD-2/TS_v6.DF Quantification Deferral; panel Sonnet56/57 [!]), `docs: TS v6.3` (`5c9284c5`, doc-only, gate 1057 — thesis-first Executive Summary restructure; 8 subsections; GP-3 verbatim blocks; ×2.4 σ(pT)/pT; Stonebraker 2024 citation; 6+6 internal+external cross-team panel; AD-1+AD-2 governing future TS revisions), `docs: COMPARISON v2.1` (`e926eae3`, doc-only, gate 1057 — stack-identity restructure + cross-team panel-fix pass; 4 P1 runtime-breaking fixes: summary_fit=True→'table', range_x/y→range=, legend_stats_fields→stats=, autorange attribution 13.36→13.28), `docs: MOTIVATION.md` (`5d2d23bc`, doc-only, gate 1101 — permanent canonical motivation tracking file; GP-3 verbatim quotes preserved; cites ARCHITECT_DECISIONS.md GP-9; §7 physics analysis section marked [DRAFT — pending brainstorm session]), Phase 13.51.DF v1.5 (`b1db4740`, +21, post-audit fix pass: Batch 1 9/15 TS doc corrections + Batch 2 R-2 explicit forwarding all 4 draw() dispatch branches + `profile.py:879`/`:907` `bin_means`→`_central_values` V-3+P1-B + datetime64 guard at `compute_autorange()` entry V-2 + `draw_profile2d` conditional `central=` forward F-1 + Batch 3 `profile2d()`/`scatter3d()` wrappers S-5 + hexbin dispatch S-11 + `hist2d` `time_format=` S-8 + clean ValueError guards S-7/S-8; 23 tests in `test_phase_13_51_post_audit.py`; panel Sonnet65 [!] 5/5; tag `PHASE_13_51_DF_END`, gate 1078+FIX1=1080 architect env), Phase 13.51.DF FIX1 (`2235caef`, +2, hexbin dispatch silent-drop closed — Opus48_3 executed negative control caught that `draw(type='hexbin', facet_by=)` silently dropped modifier while direct `hexbin(facet_by=)` raised; root cause `_hexbin_allowed` filter stripped params before S-7 guards; fix: explicit `facet_by=facet_by` + `range=kwargs.get('range',None)` forwards; builtin-shadow bug self-caught during fix; T16b+T17b added; `KNOWN.hexbin_dispatch_residual_drops` registered for Batch 4; tag `PHASE_13_51_DF_FIX1_END`, gate 1080→1082 architect env), Phase 13.52.DF v1.5.1 (`9a950c7b`, +22 tests total 21 pass+1 skip, declarative overlay dual-surface engine: `overlay(expr, layers=[...])` + `draw(type="hist2d+profile")` sugar; `_OVERLAY_DENSITY`/`_OVERLAY_ALLOWED`/`_OVERLAY_GUARD_PARAMS` constants; `_overlay_kw()` helper W-1; all 6 guards hoisted before draw; range lock `set_xlim/ylim` after each overlay layer; z-order base-first; T-S2 negative control W-4; T1p documented skip profile2d-base shared-expr limitation; 6 `OVERLAY.*` features registered; [BREACH] `_OVERLAY_GUARD_PARAMS` ratified by architect "I have not yet used 2D histo fit so I do not need back compatibility"; P1-A `_OVERLAY_ALLOWED` whitelist enforced; P1-B sugar splice extended to `selection_vector`/`weights_vector`/`nan_policy`/`color`/`size`/`marker`; panel Sonnet65 [X]→[!] after v1.5.1 fixes; AD-N/13.52.DF added to ARCHITECT_DECISIONS.md; tag `PHASE_13_52_DF_END`, gate 1101/0/2). Test count 1057 → **1101** (+44 across 9 events: +21 Phase 13.51 + +2 FIX1 + +21 Phase 13.52 incl. skip). Verified 59 → 59 (unchanged — new tests are smoke/guard class; CAPABILITY_MATRIX feature claim registration pass pending before distribution). Invariance 356 → 356 (unchanged). Features 127 → **133** (+6 OVERLAY.*). Phase entries 59 → **68** (+9). Statistics Summary table extended with 9 new rows. Overview header updated to Phase 13.52.DF v1.5.1 / 1101 / 133 / 356 / 59. Sources: git.log (commits `07606c02`..`9a950c7b`), `reviewer_20260609_083733.zip` (Phase 13.51 CRR bundle, 1078/0/1), `reviewer_20260609_114847.zip` (Phase 13.52 CRR bundle, 1098/0/2→1101 post-v1.5.1), `Sonnet65_PHASE_13_51_DF_CRR_PanelSummary_20260609.md`, `Sonnet65_PHASE_13_52_DF_CRR_PanelSummary_20260609.md`, CAPABILITY_MATRIX.md (133 features / 59 Verified at HEAD `9a950c7b`). All pre-existing content preserved verbatim per append-only directive.** |
| **1.13** | **2026-06-10** | **Opus1 (coder seat) at architect request** | **Backfill of 3 phase events between Phase 13.52.DF v1.5.1 closure (`9a950c7b`, 2026-06-09 13:43) and Phase 13.54.DF closure (`348fddec`, 2026-06-10 10:52) PLUS `tests/test_layer_classification.py` gate-4 closure applied AS PART OF THIS COMMIT. Added strictly append-only — every existing entry preserved verbatim per architect's append-only directive. Events covered, in chronological commit order: (1) Phase 13.52 distribution prep #1 (`8f6f93f8`, 2026-06-09 15:05, doc-only, gate 1101 unchanged — 10 feature claims registered in `tests/feature_taxonomy.py` for the 23 tests already landed in `test_phase_13_51_post_audit.py`: 5 Verified-target features (`DRAW.R2_forwarding`, `PROFILE.central_median_1d`, `PROFILE2D.central_median_mesh`, `PROFILE.central_median_fit`, `AUTORANGE.datetime64_guard`) plus 5 VISUAL.* features (`VISUAL.facet_r2_profile/hist/scatter/hist2d`, `VISUAL.hist2d_datetime_labels`); features 133 → 143; ADF reviewer pre-distribution finding; closed Phase 13.51 spec §5.1 gate F-13 (≥5 new features) and partially closed gate 7 (VISUAL.* ≥9 reached 11); however Verified count stayed at 59 because the matching `test_layer_classification.py` entries were NOT included in the commit — that omission is closed AS PART OF THIS COMMIT (see end of this row)); (2) ADF source fix `7906cdfd` (2026-06-10 09:46, ADF team commit "ADF: source fix for BUG_20260609_lazy_nd_facet (paired with a52f5522)" — not a dfdraw change; listed for cross-team traceability as the closure of `PHASE_13_53_ADF_TS_DRAW` audit thread); (3) Phase 13.54.DF (`348fddec`, 2026-06-10 10:52, +6 tests, gallery-found bug fixes triggered by AD-TS-DRAW-001 ADF `time_series_draw.py` gallery validation on 2026-06-10: BUG_dfdraw_20260609_scatter_auto_title closed via 3-site fix in `plots/scatter.py` + `drawer.py` per Sonnet58/Sonnet62 panel finding (signature + body + R-2 forward + faceted forward); BUG_dfdraw_20260610_hist2d_time_format_epoch closed via epoch-second elif branch mirror of `draw_hist:L451-453` at 4 conversion sites in `plots/histogram.py`; 6 invariance tests T1-T6 in `test_phase_13_54_df_gallery_fixes.py`; 2 new feature claims `SCATTER.auto_title` + `HIST2D.time_format_epoch`; features 143 → 145; cross-product §6.1 enumeration executed 12 combinations per fix per AD-TS-DRAW-001 discipline lesson; panel Sonnet65 6-reviewer [!] approved (2 administrative corrections only: Verified count text + git-add staging); gallery validation 31/31 mandatory clean (fig04 + fig16 now passing); follow-up filed P2 BUG_dfdraw_20260610_hist2d_y_axis_overreach (y-axis elif over-converts non-time integer columns when `time_format=` is set; surfaced by fig16 visual check on real data) scheduled for Phase 13.55.DF; tag `PHASE_13_54_DF_END`, gate 1101 → 1107 architect env). PLUS `tests/test_layer_classification.py` gate-4 closure applied AS PART OF THIS COMMIT — 12 new entries (7 invariance: T3, T5, T9a, T9b, T9c, T10b, T10c; 5 visual_primitive: T2, T4, T6, T8, T14) close CAPABILITY_MATRIX gate 4 (Verified ≥64). Phase 13.51 audit feature claims were registered in `8f6f93f8` without matching layer-classification entries, leaving 5 features (`DRAW.R2_forwarding`, `PROFILE.central_median_1d`, `PROFILE2D.central_median_mesh`, `PROFILE.central_median_fit`, `AUTORANGE.datetime64_guard`) at Smoke-only when their tests actually contain A≡B / explicit-value assertions. With this update those 5 features promote to Verified. Test count 1101 → **1107** (+6 from Phase 13.54). Verified 59 → **64** (+5 reclassification, **gate 4 CLOSED**). Invariance 356 → **363** (+7 reclassification). visual_primitive 27 → **32** (+5 reclassification). Features 133 → **145** (+10 distribution prep + +2 Phase 13.54). Phase entries 68 → **71** (+3 across this v1.13). Overview header updated to Phase 13.54.DF / 1107 / 145 / 363 / 64. Sources: gitlog.txt (commits `9a950c7b`..`348fddec`), `Sonnet65_PHASE_13_54_DF_CRR_PanelSummary_20260610.md` (6 reviewers, [!] APPROVED with 2 administrative corrections), `Sonnet65_GalleryBugReports_PanelSummary_20260610.md` (10 reviewers, [!] APPROVED), `AD-TS-DRAW-001_Architect_Decision.md` (gallery as pre-tag mandatory validation gate), ADF gallery validation log 2026-06-10 (31/31 mandatory clean), CAPABILITY_MATRIX.md regenerated after this commit (145 features / 64 Verified / 363 invariance / 32 visual_primitive). All pre-existing content preserved verbatim per append-only directive.** |

---

**Document Status:** Updated through Phase 13.54.DF + gate-4 closure (Phase 13.54 at commit `348fddec`, tag `PHASE_13_54_DF_END`, 2026-06-10; gate 1107/0/2; Verified moved 59 → 64 via `test_layer_classification.py` update applied with this v1.13 commit). Rolling tag `PHASE_BEGIN_dfdraw` → `348fddec`. **Previous "Updated through Phase 13.52.DF v1.5.1", "Updated through Phase 13.50.DF FIX2", "Updated through Phase 13.46.DF FIX1", "Updated through Phase 13.42.DF FIX1", and "Updated through Phase 13.39.DF v1.2" baselines preserved verbatim above for audit traceability per architect's append-only directive.**
**Next Update:** After ADF time_series tests workstream, or CAPABILITY_MATRIX feature claim registration pass (PHASE_13_51 gates 4+7 — Verified ≥64, VISUAL.* ≥9 — pending before distribution to 5 audiences).
