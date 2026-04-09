# dfdraw - Phase History

## Overview

This document tracks the development history of the `dfdraw` module, a DataFrame drawing utility with ROOT TTree::Draw-like interface. Part of the dfextensions toolkit for ALICE experiment calibration and QA at CERN.

**Current Status:** Phase 13.16.DF v1.0 - Vector Expression Interface  
**Test Count:** 451 passing (43 features, 21 invariance tests, 6 Verified)  
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

**Total Development:** 15 phases, 451 tests, 43 features, 21 invariance tests, 6 Verified

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
2. **Test-first approach:** 451 tests caught regressions early
3. **Duck typing:** Clean integration without hard dependencies
4. **Style system:** Established early, avoided later refactoring
5. **Governance process:** Proposal → review → implement → test cycle caught issues before production
6. **Multi-reviewer source verification (Phase 13.16.DF):** External reviewers catching 6 P0 defects that internal approvers missed proved the Rev2→Rev3 cycle works as designed
7. **Strong A≡B invariance tests (Phase 13.16.DF):** Byte-identical axes comparison catches divergences at unit-test level instead of real-data level

### What Could Improve
1. **Earlier integration testing:** ADF `draw_figures()` duplication discovered late
2. **Documentation cadence:** Should update with each phase (codified in Org v1.24 § Update Discipline; Phase 13.16.DF is the first phase to apply this rule)
3. **Performance profiling:** Should have benchmarked earlier phases
4. **Verbose debug mode:** Would have caught the ADF defaults cascade issue faster
5. **Source verification discipline (Phase 13.16.DF):** Proposal enumeration alone is insufficient for shared-state changes — reviewers must count call sites in source
6. **Scaffolding separation (Phase 13.16.DF):** `run_tests.sh` and similar infrastructure should not share commits with feature work

### Best Practices Established
1. **Expression syntax:** ROOT-like syntax reduces learning curve
2. **Return tuples:** `(fig, ax, stats)` consistent across all methods
3. **Keyword-only args:** After first positional, all kwargs for clarity
4. **Graceful degradation:** Optional dependencies handled cleanly
5. **Option hierarchy:** More local wins (kwargs < batch < group < plot)
6. **Byte-identical invariance tests:** The quality bar for phases touching shared state (line count, colors, linestyles, xdata/ydata to 10 decimals + stats to 1e-9)
7. **Vector expressions over loops:** For N-series overlays where color/label continuity matters, vector syntax (`[y1,y2]:x`) beats scalar loops with `same=True` — especially across ADF boundaries

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-01-14 | Main Reviewer | Initial PHASE_HISTORY.md from git log |
| 1.1 | 2026-01-29 | Claude-Main | Added Phase 13.6.G.DF (stats enhancements) |
| 1.2 | 2026-03-28 | Claude41 | Added Phases 13.12.DF, 13.13.DF, 13.14.DF; interval sort fix; updated test count to 399 |
| 1.3 | 2026-04-09 | Claude41 | Added Phase 13.15.DF (test infrastructure) and Phase 13.16.DF (vector expression interface, AD-37 fix); updated test count to 451; added 7 lessons learned from Rev2→Rev3 cycle and governance incidents; added source verification discipline and scaffolding-separation best practices |

---

**Document Status:** Updated for Phase 13.16.DF v1.0 completion  
**Next Update:** After Phase 13.16.DF real-data validation and `PHASE_13_16_DF_v1_0_END` tag, or next feature phase
