# Plotting Library Comparison for dfdraw Wrapper Design

**Date:** 2026-04-09 (Phase 13.17.DF refresh)
**Original:** 2026-01-29 (Phase 13.6.G.DF initial version)
**Purpose:** Inform wrapper design for dfdraw with potential RootInteractive backend
**Author:** Claude2-dfdraw (original), Claude40 (Phase 13.17.DF status refresh)

---

## Executive Summary

This document compares dfdraw with other Python plotting libraries and ROOT to inform the design of a unified wrapper interface. The goal is to preserve ROOT's best features (one-liner syntax, rich statistics) while leveraging Python's ecosystem and avoiding ROOT's pain points.

**Update (Phase 13.6.G.DF, 2026-01-29):** Statistics enhancements complete. dfdraw has full ROOT parity for statistics features.

**Update (Phase 13.16.DF, 2026-04-09):** Vector expression interface complete. Bracket-vector syntax (`"[y1,y2,y3]:x"`) provides single-call multi-series overlay, resolving the AD-37 AliasDataFrame color-cycling bug via architectural fix. 7 strong A≡B invariance tests establish the new testing quality bar.

**Update (Phase 13.17.DF, 2026-04-09):** This document refreshed to reflect Phase 13.12–13.16 capability additions. Status tables updated; backend abstraction section annotated with current RootInteractive integration status. No structural changes; strategic/architectural framing preserved from original.

---

## Library Landscape

### Category 1: HEP-Specific Libraries

| Library | Focus | Strengths | Weaknesses |
|---------|-------|-----------|------------|
| **ROOT/PyROOT** | HEP standard | Rich statistics, TTree::Draw, fitting | Global state, memory quirks, heavy dependency |
| **uproot** | ROOT I/O | Pure Python, no ROOT dependency, NumPy/Awkward | Read-only focus, no plotting |
| **boost-histogram / hist** | Histogramming | High performance, ROOT-compatible, Pythonic | Plotting separate, no TTree::Draw syntax |
| **mplhep** | HEP plotting | ATLAS/CMS/LHCb styles, publication-ready | Matplotlib-based (static) |
| **RootInteractive** | Interactive analysis | N-dimensional, Bokeh-based, ALICE calibration | Complex API, browser-dependent |

### Category 2: General Python Plotting

| Library | Paradigm | Strengths | Weaknesses |
|---------|----------|-----------|------------|
| **Matplotlib** | Imperative | Flexible, universal, publication-ready | Verbose, static |
| **Seaborn** | Statistical | Beautiful defaults, pandas integration | Limited customization, static |
| **Plotly** | Interactive | Web-native, hover/zoom, declarative | Heavy, web-dependent |
| **Bokeh** | Interactive | Streaming, server apps, widgets | Complex for simple plots |
| **Altair** | Declarative | Grammar of graphics, concise | Limited chart types |

### Category 3: Our Stack (dfdraw + dfextensions)

| Component | Role | Integration |
|-----------|------|-------------|
| **dfdraw** | ROOT-style plotting | Matplotlib backend, TTree::Draw syntax |
| **AliasDataFrame** | Schema-driven analysis | Duck-typed axis titles |
| **RDataFrameDSL** | Python→C++ codegen | Uses dfdraw for QA plots |
| **GroupByRegressor** | Statistical fits | PyArrow output → dfdraw |

---

## ROOT TTree::Draw: What to Keep

### ✅ Good Parts (Preserve in Wrapper)

| Feature | ROOT Syntax | dfdraw Equivalent | Value |
|---------|-------------|-------------------|-------|
| **One-liner syntax** | `tree->Draw("y:x")` | `drawer.draw("y:x")` | Minimal boilerplate |
| **Computed expressions** | `tree->Draw("pt*1000:eta")` | `drawer.draw("pt*1000:eta")` | No temp columns |
| **Inline selection** | `tree->Draw("x", "x>0")` | `drawer.draw("x", selection="x>0")` | Filter + plot in one |
| **Auto-binning** | `tree->Draw("x>>h(100,0,1)")` | `drawer.draw("x", bins=100, range=(0,1))` | Smart defaults |
| **Statistics box** | `gStyle->SetOptStat(1111)` | `stats=True` or `stats=['n','mean']` | Built-in QA |
| **Profile plots** | `tree->Draw("y:x", "", "prof")` | `drawer.draw("y:x", type="profile")` | Mean vs binned x |
| **2D histograms** | `tree->Draw("y:x", "", "colz")` | `drawer.draw("y:x", type="hist2d")` | Density visualization |
| **Population std** | ddof=0 | ddof=0 (Phase 13.6.G.DF) | ROOT compatibility |
| **Range-aware stats** | Within range only | range_x, range_y (Phase 13.6.G.DF) | Filtered statistics |
| **Auto-title from expression** | Auto | `auto_title=True` (Phase 13.12.DF) | QA plot caption automation |
| **Multi-plot overlay on same axes** | `"same"` option | `same=True` (Phase 13.13.DF) | Superposition without manual axes |
| **Multi-series from column list** | `tree->Draw("y1:x"); tree->Draw("y2:x","","same")` (loop) | `drawer.draw("[y1,y2,y3]:x")` (Phase 13.16.DF) | Single-call vector expression; fixes ADF color-cycling (AD-37) |
| **Batch QA dashboards** | Canvas + Divide loop | `draw_batch(specs)` with group format (Phase 13.14.DF) | YAML/JSON spec-driven QA |

### ❌ Bad Parts (Avoid in Wrapper)

| ROOT Problem | Description | dfdraw Solution |
|--------------|-------------|-----------------|
| **Global state** | `gStyle`, `gDirectory`, `gPad` affect everything | No global state; explicit parameters |
| **Memory ownership** | Histograms deleted when file closes | Return `(fig, ax, stats)` tuple |
| **Canvas management** | Must create/switch canvases | Auto-create matplotlib figures |
| **String-based options** | `"same"`, `"colz"`, `"prof"` | Named parameters: `type="profile"` |
| **`>>` redirection** | `"x>>h(100,0,1)"` is parsing nightmare | Separate `bins=` and `range=` params |
| **Non-Pythonic API** | `SetXXX()` methods everywhere | Direct assignment / rcParams |
| **Heavy dependency** | Full ROOT install (~500MB) | Pure Python + matplotlib |

---

## Comparison Matrix: Statistics Features

| Feature | ROOT | Matplotlib | Seaborn | boost-hist | dfdraw | RootInteractive |
|---------|------|------------|---------|------------|--------|-----------------|
| N (entries) | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Mean | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Std Dev (population) | ✅ | ❌ | ❌ | ✅ | ✅ (13.6.G.DF) | ✅ |
| Min/Max | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ |
| Median | ✅ | ❌ | ❌ | ❌ | ✅ (13.6.G.DF) | ✅ |
| Quartiles | ✅ | ❌ | ❌ | ❌ | ✅ (13.6.G.DF) | ✅ |
| MAD | ✅ | ❌ | ❌ | ❌ | ✅ (13.6.G.DF) | ✅ |
| Correlation | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| Range-filtered stats | ✅ | ❌ | ❌ | ❌ | ✅ (13.6.G.DF) | ✅ |
| Stats box overlay | ✅ | Manual | Manual | Manual | ✅ | ✅ |
| Auto-detect defaults | ✅ | ❌ | ❌ | ❌ | ✅ (13.6.G.DF) | ✅ |

**Phase 13.6.G.DF Achievement:** dfdraw now has full statistics parity with ROOT.

---

## Comparison Matrix: Plot Types

| Plot Type | ROOT | Matplotlib | Seaborn | dfdraw | RootInteractive |
|-----------|------|------------|---------|--------|-----------------|
| 1D Histogram | TH1 | `hist()` | `histplot()` | `hist()` | ✅ |
| 2D Histogram | TH2 | `hist2d()` | `histplot(x,y)` | `hist2d()` | ✅ |
| Scatter | TGraph | `scatter()` | `scatterplot()` | `scatter()` | ✅ |
| Profile | TProfile | Manual | Manual | `profile()` | ✅ |
| Hexbin | - | `hexbin()` | - | `hexbin()` | ✅ |
| Facet/Grid | - | `subplots` | `FacetGrid` | `facet=True` | Layout system |
| Group overlay | - | Manual | `hue=` | `group_by=` | ✅ |
| **Same-axes superposition** | **`"same"` option** | **Manual `ax=`** | **Manual** | **`same=True` (13.13.DF)** | **figureArray** |
| **Vector expression (multi-series)** | **Loop + "same"** | **Manual loop** | **`melt()` + `hue=`** | **`"[y1,y2]:x"` (13.16.DF)** | **figureArray** |
| **Batch QA dashboard** | **Canvas + Divide** | **Manual subplots** | **FacetGrid** | **`draw_batch` group format (13.14.DF)** | **widgetArray + figureArray** |

---

## RootInteractive: Deep Dive

### Architecture
```
RootInteractive
├── Declarative configuration (JSON/dict)
├── Bokeh ColumnDataSource (backend)
├── N-dimensional histogramming
├── Client-side aggregation (JavaScript)
└── Server-side precomputation (Python/C++)
```

### Key Features for Calibration QA

1. **N-dimensional histogramming/projection:** Slice high-dimensional data interactively
2. **Aggregated statistics:** Mean, median, RMS, quantiles computed on-the-fly
3. **Data compression:** Handle O(10⁷) entries in browser
4. **Declarative layouts:** Define dashboard in dict/JSON
5. **Standalone HTML export:** No server needed for viewing

### Integration Considerations

| Aspect | dfdraw (Current) | RootInteractive | Unified Wrapper |
|--------|------------------|-----------------|-----------------|
| Backend | Matplotlib | Bokeh | Pluggable |
| Interactivity | Static PNG/PDF | Full browser | Mode-dependent |
| Syntax | `drawer.draw("y:x")` | `bokehDrawSA(...)` | TTree::Draw style |
| Statistics | Computed once | Live aggregation | Both modes |
| Output | Figure file | HTML dashboard | Format-agnostic |

---

## Proposed Wrapper Architecture

### Design Goals

1. **Unified TTree::Draw-like syntax** across backends
2. **Backend-agnostic specification** (matplotlib/Bokeh/Plotly)
3. **Progressive disclosure:** Simple cases stay simple
4. **Statistics consistency:** Same stats fields everywhere
5. **RootInteractive integration:** Export to interactive dashboards

### Proposed API Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                    User API Layer                               │
│  drawer.draw("y:x", stats=True)  # TTree::Draw syntax           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Specification Layer                              │
│  PlotSpec(expr="y:x", type="scatter", stats=['n','mean'])       │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Matplotlib      │ │ Bokeh/          │ │ Plotly          │
│ Backend         │ │ RootInteractive │ │ Backend         │
│ (Static)        │ │ (Interactive)   │ │ (Web)           │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

### Example: Backend-Agnostic Code

```python
from dfdraw import DFDraw, set_backend

# Default: matplotlib (publication-ready, static)
drawer = DFDraw(df)
drawer.draw("dEdx:momentum", type="profile", stats=True)
drawer.save("profile.pdf")

# Switch to RootInteractive (interactive dashboard)
set_backend("rootinteractive")
drawer.draw("dEdx:momentum", type="profile", stats=True)
drawer.export_html("profile_dashboard.html")

# Batch mode with backend selection
specs = {
    'profile_dEdx': {'expr': 'dEdx:p', 'type': 'profile'},
    'hist2d_eta_phi': {'expr': 'phi:eta', 'type': 'hist2d'},
}
drawer.draw_batch(specs, backend='matplotlib', save_dir='plots/')
drawer.draw_batch(specs, backend='rootinteractive', save_dir='dashboards/')
```

---

## Feature Parity Checklist

### Phase 1: Core Parity (Phase 13.5 and earlier) ✅ COMPLETE
- [x] 1D/2D histograms
- [x] Scatter plots
- [x] Profile plots
- [x] Hexbin plots
- [x] Grouping/overlay
- [x] Faceting
- [x] Statistics box
- [x] Selection/cuts
- [x] Batch processing

### Phase 2: Statistics Enhancements (Phase 13.6.G.DF) ✅ COMPLETE
- [x] 2D stats display fix (auto-detect by plot type)
- [x] Range-aware stats (range_x, range_y)
- [x] Robust stats (median, q25, q75, MAD)
- [x] Population std (ddof=0) for ROOT match
- [x] 2D n counts both-valid pairs

### Phase 2b: Interactive Features (Phase 13.12–13.16.DF) ✅ COMPLETE
- [x] **Auto-title from expression** (Phase 13.12.DF v1.2) — `auto_title=True` / style key
- [x] **Profile enhancements** (Phase 13.12.DF) — `return_data`, `min_entries`, `group_by_bins`, `group_by_quantiles`, `sort_groups`, `weights`
- [x] **Same-axes superposition** (Phase 13.13.DF) — `same=True` for all draw methods, auto-color, auto-label
- [x] **Batch group format** (Phase 13.14.DF) — `draw_batch` with defaults cascade, subplot grids, `verbose=2` debug
- [x] **Test infrastructure** (Phase 13.15.DF) — feature taxonomy (43 features), capability matrix, `run_tests.sh`, reviewer.zip packaging
- [x] **Vector expression interface** (Phase 13.16.DF) — bracket syntax `"[y1,y2,y3]:x"`, `"y:[x1,x2]"`, `"[y1,y2]:[x1,x2]"`, paren-aware split, `vector_style`/`group_style` channels
- [x] **Strong A≡B invariance testing** (Phase 13.16.DF) — 7 byte-identical axis + per-pair stats comparisons
- [x] **AD-37 architectural fix** (Phase 13.16.DF) — single-instance vector path bypasses AliasDataFrame color-cycling bug

### Phase 3: Backend Abstraction (Partially Complete)

Status updated 2026-04-09 per architect Q&A (PHASE_END_PHASE-0.2.A):

- [ ] **PlotSpec intermediate representation** — Not started
- [x] **Matplotlib backend (refactor)** — Current production backend, in use
- [ ] **Bokeh backend** — Not started
- [ ] **RootInteractive backend** — Bridging module proposed but not implemented; architecture direction (e.g., `dfdraw.backends.rootinteractive`) is architect decision pending
- [ ] **Plotly backend (optional)** — Not started

**Summary:** Backend abstraction as a formal architectural layer has not been built. dfdraw continues to use matplotlib directly. The RootInteractive foundation work (see Phase 4 below) is proceeding in the RootInteractive repository; the dfdraw-facing bridge remains proposal/prototype stage.

### Phase 4: RootInteractive Integration (Partially Complete)

Status updated 2026-04-09 per RootInteractive team Q&A (PHASE_END_PHASE-0.2.A):

- [x] **N-dimensional histogram export** — Done in RootInteractive (`HistoNdCDS` exists and is validated)
- [ ] **Dashboard layout specification** — Not started (no dfdraw-facing spec)
- [x] **HTML export** — Done in RootInteractive itself via `bokehDrawSA.fromArray()`, but not wired to dfdraw
- [ ] **Live aggregation mode** — Not started

**Summary:** RootInteractive foundation capabilities (compression, histograms, ONNX, joins) are advancing in phases 0.1.B–0.2.A of the RootInteractive repository. There is currently no implemented dfdraw→RootInteractive integration layer — none of the API patterns proposed in the [Proposed Wrapper Architecture](#proposed-wrapper-architecture) section above are functional today. When the bridging work begins, the RootInteractive team will communicate the phase ID, API shape, and stability level.

**For dfdraw users today:** use matplotlib backend (the default and only working backend). For interactive multi-dimensional exploration of ALICE data, use RootInteractive directly at https://github.com/miranov25/RootInteractive.

---

## Current Status Summary

### dfdraw Capabilities (Phase 13.16.DF, refreshed Phase 13.17.DF)

| Category | Status | Details |
|----------|--------|---------|
| **Plot Types** | ✅ Complete | 5 types: hist, scatter, profile, hist2d, hexbin |
| **Statistics** | ✅ Complete | Full ROOT parity including robust stats (Phase 13.6.G.DF) |
| **Multi-series overlay** | ✅ Complete | Vector expressions + `same=True` chaining (Phases 13.13.DF, 13.16.DF) |
| **Batch QA** | ✅ Complete | Group format, subplot grid, defaults cascade (Phase 13.14.DF) |
| **Auto-title** | ✅ Complete | Expression-derived titles (Phase 13.12.DF v1.2) |
| **Styling** | ✅ Complete | Global + per-plot + matplotlib pass-through |
| **Integration** | ✅ Complete | AliasDataFrame, PyArrow, batch mode |
| **Testing** | ✅ Complete | 451 tests, 43 features, 21 invariance tests, 100% passing |
| **Documentation** | ✅ Complete | README (refreshed Phase 13.17.DF), API_REFERENCE, PHASE_HISTORY, Technical Summary, Capability Matrix |

### ROOT Compatibility Achieved

| Feature | ROOT | dfdraw | Status |
|---------|------|--------|--------|
| Population std | ddof=0 | ddof=0 | ✅ Match |
| Range-filtered stats | Yes | Yes | ✅ Match |
| 2D both-valid n | Yes | Yes | ✅ Match |
| Auto stats fields | Yes | Yes | ✅ Match |
| Robust stats | Yes | Yes | ✅ Match |
| Multi-series overlay | `"same"` option | `same=True` / vector `"[y1,y2]:x"` | ✅ Match (two paths) |
| Auto-title | Auto | `auto_title=True` | ✅ Match |
| Batch QA dashboard | Canvas + Divide | `draw_batch` group format | ✅ Match |

---

## Recommendations

### Short-Term ✅ COMPLETED (Phase 13.6.G.DF)
1. ~~Complete statistics enhancements~~ — **DONE**
2. ~~Keep matplotlib backend~~ — Proven, publication-ready
3. ~~Document stats fields consistently~~ — **DONE**

### Medium-Term (Next Major Phase)
1. **Define PlotSpec format** — Backend-agnostic plot description
2. **Abstract rendering layer** — Interface for pluggable backends
3. **Prototype Bokeh backend** — Stepping stone to RootInteractive

### Long-Term (Integration Phase)
1. **RootInteractive backend** — Full interactive dashboards
2. **Export to RootInteractive format** — Leverage existing ALICE infrastructure
3. **Hybrid mode** — Static + interactive from same spec

---

## References

- ROOT TTree::Draw: https://root.cern/doc/master/classTTree.html
- Scikit-HEP ecosystem: https://scikit-hep.org/
- boost-histogram: https://boost-histogram.readthedocs.io/
- RootInteractive: https://github.com/miranov25/RootInteractive
- RootInteractive paper: https://arxiv.org/abs/2403.19330
- mplhep: https://github.com/scikit-hep/mplhep
- Seaborn histplot: https://seaborn.pydata.org/generated/seaborn.histplot.html

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-29 | Initial comparison document |
| 1.1 | 2026-01-29 | Updated for Phase 13.6.G.DF completion: statistics enhancements complete, ROOT parity achieved |
| 1.2 | 2026-04-09 | Phase 13.17.DF refresh: added Phase 13.12–13.16 feature capabilities (auto-title, same=True, draw_batch groups, vector expressions, strong invariance testing, AD-37 architectural fix); updated Plot Types and ROOT TTree::Draw "What to Keep" matrices with new multi-series / vector / batch rows; updated Phase 3 Backend Abstraction and Phase 4 RootInteractive Integration checklists with current status per architect Q&A (2026-02-22); test count 310 → 451, feature count → 43, added invariance test count 21; Current Status Summary updated with new capability categories. Strategic/architectural framing preserved verbatim from v1.1; no structural changes.
