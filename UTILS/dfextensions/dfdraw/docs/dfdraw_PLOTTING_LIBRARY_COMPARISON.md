# dfdraw + ADF + GB Stack — Plotting Library Comparison

**Version:** 2.1 (Phase TS_v6 cycle — major restructure + cross-team panel-fix pass)
**Date:** 2026-06-06
**Original:** 2026-01-29 (Phase 13.6.G.DF, v1.0 — Claude2-dfdraw)
**Refresh history:** 2026-04-09 (Phase 13.17.DF, v1.2 — Claude40)
**Authors:** Claude2-dfdraw (original) · Claude40 (v1.2 refresh) · Opus2 (v2.0 restructure) · Opus1 (v2.0 architect-pass + v2.1 cross-team-panel-fix pass)
**Purpose:** Position the **dfdraw + AliasDataFrame + GBregression stack** within the Python and HEP visualization landscape, aligned with `dfdraw_Technical_Summary.md` v6.3 stack identity.
**Source baseline:** HEAD `07606c02` (Phase 13.50.DF FIX2)

---

## Reading guide

This document compares the **dfdraw + ADF + GB stack** with other libraries on three layers:

1. **dfdraw as a plotting library** (vs matplotlib, seaborn, plotly, altair, bokeh, ROOT TTree::Draw, mplhep) — §3 and §4
2. **The stack as a declarative-grammar framework** (vs ggplot2 + broom + lme4, vs Vega-Altair, vs Polars + Altair) — §5 and §6
3. **The stack as an HEP analysis ecosystem** (vs Scikit-HEP: uproot + awkward + hist + mplhep + coffea, vs RootInteractive, vs RDataFrame) — §7 and §8

If you are choosing a Python plotting library for general analysis, read §3 + §4 + §6. If you are choosing an HEP analysis ecosystem, read §7 + §8. If you want the architectural argument for the stack, read §1 + §2 + §5.

---

## §1. Executive Summary

**dfdraw is one of three components of an integrated statistical analysis framework**, not a standalone plotting library. The stack — **AliasDataFrame (ADF) + GBregression + dfdraw** — is analogous to R's *ggplot2 + broom + lme4* integrated ecosystem. **No clean public Python equivalent exists.** This document substantiates that claim through systematic comparison.

The strength of the stack is **N-dimensional groupby decomposition declared in a single call**: six orthogonal data dimensions (`vector × group_by × facet_by × *_bins × selection_vector × weights_vector`) compose without nested loops and without re-binning. Statistics flow back as `stats` dict — programmatically consumable by GBregression — so visualization and analysis are not separate workflows.

Three things position the stack uniquely:

1. **Declarative grammar with Algorithm A channel assignment** — each data dimension is mapped to a distinct visual channel (color / linestyle / marker / spatial) automatically, with overflow handling and per-call overrides. No other Python library has this for the full vector / group_by / facet_by composition.
2. **Differential operations as first-class primitives** — `selection_vector=[A, B] + normalize="delta"` produces a residual plot in one call; the same convention drives ratio, pull, and reference-overlay analyses (see §6.7 of the Technical Summary).
3. **Stack-native lazy evaluation** — ADF aliases + lazy materialization make multi-TB tracking datasets queryable; GB regression coefficients register as ADF subframes so model predictions are alias expressions, not bespoke joins.

What dfdraw is **not**:
- A backend-abstraction layer (the "wrapper" design proposed in this document v1.0 was never implemented; the channel framework was built instead — see §11).
- A full interactive dashboard system (interactive use is matplotlib `widget` mode and Jupyter; the future-interactive path is RootInteractive integration — see §8).
- A replacement for matplotlib (matplotlib is the rendering backend; dfdraw composes the grammar above it).

---

## §2. The Stack Identity

The dfdraw + ADF + GB stack restores the **analyst-bandwidth iteration model** that ALICE Run 1 / Run 2 calibration QA had through `TTree::Draw` against downsampled queryable datasets. See Technical Summary §"Why this stack exists" for the full Run 1/2 → Run 3 architectural-regression narrative.

The three components and how they compose:

| Layer | Component | Responsibility | Composition point |
|---|---|---|---|
| Data | **AliasDataFrame (ADF)** | Declarative derived columns (aliases), subframe joins, lazy materialization | `adf.draw(...)` resolves aliases and forwards to dfdraw |
| Analysis | **GBregression** | Per-group OLS / WLS / robust / sliding-window fits → coefficient DataFrames | GB coefficient frames register as ADF subframes |
| Visualization | **dfdraw** | Declarative grammar (6 axes) + per-bin statistics + inline fits + differential operations | `(fig, ax, stats)` return; `stats` consumed by GB |

**None of the three alone covers the workflow.** Standalone dfdraw on a flat pandas DataFrame works, but loses the lazy-alias data layer (no multi-TB workflows) and the integrated model-prediction overlay (no GB-coefficient subframes).

This is the architectural fact every comparison in this document must account for: when you compare *X plotting library* against dfdraw, you may be comparing the wrong thing. The fair comparison is *X plotting library + X data layer + X analysis layer* against the stack.

---

## §3. Library Landscape (refreshed)

### Category 1 — HEP-specific libraries (Scikit-HEP ecosystem + ROOT)

| Library | Focus | Strengths | Weaknesses |
|---|---|---|---|
| **ROOT / PyROOT** | HEP standard | Rich statistics, TTree::Draw, fitting, file format | Global state, memory ownership quirks, ~500 MB dependency |
| **uproot** | ROOT I/O in Python | Pure Python read/write of ROOT files, no ROOT dependency | I/O only — no plotting |
| **awkward** | Jagged-array data | Idiomatic Python for HEP event data with variable-length structures | Not a DataFrame; not a plotting library |
| **hist + boost-histogram** | N-D histogramming | High-performance binning, ROOT-compatible memory model, axis types | Histograms only; plot via mplhep |
| **mplhep** | HEP matplotlib styling | ATLAS/CMS/LHCb/ALICE styles; ratio/pull/efficiency comparison panels for pre-binned data; 1.0 released Jan 2026 | Matplotlib-based (static); requires pre-binned input |
| **coffea** | HEP analysis framework | Processor model + accumulator + dask-distributed; columnar at scale | Heavy framework; analysis-only |
| **RootInteractive** | Interactive HEP analysis | N-D histograms, Bokeh-based, ALICE-specific calibration QA, browser dashboards | Complex API; browser-dependent; specialized for ALICE workflows |

### Category 2 — General Python plotting libraries

| Library | Paradigm | Strengths | Weaknesses |
|---|---|---|---|
| **Matplotlib** | Imperative + pyplot | Universal, publication-ready, extensive | Verbose for compositions; static |
| **Seaborn** | Statistical wrappers on matplotlib | Beautiful defaults, pandas-aware, `hue=` / `FacetGrid` | Limited customization; static; no groupby decomposition beyond 2-3 axes |
| **Plotly + Plotly Express** | Interactive declarative (`px`) | Web-native, hover/zoom, JSON Vega-like spec | Heavy bundle; web-dependent; per-call API |
| **Bokeh** | Interactive imperative | Streaming, server apps, widgets | Complex for simple plots; verbose |
| **Vega-Altair** | Declarative grammar of graphics | Vega-Lite spec, interaction grammar, type-checked API | Limited statistical primitives; small-data oriented (browser rendering) |
| **HoloViews + hvPlot** | Declarative on top of Bokeh/matplotlib | Lazy `.hvplot()` accessor on DataFrames; multi-backend | Indirection layer; less direct than Altair |

### Category 3 — General Python data + plotting integration

| Stack | Composition | Notes |
|---|---|---|
| **pandas + matplotlib** | `df.plot(...)` accessor | Standard; imperative composition above the dataframe |
| **pandas + seaborn** | Wide / long form input | `hue=` / `FacetGrid` for 2-3 axes |
| **Polars + Altair** | `df.plot.<mark>(...)` delegates to Altair (Polars 1.6+, current default) | Polars: lazy queries; Altair: declarative grammar. The closest *cross-domain* peer for the dfdraw stack philosophy |
| **R: ggplot2 + broom + lme4** | Grammar of graphics + tidy fits + mixed-effects models | The **closest conceptual analog** to dfdraw + ADF + GB. See §5 for full comparison |

### Category 4 — Our stack (dfdraw + dfextensions)

| Component | Role | Integration |
|---|---|---|
| **dfdraw** | Declarative statistical visualization grammar over pandas | Matplotlib backend; 6-axis composition; `(fig, ax, stats)` return contract |
| **AliasDataFrame (ADF)** | Declarative data layer | Aliases, subframe joins, lazy materialization; resolves to pandas at draw time |
| **GBregression** | Declarative analysis layer | `make_parallel_fit_v4`, sliding-window fits, prediction registration; outputs ADF subframes |
| **RDataFrameDSL** | Python → C++ codegen (future) | TS §5.3; uses dfdraw for QA |

---

## §4. ROOT TTree::Draw — What to Keep, What to Avoid

### ✅ Good parts (preserved and extended in dfdraw)

| Feature | ROOT syntax | dfdraw equivalent | Status |
|---|---|---|---|
| One-liner declarative call | `tree->Draw("y:x")` | `drawer.draw("y:x")` | ✅ Preserved |
| Computed expressions | `tree->Draw("pt*1000:eta")` | `drawer.draw("pt*1000:eta")` | ✅ Preserved |
| Inline selection | `tree->Draw("x", "x>0")` | `drawer.draw("x", selection="x>0")` | ✅ Preserved |
| Auto-binning | `tree->Draw("x>>h(100,0,1)")` | `drawer.draw("x", bins=100, range=(0,1))` | ✅ Cleaner |
| Statistics box | `gStyle->SetOptStat(1111)` | `stats=True` or `stats=["mean","std","n"]` | ✅ Per-plot, no global state |
| Profile plots | `tree->Draw("y:x", "", "prof")` | `drawer.draw("y:x", type="profile")` | ✅ Preserved |
| 2D histograms | `tree->Draw("y:x", "", "colz")` | `drawer.draw("y:x", type="hist2d")` | ✅ Preserved |
| Same-axes overlay | `"same"` option | `same=True` (Phase 13.13) | ✅ Match |
| Multi-series from list | `Draw("y1:x"); Draw("y2:x","","same")` loop | `drawer.draw("[y1,y2,y3]:x")` (Phase 13.16) | ✅ Single-call vector — **better** |
| Population std | `ddof=0` | `ddof=0` (Phase 13.6.G.DF) | ✅ ROOT-compatible |
| Range-aware stats | Within range only | `range=` (Phase 13.6.G.DF) | ✅ Match |
| Inline curve fits | `TF1::Fit` per histogram | `fit="gauss"` (Phase 13.42) | ✅ Composable with grammar |
| Summary fit figures | Manual `cd` + draw | `summary_fit='table'` (Phase 13.43) | ✅ Automatic per-group table |
| Cumulative CDF | `TH1::Draw("cumulative")` | `hist(cumulative=True/-1)` (Phase 13.40) | ✅ Match |
| Batch QA dashboards | Canvas + Divide | `draw_batch(specs)` + group format (Phase 13.14) | ✅ Spec-driven |

### ❌ Bad parts (avoided in dfdraw)

| ROOT problem | dfdraw solution |
|---|---|
| Global state (`gStyle`, `gDirectory`, `gPad`) | No global state; explicit parameters; per-plot style cascade |
| Memory ownership (histograms deleted on file close) | `(fig, ax, stats)` tuple returned; lifecycle owned by caller |
| Canvas management (`cd()`, `Divide()`) | Auto-create matplotlib figures; `facet_by=` for grids |
| String-based options (`"same"`, `"colz"`, `"prof"`) | Named kwargs: `type="profile"`, `same=True`, etc. |
| `>>` redirection (`"x>>h(100,0,1)"` parser nightmare) | Separate `bins=`, `range=` |
| Non-Pythonic API (`SetXXX()` methods everywhere) | Direct assignment, rcParams, declarative kwargs |
| Heavy dependency (~500 MB) | Pure Python + matplotlib + numpy + scipy + pandas |

### ⊕ Beyond ROOT — what dfdraw adds that TTree::Draw cannot do

| Capability | TTree::Draw | dfdraw |
|---|---|---|
| **N-D faceting (ROW × COL × FIGID)** | Manual loop + `cd()` per pad | `facet_by=["row","col","figid"]` (Phase 13.41) |
| **Channel-aware overlays** (color × marker × linestyle) | Manual color cycling | Algorithm A automatic assignment with `EXPLICIT_RULES` (Phase 13.26) |
| **Selection-vector pairs** for differential analysis | Two separate Draw calls + manual diff | `selection_vector=[A,B] + normalize="delta"` in one call (Phase 13.27) |
| **Weight-vector pairs** for systematic comparisons | Reweight + redraw loop | `weights_vector=[w1, w2, w3]` (Phase 13.27) |
| **Inline + summary fit figures composable with grammar** | `TF1::Fit` per histogram, separate | `fit="gauss" + summary_fit='table'` per group / per facet cell |
| **Quantile rendering on profile** (band / error_bars / nested_band) | Manual TGraph construction | `quantiles=[0.16, 0.5, 0.84]` with auto-detect mode (Phase 13.26) |
| **Profile2D / Scatter3D / Time-axis** | Manual TH2 + TLatex / TGraph2D | `profile2d()`, `scatter3d()`, `time_format=` (Phase 13.39) |
| **Stack-native model overlay** (GB coefficient subframes → alias → render) | Bespoke C++ per plot | `adf.add_alias("dy_corrected", "dy - GB_subframe.intercept")` then `drawer.draw("[dy, dy_corrected]:x", normalize="delta")` |

---

## §5. Stack-Level Comparison — dfdraw + ADF + GB vs. ggplot2 + broom + lme4

This is the **load-bearing comparison.** R's *tidyverse* developed an integrated triple that no Python ecosystem has matched: ggplot2 (visualization grammar) + broom (tidy model output) + lme4 (mixed-effects models). The dfdraw stack is the closest Python equivalent.

| Concern | R (tidyverse) | dfdraw + ADF + GB | Status |
|---|---|---|---|
| **Grammar of graphics** | `ggplot(df) + geom_point() + facet_wrap(~f) + aes(color=g)` | `drawer.draw("y:x", type="scatter", facet_by="f", group_by="g")` | ✅ Equivalent expressiveness |
| **Tidy data convention** | `tidyr` + `pivot_longer` / `pivot_wider` | pandas + ADF aliases; vector-bracket syntax `[y1,y2]:x` for wide-to-long-at-draw-time | ✅ Different mechanism, same outcome |
| **Tidy model output** | `broom::tidy(lm(y ~ x, df))` → tidy coefficient frame | `make_parallel_fit_v4(df, gb_columns=[...], fit_columns=[...])` → coefficient DataFrame | ✅ Direct analog |
| **Mixed-effects / multi-level** | `lme4::lmer(y ~ x + (1\|group), df)` | `make_parallel_fit_v4(df, gb_columns=["group"], linear_columns=["x"])` for per-group fits; full mixed-effects via Huber / RLM in `GroupByRegressor.make_parallel_fit` | ⚠️ Partial — dfextensions stack does per-group OLS / robust regression at scale; classical mixed-effects (variance component decomposition) is not (yet) in GBregression |
| **Composition: model + grammar** | `broom::augment(model, df)` then `ggplot(...)`  | `register_subframe("CalibBias", AliasDataFrame(dfCoeffs))` then `adf.draw("[dy, dy_predicted]:row", normalize="delta")` | ✅ Equivalent — and via ADF subframe joins, **lazier** |
| **Lazy / memory-aware** | None natively; `dbplyr` for SQL-backed | ADF `draw_lazy=True` resolves only needed columns | ✅ **Better** — multi-TB native |
| **Multi-dimensional decomposition** | `facet_grid(rows ~ cols)` (2-D) | `facet_by=["ROW", "COL", "FIGID"]` (3-D, Phase 13.41) | ✅ **Better** — N-D |
| **Differential operations** | Manual: compute residual column, then plot | `normalize="delta"` + `selection_vector=[A,B]` first-class | ✅ **Better** — declarative |
| **Statistical fits inline** | `geom_smooth(method=lm)` | `fit="gauss"` / `"linear"` / `"polN"` / custom | ✅ Match |
| **Ecosystem maturity** | 15+ years, very large user base | Years, small but production ALICE user base | ❌ ggplot2 wins on community / docs / Stack Overflow |
| **Publication-grade output** | Excellent (with `ggsave`) | Excellent (matplotlib `savefig`) | ✅ Match |
| **Interactive (browser)** | `plotly::ggplotly()` or `shiny` | Future via RootInteractive integration | ❌ R wins today |

**Verdict:** The dfdraw + ADF + GB stack matches or exceeds ggplot2 + broom + lme4 on grammar expressiveness, N-D decomposition, lazy evaluation, and differential operations. R wins on ecosystem maturity, classical mixed-effects models, and interactive rendering. **The architectural argument: there is no Python triple that occupies this space; the stack fills the gap.**

---

## §6. Grammar-Layer Comparison — declarative visualization libraries

The newer Python library generation is converging on **declarative grammars** based on Vega-Lite or grammar-of-graphics principles. dfdraw is in this family but distinguishes itself on three axes: channel-aware composition with Algorithm A, statistics return contract, and stack-integrated lazy evaluation.

| Concern | Vega-Altair | Plotly Express | HoloViews | dfdraw |
|---|---|---|---|---|
| **Foundation** | Vega-Lite JSON spec → browser renderer | Plotly.js JSON → browser renderer | Bokeh / matplotlib backends | matplotlib direct |
| **Output** | HTML / JSON | HTML | HTML / static | static (publication) + Jupyter widget |
| **Encoding model** | `encode(x=, y=, color=, size=, ...)` | `px.scatter(df, x=, y=, color=, ...)` | `kdims=`, `vdims=` | Channel framework + `group_by` / `facet_by` / `vector` |
| **Channel-collision resolution** | None — user-driven | None — user-driven | None — user-driven | **Algorithm A** — automatic with `EXPLICIT_RULES` table (Phase 13.26) |
| **Faceting** | `facet=` 1-D, `row=` `column=` 2-D | `facet_row=`, `facet_col=` 2-D | `groupby=` | `facet_by=` 1-D, `[row, col]` 2-D, `[row, col, figid]` 3-D (Phase 13.41) |
| **Differential operations** | Manual via `transform_calculate` | Manual via DataFrame ops | Manual | **First-class** `normalize="delta"/"ratio"/"pull"` + `selection_vector` |
| **Statistics output** | Stats inside spec; not programmatically returned | Plot only | Plot only | `stats` dict — programmatic, GB-consumable |
| **Inline fits** | `transform_regression` (limited) | `trendline="ols"` | Manual | `fit="gauss"` / `"linear"` / `"polN"` / custom callable + `stats['fit']` |
| **Data scale** | Browser-bound (~100K rows) | Browser-bound (~1M with WebGL) | Larger via DataShader | Multi-TB via ADF lazy |
| **Interactive** | Yes (`brush`, linked views) | Yes (hover, zoom) | Yes | Future via RootInteractive |
| **Publication output** | Limited (PNG via vl-convert) | Yes (Kaleido) | Yes | Yes (matplotlib native) |

**Polars + Altair** deserves a separate note: as of Polars 1.6+, `df.plot.<mark>(...)` delegates to Altair via Narwhals, making the Polars-Altair pair a *cross-domain* declarative duo — fast columnar data + grammar-based visualization. This is the most similar **public stack** to ADF + dfdraw in spirit. The differences:

| Aspect | Polars + Altair | ADF + dfdraw |
|---|---|---|
| Lazy evaluation | Polars LazyFrame (column-projection pushdown) | ADF lazy aliases (compute on draw) |
| Grammar | Altair / Vega-Lite (browser) | dfdraw (matplotlib + stats return) |
| Statistics return | None (visual only) | `stats` dict programmatically returned |
| Statistical fits | Manual / `transform_regression` | `fit=` first-class + GBregression integration |
| Scale ceiling | Browser-bound for plotting | Multi-TB via lazy + matplotlib direct |
| Audience | General data science | HEP calibration + scientific data |

Polars + Altair is excellent for general data science workflows. The dfdraw stack is **specialized for declarative scientific QA where statistics must flow back as data** — which is the workflow ALICE calibration requires and where no public stack offers an equivalent.

---

## §7. HEP Ecosystem Comparison — Scikit-HEP, coffea, RootInteractive, RDataFrame

The HEP-Python ecosystem has matured rapidly. Scikit-HEP is an established domain-specific Python ecosystem for HEP analysis. The standard analysis pipeline today is:

```
uproot (I/O) → awkward (jagged arrays) → coffea (processor framework)
             → hist + boost-histogram (binning) → mplhep (plotting)
```

| Concern | Scikit-HEP standard | dfdraw stack |
|---|---|---|
| **I/O from ROOT** | uproot (direct) | uproot or pandas (via ADF for column subset) |
| **Event-level data** | awkward (nested arrays) — hierarchy as in-memory nested types | ADF (relational) — hierarchy as chain of subframes joined by index columns; one row per object per level |
| **Binning** | hist + boost-histogram (typed axes, N-D) | dfdraw `bins=`, `*_bins`, `*_quantiles` |
| **Plotting** | mplhep (matplotlib styling + ratio panels) | dfdraw (declarative grammar + stats return) |
| **Statistics return** | `hist` object has stats; plotting separate | `(fig, ax, stats)` — combined |
| **Composition** | hist + mplhep — per-call manual | dfdraw — 6-axis declarative |
| **Distributed processing** | coffea + dask (production-scale) | None native; production via in-memory pandas after pre-reduction (filtering / downsampling / projection); subframe hierarchy preserved |
| **N-D decomposition** | Manual loops over axes | `facet_by=["row","col","figid"]` declarative |
| **Differential operations** | mplhep `histplot(comparison="ratio")` for ratio panels | `normalize="delta"/"ratio"/"pull"` first-class on all plot types |
| **Iteration cadence** | Re-run processor on changes | `adf.draw(...)` over loaded DataFrame — seconds |

**Where Scikit-HEP wins:** production-scale event-level processing with distributed compute. coffea + dask is the right answer for processing CMS/ATLAS sample sizes. mplhep has ATLAS/CMS/LHCb/ALICE house-style sheets that dfdraw does not (yet) provide.

**dfdraw + ADF handles the same hierarchical HEP data:** events with variable-length tracks, tracks with calibrations, V0 → track → collision → calibration. ADF expresses the hierarchy as multiple tables joined via foreign-key indices through subframe registration; awkward expresses the same hierarchy as in-memory nested arrays. The two are different abstractions over the same data structure, not different capabilities. Multi-level aliases resolve the chain lazily at draw time (e.g., a V0-level alias can reference `track.collision.calib.<column>` through the registered subframe chain).

**Where the dfdraw stack wins:** calibration QA, **multi-level physics analysis** (V0 → track → collision → calibration with corrections flowing up the hierarchy via alias resolution), and any iteration-heavy workflow where the 6-axis grammar + differential operations + stats-return contract make iteration measurably faster than `hist + mplhep` for the multi-dimensional decomposition pattern (sector × side × residual × time × momentum × model).

**Architectural complement, not competitor:** Production pipelines use both abstractions. coffea + hist + mplhep handles event-level reduction at production scale; ADF + dfdraw handles the analyst-facing iteration loop — calibration QA, alignment, and physics analysis — through the relational subframe abstraction. See `examples/time_series.py` for the calibration QA pattern; the physics-analysis pattern follows the same subframe-chain idiom.

---

## §8. RootInteractive — Deep Dive (preserved from v1.x, partial refresh)

> **Section status:** preserved per architect (2026-06-06). Integration status sections updated to reflect current state. RootInteractive remains the future-interactive direction for the stack; current dfdraw output is matplotlib static + Jupyter widget.

### Architecture

```
RootInteractive
├── Declarative configuration (JSON / dict)
├── Bokeh ColumnDataSource (backend)
├── N-dimensional histogramming (HistoNdCDS)
├── Client-side aggregation (JavaScript)
└── Server-side precomputation (Python / C++)
```

### Key features for calibration QA

1. **N-dimensional histogram and projection** — slice high-dimensional data interactively
2. **Aggregated statistics on the fly** — mean, median, RMS, quantiles computed client-side
3. **Data compression** — handles O(10⁷) entries in browser
4. **Declarative layout** — dashboard defined in dict / JSON
5. **Standalone HTML export** — no server required for viewing

Reference: arXiv:2403.19330 (RootInteractive paper); repository at https://github.com/miranov25/RootInteractive.

### Integration considerations — current state

| Aspect | dfdraw (current) | RootInteractive | Future unified path |
|---|---|---|---|
| Backend | Matplotlib | Bokeh | Spec / multi-backend (deferred) |
| Interactivity | Static + Jupyter widget | Full browser | Mode-dependent (deferred) |
| Syntax | `drawer.draw("y:x")` | `bokehDrawSA(...)` | TBD |
| Statistics | Computed at draw (returned in `stats`) | Live aggregation | Both (deferred) |
| Output | Figure + `stats` dict | HTML dashboard | Format-agnostic (deferred) |

### Status — what's built, what's not (refreshed 2026-06-06)

- ✅ **N-dimensional histogram export (RootInteractive side)** — `HistoNdCDS` exists and is validated
- ✅ **HTML export (RootInteractive side)** — via `bokehDrawSA.fromArray()`
- ⬜ **dfdraw → RootInteractive bridging module** — proposed but not implemented; architect decision pending
- ⬜ **Dashboard layout specification** — not started (no dfdraw-facing spec)
- ⬜ **Live aggregation mode** — not started

**For dfdraw users today:** use the matplotlib backend (default and only working backend). For interactive multi-dimensional exploration of ALICE data, use RootInteractive directly. The integration bridge will be communicated by the RootInteractive team when it begins.

---

## §9. Worked Examples — side-by-side

### Example A — Profile with grouped overlay

Plot `y` vs `x`, mean per binned `x`, grouped by `sector` (color), faceted by `side` (subplot).

**ROOT TTree::Draw**:
```cpp
for (int s = 0; s < 2; s++) {
    canvas->cd(s+1);
    for (int sec = 0; sec < 18; sec++) {
        tree->Draw(Form("y:x>>h%d_%d(50)", s, sec),
                   Form("side==%d && sector==%d", s, sec),
                   sec == 0 ? "prof" : "prof same");
        // Manual color cycling, legend, label
    }
}
```

**matplotlib + pandas**:
```python
fig, axes = plt.subplots(1, 2)
for s, ax in enumerate(axes):
    for sec, g in df[df['side']==s].groupby('sector'):
        binned = g.groupby(pd.cut(g['x'], bins=50))['y'].agg(['mean', 'std'])
        ax.errorbar(binned.index.mid, binned['mean'], yerr=binned['std'],
                    label=f"sector={sec}")
    ax.legend()
```

**Seaborn**:
```python
g = sns.FacetGrid(df, col="side", hue="sector")
g.map(sns.lineplot, "x", "y", errorbar="sd")
# No native mean-per-binned-x; bin manually first
```

**ggplot2 (R)**:
```r
ggplot(df, aes(x=x, y=y, color=factor(sector))) +
  stat_summary_bin(fun.data=mean_se, bins=50) +
  facet_wrap(~ side)
```

**Vega-Altair**:
```python
alt.Chart(df).mark_point().transform_aggregate(
    mean_y='mean(y)', count='count()', groupby=['x_bin', 'sector', 'side']
).encode(x='x_bin', y='mean_y', color='sector:N', column='side:N')
# Binning must be pre-computed
```

**dfdraw** (single call):
```python
drawer.profile("y:x", group_by="sector", facet_by="side", bins=50)
```

---

### Example B — Differential analysis: residual after model

Compare `dy` against model prediction; render the residual as a delta.

**ROOT TTree::Draw** (no first-class support):
```cpp
// Compute dy_model in friend tree first
tree->AddFriend(modelTree);
tree->Draw("(dy - dy_model):x>>hdy(50)", "isOK", "prof");
// Manual residual; no normalize semantics
```

**matplotlib + pandas + scipy**:
```python
# Fit per group manually
groups = df.groupby('sector')
models = {sec: scipy.stats.linregress(g.x, g.y) for sec, g in groups}
df['dy_pred'] = df.apply(lambda r: models[r.sector].slope * r.x + models[r.sector].intercept, axis=1)
df['residual'] = df.y - df.dy_pred
df.groupby(['sector', pd.cut(df.x, 50)])['residual'].mean().unstack(0).plot()
```

**dfdraw + ADF + GB** (declarative):
```python
# 1. GB regression: per-sector fit
_, dfCoeffs = make_parallel_fit_v4(
    df=adf.df, gb_columns=["sector"], fit_columns=["dy"],
    linear_columns=["x"], suffix="_Model",
)
# 2. ADF: register coefficients as subframe, define aliases
adf.register_subframe("Model", AliasDataFrame(dfCoeffs), index_columns=["sector"])
adf.add_alias("dy_pred",
              "Model.dy_intercept_Model + Model.dy_slope_x_Model * x")
adf.add_alias("dy_residual", "dy - dy_pred")
# 3. dfdraw: declarative differential rendering
adf.draw("[dy, dy_pred]:x", normalize="delta",
         group_by="sector", facet_by="side", fit="gauss")
```

The dfdraw call produces: 2 facet panels × 18 sector overlays × 50 bins × per-cell Gaussian fit on the residual, with `stats['fit']` containing each per-cell fit's parameters for downstream consumption. **No equivalent in any other Python plotting library.**

---

### Example C — N-D faceting (Phase 13.41)

Decompose `dy` across sector (rows) × side (columns) × fill (figures), with q/pT quantile overlay within each cell.

**ROOT TTree::Draw**:
```cpp
for (int fill = 0; fill < n_fills; fill++) {
    TCanvas *c = new TCanvas(Form("c_fill%d", fill), "", 1200, 900);
    c->Divide(2, 18);  // side × sector
    for (int s = 0; s < 2; s++)
        for (int sec = 0; sec < 18; sec++) {
            c->cd(...); /* manual pad index */
            for (int q = 0; q < 5; q++) {
                tree->Draw(...);  /* per-quantile cut, color manage */
            }
        }
}
```

**dfdraw**:
```python
adf.draw(
    "dy:row",
    facet_by=["sector", "side", "fill"],
    facet_by_bins=[None, None, None],     # use existing values
    group_by="qpt", group_by_quantiles=5,
    fit="gauss",
)
```

**No other Python library** supports 3-D faceting with intra-cell quantile overlay declaratively. ggplot2 supports 2-D (`facet_grid(row ~ col)`); for the third axis you must loop and arrange manually.

---

## §10. Capability Matrix — Phase 13.18 → 13.50 capabilities

| Capability | Phase | dfdraw | matplotlib | seaborn | ggplot2 | altair | mplhep | RootInteractive |
|---|---|---|---|---|---|---|---|---|
| 1-D histogram | core | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 2-D histogram / hist2d | core | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Profile (mean per binned x) | core | ✅ | manual | manual | `stat_summary_bin` | manual | manual | ✅ |
| Hexbin | core | ✅ | ✅ | - | ✅ | - | - | ✅ |
| Profile2D | 13.39 | ✅ | manual | `heatmap` | `geom_bin_2d + fun` | manual | manual | ✅ |
| Scatter3D | 13.39 | ✅ | `mpl_toolkits` | - | - | - | - | ✅ |
| Time-axis x-axis | 13.39 | ✅ (`time_format=`) | manual `DateFormatter` | partial | scale_x_datetime | `:T` type | manual | ✅ |
| Cumulative histogram | 13.40 | ✅ | `cumulative=True` | manual | `stat_ecdf` | manual | manual | ✅ |
| Faceting 1-D | core | ✅ (`facet_by=`) | manual subplots | FacetGrid | `facet_wrap` | `facet=` | manual | layout system |
| Faceting 2-D | core | ✅ (`["row","col"]`) | manual subplots | FacetGrid | `facet_grid` | `row=, column=` | manual | layout |
| **Faceting 3-D (ROW × COL × FIGID)** | 13.41 | **✅ (Phase 13.41)** | manual loop | - | **manual loop** | - | - | layout |
| Group overlay (color) | core | ✅ (`group_by=`) | manual | `hue=` | `aes(color=)` | `color:N` | manual | ✅ |
| Quantile rendering on profile | 13.26 | ✅ band / error_bars / nested_band | manual | manual | manual | manual | manual | ✅ |
| Vector expression `[y1,y2]:x` | 13.16 | ✅ | melt + loop | melt + hue | facet / melt | manual | manual | figureArray |
| Same-axes superposition | 13.13 | ✅ (`same=True`) | shared `ax=` | manual | `+` layer | layer | manual | figureArray |
| **`selection_vector=` differential** | 13.27 | **✅** | manual | manual | manual | `transform_filter` | - | - |
| **`weights_vector=` weighted comparison** | 13.27 | **✅** | manual | manual | `weight=` aes | `weight=` | partial | - |
| **`normalize=delta/ratio/pull`** | 13.27/13.33 | **✅ first-class** | manual | manual | manual | `transform_calculate` | partial (`comparison=`) | - |
| **Inline fit `fit="gauss"`** | 13.42 | **✅** | scipy + annotate | `regplot` (lm only) | `geom_smooth(method=)` | `transform_regression` | manual | scipy + Bokeh |
| **Summary fit figure** | 13.43 | **✅** | manual | manual | manual | manual | manual | ✅ |
| **Channel framework Algorithm A** | 13.26 | **✅** | - | - | - | - | - | - |
| **Robust hybrid autorange** | 13.28 | **✅** | manual | manual | manual | manual | manual | - |
| Statistics in `stats` dict | core | ✅ | - | - | - | partial | hist object | partial |
| `(fig, ax, stats)` return | core | ✅ | (fig, ax) | (axes) | (plot object) | (chart) | (fig, ax) | (figure) |
| ROOT population std (`ddof=0`) | 13.6.G | ✅ | - | - | - | - | hist | ✅ |
| Range-filtered statistics | 13.6.G | ✅ | - | - | - | - | manual | ✅ |
| Auto-title from expression | 13.12 | ✅ | - | - | - | - | - | - |
| Batch QA dashboards | 13.14 | ✅ (`draw_batch`) | manual | `FacetGrid.map_dataframe` | manual | `vconcat`/`hconcat` | manual | widgetArray |
| Multi-channel decomposition (≥4 axes) | 13.26 | ✅ | very hard | hard | works for 2–3 | works for 2–3 | very hard | ✅ (declarative) |

**Verdict:** dfdraw is the only general-purpose Python plotting library with first-class support for: channel framework Algorithm A, `selection_vector=` / `weights_vector=` / `normalize=` differential operations, summary fit figures composable with grammar, robust hybrid autorange, and N-D faceting beyond 2-D. Combined with ADF lazy materialization and GB regression integration, this is the full stack.

---

## §11. The "Wrapper Architecture" That Was Not Built (historical note)

The v1.0 of this document (2026-01-29) proposed a backend-abstraction layer:

```
┌─ User API (TTree::Draw style) ─┐
└────────────┬────────────────────┘
             ▼
    ┌─ PlotSpec layer ─┐
    └────────┬─────────┘
       ┌─────┼─────┐
       ▼     ▼     ▼
   matplotlib  bokeh  plotly
```

**This was never built.** The actual architectural direction taken after 2026-01-29 diverged toward a **declarative grammar layer** (the channel framework, Phase 13.26 / AD-57) rather than a backend layer. Reasons:

1. **PlotSpec adds an indirection without solving the composition problem.** The hard part of multi-axis decomposition (channel collision, legend topology, facet routing) is upstream of the backend choice. PlotSpec would have proxied the same problem to multiple backends without solving it once.
2. **Backend portability was not the bottleneck.** matplotlib + Jupyter widget covers static publication and interactive exploration. The interactive-dashboard need is served by direct RootInteractive use; a unified backend layer would have added complexity for a use case better served by a separate tool.
3. **Statistics return required first-class API.** The `(fig, ax, stats)` contract is dfdraw-specific; routing it through a backend-portable spec would have weakened the GB regression integration that is the stack's distinguishing feature.

The channel framework (Algorithm A, `EXPLICIT_RULES`, `channels.cycles.*`) is the architectural answer that actually shipped. See Technical Summary §1.4.2 and AD-57.

---

## §12. Feature Parity Checklist — through Phase 13.50

### Phase 1 — Core (≤ Phase 13.5) ✅ COMPLETE
- [x] 1-D / 2-D histograms, scatter, profile, hexbin
- [x] Grouping / overlay
- [x] Faceting
- [x] Statistics box
- [x] Selection / cuts
- [x] Batch processing

### Phase 2 — Statistics enhancements (Phase 13.6.G.DF) ✅ COMPLETE
- [x] 2-D stats display fix
- [x] Range-aware stats
- [x] Robust stats (median, q25, q75, MAD)
- [x] Population std (`ddof=0`)
- [x] 2-D `n` counts both-valid pairs

### Phase 2b — Interactive features (Phase 13.12 – 13.16) ✅ COMPLETE
- [x] Auto-title from expression (13.12)
- [x] Profile enhancements (`return_data`, `min_entries`, `group_by_bins`, `group_by_quantiles`, `sort_groups`, `weights`)
- [x] Same-axes superposition `same=True` (13.13)
- [x] `draw_batch` group format (13.14)
- [x] Vector expression `[y1,y2,y3]:x` (13.16)
- [x] A≡B invariance testing (13.16)
- [x] AD-37 architectural fix

### Phase 3 — Declarative grammar (Phase 13.26 – 13.33) ✅ COMPLETE
- [x] **Visualization Grammar — channel framework with Algorithm A** (13.26 / AD-57)
- [x] **Quantile rendering on profile** (band / error_bars / nested_band) (13.26)
- [x] **`selection_vector=` + `weights_vector=` differential operations** (13.27)
- [x] **`normalize="delta"/"ratio"/"pull"` modes** (13.27 / 13.33)
- [x] **N-D faceting (ROW × COL × FIGID)** (13.41)
- [x] **Time-axis rendering** `time_format=` (13.39)
- [x] **Profile2D, Scatter3D** (13.39)
- [x] **Cumulative histogram** (13.40)
- [x] **Robust hybrid autorange** (13.28)
- [x] **Column-name faceting** (Phase 13.32)

### Phase 4 — Fitting (Phase 13.42 – 13.50) ✅ COMPLETE
- [x] **Inline fits `fit="gauss"/"linear"/"polN"/custom`** (13.42)
- [x] **Summary fit figures** `summary_fit='table'` (13.43)
- [x] **Fit rendering overhaul** (13.50) — `fit.value_format`, `fit.error_format`, `precision_mode='physics'`

### Phase 5 — Test infrastructure (Phase 13.48 – 13.49) ✅ COMPLETE
- [x] **Visual-primitive testing layer** (Phase 13.48) — 27 deterministic primitive-only tests
- [x] **Invariance test markers** §9 (Phase 13.49) — 356 markers, governance-locked
- [x] **Capability matrix traceability** (Phase 13.49)
- [x] **`run_tests.sh` + feature taxonomy** (Phase 13.15)

### Phase 6 — Stack integration (Phase 13.x – ongoing) ⚠️ PARTIAL
- [x] **AliasDataFrame integration** — aliases resolve at `adf.draw(...)`
- [x] **GBregression integration** — coefficient frames register as ADF subframes
- [x] **§5.4 Technical Summary GB integration documented** (TS v6.3)
- [ ] **`time_series.py` v2 extension** — 9 missing-feature production examples (deferred — see TS §A.6)
- [ ] **ADF-dfdraw interface update** — pending phase

### Phase 7 — Backend abstraction ❌ NOT PURSUED (see §11)
- [x] **Decision recorded:** declarative grammar layer (channel framework) was built instead — see §11

### Phase 8 — RootInteractive integration ⏳ FUTURE
- [x] **RootInteractive-side foundations** — `HistoNdCDS`, ONNX, joins (RootInteractive repo)
- [ ] **dfdraw → RootInteractive bridging module** — not implemented
- [ ] **Dashboard layout specification** — not started
- [ ] **Live aggregation mode** — not started

---

## §13. Current Status Summary

### dfdraw library scope (HEAD `07606c02`, Phase 13.50.DF FIX2)

| Category | Status | Details |
|---|---|---|
| Plot types | ✅ Complete | 9 types: hist, scatter, profile, hist2d, hexbin, profile2d, scatter3d, cumulative hist, time-axis variants |
| Statistics | ✅ Complete | ROOT-parity + robust + range-aware + structured return |
| Composition | ✅ Complete | Channel framework Algorithm A; 6-axis grammar |
| Differential ops | ✅ Complete | `selection_vector=`, `weights_vector=`, `normalize=delta/ratio/pull` |
| Faceting | ✅ Complete | 1-D channel, 1-D column, N-D `[row,col,figid]` |
| Inline fitting | ✅ Complete | gaussian, linear, polN, lorentz, expo, powerlaw, custom |
| Summary fit | ✅ Complete | Per-group fit tables + figures |
| Test suite | ✅ Complete | 1,057 passing + 1 xfailed + 1 skipped · 127 features · 356 invariance tests · 59 verified · 27 visual-primitive |
| Documentation | ✅ Complete | Technical Summary v6.3; PHASE_HISTORY; ARCHITECT_DECISIONS; CAPABILITY_MATRIX |
| Stack integration | ✅ Complete (data/analysis); ⏳ planned (interactive) | ADF lazy + GB subframes integrated at the grammar level |

### Production users (validation layers built on the library)

ALICE: TPC calibration QA, ITS/TRD alignment, multiplicity, time-series QA, and physics analysis (V0 → track → collision → calibration via ADF subframe chains). Production-validated convergence across calibration cycles; physics analysis follows the same subframe-chain pattern. See `CAPABILITY_MATRIX.md` and `PHASE_HISTORY.md` for per-phase details. Specific quantification of convergence and pipeline-time improvements deferred pending official confirmation, consistent with the same quantification discipline in `dfdraw_Technical_Summary.md` v6.3.

---

## §14. Roadmap — what's next

### Short-term (Phase 13.51+)
1. **`time_series.py` v2 extension** — production canonical example covering 9 currently-missing features (TS §A.6 backlog)
2. **`time_format=` on y-axis** — symmetric extension to y-axis time formatting (currently x-axis only)
3. **ADF–dfdraw interface update** — formalize the coupling between aliases and channel framework
4. **GP-9 codification in ARCHITECT_DECISIONS.md** — archival discipline for chat-only deliberation

### Medium-term (TS v6 cycle close + cross-team feedback)
1. **`dfextensions_VISION.md`** — extract motivation source materials from chat into a permanent foundations doc
2. **API_REFERENCE.md rename to `dfdraw_api_summary.md`** — close the cross-link discrepancy
3. **mplhep-style experiment style sheets** — ALICE-house publication styling (currently relies on matplotlib base)
4. **Cross-team feedback integration** from ADF / GB / O2DistAI / TimeAI / calibration audiences

### Long-term (multi-phase)
1. **RootInteractive integration bridge** — dfdraw → RootInteractive HTML dashboard export
2. **Polars-DataFrame backend** — investigate Polars as an ADF backend for very-large workflows (currently pandas-bound)
3. **AI-agent for online help** (Phase TS_v6 forward direction §A.7)

---

## §15. References

- **Stack foundation**
  - `dfdraw_Technical_Summary.md` v6.3 — the canonical reference for the stack identity
  - `ARCHITECT_DECISIONS.md` v1.0.1 — decision registry (AD-N, AD-N/PHASE.DF)
  - `PHASE_HISTORY.md` — chronological phase log
  - `CAPABILITY_MATRIX.md` — feature inventory at HEAD

- **HEP ecosystem**
  - ROOT TTree::Draw: https://root.cern/doc/master/classTTree.html
  - Scikit-HEP: https://scikit-hep.org/
  - uproot: https://uproot.readthedocs.io/
  - awkward: https://awkward-array.org/
  - hist + boost-histogram: https://hist.readthedocs.io/, https://boost-histogram.readthedocs.io/
  - mplhep: https://github.com/scikit-hep/mplhep (1.0 released January 2026)
  - coffea: https://github.com/scikit-hep/coffea
  - RootInteractive: https://github.com/miranov25/RootInteractive
  - RootInteractive paper: https://arxiv.org/abs/2403.19330

- **Declarative grammar / general Python plotting**
  - Vega-Altair: https://altair-viz.github.io/ (6.1 dev as of Q2 2026)
  - Vega-Lite: https://vega.github.io/vega-lite/
  - Plotly Express: https://plotly.com/python/plotly-express/
  - HoloViews / hvPlot: https://holoviews.org/, https://hvplot.holoviz.org/
  - Polars + Altair plotting: https://docs.pola.rs/api/python/stable/reference/dataframe/plot.html

- **R analog (the architectural reference)**
  - ggplot2: https://ggplot2.tidyverse.org/
  - broom: https://broom.tidymodels.org/
  - lme4: https://cran.r-project.org/package=lme4

- **Industry background**
  - Stonebraker & Pavlo, "What Goes Around Comes Around... And Around" (2024) — declarative vs blob-storage architectural history

---

## §16. Revision History

| Version | Date | Changes | Author |
|---|---|---|---|
| 1.0 | 2026-01-29 | Initial comparison document; proposed wrapper architecture | Claude2-dfdraw |
| 1.1 | 2026-01-29 | Phase 13.6.G.DF completion — statistics ROOT parity achieved | Claude2-dfdraw |
| 1.2 | 2026-04-09 | Phase 13.17.DF refresh — added Phase 13.12–13.16 capabilities (auto-title, `same=True`, `draw_batch`, vector expressions, AD-37 fix); backend abstraction marked as not-pursued | Claude40 |
| **2.0** | **2026-06-06** | **Major restructure — Phase TS_v6 cycle.** Aligned with TS v6.3 stack identity (dfdraw + ADF + GB integrated framework). Added: §2 Stack Identity, §5 ggplot2 + broom + lme4 comparison, §6 grammar-layer comparison (Vega-Altair, Plotly Express, HoloViews, Polars + Altair), §7 HEP ecosystem comparison (Scikit-HEP, coffea), §9 worked examples (side-by-side code for 3 patterns), §10 expanded capability matrix through Phase 13.50, §11 architectural-history note on the wrapper layer that was not built, §12 feature parity checklist through Phase 13.50, §14 current roadmap. RootInteractive §8 preserved per architect. References expanded (Scikit-HEP, Polars + Altair, R tidyverse). Pre-commit panel review (7 reviewers) and architect review applied four fixes: (F1) §13 production-users line — specific numeric quantifications of convergence, improvement factors, pipeline timing, iteration counts, and fit/group counts deferred per architect 2026-06-06 — consistent with the same conservatism applied in TS v6.3 (only the ×2.4 σ(pT)/pT degradation is officially confirmed at this time); (F2) §7 removed unverifiable Scikit-HEP date claim; (F3) §10 capability matrix corrected mplhep time-axis cell to `manual` (mplhep adds no time-axis API of its own); (F4) §7 corrected the framing of hierarchical HEP data — ADF + dfdraw handles event-level hierarchical data (V0 → track → collision → calibration) through subframe registration with index-based foreign-key joins, equivalent in capability to awkward's nested-array abstraction; physics analysis added to §13 production-users line as a primary use case alongside calibration QA; "Event-level data" and "Distributed processing" rows in §7 comparison table revised to reflect ADF's subframe hierarchy rather than implying flat-pandas-only. | Opus2 (restructure) + Opus1 (panel-fix pass: F1, F2, F3, F4) |
| **2.1** | **2026-06-06** | **Cross-team panel-fix pass — 4 P1 source-verified fixes.** Cross-team panel review (8 reviewers: Sonnet54/55/56/57/58/59 + Sonnet1/ADF + Claude11/Arch) returned [X] CHANGES REQUESTED with 4 P1 runtime-breaking errors found by deep grep + source verification against `drawer.py` / `_summary_fit.py` / `_autorange.py`. All four are mechanical fixes preventing runtime errors for readers who copy-paste code examples: (P1-A) `summary_fit=True` → `summary_fit='table'` — `summary_fit=True` raises `ValueError` per `_summary_fit.py:230-237` (accepted types are `str | list | dict | None`); fixed in §4 table, §5 ggplot2 comparison, §12 checklist. (P1-B) `range_x=`, `range_y=` → `range=` — phantom kwargs that raise `TypeError`; public kwarg is `range=`; fixed in §4 "Range-aware stats" row. (P1-C) `legend_stats_fields=[...]` → `stats=True` or `stats=["mean","std","n"]` — phantom kwarg that raises `TypeError`; public kwarg is `stats=`; fixed in §4 "Statistics box" row. (P1-D) §10 capability matrix and §12 checklist: Robust hybrid autorange attributed to Phase 13.36 corrected to Phase 13.28 (per `_autorange.py:2`, confirmed by Sonnet57/58/59 grep verification). No structural changes; document content preserved. P2 items deferred to v2.2 / future iteration (8 items including "9 types" → "8 types", faceting phase labels, 3-level dot chain confirmation, Example B column-name consistency). | Opus1 (cross-team-panel-fix pass: P1-A, P1-B, P1-C, P1-D) |
