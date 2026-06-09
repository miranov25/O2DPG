# dfdraw Technical Summary

**Library version:** HEAD `9a950c7b` (Phase 13.52.DF v1.5.1)
**Library scope:** 1,101 tests passing · 1 xfailed · 2 skipped · 133 features · 356 invariance tests · 59 Verified · 27 visual-primitive tests *(these are library-level test counts; production use cases — ALICE TPC calibration, ITS/TRD alignment, multiplicity calculation, time-series QA — have separate validation layers built on top)*
**Date:** 2026-06-09
**Audience:** Users and integrators of the dfdraw + AliasDataFrame + GBregression analysis stack — including ADF/GB team members, architects, developers, and downstream consumers of dfdraw output (per-bin stats, fit results) in calibration, QA, and broader scientific-DataFrame visualization workflows
**Purpose:** Conceptual reference for dfdraw — the *what* and *why*. For *how to start*, see `dfdraw_README.md` (user guide). For *how to call exactly*, see `API_REFERENCE.md` (complete API reference; rename to `dfdraw_api_summary.md` pending). For comparison with other visualization stacks, see `dfdraw_PLOTTING_LIBRARY_COMPARISON.md`. Companion documents: `PHASE_HISTORY.md` (chronological log), `CAPABILITY_MATRIX.md` (feature inventory), `ARCHITECT_DECISIONS.md` (decision registry). The ADF and GB technical summaries are at `../../AliasDataFrame/docs/` and `../../groupby_regression/docs/` respectively.

> **⚠️ MIGRATION ALERT.** This release contains **two backward-incompatible changes** (Phase 13.50): `fit.text_format` style key removed; `summary_fit.precision` style key removed. **If you are upgrading from a release prior to `PHASE_13_50_DF_END`, read §8 Migration Notes before upgrading.** Permanent back-compat parallels are preserved where possible (e.g., `show_legend=` remains supported alongside the new `legend=` polymorphic kwarg).

> **Document organization.** §1–§5 describe **what dfdraw does** (functionality reference, organized by capability). §6 is **technical API reference** (methods, kwargs, return contracts). §7–§8 are **scope boundaries and migration notes**. Appendices cover **architecture and governance** (A), **version history** (B), **performance characteristics** (C), and **decision matrix** (D). Phase chronology is in Appendix B; AD references appear as footnotes within capability sections.

---
## 0. Quick Start — Position in the Stack

**dfdraw is a declarative statistical visualization library for pandas DataFrames.** It implements a channel-aware composition grammar (see §1.4 Visualization Grammar) that lets users describe *what* to plot — data channels compose with statistical operations — and the framework chooses *how* to render them.

### 0.1 Two usage modes

1. **Standalone.** `DFDraw(df)` on any pandas DataFrame. Full feature surface — all plot types, composition, fits, statistics, normalize operations. Use when the data is already materialized as a flat table.

2. **Stack integration.** Within the **ADF + GB + dfdraw** stack. This is the production pattern for calibration / QA / alignment workflows:

   1. **AliasDataFrame** provides aliases (declarative derived columns), subframe joins, and lazy materialization. *The lazy layer is essential: multi-TB tracking datasets cannot be flat-materialized, but ADF resolves only the columns each plot needs.*
   2. **GBregression** provides multi-dimensional parallel fits (`make_parallel_fit_v4` and other fit methods), sliding-window fits, and prediction registration. Outputs are coefficient DataFrames consumed as ADF subframes.
   3. **dfdraw** renders declaratively against ADF data; consumes GB coefficient frames; returns `(fig, ax, stats)` for downstream consumption (GB regression, custom analysis, QA tooling).

See `examples/time_series.py` for the canonical end-to-end stack pattern. The stack flow diagram (§0.2 below) visualizes the data movement.

### 0.2 Stack flow

<svg viewBox="0 0 720 220" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="ADF + GB + dfdraw stack flow">
  <style>
    .box{fill:#f7f7fb;stroke:#2840c8;stroke-width:1.5}
    .lbl{font-family:-apple-system,Segoe UI,sans-serif;font-size:13px;fill:#14143a;font-weight:600}
    .sub{font-family:-apple-system,Segoe UI,sans-serif;font-size:10.5px;fill:#444466}
    .ann{font-family:Monaco,Consolas,monospace;font-size:10.5px;fill:#993380}
    .arr{stroke:#2840c8;stroke-width:1.5;fill:none;marker-end:url(#ar)}
  </style>
  <defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="#2840c8"/></marker></defs>
  <rect class="box" x="20" y="40" width="180" height="120" rx="8"/>
  <text class="lbl" x="110" y="65" text-anchor="middle">AliasDataFrame</text>
  <text class="sub" x="110" y="86" text-anchor="middle">aliases · subframes</text>
  <text class="sub" x="110" y="102" text-anchor="middle">lazy evaluation</text>
  <text class="ann" x="110" y="132" text-anchor="middle">df.shape: (1M+, 50)</text>
  <text class="sub" x="110" y="150" text-anchor="middle" style="font-style:italic">data layer</text>
  <rect class="box" x="270" y="40" width="180" height="120" rx="8"/>
  <text class="lbl" x="360" y="65" text-anchor="middle">GBregression</text>
  <text class="sub" x="360" y="86" text-anchor="middle">parallel / sliding fits</text>
  <text class="sub" x="360" y="102" text-anchor="middle">prediction registration</text>
  <text class="ann" x="360" y="132" text-anchor="middle">dfCoeffs.shape: (200, 12)</text>
  <text class="sub" x="360" y="150" text-anchor="middle" style="font-style:italic">analysis layer</text>
  <rect class="box" x="520" y="40" width="180" height="120" rx="8"/>
  <text class="lbl" x="610" y="65" text-anchor="middle">dfdraw</text>
  <text class="sub" x="610" y="86" text-anchor="middle">grammar · channels</text>
  <text class="sub" x="610" y="102" text-anchor="middle">fits · stats output</text>
  <text class="ann" x="610" y="132" text-anchor="middle">(fig, ax, stats)</text>
  <text class="sub" x="610" y="150" text-anchor="middle" style="font-style:italic">visualization layer</text>
  <path class="arr" d="M200 100 L270 100"/>
  <path class="arr" d="M450 100 L520 100"/>
  <text class="sub" x="235" y="92" text-anchor="middle">resolved DF</text>
  <text class="sub" x="485" y="92" text-anchor="middle">coeff subframe</text>
  <text class="sub" x="360" y="195" text-anchor="middle" style="font-style:italic">Figure 0.2 — ADF + GB + dfdraw stack data flow. Each layer composes through documented contracts.</text>
</svg>

### 0.3 Minimal end-to-end example

The full ADF + GB + dfdraw pattern, drawn from `examples/time_series.py`:

```python
from dfextensions.AliasDataFrame import AliasDataFrame
from dfextensions.groupby_regression import make_parallel_fit_v4
from dfextensions.dfdraw import set_style

# 1. ADF — load data, declare aliases, enable lazy evaluation
adf = root_to_adf("time_series_tracks_0.root")
adf.draw_lazy = True
apply_meta(adf, df_TimeSeriesAliases)                  # aliases + axis titles

# 2. GBregression — multi-dimensional parallel fit produces coefficients
_, dfCoeffs = make_parallel_fit_v4(
    df=adf.df,
    gb_columns=["sector_bin180", "tgl_bin10"],
    fit_columns=["dca_y", "dca_z"],
    linear_columns=["qpt"],
    addPrediction=False,
    selection=isOK,
)
adf.register_subframe("CalibFit", AliasDataFrame(dfCoeffs),
                      index_columns=["sector_bin180", "tgl_bin10"])

# 3. dfdraw — visualize through adf.draw() (ADF resolves, dfdraw renders)
set_style({"figure.figsize": (12, 8)})
fig, ax, stats = adf.draw(
    "dca_y:tgl_bin10",
    facet_by="sector_bin180", facet_by_bins=4,
    group_by="qpt", group_by_quantiles=5,
    fit="gauss",
)
```

For full integration tutorials:
1. **ADF integration patterns:** see `examples/time_series.py` and the ADF team's Technical Summary at `../../AliasDataFrame/docs/` (authoritative for `adf.draw_lazy`, alias resolution, subframe lifecycle).
2. **GB regression integration:** see the GB team's Technical Summary at `../../groupby_regression/docs/` (authoritative for `make_parallel_fit_v4`, `make_sliding_window_fit`, and other fit methods, prediction registration patterns).
3. **dfdraw-only usage** (without ADF/GB): construct `DFDraw(df)` directly — all sections below apply.

### 0.4 Interactive backends

In Jupyter, prepending `%matplotlib widget` enables pan/zoom/hover on all dfdraw plots at no cost (requires `ipympl`). Full interactive stack (linked brushing, server-side data) is deferred to RootInteractive integration in a future phase.


## Executive Summary

### The thesis

> **Understanding is prerequisite for high-quality data.**
> — M. Ivanov, dfextensions presentation, December 2025

> *"Most important is the lack of understanding. Worse resolution is just a consequence. Good software → understanding → results close to the intrinsic resolution. Bad software → total lack of understanding."*
> — M. Ivanov, dfdraw governance session, 2026-06-05

Resolution is a consequence; understanding is the cause. Calibration QA at LHC physics scale is not a plotting problem — it is the workflow that determines whether the detector's intrinsic resolution can be recovered or not. This Technical Summary describes a stack built around this thesis.

### Why this stack exists

ALICE Run 1 and Run 2 calibration QA was an **analyst-bandwidth** workflow — by design, not by accident. The analyst-facing layer was a downsampled queryable dataset produced continuously as part of production: data was either written in queryable format or actively cached and downsampled (typically O(10⁻⁴) sampling, chosen to make distributions roughly flat in the variables of interest). `TTree::Draw` queries ran in seconds against this layer. A physicist asked a calibration question — `TTree::Draw("dca_y:tgl:sector", "...", "prof")` — and saw the result. Hypotheses tested per day was bounded by analyst thinking speed, not by the framework.

The downsampling code itself was production infrastructure. Without it, declarative queries against full reconstruction data would have been impossible even in Run 1/2. This is a general pattern, not specific to HEP: large-scale data systems serve production and analysis through separate but coordinated layers; declarative queryability lives at the analysis layer, not the production layer.

Run 3's O2 / O2Physics framework was optimized for streaming production throughput; the analyst-facing downsampled layer was not included in the initial framework design. The consequences:

1. Calibration conditions are stored in CCDB as keyed binary blobs without a standard query interface, requiring bespoke C++ to read and interpret each one.
2. Each new calibration question requires bespoke imperative code — written, compiled, debugged, validated.
3. The cognitive cost of formulating a question shifted from the analyst (declarative one-liner, seconds) to the developer (imperative code, hours to days). Calibration QA became **developer-bandwidth-bound**, not analyst-thought-bound.

Production throughput and analyst-facing declarative query are two different design points serving different purposes; they are not substitutable. The industry-wide return from blob-store architectures to declarative query languages over the past decade — Snowflake, BigQuery, Databricks SQL, RDataFrame in ROOT, modern HEP analysis ecosystems — reflects the same lesson. *(Stonebraker & Pavlo, "What Goes Around Comes Around... And Around", 2024.)*

The traditional Run 3 approach shows three failure modes:

1. **New calibration → recompile C++ → days of delay.** Hypotheses are expensive; few are tested per cycle.
2. **Debugging → gdb, print statements → limited insight.** Each bug investigation is a custom project, not a query.
3. **Understanding data → write custom code → error-prone, often impossible.** The cognitive load is dominated by the cost of writing the code, not by the question itself.

The architect's principle:

> *"I do not trust custom code — I trust standard interactive queries and derived statistics. Validation is a statistical process requiring flexibility and interactivity."*
> — M. Ivanov, dfextensions presentation, December 2025

The queryability gap is being addressed from two architectural levels in parallel: the O2CCDBAI initiative addresses the calibration-DB storage layer, and the dfextensions stack described in this document addresses the analysis layer. The downsampled-dataset pattern from Run 1/2 continues: `examples/time_series.py` is precisely such a downsampled dataset, produced at production scale, and dfextensions provides the declarative grammar over it. The two efforts together continue the dual-layer architecture pattern that worked in Run 1/2, adapted to Run 3 data.

### The measurable consequence

The productivity collapse in calibration QA has a measurable downstream effect on physics. **σ(pT) / pT degrades by a factor of approximately ×2.4 in Run 3 relative to the best Run 2 calibration.** The detector's intrinsic resolution has not changed; the workflow's ability to converge to it has.

The stack is domain-general: calibration, alignment, simulation / MC data remapping, and physics analysis share the same grammar and return contract.

### The discovery goal

The stack is built around a specific epistemology:

> *"By extracting multidimensional differential maps, we can obtain maps that are already analytically tractable. That is usually our goal — to obtain a multidimensional function that we can understand and, ideally, also describe with an analytical model or analytically derived effective parameterization."*
> — M. Ivanov, dfextensions presentation, December 2025 (collective voice adaptation)

This is the inverse of opaque-model workflows. Each fit produces parameters that can be inspected; each correction reduces to coefficients with physical meaning; each iteration is a step toward an analytically tractable description rather than a more complex pipeline.

> *"Goal: production-ready composable tools with C++ speed."*
> — M. Ivanov, dfextensions presentation, December 2025

### What dfdraw is

A declarative statistical visualization library for pandas DataFrames, implementing a channel-aware composition grammar. The grammar has three layers:

1. **Data dimensions** — `vector`, `group_by`, `facet_by`, `*_bins` / `*_quantiles`, `selection_vector`, `weights_vector` — compose orthogonally. Six axes can be exercised in one call.
2. **Visual channels** — color, linestyle, marker, spatial position — assigned automatically by Algorithm A. No collisions; each data dimension gets a distinct channel.
3. **Statistical operations** — `normalize=delta/ratio/pull`, inline `fit=`, summary fit figures, robust statistics — compose with the data dimensions, not bolted on.

See §1.4 Visualization Grammar for the full mapping and a six-axis composition example.

### The stack — restoring the analyst-bandwidth iteration model

dfdraw is one of three components of an integrated statistical analysis framework. **None of the three alone covers the workflow; the stack identity matters.**

1. **AliasDataFrame (ADF)** — data layer. Declarative derived columns (aliases) defined once and referenced anywhere. Subframe joins (e.g., GB-regression coefficient frames register as subframes). **Lazy evaluation is essential:** multi-TB tracking datasets cannot be flat-materialized into pandas. ADF resolves only the columns each plot needs, and only when the plot is drawn. Without lazy evaluation, the stack does not work at the data scales we operate on.
2. **GBregression** — analysis layer. Multi-dimensional parallel fits (`make_parallel_fit_v4` and other fit methods), sliding-window fits, prediction registration. Outputs are coefficient DataFrames consumed as ADF subframes. **N-D decomposition via ADF subframes + aliases is one of the most important functionalities of the stack:** GB produces coefficients → ADF registers as subframe → aliases expose model predictions → dfdraw renders `[raw, model]:x` with `normalize="delta"` in one call.
3. **dfdraw** — visualization layer. The grammar above + per-bin statistics + inline fits + the differential operations (`normalize=`, `selection_vector=`, `weights_vector=`). Output is `(fig, ax, stats)` — `stats` is structured and is what GB and downstream consumers read.

Together this stack is analogous to R's *ggplot2 + broom + lme4* integrated ecosystem — with no clean public Python equivalent. Standard composable tools rather than custom code; declarative queries rather than imperative scripts; statistics that flow back as data rather than locked behind plot rendering. Integrated with O2 data workflows, not replacing them.

### Core characteristics

1. **Declarative grammar** — describe what to encode (`"y:x"`, `group_by`, `facet_by`, `fit=`, `normalize=`); the framework resolves channel assignment, layout, legends, and statistics.
2. **Explicit aggregation** — `type='profile'`, `hist2d`, `hexbin` are user-chosen; no implicit `groupby().mean()` invocation.
3. **Statistics return** — every call returns `(fig, ax, stats)`; programmatically consumable downstream.
4. **Composition** — channel framework composes data dimensions with visual channels (color / linestyle / marker / spatial); N-D faceting (ROW × COL × FIGID).
5. **Integrated analysis** — inline fits, summary fit figures, normalize/delta/ratio/pull operations, robust statistics; not just rendering.
6. **Stack-native** — ADF aliases + lazy evaluation + GB regression coefficient subframes integrate at the grammar level (`adf.draw(...)`); standalone use also supported.
7. **Production-tested** — library scope: 1,057 tests passing · 1 xfailed · 1 skipped · 127 features · 356 invariance tests · 59 verified · 27 visual-primitive tests at HEAD `07606c02`. Independent production use cases (TPC calibration, ITS/TRD alignment, multiplicity, time-series QA) are additional validation layers.

### Comparison with other stacks

The strength is **N-D groupby decomposition** declared in a single call. Brief summary — see §1.4.4 for detail:

| Operation | pandas + matplotlib | ROOT TTree::Draw | ggplot2 (R) | dfdraw + ADF + GB |
|---|---|---|---|---|
| Profile y vs x | `groupby.mean()` + `plt.plot()` | `TTree::Draw("y:x", "", "prof")` | `geom_smooth()` | `draw("y:x", type="profile")` |
| 3 groups × 5 facets | nested loops + 15 subplots | manual loops | `facet_wrap(~f) + aes(color=g)` | `draw("y:x", group_by="g", facet_by="f")` |
| Differential ratio | ~20 lines | ~20 lines | manual | `draw(..., selection_vector=[A,B], normalize="ratio")` |
| Inline fits per group | scipy + manual annotation | `TF1::Fit` per histogram | `geom_smooth(method=)` | `draw(..., fit="gauss")` |
| Lazy materialization | none built-in | TTree branches | none | ADF `draw_lazy=True` |
| Multi-dim regression | scipy per group | `TF1::Fit` per slice | `lme4 / lm()` | GBregression (multiple methods) |
| **N-D decomposition** | very hard | very hard | works for 2–3 axes | declarative for N axes |

### Cross-references

1. **What's in this document:** §1–§5 functionality (plot types, grammar, composition, fitting, statistics, style, integration); §6 API reference; §7–§8 boundaries and migration; Appendices A–D.
2. `./PHASE_HISTORY.md` — chronological phase log.
3. `./CAPABILITY_MATRIX.md` — generated feature inventory at HEAD.
4. `./dfdraw_Technical_Summary.html` — rendered HTML companion.
5. `./dfdraw_PLOTTING_LIBRARY_COMPARISON.md` — detailed comparison.
6. `./API_REFERENCE.md` *(rename to `dfdraw_api_summary.md` pending)* — complete API reference.
7. `./ARCHITECT_DECISIONS.md` v1.0.1 — numbered decision registry; cited inline as `AD-N` / `AD-N/PHASE.DF`.
8. `../../AliasDataFrame/docs/` — ADF technical summary.
9. `../../groupby_regression/docs/` — GB regression technical summary.
10. `examples/time_series.py` — canonical end-to-end ADF + GB + dfdraw integration example; downsampled queryable dataset (see Executive Summary).


---
## 1. Plot Types and Composition

### 1.1 Available Plot Types

| Type | Method | Expression Syntax | Use Case |
|------|--------|------------------|----------|
| **Histogram (1D)** | `hist()` | `'x'` | Distribution of single variable |
| **Scatter (2D)** | `scatter()` | `'y:x'` | Individual points, correlations |
| **Profile (2D)** | `profile()` | `'y:x'` | Mean y per x bin with error bars |
| **Hist2D** | `hist2d()` | `'y:x'` | Density heatmap |
| **Hexbin** | `hexbin()` | `'y:x'` | Hexagonal binning (large datasets) |
| **Profile2D** (Phase 13.39) | `profile2d()` or `draw()` | `'z:y:x'` | Heatmap of aggregated z per (x, y) bin — §1.13 |
| **Scatter3D** (Phase 13.39) | `scatter3d()` or `draw(type='scatter3d')` | `'z:y:x'` | 3D scatter via `mpl_toolkits.mplot3d` — §1.14 |
| **Cumulative Histogram** (Phase 13.40) | `hist(cumulative=True/-1)` | `'x'` | Binned CDF; matches ROOT `TH1::Draw("cumulative")` — §1.15 |
| **Time-Axis** (Phase 13.39) | any plot method with `time_format=` | depends on plot | datetime64 / epoch-seconds x-axis — §1.16 |

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

### 1.3 Vector Expression Syntax

> **Vector broadcast extends beyond expression syntax.** The same broadcast convention applies to `selection_vector=[...]`, `weights_vector=[...]`, and per-curve fit lists `fit=[...]` — all couple element-wise to expression position. See §1.17 for differential comparison patterns and §2.4 for fit broadcast.


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

### 1.4 Visualization Grammar — Data Dimensions × Visual Channels

dfdraw's grammar separates **what data to show** (data dimensions) from **how to encode it visually** (visual channels). A single `draw()` call describes the data decomposition; Algorithm A assigns each data dimension to a distinct visual channel, with no collisions. This is the **declarative core** of the library.

#### 1.4.1 The six data dimensions

| # | Data dimension | What it encodes | Default visual channel | Example kwarg |
|---|---|---|---|---|
| 1 | `group_by="col"` | Discrete groups within a panel (overlay) | **Color** (one color per group) | `group_by="sec"` |
| 2 | `facet_by="col"` or `facet_by="channel"` | Separate panels (subplot grid) | **Spatial** (one panel per group) | `facet_by="side"` |
| 3 | Vector position in `"[y1, y2]:x"` | Multiple series in one expression | **Marker** or **linestyle** (cycled) | `"[dy_new, dy_ref]:row"` |
| 4 | `*_bins=N` / `*_quantiles=N` | Binning of a continuous group axis | Bin label (paired with color/marker) | `group_by="qpt", group_by_quantiles=5` |
| 5 | `selection_vector=[...]` | Differential pair (filter coupling) | Coupled to vector position | `selection_vector=["A", "C"]` |
| 6 | `weights_vector=[...]` | Weighted comparison | Coupled to vector position | `weights_vector=[None, "n_track"]` |

#### 1.4.2 Channel framework — visual mapping

<svg viewBox="0 0 740 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Channel framework: data dimensions to visual channels">
  <style>
    .ddim{fill:#eef1fb;stroke:#2840c8;stroke-width:1.3}
    .vchan{fill:#fff8e6;stroke:#c98a00;stroke-width:1.3}
    .lbl{font-family:-apple-system,Segoe UI,sans-serif;font-size:12px;fill:#14143a;font-weight:600}
    .arr{stroke:#2840c8;stroke-width:1.3;fill:none;marker-end:url(#a2)}
    .arrA{stroke:#888;stroke-width:1;fill:none;stroke-dasharray:3 3;marker-end:url(#a2g)}
    .cap{font-family:-apple-system,Segoe UI,sans-serif;font-size:10.5px;fill:#444466}
  </style>
  <defs>
    <marker id="a2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#2840c8"/></marker>
    <marker id="a2g" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="#888"/></marker>
  </defs>
  <text class="cap" x="80" y="22" text-anchor="middle" style="font-weight:600;text-transform:uppercase;letter-spacing:.06em;fill:#555568">Data dimensions</text>
  <text class="cap" x="640" y="22" text-anchor="middle" style="font-weight:600;text-transform:uppercase;letter-spacing:.06em;fill:#555568">Visual channels</text>
  <g><rect class="ddim" x="10" y="40" width="180" height="32" rx="5"/><text class="lbl" x="100" y="60" text-anchor="middle">group_by</text></g>
  <g><rect class="ddim" x="10" y="82" width="180" height="32" rx="5"/><text class="lbl" x="100" y="102" text-anchor="middle">facet_by</text></g>
  <g><rect class="ddim" x="10" y="124" width="180" height="32" rx="5"/><text class="lbl" x="100" y="144" text-anchor="middle">vector position [y1,y2]:x</text></g>
  <g><rect class="ddim" x="10" y="166" width="180" height="32" rx="5"/><text class="lbl" x="100" y="186" text-anchor="middle">*_bins / *_quantiles</text></g>
  <g><rect class="ddim" x="10" y="208" width="180" height="32" rx="5"/><text class="lbl" x="100" y="228" text-anchor="middle">selection_vector</text></g>
  <g><rect class="ddim" x="10" y="250" width="180" height="32" rx="5"/><text class="lbl" x="100" y="270" text-anchor="middle">weights_vector</text></g>
  <g><rect class="vchan" x="550" y="50" width="180" height="32" rx="5"/><text class="lbl" x="640" y="70" text-anchor="middle">Color</text></g>
  <g><rect class="vchan" x="550" y="92" width="180" height="32" rx="5"/><text class="lbl" x="640" y="112" text-anchor="middle">Spatial (subplot)</text></g>
  <g><rect class="vchan" x="550" y="134" width="180" height="32" rx="5"/><text class="lbl" x="640" y="154" text-anchor="middle">Marker</text></g>
  <g><rect class="vchan" x="550" y="176" width="180" height="32" rx="5"/><text class="lbl" x="640" y="196" text-anchor="middle">Linestyle</text></g>
  <g><rect class="vchan" x="550" y="218" width="180" height="32" rx="5"/><text class="lbl" x="640" y="238" text-anchor="middle">Bin label / pair</text></g>
  <path class="arr" d="M190 56 L550 66"/>
  <path class="arr" d="M190 98 L550 108"/>
  <path class="arr" d="M190 140 L550 150"/>
  <path class="arr" d="M190 182 L550 234"/>
  <path class="arr" d="M190 224 L550 234"/>
  <path class="arr" d="M190 266 L550 234"/>
  <path class="arrA" d="M190 56 L550 192"/>
  <path class="arrA" d="M190 140 L550 192"/>
  <text class="cap" x="370" y="310" text-anchor="middle">Solid arrows: default Algorithm A assignment.</text>
  <text class="cap" x="370" y="324" text-anchor="middle">Dashed: alternative under override (resolution chain: per-call *_style → channels.default.* → EXPLICIT_RULES → channels.overflow).</text>
  <text class="cap" x="370" y="338" text-anchor="middle" style="font-style:italic">Figure 1.4.2 — Channel framework. Each data dimension maps to a distinct visual channel; no collisions by default.</text>
</svg>

**Composition principle.** Each data dimension is assigned a distinct visual channel. No two dimensions share a channel by default. Algorithm A resolves channel assignment, legend topology, and rendering order. The default assignment can be overridden through a **multi-key resolution chain** (highest priority first):

1. **Per-call `*_style=` kwargs** — `quantile_style='color'`, `vector_style='marker'`, etc., applied at a single `draw()` call.
2. **`channels.default.*` style keys** — set globally via `set_style({'channels.default.quantile': 'color', ...})` to change project-wide defaults.
3. **`EXPLICIT_RULES`** — the immutable channel-precedence table inside `channels.py`; the architecture default.
4. **`channels.overflow`** — when too many active data dimensions collide on a single channel, controls overflow behavior (`'warn'` truncates and warns; `'error'` raises).

Appendix A.1.2 details the algorithm and the full priority table.

#### 1.4.3 Six axes in one call — the multi-dimensional differential pattern

The grammar's strength is exercising many orthogonal axes in a single declarative call. The TPC calibration QA pattern uses six:

```python
# TPC calibration QA: sector × side × residual-class × time-decomposition × momentum-quantile × fit
adf.draw(
    "[dy_new, dy_old]:row",                                 # vector → marker (axis 1)
    selection_vector=["time_s < t_mid", "time_s >= t_mid"], # paired filters (axis 2)
    normalize="delta",                                      # differential mode
    group_by="sec",                                         # → color (axis 3)
    group_by_quantiles=5,                                   # → bin labels (axis 4)
    facet_by="side",                                        # → subplot (axis 5)
    fit="gauss",                                            # → per-group/cell fit (axis 6)
)
```

This call produces: 2 facet panels × 5 sector-quantile groups × 2 epoch-filtered curves with delta normalization and gaussian fits per group/cell. The resulting `stats['fit']` is structured for GBregression consumption (see §2.8).

#### 1.4.4 Multi-dimensional decomposition — visual

The single call above decomposes hierarchically into the visual cross-product:

<svg viewBox="0 0 740 320" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Multi-level groupby decomposition tree">
  <style>
    .root{fill:#14143a;stroke:none}
    .lvl1{fill:#2840c8;stroke:none}
    .lvl2{fill:#5060d0;stroke:none}
    .lvl3{fill:#888aa0;stroke:none}
    .nodelbl{font-family:-apple-system,Segoe UI,sans-serif;font-size:12px;fill:#fff;font-weight:600}
    .lblD{font-family:-apple-system,Segoe UI,sans-serif;font-size:11px;fill:#14143a}
    .ln{stroke:#888;stroke-width:1;fill:none}
    .cap{font-family:-apple-system,Segoe UI,sans-serif;font-size:10.5px;fill:#444466}
  </style>
  <g>
    <rect class="root" x="320" y="20" width="100" height="30" rx="5"/>
    <text class="nodelbl" x="370" y="40" text-anchor="middle">draw()</text>
  </g>
  <path class="ln" d="M370 50 L150 90"/>
  <path class="ln" d="M370 50 L590 90"/>
  <g><rect class="lvl1" x="100" y="90" width="100" height="26" rx="4"/><text class="nodelbl" x="150" y="108" text-anchor="middle">facet: side=A</text></g>
  <g><rect class="lvl1" x="540" y="90" width="100" height="26" rx="4"/><text class="nodelbl" x="590" y="108" text-anchor="middle">facet: side=C</text></g>
  <path class="ln" d="M150 116 L60 150"/>
  <path class="ln" d="M150 116 L110 150"/>
  <path class="ln" d="M150 116 L160 150"/>
  <path class="ln" d="M150 116 L210 150"/>
  <path class="ln" d="M150 116 L260 150"/>
  <g><rect class="lvl2" x="35" y="150" width="50" height="22" rx="3"/><text class="nodelbl" x="60" y="166" text-anchor="middle" style="font-size:10px">sec Q1</text></g>
  <g><rect class="lvl2" x="90" y="150" width="50" height="22" rx="3"/><text class="nodelbl" x="115" y="166" text-anchor="middle" style="font-size:10px">sec Q2</text></g>
  <g><rect class="lvl2" x="145" y="150" width="50" height="22" rx="3"/><text class="nodelbl" x="170" y="166" text-anchor="middle" style="font-size:10px">sec Q3</text></g>
  <g><rect class="lvl2" x="200" y="150" width="50" height="22" rx="3"/><text class="nodelbl" x="225" y="166" text-anchor="middle" style="font-size:10px">sec Q4</text></g>
  <g><rect class="lvl2" x="255" y="150" width="50" height="22" rx="3"/><text class="nodelbl" x="280" y="166" text-anchor="middle" style="font-size:10px">sec Q5</text></g>
  <path class="ln" d="M590 116 L500 150"/>
  <path class="ln" d="M590 116 L555 150"/>
  <path class="ln" d="M590 116 L600 150"/>
  <path class="ln" d="M590 116 L655 150"/>
  <path class="ln" d="M590 116 L705 150"/>
  <g><rect class="lvl2" x="478" y="150" width="50" height="22" rx="3"/><text class="nodelbl" x="503" y="166" text-anchor="middle" style="font-size:10px">sec Q1</text></g>
  <g><rect class="lvl2" x="533" y="150" width="50" height="22" rx="3"/><text class="nodelbl" x="558" y="166" text-anchor="middle" style="font-size:10px">sec Q2</text></g>
  <g><rect class="lvl2" x="588" y="150" width="50" height="22" rx="3"/><text class="nodelbl" x="613" y="166" text-anchor="middle" style="font-size:10px">sec Q3</text></g>
  <g><rect class="lvl2" x="643" y="150" width="50" height="22" rx="3"/><text class="nodelbl" x="668" y="166" text-anchor="middle" style="font-size:10px">sec Q4</text></g>
  <g><rect class="lvl2" x="698" y="150" width="50" height="22" rx="3"/><text class="nodelbl" x="723" y="166" text-anchor="middle" style="font-size:10px">sec Q5</text></g>
  <path class="ln" d="M60 172 L48 200"/>
  <path class="ln" d="M60 172 L72 200"/>
  <g><rect class="lvl3" x="30" y="200" width="36" height="18" rx="3"/><text class="nodelbl" x="48" y="213" text-anchor="middle" style="font-size:9px">early</text></g>
  <g><rect class="lvl3" x="54" y="200" width="36" height="18" rx="3"/><text class="nodelbl" x="72" y="213" text-anchor="middle" style="font-size:9px">late</text></g>
  <text class="cap" x="48" y="240" text-anchor="middle" style="font-size:9px">[dy_new]</text>
  <text class="cap" x="72" y="240" text-anchor="middle" style="font-size:9px">[dy_old]</text>
  <text class="cap" x="60" y="256" text-anchor="middle" style="font-style:italic">delta + fit</text>
  <text class="lblD" x="370" y="125" text-anchor="middle">facet_by="side" → 2 panels</text>
  <text class="lblD" x="370" y="190" text-anchor="middle">group_by="sec" + quantiles=5 → 5 colors / panel</text>
  <text class="lblD" x="370" y="220" text-anchor="middle">selection_vector → 2 epochs / curve</text>
  <text class="lblD" x="370" y="240" text-anchor="middle">vector [y1,y2] → 2 markers / curve · normalize="delta" · fit="gauss"</text>
  <text class="cap" x="370" y="305" text-anchor="middle" style="font-style:italic">Figure 1.4.4 — Six-axis decomposition. 2 × 5 × 2 × 2 = 40 fit results from one declarative call.</text>
</svg>

This is the multi-dimensional differential analysis pattern the entire stack is designed to support. See §1.17 for `selection_vector` / `weights_vector` and §2 for fitting details.

---

### 1.5 Quantile Rendering on Profile

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

**Parameters on `profile()`:**
- `quantiles: Optional[List[float]]` — list of fractions in (0, 1)
- `central: Optional[str]` — `'mean'` (default), `'median'`, `'both'`, `'none'`
- `quantile_mode: str = "auto"` — `'auto'` dispatches via `_detect_quantile_mode()`; explicit override available with `'discrete'`, `'band'`, `'error_bars'`, `'nested_band'`

> **Phase 13.51 update (audit S-2):** `central='median'` now correctly renders the median line for the non-grouped 1D and 2D profile paths (fixes at `profile.py:879` for the rendering site and `:907` for the inline-fit `y_data`, plus explicit forwarding to `draw_profile2d` for the 2D path). The grouped path (`group_by=`) does NOT yet support `central='median'` — `profile.py:1347`/`:1359` still use hardcoded `bin_means` and the per-group `_central_values` is not computed; tracked as `KNOWN.grouped_central_median` for Batch 4 / a future phase. Previously these silent-no-op cases ignored `central='median'` entirely; behavior change is `[ADDITIVE]` because no documented correct caller existed (all callers passing `central='median'` were getting silently-wrong mean).

**Stats dict keys when quantiles set:** `q_lower_per_bin`, `q_upper_per_bin`, `quantiles_per_bin` (per mode).

**Style keys (4):** `quantile.band.alpha`, `quantile.band.hatch`, `quantile.error_bars.capsize`, `quantile.central_default`.

### 1.6 Channel-Aware Visual Encoding

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

**Parameter on `profile()`:**
- `quantile_style: Optional[str]` — channel for quantile dimension when combined with group_by

**Style keys (10):** `channels.priority.{categorical,ordinal}`, `channels.cycles.{linestyle,marker,color_count}`, `channels.default.{vector,group_by,quantiles}`, `channels.overflow`, `channels.legend.factored`.

**Capacity:** Each channel has a cycle limit (`channels.cycles.color_count=10`, `linestyle` has 4 entries by default, `marker` has 8). Overflow mode is `'error'` by default with actionable suggestions (`top_k=`, `facet=True`, `group_by_bins=`); `'warn'` permits truncation with warning.

**Factored legend** (default `True`): entry count is sum of cardinalities, not product. 3-channel call with `|group|=5, |vector|=3, |quantiles|=5` produces 13 entries factored vs 75 unfactored.

**Nested-band auto-detection (AD-57):** Symmetric quantile lists with ≥4 non-0.5 entries auto-route to `'nested_band'` mode (alpha-stacked filled regions). Central line handled independently via `central=` parameter.

### 1.7 Faceting — Subplot Grids

Faceting splits a single plot into a grid of subplots, one panel per value of a chosen data dimension. Use it to compare the same statistic across different categorical or binned splits without manual subplot bookkeeping. The framework handles layout, axis sharing, legend position, and per-panel statistics.

**Three faceting modes** (all four plot types — `profile`, `hist`, `hist2d`, `scatter` — support them since Phase 13.32):

1. **Facet by channel name** — pass a channel name (`'group_by'`, `'vector'`, `'quantiles'`, `'selection_vector'`, `'weights_vector'`) to use the existing data dimension as the facet axis.
2. **Facet by DataFrame column name** — pass any column name; one panel per unique value (`facet_by="side"` → one panel per detector side). Symmetric to `group_by`. Details in §1.9 Column-Name Faceting.
3. **N-D faceting** — pass a list (`['ROW', 'COL']` or `['ROW', 'COL', 'FIGID']`) for two- and three-dimensional subplot grids. Details in §1.7.4 below.

#### 1.7.1 Basic syntax

```python
# Facet by channel — one subplot per group value, no overlay
d.profile("y:x", group_by="sector", facet_by="group_by")

# Facet by vector — one subplot per vector y element
d.profile("[y1,y2,y3]:x", facet_by="vector")

# Facet by quantile (with quantile_mode='discrete')
d.profile("y:x", quantiles=[0.25, 0.5, 0.75],
          quantile_mode="discrete", facet_by="quantiles")

# Facet by DataFrame column directly
d.profile("y:x", facet_by="side")
```

#### 1.7.2 Mutual exclusion with `same=True`

`facet_by` and `same=True` raise `ValueError` (`"facet_by and same=True are mutually exclusive"`). Faceting creates new axes; `same=True` overlays onto existing axes — the semantics conflict.

#### 1.7.3 Capacity and overflow

Subplot count is bounded by `channels.cycles.facet_max=16`. Overflow behavior:
- `'warn'` truncates to the cap and emits a warning naming the truncated values
- `'error'` raises a `ValueError` with an actionable message (target values listed)

Backward-compat: `facet=True` ≡ `facet_by='group_by'`, byte-identical output locked by invariance test `test_facet_true_eqivalent_to_facet_by_groupby`.

#### 1.7.4 N-D Faceting — ROW × COL × FIGID (Phase 13.41)

For multi-dimensional decomposition where two or three orthogonal data dimensions need spatial separation, pass `facet_by` as a list:

```python
# 2D faceting: rows × columns
adf.draw(
    "dy:row",
    facet_by=["sector", "side"],            # rows = sector, cols = side
    facet_by_bins=[4, None],                # bin sector into 4 rows; side as-is
    group_by="qpt", group_by_quantiles=5,   # color overlay within each cell
    fit="gauss",
)
# Produces: a 4-row × 2-column grid of subplots, each with 5-color overlay + per-cell fit

# 3D faceting: ROW × COL × FIGID (one figure per FIGID value)
adf.draw(
    "dy:row",
    facet_by=["sector", "side", "fill"],   # produces multiple figures
    facet_by_bins=[4, None, 3],            # sector→4 bins, side as-is, fill→3 bins
    group_by="qpt", group_by_quantiles=5,
)
# Produces: 3 figures (one per fill bin), each a 4×2 grid
```

**Axis sharing.** By default, all subplots in a row share the x-axis; all subplots in a column share the y-axis. Override via `share_x=`, `share_y=` (valid tokens, per `drawer.py:69-77`: `'all'` / `'row'` / `'col'` / `'none'`).

**Return contract** (differs from standard `draw()` — see §6.1.4):
- 2D faceting returns `(fig, axes_array, stats_array)` where `axes_array.shape == (n_rows, n_cols)` and `stats_array` is a same-shaped object array
- 3D faceting returns `(figs_list, axes_arrays_list, stats_arrays_list)` — one entry per FIGID value

**Capacity bounds:** Each per-axis (ROW, COL) is independently bounded by `channels.cycles.facet_max=16`. **FIGID figures are unbounded** at the framework layer; for very high-cardinality FIGID columns, pre-filter the DataFrame or use `top_k=` to limit the number of figures emitted (the FIGID dispatch loop at `drawer.py:4085-4130` does not impose a cap).

**Implementation notes:** Phase 13.41 unified ROW/COL routing through the same `_dispatch_faceted_render` channel that 1D faceting uses; the only difference is the dimensionality of the axes array returned. AD-1/13.41.DF locks the return-shape semantics.

#### 1.7.5 Composition with other axes

Faceting composes with all other data dimensions:
- **`group_by`** — overlay within each panel
- **`vector`** — multi-curve per panel (`"[y1,y2]:x"`)
- **`selection_vector`** / **`weights_vector`** — differential pairs per panel
- **`fit=`** — analytic overlay per panel, with per-panel `stats['fit']` entries
- **`normalize=`** — `delta` / `ratio` / `pull` modes apply per panel

This is the multi-dimensional differential analysis pattern §1.4.3 named, scaled to the N-D faceting grid.

### 1.8 Robust Data Handling

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

**Parameter on all 5 plot methods (`draw / hist / hist2d / profile / scatter`):**
- `nan_policy: str = "filter"` — `'filter'` (silent drop + counters, default), `'warn'` (drop + UserWarning), `'raise'` (ValueError)

**`range=` extended:** Now accepts strategy strings `'hybrid'` (default), `'minmax'`, `'percentile_99'`, `'percentile_95'`, `'robust_3mad'`, `'robust_4mad'` in addition to numeric tuples.

**Stats dict keys (always populated):**
- Sanitize counters: `n_input`, `n_filtered`, `n_inf_x`, `n_nan_x`, `n_inf_y`, `n_nan_y`
- Autorange diagnostics: `autorange_used` (the actual numeric range used), `autorange_strategy` (`'hybrid'`, `'minmax'`, etc., or `'explicit'` when user passed numeric range)

**Critical semantics lock (AD-77):** `stats['n']` is the post-sanitize finite count, NOT range-clipped. Range filtering is **VISUAL ONLY** — `range=` does not reduce `stats['n']`.

**Style keys (5):** `data.nan_policy`, `autorange.strategy`, `autorange.k_robust`, `autorange.k_outlier`, `autorange.percentile`.

**Hybrid autorange algorithm:** Computes robust window `(median ± k_robust·sigma_MAD)`. Per-side, declares outlier on side S if data extreme exceeds median by more than `(k_outlier · k_robust · sigma_MAD)`. Uses robust bound when outlier present, else uses data extreme. Clean Gaussians get full `(min, max)`; outlier-bearing data gets clipped only on the affected side. 2D autorange is per-axis independent.

---

### 1.9 Column-Name Faceting

`facet_by=` accepts DataFrame column names in addition to channel-name enum. Triggered by the production reproducer `adfSec.draw("dyp_I0345_rms:row", facet_by="side", ...)` — natural mental model matching ggplot2 `facet_wrap(~side)`.

```python
# Faceting by a column value — one subplot per unique value in df['side']
d.profile("y:row", facet_by="side", group_by="z", group_by_bins=5)
# Returns: (fig, axes_array, stats_dict) — standard 3-tuple, axes_array is 1D over facet panels

# Original channel-name form still works (backward-compat lock — AD-78)
d.profile("y:row", facet_by="group_by")       # facets over group_by channel
d.profile("y:row", facet_by="vector")          # facets over vector expression
```

**Disambiguation order:** channel-name enum (`'group_by'`, `'vector'`, `'quantiles'`) checked first; then `df.columns` check; then `ValueError` naming both alternatives. Channel-name and column-name namespaces are disjoint by construction; the order is deterministic and documented.

*Decision rationale: AD-78 in `ARCHITECT_DECISIONS.md` (most-cited AD in codebase — 30 occurrences across 7 files).*

---

### 1.10 Symmetric `facet_by_bins` / `facet_by_quantiles`

Bin or quantile-bin the `facet_by` column before partitioning subplots — symmetric to Phase 13.12's `group_by_bins` / `group_by_quantiles` (F3).

```python
# Bin a continuous facet_by column into 4 panels
d.profile("y:row", facet_by="mP4", facet_by_bins=4)

# Quantile-bin instead (equal-population panels)
d.profile("y:row", facet_by="mP4", facet_by_quantiles=4)

# Compose with group_by within each facet panel
d.profile("y:row", facet_by="sec", facet_by_bins=4,
                   group_by="z", group_by_quantiles=5)
```

Binning is hoisted to `_dispatch_faceted_render` — single implementation across all 4 plot types (`profile`, `hist`, `hist2d`, `scatter`).

*Decision rationale: AD-79 in `ARCHITECT_DECISIONS.md`.*

---

### 1.11 Normalized Differential Profiles

Compare two profiles directly — delta, ratio, or pull — without manual subtraction. For calibration QA: residuals between fills, epochs, or correction iterations.

```python
# Vector expression: vector[0] = signal (new), vector[1] = reference
fig, ax, stats = d.profile("[dy_new, dy_ref]:row", normalize="delta")
# Renders delta = dy_new - dy_ref per row bin with combined error bars

# Ratio mode
d.profile("[dy_new, dy_ref]:row", normalize="ratio")    # v[0] / v[1]

# Pull mode — bands at ±1σ and ±2σ for significance reference
d.profile("[dy_new, dy_ref]:row", normalize="pull")
# Formula: (v[0] − v[1]) / √(σ₀²/n₀ + σ₁²/n₁)

# Composes with group_by AND facet_by from day one
d.profile("[dy_new, dy_ref]:row", normalize="delta",
          group_by="sector", facet_by="side")
```

**Sign convention (AD-80 — locked):**

| Vector position | Role | In formula |
|---|---|---|
| `vector[0]` | signal / new | numerator / subject |
| `vector[1]` | reference | denominator / baseline |
| **`delta`** | `v[0] − v[1]` | positive when signal > reference |
| **`ratio`** | `v[0] / v[1]` | matches ROOT `h_data.Divide(h_mc)` |
| **`pull`** | `(v[0] − v[1]) / √(σ₀²/n₀ + σ₁²/n₁)` | dimensionless significance |

> **Read this twice.** The convention is locked by AD-80 — `vector[0]` is the signal, `vector[1]` is the reference. Production sign-violation locks are in `tests/test_normalize.py §9.NSC.1`.

**Style keys:** `normalize.pull.band_1sigma_alpha`, `normalize.pull.band_2sigma_alpha`, `normalize.panel.ref_line_color`, `normalize.panel.ref_line_style`. *(Phase 13.51 Batch 1 item 12: dot-notation keys replace underscore form.)*

*Decision rationale: AD-80 (reference convention), AD-81 (group_by + facet_by both in-scope from v1.0), AD-82 (pull mode with ±σ bands) — all in `ARCHITECT_DECISIONS.md`.*

---

### 1.12 User Style Kwarg Precedence

When a user passes a style kwarg (`color=`, `marker=`, `linestyle=`, `markersize=`) at the call site, it takes priority over channel auto-assigned cycle values and applies **uniformly to ALL groups** in the call.

```python
# Without explicit kwarg — channel auto-cycle assigns per-group colors
d.profile("y:row", group_by="sec")
# Each sector gets a different color from the palette cycle

# With explicit kwarg — all groups rendered with same color
d.profile("y:row", group_by="sec", color="red")
# All sectors red — user kwarg overrides cycle uniformly

# Same rule for marker, linestyle, markersize
d.profile("y:row", group_by="sec", color="blue", linestyle="--")
```

**Sentinel rule:** `None` = user did not pass; non-`None` = user explicitly passed, apply uniformly. Named-param forwarding (not kwargs pop) — implemented as `_user_color`, `_user_marker`, `_user_markersize` parameters with `is None` checks before palette/cycle assignment.

*Decision rationale: AD-1/13.36.DF in `ARCHITECT_DECISIONS.md`.*

---

### 1.13 2D Profile — `profile2d`

Aggregates z over (x, y) bins, rendering a heatmap of the per-bin statistic. `z:y:x` expression syntax dispatches `draw_profile2d()` (triggered by `colon_count == 2`). `scipy.stats.binned_statistic_2d` backend.

```python
# Heatmap of mean dy per (row, sector) bin
d.profile2d("dy:row:sector")

# Explicit ranges — outer list is [x_range, y_range]
d.profile2d("dy:row:sector", range=[(0, 152), (0, 18)])

# With statistics other than mean — via stat_fields= passthrough
d.profile2d("dy:row:sector", stat_fields=["median"])
```

**Scope boundary:** `group_by` raises a clean `ValueError` on `profile2d` calls (explicit out-of-scope per AD-1/13.39.DF). The 2D heatmap channel surface is already saturated by the z-color axis.

*Decision rationale: AD-1/13.39.DF in `ARCHITECT_DECISIONS.md`.*

---

### 1.14 3D Scatter — `scatter3d`

Three-dimensional scatter rendering via `mpl_toolkits.mplot3d`. `type='scatter3d'` dispatches `draw_scatter3d()`. Stats dict locks `mean_x`, `mean_y`, `mean_z` (all to 1e-9 deterministic precision per `tests/test_phase_13_39_df_profile2d_timeaxis.py`). scipy is a required dependency for this path — no fallback.

```python
# Both equivalent
d.scatter3d("z:y:x")
d.draw("z:y:x", type='scatter3d')

# With viewing angle (matplotlib mplot3d kwargs are forwarded)
fig, ax, stats = d.scatter3d("z:y:x")
ax.view_init(elev=20, azim=45)
```

*Decision rationale: AD-3/13.39.DF in `ARCHITECT_DECISIONS.md`.*

---

### 1.15 Cumulative Histogram

Binned CDF via matplotlib's native `cumulative` parameter. Matches ROOT `TH1::Draw("cumulative")` semantics.

```python
# Ascending CDF
d.hist("y", cumulative=True)

# Descending CDF (1 - F(x))
d.hist("y", cumulative=-1)

# Normalized cumulative
d.hist("y", cumulative=True, hist_norm=True)

# Per-group CDF overlays
d.hist("y", cumulative=True, group_by="sec")
```

Composes with `hist_norm`, `group_by`, `facet_by`, `stacked`.

> **⚠️ Guard (AD-2/13.40.DF).** `hist_errors=True` combined with `cumulative=True/-1` raises `NotImplementedError`. Binned CDF with Poisson error bars is statistically ill-defined.

*Decision rationale: AD-1/13.40.DF (capability), AD-2/13.40.DF (guard) in `ARCHITECT_DECISIONS.md`.*

---

### 1.16 Time-Axis Rendering

The `time_format=` parameter renders the x-axis as datetime-formatted labels. Supports both `datetime64[ns]` columns and epoch-seconds (`float`/`int`) columns; dtype is auto-detected at dispatch.

```python
# datetime64 column
adf.draw("rate:timestamp", time_format="auto", type="profile")

# epoch-seconds column
adf.draw("rate:time_s", time_format="%Y-%m-%d", type="profile")
```

The `"auto"` format selects an appropriate matplotlib `DateFormatter` based on the data range (sub-second to multi-year).

**Current scope and planned extension.** Source verification at HEAD `07606c02` confirms `time_format=` applies the formatter to **the x-axis only** (`ax.xaxis.set_major_formatter`, `histogram.py:847`, `profile.py:961`). Symmetric y-axis time formatting is a planned extension — see the ADF/dfdraw interface roadmap; until shipped, y-axis time data must be pre-formatted before passing to dfdraw or rendered as ordinary numeric. *Decision rationale: AD-2/13.39.DF in `ARCHITECT_DECISIONS.md`.*

> **Phase 13.51 update (audit S-4):** the datetime64 + faceted dispatch crash at `_autorange.py:67` is fixed in this phase. The guard now lives at `compute_autorange()` entry and covers all 5 autorange strategies (`hybrid`, `robust_3mad`, `robust_4mad`, `percentile_99`, `percentile_95`, `minmax`). Full per-panel `DateFormatter` application in faceted layouts remains audit finding S-4 and is scheduled for Batch 4 / a future phase. `hist2d` gains `time_format=` symmetry in this phase (audit S-8 + I-1), bringing it to parity with `hist`/`scatter`/`profile`.


---

### 1.17 Selection Vectors, Weight Vectors, and Differential Comparison

**The core MDA (Multi-dimensional Differential Analysis) operation.** Used in every production differential comparison: geographic splits, temporal splits, before-vs-after corrections, distribution residuals. The vector-broadcast convention from §1.3 (scalar broadcast and per-curve lists) applies here too — `selection_vector`, `weights_vector`, and per-curve `fit=` lists couple element-wise to expression position.

The pattern: declare two (or more) selection criteria or weight schemes; the framework renders the comparison cross-product. Composes with `group_by`, `facet_by`, `normalize=`, and `fit=`.

#### 1.17.1 `selection_vector=` — two-sided selection comparison

```python
# Geographic split: A-side vs C-side of the detector
fig, ax, stats = adf.draw(
    "[dy_a, dy_c]:row",
    selection_vector=["side == 'A'", "side == 'C'"],
    normalize="delta",
)
# Renders δ = dy_a − dy_c per row bin; vector position 0 = signal (A), 1 = reference (C)

# Temporal split: early vs late in run
adf.draw(
    "[dy_early, dy_late]:row",
    selection_vector=["time_s < t_mid", "time_s >= t_mid"],
    normalize="ratio",
)
# Renders dy_early / dy_late; useful for run-stability QA
```

The vector expression `[signal, reference]:x` couples to `selection_vector=` element-wise: position 0 of the expression is filtered by selection 0; position 1 by selection 1.

#### 1.17.2 `weights_vector=` — multi-weight scheme comparison

```python
# Compare unweighted vs population-weighted vs error-weighted
adf.draw(
    "[dy_raw, dy_popw, dy_errw]:row",
    weights_vector=[None, "track_count", "1.0/dy_err**2"],
)
# Three curves, each using a different weight expression
```

`None` in `weights_vector=` denotes unweighted; other entries are column-name or expression strings evaluated against the DataFrame.

#### 1.17.3 `normalize=` — differential modes

Three modes turn a multi-vector overlay into a quantitative comparison. **Vector position convention (AD-80, locked):** `vector[0]` = signal, `vector[1]` = reference.

| Mode | Formula | Use case |
|---|---|---|
| `normalize="delta"` | `v[0] − v[1]` per bin | Residual analysis; positive = signal above reference |
| `normalize="ratio"` | `v[0] / v[1]` per bin | Multiplicative comparison; matches ROOT `h_data.Divide(h_mc)` |
| `normalize="pull"` | `(v[0]−v[1]) / √(σ₀²/n₀ + σ₁²/n₁)` | Significance test; dimensionless; bands at ±1σ, ±2σ rendered |

```python
# Pull mode with reference bands shown
adf.draw("[dy_new, dy_old]:row",
         selection_vector=["epoch == 'after'", "epoch == 'before'"],
         normalize="pull")
```

**Style keys** (Phase 13.33): `normalize.pull.band_1sigma_alpha`, `normalize.pull.band_2sigma_alpha`, `normalize.panel.ref_line_color`, `normalize.panel.ref_line_style`. *(Phase 13.51 Batch 1 item 12: dot-notation keys replace underscore form.)*

#### 1.17.4 Subframe-qualified vector form (ADF subframe merges)

When ADF has registered a subframe (e.g., a coefficient frame from GB regression — see §0 Quick Start), the vector expression can reference subframe columns directly:

```python
adf.register_subframe("CalibFit", AliasDataFrame(dfCoeffs),
                      index_columns=gbVars)
adf.draw("[vertex_x_intercept, CalibFit.vertex_x_intercept_decomp]:quantile_bin",
         normalize="delta")
# Compares raw vs decomposed (model-predicted) value through the subframe alias path
```

#### 1.17.5 Composition

`selection_vector=` / `weights_vector=` / `normalize=` all compose with:
- `group_by=` (overlays by another column within each selection/weight pair)
- `facet_by=` (subplot grid)
- `fit=` (a fit is computed on the differential result, not on the raw curves)

```python
# Full composition — sector-wise residuals across sides, with gaussian fit
adf.draw(
    "[dy_a, dy_c]:row",
    selection_vector=["side == 'A'", "side == 'C'"],
    normalize="delta",
    group_by="sec", facet_by="layer", facet_by_bins=4,
    fit="gauss",
)
```

#### 1.17.6 Stats output

`stats['normalize_mode']` records the mode used; per-bin numerator and denominator values are preserved under `stats['normalize_data']` for downstream consumption (Phase 13.51 Batch 1 item 8: key is 'normalize_data', not 'vector_data'). For GB regression workflows that consume differential outputs, see §5.4.

*Decision rationale: AD-80 (reference convention locked), AD-81 (group_by + facet_by both in-scope from v1.0), AD-82 (pull mode with ±σ bands) in `ARCHITECT_DECISIONS.md`. Implementation: Phase 13.27 Commit 2 (selection/weights vectors) + Phase 13.33 (normalize modes).*

### 1.18 Declarative Overlay — `type="A+B"` and `overlay(layers=[...])`

Phase 13.52.DF introduces composition of multiple plot primitives onto a single shared Axes. Two surfaces, one engine:

- **Sugar (ergonomic):** `adf.draw("y:x", type="hist2d+profile", bins=40, fit="gauss")` — a `+` in the `type` string desugars to layers and calls the engine.
- **Configurable:** `d.overlay("y:x", layers=[{"type":"hist2d","bins":40}, {"type":"profile","bins":15,"fit":"gauss"}])` — full per-layer control.

Both return `(fig, ax, {"layers": [stats_per_layer]})`. `stats["layers"][0]` is the base layer's stats; subsequent entries are overlays.

**Valid compositions.** Density base (one of `hist2d`, `hexbin`, `profile2d`) plus zero or more overlays from `profile`, `scatter`. The base layer is `layers[0]` and owns the colorbar; the engine enforces exactly one density layer per composition.

**Engine guards (clean `ValueError` BEFORE any draw, both surfaces):**
1. empty `layers`
2. non-method `type` token (e.g. `{"type":"quantiles"}` — that's a kwarg, not a type)
3. >1 density layer (two colorbars not supported)
4. non-density base layer (e.g. `"profile+hist2d"` — base must be a density type)
5. 3D layer (`scatter3d`) — no shared 2D axes
6. any faceting/`share_*` param on any layer — overlay composes onto one Axes; faceted overlay is Phase 13.53 scope

**String kwarg routing (per spec §1.4).** When using the sugar form, kwargs are routed as follows:

| Class | Examples | Routing |
|---|---|---|
| Layer-unique | `fit`→profile, `cmap`→hist2d | → owning layer |
| Whole-plot | `selection`, `sample`, `nan_policy`, `selection_vector`, `weights_vector` | → replicated to every layer |
| Shared, layer-specific | `bins`, `range`, `stats` | → base layer only |
| Faceting / `share_*` | `facet_by`, `share_x`, etc. | → `ValueError` (Phase 13.53 scope) |
| Not accepted by any layer | (typos, invented kwargs) | → `ValueError` with "use `layers=[...]`" hint |

For per-overlay-layer control of shared params (e.g. different `bins` on hist2d vs profile), use the `layers=[...]` form directly.

**`summary_fit` interaction with hist2d (Phase 13.51 cross-reference).** In the sugar form, `summary_fit=` routes to the hist2d base layer where it is a no-op without a corresponding `fit=`. Adding `fit=` to hist2d triggers Phase 13.51's S-8 guard (`hist2d() does not support fit= in Phase 13.51`). For a profile fit summary in a `hist2d+profile` composition, use the layers form: `layers=[..., {"type":"profile", "fit":"gauss", "summary_fit":"table"}]`.

**Guard-only param handling (Phase 13.52 implementation finding).** Phase 13.51 added `fit=` (hist2d) and `range=`/`facet_by=` (hexbin) as guard-only signature entries to surface clean errors instead of cryptic matplotlib failures. Without special handling, signature-based ownership inference in the desugar would route `fit=` to the hist2d base, where the guard would fire instead of the user's intent reaching profile. The desugar excludes these guard-only params from ownership inference via a `_OVERLAY_GUARD_PARAMS` mapping in `drawer.py`, restoring the spec §1.4 promise "layer-unique `fit→profile`". This honors the routing-policy intent while preserving the original S-7/S-8 guards on the direct primitive call paths.

**Range lock.** After the base layer renders, the engine captures `ax.get_xlim()` / `ax.get_ylim()` and re-applies them after every subsequent overlay layer. No primitive has a public `x_range=`/`y_range=` kwarg, so post-draw `set_xlim`/`set_ylim` is the synchronization mechanism. Holds after `plt.draw()` and `fig.canvas.draw()` (locked to `rtol=1e-6`).

**Z-order.** Density base draws first → indexed before overlays in `ax.collections`. Matplotlib walks `ax.collections` in addition order, so overlay artists (profile errorbar, scatter points) render on top of the density mesh naturally.

**Out of scope (Phase 13.53 candidates).** Faceted overlays (engine-rejected with a clear error message, not silently broken); density+density compositions; cross-layer legend synthesis; primitive signature changes; per-overlay-layer control of shared params via the string surface (use `layers=[...]`).

**Examples:**

```python
# Sugar form — most common usage
d.draw("y:x", type="hist2d+profile", bins=40, fit="gauss")

# Sugar with whole-plot kwargs (selection applies to both layers)
d.draw("y:x", type="hist2d+profile", selection="cut_pass", bins=40)

# Layers form — per-layer binning
d.overlay("y:x", layers=[
    {"type": "hist2d", "bins": 40, "cmap": "viridis"},
    {"type": "profile", "bins": 15, "fit": "gauss", "summary_fit": "table"},
])

# Hexbin alternative density base
d.overlay("y:x", layers=[
    {"type": "hexbin", "gridsize": 30},
    {"type": "profile", "bins": 20},
])
```

*Decision rationale: PHASE_13_52_OverlayProposal_v1_5.md (approved by Sonnet65 v1.4 panel, [!] 5/5 0 P1); Appendix C (faceting incompatibility reproduced by 3/5 v1.4 reviewers) drove engine-level rejection of faceting params on both surfaces. Implementation: Phase 13.52 (drawer.py engine + `_desugar_overlay` + sugar trigger; 19 invariance tests, 18 passing + 1 skip for profile2d-base shared-expr limitation).*

---
---
## 2. Fitting

dfdraw supports analytic curve fitting directly on plots. The fitting system covers inline fits attached to a plot, summary figures aggregating fit results across groups or facet cells, and a parameter-rendering convention oriented to physics-style presentation. Fit numeric results are exposed in the `stats` return for downstream consumption by GBregression and analysis pipelines.

### 2.1 Inline fits — `fit=` kwarg

Inline fits attach an analytic model to a plotted curve and overlay the fit result on the plot. Available on `profile()`, `hist()`, and `scatter()`. Three input forms:

```python
# Predefined function from the registry (most common)
fig, ax, stats = d.profile("y:x", fit="gauss")

# Dict spec with full control over range, initial values, options
d.profile("y:x", fit={
    "name": "gauss",
    "range": (0, 5),
    "initial": {"mu": 0, "sigma": 1},
})

# Custom callable — your own model function
def my_model(x, A, k):
    return A * np.exp(-k * x)
d.scatter("y:x", yerr="y_err", fit=my_model)
```

Backend: `scipy.optimize.curve_fit`. The fit curve is overlaid on the plot; the parameter values and errors render in an inline textbox; full numeric results are returned in `stats['fit']` for programmatic access.

### 2.2 Predefined function registry

| Name | Aliases | Function form |
|---|---|---|
| `gauss` | `gaussian`, **`gaus`** (ROOT TF1 alias, Phase 13.46) | A · exp(−(x − μ)² / 2σ²) |
| `linear` | — | a + b·x (rendered as p0 + p1·x) |
| `pol0` … `pol5` | — | Polynomial of degree N, ascending powers (p0, p1, p2, …) |
| `expo` | `exponential` | A · exp(b·x) |
| `lorentz` | `lorentzian` | Lorentzian distribution |
| `powerlaw` | — | A · x^b |

Additional shapes (landau, crystalball, breitwigner) are not in the registry as of HEAD `07606c02` — architect noted these "can be done later". Custom callable is always available as escape hatch.

### 2.3 Initial parameter resolution — 4-tier strategy

Highest priority first:

1. **User `initial` dict** in the fit spec
2. **User `guess` callable** that returns initial values from the data
3. **Registry heuristic** — default per-function estimator (e.g. gaussian heuristic derived from data mean and std)
4. **scipy default** + `UserWarning` fallback

```python
# (1) Explicit initial values
d.profile("y:x", fit={"fun": "gauss", "initial": {"mu": 0.5, "sigma": 0.1}})

# (2) Custom guess callable
def my_guess(x, y):
    return {"A": y.max(), "mu": x[y.argmax()], "sigma": 0.1}
d.profile("y:x", fit={"fun": "gauss", "guess": my_guess})

# (3) Default heuristic — no initial values needed for registry functions
d.profile("y:x", fit="gauss")
```

### 2.4 Vector convention

Multi-vector expressions broadcast fits per the vector-expression convention (§1.3):

```python
# Scalar fit — same function applied to all curves
d.profile("[y1, y2]:x", fit="gauss")

# Per-curve list — different fit per curve
d.profile("[y1, y2]:x", fit=["gauss", "linear"])
```

### 2.5 Composition with `group_by`, `facet_by`, `stacked`

Fits compose with all overlay and faceting features. One fit is produced per `group_by` group, per `facet_by` cell — the cardinality follows the visual cardinality:

```python
# One fit per sector group, per facet panel
d.profile("y:x", group_by="sec", facet_by="side", fit="gauss")

# stacked + group_by + fit: N fits are produced (one per group)
# stacking is purely visual — does NOT alter fit targets
d.hist("y", group_by="sec", stacked=True, fit="gauss")

# yerr= alone (no use_errors flag needed) opts scatter into error-weighted fit
d.scatter("y:x", yerr="y_err", fit="linear")
```

### 2.6 Summary fit figures — `summary_fit=`

For comparing fit results across groups or facet cells, `summary_fit=` produces standalone summary figures (tabular fit results, parameter trend plots). The `draw()` return contract is **unchanged** — summary figures are attached to `stats['summary_fit']`, never replacing the main figure.

Three placement modes:

```python
# (figure) — separate Figure object, useful for non-faceted calls
fig, ax, stats = d.profile("y:x", group_by="sec", fit="gauss",
                           summary_fit="figure")
# stats['summary_fit']['figure'] is the matplotlib.Figure object (Phase 13.51 Batch 1 item 6: key is 'figure' singular, value is a Figure or list depending on summary_fit mode)

# (subfigure) — per-panel inset on faceted calls
# Each facet panel gets an inset showing ONLY that panel's fits (per-panel slices,
# NOT a full-table copy in each panel)
d.profile("y:x", group_by="sec", facet_by="side",
          fit="gauss", summary_fit="subfigure")

# (pad) — GridSpec pad column pre-planned at figure construction time
# Pad axes are NOT in the returned ax array; reachable via stats['summary_fit']
# or fig.axes walk. GridSpec is immutable post-creation so this mode requires
# pre-planning at draw() time.
d.profile("y:x", group_by="sec", facet_by="side",
          fit="gauss", summary_fit="pad")
```

When figure capacity is exceeded, summary figures auto-overflow into additional figures rather than truncating.

### 2.7 Fit rendering controls

Per-call style overrides via `fit_textbox_kwargs=`:

```python
d.profile("y:x", fit="gauss",
          fit_textbox_kwargs={
              "fontsize": 8,
              "loc": "upper right",
              "rename_params": {"mu": "μ", "sigma": "σ"},
              "value_format": ".4g",        # parameter value precision
              "error_format": ".1g",        # parameter error precision (physics default)
              "precision_mode": "physics",  # round error to 1 sig fig; match value to error's decimal
          })
```

**Physics precision convention** (default since Phase 13.50): error rendered to 1 significant figure; parameter value rendered to the same decimal place as the error. This matches the convention used in calibration QA reports and HEP publications.

**Display-name map (`_DISPLAY_NAMES`)**: render-only convention mapping internal parameter names to conventional symbols. For example, `linear` fits expose internal names `slope` / `intercept` programmatically (in `stats['fit']`) but render as `p1` / `p0` (ascending-powers convention) in the on-plot textbox. Polynomial fits use `p0, p1, p2, …` exclusively.

**Faceted-plot fit textbox sizing** — use the dedicated style key `fit.text_fontsize_facet` (separate from `fit.text_fontsize_default`) to set fit textbox font size on faceted plots where panels are smaller than full-figure plots:

```python
from dfdraw import set_style
set_style({
    'fit.text_fontsize_default': 9, # full-figure fit textboxes
    'fit.text_fontsize_facet':   6, # smaller for per-panel fits in faceted calls
})
```

Used in production at `examples/time_series.py:808`. The two keys are independent — setting one does not affect the other. (Source: `style.py:133-134` — these are the only two `fit.text_fontsize*` keys at HEAD `07606c02`; the legacy un-suffixed `fit.text_fontsize` does not exist and using it raises `KeyError` at `set_style()` time.)

### 2.8 Stats output — `stats['fit']` canonical shape

For programmatic access (GBregression consumption, custom downstream analysis), fit numeric results are exposed in the standard `stats` dict:

```python
fig, ax, stats = d.profile("y:x", fit="gauss")
# stats['fit'] is a list-of-lists:
#   outer index = curve (in render order)
#   inner index = fits for that curve

# With group_by — dict-keyed by group value:
fig, ax, stats = d.profile("y:x", group_by="sec", fit="gauss")
# stats['fit'] is keyed by sector value

# With facet_by — both top-level flat list AND per-cell dict (additive):
fig, axes, stats = d.profile("y:x", facet_by="side", fit="gauss")
stats['fit']             # flat list across all facet cells
stats[(0, 0)]['fit']     # per-cell access — preserved contract
```

Each fit entry contains: parameter values, parameter errors (1σ from covariance), chi², degrees of freedom, p-value, residuals, **`x_range`** (the (min, max) tuple of the fit domain — important for GB regression workflows that need to know the data subset the fit applied to), and the canonical parameter names.

**For pandas-native consumption** — opt-in style key (replaces the rejected public `fit_results_to_df()` API):

```python
from dfdraw import set_style
set_style({"summary_fit.data_format": "pandas"})
# stats['summary_fit']['data'] is now a pd.DataFrame
```

### 2.9 Guards and edge cases

- **`hist_errors=True` + `cumulative=True/-1`** raises `NotImplementedError` — binned CDF with Poisson errors is statistically ill-defined.
- **scatter `yerr="col"`** alone is sufficient to enable error-weighted fitting; no `use_errors=True` flag required.
- **Fits with too few points** (n < parameter count + 1) fall through scipy's standard error path; no silent default values.
- **Fit fails to converge** — failure recorded in `stats['fit']` (`success=False` + error message); plot rendered without fit overlay; no exception raised at draw time.

### 2.10 Backward-incompatible changes

Two style keys were removed in the Phase 13.50 fit rendering overhaul:

| Removed | Replaced by |
|---|---|
| `fit.text_format` | `fit.value_format` + `fit.error_format` (separate parameter value and error precision) |
| `summary_fit.precision` | `fit.value_format` + `fit.error_format` + `precision_mode='physics'` |

See §8 Migration Notes for concrete migration code patterns.
---
## 3. Statistics and Data Handling

### 3.1 Statistics Dictionary

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

### 3.2 In-Plot Statistics Display

The `stats_fields=` and `stats=` kwargs control which per-bin or per-curve statistics render directly on the plot (in the legend or as inline annotations). Available on `profile()` and `hist()`. (Phase 13.51 Batch 1 item 5: legend_stats_fields= is NOT a valid kwarg; use stats=.)

```python
# Show mean and std in the legend for each group
d.profile("y:x", group_by="sec",
          stats=["mean", "std"])  # Phase 13.51 Batch 1 item 5

# Per-bin median + MAD on the profile
d.profile("y:x", stat_fields=["median", "mad"])
```

Standard fields: `mean`, `std`, `median`, `mad`, `n`, `sem`, plus min/max and percentiles depending on plot type. The complete per-plot field list is enumerated in §6.1 (Core API) and the `CAPABILITY_MATRIX.md` `STATS.*` feature group.

For programmatic access to the same numeric values (downstream consumption), use the returned `stats` dict — see §3.1 above. For fit-parameter overlays, see §2.7.

---

### 3.3 Robust Data Handling

NaN and infinity handling, hybrid autorange, sanitize diagnostics, and weighted statistics. See §1.8 for the full robust data handling description (the `nan_policy=`, hybrid autorange, sanitize counter, autorange diagnostics suite shipped in the channel-framework era).
---
## 4. Style and Labels

### 4.1 Style System

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

### 4.2 Color Options

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

### 4.3 Labels and Titles

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

### 4.4 Auto-Title

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

### 4.5 Legends

```python
# Automatic for group_by
drawer.hist('x', group_by='charge')  # Legend added

# Manual control
fig, ax, stats = drawer.hist('x', group_by='charge')
ax.legend(loc='upper right', fontsize=12)
```

### 4.6 Grid

```python
fig, ax, stats = drawer.hist('x')
ax.grid(True, alpha=0.3)
```

---
---
## 5. Integration

### 5.1 With AliasDataFrame

> **Authoritative content pending from ADF team.** The description below is the dfdraw-side view of the contract (best-effort by the dfdraw author); the ADF team's Technical Summary is canonical for `apply_meta` lifecycle, `register_subframe` semantics, `draw_lazy=True` materialization, and `axis_titles` propagation. Claude36 (ADF reviewer) has committed to drafting the authoritative ADF-side content in a follow-on round.

AliasDataFrame (ADF) is dfdraw's primary integration partner. ADF resolves aliases, joins subframes, lazily loads branches; dfdraw handles visualization and statistics. The integration is loose-coupled — ADF passes a resolved pandas DataFrame to `DFDraw(df)` and forwards user kwargs through `adf.draw(...)`.

**Responsibility split:**

| Concern | ADF | dfdraw |
|---|---|---|
| Alias / branch resolution | ✅ | — |
| Table joins / subframes | ✅ | — |
| Lazy branch loading | ✅ | — |
| Selection expression evaluation | ✅ (passed through) | ✅ (re-applied inside dfdraw) |
| Quantile binning of `group_by` / `facet_by` | — | ✅ (`AD-48` — single source of truth) |
| Per-bin statistics | — | ✅ |
| Channel framework / Algorithm A | — | ✅ |
| Fit computation | — | ✅ |
| Figure rendering | — | ✅ |

**Standard call:**

```python
fig, ax, stats = adf.draw("dy:row", facet_by="side",
                          group_by="sec", group_by_bins=5,
                          min_entries=25)
# ADF resolves aliases, builds df_subset, instantiates DFDraw(df_subset),
# forwards kwargs verbatim. dfdraw does the rest.
```

**Kwarg forwarding contract (Pattern A — FORWARDED_NAMES discipline):**

dfdraw exports tuples `_PROFILE_FORWARDED_NAMES`, `_HIST_FORWARDED_NAMES`, etc. — these enumerate every kwarg the corresponding draw method accepts. ADF passes through anything in these tuples without inspection. Class-load validation via `_validate_forwarded_names()` prevents drift.

**`ADF.axis_titles` duck-typed integration:**

If the underlying DataFrame is an AliasDataFrame instance with `.axis_titles` attribute, dfdraw consumes that mapping to set axis labels automatically. No explicit kwarg needed — duck-typed detection. See `CAPABILITY_MATRIX.md` feature `ADF.axis_titles`.

**Vector dispatch (Phase 13.16) — ADF-side requirement:**

When the expression is multi-vector (`"[y1, y2]:x"`), ADF must ensure all vector elements resolve through the same row-level filter. The Phase 13.16 contract: dfdraw computes one stats-dict per vector element with consistent indexing.

**Two OPEN items affecting ADF integration:**

> **⚠️ Known limitation — AD-37 (`same=True` color cycle reset).** AliasDataFrame currently creates a fresh `DFDraw(df_subset)` on every `.draw()` call. This resets `self._last_ax` and the color-cycle index between calls, breaking `same=True` continuity. **Workaround:** Use a single `DFDraw(df)` instance for the full overlay sequence rather than ADF's per-call construction. ADF-side fix (AD-50 Option C1b — cache `_last_ax` only, not full DFDraw) is pending; tracked in `tests/test_quantiles_profile.py:399` with skip marker.

> **⚠️ Known limitation — AD-50 (DFDraw instance caching).** ADF must NOT cache the full DFDraw instance across calls because `df_subset` is per-call-filtered (selection, entry_mask, subframe merges). Caching DFDraw would lose per-call filtering. The architect-ratified pattern is Option C1b: ADF caches only `_last_ax` and restores it onto a fresh DFDraw per call. This is an ADF-side implementation note; dfdraw exposes `_last_ax` for this purpose.

**OPEN AD verification status (`CAPABILITY_MATRIX.md`):** Both AD-37 and AD-50 carry `status: OPEN` in `ARCHITECT_DECISIONS.md` until the ADF-side Option C1b implementation ships.

**Cross-references:**
- AD verification: see `CAPABILITY_MATRIX.md` (`ADF.*`)
- Decision provenance: `ARCHITECT_DECISIONS.md` AD-37, AD-50, AD-48, AD-78
- ADF tutorials: `AliasDataFrame/tutorials/drawing/`, `AliasDataFrame/tutorials/cheatsheets/draw_cheatsheet.md`

### 5.2 With PyArrow

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

### 5.3 With RDataFrameDSL (Future)

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

### 5.4 With GBregression

> **§5.4 status (Block E — v6 update).** This section is the result of a 6-reviewer cross-team review (GB team + TimeAI team + O2DistAI team), 2026-06-05. 10 mechanical fixes applied (C-1..C-5 column naming / `adf.eval` / parallel variants / `draw_figures` attribution / `polDyp` note; T-1 boundary warning; T-2 stats dict; T-5 7× TPC; T-6 TRD+TOF; T-7 dy/dx convergence split). Two items remain pending architect ruling on scope expansion: T-3/T-4 (TimeAI patterns), T-8 (O2DistAI production patterns). One inline marker `[O2DistAI-CONFIRM]` flags the convergence-narrative numbers awaiting final O2DistAI sign-off. Authoritative API reference is the GBregression Technical Summary v4.0 at `../../groupby_regression/docs/`.

GBregression is the analysis layer of the ADF + GB + dfdraw stack. It provides multi-dimensional grouped regression, sliding-window fits, and polynomial distortion modelling at production scale (25M+ fits, 100K+ groups). The integration is data-only: GBregression outputs a coefficient DataFrame; ADF registers it as a subframe; dfdraw draws against it via the standard vector + normalize grammar.

> **Authoritative API reference:** GBregression Technical Summary v4.0 at `../../groupby_regression/docs/`. This section documents the dfdraw-side integration patterns; for full regression API (all kwargs, backends, boundary modes, roofline K metric), see the GB team's Technical Summary.

#### 5.4.1 Function Surface

**Which function to use:**

```
Need per-group OLS/WLS (most common)?        → make_parallel_fit_v4()      ← recommended
Need multiple targets simultaneously?         → make_parallel_fit_v5()
Need robust/Huber fitter?                     → GroupByRegressor.make_parallel_fit()
Need sliding-window smoothed maps?           → make_sliding_window_fit()
Need pure aggregation (mean/std/count)?      → make_sliding_window_aggregate()  ← 300× faster
Need polynomial basis for global fit?        → PolynomialSpec
Need non-linear fit (named models)?          → make_nonlinear_sliding_window_fit()
Need to evaluate coefficients at positions?  → GroupByRegressionEvaluator (7 methods)
Need C++ evaluation in O2 reconstruction?    → gbe_kernel (C++ evaluator)
```

| Function | Purpose | Production calls |
|---|---|---|
| `make_parallel_fit_v4` | Per-group OLS/WLS, numba JIT + PyArrow. Returns `(df_predictions, dfCoeffs)` | 40 in `makeSmoothMapsWithTPC.py` |
| `make_parallel_fit_v5` | Batched multi-target OLS | — |
| `GroupByRegressor.make_parallel_fit` | Huber / RLM / robust estimators | — |
| `make_sliding_window_fit` | N-D sliding window regression, boundary handling (`'symmetric'`, `'periodic'`) | 24 |
| `make_sliding_window_aggregate` | Windowed mean/std/count/median — 300× faster than fit with no predictors | 10 |
| `make_sliding_window_fit_parallel` | Parallel SW regression — splits by `split_columns`, `n_workers=` | — |
| `make_sliding_window_aggregate_parallel` | Parallel SW aggregation — same API as serial + `split_columns` | — |
| `make_nonlinear_sliding_window_fit` | Named models (gaussian, double_gaussian, ...) via model registry | — |
| `GroupByRegressionEvaluator` | 7 evaluation methods: `'lookup'` (fastest, integer grids), `'linear'` (~3 M/s, recommended), `'cubic'`, `'nearest'`, per-dimension `dict` dispatch | — |
| `PolynomialSpec` | Multi-variable polynomial basis (e.g., 96-term 4D Iter0 polynomial) | 14 |
| `gbe_kernel` (C++) | Evaluate fitted models in O2/O2Physics reconstruction (Python↔C++ invariance tested) | — |

#### 5.4.2 Core Integration Pattern — EXTRACT → HOLD → APPLY → VALIDATE

Every GB + ADF + dfdraw workflow follows the same four-step cycle. The residual after fit iteration N becomes the input to iteration N+1; the full history is preserved as ADF subframes (`CoeffsDyI1`, ..., `CoeffsDyI6`).

```python
# -- EXTRACT: GB regression produces per-group coefficients ------------------
from dfextensions.groupby_regression import make_parallel_fit_v4
from dfextensions.AliasDataFrame import AliasDataFrame

_, dfCoeffs = make_parallel_fit_v4(
    df=adf.df,
    gb_columns=["sector_bin180", "tgl_bin10"],
    fit_columns=["dcar_tpc_vertex", "dcar_itstpc"],    # joint multi-target
    linear_columns=["qpt"],
    selection=adf.df.eval("abs(dca_y) < 3"),
    fit_intercept=True,
    min_stat=20,
    addPrediction=False,
    backend='pyarrow',
    suffix="_CalibBias",
)
# dfCoeffs columns: sector_bin180, tgl_bin10,
#   dcar_tpc_vertex_intercept_CalibBias, dcar_tpc_vertex_slope_qpt_CalibBias,
#   dcar_tpc_vertex_rms, dcar_tpc_vertex_mad, ...

# -- HOLD: register as ADF subframe for lazy alias access -------------------
adf.register_subframe("CalibBias", AliasDataFrame(dfCoeffs),
                      index_columns=["sector_bin180", "tgl_bin10"])

# -- APPLY: define residuals as lazy aliases (zero copies) ------------------
for var in ["dcar_tpc_vertex", "dcar_itstpc"]:
    adf.add_alias(f"{var}_predicted",
        f"CalibBias.{var}_intercept_CalibBias + CalibBias.{var}_slope_qpt_CalibBias * qpt",
        dtype=np.float16, fill_value=0)
    adf.add_alias(f"{var}_residual", f"{var} - {var}_predicted",
        dtype=np.float16, fill_value=0)

# -- VALIDATE: visualize with dfdraw multi-channel composition --------------
fig, ax, stats = adf.draw(
    "dcar_tpc_vertex_residual:tgl_bin10",
    type="profile",
    group_by="sector_bin180", group_by_quantiles=5,   # → color channel
    facet_by="side",                                    # → subplot
    quantiles=[0.1, 0.5, 0.9],                         # → shaded bands
    fit="gauss",                                        # → analytic overlay
    auto_title=True, min_entries=50,
)
```

**Key point:** dfdraw never calls GBregression directly. It sees only ADF aliases. `CalibBias.dcar_tpc_vertex_intercept_CalibBias` is resolved lazily at `adf.draw()` time through ADF's subframe join — dfdraw receives a resolved pandas Series like any other column.

This cycle runs: **7× TPC distortion (Iter0 polynomial + I1–I6; was 6× before v1.3) · 6× ITS alignment · 5 steps TRD · TRD+TOF (in development, not yet production)**.

**Convergence (single TF, gr11)** *[O2DistAI-CONFIRM: production narrative for these numbers]*:
- `dy` std 498 → 132 μm (×3.8, converged at I5/I6)
- `dx` std 1470 → 610 μm (×2.4, slower convergence — additional per-sector iterations needed for full dx closure)

#### 5.4.3 Polynomial Distortion Map — Iter0 Pattern

```python
from dfextensions.AliasDataFrame.PolynomialSpec import PolynomialSpec

# 4D polynomial: xM × driftM × dsectorM × tgSlp — 96 terms
spec = PolynomialSpec(
    columns=["xM", "driftM", "dsectorM", "tgSlp"],
    degrees=(3, 3, 2, 1),
)
# Split dy (tgSlp^0 terms) from dx (tgSlp^1 terms)
spec_dy = PolynomialSpec(columns=["xM", "driftM", "dsectorM", "tgSlp"],
                         degrees=(3, 3, 2, 1),
                         term_filter=lambda e: e[3] == 0)

_, dfCoeffsPoly = make_parallel_fit_v4(
    df=adf.df, gb_columns=["sec"],
    fit_columns=["dy"],
    linear_columns=spec_dy.basis_expressions(),
    selection=isOK, suffix="_PIter0", fit_intercept=False,
)

# Register + alias + draw correction quality
adf.register_subframe("PolyFit", AliasDataFrame(dfCoeffsPoly), index_columns=["sec"])
# polDyp is a runtime-generated function built by PolynomialSpec.numba_evaluator()
# during apply_schema() at read_tree time. Requires numba. See "[apply_schema]
# Reconstructed N polynomial functions" log message to confirm availability.
adf.add_alias("dy_corr0", "polDyp(xM, driftM, dsectorM, tgSlp)")
adf.add_alias("dyC1", "dy - dy_corr0", dtype=np.float16)

adf.draw("[dy, dyC1]:row", normalize="delta",
         group_by="sec", facet_by="side", auto_title=True)
```

#### 5.4.4 Sliding Window — Aggregation Then Fit

Two-stage production pattern: (1) aggregate raw tracks into per-bin statistics, then (2) fit polynomial corrections on the aggregated frame.

```python
from dfextensions.groupby_regression.groupby_regression_sliding_window import (
    make_sliding_window_fit, make_sliding_window_aggregate,
)

# Stage 1: aggregated statistics map (fast)
dfGB = make_sliding_window_aggregate(
    df=adf.df,
    gb_columns=["sec", "row_bin5", "dsectorM_bin8", "driftM_bin8"],
    agg_columns=["dy", "dz", "dtgl"],
    window_spec={"sec": 1, "row_bin5": 2, "dsectorM_bin8": 2, "driftM_bin8": 2},
    boundary="symmetric",       # mirror at sector boundaries
    n_sigma_cut=3.0,
)

# Stage 2: sliding-window fit on aggregated bins
dfCoeffsSW = make_sliding_window_fit(
    df=adf.df,
    gb_columns=["sec", "row_bin5", "dsectorM_bin8", "driftM_bin8"],
    fit_columns=["dy_I0T"],
    linear_columns=["tgSlp"],
    window_spec={"sec": 1, "row_bin5": 2, "dsectorM_bin8": 2, "driftM_bin8": 2},
    boundary="symmetric",
    algorithm="recompute",
    # NOTE: boundary='symmetric' with algorithm='recompute' is a known limitation
    # (bug instance #9): V1/V2 recompute path silently ignores boundary parameter.
    # Tracked as Phase 13.XX.GB-BoundaryV1V2. For correct boundary handling,
    # use algorithm='incremental'. Production uses 'recompute' accepting this limitation.
    backend="numba",
)

adf.register_subframe("CoeffsDyI1", AliasDataFrame(dfCoeffsSW),
                      index_columns=["sec", "row_bin5", "dsectorM_bin8", "driftM_bin8"])
adf.draw("CoeffsDyI1.dy_I0T_intercept:row_bin5",
         type="profile", group_by="sec", facet_by="side", auto_title=True)
```

**Boundary modes:** `'full'` (default), `'symmetric'` (mirror — TPC sectors), `'periodic'` (wrap-around — φ-periodic dimensions). Without `boundary='symmetric'`, edge bins pull from fewer neighbors, introducing systematic bias at TPC sector boundaries — this is why 24 out of 24 SW fit calls in production use it.

#### 5.4.5 GroupByRegressionEvaluator — Coefficient Application

```python
from dfextensions.groupby_regression.groupby_regression_evaluator import (
    GroupByRegressionEvaluator,
)

ev = GroupByRegressionEvaluator.from_dfGB(
    dfCoeffs, group_columns=["sector", "padRow"],
    predictor_columns=["tgSlp"], targets="dy", suffix="",
)

# 7 methods — per-dimension dispatch for mixed grids
prediction = ev.evaluate(
    positions={"sector": df.sector, "padRow": df.padRow},
    predictors={"tgSlp": df.tgSlp},
    method={"sector": "lookup", "padRow": "linear"},   # mixed int/continuous
)

# Register prediction as alias, draw differential overlay
adf.add_alias("dy_corrected", "dy - dy_predicted", dtype=np.float32)
adf.draw("[dy, dy_corrected]:row", normalize="delta",
         group_by="sec", group_by_quantiles=4, auto_title=True)
```

| Method | Speed | Use case |
|---|---|---|
| `'lookup'` | Fastest | Integer grids — direct indexing |
| `'linear'` | ~3 M/s | **Recommended** — scipy `map_coordinates` |
| `'cubic'` | ~0.7 M/s | Smooth C1 output (derivatives) |
| `'nearest'` / `'nearest_fast'` | Fast | Discrete lookup |
| `'multilinear'` | ~1 M/s | Legacy; supports `use_errors=True` |
| `dict` | Variable | Per-dimension dispatch for mixed grids |

*Speed figures are directional estimates on small synthetic workloads. Production-scale benchmarks pending (GBAI TECHNICAL_SUMMARY v4.0 §Unverified Claims).*

#### 5.4.6 What dfdraw Returns to GB Workflows — `stats` Dict

| `stats` key | Type | GB / ADF use |
|---|---|---|
| `means`, `stds`, `sems` | array | Per-bin residual diagnostics; weight column for WLS pass |
| `medians`, `mads` | array | Robust dispersion for outlier detection |
| `count` | array | Per-bin count (singular, not `counts`) for `min_stat` validation |
| `bins`, `bin_centers` | array | Bin geometry for evaluator coordinate matching |
| `stats['fit']` | list[dict] | Per-group fit parameters (mean, sigma, χ², ndf) — quality gates |
| `stats['summary_fit']` | tuple `(figs_dict, data, note)` | When `summary_fit=` set. `figs_dict` keys: `'table'`, `'figure'`. For vector dispatch: `stats[0]['summary_fit']` |
| `stats['profile_data']` | DataFrame | When `return_data=True`: columns `x_center`, `y_mean`, `y_std`, `count` — direct input to next GB regression pass |
| `n`, `n_input`, `n_filtered` | int | Sanity check that selection + NaN policy applied correctly |

#### 5.4.7 Batch QA Pattern

The O2DistAI production pipeline uses `adf.draw_figures(specs, save_dir=...)` for all QA figures in one call. **Note:** `adf.draw_figures()` is an AliasDataFrame method that delegates to dfdraw internally; the dfdraw-native equivalent is `DFDraw(df).draw_batch(specs)`.

```python
specs = [
    {"expr": "dy:row", "type": "profile", "fit": "gauss",
     "group_by": "sec", "facet_by": "side"},
    {"expr": "dz:row", "type": "profile", "fit": "gauss",
     "group_by": "sec", "facet_by": "side"},
]
adf.draw_figures(specs, save_dir="output/calibration_QA/")
```

For interactive time-series analysis (TimeAI workflows), the pattern is individual `adf.draw()` + `fig.savefig()` calls rather than batch mode.

#### 5.4.8 Performance

| Metric | Value |
|---|---|
| Pipeline wall time (82M rows) | 722 s (was 1452 s, 2× improvement) |
| Roofline K (SW fit, 100K rows) | K = 2.9 (42 ms / 15 ms ideal) |
| Tests | 575 + 10 roofline |
| `make_sliding_window_aggregate` vs fit | ~300× faster for pure aggregation |

Roadmap: Phase 13.24 (V5 aggregate, K → ~2.5), Phase 13.25 (BLAS-batched OLS, K → ~1.5).

**For verification status of GBregression-relevant dfdraw features (`FIT.*`, `SUMMARY_FIT.*`, `HIST.cumulative`, `STATS.*`), see `CAPABILITY_MATRIX.md`.**

---

---

### 5.5 Batch Processing — `draw_batch()`

#### 5.5.1 Dict Format (Original)

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

#### 5.5.2 List Format — Group-Based with Defaults Hierarchy

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

#### 5.5.3 Batch Saving

```python
results = drawer.draw_batch(specs, save_dir='plots/')
# Automatically saves all plots to plots/ directory
```

#### 5.5.4 Error Handling

```python
results = drawer.draw_batch(specs, verbose=True)

# Check summary
print(results['_summary'])
# {'success': 2, 'failed': 1, 'errors': {...}}

# Access failed plots
if results['_summary']['failed'] > 0:
    print(results['_summary']['errors'])
```

#### 5.5.5 Verbose Levels

| Value | Behavior |
|-------|----------|
| `False` / `0` | Silent |
| `True` / `1` | Progress — group names, save paths, summary |
| `2` | Debug — also prints merged parameters per plot |

---
---
## 6. API Reference

API surface for dfdraw — methods, common parameters, return contracts, expressions, and call-time controls. **Use §1–§5 to discover *what* exists; use this section to look up *how to invoke*.**

### 6.1 Core API

### 6.1.1 Core Methods

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

# Phase 13.39 — new plot types
drawer.profile2d(expr, **kwargs) -> (fig, ax, stats)    # z:y:x — 2D heatmap profile
drawer.scatter3d(expr, **kwargs) -> (fig, ax, stats)    # 3D scatter (mpl_toolkits.mplot3d)
# Time axis: any plot method accepts time_format= with datetime64 or epoch-seconds input

# Properties
drawer.backend -> str  # 'pandas' or 'pyarrow'
drawer.memory_info() -> dict
```

### 6.1.2 Style Functions

```python
from dfdraw import set_style, get_style, list_styles

set_style(style_dict)    # Set global style
get_style() -> dict      # Get current style
list_styles() -> list    # List available styles
```

### 6.1.3 Statistics Functions

In-plot statistics are surfaced via the `stat_fields=` and `stats=` parameters on `profile()` and `hist()` calls themselves — there is no separate post-hoc helper. See §3.2 (In-Plot Statistics Display) for usage.

For programmatic access to the same numeric values, consume the `stats` dict returned by every draw call — see §3.1.

### 6.1.4 Return Contracts — Standard vs N-D Faceted

⚠️ **Non-standard return type for `FIGID` calls.** Calls that include `'FIGID'` in `facet_by` return a *triple of lists* instead of the standard 3-tuple. Any caller iterating the return value as `(fig, ax, stats)` will break.

**Standard return — single-figure calls (default behaviour):**

```python
fig, ax, stats = d.profile("y:x")                                 # 3-tuple
fig, axes, stats = d.profile("y:x", facet_by="sec")               # 3-tuple, axes is 1D array
fig, axes, stats = d.profile("y:x", facet_by=['ROW', 'COL'])      # 3-tuple, axes is 2D array
```

**N-D faceted return — `FIGID` calls return triple-of-lists:**

```python
figures, axes_per_fig, stats_per_fig = d.profile(
    "y:x", facet_by=['ROW', 'COL', 'FIGID'])
# figures        : List[matplotlib.Figure]    — one per FIGID value
# axes_per_fig   : List[axes_2d_array]         — one 2D axes array per figure
# stats_per_fig  : List[stats_dict]            — one stats dict per figure
```

This deviation from the standard 3-tuple is **explicit and prominent** per AD-2/13.41.DF — the alternative (returning a single nested structure) would have hidden the figure-multiplicity from caller code that didn't expect it.

**Composition with shared axes:**

```python
# Per-axis sharing
d.profile("y:x", facet_by=['ROW', 'COL'],
          share_x='row',    # default — share x within each row
          share_y='col',    # default — share y within each column
          share_across_figures=True)   # default — across FIGID figures too
```

Dispatch dict: `{'all': True, 'row': 'row', 'col': 'col', 'none': False}` — symmetric for both axes. *Decision rationale: AD-3/13.41.DF.*

### 6.1.5 Fit System Surface (Phases 13.42, 13.42-FIX1, 13.43, 13.50)

```python
# Inline fit on profile / hist / scatter
d.profile("y:x", fit="gauss")                        # str — predefined registry
d.profile("y:x", fit={"fun":  "gauss",               # dict — full spec (Phase 13.51 Batch 1 item 4: key is "fun" not "name")
                      "range": (0, 5),
                      "initial": {"mu": 0, "sigma": 1}})
d.profile("y:x", fit=my_callable)                    # callable — custom fit fn

# Vector broadcasting (Phase 13.16 convention — AD-2/13.42.DF)
d.profile("[y1, y2]:x", fit=["gauss", "linear"])    # per-curve list
d.profile("[y1, y2]:x", fit="gauss")                 # scalar — broadcast to all curves

# Per-call fit textbox configuration
d.profile("y:x", fit="gauss",
          fit_textbox_kwargs={"fontsize": 8,         # per-call fontsize works (AD-3/13.42.DF-FIX1)
                              "loc": "upper right"})

# Scatter fit with error weighting (simplified per AD-1/13.42.DF-FIX1)
d.scatter("y:x", yerr="y_err", fit="linear")        # yerr= alone is opt-in; use_errors=True NOT required

# Summary fit figures (Phase 13.43)
d.profile("y:x", group_by="sec", fit="gauss",
          summary_fit="figure")                      # separate fig in stats['summary_fit']
d.profile("y:x", group_by="sec", fit="gauss",
          summary_fit="subfigure")                   # per-panel inset (per-panel slices, not full-table)
d.profile("y:x", group_by="sec", fit="gauss",
          summary_fit="pad")                         # GridSpec pad column pre-planned at construction

# Predefined registry — what scipy.optimize.curve_fit understands (AD-5/13.42.DF)
#   gauss / gaussian, linear, pol0..pol5, expo / exponential,
#   lorentz / lorentzian, powerlaw
#   Deferred: landau, crystalball, breitwigner ("can be done later" per architect)
```

**4-tier initial-parameter resolution (AD-6/13.42.DF):**
1. User `initial` dict (highest priority)
2. User `guess` callable
3. Registry heuristic (default per-function)
4. scipy default + UserWarning (fallback)

### 6.1.6 Legend — Polymorphic + Permanent Back-Compat

```python
# Polymorphic legend= kwarg (Phase 13.50)
d.draw("y:x", group_by="sec", legend=True)            # default — show
d.draw("y:x", group_by="sec", legend=False)           # suppress
d.draw("y:x", group_by="sec", legend='shared')        # shared figure-level
d.draw("y:x", group_by="sec", legend='first')         # only first subplot in faceted
d.draw("y:x", group_by="sec", legend={"loc": "upper right",  # full kwargs dict
                                       "fontsize": 8})

# show_legend= permanent back-compat parallel (NOT deprecated, NOT removed)
d.draw("y:x", group_by="sec", show_legend=True)       # still works
# If both passed: legend= wins. No warning emitted.
```

---



---

### 6.2 Common Parameters

### 6.2.1 All Plot Types

```python
expr: str               # Expression to plot
ax: Axes               # Matplotlib axes (optional)
selection: str         # Data filter
entry_mask: ndarray    # Boolean mask
group_by: str          # Column for grouping/overlay
same: bool             # Overlay on last axes (Phase 13.13.DF)
auto_title: bool|str   # Automatic title (Phase 13.12.DF v1.2)
```

> **Note on row-range parameters.** `entry_begin` / `entry_end` are *not* DFDraw kwargs — they are ADF-layer parameters consumed by `adf.draw(...)` and resolved before dfdraw runs. See §5.1 for the ADF integration contract. Standalone DFDraw users should subset the DataFrame before passing it in: `DFDraw(df.iloc[begin:end]).draw(...)`.

### 6.2.2 Histogram-Specific

```python
bins: int or array     # Number of bins or edges
range: tuple           # (min, max) range
color: str             # Bar color
alpha: float           # Transparency [0,1]
edgecolor: str         # Bin edge color
```

### 6.2.3 Scatter-Specific

```python
color: str or array    # Point color or third variable
s: float or array      # Point size
marker: str            # Marker style ('o', 's', '^', etc.)
alpha: float           # Transparency
cmap: str              # Colormap (when color is array)
```

### 6.2.4 Profile-Specific

```python
bins: int              # Number of x bins
marker: str            # Point marker
markersize: float      # Marker size (forwarded to matplotlib)
linestyle: str         # Line style
linewidth: float       # Line width (forwarded to matplotlib)
capsize: float         # Error bar cap size
return_data: bool      # Export profile DataFrame (Phase 13.12.DF)
min_entries: int       # Min entries per bin (Phase 13.12.DF)
group_by_bins: int     # Equal-width bins for group_by (Phase 13.12.DF)
group_by_quantiles: int # Quantile bins for group_by (Phase 13.12.DF)
sort_groups: bool      # Sort legend order (Phase 13.12.DF)
weights: str           # Weight column (Phase 13.12.DF v1.1)
```

### 6.2.5 Quantile Rendering

```python
quantiles: List[float]            # Fractions in (0, 1) — symmetric pair, triple, or 4+
central: str                      # 'mean' (default), 'median', 'both', 'none'
quantile_mode: str                # 'auto' (default), 'discrete', 'band',
                                  #     'error_bars', 'nested_band'
```

### 6.2.6 Channel-Aware Encoding

```python
quantile_style: str               # Channel for quantile dimension
                                  #     ('color', 'linestyle', 'marker')
                                  # Used when combined with group_by
                                  # Algorithm A auto-resolves if not set
```

### 6.2.7 Facet Routing

```python
facet_by: str                     # Facet channel — valid values:
                                  #     'group_by', 'vector', 'quantiles',
                                  #     'selection_vector', 'weights_vector'
                                  # Also accepts: any DataFrame column name (§1.9),
                                  #     or list ['ROW','COL'] / ['ROW','COL','FIGID'] (§1.7.4)
```

### 6.2.8 Robust Data Handling

```python
nan_policy: str                   # 'filter' (default), 'warn', 'raise'
range: tuple | str                # Numeric (lo, hi) or strategy name:
                                  #     'hybrid' (default), 'minmax',
                                  #     'percentile_99', 'percentile_95',
                                  #     'robust_3mad', 'robust_4mad'
```

---



---

### 6.3 Expression Evaluation

### 6.3.1 Supported Expressions

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

### 6.3.2 Expression Scope — Where Each Operation Lives in the Stack

dfdraw's expression syntax (`"y:x"`, `"[y1, y2]:x"`, `selection=`, `weights=`) evaluates against a resolved pandas DataFrame. Operations on the data *before* it reaches dfdraw — column derivation, aggregation, table joins — are not dfdraw's concern; they live upstream in the data layer.

| Operation | Where it lives | dfdraw-side syntax |
|---|---|---|
| Column derivation (e.g., `pt = sqrt(px² + py²)`) | **ADF** via `apply_meta()` aliases | dfdraw sees the alias as a column |
| Multi-table joins / subframes (e.g., calibration coefficients) | **ADF** via `register_subframe()` | Subframe columns accessible via dotted form: `"CalibFit.coeff:x"` |
| Aggregations / GROUP BY | **dfdraw** via `type="profile"`, `group_by=`, `*_bins=` | Aggregation is declarative and explicit |
| Multi-dim regression / model fitting | **GBregression** via `make_parallel_fit_v4()` and sibling fit methods | dfdraw consumes the coefficient frame as an ADF subframe |
| Standalone use without ADF/GB | **dfdraw** on plain pandas DataFrame | Full feature surface; user is responsible for column derivation |

This is an *architecture boundary*, not a *missing feature*. Each operation lives at the layer where it is most efficient and most testable. For standalone use without ADF, derive columns into the DataFrame before passing to `DFDraw(df)`.

---



---

### 6.4 Selection and Filtering

### 6.4.1 Selection Syntax

**String expression (pandas query syntax):**
```python
drawer.draw('x', selection='pt > 0.5')
drawer.draw('x', selection='(pt > 0.5) & (eta < 1.0)')
drawer.draw('x', selection='(charge == 1) | (charge == -1)')
```

**Important:** Use Python syntax (`&`, `|`), NOT C++ syntax (`&&`, `||`)

### 6.4.2 Entry Range (ADF-mediated)

`entry_begin` / `entry_end` are ADF-layer parameters, not DFDraw kwargs (see §6.2.1 note). Standalone DFDraw users subset the DataFrame before construction:

```python
# Standalone DFDraw — subset the DataFrame
DFDraw(df.iloc[:10000]).draw('x')    # first 10k entries
DFDraw(df.iloc[1000:]).draw('x')     # skip first 1000

# ADF integration — entry_begin / entry_end consumed by adf.draw(...)
adf.draw('x', entry_begin=0, entry_end=10000)
adf.draw('x', entry_begin=1000)
```

### 6.4.3 Boolean Mask

```python
import numpy as np

mask = (df['pt'] > 1.0) & (df['charge'] != 0)
drawer.draw('x', entry_mask=mask.values)
```

---



---

### 6.5 Size Control

### 6.5.1 Figure Size

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

### 6.5.2 Subplot Layout (Faceting)

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



---

### 6.6 Current Defaults

### 6.6.1 Figure Settings

```python
# From matplotlib defaults (can be overridden via set_style)
figure.figsize: (8, 6)      # inches
figure.dpi: 100             # screen display
savefig.dpi: 150            # saved figures
```

### 6.6.2 Histogram Defaults

```python
bins: 50                    # Number of bins
range: None                 # Auto from data
histtype: 'stepfilled'      # Bar style
alpha: 0.7                  # Transparency
edgecolor: 'black'          # Bin edge color
```

### 6.6.3 Scatter Defaults

```python
s: 20                       # Point size (marker size)
alpha: 0.6                  # Transparency
marker: 'o'                 # Circle markers
```

### 6.6.4 Profile Defaults

```python
bins: 50                    # X-axis bins
marker: 'o'                 # Point marker
linestyle: '-'              # Connecting line
capsize: 3                  # Error bar caps
```

### 6.6.5 Hexbin Defaults

```python
gridsize: 50                # Hexagon grid size
cmap: 'viridis'             # Colormap
mincnt: 1                   # Min count to show
```

---



---

### 6.7 Overlay and Multi-Plot Support

### 6.7.1 Overlays — Two Patterns

dfdraw supports two overlay mechanisms that serve different use cases. **They are not parallel alternatives** — each addresses a specific situation.

#### Pattern 1 — `group_by=` (primary production pattern)

A single `draw()` call produces a multi-curve overlay automatically; one curve per group value, colored by Algorithm A. This is the declarative production tool.

```python
# Production: one call → multi-color sector overlay × per-side facets × gaussian fits
adf.draw("dy:row", group_by="sec", facet_by="side", fit="gauss")
```

**Supported for:** `profile`, `hist`, `scatter` (overlaid with distinct colors per group).
**Composes with:** `facet_by`, `fit=`, `normalize=`, `selection_vector=`.
**Not used for** `hist2d` / `hexbin` — overlapping density surfaces cannot be visually disambiguated; use `facet_by` for density-plot separation, or Pattern 2 below for mixed-type overlays.

#### Pattern 2 — `same=True` (cross-plot-type overlay + interactive convenience)

Sequential `draw()` calls render onto the most recently created axes when `same=True` is passed. Two legitimate use cases:

1. **Mixed-type overlays** that `group_by=` cannot express — e.g., a profile or quantile bands on top of a 2D density heatmap. This is a valid production pattern when the desired overlay crosses plot types.
2. **Interactive shell exploration** at the Jupyter / IPython REPL — "plot this on top of what I just drew".

```python
# Production: profile residuals overlaid on a 2D density map
drawer.hist2d('y:x', norm='log')
drawer.profile('y:x', same=True)

# Production: quantile bands superimposed on hist2d
drawer.hist2d('y:x')
drawer.profile('y:x', quantiles=[0.16, 0.50, 0.84], same=True)

# Interactive: simple multi-curve overlay in REPL
drawer.profile('y1:x')
drawer.profile('y2:x', same=True)
drawer.profile('y3:x', same=True)
# → Three curves, different colors, auto-labeled, legend shown
```

**Automatic features with `same=True`:**
- Auto-increment colors from palette (AD-16)
- Auto-generate labels from expression (AD-17)
- Append to title when `auto_title=True` (AD-18)
- Legend shown automatically

**Precedence rules:**
- `ax=` always wins over `same=True`
- Explicit `color=`, `label=`, `title=` override auto-features
- `title=` replaces entire title (no append)

**Not composable with** `facet_by` — faceting creates new axes; `same=True` reuses existing.

**Instance tracking (AD-15):**
- Uses `self._last_ax` to track last axes
- Falls back to `plt.gca()` when no previous draw
- For safe AliasDataFrame integration, AD-37 requires caching the `DFDraw` instance

#### Pattern 3 (escape hatch) — Manual overlay via shared `ax=`

When neither Pattern 1 nor Pattern 2 fits — e.g., heterogeneous selections drawn into specific subplot cells — pass an `ax=` directly:

```python
fig, ax = plt.subplots()
drawer.draw('x', selection='charge==1',  ax=ax, label='Positive', alpha=0.5)
drawer.draw('x', selection='charge==-1', ax=ax, label='Negative', alpha=0.5)
ax.legend()
```

#### Summary

- **`group_by=`** — same-plot-type overlay, declarative, composes with everything (production primary)
- **`same=True`** — cross-plot-type overlay (production) + interactive REPL convenience
- **shared `ax=`** — escape hatch for arbitrary layouts

### 6.7.2 Faceting (Separate Subplots)

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

### 6.7.3 Reference Overlays

**Add analytic reference distributions onto an existing plot:**

```python
fig, ax, stats = drawer.hist('x', bins=50)

# Add Gaussian overlay (production pattern from examples/generate_gallery.py:515)
drawer.add_reference_overlay(
    ax,
    func='gaussian',
    mu=0,
    sigma=1,
    color='red',
    linestyle='--',
    label='Expected',
)
```

**Available `func=` values** (source: `drawer.py:6522`):
- `'gaussian'` — Normal distribution; takes `mu=`, `sigma=`
- callable — any user-supplied function `f(x)` evaluated on the current axis range

`add_reference_overlay` is a method of the `DFDraw` instance, not a separate module. The previous "Phase 12.4b5 — `from dfdraw.stats import add_reference_overlay`" pattern was retired; that module path does not exist at HEAD `07606c02`.

---



---

### 6.8 Profile Enhancements

### 6.8.1 Return Profile Data (F1)

```python
fig, ax, stats = drawer.profile('y:x', return_data=True)
profile_df = stats['profile_data']
# DataFrame with: x_center, x_low, x_high, y_mean, y_std, y_sem, count
```

### 6.8.2 Minimum Entries Filter (F2)

```python
# Suppress bins with < 3 entries (default)
drawer.profile('y:x', min_entries=3)

# More restrictive
drawer.profile('y:x', min_entries=20)
```

### 6.8.3 Auto-Bin Float Group-By (F3)

```python
# Equal-width bins for float grouping variable
drawer.profile('y:x', group_by='mP3', group_by_bins=8)

# Equal-count quantile bins
drawer.profile('y:x', group_by='mP3', group_by_quantiles=5)
```

### 6.8.4 Sorted Groups (F4)

```python
# Sorted legend order (default)
drawer.profile('y:x', group_by='sector', sort_groups=True)
```

### 6.8.5 Weighted Statistics

```python
# Weighted mean/std/sem
drawer.profile('y:x', weights='w')
```

---
---
## 7. Scope Boundaries

### 7.1 Where Each Capability Lives in the Stack

dfdraw is the visualization layer of a multi-component stack. Several capabilities a user might expect from dfdraw are deliberately delegated to upstream components — they are not "limitations" of dfdraw but capabilities of other layers. The table below names the canonical owner of each capability.

| Capability | Owner | Where to find it |
|---|---|---|
| **Schema validation / aliases** | AliasDataFrame | ADF Technical Summary; `examples/time_series.py` (`apply_meta(adf, df_TimeSeriesAliases)`) |
| **Table joins / subframe merges** | AliasDataFrame | ADF Technical Summary; `adf.register_subframe()` pattern in `time_series.py` |
| **Lazy evaluation** | AliasDataFrame | ADF Technical Summary; `adf.draw_lazy=True` in `time_series.py` |
| **Regression fits (multi-dimensional, parallel, sliding-window)** | GBregression | GB Technical Summary; `make_parallel_fit_v4(df=adf.df, gb_columns=..., addPrediction=...)` and `make_sliding_window_fit(...)` in `time_series.py` |
| **Inline curve fits on a plot (gauss, linear, polN, ...)** | **dfdraw — Phase 13.42** | §2 (Fitting — full reference) |
| **Interactive plot widgets (pan/zoom/hover/sliders)** | matplotlib + ipympl | `%matplotlib widget` in Jupyter is the zero-cost path; full interactive stack (linked brushing, server-side data) deferred to future RootInteractive integration |

### 7.2 Technical Limitations

**Memory:**
- Large scatter plots (>1M points) may be slow
- Recommendation: Use hexbin for >100k points

**Performance:**
- Expression evaluation in Python (not compiled)
- For performance-critical workflows, materialize columns first

**Stats computation:**
- Per-bin statistics (the `stats` dict) and fit results (`stats['fit']`) are structured for downstream programmatic consumption by GBregression and custom analysis pipelines — see §5.4.6.
- For statistical modeling **beyond dfdraw's scope** (custom likelihood fits, Bayesian inference, robust regression beyond GBregression's offering), use scipy / statsmodels on the raw data.

**Vector expressions (Phase 13.16.DF):**
- Nested brackets not supported: `"[arr[0],arr[1]]:x"` — use explicit column names
- Broadcasting is strict N:1, 1:N, N:N — mismatched shapes raise `ValueError`
- `hist2d`/`hexbin` do not accept vector input (fail-fast with clear message)
- Typical N ≤ 20; complexity is O(N × draw_cost), not optimized for N ≥ 100
- `stats()` with vector returns `list[dict]` instead of a single `dict`

### 7.3 Current Bugs / Issues

**No tests failing as of Phase 13.50.DF FIX2** (1057/1057 tests passing on `PHASE_13_50_DF_FIX2_END`, HEAD `07606c02`).

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

**Batch entry points — both are working public API:**
- **`adf.draw_figures(specs, save_dir=...)`** — the **ADF-side batch entry point**. The primary way to do batch plotting from AliasDataFrame. Used in production scripts including `makeSmoothMapsWithTPC.py` (10+ calls). Delegates internally through ADF's draw infrastructure with ADF mediation (alias resolution, subframe joins, lazy materialization).
- **`DFDraw(df).draw_batch(specs)`** — the **dfdraw-side batch entry point**. Direct dfdraw API for callers who already have a materialized DataFrame and do not need ADF mediation. See §5.5 for the full reference.

Both are supported and have been since their respective phases. Pick the entry point that matches whether the caller has an ADF instance or a raw DataFrame; **neither is a "workaround" for the other**.

(Historical note: an earlier integration gap was recorded around Phase 13.13.ADF concerning group-level `defaults` cascade through `adf.draw_figures()`. If you encounter that specific cascade behaviour, please file with ADF team; the current ADF→dfdraw delegation path resolves it.)

---
---
## 8. Migration Notes

Read this appendix **before upgrading** if your code base is at or before `PHASE_13_49_DF_FIX1_END`. There are two backward-incompatible changes and two semantic locks that may affect existing code.

### 8.1 ⚠️ BREACH — `fit.text_format` style key REMOVED (Phase 13.50 — AD-1/13.50.DF)

The unified `fit.text_format` style key has been **removed**. It is replaced by two separate keys: `fit.value_format` and `fit.error_format`.

**Before (Phase ≤ 13.49 FIX1):**
```python
set_style({'fit.text_format': '.4g'})
# Both fit values and errors rendered with .4g format
```

**After (Phase ≥ 13.50):**
```python
set_style({
    'fit.value_format': '.2g',    # default — for parameter values
    'fit.error_format': '.1g',    # default — for parameter errors (physics convention: 1 sig fig)
})
```

**Migration steps:**
1. Search code base for `'fit.text_format'`:
   ```bash
   grep -rn "fit\.text_format" --include="*.py" .
   ```
2. Replace each occurrence with separate `value_format` and `error_format` settings.
3. If you want the old unified behaviour back, set both to the same format string: `{'fit.value_format': '.4g', 'fit.error_format': '.4g'}`.

**Rationale:** Physics convention separates value precision from error precision. Errors are rendered to 1 significant figure by default; values match the error's decimal place.

*Decision: AD-1/13.50.DF + CRR §2.1 [BREACH] in `ARCHITECT_DECISIONS.md`.*

### 8.2 ⚠️ BREACH — `summary_fit.precision` style key REMOVED (Phase 13.50 — AD-2/13.50.DF)

The `summary_fit.precision` integer style key has been **removed**. It is replaced by the new `precision_mode='physics'` option together with the new `fit.value_format` / `fit.error_format` keys.

**Before (Phase ≤ 13.49 FIX1):**
```python
set_style({'summary_fit.precision': 4})    # 4 decimal places
```

**After (Phase ≥ 13.50):**
```python
set_style({
    'fit.value_format': '.4g',
    'fit.error_format': '.1g',
    'precision_mode': 'physics',     # error → 1 sig fig, value matched to error's decimal place
})
```

**Migration steps:**
1. Search code base for `'summary_fit.precision'`:
   ```bash
   grep -rn "summary_fit\.precision" --include="*.py" .
   ```
2. Replace each occurrence — see "After" pattern above.

**Rationale:** Same as D.1 — physics convention rendering.

*Decision: AD-2/13.50.DF + CRR §2.2 [BREACH] in `ARCHITECT_DECISIONS.md`.*

### 8.3 Semantic correction — Scatter `range=` is a point filter (Phase 13.46 FIX1 — AD-1/13.46.DF-FIX1)

Scatter `range=` was previously documented as view-clipping in some examples. **It is a point filter** — out-of-range points are removed before `stats` computation. `stats_dict` values reflect post-filter counts.

**Before (any version, ambiguous):**
```python
fig, ax, stats = d.scatter("y:x", range=((0, 10), (0, 5)))
# stats['n'] = total points in DataFrame  (view-clip semantics — what you might have assumed)
```

**After (Phase ≥ 13.46 FIX1, locked):**
```python
fig, ax, stats = d.scatter("y:x", range=((0, 10), (0, 5)))
# stats['n'] = points within range only  (point-filter semantics — matches hist/profile range= behaviour)
```

**Migration steps:**
- If your code depended on `stats['n']` being total-population, update to use a separate selectionless call OR use `selection=` for explicit filtering.
- If your code expected view-only clipping (e.g., points still drawn but stats unaffected), this is no longer available — scatter `range=` always filters.

*Decision: AD-1/13.46.DF-FIX1 in `ARCHITECT_DECISIONS.md`. Architect ruling 2026-05-28.*

### 8.4 OPEN cross-team items — workarounds until ADF-side fixes ship

These are NOT BREACHes — they are existing dfdraw behaviour with ADF-side fix work pending. Workarounds exist; production code at or before `PHASE_13_49_DF_FIX1_END` continues to function with the same caveats.

#### D.4.1 AD-37 — `same=True` color cycle reset across ADF calls

**Issue:** AliasDataFrame creates a fresh `DFDraw(df_subset)` on every `.draw()` call. This resets `self._last_ax` and the color-cycle index. `same=True` overlays from sequential ADF calls do NOT preserve color continuity.

**Workaround:** Build a single `DFDraw(df)` instance for the overlay sequence:
```python
# Instead of:
adf.draw("y1:x")
adf.draw("y2:x", same=True)    # color cycle reset between calls

# Use:
from dfdraw import DFDraw
d = DFDraw(adf.df)
d.draw("y1:x")
d.draw("y2:x", same=True)      # color cycle continuous within one DFDraw lifetime
```

**ADF-side fix:** Pending — Option C1b (cache `_last_ax` only, restore onto fresh DFDraw per call). Tracked in `tests/test_quantiles_profile.py:399` with skip marker. *Decision: AD-37, AD-50 in `ARCHITECT_DECISIONS.md`.*

#### D.4.2 AD-50 — DFDraw instance caching pattern (ADF-side implementation note)

**For ADF maintainers, not end users:** When implementing the AD-37 fix, do NOT cache the full DFDraw instance. `df_subset` is per-call-filtered (selection, entry_mask, subframe merges at `AliasDataFrame.py:10870, 11868, 12348`). Caching DFDraw would lose per-call filtering. The architect-ratified pattern is to cache `self._last_ax` only and restore it onto a fresh DFDraw per call. *Decision: AD-50 in `ARCHITECT_DECISIONS.md`.*

### 8.5 Non-breaking parallels — back-compat preserved

The following changes did NOT remove the old surface; they added a new one alongside. Existing code continues to work.

#### D.5.1 `legend=` polymorphic + `show_legend=` permanent back-compat (Phase 13.50 — AD-4/13.50.DF)

The new `legend=` kwarg accepts `True / False / 'shared' / 'first' / dict`. The original `show_legend=True/False` continues to work as a **permanent** back-compat parallel — NOT deprecated, NOT removed. If both are passed in the same call, `legend=` wins; no warning is emitted.

No migration action required. New code should prefer `legend=`.

#### D.5.2 `facet_by="column"` dual-path (Phase 13.31 — AD-78)

`facet_by=` now accepts DataFrame column names. The original channel-name enum (`'group_by'`, `'vector'`, `'quantiles'`) continues to work — they are disjoint by construction (channel names cannot be column names; the validator rejects collisions at dispatch time).

No migration action required. Existing `facet_by='group_by'` calls continue to function unchanged.

### 8.6 Test count and feature inventory drift

If you were verifying coverage claims from the previous Technical Summary (Phase 13.27 Commit 1: 663 tests / 62 features / 7 Verified), the current numbers are 1057 / 127 / 59. Always consult `CAPABILITY_MATRIX.md` for the live per-feature breakdown — it is auto-generated and refreshed at every phase close.

---
---
## Appendix A: Architecture and Governance

*Architecture context and governance discipline — useful for reviewers and contributors; not required reading for integration users.*

### A.1 MultiGraph Framework — Channel Arc (Phases 13.25 / 13.26 / 13.27)

The MultiGraph Framework is a coordinated three-phase effort that elevates
`dfdraw` from a per-call plotter to a **systematic visualization grammar**
for multi-dimensional differential analysis. Each phase locked one piece
of the grammar; together they let an analyst describe an N-channel plot
declaratively and let the framework decide how each channel is rendered.

#### A.1.1 Phase Hierarchy

| Phase | Role | Status |
|---|---|---|
| **13.25 Phase A** | Quantile rendering primitive on `profile()` | ✅ Closed (v1.3 + FIX1 + FIX2) |
| **13.26 Phase B** | N-channel framework, Algorithm A, channel-aware rendering | ✅ Closed (v1.2 implementation; FIX1 pending) |
| **13.27 Phase D** | Selection / weights vectors + facet refactor + delta facet | ✅ Shipped — Commit 1 (facet refactor) + Commit 2 (`selection_vector` / `weights_vector` / `delta_facet` per AD-61..AD-67); extended to all 4 plot types in Phase 13.32 (AD-79) |

#### A.1.2 The Visualization Grammar

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
across phases — see GP-2 in §21.1) plus a
priority-list fallback (`channels.priority.categorical` defaults to
`["color", "linestyle", "marker"]`).

#### A.1.3 Architectural Properties Locked Across the Three Phases

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

#### A.1.4 Phase 13.27 Commit 2 — Delivered (formerly listed as out-of-scope)

The Phase 13.27 Commit 2 deliverables — `selection_vector`, `weights_vector`, `delta_facet` (column-name `facet_by`), and hist + scatter facet integration — all **shipped** in subsequent phases:

- `selection_vector` / `weights_vector`: Phase 13.27 Commit 2 (AD-61..AD-68); 17 source citations across `drawer.py`, `channels.py`, `style.py`, `tests/`
- `delta_facet` (column-name `facet_by`): Phase 13.31 (AD-78) — 30 source citations across 7 files (most-cited AD in the codebase)
- Hist + scatter facet integration: Phase 13.32 (AD-79) — symmetric `facet_by_bins` / `facet_by_quantiles` extended to all 4 plot types via `_dispatch_faceted_render` hoisting

See §1.8 (column-name faceting), §1.9 (symmetric binning), §6.1 (N-D faceted return contract), and Appendix B for full phase history.

---



---

### A.2 dfdraw as Multidimensional Differential Analysis (MDA) Substrate

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
| 13.27 (Commit 2 — ✅ shipped) | `selection_vector` + `weights_vector` + `delta_facet` | The **core MDA operation** — declare M selections × N weight schemes and let the framework render the cross-product as a facet grid |
| 13.28 | Robust autorange + NaN/inf safety | Reliable plots on raw production data without per-call data hygiene |

The framework's contract — **per-pair invariance, byte-identical axes
state, factored legends, channel-aware encoding** — is precisely what an
MDA workflow needs to compare distributions across slicing dimensions
without rendering artifacts masking real differences.

---



---

### A.3 Governance and Review Discipline

dfdraw development follows a formal phase-lifecycle (proposal → multi-reviewer
panel → consolidated review → implementation → tag) with versioned governance
documents. This section summarises the discipline; full text in
`docs/ARCHITECT_DECISIONS.md` v1.0.1 (canonical AD registry, GP-1..GP-8) and
`docs/Organization-structure.md`.

#### A.3.1 Governance Principles (GP-1 through GP-8)

- **GP-1** — Style configurability lands at interface introduction, not deferred.
- **GP-2** — Internal APIs accepting new data-channel types must be list-based from day one (`EXPLICIT_RULES` is append-only).
- **GP-3** — Architect signals preserved verbatim with typos. Reformulating quotes has caused production bugs.
- **GP-4** — Backward-compat scope must be justified by production-usage verification (`grep`/AST against production scripts).
- **GP-5** — Drafter rotation across phases is healthy (different drafters bring different lenses; precedent: Phase 13.26 v1.0/v1.1 → v1.2 drafter handoff caught a structural improvement).
- **GP-6 (Render-the-artifact, Phase 13.49 FIX1)** — For HTML/PDF/notebook deliverables, at least one reviewer in the panel must render and navigate the artifact, not only read the diff. Diff-reading reviewers missed three functional HTML bugs (H-1 / H-2 / H-3) that were only caught after actual rendering. *Cross-subproject scope: dfdraw / ADF / GBAI.*
- **GP-7 (§0 pre-CRR gap-audit attestation, Phase 13.50)** — Coder must perform an explicit §0 attestation against the proposal's load-bearing list before opening the CRR; silent deferral of items is a Coder QRC R17 violation. *Cross-subproject scope.*
- **GP-8 (3-recurrence → infrastructure enforcement, Phase 13.50 FIX2)** — When the same class of issue recurs three times across distinct phases despite voluntary discipline (e.g., the `CAPABILITY_MATRIX.html` packaging gap), the next remediation must be a mechanical enforcement (a `run_tests.sh` check, a class-load assertion, a CI gate), not another reminder. *Cross-subproject scope.*

#### A.3.2 Coder Quick Reference Card v1.32+ — Reviewer QRC v1.31.1

**Binding §9 assertion-marker rule.** Each test class in a phase's invariance
test file must contain at least one assertion drawn from the proposal's §9
load-bearing list, marked inline as `# §9.<class>.<id>` (e.g.
`# §9.LegacyEquiv.1`). This pins each test body to a specific architect-
locked invariant; reviewers verify against the proposal §9 list.

#### A.3.3 Cross-Group Review Standard

Phases that touch the user-facing API surface (`DFDraw.draw / .hist / .hist2d /
.profile / .scatter`) follow a 3+3 cross-group review standard: 3 dfdraw
reviewers + 3 ADF reviewers (or equivalent cross-subproject). Phase 13.28
closure (5-0: Claude40 + Claude48 + GPT4×2 + Claude32 ADF) is the most
recent example.

#### A.3.4 Multi-Reviewer Model — Empirical Evidence

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

#### A.3.5 Two-Commit Patterns

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



---

### A.4 Testing Coverage

**Test suite:** 1057 tests (100% passing as of Phase 13.50.DF FIX2 / HEAD `07606c02`)
**Features:** 127
**Invariance tests:** 356 (A≡B semantic contracts; §9 markers introduced Phase 13.27, governance-locked Phase 13.49)
**Verified features:** 59 (features with at least one passing invariance test)
**Visual-primitive tests:** 27 (NEW layer introduced Phase 13.48 — primitive-only, renderer-free, deterministic)
**Smoke-only features:** 67
**Broken features:** 0

#### A.4.1 Capability matrix vs. test suite — what each guarantees

The library uses **two independent artifacts** to defend against regression. They are not redundant — each answers a different question, and neither alone is sufficient.

| Artifact | Answers the question | What it lists | How it is produced |
|---|---|---|---|
| **`CAPABILITY_MATRIX.md`** | *What does the library claim to do?* | One row per feature: name, phase introduced, AD reference, intended behavior, verification status | Auto-generated from source at HEAD via `scripts/generate_capability_matrix.py` |
| **Test suite** (`tests/`) | *Does the library actually do it?* | Behavioral validation — one or more tests per feature; checks the feature on synthetic inputs and asserts the expected output | Hand-written + auto-collected; runs via `run_tests.sh` |

**The capability matrix tells you what *exists*; the test suite tells you what *works*. Both are needed.**

A feature can be in the matrix without being in the test suite — that means it's been *implemented* but its behavior isn't yet *validated*. At HEAD `07606c02`, 67 features fall in this category (the **Smoke-only** count) — they don't crash on input, but their semantics have no invariance lock yet. Distinguishing implemented-and-validated from implemented-only is the matrix's job.

Tests come in three flavors, each catching a different class of bug:

1. **Unit tests** — does the feature produce the expected result on a synthetic input? (Catches: implementation bugs, regression after refactoring.)

2. **Invariance tests** (the `A ≡ B` markers in `tests/test_*invariance*.py`) — do two semantically-equivalent ways of calling the feature produce byte-identical state? Example: `adf.draw("[y1, y2]:x")` vector path ≡ two sequential `same=True` calls. Locked by the **§9 invariance markers** introduced in Phase 13.27 and governance-locked in Phase 13.49 — 356 markers at HEAD. (Catches: semantic drift between equivalent code paths, the most insidious regression class.)

3. **Visual-primitive tests** (Phase 13.48 — `tests/test_visual_primitive_*.py`) — do the rendering primitives produce the expected pixels? Tier-1 deterministic checks decoupled from the matplotlib renderer (renderer-free, no PNG-diff fragility). 27 tests at HEAD. (Catches: rendering-layer regressions that unit tests miss.)

The **59 "Verified" features** are those with at least one invariance test or visual-primitive test attached — the highest-confidence layer of the matrix. The remaining 67 features have only smoke-level coverage and are candidates for invariance lock as the project matures.

**When to consult which:**
- *"Does dfdraw support X?"* → `CAPABILITY_MATRIX.md`
- *"Is X verified to behave correctly?"* → matrix entry's verification status column; for the contract specifics, the linked invariance test
- *"What's the test coverage for X?"* → grep `tests/` for the feature name; the matrix lists the explicit per-feature test list under `tests:[...]` (Phase 13.49 — capability matrix traceability)
- *"Has X regressed?"* → run `./run_tests.sh` and compare against the green baseline tag

> For the current per-feature breakdown (which features are Verified vs Smoke-only vs Visual-evidenced), see `CAPABILITY_MATRIX.md` at HEAD — this is auto-generated by `scripts/generate_capability_matrix.py` and refreshed at every phase close. **The Technical Summary intentionally does NOT restate per-feature verification status; it goes stale fast. Always consult the live matrix.**

**Coverage areas (categorical):**
- All plot types: hist, scatter, profile, hist2d, hexbin, **profile2d** (NEW 13.39), **scatter3d** (NEW 13.39), **time-axis** (NEW 13.39), **cumulative histogram** (NEW 13.40) ✅
- Expression evaluation, vector expressions, multi-vector ✅
- Selection / filtering / weights / `selection_vector` / `weights_vector` ✅
- Group-by overlays + `group_by_bins` + `group_by_quantiles` ✅
- N-D faceting (ROW/COL/FIGID) (Phase 13.41) ✅
- Column-name `facet_by` dual-path (Phase 13.31, AD-78) ✅
- Symmetric `facet_by_bins` / `facet_by_quantiles` (Phase 13.32) ✅
- Normalized differential profiles (`delta`/`ratio`/`pull` — Phase 13.33) ✅
- Batch processing (dict and list formats) ✅
- PyArrow integration ✅ **Verified**
- Robust data handling (Phase 13.28) ✅
- Inline fits (Phase 13.42) — registry, vector convention, group_by composition ✅
- Summary fit figures (Phase 13.43) — 3 placement modes ✅
- Fit rendering overhaul (Phase 13.50) — `value_format`/`error_format` split, `precision_mode`, `_DISPLAY_NAMES` ✅
- Hist Poisson errors + linestyle cycling (Phase 13.37) ✅
- Scatter `xerr=`/`yerr=` + expression `color=`/`marker=` (Phase 13.38) ✅
- User style kwarg precedence (Phase 13.36) ✅
- Capability matrix traceability — `tests:[explicit list]` per feature (Phase 13.49) ✅
- Tier-1 visual testing framework (Phase 13.48) — 27 visual_primitive tests across feature categories
- `same=True` superposition + auto-color / auto-label (Phase 13.13) ✅ **Verified** (production-incident lock §9.NSC.1 against AD-80 sign-violation)

**Integration tests:**
- AliasDataFrame integration ✅ (AD-37 / AD-50 OPEN — ADF-side Option C1b pending)
- Lazy loading + subframe joins ✅
- ADF vector dispatch (Phase 13.16.DF) ✅
- Sanitize-for-plot + autorange across all 5 plot types (Phase 13.28.DF) ✅
- Inline-fit composition with `group_by` × `facet_by` (Phase 13.42+) ✅

**Test infrastructure (Phase 13.15.DF, extended through Phase 13.50.DF):**
- `tests/feature_taxonomy.py` — 127 features enumerated; each carries `tests:[explicit list]` field (AD-1/13.49.DF — no grandfathering)
- `tests/test_layer_classification.py` — smoke vs invariance vs `visual_primitive` (3-way classification post-Phase 13.48)
- `tests/test_meta_capability_matrix.py` — M.1 mandatory pass on commit, M.4 visual-tests claimable by any feature category (AD-3/13.49.DF), `KNOWN_UNCLAIMED §3.7` governance (AD-2/13.49.DF)
- `scripts/generate_capability_matrix.py` → `docs/CAPABILITY_MATRIX.md` + `docs/CAPABILITY_MATRIX.html` (HTML packaging mechanically enforced in `run_tests.sh` since Phase 13.50.DF FIX2 — closes 3-recurrence gap per GP-8)
- `run_tests.sh` — full/quick/matrix modes + `reviewer.zip` packaging + HTML rendering enforcement
- `_validate_forwarded_names()` — class-load R6 check on forwarder/signature parity (Pattern A discipline)

---



---

### A.5 Documentation References and Multi-Doc Architecture

The dfdraw documentation set:

| Document | Location | Role | Status |
|---|---|---|---|
| **`dfdraw_Technical_Summary.md`** | `./` (this document) | Conceptual reference — *what* and *why* | Current (v6) |
| **`dfdraw_Technical_Summary.html`** | `./` (rendered companion) | Same content, HTML-rendered | Generated from v6 |
| **`dfdraw_README.md`** | `./` | User guide — *how to start*; runnable intro | Existing — needs decision on overlap with Technical Summary |
| **`dfdraw_api_summary.md`** | `./` | Complete API reference — *how to call exactly* | Existing — needs rewrite (dfdraw team) |
| **`dfdraw_PLOTTING_LIBRARY_COMPARISON.md`** | `./` | Comparison with other visualization stacks | Existing — needs rewrite (dfdraw team) |
| **`ARCHITECT_DECISIONS.md`** | `./` | Numbered decision registry (`AD-N`, `AD-N/PHASE.DF`) | Current at v1.0.1 (commit `ee698002`) |
| **`CAPABILITY_MATRIX.md`** | `./` | Generated feature inventory at HEAD | Current at HEAD `07606c02` |
| **`PHASE_HISTORY.md`** | `./` | Chronological phase log (Appendix B of this document is the summary view) | Current at v1.11 |
| **AliasDataFrame Technical Summary** | `../../AliasDataFrame/docs/` | Authoritative for `apply_meta`, `register_subframe`, `draw_lazy`, alias resolution | ADF team-maintained |
| **GBregression Technical Summary** | `../../groupby_regression/docs/` | Authoritative for `make_parallel_fit_v4` and sibling fit methods, prediction registration | GB team-maintained |
| **`examples/time_series.py`** | ADF repo | Canonical end-to-end ADF + GB + dfdraw integration | Needs extension per §A.6 (9 missing-feature functions) |
| **`AI_Assistant_Brief.md`** | TBD | Web-based help system loading TS + API + tutorial | Separate phase — see Block G of v6 proposal |

**For integration questions:**
- Contact: Team3-dfdraw
- Documentation entry point: this Technical Summary
- Tutorial entry point: `examples/time_series.py`

**For bug reports:**
- Check: test suite (1,057 tests at HEAD `07606c02`)
- Verify: Minimal reproduction case
- Report: With test.log output

**For feature requests:**
- Check: Scope Boundaries (Section 7.1)
- Propose: Via specification document
- Discuss: Architecture team review


---

### A.6 Example Benchmark Plan — Extending `time_series.py` (architect-directed)

**Problem identified by 2026-06-04 cross-team review; tracked in v6 proposal Block E.** The functionality-completeness check for this Technical Summary was performed against the canonical production examples (`examples/time_series.py`, `examples/time_series_TroubleShooting.py`, `examples/makeSmoothMapsWithTPC.py`). Empirical audit (Claude36 + Sonnet26/27) found that **9 features shipped after Phase 13.27 have zero production-script witness**:

| Feature | Phase | Where described | Production witness |
|---|---|---|---|
| `weights_vector=` | 13.27 C2 | §1.17 | NONE (only `selection_vector` used) |
| `scatter3d()` | 13.39 | §1.14 | NONE |
| `profile2d()` | 13.39 | §1.13 | NONE |
| `time_format='auto'` | 13.39 | §1.16 | NONE |
| `hist_errors=True` | 13.37 | (needs §1.x — pending P2-8) | NONE |
| `cumulative=True/-1` | 13.40 | §1.15 | NONE |
| `same=True` (cross-method) | 13.13 | §6.7.1 | NONE |
| `draw_batch(specs)` direct | 13.14 | §5.5 | NONE (production uses `adf.draw_figures()`) |
| Inline `fit=` + `summary_fit=` | 13.42/13.43 | §2 | NONE |

**Consequence:** documentation against frozen Phase 13.27-era examples will recur every revision until examples cover the surface.

**Required action** (architect-directed, 2026-06-04): extend `examples/time_series.py` (or create a sibling `time_series_v2.py`) with one runnable function per major post-13.27 feature. The extension also serves as a **smoke-test benchmark layer** — each function should at minimum not crash on a known input dataset, providing a mechanical guarantee that the documented feature exists and dispatches.

**Proposed naming pattern** (following existing `time_series.py` conventions):
- `drawNclExampleSeries()` — scatter3d
- `drawProfile2DResidual()` — profile2d
- `drawTimeAxisRunStability()` — time-axis
- `drawHistWithPoissonErrors()` — hist_errors + linestyle_cycle
- `drawCumulativeDistribution()` — cumulative histogram
- `drawSameMethodOverlay()` — same=True cross-method
- `drawBatchDashboard()` — `adf.draw_figures(specs)` + `DFDraw.draw_batch(specs)` parity demonstration
- `drawWeightedComparison()` — `weights_vector=`
- `drawInlineFit()` + `drawSummaryFitFigure()` — `fit=` and `summary_fit=`

**Ownership:** Claude36 (ADF team) committed to co-drafting the extended `time_series.py` in a follow-on round. The dfdraw side commits to keeping `CAPABILITY_MATRIX.md` updated with `tests:[explicit list]` references to these example functions so the audit pattern (production-grep → doc cross-match) is mechanical.

**Verification loop going forward:**

```bash
# Architect-directed empirical audit method (Claude36 / Sonnet26-27 invented)
grep -h "\.draw\|set_style\|draw_figures\|draw_batch" \
  examples/time_series.py \
  examples/time_series_TroubleShooting.py \
  examples/makeSmoothMapsWithTPC.py \
  | grep -oP "[a-zA-Z_]+=\S+" | sort -u
# Any kwarg not documented in §1–§8 of this Technical Summary is a P1 candidate.
```

This audit pattern is proposed for adoption into the standard Technical Summary review discipline (potential addition to Coder QRC R17 or as a new R-rule).

---

### A.7 AI Agent for Online Help (Forward Direction)

**Architect-scoped simpler design** (Block G ruling, v6 proposal, 2026-06-04):

The agent is a **web-based help system** rather than a full code-generation agent. A user opens a cloud web interface and the system loads:

1. **dfdraw Technical Summary** (this document — md + html)
2. **dfdraw API summary** (`./dfdraw_api_summary.md`)
3. **AliasDataFrame Technical Summary** (`../../AliasDataFrame/docs/`)
4. **GBregression Technical Summary** (`../../groupby_regression/docs/`)
5. **The canonical tutorial** (`examples/time_series.py` and its extensions per §A.6)

The user can then ask questions and the agent provides online help grounded in this corpus.

**Why this is sufficient** (architect rationale): a well-written documentation set + a canonical example covers the question surface for users of the stack. The agent's value floor is the doc set's accuracy; if the docs are right, the agent doesn't need to do model-heavy code generation — it needs to retrieve and cite.

**Prototype scope:**
- Phase: `PHASE_DOC_AI_AGENT` (sibling to `PHASE_DOC_UPDATE`)
- Resources: to be estimated during prototyping
- Strong precondition: v6 + README + api_summary stable

**Strong precondition reinforced:** *doc accuracy is the agent's ceiling.* Build only after the documentation set is stable and the canonical example covers the feature surface (§A.6 extension).

---

---
---
## Appendix B: Version History

**Phase 13.50.DF FIX2 (Current — at HEAD `07606c02`, tag `PHASE_13_50_DF_FIX2_END`):**
- `run_tests.sh` mechanical enforcement: `CAPABILITY_MATRIX.html` packaging in reviewer zip is now a non-fatal post-zip assertion (closes 3-recurrence gap per GP-8)
- Tooling commit `df3057a3` tag-drift WARN downgrade scoped to declarative `'tag PHASE_X_END'` refs only
- AD-1..6/13.50.DF (Phase 13.50 ratifications carried forward)
- 1057 tests passing, 127 features, 356 invariance tests, 59 Verified, 27 visual_primitive

**Phase 13.50.DF FIX1 (tag `PHASE_13_50_DF_FIX1_END`):**
- P2-1 stale-taxonomy fix: `SUMMARY_FIT.orientation` "name" field updated post-rename (3/5 reviewers caught — Sonnet54, Sonnet56, Sonnet57)
- Two `_summary_fit.py` mop-up sites via parallel-sweep follow-on
- Coder QRC R17 EXTENSION added: parallel doc-surface sweep on rename / scope-change events; `feature_taxonomy.py` "name" fields + `CAPABILITY_MATRIX.md/.html` generated content + inline source comments fingerprinted via `grep -rn "OLD_TOKEN" tests/feature_taxonomy.py plots/ docs/`
- §4-facts-from-logs rule: CRR §4 sourced from `test_logs/test_full_*.log`, not working memory
- 1057 tests (same suite, stricter taxonomy)

**Phase 13.50.DF (tag `PHASE_13_50_DF_END`, gate 1057):**
- Fit rendering overhaul (proposal v2.5 after 4-round panel convergence v2.2 → v2.3 → v2.4 → v2.5)
- ⚠️ **BREACH §2.1**: `fit.text_format` style key REMOVED. Replaced by `fit.value_format` (default `.2g`) and `fit.error_format` (default `.1g`)
- ⚠️ **BREACH §2.2**: `summary_fit.precision` style key REMOVED. Replaced by `fit.value_format`/`fit.error_format` + `precision_mode='physics'`
- `FIT.display_names`: render-only `_DISPLAY_NAMES` map (slope→p1, intercept→p0 per ascending-powers convention)
- `FIT.textbox_kwargs_extensions`: `rename_params` / `value_format` / `error_format` / `precision_mode` per-call overrides
- `LEGEND.modes`: polymorphic `legend=` kwarg (`bool|str|dict|None`); `show_legend=` retained as permanent back-compat parallel
- `SUMMARY_FIT.placement`: 3 modes (figure / subfigure / pad) — pad pre-planned at GridSpec construction per Sonnet52_R1 P1-A
- 19 tests: F1-F16 visual_primitive + F17 cross-variant equivalence + F18 4-tuple `legend_topology` + `test_normalize_legend_spec_idempotent`
- §0 pre-CRR gap-audit attestation pattern introduced (proposed Coder QRC R17 — now GP-7)
- F19 textbox-bbox-overlap deferred to Tier-2 Phase 13.5X (architecturally — needs renderer)
- AD-1..6/13.50.DF; GP-7, GP-8 ratified

**Phase 13.49.DF FIX1 (tag `PHASE_13_49_DF_FIX1_END`):**
- HTML rendering fixes H-1 / H-2 / H-3 (architect rendered HTML in browser, found 3 bugs that 8-reviewer v1.1 panel missed)
- M.3 extended invariants: category-uniqueness + MD↔HTML count agreement
- Direct trigger for Reviewer QRC v1.31 Rule 14 caveat — now GP-6

**Phase 13.49.DF (tag `PHASE_13_49_DF_END`, gate 1038):**
- Capability Matrix Traceability: per-feature `tests:[explicit list]` field (Option A panel 9×[!])
- HTML matrix with expandable per-feature tests panel; orthogonal Visual column with 👁 badge
- 4 M-tests M.1-M.4; KNOWN_UNCLAIMED §3.7 governance with 64 seeded SPECIFIC target_phases
- AD-1..3/13.49.DF; META.capability_matrix feature added; Verified 56→57

**Phase 13.48.DF (tag `PHASE_13_48_DF_END`, gate 1034):**
- Tier-1 automated visual testing framework `VisualCheck` + 10 V-checks (V.1-V.10) + V.2 ragged-padding-safety lock
- New `visual_primitive` test layer (0 → 11 tests at close)
- +6 `VISUAL.*` features (Smoke-only at close)
- Tier 1 / Tier 2 split established (Tier 2 = renderer-driven, deferred)
- AD-1/13.48.DF

**Phase 13.46.DF + FIX1 (tags `PHASE_13_46_DF_END`, `PHASE_13_46_DF_FIX1_END`):**
- v1.0 §2.1 Option-1: faceted scatter `range=` shared-global view ruling — prevents last-cell-wins artifact
- FIX1: scatter `range=` semantic locked as point filter (removes out-of-range points, NOT view clip) per architect directive 2026-05-28
- `stats_dict` computed AFTER filter — honest counts
- AD-1/13.46.DF, AD-1/13.46.DF-FIX1

**Phase 13.43.DF (tag `PHASE_13_43_DF_END`):**
- `summary_fit=` kwarg: standalone fit-result summary figures (table / parameter-trend)
- 3 placement modes: `'figure'` (separate Figure), `'subfigure'` (per-panel inset slices), `'pad'` (GridSpec pre-planned)
- `summary_fit.data_format` style key — opt-in pandas DataFrame ships alongside figures (rejected public `fit_results_to_df()` per architect override)
- Per-channel fit aggregation Option A (top-level `stats['fit']` flat list + per-cell dict — additive)
- Kwarg renamed `summary_pad` → `summary_fit` throughout (OQ-A1)
- `same=True` → replace mode (accumulate deferred per OQ-A2)
- AD-1..7/13.43.DF; 13 `summary_fit.*` style keys

**Phase 13.42.DF + FIX1 (tags `PHASE_13_42_DF_END`, `PHASE_13_42_DF_FIX1_END`):**
- Inline fits: `fit=` kwarg with str / dict / callable forms; scipy.optimize.curve_fit backend
- Vector-convention broadcasting (Phase 13.16 standard — scalar broadcast, list per-curve N:1/N:N)
- Predefined registry: gauss, linear, pol0..pol5, expo, lorentz, powerlaw; landau / crystalball / breitwigner deferred
- 4-tier initial-parameter resolution: user `initial` → user `guess` callable → registry heuristic → scipy default + warning
- `stats['fit']` canonical shape locked: outer list = curves (render order); inner list = fits per curve; dict-keyed with group_by/facet_by
- FIX1 (architect MODIFY ratifications): `yerr=` alone is opt-in (R3); stacked + group_by + fit → N per-group fits (R4); `fit_textbox_kwargs` LOCKED including per-call `fontsize` (R5, regression test F.33)
- ⚠️ **BREACH FIX1 D-1**: `use_errors` default flip; **BREACH FIX1 D-2**: `_style_get` silent fix
- 7 `fit.*` style keys (Phase 13.50 BREACH later reorganizes these)
- AD-1..6/13.42.DF, AD-1..3/13.42.DF-FIX1

**Phase 13.41.DF (tag `PHASE_13_41_DF_END`):**
- ROW/COL/FIGID convention LOCKED for N-D faceting (carried unchanged through v1.6)
- `facet_by=['ROW', 'COL']` syntax with `FIGID` as figure-level grouping channel
- Return contract for `FIGID` calls: `(List[Figure], List[axes_2d], List[stats_dict])` — non-standard, explicit
- `share_x='row'` / `share_y='col'` / `share_across_figures=True` defaults
- AD-1..3/13.41.DF

**Phase 13.40.DF (tag `PHASE_13_40_DF_END`):**
- Cumulative histogram via matplotlib's native `cumulative` parameter (matches ROOT `TH1::Draw("cumulative")`)
- Composes with `hist_norm` / `group_by` / `facet_by` / `stacked`
- M5 correctness guard: `hist_errors=True` + `cumulative=True/-1` raises `NotImplementedError`
- AD-1..2/13.40.DF

**Phase 13.39.DF (tag `PHASE_13_39_DF_END`):**
- 2D Profile: `draw_profile2d()` dispatched on `colon_count==2`; scipy.stats.binned_statistic_2d backend
- Time Axis: `time_format=` dtype auto-detect (datetime64 vs epoch-seconds — two distinct paths)
- Scatter3D: `type='scatter3d'` via mpl_toolkits.mplot3d; scipy required dep
- AD-1..3/13.39.DF

**Phase 13.38.DF (tag `PHASE_13_38_DF_END`):**
- `_process_color()` dispatch fix: column-name check precedes `to_rgba()` conversion (Sonnet53_R2 + Opus2 + Sonet50 3-reviewer convergence)
- Scatter error bars: `xerr=`, `yerr=` column-expression parameters; NaN/inf in error column subject to `nan_policy`
- Expression-based `color=` and `marker=` for scatter (per-point encoding)
- AD-1..3/13.38.DF; backward-compat lock test §9.ECM.6

**Phase 13.37.DF (tag `PHASE_13_37_DF_END`):**
- `hist_errors=True` adds Poisson ±√N error bars on `hist()`
- `linestyle_cycle=True` cycles linestyles alongside color within group_by; respects AD-1/13.36.DF explicit-kwarg precedence
- BUG-016 retained in PHASE_NEXT_STEPS per explicit architect sign-off (*"Yes. I added it and I want to have it there."*)
- AD-1..3/13.37.DF + CP items

**Phase 13.36.DF (tag `PHASE_13_36_DF_END`):**
- User-specified per-call style kwargs take priority over channel auto-cycle; applied uniformly to ALL groups in the call
- Sentinel: `None` = user did not pass; non-`None` = uniform apply
- Named-param forwarding (`_user_color`, `_user_marker`, `_user_markersize`), not kwargs pop
- AD-1/13.36.DF

**Phase 13.35.DF (tag `PHASE_13_35_DF_END`):**
- `group_by_bins=` and `hist_norm=` now work on `hist()`
- Production reproducer: `adf.draw("dyp_I6-dyp_recoV2", type="hist", group_by="z", group_by_bins=5, ..., facet_by="sec")` no longer raises `AttributeError`
- AD-1/13.35.DF

**Phase 13.34.DF + FIX1 + FIX2 (governance — taxonomy refresh):**
- Capability matrix update-at-phase-end rule established (AD-1/13.34.DF)
- BUG-010 untracked test file; BUG-011 `run_tests.sh` pre-bundle staging check
- §9.MED.1 lock: `central='median'` must use MAD-sigma (1 xfailed test confirms pre-existing inconsistency from Phase 13.33 CRR §11)

**Phase 13.33.DF (tag `PHASE_13_33_DF_END`):**
- Normalized differential profiles: `normalize='delta'|'ratio'|'pull'` modes
- Reference convention LOCKED: `vector[0]` = signal, `vector[1]` = reference; `delta = v[0]−v[1]`; matches ROOT `h_data.Divide(h_mc)`
- Pull mode with ±1σ, ±2σ bands (formula: `(v[0]−v[1]) / √(σ₀²/n₀ + σ₁²/n₁)`)
- `group_by` + `facet_by` both in scope from v1.0 — no deferral (architect: *"Has to be supported from beginning"*)
- AD-80..82 (renumbered from AD-70..72 in v1.1 because AD-69..77 taken by Phase 13.28)

**Phase 13.32.DF (tag `PHASE_13_32_DF_END`):**
- `facet_by_bins` / `facet_by_quantiles` symmetric across all 4 plot types (profile / hist / hist2d / scatter)
- Binning hoisted to `_dispatch_faceted_render` BEFORE per-subplot recursion (single implementation, no drift)
- Closes production incident: `quantiles=` silently dropped when `group_by=` set; `facet_by='group_by'` ignored `group_by_bins/_quantiles`
- AD-79

**Phase 13.31.DF (tag `PHASE_13_31_DF_END`):**
- `facet_by=` accepts DataFrame column names in addition to channel-name enum (dual-path tagged union)
- Disambiguation: channel-name first, then column check, then `ValueError` naming both alternatives
- Sonnet P1 catch: `facet_by` MUST NOT be added to `_PROFILE_COLUMN_REFERENCES` — validation moved to `_dispatch_faceted_render()` so existing `facet_by="group_by"` calls don't break (would have failed Phase 13.30's column-reference validator)
- Direct trigger: production reproducer M. Ivanov 2026-05-12 — `adfSec.draw("dyp_I0345_rms:row", facet_by="side", ...)`
- AD-78 (most-cited AD in codebase — 30 occurrences across 7 files)

**Phase 13.27.DF Commit 1 (MultiGraph Phase D, profile-only):**
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
---
## Appendix D: Quick Decision Matrix

**"Should dfdraw do X?"**

| Question | Answer | Reason |
|----------|--------|--------|
| Plot types beyond 5 core types? | Yes — 3 added Phase 13.39 | `profile2d`, `scatter3d`, time-axis (any plot type accepts `time_format=`) |
| Automatic fitting? | Yes — Phase 13.42 | `fit=` kwarg, scipy.optimize backend; see §2 |
| Schema validation? | No | Flexibility by design |
| Custom aggregations? | No | Use pandas first, then plot |
| Interactive widgets? | No | Matplotlib backend concern |
| 3D plots? | Yes — Phase 13.39 | `scatter3d` only; not extending to full 3D framework |
| Animation? | No | Out of scope |
| Size control? | Yes | Already supported |
| Style control? | Yes | Already supported + new `precision_mode`, `legend=` polymorphic |
| Statistics display? | Yes | Already supported + `summary_fit=` Phase 13.43 |

---
---
**Document Status:** Updated through Phase 13.50.DF FIX2 (HEAD `07606c02`, 1057 tests, 127 features, 356 invariance tests, 59 Verified, 27 visual_primitive)
**Maintainer:** Team3-dfdraw
**Last Updated:** 2026-06-04
**Version:** Phase 13.50.DF FIX2
**Companion documents:** `ARCHITECT_DECISIONS.md` v1.0.1 (canonical AD registry, supersedes `STYLING_FRAMEWORK_DECISIONS.md` §2 AD catalog), `CAPABILITY_MATRIX.md` (per-feature verification status), `PHASE_HISTORY.md` v1.11 (phase chronology), `examples/time_series.py` (canonical ADF + GB regression + dfdraw end-to-end tutorial — ADF team-maintained)
**Next Update:** After PHASE_DOC_UPDATE closure. §5.1 (ADF integration) and §5.2 (GB regression integration) are stubs awaiting authoritative content from the ADF and GB teams.

