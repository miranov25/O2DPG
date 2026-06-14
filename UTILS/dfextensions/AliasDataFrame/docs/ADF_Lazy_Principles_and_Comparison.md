# ADF_Lazy_Principles_and_Comparison.md — what other systems do, and our preliminary principles for lazy loading and evaluation

**Status:** PRELIMINARY v0.6 — *not binding.* Fixes a self-containment defect: "Option A/B" was referenced in §6 but no longer defined after the v0.3 dissolution — both options are now defined in §4, and the two architect decisions in §6 are restated in plain self-contained terms. A-2 updated to the integrated-calibration form. **Remaining before `[OK]`/ratification: architect to confirm the §6 A-1/A-2 decisions read correctly.**
**Author:** fable5_5 (ADF) · **Date:** 2026-06-13 · **Readability rule:** full sentences; every term defined at first use; an example for every principle and open question.
**Source bundle for all line references:** `sources_adf.zip`, `AliasDataFrame.py` = 14,238 lines, `LazyTreeReader.py` = 198, `LazyChainReader.py` = 371. (No commit SHA embedded in the bundle; line counts drift across HEADs — other review bundles reported 13,666 and 14,148.)

### Changelog v0.5 → v0.6 (architect review)
- **Self-containment fix.** "Option A" and "Option B" are now **defined in §4** (they were referenced in §6 but undefined after the v0.3 dissolution of the A/B section). §6 A-1 and A-2 are restated in plain, self-contained terms, not just as labels.
- **A-2 updated.** Calibration is now stated as an **integrated** capability: ADF both runs the calibration (measure + store a per-platform profile) and reads the stored calibration automatically when estimating — per architect refinement 2026-06-13.

### Changelog v0.4 → v0.5 (from the v0.4 review panel)
- **PP-6 (P2-1, source-confirmed).** The hook-alias persistence caveat is qualified to **`draw()` only**. `draw()` calls `_ensure_vector_kwargs_aliases` (`:11266`) *before* the `already_materialized` snapshot (`:11295`), so hook aliases persist; `draw_batch()` (snapshot `:12379` before ensure `:12411`) and `draw_figures()` (snapshot `:12656` before ensure `:12731`) snapshot first and clean them up — no gap there.
- **PP-4 (ADV-1).** Clarified that `read_tree_lazy` and `read_chain_lazy` are distinct constructors; "same" means same API shape and same numerical results, not a single entry point.
- **§4 (ADV-2).** Named the state mechanism for the column-awareness caveat: `_subframe_loaded[name] = True` (`:2136`).
- **§6 (P2-2).** A-1/A-2 quotes retained; flagged for architect confirmation (both are phrased "I assume…", so confirm the decision reading matches intent).

### Changelog v0.3 → v0.4 (from the review panel)
- **§2 / PP-7 (P1-1, source-confirmed).** `estimate_memory` works in eager and **chain-lazy** mode only; **single-tree lazy raises `AttributeError`** because `LazyTreeReader` has no `estimate_memory` (only `LazyChainReader` does, `:307`). The earlier "both modes" claim is corrected; adding the method to `LazyTreeReader` is part of the PP-7 work.
- **PP-4 (P1-2, source-confirmed).** Rewritten: the chain is **branch-lazy, not file-touch-lazy** — `LazyChainReader.load_branches` (`:231`) reads each requested branch from *all* files. Row-window / file-touch laziness is future scope.
- **§6 A-1/A-2 (P2-1).** Architect decisions now carry verbatim quotes with `[ARCHITECT-VERBATIM]` markers.
- **PP-3 (P2-2).** The `register_function(input_columns=…)` contract is marked **proposed**; `register_function` (`:11522`) currently has no such parameter. `register_evaluator`'s `coord_columns` (`:11619`) is the existing declaration path.
- **§2/§7 (P2-3).** Line count corrected to 14,238 (this bundle) with drift note.
- **§3 (P2-4).** uproot row now distinguishes the user-facing column-selection syntax (regex/glob — borrowed) from the internal regex dependency resolver (deprecated in §1).
- **PP-6 (P2-5).** Added the hook-alias persistence caveat: vector-kwargs aliases persist regardless of a "drop" strategy (open cleanup).
- **§4 (Appendix A' P2).** Added the unification-vs-pruning caveat: the subframe must stay column-aware after first access.
- **PP-3 (ADV-2).** "fail loudly" now defaults to **raise**, not warn.
- **§6 (ADV-1).** `register_subframe_chain` promoted to A-1(b).

---

## §0 Plain summary (read this first)

We want to read and compute only the data a task actually needs, so analysis of datasets far larger than memory stays fast and small. Two mechanisms do this together: **lazy loading** (read a column from disk only when touched) and **lazy evaluation** (compute a derived column only when used). We are not inventing this — mature columnar and database systems use related forms of the idea under standard names (§3). This document states the principles we are converging on with an example for each (§5), lists the open questions (§6), and is honest about what already exists in the code and what does not (§2). The subframe question that earlier looked like a blocking either/or is an additive policy choice (§4).

---

## §0b Glossary

- **branch / column:** one field of a table, e.g. `pt`.
- **table / tree:** one set of columns sharing the same rows (a ROOT TTree).
- **alias:** a derived column defined by a formula over other columns, e.g. `r = sqrt(x*x + y*y)`; behaves like a real column but is computed.
- **lazy loading:** reading a column from disk only when first touched.
- **lazy evaluation:** computing an alias only when first needed.
- **branch-lazy vs file-touch-lazy:** *branch-lazy* = load only the requested columns (but from all files of a chain); *file-touch-lazy* = additionally skip files whose row range is not needed. ADF is branch-lazy today; file-touch is future scope.
- **AST:** the parsed structure of a formula; from it we can list the columns it references without running it.
- **subframe:** a second table joined to the main one (e.g. AO2D `Tracks` joined to `Events`).
- **join index:** the column(s) linking two tables.
- **materialize:** to actually load or compute a column into memory.
- **caching strategy:** the rule for what happens to a column after use (§6, A-4).
- **projection pushdown:** "read only the columns the query references."
- **predicate pushdown:** "read only the rows that pass the filter."
- **cost model / calibration:** a formula estimating time or memory from counts × per-unit constants; calibration measures those constants per hardware.
- **reader / seam:** the one piece of code answering "give me columns X and Y of table T for rows A to B."
- **typetracer:** a fake array with shape and types but no data, run only to see which columns get touched.

---

## §1 Goal (architect statement, 2026-06-13)

Lazy evaluation **and loading** with **minimal CPU and memory**, where:

1. the columns a task needs are found by taking the formula apart (AST), or, for an opaque function we cannot take apart, by the function declaring its inputs;
2. caching is governed by strategies the user can choose;
3. those strategies can be chosen automatically when the user prefers, based on a cost estimate;
4. CPU time, load time, and memory can be estimated by analytical formulas, with constants measured per machine and storage type and stored as a reusable per-platform profile;
5. we reuse the existing analysis machinery rather than writing a second one. **Caveat:** two analysis paths exist today — `_analyze_expression` (the AST analyzer, `:3755`) and the text-split parse inside `get_required_branches` (`:6828`). Reuse means consolidating onto the AST path, which is step 0 of the work (§8), not a finished foundation.

---

## §2 What already exists in AliasDataFrame today (honest inventory, source-verified)

| Capability | State today |
|---|---|
| Lazy reading of one tree: touch a column, that column loads | Works, tested |
| Lazy reading of a chain | Works, **branch-lazy** (loads requested branches from all files; `LazyChainReader.load_branches` `:231` loops every file); file-handle cache defaults to 8; file-touch laziness not yet implemented (see PP-4) |
| `get_required_branches` (`:6776`) | Handles `expr`, `selection`, `group_by`, `color`; its `expr` parse uses a text split (`:6828`), not the AST analyzer. Draw call sites (`draw()` `:11277`, `draw_batch()` `:12357`, `draw_figures()` `:12692`) pass only those four, so branches named only via `weights`, `facet_by`, `selection_vector`, or fit columns may be missed for branch pre-loading |
| Alias materialization (`_parse_expr_aliases` `:10902`; vector-kwargs path) | Receives `weights` and handles `selection_vector`/`weights_vector`/`facet_by` for *materialization* — separate from branch pre-loading above |
| AST analyzer (`_analyze_expression` `:3755`) | AST-based, but its `known_funcs` (`:3803–3815`) omits registered functions, so `corr(xM, driftM)` misclassifies `corr` as a column. Closing this is part of the lazy track |
| Memory estimator (`estimate_memory` `:6544`) | **Works in eager and chain-lazy mode only.** Single-tree lazy (`read_tree_lazy`) raises `AttributeError` — `LazyTreeReader` has no `estimate_memory`; only `LazyChainReader` does (`:307`). Adding it to `LazyTreeReader` is part of the PP-7 work |
| Plain `selection=`/`weights=` alias strings (F-A contract) | Require `lazy=True` on all surfaces (BUG_20260610 v1.1); must be in the FormularV3 audit scope |
| Eager subframe join | Works |
| Lazy subframe from a chain (`register_subframe_chain` `:1885`) | Works |
| Lazy subframe from a single tree (`register_subframe_lazy` `:1729`) | Implemented; per the architect never used in production. Its init crash bug (BUG_20260116) is a separate defect from the load-all *policy* (§4) |
| `keep_materialized` flag (`:1024`) | A single flag, not yet a chooseable strategy; the seed of PP-6 |

Reader layer is already separate (`LazyTreeReader.py` 198 ln, `LazyChainReader.py` 371 ln); `AliasDataFrame.py` ≈14,238 ln (this bundle; count drifts with HEAD). This matters for §7.

---

## §3 What other systems do (the comparison)

| System | How it picks which columns to load | How it caches | How it estimates cost | What ADF borrows |
|---|---|---|---|---|
| uproot (our layer) | reads only the columns the formula, cut, and aliases name | built-in array cache | — | the user-facing regex/glob column-selection **syntax** (not an internal regex resolver) |
| ROOT TTree::Draw | reads only the branches the formulas name | — | — | the original proof that "load only what the formula names" works |
| ROOT RDataFrame | reads only the columns its graph uses | — | — | the single reader seam for any format |
| coffea / dask-awkward | load-on-access, or trace on a fake array | three modes: on-access / lazy-graph / all | — | the trace fallback; chooseable modes |
| Vaex | computed columns treated like real ones | streams from disk | — | the stored-≡-computed-column idea |
| Polars | projection + predicate pushdown; reuses sub-expressions | toggleable optimizer passes | — | configurable optimization; sub-expression reuse |
| Spark | derives needed columns from the bound query | user picks a storage level | cost-based + runtime re-plan | the storage-level cache model; estimate-then-plan |
| PostgreSQL / DuckDB | binds columns against the catalog | — | counts × calibrated per-unit constants | the cost-formula shape; showing the estimate |

Plain notes: **uproot** reads only the branches its formulas/cut/aliases name; its column selector takes a list, wildcard, or regex — we borrow that *user-facing selection syntax*, which is a different thing from the internal regex dependency resolver §1 deprecates. **TTree::Draw** activates only the formula's branches. **RDataFrame** reads any format through one replaceable reader. **coffea / dask-awkward** offers three user-chosen modes and traces opaque functions on a fake array; its default-on-failure can be to read everything, the opposite of our PP-3 rule. **Vaex** is the closest in spirit — a design reference; adoption risk to be assessed separately. **Polars** is the maintained version of goal point 1, with toggleable passes. **Spark / SQL DBs** show caching and cost: storage levels, runtime re-planning, and counts × hardware-calibrated constants.

*Scope note:* ADF has a sound basis for **projection pushdown**. True **predicate pushdown** (push the row filter into the reader) is **future scope**; today selection masks apply after the needed columns load. It must not be an implicit acceptance criterion for the first audit.

---

## §4 How subframe loading works today, and how column-level pruning is added (additively)

**The two options (defined here so the document is self-contained — these are the labels the architect used):**
- **Option A:** column-level loading applies to the **main table only**. Subframes keep today's behavior: touching any one column of a subframe loads **all** of its registered columns, and the subframe then becomes permanently in-memory.
- **Option B (decided):** column-level loading applies to **subframes too** — touching `Tracks.pt` loads only `pt` plus the join-index columns, not the whole 40-column table. It is added **additively**: existing/eager callers keep the old behavior unchanged, and pruning happens on the lazy path, so nothing breaks and no governance re-ratification is needed.

**Today ("7.5a", originally a coder proposal, ratified):** #3 "load all, join once" — touching any column of a registered-lazy subframe loads all its registered columns; #5 "unification" — once loaded, the subframe becomes permanently in-memory (docstring at `:2101`).

**Why it hurts:** an AO2D `Tracks` subframe has 20–40 columns; touching `Tracks.pt` loads all 40 and pins them.

**Key source fact:** column-level loading is **not missing**. `_load_lazy_subframe` (`:2110–2117`) already loads a subset when columns were given at registration (`if config['columns'] is not None:` → listed columns ∪ index, `:2113`); it loads all only when `columns is None` (`:2116`). So pruning is **additive**:
- **Default stays load-all** for eager/`columns=None` callers → nothing breaks, **no re-ratification** needed (additive-only rule).
- **On the lazy path, pruning is the default** — the access/draw path computes the needed subframe columns per access and passes them to the existing subset path.
- **Minimal load set:** *requested child columns + child join-index + parent join-index* — so a join never silently fails or over-loads.

**Implementation caveat (panel, must be resolved before coding).** Per-access pruning conflicts with decision #5 as written: if the first access to `Tracks.pt` loads `{pt, index}` and then marks the subframe fully loaded/eager, a later access to `Tracks.eta` cannot re-enter the lazy path and would error or return wrong values. The implementation must therefore keep the subframe **column-aware after first access** — either do not finalize it as fully-loaded until all required columns are loaded, or allow the lazy path to re-enter for not-yet-loaded columns. The relevant state flag today is `_subframe_loaded[name] = True` (`:2136`), set on load; the fix governs when/whether that flips. This is an implementation-spec item, not a blocker for this document, but it is recorded here so the coder does not trip on it.

This dissolves the old A/B binary into a PP-6 policy value (default-by-mode).

---

## §5 Preliminary principles (each with an example)

- **PP-1 — One mechanism for stored and computed columns.** Resolved by the same step. *Everywhere.* *Example: asking for `r=sqrt(x*x+y*y)` and for stored `x` go through one resolver.*
- **PP-2 — Load and compute only what a task needs.** Touch a column → that column (and, for a subframe, the join-index columns). *Main table now; subframes additively on the lazy path (§4).* *Example: `adf.draw('pt:time')` on a 100-column tree loads 2 columns.*
- **PP-3 — Find needed columns by formula first, by declaration second, and **raise** otherwise.** Use the AST for our formulas. For an opaque function, require it to declare its inputs. *A declaration path already exists for evaluators (`register_evaluator` requires `coord_columns`, `:11619`).* The remaining gap is generic callables via `register_function` (`:11522`), which today takes only `(name, func, overwrite)`. **Proposed** first contract: add explicit input declaration (e.g. `register_function("f", fn, input_columns=["x","y"])`) — this parameter does not exist yet. Typetracer tracing is **future** and works only for typetracer-compatible functions, not arbitrary C++/Python lambdas. If the column set cannot be determined, **raise** — not a warn-then-full-load fallback. *Everywhere.*
- **PP-4 — One file and a chain behave the same (same API shape and same numerical results; note `read_tree_lazy` and `read_chain_lazy` are distinct constructors, not one entry point).** *Current implementation is **branch-lazy** across the whole chain: a requested branch is read from all files (`LazyChainReader.load_branches` `:231`). Row-window / file-touch laziness — skipping files whose entry range is not needed — is a **future optimization**, not current behavior.* *Example: switching from one file to a directory of a hundred needs no analysis-code change; today it still reads the requested branches from all hundred.*
- **PP-5 — Lazy gives the same numbers as eager.** *Everywhere; the master contract. Covered by the I1 invariance test (`read_tree_lazy() + materialize == read_tree()`); the audit extends it.*
- **PP-6 — Caching is a user-chosen strategy: simple switches with named shortcuts.** Switches: keep / spill / drop / pin-by-pattern / pin-by-alias-closure / subframe-prune. **No silent eviction:** raw and user-created non-alias columns are protected unless the user explicitly opts into dropping them (consistent with `dematerialize()`). **Caveat (`draw()` only):** vector-kwargs/hook aliases (`facet_by`, `selection_vector`, `weights_vector`) materialized via `_ensure_vector_kwargs_aliases` persist regardless of a "drop" strategy *in `draw()`*, because `draw()` calls `_ensure_vector_kwargs_aliases` (`:11266`) before its `already_materialized` snapshot (`:11295`), so those aliases are not in `we_added` and are never dropped. `draw_batch()` (snapshot `:12379` before ensure `:12411`) and `draw_figures()` (snapshot `:12656` before ensure `:12731`) take the snapshot first and correctly include hook aliases in `we_added`, so they have no gap. This is a known open cleanup for `draw()` (CLEANUP.hook_alias_tracking; AD-2 item 7; TS v1.8 §8.4); PP-6's "drop" does not yet apply to hook aliases on the `draw()` surface. *Example: pin all `Tracks.*pt*` (pattern) and separately pin everything alias `dca` needs (closure).*
- **PP-7 — Estimate the cost, show it, then optionally let the system choose.** *The memory/bytes term exists for eager and chain-lazy (`estimate_memory` `:6544`); single-tree lazy needs the method added to `LazyTreeReader`, and the time/CPU term (calibrated constants) is new.* First implementation is **advisory only**; automatic strategy choice waits until estimates are validated against measurements. *Everywhere.*
- **PP-8 — One reader behind one seam; nothing above it is ROOT-specific.** *Everywhere.* *Example: adding RNtuple means a new reader and no change to the analysis layer.*

---

## §6 Open questions (NOT yet principles)

- **A-1 — DECIDED: Option B.** *Plain, self-contained statement of the decision:* on the lazy path, touching a subframe column loads only that column plus the join-index columns — not the whole subframe; existing/eager callers keep today's load-all. Added additively (§4 defines both options and the mechanism). `[ARCHITECT-VERBATIM, 2026-06-13]`: *"I assume we have already decided on option B."* (The quote is phrased "I assume" — architect to confirm it means this decision.)
  - **A-1(b) — open:** confirm the same pruning applies to `register_subframe_chain` (`:1885`), which has the same `columns` conditional.
- **A-2 — DECIDED: integrated calibration.** *Plain, self-contained statement:* both halves are built into ADF, not a manual external step — (1) a **calibration process** that measures the cost constants (time per byte read, decompression per byte, per-column overhead, CPU per row per operation, memory per column) on the current hardware/storage and stores them as a named per-platform profile; and (2) **reading the calibration** — automatically loading the matching stored profile when ADF produces an estimate. `[ARCHITECT-VERBATIM, 2026-06-13]`: *"Can we calibrate it and store calibration data for a given hardware and software setup? I assume that is the standard approach?"* Architect refinement (2026-06-13): *the calibration process and the reading of the calibration must both be integrated.* No numbers enter the design until measured; acceptance is statistical. *Scenarios to measure: local-SSD cold vs hot; chain over ~100 network-disk directories; compressed vs uncompressed; 2-of-100 columns vs narrow tree; alias-heavy vs raw.*
- **A-3 — Unknown dependencies.** Require declared inputs for opaque functions (gap = generic `register_function`); otherwise raise. Typetracer later.
- **A-4 — Caching strategy names.** `keep` / `drop` first (mapped to `keep_materialized`), `bounded` later. Open: match Spark vocabulary (`MEMORY_ONLY`, `DISK_ONLY`) or keep the simple aliases. *keep:* stay in memory (interactive). *drop:* free after use (batch). *bounded:* keep a budget of most-recently-used columns.
- **A-5 — Cost-model depth.** Level 1 (advisory estimate) first — which means adding `estimate_memory` to `LazyTreeReader` and extending all paths with rows, operation count, and calibrated time. Level 2 (auto-select) later; level 3 (runtime re-plan) deferred.
- **A-6 — Generalizing to a shared parent.** Defer until the lazy instance is validated. *The dfdraw lesson: `dfdraw_PRINCIPLES.md` was drafted before the draw-routing architecture (AD-1/13.55.ADF) was finalized, so several principles had to be rewritten.*
- **A-7 — RDataFrameDSL boundary.** Is it a reader backend behind the seam (PP-8) or a consumer of ADF output? One architect sentence fixes it; gates the FormularV3 audit's Block F scope. Forward scope (RDataFrameDSL unfinished; bridge partial), not an audit gate.

---

## §7 Can this be done with the current monolithic AliasDataFrame.py?

Short answer: **yes, without splitting the monolith — new capabilities as new sibling files.**

- The loading/dependency work is **additive** to code already there.
- **Subframe pruning (§4)** needs two coordinated changes: (1) upstream per-access column computation; (2) passing that set to `_load_lazy_subframe`'s existing subset path. The edit to `_load_lazy_subframe` is small; the new piece is upstream — plus the column-awareness reconciliation in §4's caveat.
- The **new pieces** — the cost estimator's time/CPU term, the `LazyTreeReader.estimate_memory` addition, and the caching-strategy object — should be **separate modules**, as `LazyTreeReader`/`LazyChainReader` already are, so the ≈14,200-line file does not grow.
- The "many small, safe iterations" property holds **only after step 0** (§8): consolidating column resolution onto one AST path. Until then the two paths are split-brained and changes ripple.
- Each step is checked by running the task twice — lazy and eager — and comparing numbers (PP-5).
- **Not coupled here:** a full monolith decomposition (a separate decision, part of the Phase-14 ADFStore idea).

---

## §8 Sequence (small steps, each checkable)

**Step 0:** migrate `get_required_branches` `expr=` parsing onto `_analyze_expression`, and close the registered-function gap in `_analyze_expression` (`:3803–3815`). Until this lands, §7's small-iteration property does not hold.

Then: this document (with decisions) → mixed-panel ratification → benchmark step to measure and store the A-2 constants → build the PP-7 advisory estimator (add `LazyTreeReader.estimate_memory`; add rows + ops + calibrated time) → lazy audit (FormularV3) running every task twice and comparing lazy vs eager → fix step (subframe pruning §4 incl. the column-awareness caveat; the F-A / `facet_by` / `weights` branch-load gaps; file-touch laziness if scoped in) → add the lazy chapter to the Technical Summary → generalize to the shared parent (A-6) → propose ADFStore only if the audit proves it is needed.

---

## §9 References (sources for §3)

uproot documentation (uproot.readthedocs.io); ROOT TTree and RDataFrame documentation incl. `ROOT::RDF::RDataSource` (root.cern); coffea NanoEvents and dask-awkward "necessary columns" documentation; Vaex documentation (vaex.io); Polars lazy optimizations (docs.pola.rs); Apache Spark StorageLevel / cache / adaptive query execution (spark.apache.org); PostgreSQL query-planning cost constants (postgresql.org); DuckDB internals / pushdown (duckdb.org).
