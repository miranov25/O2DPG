# DRAW_PATH_BEHAVIOR_MATRIX.md — PHASE_13_76_ADF Stage A (A1 + A2)

*Part I: behavior matrix. Part II: source-path/semantic-owner inventory (the
evidence base for the "source owner" column). One file by architect
preference, 2026-07-19.*

**Status:** Stage-A baseline (Gate-A candidate) + Stage-B fix log (append-only below).
**Governing classification rule:** AD-4/13.76.ADF (symmetry-by-default,
ratified 2026-07-19, verbatim in `docs/ARCHITECT_DECISIONS.md` v1.3.0):
observed asymmetry => Repair unless genuinely semantically inapplicable or
explicitly architect-approved; proposed exceptions stay Unspecified until
approved; no asymmetry freezes as a compatibility contract by existence;
ADF ships contract tests + ownership tracking, never defensive duplication
of dfdraw logic. Rows below carry EXECUTED
evidence only; cells without executed evidence are absent, not guessed
(§0 no-fabrication). The matrix freezes at Gate A; after the
`PHASE_13_76_ADF_GATE_A` tag, changes are append-only amendments.

**Baseline:** `AliasDataFrame.py` MD5 `c73f0c999b4c0503f9b8217653ecf465`
(17,246 lines, post-13.75 HEAD `62ea7137`; phase branch head at seeding time
`bd71fd4d`). dfdraw `drawer.py` MD5 `115085175daeddde7a6ed4fea12d1254`.

## 1. Row schema (Rev2 §8)

```
surface | input form | specification layer | slot/modifier | data state |
classification | error_owner | current behavior | intended behavior |
source owner (code anchor) | test ID | architect ruling / deferral
```

Classifications: `Preserve` (ratified/intended) · `Repair` (defect with
intended-result + fail-before evidence) · `Refused` (deliberately
unsupported) · `Unspecified` (architect decision required; never frozen
undecided) · `BaselineArtifact` (defect in test infrastructure or stale
expectation, not in production code — §8.9 category "stale mapping or
baseline failure").

`error_owner` ∈ {ADF, dfdraw, backend, caller-validation, mixed} (§8.2).
A dfdraw-owned Repair is FILED to dfdraw, never fixed in this phase (R-4).

Enumerated axes (surfaces §8.3, layers §8.4, slots §8.5/8.6, modifiers §8.7,
data states §8.8) are as ratified in Rev2 and not restated here.

## 2. Seeded rows (all executed 2026-07-19, sandbox + alma2 where noted)

### SEED-3 — caller-passed `ax` identity (Risk-1, panel directive)

| Cell | Surface / form | Classification | error_owner | Current behavior (executed) | Intended | Anchor | Test |
|---|---|---|---|---|---|---|---|
| SEED-3.a | `draw` / `ax=` kwarg | **Preserve** | — | renders into caller Axes (1 artist observed) | same | no copy in `draw` span 13927–14300 | `test_seed3_1` (PASS) |
| SEED-3.b | `draw_batch` / top-level `ax=` kwarg | **Preserve** | — | kwargs forwarded verbatim via `plotter.draw_batch(..., **kwargs)`; renders into caller Axes | same | delegation at `:15439–15445` | `test_seed3_2` (PASS) |
| SEED-3.c | `draw_batch` / per-spec `{'ax': ...}` | **Repair** | ADF | `specs = _copy.deepcopy(specs)` replaces Axes with disconnected phantom carrying its own Figure; dfdraw renders into phantom; caller subplot stays empty; **zero diagnostics** | render into caller Axes | `AliasDataFrame.py:15140` | `test_seed3_3` (strict xfail) |
| SEED-3.d | `draw_batch` / `defaults={'ax': ...}` | **Repair** | ADF | same phantom via `defaults = _copy.deepcopy(defaults)` | render into caller Axes | `:15142` | `test_seed3_4` (strict xfail) |
| SEED-3.e1 | `draw_figures` / per-plot OR `defaults` `ax` — semantics | **Refused — RATIFIED (AD-6)**: clean-refusal exception approved | ADF | EXECUTED 2026-07-19: both forms crash `TypeError: ...multiple values for keyword argument 'ax'` at `_draw_single_figure:16126` (composer passes its own grid `ax=` while the deep-copied caller ax rides `**merged`) | AD-6 ratified: reject caller ax with clean ValueError before any figure/axes creation; future figure=/axes= API noted as separate idea | `:16126`; deepcopy `:15532/:15534` | `test_seed3_5`, `test_seed3_6` (PASS, pin crash) |
| SEED-3.e2 | same forms — error quality | **Repair** (AD-6; fix = Stage-B spec validation implementing the ratified refusal) | ADF | bare TypeError is a backend-style collision, not a preparation-time message | clear ADF preparation error naming the surface and the alternative (`draw`/`draw_batch`), OR working symmetric behavior per Q-C | `_draw_single_figure:16126` | same tests (assertion updates with the fix) |

Repair fix location: Stage-B structural-copy normalizer (Rev2 §9), ratified
by architect 2026-07-19 ("If there is a bug, it has to be fixed" — timing
delegated to coder; Stage B chosen because the defective lines are the exact
lines Stage B replaces). Corpus exposure: census shows all 24 existing `ax=`
usages are SEED-3.a form — no current production caller is affected.

### SEED-1 — `test_K1_3_draw_batch_forwards_batch_kwargs` (capability-matrix ❌)

| Cell | Finding (executed) | Classification | error_owner | Anchor |
|---|---|---|---|---|
| SEED-1.a | `bins` from batch-level kwargs never reaches the inner `scatter` call. ADF is NOT the dropper: ADF forwards `**kwargs` verbatim (`:15439–15445`, verified). dfdraw deliberately warn-and-ignores params not used by the plot type — observed live: `UserWarning: Parameter 'bins' is not used by type='scatter' and is ignored (dfdraw Phase 13.57.DF K-3)` (`drawer.py:4869`) | **Repair — DEFERRED** [AD-4 ruling 2026-07-19]: symmetric binning semantics intended; RESOLVED by AD-5 (ratified 2026-07-19): shared bins silently inapplicable to scatter; explicit bins on scatter = clean error naming profile/hist2d/hexbin. Filed to dfdraw; not fixed in 13.76 (R-4) | dfdraw | `drawer.py:4869` (ignore rule), `:7796` (batch inner call) |
| SEED-1.b | Secondary `TypeError: cannot unpack non-iterable NoneType` during the test | **BaselineArtifact** | — (test fixture) | spy fixture returns `None` on inner exception (`test_K1_...py:99–105`), production `fig, ax, stats` unpack at `drawer.py:7796` then fails — artifact does not occur outside the spy |
| SEED-1.c | The K1_3 run's inner call itself raised `ValueError` (capacity class) | folds into SEED-2 | dfdraw | see SEED-2 |

**RULING RECORDED (was Q-A):** architect classified SEED-1.a Repair —
deferred, owner dfdraw (AD-4). Tests: current behavior pinned by
`test_seed1_1_batch_bins_scatter_forwarded_then_warn_ignored` (PASS);
deferred acceptance = `test_K1_3_...` (strict xfail, AD-4-linked).
Capability-matrix ❌ stays as Repair-deferred, NOT cleared as Refused.
Q1 RESOLVED by AD-5; acceptance tests: test_K1_3 (shared-silent) + test_seed1_2 (explicit-error), both strict xfail.

### SEED-2 — `test_K2_3_production_reproducer_mirror` (capability-matrix ❌)

| Cell | Finding (executed) | Classification | error_owner | Anchor |
|---|---|---|---|---|
| SEED-2.a | `adf.draw(...)` with `top_k=4, facet=True, group_by_bins=4` raises `ValueError: vector (6 values) exceeds linestyle cycle capacity (4)` — the channel capacity check fires on the pre-`top_k` cardinality (6) although the caller limited to 4 | **Repair — DEFERRED, RATIFIED** [AD-4 ruling 2026-07-19]: intended = top_k limits the effective channel set BEFORE capacity validation. FILED to dfdraw (R-4: separate dfdraw ownership; not fixed in 13.76) | dfdraw | capacity check `channels.py:318` (step-5 of the documented binding algorithm, `channels.py:179`); exact top_k-vs-capacity ordering trace pending |

**RULING RECORDED (was Q-B):** filing confirmed by architect. Deferred
acceptance = `test_K2_3_production_reproducer_mirror` (strict xfail,
AD-4-linked; fail-before preserved).

### C-1 — caller non-mutation before-state (OBS-1)

| Cell | Surface | Classification | Current mechanism | Test |
|---|---|---|---|---|
| C-1.a | `draw_batch` | **Preserve** (contract) | blanket deepcopy `:15140/:15142` — mechanism will change in Stage B (§9 structural copy), contract must not | `test_c1_1` (PASS) |
| C-1.b | `draw_figures` | **Preserve** (contract) | blanket deepcopy `:15532/:15534` | `test_c1_2` (PASS) |

### C-3 — projection-guard placement (structural fact for the matrix)

All three surfaces are guarded (13.75 contract holds), but placement is
asymmetric: `draw:14268` and `draw_batch:15438` guard in the surface body;
the `draw_figures` guard lives in the helper `_draw_single_figure:16058`.
Classification: **Preserve** (behavior) with the placement recorded so
Stage-B consolidation (one projection finalizer, §11.6) is reviewed against
it.

### O-1 / O-2 — executed oracle rows (all Preserve; Stage B must keep green)

| Cell | Statement (executed 2026-07-19) | Test |
|---|---|---|
| O-1.a-c | identical plot request through `draw` / `draw_batch` / `draw_figures` yields identical statistics (n exact; mean/std/median to 1e-12) for plain, `selection`, and `weights` forms | `TestO1CrossSurfaceStatsEquivalence` (3 params, PASS) |
| O-2.a | all 8 combinations of `draw_lazy` × `draw_keep_materialized` × `draw_clear_after` yield identical statistics for the same alias draw on `draw` and `draw_batch`; policies change lifecycle only, never numbers | `TestO2PolicyIndependence` (8 params, PASS) |
| POLICY-1 | `draw_lazy=False` (default) REQUIRES explicit `materialize_aliases(names=[...])` before drawing a registered alias by name; error today is `ValueError: Cannot evaluate expression 'z'` — documented instance-policy contract (`AliasDataFrame.py:1034`), classification **Preserve** (behavior) with an error-quality note: the message does not mention the policy or the remedy | probed; protocol encoded inside O-2 |
| SWEEP-1 | slots `selection`/`weights`/`group_by`/`facet_by`/`color` all accepted with stats on `draw` AND `draw_batch`; numeric batch≡draw equality additionally proven for `selection` and `weights` | `TestSlotSurfaceSweep` (5 params, PASS) |

### SWEEP-2 / SWEEP-3 — executed rows (increment 6)

| Cell | Statement (executed 2026-07-19/20) | Classification | Test |
|---|---|---|---|
| SWEEP-2.a | vector context (`'[y1,y2]:x'`): `selection_vector` filters PER CHANNEL — proven numerically (ch0 n=84 = y1>0 count, ch1 n=164 = y2>4 count); `weights_vector` accepted; per-channel stats list on `draw` and `draw_batch` | **Preserve** | `test_sweep2_1`, `test_sweep2_2` (PASS) |
| SWEEP-2.c | scalar context: `selection_vector`/`weights_vector` on a plain scalar draw are SILENTLY inert — no filtering (n stays unfiltered), no warning | **Repair candidate (Unspecified)** — explicit-but-inapplicable input silently no-ops, the exact pattern AD-5 ruled must error; goes to the Gate-A ruling batch | `test_sweep2_3` (PASS, pins current) |
| SWEEP-3.a-d | `selection`/`weights`/`group_by`/`color` accepted with stats on `draw_figures`; numeric figures≡draw equality proven for `selection`/`weights` — completes the surface axis of SWEEP-1 | **Preserve** | `TestSweep3FiguresColumn` (4 params, PASS) |
| SWEEP-3.f | `facet_by` in a `draw_figures` panel: DELIBERATE clean refusal with actionable message naming the alternative (`adf.draw(expr, facet_by=...)`) and the tracked dfdraw work (`BUG_dfdraw_20260611_facet_by_ax_ignored`, nested sub-gridspec) | **Repair — deferred (already dfdraw-tracked); interim refusal is Preserve-quality** (AD-4 consistent: asymmetry stays a Repair, the loud clean interim refusal needs no new ruling) | `test_sweep3_facet_by_...` (PASS, pins refusal + message) |

### STATE-1 — data-state equivalence rows (increment 7; all Preserve)

| Cell | Statement (executed 2026-07-20) | Test |
|---|---|---|
| STATE-1.a | lazy-tree draw stats identical to eager (n exact, mean/std/median 1e-12) | `test_state1_1` (PASS) |
| STATE-1.b | lazy 2-file chain identical to eager concat, n=400 exact | `test_state1_2` (PASS) |
| STATE-1.c | O-1 cross-surface oracle holds on the lazy-tree state (draw ≡ draw_batch) | `test_state1_3` (PASS) |
| STATE-1.d | weighted+selected draw on chain identical to eager | `test_state1_4` (PASS) |

Slot-on-lazy coverage note: per-slot draws on lazy tree/chain (incl.
`group_by`, `facet_by`, `color`, vector kwargs) are already exercised by the
13.75 suite's draw section (82 tests, green); STATE-1 adds the cross-state
numeric equality those tests did not assert.

### ENTRY-1 — entry-selection layer rows (increment 8)

| Cell | Statement (executed 2026-07-20) | Classification | Test |
|---|---|---|---|
| ENTRY-1.a/b | `draw`: `entry_begin/entry_end` and `entry_mask` numerically exact (n and mean vs manual slice, 1e-12) | **Preserve** | `test_entry1_1/2` (PASS) |
| ENTRY-1.c | `draw_figures`: entry window exact via named params + `_apply_entry_selection:15682` | **Preserve** | `test_entry1_3` (PASS) |
| ENTRY-1.d | `draw_batch`: NO entry layer — signature lacks the params; kwargs fall through to matplotlib, dying with raw `Polygon.set() ... 'entry_begin'` | **Repair** (AD-4 asymmetry; owner ADF; fix = Stage-B EffectiveDrawSpec entry layer) | crash pin `test_entry1_4` (PASS) + acceptance `test_entry1_5` (strict xfail) |

### A7 — duplication-cost measurements (executed 2026-07-20, sandbox; method: helper call-counters + wall time; re-run on alma2 rides in the Gate-A verification block)

| Scenario | struct-catalog ensures | struct rewrites | note |
|---|---|---|---|
| `draw` (struct expr, lazy) | 4 | 1 | baseline single surface |
| `draw_batch` 2 plots (struct, lazy) | **7** | **5** | repeated passes measured, matching the static 3×-sites finding |
| `draw_figures` 2 plots (struct, lazy) | **7** | **5** | same duplication shape |
| plain-column draws | 2–6 ensures | 0 rewrites | rewrite engages only with struct data (honest scope note) |

Stage-B §14 target: one ensure + one rewrite per call, identical statistics
(O-1..O-2, STATE-1), no >10% wall-time regression on the single-plot path.

## 3. Open cells queue (next characterization increments)

1. ~~SEED-3.e~~ DONE (Q-C resolved by AD-6; acceptance test_seed3_7 strict xfail).
2. ~~Slot × surface sweep (scalar slots × draw/draw_batch)~~ DONE (SWEEP-1);
   remaining: data states (§8.8: lazy tree, lazy chain, subframes).
3. ~~entry layer rows~~ DONE (ENTRY-1).
4. Modifier precedence rows (§8.7).
5. `error_owner` sweep (T-G) once O-7 oracle exists.

---

# Part II — Source-Path and Semantic-Owner Inventory (A2)

**Status:** Stage-A working document, increment 2. Every anchor below was
measured by scripted scan of the canonical bytes (not by reading the
proposal): `AliasDataFrame.py` MD5 `c73f0c999b4c0503f9b8217653ecf465`,
17,246 lines, post-13.75 HEAD `62ea7137` (phase branch head at scan time
`bd71fd4d`). Line numbers are secondary citations; the primary citation is
the marker string + enclosing def (line numbers drift under upstream edits).
AST surface spans: `draw` 13927–14300 · `draw_batch` 15098–15456 ·
`draw_figures` 15462–15934 · `_draw_single_figure` 15989–16171.

## 1. Responsibility × surface anchor map (the duplication, measured)

Marker strings scanned: `deepcopy`, `_ensure_struct_catalog`,
`_autoload_expr_branches`, `_struct_rewrite_draw_slots`,
`merged_defaults = {**`, `_md_dict = {**`, `_ensure_vector_kwargs_aliases`,
`_normalize_vector_compose_kwargs`, `subframe_replacements`,
`_dict_dispatch_columns`, `_assert_struct_projection`,
`from dfextensions.dfdraw import DFDraw`, `materialize_aliases`.

| Responsibility (§5 list) | draw | draw_batch | draw_figures | _draw_single_figure |
|---|---|---|---|---|
| copy/normalize caller input (deepcopy) | **none** | 15140, 15142 | 15532, 15534 | — |
| struct catalog ensure | 14097 | 15146, **15292** | 15538, **15700** | — |
| expr-branch autoload | (inside rewrite path) | 15157 | 15549, 15558 | — |
| struct-ref slot rewrite | 14100 | 15158, **15301, 15432** | 15550, 15559, **15719, 15887** | — |
| defaults/kwargs merge | (kwargs direct) | 15180, **15223, 15320** | 15576 | — |
| union-projection dict (`_md_dict`) | — | 15285 | 15695 | — |
| vector-kwargs alias ensure | 14004 | 15250 | 15666 | — |
| vector-compose normalize | 14008 | 15270 | 15669 | — |
| subframe resolve/join (site clusters) | 14120–14243 (8) | 15279–15419 (8) | 15773–15872 (9) | — |
| dict-dispatch column discovery | 14102 | 15307 | 15728 | — |
| projection guard | 14268 | 15438 | **— (delegated)** | 16058 |
| dfdraw import/delegation | 13989 | 15161 (+call 15439) | 15562 | 16012 |
| materialization | 14078, 14183 | 15241, 15382 | 15655, 15815 | — |

**Bold** = the load-bearing findings:

1. **Intra-surface duplication**, not only cross-surface: `draw_batch`
   executes struct-rewrite **3×** and defaults-merge **3×** within one call
   path; `draw_figures` executes struct-rewrite **4×** and struct-catalog
   ensure **2×**. Stage-B's "one owner per responsibility" (R-1) therefore
   removes both cross-surface AND repeated-pass duplication; the A7 cost
   measurement must count these repeated passes.
2. **`draw` has no caller-input copy step at all** — the SEED-3 asymmetry is
   structural, not incidental: the non-mutation mechanism (13.75 Delta-2)
   was added to the two batch surfaces only.
3. **Guard placement asymmetry**: `draw_figures` relies on
   `_draw_single_figure:16058` for the projection guard; the other surfaces
   guard in the surface body (C-3 satisfied, placement differs — matrix
   row C-3).

## 2. Delegation forms (what dfdraw actually receives today)

| Surface | Delegation | What crosses the seam |
|---|---|---|
| `draw` | `DFDraw` import `:13989`; single plot call | kwargs passed through by identity (no copy) |
| `draw_batch` | `plotter.draw_batch(specs=specs, save_dir=..., defaults=defaults, on_error=..., verbose=..., **kwargs)` `:15439–15445` | the **deep-copied** `specs` and `defaults`; **original** top-level kwargs |
| `draw_figures` | `DFDraw` `:15562`, per-figure composition via `_draw_single_figure` (`:16012`) | deep-copied figure specs; composition owns fig/axes creation |

Inner dfdraw batch loop: `fig, ax, stats = self.draw(expr, type=plot_type,
**merged)` at `drawer.py:7796`; per-item failure recorded in
`results['_errors']` with `on_error` semantics (`:7817–7825`).

## 3. Semantic-owner statements (current de-facto owners)

| Semantic | De-facto owner today | Stage-B intended owner (Rev2 §11) |
|---|---|---|
| precedence (defaults/overrides) | re-derived per surface AND per pass (3× in batch) | EffectiveDrawSpec (one owner) |
| instance policy (`draw_lazy`/`draw_keep_materialized`/`draw_clear_after`, `:1034–:1036`) | read ad-hoc inside each surface | DrawExecutionPolicy |
| dependency discovery | `_dict_dispatch_columns` per surface + slot-scoped autoload loops | DrawDependencyPlan |
| effect execution (loads/materializations/joins/temp columns) | interleaved with discovery in each surface | `execute_draw_plan` (sole owner) |
| name translation (struct/subframe rewrite) | `_struct_rewrite_draw_slots` + subframe replacement loops, repeated per pass | plan/executor, once |
| projection + validation | `_assert_struct_projection` × 3 placements | PreparedDraw finalizer |
| kwarg filtering per plot type | dfdraw (warn-and-ignore, `drawer.py:4869`, 13.57.DF K-3) | unchanged (dfdraw-owned; SEED-1.a) |
| channel capacity refusal | dfdraw `channels.py:318` | unchanged (dfdraw-owned; SEED-2.a) |

## 4. Inputs to A7 (cost measurement plan)

Countable from this inventory: repeated struct-catalog ensures (2× figures,
2× batch), repeated rewrites (3×/4×), repeated defaults merges (3× batch),
duplicated subframe clusters (8–9 sites/surface). A7 measures wall time and
call counts for these on representative corpus draws before Stage B, and the
identical measurement after, feeding the §14 invariants.


---

## Stage-B fix log (append-only; each entry names the flipped cells)

### B1 — 2026-07-20 (first production change of the phase)

| Cell | Was | Now | Mechanism |
|---|---|---|---|
| SEED-3.c, SEED-3.d | Repair (phantom-`ax` via deepcopy; silent empty caller subplot) | **FIXED** — batch per-spec and defaults `ax` render into the caller's Axes; acceptance `test_seed3_3/4` un-xfailed and PASS | `_structural_copy_spec_tree`: containers copied (13.75 P0-4 no-mutation preserved — 82/82 regression green), non-containers (Axes/Figure/arrays/callables) by reference; applied on BOTH batch and figures surfaces (AD-4 symmetry, figures deepcopy had the same latent shape) |
| ENTRY-1.d | Repair (no entry layer on batch; raw matplotlib crash) | **FIXED** — `entry_begin/entry_end/entry_mask` are named params routed through `_apply_entry_selection`, identical semantics to draw/figures; acceptance `test_entry1_5` un-xfailed and PASS (n=100, mean exact) | signature + subset base before projection |
| SEED-3.e1, SEED-3.e2 | Repair (raw TypeError deep in `_draw_single_figure`) | **FIXED** — AD-6 clean ValueError before any figure/axes creation, covering kwargs / defaults / per-figure defaults / per-plot specs, message names the `adf.draw(expr, ax=...)` alternative; acceptance `test_seed3_7` un-xfailed and PASS | early validation block after `_validate_figure_specs` |

Retired with the fixes (pre-declared in their own docstrings): crash pins
`test_seed3_5`, `test_seed3_6`, `test_entry1_4`. Char suite 45 → 42 tests
(41 pass + 1 xfail = dfdraw-owned seed1_2). Remaining open Repairs: AD-5
bins×scatter + SEED-2 top_k (dfdraw-owned, tracker), R-1/R-2 (awaiting
architect ruling), consolidation proper (§14 one-ensure-one-rewrite).


### B2 — 2026-07-22 (consolidation: one preparation pass per draw call) — **SUPERSEDED** (the cache described here was reduced to counters-only on 2026-07-24; one-pass-per-call arrives with B3.2; see the correction entries below)

Plain-language summary: before this change, one call to the multi-plot
drawing functions repeated the same internal preparation work several times —
the "struct catalog" check (which hashes the full branch list and scans
columns) ran 7 times and the expression rewrite ran 5 times for a two-plot
batch. Now each runs once per call: the three drawing entry points open a
small per-call scope, and repeat invocations find the work already done. No
code was removed — every existing call site still runs, later calls are
simply free. Measured after: catalog 1× (was 7×), rewrites 3× (was 5×; three
is correct — two plot dictionaries plus the shared defaults dictionary, one
rewrite each). Numbers proven unchanged by the full oracle battery
(cross-surface, policy-independence, data-state equivalence) plus the 13.75
and 13.66 regression suites. Executable contract: TestB2ConsolidationContract
(4 tests) asserts the exact counts and result identity.


### B3.1 — 2026-07-23 (first pipeline owners, on draw) — **SUPERSEDED IN PART** (public exports reverted, records privatized and purified on 2026-07-24; see the correction entries below)

Plain-language summary: two small classes now own what used to be loose code
at the top of draw(). DrawExecutionPolicy owns the flag resolution (call
argument beats instance setting beats default). EffectiveDrawSpec owns the
normalized plot request: it runs the vector-parameter normalization exactly
once and is the single source for every "which parameters may reference
columns" question — its slot list explicitly contains the four parameters
historical scans kept missing (facet_by, weights, weights_vector,
selection_vector), pinned by test. draw() builds both records and feeds its
existing body from them. The pre-B3 inline code is kept VERBATIM behind the
environment switch ADF_B3_OLD_DRAW_PATH=1 for head-to-head testing and is
removed in step B3.4. Proof: old and new paths produce identical statistics
on four request shapes plus lazy struct data, and the real helper methods
are called the same number of times under both paths — counted by wrapping
the real methods (independent oracle, per the GPT24 review), not by reading
the implementation's own counters. TestB31EffectiveSpecOnDraw, 7 tests.


### B3.1 correction pass — 2026-07-24 (GPT24 [!] + GPT25 [X] pre-commit reviews applied)

Plain-language summary of the four corrections. First: the two new pipeline
records are PRIVATE (_EffectiveDrawSpec, _DrawExecutionPolicy) — the phase
introduces no new public API; the package export added earlier is reverted
and tests import the implementation module directly. Second: building the
specification record is now PURE — it loads nothing, materializes nothing,
mutates nothing (enforced by a test); the effect-producing normalization
stays an explicit draw()-side step until the B3.2 executor becomes its
proper owner. Third: the B2 cache's early-returns were behavior-changing
suppression, not instrumentation — they are REMOVED; only counters remain,
and the three one-pass count tests are converted to strict expected-failure
acceptance tests that will turn green when B3.2 makes one-pass preparation
true by construction. Fourth, smaller items: the old-path environment
switch now requires exactly the value "1"; the subframe pre-scan text is
derived from the single slot list and covers the vector slots (proven by a
source-derived marker test); and spec-construction purity has its own test.
Suite shape: 51 passed / 4 expected-failures, identical under module-style
and package-style imports.


### B3.1 second correction pass — 2026-07-24 (Main-Reviewer reversal synthesis applied)

Plain-language summary. Dead machinery from the removed suppression logic
(the rewritten-identifiers set: built, appended, discarded, never read) is
deleted entirely, and the per-call-scope docstring now states the truth:
counters only, nothing suppressed, every call executes. The pre-scan text
builder is type-safe — only real strings and string elements of lists reach
the join; a numpy array in a vector slot previously crashed with "truth
value of an array is ambiguous" and now cannot (test with array-valued
slots). The draw() path's pre-scan is byte-equivalent to the old path
(scalar slots only) — widening it to vector slots is a behavior change and
lands with its owner in B3.2; a test captures the pre-scan argument under
both paths and asserts equality. GPT27's ordering question is answered by
execution: subframe-qualified references inside vector slots are refused by
the EXISTING tracked guard (BUG_20260701_ADF_subframe_ref_slot_symmetry)
with the identical message under both paths, before any materialization —
pinned by test. The specification record now holds a structural copy of the
style dictionary: rewriting the caller's dict after construction cannot
alter the record (isolation test). The old/new equivalence oracle compares
every returned channel, not just the first. Suite: 54 passed / 4 expected
failures, identical under module-style and package-style imports.


### B3.2 — 2026-07-24 (dependency plan + single side-effect executor, on draw_batch) — **SUPERSEDED IN PART**

> **Superseded 2026-07-25** (B3.2 part-1 panel `[X]`, findings B32P1-7 /
> GPT24 / GPT26). Two claims in the paragraph below are known false and
> are corrected in the part-1 entry at the end of this log: *"every
> preparation effect happens here and nowhere else"* — alias
> materialization, vector-slot materialization, subframe joins,
> temporary columns and cleanup still run in `draw_batch` after the
> executor returns — and *"the auditable answer to which effects ran"*
> — the record as written then could omit real reads and could report a
> struct completion that did not happen. Read this entry as history,
> not as contract.

Plain-language summary. Three new private owners land: the dependency plan
(everything one call needs, computed once from the effective specifications
— pure, no effects), the single side-effect executor (subframe pre-scan
once, union branch load once, catalog check once, autoload and struct
rewrite once per dictionary — every preparation effect happens here and
nowhere else), and the preparation-state record (the auditable answer to
"which effects ran"). draw_batch now builds one plan and calls the executor
once; three superseded preparation blocks are removed from it (the 13.75
P0-2 early defaults pass, the Phase-7.3 per-spec loop, the 13.75 D3
pre-projection pass, and the 13.66 trailing rewrite loop). Measured result:
rewrites are one-per-dictionary BY CONSTRUCTION (acceptance test green);
the executor performs the one owned catalog ensure, and the remaining
catalog invocations are defensive re-checks inside shared helpers
(get_required_branches, _dict_dispatch_columns) that serve standalone
callers — their removal for executor-owned flows is the B3.4 demolition
step, and the catalog-count acceptance tests are re-anchored there,
disclosed, not silently. The pre-scan stays scalar-only with its reason now
recorded in the plan's own docstring: pre-scanning vector slots would
materialize a subframe immediately before the existing guard refuses it
(BUG_20260701_ADF_subframe_ref_slot_symmetry) — an effect-before-refusal
inversion; widening waits for that guard's symmetry fix. New tests: plan
purity, mid-call union coverage on batch (the reviewers' adversarial
requirement), executor-state contract. Suite: 63 passed / 4 expected
failures under both import styles.


### B3.2 correction pass — 2026-07-24 (panel [X], F-1 unanimous P0 fixed) — **SUPERSEDED IN PART**

> **Superseded 2026-07-25**: F-1 and F-4 remain closed exactly as
> described. The effect-ownership and preparation-state statements are
> corrected by the part-1 entry at the end of this log.

F-1: the plot-name expr fallback was written into the raw spec before the
defaults merge, poisoning defaults-supplied expressions (caught by the
pre-existing test_batch_with_defaults). Final form: NO name-write into raw
specs at all — pre-B3.2 it only ever ran inside the struct guard; dfdraw's
own defaults merge resolves absent expr, and the plan applies the fallback
read-only for branch analysis. A non-skipping regression guard mirrors the
catching test on a fixture that runs everywhere. F-3: batch flags migrate
onto _DrawExecutionPolicy. F-4: the plan docstring states the mutation
work-list truth — it holds the B1 structural copies plus ADF-internal
kwargs, never caller-owned dictionaries. F-5: equivalence evidence for the
corrected batch = the full oracle battery (O-1, policy, sweeps, state
equivalence, b2_4) plus the defaults guard; batch has no old-path switch by
design. F-6: the switch-removal target B3.4 is the architect-ratified
bounded plan's own step (one-line re-acknowledgment requested). F-7:
examples/time_series never staged (standing rule). Suite 64 passed /
4 expected failures both import styles.


### B3.2 second correction pass, part 1 of 2 — 2026-07-25 (architect Ruling 2; catalog effect ownership)

**Ruling 2 (2026-07-25):** *"Do not defer the catalog requirement solely on
assertion. Include the adversarial test. A reachable effect must move under
the executor; a proven no-op may be physically removed in B3.4 only after an
explicit recorded ruling."*

**Executed answer: the effect IS reachable — the previous prediction was
wrong, and the reason is worth recording.** `_ensure_struct_catalog` has two
legs. The *detection* leg (`detect_structs`) is guarded by a fingerprint over
the reader's `available_branches`, which is static per file; that leg indeed
cannot re-fire mid-call, and that is what the earlier no-effect prediction
reasoned about. The *D-3 full-structure-completion* leg has no such guard: it
re-tests the frame's columns on every invocation and loads branches whenever a
preceding load left a struct half-populated. Two legs, one guard.

Consequence before this pass, traced end-to-end on a real `draw_batch` call
(`TestB32ExecutorBoundary`): the executor loaded one struct member and
returned; the defensive catalog re-check inside `get_required_branches` /
`_dict_dispatch_columns` then completed the struct — `ensure_struct` plus a
second `ensure_branches`, i.e. real branch I/O **outside** the single effect
owner. It happened precisely when the struct reference lived in a per-spec
dictionary rather than in `defaults`, because the plan's autoload work-list
covered only `[defaults, kwargs]`. Whether a preparation effect stayed inside
the executor therefore depended on which dictionary the user happened to put
the reference in.

**Fix — positional, not list-shaped.** The D-3 leg is extracted to
`_complete_partial_structs()` (one owner, called from both places) and
`_execute_draw_plan` invokes it immediately after its own branch load.
Widening `autoload_dicts` to the per-spec dictionaries would have made
correctness depend on a list's contents again — the same shape as the defect.
The first catalog check stays where it is and is **not** relocated: the very
next executor step resolves required branches through struct-aware expression
analysis, so the catalog must exist before the analysis that decides what to
load. Addition, not movement.

**Cells flipped**

| Cell | Was | Now |
|---|---|---|
| Catalog effect ownership (batch, struct ref in a per-spec dict) | Repair — completion escapes the executor | Preserve — executor-owned, `TestB32ExecutorBoundary::test_b32_5` |
| Preparation-state completeness for struct completion | absent | `_DrawPreparationState.structs_completed`, `test_b32_5b` |
| Catalog *invocation*-count acceptances (`b2_1`, `b2_2b`) | B3.4, deferred on assertion | B3.4, deferred **on executed evidence** (`test_b32_6` + `test_b32_7`) |
| Rewrite-count assertion | fixture-shaped literal `3` / `2` | structural, `_expected_rewrite_count` (Ruling 3) |

**Why the deferral of physical removal is now safe, and how that safety is
kept honest.** `test_b32_6` shows the residual re-checks change nothing after
a real call. On its own that would be a weak guard: it would keep passing if
the D-3 leg were deleted or silenced, and B3.4's removal argument would
expire without anyone noticing. `test_b32_7` is its deliberate counterpart —
it constructs the partial-struct state by hand and asserts the same residual
path *does* act. The pair states the real property: the residual calls are
inert **because the executor completed the structs first**, not because the
code path is incapable.

**Scope, stated honestly.** This pass does not yet make the executor the sole
owner of *every* preparation effect. Alias materialization, vector-slot alias
materialization, subframe joins with their temporary columns, and the
post-draw cleanup are still performed by `draw_batch` after the executor
returns; the executor docstring now says so rather than claiming otherwise,
and `test_b32_5c` is the strict-xfail fail-before evidence that flips when
that migration lands in part 2. The spec-normalisation shims (plot-type,
`vector_compose`) and the entry-window selection are deliberately **not**
migration targets — they belong to `_EffectiveDrawSpec` (§11.1) and the
projection stage (§11.6) respectively.

Suite: focused 74 passed / 5 expected failures (was 64 / 4 — ten new tests,
one new fail-before marker). Battery 222 passed / 4 skipped / 7 expected
failures. Full sandbox sweep: failure identities byte-identical before and
after (91 before, 91 after, zero new, zero accidentally fixed). Mutation-
verified: removing the one-line fix fails `test_b32_5` on both spec-side
cases and `test_b32_5b`.


### B3.2 part-1 correction — 2026-07-25 (panel `[X]`; preparation-state truthfulness)

Eight reviews on `reviewer_20260725_122628.zip`. Six landed `[!]`/`[OK]`;
GPT24, GPT25 and GPT26 landed `[X]` — and the Main Reviewer overrode the
numeric majority, correctly. The three `[X]` seats did not merely flag the
eager path as untested, which is what every Sonnet-family seat and the coder's
own review request had said. They **ran it**, independently, three times, with
different fixtures, and got matching falsifying results. The coder had named
this exact risk in §4 of the review request as "plausible, not covered by a
test"; the honest description was "affirmatively false when executed". That is
the same failure this phase exists to teach, one level up: reasoning about a
path is not evidence about a path.

**Three ways an intent-derived record lied** (each now reproduced by the coder
independently before accepting the finding):

| ID | Defect | Observed |
|---|---|---|
| B32P1-3 | Eager frames: `ensure_struct()` is a silent no-op without a reader, but the struct name was appended to `structs_completed` regardless | columns before `['x','a__S']`, after `['x','a__S']`, `structs_completed == ('S',)` |
| B32P1-1 | Reads performed by full-structure completion were absent from `branches_loaded`, which was written from the union-load intent and never revisited | recorded `['dedxTPC/dEdxMaxTPC','mult']`, actual reader state also held `dEdxMaxIROC`, `dEdxTotTPC` |
| B32P1-2 | A struct already partial on entry is completed by the **initial** catalog call, before the union-load line runs — leaving no trace in either field | 2 struct columns appeared; `structs_completed == ()`, `branches_loaded == ('mult','tgl')` |

**Corrections.** Every read/column field of `_DrawPreparationState` is now a
**measured before/after delta** taken at each stage boundary
(`_observe_prep_effects`), never the set of branches the executor asked for.
Intent is kept, labelled, and kept apart: `requested_reads`. Effects are
attributed by stage — `reads_by_catalog`, `reads_by_union_load`,
`reads_by_completion`, `reads_by_autoload` — and the total is measured across
the whole call rather than summed from the stages, so an unattributed effect
still shows up in the reconciliation. `_complete_partial_structs()` re-reads
the frame and records a struct **only when every member is verifiably
present**: attempt and outcome are different events, and only the outcome is
reported.

**Eager partial structs are TOLERATED, not completed.** That is pre-existing
behaviour — the D-3 leg has always run on eager frames and has always done
nothing there — preserved deliberately rather than changed inside a correction
pass, and now pinned by `test_b32_10`. Whether a registered-but-partial struct
on an eager frame should instead be **refused loudly**, as D-3's own error text
implies, is an **open cell requiring an architect ruling**: Preserve or Repair.
It is out of scope for this correction either way.

**The GPT25/GPT26 double-completion dispute is resolved by direct trace, and
both were right at different layers.** GPT25 observed that the second
`ensure_struct` issues a second `ensure_branches` request — confirmed. GPT26
concluded no duplicate branch I/O occurs — also confirmed: the reader performs
exactly two real reads for the whole call. But GPT26's stated mechanism is
wrong; the filtering happens inside `ensure_branches`, not inside
`ensure_struct`'s missing-member check. So the redundancy is **invocation
overhead, not duplicated I/O**, which keeps it a B3.4 demolition item rather
than part-2 scope. `test_b32_11` pins both halves so the answer cannot drift,
and fails loudly if a second real read ever appears.

**Cells flipped**

| Cell | Was | Now |
|---|---|---|
| `structs_completed` truthfulness | can be affirmatively false (eager) | verified-then-recorded, `test_b32_10` |
| `branches_loaded` completeness | union-load intent only | measured total, reconciled against the reader, `test_b32_8` |
| Completion during the initial catalog call | untraced | `reads_by_catalog` + `structs_completed`, `test_b32_9` |
| Effect attribution by stage | absent | four stage fields, sum reconciled against the measured total |
| C5 fail-before tracer | 3 method names | 7 — adds `materialize_alias`, `_ensure_vector_kwargs_aliases`, `_prepare_subframe_joins`, `dematerialize` |
| `test_b32_7` non-vacuity | bare `before != after` | exact expected column and read deltas, plus an explicit absent-before precondition |
| Rewrite-count assertions | fixture-shaped literals | structural (Ruling 3) |
| Earlier B3.2 matrix entries | unmarked full-ownership claims | `SUPERSEDED IN PART`, with the false sentences quoted |

**Still open, deliberately.** The executor is not yet the sole owner of every
preparation effect; `test_b32_5c` remains the strict fail-before marker and its
tracer now watches all seven relevant methods, so it can no longer XPASS early.
Plan purity (`test_b32_1` never calls `required_branches(adf)`) is unchanged
and carries into part 2. `draw()` and `draw_figures()` have **not** been
examined for the same completion escape — the coder named this in the review
request and no reviewer closed it; it is the first thing part 2 must check
before the matrix claims a general closure.

Suite: focused 74 → **78 passed / 5 expected failures** (four new reconciliation
tests). Mutation-verified per correction: reverting verify-then-record fails
`test_b32_10`; reverting the measured total fails `test_b32_8`, `test_b32_9`,
`test_b32_5b`; reverting the initial-catalog attribution fails `test_b32_9`.


### B3.2 part-1 round-2 correction — 2026-07-25 (panel `[X]`; second pass on the same record)

Nine reviews. Sonet29's `[OK]` was issued from four seats before the GPT
reviews landed and self-labelled provisional, asking to be superseded rather
than reconciled once they arrived. They arrived: GPT24 `[X]`, GPT26 `[X]`,
GPT25 `[!]`, GPT27 `[!]`. Two independent executed reproductions of the same
P0 → `[X]` stands and the `[OK]` is superseded.

**F1 (P0) — the same defect, a second representation.** Round 1 closed
"initial-catalog completion is untraced" for structs already registered with
internal member columns. It stayed open for the D4 shape: a struct arriving in
PHYSICAL form (`dedxTPC/dEdxMaxTPC`), not yet registered. The catalog call then
registers it, renames the column and loads the siblings all in one stage, and
the membership snapshot taken before that call could not see a struct that did
not yet exist. Reads were recorded; the completion that caused them was not.
Reproduced by GPT24 and GPT26 independently, then by the coder before
acceptance.

Fixed by measuring the ENTRY column set with the definitions known AFTER
registration (`_struct_membership_in`), and by counting a member present under
either its internal or its physical name. The naive two-snapshot fix would have
reported a *false* completion for a struct that was already whole in physical
form and merely got registered and renamed — `test_b32_13` is the control that
pins that, and it matters as much as the finding.

**F2 (P1) — a behaviour change documented as a preservation.** GPT27 traced
what four Sonnet-family seats and the Main Reviewer accepted at face value: the
claim that "the D-3 leg has always run on eager frames and has always done
nothing there" is false. `_ensure_struct_catalog()` returns at its **second
statement** when `_lazy_reader is None`, so that leg never reached an eager
frame at all. The round-1 unconditional call was therefore a NEW eager
invocation described as preserving prior behaviour — the exact failure this
phase keeps paying for, this time in the correction pass whose subject was not
asserting things. The call is now gated to lazy frames, which is also the
useful shape: an eager frame has no reader to complete a struct from.

**F3 (P1) — misattributed pre-scan reads.** The subframe pre-scan loads the
index columns a lazy subframe needs to join. Those reads fell inside the
union-load observation window and were reported as union-load reads. Own
boundary, own field (`reads_by_prescan`). The fixture is the point here: the
first version of the stage-disjointness assertion passed with the boundary
deliberately broken, because the struct fixture's pre-scan loads nothing. A new
two-tree fixture (`_write_tree_with_subframe`) exercises it — measured
`reads_by_prescan=('sec',)` against `requested_reads=('x',)`.

**Architect ruling, 2026-07-25 — partial branch sets will be supported.** This
settles the eager question and rules OUT the Repair/refuse-loudly option that
GPT24, GPT25 and three Sonnet seats recommended. Refusing to work with a
registered-but-incomplete struct would block a capability the architect has
stated is coming, and Rev 2 §25 already defers member-exact loading to its own
decision. Measured facts behind the ruling: a present member draws correctly;
an absent member raises the 13.75 C3 projection guard with logical, internal
and physical names; `eval()` on an absent member raises `NameError`. Nobody
computes on data that is not there.

Consequences for the record, deliberately chosen:
* incompleteness is a **neutral fact** — `struct_members_present` records
  which members are present per struct, with **no warning and no refusal**. A
  warning would become noise the moment partial working is normal, and a field
  named for a fault would not survive the future mode.
* the automatic completion of a partial struct on a **lazy** frame is pinned as
  **today's policy, not an eternal invariant**, with the tag written into
  `test_b32_8` so the future partial-loading mode finds it instead of hitting a
  test wall.

**Mechanical items closed:** `test_b32_11` now asserts exact read batches and
no repeated branch name across them, not just a call count (R2-P2-1); stage
attribution asserts **pairwise disjointness** as well as summing to the total
(R2-P2-2); the state docstring names `requested_reads` as the one deliberate
intent field rather than claiming everything is observational (F5).

**One removal worth disclosing.** `_structs_completed_between` carried a `not
_present_before` clause for "a struct loaded from nothing is not a completion".
No reachable path triggers it — the catalog stage can only take a struct from
partial to complete — and a mutation test could not tell whether it was doing
anything. It was removed rather than left as untestable defensive code, and the
contract is now pinned behaviourally by `test_b32_17`, which is what catches it
if the measurement ever moves.

**Still open, unchanged:** `draw()` / `draw_figures()` parity for the same
completion escape (named since round 1, closed by nobody, first task of part 2);
plan purity (`test_b32_1` still never calls `required_branches(adf)`); the
remaining effect migration behind `test_b32_5c`; cache effects and cleanup
candidates still unrepresented against Rev 2 §11.5. GPT24's R2-P1-2 stands: the
seven-method tracer still cannot see a direct `df_for_plot[flat_ref] = ...`
temp-column write, so external frame observation must be added in part 2 before
that marker is trusted as the migration-complete signal.

Suite: focused 78 → **84 passed / 5 expected failures**. Battery **260 passed**
/ 4 skipped / 7 expected failures. Full sandbox sweep: failure identities
byte-identical to the pre-part-1 baseline (91 before, 91 after). Mutation-
verified: reverting the F1 measurement fails `test_b32_9` and `test_b32_12`;
reverting the eager gate fails `test_b32_14`; reverting the pre-scan boundary
fails `test_b32_16`.


### B3.2 part-1 round-3 correction — 2026-07-25 (reader-graph observation)

Ten seats reviewed `reviewer_20260725_135249.zip`. The Main Reviewer overturned
its own interim `[OK]` on the full panel — worth recording as method, not just
outcome: the interim verdict was issued from four seats before the remaining
GPT reviews arrived, and it was wrong.

**Two of that synthesis's findings were already closed** in the bytes that
followed it (`aff20f48`, packet `...150027`), which the panel had not yet seen:
P0-PhysicalPreload (the `_partial_struct_names()` function it quotes no longer
exists) and P1-EagerHistory. Re-litigating closed findings is the cost of a
review round landing on superseded bytes; the round-3 note exists to stop that
repeating.

**P0-ReaderGraph — the one genuinely new blocker, and the deepest of the three
rounds.** GPT31 was the only seat across ten reviews and three rounds to build
a *subframe* scenario rather than confirm the fix against the scenarios it was
designed for. `_observe_prep_effects()` — the single point every measured field
derives from — looked at `self._lazy_reader` and `self.df.columns` and nothing
else. A draw slot referencing a lazy subframe makes the executor's pre-scan
materialize that subframe, reading branches through the SUBFRAME'S OWN reader
and building the subframe's own frame. None of it was visible, by construction.
Reproduced here before acceptance: with `SectorCalib.corr:x`, the string
`corr` appeared in **no field of the record**.

This is the third consecutive round of the same class of error, each one level
further out: first the claim outran the code, then the record described intent
instead of effects, now the observation was narrower than the thing it claimed
to observe. Naming the pattern rather than just fixing the instance: *a
measurement is only as honest as its scope, and scope is exactly what a test
written against the same assumption cannot check.*

Fixed by walking the whole graph — this frame's reader, every registered lazy
subframe reader, and every materialized subframe's frame, recursively, with a
visited set. Names are qualified by owner (`SectorCalib::corr`) so a branch of
the same name in two readers cannot collapse into one entry and under-report.
Main-frame names stay unqualified, so every earlier reconciliation test keeps
its exact meaning. Permanent tests `test_b32_18` (subframe effects recorded)
and `test_b32_19` (graph reads still attributed to exactly one stage, stages
still pairwise disjoint and summing to the measured total).

**P2-DoubleAssign — Sonet25 was right and the coder's correction of it was
wrong.** It was argued that the first of the two `structs_completed`
assignments was load-bearing on the eager path. It is not:
`_ensure_struct_catalog()` returns immediately without a reader, so `_by_catalog`
is necessarily empty on an eager frame. A mutation test settled it — restoring
the two-assignment form changed no test outcome. Now a single expression, for
readability rather than to preserve a value, and the comment says so.

**Peak-RSS, partial isolation evidence.** The sandbox sweep showed 90 failure
identities against the 91-identity baseline — `test_peak_rss_dict_below_full_frame`
passed. Rather than report a one-off improvement, it was isolated: the test
passes **8/8 standalone** (5 runs on these bytes, 3 on the pre-part-1 baseline)
and fails only under 12-way parallel load, on both trees. That is evidence the
failure is load-related and not attributable to this phase. It is not the alma2
isolation run, which is still owed; it is the first actual measurement anyone
has attached to that claim in six rounds of asserting it.

**Cells flipped**

| Cell | Was | Now |
|---|---|---|
| Reader-graph observation | main reader + main frame only | whole graph, owner-qualified, `test_b32_18` |
| Stage attribution under a subframe reference | untested | reconciles and stays disjoint, `test_b32_19` |
| `structs_completed` assignment | two assignments, one argued live | one expression, argument retracted |
| Peak-RSS attribution | asserted, never measured | measured 8/8 in isolation, load-related |

**Still open, unchanged:** `draw()` / `draw_figures()` parity (named since round
1, closed by nobody, first task of part 2); plan purity; `test_b32_5c`'s tracer
cannot see a direct `df_for_plot[flat_ref] = ...` write (GPT24 R2-P1-2, GPT27,
GPT30, GPT31); cache effects and cleanup candidates unrepresented against
Rev 2 §11.5.

Suite: focused 84 → **86 passed / 5 expected failures**. Battery **262 passed**
/ 4 skipped / 7 expected failures. Mutation-verified: reverting the graph walk
fails `test_b32_18` and `test_b32_19`.


### B3.2 part-1 round-4 correction — 2026-07-25 (one falsehood fixed, two omissions disclosed)

Nine seats. **Seven approved; two did not, and the two were right.** GPT26 and
GPT30 each built a different input shape nobody had tried and each found a real
gap in the mechanism that had just closed the previous round's two P0s. The
Main Reviewer overturned its own `[OK]` for the third round running and said so
plainly. Both mechanisms were reproduced by the coder before acceptance.

**The standing bar, adopted this round.** Four rounds have shown that someone
can always construct an input shape the suite does not cover, so "the record
covers every input" does not terminate. "The record never states something
untrue" does, and it is checkable. That line separates this round's two
findings cleanly and is now the blocking criterion for `_DrawPreparationState`:

| | Meaning | Disposition |
|---|---|---|
| **Falsehood** | the record asserts something that did not happen | **fix — blocking** |
| **Omission** | the record does not cover a shape; nothing untrue is said | **disclose in the documented scope — not blocking** |

**Fixed — P0-ChainSyntheticRead (GPT30), a falsehood.** `LazyChainReader` adds
a synthetic `__file_idx__` bookkeeping column to `loaded_branches` while
deliberately excluding it from `available_branches`; its own docstring says so
in two places. The graph walk copied loaded names unfiltered, so a name that was
never read from a file appeared in `branches_loaded` and `reads_by_union_load`
— telling a consumer that I/O occurred which did not. Reads are now filtered
through each reader's own `available_branches`. The column still appears in
`columns_created`, which is accurate: it is a real column. Readers that do not
expose `available_branches` are left unfiltered — recording a superset beats
silently dropping real reads. `test_b32_20`; `test_b32_21` guards against
over-filtering. Mutation-verified.

**Disclosed — P0-SharedNodeAlias (GPT26), an omission.** The cycle guard is a
single identity set, so a child object registered under two subframe names is
walked once and the second owner path's effects are omitted. Nothing false is
recorded; the first path is correct. GPT26 framed this as needing a ruling and
that framing is accepted: either forbid the registration with a clear error, or
support it properly by separating physical-reader identity from logical
owner-path provenance — a design change, not a correction. **Open architect
ruling.** Pinned by `test_b32_23`, written so that closing the limit fails the
test rather than leaving a stale docstring.

**Disclosed — struct inside a subframe, an omission.** Raised independently as
a hypothesis by three seats (Sonet25, Sonet27, Fabble5_7) and executed by the
coder rather than left unconfirmed: **it holds.** Completion is `self`-scoped
while observation is now graph-scoped, so such a struct is left partial. The
record does not claim otherwise — `structs_completed` is correctly empty.
Whether the executor should reach into subframe registries is a scope ruling.
**Open architect ruling.** Pinned by `test_b32_22`.

Both limits are now written into `_observe_prep_effects`'s docstring as an
explicit covered / not-covered list, so the scope is stated where the code is,
not only in this log.

**Worth recording about the method.** GPT25 and GPT31 both went looking for
more — nested subframes, repeated branch names across readers, a deliberate
graph cycle — and found the mechanism sound. That is useful negative evidence,
not a miss: it narrows where the remaining risk lives. What no round has
produced is a way to know in advance which untried shape matters, which is
precisely why the bar moved from coverage to truthfulness.

Suite: focused 86 → **90 passed / 5 expected failures**. Battery **266 passed**
/ 4 skipped / 7 expected failures. Full sandbox sweep: 90 identities against the
91-identity baseline, the difference being `test_peak_rss_dict_below_full_frame`
which passes 8/8 standalone on both trees and fails only under 12-way parallel
load — load-related, not attributable. Mutation-verified: reverting the
`available_branches` filter fails `test_b32_20`.


### B3.2 part 2 — 2026-07-25 (the executor becomes the sole owner on draw_batch)

Part 1 closed the effect-accounting layer over four review rounds. Part 2 does
what B3.2 was actually for: move the remaining preparation effects under the
executor, so "one owner" is a property of the code rather than a sentence in a
docstring.

**Architect ruling — the two-phase (in fact three-phase) executor.** Two of the
four remaining effects cannot live in a pre-draw executor, and that is physics:

* the single-level subframe join writes its flattened column into the
  **reduced** frame, which does not exist until projection. Moving it earlier
  means growing `self.df`, and the D-ADF-DICT contract says in as many words
  that the big frame is never copied or grown — on a ten-million-row TPC frame
  that is not a cosmetic difference;
* **cleanup** runs after dfdraw has rendered. There is no "before the draw"
  that contains it.

Presented as three options — two-phase executor, narrow the claim to three
separate owners, or force everything pre-draw and grow the big frame — the
architect chose the two-phase executor (2026-07-25). Narrowing the claim would
have re-created the very arrangement this phase exists to remove; forcing
everything pre-draw would have traded a documentation problem for a memory
problem.

**Migrated into the preparation phase:** alias materialization, vector-slot
alias materialization, subframe joins. **Named phases added:**
`_execute_draw_projection_effects` (reduced-frame temporary columns) and
`_execute_draw_cleanup` (dematerialization after the render). All three report
into one `_DrawPreparationState`, now 20 fields.

**Measured result — zero escapes.** Effect trace on three call shapes:

```
alias + clear_after   escapes = NONE
struct in spec        escapes = NONE
vector slots          escapes = NONE
```

**`test_b32_5c` flipped to XPASS(strict) and its marker was removed.** That is
the fail-before mechanism working as designed rather than as a formality: it was
written while alias materialization still escaped, and it went off the moment
the migration landed instead of waiting for someone to remember. Its tracer now
counts all three phases as "inside the executor", which is what the ruling
makes true.

**GPT24's R2-P1-2 closed, by measurement rather than by watching.** A tracer
that wraps methods can never observe `df_for_plot[flat_ref] = ...` — the write
has no method to wrap. The projection phase therefore measures the reduced
frame before and after, and records what it gained as `temporary_columns`,
asserted distinct from `columns_created` and asserted absent from `self.df`.

**P1-PlanPurity closed by deletion, not by a test.** Open since round 1 and
flagged by four GPT seats across four rounds: `_DrawDependencyPlan` documented
itself as PURE while carrying `required_branches(adf)`, which reached through
`get_required_branches` into the struct catalog. The reviewers proposed a test
that calls the method and asserts no effect. Removing the method is stronger —
purity then holds by construction, and cannot rot when someone forgets the
assertion. Resolution moved to `_resolve_required_branches` on the executor.
`test_b32_24` pins the structural property: the plan exposes no method that
needs an ADF to act on.

**Falsehood found by the coder and fixed.** While probing `draw()` /
`draw_figures()` parity, the preparation record turned out to survive a call on
an unmigrated surface: `draw_batch` then `draw()` left the batch call's reads
in place while `tgl` had in fact been read. Under the standing bar that is a
falsehood, not a coverage gap — an absent record is honest, a stale one is not.
All three public surfaces now clear the record on entry, so an unmigrated
surface leaves `None`.

**Rev 2 §11.5 completeness.** `cache_effects` is the last field group, recorded
as a measured transition rather than as "a cache was touched": a stable catalog
reports nothing, an unset one reports the fingerprint transition, and the
subframe join-index cache reports growth. A field that fires on every call
would mean nothing.

**Cells flipped**

| Cell | Was | Now |
|---|---|---|
| Sole effect ownership on `draw_batch` | 4 effects outside the executor | zero escapes, three named phases |
| Reduced-frame temporary columns | unobservable by the tracer | measured, `temporary_columns` |
| Post-render cleanup | surface-local step | executor phase, candidates from the record |
| Plan purity | documented, untested, effectful method present | true by construction |
| Preparation record after an unmigrated surface | stale, readable | `None` |
| Rev 2 §11.5 field groups | writes and cleanup missing | complete, 20 fields |

**Still open, and stated rather than implied.** `draw()` and `draw_figures()`
are unmigrated B3.3 scope, and this was *measured*, not assumed — both still
complete a partial struct through a residual catalog re-check after their own
load, exactly as `draw_batch` did before part 1. The two disclosed limits from
round 4 are unchanged and still await rulings: the same child frame registered
under two subframe names, and structs living inside a subframe. The alma2
peak-RSS isolation run is still owed.

Suite: focused **97 passed / 4 expected failures** (one fewer expected failure
than part 1 — the removed marker). Battery **273 passed** / 4 skipped / 6
expected failures. Full sandbox sweep: **91 identities, identical to the
pre-phase baseline**. Mutation-verified: sourcing cleanup candidates from
anywhere but the record fails `test_b32_26`/`27`; a silent projection phase
fails `test_b32_28`; restoring the plan's effectful method fails `test_b32_24`.

---

## Fix log — B3.2 part 2, correction round (2026-07-25)

**Panel verdict on the part-2 bytes was `[X]`.** Five GPT seats found the same
thing independently, and they were right. The entry above says "zero escapes,
three named phases". The code did not support it: the 118-line subframe
resolution block was still inline in `draw_batch`, and
`_execute_draw_projection_effects` was called *afterwards* to diff two column
lists. That is a good oracle. An oracle is not an owner. Everything in the
previous section about the projection phase is **SUPERSEDED IN PART** — the
measurements it describes were real, the ownership claim attached to them was
not.

**P0-ProjectionOwnership — the phase now executes.** The join, the child-frame
alias materialization, the multi-level `_prepare_subframe_joins` call and the
spec rewrite all run *inside* `_execute_draw_projection_effects`, which returns
`(df_for_plot, subframe_replacements)`. `draw_batch` is a caller, not a
co-owner holding half the state. `test_b32_30` is the mutation form of the
claim: replace the phase with a pass-through and the flattened column can no
longer appear. The previous tests could not have caught this — they would have
passed just as happily against the inline arrangement, which is how the
arrangement survived a round of review.

**P0-SubframeCleanupRegression.** Cleanup called `self.dematerialize()`, which
can only reach *this* frame. An alias the projection phase materialized on a
CHILD frame was therefore listed as a cleanup candidate and then quietly
survived the call — a record that lists a candidate never dropped is not
incomplete, it is false. Cleanup now goes through `_dematerialize_qualified`,
which reaches every frame in the graph, and `aliases_dropped` is measured after
the attempt so an undroppable candidate shows up as the difference between the
two fields rather than as a claim (`test_b32_32`, `test_b32_33`).

Related: `aliases_materialized` was being measured against the set of
*declared* aliases, which materialization does not change. Measured against the
set of aliases that currently have a backing column instead
(`_materialized_frame_aliases`).

**P0-NestedPersistentMislabeled.** Classification is now by **where the write
landed**, not by which phase observed it. A multi-level reference goes through
`_prepare_subframe_joins`, which writes a PERSISTENT column onto `self.df`; the
previous version measured the reduced frame alone, saw the column appear there,
and filed it under `temporary_columns` — whose documented meaning is "discarded
when the call returns". `test_b32_34` asserts the persistent case,
`test_b32_35` the control.

**P2-DeadLocal.** `already_materialized` at the old `:16038` was deleted. Once
cleanup became a phase sourcing its bracket from the record, nothing read it; a
retained-for-symmetry local is a claim that the surface still participates in
the alias lifecycle, which is what this increment removed.

### Architect rulings implemented this round

| # | Ruling | Implementation |
|---|---|---|
| **D1** | Same child registered under two subframe names is legal and must be supported | Cycle guard moved from one global identity set to the **ancestor path**, so a child reachable as `A` and `B` is walked under both prefixes. A genuine cycle still terminates. New field `frame_aliases`. `test_b32_23`, `test_b32_23b` |
| **D2** | Full functionality within child tables | `_complete_partial_structs` is graph-scoped, gated **per node** on that node's own lazy reader; `_struct_membership_graph` reports membership under qualified names on the same scope. `test_b32_22`, `test_b32_22b` |
| **D3** | Cleanup-on-render-failure must be an option | New `clear_after_on_error=False` on `draw_batch`. Default preserves pre-B3.2 behaviour exactly (a raised render leaves the columns in place — they are the evidence someone debugging wants). New field `cleanup_outcome` records which of the five outcomes occurred, so the previously ambiguous "no candidates" state is readable. `test_b32_36`–`38` |

**Both round-4 disclosures are now CLOSED** — by ruling, not by the coder
deciding a scope question mid-correction. The `_observe_prep_effects` docstring
was rewritten accordingly; it now discloses one remaining gap instead of two: a
subframe registry entry that is not an AliasDataFrame has no frame to observe,
so its effects are omitted and never misreported.

**One walk, one guard.** `_observe_prep_effects` used to carry its own copy of
the graph traversal. Both it and the new phases now use `_iter_frame_graph`.
Two walks with two guards is how the record and the cleanup bracket came to
disagree about which frames existed in the first place.

**Newly disclosed, pre-existing, NOT fixed here.** A *lazy* subframe referenced
only through an ALIAS (`Sub.some_alias`) never gets its join index columns
pre-scanned, so the join fails before the projection phase is reached. The
child-alias tests therefore use eager frames. This is a gap in the lazy
pre-scan, not in this increment; repairing it inside a correction pass would
mix two changes and the evidence would no longer say which one it covers.

**Cells flipped (correcting the previous section)**

| Cell | Previous entry claimed | Actually now |
|---|---|---|
| Projection phase | "one owner, three named phases" | true — the phase executes; `test_b32_30` fails if the work moves back out |
| Cleanup scope | this frame | whole graph, owner-qualified |
| Multi-level join column | reported temporary | reported persistent (`columns_created`) |
| Struct completion scope | this frame (disclosed limit) | whole graph, per-node reader gate (D2) |
| Aliased child registration | walked once (disclosed limit) | walked under both owner paths (D1) |
| Cleanup on render failure | fixed: never runs | caller's choice, recorded either way (D3) |

---

## Fix log — B3.2 part 2, correction round 2 (2026-07-27)

**Panel verdict `[X]`, 5 of 8 seats, and the five were right.** GPT25, GPT26,
GPT27, GPT30 and GPT31 each independently executed `draw_batch` with a
subframe reference *and* an entry selection, and each got the same failure.
Three Sonnet seats approved; the synthesis (Sonet29) overturned its own `[OK]`
under the Verdict-from-Convergence rule. Every finding below was reproduced by
the coder before it was accepted.

**P0-EntryProjection — the composition nobody had built.** `entry_begin` /
`entry_end` / `entry_mask` became a `draw_batch` contract in B1. Subframe
projection is what part 2 consolidated. The join is computed over the WHOLE
parent frame (`_compute_join_indices` is defined that way), so its full-length
result could not be assigned into the entry-selected reduced frame. The
mismatch raised, the broad subframe `except` downgraded it to a warning, the
dotted reference was never rewritten, and dfdraw then failed with a NameError
about an undefined subframe name — which reads like a user typo. All three
entry forms were affected.

Fixed by carrying the selected parent-row POSITIONS
(`_entry_selection_positions`) into the projection phase and slicing the join
by them. Positional, not label-reindexed: nothing forbids duplicate index
labels, and label alignment on such a frame silently fans out or picks the
wrong row (`test_b32_44` is that case). The multi-level route's persistent
copy became positional for the same reason.

**Verified against a hand-computed join**, not against absence of an
exception: `test_b32_39`–`44` each compare the flattened values with an
independent join restricted to the same rows. "It didn't throw" would have
passed against a projection producing the wrong numbers.

**Disclosure: this P0 predates the correction.** The same assignment shape
existed in the rejected inline arrangement — verified by running the failing
call against `f125375f`, where it fails identically. It blocks B3.2 anyway,
because the broken composition now lives inside the code that claims sole
ownership, and closing the increment would freeze a known hole in the new
owner.

**P0-DefaultsRewrite (GPT25).** A subframe reference supplied through
`defaults` or a top-level kwarg — a supported way to give one expression to a
whole batch — was joined and got its flattened column, but the rewrite loop
walked only the per-plot spec dicts. The rewrite set is now the same set the
reference text was collected FROM. The caller's own dictionary is still never
mutated (B1's structural copy); `test_b32_45` pins both halves.

**P1-RepeatedCallMisclassify (GPT25).** Classification now asks where a column
LIVES, not what this call wrote. On a second call the persistent multi-level
column already exists, so it was absent from this call's write diff and was
filed as a reduced-frame temporary — a column documented as "discarded when
the call returns" that in fact sits on `self.df`.

**P1-CleanupOutcomeWrong (GPT27) and P1-FailureStateTruthfulness (GPT27).**
One `else` covered two situations, so a render failure with `clear_after=False`
reported `skipped_render_failed` when cleanup had never been requested. And the
bracket covered only the render, so a failure in preparation or projection left
an alias materialized with an outcome of `not_requested` and no sign anything
had gone wrong. There are now three brackets — preparation, projection, render
— feeding one `_record_draw_failure`, a new `failure_phase` field, and outcomes
renamed to `skipped_after_failure` / `ran_after_failure` since they are no
longer render-specific. `_execute_draw_plan` publishes its record the moment it
exists, so a half-finished preparation is still readable.

**P1-BaseExceptionTooBroad (GPT30).** `except Exception`. A `KeyboardInterrupt`
is the user stopping the session, not a failed plot; treating it as one deleted
their columns on the way out (`test_b32_51`).

### Architect rulings this round

| # | Ruling | Implementation |
|---|---|---|
| **D1 (Option 3)** | A failed subframe resolution raises by default, with an escape hatch | `on_subframe_error='raise'` (default) / `'warn'`. The old warn-and-continue is exactly what hid P0-EntryProjection for the whole of part 2: the warning fired in every run and nothing watched for it. `test_b32_48`, `test_b32_49` |
| **D2** | **REVERSES the 2026-07-25 D1 ruling.** Registering the same AliasDataFrame *object* under two subframe names is now FORBIDDEN | `_refuse_duplicate_frame_registration`. One mutable object under two logical names shares aliases, materialization, caches and cleanup, and forces one physical effect to be reported under two identities — which is exactly the `structs_completed` / `struct_members_present` contradiction GPT27 found. The architect's actual use case (one source, two independent analysis contexts with different parameterized aliases) is two INSTANCES, which stays legal and is tested. `test_b32_23`, `test_b32_23a` |

**Scope of the D2 refusal, pinned as a matrix (`test_b32_52`–`58`) because
getting the width wrong breaks something real in either direction.** The rule
is a property of the RESULT: after this registration, would one object be
reachable by two distinct paths *in this graph*?

| Shape | Verdict |
|---|---|
| Same child into two parents that do NOT share a graph | **legal** — `time_series_TroubleShooting.py` does exactly this |
| Same child reachable twice in one graph | refused, whichever registration completes the second path |
| Same object under two names on one parent | refused |
| Re-registering the same name | legal — an update |
| Distinct instances under distinct names | legal — this is the shape D2 directs users to |
| A frame registering itself | legal — a cycle with one name, already refused by `materialize_aliases` |

Two implementation misses are recorded because the tests exist to prevent them
coming back. The first version was **too wide** and broke
`test_N1_7_cycle_detection` by refusing self-registration. The second was
**order-dependent**: it asked "have I already seen this object", so
`root←C, mid←C, root←mid` was accepted while the same three registrations in
another order were refused — one structure, two answers.

**This closes P1-D1xD2Inconsistency by prohibition rather than by
reconciliation.** There was no non-arbitrary answer to "how many completions
happened" for one object under two owner names; the ruling makes the question
unaskable instead of picking an answer.

**Mechanical items closed:** `clear_after_on_error` documented in `Args`;
"two-phase" → "multi-phase" throughout; `test_b32_30` asserts the specific
consequence and that the spec was *not* rewritten; the unsupported
sandbox-sweep sentence removed from the commit message; the staging check made
fail-closed with an exact path set and index-hash comparison.

**Cells flipped**

| Cell | Was | Now |
|---|---|---|
| Entry selection × subframe column | silently produced no column | correct values for the selected rows, all three entry forms |
| Entry selection × nested subframe | index-label alignment, accidentally right | positional |
| Subframe ref in `defaults` / kwargs | joined, never rewritten | rewritten; caller's dict still untouched |
| Repeat call on one instance | persistent column relabelled temporary | classified by where it lives |
| Failed subframe resolution | warning, then a confusing downstream NameError | ADF-owned error naming the reference; `'warn'` opt-out |
| Failure record | render only, one branch for two cases | three phases, `failure_phase`, distinct outcomes |
| Ctrl-C during a batch | treated as a render failure | propagates untouched |
| Same object under two subframe names | supported (2026-07-25) | refused (2026-07-27); two instances instead |


### Backward-compatibility check against real user code

Asked directly by the architect: *will the old scripts stop working?* Checked
rather than asserted.

| Call site | Affected by D1 (raise) | Affected by D2 (refusal) |
|---|---|---|
| `examples/time_series/time_series.py` — 8 `register_subframe`, all distinct instances | no | no |
| `examples/time_series/time_series_TroubleShooting.py` — incl. one frame registered into **two parents** (`adf` and `adfgbTPCDSec20`, lines 561–562) | no | **no** — the two parents do not share a graph |
| `examples/time_series/time_series_draw.py` — `entry_begin/entry_end` on `draw`, no subframe in the same call | no | no |
| `tutorials/drawing/*`, `tutorials/cheatsheets/*` | no | no |
| `scripts/census_draw_path.py` | no | no |

`tutorials/drawing/with_subframes.py` and `selections_groupby.py` were executed
on both trees; behaviour is byte-identical, including two failures that
`with_subframes.py` already had (`available_branches[:5]` slices a set; a
`corrected` alias the script never defines). Neither is caused by this
increment and neither is fixed here.

**The residual risk, stated rather than implied.** If
`adfgbTPCDSec20` is ever registered *into* `adf`, line 562 becomes a second
path to the same object and will be refused. That is the ruling working as
intended, but it is a change that would bite an existing script, so it is
recorded here rather than left to be discovered.

**D1's raise is reachable only where the old code already failed.** Every case
that now raises previously emitted `[draw_batch] Failed to resolve subframe
ref ...` and then handed dfdraw an unresolved dotted reference, which failed a
few frames later. No call that produced a plot before produces an error now;
the error simply arrives at the right place with the right message. `'warn'`
restores the old sequence exactly for anyone who was catching the downstream
failure.

---

## Fix log — B3.2 part 2, correction round 3 (2026-07-27)

**Panel `[X]` again, and the finding was the worst category this chain
tracks: silent wrong numbers.** All five GPT seats executed a subframe join
with a *missing* parent key. None of the three Sonnet seats did, although the
review request listed that scenario by name. `_compute_join_indices` returns
`-1` as its missing-key sentinel plus a `missing` mask; the projection phase
captured `missing`, never read it, and did `values[join_idx]`. NumPy reads
`-1` as "last row", so a parent key with no child match received the child's
final value — no exception, no warning.

**Why 2,355 passing tests coexisted with it.** Every subframe fixture in the
characterization file generated parent keys inside the child's key range, so
the missing branch could not fire. The round-2 "compare against a
hand-computed join" oracle was built on a fixture where every key matched — an
oracle that could not fail in the way that mattered.

**Standing rule adopted this round: no value oracle may use a fixture where
every parent key is present in the child.** Every fixture in
`TestB32MissingJoinKeys` carries at least one unmatched key.

### The fix, and what it closed for free

The projection phase now **borrows** `_extract_subframe_values_cached`, the
established owner of missing-mask handling, `fill_missing`, and dtype policy,
instead of restating the gather. `join_idx` **and** `missing` are sliced by the
selected positions first, so the gather is already reduced-frame sized.

That one change also closed both **D1 escape paths** (GPT26, GPT27): the
helper owns child-alias materialization and raises `KeyError` for an absent
column, so a failed child alias and a missing leaf column now reach the
projection `except` and therefore `_fail()`. They previously warned, continued,
and died inside dfdraw with `failure_phase="render"` for a projection failure.
Borrowing a contract closed the paths that restating it had left open — the
general lesson of this round.

### Everything else

| Finding | Raised by | Fix |
|---|---|---|
| Self-registration under two **different** names accepted twice | GPT27 | The graph walk skips the root (no subframe name) and the ancestor guard never turns a self-path into a walked node, so the two were never compared. A direct registry scan closes it; a single self-registration stays legal, which is what the cycle contract and `test_N1_7_cycle_detection` own |
| Shared descendant of two registered subtrees accepted | GPT25, GPT26 | Already closed by the order-independence change (the incoming subtree's identities are compared, not just its root) |
| Partial preparation failure leaves the alias fields empty | GPT26 | `_record_draw_failure` re-measures before deciding. A failing path costs one graph walk; the alternative is a record that says nothing happened while an alias sits materialized and uncleanable |
| Entry validation outside every bracket | GPT27 | Bracketed; `failure_phase="entry_selection"` |
| Struct assertion + plotter construction between two brackets | GPT27 | Bracketed with the projection guard |
| `cleanup_outcome` relabelled `nothing_to_clean` after a failure with no candidates | Sonet27 | Outcome written **after** the cleanup phase, not before |
| `on_subframe_error` accepted any string | GPT26, GPT30 | Validated to `{'raise','warn'}`. A typo silently changing failure behaviour is the same class of defect as the warning nobody watched |
| `on_error='skip'` × `on_subframe_error` undocumented | GPT25, GPT27 | Documented in `Args` and pinned by `test_b32_71`: they act at different phases and do not substitute for each other |
| D1 ruling date used interchangeably with the 07-25 batch | GPT27 | Corrected |
| Docstring said warn "then fails" unconditionally | GPT30 | Qualified |
| Trailing whitespace on added lines | GPT26 | Cleaned |

### Architect rulings, 2026-07-27 (second batch)

**D2 is GRAPH-LOCAL.** Within one reachable graph an object may not appear at
two logical paths. Across *disconnected* graphs it may — which preserves
`time_series_TroubleShooting.py`, where one grouped frame is registered into
both `adf` and `adfgbTPCDSec20`.

**The cost is stated, not dressed up.** Two disconnected parents holding the
same object share mutable state: materialized aliases, struct state, columns,
caches and cleanup performed through one are visible through the other. This
is **not** an enforced read-only or copy-on-write mode; a real shared-memory
mode would be a separate design. `test_b32_82` asserts the sharing so the
hazard cannot go stale in prose while the code changes underneath it. For
independent contexts the supported model remains two instances over the same
source.

**The lazy-subframe gap is fixed here, not deferred.** And the disclosure it
replaces was wrong. Two rounds described it as "a lazy subframe referenced only
through an ALIAS"; executing it shows a plain physical column fails
identically. The real condition: once a lazy subframe has been materialized by
anything — `get_subframe()`, an earlier draw, adding an alias to it — the
PARENT's join index columns are never loaded, and every later `draw_batch`
reference fails with `None of [Index(['sec'])] are in the [columns]`. The
parent's index load had been nested inside "is the child still unloaded",
which are unrelated conditions. `test_b32_75`–`78`, including the realistic
shape: two draws in a row, where the first one broke the second.

**Cells flipped**

| Cell | Was | Now |
|---|---|---|
| Missing join key | child's last row, silently | NaN, or the configured `fill_missing` |
| Missing key × entry selection | wrong value at the wrong row | correct, all selection forms |
| Missing leaf column / failed child alias | warned, died later as `render` | raises at `projection`, naming the reference |
| Self under two names | accepted | refused |
| Lazy subframe after any touch | every later reference failed | works; two draws in a row work |
| Entry-validation failure | `failure_phase=""` | `"entry_selection"` |
| Partial preparation failure | record said nothing happened | reconciled before the cleanup decision |
| Disconnected shared frames | undescribed | allowed by ruling, hazard asserted by test |

---

## Fix log — B3.2 part 2, correction round 4 (2026-07-28)

Panel `[X]`, three findings, all reproduced by the coder before acceptance.
Architect rulings recorded as **AD-7 / AD-8 / AD-9** in
`docs/ARCHITECT_DECISIONS.md`.

**P0 — dtype regression (AD-7).** Round 3 fixed missing keys by borrowing
`_extract_subframe_values_cached`; that helper allocates
`np.full(n, np.nan, dtype=float64)` for every non-floating column. Four GPT
seats executed it: `object`/`category` RAISED `could not convert string to
float`, and `int`/`bool`/`datetime` were silently coerced — datetime to raw
epoch nanoseconds, which still plots.

Fixed in two parts:

- **All keys matched → gather directly in the source dtype, allocating
  nothing.** This alone restores every dtype for matched joins, which is the
  case that regressed.
- **Keys missing → a representation the dtype can hold**: `NaT` for
  datetime/timedelta, `None` for object, native for category. Measured, not
  assumed (24→24 bytes, 27→27 bytes).

**And a correction the coder got wrong first, recorded because it is the
ruling's own point.** The first implementation RAISED for `int`/`bool` with
missing keys. That invented a policy where one already existed. The project
settled it in April: this layer yields `NaN`, and the user's declared dtype is
restored at the ALIAS layer by `_safe_dtype_cast` (fills `0`/`False`,
preserves dtype, warns) — the "recipe for default values on failure" the
architect was pointing at. Three tests predating this phase said so:
`test_A5_missing_child_key`, `test_D1_int8_dtype_preserved_through_join`,
`test_D2_bool_dtype_preserved_through_join`. **The full sweep caught it; the
focused suite did not, because the focused suite is the coder's and the
contract is not.** `test_b32_85` is kept and inverted so the mistake cannot be
made twice, and `test_b32_87` shows the two layers working end to end.

**P0 — D2 defeated by delayed connection (AD-8).** Attach two parents to a
root, THEN give each the same child: neither registration can see the other,
because a frame holds no back-reference to its parents. Registration-order
defences have now failed three times, so the check moved to where it cannot be
outrun: `_validate_frame_graph_ownership()` runs at `draw_batch()` entry,
**before any effect**, and refuses a graph in which one object is reachable by
two paths, naming both. Order-independent by construction — it sees the graph
that resulted, not the sequence that built it. Strictly read-only;
`test_b32_93` asserts it changes no frame, reader, alias, cache or record, and
`test_b32_95` asserts nothing is materialized before the refusal. The
registration-time check stays as an early, friendlier error.

**P1 — cleanup was the last unbracketed boundary.** A raising cleanup escaped
with `failure_phase=""` and `cleanup_outcome="not_requested"` while an alias
sat undropped. Now `failure_phase="cleanup"` / `cleanup_outcome="failed"`,
written directly rather than through `_record_draw_failure` — which would call
the same failing cleanup a second time (`test_b32_97` asserts it is entered
exactly once). GPT27's separate normalization/dispatch interval is bracketed
too (`failure_phase="normalization"`).

**The four "untested — status unknown" combinations are now statuses.** Missing
key on a lazy subframe, missing key on a nested path, empty child table, and
`fill_mode='safe'` all execute and all behave correctly (`test_b32_98`–`101`).
Leaving a category called "unknown" open across rounds is how the missing-key
P0 survived four of them.

**Whitespace cleaned BEFORE packaging**, per GPT26: stripping it afterwards
would change the reviewed bytes and their fingerprints.

**Cells flipped**

| Cell | Was | Now |
|---|---|---|
| Matched join, any dtype | float64 or a raise | caller's exact dtype, no allocation |
| Missing key, datetime/timedelta | epoch floats, silently | `NaT`, dtype kept |
| Missing key, object / category | raised | `None` / native, dtype kept |
| Missing key, int / bool | (first draft: raised) | unchanged established contract: NaN here, dtype restored by the alias layer |
| Duplicate owner via delayed attachment | accepted, ran, produced an ambiguous record | refused at `draw_batch` entry before any effect, naming both paths |
| Cleanup exception | `failure_phase=""`, `not_requested` | `"cleanup"` / `"failed"`, candidates retained |
| Normalization exception | unbracketed | `failure_phase="normalization"` |
| Lazy / nested / empty-child / safe-mode missing keys | unknown | executed and correct |


---

## Fix log — B3.2 part 2, correction round 5 (2026-07-28)

Two panels reviewed round 4 (GPT25/26/27/31 as full reviews; Sonet25/27/28/29,
Fabble5_7, GPT26, GPT30 through the synthesis). The union is eight findings,
all reproduced by the coder before acceptance, and **one of them is a false
statement in the round-4 CRR** rather than a defect in the code.

### The structural change: one primitive, not a branch per dtype

The architect asked whether the framework was being symmetrized or whether we
were "using a special `if` for each particular case". **We were.** The
implementation branched on datetime, category and object; this round's findings
would have added complex, timezone-aware, nullable-extension and interval
branches to the same chain. Three of five B3.2 rounds landed in the dtype
domain for exactly that reason — pandas' dtype surface is larger than any list
a person maintains, so enumerating it keeps missing a different corner.

The gather is now **one call**:
`pandas.api.extensions.take(arr, idx, allow_fill=True[, fill_value=])`, which
already speaks the `-1` missing sentinel `_compute_join_indices` produces.
Measured end to end through `draw_batch`:

| source dtype | matched | missing |
|---|---|---|
| float32/64, complex64/128, datetime64, **tz-aware datetime**, timedelta64, object, category, **Int64**, **boolean**, **string**, period | preserved | preserved |
| int8/int64/uint32 | preserved | `float64` — the ratified `_safe_dtype_cast` contract |
| bool | preserved | `object` — same contract |
| interval[int64] | preserved | **refused** per AD-11 (subtype would widen) |

Empty child tables fall out of the same call with no special case. Fill
representability is pandas' to decide, which closes GPT26's categorical-fill P1
by construction rather than by a hand-rolled membership check.

The matrix that guards it is **generated** (`DTYPE_CASES` × matched / missing /
empty-child / entry-selection): adding a dtype exercises every combination
automatically, so coverage is a property of the table rather than of what
anyone thought of on the day — Sonet29's structural recommendation, adopted.

### Findings closed

| Finding | Raised by | Fix |
|---|---|---|
| tz-aware datetime loses its timezone, **matched and missing** | GPT25/26/27/30/31 | the fast path returned `.values`, which strips extension metadata; it returns the array now |
| `Int64`/`boolean` → `object` on a matched join | GPT25/27/30/31 | same |
| complex + missing → `float64`, imaginary discarded | GPT26/27/30/31 | the symmetric primitive; `np.floating`-only checks are gone |
| empty **non-numeric** child raises out-of-bounds | GPT31 | same primitive; the round-4 test passed only because it used a float column |
| categorical fill by an existing category refused | GPT26 | pandas owns representability |
| symmetric `pre_index=True` fails: key is both index level and column | GPT26 | both key tables rebuilt from column values with a fresh positional index; `__sub_row__` mapping preserved |
| **back-edge cycle bypasses the AD-8 validator** | GPT30 | one guard was doing two jobs: the ancestor check terminates recursion AND was deciding what got compared. Edges are enumerated separately from the walk, so `root←A, A←root` is visible even though recursing into it would not terminate. Single self-registration stays legal |
| **dispatch interval still unbracketed — the round-4 CRR said it was closed** | GPT27/GPT30 | bracketed as `failure_phase="dispatch"`. The previous round bracketed the *normalization* loop and the claim was written as if that covered dispatch |
| partial cleanup drop not recorded | GPT25 | `aliases_dropped` measured in a `finally` |
| cleanup failure replaces the original exception | GPT25/27 | AD-10: original stays primary, cleanup kept in `secondary_error` — a field, not `raise ... from`, because chaining would read as causation |

### Mechanical

Stale `test_b32_86` citation corrected to `test_b32_93`; `test_b32_86` renamed
and now actually configures a fill; the `failure_phase` value list completed
(`entry_selection`, `normalization`, `dispatch`, `cleanup` were missing);
"allocating nothing" narrowed — `Series.take` does allocate the gathered
result, and the earlier wording overstated the property.

### Still open and NOT closed by this round

`_DrawDependencyPlan` remains an intermediate carrier rather than the full
Rev-2 plan contract. GPT27 raises this as a substantive P1 that independently
blocks closure, and it needs an architect ruling — complete it inside B3.2, or
assign it to a named later increment. The coder has asked twice and has no
answer; it is carried here so closure cannot happen by silence.

---

## Correction round 6 — GPT31's decision set (architect-approved 2026-07-28)

Four architect decisions, a standing symmetry requirement, and one item ruled
"fix it NOW". Recorded as **AD-12 … AD-17** in `docs/ARCHITECT_DECISIONS.md`.
Every finding below was reproduced on the round-5 bytes (`1e63052b`) before it
was fixed.

### The root cause, which is one thing and not six

The gather routed on `dtype.kind == 'f'`. `.kind` is defined on pandas
ExtensionDtypes as well as NumPy dtypes:

```
pd.Float64Dtype().kind           -> 'f'
pd.SparseDtype(np.float64).kind  -> 'f'
pd.SparseDtype(np.int64).kind    -> 'i'
```

A predicate that looked general was a per-dtype assumption wearing a general
face — the same shape of defect the round-5 symmetrization was supposed to
have removed, one level further in. Replaced by `_is_plain_float_dtype()`:
`isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.floating)`, which
asks the question that actually matters — *is this a real NumPy float buffer
that holds NaN natively?*

### Measured, before → after

| case | before (round 5) | after |
|---|---|---|
| `Float64` / `Float32`, **fully matched** | **`object`** | preserved |
| `Float64` / `Float32`, missing key | `float64` | `Float64` / `Float32` with `<NA>` |
| `Sparse[float64]`, matched or missing | densified `float64` | preserved |
| `Sparse[int64]` + missing key | dense `float64` — densified AND widened | refused, with the remedy named |
| `bool` + `fill_missing=0` | `object` holding `[True, False, 0, False]` | `bool` |
| `bool` + `fill_missing=2` | `object` holding a literal `2` | refused |
| `int64` + `fill_missing=1.5` | `int64` holding `1` — silent truncation | refused |
| category + a non-member fill | `NaN` | refused; a category is never added |
| `object` column + `fill_missing='NA'` | `TypeError: must be numeric` | works |
| complex + `fill_mode='safe'`, `fill_nan`/`fill_inf` | silently ignored | applied, identically to float |
| child indexed with `set_index(drop=True)` | bare `KeyError: 'kc'` | joins; a genuinely absent key still names itself |
| facet alias in `draw_batch` | excluded from the single bulk materialization | included |

### Decision 2's scope boundary — the thing that needed a ruling, not a fix

Decision 2 says *"do not silently change an integer column to `float64` or a
Boolean column to `object`."* Applied literally to the shared gather it
contradicts a contract ratified in April and pinned by three tests that
predate this phase (`test_A5_missing_child_key`,
`test_D1_int8_dtype_preserved_through_join`,
`test_D2_bool_dtype_preserved_through_join`): the join layer yields NaN and
`_safe_dtype_cast` restores the declared dtype at the ALIAS layer.

Round 4 broke exactly those three tests by inventing a policy where one
already existed. Rather than break them again by the opposite reasoning, the
conflict was reported before any code was written. AD-13 records the boundary:
Decision 2 governs matched joins, joins with a configured fill, and
extension/sparse integer dtypes; the April contract stands for a plain NumPy
int/bool column with a missing key and NO configured fill — documented and
tested, therefore not *silent*. Changing it would also draw a real `0` where a
TPC map currently leaves a gap.

### `facet_by` — a two-list problem, not a typo

`_parse_expr_aliases` took five of the six scalar draw slots. The branch scan
(`required_branch_kwargs`) derives its slot list from
`_EffectiveDrawSpec.SLOT_NAMES` and was correct; the alias scan hand-wrote the
same list and was silently short. One derived, one copied, and only the copied
one was wrong — which is the concrete argument for the standing symmetry
requirement (AD-16: B3.3 adopt, B3.4 delete duplicates, B3.5 prove the
matrix).

### Mechanical / P2

`_DrawDependencyPlan`'s docstring claimed a union-of-required-branches field
it does not hold (resolution moved to the executor two refactors ago);
`_extract_subframe_values_cached` documented an `np.ndarray` return while it
deliberately returns the column's own array; the AD-8 validator's placement
comment justified itself with a catalog effect that plan construction no
longer has; the "one call" claim narrowed to *one call per array kind* —
`ExtensionArray.take` or `pandas.api.extensions.take`, a dispatch on the array
protocol rather than on the dtype.

### Still open

`aliases_pre_existing` is frame-scoped while `aliases_materialized` is
graph-scoped (narrowed in documentation, not widened);
`makeSmoothMapsWithTPC.py` is not in this tree; `draw()` / `draw_figures()`
are B3.3; the alma2 peak-RSS isolation run is owed. The full Rev-2 dependency
plan is no longer "open" — AD-12 assigns it to B3.2b, before B3.3.

---

## Correction round 7 — two P0s, both the coder's, both executed by four seats

Round 6 was rejected `[X]` by GPT25, GPT27, GPT30 and GPT31, and the Main
Reviewer (Sonet29) **overturned its own `[OK]`** after re-verifying against
source. Two P0s converged 4/4 — the strongest convergence of the phase.

### The pattern worth naming: I fixed the symptom and re-typed the cause

Round 6's headline fix was removing `dtype.kind` from the gather router,
because `.kind` is defined on pandas ExtensionDtypes and therefore lies about
storage family. The same round left this standing one helper over:

```python
isinstance(dtype, np.dtype) and dtype.kind in "fc"      # the fill router
```

So `Float64` and `Sparse[float64]` silently discarded an explicitly configured
`fill_nan` / `fill_inf`, while the identical call on `float64` applied it. The
correct fix and the surviving defect were written in the same commit.

The second P0 has the same shape at a different level: round 6 *documented* an
invariant (AD-17: an index level and a same-named column are one key) without
ever *checking* it.

### P0-1 — every fill knob now obeys AD-14, on every storage family

`_coerce_fill_to_dtype` existed since round 6 and was called from exactly one
site. Measured before → after:

| call | round 6 | round 7 |
|---|---|---|
| `float64` + `fill_nan="BAD"` | `object` holding `"BAD"` | refused, knob + dtype named |
| `complex128` + `fill_inf="BAD"` | `object` | refused |
| `float64` + `fill_missing=Decimal("1.25")` | `object` | `float64`, value `1.25` |
| `Float64` / `Float32` + `fill_nan=99`, `fill_inf=77` | ignored | applied, dtype preserved |
| `Sparse[float64]` / `Sparse[float32]` + same | ignored | applied, sparsity preserved |
| `float64` control | applied | applied (unchanged) |

Applicability is now asked by **capability** — `pandas.api.types.is_float_dtype`
/ `is_complex_dtype`, which answer across plain NumPy, nullable and sparse
alike — never by `isinstance(dtype, np.dtype)`. Every knob writes through one
primitive, `_place_fill`, which coerces first and then assigns; arrays that
refuse item assignment (`SparseArray`) go through a dense **temporary** and are
restored to their exact dtype, so the result is never densified. That is one
try/except on the array protocol, not a branch per dtype.

`direct` mode still touches missing keys only — pinned per dtype family, since
the guard moved into the shared helper.

### P0-2 — an ambiguous join key is refused, and it was OUR regression

```
child index k=[0,1,2], child column k=[2,1,0], parent k=[0,1,2]
round 6: [30.0, 20.0, 10.0]      silently reversed, no error
round 7: ValueError naming the subframe, the side, and both value samples
```

Checked against the pre-phase baseline `c73f0c99`, pandas had been refusing
this shape itself:

```
ValueError: 'k' is both an index level and a column label, which is ambiguous.
```

The round-4 ambiguity normalization — added to make `pre_index=True` work,
where the two spellings *always* agree — rebuilt both key tables from column
values and removed that refusal for the case where they do not. So this is a
defect introduced by this phase, not a pre-existing gap, and round 6 then wrote
a decision asserting the surviving behaviour was safe.

The rule is now checked on both sides, at registration (before the registry is
written, so a refused registration leaves no partial state) and again at graph
consumption, since a frame can be re-indexed afterwards.

Also closed under the same helper (GPT31 P1): `pre_index=True` on a child with
a MultiIndex joined on a **subset** of its levels raised
`KeyError: "None of ['a'] are in the columns"`. The predicate now asks, per
key, whether that key is reachable, instead of comparing whole index-name
lists.

### Record items

`test_fill_missing_rejects_non_numeric` renamed to
`test_fill_accepts_any_scalar_and_rejects_containers` (it proved the opposite
of its name); AD-17 corrected from "ratified by implementation" to **PROPOSED**
— implementation cannot ratify a decision (GPT27, GPT30); the revision history
re-ordered chronologically; `set_global_fill` / `set_subframe_fill` docstrings
no longer describe the removed numeric-only contract; `np.array(1.0)` accepted
as the scalar it is (GPT25 P2-3); `run_tests.sh` now ships a separate
focused-suite log in the packet so the CRR's headline count is verifiable
rather than trusted (GPT30 P2-1).

### GPT27's AD-12 objection — not upheld, by the Main Reviewer

GPT27 read the ratified decision as making B3.2b a mandatory sub-increment
that blocks B3.2 closure. Sonet29 checked the primary ratified text rather than
a paraphrase: the only sequencing constraint approved is **B3.2b before the
`draw()` / `draw_figures()` migration**. AD-12 stands, with the challenge and
its adjudication now recorded in the entry itself.

### Two items still need the architect, not code

- **AD-13's direct-slot reading** — GPT27 alone holds that Decision 2's literal
  text forbids the ratified April NaN-at-join contract on the *direct*
  (non-alias) path. GPT25 explicitly ratifies AD-13's boundary as written.
  Needs a recorded answer, not a code change.
- **AD-17's ratification** — the entry is PROPOSED and stays that way until the
  architect rules.

---

## Correction round 8 — the round-7 panel, and one architect ruling

Round 7: **GPT25, GPT26, GPT27, GPT31 → `[X]`; Fabble5_7 → `[OK]`.**
Everything below was reproduced on the round-7 bytes (`423c0d36`) before a
line was changed.

### The architect ruling that came with it — AD-13a

The panel split 3–2 on the direct (non-alias) int/bool path: GPT25/26/27 read
Decision 2 as forbidding the widening outright; GPT31 and Fabble5_7 held that
the widening *is* the missing-ness. The architect separated the two questions
and chose **Option 3**:

| aspect | ruling |
|---|---|
| the value | stays a gap — ADF never invents a measurement |
| the dtype change | **reported**, per column per call, `FutureWarning` |
| the future | the notice says it **will become an error** |
| the remedy | `set_subframe_fill(fill_missing=...)`, which preserves the dtype today |

Scoped to the one call site with no alias restoration downstream. The alias
path is untouched — `_safe_dtype_cast` already restores and already warns
there, and a second notice would train users to ignore the one that matters.
No script that produces a figure today stops producing one.

### 4/4 — `pd.NA` in a join key raised a raw `TypeError`

```
register_subframe(...)   ->  TypeError: boolean value of NA is ambiguous
```

for `object`, `string`, `boolean` **and** `Int64` keys. Only the plain float
`NaN` control survived, which is exactly why the round-7 matrix passed. The
comparison resolved missing-ness *after* the reduction instead of before it.

Now: both-missing = equal, one-sided = different, compare only the non-missing
subset — and the spelling of the gap (`None` / `np.nan` / `pd.NA` / `pd.NaT`)
is not significant, because pandas normalises between them on a `set_index()`
round trip without the user asking.

### The round-7 record was false about "before any effect"

I wrote that a refused registration "leaves no partial state", and tested the
registry entry. GPT31 tested the rest: the child's `_schema` had already been
auto-populated and the parent's join cache invalidated before the raise.

Every check now runs above the first mutation. `test_b32_168` snapshots child
schema, parent schema, join cache, child index and child columns across a
refusal, for conflicting keys and absent keys, with and without `pre_index`.

### A pre-existing hole the same move closed — and two wrong attempts at it

The key-existence check had lived inside the `right_index_columns is not None`
branch since PHASE_13_65, so the **ordinary symmetric call** — the one every
existing script makes — never ran it on the CHILD. A child without the join
key registered successfully and wrote both registry and schema (GPT25).

**Getting the scope right took two failed attempts, both caught by the full
sweep and neither by the focused suite** — the same asymmetry as round 4, and
worth recording because the reflex to widen a validation is the recurring
failure mode of this phase:

| attempt | what broke | why |
|---|---|---|
| check both sides always | **29 tests** | a parent legitimately gains its key *after* registration — a declared alias materialized later, or an unloaded branch under a lazy reader |
| drop the parent check entirely | `test_A6` (PHASE_13_65) | an unknown PARENT name **is** a registration error when `right_index_columns` is given explicitly |

The original placement was therefore not the defect. Parent-side existence is
a ratified contract of the **asymmetric** call and stays exactly there; only
the CHILD-side check widens to the symmetric call, which is precisely what
GPT25 reported.

One further consequence: the round-2 fixture `_broken()` relied on the hole,
so it is repointed at the shape D1 was actually ruled for (a missing *column*,
not a missing *key*), and the old shape is kept as `test_b32_47b`.

### Portability: a pandas primitive is not a dtype guarantee

```
pandas 1.5.3   Sparse[float32].take(...) -> Sparse[float32, nan]
pandas 3.0.2   Sparse[float32].take(...) -> Sparse[float64, nan]
```

— and on the newer pandas it widens even for **matched** positions. Round 7's
exact-preservation claim held only on the coder's and the architect's pandas,
and the matrix that "proved" it fails elsewhere. GPT26 found it by executing on
pandas 2.2.3, a runtime no seat had used before; that is the whole reason it
took seven rounds. Results are now normalized back to the verified source
dtype, with a losslessness check that reads only the array's *stored* values.

### The public-path requirement (GPT27 QRC Rule 4)

The round-7 matrices called private helpers. Every round-7 value/dtype claim is
re-proved here through `draw_batch()`, asserting on the frame actually
delegated to dfdraw: `test_b32_159`–`164`.

### Measured, not deferred

The `_place_fill` dense fallback costs **~5.25× the dense column** — 10M rows:
420 MB peak against an 80 MB dense equivalent, independent of density, and only
for a sparse column that also has a fill knob configured. Removing the
defensive `.copy()` changed nothing measurable, so the honest statement is that
the whole densify-fill-resparsify round trip costs that, not that one line
does. A sparse-index reconstruction that never densifies is a named **B3.2b**
item.

### Record items

AD-18 relabelled an **implementation consequence of AD-14**, not a ratified
decision — "ratified by implication" was the same invalid move the same
document had corrected for AD-17 one section earlier. `_get_fill_config`'s
return documentation no longer says `float or None`.

---

## Correction round 9 — AD-19 ratified, AD-13a superseded

Round 8: GPT27, GPT30, GPT31 → `[X]`; Fabble5_7 → `[OK]`. One confirmed P0
producing **silently wrong scientific values**. Everything below was
reproduced on the round-8 bytes (`99e911d8`) before a line was changed.

### The ruling that reshaped the round

The architect ratified **AD-19** in his own words, and explicitly **rejected**
the weaker formulation the review had proposed:

> A missing-key operation may not change the explicitly supplied dtype and may
> never change any non-missing value. If the dtype cannot represent the gap,
> ADF must require an explicit compatible fill or refuse clearly.

and superseded **AD-13a** — *"widen now, warn, error later was not my decision
and contradicts AD-19"*. **AD-17** is ratified.

So the round-8 policy was not merely incomplete: warning-and-widening is not a
permitted transitional state at all, because the widening can change values,
and no warning makes a changed measurement acceptable.

### The P0 — measured, and why eight rounds of matrices missed it

```
source (int64)   1152921504606846977, ...979, ...981, ...983
matched          exact
one key missing  1.152921504606847e+18  x3, NaN
                 -> three distinct measurements collapsed into ONE
via alias dtype="int64"
                 1152921504606846976 x3, 0
                 -> cast back to int64: type-correct, value-wrong
```

`uint64` above 2**63 behaves identically. `int32` does not — and that is the
whole explanation for the blindness: **every integer in every dtype matrix
built in rounds 6, 7 and 8 was small enough to be exactly representable as
float64**, so the table could not fail. The same shape as the round-3 fixture
where every key matched.

These are not exotic values here: a track/timeframe uid, a nanosecond
timestamp, a bunch-crossing id. One missing join key silently merged distinct
tracks.

### Behaviour now

| case | behaviour |
|---|---|
| fully matched | exact dtype, exact values — always |
| missing key, compatible fill configured | exact dtype, gap carries the fill |
| missing key, widening would change a value | **refused**, remedy named |
| missing key, widening provably lossless, no declared dtype | ratified April representation stands |

Never widen-and-warn. Never cast rounded floats back to an integer dtype. The
guard runs on **both** the direct and the alias path, because the alias path
was the worse of the two — it restored the dtype and therefore disguised the
corruption.

**Implementation note that is itself a finding.** The first version of the
guard compared values through `to_numpy(dtype=object)`. That is 3× slower and
allocates one Python object per row — roughly 600 MB of boxed integers for a
10M-row child column, on a path inside every draw. It would have violated the
D-ADF-DICT contract in the act of enforcing AD-19. Replaced by a vectorised
numeric round trip through the source dtype, which is exact for this question
and allocation-free.

### One normalization point, matched as well as missing

Round 8 normalized only the missing-key path, so a **fully matched**
`Sparse[float32]` was delegated as `Sparse[float64]` on pandas 2.2.3 — and the
round-8 CRR claimed otherwise (GPT27 FIX8-P0-1, GPT30 B32F8-P0-1, both
executed). The gather primitive's dtype is not a contract on any version:

```
1.5.3  preserves matched AND missing
2.2.3  widens    matched AND missing
3.0.2  preserves matched, widens missing      (Fabble5_7)
```

Three adjacent versions, three behaviours. The dtype is now verified and
restored at one point covering both branches, and a restoration that would
change a value is refused rather than applied.

**And the test no longer depends on the runner's pandas** (GPT30 B32F8-P2-1):
`test_b32_178` monkeypatches the primitive to widen unconditionally, so
deleting the restoration fails the suite on 1.5.3 too.

### Lazy-reader branches — my regression, fourth iteration

A branch a lazy reader *advertises* is present: it is physical data the frame
owns and has not loaded yet. Round 8 refused it on the child side (GPT27
FIX8-P0-2 — round 7 accepted that shape, so this was a regression I
introduced) and on the parent-asymmetric side (GPT30, GPT31).

The round-8 source comment said an unloaded lazy branch is a legitimate
deferred key, and the predicate written directly beneath it did not check for
one. `_has_key` now reads `available_branches` from `_lazy_reader` /
`_chain_reader` and from lazily-registered subframe readers — read-only, never
loading — and a genuinely unknown key is still refused (`test_b32_180`–`184`,
including one that asserts the validator loads nothing).

### Record

`AD-19` recorded canonically in the architect's words; `AD-17` ratified;
`AD-13a` superseded with its original text kept for the record; the warning
text no longer claims "the VALUE is correct" — it states that matched values
have been *verified*, and names the round-8 claim as false. AD registry is
v2.0.0: AD-19 changes a public behaviour that had held since April.

---

## Correction round 10 — AD-19 ratified with an operational definition

Round 9: GPT27, GPT30, GPT31 → `[X]`; Fabble5_7 → `[OK]`; the Main Reviewer
overturned his own `[!]`. Everything below was reproduced on the round-9 bytes
(`833638dc`) before a line was changed.

### The architect's addition that made this implementable

> **ADF does not need to know whether the user consciously typed `dtype=...`.**
> Every dtype observable from source metadata, an existing physical column,
> schema metadata, an explicit alias declaration, or the first successful
> creation/materialization is authoritative. ADF must preserve it thereafter.
> If a missing value cannot be represented in that dtype, ADF must use an
> explicitly configured compatible fill or refuse clearly.

That removes the guessing the coder was stuck on and it settles AD-19-SCOPE as
**Option 1**. The round-9 "lossless widening may stand" exception is gone: the
dtype changing at all is the violation, not whether the numbers survived.

### The P0 that survived round 9's own fix

```
add_alias("d", "S.v", dtype="int64")     ALL FOUR KEYS MATCHED, no missing key
source   1152921504606846977, ...979, ...981, ...983
round 9  1152921504606846976 x4
```

`_safe_dtype_cast` ran `np.asarray(result, dtype=np.float64)` **unconditionally**
— with zero NaN present. Round 9's exactness guard is in the GATHER;
`_safe_dtype_cast` runs downstream of it on the alias path, and was untouched
by the round-9 diff. So a fully matched declared alias was corrupted by the
very function whose job is to preserve its dtype, and it stayed corrupted even
when a subframe or global fill was correctly configured.

**Both reviewer traces were right, exactly as the Main Reviewer suspected.**
Sonet28 was right that the gather guard is shared by both paths; GPT30 was
right about what happens after it. The Main Reviewer declined to resolve it by
reading and asked for one executed reproduction. It settled on execution.

The same five lines held the second defect: `fill = False if kind=='b' else 0`
— the automatic neutral value AD-19 forbids. 0 is neutral for an additive
correction, 1 for a multiplicative one, and a dtype cannot tell them apart.

### Configured fills — all three mechanisms now reach the gather

```
alias dtype=int64, fill_value=0  (additive)        -> int64, gap 0, matched exact
alias dtype=int64, fill_value=1  (multiplicative)  -> int64, gap 1, matched exact
set_subframe_fill(fill_missing=7)                  -> int64, gap 7
set_global_fill(fill_missing=5)                    -> int64, gap 5
nothing configured                                 -> REFUSED, all three named
```

**Precedence unchanged and pinned** — the architect asked that historical
behaviour not change silently, and the measured order is:

```
subframe fill  >  global fill  >  alias fill  >  refusal
```

Round 10's only change is that the alias value reaches the *gather* rather
than being applied after it. Same observable order; what changes is that it
now works for values that cannot survive the intermediate, which round 9
refused outright.

### Blast radius — measured, and much smaller than expected

A ruling that turns a widening into a refusal could have broken a great deal.
Across the whole suite it broke **four** tests, three of which the architect
named himself:

```
test_A5_missing_child_key                     (13.65)
test_D1_int8_dtype_preserved_through_join
test_D2_bool_dtype_preserved_through_join
test_I3_9_flat_normalized_equivalence
```

All four are revised to the AD-19 contract — each now asserts *both* halves:
the refusal without a configured fill, and exact dtype plus exact matched
values with one. Plus 29 in the phase's own file, all of which encoded the
superseded behaviour.

After the revisions the sweep identity set is **identical to round 9**, which
is identical to rounds 5–8: the sixth consecutive round with a character-stable
set.

### Record

`_warn_direct_slot_widening` is **deleted**, not disabled — AD-13a is
superseded and had nothing left to stand on; `test_b32_154` asserts its
absence. `_matched_values_survive` gained a same-domain check after GPT27
showed `source 1, gathered 1.5` passed the round-trip alone; the round-9 CRR's
claim that it proved "no non-missing value may ever change" was stronger than
the code. AD registry v2.1.0.

---

## SUPERSESSION NOTICE — "Configured fills — all three mechanisms now reach the gather"

**Recorded:** 2026-08-01, checkpoint fix11a. **Authority:** ratified contract
v1.4.2 as amended by v1.4.4, **AR-3**.

The section above titled *"Configured fills — all three mechanisms now reach
the gather"* records **ROUND-10 behaviour and is superseded.** Its claim —

> *"Round 10's only change is that the alias value reaches the gather rather
> than being applied after it. Same observable order"*

— is false for any expression that is not the bare reference `S.v`. Measured:

```text
alias = "S.v + x", dtype=int64, fill_value=1, key missing, no operand fill
    pre-phase c73f0c99  ->  [13, 1]
    round 10 f9781761   ->  [13, 21]
```

The flat precedence list `subframe > global > alias > refusal` is likewise
superseded: it describes one stage, and there are two.

**Ratified contract:** operand fills (`set_subframe_fill`, `set_global_fill`)
apply before evaluation and DEFINE the operand — precedence among them stays
`subframe-specific -> global -> native gap / refusal`. Alias `fill_value`
applies after evaluation, only to a final result that is still undefined or
invalid, and is never injected into a subframe operand.

**Checkpoint status:** the fix11a bytes still implement the superseded
behaviour. Pinned by `b32_195`–`197` as `xfail(strict=True)`; corrected by
`D_4`/`D_5`. Historical entries above are retained and marked superseded, not
deleted.
