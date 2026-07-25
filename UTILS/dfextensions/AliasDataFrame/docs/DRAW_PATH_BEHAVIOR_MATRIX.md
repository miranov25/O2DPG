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
