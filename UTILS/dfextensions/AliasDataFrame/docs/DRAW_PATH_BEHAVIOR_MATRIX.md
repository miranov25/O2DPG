# DRAW_PATH_BEHAVIOR_MATRIX.md — PHASE_13_76_ADF Stage A (A1 + A2)

*Part I: behavior matrix. Part II: source-path/semantic-owner inventory (the
evidence base for the "source owner" column). One file by architect
preference, 2026-07-19.*

**Status:** Stage-A working document, increment 7 (data-states axis complete — sweep DONE on all axes).
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

## 3. Open cells queue (next characterization increments)

1. ~~SEED-3.e~~ DONE (Q-C resolved by AD-6; acceptance test_seed3_7 strict xfail).
2. ~~Slot × surface sweep (scalar slots × draw/draw_batch)~~ DONE (SWEEP-1);
   remaining: data states (§8.8: lazy tree, lazy chain, subframes).
3. `entry_begin/entry_end/entry_mask` layer rows — REMAINING (Gate-A candidate: characterize or explicitly defer with owner).
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
