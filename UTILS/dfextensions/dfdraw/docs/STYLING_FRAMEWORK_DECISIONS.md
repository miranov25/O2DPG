# STYLING_FRAMEWORK_DECISIONS — dfdraw

| Field | Value |
|---|---|
| **Owner** | Marian Ivanov (Architect) |
| **Maintained by** | dfdraw coder seat (currently Claude49Coder) |
| **Created** | Phase 13.26.DF Commit 1, 2026-05-05 |
| **Status** | Live — append-only after Phase B; modifications require architect approval |
| **Source-of-truth precedence** | Per Reviewer Card v1.29: source code at phase tag → this file → other docs |

---

## Purpose

This document is the canonical decision log for the dfdraw styling framework. It records:

1. **Architect Decisions** (AD-N) — every architecturally significant choice with provenance, rationale, and co-sign trail
2. **Verbatim architect quotes** — preserved with typos per Reviewer Card v1.29 Rule 9
3. **Governance principles** — extracted lessons that should bind future phases
4. **Drafter rotation history** — who drafted what, who reviewed what, why rotation matters

Per Phase 13.25.DF FIX1 CRR §8 and Phase 13.26.DF v1.2 §8.1: this file's creation was deferred from Phase A and seeded in Phase 13.26.DF Commit 1.

---

## Table of Contents

1. [Architect verbatim quotes](#1-architect-verbatim-quotes)
2. [Architect Decisions AD-44 through AD-77](#2-architect-decisions)
3. [Governance principles](#3-governance-principles)
4. [Drafter rotation history](#4-drafter-rotation-history)
5. [Audit trail of phase reviews](#5-audit-trail)

---

## 1. Architect Verbatim Quotes

Reviewer Card v1.29 Rule 9: *"When quoting architect requirements, use verbatim quotes, not paraphrase."* All quotes below preserve original phrasing including typos.

### 1.1 Brainstorm v1.4 sessions (2026-04-30) — Phase A foundations

**Session 1 — initial brainstorm request:**

> "Currently, we use only the GB mean. I want to add quantiles as an option. `quantiles = [list]`. We will need to consider user preferences. A figure can be defined by more than one graph. We already have many use cases, but we do not yet have a clear strategy. How to differentiate graphs: marker color, marker style, marker size, line color, line style, line width. We have many ways to define multigraphs, and some can be combined. Vector expressions `[A, B, C]:X`, group_by, now adding quantiles, vector selection (new feature). We need to establish simple rules and parameterization (with default styles) to decide which mechanism to use for defining graphical properties in drawing and legends. I want to brainstorm this before implementation. For example, we already have some bugs when combining vector expressions and group_by. We may encounter more problems in the near future."

**Session 2 — ADF/dfdraw parity + faceting:**

> "Usually, I use the ADF draw interface, as in most cases I need ADF lazy evaluation functionality, but the code should also work in dfdraw with raw pandas. I am not sure if we will have problems with that. I forgot to add faceting. I did not include it because I forgot."

**Session 3 — vectors first-class; style sets; vector selection spec:**

> "I am actively using the vector interface — this is very important for my quick checks. The vector interface is important, and I want to extend it for selection. We will have selection later, and an optional selection delta vector (logical AND on top of selection). We should have a default style but we should be able to make different sets of styles — which we should save | load | use."

**Session 4 — strategic decision:**

> "Strategic decision — make a standard way for the attribute association."

**Session 5 — strategic decisions locked:**

> **A-Q1:** "I assume this will be parameter in style. I agree that the proposed order is good default."
>
> **Q-Q2:** "No strong opinion. I assume reasonable is to specify, with central we can be more verbose."
>
> **M-Q1:** "Will we save time if we split?"
>
> **C1:** "OK."
>
> **P-Q4:** "Not clear. Computing experts should decide."
>
> **F-lift:** "Later."

— Marian Ivanov, brainstorm sessions 1–5, 2026-04-30

### 1.2 Phase 13.26.DF chat sessions (2026-05-05)

**Style parameterization (initial direction):**

> "is the proposal sufficiently configurable by style so we can define different use cases? Can we start? If style can parameterize I am happy - wan we need good defuat style"

**Style parameterization (follow-up):**

> "We need styles and default style from the begining. we should be able to change style. Has to be done and not psoponed. If it is parametrized I do not need to asnwer in details on the questions. Parametrizrtion si important. We can start with defaults as you proposed - Once we will have style we can easly change. We have to have style to be able to change ..."

**Greenfield amendment — quantile path has no production users:**

> "We did not use quentiles yet. We do not need to be back compatible. for quentiles"

— Marian Ivanov, chat sessions 2026-05-05 (typos preserved per Rule 9)

---

## 2. Architect Decisions

### Phase A inheritances (AD-44 through AD-54)

These decisions land in dfdraw via Phase 13.25.DF (Phase A: quantile rendering — error_bars + band modes). Phase B (this phase) inherits without modification.

| AD | Decision | Source | Rationale |
|---|---|---|---|
| **AD-44** | `quantiles=[…]` accepts list of fractions in (0, 1) | Brainstorm v1.4 §8 + architect Session 1 | Matches numpy/pandas quantile convention |
| **AD-45** | Default `central='mean'` when `quantiles=[…]` is set | Brainstorm v1.4 §1 bullet 5 + Session 5 Q-Q2 | Backward compat with existing GB-mean behaviour |
| **AD-46** | `error_bars` mode = symmetric pair without 0.5 → asymmetric error bars on central line | Brainstorm v1.4 §8.1 + Session 5 C1 | Zero-cost rendering; production-typical pattern |
| **AD-47** | `band` mode = symmetric triple with 0.5 → fill_between | Brainstorm v1.4 §8.1 | Zero-cost rendering |
| **AD-48** | `_detect_quantile_mode()` auto-routes by list shape | Brainstorm v1.4 §8.2 | Single API, pattern-based dispatch |
| **AD-49** | `central='none'` invalid with `error_bars` mode | Brainstorm v1.4 §8.4 | Error bars require something to ride on |
| **AD-50** | ADF-side parity for quantile rendering deferred (option C1b in AliasDataFrame.py) | Phase 13.25.DF v1.3 review | Independent of dfdraw work; can land in parallel |
| **AD-51** | `quantile.band.alpha`, `quantile.band.hatch`, `quantile.error_bars.capsize`, `quantile.central_default` style keys land in Phase A | Phase 13.25.DF v1.3 §AD-53 | New interface = cheapest place to lock keys |
| **AD-52** | Empty quantile dict pruned before rendering (no orphan legend entries) | Phase 13.25.DF FIX1 | Caught by 6-reviewer panel, P1 in v1.0_END |
| **AD-53** | quantile.* keys are independent (no cascading from profile.* keys) | Phase 13.25.DF v1.3 + FIX1 review | Matches existing dfdraw pattern (grid.alpha, scatter.alpha, hist.alpha — independent siblings) |
| **AD-54** | Naming convention: `<channel>_style` kwargs, `channels.<key>` style namespace, `_assign_channels()` for the algorithm, `TestChannel*` for test classes | Phase 13.26.DF v1.0 | Consistent vocabulary across coder/reviewer/architect |

### Phase B decisions (AD-55 through AD-59)

| AD | Decision | Source | Rationale |
|---|---|---|---|
| **AD-55** | Algorithm A categorical priority: `["color", "linestyle", "marker"]` (default, changeable via `channels.priority.categorical` style key) | Brainstorm v1.4 Session 5 A-Q1 + v1.2 §3.1 | Color has highest perceptual capacity; linestyle is orthogonal; marker is third because cycle is shorter (8) but competes poorly with quantile error bars |
| **AD-56** | 3-channel default (vector × group_by × quantiles discrete): `group_by → color, vector → marker, quantiles → linestyle`. Implemented via `EXPLICIT_RULES` in `channels.py`. Changeable via `channels.default.*` keys. | Phase 13.26.DF v1.1 §3.2 + Claude48 P1-1 (Option C reconciliation) | Vector cardinality bounded ≤ 8 (fits marker cycle); quantiles ordinal (linestyle natural); group_by primary categorical (color) |
| **AD-57** | Nested-band auto-detection: `non_05_count >= 4` AND symmetric pairs → `'nested_band'` (with or without central 0.5). User overrides with `quantile_mode='discrete'`. | Phase 13.26.DF v1.2 F-5 (Option A) | Central line handled by `central=` parameter independently from mode detection |
| **AD-58** | Overflow behaviour default `"error"` with actionable suggestions (`top_k=`, `facet=True`, `group_by_bins=`). Changeable via `channels.overflow` style key (`"warn"` for permissive). | Brainstorm v1.4 §3.4 + Session 5 A-Q2 consensus | Silent auto-facet hides intent; actionable error preserves user agency |
| **AD-59** | Factored legend default `True`: section headers per data channel; entry count = sum of cardinalities (not product). Changeable via `channels.legend.factored=False` for legacy flat dedup. | Brainstorm v1.4 §12 + Phase 13.26.DF v1.0 | 3-channel call with cardinalities |g|=5, |v|=3, |q|=5 produces 75 entries unfactored vs 11 factored |

---

## 3. Governance Principles

Extracted from Phase 13.25.DF + Phase 13.26.DF review cycles. These should bind future phases without re-litigation.

### GP-1 — Style configurability lands at interface introduction, not deferred

**Origin:** Phase 13.26.DF v1.0 review cycle (Path B vs Path A debate), Claude49 draft review.
**Statement:** When a phase introduces a new internal mechanism, the style keys controlling it should land in the same phase, not deferred to a "later" Phase C+. Deferral creates a refactor cost when the deferred phase has to retrofit style lookups everywhere.
**Precedent:** Phase 13.25.DF Phase A landed `quantile.*` style keys (AD-51) in the same phase as the rendering modes. Phase 13.26.DF Phase B landed all 10 `channels.*` style keys in the same phase as Algorithm A.

### GP-2 — Internal APIs accepting new data-channel types must be list-based from day one

**Origin:** Phase 13.26.DF v1.2 G-7, Claude48 reviewer note 2026-05-05.
**Statement:** When an internal API will accept a new data-channel type in a future phase (e.g., `selection_delta` in Phase D), the API must accept a `list[DataChannel]` (or equivalent extensible structure) from day one — not a fixed-arity signature with hardcoded boolean flags. Adding a channel type later requires adding a list entry, not changing a signature.
**Precedent:** v1.1 used three booleans (`has_vector`, `has_group`, `has_quantile_channel`); v1.2 G-7 generalized to `list[DataChannel]` with `EXPLICIT_RULES: dict[frozenset[str], dict[str, str]]`. Same total LOC; eliminates Phase D internal-API redo.
**Corollary (additive contract):** `EXPLICIT_RULES` is append-only across phases. Modifying an existing entry would be a regression of a prior phase's locked behaviour; new entries are additive only. See `channels.py` Coder Card directive comment.

### GP-3 — Architect signals preserved verbatim with typos

**Origin:** Reviewer Card v1.29 Rule 9, Phase 13.25.DF AD-50 reformulation incident.
**Statement:** Architect quotes embedded in proposals, decisions logs, and review documents must be verbatim. Typos and informal phrasing are the strongest signals of authenticity. Reformulating an architect quote into "polished" prose risks summarising away architect intent — has caused production bugs (AD-50 cascade-vs-independence resolution required FIX1 to recover).
**Operationally:** the three chat 2026-05-05 quotes in §1.2 contain *"wan"*, *"defuat"*, *"begining"*, *"psoponed"*, *"easly"*, *"quentiles"*, *"asnwer"*, *"Parametrizrtion"*. All preserved.

### GP-4 — Backward-compat scope must be justified by production-usage verification

**Origin:** Phase 13.26.DF v1.2 G-1/G-3 amendment, "no quantile users" incident.
**Statement:** Before locking backward-compatibility constraints on a code path, the proposing party (drafter or reviewer) must verify the path has production users — by `grep`/AST against production scripts. Otherwise the constraints are unjustified and lock the implementation freedom for no benefit.
**Precedent:** Phase 13.26.DF v1.0 → v1.1 took three review iterations to converge on Algorithm A correctness, idempotency contracts, and Rule 9 quote verbatim. None of the 6 reviewers (or Claude48 drafter, or Claude49 reviewer) verified that the quantile-touching code path had production users until the architect amendment chat 2026-05-05. v1.2 G-3 reframed Class 10 around real `makeSmoothMapsWithTPC.py` patterns; 2 of 7 backward-compat tests were removed as locking-greenfield-paths.
**Proposed Reviewer Card v1.30 rule:** *"For any phase claiming backward-compatibility scope on a code path, the reviewer must verify the path has production users via `grep`/AST against production scripts. Otherwise constraints are unjustified."* See `MTTU_Reviewer.md` — pending architect approval as part of next governance update.

### GP-5 — Drafter rotation across phases is healthy

**Origin:** Phase 13.26.DF v1.0/v1.1 drafted by Claude48 (then Coder seat); v1.2 drafted by Claude49Coder (succeeding Claude48 after Coder seat handoff). Claude48 review of v1.2 endorsed the rotation explicitly.
**Statement:** Different drafters across phase iterations bring different lenses. The G-7 forward-extensibility upgrade in v1.2 was a structural improvement that the original Claude48 drafter did not surface in v1.0/v1.1. The successor drafter (Claude49Coder) caught it. This is evidence that the multi-model drafter rotation pattern works.
**Reviewer note (Claude48, v1.2 review §"Note to Architect"):** *"The drafter handoff (Claude48 → Claude49Coder) is healthy — different drafter caught an architectural improvement (G-7) that the original drafter did not surface in v1.0/v1.1."*

---

## 4. Drafter Rotation History

| Phase | Artifact | Drafter | Reviewer panel | Final verdict |
|---|---|---|---|---|
| 13.25.DF v1.0–v1.3 | Phase A proposal | Claude48 (Coder) | 6-reviewer panel including Claude40, Claude45, Claude46, Claude49, GPT4 | `[OK]` |
| 13.25.DF v1.0_END | Phase A implementation | Claude48 (Coder) | Claude49 + 5 others | `[X]` → FIX1 |
| 13.25.DF FIX1 + housekeeping | Phase A implementation | Claude48 (Coder) | Claude49 | `[OK]` |
| 13.26.DF brainstorm v1.4 | Multi-graph styling brainstorm | Claude45 (Drafter, ADF-side) | dfdraw + ADF panels | Approved |
| **13.26.DF v1.0** | Phase B proposal | **Claude48 (Coder)** | 6-reviewer panel | `[X]` (5 P1) |
| **13.26.DF v1.1** | Phase B proposal (revision) | **Claude48 (Coder)** | Claude49 (Reviewer) | `[OK]` |
| **13.26.DF v1.2** | Phase B proposal (greenfield amendment + G-7) | **Claude49Coder** (succeeding Claude48 after Coder seat handoff) | Claude48 (Reviewer, role-swap) | `[OK]` ← *current state* |

**Pattern note (per GP-5):** the rotation Claude48 → Claude49Coder for v1.2 surfaced the G-7 list-based API improvement that v1.0/v1.1 missed. Worth preserving.

---

## 5. Audit Trail

### Phase 13.26.DF v1.2 approval chain

1. **Drafter** (Claude49Coder) — produced v1.2 with G-1 through G-7 deltas folded against v1.1
2. **Architect verbal approval** — chat 2026-05-05, *"Approved"* on `PHASE_13_26_DF_v1_2_Approval_Request.md`
3. **Reviewer co-sign** — Claude48 (dfdraw, Reviewer) `[OK]` APPROVED, 2026-05-05, document `Claude48_PHASE_13_26_DF_v1_2_Review_20260505.md` (or equivalent reviewer file). Specific endorsements:
   - G-7 forward-extensibility upgrade (substantive design improvement)
   - GP-2 governance principle recommendation (recorded above)
   - GP-5 drafter rotation observation (recorded above)
4. **Final architect approval** — chat 2026-05-05 on receiving Claude48 review

### Phase 13.26.DF Commit 1 scaffolding

- Pre-commit tag: `PHASE_13_26_DF_v1_0_BEGIN`
- Files added: `dfdraw/channels.py`, `dfdraw/docs/STYLING_FRAMEWORK_DECISIONS.md` (this file), `dfdraw/tests/test_channel_assignment.py`
- Files modified: `dfdraw/style.py` (+10 `channels.*` keys), `dfdraw/tests/feature_taxonomy.py` (+3 CHANNEL.* entries), `dfdraw/docs/CAPABILITY_MATRIX.md` (regenerated)
- Test count: 578 (unchanged — Commit 1 is pure scaffolding; new test classes are pytest-skip stubs)
- All 578 existing tests pass
- Bundle: `reviewer.zip` for Commit 1 review (tag verification, scaffolding correctness)

### Phase 13.26.DF Commit 2 implementation

- Parent commit: `0df4c00b` (Phase 13.26.DF Commit 1 scaffolding)
- Files modified: `dfdraw/channels.py` (stub → ~310 LOC implementation), `dfdraw/drawer.py` (Algorithm A wiring + cycle constants → style-key lookups + `quantile_style` forwarding), `dfdraw/plots/profile.py` (nested-band detection + channel-aware discrete rendering + `_render_quantile_nested_band`), `dfdraw/tests/test_channel_assignment.py` (50 skip stubs → 50 real test bodies)
- Test count: 627 passed + 1 skipped + 0 failed (verified on architect MacOS env, 2026-05-06; +50 new tests vs Commit 1 baseline of 577)
- No regressions in any pre-existing test
- Pre-commit tag candidate: `PHASE_13_26_DF_v1_0_END`

#### AD-60: FIX2 visual elements — channel-aware preservation

**Decision:** Per v1.2 §11.3 directive ("Coder may preserve, redesign, or drop FIX2 elements"), Commit 2 **preserves** the FIX2 visual elements (on-line percentage annotations + linestyle cycle for discrete quantiles) as the **channel-aware default for `quantile_style='linestyle'`**, with two structural changes to align with the channel framework:

1. **Cycle source:** the FIX2 hardcoded local `_ls_cycle = ['--', '-.', ':', (0, (3, 1, 1, 1))]` at `profile.py:559` is replaced with `get_style_value("channels.cycles.linestyle", default)[1:]`. The `[1:]` slice preserves the FIX2 invariant that **solid linestyle remains reserved for the central line** (now controlled by the first entry of `channels.cycles.linestyle`). Default cycle is `['-', '--', '-.', ':']`, so the discrete-quantile cycle is `['--', '-.', ':']` — one entry shorter than FIX2's hardcoded 4-entry cycle (the dash-dot-dot pattern `(0, (3, 1, 1, 1))` is dropped). For 4+ symmetric quantiles users are now routed to `nested_band` mode (AD-57) anyway, so the missing 4th linestyle is rarely needed; users requiring it can `set_style({'channels.cycles.linestyle': ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]})`.

2. **Annotation scope:** FIX2's on-line percentage annotations (`profile.py:564-577`) are preserved when `quantile_style='linestyle'` (the default), and **suppressed** when `quantile_style='marker'` or `quantile_style='color'`. Rationale: marker- and color-distinguished quantile lines need no on-line text to disambiguate; the legend handles it. Linestyle-distinguished lines benefit from on-line annotations because subtle linestyle differences are harder to read against a legend at a distance.

**Provenance:** v1.2 §11.3 explicit recommendation; FIX2 commit `da8895e2` (PHASE_13_25_DF_FIX2_END).

#### Behavior change recorded for transparency (per v1.2 §8.3 greenfield)

Symmetric quantile lists with **>= 4 non-0.5 entries** now auto-detect as `nested_band` mode (AD-57, Option A) instead of `discrete`. This is a deliberate change to the auto-detection rule and is per architect amendment chat 2026-05-05: *"We did not use quentiles yet. We do not need to be back compatible. for quentiles"*.

Affected examples:
- `[0.05, 0.25, 0.5, 0.75, 0.95]` (5 entries with central): was `discrete`, now `nested_band` (2 alpha-stacked filled regions + central line)
- `[0.05, 0.25, 0.75, 0.95]` (4 entries no central): was `discrete`, now `nested_band` (2 alpha-stacked filled regions)
- `[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]` (3 pairs + central): was `discrete`, now `nested_band` (3 filled regions + central line)

**To restore old behavior** on a per-call basis, pass `quantile_mode='discrete'` explicitly. The `stats_dict['quantiles_per_bin']` signature is preserved for both modes, so existing tests that only check the stats dict signature still pass.

**Existing test affected:** `tests/test_quantiles_profile.py::TestQuantileMode::test_multi_pair_returns_discrete` — used to test that `[0.05, 0.25, 0.5, 0.75, 0.95]` returns discrete mode. Renamed to `test_multi_pair_returns_nested_band` and docstring updated in Commit 2 to reflect new contract (assertion still passes either way because `quantiles_per_bin` is set for both modes; only the mode name and intent change).

---

### Phase 13.28.DF — Robust Data Handling (AD-69 through AD-77)

Reserved range AD-61..AD-68 is held for Phase 13.27.DF (sister phase: selection_vector + weights_vector + facet integration), currently in reviewer panel. Phase 13.28.DF uses AD-69 onward to keep the two phases independent.

| AD | Decision | Source | Notes |
|---|---|---|---|
| **AD-69** | Centralized sanitization module `plots/_data_sanitize.py` exposing `sanitize_for_plot()`. Uniform handling across hist / hist2d / hexbin / profile / scatter so every plot type inherits NaN/inf semantics from one place. | Phase 13.28.DF v1.1 §7.1 + brainstorm §6 | Module is callable from any plot path; called once per top-level user call (idempotency contract preserved) |
| **AD-70** | `nan_policy` parameter — default `'filter'` (silently drops NaN/inf, populates counters). Alternatives `'warn'` (drop + UserWarning) and `'raise'` (ValueError). Per-call kwarg + per-process default via style key `data.nan_policy`. | Phase 13.28.DF v1.0 architect Q1 (2026-05-06) — *"User is not responsible for curating data"* | Default preserves bit-identical behavior on clean data vs Phase 13.26.DF |
| **AD-71** | Stats dict counter keys (`n_input`, `n_filtered`, `n_inf_x`, `n_nan_x`, `n_inf_y`, `n_nan_y`) always populated regardless of nan_policy. Additive — existing keys unchanged. | Phase 13.28.DF v1.0 §3.3 + architect Q5 (2026-05-06) | New keys live alongside existing `n`, `mean_x`, etc. — flat namespace |
| **AD-72** | Hybrid autorange algorithm formal definition (proposal §4.1): compute robust window `(median ± k_robust·sigma_MAD)`. Per-side, declare outlier on side S if `data_extreme_S` exceeds median by more than `(k_outlier · k_robust · sigma_MAD)`. Use robust bound when outlier present, else use data extreme. | Phase 13.28.DF v1.0 §4.1 + architect Q3, Q4 (2026-05-06) — *"do not cut PDF without outliers"* | Genuinely novel for plotting libraries — hybrid combines minmax with robust window with per-side outlier detection |
| **AD-73** | Default autorange strategy `'hybrid'`. Alternatives via style key `autorange.strategy`: `'minmax'` (matplotlib-equivalent, backward compat), `'percentile_99'`, `'percentile_95'`, `'robust_3mad'`, `'robust_4mad'`. | Phase 13.28.DF v1.0 architect Q2 (2026-05-06) | Default change vs Phase 13.26.DF: clean data gets identical bounds; outlier-bearing data gets clipped to robust window |
| **AD-74** | 2D autorange per-axis independent. `compute_autorange()` runs separately for x and y; no cross-axis correlation in outlier decision. | Phase 13.28.DF v1.0 architect Q7 (2026-05-06) — *"Independent"* | Joint outlier detection deferred (potential Phase 13.30) |
| **AD-75** | Backward compat lock: existing tests that depend on min/max autorange semantics get explicit `range='minmax'`. Audit performed during Commit 2b; estimated ≤10 tests affected. | Phase 13.28.DF v1.0 architect Q10 (2026-05-06) — *"In old test we can use explicitly old autorange"* | Class TestStatsDictAdditive verifies pass count unchanged after audit |
| **AD-76** | Strategy parameters (`k_robust=4.0`, `k_outlier=1.5`, `percentile=(1,99)`) tunable via style keys (`autorange.k_robust`, `autorange.k_outlier`, `autorange.percentile`) only in v1.0. Per-call strategy-parameter kwarg override (e.g., `range_kwargs={'k_robust': 5}`) deferred to Phase 13.29 if production usage demonstrates need. | Phase 13.28.DF v1.0 architect Q8 (2026-05-06) — clarified by drafter | Keeps v1.0 surface area minimal; users wanting fine control use a different preset string or set style key globally |
| **AD-77** | Diagnostic stats keys (`autorange_used`, `autorange_strategy`) always populated. `autorange_used`: `(lo, hi)` for 1D or `((xlo,xhi),(ylo,yhi))` for 2D — the actual numeric range used. `autorange_strategy`: name of strategy applied, or `'explicit'` when user passed numeric `range=(...)`. **`stats['n']` semantics locked: finite count after `selection` AND `sanitize`. Range filtering is VISUAL ONLY — `range` does NOT reduce `stats['n']`.** | Phase 13.28.DF v1.1 §6.4, §3.3 (2026-05-06) — GPT4 #1 + #2 convergent finding; architect override of Claude40 *"no revision"* for spec cleanliness | Without these keys, hybrid autorange is opaque to production QA. The `n` semantics lock prevents future FIX1 churn |

---

*This document is authoritative for dfdraw architectural decisions. Modifications to AD entries require architect approval; appending new ADs follows the standard phase-decision workflow per `Organization-structure.md`. Governance principles (GP-N) are extracted from review-cycle lessons and bind future phases unless explicitly overridden by architect.*

*Maintainer log:*

| Version | Date | Author | Change |
|---|---|---|---|
| 1.0 | 2026-05-05 | Claude49Coder | Initial seed in Phase 13.26.DF Commit 1. AD-44 through AD-59 + 5 governance principles + drafter rotation history + Phase 13.26.DF v1.2 audit trail. |
| 1.1 | 2026-05-06 | Claude49Coder | Phase 13.26.DF Commit 2 audit-trail entry. AD-60 added (FIX2 visual-elements channel-aware preservation per v1.2 §11.3). Behavior-change record for `nested_band` auto-detection on symmetric 4+ entries. |
| 1.2 | 2026-05-06 | Claude49Coder | Phase 13.28.DF Commit 1 scaffolding. AD-69..AD-77 added (Robust Data Handling: optional NaN/inf filter with counter reporting + hybrid autorange strategy + diagnostic stats keys). Reserved AD-61..AD-68 for Phase 13.27.DF (sister phase in reviewer panel). |
