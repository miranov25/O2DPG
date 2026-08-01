# ARCHITECT_DECISIONS.md
# AliasDataFrame — Architect Decision Registry
# Version: 2.1.0 (created in PHASE_13_55_ADF per §11 item 6)
# Date: 2026-06-10
# Maintainer: Marian Ivanov (architect)
#
# This document is the canonical AD registry for the AliasDataFrame project.
# Conventions mirror dfdraw/docs/ARCHITECT_DECISIONS.md v1.1.0:
#
#   AD-N/PHASE    — per-phase counter, Phase tag matches canonical PHASE_*.md identifier
#   AD-N/BUG-ID   — bug-driven decisions
#
# Source-code citation pattern:
#   # AD-1/13.55.ADF: dispatch routes through DFDraw.draw()
#
# Rules:
#   - AD identifiers are immutable once ratified
#   - Withdrawn decisions keep their number with status: withdrawn
#   - Numbers are never reused
#   - Architect quotes preserved verbatim including typos (dfdraw GP-3)
#
# Backfill status: this registry starts at Phase 13.55.ADF. A dedicated
# reviewer round scanning all prior ADF PHASE documents (mirroring the
# dfdraw Sonnet56/Sonnet57 backfill) is proposed for AFTER Phase 13.55.ADF
# closure; legacy entries will be appended without renumbering.

---

## AD-1/13.55.ADF — Dispatch routing, error-visibility defaults, and exceptions

**Date:** 2026-06-10 | **Phase:** PHASE_13_55_ADF | **Status:** ratified

**Decision:**

1. `adf.draw()` and `adf.draw_figures()` route through `DFDraw.draw()` (Option B). The ADF `'auto'` sentinel is pre-resolved via `_resolve_plot_type` before routing (option (b)). Architectural widening: overlay strings (`'hist2d+profile'`) and type aliases (`'histo'`) are accepted at the ADF surface; any future `DFDraw.draw()` dispatch additions propagate to ADF automatically.
2. `adf.draw_figures` default `on_error` changes `'skip'` → `'raise'` (BREAKING; opt back in with `on_error='skip'`).
3. `adf.draw_batch` default `on_error` changes `'skip'` → `'raise'` (BREAKING; §11 item 4 Option A).
4. `adf.draw_batch` dispatch is not modified — it inherits the landed dfdraw Phase 13.55.DF fix transitively.
5. `_resolve_plot_type` is retained: production role = `'auto'` pre-resolution; plus `adf.draw_help()` introspection. Not the dispatch authority for explicit types.
6. **Documented exception:** `draw_fit_summary` keeps `on_error='skip'` (§11 item 5). Per-panel fit failures are data-dependent; the QA-dashboard placeholder is the intended UX.

**Ratification (verbatim, per GP-3 — typos preserved):**

> *"OK. Approve. You can strt implmnetation."*
> — M. Ivanov, 2026-06-10 (§11 items 1–4)

> *"They were approved, but if you found a bug, you found the bug."*
> — M. Ivanov, 2026-06-10 (reconfirmation of items 1–4 after A-8 finding)

> *"OK. B"*
> — M. Ivanov, 2026-06-10 (Q1 `'auto'` pre-resolution, option (b))

> *"Skip"*
> — M. Ivanov, 2026-06-10 (§11 item 5: `draw_fit_summary` keeps `'skip'`)

> *"Use only my verbatime quote."*
> — M. Ivanov, 2026-06-10 (§11 item 6: registry seeded with verbatim quotes only; no backfill in this phase)

> *"Extend the test as much as needed."*
> — M. Ivanov, 2026-06-10 (gallery G8 extension pulled into this phase)

**Source:** `PHASE_13_55_ADF_DrawFiguresAudit_Proposal_v1_2.md` §0, §3, §11; coder Q&A session 2026-06-10.

**Code citations:**
- `AliasDataFrame.py` — `adf.draw` dispatch site (route-through + 'auto' pre-resolution)
- `AliasDataFrame.py` — `adf.draw_figures` dispatch site (route-through + 'auto' pre-resolution + scatter3d guard)
- `AliasDataFrame.py` — `draw_figures` / `draw_batch` signature defaults
- `AliasDataFrame.py` — `draw_fit_summary` signature (unchanged, exception per item 6)

---

## AD-2/13.56.ADF — Batch type shims; figures guards; D4 error; help; R-1 rescope

**Date:** 2026-06-11 | **Phase:** PHASE_13_56_ADF | **Status:** ratified

**Decision:**

1. **Amends AD-1/13.55.ADF item 4 scope:** `adf.draw_batch` gains per-spec type shims (literal `'auto'` pre-resolution; 3-var `'profile'`→`'profile2d'` promotion) in the ADF wrapper loop, **pre-delegation** — the transitive-inheritance rationale of AD-1 is preserved (dispatch itself remains dfdraw's). (D1=A)
2. `draw_figures` guards for `type='profile2d'` and `facet_by` in specs: A-10 semantics (raise default; labelled `[ERROR]` placeholder under explicit skip). **Both temporary**, removal tied to `BUG_dfdraw_20260611_profile2d_ax_ignored` / `BUG_dfdraw_20260611_facet_by_ax_ignored` (dfdraw next steps: honour `ax=`; nested sub-gridspec). (D2=B, D3)
3. Alias-eval `astype(int)` class gets a clean actionable error naming the quoted dtype form; eval-namespace lambdas unchanged. Deferred feature: EXPR.astype_type_tokens. (D4=A)
4. Gallery extended by `weights=`, `on_error='skip'` placeholder demonstration, `entry_*` window — mandatory 36→39. (D5: "More is better; I do not want to be surprised in production.")
5. `draw_help()` lists the live-introspected type surface incl. aliases and overlay syntax; deferred: HELP.live_introspection. ("I prefer to get full help.")
6. R-1 rescoped to regression lock + docstring (panel M-1: no crash exists at HEAD; FM-Probe-1 candidate recorded).
7. `keep_materialized` hook-alias residue (panel P3-1): disposition (a) — TS qualification; follow-up item CLEANUP.hook_alias_tracking.

**Ratification (verbatim, GP-3):**

> *"D1: A · D2: B · D3: I think that will be fixed in dfdraw later. We should add a comment about the next step in dfdraw. · D4: Not sure what is safer. I assume we can do more expression evaluation in ADF. · D5: More is better; I do not want to be surprised in production."* — M. Ivanov, 2026-06-11
> *"R-1- Confirmed"* — M. Ivanov, 2026-06-11 (subsequently rescoped per panel M-1)
> *"I prefer to get full help."* — M. Ivanov, 2026-06-11
> *"Approved. Plase start coding"* — M. Ivanov, 2026-06-11 (v1.2 GO)

**Source:** PHASE_13_56_ADF_PostAuditFixes_Proposal_v1_2.md §0; v1.0/v1.1 consolidated review summaries.


## AD-3/13.59.ADF — Metadata read/write precedence and back-compatibility

**Date:** 2026-06-15 | **Phase:** PHASE_13_59_ADF | **Status:** ratified

**Decision:**

1. **Reading precedence** (first source that succeeds wins):
   1. **ROOT UserInfo** read — if ROOT is available.
   2. **uproot UserInfo** read (`uproot.open(..., minimal_ttree_metadata=False)`) — the ROOT-free path.
   3. **Standalone key** `<tree>__adfmeta__` (`TObjString` JSON) — read via ROOT if available, else uproot.
   4. **Names reconstruction** from `<tree>__subframe__*` sibling key names — structure only; carries `schema_source='names_only'` + warning; never overrides 1–3.
2. **Writing precedence:** ROOT writes **UserInfo** whenever ROOT is available (primary, trusted, keeps old-reader compatibility); uproot writes the **standalone key only** when ROOT is absent (uproot cannot write UserInfo). **Dual-write** is reserved as a future opt-in (`dual_write=False` default), out of scope for 13.59.
3. **Standalone-key convention:** `<tree>__adfmeta__`, a `TObjString` of the schema JSON, sibling to the tree — mirrors the existing `<tree>__subframe__<name>` convention.
4. **Backward compatibility is paramount:** decade-old UserInfo files must remain readable indefinitely; no path silently loses or fabricates metadata (failure is explicit).

**Mechanism note (panel-unanimous interpretation, architect-confirmed):** "up-to-date UserInfo read" (level 2) is the **uproot** read using `minimal_ttree_metadata=False`; ROOT's default skips UserInfo, so this flag is required. Named explicitly here per reviewer request (Sonnet2, Sonnet10).

**Evidence on file (2026-06-13):** uproot reads old UserInfo via `minimal_ttree_metadata=False`, validated on real `calibITS.root`/`calibTRD.root` with and without ROOT. The lazy/uproot read path previously did **not** read UserInfo (`read_tree_lazy(calibITS).lazy_subframes == []` vs UserInfo `['R','AlignDzITS5']`); PHASE_13_59_ADF closes that gap.

**Architect specification (verbatim, per GP-3 — typos preserved):**

> *"We should be backward compatible. We should use different options for metadata reading.*
> *Order:*
> *1. Use ROOT UserInfo read if available.*
> *2. Use up-to-date UserInfo read.*
> *3. If that fails, use new metadata as a standalone key, using ROOT if it exists.*
> *For writing, always use ROOT if available.*
> *I am not sure about uproot functionality, so I do not want to rely on it for writing unless I have no other option."*
> — M. Ivanov, 2026-06-13

> *"We should bea be to read data 10 years old ...."* — M. Ivanov, 2026-06-13 (motivation)

**Ratification (verbatim, GP-3):**

> *"append AD-3"* — M. Ivanov, 2026-06-15 (GO)

**Source:** `PHASE_13_59_ADF_MetadataBackCompat_v1_2_Proposal.md` §3; v1.0–v1.2 consolidated review summaries; implementation CRR v2 panel summary ([OK], 2026-06-15).

**Code citations:**
- `AliasDataFrame.py` — `read_tree_lazy` :6049–6128 (lazy read path; subframe registration added here)
- `AliasDataFrame.py` — `lazy_subframes` :2219; eager UserInfo read :5714–5719; ROOT UserInfo write :5581
- `LazyTreeReader.py` — `__init__` (resolver call; fix site, Option a)
- `adf_metadata_compat.py` — read-precedence resolver (levels 1–4)

---

## AD-4/13.76.ADF — Symmetry-by-default classification rule (draw path and beyond)

**Ratified:** 2026-07-19 (PHASE_13_76_ADF Stage A) — applies to all Stage-A
discoveries and to later phases unless superseded.

**Rule (architect's words, verbatim):**

> *"The intended design is full symmetry for every semantically applicable
> feature. When Stage A finds an asymmetry between surfaces, forms, slots, or
> equivalent plot paths, it should normally be classified as Repair, not
> Refused. A behavior may be classified as Refused only when the operation is
> genuinely semantically inapplicable or when I explicitly approve that
> limitation. Existing implementation asymmetry by itself is not evidence of
> intended refusal."* — M. Ivanov, 2026-07-19

> *"Use symmetry as the default intended contract, but do not assume it is
> already implemented. [...] Any observed asymmetry is automatically a Repair,
> unless the architect explicitly approves an exception. ADF should not
> duplicate dfdraw logic merely to protect against every dfdraw defect.
> Stage-A tests should still detect and record asymmetries so they cannot
> disappear silently. dfdraw owns and fixes dfdraw-level asymmetries; ADF only
> guards its own preparation, projection, and delegation contracts."*
> — M. Ivanov, 2026-07-19

**Operational consequences:**
1. Matrix classification default: observed asymmetry → `Repair` (deferred if
   owner is dfdraw, per R-4). `Refused` requires genuine semantic
   inapplicability or explicit architect approval; proposed permanent
   exceptions stay `Unspecified` until approved.
2. No current asymmetry is frozen as a compatibility contract by mere
   existence.
3. ADF ships contract tests + ownership tracking, not defensive duplication
   of dfdraw logic.

**Applied same day:** SEED-1.a (`bins`×scatter warn-and-ignore) → Repair
DEFERRED owner=dfdraw (acceptance semantics pending architect Q1);
SEED-2.a (capacity checked pre-`top_k`) → Repair DEFERRED owner=dfdraw.
Deferred acceptance tests: `test_K1_3_draw_batch_forwards_batch_kwargs`,
`test_K2_3_production_reproducer_mirror` (both strict xfail, reason-linked
to this AD). Current behavior pinned in
`tests/test_phase_13_76_draw_path_characterization.py`.

**Source:** architect chat rulings 2026-07-19 (two messages, quoted above);
`DRAW_PATH_BEHAVIOR_MATRIX.md` Part I §2.

---

## AD-5/13.76.ADF — `bins` × scatter: semantic-inapplicability exception (ratified)

**Ratified:** 2026-07-19. Panel: dfdraw MainReviewer synthesis (Sonnet5_4,
5 seats, 4/5 convergence with MainReviewer conceding); architect: *"Yes.
GPT consensus proposal looks reasonable, I wanted to be sure that it has no
effect. Lets go ahead"* after corpus back-compat check (0 affected usages
in examples/time_series).

**Ruling (panel draft, architect-approved verbatim):**

> Confirmed as a semantic-inapplicability exception. Shared batch-level
> `bins=` is silently inapplicable to `type='scatter'` (no warning); an
> explicitly supplied `bins=` on a scatter spec or direct call raises a
> clean, actionable error instead of the current warn-and-ignore. dfdraw
> issue to be filed and scheduled as a normal Repair; also registered as a
> seed item for a future dfdraw applicability/dispatcher-consistency phase.

**Encoded by:** deferred acceptance tests `test_K1_3_...` (shared-silent
contract) and `test_seed1_2_...` (explicit-error contract), both strict
xfail; current behavior pinned by `test_seed1_1_...`.

---

## AD-6/13.76.ADF — `draw_figures` × caller `ax`: clean refusal (ratified)

**Ratified:** 2026-07-19, same panel and architect approval as AD-5.

**Ruling (panel draft, architect-approved verbatim):**

> Confirmed as a semantic exception. `draw_figures` rejects caller-supplied
> `ax` (top-level or per-spec) with a clean, immediate `ValueError` before
> any figure/axes creation, replacing the current raw `TypeError`. A future
> `figure=`/`axes=` contract may be proposed separately for genuine
> multi-axes composition; not part of this ruling.

**Ownership note (coder, anchor-based):** the collision site is ADF's own
composer (`_draw_single_figure`), so implementing the refusal is an
ADF-owned Repair landing in Stage-B spec validation — not a dfdraw filing.
**Encoded by:** deferred acceptance `test_seed3_7_...` (strict xfail);
current crash pinned by `test_seed3_5_.../test_seed3_6_...` (retired when
the Stage-B fix lands).

---

## AD-7/13.76.ADF — subframe projection preserves the user's dtype (ratified)

**Ratified:** 2026-07-28, architect (Marian Ivanov), B3.2 part 2 correction
round 4. Supersedes nothing; states a rule that had never been written down and
was therefore violated by a correction.

**Background.** B3.2 routed subframe projection through
`_extract_subframe_values_cached` to fix a missing-key defect. That helper
allocates `np.full(n, np.nan, dtype=np.float64)` for every non-floating source
column, so `int`/`bool` were silently coerced, `datetime64` became floating
epoch nanoseconds, and `object`/`category` raised `could not convert string to
float`. Four GPT reviewers executed and confirmed it independently.

**Ruling (architect, verbatim in substance):**

> We have to keep the dtype as specified by the user. We do not want to explode
> in memory. In the past we provided a recipe for default values on failure. We
> should never change a type specified by the user. We rely on dtypes. We rely
> on default values in case of failures — you can find them in the public
> interface.

**Encoded rule.**

1. **All keys matched** — gather directly in the source dtype. No replacement
   array is allocated at all.
2. **Keys missing, and the dtype has a native missing value** — use it, because
   it changes neither dtype nor memory: `NaN` for floating, `NaT` for
   `datetime64`/`timedelta64`, `None` for `object`, the native code for
   `category`.
3. **Keys missing, `int` or `bool`** — these have no native missing value
   (measured: `int64` + NaN becomes `float64`; `bool` + NaN becomes `object`
   and grows 3 → 24 bytes). The project settled this years before the phase and
   the coder did not find it until the test suite said so: the JOIN layer
   yields `NaN`, and the user's declared dtype is restored at the ALIAS layer by
   `_safe_dtype_cast` — the "recipe for default values on failure" the ruling
   refers to. It fills `0` / `False`, preserves the dtype and warns
   (`add_alias(dtype=..., fill_value=...)`; pinned by
   `test_A5_missing_child_key`, `test_D1_int8_dtype_preserved_through_join`,
   `test_D2_bool_dtype_preserved_through_join`, all predating this phase).

   **Recorded because it is the ruling's own point:** the coder's first
   implementation RAISED here instead, inventing a policy where one already
   existed. Three long-standing tests failed and that is how it was found. The
   ruling says reuse the existing contract; the first draft did not, and no
   amount of care in the new tests would have caught it, because the new tests
   were the coder's and the contract was not.

**Existing public contract reused, not replaced:** `set_global_fill(...)` /
`set_subframe_fill(name, ...)` — `fill_missing` (documented: "if None, missing
keys produce NaN"), `fill_nan`, `fill_inf`, `fill_invalid`,
`warn_missing_keys`, `warn_threshold`, `fill_mode`. No new missing-value policy
was invented.

**Explicitly forbidden — and this wording was ambiguous in v1.5.0, corrected
here after GPT25 flagged the contradiction.** ADF must not *invent* a nullable
extension dtype (`Int64`, `boolean`, …) to solve a missing-value problem, nor
silently upcast, fall back to object, add a validity mask, or allocate a
conversion buffer only to represent missingness. It does **not** mean ADF
rejects an extension dtype the user themselves stored in the child frame —
those are preserved like any other user dtype. See AD-11.

**User-visible consequence:** none for `int`/`bool` — the established
NaN-then-`_safe_dtype_cast` contract is unchanged. What changes is that
`object`, `category`, `datetime64` and `timedelta64` subframe columns now work
where they previously raised `could not convert string to float` or were
silently converted to floating epoch values, and that a fully-matched join of
ANY dtype now returns the caller's exact dtype with no allocation.

---

## AD-8/13.76.ADF — duplicate-owner graphs refused before any effect (ratified)

**Ratified:** 2026-07-28, architect. Extends AD (D2, 2026-07-27), which is
retained: the rule stays **graph-local**.

**Background.** Registration-time refusal was defeated three times by
registration ORDER, most recently by attaching two parents to a root and only
then giving each a child that is the same object. A frame has no back-reference
to its parents, so it cannot know it is already reachable from a common root.

**Ruling.** Keep the registration-time check as an early, friendly refusal.
Additionally validate the COMPLETE graph reachable from the root at
`draw_batch()` entry, **before any effect** — before branch loading, alias
materialization, joins, cache mutation or cleanup-candidate construction.
Reject when one `AliasDataFrame` object is reachable by two distinct logical
owner paths, and name BOTH paths in the error. Validation at the point of
consumption is order-independent by construction.

**Constraints on the validator, stated because the architect asked
specifically about side effects:** read-only; graph-local; terminates on real
cycles; the single-name self-registration cycle contract is preserved; the same
object in two DISCONNECTED graphs stays legal, with the documented consequence
that those graphs share mutable state (not an enforced read-only mode). Tests
must prove the validator itself changes no frame, reader, alias, cache or
preparation record.

---

## AD-9/13.76.ADF — B3.2 / B3.2b increment boundary (delegated)

**Ratified:** 2026-07-28, architect, as an explicitly DELEGATED decision.

**Ruling.** The packaging of the broader registration-policy work is a
technical review decision and may be settled by the reviewers. It must be
technical only: it may not narrow functionality or weaken any user-facing
contract.

**Not deferrable under any packaging:**

1. `draw_batch()` refuses an ambiguous duplicate-owner graph before any effect.
2. Projection preserves the exact user dtype and the existing fill semantics.
3. Cleanup and failure-state records are truthful.

Broader registration-time enforcement across every topology-mutation order and
every graph-consuming public API may move to a separately reviewed B3.2b.

---

## AD-10/13.76.ADF — compound failure: the original error stays primary (ratified)

**Ratified:** 2026-07-28, architect, on the coder's Option 1 as drafted by the
panel and confirmed verbatim.

**Ruling (verbatim in substance):**

> When the original phase fails and cleanup also fails: re-raise the original
> exception as the user-visible primary failure; preserve `failure_phase` as
> the phase that originally failed; set `cleanup_outcome="failed"`; retain the
> cleanup exception as secondary diagnostic evidence. The cleanup failure must
> not replace the original failure. The original preparation, projection,
> normalization, or rendering error is what caused the call to fail and is
> normally what the user must debug. Cleanup failure is important but
> collateral.
>
> *Implementation caution:* do not use exception chaining in a way that falsely
> presents cleanup as the cause of the original failure. Prefer preserving the
> original traceback and attaching the cleanup exception through an explicit
> note or dedicated secondary-exception field.
>
> Also test the inverse case separately: when rendering succeeds and cleanup
> alone fails, the cleanup exception is naturally the primary exception and
> `failure_phase="cleanup"`.

**Encoded by:** `_DrawPreparationState.secondary_error` (a dedicated field, not
`raise ... from`, precisely because chaining would read as causation);
`test_b32_118` (compound) and `test_b32_119` (inverse).

**Also encoded:** `aliases_dropped` is measured in a `finally`, so a cleanup
that drops one candidate and then raises records the drop that really happened
(`test_b32_117`). A record that omits a real effect is the same class of
falsehood as one that invents an effect.

---

## AD-11/13.76.ADF — AD-7 applies to the dtypes the user supplied (ratified)

**Ratified:** 2026-07-28, architect. Resolves the ambiguity GPT25 found in
AD-7 and supersedes its "forbidden" clause's reading.

**Ruling (verbatim in substance):**

> AD-7 applies to all supported input dtypes already present in the child
> frame, including timezone-aware datetime, nullable `Int64`, nullable
> `boolean`, `string[python]`, category, period, compatible interval types, and
> NumPy numeric and complex dtypes.
>
> The statement that nullable extension dtypes are "forbidden" means that ADF
> must not silently invent or convert ordinary user data into nullable
> extension dtypes merely to solve missing-value representation. It does not
> mean that ADF rejects extension dtypes that the user explicitly supplied.
>
> ADF preserves the dtype stored by the user whenever that dtype can represent
> the projected result. ADF does not automatically promote ordinary NumPy
> dtypes to pandas nullable extension dtypes unless separately specified.
>
> **Interval:** if inserting a missing value would necessarily change the
> interval subtype, projection refuses the reference with a clear ADF-owned
> dtype error naming the reference and source dtype. A fully matched interval
> join must still preserve the exact interval dtype.

**Implementation note that matters more than the list.** The architect asked
whether the framework was being symmetrized or whether we were "using a special
`if` for each particular case". We were: the previous implementation branched on
datetime, category and object, and this round's findings would have added
complex, timezone-aware, nullable-extension and interval branches to it. Three
of five B3.2 review rounds landed in the dtype domain for that reason —
pandas' dtype surface is larger than any hand-maintained list.

The implementation is therefore ONE symmetric primitive,
`pandas.api.extensions.take(arr, idx, allow_fill=True[, fill_value=])`, which
already speaks our `-1` missing sentinel. Measured across seventeen dtypes it
preserves every one the panel found broken, handles the empty-child table with
no special case, and owns fill representability (a categorical fill that IS one
of the categories is accepted; one that is not raises — closing GPT26's P1 by
construction). Exactly two rules sit on top: a dtype change on an integer or
boolean source is the ratified `_safe_dtype_cast` contract and is accepted; any
other dtype change is refused.

**Encoded by:** `_extract_subframe_values_typed`, and a STANDING GENERATED
matrix — `DTYPE_CASES` × matched / missing / empty-child / entry-selection
(`test_b32_102`–`106`). Adding a dtype to the table exercises every
combination automatically, so coverage is a property of the table rather than
of what anyone thought of on the day (Sonet29's structural finding).

---

## AD-12/13.76.ADF — increment boundary settled: B3.2 closes, B3.2b owns the full dependency plan (ratified)

**Ratified:** 2026-07-28, architect, on GPT31's decision set, approved by every
GPT seat and by Fable. Settles the question AD-9 delegated and the question
carried open in the round-5 CRR §6.

**Ruling (verbatim in substance):**

> Finish the current B3.2 correction. Create a separate increment **B3.2b**
> for the full dependency-plan contract — branches, aliases, structs,
> subframes, joins, temporary columns, persistent columns, cache changes and
> cleanup. B3.2b must complete before B3.3 begins.

**What this closes.** GPT27 held that `_DrawDependencyPlan` being an
intermediate carrier rather than the full Rev-2 plan contract independently
blocks B3.2 closure. It no longer does: the deliverable is assigned to a named
increment with an ordering constraint, so approving B3.2 cannot silently
convert the intermediate carrier into the normative plan. The coder asked for
this ruling in three consecutive CRRs; it is now on the record.

**Challenged and upheld (round 6 review).** GPT27 read the ruling as making
B3.2b a mandatory *sub*-increment, so that B3.2 itself could not close. The
Main Reviewer (Sonet29) checked the primary ratified text rather than any
paraphrase and did not uphold the objection: the only sequencing constraint in
what was approved is **B3.2b before the `draw()` / `draw_figures()`
migration**. Nothing in the ratified text prevents B3.2 closing once the
current correction is finished. Recorded here so the question is not reopened
from a paraphrase a third time. Three of four GPT seats accepted AD-12 as
written.

---

## AD-13/13.76.ADF — integer and Boolean columns are not silently widened (ratified, with a scope boundary)

**Ratified:** 2026-07-28, architect (GPT31 Decision 2).

**Ruling (verbatim in substance):**

> Preserve integer and Boolean columns. When all join keys match, the exact
> dtype is preserved. When keys are missing, use the existing safe fill rules
> where possible. When the value cannot be represented, raise a clear ADF
> error. Do not silently change an integer column to `float64` or a Boolean
> column to `object`.

**SCOPE BOUNDARY, raised by the coder before implementation and confirmed.**
Applied literally to the shared gather, this ruling would have contradicted a
contract ratified in April and pinned by three tests that predate this phase:
`test_A5_missing_child_key` (13.65), `test_D1_int8_dtype_preserved_through_join`
and `test_D2_bool_dtype_preserved_through_join`. That contract is: at the JOIN
layer a missing key on a plain NumPy int/bool column yields NaN, and the user's
declared dtype is restored at the ALIAS layer by `_safe_dtype_cast`, which
fills `0`/`False`, preserves the dtype and warns.

Accordingly AD-13 governs:

1. **every matched join** — the exact dtype is preserved (was already true for
   plain int/bool; was NOT true for `Float64`, `Float32` or `Sparse[...]`, see
   below);
2. **every join with a configured fill** — the fill is placed in the column's
   own dtype or refused;
3. **extension and sparse integer dtypes**, which have no alias-layer
   restoration contract.

It does NOT override the April contract for a plain NumPy int/bool column with
a missing key and NO configured fill. That behaviour is documented, tested and
ratified — therefore not *silent*, which is what the ruling forbids. Changing
it would also mean a missing TPC map key draws a real `0` instead of leaving a
gap in the plot.

**Defects this closed, all reproduced on the round-5 bytes before the fix:**

| case | before | after |
|---|---|---|
| `Float64` / `Float32`, **fully matched** | `object` | preserved |
| `Sparse[float64]`, matched or missing | densified `float64` | preserved |
| `Sparse[int64]` + missing key | dense `float64` — densified AND widened | refused, with the remedy named |
| `bool` + `fill_missing=0` | `object` holding `[True, False, 0, False]` | `bool` |

**Root cause, and why it is the same root cause as AD-11.** The router asked
`dtype.kind == 'f'`. `.kind` is defined on pandas ExtensionDtypes too:
`pd.Float64Dtype().kind` and `pd.SparseDtype(np.float64).kind` are both `'f'`,
and `pd.SparseDtype(np.int64).kind` is `'i'`. A predicate that looked general
was a per-dtype assumption in disguise. Replaced by
`_is_plain_float_dtype()` — `isinstance(dtype, np.dtype) and
np.issubdtype(dtype, np.floating)` — which asks the question that actually
matters: is this a real NumPy float buffer that holds NaN natively?

**Encoded by:** `_is_plain_float_dtype`; `DTYPE_CASES` extended with
`Float64`, `Float32`, `sparse_float` and `DTYPE_REFUSED_ON_MISSING` with
`sparse_int` (four generated combinations each);
`test_b32_120`–`121`, `test_b32_127`–`128`.

---

## AD-13a/13.76.ADF — SUPERSEDED BY AD-19

**Status: SUPERSEDED, same day it was recorded.** The architect's words:

> The current AD-13a policy — widen now; emit a `FutureWarning`; make it an
> error later — **was not my decision and contradicts AD-19.** Please correct
> or supersede AD-13a.

AD-13a was written from the coder's three-option summary of a 3-2 panel split,
and the architect chose "Option 3" from that summary. His ratified AD-19 shows
the summary itself was wrong: warning-and-widening is not a permitted
transitional state, because the widening can change values, and no amount of
warning makes a changed measurement acceptable.

**What survives from AD-13a: NOTHING.** Round 9 kept the notice mechanism in a
narrowed form — lossless widenings were still allowed and reported. AD-19's
operational definition removed even that: every existing column dtype is
authoritative, so the widening is forbidden whether or not the numbers happen
to survive it, and a function whose purpose is to widen-and-warn has no ruling
left to stand on. `_warn_direct_slot_widening` is **deleted**, not disabled —
GPT30 and the Main Reviewer both flagged that leaving it callable after a
ruling saying "never widen-and-warn" was a contradiction inside the round's own
record. `test_b32_154` asserts its absence.

**What does not survive:** the claim that this is a warned transition for
explicitly supplied dtypes, and the round-8 warning text asserting "the VALUE
is correct", which was false above 2**53 and had not been verified for the
call it was printed on.

The original entry is kept below for the record.

---

### Original AD-13a text (superseded)

**Recorded:** 2026-07-29, from the coder's summary of a 3-2 panel split.
Amends AD-13; does not replace it.

**The split.** GPT25, GPT26 and GPT27 read Decision 2 (*"do not silently
change an integer column to `float64` or a Boolean column to `object`"*) as
forbidding the widening outright on the **direct** slot path — a physical
int/bool child column used in `group_by` / `color` / `facet_by` with no alias,
where `_safe_dtype_cast` never runs — and asked for a clean refusal. GPT31 and
Fabble5_7 held that the widening IS the missing-ness, and that refusing would
convert the normal state of calibration data into an error.

**Ruling (Option 3 of the three the coder put to the architect).** Both sides
are answering different questions, and they are separated:

| aspect | ruling |
|---|---|
| the **value** | stays a gap. A TPC point with no matching ITS point has no integer to show; ADF does not invent one. |
| the **dtype change** | is **reported**, per column per call, with a `FutureWarning`. Silence was the only thing Decision 2 actually forbade, and silence is what ends here. |
| the **future** | the notice states that this **will become an error**. The dissenting position is the committed destination, not a rejected one. |
| the **remedy** | `set_subframe_fill(<sf>, fill_missing=<value>)`, which already preserves the exact dtype today and is exactly the migration the future error will require. |

**Scope — load-bearing.** The notice fires ONLY where no restoration layer
exists downstream: the single direct-projection call site in `draw_batch`. On
the alias path `_safe_dtype_cast` restores the user's declared dtype and
already warns; a second notice there would be noise about a solved problem,
and would train users to ignore the message that matters.

**Backward compatibility:** none broken. Every script that produces a figure
today still produces it, with one additional warning line per affected column
per call.

**Encoded by:** `_warn_direct_slot_widening`, the `direct_slot=` flag threaded
from the draw projection through `_extract_subframe_values_cached` to
`_extract_subframe_values_typed`; `test_b32_154`-`158` (warns and keeps the
gap; matched does not warn; the named remedy silences it; the alias path does
not warn; a float column never warns).

**Rationale:** the physical reasoning is AD-19 (pending — the architect will
supply it in the next round, and the CRR must request it explicitly).

---

## AD-14/13.76.ADF — a fill is compatible with the COLUMN, not with a fixed type list (ratified)

**Ratified:** 2026-07-28, architect (GPT31 Decision 3).

**Ruling (verbatim in substance):**

> `set_subframe_fill()` accepts any fill value compatible with the actual
> column dtype: numeric, string, timestamp/NaT, complex, or an existing
> category. It never silently adds a category. It raises clearly on
> incompatibility.

**Implementation, symmetric by construction.** The numeric-only `TypeError` is
removed from `set_subframe_fill()` and `set_global_fill()`, which now reject
only containers — a fill is configured per SUBFRAME while dtypes are per
COLUMN, so compatibility is not decidable at configuration time. It is decided
at projection by `_coerce_fill_to_dtype()`, one primitive for every dtype:

```python
pd.array([fill], dtype=column_dtype)      # pandas owns representability
```

with two rules on top — it raises, or the value does not survive the round
trip. The round-trip rule is what makes it useful rather than decorative,
because several coercions succeed while changing the value:

| fill | column dtype | coerces to | verdict |
|---|---|---|---|
| `0` | `bool` | `False` | **accepted** — this is the defect that closed |
| `2` | `bool` | `True` | refused |
| `'x'` | `bool` | `True` | refused |
| `1.5` | `int64` | `1` | refused |
| `'a'` | `category[1,2,3]` | `NaN` | refused — a category is never added |

**Known strictness, recorded rather than left to be discovered:** a STRING
spelling of a timestamp (`'2020-01-01'` for a `datetime64[ns]` column) is
refused, because `pd.Timestamp('2020-01-01') == '2020-01-01'` is False in
pandas and ADF does not guess at date parsing. The error says to pass a
`pd.Timestamp` / `pd.NaT`.

**Encoded by:** `_coerce_fill_to_dtype`; `test_b32_122`–`126`, `129`.

---

## AD-15/13.76.ADF — sparse and pyarrow-backed dtypes (ratified)

**Ratified:** 2026-07-28, architect (GPT31 Decision 4).

**Ruling (verbatim in substance):**

> Sparse columns: preserve sparsity when all keys match; preserve it with
> missing keys only when the missing value is representable; otherwise raise a
> clear error. Never silently densify, and never silently widen
> `Sparse[int64]` to `Sparse[float64]`.
>
> pyarrow-backed dtypes: full support belongs to B3.2b. Until then, preserve
> them where the native operation already works, and refuse clearly where it
> does not.

**Encoded by:** the same routing predicate as AD-13; `sparse_float` in
`DTYPE_CASES`, `sparse_int` in `DTYPE_REFUSED_ON_MISSING`; `test_b32_127`
(refusal) and `test_b32_128` (the escape hatch the error message names —
`set_subframe_fill(fill_missing=0)` — actually works).

---

## AD-16/13.76.ADF — the slot symmetry requirement (recorded now, executed in B3.3–B3.5)

**Ratified:** 2026-07-28, architect, on GPT31's decision set.

**Ruling (verbatim in substance):**

> Every expression slot — `expr`, selection, `group_by`, `facet_by`, color,
> weights, and the vector selections — must go through ONE common policy, with
> an explicit decision per slot: prepared in ADF, passed unchanged to dfdraw,
> or rejected clearly. Record the requirement now; do not refactor now.
>
> * **B3.3** — all three public surfaces (`draw`, `draw_batch`,
>   `draw_figures`) use the common policy.
> * **B3.4** — the duplicated per-slot logic is deleted.
> * **B3.5** — the slot × surface matrix is proven.

**One exception, executed immediately by explicit instruction:** `facet_by`
was missing from `_parse_expr_aliases` altogether, and from all three of its
call sites. The architect ruled this "an existing `draw_batch()` defect, not
future work — fix it NOW". Consequence of the defect: a facet alias was
excluded from the single bulk `materialize_aliases()` call that `draw_batch`
exists to make, and was materialized afterwards one at a time.

**Why the defect existed at all, which is the argument for AD-16.** The slot
list in `_parse_expr_aliases`'s signature was hand-written, while
`_EffectiveDrawSpec.SCALAR_SLOT_NAMES` already held the canonical one. Two
lists, one of them silently short. The branch-requirement scan
(`required_branch_kwargs`) derives from `SLOT_NAMES` and was therefore
correct — the same information, one derived and one copied, and only the
copied one was wrong.

**Encoded by:** `facet_by=` on `_parse_expr_aliases` and its three call sites;
`_FACET_BY_CHANNEL_ENUMS` promoted to a single class-level definition shared
with `_ensure_vector_kwargs_aliases`; `test_b32_135`–`138`.

---

## AD-17/13.76.ADF — an index level IS a join key, WHEN it agrees with any same-named column (RATIFIED)

**Status: RATIFIED 2026-07-29 by the architect**, on the four-state rule and the missing-value addendum, after two rounds in which every seat that commented approved the content. Implemented in correction rounds 6–7 and
submitted for the architect's decision. GPT27 and GPT30 both flagged that the
round-6 wording called this "ratified by implementation" in its heading while
its body said "submitted for ratification", and that it was not part of
GPT31's four-decision set. **Implementation cannot ratify a decision.** The
heading is corrected and the entry stays PROPOSED until the architect rules.

**Proposed rule.** A join key held as an INDEX LEVEL is the same key as one
held in a column of the same name — *provided the two carry the same values*.

- key present only as a column → use the column;
- key present only as an index level → use the level;
- present as **both, with equal values** → one key, use either;
- present as **both, with different values** → **refuse**, before any effect,
  naming the subframe, the side, and both value samples.

**Why the last line exists, and why it is the coder's own regression.** Round 6
implemented the first three cases and *asserted* the fourth could not matter.
Four reviewers executed it independently:

```
child index k=[0,1,2], child column k=[2,1,0], parent k=[0,1,2]
projected: [30.0, 20.0, 10.0]      silently reversed, no error
```

Checked against the pre-phase baseline `c73f0c99`, pandas itself had been
**refusing** this shape:

```
ValueError: 'k' is both an index level and a column label, which is ambiguous.
```

The round-4 ambiguity normalization — added to make `pre_index=True` work,
where the two spellings *always* agree — rebuilt both key tables from column
values and thereby removed that refusal for the case where they do not. So the
silent wrong join was introduced by this phase, and AD-17 then documented the
surviving column-precedence as though it were verified. The invariant is now
checked rather than assumed, on **both** sides of the join, at registration and
again at graph consumption (a frame can be re-indexed after registration).

**Also under this entry (GPT31 P1, round 7):** `pre_index=True` on a child
carrying a MultiIndex, joined on a SUBSET of its levels, raised a bare
`KeyError: "None of ['a'] are in the columns"`. The old test compared the whole
index-name list against the requested keys instead of asking, per key, whether
that key is reachable. It now asks per key, and a partial MultiIndex is a
supported shape.

**Encoded by:** `_join_key_values`, `_key_arrays_equal`,
`SubframeRegistry.add_subframe`; `test_b32_133` (drop × pre_index ×
indexed-before), `test_b32_148`–`153` (column-only / index-only / both-equal /
both-reversed / both-disjoint × pre_index, parent side, registration-time
refusal, partial MultiIndex, genuinely absent key).

---

## AD-17/13.76.ADF addendum (round 8) — the missing-value rules for key equality

**Status:** part of AD-17, **RATIFIED 2026-07-29** together with it.

Round 7 stated the four-state rule and implemented the comparison with
`a == b` followed by a boolean reduction. For any nullable or object array
containing `pd.NA`, `a == b` yields `pd.NA` at those positions and the
reduction raises

```
TypeError: boolean value of NA is ambiguous
```

out of a public `register_subframe()` — the raw pandas error AD-17 promised
not to produce. Found independently by GPT25, GPT26, GPT27 and GPT31 (**4/4**),
and reproduced for `object`, `string`, `boolean` **and** `Int64`; only the
plain float `NaN` control survived, which is why the round-7 matrix passed.

**The rules, now explicit:**

| positions | verdict |
|---|---|
| both representations missing | **equal** — a missing key matches no child row under either spelling, so the two describe the same join |
| missing on one side only | different — refuse |
| neither missing | compare, on the non-missing subset only; a nullable comparison value never reaches a boolean reduction |

**The spelling of "missing" is deliberately not significant.** `None`,
`np.nan`, `pd.NA` and `pd.NaT` are all gaps, and pandas normalises between
them freely — a `set_index()` round trip can turn `None` into `NaN` without the
user asking. Refusing on the spelling would reject a frame that joins
identically either way. *(Decission KEY-MISSING-SPELLING, architect
2026-07-29, Option 1. Raised by Fabble5_7 in the round-7 Q6.)*

**Also in round 8:** every check a registration can fail on now runs **above
the first mutation**. Round 7 claimed refusal happened "before any effect" and
tested only the registry entry; measured, the child's `_schema` had already
been auto-populated and the parent's join cache invalidated (GPT31
B32F7-P1-2).

The key-EXISTENCE check widens on the **CHILD** side only. It had lived inside
the `right_index_columns is not None` branch since PHASE_13_65, so the ordinary
symmetric call never ran it and a child without the key registered
successfully (GPT25 B32F7-P1-2). The **PARENT** side keeps its original,
narrower rule — checked on the asymmetric call, not on the symmetric one —
because a parent legitimately acquires its join key after registration: the key
may be a declared alias materialized later, or an unloaded branch under a lazy
reader. Widening it broke 29 tests; removing it broke `test_A6`, which predates
this phase. The original placement was not the defect.

**Encoded by:** `_key_arrays_equal`, the validation block at the top of
`register_subframe`; `test_b32_165`–`169`.

---

## AD-15/13.76.ADF addendum (round 8) — a pandas primitive is not a dtype guarantee

**Status:** part of AD-15.

```
pandas 1.5.3   Sparse[float32].take(...) -> Sparse[float32, nan]
pandas 3.0.2   Sparse[float32].take(...) -> Sparse[float64, nan]
```

On pandas ≥ 2 the widening happens even for **fully matched** positions. Round
7's exact-sparse-preservation claim was therefore true only on the coder's and
the architect's pandas, and the generated matrix that "proved" it fails on a
newer runtime. GPT26 found this by executing on pandas 2.2.3 — a runtime no
seat had used before, and the reason it took seven rounds.

**Rule:** ADF does not trust a pandas primitive to preserve a dtype. A
gathered result whose dtype differs from the verified source dtype is
normalized back to the source dtype, and the cast is checked for losslessness
before it is accepted; if it would change a value, the existing refusal stands.
The check reads only the array's STORED values, so restoring a sparse column
costs its non-fill values rather than a dense copy of the frame.

**Encoded by:** `_restore_exact_dtype`, `_stored_values`; `test_b32_170`
(subtype × sparse fill_value × matched/missing) and `test_b32_171` (a lossy
cast is still refused).

**Open, measured, and assigned:** `_place_fill`'s dense fallback costs ~5.25x
the dense column (10M rows: 420 MB peak against an 80 MB dense equivalent),
independent of density, and only for a sparse column that also has a fill knob
configured. A sparse-index reconstruction that never densifies is a named
**B3.2b** item; the cost is bounded, measured and documented in the source
rather than fixed in the closing hours of a correction round.

---

## AD-19/13.76.ADF — an explicitly supplied dtype is never changed, and no non-missing value is ever changed (RATIFIED)

**Ratified:** 2026-07-29, architect, in his own words. This is the decision the
whole fill-value-plus-flag design exists to serve, and the missing premise
behind a reviewer split that ran for three rounds.

### The ruling (architect, RATIFIED 2026-07-29, verbatim in substance)

> When the user explicitly supplies or declares a dtype, ADF must never change
> that dtype. This is the reason for the fill-value plus flag design.
>
> * the ITS correction column may have an explicitly selected integer or
>   floating dtype;
> * an absent ITS contribution may use the explicitly configured neutral
>   value `0`;
> * `hasITS=False` records that the ITS measurement was absent;
> * ADF must preserve the value-column dtype.
>
> The fill value and the flag have **different meanings**: the fill is the
> physical neutral value used in the calculation; the flag records
> applicability or provenance; **an unknown value must not silently become a
> neutral value unless the user explicitly configured that policy.**
>
> 1. no non-missing value may ever change;
> 2. an explicitly supplied dtype may not be silently widened;
> 3. **warning-and-widening is not permitted**;
> 4. if a missing key cannot be represented in the explicit dtype: use a
>    compatible fill explicitly configured by the user, or raise a clear
>    ADF-owned error explaining that a fill is required.
>
> Do not automatically choose `0`, `1`, `False`, or any other fill. Those are
> physical choices made by the user. Call this a **configured fill value** or
> **configured neutral value**, not an automatic default.

### THE OPERATIONAL DEFINITION — how the code decides, without guessing intent

The architect's decisive addition, and the reason this is implementable:
**ADF does not need to know whether the user consciously typed `dtype=...`.**
It uses the dtype observable from the data source or from ADF metadata.

> Every dtype observable from source metadata, an existing physical column,
> schema metadata, an explicit alias declaration, or the first successful
> creation/materialization is **authoritative**. ADF must preserve it
> thereafter. If a missing value cannot be represented in that dtype, ADF must
> use an explicitly configured compatible fill or refuse clearly.

Determined as follows:

| # | source | authoritative dtype |
|---|---|---|
| 1 | tree / ROOT / NumPy / PyArrow branch | the dtype the branch or reader metadata declares. **The branch is not loaded merely to learn its dtype**; lazy readers should expose dtype/schema information |
| 2 | existing pandas or child-ADF physical column | the dtype already on that column — once the frame is supplied to ADF, that dtype is authoritative |
| 3 | alias with an explicit dtype | `add_alias(..., dtype=...)` defines it immediately |
| 4 | alias without an explicit dtype | the dtype produced by the **first successful materialization**, recorded in the ADF schema; later materializations must not silently change it |
| 5 | column created internally by ADF | ADF may choose it at creation; **once created it is fixed** |

This is one reason `AliasDataFrame.df` exists: after a column or alias is
first materialized its physical dtype is visible, and can be kept stable for
every subsequent operation.

### Configured fill mechanisms, and their precedence

All three existing public mechanisms remain effective and reach the common
gather:

```
add_alias(..., fill_value=...)
set_subframe_fill(..., fill_missing=...)
set_global_fill(..., fill_missing=...)
```

**Precedence is the MEASURED historical order, preserved deliberately** (the
architect asked that it not be changed silently):

```
subframe fill  >  global fill  >  alias fill  >  clear refusal
```

Round 10's only change here is that the alias value now reaches the **gather**
instead of being applied after it. The observable order is identical; what
changes is that a configured alias fill now works for values that cannot
survive the intermediate representation — round 9 refused those calls before
the fill could be used (GPT30 R9-P0-1, GPT31 B32F9-P1-1).

### Behaviour

| case | behaviour |
|---|---|
| fully matched join | exact dtype, exact values — always |
| missing key, compatible fill configured | exact dtype, gap carries the fill |
| missing key, configured fill is not representable in the dtype | **refused**, naming the dtype and the value |
| missing key, no compatible configured fill | **refused**, naming all three fill mechanisms |

Never widen-and-warn. Never cast rounded floats back to an integer dtype.

**Encoded by:** `_matched_values_survive` (called before any widening is
accepted, on both the direct and the alias path), `_restore_exact_dtype`,
`_coerce_fill_to_dtype`, `_place_fill`; `test_b32_172`-`176`.

---

## AD-18/13.76.ADF — EVERY fill knob obeys AD-14, on every storage family (IMPLEMENTATION CONSEQUENCE of AD-14, not a separate decision)

**Status: implementation consequence of AD-14. NOT a separate architect
decision, and not ratified as one.** The round-7 heading said "ratified by
implication", which GPT27 correctly rejected: implication does not ratify any
more than implementation does, and this is the same invalid move the same
document had just corrected in AD-17 one section earlier. Recorded separately
only because round 6 implemented AD-14 at exactly one call site and the panel
treated the gap as a blocking P0 in its own right; the RULE is AD-14's, and
nothing here adds to it.

Round 6 built `_coerce_fill_to_dtype` and called it only for `fill_missing`, in
only the typed gather. `fill_nan`, `fill_inf`, the `fill_invalid` expansion and
the whole plain-float fast path assigned the raw configured value. Measured on
the round-6 bytes, confirmed independently by GPT25, GPT27 and GPT30, and
re-verified by the Main Reviewer against source:

| call | round 6 | round 7 |
|---|---|---|
| `float64` + `fill_nan="BAD"` | `object` holding `"BAD"` | refused, knob and dtype named |
| `float64` + `fill_missing=Decimal("1.25")` | `object` | `float64`, value `1.25` |
| `Float64` / `Float32` + `fill_nan=99`, `fill_inf=77` | silently ignored | applied, dtype preserved |
| `Sparse[float64]` + same | silently ignored | applied, sparsity preserved |
| `complex128` + `fill_inf="BAD"` | `object` | refused |

**Root cause, and it is the same root cause twice.** The gate was

```python
isinstance(dtype, np.dtype) and dtype.kind in "fc"
```

— the exact storage-family assumption round 6 had *just removed* from the
gather router (AD-13), re-typed one helper over. `Float64Dtype` and
`SparseDtype(float64)` are not `np.dtype` instances, so an explicitly
configured public policy was discarded without a word.

**Rule.** Applicability is asked by CAPABILITY
(`pandas.api.types.is_float_dtype` / `is_complex_dtype`), never by storage
family. Every knob writes through ONE primitive, `_place_fill`, which coerces
via `_coerce_fill_to_dtype` and then assigns, falling back to a dense
temporary and restoring the exact dtype for arrays that refuse item assignment
— so a sparse column is never densified in the result. `direct` mode continues
to touch missing keys only.

**Encoded by:** `_place_fill`, `_carries_nan_or_inf`, `_nan_inf_probe`;
`INVALID_FILL_CASES` × {`fill_nan`, `fill_inf`, `fill_invalid`} ×
{safe, direct} × {compatible, incompatible} (`test_b32_139`–`143`), plus
`test_b32_144`–`147` for `fill_missing` on the fast path.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-06-10 | Registry created (PHASE_13_55_ADF §11 item 6). Seeded with AD-1/13.55.ADF. Legacy backfill scan proposed post-13.55. |
| 1.1.0 | 2026-06-11 | AD-2/13.56.ADF added (PHASE_13_56_ADF ratifications; amends AD-1 item-4 scope). |
| 1.2.0 | 2026-06-15 | AD-3/13.59.ADF added (PHASE_13_59_ADF metadata read/write precedence; ratified). |
| 1.3.0 | 2026-07-19 | AD-4/13.76.ADF added (symmetry-by-default classification rule; SEED-1.a/SEED-2.a applications; ratified in chat). |
| 1.4.0 | 2026-07-19 | AD-5/13.76.ADF (bins×scatter shared/explicit split) and AD-6/13.76.ADF (draw_figures ax clean refusal) added; panel-reviewed, architect-ratified. |
| 1.5.0 | 2026-07-28 | AD-7/13.76.ADF (subframe projection preserves the user's dtype; reuses the existing fill contract), AD-8/13.76.ADF (duplicate-owner graphs refused before any effect; read-only validator at draw_batch entry), AD-9/13.76.ADF (B3.2/B3.2b boundary, delegated to reviewers) added; architect-ratified during B3.2 part 2 correction round 4. |
| 1.6.0 | 2026-07-28 | AD-10/13.76.ADF (compound failure: original error stays primary; cleanup failure is secondary evidence, not a chained cause) and AD-11/13.76.ADF (AD-7 applies to user-supplied extension/timezone/complex dtypes; interval-with-missing refused; implemented as one symmetric pandas primitive rather than a branch per dtype) added; AD-7's "forbidden" clause disambiguated. Architect-ratified during B3.2 part 2 correction round 5. |
| 1.7.0 | 2026-07-28 | AD-12 (B3.2 closes / B3.2b owns the full dependency plan, before B3.3), AD-13 (int and Boolean columns not silently widened; scope boundary against the ratified April alias-layer contract), AD-14 (a fill is compatible with the COLUMN, checked by one round-trip primitive), AD-15 (sparse and pyarrow-backed dtypes), AD-16 (slot symmetry requirement recorded for B3.3-B3.5, `facet_by` fixed immediately by explicit instruction) and AD-17 (an index level IS a join key) added. GPT31's decision set, approved by all GPT seats and Fable; ratified during B3.2 part 2 correction round 6. |
| 1.8.0 | 2026-07-28 | AD-17 corrected to **PROPOSED** (implementation cannot ratify a decision — GPT27/GPT30) and extended: an index level equals a same-named column only when their VALUES agree, otherwise the frame is refused before any effect; partial-MultiIndex `pre_index` supported. AD-18 added (every fill knob obeys AD-14 on every storage family; applicability asked by capability, not by NumPy-dtype identity). AD-12 challenged by GPT27 and upheld by the Main Reviewer against the primary ratified text. Revision history re-ordered. B3.2 part 2 correction round 7. |
| 1.9.0 | 2026-07-29 | AD-13a added (direct-slot int/bool widening is REPORTED with a FutureWarning that says it will become an error; the value stays a gap, the remedy is `set_subframe_fill`; scoped to the one call site with no alias restoration). Resolves the 3-2 panel split on AD-13's scope boundary. AD-19 (the physical rationale for gap-vs-zero) is pending from the architect and must be explicitly requested in the next CRR. |
| 1.10.0 | 2026-07-29 | AD-17 addendum (missing-value rules for key equality: both-missing = equal, one-sided = different, spelling of the gap not significant — Decission KEY-MISSING-SPELLING Option 1; all registration validation moved above the first mutation; symmetric key-existence check restored). AD-15 addendum (a pandas primitive is not a dtype guarantee — `SparseArray.take` widens float32 on pandas >= 2; results are normalized back to the verified source dtype, with the `_place_fill` dense-temporary cost measured at ~5.25x and assigned to B3.2b). AD-18 relabelled an implementation consequence of AD-14, not a ratified decision. B3.2 part 2 correction round 8. |
| 2.0.0 | 2026-07-29 | **AD-19 RATIFIED and recorded canonically** in the architect's own words: an explicitly supplied dtype is never changed, no non-missing value is ever changed, warning-and-widening is not permitted, and a gap that cannot be represented requires an explicit compatible fill or a clear refusal. The weaker reviewer formulation was explicitly rejected. **AD-17 RATIFIED**, including the missing-value addendum. **AD-13a SUPERSEDED by AD-19** on the architect's instruction — warning-and-widening was not his decision. Major version bump: AD-19 changes a public behaviour that had held since April. B3.2 part 2 correction round 9. |
| 2.1.0 | 2026-07-29 | AD-19 completed with the architect's **operational definition** (Option 1): every dtype observable from source metadata, an existing physical column, schema metadata, an explicit alias declaration, or the first successful materialization is authoritative — ADF never has to guess user intent. The round-9 lossless-widening exception is removed; `_warn_direct_slot_widening` is deleted. All three configured-fill mechanisms now reach the common gather, with the measured historical precedence (subframe > global > alias) preserved and pinned. `_safe_dtype_cast` no longer routes exact integers through float64 and no longer invents 0/False. `test_A5`, `test_D1`, `test_D2`, `test_I3_9` and `test_b32_175` revised on the architect's explicit instruction. B3.2 part 2 correction round 10. |

---

## SUPERSESSION NOTICE — AD-19 "Configured fill mechanisms, and their precedence"

**Recorded:** 2026-08-01, B3.2 part 2 checkpoint fix11a.
**Superseding authority:** `PHASE_13_76_ADF_AD19_DTYPE_AND_FILL_BRAINSTORM_v1_4_2_RATIFIED.md`
(architect-ratified 2026-07-30) as amended by `v1_4_4` (approved 2026-07-31), **AR-3**.
**AD number for the v1.4.4 ratification is to be assigned by the architect** —
per v1.4.4 §14.2 this document does not invent one.

### What is superseded, and it is a factual error in this registry

AD-19 §"Configured fill mechanisms, and their precedence" states:

> *"Round 10's only change here is that the alias value now reaches the gather
> instead of being applied after it. The observable order is identical."*

**That sentence is FALSE and was written by the coder, not by the architect.**
It is true only for the trivial expression `alias = S.v`, where the operand
stage and the final-result stage coincide. For a compound expression the two
stages give different scientific values. Measured on both bytes:

```text
add_alias("d", "S.v + x", dtype="int64", fill_value=1), S.v key missing,
no operand fill configured

    pre-phase c73f0c99   ->  [13, 1]     alias fill is a FINAL-RESULT policy
    round 10 f9781761    ->  [13, 21]    alias fill was applied as an OPERAND fill
```

The single flat precedence list `subframe > global > alias > refusal` is
therefore also wrong as stated: the mechanisms do not live at one stage.

### The ratified contract (AR-3)

```text
OPERAND stage   set_subframe_fill(...) / set_global_fill(...)
                applied BEFORE expression evaluation; they DEFINE the operand,
                and after they apply there is no remaining undefinedness.
                Precedence among them is unchanged and remains the measured
                historical order:
                    subframe-specific  ->  global  ->  native gap / refusal

RESULT stage    add_alias(..., fill_value=...)
                applied AFTER evaluation, and ONLY when the final alias result
                is still undefined or invalid. It is never injected into a
                subframe operand.
```

Decisive controls (v1.4.4 §14.3):

```text
subframe fill=0, alias fill=1, alias=S.v+x, S.v missing   ->  0+x
no operand fill, alias fill=1, alias=S.v+x, S.v missing   ->  1
```

**AR-5 states why this matters physically:** a configured fill is a
*definition*, not a repair. A calibration defined only on TPC points is
deliberately extended to the whole frame by the operation's neutral element —
`0` additive, `1` multiplicative — so the transformation is the identity
outside its domain, and the filled value is then consumed by later arithmetic
as ordinary data. Applicability is carried separately by a flag such as
`row < 152`. ADF must never choose the neutral value itself.

### Implementation status at this checkpoint — NOT yet corrected

The fix11a bytes **still contain the superseded round-10 behaviour**.
`_active_alias_fill` still publishes the alias fill into the gather.

```text
pinned by     tests b32_195, b32_196, b32_197   xfail(strict=True)
owned by      D_4 / D_5 — delete _active_alias_fill; carry residual
              undefinedness in a row mask; apply the alias fill to the final
              result only
```

`strict=True` means these tests become FAILURES the moment the fix lands, so
this notice cannot outlive the defect.

### Also corrected here

AD-19 §"Encoded by" claims `_matched_values_survive` is *"called before any
widening is accepted, on both the direct and the alias path"*. It is called on
the **gather** for both paths; `_safe_dtype_cast` sits downstream on the alias
path and is not covered by it. Coder error, corrected on the record.

This checkpoint closes neither B3.2 nor B3.2b and is not production-ready for
unrestricted calibration use.

---

## Revision History (continued)

| Version | Date | Changes |
|---------|------|---------|
| 2.2.0 | 2026-08-01 | **Supersession notice appended.** AD-19's precedence subsection is corrected: the coder-written claim that round 10's move of the alias fill into the gather left "the observable order identical" is FALSE for compound expressions (measured `[13, 21]` against pre-phase `[13, 1]`). Ratified contract v1.4.4 AR-3 governs: operand fills define operands before evaluation; alias `fill_value` is a final-result policy only. AR-5 records the calibration motivation — a configured fill is a definition, not a repair. The fix11a checkpoint still implements the superseded behaviour, pinned by `b32_195`–`197` as strict xfail and owned by `D_4`/`D_5`. The `_matched_values_survive` coverage claim is also corrected. Historical text retained, marked superseded, not deleted. AD number for the v1.4.4 ratification to be assigned by the architect. |
