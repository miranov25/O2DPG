# ARCHITECT_DECISIONS.md
# AliasDataFrame — Architect Decision Registry
# Version: 1.0.0 (created in PHASE_13_55_ADF per §11 item 6)
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

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-06-10 | Registry created (PHASE_13_55_ADF §11 item 6). Seeded with AD-1/13.55.ADF. Legacy backfill scan proposed post-13.55. |
| 1.1.0 | 2026-06-11 | AD-2/13.56.ADF added (PHASE_13_56_ADF ratifications; amends AD-1 item-4 scope). |
| 1.2.0 | 2026-06-15 | AD-3/13.59.ADF added (PHASE_13_59_ADF metadata read/write precedence; ratified). |
