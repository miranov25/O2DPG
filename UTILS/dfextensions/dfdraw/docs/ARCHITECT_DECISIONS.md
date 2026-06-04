# ARCHITECT_DECISIONS.md
# dfdraw — Architect Decision Registry
# Version: 1.0.1 (touch-ups applied per Org panel review)
# Date: 2026-06-04 (v1.0) / 2026-06-04 (v1.0.1)
# Maintainer: Marian Ivanov (architect)
#
# This document is the canonical AD registry for the dfdraw project.
# STYLING_FRAMEWORK_DECISIONS.md §2 is superseded by this file.
#
# Naming conventions:
#   AD-N          — legacy format, Phases 13.12–13.33 (single sequential counter)
#   AD-N/PHASE    — current format, Phase 13.34+ (counter restarts per phase)
#   AD-N/BUG-ID   — bug-driven decisions
#   GP-N          — General Principles (frozen set GP-1..GP-8 as of v1.0)
#
# Note on AD-39b: legacy letter-suffix grandfathered. New AD-N/PHASE format uses integer N only.
#
# Source-code citation pattern:
#   # AD-67: 1-element list silently degrades to scalar          (legacy)
#   # AD-2/13.37.DF: Poisson error bars (hist_errors=True)       (current)
#
# Grep pattern (matches all forms including GP):
#   grep -rnE "(AD-[0-9]+[a-z]?(/[A-Za-z0-9._-]+)?|GP-[0-9]+)" --include="*.py" .
#
# Rules:
#   - AD identifiers are immutable once ratified
#   - Withdrawn decisions keep their number with status: withdrawn
#   - Numbers are never reused
#   - GP-3: architect quotes preserved verbatim including typos

---

## About this document

This file consolidates all dfdraw architect decisions from Phase 13.12.DF onward.

**Source of truth:** This file is the canonical AD registry for dfdraw. `STYLING_FRAMEWORK_DECISIONS.md §2` is superseded by this document. In case of conflict, this file takes precedence.

**Legacy format (AD-1 through AD-82):** single sequential global counter, ratified through Phase 13.33.DF. Source text from Sonnet56 (verbatim, highest-version proposal) + Sonnet57 (source code references at HEAD `07606c02`).

**Current format (AD-N/PHASE):** per-phase counter restarts at 1, Phase tag matches canonical PHASE_*.md identifier. Ratified from Phase 13.34.DF onward. This registry supersedes the candidate table (`AD_candidate_table_FINAL.md`) — entries here are the ratified record.

**Schema variation:** Legacy entries (AD-1..AD-82) carry richer fields (verbatim decision text block, version history, rejected alternatives) because they were backfilled from full proposal documentation. New-format entries (AD-N/PHASE) use a leaner schema. Both are valid; the minimum required fields are: Decision, Ratification, Source.

**Gaps in legacy numbering:** AD-4 through AD-14, AD-19 through AD-28, AD-33 and AD-34 are not in the archive — they predate the Phase 13.12 archive and were never promoted from brainstorming to formal decisions. These gaps are permanent.

---

## General Principles

**Transitional authority.** GP-1..GP-8 are the frozen legacy GP set as of v1.0. GP-6..GP-8 record architect-ratified cross-subproject principles originating from dfdraw incidents. They bind dfdraw in this registry. Their adoption in AliasDataFrame and GBAI follows the Option B decision (2026-06-04): each subproject's own `ARCHITECT_DECISIONS.md` records adoption. This file is the provenance source.

General Principles (GP-N) are project-wide governance rules that bind all phases and all reviewers. They are derived from real incidents — the triggering failure case is documented. Rules are never applied retroactively to closed phases.

GPs are distinct from Architect Decisions (AD-N): ADs record what was decided about the implementation; GPs record how the project governs itself.

**Cross-team applicability:** GP-1 through GP-5 apply to dfdraw. GP-6 through GP-8 apply to all three subprojects (dfdraw, AliasDataFrame, GBAI).

---

### GP-1 — Style configurability lands at interface introduction, not deferred

**Date:** 2026-05-05 (Phase 13.26.DF) | **Scope:** dfdraw

**Origin:** Phase 13.26.DF v1.0 review cycle (Path B vs Path A debate), Claude49 draft review.

**Statement:** When a phase introduces a new internal mechanism, the style keys controlling it must land in the same phase — not deferred to a "later" Phase C+. Deferral creates a refactor cost when the deferred phase has to retrofit style lookups everywhere.

**Precedent:** Phase 13.25.DF Phase A landed `quantile.*` style keys (AD-53) in the same phase as the rendering modes. Phase 13.26.DF Phase B landed all 10 `channels.*` style keys in the same phase as Algorithm A.

---

### GP-2 — Internal APIs accepting new data-channel types must be list-based from day one

**Date:** 2026-05-05 (Phase 13.26.DF v1.2) | **Scope:** dfdraw

**Origin:** Phase 13.26.DF v1.2 G-7, Claude48 reviewer note 2026-05-05.

**Statement:** When an internal API will accept a new data-channel type in a future phase (e.g., `selection_delta` in Phase D), the API must accept a `list[DataChannel]` (or equivalent extensible structure) from day one — not a fixed-arity signature with hardcoded boolean flags. Adding a channel type later requires adding a list entry, not changing a signature.

**Precedent:** v1.1 used three booleans (`has_vector`, `has_group`, `has_quantile_channel`); v1.2 G-7 generalized to `list[DataChannel]` with `EXPLICIT_RULES: dict[frozenset[str], dict[str, str]]`. Same total LOC; eliminates Phase D internal-API redo.

**Corollary — additive contract:** `EXPLICIT_RULES` is append-only across phases. Modifying an existing entry would be a regression of a prior phase's locked behaviour; new entries are additive only.

---

### GP-3 — Architect signals preserved verbatim with typos

**Date:** 2026-05-03 (Phase 13.25.DF) | **Scope:** dfdraw

**Origin:** Reviewer Card v1.29 Rule 9, Phase 13.25.DF AD-50 reformulation incident.

**Statement:** Architect quotes embedded in proposals, decision logs, and review documents must be verbatim. Typos and informal phrasing are the strongest signals of authenticity. Reformulating an architect quote into "polished" prose risks summarising away architect intent — has caused production bugs (AD-50 cascade-vs-independence resolution required FIX1 to recover).

**Operationally:** the 2026-05-05 quotes in STYLING_FRAMEWORK_DECISIONS.md §1.2 contain *"wan"*, *"defuat"*, *"begining"*, *"psoponed"*, *"easly"*, *"quentiles"*, *"asnwer"*, *"Parametrizrtion"*. All preserved.

---

### GP-4 — Backward-compat scope must be justified by production-usage verification

**Date:** 2026-05-05 (Phase 13.26.DF v1.2) | **Scope:** dfdraw

**Origin:** Phase 13.26.DF v1.2 G-1/G-3 amendment, "no quantile users" incident.

**Statement:** Before locking backward-compatibility constraints on a code path, the proposing party (drafter or reviewer) must verify the path has production users — by `grep`/AST against production scripts. Otherwise the constraints are unjustified and lock implementation freedom for no benefit.

**Precedent:** Phase 13.26.DF v1.0 → v1.1 took three review iterations to converge. None of the 6 reviewers verified that the quantile-touching code path had production users until the architect amendment chat 2026-05-05. v1.2 G-3 reframed Class 10 around real `makeSmoothMapsWithTPC.py` patterns; 2 of 7 backward-compat tests were removed as locking-greenfield-paths.

---

### GP-5 — Drafter rotation across phases is healthy

**Date:** 2026-05-05 (Phase 13.26.DF v1.2) | **Scope:** dfdraw

**Origin:** Phase 13.26.DF v1.0/v1.1 (Claude48) → v1.2 (Claude49Coder) handoff. Claude48 review of v1.2 endorsed the rotation explicitly.

**Statement:** Different drafters across phase iterations bring different lenses. The G-7 forward-extensibility upgrade in v1.2 was a structural improvement the original drafter did not surface in v1.0/v1.1. The successor drafter caught it. This is evidence that the multi-model drafter rotation pattern works.

**Reviewer note (Claude48, v1.2 review):** *"The drafter handoff (Claude48 → Claude49Coder) is healthy — different drafter caught an architectural improvement (G-7) that the original drafter did not surface in v1.0/v1.1."*

---

### GP-6 — Rendered artifacts require actual rendering before approval

**Date:** 2026-05-30 (Phase 13.49.DF FIX1) | **Scope:** dfdraw, AliasDataFrame, GBAI

**Origin:** Phase 13.49.DF FIX1 — architect rendered the HTML in a browser and found 3 bugs (H-1 stale artifact, H-2 JS selector scope, H-3 duplicate category headers) that the 8-reviewer v1.1 panel missed by reading the diff only.

**Statement:** When a phase delivers an HTML or other rendered artifact, at least one reviewer MUST render and navigate it before approval. Diff-reading does not discharge visual-render verification. This is now formalized as Reviewer QRC v1.31 Rule 14 caveat.

**Triggering failure:** H-1 META Broken in HTML vs Verified in MD (stale-artifact hypothesis); H-2 JS selector `[data-status]` too broad — clicking feature `<tr>` row hijacked filter state; H-3 15 duplicate category headers. All three missed by every diff-reading reviewer; all three caught on first render.

---

### GP-7 — Pre-CRR source-vs-spec audit before writing the CRR (§0 attestation)

**Date:** 2026-05-30 (Phase 13.50.DF) | **Scope:** dfdraw, AliasDataFrame, GBAI

**Origin:** Phase 13.50.DF CRR v1.0 — architect's "Was all functionality from spec implemented?" challenge caught 4 undisclosed partials in the v1.0 draft. Draft was scrapped. §0 audit table cross-checked all 20 v2.5 in-scope items; second sweep confirmed conformance; THEN the CRR was drafted. Proposed as Coder QRC R17.

**Statement:** Before writing the CRR, the coder must perform a source-vs-spec audit: list every in-scope item from the proposal, verify each against source, and attest in §0 of the CRR. Catching partial implementations at CRR time is the right gate — catching them in the reviewer round wastes panel time.

**R17 extension (Phase 13.50.DF FIX1):** When the audit covers a rename or scope change, the sweep MUST include documentation surfaces — `feature_taxonomy.py` "name" fields, `CAPABILITY_MATRIX.md/.html` generated content, inline source comments. Fingerprint command: `grep -rn "OLD_TOKEN" tests/feature_taxonomy.py plots/ docs/`.

---

### GP-8 — Three consecutive recurrences triggers infrastructure enforcement

**Date:** 2026-06-03 (Phase 13.50.DF FIX2) | **Scope:** dfdraw, AliasDataFrame, GBAI

**Origin:** Phase 13.50.DF FIX2 — `CAPABILITY_MATRIX.html` missing from reviewer zip recurred in Phase 13.48 (first occurrence), Phase 13.49 (second), and Phase 13.50 (third). On third recurrence, a mechanical enforcement gate was added to `run_tests.sh` rather than another flag-and-promise cycle.

**Statement:** When a governance failure recurs three times in consecutive phases despite flag-and-promise resolutions, the correct response is infrastructure enforcement — a mechanical check that makes the failure impossible — not another verbal commitment.

**Implementation:** `unzip -l reviewer_bundle.zip | grep -q '\.html'` non-fatal check added to `run_tests.sh`. Closes the gap mechanically at commit df3057a3 + gate enforcement at 07606c02.

---


## Phase 13.12.DF — Profile enhancements (AD-1 to AD-3)

**Phase trigger:** TPC calibration plot review — rows 0–3 showed huge error bars (N < 3). Calibration workflow requires per-bin statistics exported for downstream fitting.

---

### AD-1 — `min_entries=3` default

**Phase:** 13.12.DF | **Source:** `PHASE_13_12_DF_Proposal.md` v1.1, §2 | **Date:** 2026-03-22

**Decision text (verbatim):**
> `min_entries=3` default — Minimum to have error bar defined (SEM requires n ≥ 2, n=3 for stability)

**Motivation:** `plots/profile.py` was producing bins with undefined/unstable error bars. SEM requires n ≥ 2; n=3 provides one degree of stability margin.

**Verbatim architect quote:** *"3 as default minimum — to have error bar defined"*

**Rejected alternatives:** None documented; default value was architect-determined.

**Ratification:** v1.1 2026-03-22 — listed in revision history as architect decision.

**Source references:**
- `drawer.py:5394` — `AD-1: default=3 for stable error bars. Phase 13.12.DF F2.`
- `plots/profile.py:16` — module header: `F2: min_entries=3 → suppress low-statistics bins (AD-1)`
- `plots/profile.py:277` — `return_data=True. AD-1: default=3 for stable error bars.`

---

### AD-2 — Features F1–F4 apply to `profile()` only

**Phase:** 13.12.DF | **Source:** `PHASE_13_12_DF_Proposal.md` v1.1, §2 + §4 | **Date:** 2026-03-22

**Decision text (verbatim):**
> Features F1–F4 apply to `profile()` only — Other plot types (`hist`, `scatter`) not in scope

**Motivation:** Scope discipline — calibration workflow enhancements were specifically designed for profile plots. Extending to hist/scatter was not needed for the TPC dE/dx use case.

**Verbatim architect quote:** *"Only and only to profile"* (§4 Scope)

**Rejected alternatives:** Implementing F1–F4 across all plot types — explicitly rejected.

**Ratification:** v1.1 2026-03-22.

**Source references:** *(enforced by parameter placement — no inline AD-2 comment in source)*

---

### AD-3 — Interval label format: custom `0.0-1.5`

**Phase:** 13.12.DF | **Source:** `PHASE_13_12_DF_Proposal.md` v1.1, §2 + §3.3 | **Date:** 2026-03-22

**Decision text (verbatim):**
> Interval label format: custom `0.0-1.5` — Cleaner than pandas default `(0.0, 1.5]`

**Motivation:** pandas default interval notation `(0.0, 1.5]` is less readable in physics plot legends.

**Verbatim architect quote:** *"custom format"*

**Implementation (from proposal):**
```python
def _format_interval_label(interval) -> str:
    """Format interval as 'low-high' (AD-3: custom format)."""
    return f"{interval.left:.2g}-{interval.right:.2g}"
```

**Rejected alternatives:** pandas default `(0.0, 1.5]` notation — rejected as less clean for display.

**Ratification:** v1.1 2026-03-22.

**Source references:**
- `plots/profile.py:40` — `# Phase 13.12.DF: Interval label formatting (AD-3)`
- `plots/profile.py:47` — `AD-3: Custom format instead of pandas default '(0.0, 1.5]'.`
- `tests/test_profile_phase13_12.py:209` — `"""Verify custom interval label format (AD-3)."""`

---

## [GAP: AD-4 through AD-14]

These AD numbers do not appear in any governance document in the archive. Brainstorming v1.1/v1.2 used AO-x (Architect Opinion) notation; Phase 13.13 was the first to promote AO items to numbered ADs. The gap from AD-3 to AD-15 is intentional — early brainstorming AOs were renumbered starting at AD-15 in Phase 13.13.

---

## Phase 13.13.DF — `same=True` superposition (AD-15 to AD-18, AD-35 to AD-37)

**Phase trigger:** ROOT users expect `Draw("same")` overlay syntax. Current dfdraw required capturing and passing axes explicitly — verbose and error-prone in interactive calibration analysis.

**Production use cases:** A-side vs C-side calibration comparison, before/after correction, multi-variable overlay.

---

### AD-15 — `same=True` uses `self._last_ax` with `plt.gca()` fallback

**Phase:** 13.13.DF | **Source:** `PHASE_13_13_DF_v1_0_Proposal_Same.md` §3.1, §4 | **Date:** 2026-03-25

**Decision text (verbatim):**
> `same=True` uses `self._last_ax` with `plt.gca()` fallback

**Motivation:** Two implementation options:
- **Option A:** `plt.gca()` only — simple but risky in Jupyter (cell execution order can give wrong axes)
- **Option B (chosen):** `self._last_ax` with `plt.gca()` fallback — each DFDraw tracks its own axes; predictable in notebooks

**Verbatim architect quote:** *"I assume Option B is safer."*

**Rejected alternatives:** Option A (`plt.gca()` only) — rejected due to Jupyter notebook hazard.

**Ratification:** Architect approval of Option B. Proposal v1.0 §4 AD table.

**Source references (19 occurrences — every draw method docstring):**
- `drawer.py:363` — `# Phase 13.13.DF: Track last axes for same=True (AD-15)`
- `drawer.py:995` — `AD-15: self._last_ax with plt.gca() fallback.`
- `drawer.py:1027` — `# Fallback to plt.gca() (AD-15)`
- `drawer.py:4163, 4261, 4502, 4617, 4924, 5023, 5280, 5412, 5996, 6071, 6244, 6301` — `same=True` parameter docstrings
- `tests/test_same.py:4, 67, 196`

---

### AD-16 — Auto-increment colors from palette when `same=True`

**Phase:** 13.13.DF | **Source:** `PHASE_13_13_DF_v1_0_Proposal_Same.md` §3.2, §4 | **Date:** 2026-03-25

**Decision text (verbatim):**
> Auto-increment colors from palette when `same=True`

**Motivation:** Without automatic color cycling, all overlaid curves render in the same color — "useless" per reviewer consensus.

**Note:** Color cycle starts at index 1 (first call uses index 0). **Resets per-figure — intentional.** ADF-side continuity is AD-50/AD-37 scope.

**Ratification:** Reviewer vote V0.1 (4-0 unanimous).

**Source references:**
- `drawer.py:365` — `self._color_cycle_index = 1  # Start at 1: first plot uses index 0 (AD-16, A2)`
- `drawer.py:1041` — `AD-16: Auto-increment colors from palette.`
- `tests/test_same.py:5, 164, 264`

---

### AD-17 — Auto-generate label from expression when `same=True`

**Phase:** 13.13.DF | **Source:** `PHASE_13_13_DF_v1_0_Proposal_Same.md` §3.2, §4 | **Date:** 2026-03-25

**Decision text (verbatim):**
> Auto-generate label from expression when `same=True`

**Motivation:** Legend mandatory for overlaid plots. Requiring manual `label=` is a DRY violation when the expression string already describes the curve.

**Ratification:** Reviewer vote V0.2 (3-1).

**Source references:**
- `drawer.py:1062` — `AD-17: Auto-generate label from expression (user can override with label=).`
- `tests/test_same.py:6, 181, 292, 300`

---

### AD-18 — Append to title when `same=True` + `auto_title=True`

**Phase:** 13.13.DF | **Source:** `PHASE_13_13_DF_v1_0_Proposal_Same.md` §3.3, §4 | **Date:** 2026-03-25

**Decision text (verbatim):**
> Append to title when `same=True` + `auto_title=True`

**Verbatim architect quotes:**
> *"We need same with auto_title."*
> *"In my case title is basic — it should show which query and how was done."*

**Ratification:** AO-0 promoted to AD-18.

**Source references:**
- `drawer.py:988` — `# Phase 13.13.DF: same=True Support (AD-15 through AD-18)`
- `drawer.py:2128` — `# AD-18: Append to title when same=True + auto_title=True`
- `plots/_auto_title.py:135` — `AD-18: Append to title when same=True + auto_title=True.`
- `tests/test_same.py:7, 107`

---

### AD-35 — No hard limit on title lines

**Phase:** 13.13.DF | **Source:** `PHASE_13_13_DF_v1_0_Proposal_Same.md` §3.3, §4 | **Date:** 2026-03-25

**Decision text (verbatim):**
> No hard limit on title lines; user decides when plot becomes unreadable

**Motivation:** Reviewers voted for a max 3-line title limit. Architect overrode.

**Verbatim architect quote:** *"User should know consequence — we should not limit him until plot is still visible."*

**Rejected alternatives:** Max 3-line limit proposed by reviewers — rejected by architect override.

**Ratification:** Architect override of reviewer vote.

**Source references:**
- `plots/_auto_title.py:136` — `AD-35: No hard limit on title lines.`
- `tests/test_same.py:8, 119`

---

### AD-36 — Truncation indicator `(+N more)`

**Phase:** 13.13.DF | **Source:** `PHASE_13_13_DF_v1_0_Proposal_Same.md` §4 | **Date:** 2026-03-25

**Decision text (verbatim):**
> Truncation indicator: `(+N more)` when needed

**Ratification:** Vote V6.2 (4-0).

**Source references:**
- `tests/test_same.py:9` — `AD-36: Truncation: (+N more)`

---

### AD-37 — AliasDataFrame must cache DFDraw instance for safe `same=True` (OPEN)

**Phase:** 13.13.DF | **Source:** `PHASE_13_13_DF_v1_0_Proposal_Same.md` §11.1, §4 | **Date:** 2026-03-25

**Decision text (verbatim):**
> AliasDataFrame must cache DFDraw instance for `same=True` to work safely.

**Motivation:**
```python
# Problem: new DFDraw instance each call
aDF.draw("y1:x")            # DFDraw #1, stores _last_ax
aDF.draw("y2:x", same=True) # DFDraw #2, _last_ax is None → fallback to plt.gca()
```

**Status:** OPEN PROBLEM; implementation strategy superseded by pending AD-50. The `same=True` continuity problem remains unresolved on the ADF side; full `DFDraw` instance caching must not be implemented. See AD-50 for the admissible C1b pattern.

**Rejected alternatives:** Relying on `plt.gca()` fallback — unreliable in notebooks.

**Ratification:** Architect decision. AD-37 is an ADF-team responsibility.

**Source references (11 occurrences across 5 files):**
- `drawer.py:1014` — `tracked _last_ax. AD-37 requires AliasDataFrame to cache DFDraw instance`
- `examples/generate_gallery.py:30, 368, 371, 388, 397, 418` — gallery centerpiece before/after
- `tests/test_phase_13_48_df_visual_testing.py:296` — `"""V.4 — AD-37: per-group colors are distinct (no color-cycle reset)."""`
- `tests/test_vector.py:672, 687` — `f"Expected >= 3 distinct colors (AD-37 fix), got {all_colors}"`
- `tests/feature_taxonomy.py:1687` — `"per-group colors distinct — color-cycle not reset (AD-37 class)"`

---

## [GAP: AD-19 through AD-28]

These AD numbers do not appear in any governance document. Brainstorming v1.2 lists candidate AOs/ADs in this range (quantile interface, multi-group layout options) but they were never formally promoted to numbered ADs in a phase proposal. AD-29 through AD-34 were proposed in Phase 13.13.DF v1.1 / 13.14.DF for batch/layout features.

---

## Phase 13.14.DF — `draw_batch` group format (AD-29 to AD-32)

**Phase trigger:** `draw_batch` took a flat dict — no shared defaults, no subplot grid. Production calibration dashboards need controlled layout.

---

### AD-29 — `layout=(rows, cols)` parameter; `layout` overrides `ncols`

**Phase:** 13.13.DF v1.1 / 13.14.DF | **Source:** `PHASE_13_14_DF_v1_0_Proposal_Batch_Rev2.md` §AD table | **Date:** 2026-03-25+

**Decision text (verbatim):**
> `layout=(rows, cols)` — parameter name for explicit grid control in `draw_batch()`

**Motivation:** Batch drawing needs explicit grid layout. `layout=(nrows, ncols)` overrides `ncols` shorthand.

**Ratification:** Architect decision D1. Phase 13.14 AD compliance table.

**Source references:**
- `drawer.py:6913` — `# Layout resolution: layout > ncols > auto (AD-29)`

---

### AD-30 — `layout=` works with and without `facet=True`

**Phase:** 13.14.DF | **Source:** `PHASE_13_14_DF_v1_0_Proposal_Batch_Rev2.md`

**Decision text (verbatim):**
> `layout=` works with and without `facet=True` (needed for `vary=` use case)

**Ratification:** Phase 13.14 AD compliance table. *(no inline source comment)*

---

### AD-31 — Extend `draw_batch()`, reject `draw_multi()`

**Phase:** 13.14.DF | **Source:** `PHASE_13_14_DF_v1_0_Proposal_Batch_Rev2.md`

**Decision text (verbatim):**
> Reject `draw_multi()` — extend `draw_batch()` with `layout=` parameter

**Motivation:** Extending `draw_batch()` keeps batch API consolidated. No API surface proliferation.

**Ratification:** Phase 13.14 AD compliance table. *(no inline source comment)*

---

### AD-32 — Non-ASCII filenames: sanitize by default

**Phase:** 13.14.DF | **Source:** `PHASE_13_14_DF_v1_0_Proposal_Batch_Rev2.md`

**Decision text (verbatim):**
> Non-ASCII filenames: sanitize by default, style option to control

**Ratification:** Phase 13.14 AD compliance table. *(no inline source comment)*

---

## [GAP: AD-33, AD-34]

Referenced in brainstorming v1.2 but not promoted in formal proposals. Not recoverable from current archive.

---

### AD-35 — (see Phase 13.13.DF section above)

### AD-36 — (see Phase 13.13.DF section above)

### AD-37 — (see Phase 13.13.DF section above)

---

## Phase 13.18.DF — Robust statistics (AD-38 to AD-43)

**Phase trigger:** Architect observed `std=0.613` on a distribution with visible FWHM ≈ 0.2 — a 3× overestimate for heavy-tailed TPC calibration distributions (χ²/ndf with misassociations).

---

### AD-38 — Parameter name: `stat_fields`

**Phase:** 13.18.DF | **Source:** `PHASE_13_18_DF_v1_2_Proposal_RobustStats.md` §AD table | **Date:** 2026-04-29

**Decision text (verbatim):**
> Parameter name: **`stat_fields`**

**Motivation:** Decouples computation from display — `stat_fields` controls what gets computed; existing `stats=` controls what gets displayed. The architect's key observation: `std=0.613` on FWHM ≈ 0.2 distribution — a 3× overestimate for heavy-tailed data.

**Ratification:** Architect + 5/5 reviewers unanimous.

**Source references:** *(expressed throughout the codebase — no single inline AD-38 comment)*

---

### AD-39 — No scipy dependency — numpy manual implementation

**Phase:** 13.18.DF | **Source:** `PHASE_13_18_DF_v1_2_Proposal_RobustStats.md` §AD table | **Date:** 2026-04-29

**Decision text (verbatim):**
> Scipy dependency: **Option A: numpy manual** (no dependency)

**Motivation:** scipy was undesirable for the O2DPG build environment. The robust statistics formulas are short numpy expressions (4 lines each).

**Rejected alternatives:** `scipy.stats.skew/kurtosis` — rejected to avoid dependency.

**Ratification:** 5/5 reviewers unanimous.

**Source references:** *(no inline AD-39 comment — expressed by absence of scipy import)*

---

### AD-39b — ddof=0 for skewness/kurtosis (ROOT-compatible)

**Phase:** 13.18.DF | **Source:** `PHASE_13_18_DF_v1_2_Proposal_RobustStats.md` §AD table v1.2 | **Date:** 2026-04-29

**Decision text (verbatim):**
> ddof for skewness/kurtosis: **ddof=0** (ROOT-compatible, matches dfdraw's `std`)

**Motivation:** ROOT uses population-based (ddof=0). Using ddof=1 (scipy default) would produce values inconsistent with ROOT.

**Ratification:** Claude42 raised; reviewer consensus in v1.2.

---

### AD-40 — `group_by_quantiles=True` guard bundled as P2 bugfix

**Phase:** 13.18.DF | **Source:** `PHASE_13_18_DF_v1_2_Proposal_RobustStats.md` §AD table | **Date:** 2026-04-29

**Decision text (verbatim):**
> `group_by_quantiles=True` guard: **Bundle** as P2 bugfix

**Ratification:** 4/5 reviewers.

**Source references:**
- `drawer.py:1287` — `- boolean True passed (must be integer, per AD-40 pattern)`
- `plots/profile.py:321` — `# Phase 13.18.DF (AD-40): guard against boolean True (must be integer)`

---

### AD-41 — profile() summary stats included; per-bin median/MAD deferred

**Phase:** 13.18.DF | **Source:** `PHASE_13_18_DF_v1_2_Proposal_RobustStats.md` §AD table | **Date:** 2026-04-29

**Decision text (verbatim):**
> profile() stats: **Summary stats included** (same code as hist2d); per-bin median/MAD deferred

**Motivation:** Include the same summary stat computation in profile() as in hist2d. Per-bin robust statistics requires restructuring `_compute_profile()` — deferred.

**Ratification:** Architect directive + Claude40 recommendation.

**Source references:** *(no inline AD-41 comment — expressed by implementation scope)*

---

### AD-42 — Reserve `fit=` kwarg with `NotImplementedError` guard

**Phase:** 13.18.DF | **Source:** `PHASE_13_18_DF_v1_2_Proposal_RobustStats.md` §AD table | **Date:** 2026-04-29

**Decision text (verbatim):**
> Reserve `fit=` kwarg: **Yes** — `NotImplementedError` guard

**Motivation:** Reserves the parameter name for Phase 13.42. Raises clear error rather than silent kwarg ignore. Guard removed when Phase 13.42 ships.

**Ratification:** 5/5 reviewers + o2DistAI reviewer unanimous.

**Source references:**
- `drawer.py:4316` — `# Phase 13.42.DF: AD-42 guard removed; 'fit=' is now the inline-fit`

---

### AD-43 — Defer `stat_fields='core'` to v2

**Phase:** 13.18.DF | **Source:** `PHASE_13_18_DF_v1_2_Proposal_RobustStats.md` §AD table | **Date:** 2026-04-29

**Decision text (verbatim):**
> Defer `stat_fields='core'` to v2

**Motivation:** `stat_fields='core'` would be shorthand for the most common use case. Deferred to keep Phase 13.18 scope manageable.

**Ratification:** 5/5 reviewers + o2DistAI reviewer unanimous.

**Source references:** *(no inline AD-43 comment)*

---

## Phase 13.25.DF — Quantiles on profile (AD-44 to AD-54)

**Phase trigger:** Architect needs `quantiles=[…]` on `profile()` for batch calibration QA. Heavy-tailed distributions make `std` unreliable. Brainstorm v1.4 (2026-04-30) established the framework. Phase A = zero-channel-cost modes (error_bars, band) only.

---

### AD-44 — Phase A scope: `quantiles=` on `profile()` with error_bars + band only

**Phase:** 13.25.DF | **Source:** `PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md` §AD table | **Date:** 2026-05-03+

**Decision text (verbatim):**
> Phase A scope = `quantiles=` on `profile()` with error_bars + band only

**Motivation:** Zero-channel-cost modes per brainstorm v1.4 §3.1 Step 0. `quantiles=` provides IQR and custom quantile bands without needing scipy.

**Ratification:** Session 5 M-Q1. ✅ Architect-approved.

**Source references:**
- `channels.py:21` — block reference: `AD-44 through AD-59 — see docs/STYLING_FRAMEWORK_DECISIONS.md`
- `style.py:170` — same block reference

---

### AD-45 — `central='mean'` default; `None` sentinel resolves via style key

**Phase:** 13.25.DF | **Source:** `PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md` §AD table | **Date:** 2026-05-03+

**Decision text (verbatim):**
> `central='mean'` default

**Note (from proposal):** Signature literal default is `None`, which resolves to `'mean'` via `quantile.central_default` style key. The public-documented default is `'mean'`.

**Ratification:** Session 5 Q-Q2. ✅ Architect-approved.

**Source references:**
- `style.py:104` — `# behavior per AD-45.`

---

### AD-46 — 4×4 interaction matrix is the test specification

**Phase:** 13.25.DF | **Source:** `PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md` §AD table + §P3.4 | **Date:** 2026-05-03+

**Decision text (verbatim):**
> 4×4 matrix is test spec

**Motivation:** The 4×4 matrix of `central=` × `quantile_mode=` interactions defines all valid and invalid combinations exhaustively. Machine-checkable completeness.

**Ratification:** Session 5 Q-Q2. ✅ Architect-approved.

**Source references:** *(expressed in test structure — no inline AD-46 comment)*

---

### AD-47 — `error_bars` locked as 4th error mode

**Phase:** 13.25.DF | **Source:** `PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md` §AD table + §P3.1/§P3.2 | **Date:** 2026-05-03+

**Decision text (verbatim):**
> `error_bars` locked as 4th mode

**Meaning:** The existing `error=` kwarg (`'sem' | 'std' | 'none'`) gains `'quantile'` as a fourth value, extending the existing error-mode slot.

**Ratification:** Session 5 C1. ✅ Architect-approved.

**Source references:** *(no inline AD-47 comment)*

---

### AD-48 — Quantile binning lives in dfdraw (not ADF)

**Phase:** 13.25.DF | **Source:** `PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md` §AD table | **Date:** 2026-05-03+

**Decision text (verbatim):**
> Quantile binning lives in dfdraw

**Motivation:** Architectural boundary — per-bin quantile computation belongs in dfdraw, not AliasDataFrame. ADF passes data; dfdraw computes all per-bin statistics. Single source of truth principle (brainstorm P-Q3).

**Ratification:** Brainstorm P-Q3. ✅ Architect-approved.

**Source references:**
- `plots/profile.py:1488` — `Single source of truth for quantile computation per AD-48.`

---

### AD-49 — `fill_style` is renderer machinery in Phase A

**Phase:** 13.25.DF | **Source:** `PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md` §AD table + §P3.6 | **Date:** 2026-05-03+

**Decision text (verbatim):**
> `fill_style` is renderer machinery in Phase A

**Meaning:** `fill_style` is internal renderer state for `_render_quantile_band()`; NOT in `_VALID_STYLE_CHANNELS`; NOT a user-facing parameter in Phase A.

**Ratification:** Session 5 M-Q1. ✅ Architect-approved.

**Source references:** *(no inline AD-49 comment)*

---

### AD-50 — Cached `_last_ax` only (NOT cached DFDraw); color cycle resets per-figure

**Phase:** 13.25.DF | **Source:** `PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md` §AD table + §P3.5 | **Date:** 2026-05-03+; revised v1.2 + v1.3

**Decision text (verbatim, v1.3 final):**
> P-Q4: cached `_last_ax` (Option C1b) — **NOT cached DFDraw** — fixes `same=True` continuity. Color cycle behavior unchanged from current source.

**Relationship to AD-37:** AD-50 revises the implementation strategy proposed in AD-37. The original problem identified by AD-37 remains valid; caching the complete `DFDraw` instance is rejected because it would break per-call filtering. The admissible pending solution is the `_last_ax`-only pattern described here.

**Motivation:** Per architect Session 5: *"Not clear. Computing experts should decide."* dfdraw + ADF teams co-decided Option C1b. ADF constructs `DFDraw(df_subset)` where `df_subset` is per-call-filtered at `AliasDataFrame.py:10870, 11868, 12348`. Caching full DFDraw would lose per-call filtering.

**Rejected alternatives:**
- Option C1a: cached DFDraw instance in AliasDataFrame — rejected; breaks per-call filtering semantics
- Option A: `plt.gca()` only — unreliable in Jupyter

**Version history:**
- v1.1: incorrect characterization of `_last_ax` behavior
- v1.2: corrected per Claude49 source verification
- v1.3: C1b implementation pattern added per Claude38 P1-1

**Ratification:** Pending Claude49 + ADF team co-sign at v1.3 review. Status: OPEN.

**Source references:**
- `tests/test_quantiles_profile.py:399` — `@pytest.mark.skip(reason="AD-50 ADF-side Option C1b not yet shipped")`

---

### AD-51 — `central='none'` + `error_bars` raises ValueError

**Phase:** 13.25.DF | **Source:** `PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md` §AD table + §P3.4 | **Date:** 2026-05-03+

**Decision text (verbatim):**
> `central='none'` + `error_bars` raises ValueError

**Motivation:** `error_bars` mode requires a central line to attach bars to. Geometrically incoherent without one.

**Ratification:** ✅ Locked (Claude40 + Claude49).

**Source references:**
- `plots/profile.py:405` — `# AD-51: central='none' + error_bars is invalid`

---

### AD-52 — Default `error="sem"` → `"quantile"` rebinding when `quantiles=` set

**Phase:** 13.25.DF | **Source:** `PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md` §AD table + §P3.3 | **Date:** 2026-05-03+

**Decision text (verbatim):**
> Default `error="sem"` → `"quantile"` rebinding

**Meaning:** When `quantiles=` is set and `error=` is at default `"sem"`, internally rebind to `"quantile"` for `error_bars` mode only. Explicit `error="sem"` is honored (uses `None` sentinel to distinguish).

**Ratification:** ✅ Locked (Claude40 + Claude49).

**Source references:**
- `plots/profile.py:427` — `# AD-52 FIX1: rebind error only when user did NOT explicitly set it.`
- `tests/test_quantiles_profile.py:5` — `I-1: AD-52 error=None sentinel (distinguish explicit from default)`
- `tests/test_quantiles_profile.py:337, 351` — FIX1 sentinel behavior tests

---

### AD-53 — 4 `quantile.*` style keys independent; no cascading from `profile.*`

**Phase:** 13.25.DF | **Source:** `PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md` §AD table + §P3.8 | **Date:** 2026-05-03+; revised v1.2

**Decision text (verbatim, v1.2/v1.3 final):**
> 4 `quantile.*` keys in Phase A; **independent (no cascading from `profile.*`)**

**Motivation:** Consistent with existing dfdraw pattern (`grid.alpha`, `scatter.alpha`, `hist.alpha` are all independent siblings — verified at `style.py:37,41,48,73`).

**Version history:**
- v1.1: cascading from `profile.*` was considered
- v1.2: rejected per Claude49 + Claude37 convergence

**Ratification:** Pending Claude49 flip at v1.3 review.

**Source references:**
- `style.py:83` — `# Phase 13.25.DF (Phase A): Quantile rendering style keys (AD-53)`
- `plots/profile.py:1680` — `Capsize read from quantile.error_bars.capsize style key (AD-53).`
- `plots/profile.py:1712` — `style keys (AD-53). Band color matches the central line's color.`
- `tests/test_quantiles_profile.py:264` — `"""AD-53 lock: rendered cap sizes must match each key independently.`

---

### AD-54 — Naming conventions locked

**Phase:** 13.25.DF | **Source:** `PHASE_13_25_DF_v1_3_Proposal_QuantilesProfile.md` §AD table + §P3.9 | **Date:** 2026-05-03+

**Decision text (verbatim):**
> Naming conventions locked

**Meaning:** Parameter names (`quantiles=`, `central=`, `quantile_mode=`), style key prefix (`quantile.*`), and mode names (`'error_bars'`, `'band'`, `'discrete'`) are locked — cannot change without [BREACH] disclosure.

**Ratification:** ✅ Locked (Claude40 + Claude49).

**Source references:** *(expressed throughout naming — no single inline AD-54 comment)*

---

## Phase 13.26.DF — N-Channel Framework Phase B (AD-55 to AD-60)

**Phase trigger:** Collision problem — 3 active channels (vector × group_by × quantiles discrete) produce indistinguishable lines (vector and quantiles both default to `linestyle`). Algorithm A (brainstorm v1.4) resolves this.

**Key finding (v1.2 amendment):** Zero production quantile users (grep-verified against `makeSmoothMapsWithTPC.py`). Full greenfield design freedom for quantile path.

**Verbatim architect quotes (2026-05-05, typos preserved per GP-3):**
> *"is the proposal sufficiently configurable by style so we can define different use cases? Can we start? If style can parameterize I am happy - wan we need good defuat style"*
> *"We need styles and default style from the begining. we should be able to change style. Has to be done and not psoponed. If it is parametrized I do not need to asnwer in details on the questions. Parametrizrtion si important. We can start with defaults as you proposed - Once we will have style we can easly change. We have to have style to be able to change ..."*
> *"We did not use quentiles yet. We do not need to be back compatible. for quentiles"*
> Session 5 A-Q1: *"I assume this will be parameter in style. I agree that the proposed order is good default."*

---

### AD-55 — Channel priority order: `["color", "linestyle", "marker"]`

**Phase:** 13.26.DF | **Source:** `PHASE_13_26_DF_v1_2_Proposal_NChannelFramework.md` §AD table | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Priority order: Default `["color", "linestyle", "marker"]` — changeable via `channels.priority.categorical`

**Motivation:** Color is the strongest visual discriminator (maximum contrast, largest preattentive pop-out). Linestyle second. Marker third. All changeable via style per architect direction.

**Ratification:** Per architect direction. Reviewer consensus resolved AD-55 through AD-59 as "use proposed defaults, all changeable via `set_style()`."

**Source references:**
- `style.py:174` — `# data-channel combinations). AD-55.`
- `tests/test_channel_assignment.py:10` — module header citing AD-55 through AD-59

---

### AD-56 — 3-channel assignment: `group_by→color, vector→marker, quantiles→linestyle`

**Phase:** 13.26.DF | **Source:** `PHASE_13_26_DF_v1_2_Proposal_NChannelFramework.md` §AD table | **Date:** 2026-05-06+

**Decision text (verbatim):**
> 3-channel assignment: Explicit-case rule (Option C): `group_by→color, vector→marker, quantiles→linestyle` — changeable via `channels.default.*`

**Motivation:** Vector cardinality bounded (≤ 8, fits marker cycle). Quantiles ordinal (linestyle natural). `group_by` primary categorical dimension gets color. Implemented in `EXPLICIT_RULES` dict (append-only per GP-2).

**Ratification:** Reviewer consensus.

**Source references:**
- `channels.py:96` — `# Provenance: AD-56 (3-channel default), brainstorm v1.4 §13 deliverable table.`
- `channels.py:110` — `# 3-channel case (AD-56)`
- `style.py:188` — `# AD-56.`
- `plots/profile.py:808` — `# picked by Algorithm A (default 'linestyle' per AD-56). The`
- `tests/test_channel_assignment.py:184` — `"""3-channel default assignment per AD-56."""`

---

### AD-57 — Nested-band auto-detection: `'nested_band'` for ≥2 symmetric pairs

**Phase:** 13.26.DF | **Source:** `PHASE_13_26_DF_v1_2_Proposal_NChannelFramework.md` §AD table | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Nested-band auto-detection: Default `'nested_band'` for ≥2 symmetric pairs (with or without 0.5) — user overrides with `quantile_mode='discrete'`

**Motivation:** Multi-pair symmetric quantile lists produce most readable nested-band visualizations. This is greenfield (zero production quantile users). Max 3 nested bands — silent truncation of outermost.

**Ratification:** Reviewer consensus. No hard lock — changeable via `set_style()`.

**Source references:**
- `plots/profile.py:781` — `# Phase 13.26.DF Phase B (Option A, AD-57): nested alpha-stacked`
- `plots/profile.py:1456` — `# Phase 13.26.DF Phase B (Option A, AD-57): multiple symmetric pairs`
- `plots/profile.py:1741` — docstring: `Phase 13.26.DF Phase B (Option A, AD-57).`
- `plots/profile.py:1782` — `# Max 3 nested bands per AD-57 / v1.2 §7.2 (silent truncation, outermost 3).`
- `tests/test_quantiles_profile.py:181` — `"""Multi-pair symmetric (≥2 pairs) → nested_band mode (AD-57, Phase 13.26.DF)."""`
- `tests/test_channel_assignment.py:473` — `"""Nested-band detection (Option A, AD-57) and rendering."""`

---

### AD-58 — Overflow behavior default `"error"` with actionable suggestions

**Phase:** 13.26.DF | **Source:** `PHASE_13_26_DF_v1_2_Proposal_NChannelFramework.md` §AD table | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Overflow behavior: Default `"error"` — changeable via `channels.overflow`

**Meaning:** When active channels exceed the framework's capacity (3 visual + 1 spatial), raises error with actionable suggestions (`top_k=`, `facet_by=`, etc.). Changeable to `"warn"` or `"clip"` via style.

**Ratification:** Reviewer consensus.

**Source references:**
- `style.py:194` — `# assigned visual channel's capacity. AD-58.`
- `tests/test_phase_13_27_facet_refactor.py:246` — `# Ensure overflow is 'error' (default per Phase 13.26 AD-58)`
- `tests/test_channel_assignment.py:299` — `"""Capacity check per v1.2 §3.1 Step 5 + AD-58."""`

---

### AD-59 — Factored legend default `True`

**Phase:** 13.26.DF | **Source:** `PHASE_13_26_DF_v1_2_Proposal_NChannelFramework.md` §AD table | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Legend style: Default `True` (factored) — changeable via `channels.legend.factored`

**Meaning:** Multiple channels → one legend entry per channel value (not per combination). 3 channels with cardinalities 5/3/5 → 11 entries vs 75 unfactored.

**Ratification:** Reviewer consensus.

**Source references:**
- `style.py:199` — `# Legend factoring. AD-59.`
- `tests/test_channel_assignment.py:522` — `"""Factored legend (AD-59) — sum-not-product entry count."""`

---

### AD-60 — FIX2 visual elements preserved as channel-aware defaults

**Phase:** 13.26.DF Commit 2 / FIX2 | **Source:** `PHASE_13_26_DF_v1_0_END_CODE_REVIEW_REQUEST.md` §4.4; `STYLING_FRAMEWORK_DECISIONS.md §5` | **Date:** 2026-05-06+

**Decision text (verbatim):**
> FIX2 visual elements (on-line percentage annotations + linestyle cycle for discrete quantiles) preserved as channel-aware default for `quantile_style='linestyle'`. Solid linestyle reserved for central line via `[1:]` slice. Annotations suppressed for marker/color styles (self-disambiguating).

**Ratification:** CRR §4.4 note; STYLING_FRAMEWORK_DECISIONS.md AD-60 entry.

**Source references:** *(no inline AD-60 comment — expressed via channels.cycles.linestyle[1:] pattern)*

---

## Phase 13.27.DF Commit 2 — Selection/Weights/DeltaFacet (AD-61 to AD-68)

**Phase trigger:** Production scripts (`makeSmoothMapsWithTPC.py`) need multiple cuts of the same plot (different `abs(trkAngle-tgSlp)` thresholds, weights, sector overlays). Requires manual loops today. `selection_vector`/`weights_vector` make it declarative.

---

### AD-61 — `selection_delta`/`weights_delta` ordinal default; kwarg-controlled

**Phase:** 13.27.DF Commit 2 | **Source:** `PHASE_13_27_DF_v1_2_Proposal_SelectionWeightDeltaFacet.md` §10 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> `selection_delta` and `weights_delta` semantics — ordinal default, kwarg-controlled (per architect Q9 2026-05-06)

**Motivation:** Production cuts are ordered (loose-to-tight); ordinal encoding (linestyle) conveys order naturally. Architect Q9: partition-first-then-vector composition is correct semantics (matches Vega-Lite "facet defines panel, vector defines curves").

**Ratification:** Architect Q9 confirmation 2026-05-06. Originally proposed in v1.0; preserved unchanged in v1.2.

**Source references:**
- `channels.py:117` — `# AD-61: selection_delta and weights_delta are introduced as new`
- `drawer.py:1631` — `# not added to _channels). AD-61: is_categorical controlled by`
- `drawer.py:4176, 4527, 4944, 5302` — `# AD-61, AD-62, AD-65, AD-66, AD-67. Method body wiring lands in Turn 3.`
- `style.py:209, 214`
- `tests/feature_taxonomy.py:754, 765`

---

### AD-62 — Composition: inner default; outer opt-in via `vector_compose='outer'`

**Phase:** 13.27.DF Commit 2 | **Source:** `PHASE_13_27_DF_v1_2_Proposal_SelectionWeightDeltaFacet.md` §10 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Composition rule: inner default mirrors `[Y1,Y2]:X` semantics; outer opt-in via `vector_compose='outer'`

**Motivation:** `[Y1,Y2]:X` multi-expression syntax (Phase 13.26) uses inner/zip composition. `selection_vector`/`weights_vector` mirror this. Outer product available but requires explicit opt-in to prevent accidental combinatorial explosion.

**Ratification:** Originally proposed in v1.0; preserved unchanged in v1.2.

**Source references:**
- `channels.py:121` — `# AD-62: composition rule (inner default / outer opt-in) governs how the`
- `style.py:218` — `# AD-62 (NEW IN v1.1 §5.4).`
- `drawer.py:4176, 4527, 4944, 5302`
- `tests/feature_taxonomy.py:776, 787`

---

### AD-63 — Facet as 4th visual encoding ("spatial channel")

**Phase:** 13.27.DF Commit 1 / 13.31 / 13.32 | **Source:** `PHASE_13_27_DF_v1_2_Proposal_SelectionWeightDeltaFacet.md` §10 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Facet as 4th visual encoding ("spatial channel") — `'facet'` in `_VALID_STYLE_CHANNELS`, treated by Algorithm A like color/linestyle/marker

**Meaning:** `facet_by` is not a special case. It is a fourth channel type in Algorithm A's resolution chain.

**Ratification:** Originally proposed in v1.0; realized via Phase 13.27 Commit 1 + 13.31 + 13.32.

**Source references:**
- `style.py:222` — `# legend entries. Used by Commit 2 (selection/weights vectors). AD-63, AD-64.`

---

### AD-64 — API surface: `facet_by='<channel>'` primary; `facet=True` legacy alias

**Phase:** 13.27.DF Commit 1 / 13.31 / 13.32 | **Source:** `PHASE_13_27_DF_v1_2_Proposal_SelectionWeightDeltaFacet.md` §10 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> API surface for facet: `facet_by='<channel>'` primary + `facet=True` legacy alias + `<channel>_style='facet'` internal — all three accepted, collision is error

**Ratification:** Originally proposed in v1.0; preserved unchanged in v1.2.

**Source references:**
- `style.py:222` — same comment as AD-63

---

### AD-65 — Visual-channel budget: max 4 (3 visual + 1 spatial)

**Phase:** 13.27.DF Commit 2 | **Source:** `PHASE_13_27_DF_v1_2_Proposal_SelectionWeightDeltaFacet.md` §7 + §10 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Visual-channel budget: max 4 encodings (3 visual + 1 spatial); 5+ channel cases refuse with actionable error

**Motivation:** 4 simultaneous visual dimensions is the practical readable maximum for physics plots. Beyond 4, the plot is uninterpretable. Actionable error guides the user.

**Ratification:** Originally proposed in v1.0; preserved unchanged in v1.2.

**Source references:**
- `channels.py:123` — `# AD-65/66/67: 4-channel non-facet cases intentionally omitted — they`
- `drawer.py:4176, 4527, 4944, 5302`
- `style.py:227` — `# Used by Commit 2. AD-65, AD-66.`

---

### AD-66 — Logical composition: AND for selections, MULTIPLY for weights

**Phase:** 13.27.DF Commit 2 | **Source:** `PHASE_13_27_DF_v1_2_Proposal_SelectionWeightDeltaFacet.md` §10 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Logical composition: `selection ∧ selection_vector[i]` for filters; `weights × weights_vector[i]` for weights

**Meaning:** Base `selection` AND each vector element (restricts within base). Base `weights` TIMES each vector element (scales the base weight).

**Ratification:** Originally proposed in v1.0; preserved unchanged in v1.2.

**Source references:**
- `drawer.py:4176, 4527, 4944, 5302`
- `style.py:227`
- `tests/feature_taxonomy.py:754, 765`

---

### AD-67 — 1-element list kwargs silently degrade to scalar (cost-0 channel)

**Phase:** 13.27.DF Commit 2 | **Source:** `PHASE_13_27_DF_v1_2_Proposal_SelectionWeightDeltaFacet.md` §10 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> 1-element list-valued kwargs silently degrade to scalar (cost-0 channel)

**Meaning:** `selection_vector=["sector==1"]` (1-element list) is treated identically to `selection="sector==1"` (scalar). No channel consumed. Avoids penalizing users who conditionally build lists.

**Ratification:** Originally proposed in v1.0; preserved unchanged in v1.2.

**Source references (17 occurrences — most widely cited non-AD-78 pattern):**
- `drawer.py:1121` — FORWARDED_NAMES `# ...(AD-67)`
- `drawer.py:1344, 1353` — `per AD-67: 1-element silently degrades to scalar`
- `drawer.py:1630` — `# AD-67: 1-element list silently degrades to scalar (cost 0,`
- `drawer.py:4176, 4527, 4944, 5302, 4779, 5160, 5292, 5852`
- `tests/test_phase_13_27_commit2_selection_weights.py:492` — `"""§9.CIO.7: 1-element list degrades silently (AD-67) — cost-0 channel."""`
- `tests/feature_taxonomy.py:754, 765`

---

### AD-68 — 2D faceting deferred to Phase E

**Phase:** 13.27.DF Commit 1 / 13.31 / 13.32 | **Source:** `PHASE_13_27_DF_v1_2_Proposal_SelectionWeightDeltaFacet.md` §10 + §13 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> 2D faceting deferred to Phase E — Phase 13.27 supports 1D facet only

**Motivation:** 2D faceting has substantial design effort with subtle semantic questions (ROW/COL ordering, figure-level vs axes-level layout, shared-axis contract in 2D). Reviewer Q5: *"AD-68 should be LOCKED — 2D facet is a substantial design effort, not casual extension."*

**Ratification:** Locked per reviewer Q5. Originally proposed in v1.0; preserved unchanged in v1.2.

**Source references:**
- `drawer.py:3691` — `# AD-68: ax provided per-subplot; selection already applied at top`
- `style.py:209` — `# extends to facet via channels.cycles.facet_max. AD-61..AD-68.`

---

## Phase 13.28.DF — Robust Data Handling (AD-69 to AD-77)

**Phase trigger:** Three production failure modes from Phase 13.26 real-data validation on TPC calibration scripts: (1) `inf` in expression (`y/x` with `x==0`) crashed matplotlib; (2) NaN in column produced silent `n=0`/empty histogram; (3) `stats['n']` reflected full selection even with `range=` set.

**Verbatim architect principle:**
> *"the user is not responsible for curating data produced by automated pipelines (reconstruction, online calibration, fit failures). dfdraw should be robust by default and configurable when the user wants strict mode."*
> *"do not cut PDF without outliers"* (2026-05-06)

---

### AD-69 — Centralized `_data_sanitize.py` module

**Phase:** 13.28.DF | **Source:** `PHASE_13_28_DF_v1_0_Proposal_RobustDataHandling.md` §AD table; confirmed v1.1 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Centralized sanitization module `plots/_data_sanitize.py` — uniform handling across hist/hist2d/hexbin/profile/scatter

**Motivation:** All 5 plot types benefit from uniform NaN/inf semantics. Single implementation prevents drift.

**Ratification:** v1.0. Confirmed v1.1.

**Source references:**
- `plots/_data_sanitize.py:10` — module header: `- AD-69 (centralized sanitization module)`
- `plots/histogram.py:415, 1258, 1502`
- `plots/profile.py:486`
- `plots/scatter.py:176`
- `tests/test_data_sanitize_autorange.py:11` — test module header

---

### AD-70 — `nan_policy` default `'filter'`; alternatives `'warn'`, `'raise'`

**Phase:** 13.28.DF | **Source:** `PHASE_13_28_DF_v1_0_Proposal_RobustDataHandling.md` §AD table; confirmed v1.1 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> `nan_policy` parameter — default `'filter'` (backward compat); alternatives `'warn'`, `'raise'`. Per-call kwarg + per-process style key

**Verbatim architect quote:** *"do not cut PDF without outliers"* (2026-05-06)

**Ratification:** v1.0 per architect Q1. Confirmed v1.1.

**Source references (21 occurrences — 4th most cited AD):**
- `plots/_data_sanitize.py:11` — module header: `- AD-70 (nan_policy default 'filter')`
- `drawer.py:1120, 1157, 1206, 1245, 4165, 4506, 4926, 5290, 6000` — FORWARDED_NAMES + entry points
- `plots/histogram.py:216, 415, 1173, 1258, 1422, 1502`
- `plots/profile.py:203, 486`
- `plots/scatter.py:53, 176`

---

### AD-71 — Counter keys always populated regardless of `nan_policy`

**Phase:** 13.28.DF | **Source:** `PHASE_13_28_DF_v1_0_Proposal_RobustDataHandling.md` §AD table; confirmed v1.1 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Counter keys (`n_input`, `n_filtered`, `n_inf_*`, `n_nan_*`) always populated regardless of nan_policy. Additive — existing keys unchanged

**Motivation:** Diagnostic visibility even in `'filter'` mode for downstream programmatic inspection.

**Ratification:** v1.0 per architect Q5. Confirmed v1.1.

**Source references:**
- `plots/_data_sanitize.py:12` — module header: `- AD-71 (counters always populated)`
- `plots/histogram.py:477, 1281, 1511`
- `plots/profile.py:532`
- `plots/scatter.py:231`

---

### AD-72 — Hybrid autorange algorithm formal definition

**Phase:** 13.28.DF | **Source:** `PHASE_13_28_DF_v1_0_Proposal_RobustDataHandling.md` §4.1 + §AD table; confirmed v1.1 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Hybrid autorange algorithm formal definition (§4.1) — combines robust window with `(min, max)` based on per-side outlier detection

**Motivation:** `(min, max)` destroyed by single outliers. Percentile clipping trims clean distributions arbitrarily. Hybrid: clean data → full range; outlier-bearing → robust bounds on affected side only.

**Verbatim architect:** *"do not cut PDF without outliers"*

**Ratification:** v1.0. Confirmed v1.1.

**Source references:**
- `plots/_autorange.py:11` — module header: `- AD-72 (hybrid algorithm formal definition)`

---

### AD-73 — Default autorange strategy `'hybrid'`

**Phase:** 13.28.DF | **Source:** `PHASE_13_28_DF_v1_0_Proposal_RobustDataHandling.md` §AD table; confirmed v1.1 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Default autorange strategy `'hybrid'`. Alternatives: `'minmax'` (backward compat), `'percentile_99'`, `'percentile_95'`, `'robust_3mad'`, `'robust_4mad'`. Set via style key

**Verbatim architect motivation:** *"do not cut PDF without outliers"* — `'hybrid'` is the minimum-surprise default that does the right thing for both clean and outlier-bearing distributions.

**Ratification:** v1.0 per architect Q2. Confirmed v1.1.

**Source references:**
- `plots/_autorange.py:12, 20` — module header + `# Valid strategy preset names (AD-73)`
- `style.py:73, 78` — `# Phase 13.28.DF FIX1: autorange.* style keys (AD-73 / AD-77).` + default assignment
- `plots/histogram.py:455, 1263`
- `plots/profile.py:515`

---

### AD-74 — 2D autorange per-axis independent

**Phase:** 13.28.DF | **Source:** `PHASE_13_28_DF_v1_0_Proposal_RobustDataHandling.md` §4.2 + §AD table; confirmed v1.1 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> 2D autorange per-axis independent — no cross-axis correlation in outlier decision (per architect Q7 2026-05-06)

**Verbatim architect:** *"Independent"* (Q7 2026-05-06)

**Ratification:** v1.0 per architect Q7. Confirmed v1.1.

**Source references:**
- `plots/_autorange.py:13, 258, 298` — module header + `per-axis independent — AD-74`
- `plots/histogram.py:1263`

---

### AD-75 — Backward compat: existing tests use explicit `range='minmax'`

**Phase:** 13.28.DF | **Source:** `PHASE_13_28_DF_v1_0_Proposal_RobustDataHandling.md` §6.1 + §AD table; confirmed v1.1 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Backward compat — existing tests use explicit `range='minmax'` to lock pre-Phase-13.28 behavior (per architect Q10 2026-05-06)

**Verbatim architect:** *"In old test we can use explicitly old autorange"* (Q10 2026-05-06)

**Ratification:** v1.0 per architect Q10. Confirmed v1.1.

**Source references:** *(expressed in test annotations — no single inline AD-75 comment)*

---

### AD-76 — Strategy parameters tunable via style keys only in v1.0

**Phase:** 13.28.DF | **Source:** `PHASE_13_28_DF_v1_0_Proposal_RobustDataHandling.md` §AD table; confirmed v1.1 | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Strategy parameters (`k_robust=4.0`, `k_outlier=1.5`) tunable via style keys only in v1.0. Per-call strategy-parameter kwarg override deferred to Phase 13.29

**Motivation:** Keeps Phase 13.28 scope manageable. Style key tuning covers the production use case.

**Ratification:** v1.0. Confirmed v1.1.

**Source references:** *(no inline AD-76 comment)*

---

### AD-77 — Diagnostic stats keys always populated; `stats['n']` semantics locked

**Phase:** 13.28.DF v1.1 (NEW in v1.1) | **Source:** `PHASE_13_28_DF_v1_1_Proposal_RobustDataHandling.md` §AD table | **Date:** 2026-05-06+

**Decision text (verbatim):**
> Diagnostic stats keys (`autorange_used`, `autorange_strategy`) always populated; `autorange_strategy='explicit'` when user passes numeric range. `stats['n']` semantics: finite count after selection + sanitize, NOT clipped to range (§6.4)

**Motivation:** Without surfacing `autorange_used` and `autorange_strategy`, the hybrid autorange is opaque — production QA cannot verify what range was actually used. The `stats['n']` semantics clarification: `n` counts finite points after selection and sanitization; range filtering is visual only and does NOT reduce `n`.

**Origin:** Added in v1.1 after GPT4 #1 + #2 convergent P1 findings; architect overrode Claude40's "no revision needed" verdict.

**Ratification:** v1.1 per architect amendment post-review (2026-05-06).

**Source references:**
- `plots/_autorange.py:14, 152` — module header + `Caller stores in stats['autorange_used'] (AD-77).`
- `style.py:73` — `# Phase 13.28.DF FIX1: autorange.* style keys (AD-73 / AD-77).`
- `plots/histogram.py:455, 477, 1263, 1281`
- `plots/profile.py:515, 532`
- `tests/test_data_sanitize_autorange.py:380` — `n stats reflect post-sanitize count, NOT range-clipped count (per AD-77 / §6.4).`
- `tests/feature_taxonomy.py:863` — `"Stats keys autorange_used + autorange_strategy (AD-77)"`

---

## Phase 13.31.DF — `facet_by` column-name mode (AD-78)

**Phase trigger:** Production reproducer (M. Ivanov, 2026-05-12):
```python
adfSec.draw("dyp_I0345_rms:row", facet_by="side", ...)
# Phase 13.27 Commit 1: ValueError: facet_by must be one of (group_by, vector, quantiles)
# Phase 13.31: works — 2 subplots, each with 5 quantile-band overlays
```
The natural mental model for physicists is column-name semantics (matching ggplot2 `facet_wrap(~side)`).

---

### AD-78 — `facet_by` accepts column names (dual-path tagged union)

**Phase:** 13.31.DF | **Source:** `PHASE_13_31_DF_v1_0_Proposal_FacetByColumnName.md` §1 + §2; `AD_facet_by_dual_path.md` | **Date:** 2026-05-12

**Decision text (verbatim):**
> `facet_by` accepts DataFrame column names in addition to channel-name enum. Disambiguation order: channel-enum first, then column check. Channel-enum and `df.columns` are disjoint by construction.

**Motivation:** Direct production need. Validation moved to `_dispatch_faceted_render()` rather than `_PROFILE_COLUMN_REFERENCES` — this was a Sonnet P1 cross-review finding: adding `facet_by` to `_PROFILE_COLUMN_REFERENCES` would break every existing `facet_by="group_by"` call because Phase 13.30's validator raises if value not in `df.columns`.

**Ratification:** AD-78 signed 2026-05-12. Confirmed AD-78 numbering in cross-review (Sonnet, 2026-05-12).

**Source references (30 occurrences — most cited AD in codebase, 7 files):**
- `drawer.py:1450-1454` — `# Phase 13.31.DF (AD-78) NOTE: 'facet_by' is a tagged union... 'facet_by' must NOT be added to these tuples — Sonnet P1 catch in AD-78`
- `drawer.py:3241-3253` — full disambiguation algorithm comment block
- `drawer.py:3390, 3681, 3717, 3843` — column-mode facet implementation sites
- `drawer.py:1159, 1207` — FORWARDED_NAMES for hist, scatter
- `tests/test_phase_13_31_facet_by_column.py:13, 23, 44, 122, 160, 184, 189, 263` — test module header + test bodies
- `tests/test_phase_13_27_facet_refactor.py:200, 210, 214` — updated for AD-78
- `tests/feature_taxonomy.py:907, 911`

---

## Phase 13.32.DF — `group_by × quantiles` + symmetric `facet_by` binning (AD-79)

**Phase trigger:** Two production incidents: (A) `quantiles=` silently dropped when `group_by=` is set; (B) `facet_by='group_by'` ignores `group_by_bins/_quantiles`. Plus missing `facet_by_bins`/`facet_by_quantiles` API.

---

### AD-79 — Symmetric `facet_by_bins`/`facet_by_quantiles` apply to all plot types

**Phase:** 13.32.DF | **Source:** `PHASE_13_32_DF_v1_2_Proposal_GroupByQuantilesFacetBinning.md` §0 changelog + §3.3 | **Date:** 2026-05-14

**Decision text (verbatim):**
> Symmetric `facet_by_bins`/`facet_by_quantiles` apply to all plot types (profile, hist, hist2d, scatter). Extension of Phase 13.12 F3, implied by AD-78 column-name semantics.

**Motivation:** AD-78 established column-name faceting. `facet_by_bins`/`facet_by_quantiles` is the natural symmetric extension — if you can facet by a column, you should be able to bin that column for faceting, just as `group_by_bins`/`group_by_quantiles` bins the group_by column (Phase 13.12 F3). Hoisting binning to `_dispatch_faceted_render` gives single implementation across all 4 plot types.

**Ratification:** AD-79 signed 2026-05-14.

**Source references (22 occurrences — 3rd most cited AD, 4 files):**
- `drawer.py:1122, 1160, 1208, 1247` — FORWARDED_NAMES in all 4 draw methods
- `drawer.py:1274, 3398, 3634, 3726, 4167, 4518, 4774, 4928, 5156, 5294, 5888, 6002, 6145` — dispatcher implementation sites
- `tests/test_phase_13_32_groupby_quantiles_facet.py:20, 372`
- `tests/feature_taxonomy.py:929, 933`

---

## Phase 13.33.DF — Normalized Differential Profiles (AD-80 to AD-82)

**Phase trigger:** Architect's daily calibration workflow — comparing residual profiles between fills, epochs, track-selection cuts requires 3 manual steps today. MDA foundation (CHEP 2023): σ(A ⊖ A_ref) ≤ σ(A) ⊕ σ(A_ref) — differential map is alarm-grade signal.

**Verbatim architect quotes (2026-05-16, typos preserved):**
> *"Deltas are very useful — only we should invent better user interface which will look natural."*
> *"Has to be supported from beginning — also facet_by should be supported. We should not postpone proper solution. Postponing means more work."*
> *"Yes we need pull mode also."*

---

### AD-80 — Reference convention: `vector[0]` = signal, `vector[1]` = reference

**Phase:** 13.33.DF | **Source:** `PHASE_13_33_DF_v1_1_Proposal_NormalizedDifferentialProfiles.md` §AD table | **Date:** 2026-05-16+

**Decision text (verbatim):**
> Reference convention: `vector[0]` = new/signal, `vector[1]` = reference. `delta = v[0] − v[1]`, `ratio = v[0] / v[1]`. Matches `"[dy_new, dy_ref]:row"` and ROOT `h_data.Divide(h_mc)`.

**Motivation:** Physicist's natural ordering: subject first, reference second. Matches ROOT `h_data.Divide(h_mc)` convention.

**Version history:** AD number renumbered from AD-70 to AD-80 in v1.1 (AD-69..77 taken by Phase 13.28).

**Ratification:** Architect D1 2026-05-16.

**Source references (26 occurrences — 2nd most cited AD, 6 files):**
- `plots/profile.py:1804, 1824` — `AD-80 sign convention: stats_0 is the signal (vector[0]), stats_1 is the reference`
- `drawer.py:2280` — `# Per-curve label: AD-80 — signal first, reference second.`
- `drawer.py:5511` — `# (d) Exactly 2 vector elements required (AD-80: vector[0]=signal,`
- `style.py:241` — `# AD-80 fixes sign convention (vector[0]=signal, vector[1]=reference; delta = v[0] − v[1]).`
- `tests/test_normalize.py:648, 652, 668` — `"""§9.NSC.1 (AD-80 LOCK): when vector[0] mean > vector[1] mean, delta must be positive"""` + `f"AD-80 SIGN VIOLATION:..."`
- `tests/feature_taxonomy.py:1049`

---

### AD-81 — `group_by` + `facet_by` both in scope for v1.0; no deferral

**Phase:** 13.33.DF | **Source:** `PHASE_13_33_DF_v1_1_Proposal_NormalizedDifferentialProfiles.md` §AD table + §3.6 + §3.7 | **Date:** 2026-05-16+

**Decision text (verbatim):**
> `group_by` + `facet_by` both in scope for v1.0. No deferral. Per-group differential from day one.

**Verbatim architect quote:** *"Has to be supported from beginning — also facet_by should be supported. We should not postpone proper solution. Postponing means more work."*

**Version history:** AD number renumbered from AD-71 to AD-81 in v1.1.

**Ratification:** Architect D2 2026-05-16.

**Source references (10 occurrences):**
- `drawer.py:2189` — `see AD-81 stats contract)`
- `drawer.py:2434, 2457, 2462, 2787`
- `style.py:242` — `# AD-81 covers group_by + facet_by composition (per-group differential, K×2 facet grid).`
- `tests/test_normalize.py:167, 536`
- `tests/test_phase_13_34_df_m2_robustness.py:140`

---

### AD-82 — `normalize="pull"` with ±1σ, ±2σ bands

**Phase:** 13.33.DF | **Source:** `PHASE_13_33_DF_v1_1_Proposal_NormalizedDifferentialProfiles.md` §AD table + §3.5 | **Date:** 2026-05-16+

**Decision text (verbatim):**
> `normalize="pull"` included. Formula: `(v[0] − v[1]) / √(σ₀²/n₀ + σ₁²/n₁)`. With ±1σ, ±2σ bands.

**Motivation:** Statistical significance of the differential. The ±1σ and ±2σ bands provide immediate visual reference lines at standard significance thresholds.

**Verbatim architect quote:** *"Yes we need pull mode also."*

**Version history:** AD number renumbered from AD-72 to AD-82 in v1.1.

**Ratification:** Architect D3 2026-05-16.

**Source references:**
- `plots/profile.py:1982` — `# Phase 13.33.DF: Normalize-panel rendering (AD-82 pull bands; AD-80 ref line)`
- `style.py:243, 270` — `# AD-82 covers pull mode formula. Pull-mode bands (AD-82): ±1σ and ±2σ shaded regions`
- `tests/test_normalize.py:389` — `# Pull errors must be exactly 1.0 (AD-82 — pull is already in σ).`
- `tests/feature_taxonomy.py:1021`

---

---

# ─────────────────────────────────────────────────────────────────────────────
# AD-N/PHASE FORMAT — Phase 13.34.DF onward
# ─────────────────────────────────────────────────────────────────────────────

## AD-1/13.34.DF — Capability matrix updated at every phase end

**Phase:** 13.34.DF | **Date:** 2026-05-18 | **Original notation:** Q2

**Decision:** The capability matrix must be updated at the end of every phase. Registry update is part of every phase commit checklist.

**Motivation:** Phase 13.34 established the capability matrix as a live project artifact, not a periodic snapshot. Letting it drift between phases was identified as a governance gap — reviewers could not verify coverage claims.

**Ratification:** Architect Q2. `PHASE_13_34_DF_v1_0_Proposal_CapabilityMatrixRefresh.md` §10.

**Source:** `tests/feature_taxonomy.py`, `tests/test_layer_classification.py`

---

## AD-1/13.35.DF — `group_by_bins` + `hist_norm` must work on `hist()`

**Phase:** 13.35.DF | **Date:** 2026-05-20 | **Original notation:** Production call verbatim

**Decision:** `group_by_bins=` and `hist_norm=` must work on `hist()`. The acceptance criterion is that the architect's production call no longer raises `AttributeError`.

**Motivation — architect production call (2026-05-20, verbatim):**
```python
adf.draw("dyp_I6-dyp_recoV2", selection="(detType==1)&(sec<9)",
         type="hist", group_by="z", group_by_bins=5,
         min_entries=25, facet_by="sec")
# Was: AttributeError: Polygon.set() got an unexpected keyword argument 'group_by_bins'
```

**Ratification:** Architect production call. `PHASE_13_35_DF_v1_3_Proposal_HistGroupByNorm.md` §3.1.

**Source:** `plots/histogram.py`, `drawer.py:_HIST_FORWARDED_NAMES`

---

## AD-1/13.36.DF — User style kwargs take priority over channel auto-cycle

**Phase:** 13.36.DF | **Date:** 2026-05-20 | **Original notation:** §3.1 Priority rule

**Decision:** User-specified per-call style kwargs take priority over channel auto-assigned cycle values. Applied uniformly to ALL groups in the call. `None` = user did not pass. Non-`None` = user explicitly passed; apply uniformly.

**Motivation — architect production call (2026-05-18, verbatim):**
```python
adf.draw("dyp_I0:row", group_by="sec", color="red")
# Expected: all groups rendered red
# Was: only first group red; rest reverted to palette cycling
```

**Implementation pattern:** `_user_color`, `_user_marker`, `_user_markersize` as named params (default `None`); `is None` check before palette/cycle assignment. Named-param forwarding, not kwargs pop.

**Ratification:** §3.1 architect decision 2026-05-20; Sonnet53_R2 Option A architect-greenlit. `PHASE_13_36_DF_v1_2_Proposal_UserStyleOverride.md` §3.1, §5.

**Source:** `plots/profile.py:_draw_profile_grouped`, `plots/histogram.py:_draw_hist_grouped`

---

## AD-1/13.37.DF — BUG-016 retained in PHASE_NEXT_STEPS

**Phase:** 13.37.DF | **Date:** 2026-05-21 | **Original notation:** CP1-11 + architect sign-off

**Decision:** BUG-016 (`_interval_sort_key` lexicographic sort on interval labels) is kept in PHASE_NEXT_STEPS rather than fixed in Phase 13.37.

**Verbatim architect quote (2026-05-21):** *"Yes. I added it and I want to have it there."*

**Ratification:** Explicit architect retention sign-off. `PHASE_13_37_DF_v1_1_Proposal.md` §5, CP1-11.

**Source:** `plots/profile.py`

---

## AD-2/13.37.DF — Poisson error bars on `hist()` (`hist_errors=True`)

**Phase:** 13.37.DF | **Date:** 2026-05-21 | **Original notation:** Item 4

**Decision:** `hist_errors=True` on `hist()` adds Poisson ±√N error bars. Available on ungrouped, grouped (non-stacked), and faceted paths. Composition with `group_by` and `facet_by` in scope.

**Motivation:** Physics histograms require statistical error visualization. Poisson errors (±√N) are the standard for count distributions.

**Ratification:** `PHASE_13_37_DF_v1_1_Proposal.md` §6.

**Source:** `plots/histogram.py`

---

## AD-3/13.37.DF — Per-group linestyle cycling (`linestyle_cycle=True`)

**Phase:** 13.37.DF | **Date:** 2026-05-21 | **Original notation:** Item 5

**Decision:** `linestyle_cycle=True` on `hist()` cycles linestyles alongside color within a group_by call. Respects AD-1/13.36.DF explicit-kwarg precedence — user-passed `linestyle=` overrides the cycle.

**Ratification:** `PHASE_13_37_DF_v1_1_Proposal.md` §7.

**Source:** `plots/histogram.py`

---

## AD-1/13.38.DF — `_process_color()` dispatch: column-name check before `to_rgba()`

**Phase:** 13.38.DF | **Date:** 2026-05-22 | **Original notation:** CP0-1 (3-reviewer convergence)

**Decision:** In `_process_color()`, the DataFrame column-name check MUST precede `to_rgba()` conversion. `to_rgba('b')` succeeds for matplotlib blue regardless of `df.columns` membership — without this ordering, columns named after matplotlib colors ('b', 'g', 'r', etc.) are silently misinterpreted as fixed colors.

**Rejected alternative:** `to_rgba()` first — rejected because `to_rgba('b')` returns RGBA for matplotlib blue, meaning a column named 'b' would never reach the column-name path.

**Ratification:** 3-reviewer convergence (Opus2, Sonnet53_R2, Sonet50). Backward-compat lock test §9.ECM.6. `PHASE_13_38_DF_v1_1_ScatterEnhancements_Proposal.md` §4.

**Source:** `plots/scatter.py:_process_color`

---

## AD-2/13.38.DF — Scatter error bars (`xerr=`, `yerr=`)

**Phase:** 13.38.DF | **Date:** 2026-05-22 | **Original notation:** Item 2

**Decision:** `scatter()` gains `xerr=` and `yerr=` column-expression parameters. NaN/inf in error columns subject to `nan_policy`. `yerr=` alone is the opt-in (simplified later in AD-1/13.42.DF-FIX1 — `use_errors=True` no longer required). [Annotation added 2026-06-04: original Phase 13.38 spec required `use_errors=True` alongside `yerr=`; architect R3 simplified this.]

**Ratification:** `PHASE_13_38_DF_v1_1_ScatterEnhancements_Proposal.md` §3.

**Source:** `plots/scatter.py`

---

## AD-3/13.38.DF — Expression-based `color=` and `marker=` for scatter

**Phase:** 13.38.DF | **Date:** 2026-05-22 | **Original notation:** Item 3

**Decision:** `scatter()` accepts column-name strings for `color=` and `marker=` (per-point encoding). Dispatch follows AD-1/13.38.DF order: column-name check first, then `to_rgba()`.

**Ratification:** `PHASE_13_38_DF_v1_1_ScatterEnhancements_Proposal.md` §4.

**Source:** `plots/scatter.py`

---

## AD-1/13.39.DF — 2D Profile (`z:y:x` → `pcolormesh`)

**Phase:** 13.39.DF | **Date:** 2026-05-23 | **Original notation:** Item 1

**Decision:** `draw_profile2d()` dispatched when `colon_count==2` in the expression. `scipy.stats.binned_statistic_2d` backend. `range=` as single-level outer list `[x_range, y_range]`. `group_by` raises clean `ValueError` (explicit scope boundary).

**Ratification:** `PHASE_13_39_DF_v1_2_Profile2D_TimeAxis_Scatter3D_Proposal.md` §2.

**Source:** `drawer.py`, `plots/profile.py`

---

## AD-2/13.39.DF — Time Axis dtype auto-detection

**Phase:** 13.39.DF | **Date:** 2026-05-23 | **Original notation:** Item 2, CP1-4/N3

**Decision:** `time_format=` auto-detects dtype: if `np.issubdtype(x_data.dtype, np.datetime64)` → `mdates.date2num(x_data)`; else unit='s' numeric conversion. The two paths (datetime64 and epoch-seconds) are distinct and must not be conflated.

**Ratification:** CP1-4/N3 resolution. `PHASE_13_39_DF_v1_2_Profile2D_TimeAxis_Scatter3D_Proposal.md` §3.

**Source:** `plots/profile.py`, `drawer.py`

---

## AD-3/13.39.DF — Scatter3D (`type='scatter3d'`)

**Phase:** 13.39.DF | **Date:** 2026-05-23 | **Original notation:** Item 3

**Decision:** `type='scatter3d'` dispatches `draw_scatter3d()`. `mpl_toolkits.mplot3d` backend. Stats dict locks `mean_x`, `mean_y`, `mean_z` all to 1e-9. scipy is a required dependency — no fallback.

**Ratification:** `PHASE_13_39_DF_v1_2_Profile2D_TimeAxis_Scatter3D_Proposal.md` §4.

**Source:** `drawer.py`

---

## AD-1/13.40.DF — Cumulative histogram via matplotlib native `cumulative`

**Phase:** 13.40.DF | **Date:** 2026-05-24 | **Original notation:** Design choice §1

**Decision:** Binned CDF implemented via matplotlib's native `ax.hist(cumulative=±1)`. Not a post-processing step. Matches ROOT `TH1::Draw("cumulative")`. Composes with `hist_norm`/`group_by`/`facet_by`/`stacked`.

**Ratification:** `PHASE_13_40_DF_v1_2_CumulativeHist_Proposal.md` §1.

**Source:** `plots/histogram.py`

---

## AD-2/13.40.DF — `hist_errors=True` + `cumulative` raises `NotImplementedError`

**Phase:** 13.40.DF | **Date:** 2026-05-24 | **Original notation:** M5 correctness guard

**Decision:** `hist_errors=True` combined with `cumulative=True` or `cumulative=-1` raises `NotImplementedError`. Binned CDF with Poisson error bars is statistically ill-defined. Explicit guard with diagnostic message.

**Ratification:** `PHASE_13_40_DF_v1_2_CumulativeHist_Proposal.md` §4, item 6.

**Source:** `plots/histogram.py`

---

## AD-1/13.41.DF — ROW/COL/FIGID convention LOCKED

**Phase:** 13.41.DF | **Date:** 2026-05-25 | **Original notation:** §3 interface, locked v1.2

**Decision:** `facet_by=['ROW', 'COL']` syntax with `FIGID` as figure-level grouping channel. ROW = subplot rows within a figure, COL = subplot columns, FIGID = separate figures. Convention locked in v1.2 and carried unchanged through v1.6.

**Ratification:** `PHASE_13_41_DF_v1_6_RowColumnFigIDFaceting_Proposal.md` §1–§4.

**Source:** `drawer.py`, `facet.py`

---

## AD-2/13.41.DF — N-D faceted return contract

**Phase:** 13.41.DF | **Date:** 2026-05-25 | **Original notation:** §3 return type, CP2-3

**Decision:** N-D faceted calls with `FIGID` return `(List[Figure], List[axes_2d], List[stats_dict])`. Single-figure faceted calls return `(fig, axes_array, stats)`. The deviation from the standard 3-tuple is explicit and prominently documented.

**Ratification:** `PHASE_13_41_DF_v1_6_RowColumnFigIDFaceting_Proposal.md` §3.

**Source:** `drawer.py`

---

## AD-3/13.41.DF — `share_x`/`share_y`/`share_across_figures` defaults

**Phase:** 13.41.DF | **Date:** 2026-05-25 | **Original notation:** §4, CP0-1

**Decision:** `share_x='row'` (default), `share_y='col'` (default), `share_across_figures=True` (default). Dispatch dict `{'all': True, 'row': 'row', 'col': 'col', 'none': False}` — symmetric for both axes.

**Ratification:** `PHASE_13_41_DF_v1_6_RowColumnFigIDFaceting_Proposal.md` §4.

**Source:** `drawer.py`, `facet.py`

---

## AD-1/13.42.DF — Single `fit=` kwarg; no convenience top-level kwargs

**Phase:** 13.42.DF | **Date:** 2026-05-22 | **Original notation:** Q-B (architect)

**Decision:** All per-fit configuration travels in the dict spec. Single kwarg surface: `fit=`. No `fit_range=`, `fit_initial=`, or other convenience top-level kwargs.

**Motivation:** v1.0 was rejected by architect on this point. The unified `fit=` surface is cleaner and composable with the vector convention.

**Ratification:** Architect Q-B. `PHASE_13_42_DF_v1_4_InlineFits_Proposal.md` §2 v1.0→v1.1.

**Source:** `drawer.py`

---

## AD-2/13.42.DF — `fit=` follows standard Phase 13.16 vector convention

**Phase:** 13.42.DF | **Date:** 2026-05-22 | **Original notation:** Q-A (architect)

**Decision:** scalar → broadcast to all curves; list → per-curve inner pairing via N:1/N:N broadcasting. Per-channel different fits IS in v1.0 scope.

**Verbatim architect quote:** *"exactly the same as for vectors ... This is our standard, we should not change it."*

**Ratification:** Architect Q-A. `PHASE_13_42_DF_v1_4_InlineFits_Proposal.md` §2, §3.4.

**Source:** `drawer.py`, `plots/fits.py`

---

## AD-3/13.42.DF — scipy.optimize.curve_fit only; lmfit dropped

**Phase:** 13.42.DF | **Date:** 2026-05-22 | **Original notation:** Q-3 (architect)

**Decision:** scipy.optimize.curve_fit is the only fit engine. lmfit dropped.

**Verbatim architect quote:** *"adds soft dependency for marginal benefit."*

**Rejected alternative:** lmfit — rejected by architect.

**Ratification:** Architect Q-3. `PHASE_13_42_DF_v1_4_InlineFits_Proposal.md` §2, engine table.

**Source:** `plots/fits.py`

---

## AD-4/13.42.DF — `stats['fit']` canonical shape LOCKED

**Phase:** 13.42.DF | **Date:** 2026-05-22 | **Original notation:** CP1-3, v1.3 lock

**Decision:** `stats['fit']` canonical shape: outer list = curves (render order); inner list = fits per curve. With group_by/facet_by: dict-keyed.

**Ratification:** v1.3 panel lock (CP1-3). `PHASE_13_42_DF_v1_4_InlineFits_Proposal.md` §3.5.

**Source:** `plots/fits.py`, `drawer.py`

---

## AD-5/13.42.DF — Predefined fit registry minimum set

**Phase:** 13.42.DF | **Date:** 2026-05-22 | **Original notation:** OQ1, locked v1.3

**Decision:** Predefined registry minimum set: gauss/gaussian, linear, pol0–pol5, expo/exponential, lorentz/lorentzian, powerlaw. Deferred: landau, crystalball, breitwigner.

**Verbatim architect quote:** *"can be done later."*

**Ratification:** OQ1 locked at v1.3. `PHASE_13_42_DF_v1_4_InlineFits_Proposal.md` §12.

**Source:** `plots/fits.py:_initialize_registry()`

---

## AD-6/13.42.DF — Initial-parameter resolution order (4-tier)

**Phase:** 13.42.DF | **Date:** 2026-05-22 | **Original notation:** OQ2, v1.2 lock

**Decision:** Initial-parameter resolution order: (1) user `initial` → (2) user `guess` callable → (3) registry heuristic → (4) scipy default + warning.

**Ratification:** OQ2 locked at v1.2. `PHASE_13_42_DF_v1_4_InlineFits_Proposal.md` §3.3.

**Source:** `plots/fits.py`

---

## AD-1/13.42.DF-FIX1 — `yerr=` alone is opt-in for scatter fits

**Phase:** 13.42.DF-FIX1 | **Date:** 2026-05-26 | **Original notation:** R3 (architect MODIFY)

**Decision:** `yerr="col"` alone is the opt-in for scatter fit error weighting. `use_errors=True` NOT required when `yerr=` is provided.

**Verbatim architect quote:** *"optional, used if provided."*

**Ratification:** Architect R3 modification. `PHASE_13_42_DF_FIX1_v1_2_Proposal.md` §3.3.

**Source:** `plots/scatter.py`

---

## AD-2/13.42.DF-FIX1 — stacked + group_by + fit → N per-group fits

**Phase:** 13.42.DF-FIX1 | **Date:** 2026-05-26 | **Original notation:** R4 (architect MODIFY)

**Decision:** stacked + group_by + fit produces N fits (one per group_by group). Stacking is purely visual and does NOT alter fit targets.

**Verbatim architect quote:** *"fit all figures, all gb."*

**Ratification:** Architect R4 modification. `PHASE_13_42_DF_FIX1_v1_2_Proposal.md` §3.4.

**Source:** `plots/histogram.py`, `plots/fits.py`

---

## AD-3/13.42.DF-FIX1 — `fit_textbox_kwargs` LOCKED including per-call fontsize

**Phase:** 13.42.DF-FIX1 | **Date:** 2026-05-26 | **Original notation:** R5 (architect OK with emphasis)

**Decision:** `fit_textbox_kwargs` interface locked. Per-call `fontsize` override must work. Regression test F.33 added to guard it.

**Verbatim architect quote:** *"but we need also font size working."*

**Ratification:** Architect R5 emphasis. `PHASE_13_42_DF_FIX1_v1_2_Proposal.md` §3.5.

**Source:** `plots/_fit_render.py`

---

## AD-1/13.43.DF — summary_fit main goal: quick prototyping

**Phase:** 13.43.DF | **Date:** 2026-05-27 | **Original notation:** Architect framing v0→v0.1

**Decision:** MAIN GOAL is quick prototyping — sensible defaults over many knobs. Sophistication is the user's responsibility, not dfdraw's. Two pad kinds, non-exclusive: table and parameter-trend plot. Auto-overflow to new figure when capacity exceeded. Heatmap explicitly rejected.

**Verbatim architect quotes (2026-05-27):**
- *"quick prototyping — sensible defaults > many knobs. Sophistication is the user's responsibility, not dfdraw's."*
- *"I hate heatmap — I do not see values."*

**Ratification:** Architect v0→v0.1 framing + Q-B1. `PHASE_13_43_DF_v1_1_SummaryFit_Proposal.md` §0.

**Source:** `plots/_summary_fit.py`

---

## AD-2/13.43.DF — `draw()` return contract unchanged; summary inside `stats`

**Phase:** 13.43.DF | **Date:** 2026-05-27 | **Original notation:** §3.2 return contract

**Decision:** `draw()` returns exactly `(fig, ax, stats)` 3-tuple — unchanged. Phase 13.43 attaches summary figures only inside `stats['summary_fit']`. No change to the public return contract.

**Ratification:** Architect 2026-05-27. `PHASE_13_43_DF_v1_1_SummaryFit_Proposal.md` §3.2.

**Source:** `drawer.py`, `plots/_summary_fit.py`

---

## AD-3/13.43.DF — summary_fit data format controlled via style key

**Phase:** 13.43.DF | **Date:** 2026-05-27 | **Original notation:** C-10, architect override

**Decision:** `stats['summary_fit']['data']` ships alongside figures. Default format = dict; opt-in pandas via `style['summary_fit.data_format'] = 'pandas'`. Public `fit_results_to_df()` rejected in favour of style-controlled format.

**Ratification:** Architect override of panel recommendation (Sonet50). `PHASE_13_43_DF_v1_1_SummaryFit_Proposal.md` C-10.

**Source:** `plots/_summary_fit.py`

---

## AD-4/13.43.DF — `summary_fit` is outer-layer consumed; not in FORWARDED_NAMES

**Phase:** 13.43.DF | **Date:** 2026-05-27 | **Original notation:** P1-2

**Decision:** `summary_fit` is consumed at top-level dispatcher before `_dispatch_faceted_render`. It MUST NOT be in `_*_FORWARDED_NAMES`. `fit_textbox_kwargs` IS forwarded (Pattern B). This distinction — Pattern A (outer-consumed) vs Pattern B (forwarded) — is locked.

**Ratification:** P1-2 (Claude48 + Sonnet53_R2). `PHASE_13_43_DF_v1_1_SummaryFit_Proposal.md` P1-2.

**Source:** `drawer.py`

---

## AD-5/13.43.DF — Faceted stats aggregation: Option A

**Phase:** 13.43.DF | **Date:** 2026-05-27 | **Original notation:** P1-3, architect V2

**Decision:** `_dispatch_faceted_render` aggregates per-cell `stats[(row,col)]['fit']` into a top-level `stats['fit']` flat list before returning. Both top-level and per-cell exist (additive). Phase 13.42 D4 per-cell contract preserved.

**Verbatim architect quote (V2):** *"cleaner option, do not postpone."*

**Rejected alternative:** Option B (caller aggregates) — rejected per architect V2.

**Ratification:** Architect V2. `PHASE_13_43_DF_v1_1_SummaryFit_Proposal.md` P1-3, §4.2.0.

**Source:** `drawer.py:_dispatch_faceted_render`

---

## AD-6/13.43.DF — Kwarg renamed `summary_pad` → `summary_fit`

**Phase:** 13.43.DF | **Date:** 2026-05-27 | **Original notation:** OQ-A1 (architect)

**Decision:** Kwarg renamed `summary_pad` → `summary_fit` throughout. `stats['summary_fit']` is the extension key.

**Ratification:** Architect OQ-A1. `PHASE_13_43_DF_v1_1_SummaryFit_Proposal.md` §0.

**Source:** `drawer.py:summary_fit=`, `plots/_summary_fit.py`

---

## AD-7/13.43.DF — `same=True` replace mode for summary_fit

**Phase:** 13.43.DF | **Date:** 2026-05-27 | **Original notation:** OQ-A2 (architect "OK")

**Decision:** `same=True` → replace mode: each `draw(same=True)` replaces previous `stats['summary_fit']`. Accumulate mode deferred.

**Ratification:** Architect OQ-A2 approval. `PHASE_13_43_DF_v1_1_SummaryFit_Proposal.md` §3.11 + §0.

**Source:** `drawer.py`

---

## AD-1/13.46.DF — Faceted scatter `range=`: shared-global view

**Phase:** 13.46.DF | **Date:** 2026-05-28 | **Original notation:** §2.1 Option-1

**Decision:** For faceted scatter: `range=` applies the shared-global view — a single range computed across all facet cells, each cell filters its own points to that global range, shared axes autoscale to the union of filtered data. Prevents last-cell-wins artifact.

**Ratification:** §2.1 Option-1. `PHASE_13_46_DF_v1_0_AuditFixes_Proposal.md` §2 footnote.

**Source:** `plots/scatter.py`, `drawer.py`

---

## AD-1/13.46.DF-FIX1 — Scatter `range=` removes points (point filter)

**Phase:** 13.46.DF-FIX1 | **Date:** 2026-05-28 | **Original notation:** Architect ruling

**Decision:** Scatter `range=` removes/filters out-of-range points (a point filter), consistent with how `hist`/`profile` `range=` excludes points from binning. It does NOT merely clip the view. `stats_dict` is computed AFTER the filter — honest counts.

**Motivation:** Semantic consistency across all plot types. Clipping the view would leave misleading stats (n counts points outside the visible range).

**Ratification:** Architect ruling 2026-05-28. `PHASE_13_46_DF_FIX1_Code_Review_Request.md` §1–§3.

**Source:** `plots/scatter.py`

---

## AD-1/13.48.DF — Tier 1 visual testing: primitive-only, renderer-free

**Phase:** 13.48.DF | **Date:** 2026-05-30 | **Original notation:** ⚑ AUTHOR DECISION N-2 (architect delegated)

**Decision:** Tier 1 only: primitive-only, renderer-free, deterministic, backend-independent. New `visual_primitive` layer in `feature_taxonomy.py` distinct from `invariance`/`smoke`. Tier 2 (text-extent, overlap, clipping — needs real renderer) is out of scope, gated behind a future phase.

**Ratification:** ⚑ AUTHOR DECISION N-2 — architect delegated. `PHASE_13_48_DF_v1_4_VisualTesting_Proposal.md` §0.

**Source:** `tests/feature_taxonomy.py`, `tests/test_layer_classification.py`, `tests/test_phase_13_48_df_visual_testing.py`

---

## AD-1/13.49.DF — Explicit `tests:[list]` per feature; no grandfathering

**Phase:** 13.49.DF | **Date:** 2026-05-28 | **Original notation:** §3, Option A, panel 9×[!]

**Decision:** Each feature in `feature_taxonomy.py` carries `"tests": [explicit node-ID list]` — a direct link, not pattern matching. No grandfathering — M.1 must pass on commit.

**Ratification:** Option A, Sonet50 recommendation, 9×[!] panel. `PHASE_13_49_DF_v1_2_CapabilityMatrixTraceability_Proposal.md` §3.

**Source:** `tests/feature_taxonomy.py`, `tests/test_meta_capability_matrix.py`

---

## AD-2/13.49.DF — `KNOWN_UNCLAIMED` governance

**Phase:** 13.49.DF | **Date:** 2026-05-28 | **Original notation:** §3.7, v1.2

**Decision:** Bounded, versioned allow-list for tests classified in `TEST_LAYERS` but not yet claimed by a feature. Required fields: `reason` ≥ 10 chars, valid `target_phase`. Growth discipline: new entries require reviewer justification at commit. Visual-primitive tests NEVER enter `KNOWN_UNCLAIMED` — must always be claimed by a feature.

**Ratification:** v1.2 panel. `PHASE_13_49_DF_v1_2_CapabilityMatrixTraceability_Proposal.md` §3.7.

**Source:** `tests/test_meta_capability_matrix.py`

---

## AD-3/13.49.DF — Visual tests claimable by any feature category

**Phase:** 13.49.DF | **Date:** 2026-05-28 | **Original notation:** M.4 relaxation, v1.2

**Decision:** Visual tests can be claimed by ANY feature category (`FIT.*`, `PROF.*`, `HIST.*`, `VISUAL.*`) — not restricted to `VISUAL.*` only. Visual evidence is orthogonal; any feature can add a V-check.

**Ratification:** v1.2 M.4 relaxation. `PHASE_13_49_DF_v1_2_CapabilityMatrixTraceability_Proposal.md` §3.7, M.4.

**Source:** `tests/test_meta_capability_matrix.py`

---

## AD-1/13.50.DF — `fit.text_format` removed; split into value/error format [BREACH]

**Phase:** 13.50.DF | **Date:** 2026-05-30 | **Original notation:** [BREACH], §3.3

**Decision:** Single `fit.text_format='.4g'` style key removed. Replaced by `fit.value_format` (default `.2g`) and `fit.error_format` (default `.1g`). Physics convention: errors → 1 sig fig; values → matched to error's decimal place.

**Ratification:** [BREACH] ratified. `PHASE_13_50_DF_v2_5_FitRenderingOverhaul_Proposal.md` §3.3; `PHASE_13_50_DF_CRR_v1_0.md` §2.1.

**Source:** `style.py`, `plots/_fit_render.py`

---

## AD-2/13.50.DF — `summary_fit.precision` removed; `precision_mode='physics'` [BREACH]

**Phase:** 13.50.DF | **Date:** 2026-05-30 | **Original notation:** [BREACH], §3.2

**Decision:** `summary_fit.precision` style key removed. Replaced by `fit.value_format`/`fit.error_format` + `precision_mode='physics'`. `precision_mode='physics'`: error rounded to 1 sig fig, value matched to error's decimal place.

**Ratification:** [BREACH] ratified. `PHASE_13_50_DF_v2_5_FitRenderingOverhaul_Proposal.md` §3.2; `PHASE_13_50_DF_CRR_v1_0.md` §2.2.

**Source:** `style.py`, `plots/_summary_fit.py`

---

## AD-3/13.50.DF — `_DISPLAY_NAMES` render-only map in `_fit_render.py`

**Phase:** 13.50.DF | **Date:** 2026-05-30 | **Original notation:** §3.1

**Decision:** `_DISPLAY_NAMES` dict maps canonical parameter names (from `fits.py` signatures) to display names for textbox rendering. Lives in `_fit_render.py`, NOT in `fits.py`. No changes to `fits.py` signatures. Example: `'slope'→'p1'`, `'intercept'→'p0'` (ascending-powers convention: c0 = constant = intercept, c1 = x-coeff = slope).

**Ratification:** `PHASE_13_50_DF_v2_5_FitRenderingOverhaul_Proposal.md` §3.1.

**Source:** `plots/_fit_render.py`

---

## AD-4/13.50.DF — `legend=` polymorphic + `show_legend=` permanent back-compat

**Phase:** 13.50.DF | **Date:** 2026-05-30 | **Original notation:** §3.4 + §3.1 back-compat rule

**Decision:** `legend=` accepts `True`/`False`/`'shared'`/`'first'`/dict. `show_legend=` retained as permanent back-compat parallel — NOT deprecated, NOT removed. If both passed, `legend=` wins. No warning emitted. Back-compat guarantee is permanent (not time-limited).

**Ratification:** `PHASE_13_50_DF_v2_5_FitRenderingOverhaul_Proposal.md` §3.4.

**Source:** `drawer.py`, `plots/_legend.py`

---

## AD-5/13.50.DF — `summary_fit.placement` — 3 modes

**Phase:** 13.50.DF | **Date:** 2026-05-30 | **Original notation:** §3.5

**Decision:** Three placement modes:
- `'figure'` — separate Figure object returned in `stats['summary_fit']`
- `'subfigure'` — per-panel inset axes showing **only that panel's fits** (per-panel slices, NOT full-table copies per panel)
- `'pad'` — GridSpec pad column pre-planned before `plt.subplots` (immutable post-creation); requires pre-planning at top of `_dispatch_faceted_render`

`ax` returned to caller is always the main-plot axes array. Pad axes not in `ax`; reachable via `stats['summary_fit']` or by walking `fig.axes`.

**Ratification:** `PHASE_13_50_DF_v2_5_FitRenderingOverhaul_Proposal.md` §3.5.

**Source:** `drawer.py`, `plots/_summary_fit.py`

---

## AD-6/13.50.DF — F19 (textbox-bbox-overlap) architecturally deferred to Phase 13.5X

**Phase:** 13.50.DF | **Date:** 2026-05-30 | **Original notation:** §0 Note + §3.8

**Decision:** F19 (Tier 2 textbox-bbox-overlap test that proves the spacing bug is fixed) is architecturally deferred to Phase 13.5X. Phase 13.50 green is necessary but not sufficient evidence for the spacing bug being cured. Tier 2 requires `fig.canvas.get_renderer()` draw cycles that the Phase 13.48 framework does not yet support.

**Motivation:** The motivating bug — architect-flagged *"problems only with spacing of the fits in the latest test"* (2026-05-30) — is empirically addressed by this phase's API surface. The regression lock lands in 13.5X.

**Ratification:** `PHASE_13_50_DF_v2_5_FitRenderingOverhaul_Proposal.md` §0 Note + §3.8.

**Source:** `tests/` (future Phase 13.5X)

---

*dfdraw ARCHITECT_DECISIONS.md v1.0.1 — 2026-06-04*
*v1.0 committed same day; v1.0.1 applies Org panel convergent touch-ups (header, AD-37/50 supersession, GP transitional authority, GP Date fields, About additions)*
*Legacy AD-1..AD-82: Sonnet56 (verbatim text) + Sonnet57 (source refs at HEAD 07606c02)*
*AD-N/PHASE entries: Sonnet57 final candidate table, Sonnet56 additive corrections, naming convention per 2026-06-04*
