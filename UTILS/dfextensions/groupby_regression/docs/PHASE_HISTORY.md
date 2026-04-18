# Phase History — GroupBy Regression

**Last Updated:** 2026-04-09
**Current Phase:** 13.17.GB (⏳ **in progress** — coder Claude22 active; this document records plan, not outcomes, for 13.17.GB)
**Document Version:** 6.1
**Drafted by:** Claude23 (GBAI Reviewer) at architect request 2026-04-09

> **v6.1 scope.** This revision absorbs **two** units of work. (1) **Phase 13.16.GB-FIX2 retroactive entry** — FIX2 was committed on 2026-04-09 (`9e88eacd`, tag `PHASE_13_16_GB_FIX2_END`) but the FIX2 commit message claimed `PHASE_HISTORY.md: v6.0 → v6.1` without the file actually being updated on disk. This v6.1 closes that gap. (2) **Phase 13.17.GB in-progress record** — PHASE_13_17_GB_v1.3_Proposal is APPROVED, Coder Claude22 is implementing, commit expected in ~3.5 working days. v6.1 records the phase plan so that the in-flight work has a landing zone in the paper trail; a minor v6.1a revision at commit time will fill in the final test counts, commit hash, and outcome narrative. All v6.0 sections are **preserved verbatim** except where a specific correction is called out in the Document History row.

## Overview

The GroupBy Regression module provides high-performance grouped linear regression for CERN/ALICE particle physics calibration workflows. Development follows a phased approach with multi-LLM architecture review before each implementation.

**Scale:** 25M fits in production, 100K+ groups, strict memory constraints per core.

---

## Phase Summary

| Phase | Name | Date | Status |
|-------|------|------|--------|
| — | Package Structure | Oct 25, 2025 | ✅ Complete |
| — | v2/v3/v4 Engines | Oct 25, 2025 | ✅ Complete |
| 12.8.GB | Batch Fitting (v5) | Dec 19, 2025 | ✅ Complete |
| 12.9.GB | Numba Parallel Kernel | Dec 20, 2025 | ✅ Complete |
| 12.10.BF | Benchmark Framework | Dec 23-25, 2025 | ✅ Complete |
| 12.11 | n_jobs=1 Fix + Profiling | Dec 26, 2025 | ✅ Complete |
| 13.1.GB | PyArrow Backend | Dec 16, 2025 | ✅ Complete |
| 12.14.GB | Shared Numba Kernel | Dec 31, 2025 | ✅ Complete |
| 12.14a.GB | Test Infrastructure | Dec 31, 2025 | ✅ Complete |
| 12.14b.GB | BF Integration | Jan 1, 2026 | ✅ Complete |
| 12.14b.GB-add | Dual Timing + cProfile | Jan 1, 2026 | ✅ Complete |
| 12.14c.GB | ADF Visualization | Jan 3-4, 2026 | ✅ Complete |
| 13.7.GB | Capability Matrix Infrastructure | Feb 7-8, 2026 | ✅ Complete |
| 13.8.GB | SW API Refactor + Invariance Tests | Feb 9-10, 2026 | ✅ Complete |
| 13.9.GB | V3 Incremental Algorithm | Feb 10, 2026 | ✅ Complete |
| 13.8.SW | Parallel Sliding Window + Benchmarks | Feb 12-14, 2026 | ✅ Complete |
| 13.10.GB | Non-Linear SW Fit + Evaluator Extension | Feb 16-17, 2026 | ✅ Complete |
| 13.10.GB-T | K0s THn Tutorial + MC Truth Validation | Feb 18, 2026 | ✅ Complete |
| 13.9.GB-Ext | agg_columns + WLS Fix + fit_intercept + Lean Output + Parallel Parity | Feb 23 – Mar 13, 2026 | ✅ Complete |
| 13.13.GB | Expression-Based Linear Columns | Mar 23, 2026 | ✅ Complete |
| 13.14.GB | Dedicated Sliding Window Aggregation | Mar 24, 2026 | ✅ Complete |
| 13.15.GB | N-Sigma Cut in SW Aggregation | Mar 26, 2026 | ✅ Complete |
| 13.16.GB | Evaluator Lookup + scipy Methods | Mar 28-29, 2026 | ✅ Complete |
| P0-Fix | fit_intercept hardcoded in SW numba (3 locations) | Mar 28-29, 2026 | ✅ Fixed |
| **13.16.GB-FIX2** | **Evaluator bug fixes F2/F3/F4/F5 + C3/C8/C10** | **Apr 9, 2026** | **✅ Complete (commit `9e88eacd`, tag `PHASE_13_16_GB_FIX2_END`)** |
| **13.17.GB** | **`boundary='symmetric'` in `make_sliding_window_aggregate` — F1 fix + 16 invariance tests** | **Apr 9, 2026 →** | **⏳ In progress** (Coder Claude22, Main Reviewer Claude20, PHASE_13_17_GB_v1.3_Proposal APPROVED) |
| **13.17.GB-MedianFix** | **`agg_median=True` + `boundary` interaction** (D1 deferral from 13.17.GB per architect direction 2026-04-09) | **TBD** | **📋 Scheduled** (immediate follow-up to 13.17.GB; ~6-10h per Claude23 estimate) |
| 13.11.GB | Chi2 Audit + Histogram Accumulation | — | 📋 Approved, deferred |
| 13.12.GB | SW Histogram Accumulation Mode | — | 📋 Proposed, deferred |
| 12.15.GB | V4 Integration | — | 📋 Planned |

**Current test count (canonical `alma2`):** **529 passed** / 2 failed (pre-existing, unrelated to any in-flight phase) / 19 skipped / 133 features. Scheduled target at 13.17.GB close: **545 passed** (529 + 16 new invariance tests).

> **Known documentation defect carried from FIX2 commit:** `TECHNICAL_SUMMARY.md` v3.3 records the canonical passed-test count as **528** at four locations (header summary, Current State table, arithmetic narrative, v3.3 document history row). The correct canonical number is **529** — the FIX2 commit flips `test_select_backend_auto_sequential` from failed to passed (F5 cleanup), which adds `+1` to the passed column that the Coder's arithmetic `517 + 11 = 528` omitted. Three reviewers flagged this during the FIX2 review cycle; the correction was not absorbed before commit. Closure is scheduled as part of TECHNICAL_SUMMARY v3.4 in Phase 13.17.GB per proposal §7.3. **This v6.1 uses 529 as the authoritative baseline.**

**Capability Matrix:** Phase 13.16.GB-FIX2 — 1 broken (F1 `boundary='symmetric'` in aggregate path, in-flight fix), 1 planned. Post-13.17.GB target: 0 broken for the mean/std/count path, with the median subpath retained as a scheduled Known Limitation until 13.17.GB-MedianFix.

---

## Critical Incident 1: Silent Numba Regression (Nov 2025)

### What Happened

On **November 14, 2025** (commit `db0eb019`), the V4 Numba JIT kernel was accidentally removed during metadata refactoring. This caused a **10× performance regression** that went undetected for **6 weeks**.

### Resolution

Phases 12.14.GB through 12.14c.GB address this with shared kernel module, Numba/NumPy parity tests, performance gates, memory stability tests, and benchmark framework.

---

## Critical Incident 2: COG/agg_columns Deferral (Feb 2026)

### What Happened

The original specification (Oct 2025) included aggregation of groupby variables (center-of-gravity) within sliding windows. This was repeatedly deferred across Phases 13.7–13.10 as "V4/V5 extension" without architect approval. Discovered during real-data TPC calibration testing (Feb 19, 2026).

### Root Cause

Same class as Phase 0.1B: scope reduction without architect sign-off (governance failure mode #8). No single reviewer flagged the gap because synthetic tutorial data masked it.

### Resolution

- `agg_columns` parameter added in Phase 13.9.GB Extension
- MTTU_Reviewer v1.11+ Authority Boundary rules cover this case
- Specification compliance checklist added to phase boundary reviews

---

## Critical Incident 3: `fit_intercept` Parameter Not Propagated to SW Numba (Mar 28-29, 2026)

### What Happened

`make_sliding_window_fit` with `fit_intercept=False` silently failed on 100% of bins when using the numba backend. All bins returned `quality_flag='fit_failed'`. No error was raised. The same data worked perfectly with `make_parallel_fit_v4`. **Found by O2DistAI team in production**, not by the test suite.

### Root Cause

Three hardcoded assumptions in `_fit_window_regression_numba`, all ignoring `fit_intercept=False`:

| Location | Bug | Fix |
|----------|-----|-----|
| Line 994 | `fit_intercept=True` hardcoded in kernel call | Pass `fit_intercept` parameter through |
| Line 913 | `n_params = n_pred + 1` hardcoded | `n_params = n_pred + (1 if fit_intercept else 0)` |
| Line 1011 | Result unpacking `out_beta[i, j+1]` assumed intercept at index 0 | `offset = 1 if fit_intercept else 0` |

**Chain of compensating bugs:** With `n_params` wrong, `out_beta` was oversized, so `j+1` indexing didn't crash — it silently read garbage. When `n_params` was fixed, the unpacking crash exposed the third bug. With `fit_intercept=False` and a polynomial basis containing a constant term, the kernel added a second constant column → X'X singular → Cholesky failed → every bin returned `fit_failed`.

### This Is the Third Instance of the Parameter-Not-Propagated Bug Class

| # | Bug | Path | Phase Found |
|---|-----|------|-------------|
| 1 | WLS `weights` silently ignored | V1/V2/V3 regression | 13.9.GB-Ext |
| 2 | `fit_intercept=False` produces intercept columns | `_assemble_results` | 13.9.GB-Ext |
| 3 | `fit_intercept` hardcoded `True` in SW numba (3 locations) | `_fit_window_regression_numba` | This fix |

Pattern: parameters accepted at the API surface but silently ignored in one or more internal code paths. Each path was written independently and not checked against the full parameter contract.

> **Catalog update (v6.1).** Two more instances of this class have been identified since this incident: #4 F2 (`method=dict` silently drops interpolation orders, fixed in FIX2 — see Incident 6); #5 F1 (`boundary` silently dropped in `make_sliding_window_aggregate`, in-flight fix in 13.17.GB — see Incident 7). The bug class count is now **5 instances**. Note: an earlier v1.3 proposal draft stated "7 instances"; that draft conflated overall Incident numbers (1–7) with parameter-not-propagated class instance numbers (1–5). The authoritative count is **5 class instances**, confirmed against the FIX2 commit message line 81 (`"instance #5 of parameter-not-propagated bug class, fix in 13.17.GB"`).

### Why It Survived 4 Review Rounds

All invariance tests used `window=0` (which routes to V5, not V2 numba) or `backend='auto'` (which selected V5). The numba recompute path was never exercised with `fit_intercept=False` and `window > 0`. The tests passed but didn't test the code path where the bug lived.

### Resolution

- 3-part fix across 3 commits: `a9c6d07f`, `0ed8a56c`, `1278986b`
- 10 cross-fitter invariance tests in `test_fit_intercept_all_fitters.py`
- Each test explicitly specifies `backend='numpy'` or `backend='numba'`
- 2 full-chain tests use `window > 0` to exercise the recompute path:
  - `test_sw_window1_numba_matches_manual_windowed_v4`
  - `test_sw_window1_numba_matches_numpy`

### Governance Rule Added (MTTU_Reviewer v1.16, Failure Mode #11)

> Invariance tests must explicitly set path-controlling parameters — no auto/default dispatch. When a parameter is handled by multiple actively used paths, each path needs its own test for that parameter. Reviewers must verify that the test reaches the claimed code path, not just that it passes.

---

## Critical Incident 4: `boundary='symmetric'` — Specified, Approved, Partially Resolved (Mar–Apr 2026)

> **v6.1 status change.** v6.0 recorded this incident as "Status Disputed — Requires Verification." As of 2026-04-09 (FIX2 era + 13.17.GB proposal cycle), the dispute is **partially resolved**. The fit-path implementation exists and is correct; the aggregate-path implementation is confirmed broken and is the subject of in-flight Phase 13.17.GB (see **Incident 7** below). This incident entry is preserved for historical context; the aggregate-path bug has been split out as its own incident.

### What Happened

The original Phase 13.8.GB proposal specified `boundary='symmetric'` mode. The architect explicitly requested it with reasoning preserved verbatim:

> "Symmetric will truncate... For TPC — drift, radius, and rphi — all should be symmetric. **I do not want to introduce edge bias.**"
>
> — Main Architect (MI)

The proposal definition (architect-approved):

> `boundary='symmetric'`: Truncate window to maximum symmetric extent. If the center can only reach k bins left, also limit to k bins right.

The Phase 13.8.GB commit message (`bd6f042e`, Feb 10, 2026) claims:

> "V3b (boundary + bin weights): boundary='full'|'symmetric'|'periodic', per-dimension"

### Resolution Status (v6.1)

**Fit path (`make_sliding_window_fit`):** ✅ **Implemented correctly.** The fit path routes through `_get_neighbor_bins_v2` (line 377) which implements per-dimension `eff_w = min(w, c-lo, hi-c)` for `boundary='symmetric'` and wrap-around for `boundary='periodic'`. Four invariance tests in `test_invariance_sliding_window.py::TestSWV3bBoundary` cover this path and pass on the canonical baseline:
- `test_symmetric_reduces_corner_window`
- `test_symmetric_interior_equals_full`
- `test_symmetric_per_dimension`
- `test_symmetric_gaussian_interior_same_as_full_gaussian`

The original "status disputed" framing was partially correct: the code **did** exist but was never documented in the Technical Summary and was never surfaced in the Capability Matrix. TECHNICAL_SUMMARY v3.3 (committed with FIX2) added the fit-path documentation, closing the documentation half of the original dispute.

**Aggregate path (`make_sliding_window_aggregate` and parallel sibling):** 🧨 **Confirmed broken — silently drops the `boundary` parameter entirely.** The architect surfaced this during real-data TPC testing in the 13.16.GB era. Verified by smoke test on a linear-rise-in-`dsector` fixture: corner bin 0 with `boundary='symmetric'` and `window=1` returns mean ≈ 0.5 and count = 100 instead of the expected mean ≈ 0.0 and count = 50. This is **Incident 7** (F1) below and is the subject of in-flight Phase 13.17.GB.

### Root Cause (of the dispute itself, not the aggregate-path bug)

The original 13.8.GB implementation covered only the fit path (V3b). The aggregate path was added in Phase 13.14.GB (`make_sliding_window_aggregate`) and, independently of the fit path, accepted the `boundary` parameter at the API surface without propagating it to either the numba kernel or the numpy fallback. The "disputed" characterization arose because (a) no single document listed which path had which `boundary` support, (b) the commit message was aspirational for fit but silent about aggregate, and (c) the 4 existing tests only covered the fit path. The architect's experience of "only `boundary='full'` actually works" was correct for the aggregate path (which they were using for TPC calibration) even though the fit path worked correctly.

### Lessons (applied to v6.1 documentation discipline)

- **Two paths, two statuses.** Any feature spanning multiple entry points must be documented per-entry-point in the Technical Summary and tested per-entry-point with explicit path-controlling parameters.
- **Commit-message claims must be verified.** The Phase 13.8.GB commit message ("boundary='full'|'symmetric'|'periodic'") was interpreted as covering all of SW, when it covered only the fit path. Commit messages are not authoritative documentation.
- **The dispute resolution itself exposed a new paper-trail defect:** the FIX2 commit message on 2026-04-09 claimed `PHASE_HISTORY.md: v6.0 → v6.1` while leaving the file on disk at v6.0 — the same "aspirational claim not matching disk reality" pattern. v6.1 (this document) corrects the gap.

---

## Critical Incident 5: (Reserved)

> **Incident 5 is reserved.** Per the project cross-team incident numbering maintained by the Main Architect and the Main Reviewer, Incident 5 was assigned to another subproject (outside GBAI scope) during the 2026-04-07 → 2026-04-09 window. GBAI does not own this number. The next two GBAI incidents are Incident 6 (below) and Incident 7 (below). This placeholder is preserved so that future readers of this document are not confused by the numbering gap.

---

## Critical Incident 6: F2 — `method=dict` Silently Drops Interpolation Orders (FIX2, Apr 9, 2026)

### What Happened

`GroupByRegressionEvaluator.evaluate(method={'a': 'linear', 'b': 'cubic'})` silently used a single interpolation order (the first dimension's `order`) for **both** dimensions, producing wrong results for the second dimension without raising an error. Discovered during Coder pre-implementation review of Phase 13.16.GB-FIX2.

### Root Cause

`scipy.ndimage.map_coordinates` accepts only a **scalar** `order` argument — it cannot honor mixed interpolation orders across different dimensions in a single call. The `_eval_per_dimension` dispatch in `groupby_regression_evaluator.py` extracted the first non-`'lookup'` dimension's order and passed that scalar to `map_coordinates`, silently discarding any other dimension's requested order. The bug was reachable whenever the per-dimension dict contained two or more non-lookup methods with different `order` values.

### Fix (Phase 13.16.GB-FIX2, commit `9e88eacd`, 2026-04-09)

`_eval_per_dimension` now validates that all non-lookup dimensions share a single interpolation order. If the dimensions disagree, `ValueError` is raised at dispatch time with the full `method_dict` in the message. `'nearest'` and `'nearest_fast'` are treated as equivalent (both order 0). Unknown method strings (e.g. typo `'linaer'`) are also rejected at dispatch (bonus C10 in the consolidated comments). Supported patterns:
- Zero or more `'lookup'` dimensions + **zero or one** non-lookup interpolation method (any number of copies of the same order)
- `{'a': 'lookup', 'b': 'linear', 'c': 'linear'}` ✅
- `{'a': 'lookup', 'b': 'nearest', 'c': 'nearest_fast'}` ✅ (both order 0)
- `{'a': 'linear', 'b': 'cubic'}` ❌ raises `ValueError` — call `evaluate()` twice with single methods instead

**Severity:** P0 — Hard Constraint #3 (Safety) violation, silent wrong results in a documented code path.

**Failure-mode class:** Parameter not propagated — **instance #4** of the parameter-not-propagated bug class (see Incident 3 catalog, updated).

**Tests added in FIX2:** 5 dedicated F2 tests in `test_evaluator_lookup.py` (validation table in TECHNICAL_SUMMARY v3.3 § Summary Coverage Map), all with explicit path-controlling parameters per failure mode #11 and `pytest.raises(..., match=...)` per consolidated comment C7.

### Why It Survived

Three contributing factors:
1. **No docstring coverage of the dict case** — F3 (P1, fixed in FIX2 alongside F2) — the `evaluate()` docstring did not document the per-dimension dict method at all, so no user knew to test the mixed-order case.
2. **Two-dimension test fixtures** where both dimensions happened to request the same order (`{'a': 'lookup', 'b': 'linear'}`), which silently works because only one non-lookup order exists. The bug only triggered with two or more distinct non-lookup orders.
3. **Type hint was `str`** — the public API declared `method: str`, so static analysis never flagged the dict case as a distinct contract.

### Sibling Fixes in the Same Commit

FIX2 also fixed F3 (docstring gap, P1), F4 (`method='lookup' + bounds='extrapolate'` produced opaque `IndexError` instead of `ValueError`, P1), and F5 (stale `test_select_backend_auto_sequential` flipped from fail to pass after Phase 12.11 auto-dispatch change, P2 cleanup). The F5 flip is the origin of the canonical `528 → 529` test count correction that TECHNICAL_SUMMARY v3.3 failed to absorb.

---

## Critical Incident 7: F1 — `boundary` Parameter Silently Dropped in `make_sliding_window_aggregate` (In progress — Phase 13.17.GB, Apr 9, 2026)

> **Status:** In flight at the time of this document revision. Coder Claude22 is implementing the fix per `PHASE_13_17_GB_v1.3_Proposal`. Commit expected within ~3.5 working days. This incident entry records the diagnosis, the fix plan, and the test expansion; the **Resolution** subsection will be completed (commit hash, final test count, outcome verdict) in a v6.1a revision at commit time.

### What Happened

`make_sliding_window_aggregate(boundary='symmetric')` (and the parallel wrapper `make_sliding_window_aggregate_parallel`) silently used `boundary='full'` behavior regardless of what the caller requested. The architect-required "no edge bias" semantic for TPC distortion calibration (drift, radius, rphi) was violated at every edge bin of every grid computed through this function since Phase 13.14.GB.

**Verified by independent smoke test** (Reviewer Claude23, 2026-04-09) on a linear-rise-in-`dsector` fixture: at corner bin 0 with `window=1`, both `boundary='full'` and `boundary='symmetric'` produce bit-identical output (`mean = 0.499896, count = 100`) instead of the expected symmetric behavior (`mean ≈ 0.0, count = 50`). The differential between `'full'` and `'symmetric'` at corner is **exactly zero** — the `boundary` parameter is not "applied incorrectly," it is **completely dropped** on all three execution paths of the function (primary kernel at line 4381, sigma-cut recompute kernel at line 4448, median slow path at line 4492).

### Root Cause

Lines 4358–4359 of `groupby_regression_sliding_window.py` compute `boundary_resolved = _resolve_boundary(boundary, gb_columns)` and then never read `boundary_resolved` again in the function body (lines 4359–4549 and the parallel wrapper at 4552–4694). The numba accumulation kernel (`_accumulate_numba` at line 4161) and the numpy fallback (`_accumulate_window_agg_numpy` at line 4100) receive only the unfiltered `neighbor_offsets` array — the boundary mode is validated at the entry point and then silently discarded.

**Severity:** P0 — Hard Constraint #3 (Safety) violation, silent wrong results in production-used code path.

**Failure-mode class:** Parameter not propagated — **instance #5** of the parameter-not-propagated bug class. This is the direct continuation of Incident 4 (fit path implemented, aggregate path not) and the fourth incident of this class in six weeks:

| # | Bug | Path | Phase Found |
|---|-----|------|-------------|
| 1 | WLS `weights` silently ignored | V1/V2/V3 regression | 13.9.GB-Ext |
| 2 | `fit_intercept=False` produces intercept columns | `_assemble_results` | 13.9.GB-Ext |
| 3 | `fit_intercept` hardcoded `True` in SW numba (3 locations) | `_fit_window_regression_numba` | P0 Fix (Mar 2026) |
| 4 | `method=dict` silently drops interpolation orders | `_eval_per_dimension` | FIX2 (Apr 9, 2026) — Incident 6 |
| **5** | **`boundary` silently dropped in SW aggregate (3 execution paths)** | **`make_sliding_window_aggregate` + sigma-cut + median** | **13.17.GB (in progress)** |

### Fix Plan (per PHASE_13_17_GB_v1.3_Proposal)

**Strategy: Path 2 — precomputed per-bin valid-offset mask + per-bin wrap flag.** A new helper `_precompute_aggregate_boundary_mask` (added to `groupby_regression_sliding_window.py`, immediately above the aggregate function) mirrors the per-dimension `eff_w` logic already in `_get_neighbor_bins_v2` (the fit path reference implementation) and returns three arrays: `(valid_offset_mask, wrap_flag, wrapped_coords_per_edge_bin)`. These are threaded into the numba kernel and the numpy fallback at **two** execution paths (the median slow path is deferred — see below). For the default `boundary='full'` case the helper returns an all-True mask, an all-False wrap flag, and an empty wrapped-coords array, so the kernel's `if not valid_offset_mask[bi, ni]: continue` check always passes and the fast path is bit-identical to pre-fix. A captured-literal baseline regression gate (**T9**) verifies this.

**Memory safety (§5.1b):** Explicit `mask_bytes` check with a 100 MB `PerformanceWarning` threshold and a 1 GB `MemoryError` hard limit, per the P2/P1 findings of the v1.1 reviewers.

**Deferred sub-scope (D1):** The median slow path at line 4492 is a ~30-line pure-Python triple-nested loop with dict-based `bin_rows` lookup. Integrating it with Path 2 requires either duplicating the fast/slow branching logic in Python (ugly) or restructuring `bin_rows` to a CSR-style flat format (a ~6–10h mini-phase). Per architect direction 2026-04-09 — *"I prefer to fix the bugs — and whatever is too complex, postpone for next phase. You can indicate and quote me — if median is too complicated we can postpone..."* — the median path is deferred to a dedicated follow-up phase **13.17.GB-MedianFix**. The median column continues to silently use `'full'` behavior in 13.17.GB; a `WARNING` paragraph is added to the `make_sliding_window_aggregate` docstring and a Known Limitations row is added to TECHNICAL_SUMMARY v3.4.

### Test Plan (16 tests — unified Claude22 + Claude23 proposal)

All new tests in `tests/test_aggregate_boundary.py`. Each test specifies `boundary=` explicitly per failure mode #11, calls the production entry point `make_sliding_window_aggregate` per failure mode #12, uses the linear-rise-in-`dsector` synthetic fixture (per architect direction), and quotes the architect requirement in its docstring.

**Invariance-to-assertion ratio:** 8 of 16 (50%), up from v1.2's 25% (3 of 12), honoring the architect's D4 note *"The test should be 'Invariant' if possible"*.

| # | Name | Kind | Gate? |
|---|------|------|-------|
| T1 | full corner is biased (reference documentation) | assertion | no |
| T2 | **symmetric corner is unbiased** | **gate** | **primary bug gate** |
| T3 | symmetric interior ≡ full interior | invariance | no |
| T4 | symmetric corner count = 50 | assertion | no |
| T5 | full corner count = 100 | assertion | no |
| T6 | per-dimension dict dispatch | assertion | no |
| T7 | periodic wraps at edges | assertion | no |
| T8 | invalid boundary raises ValueError | regression | no |
| T9 | **default boundary='full' bit-identical to captured literal baseline** | **gate** | **regression gate** |
| T10 | parallel symmetric ≡ serial (2-D fixture) | invariance | no |
| T11 | **symmetric + sigma-cut corner unbiased, interior sym+sigma ≡ interior full+sigma** | **gate + invariance** | **sigma-cut path gate** |
| T13 | **manual oracle ≡ aggregate output** (2-D non-linear fixture, using `_get_neighbor_bins_v2` as the neighbor enumerator + `make_sliding_window_fit(linear_columns=[])` API parity cross-check) | **invariance (oracle)** | strongest test |
| T14 | **numba ≡ numpy cross-backend** via `_get_numba_agg_kernel` monkeypatch × (3 boundaries × 2 paths = 6 combinations) | **invariance (cross-backend)** | direct analog of `test_cross_fitter_parity_fit_intercept_false` from the P0 fit_intercept fix |
| T15 | periodic shift invariance + topology (`n_neighbors_used_sw == prod(2w+1)` on fully-periodic grid) | invariance (two assertion blocks) | no |
| T16 | window=0 ≡ pandas groupby across all 3 boundary modes | invariance (external oracle) | no |
| T17 | constant-field (value = 1.0) invariance across modes and paths | invariance (degenerate) | gross-correctness canary |

Note: **T12 was removed** per the D1 deferral. T13–T17 are new in v1.3 per the Claude22+Claude23 unified test plan. The cross-backend gap (T14) was identified by the fresh reviewer Claude23 and none of the four prior proposal reviewers caught it — the structural analog of the `fit_intercept` P0 test gap is now explicitly pinned in the test suite.

### Resolution

⏳ **Pending implementation.** To be completed in v6.1a at commit time. Expected outcome: **545 passed** canonical (529 baseline + 16 new invariance tests), 2 failed unchanged (the two pre-existing unrelated failures: `test_multiple_fits_match_v4_merged`, `test_tpc_distortion_recovery`), 19 skipped unchanged. Capability Matrix: Broken count 1 → 0 for mean/std/count, with median subpath retained as a scheduled Known Limitation until 13.17.GB-MedianFix.

### Pre-existing Broken Test Flagged for Cleanup

During the 13.17.GB proposal review cycle, Coder Claude22 identified that the existing `test_aggregate_numba_matches_numpy` at line 219 of `test_sliding_window_aggregate.py` is a **false-positive cross-backend invariance** — it calls the same numba backend twice and compares output to itself (auto-dispatch failure mode #11 hiding a real gap). The test has been in the codebase since Phase 13.14.GB and was never caught. T14 supersedes it. Per scope rule 1, the broken test is **not** deleted in 13.17.GB; it is surfaced in the Review Packet Known Issues section with a recommendation for deletion in a separate micro-task.

---

## Critical Incident 8: `_fit_window_regression_numba` — `fit_intercept` Hardcoded True (Found Apr 2026, Fixed in Phase 13.19.GB-PERF)

### What Happened

`_fit_window_regression_numba` hardcoded `fit_intercept=True` at the V1/V2 call site, ignoring the caller's `fit_intercept` parameter. Users calling `make_sliding_window_fit(fit_intercept=False, algorithm='recompute', backend='numba')` received results with an intercept term regardless.

### Root Cause

Same failure pattern as Incident 3 (fit_intercept in SW numba): parameter accepted at the API surface but hardcoded at the internal call site.

**Severity:** P1 — production TPC calibration typically uses `fit_intercept=True` (default), so the bug had no impact on production. Users explicitly setting `fit_intercept=False` would get silently wrong results.

**Failure-mode class:** Parameter not propagated — **instance #8**.

### Resolution

`[FOUND-WHILE-IMPLEMENTING-F1]` — discovered by Coder Claude22 while rewiring the V1/V2 dispatch in Phase 13.19.GB-PERF. Fixed inline: `fit_intercept` parameter added to `_fit_window_regression_numba` signature and threaded through. Authorized by architect as an inline fix. Covered by `test_v1v2_boundary_parameter_silently_dropped` with `parametrize(fit_intercept=[True, False])` and `test_2d_ols_no_intercept` in `test_fit_path_perf_invariance.py`.

---

## Critical Incident 9: V1/V2 Recompute Path Silently Ignores `boundary` Parameter (Pre-existing, Surfaced Apr 2026)

### What Happened

`make_sliding_window_fit(boundary='symmetric')` with `algorithm='recompute'` (the default V1/V2 path) silently uses `boundary='full'` regardless of the caller's request. The `boundary` parameter is validated at the entry point (`_resolve_boundary`, line ~341) but never read by `_aggregate_window_zerocopy` (line ~509) or its replacement `_aggregate_window_dense` (Phase 13.19.GB-PERF). Both functions hardcode the full-truncation boundary via their `bounds_lo/hi` mask.

### Root Cause

This is the mirror of **Incident 7** (F1, aggregate path `make_sliding_window_aggregate`): Incident 7 resolved the silent-drop in the aggregate path; Incident 9 records the still-present silent-drop in the parallel fit-path surface. Same failure pattern: parameter accepted at the API surface but silently ignored in the internal code path.

**Severity:** P1 — production TPC calibration uses `algorithm='recompute'` (default) with `boundary='symmetric'` (architect requirement). The `boundary` parameter has no effect on the V1/V2 path. Only users who explicitly set `algorithm='incremental'` (V3/V5) get correct boundary handling.

**Failure-mode class:** Parameter not propagated — **instance #9**.

### Resolution

Deferred to a dedicated follow-up phase (Phase 13.XX.GB-BoundaryV1V2). Phase 13.19.GB-PERF scope was restricted to performance routing ("zero behavioral change"). Adding boundary support would be a behavioral change requiring its own invariance tests. Precedent: Phase 13.17.GB needed 28 tests for boundary in the aggregate path — the 13.XX.GB-BoundaryV1V2 phase should follow the same template, threading the boundary parameter through `_aggregate_window_dense` the same way 13.17.GB threaded it through the aggregate path's kernel.

Regression-locked by `test_v1v2_boundary_parameter_silently_dropped` in `test_fit_path_perf_invariance.py` — test documents the current behaviour and will correctly fail when the fix lands.

---

## March 2026 Phases

### Phase 13.10.GB: Non-Linear Sliding Window Fit (Feb 16-17, 2026)
**Commit:** `e1650530` | **Tags:** `PHASE_13_10_GB_BEGIN`, `PHASE_13_10_GB_END`
**Review:** 4/5 approved (Round 2)

Separate `make_nonlinear_sliding_window_fit()` entry point for non-linear models. Named model registry with 6 built-in models (gaussian_plus_line, polynomial2, etc.). Evaluator extended with Option A (interpolate params → evaluate) and Option B (evaluate at corners → interpolate values).

**Deliverables:**
- `groupby_regression_models.py` — 322 lines, model registry + p0 estimators
- `groupby_regression_nonlinear.py` — 652 lines, non-linear SW entry point
- Evaluator: `evaluate_model`, `evaluate_function`, `evaluate_params`
- 39 new tests (7 invariance, 4 integration)
- `scripts/phase_tag.sh` — automated phase tagging

**Key decision:** Separate entry point (Approach B), not extending `make_sliding_window_fit` — non-linear has different parameter space, metadata, and output columns.

**Tests:** 447 passed, 3 failed (pre-existing). 126 features, 43 Verified, 0 Broken.

---

### Phase 13.10.GB-T: K0s THn Tutorial (Feb 18, 2026)
**Review:** 5/5 approved

End-to-end MC truth validation of non-linear SW pipeline. Synthetic K0s invariant mass spectra in 4D phase space (1/pt, tgl, phi, occupancy) with `gaussian_plus_line` fits.

**Issues discovered (tutorial's primary purpose):**
- ISSUE-1: χ²/ndf scales with statistics → needs chi2 audit (Phase 13.11.GB)
- ISSUE-2: SW concatenates rows instead of summing histograms → design limitation (Phase 13.12.GB)
- ISSUE-3: Offset/slope pull σ ≈ 0.66 → parameter correlation, not a bug
- ISSUE-4: SW sigma pull grows with statistics → consequence of ISSUE-2

---

### Phase 13.9.GB Extensions (Feb 23 – Mar 13, 2026)
**Commits:** `ce1361d6` through `9a13130e` | **Review:** 3/3 to 4/4 approved per sub-extension

Five sub-extensions addressing specification gaps and bugs found during real-data testing:

**Extension 1: agg_columns (Feb 23)**
- `agg_columns` parameter for COG/window statistics (mean/std of arbitrary columns)
- All 3 code paths: zerocopy, V3 incremental, V5 numba
- Kernel-weighted aggregation for non-uniform kernels
- 8 new tests. Resolves COG gap from original specification.

**Extension 2: WLS Weight Fix (Mar 11)**
- **P0 bug:** `weights` parameter silently ignored in regression (used only for aggregation)
- Fix: `sqrt(w)` transform in V1 numpy, V2 falls back to V1, V3 weighted XtX/XtY
- R²/RMSE on unweighted residuals (documented convention)
- 7 new tests

**Extension 3: fit_intercept=False Column Fix (Mar 11)**
- **P0 bug:** Output contained intercept columns with 0/NaN when `fit_intercept=False`
- Fix: `_assemble_results` conditional on `fit_intercept`
- 2 new tests

**Extension 4: Lean Output (Mar 12)**
- **Architect decision:** Remove default fit_column stats (mean/std/median/entries/r_squared per target)
- 52 → ~28 columns for 4-target fits
- Opt-in via `agg_columns` to restore. Canonical: `agg_columns = gb_columns + linear_columns`
- 2 new tests, 6 test files updated

**Extension 5: Parallel Parity (Mar 13)**
- Propagate `agg_columns`, `agg_median`, `fit_intercept` to parallel workers
- Option B (minimal parameter propagation, no architecture change)
- RuntimeError if all parallel units fail (was silent empty DataFrame)
- forkserver/spawn context for Numba thread conflicts (256-core fix)
- 4 new tests. 466 passed.

---

### Phase 13.13.GB: Expression-Based Linear Columns (Mar 23, 2026)
**Commits:** `f0cb33f3`, `bee8226b` | **Tags:** `PHASE_13_13_GB_BEGIN`, `PHASE_13_13_GB_END`
**Review:** 4/4 approved

`linear_columns` accepts `(key_name, expression)` tuples alongside strings. Expressions evaluated via `df.eval()` at entry point, kernels unchanged.

**Key design:** Preprocessing layer (`_preprocess_linear_columns`) at API boundary. Kernels receive augmented DataFrame with all columns materialized. Clean naming via explicit key — no `slope_xM**2 * driftM` invalid identifiers.

**Integrated into:** `make_parallel_fit_v4` and `make_sliding_window_fit`. Parallel SW deferred (P2).

**Metadata:** `linear_column_map` + `linear_columns_normalized` stored in output.

**Performance:** `df.eval()` ~13% slower than numpy for 20 expressions — acceptable.

**Tests:** 19 new. 485 passed, 3 failed (pre-existing).

---

### Phase 13.14.GB: Dedicated Sliding Window Aggregation (Mar 24, 2026)
**Commits:** `190dbd63` through `030543737` | **Tags:** `PHASE_13_14_GB_BEGIN`, `PHASE_13_14_GB_END`
**Review:** 4/4 approved (v1.1 corrected proposal)

New `make_sliding_window_aggregate` for pure aggregation (mean/std/count) without regression. Uses per-bin sufficient statistics and scalar window accumulation.

**Motivation:** `make_sliding_window_fit` with `linear_columns=[]` took 440s per sector for 815k bins. Aggregation does not need design matrices or regression.

**Algorithm:**
1. Bin mapping (counting sort) — O(N)
2. Per-bin sufficient stats (np.bincount) — O(N×C)
3. Window accumulation (Numba prange) — O(B×W×C) scalar adds
4. Compute mean/std — O(B×C)

**Performance optimizations (3 rounds):**
- Step 2: `np.bincount` replaces Python loop (2.3s → 0.013s, 177×)
- Step 3: `nb.prange` parallelizes over bins
- `_build_dense_lookup`: vectorized numpy (3.6s → <0.05s)
- **Total: 440s → ~1.5s per time frame (~300× speedup)**

**Key formula correction (all 4 reviewers caught):** Variance uses `kw` not `kw²` for kernel weights. `kw²` is for error propagation (different operation).

**Tests:** 10 new (4 invariance, 1 performance gate). 494 passed.

> **v6.1 note.** 13.14.GB is the phase where F1 (Incident 7) was introduced. The `boundary` parameter was accepted at the API surface from this phase onward but never propagated to the kernel. Discovered during real-data TPC testing in the 13.16.GB era. In-flight fix in Phase 13.17.GB. 13.14.GB is also the phase where the false-positive `test_aggregate_numba_matches_numpy` test was introduced at line 219 of `test_sliding_window_aggregate.py` — this test calls the same numba backend twice and compares output to itself (failure mode #11), hiding the cross-backend invariance gap that the 13.17.GB Claude23-proposed T14 now closes.

---

### Phase 13.15.GB: N-Sigma Cut in SW Aggregation (Mar 26, 2026)
**Commit:** `efdfb130` | **Tags:** `PHASE_13_15_GB_BEGIN`, `PHASE_13_15_GB_END`
**Review:** 3/3 approved

Two-pass sigma-clipped aggregation: Pass 1 computes mean/std, sigma cut flags outliers per row, Pass 2 recomputes clean statistics. Reuses existing sufficient stats + Numba kernel.

**Parameter:** `n_sigma_cut` (default `None` — no change). Cost: ~1.7× current.

**Guards:** `safe_std = inf` for std=0 bins (prevents all-outlier marking).

**Design choice:** Cut uses per-bin window mean/std (smoothed by neighbors), not raw per-bin stats. This gives better statistics for sparse bins.

**Tests:** 6 new (5 invariance, 1 smoke). 500 passed, 4 failed (pre-existing).

---

### Phase 13.16.GB: Evaluator Lookup + scipy Interpolation Methods (Mar 28-29, 2026)
**Commit:** `0ed8a56c` (combined with P0 fix) | **Review:** Approved

Extended `GroupByRegressionEvaluator.evaluate()` with new interpolation methods for performance and mixed-grid support.

**New methods:**

| Method | Implementation | Speed | Use case |
|--------|---------------|-------|----------|
| `'multilinear'` | Python corner-weighted (legacy) | Baseline | Default; supports `use_errors=True` |
| `'linear'` | scipy `map_coordinates` order=1 | ~3× faster | Recommended default |
| `'cubic'` | scipy `map_coordinates` order=3 | ~2× faster | C1-continuous output |
| `'nearest'` | searchsorted + snap | Fast | Discrete lookup |
| **`'lookup'`** | **Direct integer array indexing** | **>7× faster** | **Integer grids only** |
| **`dict`** | **Per-dimension method dispatch** | **Variable** | **Mixed integer/continuous grids** |

**Key design — `method='lookup'`:** Skips `_find_cell` (and `searchsorted`) entirely. Positions are used as raw grid indices via numpy fancy indexing (`grid[sector_array, padrow_array]`). Requires integer-like positions on a 0-based contiguous grid. Bounds handling via existing `bounds` parameter (clamp/nan). `ValueError` on non-integer positions.

**Key design — per-dimension method dict:**

```python
ev.evaluate(
    positions={'sector': sectors, 'padrow': padrows, 'zBin': z_bins},
    method={'sector': 'lookup', 'padrow': 'lookup', 'zBin': 'linear'},
)
```

Each dimension uses its specified method independently. Missing keys default to `'linear'`. Lookup dimensions use direct indexing; interpolation dimensions use `map_coordinates`. The mixed case builds a full D-dimensional coordinate array and passes it to `map_coordinates` — integer dimensions act as exact grid positions.

**Motivation:** Detector calibration maps (TPC as example) have mixed grid structure — sector (0–35) and padrow (0–152) are integer categorical, drift/z are continuous. `searchsorted` on integer indices is pure overhead. For 82M track positions × 3D integer grid: ~60s with `nearest` → ~8s with `lookup` (estimated >7×).

**Tests:** 6 new in `test_evaluator_lookup.py`
- `test_lookup_equals_nearest_for_integers` (invariance gate, `assert_array_equal`)
- `test_lookup_out_of_bounds_nan`, `test_lookup_out_of_bounds_clamp`
- `test_lookup_non_integer_raises`
- `test_per_dimension_method_dict` (mixed lookup + linear)
- `test_lookup_performance` (directional, >1.5× faster)

**Tests total:** 517 passed, 3 failed (pre-existing), 0 regressions.

> **v6.1 follow-up note.** Phase 13.16.GB shipped a silent bug in the `method=dict` dispatch (F2) — see Incident 6. F2 was found during Coder pre-implementation review of 13.16.GB-FIX2 and fixed in that follow-up phase. The original 13.16.GB review cycle did not catch F2 because all tests of the `method=dict` case used two dimensions with the same interpolation order (e.g. `{'sector': 'lookup', 'padrow': 'linear'}`), which silently works because only one non-lookup order exists.

---

### P0 Fix: `fit_intercept` in SW Numba Path (Mar 28-29, 2026)
**Commits:** `a9c6d07f`, `0ed8a56c`, `1278986b`
**Review:** Approved after 2 rounds (first round: tests detected unfixed bug)

See "Critical Incident 3" above for the full root cause analysis.

**Deliverables:**
- 3-line fix in `_fit_window_regression_numba` (parameter, n_params, unpacking offset)
- `test_fit_intercept_all_fitters.py` — 10 cross-fitter invariance tests
- Each test explicitly specifies `backend=` to prevent auto-select masking
- 2 full-chain tests with `window > 0` exercising the complete recompute path
- V2 excluded (legacy API does not support `fit_intercept`)

---

## April 2026 Phases

### Phase 13.16.GB-FIX2: Evaluator Bug Fixes F2/F3/F4/F5 + C3/C8/C10 (Apr 9, 2026)
**Commit:** `9e88eacd` | **Tag:** `PHASE_13_16_GB_FIX2_END`
**Review:** 5 APPROVED + 1 CONDITIONALLY APPROVED (conditions resolved pre-commit)
**Coder:** Claude21 (GBAI)
**Main Reviewer:** Claude20 (GBAI)
**Proposal:** `PHASE_13_16_GB_FIX2_v1.0_Proposal` (author: Claude20)
**Review summary:** `PHASE_13_16_GB_FIX2_ReviewSummary_v1_0.md`
**Baseline:** 517 passed / 3 failed / 19 skipped (canonical `alma2`)

Post-13.16.GB fix phase addressing four evaluator bugs (F2/F3/F4/F5) and three consolidated review comments (C3/C8/C10). The proposal was drafted by Main Reviewer Claude20 after the 13.16.GB commit cycle, reviewed by five reviewers (including an Architect Reviewer role), and approved after a single review round.

**Bugs fixed:**

- **F2 (P0, instance #4 of parameter-not-propagated bug class)** — `method=dict` with mixed interpolation orders silently picked the first dimension's order. Fix: `_eval_per_dimension` now raises `ValueError` at dispatch time when non-lookup dimensions disagree on `order`. `'nearest'` and `'nearest_fast'` treated as equivalent (both order 0). See **Incident 6**.
- **F3 (P1, public API doc gap)** — `evaluate()` and `get_coefficients()` docstrings did not document `'lookup'`, `'nearest_fast'`, or the per-dimension `dict` shape. Fix: full docstring rewrite + two runnable examples, verified by `test_evaluate_docstring_examples_run`. Type hint corrected from `method: str` to `method: Union[str, Dict[str, str]]`.
- **F4 (P1, error-class drift)** — `evaluate(method='lookup', bounds='extrapolate')` fell through to an opaque `IndexError` from numpy fancy indexing. Fix: `_eval_lookup` rejects unsupported `bounds` values at dispatch time with a clear `ValueError`.
- **F5 (P2, stale test)** — `test_select_backend_auto_sequential` was failing from Phase 12.11 onward (auto-dispatch changed to return `'numba'` instead of `'numpy'` for `n_jobs=1`). Fix: update assertion + add skipif guard + full provenance comment.

**Sibling improvements in the same commit (consolidated comments C3/C8/C10):**
- **C3** — Treat `'nearest'` and `'nearest_fast'` as equivalent for the F2 order-agreement check (both order 0)
- **C8** — Docstring examples runnable as pytest tests (`test_evaluate_docstring_examples_run`)
- **C10** — Unknown method strings in per-dimension dict (e.g. typo `'linaer'`) raise `ValueError` at dispatch time (one-line addition alongside F2)

**Tests added:** 11 new in `test_evaluator_lookup.py` + 1 stale fix in `test_phase_12_9_gb.py`. All new tests specify path-controlling parameters per failure mode #11 and use `pytest.raises(..., match=...)` per consolidated comment C7.

**Test count delta:** 517 → **529 passed** (canonical `alma2`), 3 → 2 failed (F5 flipped from fail to pass), 19 skipped unchanged. **Note:** TECHNICAL_SUMMARY v3.3 committed alongside this phase records `528 passed` at four locations due to an arithmetic error in the Coder's Review Packet (`517 + 11 = 528` omits the `+1` from the F5 flip). Three reviewers flagged this during review; the correction was not absorbed before commit. Canonical number is **529**, confirmed against `SUMMARY_20260409_112129.txt` in the FIX2 reviewer packet. Closure of the 528→529 discrepancy is scheduled as part of TECHNICAL_SUMMARY v3.4 in Phase 13.17.GB.

**Broken feature count:** 2 → **1** (F2 fixed in this phase; F1 remains for Phase 13.17.GB).

**Documentation delivered:** TECHNICAL_SUMMARY.md v3.2 → v3.3, CAPABILITY_MATRIX.md regenerated. **Paper-trail defect:** the FIX2 commit message also claims `PHASE_HISTORY.md: v6.0 → v6.1` but the file on disk was not actually updated. This v6.1 (drafted 2026-04-09 at architect request) closes that gap.

**Known follow-up (out of FIX2 scope, tracked for 13.17.GB or separate cleanup):**
- `test_evaluator_lookup.py` not discovered by `feature_taxonomy.py` (pre-existing Phase 13.16.GB gap)
- `test_aggregate_numba_matches_numpy` at line 219 of `test_sliding_window_aggregate.py` is a false-positive cross-backend invariance (identified during 13.17.GB pre-implementation review by Coder Claude22; superseded by T14 in 13.17.GB; deletion in a separate micro-task after 13.17.GB commits)

---

### Phase 13.17.GB: `boundary='symmetric'` in `make_sliding_window_aggregate` — F1 Fix (In progress, started Apr 9, 2026)
**Status:** ⏳ **In progress** — Coder Claude22 active, commit expected within ~3.5 working days
**Proposal:** `PHASE_13_17_GB_v1.3_Proposal` — APPROVED (v1.1 3/3 APPROVED; v1.2 APPROVED by Architect Reviewer + Claude11; v1.3 absorbs D1 deferral and the Claude22+Claude23 unified 16-test plan, no re-review cycle required)
**Tier:** Tier 1 (Full panel — touches numba kernel and a parameter used in production)
**Coder:** Claude22 (GBAI)
**Main Reviewer:** Claude20 (GBAI)
**Reviewers (v1.1):** Anonymous Claude Opus, GPT10, GPT11 (3/3 APPROVED)
**Reviewers (v1.2 / v1.3):** Claude Opus ArchReviewer, Claude11 APPROVED; Claude23 APPROVED WITH COMMENTS (3 P1 / 3 P2) → APPROVED after v1.3 absorbed D1 deferral
**Predecessor:** Phase 13.16.GB-FIX2 (`9e88eacd`)
**Baseline:** 529 passed / 2 failed / 19 skipped (canonical `alma2`)
**Target at close:** 545 passed (529 + 16 new invariance tests), 2 failed unchanged, 19 skipped unchanged

Fixes the F1 bug in `make_sliding_window_aggregate` (and parallel sibling) — `boundary='symmetric'` and `boundary='periodic'` silently dropped — on **two** execution paths: the primary kernel call at line 4381 and the sigma-cut recompute kernel call at line 4448. The median slow path at line 4492 is **deferred** to Phase 13.17.GB-MedianFix per architect direction (*"if median is too complicated we can postpone"*, 2026-04-09) — the median column continues to use `'full'` behavior in this phase, documented as a Known Limitation.

See **Incident 7** above for the full root cause analysis, the fix strategy (Path 2 — precomputed mask + per-bin wrap flag), the 16-test unified plan, and the pre-existing broken test flagged for cleanup.

**Key technical decisions (new to 13.17.GB):**
- **Path 2** (mask + per-bin wrap flag) over Path 1 (dual dense arrays) — Path 1 is ~700 MB for 6D worst case, unacceptable at scale
- **Path 3 fallback pre-authorized** — if Path 2 compromises the T9 bit-identity regression gate, the Coder may fall back to the numpy path for `boundary='periodic'` without re-asking (architect-accepted contingency; not yet triggered at time of this v6.1)
- **Explicit memory safety check** (`_check_mask_memory_safety`) — 100 MB `PerformanceWarning` threshold, 1 GB `MemoryError` hard limit
- **Signature-chain grep check** before commit — prevents the same "3 hardcoded assumptions at different locations" failure pattern that made the `fit_intercept` P0 a 3-commit fix
- **T9 runs first** — the regression gate is verified before any other test; if T9 fails, the fix is wrong regardless of what the other tests show
- **D1 (median path) deferred to 13.17.GB-MedianFix** — ~6–10h follow-up phase, not the 15-line fix the v1.2 sketch implied

**Governance lesson embedded in the test plan (new in v1.3):** The T14 cross-backend invariance test (`_get_numba_agg_kernel` monkeypatch × 3 boundaries × 2 paths = 6 combinations) was proposed by Reviewer Claude23 during option (b) verification (paper review + independent F1 smoke test against source). Neither Claude20 (Main Reviewer), Anonymous Claude Opus, GPT10, GPT11, nor Claude Opus ArchReviewer flagged the cross-backend gap in the v1.1 or v1.2 review cycles. T14 is the structural analog of `test_cross_fitter_parity_fit_intercept_false` from the P0 `fit_intercept` fix — the same class of test that would have caught the fit_intercept bug before it reached production, now added pre-emptively to the aggregate path. **Claude23's finding validates the fresh-reviewer rotation discipline in MTTU v1.20** and is the case for continuing to add fresh reviewer slots to Tier 1 panels.

### Resolution

⏳ **Pending commit.** v6.1a revision at commit time will fill in: final commit hash, final canonical/coder-env test counts, Capability Matrix Broken count update, TECHNICAL_SUMMARY v3.4 link, and any fallback-protocol invocations (Path 2 → Path 3 for periodic) or memory-safety warnings encountered during implementation.

---

### Phase 13.19.GB-PERF: Fit Path Performance Parity with Aggregate Path (Apr 18, 2026)

**Commits:** `ae565fd5` (F1 implementation), `180bf206` (T1 tests per v1.2 spec)
**Proposal:** PHASE_13_19_GB_PERF_v1.0_Proposal.md (Claude23, profile-driven)
**Implementation spec:** v1.2 (Claude24 + Claude25 consolidated)
**Profile evidence:** `profile_gr11_tf0.prof` — 82M rows, 1452s total

Routes V1/V2 recompute path (the default in production calibration) through the dense-lookup infrastructure proven in Phase 13.14.GB's aggregate path. Eliminates two profiled bottlenecks:

- `_build_bin_index_map` (205s cumulative, pure-Python dict with tuple hashing per row) → replaced by `_assign_bin_ids_fast` (vectorized numpy ravel_multi_index, ~1.6s)
- `_get_neighbor_bins` V3a (152s cumulative, 2.5M per-bin Python function calls) → inlined into `_aggregate_window_dense` (vectorized offset + dense array lookup)

Predicted total savings: ~355s on the 82M-row calibration workload (~24% of pipeline). T2 re-profile on alma2 pending (gates `PHASE_13_19_GB_PERF_END` tag).

**`[FOUND-WHILE-IMPLEMENTING-F1]`** `_fit_window_regression_numba` gains `fit_intercept` parameter — was hardcoded `True` at V1/V2 call site. Latent bug fix, parameter-not-propagated class instance #8 (see Incident 8).

**Known Limitation surfaced:** V1/V2 recompute path silently ignores the `boundary` parameter — parameter-not-propagated class instance #9 (see Incident 9). Pre-existing, not introduced by this phase. Regression-locked by `test_v1v2_boundary_parameter_silently_dropped`.

**Behavior change:** output DataFrame row ordering from `make_sliding_window_fit(algorithm='recompute')` is now sorted lexicographically by bin coordinates (was data-encounter order). Values identical; row ordering differs. Locked by `test_output_row_order_sorted_lex`.

**F2 (per-window median batching):** deferred. Proposal author (Claude23) confirmed the original sketch was wrong for the fit-path median (overlapping windows cannot be batched by per-bin pre-sort). Architect: "if not primitive, postpone."

**Tests:** 14 T1 invariance tests in `test_fit_path_perf_invariance.py` (6 original + 8 per v1.2 spec). All 14 passed on alma2. Tolerance: `rtol=1e-12`, max observed divergence `rtol=2.96e-13` in error columns only (accumulation-order rounding). Coefficients unaffected.

**Review panel:** Claude20 (Main), Claude21, Claude23, Claude24 (source-read diff walk, P0-1 finding), Claude25 (BLOCKED then source-read, M-5/M-6 CRR additions). Claude24 produced the strongest source-read review — caught undisclosed `fit_intercept` fix that 3 prior reviewers missed.

---

## Governance Observations (NEW in v6.1)

### Observation 1: FIX2 commit-message claim mismatched disk reality

The Phase 13.16.GB-FIX2 commit message (2026-04-09, `9e88eacd`) includes the line:

> `PHASE_HISTORY.md: v6.0 -> v6.1`

but the `PHASE_HISTORY.md` file on disk remained at v6.0 throughout the FIX2 commit cycle and beyond. The PHASE_HISTORY v6.1 revision was drafted only at architect request on 2026-04-09 (this document). This is the same **"aspirational commit message not matching disk reality"** pattern as:
- **TECHNICAL_SUMMARY v3.3 `528` vs canonical `529`** — three reviewers flagged the arithmetic error; the correction was not absorbed before commit
- **Phase 13.8.GB commit message `boundary='full'|'symmetric'|'periodic'`** — covered only the fit path, not the aggregate path, which gave rise to the original Incident 4 "disputed" framing

**Lesson:** Commit-message claims about documentation updates must be verified against disk before commit. A simple pre-commit check — "does `git diff --stat` show the file the commit message claims to have updated?" — would catch all three instances.

### Observation 2: Fresh-reviewer slot caught what 4 prior reviewers missed

The T14 cross-backend invariance gap for Phase 13.17.GB was identified by Reviewer Claude23 during first exposure to the proposal. The four prior reviewers (Anonymous Claude Opus, GPT10, GPT11, and Claude Opus ArchReviewer) all performed paper review only and did not reproduce F1 independently. Claude23's option (b) verification depth (paper review + independent smoke test on the linear-rise-in-dsector fixture) revealed the gap during source indexing, before the verdict was written.

**Lesson:** The fresh-reviewer rotation discipline in MTTU v1.20 has measurable value. Tier 1 panels should retain at least one reviewer slot for a reviewer who has no prior context in the subproject, and that reviewer should be encouraged to perform option (b) or (c) verification depth (hands-on reproduction) rather than option (a) (paper review only). The architect is considering adding 2 new fresh GPT reviewers in the next round — this observation is supportive evidence.

### Observation 3: Parameter-not-propagated bug class is still producing instances

At the time of the P0 `fit_intercept` fix (Mar 29, 2026), the parameter-not-propagated bug class had 3 known instances. In the six weeks since, two more have been found — F2 (`method=dict` in FIX2) and F1 (`boundary` in the aggregate path, in-flight in 13.17.GB). The class instance count is now **5**. The `test_aggregate_numba_matches_numpy` pre-existing broken test (identified during 13.17.GB pre-implementation review) is a **sixth latent instance** of the related failure mode #11 (auto-dispatch hiding a code-path gap) that has been in the codebase since Phase 13.14.GB and was not caught until Coder Claude22 reviewed the source for the unified test plan.

**Lesson:** The bug class is not dormant. Every new public parameter added to a function that has multiple backends or multiple call sites must be checked at commit time with an explicit **per-path × per-backend** test matrix. The 13.17.GB T14 test (6 combinations: 3 boundaries × 2 execution paths) is the template for this check going forward. MTTU v1.20 failure mode #11 already requires explicit path-controlling parameters in invariance tests; v1.21 or later should consider promoting "per-path × per-backend test matrix for new parameters" from SHOULD to MUST.

---

## Key Technical Decisions

### From Phases 13.10.GB – 13.15.GB (preserved from v5.0)

| Decision | Rationale | Phase |
|----------|-----------|-------|
| Separate non-linear entry point | Different parameter space, metadata, output columns | 13.10.GB |
| Both evaluator options (A+B) | Option A for smooth interpolation, Option B for exact at corners | 13.10.GB |
| `agg_columns` not new parameter — uses existing mechanism | No API proliferation; fit_columns stats opt-in | 13.9.GB-Ext |
| `sqrt(w)` WLS transform (non-mutating) | Keep originals for unweighted diagnostics | 13.9.GB-Ext |
| V2 falls back to V1 for WLS | Same pattern as V5→V3; keep numba kernel simple | 13.9.GB-Ext |
| R²/RMSE on unweighted residuals | Consistent with OLS behavior, backward-compatible | 13.9.GB-Ext |
| Default output: no fit_column stats | Architect direction; 52→28 columns; opt-in via agg_columns | 13.9.GB-Ext |
| Tuple `(key, expr)` for linear_columns | Clean naming; `df.eval()` at entry point, kernels unchanged | 13.13.GB |
| Sufficient statistics for aggregation | O(B×W×C) scalar adds vs O(N×W) per-row; 300× speedup | 13.14.GB |
| Variance formula: `kw` not `kw²` | Weighted variance of data, not error propagation | 13.14.GB |
| Per-column NaN counts | Different NaN patterns across columns; 39 MB acceptable | 13.14.GB |
| Dense lookup array for Numba | O(1) coord→bin; ~16 MB; Numba-friendly (no dict) | 13.14.GB |
| Single-pass sigma clipping | Sufficient for TPC outliers (extreme, sparse); iterative deferred | 13.15.GB |
| Sigma cut uses window mean/std | Better statistics for sparse bins; avoids neighbor contamination | 13.15.GB |

### From Phase 13.16.GB and P0-Fix (preserved from v6.0)

| Decision | Rationale | Phase |
|----------|-----------|-------|
| `'lookup'` method skips `_find_cell` entirely | O(1) array indexing for integer grids; fastest possible path | 13.16.GB |
| Per-dimension method dict (not single mode) | Detector grids have mixed integer/continuous dimensions; single string can't express this | 13.16.GB |
| scipy `map_coordinates` for linear/cubic | C-implemented, ~3× faster than Python multilinear | 13.16.GB |
| Mixed dict builds full coordinate array for `map_coordinates` | `map_coordinates` handles integer coords as exact grid positions; no special case needed | 13.16.GB |
| Cross-fitter invariance tests must specify `backend=` explicitly | `backend='auto'` masked the fit_intercept bug for 4 review rounds | P0-Fix |
| SW invariance tests must use `window > 0` | `window=0` routes to V5, bypassing V2 numba recompute path | P0-Fix |
| V2 excluded from fit_intercept tests | Legacy API does not support `fit_intercept` parameter | P0-Fix |

### From Phase 13.16.GB-FIX2 (new in v6.1)

| Decision | Rationale | Phase |
|----------|-----------|-------|
| `_eval_per_dimension` rejects mixed interpolation orders at dispatch time | scipy `map_coordinates` accepts only scalar `order`; per-axis mixing is impossible in a single call — raising `ValueError` is the only correctness-preserving option | 13.16.GB-FIX2 |
| `'nearest'` and `'nearest_fast'` treated as equivalent for order-agreement check | Both are scipy order=0; the name difference is implementation, not semantics | 13.16.GB-FIX2 |
| Docstring examples run as pytest tests | Closes the C8 gap — documentation claims about runnable examples must be verified by an executable test, not by reviewer inspection | 13.16.GB-FIX2 |
| Unknown method strings in per-dimension dict raise `ValueError` (typo detection) | One-line addition alongside F2; catches `'linaer'` before it silently defaults to `'linear'` | 13.16.GB-FIX2 |
| Stale test correction committed alongside feature fix | Inherited failing tests are a form of technical debt; fixing them in the same commit as related work is cheaper than a separate cleanup phase | 13.16.GB-FIX2 |

### From Phase 13.17.GB (new in v6.1, plan — subject to change at commit)

| Decision | Rationale | Phase |
|----------|-----------|-------|
| **Path 2** (mask + per-bin wrap flag) over Path 1 (dual dense arrays) | Path 1 is ~700 MB for 6D worst case, unacceptable at scale; Path 2 bounds the per-bin wrapped-coords storage to edge bins only | 13.17.GB |
| Path 3 (numpy fallback for periodic) pre-authorized as T9-failure contingency | Preserves Hard Constraint #3 (Safety) over performance — if the default path is not bit-identical, correctness wins and periodic accepts a performance cost | 13.17.GB |
| Explicit memory safety check with 100 MB warn / 1 GB hard limit | Raised as P1/P2 by GPT11 and Anonymous Claude Opus in v1.1 review; the alternative is a silent OOM on 6D grids | 13.17.GB |
| Signature-chain grep check before commit | Prevents the `fit_intercept` failure pattern where 3 hardcoded assumptions at different locations each hid the others | 13.17.GB |
| T9 (bit-identity against captured literal baseline) runs first | The regression gate must pass before any other test; a captured literal baseline is preferred over "compare to pre-fix output" because pre-fix output no longer exists after the fix | 13.17.GB |
| T14 cross-backend invariance via `_get_numba_agg_kernel` monkeypatch | Direct structural analog of `test_cross_fitter_parity_fit_intercept_false` from the P0 `fit_intercept` fix; closes the auto-dispatch failure mode #11 gap on the aggregate path | 13.17.GB |
| D1 (median path + `boundary`) deferred to 13.17.GB-MedianFix | Architect direction 2026-04-09: *"if median is too complicated we can postpone"* — the median path is a ~6-10h mini-phase due to dict-based `bin_rows` restructuring, not a 15-line fix | 13.17.GB |

---

## Performance Reference

### Sliding Window Aggregation (Phase 13.14.GB, Linux aarch64)

| Scenario | Before | After | Speedup |
|----------|--------|-------|---------|
| 1 sector (815k bins, 2M rows, 4D window) | 440s | ~1.5s | **~300×** |
| 36 sectors sequential | ~4.4 hours | ~54s | **~300×** |
| With `n_sigma_cut=3.0` | — | ~2.5s | ~1.7× vs no cut |

### Evaluator Lookup (Phase 13.16.GB)

| Scenario | nearest (searchsorted) | lookup (direct index) | Speedup |
|----------|----------------------|----------------------|---------|
| 100k rows, D=3, integer grid | ~0.07s | ~0.01s | >5× |
| 10M rows, D=3, integer grid | ~7s (est.) | <1s (est.) | >7× |
| 82M rows, D=3, integer grid | ~60s (est.) | ~8s (est.) | >7× |

### Full makeIterationFit0 Pipeline (Mar 31, 2026)

| Configuration | Time | Notes |
|---------------|------|-------|
| `sec` (0–35) included in `gb_columns` (dense 69M-cell grid) | 354s | `np.full` and `_build_dense_lookup` overhead dominated |
| Per-sector loop (1.9M cells × 36 sectors) | 18s | **~20× speedup** |

**Root cause:** Including high-cardinality grouping variables in `gb_columns` creates overly large dense grids. Looping per sector reduces grid size by ~36×. The fix is in user code, not the library, but documents a usage pattern other teams should follow. The `np.full` and `_build_dense_lookup` overhead dropped from 64s to ~1s after the fix.

### Capability Matrix (Phase 13.16.GB-FIX2, canonical `alma2`)

| Metric | Value |
|--------|-------|
| Total features | 133 |
| Verified (✅) | 43+ (32.3%) (+11 new FIX2 evaluator tests not yet counted in Capability Matrix — tracked in TECHNICAL_SUMMARY v3.3 § Summary Coverage Map) |
| Smoke-only (☑️) | 89 (66.9%) |
| Broken (🧨) | 1 (F1, in-flight fix in 13.17.GB) |
| Planned (📋) | 1 (0.8%) |
| Total tests passed (canonical) | **529** |
| Total tests passed (TECHNICAL_SUMMARY v3.3 on disk, records `528` due to F5-flip arithmetic error — closure scheduled in v3.4) | 528 (incorrect) |
| Pre-existing failures | **2** (was 3 before FIX2 F5 flip) |
| Skipped | **19** |

---

## Failure Modes Catalog (Cumulative)

| # | Mode | Origin Incident | Prevention Mechanism |
|---|------|-----------------|----------------------|
| 1 | Silent regression (no test) | Numba kernel removed Nov 2025 | Performance gates, parity tests, benchmark framework |
| 2 | Architect requirement deferred without sign-off | COG/agg_columns | Authority Boundary rules (MTTU v1.11+, failure mode #8) |
| 3 | Parameter accepted at API but ignored in path | WLS weights silently ignored | Cross-path invariance tests |
| 4 | Output columns don't match parameter | `fit_intercept` columns with 0/NaN | Output assertion tests |
| 5 | Chain of compensating bugs | `fit_intercept` n_params + unpacking | Full-chain tests, fix one assumption at a time |
| 6 | Tests bypass buggy code path via auto-dispatch | `backend='auto'`, `window=0` masked numba bug | Explicit `backend=` and `window>0` in invariance tests (failure mode #11, MTTU v1.16) |
| 7 | Architect requirement implemented but never tested or documented | `boundary='symmetric'` fit path (Incident 4) | Specification traceability — track every architect requirement to test + doc |
| **8** | **Cross-backend test that calls the same backend twice** (false-positive invariance) | **`test_aggregate_numba_matches_numpy` at line 219 of `test_sliding_window_aggregate.py` — identified during 13.17.GB pre-implementation review** | **Monkeypatch `_get_numba_agg_kernel` (or equivalent) to force the numpy fallback path; compare bit-identically against a normal numba run (T14 template in Phase 13.17.GB)** |
| **9** | **Commit message claims documentation update without touching disk** | **FIX2 commit `9e88eacd` claimed `PHASE_HISTORY.md: v6.0 → v6.1` while leaving the file at v6.0; earlier: Phase 13.8.GB commit claimed `boundary='full'|'symmetric'|'periodic'` for all of SW when only the fit path had it** | **Pre-commit check: verify `git diff --stat` shows every file the commit message claims to have modified** |
| **10** | **Multi-reviewer finding not absorbed into committed document** | **TECHNICAL_SUMMARY v3.3 `528 passed` vs canonical `529` — three reviewers flagged the arithmetic error during FIX2 review; correction was not absorbed before commit** | **Main Reviewer cross-checks final committed document against consolidated review summary before approving commit** |

---

## Planned Phases

| Phase | Content | Status |
|-------|---------|--------|
| **13.17.GB-MedianFix** | **`agg_median=True` + `boundary` interaction — median path honors mask** (D1 deferral from 13.17.GB per architect direction 2026-04-09) | **📋 Scheduled** (immediate follow-up after 13.17.GB commits; ~6-10h) |
| 13.11.GB | Chi2 audit + `--exact-model` diagnostic | Approved, deferred |
| 13.12.GB | SW histogram accumulation mode (`fit_mode='histogram'`) | Proposed, deferred |
| 13.13.GB-B | Batched expression evaluation (memory) | Proposed |
| — | Evaluator Numba bulk evaluate (only if 82M-row profiling demands it) | Conditional |
| — | `register_evaluator` on AliasDataFrame | ADF team scope |
| — | Summary Coverage Map for TECHNICAL_SUMMARY (per Org-structure v1.24) | In progress (v3.3 has partial coverage; v3.4 will expand) |
| — | Delete pre-existing broken `test_aggregate_numba_matches_numpy` (superseded by T14) | Separate micro-task after 13.17.GB commit |
| — | Fix `feature_taxonomy.py` to discover `test_evaluator_lookup.py` | Separate micro-task; pre-existing Phase 13.16.GB gap |
| 12.15.GB | V4 shared kernel integration | Planned |

---

## Document History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | Dec 16, 2025 | Initial version |
| 2.0 | Dec 31, 2025 | Added Phases 12.14.GB, 12.14a.GB, incident analysis |
| 3.0 | Jan 4, 2026 | Added Phases 12.14b.GB, 12.14b.GB-addendum, 12.14c.GB |
| 4.0 | Feb 14, 2026 | Added Phases 13.7.GB–13.8.SW. Updated capability matrix (102 features) |
| 5.0 | Mar 26, 2026 | Added Phases 13.10.GB–13.15.GB. COG incident. 133 features, 500 tests. New functions: make_nonlinear_sliding_window_fit, make_sliding_window_aggregate, make_sliding_window_aggregate_parallel. Expression-based linear columns. WLS fix, fit_intercept fix, lean output, sigma cut. |
| 6.0 | Apr 7, 2026 | Phase 13.16.GB (evaluator lookup + scipy methods + per-dimension dict). P0 fit_intercept incident in SW numba (3 locations, 10 cross-fitter tests, governance failure mode #11 added). boundary='symmetric' incident documented (status disputed — architect requirement, commit message claims implementation, architect testimony says only 'full' works). Failure Modes Catalog added (7 entries). makeIterationFit0 performance reference (354s → 18s via per-sector loop). 517 tests, 3 pre-existing failures. |
| **6.1** | **Apr 9, 2026** | **Phase 13.16.GB-FIX2 retroactive entry** (commit `9e88eacd`, tag `PHASE_13_16_GB_FIX2_END`) — the FIX2 commit claimed this revision but did not actually produce it; v6.1 closes the gap. **Phase 13.17.GB in-progress entry** — Coder Claude22 active at time of this revision, proposal v1.3 APPROVED, commit expected within ~3.5 working days; v6.1a follow-up at commit time will fill in final test counts and outcome narrative. **Incident 4 status changed** from "disputed" to "partially resolved" (fit path confirmed working with 4 invariance tests; aggregate path confirmed broken and split out as Incident 7). **Incident 5 reserved** with placeholder explaining the cross-team numbering gap. **Incident 6 added** — F2 `method=dict` silently drops interpolation orders (fixed in FIX2, parameter-not-propagated class instance #4). **Incident 7 added** — F1 `boundary` silently dropped in `make_sliding_window_aggregate` (in-flight fix in 13.17.GB, parameter-not-propagated class instance #5). **Parameter-not-propagated bug class catalog updated** from 3 to 5 instances; earlier v1.3 proposal draft incorrectly stated "7 instances" (conflation of overall Incident numbers with class instance numbers), corrected here against FIX2 commit message line 81 evidence. **Governance Observations section added** documenting three paper-trail defects surfaced during FIX2 and 13.17.GB review cycles: (1) FIX2 commit-message claim not matching disk, (2) TECHNICAL_SUMMARY v3.3 528/529 arithmetic error carried through despite 3 reviewer flags, (3) fresh-reviewer Claude23 found the T14 cross-backend gap that 4 prior reviewers missed (validates MTTU v1.20 fresh-reviewer rotation discipline; supports the architect's consideration of adding 2 new GPT reviewers). **Failure Modes Catalog expanded** from 7 to 10 entries: #8 false-positive cross-backend test (calls same backend twice), #9 commit-message documentation claim without disk update, #10 multi-reviewer finding not absorbed into committed document. **Key Technical Decisions table expanded** with FIX2 (5 new rows) and 13.17.GB (7 new rows) sections. **Planned Phases updated:** 13.17.GB-MedianFix added as immediate follow-up; two micro-tasks (delete broken `test_aggregate_numba_matches_numpy`, fix `feature_taxonomy.py` discovery gap) added to the scheduled work queue. **Test count metadata corrected to canonical 529** throughout; the 528/529 discrepancy with TECHNICAL_SUMMARY v3.3 is explicitly noted in three places (header metadata, FIX2 phase section, Capability Matrix). **Drafted by Claude23 (GBAI Reviewer) at architect request 2026-04-09 during Phase 13.17.GB implementation. v6.1a revision at 13.17.GB commit time will close the in-flight placeholders.** |
| **6.2** | **Apr 18, 2026** | **Phase 13.19.GB-PERF landed.** V1/V2 recompute path routed through dense-lookup infrastructure. `_build_bin_index_map` (205s) + `_get_neighbor_bins` V3a (152s) replaced by `_assign_bin_ids_fast` + `_aggregate_window_dense`. **Incident 8 added** — `_fit_window_regression_numba` `fit_intercept` hardcoded True, found while implementing F1, parameter-not-propagated instance #8. **Incident 9 added** — V1/V2 recompute path silently ignores `boundary` parameter, pre-existing, parameter-not-propagated instance #9, deferred to Phase 13.XX.GB-BoundaryV1V2. 14 T1 invariance tests. Coder: Claude22. Reviewers: Claude20, Claude21, Claude23, Claude24, Claude25. |
