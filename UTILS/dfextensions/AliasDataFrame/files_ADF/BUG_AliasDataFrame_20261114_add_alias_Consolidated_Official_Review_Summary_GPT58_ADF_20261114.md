# [GPT58.ADF] [AliasDataFrame] [Main Reviewer] [BUG_AliasDataFrame_20261114] [!] APPROVED_WITH_COMMENTS — immediate independent-coder hotfix

**Review date:** 2026-11-14
**Reviewer / consolidation:** `GPT58.ADF`
**Bug owner:** AliasDataFrame; originating reporter `Claude2:dEdxAI`
**Implementation reviewed:** proposal only, **not** an actual candidate code diff/commit.
**Decision:** **APPROVE a separate coder implementing the minimal `add_alias` performance correction from the latest verified ADF snapshot; PRECOMMIT remains mandatory.** No production code was modified by this review.

## 0. Executive conclusion and authorization boundary

The four ADF reviews agree that the defect is real and urgent. `_resolve_dependencies` scans all existing alias expressions; for each token it reads `self.aliases`, a property that reconstructs the entire alias mapping. With N aliases and average T tokens, the dependency scan thus costs approximately O(N² T) *per add*. A local alias-mapping snapshot eliminates the extra factor of N, approximately O(N T) per add, retaining a full graph/cycle check. The reporter's 1,600-alias benchmark improved from 34.7 s to 0.21 s (~165×). These are reporter values; Claude50 independently reproduced the scaling and found 32.574 s → 0.077 s on a separate two-CPU environment.

**APPROVED NOW:** one local mapping snapshot in `_resolve_dependencies`, one focused, deterministic regression test, measurements and exact-diff evidence.
**NOT APPROVED:** global alias caching, incremental dependency graph, unrelated parser changes, other performance fixes, rollback-semantics repair, changes to dfdraw, or incorporating production code into test-only `PHASE_13_110_ADFDF`.
**NO MERGE AUTHORIZATION YET:** second coder must supply an actual PRECOMMIT package for independent code review.

## 1. Review coverage and evidence ranking

| Reviewer | Vote | Source verification | Independent execution/evidence | Disposition |
|---|---|---|---|---|
| `Claude50.ADF` | `[!] APPROVED_WITH_COMMENTS` | Exact `AliasDataFrame.py` MD5 `2b01116110c1925032abd40fe5100bb0`, reported `f248455` bytes | Original reproducer, patched reproducer, dependency equivalence, lazy-load probe, property-read mutant and 4,096-test comparison | Strongest integrated quantitative validation; identified follow-up defects |
| `GPT56.ADF` | APPROVE immediate bounded fix | Exact MD5-matched source; owner functions and call chain | Extracted old/new `_resolve_dependencies` mappings for 3, 4, 50, 100 aliases; 100-alias 299→1 property accesses | Strong independent code-path and negative-control evidence |
| `GPT57.ADF` | APPROVE immediate separate hotfix | Earlier Nov-6 source snapshot, not exact active Nov-14 tree | No independent current benchmark; checked algorithm and proposed acceptance gates | Sound independent scope review |
| `GPT58.ADF` | APPROVE targeted priority bugfix | Earlier ADF snapshot, relevant function bodies, not exact Nov-14 active tree | Analysis, not a new ROOT/benchmark execution | Concurs; require minimal patch and current target fingerprint |

**Consensus:** 4/4 approve implementation of the bounded fix; no reviewer approves uninspected production bytes or bypassing PRECOMMIT tests. Evidence strength varies by source and test access; do not report the four votes as four independent reproductions of the full benchmark.

### Attached executable evidence inspected for consolidation

All four attached `.py` files parsed successfully as Python source (AST check), **but were not executed against a live ADF installation during this consolidation**.

| Attached file | MD5 | Intended role | Qualification |
|---|---|---|---|
| `repro_add_alias_scaling.py` | `5e5be911123e5f8385f53cb4923ec464` | One `add_alias` after N prepopulated aliases, 39-term source-like expression | Baseline/candidate identical machine; diagnostic benchmark, not portable hard timing gate |
| `probe_bug_20261114_scaling.py` | `896656d96cec0882c83139d83348caaf` | Alias-property access, cumulative registration, lazy load, materialization profile | Its `lazyread` mode hardcodes `/home/claude/work`; substitute a valid scratch directory on alma2 before use |
| `probe_bug_20261114_equiv.py` | `bde2842165028a7185fb13aba7fa7437` | 300 acyclic + 100 cyclic generated frames, 200 alias-add sequences, subframes/redefinitions; JSON comparison | Run old/new with the **same** `PYTHONHASHSEED`; compare output files byte-for-byte |
| `test_BUG_20261114_add_alias_cycle_check_scaling.py` | `9c043fa55ebb62ebec87a5fffab7ba96` | Deterministic count of alias-property reads, chain/subframe mapping, indirect-cycle rejection | Negative control must be RED old and GREEN fixed, with assertion failure rather than import/setup failure |

## 2. Consolidated findings

| Finding | Priority | Status | Required action |
|---|---|---|---|
| `P1-01` repeated mapping rebuild in `_resolve_dependencies` | **P1, blocking actual dEdx calibration** | Confirmed by source and independent performance evidence | **FIX NOW:** local snapshot, replace repeated membership access |
| `F-1` lazy read also repeatedly registers aliases | P1 impact extension | Claude50 measured 44.78 s → 0.65 s for 200 aliases | Covered automatically by `P1-01`; add it to impact/evidence, no loader change |
| `F-2` remaining O(N²T) total sequential alias registration after patch | P2 performance follow-up | Measured by Claude50 (1,600 aliases: 37.34 s cumulative fixed) | Do not expand hotfix; separate benchmarking and potential bulk-load design |
| `F-3` rejected cycle/redefinition not rolled back | P2 **separate correctness bug** | Claude50 reproduced invalid state after cycle refusal | New issue and explicit Architect decision; not in this patch |
| `F-4` strict wall-time CI thresholds are flaky | P2 test design | 800→1600 fixed timings do not obey hard 2× rule | Use deterministic property-read test in CI; benchmarks informational with medians |
| `F-5` global cache/incremental cycle search | P3 architectural option | Not required, broader state-invalidation risk | Defer; do not introduce new state in this repair |
| `F-6` costly repeated alias reads in `materialize_alias` / `get_alias_series` | P3 follow-up | Claude50's 800-alias materialization profile: 15.4 s property time of 39.4 s | Separate measured hot-loop audit and changes later |

**Important scientific/safety qualifier:** the current patch preserves the existing rejection behavior, *including* the pre-existing rollback defect F-3. Fixing F-3 in the same commit would be an unreviewed semantic change and impair minimal-risk regression attribution.

## 3. Exact recommended patch scope

Base the patch on **the latest checked ADF product bytes**, not blindly on historical commit `f248455`. If that commit's `AliasDataFrame.py` is still present, the historical MD5 is `2b01116110c1925032abd40fe5100bb0` (as independently confirmed by Claude50 and GPT56). If it differs, inspect the current `_resolve_dependencies` before changing it. If another coder already eliminated the repeated lookup, report the bug already corrected, rather than add a redundant patch.

Minimal conceptual diff:

```diff
- for name, expr in self.aliases.items():
+ aliases = self.aliases
+ for name, expr in aliases.items():
       # Preserve existing subframe handling and re.findall token loop.
-      if token in self.aliases:
+      if token in aliases:
           dependencies[name].add(token)
```

No new helper, no token `set(...)` conversion necessary, no extra `dict(...)` copy necessary if the property already yields a fresh read-only mapping, and no changes to cycle algorithm/exception messages.

The exact diff may contain alignment/indentation and a short comment; the coder must demonstrate equivalent behavior rather than rely on this abstract diff as a mechanically applicable patch.

## 4. Separate-coder plan; concurrency and merge safety

**YES, assign a second coder now.** Let that coder work in an **isolated Git worktree/branch or independent clone** based on the most recent verified committed target ref. Obtain the **latest working-tree snapshot** as well if the main coder has uncommitted changes that may affect `AliasDataFrame.py`; a snapshot is evidence, not permission to overwrite the live worktree. Record current HEAD, branch, scoped dirty paths and MD5/SHA-256 of the actual source. If the main coder is editing the same file, the second coder prepares a small patch and review bundle; integration owner rebases/cherry-picks after checking conflicts. Never copy a historic whole module on top of the shared tree.

**Authorized changed files:** `UTILS/dfextensions/AliasDataFrame/AliasDataFrame.py`, one focused regression test under existing `tests/`, and an optional concise bug-status documentation update. No changes to unrelated ADF/dfdraw implementation, scripts, old oracle semantics, capability matrix, or default settings. All test logs go to the established `test_logs/` directory.

**Required coder package:**

1. **BASELINE:** latest commit/snapshot provenance, scoped status, exact original module hash and a narrowly scoped old→new diff. Note any concurrent work or conflicts.
2. **NEGATIVE CONTROL:** run the deterministic `test_add_alias_reads_aliases_a_constant_number_of_times` on old bytes; demonstrate a **meaningful assertion failure** (Claude50 reported 843 vs 8,043 alias property reads at N=20/200). Run on candidate; expect constant reads (Claude50 reported 3/3).
3. **EQUIVALENCE:** compare old/new dependency maps and cycle/topological/add_alias results for chained aliases, repeated tokens, dotted subframes, self- and indirect cycles, redefinitions and invalidation. Use reproducible equivalence JSON probe and focused pytest; preserve exact expected exception behavior on current baseline.
4. **BENCHMARK:** measure one `add_alias` with N=200/400/800/1600 and actual dEdxAI 84-alias Taylor registration on the same pinned environment; report medians, relative scaling and approximate speedup. **Do not** gate CI on `<0.5 s` or a hard ≤2× ratio across machines.
5. **REGRESSION:** run ADF alias/cycle/subframe/invalidation focused tests and authorized canonical `run_tests.sh` and gallery checks on alma2 with ROOT. Compare failure **identities**, not merely pass totals; old pre-existing reds are not called GREEN. No repeated full run for each small iteration.
6. **PRECOMMIT:** supply self-contained CRR (what changed, source hashes, baseline/candidate timings, old RED/new GREEN, equivalence, focused/full outcomes, untouched paths), original logs and code diff. Obtain independent short PRECOMMIT acceptance before integration and commit.
7. **INTEGRATION / S0:** if `PHASE_13_110_ADFDF` S0 has already pinned ADF product hashes, explicitly re-ratify the new imported module fingerprint; if not, freeze the post-fix bytes at S0. Do not install production changes *as part of* 13.110's test-only implementation.

### Recommended message to the second coder

> Implement `BUG_AliasDataFrame_20261114` as a **standalone P1 ADF performance hotfix** from the latest verified ADF snapshot. First record the active `AliasDataFrame.py` hash, source commit, scoped worktree status and whether the main coder has conflicting edits. Work independently, changing only `_resolve_dependencies()` to read `self.aliases` once and reuse that snapshot; keep tokenization and cycle semantics unchanged. Include the supplied property-access regression test and old RED/new GREEN, dependency-graph equivalence, performance comparison including the real dEdx 84-alias scenario, and focused/full ADF regression evidence. Do not fix cyclic rollback, `materialize_alias`, `get_alias_series`, loaders or dfdraw in this commit. Deliver the exact-source PRECOMMIT CRR/diff/logs for independent review before anyone installs or merges your candidate.

## 5. Final consolidated vote

**`[!] APPROVED_WITH_COMMENTS — IMPLEMENT IMMEDIATELY IN ISOLATED SECOND-CODER BRANCH.`** The bug is confirmed; the fix is small and not architecturally ambiguous. The main coder may continue existing work. **The second coder may begin now without another proposal round.** PRECOMMIT/merge approval remains conditional on the actual candidate diff, red/green proof and normal ADF regression gates. Additional performance/correctness issues from Claude50's report should become separate backlog items, not requirements for this hotfix.

**Signed: `GPT58.ADF` — AliasDataFrame Main Reviewer, 2026-11-14.**
