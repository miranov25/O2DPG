# [GPT30.dfdraw] [dfdraw] [CODER] [PHASE_13_103_DF / B4] [PRECOMMIT REVIEW REQUEST]

## B4 — Explicit `figsize` forwarding through scalar faceted `DFDraw.draw()`

**Date:** 2026-11-06
**Author / coder seat:** `GPT30.dfdraw`
**Status:** **ALMA2 RED → GREEN, CANONICAL REGRESSION GREEN; INDEPENDENT PRECOMMIT REVIEW REQUESTED; NOT COMMITTED**
**Controlling specification:** `PHASE_13_103_DF_BugFixes_Proposal_v0_2_GPT19_dfdraw_20261105.md`, §5 B4, approved by the architect.
**Reference starting commit:** `f58683b71da751e529ef89653198731174a0ca2c` (S8 implementation checkpoint).
**PHASE_13_103_DF_BEGIN:** present at `f58683b7`; **do not create END tag**.
**PHASE_13_82_DF:** OPEN and suspended; no S8 source changes permitted in B4.

## 0. Source-to-spec audit (GP-7)

| Approved requirement | Implemented or tested | Alma2 evidence |
|---|---|---|
| Forward explicit `(16,10)` on scalar `profile` ordinary facet, 1-D/2-D | `drawer.py` generic dispatch, existing typed facet owner | B4-01, 2 passing cases |
| Forward explicit `(16,10)` on scalar `hist` ordinary facet, 1-D/2-D | Same isolated dispatch guard | B4-01, 2 passing cases |
| Forward explicit `(16,10)` on scalar `scatter` ordinary facet, 1-D/2-D | Same isolated dispatch guard | B4-01, 2 passing cases |
| Grouped scalar profile follows explicit dimensions | Existing group-by facet rendering | B4-03, 2 passing cases |
| Omitted `figsize` preserves historical automatic sizes | Compare generic `draw()` against existing typed routes | B4-02, 6 passing cases |
| Direct typed `profile`/`hist`/`scatter` remain correct | Existing typed owner unchanged | B4-04, 6 passing cases |
| Vector-profile facet route remains correct | Existing dedicated vector path unchanged | B4-05, 2 passing cases |
| `figure_margins` and `subplot_spacing` remain authoritative | Check six `subplotpars` numbers | B4-06, 2 passing cases |
| Public `AliasDataFrame.draw()` matches `DFDraw.draw()` | Actual installed ADF on Alma2 | B4-07, 2 passing cases, **no skip** |
| Numerical/statistical behavior unchanged | No numerical code changed; full regression | 1,924 passed, 0 failed |
| Capability Matrix registration is complete | New `INTEGRATION.explicit_facet_figure_geometry` taxonomy + 26 classified nodes | Matrix: 196 features, 108 verified, 0 broken |

**Scope limit:** B4 corrects explicit figure-size forwarding for *ordinary faceted scalar* profile/hist/scatter. It does not claim to fix every original RootInteractive PDF annotation or footer collision; that is distinct layout work. It does not change nonfaceted `figsize`, 2-D histogram or hexbin dispatch, S8 state, or other PHASE_13_103 slices. Regenerating the original RootInteractive real-data PDF is desirable for the upstream integration audit but is not part of the B4 numerical/geometry synthetic test claim.

## 1. Exact changed source paths and postimages

The candidate comprises **exactly four** repository-relative source/test paths:

| Path | MD5 | SHA-256 |
|---|---|---|
| `drawer.py` | `687349190d7bf68b516b86c022dcce43` | `5ee60b4319656d8b1574fba41c987e829bc7fbe476f7a2504a7cc865c3cb42d2` |
| `tests/feature_taxonomy.py` | `66a0ba1b0b12cde4685bc38f207a24d2` | `29fbd9a472a80daea6ac89a3700df9f8a2336f51fe742e6da17f63110938b3c7` |
| `tests/test_layer_classification.py` | `c1342614a5d2f0e5913a60e7e3c8c34b` | `fd44697dd2fdfd181699a4ed63f03e45b78515bfda95dbc3b4002153f33881a7` |
| `tests/test_phase_13_103_b4_figsize_forwarding_red.py` | `669ff828bd9d0cebe13e0efdf5f52c59` | `41f9b409451da31eb7cfed01ce5dbfeec366a2bb2f7040b529ed2ee9cdbd3701` |

Implementation patch: `PHASE_13_103_DF_B4_Figsize_Forwarding_Candidate_v0_1_GPT30_20261105.patch`; MD5 `7c09f21fcd2a83cdd2a025e8a1350aa5`; SHA-256 `28ca8134a2b62a7c7da713278a486f82accbdca4aa9b187b9910039470bde145`. It edits `drawer.py` and the two registry files. The new RED/GREEN test module was installed separately and is byte-identified above. Full four-file diff is present in the reviewer ZIP's `test_logs/diff_to_phase_20261106_010454.txt` alongside a generated documentation diff. For a cleaner independent review, the exact four source/test diff sections were extracted **without alteration** as `PHASE_13_103_DF_B4_Figsize_Forwarding_PRECOMMIT_FourFile_Diff_v0_3_GPT30_20261106.patch` (MD5 `953d1a65c940cfafff1569041a758621`, SHA-256 `01f9b77a0b01be0dbffa8afe6bd733b9e33feda7b9a90f3964d9cd3eb96a36d0`); `git apply --reverse --check` against the locally reconstructed postimage passed. This reviewer diff is not a new implementation and must not be reapplied on Alma2.

**Implementation mechanism:** `DFDraw.draw()` binds `figsize` as a named parameter; thus scalar typed delegates cannot see it through `**kwargs`. A single `_scalar_facet_size` dictionary forwards it to scalar `hist`, `scatter` and `profile` only when `facet_by is not None` and `figsize is not None`. The typed facet renderer still owns Figure creation. This avoids changing existing omitted-size defaults and numerical semantics.

## 2. Alma2 commissioning record — all gates passed

| Gate | Command/result | Saved evidence |
|---|---|---|
| STEP25 | Source hashes verified; `PHASE_13_103_DF_BEGIN` created at `f58683b7`; RED module installed | Operator terminal output, 2026-11-06 00:58 |
| STEP26 RED | **12 failed / 14 passed**, expected: six explicit scalar cases, two grouped profile cases, two margins cases and two ADF parity cases; original implementation unchanged | `test_logs/PHASE_13_103_B4_RED_20261106_005831.log` |
| STEP27 PATCH | Dry run PASS; application PASS; all four exact source/test hashes PASS; whitespace PASS | `test_logs/PHASE_13_103_B4_PATCH_20261106_010233.log` |
| STEP28 focused GREEN | **30 passed, 0 failed, 0 skipped**, including installed ADF parity tests and Capability Matrix meta-tests | `test_logs/PHASE_13_103_B4_GREEN_20261106_010246.log` |
| STEP29 canonical | **1,924 passed, 0 failed, 0 errors, 2 skipped, 9 xfailed, 78 warnings**, exit 0 | `test_logs/PHASE_13_103_B4_STEP29_20261106_010454.log` |
| STEP29 reviewer archive | Nine entries; `unzip -t` PASS; `test_failures` file zero bytes | `test_logs/reviewer_20261106_010454.zip` |

**Verified ZIP identity:** MD5 `45244edec5fdee94133a020aa55408c8`; SHA-256 `c9696cf26965dc96e09c81dca87c9ca83a9dc25c320609fcd2bb644c93b6390a`.

**Runner summary:** Python 3.10.19 on Linux aarch64, pytest 7.2.2, 12 workers, HEAD `f58683b7`, 1,935 collected cases. The canonical summary records 1,924 passed, two skipped, nine expected failures, and zero errors or ordinary failures. The full runner reported `All tests passed` and emitted a valid reviewer archive.

**Capability Matrix:** 196 features; 108 Verified, 86 Smoke-only, two Planned, zero Broken; 1,290 proof tests, 704 invariance tests, 94 visual tests. These are cumulative project totals, not 196 new B4 features.

## 3. Exact reviewer evidence and boundaries

The *original, unmodified* canonical ZIP includes:

1. `test_logs/SUMMARY_20261106_010454.txt`
2. `test_logs/test_failures_20261106_010454.log` (empty)
3. `test_logs/test_full_20261106_010454.log`
4. `test_logs/CAPABILITY_MATRIX_20261106_010454.md`
5. `test_logs/diff_last_commit_20261106_010454.txt`
6. `test_logs/diff_to_phase_20261106_010454.txt`
7. `test_logs/git_status_20261106_010454.txt`
8. `docs/CAPABILITY_MATRIX.md`
9. `docs/CAPABILITY_MATRIX.html`

The ZIP status confirms exactly the four B4 paths in source/test scope: three modified tracked files plus the new staged test module. Separately, generated `docs/CAPABILITY_MATRIX.md` is modified (not staged), and several unrelated untracked documentation/environment files exist. **Do not accidentally include them in the B4 source commit.** The runner's suggestion to include the generated matrix at *phase* commit may be handled as a separate, explicitly reviewed documentation decision; it is not authorization to broaden this four-file B4 patch. The runner also reports a nonblocking PHASE_HISTORY tag-drift warning for a placeholder `PHASE_X_END`; not caused by B4.

There is no evidence of a change to `partitioned_state.py` or `statistical_state.py` in this B4 worktree. PHASE_13_82 remains OPEN; its separately banked payload-axis type-restoration review finding remains outside this repair.

## 4. Questions to independent PRECOMMIT reviewers

Please provide **APPROVE / APPROVE WITH CONDITIONS / HOLD**, with line-specific findings and explicit severity, addressing:

1. Is guarded forwarding in `DFDraw.draw()` the correct first owner, without crossing into typed rendering or ADF code?
2. Does `facet_by is not None` correctly bound ordinary scalar-facet support? Does it improperly exclude a supported legacy `facet=True` path, or is that a distinct contract?
3. Does conditional forwarding preserve preexisting nonfaceted and omitted-`figsize` behavior, direct typed methods, vector-profile routing, and user margins/spacing?
4. Do 26 B4 tests (including actual Alma2 ADF parity) establish the specified geometry regression without silently changing statistics, alias normalization, fit metadata or figure ownership?
5. Are the tests registered correctly in taxonomy/classification and are there any newly invented invariance claims?
6. Is the four-source-file commit scope correct, with generated Capability Matrix and unrelated files excluded, while retaining the original ZIP as permanent `test_logs/` evidence?
7. Are any issues blocking a narrowly scoped B4 commit, as distinct from the later B2/B3 layout defects and original RootInteractive PDF regeneration?

Request **one independent combined PRECOMMIT review round**. Do not ask separate reviewers to change implementation before reporting their findings. Do not regenerate the ZIP or rerun canonical tests unless a source byte changes or a reviewer raises a reproducible blocking defect.

## 5. Commissioning recommendation

**Coder status: READY FOR INDEPENDENT B4 PRECOMMIT REVIEW.** All prescribed RED, focused GREEN, public ADF and canonical gates passed on Alma2 against the exact candidate. **This CRR is not an independent approval and does not authorize a commit by itself.** After independent reviewer authorization, use `git commit --only -F - -- drawer.py tests/feature_taxonomy.py tests/test_layer_classification.py tests/test_phase_13_103_b4_figsize_forwarding_red.py` with a documented B4 message. Keep PHASE_13_103 open for B5-A, B1, B2, B3 and B5-B, and PHASE_13_82 OPEN and suspended.

**Signed:** `GPT30.dfdraw` — implementation/coding seat, 2026-11-06.
