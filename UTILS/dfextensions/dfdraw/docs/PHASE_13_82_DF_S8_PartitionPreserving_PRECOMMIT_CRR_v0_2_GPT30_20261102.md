# PHASE_13_82_DF — S8 Partition-Preserving State + M14 Completion
## Official six-file PRECOMMIT Code Review Request (CRR) v0.2

**Author / coder seat:** `GPT30.dfdraw`
**Date:** 2026-11-02
**Status:** **ALMA2 GREEN; INDEPENDENT PRECOMMIT REVIEW REQUESTED; NOT COMMITTED**
**Phase:** `PHASE_13_82_DF` remains OPEN; no END tag authorized.
**Committed base:** `fd01ccb08b9c57dd3fcec72b9f4518f9812b2d0a` (S7-M14).
**Change control:** exact six-file candidate only; patch MD5 `302917de80b3456b76518770377e0226`.
**Earlier Stage-B CRR:** `PHASE_13_82_DF_S8_PartitionPreserving_Candidate_STAGEB_CRR_v0_1_GPT30_20261102.md` (commissioning plan; status superseded by this current evidence-bearing CRR).
**Architect decision:** preserve *explicit user-declared scientific category order* as authoritative; no separate S7/M14 architecture review and no S9 collapse in S8.

---

## 0. First-page decision interface

**Review requested:** `APPROVE`, `APPROVE_WITH_CONDITIONS`, or `REQUEST_CHANGES` **for the unchanged six-file S8 patch**, with explicit conditions and precise source/test references. Do not vote to close the entire phase.

**Operator-observed evidence:**

| Gate | Result | Evidence |
|---|---|---|
| STEP19, exact-byte patch installation on Alma2 | six destination files `OK`; dry-run PASS; patch applied | Operator console, 2026-11-02 17:40 |
| STEP20, focused/meta | **117 passed, exit 0**, 2.34 s | `test_logs/phase_13_82_s8_step20_20261102_174113.log`, operator summary |
| STEP21, canonical `./run_tests.sh` | **1,898 passed, 0 failed, 0 errors, 2 skipped, 9 xfailed, exit 0** | `test_logs/reviewer_20261102_175335.zip`, original unmodified runner archive |
| Capability Matrix | 195 total / 107 Verified / 86 Smoke-only / 0 Broken / 2 Planned; 1,264 proof tests; 698 invariance tests | ZIP's `CAPABILITY_MATRIX_20261102_175335.md` and `SUMMARY` |
| Reviewed source commit | HEAD **still S7** `fd01ccb0`; S8 candidate uncommitted | ZIP `git_status_20261102_175335.txt` |

The canonical reviewer ZIP has **nine entries** and passes ZIP integrity verification. It contains the full run log, zero-byte failure log, Capability Matrix, status and diffs. The summary's `PHASE_X_END` tag-history warning concerns an unrelated template/history entry and is nonblocking; it does **not** authorize creation of any phase-END tag. No extra test run is requested for unchanged bytes.

**Independent review focus:** Whether the new partitioned object is scientifically safe and compatible, especially identity, ordering, payload restoration and refusal of already pooled inputs. Passing tests alone is not a substitute for source-level inspection.

## 1. Scientific purpose and user decisions

Partition-preserving aggregation is distinct from exact pooling. If source A and source B each hold weighted per-coordinate sufficient statistics, S8 stores both statistical leaves separately and exposes a combined, queryable `partition_id × x_bin × categories` view. **No moments are summed across partitions at S8.** S9 will separately implement explicit collapse/pooling and independently validate its arithmetic.

S8 completes three bounded M14 follow-ups, *without reopening the accepted S7 commit*:

1. **M14-A, signed zero:** numeric float categorical `-0.0` normalizes to `+0.0` in new builds and restored nodes. A restored domain explicitly listing both signed zeros is refused as ambiguous.
2. **M14-B, exact types:** six S7 round-trip tests now assert actual types for Boolean, integer and floating categories rather than Python's cross-type value equality.
3. **M14-C, category order:** an explicitly declared scientific domain/order takes precedence, including currently unoccupied categories. Ordered pandas categoricals and `category_orders` are supported. Conflicting explicit orders are refused. With no declared scientific order, S8 computes a deterministic homogeneous scalar category union. Original order remains available on each independent source leaf. **Legacy `merge_exact` left/input ordering is not changed.**

ADF/pandas dtype metadata may convey a trustworthy declared dtype, but upstream conversion may already have erased distinctions; S8 does not claim to recover lost original types or integrate directly into ADF dispatch.

### Concrete scientific example

```python
from dfextensions.dfdraw import statistical_state as ss
from dfextensions.dfdraw.partitioned_state import preserve_partitions

order = ['IROC', 'OROC1', 'OROC2', 'OROC3']
a = ss.build_histogram_state([0.2], edges=[0., 1.], name='residual',
    partition_id='run-A', categories={'region': ['OROC2']},
    category_orders={'region': order})
b = ss.build_histogram_state([0.3], edges=[0., 1.], name='residual',
    partition_id='run-B', categories={'region': ['OROC1']},
    category_orders={'region': order})
result = preserve_partitions([a, b])
assert result.axes['region']['values'] == order
assert result.partition_ids == ('run-A', 'run-B')
# Each original StatisticalNode is recoverable by get_partition(...).
# No result.collapse_partitions() is implemented in S8.
```

## 2. Binding six-file patch and byte fingerprints

**Patch filename:** `PHASE_13_82_DF_S8_PartitionPreserving_CategoricalOrder_Candidate_v0_1_GPT30_20261102.patch`
MD5 **`302917de80b3456b76518770377e0226`**
SHA-256 **`227c0a2ddeb90c1cdac5b5e064152a72024b1d038f09d9dde5ba8e227d05d155`**
Applied by operator after clean `patch --dry-run -p1` on S7 base; all six destination checks reported `OK`.

| Authorized path | Source change | POST MD5 | POST SHA-256 |
|---|---|---|---|
| `statistical_state.py` | Normalize signed zero, accept declared ordered domain | `a2afd3db1a9d01dee5fe1341c13f8388` | `af87720f0afcaf93ad89a96c2bff48f4f536a9e0a468c81de585ea1861d27da0` |
| `partitioned_state.py` **NEW** | Partition-preserving `PartitionedStatisticalState`, nested flatten, ordered aggregate view, payload | `ea336c028855804209cc99eb6e812dd7` | `ef9baa3b5e0a56a257f50041a00fc5c3825fd5a68f3679b804a404af61e2b61f` |
| `tests/test_phase_13_82_m14_category_identity_red.py` | Type-strict S7 round trips | `9666ef731adaa0a7630c4f39de74f5cd` | `f8f1e6ead194d5da51838b04fd6aa1f9864ea87f271736e92f82e1b16d8b0efb` |
| `tests/test_phase_13_82_s8_partition_preserve.py` **NEW** | 29 S8 tests | `4a97df3d97075b45d5ae63c13484ccf8` | `4573a29c59af69853baf0e1be32716c74da095e3072f040e508d74f44b67717f` |
| `tests/feature_taxonomy.py` | Two S8 capability features and mappings | `2a0e225d0abff3a6c057e3aee12756e2` | `b52f894fa42109cc6dfe39176170ec19478b64f882384212cad1861637cc9e2f` |
| `tests/test_layer_classification.py` | S8 smoke/integration/invariance mapping | `6368c04db17c767a9067b0327b6f6340` | `b05c3aacded2b087b28dabecad799aab567512e84854f66d8b1b6ecaf228dc94` |

The ZIP's Git-status snapshot lists only these six authorized source/test paths as relevant S8 changes; the two new files were **staged for runner evidence, not committed**, while four preexisting files were modified but unstaged. Generated `docs/CAPABILITY_MATRIX.md/.html`, working-style documents, images, `gitlog.txt`, `scripts/env.sh` and any unrelated changes **must not** be included in S8's explicit-path commit. Do not confuse the ZIP's cumulative `diff_to_phase` with the six-file S8 patch.

## 3. Implementation contracts to inspect in source

1. `partitioned_state.PartitionedStatisticalState`: acceptance of multiple independent `StatisticalNode`s, deterministic partition-identity ordering, unmodified source leaf state and provenance, copy-on-access semantics, computed sparse combined table.
2. `partitioned_state.preserve_partitions`: nested partition-preserving composition flattened without statistical pooling; refuse repeat `partition_id`s and nodes whose provenance reports `contributing_partition_ids` from previous pooled `merge_exact`.
3. `partitioned_state._canonical_axes`: physical edges/statistic kind/schema/semantic compatibility; entire explicit ordered domains including zero-observation values; distinct types not coalesced; conflicting explicit scientific orders refused; deterministic unordered canonical scalar ordering.
4. `statistical_state`: normalization `-0.0 → +0.0` on accepted categorical values and restored axes/state, with ambiguous restored dual-zero domain refusal. Impact on previously serialized valid v1 payloads must be examined; **leaf schema remains v1**, partitioned logical payload has separate `partitioned_schema_version=1`.
5. `tests/test_phase_13_82_s8_partition_preserve.py`: numerical truth for separately retained weighted moments, including signed and zero weights; distinct partition identities; readback and mutation isolation; explicit ordering; preexisting `merge_exact` ordering remains unchanged.
6. `tests/feature_taxonomy.py` plus `tests/test_layer_classification.py`: 29 S8 cases represented once and correctly classified across `STATS.s8_partition_preserving_state` and `STATS.s8_category_order_and_signed_zero`; four true independent/numerical invariance cases, rather than claiming 29 independent numerical proofs.

### Known boundaries — not a hidden deferral

- No S9 `collapse_partitions`, no durable persistence backend, no recursive `StatisticalResult`, no general physical N-D bins, no `adf.draw` integration, no executor replacement.
- `partition_id` is an internally validated declared scientific source identity, **not** cryptographic authentication of external data origins.
- ADF/pandas may have coerced types before reaching S8; that lost information is unrecoverable.
- The combined `state` is a derived dataframe view, not an additional authoritative pooled accumulator.
- Per-leaf order is preserved; a single shared aggregate cannot retain conflicting user-declared scientific orders without explicit representation or refusal.
- Existing `merge_exact` legacy order remains input-dependent; S8 does **not** change it silently.

## 4. Key reviewer questions — request recommendations, not merely votes

**R8-01 — Source/provenance and duplicate prevention.** Inspect the actual source to determine whether `partition_id`, nested flattening and contributor provenance prevent double counting and unintentional recovery of pooled leaves. Are there malformed accepted inputs or identity aliases not covered by oracles? Distinguish internal integrity from external truth/authenticity.

**R8-02 — Explicit ordering preservation.** Confirm that ordered ADF/pandas metadata or `category_orders` survive construction, partition preservation, logical round-trip and unoccupied labels; that conflicting explicit order declarations are rejected; and that unordered union is merge-order independent. If a *partial explicit order* union would be better than strict identical declared domains, specify its scientific consequences; do not alter already approved ordering without an architect choice.

**R8-03 — M14 signed-zero/type compatibility.** Probe signed-zero normalization and refusal for both builder and restored payload, especially mixed unused coordinates, signed-zero row ordering, float labels and schema-v1 compatibility. Are there remaining collisions requiring a fix before this exact commit?

**R8-04 — Statistical numerical truth.** Verify that S8 never pools source sums; independently inspect per-partition `entries`, Σw, Σw², and profile Σwy/Σwy² against directly computed raw inputs, including signed-weight cancellation/zero-weight occupied cells. Does preserved sparse occupancy remain unambiguous?

**R8-05 — Payload/immutability/scale.** Assess `to_payload`/`from_payload` refusal of payload tampering, category-axis mismatch, duplicate partitions and previously pooled inputs. Consider CPU/memory implications of deep copies and repeated computed `state` at O(thousands) of partitions, with concrete recommendation if necessary.

**R8-06 — Scope and regression.** Do these exact six source/test changes preserve S7, P1, existing `merge_exact`, and nonstatistical dfdraw behavior? Is the 1,898/0 canonical gate adequate for this bounded S8 slice? Identify any source-level concern, not already-covered feature wishes for S9.

**R8-07 — Follow-up advice.** List only required follow-ups with priority: **must fix before S8 commit**, **can defer until S9**, or **out of phase**. Keep review overhead proportional to scientific risk. Propose any genuinely better alternative with user-visible consequences.

## 5. Four-file independent reviewer packet (flat `files_dfdraw/`)

1. **This current CRR** — `PHASE_13_82_DF_S8_PartitionPreserving_PRECOMMIT_CRR_v0_2_GPT30_20261102.md`.
2. **Exact S8 patch** — `PHASE_13_82_DF_S8_PartitionPreserving_CategoricalOrder_Candidate_v0_1_GPT30_20261102.patch`.
3. **Original canonical runner ZIP** — `reviewer_20261102_175335.zip` (nine integrity-verified entries, MD5 `aea80f642412999ffc2bf7b17075beee`, SHA-256 `998e6f330bd215cce9c9bb1284e0e4be702e4501e305483e9a3059b320ea3a3b`). Original is retained under canonical `test_logs/` and copied byte-for-byte to `files_dfdraw/` for distribution.
4. **Immediately preceding consolidated S7 review** — `PHASE_13_82_DF_S7_v03_Code_Review_Summary_GPT29_20261101.md` (canonical name for the received `(1)` copy; MD5 `9adb65bc34c5815965c4492faa075f85`, SHA-256 `31f5bee520326ca9a45288d1f76d0962dcdd7e7b32d5ad08ec4c59764a34febf`). This records the five independent S7 decisions and the M14 A/B/C follow-ups now included in S8.

**Why no PDF:** S8 is a statistics-only, non-rendering slice. No new real-data multipage plot/PDF evidence was produced; the canonical ZIP and numerical oracles are the relevant evidence. The preceding S7 summary provides review continuity without extra uploads. The scientific Stage-A CRR and earlier Stage-B plan are named in this document, but are not required as separate reviewer uploads.

## 6. Approval and commit custody

Require a source-read declaration and an explicit `APPROVE`, `APPROVE_WITH_CONDITIONS`, or `REQUEST_CHANGES` decision for this exact six-file candidate, including any mandatory conditions. On aggregate reviewer approval, operator must recheck all six live postimage hashes, S7 ancestry, phase-open status and scoped whitespace/diff; then use `git commit --only -F - --` with the six exact paths and a human-readable detailed message. No re-run of the full suite merely for evidence repackaging. **Phase 13.82 remains OPEN; no END tag.**

**Signed:** `GPT30.dfdraw` — S8 implementation author / PRECOMMIT CRR preparer, 2026-11-02. **No independent reviewer approval is claimed in this document.**
