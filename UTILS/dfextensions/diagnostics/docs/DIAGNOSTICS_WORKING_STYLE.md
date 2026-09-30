# Diagnostics Working Style

**Scope:** shared working conventions for PHASE_13_74_ADF diagnostics coders, reviewers, and the Architect.

This document complements the organization governance and the PHASE_13_74_ADF normative proposal.
If governance or a ratified architect decision is stricter than this document, the stricter rule wins.

This file is the single current canonical working-style document under
`$DIAGNOSTICS_CODE/docs/`. Historical or dated variants belong under `$DIAGNOSTICS_DOCS`.

---

## 1. Canonical environment

Load the canonical diagnostics environment before substantial work:

```bash
source "$O2DPG/UTILS/dfextensions/diagnostics/scripts/env.sh"
```

Canonical variables:

```text
$DFEXT_CODE
    shared dfextensions root

$DIAGNOSTICS_CODE
    diagnostics source/code root
    default: $O2DPG/UTILS/dfextensions/diagnostics

$DIAGNOSTICS_DOCS
    PHASE proposals, CRRs, review prompts, review summaries, closure records
    default: $NOTES/JIRA/O2-6532/docs/diagnostics

$DIAGNOSTICS_DATA
    generated monitoring data / large runtime evidence
    default: $NOTESData/JIRA/O2-6532/diagnostics

$DIAGNOSTICS_FILES
    temporary reviewer-distribution / packaging workspace
    NEVER COMMIT

$DIAGNOSTICS_INSTALL_BY_MD5
    shared exact-byte installation helper
```

Do not silently create a second set of project roots in ad-hoc scripts.

---

## 2. Communication and copy/paste format

Keep answers concise and operational. Prefer short explanations followed by explicit copy/paste commands.

Every formal execution step must have:

1. a real Markdown `### STEP<N> — ...` header;
2. a visible dated marker in the terminal/IPython transcript.

Shell:

```bash
echo "[$(date '+%F %T')] STEP1: <Short description>"
```

IPython/Python:

```python
from datetime import datetime; print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] STEP1: <Short description>")
```

Rules:

- keep the same step number in the heading and marker;
- do not use shell `exit` commands in instructions;
- prefer one readable command over a long opaque pipeline;
- do not assume a narrow/mobile terminal;
- use roughly 120-character lines when practical.

---

## 3. Project locations

Use the locations consistently:

```text
$DIAGNOSTICS_CODE/
    production source, tests, scripts, canonical technical documentation

$DIAGNOSTICS_CODE/docs/
    current canonical project documentation only

$DIAGNOSTICS_CODE/test_logs/
    raw canonical test logs and generated review evidence
    NEVER COMMIT

$DIAGNOSTICS_CODE/files_diag/
    temporary reviewer-distribution / packaging workspace
    NEVER COMMIT

$DIAGNOSTICS_DOCS/
    PHASE proposals/specifications, CRRs, review requests/prompts,
    review summaries and phase-specific closure records

$DIAGNOSTICS_DATA/
    host_diag_* bundles, rendered reports, large runtime evidence
```

`test_logs/` and `files_diag/` must be ignored by Git and must never be staged.

Generated monitoring data must not be written into the source tree by default.

Canonical `run_tests.sh` output uses:

```text
$DIAGNOSTICS_CODE/test_logs/
    bash/pytest logs, evidence bundles and rendered test reports

$DIAGNOSTICS_CODE/files_diag/
    reviewer_<timestamp>.zip / diagnostics_reviewer_<timestamp>.zip
```

---

## 4. Exact-byte handoff

Exact-byte delivery is mandatory for downloadable source, tests, documentation, CRRs, review summaries and scripts.

Every AI-delivered file must come with:

```text
1. the file itself
2. download filename/glob
3. canonical destination
4. MD5
5. SHA256
6. exact install command
7. exact `check_ref` verification command
8. exact focused execution/test command when applicable
```

Use the shared helper; do not create a diagnostics-local fork:

```bash
source "$DIAGNOSTICS_INSTALL_BY_MD5"
```

Preferred installation pattern:

```bash
source "$O2DPG/UTILS/dfextensions/diagnostics/scripts/env.sh"
source "$DIAGNOSTICS_INSTALL_BY_MD5"

install_newest_by_md5 \
  '<download-pattern>' \
  '<canonical-destination>' \
  <MD5>

check_ref \
  '<canonical-destination>' \
  <MD5> \
  <SHA256>
```

`install_newest_by_md5` searches under `$Downloads` itself. Its first argument is a filename/glob relative to
`$Downloads`; do **not** prefix it with `$Downloads/`.

Do not rely on browser suffixes such as `(1)` or `(2)`: use a glob plus the expected MD5 and let the helper select the
matching bytes.

Do not ask the Architect to calculate or visually compare checksums. The AI/coder supplies the expected MD5 and SHA256;
`check_ref` performs the verification and must fail on a mismatch.

When a candidate contains multiple source/test files, treat the **complete candidate set** as the transfer/review unit.
Do not update production source while leaving stale tests or helper files.

Keep these operations distinct:

```text
prepare/download candidate
    -> install exact bytes
    -> verify exact bytes with check_ref
    -> run focused tests
    -> run canonical gate when required
    -> prepare review evidence / CRR
    -> stage exact reviewed files
    -> inspect staged diff
    -> commit only after the review gate allows it
```

Installation/verification is a separate gate from Git staging/commit. Never combine download/copy/check commands with
`git add` or `git commit`.

For cross-site validation, copy the exact reviewed candidate set, verify the bytes on the target host, and only then run
the target-environment test.

---

## 5. Scientific-goal anchor

Infrastructure is not the final product.

The phase goal is to measure the target workload and surrounding shared-host activity on a synchronized
timeline so that changes in job performance can be distinguished from background load, infrastructure
activity and specific host pathology.

Every work order and CRR must state in one short paragraph how the increment:

- advances;
- enables;
- verifies; or
- protects

that synchronized job-versus-host diagnostic product.

A review must not silently narrow the phase to:

- THP/compaction-only diagnosis;
- plotting whichever counters are nonzero;
- host monitoring disconnected from job performance;
- collection without analytics;
- analytics without canonical detailed evidence; or
- package/test infrastructure treated as the final product.

---

## 6. Coding discipline

Before adding a helper, parser, converter, generator, API or execution path:

1. search existing diagnostics source/scripts;
2. check ADF/dfdraw capabilities where relevant;
3. check Technical Summary / capability documentation where relevant;
4. reuse or extend the existing tested implementation by default;
5. if new code is still needed, state briefly why reuse was insufficient.

Keep examples and runbooks readable for physicists and operators.

Avoid unrelated cleanup in functional patches:

- no whitespace-only churn;
- no opportunistic reformatting;
- no unrelated renames;
- no tolerance widening to make tests pass.

Comments should describe the public contract, physical meaning or non-obvious invariant.

---

## 7. Shared-file rule

Any change outside `$DIAGNOSTICS_CODE` must be explicitly disclosed before review.

This applies especially to:

```text
$DFEXT_CODE/__init__.py
$DFEXT_CODE/AliasDataFrame/
other shared ADF/dfdraw/GB state
```

For every shared-file change, the CRR must state:

- exact file path;
- why diagnostics must change it;
- user-visible/public-API consequence;
- architect approval when required by governance;
- focused regression evidence;
- cross-subproject blast-radius check.

Do not hide a shared-file modification inside a diagnostics-only change list.

---

## 8. Test-first defect correction

For correctness or public-contract defects:

1. reproduce with the smallest synthetic/public-path test;
2. demonstrate RED on the reviewed baseline;
3. patch only the bounded defect;
4. demonstrate GREEN with the same test;
5. run nearby regression tests;
6. run a real-host/real-data smoke when environment structure matters;
7. run the canonical diagnostics gate before review/commit.

A public-API defect should be tested through the public API whenever feasible.

Do not modify production code after the final green evidence without rerunning the affected tests.

---

## 9. Validation hierarchy

Use the cheapest sufficient gate first:

```text
syntax / compile
focused synthetic pytest
bounded public-path test
real-host / real-data smoke or invariant
canonical diagnostics/run_tests.sh
```

Use `pytest` for bounded gates.

Use the canonical `run_tests.sh` for final CRR/commit evidence.

Skipped tests and skip reasons must remain visible in canonical evidence.

For numerical behavior, prefer exact invariants or diagnostic comparisons over tolerance widening.

---

## 10. Reviewer-packet reproducibility

If a reviewer packet says a reviewer can execute a command from the packet, that claim must be tested.

Required packet test pattern:

```text
build packet
    ↓
extract into a fresh empty directory
    ↓
run the advertised command exactly
    ↓
require success
```

The packet must carry enough exact source to reproduce every test it instructs the reviewer to run.

For own-team implementation reviews, include the actual phase diff or an equivalent complete phase-owned
patch artifact so reviewers can perform the required full-diff audit.

Recommended provenance artifacts:

```text
provenance/diff_to_phase_13_74_ADF.txt
provenance/git_log_phase_13_74_ADF.txt
provenance/MANIFEST.md5
```

When concurrent phases share the branch, also provide the PHASE_13_74_ADF-owned commit list/patch series.

---

## 11. Monitoring execution and data

Use `$DIAGNOSTICS_DATA` as the default parent for runtime monitoring bundles.

A bounded background watch is preferred over an unbounded daemon unless a later architect decision changes
the parked continuous-watch scope.

For an actual performance study, prefer `dfx_run_with_diagnostics.py` around the target workload so the
bundle contains a synchronized pre-job baseline, job window and post-job recovery period.

Do not treat the human-readable report as the sole evidence. Preserve the canonical machine-readable bundle.

---

## 12. Git and commit discipline

Diagnostics lives inside a shared repository. Avoid repository-wide noise.

Normal diagnostics inspection:

```bash
git -C "$DIAGNOSTICS_CODE" status --short .
```

When a shared `dfextensions` file is part of the reviewed unit:

```bash
git -C "$DFEXT_CODE" status --short -- __init__.py diagnostics/
```

Never use broad staging/commit shortcuts:

```bash
git add .
git commit -a
```

Do not amend shared-history commits unless the Architect explicitly requests it.

Stage and commit explicit reviewed files only.

Before staging:

```bash
git diff --check -- <explicit files>
git diff --stat -- <explicit files>
git status --short -- <explicit files>
```

Stage the exact reviewed unit:

```bash
git add <explicit files>
```

Before commit, inspect and validate the staged bytes:

```bash
git diff --cached --check
git diff --cached --stat
git diff --cached --name-only
git diff --cached -- <explicit files>
git status --short
```

Run the required focused tests and the canonical gate appropriate to the review boundary before committing.

Commit messages must describe what actually changed. For a nontrivial coding unit use:

```text
subject: PHASE_13_74_ADF: concise human-readable change

body:
- what changed / public or operational contract
- why it changed
- validation or behavioral intent
```

A preferred command form is:

```bash
git commit \
  -m 'PHASE_13_74_ADF: <concise change>' \
  -m '<what changed and why>' \
  -m '<validation / preserved contract>'
```

Do not mix unrelated source changes, generated evidence, backup files, temporary monitoring output, unrelated PHASE
documents or local scratch files into one commit.

**Review/commit boundary:** copy/install/check/test work stays separate from commit work. The pre-commit CRR must state
the exact proposed staging boundary and exact proposed commit command. The actual commit command is executed only after
the reviewed bytes and evidence have been inspected and the applicable review gate permits banking.

---

## 13. Review and CRR cadence

Recommended cadence:

```text
focused implementation unit
    ↓
bounded CRR/review
    ↓
reviewer P0/P1 converted to executable regression where practical
    ↓
bounded correction
    ↓
narrow confirmation review
    ↓
phase-closure review
```

Every pre-commit CRR must include:

- exact review target;
- exact changed files/fingerprints;
- relevant phase/proposal references;
- tests executed and host/environment;
- exact proposed staging boundary;
- exact proposed commit command, including subject and body;
- reviewer-bundle fingerprint when a bundle exists;
- shared-state disclosures;
- scientific-goal impact;
- known residue/deferrals;
- explicit reviewer questions and requested disposition.

Prepare the CRR and proposed commit instructions together so the reviewed bytes, staging scope and commit description
cannot drift apart. The coder must not self-approve.

A confirmation review verifies the identified blockers and regression safety. It should not reopen accepted
architecture without new evidence.

---

## 14. Reviewer obligations

Reviewers use the assigned `reviewerID` verbatim.

Own-team implementation reviewers inspect:

- exact source;
- full phase diff or equivalent complete patch artifact;
- tests;
- relevant normative documentation;
- Capability Matrix when applicable;
- shared-state removals/additions;
- advertised packet-reproduction commands.

Separate:

- P0;
- P1;
- P2/advisory;
- architectural disagreement;
- concrete implementation defects.

Do not approve phase closure while a genuine P0/P1 blocker remains.

---

## 15. Minimal copy/paste transcript convention

Example:

### STEP1 — Load diagnostics environment

```bash
echo "[$(date '+%F %T')] STEP1: Load diagnostics environment"
source "$O2DPG/UTILS/dfextensions/diagnostics/scripts/env.sh"
```

### STEP2 — Inspect diagnostics working state

```bash
echo "[$(date '+%F %T')] STEP2: Inspect diagnostics working state"
git -C "$DIAGNOSTICS_CODE" status --short .
```

### STEP3 — Run focused tests

```bash
echo "[$(date '+%F %T')] STEP3: Run focused tests"
cd "$DIAGNOSTICS_CODE" && python3 -m pytest -q <focused-test> -rs
```

### STEP4 — Run canonical diagnostics gate

```bash
echo "[$(date '+%F %T')] STEP4: Run canonical diagnostics gate"
cd "$DIAGNOSTICS_CODE" && bash run_tests.sh
```

### STEP5 — Stage the reviewed unit

```bash
echo "[$(date '+%F %T')] STEP5: Stage the reviewed unit"
git -C "$DFEXT_CODE" add <explicit files>
git -C "$DFEXT_CODE" diff --cached --check
git -C "$DFEXT_CODE" diff --cached --stat
git -C "$DFEXT_CODE" diff --cached --name-only
```

### STEP6 — Commit the approved reviewed unit

Run only after the applicable pre-commit review gate permits banking.

```bash
echo "[$(date '+%F %T')] STEP6: Commit approved PHASE_13_74_ADF unit"
git -C "$DFEXT_CODE" commit \
  -m 'PHASE_13_74_ADF: <concise change>' \
  -m '<what changed and why>' \
  -m '<validation / preserved contract>'
```

### STEP7 — Verify committed boundary

```bash
echo "[$(date '+%F %T')] STEP7: Verify committed PHASE_13_74_ADF boundary"
git -C "$DFEXT_CODE" show --stat --oneline --decorate -1
git -C "$DFEXT_CODE" status --short -- diagnostics/ __init__.py
```

Optional cleanup comes only after required work and must be labelled `Optional` unless stale files would
contaminate packaging, testing, source snapshots or formal evidence.

---

## 16. Practical default

When several approaches are possible, prefer the one that is:

```text
easiest to reproduce
easiest to review
least likely to change unrelated behavior
protected by an executable test
clear to a physicist returning later
```

The objective is not maximum process. The objective is traceability: which bytes changed, why they changed,
which executable evidence demonstrates the behavior, and which review accepted it.
