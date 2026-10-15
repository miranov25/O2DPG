# dfdraw Working Style

**File:** `docs/dfdraw_WORKING_STYLE_20261015_v3.md`  
**Date:** 2026-10-15  
**Status:** revised operational working-style candidate — commit-message semantics and pre-commit verification strengthened  
**Scope:** dfdraw development, review, commissioning, bug fixing, and cross-project integration

**Revision note:** v2 incorporates the Architect's four binding operational rules and the 2026-10-15 panel review: useful subject+body commit messages; exact-file heredoc/pathspec commits; `git status --short .`; numbered/timestamped executable steps; canonical `$DFDRAW_CODE`/`$DFDRAW_DOCS` custody; and mandatory exact-byte install/check discipline when files are delivered externally.

**v3 revision note:** commit-message quality is now defined semantically, not merely syntactically. A commit message must explain the actual observable change, defect, evidence, or behavior. Process labels such as “gate”, “checkpoint”, “characterization”, “review”, “CRR”, “Stage A”, or “update” are not sufficient unless the message also states what was actually found or changed. Pre-commit verification now explicitly repeats `check_ref` for externally delivered reviewed artifacts before staging/commit and requires inspection of the exact staged diff.

---

## 1. Purpose

This document defines the day-to-day engineering and review workflow for dfdraw.

It is intentionally operational. It does **not** replace the canonical Architect Decision registry, phase proposals, governance documents, or ratified review decisions.

If this document conflicts with a ratified Architect Decision or a phase-specific approved proposal, the ratified decision/proposal wins and this document must be updated.

Primary references:

- `docs/ARCHITECT_DECISIONS.md`
- `docs/Organization-structure.md`
- the active `PHASE_*` proposal and its ratified Architect Decisions
- the latest approved CRR/review summary
- the exact source snapshot used for the current review or implementation

The working style should be kept current. When a process lesson recurs or becomes binding, update this file rather than relying on conversational memory.

---

## 1.1 Canonical environment, repositories, and document locations

Use the existing dfextensions environment. Do not invent parallel variables or locations for the same purpose.

For normal laptop/local dfdraw work, initialize with:

```bash
source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh
```

The current local environment defines at least:

```text
$DFDRAW_CODE    dfdraw source repository area
$DFDRAW_DOCS    formal dfdraw project-document area
```

The same environment also sources the shared `install_by_md5.sh` helper. Therefore normal copy/install instructions should source `env.local.sh` once rather than separately sourcing the checksum helper again.

Canonical custody:

```text
$DFDRAW_DOCS/
    formal PHASE proposals
    Architect-decision records when kept with the phase
    CRRs
    consolidated review summaries
    technical/project notes intended for the formal project record

$DFDRAW_CODE/docs/
    long-term operational/code-team documentation
    Working Style
    source-facing technical documentation

$O2DPG/UTILS/dfextensions/files_dfext/
    snapshot / reviewer-transfer staging when used by the dfextensions workflow
    not the canonical home of PHASE documents
    do not treat a transfer copy as authoritative merely because it is convenient
```

**Formal PHASE documents must not be committed from `$DFDRAW_CODE`.** Their canonical committed copy belongs under `$DFDRAW_DOCS`.

If a phase uses another explicit reviewer-transfer staging directory, state it in that phase. Transfer staging is not a substitute for the canonical document/source location.

---

## 1.2 Formal executable-step format

All formal install, test, Git, review, snapshot, commissioning, and commit instructions must be delivered as numbered copy/paste steps.

Each logical step has:

1. a Markdown header;
2. a visible dated marker as the first executed line.

Example:

```markdown
### STEP3 — Run focused PHASE_13_86 tests
```

```bash
echo "[$(date '+%F %T')] STEP3: Run focused PHASE_13_86 tests"
source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh
cd "$DFDRAW_CODE"
python -m pytest tests/test_phase_13_86_stage_a.py -q
```

Binding rules:

- Continue step numbering across follow-up messages in the same workflow.
- Restart at `STEP1` only for a genuinely new workflow.
- Evidence intended for review, CRR, commissioning, copy/install verification, or commit preparation requires the timestamp marker.
- Keep each block self-contained: source the canonical environment, enter the owning directory, and do not assume hidden shell state from an earlier block.
- Prefer explicit filenames over temporary shell variables when the names fit readably on screen.
- Do not use `set -e` in Architect copy/paste blocks unless an approved workflow explicitly requires it.
- Prefix slow substantive commands with `time` where useful.


---

## 1.3 Canonical reviewed-file workflow

For externally delivered reviewed artifacts, the normal workflow is:

```text
install_newest_by_md5
    ↓
check_ref (MD5 + SHA256)
    ↓
review/test
    ↓
repeat check_ref on canonical destination before commit
    ↓
git add -- exact files
    ↓
git diff --cached --check -- exact files
    ↓
git diff --cached -- exact files
    ↓
git commit --only -F - -- exact files <<'EOF'
    useful subject + concrete body
EOF
    ↓
git show --stat --format=fuller HEAD -- exact files
    ↓
git status --short .
```

Do not skip the installation/integrity steps because a filename “looks present”.
Do not replace byte verification with `ls -l`.
Do not replace staged-content inspection with a filename-only check.
Do not replace a concrete commit description with an internal process label.


---

## 2. Engineering goals

dfdraw development should optimize for four things simultaneously:

1. **Correct scientific semantics.** A plot that renders successfully but silently loses a branch, selection, weight, facet, normalization, or statistic is a failure.
2. **Symmetry and consistency.** Semantically equivalent public forms should resolve to the same meaning unless a difference is explicitly ratified.
3. **Low-disruption evolution.** Existing analyses depend on dfdraw. Large refactors must be staged, shadow-validated, and migrated incrementally.
4. **Maintainability for humans and AI reviewers.** Code, tests, proposals, and examples must be understandable without reconstructing months of project history.

A technically clever solution that is hard to review, hard to explain, or difficult to validate is not the preferred solution.

---

## 3. Readability is a specification requirement

Do not use unexplained project shorthand as if it were ordinary language.

At first meaningful use, write the full term. Add a shorthand only when it genuinely improves readability.

Bad:

```text
IR -> S4 -> RESOLVED -> Gate 1A
```

Better:

```text
canonical semantic representation
    -> resolved semantic request
    -> statistical state
    -> rendering
```

Every substantial proposal or CRR should begin with a plain-language explanation of:

- the problem;
- the intended user-visible result;
- what changes;
- what explicitly does not change.

A competent reviewer who was not present in earlier discussions should be able to understand the document from the document itself plus the cited source files.

Code names, document names, and architectural names should refer to the same concepts consistently.

---

## 4. Roles and decision ownership

### 4.1 Architect

The Architect owns scientific intent, public semantics, scope boundaries, and unresolved architectural decisions.

Architect decisions should not be silently inferred or widened by coders or reviewers.

Where an Architect statement is being recorded as an official decision, preserve its meaning exactly. The canonical Architect Decision registry remains authoritative.

### 4.2 Coder

The coder:

- implements only approved scope;
- preserves source custody;
- produces RED evidence before bug fixes when practical;
- runs focused tests and the canonical full gate;
- prepares exact CRR evidence;
- does not silently defer approved items;
- does not mix unrelated work into a phase commit.

The coder may diagnose and propose, but does not self-approve architectural changes.

### 4.3 Reviewer

The reviewer is independent from the implementation under review.

The reviewer should:

- source-read the relevant implementation and tests;
- verify the exact candidate bytes;
- test semantic ownership and failure modes, not only happy-path rendering;
- distinguish code defects from oracle defects;
- verify that tests prove the intended contract;
- explicitly state approval, conditional approval, or refusal.

For official review documents, identify the reviewer seat/ID consistently.

---

## 5. New-reviewer onboarding

Do **not** ask a new reviewer to read the entire historical PHASE archive.

Recommended order:

1. current governance / reviewer rules;
2. `dfdraw_Technical_Summary.md` or the latest equivalent technical orientation;
3. the latest exact source snapshot;
4. the active phase proposal;
5. current Architect Decisions relevant to that phase;
6. the latest consolidated review/approval summary;
7. only the source files and tests relevant to the current change.

Historical phase documents are consulted when a specific precedent or decision is needed, not as default onboarding.

---

## 6. Phase lifecycle

A normal dfdraw phase should have a finite, auditable lifecycle.

```text
problem / goal
    ↓
proposal
    ↓
Architect decisions where needed
    ↓
independent proposal review
    ↓
BEGIN tag / exact source custody
    ↓
small implementation units
    ↓
focused tests
    ↓
full canonical test gate
    ↓
downstream/oracle validation when applicable
    ↓
CRR
    ↓
explicit-path commit
    ↓
END tag
```

Do not leave a broad phase open indefinitely when a finite architecture checkpoint can be closed and later work can use a successor phase.

Concurrent phases may share a branch, but each phase must keep explicit file and commit custody.

---

## 7. Source custody

Exact bytes matter more than filenames.

Before implementation or review, record as appropriate:

```text
SOURCE_COMMIT
REPO_HEAD
CLEAN or DIRTY state
ZIP/TAR/text-bundle identity
MD5
SHA256
key-file hashes
```

For a **clean committed-ref snapshot**, the commit is the authoritative source.

For a **dirty worktree snapshot**, the snapshot bytes are authoritative; `REPO_HEAD` is only generation context.

Never describe a dirty package as if it were a committed reference.

Use the existing audit-snapshot tooling where available.

Typical tools already used in the project:

```bash
check_ref <file> <md5> <sha256>

../scripts/make_audit_snapshot.sh \
  ../dfdraw \
  dfdraw \
  . \
  <optional-git-ref>
```

When source identity changes after a fix, downstream teams must receive the new exact reference.

---

## 8. Exact-byte installation and copy discipline

Exact-byte installation/verification is a separate gate from Git staging and commit.

When a coder/reviewer delivers a repository-bound file through `$Downloads`, provide all of:

```text
filename / download pattern
canonical destination
MD5
SHA256
exact install command
exact check_ref command
```

The canonical environment already sources the shared installation helper:

```bash
source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh
```

Do not normally source `install_by_md5.sh` a second time.

The preferred pattern is:

```bash
install_newest_by_md5 \
  '<download-pattern-relative-to-$Downloads>' \
  '<canonical-destination>' \
  '<MD5>'

check_ref \
  '<canonical-destination>' \
  '<MD5>' \
  '<SHA256>'
```

Rules:

- The first `install_newest_by_md5` argument is a glob relative to `$Downloads`; do not prepend `$Downloads/`.
- Treat the install operation as consuming/moving the selected download; do not depend on the browser copy remaining in place afterward.
- Install one reviewed implementation unit, verify its exact bytes, then run syntax/focused checks before installing the next independent unit.
- Do not ask the Architect to edit delivered files manually when exact bytes can be installed.
- Do not rely on browser suffixes such as `(1)` or `(2)`; use a controlled glob plus exact MD5.
- Copy/install + `check_ref` is one formal step. Git commit is a later, separate formal step.
- **Never assume that a delivered/reviewed file has already been copied into its canonical destination.** If the file arrived through Downloads/chat/reviewer delivery, start with install + `check_ref`.
- Before staging/committing externally delivered reviewed files, repeat `check_ref` on the canonical destination so the bytes being committed are proven to be the reviewed bytes.
- If a file was created directly in the canonical repository/document tree and no external candidate bytes are involved, do not fabricate an install step. Verify the exact reviewed bytes by the applicable source/hash/review mechanism instead.

### Example exact-byte install step

```markdown
### STEP<n> — Install and verify the reviewed dfdraw document
```

```bash
echo "[$(date '+%F %T')] STEP<n>: Install and verify the reviewed dfdraw document"
source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh
cd "$DFDRAW_DOCS"

install_newest_by_md5 \
  'PHASE_example*.md' \
  "$DFDRAW_DOCS/PHASE_example.md" \
  '<MD5>'

check_ref \
  "$DFDRAW_DOCS/PHASE_example.md" \
  '<MD5>' \
  '<SHA256>'
```

Do not combine the install/check block with the eventual `git commit` block.

---

## 9. Small implementation units

For risky or semantically broad changes, work in small rounds.

Preferred sequence:

```text
1. custody / current-state census
2. RED characterization or independent oracle
3. smallest routing/ownership change
4. smallest numerical/statistical change
5. focused GREEN
6. regression gates
7. CRR
```

After each nontrivial step, inspect the result before widening the implementation.

Avoid long chains of speculative edits.

If a change starts expanding beyond the approved scope, stop and return to review rather than completing an accidental refactor.

---

## 10. Bug-fix workflow

For a real bug, especially a user-visible semantic bug:

### 10.1 Reproduce first

Create the smallest deterministic reproducer that demonstrates the defect.

Prefer an independent truth source where possible:

- direct NumPy/statistical calculation;
- hand-designed fixture;
- exact branch membership;
- known analytical result;
- current production behavior when explicitly intended as invariance.

### 10.2 RED before product edit

When practical, install and run the regression test before editing product code.

The RED should fail for the intended reason.

A RED test that fails because of an unrelated setup/import error is not useful evidence.

### 10.3 Fix the owner, not the symptom

Reuse the existing semantic/statistical owner.

Do not duplicate:

- vector composition;
- histogram arithmetic;
- profile arithmetic;
- normalization transforms;
- layout ownership;
- selection/weight combination rules.

If two paths need the same meaning, converge them onto one owner rather than copy logic.

### 10.4 GREEN plus invariance

After the fix:

- run the new focused test;
- run the nearest historical regression tests;
- run tests for recently repaired semantic classes that the change could affect;
- then run the canonical full gate.

---

## 11. Oracle discipline

A test suite proves implementation consistency. It does not automatically prove scientific correctness.

For semantic changes, prefer independent or injected-truth oracles.

Important rules:

- do not weaken an oracle because product code fails;
- do not widen numerical tolerances without an explicit technical reason and review;
- identify the first disagreement layer where possible;
- distinguish product defect, oracle defect, and unsupported/unknown state;
- `UNKNOWN` is not equivalent to PASS.

For plot semantics, a useful oracle checks both:

```text
machine-readable numerical/state result
+
rendered/visual structure
```

When downstream ADF galleries cover the modified dfdraw behavior, use them as a cross-project safety net.

---

## 12. Visual validation

For user-visible plotting behavior, tests should not stop at “figure object exists.”

Check relevant properties such as:

- number of branches;
- branch order;
- bin values;
- errors;
- facet/grid cardinality;
- axis count;
- normalization panel;
- labels/legends where semantically relevant;
- masks/undefined bins;
- layout invariants.

When reviewing HTML, PDF, notebook, or generated gallery artifacts, at least one reviewer should render and inspect the artifact, not only read source or diffs.

This follows the established render-the-artifact governance rule.

---

## 13. Canonical full test gate

Focused tests are necessary but not sufficient for final approval.

Use the canonical project gate for CRR/closure:

```bash
./run_tests.sh
```

Do not substitute a narrow “fast” mode for final CRR evidence unless the active approved proposal explicitly permits it.

Record:

```text
passed
failed
errors
skipped
xfail/xpass where relevant
exit code
```

Unexpected new XFAIL/skip is not automatically acceptable; explain it.

---

## 14. Public-surface symmetry

The intended design is symmetry for semantically applicable features.

Equivalent user-facing forms should converge, for example where applicable:

```text
draw(...)
typed method(...)
dictionary / recursive request form
```

A behavior should be refused only when:

- the operation is genuinely semantically inapplicable; or
- the Architect explicitly approves the limitation.

Existing implementation asymmetry alone is not evidence that the asymmetry is intended.

Silent dropping of an applicable parameter is a high-severity semantic defect.

---

## 15. Current architectural evolution rule

dfdraw is moving toward a generic statistical representation, persistence/merge support, and later execution unification.

The transition must remain low disruption.

Binding working rule:

```text
current executor remains authoritative
    ↓
additive persistence / statistical-state work
    ↓
canonical semantic/statistical representation in read-only or shadow mode
    ↓
systematic equivalence/oracle validation
    ↓
bounded migration of execution
    ↓
retirement only by explicit decision
```

Do not use an urgent bug fix as an excuse to implement an unfinished future architecture.

Do not use future architecture work as an excuse to leave an important current bug unfixed when a bounded compatibility fix is available.

Unresolved schema choices must be explicitly ratified before persisted formats make them expensive to change.

---

## 16. Statistical-model direction

The accepted mental model for binned statistics is generic and N-dimensional:

```text
named coordinate dimensions
    +
statistic-specific retained state
```

Physical bin axes and semantic axes such as group/facet/branch belong to the coordinate model when they represent independent result coordinates.

Examples:

```text
x_bin
y_bin
group
facet
branch
```

Scalar selection filters rows; scalar weight modifies accumulation.

Vectorized selections or weights may create branch coordinates.

Histogram/profile families should not be special-cased into unrelated persistence models merely because they are 1D, 2D, or higher dimensional.

However, this working-style document does **not** freeze unresolved PHASE_13_82 persistence-schema details. Those remain governed by the ratified Architect Decisions for that architecture round.

---

## 17. Persistence and merge principles

Keep two concepts separate:

```text
exact state merge
    compatible retained state
    -> same statistic on combined partitions

derived aggregation
    child/finalized statistics
    -> new named statistic
```

Derived approximations are allowed when explicitly named and interpreted.

For binned exact merge:

- bin definitions must be compatible;
- incompatible binning should refuse loudly;
- automatic rebinning is not an implicit fallback.

Persistence should store enough state to support the operations promised by the schema, not only the displayed numbers.

---

## 18. Reuse before create

Before adding a new helper or semantic layer, inspect existing owners.

Questions to ask:

```text
Does dfdraw already compute this quantity?
Does an existing helper own this rule?
Can the new path consume the existing state?
Can two old implementations converge onto one shared helper?
```

Creating a second implementation of the same scientific rule is usually worse than a slightly broader reusable owner.

This is particularly important for:

- weight composition;
- selection composition;
- binning;
- histogram/profile errors;
- normalization;
- fit summaries;
- facet/group/vector semantics.

---

## 19. Example code is also documentation

Files under `examples/` are not disposable test machinery.

When an example is used as an oracle or commissioning driver, it must still remain:

- readable;
- commented where non-obvious;
- useful to a human user;
- representative of supported public APIs.

A CRR that modifies example code should explicitly ask reviewers whether the example remains understandable and usable as an example.

Do not turn examples into cryptic test harnesses.

---

## 20. Git inspection, generated files, and unrelated dirt

dfdraw and the formal dfdraw documents live inside larger repositories. Routine Git inspection must therefore be scoped.

**Never use repository-global `git status` for normal dfdraw work.**

Use:

```bash
git status --short .
```

or, for a smaller reviewed unit:

```bash
git status --short -- path/to/file1 path/to/file2
```

Never use:

```text
git status
git status --short
git status --porcelain
```

without `.` or another explicit path scope when the intent is only the current dfdraw/code/document area.

Never stage broadly:

```text
git add .
git add -A
git commit -a
```

Stage only explicit reviewed new/modified files when staging is needed for review:

```bash
git add -- \
  path/to/file1 \
  path/to/file2 \
  path/to/test
```

Before commit/review, verify the exact reviewed paths.

For externally delivered reviewed files, first repeat the exact-byte check on the canonical destination:

```bash
check_ref \
  path/to/file1 \
  '<MD5>' \
  '<SHA256>'
```

Then stage only the intended files and inspect the exact staged content:

```bash
git add -- \
  path/to/file1 \
  path/to/file2 \
  path/to/test

git diff --cached --check -- \
  path/to/file1 \
  path/to/file2 \
  path/to/test

git diff --cached -- \
  path/to/file1 \
  path/to/file2 \
  path/to/test

git status --short .
```

`ls -l` may be useful for orientation, but it is **not** an integrity or review check and must never replace `check_ref` or staged-diff inspection.

For exact reviewed/frozen paths, the unstaged `git diff -- <paths>` should be empty before the final commit. If it is not empty, the current working bytes differ from the reviewed/staged candidate and must be reconciled before commit.

Do not accidentally include generated capability matrices, logs, temporary patches, backup files, reviewer packets, gallery exports, or unrelated phase work.

If unrelated dirt exists, record and protect it explicitly. Do not “clean up” unrelated work merely to simplify the current phase.

---

## 21. Commit discipline — exact files and useful history

The final commit command must itself show **exactly which files are being committed**.

**Never use a plain `git commit` that relies on whatever happens to be staged.**

Canonical form:

```bash
git commit --only -F - -- \
  path/to/file1 \
  path/to/file2 \
  path/to/test <<'EOF'
PHASE_XX_YY_DF: describe the actual behavior or documents committed

Explain in plain language what changed and why.
State important validation, compatibility boundaries, or deliberate non-changes.
EOF
```

Binding rules:

- Every final commit names the exact file pathspecs after `--`.
- Use `--only` so unrelated staged files cannot silently enter the intended commit.
- Brand-new untracked files must first be introduced explicitly with `git add -- <new-file>`.
- Never use `git commit -a`.
- Never use a cryptic one-line commit such as `fix B2`, `update stuff`, `CRR2`, `record review gate`, `bank checkpoint`, or `Stage-A characterization`.
- Do not use `git commit --amend` in the normal reviewed workflow; a correction is a new commit unless the Architect explicitly directs otherwise.
- The exact committed files must be visible by eye in the command.
- A nontrivial commit message contains:
  - a descriptive subject/header, normally beginning with the PHASE/BUG identifier;
  - the **actual observable change, defect, evidence, or behavior**;
  - why the change/evidence matters;
  - important validation and/or scope boundaries where useful.

### Commit-message semantic quality rule

A commit message is **not acceptable** merely because it names the phase, stage, review type, or process step.

Process words such as:

```text
gate
checkpoint
characterization
review
CRR
Stage A / Stage B
update
cleanup
follow-up
```

do not explain what happened.

The message must answer, in plain language:

```text
What actually changed or was demonstrated?
What user-visible / semantic / test behavior is involved?
What was verified?
What remains deliberately unchanged or blocked, if relevant?
```

Bad:

```text
PHASE_13_86_DF: record Stage-A RED characterization and review gate
```

Better:

```text
PHASE_13_86_DF: document reproduced facet/vector failures before fixing dfdraw
```

The body should then state the concrete findings, for example:

```text
- faceted scatter drops xerr/yerr;
- faceted time_format is lost;
- selection_vector + facet collapses two branches into one scalar plot;
- the old xerr+facet test passes falsely with zero error bars;
- 70 historical neighbor tests remain green;
- stronger RED oracles are required before product code changes.
```

The commit message should remain useful from `git log` months later **without reopening chat, the CRR, or the review summary**.

### Canonical formal commit step

```markdown
### STEP<n> — Commit the exact reviewed files
```

```bash
echo "[$(date '+%F %T')] STEP<n>: Commit the exact reviewed files"
source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh
cd "$DFDRAW_CODE"

git status --short .

git diff --check -- \
  drawer.py \
  tests/test_phase_example.py

git diff -- \
  drawer.py \
  tests/test_phase_example.py

git diff --cached --name-only -- .
git diff --cached --check -- \
  drawer.py \
  tests/test_phase_example.py

git commit --only -F - -- \
  drawer.py \
  tests/test_phase_example.py <<'EOF'
PHASE_XX_YY_DF: preserve faceted error bars instead of silently dropping xerr/yerr

Forward the existing xerr/yerr semantics through the facet path so every
facet renders the same error-bar information as the single-panel owner.

Add a content-level regression test that counts real ErrorbarContainer
objects and verifies the requested x/y error semantics. The change does not
introduce a new statistics engine or alter unrelated scalar rendering.
EOF

git show --stat --format=fuller HEAD -- \
  drawer.py \
  tests/test_phase_example.py

git status --short .
```

The commit step is separate from the earlier install/check step and from the independent review decision.

Commit/operator commands should normally be supplied as a separate operator-step block/document rather than mixed into the technical CRR body. The CRR records the proposed commit scope/message and review evidence; the operator step performs the approved commit.

---

## 22. CRR expectations

A final Code Review Request should normally contain:

```text
scope
exact source/candidate identity
what was actually changed
what was deliberately not changed
relevant Architect Decisions
source files changed
tests added/modified
RED evidence where applicable
focused GREEN evidence
regression evidence
full run_tests.sh result
downstream/oracle evidence where applicable
known warnings / known gaps
exact staged file list
proposed commit subject/body and exact committed-file list
```

Include a short plain-language “what was actually done” section. Hashes and commands alone are not sufficient.

Reviewers should be able to distinguish:

```text
proposal approval
implementation approval
commit approval
phase closure approval
```

Do not conflate them.

---

## 23. Cross-project changes

ADF and other projects depend on dfdraw behavior.

For changes that touch shared/public semantics:

- freeze the exact dfdraw dependency;
- run relevant downstream oracle/gallery tests;
- report ownership of any remaining RED;
- give downstream teams the exact new committed-ref snapshot after closure.

Do not tell a downstream project to continue against an ambiguous or changing dependency boundary.

---

## 24. Handover discipline

Create or refresh a coder/reviewer handover when:

- the coder/model is being replaced;
- a phase is interrupted;
- context has become too large;
- work moves temporarily to another phase;
- there is protected uncommitted work.

A handover should contain:

```text
current branch / HEAD
current `PHASE_HISTORY.md` status and whether it is up to date
phase BEGIN/END state
exact modified/untracked files
protected unrelated work
current source hashes
approved scope
decisions already made
tests already run
current RED/GREEN state
next smallest implementation step
what must NOT be changed
```

Do not rely on the next coder reconstructing state from chat history.

---

## 25. Review packets

A reviewer packet should contain enough evidence to review without depending on the coder's local working tree.

Typical contents:

```text
proposal / ratified decision
CRR
exact source diff
source snapshot or declared exact source reference
focused test logs
full test log/summary
oracle/gallery evidence where applicable
git status / staged-file list
artifact hashes
```

Reviewers must be told whether the package is:

```text
clean committed-ref
or
dirty worktree snapshot
```

For dirty snapshots, snapshot bytes are authoritative.

---

## 26. When to ask the Architect

Ask only when a genuine decision is unresolved.

Good reasons:

- two reasonable semantic interpretations;
- compatibility policy would change user-visible behavior;
- scope needs widening;
- an existing Architect Decision conflicts with the simplest implementation;
- persistence/schema choice will be expensive to reverse.

Do not ask the Architect to choose implementation trivia that can be resolved by measurement, source inspection, or existing precedent.

When asking, present:

```text
the concrete question
2–3 options
recommended option
consequences
what must be decided now
what can be deferred
```

---

## 27. Performance work

Do not optimize from intuition alone.

Preferred cycle:

```text
reported problem
    ↓
minimal reproducer / correctness oracle
    ↓
benchmark
    ↓
smallest justified optimization
    ↓
same oracle
    ↓
review
```

Correctness remains authoritative.

Avoid performance changes that silently alter statistical semantics.

---

## 28. Refusal and failure behavior

Prefer loud, informative refusal over silent semantic degradation.

A refusal should name:

- the unsupported combination;
- the conflicting dimensions/options;
- the expected compatible form when useful.

Do not silently:

- drop vector branches;
- ignore normalization;
- discard facet/group dimensions;
- widen or reinterpret scientific state;
- rebin incompatible data;
- replace user-specified semantics with a convenient fallback.

---

## 29. Documentation maintenance

This file should be updated when:

- a new governance principle becomes binding;
- a recurring process failure needs mechanical prevention;
- the canonical review lifecycle changes;
- source-custody tooling changes;
- a major architecture transition is ratified.

Prefer a dated immutable revision:

```text
docs/dfdraw_WORKING_STYLE_YYYYMMDD.md
docs/dfdraw_WORKING_STYLE_YYYYMMDD_vN.md    # when multiple approved revisions occur on one day
```

and, if desired by the repository convention, a convenience symlink:

```text
docs/dfdraw_WORKING_STYLE.md
    -> dfdraw_WORKING_STYLE_YYYYMMDD.md
```

Do not overwrite historical dated revisions without an explicit reason.

---

## 30. Short operational checklist

Before coding:

```text
[ ] correct phase / scope
[ ] canonical environment sourced
[ ] exact source baseline known
[ ] formal PHASE documents located under $DFDRAW_DOCS
[ ] unrelated dirt identified/protected
[ ] proposal / Architect Decisions understood
[ ] smallest acceptance oracle defined
[ ] formal execution steps use continued STEP numbering + dated markers
```

Before CRR:

```text
[ ] focused tests green
[ ] relevant invariance/oracle tests green
[ ] full canonical test gate green
[ ] user-visible artifact rendered where applicable
[ ] exact candidate hashes recorded
[ ] external candidate installed with install_newest_by_md5 + verified with check_ref where applicable
[ ] check_ref repeated on canonical destination immediately before staging/commit where applicable
[ ] git status inspected only with . / explicit paths
[ ] exact staged diff inspected with git diff --cached -- <explicit paths>
[ ] staged/reviewed files explicit and clean
[ ] final commit command names exact files after --
[ ] commit subject states the actual change/evidence, not only the process/stage label
[ ] commit body explains concrete behavior/findings and validation without reopening the CRR
[ ] plain-language change summary written
```

Before closure:

```text
[ ] independent review approval
[ ] exact reviewed bytes committed
[ ] downstream dependency/oracle evidence captured where applicable
[ ] committed-ref snapshot created when needed
[ ] END tag created and verified
[ ] phase_info reports the intended closed range
```

---

# Working principle

> Make the scientific meaning explicit, keep one owner for each semantic rule, change the system in small auditable steps, and never trade reviewability for cleverness.

That principle applies equally to current bug fixes and to the longer-term statistical-state/persistence refactor.
