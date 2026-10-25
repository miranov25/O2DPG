# AliasDataFrame Working Style

**Version:** v1.5 · **Date:** 2026-10-24 · **Owner:** the ADF team, maintained by the active coder seat
**Location:** `AliasDataFrame/docs/ADF_WORKING_STYLE_20261024_v1_5.md`; the stable entry point `AliasDataFrame/docs/ADF_WORKING_STYLE.md` links to the currently **approved and installed** revision (§1.2). The new file does not become the current revision until the symlink is updated after review.

**Revision note (v1.4):** retains every v1.3 rule and adds the reviewer-handoff custody workflow in §14.1, based on the repeatedly used ADF commissioning process. Reviewer inputs are co-located under `files_ADF/` without confusing that transient handoff directory with the formal docs repository: ChatGPT/downloaded documents are installed by exact MD5 and verified with `check_ref`; repository-local evidence already produced under `test_logs/` is copied directly with `cp -p`; generic gallery filenames are renamed to checkpoint-specific names to prevent collisions; reviewer packets are copied as produced and verified against their recorded digests; installer backups are removed only when they were created by the current install and are named explicitly. §20 gains the corresponding handoff checklist. No architectural or phase-specific rule is changed.

**Revision note (v1.5):** implements the Architect's 2026-10-24 corrections to the standing operational rules. (1) Formal documents go directly to the formal documentation repository using `install_newest_by_md5` followed by `check_ref`, not a separate staging helper. (2) Shell instructions never invoke the shell-termination command, not even inside subshells; guarded `if`/`&&` chains and an explicit failure result prevent later actions after an error. (3) Every commit, including documentation-only commits, uses an informative, non-cryptic subject and a multiline plain-language body stating what changed, why, evidence/status, and what did not change. Also corrects the description of the install helper's default Downloads retention, as independently observed from its installed definition. No ADF or dfdraw functionality, architecture or acceptance-oracle policy changes.

**Lineage:** v1 `c94ae685…` → v1.2 `0521c6f0…` (absorbed GPT49, Fabble5_8, GPT53 and a separate seat's v1.1 `a07d8143…`) → v1.3 → v1.4 → v1.5.

---

## Terms used in this document

A **phase** is a numbered unit of work such as `PHASE_13_76_ADF`; it has a proposal, increments, and a closure. A **CRR** (code-review request) is what a coder submits with an increment: what changed, what was measured, what reviewers should attack. An **AD** is a ratified Architect Decision, always cited in the form the registry `docs/ARCHITECT_DECISIONS.md` uses for its headings — `AD-13/13.76.ADF`, `AD-1/13.55.ADF` — that is, decision number, then the short phase number, then the team. A decision local to a proposal and not yet ratified is a **PD**; it lives in its proposal, never in the registry, and is cited as that proposal defines it. **Nobody invents an `AD-` number** — only the Architect assigns one, on ratification. A **STEP** is one numbered, copy-pasteable block of commands (§4). An **oracle** is the independent source of an expected answer (§11). A **tag** is a permanent git name for one commit (§15).

---

## 0. Four rules the Architect added explicitly

Each was enforced by hand often enough to be stated first.

```text
1  NEVER make a flat commit.
   Every commit names its files after `--`.             -> §8

2  NEVER use a flat `git status`.
   Always scope it: `git status --short .`             -> §7

3  Every executable step is a Markdown HEADER
   (### STEPn — what it does), so the Architect can
   see and select steps in the prompt.                  -> §4

4  Every executable block begins with a dated STEP
   marker — in bash AND in python — so the transcript
   records what was done, when, and why.                -> §4
```

The reason behind all four: this repository is shared by coders working in parallel, and the Architect works by copy-pasting **selected** steps from Markdown into a shell or an interactive Python session. A flat commit or status silently reaches into other people's work; an unmarked step leaves a transcript nobody can reconstruct a week later.

**Additional non-negotiable Architect requirements (2026-10-24):** Shell instructions never run `exit` anywhere, including within a subshell (§4.2). Every commit has a meaningful, human-readable subject and explanatory multiline body (§8).

---

## 1. Purpose, precedence and versioning

### 1.1 What this document is

This document holds the **standing** rules for day-to-day ADF engineering and review. It is operational, not architectural.

**It is not a handover.** A handover (§18) records state at one moment — HEAD, hashes, which tests are red. A working style records what stays true across seats and phases. `CODER_HANDOVER_ADF.md` mixed both; its phase-state section went stale the day it was written, while its generic sections carried this team for three months and are the source of most of this document.

**Precedence.** If this document conflicts with a ratified AD or an approved phase proposal, the AD or proposal wins and this document is updated.

Read at onboarding:

```text
docs/ARCHITECT_DECISIONS.md     ratified rulings
docs/PHASE_HISTORY.md           what each phase did
the active PHASE_* proposal and its ratified decisions
the latest approved CRR / review summary
the exact source snapshot under review
```

### 1.2 Finding the current revision

```text
docs/ADF_WORKING_STYLE_YYYYMMDD[_vN_M].md   an approved revision — never rewritten
docs/ADF_WORKING_STYLE.md                   git-tracked relative symlink to the
                                            current approved revision
```

A new coder reads `docs/ADF_WORKING_STYLE.md`. To revise: start from the symlink target, create a new dated revision with a revision note, update the symlink, and commit both together.

### 1.3 When to update this document

A recurring process lesson; a new Architect rule; a tooling change (`run_tests.sh`, `install_newest_by_md5`, `env.local.sh`, `phase_tag.sh`); or a ratified decision that changes a workflow. Lessons that live only in chat get re-learned, expensively. **A revision that drops a rule the previous one carried is a defect.** In particular, v1.5 strengthens the historical shell-safety constraint: even a subshell may not run the shell-termination command (§4.2).

---

## 2. Readability is a specification requirement

Do not use project shorthand as if it were ordinary language.

At first meaningful use, write the full term. A shorthand is acceptable after it has been defined, and only if it genuinely helps. This applies to proposals, CRRs, reviews, commit messages and delivered instructions — anything a person must act on or read later.

Illustrative wording — this shows the form, not a record of a real change.

Bad:

```text
Per AD-13, F1–F4 closed; B3.4 unblocked.
```

Better:

```text
Integer and Boolean columns are no longer silently widened when a fill
is applied — the rule ratified as AD-13/13.76.ADF. The four findings
from the previous review round are closed. The legacy draw-path cleanup
(B3.4 in the 13.76 plan) is unblocked.
```

The "Better" text defines every shorthand the first time it uses it, and still reads correctly without the proposal open.

Every substantial proposal or CRR opens with one to four plain sentences: the problem, the intended user-visible result, what changes, and what explicitly does not. A competent reviewer who missed the earlier discussion must be able to understand the document from the document plus its cited sources.

---

## 3. Canonical environment, locations and ownership

Use the existing dfextensions environment. Do not invent parallel variables or locations.

```bash
source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh
```

This also sources the shared installation helpers (§5), so instructions source `env.local.sh` once.

```text
code   /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame
       ADF source, tests, examples, run_tests.sh, long-term docs/
docs   /Users/miranov25/NOTES/alice-tpc-notes/JIRA/O2-6532/docs/AliasDataFrame
       formal phase documents — proposals, CRRs, review summaries,
       handovers — each phase under its own PHASE_*/ subdirectory
```

**Formal phase documents are never committed from the code tree.**

**Canonical test environment:** alma2 — Python 3.10.19, pandas 1.5.3, numpy 1.24.2, aarch64 — via `./run_tests.sh`. A laptop or sandbox result is supporting evidence only.

### 3.1 Ownership inside `examples/time_series/`

That directory holds files with two different owners:

```text
architect-owned
    time_series.py · time_series_draw.py · time_series_TroubleShooting.py
    setup_env.sh · the data files

phase-owned — coder-delivered under the owning phase
    time_series_draw_invariance.py        PHASE_13_77 real-data harness
```

**Architect-owned files may be read, source-read and audited** — and must be, when a review depends on them; the gallery is only trustworthy if reviewers can see what it runs. What a coder may not do is **modify or stage** them without explicit authorization for that task. Explicit authorization for a named task permits the change; it does not transfer ownership.

A new file in that directory states its owner in the CRR that introduces it.

---

## 4. Executable-step format

All formal install, test, git, review, snapshot, commissioning and commit instructions are delivered as **numbered copy/paste steps**, in a `.md` file or message — never `.txt`, preferably not `.sh`.

### 4.1 Each step has a header and a dated marker

**The header** makes the step visible and selectable in the prompt:

```markdown
### STEP3 — Run the focused 13.76 characterization tests
```

**The marker** is the first executed line, and it states the purpose, not only the number — it is what turns a terminal transcript into a record of what was done, when, and why.

In bash:

```bash
echo "[$(date '+%F %T')] STEP3: Run focused 13.76 characterization tests before committing the fix"
(
  if source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh &&
     cd /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame &&
     time python -m pytest tests/test_phase_13_76_draw_path_characterization.py -q; then
    echo "STEP3 PASS: focused characterization tests completed"
  else
    echo "STEP3 STOP: setup or focused tests failed; no further action in this step" >&2
    false
  fi
)
```

The parentheses isolate the step's working directory and environment. The `if` guard runs the test **only if** setup succeeds; a failed test produces a nonzero step result without terminating the Architect's shell. §4.2 requires this form for all delivered shell commands.

In Python, interactive or scripted:

```python
from datetime import datetime
print(f"[{datetime.now():%F %T}] STEP4: Build eager ADF on the 20% sample and check the vector-facet oracle", flush=True)
```

For a script with several steps:

```python
from datetime import datetime
def step(n, what):
    print(f"[{datetime.now():%F %T}] STEP{n}: {what}", flush=True)

step(4, "Build eager ADF on the 20% sample")
step(5, "Compare draw() and draw_batch() statistics for the faceted profile")
```

`flush=True` matters: without it a buffered log can show the marker after the output it introduces. The Python marker matters as much as the bash one — interactive analysis is where most ADF behaviour is exercised, and an IPython history without markers cannot say which cell made which figure.

### 4.2 Binding rules

**Shell safety — NEVER invoke `exit`, anywhere, under any circumstances.** This prohibition includes the main shell, subshells `( … )`, functions, sourced scripts in examples, conditional branches and failure handlers. Do not use `logout`, `set -e` or a top-level `return` as substitutes. The user must never lose their interactive session because of a pasted ADF command.

**Failure gating — a failed prerequisite must prevent every dependent operation in the step.** Use `if` with `&&` and a final `false` in the failure branch. If a step performs several independent actions, report each action's failure explicitly; for destructive, Git, installation or test-dependent steps, use a fail-closed chain.

The required pattern, with an illustrative command:

```bash
echo "[$(date '+%F %T')] STEP3: Verify prerequisites before running the next command"
(
  if source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh &&
     cd /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame &&
     test -f AliasDataFrame.py; then
    echo "STEP3 PASS: required file is present"
  else
    echo "STEP3 STOP: environment, directory or required file unavailable" >&2
    false
  fi
)
```

**Why not use `|| return 1` at top level?** It is not valid outside a function or sourced script. A previous real incident showed that the next command then ran in the wrong directory. **Why not use the former subshell exception?** It was technically less dangerous than terminating the login shell, but it violates the Architect's absolute prohibition. The `if`/`&&` pattern above makes the exception unnecessary.

- The dated marker (§4.1) goes **before** the `(`, so it prints even when the step fails.
- Do not hide a failure with `|| true` or print `PASS` after an unchecked command.
- Any command that must not run after a failed hash check, test, `cd` or staged-diff check stays inside the same successful `if` branch or `&&` chain.
- An exploratory diagnostic may report multiple independent missing files rather than stop at the first; label it clearly and do not perform a dependent commit.

**Self-containment — the Architect restarts sessions and runs steps selectively:**

- Every block sources `env.local.sh` by its **full absolute path** and `cd`s to an absolute path.
- **Never `~/` or `$HOME`** in a path. Write `/Users/miranov25/...` in full.
- Never rely on a variable, directory or function set by an earlier block. Project variables defined by `env.local.sh` are fine, because every block sources it.
- The subshell form fits this rule exactly: nothing a step sets inside `( … )` leaks into the next step, so no step can silently depend on another.

**Form:**

- Number continuously across follow-up messages in one workflow; restart at `STEP1` only for a genuinely new workflow.
- Prefer explicit filenames over temporary shell variables when they fit on screen.
- Prefix slow commands with `time`.
- One purpose per step. A step that installs, tests and commits is three steps.
- Single-command fragments elsewhere in this document illustrate **syntax only**. A delivered instruction always wraps them in a complete STEP: header, dated marker, `source` by absolute path, `cd` by absolute path.

---

## 5. Exact-byte delivery and installation

Candidate files are delivered as **individual files**, each with a declared MD5 and SHA256 — never as an edit script that patches the tree, and never bundled in an archive the Architect must unpack to install. Archives remain correct for **evidence and custody**: reviewer packets (§14) and source snapshots (§6) are archives by design.

Delivery is always **two separate stages**:

```text
INSTALL STAGE   install by MD5, verify with check_ref. For source-code
                 changes, run focused tests and ./run_tests.sh and print
                 the complete summary. NO GIT AT ALL.
COMMIT STAGE    after review only: re-verify, stage, inspect, commit,
                 then check the committed bytes.            (§8)
```

Never mix them, and never ask the Architect to hand-edit a file.

```markdown
### STEP1 — Install the reviewed harness by MD5 and verify its bytes
```

```bash
echo "[$(date '+%F %T')] STEP1: Install reviewed A2 harness by MD5 and verify, no git"
(
  if source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh &&
     cd /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame &&
     install_newest_by_md5 \
       'time_series_draw_invariance_a2_v03*.py' \
       'examples/time_series/time_series_draw_invariance.py' \
       '<MD5>' &&
     check_ref \
       'examples/time_series/time_series_draw_invariance.py' \
       '<MD5>' '<SHA256>'; then
    echo "STEP1 PASS: reviewed harness installed and verified"
  else
    echo "STEP1 STOP: source, install or byte verification failed" >&2
    false
  fi
)
```

Rules:

- The first `install_newest_by_md5` argument is a **glob relative to `$Downloads`** — do not prepend `$Downloads/`, and do not rely on browser suffixes such as `(1)`. The glob plus the MD5 select the right bytes.
- The verified installed helper **keeps** matching Downloads files by default; deletion happens only if `--delete-matching` is explicitly supplied. Do not request that option in delivered installation steps. The installed destination, not the Downloads copy, is authoritative after `check_ref`.
- Install one reviewed unit, verify it, run its focused checks, then install the next.
- **Never assume a delivered file is already in place.** If it arrived through Downloads, chat or a reviewer, start with install and `check_ref`.
- A file created directly in its canonical location needs no install step; verify its bytes by the applicable hash instead.

**The index is not the working tree.** After an install, `git status` may show `A`, `AM` or `M` depending on existing index state. The index can still hold previously staged bytes. Verify the **working tree** with `check_ref`.

**Documentation-only delivery:** Install formal Markdown files **directly into** the appropriate `PHASE_*/` subdirectory of the formal documentation repository (§3, §14), one exact named file per helper invocation. Verify each with `check_ref`. Documentation-only changes require document/content inspection and a whitespace check before commit; they do not require running ADF scientific tests when no code or oracle changed. Installation and Git actions remain separate stages.

### 5.1 Hashes must match everywhere

The MD5 in the CRR, the bytes in the reviewer packet, and the values in the commit step's `check_ref` must be **the same**. A reviewer packet must contain **every** candidate file the CRR declares.

This is not bookkeeping. In PHASE_13_76 B3.2b, a CRR declared "no production change" while the commit it led to changed `AliasDataFrame.py`. Matching hashes across the three documents is what catches that.

### 5.1a Base hashes for shared files

When a CRR changes a shared or high-risk file — `AliasDataFrame.py` above all — it records the **base commit and the base MD5** of that file it started from. Before delivery, re-check that the file in the tree still has that base MD5. If it does not, another coder changed it in the meantime and the delivery is rebased, not overwritten.

### 5.1b Commit approved bytes before stacking

Once an increment's bytes are approved, **commit them before starting the next semantic increment** on the same files. Stacking a second change on uncommitted approved bytes means the next review cannot tell which change produced which result, and a rollback loses both.

### 5.2 Installer backups

The installer saves the previous file as `*.bak.<timestamp>`. Those files are never staged (§8).

Remove **only the backups your own install step just created, named explicitly**. Never remove backups by pattern: in a shared tree, another coder's `AliasDataFrame.py.bak.*` may be the only copy of their uncommitted work.

---

## 6. Source custody and snapshots

Reviews depend on knowing exactly which bytes were reviewed. Snapshots are produced by `scripts/make_audit_snapshot.sh` and carry `_AUDIT_*` metadata.

**Run it; never `source` it.** It is an executable workflow, not an environment script. Its arguments are the package directory, the package name and the output directory; a fourth argument, a git ref, produces a clean snapshot of a committed ref instead of the working tree:

```text
../scripts/make_audit_snapshot.sh ../AliasDataFrame adf .
```

It writes three containers holding the same file list: a **ZIP**, a **TAR**, and a plain-text **LLMBUNDLE**. The ZIP owns the membership — the LLMBUNDLE is built **from** the completed ZIP, and the tool verifies the containers hold the same member count. A reviewer citing the LLMBUNDLE quotes its entry count and manifest SHA256 from **both** its header and its trailer, declares all three container hashes, and states which container was opened.

**Fail closed.** If the tool reports a member-count mismatch, or a file declared but absent, that snapshot is not used for review.

```text
CLEAN snapshot of a committed ref
    the commit is authoritative.

DIRTY worktree snapshot
    the snapshot bytes are authoritative.
    REPO_HEAD is generation context only — not the source.
```

Always record: `SOURCE_COMMIT`, CLEAN or DIRTY, the ZIP / TAR / LLMBUNDLE identity where applicable, and key-file hashes. When the snapshot tool reports untracked `.py` files missing from the snapshot, **disclose them** — a test file absent from the snapshot is a test no reviewer ran.

A phase freezes its reference snapshot at phase start and does not recut it mid-phase without saying so in the CRR.

---

## 7. Git inspection — never flat

ADF lives inside the larger O2DPG repository, and other coders work in the same tree. Git inspection is therefore always scoped.

**Never use repository-global `git status` for ADF work.**

```bash
git status --short .
```

or, for a reviewed set of files:

```bash
git status --short -- path/to/file1 path/to/file2
```

Never, when the intent is the current work area:

```text
git status
git status --short
git status --porcelain
```

**Stage explicitly, then inspect.** The order matters:

```bash
git add -- path/to/file1 path/to/file2
git diff --cached --check -- path/to/file1 path/to/file2
git diff --cached        -- path/to/file1 path/to/file2
git diff --cached --name-only -- .
```

**Why `git add` must come first.** `git commit --only -- <paths>` records the **working-tree** bytes of those paths. `git diff --cached` shows the **index**. Without a fresh `git add`, the index can still hold an older version, and the inspection then shows different bytes from those committed. Demonstrated before v1.3: with an older version staged, `git diff --cached` showed `v1-STAGED` while the commit recorded `v2-REVIEWED`. After an explicit `git add -- <paths>`, the inspection and the commit showed the same bytes. v1.2 of this document omitted the `git add`; it was wrong.

The second and third lines prove the intended paths are right. **Only the fourth shows an unrelated path someone else staged** — demonstrated: it listed another coder's staged file alongside the intended one. `--only` on the commit (§8) still keeps that file out of the commit; this check is how you see it is there.

`ls -l` is for orientation only; it is never an integrity check.

---

## 8. Commit discipline — never flat

**The commit command itself must show exactly which files are committed.** Never use a plain `git commit` that takes whatever happens to be staged.

**Commit-message quality is a binding requirement for every commit, including documentation-only commits.** Use an informative, non-cryptic subject and a readable multiline body. The subject should name the concrete change and its effect, not merely version numbers or process labels. The body explains **what changed, why it was needed, what evidence or review status supports it, and what deliberately did not change**. Distinguish *proposed* documents from *ratified* decisions and *reported reproduction* from *canonically executed* evidence.

The standard command is `git commit --only -F - -- <explicit paths>` with a here-document (`<<'COMMIT_MESSAGE'`). Do not replace it with a short collection of `-m` flags when writing formal commits. The step assumes the exact paths have already been explicitly staged and inspected (§7), and that pre-commit hash checks passed (§5).

**Illustrative example — the change described below is fictional and is not evidence of an actual fix:**

```bash
echo "[$(date '+%F %T')] STEP7: Commit reviewed integer fill-contract fix by explicit paths"
(
  if source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh &&
     cd /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame; then
    if git commit --only -F - -- \
      AliasDataFrame.py \
      tests/test_phase_13_76_draw_path_characterization.py <<'COMMIT_MESSAGE'
AliasDataFrame: preserve integer column types when applying missing-value fills

Previously, filling a missing integer value widened the column to a
floating-point type. A subsequent equality join could then compare
incompatible keys and silently produce incorrect matches.

Preserve the integer dtype and add an injected-truth regression test
covering missing rows. The focused regression and canonical test suite
were completed on the reviewed source bytes.

The public draw interfaces and dfdraw implementation were not changed.
COMMIT_MESSAGE
    then
      echo "STEP7 PASS: reviewed source and test committed"
    else
      echo "STEP7 STOP: commit failed; inspect before retrying" >&2
      false
    fi
  else
    echo "STEP7 STOP: environment or repository unavailable" >&2
    false
  fi
)
```

The subject names the user-relevant behavior; the body explains the mechanism, independent evidence, and intentional limits. A real commit must accurately describe **what was actually done and verified**, with the evidence status from §10. Never claim a test passed solely because it was planned.

Binding rules:

- **Every commit names its paths after `--`, and uses `--only`.** A pathspec commit records the working-tree bytes of exactly those paths and leaves everyone else's index untouched. A bare `git commit` in a shared tree can commit another coder's stale staged work under your message.
- **Never** `git add -A`, `git add .`, or `git commit -a`. A brand-new file is introduced explicitly with `git add -- <new-file>`.
- **Never** `--amend` in the reviewed workflow. A correction is a new commit.
- **Commit subjects are meaningful, human-readable descriptions.** Spell out the actual capability or defect before using an abbreviation; do not substitute a cryptic stage name, review number, phase ID or generic "update" as the subject.
- **Use an explanatory multiline commit body.** State what changed; why it mattered; relevant evidence, status, or reviewers' conclusions; and what did **not** change. For document-only work, identify the actual documents and whether the content is *proposed*, *approved* or *historical*.
- **No unsupported claims.** An example in a commit message, CRR or this document is either true or labelled illustrative. v1 presented an open dfdraw defect as an ADF fix; v1.2 replaced it with a description of a fix whose details could not be verified. Both are the "defect in the sentence" of §16.

### 8.1 Check the committed bytes

```bash
git show HEAD:./AliasDataFrame.py | md5sum
```

The `./` resolves the path from the current directory rather than from the O2DPG repository root. If the MD5 does not match the reviewed value, **stop and report**. Never amend.

### 8.2 What is never committed

`*.bak.*` files, logs, reviewer packets, gallery exports, and unrelated phase work. **Capability matrices** (`docs/CAPABILITY_MATRIX.{md,html,json}`) are generated from the taxonomy, generators and tests: they are not committed at intermediate checkpoints; Markdown and HTML are published only at phase closure; JSON is not committed at all unless a repository consumer appears — it travels in reviewer packets.

**Unrelated dirt is recorded and protected, never cleaned up** to simplify the current phase.

---

## 9. Parallel work on a shared tree

Several coders may work in one working tree at once — for example one on `PHASE_13_76` production code and one on the `PHASE_13_77` real-data harness.

- **One holder per file at a time.** If two phases need the same file, agree in writing who holds it before the first delivery. Otherwise the last writer wins silently.
- **Prefer making the rule structural.** A phase that must not change production code says so, and then never delivers that file — stronger than coordinating around it.
- **Each phase owns its own test file.** Never add a test to another phase's file. A phase's strict-xfail ledger is its closure criterion — *no strict xfail names `<stage>`*, answerable by one `grep` — and a foreign test in that file corrupts the answer.
- **Evidence from a shared tree is attributable only if the delta is.** A run carrying another coder's uncommitted change cannot say whose change caused a new failure. Either the other change is committed first, or the run is labelled non-authoritative and a clean rerun is owed before closure.

---

## 10. Evidence that counts

| Evidence | Why |
|---|---|
| **Failure-identity diff against a named baseline** | Counts drift for innocent reasons. Compare the set of failing test *names* against a named, recorded baseline — never "the most recent run", or a flapper that happened to be green manufactures a regression. |
| **Departures count too** | A test that **leaves** the failing set is a change and must be explained. A baseline records the dfdraw byte it ran with as well as the ADF one; otherwise a dfdraw change looks like an ADF fix. |
| **Every strict xfail run with `--runxfail`** | The recorded failure must match the reason string. *"They all xfail"* and *"they all xfail for the stated reason"* are different claims. |
| **Measured on the canonical stack first** | Any claim about pandas or numpy behaviour is measured on alma2's pandas 1.5.3 **before** it is written. Two P0s in this project came from claims verified on the wrong version. |
| **Registered flappers** | Known timing-threshold and peak-RSS tests that flip independently of code changes are listed by name, and are never claimed as a fix when they turn green. |
| **Every result labelled** | `EXECUTED`, `SOURCE-PROVED`, `REASONED` or `NOT RUN`. A reasoned claim stated as if executed is the most common way a review approves a defect. |

**The canonical final gate is `./run_tests.sh`.** Focused tests are necessary but not sufficient for a final CRR or closure. Record passed, failed, errors, skipped, xfail, xpass and exit code. A reduced or fast lane does not replace this gate unless the active approved proposal says so explicitly.

**Print the complete summary.** `cat` the whole `SUMMARY_*.txt`; never substitute `tail` or a selected `grep`. A filtered summary hides exactly the line nobody thought to look for.

**Real-data gallery evidence is durable evidence.** For draw-path and oracle changes, the real-data gallery run — its manifest, its PDF and its log, each identified by hash — is part of the evidence, not a demonstration. When loading the data dominates the run time, run the **full applicable gallery** rather than a hand-picked subset: once the data is loaded, the additional cases cost little, and a subset chosen by the coder tests what the coder expected to break.

---

## 11. Oracle discipline

A test suite proves implementation consistency. **It does not prove scientific correctness.** All three public drawing surfaces can share one wrong implementation and agree perfectly.

- **Prefer independent or injected-truth oracles.** Injected truth — data built so the answer is known before the code runs — is the only class that catches every surface being wrong the same way.
- **Distinguish product defect, oracle defect, and unknown.** `UNKNOWN` is never equivalent to `PASS`.
- **Attribution is measured, not declared.** For a red case, record the measured value at each layer — independent formula, ADF materialized output, direct dfdraw on plain data, full ADF surface — and name the first layer that disagrees. A declared owner is a prediction the measurement can falsify, not evidence. In the 13.77 real-data oracle, four of seven reds first classified as dfdraw defects turned out to be oracle defects.
- **Check precision on both sides.** A float16 source against a float64 reference, or a weighted mean accumulated in float32 over millions of rows, produces a real disagreement that is not a product defect. Record the dtype of both sides.
- **Never weaken an oracle because product code fails.** A tolerance is **derived** — from dtype, row count and value scale, with the derivation recorded — never picked or widened until the test passes.
- **Every guard has a negative control, paired with a positive one.** A guard that should refuse something gets a test proving it refuses — `normalize=` on a histogram once vanished silently because no test checked the refusal. The paired positive control proves the guard does not refuse everything.
- **An oracle must be seen failing.** Before a gate is relied on, break what it guards and confirm it goes red for the stated reason.
- **For plots, check both** the machine-readable result and the rendered structure, and tag each gallery page by proof class — `PRIMARY ORACLE`, `CONSISTENCY ORACLE`, or `SMOKE / COVERAGE` — so a reviewer knows what to verify on it.

---

## 12. Review procedures

Reviews run falsification procedures, not only a check of the CRR's claim list. A seat that verifies the claims it was handed tends to approve; the defects in this project were almost always in a case the CRR never named.

**Every review opens with a source-read declaration:** what was read, what was executed, and **the reference bytes it ran against** — `AliasDataFrame.py` MD5, `drawer.py` MD5, snapshot `SOURCE_COMMIT`. A finding has nearly been discounted as "run on stale bytes" when it had in fact been executed on the exact candidate; the declaration settles that.

**Four procedures as a floor, plus a mandatory fifth:**

```text
1  TAUTOLOGY      Can a test pass while proving nothing? A getattr with a
                  default that hides a missing field; a fixture that never
                  reaches the branch it names; an assertion in a dead branch.
2  LIFECYCLE      create -> read -> mutate -> delete -> recreate. State
                  that survives the deletion of its subject is a defect.
3  SCOPE          For each obligation of the phase: closed, open with a
                  named owner, or unowned? Unowned is a finding.
4  PUBLIC PATH    Enumerate the documented API paths reaching the new state
                  — from the API surface, not from the CRR's own list.
5  FREE ATTACK    Two hypotheses the CRR did not name, on two dimensions.
```

Every result is labelled (§10). A seat may file a **question** rather than a finding when it lacks the evidence to decide — filing uncertainty as a question is correct; asserting it resolved is not.

### 12.1 Roles and approval

```text
coder            delivers the increment and its CRR.
                 NEVER approves its own work — including its own documents.
reviewer         independent of the coder; reviews the exact candidate bytes.
Main Reviewer    writes ONE review summary that disposes EVERY finding:
                 accepted, rejected with reason, or deferred with an owner.
Architect        rules on decisions; approves closure and phase tags.
```

The individual reviews are kept for provenance; **the summary is the decision record**. Two reviewers disagreeing on a P0 is escalated to the Architect, not averaged.

### 12.2 Correction reviews carry findings forward

A review of a corrected increment first lists **every finding from the previous round** with its status, and only then any new findings, separately:

```text
CLOSED       fixed, and the fix was verified
OPEN         not fixed — still blocking or still owed
DEFERRED     moved to a named later step, with an owner
SUPERSEDED   no longer applies, with the reason
```

A finding that silently disappears between rounds is the commonest way a defect survives review.

---

## 13. CRR expectations

A CRR opens with the plain-language summary of §2, then states:

- the exact files with MD5 and SHA256, and the baseline commit;
- what changed, and what explicitly did not;
- the evidence, labelled per §10, including the failure-identity diff;
- **disclosures** — any superseded test, any finding from a previous round not fixed, any near-miss caught by re-execution;
- **reuse** — for every new helper, the existing function it reused, or why none fit;
- what to attack — the places the coder is least sure of.

**Reuse before create.** Before writing a helper, census the existing ones. Two incidents in this project began as well-tested new code written beside an existing owner of the same fact — and two producers of one fact eventually disagree (§16).

**Naming a hole is not closing it.** If a CRR says *"I did not check X"*, either check X or record it as an open item with an owner. Writing it as an attack point and shipping anyway has twice turned a fixable gap into a shipped defect.

**A superseded test is disclosed loudly**, with the reasoning in its own docstring. A correction and a weakening look identical in a diff.

**A question is not an authorization.** When the Architect asks *"could X be done?"* or *"why does Y happen?"*, answer it. Do not edit code until the change is explicitly requested.

**An exploratory probe is not an acceptance gate.** A quick script written to understand a behaviour may be thrown away. If its finding becomes a claim, the claim is proved by a test in the suite, not by the probe.

**Candidate files have meaningful, unique names** — `time_series_draw_invariance_a2_v03.py`, not `fix.py` or `new_version.py` — so a download, a CRR line and a commit can be matched by name as well as by hash.

---

## 14. Review packets and documents

`./run_tests.sh` produces `test_logs/reviewer_<timestamp>.zip`: full, focused and `--runxfail` logs, the MD5 manifest for working tree **and** staged blobs, package versions, and diffs. **That zip is the evidence a review cites** — not a pasted summary. Do not hand-roll a packet when the standard one exists.

**Formal phase documents are installed directly into the formal documentation repository** at `/Users/miranov25/NOTES/alice-tpc-notes/JIRA/O2-6532/docs/AliasDataFrame/PHASE_*/`, using `install_newest_by_md5` with the exact named destination followed by `check_ref` on MD5 and SHA-256 (§5). A temporary handoff under `files_ADF/` is optional for reviewer distribution, not a required staging step before permanent installation. Once approved, commit only the explicitly named Markdown paths in the documentation repository, using the human-readable multiline format of §8. Never mix documentation files into the ADF source-code commit.

**For several available documents**, attempt each exact-MD5 install separately, print `SAVED` or `NOT AVAILABLE` for every document, and then verify all available destinations as a distinct step. A missing optional document must not prevent other independent files from being preserved; it **does** prevent claiming that an intended complete review packet was installed or committed.

### 14.1 Reviewer handoff staging in `files_ADF/`

`files_ADF/` is the **transient reviewer-handoff directory inside the ADF working tree**. It exists so a reviewer receives one coherent set of CRRs, review documents and evidence without hunting across Downloads and `test_logs/`. It is **not** the permanent formal-docs repository of §3, and its contents are never included in the code commit.

Use the source of an artifact to decide how it enters `files_ADF/`:

```text
artifact arrived through ChatGPT / browser Downloads
    -> install_newest_by_md5 into files_ADF/<exact target name>
    -> check_ref the installed MD5 and SHA256

artifact was already produced by the repository itself
(e.g. test_logs/reviewer_<timestamp>.zip, SUMMARY_*.txt, gallery manifest/PDF/log)
    -> copy it directly with cp -p
    -> do NOT download it again and do NOT run it through the installer
```

The distinction is a custody rule. `install_newest_by_md5` proves which downloaded bytes were selected. Repository-local evidence already has a canonical path and was produced by the run being reviewed; copying it preserves that provenance instead of manufacturing a second download/install lineage.

**Path rule.** From the AliasDataFrame repository root the handoff directory is exactly `files_ADF/`. Do not invent `../files_ADF`, a sibling cache, or a second reviewer directory. If a command is run elsewhere, first `cd` to the absolute ADF root as §4 requires.

**Collision rule.** Generic gallery names such as `fast_manifest.json`, `fast_gallery.pdf` and `fast_gallery.log` are renamed on handoff to include the phase/checkpoint and run timestamp. A later run must never silently overwrite an earlier reviewer artifact. The canonical run directory under `test_logs/` remains unchanged.

**Reviewer-packet rule.** The standard `run_tests.sh` reviewer ZIP is copied as produced. Verify its MD5/SHA256 against `reviewer_digests_<timestamp>.txt` or the printed run summary; do not rebuild a second packet for the same run. The ZIP remains the evidence packet defined by §14.

**Backup rule.** Installing downloaded Markdown into `files_ADF/` may create `*.bak.<timestamp>`. Remove only backups created by the current install, and name them explicitly. Never run a broad `rm files_ADF/*.bak.*` cleanup in the shared tree.

**Formal-document rule.** A CRR or review may temporarily appear in `files_ADF/` for reviewer distribution. Permanent approved phase documents are installed **directly** in their canonical `PHASE_*/` directory within the formal docs repository, with `install_newest_by_md5` and `check_ref` (§5). The temporary `files_ADF/` copy is never the publication record; the documentation-repository commit is.

A normal handoff contains only the artifacts relevant to that review round, for example:

```text
files_ADF/
    PHASE_..._CRR_....md
    PHASE_..._OfficialReview_....md
    reviewer_<timestamp>.zip
    SUMMARY_<timestamp>.txt
    PHASE_..._fast_manifest_<timestamp>.json
    PHASE_..._fast_gallery_<timestamp>.pdf
    PHASE_..._fast_gallery_<timestamp>.log
```

Illustrative delivery form:

```markdown
### STEPn — Collect the approved CRR and existing run evidence for reviewer handoff
```

```bash
echo "[$(date '+%F %T')] STEPn: Collect reviewed CRR and existing test evidence in files_ADF"
(
  if source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh &&
     cd /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame &&
     mkdir -p files_ADF &&
     install_newest_by_md5 \
       'PHASE_example_CRR_v01*.md' \
       'files_ADF/PHASE_example_CRR_v01.md' \
       '<MD5>' &&
     check_ref \
       'files_ADF/PHASE_example_CRR_v01.md' \
       '<MD5>' '<SHA256>' &&
     cp -p 'test_logs/reviewer_YYYYMMDD_HHMMSS.zip' files_ADF/ &&
     cp -p 'test_logs/SUMMARY_YYYYMMDD_HHMMSS.txt' files_ADF/; then
    echo "STEPn PASS: named handoff documents and evidence collected"
  else
    echo "STEPn STOP: one or more handoff requirements failed" >&2
    false
  fi
)
```

This is an illustrative example: replace the placeholder digests, timestamped evidence paths and sample document name with the exact reviewed identities. It intentionally does not `git add` anything. Handoff collection and source or documentation commits are separate operations.

---

## 15. Phase tags

A tag is a permanent name for one commit. ADF uses tags to mark where a phase starts, each reviewed checkpoint inside it, and where it closes, so any reviewer can recover the exact bytes a decision was made on.

```text
PHASE_13_NN_ADF_BEGIN
    the commit the phase's reference source snapshot was taken from —
    the custody anchor every increment is diffed against.
    Set once, when implementation starts.

PHASE_13_NN_ADF_GATE_<increment>          e.g. PHASE_13_76_ADF_GATE_B3_2
    a commit whose bytes passed review and match the CRR hashes.
    Means "this checkpoint was reviewed". Never means "closed".

PHASE_13_NN_ADF_END
    closure. Set only after: final reviewed bytes are committed, the
    closure review is approved, the PHASE_HISTORY.md row is added, and
    the registry exit is recorded.
```

**Naming.** `_END` is the established ADF form — `PHASE_HISTORY.md` records it for 13.26, 13.27, 13.35, 13.36 and 13.56. `_GATE_<increment>` was used in 13.76. `_BEGIN` matches the dfdraw lifecycle and is new for ADF.

**Rules:**

- **Create tags only with `scripts/phase_tag.sh`**, never with a hand-typed `git tag`, so names stay consistent across teams. The wrapper sets `SUBPROJECT="AliasDataFrame"` and `TAGSUFFIX="ADF"`; all logic, including the subcommands, lives in `/Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/scripts/phase_tag_common.sh`. This document deliberately does not copy those subcommands — two copies of one command list would drift.
- **Annotated, never lightweight.** The tag message names the CRR or review summary and its verdict.
- **A tag points at reviewed bytes.** Verify it against the CRR's MD5 before relying on it.
- **Tags are never moved or deleted.** A wrong tag is superseded by a new one, as a wrong commit is corrected by a new commit. A moved tag also diverges silently between clones: the laptop, alma2 and GSI each keep whichever version they first fetched.
- **"Latest progress" is derived, not a moving tag** — it is the newest `_GATE_` tag.
- **A tag never means more than its name.** Only `_END` means closed.

```markdown
### STEPn — Verify a phase tag before relying on it
```

```bash
echo "[$(date '+%F %T')] STEPn: Verify GATE_B3_2 tag and the reviewed AliasDataFrame.py"
(
  if source /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame/scripts/env.local.sh &&
     cd /Users/miranov25/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame &&
     git cat-file -e PHASE_13_76_ADF_GATE_B3_2; then
    git tag -l 'PHASE_13_76_ADF_*' --sort=creatordate
    git cat-file -t PHASE_13_76_ADF_GATE_B3_2
    git show PHASE_13_76_ADF_GATE_B3_2:./AliasDataFrame.py | md5sum
    git tag -l 'PHASE_13_76_ADF_GATE_*' --sort=-creatordate | head -1
  else
    echo "STEPn STOP: repository or requested tag unavailable" >&2
    false
  fi
)
```

`git cat-file -t` must print `tag` — `commit` means a lightweight tag. The `md5sum` must equal the CRR's value. The last line is the derived "latest progress".

**Ratified: there is no movable `PROGRESS` tag.** A movable tag was proposed and rejected by every reviewer who addressed it: git does not update an existing tag on fetch, so a moved tag silently shows different commits on the laptop, alma2 and GSI. Latest progress is the newest `_GATE_` tag, derived as above.

---

## 16. Recurring defect classes

The shapes that produced most findings across 13.76 and 13.77. Check for them deliberately.

- **Declared state nobody reads.** A field validated and serialised but never consulted by any decision is documentation wearing a schema's clothes.
- **The mechanism built to close a class is itself an instance.** An audit, allow-list or guard added to stop a defect class is checked as strictly as the code it guards.
- **Two producers of one fact.** If two functions compute overlapping facts, they will eventually disagree. Derive every view from one authority.
- **A test that encodes a defect** is worse than a missing test: it turns the defect into a protected property.
- **A test that cannot reach its branch.** Assert that the mechanism *fired* — mask survived, branch entered — not only that the outcome matched.
- **A criterion with two halves gets both halves mutated.** Proving one half works says nothing about the other.
- **The empty case.** A gate over a declared set must refuse an empty set; zero cases is not success.
- **A name is not a type.** Static analysis matching by variable name, across scopes, certifies the wrong things.
- **The defect is usually in the sentence, not the line.** The claim in a docstring, comment, CRR or example outruns what the code does.

---

## 17. Examples are documentation

Files under `examples/` are user-facing example code as well as commissioning and oracle infrastructure. When an example changes:

- the public ADF / dfdraw query stays easy to find;
- comments stay scientifically useful;
- bulky oracle or diagnostic machinery moves into the harness or a helper;
- the CRR asks reviewers whether the example is still understandable and usable as an example.

---

## 18. Handover discipline

A handover is written when a coder or model is replaced, a phase is interrupted, context has grown too large, work moves temporarily to another phase, or there is protected uncommitted work. It is named `CODER_HANDOVER_ADF_<date>.md` and committed to the docs repository.

It contains only **state**, never standing rules — those live here:

```text
who                 outgoing and incoming seat identifiers
what governs        the controlling proposal, and the ADs in force — full form
where               current branch and HEAD; the phase's BEGIN and latest GATE tags
which bytes         canonical source snapshot identity (ZIP / TAR / LLMBUNDLE
                    hashes); latest reviewer packet identity
history             PHASE_HISTORY.md status, and whether it is current
the tree            exact modified and untracked files, with hashes;
                    protected unrelated work
decided             approved scope and decisions already made
do not reopen       items already ruled on, with where they were ruled
evidence            tests already run, with the named identity baseline
                    (ADF byte AND dfdraw byte)
state               current RED/GREEN state, each red with its measured
                    first disagreement
next                the next smallest implementation step, and the exact
                    set of files it will touch
forbidden           what must NOT be changed
```

The **do-not-reopen** list matters as much as the open list. Without it, an incoming seat re-litigates settled decisions, and each one costs a review round.

Do not rely on the next coder reconstructing state from chat history.

---

## 19. When to ask the Architect

Ask in plain text, with the options listed inline so the answer can be typed — never with a multiple-choice widget:

- a decision that changes ratified scope, or defers a ratified item;
- a fix that would require production code the phase may not touch;
- two reviewers who disagree on a P0;
- any action outside the agreed file set;
- creating or superseding a phase tag.

State what you would do and why, so the Architect can approve rather than decide from scratch. Ask before acting, not after.

---

## 20. Short operational checklist

```text
before delivering
  [ ] plain-language opening; every shorthand defined at first use
  [ ] each file individually, with MD5 and SHA256
  [ ] install stage has NO git
  [ ] every step: ### STEPn header + dated marker (bash and python)
  [ ] no shell `exit` command anywhere, including inside subshells
  [ ] guard dependent operations with `if` + `&&`; failure branch reports + `false`
  [ ] no top-level `return`, no `logout`, no `set -e`, no hidden failure
  [ ] full absolute paths; never ~/ or $HOME
  [ ] results labelled EXECUTED / SOURCE-PROVED / REASONED / NOT RUN
  [ ] CRR hashes == packet bytes == check_ref values; every file in packet
  [ ] shared files: base commit + base MD5 recorded, re-checked before delivery
  [ ] previous increment's approved bytes already committed

before committing
  [ ] check_ref on the working tree, not the index
  [ ] git status --short .                       never flat
  [ ] git add -- <paths>                         FIRST, then inspect
  [ ] git diff --cached -- <paths>               now shows the committed bytes
  [ ] git diff --cached --name-only -- .         the WHOLE staged set
  [ ] git commit --only -F - -- <paths>          never flat
  [ ] git show HEAD:./<path> | md5sum            after the commit
  [ ] commit subject describes the actual change, not a cryptic label
  [ ] multiline commit body explains what/why/evidence/status/not-changed
  [ ] no *.bak.*, logs, matrices, packets, exports staged

before handing evidence to reviewers
  [ ] temporary handoff: downloaded artifacts -> install_newest_by_md5 + check_ref into files_ADF/
  [ ] formal docs: downloaded Markdown -> install directly in formal PHASE_*/ + check_ref
  [ ] repository-local test_logs evidence -> cp -p directly; never re-download/reinstall
  [ ] generic gallery filenames renamed with phase/checkpoint + timestamp
  [ ] reviewer ZIP digest checked against the producing run
  [ ] only current-install backups removed, named explicitly
  [ ] files_ADF/ treated as transient handoff, never as the permanent docs repository

before closing a review round
  [ ] every previous finding: CLOSED / OPEN / DEFERRED / SUPERSEDED
  [ ] the coder has not approved their own work

before claiming a red is someone else's
  [ ] first disagreeing layer MEASURED, not declared
  [ ] dtype and precision recorded on both sides
  [ ] baseline records both the ADF and the dfdraw byte
```

---

## Working principle

Make every result traceable: which bytes, which step, which command, when, and why. A finding that cannot be reproduced from the record is not a finding, and a commit that cannot be understood a week later is not finished.
