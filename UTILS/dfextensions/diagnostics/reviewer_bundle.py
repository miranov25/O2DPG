#!/usr/bin/env python3
"""
reviewer_bundle.py - dfextensions/diagnostics: build the reviewer packet zip.

One command produces everything a reviewer needs (the CRR reviewer-instruction
recipe is the entry point inside):

  python3 diagnostics/reviewer_bundle.py [-o OUT.zip] [--evidence DIR ...]

Contents: code/ (all modules + collector script), tests/, docs/ (README,
run recipe), logs/ (fresh run_tests.sh output), provenance/
(git log for the phase, file MD5 MANIFEST), plus any --evidence directories
(rendered reports, real bundles, validation/) copied verbatim.
The zip's own MD5 is printed last - cite it when distributing (AD-10).
"""
from __future__ import annotations
import argparse
import hashlib
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

PHASE_BASELINE = os.environ.get(
    "DFX_REVIEW_BASELINE",
    "e0c3f0891cf856a61af7203c90125364fc01e0dd",
)
PHASE_CANDIDATE_FILES = (
    "dfx_host_diagnostics.sh",
    "collector.py",
    "query_diagnostics.py",
    "run_tests.sh",
    "reviewer_bundle.py",
    "tests/test_dfx_host_diagnostics.sh",
    "tests/test_process_catalog.py",
    "tests/test_query_diagnostics.py",
)

CODE = ["dfx_host_diagnostics.sh", "collector.py", "schema.py", "audit.py",
        "report_diagnostics.py", "run_metrics.py", "explain_bundle.py",
        "dfx_run_with_diagnostics.py", "run_tests.sh",
        "job_host_analysis.py", "conclusion_model.py",
        "capabilities.py",              # registry: lets a reviewer reproduce T-P9 from the packet
        "query_diagnostics.py",         # PHASE_13_75 DuckDB/SQL query layer
        "reviewer_bundle.py"]
DOCS = ["README.md", "docs/CAPABILITY_MATRIX.md", "docs/DIAGNOSTICS_WORKING_STYLE.md"]
# LEDGER-PKT [architect ruling AR-4, 2026-07-22]: the completion ledger is
# carried by EVERY reviewer packet until the phase closes, so a reviewer never
# has to reconstruct the remaining-work state from conversation history.
LEDGER_NAMES = ["PHASE_13_74_ADF_v8_Completion_Ledger_for_CRR_v4_Rev2.md",
                "PHASE_13_74_ADF_v8_Completion_Ledger_for_CRR_v4_Rev1.md"]  # Rev2 preferred; Rev1 fallback

# Governance documents that make the packet a COMPLETE, self-contained review
# request - so a reviewer never reconstructs the "why / what / status" from
# conversation history [architect: reviewer packet must carry everything needed
# for review automatically]. Each is auto-included if found; a missing one is
# noted, not fatal. Search order: --docs-dir (the GitLab docs tree), then the
# subproject dir and its parent. Filenames are prefix-matched newest-first so a
# later Rev supersedes an earlier one without editing this list.
GOVERNANCE_DOCS = [
    ("PHASE_13_74_ADF_MOTIVATION",              "the inherited WHY - motivation, goal, non-goals"),
    ("PHASE_13_75_DIAG_Proposal",               "the current PHASE_13_75 normative specification"),
    ("PHASE_13_75_DIAG_ProcessCatalog_DuckDB_v", "the latest authoritative consolidated review summary"),
    ("PHASE_13_75_DIAG_GSI_Commissioning_Evidence",
                                                  "the current GSI commissioning evidence"),
    ("PHASE_13_75_DIAG_ProcessCatalog_DuckDB_Commissioning_PRECOMMIT_CRR",
                                                  "the current Process Catalog + DuckDB CRR"),
    ("PHASE_13_75_DIAG_Increment",              "other current PHASE_13_75 increment CRR, if present"),
    ("PHASE_13_74_ADF_v8_Proposal",             "the predecessor specification retained for transferred scope"),
]
# Phase documents live in the GitLab NOTES tree (repo split: code on GitHub,
# documents on GitLab). Override with --docs-dir.
DEFAULT_DOCS_DIR = os.environ.get(
    "DIAGNOSTICS_DOCS",
    "/Users/miranov25/NOTES/alice-tpc-notes/JIRA/O2-6532/docs/diagnostics",
)


def _newest_match(dirs, prefix):
    """Newest file whose name starts with `prefix` and ends in .md, across dirs."""
    hits = []
    for d in dirs:
        d = Path(d)
        if d.is_dir():
            hits += [f for f in d.glob(f"{prefix}*.md") if f.is_file()]
    return max(hits, key=lambda f: f.stat().st_mtime) if hits else None


import getpass
import re

_IDENTS = [getpass.getuser(), os.path.expanduser("~")]
_TEXT_EXT = {".md", ".txt", ".log", ".json", ".kv", ".csv", ".html", ".py", ".sh"}


def _scrub(data, name):
    """Replace the packaging user's identity and home path in text payloads
    [panel P0-5: privacy holds for the COMPLETE packet, not one directory]."""
    if Path(name).suffix not in _TEXT_EXT:
        return data
    try:
        t = data.decode("utf-8")
    except Exception:
        return data
    home = os.path.expanduser("~")
    repo = str(Path(__file__).resolve().parents[1])   # .../dfextensions
    t = t.replace(repo, "~repo").replace(home, "~")
    u = getpass.getuser()
    t = re.sub(rf"\b{re.escape(u)}\b", "[user]", t)
    return t.encode("utf-8")


def _leak_scan(entries):
    """RETAINED BUT NOT INVOKED - see _scrub() [Decision 3, 2026-07-23]."""
    hits = []
    u = getpass.getuser()
    home = os.path.expanduser("~").encode()
    repo = str(Path(__file__).resolve().parents[1]).encode()
    for name, data in entries:
        if Path(name).suffix not in _TEXT_EXT:
            continue
        if re.search(rf"\b{re.escape(u)}\b".encode(), data) or home in data \
                or repo in data:
            hits.append(name)
    return hits


def _git(repo_root, *args, check=True):
    cp = subprocess.run(
        ["git", *args], cwd=repo_root, capture_output=True, text=True,
    )
    if check and cp.returncode != 0:
        raise RuntimeError(
            "git " + " ".join(args) + " failed: " + (cp.stderr or cp.stdout).strip()
        )
    return cp


def _phase_diff_artifacts():
    """Build the mandatory complete baseline->candidate patch for this review unit."""
    root_cp = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=HERE, capture_output=True, text=True,
    )
    if root_cp.returncode != 0:
        # A reviewer may rerun run_tests.sh from an extracted packet where no
        # .git directory exists. Reuse only the packet's already-shipped exact
        # phase diff; never invent or silently omit provenance.
        provenance = HERE.parent / "provenance"
        patch_path = provenance / "diff_to_phase_13_75_candidate.patch"
        files_path = provenance / "phase_candidate_files.txt"
        baseline_path = provenance / "phase_baseline.txt"
        if not (patch_path.is_file() and files_path.is_file() and baseline_path.is_file()):
            raise RuntimeError("cannot locate Git repository or shipped phase-diff provenance")
        patch = patch_path.read_text(encoding="utf-8")
        candidate_text = files_path.read_text(encoding="utf-8")
        baseline = baseline_path.read_text(encoding="utf-8").strip()
        if baseline != PHASE_BASELINE:
            raise RuntimeError(
                f"shipped phase baseline {baseline!r} != expected {PHASE_BASELINE!r}"
            )
        missing = [rel for rel in PHASE_CANDIDATE_FILES if rel not in patch]
        if missing:
            raise RuntimeError(
                "shipped phase diff is incomplete: " + ", ".join(missing)
            )
        return patch, candidate_text, "(packet extraction; using shipped complete phase diff)\n"
    repo_root = Path(root_cp.stdout.strip()).resolve()
    diag_rel = HERE.resolve().relative_to(repo_root)
    relpaths = [(diag_rel / rel).as_posix() for rel in PHASE_CANDIDATE_FILES]

    _git(repo_root, "cat-file", "-e", PHASE_BASELINE + "^{commit}")
    for rel in relpaths:
        if not (repo_root / rel).is_file():
            raise RuntimeError(f"candidate file missing while building phase diff: {rel}")

    tracked = []
    new = []
    for rel in relpaths:
        exists = _git(
            repo_root, "cat-file", "-e", f"{PHASE_BASELINE}:{rel}", check=False
        ).returncode == 0
        (tracked if exists else new).append(rel)

    chunks = []
    if tracked:
        cp = _git(
            repo_root, "diff", "--binary", "--full-index", PHASE_BASELINE, "--", *tracked
        )
        if cp.stdout:
            chunks.append(cp.stdout)
    for rel in new:
        cp = _git(
            repo_root, "diff", "--no-index", "--binary", "--", "/dev/null", rel,
            check=False,
        )
        if cp.returncode not in (0, 1):
            raise RuntimeError(f"git diff --no-index failed for {rel}: {cp.stderr.strip()}")
        if cp.stdout:
            chunks.append(cp.stdout)

    patch = "".join(chunks)
    missing = [rel for rel in relpaths if rel not in patch]
    if missing:
        raise RuntimeError(
            "phase diff is incomplete; candidate path(s) absent: " + ", ".join(missing)
        )

    status = _git(repo_root, "status", "--short", "--", *relpaths).stdout
    candidate_text = (
        f"baseline={PHASE_BASELINE}\n"
        + "\n".join(relpaths)
        + "\n"
    )
    return patch, candidate_text, status


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--evidence", action="append", default=[],
                    help="directory to include under evidence/ (repeatable)")
    ap.add_argument("--skip-tests", action="store_true",
                    help="do not run the suites (used by run_tests.sh, which "
                         "just ran them; standalone official packets rerun)")
    ap.add_argument("--crr", default=None,
                    help="explicit CRR path (else auto-found in --docs-dir)")
    ap.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR,
                    help="GitLab docs tree holding governance documents "
                         "(spec, motivation, CRR, ledger)")
    ap.add_argument("--logs", default=None,
                    help="log directory to package (default: <here>/test_logs)")
    a = ap.parse_args(argv)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(a.out or f"diagnostics_reviewer_{ts}.zip")

    logdir = Path(a.logs) if a.logs else HERE / "test_logs"
    if not a.skip_tests:
        print("[bundle] running suites (this IS the point - packets carry fresh logs)")
        rc = subprocess.run(["bash", str(HERE / "run_tests.sh"), str(logdir)]).returncode
        if rc != 0:
            print("[bundle] SUITES FAILING - packet will say so; fix before distributing")

    manifest, entries = [], []

    def add_bytes(data, arc):
        if isinstance(data, str):
            data = data.encode("utf-8")
        entries.append((arc, data))
        manifest.append(f"{hashlib.md5(data).hexdigest()}  {arc}")

    def add(path, arc):
        # Payloads ship unmodified so packet fingerprints equal repository bytes.
        add_bytes(Path(path).read_bytes(), arc)

    # Shared umbrella dfextensions/__init__.py lives one level ABOVE the
    # subproject; ship it in package layout so a reviewer can fingerprint it
    # here instead of hunting sources_adf.zip [closes round-1 P0-1 in-packet].
    _umbrella = HERE.parent / "__init__.py"
    if _umbrella.is_file():
        add(_umbrella, "dfextensions/__init__.py")
    else:
        manifest.append("MISSING                           dfextensions/__init__.py")

    for f in CODE:
        p = HERE / f
        if p.is_file():
            add(p, f"diagnostics/{f}")   # REAL layout: recipe commands work as-is
        else:
            manifest.append(f"MISSING                           diagnostics/{f}")
    for f in sorted((HERE / "tests").glob("test_*")):
        add(f, f"diagnostics/tests/{f.name}")
    for f in DOCS:
        p = HERE / f
        if p.is_file():
            # P0-2 fix: CAPABILITY_MATRIX.md is consumed by capabilities.py and
            # test_capabilities.py, which resolve docs/ as a SIBLING of the code
            # (diagnostics/docs/). Ship it there so a fresh packet extraction
            # reproduces T-P9. Human-facing docs (README) stay at packet-top docs/.
            if Path(f).name == "CAPABILITY_MATRIX.md":
                add(p, f"diagnostics/docs/{Path(f).name}")
            else:
                add(p, f"docs/{Path(f).name}")
    _doc_search = [a.docs_dir, HERE, HERE.parent]
    _crr_name = None
    for _prefix, _role in GOVERNANCE_DOCS:
        # explicit --crr wins for the CRR slot
        if "CRR" in _prefix and a.crr and Path(a.crr).is_file():
            _doc = Path(a.crr)
        else:
            _doc = _newest_match(_doc_search, _prefix)
        if _doc and _doc.is_file():
            add(_doc, f"governance/{_doc.name}")
            if "CRR" in _prefix:
                _crr_name = _doc.name
        else:
            print(f"[bundle] NOTE: governance doc not found ({_role}): {_prefix}*.md")
    # LEDGER-PKT: the completion ledger, from the docs tree or subproject root
    _ledger_dirs = [a.docs_dir, HERE, HERE.parent]
    if a.crr:
        _ledger_dirs.insert(0, str(Path(a.crr).parent))
    _shipped = False
    for _name in LEDGER_NAMES:
        for _d in _ledger_dirs:
            _lp = Path(_d) / _name
            if _lp.is_file():
                add(_lp, f"governance/{_name}")
                _shipped = True
                break
        if _shipped:
            break
    if not _shipped:
        print("[bundle] NOTE: completion ledger not found in " + str(_ledger_dirs))
    # P1-LogSelect (round-3): one newest log of EACH kind by explicit name
    # pattern - mtime alone only coincidentally selected a bash+pytest pair
    # Package the newest log of every canonical suite kind.  Keep the
    # historical single pytest_<timestamp>.log, and also preserve split
    # pytest_fast / pytest_e2e logs when a diagnostic run uses them.
    for pat in (
        "bash_suite_*.log",
        "pytest_[0-9]*.log",
        "pytest_fast_*.log",
        "pytest_e2e_*.log",
        "run_status_*.txt",
    ):
        cand = sorted(logdir.glob(pat), key=lambda p: p.stat().st_mtime)
        if cand:
            add(cand[-1], f"logs/{cand[-1].name}")
    # C0 / own-team governance: the packet MUST carry the complete exact
    # baseline->candidate patch for the explicit review unit, including new files.
    try:
        phase_patch, candidate_text, candidate_status = _phase_diff_artifacts()
    except Exception as exc:
        print(f"[bundle] FATAL: cannot build complete phase diff: {exc}")
        return 2
    add_bytes(phase_patch, "provenance/diff_to_phase_13_75_candidate.patch")
    add_bytes(candidate_text, "provenance/phase_candidate_files.txt")
    add_bytes(candidate_status or "(clean candidate paths)\n",
              "provenance/phase_candidate_status.txt")
    add_bytes(PHASE_BASELINE + "\n", "provenance/phase_baseline.txt")
    print("[bundle] phase diff: COMPLETE baseline->candidate patch included")

    try:
        gl = subprocess.run(["git", "log", "--oneline", "--stat", "PHASE_13_74_ADF_BEGIN..HEAD"],
                            capture_output=True, text=True, cwd=HERE).stdout
    except Exception:
        gl = "(git unavailable)\n"
    for ev in a.evidence:
        for f in sorted(Path(ev).rglob("*")):
            if f.is_file():
                add(f, f"evidence/{Path(ev).name}/{f.relative_to(ev)}")

    gl = gl or "(no commits in range)\n"
    start_here = (
        "Reviewer packet - dfextensions/diagnostics (PHASE_13_75_DIAG successor work)\n"
        "This packet is SELF-CONTAINED: everything needed to review is inside.\n\n"
        "READING ORDER:\n"
        "  0. governance/PHASE_13_74_ADF_MOTIVATION*.md   - inherited WHY / scientific goal\n"
        "  1. governance/PHASE_13_75_DIAG_Increment*.md   - current Code Review Request\n"
        "  2. governance/PHASE_13_75_DIAG_Proposal*.md    - current normative specification\n"
        "  3. governance/*Completion_Ledger*.md           - remaining-work status matrix\n\n"
        "VERIFY & RUN:\n"
        "  4. read provenance/diff_to_phase_13_75_candidate.patch - COMPLETE reviewed diff\n"
        "  5. verify provenance/MANIFEST.md5 against diagnostics/ (incl. dfextensions/__init__.py)\n"
        "  6. run: bash diagnostics/run_tests.sh  (layout matches a checkout)\n"
        "  7. logs/ holds the suite runs made when this zip was built\n"
        "  8. evidence/ holds a REAL bundle + rendered report.html + validation/\n"
        "  9. provenance/git_log.txt anchors the review target commit\n")
    manifest_text = "\n".join(manifest) + "\n"
    # [Decision 3, 2026-07-23] the build-time identity gate is removed as a
    # normative requirement.  It previously failed the packet whenever a real
    # username or home path survived; information now propagates in full, so
    # there is nothing to fail on.  _scrub()/_leak_scan() remain defined above
    # but are not invoked.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for arc, data in entries:
            z.writestr(arc, data)
        z.writestr("provenance/git_log.txt", gl)
        z.writestr("provenance/MANIFEST.md5", manifest_text)
        z.writestr("START_HERE.txt", start_here)
    print(f"[bundle] wrote {out}  files={len(entries)}")
    print(f"[bundle] zip md5: {md5(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
