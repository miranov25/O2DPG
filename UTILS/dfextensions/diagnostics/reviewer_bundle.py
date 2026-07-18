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
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

CODE = ["dfx_host_diagnostics.sh", "collector.py", "schema.py", "audit.py",
        "report_diagnostics.py", "run_metrics.py", "explain_bundle.py",
        "dfx_run_with_diagnostics.py", "run_tests.sh"]
DOCS = ["README.md"]


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
                    help="CRR document to include under docs/ (official packets)")
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
    def add(path, arc):
        entries.append((Path(path), arc))
        manifest.append(f"{md5(path)}  {arc}")

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
            add(p, f"docs/{f}")
    if a.crr and Path(a.crr).is_file():
        add(Path(a.crr), f"docs/{Path(a.crr).name}")
    for lg in sorted(logdir.glob("*.log"))[-4:]:
        add(lg, f"logs/{lg.name}")
    try:
        gl = subprocess.run(["git", "log", "--oneline", "PHASE_13_74_ADF_BEGIN..HEAD"],
                            capture_output=True, text=True, cwd=HERE).stdout
    except Exception:
        gl = "(git unavailable)\n"
    for ev in a.evidence:
        for f in sorted(Path(ev).rglob("*")):
            if f.is_file():
                add(f, f"evidence/{Path(ev).name}/{f.relative_to(ev)}")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in entries:
            z.write(p, arc)
        z.writestr("provenance/git_log.txt", gl or "(no commits in range)\n")
        z.writestr("provenance/MANIFEST.md5", "\n".join(manifest) + "\n")
        z.writestr("START_HERE.txt",
                   "Reviewer packet - dfextensions/diagnostics (PHASE_13_74_ADF)\n"
                   "1. verify provenance/MANIFEST.md5 against code/ and tests/\n"
                   "2. read docs/PHASE_13_74_ADF_CRR*.md (the Code Review Request), then\n"
                   "   run: bash diagnostics/run_tests.sh  (layout matches a checkout)\n"
                   "3. logs/ contains the suite runs made when this zip was built\n"
                   "4. evidence/ holds a REAL bundle + rendered report.html + validation/\n""   collected on the packet-builder host at build time\n")
    print(f"[bundle] wrote {out}  files={len(entries)}")
    print(f"[bundle] zip md5: {md5(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
