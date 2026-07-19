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
        "dfx_run_with_diagnostics.py", "run_tests.sh",
        "job_host_analysis.py", "conclusion_model.py",
        "reviewer_bundle.py"]
DOCS = ["README.md"]


import getpass
import os
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
    """After scrubbing: zero tolerance. A hit fails the packet build."""
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
        # payload is scrubbed BEFORE fingerprinting: the manifest describes
        # the bytes reviewers actually receive [panel P0-5]
        data = _scrub(Path(path).read_bytes(), arc)
        entries.append((arc, data))
        import hashlib as _h
        manifest.append(f"{_h.md5(data).hexdigest()}  {arc}")

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
    # P1-LogSelect (round-3): one newest log of EACH kind by explicit name
    # pattern - mtime alone only coincidentally selected a bash+pytest pair
    for pat in ("bash_suite_*.log", "pytest_*.log"):
        cand = sorted(logdir.glob(pat), key=lambda p: p.stat().st_mtime)
        if cand:
            add(cand[-1], f"logs/{cand[-1].name}")
    try:
        gl = subprocess.run(["git", "log", "--oneline", "--stat", "PHASE_13_74_ADF_BEGIN..HEAD"],
                            capture_output=True, text=True, cwd=HERE).stdout
    except Exception:
        gl = "(git unavailable)\n"
    for ev in a.evidence:
        for f in sorted(Path(ev).rglob("*")):
            if f.is_file():
                add(f, f"evidence/{Path(ev).name}/{f.relative_to(ev)}")

    gl = _scrub((gl or "(no commits in range)\n").encode(), "git_log.txt").decode()
    start_here = (
        "Reviewer packet - dfextensions/diagnostics (PHASE_13_74_ADF)\n"
        "1. verify provenance/MANIFEST.md5 against code/ and tests/\n"
        + (("2. read docs/" + Path(a.crr).name + " (the Code Review Request), then\n")
           if a.crr and Path(a.crr).is_file() else
           "2. the CRR document is distributed separately via GitLab (not in this packet)\n"
           "   then\n")
        + "   run: bash diagnostics/run_tests.sh  (layout matches a checkout)\n"
        "3. logs/ contains the suite runs made when this zip was built\n"
        "4. evidence/ holds a REAL bundle + rendered report.html + validation/"
        "   collected on the packet-builder host at build time\n")
    start_here = _scrub(start_here.encode(), "START_HERE.txt").decode()
    manifest_text = "\n".join(manifest) + "\n"
    # P1-LeakScope (round-3): EVERYTHING written to the zip is scanned -
    # including the three payloads previously assembled outside `entries`
    scan_set = entries + [("provenance/git_log.txt", gl.encode()),
                          ("provenance/MANIFEST.md5", manifest_text.encode()),
                          ("START_HERE.txt", start_here.encode())]
    leaks = _leak_scan(scan_set)
    if leaks:
        print("[bundle] PRIVACY LEAK after scrub - packet build FAILED:")
        for n in leaks[:10]:
            print(f"[bundle]   {n}")
        sys.exit(3)
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
