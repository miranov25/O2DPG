#!/usr/bin/env python3
"""
test_e2e_presence.py - PHASE_13_74_ADF: the presence-path acceptance test
(GPT22's vertical instruction, full-depth variant). Runs the REAL wrapper
around a tiny real workload that emits an in-process run record with
progress events, lets the REAL collector take a short bounded run, renders
the REAL report, and asserts the presence path end to end:

  - the HTML does NOT say "no run_metrics records supplied"
  - the current run's records appear (external + in-process)
  - a conclusion code is rendered
  - the persisted audit contains S7/S8 digests (not SKIP)

ADF tier: requires the full render stack; skipped where AliasDataFrame is
unavailable (sandbox), executed on alma2 - this is the packet-evidence test
the CRR-v3 panel required before any clean checkpoint.
Run: pytest -q diagnostics/tests/test_e2e_presence.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

DIAG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DIAG))
pytest.importorskip("pandas")
try:  # ADF tier gate
    sys.path.insert(0, str(DIAG.parent / "AliasDataFrame"))
    import AliasDataFrame  # noqa: F401
    HAVE_ADF = True
except Exception:
    HAVE_ADF = False

pytestmark = pytest.mark.skipif(not HAVE_ADF, reason="ADF tier (alma2)")

WORKLOAD = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
from run_metrics import RunMetrics
with RunMetrics("e2e_job", out=sys.argv[2]) as rm:
    v = 0
    for i in range(6):
        time.sleep(0.6)
        v += 100
        rm.record_event("progress", {"value": v})
"""


def test_e2e_wrapper_presence_path(tmp_path):
    out = tmp_path / "data"
    wl = tmp_path / "workload.py"
    wl.write_text(WORKLOAD)
    r = subprocess.run(
        [sys.executable, str(DIAG / "dfx_run_with_diagnostics.py"),
         "--out", str(out), "--label", "e2e", "--interval", "1",
         "--max-samples", "8", "--pre", "0", "--post", "0", "--report", "--",
         sys.executable, str(wl), str(DIAG), str(out)],
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"wrapper rc={r.returncode}\n{r.stdout}\n{r.stderr}"

    orch = json.loads((out / "orchestration.json").read_text())
    assert orch["outcome"] == "success"
    run_id = orch["run_id"]

    reports = list(out.glob("report_*/report.html"))
    assert reports, "no report rendered"
    html = reports[0].read_text()
    assert "no run_metrics records supplied" not in html, \
        "presence path still renders the absence message"
    assert run_id in html, "current run_id not in the rendered report"
    assert "Bundle conclusion [CM-" in html, "no conclusion code rendered"
    assert "e2e_job" in html, "in-process record not rendered"

    checks = list(out.glob("report_*/validation/transition_checks.csv"))
    assert checks, "audit not persisted"
    rows = checks[0].read_text()
    assert '"I-COV-S7_job_host_analysis"' in rows and '"PASS"' in \
        [l.split(",")[2].strip() for l in rows.splitlines()
         if "I-COV-S7" in l][0], "S7 not covered in persisted audit"
    assert any("I-COV-S8" in l and "PASS" in l for l in rows.splitlines()), \
        "S8 not covered in persisted audit"
