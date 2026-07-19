#!/usr/bin/env python3
"""
test_wrapper_vertical.py - PHASE_13_74_ADF correction pass round 2
(GPT22/GPT24): the EXECUTED wrapper sequence, not the helper in isolation.

The prior IC-4 pre-created orchestration.json and tested only the helper -
so it could not see that the wrapper asked for a record that did not exist
yet. These tests run the wrapper's real main() with a stub report module
injected into sys.modules that captures the actual run_records argument.

VT-1 the record EXISTS and is PASSED at report time, with outcome=success
VT-2 failed workload -> outcome=failed carried into the record, never CM-1
VT-3 stale records from other run_ids are excluded
VT-4 successful job + failing report -> exit 70 (finalization contract)
Run: pytest -q diagnostics/tests/test_wrapper_vertical.py
"""
import json
import sys
import types
from pathlib import Path

import pytest

DIAG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DIAG))
pd = pytest.importorskip("pandas")
import dfx_run_with_diagnostics as wrap  # noqa: E402
import conclusion_model as cm  # noqa: E402
import job_host_analysis as jha  # noqa: E402


class _Capture:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def generate(self, bundles, run_records=(), **kw):
        self.calls.append(list(run_records))
        if self.fail:
            raise RuntimeError("stub report failure")
        out = Path(kw.get("out_dir", "/tmp/x")); out.mkdir(parents=True, exist_ok=True)
        p = out / "report.html"; p.write_text("<html>stub</html>")
        return p


def run_main(tmp_path, workload, capture, monkeypatch):
    stub = types.ModuleType("report_diagnostics")
    stub.generate = capture.generate
    monkeypatch.setitem(sys.modules, "report_diagnostics", stub)
    monkeypatch.setenv("DFX_COLLECTOR_DISABLE", "1")   # no real collector needed
    fb = tmp_path / "host_diag_test_19700101T000000Z_00000000"
    fb.mkdir(parents=True, exist_ok=True)
    (fb / "manifest.kv").write_text("verdict=OK\n")     # minimal discovered bundle
    monkeypatch.setattr(wrap.glob, "glob", lambda pat: [str(fb)])
    argv = ["--out", str(tmp_path), "--label", "vt", "--report", "--"] + workload
    try:
        rc = wrap.main(argv)
    except SystemExit as e:
        rc = int(e.code or 0)
    return rc


def test_vt1_record_exists_and_passed_success(tmp_path, monkeypatch):
    cap = _Capture()
    rc = run_main(tmp_path, ["true"], cap, monkeypatch)
    assert rc == 0
    assert cap.calls, "generate never called"
    recs = cap.calls[-1]
    assert recs, "run_records EMPTY at report time - the panel's exact defect"
    orec = json.loads(Path([r for r in recs if r.endswith("orchestration.json")][0]).read_text())
    assert orec["outcome"] == "success" and orec["workload_rc"] == 0
    on_disk = json.loads((tmp_path / "orchestration.json").read_text())
    assert on_disk["run_id"] == orec["run_id"]

def test_vt2_failed_workload_outcome_failed_never_cm1(tmp_path, monkeypatch):
    cap = _Capture()
    rc = run_main(tmp_path, ["bash", "-c", "exit 7"], cap, monkeypatch)
    assert rc == 7
    orec = json.loads(Path([r for r in cap.calls[-1]
                            if r.endswith("orchestration.json")][0]).read_text())
    assert orec["outcome"] == "failed"
    # through the real analysis + conclusion chain: never reference-grade
    hf = pd.DataFrame({"ts": [0.0], "cpu_busy_pct": [1.0]})
    concl = cm.evaluate("OK", [], jha.analyze([orec], hf))
    assert concl["records"][0]["job_state"] == "failed"
    assert concl["records"][0]["code"] != "CM-1"

def test_vt3_foreign_run_records_excluded(tmp_path, monkeypatch):
    (tmp_path / "run_old_19700101T000000Z_1.json").write_text(
        json.dumps({"run_id": "FOREIGN", "label": "old"}))
    cap = _Capture()
    run_main(tmp_path, ["true"], cap, monkeypatch)
    recs = cap.calls[-1]
    for r in recs:
        assert json.loads(Path(r).read_text()).get("run_id") != "FOREIGN"

def test_vt4_report_failure_after_success_exits_70(tmp_path, monkeypatch):
    cap = _Capture(fail=True)
    rc = run_main(tmp_path, ["true"], cap, monkeypatch)
    assert rc == 70, f"successful job with failed report returned {rc}, not 70"


def test_vt5_runmetrics_env_handoff_discoverable(tmp_path, monkeypatch):
    """P1-RunMetrics: ordinary RunMetrics("job") with NO out= must land where
    the wrapper's env handoff points, and be found by build_report_records."""
    import run_metrics as rmmod
    monkeypatch.setenv("DFX_RUN_METRICS_OUT", str(tmp_path))
    monkeypatch.setenv("DFX_RUN_ID", "envrun01")
    with rmmod.RunMetrics("plainjob") as rm:
        pass
    assert Path(rm.record_path).parent == tmp_path
    rec = json.loads(Path(rm.record_path).read_text())
    recs = wrap.build_report_records(tmp_path, rec["run_id"])
    assert any(Path(p).name == Path(rm.record_path).name for p in recs)

def test_vt6_legacy_run_metrics_subdir_scanned(tmp_path):
    sub = tmp_path / "run_metrics"; sub.mkdir()
    (sub / "run_x_19700101T000000Z_1.json").write_text(
        json.dumps({"run_id": "R9", "label": "legacy"}))
    recs = wrap.build_report_records(tmp_path, "R9")
    assert any("run_metrics" in p for p in recs)
