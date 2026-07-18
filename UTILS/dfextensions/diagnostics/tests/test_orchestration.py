#!/usr/bin/env python3
"""
test_orchestration.py - PHASE_13_74_ADF v8 section 3.3 failure-contract tests.
Hermetic: fixture /proc via env; short windows; every test asserts the ONE
non-negotiable property first - the workload's exit status is propagated
UNCHANGED. Covered: T-O1..T-O9. Deferred to the acceptance run (disclosed):
mid-workload operator interrupt; report-step failure isolation on real ADF.
Run: pytest -q diagnostics/tests/test_orchestration.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

DIAG = Path(__file__).resolve().parent.parent
WRAP = DIAG / "dfx_run_with_diagnostics.py"

pytestmark = pytest.mark.skipif(not (DIAG / "dfx_host_diagnostics.sh").exists(),
                                reason="collector script missing")


def fixture_env(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_collector import build_fixture
    proc = tmp_path / "proc"
    build_fixture(proc)                       # process entries + loadavg + boot_id
    # the bash collector's REQUIRED evidence (vmstat/thp/meminfo) + basics -
    # without these, preflight legitimately exits UNKNOWN and no sampling runs
    (proc / "vmstat").write_text("compact_stall 100\ncompact_fail 40\n"
                                 "thp_fault_alloc 120\nthp_fault_fallback 100\n"
                                 "thp_collapse_alloc 7\npgscan_direct 3\n"
                                 "allocstall_normal 20\npswpin 1\npswpout 2\n")
    (proc / "meminfo").write_text("MemTotal:  100000 kB\nMemAvailable:  50000 kB\n")
    (proc / "uptime").write_text("100.00 95.00\n")
    (proc / "stat").write_text("cpu  1000 0 500 8000 100 0 0 0 0 0\nctxt 123456\n"
                               "procs_running 2\n")
    sysd = tmp_path / "sys" / "kernel" / "mm" / "transparent_hugepage"
    sysd.mkdir(parents=True, exist_ok=True)
    (sysd / "enabled").write_text("always madvise [never]\n")
    (sysd / "defrag").write_text("always defer [madvise] never\n")
    env = dict(os.environ, PROC_ROOT=str(proc), SYS_ROOT=str(tmp_path / "sys"),
               CGROUP_ROOT=str(tmp_path / "nocg"), CLK_TCK_OVERRIDE="100",
               DFX_PROCESS_SAMPLER="off", MPLBACKEND="Agg")
    return env


def run_wrap(tmp_path, workload, extra=(), env=None):
    out = tmp_path / "data"
    r = subprocess.run([sys.executable, str(WRAP), "--out", str(out),
                        "--interval", "1", "--max-samples", "30",
                        "--pre", "0.3", "--post", "0.3", *extra, "--", *workload],
                       capture_output=True, text=True,
                       env=env or fixture_env(tmp_path), timeout=60)
    oj = out / "orchestration.json"
    orch = json.loads(oj.read_text()) if oj.is_file() else None
    return r, out, orch


def steps(orch, name):
    return [s for s in orch["steps"] if s["step"] == name]


def test_o1_exit_zero_passthrough(tmp_path):
    r, out, orch = run_wrap(tmp_path, ["true"])
    assert r.returncode == 0 and orch["workload_rc"] == 0

def test_o2_nonzero_rc_unchanged(tmp_path):
    r, _, orch = run_wrap(tmp_path, ["bash", "-c", "exit 7"])
    assert r.returncode == 7 and orch["workload_rc"] == 7

def test_o3_missing_workload_rc127_collector_stopped(tmp_path):
    r, _, orch = run_wrap(tmp_path, ["/nonexistent/definitely_missing"])
    assert r.returncode == 127
    assert steps(orch, "collector_stop")[0]["status"] in ("ok", "ERROR")

def test_o4_pidfile_registered_and_cleaned(tmp_path):
    r, out, orch = run_wrap(tmp_path,
                            ["bash", "-c", 'sleep 0.5; ls "$(dirname "$0")" >/dev/null'])
    assert r.returncode == 0
    ws = steps(orch, "workload_start")[0]
    assert ws["pid"] > 0
    assert not list(out.glob(".target_*.pid"))          # cleaned afterwards

def test_o5_env_handoff_to_workload(tmp_path):
    r, out, orch = run_wrap(tmp_path,
                            ["bash", "-c", 'echo "RID=$DFX_RUN_ID BD=$DFX_BUNDLE_DIR"'])
    assert r.returncode == 0
    assert f"RID={orch['run_id']}" in r.stdout
    assert "BD=" in r.stdout

def test_o6_bundle_complete_with_clean_stop(tmp_path):
    r, out, orch = run_wrap(tmp_path, ["bash", "-c", "sleep 2"])
    assert r.returncode == 0
    b = Path(steps(orch, "bundle_discovered")[0]["bundle"])
    man = (b / "manifest.kv").read_text()
    assert "verdict=" in man                            # clean stop wrote the verdict
    assert "stop_reason=signal_clean_stop" in man
    assert "samples_taken=" in man

def test_o7_failed_workload_rc_never_masked(tmp_path, monkeypatch):
    """FAILED workload always propagates its own rc - even with broken
    diagnostics (v8 3.3; exit-70 applies only to rc==0, see T-M21)."""
    env = fixture_env(tmp_path)
    env["PROC_ROOT"] = "/nonexistent_proc_root"
    r, _, orch = run_wrap(tmp_path, ["bash", "-c", "exit 5"], env=env)
    assert r.returncode == 5 and orch["workload_rc"] == 5

def test_m21_success_plus_diag_finalization_failure_exits_70(tmp_path):
    """v8:1054 / T-M21 (CRR-5): workload rc==0 but diagnostics finalization
    failed -> wrapper exits 70 and records the failed stage."""
    import shutil
    iso = tmp_path / "iso"; iso.mkdir()
    shutil.copy(WRAP, iso / WRAP.name)          # collector script ABSENT here
    r = subprocess.run([sys.executable, str(iso / WRAP.name), "--out",
                        str(tmp_path / "d"), "--pre", "0", "--post", "0",
                        "--", "true"],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 70
    orch = json.loads((tmp_path / "d" / "orchestration.json").read_text())
    fin = [s for s in orch["steps"] if s["step"] == "diagnostics_finalization"]
    assert fin and fin[0]["status"] == "FAILED" and fin[0]["exit_code"] == 70
    assert orch["workload_rc"] == 0

def test_o8_workload_stdout_stderr_passthrough(tmp_path):
    r, _, _ = run_wrap(tmp_path, ["bash", "-c", "echo OUT_MARKER; echo ERR_MARKER >&2"])
    assert "OUT_MARKER" in r.stdout and "ERR_MARKER" in r.stderr

def test_o9_orchestration_record_complete(tmp_path):
    r, _, orch = run_wrap(tmp_path, ["true"])
    names = [s["step"] for s in orch["steps"]]
    for required in ("run_id", "collector_start", "bundle_discovered",
                     "workload_start", "workload_end", "collector_stop"):
        assert required in names
    assert all("t" in s and "status" in s for s in orch["steps"])
