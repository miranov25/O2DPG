#!/usr/bin/env python3
"""
test_audit.py - PHASE_13_74_ADF v8 validation/ audit engine tests (T-V1..T-V6).
The decisive property: a seeded inconsistency must be DETECTED and LOCALIZED
to its stage - that is the architect's statistical debuggability, tested.
Run: pytest -q diagnostics/tests/test_audit.py
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pd = pytest.importorskip("pandas")
import audit  # noqa: E402
import schema  # noqa: E402
from test_diagnostics_py import make_bundle  # reuse the bundle fixture  # noqa: E402


def host_frame(tmp_path, **kw):
    return schema.samples_frame(schema.load_bundle(make_bundle(tmp_path, **kw)))


def test_v1_conservation_pass_on_consistent_fixture(tmp_path):
    a = audit.Audit()
    df = host_frame(tmp_path)
    a.check_conservation(df)
    cons = [c for c in a.checks if c["check_id"].startswith("I-CONS-") and c["status"] != "SKIP"]
    assert cons and all(c["status"] == "PASS" for c in cons)
    assert a.first_inconsistency() is None

def test_v2_seeded_corruption_detected_and_localized(tmp_path):
    """Corrupt ONE rate value post-collection -> conservation FAILS and the
    first-inconsistency pointer names the stage and the channel."""
    b = make_bundle(tmp_path)
    csv = (b / "samples.csv")
    csv.write_text(csv.read_text().replace("100.000", "150.000", 1))
    a = audit.Audit()
    a.check_conservation(schema.samples_frame(schema.load_bundle(b)))
    fi = a.first_inconsistency()
    assert fi is not None
    assert fi["check_id"] == "I-CONS-compact_stall_total"
    assert fi["stage"] == "S2_parse_derive"
    assert float(fi["discrepancy"]) > 100      # 50/s over a 10s interval

def test_v3_hierarchy_pass_and_violation(tmp_path):
    ts = [1010, 1020]
    proc = pd.DataFrame({"ts": ts, "is_target_job": [1, 1],
                         "cpu_pct": [50.0, 60.0], "pid": [50, 50]})
    user = pd.DataFrame({"ts": ts, "is_current_user": [1, 1],
                         "cpu_cores": [0.9, 0.9]})
    a = audit.Audit()
    a.check_hierarchy(None, proc, user)
    assert [c for c in a.checks if c["check_id"] == "I-HIER-cpu"][0]["status"] == "PASS"
    a2 = audit.Audit()
    user_bad = pd.DataFrame({"ts": ts, "is_current_user": [1, 1],
                             "cpu_cores": [0.1, 0.1]})   # job 0.5-0.6 cores > user 0.1
    a2.check_hierarchy(None, proc, user_bad)
    assert [c for c in a2.checks if c["check_id"] == "I-HIER-cpu"][0]["status"] == "FAIL"

def test_v4_rss_hierarchy_is_prohibited_not_checked():
    a = audit.Audit()
    a.check_hierarchy(None, pd.DataFrame({"ts": [], "is_target_job": [],
                                          "cpu_pct": [], "pid": []}),
                      pd.DataFrame({"ts": [], "is_current_user": [], "cpu_cores": []}))
    note = [c for c in a.checks if c["check_id"] == "I-HIER-note-rss"]
    assert note and note[0]["status"] == "SKIP" and "double-counts" in note[0]["detail"]

def test_v5_writers_produce_four_artifacts_and_pointer(tmp_path):
    a = audit.Audit()
    df = host_frame(tmp_path)
    a.add_stage("S2_parse_derive", "hostA:samples", df)
    a.check_conservation(df)
    v = a.write(tmp_path / "rep")
    assert (v / "stage_stats.jsonl").is_file()
    assert (v / "transition_checks.csv").is_file()
    assert (v / "conclusion_trace.json").is_file()
    text = (v / "summary.md").read_text()
    assert "no inconsistencies" in text
    first = json.loads((v / "stage_stats.jsonl").read_text().splitlines()[0])
    assert {"stage", "table", "column", "rows"} <= set(first)

def test_v6_generate_default_on_and_opt_out(tmp_path):
    """Audit ships with EVERY report by default (A-1); --no-audit opts out."""
    import report_diagnostics as rd
    b = make_bundle(tmp_path)
    out1 = tmp_path / "r1"; out2 = tmp_path / "r2"
    try:
        rd.generate([b], out_dir=out1)                      # ADF present: full path
    except ImportError:
        pytest.skip("ADF stack absent: generate() unreachable in sandbox tier")
    assert (out1 / "validation" / "transition_checks.csv").is_file()
    rd.generate([b], out_dir=out2, audit=False)
    assert not (out2 / "validation").exists()
