#!/usr/bin/env python3
"""
test_run_metrics.py - PHASE_13_74_ADF D2 tests (T-M1..T-M9).
Sandbox tier runs anywhere with /proc; the ADF test (T-M9) is alma2-gated.
Run: pytest -q diagnostics/tests/test_run_metrics.py
"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from run_metrics import RunMetrics, load_record  # noqa: E402

HAVE_PROC = Path("/proc/self/status").exists()
proc_only = pytest.mark.skipif(not HAVE_PROC, reason="needs Linux /proc")


def run_ok(tmp_path, **kw):
    with RunMetrics("t1", out=tmp_path, sample_interval_s=kw.pop("iv", 0.0), **kw) as rm:
        rm.set_label("case", "unit")
    return load_record(rm.record_path)


def test_m1_success_record_valid_json(tmp_path):
    rec = run_ok(tmp_path)
    assert rec["outcome"] == "success" and rec["exception"] is None
    assert rec["schema_version"] == 1 and rec["label"] == "t1"
    assert rec["labels"] == {"case": "unit"}

def test_m2_exception_written_and_reraised(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        with RunMetrics("t2", out=tmp_path) as rm:
            raise RuntimeError("boom")
    rec = load_record(rm.record_path)
    assert rec["outcome"] == "exception"
    assert rec["exception"] == {"type": "RuntimeError", "message": "boom"}

def test_m3_field_types_and_ranges(tmp_path):
    with RunMetrics("t3", out=tmp_path) as rm:
        time.sleep(0.05)
    rec = load_record(rm.record_path)
    assert rec["time"]["wall_s"] >= 0.05
    assert rec["time"]["user_s"] >= 0 and rec["time"]["system_s"] >= 0
    assert isinstance(rec["faults"]["minor"], int)
    assert rec["rss_kb"]["ru_maxrss_lifetime"] > 0          # R-6: lifetime, own name

def test_m4_record_event_aggregation_and_validation(tmp_path):
    with RunMetrics("t4", out=tmp_path) as rm:
        rm.record_event("load", {"branches": 3})
        time.sleep(0.01)
        rm.record_event("draw", {"figure": "f1"})
        with pytest.raises(TypeError):
            rm.record_event("bad", "not-a-dict")
        with pytest.raises(TypeError):
            rm.record_event("", {})
        with pytest.raises(TypeError):
            rm.record_event("unserializable", {"x": object()})
    rec = load_record(rm.record_path)
    ev = rec["events"]
    assert [e["kind"] for e in ev] == ["load", "draw"]
    assert ev[0]["payload"] == {"branches": 3}
    assert ev[1]["t_rel_s"] >= ev[0]["t_rel_s"]              # monotonic

def test_m5_provenance_fields(tmp_path):
    rec = run_ok(tmp_path)
    for k in ("tool", "tool_version", "utc", "pid", "python", "schema_version"):
        assert k in rec
    assert rec["tool"] == "run_metrics"

def test_m6_wrapper_invariance(tmp_path):
    """Run twice, with and without wrapper: result and exception behavior equal."""
    def work(x):
        if x < 0:
            raise ValueError("neg")
        return sum(i * i for i in range(x))
    bare = work(500)
    with RunMetrics("t6", out=tmp_path):
        wrapped = work(500)
    assert wrapped == bare
    with pytest.raises(ValueError):
        work(-1)
    with pytest.raises(ValueError):
        with RunMetrics("t6b", out=tmp_path):
            work(-1)

@proc_only
def test_m7_synthetic_allocation_peak_rss(tmp_path):
    with RunMetrics("t7", out=tmp_path, sample_interval_s=0.02) as rm:
        blob = bytearray(60 * 1024 * 1024)      # 60 MiB
        blob[::4096] = b"x" * len(blob[::4096]) # touch pages
        time.sleep(0.15)                        # let the sampler see it
        del blob
    rec = load_record(rm.record_path)
    r = rec["rss_kb"]
    assert r["start"] is not None and r["peak_context_sampled"] is not None
    assert r["peak_context_sampled"] - r["start"] > 40 * 1024   # > 40 MiB seen

@proc_only
def test_m8_synthetic_read_io_counters(tmp_path):
    payload = tmp_path / "blob.bin"
    payload.write_bytes(b"z" * (8 * 1024 * 1024))
    with RunMetrics("t8", out=tmp_path) as rm:
        data = payload.read_bytes()
    assert len(data) == 8 * 1024 * 1024
    rec = load_record(rm.record_path)
    assert rec["io"]["state"] == "available"
    assert rec["io"]["rchar"] >= 8 * 1024 * 1024              # logical reads

def test_load_record_rejects_foreign_json(tmp_path):
    p = tmp_path / "x.json"; p.write_text(json.dumps({"tool": "other"}))
    with pytest.raises(ValueError):
        load_record(p)

# --------------------------- ADF tier (alma2 gate) --------------------------
_ADF_DIR = Path(__file__).resolve().parent.parent.parent / "AliasDataFrame"
if _ADF_DIR.is_dir():
    sys.path.insert(0, str(_ADF_DIR))
try:
    from AliasDataFrame import AliasDataFrame
    HAVE_ADF = True
except Exception as _e:
    HAVE_ADF = False
    _REASON = f"ADF tier requires the locked stack (alma2 gate): {_e}"

@pytest.mark.skipif(not HAVE_ADF, reason="" if HAVE_ADF else _REASON)
def test_m9_adf_describe_before_after_no_materialization(tmp_path):
    import pandas as pd
    adf = AliasDataFrame(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
    adf.add_alias("y", "x*2")
    with RunMetrics("t9", adf=adf, out=tmp_path) as rm:
        _ = adf.df["x"].sum()          # workload touches no alias
    rec = load_record(rm.record_path)
    assert rec["adf_before"]["ok"] and rec["adf_after"]["ok"]
    # diagnostic-only claim: describe_lazy must not have materialized 'y'
    assert "y" not in adf.df.columns
