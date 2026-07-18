#!/usr/bin/env python3
"""
test_job_host_analysis.py - PHASE_13_74_ADF Layer-C-foundation tests
(review-mandated set, GPT18 2026-07-18). The decisive additions over the
first draft: JH-1 proves the END-stamp time convention with known
timestamps against run_metrics BY EXECUTION - the first draft's fixtures
encoded the coder's wrong start-stamp assumption and passed anyway.
Run: pytest -q diagnostics/tests/test_job_host_analysis.py
"""
import sys
import time
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pd = pytest.importorskip("pandas")
import numpy as np  # noqa: E402
import job_host_analysis as jha  # noqa: E402

T0 = 1_000_000          # job START epoch used throughout
WALL = 60.0


def stamp(epoch):
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(epoch))


def record(start=T0, wall=WALL, rss=None, npt=13, events=None, rtype="in_process"):
    """END-stamp convention: utc = start + wall (as run_metrics writes it)."""
    rss = rss if rss is not None else [(i * 5.0, 1000 + 100 * i) for i in range(npt)]
    r = {"label": "job", "run_id": "r1", "record_type": rtype,
         "utc": stamp(start + wall), "time": {"wall_s": wall},
         "rss_series": rss}
    if events is not None:
        r["events"] = events
    return r


def host_frame(n=20, start=T0 - 20, step=5, busy=None, const_ch=0.0):
    ts = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "ts": ts,
        "cpu_busy_pct": busy if busy is not None else [10 + i for i in range(n)],
        "loadavg1": [const_ch] * n,
    })


def rollup(n=20, start=T0 - 20, step=5, ramp=True):
    rows = []
    for i in range(n):
        ts = start + i * step
        rows.append((ts, "target_job", 1.0))
        rows.append((ts, "other_visible_workloads", (n - i) * 0.1 if ramp else 0.5))
        rows.append((ts, "current_user_non_job", 0.0))
    return pd.DataFrame(rows, columns=["ts", "scope", "cpu_cores"])


def corr(res, jm, hm):
    return next(c for c in res["correlations"]
                if c["job_metric"] == jm and c["host_metric"] == hm)


# ---------------------------------------------------------------------------
def test_jh1_end_stamp_convention_proven_by_execution(tmp_path):
    """utc is the END stamp: run run_metrics for real, then require the
    reconstructed window to bracket the true execution interval."""
    from run_metrics import RunMetrics, load_record
    before = time.time()
    with RunMetrics("probe", out=tmp_path) as rm:
        time.sleep(1.2)
    after = time.time()
    rec = load_record(rm.record_path)
    state, s, e = jha._window(rec)
    assert state == "ok"
    assert s == pytest.approx(before, abs=2.0)
    assert e == pytest.approx(after, abs=2.0)
    assert e - s == pytest.approx(rec["time"]["wall_s"], abs=0.01)

def test_jh2_window_reconstruction_exact_known_stamps():
    res = jha.analyze_record(record(), host_frame())
    assert res["window"]["state"] == "ok"
    assert res["window"]["t0"] == T0 and res["window"]["t1"] == T0 + WALL

def test_jh3_rss_points_absolute_alignment():
    """A wrong (start-stamp) convention would shift points by WALL and find
    zero in-tolerance pairs against this deliberately tight host window."""
    hf = host_frame(n=13, start=T0, step=5)           # exactly the job window
    res = jha.analyze_record(record(), hf)
    c = corr(res, "job_rss", "cpu_busy_pct")
    assert c["state"] == "ok" and c["coverage_fraction"] == 1.0

def test_jh4_cadence_mismatch_and_tolerance_no_extrapolation():
    hf = host_frame(n=5, start=T0, step=7)            # host 7s vs rss 5s cadence
    rss = [(i * 5.0, 1000 + i) for i in range(13)]    # last points beyond host end
    res = jha.analyze_record(record(rss=rss), hf)
    c = corr(res, "job_rss", "cpu_busy_pct")
    assert c["alignment_method"] == "nearest_within_tolerance"
    assert c["time_tolerance_s"] == jha.ALIGN_TOL_S
    assert c["coverage_fraction"] < 1.0               # far points EXCLUDED, not extrapolated
    assert c["n_pairs"] < 13

def test_jh5_anticorrelated_background_detected():
    res = jha.analyze_record(record(), host_frame(), rollup(ramp=True))
    c = corr(res, "job_rss", "background_cpu_cores")
    assert c["state"] == "ok" and c["r"] < -0.9

def test_jh6_constant_channels_classified_warning_free():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = jha.analyze_record(record(), host_frame())
    c = corr(res, "job_rss", "loadavg1")
    assert c["state"] == "unavailable_constant" and c["r"] is None
    st, r, n = jha.safe_corr([1.0] * 6, [1, 2, 3, 4, 5, 6])
    assert st == "unavailable_constant"

def test_jh7_baseline_states():
    r = record()
    assert jha.analyze_record(r, host_frame(n=30, start=T0 - 60))["baseline_state"] == "ok"
    assert jha.analyze_record(r, host_frame(n=17, start=T0 - 25))["baseline_state"] == "pre_only"
    assert jha.analyze_record(r, host_frame(n=17, start=T0 + 5))["baseline_state"] == "post_only"
    assert jha.analyze_record(r, host_frame(n=12, start=T0 + 1))["baseline_state"] == "no_outside_window"
    assert jha.analyze_record(r, host_frame(n=14, start=T0 - 5))["baseline_state"] == "insufficient_baseline"

def test_jh8_record_types():
    hf = host_frame()
    r_no_series = record(); r_no_series.pop("rss_series")
    assert jha.analyze_record(r_no_series, hf)["correlations"][0]["state"] == "no_series"
    ext = {"tool": "dfx_run_with_diagnostics", "run_id": "x",
           "steps": [{"step": "workload_start", "t": T0, "status": "ok"},
                     {"step": "workload_end", "t": T0 + WALL, "status": "ok"}]}
    re_ = jha.analyze_record(ext, hf)
    assert re_["window"]["state"] == "ok_external"
    assert re_["correlations"][0]["state"] == "no_series"
    assert len(re_["influence"]) > 0                  # influence still computed
    assert jha.analyze_record({"record_type": "martian"}, hf)["window"]["state"] \
        == "unsupported_record_type"

def test_jh9_multiple_records_same_runid_all_analyzed():
    out = jha.analyze([record(), record()], host_frame())
    assert len(out) == 2 and all(o["run_id"] == "r1" for o in out)

def test_jh10_influence_oracle_independent_recompute():
    busy = [5.0] * 4 + [50.0] * 13 + [5.0] * 3
    hf = host_frame(busy=busy)
    res = jha.analyze_record(record(), hf)
    e = next(x for x in res["influence"] if x["channel"] == "cpu_busy_pct")
    ts = hf["ts"]
    oracle_in = hf[(ts >= T0) & (ts <= T0 + WALL)]["cpu_busy_pct"]
    oracle_out = hf[(ts < T0) | (ts > T0 + WALL)]["cpu_busy_pct"]
    assert e["window_mean"] == pytest.approx(float(oracle_in.mean()))
    assert e["baseline_mean"] == pytest.approx(float(oracle_out.mean()))

def test_jh11_progress_throughput_vs_background():
    """Throughput SLOWS while background RISES -> negative correlation.
    This is the job-performance question, not the RSS proxy."""
    ev = [{"kind": "progress", "payload": {"value": v}, "t_rel_s": t}
          for t, v in [(0, 0), (10, 500), (20, 900), (30, 1150),
                       (40, 1300), (50, 1400), (60, 1450)]]
    roll = rollup(n=20, start=T0 - 20, ramp=False)
    roll.loc[roll["scope"] == "other_visible_workloads", "cpu_cores"] = \
        [i * 0.2 for i in range(20)]                  # background RAMPS UP
    res = jha.analyze_record(record(events=ev), host_frame(), roll)
    assert res["progress"]["state"] == "ok"
    pc = next(c for c in res["progress"]["correlations"]
              if c["host_metric"] == "background_cpu_cores")
    assert pc["state"] == "ok" and pc["r"] < -0.8
    res2 = jha.analyze_record(record(), host_frame())
    assert res2["progress"]["state"] == "no_progress_events"

def test_jh12_stage_durations_background_classes():
    ev = [{"kind": "stage_start", "payload": {"stage": "load"}, "t_rel_s": 0.0},
          {"kind": "stage_end", "payload": {"stage": "load"}, "t_rel_s": 20.0},
          {"kind": "stage_start", "payload": {"stage": "fit"}, "t_rel_s": 25.0},
          {"kind": "stage_end", "payload": {"stage": "fit"}, "t_rel_s": 55.0}]
    res = jha.analyze_record(record(events=ev), host_frame(), rollup(ramp=True))
    assert res["stages"]["state"] == "ok"
    rows = {r["stage"]: r for r in res["stages"]["rows"]}
    assert rows["load"]["duration_s"] == 20.0 and rows["fit"]["duration_s"] == 30.0
    assert rows["load"]["background_class"] in ("low", "normal", "high")
    res2 = jha.analyze_record(record(), host_frame())
    assert res2["stages"]["state"] == "no_stage_events"

def test_jh13_analyze_never_raises(tmp_path):
    bad = tmp_path / "broken.json"; bad.write_text("{not json")
    out = jha.analyze([bad], host_frame())
    assert out[0]["window"]["state"] == "error"
