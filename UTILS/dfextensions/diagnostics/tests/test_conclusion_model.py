#!/usr/bin/env python3
"""
test_conclusion_model.py - PHASE_13_74_ADF CRR-6 tests.
Every dimension has a state oracle; the combination table is exercised on
its decisive cells including UNKNOWN-domination; the evidence gates
(r, n_pairs, coverage) are proven to gate - a strong r with three points
must NOT produce 'active_correlated'.
Run: pytest -q diagnostics/tests/test_conclusion_model.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import conclusion_model as cm  # noqa: E402


def jrec(bg_mean=2.0, bg_base=0.2, baseline_state="ok", r=None, n=10,
         cov=0.9, window_state="ok", progress_r=None):
    rec = {"run_id": "r1", "label": "job", "outcome": "success",
           "window": {"state": window_state},
           "baseline_state": baseline_state,
           "influence": [{"channel": "background_cpu_cores", "state": "ok",
                          "window_mean": bg_mean, "window_max": bg_mean,
                          "baseline_mean": bg_base}],
           "correlations": [], "progress": None}
    if r is not None:
        rec["correlations"].append({"job_metric": "job_rss",
                                    "host_metric": "background_cpu_cores",
                                    "state": "ok", "r": r, "n_pairs": n,
                                    "coverage_fraction": cov})
    if progress_r is not None:
        rec["progress"] = {"state": "ok", "correlations": [
            {"job_metric": "throughput", "host_metric": "background_cpu_cores",
             "state": "ok", "r": progress_r, "n_pairs": n,
             "coverage_fraction": cov}]}
    return rec


# ---- D1 ----
def test_host_state_oracle():
    assert cm.host_state("OK", []) == "clean"
    assert cm.host_state("IN-PROGRESS", []) == "in_progress"
    assert cm.host_state(None, []) == "unknown"
    assert cm.host_state("WARN", ["THP-01"]) == "stressed"
    assert cm.host_state("WARN", ["MEM-02"]) == "pathological"

# ---- D2 ----
def test_background_state_oracle_and_gates():
    assert cm.background_state(jrec(bg_mean=0.1, bg_base=0.1)) == "quiet"
    assert cm.background_state(jrec(bg_mean=0.1, baseline_state="pre_only")) \
        == "unknown_no_baseline"
    assert cm.background_state(jrec(bg_mean=3.0)) == "active_uncorrelated"
    assert cm.background_state(jrec(bg_mean=3.0, r=-0.9)) == "active_correlated"
    # the gates GATE: strong r with too few points / low coverage must NOT count
    assert cm.background_state(jrec(bg_mean=3.0, r=-0.99, n=3)) \
        == "active_uncorrelated"
    assert cm.background_state(jrec(bg_mean=3.0, r=-0.99, cov=0.2)) \
        == "active_uncorrelated"
    assert cm.background_state(jrec(bg_mean=3.0, r=0.3)) == "active_uncorrelated"
    # progress-throughput correlation is accepted evidence too
    assert cm.background_state(jrec(bg_mean=3.0, progress_r=-0.8)) \
        == "active_correlated"
    assert cm.background_state(jrec(window_state="no_overlap")) == "no_data"
    # ratio path: modest cores but 10x its own baseline -> active
    assert cm.background_state(jrec(bg_mean=0.8, bg_base=0.05)) \
        == "active_uncorrelated"

# ---- D3 + combination ----
def test_conclusions_decisive_cells():
    assert cm.conclude("clean", "quiet", "success")[0] == "CM-1"
    assert cm.conclude("pathological", "quiet", "success")[0] == "CM-2"
    assert cm.conclude("pathological", "quiet", "failed")[0] == "CM-3"
    assert cm.conclude("clean", "active_correlated", "failed")[0] == "CM-4"
    assert cm.conclude("clean", "quiet", "failed")[0] == "CM-5"
    assert cm.conclude("clean", "active_correlated", "success")[0] == "CM-6"
    assert cm.conclude("clean", "active_uncorrelated", "success")[0] == "CM-7"
    assert cm.conclude("stressed", "quiet", "success")[0] == "CM-8"

def test_unknown_domination():
    assert cm.conclude("unknown", "quiet", "success")[0] == "CM-U1"
    assert cm.conclude("in_progress", "active_correlated", "failed")[0] == "CM-U1"
    assert cm.conclude("clean", "no_data", "success")[0] == "CM-U2"
    assert cm.conclude("clean", "unknown_no_baseline", "success")[0] == "CM-U3"
    assert cm.conclude("clean", "quiet", "no_records")[0] == "CM-0"

def test_evaluate_bundle_worst_of_and_versioning():
    res = cm.evaluate("OK", [], [jrec(bg_mean=0.1, bg_base=0.1),
                                 jrec(bg_mean=3.0, r=-0.9)])
    assert res["model_version"] == cm.MODEL_VERSION
    assert res["thresholds"]["corr_r_min"] == cm.CORR_R_MIN
    codes = [r["code"] for r in res["records"]]
    assert "CM-1" in codes and "CM-6" in codes
    assert res["bundle_conclusion"]["code"] == "CM-6"      # worst-of wins
    assert all(r["job_state"] == "performance_unassessed"  # CRR-4 honesty
               for r in res["records"])

def test_evaluate_no_records():
    res = cm.evaluate("OK", [], [])
    assert res["bundle_conclusion"]["code"] == "CM-0"
