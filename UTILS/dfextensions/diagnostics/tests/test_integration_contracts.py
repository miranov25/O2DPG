#!/usr/bin/env python3
"""
test_integration_contracts.py - PHASE_13_74_ADF correction pass
(CRR-v3 panel P0-A/B/C/E): the tests the panel said were missing - they
exercise the PRESENCE path across module seams, not the absence path.

IC-1 is the P0-B catcher: it feeds _render_layer_c the REAL output of
job_host_analysis.analyze + conclusion_model.evaluate. Written BEFORE the
fix; it must fail with KeyError on the unfixed renderer.
Run: pytest -q diagnostics/tests/test_integration_contracts.py
"""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pd = pytest.importorskip("pandas")
import conclusion_model as cm  # noqa: E402
import job_host_analysis as jha  # noqa: E402
import report_diagnostics as rd  # noqa: E402

T0 = 1_000_000
WALL = 60.0


def real_record(outcome="success"):
    return {"label": "job", "run_id": "rX", "record_type": "in_process",
            "utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(T0 + WALL)),
            "time": {"wall_s": WALL}, "outcome": outcome,
            "rss_series": [(i * 5.0, 1000 + 100 * i) for i in range(13)]}


def host_frame():
    return pd.DataFrame({"ts": [T0 - 20 + i * 5 for i in range(30)],
                         "cpu_busy_pct": [10 + i for i in range(30)]})


def test_ic1_renderer_accepts_real_engine_output():
    """P0-B: the renderer must consume what the engine actually produces -
    proven by rendering REAL analyze() output, never a hand-written fixture."""
    res = jha.analyze([real_record()], host_frame())
    concl = cm.evaluate("OK", [], res)
    html = rd._render_layer_c("alma2", res, concl)
    assert "job window" in html and "baseline" in html
    # every correlation row rendered with its real metadata
    assert "job_rss" in html and "cpu_busy_pct" in html
    assert "n_pairs" in html or "n</th>" in html

def test_ic2_renderer_survives_every_window_state():
    hf = host_frame()
    records = [real_record(),                                   # ok
               {"label": "far", "run_id": "r2", "record_type": "in_process",
                "utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(T0 + 10_000)),
                "time": {"wall_s": 5}, "rss_series": []},        # no_overlap
               {"record_type": "martian"}]                       # unsupported
    res = jha.analyze(records, hf)
    concl = cm.evaluate("OK", [], res)
    html = rd._render_layer_c("alma2", res, concl)
    assert "no_overlap" in html and "unsupported_record_type" in html

def test_ic3_outcome_carried_and_failed_job_never_reference_grade():
    """P0-E: a FAILED job must never conclude CM-1; missing outcome must be
    unknown-dominant, never defaulted to success."""
    res = jha.analyze([real_record(outcome="exception")], host_frame())
    assert res[0].get("outcome") == "exception"          # carried through
    concl = cm.evaluate("OK", [], res)
    assert concl["records"][0]["code"] != "CM-1"
    assert concl["records"][0]["job_state"] == "failed"
    # missing outcome -> unknown, never success
    r2 = real_record(); r2.pop("outcome")
    concl2 = cm.evaluate("OK", [], jha.analyze([r2], host_frame()))
    assert concl2["records"][0]["code"] == "CM-U4"
    assert concl2["records"][0]["job_state"] == "unknown_outcome"

# IC-4 removed: it pre-created orchestration.json and tested the helper in
# isolation, missing the wrapper sequence defect [GPT24]. Superseded by the
# executed vertical suite in test_wrapper_vertical.py (VT-1..VT-4).


def test_ic5_audit_persisted_after_conclusion_stages(tmp_path):
    """P0-C: generate() must write the audit AFTER S7/S8; verified
    structurally on the source: the finalize call must come after the
    Layer-C block in the function body."""
    src = Path(rd.__file__).read_text()
    body = src[src.find("def generate("):]
    i_layer = body.find('_render_layer_c(')
    i_fin = body.rfind('_finalize_audit(aud')
    assert i_layer != -1 and i_fin != -1
    assert i_fin > i_layer, "audit persisted before Layer-C/conclusion stages ran"
    helper = src[src.find("def _finalize_audit("):src.find("def generate(")]
    assert "aud.write(" in helper and "check_stage_coverage" in helper
