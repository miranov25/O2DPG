#!/usr/bin/env python3
"""
conclusion_model.py - PHASE_13_74_ADF CRR-6: the three-dimension conclusion
model, versioned.

The report previously carried only the HOST verdict (schema RULE_TABLE). The
benchmark question needs three independent dimensions, each with explicit
states including honest unknowns, combined by a versioned precedence table:

  D1 HOST STATE        (from the bundle verdict + fired rules)
       clean | stressed | pathological | in_progress | unknown
  D2 BACKGROUND STATE  (from job_host_analysis influence + correlations)
       quiet | active_uncorrelated | active_correlated
       | unknown_no_baseline | no_data
  D3 JOB STATE         (from the run record outcome)
       success | failed | no_records | performance_unassessed

D3 honesty: WITHOUT a reference-run comparison (the open CRR-4 remainder) a
successful job cannot be graded fast/slow; it is 'success' with performance
explicitly unassessed - this model does NOT pretend otherwise.

Combination: precedence UNKNOWN-dominant per dimension where knowledge is
required for the claim; conclusion codes are stable IDs (CM-x) so bundles
remain comparable across report versions. Any threshold change bumps
MODEL_VERSION. Statistical significance is not assessed (n_pairs and
coverage gates only).
"""
from __future__ import annotations

MODEL_VERSION = "1.0"

# versioned thresholds (bump MODEL_VERSION on any change)
BG_ACTIVE_CORES = 1.0          # mean background cores in window counted "active"
BG_ACTIVE_RATIO = 1.5          # or window/baseline ratio above this
CORR_R_MIN = 0.6               # |r| at or above -> correlation evidence
CORR_N_MIN = 6                 # minimum aligned pairs
CORR_COVERAGE_MIN = 0.5        # minimum fraction of job points aligned

HOST_STATES = ("clean", "stressed", "pathological", "in_progress", "unknown")
BG_STATES = ("quiet", "active_uncorrelated", "active_correlated",
             "unknown_no_baseline", "no_data")
JOB_STATES = ("success", "failed", "no_records", "unknown_outcome",
              "performance_unassessed")



# THE canonical code glosses - extracted from conclude()'s own return texts
# and asserted equal to them by test. Renderers MUST build any legend from
# this table, never hand-write one [UID-delta panel P1-3: a hand-written
# legend inverted CM-6's outcome].
CODE_LEGEND = {
    "CM-0": "No job records supplied: host assessment only; no job-environment conclusion possible.",
    "CM-1": "Job succeeded on a clean host with quiet background: the run qualifies as reference-grade for comparisons.",
    "CM-2": "Job succeeded but the host shows pathology: results are valid, runtimes are NOT comparable to healthy hosts.",
    "CM-3": "Job failed on a pathological host: investigate the host pathology before blaming the job.",
    "CM-4": "Job failed while background activity was high and correlated with the job's series: environment interference is a concrete suspect.",
    "CM-5": "Job failed on an unremarkable host with no background evidence: the job itself is the first suspect.",
    "CM-6": "Job succeeded; background was active AND correlated with the job's series: runtime comparisons should exclude or annotate this run.",
    "CM-7": "Job succeeded; background was active but shows no correlation with the job at the model's evidence gates: no interference demonstrated.",
    "CM-8": "Job succeeded on a stressed but non-pathological host with quiet background: acceptable for comparisons, with the stress rules noted.",
    "CM-U1": "Host state is not established (collector did not reach a final verdict): job-vs-host conclusion withheld.",
    "CM-U2": "Background could not be assessed for the job window: conclusion limited to the host verdict.",
    "CM-U3": "Background activity is measured but has no valid baseline: influence direction cannot be established.",
    "CM-U4": "The run record carries no outcome: job state is unknown, so no success-dependent conclusion is made."
}

def host_state(verdict, rules_fired):
    """D1 from the bundle's own versioned verdict machinery."""
    v = (verdict or "").upper()
    if v == "IN-PROGRESS":
        return "in_progress"
    if v == "UNKNOWN" or not v:
        return "unknown"
    warn = [r for r in (rules_fired or ())]
    if v in ("OK", "CLEAN") and not warn:
        return "clean"
    sev = {r: True for r in warn}
    if any(r.startswith(("MEM-", "PSI-", "STALL-")) for r in sev):
        return "pathological"
    return "stressed" if warn else "clean"


def background_state(jha_record):
    """D2 from one job_host_analysis record result."""
    w = (jha_record or {}).get("window") or {}
    if w.get("state") in (None, "no_window", "no_overlap",
                          "unsupported_record_type", "error"):
        return "no_data"
    inf = {e.get("channel"): e for e in jha_record.get("influence", [])}
    bg = inf.get("background_cpu_cores")
    if bg is None or bg.get("state") != "ok":
        return "no_data"
    baseline_ok = jha_record.get("baseline_state") == "ok"
    active = bg.get("window_mean", 0.0) >= BG_ACTIVE_CORES
    ratio = bg.get("baseline_mean")
    if not active and baseline_ok and ratio not in (None, 0):
        active = (bg["window_mean"] / ratio) >= BG_ACTIVE_RATIO
    if not active:
        return "quiet" if baseline_ok else "unknown_no_baseline"
    # active: is there correlation EVIDENCE meeting the versioned gates?
    pools = list(jha_record.get("correlations", []))
    prog = jha_record.get("progress") or {}
    if prog.get("state") == "ok":
        pools += prog.get("correlations", [])
    for c in pools:
        if (c.get("host_metric") == "background_cpu_cores"
                and c.get("state") == "ok"
                and c.get("r") is not None and abs(c["r"]) >= CORR_R_MIN
                and c.get("n_pairs", 0) >= CORR_N_MIN
                and c.get("coverage_fraction", 0.0) >= CORR_COVERAGE_MIN):
            return "active_correlated"
    return "active_uncorrelated"


def job_state(outcome):
    """D3 from the run record outcome. A missing outcome is UNKNOWN - the
    model never invents success [P0-E]. Performance grading NEEDS the
    reference-run comparison (open CRR-4 remainder) - not pretended here."""
    if outcome is None:
        return "unknown_outcome"
    if outcome == "success":
        return "success"
    return "failed"


# stable conclusion codes: (D1-class, D2, D3) -> (code, sentence)
def conclude(h, b, j):
    """Versioned precedence. UNKNOWN dominates wherever the claim needs the
    missing knowledge; codes CM-x are stable across report versions."""
    if j == "unknown_outcome":
        return ("CM-U4", CODE_LEGEND["CM-U4"])
    if j == "no_records":
        return ("CM-0", CODE_LEGEND["CM-0"])
    if h in ("in_progress", "unknown"):
        return ("CM-U1", CODE_LEGEND["CM-U1"])
    if b == "no_data":
        return ("CM-U2", CODE_LEGEND["CM-U2"])
    if b == "unknown_no_baseline":
        return ("CM-U3", CODE_LEGEND["CM-U3"])
    if j == "failed":
        if h == "pathological":
            return ("CM-3", CODE_LEGEND["CM-3"])
        if b == "active_correlated":
            return ("CM-4", CODE_LEGEND["CM-4"])
        return ("CM-5", CODE_LEGEND["CM-5"])
    # j == success
    if h == "pathological":
        return ("CM-2", CODE_LEGEND["CM-2"])
    if b == "active_correlated":
        return ("CM-6", CODE_LEGEND["CM-6"])
    if b == "active_uncorrelated":
        return ("CM-7", CODE_LEGEND["CM-7"])
    if h == "stressed":
        return ("CM-8", CODE_LEGEND["CM-8"])
    return ("CM-1", CODE_LEGEND["CM-1"])


def evaluate(verdict, rules_fired, jha_results):
    """Bundle-level evaluation -> versioned dict for report + trace."""
    h = host_state(verdict, rules_fired)
    out = {"model_version": MODEL_VERSION, "host_state": h, "records": [],
           "thresholds": {"bg_active_cores": BG_ACTIVE_CORES,
                          "bg_active_ratio": BG_ACTIVE_RATIO,
                          "corr_r_min": CORR_R_MIN, "corr_n_min": CORR_N_MIN,
                          "corr_coverage_min": CORR_COVERAGE_MIN}}
    if not jha_results:
        code, text = conclude(h, "no_data", "no_records")
        out["bundle_conclusion"] = {"code": code, "text": text}
        return out
    worst_rank, worst = -1, None
    severity = ["CM-1", "CM-8", "CM-7", "CM-0", "CM-U3", "CM-U2", "CM-U1",
                "CM-U4", "CM-6", "CM-2", "CM-5", "CM-4", "CM-3"]
    for r in jha_results:
        b = background_state(r)
        j = job_state(r.get("outcome"))     # P0-E: no default, unknown-dominant
        code, text = conclude(h, b, j)
        rec = {"run_id": r.get("run_id"), "label": r.get("label"),
               "background_state": b, "job_state":
                   ("performance_unassessed" if j == "success" else j),
               "record_role": r.get("record_role", "?"),
               "code": code, "conclusion": text}
        out["records"].append(rec)
        rank = severity.index(code) if code in severity else 0
        if rank > worst_rank:
            worst_rank, worst = rank, rec
    out["bundle_conclusion"] = {"code": worst["code"], "text": worst["conclusion"],
                                "from_run": worst["run_id"]}
    return out
