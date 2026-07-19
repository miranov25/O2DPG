#!/usr/bin/env python3
"""
job_host_analysis.py - PHASE_13_74_ADF: job-vs-host analysis (Layer-C
FOUNDATION - alignment, background influence, correlations; NOT full Layer-C
closure: reference-run comparison remains open in the CRR matrix).

Renamed from layer_c.py on review (GPT18 2026-07-18): the module is named for
its responsibility, not for a proposal section.

TIME CONVENTION (proven by test JH-1 against run_metrics by execution):
  the record's `utc` stamp is written at __exit__  ->  END of the run;
  start = utc_epoch - time.wall_s;  every t_rel_s (rss_series, events) is
  relative to START. Getting this wrong shifts every result by one full run
  duration - the exact defect the review caught in the first draft.

Record types (v8 correction P0-1): only `record_type == "in_process"` carries
series; an orchestration record (tool == "dfx_run_with_diagnostics") yields a
window from its step timestamps; anything else -> `unsupported_record_type`.

Every quantity carries an explicit state, never a bare number. Statistical
significance (p-values) is NOT provided; n_pairs and coverage_fraction are
reported so a 3-point correlation can never impersonate a 300-point one.
"""
from __future__ import annotations

import calendar
import json
import time
from pathlib import Path

MIN_POINTS = 4
ALIGN_TOL_S = 5.0
ALIGNMENT_METHOD = "nearest_within_tolerance"


# ----------------------------- primitives -----------------------------------
def safe_corr(x, y):
    """Pearson r with CRR-14 semantics: (state, r|None, n_pairs).
    Zero-variance input is CLASSIFIED, never computed-and-warned."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < MIN_POINTS:
        return ("insufficient_overlap", None, n)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return ("unavailable_constant", None, n)
    return ("ok", float(np.corrcoef(x, y)[0, 1]), n)


def _utc_epoch(stamp):
    try:
        return calendar.timegm(time.strptime(stamp, "%Y%m%dT%H%M%SZ"))
    except Exception:
        return None


def load_run_record(path_or_dict):
    if isinstance(path_or_dict, dict):
        return path_or_dict
    return json.loads(Path(path_or_dict).read_text())


def background_series(rollup_df):
    """workload_rollup -> per-ts background CPU cores (NOT the target job)."""
    bg = rollup_df[rollup_df["scope"].isin(
        ["current_user_non_job", "other_visible_workloads"])]
    return bg.groupby("ts")["cpu_cores"].sum()


def classify_background(bg_series):
    """Tercile classes over the bundle: value -> low / normal / high."""
    import numpy as np
    v = bg_series.dropna()
    if len(v) < 3 or float(v.max()) == float(v.min()):
        return None, None
    lo, hi = np.percentile(v, [33.3, 66.7])
    def cls(x):
        return "low" if x <= lo else ("high" if x >= hi else "normal")
    return cls, (float(lo), float(hi))


def _window(rec):
    """(state, start, end) with the proven END-stamp convention."""
    rtype = rec.get("record_type")
    if rtype == "in_process":
        end = _utc_epoch(rec.get("utc", ""))
        wall = (rec.get("time") or {}).get("wall_s")
        if end is None or wall is None:
            return ("no_window", None, None)
        return ("ok", end - float(wall), float(end))
    if rec.get("tool") == "dfx_run_with_diagnostics":       # external record
        ts = {s.get("step"): s.get("t") for s in rec.get("steps", [])}
        if ts.get("workload_start") and ts.get("workload_end"):
            return ("ok_external", float(ts["workload_start"]),
                    float(ts["workload_end"]))
        return ("no_window", None, None)
    return ("unsupported_record_type", None, None)


def _align(abs_t, values, host_ts, tol=ALIGN_TOL_S):
    """Nearest host tick within tolerance; NO extrapolation. Returns
    (idx, mask, coverage_fraction over the supplied points)."""
    import numpy as np
    abs_t = np.asarray(abs_t, dtype=float)
    idx = np.abs(abs_t[:, None] - host_ts[None, :]).argmin(axis=1)
    ok = np.abs(abs_t - host_ts[idx]) <= tol
    cov = float(ok.sum()) / len(abs_t) if len(abs_t) else 0.0
    return idx, ok, cov


def _corr_row(job_metric, host_metric, jv, hv, coverage):
    state, r, n = safe_corr(jv, hv)
    return {"job_metric": job_metric, "host_metric": host_metric,
            "state": state, "r": None if r is None else round(r, 3),
            "n_pairs": n, "coverage_fraction": round(coverage, 3),
            "time_tolerance_s": ALIGN_TOL_S, "alignment_method": ALIGNMENT_METHOD}


# ------------------------------- analysis -----------------------------------
def analyze_record(rec, host_df, rollup_df=None, channels=None):
    import numpy as np
    out = {"label": rec.get("label"), "run_id": rec.get("run_id"),
           "outcome": rec.get("outcome"),          # P0-E: carried, never invented
           "record_type": rec.get("record_type") or rec.get("tool") or "unknown",
           "window": None, "baseline_state": None, "influence": [],
           "correlations": [], "progress": None, "stages": None}
    wstate, t0, t1 = _window(rec)
    if wstate in ("no_window", "unsupported_record_type"):
        out["window"] = {"state": wstate}
        return out
    ts = host_df["ts"].astype(float)
    inw = host_df[(ts >= t0) & (ts <= t1)]
    pre = host_df[ts < t0]
    post = host_df[ts > t1]
    out["window"] = {"state": wstate if len(inw) else "no_overlap",
                     "t0": t0, "t1": t1,
                     "host_samples_in_window": int(len(inw)),
                     "baseline_pre": int(len(pre)), "baseline_post": int(len(post))}
    if len(inw) == 0:
        return out
    nb = len(pre) + len(post)
    out["baseline_state"] = ("no_outside_window" if nb == 0 else
                             "insufficient_baseline" if nb < MIN_POINTS else
                             "pre_only" if len(post) == 0 else
                             "post_only" if len(pre) == 0 else "ok")

    channels = channels or [c for c in
                            ("cpu_busy_pct", "loadavg1", "psi_mem_some_avg10",
                             "compact_stall_per_s", "allocstall_per_s")
                            if c in host_df.columns]
    outw = host_df[(ts < t0) | (ts > t1)]
    for c in channels:
        wi = inw[c].dropna()
        ba = outw[c].dropna()
        if len(wi) == 0:
            out["influence"].append({"channel": c, "state": "no_data"})
            continue
        e = {"channel": c, "state": "ok", "baseline_state": out["baseline_state"],
             "window_mean": float(wi.mean()), "window_max": float(wi.max()),
             "baseline_mean": float(ba.mean()) if len(ba) else None}
        if e["baseline_mean"] not in (None, 0):
            e["window_over_baseline"] = round(e["window_mean"] / e["baseline_mean"], 3)
        out["influence"].append(e)
    bg = None
    if rollup_df is not None and len(rollup_df):
        bg = background_series(rollup_df)
        bgw = bg[(bg.index >= t0) & (bg.index <= t1)]
        if len(bgw):
            rest = bg.drop(bgw.index)
            out["influence"].append({"channel": "background_cpu_cores",
                                     "state": "ok",
                                     "baseline_state": out["baseline_state"],
                                     "window_mean": float(bgw.mean()),
                                     "window_max": float(bgw.max()),
                                     "baseline_mean": float(rest.mean()) if len(rest) else None})

    if out["record_type"] != "in_process":
        out["correlations"].append(_corr_row("(external record)", "*", [], [], 0.0)
                                   | {"state": "no_series"})
        return out

    # ---- per-tick correlations: rss_series -----------------------------------
    series = rec.get("rss_series") or []
    hts = inw["ts"].astype(float).to_numpy()
    if len(series) < MIN_POINTS:
        out["correlations"].append({"job_metric": "job_rss", "host_metric": "*",
                                    "state": "no_series", "n_pairs": len(series),
                                    "r": None, "coverage_fraction": 0.0,
                                    "time_tolerance_s": ALIGN_TOL_S,
                                    "alignment_method": ALIGNMENT_METHOD})
    elif len(hts) < MIN_POINTS:
        out["correlations"].append({"job_metric": "job_rss", "host_metric": "*",
                                    "state": "insufficient_overlap",
                                    "n_pairs": int(len(hts)), "r": None,
                                    "coverage_fraction": 0.0,
                                    "time_tolerance_s": ALIGN_TOL_S,
                                    "alignment_method": ALIGNMENT_METHOD})
    else:
        st = np.array([t0 + float(p[0]) for p in series])
        sv = np.array([float(p[1]) for p in series])
        idx, okm, cov = _align(st, sv, hts)
        for c in channels:
            hv = inw[c].to_numpy(dtype=float)[idx]
            out["correlations"].append(_corr_row("job_rss", c, sv[okm], hv[okm], cov))
        if bg is not None and len(bg) >= MIN_POINTS:
            bts = bg.index.to_numpy(dtype=float)
            bidx, bok, bcov = _align(st, sv, bts)
            out["correlations"].append(_corr_row(
                "job_rss", "background_cpu_cores",
                sv[bok], bg.to_numpy(dtype=float)[bidx][bok], bcov))

    # ---- progress-derived throughput (events seam) ---------------------------
    ev = rec.get("events") or []
    prog = [(e["t_rel_s"], e["payload"].get("value"))
            for e in ev if e.get("kind") == "progress"
            and isinstance(e.get("payload"), dict)
            and isinstance(e["payload"].get("value"), (int, float))]
    if len(prog) < 3:
        out["progress"] = {"state": "no_progress_events", "n_events": len(prog)}
    else:
        tt = np.array([t0 + p[0] for p in prog])
        vv = np.array([p[1] for p in prog], dtype=float)
        dt = np.diff(tt); dv = np.diff(vv)
        good = dt > 0
        thr = dv[good] / dt[good]
        mid = (tt[1:] + tt[:-1])[good] / 2.0
        out["progress"] = {"state": "ok", "n_events": len(prog),
                           "throughput_mean": float(thr.mean()),
                           "correlations": []}
        idx, okm, cov = _align(mid, thr, hts)
        for c in channels:
            hv = inw[c].to_numpy(dtype=float)[idx]
            out["progress"]["correlations"].append(
                _corr_row("throughput", c, thr[okm], hv[okm], cov))
        if bg is not None and len(bg) >= MIN_POINTS:
            bts = bg.index.to_numpy(dtype=float)
            bidx, bok, bcov = _align(mid, thr, bts)
            out["progress"]["correlations"].append(_corr_row(
                "throughput", "background_cpu_cores",
                thr[bok], bg.to_numpy(dtype=float)[bidx][bok], bcov))

    # ---- stage durations vs background class ---------------------------------
    starts, stages = {}, []
    for e in ev:
        if e.get("kind") == "stage_start":
            starts[e["payload"].get("stage")] = e["t_rel_s"]
        elif e.get("kind") == "stage_end":
            sname = e["payload"].get("stage")
            if sname in starts:
                stages.append((sname, t0 + starts.pop(sname), t0 + e["t_rel_s"]))
    if not stages:
        out["stages"] = {"state": "no_stage_events"}
    else:
        cls, cuts = (classify_background(bg) if bg is not None else (None, None))
        rows = []
        for sname, s0, s1 in stages:
            row = {"stage": sname, "duration_s": round(s1 - s0, 3)}
            if cls is not None:
                bgw = bg[(bg.index >= s0) & (bg.index <= s1)]
                row["background_class"] = (cls(float(bgw.mean())) if len(bgw)
                                           else "no_background_overlap")
            else:
                row["background_class"] = "background_constant_or_absent"
            rows.append(row)
        out["stages"] = {"state": "ok", "tercile_cuts": cuts, "rows": rows}
    return out


def analyze(records, host_df, rollup_df=None):
    """All records (paths or dicts) vs one bundle; duplicates of the same
    run_id are all analyzed and labeled - selection is the reader's, not
    silently ours. Never raises."""
    results = []
    for rp in records:
        try:
            results.append(analyze_record(load_run_record(rp), host_df, rollup_df))
        except Exception as e:
            results.append({"label": str(rp), "record_type": "unknown",
                            "outcome": None,
                            "window": {"state": "error",
                                       "detail": f"{type(e).__name__}: {e}"},
                            "baseline_state": None,
                            "influence": [], "correlations": [],
                            "progress": None, "stages": None})
    return results
