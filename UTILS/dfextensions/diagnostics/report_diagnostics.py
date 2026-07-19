#!/usr/bin/env python3
"""
report_diagnostics.py - PHASE_13_74_ADF D4: CSV -> ADF -> Draw, CSV -> ADF -> Summary.

Loads one or more D1 diagnostic bundles into AliasDataFrame, derives severity
bands as ALIASES, renders time-series figures through adf.draw (dfdraw) and a
summary table, and writes report.html + report_summary.json (+ report_it.md in
it_report mode). The architect's pattern verbatim (T-1).

Frozen public signature (R-14, with the panel-mandated `mode` amendment V2-2):
    generate(bundles, run_records=(), labels=None, sections=None,
             out_dir=".", mode="technical") -> Path

NOTE (T-R8, honest): AliasDataFrame has no generic data-statistics describe -
describe_lazy()/describe_structure() are state/structure surfaces. The summary
table is therefore computed report-locally from the materialized frame
(adf.df[cols].describe()-equivalent) and verified against an independent
pandas oracle in tests (T-R7).

AliasDataFrame/dfdraw are imported lazily inside generate(): schema-level
functions in this module stay importable on hosts without the ADF stack.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")   # headless rendering (D4 contract)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# adf.draw imports dfextensions.dfdraw -> the PARENT of dfextensions must be on
# sys.path when this file is run as a plain script (pytest adds it, CLI doesn't)
_UTILS = Path(__file__).resolve().parent.parent.parent
if (_UTILS / "dfextensions" / "dfdraw").is_dir():
    sys.path.insert(0, str(_UTILS))
import schema  # noqa: E402
import audit as audit_mod  # noqa: E402
import job_host_analysis as jha_mod  # noqa: E402
import conclusion_model as cm_mod  # noqa: E402

# channels drawn/summarized when present (rates + PSI are the pathology signals)
DEFAULT_CHANNELS = [
    "compact_stall_per_s", "compact_fail_per_s", "thp_fault_alloc_per_s",
    "thp_fault_fallback_per_s", "allocstall_per_s",
    "kcompactd_cpu_per_s", "khugepaged_cpu_per_s",
    "psi_mem_some_avg10", "psi_mem_full_avg10",
    # derived by schema.samples_frame from *_total columns (coverage-gap fix):
    "pgscan_direct_per_s", "disk_read_sectors_per_s",
    "pswpin_per_s", "pswpout_per_s", "thp_collapse_alloc_per_s",
]
DEFAULT_SECTIONS = ["environment", "vitals", "processes", "rates", "psi",
                    "runs", "crosshost", "evidence"]

# v8 output rule 1: host vitals are ALWAYS drawn when present - never subject
# to plot-what-moves (a flat vital is itself information; page-1 dashboard)
VITAL_PANELS = [
    ("cpu_busy_pct",              "CPU busy [% of all cores]"),
    ("[loadavg1,procs_running]",  "load (1min) and runnable processes"),
    ("mem_available_gb",          "MemAvailable [GB]"),
    ("ctxt_per_s",                "context switches / s"),
]

# severity-band alias expressions (registered on the ADF - T-R8 exact path)
# the trailing +0*<input> term propagates NaN: severity of a missing rate is
# NaN, never a silent 0 (first sample row has NaN rates by design)
SEVERITY_ALIASES = {
    "sev_mem_pressure":
        "(psi_mem_full_avg10>=5.0)*2 + ((psi_mem_full_avg10>=1.0)&(psi_mem_full_avg10<5.0))*1"
        " + 0*psi_mem_full_avg10",
    "sev_compaction":
        "(compact_stall_per_s>=10.0)*2 + ((compact_stall_per_s>=1.0)&(compact_stall_per_s<10.0))*1"
        " + 0*compact_stall_per_s",
}


# ---------------------- process/user pivots (v8 D4) ------------------------
def _san(name):
    out = "".join(c if c.isalnum() else "_" for c in str(name))
    return out if out and not out[0].isdigit() else "t_" + out

def _pivot_users(udf, k=4):
    """user_samples -> wide frame: t_rel + cpu_<tenant> [cores] + rss_gb_<tenant>.
    Keeps the current user ALWAYS + top-k other tenants by window CPU."""
    udf = udf.copy()
    udf["t_rel"] = udf["ts"] - udf["ts"].min()
    tot = udf.groupby("tenant")["cpu_cores"].sum().sort_values(ascending=False)
    own = set(udf.loc[udf["is_current_user"] == 1, "tenant"])
    keep = list(own) + [t for t in tot.index if t not in own][:k]
    wide, names = None, {}
    import pandas as pd
    for t in keep:
        sub = udf[udf["tenant"] == t].set_index("t_rel")
        col_c, col_r = f"cpu_{_san(t)}", f"rss_gb_{_san(t)}"
        names[t] = col_c
        f = pd.DataFrame({col_c: sub["cpu_cores"],
                          col_r: sub["rss_kb"] / 1048576.0})
        wide = f if wide is None else wide.join(f, how="outer")
    wide = wide.reset_index()
    return wide, names

def _pivot_rollup(wdf):
    """workload_rollup -> wide frame: t_rel + cpu cores per scope."""
    import pandas as pd
    wdf = wdf.copy()
    wdf["t_rel"] = wdf["ts"] - wdf["ts"].min()
    wide = None
    for scope in ("target_job", "current_user_non_job", "other_visible_workloads"):
        sub = wdf[wdf["scope"] == scope].set_index("t_rel")
        f = pd.DataFrame({f"cpu_{scope}": sub["cpu_cores"]})
        wide = f if wide is None else wide.join(f, how="outer")
    return wide.reset_index()

def _top_consumers(pdf, n=10):
    """process_samples -> window top-n table rows (tenant, proc, max cpu, max rss)."""
    g = pdf.groupby(["tenant", "proc"]).agg(
        cpu_max=("cpu_pct", "max"), cpu_mean=("cpu_pct", "mean"),
        rss_max_kb=("rss_kb", "max"), samples=("ts", "count")).reset_index()
    g = g.sort_values(["cpu_max", "rss_max_kb"], ascending=False).head(n)
    return [dict(r) for _, r in g.iterrows()]


RENDER_WARNINGS = 0     # GPT18-F: drawing-library warnings are counted and

                        # reported, never silently flooded to stderr


def _draw_expr(adf, expr, title, ylab, fig_dir, fname):
    """One draw through the ADF/dfdraw grammar (incl. [a,b]:x multi-curve),
    individual-curve fallback if the vector form is rejected."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    import warnings as _w
    try:
      with _w.catch_warnings(record=True) as _wl:
        _w.simplefilter("always")
        fig, ax, _ = adf.draw(f"{expr}:t_rel", type="scatter", title=title,
                              xlabel="t_rel [s]", ylabel=ylab)
        p = fig_dir / fname
        fig.savefig(p, dpi=110, bbox_inches="tight")
        global RENDER_WARNINGS
        RENDER_WARNINGS += len(_wl)
        try:
            import matplotlib.pyplot as plt; plt.close(fig)
        except Exception:
            pass
        return [p.name]
    except Exception:
        if expr.startswith("["):        # fallback: split the vector
            names = []
            for one in expr.strip("[]").split(","):
                names += _draw_expr(adf, one.strip(), f"{title}: {one}", one,
                                    fig_dir, fname.replace(".png", f"_{_san(one)}.png"))
            return names
        return []


def _aux_tables(bundle):
    """process/user tables if the bundle has them (v8 D1.6); else (None, None)."""
    import pandas as pd
    pp = bundle.path / "process_samples.csv"
    up = bundle.path / "user_samples.csv"
    wp = bundle.path / "workload_rollup.csv"
    pdf_ = udf_ = wdf_ = None
    try:
        if pp.is_file() and len(pp.read_text().splitlines()) > 1:
            pdf_ = pd.read_csv(pp)
        if up.is_file() and len(up.read_text().splitlines()) > 1:
            udf_ = pd.read_csv(up)
        if wp.is_file() and len(wp.read_text().splitlines()) > 1:
            wdf_ = pd.read_csv(wp)
    except Exception:
        pass
    return pdf_, udf_, wdf_


def _summarize(df, cols):
    """Report-local summary: per-channel count/mean/std/min/max/last on VALID
    rows. Pure-pandas on the materialized frame; oracle-tested (T-R7/T-R8)."""
    out = {}
    valid = df[df["row_valid"]] if "row_valid" in df.columns else df
    for c in cols:
        if c not in valid.columns:
            continue
        s = valid[c].dropna()
        if len(s) == 0:
            out[c] = {"count": 0}
            continue
        out[c] = {
            "count": int(s.count()), "mean": float(s.mean()),
            "std": float(s.std()) if s.count() > 1 else 0.0,
            "min": float(s.min()), "max": float(s.max()),
            "last": float(s.iloc[-1]),
        }
    return out


def _build_adf(frame):
    """DataFrame -> AliasDataFrame with severity-band aliases materialized.
    Exact T-R8 path: ADF -> add_alias -> materialize -> use adf.df."""
    try:
        from AliasDataFrame import AliasDataFrame  # locked stack (pandas 1.5.3)
    except ImportError:
        _sib = Path(__file__).resolve().parent.parent / "AliasDataFrame"
        if _sib.is_dir():
            sys.path.insert(0, str(_sib))
        from AliasDataFrame import AliasDataFrame
    adf = AliasDataFrame(frame)
    reg = []
    for name, expr in SEVERITY_ALIASES.items():
        cols_needed = [c for c in DEFAULT_CHANNELS if c in expr]
        if all(c in frame.columns and frame[c].notna().any() for c in cols_needed):
            adf.add_alias(name, expr)
            reg.append(name)
    if reg:
        adf.materialize_aliases(names=reg)
    return adf, reg


def _draw_series(adf, cols, fig_dir, host):
    """Time-series figures through adf.draw (dfdraw path, T-R3)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for c in cols:
        if c not in adf.df.columns or not adf.df[c].notna().any():
            continue
        fig, ax, _stats = adf.draw(f"{c}:t_rel", type="scatter",
                                   title=f"{host}: {c}",
                                   xlabel="t_rel [s]", ylabel=c)
        p = fig_dir / f"{host}_{c}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        written.append(p.name)
        try:
            import matplotlib.pyplot as plt
            plt.close(fig)
        except Exception:
            pass
    return written


def _img_datauri(path):
    """Embed a figure as a base64 data-URI: report.html stays valid when moved
    or mailed WITHOUT its figures/ directory (2026-07-16 'empty report' lesson)."""
    import base64
    return ("data:image/png;base64," +
            base64.b64encode(Path(path).read_bytes()).decode("ascii"))


def _channels_to_draw(df, cols):
    """Split channels into (active, flat_zero, no_data) for the window.
    Flat-zero channels are summarized in one line instead of an empty-looking
    plot each - on an affected host only the pathological channels plot."""
    active, flat, nodata = [], [], []
    valid = df[df["row_valid"]] if "row_valid" in df.columns else df
    for c in cols:
        if c not in valid.columns:
            continue
        s = valid[c].dropna()
        if len(s) == 0:
            nodata.append(c)
        elif (s == 0).all():
            flat.append(c)
        else:
            active.append(c)
    return active, flat, nodata


def _html(title, sections):
    body = "\n".join(sections)
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{title}</title><style>body{{font-family:sans-serif;max-width:1100px;"
            "margin:auto}table{border-collapse:collapse}td,th{border:1px solid #999;"
            "padding:3px 8px;font-size:13px}img{max-width:100%}</style></head>"
            f"<body><h1>{title}</h1>\n{body}\n</body></html>")

def _table(d, headers=("key", "value")):
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in d.items())
    return (f"<table><tr>{''.join(f'<th>{h}</th>' for h in headers)}</tr>{rows}</table>")


def _it_report(bundles, summaries, out_dir):
    """C-5/D4-IT: one-page plain-language admin report. Redaction is MANDATORY
    in this mode; per-user aggregates suppressed below the k-anonymity floor."""
    lines = ["# Infrastructure incident report (generated by dfx diagnostics)", ""]
    for b in bundles:
        if b.redaction == "raw":
            raise ValueError(
                f"{b.path}: bundle recorded redaction=raw - it_report mode requires "
                "shareable (redacted) bundles; re-collect without -r (checklist C-1)")
        lines += [f"## Host {b.host}  ({b.manifest.get('utc','')})",
                  f"- Verdict: **{b.verdict}**  (rules:{' ' + ' '.join(b.rules_fired) if b.rules_fired else ' none'};"
                  f" rule table {b.manifest.get('rule_table_version', schema.RULE_TABLE_VERSION)})",
                  f"- Bundle fingerprint: tool_md5={b.manifest.get('tool_md5','?')}, "
                  f"collected as user {b.manifest.get('user','?')}, redaction={b.redaction}"]
        for s in schema.select_impact(b.rules_fired, mode="it_report"):
            lines.append(f"- {s}")
        summ = summaries.get(b.host, {})
        psi = summ.get("psi_mem_full_avg10")
        if psi and psi.get("count"):
            lines.append(f"- Measured PSI memory-full avg10 over the window: "
                         f"mean {psi['mean']:.2f}%, max {psi['max']:.2f}% "
                         f"(kernel-native stall-time accounting, all users).")
        lines.append("- Our-job vs host-level: the evidence above is SYSTEM-WIDE kernel "
                     "accounting; it does not attribute load to any user. Our own job "
                     "metrics are reported separately (run panel) and are the impact, "
                     "not the cause claim.")
        lines.append(f"- Privacy: no other user's process names, command lines or "
                     f"identities are included; per-user aggregates are suppressed below "
                     f"{schema.K_ANONYMITY_MIN_USERS} concurrent users (k-anonymity floor).")
        lines.append("")
    lines += ["## Requested action",
              "Please review the flagged host-level items (questions embedded per finding "
              "above). This report diagnoses only - any remediation is an admin decision.", ""]
    p = Path(out_dir) / "report_it.md"
    p.write_text("\n".join(lines))
    return p


# ============================ public API (frozen) ===========================

def _render_layer_c(host, res, concl):
    """Render one bundle's Layer-C section from job_host_analysis results and
    the conclusion-model evaluation. Extracted from generate() [CRR-15]; the
    renderer<->engine contract is enforced by tests that feed this function
    REAL analyze() output [P0-B fix: keys = the engine's actual schema]."""
    rows_html = []
    for r in res:
        w = r.get("window") or {}
        rows_html.append(f"<h3>{host} - run '{r.get('label') or r.get('run_id') or 'unlabeled'}' "
                         f"(run_id {r.get('run_id')}, {r.get('record_type')}) "
                         f"- window {w.get('state')}</h3>")
        if w.get("state") in ("ok", "ok_external"):
            rows_html.append(
                f"<p>{w.get('host_samples_in_window', 0)} host samples in the "
                f"job window; baseline: {w.get('baseline_pre', 0)} before + "
                f"{w.get('baseline_post', 0)} after "
                f"(baseline state: {r.get('baseline_state')}).</p>")
            inf = "".join(
                f"<tr><td>{e.get('channel')}</td><td>{e.get('state')}</td>"
                f"<td>{e.get('window_mean','')}</td><td>{e.get('window_max','')}</td>"
                f"<td>{e.get('baseline_mean','')}</td>"
                f"<td>{e.get('window_over_baseline','')}</td></tr>"
                for e in r.get("influence", []))
            rows_html.append("<table><tr><th>channel</th><th>state</th>"
                             "<th>window mean</th><th>window max</th>"
                             "<th>baseline mean</th><th>window/baseline</th></tr>"
                             + inf + "</table>")
            cor = "".join(
                f"<tr><td>{c.get('job_metric')}~{c.get('host_metric')}</td>"
                f"<td>{c.get('state')}</td>"
                f"<td>{'' if c.get('r') is None else c['r']}</td>"
                f"<td>{c.get('n_pairs','')}</td>"
                f"<td>{c.get('coverage_fraction','')}</td></tr>"
                for c in r.get("correlations", []))
            rows_html.append("<table><tr><th>job~host pair</th><th>validity</th>"
                             "<th>r</th><th>n_pairs</th><th>coverage</th></tr>"
                             + cor + "</table>")
            prog = r.get("progress")
            if prog and prog.get("state") == "ok":
                pc = "".join(
                    f"<tr><td>{c.get('job_metric')}~{c.get('host_metric')}</td>"
                    f"<td>{c.get('state')}</td>"
                    f"<td>{'' if c.get('r') is None else c['r']}</td>"
                    f"<td>{c.get('n_pairs','')}</td></tr>"
                    for c in prog.get("correlations", []))
                rows_html.append(
                    f"<p>Progress: {prog.get('n_events')} events, mean throughput "
                    f"{prog.get('throughput_mean'):.4g}/s.</p>"
                    "<table><tr><th>pair</th><th>validity</th><th>r</th>"
                    "<th>n_pairs</th></tr>" + pc + "</table>")
            elif prog:
                rows_html.append(f"<p>Progress: {prog.get('state')}.</p>")
            st = r.get("stages")
            if st and st.get("state") == "ok":
                sr = "".join(f"<tr><td>{x['stage']}</td><td>{x['duration_s']}</td>"
                             f"<td>{x['background_class']}</td></tr>"
                             for x in st.get("rows", []))
                rows_html.append("<table><tr><th>stage</th><th>duration s</th>"
                                 "<th>background class</th></tr>" + sr + "</table>")
        elif w.get("detail"):
            rows_html.append(f"<p>{w['detail']}</p>")
    bc = concl["bundle_conclusion"]
    # identical conclusions from components of the same run collapse into ONE
    # row listing the component roles [round-4: duplicate CM-row finding]
    merged = {}
    for r in concl["records"]:
        k = (r.get("run_id"), r["background_state"], r["job_state"], r["code"])
        merged.setdefault(k, {"r": r, "roles": []})["roles"].append(
            r.get("record_role", "?"))
    per = "".join(
        f"<tr><td>{k[0]}</td><td>{'+'.join(v['roles'])}</td><td>{k[1]}</td>"
        f"<td>{k[2]}</td><td>{k[3]}</td><td>{v['r']['conclusion']}</td></tr>"
        for k, v in merged.items())
    guidance = ""
    if str(bc["code"]).startswith("CM-U"):
        guidance = ("<p><i>Inconclusive: collect a longer run with non-zero "
                    "pre/post baseline windows (wrapper --pre/--post) for a "
                    "decisive read.</i></p>")
    rows_html.append(
        f"<h3>Conclusion (model v{concl['model_version']}, host state: "
        f"{concl['host_state']})</h3>"
        "<table><tr><th>run</th><th>components</th><th>background</th>"
        "<th>job</th><th>code</th><th>conclusion</th></tr>" + per + "</table>"
        f"<p><b>Bundle conclusion [{bc['code']}]</b>: {bc['text']}</p>"
        + guidance +
        "<p><i>Note: correlation tables above may show real correlations "
        "against channels (e.g. cpu_busy_pct) that are NOT eligible conclusion "
        "inputs - the background verdict is gated on background_cpu_cores "
        "specifically, so the job's own activity is not read as interference."
        "</i></p>"
        "<details><summary>CM code legend (from conclusion_model.CODE_LEGEND)"
        "</summary><p>" +
        " ".join(f"<b>{c}</b>: {t}" for c, t in
                 __import__("conclusion_model").CODE_LEGEND.items()) +
        "<br>Evidence states: unavailable_constant = channel present but "
        "constant (no signal), ok = evaluated, absent = not collected on "
        "this platform.</p></details>")
    return "<h2>Layer-C: job vs background</h2>" + "".join(rows_html)


def _finalize_audit(aud, out_dir, loaded, mode):
    """Persist the audit LAST, after every stage has run [P0-C fix: the
    previous call site wrote the audit before the Layer-C and conclusion
    stages executed, so S7/S8 and the conclusion trace never reached the
    evidence]. Called at every generate() exit that has an audit."""
    if aud is None:
        return
    import pandas as _pd
    aud.add_stage("S9_render", "figures",
                  _pd.DataFrame([{"figures": len(list((Path(out_dir) / "figures").glob("*.png")))
                                  if (Path(out_dir) / "figures").is_dir() else 0,
                                  "render_warnings": RENDER_WARNINGS}]))
    aud.check_stage_coverage()
    aud.trace["stage_list_version"] = aud.STAGE_LIST_VERSION
    aud.trace["render_warnings"] = RENDER_WARNINGS
    aud.trace["bundles"] = [Path(b.path).name for b in loaded]   # privacy: relative
    aud.trace["mode"] = mode
    aud.write(out_dir)


def generate(bundles, run_records=(), labels=None, sections=None,
             out_dir=".", mode="technical", audit=True):
    """Render diagnostic bundles. Returns the path of the primary artifact
    (report.html for technical mode, report_it.md for it_report mode)."""
    # UID-delta round 2 [panel P1-2]: Layer-C state is LOCAL per call - the
    # module-global stash made a second no-record call in the same process
    # ship the FIRST call's runs/conclusion in report_summary.json, and kept
    # only the last host in multi-host reports. Reproduced, now structural.
    layerc_analyses = []                 # (host, record) pairs - attributable
    layerc_conclusions = {}              # host -> bundle_conclusion
    global RENDER_WARNINGS
    RENDER_WARNINGS = 0                  # per-call, never inherited
    if mode not in ("technical", "it_report"):
        raise ValueError(f"mode must be 'technical' or 'it_report', got {mode!r}")
    sections = list(sections) if sections else list(DEFAULT_SECTIONS)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    loaded = [b if isinstance(b, schema.Bundle) else schema.load_bundle(b) for b in bundles]
    if not loaded:
        raise ValueError("no bundles given")

    def _prog(msg):
        print(f"[report] {msg}", file=sys.stderr, flush=True)
    aud = audit_mod.Audit() if audit else None
    summaries, figures, html_parts = {}, {}, []
    for b in loaded:
        _prog(f"bundle {b.host}: environment")
        env = {"verdict": (b.verdict + " (collector still sampling - verdict is "
                            "written at completion; re-render when finished)"
                            if b.in_progress else b.verdict),
               "rules": " ".join(b.rules_fired) or "none",
               **schema.config_summary(b),
               "missing_required": ",".join(schema.required_missing(b)) or "none"}
        html_parts.append(f"<h2>{b.host} - environment &amp; validity</h2>" + _table(env))
        if aud is not None:
            import pandas as _pd
            aud.add_stage("S6_severity", f"{b.host}:rules",
                          _pd.DataFrame([{"rule": r} for r in b.rules_fired]
                                        or [{"rule": "none"}]))
        for s in schema.select_impact(b.rules_fired, mode="technical"):
            html_parts.append(f"<p><b>{s.split(']')[0]}]</b>{s.split(']',1)[1]}</p>")
        # platform-limitation honesty: name the rules that CANNOT fire here
        st = schema.evidence_states(b)
        limits = []
        if st.get("kthreads") != "available":
            limits.append("KC-01 (kernel threads not visible on this host/container)")
        if st.get("psi") != "available":
            limits.append("PSI-01 (kernel exposes no pressure-stall accounting)")
        if limits:
            html_parts.append("<p><b>Platform limitation:</b> the following rules can "
                              "structurally never fire on this host - a PASS here does "
                              "NOT clear them: " + "; ".join(limits) + ". Collect on the "
                              "production node for these signals.</p>")
        if b.samples_path is not None:
            frame = schema.samples_frame(b)
            if aud is not None:
                import pandas as _pd
                aud.add_stage("S0_manifest", f"{b.host}:manifest",
                              _pd.DataFrame([{"k": k, "v": str(v)}
                                             for k, v in sorted(b.manifest.items())]))
                aud.add_stage("S1_load_tables", f"{b.host}:host_table_shape",
                              _pd.DataFrame([{"rows": len(frame),
                                              "cols": len(frame.columns),
                                              "legacy": bool(getattr(b, "legacy_host_table", False))}]))
                aud.add_stage("S2_parse_derive", f"{b.host}:samples", frame)
                aud.check_conservation(frame)
                aud.check_counts(frame)
                audit_mod.check_rule_evidence(aud, b, schema)   # CRR-7
                pdf_, udf_, _wdf_a = _aux_tables(b)
                if _wdf_a is not None and len(_wdf_a):
                    aud.add_stage("S4_pivots", f"{b.host}:rollup", _wdf_a)
                if pdf_ is not None:
                    aud.add_stage("S2_parse_derive", f"{b.host}:process_samples", pdf_)
                aud.check_hierarchy(frame, pdf_, udf_)
            adf, reg = _build_adf(frame)
            summaries[b.host] = _summarize(adf.df, DEFAULT_CHANNELS + reg)
            if aud is not None:
                import pandas as _pd
                aud.add_stage("S5_summary", f"{b.host}:summary",
                              _pd.DataFrame(summaries[b.host]))
                aud.check_reproducibility(summaries[b.host], adf.df,
                                          DEFAULT_CHANNELS + reg)
            # ---- v8 page-1: host vitals - ALWAYS drawn when present ----
            _prog(f"bundle {b.host}: vitals")
            if "vitals" in sections:
                if "mem_available_kb" in adf.df.columns:
                    adf.add_alias("mem_available_gb", "mem_available_kb/1048576.0")
                    adf.materialize_aliases(names=["mem_available_gb"])
                vfigs = []
                for expr, title in VITAL_PANELS:
                    cols = expr.strip("[]").split(",")
                    if all(c in adf.df.columns and adf.df[c].notna().any() for c in cols):
                        vfigs += _draw_expr(adf, expr, f"{b.host}: {title}", title,
                                            out_dir / "figures",
                                            f"{b.host}_vital_{_san(expr)}.png")
                if vfigs:
                    html_parts.append(f"<h2>{b.host} - host vitals</h2>" +
                                      "".join(f"<img src='{_img_datauri(out_dir / 'figures' / f)}' "
                                              f"alt='{f}'>" for f in vfigs))
                else:
                    html_parts.append(f"<h2>{b.host} - host vitals</h2><p>no vital "
                                      "channels in this bundle (pre-anchor collector) - "
                                      "re-collect with the current script</p>")
            # ---- v8: processes & background (the job-vs-environment picture) ----
            _prog(f"bundle {b.host}: processes & background")
            pdf_x, udf_x, wdf_x = _aux_tables(b)
            if "processes" in sections and udf_x is not None:
                import pandas as pd
                pfigs = []
                wide_u, _names = _pivot_users(udf_x)
                from AliasDataFrame import AliasDataFrame as _ADF
                adf_u = _ADF(wide_u)
                ccols = [c for c in wide_u.columns if c.startswith("cpu_")]
                rcols = [c for c in wide_u.columns if c.startswith("rss_gb_")]
                if ccols:
                    pfigs += _draw_expr(adf_u, "[" + ",".join(ccols) + "]",
                                        f"{b.host}: CPU by user [cores]", "cores",
                                        out_dir / "figures", f"{b.host}_users_cpu.png")
                if rcols:
                    pfigs += _draw_expr(adf_u, "[" + ",".join(rcols) + "]",
                                        f"{b.host}: RSS by user [GB]", "GB",
                                        out_dir / "figures", f"{b.host}_users_rss.png")
                if wdf_x is not None:
                    wide_w = _pivot_rollup(wdf_x)
                    adf_w = _ADF(wide_w)
                    wcols = [c for c in wide_w.columns if c.startswith("cpu_")]
                    if wcols:
                        pfigs += _draw_expr(adf_w, "[" + ",".join(wcols) + "]",
                                            f"{b.host}: background vs job [CPU cores]",
                                            "cores", out_dir / "figures",
                                            f"{b.host}_background_vs_job.png")
                html_parts.append(f"<h2>{b.host} - processes &amp; background</h2>" +
                                  "".join(f"<img src='{_img_datauri(out_dir / 'figures' / f)}' "
                                          f"alt='{f}'>" for f in pfigs))
                if pdf_x is not None:
                    rows = _top_consumers(pdf_x)
                    tbl = ("<table><tr><th>tenant</th><th>process</th><th>cpu max %</th>"
                           "<th>cpu mean %</th><th>rss max MB</th><th>samples</th></tr>" +
                           "".join("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td>"
                                   "<td>{:.0f}</td><td>{}</td></tr>".format(
                                       r["tenant"], r["proc"],
                                       ("-" if r["cpu_max"] != r["cpu_max"]
                                        else f"{r['cpu_max']:.1f}"),
                                       ("-" if r["cpu_mean"] != r["cpu_mean"]
                                        else f"{r['cpu_mean']:.1f}"),
                                       r["rss_max_kb"] / 1024.0, r["samples"])
                                   for r in rows) + "</table>")
                    html_parts.append(f"<h3>{b.host} - top consumers over the window</h3>" + tbl)
            _prog(f"bundle {b.host}: pathology channels")
            if "rates" in sections or "psi" in sections:
                active, flat, nodata = _channels_to_draw(adf.df, DEFAULT_CHANNELS + reg)
                figures[b.host] = _draw_series(adf, active, out_dir / "figures", b.host)
                if flat:
                    html_parts.append(f"<p><b>{b.host}:</b> flat ZERO over the whole "
                                      f"window (not plotted): {', '.join(flat)}</p>")
                if nodata:
                    html_parts.append(f"<p><b>{b.host}:</b> no data on this platform: "
                                      f"{', '.join(nodata)}</p>")
        else:
            summaries[b.host] = {}

    _prog("writing audit + report")
    if RENDER_WARNINGS:
        html_parts.append(f"<p class='note'>rendering produced {RENDER_WARNINGS} "
                          "drawing-library warnings (captured, not shown; "
                          "constant-series statistics inside the drawing backend - "
                          "filed against dfdraw).</p>")
    if mode == "it_report":
        _finalize_audit(aud, out_dir, loaded, mode)
        return _it_report(loaded, summaries, out_dir)

    for host, summ in summaries.items():
        if summ:
            html_parts.append(f"<h2>{host} - channel summary</h2>" + _table(
                {c: (f"n={v.get('count',0)} mean={v.get('mean',float('nan')):.3g} "
                     f"max={v.get('max',float('nan')):.3g} last={v.get('last',float('nan')):.3g}"
                     if v.get("count") else "no data") for c, v in summ.items()}))
    for host, figs in figures.items():
        if not figs:
            html_parts.append(f"<h2>{host} - time series</h2><p>all channels flat "
                              "zero or without data over this window - nothing to plot "
                              "- no activity observed in these channels during the "
                              "window. This is a VALID observation only because the "
                              "collector is certified live via the anchor channels; "
                              "flat pathology channels never by themselves prove a "
                              "healthy host (see validation/summary.md).</p>")
            continue
        html_parts.append(f"<h2>{host} - time series</h2>" +
                          "".join(f"<img src='{_img_datauri(out_dir / 'figures' / f)}' "
                                  f"alt='{f}'>" for f in figs))
    if "crosshost" in sections and len(loaded) > 1:
        diff = schema.config_diff(loaded)
        html_parts.append("<h2>Cross-host configuration differences</h2>" +
                          (_table({k: json.dumps(v) for k, v in diff.items()})
                           if diff else "<p>no configuration differences detected</p>"))
    if "runs" in sections:
        if not run_records:
            html_parts.append("<h2>Layer-C: job vs background</h2>"
                              "<p>no run_metrics records supplied (D2) - "
                              "background influence cannot be assessed.</p>")
        else:
            # CRR-4: records are LOADED, ALIGNED and CORRELATED - not counted
            lc_all = []
            for b in loaded:
                if b.samples_path is None:
                    continue
                hostf = schema.samples_frame(b)
                _p, _u, roll = _aux_tables(b)
                res = jha_mod.analyze(run_records, hostf, roll)
                concl = cm_mod.evaluate(b.verdict, b.rules_fired, res)
                layerc_analyses.extend(          # host-tagged [GPT25 P1-4]:
                    (b.host, r) for r in res)    # multi-host rows stay attributable
                layerc_conclusions[b.host] = concl.get("bundle_conclusion")
                lc_all.append((b.host, res, concl))
                if aud is not None:
                    import pandas as _pd
                    aud.trace.setdefault("job_host_analysis", {})[b.host] = res
                    aud.trace.setdefault("conclusion_model", {})[b.host] = concl
                    aud.add_stage("S7_job_host_analysis", f"{b.host}:jha",
                                  _pd.DataFrame([{"run": r.get("run_id"),
                                                  "window": (r.get("window") or {}).get("state"),
                                                  "n_corr": len(r.get("correlations", []))}
                                                 for r in res]))
                    aud.add_stage("S8_conclusion", f"{b.host}:conclusion",
                                  _pd.DataFrame([{"run": r["run_id"], "code": r["code"]}
                                                 for r in concl["records"]]
                                                or [{"run": "none",
                                                     "code": concl["bundle_conclusion"]["code"]}]))
            for host, res, concl in lc_all:
                html_parts.append(_render_layer_c(host, res, concl))
    if "evidence" in sections:
        for b in loaded:
            html_parts.append(f"<h2>{b.host} - evidence states</h2>" +
                              _table(schema.evidence_states(b)))

    # summary JSON (canonical values; html is a rendering of the same numbers)
    (out_dir / "report_summary.json").write_text(json.dumps(
        {"schema_version": schema.SCHEMA_VERSION,
         "rule_table_version": schema.RULE_TABLE_VERSION,
         # round-4 machine-legibility fields [GPT21]: run identity+roles,
         # window counts, baseline validity, conclusion inputs/code
         "runs": [{"host": h, "run_id": a.get("run_id"), "label": a.get("label"),
                   "record_role": a.get("record_role"),
                   "outcome": a.get("outcome"),
                   "baseline_state": a.get("baseline_state"),
                   "window": a.get("window"),
                   "n_correlations": len(a.get("correlations", []))}
                  for h, a in layerc_analyses],
         "conclusion": (next(iter(layerc_conclusions.values()))
                        if len(layerc_conclusions) == 1 else None),
         "conclusions_by_host": layerc_conclusions,
         "hosts": {b.host: {"verdict": b.verdict, "rules": b.rules_fired,
                            "summary": summaries.get(b.host, {})} for b in loaded}},
        indent=1, sort_keys=True))

    p = out_dir / "report.html"
    p.write_text(_html("dfx host diagnostics report", html_parts))
    _finalize_audit(aud, out_dir, loaded, mode)
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Render diagnostic bundle(s) into a report")
    ap.add_argument("bundles", nargs="+", help="bundle directories (D1 output)")
    ap.add_argument("-o", "--out-dir", default="diag_report")
    ap.add_argument("--mode", choices=["technical", "it_report"], default="technical")
    ap.add_argument("--no-audit", action="store_true",
                    help="skip validation/ audit (recorded; audit-less reports "
                         "cannot serve as CRR/official evidence)")
    a = ap.parse_args()
    print(generate(a.bundles, out_dir=a.out_dir, mode=a.mode, audit=not a.no_audit))
