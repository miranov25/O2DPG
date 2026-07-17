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
DEFAULT_SECTIONS = ["environment", "rates", "psi", "runs", "crosshost", "evidence"]

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


# --------------------------------------------------------------------------
def _aux_tables(bundle):
    """process/user tables if the bundle has them (v8 D1.6); else (None, None)."""
    import pandas as pd
    pp = bundle.path / "process_samples.csv"
    up = bundle.path / "user_samples.csv"
    pdf_ = udf_ = None
    try:
        if pp.is_file() and len(pp.read_text().splitlines()) > 1:
            pdf_ = pd.read_csv(pp)
        if up.is_file() and len(up.read_text().splitlines()) > 1:
            udf_ = pd.read_csv(up)
    except Exception:
        pass
    return pdf_, udf_


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
def generate(bundles, run_records=(), labels=None, sections=None,
             out_dir=".", mode="technical", audit=True):
    """Render diagnostic bundles. Returns the path of the primary artifact
    (report.html for technical mode, report_it.md for it_report mode)."""
    if mode not in ("technical", "it_report"):
        raise ValueError(f"mode must be 'technical' or 'it_report', got {mode!r}")
    sections = list(sections) if sections else list(DEFAULT_SECTIONS)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    loaded = [b if isinstance(b, schema.Bundle) else schema.load_bundle(b) for b in bundles]
    if not loaded:
        raise ValueError("no bundles given")

    aud = audit_mod.Audit() if audit else None
    summaries, figures, html_parts = {}, {}, []
    for b in loaded:
        env = {"verdict": (b.verdict + " (collector still sampling - verdict is "
                            "written at completion; re-render when finished)"
                            if b.in_progress else b.verdict),
               "rules": " ".join(b.rules_fired) or "none",
               **schema.config_summary(b),
               "missing_required": ",".join(schema.required_missing(b)) or "none"}
        html_parts.append(f"<h2>{b.host} - environment &amp; validity</h2>" + _table(env))
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
                aud.add_stage("S2_parse_derive", f"{b.host}:samples", frame)
                aud.check_conservation(frame)
                aud.check_counts(frame)
                pdf_, udf_ = _aux_tables(b)
                if pdf_ is not None:
                    aud.add_stage("S2_parse_derive", f"{b.host}:process_samples", pdf_)
                aud.check_hierarchy(frame, pdf_, udf_)
            adf, reg = _build_adf(frame)
            summaries[b.host] = _summarize(adf.df, DEFAULT_CHANNELS + reg)
            if aud is not None:
                aud.check_reproducibility(summaries[b.host], adf.df,
                                          DEFAULT_CHANNELS + reg)
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

    # summary JSON (canonical values; html is a rendering of the same numbers)
    (out_dir / "report_summary.json").write_text(json.dumps(
        {"schema_version": schema.SCHEMA_VERSION,
         "rule_table_version": schema.RULE_TABLE_VERSION,
         "hosts": {b.host: {"verdict": b.verdict, "rules": b.rules_fired,
                            "summary": summaries.get(b.host, {})} for b in loaded}},
        indent=1, sort_keys=True))

    if aud is not None:
        aud.trace["bundles"] = [str(b.path) for b in loaded]
        aud.trace["mode"] = mode
        aud.write(out_dir)
    if mode == "it_report":
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
                              "(a good sign on a healthy host; see summary table)</p>")
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
        html_parts.append("<h2>Run panel</h2><p>" +
                          (f"{len(run_records)} run record(s) supplied." if run_records
                           else "no run_metrics records supplied (D2).") + "</p>")
    if "evidence" in sections:
        for b in loaded:
            html_parts.append(f"<h2>{b.host} - evidence states</h2>" +
                              _table(schema.evidence_states(b)))

    p = out_dir / "report.html"
    p.write_text(_html("dfx host diagnostics report", html_parts))
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
