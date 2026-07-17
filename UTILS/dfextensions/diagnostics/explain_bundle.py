#!/usr/bin/env python3
"""
explain_bundle.py - PHASE_13_74_ADF: make a diagnostic bundle HUMAN-DEBUGGABLE.

Walks one bundle step by step and prints, for every column and key value:
WHAT it is, WHAT WAS MEASURED, WHAT SHOULD BE THERE on a healthy vs an
affected host, and a per-column FLAG. Ends with a plain-language
self-consistency verdict based on ANCHOR channels (loadavg/CPU/memory -
signals that are never flat on a live host): anchors moving + pathology flat
= healthy host AND working collector, proven; anchors flat = do not trust
the bundle.

Pure pandas + stdlib. No ADF needed. This is the reviewer's entry point.

Usage:  python3 explain_bundle.py <bundle_dir> [...]
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402

# column -> (meaning, healthy expectation, affected signature)
DOC = {
    "loadavg1":            ("1-min load average [runnable procs]",
                            "ANCHOR: >0 and varying on any live host",
                            "may spike; if FLAT ZERO the collector/platform is broken"),
    "cpu_busy_pct":        ("CPU busy % (all cores, incl. iowait as idle)",
                            "ANCHOR: 1-100, varying; never constant for hours",
                            "high + stalls -> compaction burns CPU"),
    "mem_available_kb":    ("kernel estimate of allocatable memory",
                            "ANCHOR: large, wiggles with activity",
                            "trending to ~0 -> memory pressure incident"),
    "ctxt_per_s":          ("context switches per second",
                            "ANCHOR: hundreds..hundreds of thousands, never 0",
                            "collapse can indicate a frozen host"),
    "procs_running":       ("currently runnable processes",
                            "ANCHOR: >=1 (this collector itself runs)",
                            "large + low CPU -> run-queue congestion"),
    "compact_stall_per_s": ("processes stalled waiting for memory compaction",
                            "EXPECTED 0 on a healthy host",
                            "THE SMOKING GUN: >0 sustained during slowdowns"),
    "compact_fail_per_s":  ("failed compaction attempts",
                            "EXPECTED 0", ">0 with stalls -> fragmentation"),
    "thp_fault_alloc_per_s": ("THP allocations on page fault",
                            "0..small on madvise; larger on enabled=[always]",
                            "high + fallbacks -> THP churn"),
    "thp_fault_fallback_per_s": ("THP wanted but unavailable -> 4k fallback",
                            "EXPECTED ~0", "sustained >0 -> fragmentation pressure"),
    "allocstall_per_s":    ("direct-reclaim allocation stalls",
                            "EXPECTED 0", ">0 -> allocations block on reclaim"),
    "kcompactd_cpu_per_s": ("kernel compaction thread CPU rate",
                            "EXPECTED ~0 (invisible in containers - see states)",
                            ">0.05 sustained -> active compaction pathology"),
    "khugepaged_cpu_per_s": ("khugepaged CPU rate", "EXPECTED ~0",
                            ">0 sustained -> collapse scanning churn"),
    "psi_mem_some_avg10":  ("PSI: % time SOME tasks stalled on memory",
                            "0 (absent on old/container kernels - see states)",
                            ">1 noteworthy; >10 severe"),
    "psi_mem_full_avg10":  ("PSI: % time ALL tasks stalled on memory",
                            "0", ">=1 live pathology; >=5 fires PSI-01"),
    "pgscan_direct_per_s": ("pages scanned by DIRECT reclaim (allocating task blocked)",
                            "EXPECTED ~0; ANY sustained value is noteworthy",
                            "hundreds+/s -> allocation paths doing reclaim work"),
    "disk_read_sectors_per_s": ("disk sectors read/s (whole host)",
                            "varies with I/O; just context",
                            "high + memory pressure -> thrash/readback"),
    "pswpin_per_s":        ("swap-ins/s", "0 on no-swap or healthy hosts",
                            ">0 sustained -> swap thrash"),
    "pswpout_per_s":       ("swap-outs/s", "0", ">0 sustained -> memory shortage"),
    "sample_wall_s":       ("collector's own cost per sample",
                            "< 0.5 s (C-9 draft budget)", "overruns -> reduce tier"),
}
ANCHORS = ("loadavg1", "cpu_busy_pct", "mem_available_kb", "ctxt_per_s", "procs_running")
PATHOLOGY = ("compact_stall_per_s", "compact_fail_per_s", "thp_fault_fallback_per_s",
             "allocstall_per_s", "kcompactd_cpu_per_s", "psi_mem_full_avg10",
             "pgscan_direct_per_s", "pswpin_per_s", "pswpout_per_s")


def flag_column(name, series):
    s = series.dropna()
    if len(s) == 0:
        return "NO-DATA", None
    varying = s.nunique() > 1
    nonzero = (s != 0).any()
    if name in ANCHORS:
        if not nonzero:
            return "SUSPECT-ANCHOR-FLAT-ZERO", s
        return ("ANCHOR-OK" if varying or name == "procs_running"
                else "ANCHOR-CONSTANT(check)"), s
    if name in PATHOLOGY:
        return ("SIGNAL(!)" if nonzero else "EXPECTED-ZERO"), s
    return ("data" if nonzero else "zero"), s


def explain(bundle_dir, out=print):
    b = schema.load_bundle(bundle_dir)
    st = schema.evidence_states(b)
    out(f"================ BUNDLE {b.path.name} ================")
    out(f"[1] PROVENANCE  host={b.host}  collected_by={b.manifest.get('user')}  "
        f"utc={b.manifest.get('utc')}  tool_md5={b.manifest.get('tool_md5','?')[:8]}...")
    out(f"    verdict={b.verdict}"
        + ("  <- collector still running; verdict comes at completion" if b.in_progress else
           f"  rules_fired={' '.join(b.rules_fired) or 'none'}"))
    out(f"[2] PLATFORM  kernel={b.snapshot.get('sys.kernel','?')}  virt={b.snapshot.get('sys.virt','?')}")
    limits = []
    if st.get("kthreads") != "available":
        limits.append("kernel threads INVISIBLE -> kcompactd channels can never show data here")
    if st.get("psi") != "available":
        limits.append("PSI NOT EXPOSED -> psi_* channels can never show data here")
    for l in limits:
        out(f"    LIMITATION: {l}")
    out(f"[3] CONFIG  thp.enabled={b.snapshot.get('thp.enabled','?')}  "
        f"thp.defrag={b.snapshot.get('thp.defrag','?')}")
    out("[4] EVIDENCE STATES  " + "  ".join(f"{k}={v}" for k, v in sorted(st.items())))

    if b.samples_path is None:
        out("[5] no samples.csv (snapshot-only run) - rerun with -s/-n for time series")
        return None
    df = schema.samples_frame(b)
    valid = df[df["row_valid"]]
    out(f"[5] SAMPLES  rows={len(df)} (valid={len(valid)})  "
        f"window={float(df['t_rel'].iloc[-1]):.0f}s")
    out(f"    {'column':26s} {'n':>3s} {'min':>10s} {'mean':>10s} {'max':>10s}  FLAG   meaning | healthy | affected")
    anchors_ok, anchors_bad, signals = [], [], []
    for name in DOC:
        if name not in df.columns:
            continue
        fl, s = flag_column(name, valid[name] if name in valid else df[name])
        meaning, healthy, affected = DOC[name]
        if s is None:
            out(f"    {name:26s} {0:>3d} {'-':>10s} {'-':>10s} {'-':>10s}  {fl:22s} {meaning}")
            continue
        out(f"    {name:26s} {len(s):>3d} {s.min():>10.3g} {s.mean():>10.3g} {s.max():>10.3g}  "
            f"{fl:22s} {meaning} | {healthy} | {affected}")
        if fl == "ANCHOR-OK":
            anchors_ok.append(name)
        elif fl.startswith("SUSPECT"):
            anchors_bad.append(name)
        elif fl == "SIGNAL(!)":
            signals.append(name)

    out("[6] SELF-CONSISTENCY VERDICT (plain language):")
    if anchors_bad and not anchors_ok:
        out("    !! ALL anchors flat -> DO NOT TRUST this bundle: collector or platform broken.")
        verdict = "UNTRUSTWORTHY"
    elif anchors_bad:
        out(f"    ?? some anchors flat ({', '.join(anchors_bad)}) - investigate those channels first.")
        verdict = "PARTIAL"
    elif anchors_ok:
        out(f"    OK anchors alive ({', '.join(anchors_ok)}): the collector is measuring a live host.")
        verdict = "CONSISTENT"
    else:
        out("    (old-format bundle: no anchor columns - re-collect with the current script)")
        verdict = "NO-ANCHORS"
    if signals:
        out(f"    PATHOLOGY SIGNALS PRESENT: {', '.join(signals)} - inspect these plots first.")
    else:
        out("    no pathology signals in this window - zeros in pathology channels are the "
            "EXPECTED healthy value (see column table).")
    return verdict


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    for d in sys.argv[1:]:
        explain(d)
