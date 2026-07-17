# dfextensions diagnostics (PHASE_13_74_ADF)

Host-health collection, run instrumentation and summary analytics for the
dfextensions ecosystem (AliasDataFrame, dfdraw, GroupByRegression consumers).
Motivated by the 2026-07-14 THP/kcompactd incidents: some hosts run identical
workloads 5-10x slower, and an affected host silently corrupts every
performance/memory measurement taken on it (incl. PlanScope stage-0 M-0).

**This tool set diagnoses and reports. It never remediates - THP/allocator
changes are an admin decision.**

## Components

| File | Role |
|---|---|
| `dfx_host_diagnostics.sh` | read-only host collector -> diagnostic bundle |
| `run_metrics.py` | process-level run instrumentation (context manager) |
| `schema.py` | bundle parsing, kv->JSON, evidence classes, verdict-rule table |
| `report_diagnostics.py` | bundles -> AliasDataFrame -> dfdraw figures + summary + IT report |
| `tests/` | hermetic fixture suites (bash + pytest); ADF tier gates on alma2 |

## Reviewer recipe (execute, do not just read)

1. Verify `provenance/MANIFEST.md5` in the packet against `code/` and `tests/`.
2. Run the gate yourself: `bash diagnostics/run_tests.sh` - expect `SUMMARY: diagnostics OK`.
3. Collect on a live host: `bash diagnostics/dfx_host_diagnostics.sh -o /tmp/rv -s 5 -n 6`.
4. Render + audit: `python3 diagnostics/report_diagnostics.py /tmp/rv/host_diag_* -o /tmp/rv/rep`;
   read `/tmp/rv/rep/validation/summary.md` FIRST (trust check).
5. Inspect the report: vitals ALWAYS drawn; processes-and-background section present;
   rank fields EMPTY (never 0) when a process is outside a top list.
6. Verdict per the Reviewer QRC: findings P0/P1/P2 with file:line evidence.

## How to run collection (runbook)

**A. Health snapshot** (5 s): `bash diagnostics/dfx_host_diagnostics.sh -o <data_dir>`
**B. Background watch**: `nohup bash ... -o <data_dir> -s 60 -n 120 > <data_dir>/run.log 2>&1 &`
(process/user/rollup tables sample automatically at 5 s; stop cleanly with
`kill -TERM <pid>`; NEVER move/delete the bundle dir mid-run - loud FATAL.)
**C. Around your job**: add `-R <run_id> -F <pidfile>` and run the job with
`DFX_RUN_ID=<run_id>` so run_metrics records join the bundle; the job's tree
is marked `is_target_job=1` in process_samples.csv.
**D. Production node, no checkout**: copy exactly two files -
`dfx_host_diagnostics.sh` + `collector.py` - run as in A/B, scp the bundle
back (shareable redaction is the default).
**Forgot a running watch?** `pgrep -af dfx_host_diagnostics` lists running
collectors; `grep -L '^verdict=' <data>/host_diag_*/manifest.kv` lists bundles
still in progress. Watches are ALWAYS bounded (`-n` required) and stop
themselves; heartbeat lines land in run.log (`tail -f`).
**E. Analyze**: `report_diagnostics.py <bundles...> -o <out>` then read
`<out>/validation/summary.md` FIRST (trust check), `explain_bundle.py <bundle>`
for the column-by-column reading, `--mode it_report` for the admin one-pager.

Flags: `-s` interval s / `-n` samples / `-o` outdir / `-R` run_id /
`-F` target-pid-file / `-G` job-cgroup / `-r` raw (local only) /
`-A` foreign-PID ack / `-p` PID deep-dive / `-d` datapath probe.
Env: `PROC_INTERVAL_OVERRIDE`, `NALL`, `NUSER`, `DFX_PROCESS_SAMPLER=off`.

## Quick start

```bash
DIAG=/Users/miranov25/tmp/dfx_diag        # persistent tree - NEVER node-local /tmp
DFX=/Users/miranov25/alicesw/O2DPG/UTILS/dfextensions
mkdir -p "$DIAG/data" "$DIAG/reports"

# one snapshot
bash "$DFX/diagnostics/dfx_host_diagnostics.sh" -o "$DIAG/data"

# 2h sampling watch (rates + PSI), e.g. before/while running risky work
nohup bash "$DFX/diagnostics/dfx_host_diagnostics.sh" -o "$DIAG/data" -s 60 -n 120 \
      > "$DIAG/data/run.log" 2>&1 &

# analyze (any time - the CSV is append-only)
python3 "$DFX/diagnostics/report_diagnostics.py" "$DIAG"/data/host_diag_* -o "$DIAG/reports/tech"
python3 "$DFX/diagnostics/report_diagnostics.py" "$DIAG"/data/host_diag_* \
        -o "$DIAG/reports/it" --mode it_report
```

Cross-host ("why some servers, not all"): collect a bundle on an affected AND
a healthy node, pass both paths in ONE report command - the report includes a
configuration-difference table and per-host overlays.

## The bundle

`host_diag_<host>_<utc>_<pid>/` containing `manifest.kv` (provenance, verdict,
self-overhead), `snapshot.kv` (all point-in-time fields + per-field evidence
states), `samples.csv` (sampling mode: `*_total` cumulative AND `*_per_s` rate
columns, PSI, diskstats, per-sample wall + overrun flag; first row has empty
rates by design - empty != 0), `report.txt` (human rendering of the same
values - never the data source). Canonical JSON (`manifest.json`,
`snapshot.json`) is produced by `schema.bundle_to_json()` - the bash collector
emits flat key=value/CSV only.

## Exit contract (collector)

| code | meaning |
|---|---|
| 0 | bundle produced; verdict PASS / WARN / UNHEALTHY recorded inside |
| 1 | invocation or internal error (all CLI validation failures; vanished bundle dir) |
| 2 | verdict UNKNOWN: REQUIRED evidence unreadable / unsupported platform (non-Linux) |

## Evidence classes and UNKNOWN semantics

Every field carries a state (`available/unavailable/permission_denied/
not_supported/parse_error`). REQUIRED floor: `/proc/vmstat`, THP sysfs,
`/proc/meminfo` - any of these missing forces verdict UNKNOWN (never
"healthy"). Everything else is OPTIONAL: absence is recorded per-field and
does NOT force UNKNOWN (a PSI-less kernel degrades gracefully).

## Verdict rules (stable IDs; table versioned in `schema.RULE_TABLE`)

| ID | kind | condition (v1-draft thresholds, pending C-9 ratification) |
|---|---|---|
| THP-01 | config | `transparent_hugepage/enabled = [always]` |
| THP-02 | config | defrag in `{[always], [defer+madvise]}` |
| KC-01 | historical | kcompactd CPU >= 300 s per uptime-day (uptime-normalized) |
| PSI-01 | live | PSI memory `full avg60 >= 5.0` |

Severity: config -> WARN; historical/live -> UNHEALTHY. Reports cite rule IDs;
impact sentences come from the fixed reviewed library in `schema.RULE_TABLE`
(never free-generated).

## Privacy (shared nodes)

Default bundles are **shareable**: other users' names are hashed, commands
masked; `-r` (raw) keeps everything, local use only, recorded in the manifest.
Deep-dive on a process you do not own requires the explicit `-A` flag.
`--mode it_report` REFUSES raw bundles, separates host-level pathology
(system-wide kernel accounting, affects all users) from our-job impact, and
suppresses per-user aggregates below a k-anonymity floor of 5 users.

## run_metrics (instrumenting a workload)

```python
from run_metrics import RunMetrics
with RunMetrics("fitDCAITS_run42", adf=adf, out=f"{DIAG}/runs") as rm:
    rm.record_event("load", {"branches": 12})
    ...workload...
```
Notes: `io.rchar` counts logical reads (page cache included), `io.read_bytes`
physical disk reads - the difference IS the cache-hit picture.
`rss_kb.peak_context_sampled` is the context-local peak (sampled);
`rss_kb.ru_maxrss_lifetime` is the process-lifetime peak - deliberately
distinct names. The record is written on exceptions too, then re-raised.

## Testing

```bash
bash diagnostics/tests/test_dfx_host_diagnostics.sh      # collector, hermetic fixtures
python3 -m pytest -q diagnostics/tests/                  # schema/report/run_metrics
```
The ADF tier (add_alias -> materialize -> adf.draw path) auto-skips where the
locked stack (pandas 1.5.3 + AliasDataFrame + dfdraw) is absent and is
BLOCKING on alma2. Fixture roots: `PROC_ROOT`, `SYS_ROOT`, `CGROUP_ROOT`,
`CLK_TCK_OVERRIDE`, `PS_CMD_OVERRIDE`.


# Reviewer recipe (CRR packet entry point)

You are reviewing a host-pathology measurement tool. You do NOT need to read
the CSV raw — the tool explains its own data. Review = run six commands and
judge the outputs against the expectations below.

## The 6-step review (copy-paste; needs only python3+pandas and bash)

```bash
cd <packet>/code
# 1. hermetic test suites (fixtures - no host dependence)
bash ../tests/test_dfx_host_diagnostics.sh | tail -1        # expect PASS=54 FAIL=0
python3 -m pytest -q ../tests/                              # expect all pass (ADF tier may skip off-alma2)

# 2. collect 3 samples on YOUR machine (read-only, writes one bundle dir)
bash dfx_host_diagnostics.sh -o /tmp/rev -s 1 -n 3

# 3. THE ENTRY POINT - the tool explains the bundle column by column:
python3 explain_bundle.py /tmp/rev/host_diag_*
#    read [6] SELF-CONSISTENCY VERDICT: anchors must be alive on your machine.

# 4. oracle check (trust nothing): recompute one rate yourself
python3 - <<'EOF'
import sys; sys.path.insert(0,'.'); import schema, glob
b = schema.load_bundle(glob.glob('/tmp/rev/host_diag_*')[0]); df = schema.samples_frame(b)
print("max |csv - oracle| =", ((df.compact_stall_total.diff()/df.ts.diff())
      .iloc[1:] - df.compact_stall_per_s.iloc[1:]).abs().max())   # expect 0 or NaN-free tiny
EOF

# 5. break it on purpose - the tool must FAIL LOUDLY, never lie:
PROC_ROOT=/nonexistent bash dfx_host_diagnostics.sh -o /tmp/rev; echo "exit=$? (expect 2 UNKNOWN)"

# 6. render and eyeball
python3 report_diagnostics.py /tmp/rev/host_diag_* -o /tmp/rev_report   # open report.html
```

## What SHOULD be there (the expectation table)

| Channel class | Healthy host | Affected host (the 2026-07-14 incidents) | Broken collector |
|---|---|---|---|
| **ANCHORS**: loadavg1, cpu_busy_pct, mem_available_kb, ctxt_per_s, procs_running | alive and moving — CPU 1–100%, ctxt hundreds+, load >0 | alive (possibly extreme) | **flat zero — the tell** |
| compaction/THP rates (compact_stall/s, thp_fault_fallback/s, allocstall/s) | **all 0 — zero is the healthy value** | >0 sustained; compact_stall/s is the smoking gun | 0 (indistinguishable without anchors — that is WHY anchors exist) |
| kcompactd/khugepaged CPU | ~0; INVISIBLE in containers (evidence state says so) | >0.05/s sustained | 0 with state=available and anchors dead |
| PSI mem full avg10/60 | 0; ABSENT on old/container kernels (state says so) | ≥1 live pathology, ≥5 fires PSI-01 | — |
| pgscan_direct/s, swap in/out per s | ~0 | hundreds+/s = reclaim pressure | — |
| sample_wall_s | < 0.5 s | < 0.5 s | overruns |

**Decision logic (also printed by explain_bundle.py):** anchors moving +
pathology zero ⇒ healthy host, working collector — proven, not assumed.
Anchors moving + pathology nonzero ⇒ incident data — the whole point.
Anchors flat ⇒ do not trust anything else in the bundle.

## Where to hunt for bugs (CRR §6, condensed)

1. Rerun step-4's oracle for EVERY *_per_s column, not just one.
2. Feed hostile fixtures via PROC_ROOT/SYS_ROOT (weird THP strings, huge/tiny
   uptime, missing files mid-run, prefix-colliding vmstat keys).
3. Hunt redaction leaks: any other-user string reaching snapshot/report/IT output.
4. awk portability (mawk/busybox) — printf-argument bugs bit this tool twice.
5. run_metrics under forks/threads; exception path must write-then-reraise.
6. explain_bundle's flags: construct a bundle that fools the self-consistency verdict.

## History you should know (so you don't re-litigate)

Rate arithmetic was orally accused and ORACLE-CLEARED on real data (error
exactly 0.0). The real bug was a coverage gap: pgscan_direct ran at 357/s
(bursts 2962/s) while every plotted channel was zero — fixed by deriving
rates for all *_total columns. Anchors were added after reviewers and the
architect could not tell "healthy" from "broken" — that failure mode is now
mechanically detectable. Two alma2 test failures were a harness race
(non-atomic fixture swap); the product responded with an honest INVALID row.
