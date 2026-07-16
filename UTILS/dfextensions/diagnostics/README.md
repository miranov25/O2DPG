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
