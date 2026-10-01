#!/bin/bash
# run_tests.sh - dfextensions/diagnostics subproject runner (PHASE_13_74_ADF).
# ORG CONVENTION (same as ADF/dfdraw): running the tests ALSO produces the
# reviewer zip - one command, one gate, one distributable artifact.
#
# Usage:  bash diagnostics/run_tests.sh [logdir]
#   default logs/evidence: diagnostics/test_logs/ (Git-ignored)
#   reviewer zip:          diagnostics/files_diag/ (Git-ignored)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

check_validation_csv() {
  python3 - "$1" <<'PYEOF'
import csv
import sys

path = sys.argv[1]
bad = []
with open(path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        status = (row.get("status") or "").strip().upper()
        if status in {"FAIL", "ERROR"}:
            bad.append((row.get("check_id", "?"), status, row.get("detail", "")))

for check_id, status, detail in bad:
    print(f"[run_tests] validation {status}: {check_id}: {detail}")
sys.exit(1 if bad else 0)
PYEOF
}

# Test hook: exercise the exact global validation gate without running the full
# diagnostics suite. Canonical execution below calls the same function.
if [ -n "${DFX_VALIDATE_ONLY_CSV:-}" ]; then
  check_validation_csv "$DFX_VALIDATE_ONLY_CSV"
  exit $?
fi
LOGDIR="${1:-$HERE/test_logs}"
FILEDIR="${DIAGNOSTICS_FILES:-$HERE/files_diag}"
mkdir -p "$LOGDIR" "$FILEDIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
BLOG="$LOGDIR/bash_suite_$TS.log"
PLOG="$LOGDIR/pytest_$TS.log"

echo "[run_tests] diagnostics subproject - $TS"
echo "[$(date '+%F %T')] [run_tests] bash suite: START"
bash "$HERE/tests/test_dfx_host_diagnostics.sh" > "$BLOG" 2>&1 &
BASH_PID=$!

echo "[$(date '+%F %T')] [run_tests] pytest suite: START"
python3 -m pytest -v "$HERE/tests/" > "$PLOG" 2>&1 &
PYTEST_PID=$!

wait "$BASH_PID"; BRC=$?
BLINE=$(tail -1 "$BLOG")
echo "[$(date '+%F %T')] [run_tests] bash suite: DONE rc=$BRC"

wait "$PYTEST_PID"; PRC=$?
PLINE=$(grep -E "[0-9]+ (passed|failed|error)" "$PLOG" | tail -1 | sed "s/=//g; s/^ *//; s/ *$//")
echo "[$(date '+%F %T')] [run_tests] pytest suite: DONE rc=$PRC"

echo "[run_tests] bash : $BLINE"
echo "[run_tests] pytest: $PLINE"
echo "[run_tests] logs : $BLOG  $PLOG"

FAIL=0
[ "$BRC" -ne 0 ] && FAIL=1
case "$BLINE" in *"FAIL=0"*) : ;; *) FAIL=1;; esac
[ "$PRC" -ne 0 ] && FAIL=1

if [ "$FAIL" = 0 ]; then
  # CRR-12: every packet carries REAL rendered evidence from THIS host
  EV="$LOGDIR/evidence_$TS"; mkdir -p "$EV"
  EVN="${DFX_EVIDENCE_SAMPLES:-20}"   # 20 x 0.5s: denser evidence with shorter wall time
  echo "[$(date '+%F %T')] [run_tests] evidence run: START (-s 0.5 -n $EVN)"
  # P0-6 (round-3 panel): the packet's evidence is produced by the REAL
  # wrapper around a REAL recorded workload, so the SHIPPED report shows the
  # presence path - not merely a test proving it possible.
  WL="$EV/evidence_workload.py"
  mkdir -p "$EV"
  cat > "$WL" << 'PYEOF'
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])) if False else sys.argv[1])
from run_metrics import RunMetrics
n = int(sys.argv[2])
with RunMetrics("evidence_job") as rm:      # default out: wrapper env handoff
    v = 0
    for i in range(max(2, n - 4)):
        time.sleep(0.5)
        v += 100
        rm.record_event("progress", {"value": v})
PYEOF
  HAVE_ADF=0; python3 -c "import sys; sys.path.insert(0,'$HERE/../AliasDataFrame'); import AliasDataFrame" 2>/dev/null && HAVE_ADF=1
  RPT_OPT=""; [ "$HAVE_ADF" = 1 ] && RPT_OPT="--report"
  # shellcheck disable=SC2086
  python3 "$HERE/dfx_run_with_diagnostics.py" --out "$EV" --label evidence \
      --interval 0.5 --max-samples "$EVN" --pre "${DFX_EVIDENCE_PRE:-1}" --post "${DFX_EVIDENCE_POST:-2}" $RPT_OPT -- \
      python3 "$WL" "$HERE" "$EVN" > "$EV/collect.log" 2>&1 \
    || echo "[run_tests] evidence wrapper degraded rc=$? (see $EV/collect.log)"
  echo "[$(date '+%F %T')] [run_tests] evidence run: DONE"
  EB=$(ls -d "$EV"/host_diag_* 2>/dev/null | head -1)
  EVREP=$(ls -d "$EV"/report_*/report.html 2>/dev/null | head -1)
  # PHASE_13_75 full-fidelity identity invariant.
  EVB=$(ls -d "$EV"/host_diag_* 2>/dev/null | head -1)
  if [ -n "$EVB" ]; then
    EMUSER=$(grep "^user=" "$EVB/manifest.kv" | cut -d= -f2)
    CURUSER=$(awk -F '\t' '$4==1{print $3; exit}' "$EVB/user_samples.tsv" 2>/dev/null)
    if [ -n "$CURUSER" ]; then
      [ "$CURUSER" = "$EMUSER" ] && echo "[run_tests] identity invariant OK: user_samples current == manifest ($EMUSER)" \
        || { echo "[run_tests] IDENTITY INVARIANT FAILED: user_samples=$CURUSER manifest=$EMUSER"; FAIL=1; }
    fi
    if [ -f "$EVB/process_catalog.tsv" ]; then
      grep -q $'\t'"$EMUSER"$'\t' "$EVB/process_catalog.tsv" \
        && echo "[run_tests] identity invariant OK: process_catalog carries real username" \
        || { echo "[run_tests] IDENTITY INVARIANT FAILED: process_catalog missing $EMUSER"; FAIL=1; }
      awk -F '\t' 'NR>1 && $19==1{found=1} END{if(!found) fail=1; if(fail) print "missing"; else print "ok"}' "$EVB/process_catalog.tsv" | grep -q '^ok$' \
        && echo "[run_tests] target hierarchy evidence OK: process_catalog has target-job row" \
        || { echo "[run_tests] TARGET HIERARCHY EVIDENCE FAILED: no target-job process in catalog"; FAIL=1; }
    fi
  fi
  if [ -n "$EVREP" ]; then
    # build-time PRESENCE assertion: shipped evidence must show the job
    if grep -q "no run_metrics records supplied" "$EVREP"; then
      echo "[run_tests] EVIDENCE PRESENCE CHECK FAILED: report shows the absence path"
      FAIL=1
    else
      echo "[run_tests] evidence presence check OK (Layer-C populated)"
    fi
    AUDITCSV="$(dirname "$EVREP")/validation/transition_checks.csv"
    if check_validation_csv "$AUDITCSV"; then
      echo "[run_tests] evidence audit: global PASS (PASS/SKIP only)"
    else
      echo "[run_tests] EVIDENCE AUDIT FAILED: validation contains FAIL/ERROR"
      FAIL=1
    fi
    grep -q '"I-COV-S7_job_host_analysis","[^"]*","PASS"' "$AUDITCSV" \
      && echo "[run_tests] evidence audit: S7 PASS" \
      || { echo "[run_tests] EVIDENCE AUDIT: S7 not PASS"; FAIL=1; }
    REPHTML="$LOGDIR/diagnostics_report_$TS.html"
    cp "$EVREP" "$REPHTML" 2>/dev/null || REPHTML="$EVREP"
    echo "[run_tests] EVIDENCE REPORT (open in browser): $REPHTML"
    echo "[run_tests] EVIDENCE AUDIT : $(dirname "$EVREP")/validation/summary.md"
  elif [ -n "$EB" ] && [ "$HAVE_ADF" != 1 ]; then
    echo "[run_tests] render tier unavailable here (no ADF): bundle + records collected; presence path proven on the render host"
  fi
fi

ZIP_TAG=""
[ "$FAIL" -ne 0 ] && ZIP_TAG="_FAILED"
ZIP="$FILEDIR/diagnostics_reviewer${ZIP_TAG}_$TS.zip"
CRROPT=""
[ -n "${DFX_CRR:-}" ] && [ -f "${DFX_CRR:-}" ] && CRROPT="--crr $DFX_CRR"
EVOPT=""
[ -n "${EV:-}" ] && [ -d "${EV:-}" ] && EVOPT="--evidence $EV"

STATUS="$LOGDIR/run_status_$TS.txt"
{
  echo "timestamp=$TS"
  echo "bash_rc=$BRC"
  echo "bash_summary=$BLINE"
  echo "pytest_rc=$PRC"
  echo "pytest_summary=$PLINE"
  echo "gate_fail=$FAIL"
} > "$STATUS"

echo "[$(date '+%F %T')] [run_tests] reviewer packet: START state=$([ "$FAIL" -eq 0 ] && echo OK || echo FAILING)"
# shellcheck disable=SC2086
if python3 "$HERE/reviewer_bundle.py" -o "$ZIP" --logs "$LOGDIR" --skip-tests $EVOPT $CRROPT \
    && [ -s "$ZIP" ]; then
  echo "[$(date '+%F %T')] [run_tests] reviewer packet: DONE"
  echo "[run_tests] reviewer zip: $ZIP"
else
  echo "[run_tests] reviewer packet creation FAILED"
  FAIL=1
fi

if [ "$FAIL" = 0 ]; then
  echo "SUMMARY: diagnostics OK - bash[$BLINE] pytest[$PLINE] zip[$ZIP] report[${REPHTML:-none}]"
else
  echo "SUMMARY: diagnostics FAILING - bash rc=$BRC [$BLINE] pytest rc=$PRC [$PLINE] zip[$ZIP]"
fi

test "$FAIL" -eq 0

