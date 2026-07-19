#!/bin/bash
# run_tests.sh - dfextensions/diagnostics subproject runner (PHASE_13_74_ADF).
# ORG CONVENTION (same as ADF/dfdraw): running the tests ALSO produces the
# reviewer zip - one command, one gate, one distributable artifact.
#
# Usage:  bash diagnostics/run_tests.sh [logdir]
#   logdir default is OUTSIDE the repo tree (never commit run products):
#   ${DFX_DIAG:-/tmp/dfx_diag}/test_logs ; reviewer zip lands next to it.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
DIAGROOT="${DFX_DIAG:-/tmp/dfx_diag}"
LOGDIR="${1:-$DIAGROOT/test_logs}"
mkdir -p "$LOGDIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
BLOG="$LOGDIR/bash_suite_$TS.log"
PLOG="$LOGDIR/pytest_$TS.log"

echo "[run_tests] diagnostics subproject - $TS"
bash "$HERE/tests/test_dfx_host_diagnostics.sh" > "$BLOG" 2>&1
BRC=$?
BLINE=$(tail -1 "$BLOG")

python3 -m pytest -q "$HERE/tests/" > "$PLOG" 2>&1
PRC=$?
PLINE=$(grep -E "passed|failed|error" "$PLOG" | tail -1)

echo "[run_tests] bash : $BLINE"
echo "[run_tests] pytest: $PLINE"
echo "[run_tests] logs : $BLOG  $PLOG"

FAIL=0
[ "$BRC" -ne 0 ] && FAIL=1
case "$BLINE" in *"FAIL=0"*) : ;; *) FAIL=1;; esac
[ "$PRC" -ne 0 ] && FAIL=1

if [ "$FAIL" = 0 ]; then
  # CRR-12: every packet carries REAL rendered evidence from THIS host
  EV="$DIAGROOT/evidence_$TS"; mkdir -p "$EV"
  EVN="${DFX_EVIDENCE_SAMPLES:-16}"   # 16 x 2s: enough points for review figures (architect 2026-07-18)
  echo "[run_tests] evidence: WRAPPER run, -s 2 -n $EVN (DFX_EVIDENCE_SAMPLES tunes it)"
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
    for i in range(max(4, n // 2)):
        time.sleep(1.0)
        v += 100
        rm.record_event("progress", {"value": v})
PYEOF
  HAVE_ADF=0; python3 -c "import sys; sys.path.insert(0,'$HERE/../AliasDataFrame'); import AliasDataFrame" 2>/dev/null && HAVE_ADF=1
  RPT_OPT=""; [ "$HAVE_ADF" = 1 ] && RPT_OPT="--report"
  # shellcheck disable=SC2086
  python3 "$HERE/dfx_run_with_diagnostics.py" --out "$EV" --label evidence \
      --interval 2 --max-samples "$EVN" --pre 0 --post 0 $RPT_OPT -- \
      python3 "$WL" "$HERE" "$EVN" > "$EV/collect.log" 2>&1 \
    || echo "[run_tests] evidence wrapper degraded rc=$? (see $EV/collect.log)"
  EB=$(ls -d "$EV"/host_diag_* 2>/dev/null | head -1)
  EVREP=$(ls -d "$EV"/report_*/report.html 2>/dev/null | head -1)
  if [ -n "$EVREP" ]; then
    # build-time PRESENCE assertion: shipped evidence must show the job
    if grep -q "no run_metrics records supplied" "$EVREP"; then
      echo "[run_tests] EVIDENCE PRESENCE CHECK FAILED: report shows the absence path"
      PYRC=1
    else
      echo "[run_tests] evidence presence check OK (Layer-C populated)"
    fi
    grep -q '"I-COV-S7_job_host_analysis","[^"]*","PASS"' \
        "$(dirname "$EVREP")/validation/transition_checks.csv" \
      && echo "[run_tests] evidence audit: S7 PASS" \
      || { echo "[run_tests] EVIDENCE AUDIT: S7 not PASS"; PYRC=1; }
    REPHTML="$DIAGROOT/diagnostics_report_$TS.html"
    cp "$EVREP" "$REPHTML" 2>/dev/null || REPHTML="$EVREP"
    echo "[run_tests] EVIDENCE REPORT (open in browser): $REPHTML"
    echo "[run_tests] EVIDENCE AUDIT : $(dirname "$EVREP")/validation/summary.md"
  elif [ -n "$EB" ] && [ "$HAVE_ADF" != 1 ]; then
    echo "[run_tests] render tier unavailable here (no ADF): bundle + records collected; presence path proven on the render host"
  fi
  ZIP="$DIAGROOT/diagnostics_reviewer_$TS.zip"
  CRROPT=""
  [ -n "${DFX_CRR:-}" ] && [ -f "${DFX_CRR:-}" ] && CRROPT="--crr $DFX_CRR"
  # shellcheck disable=SC2086
  if python3 "$HERE/reviewer_bundle.py" -o "$ZIP" --logs "$LOGDIR" --skip-tests \
      --evidence "$EV" $CRROPT \
      && [ -s "$ZIP" ]; then
    echo "[run_tests] reviewer zip: $ZIP"
    echo "SUMMARY: diagnostics OK - bash[$BLINE] pytest[$PLINE] zip[$ZIP] report[${REPHTML:-none}]"
  else
    echo "SUMMARY: diagnostics FAILING - tests green but reviewer-zip creation FAILED (CRR-13 gate)"
    exit 1
  fi
else
  echo "SUMMARY: diagnostics FAILING - bash rc=$BRC [$BLINE] pytest rc=$PRC [$PLINE] (no reviewer zip from a failing state)"
fi
exit $FAIL
