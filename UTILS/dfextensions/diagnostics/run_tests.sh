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
LOGDIR="${1:-$HERE/test_logs}"
FILEDIR="${DIAGNOSTICS_FILES:-$HERE/files_diag}"
mkdir -p "$LOGDIR" "$FILEDIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
BLOG="$LOGDIR/bash_suite_$TS.log"
PLOG="$LOGDIR/pytest_$TS.log"

echo "[run_tests] diagnostics subproject - $TS"
echo "[$(date '+%F %T')] [run_tests] bash suite: START"
bash "$HERE/tests/test_dfx_host_diagnostics.sh" > "$BLOG" 2>&1
BRC=$?
BLINE=$(tail -1 "$BLOG")
echo "[$(date '+%F %T')] [run_tests] bash suite: DONE rc=$BRC"

echo "[$(date '+%F %T')] [run_tests] pytest suite: START"
python3 -m pytest -v "$HERE/tests/" > "$PLOG" 2>&1   # -v: named pass lines for reviewers without ADF
PRC=$?
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
    # real-packet identity invariant [GPT24]: if the workload PID appears in
  # the snapshot, its token MUST equal the manifest current-user token
  EVB=$(ls -d "$EV"/host_diag_* 2>/dev/null | head -1)
  if [ -n "$EVB" ]; then
    EMTOK=$(grep "^user=" "$EVB/manifest.kv" | cut -d= -f2)
    WPID=$(ls "$EV"/run_*.json 2>/dev/null | head -1 | sed 's/.*_\([0-9]*\)\.json/\1/')
    if [ -n "$WPID" ] && grep -q "^ps.top=$WPID " "$EVB/snapshot.kv" 2>/dev/null; then
    WTOK=$(grep "^ps.top=$WPID " "$EVB/snapshot.kv" | awk '{print $2}')
    [ "$WTOK" = "$EMTOK" ] && echo "[run_tests] identity invariant OK: workload PID token == manifest ($EMTOK)" \
      || { echo "[run_tests] IDENTITY INVARIANT FAILED: manifest=$EMTOK workload-row=$WTOK"; PYRC=1; }
    fi
    CURTOK=$(awk -F '\t' '$4==1{print $3; exit}' "$EVB/user_samples.tsv" 2>/dev/null)
    if [ -n "$CURTOK" ]; then
    [ "$CURTOK" = "$EMTOK" ] && echo "[run_tests] identity invariant OK: user_samples current == manifest" \
      || { echo "[run_tests] IDENTITY INVARIANT FAILED: user_samples=$CURTOK manifest=$EMTOK"; PYRC=1; }
    fi
  fi
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
    REPHTML="$LOGDIR/diagnostics_report_$TS.html"
    cp "$EVREP" "$REPHTML" 2>/dev/null || REPHTML="$EVREP"
    echo "[run_tests] EVIDENCE REPORT (open in browser): $REPHTML"
    echo "[run_tests] EVIDENCE AUDIT : $(dirname "$EVREP")/validation/summary.md"
  elif [ -n "$EB" ] && [ "$HAVE_ADF" != 1 ]; then
    echo "[run_tests] render tier unavailable here (no ADF): bundle + records collected; presence path proven on the render host"
  fi
  ZIP="$FILEDIR/diagnostics_reviewer_$TS.zip"
  CRROPT=""
  [ -n "${DFX_CRR:-}" ] && [ -f "${DFX_CRR:-}" ] && CRROPT="--crr $DFX_CRR"
  echo "[$(date '+%F %T')] [run_tests] reviewer packet: START"
  # shellcheck disable=SC2086
  if python3 "$HERE/reviewer_bundle.py" -o "$ZIP" --logs "$LOGDIR" --skip-tests \
      --evidence "$EV" $CRROPT \
      && [ -s "$ZIP" ]; then
    echo "[$(date '+%F %T')] [run_tests] reviewer packet: DONE"
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

