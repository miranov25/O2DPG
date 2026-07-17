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
  ZIP="$DIAGROOT/diagnostics_reviewer_$TS.zip"
  python3 "$HERE/reviewer_bundle.py" -o "$ZIP" --logs "$LOGDIR" --skip-tests \
    && echo "[run_tests] reviewer zip: $ZIP"
  echo "SUMMARY: diagnostics OK - bash[$BLINE] pytest[$PLINE] zip[$ZIP]"
else
  echo "SUMMARY: diagnostics FAILING - bash rc=$BRC [$BLINE] pytest rc=$PRC [$PLINE] (no reviewer zip from a failing state)"
fi
exit $FAIL
