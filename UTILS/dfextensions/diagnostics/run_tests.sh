#!/bin/bash
# run_tests.sh - dfextensions/diagnostics subproject runner (PHASE_13_74_ADF,
# architect D4-ruling: diagnostics uses the standard per-subproject
# infrastructure like ADF/dfdraw/GB).
#
# Runs BOTH suites, writes logs, prints one standard SUMMARY line, exits
# nonzero on any failure. Usage:  bash diagnostics/run_tests.sh [logdir]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="${1:-$HERE/test_logs}"
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

echo "[run_tests] bash : $BLINE  (log: $BLOG)"
echo "[run_tests] pytest: $PLINE  (log: $PLOG)"

FAIL=0
[ "$BRC" -ne 0 ] && FAIL=1
case "$BLINE" in *"FAIL=0"*) : ;; *) FAIL=1;; esac
[ "$PRC" -ne 0 ] && FAIL=1

if [ "$FAIL" = 0 ]; then
  echo "SUMMARY: diagnostics OK - bash[$BLINE] pytest[$PLINE]"
else
  echo "SUMMARY: diagnostics FAILING - bash rc=$BRC [$BLINE] pytest rc=$PRC [$PLINE]"
  echo "  full logs: $BLOG  $PLOG"
fi
exit $FAIL
