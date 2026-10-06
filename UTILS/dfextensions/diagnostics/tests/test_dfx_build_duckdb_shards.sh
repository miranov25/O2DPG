#!/usr/bin/env bash
set -u

SCRIPT=${1:-}
if [ -z "$SCRIPT" ] || [ ! -f "$SCRIPT" ]; then
  echo "usage: $0 /path/to/dfx_build_duckdb_shards.sh" >&2
  exit 2
fi

PASS=0
FAIL=0

ok() {
  PASS=$((PASS + 1))
  echo "PASS  $*"
}

bad() {
  FAIL=$((FAIL + 1))
  echo "FAIL  $*" >&2
}

TMP=$(mktemp -d "${TMPDIR:-/tmp}/dfx_shards_test.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

ROOT="$TMP/production"
HOST="testhost"
TRACE="$TMP/trace"
mkdir -p "$ROOT/$HOST" "$TRACE"

make_status() {
  local slot="$1"
  local state="$2"
  local bundle="$3"
  mkdir -p "$slot"
  cat > "$slot/status_001.kv" <<EOF
state=$state
bundle=$bundle
EOF
}

A="$ROOT/$HOST/2026/10/01/slot_00/host_diag_A"
B="$ROOT/$HOST/2026/10/01/slot_06/host_diag_B"
C="$ROOT/$HOST/2026/10/01/slot_12/host_diag_C"
D="$ROOT/$HOST/2026/10/01/slot_18/host_diag_D"
E="$ROOT/$HOST/2026/10/02/slot_00/host_diag_E"

mkdir -p "$A" "$B" "$C" "$D" "$E"

make_status "$(dirname "$A")" COMPLETE "$A"
make_status "$(dirname "$B")" COMPLETE "$B"
make_status "$(dirname "$C")" COMPLETE "$C"
make_status "$(dirname "$D")" RUNNING "$D"
make_status "$(dirname "$E")" FAILED "$E"

printf 'already-built\n' > "$(dirname "$B")/$(basename "$B").duckdb"

FAKE="$TMP/fake_query.py"
cat > "$FAKE" <<'PY'
#!/usr/bin/env python3
import os
import pathlib
import sys
import time

args = sys.argv[1:]
if len(args) < 5 or args[0] != "build-db":
    raise SystemExit(2)

bundle = pathlib.Path(args[args.index("--bundle") + 1])
db = pathlib.Path(args[args.index("--db") + 1])
trace = pathlib.Path(os.environ["DFX_TEST_TRACE_DIR"])
trace.mkdir(parents=True, exist_ok=True)

pid = os.getpid()
(trace / f"start.{pid}").write_text(str(time.time_ns()))
time.sleep(0.40)
db.write_text(f"built from {bundle}\n")
(trace / f"end.{pid}").write_text(str(time.time_ns()))
print(f"db={db}")
PY
chmod 755 "$FAKE"

export DFX_TEST_TRACE_DIR="$TRACE"

OUT1="$TMP/out1.txt"
bash "$SCRIPT" \
  --root "$ROOT" \
  --host "$HOST" \
  --query-tool "$FAKE" \
  -j 2 \
  >"$OUT1" 2>&1
RC1=$?

[ "$RC1" -eq 0 ] && ok "parallel shard build exits 0" || bad "parallel shard build rc=$RC1"

DB_A="$(dirname "$A")/$(basename "$A").duckdb"
DB_B="$(dirname "$B")/$(basename "$B").duckdb"
DB_C="$(dirname "$C")/$(basename "$C").duckdb"
DB_D="$(dirname "$D")/$(basename "$D").duckdb"

[ -s "$DB_A" ] && [ -s "$DB_C" ] && ok "missing COMPLETE shards built" || bad "missing COMPLETE shards"
[ "$(cat "$DB_B")" = "already-built" ] && ok "existing shard preserved" || bad "existing shard changed"
[ ! -e "$DB_D" ] && ok "RUNNING bundle ignored" || bad "RUNNING bundle was built"

python3 - "$TRACE" <<'PY'
import pathlib, sys
trace = pathlib.Path(sys.argv[1])
starts = {p.name.split(".",1)[1]: int(p.read_text()) for p in trace.glob("start.*")}
ends = {p.name.split(".",1)[1]: int(p.read_text()) for p in trace.glob("end.*")}
pids = sorted(set(starts) & set(ends))
overlap = False
for i, a in enumerate(pids):
    for b in pids[i+1:]:
        if starts[a] < ends[b] and starts[b] < ends[a]:
            overlap = True
if not overlap:
    raise SystemExit(1)
PY
[ "$?" -eq 0 ] && ok "at least two shard builds overlapped" || bad "parallel execution not observed"

grep -q 'complete_statuses=3' "$OUT1" &&
grep -q 'running_statuses=1' "$OUT1" &&
grep -q 'other_statuses=1' "$OUT1" &&
grep -q 'existing=1' "$OUT1" &&
grep -q 'built=2' "$OUT1" &&
grep -q 'failed=0' "$OUT1" &&
  ok "first-run summary" || bad "first-run summary"

OUT2="$TMP/out2.txt"
bash "$SCRIPT" \
  --root "$ROOT" \
  --host "$HOST" \
  --query-tool "$FAKE" \
  -j 4 \
  >"$OUT2" 2>&1
RC2=$?

[ "$RC2" -eq 0 ] && ok "idempotent rerun exits 0" || bad "idempotent rerun rc=$RC2"

grep -q 'existing=3' "$OUT2" &&
grep -q 'missing_to_build=0' "$OUT2" &&
grep -q 'built=0' "$OUT2" &&
  ok "rerun skips all existing shards" || bad "rerun did not skip existing shards"

echo
echo "RESULT: PASS=$PASS FAIL=$FAIL"

[ "$FAIL" -eq 0 ]
