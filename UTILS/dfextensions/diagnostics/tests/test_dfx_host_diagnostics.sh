#!/bin/bash
# test_dfx_host_diagnostics.sh — PHASE_13_74_ADF D1 test suite (fixture-first, C-8)
# Every test hermetic where possible via PROC_ROOT/SYS_ROOT/CLK_TCK_OVERRIDE fixtures.
set -u
SCRIPT="$(cd "$(dirname "$0")/.." && pwd)/dfx_host_diagnostics.sh"
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
PASS=0; FAIL=0
ok(){ PASS=$((PASS+1)); echo "PASS  $1"; }
bad(){ FAIL=$((FAIL+1)); echo "FAIL  $1"; }
check(){ [ "$2" = "$3" ] && ok "$1 (got $2)" || bad "$1 (got '$2', want '$3')"; }

# ---------- fixture builders ----------
mkfix(){ # $1=root  builds a healthy synthetic /proc + /sys
  local R="$1"
  mkdir -p "$R/proc/pressure" "$R/proc/self" "$R/sys/kernel/mm/transparent_hugepage/khugepaged" "$R/sys/kernel/mm/hugepages/hugepages-2048kB"
  cat > "$R/proc/vmstat" <<'EOF'
compact_stall 100
compact_fail 40
thp_fault_alloc 120
thp_fault_fallback 100
thp_fault_fallback_charge 999
thp_collapse_alloc 7
pgscan_direct 3
allocstall_normal 12
allocstall_movable 8
pgmajfault 5
pswpin 1
pswpout 2
oom_kill 0
EOF
  printf 'MemTotal: 100 kB\nMemFree: 50 kB\nMemAvailable: 60 kB\nAnonHugePages: 0 kB\nSwapTotal: 10 kB\nSwapFree: 9 kB\nDirty: 1 kB\nSlab: 2 kB\n' > "$R/proc/meminfo"
  echo "100.00 90.00" > "$R/proc/uptime"
  printf 'some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n' > "$R/proc/pressure/memory"
  cp "$R/proc/pressure/memory" "$R/proc/pressure/cpu"; cp "$R/proc/pressure/memory" "$R/proc/pressure/io"
  echo "always madvise [never]" > "$R/sys/kernel/mm/transparent_hugepage/enabled"
  echo "always defer defer+madvise [madvise] never" > "$R/sys/kernel/mm/transparent_hugepage/defrag"
  echo 0 > "$R/sys/kernel/mm/transparent_hugepage/khugepaged/full_scans"
  # readable processes for the v8 process sampler (status+statm present)
  mkdir -p "$R/proc/500"; echo pytest > "$R/proc/500/comm"
  echo "500 (pytest) R 1 0 0 0 -1 0 0 0 0 0 300 300 0 0 20 0 1 0 40 1048576 250 18446744073709551615 0 0 0 0 0 0 0 0 0 0 0 0 17 1 0 0 0 0 0 0 0 0 0 0 0 0 0" > "$R/proc/500/stat"
  printf '260 250 5 1 0 1 0\n' > "$R/proc/500/statm"
  printf 'Name:\tpytest\nUid:\t%s\t%s\t%s\t%s\n' "$(id -u)" "$(id -u)" "$(id -u)" "$(id -u)" > "$R/proc/500/status"
  mkdir -p "$R/proc/net"
  printf 'Inter-|   Receive                                                |  Transmit\n face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n    lo: 1000 10 0 0 0 0 0 0 2000 20 0 0 0 0 0 0\n  eth0: 5000 50 1 2 0 0 0 0 7000 70 3 4 0 0 0 0\n' > "$R/proc/net/dev"
  # fake kcompactd kthread pid 61: utime=400000 stime=143210 ticks -> (543210)/100 = 5432.10 s
  mkdir -p "$R/proc/61"; echo kcompactd0 > "$R/proc/61/comm"
  printf 'Name:\tkcompactd0\nUid:\t0\t0\t0\t0\n' > "$R/proc/61/status"
  echo "61 (kcompactd0) S 2 0 0 0 -1 0 0 0 0 0 400000 143210 0 0 20 0 1 0 30 0 0 18446744073709551615 0 0 0 0 0 0 0 2147483647 0 0 0 0 17 1 0 0 0 0 0 0 0 0 0 0 0 0 0" > "$R/proc/61/stat"
  # self/stat for overhead accounting (fields 14,15)
  echo "1 (test) S 0 0 0 0 -1 0 0 0 0 0 10 10 0 0 20 0 1 0 1 0 0 18446744073709551615 0 0 0 0 0 0 0 0 0 0 0 0 17 1 0 0 0 0 0 0 0 0 0 0 0 0 0" > "$R/proc/self/stat"
  echo "fixture" > "$R/proc/cmdline"; echo "model name : Fixture CPU" > "$R/proc/cpuinfo"
  echo "1.25 0.80 0.50 2/345 9999" > "$R/proc/loadavg"
  printf 'cpu  1000 0 500 8000 100 0 0 0 0 0\nctxt 123456\n' > "$R/proc/stat"
  printf 'Node 0, zone Normal 10 10 10 10 10 10 10 10 10 10 10\n' > "$R/proc/buddyinfo"
  printf ' 8 0 sda 100 0 5000 0 0 0 0 0 0 0 0\n' > "$R/proc/diskstats"
}

FIX="$WORK/fix"; mkfix "$FIX"
export CLK_TCK_OVERRIDE=100
export PS_CMD_OVERRIDE="cat $WORK/psfix"
ME=$(id -un)   # own-user row built from the INVOKING user — never hardcoded (alma2 lesson)
printf 'PID USER VSZ RSS PCPU COMM\n1 %s 1000 100 0.1 python3\n2 otheruser 4000 100 0.2 secretjob\n' "$ME" > "$WORK/psfix"

export DFX_PROCESS_SAMPLER=off   # speed: python sampler only in T-D20
run(){ PROC_ROOT="$FIX/proc" SYS_ROOT="$FIX/sys" CGROUP_ROOT="$FIX/nocg" bash "$SCRIPT" "$@"; }

echo "== T-D1: snapshot on fixture -> exit 0, bundle complete =="
O="$WORK/o1"; mkdir "$O"; run -o "$O" > "$WORK/t1.out" 2> "$WORK/t1.err"; rc=$?
check "T-D1 exit" "$rc" 0
[ "$rc" != 0 ] && { echo "--- T-D1 collector stderr ---"; cat "$WORK/t1.err"; echo "-----------------------------"; }
B=$(ls -d "$O"/host_diag_* 2>/dev/null | head -1)
for f in manifest.kv snapshot.kv report.txt; do [ -s "$B/$f" ] && ok "T-D1 $f present" || bad "T-D1 $f missing"; done
check "T-D1 stderr empty" "$(wc -c < "$WORK/t1.err")" 0

echo "== T-D4: fabricated kthread -> exact nonzero CPU seconds in snapshot AND report =="
v=$(grep '^kthread.kcompactd_cpu_sum=' "$B/snapshot.kv" | cut -d= -f2)
check "T-D4 kcompactd_cpu_sum" "$v" "5432.10"
grep -q "kcompactd0 61 5432.10" "$B/report.txt" && ok "T-D4 report line" || bad "T-D4 report line"

echo "== T-D5: exact-key vmstat -> fallback=100 captured, _charge captured separately, no bleed =="
check "T-D5 thp_fault_fallback" "$(grep '^vmstat.thp_fault_fallback=' "$B/snapshot.kv" | cut -d= -f2)" 100
check "T-D5 fallback_charge own key" "$(grep '^vmstat.thp_fault_fallback_charge=' "$B/snapshot.kv" | cut -d= -f2)" 999
check "T-D5 allocstall_sum" "$(grep '^vmstat.allocstall_sum=' "$B/snapshot.kv" | cut -d= -f2)" 20

echo "== C-1: default ps redaction — other user hashed, comm masked; own visible =="
grep -q 'secretjob' "$B/snapshot.kv" && bad "C-1 other-user comm leaked" || ok "C-1 other-user comm masked"
grep -q 'otheruser' "$B/snapshot.kv" && bad "C-1 other-user name leaked" || ok "C-1 other-user name hashed"
grep -Eq "^ps.top=1 u_[0-9a-f]+ .*python3" "$B/snapshot.kv" && ok "C-1 own row visible (comm preserved, user tokenized per P0-F)" || bad "C-1 own row visible (comm preserved, user tokenized per P0-F)"
grep -Eq "^ps.top=1 $ME " "$B/snapshot.kv" && bad "C-1 own username absent from ps table" || ok "C-1 own username absent from ps table"

echo "== T-D6: REQUIRED evidence missing -> UNKNOWN, exit 2, manifest records reason =="
FIX2="$WORK/fix2"; mkfix "$FIX2"; rm "$FIX2/proc/vmstat"
O2="$WORK/o2"; mkdir "$O2"
PROC_ROOT="$FIX2/proc" SYS_ROOT="$FIX2/sys" bash "$SCRIPT" -o "$O2" > /dev/null 2> "$WORK/t6.err"; rc=$?
check "T-D6 exit" "$rc" 2
B2=$(ls -d "$O2"/host_diag_* | head -1)
grep -q '^verdict=UNKNOWN' "$B2/manifest.kv" && ok "T-D6 manifest UNKNOWN" || bad "T-D6 manifest UNKNOWN"
grep -q 'vmstat' "$WORK/t6.err" && ok "T-D6 reason names evidence" || bad "T-D6 reason names evidence"

echo "== T-D7/T-D14: CLI matrix incl. attached and mixed forms (C-6) =="
run -s 0 -n 0 >/dev/null 2>&1; check "T-D14 -s 0 -n 0 rejected" "$?" 1
run -s 5 >/dev/null 2>&1;      check "T-D14 -s alone rejected" "$?" 1
run -n 5 >/dev/null 2>&1;      check "T-D14 -n alone rejected" "$?" 1
O3="$WORK/o3"; mkdir "$O3"
run -o "$O3" -s1 -n2 >/dev/null 2> "$WORK/t7.err"; check "T-D14 ATTACHED -s1 -n2 accepted" "$?" 0
O4="$WORK/o4"; mkdir "$O4"
run -o "$O4" -n2 -s 1 >/dev/null 2>&1; check "T-D14 MIXED -n2 -s 1 accepted" "$?" 0
run extra-garbage >/dev/null 2>&1;  check "T-D14 positional rejected" "$?" 1
run -Z >/dev/null 2>&1;             check "T-D14 unknown flag rejected" "$?" 1
run -s >/dev/null 2>&1;             check "T-D14 missing value rejected" "$?" 1
run -s abc -n 2 >/dev/null 2>&1;    check "T-D14 non-integer rejected" "$?" 1
run -p abc >/dev/null 2>&1;         check "T-D14 bad PID rejected" "$?" 1

echo "== T-D3/T-D3b: sampling on CONTROLLED fixture -> exact rate oracle, empty stderr =="
# sample 1 reads state A; harness swaps to state B during sleep; sample 2 -> dt=10.00, compact_stall 100->1100 => 100.000/s
FIX3="$WORK/fix3"; mkfix "$FIX3"
O5="$WORK/o5"; mkdir "$O5"
( # EVENT-DRIVEN swap (2026-07-17 alma2 lesson): fixed sleeps race in BOTH
  # directions on slow filesystems (0.45s swap fired before a slow script even
  # took sample 1). Wait until sample 1 is WRITTEN, then swap atomically.
  CSVF=""
  for _i in $(seq 1 200); do
    CSVF=$(ls "$O5"/host_diag_*/host_samples.csv 2>/dev/null | head -1)
    [ -n "$CSVF" ] && [ "$(wc -l < "$CSVF")" -ge 2 ] && break
    sleep 0.05
  done
  sed 's/^compact_stall 100/compact_stall 1100/' "$FIX3/proc/vmstat" > "$FIX3/vm.new" && mv "$FIX3/vm.new" "$FIX3/proc/vmstat"
  printf '110.00 95.00\n' > "$FIX3/up.new" && mv "$FIX3/up.new" "$FIX3/proc/uptime" ) &
PROC_ROOT="$FIX3/proc" SYS_ROOT="$FIX3/sys" bash "$SCRIPT" -o "$O5" -s 1 -n 2 > /dev/null 2> "$WORK/t3.err"; rc=$?
wait
check "T-D3b exit" "$rc" 0
[ "$rc" != 0 ] && { echo "--- T-D3 collector stderr ---"; cat "$WORK/t3.err"; echo "-----------------------------"; }
check "T-D3b stderr empty" "$(wc -c < "$WORK/t3.err")" 0
B5=$(ls -d "$O5"/host_diag_* | head -1)
hdr=$(head -1 "$B5/host_samples.csv" | awk -F, '{print NF}')
r2=$(sed -n 3p "$B5/host_samples.csv")
check "T-D3b column count rows==header" "$(echo "$r2" | awk -F, '{print NF}')" "$hdr"
check "T-D3 exact dt oracle" "$(echo "$r2" | cut -d, -f2)" "10.00"
check "T-D3 exact compact_stall rate oracle" "$(echo "$r2" | cut -d, -f4)" "100.000"
check "T-D3 totals column keeps cumulative" "$(echo "$r2" | cut -d, -f3)" "1100"

echo "== T-D2: no writes outside the bundle (pre/post directory diff) =="
FIX4="$WORK/fix4"; mkfix "$FIX4"; O6="$WORK/o6"; mkdir "$O6"
LST=$(mktemp -d)
find "$WORK" -not -path "$O6/*" | sort > "$LST/pre.lst"
PROC_ROOT="$FIX4/proc" SYS_ROOT="$FIX4/sys" bash "$SCRIPT" -o "$O6" > /dev/null 2>&1
find "$WORK" -not -path "$O6/*" | sort > "$LST/post.lst"
diff -q "$LST/pre.lst" "$LST/post.lst" > /dev/null && ok "T-D2 no external writes" || bad "T-D2 external writes detected"
rm -rf "$LST"

echo "== T-D10: manifest provenance complete =="
for k in schema_version tool_version tool_md5 invocation host utc redaction verdict self_cpu_s self_wall_s; do
  grep -q "^$k=" "$B/manifest.kv" && ok "T-D10 $k" || bad "T-D10 $k missing"
done

echo "== collision guard: same-second bundles distinct (V2-6/R-8, structural) =="
O7="$WORK/o7"; mkdir "$O7"; run -o "$O7" >/dev/null 2>&1 & run -o "$O7" >/dev/null 2>&1; wait
check "collision: two bundles" "$(ls -d "$O7"/host_diag_* | wc -l)" 2

echo "== T-D15: anchor channels present; loadavg captured in snapshot and CSV =="
grep -q '^anchor.loadavg1=1.25' "$B/snapshot.kv" && ok "T-D15 snapshot anchor" || bad "T-D15 snapshot anchor"
hdrA=$(head -1 "$B5/host_samples.csv")
case "$hdrA" in *loadavg1,cpu_busy_pct,mem_available_kb,ctxt_per_s,procs_running) ok "T-D15 CSV anchor columns";; *) bad "T-D15 CSV anchor columns";; esac
la=$(sed -n 3p "$B5/host_samples.csv" | awk -F, '{print $(NF-4)}')
check "T-D15 loadavg1 value in row" "$la" "1.25"

echo "== T-D19: C-2 bundle naming - shareable has run_id not PID; raw has PID =="
ON="$WORK/on"; mkdir "$ON"; run -o "$ON" >/dev/null 2>&1
BN=$(basename "$(ls -d "$ON"/host_diag_*)")
case "$BN" in *_$$_*|*_$$) bad "T-D19 shareable name leaks PID ($BN)";; *) ok "T-D19 shareable name has no PID";; esac
echo "$BN" | grep -qE '_[0-9a-f]{8}$' && ok "T-D19 run_id suffix present" || bad "T-D19 run_id suffix present ($BN)"
OR="$WORK/or"; mkdir "$OR"; run -o "$OR" -r >/dev/null 2>&1
BR=$(basename "$(ls -d "$OR"/host_diag_*)")
echo "$BR" | grep -qE '_[0-9]+_[0-9a-f]{8}$' && ok "T-D19 raw name carries PID" || bad "T-D19 raw name carries PID ($BR)"

echo "== T-D20: sampling mode starts process sampler -> three CSVs with rows =="
OS="$WORK/os"; mkdir "$OS"
DFX_PROCESS_SAMPLER=on PROC_INTERVAL_OVERRIDE=1 PROC_ROOT="$FIX/proc" SYS_ROOT="$FIX/sys" CGROUP_ROOT="$FIX/nocg" DFX_UID_OVERRIDE=$(id -u) \
  bash "$SCRIPT" -o "$OS" -s 1 -n 2 >/dev/null 2>&1
BS=$(ls -d "$OS"/host_diag_* | head -1)
TD20_FAIL=0
for f in process_samples.csv user_samples.csv workload_rollup.csv; do
  if [ -f "$BS/$f" ] && [ "$(wc -l < "$BS/$f")" -gt 1 ]; then ok "T-D20 $f has rows"; else bad "T-D20 $f has rows"; TD20_FAIL=1; fi
done
if [ "$TD20_FAIL" = 1 ]; then
  echo "--- T-D20 SELF-DIAGNOSIS ---"
  echo "bundle: $BS"; ls -la "$BS" 2>/dev/null
  echo "--- sampler.log ---"; cat "$BS/sampler.log" 2>/dev/null || echo "(no sampler.log)"
  echo "--- manifest sampler lines ---"; grep -E "sampler|process_interval|run_id" "$BS/manifest.kv" 2>/dev/null
  echo "--- python3 -S smoke ---"; python3 -S -c "import pwd,signal,hashlib;print('stdlib OK')" 2>&1
  echo "----------------------------"
fi
grep -q '^process_interval_s=' "$BS/manifest.kv" && ok "T-D20 manifest cadence" || bad "T-D20 manifest cadence"
grep -q '^run_id=' "$BS/manifest.kv" && ok "T-D20 manifest run_id" || bad "T-D20 manifest run_id"

echo "== T-D22: v8 canonical per-entity tables (disk_samples/network_samples) =="
BS22=$(ls -d "$OS"/host_diag_* | head -1)
for f in disk_samples.csv network_samples.csv; do
  if [ -f "$BS22/$f" ] && [ "$(wc -l < "$BS22/$f")" -gt 1 ]; then ok "T-D22 $f has rows"; else bad "T-D22 $f has rows"; fi
done
grep -q "^ts,device,reads_completed" "$BS22/disk_samples.csv" && ok "T-D22 disk header canonical" || bad "T-D22 disk header canonical"
awk -F, 'NR>1{print $2}' "$BS22/network_samples.csv" | sort -u | grep -q "eth0" && ok "T-D22 net per-interface rows (eth0)" || bad "T-D22 net per-interface rows (eth0)"
TX=$(awk -F, '$2=="eth0"{print $7; exit}' "$BS22/network_samples.csv")
[ "$TX" = 7000 ] && ok "T-D22 eth0 tx_bytes oracle (=7000, P0-G)" || bad "T-D22 eth0 tx_bytes oracle (got '$TX', want 7000)"

echo "== T-D23: bundle-wide identity hunt (P0-F: shareable = tokens only) =="
ME23=$(id -un)
LEAK=$(grep -rlw "$ME23" "$BS22" 2>/dev/null | wc -l | tr -d ' ')
[ "$LEAK" = 0 ] && ok "T-D23 no real username anywhere in shareable bundle" || bad "T-D23 username found in $LEAK file(s): $(grep -rlw "$ME23" "$BS22" | head -2 | tr '\n' ' ')"
grep -q "^user=u_" "$BS22/manifest.kv" && ok "T-D23 manifest user tokenized" || bad "T-D23 manifest user tokenized"
grep "^invocation=" "$BS22/manifest.kv" | grep -q "/" && bad "T-D23 invocation path-free" || ok "T-D23 invocation path-free"
grep -q "^redaction=shareable-tokens" "$BS22/manifest.kv" && ok "T-D23 redaction declared" || bad "T-D23 redaction declared"
echo "== T-D24: ONE token namespace (P1-1) - manifest == snapshot own-row token =="
MTOK=$(grep "^user=" "$BS22/manifest.kv" | cut -d= -f2)
STOK=$(grep -E "^ps.top=1 " "$BS22/snapshot.kv" | awk '{print $2}' | head -1)
[ -n "$MTOK" ] && [ "$MTOK" = "$STOK" ] && ok "T-D24 manifest/snapshot token identical ($MTOK)" || bad "T-D24 token mismatch: manifest=$MTOK snapshot=$STOK"
US24="$BS22/user_samples.csv"
if [ -f "$US24" ]; then
  grep -q ",$MTOK," "$US24" && ok "T-D24 user_samples carries the same token" || bad "T-D24 user_samples token differs from manifest"
fi

OVS=$(grep "^self_cpu_s=" "$BS22/manifest.kv" | cut -d= -f2)
awk -v v="$OVS" 'BEGIN{exit (v>=0)?0:1}' && ok "T-D23 self_cpu_s non-negative (P0-H)" || bad "T-D23 self_cpu_s negative: $OVS"

echo "== T-D21: TERM during bounded run -> CLEAN stop: verdict + stop_reason =="
OC="$WORK/oc"; mkdir "$OC"
PROC_ROOT="$FIX/proc" SYS_ROOT="$FIX/sys" CGROUP_ROOT="$FIX/nocg" bash "$SCRIPT" -o "$OC" -s 1 -n 30 > "$WORK/cs.out" 2>/dev/null &
CPID=$!
sleep 2.6
kill -TERM "$CPID"; wait "$CPID"; rcc=$?
check "T-D21 exit 0 on TERM" "$rcc" 0
BC=$(ls -d "$OC"/host_diag_* | head -1)
grep -q '^verdict=' "$BC/manifest.kv" && ok "T-D21 verdict written" || bad "T-D21 verdict written"
grep -q '^stop_reason=signal_clean_stop' "$BC/manifest.kv" && ok "T-D21 stop_reason" || bad "T-D21 stop_reason"
NSC=$(grep '^samples_taken=' "$BC/manifest.kv" | cut -d= -f2)
[ "${NSC:-0}" -ge 2 ] && ok "T-D21 samples_taken>=2 (got $NSC)" || bad "T-D21 samples_taken>=2 (got $NSC)"

echo "== T-D11a: bundle dir vanishes mid-run -> loud FATAL, exit 1 (2026-07-16 incident) =="
FIXV="$WORK/fixv"; mkfix "$FIXV"; OV="$WORK/ov"; mkdir "$OV"
( sleep 0.4; rm -rf "$OV"/host_diag_* ) &
PROC_ROOT="$FIXV/proc" SYS_ROOT="$FIXV/sys" bash "$SCRIPT" -o "$OV" -s 1 -n 3 >/dev/null 2> "$WORK/tv.err"; rcv=$?
wait
check "T-D11a exit" "$rcv" 1
grep -q "bundle directory vanished" "$WORK/tv.err" && ok "T-D11a loud FATAL message" || bad "T-D11a loud FATAL message"

echo "== verdict rules: THP-01 fires on [always] fixture =="
FIX5="$WORK/fix5"; mkfix "$FIX5"; echo "[always] madvise never" > "$FIX5/sys/kernel/mm/transparent_hugepage/enabled"
# quiet the kthread so ONLY the config rule fires (base fixture deliberately trips KC-01)
sed -i 's/400000 143210/10 10/' "$FIX5/proc/61/stat"
O8="$WORK/o8"; mkdir "$O8"
PROC_ROOT="$FIX5/proc" SYS_ROOT="$FIX5/sys" bash "$SCRIPT" -o "$O8" >/dev/null 2>&1
B8=$(ls -d "$O8"/host_diag_* | head -1)
check "THP-01 verdict WARN" "$(grep '^verdict=' "$B8/manifest.kv" | cut -d= -f2)" "WARN"
grep -q '^verdict_rules= THP-01' "$B8/manifest.kv" && ok "THP-01 rule id recorded" || bad "THP-01 rule id recorded"

echo "== KC-01 fires on base fixture (5432s kthread CPU @ 100s uptime, uptime-normalized) =="
check "KC-01 base verdict UNHEALTHY" "$(grep '^verdict=' "$B/manifest.kv" | cut -d= -f2)" "UNHEALTHY"
grep -q 'KC-01' "$B/manifest.kv" && ok "KC-01 rule id recorded" || bad "KC-01 rule id recorded"

echo "== PSI-01 fires on high mem-full fixture -> UNHEALTHY =="
FIX6="$WORK/fix6"; mkfix "$FIX6"
printf 'some avg10=9.0 avg60=8.0 avg300=1.0 total=1\nfull avg10=7.0 avg60=6.5 avg300=1.0 total=1\n' > "$FIX6/proc/pressure/memory"
O9="$WORK/o9"; mkdir "$O9"
PROC_ROOT="$FIX6/proc" SYS_ROOT="$FIX6/sys" bash "$SCRIPT" -o "$O9" >/dev/null 2>&1
B9=$(ls -d "$O9"/host_diag_* | head -1)
check "PSI-01 verdict UNHEALTHY" "$(grep '^verdict=' "$B9/manifest.kv" | cut -d= -f2)" "UNHEALTHY"

echo
echo "==================== RESULT: PASS=$PASS FAIL=$FAIL ===================="
[ "$FAIL" = 0 ]
