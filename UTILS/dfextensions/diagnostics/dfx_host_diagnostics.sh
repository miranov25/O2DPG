#!/bin/bash
# =============================================================================
# dfx_host_diagnostics.sh — READ-ONLY host memory/THP pathology collector
# PHASE_13_74_ADF D1 · v2.0 · dfextensions-scoped (ADF, GB, dfdraw consumers)
#
# Emits a DIAGNOSTIC BUNDLE (directory): manifest.kv, snapshot.kv,
# samples.csv (sampling mode), report.txt. Flat key=value + CSV only —
# JSON conversion is schema.py's job (checklist C-13). Read-only against
# the host: writes ONLY inside its own bundle directory.
#
# USAGE:
#   dfx_host_diagnostics.sh [-o DIR] [-p PID [-A]] [-d DATA_PATH] [-r]
#                           [-s INTERVAL_S -n NSAMPLES]
#     -o DIR        parent directory for the bundle (default: .)
#     -p PID        per-process deep-dive; non-owned PID requires -A (C-1)
#     -A            acknowledge deep-dive on a process you do not own
#     -d DATA_PATH  record filesystem/mount type of a named data path
#     -r            raw mode: NO redaction (local use only; recorded in manifest)
#     -s/-n         sampling: NSAMPLES samples, INTERVAL_S seconds apart
#
# EXIT CONTRACT (v2 §2, R-4): 0 = bundle produced (PASS/WARN/UNHEALTHY inside);
#   1 = invocation/internal error; 2 = verdict UNKNOWN (REQUIRED evidence
#   unreadable / unsupported platform) — a minimal bundle is still written
#   when possible so UNKNOWN is itself evidence.
#
# TESTABILITY (checklist C-8): every read path honors PROC_ROOT, SYS_ROOT,
#   CGROUP_ROOT, CLK_TCK_OVERRIDE, PS_CMD_OVERRIDE for hermetic fixtures.
# =============================================================================
set -u
VERSION="2.0"
PROC="${PROC_ROOT:-/proc}"
SYS="${SYS_ROOT:-/sys}"
CGR="${CGROUP_ROOT:-/sys/fs/cgroup}"
CLK_TCK="${CLK_TCK_OVERRIDE:-$(getconf CLK_TCK 2>/dev/null || echo 100)}"
SELF_USER="$(id -un 2>/dev/null || echo unknown)"

usage(){ echo "usage: $0 [-o DIR] [-p PID [-A]] [-d DATA_PATH] [-r] [-s INTERVAL_S -n NSAMPLES]" >&2; exit 1; }

# ---- argument parsing: flags tracked INSIDE getopts (C-6: never re-parse $*);
# ---- leading-colon silent mode separates missing-value from unknown-flag (C-7)
PID=""; INTERVAL=0; NSAMPLES=0; RUN_ID=""; JOB_CGROUP=""; TARGET_PID_FILE=""; OUTDIR="."; DATAPATH=""; RAW=0; ACK=0
S_GIVEN=0; N_GIVEN=0
while getopts ":p:s:n:o:d:rAR:G:F:" opt; do case $opt in
  R) RUN_ID=$OPTARG;; G) JOB_CGROUP=$OPTARG;; F) TARGET_PID_FILE=$OPTARG;;
  p) PID=$OPTARG;;
  s) INTERVAL=$OPTARG; S_GIVEN=1;;
  n) NSAMPLES=$OPTARG; N_GIVEN=1;;
  o) OUTDIR=$OPTARG;;
  d) DATAPATH=$OPTARG;;
  r) RAW=1;;
  A) ACK=1;;
  :)  echo "option -$OPTARG requires a value" >&2; usage;;
  \?) echo "unknown option -$OPTARG" >&2; usage;;
esac; done
shift $((OPTIND-1))
[ $# -gt 0 ] && { echo "unexpected arguments: $*" >&2; usage; }
case "$PID" in *[!0-9]*) echo "invalid PID '$PID'" >&2; usage;; esac
case "$INTERVAL$NSAMPLES" in *[!0-9]*) echo "interval/samples must be integers" >&2; usage;; esac
[ "$S_GIVEN" != "$N_GIVEN" ] && { echo "-s and -n must be given together" >&2; usage; }
if [ "$S_GIVEN" = 1 ] && { [ "$INTERVAL" -le 0 ] || [ "$NSAMPLES" -le 0 ]; }; then
  echo "interval and samples must be positive" >&2; usage; fi
[ -d "$OUTDIR" ] || { echo "output dir '$OUTDIR' does not exist" >&2; exit 1; }
if [ -n "$PID" ]; then
  [ -d "$PROC/$PID" ] || { echo "no such process $PID" >&2; exit 1; }
  P_OWNER="$(stat -c %U "$PROC/$PID" 2>/dev/null || echo unknown)"
  if [ "$P_OWNER" != "$SELF_USER" ] && [ "$ACK" != 1 ]; then
    echo "pid $PID is owned by '$P_OWNER', not you; re-run with -A to acknowledge deep-dive on another user's process (C-1)" >&2
    exit 1
  fi
fi

# ---- bundle: directory creation is the atomic collision guard (V2-6)
HOST=$(hostname -s 2>/dev/null || echo unknown-host)
UTC=$(date -u +%Y%m%dT%H%M%SZ)
# run_id: orchestration-supplied (-R) or generated; C-2: shareable bundle name
# carries run_id only - raw PID appears in the name ONLY in raw/local (-r) mode
[ -n "$RUN_ID" ] || RUN_ID=$(printf '%s%s' "$(date +%s)" "$$" | md5sum | cut -c1-8)
if [ "$RAW" = 1 ]; then
  BUNDLE="$OUTDIR/host_diag_${HOST}_${UTC}_$$_${RUN_ID}"
else
  BUNDLE="$OUTDIR/host_diag_${HOST}_${UTC}_${RUN_ID}"
fi
umask 077
mkdir "$BUNDLE" 2>/dev/null || { echo "bundle dir '$BUNDLE' already exists — refusing" >&2; exit 1; }
MAN="$BUNDLE/manifest.kv"; SNAP="$BUNDLE/snapshot.kv"; REP="$BUNDLE/report.txt"
# v8 canonical bundle tables (proposal lines 720-724): host totals + per-entity
CSV="$BUNDLE/host_samples.csv"
DISKCSV="$BUNDLE/disk_samples.csv"
NETCSV="$BUNDLE/network_samples.csv"
: > "$MAN"; : > "$SNAP"; : > "$REP"
man(){ echo "$1=$2" >> "$MAN"; }
kv(){ echo "$1=$2" >> "$SNAP"; }
st(){ echo "state.$1=$2" >> "$SNAP"; }   # evidence-class states (C-2): available/unavailable/permission_denied/not_supported/parse_error
say(){ echo "$@" >> "$REP"; }
sec(){ say ""; say "===== $1 ====="; }

SELF_MD5=$( { md5sum "$0" 2>/dev/null || md5 -q "$0" 2>/dev/null; } | awk '{print $1}' )
man schema_version 1
man tool dfx_host_diagnostics
man tool_version "$VERSION"
man tool_md5 "${SELF_MD5:-unknown}"
man invocation "$0 $*${PID:+ -p $PID}${S_GIVEN:+ -s $INTERVAL -n $NSAMPLES}"
man run_id "$RUN_ID"
man host "$HOST"; man utc "$UTC"; man user "$SELF_USER"
man redaction $([ "$RAW" = 1 ] && echo raw || echo shareable)
man proc_root "$PROC"; man sys_root "$SYS"; man clk_tck "$CLK_TCK"

# ---- self-overhead accounting start (C-9 draft thresholds; measured, not asserted)
read_self_cpu(){ awk '{print ($14+$15)}' "$PROC/self/stat" 2>/dev/null || echo 0; }
OV_CPU0=$(read_self_cpu); OV_T0=$(date +%s.%N 2>/dev/null || date +%s)

# ---- REQUIRED evidence preflight (C-2 floor: vmstat, THP enabled, meminfo)
REQ_MISSING=""
[ -r "$PROC/vmstat" ]  || REQ_MISSING="$REQ_MISSING vmstat"
[ -r "$SYS/kernel/mm/transparent_hugepage/enabled" ] || REQ_MISSING="$REQ_MISSING thp_sysfs"
[ -r "$PROC/meminfo" ] || REQ_MISSING="$REQ_MISSING meminfo"
if [ -n "$REQ_MISSING" ]; then
  man verdict UNKNOWN
  man verdict_reason "REQUIRED evidence unreadable:$REQ_MISSING"
  say "VERDICT: UNKNOWN — REQUIRED evidence unreadable:$REQ_MISSING"
  echo "VERDICT: UNKNOWN — REQUIRED evidence unreadable:$REQ_MISSING (bundle: $BUNDLE)" >&2
  exit 2
fi

# =================== collectors (each records its evidence state) ===================
get(){ cat "$1" 2>/dev/null; }   # single-file read helper

collect_identity(){
  kv sys.kernel "$(uname -r 2>/dev/null || echo unknown)"
  kv sys.nproc "$(nproc 2>/dev/null || echo 0)"
  kv sys.uptime_s "$(awk '{print $1}' "$PROC/uptime" 2>/dev/null || echo 0)"
  if [ -r /etc/os-release ]; then kv sys.os "$(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-unknown}")"; st os_release available
  else st os_release unavailable; fi
  m=$(grep -m1 "model name" "$PROC/cpuinfo" 2>/dev/null | cut -d: -f2- | sed 's/^ //'); kv sys.cpu_model "${m:-unknown}"
  kv sys.cmdline "$(get "$PROC/cmdline" | tr '\0' ' ')"
  kv sys.glibc "$(ldd --version 2>/dev/null | head -1 | awk '{print $NF}')"
  v=$(systemd-detect-virt 2>/dev/null); kv sys.virt "${v:-unknown}"
  kv sys.slurm_job "${SLURM_JOB_ID:-none}"
  n=$(ls -d "$SYS"/devices/system/node/node[0-9]* 2>/dev/null | wc -l); kv sys.numa_nodes "$n"
}

collect_thp(){
  for f in enabled defrag; do
    v=$(get "$SYS/kernel/mm/transparent_hugepage/$f"); kv "thp.$f" "${v:-unreadable}"
  done
  st thp available
  for f in defrag pages_collapsed full_scans scan_sleep_millisecs; do
    v=$(get "$SYS/kernel/mm/transparent_hugepage/khugepaged/$f") && kv "thp.khugepaged.$f" "$v"
  done
  hs=$(ls -d "$SYS"/kernel/mm/hugepages/hugepages-* 2>/dev/null | sed 's|.*/hugepages-||' | tr '\n' ',' )
  if [ -n "$hs" ]; then kv thp.hugepage_sizes "${hs%,}"; st hugepages available; else st hugepages not_supported; fi
}

VMKEYS="compact_stall compact_fail compact_success compact_isolated thp_fault_alloc thp_fault_fallback thp_fault_fallback_charge thp_collapse_alloc thp_collapse_alloc_failed pgscan_direct pgscan_direct_throttle pgsteal_direct pgmajfault pgscan_kswapd pgsteal_kswapd oom_kill pswpin pswpout"
vmk(){ awk -v k="$1" '$1==k{print $2; f=1} END{if(!f) print ""}' "$PROC/vmstat" 2>/dev/null; }  # exact key (C-4/R-3)
collect_vmstat(){
  ok=0
  for k in $VMKEYS; do v=$(vmk "$k"); [ -n "$v" ] && { kv "vmstat.$k" "$v"; ok=1; }; done
  kv vmstat.allocstall_sum "$(awk '$1 ~ /^allocstall/{s+=$2} END{print s+0}' "$PROC/vmstat" 2>/dev/null)"
  [ "$ok" = 1 ] && st vmstat available || st vmstat parse_error
}

kthread_cpu(){ # BARE NUMERIC: name pid cpu_seconds (C-1 fix lineage); fixture-safe via PROC_ROOT
  for p in "$PROC"/[0-9]*; do
    read -r n < "$p/comm" 2>/dev/null || continue
    case "$n" in kcompactd*|khugepaged*)
      awk -v n="$n" -v pid="${p##*/}" -v clk="$CLK_TCK" \
        '{printf "%s %s %.2f\n", n, pid, ($14+$15)/clk}' "$p/stat" 2>/dev/null;; esac
  done; }
collect_kthreads(){
  any=0
  kthread_cpu | while read -r n p s; do echo "kthread.$n.pid=$p"; echo "kthread.$n.cpu_seconds=$s"; done >> "$SNAP"
  grep -q '^kthread\.' "$SNAP" && st kthreads available || st kthreads unavailable
  kv kthread.kcompactd_cpu_sum "$(kthread_cpu | awk '$1~/^kcompactd/{s+=$3} END{printf "%.2f", s+0}')"
  kv kthread.khugepaged_cpu_sum "$(kthread_cpu | awk '$1~/^khugepaged/{s+=$3} END{printf "%.2f", s+0}')"
}

collect_meminfo(){
  for k in MemTotal MemFree MemAvailable AnonHugePages Committed_AS CommitLimit SwapTotal SwapFree Dirty Writeback PageTables Slab SReclaimable SUnreclaim Active Inactive Mapped; do
    v=$(awk -v k="$k:" '$1==k{print $2}' "$PROC/meminfo" 2>/dev/null); [ -n "$v" ] && kv "meminfo.$k" "$v"
  done
  st meminfo available
}

collect_psi(){
  if [ -d "$PROC/pressure" ]; then
    for r in memory cpu io; do
      f="$PROC/pressure/$r"; [ -r "$f" ] || continue
      while read -r line; do
        typ=${line%% *}
        a10=$(echo "$line" | sed -n 's/.*avg10=\([0-9.]*\).*/\1/p')
        a60=$(echo "$line" | sed -n 's/.*avg60=\([0-9.]*\).*/\1/p')
        kv "psi.$r.$typ.avg10" "${a10:-}"; kv "psi.$r.$typ.avg60" "${a60:-}"
      done < "$f"
    done
    st psi available
  else st psi not_supported; fi
}

collect_cgroup(){
  d="$CGR"
  if [ -r "$d/memory.pressure" ] || [ -r "$d/memory.current" ]; then
    for f in memory.current memory.max memory.high cpu.max; do
      v=$(get "$d/$f"); [ -n "$v" ] && kv "cgroup.$f" "$(echo "$v" | tr '\n' ' ')"
    done
    [ -r "$d/memory.events" ] && awk '{print "cgroup.memory.events."$1"="$2}' "$d/memory.events" >> "$SNAP"
    st cgroup_v2 available
  else st cgroup_v2 not_supported; fi
}

collect_buddy(){
  if [ -r "$PROC/buddyinfo" ]; then
    awk '{node=$2; sub(",","",node); zone=$4; printf "buddy.node%s.%s=", node, zone; for(i=5;i<=NF;i++) printf "%s%s", $i, (i<NF?",":"\n")}' "$PROC/buddyinfo" >> "$SNAP"
    st buddyinfo available
  else st buddyinfo unavailable; fi
}

collect_ps(){ # redacted by default (C-1): other users -> hashed id, comm masked
  PSC="${PS_CMD_OVERRIDE:-ps -eo pid,user,vsz,rss,pcpu,comm --sort=-vsz}"
  $PSC 2>/dev/null | head -13 | awk -v me="$SELF_USER" -v raw="$RAW" 'NR==1{print "pid user vsz rss pcpu comm virt_res"; next}
    { u=$2; c=$6; if (raw!=1 && u!=me) { cmd="printf %s "u" | cksum"; cmd | getline h; close(cmd); split(h,a," "); u="u"a[1]; c="[other]" }
      r=($4>0)? sprintf("%.1f",$3/$4) : "inf";
      print $1" "u" "$3" "$4" "$5" "c" "r }' | while read -r line; do echo "ps.top=$line"; done >> "$SNAP"
  st ps_table available
}

collect_deepdive(){
  [ -z "$PID" ] && return 0
  kv "pid.$PID.owner" "${P_OWNER:-unknown}"
  if [ "$RAW" = 1 ] || [ "$P_OWNER" = "$SELF_USER" ]; then
    kv "pid.$PID.cmd" "$(tr '\0' ' ' < "$PROC/$PID/cmdline" 2>/dev/null)"
  else kv "pid.$PID.cmd" "[redacted]"; fi
  for k in VmSize VmRSS VmSwap Threads RssAnon RssFile RssShmem voluntary_ctxt_switches nonvoluntary_ctxt_switches; do
    v=$(awk -v k="$k:" '$1==k{print $2}' "$PROC/$PID/status" 2>/dev/null); [ -n "$v" ] && kv "pid.$PID.$k" "$v"
  done
  for k in Pss Private_Dirty AnonHugePages; do
    v=$(awk -v k="$k:" '$1==k{s+=$2} END{print s+0}' "$PROC/$PID/smaps_rollup" 2>/dev/null); kv "pid.$PID.smaps.$k" "${v:-0}"
  done
  kv "pid.$PID.wchan" "$(get "$PROC/$PID/wchan")"
  kv "pid.$PID.fd_count" "$(ls "$PROC/$PID/fd" 2>/dev/null | wc -l)"
  kv "pid.$PID.oom_score" "$(get "$PROC/$PID/oom_score")"
  # glibc arena fingerprint: ~64MiB-class anonymous mappings (Fabble5_6 / v2 §2)
  am=$(awk '$6=="" { split($1,r,"-"); sz=strtonum("0x"r[2])-strtonum("0x"r[1]); if (sz>=50*1048576 && sz<=70*1048576) c++ } END{print c+0}' "$PROC/$PID/maps" 2>/dev/null)
  kv "pid.$PID.arena_map_count" "${am:-0}"
  if [ -r "$PROC/$PID/io" ]; then awk -v P="$PID" '{gsub(":",""); print "pid."P".io."$1"="$2}' "$PROC/$PID/io" >> "$SNAP"; st pid_io available; else st pid_io permission_denied; fi
  st deepdive available
}

collect_datapath(){
  [ -z "$DATAPATH" ] && return 0
  if [ -e "$DATAPATH" ]; then
    df -PT "$DATAPATH" 2>/dev/null | awk 'NR==2{print "datapath.device="$1; print "datapath.fstype="$2}' >> "$SNAP"
    kv datapath.path "$DATAPATH"; st datapath available
  else kv datapath.path "$DATAPATH"; st datapath unavailable; fi
}

collect_allocenv(){
  kv env.MALLOC_ARENA_MAX "${MALLOC_ARENA_MAX:-unset}"
  kv env.LD_PRELOAD "${LD_PRELOAD:-unset}"
  kv env.GLIBC_TUNABLES "${GLIBC_TUNABLES:-unset}"
}

collect_anchors(){  # ground-truth channels: NEVER flat on a live host
  if [ -r "$PROC/loadavg" ]; then
    read -r l1 l5 l15 running _ < "$PROC/loadavg"
    kv anchor.loadavg1 "$l1"; kv anchor.loadavg5 "$l5"; kv anchor.loadavg15 "$l15"
    kv anchor.procs_running "${running%%/*}"
    st anchors available
  else st anchors unavailable; fi
  kv anchor.mem_available_kb "$(awk '$1=="MemAvailable:"{print $2}' "$PROC/meminfo" 2>/dev/null)"
}

collect_identity; collect_anchors; collect_thp; collect_vmstat; collect_kthreads; collect_meminfo
collect_psi; collect_cgroup; collect_buddy; collect_ps; collect_deepdive; collect_datapath; collect_allocenv

# =================== sampling mode ===================
if [ "$S_GIVEN" = 1 ]; then
  echo "ts,elapsed_s,compact_stall_total,compact_stall_per_s,compact_fail_total,compact_fail_per_s,thp_fault_alloc_total,thp_fault_alloc_per_s,thp_fault_fallback_total,thp_fault_fallback_per_s,thp_collapse_alloc_total,pgscan_direct_total,allocstall_total,allocstall_per_s,kcompactd_cpu_s_total,kcompactd_cpu_per_s,khugepaged_cpu_s_total,khugepaged_cpu_per_s,psi_mem_some_avg10,psi_mem_full_avg10,pswpin_total,pswpout_total,disk_read_sectors_total,sample_wall_s,overrun,loadavg1,cpu_busy_pct,mem_available_kb,ctxt_per_s,procs_running" > "$CSV"
  echo "ts,device,reads_completed,sectors_read,writes_completed,sectors_written,io_in_progress,io_time_ms" > "$DISKCSV"
  echo "ts,iface,rx_bytes,rx_packets,rx_errs,rx_drop,tx_bytes,tx_packets,tx_errs,tx_drop" > "$NETCSV"
  p_t=""; p_cs=""; p_cf=""; p_fa=""; p_fb=""; p_as=""; p_kc=""; p_kh=""
  p_cpu_idle=""; p_cpu_total=""; p_ctxt=""
  OVERRUNS=0
  # ---- v8 D1.6: start the process/user/rollup sampler (collector.py) ----
  SAMPLER_PID=""
  PROC_INTERVAL=${PROC_INTERVAL_OVERRIDE:-5}
  if [ "${DFX_PROCESS_SAMPLER:-on}" != "off" ] && command -v python3 >/dev/null 2>&1 && [ -f "$(dirname "$0")/collector.py" ]; then
    SARGS="-o $BUNDLE --run-id $RUN_ID --interval $PROC_INTERVAL --nall ${NALL:-20} --nuser ${NUSER:-20}"
    [ -n "$TARGET_PID_FILE" ] && SARGS="$SARGS --target-pid-file $TARGET_PID_FILE"
    [ -n "$JOB_CGROUP" ] && SARGS="$SARGS --job-cgroup $JOB_CGROUP"
    [ "$RAW" = 1 ] && SARGS="$SARGS --raw"
    # shellcheck disable=SC2086
    # -S: skip site-packages scan (collector is stdlib-only) - 10x faster start on slow FS
    python3 -S "$(dirname "$0")/collector.py" $SARGS >> "$BUNDLE/sampler.log" 2>&1 &
    SAMPLER_PID=$!
    st process_sampler available
    man process_interval_s "$PROC_INTERVAL"; man nall "${NALL:-20}"; man nuser "${NUSER:-20}"
  else
    st process_sampler unavailable
  fi
  # clean-stop (architect token 2026-07-17, "continue"): TERM/INT during a
  # bounded run finishes the current sample, stops the sampler, writes the
  # verdict and exits 0 - interruption leaves a VALID bundle (T-D11 class).
  STOP_REQ=0
  trap 'STOP_REQ=1' TERM INT
  i=1
  while [ "$i" -le "$NSAMPLES" ]; do
    [ "$STOP_REQ" = 1 ] && break
    # T-D11a guard: if the bundle directory vanishes mid-run (mv/rm/tmp-cleaner),
    # fail LOUDLY instead of appending forever into deleted files (2026-07-16 incident)
    [ -d "$BUNDLE" ] || { echo "FATAL: bundle directory vanished: $BUNDLE (sample $i/$NSAMPLES)" >&2; exit 1; }
    SW0=$(date +%s.%N 2>/dev/null || date +%s)
    cs=$(vmk compact_stall); cf=$(vmk compact_fail); fa=$(vmk thp_fault_alloc)
    fb=$(vmk thp_fault_fallback); ca=$(vmk thp_collapse_alloc); pd=$(vmk pgscan_direct)
    as=$(awk '$1 ~ /^allocstall/{s+=$2} END{print s+0}' "$PROC/vmstat" 2>/dev/null)
    kc=$(kthread_cpu | awk '$1~/^kcompactd/{s+=$3} END{printf "%.2f", s+0}')
    kh=$(kthread_cpu | awk '$1~/^khugepaged/{s+=$3} END{printf "%.2f", s+0}')
    pm_s=$(sed -n 's/^some.*avg10=\([0-9.]*\).*/\1/p' "$PROC/pressure/memory" 2>/dev/null | head -1)
    pm_f=$(sed -n 's/^full.*avg10=\([0-9.]*\).*/\1/p' "$PROC/pressure/memory" 2>/dev/null | head -1)
    si=$(vmk pswpin); so=$(vmk pswpout)
    dr=$(awk '{s+=$6} END{print s+0}' "$PROC/diskstats" 2>/dev/null)
    # ---- anchors (ground truth; cpu/ctxt need previous sample) ----
    read -r la1 _ _ runn _ < "$PROC/loadavg" 2>/dev/null || { la1=""; runn=""; }
    runn=${runn%%/*}
    mavail=$(awk '$1=="MemAvailable:"{print $2}' "$PROC/meminfo" 2>/dev/null)
    read -r _ c_user c_nice c_sys c_idle c_iow c_irq c_sirq c_steal _ < "$PROC/stat" 2>/dev/null || c_idle=""
    if [ -n "${c_idle:-}" ]; then
      cpu_total=$((c_user+c_nice+c_sys+c_idle+c_iow+c_irq+c_sirq+c_steal)); cpu_idle=$((c_idle+c_iow))
    else cpu_total=""; cpu_idle=""; fi
    ctxt=$(awk '$1=="ctxt"{print $2}' "$PROC/stat" 2>/dev/null)
    t=$(awk '{print $1}' "$PROC/uptime")
    if [ -n "$p_t" ]; then
      rates=$(awk -v t="$t" -v pt="$p_t" -v cs="${cs:-0}" -v pcs="${p_cs:-0}" -v cf="${cf:-0}" -v pcf="${p_cf:-0}" \
                  -v fa="${fa:-0}" -v pfa="${p_fa:-0}" -v fb="${fb:-0}" -v pfb="${p_fb:-0}" \
                  -v as="${as:-0}" -v pas="${p_as:-0}" -v kc="${kc:-0}" -v pkc="${p_kc:-0}" -v kh="${kh:-0}" -v pkh="${p_kh:-0}" \
        'BEGIN{dt=t-pt; bad=(dt<=0);
          if (bad) { printf "INVALID 0 0 0 0 0 0 0 0" }
          else printf "%.2f %.3f %.3f %.3f %.3f %.3f %.4f %.4f OK",
            dt,(cs-pcs)/dt,(cf-pcf)/dt,(fa-pfa)/dt,(fb-pfb)/dt,(as-pas)/dt,(kc-pkc)/dt,(kh-pkh)/dt}')
      set -f; set -- $rates; set +f
      cpu_pct=""; ctxt_ps=""
      if [ -n "${p_cpu_total:-}" ] && [ -n "${cpu_total:-}" ]; then
        cpu_pct=$(awk -v i="$cpu_idle" -v pi="$p_cpu_idle" -v t2="$cpu_total" -v pt2="$p_cpu_total"           'BEGIN{d=t2-pt2; if(d<=0){print ""} else printf "%.1f", 100.0*(1-(i-pi)/d)}')
      fi
      if [ -n "${p_ctxt:-}" ] && [ -n "${ctxt:-}" ]; then
        ctxt_ps=$(awk -v c="$ctxt" -v pc="$p_ctxt" -v t="$t" -v pt="$p_t"           'BEGIN{dt=t-pt; if(dt<=0){print ""} else printf "%.1f",(c-pc)/dt}')
      fi
      if [ "$1" = "INVALID" ]; then
        echo "$(date +%s),INVALID,${cs:-},,${cf:-},,${fa:-},,${fb:-},,${ca:-},${pd:-},${as:-},,${kc:-0},,${kh:-0},,${pm_s:-},${pm_f:-},${si:-},${so:-},${dr:-},,,${la1:-},,${mavail:-},,${runn:-}" >> "$CSV"
      else
        dt_s=$1; r_cs=$2; r_cf=$3; r_fa=$4; r_fb=$5; r_as=$6; r_kc=$7; r_kh=$8
        SW1=$(date +%s.%N 2>/dev/null || date +%s)
        swall=$(awk -v a="$SW0" -v b="$SW1" 'BEGIN{printf "%.3f", b-a}')
        ovr=$(awk -v w="$swall" -v iv="$INTERVAL" 'BEGIN{print (w>iv)?"1":"0"}')
        [ "$ovr" = 1 ] && OVERRUNS=$((OVERRUNS+1))
        echo "$(date +%s),$dt_s,${cs:-},$r_cs,${cf:-},$r_cf,${fa:-},$r_fa,${fb:-},$r_fb,${ca:-},${pd:-},${as:-},$r_as,${kc:-0},$r_kc,${kh:-0},$r_kh,${pm_s:-},${pm_f:-},${si:-},${so:-},${dr:-},$swall,$ovr,${la1:-},${cpu_pct:-},${mavail:-},${ctxt_ps:-},${runn:-}" >> "$CSV"
      fi
    else
      echo "$(date +%s),,${cs:-},,${cf:-},,${fa:-},,${fb:-},,${ca:-},${pd:-},${as:-},,${kc:-0},,${kh:-0},,${pm_s:-},${pm_f:-},${si:-},${so:-},${dr:-},,,${la1:-},,${mavail:-},,${runn:-}" >> "$CSV"
    fi
    p_t=$t; p_cs=$cs; p_cf=$cf; p_fa=$fa; p_fb=$fb; p_as=$as; p_kc=$kc; p_kh=$kh
    p_cpu_idle=$cpu_idle; p_cpu_total=$cpu_total; p_ctxt=$ctxt
    TSNOW=$(date +%s)
    awk -v ts="$TSNOW" 'NF>=13 {print ts","$3","$4","$6","$8","$10","$12","$13}' \
      "$PROC/diskstats" >> "$DISKCSV" 2>/dev/null || true
    awk -v ts="$TSNOW" 'NR>2 {gsub(":"," ",$0); print ts","$1","$2","$3","$4","$5","$9","$10","$11","$12}' \
      "$PROC/net/dev" >> "$NETCSV" 2>/dev/null || true
    echo "[dfx $(date -u +%H:%M:%SZ)] sample $i/$NSAMPLES wall=${swall:-?}s overrun=${ovr:-0} bundle=$(basename "$BUNDLE")"
    if [ "$STOP_REQ" = 0 ] && [ "$i" -lt "$NSAMPLES" ]; then
      sleep "$INTERVAL" & SLPID=$!; wait "$SLPID" 2>/dev/null   # interruptible sleep
    fi
    i=$((i+1))
  done
  trap - TERM INT
  man samples_taken "$((i-1))"
  [ "$STOP_REQ" = 1 ] && man stop_reason signal_clean_stop
  if [ -n "$SAMPLER_PID" ]; then
    # wait (bounded) for the first-scan readiness marker: a TERM during slow
    # interpreter startup would kill the sampler before any scan (alma2 race)
    w=0
    while [ ! -f "$BUNDLE/.sampler_ready" ] && [ "$w" -lt 24 ] && kill -0 "$SAMPLER_PID" 2>/dev/null; do
      sleep 0.25; w=$((w+1))
    done
    kill -TERM "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null
    if [ -f "$BUNDLE/.sampler_ready" ]; then man process_sampler_stopped clean
    else man process_sampler_stopped timeout_no_ready; fi
    rm -f "$BUNDLE/.sampler_ready"
  fi
  man samples "$NSAMPLES"; man sample_interval_s "$INTERVAL"; man sample_overruns "$OVERRUNS"
  or_rule=$(awk -v o="$OVERRUNS" -v n="$NSAMPLES" 'BEGIN{print (n>0 && o/n>0.10)?"WARN":"OK"}')
  man overrun_status "$or_rule"
fi

# =================== verdict engine (rule IDs; normative table -> D3, draft v0 pending C-9) ===================
RULES_FIRED=""; SEV=0   # 0 PASS, 1 WARN(config), 2 UNHEALTHY(historical/live)
fire(){ RULES_FIRED="$RULES_FIRED $1"; [ "$2" -gt "$SEV" ] && SEV=$2; kv "verdict.rule.$1" fired; }
EN=$(get "$SYS/kernel/mm/transparent_hugepage/enabled")
DF=$(get "$SYS/kernel/mm/transparent_hugepage/defrag")
case "$EN" in *"[always]"*) fire THP-01 1;; esac
case "$DF" in *"[always]"*|*"[defer+madvise]"*) fire THP-02 1;; esac
UPT=$(awk '{print $1}' "$PROC/uptime" 2>/dev/null || echo 0)
KCS=$(kthread_cpu | awk '$1~/^kcompactd/{s+=$3} END{printf "%.2f", s+0}')
# KC-01 (draft): uptime-normalized — >=300 kcompactd CPU seconds per uptime-day (C-7 lineage)
kcday=$(awk -v k="$KCS" -v u="$UPT" 'BEGIN{ if (u<=0) print 0; else printf "%.1f", k/(u/86400) }')
kv verdict.kcompactd_cpu_per_day "$kcday"
awk -v x="$kcday" 'BEGIN{exit !(x>=300)}' && fire KC-01 2
# PSI-01 (draft): memory full avg60 >= 5.0 => live stall pathology
PF60=$(sed -n 's/^full.*avg60=\([0-9.]*\).*/\1/p' "$PROC/pressure/memory" 2>/dev/null | head -1)
[ -n "${PF60:-}" ] && awk -v x="$PF60" 'BEGIN{exit !(x>=5.0)}' && fire PSI-01 2
case $SEV in 0) VERDICT=PASS;; 1) VERDICT=WARN;; 2) VERDICT=UNHEALTHY;; esac
man verdict "$VERDICT"; man verdict_rules "${RULES_FIRED:- none}"; man rule_table_version 1-draft

# =================== human report (rendering of the SAME values) ===================
say "dfx_host_diagnostics v$VERSION | host=$HOST | $UTC | bundle=$(basename "$BUNDLE")"
sec "VERDICT"
say "verdict: $VERDICT   rules fired:${RULES_FIRED:- none}   (rule table v1-draft; thresholds pending architect ratification per C-9)"
say "evidence classes: configuration risk (THP-xx) | historical (KC-01, uptime-normalized: ${kcday}s/day) | live (PSI-01: mem full avg60=${PF60:-n/a})"
sec "THP CONFIGURATION"
say "enabled: ${EN:-unreadable}"; say "defrag:  ${DF:-unreadable}"
sec "KERNEL COMPACTION/THP COUNTERS (exact-key)"
for k in $VMKEYS; do v=$(vmk "$k"); [ -n "$v" ] && say "$k $v"; done
sec "KERNEL THREAD ACCUMULATED CPU"
kthread_cpu >> "$REP"
sec "MEMORY / SWAP"
grep -E '^meminfo\.' "$SNAP" | sed 's/^meminfo\.//; s/=/ /' >> "$REP"
sec "PSI (pressure stall information)"
grep -E '^psi\.' "$SNAP" | sed 's/^psi\.//; s/=/ /' >> "$REP" || say "not supported on this kernel"
sec "TOP MEMORY PROCESSES (VIRT/RES = screening clue — inspect mappings and allocator context)"
grep -E '^ps\.top=' "$SNAP" | sed 's/^ps\.top=//' >> "$REP"
sec "EVIDENCE STATES"
grep -E '^state\.' "$SNAP" | sed 's/^state\.//; s/=/ /' >> "$REP"
[ "$S_GIVEN" = 1 ] && { sec "SAMPLING"; say "samples=$NSAMPLES interval=${INTERVAL}s overruns=$OVERRUNS -> $(basename "$CSV")"; }
say ""
say "Cross-host use: run on affected AND healthy servers and compare bundles (or use"
say "report_diagnostics.py). Remediation is an admin decision — this tool only diagnoses."

# ---- self-overhead close-out
OV_CPU1=$(read_self_cpu); OV_T1=$(date +%s.%N 2>/dev/null || date +%s)
man self_cpu_s "$(awk -v a="$OV_CPU0" -v b="$OV_CPU1" -v c="$CLK_TCK" 'BEGIN{printf "%.3f",(b-a)/c}')"
man self_wall_s "$(awk -v a="$OV_T0" -v b="$OV_T1" 'BEGIN{printf "%.3f", b-a}')"
man files "manifest.kv snapshot.kv report.txt$([ "$S_GIVEN" = 1 ] && echo ' host_samples.csv disk_samples.csv network_samples.csv')"
man json_conversion deferred_to_schema_py

echo "bundle: $BUNDLE"
exit 0
