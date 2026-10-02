#!/usr/bin/env bash
# PHASE_13_76_DIAG Regime-1 production prototype.
# Host-native, bounded, non-overlapping historical monitoring.
set -u
umask 077

usage() {
  cat >&2 <<'USAGE'
usage: dfx_cron_host_monitor.sh [options]

Options:
  --interval SEC     Host sample interval (default: 10)
  --duration SEC     Runtime used to derive -n (default: 21300 = 5h55m)
  --root DIR         Production monitoring root
  --dry-run          Print resolved slot/command; do not collect
  --status           Print latest host status
  -h, --help         Show help

Environment:
  O2DPG                         default: $HOME/O2DPG
  DIAGNOSTICS_CODE              default: $O2DPG/UTILS/dfextensions/diagnostics
  DFX_CRON_ROOT                 production data root; alternative to --root
  DFX_CRON_PROCESS_INTERVAL     process interval; default = host interval
  DFX_EXPECT_HOST               optional deployment guard; refuse another host
  DFX_HOST_OVERRIDE             host override for tests only
USAGE
}

die() { printf 'ERROR: %s\n' "$*" >&2; return 2; }

is_pos_int() {
  case "$1" in ''|*[!0-9]*|0) return 1 ;; *) return 0 ;; esac
}

INTERVAL="${DFX_CRON_INTERVAL:-10}"
DURATION="${DFX_CRON_DURATION:-21300}"
ROOT_OVERRIDE=""
MODE="run"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --interval) [ "$#" -ge 2 ] || { usage; die "--interval requires a value"; exit $?; }; INTERVAL="$2"; shift 2 ;;
    --duration) [ "$#" -ge 2 ] || { usage; die "--duration requires a value"; exit $?; }; DURATION="$2"; shift 2 ;;
    --root) [ "$#" -ge 2 ] || { usage; die "--root requires a value"; exit $?; }; ROOT_OVERRIDE="$2"; shift 2 ;;
    --dry-run) MODE="dry-run"; shift ;;
    --status) MODE="status"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown argument: $1"; exit $? ;;
  esac
done

is_pos_int "$INTERVAL" || { die "--interval must be a positive integer"; exit $?; }
is_pos_int "$DURATION" || { die "--duration must be a positive integer"; exit $?; }

O2DPG="${O2DPG:-$HOME/O2DPG}"
DIAGNOSTICS_CODE="${DIAGNOSTICS_CODE:-$O2DPG/UTILS/dfextensions/diagnostics}"

COLLECTOR="$DIAGNOSTICS_CODE/dfx_host_diagnostics.sh"
[ -r "$COLLECTOR" ] || { die "collector not readable: $COLLECTOR"; exit $?; }

HOST="${DFX_HOST_OVERRIDE:-$(hostname -s 2>/dev/null || hostname)}"
HOST="${HOST%%.*}"

# Generic launcher: site/host policy belongs to deployment configuration.
if [ -n "${DFX_EXPECT_HOST:-}" ] && [ "$HOST" != "$DFX_EXPECT_HOST" ]; then
  die "host mismatch: actual '$HOST', expected '$DFX_EXPECT_HOST'"
  exit $?
fi

if [ -n "$ROOT_OVERRIDE" ]; then
  ROOT="$ROOT_OVERRIDE"
elif [ -n "${DFX_CRON_ROOT:-}" ]; then
  ROOT="$DFX_CRON_ROOT"
elif [ -n "${DIAGNOSTICS_DATA:-}" ]; then
  ROOT="$DIAGNOSTICS_DATA/gsi_monitoring/production"
else
  die "production data root is not set; use --root DIR or DFX_CRON_ROOT"
  exit $?
fi
HOST_ROOT="$ROOT/$HOST"
YEAR="$(date '+%Y')"; MONTH="$(date '+%m')"; DAY="$(date '+%d')"; HOUR="$(date '+%H')"
SLOT_NUM=$((10#$HOUR / 6 * 6)); printf -v SLOT_HOUR '%02d' "$SLOT_NUM"
SLOT_DIR="$HOST_ROOT/$YEAR/$MONTH/$DAY/slot_$SLOT_HOUR"
LOCK_DIR="$HOST_ROOT/.collector.lock"
HOST_STATUS="$HOST_ROOT/status.latest.kv"

SAMPLES=$((DURATION / INTERVAL))
[ "$SAMPLES" -ge 2 ] || { die "duration/interval must yield at least 2 samples"; exit $?; }
PROC_INTERVAL="${DFX_CRON_PROCESS_INTERVAL:-$INTERVAL}"
is_pos_int "$PROC_INTERVAL" || { die "DFX_CRON_PROCESS_INTERVAL must be a positive integer"; exit $?; }

if [ "$MODE" = "status" ]; then
  if [ -f "$HOST_STATUS" ]; then cat "$HOST_STATUS"; else printf 'state=NO_STATUS\nhost=%s\nstatus_file=%s\n' "$HOST" "$HOST_STATUS"; fi
  exit 0
fi

STAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
RUN_LOG="$SLOT_DIR/collector_${STAMP}.log"
SLOT_STATUS="$SLOT_DIR/status_${STAMP}.kv"
CRON_LOG="$HOST_ROOT/cron.log"

if [ "$MODE" = "dry-run" ]; then
  cat <<EOF2
mode=dry-run
host=$HOST
root=$ROOT
slot_dir=$SLOT_DIR
interval_s=$INTERVAL
process_interval_s=$PROC_INTERVAL
duration_s=$DURATION
samples=$SAMPLES
collector=$COLLECTOR
command=PROC_INTERVAL_OVERRIDE=$PROC_INTERVAL bash $COLLECTOR -o $SLOT_DIR -s $INTERVAL -n $SAMPLES
EOF2
  exit 0
fi

mkdir -p "$SLOT_DIR" "$HOST_ROOT"
log() { local m="[$(date '+%F %T')] [dfx-cron] $*"; printf '%s\n' "$m"; printf '%s\n' "$m" >> "$CRON_LOG"; }

COLLECTOR_PID=""; STARTED_UTC=""; LOCK_OWNED=0; STOP_REQUESTED=0

write_status() {
  local state="$1" rc="${2:-}" bundle="${3:-}" verdict="" samples_taken=""
  if [ -n "$bundle" ] && [ -f "$bundle/manifest.kv" ]; then
    verdict="$(sed -n 's/^verdict=//p' "$bundle/manifest.kv" | tail -1)"
    samples_taken="$(sed -n 's/^samples_taken=//p' "$bundle/manifest.kv" | tail -1)"
  fi
  local tmp="$SLOT_STATUS.tmp.$$"
  cat > "$tmp" <<EOF2
state=$state
host=$HOST
slot=$YEAR-$MONTH-${DAY}T${SLOT_HOUR}
slot_dir=$SLOT_DIR
launcher_pid=$$
collector_pid=$COLLECTOR_PID
interval_s=$INTERVAL
process_interval_s=$PROC_INTERVAL
duration_s=$DURATION
samples_requested=$SAMPLES
samples_taken=$samples_taken
collector_rc=$rc
verdict=$verdict
bundle=$bundle
started_utc=$STARTED_UTC
updated_utc=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
log=$RUN_LOG
EOF2
  mv "$tmp" "$SLOT_STATUS"
  cp -p "$SLOT_STATUS" "$HOST_STATUS"
}

acquire_lock() {
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"; printf '%s\n' "$STAMP" > "$LOCK_DIR/started_utc"; return 0
  fi
  local old_pid=""; [ -r "$LOCK_DIR/pid" ] && old_pid="$(cat "$LOCK_DIR/pid" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    log "SKIPPED overlap: launcher PID $old_pid still owns $LOCK_DIR"
    STARTED_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"; write_status "SKIPPED_OVERLAP" "" ""; return 1
  fi
  log "removing stale lock: $LOCK_DIR old_pid=${old_pid:-unknown}"; rm -rf "$LOCK_DIR"
  if mkdir "$LOCK_DIR" 2>/dev/null; then
    printf '%s\n' "$$" > "$LOCK_DIR/pid"; printf '%s\n' "$STAMP" > "$LOCK_DIR/started_utc"; return 0
  fi
  log "SKIPPED overlap: another launcher acquired lock during stale-lock recovery"
  STARTED_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"; write_status "SKIPPED_OVERLAP" "" ""; return 1
}

release_lock() { if [ "$LOCK_OWNED" -eq 1 ]; then rm -rf "$LOCK_DIR"; LOCK_OWNED=0; fi; }
request_stop() {
  STOP_REQUESTED=1
  if [ -n "$COLLECTOR_PID" ] && kill -0 "$COLLECTOR_PID" 2>/dev/null; then
    log "signal received; requesting clean collector stop pid=$COLLECTOR_PID"; kill -TERM "$COLLECTOR_PID" 2>/dev/null || true
  fi
}
trap release_lock EXIT
trap request_stop TERM INT HUP

if ! acquire_lock; then exit 0; fi
LOCK_OWNED=1
STARTED_UTC="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
log "START host=$HOST slot=$SLOT_HOUR interval=${INTERVAL}s process_interval=${PROC_INTERVAL}s samples=$SAMPLES"
log "data=$SLOT_DIR"

PROC_INTERVAL_OVERRIDE="$PROC_INTERVAL" bash "$COLLECTOR" -o "$SLOT_DIR" -s "$INTERVAL" -n "$SAMPLES" > "$RUN_LOG" 2>&1 &
COLLECTOR_PID=$!
write_status "RUNNING" "" ""
wait "$COLLECTOR_PID"; RC=$?
if kill -0 "$COLLECTOR_PID" 2>/dev/null; then wait "$COLLECTOR_PID"; RC=$?; fi

BUNDLE="$(find "$SLOT_DIR" -maxdepth 1 -type d -name 'host_diag_*' 2>/dev/null | sort | tail -1)"
case "$RC" in 0) STATE="COMPLETE" ;; 2) STATE="UNKNOWN" ;; *) STATE="FAILED" ;; esac
[ "$STOP_REQUESTED" -eq 1 ] && [ "$RC" -eq 0 ] && STATE="CLEAN_STOP"
write_status "$STATE" "$RC" "$BUNDLE"
log "DONE state=$STATE rc=$RC bundle=${BUNDLE:-NONE}"
exit "$RC"
