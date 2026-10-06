#!/usr/bin/env bash
# dfx_build_duckdb_shards.sh
#
# Build one derived DuckDB shard beside each COMPLETE diagnostics bundle.
#
# Safe to source:
#   source scripts/dfx_build_duckdb_shards.sh
#   dfx_build_duckdb_shards --root "$ROOT" --host ccsub0002 -j 4
#
# Also executable as a CLI:
#   bash scripts/dfx_build_duckdb_shards.sh --root "$ROOT" --host ccsub0002 -j 4

_dfx_shards_usage() {
  cat <<'EOF'
Usage:
  dfx_build_duckdb_shards --root DIR [--host HOST] [-j N] [--dry-run]

Options:
  --root DIR        Production root containing <host>/YYYY/MM/DD/slot_*/.
                    Defaults to DFX_CRON_ROOT when set.
  --host HOST       Host subtree to process. Defaults to hostname -s.
  -j, --jobs N      Maximum concurrent shard builds. Default: DFX_SHARD_JOBS or 4.
  --query-tool FILE query_diagnostics.py to use. Normally auto-detected.
  --dry-run         Discover and report without creating DuckDB files.
  -h, --help        Show this help.

Shard naming:
  <slot>/<host_diag_bundle_name>.duckdb

Only status_*.kv records with state=COMPLETE are eligible.
Existing non-empty .duckdb shards are skipped.
RUNNING and other non-COMPLETE states are never built.
EOF
}

_dfx_shards_value() {
  sed -n "s/^$2=//p" "$1" | tail -1
}

_dfx_shards_build_one() {
  local bundle="$1"
  local query_tool="$2"
  local python_cmd="$3"
  local result_file="$4"

  local slot name db lock log rc
  slot=$(dirname "$bundle")
  name=$(basename "$bundle")
  db="$slot/${name}.duckdb"
  lock="${db}.build.lock"
  log="${db}.build.tmp.$$"

  if [ -s "$db" ]; then
    printf '%s\n' "EXISTS" > "$result_file"
    printf 'EXISTS  %s\n' "$db"
    return 0
  fi

  if [ -e "$db" ] && [ ! -s "$db" ]; then
    printf '%s\n' "BROKEN_EXISTING" > "$result_file"
    printf 'ERROR   zero-size existing shard: %s\n' "$db" >&2
    return 1
  fi

  if ! mkdir "$lock" 2>/dev/null; then
    if [ -s "$db" ]; then
      printf '%s\n' "EXISTS" > "$result_file"
      printf 'EXISTS  %s\n' "$db"
      return 0
    fi
    printf '%s\n' "LOCKED" > "$result_file"
    printf 'LOCKED  %s\n' "$db"
    return 1
  fi

  if "$python_cmd" "$query_tool" \
      build-db \
      --bundle "$bundle" \
      --db "$db" \
      >"$log" 2>&1; then
    rm -f "$log"
    rmdir "$lock" 2>/dev/null || true
    printf '%s\n' "BUILT" > "$result_file"
    printf 'BUILT   %s\n' "$db"
    return 0
  fi

  rc=$?
  mv -f "$log" "${db}.build_failed.log" 2>/dev/null || true
  rmdir "$lock" 2>/dev/null || true
  printf '%s\n' "FAILED" > "$result_file"
  printf 'FAILED  %s rc=%s log=%s\n' "$db" "$rc" "${db}.build_failed.log" >&2
  return "$rc"
}

dfx_build_duckdb_shards() {
  local root="${DFX_CRON_ROOT:-}"
  local host
  host=$(hostname -s)
  local jobs="${DFX_SHARD_JOBS:-4}"
  local query_tool=""
  local dry_run=0
  local python_cmd="${DFX_PYTHON:-python3}"

  while [ "$#" -gt 0 ]; do
    case "$1" in
      --root)
        [ "$#" -ge 2 ] || { echo "ERROR: --root requires a value" >&2; return 2; }
        root="$2"
        shift 2
        ;;
      --host)
        [ "$#" -ge 2 ] || { echo "ERROR: --host requires a value" >&2; return 2; }
        host="$2"
        shift 2
        ;;
      -j|--jobs)
        [ "$#" -ge 2 ] || { echo "ERROR: $1 requires a value" >&2; return 2; }
        jobs="$2"
        shift 2
        ;;
      --query-tool)
        [ "$#" -ge 2 ] || { echo "ERROR: --query-tool requires a value" >&2; return 2; }
        query_tool="$2"
        shift 2
        ;;
      --dry-run)
        dry_run=1
        shift
        ;;
      -h|--help)
        _dfx_shards_usage
        return 0
        ;;
      *)
        echo "ERROR: unknown argument: $1" >&2
        _dfx_shards_usage >&2
        return 2
        ;;
    esac
  done

  if [ -z "$root" ]; then
    echo "ERROR: --root or DFX_CRON_ROOT is required" >&2
    return 2
  fi

  case "$jobs" in
    ''|*[!0-9]*)
      echo "ERROR: jobs must be a positive integer: $jobs" >&2
      return 2
      ;;
  esac
  if [ "$jobs" -lt 1 ]; then
    echo "ERROR: jobs must be >= 1" >&2
    return 2
  fi

  if [ -z "$query_tool" ]; then
    if [ -n "${DIAGNOSTICS_CODE:-}" ]; then
      query_tool="$DIAGNOSTICS_CODE/query_diagnostics.py"
    elif [ -n "${O2DPG:-}" ]; then
      query_tool="$O2DPG/UTILS/dfextensions/diagnostics/query_diagnostics.py"
    else
      echo "ERROR: cannot locate query_diagnostics.py; set DIAGNOSTICS_CODE/O2DPG or use --query-tool" >&2
      return 2
    fi
  fi

  if [ ! -f "$query_tool" ]; then
    echo "ERROR: query tool not found: $query_tool" >&2
    return 2
  fi
  if ! command -v "$python_cmd" >/dev/null 2>&1; then
    echo "ERROR: Python command not found: $python_cmd" >&2
    return 2
  fi

  local host_root="$root/$host"
  if [ ! -d "$host_root" ]; then
    echo "ERROR: host production root not found: $host_root" >&2
    return 2
  fi

  local tmpdir
  tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/dfx_shards.XXXXXX") || return 2

  local status state bundle
  local status_count=0
  local complete_statuses=0
  local running_statuses=0
  local other_statuses=0
  local missing_bundle_statuses=0
  local existing=0
  local broken_existing=0
  local candidate_count=0

  while IFS= read -r status; do
    [ -n "$status" ] || continue
    status_count=$((status_count + 1))
    state=$(_dfx_shards_value "$status" state)

    case "$state" in
      COMPLETE)
        complete_statuses=$((complete_statuses + 1))
        bundle=$(_dfx_shards_value "$status" bundle)
        if [ -z "$bundle" ] || [ ! -d "$bundle" ]; then
          missing_bundle_statuses=$((missing_bundle_statuses + 1))
          printf 'MISSING bundle from COMPLETE status: %s\n' "$status" >&2
          continue
        fi
        printf '%s\n' "$bundle" >> "$tmpdir/complete_bundles.raw"
        ;;
      RUNNING)
        running_statuses=$((running_statuses + 1))
        ;;
      *)
        other_statuses=$((other_statuses + 1))
        ;;
    esac
  done < <(find "$host_root" -type f -name 'status_*.kv' -print | sort)

  if [ -f "$tmpdir/complete_bundles.raw" ]; then
    sort -u "$tmpdir/complete_bundles.raw" > "$tmpdir/complete_bundles"
  else
    : > "$tmpdir/complete_bundles"
  fi

  : > "$tmpdir/tasks"

  while IFS= read -r bundle; do
    [ -n "$bundle" ] || continue
    local slot name db
    slot=$(dirname "$bundle")
    name=$(basename "$bundle")
    db="$slot/${name}.duckdb"

    if [ -s "$db" ]; then
      existing=$((existing + 1))
      continue
    fi
    if [ -e "$db" ] && [ ! -s "$db" ]; then
      broken_existing=$((broken_existing + 1))
      printf 'BROKEN existing zero-size shard: %s\n' "$db" >&2
      continue
    fi

    candidate_count=$((candidate_count + 1))
    printf '%s\t%s\n' "$candidate_count" "$bundle" >> "$tmpdir/tasks"
  done < "$tmpdir/complete_bundles"

  echo "root=$root"
  echo "host=$host"
  echo "query_tool=$query_tool"
  echo "jobs=$jobs"
  echo "status_records=$status_count"
  echo "complete_statuses=$complete_statuses"
  echo "running_statuses=$running_statuses"
  echo "other_statuses=$other_statuses"
  echo "existing=$existing"
  echo "missing_to_build=$candidate_count"

  if [ "$dry_run" -eq 1 ]; then
    while IFS=$'\t' read -r _ bundle; do
      [ -n "$bundle" ] || continue
      local slot name
      slot=$(dirname "$bundle")
      name=$(basename "$bundle")
      printf 'WOULD_BUILD %s/%s.duckdb\n' "$slot" "$name"
    done < "$tmpdir/tasks"

    echo "SUMMARY: complete=$complete_statuses existing=$existing would_build=$candidate_count" \
         "running=$running_statuses other=$other_statuses missing_bundle=$missing_bundle_statuses" \
         "broken_existing=$broken_existing"
    rm -rf "$tmpdir"

    if [ "$missing_bundle_statuses" -ne 0 ] || [ "$broken_existing" -ne 0 ]; then
      return 1
    fi
    return 0
  fi

  local -a pids=()
  local idx result_file
  local batch_count=0

  while IFS=$'\t' read -r idx bundle; do
    [ -n "$idx" ] || continue
    result_file="$tmpdir/result.$idx"

    _dfx_shards_build_one \
      "$bundle" \
      "$query_tool" \
      "$python_cmd" \
      "$result_file" &

    pids+=("$!")
    batch_count=$((batch_count + 1))

    if [ "$batch_count" -ge "$jobs" ]; then
      local pid
      for pid in "${pids[@]}"; do
        wait "$pid" || true
      done
      pids=()
      batch_count=0
    fi
  done < "$tmpdir/tasks"

  if [ "${#pids[@]}" -gt 0 ]; then
    local pid
    for pid in "${pids[@]}"; do
      wait "$pid" || true
    done
  fi

  local built=0
  local failed=0
  local locked=0
  local raced_existing=0
  local result

  for result_file in "$tmpdir"/result.*; do
    [ -f "$result_file" ] || continue
    result=$(cat "$result_file")
    case "$result" in
      BUILT) built=$((built + 1)) ;;
      EXISTS) raced_existing=$((raced_existing + 1)) ;;
      LOCKED) locked=$((locked + 1)) ;;
      FAILED|BROKEN_EXISTING) failed=$((failed + 1)) ;;
      *) failed=$((failed + 1)) ;;
    esac
  done

  existing=$((existing + raced_existing))

  echo "SUMMARY: complete=$complete_statuses existing=$existing built=$built failed=$failed" \
       "locked=$locked running=$running_statuses other=$other_statuses" \
       "missing_bundle=$missing_bundle_statuses broken_existing=$broken_existing"

  rm -rf "$tmpdir"

  if [ "$failed" -ne 0 ] || [ "$locked" -ne 0 ] || \
     [ "$missing_bundle_statuses" -ne 0 ] || [ "$broken_existing" -ne 0 ]; then
    return 1
  fi
  return 0
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  dfx_build_duckdb_shards "$@"
fi
