#!/usr/bin/env bash
set -u
SCRIPT="${1:-$(cd "$(dirname "$0")/.." && pwd)/scripts/dfx_cron_host_monitor.sh}"
TMP="$(mktemp -d)"; PASS=0; FAIL=0; P1=""
cleanup(){ [ -n "$P1" ] && kill -TERM "$P1" 2>/dev/null || true; rm -rf "$TMP"; }
trap cleanup EXIT
ok(){ PASS=$((PASS+1)); printf 'PASS  %s\n' "$1"; }
bad(){ FAIL=$((FAIL+1)); printf 'FAIL  %s\n' "$1"; }
O2="$TMP/O2DPG"; CODE="$O2/UTILS/dfextensions/diagnostics"; DATA="$TMP/data"
mkdir -p "$CODE/scripts" "$DATA"
cat > "$CODE/scripts/env.sh" <<EOF2
export DIAGNOSTICS_CODE="$CODE"
export DIAGNOSTICS_DATA="$DATA"
EOF2
cat > "$CODE/dfx_host_diagnostics.sh" <<'EOF2'
#!/usr/bin/env bash
set -u
OUT="."; N=2
while [ "$#" -gt 0 ]; do case "$1" in -o) OUT="$2"; shift 2 ;; -s) shift 2 ;; -n) N="$2"; shift 2 ;; *) shift ;; esac; done
B="$OUT/host_diag_test_$$"; mkdir -p "$B"
trap 'printf "samples_taken=1\nverdict=PASS\n" > "$B/manifest.kv"; exit 0' TERM
sleep 2
printf 'samples_taken=%s\nverdict=PASS\n' "$N" > "$B/manifest.kv"
EOF2
chmod 755 "$CODE/dfx_host_diagnostics.sh"
DRY="$(O2DPG="$O2" DFX_HOST_OVERRIDE=ccsub0001 "$SCRIPT" --root "$DATA/prod" --interval 10 --duration 21300 --dry-run)"
printf '%s\n' "$DRY" | grep -q '^samples=2130$' && ok "5h55/10s -> 2130 requested samples" || bad "sample calculation"
printf '%s\n' "$DRY" | grep -Eq '/ccsub0001/[0-9]{4}/[0-9]{2}/[0-9]{2}/slot_(00|06|12|18)$' && ok "host/date/slot path" || bad "slot path"
printf '%s\n' "$DRY" | grep -q '^process_interval_s=10$' && ok "process cadence defaults to host cadence" || bad "process cadence"

GENERIC="$(O2DPG="$O2" DFX_HOST_OVERRIDE=generic-host "$SCRIPT" --root "$DATA/prod_generic" --interval 10 --duration 60 --dry-run)"
printf '%s\n' "$GENERIC" | grep -q '^host=generic-host$' && ok "generic launcher accepts arbitrary host" || bad "generic host support"

if O2DPG="$O2" DFX_HOST_OVERRIDE=wrong-host DFX_EXPECT_HOST=expected-host "$SCRIPT" --root "$DATA/prod_guard" --dry-run >/dev/null 2>&1; then
  bad "optional deployment host guard"
else
  ok "optional deployment host guard"
fi

MINIMAL="$(
  env -i \
    HOME="$TMP/home" \
    USER=tester \
    LOGNAME=tester \
    SHELL=/bin/bash \
    PATH=/usr/bin:/bin \
    O2DPG="$O2" \
    DFX_HOST_OVERRIDE=minimal-host \
    /bin/bash "$SCRIPT" \
      --root "$DATA/prod_minimal" \
      --interval 10 \
      --duration 60 \
      --dry-run
)"
printf '%s\n' "$MINIMAL" | grep -q '^host=minimal-host$' \
  && ok "cron-like minimal environment works without NOTES/NOTESData" \
  || bad "minimal cron environment"
O2DPG="$O2" DFX_HOST_OVERRIDE=ccsub0001 "$SCRIPT" --root "$DATA/prod" --interval 1 --duration 10 >/dev/null 2>&1 & P1=$!
sleep 0.3
SECOND="$(O2DPG="$O2" DFX_HOST_OVERRIDE=ccsub0001 "$SCRIPT" --root "$DATA/prod" --interval 1 --duration 10 2>&1)"
printf '%s\n' "$SECOND" | grep -q 'SKIPPED overlap' && ok "live overlap refused" || bad "overlap guard"
kill -TERM "$P1" 2>/dev/null || true; wait "$P1" 2>/dev/null || true; P1=""
STATUS="$(O2DPG="$O2" DFX_HOST_OVERRIDE=ccsub0001 "$SCRIPT" --root "$DATA/prod" --status)"
printf '%s\n' "$STATUS" | grep -Eq '^state=(CLEAN_STOP|COMPLETE)$' && ok "final status recorded" || bad "final status"
printf '\nRESULT: PASS=%d FAIL=%d\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
