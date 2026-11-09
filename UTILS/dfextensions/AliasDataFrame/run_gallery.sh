#!/usr/bin/env bash
# AliasDataFrame real-data gallery runner — PHASE_13_105 S3-0.
#
#   ./run_gallery.sh                     same as --all: every target, then the audit, one ZIP (the gate)
#   ./run_gallery.sh --all
#   ./run_gallery.sh --new               = --target pytest --target gallery --target 13_105 (daily work, no old harness)
#   ./run_gallery.sh --target 13_105     one or more targets (repeatable); the audit always runs; result = SUBSET
#   ./run_gallery.sh --target 13_105 --case <CASE-ID>
#   ./run_gallery.sh --target ?          list the targets
#
# Targets
#   pytest    every synthetic test of the 13.101 and 13.105 files, including the tests of this runner
#   gallery   the 59-page readable gallery, time_series_draw.run_all_pdf
#   13_101    the OLD harness: 23 historical cases + 3 ADF101 cases (its own process until S5)
#   13_105    the NEW declarative test: all migrated cases, strict
#   audit     evidence audit over whatever ran; cannot be deselected
#
# The data is loaded ONCE for the gallery and 13_105 targets: both run inside one Python process,
# time_series_draw_invariancetest.py, which loops over the requested targets. The old harness loads
# its own copy because its code is frozen until S5.
#
# Default fixture: the first 1,000,000 ROOT entries, no post-read sampling (S0 input identity).
# Every run writes test_logs/gallery_<timestamp>_<unique>/ and one flat reviewer ZIP.
# This script only selects targets and packages evidence; no scientific truth lives here.

set -o pipefail

ALL_TARGETS=(pytest gallery 13_101 13_105)

usage() {
    cat <<'HELP'
AliasDataFrame gallery runner (PHASE_13_105 S3-0)

  ./run_gallery.sh                       same as --all
  ./run_gallery.sh --all                 every target, audit with migration crosswalk, one ZIP — the gate
  ./run_gallery.sh --new                 pytest + gallery + 13_105: daily work, without the old harness
  ./run_gallery.sh --target NAME ...     only these targets (repeatable); audit always; labelled SUBSET
  ./run_gallery.sh --target 13_105 --case CASE-ID
  ./run_gallery.sh --target ?            list the targets
  ./run_gallery.sh --help

Targets: pytest  gallery  13_101  13_105   (audit always runs)
HELP
}

list_targets() {
    cat <<'LIST'
TARGET    WHAT IT RUNS                                                    IN --all
pytest    synthetic tests: 13.101 files (incl. runner tests) + 13.105 file   yes
gallery   the 59-page readable gallery (time_series_draw.run_all_pdf)     yes
13_101    old harness: 23 historical Stage-A cases + 3 ADF101 cases       yes, until S5
13_105    new declarative test: all migrated cases, strict                yes
audit     evidence audit over what ran; with --all also the crosswalk     always
LIST
}

# A stage runs the original code path with its own log; a failed stage never prevents packaging.
run_stage() {
    local label="$1"
    local logfile="$2"
    shift 2
    printf '\n[%s] RUN %s\n' "$(date '+%F %T')" "$label"
    if "$@" 2>&1 | tee "$logfile"; then
        printf '%s\tPASS\n' "$label" >> "$results_file"
        printf '[%s] PASS %s\n' "$(date '+%F %T')" "$label"
    else
        printf '%s\tFAIL\n' "$label" >> "$results_file"
        printf '[%s] FAIL %s (see %s)\n' "$(date '+%F %T')" "$label" "$logfile" >&2
        failures=$((failures + 1))
    fi
}

has_target() {
    local t
    for t in "${targets[@]}"; do
        if [[ "$t" == "$1" ]]; then return 0; fi
    done
    return 1
}

# The audit: counts for what ran; with --all, the migration crosswalk between old and new.
check_outputs() {
    python - "$run_dir" "$mode" "${targets[@]}" <<'PY'
import hashlib, json, re, sys
sys.path.insert(0, "examples/time_series")
folder, mode, targets = sys.argv[1], sys.argv[2], set(sys.argv[3:])
from pathlib import Path
failures = []

def pdf_pages(name, expected):
    file = Path(folder) / name
    if not file.is_file():
        failures.append(f"{name}: missing"); return
    data = file.read_bytes()
    if not data.startswith(b"%PDF-"):
        failures.append(f"{name}: missing PDF header"); return
    pages = len(re.findall(rb"/Type\s*/Page\b", data))
    print(f"PDF {name}: {pages} page objects, {len(data)} bytes, SHA256={hashlib.sha256(data).hexdigest()}")
    if expected is not None and pages != expected:
        failures.append(f"{name}: expected {expected} page objects, found {pages}")

def load(name):
    file = Path(folder) / name
    if not file.is_file():
        failures.append(f"{name}: missing"); return None
    try:
        content = json.loads(file.read_text())
    except (OSError, ValueError) as exc:
        failures.append(f"{name}: unreadable: {exc}"); return None
    print(f"MANIFEST {name}: SHA256={hashlib.sha256(file.read_bytes()).hexdigest()}")
    return content

old_ids = []
if "13_101" in targets:
    for name, expected in (("historical_stageA_manifest.json", 23), ("adf101_manifest.json", 3)):
        m = load(name)
        if m is None:
            continue
        cases = m.get("cases", []); rec = m.get("reconciliation", {})
        old_ids += [c.get("case_id") for c in cases]
        if len(cases) != expected or rec.get("n_declared") != expected or rec.get("n_results") != expected:
            failures.append(f"{name}: expected {expected} declared/executed cases")
        if rec.get("gating") or any(c.get("status") != "PASS" for c in cases):
            failures.append(f"{name}: not all cases PASS")
        if name == "historical_stageA_manifest.json":
            prov = m.get("provenance", {})
            if (prov.get("sample_mode") != "FULL" or prov.get("bounded_fixture_kind") != "FIRST_N_NO_FRACTION"
                    or prov.get("entry_start") != 0 or prov.get("entry_stop") != 1000000):
                failures.append(f"{name}: bounded unsampled provenance mismatch")
    pdf_pages("historical_stageA.pdf", None)

new = None
if "13_105" in targets or "gallery" in targets:
    new = load("new_13_105_manifest.json")
if new is not None:
    prov, rec = new.get("provenance", {}), new.get("reconciliation", {})
    if rec.get("exit_code") != 0:
        failures.append("new_13_105_manifest.json: strict gate failed (nonzero code)")
    if "13_105" in targets:
        if rec.get("missing_mandatory"):
            failures.append(f"new manifest: mandatory cases missing {rec['missing_mandatory']}")
        if any(c.get("status") != "PASS" for c in new.get("cases", [])):
            failures.append("new manifest: not all cases PASS")
        versus = prov.get("input_identity_versus_s0") or {}
        if versus.get("ok") is not True:
            failures.append(f"new manifest: input identity does not match S0: {versus}")
        if not all((prov.get("loaded_equals_raw") or {}).values()):
            failures.append("new manifest: loaded columns differ from the ROOT values")
        pdf_pages("new_13_105.pdf", None)
    if "gallery" in targets:
        g = (new.get("targets") or {}).get("gallery") or {}
        if not g.get("ok"):
            failures.append("gallery target did not report OK")
        pdf_pages("original_gallery.pdf", 59)

if mode == "all" and new is not None and old_ids:
    import time_series_draw_invariancetest as T
    x = T.migration_crosswalk(old_ids, new)
    print(f"CROSSWALK old {len(old_ids)} = migrated {len(x['migrated'])} + still old {len(x['still_old'])}"
          f" | in both {x['in_both']} | in neither {x['in_neither']} | migrated but not PASS {x['migrated_but_not_pass']}")
    if not x["ok"]:
        failures.append(f"migration crosswalk: in_both={x['in_both']} in_neither={x['in_neither']} "
                        f"not_pass={x['migrated_but_not_pass']}")

if failures:
    for error in failures:
        print("EVIDENCE FAIL:", error)
    raise AssertionError("Evidence audit failed")
print("EVIDENCE PASS" if mode == "all" else "EVIDENCE PASS (SUBSET: not a gate)")
PY
}

main() {
    local mode="all"
    local case_id=""
    targets=()
    while (( $# > 0 )); do
        case "$1" in
            --help|-h) usage; return 0 ;;
            --all) mode="all"; shift ;;
            --new) targets+=(pytest gallery 13_105); mode="subset"; shift ;;
            --target)
                if [[ "${2:-}" == "?" ]]; then list_targets; return 0; fi
                if [[ -z "${2:-}" ]]; then printf 'ERROR: --target needs a name (or ?)\n' >&2; return 2; fi
                case "$2" in
                    pytest|gallery|13_101|13_105) targets+=("$2") ;;
                    *) printf 'ERROR: unknown target %s\n\n' "$2" >&2; list_targets >&2; return 2 ;;
                esac
                mode="subset"; shift 2 ;;
            --case)
                if [[ -z "${2:-}" ]]; then printf 'ERROR: --case needs a CASE-ID\n' >&2; return 2; fi
                case_id="$2"; shift 2 ;;
            *) usage >&2; printf 'ERROR: unknown argument: %s\n' "$1" >&2; return 2 ;;
        esac
    done
    if [[ "$mode" == "all" ]]; then
        targets=("${ALL_TARGETS[@]}")
    fi
    if [[ -n "$case_id" ]]; then
        if ! has_target 13_105; then printf 'ERROR: --case applies to --target 13_105 only\n' >&2; return 2; fi
    fi
    if [[ -n "$case_id" ]]; then mode="subset"; fi

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
    for required in scripts/env.local.sh examples/time_series/setup_env.sh; do
        if [[ ! -f "$script_dir/$required" ]]; then printf 'ERROR: missing %s\n' "$required" >&2; return 2; fi
    done
    if ! source "$script_dir/scripts/env.local.sh"; then printf 'ERROR: cannot initialize ADF environment\n' >&2; return 2; fi
    if ! source "$script_dir/examples/time_series/setup_env.sh"; then printf 'ERROR: cannot initialize ROOT environment\n' >&2; return 2; fi
    if ! cd "$script_dir"; then printf 'ERROR: cannot enter ADF repository\n' >&2; return 2; fi

    root_path="$script_dir/examples/time_series/time_series_tracks_0.root"
    identity_ref="$script_dir/test_logs/phase_13_105_S0/input_identity_first1M.json"
    if [[ ! -f "$root_path" ]]; then printf 'ERROR: ROOT fixture missing: %s\n' "$root_path" >&2; return 2; fi
    if has_target 13_105; then
        if [[ ! -f "$identity_ref" ]]; then printf 'ERROR: S0 input identity missing: %s\n' "$identity_ref" >&2; return 2; fi
    fi
    if ! command -v zip >/dev/null 2>&1; then printf 'ERROR: zip executable required\n' >&2; return 2; fi

    mkdir -p "$script_dir/test_logs"
    run_dir="$(mktemp -d "$script_dir/test_logs/gallery_$(date '+%Y%m%d_%H%M%S')_XXXXXX")"
    if [[ -z "$run_dir" ]]; then printf 'ERROR: unable to create evidence directory\n' >&2; return 2; fi
    local run_id="${run_dir##*/}"
    local target_tag; target_tag="$(IFS=+; printf '%s' "${targets[*]}")"
    if [[ "$mode" == "all" ]]; then target_tag="all"; fi
    results_file="$run_dir/RESULTS.tsv"
    : > "$results_file"
    failures=0

    {
        printf 'run_id: %s\nmode: %s\ntargets: %s\ncase: %s\nfixture: %s\nFAST ROOT entry_stop: 1000000\n' \
            "$run_id" "$mode" "${targets[*]}" "${case_id:-ALL}" "$root_path"
        stat -L -c 'ROOT file (resolved) bytes: %s; mtime: %y' "$root_path"
        printf 'ROOT file resolved path: %s\n' "$(readlink -f "$root_path")"
        git rev-parse HEAD
    } > "$run_dir/PROVENANCE.txt"
    sha256sum \
        run_gallery.sh \
        examples/time_series/time_series_draw.py \
        examples/time_series/time_series_draw_invariance.py \
        examples/time_series/time_series_draw_invariancetest.py \
        examples/time_series/phase_13_101_realdata_oracles.py \
        tests/test_time_series_invariancetest.py \
        tests/test_phase_13_101_realdata_fast_registry.py \
        tests/test_phase_13_101_dfdraw_adf_oracles.py \
        tests/test_phase_13_101_ts_entry_stop.py \
        tests/test_phase_13_101_bounded_stage_a.py \
        > "$run_dir/SOURCE_SHA256.txt"
    if [[ $? -ne 0 ]]; then printf 'ERROR: cannot fingerprint all source and test files\n' >&2; return 2; fi

    printf '\nRUN_GALLERY — %s — targets: %s — %s\n' "$mode" "${targets[*]}" "$run_id"
    printf 'Evidence directory: %s\n' "$run_dir"

    if has_target pytest; then
        run_stage focused-pytest "$run_dir/focused_pytest.log" \
            python -m pytest -q \
            tests/test_phase_13_101_realdata_fast_registry.py \
            tests/test_phase_13_101_dfdraw_adf_oracles.py \
            tests/test_phase_13_101_ts_entry_stop.py \
            tests/test_phase_13_101_bounded_stage_a.py \
            tests/test_phase_13_101_facet_residual_reference.py \
            tests/test_time_series_invariancetest.py
    fi

    if has_target 13_101; then
        run_stage historical-23-bounded "$run_dir/historical_stageA.log" \
            python examples/time_series/time_series_draw_invariance.py \
            "$root_path" --bounded-entries 1000000 \
            --manifest "$run_dir/historical_stageA_manifest.json" \
            --pdf "$run_dir/historical_stageA.pdf" --strict
        run_stage adf101-three-oracles "$run_dir/adf101_oracles.log" \
            python examples/time_series/time_series_draw_invariance.py \
            "$root_path" --adf101-fast \
            --manifest "$run_dir/adf101_manifest.json" --pdf "$run_dir/adf101_oracles.pdf" --strict
    fi

    # gallery and 13_105 share ONE Python process and ONE data load
    if has_target gallery || has_target 13_105; then
        local -a new_args=(python examples/time_series/time_series_draw_invariancetest.py "$root_path"
                           --bounded-entries 1000000 --strict --input-identity "$identity_ref"
                           --manifest "$run_dir/new_13_105_manifest.json")
        if has_target gallery; then new_args+=(--target gallery --gallery-pdf "$run_dir/original_gallery.pdf"); fi
        if has_target 13_105; then new_args+=(--target 13_105 --pdf "$run_dir/new_13_105.pdf"); fi
        if [[ -n "$case_id" ]]; then new_args+=(--case "$case_id"); fi
        run_stage new-gallery+13_105-one-load "$run_dir/new_13_105.log" "${new_args[@]}"
    fi

    run_stage evidence-audit "$run_dir/evidence_audit.log" check_outputs

    cp run_gallery.sh "$run_dir/run_gallery.sh"
    git diff -- examples/time_series tests run_gallery.sh > "$run_dir/WORKTREE_DIFF.txt"
    git diff --cached -- examples/time_series tests run_gallery.sh > "$run_dir/STAGED_DIFF.txt"

    {
        printf 'RUN_GALLERY %s\nmode=%s\ntargets=%s\ncase=%s\n' "$run_id" "$mode" "${targets[*]}" "${case_id:-ALL}"
        printf 'FAST fixture=first 1000000 source entries, no post-read sampling\n'
        if has_target 13_101; then printf 'old_13_101_scientific_cases=26\n'; fi
        if has_target 13_105; then printf 'new_13_105_cases=%s\n' "$(python -c "import json,sys; print(len(json.load(open(sys.argv[1]))['cases']))" "$run_dir/new_13_105_manifest.json" 2>/dev/null || echo unknown)"; fi
        printf 'post_read_sampling=NONE\n'
        printf 'failed_stages=%s\n' "$failures"
        cat "$results_file"
    } > "$run_dir/SUMMARY.txt"

    local packet="$script_dir/test_logs/gallery_reviewer_${run_id}_${target_tag}.zip"
    if zip -q -j "$packet" "$run_dir"/*; then
        printf 'Reviewer packet: %s\n' "$packet"
        sha256sum "$packet" > "$script_dir/test_logs/gallery_reviewer_${run_id}_${target_tag}.sha256"
    else
        printf 'ERROR: failed to produce gallery reviewer ZIP\n' >&2
        failures=$((failures + 1))
    fi

    cat "$run_dir/SUMMARY.txt"
    if (( failures > 0 )); then
        printf 'RUN_GALLERY FAIL (%s): %s stage(s) failed; evidence retained.\n' "$mode" "$failures" >&2
        return 1
    fi
    if [[ "$mode" == "all" ]]; then
        printf 'RUN_GALLERY PASS (all targets). Visual review and official CRR remain separate.\n'
    else
        printf 'RUN_GALLERY SUBSET PASS (targets: %s) — not a gate.\n' "${targets[*]}"
    fi
    return 0
}

main "$@"
