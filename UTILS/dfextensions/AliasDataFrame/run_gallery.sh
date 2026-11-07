#!/usr/bin/env bash
# AliasDataFrame real-data gallery runner — PHASE_13_101 checkpoint.
#
# Usage (from AliasDataFrame repository):
#   ./run_gallery.sh            ALL: first 1M rows, 59-page gallery, 23+3 oracles, focused tests
#   ./run_gallery.sh --help     Explain coverage and output
#
# Default mode runs ALL currently implemented gallery tests on the first
# million ROOT entries, without post-read sampling. The old canonical 20%
# entrypoint is not part of this runner.
#
# Every run writes an isolated test_logs/gallery_<timestamp>_<unique>/ folder
# and a gallery_reviewer_*.zip. Errors retain evidence and make the command
# return a nonzero status. This is orchestration, NOT a second oracle engine.
#
# Shell working style: one step at a time; scoped Git inspections.

set -o pipefail

usage() {
    cat <<'HELP'
AliasDataFrame gallery runner

  ./run_gallery.sh            COMPLETE BOUNDED: first 1,000,000 ROOT entries
  ./run_gallery.sh --fast     Same as default
  ./run_gallery.sh --help     Show this message

Default executes all CURRENT scientific cases:
  - all focused/synthetic PHASE_13_101 pytest tests
  - full original 59-page readable time-series gallery
  - 23 historical Stage-A cases on first 1M unsampled ROOT entries
  - three ADF101 scientific oracles on the same bounded ROOT window
  - manifest, PDF, logs, flat reviewer ZIP

Both real-data suites are strict. The old 20%/seed42 mode is not invoked
by this launcher.

Evidence: test_logs/gallery_<timestamp>_<unique>/* and
          test_logs/gallery_reviewer_<timestamp>_<unique>.zip
HELP
}

# Executing a stage uses the original code path, with a separately retained log.
# A failed stage never prevents packaging the evidence from other stages.
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

# Keep the gallery's actual draw calls in time_series_draw.py.
# This tiny launcher only supplies the bounded ROOT fixture and output path.
run_original_gallery() {
    python - "$root_path" "$gallery_pdf" <<'PY'
import sys
import time
from pathlib import Path
from time_series_draw import build_adf, run_all_pdf

root, output = sys.argv[1:]
t0 = time.perf_counter()
adf = build_adf(root, sample=None, lazy=False, entry_stop=1_000_000)
t1 = time.perf_counter()
result = run_all_pdf(adf, output)
t2 = time.perf_counter()

print(f"ROOT entries: {len(adf.df):,}")
print(f"Loading: {t1 - t0:.2f}s; gallery: {t2 - t1:.2f}s")
print(f"Original gallery: {result}; PDF: {output}")
assert len(adf.df) == 1_000_000, "FAST input must contain exactly 1M rows"
assert result is True, "Historical gallery reported failure"
assert Path(output).is_file(), "Historical gallery PDF missing"
assert Path(output).stat().st_size > 1000, "Historical gallery PDF empty"
PY
}

# This is only an output/provenance check. Scientific truth stays in Stage-A.
# Matplotlib's generated PDFs expose /Type /Page objects in plain PDF syntax.
# It does not replace a human visual inspection or a full PDF library parser.
check_outputs() {
    python - "$run_dir" "$mode" <<'PY'
import hashlib
import json
import re
import sys
from pathlib import Path

folder = Path(sys.argv[1])
mode = sys.argv[2]
expected = [('original_gallery.pdf', 59), ('historical_stageA.pdf', None),
            ('adf101_oracles.pdf', 3)]

failures = []
for name, pages in expected:
    file = folder / name
    if not file.is_file():
        failures.append(f'{name}: missing')
        continue
    data = file.read_bytes()
    if not data.startswith(b'%PDF-'):
        failures.append(f'{name}: missing PDF header')
        continue
    number = len(re.findall(rb'/Type\s*/Page\b', data))
    digest = hashlib.sha256(data).hexdigest()
    print(f'PDF {name}: {number} page objects, {len(data)} bytes, SHA256={digest}')
    if pages is not None and number != pages:
        failures.append(f'{name}: expected {pages} page objects, found {number}')

# Fail closed on missing cases, incomplete execution and unobserved science.
manifest_specs = [('historical_stageA_manifest.json', 23),
                  ('adf101_manifest.json', 3)]
for name, expected_cases in manifest_specs:
    file = folder / name
    if not file.is_file():
        failures.append(f'{name}: missing')
        continue
    try:
        content = json.loads(file.read_text())
    except (OSError, ValueError) as exc:
        failures.append(f'{name}: unreadable: {exc}')
        continue
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    actual_cases = content.get('cases', [])
    rec = content.get('reconciliation', {})
    print(f'MANIFEST {name}: cases={len(actual_cases)}, SHA256={digest}')
    if len(actual_cases) != expected_cases:
        failures.append(f'{name}: expected {expected_cases} cases, got {len(actual_cases)}')
    if rec.get('n_declared') != expected_cases or rec.get('n_results') != expected_cases:
        failures.append(f'{name}: incomplete declared/executed case reconciliation')
    if rec.get('gating'):
        failures.append(f'{name}: has {len(rec["gating"])} gating RED results')
    if any(row.get('status') != 'PASS' for row in actual_cases):
        failures.append(f'{name}: not all cases report PASS')
    if name == 'historical_stageA_manifest.json':
        prov = content.get('provenance', {})
        if (prov.get('sample_mode') != 'FULL'
                or prov.get('bounded_fixture_kind') != 'FIRST_N_NO_FRACTION'
                or prov.get('entry_start') != 0 or prov.get('entry_stop') != 1000000
                or prov.get('sample_fraction') is not None):
            failures.append(f'{name}: bounded unsampled provenance mismatch')
    if name == 'adf101_manifest.json':
        for row in actual_cases:
            observed = row.get('observed', {})
            if not isinstance(observed, dict):
                failures.append(f'{name}: missing observed details')
                continue
            if int(row.get('executed_comparisons', 0)) <= 0:
                failures.append(f'{name}: {row.get("case_id")} no comparisons')
            if not observed.get('falsifier', {}).get('detected'):
                failures.append(f'{name}: {row.get("case_id")} falsifier not detected')

if failures:
    for error in failures:
        print('EVIDENCE FAIL:', error)
    raise AssertionError('Evidence audit failed')
print('EVIDENCE PASS: expected files and Matplotlib PDF page objects present')
PY
}

main() {
    local mode="fast"
    local script_dir

    case "${1:---fast}" in
        --help|-h)
            usage
            return 0
            ;;
        --fast)
            mode="fast"
            ;;
        *)
            usage >&2
            printf 'ERROR: unknown mode: %s\n' "$1" >&2
            return 2
            ;;
    esac
    if (( $# > 1 )); then
        printf 'ERROR: pass only one mode argument\n' >&2
        return 2
    fi

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"
    if [[ ! -f "$script_dir/scripts/env.local.sh" ]]; then
        printf 'ERROR: missing scripts/env.local.sh\n' >&2
        return 2
    fi
    if [[ ! -f "$script_dir/examples/time_series/setup_env.sh" ]]; then
        printf 'ERROR: missing examples/time_series/setup_env.sh\n' >&2
        return 2
    fi
    if ! source "$script_dir/scripts/env.local.sh"; then
        printf 'ERROR: cannot initialize ADF environment\n' >&2
        return 2
    fi
    if ! source "$script_dir/examples/time_series/setup_env.sh"; then
        printf 'ERROR: cannot initialize ROOT/time-series environment\n' >&2
        return 2
    fi
    if ! cd "$script_dir"; then
        printf 'ERROR: cannot enter ADF repository\n' >&2
        return 2
    fi

    root_path="$script_dir/examples/time_series/time_series_tracks_0.root"
    if [[ ! -f "$root_path" ]]; then
        printf 'ERROR: ROOT fixture missing: %s\n' "$root_path" >&2
        return 2
    fi
    if ! command -v zip >/dev/null 2>&1; then
        printf 'ERROR: zip executable required to make reviewer packet\n' >&2
        return 2
    fi

    mkdir -p "$script_dir/test_logs"
    run_dir="$(mktemp -d "$script_dir/test_logs/gallery_$(date '+%Y%m%d_%H%M%S')_XXXXXX")"
    if [[ -z "$run_dir" ]]; then
        printf 'ERROR: unable to create unique evidence directory\n' >&2
        return 2
    fi
    local run_id="${run_dir##*/}"
    results_file="$run_dir/RESULTS.tsv"
    : > "$results_file"
    failures=0

    gallery_pdf="$run_dir/original_gallery.pdf"
    local new_pdf="$run_dir/adf101_oracles.pdf"
    local new_manifest="$run_dir/adf101_manifest.json"
    local packet="$script_dir/test_logs/gallery_reviewer_${run_id}.zip"

    printf 'run_id: %s\nmode: %s\nfixture: %s\nFAST ROOT entry_stop: 1000000\n' \
        "$run_id" "$mode" "$root_path" > "$run_dir/PROVENANCE.txt"
    stat -c 'ROOT file bytes: %s; mtime: %y' "$root_path" >> "$run_dir/PROVENANCE.txt"
    git rev-parse HEAD >> "$run_dir/PROVENANCE.txt"
    sha256sum \
        run_gallery.sh \
        examples/time_series/time_series_draw.py \
        examples/time_series/time_series_draw_invariance.py \
        examples/time_series/phase_13_101_realdata_oracles.py \
        tests/test_phase_13_101_realdata_fast_registry.py \
        tests/test_phase_13_101_dfdraw_adf_oracles.py \
        tests/test_phase_13_101_ts_entry_stop.py \
        tests/test_phase_13_101_bounded_stage_a.py \
        > "$run_dir/SOURCE_SHA256.txt"
    if [[ $? -ne 0 ]]; then
        printf 'ERROR: cannot fingerprint all source and test files\n' >&2
        return 2
    fi

    printf '\nRUN_GALLERY — %s — %s\n' "$mode" "$run_id"
    printf 'Evidence directory: %s\n' "$run_dir"
    printf 'COMPLETE BOUNDED: all current cases, first 1M unsampled ROOT rows.\n'

    run_stage focused-pytest "$run_dir/focused_pytest.log" \
        python -m pytest -q \
        tests/test_phase_13_101_realdata_fast_registry.py \
        tests/test_phase_13_101_dfdraw_adf_oracles.py \
        tests/test_phase_13_101_ts_entry_stop.py

    run_stage original-59-page-gallery "$run_dir/original_gallery.log" \
        run_original_gallery

    run_stage historical-23-bounded "$run_dir/historical_stageA.log" \
        python examples/time_series/time_series_draw_invariance.py \
        "$root_path" --bounded-entries 1000000 \
        --manifest "$run_dir/historical_stageA_manifest.json" \
        --pdf "$run_dir/historical_stageA.pdf" --strict

    run_stage adf101-three-oracles "$run_dir/adf101_oracles.log" \
        python examples/time_series/time_series_draw_invariance.py \
        "$root_path" --adf101-fast \
        --manifest "$new_manifest" --pdf "$new_pdf" --strict

    run_stage evidence-audit "$run_dir/evidence_audit.log" check_outputs

    cp -p "$script_dir/run_gallery.sh" "$run_dir/run_gallery.sh"
    git diff -- \
        examples/time_series/time_series_draw_invariance.py \
        examples/time_series/phase_13_101_realdata_oracles.py \
        tests/test_phase_13_101_realdata_fast_registry.py \
        > "$run_dir/WORKTREE_DIFF.txt"
    git diff --cached -- \
        examples/time_series/time_series_draw_invariance.py \
        examples/time_series/phase_13_101_realdata_oracles.py \
        tests/test_phase_13_101_realdata_fast_registry.py \
        > "$run_dir/STAGED_DIFF.txt"

    {
        printf 'RUN_GALLERY %s\n' "$run_id"
        printf 'mode=%s\n' "$mode"
        printf 'FAST fixture=first 1000000 source entries\n'
        printf 'historical_23_case_stageA=BOUNDED_FIRST_1000000\n'
        printf 'new_ADF101_oracles=BOUNDED_FIRST_1000000\n'
        printf 'total_scientific_cases=26\n'
        printf 'post_read_sampling=NONE\n'
        printf 'failed_stages=%s\n' "$failures"
        cat "$results_file"
    } > "$run_dir/SUMMARY.txt"

    # Keep the evidence folder even after failures; ZIP everything with flat
    # filenames for direct reviewer inspection. This runner has no external
    # PDF reader dependency (unlike the previous ad-hoc validator).
    if zip -q -j "$packet" "$run_dir"/*; then
        printf 'Reviewer packet: %s\n' "$packet"
        sha256sum "$packet" > "$script_dir/test_logs/gallery_reviewer_${run_id}.sha256"
    else
        printf 'ERROR: failed to produce gallery reviewer ZIP\n' >&2
        failures=$((failures + 1))
    fi

    cat "$run_dir/SUMMARY.txt"
    if (( failures > 0 )); then
        printf 'RUN_GALLERY FAIL: %s stage(s) failed; evidence retained.\n' "$failures" >&2
        return 1
    fi
    printf 'RUN_GALLERY PASS (%s). Visual review and official CRR remain separate.\n' "$mode"
    return 0
}

main "$@"
