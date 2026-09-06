"""PHASE_13_80 A14/A15 timing instrumentation acceptance tests."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap


def _runner_path() -> Path:
    override = os.environ.get("ADF_RUNNER_UNDER_TEST")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "run_tests.sh"


def _runner_text() -> str:
    return _runner_path().read_text(encoding="utf-8")


def test_runner_shell_syntax_is_valid():
    proc = subprocess.run(
        ["bash", "-n", str(_runner_path())],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_timing_artifact_schema_and_timezone_are_declared():
    text = _runner_text()
    assert 'ADF_RUN_TIMING/1' in text
    assert 'timing_${TS}.txt' in text
    assert 'date +"%Y-%m-%d %H:%M:%S %z"' in text
    assert 'total_seconds:' in text
    assert 'unclassified_other_seconds:' in text


def test_full_pytest_pipestatus_capture_is_immediate():
    """No executed command may appear between full pytest pipeline and capture."""
    lines = _runner_text().splitlines()
    tee_index = next(
        i for i, line in enumerate(lines)
        if '2>&1 | tee "$LOG_FILE"' in line
    )
    next_nonblank = next(
        line.strip() for line in lines[tee_index + 1 :] if line.strip()
    )
    assert next_nonblank == 'TEST_EXIT=${PIPESTATUS[0]}'


def test_focused_pytest_pipestatus_capture_is_immediate():
    lines = _runner_text().splitlines()
    tee_index = next(
        i for i, line in enumerate(lines)
        if '2>&1 | tee "$FOCUSED_LOG"' in line
    )
    next_nonblank = next(
        line.strip() for line in lines[tee_index + 1 :] if line.strip()
    )
    assert next_nonblank == 'FOCUSED_EXIT=${PIPESTATUS[0]}'


def test_timing_helpers_do_not_execute_test_or_packaging_commands():
    text = _runner_text()
    start = text.index("timing_start()")
    end = text.index("PYTEST_WORKERS=", start)
    helper_block = text[start:end]
    forbidden = (
        "python3 -m pytest",
        " zip ",
        "tar -",
        "make_llm_bundle.py",
    )
    for token in forbidden:
        assert token not in helper_block


def test_major_stage_names_are_visible():
    text = _runner_text()
    required = {
        "diff_last_commit",
        "diff_to_phase",
        "environment / provenance",
        "full pytest",
        "focused collection",
        "focused pytest",
        "raw-XFAIL pytest",
        "Capability Matrix Markdown/JSON generation",
        "Capability Matrix HTML generation",
        "candidate hashing / MD5 manifest",
        "summary generation",
        "pre-bundle custody checks",
        "reviewer ZIP creation",
        "reviewer TAR creation",
        "LLMBUNDLE conversion",
        "reviewer package digest generation",
    }
    missing = sorted(stage for stage in required if stage not in text)
    assert not missing, missing


def test_packet_contains_frozen_timing_snapshot_not_mutating_final_timing_file():
    text = _runner_text()
    assert 'timing_packet_${TS}.txt' in text
    assert '"$LOG_DIR/timing_packet_${TS}.txt"' in text
    assert "timing_snapshot_for_packet" in text
    assert "self-referential packet-byte drift" in text
    # The mutable final timing artifact itself must not be added to ZIP_FILES.
    zip_section = text.split('echo "--- Packaging reviewer.zip ---"', 1)[1]
    zip_inventory = zip_section.split(
        "# Reviewer source custody:", 1
    )[0]
    assert '"$LOG_DIR/timing_${TS}.txt"' not in zip_inventory


def test_timing_is_not_written_into_capability_matrix_artifacts():
    text = _runner_text()
    for target in (
        'docs/CAPABILITY_MATRIX.json',
        'docs/CAPABILITY_MATRIX.html',
        'docs/CAPABILITY_MATRIX.md',
    ):
        # Matrix files may be generated/copied as before, but timing writes must
        # not target them.
        assert not re.search(
            rf'(timing_|ADF_RUN_TIMING).*?>\s*["\']?{re.escape(target)}',
            text,
        )


def test_final_summary_reports_other_and_total():
    text = _runner_text()
    assert '"unclassified/other"' in text
    assert '"TOTAL"' in text
    assert 'stage_sum=$(awk' in text
    assert 'other=$((total_seconds - stage_sum))' in text


def test_deliberately_failing_pytest_pipeline_preserves_nonzero_summary(tmp_path):
    """Executable falsifier for the exact pipeline/capture ordering."""
    fail_test = tmp_path / "test_deliberate_failure.py"
    fail_test.write_text(
        "def test_deliberate_failure():\n    assert False\n",
        encoding="utf-8",
    )
    log_file = tmp_path / "pytest.log"
    summary_file = tmp_path / "SUMMARY.txt"
    timing_file = tmp_path / "timing.txt"
    harness = tmp_path / "harness.sh"
    harness.write_text(
        textwrap.dedent(
            r"""
            #!/bin/bash
            TIMING_FILE="$4"

            timing_start() {
                local stage="$1"
                TIMING_STAGE_START_EPOCH=$(date +%s)
                TIMING_STAGE_START_WALL=$(date +"%Y-%m-%d %H:%M:%S %z")
                printf '[%s] START %s\n' "$TIMING_STAGE_START_WALL" "$stage"
                printf 'stage: %s\nstart: %s\n' \
                    "$stage" "$TIMING_STAGE_START_WALL" >> "$TIMING_FILE"
            }

            timing_end() {
                local stage="$1"
                local result="$2"
                local end_epoch end_wall elapsed
                end_epoch=$(date +%s)
                end_wall=$(date +"%Y-%m-%d %H:%M:%S %z")
                elapsed=$((end_epoch - TIMING_STAGE_START_EPOCH))
                printf '[%s] END %s elapsed=%ss result=%s\n' \
                    "$end_wall" "$stage" "$elapsed" "$result"
                printf 'end: %s\nelapsed_seconds: %s\nresult: %s\n' \
                    "$end_wall" "$elapsed" "$result" >> "$TIMING_FILE"
            }

            timing_start "full pytest"
            python3 -m pytest "$1" -q 2>&1 | tee "$2"
            TEST_EXIT=${PIPESTATUS[0]}
            timing_end "full pytest" "$TEST_EXIT"
            printf 'Exit: %s\n' "$TEST_EXIT" > "$3"
            exit "$TEST_EXIT"
            """
        ).lstrip(),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            "bash",
            str(harness),
            str(fail_test),
            str(log_file),
            str(summary_file),
            str(timing_file),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    summary = summary_file.read_text(encoding="utf-8")
    match = re.search(r"Exit:\s*(\d+)", summary)
    assert match
    assert int(match.group(1)) != 0
    timing = timing_file.read_text(encoding="utf-8")
    assert "stage: full pytest" in timing
    assert re.search(r"result:\s*[1-9]\d*", timing)
    assert re.search(r"\+\d{4}", timing)
