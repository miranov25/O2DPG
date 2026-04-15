"""Phase 13.18.GB Turn 5 — Layer A safety, adversarial, edge-case tests.

Layer A is feature-complete after this turn. This test file adds the
final Safety coverage that was not captured in the lookup / linear
parity tests (Turns 3-4):

  1. P1-β closure (Claude23 P1-2): all-corners-invalid linear query
     returns NaN, not 0/0 from zero-division, not zero, not edge value.
     Uses an inline minimal 2D sparse fixture constructed in this
     test; not a fixture file because the structural requirement
     (every 2^N corner invalid) does not cleanly map into the
     orthogonal-array 24-fixture scheme.

  2. Adversarial consolidation: the 4 adversarial fixtures F_21-F_24
     are already covered by the parity tests and specific assertions
     in test_layer_a_lookup.py and test_layer_a_linear.py. This file
     does NOT re-assert them (would be redundant) but references them
     for audit traceability.

  3. Out-of-bounds behaviour — constructed inline:
     - bounds='nan' with out-of-grid position returns NaN
     - bounds='clamp' with out-of-grid position snaps to edge

  4. Input validation — the cli_runner returns structured error
     responses for malformed fixtures; these tests exercise the
     constructor validation paths.

The test approach: build custom fixture JSON dynamically in the test
and invoke cli_runner against it, mirroring the Turn 3/4 subprocess
harness. No new C++ code is expected from this turn; the tests
exercise already-implemented behaviour.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

CPP_ROOT = Path(__file__).resolve().parent.parent
BUILD_DIR = CPP_ROOT / "build"
CLI_RUNNER = BUILD_DIR / "cli_runner"


@pytest.fixture(scope="session")
def cli_runner_binary() -> Path:
    """Compile once per session (reuses Turn 3 target)."""
    result = subprocess.run(
        ["make", "-C", str(CPP_ROOT), "layer_a_lookup"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"make layer_a_lookup failed:\n{result.stdout}\n{result.stderr}"
        )
    if not CLI_RUNNER.exists():
        pytest.fail(f"cli_runner binary not found at {CLI_RUNNER}")
    return CLI_RUNNER


def _run(binary: Path, fixture: dict, tmp_path: Path,
         *, expect_error: bool = False) -> dict:
    """Write fixture dict to a tmp file and invoke cli_runner."""
    p = tmp_path / "safety_fixture.json"
    p.write_text(json.dumps(fixture))
    result = subprocess.run(
        [str(binary), str(p)],
        capture_output=True, text=True, check=False,
    )
    try:
        out = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"cli_runner stdout was not valid JSON:\n"
            f"{result.stdout}\nparse error: {e}"
        )
    if expect_error:
        assert out.get("status") == "error", (
            f"expected error response, got ok: {out}"
        )
        assert result.returncode != 0, (
            f"expected non-zero exit on error; got 0"
        )
    else:
        assert out.get("status") == "ok", (
            f"cli_runner reported error: {out.get('error')!r}"
        )
    return out


# ---------- Helper: build a minimal linear fixture in-memory ----------


def _build_linear_fixture(
    *,
    group_columns: list[str],
    predictor_columns: list[str],
    targets: list[str],
    suffix: str,
    fit_intercept: bool,
    subframe_rows: list[dict],
    query_positions: list[list[float]],
    predictor_values_per_query: list[list[float]],
    bounds: str,
    method: str = "linear",
    expected_output: dict | None = None,
) -> dict:
    """Build the full 5-key fixture dict matching FIXTURE_SPEC §4."""
    return {
        "fixture_id": "SAFETY_INLINE",
        "axis_values": {
            "fit_intercept": fit_intercept,
            "bounds": bounds,
            "dimensions": len(group_columns),
            "method": method,
            "coverage": "sparse",
        },
        "input": {
            "gb_columns": group_columns,
            "predictor_columns": predictor_columns,
            "targets": targets,
            "suffix": suffix,
            "fit_intercept": fit_intercept,
            "subframe_rows": subframe_rows,
            "query_positions": query_positions,
            "predictor_values_per_query": predictor_values_per_query,
            "method": method,
            "bounds": bounds,
        },
        "expected_output": expected_output or {t: [] for t in targets},
        "intermediates": {
            "expected_bin_centers": [],
            "expected_valid_mask_flat": [],
            "expected_coefficient_shape": [],
            "expected_remap": [],
        },
        "metadata": {
            "generator_version": "inline-safety-test",
            "generator_timestamp": "2026-04-13T00:00:00Z",
            "python_evaluator_hash": "inline",
            "notes": "Turn 5 inline safety fixture",
        },
    }


# ======================================================================
# P1-β: all-corners-invalid linear query -> NaN (Claude23 P1-2)
# ======================================================================


def test_linear_all_corners_invalid_returns_nan(
    cli_runner_binary: Path, tmp_path: Path) -> None:
    """Construct a minimal 2D sparse grid where the ONLY valid cells
    are at positions (0,0) and (2,2), and query the midpoint (1.5, 1.5).

    Grid after construction: 3x3 (bin_centers = [0,1,2] per dim).
    Valid cells: (0,0) and (2,2).
    Query at (1.5, 1.5) -> 4 corners are (1,1), (1,2), (2,1), (2,2).
    - (1,1): missing
    - (1,2): missing
    - (2,1): missing
    - (2,2): valid

    So with 3 of 4 corners invalid, query returns the renormalized
    value — not the all-invalid case.

    To force all-invalid, we query at (1.0, 1.0) where corners are
    (0,0), (0,1), (1,0), (1,1) — none valid except (0,0).
    Still not all-invalid.

    Correct construction: to get ALL 4 corners invalid, we need a
    query whose 4 bilinear corners are all missing. With valid cells
    at (0,0) and (2,2) only, query at (1.5, 0.5) has corners
    (1,0), (1,1), (2,0), (2,1) — ALL missing. Valid cells (0,0) and
    (2,2) are not among them. This is the all-corners-invalid case.

    Expected: NaN (Safety contract). NOT 0/0 zero-division. NOT 0.0.
    NOT edge value.
    """
    rows = [
        {"bx": 0, "by": 0, "y_intercept_fit": 1.0},
        {"bx": 2, "by": 2, "y_intercept_fit": 9.0},
    ]
    fx = _build_linear_fixture(
        group_columns=["bx", "by"],
        predictor_columns=[],
        targets=["y"],
        suffix="_fit",
        fit_intercept=True,
        subframe_rows=rows,
        # Query at (1.5, 0.5): corners are (1,0), (1,1), (2,0), (2,1)
        # - All 4 corners are MISSING. Expected: NaN.
        query_positions=[[1.5, 0.5]],
        predictor_values_per_query=[[]],
        bounds="nan",
    )
    out = _run(cli_runner_binary, fx, tmp_path)
    got = out["predictions"]["y"][0]

    # Must be NaN — explicit string match
    assert isinstance(got, str) and got == "NaN", (
        f"P1-β FAILURE: all-corners-invalid linear query MUST return "
        f"NaN per Safety contract. Got {got!r}.\n"
        f"  If got == 0.0: zero-division or unguarded 0/0 reduction bug\n"
        f"  If got finite: weight_sum check missing, edge value leaked\n"
        f"  If got large finite: renormalization divided by tiny weight_sum"
    )


def test_linear_all_corners_invalid_3D(
    cli_runner_binary: Path, tmp_path: Path) -> None:
    """3D variant: 8 corners, all invalid. Confirms the logic scales
    beyond 2D and is not an artifact of a small corner count."""
    rows = [
        # Only one valid cell far from the query's corners
        {"bx": 0, "by": 0, "bz": 0, "y_intercept_fit": 1.0},
        {"bx": 3, "by": 3, "bz": 3, "y_intercept_fit": 8.0},
    ]
    fx = _build_linear_fixture(
        group_columns=["bx", "by", "bz"],
        predictor_columns=[],
        targets=["y"],
        suffix="_fit",
        fit_intercept=True,
        subframe_rows=rows,
        # Query at (1.5, 1.5, 1.5) — all 8 corners are between (1,1,1)
        # and (2,2,2), none of which is populated. All invalid.
        query_positions=[[1.5, 1.5, 1.5]],
        predictor_values_per_query=[[]],
        bounds="nan",
    )
    out = _run(cli_runner_binary, fx, tmp_path)
    got = out["predictions"]["y"][0]
    assert isinstance(got, str) and got == "NaN", (
        f"3D all-corners-invalid must return NaN; got {got!r}"
    )


# ======================================================================
# Out-of-bounds behaviour
# ======================================================================


def test_out_of_grid_nan_mode_returns_nan(
    cli_runner_binary: Path, tmp_path: Path) -> None:
    """bounds='nan' with position outside [0, N-1] in any dim -> NaN."""
    rows = [
        {"bx": 0, "y_intercept_fit": 1.0},
        {"bx": 1, "y_intercept_fit": 2.0},
        {"bx": 2, "y_intercept_fit": 3.0},
    ]
    fx = _build_linear_fixture(
        group_columns=["bx"], predictor_columns=[], targets=["y"],
        suffix="_fit", fit_intercept=True,
        subframe_rows=rows,
        query_positions=[[-0.5], [3.5]],  # below 0, above N-1=2
        predictor_values_per_query=[[], []],
        bounds="nan",
    )
    out = _run(cli_runner_binary, fx, tmp_path)
    got = out["predictions"]["y"]
    assert isinstance(got[0], str) and got[0] == "NaN", (
        f"bounds='nan' with pos=-0.5 must return NaN; got {got[0]!r}"
    )
    assert isinstance(got[1], str) and got[1] == "NaN", (
        f"bounds='nan' with pos=3.5 must return NaN; got {got[1]!r}"
    )


def test_out_of_grid_clamp_mode_returns_edge(
    cli_runner_binary: Path, tmp_path: Path) -> None:
    """bounds='clamp' with position outside [0, N-1] -> snap to edge.
    Linear at exactly the clamped edge == the edge cell's value."""
    rows = [
        {"bx": 0, "y_intercept_fit": 10.0},
        {"bx": 1, "y_intercept_fit": 20.0},
        {"bx": 2, "y_intercept_fit": 30.0},
    ]
    fx = _build_linear_fixture(
        group_columns=["bx"], predictor_columns=[], targets=["y"],
        suffix="_fit", fit_intercept=True,
        subframe_rows=rows,
        # Below left edge clamps to bx=0 -> 10.0
        # Above right edge clamps to bx=2 -> 30.0
        query_positions=[[-5.0], [99.0]],
        predictor_values_per_query=[[], []],
        bounds="clamp",
    )
    out = _run(cli_runner_binary, fx, tmp_path)
    got = out["predictions"]["y"]
    assert got[0] == 10.0, f"clamp to left edge expected 10.0; got {got[0]!r}"
    assert got[1] == 30.0, f"clamp to right edge expected 30.0; got {got[1]!r}"


def test_missing_bin_lookup_returns_nan(
    cli_runner_binary: Path, tmp_path: Path) -> None:
    """Direct query at a sparse-invalid cell via lookup returns NaN,
    matching std::nan("") semantic (not a neighbor leak).

    Construction note: bin_centers is inferred from distinct values in
    rows. To get a 'missing bin' we need THREE distinct values in
    rows (so the grid has shape 3) but only TWO of them fully
    populated. Using 2D: the missing cell at (0,1) is a bin_centers
    position but has no row.
    """
    rows = [
        {"bx": 0, "by": 0, "y_intercept_fit": 1.0},
        # (0, 1) missing deliberately
        {"bx": 0, "by": 2, "y_intercept_fit": 3.0},
        {"bx": 1, "by": 0, "y_intercept_fit": 4.0},
        {"bx": 1, "by": 1, "y_intercept_fit": 5.0},
        {"bx": 1, "by": 2, "y_intercept_fit": 6.0},
    ]
    fx = _build_linear_fixture(
        group_columns=["bx", "by"], predictor_columns=[], targets=["y"],
        suffix="_fit", fit_intercept=True,
        subframe_rows=rows,
        # Compact indices after remap: bx in {0,1}, by in {0,1,2}
        query_positions=[[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
        predictor_values_per_query=[[], [], []],
        bounds="nan",
        method="lookup",
    )
    out = _run(cli_runner_binary, fx, tmp_path)
    got = out["predictions"]["y"]
    assert got[0] == 1.0, f"valid (0,0) expected 1.0; got {got[0]!r}"
    assert isinstance(got[1], str) and got[1] == "NaN", (
        f"missing (0,1) MUST be NaN; got {got[1]!r} — neighbor leak bug"
    )
    assert got[2] == 3.0, f"valid (0,2) expected 3.0; got {got[2]!r}"


# ======================================================================
# Input validation — cli_runner returns structured errors
# ======================================================================


def test_missing_coefficient_column_raises(
    cli_runner_binary: Path, tmp_path: Path) -> None:
    """A row missing its required intercept coefficient -> error."""
    rows = [
        {"bx": 0, "y_intercept_fit": 1.0},
        {"bx": 1},  # missing y_intercept_fit
    ]
    fx = _build_linear_fixture(
        group_columns=["bx"], predictor_columns=[], targets=["y"],
        suffix="_fit", fit_intercept=True,
        subframe_rows=rows,
        query_positions=[[0.0]],
        predictor_values_per_query=[[]],
        bounds="nan",
    )
    out = _run(cli_runner_binary, fx, tmp_path, expect_error=True)
    assert "missing required coefficient column" in out["error"].lower() \
        or "missing" in out["error"].lower(), (
        f"expected missing-column error; got {out['error']!r}"
    )


def test_empty_rows_raises(
    cli_runner_binary: Path, tmp_path: Path) -> None:
    """Zero subframe rows -> structured error at construction."""
    fx = _build_linear_fixture(
        group_columns=["bx"], predictor_columns=[], targets=["y"],
        suffix="_fit", fit_intercept=True,
        subframe_rows=[],
        query_positions=[[0.0]],
        predictor_values_per_query=[[]],
        bounds="nan",
    )
    out = _run(cli_runner_binary, fx, tmp_path, expect_error=True)
    assert "rows" in out["error"].lower() \
        or "non-empty" in out["error"].lower(), (
        f"expected 'rows must be non-empty' class error; got {out['error']!r}"
    )


def test_extra_top_level_key_rejected(
    cli_runner_binary: Path, tmp_path: Path) -> None:
    """Restricted-parser contract: extra top-level key rejected (P1-η)."""
    fx = _build_linear_fixture(
        group_columns=["bx"], predictor_columns=[], targets=["y"],
        suffix="_fit", fit_intercept=True,
        subframe_rows=[{"bx": 0, "y_intercept_fit": 1.0}],
        query_positions=[[0.0]],
        predictor_values_per_query=[[]],
        bounds="nan",
    )
    fx["unexpected_extra_key"] = "should be rejected"
    out = _run(cli_runner_binary, fx, tmp_path, expect_error=True)
    assert "unknown top-level" in out["error"].lower() \
        or "unexpected_extra_key" in out["error"].lower(), (
        f"expected unknown-key error; got {out['error']!r}"
    )


def test_wrong_arity_position_raises(
    cli_runner_binary: Path, tmp_path: Path) -> None:
    """Query position with wrong number of dimensions -> error."""
    rows = [
        {"bx": 0, "by": 0, "y_intercept_fit": 1.0},
        {"bx": 1, "by": 1, "y_intercept_fit": 2.0},
    ]
    fx = _build_linear_fixture(
        group_columns=["bx", "by"], predictor_columns=[], targets=["y"],
        suffix="_fit", fit_intercept=True,
        subframe_rows=rows,
        # Only 1 dim provided for a 2D schema
        query_positions=[[0.5]],
        predictor_values_per_query=[[]],
        bounds="nan",
    )
    out = _run(cli_runner_binary, fx, tmp_path, expect_error=True)
    assert ("position" in out["error"].lower()
            or "size" in out["error"].lower()
            or "group_columns" in out["error"].lower()), (
        f"expected arity-mismatch error; got {out['error']!r}"
    )
