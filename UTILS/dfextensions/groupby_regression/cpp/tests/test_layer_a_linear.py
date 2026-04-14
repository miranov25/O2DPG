"""Phase 13.18.GB Turn 4 — Layer A linear subprocess tests.

Invokes cpp/build/cli_runner with method=linear fixtures, parses
structured JSON stdout, asserts parity at rtol=1e-12. FP-1
relaxation to rtol=1e-10 is pre-authorized only if (a) fixture size
N > 10000 OR (b) Kahan-disabled Python ref also fails — neither
condition applies to any of the 12 small-grid fixtures, so we expect
strict 1e-12 (and in practice, given -ffp-contract=off in Makefile,
bit-exact equality).

Linear fixtures (12) per FIXTURE_SPEC v1.0 §5: F_02, F_04, F_06,
F_09, F_11, F_14, F_15, F_18, F_20, F_21, F_23, F_24.

Adversarial sparse fixtures get extra explicit assertions:
  F_21 — in-grid corner with >=4 out-of-grid trilinear neighbours
  F_23 — clamp-into-missing-corner returns NaN
  F_24 — two-query: midpoint renormalization AND missing-bin NaN
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

CPP_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = CPP_ROOT / "fixtures"
BUILD_DIR = CPP_ROOT / "build"
CLI_RUNNER = BUILD_DIR / "cli_runner"

LINEAR_FIXTURES = [
    "F_02_I_n_1D_X_d",
    "F_04_I_c_1D_X_d",
    "F_06_I_n_2D_X_d",
    "F_09_N_n_1D_X_d",
    "F_11_N_c_1D_X_d",
    "F_14_N_c_2D_X_d",
    "F_15_I_n_3D_X_d",
    "F_18_I_c_3D_X_d",
    "F_20_N_c_3D_X_d",
    "F_21_N_n_3D_X_s",
    "F_23_I_c_2D_X_s",
    "F_24_N_n_2D_X_s",
]

# Strict tolerance per proposal v1.1 §6.3. FP-1 not invoked unless
# (N > 10000) OR (Kahan-disabled Python ref also fails); neither applies
# to these small fixtures.
RTOL = 1e-12


@pytest.fixture(scope="session")
def cli_runner_binary() -> Path:
    """Compile cli_runner once per session (reuses Turn 3 target)."""
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


def _run_cli(binary: Path, fixture_path: Path) -> dict:
    """Invoke cli_runner, parse stdout JSON."""
    result = subprocess.run(
        [str(binary), str(fixture_path)],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"cli_runner exit={result.returncode} for {fixture_path.name}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"cli_runner stdout was not valid JSON for {fixture_path.name}:\n"
            f"{result.stdout}\nparse error: {e}"
        )


def _values_equal(expected, actual, rtol: float) -> tuple[bool, str]:
    """Compare with NaN-aware semantics; return (ok, diagnostic)."""
    e_nan = isinstance(expected, str) and expected == "NaN"
    a_nan = isinstance(actual, str) and actual == "NaN"
    if e_nan and a_nan:
        return True, "NaN match"
    if e_nan or a_nan:
        return False, f"NaN mismatch: expected={expected!r} actual={actual!r}"
    # Both finite numbers
    if expected == actual:
        return True, "bit-exact"
    denom = max(abs(expected), abs(actual), 1e-300)
    rel = abs(expected - actual) / denom
    if rel <= rtol:
        return True, f"rtol={rel:.2e}"
    return False, (
        f"rtol={rel:.2e} exceeds {rtol:.0e} "
        f"(expected={expected!r} actual={actual!r})"
    )


@pytest.mark.parametrize("fixture_id", LINEAR_FIXTURES)
def test_linear_fixture(cli_runner_binary: Path, fixture_id: str) -> None:
    fixture_path = FIXTURES_DIR / f"{fixture_id}.json"
    assert fixture_path.exists(), f"missing fixture: {fixture_path}"

    with fixture_path.open() as f:
        fixture = json.load(f)

    response = _run_cli(cli_runner_binary, fixture_path)
    assert response.get("status") == "ok", (
        f"cli_runner reported error: {response.get('error')!r}"
    )

    predictions = response["predictions"]
    expected_output = fixture["expected_output"]

    assert set(predictions.keys()) == set(expected_output.keys()), (
        "target key set mismatch"
    )

    for target in expected_output:
        exp_vec = expected_output[target]
        act_vec = predictions[target]
        assert len(act_vec) == len(exp_vec), (
            f"{target}: length {len(act_vec)} != expected {len(exp_vec)}"
        )
        for qi, (e, a) in enumerate(zip(exp_vec, act_vec)):
            ok, diag = _values_equal(e, a, RTOL)
            assert ok, (
                f"{fixture_id} target={target} query={qi}: {diag}"
            )


def test_F21_corner_with_out_of_grid_neighbours(
    cli_runner_binary: Path) -> None:
    """F_21: in-grid query near corner with >=4 trilinear neighbours
    out-of-grid; bounds='nan' excludes them; remaining valid corners
    are renormalized; result must be finite (not NaN). Sparse interior
    corner (1,1,1) is also missing — exercises both out-of-grid AND
    sparse-invalid corner handling in the same query."""
    fixture_path = FIXTURES_DIR / "F_21_N_n_3D_X_s.json"
    response = _run_cli(cli_runner_binary, fixture_path)
    assert response["status"] == "ok"
    act = response["predictions"]["y"][0]
    assert not (isinstance(act, str) and act == "NaN"), (
        "F_21 query at in-grid corner must produce finite renormalized "
        f"value, got NaN. This indicates either (a) bounds='nan' is not "
        f"excluding out-of-grid neighbours correctly, or (b) "
        f"renormalization is not happening when valid corners exist."
    )
    assert math.isfinite(act), (
        f"F_21 query result must be finite, got {act!r}"
    )


def test_F23_clamp_into_missing_returns_nan(cli_runner_binary: Path) -> None:
    """F_23: bounds='clamp' query (5,5) snaps to (2,2) which is
    sparse-invalid. Must return NaN per Safety contract — clamp does
    NOT supersede valid_mask."""
    fixture_path = FIXTURES_DIR / "F_23_I_c_2D_X_s.json"
    with fixture_path.open() as f:
        fixture = json.load(f)
    response = _run_cli(cli_runner_binary, fixture_path)
    assert response["status"] == "ok"

    # Query 0: in-grid valid -> finite
    act0 = response["predictions"]["y"][0]
    exp0 = fixture["expected_output"]["y"][0]
    assert not (isinstance(act0, str) and act0 == "NaN"), (
        f"F_23 query 0 (in-grid valid) must be finite"
    )
    ok, diag = _values_equal(exp0, act0, RTOL)
    assert ok, f"F_23 query 0: {diag}"

    # Query 1: clamps into missing corner -> NaN
    act1 = response["predictions"]["y"][1]
    assert isinstance(act1, str) and act1 == "NaN", (
        f"F_23 query 1 (clamp into missing corner) MUST return NaN — "
        f"clamp does not supersede valid_mask. Got {act1!r}. If finite, "
        f"this indicates clamp-to-nearest-valid-bin bug, a Safety "
        f"failure."
    )


def test_F24_renormalization_and_missing(cli_runner_binary: Path) -> None:
    """F_24: two queries.
      Query 0: midpoint between valid + missing corner. Must equal the
               renormalized weighted sum over the 3 valid corners.
      Query 1: query AT a missing bin. Must return NaN.
    """
    fixture_path = FIXTURES_DIR / "F_24_N_n_2D_X_s.json"
    with fixture_path.open() as f:
        fixture = json.load(f)
    response = _run_cli(cli_runner_binary, fixture_path)
    assert response["status"] == "ok"

    act = response["predictions"]["y"]
    exp = fixture["expected_output"]["y"]

    # Query 0: renormalized midpoint
    assert not (isinstance(act[0], str) and act[0] == "NaN"), (
        f"F_24 query 0 (renormalized midpoint) must be finite, got NaN. "
        f"This indicates renormalization-by-valid-corner-weight is "
        f"missing — the C++ would return raw weighted sum without "
        f"dividing by sum-of-valid-weights."
    )
    ok, diag = _values_equal(exp[0], act[0], RTOL)
    assert ok, (
        f"F_24 query 0 renormalization mismatch: {diag}. If off by ~25%, "
        f"weights may be summing to 0.75 instead of being renormalized "
        f"to 1.0."
    )

    # Query 1: missing bin -> NaN
    assert isinstance(act[1], str) and act[1] == "NaN", (
        f"F_24 query 1 (query AT missing bin) MUST return NaN. "
        f"Got {act[1]!r}."
    )
