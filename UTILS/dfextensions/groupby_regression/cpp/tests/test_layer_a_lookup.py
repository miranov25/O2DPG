"""Phase 13.18.GB Turn 3 — Layer A lookup subprocess tests.

Invokes cpp/build/cli_runner as a subprocess, one fixture per test,
parses the structured JSON stdout, and asserts bit-exact parity
with each fixture's expected_output.

Lookup fixtures (12): F_01, F_03, F_05, F_07, F_08, F_10, F_12, F_13,
F_16, F_17, F_19, F_22 (per PHASE_13_18_GB_Fixture_Specification_v1.0 §5).

The compile step is a session-scoped fixture: run `make layer_a_lookup`
once; if build fails, every test errors with the compile output.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

# cpp/ importable via cpp/conftest.py
CPP_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = CPP_ROOT / "fixtures"
BUILD_DIR = CPP_ROOT / "build"
CLI_RUNNER = BUILD_DIR / "cli_runner"

LOOKUP_FIXTURES = [
    "F_01_I_n_1D_L_d",
    "F_03_I_c_1D_L_d",
    "F_05_I_n_2D_L_d",
    "F_07_I_c_2D_L_d",
    "F_08_N_n_1D_L_d",
    "F_10_N_c_1D_L_d",
    "F_12_N_n_2D_L_d",
    "F_13_N_c_2D_L_d",
    "F_16_I_c_3D_L_d",
    "F_17_N_n_3D_L_d",
    "F_19_N_c_3D_L_d",
    "F_22_I_n_3D_L_s",
]


@pytest.fixture(scope="session")
def cli_runner_binary() -> Path:
    """Compile cli_runner once per session."""
    result = subprocess.run(
        ["make", "-C", str(CPP_ROOT), "layer_a_lookup"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        pytest.fail(
            f"make layer_a_lookup failed:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
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
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        pytest.fail(
            f"cli_runner stdout was not valid JSON for "
            f"{fixture_path.name}:\n{result.stdout}\nparse error: {e}"
        )


def _values_equal(expected, actual) -> bool:
    """Compare two fixture values with explicit NaN handling."""
    e_nan = isinstance(expected, str) and expected == "NaN"
    a_nan = isinstance(actual, str) and actual == "NaN"
    if e_nan and a_nan:
        return True
    if e_nan or a_nan:
        return False
    # Float values: bit-exact integer lookup must be exactly equal
    return expected == actual


@pytest.mark.parametrize("fixture_id", LOOKUP_FIXTURES)
def test_lookup_fixture(cli_runner_binary: Path, fixture_id: str) -> None:
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

    # Same targets
    assert set(predictions.keys()) == set(expected_output.keys()), (
        f"target key set mismatch"
    )

    # Each target, each query, bit-exact (or NaN match)
    for target in expected_output:
        exp_vec = expected_output[target]
        act_vec = predictions[target]
        assert len(act_vec) == len(exp_vec), (
            f"{target}: length {len(act_vec)} != expected {len(exp_vec)}"
        )
        for qi, (e, a) in enumerate(zip(exp_vec, act_vec)):
            assert _values_equal(e, a), (
                f"{fixture_id} target={target} query={qi} "
                f"expected={e!r} actual={a!r}"
            )


def test_F22_safety_contract(cli_runner_binary: Path) -> None:
    """F_22 is the Safety-critical fixture: query at valid bin returns
    finite, query at missing bin returns NaN. Explicit assertion with
    math.isnan() semantics to catch the 'neighbour leak' bug class."""
    fixture_path = FIXTURES_DIR / "F_22_I_n_3D_L_s.json"
    with fixture_path.open() as f:
        fixture = json.load(f)

    response = _run_cli(cli_runner_binary, fixture_path)
    assert response["status"] == "ok"

    act = response["predictions"]["y"]
    exp = fixture["expected_output"]["y"]

    # Query 0: valid bin (0,0,0) — must be finite and match expected
    assert not (isinstance(act[0], str) and act[0] == "NaN"), (
        "F_22 query 0 (valid bin) must be finite, got NaN"
    )
    assert math.isfinite(act[0]), (
        f"F_22 query 0 (valid bin) must be finite, got {act[0]!r}"
    )
    assert act[0] == exp[0], (
        f"F_22 query 0 value mismatch: expected {exp[0]} actual {act[0]}"
    )

    # Query 1: missing bin (1,1,1) — must be NaN (not finite neighbour)
    assert isinstance(act[1], str) and act[1] == "NaN", (
        f"F_22 query 1 (missing bin) MUST return NaN to satisfy "
        f"Safety contract; got {act[1]!r}. If finite, this indicates "
        f"a neighbour-leak bug — catastrophic Safety failure."
    )
