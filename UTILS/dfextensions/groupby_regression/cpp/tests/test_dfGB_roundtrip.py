"""Phase 13.18.GB Turn 2 — Python-only round-trip test for dfGB_to_root.

This test is the P1-ε gate (Claude21 P1-2): if `dfGB_to_root.py` has
any dtype conversion path that loses precision, it surfaces HERE,
not misattributed to C++ port parity failures in Turn 4.

Test strategy
-------------
1. Build synthetic dfGB DataFrames covering the same axis values as the
   24-fixture test matrix.
2. dfGB_to_root -> root_to_dfGB round-trip.
3. np.array_equal on every column (NOT np.allclose — strict, bit-exact,
   catches every conversion).
4. Schema sidecar round-trip: written JSON matches read JSON exactly.
5. Unknown-suffix columns (*_err_sw, *_rmse_sw, *_n_fitted_sw) survive
   round-trip (P1-δ fixture - C++ side tolerates at load time, but
   writer must not drop them).

Run via:
    pytest cpp/tests/test_dfGB_roundtrip.py -v
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# dfGB_to_root is importable via cpp/tests/conftest.py which prepends
# cpp/ to sys.path at pytest collection time.
from dfGB_to_root import dfGB_to_root, root_to_dfGB


# ============================================================
# Fixture builders for round-trip scenarios
# ============================================================


def _minimal_dfGB_1D_intercept() -> pd.DataFrame:
    """Simplest case: 1D, fit_intercept=True, no predictors, one target."""
    return pd.DataFrame({
        "bin_x": np.array([0, 1, 2, 3], dtype=np.int64),
        "y_intercept_fit": np.array([1.0, 2.5, 4.0, 5.5], dtype=np.float64),
    })


def _dfGB_2D_with_slopes() -> pd.DataFrame:
    """2D, fit_intercept=True, 2 predictors, 1 target."""
    return pd.DataFrame({
        "bin_x": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64),
        "bin_y": np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64),
        "y_intercept_fit": np.linspace(-1.0, 1.0, 9, dtype=np.float64),
        "y_slope_x_fit": np.linspace(-0.5, 0.5, 9, dtype=np.float64),
        "y_slope_z_fit": np.linspace(-2.0, 2.0, 9, dtype=np.float64),
    })


def _dfGB_with_error_columns() -> pd.DataFrame:
    """dfGB with the unknown-suffix columns C++ Layer A must tolerate."""
    return pd.DataFrame({
        "bin_x": np.array([0, 1, 2, 3], dtype=np.int64),
        "y_intercept_fit": np.array([1.0, 2.5, 4.0, 5.5], dtype=np.float64),
        # Unknown-suffix columns per P1-δ:
        "y_intercept_err_fit": np.array([0.01, 0.02, 0.03, 0.04],
                                         dtype=np.float64),
        "y_rmse_fit": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        "y_n_fitted_fit": np.array([100, 200, 300, 400], dtype=np.int64),
    })


def _dfGB_natural_labels() -> pd.DataFrame:
    """dfGB with non-compact natural-label index columns (sector in {2,5,9})."""
    rows = []
    for sx in [2, 5, 9]:
        for sy in [0, 1, 2]:
            rows.append({
                "sector": sx, "padRow": sy,
                "y_intercept_fit": float(sx * 10 + sy),
                "y_slope_meanIDC_fit": float(sx * 0.1 + sy * 0.01),
            })
    return pd.DataFrame(rows)


# ============================================================
# Tests
# ============================================================


@pytest.fixture
def tmp_root_path(tmp_path: Path) -> Path:
    return tmp_path / "test.root"


def _roundtrip_assert_equal(df: pd.DataFrame, tmp_root_path: Path,
                            **schema: object) -> None:
    """Write df, read back, assert bit-exact equality on every column."""
    dfGB_to_root(df, tmp_root_path, **schema)
    df_back, schema_back = root_to_dfGB(tmp_root_path)

    # Columns must match (order can differ — sort both)
    assert set(df.columns) == set(df_back.columns), (
        f"column set mismatch: "
        f"original={sorted(df.columns)} vs roundtrip={sorted(df_back.columns)}")

    # Every column bit-exact
    for col in df.columns:
        a = df[col].to_numpy()
        b = df_back[col].to_numpy()
        assert a.shape == b.shape, f"{col}: shape {a.shape} != {b.shape}"
        if np.issubdtype(a.dtype, np.floating):
            # Strict: NO precision loss allowed
            assert np.array_equal(a, b, equal_nan=True), (
                f"{col}: bit-exact equality failed (float path). "
                f"original={a}, roundtrip={b}")
        else:
            assert np.array_equal(a, b), (
                f"{col}: bit-exact equality failed (integer path). "
                f"original={a}, roundtrip={b}")

    # Schema sidecar bit-exact
    expected_schema = {
        "group_columns": list(schema["group_columns"]),
        "predictor_columns": list(schema["predictor_columns"]),
        "targets": list(schema["targets"]),
        "suffix": schema["suffix"],
        "fit_intercept": bool(schema["fit_intercept"]),
    }
    assert schema_back == expected_schema, (
        f"schema roundtrip mismatch: "
        f"written={expected_schema} vs read={schema_back}")


def test_roundtrip_1D_intercept_only(tmp_root_path: Path) -> None:
    """Simplest case: 1D dfGB, fit_intercept=True, no predictors."""
    df = _minimal_dfGB_1D_intercept()
    _roundtrip_assert_equal(
        df, tmp_root_path,
        group_columns=["bin_x"], predictor_columns=[], targets=["y"],
        suffix="_fit", fit_intercept=True,
    )


def test_roundtrip_2D_with_slopes(tmp_root_path: Path) -> None:
    """2D dfGB with 2 predictors; validates multi-column float64 path."""
    df = _dfGB_2D_with_slopes()
    _roundtrip_assert_equal(
        df, tmp_root_path,
        group_columns=["bin_x", "bin_y"], predictor_columns=["x", "z"],
        targets=["y"], suffix="_fit", fit_intercept=True,
    )


def test_roundtrip_with_error_columns(tmp_root_path: Path) -> None:
    """Unknown-suffix columns (*_err_fit, *_rmse_fit, *_n_fitted_fit)
    survive round-trip exactly. P1-δ gate."""
    df = _dfGB_with_error_columns()
    _roundtrip_assert_equal(
        df, tmp_root_path,
        group_columns=["bin_x"], predictor_columns=[], targets=["y"],
        suffix="_fit", fit_intercept=True,
    )


def test_roundtrip_natural_labels(tmp_root_path: Path) -> None:
    """Non-compact natural-label index columns round-trip as integer."""
    df = _dfGB_natural_labels()
    _roundtrip_assert_equal(
        df, tmp_root_path,
        group_columns=["sector", "padRow"],
        predictor_columns=["meanIDC"], targets=["y"],
        suffix="_fit", fit_intercept=True,
    )


def test_missing_required_column_raises(tmp_root_path: Path) -> None:
    """Missing a required coefficient column raises ValueError."""
    df = pd.DataFrame({
        "bin_x": np.array([0, 1, 2], dtype=np.int64),
        # missing y_intercept_fit
    })
    with pytest.raises(ValueError, match="missing required columns"):
        dfGB_to_root(
            df, tmp_root_path,
            group_columns=["bin_x"], predictor_columns=[], targets=["y"],
            suffix="_fit", fit_intercept=True,
        )


def test_roundtrip_float32_preserves_value(tmp_root_path: Path) -> None:
    """float32 coefficients promote to float64 but preserve value exactly
    (float32 exactly representable as float64)."""
    df = pd.DataFrame({
        "bin_x": np.array([0, 1, 2], dtype=np.int32),
        "y_intercept_fit": np.array([1.5, 2.75, 3.125], dtype=np.float32),
    })
    dfGB_to_root(
        df, tmp_root_path,
        group_columns=["bin_x"], predictor_columns=[], targets=["y"],
        suffix="_fit", fit_intercept=True,
    )
    df_back, _ = root_to_dfGB(tmp_root_path)
    # Values must be the same numerically (exact, since these are
    # representable in both float32 and float64)
    assert np.array_equal(
        df["y_intercept_fit"].to_numpy().astype(np.float64),
        df_back["y_intercept_fit"].to_numpy(),
    )
