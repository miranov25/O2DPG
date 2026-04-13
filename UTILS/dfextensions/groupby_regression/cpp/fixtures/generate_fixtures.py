#!/usr/bin/env python3
"""Phase 13.18.GB Turn 2 — Fixture generator.

Produces 24 deterministic JSON fixtures at cpp/fixtures/fixture_*.json
matching PHASE_13_18_GB_Fixture_Specification_v1.0.md §3.

Design constraints
------------------
- Deterministic (seeded RNG everywhere).
- Self-contained (no dependency on the repo's groupby_regression source
  tree beyond what Turn 3 will need; this generator builds synthetic
  dfGB DataFrames directly and evaluates them via a small reference
  kernel — same math the C++ port must match).
- Reference kernel is written in pure numpy in this file and ACTS as
  the cross-language parity oracle.  The real Python evaluator in
  groupby_regression_evaluator.py has the same semantics for
  method='lookup' and method='linear' with bounds='nan' / 'clamp'; we
  intentionally reimplement the math here so the fixtures can be
  regenerated without requiring a working groupby_regression install
  in the container.
- JSON schema per FIXTURE_SPEC §4: top-level keys
  fixture_id, axis_values, input, expected_output, intermediates,
  metadata.  Nothing else.  NaN values serialized as the JSON string
  "NaN".
- Each fixture < 100 KB (FP-2 not triggered) — we use small grids
  (N_d ≤ 8 per dimension) and small numbers of query positions.

Usage
-----
    python cpp/fixtures/generate_fixtures.py

Produces 24 JSON files in the script's own directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


GENERATOR_VERSION = "0.1.0"
FIXTURES_DIR = Path(__file__).resolve().parent
TIMESTAMP_UTC = datetime.now(timezone.utc).isoformat(timespec="seconds")


# ============================================================
# Reference kernel (pure numpy) — the parity oracle for C++
# ============================================================


@dataclass
class ReferenceModel:
    """Dense N-D representation of a trained dfGB model.

    coeff_arrays[name] has shape grid_shape + ( ,) as N-D array indexed
    by per-dimension integer indices.  valid_mask is N-D bool.
    """

    group_columns: list[str]
    predictor_columns: list[str]
    targets: list[str]
    suffix: str
    fit_intercept: bool
    bin_centers: list[list[int]]  # per-dim sorted distinct integer values
    remap: list[dict[int, int]]  # per-dim natural-label -> compact index
    grid_shape: tuple[int, ...]
    coeff_arrays: dict[str, np.ndarray]  # coefficient_name -> N-D array
    valid_mask: np.ndarray  # N-D bool


def _coeff_names_for(target: str, predictor_columns: list[str],
                     suffix: str, fit_intercept: bool) -> list[str]:
    """Return the coefficient column names for one target."""
    names: list[str] = []
    if fit_intercept:
        names.append(f"{target}_intercept{suffix}")
    for pred in predictor_columns:
        names.append(f"{target}_slope_{pred}{suffix}")
    return names


def build_reference_model(
    group_columns: list[str],
    predictor_columns: list[str],
    targets: list[str],
    suffix: str,
    fit_intercept: bool,
    subframe_rows: list[dict[str, Any]],
) -> ReferenceModel:
    """Build the dense N-D in-memory representation from sparse rows.

    Mirrors the ADF bridge's §4.2 load-time transformation.
    """
    # Per-dim sorted distinct values
    bin_centers: list[list[int]] = []
    remap: list[dict[int, int]] = []
    for gc in group_columns:
        vals = sorted({int(row[gc]) for row in subframe_rows})
        bin_centers.append(vals)
        remap.append({v: i for i, v in enumerate(vals)})

    grid_shape = tuple(len(bc) for bc in bin_centers)

    # Initialise coefficient arrays and valid_mask
    coeff_arrays: dict[str, np.ndarray] = {}
    for target in targets:
        for cname in _coeff_names_for(target, predictor_columns,
                                      suffix, fit_intercept):
            coeff_arrays[cname] = np.full(grid_shape, np.nan, dtype=np.float64)

    valid_mask = np.zeros(grid_shape, dtype=bool)

    for row in subframe_rows:
        idx = tuple(remap[d][int(row[gc])] for d, gc in enumerate(group_columns))
        valid_mask[idx] = True
        for target in targets:
            for cname in _coeff_names_for(target, predictor_columns,
                                          suffix, fit_intercept):
                coeff_arrays[cname][idx] = float(row[cname])

    return ReferenceModel(
        group_columns=list(group_columns),
        predictor_columns=list(predictor_columns),
        targets=list(targets),
        suffix=suffix,
        fit_intercept=fit_intercept,
        bin_centers=bin_centers,
        remap=remap,
        grid_shape=grid_shape,
        coeff_arrays=coeff_arrays,
        valid_mask=valid_mask,
    )


def _clamp_index(i: float, n: int) -> int:
    """Clamp float/int index into [0, n-1]."""
    if i < 0:
        return 0
    if i >= n:
        return n - 1
    return int(i)


def _evaluate_lookup_one(
    model: ReferenceModel,
    position: list[int],
    predictor_values: list[float],
    bounds: str,
) -> dict[str, float]:
    """Lookup one query position.  Position is integer per group_column."""
    # Bounds handling
    idx: list[int] = []
    for d, pos_d in enumerate(position):
        n = model.grid_shape[d]
        if pos_d < 0 or pos_d >= n:
            if bounds == "nan":
                # Out of grid -> NaN for all targets
                return {t: float("nan") for t in model.targets}
            elif bounds == "clamp":
                idx.append(_clamp_index(pos_d, n))
            else:
                raise ValueError(f"Unknown bounds mode {bounds!r}")
        else:
            idx.append(int(pos_d))
    idx_t = tuple(idx)

    # valid_mask check (Safety Hard Constraint)
    if not model.valid_mask[idx_t]:
        return {t: float("nan") for t in model.targets}

    # Evaluate each target
    result: dict[str, float] = {}
    for target in model.targets:
        val = 0.0
        if model.fit_intercept:
            cname = f"{target}_intercept{model.suffix}"
            val += float(model.coeff_arrays[cname][idx_t])
        for pi, pred in enumerate(model.predictor_columns):
            cname = f"{target}_slope_{pred}{model.suffix}"
            val += float(model.coeff_arrays[cname][idx_t]) * float(predictor_values[pi])
        result[target] = val
    return result


def _evaluate_linear_one(
    model: ReferenceModel,
    position: list[float],
    predictor_values: list[float],
    bounds: str,
) -> dict[str, float]:
    """Linear (multilinear) interpolation at one query position."""
    n_dim = len(position)

    # Per-dim: compute floor/ceil + frac, applying bounds
    floors: list[int] = []
    fracs: list[float] = []
    any_nan_from_bounds = False
    for d, pos_d in enumerate(position):
        n = model.grid_shape[d]
        if bounds == "nan":
            if pos_d < 0 or pos_d > n - 1:
                any_nan_from_bounds = True
                floors.append(0)
                fracs.append(0.0)
                continue
        elif bounds == "clamp":
            if pos_d < 0:
                pos_d = 0.0
            elif pos_d > n - 1:
                pos_d = float(n - 1)
        # else: unknown bounds -> handled in lookup path
        f = math.floor(pos_d)
        if f >= n - 1:
            f = n - 2 if n >= 2 else 0
        if f < 0:
            f = 0
        frac = pos_d - f
        if frac < 0.0:
            frac = 0.0
        elif frac > 1.0:
            frac = 1.0
        floors.append(int(f))
        fracs.append(float(frac))

    if any_nan_from_bounds:
        return {t: float("nan") for t in model.targets}

    # Iterate 2^n_dim corners, accumulate weight-sum only over valid corners
    result: dict[str, float] = {t: 0.0 for t in model.targets}
    weight_sum = 0.0
    for corner in range(1 << n_dim):
        corner_idx: list[int] = []
        weight = 1.0
        for d in range(n_dim):
            # Handle 1-cell grids: if grid_shape[d] == 1 there is only one
            # cell and 'upper' does not exist.  Treat as degenerate: weight
            # from this dim is 1.0 regardless, idx is 0.
            if model.grid_shape[d] == 1:
                corner_idx.append(0)
                continue
            bit = (corner >> d) & 1
            if bit == 0:
                corner_idx.append(floors[d])
                weight *= (1.0 - fracs[d])
            else:
                corner_idx.append(floors[d] + 1)
                weight *= fracs[d]
        idx_t = tuple(corner_idx)
        if not model.valid_mask[idx_t]:
            continue
        weight_sum += weight
        for target in model.targets:
            val = 0.0
            if model.fit_intercept:
                cname = f"{target}_intercept{model.suffix}"
                val += float(model.coeff_arrays[cname][idx_t])
            for pi, pred in enumerate(model.predictor_columns):
                cname = f"{target}_slope_{pred}{model.suffix}"
                val += (float(model.coeff_arrays[cname][idx_t])
                        * float(predictor_values[pi]))
            result[target] += weight * val

    # If ALL corners were invalid, return NaN (Safety, P1-β Turn 5)
    if weight_sum == 0.0:
        return {t: float("nan") for t in model.targets}

    # Renormalize by the sum of valid-corner weights (closes F_24 case)
    for target in model.targets:
        result[target] /= weight_sum
    return result


def evaluate(
    model: ReferenceModel,
    query_positions: list[list[float]],
    predictor_values_per_query: list[list[float]],
    method: str,
    bounds: str,
) -> dict[str, list[float]]:
    """Evaluate a list of query positions.  Returns per-target lists."""
    out: dict[str, list[float]] = {t: [] for t in model.targets}
    for i, pos in enumerate(query_positions):
        pv = (predictor_values_per_query[i] if predictor_values_per_query
              else [])
        if method == "lookup":
            r = _evaluate_lookup_one(model, [int(x) for x in pos], pv, bounds)
        elif method == "linear":
            r = _evaluate_linear_one(model, [float(x) for x in pos], pv, bounds)
        else:
            raise ValueError(f"Unknown method {method!r}")
        for t in model.targets:
            out[t].append(r[t])
    return out


# ============================================================
# Fixture construction helpers
# ============================================================


def _serialize_float(v: float) -> Any:
    """NaN serialized as string "NaN" per FIXTURE_SPEC §4."""
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    return v


def _make_fixture_dict(
    fixture_id: str,
    axis_values: dict[str, Any],
    group_columns: list[str],
    predictor_columns: list[str],
    targets: list[str],
    suffix: str,
    fit_intercept: bool,
    subframe_rows: list[dict[str, Any]],
    query_positions: list[list[float]],
    predictor_values_per_query: list[list[float]],
    method: str,
    bounds: str,
    notes: str,
) -> dict[str, Any]:
    """Build a complete fixture dict ready for JSON serialization."""
    model = build_reference_model(
        group_columns, predictor_columns, targets, suffix,
        fit_intercept, subframe_rows)
    expected = evaluate(model, query_positions, predictor_values_per_query,
                        method, bounds)

    # Intermediates
    flat_mask = [bool(x) for x in model.valid_mask.flatten().tolist()]
    remap_serial = [{str(k): v for k, v in dd.items()} for dd in model.remap]

    # Sparse/dense for axis_values / metadata
    axis_values.setdefault("coverage",
                           "sparse" if not model.valid_mask.all() else "dense")

    fixture: dict[str, Any] = {
        "fixture_id": fixture_id,
        "axis_values": axis_values,
        "input": {
            "gb_columns": list(group_columns),
            "predictor_columns": list(predictor_columns),
            "targets": list(targets),
            "suffix": suffix,
            "fit_intercept": fit_intercept,
            "subframe_rows": subframe_rows,
            "query_positions": [list(map(float, p)) for p in query_positions],
            "predictor_values_per_query": [list(map(float, pv))
                                           for pv in predictor_values_per_query],
            "method": method,
            "bounds": bounds,
        },
        "expected_output": {
            t: [_serialize_float(v) for v in expected[t]]
            for t in model.targets
        },
        "intermediates": {
            "expected_bin_centers": [list(bc) for bc in model.bin_centers],
            "expected_valid_mask_flat": flat_mask,
            "expected_coefficient_shape": list(model.grid_shape),
            "expected_remap": remap_serial,
        },
        "metadata": {
            "generator_version": GENERATOR_VERSION,
            "generator_timestamp": TIMESTAMP_UTC,
            "python_evaluator_hash": "reference-kernel-v0.1.0",
            "notes": notes,
        },
    }
    return fixture


def _write_fixture(fixture: dict[str, Any]) -> Path:
    """Write fixture JSON atomically; enforce 100 KB cap."""
    path = FIXTURES_DIR / f"{fixture['fixture_id']}.json"
    text = json.dumps(fixture, indent=2, ensure_ascii=False, sort_keys=False)
    if len(text.encode("utf-8")) > 100 * 1024:
        raise RuntimeError(
            f"Fixture {fixture['fixture_id']} exceeds 100 KB — FP-2 trigger. "
            "Either shrink the fixture or invoke FP-2 compression.")
    path.write_text(text, encoding="utf-8")
    return path


# ============================================================
# Synthetic data helpers — seeded RNGs for determinism
# ============================================================


def _make_rng(seed_str: str) -> np.random.Generator:
    h = hashlib.sha256(seed_str.encode("utf-8")).digest()
    seed = int.from_bytes(h[:4], "little")
    return np.random.default_rng(seed)


def _fill_coeffs(rng: np.random.Generator, shape: tuple[int, ...],
                 scale: float = 1.0) -> np.ndarray:
    """Deterministic coefficient array filled with smooth bounded values."""
    return rng.uniform(-scale, scale, size=shape)


def _dense_subframe_rows(
    group_columns: list[str],
    grid_shape: tuple[int, ...],
    predictor_columns: list[str],
    targets: list[str],
    suffix: str,
    fit_intercept: bool,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """One row per grid cell, deterministic coefficient values."""
    coeff_names = []
    for t in targets:
        coeff_names.extend(_coeff_names_for(t, predictor_columns, suffix,
                                            fit_intercept))

    rows: list[dict[str, Any]] = []
    for flat in range(int(np.prod(grid_shape))):
        idx = np.unravel_index(flat, grid_shape)
        row: dict[str, Any] = {}
        for d, gc in enumerate(group_columns):
            row[gc] = int(idx[d])
        for cname in coeff_names:
            row[cname] = float(rng.uniform(-1.0, 1.0))
        rows.append(row)
    return rows


def _sparse_subframe_rows(
    group_columns: list[str],
    grid_shape: tuple[int, ...],
    predictor_columns: list[str],
    targets: list[str],
    suffix: str,
    fit_intercept: bool,
    rng: np.random.Generator,
    missing_cells: list[tuple[int, ...]],
) -> list[dict[str, Any]]:
    """Dense data minus the listed missing cells."""
    dense = _dense_subframe_rows(group_columns, grid_shape, predictor_columns,
                                 targets, suffix, fit_intercept, rng)
    missing_set = set(missing_cells)
    out = []
    for row in dense:
        idx = tuple(int(row[gc]) for gc in group_columns)
        if idx in missing_set:
            continue
        out.append(row)
    return out


# ============================================================
# 24 fixture definitions
# ============================================================
#
# Each function builds and writes exactly one fixture per FIXTURE_SPEC §3.
# Naming mirrors FIXTURE_SPEC identifiers.
# ============================================================


def fixture_01() -> None:  # F_01_I_n_1D_L_d
    rng = _make_rng("F_01")
    gc = ["bin_x"]
    tgt = ["y"]
    rows = _dense_subframe_rows(gc, (4,), [], tgt, "_fit", True, rng)
    qp = [[0.0], [1.0], [2.0], [3.0]]
    f = _make_fixture_dict(
        fixture_id="F_01_I_n_1D_L_d",
        axis_values={"fit_intercept": True, "bounds": "nan",
                     "dimensions": 1, "method": "lookup"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=[[]] * len(qp),
        method="lookup", bounds="nan",
        notes="Baseline 1D integer lookup with intercept.")
    _write_fixture(f)


def fixture_02() -> None:  # F_02_I_n_1D_X_d
    rng = _make_rng("F_02")
    gc = ["bin_x"]
    tgt = ["y"]
    rows = _dense_subframe_rows(gc, (4,), [], tgt, "_fit", True, rng)
    qp = [[0.0], [0.5], [1.25], [2.75], [3.0]]
    f = _make_fixture_dict(
        fixture_id="F_02_I_n_1D_X_d",
        axis_values={"fit_intercept": True, "bounds": "nan",
                     "dimensions": 1, "method": "linear"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=[[]] * len(qp),
        method="linear", bounds="nan",
        notes="1D linear interpolation with intercept; mid-point + endpoint.")
    _write_fixture(f)


def fixture_03() -> None:  # F_03_I_c_1D_L_d
    rng = _make_rng("F_03")
    gc = ["bin_x"]
    tgt = ["y"]
    rows = _dense_subframe_rows(gc, (4,), [], tgt, "_fit", True, rng)
    qp = [[-2.0], [0.0], [3.0], [7.0]]  # corner clamps on both sides
    f = _make_fixture_dict(
        fixture_id="F_03_I_c_1D_L_d",
        axis_values={"fit_intercept": True, "bounds": "clamp",
                     "dimensions": 1, "method": "lookup"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=[[]] * len(qp),
        method="lookup", bounds="clamp",
        notes="1D lookup + clamp at both edges.")
    _write_fixture(f)


def fixture_04() -> None:  # F_04_I_c_1D_X_d
    rng = _make_rng("F_04")
    gc = ["bin_x"]
    tgt = ["y"]
    rows = _dense_subframe_rows(gc, (4,), [], tgt, "_fit", True, rng)
    qp = [[-0.5], [0.0], [2.5], [3.0], [4.5]]  # clamp+linear edge behaviour
    f = _make_fixture_dict(
        fixture_id="F_04_I_c_1D_X_d",
        axis_values={"fit_intercept": True, "bounds": "clamp",
                     "dimensions": 1, "method": "linear"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=[[]] * len(qp),
        method="linear", bounds="clamp",
        notes="1D linear + clamp at edges; edge queries return edge values.")
    _write_fixture(f)


def fixture_05() -> None:  # F_05_I_n_2D_L_d
    rng = _make_rng("F_05")
    gc = ["bin_x", "bin_y"]
    tgt = ["y"]
    rows = _dense_subframe_rows(gc, (3, 4), [], tgt, "_fit", True, rng)
    qp = [[0.0, 0.0], [1.0, 2.0], [2.0, 3.0], [0.0, 3.0], [2.0, 0.0]]
    f = _make_fixture_dict(
        fixture_id="F_05_I_n_2D_L_d",
        axis_values={"fit_intercept": True, "bounds": "nan",
                     "dimensions": 2, "method": "lookup"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=[[]] * len(qp),
        method="lookup", bounds="nan",
        notes="2D lookup; asymmetric grid shape catches linearization swap.")
    _write_fixture(f)


def fixture_06() -> None:  # F_06_I_n_2D_X_d
    rng = _make_rng("F_06")
    gc = ["bin_x", "bin_y"]
    tgt = ["y"]
    rows = _dense_subframe_rows(gc, (3, 4), [], tgt, "_fit", True, rng)
    qp = [[0.0, 0.0], [1.0, 1.0], [0.5, 1.5], [1.25, 2.75], [2.0, 3.0]]
    f = _make_fixture_dict(
        fixture_id="F_06_I_n_2D_X_d",
        axis_values={"fit_intercept": True, "bounds": "nan",
                     "dimensions": 2, "method": "linear"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=[[]] * len(qp),
        method="linear", bounds="nan",
        notes="2D bilinear with integer + fractional positions.")
    _write_fixture(f)


def fixture_07() -> None:  # F_07_I_c_2D_L_d
    rng = _make_rng("F_07")
    gc = ["bin_x", "bin_y"]
    tgt = ["y"]
    rows = _dense_subframe_rows(gc, (3, 4), [], tgt, "_fit", True, rng)
    qp = [[-1.0, -1.0], [-1.0, 5.0], [5.0, -1.0], [5.0, 5.0], [1.0, 2.0]]
    f = _make_fixture_dict(
        fixture_id="F_07_I_c_2D_L_d",
        axis_values={"fit_intercept": True, "bounds": "clamp",
                     "dimensions": 2, "method": "lookup"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=[[]] * len(qp),
        method="lookup", bounds="clamp",
        notes="2D lookup + simultaneous clamp on both dims (per-dim leak).")
    _write_fixture(f)


def fixture_08() -> None:  # F_08_N_n_1D_L_d  (fit_intercept=False path)
    rng = _make_rng("F_08")
    gc = ["bin_x"]
    tgt = ["y"]
    pred = ["x"]
    rows = _dense_subframe_rows(gc, (4,), pred, tgt, "_fit", False, rng)
    qp = [[0.0], [1.0], [2.0], [3.0]]
    pv = [[0.5], [1.5], [-1.0], [2.0]]  # predictor values per query
    f = _make_fixture_dict(
        fixture_id="F_08_N_n_1D_L_d",
        axis_values={"fit_intercept": False, "bounds": "nan",
                     "dimensions": 1, "method": "lookup"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="lookup", bounds="nan",
        notes="fit_intercept=False + 1 predictor; no intercept column.")
    _write_fixture(f)


def fixture_09() -> None:  # F_09_N_n_1D_X_d
    rng = _make_rng("F_09")
    gc = ["bin_x"]
    tgt = ["y"]
    pred = ["x"]
    rows = _dense_subframe_rows(gc, (4,), pred, tgt, "_fit", False, rng)
    qp = [[0.0], [0.5], [1.25], [2.75]]
    pv = [[0.5], [1.0], [1.5], [2.0]]
    f = _make_fixture_dict(
        fixture_id="F_09_N_n_1D_X_d",
        axis_values={"fit_intercept": False, "bounds": "nan",
                     "dimensions": 1, "method": "linear"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="linear", bounds="nan",
        notes="fit_intercept=False + linear; composition of both paths.")
    _write_fixture(f)


def fixture_10() -> None:  # F_10_N_c_1D_L_d
    rng = _make_rng("F_10")
    gc = ["bin_x"]
    tgt = ["y"]
    pred = ["x"]
    rows = _dense_subframe_rows(gc, (4,), pred, tgt, "_fit", False, rng)
    qp = [[-1.0], [0.0], [3.0], [6.0]]
    pv = [[0.5], [1.0], [1.5], [2.0]]
    f = _make_fixture_dict(
        fixture_id="F_10_N_c_1D_L_d",
        axis_values={"fit_intercept": False, "bounds": "clamp",
                     "dimensions": 1, "method": "lookup"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="lookup", bounds="clamp",
        notes="fit_intercept=False + clamp.")
    _write_fixture(f)


def fixture_11() -> None:  # F_11_N_c_1D_X_d
    rng = _make_rng("F_11")
    gc = ["bin_x"]
    tgt = ["y"]
    pred = ["x"]
    rows = _dense_subframe_rows(gc, (4,), pred, tgt, "_fit", False, rng)
    qp = [[-0.5], [0.0], [2.5], [3.5]]
    pv = [[0.5], [1.0], [1.5], [2.0]]
    f = _make_fixture_dict(
        fixture_id="F_11_N_c_1D_X_d",
        axis_values={"fit_intercept": False, "bounds": "clamp",
                     "dimensions": 1, "method": "linear"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="linear", bounds="clamp",
        notes="fit_intercept=False + linear + clamp (triple-interaction).")
    _write_fixture(f)


def fixture_12() -> None:  # F_12_N_n_2D_L_d
    rng = _make_rng("F_12")
    gc = ["bin_x", "bin_y"]
    tgt = ["y"]
    pred = ["x", "z"]
    rows = _dense_subframe_rows(gc, (3, 4), pred, tgt, "_fit", False, rng)
    qp = [[0.0, 0.0], [1.0, 2.0], [2.0, 3.0]]
    pv = [[0.5, -0.3], [1.0, 0.7], [-1.0, 1.5]]
    f = _make_fixture_dict(
        fixture_id="F_12_N_n_2D_L_d",
        axis_values={"fit_intercept": False, "bounds": "nan",
                     "dimensions": 2, "method": "lookup"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="lookup", bounds="nan",
        notes="2D fit_intercept=False + 2 predictors; slope-column mapping.")
    _write_fixture(f)


def fixture_13() -> None:  # F_13_N_c_2D_L_d
    rng = _make_rng("F_13")
    gc = ["bin_x", "bin_y"]
    tgt = ["y"]
    pred = ["x"]
    rows = _dense_subframe_rows(gc, (3, 4), pred, tgt, "_fit", False, rng)
    qp = [[-1.0, -1.0], [5.0, 5.0], [1.0, 2.0]]
    pv = [[0.5], [1.0], [-0.5]]
    f = _make_fixture_dict(
        fixture_id="F_13_N_c_2D_L_d",
        axis_values={"fit_intercept": False, "bounds": "clamp",
                     "dimensions": 2, "method": "lookup"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="lookup", bounds="clamp",
        notes="2D fit_intercept=False + clamp.")
    _write_fixture(f)


def fixture_14() -> None:  # F_14_N_c_2D_X_d  (NEW — Option B)
    rng = _make_rng("F_14")
    gc = ["bin_x", "bin_y"]
    tgt = ["y"]
    pred = ["x"]
    rows = _dense_subframe_rows(gc, (3, 4), pred, tgt, "_fit", False, rng)
    qp = [[-0.5, -0.5], [2.5, 3.5], [1.5, 2.0], [3.2, 3.5]]
    pv = [[0.5], [1.0], [-0.5], [2.0]]
    f = _make_fixture_dict(
        fixture_id="F_14_N_c_2D_X_d",
        axis_values={"fit_intercept": False, "bounds": "clamp",
                     "dimensions": 2, "method": "linear"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="linear", bounds="clamp",
        notes="2D fit_intercept=False + clamp + linear (quadruple interaction).")
    _write_fixture(f)


def fixture_15() -> None:  # F_15_I_n_3D_X_d
    rng = _make_rng("F_15")
    gc = ["bin_x", "bin_y", "bin_z"]
    tgt = ["y"]
    rows = _dense_subframe_rows(gc, (3, 3, 3), [], tgt, "_fit", True, rng)
    qp = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 1.5, 2.0], [2.0, 2.0, 2.0]]
    f = _make_fixture_dict(
        fixture_id="F_15_I_n_3D_X_d",
        axis_values={"fit_intercept": True, "bounds": "nan",
                     "dimensions": 3, "method": "linear"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=[[]] * len(qp),
        method="linear", bounds="nan",
        notes="3D trilinear with intercept; 8-corner weighted sum.")
    _write_fixture(f)


def fixture_16() -> None:  # F_16_I_c_3D_L_d
    rng = _make_rng("F_16")
    gc = ["bin_x", "bin_y", "bin_z"]
    tgt = ["y"]
    rows = _dense_subframe_rows(gc, (3, 3, 3), [], tgt, "_fit", True, rng)
    qp = [[-1.0, -1.0, -1.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0]]
    f = _make_fixture_dict(
        fixture_id="F_16_I_c_3D_L_d",
        axis_values={"fit_intercept": True, "bounds": "clamp",
                     "dimensions": 3, "method": "lookup"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=[[]] * len(qp),
        method="lookup", bounds="clamp",
        notes="3D lookup + clamp in all three dimensions simultaneously.")
    _write_fixture(f)


def fixture_17() -> None:  # F_17_N_n_3D_L_d
    rng = _make_rng("F_17")
    gc = ["sector", "padRow", "zBin"]  # non-compact natural labels test
    tgt = ["y"]
    pred = ["meanIDC", "t"]
    # Build dense rows with NON-COMPACT natural labels to exercise remap
    rows: list[dict[str, Any]] = []
    natural_x = [2, 5, 9]  # sector labels with gaps
    natural_y = [0, 1, 2]
    natural_z = [10, 20]
    for sx in natural_x:
        for sy in natural_y:
            for sz in natural_z:
                row = {"sector": sx, "padRow": sy, "zBin": sz}
                for cname in _coeff_names_for("y", pred, "_fit", False):
                    row[cname] = float(rng.uniform(-1.0, 1.0))
                rows.append(row)
    qp = [[2.0, 0.0, 10.0], [5.0, 1.0, 20.0], [9.0, 2.0, 10.0]]
    pv = [[0.3, 0.7], [-0.5, 1.2], [1.0, -0.4]]
    # Translate natural-label query positions to compact-index positions
    # for the reference kernel (which expects compact indices).
    remap_x = {v: i for i, v in enumerate(natural_x)}
    remap_y = {v: i for i, v in enumerate(natural_y)}
    remap_z = {v: i for i, v in enumerate(natural_z)}
    qp_compact = [[float(remap_x[int(p[0])]),
                   float(remap_y[int(p[1])]),
                   float(remap_z[int(p[2])])] for p in qp]
    f = _make_fixture_dict(
        fixture_id="F_17_N_n_3D_L_d",
        axis_values={"fit_intercept": False, "bounds": "nan",
                     "dimensions": 3, "method": "lookup"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp_compact,
        predictor_values_per_query=pv,
        method="lookup", bounds="nan",
        notes="3D fit_intercept=False lookup; NON-COMPACT natural labels "
              "exercise the remap builder (sector in {2,5,9}).")
    _write_fixture(f)


def fixture_18() -> None:  # F_18_I_c_3D_X_d
    rng = _make_rng("F_18")
    gc = ["bin_x", "bin_y", "bin_z"]
    tgt = ["y"]
    rows = _dense_subframe_rows(gc, (3, 3, 3), [], tgt, "_fit", True, rng)
    qp = [[0.5, 0.5, 0.5], [-1.0, -1.0, -1.0], [3.5, 3.5, 3.5],
          [1.5, 1.5, 1.5]]
    f = _make_fixture_dict(
        fixture_id="F_18_I_c_3D_X_d",
        axis_values={"fit_intercept": True, "bounds": "clamp",
                     "dimensions": 3, "method": "linear"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=[[]] * len(qp),
        method="linear", bounds="clamp",
        notes="3D linear + clamp; performance canary.")
    _write_fixture(f)


def fixture_19() -> None:  # F_19_N_c_3D_L_d  (NEW — Option B)
    rng = _make_rng("F_19")
    gc = ["bin_x", "bin_y", "bin_z"]
    tgt = ["y"]
    pred = ["x"]
    rows = _dense_subframe_rows(gc, (3, 3, 3), pred, tgt, "_fit", False, rng)
    qp = [[-1.0, -1.0, -1.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0]]
    pv = [[0.5], [1.0], [-0.5]]
    f = _make_fixture_dict(
        fixture_id="F_19_N_c_3D_L_d",
        axis_values={"fit_intercept": False, "bounds": "clamp",
                     "dimensions": 3, "method": "lookup"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="lookup", bounds="clamp",
        notes="3D fit_intercept=False + clamp lookup.")
    _write_fixture(f)


def fixture_20() -> None:  # F_20_N_c_3D_X_d  (NEW — Option B, hardest dense)
    rng = _make_rng("F_20")
    gc = ["bin_x", "bin_y", "bin_z"]
    tgt = ["y"]
    pred = ["x"]
    rows = _dense_subframe_rows(gc, (3, 3, 3), pred, tgt, "_fit", False, rng)
    qp = [[0.5, 0.5, 0.5], [-1.0, -1.0, -1.0], [3.5, 3.5, 3.5]]
    pv = [[0.5], [1.0], [-0.5]]
    f = _make_fixture_dict(
        fixture_id="F_20_N_c_3D_X_d",
        axis_values={"fit_intercept": False, "bounds": "clamp",
                     "dimensions": 3, "method": "linear"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="linear", bounds="clamp",
        notes="3D fit_intercept=False + clamp + linear — hardest dense.")
    _write_fixture(f)


# ---------- sparse fixtures ----------


def fixture_21() -> None:  # F_21_N_n_3D_X_s
    """F_adv_1 — query in-grid near corner; ≥4 trilinear neighbours out-of-grid
    because the query lies at a corner cell; remaining interior corners may
    be sparse-invalid."""
    rng = _make_rng("F_21")
    gc = ["bin_x", "bin_y", "bin_z"]
    tgt = ["y"]
    pred = ["x"]
    # 3x3x3 grid; query near corner (0.5, 0.5, 0.5) places it between
    # corners {0,1}^3. Mark one interior-adjacent corner missing.
    rows = _sparse_subframe_rows(gc, (3, 3, 3), pred, tgt, "_fit", False, rng,
                                 missing_cells=[(1, 1, 1)])
    qp = [[0.5, 0.5, 0.5]]
    pv = [[0.5]]
    f = _make_fixture_dict(
        fixture_id="F_21_N_n_3D_X_s",
        axis_values={"fit_intercept": False, "bounds": "nan",
                     "dimensions": 3, "method": "linear"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="linear", bounds="nan",
        notes="F_adv_1 — query in-grid near corner; interior neighbour "
              "(1,1,1) missing; renormalization over remaining 7 corners.")
    _write_fixture(f)


def fixture_22() -> None:  # F_22_I_n_3D_L_s
    """F_adv_2 — two query positions: populated-adjacent-to-missing (finite),
    missing (NaN)."""
    rng = _make_rng("F_22")
    gc = ["bin_x", "bin_y", "bin_z"]
    tgt = ["y"]
    # 3x3x3 with missing cluster at (1,1,1) and (1,1,2)
    rows = _sparse_subframe_rows(gc, (3, 3, 3), [], tgt, "_fit", True, rng,
                                 missing_cells=[(1, 1, 1), (1, 1, 2)])
    qp = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]  # valid, missing
    pv = [[], []]
    f = _make_fixture_dict(
        fixture_id="F_22_I_n_3D_L_s",
        axis_values={"fit_intercept": True, "bounds": "nan",
                     "dimensions": 3, "method": "lookup"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="lookup", bounds="nan",
        notes="F_adv_2 — two assertions: (0,0,0) finite, (1,1,1) NaN.")
    _write_fixture(f)


def fixture_23() -> None:  # F_23_I_c_2D_X_s
    """F_adv_3 — clamp-into-missing-bin: out-of-grid query clamped INTO a
    sparse-invalid cell must return NaN, not clamp-to-nearest-valid."""
    rng = _make_rng("F_23")
    gc = ["bin_x", "bin_y"]
    tgt = ["y"]
    # 3x3 grid. Mark the corner cell (2,2) as missing. Query at (5,5)
    # clamps to (2,2) -> must return NaN.
    rows = _sparse_subframe_rows(gc, (3, 3), [], tgt, "_fit", True, rng,
                                 missing_cells=[(2, 2)])
    qp = [[1.0, 1.0], [5.0, 5.0]]  # valid interior, clamp-into-missing
    pv = [[], []]
    f = _make_fixture_dict(
        fixture_id="F_23_I_c_2D_X_s",
        axis_values={"fit_intercept": True, "bounds": "clamp",
                     "dimensions": 2, "method": "linear"},
        group_columns=gc, predictor_columns=[], targets=tgt,
        suffix="_fit", fit_intercept=True, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="linear", bounds="clamp",
        notes="F_adv_3 — clamp destination is sparse-invalid; must return NaN.")
    _write_fixture(f)


def fixture_24() -> None:  # F_24_N_n_2D_X_s
    """F_adv_4 — two query positions: missing-bin-neighbour midpoint
    (renormalized finite), missing-bin (NaN)."""
    rng = _make_rng("F_24")
    gc = ["bin_x", "bin_y"]
    tgt = ["y"]
    pred = ["x"]
    # 3x3 grid; mark (1,1) missing
    rows = _sparse_subframe_rows(gc, (3, 3), pred, tgt, "_fit", False, rng,
                                 missing_cells=[(1, 1)])
    qp = [[0.5, 0.5], [1.0, 1.0]]  # midpoint between valid + missing, missing
    pv = [[0.5], [1.0]]
    f = _make_fixture_dict(
        fixture_id="F_24_N_n_2D_X_s",
        axis_values={"fit_intercept": False, "bounds": "nan",
                     "dimensions": 2, "method": "linear"},
        group_columns=gc, predictor_columns=pred, targets=tgt,
        suffix="_fit", fit_intercept=False, subframe_rows=rows,
        query_positions=qp, predictor_values_per_query=pv,
        method="linear", bounds="nan",
        notes="F_adv_4 — (0.5,0.5) renormalized finite; (1.0,1.0) NaN.")
    _write_fixture(f)


# ============================================================
# Main
# ============================================================


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    # Clean old fixture files
    for old in FIXTURES_DIR.glob("F_*.json"):
        old.unlink()

    fixture_fns = [
        fixture_01, fixture_02, fixture_03, fixture_04, fixture_05,
        fixture_06, fixture_07, fixture_08, fixture_09, fixture_10,
        fixture_11, fixture_12, fixture_13, fixture_14, fixture_15,
        fixture_16, fixture_17, fixture_18, fixture_19, fixture_20,
        fixture_21, fixture_22, fixture_23, fixture_24,
    ]
    for fn in fixture_fns:
        fn()

    # Verify
    produced = sorted(FIXTURES_DIR.glob("F_*.json"))
    if len(produced) != 24:
        print(f"ERROR: produced {len(produced)} fixtures, expected 24",
              file=sys.stderr)
        return 1

    total_bytes = sum(p.stat().st_size for p in produced)
    max_bytes = max(p.stat().st_size for p in produced)
    print(f"Generated 24 fixtures in {FIXTURES_DIR}")
    print(f"  Total size:   {total_bytes:,} bytes")
    print(f"  Max fixture:  {max_bytes:,} bytes (cap 102,400)")
    print(f"  Timestamp:    {TIMESTAMP_UTC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
