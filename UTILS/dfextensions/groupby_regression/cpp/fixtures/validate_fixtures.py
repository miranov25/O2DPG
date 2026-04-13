#!/usr/bin/env python3
"""Phase 13.18.GB Turn 2 — Fixture validator.

Parses all cpp/fixtures/F_*.json files and validates:
  1. Exactly 24 files present
  2. Each file has the 5 top-level keys: fixture_id, axis_values,
     input, expected_output, intermediates, metadata (FIXTURE_SPEC §4)
  3. No extra top-level keys (restricted-parser contract per P1-η)
  4. axis_values matches the fixture_id tokens (drift check)
  5. Axis coverage closes to 12:12 / 12:12 / 8:8:8 / 12:12 / 20:4
     (proposal v1.1 §6.1 and FIXTURE_SPEC v1.0 §2)
  6. All 24 orthogonal cells covered exactly once, no missing cells,
     no duplicated cells
  7. Each fixture file < 100 KB (FP-2 not triggered)
  8. NaN values serialized as string "NaN", not null

Usage
-----
    python cpp/fixtures/validate_fixtures.py

Exits non-zero on validation failure.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent
REQUIRED_TOP_KEYS = {"fixture_id", "axis_values", "input",
                     "expected_output", "intermediates", "metadata"}
MAX_BYTES = 100 * 1024


def parse_fixture_id(fixture_id: str) -> dict[str, str]:
    """Parse the F_NN_<fit>_<bnd>_<dim>_<method>_<cov> ID into tokens."""
    parts = fixture_id.split("_")
    if len(parts) != 7:
        raise ValueError(f"Malformed fixture_id: {fixture_id!r}")
    return {
        "_": parts[0],
        "seq": parts[1],
        "fit": parts[2],
        "bnd": parts[3],
        "dim": parts[4],
        "method": parts[5],
        "cov": parts[6],
    }


def validate_one(path: Path) -> tuple[dict[str, str], list[str]]:
    """Return (id_tokens, list_of_errors)."""
    errors: list[str] = []
    size = path.stat().st_size
    if size > MAX_BYTES:
        errors.append(f"{path.name}: size {size} > {MAX_BYTES} (FP-2 trigger)")

    with path.open() as f:
        fx = json.load(f)

    # 1 + 2: top-level keys
    top_keys = set(fx.keys())
    missing = REQUIRED_TOP_KEYS - top_keys
    extra = top_keys - REQUIRED_TOP_KEYS
    if missing:
        errors.append(f"{path.name}: missing top-level keys: {sorted(missing)}")
    if extra:
        errors.append(f"{path.name}: extra top-level keys: {sorted(extra)}")

    # 3: fixture_id matches filename stem
    stem = path.stem
    if fx.get("fixture_id") != stem:
        errors.append(f"{path.name}: fixture_id {fx.get('fixture_id')!r} != "
                      f"stem {stem!r}")

    # 4: axis_values match fixture_id tokens
    id_tokens = parse_fixture_id(stem)
    av = fx.get("axis_values", {})
    expect_fit = {"I": True, "N": False}[id_tokens["fit"]]
    expect_bnd = {"n": "nan", "c": "clamp"}[id_tokens["bnd"]]
    expect_dim = {"1D": 1, "2D": 2, "3D": 3}[id_tokens["dim"]]
    expect_method = {"L": "lookup", "X": "linear"}[id_tokens["method"]]
    expect_cov = {"d": "dense", "s": "sparse"}[id_tokens["cov"]]
    if av.get("fit_intercept") != expect_fit:
        errors.append(f"{path.name}: axis_values.fit_intercept "
                      f"{av.get('fit_intercept')!r} != id token {expect_fit}")
    if av.get("bounds") != expect_bnd:
        errors.append(f"{path.name}: axis_values.bounds "
                      f"{av.get('bounds')!r} != id token {expect_bnd!r}")
    if av.get("dimensions") != expect_dim:
        errors.append(f"{path.name}: axis_values.dimensions "
                      f"{av.get('dimensions')!r} != id token {expect_dim}")
    if av.get("method") != expect_method:
        errors.append(f"{path.name}: axis_values.method "
                      f"{av.get('method')!r} != id token {expect_method!r}")
    if av.get("coverage") != expect_cov:
        errors.append(f"{path.name}: axis_values.coverage "
                      f"{av.get('coverage')!r} != id token {expect_cov!r}")

    # 5: check expected_output NaN serialization uses "NaN" string, not null
    def check_nan(container: object, path_: str) -> None:
        if isinstance(container, dict):
            for k, v in container.items():
                check_nan(v, f"{path_}.{k}")
        elif isinstance(container, list):
            for i, v in enumerate(container):
                check_nan(v, f"{path_}[{i}]")
        elif container is None:
            errors.append(f"{path.name}: null found at {path_} — use \"NaN\"")

    check_nan(fx.get("expected_output", {}), "expected_output")

    # 6: input.method / input.bounds consistency with axis_values
    inp = fx.get("input", {})
    if inp.get("method") != expect_method:
        errors.append(f"{path.name}: input.method "
                      f"{inp.get('method')!r} != axis {expect_method!r}")
    if inp.get("bounds") != expect_bnd:
        errors.append(f"{path.name}: input.bounds "
                      f"{inp.get('bounds')!r} != axis {expect_bnd!r}")

    return id_tokens, errors


def main() -> int:
    all_errors: list[str] = []
    files = sorted(FIXTURES_DIR.glob("F_*.json"))

    if len(files) != 24:
        all_errors.append(f"expected 24 fixture files, found {len(files)}")

    per_fixture_tokens: list[dict[str, str]] = []
    for p in files:
        try:
            tokens, errs = validate_one(p)
            per_fixture_tokens.append(tokens)
            all_errors.extend(errs)
        except Exception as e:
            all_errors.append(f"{p.name}: exception during validation: {e}")

    # Axis coverage + orthogonal-cell check
    if not all_errors and per_fixture_tokens:
        fit_c = Counter(t["fit"] for t in per_fixture_tokens)
        bnd_c = Counter(t["bnd"] for t in per_fixture_tokens)
        dim_c = Counter(t["dim"] for t in per_fixture_tokens)
        mth_c = Counter(t["method"] for t in per_fixture_tokens)
        cov_c = Counter(t["cov"] for t in per_fixture_tokens)

        expected = {
            "fit_intercept (I:True, N:False) -> 12:12":
                fit_c.get("I") == 12 and fit_c.get("N") == 12,
            "bounds (n:nan, c:clamp) -> 12:12":
                bnd_c.get("n") == 12 and bnd_c.get("c") == 12,
            "dimensions (1D/2D/3D) -> 8:8:8":
                (dim_c.get("1D") == 8 and dim_c.get("2D") == 8
                 and dim_c.get("3D") == 8),
            "method (L/X) -> 12:12":
                mth_c.get("L") == 12 and mth_c.get("X") == 12,
            "coverage (d/s) -> 20:4":
                cov_c.get("d") == 20 and cov_c.get("s") == 4,
        }
        for label, ok in expected.items():
            if not ok:
                all_errors.append(f"axis coverage failed: {label} "
                                  f"(actual: fit={dict(fit_c)}, "
                                  f"bnd={dict(bnd_c)}, dim={dict(dim_c)}, "
                                  f"mth={dict(mth_c)}, cov={dict(cov_c)})")

        # Orthogonal cell check: every (fit, bnd, dim, method) cell covered
        # exactly once, ignoring coverage axis.
        cells = Counter((t["fit"], t["bnd"], t["dim"], t["method"])
                        for t in per_fixture_tokens)
        all_cells = {(f, b, d, m)
                     for f in ("I", "N") for b in ("n", "c")
                     for d in ("1D", "2D", "3D") for m in ("L", "X")}
        missing_cells = all_cells - set(cells.keys())
        dup_cells = [c for c, n in cells.items() if n > 1]
        if missing_cells:
            all_errors.append(f"orthogonal cells missing: "
                              f"{sorted(missing_cells)}")
        if dup_cells:
            all_errors.append(f"orthogonal cells duplicated: "
                              f"{sorted(dup_cells)}")

    if all_errors:
        print("VALIDATION FAILED")
        for e in all_errors:
            print(f"  {e}")
        return 1

    total_bytes = sum(p.stat().st_size for p in files)
    print(f"VALIDATION PASSED")
    print(f"  24 fixtures, total {total_bytes:,} bytes, "
          f"max {max(p.stat().st_size for p in files):,} bytes")
    print(f"  axis coverage: fit 12:12, bnd 12:12, dim 8:8:8, "
          f"method 12:12, cov 20:4")
    print(f"  all 24 orthogonal cells covered exactly once")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
