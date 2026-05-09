#!/usr/bin/env python
"""Phase 13.22.GB-RooflineTier1 — Baseline Consolidation Script (D4).

Reads per-machine primitive measurements and K-measured values from
Tier 2 benchmark outputs, derives K thresholds using the architect-
directed formula:

    K_floor = max(K_median[alma2], K_median[ccsub0001])
    K_threshold[machine] = K_floor + 6 × 1.4826 × MAD(K[machine])

Outputs: benchmarks/baselines/roofline_baseline.json

Usage:
    python benchmarks/scripts/update_roofline_baseline.py \\
        --primitives bench_out/primitives_alma2.json bench_out/primitives_ccsub0001.json \\
        --k-values bench_out/roofline_K_alma2.json bench_out/roofline_K_ccsub0001.json \\
        --output benchmarks/baselines/roofline_baseline.json \\
        --phase 13.22.GB-RooflineTier1

Threshold derivation per AI_Review_Scientific_Methodology v1.0-FINAL
§ The Correctness Anchors / 3. Hardware-Limit Performance:
"the bound must be known."
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def _sigma_from_mad(mad_value):
    """Convert MAD to Gaussian-equivalent sigma: σ = 1.4826 × MAD."""
    return 1.4826 * mad_value


def derive_thresholds(machines_data, test_names):
    """Derive K thresholds from per-machine K measurements.

    Formula (architect directive 2026-05-08):
        K_floor = max(K_median across machines)
        K_threshold[machine] = K_floor + 6 × σ_K[machine]
        where σ_K = 1.4826 × MAD(K)
    """
    thresholds = {}
    for test_name in test_names:
        # Collect K_median from all machines
        machine_K = {}
        for machine_name, mdata in machines_data.items():
            k_entry = mdata.get("test_K_measured", {}).get(test_name)
            if k_entry is None:
                continue
            machine_K[machine_name] = k_entry

        if not machine_K:
            continue

        # Dispatch count tests don't use K-roofline formula
        if "dispatch_count" in test_name:
            thresholds[test_name] = {
                "type": "dispatch_count",
                "threshold": 1000,
                "derivation": "Hard limit: ≤1000 np.median calls (was 1.26M pre-Phase 13.21 F3)"
            }
            continue

        # Structural tests don't have thresholds
        if "kernel_present" in test_name or "self_consistent" in test_name:
            thresholds[test_name] = {
                "type": "structural",
                "derivation": "Boolean pass/fail, no K threshold"
            }
            continue

        # K-roofline formula
        k_medians = {m: d["K_median"] for m, d in machine_K.items()}
        k_floor = max(k_medians.values())

        threshold_entry = {
            "K_floor": round(k_floor, 1),
            "derivation_formula": "K_floor + 6 × 1.4826 × MAD(K)",
        }

        for machine_name, k_entry in machine_K.items():
            k_mad = k_entry.get("K_mad", 0)
            sigma_k = _sigma_from_mad(k_mad)
            k_thresh = k_floor + 6 * sigma_k
            threshold_entry[f"K_threshold_{machine_name}"] = round(k_thresh, 1)
            threshold_entry[f"sigma_{machine_name}"] = round(sigma_k, 2)
            threshold_entry[f"K_median_{machine_name}"] = round(k_medians[machine_name], 1)

        thresholds[test_name] = threshold_entry

    return thresholds


def main():
    parser = argparse.ArgumentParser(description="Consolidate roofline baseline")
    parser.add_argument("--primitives", nargs="+", required=True,
                        help="Per-machine primitive JSON files")
    parser.add_argument("--k-values", nargs="+", required=True,
                        help="Per-machine K-measured JSON files")
    parser.add_argument("--output", type=str,
                        default="benchmarks/baselines/roofline_baseline.json")
    parser.add_argument("--phase", type=str, default="13.22.GB-RooflineTier1")
    args = parser.parse_args()

    # Load per-machine data
    machines = {}

    # Load primitives
    for prim_path in args.primitives:
        with open(prim_path) as f:
            pdata = json.load(f)
        machine_name = pdata["machine"]
        if machine_name not in machines:
            machines[machine_name] = {}
        machines[machine_name]["primitives"] = pdata["primitives"]
        machines[machine_name]["cpu"] = pdata.get("cpu", "unknown")
        machines[machine_name]["python_version"] = pdata.get("python_version", "unknown")
        machines[machine_name]["numpy_version"] = pdata.get("numpy_version", "unknown")
        machines[machine_name]["numba_version"] = pdata.get("numba_version", "unknown")
        machines[machine_name]["numba_threading_layer"] = pdata.get("numba_threading_layer", "unknown")
        machines[machine_name]["numba_num_threads"] = pdata.get("numba_num_threads", 1)
        machines[machine_name]["calibration_date"] = pdata.get("timestamp", "unknown")

    # Load K-measured values + fixture metadata + threading config
    for k_path in args.k_values:
        with open(k_path) as f:
            kdata = json.load(f)
        machine_name = kdata["machine"]
        if machine_name not in machines:
            machines[machine_name] = {}
        machines[machine_name]["test_K_measured"] = kdata["test_K_measured"]
        # Fix 5: propagate fixture metadata per v1.10 §4.6 schema
        if "fixture_SW2D" in kdata:
            machines[machine_name]["operation_counts_SW2D"] = kdata["fixture_SW2D"]
        if "fixture_S2" in kdata:
            machines[machine_name]["operation_counts_S2"] = kdata["fixture_S2"]
        # v1.12: propagate threading config for meta-test
        if "numba_threading_layer" in kdata:
            machines[machine_name]["numba_threading_layer"] = kdata["numba_threading_layer"]
        if "numba_num_threads" in kdata:
            machines[machine_name]["numba_num_threads"] = kdata["numba_num_threads"]

    # Collect all test names
    all_test_names = set()
    for mdata in machines.values():
        all_test_names |= set(mdata.get("test_K_measured", {}).keys())

    # Derive thresholds
    thresholds = derive_thresholds(machines, sorted(all_test_names))

    # Build output
    baseline = {
        "schema_version": "1.0",
        "phase_anchored": args.phase,
        "calibration_date": datetime.now().strftime("%Y-%m-%d"),
        "commands": {
            "primitive_calibration": "python benchmarks/scripts/measure_primitives.py",
            "baseline_consolidation": (
                f"python benchmarks/scripts/update_roofline_baseline.py "
                f"--primitives {' '.join(args.primitives)} "
                f"--k-values {' '.join(args.k_values)} "
                f"--output {args.output}"
            ),
        },
        "machines": machines,
        "thresholds": thresholds,
    }

    # Write
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(baseline, f, indent=2)

    print(f"Baseline written to {out_path}")
    print(f"  Machines: {', '.join(sorted(machines.keys()))}")
    print(f"  Tests: {len(thresholds)}")
    print(f"  Phase: {args.phase}")

    # Summary table
    print(f"\n{'Test':<45} {'K_floor':>8} ", end="")
    for m in sorted(machines.keys()):
        print(f"{'K_thresh_' + m:>18}", end=" ")
    print()
    for test_name, t in sorted(thresholds.items()):
        if "K_floor" not in t:
            continue
        print(f"  {test_name:<43} {t['K_floor']:>8.1f} ", end="")
        for m in sorted(machines.keys()):
            key = f"K_threshold_{m}"
            if key in t:
                print(f"{t[key]:>18.1f}", end=" ")
            else:
                print(f"{'N/A':>18}", end=" ")
        print()


if __name__ == "__main__":
    main()
