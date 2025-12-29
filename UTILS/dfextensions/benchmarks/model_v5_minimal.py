#!/usr/bin/env python3
"""
Phase 12.12b.0: Minimal V5 Performance Model
Formula-based prediction with NO calibration factors.

Usage:
    PYTHONPATH=. python benchmarks/model_v5_minimal.py \
        --primitives benchmarks/primitives_v3_S4.json \
        --measured benchmarks/results/v5_scientific.json \
        --output benchmarks/results/v5_model.json

Purpose:
    Compute T_ideal from Phase 12.12a primitives and compare to measured.
    Efficiency Gap = T_measured / T_ideal (target: < 2.0)
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# Primitive Lookup Helpers
# =============================================================================

def find_primitive(benchmarks: list, primitive_id: str, impl: str = None) -> Optional[Dict]:
    """
    Find a primitive by ID and optionally implementation.
    
    Returns the benchmark entry or None if not found.
    """
    for b in benchmarks:
        bid = b.get("id", "")
        if primitive_id in bid:
            if impl is None or impl in bid:
                return b
    return None


def get_primitive_time(benchmarks: list, primitive_id: str, impl: str = None) -> Optional[float]:
    """
    Get time_per_call_ns or time_per_row_ns from a primitive.
    """
    prim = find_primitive(benchmarks, primitive_id, impl)
    if prim is None:
        return None
    
    # Try different time fields
    for field in ["time_per_call_ns", "time_per_row_ns"]:
        if field in prim:
            return prim[field]
    
    return None


# =============================================================================
# Analytical Model
# =============================================================================

def compute_t_ideal(
    primitives: Dict[str, Any],
    n_rows: int,
    n_groups: int,
    n_fits: int,
    n_params: int,
) -> Dict[str, Any]:
    """
    Compute T_ideal from primitives WITHOUT calibration factors.
    
    Formula:
        T_ideal = T_sort + T_boundaries + T_materialize + T_kernel + T_output
    
    Where:
        T_sort = N × t_S1 (argsort)
        T_boundaries = N × t_S2 (group boundaries)
        T_materialize = N × cols × 8 / BW_gather
        T_kernel = G × F × (t_C1 + t_C2 + t_C6 + t_C8)
        T_output = G × F × n_out × 8 / BW_write
    """
    benchmarks = primitives.get("benchmarks", [])
    
    # Extract primitives (prefer numba implementations)
    t_S1 = get_primitive_time(benchmarks, "S1", "numpy_argsort")  # argsort
    t_S2 = get_primitive_time(benchmarks, "S2", "numba") or get_primitive_time(benchmarks, "S2", "numpy")
    t_C1 = get_primitive_time(benchmarks, "C1", "numba") or get_primitive_time(benchmarks, "C1", "numpy")
    t_C2 = get_primitive_time(benchmarks, "C2", "numba") or get_primitive_time(benchmarks, "C2", "numpy")
    t_C6 = get_primitive_time(benchmarks, "C6", "numba") or get_primitive_time(benchmarks, "C6", "numpy")
    t_C8 = get_primitive_time(benchmarks, "C8", "numba") or get_primitive_time(benchmarks, "C8", "numpy")
    
    # Bandwidths
    gather_bw = primitives.get("summary", {}).get("gather_bandwidth_gbs", 2.0)  # GB/s
    stream_bw = primitives.get("summary", {}).get("stream_bandwidth_gbs", 40.0)  # GB/s
    
    # Check for missing primitives
    missing = []
    if t_S1 is None: missing.append("S1")
    if t_S2 is None: missing.append("S2")
    if t_C1 is None: missing.append("C1")
    if t_C2 is None: missing.append("C2")
    if t_C6 is None: missing.append("C6")
    if t_C8 is None: missing.append("C8")
    
    if missing:
        return {"error": f"Missing primitives: {missing}"}
    
    # Compute each term
    # T_sort: O(N log N), but primitive measures per-row
    T_sort_ms = (n_rows * t_S1) / 1e6
    
    # T_boundaries: O(N)
    T_boundaries_ms = (n_rows * t_S2) / 1e6
    
    # T_materialize: Read all columns via gather
    n_cols = n_params + 1 + 1 + n_fits  # X columns + y + w + targets
    bytes_to_gather = n_rows * n_cols * 8  # float64
    T_materialize_ms = (bytes_to_gather / (gather_bw * 1e9)) * 1000
    
    # T_kernel: Per-fit compute
    # Each fit: XtWX (C1) + solve (C2) + residual (C8) + MAD (C6)
    t_fit = t_C1 + t_C2 + t_C8 + t_C6  # ns per fit
    T_kernel_ms = (n_groups * n_fits * t_fit) / 1e6
    
    # T_output: Write results
    n_outputs_per_fit = n_params + 4  # beta + errors + rms + mad + diagnostics
    bytes_to_write = n_groups * n_fits * n_outputs_per_fit * 8
    # Use stream bandwidth for sequential write
    T_output_ms = (bytes_to_write / (stream_bw * 1e9)) * 1000
    
    # Total
    T_ideal_ms = T_sort_ms + T_boundaries_ms + T_materialize_ms + T_kernel_ms + T_output_ms
    
    return {
        "components": {
            "T_sort": {
                "formula": f"N × t_S1 = {n_rows} × {t_S1:.1f} ns",
                "value_ms": round(T_sort_ms, 2),
                "primitive": "S1:argsort",
            },
            "T_boundaries": {
                "formula": f"N × t_S2 = {n_rows} × {t_S2:.2f} ns",
                "value_ms": round(T_boundaries_ms, 2),
                "primitive": "S2:group_boundaries",
            },
            "T_materialize": {
                "formula": f"N × cols × 8 / BW = {n_rows} × {n_cols} × 8 / {gather_bw} GB/s",
                "value_ms": round(T_materialize_ms, 2),
                "primitive": "M2:gather",
            },
            "T_kernel": {
                "formula": f"G × F × t_fit = {n_groups} × {n_fits} × {t_fit:.1f} ns",
                "value_ms": round(T_kernel_ms, 2),
                "primitives": ["C1:XtWX", "C2:solve", "C8:residual", "C6:MAD"],
                "t_fit_ns": round(t_fit, 1),
            },
            "T_output": {
                "formula": f"G × F × n_out × 8 / BW = {n_groups} × {n_fits} × {n_outputs_per_fit} × 8 / {stream_bw} GB/s",
                "value_ms": round(T_output_ms, 2),
                "primitive": "M1:stream_write",
            },
        },
        "T_ideal_ms": round(T_ideal_ms, 2),
        "breakdown_pct": {
            "sort": round(T_sort_ms / T_ideal_ms * 100, 1),
            "boundaries": round(T_boundaries_ms / T_ideal_ms * 100, 1),
            "materialize": round(T_materialize_ms / T_ideal_ms * 100, 1),
            "kernel": round(T_kernel_ms / T_ideal_ms * 100, 1),
            "output": round(T_output_ms / T_ideal_ms * 100, 1),
        },
        "primitives_used": {
            "t_S1_ns": t_S1,
            "t_S2_ns": t_S2,
            "t_C1_ns": t_C1,
            "t_C2_ns": t_C2,
            "t_C6_ns": t_C6,
            "t_C8_ns": t_C8,
            "gather_bandwidth_gbs": gather_bw,
            "stream_bandwidth_gbs": stream_bw,
        },
    }


def compute_efficiency_gap(
    t_measured_ms: float,
    t_ideal_ms: float,
) -> Dict[str, Any]:
    """
    Compute efficiency gap = T_measured / T_ideal.
    
    Target: Gap < 2.0 (acceptable overhead for real-world code)
    """
    gap = t_measured_ms / t_ideal_ms if t_ideal_ms > 0 else float('inf')
    
    # Diagnose
    if gap < 1.5:
        diagnosis = "✅ EXCELLENT: V5 is within 50% of theoretical optimum"
    elif gap < 2.0:
        diagnosis = "✅ GOOD: V5 has acceptable overhead (<2×)"
    elif gap < 5.0:
        diagnosis = f"⚠️ MODERATE: V5 is {gap:.1f}× slower than ideal — optimization possible"
    else:
        diagnosis = f"🔴 HIGH: V5 is {gap:.1f}× slower than ideal — significant optimization needed"
    
    return {
        "T_measured_ms": round(t_measured_ms, 2),
        "T_ideal_ms": round(t_ideal_ms, 2),
        "efficiency_gap": round(gap, 2),
        "overhead_ms": round(t_measured_ms - t_ideal_ms, 2),
        "overhead_pct": round((gap - 1) * 100, 1),
        "diagnosis": diagnosis,
    }


# =============================================================================
# Main Runner
# =============================================================================

def run_model(
    primitives_path: str,
    measured_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the minimal analytical model.
    
    Parameters
    ----------
    primitives_path : str
        Path to primitives JSON (from Phase 12.12a)
    measured_path : str, optional
        Path to measured results JSON (from bench_v5_scientific.py)
    verbose : bool
        Print progress
    """
    # Load primitives
    with open(primitives_path) as f:
        primitives = json.load(f)
    
    # Extract scenario parameters
    params = primitives.get("meta", {}).get("parameters", {})
    n_rows = params.get("n_rows", 500_000)
    n_groups = params.get("n_groups", 5_000)
    n_fits = params.get("n_fits", 6)
    n_params = params.get("n_params", 3)
    scenario = primitives.get("meta", {}).get("scenario", "S4")
    
    if verbose:
        print("=" * 70)
        print("Phase 12.12b.0: Minimal V5 Performance Model")
        print("=" * 70)
        print(f"Primitives: {primitives_path}")
        print(f"Scenario: {scenario}")
        print(f"  N={n_rows:,}, G={n_groups:,}, F={n_fits}, P={n_params}")
        print()
    
    # Compute T_ideal
    model = compute_t_ideal(primitives, n_rows, n_groups, n_fits, n_params)
    
    if "error" in model:
        print(f"ERROR: {model['error']}")
        return model
    
    if verbose:
        print("Predicted Components:")
        for name, comp in model["components"].items():
            print(f"  {name:15} {comp['value_ms']:8.2f} ms  ({comp['formula']})")
        print()
        print(f"T_ideal: {model['T_ideal_ms']:.2f} ms")
        print()
        print("Breakdown:")
        for name, pct in model["breakdown_pct"].items():
            print(f"  {name:15} {pct:5.1f}%")
    
    # Build result
    result = {
        "meta": {
            "phase": "12.12b.0",
            "type": "analytical_model",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        },
        "inputs": {
            "primitives_json": primitives_path,
            "scenario": scenario,
            "parameters": params,
        },
        "model": model,
    }
    
    # Load and compare to measured if provided
    if measured_path:
        if verbose:
            print()
            print("-" * 70)
            print("Comparison to Measured")
            print("-" * 70)
        
        with open(measured_path) as f:
            measured = json.load(f)
        
        # Get V5 steady state time
        v5_steady_ms = measured.get("v5_sequential", {}).get("steady_ms")
        
        if v5_steady_ms:
            gap = compute_efficiency_gap(v5_steady_ms, model["T_ideal_ms"])
            result["efficiency_gap"] = gap
            
            if verbose:
                print(f"  T_measured (V5 steady): {gap['T_measured_ms']:.2f} ms")
                print(f"  T_ideal (model):        {gap['T_ideal_ms']:.2f} ms")
                print(f"  Efficiency Gap:         {gap['efficiency_gap']:.2f}×")
                print(f"  Overhead:               {gap['overhead_ms']:.2f} ms ({gap['overhead_pct']:.1f}%)")
                print()
                print(f"  {gap['diagnosis']}")
        else:
            if verbose:
                print("  WARNING: No v5_sequential.steady_ms found in measured data")
    
    return result


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 12.12b.0: Minimal V5 Performance Model",
    )
    parser.add_argument(
        "--primitives", "-p",
        required=True,
        help="Path to primitives JSON file (from Phase 12.12a)",
    )
    parser.add_argument(
        "--measured", "-m",
        default=None,
        help="Path to measured results JSON (from bench_v5_scientific.py)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output JSON file",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output",
    )
    
    args = parser.parse_args()
    
    result = run_model(
        primitives_path=args.primitives,
        measured_path=args.measured,
        verbose=not args.quiet,
    )
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        print(f"\nModel written to: {args.output}")
    elif args.quiet:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
