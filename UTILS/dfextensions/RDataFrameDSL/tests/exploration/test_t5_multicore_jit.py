#!/usr/bin/env python3
"""
Phase 13.5.A - Test T5: Multicore JIT Benchmark (Critical)

Objective: Quantify JIT overhead in multicore scenarios

FROZEN RULE #1 COMPLIANCE:
This test uses NAMED FUNCTIONS, not lambdas.

Decision Gate (per spec):
- If speedup < 4× → Multicore JIT overhead is SIGNIFICANT (precompilation essential)
- If speedup > 6× → Multicore JIT overhead is acceptable (precompilation optional)
- If 4× ≤ speedup ≤ 6× → Situational (user choice)

Per Phase 13.5 v0.3 specification.
"""

import sys
import time
from datetime import datetime

RESULTS = {
    "test": "T5: Multicore JIT Benchmark",
    "timestamp": datetime.now().isoformat(),
    "status": "NOT_RUN",
    "findings": [],
    "time_single": None,
    "time_multi": None,
    "speedup": None,
    "severity": None,
    "recommendation": None,
}


def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    prefix = {"INFO": "ℹ️ ", "PASS": "✅", "FAIL": "❌", "WARN": "⚠️ ", "DATA": "📊"}
    print(f"{prefix.get(level, '')} {msg}")
    RESULTS["findings"].append(f"[{level}] {msg}")


def benchmark_jit_overhead():
    """
    Measure JIT compilation overhead with multicore RDataFrame.
    
    FROZEN RULE #1 COMPLIANCE:
    This test uses NAMED FUNCTIONS, not lambdas.
    """
    
    log("Starting T5: Multicore JIT Benchmark")
    log("FROZEN RULE #1: Using named functions (no lambdas)")
    
    try:
        import ROOT
    except ImportError:
        log("ROOT not available - cannot run this test", "FAIL")
        RESULTS["status"] = "SKIP_NO_ROOT"
        return None
    
    # Test parameters
    n_events = 1_000_000
    n_functions = 20
    n_threads = 8  # Adjust based on available cores
    
    log(f"Parameters: {n_events:,} events, {n_functions} functions, {n_threads} threads")
    
    # ==========================================================================
    # Phase 1: Single-threaded baseline
    # ==========================================================================
    log("\n" + "="*50)
    log("Phase 1: Single-threaded baseline")
    log("="*50)
    
    # Ensure single-threaded
    ROOT.DisableImplicitMT()
    
    # Pre-declare named functions (FROZEN RULE #1 compliant)
    log("Declaring named functions for single-threaded test...")
    for i in range(n_functions):
        func_code = f'''
        double benchmark_st_func_{i}(double x) {{
            return std::sqrt(x * x + {i}.0);
        }}
        '''
        ROOT.gInterpreter.Declare(func_code)
    
    # Measure single-threaded execution
    log("Running single-threaded benchmark...")
    start = time.time()
    
    df = ROOT.RDataFrame(n_events)
    df = df.Define("x", "gRandom->Gaus()")
    
    # Define columns using named functions
    for i in range(n_functions):
        df = df.Define(f"col_{i}", f"benchmark_st_func_{i}(x)")
    
    # Trigger execution
    result_single = df.Sum("col_0").GetValue()
    time_single = time.time() - start
    
    log(f"Single-threaded time: {time_single:.3f}s", "DATA")
    RESULTS["time_single"] = time_single
    
    # ==========================================================================
    # Phase 2: Multi-threaded (ImplicitMT)
    # ==========================================================================
    log("\n" + "="*50)
    log(f"Phase 2: Multi-threaded ({n_threads} threads)")
    log("="*50)
    
    # Enable multi-threading
    ROOT.EnableImplicitMT(n_threads)
    
    # Declare NEW functions for multi-threaded test
    # (Using same names would test caching, not JIT overhead)
    log("Declaring named functions for multi-threaded test...")
    for i in range(n_functions):
        func_code = f'''
        double benchmark_mt_func_{i}(double x) {{
            return std::sqrt(x * x + {i}.0);
        }}
        '''
        ROOT.gInterpreter.Declare(func_code)
    
    # Measure multi-threaded execution
    log("Running multi-threaded benchmark...")
    start = time.time()
    
    df = ROOT.RDataFrame(n_events)
    df = df.Define("x", "gRandom->Gaus()")
    
    # Define columns using named functions
    for i in range(n_functions):
        df = df.Define(f"col_{i}", f"benchmark_mt_func_{i}(x)")
    
    # Trigger execution
    result_multi = df.Sum("col_0").GetValue()
    time_multi = time.time() - start
    
    log(f"Multi-threaded time: {time_multi:.3f}s", "DATA")
    RESULTS["time_multi"] = time_multi
    
    # Disable MT for cleanup
    ROOT.DisableImplicitMT()
    
    # ==========================================================================
    # Analysis
    # ==========================================================================
    log("\n" + "="*50)
    log("Analysis")
    log("="*50)
    
    # Calculate speedup
    if time_multi > 0:
        speedup = time_single / time_multi
    else:
        speedup = float('inf')
    
    RESULTS["speedup"] = speedup
    
    log(f"Speedup: {speedup:.2f}x", "DATA")
    log(f"Theoretical max (ideal): {n_threads}x")
    log(f"Efficiency: {(speedup / n_threads) * 100:.1f}%")
    
    # Interpretation per spec
    if speedup < 4.0:
        severity = "HIGH"
        recommendation = "Precompilation is ESSENTIAL (P0)"
        log("\n⚠️  T5 FINDING: SIGNIFICANT multicore JIT overhead", "WARN")
        log(f"   Speedup < 4x ({speedup:.2f}x) indicates compilation redundancy")
        log(f"   → {recommendation}")
    elif speedup > 6.0:
        severity = "LOW"
        recommendation = "Precompilation is OPTIONAL (P1, nice-to-have)"
        log("\n✅ T5 FINDING: Multicore JIT overhead acceptable", "PASS")
        log(f"   Speedup > 6x ({speedup:.2f}x) indicates compilation is shared")
        log(f"   → {recommendation}")
    else:
        severity = "MEDIUM"
        recommendation = "Precompilation is RECOMMENDED (P1, situational)"
        log("\n⚠️  T5 FINDING: MODERATE multicore JIT overhead", "WARN")
        log(f"   Speedup 4-6x ({speedup:.2f}x) indicates partial sharing")
        log(f"   → {recommendation}")
    
    RESULTS["severity"] = severity
    RESULTS["recommendation"] = recommendation
    
    # Verify numerical consistency
    if abs(result_single) > 0 and abs(result_multi) > 0:
        ratio = result_single / result_multi
        if 0.9 < ratio < 1.1:
            log(f"\n✅ Numerical consistency verified (ratio: {ratio:.4f})", "PASS")
        else:
            log(f"\n⚠️  Numerical inconsistency detected (ratio: {ratio:.4f})", "WARN")
    
    return {
        'time_single': time_single,
        'time_multi': time_multi,
        'speedup': speedup,
        'severity': severity,
        'recommendation': recommendation,
        'n_threads': n_threads,
        'n_functions': n_functions,
        'n_events': n_events,
    }


def print_summary():
    """Print test summary."""
    print("\n" + "="*60)
    print("T5 MULTICORE JIT BENCHMARK SUMMARY")
    print("="*60)
    print(f"Status: {RESULTS['status']}")
    print(f"Timestamp: {RESULTS['timestamp']}")
    print("")
    print(f"Single-threaded: {RESULTS['time_single']:.3f}s" if RESULTS['time_single'] else "Single-threaded: N/A")
    print(f"Multi-threaded:  {RESULTS['time_multi']:.3f}s" if RESULTS['time_multi'] else "Multi-threaded: N/A")
    print(f"Speedup:         {RESULTS['speedup']:.2f}x" if RESULTS['speedup'] else "Speedup: N/A")
    print("")
    print(f"Severity: {RESULTS['severity']}")
    print(f"Recommendation: {RESULTS['recommendation']}")
    print("")
    print("Decision Criteria (per Phase 13.5 v0.3 spec):")
    print("  - If speedup < 4× → Precompilation is ESSENTIAL (P0)")
    print("  - If speedup > 6× → Precompilation is OPTIONAL (P1)")
    print("  - If 4× ≤ speedup ≤ 6× → Situational (user choice)")
    print("="*60)


if __name__ == "__main__":
    print("="*60)
    print("Phase 13.5.A - Test T5: Multicore JIT Benchmark")
    print("="*60)
    print("FROZEN RULE #1: Using named functions (no lambdas)")
    print("")
    
    results = benchmark_jit_overhead()
    
    if results:
        RESULTS["status"] = "PASS"
    else:
        RESULTS["status"] = "SKIP"
    
    print_summary()
    
    if results:
        print(f"\n✅ T5 Complete: Severity={results['severity']}")
        sys.exit(0)
    else:
        print("\n❌ T5 could not complete")
        sys.exit(1)
