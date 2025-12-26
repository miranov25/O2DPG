"""
Phase 12.12a: Scaling & Contention Primitives

Additional primitives to diagnose parallel scaling failures.

Primitives:
- FS1: false_sharing_adjacent — Threads write to adjacent memory (contention)
- FS2: false_sharing_padded — Threads write with cache-line padding (baseline)
- M2s: gather_scaling — Gather bandwidth vs thread count
- M3s: scatter_reduce_scaling — Scatter bandwidth vs thread count
- SCH1: prange_grain_sweep — Minimum work per iteration for speedup
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

try:
    import numba
    from numba import njit, prange, set_num_threads, get_num_threads
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class ScalingResult:
    """Result from a scaling benchmark."""
    primitive_id: str
    name: str
    implementation: str
    size: int
    thread_counts: List[int]
    times_s: List[float]
    speedups: List[float]
    bandwidths_gbs: List[float]
    status: str
    diagnosis: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "id": f"primitive:{self.primitive_id}:{self.name}:{self.implementation}",
            "name": self.name,
            "implementation": self.implementation,
            "size": self.size,
            "thread_counts": self.thread_counts,
            "times_s": [round(t, 6) for t in self.times_s],
            "speedups": [round(s, 2) for s in self.speedups],
            "bandwidths_gbs": [round(b, 2) for b in self.bandwidths_gbs],
            "status": self.status,
            "diagnosis": self.diagnosis,
        }


def _run_timed(func, *args, warmup: int = 2, runs: int = 5, **kwargs) -> Tuple[float, float]:
    """Run function with warmup and return (mean_time, std_time) in seconds."""
    for _ in range(warmup):
        func(*args, **kwargs)
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times)


# =============================================================================
# FS1: False Sharing — Adjacent Writes (Contention Expected)
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _false_sharing_adjacent(out: np.ndarray, n_iters: int) -> None:
        """
        Each thread writes to adjacent elements — causes false sharing.
        
        Threads will contend for the same cache lines, causing
        invalidation traffic and destroying parallel speedup.
        """
        n = len(out)
        for i in prange(n):
            for _ in range(n_iters):
                out[i] += 1.0


# =============================================================================
# FS2: False Sharing — Padded Writes (No Contention)
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _false_sharing_padded(out: np.ndarray, n_iters: int, stride: int) -> None:
        """
        Each thread writes to memory locations separated by stride.
        
        With stride >= 8 (64 bytes = cache line), threads use
        different cache lines, eliminating false sharing.
        """
        n = len(out) // stride
        for i in prange(n):
            idx = i * stride
            for _ in range(n_iters):
                out[idx] += 1.0


def bench_false_sharing(
    n_elements: int = 1000,
    n_iters: int = 10000,
    thread_counts: List[int] = None,
    runs: int = 5,
) -> Tuple[ScalingResult, ScalingResult]:
    """
    FS1/FS2: Measure false sharing impact on parallel scaling.
    
    Returns (adjacent_result, padded_result).
    
    If padded scales well but adjacent doesn't → false sharing is the problem.
    """
    if not NUMBA_AVAILABLE:
        empty = ScalingResult(
            "FS1", "false_sharing", "numba", 0, [], [], [], [],
            "ERROR:NUMBA_NOT_AVAILABLE"
        )
        return empty, empty
    
    if thread_counts is None:
        max_threads = get_num_threads()
        thread_counts = [1, 2, 4, min(8, max_threads), max_threads]
        thread_counts = sorted(set(thread_counts))
    
    # Allocate arrays
    out_adjacent = np.zeros(n_elements, dtype=np.float64)
    out_padded = np.zeros(n_elements * 16, dtype=np.float64)  # 16 = stride for 128 bytes
    
    # Warmup
    original_threads = get_num_threads()
    set_num_threads(1)
    _false_sharing_adjacent(out_adjacent, 10)
    _false_sharing_padded(out_padded, 10, 16)
    
    # Measure adjacent (FS1)
    adjacent_times = []
    for t in thread_counts:
        set_num_threads(t)
        out_adjacent[:] = 0
        mean_time, _ = _run_timed(_false_sharing_adjacent, out_adjacent, n_iters, warmup=1, runs=runs)
        adjacent_times.append(mean_time)
    
    # Measure padded (FS2)
    padded_times = []
    for t in thread_counts:
        set_num_threads(t)
        out_padded[:] = 0
        mean_time, _ = _run_timed(_false_sharing_padded, out_padded, n_iters, 16, warmup=1, runs=runs)
        padded_times.append(mean_time)
    
    # Restore threads
    set_num_threads(original_threads)
    
    # Calculate speedups
    adjacent_speedups = [adjacent_times[0] / t for t in adjacent_times]
    padded_speedups = [padded_times[0] / t for t in padded_times]
    
    # Diagnose
    max_adjacent_speedup = max(adjacent_speedups)
    max_padded_speedup = max(padded_speedups)
    
    if max_padded_speedup > 2 * max_adjacent_speedup:
        diagnosis = f"❌ FALSE SHARING DETECTED: Padded {max_padded_speedup:.1f}× vs Adjacent {max_adjacent_speedup:.1f}×"
    elif max_adjacent_speedup < 1.5:
        diagnosis = f"⚠️ NO SCALING: Both patterns show <1.5× speedup"
    else:
        diagnosis = f"✅ OK: Both patterns scale similarly"
    
    adjacent_result = ScalingResult(
        primitive_id="FS1",
        name="false_sharing_adjacent",
        implementation="numba_parallel",
        size=n_elements,
        thread_counts=thread_counts,
        times_s=adjacent_times,
        speedups=adjacent_speedups,
        bandwidths_gbs=[0.0] * len(thread_counts),  # Not bandwidth-focused
        status="OK",
        diagnosis=diagnosis,
    )
    
    padded_result = ScalingResult(
        primitive_id="FS2",
        name="false_sharing_padded",
        implementation="numba_parallel",
        size=n_elements,
        thread_counts=thread_counts,
        times_s=padded_times,
        speedups=padded_speedups,
        bandwidths_gbs=[0.0] * len(thread_counts),
        status="OK",
        diagnosis=diagnosis,
    )
    
    return adjacent_result, padded_result


# =============================================================================
# M2s: Gather Scaling — Bandwidth vs Thread Count
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _gather_parallel(data: np.ndarray, perm: np.ndarray, out: np.ndarray) -> None:
        """Parallel gather: out[i] = data[perm[i]]."""
        n = len(perm)
        for i in prange(n):
            out[i] = data[perm[i]]


def bench_gather_scaling(
    n_rows: int = 500_000,
    thread_counts: List[int] = None,
    runs: int = 10,
) -> ScalingResult:
    """
    M2s: Measure gather (random access) bandwidth vs thread count.
    
    Shows if memory bandwidth saturates at low thread counts.
    """
    if not NUMBA_AVAILABLE:
        return ScalingResult(
            "M2s", "gather_scaling", "numba", 0, [], [], [], [],
            "ERROR:NUMBA_NOT_AVAILABLE"
        )
    
    if thread_counts is None:
        max_threads = get_num_threads()
        thread_counts = [1, 2, 4, min(8, max_threads), max_threads]
        thread_counts = sorted(set(thread_counts))
    
    # Allocate data
    data = np.random.randn(n_rows)
    perm = np.random.permutation(n_rows)
    out = np.empty(n_rows, dtype=np.float64)
    
    # Warmup
    original_threads = get_num_threads()
    set_num_threads(1)
    _gather_parallel(data, perm, out)
    
    # Measure at each thread count
    times = []
    bandwidths = []
    bytes_accessed = data.nbytes + perm.nbytes + out.nbytes
    
    for t in thread_counts:
        set_num_threads(t)
        mean_time, _ = _run_timed(_gather_parallel, data, perm, out, warmup=2, runs=runs)
        times.append(mean_time)
        bandwidths.append(bytes_accessed / mean_time / 1e9)
    
    # Restore threads
    set_num_threads(original_threads)
    
    # Calculate speedups
    speedups = [times[0] / t for t in times]
    
    # Diagnose
    max_speedup = max(speedups)
    if max_speedup < 1.5:
        diagnosis = f"⚠️ MEMORY SATURATED: Max speedup {max_speedup:.2f}× (bandwidth-limited)"
    elif max_speedup < len(thread_counts) / 2:
        diagnosis = f"⚠️ PARTIAL SCALING: {max_speedup:.2f}× (sublinear)"
    else:
        diagnosis = f"✅ GOOD SCALING: {max_speedup:.2f}×"
    
    return ScalingResult(
        primitive_id="M2s",
        name="gather_scaling",
        implementation="numba_parallel",
        size=n_rows,
        thread_counts=thread_counts,
        times_s=times,
        speedups=speedups,
        bandwidths_gbs=bandwidths,
        status="OK",
        diagnosis=diagnosis,
    )


# =============================================================================
# M3s: Scatter Reduce Scaling — Per-Group Accumulation vs Thread Count
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _scatter_reduce_parallel(values: np.ndarray, indices: np.ndarray, out: np.ndarray) -> None:
        """
        Parallel scatter-reduce with per-thread local buffers.
        
        Note: True atomic scatter would be slower; this simulates
        the pattern where each thread processes a subset of groups.
        """
        n = len(values)
        n_groups = len(out)
        # Each iteration handles one value
        for i in prange(n):
            # This would need atomics for true correctness
            # Here we just measure the access pattern cost
            idx = indices[i] % n_groups
            out[idx] += values[i]


def bench_scatter_reduce_scaling(
    n_rows: int = 500_000,
    n_groups: int = 5_000,
    thread_counts: List[int] = None,
    runs: int = 10,
) -> ScalingResult:
    """
    M3s: Measure scatter-reduce bandwidth vs thread count.
    """
    if not NUMBA_AVAILABLE:
        return ScalingResult(
            "M3s", "scatter_reduce_scaling", "numba", 0, [], [], [], [],
            "ERROR:NUMBA_NOT_AVAILABLE"
        )
    
    if thread_counts is None:
        max_threads = get_num_threads()
        thread_counts = [1, 2, 4, min(8, max_threads), max_threads]
        thread_counts = sorted(set(thread_counts))
    
    # Allocate data
    values = np.random.randn(n_rows)
    indices = np.random.randint(0, n_groups, size=n_rows)
    out = np.zeros(n_groups, dtype=np.float64)
    
    # Warmup
    original_threads = get_num_threads()
    set_num_threads(1)
    _scatter_reduce_parallel(values, indices, out)
    
    # Measure at each thread count
    times = []
    bandwidths = []
    bytes_accessed = values.nbytes + indices.nbytes + out.nbytes
    
    for t in thread_counts:
        set_num_threads(t)
        out[:] = 0
        mean_time, _ = _run_timed(_scatter_reduce_parallel, values, indices, out, warmup=2, runs=runs)
        times.append(mean_time)
        bandwidths.append(bytes_accessed / mean_time / 1e9)
    
    # Restore threads
    set_num_threads(original_threads)
    
    # Calculate speedups
    speedups = [times[0] / t for t in times]
    
    # Diagnose
    max_speedup = max(speedups)
    if max_speedup < 1.2:
        diagnosis = f"❌ NO SCALING: Max {max_speedup:.2f}× — likely write contention"
    elif max_speedup < 2.0:
        diagnosis = f"⚠️ POOR SCALING: Max {max_speedup:.2f}×"
    else:
        diagnosis = f"✅ OK SCALING: {max_speedup:.2f}×"
    
    return ScalingResult(
        primitive_id="M3s",
        name="scatter_reduce_scaling",
        implementation="numba_parallel",
        size=n_rows,
        thread_counts=thread_counts,
        times_s=times,
        speedups=speedups,
        bandwidths_gbs=bandwidths,
        status="OK",
        diagnosis=diagnosis,
    )


# =============================================================================
# SCH1: Prange Grain Sweep — Minimum Work for Speedup
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _prange_with_work(n: int, work_iters: int) -> float:
        """Parallel loop with configurable work per iteration."""
        total = 0.0
        for i in prange(n):
            # Simulate work
            x = float(i)
            for _ in range(work_iters):
                x = x * 1.0001 + 0.0001
            total += x
        return total


def bench_prange_grain_sweep(
    n_iterations: int = 5000,
    work_levels: List[int] = None,
    thread_counts: List[int] = None,
    runs: int = 5,
) -> Dict:
    """
    SCH1: Find minimum work per iteration needed for parallel speedup.
    
    Returns dict with speedup at each (work_level, thread_count) combination.
    """
    if not NUMBA_AVAILABLE:
        return {"status": "ERROR:NUMBA_NOT_AVAILABLE"}
    
    if work_levels is None:
        work_levels = [1, 10, 100, 1000, 10000]
    
    if thread_counts is None:
        max_threads = get_num_threads()
        thread_counts = [1, max_threads]
    
    # Warmup
    original_threads = get_num_threads()
    set_num_threads(1)
    _prange_with_work(100, 10)
    
    results = {}
    
    for work in work_levels:
        times_by_threads = {}
        
        for t in thread_counts:
            set_num_threads(t)
            mean_time, _ = _run_timed(_prange_with_work, n_iterations, work, warmup=1, runs=runs)
            times_by_threads[t] = mean_time
        
        # Calculate speedup (max threads vs 1 thread)
        t1 = times_by_threads.get(1, times_by_threads[min(thread_counts)])
        t_max = times_by_threads[max(thread_counts)]
        speedup = t1 / t_max if t_max > 0 else 0
        
        results[work] = {
            "times_s": times_by_threads,
            "speedup": round(speedup, 2),
            "work_ns_per_iter": round(t1 / n_iterations * 1e9, 1),
        }
    
    # Restore threads
    set_num_threads(original_threads)
    
    # Find crossover point
    crossover_work = None
    for work in sorted(work_levels):
        if results[work]["speedup"] >= 2.0:
            crossover_work = work
            break
    
    return {
        "primitive_id": "SCH1",
        "name": "prange_grain_sweep",
        "n_iterations": n_iterations,
        "thread_counts": thread_counts,
        "results": results,
        "crossover_work": crossover_work,
        "diagnosis": f"Speedup >= 2× requires {crossover_work or '>10000'} work iterations per prange iteration",
        "status": "OK",
    }


# =============================================================================
# M3w_parallel: Parallel Scatter Write (No Reduction)
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _scatter_write_parallel(values: np.ndarray, indices: np.ndarray, out: np.ndarray) -> None:
        """
        Parallel scatter write: out[indices[i]] = values[i]
        
        No reduction, no atomics. Last write wins.
        This isolates pure write contention from reduction overhead.
        """
        n = len(values)
        for i in prange(n):
            out[indices[i]] = values[i]


def bench_scatter_write_parallel(
    n_rows: int = 500_000,
    n_groups: int = 5_000,
    thread_counts: List[int] = None,
    runs: int = 10,
) -> ScalingResult:
    """
    M3w_parallel: Measure pure scatter write (no reduction) scaling.
    
    Distinguishes write bandwidth / cache-line contention from reduction logic.
    """
    if not NUMBA_AVAILABLE:
        return ScalingResult(
            "M3w_p", "scatter_write_parallel", "numba", 0, [], [], [], [],
            "ERROR:NUMBA_NOT_AVAILABLE"
        )
    
    if thread_counts is None:
        max_threads = get_num_threads()
        thread_counts = [1, 2, 4, min(8, max_threads), max_threads]
        thread_counts = sorted(set(thread_counts))
    
    # Allocate data
    values = np.random.randn(n_rows)
    indices = np.random.randint(0, n_groups, size=n_rows).astype(np.int64)
    out = np.zeros(n_groups, dtype=np.float64)
    
    # Warmup
    original_threads = get_num_threads()
    set_num_threads(1)
    _scatter_write_parallel(values, indices, out)
    
    # Measure at each thread count
    times = []
    bandwidths = []
    bytes_accessed = values.nbytes + indices.nbytes + out.nbytes
    
    for t in thread_counts:
        set_num_threads(t)
        out[:] = 0
        mean_time, _ = _run_timed(_scatter_write_parallel, values, indices, out, warmup=2, runs=runs)
        times.append(mean_time)
        bandwidths.append(bytes_accessed / mean_time / 1e9)
    
    # Restore threads
    set_num_threads(original_threads)
    
    # Calculate speedups
    speedups = [times[0] / t for t in times]
    
    # Diagnose
    max_speedup = max(speedups)
    if max_speedup < 1.2:
        diagnosis = f"❌ NO SCALING: Max {max_speedup:.2f}× — pure write contention (cache-line bouncing)"
    elif max_speedup < 2.0:
        diagnosis = f"⚠️ POOR SCALING: Max {max_speedup:.2f}× — partial write contention"
    else:
        diagnosis = f"✅ OK SCALING: {max_speedup:.2f}×"
    
    return ScalingResult(
        primitive_id="M3w_p",
        name="scatter_write_parallel",
        implementation="numba_parallel",
        size=n_rows,
        thread_counts=thread_counts,
        times_s=times,
        speedups=speedups,
        bandwidths_gbs=bandwidths,
        status="OK",
        diagnosis=diagnosis,
    )


# =============================================================================
# Combined Scaling Benchmark Runner
# =============================================================================

def run_scaling_benchmarks(
    n_rows: int = 500_000,
    n_groups: int = 5_000,
    verbose: bool = True,
) -> Dict:
    """
    Run all scaling benchmarks and return combined results.
    """
    if verbose:
        print("=" * 70)
        print("SCALING & CONTENTION PRIMITIVES")
        print("=" * 70)
    
    results = {}
    
    # FS1/FS2: False sharing
    if verbose:
        print("\nRunning FS1/FS2 (false sharing)...")
    fs1, fs2 = bench_false_sharing()
    results["FS1"] = fs1.to_dict()
    results["FS2"] = fs2.to_dict()
    if verbose:
        print(f"  FS1 (adjacent): speedups = {fs1.speedups}")
        print(f"  FS2 (padded): speedups = {fs2.speedups}")
        print(f"  Diagnosis: {fs1.diagnosis}")
    
    # M2s: Gather scaling
    if verbose:
        print("\nRunning M2s (gather scaling)...")
    m2s = bench_gather_scaling(n_rows=n_rows)
    results["M2s"] = m2s.to_dict()
    if verbose:
        print(f"  Thread counts: {m2s.thread_counts}")
        print(f"  Speedups: {m2s.speedups}")
        print(f"  Bandwidths: {m2s.bandwidths_gbs} GB/s")
        print(f"  Diagnosis: {m2s.diagnosis}")
    
    # M3s: Scatter reduce scaling
    if verbose:
        print("\nRunning M3s (scatter reduce scaling)...")
    m3s = bench_scatter_reduce_scaling(n_rows=n_rows, n_groups=n_groups)
    results["M3s"] = m3s.to_dict()
    if verbose:
        print(f"  Thread counts: {m3s.thread_counts}")
        print(f"  Speedups: {m3s.speedups}")
        print(f"  Diagnosis: {m3s.diagnosis}")
    
    # M3w_parallel: Scatter write scaling (no reduction)
    if verbose:
        print("\nRunning M3w_parallel (scatter write scaling)...")
    m3wp = bench_scatter_write_parallel(n_rows=n_rows, n_groups=n_groups)
    results["M3w_p"] = m3wp.to_dict()
    if verbose:
        print(f"  Thread counts: {m3wp.thread_counts}")
        print(f"  Speedups: {m3wp.speedups}")
        print(f"  Diagnosis: {m3wp.diagnosis}")
    
    # SCH1: Grain sweep
    if verbose:
        print("\nRunning SCH1 (grain sweep)...")
    sch1 = bench_prange_grain_sweep()
    results["SCH1"] = sch1
    if verbose:
        print(f"  Crossover work: {sch1.get('crossover_work', 'N/A')}")
        print(f"  Diagnosis: {sch1.get('diagnosis', 'N/A')}")
    
    return results
