"""
Phase 12.12: Roofline Model and Performance Analysis

Provides theoretical bounds and efficiency calculations for
understanding parallel performance limitations.

Key Concepts:
- Roofline: min(peak_compute, memory_bandwidth × arithmetic_intensity)
- Parallel Efficiency: actual_speedup / theoretical_speedup
- Overhead Dominated: thread_overhead > work_per_group
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RooflineAnalysis:
    """Result of roofline analysis."""
    peak_bandwidth_gbs: float
    measured_bandwidth_gbs: float
    bandwidth_efficiency: float
    
    parallel_efficiency: float
    theoretical_speedup: float
    actual_speedup: float
    
    is_bandwidth_bound: bool
    is_overhead_dominated: bool
    
    diagnosis: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "peak_bandwidth_gbs": round(self.peak_bandwidth_gbs, 2),
            "measured_bandwidth_gbs": round(self.measured_bandwidth_gbs, 2),
            "bandwidth_efficiency": round(self.bandwidth_efficiency, 3),
            "parallel_efficiency": round(self.parallel_efficiency, 3),
            "theoretical_speedup": round(self.theoretical_speedup, 2),
            "actual_speedup": round(self.actual_speedup, 2),
            "is_bandwidth_bound": self.is_bandwidth_bound,
            "is_overhead_dominated": self.is_overhead_dominated,
            "diagnosis": self.diagnosis,
        }


# =============================================================================
# Core Roofline Equations
# =============================================================================

def parallel_overhead_bound(
    n_groups: int,
    overhead_per_spawn_us: float,
) -> float:
    """
    Calculate theoretical parallel overhead time.
    
    Parameters
    ----------
    n_groups : int
        Number of parallel work items
    overhead_per_spawn_us : float
        Overhead per thread spawn in microseconds (from O4 primitive)
        
    Returns
    -------
    Total overhead time in seconds
    """
    return n_groups * overhead_per_spawn_us / 1e6


def parallel_efficiency(
    t_serial: float,
    t_parallel: float,
    n_threads: int,
) -> float:
    """
    Calculate parallel efficiency.
    
    Parameters
    ----------
    t_serial : float
        Time with 1 thread (seconds)
    t_parallel : float
        Time with n_threads (seconds)
    n_threads : int
        Number of threads used
        
    Returns
    -------
    Efficiency in range [0, 1]
    
    efficiency = 1.0 means perfect scaling
    efficiency < 0.5 indicates significant overhead
    """
    if t_parallel <= 0 or n_threads <= 0:
        return 0.0
    
    theoretical_speedup = n_threads
    actual_speedup = t_serial / t_parallel
    
    return actual_speedup / theoretical_speedup


def is_overhead_dominated(
    t_work_per_group_us: float,
    t_overhead_us: float,
    threshold: float = 0.5,
) -> bool:
    """
    Check if thread overhead exceeds useful work.
    
    Parameters
    ----------
    t_work_per_group_us : float
        Time to process one group (microseconds)
    t_overhead_us : float
        Thread overhead per group (microseconds)
    threshold : float
        Overhead/work ratio threshold (default 0.5 = 50%)
        
    Returns
    -------
    True if overhead is too high for effective parallelism
    """
    if t_work_per_group_us <= 0:
        return True
    
    ratio = t_overhead_us / t_work_per_group_us
    return ratio > threshold


def roofline_bound(
    flops: float,
    bytes_accessed: float,
    peak_gflops: float,
    peak_bandwidth_gbs: float,
) -> Tuple[float, str]:
    """
    Calculate roofline-limited performance.
    
    Parameters
    ----------
    flops : float
        Floating point operations count
    bytes_accessed : float
        Bytes read/written
    peak_gflops : float
        Peak compute throughput (GFLOP/s)
    peak_bandwidth_gbs : float
        Peak memory bandwidth (GB/s)
        
    Returns
    -------
    (max_gflops, bound_type) where bound_type is "compute" or "memory"
    """
    if bytes_accessed <= 0:
        return peak_gflops, "compute"
    
    arithmetic_intensity = flops / bytes_accessed  # FLOP/byte
    
    # Roofline: min(peak, bandwidth × intensity)
    memory_bound_gflops = peak_bandwidth_gbs * arithmetic_intensity
    
    if memory_bound_gflops < peak_gflops:
        return memory_bound_gflops, "memory"
    else:
        return peak_gflops, "compute"


# =============================================================================
# Diagnostic Functions
# =============================================================================

def diagnose_parallel_failure(
    t_serial: float,
    t_parallel: float,
    n_threads: int,
    t_overhead_per_group_us: float,
    n_groups: int,
    t_work_per_group_us: float,
) -> List[str]:
    """
    Diagnose why parallel speedup failed.
    
    Returns list of diagnostic messages.
    """
    diagnosis = []
    
    # Calculate efficiency
    eff = parallel_efficiency(t_serial, t_parallel, n_threads)
    actual_speedup = t_serial / t_parallel if t_parallel > 0 else 0
    
    diagnosis.append(f"Parallel efficiency: {eff:.1%}")
    diagnosis.append(f"Actual speedup: {actual_speedup:.2f}× (expected {n_threads}×)")
    
    # Check overhead domination
    if is_overhead_dominated(t_work_per_group_us, t_overhead_per_group_us):
        overhead_ratio = t_overhead_per_group_us / t_work_per_group_us if t_work_per_group_us > 0 else float('inf')
        diagnosis.append(f"⚠️ OVERHEAD DOMINATED: Thread overhead ({t_overhead_per_group_us:.1f} μs) is {overhead_ratio:.1f}× work per group ({t_work_per_group_us:.1f} μs)")
        diagnosis.append("→ Recommendation: Increase work per thread (batch groups) or reduce parallelism")
    
    # Check total overhead vs total time
    total_overhead = parallel_overhead_bound(n_groups, t_overhead_per_group_us)
    overhead_fraction = total_overhead / t_parallel if t_parallel > 0 else 0
    
    if overhead_fraction > 0.2:
        diagnosis.append(f"⚠️ HIGH OVERHEAD FRACTION: {overhead_fraction:.1%} of parallel time is overhead")
    
    # Check if speedup is actually negative
    if actual_speedup < 1.0:
        diagnosis.append("❌ NEGATIVE SPEEDUP: Parallel is slower than serial")
        diagnosis.append("→ Possible causes: false sharing, synchronization, memory contention")
    
    return diagnosis


def calculate_v5_theoretical_time(
    n_rows: int,
    n_groups: int,
    n_fits: int,
    n_params: int,
    primitive_times: Dict[str, float],  # ID -> time in seconds
) -> float:
    """
    Calculate theoretical V5 execution time from primitives.
    
    Parameters
    ----------
    n_rows : int
        Total rows
    n_groups : int
        Number of groups
    n_fits : int
        Number of fit columns
    n_params : int
        Number of parameters (intercept + linear columns)
    primitive_times : dict
        Primitive timings: {
            "S1": sort_time_s,
            "S2": boundaries_time_s,
            "C2": solve_time_s,
            "C5": median_time_s,
            "C6": mad_time_s,
            ...
        }
        
    Returns
    -------
    Predicted time in seconds
    """
    rows_per_group = n_rows / n_groups if n_groups > 0 else 0
    
    # Sort + boundaries (once)
    t_sort = primitive_times.get("S1", 0)
    t_boundaries = primitive_times.get("S2", 0)
    
    # Per group × per fit operations
    t_solve = primitive_times.get("C2", 0)
    t_mad = primitive_times.get("C6", 0)
    
    # Total
    t_per_group_fit = t_solve + t_mad
    t_kernel = n_groups * n_fits * t_per_group_fit
    
    return t_sort + t_boundaries + t_kernel


def calculate_parallel_potential(
    t_serial: float,
    t_overhead_total: float,
    n_threads: int,
) -> Tuple[float, float]:
    """
    Calculate potential parallel time and maximum achievable speedup.
    
    Parameters
    ----------
    t_serial : float
        Serial execution time (seconds)
    t_overhead_total : float
        Total parallel overhead (seconds)
    n_threads : int
        Number of threads
        
    Returns
    -------
    (t_parallel_optimal, max_speedup)
    
    t_parallel_optimal = t_serial / n_threads + t_overhead
    max_speedup = t_serial / t_parallel_optimal
    """
    if n_threads <= 0:
        return t_serial, 1.0
    
    t_parallel_optimal = t_serial / n_threads + t_overhead_total
    max_speedup = t_serial / t_parallel_optimal if t_parallel_optimal > 0 else 0
    
    return t_parallel_optimal, max_speedup
