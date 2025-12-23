"""
Benchmark Framework v1.0 — Memory Profiling

Cross-platform memory profiling utilities.

Phase 12.10.BF: Memory tracking for batch farm constraints.

Metrics:
- peak_rss_mb: Process-wide peak RSS (primary metric)
- peak_tracemalloc_mb: Python allocations only (optional, for debugging)

Platform Support:
- macOS (Darwin): ru_maxrss returns bytes
- Linux: ru_maxrss returns KB
- Windows: Not supported (resource module unavailable)

Note on RSS Semantics:
    `resource.ru_maxrss` returns the process-lifetime maximum RSS, not a
    per-benchmark measurement. In in-process execution mode, this means
    later benchmarks may report the same peak even if they didn't cause it.
    
    For v1.0, this is acceptable because:
    - We compare same benchmark across runs (not benchmarks within a run)
    - Regression detection uses history from same benchmark+scenario+params
"""

import platform
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Callable, Any

# resource module is Unix-only
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    HAS_RESOURCE = False


# =============================================================================
# PLATFORM CHECK
# =============================================================================

def check_platform_support() -> tuple[bool, str]:
    """
    Check if current platform supports memory profiling.
    
    Returns:
        (supported, message)
    """
    system = platform.system()
    
    if system == "Windows":
        return False, "Windows not supported in v1.0 (resource module unavailable)"
    
    if not HAS_RESOURCE:
        return False, "resource module not available"
    
    return True, f"Platform {system} supported"


# =============================================================================
# RSS MEASUREMENT
# =============================================================================

def get_peak_rss_mb() -> float:
    """
    Get peak RSS in MB (cross-platform for Unix).
    
    Returns:
        Peak RSS in megabytes.
        
    Note:
        - macOS (Darwin): ru_maxrss is in bytes
        - Linux: ru_maxrss is in KB
        - Returns 0.0 if resource module unavailable (Windows)
    """
    if not HAS_RESOURCE:
        return 0.0
    
    usage = resource.getrusage(resource.RUSAGE_SELF)
    
    if platform.system() == 'Darwin':
        # macOS: ru_maxrss is in bytes
        return usage.ru_maxrss / (1024 * 1024)
    else:
        # Linux: ru_maxrss is in KB
        return usage.ru_maxrss / 1024


def get_current_rss_mb() -> float:
    """
    Get current RSS in MB (not peak).
    
    Uses /proc/self/status on Linux, sysctl on macOS.
    Returns 0.0 if unavailable.
    """
    system = platform.system()
    
    if system == "Linux":
        try:
            with open('/proc/self/status', 'r') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        # Format: "VmRSS:    12345 kB"
                        parts = line.split()
                        return float(parts[1]) / 1024  # KB → MB
        except (FileNotFoundError, PermissionError, ValueError):
            pass
    
    elif system == "Darwin":
        try:
            import subprocess
            pid = str(os.getpid())
            result = subprocess.check_output(
                ['ps', '-o', 'rss=', '-p', pid],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return float(result) / 1024  # KB → MB
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
            pass
    
    return 0.0


# =============================================================================
# MEMORY TRACKING CONTEXT MANAGERS
# =============================================================================

@dataclass
class MemoryStats:
    """Memory statistics from a profiled execution."""
    peak_rss_mb: float
    peak_tracemalloc_mb: Optional[float] = None
    current_rss_mb: Optional[float] = None
    top_allocations: Optional[list] = None


@contextmanager
def track_peak_rss():
    """
    Context manager to track peak RSS during execution.
    
    Usage:
        with track_peak_rss() as stats:
            run_benchmark()
        print(f"Peak RSS: {stats.peak_rss_mb:.1f} MB")
    
    Note:
        Peak RSS is process-wide and non-decreasing within a process.
        The returned value is the peak RSS at the end of the context,
        not necessarily caused by the code within the context.
    """
    stats = MemoryStats(peak_rss_mb=0.0)
    
    try:
        yield stats
    finally:
        stats.peak_rss_mb = get_peak_rss_mb()
        stats.current_rss_mb = get_current_rss_mb()


@contextmanager
def track_memory_detailed(top_n: int = 10):
    """
    Track memory with tracemalloc for detailed breakdown.
    
    Usage:
        with track_memory_detailed() as stats:
            run_benchmark()
        print(f"Peak traced: {stats.peak_tracemalloc_mb:.1f} MB")
        for alloc in stats.top_allocations:
            print(f"  {alloc}")
    
    Parameters:
        top_n: Number of top allocations to capture
    
    Note:
        tracemalloc only tracks Python allocations, not native/C allocations.
        For Numba/NumPy heavy code, peak_rss_mb is more accurate.
    """
    tracemalloc.start()
    
    stats = MemoryStats(peak_rss_mb=0.0)
    
    try:
        yield stats
    finally:
        # Get tracemalloc stats
        current, peak = tracemalloc.get_traced_memory()
        stats.peak_tracemalloc_mb = peak / (1024 * 1024)
        
        # Get top allocations
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')[:top_n]
        
        stats.top_allocations = []
        for rank, stat in enumerate(top_stats, 1):
            frame = stat.traceback[0]
            stats.top_allocations.append({
                "rank": rank,
                "file": frame.filename.split('/')[-1],  # Just filename
                "line": frame.lineno,
                "size_mb": stat.size / (1024 * 1024),
            })
        
        tracemalloc.stop()
        
        # Also capture RSS
        stats.peak_rss_mb = get_peak_rss_mb()
        stats.current_rss_mb = get_current_rss_mb()


# =============================================================================
# MEMORY GUARD
# =============================================================================

class MemoryGuard:
    """
    Enforce memory limit during benchmark.
    
    Usage:
        guard = MemoryGuard(limit_mb=8192)  # 8 GB
        guard.check()  # Raises MemoryError if exceeded
    
    For batch farm environments with 2 GB/core limits.
    """
    
    def __init__(self, limit_mb: float):
        self.limit_mb = limit_mb
    
    def check(self) -> bool:
        """Return True if within limit, False if exceeded."""
        current = get_peak_rss_mb()
        return current <= self.limit_mb
    
    def assert_within_limit(self):
        """Raise MemoryError if memory limit exceeded."""
        current = get_peak_rss_mb()
        if current > self.limit_mb:
            raise MemoryError(
                f"Memory limit exceeded: {current:.1f} MB > {self.limit_mb:.1f} MB"
            )
    
    def get_usage_pct(self) -> float:
        """Get current memory usage as percentage of limit."""
        current = get_peak_rss_mb()
        return (current / self.limit_mb) * 100 if self.limit_mb > 0 else 0.0


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@dataclass
class TimingResult:
    """Result from timed execution."""
    elapsed_s: float
    result: Any = None
    error: Optional[str] = None


def time_function(
    func: Callable,
    *args,
    **kwargs,
) -> TimingResult:
    """
    Time a single function execution.
    
    Returns:
        TimingResult with elapsed time and result/error.
    """
    start = time.perf_counter()
    
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return TimingResult(elapsed_s=elapsed, result=result)
    
    except Exception as e:
        elapsed = time.perf_counter() - start
        return TimingResult(elapsed_s=elapsed, error=str(e))


def time_function_n(
    func: Callable,
    n_runs: int,
    *args,
    warmup_runs: int = 0,
    **kwargs,
) -> tuple[list[float], Any]:
    """
    Time a function over multiple runs.
    
    Parameters:
        func: Function to time
        n_runs: Number of timed runs
        *args: Positional arguments for func
        warmup_runs: Number of warmup runs (not timed)
        **kwargs: Keyword arguments for func
    
    Returns:
        (list of times, last result)
    
    Raises:
        Exception from func if any run fails.
    """
    # Warmup runs
    result = None
    for _ in range(warmup_runs):
        result = func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    return times, result


# =============================================================================
# COMBINED BENCHMARK RUNNER
# =============================================================================

def run_benchmark_with_memory(
    func: Callable,
    *args,
    n_runs: int = 3,
    warmup_runs: int = 2,
    profile: bool = False,
    top_n: int = 10,
    **kwargs,
) -> tuple[list[float], MemoryStats, Any]:
    """
    Run benchmark with timing and memory tracking.
    
    Parameters:
        func: Benchmark function
        *args: Positional arguments
        n_runs: Number of timed runs
        warmup_runs: Number of warmup runs (for JIT)
        profile: If True, enable tracemalloc
        top_n: Number of top allocations to capture
        **kwargs: Keyword arguments
    
    Returns:
        (times, memory_stats, last_result)
    
    Example:
        times, mem, result = run_benchmark_with_memory(
            make_parallel_fit_v5,
            df=df,
            gb_columns=['group'],
            n_runs=3,
            warmup_runs=2,
        )
    """
    if profile:
        with track_memory_detailed(top_n=top_n) as mem_stats:
            times, result = time_function_n(
                func, n_runs, *args, 
                warmup_runs=warmup_runs, 
                **kwargs
            )
    else:
        with track_peak_rss() as mem_stats:
            times, result = time_function_n(
                func, n_runs, *args,
                warmup_runs=warmup_runs,
                **kwargs
            )
    
    return times, mem_stats, result


# =============================================================================
# EXPORTS
# =============================================================================

# Need os for get_current_rss_mb on macOS
import os

__all__ = [
    # Platform
    "check_platform_support",
    "HAS_RESOURCE",
    # RSS
    "get_peak_rss_mb",
    "get_current_rss_mb",
    # Context managers
    "MemoryStats",
    "track_peak_rss",
    "track_memory_detailed",
    # Guard
    "MemoryGuard",
    # Timing
    "TimingResult",
    "time_function",
    "time_function_n",
    # Combined
    "run_benchmark_with_memory",
]
