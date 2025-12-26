"""
Benchmark Framework v1.0 — Memory and CPU Profiling

Cross-platform memory and CPU profiling utilities.

Phase 12.10.BF: Memory tracking for batch farm constraints.
Phase 12.11: CPU profiling for performance recovery.

Metrics:
- peak_rss_mb: Process-wide peak RSS (primary metric)
- peak_tracemalloc_mb: Python allocations only (optional, for debugging)
- wall_time_s: Wall-clock time (always measured)
- cpu_total_time_s: cProfile total time (when CPU profiling enabled)

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

import cProfile
import io
import json
import os
import platform
import pstats
import sys
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Callable, Any, List, Dict

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
# CPU PROFILING (Phase 12.11)
# =============================================================================

@dataclass
class CPUProfileResult:
    """
    CPU profile results for archiving.
    
    Phase 12.11: Per-function CPU profiling with cProfile.
    
    Note on timing:
        - wall_time_s: Always measured with time.perf_counter() (consistent metric)
        - cpu_total_time_s: cProfile's total_tt (only when profiling enabled)
    """
    enabled: bool
    wall_time_s: float = 0.0                    # Always measured (consistent!)
    cpu_total_time_s: Optional[float] = None    # cProfile time (when enabled)
    prof_path: Optional[str] = None             # Relative path to .prof
    txt_path: Optional[str] = None              # Relative path to .txt
    json_path: Optional[str] = None             # Relative path to _cpu.json
    total_calls: int = 0
    top_functions: List[Dict] = field(default_factory=list)
    sort_key: str = "cumulative"                # Sort key used for top_functions


def profile_function_cpu(
    func: Callable,
    *args,
    output_dir: str,
    name: str,
    top_n: int = 10,
    enabled: bool = True,
    **kwargs,
) -> tuple[Any, CPUProfileResult]:
    """
    Profile a single function call with cProfile.
    
    Parameters
    ----------
    func : Callable
        Function to profile
    output_dir : str
        Directory for profile output files (profiles subdirectory)
    name : str
        Base name for output files (e.g., "v5_S3_n1")
    top_n : int, default=10
        Number of top functions to include in summary (user-configurable)
    enabled : bool, default=True
        If False, skip CPU profiling (still measure wall-time)
    *args, **kwargs
        Arguments passed to func
        
    Returns
    -------
    result : Any
        Return value of func
    profile : CPUProfileResult
        Profile data for archiving (paths are RELATIVE to run directory)
        
    Note
    ----
    wall_time_s is ALWAYS measured, regardless of whether CPU profiling is enabled.
    This ensures consistent timing metrics across all runs.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Relative paths for JSON storage (portable across machines)
    rel_prof_path = f"profiles/{name}.prof"
    rel_txt_path = f"profiles/{name}.txt"
    rel_json_path = f"profiles/{name}_cpu.json"
    
    # Absolute paths for file writing
    abs_prof_path = out_dir / f"{name}.prof"
    abs_txt_path = out_dir / f"{name}.txt"
    abs_json_path = out_dir / f"{name}_cpu.json"
    
    if not enabled:
        # Still measure wall-time even when profiling disabled
        start = time.perf_counter()
        result = func(*args, **kwargs)
        wall_time = time.perf_counter() - start
        
        return result, CPUProfileResult(
            enabled=False,
            wall_time_s=round(wall_time, 6),
            cpu_total_time_s=None,
        )
    
    # Run with profiling + wall-time measurement
    profiler = cProfile.Profile()
    
    start = time.perf_counter()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()
    wall_time = time.perf_counter() - start
    
    # Save binary (.prof)
    profiler.dump_stats(str(abs_prof_path))
    
    # Generate text summary
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats("cumulative")
    stats.print_stats(top_n * 2)  # Extra for text file
    abs_txt_path.write_text(stream.getvalue())
    
    # Extract top N functions using fcn_list (CORRECT: sorted order)
    # After sort_stats(), fcn_list contains function keys in sorted order
    top_functions = []
    
    for func_key in stats.fcn_list[:top_n]:
        filename, lineno, func_name = func_key
        # pstats tuple: (primitive_calls, total_calls, tottime, cumtime, callers)
        cc, nc, tt, ct, callers = stats.stats[func_key]
        top_functions.append({
            "function": func_name,
            "file": filename.split("/")[-1] if "/" in filename else filename,
            "line": lineno,
            "ncalls": nc,         # Total calls (not primitive)
            "tottime_s": round(tt, 6),
            "cumtime_s": round(ct, 6),
        })
    
    cpu_total_time = stats.total_tt
    # Total calls: sum of nc (total calls) across all functions
    total_calls = sum(stats.stats[k][1] for k in stats.stats)
    
    profile_result = CPUProfileResult(
        enabled=True,
        wall_time_s=round(wall_time, 6),          # Always wall-time
        cpu_total_time_s=round(cpu_total_time, 6),
        prof_path=rel_prof_path,                  # Relative path
        txt_path=rel_txt_path,                    # Relative path
        json_path=rel_json_path,                  # Relative path
        total_calls=total_calls,
        top_functions=top_functions,
        sort_key="cumulative",
    )
    
    # Save JSON summary (with relative paths)
    abs_json_path.write_text(json.dumps(asdict(profile_result), indent=2))
    
    return result, profile_result


# =============================================================================
# COMBINED PROFILING (CPU + Memory)
# =============================================================================

@dataclass
class CombinedProfileResult:
    """Combined CPU and memory profile results."""
    cpu: CPUProfileResult
    peak_rss_mb: float
    tracemalloc_peak_mb: Optional[float] = None


def profile_function_full(
    func: Callable,
    *args,
    output_dir: str,
    name: str,
    top_n: int = 10,
    cpu_enabled: bool = True,
    memory_enabled: bool = True,
    **kwargs,
) -> tuple[Any, CombinedProfileResult]:
    """
    Profile function with both CPU and memory tracking.
    
    Parameters
    ----------
    func : Callable
        Function to profile
    output_dir : str
        Directory for profile output files
    name : str
        Base name for output files
    top_n : int, default=10
        Number of top functions in CPU profile
    cpu_enabled : bool, default=True
        Enable CPU profiling (cProfile)
    memory_enabled : bool, default=True
        Enable memory tracking (tracemalloc)
    *args, **kwargs
        Arguments passed to func
        
    Returns
    -------
    result : Any
        Return value of func
    profile : CombinedProfileResult
        Combined profile data
        
    Note
    ----
    wall_time_s is always measured in the CPU profile, regardless of cpu_enabled.
    """
    tracemalloc_peak = None
    
    if memory_enabled:
        tracemalloc.start()
    
    # CPU profile (or just wall-time if disabled)
    result, cpu_profile = profile_function_cpu(
        func, *args,
        output_dir=output_dir,
        name=name,
        top_n=top_n,
        enabled=cpu_enabled,
        **kwargs,
    )
    
    if memory_enabled:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc_peak = round(peak / (1024 * 1024), 2)
    
    peak_rss = get_peak_rss_mb()
    
    return result, CombinedProfileResult(
        cpu=cpu_profile,
        peak_rss_mb=peak_rss,
        tracemalloc_peak_mb=tracemalloc_peak,
    )


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
    # Combined (legacy)
    "run_benchmark_with_memory",
    # CPU profiling (Phase 12.11)
    "CPUProfileResult",
    "CombinedProfileResult",
    "profile_function_cpu",
    "profile_function_full",
]
