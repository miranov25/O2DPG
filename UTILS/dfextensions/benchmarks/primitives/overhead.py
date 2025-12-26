"""
Phase 12.12: Overhead Primitives (O1-O4)

Measures loop and dispatch overhead to determine if thread
coordination cost exceeds useful work per group.

Primitives:
- O1: python_loop — Empty Python for-loop overhead
- O2: numba_loop — Empty Numba prange overhead  
- O3: numba_dispatch — JIT function call overhead
- O4: parallel_overhead — Thread fork/join cost (PRIORITY)
"""

import time
import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass

try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


@dataclass
class PrimitiveResult:
    """Result from a primitive benchmark."""
    primitive_id: str
    name: str
    implementation: str
    size: int
    calls: int
    wall_time_s: float
    time_per_call_ns: float
    status: str  # OK, SLOW, ERROR
    
    def to_dict(self) -> Dict:
        return {
            "id": f"primitive:{self.primitive_id}:{self.name}:{self.implementation}",
            "name": self.name,
            "implementation": self.implementation,
            "size": self.size,
            "calls": self.calls,
            "wall_time_s": round(self.wall_time_s, 6),
            "time_per_call_ns": round(self.time_per_call_ns, 2),
            "status": self.status,
        }


def _run_timed(func, *args, warmup: int = 2, runs: int = 5, **kwargs) -> Tuple[float, float]:
    """
    Run function with warmup and return (mean_time, std_time).
    
    Returns time in seconds.
    """
    # Warmup (excluded from timing)
    for _ in range(warmup):
        func(*args, **kwargs)
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)
    
    return np.mean(times), np.std(times)


# =============================================================================
# O1: Python Loop Overhead
# =============================================================================

def _python_loop(n: int) -> int:
    """Empty Python for-loop."""
    total = 0
    for i in range(n):
        total += 1
    return total


def bench_python_loop(n_iterations: int = 5000, runs: int = 100) -> PrimitiveResult:
    """
    O1: Measure empty Python for-loop overhead.
    
    Parameters
    ----------
    n_iterations : int
        Number of loop iterations (simulates n_groups)
    runs : int
        Number of benchmark runs
        
    Returns
    -------
    PrimitiveResult with time per iteration in nanoseconds
    """
    mean_time, _ = _run_timed(_python_loop, n_iterations, warmup=2, runs=runs)
    time_per_iter_ns = (mean_time / n_iterations) * 1e9
    
    return PrimitiveResult(
        primitive_id="O1",
        name="python_loop",
        implementation="python",
        size=n_iterations,
        calls=runs,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_iter_ns,
        status="OK",
    )


# =============================================================================
# O2: Numba Loop Overhead
# =============================================================================

if NUMBA_AVAILABLE:
    @njit
    def _numba_loop_serial(n: int) -> int:
        """Empty Numba for-loop (serial)."""
        total = 0
        for i in range(n):
            total += 1
        return total
    
    @njit(parallel=True)
    def _numba_loop_parallel(n: int) -> int:
        """Empty Numba prange loop (parallel)."""
        total = 0
        for i in prange(n):
            total += 1
        return total


def bench_numba_loop(n_iterations: int = 5000, runs: int = 100, parallel: bool = True) -> PrimitiveResult:
    """
    O2: Measure empty Numba loop overhead.
    
    Parameters
    ----------
    n_iterations : int
        Number of loop iterations (simulates n_groups)
    runs : int
        Number of benchmark runs
    parallel : bool
        If True, use prange; if False, use range
        
    Returns
    -------
    PrimitiveResult with time per iteration in nanoseconds
    """
    if not NUMBA_AVAILABLE:
        return PrimitiveResult(
            primitive_id="O2",
            name="numba_loop",
            implementation="numba_parallel" if parallel else "numba_serial",
            size=n_iterations,
            calls=0,
            wall_time_s=0.0,
            time_per_call_ns=0.0,
            status="ERROR:NUMBA_NOT_AVAILABLE",
        )
    
    func = _numba_loop_parallel if parallel else _numba_loop_serial
    impl = "numba_parallel" if parallel else "numba_serial"
    
    mean_time, _ = _run_timed(func, n_iterations, warmup=2, runs=runs)
    time_per_iter_ns = (mean_time / n_iterations) * 1e9
    
    return PrimitiveResult(
        primitive_id="O2",
        name="numba_loop",
        implementation=impl,
        size=n_iterations,
        calls=runs,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_iter_ns,
        status="OK",
    )


# =============================================================================
# O3: Numba Dispatch Overhead
# =============================================================================

if NUMBA_AVAILABLE:
    @njit
    def _numba_noop() -> int:
        """Minimal Numba function for dispatch overhead."""
        return 0


def bench_numba_dispatch(n_calls: int = 10000, runs: int = 5) -> PrimitiveResult:
    """
    O3: Measure Numba JIT function call overhead.
    
    Parameters
    ----------
    n_calls : int
        Number of function calls per run
    runs : int
        Number of benchmark runs
        
    Returns
    -------
    PrimitiveResult with time per call in nanoseconds
    """
    if not NUMBA_AVAILABLE:
        return PrimitiveResult(
            primitive_id="O3",
            name="numba_dispatch",
            implementation="numba",
            size=n_calls,
            calls=0,
            wall_time_s=0.0,
            time_per_call_ns=0.0,
            status="ERROR:NUMBA_NOT_AVAILABLE",
        )
    
    # Warmup
    for _ in range(10):
        _numba_noop()
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        for _ in range(n_calls):
            _numba_noop()
        times.append(time.perf_counter() - start)
    
    mean_time = np.mean(times)
    time_per_call_ns = (mean_time / n_calls) * 1e9
    
    return PrimitiveResult(
        primitive_id="O3",
        name="numba_dispatch",
        implementation="numba",
        size=n_calls,
        calls=runs,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_call_ns,
        status="OK",
    )


# =============================================================================
# O4: Parallel Overhead (PRIORITY)
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _parallel_spawn_work(n: int, work_array: np.ndarray) -> None:
        """
        Parallel loop with minimal work to measure spawn overhead.
        
        Each iteration writes to a separate array element to force
        actual parallel execution (no reduction that could be optimized away).
        """
        for i in prange(n):
            work_array[i] = i


def bench_parallel_overhead(
    n_groups: int = 5000,
    runs: int = 20,
) -> PrimitiveResult:
    """
    O4: Measure thread fork/join overhead per parallel spawn.
    
    This is the PRIORITY primitive — if thread overhead exceeds
    work per group, parallelism cannot provide speedup.
    
    Parameters
    ----------
    n_groups : int
        Number of parallel work items (simulates n_groups in V5)
    runs : int
        Number of benchmark runs
        
    Returns
    -------
    PrimitiveResult with overhead per spawn in nanoseconds
    
    Notes
    -----
    The "overhead" here includes:
    - Thread pool wake-up
    - Work distribution
    - Synchronization barrier
    
    If overhead_ns > work_per_group_ns, parallel speedup is impossible.
    """
    if not NUMBA_AVAILABLE:
        return PrimitiveResult(
            primitive_id="O4",
            name="parallel_overhead",
            implementation="numba_parallel",
            size=n_groups,
            calls=0,
            wall_time_s=0.0,
            time_per_call_ns=0.0,
            status="ERROR:NUMBA_NOT_AVAILABLE",
        )
    
    # Allocate work array (forces actual parallel execution)
    work_array = np.zeros(n_groups, dtype=np.float64)
    
    # Warmup (ensures threads are spawned)
    for _ in range(5):
        _parallel_spawn_work(n_groups, work_array)
    
    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _parallel_spawn_work(n_groups, work_array)
        times.append(time.perf_counter() - start)
    
    mean_time = np.mean(times)
    
    # Time per group (this is the "overhead" per parallel unit)
    time_per_group_ns = (mean_time / n_groups) * 1e9
    
    # Also measure serial baseline for comparison
    serial_result = bench_numba_loop(n_groups, runs=runs, parallel=False)
    
    # Overhead = parallel time - serial time (per iteration)
    overhead_ns = time_per_group_ns - serial_result.time_per_call_ns
    
    return PrimitiveResult(
        primitive_id="O4",
        name="parallel_overhead",
        implementation="numba_parallel",
        size=n_groups,
        calls=runs,
        wall_time_s=mean_time,
        time_per_call_ns=max(0, overhead_ns),  # Can't be negative
        status="OK",
    )


# =============================================================================
# Diagnostic: Get Threading Info
# =============================================================================

def get_threading_info() -> Dict:
    """
    Get Numba threading layer information.
    
    Returns dict with:
    - layer: Threading layer name (omp, tbb, workqueue)
    - num_threads: Number of threads available
    - parallel_active: Whether parallel compilation succeeded
    - parallel_valid: Whether results should be trusted for parallel analysis
    """
    if not NUMBA_AVAILABLE:
        return {
            "layer": "unavailable",
            "num_threads": 0,
            "parallel_active": False,
            "parallel_valid": False,
            "diagnostics": "Numba not installed",
        }
    
    # Force initialization by running a parallel function
    try:
        _numba_loop_parallel(100)
        layer = numba.threading_layer()
        num_threads = numba.get_num_threads()
    except Exception as e:
        return {
            "layer": "error",
            "num_threads": 0,
            "parallel_active": False,
            "parallel_valid": False,
            "diagnostics": str(e),
        }
    
    # Check if parallel is actually working
    try:
        # Capture diagnostics output
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        _numba_loop_parallel.parallel_diagnostics(level=4)
        
        diag_str = buffer.getvalue()
        sys.stdout = old_stdout
        
        if not diag_str:
            diag_str = "EMPTY"
        
        # Check for actual parallel loop listing
        parallel_active = "Parallel loop listing" in diag_str or "loop #" in diag_str
    except Exception as e:
        diag_str = f"ERROR: {str(e)}"
        parallel_active = False
    
    return {
        "layer": layer,
        "num_threads": num_threads,
        "parallel_active": parallel_active,
        "parallel_valid": parallel_active and num_threads > 1,
        "diagnostics": diag_str[:500] if len(diag_str) > 500 else diag_str,
    }
