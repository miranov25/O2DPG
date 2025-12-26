"""
Phase 12.12: Compute Primitives (C1-C8)

Measures computational primitives for per-group operations.

Primitives:
- C1: dot_XtWX — Matrix accumulation X.T @ diag(W) @ X
- C2: solve_3x3 — Linear system solve
- C3: cholesky_3x3 — Cholesky decomposition
- C4: cond_number — Condition number
- C5: median — Median calculation
- C6: mad — Median Absolute Deviation
- C7: matrix_vector — Xβ computation
- C8: residual — y - Xβ
"""

import time
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

try:
    import numba
    from numba import njit
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
    status: str
    ratio_vs_baseline: Optional[float] = None
    
    def to_dict(self) -> Dict:
        d = {
            "id": f"primitive:{self.primitive_id}:{self.name}:{self.implementation}",
            "name": self.name,
            "implementation": self.implementation,
            "size": self.size,
            "calls": self.calls,
            "wall_time_s": round(self.wall_time_s, 6),
            "time_per_call_ns": round(self.time_per_call_ns, 2),
            "status": self.status,
        }
        if self.ratio_vs_baseline is not None:
            d["ratio_vs_numpy"] = round(self.ratio_vs_baseline, 2)
        return d


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
# C1: dot_XtWX — Matrix accumulation
# =============================================================================

def _dot_XtWX_numpy(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute X.T @ diag(W) @ X using NumPy."""
    # Efficient: (X.T * W) @ X
    return (X.T * W) @ X


if NUMBA_AVAILABLE:
    @njit
    def _dot_XtWX_numba(X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Compute X.T @ diag(W) @ X using Numba."""
        n_rows, n_cols = X.shape
        result = np.zeros((n_cols, n_cols))
        
        for i in range(n_rows):
            w = W[i]
            for j in range(n_cols):
                for k in range(n_cols):
                    result[j, k] += X[i, j] * w * X[i, k]
        
        return result


def bench_dot_XtWX(
    n_rows: int = 100,
    n_cols: int = 3,
    n_calls: int = 30000,
    runs: int = 5,
    implementation: str = "numpy",
) -> PrimitiveResult:
    """
    C1: Measure X.T @ diag(W) @ X performance.
    """
    X = np.random.randn(n_rows, n_cols)
    W = np.random.rand(n_rows)
    
    if implementation == "numpy":
        func = _dot_XtWX_numpy
    elif implementation == "numba" and NUMBA_AVAILABLE:
        _dot_XtWX_numba(X[:10], W[:10])  # Warmup JIT
        func = _dot_XtWX_numba
    else:
        return PrimitiveResult(
            "C1", "dot_XtWX", implementation, n_rows, 0, 0.0, 0.0,
            "ERROR:NOT_AVAILABLE"
        )
    
    def run_batch():
        for _ in range(n_calls):
            func(X, W)
    
    mean_time, _ = _run_timed(run_batch, warmup=1, runs=runs)
    time_per_call_ns = (mean_time / n_calls) * 1e9
    
    return PrimitiveResult(
        primitive_id="C1",
        name="dot_XtWX",
        implementation=implementation,
        size=n_rows,
        calls=n_calls,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_call_ns,
        status="OK",
    )


# =============================================================================
# C2: solve_3x3 — Linear system solve
# =============================================================================

def _solve_numpy(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ax = b using NumPy."""
    return np.linalg.solve(A, b)


if NUMBA_AVAILABLE:
    @njit
    def _solve_numba(A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve Ax = b using Numba (calls LAPACK internally)."""
        return np.linalg.solve(A, b)


def bench_solve(
    n_params: int = 3,
    n_calls: int = 30000,
    runs: int = 5,
    implementation: str = "numpy",
) -> PrimitiveResult:
    """
    C2: Measure linear system solve performance.
    """
    # Create positive definite matrix
    A = np.random.randn(n_params, n_params)
    A = A @ A.T + np.eye(n_params) * 0.1
    b = np.random.randn(n_params)
    
    if implementation == "numpy":
        func = _solve_numpy
    elif implementation == "numba" and NUMBA_AVAILABLE:
        _solve_numba(A, b)  # Warmup JIT
        func = _solve_numba
    else:
        return PrimitiveResult(
            "C2", "solve", implementation, n_params, 0, 0.0, 0.0,
            "ERROR:NOT_AVAILABLE"
        )
    
    def run_batch():
        for _ in range(n_calls):
            func(A, b)
    
    mean_time, _ = _run_timed(run_batch, warmup=1, runs=runs)
    time_per_call_ns = (mean_time / n_calls) * 1e9
    
    return PrimitiveResult(
        primitive_id="C2",
        name="solve",
        implementation=implementation,
        size=n_params,
        calls=n_calls,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_call_ns,
        status="OK",
    )


# =============================================================================
# C3: cholesky_3x3 — Cholesky decomposition
# =============================================================================

def _cholesky_numpy(A: np.ndarray) -> np.ndarray:
    """Cholesky decomposition using NumPy."""
    return np.linalg.cholesky(A)


if NUMBA_AVAILABLE:
    @njit
    def _cholesky_numba(A: np.ndarray) -> np.ndarray:
        """Cholesky decomposition using Numba."""
        return np.linalg.cholesky(A)


def bench_cholesky(
    n_params: int = 3,
    n_calls: int = 30000,
    runs: int = 5,
    implementation: str = "numpy",
) -> PrimitiveResult:
    """
    C3: Measure Cholesky decomposition performance.
    """
    A = np.random.randn(n_params, n_params)
    A = A @ A.T + np.eye(n_params) * 0.1
    
    if implementation == "numpy":
        func = _cholesky_numpy
    elif implementation == "numba" and NUMBA_AVAILABLE:
        _cholesky_numba(A)  # Warmup JIT
        func = _cholesky_numba
    else:
        return PrimitiveResult(
            "C3", "cholesky", implementation, n_params, 0, 0.0, 0.0,
            "ERROR:NOT_AVAILABLE"
        )
    
    def run_batch():
        for _ in range(n_calls):
            func(A)
    
    mean_time, _ = _run_timed(run_batch, warmup=1, runs=runs)
    time_per_call_ns = (mean_time / n_calls) * 1e9
    
    return PrimitiveResult(
        primitive_id="C3",
        name="cholesky",
        implementation=implementation,
        size=n_params,
        calls=n_calls,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_call_ns,
        status="OK",
    )


# =============================================================================
# C4: cond_number — Condition number
# =============================================================================

def _cond_numpy(A: np.ndarray) -> float:
    """Condition number using NumPy (SVD-based)."""
    return np.linalg.cond(A)


if NUMBA_AVAILABLE:
    @njit
    def _cond_numba_proxy(A: np.ndarray) -> float:
        """
        Condition number proxy using Numba (Cholesky-based).
        
        Uses ratio of diagonal elements as a proxy for condition number.
        Much faster than SVD for small matrices.
        """
        try:
            L = np.linalg.cholesky(A)
            diag = np.diag(L)
            return (np.max(diag) / np.min(diag)) ** 2
        except:
            return np.inf


def bench_cond(
    n_params: int = 3,
    n_calls: int = 30000,
    runs: int = 5,
    implementation: str = "numpy",
) -> PrimitiveResult:
    """
    C4: Measure condition number performance.
    """
    A = np.random.randn(n_params, n_params)
    A = A @ A.T + np.eye(n_params) * 0.1
    
    if implementation == "numpy":
        func = _cond_numpy
    elif implementation == "numba" and NUMBA_AVAILABLE:
        _cond_numba_proxy(A)  # Warmup JIT
        func = _cond_numba_proxy
    else:
        return PrimitiveResult(
            "C4", "cond", implementation, n_params, 0, 0.0, 0.0,
            "ERROR:NOT_AVAILABLE"
        )
    
    def run_batch():
        for _ in range(n_calls):
            func(A)
    
    mean_time, _ = _run_timed(run_batch, warmup=1, runs=runs)
    time_per_call_ns = (mean_time / n_calls) * 1e9
    
    return PrimitiveResult(
        primitive_id="C4",
        name="cond",
        implementation=implementation,
        size=n_params,
        calls=n_calls,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_call_ns,
        status="OK",
    )


# =============================================================================
# C5: Median
# =============================================================================

def _median_numpy(arr: np.ndarray) -> float:
    """NumPy median (C implementation)."""
    return np.median(arr)


if NUMBA_AVAILABLE:
    @njit
    def _median_numba(arr: np.ndarray) -> float:
        """
        Numba median implementation (sort-based).
        
        This is the likely implementation in V5 _process_chunk_numba.
        O(n log n) complexity due to full sort.
        """
        sorted_arr = np.sort(arr)
        n = len(sorted_arr)
        if n % 2 == 0:
            return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0
        else:
            return sorted_arr[n // 2]
    
    @njit
    def _median_numba_partition(arr: np.ndarray) -> float:
        """
        Numba median using partition (O(n) average).
        
        This is the optimized approach — uses np.partition internally.
        """
        n = len(arr)
        if n % 2 == 1:
            return np.partition(arr, n // 2)[n // 2]
        else:
            # For even n, need two middle elements
            mid = n // 2
            partitioned = np.partition(arr, [mid - 1, mid])
            return (partitioned[mid - 1] + partitioned[mid]) / 2.0


def bench_median_numpy(
    array_size: int = 100,
    n_calls: int = 60000,  # 5K groups × 6 fits × 2 calls
    runs: int = 5,
) -> PrimitiveResult:
    """
    C5: Measure NumPy median performance.
    
    Parameters
    ----------
    array_size : int
        Size of array to compute median (rows per group)
    n_calls : int
        Total number of median calls to simulate
    runs : int
        Number of benchmark runs
        
    Returns
    -------
    PrimitiveResult with time per call in nanoseconds
    """
    # Create test array
    arr = np.random.randn(array_size)
    
    def run_batch():
        for _ in range(n_calls):
            _median_numpy(arr)
    
    mean_time, _ = _run_timed(run_batch, warmup=1, runs=runs)
    time_per_call_ns = (mean_time / n_calls) * 1e9
    
    return PrimitiveResult(
        primitive_id="C5",
        name="median",
        implementation="numpy",
        size=array_size,
        calls=n_calls,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_call_ns,
        status="OK",
    )


def bench_median_numba(
    array_size: int = 100,
    n_calls: int = 60000,
    runs: int = 5,
    use_partition: bool = False,
) -> PrimitiveResult:
    """
    C5: Measure Numba median performance.
    
    Parameters
    ----------
    array_size : int
        Size of array to compute median (rows per group)
    n_calls : int
        Total number of median calls
    runs : int
        Number of benchmark runs
    use_partition : bool
        If True, use O(n) partition-based median
        If False, use O(n log n) sort-based median (like V5)
        
    Returns
    -------
    PrimitiveResult with time per call in nanoseconds
    """
    if not NUMBA_AVAILABLE:
        return PrimitiveResult(
            primitive_id="C5",
            name="median",
            implementation="numba",
            size=array_size,
            calls=0,
            wall_time_s=0.0,
            time_per_call_ns=0.0,
            status="ERROR:NUMBA_NOT_AVAILABLE",
        )
    
    # Create test array
    arr = np.random.randn(array_size)
    
    # Select implementation
    median_func = _median_numba_partition if use_partition else _median_numba
    impl_name = "numba_partition" if use_partition else "numba_sort"
    
    # Warmup Numba JIT
    for _ in range(10):
        median_func(arr)
    
    def run_batch():
        for _ in range(n_calls):
            median_func(arr)
    
    mean_time, _ = _run_timed(run_batch, warmup=1, runs=runs)
    time_per_call_ns = (mean_time / n_calls) * 1e9
    
    return PrimitiveResult(
        primitive_id="C5",
        name="median",
        implementation=impl_name,
        size=array_size,
        calls=n_calls,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_call_ns,
        status="OK",
    )


def compare_median(
    array_size: int = 100,
    n_calls: int = 60000,
    runs: int = 5,
) -> Tuple[PrimitiveResult, PrimitiveResult, float]:
    """
    Compare NumPy vs Numba median performance.
    
    Returns
    -------
    (numpy_result, numba_result, ratio)
    
    ratio > 1 means Numba is slower than NumPy
    """
    numpy_result = bench_median_numpy(array_size, n_calls, runs)
    numba_result = bench_median_numba(array_size, n_calls, runs, use_partition=False)
    
    if numba_result.time_per_call_ns > 0:
        ratio = numba_result.time_per_call_ns / numpy_result.time_per_call_ns
        numba_result.ratio_vs_baseline = ratio
        
        # Status based on ratio
        if ratio > 3.0:
            numba_result.status = "SLOW"
        elif ratio > 1.5:
            numba_result.status = "WARN"
    else:
        ratio = float('inf')
    
    return numpy_result, numba_result, ratio


# =============================================================================
# C6: MAD (Median Absolute Deviation)
# =============================================================================

def _mad_numpy(arr: np.ndarray) -> float:
    """NumPy MAD (2 median calls)."""
    med = np.median(arr)
    return np.median(np.abs(arr - med))


if NUMBA_AVAILABLE:
    @njit
    def _mad_numba(arr: np.ndarray) -> float:
        """
        Numba MAD implementation.
        
        Uses sort-based median (like V5).
        """
        # First median
        sorted_arr = np.sort(arr)
        n = len(sorted_arr)
        if n % 2 == 0:
            med = (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0
        else:
            med = sorted_arr[n // 2]
        
        # Absolute deviations
        abs_dev = np.abs(arr - med)
        
        # Second median
        sorted_dev = np.sort(abs_dev)
        if n % 2 == 0:
            return (sorted_dev[n // 2 - 1] + sorted_dev[n // 2]) / 2.0
        else:
            return sorted_dev[n // 2]


def bench_mad_numpy(
    array_size: int = 100,
    n_calls: int = 30000,  # 5K groups × 6 fits
    runs: int = 5,
) -> PrimitiveResult:
    """
    C6: Measure NumPy MAD performance.
    """
    arr = np.random.randn(array_size)
    
    def run_batch():
        for _ in range(n_calls):
            _mad_numpy(arr)
    
    mean_time, _ = _run_timed(run_batch, warmup=1, runs=runs)
    time_per_call_ns = (mean_time / n_calls) * 1e9
    
    return PrimitiveResult(
        primitive_id="C6",
        name="mad",
        implementation="numpy",
        size=array_size,
        calls=n_calls,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_call_ns,
        status="OK",
    )


def bench_mad_numba(
    array_size: int = 100,
    n_calls: int = 30000,
    runs: int = 5,
) -> PrimitiveResult:
    """
    C6: Measure Numba MAD performance.
    """
    if not NUMBA_AVAILABLE:
        return PrimitiveResult(
            primitive_id="C6",
            name="mad",
            implementation="numba",
            size=array_size,
            calls=0,
            wall_time_s=0.0,
            time_per_call_ns=0.0,
            status="ERROR:NUMBA_NOT_AVAILABLE",
        )
    
    arr = np.random.randn(array_size)
    
    # Warmup
    for _ in range(10):
        _mad_numba(arr)
    
    def run_batch():
        for _ in range(n_calls):
            _mad_numba(arr)
    
    mean_time, _ = _run_timed(run_batch, warmup=1, runs=runs)
    time_per_call_ns = (mean_time / n_calls) * 1e9
    
    return PrimitiveResult(
        primitive_id="C6",
        name="mad",
        implementation="numba",
        size=array_size,
        calls=n_calls,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_call_ns,
        status="OK",
    )


def compare_mad(
    array_size: int = 100,
    n_calls: int = 30000,
    runs: int = 5,
) -> Tuple[PrimitiveResult, PrimitiveResult, float]:
    """
    Compare NumPy vs Numba MAD performance.
    
    Returns (numpy_result, numba_result, ratio)
    """
    numpy_result = bench_mad_numpy(array_size, n_calls, runs)
    numba_result = bench_mad_numba(array_size, n_calls, runs)
    
    if numba_result.time_per_call_ns > 0:
        ratio = numba_result.time_per_call_ns / numpy_result.time_per_call_ns
        numba_result.ratio_vs_baseline = ratio
        
        if ratio > 3.0:
            numba_result.status = "SLOW"
        elif ratio > 1.5:
            numba_result.status = "WARN"
    else:
        ratio = float('inf')
    
    return numpy_result, numba_result, ratio


# =============================================================================
# C7: matrix_vector — Xβ computation
# =============================================================================

def _matrix_vector_numpy(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute X @ beta using NumPy."""
    return X @ beta


if NUMBA_AVAILABLE:
    @njit
    def _matrix_vector_numba(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute X @ beta using Numba."""
        return X @ beta


def bench_matrix_vector(
    n_rows: int = 100,
    n_cols: int = 3,
    n_calls: int = 30000,
    runs: int = 5,
    implementation: str = "numpy",
) -> PrimitiveResult:
    """
    C7: Measure matrix-vector multiplication performance.
    """
    X = np.random.randn(n_rows, n_cols)
    beta = np.random.randn(n_cols)
    
    if implementation == "numpy":
        func = _matrix_vector_numpy
    elif implementation == "numba" and NUMBA_AVAILABLE:
        _matrix_vector_numba(X, beta)  # Warmup JIT
        func = _matrix_vector_numba
    else:
        return PrimitiveResult(
            "C7", "matrix_vector", implementation, n_rows, 0, 0.0, 0.0,
            "ERROR:NOT_AVAILABLE"
        )
    
    def run_batch():
        for _ in range(n_calls):
            func(X, beta)
    
    mean_time, _ = _run_timed(run_batch, warmup=1, runs=runs)
    time_per_call_ns = (mean_time / n_calls) * 1e9
    
    return PrimitiveResult(
        primitive_id="C7",
        name="matrix_vector",
        implementation=implementation,
        size=n_rows,
        calls=n_calls,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_call_ns,
        status="OK",
    )


# =============================================================================
# C8: residual — y - Xβ
# =============================================================================

def _residual_numpy(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Compute y - X @ beta using NumPy."""
    return y - X @ beta


if NUMBA_AVAILABLE:
    @njit
    def _residual_numba(y: np.ndarray, X: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Compute y - X @ beta using Numba."""
        return y - X @ beta


def bench_residual(
    n_rows: int = 100,
    n_cols: int = 3,
    n_calls: int = 30000,
    runs: int = 5,
    implementation: str = "numpy",
) -> PrimitiveResult:
    """
    C8: Measure residual computation performance.
    """
    y = np.random.randn(n_rows)
    X = np.random.randn(n_rows, n_cols)
    beta = np.random.randn(n_cols)
    
    if implementation == "numpy":
        func = lambda: _residual_numpy(y, X, beta)
    elif implementation == "numba" and NUMBA_AVAILABLE:
        _residual_numba(y, X, beta)  # Warmup JIT
        func = lambda: _residual_numba(y, X, beta)
    else:
        return PrimitiveResult(
            "C8", "residual", implementation, n_rows, 0, 0.0, 0.0,
            "ERROR:NOT_AVAILABLE"
        )
    
    def run_batch():
        for _ in range(n_calls):
            func()
    
    mean_time, _ = _run_timed(run_batch, warmup=1, runs=runs)
    time_per_call_ns = (mean_time / n_calls) * 1e9
    
    return PrimitiveResult(
        primitive_id="C8",
        name="residual",
        implementation=implementation,
        size=n_rows,
        calls=n_calls,
        wall_time_s=mean_time,
        time_per_call_ns=time_per_call_ns,
        status="OK",
    )


# =============================================================================
# Comparison Functions for C1-C4
# =============================================================================

def compare_dot_XtWX(
    n_rows: int = 100,
    n_cols: int = 3,
    n_calls: int = 30000,
    runs: int = 5,
) -> Tuple[PrimitiveResult, PrimitiveResult, float]:
    """Compare NumPy vs Numba for X.T @ W @ X."""
    numpy_result = bench_dot_XtWX(n_rows, n_cols, n_calls, runs, "numpy")
    numba_result = bench_dot_XtWX(n_rows, n_cols, n_calls, runs, "numba")
    
    if numba_result.time_per_call_ns > 0 and numpy_result.time_per_call_ns > 0:
        ratio = numba_result.time_per_call_ns / numpy_result.time_per_call_ns
        numba_result.ratio_vs_baseline = ratio
    else:
        ratio = float('inf')
    
    return numpy_result, numba_result, ratio


def compare_solve(
    n_params: int = 3,
    n_calls: int = 30000,
    runs: int = 5,
) -> Tuple[PrimitiveResult, PrimitiveResult, float]:
    """Compare NumPy vs Numba for linear solve."""
    numpy_result = bench_solve(n_params, n_calls, runs, "numpy")
    numba_result = bench_solve(n_params, n_calls, runs, "numba")
    
    if numba_result.time_per_call_ns > 0 and numpy_result.time_per_call_ns > 0:
        ratio = numba_result.time_per_call_ns / numpy_result.time_per_call_ns
        numba_result.ratio_vs_baseline = ratio
    else:
        ratio = float('inf')
    
    return numpy_result, numba_result, ratio


def compare_cholesky(
    n_params: int = 3,
    n_calls: int = 30000,
    runs: int = 5,
) -> Tuple[PrimitiveResult, PrimitiveResult, float]:
    """Compare NumPy vs Numba for Cholesky decomposition."""
    numpy_result = bench_cholesky(n_params, n_calls, runs, "numpy")
    numba_result = bench_cholesky(n_params, n_calls, runs, "numba")
    
    if numba_result.time_per_call_ns > 0 and numpy_result.time_per_call_ns > 0:
        ratio = numba_result.time_per_call_ns / numpy_result.time_per_call_ns
        numba_result.ratio_vs_baseline = ratio
    else:
        ratio = float('inf')
    
    return numpy_result, numba_result, ratio


def compare_cond(
    n_params: int = 3,
    n_calls: int = 30000,
    runs: int = 5,
) -> Tuple[PrimitiveResult, PrimitiveResult, float]:
    """Compare NumPy vs Numba for condition number."""
    numpy_result = bench_cond(n_params, n_calls, runs, "numpy")
    numba_result = bench_cond(n_params, n_calls, runs, "numba")
    
    if numba_result.time_per_call_ns > 0 and numpy_result.time_per_call_ns > 0:
        ratio = numba_result.time_per_call_ns / numpy_result.time_per_call_ns
        numba_result.ratio_vs_baseline = ratio
    else:
        ratio = float('inf')
    
    return numpy_result, numba_result, ratio
