"""
Phase 12.12: Sort/Index Primitives (S1-S4)

Measures sorting and grouping operations.

Primitives:
- S1: argsort — Sort indices (lexsort for multi-column)
- S2: group_boundaries — Find group starts from sorted data
- S3: unique_count — Count unique groups
- S4: boundary_to_slices — Materialize slices from boundaries
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
    time_per_row_ns: float
    status: str
    
    def to_dict(self) -> Dict:
        return {
            "id": f"primitive:{self.primitive_id}:{self.name}:{self.implementation}",
            "name": self.name,
            "implementation": self.implementation,
            "size": self.size,
            "calls": self.calls,
            "wall_time_s": round(self.wall_time_s, 6),
            "time_per_row_ns": round(self.time_per_row_ns, 2),
            "status": self.status,
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
# S1: Argsort
# =============================================================================

def bench_argsort(
    n_rows: int = 500_000,
    n_keys: int = 1,
    runs: int = 10,
) -> PrimitiveResult:
    """
    S1: Measure argsort performance.
    
    Parameters
    ----------
    n_rows : int
        Number of rows to sort
    n_keys : int
        Number of sort keys (1 = simple argsort, >1 = lexsort)
    runs : int
        Number of benchmark runs
    """
    if n_keys == 1:
        # Single key argsort
        data = np.random.randint(0, n_rows // 100, size=n_rows)
        
        def sort_func():
            return np.argsort(data)
    else:
        # Multi-key lexsort
        keys = [np.random.randint(0, 100, size=n_rows) for _ in range(n_keys)]
        
        def sort_func():
            return np.lexsort(keys)
    
    mean_time, _ = _run_timed(sort_func, warmup=2, runs=runs)
    time_per_row_ns = (mean_time / n_rows) * 1e9
    
    return PrimitiveResult(
        primitive_id="S1",
        name="argsort",
        implementation=f"numpy_{'lexsort' if n_keys > 1 else 'argsort'}",
        size=n_rows,
        calls=runs,
        wall_time_s=mean_time,
        time_per_row_ns=time_per_row_ns,
        status="OK",
    )


# =============================================================================
# S2: Group Boundaries
# =============================================================================

def _find_boundaries_numpy(sorted_groups: np.ndarray) -> np.ndarray:
    """Find group boundary indices from sorted group labels."""
    # Find where groups change
    diff = np.diff(sorted_groups)
    change_points = np.where(diff != 0)[0] + 1
    # Add start and end
    boundaries = np.concatenate([[0], change_points, [len(sorted_groups)]])
    return boundaries


if NUMBA_AVAILABLE:
    @njit
    def _find_boundaries_numba(sorted_groups: np.ndarray) -> np.ndarray:
        """Numba version of group boundary finding."""
        n = len(sorted_groups)
        # First pass: count boundaries
        n_boundaries = 2  # start and end
        for i in range(1, n):
            if sorted_groups[i] != sorted_groups[i-1]:
                n_boundaries += 1
        
        # Second pass: fill boundaries
        boundaries = np.empty(n_boundaries, dtype=np.int64)
        boundaries[0] = 0
        idx = 1
        for i in range(1, n):
            if sorted_groups[i] != sorted_groups[i-1]:
                boundaries[idx] = i
                idx += 1
        boundaries[n_boundaries - 1] = n
        
        return boundaries


def bench_group_boundaries(
    n_rows: int = 500_000,
    n_groups: int = 5_000,
    runs: int = 20,
    implementation: str = "numpy",
) -> PrimitiveResult:
    """
    S2: Measure group boundary finding performance.
    """
    # Create sorted group labels
    group_labels = np.repeat(np.arange(n_groups), n_rows // n_groups)
    
    if implementation == "numpy":
        func = _find_boundaries_numpy
    elif implementation == "numba" and NUMBA_AVAILABLE:
        # Warmup JIT
        _find_boundaries_numba(group_labels[:100])
        func = _find_boundaries_numba
    else:
        return PrimitiveResult(
            "S2", "group_boundaries", implementation, n_rows, 0, 0.0, 0.0,
            "ERROR:NOT_AVAILABLE"
        )
    
    mean_time, _ = _run_timed(func, group_labels, warmup=2, runs=runs)
    time_per_row_ns = (mean_time / n_rows) * 1e9
    
    return PrimitiveResult(
        primitive_id="S2",
        name="group_boundaries",
        implementation=implementation,
        size=n_rows,
        calls=runs,
        wall_time_s=mean_time,
        time_per_row_ns=time_per_row_ns,
        status="OK",
    )


# =============================================================================
# S3: Unique Count
# =============================================================================

def bench_unique_count(
    n_rows: int = 500_000,
    n_groups: int = 5_000,
    runs: int = 20,
) -> PrimitiveResult:
    """
    S3: Measure unique group counting performance.
    """
    group_labels = np.random.randint(0, n_groups, size=n_rows)
    
    def count_unique():
        return len(np.unique(group_labels))
    
    mean_time, _ = _run_timed(count_unique, warmup=2, runs=runs)
    time_per_row_ns = (mean_time / n_rows) * 1e9
    
    return PrimitiveResult(
        primitive_id="S3",
        name="unique_count",
        implementation="numpy",
        size=n_rows,
        calls=runs,
        wall_time_s=mean_time,
        time_per_row_ns=time_per_row_ns,
        status="OK",
    )


# =============================================================================
# S4: Boundary to Slices
# =============================================================================

def bench_boundary_to_slices(
    n_groups: int = 5_000,
    rows_per_group: int = 100,
    runs: int = 20,
) -> PrimitiveResult:
    """
    S4: Measure slice materialization from boundaries.
    
    This measures the overhead of converting boundary indices
    to actual slice objects or index arrays.
    """
    n_rows = n_groups * rows_per_group
    boundaries = np.arange(0, n_rows + 1, rows_per_group)
    data = np.random.randn(n_rows, 3)
    
    def materialize_slices():
        """Extract all group slices."""
        slices = []
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i+1]
            slices.append(data[start:end])
        return slices
    
    mean_time, _ = _run_timed(materialize_slices, warmup=2, runs=runs)
    time_per_group_ns = (mean_time / n_groups) * 1e9
    
    return PrimitiveResult(
        primitive_id="S4",
        name="boundary_to_slices",
        implementation="numpy",
        size=n_groups,
        calls=runs,
        wall_time_s=mean_time,
        time_per_row_ns=time_per_group_ns,  # Actually per-group
        status="OK",
    )


# =============================================================================
# Comparison Functions
# =============================================================================

def compare_boundaries(
    n_rows: int = 500_000,
    n_groups: int = 5_000,
    runs: int = 20,
) -> Tuple[PrimitiveResult, PrimitiveResult, float]:
    """Compare NumPy vs Numba group boundary finding."""
    numpy_result = bench_group_boundaries(n_rows, n_groups, runs, "numpy")
    numba_result = bench_group_boundaries(n_rows, n_groups, runs, "numba")
    
    if numba_result.time_per_row_ns > 0 and numpy_result.time_per_row_ns > 0:
        ratio = numba_result.time_per_row_ns / numpy_result.time_per_row_ns
    else:
        ratio = float('inf')
    
    return numpy_result, numba_result, ratio
