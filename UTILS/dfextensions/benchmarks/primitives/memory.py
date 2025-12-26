"""
Phase 12.12: Memory Primitives (M1-M5)

Measures memory bandwidth and access patterns.

Primitives:
- M1: stream_read — Contiguous array scan
- M2: gather — y = x[perm] random access
- M3: scatter_reduce — Per-group accumulation
- M4: alloc_copy — Array allocation + copy
- M4a: inner_loop_alloc — Per-group temp arrays
- M5: strided_gather — Non-contiguous index access
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
    bandwidth_gbs: float  # GB/s for memory primitives
    status: str
    
    def to_dict(self) -> Dict:
        return {
            "id": f"primitive:{self.primitive_id}:{self.name}:{self.implementation}",
            "name": self.name,
            "implementation": self.implementation,
            "size": self.size,
            "calls": self.calls,
            "wall_time_s": round(self.wall_time_s, 6),
            "bandwidth_gbs": round(self.bandwidth_gbs, 2),
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
# M1: Stream Read (Contiguous Scan)
# =============================================================================

def bench_stream_read(n_rows: int = 500_000, n_cols: int = 10, runs: int = 20) -> PrimitiveResult:
    """
    M1: Measure contiguous memory read bandwidth.
    
    This is the theoretical upper bound for memory operations.
    """
    data = np.random.randn(n_rows, n_cols)
    
    # Warmup
    _ = data.sum()
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = data.sum()  # Forces read of all data
        times.append(time.perf_counter() - start)
    
    mean_time = np.mean(times)
    bytes_read = data.nbytes
    bandwidth_gbs = bytes_read / mean_time / 1e9
    
    return PrimitiveResult(
        primitive_id="M1",
        name="stream_read",
        implementation="numpy",
        size=n_rows * n_cols,
        calls=runs,
        wall_time_s=mean_time,
        bandwidth_gbs=bandwidth_gbs,
        status="OK",
    )


# =============================================================================
# M2: Gather (Random Access)
# =============================================================================

def bench_gather(n_rows: int = 500_000, n_cols: int = 10, runs: int = 20) -> PrimitiveResult:
    """
    M2: Measure random access (gather) bandwidth.
    
    y = x[perm] where perm is a random permutation.
    This simulates the access pattern in V5 when extracting group data.
    """
    data = np.random.randn(n_rows, n_cols)
    perm = np.random.permutation(n_rows)
    
    # Warmup
    _ = data[perm]
    
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = data[perm]
        times.append(time.perf_counter() - start)
    
    mean_time = np.mean(times)
    bytes_read = data.nbytes
    bandwidth_gbs = bytes_read / mean_time / 1e9
    
    return PrimitiveResult(
        primitive_id="M2",
        name="gather",
        implementation="numpy",
        size=n_rows * n_cols,
        calls=runs,
        wall_time_s=mean_time,
        bandwidth_gbs=bandwidth_gbs,
        status="OK",
    )


# =============================================================================
# M3: Scatter Reduce (Per-Group Accumulation)
# =============================================================================

def bench_scatter_reduce(
    n_rows: int = 500_000,
    n_groups: int = 5_000,
    runs: int = 20,
) -> PrimitiveResult:
    """
    M3: Measure scatter-reduce (per-group accumulation) performance.
    
    This simulates accumulating values into group bins.
    """
    values = np.random.randn(n_rows)
    group_labels = np.random.randint(0, n_groups, size=n_rows)
    
    def scatter_reduce():
        result = np.zeros(n_groups)
        np.add.at(result, group_labels, values)
        return result
    
    mean_time, _ = _run_timed(scatter_reduce, warmup=2, runs=runs)
    time_per_group_ns = (mean_time / n_groups) * 1e9
    
    return PrimitiveResult(
        primitive_id="M3",
        name="scatter_reduce",
        implementation="numpy",
        size=n_groups,
        calls=runs,
        wall_time_s=mean_time,
        bandwidth_gbs=values.nbytes / mean_time / 1e9,
        status="OK",
    )


# =============================================================================
# M4: Allocation + Copy
# =============================================================================

def bench_alloc_copy(
    n_rows: int = 500_000,
    n_cols: int = 10,
    runs: int = 20,
) -> PrimitiveResult:
    """
    M4: Measure array allocation and copy overhead.
    
    This measures the cost of creating a new array and copying data.
    """
    source = np.random.randn(n_rows, n_cols)
    
    def alloc_copy():
        return source.copy()
    
    mean_time, _ = _run_timed(alloc_copy, warmup=2, runs=runs)
    bandwidth_gbs = source.nbytes / mean_time / 1e9
    
    return PrimitiveResult(
        primitive_id="M4",
        name="alloc_copy",
        implementation="numpy",
        size=n_rows * n_cols,
        calls=runs,
        wall_time_s=mean_time,
        bandwidth_gbs=bandwidth_gbs,
        status="OK",
    )


# =============================================================================
# M4a: Inner Loop Allocation (Per-Group Temp Arrays)
# =============================================================================

def bench_inner_loop_alloc(
    n_groups: int = 5_000,
    rows_per_group: int = 100,
    n_cols: int = 3,
    runs: int = 10,
) -> PrimitiveResult:
    """
    M4a: Measure per-group temporary array allocation cost.
    
    This simulates the cost of allocating small arrays inside a loop,
    which is a common pattern in V5 kernel.
    """
    def inner_loop_alloc():
        results = []
        for _ in range(n_groups):
            # Simulate per-group allocation
            temp = np.empty((rows_per_group, n_cols))
            results.append(temp)
        return results
    
    mean_time, _ = _run_timed(inner_loop_alloc, warmup=1, runs=runs)
    time_per_group_ns = (mean_time / n_groups) * 1e9
    
    total_bytes = n_groups * rows_per_group * n_cols * 8
    bandwidth_gbs = total_bytes / mean_time / 1e9
    
    return PrimitiveResult(
        primitive_id="M4a",
        name="inner_loop_alloc",
        implementation="numpy",
        size=n_groups,
        calls=runs,
        wall_time_s=mean_time,
        bandwidth_gbs=bandwidth_gbs,
        status="OK",
    )


# =============================================================================
# M5: Strided Gather (Non-Contiguous Index Access)
# =============================================================================

def bench_strided_gather(
    n_rows: int = 500_000,
    n_cols: int = 10,
    stride: int = 100,
    runs: int = 20,
) -> PrimitiveResult:
    """
    M5: Measure strided (non-contiguous) memory access performance.
    
    This measures access patterns like chunk_perm where indices
    are not sequential.
    """
    data = np.random.randn(n_rows, n_cols)
    # Create strided indices (every stride-th row)
    indices = np.arange(0, n_rows, stride)
    
    def strided_gather():
        return data[indices]
    
    mean_time, _ = _run_timed(strided_gather, warmup=2, runs=runs)
    bytes_read = len(indices) * n_cols * 8
    bandwidth_gbs = bytes_read / mean_time / 1e9
    
    return PrimitiveResult(
        primitive_id="M5",
        name="strided_gather",
        implementation="numpy",
        size=len(indices) * n_cols,
        calls=runs,
        wall_time_s=mean_time,
        bandwidth_gbs=bandwidth_gbs,
        status="OK",
    )
