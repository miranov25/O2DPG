"""
Phase 12.12: Primitive Microbenchmarks

This package provides standardized measurements for individual
computational primitives used in groupby regression.

Modules:
- overhead: O1-O4 (loop overhead, parallel spawn cost)
- compute: C1-C8 (linear algebra, statistics)
- memory: M1-M5 (bandwidth, gather/scatter)
- sort: S1-S4 (argsort, group boundaries)
"""

from .overhead import (
    bench_python_loop,
    bench_numba_loop,
    bench_numba_dispatch,
    bench_parallel_overhead,
    get_threading_info,
    NUMBA_AVAILABLE,
)

from .compute import (
    bench_median_numpy,
    bench_median_numba,
    bench_mad_numpy,
    bench_mad_numba,
    compare_median,
    compare_mad,
    bench_dot_XtWX,
    bench_solve,
    bench_cholesky,
    bench_cond,
    bench_matrix_vector,
    bench_residual,
    compare_dot_XtWX,
    compare_solve,
    compare_cholesky,
    compare_cond,
)

from .memory import (
    bench_stream_read,
    bench_gather,
    bench_scatter_reduce,
    bench_alloc_copy,
    bench_inner_loop_alloc,
    bench_strided_gather,
)

from .sort import (
    bench_argsort,
    bench_group_boundaries,
    bench_unique_count,
    bench_boundary_to_slices,
    compare_boundaries,
)

__all__ = [
    # Overhead (O1-O4)
    'bench_python_loop',
    'bench_numba_loop', 
    'bench_numba_dispatch',
    'bench_parallel_overhead',
    'get_threading_info',
    'NUMBA_AVAILABLE',
    # Compute (C1-C8)
    'bench_median_numpy',
    'bench_median_numba',
    'bench_mad_numpy',
    'bench_mad_numba',
    'compare_median',
    'compare_mad',
    'bench_dot_XtWX',
    'bench_solve',
    'bench_cholesky',
    'bench_cond',
    'bench_matrix_vector',
    'bench_residual',
    'compare_dot_XtWX',
    'compare_solve',
    'compare_cholesky',
    'compare_cond',
    # Memory (M1-M5)
    'bench_stream_read',
    'bench_gather',
    'bench_scatter_reduce',
    'bench_alloc_copy',
    'bench_inner_loop_alloc',
    'bench_strided_gather',
    # Sort (S1-S4)
    'bench_argsort',
    'bench_group_boundaries',
    'bench_unique_count',
    'bench_boundary_to_slices',
    'compare_boundaries',
]
