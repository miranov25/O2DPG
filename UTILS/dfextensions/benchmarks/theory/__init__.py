"""
Phase 12.12: Theoretical Performance Models

This package provides roofline analysis and performance prediction
for V4 and V5 groupby regression implementations.

Modules:
- roofline: Roofline model equations and efficiency calculation
- model_v4: V4 theoretical time prediction (future)
- model_v5: V5 theoretical time prediction (future)
"""

from .roofline import (
    parallel_overhead_bound,
    parallel_efficiency,
    is_overhead_dominated,
    roofline_bound,
    diagnose_parallel_failure,
)

__all__ = [
    'parallel_overhead_bound',
    'parallel_efficiency',
    'is_overhead_dominated',
    'roofline_bound',
    'diagnose_parallel_failure',
]
