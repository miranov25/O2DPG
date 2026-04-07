"""
Tests for Phase 13.16.GB — Evaluator Lookup Mode.

Tests method='lookup' (direct integer indexing) and per-dimension method dict.
Key invariance: lookup ≡ nearest on integer grids (bit-identical).
"""
import numpy as np
import pandas as pd
import pytest
import time

try:
    from groupby_regression_evaluator import GroupByRegressionEvaluator
    from groupby_regression_optimized import make_parallel_fit_v4
except ImportError:
    from ..groupby_regression_evaluator import GroupByRegressionEvaluator
    from ..groupby_regression_optimized import make_parallel_fit_v4


# ── Fixtures ──

@pytest.fixture
def evaluator_3d():
    """3D evaluator with 10×10×10 grid, 1 predictor, 1 target."""
    rng = np.random.RandomState(42)
    n_per_bin = 50
    g0 = np.repeat(np.arange(10), n_per_bin * 100)
    g1 = np.tile(np.repeat(np.arange(10), n_per_bin * 10), 10)
    g2 = np.tile(np.repeat(np.arange(10), n_per_bin), 100)
    n_total = len(g0)

    df = pd.DataFrame({
        'g0': g0, 'g1': g1, 'g2': g2,
        'x': rng.normal(0, 1, n_total),
        'y': 2.0 * g0 + 0.5 * g1 - g2 + rng.normal(0, 0.1, n_total),
    })

    _, dfGB = make_parallel_fit_v4(
        df=df, gb_columns=['g0', 'g1', 'g2'],
        fit_columns=['y'], linear_columns=['x'], suffix='_f',
    )

    ev = GroupByRegressionEvaluator.from_dfGB(
        dfGB, group_columns=['g0', 'g1', 'g2'],
        predictor_columns=['x'], targets=['y'], suffix='_f',
    )
    return ev


# ═══════════════════════════════════════════════════════════════
# Test 1: Lookup ≡ nearest for integers (INVARIANCE — key gate)
# ═══════════════════════════════════════════════════════════════

def test_lookup_equals_nearest_for_integers(evaluator_3d):
    """On integer grid, method='lookup' ≡ method='nearest' (bit-identical)."""
    rng = np.random.RandomState(123)
    n = 10000

    positions = {
        'g0': rng.randint(0, 10, n),
        'g1': rng.randint(0, 10, n),
        'g2': rng.randint(0, 10, n),
    }
    predictors = {'x': rng.normal(0, 1, n)}

    result_nearest = evaluator_3d.evaluate(
        positions=positions, predictors=predictors, method='nearest')
    result_lookup = evaluator_3d.evaluate(
        positions=positions, predictors=predictors, method='lookup')

    np.testing.assert_array_equal(
        result_nearest['y'], result_lookup['y'],
        err_msg="lookup ≠ nearest on integer grid")


# ═══════════════════════════════════════════════════════════════
# Test 2: Out-of-bounds with bounds='nan' (smoke)
# ═══════════════════════════════════════════════════════════════

def test_lookup_out_of_bounds_nan(evaluator_3d):
    """Out-of-bounds positions with bounds='nan' return NaN."""
    positions = {
        'g0': np.array([0, 5, -1, 15]),
        'g1': np.array([0, 5, 0, 5]),
        'g2': np.array([0, 5, 0, 5]),
    }
    predictors = {'x': np.array([1.0, 1.0, 1.0, 1.0])}

    result = evaluator_3d.evaluate(
        positions=positions, predictors=predictors,
        method='lookup', bounds='nan')

    vals = result['y']
    assert np.isfinite(vals[0]), "In-bounds point should be finite"
    assert np.isfinite(vals[1]), "In-bounds point should be finite"
    assert np.isnan(vals[2]), "Out-of-bounds (negative) should be NaN"
    assert np.isnan(vals[3]), "Out-of-bounds (too large) should be NaN"


# ═══════════════════════════════════════════════════════════════
# Test 3: Out-of-bounds with bounds='clamp' (smoke)
# ═══════════════════════════════════════════════════════════════

def test_lookup_out_of_bounds_clamp(evaluator_3d):
    """Out-of-bounds positions with bounds='clamp' clip to edge."""
    positions_oob = {
        'g0': np.array([-1, 15]),
        'g1': np.array([0, 0]),
        'g2': np.array([0, 0]),
    }
    positions_edge = {
        'g0': np.array([0, 9]),
        'g1': np.array([0, 0]),
        'g2': np.array([0, 0]),
    }
    predictors = {'x': np.array([1.0, 1.0])}

    result_oob = evaluator_3d.evaluate(
        positions=positions_oob, predictors=predictors,
        method='lookup', bounds='clamp')
    result_edge = evaluator_3d.evaluate(
        positions=positions_edge, predictors=predictors,
        method='lookup', bounds='clamp')

    np.testing.assert_array_equal(
        result_oob['y'], result_edge['y'],
        err_msg="Clamped out-of-bounds ≠ edge values")


# ═══════════════════════════════════════════════════════════════
# Test 4: Non-integer positions raise ValueError (smoke)
# ═══════════════════════════════════════════════════════════════

def test_lookup_non_integer_raises(evaluator_3d):
    """Float positions with method='lookup' raise ValueError."""
    positions = {
        'g0': np.array([0.5, 1.7, 2.3]),
        'g1': np.array([0, 1, 2]),
        'g2': np.array([0, 1, 2]),
    }
    predictors = {'x': np.array([1.0, 1.0, 1.0])}

    with pytest.raises(ValueError, match="integer"):
        evaluator_3d.evaluate(
            positions=positions, predictors=predictors, method='lookup')


# ═══════════════════════════════════════════════════════════════
# Test 5: Per-dimension method dict (smoke)
# ═══════════════════════════════════════════════════════════════

def test_per_dimension_method_dict(evaluator_3d):
    """Mixed lookup + linear produces valid output."""
    rng = np.random.RandomState(42)
    n = 1000

    positions = {
        'g0': rng.randint(0, 10, n),
        'g1': rng.randint(0, 10, n),
        'g2': rng.uniform(0, 9, n),
    }
    predictors = {'x': rng.normal(0, 1, n)}

    result = evaluator_3d.evaluate(
        positions=positions, predictors=predictors,
        method={'g0': 'lookup', 'g1': 'lookup', 'g2': 'linear'},
    )

    vals = result['y']
    assert len(vals) == n
    assert np.isfinite(vals).all(), "All points should be finite"

    # At integer g2 positions, mixed should equal pure lookup
    positions_int = positions.copy()
    positions_int['g2'] = np.round(positions['g2']).astype(int)

    result_lookup = evaluator_3d.evaluate(
        positions=positions_int, predictors=predictors, method='lookup')

    at_integer = np.abs(positions['g2'] - np.round(positions['g2'])) < 1e-10
    if at_integer.sum() > 0:
        np.testing.assert_allclose(
            vals[at_integer], result_lookup['y'][at_integer],
            rtol=1e-10,
            err_msg="At integer positions, mixed method ≠ lookup")


# ═══════════════════════════════════════════════════════════════
# Test 6: Lookup performance (performance — directional)
# ═══════════════════════════════════════════════════════════════

def test_lookup_performance(evaluator_3d):
    """Lookup is faster than nearest on integer grid."""
    rng = np.random.RandomState(42)
    n = 100_000

    positions = {
        'g0': rng.randint(0, 10, n),
        'g1': rng.randint(0, 10, n),
        'g2': rng.randint(0, 10, n),
    }
    predictors = {'x': rng.normal(0, 1, n)}

    # Warmup
    evaluator_3d.evaluate(positions=positions, predictors=predictors, method='lookup')
    evaluator_3d.evaluate(positions=positions, predictors=predictors, method='nearest')

    # Time lookup
    t0 = time.time()
    for _ in range(5):
        evaluator_3d.evaluate(positions=positions, predictors=predictors, method='lookup')
    t_lookup = (time.time() - t0) / 5

    # Time nearest
    t0 = time.time()
    for _ in range(5):
        evaluator_3d.evaluate(positions=positions, predictors=predictors, method='nearest')
    t_nearest = (time.time() - t0) / 5

    speedup = t_nearest / t_lookup
    print(f"\n  [PERF] lookup: {t_lookup:.4f}s, nearest: {t_nearest:.4f}s, speedup: {speedup:.1f}×")

    assert speedup > 1.5, \
        f"Lookup should be faster than nearest: {speedup:.1f}× (expect >1.5×)"
