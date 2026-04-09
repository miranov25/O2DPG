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


# ═══════════════════════════════════════════════════════════════
# Phase 13.16.GB-FIX2 — Bug fix tests (F2, F3, F4)
# ═══════════════════════════════════════════════════════════════
#
# Each test specifies path-controlling parameters explicitly per failure
# mode #11 (invariance tests must not rely on auto/default dispatch). Each
# ValueError assertion uses pytest.raises(..., match=...) per C7 to prevent
# silent message drift in future refactors.


# ─── F2: method=dict validation (detect-and-reject mixed interp orders) ───

def test_dict_lookup_plus_two_same_interp_works(evaluator_3d):
    """F2 anti-regression: {lookup, linear, linear} must continue to work.

    Per C3: this is the legitimate supported shape — any number of 'lookup'
    dimensions plus any number of copies of the SAME interpolation method.
    The bug fix must not accidentally reject this case.
    """
    rng = np.random.RandomState(17)
    n = 500

    # g0 integer (lookup), g1 and g2 continuous (both linear)
    positions = {
        'g0': rng.randint(0, 10, n),
        'g1': rng.uniform(0, 9, n),
        'g2': rng.uniform(0, 9, n),
    }
    predictors = {'x': rng.normal(0, 1, n)}

    result = evaluator_3d.evaluate(
        positions=positions, predictors=predictors,
        method={'g0': 'lookup', 'g1': 'linear', 'g2': 'linear'},
    )
    vals = result['y']
    assert len(vals) == n
    assert np.isfinite(vals).all(), \
        "Supported shape {lookup, linear, linear} should produce finite values"


def test_dict_mixed_linear_cubic_raises(evaluator_3d):
    """F2: {'g1':'linear', 'g2':'cubic'} must raise ValueError.

    This is the pure bug case: no lookup dims, two distinct interp orders.
    Before the fix, this silently picked the first dim's order and ran.
    """
    rng = np.random.RandomState(18)
    n = 100
    positions = {
        'g0': rng.randint(0, 10, n),
        'g1': rng.uniform(0, 9, n),
        'g2': rng.uniform(0, 9, n),
    }
    predictors = {'x': rng.normal(0, 1, n)}

    with pytest.raises(ValueError, match="at most one interpolation order"):
        evaluator_3d.evaluate(
            positions=positions, predictors=predictors,
            method={'g0': 'linear', 'g1': 'linear', 'g2': 'cubic'},
        )


def test_dict_lookup_plus_mixed_interp_raises(evaluator_3d):
    """F2: lookup + mixed interp (linear+cubic) must raise ValueError.

    Mixed case with lookup present. Same bug, different branch in
    _eval_per_dimension (the mixed case at line ~1430).
    """
    rng = np.random.RandomState(19)
    n = 100
    positions = {
        'g0': rng.randint(0, 10, n),
        'g1': rng.uniform(0, 9, n),
        'g2': rng.uniform(0, 9, n),
    }
    predictors = {'x': rng.normal(0, 1, n)}

    with pytest.raises(ValueError, match="at most one interpolation order"):
        evaluator_3d.evaluate(
            positions=positions, predictors=predictors,
            method={'g0': 'lookup', 'g1': 'linear', 'g2': 'cubic'},
        )


def test_dict_lookup_plus_nearest_plus_linear_raises(evaluator_3d):
    """F2: lookup + nearest + linear must raise (order 0 vs order 1).

    Distinct interp orders: nearest is order=0, linear is order=1. The
    ValueError message must include the full method dict per C5.
    """
    rng = np.random.RandomState(20)
    n = 100
    positions = {
        'g0': rng.randint(0, 10, n),
        'g1': rng.randint(0, 10, n),
        'g2': rng.uniform(0, 9, n),
    }
    predictors = {'x': rng.normal(0, 1, n)}

    with pytest.raises(ValueError, match="at most one interpolation order") as exc_info:
        evaluator_3d.evaluate(
            positions=positions, predictors=predictors,
            method={'g0': 'lookup', 'g1': 'nearest', 'g2': 'linear'},
        )
    # C5: full dict must appear in the error message for debuggability
    msg = str(exc_info.value)
    assert 'g1' in msg and 'g2' in msg, \
        f"Error message must name the conflicting dimensions. Got: {msg}"


def test_dict_all_lookup_unchanged(evaluator_3d):
    """F2 anti-regression: all-lookup dict ≡ scalar method='lookup'.

    Bit-identical equivalence check. The all-lookup branch delegates to
    _eval_lookup directly and must not be touched by the F2 validation.
    """
    rng = np.random.RandomState(21)
    n = 1000
    positions = {
        'g0': rng.randint(0, 10, n),
        'g1': rng.randint(0, 10, n),
        'g2': rng.randint(0, 10, n),
    }
    predictors = {'x': rng.normal(0, 1, n)}

    result_dict = evaluator_3d.evaluate(
        positions=positions, predictors=predictors,
        method={'g0': 'lookup', 'g1': 'lookup', 'g2': 'lookup'},
    )
    result_scalar = evaluator_3d.evaluate(
        positions=positions, predictors=predictors,
        method='lookup',
    )
    np.testing.assert_array_equal(
        result_dict['y'], result_scalar['y'],
        err_msg="All-lookup dict must be bit-identical to method='lookup'")


# ─── C3 (from Claude21 P1 #3): nearest/nearest_fast equivalence in dict ───

def test_dict_nearest_plus_nearest_fast_works(evaluator_3d):
    """C3: 'nearest' and 'nearest_fast' are both order=0 and must mix freely.

    Per Claude21 P1 #3 and Claude20 consolidated comment C3: the two names
    are semantically equivalent (both map to scipy order=0). Rejecting this
    combination would surprise a user who is doing nothing wrong. The F2
    validation treats them as equivalent via _METHOD_ORDER.
    """
    rng = np.random.RandomState(22)
    n = 500
    positions = {
        'g0': rng.randint(0, 10, n),
        'g1': rng.uniform(0, 9, n),
        'g2': rng.uniform(0, 9, n),
    }
    predictors = {'x': rng.normal(0, 1, n)}

    # This must NOT raise — both are order=0
    result = evaluator_3d.evaluate(
        positions=positions, predictors=predictors,
        method={'g0': 'lookup', 'g1': 'nearest', 'g2': 'nearest_fast'},
    )
    vals = result['y']
    assert len(vals) == n
    assert np.isfinite(vals).all(), \
        "lookup + nearest + nearest_fast should produce finite values"


# ─── C10 (bonus, one-line addition): unknown method detection ───

def test_dict_unknown_method_raises(evaluator_3d):
    """C10: typo in dict value must raise clear error, not silently default.

    Before the fix, {'g0':'lookup','g1':'linaer'} silently fell through to
    the default 'linear' case. The F2 validation now catches unknown
    method strings at dispatch time.
    """
    rng = np.random.RandomState(23)
    n = 100
    positions = {
        'g0': rng.randint(0, 10, n),
        'g1': rng.uniform(0, 9, n),
        'g2': rng.uniform(0, 9, n),
    }
    predictors = {'x': rng.normal(0, 1, n)}

    with pytest.raises(ValueError, match="unknown method"):
        evaluator_3d.evaluate(
            positions=positions, predictors=predictors,
            method={'g0': 'lookup', 'g1': 'linaer', 'g2': 'linear'},  # typo
        )


# ─── F3: evaluate() docstring coverage ───

def test_evaluate_docstring_mentions_lookup():
    """F3: evaluate.__doc__ must document 'lookup' method."""
    doc = GroupByRegressionEvaluator.evaluate.__doc__
    assert doc is not None, "evaluate() must have a docstring"
    assert "'lookup'" in doc or '"lookup"' in doc or "``'lookup'``" in doc, \
        "docstring must document method='lookup' (added in Phase 13.16.GB)"


def test_evaluate_docstring_mentions_dict():
    """F3: evaluate.__doc__ must document the per-dimension dict shape."""
    doc = GroupByRegressionEvaluator.evaluate.__doc__
    assert doc is not None
    # Must mention dict and per-dimension (case-insensitive)
    doc_lower = doc.lower()
    assert 'dict' in doc_lower, "docstring must mention dict method shape"
    assert 'per-dimension' in doc_lower or 'per dimension' in doc_lower, \
        "docstring must mention per-dimension dispatch"


# ─── C8 (from Claude21 P1 #2): runnable docstring examples ───

def test_evaluate_docstring_examples_run(evaluator_3d):
    """C8: the method='lookup' and method=dict examples in the docstring
    must produce valid output when run against a real evaluator.

    Catches future regressions where someone edits the docstring example
    but not the behavior (or vice versa). The examples in the docstring
    are marked +SKIP for doctest because they need a live evaluator, so
    we reproduce them here against the real evaluator_3d fixture.
    """
    rng = np.random.RandomState(24)
    n = 100

    # Example 1: method='lookup' on integer grid (paraphrased with fixture names)
    track_g0 = rng.randint(0, 10, n)
    track_g1 = rng.randint(0, 10, n)
    track_g2 = rng.randint(0, 10, n)
    track_x = rng.normal(0, 1, n)

    result1 = evaluator_3d.evaluate(
        positions={'g0': track_g0, 'g1': track_g1, 'g2': track_g2},
        predictors={'x': track_x},
        method='lookup',
    )
    assert 'y' in result1
    assert np.isfinite(result1['y']).all(), \
        "Docstring example 1 (method='lookup') must produce finite output"

    # Example 2: method=dict per-dimension dispatch
    g0s = rng.randint(0, 10, n)
    g1s = rng.randint(0, 10, n)
    g2s = rng.uniform(0, 9, n)
    xs = rng.normal(0, 1, n)

    result2 = evaluator_3d.evaluate(
        positions={'g0': g0s, 'g1': g1s, 'g2': g2s},
        predictors={'x': xs},
        method={'g0': 'lookup', 'g1': 'lookup', 'g2': 'linear'},
    )
    assert 'y' in result2
    assert np.isfinite(result2['y']).all(), \
        "Docstring example 2 (method=dict) must produce finite output"


# ─── F4: bounds='extrapolate' with method='lookup' ───

def test_lookup_with_extrapolate_bounds_raises(evaluator_3d):
    """F4: method='lookup' + bounds='extrapolate' must raise ValueError.

    Before the fix, this fell through _eval_lookup's if/elif chain to
    `idx = raw` with no bounds handling, then raised an opaque IndexError
    from numpy fancy indexing when any raw value was out of range.

    Direct integer indexing has no interpolation to extrapolate from —
    the combination is nonsensical and must be rejected at dispatch time
    with a clear message.
    """
    positions = {
        'g0': np.array([0, 5, 9]),  # all in-range
        'g1': np.array([0, 5, 9]),
        'g2': np.array([0, 5, 9]),
    }
    predictors = {'x': np.array([1.0, 1.0, 1.0])}

    # Must raise even for in-range positions — the combination itself is invalid
    with pytest.raises(ValueError, match="extrapolate"):
        evaluator_3d.evaluate(
            positions=positions, predictors=predictors,
            method='lookup', bounds='extrapolate',
        )
