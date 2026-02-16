"""
Invariance Tests for GroupByRegressionEvaluator — Phase 13.9.GB

Analytical invariance checks that verify mathematical correctness,
not just "doesn't crash". Each test has an expected analytical result.

Test categories:
  I1. Nearest evaluation invariances (4 tests)
  I2. Multilinear interpolation invariances (8 tests)
  I3. Inverse-variance weighting invariances (4 tests)
  I4. Boundary handling invariances (5 tests)
  I5. Sparse grid invariances (4 tests)
  I6. Multi-target consistency invariances (3 tests)
  I7. Export/import numerical invariances (3 tests)

Total: 31 analytical invariance tests
"""

import itertools
import numpy as np
import pandas as pd
import pytest

from ..groupby_regression_evaluator import GroupByRegressionEvaluator


# ================================================================== #
#  Helper: build evaluator from known coefficient functions
# ================================================================== #

def _evaluator_from_function(
    func_intercept, func_slope, dims, n_bins_per_dim,
    predictor='pred', target='val',
):
    """
    Build an evaluator where coefficients are a known function of coordinates.

    Parameters
    ----------
    func_intercept : callable(coords_dict) -> float
        Analytical intercept as function of bin coordinates.
    func_slope : callable(coords_dict) -> float
        Analytical slope as function of bin coordinates.
    dims : list of str
        Dimension names.
    n_bins_per_dim : list of int
        Number of bins per dimension.
    """
    grid_shape = tuple(n_bins_per_dim)
    bin_centers = {d: np.arange(n, dtype=np.float64) for d, n in zip(dims, n_bins_per_dim)}

    intercept = np.empty(grid_shape, dtype=np.float64)
    slope = np.empty(grid_shape, dtype=np.float64)

    for idx in itertools.product(*(range(n) for n in n_bins_per_dim)):
        coords = {d: float(idx[i]) for i, d in enumerate(dims)}
        intercept[idx] = func_intercept(coords)
        slope[idx] = func_slope(coords)

    coefficients = {
        target: {
            'intercept': intercept,
            f'slope_{predictor}': slope,
        }
    }

    return GroupByRegressionEvaluator(
        grid_shape=grid_shape,
        group_columns=dims,
        predictor_columns=[predictor],
        targets=[target],
        bin_centers=bin_centers,
        coefficients=coefficients,
    )


def _evaluator_constant(value=5.0, slope_value=2.0, dims=None, n=5):
    """Evaluator with spatially constant coefficients."""
    if dims is None:
        dims = ['x', 'y']
    n_bins = [n] * len(dims)
    return _evaluator_from_function(
        lambda c: value, lambda c: slope_value, dims, n_bins)


def _evaluator_linear_2d(nx=10, ny=10):
    """
    2D evaluator: intercept = 1 + 2*x + 3*y, slope = 0.5 + 0.1*x + 0.2*y.
    Multilinear interpolation is exact for this field.
    """
    return _evaluator_from_function(
        func_intercept=lambda c: 1.0 + 2.0 * c['x'] + 3.0 * c['y'],
        func_slope=lambda c: 0.5 + 0.1 * c['x'] + 0.2 * c['y'],
        dims=['x', 'y'], n_bins_per_dim=[nx, ny])


def _evaluator_linear_3d(nx=6, ny=6, nz=6):
    """
    3D evaluator: intercept = 1 + 0.5*x + 0.3*y + 0.2*z.
    slope = 2 + 0.1*x + 0.1*y + 0.1*z.
    """
    return _evaluator_from_function(
        func_intercept=lambda c: 1.0 + 0.5 * c['x'] + 0.3 * c['y'] + 0.2 * c['z'],
        func_slope=lambda c: 2.0 + 0.1 * c['x'] + 0.1 * c['y'] + 0.1 * c['z'],
        dims=['x', 'y', 'z'], n_bins_per_dim=[nx, ny, nz])


# ================================================================== #
#  I1. Nearest evaluation invariances
# ================================================================== #

class TestNearestInvariance:
    """Analytical checks for nearest-neighbor evaluation."""

    def test_nearest_at_bin_center_exact(self):
        """I1.1: evaluate(nearest) at bin center = exact coefficient values."""
        ev = _evaluator_linear_2d()
        for ix in range(10):
            for iy in range(10):
                result = ev.evaluate(
                    {'x': float(ix), 'y': float(iy)}, {'pred': 0.0},
                    method='nearest')
                expected = 1.0 + 2.0 * ix + 3.0 * iy
                assert abs(result['val'] - expected) < 1e-12, \
                    f"Nearest at ({ix},{iy}): got {result['val']}, expected {expected}"

    def test_nearest_prediction_formula(self):
        """I1.2: y = intercept + slope * predictor at all bins."""
        ev = _evaluator_linear_2d()
        rng = np.random.RandomState(42)
        for _ in range(50):
            ix, iy = rng.randint(0, 10, size=2)
            pred_val = rng.uniform(-5, 5)
            result = ev.evaluate(
                {'x': float(ix), 'y': float(iy)}, {'pred': pred_val},
                method='nearest')
            intercept = 1.0 + 2.0 * ix + 3.0 * iy
            slope = 0.5 + 0.1 * ix + 0.2 * iy
            expected = intercept + slope * pred_val
            assert abs(result['val'] - expected) < 1e-12

    def test_nearest_snaps_to_closer_bin(self):
        """I1.3: At x=1.3, nearest snaps to bin 1; at x=1.7, to bin 2."""
        ev = _evaluator_linear_2d()
        r1 = ev.evaluate({'x': 1.3, 'y': 0.0}, {'pred': 0.0}, method='nearest')
        r_bin1 = ev.evaluate({'x': 1.0, 'y': 0.0}, {'pred': 0.0}, method='nearest')
        assert abs(r1['val'] - r_bin1['val']) < 1e-12

        r2 = ev.evaluate({'x': 1.7, 'y': 0.0}, {'pred': 0.0}, method='nearest')
        r_bin2 = ev.evaluate({'x': 2.0, 'y': 0.0}, {'pred': 0.0}, method='nearest')
        assert abs(r2['val'] - r_bin2['val']) < 1e-12

    def test_nearest_constant_field_everywhere_equal(self):
        """I1.4: Constant coefficient field → same value everywhere."""
        ev = _evaluator_constant(value=7.0, slope_value=3.0)
        rng = np.random.RandomState(42)
        for _ in range(20):
            pos = {'x': rng.uniform(0, 4), 'y': rng.uniform(0, 4)}
            result = ev.evaluate(pos, {'pred': 2.0}, method='nearest')
            assert abs(result['val'] - (7.0 + 3.0 * 2.0)) < 1e-12


# ================================================================== #
#  I2. Multilinear interpolation invariances
# ================================================================== #

class TestMultilinearInvariance:
    """Analytical checks for multilinear interpolation."""

    def test_multilinear_exact_at_grid_points(self):
        """I2.1: Interpolation at grid points = nearest evaluation."""
        ev = _evaluator_linear_2d()
        for ix in range(10):
            for iy in range(10):
                nearest = ev.evaluate(
                    {'x': float(ix), 'y': float(iy)}, {'pred': 1.0},
                    method='nearest')
                interp = ev.evaluate(
                    {'x': float(ix), 'y': float(iy)}, {'pred': 1.0},
                    method='multilinear')
                assert abs(nearest['val'] - interp['val']) < 1e-12

    def test_multilinear_recovers_linear_field_2d(self):
        """I2.2: Linear coefficient field → multilinear is exact (2D)."""
        ev = _evaluator_linear_2d()
        rng = np.random.RandomState(123)
        for _ in range(100):
            x = rng.uniform(0, 9)
            y = rng.uniform(0, 9)
            pred_val = rng.uniform(-5, 5)
            result = ev.evaluate(
                {'x': x, 'y': y}, {'pred': pred_val},
                method='multilinear')
            intercept = 1.0 + 2.0 * x + 3.0 * y
            slope = 0.5 + 0.1 * x + 0.2 * y
            expected = intercept + slope * pred_val
            assert abs(result['val'] - expected) < 1e-10, \
                f"At ({x:.3f},{y:.3f}): got {result['val']:.6f}, expected {expected:.6f}"

    def test_multilinear_recovers_linear_field_3d(self):
        """I2.3: Linear coefficient field → multilinear is exact (3D)."""
        ev = _evaluator_linear_3d()
        rng = np.random.RandomState(456)
        for _ in range(100):
            x = rng.uniform(0, 5)
            y = rng.uniform(0, 5)
            z = rng.uniform(0, 5)
            pred_val = rng.uniform(-5, 5)
            result = ev.evaluate(
                {'x': x, 'y': y, 'z': z}, {'pred': pred_val},
                method='multilinear')
            intercept = 1.0 + 0.5 * x + 0.3 * y + 0.2 * z
            slope = 2.0 + 0.1 * x + 0.1 * y + 0.1 * z
            expected = intercept + slope * pred_val
            assert abs(result['val'] - expected) < 1e-10

    def test_multilinear_midpoint_is_average(self):
        """I2.4: Midpoint of two bins = arithmetic average of their values."""
        # 1D test (y fixed at 0)
        ev = _evaluator_from_function(
            func_intercept=lambda c: 10.0 * c['x'],
            func_slope=lambda c: 0.0,
            dims=['x', 'y'], n_bins_per_dim=[5, 1])
        for ix in range(4):
            mid = ev.evaluate(
                {'x': ix + 0.5, 'y': 0.0}, {'pred': 0.0},
                method='multilinear')
            expected = (10.0 * ix + 10.0 * (ix + 1)) / 2.0
            assert abs(mid['val'] - expected) < 1e-12

    def test_multilinear_constant_field(self):
        """I2.5: Constant field → interpolation = constant everywhere."""
        ev = _evaluator_constant(value=42.0, slope_value=0.0)
        rng = np.random.RandomState(789)
        for _ in range(50):
            pos = {'x': rng.uniform(0, 4), 'y': rng.uniform(0, 4)}
            result = ev.evaluate(pos, {'pred': 0.0}, method='multilinear')
            assert abs(result['val'] - 42.0) < 1e-12

    def test_multilinear_symmetry(self):
        """I2.6: Symmetric field → symmetric evaluation."""
        # Symmetric: intercept = x + y
        ev = _evaluator_from_function(
            func_intercept=lambda c: c['x'] + c['y'],
            func_slope=lambda c: 0.0,
            dims=['x', 'y'], n_bins_per_dim=[5, 5])
        # f(1.3, 2.7) should equal f(2.7, 1.3) by symmetry
        r1 = ev.evaluate({'x': 1.3, 'y': 2.7}, {'pred': 0.0}, method='multilinear')
        r2 = ev.evaluate({'x': 2.7, 'y': 1.3}, {'pred': 0.0}, method='multilinear')
        assert abs(r1['val'] - r2['val']) < 1e-12

    def test_multilinear_continuity_lipschitz(self):
        """I2.7: |f(x+eps) - f(x)| <= L * eps for bounded L."""
        ev = _evaluator_linear_2d()
        # Max gradient: d(intercept)/dx = 2.0, d(slope)/dx = 0.1
        # With pred=1: max derivative ~2.1
        base = ev.evaluate({'x': 3.0, 'y': 4.0}, {'pred': 1.0},
                           method='multilinear')
        eps = 1e-6
        perturbed = ev.evaluate({'x': 3.0 + eps, 'y': 4.0}, {'pred': 1.0},
                                method='multilinear')
        diff = abs(perturbed['val'] - base['val'])
        # L * eps where L ~ 2.1 for this field
        assert diff < 5.0 * eps  # generous bound

    def test_multilinear_bilinear_weight_formula(self):
        """
        I2.8: Verify bilinear formula explicitly.

        At (x=0.3, y=0.7) between bins (0,0), (1,0), (0,1), (1,1):
        f = (1-fx)(1-fy)*f00 + fx*(1-fy)*f10 + (1-fx)*fy*f01 + fx*fy*f11
        """
        ev = _evaluator_from_function(
            func_intercept=lambda c: c['x'] ** 2 + c['y'] ** 2,  # non-linear!
            func_slope=lambda c: 0.0,
            dims=['x', 'y'], n_bins_per_dim=[5, 5])

        fx, fy = 0.3, 0.7
        f00 = 0.0 ** 2 + 0.0 ** 2  # 0
        f10 = 1.0 ** 2 + 0.0 ** 2  # 1
        f01 = 0.0 ** 2 + 1.0 ** 2  # 1
        f11 = 1.0 ** 2 + 1.0 ** 2  # 2

        expected = ((1 - fx) * (1 - fy) * f00 + fx * (1 - fy) * f10
                    + (1 - fx) * fy * f01 + fx * fy * f11)

        result = ev.evaluate({'x': 0.3, 'y': 0.7}, {'pred': 0.0},
                             method='multilinear')
        assert abs(result['val'] - expected) < 1e-12, \
            f"Bilinear: got {result['val']}, expected {expected}"


# ================================================================== #
#  I3. Inverse-variance weighting invariances
# ================================================================== #

class TestIVarInvariance:
    """Analytical checks for inverse-variance weighted interpolation."""

    def test_ivar_uniform_errors_equals_unweighted(self):
        """I3.1: Uniform errors → ivar result = unweighted result."""
        grid_shape = (5, 5)
        dims = ['x', 'y']
        rng = np.random.RandomState(42)
        intercept = rng.randn(*grid_shape)
        slope = rng.randn(*grid_shape)
        err = np.full(grid_shape, 0.1)  # uniform errors

        ev = GroupByRegressionEvaluator(
            grid_shape=grid_shape, group_columns=dims,
            predictor_columns=['pred'], targets=['val'],
            bin_centers={d: np.arange(5, dtype=float) for d in dims},
            coefficients={'val': {
                'intercept': intercept, 'slope_pred': slope,
                'intercept_err': err, 'slope_pred_err': err,
            }},
        )

        pos = {'x': 1.3, 'y': 2.7}
        pred = {'pred': 1.0}
        unweighted = ev.evaluate(pos, pred, method='multilinear',
                                 use_errors=False)
        weighted = ev.evaluate(pos, pred, method='multilinear',
                               use_errors=True)
        assert abs(unweighted['val'] - weighted['val']) < 1e-12

    def test_ivar_extreme_error_ratio(self):
        """I3.2: One corner has err=0.01, other err=100 → result ≈ precise corner."""
        ev = GroupByRegressionEvaluator(
            grid_shape=(2, 1), group_columns=['x', 'y'],
            predictor_columns=['pred'], targets=['val'],
            bin_centers={'x': np.array([0.0, 1.0]),
                         'y': np.array([0.0])},
            coefficients={'val': {
                'intercept': np.array([[10.0], [20.0]]),
                'slope_pred': np.array([[0.0], [0.0]]),
                'intercept_err': np.array([[0.01], [100.0]]),
                'slope_pred_err': np.array([[0.01], [100.0]]),
            }},
        )
        # Midpoint: unweighted = 15.0, weighted ≈ 10.0 (precise corner)
        weighted = ev.evaluate({'x': 0.5, 'y': 0.0}, {'pred': 0.0},
                               method='multilinear', use_errors=True)
        # With err ratio 10000:1, weight ratio ~ 1e8:1
        assert abs(weighted['val'] - 10.0) < 0.01, \
            f"Expected ~10.0, got {weighted['val']}"

    def test_ivar_zero_error_gets_all_weight(self):
        """I3.3: Corner with err→0 dominates (limit case)."""
        ev = GroupByRegressionEvaluator(
            grid_shape=(2, 1), group_columns=['x', 'y'],
            predictor_columns=['pred'], targets=['val'],
            bin_centers={'x': np.array([0.0, 1.0]),
                         'y': np.array([0.0])},
            coefficients={'val': {
                'intercept': np.array([[10.0], [20.0]]),
                'slope_pred': np.array([[0.0], [0.0]]),
                'intercept_err': np.array([[1e-10], [1.0]]),
                'slope_pred_err': np.array([[1e-10], [1.0]]),
            }},
        )
        weighted = ev.evaluate({'x': 0.5, 'y': 0.0}, {'pred': 0.0},
                               method='multilinear', use_errors=True)
        assert abs(weighted['val'] - 10.0) < 1e-6

    def test_ivar_proportional_weighting(self):
        """I3.4: At midpoint with err1=1, err2=2: w1/w2 = 4 (inverse variance)."""
        ev = GroupByRegressionEvaluator(
            grid_shape=(2, 1), group_columns=['x', 'y'],
            predictor_columns=['pred'], targets=['val'],
            bin_centers={'x': np.array([0.0, 1.0]),
                         'y': np.array([0.0])},
            coefficients={'val': {
                'intercept': np.array([[0.0], [100.0]]),
                'slope_pred': np.array([[0.0], [0.0]]),
                'intercept_err': np.array([[1.0], [2.0]]),
                'slope_pred_err': np.array([[1.0], [2.0]]),
            }},
        )
        # At midpoint: geometric weights = 0.5 each
        # ivar: w1 = 0.5 * (1/1) = 0.5, w2 = 0.5 * (1/4) = 0.125
        # result = (0.5*0 + 0.125*100) / (0.5 + 0.125) = 12.5/0.625 = 20.0
        result = ev.evaluate({'x': 0.5, 'y': 0.0}, {'pred': 0.0},
                             method='multilinear', use_errors=True)
        assert abs(result['val'] - 20.0) < 1e-10, \
            f"Expected 20.0, got {result['val']}"

    def test_use_errors_false_ignores_error_columns(self):
        """I3.5: use_errors=False must NOT apply inverse-variance weighting.

        Regression test for operator precedence bug: with non-uniform errors,
        use_errors=False must produce pure geometric multilinear interpolation
        identical to a grid without any error columns.
        """
        grid_shape = (3, 3)
        dims = ['x', 'y']
        rng = np.random.RandomState(99)
        intercept = rng.randn(*grid_shape)
        slope = rng.randn(*grid_shape)
        # Highly non-uniform errors — if ivar is applied, result differs
        err_intercept = np.array([[0.01, 1.0, 100.0],
                                   [0.01, 1.0, 100.0],
                                   [0.01, 1.0, 100.0]])
        err_slope = np.array([[100.0, 1.0, 0.01],
                               [100.0, 1.0, 0.01],
                               [100.0, 1.0, 0.01]])

        # Evaluator WITH error columns
        ev_with_err = GroupByRegressionEvaluator(
            grid_shape=grid_shape, group_columns=dims,
            predictor_columns=['pred'], targets=['val'],
            bin_centers={d: np.arange(3, dtype=float) for d in dims},
            coefficients={'val': {
                'intercept': intercept, 'slope_pred': slope,
                'intercept_err': err_intercept,
                'slope_pred_err': err_slope,
            }},
        )
        # Evaluator WITHOUT error columns (pure geometric baseline)
        ev_no_err = GroupByRegressionEvaluator(
            grid_shape=grid_shape, group_columns=dims,
            predictor_columns=['pred'], targets=['val'],
            bin_centers={d: np.arange(3, dtype=float) for d in dims},
            coefficients={'val': {
                'intercept': intercept, 'slope_pred': slope,
            }},
        )

        pos = {'x': 0.7, 'y': 1.3}
        pred = {'pred': 2.5}
        result_with = ev_with_err.evaluate(pos, pred, method='multilinear',
                                            use_errors=False)
        result_without = ev_no_err.evaluate(pos, pred, method='multilinear',
                                             use_errors=False)
        assert abs(result_with['val'] - result_without['val']) < 1e-12, \
            (f"use_errors=False should ignore error columns. "
             f"With errors: {result_with['val']}, "
             f"Without errors: {result_without['val']}")


# ================================================================== #
#  I4. Boundary handling invariances
# ================================================================== #

class TestBoundaryInvariance:
    """Analytical checks for boundary/extrapolation behavior."""

    def test_clamp_at_edge_equals_edge_bin(self):
        """I4.1: With clamp, evaluate far outside = edge bin value."""
        ev = _evaluator_linear_2d()
        # Far left: should clamp to x=0
        far_left = ev.evaluate({'x': -100.0, 'y': 0.0}, {'pred': 0.0},
                               method='multilinear', bounds='clamp')
        at_zero = ev.evaluate({'x': 0.0, 'y': 0.0}, {'pred': 0.0},
                              method='nearest')
        assert abs(far_left['val'] - at_zero['val']) < 1e-12

    def test_nan_outside_grid(self):
        """I4.2: With bounds='nan', outside grid → NaN."""
        ev = _evaluator_linear_2d()
        result = ev.evaluate({'x': -0.5, 'y': 5.0}, {'pred': 0.0},
                             method='multilinear', bounds='nan')
        assert np.isnan(result['val'])

    def test_nan_inside_grid_not_nan(self):
        """I4.3: With bounds='nan', inside grid → not NaN (fully populated)."""
        ev = _evaluator_linear_2d()
        result = ev.evaluate({'x': 3.5, 'y': 4.5}, {'pred': 0.0},
                             method='multilinear', bounds='nan')
        assert not np.isnan(result['val'])

    def test_clamp_vs_edge_bin_all_corners(self):
        """I4.4: Clamping to all four corners of 2D grid."""
        ev = _evaluator_linear_2d(nx=5, ny=5)
        corners = [
            ((-10, -10), (0, 0)),     # bottom-left
            ((100, -10), (4, 0)),     # bottom-right
            ((-10, 100), (0, 4)),     # top-left
            ((100, 100), (4, 4)),     # top-right
        ]
        for (ox, oy), (ex, ey) in corners:
            clamped = ev.evaluate(
                {'x': float(ox), 'y': float(oy)}, {'pred': 0.0},
                method='multilinear', bounds='clamp')
            expected = ev.evaluate(
                {'x': float(ex), 'y': float(ey)}, {'pred': 0.0},
                method='nearest')
            assert abs(clamped['val'] - expected['val']) < 1e-12, \
                f"Corner ({ox},{oy}): got {clamped['val']}, expected {expected['val']}"

    def test_inside_grid_clamp_and_nan_agree(self):
        """I4.5: For interior points, clamp and nan modes give same result."""
        ev = _evaluator_linear_2d()
        rng = np.random.RandomState(42)
        for _ in range(50):
            x = rng.uniform(0, 9)
            y = rng.uniform(0, 9)
            pos = {'x': x, 'y': y}
            pred = {'pred': rng.uniform(-5, 5)}
            r_clamp = ev.evaluate(pos, pred, method='multilinear', bounds='clamp')
            r_nan = ev.evaluate(pos, pred, method='multilinear', bounds='nan')
            assert abs(r_clamp['val'] - r_nan['val']) < 1e-12


# ================================================================== #
#  I5. Sparse grid invariances
# ================================================================== #

class TestSparseGridInvariance:
    """Analytical checks for sparse grid (missing bins) handling."""

    def _make_sparse_evaluator(self, missing_idx=(1, 1)):
        """3x3 grid with one missing bin. intercept = uniform 10.0."""
        grid_shape = (3, 3)
        intercept = np.full(grid_shape, 10.0)
        slope = np.full(grid_shape, 1.0)
        valid = np.ones(grid_shape, dtype=bool)
        intercept[missing_idx] = np.nan
        slope[missing_idx] = np.nan
        valid[missing_idx] = False

        return GroupByRegressionEvaluator(
            grid_shape=grid_shape, group_columns=['x', 'y'],
            predictor_columns=['pred'], targets=['val'],
            bin_centers={'x': np.arange(3, dtype=float),
                         'y': np.arange(3, dtype=float)},
            coefficients={'val': {'intercept': intercept, 'slope_pred': slope}},
            valid_mask=valid,
        )

    def test_nan_strategy_with_invalid_corner(self):
        """I5.1: invalid_strategy='nan' → NaN when interpolation touches invalid bin."""
        ev = self._make_sparse_evaluator(missing_idx=(1, 1))
        # (0.5, 0.5) uses corners (0,0),(1,0),(0,1),(1,1) — last is invalid
        result = ev.evaluate({'x': 0.5, 'y': 0.5}, {'pred': 0.0},
                             method='multilinear', invalid_strategy='nan')
        assert np.isnan(result['val'])

    def test_skip_strategy_renormalises(self):
        """I5.2: skip with uniform field → result equals the uniform value."""
        ev = self._make_sparse_evaluator(missing_idx=(1, 1))
        # All valid corners have intercept=10.0, slope=1.0
        # Skipping invalid corner and renormalising should give same value
        result = ev.evaluate({'x': 0.5, 'y': 0.5}, {'pred': 2.0},
                             method='multilinear', invalid_strategy='skip')
        expected = 10.0 + 1.0 * 2.0
        assert abs(result['val'] - expected) < 1e-12

    def test_nan_strategy_far_from_invalid_bin(self):
        """I5.3: Point whose interpolation cell has all valid corners → not NaN."""
        ev = self._make_sparse_evaluator(missing_idx=(1, 1))
        # At (1.5, 0.3): corners are (1,0),(2,0),(1,1),(2,1) — (1,1) invalid → NaN
        # At (1.5, 1.5): corners are (1,1),(2,1),(1,2),(2,2) — (1,1) invalid → NaN
        # At (0.3, 1.5): corners are (0,1),(1,1),(0,2),(1,2) — (1,1) invalid → NaN
        # Need a cell that doesn't touch (1,1):
        # (1.5, 0.0): uses corners (1,0),(2,0) — all valid (single-bin y)
        # Actually on 3x3 grid: (1.3, 0.3) uses (1,0),(2,0),(1,1),(2,1)
        # The only safe cell is top-right: (1.5, 1.5) still hits (1,1)
        # Use missing_idx=(0, 0) instead for a cleaner test
        ev2 = self._make_sparse_evaluator(missing_idx=(0, 0))
        # At (1.5, 1.5): corners are (1,1),(2,1),(1,2),(2,2) — all valid
        result = ev2.evaluate({'x': 1.5, 'y': 1.5}, {'pred': 0.0},
                              method='multilinear', invalid_strategy='nan')
        assert not np.isnan(result['val'])
        assert abs(result['val'] - 10.0) < 1e-12  # uniform field

    def test_all_corners_valid_skip_equals_nan(self):
        """I5.4: When all corners are valid, skip and nan give same result."""
        ev = self._make_sparse_evaluator(missing_idx=(2, 2))  # far corner
        pos = {'x': 0.5, 'y': 0.5}
        pred = {'pred': 1.0}
        r_nan = ev.evaluate(pos, pred, method='multilinear',
                            invalid_strategy='nan')
        r_skip = ev.evaluate(pos, pred, method='multilinear',
                             invalid_strategy='skip')
        assert abs(r_nan['val'] - r_skip['val']) < 1e-12


# ================================================================== #
#  I6. Multi-target consistency invariances
# ================================================================== #

class TestMultiTargetInvariance:
    """Analytical checks for multi-target evaluator consistency."""

    def _make_multi_target_evaluator(self):
        """3 targets with known analytical relationships."""
        grid_shape = (5, 5)
        dims = ['x', 'y']
        bc = {d: np.arange(5, dtype=float) for d in dims}

        # dX_intercept = x + y,  dY = 2*(x+y),  dZ = 3*(x+y)
        dX_int = np.fromfunction(lambda i, j: i + j, grid_shape)
        dY_int = 2.0 * dX_int
        dZ_int = 3.0 * dX_int
        slope = np.ones(grid_shape)

        return GroupByRegressionEvaluator(
            grid_shape=grid_shape, group_columns=dims,
            predictor_columns=['pred'], targets=['dX', 'dY', 'dZ'],
            bin_centers=bc,
            coefficients={
                'dX': {'intercept': dX_int, 'slope_pred': slope},
                'dY': {'intercept': dY_int, 'slope_pred': slope},
                'dZ': {'intercept': dZ_int, 'slope_pred': slope},
            },
        )

    def test_targets_independent_evaluation(self):
        """I6.1: Each target evaluated independently, results in one dict."""
        ev = self._make_multi_target_evaluator()
        result = ev.evaluate({'x': 2.5, 'y': 1.5}, {'pred': 0.0},
                             method='multilinear')
        assert set(result.keys()) == {'dX', 'dY', 'dZ'}
        # dX = 2.5 + 1.5 = 4.0
        assert abs(result['dX'] - 4.0) < 1e-12
        # dY = 2 * 4.0 = 8.0
        assert abs(result['dY'] - 8.0) < 1e-12
        # dZ = 3 * 4.0 = 12.0
        assert abs(result['dZ'] - 12.0) < 1e-12

    def test_target_ratio_preserved(self):
        """I6.2: dY/dX = 2.0 and dZ/dX = 3.0 at all interior points."""
        ev = self._make_multi_target_evaluator()
        rng = np.random.RandomState(42)
        for _ in range(50):
            x = rng.uniform(0, 4)
            y = rng.uniform(0, 4)
            result = ev.evaluate({'x': x, 'y': y}, {'pred': 0.0},
                                 method='multilinear')
            if abs(result['dX']) > 1e-10:
                assert abs(result['dY'] / result['dX'] - 2.0) < 1e-10
                assert abs(result['dZ'] / result['dX'] - 3.0) < 1e-10

    def test_single_target_subset(self):
        """I6.3: Evaluating subset of targets gives same values."""
        ev = self._make_multi_target_evaluator()
        pos = {'x': 2.5, 'y': 1.5}
        pred = {'pred': 1.0}
        all_targets = ev.evaluate(pos, pred, method='multilinear')
        just_dX = ev.evaluate(pos, pred, method='multilinear',
                              targets=['dX'])
        assert abs(all_targets['dX'] - just_dX['dX']) < 1e-12
        assert 'dY' not in just_dX


# ================================================================== #
#  I7. Export/import numerical invariances
# ================================================================== #

class TestExportInvariance:
    """Analytical checks for roundtrip numerical precision."""

    def test_dict_roundtrip_exact(self):
        """I7.1: to_dict → from_dict preserves all coefficients exactly."""
        ev = _evaluator_linear_3d()
        ev2 = GroupByRegressionEvaluator.from_dict(ev.to_dict())
        for tgt in ev.targets:
            for key in ev._coefficients[tgt]:
                np.testing.assert_array_equal(
                    ev.coefficient_grid(tgt, key),
                    ev2.coefficient_grid(tgt, key))

    def test_roundtrip_evaluation_identical(self):
        """I7.2: Evaluation after roundtrip = evaluation before."""
        ev = _evaluator_linear_3d()
        ev2 = GroupByRegressionEvaluator.from_dict(ev.to_dict())
        rng = np.random.RandomState(42)
        for _ in range(50):
            pos = {'x': rng.uniform(0, 5), 'y': rng.uniform(0, 5),
                   'z': rng.uniform(0, 5)}
            pred = {'pred': rng.uniform(-5, 5)}
            r1 = ev.evaluate(pos, pred, method='multilinear')
            r2 = ev2.evaluate(pos, pred, method='multilinear')
            assert abs(r1['val'] - r2['val']) < 1e-12

    def test_roundtrip_preserves_valid_mask(self):
        """I7.3: Valid mask survives roundtrip exactly."""
        grid_shape = (4, 4)
        valid = np.ones(grid_shape, dtype=bool)
        valid[1, 2] = False
        valid[3, 0] = False

        intercept = np.where(valid, 1.0, np.nan)
        ev = GroupByRegressionEvaluator(
            grid_shape=grid_shape, group_columns=['x', 'y'],
            predictor_columns=['pred'], targets=['val'],
            bin_centers={'x': np.arange(4, dtype=float),
                         'y': np.arange(4, dtype=float)},
            coefficients={'val': {
                'intercept': intercept,
                'slope_pred': np.where(valid, 0.5, np.nan),
            }},
            valid_mask=valid,
        )
        ev2 = GroupByRegressionEvaluator.from_dict(ev.to_dict())
        np.testing.assert_array_equal(ev._valid_mask, ev2._valid_mask)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
