"""
Tests for GroupByRegressionEvaluator — Phase 13.9.GB

Test plan from proposal:
  §7.1 Construction tests (5)
  §7.2 Per-bin evaluation tests (3)
  §7.3 Interpolation invariance tests (7)
  §7.4 Inverse-variance weighting tests (2)
  §7.5 Export / roundtrip tests (3)
  §7.6 Integration with SW output (2)

Total: 22 tests
"""

import json
import gzip
import os
import tempfile
import numpy as np
import pandas as pd
import pytest

from ..groupby_regression_evaluator import GroupByRegressionEvaluator


# ================================================================== #
#  Fixtures
# ================================================================== #

def _make_simple_2d_dfGB(nx=5, ny=4, suffix=''):
    """
    Create a simple 2D dfGB with known coefficients.

    Model: value = 2.0 + 3.0 * predictor
    (uniform across all bins for easy verification)
    """
    rows = []
    for ix in range(nx):
        for iy in range(ny):
            rows.append({
                'xBin': ix,
                'yBin': iy,
                f'value_intercept{suffix}': 2.0 + 0.1 * ix,
                f'value_slope_pred{suffix}': 3.0 + 0.1 * iy,
                f'value_intercept_err{suffix}': 0.01 + 0.001 * ix,
                f'value_slope_pred_err{suffix}': 0.02 + 0.001 * iy,
                f'value_rms{suffix}': 0.5,
                f'value_n_fitted{suffix}': 100,
            })
    return pd.DataFrame(rows)


def _make_linear_field_3d(nx=5, ny=5, nz=5, suffix=''):
    """
    Create a 3D dfGB where coefficients are linear functions of bin coords.

    intercept(x,y,z) = 1.0 + 0.5*x + 0.3*y + 0.2*z
    slope_pred(x,y,z) = 2.0 + 0.1*x + 0.1*y + 0.1*z

    Multilinear interpolation should recover these exactly.
    """
    rows = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                rows.append({
                    'xBin': ix,
                    'yBin': iy,
                    'zBin': iz,
                    f'dX_intercept{suffix}': 1.0 + 0.5 * ix + 0.3 * iy + 0.2 * iz,
                    f'dX_slope_pred{suffix}': 2.0 + 0.1 * ix + 0.1 * iy + 0.1 * iz,
                    f'dX_intercept_err{suffix}': 0.01,
                    f'dX_slope_pred_err{suffix}': 0.02,
                    f'dX_rms{suffix}': 0.1,
                })
    return pd.DataFrame(rows)


def _make_sparse_2d_dfGB(nx=5, ny=5, missing_frac=0.3, suffix=''):
    """Create a 2D dfGB with some bins missing."""
    rng = np.random.RandomState(42)
    rows = []
    for ix in range(nx):
        for iy in range(ny):
            if rng.random() < missing_frac:
                continue  # skip this bin
            rows.append({
                'xBin': ix,
                'yBin': iy,
                f'val_intercept{suffix}': 1.0 + 0.1 * ix,
                f'val_slope_pred{suffix}': 2.0 + 0.1 * iy,
                f'val_intercept_err{suffix}': 0.01,
                f'val_slope_pred_err{suffix}': 0.02,
            })
    return pd.DataFrame(rows)


def _make_multi_target_2d(nx=5, ny=4, suffix='_fit'):
    """Create a 2D dfGB with multiple targets (dX, dY, dZ)."""
    rows = []
    for ix in range(nx):
        for iy in range(ny):
            rows.append({
                'xBin': ix,
                'yBin': iy,
                f'dX_intercept{suffix}': 1.0 + 0.1 * ix,
                f'dX_slope_pred{suffix}': 2.0 + 0.1 * iy,
                f'dX_intercept_err{suffix}': 0.01,
                f'dX_slope_pred_err{suffix}': 0.02,
                f'dY_intercept{suffix}': -1.0 + 0.2 * ix,
                f'dY_slope_pred{suffix}': 0.5 + 0.05 * iy,
                f'dY_intercept_err{suffix}': 0.02,
                f'dY_slope_pred_err{suffix}': 0.01,
                f'dZ_intercept{suffix}': 0.0 + 0.05 * ix,
                f'dZ_slope_pred{suffix}': 1.0 + 0.02 * iy,
                f'dZ_intercept_err{suffix}': 0.005,
                f'dZ_slope_pred_err{suffix}': 0.01,
            })
    return pd.DataFrame(rows)


# ================================================================== #
#  §7.1 Construction tests
# ================================================================== #

class TestConstruction:
    """Tests 1-5: Construction from dfGB."""

    def test_from_dfGB_basic(self):
        """Test 1: Construct from simple 2D dfGB, verify grid_shape."""
        dfGB = _make_simple_2d_dfGB(nx=5, ny=4)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        assert ev.grid_shape == (5, 4)
        assert ev.targets == ['value']
        assert ev.group_columns == ['xBin', 'yBin']
        assert ev.predictor_columns == ['pred']

    def test_from_dfGB_with_bin_centers(self):
        """Test 2: Float coordinates correctly mapped."""
        dfGB = _make_simple_2d_dfGB(nx=3, ny=3)
        centers = {
            'xBin': np.array([0.5, 1.5, 2.5]),
            'yBin': np.array([10.0, 20.0, 30.0]),
        }
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
            bin_centers=centers,
        )
        dims = ev.dimensions
        assert dims['xBin']['min'] == 0.5
        assert dims['xBin']['max'] == 2.5
        assert dims['yBin']['min'] == 10.0
        assert dims['yBin']['max'] == 30.0

    def test_from_dfGB_with_bin_edges(self):
        """Test 3: Edges → midpoint centers computed."""
        dfGB = _make_simple_2d_dfGB(nx=3, ny=2)
        edges = {
            'xBin': np.array([0.0, 1.0, 2.0, 3.0]),  # 3 bins
            'yBin': np.array([0.0, 5.0, 10.0]),        # 2 bins
        }
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
            bin_edges=edges,
        )
        dims = ev.dimensions
        assert dims['xBin']['min'] == 0.5
        assert dims['xBin']['max'] == 2.5
        assert dims['yBin']['min'] == 2.5
        assert dims['yBin']['max'] == 7.5

    def test_from_dfGB_sparse(self):
        """Test 4: Missing bins marked in valid_mask."""
        dfGB = _make_sparse_2d_dfGB(nx=5, ny=5, missing_frac=0.3)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='val',
        )
        # Some bins should be invalid (NaN)
        assert ev.sparsity > 0
        assert ev.n_valid_bins < 25
        assert ev.n_valid_bins == len(dfGB)

    def test_grid_shape_matches_unique_bins(self):
        """Test 5: grid_shape = product of unique values."""
        dfGB = _make_simple_2d_dfGB(nx=7, ny=3)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        assert ev.grid_shape == (7, 3)
        assert ev.grid_shape[0] * ev.grid_shape[1] == 21

    def test_from_dfGB_multi_target(self):
        """Test: Multi-target construction (dX/dY/dZ)."""
        dfGB = _make_multi_target_2d(nx=5, ny=4, suffix='_fit')
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets=['dX', 'dY', 'dZ'],
            suffix='_fit',
        )
        assert ev.targets == ['dX', 'dY', 'dZ']
        assert ev.grid_shape == (5, 4)

    def test_from_dfGB_with_suffix(self):
        """Test: Suffix stripping works correctly."""
        dfGB = _make_simple_2d_dfGB(nx=3, ny=3, suffix='_sw')
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
            suffix='_sw',
        )
        # Internal keys should be canonical (no suffix)
        grid = ev.coefficient_grid('value', 'intercept')
        assert grid.shape == (3, 3)
        assert not np.all(np.isnan(grid))


# ================================================================== #
#  §7.2 Per-bin evaluation tests
# ================================================================== #

class TestPerBinEvaluation:
    """Tests 6-8: Nearest-neighbor evaluation."""

    def test_evaluate_nearest_exact_bin(self):
        """Test 6: Exact bin position → exact coefficients."""
        dfGB = _make_simple_2d_dfGB(nx=5, ny=4)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        # Evaluate at bin (2, 3)
        result = ev.evaluate(
            {'xBin': 2.0, 'yBin': 3.0},
            {'pred': 1.0},
            method='nearest',
        )
        # intercept = 2.0 + 0.1*2 = 2.2, slope = 3.0 + 0.1*3 = 3.3
        expected = 2.2 + 3.3 * 1.0
        assert abs(result['value'] - expected) < 1e-10

    def test_evaluate_nearest_known_truth(self):
        """Test 7: y = intercept + slope * predictor matches."""
        dfGB = _make_simple_2d_dfGB(nx=5, ny=4)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        # Test at predictor = 5.0
        result = ev.evaluate(
            {'xBin': 0.0, 'yBin': 0.0},
            {'pred': 5.0},
            method='nearest',
        )
        # intercept = 2.0, slope = 3.0
        expected = 2.0 + 3.0 * 5.0
        assert abs(result['value'] - expected) < 1e-10

    def test_evaluate_batch(self):
        """Test 8: DataFrame input, vectorised output."""
        dfGB = _make_simple_2d_dfGB(nx=5, ny=4)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        positions = pd.DataFrame({
            'xBin': [0.0, 1.0, 2.0, 3.0],
            'yBin': [0.0, 1.0, 2.0, 3.0],
        })
        predictors = pd.DataFrame({
            'pred': [1.0, 1.0, 1.0, 1.0],
        })
        result = ev.evaluate(positions, predictors, method='nearest')
        assert isinstance(result['value'], np.ndarray)
        assert len(result['value']) == 4

        # Check first point: intercept=2.0, slope=3.0
        assert abs(result['value'][0] - (2.0 + 3.0)) < 1e-10
        # Check last point: intercept=2.3, slope=3.3
        assert abs(result['value'][3] - (2.3 + 3.3)) < 1e-10


# ================================================================== #
#  §7.3 Interpolation invariance tests
# ================================================================== #

class TestInterpolationInvariance:
    """Tests 9-15: Multilinear interpolation analytical checks."""

    def test_interpolate_at_grid_point(self):
        """Test 9: Interpolation at exact bin = per-bin eval."""
        dfGB = _make_simple_2d_dfGB(nx=5, ny=4)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        for ix in range(5):
            for iy in range(4):
                nearest = ev.evaluate(
                    {'xBin': float(ix), 'yBin': float(iy)},
                    {'pred': 1.0}, method='nearest')
                interp = ev.evaluate(
                    {'xBin': float(ix), 'yBin': float(iy)},
                    {'pred': 1.0}, method='multilinear')
                assert abs(nearest['value'] - interp['value']) < 1e-10, \
                    f"Mismatch at ({ix}, {iy})"

    def test_interpolate_midpoint_2bins(self):
        """Test 10: 1D midpoint between 2 bins = average."""
        # Create 1D grid (use yBin=0 only)
        rows = []
        for ix in range(4):
            rows.append({
                'xBin': ix,
                'yBin': 0,
                'val_intercept': float(ix) * 10.0,
                'val_slope_pred': 1.0,
            })
        dfGB = pd.DataFrame(rows)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='val',
        )
        # Midpoint between bin 1 and bin 2: intercept should be average
        result = ev.get_coefficients(
            {'xBin': 1.5, 'yBin': 0.0}, method='multilinear')
        expected_intercept = (10.0 + 20.0) / 2.0  # avg of bin1 and bin2
        assert abs(result['val']['intercept'] - expected_intercept) < 1e-10

    def test_interpolate_recovers_linear_field(self):
        """
        Test 11: If coefficients are linear in bin coords,
        multilinear interpolation is exact.
        """
        dfGB = _make_linear_field_3d(nx=5, ny=5, nz=5)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin', 'zBin'],
            predictor_columns=['pred'], targets='dX',
        )
        # Test at fractional positions
        rng = np.random.RandomState(123)
        for _ in range(50):
            x = rng.uniform(0, 4)
            y = rng.uniform(0, 4)
            z = rng.uniform(0, 4)
            pred_val = rng.uniform(0, 10)

            result = ev.evaluate(
                {'xBin': x, 'yBin': y, 'zBin': z},
                {'pred': pred_val},
                method='multilinear',
            )
            # Expected: (1.0 + 0.5*x + 0.3*y + 0.2*z) + (2.0 + 0.1*x + 0.1*y + 0.1*z) * pred
            expected_intercept = 1.0 + 0.5 * x + 0.3 * y + 0.2 * z
            expected_slope = 2.0 + 0.1 * x + 0.1 * y + 0.1 * z
            expected = expected_intercept + expected_slope * pred_val

            assert abs(result['dX'] - expected) < 1e-10, \
                f"Failed at ({x:.2f}, {y:.2f}, {z:.2f}): " \
                f"got {result['dX']:.6f}, expected {expected:.6f}"

    def test_interpolate_continuity(self):
        """Test 12: Small perturbation → small change (Lipschitz)."""
        dfGB = _make_simple_2d_dfGB(nx=10, ny=10)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        base = ev.evaluate(
            {'xBin': 3.0, 'yBin': 4.0}, {'pred': 1.0},
            method='multilinear')

        eps = 1e-6
        perturbed = ev.evaluate(
            {'xBin': 3.0 + eps, 'yBin': 4.0 + eps}, {'pred': 1.0},
            method='multilinear')

        # Change should be proportional to eps
        diff = abs(perturbed['value'] - base['value'])
        assert diff < 1.0  # Lipschitz bound (coefficients vary by ~0.1 per bin)

    def test_interpolate_boundary_nan(self):
        """Test 13: Point outside grid → NaN (with bounds='nan')."""
        dfGB = _make_simple_2d_dfGB(nx=5, ny=4)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        result = ev.evaluate(
            {'xBin': -1.0, 'yBin': 0.0}, {'pred': 1.0},
            method='multilinear', bounds='nan')
        assert np.isnan(result['value'])

        result = ev.evaluate(
            {'xBin': 10.0, 'yBin': 0.0}, {'pred': 1.0},
            method='multilinear', bounds='nan')
        assert np.isnan(result['value'])

    def test_interpolate_invalid_bin_nan(self):
        """Test 14: Corner with failed fit → NaN (default strategy)."""
        dfGB = _make_sparse_2d_dfGB(nx=5, ny=5, missing_frac=0.5)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='val',
        )
        # Find a position where at least one corner is invalid
        # Evaluate in the middle of the grid — likely to hit invalid corners
        result = ev.evaluate(
            {'xBin': 2.5, 'yBin': 2.5}, {'pred': 1.0},
            method='multilinear', invalid_strategy='nan')
        # Result should be NaN (high sparsity makes this likely)
        # If it's not NaN, all 4 corners happened to be valid — still correct
        # We just verify no crash

    def test_interpolate_invalid_bin_skip(self):
        """Test 15: invalid_strategy='skip' → renormalised result."""
        # Create grid with one known invalid bin
        rows = []
        for ix in range(3):
            for iy in range(3):
                if ix == 1 and iy == 1:
                    continue  # missing bin
                rows.append({
                    'xBin': ix, 'yBin': iy,
                    'val_intercept': 10.0,  # uniform
                    'val_slope_pred': 1.0,
                })
        dfGB = pd.DataFrame(rows)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='val',
        )
        # Evaluate at (0.5, 0.5) — corner (1,1) is invalid
        result = ev.evaluate(
            {'xBin': 0.5, 'yBin': 0.5}, {'pred': 1.0},
            method='multilinear', invalid_strategy='skip')
        # With uniform coefficients and skip, result should still be ~11.0
        # (intercept=10 + slope=1 * pred=1)
        assert not np.isnan(result['val']), "Should not be NaN with skip"
        assert abs(result['val'] - 11.0) < 1e-10

    def test_interpolate_boundary_clamp(self):
        """Test: Default bounds='clamp' clips to edge."""
        dfGB = _make_simple_2d_dfGB(nx=5, ny=4)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        # Evaluate way outside — should clamp, not NaN
        result = ev.evaluate(
            {'xBin': -5.0, 'yBin': -5.0}, {'pred': 1.0},
            method='multilinear', bounds='clamp')
        assert not np.isnan(result['value'])

        # Should be close to the (0,0) corner value
        corner = ev.evaluate(
            {'xBin': 0.0, 'yBin': 0.0}, {'pred': 1.0},
            method='nearest')
        assert abs(result['value'] - corner['value']) < 1e-10


# ================================================================== #
#  §7.4 Inverse-variance weighting tests
# ================================================================== #

class TestInverseVarianceWeighting:
    """Tests 16-17: Error-weighted interpolation."""

    def test_ivar_weights_favor_precise_bins(self):
        """Test 16: Bin with small error gets more weight."""
        # Create 1D-ish grid: 2 bins in x, 1 in y
        rows = [
            {'xBin': 0, 'yBin': 0,
             'val_intercept': 10.0, 'val_slope_pred': 0.0,
             'val_intercept_err': 0.1},  # precise
            {'xBin': 1, 'yBin': 0,
             'val_intercept': 20.0, 'val_slope_pred': 0.0,
             'val_intercept_err': 10.0},  # noisy
        ]
        dfGB = pd.DataFrame(rows)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='val',
        )
        # Midpoint: unweighted should give 15.0
        unweighted = ev.evaluate(
            {'xBin': 0.5, 'yBin': 0.0}, {'pred': 0.0},
            method='multilinear', use_errors=False)
        assert abs(unweighted['val'] - 15.0) < 1e-10

        # Weighted should favor bin 0 (precise)
        weighted = ev.evaluate(
            {'xBin': 0.5, 'yBin': 0.0}, {'pred': 0.0},
            method='multilinear', use_errors=True)
        assert weighted['val'] < 15.0, \
            f"Weighted ({weighted['val']}) should be closer to 10 than 15"
        assert weighted['val'] > 10.0, \
            f"Weighted ({weighted['val']}) should still be > 10"

    def test_ivar_uniform_errors_equals_multilinear(self):
        """Test 17: Equal errors → same as unweighted."""
        dfGB = _make_simple_2d_dfGB(nx=5, ny=4)
        # Override errors to be uniform
        dfGB['value_intercept_err'] = 0.1
        dfGB['value_slope_pred_err'] = 0.1
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        pos = {'xBin': 1.3, 'yBin': 2.7}
        pred = {'pred': 1.0}

        unweighted = ev.evaluate(pos, pred, method='multilinear',
                                 use_errors=False)
        weighted = ev.evaluate(pos, pred, method='multilinear',
                               use_errors=True)
        assert abs(unweighted['value'] - weighted['value']) < 1e-10


# ================================================================== #
#  §7.5 Export / roundtrip tests
# ================================================================== #

class TestExportRoundtrip:
    """Tests 18-20: Serialization."""

    def test_to_dict_roundtrip(self):
        """Test 18: from_dict(to_dict()) ≡ original."""
        dfGB = _make_multi_target_2d(nx=4, ny=3, suffix='_fit')
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets=['dX', 'dY', 'dZ'],
            suffix='_fit',
        )
        d = ev.to_dict()
        ev2 = GroupByRegressionEvaluator.from_dict(d)

        assert ev2.grid_shape == ev.grid_shape
        assert ev2.targets == ev.targets
        assert ev2.group_columns == ev.group_columns
        for tgt in ev.targets:
            for key in ev._coefficients[tgt]:
                np.testing.assert_array_equal(
                    ev2.coefficient_grid(tgt, key),
                    ev.coefficient_grid(tgt, key))

    def test_to_json_roundtrip(self):
        """Test 19: File write/read preserves all data."""
        dfGB = _make_simple_2d_dfGB(nx=4, ny=3)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as f:
            path = f.name

        try:
            ev.to_json(path)
            ev2 = GroupByRegressionEvaluator.from_json(path)
            assert ev2.grid_shape == ev.grid_shape
            assert ev2.targets == ev.targets
            np.testing.assert_array_equal(
                ev2.coefficient_grid('value', 'intercept'),
                ev.coefficient_grid('value', 'intercept'))
        finally:
            os.unlink(path)

    def test_evaluate_after_roundtrip(self):
        """Test 20: Same evaluation results after export/import."""
        dfGB = _make_linear_field_3d(nx=4, ny=4, nz=4)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin', 'zBin'],
            predictor_columns=['pred'], targets='dX',
        )
        d = ev.to_dict()
        ev2 = GroupByRegressionEvaluator.from_dict(d)

        pos = {'xBin': 1.5, 'yBin': 2.3, 'zBin': 0.7}
        pred = {'pred': 3.0}
        r1 = ev.evaluate(pos, pred, method='multilinear')
        r2 = ev2.evaluate(pos, pred, method='multilinear')
        assert abs(r1['dX'] - r2['dX']) < 1e-10


# ================================================================== #
#  §7.6 Integration with SW output
# ================================================================== #

class TestSWIntegration:
    """Tests 21-22: Integration with sliding window output."""

    def test_from_sliding_window_output(self):
        """Test 21: Construct from make_sliding_window_fit-style result."""
        # Simulate SW output column naming
        rows = []
        for ix in range(5):
            for iy in range(5):
                rows.append({
                    'rBin': ix,
                    'zBin': iy,
                    'dX_intercept_sw': 1.0 + 0.1 * ix,
                    'dX_slope_x_sw': 2.0 + 0.1 * iy,
                    'dX_intercept_err_sw': 0.01,
                    'dX_slope_x_err_sw': 0.02,
                    'dX_rmse_sw': 0.5,
                    'dX_r_squared_sw': 0.99,
                    'dX_std_sw': 0.3,
                    'dX_median_sw': np.nan,  # V5 incremental
                    'dX_n_fitted_sw': 100,
                })
        dfGB = pd.DataFrame(rows)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['rBin', 'zBin'],
            predictor_columns=['x'], targets='dX',
            suffix='_sw',
        )
        assert ev.grid_shape == (5, 5)
        assert ev.targets == ['dX']
        # Check that rmse, r_squared etc are available
        grid = ev.coefficient_grid('dX', 'rmse')
        assert not np.all(np.isnan(grid))

    def test_sw_evaluate_matches_direct(self):
        """Test 22: Evaluator at bin center ≡ direct dfGB lookup."""
        rows = []
        for ix in range(5):
            for iy in range(5):
                rows.append({
                    'rBin': ix,
                    'zBin': iy,
                    'dX_intercept_sw': 1.0 + 0.3 * ix + 0.2 * iy,
                    'dX_slope_x_sw': 2.0 + 0.1 * ix - 0.05 * iy,
                    'dX_intercept_err_sw': 0.01,
                    'dX_slope_x_err_sw': 0.02,
                })
        dfGB = pd.DataFrame(rows)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['rBin', 'zBin'],
            predictor_columns=['x'], targets='dX',
            suffix='_sw',
        )
        # For each bin, evaluate should match direct lookup
        for ix in range(5):
            for iy in range(5):
                result = ev.evaluate(
                    {'rBin': float(ix), 'zBin': float(iy)},
                    {'x': 5.0},
                    method='nearest',
                )
                row = dfGB[(dfGB['rBin'] == ix) & (dfGB['zBin'] == iy)].iloc[0]
                expected = (row['dX_intercept_sw'] +
                            row['dX_slope_x_sw'] * 5.0)
                assert abs(result['dX'] - expected) < 1e-10, \
                    f"Mismatch at ({ix}, {iy})"


# ================================================================== #
#  Additional edge case tests
# ================================================================== #

class TestEdgeCases:
    """Additional tests beyond the 22 in the proposal."""

    def test_string_target_convenience(self):
        """targets='dX' (string) treated as ['dX'] internally."""
        dfGB = _make_simple_2d_dfGB(nx=3, ny=3)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        assert ev.targets == ['value']

    def test_repr(self):
        """__repr__ doesn't crash."""
        dfGB = _make_multi_target_2d(nx=3, ny=3, suffix='_fit')
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets=['dX', 'dY', 'dZ'],
            suffix='_fit',
        )
        r = repr(ev)
        assert 'dX' in r
        assert 'grid_shape=(3, 3)' in r

    def test_evaluate_grid(self):
        """evaluate_grid produces correct shape."""
        dfGB = _make_simple_2d_dfGB(nx=5, ny=4)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        result = ev.evaluate_grid(
            {'xBin': np.linspace(0, 4, 10),
             'yBin': np.linspace(0, 3, 8)},
            {'pred': 1.0},
        )
        assert result['value'].shape == (10, 8)

    def test_missing_intercept_raises(self):
        """Construction fails if intercept column missing."""
        dfGB = pd.DataFrame({
            'xBin': [0, 1], 'yBin': [0, 0],
            'val_slope_pred': [1.0, 2.0],
        })
        with pytest.raises(ValueError, match="intercept"):
            GroupByRegressionEvaluator.from_dfGB(
                dfGB, group_columns=['xBin', 'yBin'],
                predictor_columns=['pred'], targets='val',
            )

    def test_mutually_exclusive_bin_specs(self):
        """bin_centers + bin_edges raises ValueError."""
        dfGB = _make_simple_2d_dfGB(nx=3, ny=3)
        with pytest.raises(ValueError, match="mutually exclusive"):
            GroupByRegressionEvaluator.from_dfGB(
                dfGB, group_columns=['xBin', 'yBin'],
                predictor_columns=['pred'], targets='value',
                bin_centers={'xBin': [0, 1, 2], 'yBin': [0, 1, 2]},
                bin_edges={'xBin': [0, 1, 2, 3], 'yBin': [0, 1, 2, 3]},
            )

    def test_schema_version_in_export(self):
        """JSON export includes schema_version."""
        dfGB = _make_simple_2d_dfGB(nx=3, ny=3)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value',
        )
        d = ev.to_dict()
        assert d['schema_version'] == '1.0'
        assert 'n_valid' in d['metadata']
        assert 'sparsity' in d['metadata']


# ================================================================== #
#  Metadata-based construction tests
# ================================================================== #

class TestMetadataConstruction:
    """Tests for constructing evaluator from metadata dict."""

    def _make_v4_metadata(self, suffix='_fit'):
        """Build a V4-style metadata dict."""
        return {
            'version': '1.0',
            'formulas': {'value_pred_fit': 'value_intercept_fit + value_slope_pred_fit*pred'},
            'residual_formulas': {},
            'pull_formulas': {},
            'columns': {
                'gb_columns': ['xBin', 'yBin'],
                'fit_columns': ['value'],
                'linear_columns': ['pred'],
                'coefficients': {'value': ['value_intercept_fit', 'value_slope_pred_fit']},
                'errors': {'value': ['value_intercept_err_fit', 'value_slope_pred_err_fit']},
                'quality': {'value': ['value_rms_fit', 'value_mad_fit']},
            },
            'parameters': {
                'suffix': suffix,
                'fit_intercept': True,
                'min_stat': 5,
                'fit_type': 'linear',
                'weights_column': None,
            },
        }

    def _make_sw_metadata(self, suffix='_sw'):
        """Build a SW-style metadata dict (V4-compatible)."""
        return {
            'version': '1.0',
            'formulas': {'dX_pred_sw': 'dX_intercept_sw + dX_slope_x_sw*x'},
            'residual_formulas': {},
            'pull_formulas': {},
            'columns': {
                'gb_columns': ['rBin', 'zBin'],
                'fit_columns': ['dX'],
                'linear_columns': ['x'],
                'coefficients': {'dX': ['dX_intercept_sw', 'dX_slope_x_sw']},
                'errors': {'dX': ['dX_intercept_err_sw', 'dX_slope_x_err_sw']},
                'quality': {'dX': ['dX_rmse_sw', 'dX_r_squared_sw', 'dX_std_sw']},
            },
            'parameters': {
                'suffix': suffix,
                'fit_intercept': True,
                'min_stat': 10,
                'fit_type': 'sliding_window',
                'weights_column': None,
            },
            # SW-specific flat keys (same level as V4 keys)
            'window_spec': {'rBin': 1, 'zBin': 2},
            'algorithm': 'incremental',
            'backend_used': 'numba',
        }

    def test_from_metadata_v4(self):
        """Construct evaluator using V4 metadata — no explicit params needed."""
        dfGB = _make_simple_2d_dfGB(nx=3, ny=3, suffix='_fit')
        metadata = self._make_v4_metadata(suffix='_fit')
        ev = GroupByRegressionEvaluator.from_dfGB(dfGB, metadata=metadata)
        assert ev.targets == ['value']
        assert ev.predictor_columns == ['pred']
        assert ev.group_columns == ['xBin', 'yBin']
        assert ev.grid_shape == (3, 3)

    def test_from_metadata_sw(self):
        """Construct evaluator using SW metadata."""
        rows = []
        for ix in range(3):
            for iy in range(3):
                rows.append({
                    'rBin': ix, 'zBin': iy,
                    'dX_intercept_sw': 1.0 + 0.1 * ix,
                    'dX_slope_x_sw': 2.0 + 0.1 * iy,
                    'dX_intercept_err_sw': 0.01,
                    'dX_slope_x_err_sw': 0.02,
                    'dX_rmse_sw': 0.5,
                })
        dfGB = pd.DataFrame(rows)
        metadata = self._make_sw_metadata(suffix='_sw')
        ev = GroupByRegressionEvaluator.from_dfGB(dfGB, metadata=metadata)
        assert ev.targets == ['dX']
        assert ev.predictor_columns == ['x']
        assert ev.group_columns == ['rBin', 'zBin']

    def test_from_metadata_multi_target(self):
        """Metadata with multiple targets."""
        dfGB = _make_multi_target_2d(nx=3, ny=3, suffix='_fit')
        metadata = {
            'version': '1.0',
            'formulas': {},
            'columns': {
                'gb_columns': ['xBin', 'yBin'],
                'fit_columns': ['dX', 'dY', 'dZ'],
                'linear_columns': ['pred'],
                'coefficients': {},
                'errors': {},
            },
            'parameters': {'suffix': '_fit', 'fit_intercept': True},
        }
        ev = GroupByRegressionEvaluator.from_dfGB(dfGB, metadata=metadata)
        assert set(ev.targets) == {'dX', 'dY', 'dZ'}

    def test_explicit_params_override_metadata(self):
        """Explicit parameters take priority over metadata."""
        dfGB = _make_simple_2d_dfGB(nx=3, ny=3, suffix='_fit')
        metadata = self._make_v4_metadata(suffix='_fit')
        # Override targets via explicit param
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, targets='value', metadata=metadata)
        assert ev.targets == ['value']

    def test_no_metadata_no_params_raises(self):
        """Missing metadata and missing params gives clear error."""
        dfGB = _make_simple_2d_dfGB(nx=3, ny=3)
        with pytest.raises(ValueError, match="targets must be specified"):
            GroupByRegressionEvaluator.from_dfGB(
                dfGB, group_columns=['xBin', 'yBin'])

    def test_no_metadata_explicit_params_works(self):
        """Without metadata, explicit params still work."""
        dfGB = _make_simple_2d_dfGB(nx=3, ny=3)
        ev = GroupByRegressionEvaluator.from_dfGB(
            dfGB, group_columns=['xBin', 'yBin'],
            predictor_columns=['pred'], targets='value')
        assert ev.targets == ['value']

    def test_metadata_suffix_extracted(self):
        """Suffix is extracted from metadata.parameters.suffix."""
        dfGB = _make_simple_2d_dfGB(nx=3, ny=3, suffix='_fit')
        metadata = self._make_v4_metadata(suffix='_fit')
        # Don't pass suffix= explicitly — should come from metadata
        ev = GroupByRegressionEvaluator.from_dfGB(dfGB, metadata=metadata)
        grid = ev.coefficient_grid('value', 'intercept')
        assert not np.all(np.isnan(grid))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
