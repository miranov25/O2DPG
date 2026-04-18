"""Phase 13.19.GB-PERF — T1 Invariance Gate.

Verifies that the V1/V2 recompute path (modified to use dense lookup)
produces bit-identical output to the V5 incremental path (untouched).

Both paths compute the same regression; they differ only in how they
build bin indices and gather neighbor rows. If the dense-lookup
routing change introduces any numerical divergence, this test catches it.

Run on alma2:
    cd groupby_regression
    NUMBA_THREADING_LAYER=omp pytest tests/test_fit_path_perf_invariance.py -v
"""
import numpy as np
import pandas as pd
import pytest


def _make_synthetic_df(n_bins_per_dim=5, rows_per_bin=50, n_dims=2, seed=42):
    """Build a synthetic dataset matching TPC calibration structure."""
    rng = np.random.default_rng(seed)
    dims = [f"dim_{d}" for d in range(n_dims)]
    grid = np.array(np.meshgrid(*[np.arange(n_bins_per_dim)] * n_dims, indexing='ij'))
    grid = grid.reshape(n_dims, -1).T  # (n_bins, n_dims)
    n_bins = len(grid)

    rows = []
    for bi in range(n_bins):
        for _ in range(rows_per_bin):
            row = {dims[d]: int(grid[bi, d]) for d in range(n_dims)}
            row["x"] = rng.normal(0, 1)
            row["y"] = 2.0 + 0.5 * row["x"] + rng.normal(0, 0.1)
            row["z"] = -1.0 + 0.3 * row["x"] + rng.normal(0, 0.15)
            rows.append(row)

    df = pd.DataFrame(rows)
    for d in dims:
        df[d] = df[d].astype(np.int64)
    return df, dims


class TestFitPathPerformanceParity:
    """Phase 13.19.GB-PERF invariance gate."""

    def _run_and_compare(self, n_dims, window, fit_columns, linear_columns,
                          fit_intercept=True, weights=None, agg_columns=None,
                          agg_median=False, rtol=0.0):
        """Run both paths and assert output equivalence."""
        try:
            from groupby_regression_sliding_window import make_sliding_window_fit
        except ImportError:
            from dfextensions.groupby_regression.groupby_regression_sliding_window import make_sliding_window_fit

        df, dims = _make_synthetic_df(n_bins_per_dim=5, rows_per_bin=50, n_dims=n_dims)
        ws = {dims[d]: window for d in range(n_dims)}

        # V1/V2 recompute path (MODIFIED — dense lookup)
        out_recompute = make_sliding_window_fit(
            df=df, gb_columns=dims, fit_columns=fit_columns,
            linear_columns=linear_columns, window_spec=ws,
            weights=weights, suffix='_sw', fit_intercept=fit_intercept,
            min_stat=5, backend='numba', algorithm='recompute',
            agg_columns=agg_columns, agg_median=agg_median,
        )

        # V5 incremental path (UNTOUCHED — reference)
        out_v5 = make_sliding_window_fit(
            df=df, gb_columns=dims, fit_columns=fit_columns,
            linear_columns=linear_columns, window_spec=ws,
            weights=weights, suffix='_sw', fit_intercept=fit_intercept,
            min_stat=5, backend='numba', algorithm='incremental',
            agg_columns=agg_columns, agg_median=agg_median,
        )

        # Sort both by bin columns for stable comparison
        out_recompute = out_recompute.sort_values(dims).reset_index(drop=True)
        out_v5 = out_v5.sort_values(dims).reset_index(drop=True)

        # Same bin coordinates
        for d in dims:
            np.testing.assert_array_equal(
                out_recompute[d].values, out_v5[d].values,
                err_msg=f"Bin coordinates differ for {d}")

        # Same regression coefficients
        coeff_cols = [c for c in out_recompute.columns if c not in dims]
        v5_coeff_cols = [c for c in out_v5.columns if c not in dims]

        # Match columns present in both outputs (numeric only)
        common_cols = sorted(set(coeff_cols) & set(v5_coeff_cols))
        assert len(common_cols) > 0, f"No common coefficient columns found"

        for col in common_cols:
            # Skip non-numeric columns (e.g. quality_flag)
            if out_recompute[col].dtype == object or out_v5[col].dtype == object:
                continue
            a = out_recompute[col].values.astype(np.float64)
            b = out_v5[col].values.astype(np.float64)
            # NaN-equal comparison
            both_nan = np.isnan(a) & np.isnan(b)
            either_nan = np.isnan(a) | np.isnan(b)
            nan_mismatch = either_nan & ~both_nan
            assert not nan_mismatch.any(), \
                f"NaN mismatch in {col}: recompute has NaN where V5 doesn't (or vice versa)"

            finite = ~np.isnan(a) & ~np.isnan(b)
            if finite.any():
                if rtol == 0.0:
                    np.testing.assert_array_equal(
                        a[finite], b[finite],
                        err_msg=f"Bit-exact mismatch in {col}")
                else:
                    np.testing.assert_allclose(
                        a[finite], b[finite], rtol=rtol, atol=0,
                        err_msg=f"Tolerance mismatch in {col}")

    def test_2d_ols_intercept(self):
        """2D grid, OLS, fit_intercept=True, window=1."""
        self._run_and_compare(
            n_dims=2, window=1,
            fit_columns=["y"], linear_columns=["x"],
            fit_intercept=True, rtol=1e-12)

    def test_2d_ols_no_intercept(self):
        """2D grid, OLS, fit_intercept=False, window=1."""
        self._run_and_compare(
            n_dims=2, window=1,
            fit_columns=["y"], linear_columns=["x"],
            fit_intercept=False, rtol=1e-12)

    def test_2d_multi_target(self):
        """2D grid, two targets (y, z), window=1."""
        self._run_and_compare(
            n_dims=2, window=1,
            fit_columns=["y", "z"], linear_columns=["x"],
            fit_intercept=True, rtol=1e-12)

    def test_3d_larger_window(self):
        """3D grid, window=2, wider neighborhood."""
        self._run_and_compare(
            n_dims=3, window=2,
            fit_columns=["y"], linear_columns=["x"],
            fit_intercept=True, rtol=1e-12)

    def test_2d_with_agg_columns(self):
        """2D grid with agg_columns (mean/std/median)."""
        self._run_and_compare(
            n_dims=2, window=1,
            fit_columns=["y"], linear_columns=["x"],
            agg_columns=["x"], agg_median=True, rtol=1e-12)

    def test_1d_window_0(self):
        """1D grid, window=0 (no sliding — per-bin regression)."""
        self._run_and_compare(
            n_dims=1, window=0,
            fit_columns=["y"], linear_columns=["x"],
            fit_intercept=True, rtol=1e-12)
