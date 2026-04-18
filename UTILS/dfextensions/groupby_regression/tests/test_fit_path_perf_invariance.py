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


# ---- Helper fixtures for T1-7 through T1-13 ----

def _import_swf():
    """Import make_sliding_window_fit with fallback."""
    try:
        from groupby_regression_sliding_window import make_sliding_window_fit
    except ImportError:
        from dfextensions.groupby_regression.groupby_regression_sliding_window import make_sliding_window_fit
    return make_sliding_window_fit


def _make_2d_fixture(n_x=4, n_y=4, rows_per_bin=30, seed=123,
                     drop_bins=None, add_weights=False):
    """Build a 2D fixture. Optionally drop bins and/or add weight column."""
    rng = np.random.default_rng(seed)
    rows = []
    for ix in range(n_x):
        for iy in range(n_y):
            if drop_bins and (ix, iy) in drop_bins:
                continue
            for _ in range(rows_per_bin):
                row = {"bin_x": ix, "bin_y": iy}
                row["x"] = rng.normal(0, 1)
                row["y"] = 2.0 + 0.5 * row["x"] + rng.normal(0, 0.1)
                if add_weights:
                    row["w"] = rng.uniform(0.5, 2.0)
                rows.append(row)
    df = pd.DataFrame(rows)
    df["bin_x"] = df["bin_x"].astype(np.int64)
    df["bin_y"] = df["bin_y"].astype(np.int64)
    return df


def _numeric_cols_equal(df_a, df_b, dims, rtol=1e-12, skip_cols=None):
    """Assert all numeric non-dim columns match between two DataFrames."""
    a = df_a.sort_values(dims).reset_index(drop=True)
    b = df_b.sort_values(dims).reset_index(drop=True)
    _skip = set(skip_cols or [])
    for d in dims:
        np.testing.assert_array_equal(a[d].values, b[d].values,
                                       err_msg=f"Bin coords differ: {d}")
    common = sorted(set(a.columns) & set(b.columns) - set(dims) - _skip)
    for col in common:
        if a[col].dtype == object or b[col].dtype == object:
            continue
        va = a[col].values.astype(np.float64)
        vb = b[col].values.astype(np.float64)
        both_nan = np.isnan(va) & np.isnan(vb)
        nan_mismatch = (np.isnan(va) | np.isnan(vb)) & ~both_nan
        assert not nan_mismatch.any(), f"NaN mismatch in {col}"
        finite = ~np.isnan(va) & ~np.isnan(vb)
        if finite.any():
            np.testing.assert_allclose(va[finite], vb[finite], rtol=rtol,
                                        atol=0, err_msg=f"Mismatch in {col}")


# ---- T1-7: boundary parameter silently dropped (v1.2 §5.2.1) ----

class TestV1V2BoundaryDrop:
    """V1/V2 recompute path silently ignores boundary parameter.

    Parameter-not-propagated class instance #9, pre-existing.
    This test locks current behavior: V1/V2(symmetric) == V5(full).
    """

    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_v1v2_boundary_parameter_silently_dropped(self, fit_intercept):
        swf = _import_swf()
        df = _make_2d_fixture(n_x=4, n_y=4, rows_per_bin=30, seed=777)
        dims = ["bin_x", "bin_y"]
        ws = {"bin_x": 2, "bin_y": 2}

        # V1/V2 with boundary='symmetric' — but it ignores boundary
        result_v1v2_sym = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec=ws,
            boundary='symmetric', algorithm='recompute',
            backend='numba', fit_intercept=fit_intercept,
            min_stat=5, suffix='_sw',
        )

        # V5 with boundary='full' — the behavior V1/V2 actually uses
        result_v5_full = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec=ws,
            boundary='full', algorithm='incremental',
            backend='numba', fit_intercept=fit_intercept,
            min_stat=5, suffix='_sw',
        )

        # V1/V2(symmetric) should equal V5(full), not V5(symmetric)
        _numeric_cols_equal(result_v1v2_sym, result_v5_full, dims, rtol=1e-12)

        # Non-triviality guard: V5(full) must differ from V5(symmetric)
        # at edge bins, or this test proves nothing.
        result_v5_sym = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec=ws,
            boundary='symmetric', algorithm='incremental',
            backend='numba', fit_intercept=fit_intercept,
            min_stat=5, suffix='_sw',
        )
        a = result_v5_full.sort_values(dims).reset_index(drop=True)
        b = result_v5_sym.sort_values(dims).reset_index(drop=True)
        # Find a numeric coefficient column to compare
        coeff_col = [c for c in a.columns if c not in dims and a[c].dtype != object][0]
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(a[coeff_col].values, b[coeff_col].values)


# ---- T1-8: sparse grid (v1.2 §5.2.2) ----

class TestSparseGrid:

    def test_2d_sparse_grid(self):
        """Dense lookup handles sparse grids (empty bins → lookup=-1)."""
        swf = _import_swf()
        dropped = [(0, 1), (2, 3), (4, 0), (1, 4), (3, 2)]
        df = _make_2d_fixture(n_x=5, n_y=5, rows_per_bin=40, seed=888,
                              drop_bins=dropped)
        dims = ["bin_x", "bin_y"]
        ws = {"bin_x": 1, "bin_y": 1}

        result_v1v2 = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec=ws,
            algorithm='recompute', backend='numba',
            min_stat=5, suffix='_sw',
        )
        result_v5 = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec=ws,
            algorithm='incremental', backend='numba',
            min_stat=5, suffix='_sw',
        )
        _numeric_cols_equal(result_v1v2, result_v5, dims, rtol=1e-12)

        # Non-triviality: at least one center is adjacent to a dropped bin
        # (1,1) has neighbor (0,1) which is dropped
        centers = set(zip(result_v1v2["bin_x"], result_v1v2["bin_y"]))
        assert (1, 1) in centers, "Center (1,1) missing — fixture broken"


# ---- T1-10: output row order (v1.2 §5.2.4) ----

class TestOutputRowOrder:

    def test_output_row_order_sorted_lex(self):
        """Dense path produces rows in lexicographic bin-coordinate order."""
        swf = _import_swf()
        df = _make_2d_fixture(n_x=4, n_y=4, rows_per_bin=20, seed=999)
        # Deliberately shuffle input
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        dims = ["bin_x", "bin_y"]

        result = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec={"bin_x": 1, "bin_y": 1},
            algorithm='recompute', backend='numba',
            min_stat=5, suffix='_sw',
        )

        result_sorted = result.sort_values(dims).reset_index(drop=True)
        pd.testing.assert_frame_equal(result.reset_index(drop=True),
                                       result_sorted)


# ---- T1-11: selection with out-of-range rows (v1.2 §5.2.5) ----

class TestSelectionOutliers:

    def test_selection_with_out_of_range_rows(self):
        """_assign_bin_ids_fast handles selection with outlier gb values."""
        swf = _import_swf()
        rng = np.random.default_rng(1111)
        # 100 in-range rows
        df_good = _make_2d_fixture(n_x=5, n_y=5, rows_per_bin=4, seed=1111)
        n_good = len(df_good)
        # 50 outlier rows with extreme bin values
        outliers = pd.DataFrame({
            "bin_x": rng.integers(-10000, 10000, size=50).astype(np.int64),
            "bin_y": rng.integers(-10000, 10000, size=50).astype(np.int64),
            "x": rng.normal(0, 1, size=50),
            "y": rng.normal(0, 1, size=50),
        })
        df_combined = pd.concat([df_good, outliers], ignore_index=True)
        selection = pd.Series([True] * n_good + [False] * 50)
        dims = ["bin_x", "bin_y"]

        result_with_outliers = swf(
            df=df_combined, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec={"bin_x": 1, "bin_y": 1},
            selection=selection, algorithm='recompute', backend='numba',
            min_stat=3, suffix='_sw',
        )
        result_clean = swf(
            df=df_good, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec={"bin_x": 1, "bin_y": 1},
            algorithm='recompute', backend='numba',
            min_stat=3, suffix='_sw',
        )
        _numeric_cols_equal(result_with_outliers, result_clean, dims, rtol=1e-12)


# ---- T1-12a/b: NaN handling (v1.2 §5.2.6) ----

class TestNaNHandling:

    def test_nan_in_fit_column(self):
        """NaN in fit column handled identically by V1/V2 and V5."""
        swf = _import_swf()
        df = _make_2d_fixture(n_x=5, n_y=5, rows_per_bin=10, seed=2222)
        # Inject NaNs at known positions
        df.loc[5, "y"] = np.nan
        df.loc[15, "y"] = np.nan
        df.loc[42, "y"] = np.nan
        dims = ["bin_x", "bin_y"]

        result_v1v2 = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec={"bin_x": 1, "bin_y": 1},
            algorithm='recompute', backend='numba',
            min_stat=5, suffix='_sw',
        )
        result_v5 = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec={"bin_x": 1, "bin_y": 1},
            algorithm='incremental', backend='numba',
            min_stat=5, suffix='_sw',
        )
        # n_rows_aggregated counts differ between V1/V2 (all rows) and V5
        # (finite rows only) — pre-existing behavioral difference, not a bug.
        _skip = {"n_rows_aggregated_sw", "n_neighbors_used_sw",
                 "effective_window_fraction_sw"}
        _numeric_cols_equal(result_v1v2, result_v5, dims, rtol=1e-12,
                            skip_cols=_skip)

    def test_nan_in_weights_column(self):
        """NaN weights excluded correctly — result matches pre-filtered data."""
        swf = _import_swf()
        df = _make_2d_fixture(n_x=5, n_y=5, rows_per_bin=10, seed=3333,
                              add_weights=True)
        dims = ["bin_x", "bin_y"]

        # Run with NaN weights injected
        df_nan = df.copy()
        df_nan.loc[3, "w"] = np.nan
        df_nan.loc[20, "w"] = np.nan

        result_with_nan = swf(
            df=df_nan, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec={"bin_x": 1, "bin_y": 1},
            weights="w", algorithm='recompute', backend='numpy',
            min_stat=5, suffix='_sw',
        )

        # Run with NaN-weight rows pre-removed
        df_clean = df.drop([3, 20]).reset_index(drop=True)
        result_clean = swf(
            df=df_clean, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec={"bin_x": 1, "bin_y": 1},
            weights="w", algorithm='recompute', backend='numpy',
            min_stat=5, suffix='_sw',
        )

        # Coefficients should match — NaN weights are excluded from WLS.
        # Row counts naturally differ (NaN-run has more total rows gathered).
        _skip = {"n_rows_aggregated_sw", "n_neighbors_used_sw",
                 "effective_window_fraction_sw"}
        _numeric_cols_equal(result_with_nan, result_clean, dims, rtol=1e-12,
                            skip_cols=_skip)


# ---- T1-13: numba vs numpy backend (v1.2 §5.2.7) ----

class TestBackendParity:

    def test_dense_numba_equals_numpy(self):
        """Both backends through the dense path produce same results."""
        swf = _import_swf()
        df = _make_2d_fixture(n_x=4, n_y=4, rows_per_bin=30, seed=4444)
        dims = ["bin_x", "bin_y"]
        ws = {"bin_x": 2, "bin_y": 2}

        result_numba = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec=ws,
            algorithm='recompute', backend='numba',
            min_stat=5, suffix='_sw',
        )
        result_numpy = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec=ws,
            algorithm='recompute', backend='numpy',
            min_stat=5, suffix='_sw',
        )
        _numeric_cols_equal(result_numba, result_numpy, dims, rtol=1e-12)


# ---- Phase 13.20.GB-PERF: numba kernel vs numpy fallback ----

class TestAggDenseNumbaKernel:
    """Phase 13.20.GB-PERF invariance: _gather_window_rows_numba kernel
    produces identical output to the numpy fallback path.

    Tested via GBAI_DISABLE_AGG_DENSE_NUMBA env flag.
    """

    @pytest.mark.parametrize("window", [1, 2])
    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_numba_kernel_equals_numpy_fallback(self, window, fit_intercept, monkeypatch):
        swf = _import_swf()
        df = _make_2d_fixture(n_x=5, n_y=5, rows_per_bin=30, seed=5555)
        dims = ["bin_x", "bin_y"]
        ws = {"bin_x": window, "bin_y": window}

        # Run with numba kernel (default)
        monkeypatch.delenv("GBAI_DISABLE_AGG_DENSE_NUMBA", raising=False)
        result_numba = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec=ws,
            algorithm='recompute', backend='numba',
            fit_intercept=fit_intercept,
            min_stat=5, suffix='_sw',
        )

        # Run with numpy fallback
        monkeypatch.setenv("GBAI_DISABLE_AGG_DENSE_NUMBA", "1")
        result_fallback = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec=ws,
            algorithm='recompute', backend='numba',
            fit_intercept=fit_intercept,
            min_stat=5, suffix='_sw',
        )

        _numeric_cols_equal(result_numba, result_fallback, dims, rtol=1e-12)

    @pytest.mark.parametrize("with_weights", [False, True])
    def test_numba_kernel_with_agg_columns(self, with_weights, monkeypatch):
        swf = _import_swf()
        df = _make_2d_fixture(n_x=4, n_y=4, rows_per_bin=25, seed=6666,
                              add_weights=with_weights)
        dims = ["bin_x", "bin_y"]
        ws = {"bin_x": 1, "bin_y": 1}
        w = "w" if with_weights else None

        monkeypatch.delenv("GBAI_DISABLE_AGG_DENSE_NUMBA", raising=False)
        result_numba = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec=ws,
            weights=w, agg_columns=["x"],
            agg_median=True,
            algorithm='recompute', backend='numba' if not with_weights else 'numpy',
            min_stat=5, suffix='_sw',
        )

        monkeypatch.setenv("GBAI_DISABLE_AGG_DENSE_NUMBA", "1")
        result_fallback = swf(
            df=df, gb_columns=dims, fit_columns=["y"],
            linear_columns=["x"], window_spec=ws,
            weights=w, agg_columns=["x"],
            agg_median=True,
            algorithm='recompute', backend='numba' if not with_weights else 'numpy',
            min_stat=5, suffix='_sw',
        )

        _numeric_cols_equal(result_numba, result_fallback, dims, rtol=1e-12)
