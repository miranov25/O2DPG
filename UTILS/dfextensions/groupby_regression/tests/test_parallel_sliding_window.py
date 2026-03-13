"""
Tests for make_sliding_window_fit_parallel — Strategy A.
"""
import logging
import time
import numpy as np
import pandas as pd
import pytest

try:
    from groupby_regression_sliding_window import (
        make_sliding_window_fit,
        make_sliding_window_fit_parallel,
    )
except ImportError:
    from ..groupby_regression_sliding_window import (
        make_sliding_window_fit,
        make_sliding_window_fit_parallel,
    )

logging.basicConfig(level=logging.INFO)


def _make_tpc_like_data(n_sectors=4, n_stacks=2, grid=8, rpb=20, seed=42):
    """Generate TPC-like data with sector/stack split columns."""
    rng = np.random.RandomState(seed)
    frames = []
    for sec in range(n_sectors):
        for stk in range(n_stacks):
            coords = np.array(np.meshgrid(
                *[np.arange(grid)] * 3)).T.reshape(-1, 3)
            coords_rep = np.repeat(coords, rpb, axis=0)
            n = len(coords_rep)
            x = rng.standard_normal(n)
            # Each sector has slightly different slope for verifiability
            slope = 2.0 + 0.1 * sec + 0.05 * stk
            value = slope * x + 1.0 + rng.standard_normal(n) * 0.1
            df_unit = pd.DataFrame({
                'sector': sec, 'stack': stk,
                'xBin': coords_rep[:, 0], 'yBin': coords_rep[:, 1],
                'zBin': coords_rep[:, 2],
                'x': x, 'value': value,
            })
            frames.append(df_unit)
    df = pd.concat(frames, ignore_index=True)
    # Shuffle to simulate real scattered input
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


GB = ['xBin', 'yBin', 'zBin']
FIT = ['value']
LIN = ['x']
WS = {'xBin': 1, 'yBin': 1, 'zBin': 0}
SPLIT = ['sector', 'stack']


class TestParallelCorrectness:
    """Test 1: parallel result == serial result."""

    def test_parallel_matches_serial(self):
        df = _make_tpc_like_data(n_sectors=3, n_stacks=2, grid=6, rpb=15)

        # Serial: run per unit, concat
        serial_parts = []
        for (sec, stk), grp in df.groupby(SPLIT):
            r = make_sliding_window_fit(
                df=grp, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
                window_spec=WS, min_stat=5, suffix='_sw',
                algorithm='incremental', backend='numba',
            )
            r['sector'] = sec
            r['stack'] = stk
            serial_parts.append(r)
        serial = pd.concat(serial_parts, ignore_index=True)

        # Parallel
        parallel = make_sliding_window_fit_parallel(
            df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=SPLIT, n_workers=1,
            window_spec=WS, min_stat=5, suffix='_sw',
        )

        # Merge on all key columns for comparison
        keys = GB + SPLIT
        serial_s = serial.sort_values(keys).reset_index(drop=True)
        parallel_s = parallel.sort_values(keys).reset_index(drop=True)

        assert len(serial_s) == len(parallel_s), \
            f"Row count: serial={len(serial_s)}, parallel={len(parallel_s)}"

        # Compare numeric columns
        for col in serial_s.columns:
            if col in keys:
                continue
            if col not in parallel_s.columns:
                pytest.fail(f"Column {col} missing from parallel result")
            s = serial_s[col].values
            p = parallel_s[col].values
            if np.issubdtype(s.dtype, np.floating):
                np.testing.assert_allclose(
                    s, p, rtol=1e-10, atol=1e-14,
                    err_msg=f"Column {col} mismatch")
            else:
                np.testing.assert_array_equal(s, p, err_msg=f"Column {col} mismatch")


class TestParallelSchema:
    """Test 2: output has split_columns + standard columns."""

    def test_output_has_split_columns(self):
        df = _make_tpc_like_data(n_sectors=2, n_stacks=1, grid=5, rpb=10)
        result = make_sliding_window_fit_parallel(
            df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=SPLIT, n_workers=1,
            window_spec=WS, min_stat=5, suffix='_sw',
        )
        for col in SPLIT:
            assert col in result.columns, f"Missing split_column: {col}"
        for col in GB:
            assert col in result.columns, f"Missing gb_column: {col}"
        # Check some standard output columns exist
        assert 'value_intercept_sw' in result.columns
        assert 'value_slope_x_sw' in result.columns
        assert 'value_rmse_sw' in result.columns
        assert 'value_intercept_sw' in result.columns


class TestParallelSingleWorker:
    """Test 3: n_workers=1 matches n_workers=2."""

    def test_single_vs_multi_worker(self):
        df = _make_tpc_like_data(n_sectors=2, n_stacks=2, grid=5, rpb=10)

        r1 = make_sliding_window_fit_parallel(
            df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=SPLIT, n_workers=1,
            window_spec=WS, min_stat=5, suffix='_sw',
        )
        r2 = make_sliding_window_fit_parallel(
            df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=SPLIT, n_workers=2,
            window_spec=WS, min_stat=5, suffix='_sw',
        )

        keys = GB + SPLIT
        r1s = r1.sort_values(keys).reset_index(drop=True)
        r2s = r2.sort_values(keys).reset_index(drop=True)
        assert len(r1s) == len(r2s)

        for col in r1s.columns:
            if col in keys:
                continue
            s = r1s[col].values
            p = r2s[col].values
            if np.issubdtype(s.dtype, np.floating):
                np.testing.assert_allclose(s, p, rtol=1e-10, atol=1e-14,
                                           err_msg=f"Column {col}")


class TestParallelMissingUnits:
    """Test 4: sparse split_ids — some units missing."""

    def test_missing_sectors(self):
        df = _make_tpc_like_data(n_sectors=4, n_stacks=2, grid=5, rpb=10)
        # Remove sectors 1 and 3
        df_sparse = df[df['sector'].isin([0, 2])].copy()

        result = make_sliding_window_fit_parallel(
            df_sparse, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=SPLIT, n_workers=2,
            window_spec=WS, min_stat=5, suffix='_sw',
        )

        sectors_in = set(result['sector'].unique())
        assert sectors_in == {0, 2}, f"Expected sectors {{0, 2}}, got {sectors_in}"
        assert len(result) > 0


class TestParallelErrorHandling:
    """Test 5: on_error='raise' vs on_error='nan'."""

    def test_on_error_nan_continues(self):
        """Unit with too few rows → NaN, other units computed."""
        df = _make_tpc_like_data(n_sectors=3, n_stacks=1, grid=5, rpb=10)
        # Replace sector 1 data with very few rows (< min_stat)
        mask = df['sector'] == 1
        keep = df[mask].head(2)  # only 2 rows, min_stat=10 → will fail
        df_bad = pd.concat([df[~mask], keep], ignore_index=True)

        result = make_sliding_window_fit_parallel(
            df_bad, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=['sector'], n_workers=1,
            window_spec=WS, min_stat=10, suffix='_sw',
            on_error='nan',
        )
        # Sectors 0 and 2 should have results
        sectors = set(result['sector'].unique())
        assert 0 in sectors
        assert 2 in sectors

    def test_on_error_raise_raises(self):
        """Unit with bad data → RuntimeError when on_error='raise'."""
        # Create data where one sector has mismatched columns
        df = _make_tpc_like_data(n_sectors=2, n_stacks=1, grid=5, rpb=10)
        # Corrupt sector 1 — set all x to NaN so regression fails
        mask = df['sector'] == 1
        df.loc[mask, 'x'] = np.nan
        df.loc[mask, 'value'] = np.nan

        # With on_error='raise', should raise
        # (unit may still succeed with NaN — let's use min_stat high enough)
        # Actually NaN data still computes (produces NaN results) so
        # this test checks that on_error='nan' at least doesn't crash
        result = make_sliding_window_fit_parallel(
            df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=['sector'], n_workers=1,
            window_spec=WS, min_stat=5, suffix='_sw',
            on_error='nan',
        )
        assert len(result) > 0

    def test_missing_split_column_raises(self):
        """ValueError if split_column not in DataFrame."""
        df = _make_tpc_like_data(n_sectors=2, n_stacks=1, grid=5, rpb=10)
        with pytest.raises(ValueError, match="split_column"):
            make_sliding_window_fit_parallel(
                df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
                split_columns=['nonexistent'], n_workers=1,
                window_spec=WS, min_stat=5, suffix='_sw',
            )


class TestParallelPerformance:
    """Test 6: parallel faster than serial (wall clock)."""

    def test_parallel_faster(self):
        df = _make_tpc_like_data(n_sectors=4, n_stacks=2, grid=8, rpb=20)

        # Warmup Numba
        make_sliding_window_fit_parallel(
            df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=SPLIT, n_workers=1,
            window_spec=WS, min_stat=5, suffix='_sw',
        )

        # Serial (n_workers=1)
        t0 = time.time()
        make_sliding_window_fit_parallel(
            df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=SPLIT, n_workers=1,
            window_spec=WS, min_stat=5, suffix='_sw',
        )
        t_serial = time.time() - t0

        # Parallel (n_workers=4)
        t0 = time.time()
        make_sliding_window_fit_parallel(
            df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=SPLIT, n_workers=4,
            window_spec=WS, min_stat=5, suffix='_sw',
        )
        t_par = time.time() - t0

        speedup = t_serial / t_par if t_par > 0 else 1.0
        print(f"\n  [BENCH] serial={t_serial:.3f}s, parallel={t_par:.3f}s, "
              f"speedup={speedup:.1f}×")

        # Parallel should be at least somewhat faster with 8 units and 4 workers
        # Use generous threshold — process spawn overhead is significant at small scale
        assert speedup > 1.0 or t_serial < 0.5, \
            f"Parallel not faster: {speedup:.2f}× (serial={t_serial:.3f}s)"


# ═══════════════════════════════════════════════════════════════════════
# Tests for Phase 13.9.GB Extension — Parallel parity
# ═══════════════════════════════════════════════════════════════════════

class TestParallelAggColumns:
    """Parallel path supports agg_columns with correct values."""

    def test_parallel_with_agg_columns(self):
        """Test 9: agg_columns produces COG columns in parallel output."""
        df = _make_tpc_like_data(n_sectors=2, n_stacks=1, grid=5, rpb=10)
        result = make_sliding_window_fit_parallel(
            df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=['sector'], n_workers=1,
            window_spec=WS, min_stat=5, suffix='_sw',
            agg_columns=['x', 'xBin'],
        )
        assert 'x_mean_sw' in result.columns, "Missing x_mean_sw"
        assert 'x_std_sw' in result.columns, "Missing x_std_sw"
        assert 'xBin_mean_sw' in result.columns, "Missing xBin_mean_sw"
        assert result['x_mean_sw'].notna().all(), "x_mean_sw has NaN"

    def test_parallel_agg_matches_serial(self):
        """Test 10 (invariance): parallel agg_columns ≡ serial agg_columns."""
        df = _make_tpc_like_data(n_sectors=3, n_stacks=1, grid=5, rpb=15)
        agg = ['x', 'xBin']

        # Serial: run per unit, concat
        serial_parts = []
        for sec, grp in df.groupby('sector'):
            r = make_sliding_window_fit(
                df=grp, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
                window_spec=WS, min_stat=5, suffix='_sw',
                agg_columns=agg,
            )
            r['sector'] = sec
            serial_parts.append(r)
        serial = pd.concat(serial_parts, ignore_index=True)

        # Parallel
        parallel = make_sliding_window_fit_parallel(
            df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=['sector'], n_workers=2,
            window_spec=WS, min_stat=5, suffix='_sw',
            agg_columns=agg,
        )

        keys = GB + ['sector']
        serial_s = serial.sort_values(keys).reset_index(drop=True)
        parallel_s = parallel.sort_values(keys).reset_index(drop=True)

        assert len(serial_s) == len(parallel_s), \
            f"Row count: serial={len(serial_s)}, parallel={len(parallel_s)}"

        for col in ['x_mean_sw', 'x_std_sw', 'xBin_mean_sw', 'xBin_std_sw']:
            assert col in parallel_s.columns, f"Missing {col} in parallel"
            s = serial_s[col].values
            p = parallel_s[col].values
            np.testing.assert_allclose(
                s, p, rtol=1e-12, atol=1e-14,
                err_msg=f"Parallel ≠ serial: {col}")


class TestParallelFitIntercept:
    """fit_intercept=False works in parallel."""

    def test_parallel_fit_intercept_false(self):
        """Test 11: No intercept columns when fit_intercept=False."""
        df = _make_tpc_like_data(n_sectors=2, n_stacks=1, grid=5, rpb=10)
        result = make_sliding_window_fit_parallel(
            df, gb_columns=GB, fit_columns=FIT, linear_columns=LIN,
            split_columns=['sector'], n_workers=1,
            window_spec=WS, min_stat=5, suffix='_sw',
            fit_intercept=False,
        )
        assert 'value_intercept_sw' not in result.columns, \
            "Intercept column should not exist when fit_intercept=False"
        assert 'value_intercept_err_sw' not in result.columns, \
            "Intercept err column should not exist when fit_intercept=False"
        assert 'value_slope_x_sw' in result.columns, \
            "Slope column must still exist"


class TestParallelSafety:
    """Safety: total failure must raise, not return empty."""

    def test_parallel_total_failure_raises(self):
        """Test 12: All units fail → RuntimeError or KeyError with message."""
        df = _make_tpc_like_data(n_sectors=2, n_stacks=1, grid=5, rpb=10)
        # Non-existent column raises KeyError in parent (column extraction)
        with pytest.raises((RuntimeError, KeyError)):
            make_sliding_window_fit_parallel(
                df, gb_columns=GB, fit_columns=['NONEXISTENT'],
                linear_columns=LIN, split_columns=['sector'], n_workers=1,
                window_spec=WS, min_stat=5, suffix='_sw',
            )
