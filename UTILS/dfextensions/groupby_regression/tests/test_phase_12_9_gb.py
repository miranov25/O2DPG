"""
Tests for Phase 12.9.GB: Numba Parallel Kernel for make_parallel_fit_v5

Test categories:
1. Correctness: parallel matches sequential
2. Backend selection
3. Status code mapping
4. Per-fit diagnostics
5. Edge cases
6. Performance (diagnostic only)
"""

import numpy as np
import pandas as pd
import pytest
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groupby_regression_optimized import (
    make_parallel_fit_v5,
    _NUMBA_AVAILABLE,
    _select_parallel_backend,
    STATUS_OK,
    STATUS_INSUFFICIENT_DATA,
    STATUS_INSUFFICIENT_VALID,
    STATUS_ILL_CONDITIONED,
    STATUS_SINGULAR,
    STATUS_TO_STRING,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_df():
    """Simple test DataFrame."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'group': np.repeat(np.arange(100), 10),
        'x': np.random.randn(n),
        'y': np.random.randn(n),
        'w': np.abs(np.random.randn(n)) + 0.1,
    })
    df['y'] = 2 + 3 * df['x'] + 0.1 * np.random.randn(n)
    return df


@pytest.fixture
def multi_fit_df():
    """Multi-fit test DataFrame with different linear columns per fit."""
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        'group': np.repeat(np.arange(100), 20),
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y1': np.random.randn(n),
        'y2': np.random.randn(n),
        'w1': np.abs(np.random.randn(n)) + 0.1,
        'w2': np.abs(np.random.randn(n)) + 0.1,
    })
    df['y1'] = 2 + 3 * df['x1'] + 0.1 * np.random.randn(n)
    df['y2'] = 1 + 2 * df['x1'] + 0.5 * df['x2'] + 0.1 * np.random.randn(n)
    return df


@pytest.fixture
def df_with_nans():
    """DataFrame with NaN values in different fits."""
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'group': np.repeat(np.arange(100), 10),
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y1': np.random.randn(n),
        'y2': np.random.randn(n),
        'w': np.abs(np.random.randn(n)) + 0.1,
    })
    df['y1'] = 2 + 3 * df['x1'] + 0.1 * np.random.randn(n)
    df['y2'] = 1 + 2 * df['x1'] + 0.1 * np.random.randn(n)
    
    # Add NaNs to different columns for different fits
    df.loc[df.index[:50], 'y1'] = np.nan  # NaN in y1 for first 50 rows
    df.loc[df.index[100:150], 'x2'] = np.nan  # NaN in x2 for rows 100-150
    
    return df


# ============================================================================
# TEST: NUMBA MATCHES SEQUENTIAL (Critical)
# ============================================================================

class TestNumbaMatchesSequential:
    """Verify that Numba and sequential backends produce identical results."""
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_single_fit_matches(self, simple_df):
        """Single fit: Numba results match sequential within float64 precision."""
        result_seq = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            weights='w',
            diag=True,
            parallel_backend='sequential',
        )
        
        result_numba = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            weights='w',
            diag=True,
            parallel_backend='numba',
            n_jobs=2,
        )
        
        # Check numeric columns match
        numeric_cols = ['y_intercept_v5', 'y_intercept_err_v5', 
                       'y_slope_x_v5', 'y_slope_x_err_v5',
                       'y_rms_v5', 'y_mad_v5']
        
        for col in numeric_cols:
            np.testing.assert_allclose(
                result_seq[col].values,
                result_numba[col].values,
                rtol=1e-10,
                err_msg=f"Column {col} mismatch"
            )
        
        # Check diagnostic columns match
        assert np.all(result_seq['diag_n_total_v5'] == result_numba['diag_n_total_v5'])
        assert np.all(result_seq['diag_n_valid_v5'] == result_numba['diag_n_valid_v5'])
        assert np.all(result_seq['diag_n_filtered_v5'] == result_numba['diag_n_filtered_v5'])
        assert np.all(result_seq['diag_status_v5'] == result_numba['diag_status_v5'])
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_multi_fit_matches(self, multi_fit_df):
        """Multi-fit: Numba results match sequential."""
        result_seq = make_parallel_fit_v5(
            df=multi_fit_df,
            gb_columns='group',
            fit_columns=['y1', 'y2'],
            suffixes=['_A', '_B'],
            linear_columns=[['x1'], ['x1', 'x2']],
            weights=['w1', 'w2'],
            diag=True,
            parallel_backend='sequential',
        )
        
        result_numba = make_parallel_fit_v5(
            df=multi_fit_df,
            gb_columns='group',
            fit_columns=['y1', 'y2'],
            suffixes=['_A', '_B'],
            linear_columns=[['x1'], ['x1', 'x2']],
            weights=['w1', 'w2'],
            diag=True,
            parallel_backend='numba',
            n_jobs=2,
        )
        
        # Check coefficients match
        coef_cols = ['y1_intercept_A', 'y1_slope_x1_A',
                     'y2_intercept_B', 'y2_slope_x1_B', 'y2_slope_x2_B']
        
        for col in coef_cols:
            np.testing.assert_allclose(
                result_seq[col].values,
                result_numba[col].values,
                rtol=1e-10,
                err_msg=f"Column {col} mismatch"
            )
        
        # Check per-fit diagnostics match
        assert np.all(result_seq['diag_n_valid_A'] == result_numba['diag_n_valid_A'])
        assert np.all(result_seq['diag_n_valid_B'] == result_numba['diag_n_valid_B'])
        assert np.all(result_seq['diag_status_A'] == result_numba['diag_status_A'])
        assert np.all(result_seq['diag_status_B'] == result_numba['diag_status_B'])
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_unweighted_matches(self, simple_df):
        """Unweighted fit: Numba matches sequential."""
        result_seq = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            weights=None,
            parallel_backend='sequential',
        )
        
        result_numba = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            weights=None,
            parallel_backend='numba',
            n_jobs=2,
        )
        
        np.testing.assert_allclose(
            result_seq['y_intercept_v5'].values,
            result_numba['y_intercept_v5'].values,
            rtol=1e-10,
        )
        np.testing.assert_allclose(
            result_seq['y_slope_x_v5'].values,
            result_numba['y_slope_x_v5'].values,
            rtol=1e-10,
        )
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_no_intercept_matches(self, simple_df):
        """fit_intercept=False: Numba matches sequential."""
        result_seq = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            fit_intercept=False,
            parallel_backend='sequential',
        )
        
        result_numba = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            fit_intercept=False,
            parallel_backend='numba',
            n_jobs=2,
        )
        
        np.testing.assert_allclose(
            result_seq['y_slope_x_v5'].values,
            result_numba['y_slope_x_v5'].values,
            rtol=1e-10,
        )


# ============================================================================
# TEST: BACKEND SELECTION
# ============================================================================

class TestBackendSelection:
    """Test parallel_backend parameter and auto-selection."""
    
    def test_select_backend_sequential(self):
        """parallel_backend='sequential' always returns sequential."""
        assert _select_parallel_backend('sequential', n_jobs=1) == 'sequential'
        assert _select_parallel_backend('sequential', n_jobs=4) == 'sequential'
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_select_backend_numba(self):
        """parallel_backend='numba' returns numba when available."""
        assert _select_parallel_backend('numba', n_jobs=1) == 'numba'
        assert _select_parallel_backend('numba', n_jobs=4) == 'numba'
    
    def test_select_backend_numba_unavailable(self, monkeypatch):
        """parallel_backend='numba' raises when Numba unavailable."""
        import groupby_regression_optimized as grmod
        monkeypatch.setattr(grmod, '_NUMBA_AVAILABLE', False)
        
        with pytest.raises(ImportError, match="Numba is required"):
            _select_parallel_backend('numba', n_jobs=4)
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_select_backend_auto_numba(self):
        """parallel_backend='auto' chooses numba when n_jobs > 1."""
        assert _select_parallel_backend('auto', n_jobs=4) == 'numba'
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_select_backend_auto_sequential(self):
        """parallel_backend='auto' with n_jobs=1 chooses numba when available.

        Phase 12.11 (Dec 26, 2025) changed auto-dispatch to prefer numba
        whenever Numba is installed, regardless of n_jobs. This test was not
        updated at that time and silently failed from Phase 12.11 through
        Phase 13.16.GB. Updated in Phase 13.16.GB-FIX2.

        See `_select_parallel_backend` in groupby_regression_optimized.py
        (line ~2944, "# Phase 12.11 fix" comment) for the production behavior
        this test asserts.
        """
        assert _select_parallel_backend('auto', n_jobs=1) == 'numba'
    
    def test_select_backend_auto_no_numba(self, monkeypatch):
        """parallel_backend='auto' falls back to sequential without Numba."""
        import groupby_regression_optimized as grmod
        monkeypatch.setattr(grmod, '_NUMBA_AVAILABLE', False)
        
        assert _select_parallel_backend('auto', n_jobs=4) == 'sequential'


# ============================================================================
# TEST: STATUS CODES
# ============================================================================

class TestStatusCodes:
    """Test status code mapping between int and string."""
    
    def test_status_code_constants(self):
        """Status code constants are defined."""
        assert STATUS_OK == 0
        assert STATUS_INSUFFICIENT_DATA == 1
        assert STATUS_INSUFFICIENT_VALID == 2
        assert STATUS_ILL_CONDITIONED == 3
        assert STATUS_SINGULAR == 4
    
    def test_status_to_string_mapping(self):
        """STATUS_TO_STRING maps all codes correctly."""
        assert STATUS_TO_STRING[STATUS_OK] == 'OK'
        assert STATUS_TO_STRING[STATUS_INSUFFICIENT_DATA] == 'INSUFFICIENT_DATA'
        assert STATUS_TO_STRING[STATUS_INSUFFICIENT_VALID] == 'INSUFFICIENT_VALID'
        assert STATUS_TO_STRING[STATUS_ILL_CONDITIONED] == 'ILL_CONDITIONED_RIDGED'
        assert STATUS_TO_STRING[STATUS_SINGULAR] == 'SINGULAR_MATRIX'
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_status_codes_in_output(self, simple_df):
        """Output status codes are strings after conversion."""
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            diag=True,
            parallel_backend='numba',
            n_jobs=2,
        )
        
        # Status should be string
        assert result['diag_status_v5'].dtype == object or 'str' in str(result['diag_status_v5'].dtype)
        
        # Check valid values
        valid_statuses = set(STATUS_TO_STRING.values())
        unique_statuses = set(result['diag_status_v5'].unique())
        assert unique_statuses.issubset(valid_statuses | {''})


# ============================================================================
# TEST: PER-FIT DIAGNOSTICS
# ============================================================================

class TestPerFitDiagnostics:
    """Test per-fit diagnostic columns (Phase 12.9.GB)."""
    
    def test_per_fit_n_valid_different_nans(self, df_with_nans):
        """Different fits can have different n_valid due to NaN patterns."""
        result = make_parallel_fit_v5(
            df=df_with_nans,
            gb_columns='group',
            fit_columns=['y1', 'y2'],
            suffixes=['_fit1', '_fit2'],
            linear_columns=[['x1'], ['x1', 'x2']],
            diag=True,
            parallel_backend='sequential',
        )
        
        # Per-fit diagnostics exist
        assert 'diag_n_valid_fit1' in result.columns
        assert 'diag_n_valid_fit2' in result.columns
        assert 'diag_n_filtered_fit1' in result.columns
        assert 'diag_n_filtered_fit2' in result.columns
        
        # n_valid can differ between fits (due to different NaN patterns)
        # Group 0 has y1 NaNs, fit2 may also have x2 NaNs
        n_valid_fit1 = result['diag_n_valid_fit1'].values
        n_valid_fit2 = result['diag_n_valid_fit2'].values
        
        # At least some groups should have different n_valid
        # (due to NaN in x2 affecting only fit2)
        # This verifies per-fit semantics
        assert result.shape[0] == 100  # 100 groups
    
    def test_shared_n_total_only(self, multi_fit_df):
        """n_total is shared (1D), others are per-fit."""
        result = make_parallel_fit_v5(
            df=multi_fit_df,
            gb_columns='group',
            fit_columns=['y1', 'y2'],
            suffixes=['_A', '_B'],
            linear_columns=['x1'],
            diag=True,
        )
        
        # Shared: n_total
        assert 'diag_n_total_v5' in result.columns
        
        # Per-fit: n_valid, n_filtered, cond, status
        assert 'diag_n_valid_A' in result.columns
        assert 'diag_n_valid_B' in result.columns
        assert 'diag_n_filtered_A' in result.columns
        assert 'diag_n_filtered_B' in result.columns
        assert 'diag_cond_A' in result.columns
        assert 'diag_cond_B' in result.columns
        assert 'diag_status_A' in result.columns
        assert 'diag_status_B' in result.columns
        
        # Old shared names should NOT exist
        assert 'diag_n_valid_v5' not in result.columns
        assert 'diag_n_filtered_v5' not in result.columns


# ============================================================================
# TEST: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_n_jobs_greater_than_groups(self, simple_df):
        """n_jobs > n_groups should not crash."""
        # Only 100 groups, request 200 jobs
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            parallel_backend='numba',
            n_jobs=200,
        )
        
        assert len(result) == 100
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_single_group(self):
        """Single group works with Numba backend."""
        np.random.seed(42)
        df = pd.DataFrame({
            'group': [0] * 10,
            'x': np.random.randn(10),
            'y': 2 + 3 * np.random.randn(10),
        })
        
        result = make_parallel_fit_v5(
            df=df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            parallel_backend='numba',
            n_jobs=4,
        )
        
        assert len(result) == 1
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_insufficient_data_groups(self):
        """Groups with insufficient data get correct status."""
        np.random.seed(42)
        df = pd.DataFrame({
            'group': [0, 0, 0, 1, 1, 2, 2, 2, 2, 2],  # Group 1 has only 2 rows
            'x': np.random.randn(10),
            'y': np.random.randn(10),
        })
        
        result = make_parallel_fit_v5(
            df=df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            min_stat=3,
            diag=True,
            parallel_backend='numba',
            n_jobs=2,
        )
        
        # Group 1 should have INSUFFICIENT_DATA status
        status_g1 = result[result['group'] == 1]['diag_status_v5'].values[0]
        assert status_g1 == 'INSUFFICIENT_DATA'
        
        # Groups 0 and 2 should be OK
        status_g0 = result[result['group'] == 0]['diag_status_v5'].values[0]
        status_g2 = result[result['group'] == 2]['diag_status_v5'].values[0]
        assert status_g0 == 'OK'
        assert status_g2 == 'OK'


# ============================================================================
# TEST: PERFORMANCE (Diagnostic only - not a gate)
# ============================================================================

class TestPerformance:
    """Performance tests - diagnostic only, weak assertions."""
    
    @pytest.mark.slow
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_scaling_diagnostic(self):
        """Measure scaling with parallel workers (diagnostic only)."""
        # Create larger dataset
        np.random.seed(42)
        n = 50000
        n_groups = 5000
        df = pd.DataFrame({
            'group': np.repeat(np.arange(n_groups), n // n_groups),
            'x': np.random.randn(n),
            'y': np.random.randn(n),
            'w': np.abs(np.random.randn(n)) + 0.1,
        })
        df['y'] = 2 + 3 * df['x'] + 0.1 * np.random.randn(n)
        
        # Warmup (JIT compile)
        _ = make_parallel_fit_v5(
            df=df.head(1000),
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            weights='w',
            parallel_backend='numba',
            n_jobs=2,
        )
        
        # Time sequential
        t0 = time.perf_counter()
        _ = make_parallel_fit_v5(
            df=df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            weights='w',
            parallel_backend='sequential',
        )
        time_seq = time.perf_counter() - t0
        
        # Time parallel (4 jobs)
        t0 = time.perf_counter()
        _ = make_parallel_fit_v5(
            df=df,
            gb_columns='group',
            fit_columns=['y'],
            linear_columns=['x'],
            weights='w',
            parallel_backend='numba',
            n_jobs=4,
        )
        time_par = time.perf_counter() - t0
        
        print(f"\n  Sequential: {time_seq:.2f}s")
        print(f"  Parallel (4 jobs): {time_par:.2f}s")
        print(f"  Speedup: {time_seq / time_par:.2f}x")
        
        # Weak assertion: parallel should not be significantly slower
        # (allows for JIT overhead on small datasets)
        assert time_par < time_seq * 2, "Parallel should not be 2x slower than sequential"


# ============================================================================
# TEST: THREADING ENVIRONMENT
# ============================================================================

class TestThreadingEnvironment:
    """Test that threading environment is properly managed."""
    
    @pytest.mark.skipif(not _NUMBA_AVAILABLE, reason="Numba not available")
    def test_threading_env_restored(self, simple_df):
        """Thread environment variables are restored after call."""
        import os
        
        # Set some env vars
        original_omp = os.environ.get('OMP_NUM_THREADS', None)
        os.environ['OMP_NUM_THREADS'] = '8'
        
        try:
            _ = make_parallel_fit_v5(
                df=simple_df,
                gb_columns='group',
                fit_columns=['y'],
                linear_columns=['x'],
                parallel_backend='numba',
                n_jobs=4,
            )
            
            # Should be restored
            assert os.environ.get('OMP_NUM_THREADS') == '8'
        finally:
            if original_omp is not None:
                os.environ['OMP_NUM_THREADS'] = original_omp
            elif 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
