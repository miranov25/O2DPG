"""
Tests for Phase 12.8.GB: make_parallel_fit_v5 batch fitting.

Tests cover:
1. Basic functionality - single and multiple fits
2. Precision - v5 matches v4 within float64 roundoff
3. Chunk invariance - different n_chunks produce identical results
4. Parameter validation - error handling
5. Edge cases - empty data, insufficient points, etc.
6. Metadata generation
7. Diagnostics
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from groupby_regression_optimized import (
    make_parallel_fit_v4,
    make_parallel_fit_v5,
    _compute_sort_indices_v5,
    _compute_group_boundaries_v5,
    _compute_chunk_boundaries_v5,
    _validate_v5_params,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simple_df():
    """Simple test DataFrame with 4 groups."""
    np.random.seed(42)
    n_per_group = 50
    n_groups = 4
    n = n_per_group * n_groups
    
    return pd.DataFrame({
        'g': np.repeat(np.arange(n_groups), n_per_group),
        'x1': np.random.randn(n),
        'x2': np.random.randn(n),
        'y1': np.random.randn(n),
        'y2': np.random.randn(n),
        'w': np.abs(np.random.randn(n)) + 0.1,
    })


@pytest.fixture
def multi_key_df():
    """DataFrame with multi-key groupby (like track_index, firstTForbit)."""
    np.random.seed(123)
    n = 1000
    
    # Create multi-key groups
    track_index = np.random.randint(0, 20, n)
    orbit = np.random.randint(0, 10, n)
    
    return pd.DataFrame({
        'track_index': track_index,
        'firstTForbit': orbit,
        'rrel': np.random.randn(n),
        'rrel2': np.random.randn(n),
        'dyC2': np.random.randn(n),
        'dzC2': np.random.randn(n),
        'weightTPCR': np.abs(np.random.randn(n)) + 0.1,
        'weightITSR': np.abs(np.random.randn(n)) + 0.1,
    })


@pytest.fixture
def calibration_df():
    """Larger DataFrame simulating calibration workflow."""
    np.random.seed(456)
    n = 10000
    n_tracks = 100
    n_orbits = 20
    
    track_index = np.random.randint(0, n_tracks, n)
    orbit = np.random.randint(0, n_orbits, n)
    rrel = np.random.uniform(-1, 1, n)
    
    # Generate correlated targets
    dy = 0.1 + 0.5 * rrel + 0.2 * rrel**2 + 0.05 * np.random.randn(n)
    dz = 0.2 + 0.3 * rrel + 0.03 * np.random.randn(n)
    
    return pd.DataFrame({
        'track_index': track_index,
        'firstTForbit': orbit,
        'rrel': rrel,
        'rrel2': rrel**2,
        'dyC2': dy,
        'dzC2': dz,
        'weightTPCR': np.abs(np.random.randn(n)) + 0.5,
        'weightITSR': np.abs(np.random.randn(n)) + 0.3,
    })


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

class TestV5BasicFunctionality:
    """Basic functionality tests for make_parallel_fit_v5."""
    
    def test_single_fit_returns_dataframe(self, simple_df):
        """v5 with single fit returns DataFrame."""
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1'],
            linear_columns=['x1'],
            suffixes='_v5',
        )
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 4 groups
        assert 'g' in result.columns
        assert 'y1_intercept_v5' in result.columns
        assert 'y1_slope_x1_v5' in result.columns
        assert 'y1_rms_v5' in result.columns
    
    def test_multiple_fits_same_target(self, simple_df):
        """v5 with same target, different suffixes."""
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1', 'y1'],
            suffixes=['_A', '_B'],
            linear_columns=[['x1'], ['x1', 'x2']],
        )
        
        assert 'y1_intercept_A' in result.columns
        assert 'y1_intercept_B' in result.columns
        assert 'y1_slope_x1_A' in result.columns
        assert 'y1_slope_x2_B' in result.columns
    
    def test_different_targets(self, simple_df):
        """v5 with different targets, shared suffix."""
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1', 'y2'],
            suffixes='_Fit',
            linear_columns=['x1', 'x2'],
        )
        
        assert 'y1_intercept_Fit' in result.columns
        assert 'y2_intercept_Fit' in result.columns
    
    def test_different_weights_per_fit(self, multi_key_df):
        """v5 with different weights per fit."""
        result = make_parallel_fit_v5(
            df=multi_key_df,
            gb_columns=['track_index', 'firstTForbit'],
            fit_columns=['dzC2', 'dzC2'],
            suffixes=['_TPC', '_ITS'],
            linear_columns=['rrel'],
            weights=['weightTPCR', 'weightITSR'],
        )
        
        assert 'dzC2_intercept_TPC' in result.columns
        assert 'dzC2_intercept_ITS' in result.columns
        # Coefficients should differ due to different weights
        # (not testing exact values, just that they exist)
    
    def test_different_linear_columns_per_fit(self, multi_key_df):
        """v5 with different linear columns per fit."""
        result = make_parallel_fit_v5(
            df=multi_key_df,
            gb_columns=['track_index', 'firstTForbit'],
            fit_columns=['dyC2', 'dzC2'],
            suffixes=['_Y', '_Z'],
            linear_columns=[['rrel', 'rrel2'], ['rrel']],
        )
        
        # _Y fit should have rrel2 slope
        assert 'dyC2_slope_rrel2_Y' in result.columns
        # _Z fit should NOT have rrel2 slope
        assert 'dzC2_slope_rrel2_Z' not in result.columns
    
    def test_no_intercept(self, simple_df):
        """v5 with fit_intercept=False."""
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1'],
            linear_columns=['x1'],
            suffixes='_v5',
            fit_intercept=False,
        )
        
        assert 'y1_intercept_v5' not in result.columns
        assert 'y1_slope_x1_v5' in result.columns


# ============================================================================
# PRECISION TESTS - V5 MATCHES V4
# ============================================================================

class TestV5MatchesV4:
    """Test that v5 results match v4 within float64 roundoff."""
    
    def test_single_fit_matches_v4(self, simple_df):
        """Single fit in v5 matches v4 within float64 roundoff."""
        # v4 result
        _, dfGB_v4 = make_parallel_fit_v4(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1'],
            linear_columns=['x1'],
            weights='w',
            suffix='_v4',
        )
        
        # v5 result
        dfGB_v5 = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1'],
            linear_columns=['x1'],
            weights=['w'],
            suffixes='_v5',
        )
        
        # Match group keys
        assert list(dfGB_v4['g']) == list(dfGB_v5['g'])
        
        # Match coefficients within float64 roundoff
        np.testing.assert_allclose(
            dfGB_v4['y1_intercept_v4'].values,
            dfGB_v5['y1_intercept_v5'].values,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            dfGB_v4['y1_slope_x1_v4'].values,
            dfGB_v5['y1_slope_x1_v5'].values,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            dfGB_v4['y1_rms_v4'].values,
            dfGB_v5['y1_rms_v5'].values,
            rtol=1e-12,
            atol=1e-12,
        )
    
    def test_multiple_fits_match_v4_merged(self, multi_key_df):
        """Multiple v5 fits match v4 called separately."""
        gb = ['track_index', 'firstTForbit']
        
        # v4 calls
        _, dfGB_v4_Y = make_parallel_fit_v4(
            df=multi_key_df, gb_columns=gb,
            fit_columns=['dyC2'], linear_columns=['rrel', 'rrel2'],
            weights='weightTPCR', suffix='_Y',
        )
        _, dfGB_v4_Z = make_parallel_fit_v4(
            df=multi_key_df, gb_columns=gb,
            fit_columns=['dzC2'], linear_columns=['rrel'],
            weights='weightTPCR', suffix='_Z',
        )
        
        # Merge v4 results
        dfGB_v4 = dfGB_v4_Y.merge(dfGB_v4_Z, on=gb)
        
        # v5 single call
        dfGB_v5 = make_parallel_fit_v5(
            df=multi_key_df,
            gb_columns=gb,
            fit_columns=['dyC2', 'dzC2'],
            suffixes=['_Y', '_Z'],
            linear_columns=[['rrel', 'rrel2'], ['rrel']],
            weights=['weightTPCR', 'weightTPCR'],
        )
        
        # Sort both for comparison
        dfGB_v4 = dfGB_v4.sort_values(gb).reset_index(drop=True)
        dfGB_v5 = dfGB_v5.sort_values(gb).reset_index(drop=True)
        
        # Compare coefficients
        np.testing.assert_allclose(
            dfGB_v4['dyC2_intercept_Y'].values,
            dfGB_v5['dyC2_intercept_Y'].values,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            dfGB_v4['dzC2_slope_rrel_Z'].values,
            dfGB_v5['dzC2_slope_rrel_Z'].values,
            rtol=1e-12,
            atol=1e-12,
        )
    
    def test_precision_float32_input(self):
        """Results stable with float32 input."""
        np.random.seed(789)
        n = 500
        
        df_f32 = pd.DataFrame({
            'g': np.repeat(np.arange(10), 50),
            'x': np.random.randn(n).astype(np.float32),
            'y': np.random.randn(n).astype(np.float32),
            'w': (np.abs(np.random.randn(n)) + 0.1).astype(np.float32),
        })
        
        result = make_parallel_fit_v5(
            df=df_f32,
            gb_columns='g',
            fit_columns=['y'],
            linear_columns=['x'],
            weights=['w'],
            suffixes='_v5',
        )
        
        # Should complete without error
        assert len(result) == 10
        # Results should be finite
        assert np.all(np.isfinite(result['y_intercept_v5']))


# ============================================================================
# CHUNK INVARIANCE TESTS (Critical - R2 fix validation)
# ============================================================================

class TestChunkInvariance:
    """Test that different n_chunks produce identical results."""
    
    def test_chunks_1_vs_4_identical(self, calibration_df):
        """n_chunks=1 and n_chunks=4 produce identical results."""
        result_1 = make_parallel_fit_v5(
            df=calibration_df,
            gb_columns=['track_index', 'firstTForbit'],
            fit_columns=['dyC2', 'dzC2'],
            suffixes=['_Y', '_Z'],
            linear_columns=[['rrel', 'rrel2'], ['rrel']],
            n_chunks=1,
        )
        
        result_4 = make_parallel_fit_v5(
            df=calibration_df,
            gb_columns=['track_index', 'firstTForbit'],
            fit_columns=['dyC2', 'dzC2'],
            suffixes=['_Y', '_Z'],
            linear_columns=[['rrel', 'rrel2'], ['rrel']],
            n_chunks=4,
        )
        
        # Sort for comparison
        gb = ['track_index', 'firstTForbit']
        result_1 = result_1.sort_values(gb).reset_index(drop=True)
        result_4 = result_4.sort_values(gb).reset_index(drop=True)
        
        # All numeric columns should match exactly
        for col in result_1.columns:
            if result_1[col].dtype in [np.float64, np.float32]:
                np.testing.assert_allclose(
                    result_1[col].values,
                    result_4[col].values,
                    rtol=1e-14,
                    atol=1e-14,
                    err_msg=f"Mismatch in column {col}",
                )
    
    def test_chunks_1_vs_8_identical(self, calibration_df):
        """n_chunks=1 and n_chunks=8 produce identical results."""
        result_1 = make_parallel_fit_v5(
            df=calibration_df,
            gb_columns=['track_index', 'firstTForbit'],
            fit_columns=['dyC2'],
            suffixes='_Fit',
            linear_columns=['rrel'],
            n_chunks=1,
        )
        
        result_8 = make_parallel_fit_v5(
            df=calibration_df,
            gb_columns=['track_index', 'firstTForbit'],
            fit_columns=['dyC2'],
            suffixes='_Fit',
            linear_columns=['rrel'],
            n_chunks=8,
        )
        
        gb = ['track_index', 'firstTForbit']
        result_1 = result_1.sort_values(gb).reset_index(drop=True)
        result_8 = result_8.sort_values(gb).reset_index(drop=True)
        
        pd.testing.assert_frame_equal(result_1, result_8)
    
    def test_chunk_boundaries_never_split_groups(self):
        """Verify chunk boundaries align with group boundaries."""
        # Create data with known groups
        df = pd.DataFrame({
            'g': [0, 0, 0, 1, 1, 2, 2, 2, 2, 3],
            'x': np.random.randn(10),
        })
        
        perm = _compute_sort_indices_v5(df, ['g'])
        offsets = _compute_group_boundaries_v5(df, ['g'], perm)
        
        # offsets should be [0, 3, 5, 9, 10]
        assert list(offsets) == [0, 3, 5, 9, 10]
        
        # With 2 chunks, should split between groups
        chunks = _compute_chunk_boundaries_v5(n_groups=4, n_chunks=2, offsets=offsets)
        
        # Chunk 0: groups 0-1 (rows 0-5)
        assert chunks[0][0] == 0  # g_start
        assert chunks[0][1] == 2  # g_end
        assert chunks[0][2] == 0  # row_start
        assert chunks[0][3] == 5  # row_end
        
        # Chunk 1: groups 2-3 (rows 5-10)
        assert chunks[1][0] == 2  # g_start
        assert chunks[1][1] == 4  # g_end
        assert chunks[1][2] == 5  # row_start
        assert chunks[1][3] == 10  # row_end


# ============================================================================
# PARAMETER VALIDATION TESTS
# ============================================================================

class TestV5Validation:
    """Test parameter validation and error handling."""
    
    def test_shared_suffix_duplicate_targets_raises(self):
        """Shared suffix + duplicate targets raises ValueError."""
        with pytest.raises(ValueError, match="Shared suffix.*not allowed"):
            _validate_v5_params(
                fit_columns=['y', 'y'],  # Duplicate targets
                suffixes='_Fit',  # Shared suffix
                linear_columns=['x'],
                weights=None,
            )
    
    def test_duplicate_output_columns_raises(self):
        """Duplicate (target, suffix) pairs raises ValueError."""
        # Different targets with same suffix is OK
        suffixes, _, _ = _validate_v5_params(
            fit_columns=['y1', 'y2'],
            suffixes=['_A', '_A'],  # Same suffix but different targets - OK
            linear_columns=['x'],
            weights=None,
        )
        assert suffixes == ['_A', '_A']
        
        # Same target with same suffix should raise
        with pytest.raises(ValueError, match="Duplicate"):
            _validate_v5_params(
                fit_columns=['y', 'y'],
                suffixes=['_A', '_A'],  # Same (target, suffix) pair - ERROR
                linear_columns=['x'],
                weights=None,
            )
    
    def test_mismatched_lengths_raises(self, simple_df):
        """Mismatched parameter lengths raises ValueError."""
        with pytest.raises(ValueError, match="length"):
            make_parallel_fit_v5(
                df=simple_df,
                gb_columns='g',
                fit_columns=['y1', 'y2'],
                suffixes=['_A', '_B', '_C'],  # 3 suffixes for 2 fits
                linear_columns=['x1'],
            )
    
    def test_empty_fit_columns_raises(self, simple_df):
        """Empty fit_columns raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            make_parallel_fit_v5(
                df=simple_df,
                gb_columns='g',
                fit_columns=[],
                linear_columns=['x1'],
            )
    
    def test_missing_columns_raises(self, simple_df):
        """Missing required columns raises KeyError."""
        with pytest.raises(KeyError, match="Missing"):
            make_parallel_fit_v5(
                df=simple_df,
                gb_columns='g',
                fit_columns=['nonexistent'],
                linear_columns=['x1'],
            )
    
    def test_null_in_gb_columns_raises(self):
        """Null values in gb_columns raises ValueError."""
        df = pd.DataFrame({
            'g': [1, 2, None, 4],
            'x': [1, 2, 3, 4],
            'y': [1, 2, 3, 4],
        })
        
        with pytest.raises(ValueError, match="null"):
            make_parallel_fit_v5(
                df=df,
                gb_columns='g',
                fit_columns=['y'],
                linear_columns=['x'],
            )


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestV5EdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_dataframe(self):
        """Empty DataFrame returns empty result."""
        df = pd.DataFrame({
            'g': [],
            'x': [],
            'y': [],
        })
        
        result = make_parallel_fit_v5(
            df=df,
            gb_columns='g',
            fit_columns=['y'],
            linear_columns=['x'],
        )
        
        assert len(result) == 0
    
    def test_single_group(self):
        """Single group works correctly."""
        df = pd.DataFrame({
            'g': [1, 1, 1, 1, 1],
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10],
        })
        
        result = make_parallel_fit_v5(
            df=df,
            gb_columns='g',
            fit_columns=['y'],
            linear_columns=['x'],
        )
        
        assert len(result) == 1
        # Perfect linear relationship
        np.testing.assert_allclose(result['y_slope_x_v5'].values[0], 2.0, rtol=1e-10)
    
    def test_insufficient_data_group(self):
        """Groups with insufficient data get NaN results."""
        df = pd.DataFrame({
            'g': [1, 1, 1, 1, 1, 2, 2],  # Group 2 has only 2 points
            'x': np.random.randn(7),
            'y': np.random.randn(7),
        })
        
        result = make_parallel_fit_v5(
            df=df,
            gb_columns='g',
            fit_columns=['y'],
            linear_columns=['x'],
            min_stat=3,
        )
        
        # Group 1 should have results
        assert not np.isnan(result[result['g'] == 1]['y_intercept_v5'].values[0])
        # Group 2 should have NaN
        assert np.isnan(result[result['g'] == 2]['y_intercept_v5'].values[0])
    
    def test_selection_parameter(self, simple_df):
        """Selection parameter filters rows correctly."""
        selection = simple_df['g'] <= 1  # Keep only groups 0 and 1
        
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1'],
            linear_columns=['x1'],
            selection=selection,
        )
        
        assert len(result) == 2
        assert set(result['g']) == {0, 1}
    
    def test_compute_mad_false(self, simple_df):
        """compute_mad=False skips MAD calculation."""
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1'],
            linear_columns=['x1'],
            compute_mad=False,
        )
        
        # MAD column should be all NaN
        assert np.all(np.isnan(result['y1_mad_v5']))


# ============================================================================
# METADATA TESTS
# ============================================================================

class TestV5Metadata:
    """Test metadata generation."""
    
    def test_return_metadata(self, simple_df):
        """return_metadata=True returns tuple with metadata."""
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1'],
            linear_columns=['x1'],
            return_metadata=True,
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        
        dfGB, metadata = result
        assert isinstance(dfGB, pd.DataFrame)
        assert isinstance(metadata, dict)
    
    def test_metadata_structure(self, simple_df):
        """Metadata has correct structure."""
        dfGB, metadata = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1', 'y2'],
            suffixes=['_A', '_B'],
            linear_columns=['x1'],
            return_metadata=True,
        )
        
        assert metadata['version'] == 'v5'
        assert metadata['n_fits'] == 2
        assert '_A' in metadata['fits']
        assert '_B' in metadata['fits']
    
    def test_metadata_formulas(self, simple_df):
        """Metadata contains correct formulas."""
        dfGB, metadata = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1'],
            suffixes='_Fit',
            linear_columns=['x1', 'x2'],
            return_metadata=True,
        )
        
        fit_meta = metadata['fits']['_Fit']
        
        # Prediction formula
        pred = fit_meta['formulas']['prediction']
        assert 'y1_intercept_Fit' in pred
        assert 'y1_slope_x1_Fit' in pred
        assert 'y1_slope_x2_Fit' in pred
        
        # Residual formula
        resid = fit_meta['formulas']['residual']
        assert 'y1 -' in resid


# ============================================================================
# DIAGNOSTICS TESTS
# ============================================================================

class TestV5Diagnostics:
    """Test diagnostic output."""
    
    def test_diag_true_adds_columns(self, simple_df):
        """diag=True adds diagnostic columns."""
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1'],
            linear_columns=['x1'],
            diag=True,
        )
        
        # Shared diagnostics
        assert 'diag_n_total_v5' in result.columns
        assert 'diag_n_valid_v5' in result.columns
        assert 'diag_n_filtered_v5' in result.columns
        
        # Per-fit diagnostics
        assert 'diag_cond_v5' in result.columns
        assert 'diag_status_v5' in result.columns
    
    def test_per_fit_diagnostics(self, simple_df):
        """Per-fit diagnostics have correct suffixes."""
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1', 'y2'],
            suffixes=['_A', '_B'],
            linear_columns=['x1'],
            diag=True,
        )
        
        assert 'diag_cond_A' in result.columns
        assert 'diag_cond_B' in result.columns
        assert 'diag_status_A' in result.columns
        assert 'diag_status_B' in result.columns
    
    def test_diag_n_total_correct(self, simple_df):
        """diag_n_total shows correct group sizes."""
        result = make_parallel_fit_v5(
            df=simple_df,
            gb_columns='g',
            fit_columns=['y1'],
            linear_columns=['x1'],
            diag=True,
        )
        
        # Each group has 50 points
        assert all(result['diag_n_total_v5'] == 50)


# ============================================================================
# MULTI-KEY SORT TESTS
# ============================================================================

class TestMultiKeySort:
    """Test multi-key sorting functionality."""
    
    def test_sort_indices_single_key(self):
        """Sort indices for single key."""
        df = pd.DataFrame({'g': [3, 1, 2, 1, 3]})
        perm = _compute_sort_indices_v5(df, 'g')
        
        sorted_g = df['g'].values[perm]
        assert list(sorted_g) == [1, 1, 2, 3, 3]
    
    def test_sort_indices_multi_key(self):
        """Sort indices for multiple keys."""
        df = pd.DataFrame({
            'a': [2, 1, 2, 1],
            'b': [2, 1, 1, 2],
        })
        perm = _compute_sort_indices_v5(df, ['a', 'b'])
        
        sorted_a = df['a'].values[perm]
        sorted_b = df['b'].values[perm]
        
        # Should be sorted by (a, b)
        assert list(zip(sorted_a, sorted_b)) == [(1, 1), (1, 2), (2, 1), (2, 2)]
    
    def test_group_boundaries_correct(self):
        """Group boundaries computed correctly."""
        df = pd.DataFrame({'g': [0, 0, 0, 1, 1, 2]})
        perm = _compute_sort_indices_v5(df, 'g')
        offsets = _compute_group_boundaries_v5(df, 'g', perm)
        
        assert list(offsets) == [0, 3, 5, 6]


# ============================================================================
# PERFORMANCE TESTS (marked slow)
# ============================================================================

class TestV5Performance:
    """Performance tests for v5."""
    
    @pytest.mark.slow
    def test_v5_faster_than_v4_multiple(self, calibration_df):
        """v5 should be faster than multiple v4 calls."""
        import time
        
        gb = ['track_index', 'firstTForbit']
        
        # Time v4 × 3 calls
        start = time.time()
        _, _ = make_parallel_fit_v4(
            df=calibration_df, gb_columns=gb,
            fit_columns=['dyC2'], linear_columns=['rrel', 'rrel2'],
            suffix='_Y',
        )
        _, _ = make_parallel_fit_v4(
            df=calibration_df, gb_columns=gb,
            fit_columns=['dzC2'], linear_columns=['rrel'],
            suffix='_Z1',
        )
        _, _ = make_parallel_fit_v4(
            df=calibration_df, gb_columns=gb,
            fit_columns=['dzC2'], linear_columns=['rrel'],
            suffix='_Z2',
        )
        v4_time = time.time() - start
        
        # Time v5 single call
        start = time.time()
        _ = make_parallel_fit_v5(
            df=calibration_df,
            gb_columns=gb,
            fit_columns=['dyC2', 'dzC2', 'dzC2'],
            suffixes=['_Y', '_Z1', '_Z2'],
            linear_columns=[['rrel', 'rrel2'], ['rrel'], ['rrel']],
        )
        v5_time = time.time() - start
        
        print(f"\nv4 × 3: {v4_time:.3f}s, v5: {v5_time:.3f}s")
        # v5 should be at least as fast (ideally faster)
        # Not asserting speedup in unit test since it depends on data size


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
