"""
Test suite for V2/V3/V4 output standardization (Phase 2).

This module tests the standardization of output structure across V2, V3, and V4:
- Suffix applied to ALL columns (fit results + diagnostics)
- RMS and MAD always present (not just with diag=True)
- Diagnostic columns named consistently
- Multiple fits can be merged without column collision

Phase 2 standardization ensures:
1. V3 and V4 produce identical output structure
2. Diagnostic columns include suffix for merge compatibility
3. Statistical properties (RMS/MAD) are always present
4. Real-world use case: merging dfGB_Align + dfGB_Corr works cleanly

Note: V2 tests are skipped pending V2 API standardization to match V3/V4.
"""

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# V2/V3/V4 Column Parity Tests
# =============================================================================

@pytest.mark.skip(reason="V2 has different API (positional args) - standardize later if needed")
def test_v2_v3_v4_columns_without_diag():
    """Test that V2, V3, and V4 produce identical columns when diag=False"""
    np.random.seed(42)
    n = 100
    
    test_df = pd.DataFrame({
        'group': [1, 2] * (n // 2),
        'x': np.random.uniform(50, 150, n),
        'y': np.random.uniform(0, 1, n),
        'w': np.ones(n)
    })
    
    from ..groupby_regression_optimized import (
        make_parallel_fit_v2, 
        make_parallel_fit_v3, 
        make_parallel_fit_v4
    )
    
    # V2 has different API (positional args for median_columns, weights, selection)
    # TODO: Standardize V2 API to match V3/V4 keyword-only style
    _, dfGB_v2 = make_parallel_fit_v2(
        df=test_df,
        gb_columns=['group'],
        fit_columns=['y'],
        linear_columns=['x'],
        suffix='_test',
        diag=False,
        min_stat=10
    )
    
    _, dfGB_v3 = make_parallel_fit_v3(
        df=test_df,
        gb_columns=['group'],
        fit_columns=['y'],
        linear_columns=['x'],
        suffix='_test',
        diag=False,
        min_stat=10
    )
    
    _, dfGB_v4 = make_parallel_fit_v4(
        df=test_df,
        gb_columns=['group'],
        fit_columns=['y'],
        linear_columns=['x'],
        suffix='_test',
        diag=False,
        min_stat=10
    )
    
    # Check all have same columns
    cols_v2 = set(dfGB_v2.columns)
    cols_v3 = set(dfGB_v3.columns)
    cols_v4 = set(dfGB_v4.columns)
    
    assert cols_v2 == cols_v3, f"V2/V3 column mismatch: {cols_v2 ^ cols_v3}"
    assert cols_v3 == cols_v4, f"V3/V4 column mismatch: {cols_v3 ^ cols_v4}"
    
    # Check RMS and MAD are present
    assert 'y_rms_test' in dfGB_v2.columns, "V2 missing RMS"
    assert 'y_rms_test' in dfGB_v3.columns, "V3 missing RMS"
    assert 'y_rms_test' in dfGB_v4.columns, "V4 missing RMS"
    
    assert 'y_mad_test' in dfGB_v2.columns, "V2 missing MAD"
    assert 'y_mad_test' in dfGB_v3.columns, "V3 missing MAD"
    assert 'y_mad_test' in dfGB_v4.columns, "V4 missing MAD"
    
    # Check NO diagnostic columns present
    diag_cols_v2 = [c for c in dfGB_v2.columns if 'diag_' in c]
    diag_cols_v3 = [c for c in dfGB_v3.columns if 'diag_' in c]
    diag_cols_v4 = [c for c in dfGB_v4.columns if 'diag_' in c]
    
    assert len(diag_cols_v2) == 0, f"V2 should have no diag columns with diag=False, found {diag_cols_v2}"
    assert len(diag_cols_v3) == 0, f"V3 should have no diag columns with diag=False, found {diag_cols_v3}"
    assert len(diag_cols_v4) == 0, f"V4 should have no diag columns with diag=False, found {diag_cols_v4}"


@pytest.mark.skip(reason="V2 has different API (positional args) - standardize later if needed")
def test_v2_v3_v4_columns_with_diag():
    """Test that V2, V3, and V4 produce identical columns when diag=True (including suffix)"""
    np.random.seed(42)
    n = 100
    
    test_df = pd.DataFrame({
        'group': [1, 2] * (n // 2),
        'x': np.random.uniform(50, 150, n),
        'y': np.random.uniform(0, 1, n),
        'w': np.ones(n)
    })
    
    from ..groupby_regression_optimized import (
        make_parallel_fit_v2, 
        make_parallel_fit_v3, 
        make_parallel_fit_v4
    )
    
    # V2 has different API - TODO: standardize
    _, dfGB_v2 = make_parallel_fit_v2(
        df=test_df,
        gb_columns=['group'],
        fit_columns=['y'],
        linear_columns=['x'],
        suffix='_test',
        diag=True,
        min_stat=10
    )
    
    _, dfGB_v3 = make_parallel_fit_v3(
        df=test_df,
        gb_columns=['group'],
        fit_columns=['y'],
        linear_columns=['x'],
        suffix='_test',
        diag=True,
        min_stat=10
    )
    
    _, dfGB_v4 = make_parallel_fit_v4(
        df=test_df,
        gb_columns=['group'],
        fit_columns=['y'],
        linear_columns=['x'],
        suffix='_test',
        diag=True,
        min_stat=10
    )
    
    # Check all have same columns (allowing for V3-specific columns like time_ms)
    cols_v2 = set(dfGB_v2.columns)
    cols_v3 = set(dfGB_v3.columns)
    cols_v4 = set(dfGB_v4.columns)
    
    # Core columns should match (V3 has extra timing columns)
    core_cols_v2 = {c for c in cols_v2 if 'time_ms' not in c and 'wall_ms' not in c}
    core_cols_v3 = {c for c in cols_v3 if 'time_ms' not in c and 'wall_ms' not in c}
    core_cols_v4 = {c for c in cols_v4 if 'time_ms' not in c and 'wall_ms' not in c}
    
    assert core_cols_v2 == core_cols_v4, f"V2/V4 core column mismatch: {core_cols_v2 ^ core_cols_v4}"
    assert core_cols_v3 == core_cols_v4, f"V3/V4 core column mismatch: {core_cols_v3 ^ core_cols_v4}"
    
    # Check diagnostic columns have suffix
    expected_diag_cols = [
        'diag_n_total_test',
        'diag_n_valid_test',
        'diag_n_filtered_test',
        'diag_cond_xtx_test',
        'diag_status_test'
    ]
    
    for col in expected_diag_cols:
        assert col in dfGB_v2.columns, f"V2 missing {col}"
        assert col in dfGB_v3.columns, f"V3 missing {col}"
        assert col in dfGB_v4.columns, f"V4 missing {col}"
    
    # Check RMS and MAD still present
    assert 'y_rms_test' in dfGB_v2.columns, "V2 missing RMS with diag=True"
    assert 'y_mad_test' in dfGB_v2.columns, "V2 missing MAD with diag=True"


# =============================================================================
# Suffix Application Tests
# =============================================================================

def test_suffix_applied_to_all_output_columns():
    """
    Test that suffix is applied to ALL output columns (fit + diagnostic).
    
    This validates the Phase 2 standardization where diagnostic columns
    now include suffix, enabling clean merging of multiple fits.
    """
    np.random.seed(42)
    n = 100
    
    test_df = pd.DataFrame({
        'sector': [1, 2] * (n // 2),
        'chamber': ['A', 'B'] * (n // 2),
        'x': np.random.uniform(50, 150, n),
        'y': np.random.uniform(0, 1, n)
    })
    
    from ..groupby_regression_optimized import make_parallel_fit_v3
    
    suffix = '_CustomSuffix'
    
    _, dfGB = make_parallel_fit_v3(
        df=test_df,
        gb_columns=['sector', 'chamber'],
        fit_columns=['y'],
        linear_columns=['x'],
        suffix=suffix,
        diag=True,
        min_stat=10
    )
    
    # Group keys should NOT have suffix
    group_keys = {'sector', 'chamber'}
    
    # All other columns MUST have suffix
    for col in dfGB.columns:
        if col not in group_keys:
            assert col.endswith(suffix), f"Column '{col}' missing suffix '{suffix}'"
    
    # Specifically check diagnostic columns
    diag_cols = [c for c in dfGB.columns if c.startswith('diag_')]
    assert len(diag_cols) > 0, "Should have diagnostic columns"
    
    for col in diag_cols:
        assert col.endswith(suffix), f"Diagnostic column '{col}' missing suffix"


# =============================================================================
# RMS/MAD Availability Tests
# =============================================================================

def test_rms_mad_always_present():
    """
    Test that RMS and MAD are present regardless of diag setting.
    
    Phase 2 standardization moved RMS/MAD from diagnostics to fit results,
    making them always available for quality assessment.
    """
    np.random.seed(42)
    n = 100
    
    test_df = pd.DataFrame({
        'group': [1] * n,
        'x': np.random.uniform(50, 150, n),
        'y': np.random.uniform(0, 1, n)
    })
    
    from ..groupby_regression_optimized import make_parallel_fit_v3, make_parallel_fit_v4
    
    # Test V3
    _, dfGB_v3_no_diag = make_parallel_fit_v3(
        df=test_df,
        gb_columns=['group'],
        fit_columns=['y'],
        linear_columns=['x'],
        suffix='_test',
        diag=False,
        min_stat=10
    )
    
    _, dfGB_v3_with_diag = make_parallel_fit_v3(
        df=test_df,
        gb_columns=['group'],
        fit_columns=['y'],
        linear_columns=['x'],
        suffix='_test',
        diag=True,
        min_stat=10
    )
    
    # Test V4
    _, dfGB_v4_no_diag = make_parallel_fit_v4(
        df=test_df,
        gb_columns=['group'],
        fit_columns=['y'],
        linear_columns=['x'],
        suffix='_test',
        diag=False,
        min_stat=10
    )
    
    _, dfGB_v4_with_diag = make_parallel_fit_v4(
        df=test_df,
        gb_columns=['group'],
        fit_columns=['y'],
        linear_columns=['x'],
        suffix='_test',
        diag=True,
        min_stat=10
    )
    
    # Check RMS and MAD in all cases
    for name, dfGB in [
        ('V3 diag=False', dfGB_v3_no_diag),
        ('V3 diag=True', dfGB_v3_with_diag),
        ('V4 diag=False', dfGB_v4_no_diag),
        ('V4 diag=True', dfGB_v4_with_diag)
    ]:
        assert 'y_rms_test' in dfGB.columns, f"{name} missing RMS"
        assert 'y_mad_test' in dfGB.columns, f"{name} missing MAD"
        
        # Values should be finite and positive
        rms = dfGB['y_rms_test'].values[0]
        mad = dfGB['y_mad_test'].values[0]
        
        assert np.isfinite(rms) and rms > 0, f"{name} has invalid RMS: {rms}"
        assert np.isfinite(mad) and mad >= 0, f"{name} has invalid MAD: {mad}"


# =============================================================================
# Real-World Use Case: Merge Compatibility
# =============================================================================

def test_merge_multiple_fits_with_suffix():
    """
    Test real-world scenario: merging multiple fits with different suffixes.
    
    This is the primary motivation for Phase 2 standardization. In ALICE TPC
    analysis, users commonly merge alignment fits (dfGB_Align) with correction
    fits (dfGB_Corr). Without suffix on diagnostic columns, pandas creates
    collision columns (_x, _y) which is confusing.
    
    With Phase 2 standardization:
    - diag_n_total_Align and diag_n_total_Corr (no collision!)
    - Clean comparison of fit quality across different calibrations
    """
    np.random.seed(42)
    n = 100
    
    # Same data, two different "fits" (simulating Align and Corr)
    test_df = pd.DataFrame({
        'sector': [1, 2, 3, 4] * (n // 4),
        'x': np.random.uniform(50, 150, n),
        'y_align': np.random.uniform(0, 1, n),
        'y_corr': np.random.uniform(0, 1, n)
    })
    
    from ..groupby_regression_optimized import make_parallel_fit_v3
    
    # Alignment fit
    _, dfGB_Align = make_parallel_fit_v3(
        df=test_df,
        gb_columns=['sector'],
        fit_columns=['y_align'],
        linear_columns=['x'],
        suffix='_Align',
        diag=True,
        min_stat=10
    )
    
    # Correction fit
    _, dfGB_Corr = make_parallel_fit_v3(
        df=test_df,
        gb_columns=['sector'],
        fit_columns=['y_corr'],
        linear_columns=['x'],
        suffix='_Corr',
        diag=True,
        min_stat=10
    )
    
    # Merge on sector
    merged = dfGB_Align.merge(dfGB_Corr, on='sector')
    
    # Check NO column collisions (pandas adds _x/_y if collision)
    collision_cols = [c for c in merged.columns if c.endswith('_x') or c.endswith('_y')]
    assert len(collision_cols) == 0, f"Column collisions detected: {collision_cols}"
    
    # Check both sets of diagnostic columns present with correct suffixes
    assert 'diag_n_total_Align' in merged.columns, "Missing Align diagnostic"
    assert 'diag_n_total_Corr' in merged.columns, "Missing Corr diagnostic"
    assert 'diag_status_Align' in merged.columns, "Missing Align status"
    assert 'diag_status_Corr' in merged.columns, "Missing Corr status"
    
    # Check both sets of RMS/MAD present
    assert 'y_align_rms_Align' in merged.columns, "Missing Align RMS"
    assert 'y_corr_rms_Corr' in merged.columns, "Missing Corr RMS"
    assert 'y_align_mad_Align' in merged.columns, "Missing Align MAD"
    assert 'y_corr_mad_Corr' in merged.columns, "Missing Corr MAD"
    
    # Check can compare statuses
    status_comparison = merged[['sector', 'diag_status_Align', 'diag_status_Corr']]
    assert len(status_comparison) == 4, "Should have 4 sectors"
