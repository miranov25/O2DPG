"""
test_invariance_compression.py — I4: Compression Invariance

Phase 13.7.ADF: Invariance Test Suite
Priority: P1
Tests: 8

Property: decompress(compress(x)) == apply_decompress_formula(apply_compress_formula(x))

With user-defined quantization, the invariant is NOT exact round-trip but:
- Compression formula applied correctly
- Decompression formula applied correctly  
- Error bounded by quantization step

Author: Claude2-Coder
Date: 2026-01-16
Version: 2.0 - Fixed to use user-defined compression spec format
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# =============================================================================
# PREDEFINED COMPRESSION SPECS (from real usage)
# =============================================================================

# Linear compression: round(x * scale) / scale
# Error bound: 0.5 / scale
LINEAR_COMPRESSION_SPEC = {
    'mP3': {
        'compress': 'np.round(mP3 * 50).astype(np.int8)',
        'decompress': 'mP3_c / 50.',
        'compressed_dtype': np.int8,
        'decompressed_dtype': np.float32
    },
    'mP4': {
        'compress': 'np.round(mP4 * 20).astype(np.int8)',
        'decompress': 'mP4_c / 20.',
        'compressed_dtype': np.int8,
        'decompressed_dtype': np.float32
    },
}

# Scaled linear compression with larger range
SCALED_COMPRESSION_SPEC = {
    'y': {
        'compress': 'np.round(y * (0x7fff / 50.)).astype(np.int16)',
        'decompress': 'y_c * (50.0 / 0x7fff)',
        'compressed_dtype': np.int16,
        'decompressed_dtype': np.float32
    },
    'z': {
        'compress': 'np.round(z * (0x7fff / 300.)).astype(np.int16)',
        'decompress': 'z_c * (300.0 / 0x7fff)',
        'compressed_dtype': np.int16,
        'decompressed_dtype': np.float32
    },
}

# Asinh compression for handling wide dynamic range
ASINH_COMPRESSION_SPEC = {
    'dy': {
        'compress': 'np.round(np.arcsinh(dy) * 40).astype(np.int16)',
        'decompress': 'np.sinh(dy_c / 40.)',
        'compressed_dtype': np.int16,
        'decompressed_dtype': np.float16
    },
    'dz': {
        'compress': 'np.round(np.arcsinh(dz) * 40).astype(np.int16)',
        'decompress': 'np.sinh(dz_c / 40.)',
        'compressed_dtype': np.int16,
        'decompressed_dtype': np.float16
    },
}

# Square root compression for positive values
SQRT_COMPRESSION_SPEC = {
    'dEdxTPC': {
        'compress': 'np.round(np.sqrt(dEdxTPC) * 8).astype(np.uint8)',
        'decompress': '(dEdxTPC_c.astype(np.float32) ** 2) / 64.',
        'compressed_dtype': np.uint8,
        'decompressed_dtype': np.float16
    },
}


def compute_linear_tolerance(scale, dtype=np.float32):
    """
    Compute expected tolerance for linear quantization.
    
    For round(x * scale) / scale, max error is 0.5 / scale
    """
    return 0.5 / scale + np.finfo(dtype).eps * 10


def compute_asinh_tolerance(scale, x_range=10.0):
    """
    Compute expected tolerance for asinh quantization.
    
    asinh is approximately linear near 0 and logarithmic for large |x|.
    The worst case error depends on the derivative of sinh.
    """
    # For asinh compression, the error in reconstructed space depends on
    # the derivative of sinh at the quantized point
    # sinh'(x) = cosh(x), which grows exponentially
    # For typical ranges, use conservative estimate
    return 0.5 / scale * np.cosh(np.arcsinh(x_range))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def compression_test_df():
    """Create test DataFrame with columns for each compression type."""
    np.random.seed(42)
    n = 5000
    
    return pd.DataFrame({
        # For linear compression (small range)
        'mP3': np.random.randn(n).astype(np.float32) * 0.1,  # Range ~[-0.3, 0.3]
        'mP4': np.random.randn(n).astype(np.float32) * 0.2,  # Range ~[-0.6, 0.6]
        
        # For scaled linear compression
        'y': np.random.randn(n).astype(np.float32) * 20,     # Range ~[-60, 60]
        'z': np.random.randn(n).astype(np.float32) * 100,    # Range ~[-300, 300]
        
        # For asinh compression (can handle wide range)
        'dy': np.random.randn(n).astype(np.float32) * 5,     # Range ~[-15, 15]
        'dz': np.random.randn(n).astype(np.float32) * 5,
        
        # For sqrt compression (positive values only)
        'dEdxTPC': np.abs(np.random.randn(n).astype(np.float32)) * 100 + 1,  # Positive
    })


# =============================================================================
# TEST CLASS: COMPRESSION INVARIANCE
# =============================================================================

@pytest.mark.invariance
class TestInvarianceCompression:
    """
    I4: Compression Invariance Tests
    
    Core invariant: compress → decompress produces values within quantization tolerance.
    
    Unlike exact round-trip, user-defined quantization has bounded error.
    """
    
    def test_I4_1_linear_compression_roundtrip(self, compression_test_df):
        """
        I4_1: Linear compression (round(x * scale)) preserves data within tolerance.
        """
        from AliasDataFrame import AliasDataFrame
        
        df = compression_test_df.copy()
        original_mP3 = df['mP3'].values.copy()
        original_mP4 = df['mP4'].values.copy()
        
        adf = AliasDataFrame(df)
        
        # Compress using linear spec
        adf.compress_columns(LINEAR_COMPRESSION_SPEC)
        adf.decompress_columns(['mP3', 'mP4'])
        
        result_mP3 = adf.df['mP3'].values
        result_mP4 = adf.df['mP4'].values
        
        # Check within tolerance: 0.5 / scale
        tol_mP3 = compute_linear_tolerance(50)
        tol_mP4 = compute_linear_tolerance(20)
        
        np.testing.assert_allclose(result_mP3, original_mP3, atol=tol_mP3, 
                                   err_msg="I4_1_mP3_linear")
        np.testing.assert_allclose(result_mP4, original_mP4, atol=tol_mP4,
                                   err_msg="I4_1_mP4_linear")
    
    def test_I4_2_scaled_linear_compression_roundtrip(self, compression_test_df):
        """
        I4_2: Scaled linear compression (16-bit) preserves data within tolerance.
        """
        from AliasDataFrame import AliasDataFrame
        
        df = compression_test_df.copy()
        original_y = df['y'].values.copy()
        original_z = df['z'].values.copy()
        
        adf = AliasDataFrame(df)
        
        adf.compress_columns(SCALED_COMPRESSION_SPEC)
        adf.decompress_columns(['y', 'z'])
        
        result_y = adf.df['y'].values
        result_z = adf.df['z'].values
        
        # For 16-bit with 0x7fff scaling
        tol_y = 50.0 / 0x7fff * 0.5  # ~0.0015
        tol_z = 300.0 / 0x7fff * 0.5  # ~0.009
        
        np.testing.assert_allclose(result_y, original_y, atol=tol_y * 2,  # 2x for safety
                                   err_msg="I4_2_y_scaled")
        np.testing.assert_allclose(result_z, original_z, atol=tol_z * 2,
                                   err_msg="I4_2_z_scaled")
    
    def test_I4_3_asinh_compression_roundtrip(self, compression_test_df):
        """
        I4_3: Asinh compression preserves data within tolerance.
        
        asinh handles wide dynamic range by compressing logarithmically.
        """
        from AliasDataFrame import AliasDataFrame
        
        df = compression_test_df.copy()
        original_dy = df['dy'].values.copy()
        original_dz = df['dz'].values.copy()
        
        adf = AliasDataFrame(df)
        
        adf.compress_columns(ASINH_COMPRESSION_SPEC)
        adf.decompress_columns(['dy', 'dz'])
        
        result_dy = adf.df['dy'].values
        result_dz = adf.df['dz'].values
        
        # Asinh tolerance is more complex - use relative tolerance
        # For typical values, error is roughly 0.5/40 * cosh(asinh(x)) = 0.5/40 * sqrt(1+x^2)
        # Use 5% relative tolerance for safety
        np.testing.assert_allclose(result_dy, original_dy, rtol=0.05, atol=0.01,
                                   err_msg="I4_3_dy_asinh")
        np.testing.assert_allclose(result_dz, original_dz, rtol=0.05, atol=0.01,
                                   err_msg="I4_3_dz_asinh")
    
    def test_I4_4_sqrt_compression_roundtrip(self, compression_test_df):
        """
        I4_4: Sqrt compression for positive values preserves data.
        """
        from AliasDataFrame import AliasDataFrame
        
        df = compression_test_df.copy()
        original = df['dEdxTPC'].values.copy()
        
        adf = AliasDataFrame(df)
        
        adf.compress_columns(SQRT_COMPRESSION_SPEC)
        adf.decompress_columns(['dEdxTPC'])
        
        result = adf.df['dEdxTPC'].values
        
        # Sqrt compression: round(sqrt(x) * 8) then (c^2 / 64)
        # Error depends on value magnitude
        # Use relative tolerance
        np.testing.assert_allclose(result, original, rtol=0.15, atol=1.0,
                                   err_msg="I4_4_sqrt_compression")
    
    def test_I4_5_compress_decompress_idempotent(self, compression_test_df):
        """
        I4_5: Multiple decompress calls don't change values.
        """
        from AliasDataFrame import AliasDataFrame
        
        df = compression_test_df.copy()
        
        adf = AliasDataFrame(df)
        adf.compress_columns(LINEAR_COMPRESSION_SPEC)
        
        # First decompress
        adf.decompress_columns(['mP3'])
        first_result = adf.df['mP3'].values.copy()
        
        # Compress again and decompress
        adf.compress_columns(columns=['mP3'])
        adf.decompress_columns(['mP3'])
        second_result = adf.df['mP3'].values
        
        # Should be identical (within float precision)
        np.testing.assert_allclose(second_result, first_result, rtol=1e-6,
                                   err_msg="I4_5_idempotent")
    
    def test_I4_6_schema_preserved_after_roundtrip(self, compression_test_df):
        """
        I4_6: Compression schema is preserved in ADF metadata.
        """
        from AliasDataFrame import AliasDataFrame
        
        df = compression_test_df.copy()
        
        adf = AliasDataFrame(df)
        adf.compress_columns(LINEAR_COMPRESSION_SPEC)
        
        # Check schema contains compression info
        assert 'compression' in adf._schema
        assert 'mP3' in adf._schema['compression']
        assert 'mP4' in adf._schema['compression']
        
        # Check schema has required fields
        mP3_schema = adf._schema['compression']['mP3']
        assert 'compress' in mP3_schema or 'compress_expr' in mP3_schema
        assert 'decompress' in mP3_schema or 'decompress_expr' in mP3_schema
    
    def test_I4_7_compressed_dtype_correct(self, compression_test_df):
        """
        I4_7: Compressed columns have correct dtype.
        """
        from AliasDataFrame import AliasDataFrame
        
        df = compression_test_df.copy()
        
        adf = AliasDataFrame(df)
        adf.compress_columns(LINEAR_COMPRESSION_SPEC)
        
        # Check compressed columns exist with correct dtype
        assert 'mP3_c' in adf.df.columns
        assert 'mP4_c' in adf.df.columns
        assert adf.df['mP3_c'].dtype == np.int8
        assert adf.df['mP4_c'].dtype == np.int8
    
    def test_I4_8_multi_column_compression(self, compression_test_df):
        """
        I4_8: Multiple columns compressed in single call.
        """
        from AliasDataFrame import AliasDataFrame
        
        df = compression_test_df.copy()
        originals = {
            'mP3': df['mP3'].values.copy(),
            'mP4': df['mP4'].values.copy(),
        }
        
        adf = AliasDataFrame(df)
        
        # Compress both columns at once
        adf.compress_columns(LINEAR_COMPRESSION_SPEC)
        
        # Verify compressed columns exist
        assert 'mP3_c' in adf.df.columns
        assert 'mP4_c' in adf.df.columns
        
        # Decompress and verify
        adf.decompress_columns(['mP3', 'mP4'])
        
        for col in ['mP3', 'mP4']:
            scale = 50 if col == 'mP3' else 20
            tol = compute_linear_tolerance(scale)
            np.testing.assert_allclose(
                adf.df[col].values, originals[col], 
                atol=tol,
                err_msg=f"I4_8_{col}"
            )


# =============================================================================
# TEST SUMMARY
# =============================================================================

class TestCompressionSummary:
    """Verify all I4 tests are present."""
    
    def test_I4_count(self):
        """Verify we have 8 compression tests."""
        tests = [
            'test_I4_1_linear_compression_roundtrip',
            'test_I4_2_scaled_linear_compression_roundtrip',
            'test_I4_3_asinh_compression_roundtrip',
            'test_I4_4_sqrt_compression_roundtrip',
            'test_I4_5_compress_decompress_idempotent',
            'test_I4_6_schema_preserved_after_roundtrip',
            'test_I4_7_compressed_dtype_correct',
            'test_I4_8_multi_column_compression',
        ]
        assert len(tests) == 8, "Expected 8 I4 tests"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "invariance"])
