"""
Batch 3 — I11: Compression Working-Paths Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE (§5.1 deviation note, established in Batch 1).
All tests are marked @pytest.mark.invariance.

SCOPE (per v1.2 §4.3)
---------------------
I11 covers only the WORKING linear compression path. The broken
asinh/scaled paths (COMP.roundtrip 🧨, xfail tests I4_2 and I4_3 in
test_invariance_compression.py) are explicitly out of scope per v1.2
§3.2 "Bug fixes for the 4 known-broken features — these need separate
bug-fix phases."

SCOPE DEVIATION FROM v1.2 §4.3 WORDING
---------------------------------------
The proposal §4.3 I11_1 uses shorthand:
    compress_columns({c: {'formula':'linear', 'bits':16}})
The actual source API (AliasDataFrame.py:6185) uses explicit compress/
decompress expressions:
    compress_columns({col: {'compress': 'np.round(col*scale)', ...}})
Following the pattern of existing test_invariance_compression.py
(LINEAR_COMPRESSION_SPEC). Shorthand 'formula'/'bits' keys do not
exist in current source. This is a spec-wording vs source-reality
mismatch, not a scope change — the tests still verify the linear
compression invariant as intended.

PATH-EXPLICIT DISCIPLINE (Failure Mode #11)
-------------------------------------------
Tests call compress_columns and decompress_columns directly with
explicit compress/decompress expressions. No auto-dispatch.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


# Linear compression spec with int16 and scale=100, producing max error
# of 0.5/scale = 0.005 on the reconstructed value.
# For int16 range [-32768, 32767] at scale=100: representable range is
# approximately [-327.67, 327.67] — well within our test data [-1,+1].
LINEAR_INT16_SPEC = {
    'val': {
        'compress': 'np.round(val * 100).astype(np.int16)',
        'decompress': 'val_c / 100.0',
        'compressed_dtype': np.int16,
        'decompressed_dtype': np.float32,
    }
}

# Expected max absolute error for round-to-int quantization at scale=100:
#   |x_reconstructed - x_original| <= 0.5 / scale = 0.005
LINEAR_MAX_ERROR_SCALE_100 = 0.005


@pytest.fixture
def adf_for_compression():
    """ADF with a single float column well within the int16/scale=100 range."""
    n = 200
    rng = np.random.RandomState(13122)
    df = pd.DataFrame({
        'val': rng.uniform(-1.0, 1.0, n).astype(np.float32),
        'untouched': rng.uniform(10.0, 20.0, n).astype(np.float32),
    })
    return AliasDataFrame(df)


class TestI11CompressionWorkingPathInvariance:
    """Phase 13.12 I11 — Compression working paths (linear only)."""

    @pytest.mark.invariance
    def test_I11_1_linear_compress_decompress_max_error_within_bit_budget(
        self, adf_for_compression
    ):
        """
        I11_1 INVARIANT:
            For a linear-quantization compression spec with scale=100,
            the maximum reconstruction error after
            compress_columns() -> decompress_columns() is bounded by
            0.5 / scale = 0.005.
        CODE PATH:
            adf.compress_columns(LINEAR_INT16_SPEC, drop_original=True)
            adf.decompress_columns(['val'], inplace=True)
            All parameters explicit, no defaults.
        PRODUCTION ENTRY POINT:
            adf.compress_columns({col: {compress, decompress, ...}})
            adf.decompress_columns([col])
        REGRESSION GUARD: linear quantization bit-budget contract.
        """
        adf = adf_for_compression
        original = adf.df['val'].values.copy()

        # Compress: should replace 'val' with 'val_c' (int16)
        adf.compress_columns(LINEAR_INT16_SPEC, drop_original=True)
        assert 'val_c' in adf.df.columns, (
            "I11_1: compress_columns did not produce 'val_c' as expected"
        )

        # Decompress: should restore 'val' as float32
        adf.decompress_columns(['val'], inplace=True)
        assert 'val' in adf.df.columns, (
            "I11_1: decompress_columns did not restore 'val'"
        )

        reconstructed = adf.df['val'].values
        max_error = float(np.max(np.abs(reconstructed - original)))

        assert max_error <= LINEAR_MAX_ERROR_SCALE_100, (
            f"I11_1: max error {max_error:.6f} exceeds bit-budget bound "
            f"{LINEAR_MAX_ERROR_SCALE_100:.6f} for scale=100 quantization"
        )

    @pytest.mark.invariance
    def test_I11_2_mixed_compressed_uncompressed_arithmetic(
        self, adf_for_compression
    ):
        """
        I11_2 INVARIANT:
            Arithmetic between a compressed-then-decompressed column
            and an untouched column produces the same result (within
            compression precision) as arithmetic between two untouched
            columns.
        CODE PATH:
            Path A: compress val -> decompress val -> alias 'prod = val*untouched'
            Path B: on a fresh ADF, alias 'prod_ref = val*untouched' directly
            Compare within 2 * LINEAR_MAX_ERROR_SCALE_100 * max(|untouched|).
        PRODUCTION ENTRY POINT:
            adf.compress_columns + adf.decompress_columns + add_alias
        REGRESSION GUARD: dispatch in mixed compression state — ensures
            a decompressed column participates in arithmetic identically
            to an uncompressed column, up to the compression precision.
        """
        adf = adf_for_compression
        df_snapshot = adf.df.copy()

        # Path A: compress -> decompress -> arithmetic
        adf.compress_columns(LINEAR_INT16_SPEC, drop_original=True)
        adf.decompress_columns(['val'], inplace=True)
        adf.add_alias('prod', 'val * untouched')
        adf.materialize_aliases(names=['prod'])
        prod_a = adf.df['prod'].values.copy()

        # Path B: same arithmetic on the snapshot without compression
        adf_b = AliasDataFrame(df_snapshot)
        adf_b.add_alias('prod_ref', 'val * untouched')
        adf_b.materialize_aliases(names=['prod_ref'])
        prod_b = adf_b.df['prod_ref'].values

        # Expected error bound:
        # |val_compressed * u - val * u| <= |u| * max_error_per_scalar
        # Use maximum |untouched| to form a conservative absolute bound.
        max_abs_untouched = float(np.max(np.abs(df_snapshot['untouched'].values)))
        bound = 2.0 * LINEAR_MAX_ERROR_SCALE_100 * max_abs_untouched

        max_diff = float(np.max(np.abs(prod_a - prod_b)))
        assert max_diff <= bound, (
            f"I11_2: mixed compressed/uncompressed arithmetic diff "
            f"{max_diff:.6f} exceeds bound {bound:.6f} "
            f"(LINEAR_MAX_ERROR={LINEAR_MAX_ERROR_SCALE_100}, "
            f"max_abs_untouched={max_abs_untouched:.3f})"
        )
