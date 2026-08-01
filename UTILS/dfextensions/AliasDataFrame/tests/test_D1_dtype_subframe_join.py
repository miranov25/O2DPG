"""
BUG_AliasDataFrame_20260424_dtype_loss_subframe_join — Regression test

BUG: Aliases with integer/bool dtype that go through a subframe join
lose their declared dtype. The join produces NaN for missing keys,
and pandas raises IntCastingNaNError on .astype(int8). The cast fails
silently and the result stays float32.

FIX: _safe_dtype_cast fills NaN with 0/False before casting to
integer/bool dtypes, with a warning.

ENTRY POINT: materialize_aliases() — the production path.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


class TestDtypeLossSubframeJoin:
    """Tests for dtype preservation through subframe joins."""

    @pytest.mark.invariance
    def test_D1_int8_dtype_preserved_through_join(self):
        """
        D1: alias with dtype=int8 through subframe join preserves dtype.

        REVISED under AD-19 (architect, RATIFIED 2026-07-29), on his explicit
        instruction that this test be updated.

        The April behaviour asserted here was: NaN from the missing key is
        automatically filled with 0 and a RuntimeWarning is emitted. AD-19
        forbids that:

            "an unknown value must not silently become a neutral value unless
             the user explicitly configured that policy ... do not
             automatically choose 0, 1, False, or any other fill. Those are
             physical choices made by the user."

        0 is the neutral value of an ADDITIVE correction and 1 of a
        MULTIPLICATIVE one; a dtype cannot tell them apart. So the test now
        asserts both halves of the ruling: without a configured fill the
        operation refuses, and with one the dtype AND the matched values are
        preserved exactly.
        """
        main_df = pd.DataFrame({
            'sec': np.array([0, 1, 2, 999], dtype=np.int16),
            'x': np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        })
        sub_df = pd.DataFrame({
            'sec': np.array([0, 1, 2], dtype=np.int16),
            'flag': np.array([1, 0, 1], dtype=np.int8),
        })

        adf = AliasDataFrame(main_df)
        sf = AliasDataFrame(sub_df)
        adf.register_subframe('T', sf, index_columns=['sec'])
        adf.add_alias('is_good', 'T.flag', dtype=np.int8)

        # No configured fill: ADF refuses rather than inventing 0.
        with pytest.raises(ValueError, match="authoritative dtype|neutral value"):
            adf.materialize_aliases(names=['is_good'])

        # With the physically correct value configured, int8 is preserved.
        adf2 = AliasDataFrame(main_df.copy())
        sf2 = AliasDataFrame(sub_df.copy())
        adf2.register_subframe('T', sf2, index_columns=['sec'])
        adf2.add_alias('is_good', 'T.flag', dtype=np.int8, fill_value=0)
        adf2.materialize_aliases(names=['is_good'])

        assert adf2.df['is_good'].dtype == np.int8, (
            f"D1: expected int8, got {adf2.df['is_good'].dtype}"
        )
        # sec=999 missing -> the CONFIGURED fill, not an ADF-chosen one
        assert adf2.df['is_good'].iloc[3] == 0, "Missing key takes the configured fill"
        assert adf2.df['is_good'].iloc[0] == 1, "Present key should be 1"

    @pytest.mark.invariance
    def test_D2_bool_dtype_preserved_through_join(self):
        """
        D2: alias with dtype=bool through subframe join preserves dtype.

        REVISED under AD-19, same reasoning as D1: `False` is a physical
        choice, not something a Boolean dtype implies. Without a configured
        fill ADF refuses; with one the dtype and matched values are exact.
        """
        main_df = pd.DataFrame({
            'sec': np.array([0, 1, 999], dtype=np.int16),
            'x': np.array([1.0, 2.0, 3.0], dtype=np.float32),
        })
        sub_df = pd.DataFrame({
            'sec': np.array([0, 1], dtype=np.int16),
            'is_primary': np.array([True, False], dtype=bool),
        })

        adf = AliasDataFrame(main_df)
        sf = AliasDataFrame(sub_df)
        adf.register_subframe('T', sf, index_columns=['sec'])
        adf.add_alias('primary', 'T.is_primary', dtype=bool)

        with pytest.raises(ValueError, match="authoritative dtype|neutral value"):
            adf.materialize_aliases(names=['primary'])

        adf2 = AliasDataFrame(main_df.copy())
        sf2 = AliasDataFrame(sub_df.copy())
        adf2.register_subframe('T', sf2, index_columns=['sec'])
        adf2.add_alias('primary', 'T.is_primary', dtype=bool, fill_value=False)
        adf2.materialize_aliases(names=['primary'])

        assert adf2.df['primary'].dtype == bool, (
            f"D2: expected bool, got {adf2.df['primary'].dtype}"
        )
        assert adf2.df['primary'].iloc[0] == True
        assert adf2.df['primary'].iloc[1] == False
        assert adf2.df['primary'].iloc[2] == False  # missing -> configured fill

    @pytest.mark.invariance
    def test_D3_float_dtype_unaffected(self):
        """
        D3: float dtypes with NaN pass through without warning.
        NaN is native to float — no fill needed.
        """
        main_df = pd.DataFrame({
            'sec': np.array([0, 999], dtype=np.int16),
            'x': np.array([1.0, 2.0], dtype=np.float32),
        })
        sub_df = pd.DataFrame({
            'sec': np.array([0], dtype=np.int16),
            'val': np.array([42.0], dtype=np.float32),
        })

        adf = AliasDataFrame(main_df)
        sf = AliasDataFrame(sub_df)
        adf.register_subframe('T', sf, index_columns=['sec'])
        adf.add_alias('result', 'T.val', dtype=np.float32)

        # No warning for float dtype
        adf.materialize_aliases(names=['result'])

        assert adf.df['result'].dtype == np.float32
        assert adf.df['result'].iloc[0] == 42.0
        assert np.isnan(adf.df['result'].iloc[1])  # NaN preserved

    @pytest.mark.invariance
    def test_D4_no_missing_keys_no_warning(self):
        """
        D4: int dtype with no missing keys casts cleanly, no warning.
        """
        main_df = pd.DataFrame({
            'sec': np.array([0, 1], dtype=np.int16),
            'x': np.array([1.0, 2.0], dtype=np.float32),
        })
        sub_df = pd.DataFrame({
            'sec': np.array([0, 1], dtype=np.int16),
            'flag': np.array([1, 0], dtype=np.int8),
        })

        adf = AliasDataFrame(main_df)
        sf = AliasDataFrame(sub_df)
        adf.register_subframe('T', sf, index_columns=['sec'])
        adf.add_alias('is_good', 'T.flag', dtype=np.int8)

        # No warning when no NaN values exist
        adf.materialize_aliases(names=['is_good'])

        assert adf.df['is_good'].dtype == np.int8
        assert adf.df['is_good'].iloc[0] == 1
        assert adf.df['is_good'].iloc[1] == 0

    @pytest.mark.invariance
    def test_D5_boolean_and_operator_works(self):
        """
        D5: production pattern — boolean alias from subframe used with & operator.
        This was the original failure mode: isPrimITS & isNotEdge fails
        because isPrimITS is float32 instead of bool.
        """
        main_df = pd.DataFrame({
            'sec': np.array([0, 1, 2], dtype=np.int16),
            'flag_local': np.array([True, True, False], dtype=bool),
        })
        sub_df = pd.DataFrame({
            'sec': np.array([0, 1, 2], dtype=np.int16),
            'flag_remote': np.array([True, False, True], dtype=bool),
        })

        adf = AliasDataFrame(main_df)
        sf = AliasDataFrame(sub_df)
        adf.register_subframe('T', sf, index_columns=['sec'])
        adf.add_alias('remote', 'T.flag_remote', dtype=bool)
        adf.add_alias('combined', 'flag_local & remote', dtype=bool)

        adf.materialize_aliases(names=['remote', 'combined'])

        assert adf.df['combined'].dtype == bool
        np.testing.assert_array_equal(
            adf.df['combined'].values,
            np.array([True, False, False]),
            err_msg="D5: boolean & operator failed on subframe-joined alias"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
