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
        
        Before fix: result is float32 (cast fails on NaN from missing keys).
        After fix: result is int8 (NaN filled with 0, warning emitted).
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

        with pytest.warns(RuntimeWarning, match="NaN values filled"):
            adf.materialize_aliases(names=['is_good'])

        assert adf.df['is_good'].dtype == np.int8, (
            f"D1: expected int8, got {adf.df['is_good'].dtype}"
        )
        # sec=999 missing → NaN → filled with 0
        assert adf.df['is_good'].iloc[3] == 0, "Missing key should be 0"
        assert adf.df['is_good'].iloc[0] == 1, "Present key should be 1"

    @pytest.mark.invariance
    def test_D2_bool_dtype_preserved_through_join(self):
        """
        D2: alias with dtype=bool through subframe join preserves dtype.
        Missing keys become False.
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

        with pytest.warns(RuntimeWarning, match="NaN values filled"):
            adf.materialize_aliases(names=['primary'])

        assert adf.df['primary'].dtype == bool, (
            f"D2: expected bool, got {adf.df['primary'].dtype}"
        )
        assert adf.df['primary'].iloc[0] == True
        assert adf.df['primary'].iloc[1] == False
        assert adf.df['primary'].iloc[2] == False  # missing → False

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
