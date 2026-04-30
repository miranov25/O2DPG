"""
Phase 13.24.ADF Part B — N1_11 missing column KeyError contract.

Deferred from Phase 13.23 review (P2-2). Pins the error contract for
confirmed subframe references with non-existent leaf columns.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


class TestN1_11_MissingColumnKeyError:

    @pytest.mark.invariance
    def test_N1_11_two_level_missing_column_raises(self):
        """Outer.Inner.nonexistent raises KeyError."""
        main = pd.DataFrame({
            'key_a': np.array([1, 2], dtype=np.int16),
            'x': np.array([10, 20], dtype=np.float32),
        })
        outer = pd.DataFrame({
            'key_a': np.array([1, 2], dtype=np.int16),
            'key_b': np.array([100, 200], dtype=np.int16),
        })
        inner = pd.DataFrame({
            'key_b': np.array([100, 200], dtype=np.int16),
            'val': np.array([1000, 2000], dtype=np.float32),
        })

        adf = AliasDataFrame(main)
        sf_outer = AliasDataFrame(outer)
        sf_inner = AliasDataFrame(inner)
        sf_outer.register_subframe('Inner', sf_inner, index_columns=['key_b'])
        adf.register_subframe('Outer', sf_outer, index_columns=['key_a'])

        adf.add_alias('bad', 'Outer.Inner.nonexistent', dtype=np.float32)
        with pytest.raises(KeyError, match="nonexistent"):
            adf.materialize_aliases(names=['bad'])

    @pytest.mark.invariance
    def test_N1_11b_single_level_missing_column_raises(self):
        """T.nonexistent raises KeyError."""
        main = pd.DataFrame({
            'key': np.array([1, 2], dtype=np.int16),
            'x': np.array([10, 20], dtype=np.float32),
        })
        sub = pd.DataFrame({
            'key': np.array([1, 2], dtype=np.int16),
            'val': np.array([100, 200], dtype=np.float32),
        })

        adf = AliasDataFrame(main)
        adf.register_subframe('T', AliasDataFrame(sub), index_columns=['key'])

        adf.add_alias('bad', 'T.nonexistent', dtype=np.float32)
        with pytest.raises(KeyError, match="nonexistent"):
            adf.materialize_aliases(names=['bad'])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
