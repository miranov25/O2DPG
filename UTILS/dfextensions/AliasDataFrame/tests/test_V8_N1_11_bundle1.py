"""
Bundle 1 — Cleanup tests

V8: aliases property returns read-only MappingProxyType.
    Mutation via del/setitem/update raises TypeError.

N1_11: multi-level chain with non-existent leaf column raises KeyError.
    Pins the error contract from Phase 13.23 review (Claude36 P2).
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


class TestAliasesImmutable:
    """V8: aliases dict is read-only."""

    def test_V8_1_setitem_raises(self):
        """adf.aliases['x'] = 'new' raises TypeError."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        adf.add_alias('a', 'x + 1')

        with pytest.raises(TypeError):
            adf.aliases['a'] = 'x + 999'

        # Original expression unchanged
        assert adf.aliases['a'] == 'x + 1'

    def test_V8_2_del_raises(self):
        """del adf.aliases['x'] raises TypeError."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        adf.add_alias('a', 'x + 1')

        with pytest.raises(TypeError):
            del adf.aliases['a']

        assert 'a' in adf.aliases

    def test_V8_3_update_raises(self):
        """adf.aliases.update({...}) raises AttributeError (proxy has no update)."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        adf.add_alias('a', 'x + 1')

        with pytest.raises(AttributeError):
            adf.aliases.update({'a': 'x + 999'})

    def test_V8_4_read_still_works(self):
        """Reading aliases dict works normally."""
        df = pd.DataFrame({'x': [1, 2, 3]})
        adf = AliasDataFrame(df)
        adf.add_alias('a', 'x + 1')
        adf.add_alias('b', 'x * 2')

        assert adf.aliases['a'] == 'x + 1'
        assert 'b' in adf.aliases
        assert len(adf.aliases) == 2
        assert set(adf.aliases.keys()) == {'a', 'b'}
        assert list(adf.aliases.values()) == ['x + 1', 'x * 2']


class TestN1_11_MissingColumnKeyError:
    """N1_11: multi-level chain with non-existent leaf raises KeyError."""

    @pytest.mark.invariance
    def test_N1_11_two_level_missing_column(self):
        """
        Outer.Inner.nonexistent raises KeyError, not silently returns NaN.
        Pins error contract for confirmed subframe refs with invalid leaf.
        """
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
    def test_N1_11b_single_level_missing_column(self):
        """Same contract for single-level: Sub.nonexistent raises KeyError."""
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
