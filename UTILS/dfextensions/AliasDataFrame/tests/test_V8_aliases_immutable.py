"""
Phase 13.24.ADF Part B — aliases and alias_dtypes read-only contract tests.

V8_1-V8_9: verify mutation raises TypeError, reads work, JSON/pickle/isinstance OK.
"""

import os
import sys
import json
import copy
import pickle
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


def _make_adf():
    df = pd.DataFrame({'x': np.array([1, 2, 3], dtype=np.float32)})
    adf = AliasDataFrame(df)
    adf.add_alias('a', 'x + 1', dtype=np.float32)
    adf.add_alias('b', 'x * 2', dtype=np.float32)
    return adf


class TestV8AliasesImmutable:

    def test_V8_1_setitem_raises(self):
        """adf.aliases['x'] = 'expr' raises TypeError."""
        adf = _make_adf()
        with pytest.raises(TypeError, match="add_alias"):
            adf.aliases['a'] = 'x + 999'
        assert adf.aliases['a'] == 'x + 1'

    def test_V8_2_del_raises(self):
        """del adf.aliases['x'] raises TypeError."""
        adf = _make_adf()
        with pytest.raises(TypeError, match="remove_alias"):
            del adf.aliases['a']
        assert 'a' in adf.aliases

    def test_V8_3_update_raises(self):
        """adf.aliases.update({...}) raises TypeError."""
        adf = _make_adf()
        with pytest.raises(TypeError, match="add_alias"):
            adf.aliases.update({'a': 'x + 999'})

    def test_V8_4_read_paths_unchanged(self):
        """All read patterns work normally."""
        adf = _make_adf()
        a = adf.aliases

        assert a['a'] == 'x + 1'
        assert a['b'] == 'x * 2'
        assert 'a' in a
        assert 'c' not in a
        assert len(a) == 2
        assert set(a.keys()) == {'a', 'b'}
        assert set(a.values()) == {'x + 1', 'x * 2'}
        assert dict(a) == {'a': 'x + 1', 'b': 'x * 2'}
        assert list(a.items()) == [('a', 'x + 1'), ('b', 'x * 2')]
        # Iteration
        keys = [k for k in a]
        assert set(keys) == {'a', 'b'}

    def test_V8_5_json_serializable(self):
        """json.dumps works on aliases (dict subclass)."""
        adf = _make_adf()
        # Direct serialization
        s = json.dumps(adf.aliases)
        parsed = json.loads(s)
        assert parsed == {'a': 'x + 1', 'b': 'x * 2'}

        # Via dict()
        s2 = json.dumps(dict(adf.aliases))
        assert json.loads(s2) == parsed

    def test_V8_6_isinstance_dict_compat(self):
        """isinstance(adf.aliases, dict) returns True."""
        adf = _make_adf()
        assert isinstance(adf.aliases, dict)

    def test_V8_7_alias_dtypes_setitem_raises(self):
        """adf.alias_dtypes['x'] = dtype raises TypeError."""
        adf = _make_adf()
        with pytest.raises(TypeError, match="add_alias"):
            adf.alias_dtypes['a'] = np.float64
        # Also test del
        with pytest.raises(TypeError):
            del adf.alias_dtypes['a']
        # Read still works
        assert adf.alias_dtypes['a'] == np.float32
        assert isinstance(adf.alias_dtypes, dict)

    def test_V8_8_pickle_deepcopy_roundtrip(self):
        """pickle and deepcopy of read-only wrappers work correctly."""
        adf = _make_adf()

        # Wrapper-level pickle roundtrip (the actual contract)
        aliases_dict = adf.aliases
        restored = pickle.loads(pickle.dumps(aliases_dict))
        assert restored == {'a': 'x + 1', 'b': 'x * 2'}
        with pytest.raises(TypeError):
            restored['c'] = 'new'

        # Wrapper-level deepcopy
        copied = copy.deepcopy(aliases_dict)
        assert copied == {'a': 'x + 1', 'b': 'x * 2'}
        with pytest.raises(TypeError):
            copied['c'] = 'new'

        # alias_dtypes wrapper
        dtypes_dict = adf.alias_dtypes
        restored_dt = pickle.loads(pickle.dumps(dtypes_dict))
        assert restored_dt['a'] == np.float32
        with pytest.raises(TypeError):
            restored_dt['c'] = np.int8

    def test_V8_9_setdefault_existing_key_returns_value(self):
        """setdefault on existing key is a read — returns value, no error."""
        adf = _make_adf()
        # Existing key — read path, should return value
        val = adf.aliases.setdefault('a', None)
        assert val == 'x + 1'

        # Non-existing key — write path, should raise
        with pytest.raises(TypeError, match="add_alias"):
            adf.aliases.setdefault('nonexistent', 'x + 999')


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
