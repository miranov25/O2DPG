"""
Phase 13.24.ADF Part B — constant_aliases read-only contract tests.

V9_1-V9_3: verify set mutation raises TypeError, read paths work.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


def _make_adf_with_constants():
    df = pd.DataFrame({'x': np.array([1, 2, 3], dtype=np.float32)})
    adf = AliasDataFrame(df)
    adf.add_alias('pi_approx', '3.14159', is_constant=True)
    adf.add_alias('scale', 'x * 2', dtype=np.float32)  # not constant
    return adf


class TestV9ConstantAliasesImmutable:

    def test_V9_1_add_raises(self):
        """adf.constant_aliases.add('x') raises TypeError."""
        adf = _make_adf_with_constants()
        with pytest.raises(TypeError, match="add_alias"):
            adf.constant_aliases.add('injected')
        assert 'injected' not in adf.constant_aliases

    def test_V9_2_remove_discard_clear_raise(self):
        """All set-mutation methods raise TypeError."""
        adf = _make_adf_with_constants()
        ca = adf.constant_aliases

        with pytest.raises(TypeError):
            ca.remove('pi_approx')

        with pytest.raises(TypeError):
            ca.discard('pi_approx')

        with pytest.raises(TypeError):
            ca.clear()

        with pytest.raises(TypeError):
            ca.pop()

        with pytest.raises(TypeError):
            ca.update({'injected'})

        with pytest.raises(TypeError):
            ca.intersection_update({'pi_approx'})

        with pytest.raises(TypeError):
            ca.difference_update({'pi_approx'})

        with pytest.raises(TypeError):
            ca.symmetric_difference_update({'pi_approx'})

    def test_V9_3_set_read_paths_unchanged(self):
        """All set read operations work normally."""
        adf = _make_adf_with_constants()
        ca = adf.constant_aliases

        # Membership
        assert 'pi_approx' in ca
        assert 'scale' not in ca  # not constant

        # Length
        assert len(ca) == 1

        # Iteration
        items = list(ca)
        assert items == ['pi_approx']

        # isinstance
        assert isinstance(ca, set)

        # Set operations (non-mutating)
        assert ca & {'pi_approx', 'other'} == {'pi_approx'}
        assert ca | {'other'} == {'pi_approx', 'other'}
        assert ca - {'pi_approx'} == set()
        assert ca ^ {'pi_approx'} == set()


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
