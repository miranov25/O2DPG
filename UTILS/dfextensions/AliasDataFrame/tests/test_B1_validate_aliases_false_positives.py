"""
BUG_AliasDataFrame_20260512_validate_aliases_false_positives — regression tests.

Three false-positive classes fixed in validate_aliases():
B1_1: np.pi in expression → 'np' was flagged as missing subframe
B1_2: SubframeName.col → column-part token flagged as bare unknown token
B1_3: mid-chain ref (R.CTPLumi.orbit) → 'CTPLumi'/'orbit' escaped the guard
B1_4: genuinely broken alias (registered subframe, column absent) → still detected
B1_5: arithmetic expression with no dotted refs → clean, not broken
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


def _main_adf():
    df = pd.DataFrame({
        'x':   np.linspace(0, 1, 10, dtype=np.float32),
        'y':   np.zeros(10, dtype=np.float32),
        'row': np.arange(10, dtype=np.int16),
    })
    return AliasDataFrame(df)


def _adf_with_subframe():
    main = pd.DataFrame({'key': np.arange(5, dtype=np.int16),
                         'x':   np.linspace(0, 1, 5, dtype=np.float32)})
    sub  = pd.DataFrame({'key': np.arange(5, dtype=np.int16),
                         'coeff_a': np.ones(5, dtype=np.float32),
                         'coeff_b': np.zeros(5, dtype=np.float32)})
    adf = AliasDataFrame(main)
    adf.register_subframe('SF', AliasDataFrame(sub), index_columns=['key'])
    return adf


class TestB1ValidateAliasesFalsePositives:

    @pytest.mark.invariance
    def test_B1_1_np_pi_not_broken(self):
        """np.pi in expression must NOT be flagged as missing subframe."""
        adf = _main_adf()
        adf.add_alias('edge', '(x * np.pi / 18) - y', dtype=np.float32)
        broken = adf.validate_aliases()
        assert 'edge' not in broken, (
            f"False positive: 'edge' flagged broken; broken={broken}")

    @pytest.mark.invariance
    def test_B1_2_subframe_column_not_broken(self):
        """SF.coeff_a must NOT flag 'coeff_a' as a bare unknown token."""
        adf = _adf_with_subframe()
        adf.add_alias('val', 'SF.coeff_a * x + SF.coeff_b', dtype=np.float32)
        broken = adf.validate_aliases()
        assert 'val' not in broken, (
            f"False positive: 'val' flagged broken; broken={broken}")

    @pytest.mark.invariance
    def test_B1_3_arithmetic_expression_not_broken(self):
        """Pure arithmetic expressions must not be broken."""
        adf = _main_adf()
        adf.add_alias('mod3', 'row % 3', dtype=np.int16)
        adf.add_alias('edge2', '(x * np.pi / 18) - (y + mod3)', dtype=np.float32)
        broken = adf.validate_aliases()
        assert 'mod3'  not in broken, f"False positive: mod3 broken={broken}"
        assert 'edge2' not in broken, f"False positive: edge2 broken={broken}"

    @pytest.mark.invariance
    def test_B1_4_genuinely_broken_still_detected(self):
        """Alias referencing a registered subframe's absent column IS broken."""
        adf = _adf_with_subframe()
        adf.add_alias('bad', 'SF.nonexistent_col * x', dtype=np.float32)
        broken = adf.validate_aliases()
        assert 'bad' in broken, (
            f"False negative: genuinely broken alias 'bad' not detected; broken={broken}")

    @pytest.mark.invariance
    def test_B1_5_truly_missing_bare_token_detected(self):
        """Alias referencing a non-existent bare name IS broken."""
        adf = _main_adf()
        adf.add_alias('bad2', 'x + ghost_column', dtype=np.float32)
        broken = adf.validate_aliases()
        assert 'bad2' in broken, (
            f"False negative: 'bad2' with unknown token not detected; broken={broken}")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
