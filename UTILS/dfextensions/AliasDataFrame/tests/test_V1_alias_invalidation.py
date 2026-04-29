"""
BUG_AliasDataFrame_20260427_alias_invalidation — Regression test

BUG: add_alias() with a new expression does not drop the old materialized
column from self.df. Stale values persist silently. Aliases that depend
on the changed alias also keep their stale materialized values.

FIX: _invalidate_alias_cascade() drops materialized column for the
changed alias + all transitive dependents via reverse dependency BFS.

ENTRY POINT: add_alias() + materialize_aliases() — production path.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


class TestAliasInvalidation:
    """Tests for BUG_AliasDataFrame_20260427_alias_invalidation."""

    @pytest.mark.invariance
    def test_V1_redefine_alias_drops_stale_column(self):
        """
        V1: redefining an alias drops the old materialized column.

        Before fix: df['offset'] keeps old value (1.0) after expression change.
        After fix: df['offset'] is dropped; next materialize gets new value (100.0).
        """
        df = pd.DataFrame({'x': np.array([1, 2, 3], dtype=np.float32)})
        adf = AliasDataFrame(df)

        adf.add_alias('offset', 'x + 1', dtype=np.float32)
        adf.materialize_aliases(names=['offset'])
        assert adf.df['offset'].iloc[0] == 2.0  # x[0]=1 + 1 = 2

        # Redefine with different expression
        adf.add_alias('offset', 'x + 100', dtype=np.float32)

        # Column should be dropped (invalidated)
        assert 'offset' not in adf.df.columns, (
            "V1: stale materialized column not dropped after add_alias redefine"
        )

        # Re-materialize gets new value
        adf.materialize_aliases(names=['offset'])
        assert adf.df['offset'].iloc[0] == 101.0, (
            f"V1: expected 101.0, got {adf.df['offset'].iloc[0]}"
        )

    @pytest.mark.invariance
    def test_V2_cascade_invalidates_dependents(self):
        """
        V2: redefining alias A also drops materialized aliases that depend on A.

        Chain: result = offset + 10, offset = x + 1.
        Redefine offset → result must also be invalidated.
        """
        df = pd.DataFrame({'x': np.array([1, 2, 3], dtype=np.float32)})
        adf = AliasDataFrame(df)

        adf.add_alias('offset', 'x + 1', dtype=np.float32)
        adf.add_alias('result', 'offset + 10', dtype=np.float32)
        adf.materialize_aliases(names=['offset', 'result'])
        assert adf.df['result'].iloc[0] == 12.0  # (1+1)+10 = 12

        # Redefine the upstream alias
        adf.add_alias('offset', 'x + 100', dtype=np.float32)

        # Both should be dropped
        assert 'offset' not in adf.df.columns, "V2: offset not dropped"
        assert 'result' not in adf.df.columns, "V2: dependent 'result' not dropped"

        # Re-materialize gets correct new values
        adf.materialize_aliases(names=['offset', 'result'])
        assert adf.df['result'].iloc[0] == 111.0, (
            f"V2: expected 111.0, got {adf.df['result'].iloc[0]}"
        )

    @pytest.mark.invariance
    def test_V3_deep_cascade_3_levels(self):
        """
        V3: 3-level dependency chain — changing root invalidates all.

        Chain: c = b + 1, b = a + 1, a = x + 1.
        Redefine a → b and c must also be invalidated.
        """
        df = pd.DataFrame({'x': np.array([10], dtype=np.float32)})
        adf = AliasDataFrame(df)

        adf.add_alias('a', 'x + 1', dtype=np.float32)
        adf.add_alias('b', 'a + 1', dtype=np.float32)
        adf.add_alias('c', 'b + 1', dtype=np.float32)
        adf.materialize_aliases(names=['a', 'b', 'c'])
        assert adf.df['c'].iloc[0] == 13.0  # (10+1)+1+1 = 13

        # Redefine root
        adf.add_alias('a', 'x + 1000', dtype=np.float32)

        assert 'a' not in adf.df.columns, "V3: a not dropped"
        assert 'b' not in adf.df.columns, "V3: b not dropped"
        assert 'c' not in adf.df.columns, "V3: c not dropped"

        adf.materialize_aliases(names=['a', 'b', 'c'])
        assert adf.df['c'].iloc[0] == 1012.0, (
            f"V3: expected 1012.0, got {adf.df['c'].iloc[0]}"
        )

    @pytest.mark.invariance
    def test_V4_unrelated_alias_not_dropped(self):
        """
        V4: redefining one alias does NOT drop unrelated materialized aliases.
        """
        df = pd.DataFrame({'x': np.array([1, 2], dtype=np.float32)})
        adf = AliasDataFrame(df)

        adf.add_alias('offset', 'x + 1', dtype=np.float32)
        adf.add_alias('scale', 'x * 10', dtype=np.float32)
        adf.materialize_aliases(names=['offset', 'scale'])

        # Redefine offset — scale is unrelated
        adf.add_alias('offset', 'x + 999', dtype=np.float32)

        assert 'offset' not in adf.df.columns, "V4: offset not dropped"
        assert 'scale' in adf.df.columns, "V4: unrelated 'scale' was incorrectly dropped"
        assert adf.df['scale'].iloc[0] == 10.0, "V4: scale value changed"

    @pytest.mark.invariance
    def test_V5_raw_column_never_dropped(self):
        """
        V5: invalidation never drops raw (physical) columns, even if
        an alias has the same name pattern.
        """
        df = pd.DataFrame({
            'x': np.array([1, 2, 3], dtype=np.float32),
            'y': np.array([10, 20, 30], dtype=np.float32),
        })
        adf = AliasDataFrame(df)

        adf.add_alias('offset', 'x + y', dtype=np.float32)
        adf.materialize_aliases(names=['offset'])

        adf.add_alias('offset', 'x - y', dtype=np.float32)

        # Raw columns must survive
        assert 'x' in adf.df.columns, "V5: raw column 'x' was dropped"
        assert 'y' in adf.df.columns, "V5: raw column 'y' was dropped"

    @pytest.mark.invariance
    def test_V6_new_alias_no_invalidation(self):
        """
        V6: adding a NEW alias (not redefining) does not drop anything.
        """
        df = pd.DataFrame({'x': np.array([1, 2], dtype=np.float32)})
        adf = AliasDataFrame(df)

        adf.add_alias('a', 'x + 1', dtype=np.float32)
        adf.materialize_aliases(names=['a'])

        # Add a completely new alias — a should survive
        adf.add_alias('b', 'x + 2', dtype=np.float32)

        assert 'a' in adf.df.columns, "V6: existing alias dropped when adding new one"

    @pytest.mark.invariance
    def test_V7_production_pattern_iterative_calibration(self):
        """
        V7: production pattern — iterative recalibration.

        Simulates: compute correction, materialize, update coefficients,
        redefine correction, re-materialize. The second materialize must
        use the new expression, not stale values.
        """
        rng = np.random.default_rng(427)
        n = 100
        df = pd.DataFrame({
            'x': rng.uniform(0, 10, n).astype(np.float32),
            'sec': rng.integers(0, 5, n).astype(np.int8),
        })
        adf = AliasDataFrame(df)

        # Iteration 0: offset = 0
        sf0 = AliasDataFrame(pd.DataFrame({
            'sec': np.arange(5, dtype=np.int8),
            'c0': np.zeros(5, dtype=np.float32),
        }))
        adf.register_subframe('Coeff', sf0, index_columns=['sec'])
        adf.add_alias('correction', 'Coeff.c0', dtype=np.float32)
        adf.materialize_aliases(names=['correction'])
        assert np.allclose(adf.df['correction'].values, 0.0), "V7: iter0 should be 0"

        # Iteration 1: new coefficients
        sf1 = AliasDataFrame(pd.DataFrame({
            'sec': np.arange(5, dtype=np.int8),
            'c0': np.array([1, 2, 3, 4, 5], dtype=np.float32),
        }))
        adf.register_subframe('Coeff', sf1, index_columns=['sec'])
        adf.add_alias('correction', 'Coeff.c0', dtype=np.float32)

        # Stale column must be gone
        assert 'correction' not in adf.df.columns, "V7: stale correction not dropped"

        adf.materialize_aliases(names=['correction'])
        # sec=0 → c0=1, sec=1 → c0=2, etc.
        for s in range(5):
            mask = adf.df['sec'] == s
            if mask.any():
                val = adf.df.loc[mask, 'correction'].iloc[0]
                assert val == float(s + 1), (
                    f"V7: sec={s} expected {s+1}, got {val}"
                )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
