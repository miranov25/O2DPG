"""
Phase 13.23.ADF — Multi-level dotted subframe expression resolution tests

N1_0:  Single-level backward compatibility (regression gate)
N1_1:  2-level Outer.Inner.val resolves correctly (vs pd.merge)
N1_2:  Roundtrip: export_tree → read_tree → eval (bit-exact)
N1_3:  Missing keys at each level → NaN (Safety)
N1_4:  dematerialize + re-materialize (bit-exact recovery)
N1_5:  Nested ref in compound expression
N1_6:  T.pt.round() method chain preserved
N1_7:  Cycle detection (self-ref + A→B→A)
N1_8:  3-level chain A.B.C.val
N1_9:  draw('Outer.Inner.val:x') auto-resolves
N1_10: add_alias('val', 'Outer.Inner.val') no false self-ref error

ENTRY POINT DISCIPLINE (Failure Mode #12):
Every test calls materialize_aliases(), export_tree/read_tree, draw(),
or add_alias() — never _prepare_subframe_joins directly.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame

try:
    import ROOT
    _HAS_ROOT = True
except ImportError:
    _HAS_ROOT = False

try:
    import uproot
    _HAS_UPROOT = True
except ImportError:
    _HAS_UPROOT = False

try:
    import matplotlib
    matplotlib.use('Agg')
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False


def _build_nested_adf():
    """Build a 2-level nested subframe ADF for testing."""
    main = pd.DataFrame({
        'key_a': np.array([1, 1, 2, 2, 3, 3], dtype=np.int16),
        'x': np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
    })
    outer = pd.DataFrame({
        'key_a': np.array([1, 2, 3], dtype=np.int16),
        'key_b': np.array([100, 200, 300], dtype=np.int16),
        'outer_val': np.array([1.0, 2.0, 3.0], dtype=np.float32),
    })
    inner = pd.DataFrame({
        'key_b': np.array([100, 200, 300], dtype=np.int16),
        'val': np.array([1000, 2000, 3000], dtype=np.float32),
    })
    adf = AliasDataFrame(main)
    sf_outer = AliasDataFrame(outer)
    sf_inner = AliasDataFrame(inner)
    sf_outer.register_subframe('Inner', sf_inner, index_columns=['key_b'])
    adf.register_subframe('Outer', sf_outer, index_columns=['key_a'])
    return adf


def _build_3level_adf():
    """Build a 3-level nested subframe ADF."""
    main = pd.DataFrame({
        'k1': np.array([1, 2, 3], dtype=np.int16),
        'x': np.array([10, 20, 30], dtype=np.float32),
    })
    level_a = pd.DataFrame({
        'k1': np.array([1, 2, 3], dtype=np.int16),
        'k2': np.array([10, 20, 30], dtype=np.int16),
    })
    level_b = pd.DataFrame({
        'k2': np.array([10, 20, 30], dtype=np.int16),
        'k3': np.array([100, 200, 300], dtype=np.int16),
    })
    level_c = pd.DataFrame({
        'k3': np.array([100, 200, 300], dtype=np.int16),
        'val': np.array([7.0, 8.0, 9.0], dtype=np.float32),
    })
    adf = AliasDataFrame(main)
    sf_a = AliasDataFrame(level_a)
    sf_b = AliasDataFrame(level_b)
    sf_c = AliasDataFrame(level_c)
    sf_b.register_subframe('C', sf_c, index_columns=['k3'])
    sf_a.register_subframe('B', sf_b, index_columns=['k2'])
    adf.register_subframe('A', sf_a, index_columns=['k1'])
    return adf


class TestN1MultiLevelResolution:
    """Phase 13.23.ADF — multi-level dotted subframe expression resolution."""

    @pytest.mark.invariance
    def test_N1_0_single_level_backward_compat(self):
        """
        N1_0: single-level T.col still works in mixed expression with multi-level.
        Regression gate — same behavior as Phase 13.22.
        """
        adf = _build_nested_adf()
        # Single-level: Outer.outer_val (direct column on Outer, not nested)
        adf.add_alias('single', 'Outer.outer_val', dtype=np.float32)
        adf.materialize_aliases(names=['single'])

        # Manual reference
        ref = adf.df.merge(
            pd.DataFrame({
                'key_a': np.array([1, 2, 3], dtype=np.int16),
                'outer_val': np.array([1.0, 2.0, 3.0], dtype=np.float32),
            }),
            on='key_a', how='left'
        )
        np.testing.assert_array_equal(
            adf.df['single'].values, ref['outer_val'].values,
            err_msg="N1_0: single-level resolution differs from pd.merge"
        )

    @pytest.mark.invariance
    def test_N1_1_two_level_resolution(self):
        """
        N1_1: Outer.Inner.val resolves through 2 join levels.
        Result matches independent pd.merge chain.
        """
        adf = _build_nested_adf()
        adf.add_alias('nested_val', 'Outer.Inner.val', dtype=np.float32)
        adf.materialize_aliases(names=['nested_val'])

        # Independent reference: manual pd.merge chain
        main = adf.df[['key_a', 'x']].copy()
        outer = pd.DataFrame({
            'key_a': np.array([1, 2, 3], dtype=np.int16),
            'key_b': np.array([100, 200, 300], dtype=np.int16),
        })
        inner = pd.DataFrame({
            'key_b': np.array([100, 200, 300], dtype=np.int16),
            'val': np.array([1000, 2000, 3000], dtype=np.float32),
        })
        ref = main.merge(outer, on='key_a', how='left').merge(inner, on='key_b', how='left')

        np.testing.assert_array_equal(
            adf.df['nested_val'].values, ref['val'].values,
            err_msg="N1_1: nested subframe resolution differs from pd.merge chain"
        )

    @pytest.mark.skipif(not _HAS_ROOT or not _HAS_UPROOT,
                        reason="Requires ROOT + uproot")
    @pytest.mark.invariance
    def test_N1_2_nested_roundtrip(self):
        """
        N1_2: export_tree → read_tree preserves nested subframes,
        alias eval on restored ADF produces bit-exact same values.
        """
        adf = _build_nested_adf()
        adf.add_alias('nested_val', 'Outer.Inner.val', dtype=np.float32)
        adf.materialize_aliases(names=['nested_val'])
        before = adf.df['nested_val'].values.copy()

        with tempfile.NamedTemporaryFile(suffix='.root', delete=False) as f:
            tmpfile = f.name
        try:
            adf.export_tree(tmpfile, 'tree')
            adf2 = AliasDataFrame.read_tree(tmpfile, treename='tree')
            adf2.materialize_aliases(names=['nested_val'])
            after = adf2.df['nested_val'].values

            np.testing.assert_array_equal(
                before, after,
                err_msg="N1_2: nested alias differs after roundtrip"
            )
        finally:
            os.unlink(tmpfile)

    @pytest.mark.invariance
    def test_N1_3_missing_keys_nan(self):
        """
        N1_3: missing key at either level → NaN.
        Safety Hard Constraint: never zero, never a neighbor value.

        Subcases:
        (a) both hit → correct value
        (b) outer hit, inner miss → NaN
        (c) outer miss → NaN
        (d) mixed — no cross-contamination
        """
        main = pd.DataFrame({
            'key_a': np.array([1, 2, 3, 999], dtype=np.int16),
            'x': np.array([10, 20, 30, 40], dtype=np.float32),
        })
        outer = pd.DataFrame({
            'key_a': np.array([1, 2], dtype=np.int16),  # key_a=3,999 missing
            'key_b': np.array([100, 200], dtype=np.int16),
        })
        inner = pd.DataFrame({
            'key_b': np.array([100], dtype=np.int16),  # key_b=200 missing
            'val': np.array([1000], dtype=np.float32),
        })

        adf = AliasDataFrame(main)
        sf_outer = AliasDataFrame(outer)
        sf_inner = AliasDataFrame(inner)
        sf_outer.register_subframe('Inner', sf_inner, index_columns=['key_b'])
        adf.register_subframe('Outer', sf_outer, index_columns=['key_a'])

        adf.add_alias('nested_val', 'Outer.Inner.val', dtype=np.float32)
        adf.materialize_aliases(names=['nested_val'])

        result = adf.df['nested_val'].values
        # (a) key_a=1 → key_b=100 → val=1000
        assert result[0] == 1000.0, f"(a) expected 1000, got {result[0]}"
        # (b) key_a=2 → key_b=200 → inner miss → NaN
        assert np.isnan(result[1]), f"(b) expected NaN, got {result[1]}"
        # (c) key_a=3 → outer miss → NaN
        assert np.isnan(result[2]), f"(c) expected NaN, got {result[2]}"
        # (d) key_a=999 → outer miss → NaN
        assert np.isnan(result[3]), f"(d) expected NaN, got {result[3]}"

    @pytest.mark.invariance
    def test_N1_4_dematerialize_then_recover(self):
        """
        N1_4: dematerialize(drop=[...]) then re-materialize produces
        bit-exact same values. Verifies join caching composes with
        nested subframe resolution.
        """
        adf = _build_nested_adf()
        adf.add_alias('nested_val', 'Outer.Inner.val', dtype=np.float32)
        adf.materialize_aliases(names=['nested_val'])
        before = adf.df['nested_val'].values.copy()

        adf.dematerialize(drop=['nested_val'])
        assert 'nested_val' not in adf.df.columns

        adf.materialize_aliases(names=['nested_val'])
        after = adf.df['nested_val'].values

        np.testing.assert_array_equal(
            before, after,
            err_msg="N1_4: nested alias differs after dematerialize + re-materialize"
        )

    @pytest.mark.invariance
    def test_N1_5_nested_in_compound_expression(self):
        """
        N1_5: nested subframe column in compound alias expression.
        """
        adf = _build_nested_adf()
        adf.add_alias('combined', 'x + Outer.Inner.val * 2', dtype=np.float32)
        adf.materialize_aliases(names=['combined'])

        # Manual: val=[1000,1000,2000,2000,3000,3000], x=[10,20,30,40,50,60]
        expected = adf.df['x'].values + np.array(
            [1000, 1000, 2000, 2000, 3000, 3000], dtype=np.float32
        ) * 2

        np.testing.assert_array_equal(
            adf.df['combined'].values, expected,
            err_msg="N1_5: nested in compound expression failed"
        )

    @pytest.mark.invariance
    def test_N1_6_method_chain_preserved(self):
        """
        N1_6: T.col.method() — walk stops at col (leaf),
        .method() preserved as pandas method suffix.
        """
        main = pd.DataFrame({
            'key': np.array([1, 2, 3], dtype=np.int16),
            'x': np.array([1.4, 2.6, 3.5], dtype=np.float32),
        })
        sub = pd.DataFrame({
            'key': np.array([1, 2, 3], dtype=np.int16),
            'val': np.array([1.7, 2.3, 3.9], dtype=np.float32),
        })
        adf = AliasDataFrame(main)
        adf.register_subframe('T', AliasDataFrame(sub), index_columns=['key'])
        # T.val exists; round is a pandas method, not a subframe
        adf.add_alias('rounded', 'T.val.round()', dtype=np.float32)
        adf.materialize_aliases(names=['rounded'])

        expected = np.array([2.0, 2.0, 4.0], dtype=np.float32)
        np.testing.assert_array_equal(
            adf.df['rounded'].values, expected,
            err_msg="N1_6: method chain T.val.round() failed"
        )

    @pytest.mark.invariance
    def test_N1_7_cycle_detection(self):
        """
        N1_7: circular subframe registration raises ValueError.
        (a) Self-referential: adf registers itself
        (b) A→B→A cycle
        Neither should hang.
        """
        # (a) Self-ref
        main = pd.DataFrame({
            'k': np.array([1, 2], dtype=np.int16),
            'x': np.array([10, 20], dtype=np.float32),
        })
        adf = AliasDataFrame(main)
        adf.register_subframe('Self', adf, index_columns=['k'])
        adf.add_alias('bad', 'Self.Self.x', dtype=np.float32)
        with pytest.raises(ValueError, match="[Cc]ycle"):
            adf.materialize_aliases(names=['bad'])

        # (b) A→B→A
        a_df = pd.DataFrame({'k': [1], 'v': [10.0]})
        b_df = pd.DataFrame({'k': [1], 'w': [20.0]})
        adf_a = AliasDataFrame(a_df)
        adf_b = AliasDataFrame(b_df)
        adf_a.register_subframe('B', adf_b, index_columns=['k'])
        adf_b.register_subframe('A', adf_a, index_columns=['k'])
        adf_a.add_alias('cycle_test', 'B.A.v', dtype=np.float32)
        with pytest.raises(ValueError, match="[Cc]ycle"):
            adf_a.materialize_aliases(names=['cycle_test'])

    @pytest.mark.invariance
    def test_N1_8_three_level_chain(self):
        """
        N1_8: 3-level chain A.B.C.val resolves correctly.
        Column named val__C__B__A.
        """
        adf = _build_3level_adf()
        adf.add_alias('deep', 'A.B.C.val', dtype=np.float32)
        adf.materialize_aliases(names=['deep'])

        expected = np.array([7.0, 8.0, 9.0], dtype=np.float32)
        np.testing.assert_array_equal(
            adf.df['deep'].values, expected,
            err_msg="N1_8: 3-level chain A.B.C.val failed"
        )

    @pytest.mark.skipif(not _HAS_MPL, reason="Requires matplotlib")
    @pytest.mark.invariance
    def test_N1_9_draw_nested_subframe(self):
        """
        N1_9: adf.draw('Outer.Inner.val:x') auto-resolves through
        nested path. Verifies draw resolver handles multi-level.
        """
        adf = _build_nested_adf()
        fig = adf.draw('Outer.Inner.val:x')
        assert fig is not None, "N1_9: draw with nested subframe ref should not fail"

    @pytest.mark.invariance
    def test_N1_10_add_alias_no_false_self_ref(self):
        """
        N1_10: add_alias('val', 'Outer.Inner.val') does NOT raise
        self-reference error. The 'val' in 'Outer.Inner.val' is a
        subframe column, not a reference to the alias being defined.
        """
        adf = _build_nested_adf()
        # This should NOT raise ValueError("Alias 'val' would reference itself")
        try:
            adf.add_alias('val', 'Outer.Inner.val', dtype=np.float32)
        except ValueError as e:
            if 'reference itself' in str(e):
                pytest.fail(
                    f"N1_10: add_alias raised false self-reference error: {e}"
                )
            raise  # re-raise if it's a different ValueError

        adf.materialize_aliases(names=['val'])
        expected = np.array([1000, 1000, 2000, 2000, 3000, 3000], dtype=np.float32)
        np.testing.assert_array_equal(
            adf.df['val'].values, expected,
            err_msg="N1_10: nested alias 'val' produced wrong values"
        )


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
