"""
Batch 2 — I10: Lazy ≡ Eager Pipeline Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE (§5.1 deviation note, established in Batch 1).
All tests are marked @pytest.mark.invariance.

PATH-EXPLICIT DISCIPLINE (Failure Mode #11)
-------------------------------------------
read_tree (line 4864) is the eager path; read_tree_lazy (line 5221)
is the lazy path. Both are public static methods. Tests use both
explicitly with the same ROOT file written by export_tree.

ROOT-DEPENDENT
--------------
These tests require ROOT to write/read files. They use the same
test_draw_invariance.py pattern: skip if ROOT unavailable.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame

try:
    import ROOT
    _HAS_ROOT = ROOT is not None
except ImportError:
    _HAS_ROOT = False

_requires_root = pytest.mark.skipif(
    not _HAS_ROOT, reason="ROOT not available"
)


@pytest.fixture
def small_root_file(tmp_path):
    """Write a small ADF to a ROOT file and return path + reference DF."""
    df = pd.DataFrame({
        'a': np.arange(100, dtype=np.int32),
        'b': np.linspace(0.0, 1.0, 100).astype(np.float32),
        'c': np.linspace(-1.0, 1.0, 100).astype(np.float64),
    })
    adf = AliasDataFrame(df.copy())
    path = str(tmp_path / 'i10_small.root')
    adf.export_tree(path, treename='tree')
    return path, df


class TestI10LazyEagerInvariance:
    """Phase 13.12 I10 — Lazy ≡ Eager Pipeline."""

    @_requires_root
    @pytest.mark.invariance
    def test_I10_1_read_tree_eager_equals_read_tree_lazy(
        self, small_root_file
    ):
        """
        I10_1 INVARIANT:
            read_tree() (eager) and read_tree_lazy() + ensure_branches()
            (lazy with explicit branch loading) produce DataFrames with
            byte-identical integer columns and within-tolerance float
            columns. Row order preserved.
        CODE PATH:
            Path A: AliasDataFrame.read_tree(path) — eager static
            Path B: AliasDataFrame.read_tree_lazy(path, 'tree') +
                    ensure_branches(['a','b','c'])
            Both explicit, no defaults.
        PRODUCTION ENTRY POINT:
            adf_eager = AliasDataFrame.read_tree(path)
            adf_lazy = AliasDataFrame.read_tree_lazy(path, 'tree')
            adf_lazy.ensure_branches(['a','b','c'])
        REGRESSION GUARD: lazy vs eager byte equivalence for primitive
                          dtype columns.
        """
        path, df_ref = small_root_file

        # Path A: eager
        adf_eager = AliasDataFrame.read_tree(path, treename='tree')

        # Path B: lazy with explicit branch loading
        adf_lazy = AliasDataFrame.read_tree_lazy(path, 'tree')
        adf_lazy.ensure_branches(['a', 'b', 'c'])

        # Integer column: exact equality
        np.testing.assert_array_equal(
            adf_eager.df['a'].values, adf_lazy.df['a'].values,
            err_msg="I10_1: int32 column 'a' differs eager vs lazy"
        )

        # float32: rtol=1e-6 per §5.6
        np.testing.assert_allclose(
            adf_eager.df['b'].values, adf_lazy.df['b'].values,
            rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="I10_1: float32 column 'b' differs eager vs lazy"
        )

        # float64: rtol=1e-12 per §5.6
        np.testing.assert_allclose(
            adf_eager.df['c'].values, adf_lazy.df['c'].values,
            rtol=1e-12, atol=1e-15, equal_nan=True,
            err_msg="I10_1: float64 column 'c' differs eager vs lazy"
        )

    @_requires_root
    @pytest.mark.invariance
    def test_I10_2_explicit_ensure_branches_equals_implicit_access(
        self, small_root_file
    ):
        """
        I10_2 INVARIANT:
            ensure_branches(['a','b']) followed by .df access produces
            the same column values as implicit lazy auto-load via
            adf['a']; adf['b'] one-by-one access (which triggers
            on-access loading).
        CODE PATH:
            Path A: read_tree_lazy + ensure_branches(['a','b']) + .df['a']
            Path B: read_tree_lazy + adf['a'] + adf['b']  (implicit auto-load)
        PRODUCTION ENTRY POINT:
            Both ensure_branches and adf['col'] access are public API.
        REGRESSION GUARD: explicit batch vs implicit on-access loading.
        """
        path, df_ref = small_root_file

        # Path A: explicit ensure_branches batch
        adf_explicit = AliasDataFrame.read_tree_lazy(path, 'tree')
        adf_explicit.ensure_branches(['a', 'b'])
        a_explicit = adf_explicit.df['a'].values.copy()
        b_explicit = adf_explicit.df['b'].values.copy()

        # Path B: implicit on-access loading via adf['col']
        adf_implicit = AliasDataFrame.read_tree_lazy(path, 'tree')
        a_implicit = adf_implicit['a']
        b_implicit = adf_implicit['b']
        # Convert if returned as Series
        if hasattr(a_implicit, 'values'):
            a_implicit = a_implicit.values
        if hasattr(b_implicit, 'values'):
            b_implicit = b_implicit.values

        np.testing.assert_array_equal(
            a_explicit, np.asarray(a_implicit),
            err_msg="I10_2: int32 'a' explicit vs implicit auto-load"
        )
        np.testing.assert_allclose(
            b_explicit, np.asarray(b_implicit),
            rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="I10_2: float32 'b' explicit vs implicit auto-load"
        )
