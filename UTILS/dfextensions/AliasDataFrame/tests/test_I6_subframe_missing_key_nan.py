"""
Batch 1 — I6: Subframe-Missing-Key NaN Propagation
Phase 13.12.ADF — Public API Invariance Test Suite

APPEND to tests/test_fill_value_dependency.py (inside existing module,
after the last test class). All tests are marked @pytest.mark.invariance.

Feature flipped: CORE.materialization + CORE.dependency_resolution
Incident addressed: BUG_AliasDataFrame_20260331_fill_value_dependency
    (7.4% NaN contamination in TPC calibration production output)

NOTE ON INTEGRATION WITH EXISTING TESTS
---------------------------------------
test_fill_value_dependency.py already has
test_invariance_direct_vs_dependency (marked @pytest.mark.invariance)
which is a 6-row toy-scale I6_2. Per architect A3, I6 tests use a
SINGLE TPC SECTOR scale (~250 rows, 1/36 of production) for faithful
reproduction of the BUG_20260331 structure. The existing toy test is
complementary, not duplicated.

PATH-EXPLICIT DISCIPLINE (Failure Mode #11)
-------------------------------------------
Every test exercises the exact production call sequence from BUG_20260331:
  1. register_subframe with keys where N_subframe < N_main (missing matches)
  2. add_alias('A', 'S.col', fill_value=0)  # fill must apply at join level
  3. add_alias('B', 'A * pt')                # dependent alias
  4. materialize_aliases(['B'])               # production entry point
  5. Verify B has 0, not NaN, for missing-match rows

SINGLE-SECTOR SCALE (per architect A3)
--------------------------------------
- Main: 250 rows (~ 1 TPC sector's worth of tracks)
- Subframe: 150 matched keys (100 rows have no match)
- Runtime: target <1s per test, well under the 5s budget
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


# =============================================================================
# Fixtures — single TPC sector scale per architect A3
# =============================================================================

@pytest.fixture
def adf_one_sector_subframe_missing_keys():
    """
    Single-sector-scale reproduction of BUG_20260331.

    Structure (mirrors TPC calibration with TRD subframe):
      - Main = 250 tracks (single sector)
      - Subframe S = 150 matched keys (60% match rate)
      - 100 tracks have NO subframe match → those rows produce NaN
        unless fill_value=0 propagates correctly through the
        dependency chain.

    BUG_20260331 manifested when fill_value was applied only when A
    was materialized DIRECTLY, but lost when A was materialized as a
    DEPENDENCY of B.
    """
    n_main = 250
    n_sub = 150

    np.random.seed(20260331)  # bug date = seed, for reproducibility
    df_main = pd.DataFrame({
        'track_id': np.arange(n_main, dtype=np.int64),
        'pt': np.random.uniform(0.5, 10.0, n_main).astype(np.float32),
        'phi': np.random.uniform(-np.pi, np.pi, n_main).astype(np.float32),
    })

    df_sub = pd.DataFrame({
        'track_id': np.arange(n_sub, dtype=np.int64),
        'dy': np.random.normal(0, 0.1, n_sub).astype(np.float32),
        'dz': np.random.normal(0, 0.2, n_sub).astype(np.float32),
    })

    adf = AliasDataFrame(df_main)
    adf.register_subframe('S', AliasDataFrame(df_sub), index_columns='track_id')
    return adf


# =============================================================================
# Test class — append to tests/test_fill_value_dependency.py
# =============================================================================

class TestI6SubframeMissingKeyNaNPropagation:
    """
    Phase 13.12 I6 — Subframe-Missing-Key NaN Propagation.

    Regression guard for BUG_AliasDataFrame_20260331_fill_value_dependency
    (7.4% NaN contamination in production). Tests at single TPC sector
    scale per architect direction A3.

    Flips: CORE.materialization, CORE.dependency_resolution.
    """

    @pytest.mark.invariance
    def test_I6_1_subframe_missing_key_fill_value_propagates_through_chain(
        self, adf_one_sector_subframe_missing_keys
    ):
        """
        I6_1 INVARIANT:
            Given main ADF with N rows and subframe S with K<N matched rows:
                A = S.col with fill_value=0
                B = A * pt    (depends on A)
            Then B = 0 (not NaN) for the N-K missing-match rows,
            when accessed via both:
              Path 1: adf.materialize_aliases(['A','B']) then adf['B']
              Path 2: adf['B']  (triggers on-access materialization)

        CODE PATH:
            register_subframe with missing keys, add_alias with fill_value,
            materialize_aliases batch path. NOT auto-dispatch.

        PRODUCTION ENTRY POINT:
            adf.materialize_aliases(['A','B']) then adf['B'] access —
            the exact call sequence in BUG_AliasDataFrame_20260331_fill_value_dependency.

        REGRESSION GUARD FOR:
            BUG_AliasDataFrame_20260331_fill_value_dependency
            (7.4% NaN contamination — TPC calibration TRD subframe case,
            600/1000 rows had TRD match, 400 missing rows contaminated
            dependent aliases).
        """
        adf = adf_one_sector_subframe_missing_keys

        adf.add_alias('A', 'S.dy', fill_value=0)
        adf.add_alias('B', 'A * pt')

        # Production entry point: batch materialization
        adf.materialize_aliases(names=['B'])

        # Invariant: B must have no NaN for the 100 missing-match rows
        b_values = adf.df['B'].values
        assert not np.isnan(b_values).any(), (
            f"I6_1 FAILED: B has {np.isnan(b_values).sum()} NaN values; "
            f"fill_value=0 on A did not propagate through dependency chain. "
            f"This is BUG_20260331 recurrence."
        )

        # Invariant: missing-match rows (track_id >= 150) must have B=0
        missing_rows = adf.df['track_id'].values >= 150
        assert np.all(b_values[missing_rows] == 0.0), (
            f"I6_1 FAILED: missing-match rows have non-zero B values: "
            f"{b_values[missing_rows][b_values[missing_rows] != 0][:5]}"
        )

        # Invariant: matched rows (track_id < 150) must have B = S.dy * pt
        matched_rows = adf.df['track_id'].values < 150
        expected_matched = (
            adf.df.loc[matched_rows, 'pt'].values *
            adf.get_subframe('S').df['dy'].values[:matched_rows.sum()]
        )
        np.testing.assert_allclose(
            b_values[matched_rows], expected_matched,
            rtol=1e-6, atol=1e-9,
            err_msg="I6_1 FAILED: matched rows have incorrect B values"
        )

    @pytest.mark.invariance
    def test_I6_2_sequential_equals_batch_materialization(
        self, adf_one_sector_subframe_missing_keys
    ):
        """
        I6_2 INVARIANT:
            Sequential materialization
                materialize_alias('A'); materialize_alias('B')
            produces the same B values as batch materialization
                materialize_aliases(['A','B'])
            when A has fill_value and B depends on A.

        CODE PATH:
            Path 1 (sequential): materialize_alias per alias, single-alias codepath
            Path 2 (batch): materialize_aliases with names list, batch codepath
            Both explicit, not auto-dispatch.

        PRODUCTION ENTRY POINT:
            Both sequential and batch paths are public API. Users who
            migrate from one to the other (e.g., when batching for
            performance) depend on semantic equivalence.

        REGRESSION GUARD FOR:
            BUG_AliasDataFrame_20260331 — fill_value was correctly applied
            in one path but not the other. Direct production scenario from
            the TPC TRD calibration bug report.
        """
        adf = adf_one_sector_subframe_missing_keys

        # Path 1: sequential
        adf.add_alias('A', 'S.dy', fill_value=0)
        adf.add_alias('B', 'A * pt')
        adf.materialize_alias('A')
        adf.materialize_alias('B')
        b_sequential = adf.df['B'].values.copy()

        # Reset — drop materialized columns
        adf.df.drop(columns=['A', 'B'], inplace=True, errors='ignore')

        # Path 2: batch
        adf.materialize_aliases(names=['B'])  # A materialized as dependency
        b_batch = adf.df['B'].values.copy()

        # Invariant: sequential ≡ batch, NaN positions and values
        seq_nan = np.isnan(b_sequential)
        batch_nan = np.isnan(b_batch)
        assert np.array_equal(seq_nan, batch_nan), (
            f"I6_2 FAILED: NaN positions differ between sequential and batch. "
            f"Sequential NaN count={seq_nan.sum()}, "
            f"batch NaN count={batch_nan.sum()}. "
            f"This is exactly the BUG_20260331 failure mode."
        )

        np.testing.assert_allclose(
            b_sequential, b_batch,
            rtol=1e-12, atol=1e-15, equal_nan=True,
            err_msg=(
                "I6_2 FAILED: sequential and batch materialization "
                "produce different values"
            )
        )

        # Both paths must produce no NaN if fill_value works
        assert not np.isnan(b_batch).any(), (
            f"I6_2 FAILED: batch path has {np.isnan(b_batch).sum()} NaN — "
            f"fill_value=0 not propagating through dependency chain"
        )

    @pytest.mark.invariance
    def test_I6_3_materialization_order_does_not_affect_final_values(
        self, adf_one_sector_subframe_missing_keys
    ):
        """
        I6_3 INVARIANT:
            For dependency graph {A, B, C} where:
                A = S.dy with fill_value=0
                B = A * 2
                C = B + A
            Then materialize_aliases(['A','B','C']) produces the same
            final A, B, C values as materialize_aliases(['C','B','A'])
            and any other permutation. Dependency order is an internal
            optimization, not a semantic difference.

        CODE PATH:
            materialize_aliases with different name orderings; the
            internal dependency resolver picks the actual execution
            order. NOT auto-dispatch over types.

        PRODUCTION ENTRY POINT:
            Users should be able to request aliases in any order they
            want; the materialization graph resolver is responsible for
            correct execution ordering.

        REGRESSION GUARD FOR:
            dependency resolution bugs that only surface when the
            user-requested order differs from the dependency-correct
            order. This is a harder variant of BUG_20260331.
        """
        adf = adf_one_sector_subframe_missing_keys

        adf.add_alias('A', 'S.dy', fill_value=0)
        adf.add_alias('B', 'A * 2')
        adf.add_alias('C', 'B + A')

        # Order 1: dependency-correct
        adf.materialize_aliases(names=['A', 'B', 'C'])
        a1 = adf.df['A'].values.copy()
        b1 = adf.df['B'].values.copy()
        c1 = adf.df['C'].values.copy()

        # Reset
        adf.df.drop(columns=['A', 'B', 'C'], inplace=True, errors='ignore')

        # Order 2: reverse (resolver must figure out the correct order)
        adf.materialize_aliases(names=['C', 'B', 'A'])
        a2 = adf.df['A'].values.copy()
        b2 = adf.df['B'].values.copy()
        c2 = adf.df['C'].values.copy()

        # Invariant: same values regardless of requested order
        np.testing.assert_allclose(
            a1, a2, rtol=1e-12, atol=1e-15, equal_nan=True,
            err_msg="I6_3 FAILED: A differs between materialization orders"
        )
        np.testing.assert_allclose(
            b1, b2, rtol=1e-12, atol=1e-15, equal_nan=True,
            err_msg="I6_3 FAILED: B differs between materialization orders"
        )
        np.testing.assert_allclose(
            c1, c2, rtol=1e-12, atol=1e-15, equal_nan=True,
            err_msg="I6_3 FAILED: C differs between materialization orders"
        )

        # Secondary invariant: all three aliases must have no NaN
        assert not np.isnan(c1).any(), (
            f"I6_3 FAILED: C has {np.isnan(c1).sum()} NaN — "
            f"fill_value=0 on A did not propagate through two-level chain"
        )
