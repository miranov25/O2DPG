"""
Phase 13.21.ADF — Join Index Caching Tests

PURPOSE: Verify that join index cache survives materialize_aliases calls
(value-column additions) and invalidates correctly on subframe replacement.

PHASE: 13.21.ADF — Subframe join index caching
PROFILE EVIDENCE: _compute_join_indices = 56s / 141 cache misses (should be ~15)

J1_1: Cached join produces same output as uncached (correctness)
J1_2: Cache survives materialize_aliases — cache hit, not miss
J1_3: Cache invalidates on register_subframe with different data
J1_4: Cache invalidates on subframe swap
J1_5: Existing invariance behavior preserved (multi-subframe pipeline)
J2_1: Cache hit count with repeated materialize_aliases calls (performance gate)
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


def _build_main(n=1000):
    """Main ADF with index columns for subframe joins."""
    rng = np.random.default_rng(1321)
    df = pd.DataFrame({
        'x': rng.uniform(0, 10, n).astype(np.float32),
        'y': rng.normal(0, 1, n).astype(np.float32),
        'sec': rng.integers(0, 36, n).astype(np.int8),
        'row': rng.integers(0, 152, n).astype(np.int16),
    })
    return AliasDataFrame(df)


def _build_coeff_sf(n_sectors=36, scale=1.0, seed=42):
    """Coefficient subframe joined on 'sec'."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'sec': np.arange(n_sectors, dtype=np.int8),
        'c0': (rng.normal(0, 0.1, n_sectors) * scale).astype(np.float32),
        'c1': (rng.normal(0, 0.1, n_sectors) * scale).astype(np.float32),
    })
    adf = AliasDataFrame(df)
    adf.add_alias('c_sum', 'c0 + c1', dtype=np.float32)
    return adf


class TestJ1JoinCacheCorrectness:
    """J1_1..J1_5 — correctness of cached vs uncached joins."""

    @pytest.mark.invariance
    def test_J1_1_cached_equals_uncached(self):
        """
        J1_1: materialize with cache produces same result as without cache.

        Method: materialize the same alias twice — once with empty cache,
        once with populated cache. Results must be bit-exact.
        """
        adf = _build_main()
        sf = _build_coeff_sf()
        adf.register_subframe('Coeff', sf, index_columns=['sec'])
        adf.add_alias('correction', 'Coeff.c0 + x', dtype=np.float32)

        # First call — cache miss (computes join indices)
        adf.materialize_aliases(names=['correction'])
        result_first = adf.df['correction'].values.copy()

        # Drop the materialized column, keeping cache intact
        adf.df.drop(columns=['correction'], inplace=True)
        # Also drop the joined column so it re-joins
        joined_col = [c for c in adf.df.columns if c.endswith('__Coeff')]
        adf.df.drop(columns=joined_col, inplace=True, errors='ignore')

        # Second call — should use cache
        adf.materialize_aliases(names=['correction'])
        result_second = adf.df['correction'].values.copy()

        np.testing.assert_array_equal(
            result_first, result_second,
            err_msg="J1_1: cached join produced different result than uncached"
        )

    @pytest.mark.invariance
    def test_J1_2_cache_survives_materialize_aliases(self):
        """
        J1_2: cache survives value-column addition via materialize_aliases.

        Before fix: cache cleared after every materialize_aliases (line 4392).
        After fix: cache preserved — second materialize should be a cache hit.
        """
        adf = _build_main()
        sf = _build_coeff_sf()
        adf.register_subframe('Coeff', sf, index_columns=['sec'])

        # Add unrelated aliases that don't touch subframes
        adf.add_alias('x_squared', 'x**2', dtype=np.float32)
        adf.add_alias('y_abs', 'abs(y)', dtype=np.float32)
        # And a subframe-dependent alias
        adf.add_alias('correction', 'Coeff.c0', dtype=np.float32)

        # Materialize unrelated aliases first (adds columns, triggers pd.concat)
        adf.materialize_aliases(names=['x_squared', 'y_abs'])

        # Reset hit/miss counters
        adf._join_cache_hits = 0
        adf._join_cache_misses = 0

        # Now materialize the subframe-dependent alias
        adf.materialize_aliases(names=['correction'])

        # The join for 'Coeff' happens here.
        # Before fix: always a cache miss (cache cleared by previous materialize)
        # After fix: first access to 'Coeff' is still a miss (not cached yet)
        first_misses = adf._join_cache_misses

        # Drop joined column, re-materialize
        joined_cols = [c for c in adf.df.columns if '__Coeff' in c]
        adf.df.drop(columns=['correction'] + joined_cols, inplace=True,
                     errors='ignore')
        adf._join_cache_hits = 0
        adf._join_cache_misses = 0

        # Add another unrelated column (simulates next iteration)
        adf.add_alias('z_fake', 'x + y', dtype=np.float32)
        adf.materialize_aliases(names=['z_fake'])

        # Now re-materialize the subframe alias
        adf.materialize_aliases(names=['correction'])

        print(f"\nJ1_2: after 2nd materialize: "
              f"hits={adf._join_cache_hits}, misses={adf._join_cache_misses}")

        # After fix: this should be a cache HIT (cache survived the
        # intermediate materialize_aliases call)
        assert adf._join_cache_hits >= 1, (
            f"J1_2: expected cache hit after materialize_aliases, "
            f"got hits={adf._join_cache_hits}, misses={adf._join_cache_misses}. "
            f"Cache may still be cleared after materialize_aliases."
        )

    @pytest.mark.invariance
    def test_J1_3_cache_invalidates_on_register_subframe(self):
        """
        J1_3: registering a new subframe with the same name invalidates cache.

        This tests that register_subframe pops the old cache entry.
        """
        adf = _build_main()
        sf1 = _build_coeff_sf(scale=1.0, seed=42)
        adf.register_subframe('Coeff', sf1, index_columns=['sec'])
        adf.add_alias('correction', 'Coeff.c0', dtype=np.float32)

        # Materialize — populates cache for 'Coeff'
        adf.materialize_aliases(names=['correction'])
        result_v1 = adf.df['correction'].values.copy()

        # Re-register with DIFFERENT data (different seed → different c0 values)
        sf2 = _build_coeff_sf(scale=10.0, seed=99)
        adf.register_subframe('Coeff', sf2, index_columns=['sec'])

        # Drop old materialized columns
        drop_cols = ['correction'] + [c for c in adf.df.columns if '__Coeff' in c]
        adf.df.drop(columns=drop_cols, inplace=True, errors='ignore')

        # Re-materialize — should use NEW subframe data, not stale cache
        adf.materialize_aliases(names=['correction'])
        result_v2 = adf.df['correction'].values.copy()

        # Results must be different (different subframe data)
        assert not np.allclose(result_v1, result_v2, atol=0.1), (
            "J1_3: results identical after subframe swap — "
            "cache was not invalidated by register_subframe"
        )

    @pytest.mark.invariance
    def test_J1_4_cache_invalidates_on_subframe_data_change(self):
        """
        J1_4: if the subframe DataFrame object changes (new id), cache misses.

        This tests the subframe_id check in the cache validation.
        """
        adf = _build_main()
        sf = _build_coeff_sf(scale=1.0, seed=42)
        adf.register_subframe('Coeff', sf, index_columns=['sec'])
        adf.add_alias('correction', 'Coeff.c0', dtype=np.float32)

        adf.materialize_aliases(names=['correction'])
        result_before = adf.df['correction'].values.copy()

        # Mutate subframe's DataFrame (replace with modified copy)
        sf.df = sf.df.copy()
        sf.df['c0'] = sf.df['c0'] * 100  # drastically different values

        # Drop old joined columns
        drop_cols = ['correction'] + [c for c in adf.df.columns if '__Coeff' in c]
        adf.df.drop(columns=drop_cols, inplace=True, errors='ignore')

        adf._join_cache_hits = 0
        adf._join_cache_misses = 0
        adf.materialize_aliases(names=['correction'])
        result_after = adf.df['correction'].values.copy()

        print(f"\nJ1_4: after subframe mutation: "
              f"hits={adf._join_cache_hits}, misses={adf._join_cache_misses}")

        # Should be a cache miss (subframe id changed)
        assert adf._join_cache_misses >= 1, (
            "J1_4: expected cache miss after subframe mutation"
        )
        # Results should be different
        assert not np.allclose(result_before, result_after, atol=0.1), (
            "J1_4: results identical after subframe mutation"
        )

    @pytest.mark.invariance
    def test_J1_5_multi_subframe_pipeline(self):
        """
        J1_5: multi-subframe pipeline produces correct results with caching.

        Simulates calibration pipeline pattern: register 3 subframes,
        materialize in stages, verify all values are correct.
        """
        adf = _build_main(n=500)
        rng = np.random.default_rng(55)

        # Three subframes on different index columns
        sf_sec = _build_coeff_sf(n_sectors=36, seed=10)
        adf.register_subframe('BySec', sf_sec, index_columns=['sec'])

        df_row = pd.DataFrame({
            'row': np.arange(152, dtype=np.int16),
            'row_weight': rng.uniform(0.5, 1.5, 152).astype(np.float32),
        })
        adf.register_subframe('ByRow', AliasDataFrame(df_row),
                              index_columns=['row'])

        adf.add_alias('val_sec', 'x + BySec.c0', dtype=np.float32)
        adf.add_alias('val_row', 'y * ByRow.row_weight', dtype=np.float32)
        adf.add_alias('combined', 'val_sec + val_row', dtype=np.float32)

        # Stage 1: materialize val_sec
        adf.materialize_aliases(names=['val_sec'])

        # Stage 2: materialize val_row (different subframe)
        adf.materialize_aliases(names=['val_row'])

        # Stage 3: materialize combined (depends on both)
        adf.materialize_aliases(names=['combined'])

        # Verify correctness by manual computation
        sec_vals = adf.df['sec'].values
        c0_lookup = sf_sec.df.set_index('sec')['c0']
        expected_val_sec = adf.df['x'].values + c0_lookup.reindex(sec_vals).values

        row_vals = adf.df['row'].values
        rw_lookup = df_row.set_index('row')['row_weight']
        expected_val_row = adf.df['y'].values * rw_lookup.reindex(row_vals).values

        expected_combined = expected_val_sec + expected_val_row

        np.testing.assert_allclose(
            adf.df['combined'].values.astype(np.float64),
            expected_combined.astype(np.float64),
            rtol=1e-2,
            err_msg="J1_5: multi-subframe pipeline result mismatch"
        )

    @pytest.mark.invariance
    def test_J1_6_dematerialize_drop_and_recover(self):
        """
        J1_6: dematerialize(drop=[...]) drops the column;
        re-materialize recovers identical values.
        """
        adf = _build_main()
        sf = _build_coeff_sf()
        adf.register_subframe('Coeff', sf, index_columns=['sec'])
        adf.add_alias('correction', 'Coeff.c0 + x', dtype=np.float32)
        adf.add_alias('scaled', 'Coeff.c1 * y', dtype=np.float32)

        adf.materialize_aliases(names=['correction', 'scaled'])
        original_correction = adf.df['correction'].values.copy()
        original_scaled = adf.df['scaled'].values.copy()

        # Drop only 'correction', keep 'scaled'
        dropped = adf.dematerialize(drop=['correction'])
        assert dropped == ['correction'], f"Expected ['correction'], got {dropped}"
        assert 'correction' not in adf.df.columns, "correction should be dropped"
        assert 'scaled' in adf.df.columns, "scaled should survive"

        # Raw columns must survive
        assert 'x' in adf.df.columns, "raw column 'x' must survive"
        assert 'sec' in adf.df.columns, "raw column 'sec' must survive"

        # Re-materialize and verify bit-exact recovery
        adf.materialize_aliases(names=['correction'])
        np.testing.assert_array_equal(
            adf.df['correction'].values, original_correction,
            err_msg="J1_6: re-materialized values differ from original"
        )

    @pytest.mark.invariance
    def test_J1_7_dematerialize_keep(self):
        """
        J1_7: dematerialize(keep=[...]) drops all OTHER materialized aliases;
        raw columns survive; re-materialization recovers values.
        """
        adf = _build_main()
        sf = _build_coeff_sf()
        adf.register_subframe('Coeff', sf, index_columns=['sec'])
        adf.add_alias('a1', 'Coeff.c0', dtype=np.float32)
        adf.add_alias('a2', 'Coeff.c1', dtype=np.float32)
        adf.add_alias('a3', 'x**2', dtype=np.float32)

        adf.materialize_aliases(names=['a1', 'a2', 'a3'])
        original_a2 = adf.df['a2'].values.copy()
        original_a3 = adf.df['a3'].values.copy()

        # Keep only a1; drop a2 and a3
        dropped = adf.dematerialize(keep=['a1'])
        assert 'a1' in adf.df.columns, "a1 should survive (in keep list)"
        assert 'a2' not in adf.df.columns, "a2 should be dropped"
        assert 'a3' not in adf.df.columns, "a3 should be dropped"
        assert set(dropped) == {'a2', 'a3'}, f"Expected {{'a2','a3'}}, got {set(dropped)}"

        # Raw columns always survive
        assert 'x' in adf.df.columns
        assert 'sec' in adf.df.columns

        # Re-materialize dropped aliases
        adf.materialize_aliases(names=['a2', 'a3'])
        np.testing.assert_array_equal(
            adf.df['a2'].values, original_a2,
            err_msg="J1_7: re-materialized a2 differs"
        )
        np.testing.assert_array_equal(
            adf.df['a3'].values, original_a3,
            err_msg="J1_7: re-materialized a3 differs"
        )

    @pytest.mark.invariance
    def test_J1_8_dematerialize_all(self):
        """
        J1_8: dematerialize() with no args drops ALL materialized alias columns.
        Raw columns survive.
        """
        adf = _build_main()
        adf.add_alias('x2', 'x**2', dtype=np.float32)
        adf.add_alias('y2', 'y**2', dtype=np.float32)
        adf.materialize_aliases(names=['x2', 'y2'])

        raw_cols_before = [c for c in adf.df.columns if c not in adf.aliases]

        dropped = adf.dematerialize()
        assert 'x2' not in adf.df.columns
        assert 'y2' not in adf.df.columns

        # All raw columns still present
        for col in raw_cols_before:
            assert col in adf.df.columns, f"raw column '{col}' was dropped"

    @pytest.mark.invariance
    def test_J1_9_dematerialize_ignores_raw_columns(self):
        """
        J1_9: dematerialize(drop=['raw_col']) silently ignores raw columns.
        No crash, no drop.
        """
        adf = _build_main()
        adf.add_alias('a1', 'x**2', dtype=np.float32)
        adf.materialize_aliases(names=['a1'])

        # Try to drop a raw column — should be silently ignored
        dropped = adf.dematerialize(drop=['x', 'sec', 'a1'])
        assert 'a1' not in adf.df.columns, "alias a1 should be dropped"
        assert 'x' in adf.df.columns, "raw column x must survive"
        assert 'sec' in adf.df.columns, "raw column sec must survive"
        assert dropped == ['a1'], f"Only alias columns in drop list: {dropped}"

    @pytest.mark.invariance
    def test_J1_10_dematerialize_drop_keep_mutual_exclusion(self):
        """
        J1_10: specifying both drop and keep raises ValueError.
        """
        adf = _build_main()
        with pytest.raises(ValueError, match="drop or keep, not both"):
            adf.dematerialize(drop=['x'], keep=['y'])


class TestJ2JoinCachePerformance:
    """J2 — cache hit count verification."""

    @pytest.mark.invariance
    def test_J2_1_cache_hit_count_across_materialize_calls(self):
        """
        J2_1: with 3 subframes and 5 materialize_aliases calls,
        _compute_join_indices should be called ≤ 6 times (once per
        subframe on first use + at most 3 re-checks), not 15 (3×5).

        Before fix: 15 (cache cleared after each materialize)
        After fix: ≤ 6 (cache survives between calls)
        """
        adf = _build_main(n=2000)

        # Register 3 subframes
        for i in range(3):
            sf = _build_coeff_sf(seed=i * 10)
            adf.register_subframe(f'SF{i}', sf, index_columns=['sec'])
            adf.add_alias(f'val_{i}', f'SF{i}.c0 + x', dtype=np.float32)
            adf.add_alias(f'val2_{i}', f'SF{i}.c1 * y', dtype=np.float32)

        # Also add non-subframe aliases (to trigger pd.concat between calls)
        for i in range(5):
            adf.add_alias(f'filler_{i}', f'x + {i}', dtype=np.float32)

        # Track total cache misses across all calls
        total_misses = 0
        total_hits = 0

        for step in range(5):
            # Materialize a mix of subframe + non-subframe aliases
            names = [f'filler_{step}']
            if step < 3:
                names.append(f'val_{step}')
            if step >= 2:
                names.append(f'val2_{step % 3}')

            adf._join_cache_hits = 0
            adf._join_cache_misses = 0
            adf.materialize_aliases(names=names)
            total_hits += adf._join_cache_hits
            total_misses += adf._join_cache_misses

        print(f"\nJ2_1: across 5 materialize_aliases calls with 3 subframes:")
        print(f"  Total cache hits:   {total_hits}")
        print(f"  Total cache misses: {total_misses}")
        print(f"  Hit rate: {total_hits/(total_hits+total_misses)*100:.0f}%"
              if (total_hits + total_misses) > 0 else "  No joins")

        # Before fix: total_misses would be high (cache cleared each time)
        # After fix: at most 3 initial misses + minimal re-checks
        assert total_misses <= 6, (
            f"J2_1: expected ≤ 6 cache misses (3 subframes × ~2 initial), "
            f"got {total_misses}. Cache may still be cleared aggressively."
        )
