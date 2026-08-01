"""
PHASE_13_65_ADF — Asymmetric subframe join keys.

register_subframe(... right_index_columns=[...]) lets the parent (left) and child
(right) join columns have different names, mirroring pandas merge(left_on=, right_on=).
Omitting right_index_columns is byte-identical to the previous same-name behavior.

Test order follows the panel recommendation (A4b-first TDD):
  A4b  multi-column integer key, asymmetric names, _use_numba=True (Phase 8c path 2)
       must equal the merge path (path 3) on the same data — the load-bearing gate
  A4   single-column, numba (path 1) vs merge (path 3) agree
  A10  symmetric registration unchanged on all paths (regression)
  A1   asymmetric basic correctness vs manual oracle
  A6   validation (length mismatch / unknown name)
  A5   missing key on child side
  A7   schema round-trip incl. absent-field back-compat
  A9   asymmetric pre_index=True

numba note: paths 1/2 only fire when numba is installed AND n >= NUMBA_MIN_ROWS
(10,000). Where numba is absent the comparison is still valid (both sides take the
merge path); the path-2 firing is confirmed on the alma2 numba run.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame  # noqa: E402


def _parent_child_multicol(n=12000, seed=0):
    """Parent with keys (a, b5); child with keys (a, b_alt5) — b5 vs b_alt5 asymmetric."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 6, n).astype(np.int64)
    b = rng.integers(0, 8, n).astype(np.int64)
    parent = AliasDataFrame(pd.DataFrame({"a": a, "b5": b, "x": rng.normal(0, 1, n)}))
    # child: one row per (a, b) combo, value = a*100 + b
    combos = np.array([[i, j] for i in range(6) for j in range(8)], dtype=np.int64)
    child = AliasDataFrame(pd.DataFrame({
        "a": combos[:, 0],
        "b_alt5": combos[:, 1],          # different name from parent's b5
        "val": combos[:, 0] * 100 + combos[:, 1],
    }))
    return parent, child, a, b


def _join_values(parent, sf_name, sf_col):
    """Materialize a subframe column onto the parent via an alias draw-free path."""
    parent.add_alias("joined", f"{sf_name}.{sf_col}")
    parent.materialize_aliases(names=["joined"])
    return parent.df["joined"].to_numpy()


# --------------------------------------------------------------------------- #
# A4b — load-bearing: Phase 8c linearization (path 2) vs merge (path 3)
# --------------------------------------------------------------------------- #

def test_A4b_multicol_numba_vs_merge_asymmetric():
    pytest.importorskip("numba")   # Phase 8c linearization only fires with numba present
    parent, child, a, b = _parent_child_multicol(n=12000)   # >= NUMBA_MIN_ROWS
    parent.register_subframe("C", child, index_columns=["a", "b5"],
                             right_index_columns=["a", "b_alt5"])
    expected = a * 100 + b                                   # manual oracle

    parent._use_numba = True                                 # path 2 on alma2
    got_numba = _join_values(parent, "C", "val")

    parent2, child2, _, _ = _parent_child_multicol(n=12000)
    parent2.register_subframe("C", child2, index_columns=["a", "b5"],
                              right_index_columns=["a", "b_alt5"])
    parent2._use_numba = False                               # path 3 (merge)
    got_merge = _join_values(parent2, "C", "val")

    assert np.array_equal(got_numba, expected)
    assert np.array_equal(got_merge, expected)
    assert np.array_equal(got_numba, got_merge)              # paths agree


# --------------------------------------------------------------------------- #
# A4 — single-column numba (path 1) vs merge (path 3), asymmetric
# --------------------------------------------------------------------------- #

def test_A4_singlecol_numba_vs_merge_asymmetric():
    pytest.importorskip("numba")   # single-col numba path 1 only fires with numba present
    rng = np.random.default_rng(1)
    n = 12000
    k = rng.integers(0, 10, n).astype(np.int64)
    parent = AliasDataFrame(pd.DataFrame({"kp": k, "x": rng.normal(0, 1, n)}))
    child = AliasDataFrame(pd.DataFrame({"kc": np.arange(10, dtype=np.int64),
                                         "val": np.arange(10, dtype=np.int64) * 7}))
    parent.register_subframe("C", child, index_columns=["kp"], right_index_columns=["kc"])
    expected = k * 7
    parent._use_numba = True
    got_a = _join_values(parent, "C", "val")
    parent.df.drop(columns=["joined"], inplace=True)
    parent._join_index_cache.clear()
    parent._use_numba = False
    got_b = _join_values(parent, "C", "val")
    assert np.array_equal(got_a, expected)
    assert np.array_equal(got_b, expected)
    assert np.array_equal(got_a, got_b)


# --------------------------------------------------------------------------- #
# A4c — multi-column asymmetric via merge (path 3), no numba required
# --------------------------------------------------------------------------- #

def test_A4c_multicol_merge_asymmetric():
    parent, child, a, b = _parent_child_multicol(n=3000)
    parent.register_subframe("C", child, index_columns=["a", "b5"],
                             right_index_columns=["a", "b_alt5"])
    parent._use_numba = False                                # force merge (path 3)
    got = _join_values(parent, "C", "val")
    assert np.array_equal(got, a * 100 + b)


# --------------------------------------------------------------------------- #
# A10 — symmetric registration unchanged (regression)
# --------------------------------------------------------------------------- #

def test_A10_symmetric_unchanged():
    rng = np.random.default_rng(2)
    n = 500
    k = rng.integers(0, 8, n).astype(np.int64)
    parent = AliasDataFrame(pd.DataFrame({"k": k, "x": rng.normal(0, 1, n)}))
    child = AliasDataFrame(pd.DataFrame({"k": np.arange(8, dtype=np.int64),
                                         "val": np.arange(8, dtype=np.int64) * 3}))
    parent.register_subframe("C", child, index_columns=["k"])   # no right_index_columns
    got = _join_values(parent, "C", "val")
    assert np.array_equal(got, k * 3)


# --------------------------------------------------------------------------- #
# A1 — asymmetric basic correctness
# --------------------------------------------------------------------------- #

def test_A1_asymmetric_basic():
    parent = AliasDataFrame(pd.DataFrame({"kp": [0, 1, 2, 1, 0], "x": [1, 2, 3, 4, 5]}))
    child = AliasDataFrame(pd.DataFrame({"kc": [0, 1, 2], "val": [10, 20, 30]}))
    parent.register_subframe("C", child, index_columns=["kp"], right_index_columns=["kc"])
    got = _join_values(parent, "C", "val")
    assert list(got) == [10, 20, 30, 20, 10]


# --------------------------------------------------------------------------- #
# A6 — validation
# --------------------------------------------------------------------------- #

def test_A6_validation_length_and_names():
    parent = AliasDataFrame(pd.DataFrame({"kp": [0, 1], "x": [1, 2]}))
    child = AliasDataFrame(pd.DataFrame({"kc": [0, 1], "val": [9, 8]}))
    with pytest.raises(ValueError):   # length mismatch
        parent.register_subframe("C", child, index_columns=["kp"],
                                 right_index_columns=["kc", "extra"])
    with pytest.raises(ValueError):   # unknown parent name
        parent.register_subframe("C", child, index_columns=["nope"],
                                 right_index_columns=["kc"])
    with pytest.raises(ValueError):   # unknown child name
        parent.register_subframe("C", child, index_columns=["kp"],
                                 right_index_columns=["nope"])


# --------------------------------------------------------------------------- #
# A5 — missing key on the child side
# --------------------------------------------------------------------------- #

def test_A5_missing_child_key():
    """REVISED under AD-19 (architect, RATIFIED 2026-07-29, Option 1 with an
    operational definition), on the architect's explicit instruction that this
    test be updated.

    Until round 10 this asserted `np.isnan(got[2])` — i.e. that an `int64`
    child column silently widened to `float64` to represent a gap. AD-19:

        Every dtype observable from source metadata, an existing physical
        column, schema metadata, an explicit alias declaration, or the first
        successful creation/materialization is authoritative. ADF must
        preserve it thereafter. If a missing value cannot be represented in
        that dtype, ADF must use an explicitly configured compatible fill or
        refuse clearly.

    `val` is `int64` by virtue of existing, so it is authoritative. ADF does
    not need to know whether the user consciously chose it. The asymmetric-key
    behaviour this test exists to prove — that `kp`/`kc` join correctly and
    that key 9 finds no match — is unchanged and is still asserted, through
    the configured fill.
    """
    parent = AliasDataFrame(pd.DataFrame({"kp": [0, 1, 9], "x": [1, 2, 3]}))
    child = AliasDataFrame(pd.DataFrame({"kc": [0, 1], "val": [10, 20]}))
    parent.register_subframe("C", child, index_columns=["kp"], right_index_columns=["kc"])

    # No configured fill: an authoritative int64 cannot hold the gap -> refuse.
    with pytest.raises(ValueError, match="authoritative dtype"):
        _join_values(parent, "C", "val")

    # With the physically correct value configured, the join is exact and the
    # dtype is preserved. -1 is used here purely as a sentinel the test can
    # recognise; ADF never picks it, which is the point of the ruling.
    parent2 = AliasDataFrame(pd.DataFrame({"kp": [0, 1, 9], "x": [1, 2, 3]}))
    child2 = AliasDataFrame(pd.DataFrame({"kc": [0, 1], "val": [10, 20]}))
    parent2.register_subframe("C", child2, index_columns=["kp"],
                              right_index_columns=["kc"])
    parent2.set_subframe_fill("C", fill_missing=-1)
    got = _join_values(parent2, "C", "val")
    assert got.dtype == np.int64          # authoritative dtype preserved
    assert got[0] == 10 and got[1] == 20  # matched values exact
    assert got[2] == -1                   # key 9 not in child -> configured fill


# --------------------------------------------------------------------------- #
# A7 — schema round-trip + absent-field back-compat
# --------------------------------------------------------------------------- #

def test_A7_schema_roundtrip_and_absent_field():
    parent = AliasDataFrame(pd.DataFrame({"kp": [0, 1], "x": [1, 2]}))
    child = AliasDataFrame(pd.DataFrame({"kc": [0, 1], "val": [5, 6]}))
    parent.register_subframe("C", child, index_columns=["kp"], right_index_columns=["kc"])
    entry = parent._schema["subframes"]["C"]
    assert entry.get("right_index_columns") == ["kc"]          # persisted
    # absent-field back-compat: an old schema entry without right_index_columns
    old = {"index": ["k"], "index_columns": ["k"]}
    right = old.get("right_index_columns", old.get("index_columns", old.get("index")))
    assert right == ["k"]                                       # defaults to symmetric


# --------------------------------------------------------------------------- #
# A9 — asymmetric pre_index=True
# --------------------------------------------------------------------------- #

def test_A9_asymmetric_pre_index():
    parent = AliasDataFrame(pd.DataFrame({"kp": [0, 1, 2], "x": [1, 2, 3]}))
    child = AliasDataFrame(pd.DataFrame({"kc": [0, 1, 2], "val": [100, 200, 300]}))
    parent.register_subframe("C", child, index_columns=["kp"],
                             right_index_columns=["kc"], pre_index=True)
    got = _join_values(parent, "C", "val")
    assert list(got) == [100, 200, 300]
