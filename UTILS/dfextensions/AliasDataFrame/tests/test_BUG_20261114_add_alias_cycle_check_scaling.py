"""Regression tests for BUG_AliasDataFrame_20261114: add_alias() cycle check was O(N^2 x tokens).

`AliasDataFrame.aliases` is a property that rebuilds the {name: expr} mapping on every
access (O(N)). _resolve_dependencies() read it once per token of every alias, so one
add_alias() cost O(N^2 x tokens), and building a frame alias by alias cost O(N^3 x tokens).
The fix reads the property once per call.

Test 1 counts property reads instead of timing wall clock, so it is deterministic on any
machine. Tests 2-4 pin the behaviour that must not change: the dependency map, the cycle
rejection and the invalidation of dependents when a materialized alias is redefined.

Negative control: on the unfixed AliasDataFrame.py (MD5 2b01116110c1925032abd40fe5100bb0)
test 1 fails with an AssertionError (843 reads at N=20, 8043 at N=200); tests 2-4 pass on
both the unfixed and the fixed bytes.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame  # noqa: E402

TERMS = "+".join(f"c{j % 3}*(abs(x)*y)" for j in range(10))   # 40 word tokens, no alias names


def _frame(with_subframe=False):
    adf = AliasDataFrame(pd.DataFrame({"k": np.arange(4), "x": np.arange(4.0), "y": np.ones(4),
                                       **{f"c{j}": np.ones(4) for j in range(3)}}))
    if with_subframe:
        sub = AliasDataFrame(pd.DataFrame({"k": np.arange(4), "a0": np.ones(4), "b": np.ones(4)}))
        adf.register_subframe("S", sub, index_columns=["k"])
    return adf


def _reads_during_one_add_alias(monkeypatch, n_existing):
    """Number of `aliases` property evaluations during one add_alias() on a frame with n_existing aliases."""
    adf = _frame()
    for i in range(n_existing):                       # bulk-define, as a saved schema would; no per-call check
        adf._schema["columns"][f"a{i}"] = {"expr": TERMS}
    original = AliasDataFrame.aliases
    counter = {"n": 0}

    def counting(self):
        counter["n"] += 1
        return original.fget(self)

    monkeypatch.setattr(AliasDataFrame, "aliases", property(counting, original.fset))
    adf.add_alias("probe", TERMS)
    monkeypatch.setattr(AliasDataFrame, "aliases", original)
    return counter["n"]


def test_add_alias_reads_aliases_a_constant_number_of_times(monkeypatch):
    """1. The number of `aliases` rebuilds in one add_alias() does not grow with the number of aliases."""
    small = _reads_during_one_add_alias(monkeypatch, 20)
    large = _reads_during_one_add_alias(monkeypatch, 200)
    assert large == small, f"aliases rebuilt {small} times at N=20 but {large} times at N=200"
    assert large <= 10, f"aliases rebuilt {large} times in one add_alias()"


def test_resolve_dependencies_map_unchanged():
    """2. Dependency map: subframe refs, repeated tokens, numbers, a subframe column named like an alias."""
    adf = _frame(with_subframe=True)
    adf.add_alias("a0", "S.a0*2+x")                    # S.a0 is a subframe column, not the alias a0
    adf.add_alias("b", "a0+a0+a0*40+S.b")              # repeated token; S.b is not the alias b
    adf.add_alias("c2", "a0*b+abs(b)+3.5")
    deps = adf._resolve_dependencies()
    assert {k: set(v) for k, v in deps.items()} == {"b": {"a0"}, "c2": {"a0", "b"}}


def test_indirect_cycle_still_raises():
    """3. The cycle check still rejects an indirect cycle p -> q -> p."""
    adf = _frame()
    adf.add_alias("p", "x+1")
    adf.add_alias("q", "p*2")
    with pytest.raises(ValueError, match="Cycle detected"):
        adf.add_alias("p", "q+1")


def test_redefinition_still_invalidates_materialized_dependents():
    """4. Redefining a materialized alias drops it and its dependents (found via the dependency map), not others."""
    adf = _frame()
    adf.add_alias("p", "x*2")
    adf.add_alias("q", "p+1")
    adf.add_alias("r", "y+1")
    for name in ("p", "q", "r"):
        adf.materialize_alias(name)
    adf.add_alias("p", "x*3")
    assert "p" not in adf.df.columns and "q" not in adf.df.columns
    assert "r" in adf.df.columns
    adf.materialize_alias("q")
    assert adf.df["q"].tolist() == [1.0, 4.0, 7.0, 10.0]
