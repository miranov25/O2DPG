"""Regression tests for PHASE_13_114_ADF (BUG_AliasDataFrame_20261122): the dependency graph built by
select_aliases(..., with_dependencies=True) read the alias dictionary once for every word.

Plain words. `adf.aliases` is a property: every read builds a new {name: expression} dictionary from the
schema. The graph builder inside select_aliases() read it once for every word of every alias expression, so
one call cost (number of aliases) x (words in all expressions): 2.85 s in a real 638-alias session.
materialize_aliases() makes this call, and so do draw(), draw_figures() and draw_batch() when they have to
compute an alias. The fix reads the dictionary once per call.

Tests 1-2 count dictionary reads instead of timing, so they are deterministic on any machine. On the old
code they fail with an AssertionError that prints the growing count; on the fixed code they pass.
Tests 3-7 pin behaviour that must not change, and pass on the old and the fixed code:
  3. which aliases are selected and in which order (compared with the old algorithm, in the same process,
     because the order of independent aliases follows Python's string hashing and differs between runs);
  4. values, dtypes and missing join keys of materialized aliases, including a diamond of dependencies;
  5. the refusal of a dependency cycle and its message;
  6. alias changes between calls are seen (the dictionary is read anew on every call, nothing is cached);
  7. an alias named like a column of its own subframe (val = S.val) is not a cycle.

Out of scope (separate items): the regular expressions in add_alias() (finding C), the two frame copies in
materialize_aliases() (finding B), the other reads of `aliases` inside loops, and the different handling of
dotted subframe references by add_alias() and select_aliases() (the cycle of test 5).
"""
import contextlib
import os
import random
import re
import sys

import networkx as nx
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame  # noqa: E402

# 19 terms of 3 words each: a long expression, as in the dEdxAI session (about 77 words per alias)
LONG = " + ".join(f"c{j % 6}*c{(j + 1) % 6}*{j + 1}" for j in range(19))


@contextlib.contextmanager
def counting_alias_reads():
    """Count evaluations of the `aliases` property; the original property is restored in all cases."""
    original = AliasDataFrame.aliases
    counter = {"n": 0}

    def counting(self):
        counter["n"] += 1
        return original.fget(self)

    AliasDataFrame.aliases = property(counting, original.fset, original.fdel, original.__doc__)
    try:
        yield counter
    finally:
        AliasDataFrame.aliases = original


def session(n_alias):
    """A frame with `trivial` = c0*1.0 and n_alias-1 long aliases; every 8th uses the previous one."""
    adf = AliasDataFrame(pd.DataFrame({f"c{i}": np.arange(8.0) + i for i in range(6)}))
    adf.add_alias("trivial", "c0*1.0")
    for i in range(n_alias - 1):
        adf.add_alias(f"m{i}", LONG + (f" + m{i - 1}" if i % 8 == 7 else ""))
    return adf


# ---------------------------------------------------------------- 1-2: number of dictionary reads
def _reads(n_alias, call):
    adf = session(n_alias)
    with counting_alias_reads() as counter:
        call(adf)
    return counter["n"]


def test_graph_reads_alias_dictionary_a_fixed_number_of_times():
    """1. One select_aliases(..., with_dependencies=True) reads `aliases` as often at 40 as at 160 aliases."""
    call = lambda adf: adf.select_aliases(names=["trivial"], with_dependencies=True)
    small, large = _reads(40, call), _reads(160, call)
    assert (small, large) == (large, large) and large <= 5, (
        f"select_aliases rebuilt the alias dictionary {small} times at 40 aliases and {large} times at 160")


def test_materialize_aliases_reads_alias_dictionary_a_fixed_number_of_times():
    """2. One materialize_aliases() of a trivial alias reads `aliases` as often at 40 as at 160 aliases."""
    def call(adf):
        adf.materialize_aliases(names=["trivial"])
        assert "trivial" in adf.df.columns
    small, large = _reads(40, call), _reads(160, call)
    assert (small, large) == (large, large) and large <= 40, (
        f"materialize_aliases rebuilt the alias dictionary {small} times at 40 aliases and {large} times at 160")


# ---------------------------------------------------------------- 3: selection and order unchanged
def old_expansion(adf, result):
    """The dependency expansion of select_aliases() as it was before PHASE_13_114, step by step."""
    aliases = dict(adf.aliases)
    g = nx.DiGraph()
    for alias, expr in aliases.items():
        g.add_node(alias)
        for token in re.findall(r"\b\w+\b", expr):
            if token in aliases and token != alias:
                g.add_edge(token, alias)
    expanded = set(result)
    for name in result:
        expanded |= nx.ancestors(g, name)
    ordered = list(nx.topological_sort(g.subgraph(expanded)))
    return [n for n in ordered if n in expanded], g


def generated_session(seed):
    """Aliases over base columns and a subframe S: chains, shared (diamond) dependencies, dotted references,
    an alias named like a subframe column (m3 next to S.m3) and a self-named one (val = S.val*1)."""
    rng = random.Random(seed)
    base = AliasDataFrame(pd.DataFrame({**{f"c{i}": np.arange(30.0) + i for i in range(4)}, "k": np.arange(30)}))
    sub = AliasDataFrame(pd.DataFrame({"k": np.arange(30), "v": np.arange(30.0) / 3, "m3": np.ones(30),
                                       "val": np.arange(30.0)}))
    base.register_subframe("S", sub, index_columns="k")
    base.add_alias("val", "S.val*1")
    names = ["val"]
    for i in range(rng.randint(25, 45)):
        terms = [rng.choice(["c0", "c1", "c2", "c3", "S.v", "S.m3", "2.5"]) for _ in range(rng.randint(1, 5))]
        terms += rng.sample(names, k=min(len(names), rng.choice([0, 1, 1, 2])))
        base.add_alias(f"m{i}", " + ".join(terms))
        names.append(f"m{i}")
    return base, names, rng


@pytest.mark.parametrize("seed", range(8))
def test_selection_and_order_equal_the_old_algorithm(seed):
    """3. For single and multiple requested names: same aliases, same order, and a valid dependency order."""
    adf, names, rng = generated_session(seed)
    queries = [[q] for q in rng.sample(names, 6)] + [rng.sample(names, 3) for _ in range(3)]
    for query in queries:
        result = adf.select_aliases(names=query, with_dependencies=False)
        try:
            got = adf.select_aliases(names=query, with_dependencies=True)
        except ValueError as exc:                   # a dotted reference S.m3 next to alias m3 can close a cycle
            assert str(exc).startswith("Dependency cycle detected in aliases"), str(exc)
            with pytest.raises(nx.NetworkXUnfeasible):
                old_expansion(adf, result)          # the old algorithm refuses the same query
            continue
        want, graph = old_expansion(adf, result)
        assert got == want, f"seed {seed}, query {query}: {got} != old algorithm {want}"
        position = {n: i for i, n in enumerate(got)}
        for u, v in graph.subgraph(got).edges():
            assert position[u] < position[v], f"seed {seed}: {u} must come before {v} in {got}"


# ---------------------------------------------------------------- 4: materialized values unchanged
def test_materialized_values_dtypes_and_missing_keys():
    """4. A diamond a -> (b, c) -> d and a subframe alias with one parent key missing in the child."""
    adf = AliasDataFrame(pd.DataFrame({"k": np.array([0, 1, 2, 7]), "c0": np.array([1.0, 2.0, 3.0, 4.0])}))
    adf.register_subframe("S", AliasDataFrame(pd.DataFrame({"k": np.arange(4), "v": np.array([10.0, 20.0, 30.0, 40.0])})),
                          index_columns="k")
    adf.add_alias("a", "c0+1")
    adf.add_alias("b", "a*2")
    adf.add_alias("c", "a+3")
    adf.add_alias("d", "b+c")
    adf.add_alias("s", "S.v*2")
    assert adf.select_aliases(names=["d"], with_dependencies=True)[0] == "a"
    adf.materialize_aliases(names=["d", "s"])
    c0 = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(adf.df["d"].to_numpy(), 2 * (c0 + 1) + (c0 + 1 + 3))
    assert adf.df["d"].dtype == np.float64
    np.testing.assert_array_equal(adf.df["s"].to_numpy(), np.array([20.0, 40.0, 60.0, np.nan]))
    assert adf.df["s"].dtype == np.float64


# ---------------------------------------------------------------- 5: cycle refusal and message unchanged
def test_dependency_cycle_is_refused_with_the_same_message():
    """5. a = T.b and b = T.a pass add_alias(), but select_aliases() sees the cycle a -> b -> a and refuses.
    (This disagreement between the two checks is an existing, separate defect; this test pins today's
    behaviour so that PHASE_13_114 cannot change it silently.)"""
    adf = AliasDataFrame(pd.DataFrame({"k": np.arange(3), "x": np.arange(3.0)}))
    adf.register_subframe("T", AliasDataFrame(pd.DataFrame({"k": np.arange(3), "a": np.ones(3), "b": np.ones(3)})),
                          index_columns="k")
    adf.add_alias("a", "T.b")
    adf.add_alias("b", "T.a")
    for call in (lambda: adf.select_aliases(names=["a"], with_dependencies=True),
                 lambda: adf.materialize_aliases(names=["a"])):
        with pytest.raises(ValueError) as info:
            call()
        text = str(info.value)
        assert text.startswith("Dependency cycle detected in aliases (1 total, showing first 1):\n  Cycle: ")
        assert "    a = T.b" in text and "    b = T.a" in text
        assert "Hint: Self-referential aliases often occur" in text


# ---------------------------------------------------------------- 6: no stale view between calls
def test_alias_changes_between_calls_are_seen():
    """6. Redefining or removing an alias between two calls changes the next selection."""
    adf = AliasDataFrame(pd.DataFrame({"c0": np.arange(4.0), "c1": np.arange(4.0) * 10}))
    adf.add_alias("x", "c0*2")
    adf.add_alias("y", "x+1")
    assert adf.select_aliases(names=["y"], with_dependencies=True) == ["x", "y"]
    adf.add_alias("z", "c1+0.5")
    adf.add_alias("x", "z*2")                       # x now depends on z
    assert adf.select_aliases(names=["y"], with_dependencies=True) == ["z", "x", "y"]
    adf.remove_alias("z")
    assert adf.select_aliases(names=["y"], with_dependencies=True) == ["x", "y"]


# ---------------------------------------------------------------- 7: self-named subframe alias
def test_alias_named_like_its_subframe_column_is_not_a_cycle():
    """7. val = S.val*1 contains the word `val` but must not depend on itself."""
    adf = AliasDataFrame(pd.DataFrame({"k": np.arange(3)}))
    adf.register_subframe("S", AliasDataFrame(pd.DataFrame({"k": np.arange(3), "val": np.array([1.0, 2.0, 3.0])})),
                          index_columns="k")
    adf.add_alias("val", "S.val*1")
    adf.add_alias("twice", "val*2")
    assert adf.select_aliases(names=["twice"], with_dependencies=True) == ["val", "twice"]
    adf.materialize_aliases(names=["twice"])
    np.testing.assert_array_equal(adf.df["twice"].to_numpy(), np.array([2.0, 4.0, 6.0]))
