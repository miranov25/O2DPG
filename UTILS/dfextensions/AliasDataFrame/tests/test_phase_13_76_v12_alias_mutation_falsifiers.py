"""PHASE_13_76_ADF B3.2b — deterministic alias mutation falsifiers.

M-1 systematically enriches the audited DYN-P0-1 topology with committed seeds.
M-2 checks semantic state atomicity only for mutating definition failures whose
atomic-rejection behavior is already established.
"""

import copy

import numpy as np
import pandas as pd
import pytest

try:
    from AliasDataFrame.AliasDataFrame import AliasDataFrame
except (ImportError, ModuleNotFoundError):
    from AliasDataFrame import AliasDataFrame


DYN_P0_1 = (
    "DYN-P0-1: virtual upstream alias redefinition leaves materialized "
    "descendants stale"
)
DYN_P1_1 = (
    "DYN-P1-1: rejected indirect-cycle redefinition is not state-atomic"
)

COMMITTED_DAG_SEEDS = (17, 29, 43, 71, 101, 137, 211, 307, 419, 557)


def _semantic_state(adf):
    """Semantic mutable-state fingerprint; deliberately excludes object identity."""
    columns = tuple(map(str, adf.df.columns))
    values = tuple(
        (name, str(adf.df[name].dtype), np.asarray(adf.df[name]).copy())
        for name in columns
    )
    schema_columns = copy.deepcopy(adf._schema.get("columns", {}))
    schema_subframes = copy.deepcopy(adf._schema.get("subframes", {}))
    aliases = copy.deepcopy(dict(adf.aliases))
    groups = copy.deepcopy(getattr(adf, "_alias_groups", None))
    join_cache_keys = tuple(sorted(map(str, getattr(adf, "_join_index_cache", {}).keys())))
    return {
        "columns": columns,
        "values": values,
        "schema_columns": schema_columns,
        "schema_subframes": schema_subframes,
        "aliases": aliases,
        "groups": groups,
        "join_cache_keys": join_cache_keys,
    }


def _assert_semantic_state_equal(before, after):
    assert after["columns"] == before["columns"]
    assert after["schema_columns"] == before["schema_columns"]
    assert after["schema_subframes"] == before["schema_subframes"]
    assert after["aliases"] == before["aliases"]
    assert after["groups"] == before["groups"]
    assert after["join_cache_keys"] == before["join_cache_keys"]
    assert len(after["values"]) == len(before["values"])
    for (an, adt, av), (bn, bdt, bv) in zip(after["values"], before["values"]):
        assert (an, adt) == (bn, bdt)
        np.testing.assert_array_equal(av, bv)


def _dag_for_seed(seed):
    """Build a small deterministic rooted alias DAG and its independent oracle spec."""
    rng = np.random.default_rng(seed)
    n_nodes = int(rng.integers(5, 9))
    root_scale = float(rng.integers(2, 6))
    root_shift = float(rng.integers(1, 5))
    new_scale = root_scale + float(rng.integers(7, 13))
    new_shift = root_shift + float(rng.integers(5, 11))

    # specs[name] = (kind, parent1, parent2_or_scalar)
    specs = {"a0": ("root", root_scale, root_shift)}
    aliases = ["a0"]
    for i in range(1, n_nodes):
        name = f"a{i}"
        p1 = aliases[int(rng.integers(0, len(aliases)))]
        kind = ("add", "sub", "mul", "sum")[int(rng.integers(0, 4))]
        if kind == "sum" and len(aliases) > 1:
            p2 = aliases[int(rng.integers(0, len(aliases)))]
            expr = f"{p1}+{p2}"
            specs[name] = ("sum", p1, p2)
        else:
            if kind == "sum":
                kind = "add"
            scalar = float(rng.integers(1, 5))
            op = {"add": "+", "sub": "-", "mul": "*"}[kind]
            expr = f"{p1}{op}{scalar:g}"
            specs[name] = (kind, p1, scalar)
        aliases.append(name)
        specs[name] = (*specs[name], expr)

    # Last two aliases are explicit retained sinks.  Dependencies are temporary,
    # so a0 is forced virtual after materialization on every seed.
    sinks = aliases[-2:]
    return specs, sinks, root_scale, root_shift, new_scale, new_shift


def _oracle(specs, x, root_scale, root_shift):
    vals = {"a0": x * root_scale + root_shift}
    for name in sorted((n for n in specs if n != "a0"), key=lambda n: int(n[1:])):
        kind, p1, rhs, _expr = specs[name]
        if kind == "add":
            vals[name] = vals[p1] + rhs
        elif kind == "sub":
            vals[name] = vals[p1] - rhs
        elif kind == "mul":
            vals[name] = vals[p1] * rhs
        elif kind == "sum":
            vals[name] = vals[p1] + vals[rhs]
        else:  # pragma: no cover
            raise AssertionError(kind)
    return vals


def _install_dag(adf, specs, root_scale, root_shift):
    adf.add_alias("a0", f"x*{root_scale:g}+{root_shift:g}")
    for name in sorted((n for n in specs if n != "a0"), key=lambda n: int(n[1:])):
        adf.add_alias(name, specs[name][3])


@pytest.mark.invariance
class TestV12AliasMutationFalsifiers:
    @pytest.mark.parametrize("seed", COMMITTED_DAG_SEEDS, ids=lambda s: f"seed{s}")
    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=DYN_P0_1)
    def test_m1_seeded_alias_dag_virtual_upstream_invalidation(self, seed):
        specs, sinks, scale0, shift0, scale1, shift1 = _dag_for_seed(seed)
        x = np.linspace(0.5, 5.5, 16, dtype=np.float64)
        y = np.linspace(3.0, 7.0, 16, dtype=np.float64)
        adf = AliasDataFrame(pd.DataFrame({"x": x, "y": y}))
        _install_dag(adf, specs, scale0, shift0)
        adf.add_alias("unrelated", "y*3+5")

        # FORCE the audited topology.  Named descendants stay materialized,
        # dependencies are cleaned, and a0 must remain virtual.
        adf.materialize_aliases(names=[*sinks, "unrelated"])
        assert "a0" not in adf.df.columns, (
            f"seed={seed}: trigger not established; columns={list(adf.df.columns)}"
        )
        for sink in sinks:
            assert sink in adf.df.columns
        unrelated_before = np.asarray(adf.df["unrelated"]).copy()

        adf.add_alias("a0", f"x*{scale1:g}+{shift1:g}")

        # Every retained descendant is stale after the upstream definition changes.
        for sink in sinks:
            assert sink not in adf.df.columns, (
                f"seed={seed}: stale descendant {sink}; specs={specs}; "
                f"sinks={sinks}; columns={list(adf.df.columns)}"
            )
        assert "unrelated" in adf.df.columns
        np.testing.assert_array_equal(np.asarray(adf.df["unrelated"]), unrelated_before)

        expected = _oracle(specs, x, scale1, shift1)
        for sink in sinks:
            got = np.asarray(adf.eval(sink))
            np.testing.assert_allclose(
                got, expected[sink], rtol=0.0, atol=0.0,
                err_msg=f"seed={seed}; specs={specs}; sink={sink}",
            )

    def test_m2_self_reference_rejection_is_state_atomic(self):
        adf = AliasDataFrame(pd.DataFrame({"x": [1.0, 2.0]}))
        adf.add_alias("healthy", "x+1")
        adf.materialize_aliases(names=["healthy"])
        before = _semantic_state(adf)

        with pytest.raises(ValueError):
            adf.add_alias("q", "q+1")

        _assert_semantic_state_equal(before, _semantic_state(adf))
        np.testing.assert_array_equal(np.asarray(adf.eval("healthy")), [2.0, 3.0])

    def test_m2_vector_arity_rejection_is_state_atomic(self):
        adf = AliasDataFrame(pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]}))
        adf.add_alias("healthy", "x+y")
        adf.materialize_aliases(names=["healthy"])
        before = _semantic_state(adf)

        with pytest.raises(ValueError):
            adf.add_alias(["u", "v", "w"], "x+y,x-y")

        _assert_semantic_state_equal(before, _semantic_state(adf))
        np.testing.assert_array_equal(np.asarray(adf.eval("healthy")), [4.0, 6.0])

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=DYN_P1_1)
    def test_m2_indirect_cycle_rejection_is_state_atomic(self):
        adf = AliasDataFrame(pd.DataFrame({"x": [1.0, 2.0, 3.0]}))
        adf.add_alias("a", "b+1")
        adf.add_alias("healthy", "x*10")
        adf.materialize_aliases(names=["healthy"])
        before = _semantic_state(adf)

        with pytest.raises(ValueError, match="[Cc]ycle"):
            adf.add_alias("b", "a+1")

        _assert_semantic_state_equal(before, _semantic_state(adf))
        np.testing.assert_array_equal(np.asarray(adf.eval("healthy")), [10.0, 20.0, 30.0])
        adf.add_alias("later", "x+5")
        np.testing.assert_array_equal(np.asarray(adf.eval("later")), [6.0, 7.0, 8.0])
