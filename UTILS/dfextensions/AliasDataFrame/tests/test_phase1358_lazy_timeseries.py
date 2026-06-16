"""
Phase 13.58.ADF — single-tree lazy time-series loading (use case 1).

Dedicated lazy-loading test (D5, the primary gate / AC-7) plus the resolver and estimator
gates (AC-2/3/4/5/6). ROOT-free: synthetic TTrees written with uproot.mktree (seed 42),
read back via read_tree_lazy (uproot). The matplotlib-rendering integration (AC-1 gallery
double-run via dfdraw) runs on the server; here we prove the lazy-loading *mechanism* that
D1/D2/D3 enable — get_required_branches drives exactly the needed branch loads (and no
others), and the lazily-loaded data equals the eager source.

The auto-load path mirrors draw() exactly (AliasDataFrame.py draw(): get_required_branches
-> ensure_branches), so it cannot drift from production behaviour.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame  # noqa: E402

SEED = 42
N = 450


def _write_tree(path):
    """Synthetic single tree (seed 42). 'unused' must never load in a lazy run."""
    rng = np.random.default_rng(SEED)
    data = {
        "x": rng.random(N),
        "y": rng.random(N),
        "driftM": rng.random(N),
        "sector": rng.integers(0, 4, N).astype("i4"),
        "w": rng.random(N),
        "unused": rng.random(N),
    }
    types = {k: ("int32" if v.dtype == np.int32 else "float64") for k, v in data.items()}
    with uproot.recreate(path) as fo:
        fo.mktree("t", types)
        fo["t"].extend(data)
    return data


def _autoload(adf, **draw_kwargs):
    """Replicate draw()'s lazy auto-load step using the same public calls draw() makes:
    get_required_branches(...) then ensure_branches(missing). Returns the required set."""
    required = adf.get_required_branches(
        expr=draw_kwargs.get("expr"),
        selection=draw_kwargs.get("selection"),
        group_by=draw_kwargs.get("group_by"),
        color=draw_kwargs.get("color"),
        facet_by=draw_kwargs.get("facet_by"),
        weights=draw_kwargs.get("weights"),
        weights_vector=draw_kwargs.get("weights_vector"),
        selection_vector=draw_kwargs.get("selection_vector"),
    )
    to_load = required - adf._lazy_reader.loaded_branches
    if to_load:
        adf.ensure_branches(sorted(to_load))
    return required


@pytest.fixture
def tree(tmp_path):
    path = str(tmp_path / "ts.root")
    data = _write_tree(path)
    return path, data


# --------------------------- D5 / AC-7 (primary gate) ---------------------------

@pytest.mark.invariance
def test_AC7_only_needed_branches_load(tree):
    """D5/AC-7 primary gate: a lazy single-tree draw spec loads exactly the needed
    branches (facet_by + weights included) and nothing else — provably lazy."""
    path, data = tree
    adf = AliasDataFrame.read_tree_lazy(path, "t")
    assert adf._lazy_reader.loaded_branches == set()          # provably lazy at construction

    required = _autoload(adf, expr="y:x", facet_by="sector", weights="w")
    assert required == {"x", "y", "sector", "w"}
    assert adf._lazy_reader.loaded_branches == {"x", "y", "sector", "w"}
    assert "unused" not in adf._lazy_reader.loaded_branches    # only-needed-load


@pytest.mark.invariance
def test_AC7_registered_function_alias_loads(tree):
    """D5/AC-7: a registered-function alias resolves to its branch deps and loads them
    (not the function name, not unused branches)."""
    path, data = tree
    adf = AliasDataFrame.read_tree_lazy(path, "t")
    adf.register_function("corr", lambda a, b: a * b)
    adf.add_alias("cc", "corr(x, driftM)")

    required = _autoload(adf, expr="cc:y")
    assert required == {"x", "driftM", "y"}                    # corr excluded; deps loaded
    assert adf._lazy_reader.loaded_branches == {"x", "driftM", "y"}
    assert "unused" not in adf._lazy_reader.loaded_branches
    assert "corr" not in adf._lazy_reader.loaded_branches


def test_AC3_facet_weights_preload(tree):
    """AC-3: a branch referenced ONLY via facet_by / weights pre-loads in lazy mode."""
    path, data = tree
    adf = AliasDataFrame.read_tree_lazy(path, "t")
    _autoload(adf, expr="y:x", facet_by="sector", weights="w")
    loaded = adf._lazy_reader.loaded_branches
    assert "sector" in loaded and "w" in loaded                # the D2 gap, closed


def test_AC2_lazy_eager_data_identity(tree):
    """AC-2: lazily-loaded data equals the eager source exactly (data-level identity)."""
    path, data = tree
    adf = AliasDataFrame.read_tree_lazy(path, "t")
    _autoload(adf, expr="y:x", facet_by="sector", weights="w")
    for col in ["x", "y", "sector", "w"]:
        assert np.array_equal(np.asarray(adf.df[col]), data[col]), col
    assert len(adf.df) == N


# --------------------------- D1 resolver gates ---------------------------

@pytest.mark.invariance
def test_AC6_resolver_registered_function():
    """AC-6 (must fail pre-D1): an inline registered-function call resolves to its column
    deps and does NOT include the function name or a garbage token."""
    df = pd.DataFrame({c: np.arange(5.0) for c in ["xM", "driftM", "c"]})
    adf = AliasDataFrame(df)
    adf.register_function("corr", lambda a, b: a + b)
    assert adf.get_required_branches(expr="corr(xM, driftM)") == {"xM", "driftM"}
    assert adf.get_required_branches(expr="corr(xM, driftM):c") == {"xM", "driftM", "c"}
    # the same via an alias whose expression calls the registered function
    adf.add_alias("cc", "corr(xM, driftM)")
    assert adf.get_required_branches(expr="cc") == {"xM", "driftM"}


@pytest.mark.invariance
def test_AC4_resolver_regression():
    """AC-4: existing expressions return the same branch set (no regression), plus the
    §10 compound colon-grammar case."""
    df = pd.DataFrame({c: np.arange(5.0) for c in
                       ["signal", "trackLength", "p", "a", "b", "c", "isOK", "pt"]})
    adf = AliasDataFrame(df)
    adf.add_alias("dEdx", "signal / trackLength")
    assert adf.get_required_branches(expr="dEdx:p") == {"signal", "trackLength", "p"}
    assert adf.get_required_branches(expr="pt") == {"pt"}
    assert adf.get_required_branches(expr="abs(a)/sqrt(1+b**2):c") == {"a", "b", "c"}
    assert adf.get_required_branches(expr="pt", selection="isOK && pt > 0.5") == {"isOK", "pt"}


# --------------------------- D3 estimator gate ---------------------------

@pytest.mark.invariance
def test_AC5_estimate_memory_exact(tree):
    """AC-5: read_tree_lazy(...).estimate_memory matches eager sum(nbytes) exactly
    (tolerance 0), using real per-branch dtype (not a float32 constant), no AttributeError."""
    path, data = tree
    lazy = AliasDataFrame.read_tree_lazy(path, "t")
    eager = AliasDataFrame(pd.DataFrame(data))
    for sel in (["x", "sector", "w"], ["x"], None):
        cols = sel if sel else list(data)
        assert lazy.estimate_memory(sel)["bytes"] == eager.estimate_memory(cols)["bytes"], sel
    est = lazy.estimate_memory(["x", "sector", "w"])
    assert set(est) == {"bytes", "human", "branches", "entries", "warning"}
    assert est["entries"] == N
    # real dtype: x is float64 (8B), sector int32 (4B), w float64 (8B) -> 20*N, not 12*N
    assert est["bytes"] == (8 + 4 + 8) * N != 3 * 4 * N


@pytest.mark.invariance
def test_AC4_bracket_vector_resolver():
    """Bracket-vector lock (panel B5) + the architect's composition case: the gallery uses
    '[a, b]:x' bracket-vector notation; the resolver must extract every column, and must
    combine bracket-vector with registered-function recognition (D1) so that
    '[corr(x,y), x]:z' -> {x, y, z} with corr NOT treated as a column."""
    df = pd.DataFrame({c: np.arange(5.0) for c in ["x", "y", "z", "a", "b"]})
    adf = AliasDataFrame(df)
    adf.register_function("corr", lambda u, v: u * v)
    assert adf.get_required_branches(expr="[a, b]:x") == {"a", "b", "x"}
    # registered function inside a bracket-vector, composed with a bare column and a Y axis
    assert adf.get_required_branches(expr="[corr(x, y), x]:z") == {"x", "y", "z"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
