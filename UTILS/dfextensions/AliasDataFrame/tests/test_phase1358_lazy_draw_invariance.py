"""
Phase 13.58.ADF — lazy DRAWING invariance on small synthetic data (seed 42).

Runs real ``adf.draw()`` in lazy mode (via dfdraw) and asserts, for the gallery's draw
attributes (plot types + column-name-bearing kwargs), that lazy == eager:
  (a) the draw renders without error,
  (b) the lazy ADF auto-loads EXACTLY the branches the draw needs (D1 resolver + D2 scan)
      and never the 'unused' branch — provably lazy,
  (c) for profile (which returns stats), the lazy stats equal the eager stats, and
  (d) the lazily-loaded branch data equals the eager source (data the plot consumes).

No ROOT and no 2 GB gallery file — fully synthetic (uproot TTree via mktree). This is an
INVARIANCE gate, complementary to (not a replacement for) the real-data gallery double-run
(test_phase1358_gallery_lazy.py). Skips cleanly if dfdraw is not importable.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
import uproot

import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame  # noqa: E402

# dfdraw is required for actual drawing; skip the whole module cleanly if absent.
pytest.importorskip("dfextensions.dfdraw", reason="dfdraw not importable; lazy-draw gate skipped")

SEED = 42
N = 450


def _write_tree(path):
    rng = np.random.default_rng(SEED)
    data = {
        "x": rng.random(N),
        "y": rng.random(N),
        "y2": rng.random(N),
        "driftM": rng.random(N),
        "sector": rng.integers(0, 4, N).astype("i4"),
        "w": rng.random(N),
        "w2": rng.random(N),
        "q": rng.random(N),
        "a": rng.random(N),
        "colr": rng.random(N),
        # >=3 decoys (panel exact-load standard): none of these may ever load
        "unused": rng.random(N),
        "unused2": rng.random(N),
        "unused3": rng.random(N),
    }
    types = {k: ("int32" if v.dtype == np.int32 else "float64") for k, v in data.items()}
    with uproot.recreate(path) as fo:
        fo.mktree("t", types)
        fo["t"].extend(data)
    return data


def _setup(adf):
    """Shared setup applied identically to the lazy and eager sides."""
    adf.draw_lazy = True
    adf.register_function("corr", lambda u, v: u * v)
    adf.add_alias("cc", "corr(x, driftM)")
    adf.add_alias("cc2", "cc + y")          # nested alias: alias -> alias -> registered fn
    return adf


# Draw specs modelled on the time-series gallery (time_series_draw.py): plot types and the
# column-name-bearing kwargs. `need` = base branches the lazy run must auto-load.
SPECS = [
    ("hist",            dict(expr="x", type="hist", bins=20),                          {"x"}),
    ("hist_selection",  dict(expr="x", type="hist", bins=20, selection="q>0.2"),       {"x", "q"}),
    ("hist_cumulative", dict(expr="x", type="hist", bins=20, cumulative=True),         {"x"}),
    ("scatter",         dict(expr="y:x", type="scatter"),                              {"x", "y"}),
    ("profile",         dict(expr="y:x", type="profile", bins=10),                     {"x", "y"}),
    ("profile_groupby", dict(expr="y:x", type="profile", bins=10, group_by="sector"),  {"x", "y", "sector"}),
    ("profile_facet",   dict(expr="y:x", type="profile", bins=10, facet_by="sector"),  {"x", "y", "sector"}),
    ("profile_weights", dict(expr="y:x", type="profile", bins=10, weights="w"),        {"x", "y", "w"}),
    ("profile_regfunc", dict(expr="cc:y", type="profile", bins=10),                    {"x", "driftM", "y"}),
]


@pytest.fixture
def root_path(tmp_path):
    p = str(tmp_path / "ts.root")
    data = _write_tree(p)
    return p, data


def _cmp_stats(sl, se, label):
    """Defensive lazy==eager stats comparison: equate whatever numeric keys both return."""
    compared = 0
    for k in (set(sl) & set(se)):
        try:
            a = np.asarray(sl[k], dtype=float)
            b = np.asarray(se[k], dtype=float)
        except (TypeError, ValueError):
            continue
        if a.shape == b.shape and a.size:
            assert np.allclose(a, b, equal_nan=True), f"{label}: stat '{k}' lazy != eager"
            compared += 1
    assert compared > 0, f"{label}: no comparable numeric stats produced"


@pytest.mark.invariance
@pytest.mark.parametrize("spec_id,kw,need", SPECS, ids=[s[0] for s in SPECS])
def test_lazy_draw_invariance(root_path, spec_id, kw, need):
    path, data = root_path
    lazy = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    eager = _setup(AliasDataFrame(pd.DataFrame(data)))
    assert lazy._lazy_reader.loaded_branches == set()        # provably lazy at construction

    returns_stats = (kw.get("type") == "profile")
    if returns_stats:
        out_l = lazy.draw(return_data=True, **kw)
        out_e = eager.draw(return_data=True, **kw)
        assert isinstance(out_l, tuple) and len(out_l) > 2     # rendered + stats
        _cmp_stats(out_l[2], out_e[2], spec_id)
    else:
        assert lazy.draw(**kw) is not None                     # rendered without error
        eager.draw(**kw)

    loaded = lazy._lazy_reader.loaded_branches
    # exact-load standard (panel): loaded == need exactly, not need <= loaded. With >=3
    # decoy branches in the fixture, this catches over-loading (a "load everything" bug).
    assert loaded == need, f"{spec_id}: loaded {sorted(loaded)} != need {sorted(need)} (over/under-load)"

    # data-level identity: the branch arrays the plot consumed equal the eager source
    for col in (need - {"cc"}):
        assert np.array_equal(np.asarray(lazy.df[col]), data[col]), f"{spec_id}: {col} data differs"


# ----------- other call sites and kwarg forms where lazy resolution runs -----------

@pytest.mark.invariance
def test_lazy_draw_batch_loads_union(root_path, tmp_path):
    """draw_batch(): the lazy auto-load scans every spec and loads the union of their
    branches (facet_by/weights/selection across specs), nothing else."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    adf.draw_batch({
        "p1": {"expr": "y:x", "type": "profile", "bins": 10, "facet_by": "sector"},
        "p2": {"expr": "y:x", "type": "profile", "bins": 10, "weights": "w"},
        "h1": {"expr": "x", "type": "hist", "bins": 20, "selection": "q>0.2"},
    }, save_dir=str(tmp_path / "batch"))
    loaded = adf._lazy_reader.loaded_branches
    assert loaded == {"x", "y", "sector", "w", "q"}


@pytest.mark.invariance
def test_lazy_draw_figures_loads_union(root_path, tmp_path):
    """draw_figures(): the lazy auto-load scans every subplot and loads the union."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    adf.draw_figures([{"plots": [
        {"expr": "y:x", "type": "profile", "bins": 10},
        {"expr": "y2:x", "type": "profile", "bins": 10, "weights": "w2"},
    ]}], save_dir=str(tmp_path / "figs"))
    loaded = adf._lazy_reader.loaded_branches
    assert loaded == {"x", "y", "y2", "w2"}


@pytest.mark.invariance
def test_lazy_selection_vector_loads(root_path):
    """Per-Y selection_vector (a list) contributes its branches to the lazy load."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    adf.draw(expr="y:x", type="profile", bins=10,
             selection_vector=["q>0.3", "q<=0.3"], vector_compose="outer")
    loaded = adf._lazy_reader.loaded_branches
    assert loaded == {"x", "y", "q"}


@pytest.mark.invariance
def test_lazy_nested_alias_loads(root_path):
    """A nested alias (cc2 -> cc -> corr(x, driftM)) resolves to its base branches."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    adf.draw(expr="cc2:y", type="profile", bins=10)
    loaded = adf._lazy_reader.loaded_branches
    assert loaded == {"x", "driftM", "y"}


@pytest.mark.invariance
def test_lazy_color_branch_loads(root_path):
    """A branch referenced only via color= contributes to the lazy load."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    adf.draw(expr="y:x", type="scatter", color="colr")
    loaded = adf._lazy_reader.loaded_branches
    assert loaded == {"x", "y", "colr"}


@pytest.mark.invariance
def test_lazy_weights_vector_loads(root_path):
    """T3: per-Y weights_vector (a list) contributes its branches to the lazy load."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    adf.draw(expr="y:x", type="profile", bins=10,
             weights_vector=["w", "w2"], vector_compose="outer")
    assert adf._lazy_reader.loaded_branches == {"x", "y", "w", "w2"}


@pytest.mark.invariance
def test_lazy_logical_selection_loads(root_path):
    """T8: a compound selection (numexpr '&'/'|') loads every branch it references.
    Note: ADF selection is numexpr-style ('&'/'|'), not C-style ('&&'/'||'); '&&' raises."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    adf.draw(expr="y:x", type="profile", bins=10, selection="(q>0.2) & (a<0.8)")
    assert adf._lazy_reader.loaded_branches == {"x", "y", "q", "a"}


@pytest.mark.invariance
def test_lazy_selection_side_registered_function(root_path):
    """T14: a registered-function alias on the SELECTION side resolves to its deps
    (distinct path from expr-side)."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))   # cc = corr(x, driftM)
    adf.draw(expr="y:x", type="profile", bins=10, selection="cc>0.3")
    assert adf._lazy_reader.loaded_branches == {"x", "y", "driftM"}


@pytest.mark.invariance
def test_lazy_multidraw_no_reload(root_path):
    """T10: many draws on one lazy ADF accumulate monotonically with no re-load --
    the gallery pattern (36+ draws on one build_adf)."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    adf.draw(expr="y:x", type="profile", bins=10)
    first = set(adf._lazy_reader.loaded_branches)
    assert first == {"x", "y"}
    adf.draw(expr="y:x", type="profile", bins=10)            # identical -> no re-load
    assert set(adf._lazy_reader.loaded_branches) == first
    adf.draw(expr="driftM:x", type="profile", bins=10)       # new branch -> grows by exactly driftM
    after = set(adf._lazy_reader.loaded_branches)
    assert after == first | {"driftM"}


def test_lazy_nonexistent_branch_raises(root_path):
    """T12 / PP-2 negative control: a non-existent branch in expr raises a clear error,
    it does NOT silently render an empty figure."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    with pytest.raises(ValueError):
        adf.draw(expr="nosuchbranch:x", type="profile", bins=10)


def test_d2_missing_wiring_raises_not_silent(root_path):
    """B2 negative control (committed, not prose): if D2 wiring is removed so the
    facet_by branch is NOT pre-loaded, the draw FAILS LOUD (ValueError) rather than
    silently producing an empty/wrong figure. This is the architectural proof that the
    silent-empty-figure failure mode cannot recur. Uses a RAW branch for facet_by (the
    isolation does not hold for aliases, per the panel caveat)."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    orig = adf.get_required_branches
    # simulate broken D2: drop facet_by/weights from the scan
    adf.get_required_branches = lambda **kw: orig(
        expr=kw.get("expr"), selection=kw.get("selection"),
        group_by=kw.get("group_by"), color=kw.get("color"))
    with pytest.raises(ValueError):
        adf.draw(expr="y:x", type="profile", bins=10, facet_by="sector")


def test_d2_resolver_isolation_function_level(root_path):
    """Corrected T9: function-level D2 isolation -- get_required_branches returns the
    kwarg-only branch with no draw at all (immune to dfdraw changes; the clean isolation
    that replaces the render monkeypatch)."""
    path, data = root_path
    adf = _setup(AliasDataFrame.read_tree_lazy(path, "t"))
    assert "sector" in adf.get_required_branches(expr="y:x", facet_by="sector")
    assert "w" in adf.get_required_branches(expr="y:x", weights="w")
    assert "colr" in adf.get_required_branches(expr="y:x", color="colr")
    assert "q" in adf.get_required_branches(expr="y:x", selection_vector=["q>0.2"])


@pytest.mark.invariance
def test_lazy_subframe_column_draw(tmp_path):
    """Phase 13.58: subframe-column lazy draw. A registered lazy subframe (proper index
    columns) drawn via 'A.col:x' is materialized on demand through the existing
    ensure_subframe machinery and matches eager exactly, loading only the subframe's index
    column plus the main-tree branches used (not the decoy column).

    This is the positive proof of the feature; the calibITS test (T11) exercises the same
    path on real data and the names-only boundary.
    """
    import warnings as _warnings
    rng = np.random.default_rng(7); n = 400
    run = np.arange(n); x = rng.random(n); a = rng.random(n); unused = rng.random(n)

    def _build():
        m = AliasDataFrame(pd.DataFrame({"run": run.copy(), "x": x.copy(), "unused": unused.copy()}))
        m.register_subframe("A", AliasDataFrame(pd.DataFrame({"run": run.copy(), "a": a.copy()})),
                            index_columns=["run"])
        return m

    p = str(tmp_path / "sf.root")
    src = _build()
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with uproot.recreate(p) as f:
            src._write_all_data_to_uproot(f, "t", True)
        src._write_all_metadata_to_key(p, "t")

    lazy = AliasDataFrame.read_tree_lazy(p, "t"); lazy.draw_lazy = True
    eager = _build(); eager.draw_lazy = True
    assert "A" in lazy.lazy_subframes                       # registered as a lazy subframe
    assert lazy._lazy_reader.loaded_branches == set()       # provably lazy at construction

    out_l = lazy.draw(expr="A.a:x", type="profile", bins=12, return_data=True)
    out_e = eager.draw(expr="A.a:x", type="profile", bins=12, return_data=True)
    assert isinstance(out_l, tuple) and len(out_l) > 2
    _cmp_stats(out_l[2], out_e[2], "subframe_draw")         # lazy == eager

    # exact load: subframe index 'run' + main 'x'; decoy 'unused' never loaded
    assert lazy._lazy_reader.loaded_branches == {"run", "x"}, \
        f"loaded {sorted(lazy._lazy_reader.loaded_branches)} != {{run, x}}"
    assert "A" in lazy.list_subframes()                     # materialized on demand by the draw


@pytest.mark.invariance
def test_lazy_nested_subframe_column_draw(tmp_path):
    """Phase 13.58 (nested): recursive subframe-column lazy draw. A two-level chain
    main -> A -> B drawn via 'A.B.col:x' materializes the whole chain on demand (A, then B
    registered on A's frame at materialization) and matches eager exactly, loading only the
    join key + main branch (not the decoys at any level)."""
    import warnings as _warnings
    rng = np.random.default_rng(3); n = 300
    run = np.arange(n); x = rng.random(n); col = rng.random(n)

    def _build():
        B = AliasDataFrame(pd.DataFrame({"run": run.copy(), "col": col.copy(),
                                         "bdecoy": rng.random(n)}))
        A = AliasDataFrame(pd.DataFrame({"run": run.copy(), "a": rng.random(n)}))
        A.register_subframe("B", B, index_columns=["run"])
        m = AliasDataFrame(pd.DataFrame({"run": run.copy(), "x": x.copy(),
                                         "unused": rng.random(n)}))
        m.register_subframe("A", A, index_columns=["run"])
        return m

    p = str(tmp_path / "nested.root")
    src = _build()
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        with uproot.recreate(p) as f:
            src._write_all_data_to_uproot(f, "t", True)
        src._write_all_metadata_to_key(p, "t")

    lazy = AliasDataFrame.read_tree_lazy(p, "t"); lazy.draw_lazy = True
    eager = _build(); eager.draw_lazy = True
    assert lazy._lazy_reader.loaded_branches == set()

    out_l = lazy.draw(expr="A.B.col:x", type="profile", bins=12, return_data=True)
    out_e = eager.draw(expr="A.B.col:x", type="profile", bins=12, return_data=True)
    assert isinstance(out_l, tuple) and len(out_l) > 2
    _cmp_stats(out_l[2], out_e[2], "nested_subframe_draw")        # lazy == eager

    assert lazy._lazy_reader.loaded_branches == {"run", "x"}, \
        f"loaded {sorted(lazy._lazy_reader.loaded_branches)} != {{run, x}}"
    assert "A" in lazy.list_subframes()                          # outer materialized


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))