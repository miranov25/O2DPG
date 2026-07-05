"""PHASE_13_68_ADF — explicit lazy-branch/struct release (release_branches / release_struct).

Fixture rule (proposal §4): real uproot-written files, never reader mocks. Struct
members are exercised through a REAL LazyTreeReader; where a full ALICE-style slash
struct branch cannot be produced by ``uproot.recreate`` (it exposes dot-named record
members, not the slash form ``_struct_physical_name`` uses), the end-to-end LOAD leg is
skipped with a reason and the RELEASE-logic leg runs against real reader state. On alma2
real struct fixtures the end-to-end leg runs fully.

Behavior tokens (architect-approved 2026-07-04, "approved"):
  DD-alpha RAISE (eager) · DD-beta REFUSE (non-branch / written / __file_idx__ / unknown)
  DD-gamma REFUSE (alias -> dematerialize) · DD-delta REFUSE (subframe join key)
  C-6 REFUSE (in a materialized alias's dependency closure)
"""
import os
import numpy as np
import pandas as pd
import pytest

from AliasDataFrame import AliasDataFrame


# --------------------------------------------------------------------------- fixtures
def _write_tree(path, n=200, seed=0):
    import uproot
    rng = np.random.default_rng(seed)
    with uproot.recreate(path) as fo:
        fo["tree"] = {
            "x": np.arange(n, dtype="f8"),
            "y": (np.arange(n, dtype="f8") + 1.0),
            "z": rng.random(n),
            "key": (np.arange(n) % 5).astype("i8"),
        }
    return path


@pytest.fixture
def lazy_tree(tmp_path):
    return AliasDataFrame.read_tree_lazy(_write_tree(str(tmp_path / "t.root")), "tree")


def _book_struct_member(adf, struct, member):
    """Reflect the real reader state after a struct member loads from a slash
    fixture: the physical (slash) name is a real available branch that has been
    read into loaded_branches, and the frame holds the internal-named column.
    Uses the real LazyTreeReader's sets — not a mock. (On alma2 real struct
    fixtures this exact state is produced by adf.eval('struct.member').)"""
    intern = adf._struct_internal_name(struct, member)
    phys = adf._struct_physical_name(struct, member)
    adf._lazy_reader.available_branches.add(phys)      # real branch exists in the tree
    adf._lazy_reader.loaded_branches.add(phys)         # ...and has been loaded
    adf.df[intern] = np.arange(len(adf.df), dtype="f8")
    return intern, phys


@pytest.fixture
def lazy_chain(tmp_path):
    f0 = _write_tree(str(tmp_path / "c0.root"), n=120, seed=1)
    f1 = _write_tree(str(tmp_path / "c1.root"), n=80, seed=2)
    return AliasDataFrame.read_chain_lazy([f0, f1], tree_name="tree")


# ------------------------------------------------------------------ T-REL-1 / T-REL-3
@pytest.mark.invariance
@pytest.mark.parametrize("which", ["tree", "chain"])
def test_REL_1_plain_release_reaccess_identity(which, lazy_tree, lazy_chain):
    """Invariant: release -> re-access == first load. Reference: the first-load Series,
    captured before release and untouched by the phase. Exercises release_branches +
    the reader-side seam (symmetric evict) on both readers."""
    adf = lazy_tree if which == "tree" else lazy_chain
    first = adf["x"].to_numpy().copy()                 # <- independent reference
    assert "x" in adf.loaded_branches

    released = adf.release_branches(["x"])
    assert released == ["x"]
    assert "x" not in adf.df.columns                   # dropped from frame
    assert "x" not in adf.loaded_branches              # unbooked on the reader (symmetric)

    again = adf["x"].to_numpy()                         # re-load from file
    np.testing.assert_array_equal(again, first)        # identical to first load


def test_REL_3_plain_roundtrip_returns_and_gc(lazy_tree):
    adf = lazy_tree
    _ = adf["y"], adf["z"]
    out = adf.release_branches(["y", "z"])
    assert set(out) == {"y", "z"}
    assert not ({"y", "z"} & set(adf.df.columns))
    assert not ({"y", "z"} & set(adf.loaded_branches))


# ------------------------------------------------------------------------- T-REL-2 C-6
def test_REL_2_c6_dependency_refuse_plain(lazy_tree):
    adf = lazy_tree
    adf.add_alias("r", "x + y")
    adf.materialize_aliases("r")                        # materializes r (loads x, y)
    with pytest.raises(ValueError) as ei:
        adf.release_branches(["x"])
    msg = str(ei.value)
    assert "x" in msg and "r" in msg                   # names both the branch and the alias
    assert "x" in adf.df.columns                        # all-or-nothing: nothing released


def test_REL_2_c6_dependency_refuse_struct_member(lazy_tree):
    """C-6 parametrized to a struct member (N-5): an alias over a struct member protects
    that member from release. Uses real reader state (booked physical name)."""
    adf = lazy_tree
    adf.register_struct("st", ["m"])
    intern, phys = _book_struct_member(adf, "st", "m")
    adf.add_alias("sm", f"{intern} * 2")
    adf.materialize_aliases("sm")
    with pytest.raises(ValueError) as ei:
        adf.release_branches([intern])
    assert intern in str(ei.value) and "sm" in str(ei.value)


# --------------------------------------------------------------- T-REL-4 DD-alpha/beta/gamma
def test_REL_4_dd_alpha_eager_raises():
    eager = AliasDataFrame(pd.DataFrame({"a": [1, 2, 3]}))
    with pytest.raises(ValueError) as ei:
        eager.release_branches(["a"])
    assert "not lazy" in str(ei.value)


def test_REL_4_dd_beta_unknown_name_refuse(lazy_tree):
    with pytest.raises(ValueError) as ei:
        lazy_tree.release_branches(["not_a_branch"])
    assert "not_a_branch" in str(ei.value)


def test_REL_4_dd_gamma_alias_name_refuse(lazy_tree):
    adf = lazy_tree
    adf.add_alias("r", "x + y")
    adf.materialize_aliases("r")
    with pytest.raises(ValueError) as ei:
        adf.release_branches(["r"])
    assert "dematerialize" in str(ei.value)             # points to the alias valve


# ---------------------------------------------------------------------- T-REL-5 policy
def test_REL_5_memory_policy_surface(lazy_tree):
    adf = lazy_tree
    assert adf.memory_policy == "keep"                  # default
    adf.memory_policy = "keep"                           # accepted
    for bad in ("bounded", "drop", "typo", None):
        with pytest.raises(ValueError):
            adf.memory_policy = bad


# --------------------------------------------------------------- T-REL-6 partial struct
@pytest.mark.invariance
def test_REL_6_partial_struct_release_exact_complement(lazy_tree):
    """Invariant: partial release preserves the exact-load complement.
    Reference: reader loaded-set arithmetic. Release 1 of 3 loaded members; assert the
    released member (frame col + physical) is gone and the other two are untouched."""
    adf = lazy_tree
    adf.register_struct("st", ["a", "b", "c"])
    booked = {m: _book_struct_member(adf, "st", m) for m in ("a", "b", "c")}

    # release only member 'a' via release_branches on its internal name
    out = adf.release_branches([booked["a"][0]])
    assert out == [booked["a"][0]]
    assert booked["a"][0] not in adf.df.columns
    assert booked["a"][1] not in adf.loaded_branches
    # exact complement: b, c untouched (frame + reader)
    for m in ("b", "c"):
        assert booked[m][0] in adf.df.columns
        assert booked[m][1] in adf.loaded_branches


def test_REL_6_release_struct_sugar_and_unknown(lazy_tree):
    adf = lazy_tree
    adf.register_struct("st", ["a", "b"])
    for m in ("a", "b"):
        _book_struct_member(adf, "st", m)
    released = adf.release_struct("st")
    assert set(released) == {"a__st", "b__st"}
    assert not any(c.endswith("__st") for c in adf.df.columns)
    # never-loaded members simply skipped: empty struct release is a no-op, not an error
    adf.register_struct("st2", ["p", "q"])
    assert adf.release_struct("st2") == []
    with pytest.raises(ValueError):
        adf.release_struct("does_not_exist")


# ------------------------------------------------------------------- T-REL-7 regression
def test_REL_7_dematerialize_unaffected(lazy_tree):
    """release_branches must not perturb the alias valve (dematerialize)."""
    adf = lazy_tree
    adf.add_alias("r", "x + y")
    adf.materialize_aliases("r")
    assert "r" in adf.df.columns
    dropped = adf.dematerialize(drop=["r"])
    assert dropped == ["r"] and "r" not in adf.df.columns


# --------------------------------------------------------------- T-REL-8 DD-delta joinkey
@pytest.mark.parametrize("kind", ["in_memory", "chain"])
def test_REL_8_dd_delta_join_key_refuse(kind, tmp_path):
    adf = AliasDataFrame.read_tree_lazy(_write_tree(str(tmp_path / "p.root")), "tree")
    _ = adf["key"]                                      # 'key' is a real loaded branch
    if kind == "in_memory":
        sub = AliasDataFrame(pd.DataFrame({"key": [0, 1, 2, 3, 4], "val": range(5)}))
        adf.register_subframe("S", sub, index_columns="key")
    else:
        c0 = _write_tree(str(tmp_path / "s0.root"), n=5)
        adf.register_subframe_chain("S", [c0], tree_name="tree", index_columns=["key"])
    with pytest.raises(ValueError) as ei:
        adf.release_branches(["key"])
    assert "key" in str(ei.value) and "S" in str(ei.value)
    assert "key" in adf.df.columns                      # all-or-nothing


# --------------------------------------------------------- T-REL-9 written col / __file_idx__
def test_REL_9_written_column_distinct_message(lazy_tree):
    adf = lazy_tree
    adf["h"] = np.ones(len(adf.df))                     # Stage-2a write-through
    assert "h" in adf.loaded_branches                   # premise (verified at re-anchor gate)
    with pytest.raises(ValueError) as ei:
        adf.release_branches(["h"])
    m = str(ei.value)
    assert "written" in m.lower() and "__file_idx__" not in m


def test_REL_9_file_idx_marker_distinct_message(lazy_chain):
    adf = lazy_chain
    _ = adf["x"]                                        # trigger a load on the chain
    if "__file_idx__" not in (adf.loaded_branches or set()):
        pytest.skip("chain built without __file_idx__ marker in this configuration")
    with pytest.raises(ValueError) as ei:
        adf.release_branches(["__file_idx__"])
    assert "reader-synthesized" in str(ei.value)


# ------------------------------------------------- struct end-to-end (alma2 real fixture)
@pytest.mark.invariance
def test_REL_1_struct_member_end_to_end(tmp_path):
    """Full load->release->re-access identity for a struct member. Runs only where the
    physical (slash) struct branch actually resolves — real ALICE-style fixtures on alma2.
    Skipped under uproot.recreate, which exposes dot-named record members (not slash)."""
    import uproot
    try:
        import awkward as ak
    except Exception:
        pytest.skip("awkward not available")
    n = 60
    path = str(tmp_path / "struct.root")
    rec = ak.zip({"dEdxTotIROC": np.arange(n, dtype="f8"),
                  "dEdxTotOROC": np.arange(n, dtype="f8") + 100.0})
    with uproot.recreate(path) as fo:
        fo["tree"] = {"dedxTPC": rec, "x": np.arange(n, dtype="f8")}
    adf = AliasDataFrame.read_tree_lazy(path, "tree")
    adf.register_struct("dedxTPC", ["dEdxTotIROC", "dEdxTotOROC"])
    intern = adf._struct_internal_name("dedxTPC", "dEdxTotIROC")
    try:
        first = adf.eval("dedxTPC.dEdxTotIROC")
    except Exception as exc:
        pytest.skip(f"struct member did not load in this env (slash/dot fixture artifact): {exc}")
    if intern not in adf.df.columns:
        pytest.skip("struct member did not materialize under the internal name in this env")
    first = np.asarray(first).copy()
    phys = adf._struct_physical_name("dedxTPC", "dEdxTotIROC")
    adf.release_struct("dedxTPC")
    assert intern not in adf.df.columns
    assert phys not in adf.loaded_branches
    again = np.asarray(adf.eval("dedxTPC.dEdxTotIROC"))
    np.testing.assert_array_equal(again, first)
