"""PHASE_13_76_ADF B3.2b — R10 no-metadata subframe mode matrix.

The matrix covers parent eager/lazy x child eager/lazy for physical requests and
unresolved requests under raise vs warn+skip.  A separate post-load-alias family
pins Architect Decision A without inventing a new lazy-child metadata-alias
contract.
"""

import warnings

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from AliasDataFrame.AliasDataFrame import AliasDataFrame
except (ImportError, ModuleNotFoundError):
    from AliasDataFrame import AliasDataFrame

try:
    from dfextensions.dfdraw import DFDraw  # noqa: F401
    HAVE_DFDRAW = True
except Exception:
    HAVE_DFDRAW = False

needs_dfdraw = pytest.mark.skipif(
    not HAVE_DFDRAW, reason="dfdraw not importable via dfextensions.dfdraw"
)

DYN_P1_2 = (
    "DYN-P1-2: eager/in-memory parent + lazy child physical reference does "
    "not auto-resolve"
)
B3_P0_3 = (
    "B3-P0-3: metadata-poor bad-only warn discards exact runtime unresolved "
    "evidence and terminal reconciliation false-refuses"
)


def _require_real_uproot():
    uproot = pytest.importorskip("uproot")
    if getattr(uproot, "__version__", "") == "stub":
        pytest.skip("real uproot unavailable in local coder environment")
    return uproot


@pytest.fixture
def r10_root_file(tmp_path):
    uproot = _require_real_uproot()
    path = tmp_path / "v12_r10_modes.root"
    with uproot.recreate(str(path)) as f:
        f.mktree("tree", {"k": "int64", "x": "float64", "unused": "float64"})
        f["tree"].extend({
            "k": np.arange(4, dtype=np.int64),
            "x": np.array([10.0, 20.0, 30.0, 40.0]),
            "unused": np.array([101.0, 102.0, 103.0, 104.0]),
        })
        f.mktree("Ch", {"k": "int64", "v": "float64", "unused_child": "float64"})
        f["Ch"].extend({
            "k": np.arange(4, dtype=np.int64),
            "v": np.array([1.0, 2.0, 3.0, 4.0]),
            "unused_child": np.array([201.0, 202.0, 203.0, 204.0]),
        })
    return str(path)


def _build_modes(path, parent_mode, child_mode):
    if parent_mode == "eager":
        parent = AliasDataFrame.read_tree(path, "tree")
    else:
        parent = AliasDataFrame.read_tree_lazy(path, "tree")

    if child_mode == "eager":
        # Established mixed-lazy contract: with an eager child, the parent
        # join key is structural setup, not expression-slot discovery.  This
        # is the same boundary pinned by PHASE_13_77 A4.4.
        if parent_mode == "lazy":
            parent.ensure_branches(["k"])
        child = AliasDataFrame.read_tree(path, "Ch")
        parent.register_subframe("S", child, index_columns=["k"])
    else:
        parent.register_subframe_lazy("S", path, tree_name="Ch", index_columns=["k"])
        child = None
    return parent, child


def _draw(parent, expr, *, on_subframe_error="raise", on_error="raise"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return parent.draw_batch(
                {"p": {"expr": expr, "type": "scatter"}},
                on_subframe_error=on_subframe_error,
                on_error=on_error,
                verbose=False,
            )
        finally:
            plt.close("all")


def _assert_lazy_evidence(parent, parent_mode, child_mode):
    if parent_mode == "lazy":
        loaded = set(map(str, getattr(parent._lazy_reader, "loaded_branches", set())))
        assert {"k", "x"} <= loaded
        assert "unused" not in loaded
    if child_mode == "lazy":
        assert parent._subframe_loaded.get("S", False)


PHYSICAL_CASES = [
    pytest.param("eager", "eager", id="physical-parent_eager-child_eager"),
    pytest.param(
        "eager", "lazy",
        marks=pytest.mark.xfail(strict=True, raises=ValueError, reason=DYN_P1_2),
        id="physical-parent_eager-child_lazy-DYN-P1-2",
    ),
    pytest.param("lazy", "eager", id="physical-parent_lazy-child_eager"),
    pytest.param("lazy", "lazy", id="physical-parent_lazy-child_lazy"),
]

WARN_BAD_CASES = [
    pytest.param("eager", "eager", id="bad_warn-parent_eager-child_eager"),
    pytest.param(
        "eager", "lazy",
        marks=pytest.mark.xfail(strict=True, raises=RuntimeError, reason=DYN_P1_2),
        id="bad_warn-parent_eager-child_lazy-DYN-P1-2",
    ),
    pytest.param("lazy", "eager", id="bad_warn-parent_lazy-child_eager"),
    pytest.param(
        "lazy", "lazy",
        id="bad_warn-parent_lazy-child_lazy",
    ),
]

RAISE_BAD_CASES = [
    pytest.param("eager", "eager", id="bad_raise-parent_eager-child_eager"),
    pytest.param(
        "eager", "lazy",
        marks=pytest.mark.xfail(strict=True, raises=RuntimeError, reason=DYN_P1_2),
        id="bad_raise-parent_eager-child_lazy-DYN-P1-2",
    ),
    pytest.param("lazy", "eager", id="bad_raise-parent_lazy-child_eager"),
    pytest.param("lazy", "lazy", id="bad_raise-parent_lazy-child_lazy"),
]

ALIAS_POSTLOAD_CASES = [
    pytest.param("eager", "eager", id="alias_postload-parent_eager-child_eager"),
    pytest.param("eager", "lazy", id="alias_postload-parent_eager-child_lazy"),
    pytest.param("lazy", "eager", id="alias_postload-parent_lazy-child_eager"),
    pytest.param("lazy", "lazy", id="alias_postload-parent_lazy-child_lazy"),
]


@needs_dfdraw
@pytest.mark.invariance
class TestR10SubframeModeMatrix:
    @pytest.mark.parametrize("parent_mode,child_mode", PHYSICAL_CASES)
    def test_r10_physical_existing(self, r10_root_file, parent_mode, child_mode):
        parent, _ = _build_modes(r10_root_file, parent_mode, child_mode)
        result = _draw(parent, "S.v:x")
        assert result is not None
        _assert_lazy_evidence(parent, parent_mode, child_mode)

    @pytest.mark.parametrize("parent_mode,child_mode", RAISE_BAD_CASES)
    def test_r10_unresolved_bad_raise_policy(self, r10_root_file, parent_mode, child_mode):
        parent, _ = _build_modes(r10_root_file, parent_mode, child_mode)
        with pytest.raises(ValueError):
            _draw(parent, "S.nosuch:x", on_subframe_error="raise", on_error="skip")

    @pytest.mark.parametrize("parent_mode,child_mode", WARN_BAD_CASES)
    def test_r10_unresolved_bad_warn_skip_policy(self, r10_root_file, parent_mode, child_mode):
        parent, _ = _build_modes(r10_root_file, parent_mode, child_mode)
        result = _draw(parent, "S.nosuch:x", on_subframe_error="warn", on_error="skip")
        assert result is not None
        assert parent._last_draw_prep_state.plan_reconciliation_errors == ()

    @pytest.mark.parametrize("parent_mode,child_mode", ALIAS_POSTLOAD_CASES)
    def test_r10_alias_added_after_child_data_load(self, r10_root_file, parent_mode, child_mode):
        parent, child = _build_modes(r10_root_file, parent_mode, child_mode)
        if child_mode == "lazy":
            # This family tests Architect Decision A (logical definitions can be
            # added after data load).  It intentionally does not invent a
            # separate metadata-provided lazy-child alias contract.
            parent.ensure_subframe("S")
            child = parent.get_subframe("S")
        child.add_alias("twice", "v*2")

        result = _draw(parent, "S.twice:x")
        assert result is not None
