"""PHASE_13_71_ADF - describe_lazy() user-facing lazy-state diagnostic.

Covers: eager report, main lazy reader report, no-side-effects (invariance),
max_items truncation, lazy-subframe block, chain-reader tolerance, as_dict form.
"""
import numpy as np
import pandas as pd
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame

try:
    import uproot  # noqa: F401
    _HAS_UPROOT = True
except Exception:
    _HAS_UPROOT = False

pytestmark = pytest.mark.skipif(not _HAS_UPROOT, reason="uproot required for lazy readers")

N = 30


def _make_main(tmp_path):
    df = pd.DataFrame({
        "run": np.arange(N) % 3,
        "a": np.arange(N) * 1.0, "b": np.arange(N) + 5.0,
        "c": np.arange(N) - 2.0, "d": np.arange(N) * 2.0, "e": np.arange(N) * 0.5,
    })
    f = str(tmp_path / "main.root")
    AliasDataFrame(df).export_tree(f, treename="tree")
    return f


def _make_sub(tmp_path):
    df = pd.DataFrame({"run": np.arange(3), "corr": np.arange(3) * 10.0, "w": np.arange(3) + 1.0})
    f = str(tmp_path / "sub.root")
    AliasDataFrame(df).export_tree(f, treename="tree")
    return f


# ---- T-LAZYDESC-1: eager ADF reports not lazy ----
def test_LAZYDESC_1_eager_reports_not_lazy(capsys):
    adf = AliasDataFrame(pd.DataFrame({"a": [1, 2, 3]}))
    adf.describe_lazy()
    out = capsys.readouterr().out
    assert "Not a lazy AliasDataFrame" in out


# ---- T-LAZYDESC-2: lazy ADF reports available/loaded/df-cols/not-loaded ----
def test_LAZYDESC_2_lazy_reports_branches(tmp_path, capsys):
    adf = AliasDataFrame.read_tree_lazy(_make_main(tmp_path), tree_name="tree")
    adf.ensure_columns(["a", "b"])
    adf.describe_lazy()
    out = capsys.readouterr().out
    for token in ("Lazy state", "Available branches", "Loaded branches",
                  "DataFrame columns", "Available but not loaded"):
        assert token in out, token
    assert "a" in out and "b" in out          # loaded
    assert "c" in out and "d" in out and "e" in out  # available-but-not-loaded


# ---- T-LAZYDESC-3: no side effects (INVARIANCE) ----
@pytest.mark.invariance
def test_LAZYDESC_3_no_side_effects(tmp_path):
    adf = AliasDataFrame.read_tree_lazy(_make_main(tmp_path), tree_name="tree")
    adf.ensure_columns(["a"])
    before_cols = list(adf.df.columns)
    before_loaded = set(getattr(adf._lazy_reader, "loaded_branches", set()))
    adf.describe_lazy()
    adf.describe_lazy(as_dict=True)
    after_cols = list(adf.df.columns)
    after_loaded = set(getattr(adf._lazy_reader, "loaded_branches", set()))
    assert before_cols == after_cols
    assert before_loaded == after_loaded


# ---- T-LAZYDESC-4: max_items truncates ----
def test_LAZYDESC_4_max_items_truncates(tmp_path, capsys):
    adf = AliasDataFrame.read_tree_lazy(_make_main(tmp_path), tree_name="tree")
    adf.describe_lazy(max_items=2)          # 6 available, none loaded -> truncation
    out = capsys.readouterr().out
    assert "more)" in out and "... (+" in out


# ---- T-LAZYDESC-5: lazy subframe block ----
def test_LAZYDESC_5_lazy_subframe_block(tmp_path, capsys):
    adf = AliasDataFrame.read_tree_lazy(_make_main(tmp_path), tree_name="tree")
    adf.register_subframe_lazy("corrmap", _make_sub(tmp_path), tree_name="tree",
                               index_columns=["run"])
    adf.describe_lazy()
    out = capsys.readouterr().out
    assert "Lazy subframes" in out
    assert "corrmap" in out
    assert "Available branches" in out and "Loaded branches" in out
    assert "Index columns" in out and "run" in out


# ---- T-LAZYDESC-6: chain reader tolerant (no exception) ----
def test_LAZYDESC_6_chain_reader_tolerant(tmp_path):
    f0 = _make_main(tmp_path)
    df2 = pd.DataFrame({"run": np.arange(N) % 3, "a": np.arange(N) + 100.0,
                        "b": np.arange(N).astype(float), "c": np.arange(N) * 1.0,
                        "d": np.arange(N) * 1.0, "e": np.arange(N) * 1.0})
    f1 = str(tmp_path / "main2.root"); AliasDataFrame(df2).export_tree(f1, treename="tree")
    adf = AliasDataFrame.read_chain_lazy([f0 + ":tree", f1 + ":tree"])
    adf.describe_lazy()                      # must not raise
    d = adf.describe_lazy(as_dict=True)
    assert d["lazy"] is True and d["main"] is not None


# ---- T-LAZYDESC-7: as_dict structured form (family consistency) ----
def test_LAZYDESC_7_as_dict_form(tmp_path):
    adf = AliasDataFrame.read_tree_lazy(_make_main(tmp_path), tree_name="tree")
    adf.ensure_columns(["a", "b"])
    d = adf.describe_lazy(as_dict=True)
    assert set(d) == {"lazy", "main", "subframes"}
    assert d["lazy"] is True
    assert set(d["main"]["loaded"]) == {"a", "b"}
    assert {"c", "d", "e"} <= set(d["main"]["not_loaded"])
    # eager frame -> lazy False
    e = AliasDataFrame(pd.DataFrame({"x": [1]})).describe_lazy(as_dict=True)
    assert e["lazy"] is False
