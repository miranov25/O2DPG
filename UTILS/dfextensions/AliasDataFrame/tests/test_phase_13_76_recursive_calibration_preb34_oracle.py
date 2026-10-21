"""PHASE_13_76 pre-B3.4 recursive calibration/subframe safety gate.

This test is one half of the binding pre-B3.4 entry gate.

This file protects recursive keyed calibration/subframe semantics:
- exact A -> B -> C stage truth against independent NumPy/pandas truth;
- independently shuffled calibration keys;
- one deliberately missing Calibration-A key with explicit neutral fill;
- a load-bearing wrong-key falsifier;
- alias-level ``fill_value=`` behavior;
- public ``draw(..., lazy=True)`` from an unmaterialized recursive final alias;
- a bounded 50,000-row real ROOT EAGER/LAZY gate;
- exact EAGER/LAZY product parity.

The complementary half of the B3.4 entry gate is the PHASE_13_76 FAST
real-data gallery, which provides broad draw-path coverage across public
surfaces and expression/vector/facet slots. B3.4 must not begin unless both
this recursive oracle and the FAST gallery are green and frozen.

Known gap
---------
BUG_AliasDataFrame_20261020_lazy_subframe_join_key_not_autoloaded

The current lazy root requires its subframe join key to be loaded before the
recursive request can succeed. This test pins that exact current behavior:
without preloading ``oracle_row_id`` the recursive request must raise the
named join-key ``KeyError``; on a fresh lazy graph, only ``oracle_row_id`` is
preloaded, after which ``draw(..., lazy=True)`` must load the real value
branch and produce exact recursive truth.

When the known gap is fixed, the negative pin must be superseded by a
positive automatic join-key-loading regression; it must not be silently
deleted.

No AliasDataFrame or dfdraw production code is modified by this test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from AliasDataFrame import AliasDataFrame


SYNTHETIC_N_ROWS = 4096
REAL_N_ROWS = 50000
SHUFFLE_SEED = 1376
DRAW_BINS = 32


@dataclass(frozen=True)
class RecursiveTruth:
    keys: np.ndarray
    raw: np.ndarray
    offset_a: np.ndarray
    scale_b: np.ndarray
    offset_c: np.ndarray
    stage_a: np.ndarray
    stage_b: np.ndarray
    stage_c: np.ndarray
    missing_key: int
    missing_mask: np.ndarray


@dataclass
class RecursiveGraph:
    root: AliasDataFrame
    cal_a: AliasDataFrame
    cal_b: AliasDataFrame
    cal_c: AliasDataFrame


def _binary_exact_truth(base: pd.DataFrame) -> RecursiveTruth:
    keys = np.asarray(base["oracle_row_id"], dtype=np.int64)
    raw = np.asarray(base["raw"], dtype=np.float64)
    assert len(keys) > 0
    assert len(np.unique(keys)) == len(keys)

    offset_a = 0.125 * (keys % 8)
    scale_b = 1.0 + 0.25 * (keys % 3)
    offset_c = -0.5 * (keys % 5)

    missing_pos = len(keys) // 2
    missing_key = int(keys[missing_pos])
    missing_mask = keys == missing_key

    stage_a = raw + offset_a
    stage_a = stage_a.copy()
    stage_a[missing_mask] = raw[missing_mask]
    stage_b = stage_a * scale_b
    stage_c = stage_b + offset_c

    return RecursiveTruth(
        keys=keys,
        raw=raw,
        offset_a=offset_a,
        scale_b=scale_b,
        offset_c=offset_c,
        stage_a=stage_a,
        stage_b=stage_b,
        stage_c=stage_c,
        missing_key=missing_key,
        missing_mask=missing_mask,
    )


def _build_tables(truth: RecursiveTruth, *, corrupt_a_key: bool = False):
    rng = np.random.default_rng(SHUFFLE_SEED)

    a_mask = ~truth.missing_mask
    a_keys = truth.keys[a_mask].copy()
    a_offsets = truth.offset_a[a_mask].copy()

    if corrupt_a_key:
        nonzero = np.flatnonzero(a_offsets != 0.0)
        assert len(nonzero) > 0
        victim = int(nonzero[0])
        a_keys[victim] = int(np.max(truth.keys)) + 1000003

    pa = rng.permutation(len(a_keys))
    pb = rng.permutation(len(truth.keys))
    pc = rng.permutation(len(truth.keys))

    a_df = pd.DataFrame({
        "oracle_row_id": a_keys[pa],
        "offset_A": a_offsets[pa],
    })
    b_df = pd.DataFrame({
        "oracle_row_id": truth.keys[pb],
        "raw_B": truth.raw[pb],
        "scale_B": truth.scale_b[pb],
    })
    c_df = pd.DataFrame({
        "oracle_row_id": truth.keys[pc],
        "offset_C": truth.offset_c[pc],
    })

    assert not np.array_equal(a_df["oracle_row_id"].to_numpy(), truth.keys[:len(a_df)])
    assert not np.array_equal(b_df["oracle_row_id"].to_numpy(), truth.keys)
    assert not np.array_equal(c_df["oracle_row_id"].to_numpy(), truth.keys)
    return a_df, b_df, c_df


def _attach_recursive_graph(root: AliasDataFrame, truth: RecursiveTruth, *, corrupt_a_key: bool = False) -> RecursiveGraph:
    a_df, b_df, c_df = _build_tables(truth, corrupt_a_key=corrupt_a_key)

    cal_a = AliasDataFrame(a_df)

    cal_b = AliasDataFrame(b_df)
    cal_b.register_subframe("A", cal_a, index_columns=["oracle_row_id"])
    cal_b.set_subframe_fill("A", fill_missing=0.0)
    cal_b.add_alias("stage_A", "raw_B + A.offset_A", dtype=np.float64)
    cal_b.add_alias("stage_B", "stage_A * scale_B", dtype=np.float64)

    cal_c = AliasDataFrame(c_df)
    cal_c.register_subframe("B", cal_b, index_columns=["oracle_row_id"])
    cal_c.add_alias("stage_C", "B.stage_B + offset_C", dtype=np.float64)
    cal_c.add_alias("raw_B", "B.raw_B", dtype=np.float64)

    root.register_subframe("C", cal_c, index_columns=["oracle_row_id"])
    root.add_alias(
        "oracle_final",
        "C.stage_C + (raw - C.raw_B)",
        dtype=np.float64,
    )

    assert "oracle_final" not in root.df.columns
    assert "oracle_final" not in set(root._get_materialized_aliases())

    return RecursiveGraph(root=root, cal_a=cal_a, cal_b=cal_b, cal_c=cal_c)


def _collect_stage_values(truth: RecursiveTruth, graph: RecursiveGraph):
    graph.cal_b.materialize_aliases(names=["stage_A", "stage_B"])
    graph.cal_c.materialize_aliases(names=["stage_C"])

    got_a = graph.cal_b.df.set_index("oracle_row_id").loc[truth.keys, "stage_A"].to_numpy(dtype=np.float64)
    got_b = graph.cal_b.df.set_index("oracle_row_id").loc[truth.keys, "stage_B"].to_numpy(dtype=np.float64)
    got_c = graph.cal_c.df.set_index("oracle_row_id").loc[truth.keys, "stage_C"].to_numpy(dtype=np.float64)
    got_final = graph.root.df["oracle_final"].to_numpy(dtype=np.float64)
    return got_a, got_b, got_c, got_final


def _assert_exact_stage_truth(truth: RecursiveTruth, graph: RecursiveGraph):
    got_a, got_b, got_c, got_final = _collect_stage_values(truth, graph)
    assert np.array_equal(got_a, truth.stage_a), "stage A truth mismatch"
    assert np.array_equal(got_b, truth.stage_b), "stage B truth mismatch"
    assert np.array_equal(got_c, truth.stage_c), "stage C truth mismatch"
    assert np.array_equal(got_final, truth.stage_c), "root final truth mismatch"

    pos = int(np.flatnonzero(truth.missing_mask)[0])
    assert got_a[pos] == truth.raw[pos]

    assert np.max(np.abs(got_a - truth.stage_a)) == 0.0
    assert np.max(np.abs(got_b - truth.stage_b)) == 0.0
    assert np.max(np.abs(got_c - truth.stage_c)) == 0.0
    assert np.max(np.abs(got_final - truth.stage_c)) == 0.0
    return got_a, got_b, got_c, got_final


def _draw_unmaterialized_final(truth: RecursiveTruth, graph: RecursiveGraph) -> dict:
    assert "oracle_final" not in graph.root.df.columns
    assert "oracle_final" not in set(graph.root._get_materialized_aliases())

    fig, ax, stats = graph.root.draw(
        "oracle_final",
        type="hist",
        # AliasDataFrame's established public contract materializes an
        # unmaterialized alias on draw when lazy=True.  On an eager ROOT-backed
        # frame this requests on-demand alias materialization; it does not
        # replace the root object or convert it to a lazy reader.
        lazy=True,
        bins=DRAW_BINS,
        range=(float(np.min(truth.stage_c)) - 1.0, float(np.max(truth.stage_c)) + 1.0),
        auto_title=True,
    )
    try:
        st = stats[0] if isinstance(stats, list) else stats
        assert int(st["n"]) == len(truth.stage_c)
        assert np.isclose(float(st["mean"]), float(np.mean(truth.stage_c)), rtol=1e-12, atol=1e-12)
        assert np.isclose(float(st["std"]), float(np.std(truth.stage_c)), rtol=1e-12, atol=1e-12)
        assert len(ax.lines) + len(ax.collections) + len(ax.patches) > 0
    finally:
        plt.close(fig)

    assert "oracle_final" in graph.root.df.columns
    assert "oracle_final" in set(graph.root._get_materialized_aliases())
    assert np.array_equal(graph.root.df["oracle_final"].to_numpy(dtype=np.float64), truth.stage_c)
    return st


def _synthetic_base() -> pd.DataFrame:
    keys = np.arange(SYNTHETIC_N_ROWS, dtype=np.int64)
    raw = (40 + (keys % 113)).astype(np.float64)
    return pd.DataFrame({"oracle_row_id": keys, "raw": raw})


def _resolve_real_root() -> str | None:
    explicit = os.environ.get("ADF_RECURSIVE_ORACLE_ROOT")
    if explicit:
        return explicit
    ts = os.environ.get("timeSeriesData")
    if ts:
        candidate = os.path.join(ts, "time_series_tracks_0.root")
        if os.path.exists(candidate):
            return candidate
    return None


def _write_bounded_real_fixture(source_root: str, fixture_path: Path) -> pd.DataFrame:
    import uproot

    with uproot.open(source_root) as f:
        if "treeTimeSeries" in f:
            tree = f["treeTimeSeries"]
        else:
            candidates = [obj for _, obj in f.items() if hasattr(obj, "num_entries") and hasattr(obj, "keys")]
            assert candidates, "no TTree-like object found in trusted ROOT input"
            tree = candidates[0]

        assert "ncl" in tree.keys(), "trusted real ROOT input has no 'ncl' branch"
        ncl = np.asarray(tree["ncl"].array(entry_start=0, entry_stop=REAL_N_ROWS, library="np"))

    assert len(ncl) == REAL_N_ROWS
    source = pd.DataFrame({
        "oracle_row_id": np.arange(REAL_N_ROWS, dtype=np.int64),
        "raw": ncl.astype(np.float64),
    })

    with uproot.recreate(fixture_path) as f:
        f.mktree("tree", {
            "oracle_row_id": source["oracle_row_id"].to_numpy(dtype=np.int64),
            "raw": source["raw"].to_numpy(dtype=np.float64),
        })
    return source


def _source_truth_from_fixture(fixture_path: Path) -> pd.DataFrame:
    import uproot
    with uproot.open(fixture_path) as f:
        arrays = f["tree"].arrays(["oracle_row_id", "raw"], library="np")
    return pd.DataFrame({
        "oracle_row_id": np.asarray(arrays["oracle_row_id"], dtype=np.int64),
        "raw": np.asarray(arrays["raw"], dtype=np.float64),
    })


def test_recursive_calibration_synthetic_exact_truth_and_missing_fill():
    base = _synthetic_base()
    truth = _binary_exact_truth(base)
    root = AliasDataFrame(base.copy())
    graph = _attach_recursive_graph(root, truth)
    graph.root.materialize_aliases(names=["oracle_final"])
    _assert_exact_stage_truth(truth, graph)


def test_recursive_calibration_unmaterialized_public_draw_discovers_graph():
    base = _synthetic_base()
    truth = _binary_exact_truth(base)
    graph = _attach_recursive_graph(AliasDataFrame(base.copy()), truth)
    _draw_unmaterialized_final(truth, graph)
    _assert_exact_stage_truth(truth, graph)


def test_recursive_calibration_wrong_key_falsifier_turns_oracle_red():
    base = _synthetic_base()
    truth = _binary_exact_truth(base)
    graph = _attach_recursive_graph(AliasDataFrame(base.copy()), truth, corrupt_a_key=True)
    graph.root.materialize_aliases(names=["oracle_final"])
    with pytest.raises(AssertionError, match="stage A truth mismatch"):
        _assert_exact_stage_truth(truth, graph)


def test_recursive_calibration_alias_level_fill_value_form():
    parent = AliasDataFrame(pd.DataFrame({"k": np.arange(12, dtype=np.int64), "x": np.arange(12, dtype=np.float64)}))
    child = AliasDataFrame(pd.DataFrame({"k": np.arange(8, dtype=np.int64), "delta": np.ones(8, dtype=np.float64)}))
    parent.register_subframe("S", child, index_columns=["k"])
    parent.add_alias("delta_filled", "S.delta", fill_value=0.0)
    parent.add_alias("corrected", "x + delta_filled")
    parent.materialize_aliases(names=["corrected"])
    expected = np.arange(12, dtype=np.float64)
    expected[:8] += 1.0
    assert np.array_equal(parent.df["corrected"].to_numpy(dtype=np.float64), expected)


def test_recursive_calibration_bounded_real_eager_lazy_live_graph(tmp_path):
    source_root = _resolve_real_root()
    if not source_root or not os.path.exists(source_root):
        pytest.skip("set ADF_RECURSIVE_ORACLE_ROOT (or timeSeriesData) to run the 50k real EAGER/LAZY gate")

    fixture = tmp_path / "recursive_calibration_real_50000.root"
    source = _write_bounded_real_fixture(source_root, fixture)
    independent = _source_truth_from_fixture(fixture)
    assert np.array_equal(source["oracle_row_id"].to_numpy(), independent["oracle_row_id"].to_numpy())
    assert np.array_equal(source["raw"].to_numpy(), independent["raw"].to_numpy())

    truth = _binary_exact_truth(independent)

    eager_root = AliasDataFrame.read_tree(str(fixture), "tree")
    eager_graph = _attach_recursive_graph(eager_root, truth)
    eager_stats = _draw_unmaterialized_final(truth, eager_graph)
    eager_values = _assert_exact_stage_truth(truth, eager_graph)

    lazy_gap_root = AliasDataFrame.read_tree_lazy(str(fixture), "tree")
    assert "oracle_row_id" not in set(lazy_gap_root.loaded_branches)
    assert "raw" not in set(lazy_gap_root.loaded_branches)
    lazy_gap_graph = _attach_recursive_graph(lazy_gap_root, truth)
    # Known gap:
    # BUG_AliasDataFrame_20261020_lazy_subframe_join_key_not_autoloaded
    # Pin the exact failure reason so an unrelated KeyError cannot satisfy
    # this negative-control contract.
    with pytest.raises(
        KeyError,
        match="join key 'oracle_row_id'",
    ):
        lazy_gap_graph.root.materialize_aliases(names=["oracle_final"])

    lazy_root = AliasDataFrame.read_tree_lazy(str(fixture), "tree")
    assert "oracle_row_id" not in set(lazy_root.loaded_branches)
    assert "raw" not in set(lazy_root.loaded_branches)

    lazy_root.ensure_columns("oracle_row_id")
    assert "oracle_row_id" in set(lazy_root.loaded_branches)
    assert "raw" not in set(lazy_root.loaded_branches)

    lazy_graph = _attach_recursive_graph(lazy_root, truth)
    lazy_stats = _draw_unmaterialized_final(truth, lazy_graph)
    assert "raw" in set(lazy_root.loaded_branches)
    lazy_values = _assert_exact_stage_truth(truth, lazy_graph)

    assert np.array_equal(eager_graph.root.df["oracle_row_id"].to_numpy(dtype=np.int64), lazy_graph.root.df["oracle_row_id"].to_numpy(dtype=np.int64))
    assert np.array_equal(eager_graph.root.df["raw"].to_numpy(dtype=np.float64), lazy_graph.root.df["raw"].to_numpy(dtype=np.float64))
    for eager_value, lazy_value in zip(eager_values, lazy_values):
        assert np.array_equal(eager_value, lazy_value)

    assert int(eager_stats["n"]) == int(lazy_stats["n"]) == REAL_N_ROWS
    assert np.isclose(float(eager_stats["mean"]), float(lazy_stats["mean"]), rtol=0.0, atol=0.0)
    assert np.isclose(float(eager_stats["std"]), float(lazy_stats["std"]), rtol=0.0, atol=0.0)
