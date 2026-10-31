"""PHASE_13_101_ADF: bounded time-series eager ROOT reads (no ROOT fixture).

These tests execute the actual build_adf function body in an isolated namespace,
so they probe the forwarding and fail-fast contract without importing the very
large real-data gallery or loading a ROOT file. A real 1M-entry run is still
required on alma2 before the new developer fixture can be declared validated.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_SOURCE = (Path(__file__).resolve().parents[1] / "examples" /
           "time_series" / "time_series_draw.py")


class _FakeADF:
    def __init__(self, n=4):
        self.df = pd.DataFrame({"dummy": np.arange(n, dtype=np.float64)})
        self.materialized = None
        self.draw_lazy = False

    def materialize_aliases(self, names):
        self.materialized = tuple(names)


def _load_function():
    """Execute just the real build_adf function with instrumented dependencies."""
    source = _SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_SOURCE))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef)
               and node.name == "build_adf"]
    assert len(matches) == 1

    calls = []

    def root_to_adf(*args, **kwargs):
        calls.append(("eager", args, kwargs))
        return _FakeADF()

    def read_tree_lazy(*args, **kwargs):
        calls.append(("lazy", args, kwargs))
        return _FakeADF()

    class LazyNamespace:
        @staticmethod
        def read_tree_lazy(*args, **kwargs):
            return read_tree_lazy(*args, **kwargs)

    def apply_meta(adf, metadata):
        calls.append(("meta", metadata))

    def addTimeQuantiles(adf, **kwargs):
        calls.append(("quantiles", kwargs))

    ns = {
        "np": np,
        "AliasDataFrame": LazyNamespace,
        "root_to_adf": root_to_adf,
        "apply_meta": apply_meta,
        "df_TimeSeriesAliases": {"alias": 1},
        "df_TimeSeriesMeta": {"meta": 1},
        "addTimeQuantiles": addTimeQuantiles,
    }
    body = ast.Module(body=matches, type_ignores=[])
    exec(compile(ast.fix_missing_locations(body), str(_SOURCE), "exec"), ns)
    return ns["build_adf"], calls


def test_bounded_read_passes_one_million_to_existing_eager_reader():
    build, calls = _load_function()
    adf = build("source.root", sample=None, lazy=False, entry_stop=1_000_000)
    assert calls[0] == ("eager", ("source.root",), {"entry_stop": 1_000_000})
    assert len(adf.df) == 4  # fake reader returned four rows, no post-load sampling
    assert adf.df["oracle_row_id"].tolist() == [0, 1, 2, 3]
    assert adf.materialized == ("sector", "time_s")
    assert adf.draw_lazy is True


def test_default_call_is_bytewise_compatible_at_reader_boundary():
    build, calls = _load_function()
    build("source.root")
    assert calls[0] == ("eager", ("source.root",), {})


@pytest.mark.parametrize("bad", [0, -1, True, 3.5, "1000000"])
def test_invalid_bounds_fail_before_reader_is_called(bad):
    build, calls = _load_function()
    with pytest.raises(ValueError, match="positive integer"):
        build("source.root", entry_stop=bad)
    assert calls == []


def test_lazy_bounded_read_refuses_instead_of_silently_loading_all():
    build, calls = _load_function()
    with pytest.raises(ValueError, match="eager reading only"):
        build("source.root", lazy=True, entry_stop=1_000_000)
    assert calls == []


def test_optional_sampling_happens_after_bounded_read():
    build, calls = _load_function()
    adf = build("source.root", sample=0.5, entry_stop=np.int64(1_000_000))
    assert calls[0] == ("eager", ("source.root",), {"entry_stop": 1_000_000})
    assert len(adf.df) == 2
    assert adf.df.index.tolist() == [0, 1]
    assert adf.df["oracle_row_id"].nunique() == 2
