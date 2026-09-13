"""PHASE_13_76 B3.3 draw_figures effective-defaults regressions.

Locks the common request cascade used by both preparation and rendering:
    top-level defaults < per-figure defaults < per-plot spec

These tests protect alias discovery/materialization, vector-slot preparation,
short-form/dict-form parity, and caller non-mutation.
"""

import copy

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from AliasDataFrame import AliasDataFrame


def _close_results(results):
    for result in results.values():
        fig = result.get("fig") if isinstance(result, dict) else None
        if fig is not None:
            plt.close(fig)


def test_draw_figures_effective_spec_precedence_top_figure_plot():
    merged = AliasDataFrame._draw_figures_effective_plot_spec(
        {"selection": "top", "bins": 10, "title": "top"},
        {"defaults": {"selection": "figure", "bins": 20}, "plots": []},
        {"expr": "y:x", "selection": "plot"},
    )
    assert merged["selection"] == "plot"
    assert merged["bins"] == 20
    assert merged["title"] == "top"
    assert merged["expr"] == "y:x"


def test_draw_figures_scalar_alias_only_in_figure_defaults_is_materialized_and_applied():
    adf = AliasDataFrame(pd.DataFrame({
        "x": np.arange(10.0),
        "y": np.arange(10.0),
    }))
    adf.add_alias("figure_selection", "x > 4")

    specs = [{
        "name": "scalar-default",
        "defaults": {"selection": "figure_selection"},
        "plots": [{"expr": "y", "type": "hist", "bins": 5}],
    }]
    results = adf.draw_figures(specs, lazy=True, verbose=False)
    try:
        stats = results["scalar-default"]["stats"][0]
        assert stats["n"] == 5
        assert stats["min"] == 5.0
        assert stats["max"] == 9.0
    finally:
        _close_results(results)


def test_draw_figures_selection_vector_aliases_only_in_figure_defaults_keep_two_branches():
    adf = AliasDataFrame(pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 100),
        "y": np.linspace(0.0, 10.0, 100),
    }))
    adf.add_alias("sel_lo", "x < 0")
    adf.add_alias("sel_hi", "x >= 0")

    specs = [{
        "name": "vector-default",
        "defaults": {"selection_vector": ["sel_lo", "sel_hi"]},
        "plots": [{
            "expr": "y:x",
            "type": "profile",
            "bins": 10,
            "range": [-1.0, 1.0],
        }],
    }]
    results = adf.draw_figures(specs, lazy=True, verbose=False)
    try:
        stats = results["vector-default"]["stats"][0]
        assert isinstance(stats, list)
        assert len(stats) == 2
        assert [branch["n"] for branch in stats] == [50, 50]
    finally:
        _close_results(results)


def test_draw_figures_short_form_selection_vector_in_figure_defaults_uses_same_vector_path():
    adf = AliasDataFrame(pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 100),
        "y": np.linspace(0.0, 10.0, 100),
    }))
    adf.add_alias("sel_lo", "x < 0")
    adf.add_alias("sel_hi", "x >= 0")

    specs = [{
        "name": "short-selection-vector",
        "defaults": {"selection_vector": ["sel_lo", "sel_hi"]},
        "plots": ["y:x"],
    }]
    specs_before = copy.deepcopy(specs)

    results = adf.draw_figures(specs, lazy=True, verbose=False)
    try:
        stats = results["short-selection-vector"]["stats"][0]
        assert isinstance(stats, list)
        assert len(stats) == 2
        assert [branch["n"] for branch in stats] == [50, 50]
        assert specs == specs_before
        assert isinstance(specs[0]["plots"][0], str)
    finally:
        _close_results(results)


def test_draw_figures_short_form_weights_vector_in_figure_defaults_uses_same_vector_path():
    x = np.repeat(np.linspace(-0.9, 0.9, 10), 10)
    y = np.tile(np.array([0.0, 10.0] * 5), 10)
    adf = AliasDataFrame(pd.DataFrame({"x": x, "y": y}))
    adf.add_alias("w_low", "where(y < 5, 10.0, 1.0)")
    adf.add_alias("w_high", "where(y >= 5, 10.0, 1.0)")

    specs = [{
        "name": "short-weights-vector",
        "defaults": {
            "type": "profile",
            "weights_vector": ["w_low", "w_high"],
            "bins": 10,
            "range": [-1.0, 1.0],
        },
        "plots": ["y:x"],
    }]
    specs_before = copy.deepcopy(specs)

    results = adf.draw_figures(
        specs, lazy=True, clear_after=False, verbose=False)
    try:
        stats = results["short-weights-vector"]["stats"][0]
        assert isinstance(stats, list)
        assert len(stats) == 2
        assert [branch["n"] for branch in stats] == [100, 100]
        ax = results["short-weights-vector"]["axes"][0]
        branch_central = [ax.lines[0].get_ydata(), ax.lines[3].get_ydata()]
        assert np.nanmax(branch_central[0]) < 2.0
        assert np.nanmin(branch_central[1]) > 8.0
        assert "w_low" in adf.df.columns
        assert "w_high" in adf.df.columns
        assert specs == specs_before
        assert isinstance(specs[0]["plots"][0], str)
    finally:
        _close_results(results)


def test_draw_figures_effective_defaults_do_not_mutate_caller_specs_or_defaults():
    adf = AliasDataFrame(pd.DataFrame({
        "x": np.linspace(-1.0, 1.0, 100),
        "y": np.linspace(0.0, 10.0, 100),
    }))
    adf.add_alias("sel_lo", "x < 0")
    adf.add_alias("sel_hi", "x >= 0")

    defaults = {"title": "caller-title"}
    specs = [{
        "name": "nonmutation",
        "defaults": {"selection_vector": ["sel_lo", "sel_hi"]},
        "plots": [{
            "expr": "y:x",
            "type": "profile",
            "bins": 10,
            "range": [-1.0, 1.0],
        }],
    }]
    defaults_before = copy.deepcopy(defaults)
    specs_before = copy.deepcopy(specs)

    results = adf.draw_figures(
        specs, defaults=defaults, lazy=True, verbose=False)
    try:
        assert defaults == defaults_before
        assert specs == specs_before
        assert "vector_compose" not in specs[0]["plots"][0]
    finally:
        _close_results(results)


def test_draw_figures_qualified_reference_rewrite_does_not_mutate_caller_specs():
    parent = AliasDataFrame(pd.DataFrame({
        "id": np.arange(6),
        "y": np.arange(6.0),
    }))
    child = AliasDataFrame(pd.DataFrame({
        "id": np.arange(6),
        "mask": np.array([0, 0, 0, 1, 1, 1]),
    }))
    parent.register_subframe("Child", child, "id")

    specs = [{
        "name": "qualified-nonmutation",
        "defaults": {"selection": "Child.mask > 0"},
        "plots": [{"expr": "y", "type": "hist", "bins": 3}],
    }]
    specs_before = copy.deepcopy(specs)

    results = parent.draw_figures(specs, lazy=True, verbose=False)
    try:
        assert results["qualified-nonmutation"]["stats"][0]["n"] == 3
        assert specs == specs_before
        assert specs[0]["defaults"]["selection"] == "Child.mask > 0"
    finally:
        _close_results(results)
