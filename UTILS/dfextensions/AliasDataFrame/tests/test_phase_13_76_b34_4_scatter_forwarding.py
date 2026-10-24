"""PHASE_13_76_ADF B3.4.4 scatter forwarding dependency regression."""

import os

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.container import ErrorbarContainer

import AliasDataFrame as A

try:
    from dfextensions.dfdraw import DFDraw
    _HAVE_DFDRAW = True
except Exception:
    _HAVE_DFDRAW = False

needs_dfdraw = pytest.mark.skipif(
    not _HAVE_DFDRAW, reason="dfdraw not importable via dfextensions.dfdraw")


def _fixture(n=60):
    df = pd.DataFrame({
        "x": np.arange(n, dtype=float),
        "y": 5.0 + 0.4*np.arange(n, dtype=float),
        "sel": np.linspace(-1.0, 1.0, n),
        "ex": np.linspace(0.05, 0.30, n),
        "ey": np.linspace(0.10, 0.60, n),
        "ncl": np.linspace(50.0, 150.0, n),
        "size_col": np.linspace(20.0, 80.0, n),
        "color_col": np.linspace(-2.0, 2.0, n),
    })
    adf = A.AliasDataFrame(df.copy())
    adf.add_alias("xerror", "1.5*ex")
    adf.add_alias("yerror", "2.0*ey")
    adf.add_alias("marker_alias", "ncl > 125")
    adf.add_alias("size_alias", "1.25*size_col")
    return adf, df



def _lazy_root_fixture():
    """Genuine lazy ROOT-backed fixture for B3.4.4 RC-2."""
    fixture = os.path.join(
        os.path.dirname(__file__), "lazy_struct_fixture_clean.root")
    if not os.path.exists(fixture):
        pytest.skip("struct fixture not present (make_fixtures.py)")
    return A.AliasDataFrame.read_tree_lazy(fixture, "tree")

def _selected(df):
    mask = df.eval("sel > 0").to_numpy(dtype=bool)
    assert 0 < int(mask.sum()) < len(df)
    return df.loc[mask].reset_index(drop=True)


def _errorbar(ax):
    cs = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
    assert len(cs) == 1, f"expected 1 ErrorbarContainer, got {len(cs)}"
    return cs[0]


def _yerr(eb):
    segs = eb.lines[2]
    lc = segs[-1] if isinstance(segs, tuple) else segs
    return np.array([(s[1][1] - s[0][1]) / 2.0 for s in lc.get_segments()])


def _xerr(eb):
    segs = eb.lines[2]
    lc = segs[0] if isinstance(segs, tuple) else segs
    return np.array([(s[1][0] - s[0][0]) / 2.0 for s in lc.get_segments()])


def _surface_axis(adf, surface, spec, *, lazy=False):
    if surface == "draw":
        fig, ax, _ = adf.draw(
            spec["expr"], lazy=lazy,
            **{k: v for k, v in spec.items() if k != "expr"})
        return fig, ax
    if surface == "draw_batch":
        r = adf.draw_batch({"p": dict(spec)}, lazy=lazy, verbose=False)
        return r["p"]["fig"], r["p"]["ax"]
    if surface == "draw_figures":
        r = adf.draw_figures(
            [{"name": "f", "ncols": 1, "plots": [dict(spec)]}],
            lazy=lazy, verbose=False)
        axes = np.asarray(r["f"]["axes"], dtype=object).reshape(-1)
        return r["f"]["fig"], axes[0]
    raise AssertionError(surface)


@needs_dfdraw
class TestB344ScatterForwarding:

    @pytest.mark.parametrize("marker", ["o", "s", "^"])
    @pytest.mark.parametrize(
        "surface", ["draw", "draw_batch", "draw_figures"])
    def test_lazy_root_literal_marker_across_public_surfaces(
            self, surface, marker):
        """B3.4.4 RC-2: literal marker styles are plotting values, not lazy
        branch dependencies.

        This is deliberately ROOT-backed.  The pre-RC-1 endpoint raises
        BranchNotFoundError for these requests because the marker literal is
        promoted to a fake required branch before dfdraw sees it.
        """
        adf = _lazy_root_fixture()
        before_loaded = set(adf._lazy_reader.loaded_branches)

        spec = {
            "expr": "dedxTPC.dEdxMaxTPC:mult",
            "type": "scatter",
            "marker": marker,
        }

        fig, ax = _surface_axis(adf, surface, spec, lazy=True)
        try:
            # Public rendering completed and produced a scatter collection.
            paths = [
                c for c in ax.collections
                if hasattr(c, "get_paths") and c.get_paths()
            ]
            assert paths, (
                f"{surface}: marker={marker!r} produced no scatter collection")

            # Most importantly, the plotting literal must never become a
            # physical lazy-reader dependency.
            assert marker not in set(adf._lazy_reader.loaded_branches), (
                f"{surface}: literal marker {marker!r} was loaded as a branch")

            # The request is non-vacuous: real expression dependencies did
            # load from the ROOT-backed fixture.
            after_loaded = set(adf._lazy_reader.loaded_branches)
            assert after_loaded - before_loaded, (
                f"{surface}: lazy fixture loaded no real branches")
        finally:
            plt.close("all")

    def test_direct_dfdraw_yerr_positive_control(self):
        _, df = _fixture()
        selected = _selected(df)
        fig, ax, _ = DFDraw(selected).draw("y:x", type="scatter", yerr="ey")
        try:
            np.testing.assert_allclose(
                _yerr(_errorbar(ax)), selected["ey"].to_numpy(),
                rtol=0, atol=1e-9)
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("axis,kw,col", [
        ("y", "yerr", "ey"),
        ("x", "xerr", "ex"),
    ])
    def test_physical_error_semantics_on_draw(self, axis, kw, col):
        adf, df = _fixture()
        selected = _selected(df)
        fig, ax, _ = adf.draw(
            "y:x", type="scatter", selection="sel > 0", **{kw: col})
        try:
            got = _yerr(_errorbar(ax)) if axis == "y" else _xerr(_errorbar(ax))
            np.testing.assert_allclose(
                got, selected[col].to_numpy(), rtol=0, atol=1e-9)
        finally:
            plt.close(fig)

    @pytest.mark.parametrize("axis,kw,alias,col,scale", [
        ("y", "yerr", "yerror", "ey", 2.0),
        ("x", "xerr", "xerror", "ex", 1.5),
    ])
    def test_alias_error_semantics_on_draw_lazy(
            self, axis, kw, alias, col, scale):
        adf, df = _fixture()
        selected = _selected(df)
        fig, ax, _ = adf.draw(
            "y:x", type="scatter", selection="sel > 0",
            lazy=True, **{kw: alias})
        try:
            got = _yerr(_errorbar(ax)) if axis == "y" else _xerr(_errorbar(ax))
            np.testing.assert_allclose(
                got, scale*selected[col].to_numpy(), rtol=0, atol=1e-9)
        finally:
            plt.close(fig)

    def test_marker_expression_semantics_on_draw(self):
        adf, df = _fixture()
        selected = _selected(df)
        fig, ax, _ = adf.draw(
            "y:x", type="scatter", selection="sel > 0", marker="ncl > 125")
        try:
            expected = sorted([
                int((selected["ncl"] > 125).sum()),
                int((selected["ncl"] <= 125).sum()),
            ])
            got = sorted(len(c.get_offsets()) for c in ax.collections)
            assert got == expected
            assert len({len(c.get_paths()[0].vertices) for c in ax.collections}) == 2
        finally:
            plt.close(fig)

    # Boolean *expression* marker is the dfdraw-supported contract.
    # A materialized boolean marker column is currently a dfdraw limitation:
    # its column resolver returns raw bool values rather than mapping them to
    # marker styles, so B3.4.4 does not claim marker-alias semantics.

    @pytest.mark.parametrize("surface", ["draw", "draw_batch", "draw_figures"])
    def test_physical_yerr_across_public_surfaces(self, surface):
        adf, df = _fixture()
        selected = _selected(df)
        spec = {
            "expr": "y:x", "type": "scatter",
            "selection": "sel > 0", "yerr": "ey",
        }
        fig, ax = _surface_axis(adf, surface, spec)
        try:
            np.testing.assert_allclose(
                _yerr(_errorbar(ax)), selected["ey"].to_numpy(),
                rtol=0, atol=1e-9)
        finally:
            plt.close("all")

    @pytest.mark.parametrize("surface", ["draw", "draw_batch", "draw_figures"])
    def test_alias_yerr_across_public_surfaces_lazy(self, surface):
        adf, df = _fixture()
        selected = _selected(df)
        spec = {
            "expr": "y:x", "type": "scatter",
            "selection": "sel > 0", "yerr": "yerror",
        }
        fig, ax = _surface_axis(adf, surface, spec, lazy=True)
        try:
            np.testing.assert_allclose(
                _yerr(_errorbar(ax)), (2.0*selected["ey"]).to_numpy(),
                rtol=0, atol=1e-9)
        finally:
            plt.close("all")

    @staticmethod
    def _scatter_sizes(ax):
        colls = [c for c in ax.collections if hasattr(c, "get_sizes")]
        assert colls, "no scatter PathCollection"
        return np.asarray(colls[0].get_sizes(), dtype=float)

    def test_size_physical_semantics_on_draw(self):
        adf, df = _fixture()
        selected = _selected(df)

        fig_ref, ax_ref, _ = DFDraw(selected).draw(
            "y:x", type="scatter", size="size_col")
        expected = self._scatter_sizes(ax_ref).copy()

        fig, ax, _ = adf.draw(
            "y:x", type="scatter", selection="sel > 0", size="size_col")
        try:
            got = self._scatter_sizes(ax)
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
            assert len(np.unique(got)) > 1
        finally:
            plt.close(fig_ref)
            plt.close(fig)

    def test_size_alias_semantics_on_draw_lazy(self):
        adf, df = _fixture()
        selected = _selected(df).copy()
        selected["size_alias"] = 1.25 * selected["size_col"]

        fig_ref, ax_ref, _ = DFDraw(selected).draw(
            "y:x", type="scatter", size="size_alias")
        expected = self._scatter_sizes(ax_ref).copy()

        fig, ax, _ = adf.draw(
            "y:x", type="scatter", selection="sel > 0",
            size="size_alias", lazy=True)
        try:
            got = self._scatter_sizes(ax)
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
            assert len(np.unique(got)) > 1
        finally:
            plt.close(fig_ref)
            plt.close(fig)

    def test_color_expression_positive_control(self):
        adf, _ = _fixture()
        fig, ax, _ = adf.draw(
            "y:x", type="scatter", selection="sel > 0",
            color="abs(color_col)")
        try:
            arrays = [c.get_array() for c in ax.collections
                      if c.get_array() is not None]
            assert arrays
            assert np.all(np.concatenate([np.asarray(a) for a in arrays]) >= 0)
        finally:
            plt.close(fig)
