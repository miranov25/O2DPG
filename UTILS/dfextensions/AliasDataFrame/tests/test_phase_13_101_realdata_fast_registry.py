"""PHASE_13_101 opt-in FAST runner static/contract tests (no ROOT needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# In the installed checkout these modules live under examples/time_series.
sys.path.insert(0, str(Path(__file__).resolve().parent / "../examples/time_series"))
# The above relative path is correct after tests/ installation.  When running
# this delivery bundle directly in a test sandbox, also accept the colocated
# module copies; never install a second runner as a fallback in production.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import phase_13_101_realdata_oracles as o
import time_series_draw_invariance as h


def test_declared_three_cases_validate_in_existing_stage_a_registry():
    cases = o.cases(h)
    assert tuple(c.case_id for c in cases) == o.CASE_IDS
    assert h.validate_registry(cases) == []
    assert len(h._a6_3_fraction_cases("fixture.root", gallery_module=type("G", (), {})())) == 22
    assert len(h._a6_3_fraction_runners()) == 22


def test_opt_in_cli_is_disjoint_from_historical_fraction_mode():
    parser = h.build_stage_a_cli_parser()
    fast = parser.parse_args(["source.root", "--adf101-fast", "--manifest", "x.json",
                              "--pdf", "x.pdf", "--strict"])
    canon = parser.parse_args(["source.root", "--sample", "0.20", "--seed", "42",
                               "--manifest", "x.json", "--pdf", "x.pdf", "--strict"])
    assert fast.adf101_fast and fast.sample is None
    assert canon.sample == 0.20 and not canon.adf101_fast
    assert len(o.cases(h)) == 3


def test_raw_mean_and_discrete_quantile_reference_no_product_dependency():
    x = np.array([0., .25, .5, .75, 1., np.inf, np.nan])
    y = np.array([1., 3., 7., 11., 15., 4., 5.])
    edges = np.array([0., .5, 1.])
    count, mean = o._profile_raw(x, y, edges)
    np.testing.assert_array_equal(count, [2, 3])
    np.testing.assert_allclose(mean, [2, 11])
    np.testing.assert_allclose(o._quantile_raw(x,y,edges,.5), [2, 11])
    # Last right edge is inclusive; an incorrect x branch must be detectable.
    with pytest.raises(AssertionError):
        o._assert_numbered("mutated pair", count, [3, 2])


def test_factored_legend_and_deliberate_falsifiers():
    ungrouped = ["dcar_tpc_vertex vs sector", "dcar_tpc vs sector", "q=15%", "median", "q=90%"]
    assert o._check_factored_labels(ungrouped, ["dcar_tpc_vertex", "dcar_tpc", "q=15%", "median", "q=90%"]) == 5
    assert o._legend_falsifier(ungrouped, ["dcar_tpc_vertex", "dcar_tpc", "q=15%", "median", "q=90%"], False)
    grouped = ["dcar_tpc_vertex", "dcar_tpc", "0", "1", "q=15%", "median", "q=90%"]
    assert o._check_factored_labels(grouped, grouped) == 7
    assert o._legend_falsifier(grouped, grouped, True)
    with pytest.raises(AssertionError):
        o._check_factored_labels(grouped + ["0"], grouped)


def test_six_physical_facets_and_genuine_sparse_global_groups():
    rows = []
    for t in (-1.1, 0.0, 1.1):
        for q in (-20, 20):
            for side in (0, 1):
                if t < 0 and q > 0 and side == 1:  # deliberately sparse
                    continue
                for sector in (0.25, 1.25, 2.25, 3.25):
                    rows.append((t, q, side, sector))
    df = pd.DataFrame(rows, columns=["tgl", "qpt_bin10", "side_type", "sector"])
    raw = {k: np.asarray(df[k]) for k in df.columns}
    labels, masks = o._facets(raw, np.ones(len(df), bool), grouped=True)
    assert len(labels) == len(masks) == 6
    assert all(mask.sum() > 0 for mask in masks)
    assert any((raw["side_type"][mask] == 1).sum() == 0 for mask in masks)


def test_build_failure_is_not_false_green(tmp_path):
    class FailingGallery:
        @staticmethod
        def build_adf(root_path, *, sample, lazy, entry_stop):
            assert sample is None and lazy is False and entry_stop == 1_000_000
            raise RuntimeError("fixture deliberately unavailable")
    root = tmp_path / "fixture.root"
    root.write_bytes(b"dummy fixture")
    manifest, pdf = tmp_path / "fast.json", tmp_path / "fast.pdf"
    results, doc, code = o.run_fast(str(root), manifest_path=str(manifest),
                                    pdf_path=str(pdf), h=h, gallery_module=FailingGallery())
    assert code != 0
    assert all(r.status == "INVALID_FIXTURE" for r in results)
    assert all(r.executed_comparisons == 0 for r in results)
    assert len(json.loads(manifest.read_text())["cases"]) == 3
    assert doc["provenance"]["stage_a_gate"] == "ADF101_FAST_TS_BOUNDED"


def test_mocked_public_three_case_run_reconciles_manifest_and_pdf(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    rows = []
    for t in (-1.1, -0.7, -0.3, 0.1, 0.5, 1.1):
        for qpt in (-20, 20):
            for side in (0, 1):
                for s in range(36):
                    for rep in (0, 1):
                        rows.append({"tgl": t, "qpt_bin10": qpt, "side_type": side,
                                     "sector": s + 0.25 + .05 * rep, "ncl": 100,
                                     "dcar_tpc_vertex": float(.1*np.sin(s)+.02*side+.01*rep),
                                     "dcar_tpc": float(.2*np.cos(s)+.01*t+.01*rep)})
    raw_df = pd.DataFrame(rows)
    monkeypatch.setattr(o, 'ENTRY_STOP', len(raw_df))

    class FakeAdf:
        df = raw_df
        def draw(self, expr, **kw):
            fig, axes = plt.subplots(3 if isinstance(kw.get('facet_by'), list) else 2,
                                     2 if isinstance(kw.get('facet_by'), list) else 3)
            raw = {col: np.asarray(self.df[col]) for col in self.df.columns}
            group = isinstance(kw.get('facet_by'), list)
            selected = o._selected(raw, group)
            _, masks = o._facets(raw, selected, group)
            x0 = np.asarray(raw['sector'][selected],dtype=float)
            ys = [np.asarray(raw[y][selected],dtype=float)
                  for y in ('dcar_tpc_vertex','dcar_tpc')]
            axy = list(np.asarray(axes).flat)
            if 'minmax' == kw.get('range'):
                # Build the product stats independently from the public calls
                # requested by the runner.  No ROOT/dfdraw available here.
                res=[]
                for x,y in zip((x0, x0+.25), ys):
                    edges=np.linspace(x.min(),x.max(),21)
                    items=[]
                    for facet, mask in zip(o._facets(raw,selected,False)[0],masks):
                        n, mean=o._profile_raw(x[mask],y[mask],edges)
                        items.append((o._public_interval_label(facet), pd.DataFrame({'count':n, 'y_mean':mean,
                                      'x_low':edges[:-1], 'x_high':edges[1:]})))
                    res.append({'groups':[pair[0] for pair in items],
                                'per_group': {k:{'profile_data':v} for k,v in items}})
                return fig, axes, res
            edges=np.linspace(0,36,37)
            centers=(edges[:-1]+edges[1:])/2
            groups=(0,1) if group else (None,)
            for ax, cell in zip(axy,masks):
                for g in groups:
                    mask=cell & (raw['side_type'][selected]==g) if group else cell
                    if not np.any(mask):
                        continue
                    for i, expr_name in enumerate(('dcar_tpc_vertex','dcar_tpc')):
                        for q in o.QUANTILES:
                            ys_q=o._quantile_raw(x0[mask],ys[i][mask],edges,q)
                            label = ("median" if q == .50 else f"q={q:.0%}")
                            ln,=ax.plot(centers,ys_q, label=label if not group else None)
                            if group:
                                # 2-D grouped renderer annotates artists.
                                ln._dfdraw_facet_semantic={'kind':'quantile','group':g,
                                                          'quantile':q,'expression':expr_name}
                            # 1-D ungrouped renderer uses public q-labels,
                            # without private semantic tags.
            labels=['dcar_tpc_vertex','dcar_tpc']
            if group:
                labels+=['0','1']
                for i in range(5):
                    fig.text(.01,.15+.1*i, f'({i})')
            labels+=list(o.QUANTILE_LABELS)
            fig.legend(handles=[Line2D([],[],label=label) for label in labels])
            return fig,axes,{}

    class FakeGallery:
        called=0
        @classmethod
        def build_adf(cls, root_path, **kw):
            cls.called+=1
            assert kw=={'sample':None,'lazy':False,'entry_stop':len(raw_df)}
            return FakeAdf()

    root=tmp_path/'source.root'
    root.write_bytes(b'dummy')
    results, doc, status=o.run_fast(str(root),manifest_path=str(tmp_path/'manifest.json'),
                                   pdf_path=str(tmp_path/'gallery.pdf'), h=h,
                                   gallery_module=FakeGallery)
    assert [r.status for r in results]==['PASS','PASS','PASS'], [(r.case_id,r.detail) for r in results]
    assert status==0
    assert FakeGallery.called==1
    assert doc['reconciliation']['n_results']==3 and not doc['reconciliation']['gating']
    assert all(r.executed_comparisons>0 for r in results)
    assert (tmp_path/'gallery.pdf').stat().st_size>3000
    assert len(json.loads((tmp_path/'manifest.json').read_text())['cases'])==3


def test_raw_data_derives_unmaterialized_qpt_bin10_from_physical_qpt():
    # Real time-series data carry qpt, while qpt_bin10 is a lazy alias.
    df = pd.DataFrame({
        "sector": [1.0, 2.0, 3.0, 4.0],
        "tgl": [-0.7, 0.0, 0.8, 1.0],
        "side_type": [0, 1, 0, 1],
        "dcar_tpc_vertex": [.1, .2, -.3, .1],
        "dcar_tpc": [.2, .1, -.2, -.1],
        "ncl": [100, 110, 120, 130],
        "qpt": np.array([-4.53, -.15, .15, 5.55], dtype=np.float32),
    })
    adf = type("AliasNotMaterialized", (), {"df": df})()
    assert "qpt_bin10" not in df.columns
    raw = o._raw_data(adf)
    np.testing.assert_array_equal(raw["qpt_bin10"], np.array([-45, -1, 1, 55], dtype=np.int8))
    assert raw["qpt_bin10"].dtype == np.dtype("int8")
    assert "qpt_bin10" not in df.columns  # L0 does not call ADF alias resolution


def test_raw_reference_ignores_materialized_alias_if_physical_qpt_available():
    df = pd.DataFrame({
        "sector": [.1, 1.1], "tgl": [0., .1], "side_type": [0, 1],
        "dcar_tpc_vertex": [0., 0.], "dcar_tpc": [0., 0.],
        "ncl": [90, 90], "qpt": [.12, -.22],
        "qpt_bin10": [99, 99],  # deliberately poisoned ADF materialization
    })
    raw = o._raw_data(type("BadAlias", (), {"df": df})())
    np.testing.assert_array_equal(raw["qpt_bin10"], np.array([1, -2], dtype=np.int8))
    assert not np.array_equal(raw["qpt_bin10"], df["qpt_bin10"].to_numpy())


def test_raw_data_fails_closed_when_qpt_and_qpt_bin10_both_missing():
    df = pd.DataFrame({
        "sector": [.1], "tgl": [0.], "side_type": [0],
        "dcar_tpc_vertex": [0.], "dcar_tpc": [0.], "ncl": [90],
    })
    with pytest.raises(ValueError, match="INVALID_FIXTURE: physical qpt branch absent"):
        o._raw_data(type("Missing", (), {"df": df})())


def test_public_interval_labels_match_1d_facet_api_and_not_interval_repr():
    cases = [
        (pd.Interval(-1.403, -.933, closed="right"), "-1.40--0.93"),
        (pd.Interval(0, 2.51), "0.0-2.5"),
        (pd.Interval(-20, 0), "-20-0"),
        (pd.Interval(.001, .091), "0.001-0.091"),
    ]
    for interval, display in cases:
        assert o._public_interval_label(interval) == display
        assert o._public_interval_label(interval) != str(interval)


def test_ungrouped_quantile_lines_resolve_actual_labeled_artists_without_metadata():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    curves = []
    for expr in ("dcar_tpc_vertex", "dcar_tpc"):
        for q, label in zip(o.QUANTILES, o.QUANTILE_LABELS):
            line, = ax.plot([.5, 1.5], [q, q+.01], label=label)
            curves.append(line)
    for expr_i, expr in enumerate(("dcar_tpc_vertex", "dcar_tpc")):
        for q_i, q in enumerate(o.QUANTILES):
            assert o._semantic_quantile_line(ax, expr, q) is curves[expr_i*3+q_i]
    plt.close(fig)


def test_ungrouped_quantile_fails_closed_on_missing_or_reordered_public_artist():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for label in ["q=15%", "median", "q=90%", "median", "q=15%", "q=90%"]:
        ax.plot([0, 1], [0, 1], label=label)
    with pytest.raises(AssertionError, match="quantile artist order"):
        o._semantic_quantile_line(ax, "dcar_tpc", .50)
    ax.lines[-1].remove()
    with pytest.raises(AssertionError, match="quantile artist order"):
        o._semantic_quantile_line(ax, "dcar_tpc", .90)
    plt.close(fig)


def test_ungrouped_quantile_fails_closed_on_partial_semantic_metadata():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for label in list(o.QUANTILE_LABELS) * 2:
        ax.plot([0, 1], [0, 1], label=label)
    ax.lines[0]._dfdraw_facet_semantic = {
        "kind": "quantile", "expression": "dcar_tpc_vertex", "quantile": .15}
    with pytest.raises(AssertionError, match="partial/mixed"):
        o._semantic_quantile_line(ax, "dcar_tpc", .50)
    plt.close(fig)
