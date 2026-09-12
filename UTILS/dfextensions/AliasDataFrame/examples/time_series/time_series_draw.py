"""
time_series_draw.py  — dfdraw full coverage gallery (v2.2, PHASE_13_56_ADF)

Usage from IPython:
    adf = build_adf("time_series_tracks_0.root")          # full, ~4 min
    adf = build_adf("time_series_tracks_0.root", 0.2)     # 20% sample, ~1 min

    fig07_profile_dca_sector(adf)    # run one figure
    run_all_pdf(adf)                  # save all to ts_draw_gallery.pdf

Groups:
  G1 — Basic primitives         (fig01–fig09)
  G2 — Group and facet          (fig10–fig14)
  G3 — Time axis                (fig15–fig17)
  G4 — Differential analysis    (fig18–fig22)
  G5 — Fitting                  (fig23–fig26)
  G6 — Advanced composition     (fig27–fig31)
  G7 — Full stack ADF+GB        (fig32–fig34, optional)
  G8 — ADF dispatch closure     (fig35–fig39)
  G9 — Error/window semantics   (fig40–fig42)
  G10 — Semantic oracle gallery (fig43)
  G11 — Hardening oracles       (fig44 weights-vector, fig45 public-surface)

Known limitations:
  central='median' + group_by=: silently returns mean (KNOWN.grouped_central_median).
  fig17 facet_by + time_format=: ScalarFormatter on panels, not HH:MM (S-4, Batch 4).
  hexbin: range= raises ValueError (Phase 13.51 Batch 3 guard).
"""

import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from perfmonitor import PerformanceLogger

from dfextensions.AliasDataFrame import AliasDataFrame
from dfextensions.dfdraw.drawer import DFDraw
from time_series import (
    root_to_adf, apply_meta, addTimeQuantiles,
    calibBiasResolution, calibVertex,
    df_TimeSeriesAliases, df_TimeSeriesMeta,
)

logger = PerformanceLogger("perf_log.txt")

BASE_SEL = "(ncl>60)&(abs(dcar_tpc_vertex)<10)"
ITS_SEL  = f"{BASE_SEL}&(hasITSTPC)"


def build_adf(root_path, sample=None, lazy=False, tree_name="tree"):
    """Load ADF. build_adf(path, 0.2) gives ~1 min dev run (20% sample).

    Phase 13.58.ADF (D4): the additive ``lazy=`` parameter swaps ONLY the constructor --
    eager ``root_to_adf()`` (default, behaviour unchanged) or lazy ``read_tree_lazy()`` --
    and shares every post-read step below, so the eager and lazy galleries differ only in
    data loading (no plotting logic is forked or replicated). ``lazy=True`` requires
    ``sample=None``: a lazy read makes an N-row frame, ``sample(frac)`` shrinks it, then a
    branch load returns full-N rows and crashes in ``_merge_loaded_data`` -- sampled-lazy is
    out of scope (Phase 13.58 §2). ``tree_name`` is used only by the lazy path (the eager
    default resolution is untouched); pass the gallery file's tree name if it is not
    ``"tree"``.
    """
    if lazy:
        if sample is not None:
            raise ValueError(
                "build_adf(lazy=True) does not support sampling (sample must be None): "
                "sampled-lazy crashes in _merge_loaded_data and is out of scope for "
                "Phase 13.58.ADF. Run the lazy gallery unsampled."
            )
        adf = AliasDataFrame.read_tree_lazy(root_path, tree_name)
    else:
        adf = root_to_adf(root_path)
    adf.draw_lazy = True
    apply_meta(adf, df_TimeSeriesAliases)
    apply_meta(adf, df_TimeSeriesMeta)
    addTimeQuantiles(adf, varname="timeMS", step=10000, step2=100)
    if sample is not None:
        adf.df = adf.df.sample(frac=sample, random_state=42).reset_index(drop=True)
    adf.materialize_aliases(names=["sector", "time_s"])
    print(f"ADF ready: {len(adf.df):,} tracks"
          + (f" ({int(sample*100)}% sample)" if sample else "")
          + (" [lazy]" if lazy else ""))
    return adf


def validate_lazy_vs_eager(root_path, tree_name="tree"):
    """Phase 13.58.ADF D4 / AC-1 / AC-2 -- gallery lazy-vs-eager double-run (SERVER gate).

    Requires ROOT (eager root_to_adf) + dfdraw; run on the server, unsampled. Asserts:
      * AC-1 (clean, genuinely lazy): the lazy ADF uses read_tree_lazy, branches load on
        demand (a figure-only branch like 'ncl' is NOT force-materialized by setup, and
        IS loaded after the figure that needs it), and every gallery figure renders with
        no error (AD-TS-DRAW-001).
      * AC-2 (PP-5 identity): a representative set of draws produce identical stats lazy vs
        eager at the data/stats level (NOT pixel). The comparison is defensive -- it equates
        whatever numeric stats both runs return -- so it cannot false-green on a key name.
    Sandbox cannot run this (no ROOT/dfdraw); it is the alma2 secondary integration gate.
    The primary gate is the dedicated synthetic test (tests/test_phase1358_lazy_timeseries.py).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    eager = build_adf(root_path, lazy=False, tree_name=tree_name)
    lazy = build_adf(root_path, lazy=True, tree_name=tree_name)
    assert lazy._lazy_reader is not None, "lazy build must use read_tree_lazy (not eager-in-disguise)"

    # provably lazy: a figure-only branch must not be force-materialized by setup
    forced = set(lazy._lazy_reader.loaded_branches)
    assert "ncl" not in forced, "ncl must not be force-loaded by build_adf setup"

    # AC-2 stats identity on representative draws (return_data=True; defensive key match)
    checks = [
        dict(expr="ncl", type="hist", bins=50, selection="ncl>30"),
        dict(expr="dcar_tpc_vertex:tgl", type="profile", bins=50, selection=BASE_SEL),
    ]
    for kw in checks:
        re_ = eager.draw(return_data=True, **kw)
        rl_ = lazy.draw(return_data=True, **kw)
        se = re_[2] if isinstance(re_, tuple) and len(re_) > 2 and isinstance(re_[2], dict) else {}
        sl = rl_[2] if isinstance(rl_, tuple) and len(rl_) > 2 and isinstance(rl_[2], dict) else {}
        compared = 0
        for key in (set(se) & set(sl)):
            try:
                a = np.asarray(se[key], dtype=float)
                b = np.asarray(sl[key], dtype=float)
            except (TypeError, ValueError):
                continue
            if a.shape == b.shape and a.size:
                assert np.allclose(a, b, equal_nan=True), f"lazy != eager stats for {kw}, key '{key}'"
                compared += 1
        assert compared > 0, f"no comparable numeric stats produced for {kw}"

    assert "ncl" in lazy._lazy_reader.loaded_branches, "ncl should load lazily after its draw"

    # AC-1 clean run: every gallery figure renders lazily with no error
    fig_funcs = [v for k, v in sorted(globals().items())
                 if k.startswith("fig") and callable(v)]
    for fn in fig_funcs:
        fig = fn(lazy)
        if fig is not None:
            plt.close(fig)
    print(f"validate_lazy_vs_eager: OK -- {len(fig_funcs)} figures rendered lazily; "
          f"lazy==eager stats on {len(checks)} representative draws")


def _add(pdf, fig, title):
    """Save figure to PDF with suptitle + PDF bookmark."""
    fig.suptitle(title, fontsize=8, color="gray", y=1.0, ha="left", x=0.01)
    pdf.savefig(fig, bbox_inches="tight")
    pdf.attach_note(title)
    plt.close(fig)


# ── G1 — Basic primitives ─────────────────────────────────────────────────────

def fig01_hist_ncl(adf):
    """G1.01 — hist — TPC cluster count distribution"""
    return adf.draw("ncl", type="hist", bins=50, selection="ncl>30", auto_title=True)

def fig02_hist_time(adf):
    """G1.02 — hist+time_format — Run time occupancy"""
    return adf.draw("time_s", type="hist", bins=100, time_format="%H:%M", auto_title=True)

def fig03_hist_cumulative(adf):
    """G1.03 — cumulative hist — CDF of TPC cluster count"""
    return adf.draw("ncl", type="hist", bins=50, cumulative=True, selection="ncl>30", auto_title=True)

def fig04_scatter_dca_tgl(adf):
    """G1.04 — scatter+range+sample — DCA_r vs tgl"""
    # auto_title= not supported by scatter (BUG_dfdraw_20260609_scatter_auto_title)
    return adf.draw("dcar_tpc_vertex:tgl", selection=BASE_SEL, type="scatter", sample=50000, range=((-1.5, 1.5), (-1.5, 1.5)))

def fig05_hist2d_dca_sector(adf):
    """G1.05 — hist2d+range — DCA_r density per sector"""
    # range= is (x_range, y_range) in dfdraw hist2d; expression is y:x so sector=x, dca=y
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="hist2d", bins=36, range=((0, 36), (-1.5, 1.5)), auto_title=True)

def fig06_hexbin_dca_sector(adf):
    """G1.06 — hexbin — DCA_r density per sector (hexagonal bins)"""
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="hexbin", auto_title=True)

def fig07_profile_dca_sector(adf):
    """G1.07 — profile — Mean DCA_r per sector"""
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="profile", bins=36, auto_title=True)

def fig08_profile2d_dca_tgl_sector(adf):
    """G1.08 — profile2d — Mean DCA_r in (tgl x sector) plane"""
    return adf.draw("dcar_tpc_vertex:tgl:sector", selection=BASE_SEL, type="profile", bins=36, auto_title=True)

def fig09_scatter3d(adf):
    """G1.09 — scatter3d+sample — 3D DCA_r point cloud"""
    return adf.draw("dcar_tpc_vertex:tgl:sector", selection=BASE_SEL, type="scatter3d", sample=20000, auto_title=True)


# ── G2 — Group and facet ──────────────────────────────────────────────────────

def fig10_profile_groupby_side(adf):
    """G2.10 — profile+group_by — Mean DCA_r per sector, A vs C side"""
    # side_type: 0=A, 1=C, 2=both/central — filter to A and C only
    return adf.draw("dcar_tpc_vertex:sector", selection=f"{BASE_SEL}&(side_type<2)", type="profile", bins=36, group_by="side_type", auto_title=True)

def fig11_hist_groupby_side(adf):
    """G2.11 — hist+group_by — TPC cluster count A vs C side"""
    return adf.draw("ncl", type="hist", bins=40, selection=f"(ncl>30)&(side_type<2)", group_by="side_type", auto_title=True)

def fig12_profile_facet_side(adf):
    """G2.12 — profile+facet_by — DCA_r per sector, A/C side panels"""
    return adf.draw("dcar_tpc_vertex:sector", selection=f"{BASE_SEL}&(side_type<2)", type="profile", bins=36, facet_by="side_type", auto_title=True)

def fig13_profile_facet_nd(adf):
    """G2.13 — N-D facet — DCA_r in side_type x qpt_bin10 grid"""
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="profile", bins=36, facet_by=["side_type", "qpt_bin10"], facet_by_bins=[2, 3], auto_title=True)

def fig14_group_and_facet(adf):
    """G2.14 — group_by+facet_by — ITS clusters vs time, side x sector-group"""
    return adf.draw("nClITS:time_s", selection=ITS_SEL, type="profile", bins=100, group_by="side_type", facet_by="sector", facet_by_bins=4, min_entries=20, auto_title=True)


# ── G3 — Time axis ────────────────────────────────────────────────────────────

def fig15_profile_time(adf):
    """G3.15 — profile+time_format — DCA_r drift stability vs time"""
    return adf.draw("dcar_tpc_vertex:time_s", selection=BASE_SEL, type="profile", bins=100, time_format="%H:%M", auto_title=True)

def fig16_hist2d_time(adf):
    """G3.16 — hist2d+time_format — ITS cluster density vs time"""
    return adf.draw("nClITS:time_s", selection=ITS_SEL, type="hist2d", bins=[100, 50], time_format="%H:%M", auto_title=True)

def fig17_profile_facet_time(adf):
    """G3.17 — facet_by+time_format — DCA_r vs time per side (S-4: epoch ticks expected)"""
    # KNOWN S-4: per-panel DateFormatter deferred to Batch 4. Panels show epoch seconds.
    return adf.draw("dcar_tpc_vertex:time_s", selection=BASE_SEL, type="profile", bins=50, facet_by="side_type", time_format="%H:%M", auto_title=True)


# ── G4 — Differential analysis ────────────────────────────────────────────────

def fig18_delta_sector13(adf):
    """G4.18 — selection_vector+delta — Sector-13 ITS cluster deficit vs time"""
    return adf.draw("nClITS:time_s", selection=ITS_SEL, type="profile", bins=100, selection_vector=["(abs(sector-13)<2)", "(abs(sector-13)>=2)&(sector<36)"], normalize="delta", min_entries=50, auto_title=True)

def fig19_ratio_time(adf):
    """G4.19 — normalize=ratio — DCA_r early vs late run ratio per sector"""
    t_mid = adf.df["time_s"].median()
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="profile", bins=36, selection_vector=[f"time_s<{t_mid}", f"time_s>={t_mid}"], normalize="ratio", min_entries=100, auto_title=True)

def fig20_pull_time(adf):
    """G4.20 — normalize=pull — Statistical significance of DCA_r time drift"""
    t_mid = adf.df["time_s"].median()
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="profile", bins=36, selection_vector=[f"time_s<{t_mid}", f"time_s>={t_mid}"], normalize="pull", min_entries=100, auto_title=True)

def fig21_delta_side(adf):
    """G4.21 — normalize=delta — A minus C DCA_r asymmetry per sector"""
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="profile", bins=36, selection_vector=["side_type==0", "side_type==1"], normalize="delta", auto_title=True)

G4_22_BINS = 36
G4_22_RANGE = (-0.5, 35.5)
G4_22_FACETS = (0, 1)

def fig22_delta_faceted(adf):
    """G4.22 — normalize=delta+facet_by — Early/late DCA_r delta per side panel"""
    t_mid = adf.df["time_s"].median()
    return adf.draw(
        "dcar_tpc_vertex:sector",
        selection=f"{BASE_SEL}&(side_type<2)",
        type="profile",
        bins=G4_22_BINS,
        range=G4_22_RANGE,
        selection_vector=[f"time_s<{t_mid}", f"time_s>={t_mid}"],
        normalize="delta",
        facet_by="side_type",
        auto_title=True,
    )


# ── G5 — Fitting ──────────────────────────────────────────────────────────────

def fig23_hist_fit(adf):
    """G5.23 — hist+fit=gauss — DCA_r distribution Gaussian fit"""
    return adf.draw("dcar_tpc_vertex", type="hist", bins=100, fit="gauss", selection=BASE_SEL, auto_title=True)

def fig24_profile_fit(adf):
    """G5.24 — profile+fit=pol2 — Mean DCA_r vs tgl with pol2 fit"""
    return adf.draw("dcar_tpc_vertex:tgl", selection=BASE_SEL, type="profile", bins=50, fit="pol2", auto_title=True)

def fig25_profile_fit_median(adf):
    """G5.25 — central=median+fit=pol2 — Median DCA_r vs tgl with pol2 fit"""
    return adf.draw("dcar_tpc_vertex:tgl", selection=BASE_SEL, type="profile", bins=50, central="median", fit="pol2", auto_title=True)

def fig26_summary_fit(adf):
    """G5.26 — summary_fit=table — Per-side Gaussian fit params on DCA_r vs tgl"""
    # stats['summary_fit']['table'] is saved as a second PDF page in run_all_pdf.
    return adf.draw("dcar_tpc_vertex:tgl", selection=BASE_SEL, type="profile", bins=50, group_by="side_type", fit="gauss", summary_fit="table", auto_title=True)


# ── G6 — Advanced composition ─────────────────────────────────────────────────

def fig27_vector(adf):
    """G6.27 — vector expression — TPC vs ITS-TPC DCA_r per sector"""
    return adf.draw("[dcar_tpc_vertex, dcar_tpc]:sector", selection=BASE_SEL, type="profile", bins=36, auto_title=True)

def fig28_quantile_band(adf):
    """G6.28 — quantile band — Median +-1sigma DCA_r band per sector"""
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="profile", bins=36, quantiles=[0.16, 0.5, 0.84], auto_title=True)

def fig29_central_median(adf):
    """G6.29 — central=median — Median DCA_r per sector"""
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="profile", bins=36, central="median", auto_title=True)

def fig30_overlay(adf):
    """G6.30 — overlay hist2d+profile — 2D density with mean profile overlaid"""
    # PHASE_13_55_ADF: native adf.draw dispatch (A-1 closed); DFDraw detour removed.
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="hist2d+profile", bins=36, range=((0, 36), (-1.5, 1.5)), auto_title=True)

def fig31_selection_delta_ncl(adf):
    """G6.31 — selection_vector+delta — DCA_r delta high-ncl vs low-ncl tracks"""
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="profile", bins=36, selection_vector=["ncl>100", "ncl<=100"], normalize="delta", auto_title=True)


# ── G8 — ADF dispatch closure (PHASE_13_55_ADF) ───────────────────────────────

def fig35_batch_profile2d(adf):
    """G8.35 — draw_batch profile2d — mean DCA_r per (tgl, sector) via batch"""
    res = adf.draw_batch({"p2d": {"expr": "dcar_tpc_vertex:tgl:sector",
                                  "type": "profile2d", "bins": 36,
                                  "selection": BASE_SEL, "auto_title": True}},
                         verbose=False)
    r = res["p2d"]
    return r["fig"], r["ax"], r["stats"]

def fig36_batch_overlay(adf):
    """G8.36 — draw_batch overlay — hist2d+profile via batch dispatch"""
    res = adf.draw_batch({"ovl": {"expr": "dcar_tpc_vertex:sector",
                                  "type": "hist2d+profile", "bins": 36,
                                  "range": ((0, 36), (-1.5, 1.5)),
                                  "selection": BASE_SEL, "auto_title": True}},
                         verbose=False)
    r = res["ovl"]
    return r["fig"], r["ax"], r["stats"]

def fig37_adf_draw_overlay(adf):
    """G8.37 — adf.draw overlay — A-1 closure: hist2d+profile single-call"""
    return adf.draw("dcar_tpc:sector", selection=BASE_SEL,
                    type="hist2d+profile", bins=36,
                    range=((0, 36), (-1.5, 1.5)), auto_title=True)

def fig38_adf_draw_histo_alias(adf):
    """G8.38 — adf.draw 'histo' alias — A-3 closure: ROOT-convention type alias"""
    return adf.draw("ncl", type="histo", bins=50, selection="ncl>30",
                    auto_title=True)

def fig39_figures_overlay_in_spec(adf):
    """G8.39 — draw_figures overlay-in-spec — A-2 closure: dashboard panel"""
    res = adf.draw_figures([{
        "name": "g8_dashboard",
        "suptitle": "PHASE_13_55_ADF closure — overlay + alias in dashboard",
        "plots": [
            {"expr": "dcar_tpc_vertex:sector", "type": "hist2d+profile",
             "bins": 36, "range": ((0, 36), (-1.5, 1.5)), "selection": BASE_SEL},
            {"expr": "ncl", "type": "histo", "bins": 50, "selection": "ncl>30"},
        ]}], verbose=False)
    r = res["g8_dashboard"]
    return r["fig"], r["axes"], r["stats"]


# ── G9 — Coverage closure (PHASE_13_56_ADF, audit D1 gaps) ───────────────────

def fig40_weights_alias(adf):
    """G9.40 — weights= alias — weighted vs unweighted ncl distribution"""
    adf.add_alias("w_dca", "1.0 + abs(dcar_tpc_vertex)")
    return adf.draw("ncl", type="hist", bins=50, weights="w_dca",
                    selection="ncl>30", auto_title=True)

def fig41_on_error_skip_placeholder(adf):
    """G9.41 — on_error='skip' — deliberate bad spec: visual confirmation of
    the labelled [ERROR] placeholder next to a healthy panel (programmatic
    assertions live in T-G1b/T-G2b/T13-15; this figure is visual-only)."""
    res = adf.draw_figures([{
        "name": "g9_skip_demo",
        "suptitle": "on_error='skip' placeholder demonstration (deliberate)",
        "plots": [
            {"expr": "ncl", "type": "not_a_plot_type"},
            {"expr": "ncl", "type": "hist", "bins": 50, "selection": "ncl>30"},
        ]}], on_error="skip", verbose=False)
    r = res["g9_skip_demo"]
    return r["fig"], r["axes"], r["stats"]

def fig42_entry_window(adf):
    """G9.42 — entry_begin/entry_end — first 200k entries vs full sample"""
    return adf.draw("dcar_tpc_vertex:sector", type="profile", bins=36,
                    selection=BASE_SEL, entry_begin=0, entry_end=200_000,
                    auto_title=True)


# ── G10 — Post-Stage-A semantic-oracle hardening (PHASE_13_77_ADF) ───────────

def fig43_vector_facet_summary_fit(adf):
    """G10.43 — vector×facet×fit×summary_fit — 3 time branches × 2 side facets"""
    q1, q2 = adf.df["time_s"].quantile([1.0 / 3.0, 2.0 / 3.0]).to_numpy()
    return adf.draw(
        "dcar_tpc_vertex:tgl",
        selection=f"{BASE_SEL}&(side_type<2)",
        type="profile",
        bins=30,
        range=(-1.5, 1.5),
        selection_vector=[
            f"time_s<{q1}",
            f"(time_s>={q1})&(time_s<{q2})",
            f"time_s>={q2}",
        ],
        selection_labels=["early", "middle", "late"],
        vector_compose="outer",
        facet_by="side_type",
        fit="pol1",
        summary_fit="table",
        min_entries=50,
        auto_title=True,
    )


# ── G11 — PHASE_13_77 v0.2 hardening oracles (O2 + O4) ─────────────────────

O2_WEIGHT_BRANCHES = [
    "1.0 + 0.0*abs(dcar_tpc_vertex)",
    "1.0 + abs(dcar_tpc_vertex)",
]

def fig44_weights_vector_facet_fit_oracle(adf):
    """G11.44 — weights_vector×facet×fit — 2 weight branches × 2 side facets"""
    return adf.draw(
        "dcar_tpc_vertex:tgl",
        selection=f"{BASE_SEL}&(side_type<2)",
        type="profile",
        bins=30,
        range=(-1.5, 1.5),
        weights_vector=list(O2_WEIGHT_BRANCHES),
        vector_compose="outer",
        facet_by="side_type",
        fit="pol1",
        min_entries=1,
        auto_title=True,
        return_data=True,
    )


# O4 public-surface equivalence oracle.
O4_SUBFRAME_NAME = "O4Surface"
O4_SELECTIONS = [
    f"{O4_SUBFRAME_NAME}.selector==0",
    f"{O4_SUBFRAME_NAME}.selector==1",
]

def _ensure_o4_surface_subframe(adf):
    """Register the qualified-vector fixture over the complete side_type key domain.

    Projection happens before the base selection is applied.  The fixture therefore
    has to cover every side_type key present in the parent ADF, including keys that
    the later ``side_type<2`` selection will discard; otherwise AD-19 correctly
    refuses an integer projection with row-level gaps.
    """
    registry = getattr(getattr(adf, "_subframes", None), "subframes", {}) or {}
    if O4_SUBFRAME_NAME in registry:
        return
    if not hasattr(adf, "df") or "side_type" not in adf.df.columns:
        raise RuntimeError("O4 fixture requires parent column side_type")
    keys = np.asarray(pd.unique(adf.df["side_type"].dropna()))
    if keys.size == 0:
        raise RuntimeError("O4 fixture found no side_type keys")
    keys = np.sort(keys)
    sub = AliasDataFrame(pd.DataFrame({
        "side_type": keys,
        "selector": keys.astype(np.int64, copy=False),
    }))
    adf.register_subframe(O4_SUBFRAME_NAME, sub, index_columns="side_type")

def _o4_public_spec():
    return dict(
        expr="dcar_tpc_vertex:tgl",
        selection=f"{BASE_SEL}&(side_type<2)",
        type="profile",
        bins=30,
        range=(-1.5, 1.5),
        selection_vector=list(O4_SELECTIONS),
        normalize="delta",
        return_data=True,
        auto_title=True,
    )

def _o4_stats(surface, result):
    if surface == "draw":
        return result[2]
    if surface == "draw_batch":
        if result.get("_errors"):
            raise RuntimeError(f"draw_batch errors: {result['_errors']}")
        return result["o4"]["stats"]
    entry = result["o4_surface_equivalence"]
    return entry["stats"][0]

def fig45_public_surface_equivalence_oracle(adf):
    """G11.45 — draw/draw_batch/draw_figures on a B3.3 qualified vector slot"""
    _ensure_o4_surface_subframe(adf)
    spec = _o4_public_spec()
    expr = spec.pop("expr")
    raw = {}
    raw["draw"] = adf.draw(expr, **spec)
    raw["draw_batch"] = adf.draw_batch({"o4": dict(expr=expr, **spec)})
    raw["draw_figures"] = adf.draw_figures([{
        "name": "o4_surface_equivalence",
        "plots": [dict(expr=expr, **spec)],
    }])
    stats = {name: _o4_stats(name, result) for name, result in raw.items()}

    # Close the three product-owned figures before creating the compact evidence page.
    plt.close("all")
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 7.0), sharex=True)
    ref_x = None
    ref_y = None
    for name in ("draw", "draw_batch", "draw_figures"):
        nd = stats[name]["normalize_data"]
        x = np.asarray(nd["x_center"], dtype=float)
        y = np.asarray(nd["value"], dtype=float)
        axes[0].plot(x, y, marker="o", ms=2, label=name)
        if ref_x is None:
            ref_x, ref_y = x, y
        else:
            residual = y - ref_y
            axes[1].plot(x, residual, marker="o", ms=2, label=f"{name} - draw")
    axes[0].set_ylabel("delta profile")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_ylabel("residual")
    axes[1].set_xlabel("tan λ = pz/pT")
    axes[1].legend(fontsize=8)
    fig.suptitle("O4 public-surface equivalence: qualified selection_vector")
    fig.text(
        0.5, 0.93,
        "query: dcar_tpc_vertex:tgl | selection: "
        f"{BASE_SEL}&(side_type<2) | "
        "selection_vector: O4Surface.selector==0 ; O4Surface.selector==1",
        ha="center", va="top", fontsize=7, family="monospace",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return fig, axes, {"surface_stats": stats}


# ── G7 — Full stack ADF + GB (optional, mutate adf in place) ─────────────────

def fig32_subframe_vertex(adf):
    """G7.32 — ADF subframe access — Vertex x-intercept vs time"""
    try:
        calibVertex(adf)
        return adf.draw("CalibVertex.vertex_x_intercept:time_s", type="profile", bins=100, time_format="%H:%M", auto_title=True)
    except Exception as e:
        print(f"  fig32 skipped: {e}"); return None

def fig33_gb_correction_tgl(adf):
    """G7.33 — GB correction residual — DCA_r raw vs predicted delta vs tgl"""
    try:
        calibBiasResolution(adf)
        return adf.draw("[dcar_tpc_vertex, dcar_tpc_vertex_predicted0]:tgl", selection=BASE_SEL, type="profile", bins=50, normalize="delta", auto_title=True)
    except Exception as e:
        print(f"  fig33 skipped: {e}"); return None

def fig34_gb_correction_sector(adf):
    """G7.34 — GB correction residual — DCA_r raw vs predicted delta per sector"""
    try:
        return adf.draw("[dcar_tpc_vertex, dcar_tpc_vertex_predicted0]:sector", selection=BASE_SEL, type="profile", bins=36, normalize="delta", auto_title=True)
    except Exception as e:
        print(f"  fig34 skipped: {e}"); return None


# ── Figure lists ──────────────────────────────────────────────────────────────

FIGURES_G1 = [fig01_hist_ncl, fig02_hist_time, fig03_hist_cumulative,
              fig04_scatter_dca_tgl, fig05_hist2d_dca_sector, fig06_hexbin_dca_sector,
              fig07_profile_dca_sector, fig08_profile2d_dca_tgl_sector, fig09_scatter3d]

FIGURES_G2 = [fig10_profile_groupby_side, fig11_hist_groupby_side,
              fig12_profile_facet_side, fig13_profile_facet_nd, fig14_group_and_facet]

FIGURES_G3 = [fig15_profile_time, fig16_hist2d_time, fig17_profile_facet_time]

FIGURES_G4 = [fig18_delta_sector13, fig19_ratio_time, fig20_pull_time,
              fig21_delta_side, fig22_delta_faceted]

FIGURES_G5 = [fig23_hist_fit, fig24_profile_fit, fig25_profile_fit_median, fig26_summary_fit]

FIGURES_G6 = [fig27_vector, fig28_quantile_band, fig29_central_median,
              fig30_overlay, fig31_selection_delta_ncl]

FIGURES_G8 = [fig35_batch_profile2d, fig36_batch_overlay, fig37_adf_draw_overlay,
              fig38_adf_draw_histo_alias, fig39_figures_overlay_in_spec]

FIGURES_G9 = [fig40_weights_alias, fig41_on_error_skip_placeholder, fig42_entry_window]

FIGURES_G10 = [fig43_vector_facet_summary_fit]

FIGURES_G11 = [fig44_weights_vector_facet_fit_oracle, fig45_public_surface_equivalence_oracle]

FIGURES_MANDATORY = (FIGURES_G1 + FIGURES_G2 + FIGURES_G3 + FIGURES_G4
                     + FIGURES_G5 + FIGURES_G6 + FIGURES_G8 + FIGURES_G9
                     + FIGURES_G10 + FIGURES_G11)
FIGURES_OPTIONAL  = [fig32_subframe_vertex, fig33_gb_correction_tgl, fig34_gb_correction_sector]


# ── Batch PDF / declarative extra-page accounting ─────────────────────────────

# One owner for generated pages beyond each figure's primary page.  Each value
# is an ordered tuple, so 0 / 1 / N extra pages are representable without
# figure-name conditionals in the PDF loop.
FIGURE_EXTRA_PAGE_SPECS = {
    "fig26_summary_fit": (
        {"kind": "summary_fit_table", "title_suffix": " — fit table"},
    ),
    "fig43_vector_facet_summary_fit": (
        {"kind": "summary_fit_table", "title_suffix": " — fit table"},
    ),
}

def declared_extra_page_count(figure_name):
    return len(FIGURE_EXTRA_PAGE_SPECS.get(str(figure_name), ()))

def _summary_fit_table_from_result(result):
    if not isinstance(result, tuple) or len(result) < 3:
        return None
    stats = result[2]
    if isinstance(stats, list):
        for item in stats:
            if isinstance(item, dict) and isinstance(item.get("summary_fit"), dict):
                table = item["summary_fit"].get("table")
                if table is not None:
                    return table
        return None
    if isinstance(stats, dict):
        block = stats.get("summary_fit")
        return block.get("table") if isinstance(block, dict) else None
    return None

def extract_declared_extra_pages(figure_name, result):
    """Return ``[(figure, title_suffix), ...]`` for all declared extra pages.

    A declaration is fail-closed: every declared page must resolve.  Adding a
    new page kind means adding one resolver here, while page cardinality stays
    entirely data-driven by ``FIGURE_EXTRA_PAGE_SPECS``.
    """
    pages = []
    for spec in FIGURE_EXTRA_PAGE_SPECS.get(str(figure_name), ()):
        kind = spec.get("kind")
        if kind == "summary_fit_table":
            page = _summary_fit_table_from_result(result)
        else:
            raise RuntimeError(f"unknown extra-page kind {kind!r} for {figure_name}")
        if page is None:
            raise RuntimeError(
                f"declared extra page {kind!r} missing for {figure_name}")
        pages.append((page, str(spec.get("title_suffix", ""))))
    if len(pages) != declared_extra_page_count(figure_name):
        raise RuntimeError(
            f"extra-page accounting mismatch for {figure_name}: "
            f"declared={declared_extra_page_count(figure_name)} produced={len(pages)}")
    return pages

def run_all_pdf(adf, path="ts_draw_gallery.pdf"):
    """Run all mandatory figures + optional G7, save to one PDF with titled pages."""
    errors = []
    n_pages = 0

    with PdfPages(path) as pdf:
        for fn in FIGURES_MANDATORY:
            name  = fn.__name__
            title = fn.__doc__.splitlines()[0].strip()
            logger.log(f"{name} : BEGIN")
            try:
                result = fn(adf)
                if result is None:
                    continue
                _add(pdf, result[0], title)
                n_pages += 1
                for extra_fig, suffix in extract_declared_extra_pages(name, result):
                    _add(pdf, extra_fig, title + suffix)
                    n_pages += 1
            except Exception as e:
                print(f"  ERROR {name}: {e}")
                errors.append((name, str(e)))
                plt.close("all")
            logger.log(f"{name} : END")

        for fn in FIGURES_OPTIONAL:
            name  = fn.__name__
            title = fn.__doc__.splitlines()[0].strip()
            logger.log(f"{name} : BEGIN")
            try:
                result = fn(adf)
                if result is not None:
                    _add(pdf, result[0], title)
                    n_pages += 1
                    for extra_fig, suffix in extract_declared_extra_pages(name, result):
                        _add(pdf, extra_fig, title + suffix)
                        n_pages += 1
            except Exception as e:
                print(f"  {name} skipped: {e}")
                plt.close("all")
            logger.log(f"{name} : END")

    if errors:
        print(f"\nFAILED {len(errors)} mandatory figure(s):")
        for name, msg in errors:
            print(f"  {name}: {msg}")
        return False

    print(f"\nSaved {n_pages} pages -> {path}")
    return True


# ── Command line ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python time_series_draw.py file.root [output.pdf] [sample_frac]")
        sys.exit(1)
    root_file = sys.argv[1]
    out_pdf   = sys.argv[2] if len(sys.argv) > 2 else "ts_draw_gallery.pdf"
    sample    = float(sys.argv[3]) if len(sys.argv) > 3 else None

    adf = build_adf(root_file, sample)
    ok  = run_all_pdf(adf, out_pdf)
    sys.exit(0 if ok else 1)
