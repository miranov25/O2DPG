"""
time_series_draw_110.py — PHASE_13_110 pages: combined dfdraw features, one public call per page

Each page is one ordinary `adf.draw(...)` call, written like the pages of time_series_draw.py, and it is the
subject of one oracle case in time_series_draw_invariancetest.py. That case computes, from the raw ROOT columns
and with plain NumPy/pandas, what the plot must show, and compares it with what dfdraw returned and drew.

The pages are drawn only by the oracle run (new_13_105.pdf); the 59-page gallery of time_series_draw.py is
unchanged. Every profile page passes return_data=True, so that the case can read the numbers behind the plot;
hist accepts no return_data, so its cases read the drawn bars.

Families (PHASE_13_110 proposal, S0 v1.0 §7 and v1.1 §3):
  P110-01  vectors x groups x facets in one call (profile and histogram)
"""

from time_series_draw import BASE_SEL, ITS_SEL

AC_SEL = f"{BASE_SEL}&(side_type<2)"            # A side (0) and C side (1) only
W_DCA = "1.0 + abs(dcar_tpc_vertex)"             # the gallery's weight alias w_dca (fig40)
W_FLAT = "1.0 + 0.0*abs(dcar_tpc_vertex)"        # a weight of exactly 1 written as an expression (fig44)


def early_late(adf):
    """The gallery's early/late split: time_s below / at or above the median over all rows (fig19)."""
    t_mid = adf.df["time_s"].median()
    return [f"time_s<{t_mid}", f"time_s>={t_mid}"]


# ── P110-01 — vectors x groups x facets ──────────────────────────────────────

def p110_01a_group_facet_min_entries(adf):
    """P110-01.a — group_by+facet_by+min_entries — ITS clusters vs time, per side and sector quarter (fig14)"""
    return adf.draw("nClITS:time_s", selection=ITS_SEL, type="profile", bins=100, group_by="side_type",
                    facet_by="sector", facet_by_bins=4, min_entries=20, auto_title=True, return_data=True)


def p110_01b_two_level_facets(adf):
    """P110-01.b — two-level numeric facets — DCA_r per sector in a side_type x qpt_bin10 grid (fig13)"""
    return adf.draw("dcar_tpc_vertex:sector", selection=BASE_SEL, type="profile", bins=36,
                    facet_by=["side_type", "qpt_bin10"], facet_by_bins=[2, 3], auto_title=True, return_data=True)


def p110_01c_profile_group(adf):
    """P110-01.c — profile+group_by — mean DCA_r per sector, A vs C side (fig10)"""
    return adf.draw("dcar_tpc_vertex:sector", selection=AC_SEL, type="profile", bins=36, group_by="side_type",
                    auto_title=True, return_data=True)


def p110_01d_profile_facet(adf):
    """P110-01.d — profile+facet_by — mean DCA_r per sector, one panel per side (fig12)"""
    return adf.draw("dcar_tpc_vertex:sector", selection=AC_SEL, type="profile", bins=36, facet_by="side_type",
                    auto_title=True, return_data=True)


def p110_01e_selection_vector_group(adf):
    """P110-01.e — selection_vector x group_by — mean DCA_r per sector, early and late run, A vs C side"""
    return adf.draw("dcar_tpc_vertex:sector", selection=AC_SEL, type="profile", bins=36, range=(-0.5, 35.5),
                    selection_vector=early_late(adf), selection_labels=["early", "late"], group_by="side_type",
                    auto_title=False, return_data=True)


def p110_01f_weights_vector_facet_hist(adf):
    """P110-01.f — hist weights_vector x facet_by — cluster count, weighted by 1+|DCA_r| and flat, per side"""
    return adf.draw("ncl", type="hist", bins=40, range=(60, 160), selection=AC_SEL,
                    weights_vector=[W_DCA, W_FLAT], weights_labels=["w_dca", "flat"], facet_by="side_type",
                    auto_title=False)


def p110_01g_hist_group(adf):
    """P110-01.g — hist+group_by — TPC cluster count, A vs C side (fig11)"""
    return adf.draw("ncl", type="hist", bins=40, selection="(ncl>30)&(side_type<2)", group_by="side_type",
                    auto_title=True)


def p110_01h_hist_weights(adf):
    """P110-01.h — hist weights= alias — cluster count weighted by w_dca = 1+|DCA_r| (fig40)"""
    adf.add_alias("w_dca", W_DCA)
    adf.materialize_aliases(names=["w_dca"])     # the oracle compares the alias column with its formula
    return adf.draw("ncl", type="hist", bins=50, weights="w_dca", selection="ncl>30", auto_title=True)


def p110_01i_selection_group_facet(adf):
    """P110-01.i — selection_vector x group_by x facet_by in ONE call — mean DCA_r per sector, early and late,
    A vs C side, one panel per hasITSTPC value"""
    return adf.draw("dcar_tpc_vertex:sector", selection=AC_SEL, type="profile", bins=36, range=(-0.5, 35.5),
                    selection_vector=early_late(adf), selection_labels=["early", "late"], group_by="side_type",
                    facet_by="hasITSTPC", facet_labels="outer", auto_title=False, return_data=True)


PAGES_110_01 = (p110_01a_group_facet_min_entries, p110_01b_two_level_facets, p110_01c_profile_group,
                p110_01d_profile_facet, p110_01e_selection_vector_group, p110_01f_weights_vector_facet_hist,
                p110_01g_hist_group, p110_01h_hist_weights, p110_01i_selection_group_facet)
