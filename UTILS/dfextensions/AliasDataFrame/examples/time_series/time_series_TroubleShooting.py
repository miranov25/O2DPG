"""
Troublesooting stat examples
from  time_series import *
adf=my_snippet()
"""
from  time_series import *
from  time_series_TroubleShooting import *

def checkEffiecnyADFQA(adf):
    #
    return adf



def checkDCAQA(adf):
    """
    Plats for the DCA bias
    :param adf:
    :return:
    """


    """
    FIGURE1
    
    We need  mult/occupancy as parameter for fitting 
    Sector edge bias vs. occupancy – new dfextension example for time series interactive queries:
I suspected this for a long time; now it is easy to evaluate systematically.
In the time series plot query below, you can see the sector edge bias as a function of track occupancy. This is a preliminary result for the time series:

    """
    fig, ax, stats = adf.draw("dcar_tpc_vertex:dsector",selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)&(abs(qpt)<1)&abs(tgl)<1.2",type="profile",bins=25,group_by="mult",
                              group_by_bins=5,auto_title=True,
                              facet_by="tgl", facet_by_bins=3,  min_entries=20)
    fig.savefig("dcaZFromDeltaTime_tgl.png", dpi=150, bbox_inches='tight')
    """
    FIGUERE2
    Problem at the A side/C side crossing
    Track z validation.
    The large radial distortion above tgl > 0.8 was discussed, but it was not clear to me that the effect is so significant. What I wanted to see is the extent of the effect – the effect is enormous and should be corrected in the calibration procedure, as it defines efficiency loss.
    The vertex z dependence of the dcaZ (and track tgl bias) is particularly striking. It is not surprising that we lose connection to ITS with such a bias.
    """

    fig, ax, stats = adf.draw(
        "dcaZFromDeltaTime:tgl",
        selection="(ncl>50)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)&(abs(qpt)<5)&abs(tgl)<1.4",
        type="profile", bins=50,
        group_by="vertex_z", group_by_quantiles=5,
        facet_by="qpt", facet_by_bins=3,
        auto_title=True, min_entries=20,
        linestyle='none',   # no connecting line
        markersize=10,      # larger markers (default ≈ 4)
    )
    fig.savefig("dcaZFromDeltaTime_tgl.png", dpi=150, bbox_inches='tight')

    """
    Figure 3 — DCA_r bias for tracks crossing the A/C side boundary
    
    A ±0.6 cm bias in DCA_r appears near the central electrode (CE),
    vertex-drift dependent on (vertex_z + tgl) combination — consistent
    with an uncorrected drift-velocity offset at the A/C boundary.
    The intrinsic DCA_r resolution at high pT is ~0.1 cm; the observed
    bias is 6× larger and cannot be ignored.
    
    Tracks crossing the A/C side boundary receive a strong extra kick
    requiring dedicated calibration at the CE boundary.
    """
    fig, ax, stats = adf.draw(
        "dcar_tpc:tgl",
        selection="(ncl>50)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)&(abs(qpt)<3)&abs(tgl)<1.4",
        type="profile", bins=50,
        group_by="vertex_z", group_by_quantiles=5,
        facet_by="qpt", facet_by_bins=3,
        auto_title=True, min_entries=20,
        linestyle='none',   # no connecting line
        markersize=10,      # larger markers (default ≈ 4)
    )
    fig.savefig("dcar_tpc_tgl_vertxz_qpt.png", dpi=150, bbox_inches='tight')

    """
    FIGURE4
    DCA bias  - qpt correction   
    **Imperfect calibration at CE — $\mathrm{DCA}_{r}$ bias for tracks crossing the A/C side boundary**
    A ±0.6 cm bias in $\mathrm{DCA}_{r}$ appears near the central electrode (CE), with a vertex-drift dependence on the ($z_{\mathrm{vtx}}$, $\tan\lambda$) combination — consistent with an uncorrected drift-velocity offset at the A/C boundary. The intrinsic $\mathrm{DCA}_{r}$ resolution at high $p_{T}$ is ~0.1 cm; the observed bias is 6× larger.
    **The calibration must be performed as a function of drift length (as in my calibration procedure),   NOT as a function of $\tan\lambda$ as in the current reconstruction procedure.** Tracks crossing the A/C boundary receive a strong extra kick requiring dedicated calibration at the CE boundary. This is a standard QA plot for the new distortion map creation.
    Note: positive and negative tracks are affected differently due to the $\mathbf{E}\times\mathbf{B}$ effect — a miscalibration in $r$ and in $r\varphi$ can either compensate or enhance the bias depending on track charge sign.
    """
    fig, ax, stats = adf.draw("deltaPar4:dcar_tpc_vertex",selection="( ncl>60 )&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)&(abs(qpt)<2)&abs(tgl)<1.4",type="profile",bins=50,group_by="qpt_ITSTPC",
                              group_by_quantiles=5,facet_by="dsector",facet_by_bins=16,auto_title=True,min_entries=20,linestyle='none',markersize=10,ncols=4,range=(-4,4))
    for a in fig.axes: a.set_ylim(-0.4, 0.4)
    for a in fig.axes: a.set_xlim(-4, 4)
    fig.savefig("dca_bias_qpt_sector_edge.png", dpi=150, bbox_inches='tight')
    fig.set_size_inches(20, 12); fig.tight_layout(); plt.draw()
    #
    """
    Figure 5 Sector edge q/pT bias — correction feasible in past data
    
    TPC intrinsic resolution: ~0.006 1/GeV; TPC+ITS: ~0.0005 1/GeV.
    The sector edge bias is a significant fraction of the TPC standalone resolution
    and must be corrected to exploit the full ITS+TPC momentum resolution.
    
    The q/pT bias at the sector edges is determined by the fraction of track
    length spent near the edge and the local occupancy. It can be characterised
    and corrected using the DCA of TPC-only tracks as the sole calibration
    observable — no separate occupancy correction is required.
    
    Some modelling will be needed to emulate a Kalman filter update, but the
    correction function itself is straightforward. The affected fiducial volume
    is 10–20% of the total. Correcting it would restore the edge bias in past
    data and is both feasible and worthwhile.
    """
    fig, ax, stats = adf.draw("deltaPar4:dcar_tpc_vertex",selection="( ncl>60 )&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)&(abs(qpt)<2)&abs(tgl)<1.4",type="profile",bins=50,group_by="mult",
                              group_by_quantiles=5,facet_by="dsector",facet_by_bins=16,auto_title=True,min_entries=20,linestyle='none',markersize=10,ncols=4,range=(-4,4))
    for a in fig.axes: a.set_ylim(-0.4, 0.4)
    for a in fig.axes: a.set_xlim(-4, 4)
    fig.set_size_inches(20, 12); fig.tight_layout(); plt.draw()
    fig.savefig("dca_bias_qpt_sector_edge_mult.png", dpi=150, bbox_inches='tight')

    return adf

def dcaBiasResol(adf):
    """

    :return:
    """
    """
    Figures 1.a–1.d — TPC DCA_r distribution vs q/pT: sector edge vs sector centre
    
    Quantile plots (10%, 20%, 50%, 80%, 90%) showing the full DCA_r distribution
    shape, not just the mean. Compares two regions:
    
      Fig 1.a — tracks far from sector edge (dsector > 0.75):
        Distribution centred at zero, width dominated by multiple scattering.
        MAD ~ 0.2 cm at high pT — approximately 2× the intrinsic TPC resolution
        (~0.1 cm). The spread is consistent with expected multiple scattering.
    
      Fig 1.b — tracks near sector centre (|dsector - 0.5| < 0.25):
        Reference region away from both edge and boundary.
    
      Fig 1.c — |DCA_r| mean vs q/pT, grouped by dsector (5 bins):
        Shows the absolute radial bias as a function of momentum and
        distance to the sector boundary.
    
      Fig 1.d — |DCA_r| / sqrt(1 + q/pT^2) vs q/pT, grouped by dsector:
        Normalised radial bias — removes the leading momentum dependence
        to isolate the geometric (sector-edge) component.
    """
    fig, ax, stats =adf.draw("dcar_tpc:qpt_ITSTPC",selection="(baseITSTPCCut0)&(abs(qpt_ITSTPC)<5)&((dsector)>0.75)",type="profile",
             quantiles=[0.1,0.2,0.5,0.8,0.9],facet_by="tgl",facet_by_bins=9,auto_title=True,min_entries=50)
    fig.set_size_inches(16, 8); fig.tight_layout(); plt.draw()
    fig.savefig("dcar_tpc_qptITSTPC_quantile_dsectorB75.png", dpi=150, bbox_inches='tight')
    #
    fig, ax, stats =adf.draw("dcar_tpc:qpt_ITSTPC",selection="(baseITSTPCCut0)&(abs(qpt_ITSTPC)<5)&(abs(dsector-0.5)<0.25)",type="profile",
                             quantiles=[0.1,0.2,0.5,0.8,0.9],facet_by="tgl",facet_by_bins=9,auto_title=True,min_entries=50)
    fig.set_size_inches(16, 8); fig.tight_layout(); plt.draw()
    fig.savefig("dcar_tpc_qptITSTPC_quantile_dsector05.png", dpi=150, bbox_inches='tight')
    #
    fig, ax, stats =adf.draw("abs(dcar_tpc):qpt_ITSTPC",selection="(baseITSTPCCut0)&(abs(qpt_ITSTPC)<5)",type="profile",
                             group_by="dsector",group_by_bins=5,auto_title=True,min_entries=50)
    fig.savefig("adcar_tpc_qptITSTPC_dsector.png", dpi=150, bbox_inches='tight')
    #
    fig, ax, stats =adf.draw("abs(dcar_tpc)/sqrt(1+qpt_ITSTPC**2):qpt_ITSTPC",selection="(baseITSTPCCut0)&(abs(qpt_ITSTPC)<5)",type="profile",
                             group_by="dsector",group_by_bins=5,auto_title=True,min_entries=50)
    fig.savefig("adcar_tpcNorm_qptITSTPC_dsector.png", dpi=150, bbox_inches='tight')
    #
    fig, ax, stats =adf.draw("abs(dcar_tpc)/sqrt(1+qpt_ITSTPC**2):qpt_ITSTPC",selection="(baseITSTPCCut0)&(abs(qpt_ITSTPC)<5)&((abs(dsector-0.5)<0.25))",type="profile",
                             group_by="tgl",group_by_bins=5,auto_title=True,min_entries=50)
    fig.savefig("adcar_tpcNorm_qptITSTPC_mult.png", dpi=150, bbox_inches='tight')

    fig, ax, stats =adf.draw("abs(dcar_tpc)/sqrt(1+qpt_ITSTPC**2):qpt_ITSTPC",selection="(baseITSTPCCut0)&(abs(qpt_ITSTPC)<5)&((abs(dsector-0.5)<0.25)&(abs(tgl)<1))",type="profile",
                             group_by="abs(tgl)",group_by_bins=8,auto_title=True,min_entries=50)
    fig.savefig("adcar_tpcNorm_qptITSTPC_tgl.png", dpi=150, bbox_inches='tight')
    #
    """
    Figure 1.e — Normalised |DCA_r| / sqrt(1 + q/pT²) vs q/pT
                 Tracks near sector centre (|dsector-0.5| < 0.35), |tgl| < 1
                 Grouped by tgl (5 bins), faceted by vertex_z (6 quantile panels)
    Observed in plot:
      - Tracks with tgl ≈ 0 (green, -0.20 to 0.20) — crossing the A/C
        central electrode — show  at q/pT ≈ 0
        (normalised DCA_r ~ 0.15 cm) with peaks at |q/pT| ~ 1–2
        (~ 0.30–0.35 cm). This structure is visible in all 6 vertex_z panels,
        confirming it is not vertex-position dependent.
      - All other tgl groups lie in the band 0.20–0.27 cm across the full
        q/pT range — consistent with ~2–3× the intrinsic TPC resolution
        (~0.1 cm), as expected from multiple scattering.
      - No significant vertex_z dependence is observed for tracks not crossing the A/C boundary.
    """
    fig, ax, stats =adf.draw("abs(dcar_tpc)/sqrt(1+qpt_ITSTPC**2):qpt_ITSTPC",selection="(baseITSTPCCut0)&(abs(qpt_ITSTPC)<5)&((abs(dsector-0.5)<0.35)&(abs(tgl)<1))",type="profile",
                             group_by="tgl",group_by_bins=7,auto_title=True,min_entries=20,bins=50,facet_by="vertex_z",facet_by_quantiles=6)
    fig.set_size_inches(16, 8); fig.tight_layout(); plt.draw()
    fig.savefig("adcar_tpcNorm_qptITSTPC_tgl_vertex.png", dpi=150, bbox_inches='tight')

    """
    Figure 2 — DCA_r bias vs q/pT, faceted by tgl, grouped by sector edge distance
    Diagnoses radial DCA bias as a function of momentum and detector geometry:
      1. q/pT splitting at sector edges (group_by=dsector, 5 bins):
         the radial DCA bias changes sign across the sector boundary —
      2. DCA_r scaling with tgl (facet_by=tgl, 9 panels):
         the radial bias grows at large |tgl|, confirming a drift-length
         dependent component. Complements Figure 3 (delta q/pT) —
         the same miscalibration manifests as both a radial displacement
         and a momentum bias.
    """
    fig, ax, stats =adf.draw("dcar_tpc:qpt_ITSTPC",selection="(baseITSTPCCut0)&(abs(qpt_ITSTPC)<5)",type="profile", group_by="dsector",bins=50,
             group_by_bins=5,facet_by="tgl",facet_by_bins=9,auto_title=True,min_entries=20,linestyle='none')
    fig.set_size_inches(20, 10); fig.tight_layout(); plt.draw()
    fig.savefig("dcar_tpc_qptITSTPC_dsector_tgl.png", dpi=150, bbox_inches='tight')
    #
    """
    Figure 3 — Delta q/pT (ITS-TPC outer match) vs q/pT, faceted by tgl, grouped by sector edge distance
    Diagnoses two effects simultaneously:
      1. q/pT splitting at sector edges (group_by=dsector, 5 bins):
         tracks near the sector boundary show a systematic q/pT bias
         — sign and magnitude encode the local ExB + space-charge miscalibration.
      2. q/pT scaling at |tgl| > 0.8 (facet_by=tgl, 9 panels):
         the bias grows toward the A/C side boundary, indicating
         drift-length-dependent calibration error at large pseudorapidity.
    """
    fig, ax, stats =adf.draw("deltaP4OuterITS:qpt_ITSTPC",selection="(baseITSTPCCut0)&(abs(qpt_ITSTPC)<5)&(abs(deltaP4OuterITS)<1)",type="profile", group_by="dsector",bins=50,
                             group_by_bins=5,facet_by="tgl",facet_by_bins=9,auto_title=True,min_entries=10,linestyle='none')
    fig.set_size_inches(16, 8); fig.tight_layout(); plt.draw()
    fig.savefig("deltaP4OuterITS_qptITSTPC_dsector_tgl.png", dpi=150, bbox_inches='tight')


def checkEffiecincyQA(adf):
    """

    :param adf:
    :return:
    """

    """
    FIGURE 1:
    ITS-TPC matching efficiency vs time — A/C side comparison
    
    Track matching efficiency is stable over the selected pp run interval,
    but not necessarily in general. Tracks crossing the A/C side boundary
    have significantly lower matching efficiency. Track matching should be
    performed from the TPC side toward ITS and joined with the other side
    subsequently.
    """

    fig, ax, stats = adf.draw("(hasITSTPC>0):time_s",selection="( ncl>60 )&(abs(dcar_tpc)<5)&(abs(qpt)<5)&(abs(tgl)<1.4)",type="profile",bins=30,group_by="qpt",
                              group_by_quantiles=5,facet_by="side_type",auto_title=True,min_entries=20,linestyle='none',markersize=5,range="minmax")
    fig.savefig("fig/eff_vstime_vsqpt_vsside.png", dpi=150, bbox_inches='tight')

    """
    """

    fig, ax, stats = adf.draw("(hasITSTPC>0):qpt",selection="( ncl>60 )&(abs(dcar_tpc)<5)&(abs(qpt)<2)&(abs(tgl)<1.4)",type="profile",bins=25,group_by="abs(tgl)",
                              group_by_quantiles=4,facet_by="dsector",facet_by_bins=16,auto_title=True,min_entries=20,linestyle='none',markersize=10,ncols=4,range="minmax")
    for a in fig.axes: a.set_ylim(0.7, 1.00)
    for a in fig.axes: a.set_xlim(-3, 3)
    fig.set_size_inches(20, 12); fig.tight_layout(); plt.draw()
    fig.savefig("eff_qpt_tgl_dsector.png", dpi=150, bbox_inches='tight')


def effieciencyQA(adf):
    """

    :param adf:
    :return:
    """
    fig, ax, stats = adf.draw("ncl:sector",selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)",type="profile",bins=100,auto_title=True, group_by="abs(tgl)",
                                group_by_bins=2, quantiles=[0.1,0.5, 0.9], facet_by="time_s",facet_by_bins=9, min_entries=200)
    fig, ax, stats = adf.draw("sector",selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)&(abs(qpt)>0.2)",bins=100,auto_title=True, group_by="abs(tgl)",
                            group_by_bins=3, facet_by="time_s",facet_by_bins=6, min_entries=200,    hist_norm="probability")




def drawCECross(adf):
    # mising clusters at the CE crossing
    fig, ax, stats = adf.draw("ncl:tgl", selection="(abs(qpt)<2.5)&(abs(tgl)<0.5)&(abs(dcar_tpc)<6)", type="profile", min_entries=100, auto_title=True,
             quantiles=[0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9], facet_by="vertex_z", facet_by_quantiles=12, quantile_mode='discrete',bins=100,linewidth=2,ncols=3)
    fig.set_size_inches(16, 10); fig.tight_layout(); plt.draw()
    # chi2match_ITSTPC
    fig, ax, stats = adf.draw("chi2match_ITSTPC:tgl", selection="(abs(qpt)<2.5)&(abs(tgl)<1)&(abs(dcar_tpc)<6)&(abs(vertex_z)<12)", type="profile", min_entries=100, auto_title=True,
                              quantiles=[0.1, 0.5, 0.9], facet_by="vertex_z", facet_by_quantiles=9, quantile_mode='discrete',bins=100,linewidth=2,ncols=3)
    fig.set_size_inches(16, 10); fig.tight_layout(); plt.draw()
    #
    fig, ax, stats = adf.draw("chi2match_ITSTPC:tgl", selection="(abs(qpt)<2.5)&(abs(tgl)<1)&(abs(dcar_tpc)<6)&(abs(vertex_z)<12)", type="profile", min_entries=100, auto_title=True,
                              quantiles=[0.1, 0.5, 0.9], facet_by="vertex_z", facet_by_quantiles=9, quantile_mode='discrete',bins=100,linewidth=2,ncols=3)
    fig.set_size_inches(16, 10); fig.tight_layout(); plt.draw()



def fitDCSITS(adf):
    """
    :return:
    """
    """
    fit DCA bias
    """
    adf.add_alias("phiBin180","(floor(phiITSTPCAtVertex/2/pi*180))",dtype="uint8")
    adf.add_alias("atgl","abs(tgl)",dtype="float16")
    adf.add_alias("aqpt_ITSTPC","abs(qpt_ITSTPC)",dtype="float16")
    adf.materialize_aliases(names=["phiBin180","atgl","aqpt_ITSTPC"])
    gb_columns=["phiBin180"]
    linear_columns=["qpt_ITSTPC","tgl","vertex_z","aqpt_ITSTPC","atgl"]
    fit_columns=["dcar_itstpc"]
    # ---------------------------------------------------------------
    # Pass 1: linear fit of dcar_itstpc vs predictors per phiBin180
    # ---------------------------------------------------------------
    selection="(ncl>50)&(abs(dcar_itstpc)<0.05)&(hasITSTPC>0)&(nClITS>5)"
    adf.ensure_columns(selection,gb_columns+linear_columns+fit_columns)
    selection_mask = adf.df.eval(selection)
    adf.df["wdcar_itstpc"]=(0.5/(0.5+np.abs(adf.df["aqpt_ITSTPC"]))).astype(np.float32)
    _, dfCoeffsP1 = make_parallel_fit_v4(
        df=adf.df,
        gb_columns=gb_columns,
        fit_columns=fit_columns,
        linear_columns=linear_columns,
        fit_intercept=True,
        min_stat=50,
        weights="wdcar_itstpc",
        selection=selection_mask,
        suffix="",
    )
    #
    adfVertex=AliasDataFrame(dfCoeffsP1)
    adf.register_subframe("DCABiasFitP1",adfVertex,index_columns=gb_columns)
    for v in fit_columns:
        adf.add_alias(
            f"{v}_pred",
            f"DCABiasFitP1.{v}_intercept"
            f" + DCABiasFitP1.{v}_slope_qpt_ITSTPC      * qpt_ITSTPC"
            f" + DCABiasFitP1.{v}_slope_tgl             * tgl"
            f" + DCABiasFitP1.{v}_slope_vertex_z        * vertex_z"
            f" + DCABiasFitP1.{v}_slope_aqpt_ITSTPC * aqpt_ITSTPC"
            f" + DCABiasFitP1.{v}_slope_atgl        * atgl",
        )
        adf.add_alias(f"{v}_resid", f"{v} - {v}_pred")

    # add aliase to the ADF

    adf.add_alias("q", "sign(qpt_ITSTPC)",dtype="int8")
    """
    adf.draw("dcar_itstpc:phiITSTPCAtVertex",type="profile",selection="(hasITSTPC>0)&(abs(phi-phiITSTPCAtVertex)<1)&(abs(qpt_ITSTPC)<4)&(ncl>80)&(abs(dcar_itstpc)<0.03)",
             group_by="abs(qpt_ITSTPC)",group_by_bins=5,bins=180,auto_title=True,min_entries=100)
    adf.draw("dcar_itstpc_pred:phiITSTPCAtVertex",type="profile",selection="(hasITSTPC>0)&(abs(phi-phiITSTPCAtVertex)<1)&(abs(qpt_ITSTPC)<4)&(ncl>80)&(abs(dcar_itstpc)<0.03)",
             group_by="abs(qpt_ITSTPC)",group_by_bins=5,bins=180,auto_title=True,min_entries=100)
    adf.draw("dcar_itstpc_resid:phiITSTPCAtVertex",type="profile",selection="(hasITSTPC>0)&(abs(phi-phiITSTPCAtVertex)<1)&(abs(qpt_ITSTPC)<4)&(ncl>80)&(abs(dcar_itstpc)<0.03)",
             group_by="abs(qpt_ITSTPC)",group_by_bins=5,bins=180,auto_title=True,min_entries=100)
    """
    # Plot1:
    sel = "(hasITSTPC>0)&(abs(phi-phiITSTPCAtVertex)<1)&(abs(qpt_ITSTPC)<4)&(ncl>80)&(abs(dcar_itstpc)<0.03)"
    common = dict(type="profile", selection=sel,group_by="abs(qpt_ITSTPC)", group_by_bins=5, bins=180, min_entries=100, auto_title=True)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharex=True)
    adf.draw("dcar_itstpc:phiITSTPCAtVertex",       ax=axes[0], **common)
    adf.draw("dcar_itstpc_pred:phiITSTPCAtVertex",  ax=axes[1], **common)
    adf.draw("dcar_itstpc_resid:phiITSTPCAtVertex", ax=axes[2], **common)
    fig.tight_layout()
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharex=True)
    common = dict(type="profile", selection=sel,group_by="vertex_z", group_by_quantiles=5, bins=180, min_entries=100, auto_title=True)
    adf.draw("abs(dcar_itstpc):qpt_ITSTPC",       ax=axes[0], **common)
    adf.draw("abs(dcar_itstpc_pred):qpt_ITSTPC",  ax=axes[1], **common)
    adf.draw("abs(dcar_itstpc_resid):qpt_ITSTPC", ax=axes[2], **common)
    fig.tight_layout()

    #
    #adf.draw("abs(dcar_itstpc_resid):qpt_ITSTPC",type="profile",selection="(hasITSTPC>0)&(abs(phi-phiITSTPCAtVertex)<1)&(abs(qpt_ITSTPC)<4)&(ncl>80)&(abs(dcar_itstpc)<0.03)",
    #         group_by="mult",group_by_quantiles=5,bins=180,auto_title=True,min_entries=100)
    adf.draw("abs(dcar_itstpc_resid):qpt_ITSTPC",type="profile",selection="(hasITSTPC>0)&(abs(phi-phiITSTPCAtVertex)<1)&(abs(qpt_ITSTPC)<4)&(ncl>80)&(abs(dcar_itstpc)<0.03)",
             group_by="tgl",group_by_quantiles=5,bins=40,auto_title=True,min_entries=100,facet_by="vertex_z",facet_by_quantiles=6)



def drawEdge(adf):
    adf.df["q"]=np.sign(adf.df["qpt_ITSTPC"])
    adf.draw("dcar_itstpc:phiITSTPCAtVertex",type="profile",selection="(hasITSTPC>0)&(abs(phi-phiITSTPCAtVertex)<1)&(abs(qpt_ITSTPC)<4)&(ncl>80)&(abs(dcar_itstpc))<0.03",
             group_by="abs(qpt_ITSTPC)",group_by_bins=5,bins=180,auto_title=True,facet_by="q",min_entries=100)

    # Plot 1 - DCA
    adf.draw("dcar_tpc:dsectorIn",type="profile",selection="(hasITSTPC>0)&(abs(phi-phiITSTPCAtVertex)<1)&(abs(qpt_ITSTPC)<1)&(ncl>80)&(abs(dcar_itstpc)<0.03)",
        group_by="abs(qpt_ITSTPC)",group_by_bins=5,bins=100,auto_title=True,min_entries=100)

    # NCL
    adf.draw("ncl:dsectorIn",type="profile",selection="(hasITSTPC>0)&(abs(phi-phiITSTPCAtVertex)<1)&(abs(qpt_ITSTPC)<1)&(ncl>40)&(abs(dcar_itstpc)<0.03)",
                        group_by="(qpt_ITSTPC)",group_by_bins=5,bins=100,auto_title=True,min_entries=100)


def drawEdgeHis0(adf):
    """
    :param adf:
    :return:
    Example use case of the edge effect - histomgramming part
    Tracks kind and quality selection:
    1.) Track with the TPC, Track with ITS match
    2.) Factorization phsysics is symmetric in  phi - GB(qpt,tlg) for normlization

    Effects included:
      * effiecniency to find the ITS-TPC match (cut dependent e.g DCA)
      * MC/data missmatach
      * double found tracks
      * qpt measurement on the edges is biased, phi posiition on the edge is biased mostly for the TPC only tracks
      *
    """

    # make gb count aggregation
    adf.add_alias("dsector20","dsector*20",dtype="uint8")
    adf.add_alias("tgl20","tgl*20",dtype="int8")
    adf.add_alias("qpt5","qpt*5",dtype="int8")
    adf.add_alias("qptITSTPC5","qpt_ITSTPC*5",dtype="int8")

    adf.materialize_aliases(names=["dsector20","tgl20","qpt5","dsector","qptITSTPC5"])
    isTPC=(adf.df["ncl"]>40)
    isITSTPC=((adf.df["ncl"]>40) & (adf.df["hasITSTPC"]>0))
    # TPC track histo
    dfgbTPCDSec20 = adf.df[isTPC].groupby(["dsector20","tgl20","qpt5"])[["ncl","tgl","qpt","dsector","qpt_ITSTPC"]].agg(["median"]).reset_index()
    dfgbTPCDSec20.columns = ['_'.join(c).strip('_') for c in dfgbTPCDSec20.columns.to_flat_index()]
    dfgbTPCDSec20["count"] = adf.df[isTPC].groupby(["dsector20","tgl20","qpt5"])["ncl"].count().values
    # ITS+TPC trak histo with tpc track param
    dfgbITSTPCDSec20 = adf.df[isITSTPC].groupby(["dsector20","tgl20","qpt5"])[["ncl","tgl","qpt","dsector","qpt_ITSTPC"]].agg(["median"]).reset_index()
    dfgbITSTPCDSec20.columns = ['_'.join(c).strip('_') for c in dfgbITSTPCDSec20.columns.to_flat_index()]
    dfgbITSTPCDSec20["count"] = adf.df[isITSTPC].groupby(["dsector20","tgl20","qpt5"])["ncl"].count().values
    #
    dfgbITSTPCCDSec20 = adf.df[isITSTPC].groupby(["dsector20","tgl20","qptITSTPC5"])[["ncl","tgl","qpt","dsector","qpt_ITSTPC"]].agg(["median"]).reset_index()
    dfgbITSTPCCDSec20.columns = ['_'.join(c).strip('_') for c in dfgbITSTPCCDSec20.columns.to_flat_index()]
    dfgbITSTPCCDSec20["count"] = adf.df[isITSTPC].groupby(["dsector20","tgl20","qptITSTPC5"])["ncl"].count().values
    dfgbITSTPCCDSec20["qpt5"] = dfgbITSTPCCDSec20["qptITSTPC5"].astype("int8")
    #
    dfgbTPC = adf.df[isTPC].groupby(["tgl20","qpt5"])[["ncl","tgl","qpt","dsector"]].agg(["median"]).reset_index()
    dfgbTPC.columns = ['_'.join(c).strip('_') for c in dfgbTPC.columns.to_flat_index()]
    dfgbTPC["count"] = adf.df[isTPC].groupby(["tgl20","qpt5"])["ncl"].count().values
    #
    dfgbITSTPC = adf.df[isITSTPC].groupby(["tgl20","qpt5"])[["ncl","tgl","qpt","dsector","qpt_ITSTPC"]].agg(["median"]).reset_index()
    dfgbITSTPC.columns = ['_'.join(c).strip('_') for c in dfgbITSTPC.columns.to_flat_index()]
    dfgbITSTPC["count"] = adf.df[isITSTPC].groupby(["tgl20","qpt5"])["ncl"].count().values
    # using compbined tracing parameter for the ITSTPC
    dfgbITSTPCC = adf.df[isITSTPC].groupby(["tgl20","qptITSTPC5"])[["ncl","tgl","qpt","dsector","qpt_ITSTPC"]].agg(["median"]).reset_index()
    dfgbITSTPCC.columns = ['_'.join(c).strip('_') for c in dfgbITSTPCC.columns.to_flat_index()]
    dfgbITSTPCC["count"] = adf.df[isITSTPC].groupby(["tgl20","qptITSTPC5"])["ncl"].count().values
    dfgbITSTPCC["qpt5"] = dfgbITSTPCC["qptITSTPC5"].astype("int8")
    #
    #
    adfgbTPCDSec20=AliasDataFrame(dfgbTPCDSec20)
    adfgbTPC=AliasDataFrame(dfgbTPC)
    adfgbITSTPC=AliasDataFrame(dfgbITSTPC)
    adfgbITSTPCC=AliasDataFrame(dfgbITSTPCC)
    adfgbITSTPCDSec20=AliasDataFrame(dfgbITSTPCDSec20)
    adfgbITSTPCCDSec20=AliasDataFrame(dfgbITSTPCCDSec20)

    adfgbTPCDSec20.register_subframe("adfgbITSTPCDSec20",adfgbITSTPCDSec20,index_columns=["dsector20","tgl20","qpt5"])
    adfgbTPCDSec20.register_subframe("adfgbITSTPCCDSec20",adfgbITSTPCCDSec20,index_columns=["dsector20","tgl20","qpt5"])
    adfgbTPCDSec20.register_subframe("adfgbTPC",adfgbTPC,index_columns=["tgl20","qpt5"])
    adfgbTPCDSec20.register_subframe("adfgbITSTPC",adfgbITSTPC,index_columns=["tgl20","qpt5"])
    adfgbTPCDSec20.register_subframe("adfgbITSTPCC",adfgbITSTPCC,index_columns=["tgl20","qpt5"])
    adf.register_subframe("adfgbITSTPCC",adfgbITSTPCC,index_columns=["tgl20","qpt5"])
    adfgbTPCDSec20.draw_lazy=True
    # example queries
    adfgbTPCDSec20.draw("count:adfgbITSTPCDSec20.count")
    adfgbTPCDSec20.draw("adfgbITSTPCDSec20.count:count",type="profile",group_by="qpt_median",selection="abs(tgl20)<10",group_by_bins=5,auto_title=True)
    adfgbTPCDSec20.draw("adfgbITSTPCDSec20.count/count:dsector_median",type="profile",group_by="qpt_median",selection="(abs(tgl_median)<1) & (count>20) &(abs(qpt_median)<1)",group_by_bins=10,auto_title=True)
    #
    adfgbTPCDSec20.draw("adfgbTPC.count/(count*20):dsector_median",type="profile",group_by="qpt_median",selection="(abs(tgl_median)<1) & (count>20) &(abs(qpt_median)<1)",group_by_bins=10,auto_title=True)
    #
    adfgbTPCDSec20.draw("(adfgbITSTPCDSec20.count/count)/(adfgbITSTPC.count/adfgbTPC.count):dsector_median",type="profile",group_by="qpt_median",selection="(abs(tgl_median)<1) & (count>20) &(abs(qpt_median)<1)",group_by_bins=10,auto_title=True)
    #
    adfgbTPCDSec20.draw("(20*adfgbITSTPCDSec20.count/adfgbITSTPC.count):dsector_median",type="profile",group_by="qpt_median",selection="(abs(tgl_median)<1) & (count>20) &(abs(qpt_median)<2)",group_by_bins=10,auto_title=True)
    #
    #
    adfgbTPCDSec20.draw("(20*adfgbITSTPCCDSec20.count/adfgbITSTPCC.count):dsector_median",type="profile",group_by="qpt_median",selection="(abs(tgl_median)<1) & (count>20) &(abs(qpt_median)<2)",group_by_bins=10,auto_title=True)
    #






def drawEdgeHis(adf):
    """Edge-effect QA — the histogramming (count-based) part.

    Builds count aggregations of TPC tracks on two grids and uses them to
    monitor the ITS-TPC matching efficiency and its bias near the sector edge.

    Track samples compared:
      * isTPC    = ncl > 40              — TPC tracks
      * isITSTPC = isTPC & hasITSTPC     — ITS-matched TPC tracks

    Normalization assumption:
      The physics is taken to factorize and be symmetric in phi, so the
      aggregation integrated over the sector coordinate ``dsector`` is the
      phi-symmetric reference. Two grids are therefore built per sample:
        * fine grid (dsector20, tgl20, qpt5) — resolves the edge in dsector
        * norm grid (tgl20, qpt5)            — integrated over dsector
      A second binning uses the combined ITS-TPC curvature ``qpt_ITSTPC5``
      instead of the TPC-only ``qpt5`` (only defined for matched tracks).

    Effects this is meant to expose:
      * ITS-TPC matching efficiency and its cut dependence (e.g. on DCA)
      * MC/data mismatch
      * double-found (duplicate) tracks
      * edge bias: qpt is biased near the edge; the phi position is biased on
        the edge mostly for TPC-only tracks

    :param adf: AliasDataFrame with per-track columns
                (ncl, hasITSTPC, dsector, tgl, qpt, qpt_ITSTPC).
    :return: None — registers the count subframes and issues the example draws.
    """
    def _gb_agg(df, keys, value_cols, agg=("median",)):
        """Group `df` on `keys`; per cell: `agg` of `value_cols` (suffixed) + row count.
        Returns an AliasDataFrame with one row per grid cell: the grid keys as
        plain columns, each value column aggregated and flat-named ``<col>_<agg>``,
        plus ``count`` (rows per cell). This is the 'simple-stats' StatResult
        producer used by the edge-effect example.
        """
        g = df.groupby(list(keys))
        out = g[value_cols].agg(list(agg))
        out.columns = [f"{c}_{a}" for c, a in out.columns]   # ('ncl','median') -> 'ncl_median'
        out["count"] = g.size()                              # rows per cell
        return AliasDataFrame(out.reset_index())
    def _median_over(adf_fine, keep_keys, col="count"):
        """Median of `col` over the dropped grid axes, at fixed `keep_keys`. Edge-robust norm."""
        out = adf_fine.df.groupby(keep_keys, as_index=False)[col].median()
        return AliasDataFrame(out)
    # --- 1. integer grid-coordinate aliases ----------------------------------
    adf.add_alias("dsector20",   "dsector*20",   dtype="uint8")
    adf.add_alias("tgl20",       "tgl*20",       dtype="int8")
    adf.add_alias("qpt5",        "qpt*5",        dtype="int8")
    adf.add_alias("qpt_ITSTPC5", "qpt_ITSTPC*5", dtype="int8")
    adf.materialize_aliases(names=["dsector20", "tgl20", "qpt5", "qpt_ITSTPC5", "dsector","ncl"])
    adf.ensure_columns("ncl>40&hasITSTPC>0")
    # --- 2. the two track samples --------------------------------------------
    isTPC    = adf.df["ncl"] > 40
    isITSTPC = isTPC & (adf.df["hasITSTPC"] > 0)
    COLS_TPC    = ["ncl", "tgl", "qpt", "dsector"]
    COLS_ITSTPC = COLS_TPC + ["qpt_ITSTPC"]

    # --- 3. count tables: {sample} x {grid} x {qpt binning} ------------------
    #   sample : TPC | ITSTPC          grid : fine (with dsector) | norm (integrated)
    #   qpt    : qpt5 (TPC curvature)  |  qpt_ITSTPC5 (combined ITS-TPC curvature)
    specs = {
        #  name           mask       grid keys                              value cols
        "TPC_fine":     (isTPC,    ["dsector20", "tgl20", "qpt5"],        COLS_TPC),
        "ITSTPC_fine":  (isITSTPC, ["dsector20", "tgl20", "qpt5"],        COLS_ITSTPC),
        "TPC_norm":     (isTPC,    ["tgl20", "qpt5"],                     COLS_TPC),
        "ITSTPC_norm":  (isITSTPC, ["tgl20", "qpt5"],                     COLS_ITSTPC),
        # combined ITS-TPC curvature binning (staged for the qpt_ITSTPC draws)
        "ITSTPCc_fine": (isITSTPC, ["dsector20", "tgl20", "qpt_ITSTPC5"], COLS_ITSTPC),
        "ITSTPCc_norm": (isITSTPC, ["tgl20", "qpt_ITSTPC5"],             COLS_ITSTPC),
    }
    logger.log("drawEdgeHis: Step3 BEGIN")
    gb = {name: _gb_agg(adf.df[mask], keys, cols) for name, (mask, keys, cols) in specs.items()}
    logger.log("drawEdgeHis: step3 END")
    # --- 4. register the comparison tables onto the fine TPC grid ------------
    # unqualified `count` in the draws = base (TPC_fine) count;
    # `<name>.count` reaches the registered subframe, aligned on its index cols.
    logger.log("drawEdgeHis: step4 BEGIN")
    base = gb["TPC_fine"]
    registrations = {
        #  subframe name   table              index (alignment) columns
        "ITSTPC_fine": (gb["ITSTPC_fine"], ["dsector20", "tgl20", "qpt5"]),
        "TPC_norm":    (gb["TPC_norm"],    ["tgl20", "qpt5"]),   # broadcast over dsector20
        "ITSTPC_norm": (gb["ITSTPC_norm"], ["tgl20", "qpt5"]),   # broadcast over dsector20
    }
    for name, (table, index_cols) in registrations.items():
        base.register_subframe(name, table, index_columns=index_cols)
    base.draw_lazy = True
    logger.log("drawEdgeHis: step4 END")
    #
    # median-over-dsector reference (built from *_fine, not a new raw GB pass)
    keep = ["tgl20", "qpt5"]
    base.register_subframe("TPC_normmed",    _median_over(gb["TPC_fine"],    keep), index_columns=keep)
    base.register_subframe("ITSTPC_normmed", _median_over(gb["ITSTPC_fine"], keep), index_columns=keep)

    # --- 5. example draws -----------------------------------------------------
    sel = "(abs(tgl_median)<1) & (count>20) & (abs(qpt_median)<1)"
    # (a) raw count comparison, TPC vs ITS-TPC (one point per fine cell)
    base.draw("count:ITSTPC_fine.count")
    base.draw("ITSTPC_fine.count:count", type="profile", group_by="qpt_median", group_by_bins=5, selection="abs(tgl20)<10", auto_title=True)
    # (b) matching efficiency = ITSTPC / TPC, vs distance to the sector edge
    base.draw("ITSTPC_fine.count/count:dsector_median", type="profile", group_by="qpt_median", group_by_bins=10, selection=sel, auto_title=True)
    # (c) uniformity check: norm (dsector-integrated) count vs fine count,
    #     scaled by the dsector bin factor (=20); ~flat where rate is edge-independent
    base.draw("TPC_norm.count/(count*20):dsector_median", type="profile", group_by="qpt_median", group_by_bins=10, selection=sel, auto_title=True)
    # (d) DOUBLE RATIO: edge-resolved efficiency / phi-symmetric reference efficiency
    base.draw("(ITSTPC_fine.count/count)/(ITSTPC_norm.count/TPC_norm.count):dsector_median",type="profile", group_by="qpt_median", group_by_bins=10,
              selection=sel, auto_title=True)
    # (e) ITS-TPC fine/norm count ratio (x20 dsector factor), points only
    base.draw("(20*ITSTPC_fine.count/ITSTPC_norm.count):dsector_median", type="profile",group_by="qpt_median", group_by_bins=10,
              selection="(abs(tgl_median)<1) & (count>20) & (abs(qpt_median)<2)",auto_title=True)
    # (f)  ITS-TPC fine/normed to median withoing sector
    base.draw("(ITSTPC_fine.count/ITSTPC_normmed.count):dsector_median", type="profile",group_by="qpt_median", group_by_bins=10,
              selection="(abs(tgl_median)<1) & (count>20) & (abs(qpt_median)<2)",auto_title=True)
    adf.register_subframe("counters", base, index_columns=["dsector20", "tgl20", "qpt5"]) ## should we register here or as user request?
    return base


def makeGBTPCDiff(adf, cols, group_byLocal, group_byGlobal, selection=None, out_dir=".", tag="gbtpcdiff", time_col="timeMS"):
    """Local + global GB statistics of `cols`, exported for time-series monitoring.
    1. LOCAL  — group by `group_byLocal` (fine grid); mean/median/std (+count)
                per column. Registered on the main frame as subframe 'gbLocal'.
    2. GLOBAL — subtract the per-local-cell mean from each track, then group the
                residuals by `group_byGlobal` (sector grid); mean/median/std
                (+count) of the residuals. Registered as 'gbGlobal'.
    3. EXPORT — both frames get a `timestamp` column and a timestamped filename,
                so snapshots can be merged into a time series later.

    adf=loadADFLazy()
    cols           = ["dcar_itstpc", "dcar_tpc","dcar_tpc_vertex", "chi2match_ITSTPC","dcaZFromDeltaTime"]
    for icol in range(5): cols.append(f"deltaP{icol}OuterITS"); cols.append(f"deltaPar{icol}")

    group_byLocal  = ["dsector20", "tgl20", "qpt_ITSTPC5"]   # fine grid for local mean subtraction
    group_byGlobal = ["sector180", "tgl10", "qpt_ITSTPC5"]
    selection      = "(ncl>40)&(hasITSTPC>0)&(abs(dcar_itstpc)<0.03)&(isOKITSTPC)"
    time_col="timeMS"
    plots to export: for the local test before merging time series:
    PDF report?
    Candidate figures to code -very dense beut usfull
    colunss : dcar_tpc_vertx,dcaz_tpc, dP0,dP1,dP2,dP3,dP4
    reows:
    makeGBTPCDiff(adf, cols, group_byLocal, group_byGlobal, selection=selection, out_dir=".", tag="gbtpcdiff", time_col="timeMS")
    """
    def _gb_stats(frame, keys, value_cols):
        """mean/median/std of value_cols on grid `keys` + row count; flat-named."""
        g = frame.groupby(keys)
        out = g[value_cols].agg(["mean", "median", "std"])
        out.columns = [f"{c}_{a}" for c, a in out.columns]   # ('dcar','mean') -> 'dcar_mean'
        out["count"] = g.size()
        out = out.reset_index()
        out["timestamp"] = ts
        return out
    def makeFigLocal4(var="dcar_tpc_vertex", out="fig/gb4_{var}", group_by="tgl", n=7, range=(-4.5, 4.5)):
        """
        makeFigLocal4("dcar_tpc_vertex",range=(-4.5,4.5),out="fig/gb4_{var}45")
        makeFigLocal4("dcar_tpc_vertex",range=(-1,1),out="fig/gb4_{var}10")
        makeFigLocal4("deltaP2OuterITS",range=(-1,1),out="fig/gb4_{var}10")

        """
        sel_raw   = f"(isOKITSTPC>0)&(abs({var})<10)&(abs(dsector-0.5)<0.45)&(ncl>80)"
        sel_local = "(count>10)&(abs(dsector-0.5)<0.45)"
        fig, ax = plt.subplots(2, 2, figsize=(12, 9))
        adf.draw(f"{var}:qpt_ITSTPC",            type="profile", selection=sel_raw,
                 group_by=group_by, group_by_bins=n, auto_title=True, range=range, ax=ax[0,0])
        adf.draw(f"abs({var}):qpt_ITSTPC",       type="profile", selection=sel_raw,
                 group_by=group_by, group_by_bins=n, auto_title=True, range=range, ax=ax[0,1])
        adfLocal.draw(f"{var}_std:qpt",          type="profile", selection=sel_local,
                      group_by=group_by, group_by_bins=n, auto_title=True, ax=ax[1,0],range=range)
        adfLocal.draw(f"abs({var}_mean):qpt",    type="profile", selection=sel_local,
                      group_by=group_by, group_by_bins=n, auto_title=True, ax=ax[1,1],range=range)
        fig.suptitle(var)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        stem = out.format(var=var)
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(f"{stem}.png", dpi=150, bbox_inches="tight")
        #plt.close(fig)
        return f"{stem}.pdf", f"{stem}.png"

# --- grid-coordinate aliases ---------------------------------------------
    adf.add_alias("dsector20", "dsector*20",   dtype="uint8")
    adf.add_alias("sector180", "180*(phi/pi)", dtype="int16")   # NOTE: was uint8 — phi<0 overflows
    adf.add_alias("tgl20",     "tgl*20",       dtype="int8")
    adf.add_alias("tgl10",     "tgl*10",       dtype="int8")
    adf.add_alias("qpt5",      "qpt*5",        dtype="int8")
    adf.add_alias("qpt_ITSTPC5", "qpt_ITSTPC*5", dtype="int8")
    adf.add_alias("isPrimITS01", "(abs(dcar_itstpc)<0.1)&(abs(dcaz_itstpc)<0.1)", dtype="int8")
    adf.add_alias("isOKITSTPC", "(hasITSTPC>0)&(abs(deltaP0OuterITS)<5)&(abs(deltaP2OuterITS)<1) & (abs(dcar_tpc_vertex)<10) & (ncl>80)", dtype="int8")
    # make pulls alaises for the deltaP columns
    #
    logger.log(f"makeGBTPCDiff: Load brances Step0 BEGIN")
    grid_cols = sorted(set(group_byLocal) | set(group_byGlobal))
    adf.ensure_columns(selection, cols + grid_cols,time_col)
    adf.materialize_aliases(names=grid_cols)
    logger.log(f"makeGBTPCDiff: Load brances Step0 END")

    # --- selection + snapshot timestamp --------------------------------------
    mask = adf.eval(selection) if selection else slice(None)
    df = adf.df[mask]
    ts=adf.df["timeMS"].median()


    # --- 1. LOCAL ------------------------------------------------------------
    logger.log(f"makeGBTPCDiff: LOCAL GB Step1: BEGIN")
    local = _gb_stats(df, group_byLocal, cols)
    local["timeMS"] = ts
    adfLocal=AliasDataFrame(local)
    adfLocal.draw_lazy=True
    adfLocal.add_alias("qpt", "qpt_ITSTPC5/5", dtype="float32")
    adfLocal.add_alias("tgl", "tgl20/20", dtype="float32")
    adfLocal.add_alias("dsector", "dsector20/20", dtype="float32")
    adf.register_subframe("gbLocal", adfLocal, index_columns=group_byLocal)
    logger.log(f"makeGBTPCDiff: LOCAL GB END")
    # 1.b)
    logger.log(f"makeGBTPCDiff: LOCAL GB ALIASES Step1bBEGIN")
    cols_DL=[f"{c}_DL" for c in cols]
    for c in cols: adf.add_alias(f"{c}_DL", f"{c}-gbLocal.{c}_mean", dtype="float32")
    adf.materialize_aliases(names=cols_DL)
    """
    adf.draw("dcar_tpc_vertex:qpt_ITSTPC",type="profile",selection="(isOKITSTPC>0)&(abs(dcar_tpc_vertex)<10)&(abs(dsector-0.5)<0.45)",group_by="tgl",group_by_bins=7,auto_title=True,range=(-4.5,4.5))
    adf.draw("abs(dcar_tpc_vertex):qpt_ITSTPC",type="profile",selection="(isOKITSTPC>0)&(abs(dcar_tpc_vertex)<10)&(abs(dsector-0.5)<0.45)",group_by="tgl",group_by_bins=7,auto_title=True,range=(-4.5,4.5))
    #   
    adfLocal.draw("dcar_tpc_vertex_std:qpt",type="profile",selection="(count>10)&(abs(dsector-0.5)<0.45)",group_by="tgl",group_by_bins=7,auto_title=True)
    adfLocal.draw("abs(dcar_tpc_vertex_mean):qpt",type="profile",selection="(count>10)&(abs(dsector-0.5)<0.45)",group_by="tgl",group_by_bins=7,auto_title=True)

    
    adf.draw("dcar_tpc_vertex:gbLocal.dcar_tpc_mean",type="profile",selection="((isOKITSTPC>0)&(abs(dsector-0.5)<0.45)&(isPrimITS01)",group_by="tgl",group_by_bins=7,auto_title=True)
    # 

    fig, axes, stats = adf.draw("1.44*abs(dcar_tpc_DL):qpt_ITSTPC",type="profile",selection="(hasITSTPC>0)&(isPrimITS01)&(abs(tgl)<1.2)",
        group_by="dsector",group_by_bins=10,auto_title=True,range=(-4,4),facet_by="mult",facet_by_quantiles=4,ncols=2,legend="shared")
    
    fig, axes, stats = adf.draw("1.44*abs(dcar_tpc_DL):mult",type="profile",selection="(hasITSTPC>0)&(isPrimITS01)&(abs(tgl)<1.2)&(abs(qpt_ITSTPC)<1)&abs(dsector-0.5)<0.45",
        group_by="qpt_ITSTPC",group_by_bins=9,auto_title=True,facet_by="dsector",facet_by_quantiles=9,ncols=3,legend="shared")    
        
    fig, axes, stats = adf.draw("1.44*abs(dcar_tpc_DL):mult",type="profile",selection="(hasITSTPC>0)&(isPrimITS01)&(abs(tgl)<1.2)&(abs(qpt_ITSTPC)<1)&abs(dsector-0.5)<0.45",
        group_by="qpt_ITSTPC",group_by_bins=9,auto_title=True,facet_by="tgl",facet_by_quantiles=6,ncols=3,legend="shared",bins=20)
        
        
      
    """
    logger.log(f"makeGBTPCDiff: LOCAL GB ALIASES Step1bEND")
    # --- 2. GLOBAL on local-mean-subtracted residuals ------------------------
    logger.log("makeGBTPCDiff: GLOBAL GB Step2 BEGIN")
    glob = _gb_stats(adf.df[mask], group_byGlobal, cols_DL)   # residuals already materialized
    glob["timeMS"] = ts
    adfGlobal=AliasDataFrame(glob)
    adf.register_subframe("gbGlobal", adfGlobal, index_columns=group_byGlobal)
    logger.log("makeGBTPCDiff: GLOBAL GB END")
    # --- 3. EXPORT both, timestamped -----------------------------------------
    logger.log("makeGBTPCDiff: EXPORT Step3 BEGIN")
    adfLocal.export_tree(f"{out_dir}/timeSeries_GBLocal.root")
    adfGlobal.export_tree(f"{out_dir}/timeSeries_GBGlobal.root")
    logger.log("makeGBTPCDiff: EXPORT Step3 END")

    # --- 4. EXAMPLE DRAW -----------------------------------------------------"""
    # make pdf report file?
    """
    adf.draw("(gbLocal.dcar_tpc_mean):qpt_ITSTPC",type="profile",selection="(hasITSTPC>0)&(abs(dsector-0.5)<0.45)&(isPrimITS01)",group_by="tgl",group_by_bins=7,auto_title=True,range=(-4,4))
    adf.draw("(gbLocal.dcar_tpc_mean):dsector",type="profile",selection="(hasITSTPC>0)&(abs(dsector-0.5)<0.45)&(isPrimITS01)",group_by="qpt",group_by_bins=11,auto_title=True,range=(0,1))
    adf.draw("(dcar_tpc):dsector",type="profile",selection="(hasITSTPC>0)&(abs(dsector-0.5)<0.45)&(isPrimITS01)",group_by="qpt",group_by_bins=15,auto_title=True,range=(0,1))
    adf.draw("1.44*abs(gbLocal.dcar_tpc_mean):qpt_ITSTPC",type="profile",selection="(hasITSTPC>0)&(abs(dsector-0.5)<0.45)&(isPrimITS01)",group_by="tgl",group_by_bins=7,auto_title=True,range=(-4,4))
    
    adf.draw("1.44*abs(gbGlobal.dcar_tpc_DL_mean):qpt_ITSTPC",type="profile",selection="(hasITSTPC>0)&(abs(dsector-0.5)<0.45)&(isPrimITS01)",group_by="tgl",group_by_bins=7,auto_title=True,range=(-4,4))
    adf.draw(1.44*"abs(gbGlobal.dcar_tpc_DL_mean):qpt_ITSTPC",type="profile",selection="(hasITSTPC>0)&(abs(dsector-0.5)<0.45)&(isPrimITS01)",group_by="tgl",group_by_bins=7,auto_title=True,range=(-4,4))
    #adf.draw("dcar_tpc:sector",type="profile",selection="(hasITSTPC>0)&(abs(dsector-0.5)<0.45)&(isPrimITS01)",group_by="qpt",group_by_bins=11,auto_title=True)
    adf.draw("dcar_tpc:sector",type="profile",selection="(hasITSTPC>0)&(abs(dsector-0.5)<0.45)&(isPrimITS01)",group_by="qpt",group_by_bins=11,bins=180,auto_title=True)
    #
    fig, axes, stats = adf.draw("1.44*abs(dcar_tpc_DL):mult",type="profile",selection="(hasITSTPC>0)&(isPrimITS01)&(abs(tgl)<1.2)&(abs(qpt_ITSTPC)<1)&abs(dsector-0.5)<0.45",
        group_by="qpt_ITSTPC",group_by_bins=9,auto_title=True,facet_by="dsector",facet_by_quantiles=9,ncols=3,legend="shared")    
        
    fig, axes, stats = adf.draw("1.44*abs(dcar_tpc_DL):mult",type="profile",selection="(hasITSTPC>0)&(isPrimITS01)&(abs(tgl)<1.2)&(abs(qpt_ITSTPC)<1)&abs(dsector-0.5)<0.45",
        group_by="qpt_ITSTPC",group_by_bins=9,auto_title=True,facet_by="tgl",facet_by_quantiles=6,ncols=3,legend="shared",bins=20)
    fig, axes, stats = adf.draw("1.44*abs(dcar_tpc_DL):mult",type="profile",selection="(hasITSTPC>0)&(isPrimITS01)&(abs(tgl)<1.2)&(abs(qpt_ITSTPC)<0.5)&abs(dsector-0.5)<0.40",
        group_by="qpt_ITSTPC",group_by_bins=6,auto_title=True,facet_by="tgl",facet_by_quantiles=9,ncols=3,legend="shared",bins=20)    
    
    """
    return adf

