"""

Time-series TPC track QA example for AliasDataFrame.
Tutorial macro for ALICE TPC/ITS QA workflows. Each plot is a self-contained
function with the full ``adf.draw`` call spelled out — edit the function body
to change parameters, then re-run.

Setup (once per shell session):
    $ cd examples/time_series
    $ source setup_env.sh         # adds AliasDataFrame + dfdraw to PYTHONPATH

Interactive use (IPython):
import sys,os; sys.path.insert(1, os.environ[f"O2DPG"]+"/UTILS/dfextensions/AliasDataFrame/examples/time_series");
from time_series import *
adf = root_to_adf("time_series_tracks_0.root")
    adf = root_to_adf("time_series_tracks_0.root")
    adf.draw_lazy=True
    apply_meta(adf,df_TimeSeriesAliases)
    apply_meta(adf,df_TimeSeriesMeta)

Data file convention:
    By default reads ``time_series_tracks_0.root`` from this script's directory.
    Symlink your data file:
        $ ln -s /path/to/your/file.root time_series_tracks_0.root
Requires: AliasDataFrame, uproot, pandas, matplotlib.
"""
import re
#from __future__ import annotations
import os
import sys
import pandas as pd
import uproot
import numpy as np
from matplotlib.ticker import FuncFormatter
from datetime import datetime
import awkward as ak


from dfextensions.AliasDataFrame import AliasDataFrame
from dfextensions.groupby_regression.groupby_regression_sliding_window import (make_sliding_window_fit,)
from dfextensions.groupby_regression import make_parallel_fit_v4
import matplotlib
from perfmonitor import PerformanceLogger
from dfextensions.dfdraw import set_style
from matplotlib.backends.backend_pdf import PdfPages

logger = PerformanceLogger("perf_log.txt")
time_fmt = FuncFormatter(lambda x, _: datetime.utcfromtimestamp(x).strftime('%H:%M'))

def _is_interactive():
    try:
        if hasattr(__builtins__, '__IPYTHON__'):
            return True
    except Exception:
        pass
    if 'IPython' in sys.modules:
        return True
    if hasattr(sys, 'ps1'):           # `python -i ...` also counts as interactive
        return True
    return False

if _is_interactive():
    for _backend in ('MacOSX', 'TkAgg', 'Qt5Agg'):
        try:
            matplotlib.use(_backend)
            break
        except Exception:
            continue                  # try next; fall back to mpl default
    import matplotlib.pyplot as plt
    plt.ion()
else:
    # Script / batch context: Agg is headless, no display needed.
    # setdefault so user-set MPLBACKEND wins if explicitly chosen.
    os.environ.setdefault("MPLBACKEND", "Agg")
    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    # no plt.ion() — batch jobs should not open GUI windows

# ============================================================================
# Configuration
# ============================================================================

# Default data file: local symlink (or real file) next to this script.
# Convention: `ln -s /path/to/real/data.root time_series_tracks_0.root`.
# `__file__` works with both `python time_series.py` and IPython `%run`.
DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"time_series_tracks_0.root",)

def describe_columns(df, nrows=100_000):
    """Print dtype, min, max, nunique, and per-column memory + total.
    nrows: rows used for min/max/nunique stats (None = full table).
    Memory is always computed on the full DataFrame.
    """
    sample = df if (nrows is None or len(df) <= nrows) else df.iloc[:nrows]
    mem = df.memory_usage(deep=True, index=False)

    def safe_stat(fn):
        result = {}
        for c in sample.columns:
            try:
                result[c] = fn(sample[c])
            except Exception:
                result[c] = None
        return pd.Series(result)

    info = pd.DataFrame({
        'dtype':   df.dtypes.astype(str),
        'min':     safe_stat(lambda s: s.min()),
        'max':     safe_stat(lambda s: s.max()),
        'nunique': safe_stat(lambda s: s.nunique()),
        'bytes':   mem,
        'MB':      (mem / 1024**2).round(3),
    })
    nrows_used = len(sample)
    print(f"Stats on {nrows_used:,} of {len(df):,} rows  ({nrows_used/max(len(df),1)*100:.0f}%)")
    print(info.to_string())
    total_b  = int(mem.sum())
    total_mb = total_b / 1024**2
    bpr      = total_b / max(len(df), 1)
    print(f"\nTotal: {total_mb:.2f} MB,  {len(df):,} rows × {len(df.columns)} cols,  avg {bpr:.0f} bytes/row")
    return info
def root_to_adf(path, tree_name=None, branches=None, cut=None,
                entry_start=None, entry_stop=None, use_adf_read_tree=True):
    """Read a ROOT TTree and return an AliasDataFrame.

    Fast path: if the tree was written via ``AliasDataFrame.export_tree``,
    dtypes and schema are restored from ``UserInfo``. Otherwise falls back
    to a generic uproot read (any TTree).
    """
    if use_adf_read_tree:
        try:
            return AliasDataFrame.read_tree(
                path, tree_name=tree_name, branches=branches,
                entry_start=entry_start, entry_stop=entry_stop,
            )
        except Exception as e:
            print(f"[root_to_adf] read_tree path failed ({e}); "
                  "falling back to uproot")

    with uproot.open(path) as f:
        if tree_name is None:
            tree_name = next(
                k.rsplit(";", 1)[0] for k, v in f.items()
                if hasattr(v, "num_entries")
            )
        tree = f[tree_name]
        df = tree.arrays(
            branches, cut=cut,
            entry_start=entry_start, entry_stop=entry_stop,
            library="pd",
        )

    if isinstance(df, tuple):  # jagged branches produced multiple frames
        df = pd.concat(df, axis=1)

    # df = change type of all ["dca*_.", "delta.*","dEdx.] columns to float16
    cols = [c for c in df.columns if re.match(r'(dca.*|delta.*|dEdx.*|cov.*)', c)]
    df[cols] = df[cols].astype('float16')
    return AliasDataFrame(df)





def make_row_group_mask(mask, group_size=5, threshold=2, dtype=np.uint32, add_first_last=True, missing_value=255):
    """
    Convert per-row cluster bool mask into compact grouped mask.

    For each track, rows are grouped into blocks of `group_size`.
    A group is active if the number of hit rows is > threshold.

    Example
    -------
    group_size=5, threshold=2:
        active group means at least 3 of 5 rows have clusters.

    Parameters
    ----------
    mask : awkward.Array or numpy.ndarray
        Shape (n_tracks, n_rows), e.g. n_rows=152.
    group_size : int
        Rows per output bit.
    threshold : int
        Group active if count > threshold.
    dtype : numpy dtype
        Integer dtype for packed group mask.
    add_first_last : bool
        If True, also return first/last active group as uint8.
    missing_value : int
        Sentinel for no active group. Default 255.

    Returns
    -------
    result : dict
        {
          "rowmask": np.ndarray uint32, shape (n_tracks,),
          "group_counts": np.ndarray uint8, shape (n_tracks, n_groups),
          "first": np.ndarray uint8, optional,
          "last": np.ndarray uint8, optional,
        }
    """
    arr = ak.to_numpy(mask) if isinstance(mask, ak.Array) else np.asarray(mask)
    arr = arr.astype(bool)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D mask array, got shape {arr.shape}")

    n_tracks, n_rows = arr.shape
    n_groups = (n_rows + group_size - 1) // group_size

    if n_groups > np.iinfo(dtype).bits:
        raise ValueError(
            f"{n_groups} groups do not fit into {np.dtype(dtype).name}; "
            f"use a larger dtype or multiple words"
        )

    group_counts = np.zeros((n_tracks, n_groups), dtype=np.uint8)
    for g in range(n_groups):
        lo = g * group_size
        hi = min((g + 1) * group_size, n_rows)
        group_counts[:, g] = arr[:, lo:hi].sum(axis=1)
    active = group_counts > threshold

    rowmask = np.zeros(n_tracks, dtype=dtype)
    for g in range(n_groups):
        rowmask |= active[:, g].astype(dtype) << g
    result = {
        "rowmask": rowmask,
        "group_counts": group_counts,
    }
    if add_first_last:
        any_active = active.any(axis=1)
        first = np.full(n_tracks, missing_value, dtype=np.uint8)
        last = np.full(n_tracks, missing_value, dtype=np.uint8)
        first[any_active] = np.argmax(active[any_active], axis=1).astype(np.uint8)
        # last active index = n_groups - 1 - first active in reversed mask
        last[any_active] = (
                n_groups - 1 - np.argmax(active[any_active, ::-1], axis=1)
        ).astype(np.uint8)
        result["first"] = first
        result["last"] = last

    return result



df_TimeSeriesAliases = {
    # GB binning aliases
    "sector_bin180":{"expr": f"90*(phiITSTPCAtVertex/{np.pi})","dtype": np.uint16, "title": "Sector bin180"},
    "sector_bin360":{"expr": f"180*(phiITSTPCAtVertex/{np.pi})","dtype": np.uint8, "title": "Sector bin360"},
    "tgl_bin10":{"expr": f"10*tgl","dtype": np.int8, "title": "PzPt bin10"},
    "qpt_bin5":{"expr": f"5*qpt","dtype": np.int8, "title": "q/pT bin5"},
    "qpt_bin10":{"expr": f"10*qpt","dtype": np.int8, "title": "q/pT bin10"},
    #
    "hasITS": {"expr": "hasITSTPC>0", "dtype": bool, "title": "Track matched in ITS"},
    # Expected resolution
    #
    "sectorV": {"expr": f"18*(phiITSTPCAtVertex/{np.pi})","dtype": np.float16,"title": "Float  TPC sector (0-18) from phiITSTPCAtVertex",},
    "sector": {"expr": f"18*(phi/{np.pi})","dtype": np.float16,"title": "Float  TPC sector (0-18) from phi",},
    "dsector": {"expr": f"sector-int(sector)","dtype": np.float16,"title": "TPC Δ sector",},
    "dsector_bin20": {"expr": f"(sector-int(sector))*20","dtype": np.int8,"title": "TPC Δ sector _ bin 20",},
    #
    "time_s": {"expr": "timeMS/1000","dtype": np.float64,"title": "Time stamp (s) ",},
    # adf.add_alias("sector", f"18*(phiITSTPCAtVertex/{np.pi})")
    "baseTPCCut0": {"expr": "(ncl>60)&(abs(dcaZFromDeltaTime)<10)", "dtype": bool, "title": "Base TPC track selection0- ncl>60, dcaZFromDeltaTime<10cm"},
    "baseITSTPCCut0": {"expr": "(baseTPCCut0)&(hasITSTPC>0)&(sqrt(dcar_itstpc**2+dcaZFromDeltaTime**2)<1)", "dtype": bool, "title": "Base track TPC+ITS selection0 + hasITSTPC"},
    # vertex aliases
    "vertexOK0": {"expr": "(vertex_nContributors>5)&(sqrt(vertex_x**2+vertex_y**2)<1)", "dtype": bool, "title": "Vertex quality flag: >0 contributors, |x|<1cm"},
    # adf.add_alias("dphi", "0.00299792 * 5 * mX * qpt", dtype=np.float32)
    # adf.add_alias("dphiTPCITS", "0.00299792 * 0.5 * mX * qpt_ITSTPC", dtype=np.float32). # we should ge ther B field
    "dphiTPCITSIn": {"expr": "0.00299792 * 0.5 * mX * qpt_ITSTPC", "dtype": np.float32, "title": "dφ TPC-ITS inner (rad)"},
    "phiTPCITSIn": {"expr": "phiITSTPCAtVertex+0.00299792 * 0.5 * mX * qpt_ITSTPC", "dtype": np.float32, "title": "φ TPC-ITS inner at  mX"}, # we should ge ther B field
    "fsectorIn": {"expr": "(9*((phiITSTPCAtVertex+0.00299792 * 0.5 * mX * qpt_ITSTPC)/np.pi))", "dtype": np.float32, "title": "Float  TPC sector (0-18) from phiTPCITSIn"},
    "dsectorIn": {"expr": "fsectorIn-int(fsectorIn)", "dtype": np.float32, "title": "TPC Δ sector from phiTPCITSIn"},
    "fsector": {"expr": "(9*(phi/np.pi))", "dtype": np.float32, "title": "Float  TPC sector (0-18) from phi"},
    "dsector": {"expr": "fsector-int(fsector)", "dtype": np.float32, "title": "TPC Δ sector from phi"},
}


"""
adf.add_alias("dphiTPCITSIn", "0.00299792 * 0.5 * mX * qpt_ITSTPC", dtype=np.float32) # we should ge ther B field 
adf.add_alias("phiTPCITSIn", "phiITSTPCAtVertex+0.00299792 * 0.5 * mX * qpt_ITSTPC", dtype=np.float32) # we should ge ther B field
adf.add_alias("fsectorIn", "(9*((phiITSTPCAtVertex+0.00299792 * 0.5 * mX * qpt_ITSTPC)/np.pi))", dtype=np.float16) 
adf.add_alias("dsectorIn", "fsectorIn-int(fsectorIn)", dtype=np.float16)
adf.add_alias("fsector", "(9*(phi/np.pi))", dtype=np.float16)
adf.add_alias("dsector", "fsector-int(fsector)", dtype=np.float16)

 adf.materialize_aliases(names=["dsectorIn"])
 
"""


df_TimeSeriesMeta = {
    # ── Event / trigger weights ──────────────────────────────────────────
    "triggerMask":          {"title": "Trigger mask (bitmask)"},
    "factorMinBias":        {"title": "Min-bias downscaling factor"},
    "factorPt":             {"title": r"$p_{T}$-dependent correction factor"},
    "weight":               {"title": "Event weight"},

    # ── DCA (Distance of Closest Approach) ───────────────────────────────
    "dcar_tpc_vertex":      {"title": r"TPC $\mathrm{DCA}_{r}$ to vertex (cm)"},
    "dcar_tpc":             {"title": r"TPC $\mathrm{DCA}_{r}$ (cm)"},
    "dcaz_tpc":             {"title": r"TPC $\mathrm{DCA}_{z}$ (cm)"},
    "dcar_itstpc":          {"title": r"ITS-TPC $\mathrm{DCA}_{r}$ to vertex (cm)"},
    "dcaz_itstpc":          {"title": r"ITS-TPC $\mathrm{DCA}_{z}$ to vertex (cm)"},
    "dcarW":                {"title": r"Weighted $\mathrm{DCA}_{r}$ (cm)"},
    "dcaZFromDeltaTime":    {"title": r"$\mathrm{DCA}_{z}$ derived from $\delta t$ (cm)"},

    # ── Track matching flags ─────────────────────────────────────────────
    "hasITSTPC":            {"title": "Track matched in both ITS and TPC"},

    # ── Primary vertex ───────────────────────────────────────────────────
    "vertex_x":             {"title": "Primary vertex $x$ (cm)"},
    "vertex_y":             {"title": "Primary vertex $y$ (cm)"},
    "vertex_z":             {"title": "Primary vertex $z$ (cm)"},
    "vertex_time":          {"title": r"Primary vertex time ($\mu$s)"},
    "vertex_nContributors": {"title": "Number of vertex contributors"},
    "isNearestVertex":      {"title": "Track assigned to nearest vertex (flag)"},

    # ── Kinematics ───────────────────────────────────────────────────────
    "pt":                   {"title": r"$p_{T}$ (GeV/$c$)"},
    "qpt":                  {"title": r"TPC $q/p_{T}$ (1/GeV)"},
    "qpt_ITSTPC":           {"title": r"ITS-TPC $q/p_{T}$ (1/GeV)"},
    "tgl":                  {"title": r"$\tan\lambda = p_{z}/p_{T}$"},
    "phi":                  {"title": r"Track azimuthal angle $\varphi$ (rad)"},
    "side_type":            {"title": "TPC side (A=0, C=1)"},

    # ── TPC cluster counters ─────────────────────────────────────────────
    "ncl":                  {"title": "TPC cluster count"},
    "ncl_shared":           {"title": "TPC shared cluster count"},
    "clusterMask":          {"title": "TPC cluster occupancy mask (bitmask)"},

    # ── TPC timing ───────────────────────────────────────────────────────
    "tpc_timebin":          {"title": "TPC time bin"},

    # ── dE/dx — total charge ─────────────────────────────────────────────
    "dEdxTotIROC":          {"title": r"$\mathrm{d}E/\mathrm{d}x$ total IROC (a.u.)"},
    "dEdxTotOROC1":         {"title": r"$\mathrm{d}E/\mathrm{d}x$ total OROC1 (a.u.)"},
    "dEdxTotOROC2":         {"title": r"$\mathrm{d}E/\mathrm{d}x$ total OROC2 (a.u.)"},
    "dEdxTotOROC3":         {"title": r"$\mathrm{d}E/\mathrm{d}x$ total OROC3 (a.u.)"},
    "dEdxTotTPC":           {"title": r"$\mathrm{d}E/\mathrm{d}x$ total TPC (a.u.)"},

    # ── dE/dx — max charge ───────────────────────────────────────────────
    "dEdxMaxIROC":          {"title": r"$\mathrm{d}E/\mathrm{d}x$ max IROC (a.u.)"},
    "dEdxMaxOROC1":         {"title": r"$\mathrm{d}E/\mathrm{d}x$ max OROC1 (a.u.)"},
    "dEdxMaxOROC2":         {"title": r"$\mathrm{d}E/\mathrm{d}x$ max OROC2 (a.u.)"},
    "dEdxMaxOROC3":         {"title": r"$\mathrm{d}E/\mathrm{d}x$ max OROC3 (a.u.)"},
    "dEdxMaxTPC":           {"title": r"$\mathrm{d}E/\mathrm{d}x$ max TPC (a.u.)"},

    # ── Pad-row hit counters (above / sub-threshold) ─────────────────────
    "NHitsIROC":                  {"title": "Pad-row hits IROC"},
    "NHitsSubThresholdIROC":      {"title": "Sub-threshold hits IROC"},
    "NHitsOROC1":                 {"title": "Pad-row hits OROC1"},
    "NHitsSubThresholdOROC1":     {"title": "Sub-threshold hits OROC1"},
    "NHitsOROC2":                 {"title": "Pad-row hits OROC2"},
    "NHitsSubThresholdOROC2":     {"title": "Sub-threshold hits OROC2"},
    "NHitsOROC3":                 {"title": "Pad-row hits OROC3"},
    "NHitsSubThresholdOROC3":     {"title": "Sub-threshold hits OROC3"},

    # ── Track fit quality ────────────────────────────────────────────────
    "chi2":                 {"title": r"TPC track fit $\chi^{2}/\mathrm{ndf}$"},
    "mX":                   {"title": "TPC track reference radius $x$ (cm)"},
    "mX_ITS":               {"title": "ITS track reference radius $x$ (cm)"},

    # ── ITS ──────────────────────────────────────────────────────────────
    "nClITS":               {"title": "ITS cluster count"},
    "chi2ITS":              {"title": r"ITS track fit $\chi^{2}/\mathrm{ndf}$"},
    "chi2match_ITSTPC":     {"title": r"ITS-TPC matching $\chi^{2}$"},
    "sqrtChi2Match":        {"title": r"$\sqrt{\chi^{2}}$ ITS-TPC matching"},

    # ── PID ──────────────────────────────────────────────────────────────
    "PID":                  {"title": "PID hypothesis (integer code)"},

    # ── Covariance matrix elements at vertex ─────────────────────────────
    "covTPCAtVertex0":      {"title": r"TPC $\mathrm{cov}(y,y)$ at vertex (cm$^{2}$)"},
    "covTPCAtVertex1":      {"title": r"TPC $\mathrm{cov}(z,z)$ at vertex (cm$^{2}$)"},
    "covTPCConstrVtxP2":    {"title": r"TPC constrained $P_{2}$ ($\sin\varphi$)"},
    "covTPCConstrVtxP3":    {"title": r"TPC constrained $P_{3}$ ($\tan\lambda$)"},
    "covTPCConstrVtxP4":    {"title": r"TPC constrained $P_{4}$ ($q/p_{T}$)"},
    "covITSTPCConstrVtxP2": {"title": r"ITS-TPC constrained $P_{2}$ ($\sin\varphi$)"},
    "covITSTPCConstrVtxP3": {"title": r"ITS-TPC constrained $P_{3}$ ($\tan\lambda$)"},
    "covITSTPCConstrVtxP4": {"title": r"ITS-TPC constrained $P_{4}$ ($q/p_{T}$)"},

    # ── ITS-TPC parameter deltas at vertex ────────────────────────────────
    "deltaP2ConstrVtx":     {"title": r"$\Delta\sin\varphi$ ITS-TPC at vtx"},
    "deltaP3ConstrVtx":     {"title": r"$\Delta\tan\lambda$ ITS-TPC at vtx"},
    "deltaP4ConstrVtx":     {"title": r"$\Delta q/p_{T}$ ITS-TPC at vtx (1/GeV)"},

    # ── ITS outer-matching parameter deltas ──────────────────────────────
    "deltaPar0":            {"title": r"$\Delta y$ ITS-TPC outer match (cm)"},
    "deltaPar1":            {"title": r"$\Delta z$ ITS-TPC outer match (cm)"},
    "deltaPar2":            {"title": r"$\Delta\sin\varphi$ ITS-TPC outer match"},
    "deltaPar3":            {"title": r"$\Delta\tan\lambda$ ITS-TPC outer match"},
    "deltaPar4":            {"title": r"$\Delta q/p_{T}$ ITS-TPC outer match (1/GeV)"},
    "deltaP0OuterITS":      {"title": r"$\Delta y$ outer ITS propagation (cm)"},
    "deltaP1OuterITS":      {"title": r"$\Delta z$ outer ITS propagation (cm)"},
    "deltaP2OuterITS":      {"title": r"$\Delta\sin\varphi$ outer ITS propagation"},
    "deltaP3OuterITS":      {"title": r"$\Delta\tan\lambda$ outer ITS propagation"},
    "deltaP4OuterITS":      {"title": r"$\Delta q/p_{T}$ outer ITS propagation (1/GeV)"},
    "mXOuterMatching":      {"title": "Reference $x$ outer ITS matching (cm)"},

    # ── Track parameter uncertainties ────────────────────────────────────
    "sigmaY2":              {"title": r"$\sigma^{2}(y)$ at reference surface (cm$^{2}$)"},
    "sigmaZ2":              {"title": r"$\sigma^{2}(z)$ at reference surface (cm$^{2}$)"},

    # ── TPC in/out parameter deltas ──────────────────────────────────────
    "deltaTPCParamInOutTgl": {"title": r"$\Delta\tan\lambda$ TPC inner vs outer"},
    "deltaTPCParamInOutQPt": {"title": r"$\Delta q/p_{T}$ TPC inner vs outer (1/GeV)"},

    # ── TOF ──────────────────────────────────────────────────────────────
    "tpcYDeltaAtTOF":       {"title": r"TPC $\Delta y$ at TOF radius (cm)"},
    "tpcZDeltaAtTOF":       {"title": r"TPC $\Delta z$ at TOF radius (cm)"},
    "mDXatTOF":             {"title": r"TPC-TOF $\Delta x$ at TOF (cm)"},
    "mDZatTOF":             {"title": r"TPC-TOF $\Delta z$ at TOF (cm)"},
    "mL":                   {"title": "Track length to TOF (cm)"},
    "mX2X0":                {"title": r"Material budget $X/X_{0}$ to TOF"},
    "mXRho":                {"title": r"$\int\rho\,\mathrm{d}x$ to TOF (g/cm$^{2}$)"},
    "mT[9]":                {"title": "Expected TOF signal per PID hypothesis (ps)"},
    "mTOFSignal":           {"title": "Measured TOF signal (ps)"},
    "mDeltaTTOFTPC":        {"title": r"TOF-TPC $\Delta t$ (ps)"},
    "TOFmask":              {"title": "TOF matching mask (bitmask)"},
    "TOFchannel":           {"title": "TOF channel index"},

    # ── Multiplicity & timing ────────────────────────────────────────────
    "mult":                 {"title": "Track multiplicity in time window"},
    "time_window_mult":     {"title": r"Multiplicity time window ($\mu$s)"},
    "firstTFOrbit":         {"title": "First LHC orbit of TF"},
    "timeMS":               {"title": "Track timestamp (ms)"},
    "vertexTime":           {"title": r"Vertex time ($\mu$s)"},
    "trackTime0":           {"title": r"Track reference time $t_{0}$ ($\mu$s)"},

    # ── Run / calibration ────────────────────────────────────────────────
    "run":                  {"title": "Run number"},
    "mVDrift":              {"title": r"TPC $v_{\mathrm{drift}}$ ($\mu$m/$\mu$s)"},
    "its_flag":             {"title": "ITS track quality flag (bitmask)"},

    # ── Derived / aliases ────────────────────────────────────────────────
    "phiITSTPCAtVertex":    {"title": r"ITS-TPC $\varphi$ at vertex (rad)"},
    "sector":               {"title": "TPC sector index (0–17)"},
    "time_s":               {"title": "Timestamp (s)"},
}
def apply_meta(adf, aliases_spec, errors='raise', logger=None):
    """Apply alias expressions and axis titles from a spec dict.
    Parameters
    ----------
    adf : AliasDataFrame
    aliases_spec : dict[str, dict]
        name → spec with optional 'expr'+'dtype' and/or 'title'.
    errors : {'raise', 'warn'}, default 'raise'.
    logger : object with .log(msg), or None (falls back to print).
    Returns
    -------
    {'applied': [(name, op), ...], 'failed': [(name, op, exc), ...]}.
    """
    if errors not in ('raise', 'warn'): raise ValueError(f"apply_meta: errors must be 'raise' or 'warn', got {errors!r}")
    log = (lambda m: logger.log(m)) if logger is not None else print
    applied, failed = [], []
    def _fail(op, exc):
        if errors == 'raise': raise exc
        failed.append((name, op, exc))
        log(f"apply_meta: {op}({name!r}) FAILED — {type(exc).__name__}: {exc}")
    for name, spec in aliases_spec.items():
        if "expr" in spec:
            if "dtype" not in spec:
                _fail('add_alias', KeyError(f"spec[{name!r}] has 'expr' but no 'dtype'"))
            else:
                try: adf.add_alias(name, spec["expr"], dtype=spec["dtype"]); applied.append((name, 'add_alias'))
                except Exception as exc: _fail('add_alias', exc)
        if "title" in spec:
            try: adf.set_axis_title(name, spec["title"]); applied.append((name, 'set_axis_title'))
            except Exception as exc: _fail('set_axis_title', exc)
    return {"applied": applied, "failed": failed}


def addTimeQuantiles(adf, varname="timeMS", step=10000, step2=100):
    """
    Two-level time-quantile binning with approximately flat statistics per bin.
    Level 1 (fine):   quantile_bin   — round(N/step) bins of ~step tracks sorted by varname
    Level 2 (coarse): quantile_binGB — round(n_bins/step2) coarse bins of ~step2 fine bins
    Both levels use equal-split assignment (counts differ by at most 1).
    Registers:
        column   "quantile_bin"   on adf.df
        column   "quantile_binGB" on adf.df
        subframe "TimeQuantiles"  — quantile_bin, quantile_binGB, {varname}_min/max/mean, count
    :param adf:     AliasDataFrame with column `varname`
    :param varname: time variable to bin on (default "timeMS")
    :param step:    approximate tracks per fine bin (default 10000)
    :param step2:   approximate fine bins per coarse bin (default 100)
    :return:        adf with bin columns and TimeQuantiles subframe registered
    """
    logger.log(f"addTimeQuantiles::BEGIN varname={varname} step={step} step2={step2} tracks={len(adf.df):,}")
    n        = len(adf.df)
    n_bins   = max(1, round(n / step))
    sorted_idx              = np.argsort(adf.df[varname].values, kind="stable")
    bin_labels              = np.empty(n, dtype=np.int32)
    bin_labels[sorted_idx]  = np.arange(n) * n_bins // n

    adf.df["quantile_bin"]  = bin_labels
    dfGBT = adf.df.groupby("quantile_bin")[varname].agg(
        **{f"{varname}_min": "min", f"{varname}_max": "max", f"{varname}_mean": "mean", "count": "count"}
    ).reset_index()

    n_binsGB               = max(1, round(n_bins / step2))
    dfGBT["quantile_binGB"] = np.arange(len(dfGBT)) * n_binsGB // len(dfGBT)
    adf.df["quantile_binGB"] = dfGBT.set_index("quantile_bin")["quantile_binGB"].reindex(adf.df["quantile_bin"]).values.astype(np.int32)

    adf.register_subframe("TimeQuantiles", AliasDataFrame(dfGBT), index_columns=["quantile_bin"])
    logger.log(f"addTimeQuantiles::END n_bins={n_bins} n_binsGB={n_binsGB}")
    return adf



def compressADF(adfC, column_array, gb_columns, index_columns,  subframe_name="CompressGB", dtype=np.int8, precision=5, coding="linear"):
    """
    Generic GB-aware delta compression of ADF columns with per-column distance type.

    Step 1: Resolve column_array dict {regex/colname: distance} → {col: distance}.
            GroupBy gb_columns → median and MAD per group, distance-aware:
                linear: mad = median(|var − median(var)|)
                log:    mad = median(|log(var / median(var))|)   [log-space MAD]
    Step 2: Register GB map as subframe `subframe_name`.
    Step 3: Delta-code per column using its distance type:
                linear: delta = (var − median) / mad
                log:    delta = log(var / median) / mad
            mad=0 guard: fall back to mean of per-bin MADs (TODO: expose as parameter).
    Step 4: Apply coding + quantize to dtype:
                linear: round(delta * scale)
                log:    round(sign(delta) * log1p(|delta|) * scale)
                asinh:  round(arcsinh(delta) * scale)
            where scale = dtype.max / precision
    Step 5: Register decompression aliases (distance + coding aware):
                linear dist + linear coding: median + (coded/scale) * mad
                linear dist + asinh coding:  median + sinh(coded/scale) * mad
                log dist    + linear coding: median * exp((coded/scale) * mad)
                log dist    + asinh coding:  median * exp(sinh(coded/scale) * mad)
                (log coding variants follow the same pattern)

    :param adfC:          AliasDataFrame to compress
    :param column_array:  dict {regex_or_colname: "linear"|"log"}
                          e.g. {"dcar.*|delta.*": "linear", "dEdx.*|ncl": "log"}
                          list or str also accepted (all default to "linear")
    :param gb_columns:    groupby keys e.g. ["quantile_bin"]
    :param index_columns: column name(s) copied from adfC.df into the output ADF,
    :param subframe_name: name for GB statistics subframe (default "CompressGB")
    :param dtype:         output integer dtype (default np.int8)
    :param precision:     MAD units mapped to dtype.max (default 5)
    :param coding:        quantization transform: "linear" | "log" | "asinh"
    :return:              AliasDataFrame with compressed columns, GB subframe, decompress aliases
    """
    logger.log(f"compressADF::BEGIN coding={coding} dtype={dtype} precision={precision}")

    # ── Step 1: resolve column_array → {col: distance} ───────────────────
    if isinstance(column_array, (str, list)):
        column_array = {p: "linear" for p in ([column_array] if isinstance(column_array, str) else column_array)}
    col_distance = {}
    for pattern, dist in column_array.items():
        matched = adfC.df.filter(regex=pattern).columns.tolist() if pattern not in adfC.df.columns else [pattern]
        for c in matched:
            if c not in col_distance:          # first match wins
                col_distance[c] = dist
    cols = list(col_distance.keys())
    logger.log(f"compressADF::Step1 - resolved cols={cols}")

    def _mad_linear(x): return np.median(np.abs(x - x.median()))
    def _mad_log(x):    return np.median(np.abs(np.log(x / x.median())))

    dfGB = adfC.df.groupby(gb_columns).agg(**{
        **{f"{c}_median": (c, "median")                                              for c in cols},
        **{f"{c}_mad":    (c, _mad_linear if col_distance[c] == "linear" else _mad_log) for c in cols},
    }).reset_index()

    # ── Step 2: register GB subframe ─────────────────────────────────────
    adfC.register_subframe(subframe_name, AliasDataFrame(dfGB), index_columns=gb_columns)
    logger.log(f"compressADF::Step2 - registered {subframe_name}  shape={dfGB.shape}")

    # ── Step 3+4: delta-code + quantize ──────────────────────────────────
    scale     = np.iinfo(dtype).max / precision
    lo, hi    = np.iinfo(dtype).min, np.iinfo(dtype).max
    df_merged = adfC.df[gb_columns + cols].merge(dfGB, on=gb_columns, how="left")
    df_out    = adfC.df[gb_columns].copy()
    encode    = {"linear": lambda d: d * scale,
                 "log":    lambda d: np.sign(d) * np.log1p(np.abs(d)) * scale,
                 "asinh":  lambda d: np.arcsinh(d) * scale}[coding]

    for col, dist in col_distance.items():
        mad_safe = df_merged[f"{col}_mad"].where(df_merged[f"{col}_mad"] > 0, dfGB[f"{col}_mad"].mean())
        delta    = ((df_merged[col] - df_merged[f"{col}_median"]) / mad_safe if dist == "linear"
                    else np.log(df_merged[col] / df_merged[f"{col}_median"]) / mad_safe)
        df_out[col] = np.clip(np.round(encode(delta)), lo, hi).astype(dtype)

    # ── Step 5: decompression aliases ────────────────────────────────────
    logger.log("compressADF::Step5 - registering decompression aliases")
    sf     = subframe_name
    if index_columns:
        df_out[index_columns] = adfC.df[index_columns].copy()
    adfOut = AliasDataFrame(df_out)
    adfOut.register_subframe(subframe_name, AliasDataFrame(dfGB), index_columns=gb_columns)

    _delta_decode = {"linear": f"{{col}} / {scale}",
                     "log":    f"sign({{col}}) * expm1(abs({{col}}) / {scale})",
                     "asinh":  f"sinh({{col}} / {scale})"}[coding]
    _decomp = {"linear": f"{{sf}}.{{col}}_median + ({_delta_decode}) * {{sf}}.{{col}}_mad",
               "log":    f"{{sf}}.{{col}}_median * exp(({_delta_decode}) * {{sf}}.{{col}}_mad)"}

    for col, dist in col_distance.items():
        expr = _decomp[dist].format(sf=sf, col=col)
        adfOut.add_alias(f"{col}_decomp", expr, dtype=np.float32)

    logger.log(f"compressADF::END  shape={df_out.shape}")
    return adfOut


def calibVertex(adf,qaPlots=False):
    """
    Calibrate vertex time-series properties per time-quantile bin.
    Step 1: linear fit  var = intercept + slope * vertex_z  per quantile_bin
            → captures beam-spot tilt and z-dependence of each vertex property.

    Step 2: compute spread (MAD, std) of fit residuals per quantile_bin
            → captures vertex resolution / intrinsic spread independent of z-tilt.

    Step 3: make compressed time series GB 100 bins median and mad

    Requires: addTimeQuantiles(adf) called first (quantile_bin column must exist).

    :param adf: AliasDataFrame with vertex_* columns and quantile_bin assigned
    :return:    adf with CalibVertex subframe and <var>_predicted aliases registered
    """
    adf.materialize_aliases(names=["time_s","vertexOK0"])
    vars   = ['vertex_x', 'vertex_y', 'vertex_z', 'vertex_nContributors']
    gbVars = ["quantile_bin"]
    median_columns=["time_s","vertex_z"]
    isOK= adf.df.eval("vertexOK0")
    #
    # ── Step 1: linear fit var = a + b*vertex_z per quantile_bin ─────────
    logger.log("calibVertex::Step1 - linear fit BEGIN")
    _, dfCoeffs = make_parallel_fit_v4(
        df=adf.df, gb_columns=gbVars, fit_columns=vars,
        linear_columns=["vertex_z"], fit_intercept=True,
        suffix="", min_stat=10, addPrediction=False,
        selection=isOK,
        median_columns=median_columns,
    )
    df_medians = adf.df.loc[isOK].groupby(gbVars)[median_columns].median().reset_index()
    dfCoeffs = dfCoeffs.merge(df_medians, on=gbVars, how='left')
    logger.log(f"calibVertex::Step1 - linear fit END  shape={dfCoeffs.shape}")
    adfVertex = AliasDataFrame(dfCoeffs)
    #
    # ── Step 2: residual MAD + std per quantile_bin ───────────────────────
    logger.log("calibVertex::Step2 - residual spread BEGIN")
    adf.register_subframe("CalibVertex", adfVertex, index_columns=gbVars)
    avars=[]
    for var in vars:
        adf.add_alias(f"{var}_delta",f"abs({var} - (CalibVertex.{var}_intercept + CalibVertex.{var}_slope_vertex_z * vertex_z))",dtype=np.float32, fill_value=0)
        adf.add_alias(f"a{var}_res",f"abs({var} - (CalibVertex.{var}_intercept + CalibVertex.{var}_slope_vertex_z * vertex_z))",dtype=np.float32, fill_value=0)
        avars.append(f"a{var}_res")
    adf.materialize_aliases(names=[f"a{var}_res" for var in vars])
    logger.log("calibVertex::Step2 - linear fit BEGIN")
    _, dfCoeffsMAD = make_parallel_fit_v4(
        df=adf.df, gb_columns=gbVars, fit_columns=avars,
        linear_columns=["vertex_z"], fit_intercept=True,
        suffix="", min_stat=10, addPrediction=False,
    )
    logger.log(f"calibVertex::Step2 - linear fit END  shape={dfCoeffsMAD.shape}")
    adfVertex.df=adfVertex.df.merge(dfCoeffsMAD, on=gbVars, suffixes=["", "_MAD"])
    adf.register_subframe("CalibVertex", adfVertex, index_columns=gbVars)
    """
    if qaPlots:
    #
    adf.CalibVertex.draw("avertex_x_res_intercept:time_s",type="profile")
    """
    return adfVertex



def calibBiasResolution(adf):
    #
    # make sliding window fit of the variables as function of the sector10, tgl as a function of qpt
    # 1.) make linear fit of the variables as function of qpt, register in adf as subframe and aliasses
    # 2.) Make linear fit if the var-<var> as function of qpt , register in adf as subframe and aliases
    varList=["dcar_tpc","dcar_tpc_vertex","dcaz_tpc_vertex","dcar_itstpc","dcaz_itstpc","hasITS"]
    gbVars=["sector_bin180","tgl_bin10"]
    linear=["qpt"]
    #
    # Step0: materialize binning aliases if needed
    #
    varListDCA = ["dcar_tpc_vertex", "dcar_tpc", "dcaz_tpc", "dcar_itstpc", "dcaz_itstpc"]
    # ITS-TPC matching residuals at outer surface (y, z, sin φ, tan λ, q/pT)
    varListDeltaPar = ["deltaPar0", "deltaPar1", "deltaPar2", "deltaPar3", "deltaPar4"]
    # ITS-TPC constrained-to-vertex deltas (sin φ, tan λ, q/pT)
    varListDeltaConstrVtx = ["deltaP2ConstrVtx", "deltaP3ConstrVtx", "deltaP4ConstrVtx"]
    # TPC inner vs outer consistency (sensitive to space-charge, B-field)
    varListDeltaTPCInOut = ["deltaTPCParamInOutTgl", "deltaTPCParamInOutQPt"]
    # Outer ITS propagation deltas (y, z, sin φ, tan λ, q/pT)
    varListDeltaOuterITS = ["deltaP0OuterITS", "deltaP1OuterITS", "deltaP2OuterITS","deltaP3OuterITS", "deltaP4OuterITS"]
    # Combined — all delta variables for calibBiasResolution Step 1
    varListDelta = varListDeltaPar + varListDeltaConstrVtx + varListDeltaTPCInOut + varListDeltaOuterITS
    #
    # Step1: make linear fit of the variables as function of qpt, register in adf as subframe and aliasses
    #
    logger.log("Step1 calibBiasResolution: make sliding window fit of the variables as function of the sector10, tgl as a function of qpt -BEGIN")
    varList = varListDCA +varListDelta
    gbVars  = ["sector_bin180", "tgl_bin10"]
    linear  = ["qpt"]
    suffix  = ""
    logger.log("Step1.0 calibBiasResolution: materializing binning aliases if needed -BEGIN")
    for col in gbVars + linear+ ["baseITSTPCCut0"]:
        if col not in adf.df.columns:
            adf.materialize_alias(col)
    logger.log("Step1.0 calibBiasResolution: materializing binning aliases if needed -END")
    logger.log("Step1 calibBiasResolution: Fit BEGIN")
    selection= adf.df.eval("(baseITSTPCCut0)&(hasITSTPC>0)")
    np.random.seed(42)
    subsample = np.random.rand(len(adf.df)) < 0.1
    combined_selection = selection & subsample
    _, dfCoeffs1 = make_parallel_fit_v4(
        df=adf.df,
        gb_columns=gbVars,           # ["sector_bin180", "tgl_bin10"]
        fit_columns=varList,         # DCA variables
        linear_columns=linear,       # ["qpt"]
        fit_intercept=True,
        suffix=suffix,               #
        min_stat=20,
        selection=combined_selection,
        addPrediction=False,
    )
    logger.log(f"calibBiasResolution::Step1 - Sliding window fit END  bins={len(dfCoeffs1)}")
    # ── Register coefficient table as subframe ────────────────────────────
    adfCalibBias1=AliasDataFrame(dfCoeffs1)
    adf.register_subframe("CalibBias1", adfCalibBias1, index_columns=gbVars)
    # ── Add per-variable aliases ──────────────────────────────────────────
    logger.log("calibBiasResolution::Step1 - Registering aliases BEGIN")
    varList1=[]
    for var in varList:
        ic = f"CalibBias1.{var}_intercept{suffix}"
        sc = f"CalibBias1.{var}_slope_qpt{suffix}"
        adf.add_alias(f"{var}_bias",      ic,                   dtype=np.float16, fill_value=0)
        adf.add_alias(f"{var}_slope_qpt", sc,                   dtype=np.float16, fill_value=0)
        adf.add_alias(f"{var}_predicted0", f"{ic} + {sc} * qpt", dtype=np.float16, fill_value=0)
        adf.add_alias(f"{var}_predictedD0", f"{var}-({ic} + {sc} * qpt)", dtype=np.float16, fill_value=0)
        varList1+=[f"{var}_predictedD0"]
    logger.log("calibBiasResolution::Step1 - Registering aliases END")
    # adf.export_tree("adf_snapshot0.root", "snaphsot0")
    # adfCalibBias0.export_tree("adfCalibBias0.root", "calibBias0")
    #
    # Step2: make linear fit of the var-<var> as function of tgl , register in adf as subframe and aliases
    #
    logger.log("calibBiasResolution::Step2 - Registering aliases END")
    gbVars2= ["dsector_bin20", "qpt_bin10"]
    linearColumns2="tgl"
    combined_selection = selection & subsample
    adf.materialize_aliases(names=varList1+['dsector_bin20', 'qpt_bin10'])
    _, dfCoeffs2 = make_parallel_fit_v4(
        df=adf.df,
        gb_columns=gbVars2,           # ['dsector_bin20', 'qpt_bin10']
        fit_columns=varList1,         # DCA variables
        linear_columns=linearColumns2,       # ["tgl"]
        fit_intercept=True,
        suffix="",               # "_CalibBias"
        min_stat=20,
        selection=combined_selection,
        addPrediction=False,
    )
    # ── Register coefficient table as subframe ────────────────────────────
    adfCalibBias2=AliasDataFrame(dfCoeffs2)
    adf.register_subframe("CalibBias2", adfCalibBias2, index_columns=gbVars2)
    varList2=[]
    for var in varList:
        ic = f"CalibBias2.{var}_predictedD0_intercept{suffix}"
        sc = f"CalibBias2.{var}_predictedD0_slope_tgl{suffix}"
        adf.add_alias(f"{var}_bias",      ic,                   dtype=np.float16, fill_value=0)
        adf.add_alias(f"{var}_slope_tgl", sc,                   dtype=np.float16, fill_value=0)
        adf.add_alias(f"{var}_predicted2", f"{ic} + {sc} * tgl", dtype=np.float16, fill_value=0)
        adf.add_alias(f"{var}_predictedD2", f"{var}-({ic} + {sc} * tgl)", dtype=np.float16, fill_value=0)
        varList2+=[f"{var}_predictedD2"]

def addGBBins(adf):
    #
    gb_columns=["sector_bin180","tgl_bin10","qpt_bin5"]
    vars=["dcar_tpc_vertex","dcaz_tpc","dcar_itstpc","dcaz_itstpc","hasITS"]


def drawTestBugRepoduce(adfVertex,adf):
    """
    Ths is the code to repodue bug - will be deleted later.
    :param adfVertex:
    :param adf:
    :return:
    """
    time_fmt = FuncFormatter(lambda x, _: datetime.utcfromtimestamp(x).strftime('%H:%M'))

    # ── Already working (Phase A fix landed) ──
    adfVertex.draw("vertex_x_intercept:vC.vertex_x_intercept_decomp")
    adfVertex.draw("vertex_x_intercept - vC.vertex_x_intercept_decomp", bins=50)
    adfVertex.draw("vertex_x_intercept - vC.vertex_x_intercept_decomp:quantile_bin", type="profile", bins=50, auto_title=True)
    adfVertex.draw("vC.vertex_x_intercept_decomp:vC.vertex_y_intercept_decomp")

    # ── Needs Phase 13.35.ADF (not yet committed) ──
    # Workaround: pre-materialize before draw
    adf.materialize_aliases(['sector', 'time_s'])
    t_mid = adf.df["time_s"].median()
    # 1. Delta: sector 13-14 vs rest OK
    fig, ax, stats = adf.draw(
        "nClITS:time_s",
        selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)",
        type="profile", bins=100,
        selection_vector=["(abs(sector-13)<2)", "(abs(sector-13)>=2)&(sector<36)"],
        normalize="delta",
        auto_title=True, min_entries=50,
    )
    for subplot_ax in fig.axes:
        subplot_ax.xaxis.set_major_formatter(time_fmt)
    fig.tight_layout(); plt.draw()
    # 2. Ratio: early vs late in run - ratio is close to 1 -most probable no change in 2 time intervals
    fig, ax, stats = adf.draw(
        "dcar_tpc_vertex:sector",
        selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)",
        type="profile", bins=36,
        selection_vector=[f"time_s < {t_mid}", f"time_s >= {t_mid}"],
        normalize="ratio",
        auto_title=True, min_entries=100,
    )

    # 3. Compression quality: raw vs decompressed
    fig, ax, stats = adfVertex.draw(
        "[vertex_x_intercept, vC.vertex_x_intercept_decomp]:quantile_bin",
        type="profile", bins=50,
        normalize="delta",
        auto_title=True,
    )
    # Expected: delta ≈ 0 everywhere; deviations = compression artifacts






def my_snippet(makeVertex=True):

    #  adf.df.filter(regex="delta.*").columns
    binGB=100
    adf = root_to_adf("time_series_tracks_0.root")
    adf.draw_lazy=True
    apply_meta(adf,df_TimeSeriesAliases)
    apply_meta(adf,df_TimeSeriesMeta)
    adf=addTimeQuantiles(adf,varname="timeMS",step=10000,step2=100)
    #
    # 1.) make vertex time series compression
    #
    if makeVertex:
        adfVertex=calibVertex(adf)
        adfVertex.df["quantile_binGB"]=adfVertex.df["quantile_bin"]//(binGB)
        adfVertexC=compressADF(adfVertex, column_array={"vertex_(x|y)_intercept$": "linear"}, gb_columns=["quantile_binGB"],
                               index_columns=["quantile_bin"],subframe_name="GB", dtype=np.int8, precision=5, coding="asinh")
        adfVertexC.draw_lazy=True
        adfVertex.draw_lazy=True
        adfVertex.register_subframe("vC", adfVertexC, index_columns=["quantile_bin"])
    #
    # make delta parameterrization per files
    #
    """
    adfVertex.draw("vertex_x_intercept:vC.vertex_x_intercept_decomp")
    """
    time_fmt = FuncFormatter(lambda x, _: datetime.utcfromtimestamp(x).strftime('%H:%M'))
    logger.log("Step1 my_snippet: read tree")
    return adf

def drawTimeSeries(adf):
    time_fmt = FuncFormatter(lambda x, _: datetime.utcfromtimestamp(x).strftime('%H:%M'))
    #
    # 1.) Example draw time series by sector, with cuts and grouping
    #
    fig, ax, stats = adf.draw("nClITS:time_s",selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)&(abs(sector)-7)<2",type="profile",bins=360,group_by="sector",group_by_bins=15,auto_title=True,min_entries=20)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: datetime.utcfromtimestamp(x).strftime('%H:%M:%S')))
    fig.autofmt_xdate()   # rotates labels diagonally so they don't overlap
    plt.draw()            # refresh in interactive mode

    # 2.) Example facet by sector (9 panels), with cuts and grouping
    fig, ax, stats = adf.draw("nClITS:time_s",selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)",type="profile",bins=360,group_by="sector",
                              group_by_bins=5,auto_title=True,
                              facet_by="sector", facet_by_bins=9,  min_entries=20)
    #
    # 3.
    fig, ax, stats = adf.draw("nClITS:time_s",selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)",type="profile",bins=100,group_by="sector",
                              group_by_bins=5,auto_title=True,
                              facet_by="sector", facet_by_bins=9, quantiles=[0.1,0.5,0.9], min_entries=100)
    for subplot_ax in fig.axes: subplot_ax.xaxis.set_major_formatter(time_fmt)
    subplot_ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    plt.draw()
    #
    fig, ax, stats = adf.draw("nClITS:sector",selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)",type="profile",bins=100,auto_title=True, group_by="time_s",
                              group_by_bins=2, quantiles=[0.1,0.5, 0.9], facet_by="time_s",facet_by_bins=9, min_entries=200)
    fig, ax, stats = adf.draw("nClITS:sector",selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)",type="profile",bins=100,auto_title=True, group_by="time_s",
                                group_by_bins=2, quantiles=[0.1,0.5, 0.9], facet_by="time_s",facet_by_bins=9, min_entries=200)


def drawNclExampleFacet(adf,pdf=None):
    """NCL vs tgl, basic + vertex_z faceted."""

    """
    example profile plot  - trobleshooting ncl dependence of tgl distribution  -tracks crossing CE
    """
    fig,_,_ = adf.draw("ncl:tgl", selection="(abs(qpt)<2.5)&(abs(tgl)<0.8)", type="profile", min_entries=100, auto_title=True,
             quantiles=[0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9], quantile_mode='discrete')
    if pdf is not None: pdf.savefig(fig, bbox_inches='tight');  plt.close(fig)
    """
    Draw as function of vertex_z to check if the ncl vs tgl dependence is related to tracks crossing the CE (vertex_z~0) and if the effect is symmetric in z.
    """
    fig,_,_ =adf.draw("ncl:tgl", selection="(abs(qpt)<2.5)&(abs(tgl)<1.4)&(abs(vertex_z)<12)", type="profile", min_entries=50, auto_title=True,bins=100,
             quantiles=[0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9], facet_by="vertex_z", facet_by_bins=9, quantile_mode='discrete')
    if pdf is not None: pdf.savefig(fig, bbox_inches='tight');  plt.close(fig)


def drawFitExample(adf,pdf):
    # run and save figures as an example figure
    set_style({'fit.text_fontsize_facet': 6})
    # Linear fit qpt bias
    fig, ax, stats = adf.draw( "deltaPar4:qpt_ITSTPC", selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)&(abs(qpt)<2)&abs(tgl)<1.4",
        type="profile", bins=80, group_by="tgl", group_by_quantiles=3, facet_by="dsector", facet_by_bins=9,auto_title=True, min_entries=20, range=(-4,4),fit="linear")
    if pdf is not None: pdf.savefig(fig, bbox_inches='tight');  plt.close(fig)
    #
    # Gaussian fit ncl distribution good far from edges
    fig, ax, stats=adf.draw( "ncl", selection="(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)&(abs(qpt)<2)&abs(tgl)<1.4",
              type="hist", bins=92, group_by="abs(tgl)", group_by_quantiles=6, facet_by="dsector", facet_by_bins=9,auto_title=True, min_entries=20, range=(60,152) ,fit="gaus")
    if pdf is not None: pdf.savefig(fig, bbox_inches='tight');  plt.close(fig)


def makePlots(output_path="time_series_plots.pdf"):
    pdf=PdfPages(output_path)
    adf=my_snippet()
    #drawTimeSeries(adf)
    drawNclExampleFacet(adf,pdf)
    drawFitExample(adf,pdf)
    pdf.close()

def loadADFLazy():
    adf = AliasDataFrame.read_tree_lazy("time_series_tracks_0.root","treeTimeSeries")
    apply_meta(adf,df_TimeSeriesAliases)
    apply_meta(adf,df_TimeSeriesMeta)
    adf.draw_lazy=True
    with uproot.open("time_series_tracks_0.root") as f:
        tree = f["treeTimeSeries"]   # adjust if needed
        mask = tree["clusterMask"].array(library="ak")
        res_3_5 = make_row_group_mask(mask, group_size=5, threshold=2)
        adf.df[f"rowmask_3_5"] = res_3_5["rowmask"]
        adf.df[f"first_3_5"] = res_3_5["first"].astype(np.uint8)
        adf.df[f"last_3_5"] = res_3_5["last"].astype(np.uint8)



#
# make plots if args[0] == "plot":

if __name__ == "__main__":
    makePlots()
