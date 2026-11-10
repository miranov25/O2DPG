"""Gallery invariance test for the time-series real-data gallery — declarative form.

HOW TO READ THIS FILE (navigation note, ~2 minutes)

  * Start at CASES at the bottom.  Each case names the gallery function that draws the page
    (time_series_draw.py) and DECLARES what must be true about that page:
      aliases   formulas that ADF's materialized aliases must equal exactly          (L1)
      expect    Profile(...):   per-bin count and (weighted) mean of a plotted profile   (L2)
                Delta(...):     per-bin difference or ratio of two profiles (normalize) (L2)
                RowCounts:      rows a selection keeps in each facet                   (L2)
                SameAcrossSurfaces: draw = draw_batch = draw_figures, value by value   (L3)
                SameAsCall:     a simpler public call gives the same numbers            (L3)
                SameStyleAcrossFacets: each vector branch keeps its style in every facet (L3)
                Flag:           a property the gallery function reports must be True  (L3)
                Histogram(...): per-bin counts or weighted sums of a plotted histogram   (L2)
                Branches(n):    exactly n vector branches were plotted                  (L2)
                FacetPanels:    the facet panels, found by their titles, each once     (L2)
                GaussFit(...):  centre and width of a plotted Gaussian fit             (L2)
                FacetQuantiles: each panel's quantile lines vs np.percentile per cell   (L2)
                Legend / Headings: the figure's shared legend and outer facet headings  (L3)
                GalleryPages:   the 59-page gallery itself (the visual case)           (L3)
                Refuses(...):   a call that must fail loudly with a named message      (L3)
      derived   truth columns computed here that ADF never materializes (no L1 step)
      fixture   a check of the input rows themselves                                  (L0)
  * Every name inside a declaration refers to the TRUTH FRAME: raw source columns plus the
    formulas written in this file — never ADF's alias values, never a dfdraw result.
  * The truth is computed by one small engine (truth_frame, truth_profile) with pandas
    query / cut / groupby — an independent path from dfdraw's own binning.
  * Failure owners (OWNER): L0 input data, L1 ADF alias, L2 the plotted result (ADF -> dfdraw,
    not isolated between the two), L3 a public-call contract, H this test itself.
  * A case passes only if at least one gating comparison ran and all of them passed.
  * run_gallery.sh is the entry point that calls main(); this file does no orchestration beyond
    running the requested TARGETS on ONE loaded data set:
        gallery   the readable 59-page gallery, time_series_draw.run_all_pdf, on the same loaded ADF
        13_105    the declared cases of this file
    NOT_YET_MIGRATED lists the historical case IDs still owned by the old harness; the audit in
    run_gallery.sh checks that every old case is either PASS here or on that list, never both.
    HISTORICAL_GATES lists every gating assertion of the old harness for a migrated case with its old
    tolerance; each run checks that a new comparison carries it at that tolerance or stricter.
  * After each case the PDF gets an EVIDENCE page: verdict, owner, and every comparison with its
    worst difference and tolerance, so a reviewer sees the plot and the numbers side by side.

Command line (called by run_gallery.sh; the data is loaded once for all targets):
    python time_series_draw_invariancetest.py time_series_tracks_0.root --bounded-entries 1000000 \
        --target gallery --target 13_105 --input-identity input_identity_first1M.json \
        --manifest out.json --pdf out.pdf --gallery-pdf gallery.pdf --strict
    python time_series_draw_invariancetest.py --target '?'      list the targets
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import glob
import hashlib
import inspect
import json
import os
import platform
import re
import sys
import time
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

import time_series_draw as G          # the gallery: every adf.draw() of a page lives there

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
RUN_CONTEXT: dict = {}                       # results of other targets of the same run (the gallery), set by main()
OWNER = {"L0": "INPUT DATA", "L1": "ADF ALIAS", "L2": "PLOTTED RESULT (ADF->dfdraw, not isolated)",
         "L3": "PUBLIC CALL CONTRACT", "H": "TEST HARNESS"}


# =============================================================================
# 1. Tolerances — chosen by the precision of the two numbers being compared (D12)
# =============================================================================
@dataclasses.dataclass(frozen=True)
class Tol:
    name: str
    atol: float
    rtol: float


EXACT = Tol("exact", 0.0, 0.0)
SAME_VALUES = Tol("same float64 values, different summation", 1e-9, 1e-8)
# float64 truth vs a float32 alias chain: |value| < 16 in the selection, so one float32 rounding is
# at most 2**-21 = 4.8e-7 per row, and so is any bin mean of such errors; 1e-6 is a hard bound (D12).
MIXED_PRECISION = Tol("float64 truth vs float32 alias chain (D12)", 1e-6, 0.0)
WEIGHTED = Tol("independent float64 weighted accumulation", 1e-8, 1e-7)            # historical weights case
HIST_WEIGHTED = Tol("weighted histogram sums, float64", 1e-8, 1e-10)               # historical M1 case
F16_MEAN = Tol("float16 source summation", 3e-5, 2e-4)                             # historical ADF101 cases
F16_QUANTILE = Tol("float16 source, linear quantile interpolation", 3e-4, 2e-4)
F16_EDGE = Tol("float16 x expression promotion", 2e-3, 0.0)
HIST12 = Tol("same float64 values, historical 1e-12", 1e-12, 1e-12)               # o5, C2, C3 signal
# bin centres are (low + high) / 2 of the declared edges: the historical C2/C3/O4 gate keeps the absolute
# part at 1e-14, so a centre of 0.0 that moved by 5e-13 is a failure, not a rounding difference.
BIN_CENTRES = Tol("bin centres of the declared edges, historical C2/C3/O4", 1e-14, 1e-12)
O4 = Tol("same request, historical o4", 1e-12, 1e-10)
C3_RATIO = Tol("ratio of float64 means, historical C3", 2e-13, 2e-13)
C3_REFERENCE = Tol("float64 reference means, historical C3", 2e-12, 1e-12)        # the signal side keeps HIST12
O2_VALUE = Tol("weighted sums/mean/std/sem, historical o2", 1e-7, 5e-6)
FIT10 = Tol("same fit from two public calls, historical o2", 1e-10, 1e-10)
# golden noise values come from an independent implementation; libm cos/sqrt may differ by 1 ulp.
GOLDEN = Tol("independent implementation, last-ulp libm differences", 1e-15, 0.0)


# =============================================================================
# 2. The truth — selections, binning, formulas and golden values, written here once
# =============================================================================
SELECTED = "(ncl > 60) & (abs(dcar_tpc_vertex) < 10)"            # BASE_SEL of the gallery
SIDES_A_C = SELECTED + " & (side_type < 2)"                     # INJECTED_TRUTH_SIDE_SEL
SECTOR_BINS = dict(x="sector", bins=36, range=(-0.5, 35.5))     # one bin per sector

SECTOR_ALIAS = {"sector": lambda t: 18 * (t.phi / np.pi)}       # the loader's alias (time_series.py)
INJECTED_ALIASES = {   # evaluated in this order; each step may use the previous ones
    **SECTOR_ALIAS,
    "known_delta":          lambda t: 0.08 * np.sin(2 * np.pi * t.sector / 36) + 0.03 * t.tgl + 0.015 * t.tgl * t.tgl,
    "dcar_distorted_clean": lambda t: t.dcar_tpc_vertex + t.known_delta,
    "dcar_oracle":          lambda t: t.dcar_distorted_clean + t.oracle_noise,
    "oracle_residual":      lambda t: t.dcar_oracle - t.dcar_tpc_vertex - t.known_delta,
}
TIME_ALIAS = {"time_s": lambda t: t.timeMS / 1000}                # the loader's alias, float64
CHAIN_ALIASES = {   # the qualified-subframe chain of fig54; the subframe OracleShift holds 0.04 - 0.035*side_type
    **INJECTED_ALIASES,
    "oracle_chain_l1":         lambda t: t.known_delta + 0.005 * t.tgl,
    "oracle_chain_l2":         lambda t: t.oracle_chain_l1 - 0.002 * t.sector / 36,
    "oracle_chain_with_shift": lambda t: t.oracle_chain_l2 + (0.04 - 0.035 * t.side_type.astype(float)),
}
QPT_BIN = {"qpt_bin10": lambda t: (10 * t.qpt).astype(np.int8)}    # the loader's alias (int8 truncation)
QUANTILES = (0.15, 0.50, 0.90)
RATIO_ALIASES = {   # the gallery's C3 pair (time_series_draw.py, PRECLEAN_C3_*)
    **SECTOR_ALIAS,
    "preclean_ratio_base":   lambda t: 1.0 + 0.02 * t.sector + 0.01 * t.tgl * t.tgl,
    "preclean_ratio_scaled": lambda t: 2.0 * t.preclean_ratio_base,
}
O2_WEIGHT_EXPRESSIONS = ("1.0 + 0.0*abs(dcar_tpc_vertex)", "1.0 + abs(dcar_tpc_vertex)")   # fig44's weights_vector
O2_WEIGHTS = {      # the same two weights as truth columns, in the source dtype of dcar_tpc_vertex
    "w_unity": lambda t: 1.0 + 0.0 * t.dcar_tpc_vertex.abs(),
    "w_dca":   lambda t: 1.0 + t.dcar_tpc_vertex.abs(),
}


def early(t):                     # time_s below the median of all rows (the gallery's t_mid)
    return t.time_s < t.time_s.median()


def late(t):
    return t.time_s >= t.time_s.median()
RAW_COLUMNS = ["phi", "tgl", "dcar_tpc_vertex", "dcar_tpc", "ncl", "side_type", "timeMS", "qpt"]   # ROOT branches used
SOURCE_COLUMNS = RAW_COLUMNS + ["oracle_noise", "oracle_row_id", "oracle_w_flat", "oracle_w_tgl"]

NOISE_SIGMA = 0.02                                  # declared in time_series_draw.py (seed 137)
NOISE_GOLDEN = {0: -0.013639289756689786, 1: -0.022394913111732667, 2: 0.032784057514017396,
                17: -0.0019575276417686006, 4242: -0.010185919876131678, 5999: -0.006423739361834429}


# =============================================================================
# 3. Declarations
# =============================================================================
@dataclasses.dataclass(frozen=True)
class Profile:
    """Per-bin count and float64 mean of truth column `y` over `rows`, one table per group of `by`."""
    y: str
    rows: str
    x: str
    bins: int
    range: tuple
    plotted: Callable[[Any], list]           # drawn -> [(group label, plotted profile table), ...]
    by: str | None = None
    by_bins: int | None = None               # None: categorical groups; N: N equal-width pd.cut bins
    weights: str | None = None               # truth column of per-row weights; None = unweighted
    tol: Tol = SAME_VALUES
    edges_tol: Tol | None = None             # also compare the plotted bin edges (x_low, x_high)
    spread: bool = False                     # also compare weighted y_std (population) and y_sem = std/sqrt(n_eff)


@dataclasses.dataclass(frozen=True)
class Delta:
    """Per bin: mean(signal) - mean(reference) (kind "delta") or mean(signal)/mean(reference) (kind "ratio"),
    which bins are undefined, and where the table exposes them, both central values and counts."""
    signal: Profile
    reference: Profile
    plotted: Callable[[Any], list]           # drawn -> [(group label, normalize table), ...]
    tol: Tol = SAME_VALUES                   # the plotted difference or ratio
    kind: str = "delta"
    signal_tol: Tol | None = None            # signal central values (and plotted signal line); None = tol
    reference_tol: Tol | None = None         # reference central values (and plotted reference line); None = tol
    centres_tol: Tol = BIN_CENTRES           # x_center against the declared edges
    plotted_lines: Callable[[Any], dict] | None = None   # drawn -> {group: [(x, y) signal line, (x, y) reference line]}


@dataclasses.dataclass(frozen=True)
class RowCounts:
    """Rows that `rows` keeps in each group of `by`, with finite x, y and x inside `range` (the public n)."""
    rows: Any
    by: str
    x: str
    y: str
    range: tuple
    plotted: Callable[[Any], dict]           # drawn -> {group label: plotted n}


@dataclasses.dataclass(frozen=True)
class SameAcrossSurfaces:
    """The same request through draw, draw_batch and draw_figures gives the same table, column by column,
    each column with its own tolerance (counts and masks exact)."""
    columns: dict                            # {column: Tol}
    plotted: Callable[[Any], dict]           # drawn -> {surface: table}


@dataclasses.dataclass(frozen=True)
class SameAsCall:
    """A second, simpler public call must give the same numbers as the plotted one, e.g. each weights_vector
    branch equals a draw with that single weight (composition invariant, L3)."""
    name: str
    call: Callable[[Any], Any]               # adf -> result of the simpler call
    pairs: Callable[[Any, Any], list]        # (plotted, simpler) -> [(label, {name: plotted values}, {name: simpler})]
    tol: Tol = HIST12
    tols: dict = dataclasses.field(default_factory=dict)   # per-value tolerance where it differs from tol


@dataclasses.dataclass(frozen=True)
class SameStyleAcrossFacets:
    """In every facet panel (title 'side_type=N') the first n profile lines (circle marker, visible line) have the
    same colour, line style and marker as in the first facet, branch by branch (L3)."""
    n_branches: int
    n_facets: int = 2


@dataclasses.dataclass(frozen=True)
class Flag:
    """A property the gallery function measured about the public call must be True (e.g. caller spec unchanged)."""
    name: str
    plotted: Callable[[Any], Any]


@dataclasses.dataclass(frozen=True)
class Histogram:
    """Per-bin counts (or weighted sums) of truth column `x` over `rows`, as np.histogram computes them."""
    x: str
    rows: Any                                # pandas query string, or a function of the truth frame
    bins: int
    range: tuple
    plotted: Callable[[Any], Any]            # drawn -> the plotted bar heights
    weights: str | None = None
    tol: Tol = EXACT
    label: str = ""                          # which plotted series, e.g. "side_type=1 early"


@dataclasses.dataclass(frozen=True)
class Branches:
    """The plotted result has exactly `n` vector branches; an extra or a dropped branch is a failure (L2)."""
    n: int


@dataclasses.dataclass(frozen=True)
class FacetPanels:
    """The panels titled 'side_type=N' show exactly the facets `want`, each once. Panels are found by their
    title, never by their position (L2)."""
    want: tuple
    axes: Callable[[Any], Any]               # drawn -> the facet axes


@dataclasses.dataclass(frozen=True)
class GaussFit:
    """The plotted Gaussian fit's centre and width must match mean and std of the truth sample (historical rule)."""
    y: str
    rows: str
    range: tuple
    bins: int
    plotted: Callable[[Any], dict]           # drawn -> the fit record (fit_name, params, param_names, ...)


@dataclasses.dataclass(frozen=True)
class FacetQuantiles:
    """In every facet panel, each quantile line (per y, per quantile, per group) equals np.percentile of that cell."""
    ys: tuple                                # truth columns, in branch order
    rows: str
    facets: tuple                            # ((column, number of pd.cut bins), ...); panels in product order
    x: str = "sector"
    bins: int = 36
    range: tuple = (0.0, 36.0)
    by: str | None = None                    # global group column; groups 0 and 1
    tol: Tol = F16_QUANTILE


@dataclasses.dataclass(frozen=True)
class Legend:
    """The figure has one shared legend showing each of `labels` exactly once ("name vs sector" counts as name)."""
    labels: tuple


@dataclasses.dataclass(frozen=True)
class Headings:
    """The figure shows `n` distinct outer facet headings (texts that start with "(" or "[")."""
    n: int


@dataclasses.dataclass(frozen=True)
class GalleryPages:
    """The readable gallery of the same run (target `gallery`) reported success and wrote `pages` pages."""
    pages: int


@dataclasses.dataclass(frozen=True)
class Refuses:
    """`call(adf)` must raise with `message` in its text (L3)."""
    call: Callable[[Any], Any]
    message: str


@dataclasses.dataclass(frozen=True)
class Case:
    case_id: str
    draw: Callable[[Any], Any] | None        # gallery function; None for machine-only contracts
    title: str
    aliases: dict = dataclasses.field(default_factory=dict)
    expect: tuple = ()
    fixture: Callable[[Any, pd.DataFrame], None] | None = None
    derived: dict = dataclasses.field(default_factory=dict)


def facets(stats) -> list:
    """Plotted tables of a faceted profile: [(facet value, profile_data), ...]."""
    return [(g, stats["per_group"][str(g)]["profile_data"]) for g in stats.get("groups", ())]


def single(stats) -> list:
    """The plotted table of an ungrouped profile, as one group with label 0."""
    return [(0, stats["profile_data"])]


def branch(i: int, extract=single):
    """Plotted tables of vector branch i (one result per branch of a bracket vector or selection/weights vector)."""
    return lambda drawn: extract(drawn[2][i])


def first_fit(stats) -> dict:
    """The first fit record of a plotted result (dfdraw nests it in lists)."""
    record = stats.get("fit")
    while isinstance(record, (list, tuple)) and record:
        record = record[0]
    return record if isinstance(record, dict) else {}


def titled_facets(axes) -> list:
    """[(facet value, axes)] of every panel titled 'side_type=N', in drawing order; duplicates are kept."""
    found = []
    for ax in np.asarray(axes, dtype=object).reshape(-1):
        match = re.search(r"side_type=([^\s]+)", ax.get_title()) if ax is not None else None
        if match is not None:
            found.append((int(float(match.group(1))), ax))
    return found


def facet_panel(axes, value: int):
    """The one panel titled 'side_type=<value>'; none or two of them is a failure of what was plotted (L2)."""
    found = [ax for g, ax in titled_facets(axes) if g == value]
    if len(found) != 1:
        raise CheckFailure("L2", f"{len(found)} panels titled side_type={value}")
    return found[0]


def facet_panels(drawn) -> dict:
    """{facet value: axes} for the panels titled 'side_type=N' (upper panels of a faceted normalize figure)."""
    return dict(titled_facets(drawn[0].axes))


def profile_lines(ax) -> list:
    """The plotted profile lines of a panel, in drawing order: circle marker and a visible line."""
    return [ln for ln in ax.lines if str(ln.get_marker()) == "o" and str(ln.get_linestyle()).lower() not in ("none", "")]


def overlay_lines(drawn) -> dict:
    """{facet: [(x, y) of each overlay line]} — signal first, reference second, as dfdraw draws them."""
    return {g: [(np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float)) for ln in profile_lines(ax)]
            for g, ax in facet_panels(drawn).items()}


def facet_fit_parameters(stats, g) -> np.ndarray:
    """Parameters of the fit drawn in facet g of one branch (dfdraw nests the record as [[record]])."""
    cell = stats["per_group"][str(g)]["fit"]
    while isinstance(cell, (list, tuple)):
        cell = cell[0]
    return np.asarray(cell["params"], float)


def scalar_weight_draw(expression: str):
    """fig44's request with ONE weight instead of the weights_vector (the old harness's scalar decomposition)."""
    return lambda adf: adf.draw("dcar_tpc_vertex:tgl", selection=SIDES_A_C, type="profile", bins=30, range=(-1.5, 1.5),
                                weights=expression, facet_by="side_type", fit="pol1", min_entries=1, auto_title=False,
                                return_data=True)


def branch_vs_scalar(i: int):
    """Pairs (facet, vector branch i, scalar draw): profile columns and pol1 parameters of each side_type facet."""
    columns = ("x_center", "count", "sum_weights", "y_mean", "y_std", "y_sem")

    def values(stats, g):
        table = stats["per_group"][str(g)]["profile_data"]
        return {**{c: table[c].to_numpy() for c in columns}, "pol1 parameters": facet_fit_parameters(stats, g)}
    return lambda drawn, scalar: [(f"weights branch {i} side_type={g}", values(drawn[2][i], g), values(scalar[2], g))
                                  for g in (0, 1)]


def bar_heights(ax, series: int, n_bins: int) -> np.ndarray:
    """Heights of one plotted step histogram: vertex 2i+1 of the filled step polygon carries bin i's height."""
    return np.asarray(ax.patches[series].get_xy()[1:2 * n_bins:2, 1], dtype=float)


def groups(stats) -> list:
    """Plotted tables of a grouped profile: [(group label, rows of profile_data), ...] in plotted order."""
    frame = stats["profile_data"]
    return [(g, frame[frame["group"] == g]) for g in pd.unique(frame["group"])]


def paired_x_is_sensitive(result, t: pd.DataFrame) -> None:
    """L0 for ADF100-01: binning by `sector` instead of `sector+0.25` must change the counts,
    otherwise the case could not detect a swapped X."""
    r = t.query(SELECTED)
    sector = SECTOR_ALIAS["sector"](r).astype(np.float16)            # the input stage: sector from raw phi
    shifted = (sector + 0.25).astype(float)
    edges = np.linspace(shifted.min(), shifted.max(), 21)
    right = np.histogram(shifted, edges)[0]
    wrong = np.histogram(sector.astype(float), edges)[0]
    compare(result, "swapped-X counts differ (fixture sensitivity)", int(not np.array_equal(right, wrong)), 1,
            EXACT, "L0")


def injected_noise_is_declared(result, t: pd.DataFrame) -> None:
    """L0: noise golden values and mean/std within 5 standard errors; weight columns equal their declared formulas."""
    tgl = t["tgl"].to_numpy(float)
    compare(result, "oracle_w_flat == 1", t["oracle_w_flat"], np.ones(len(t)), EXACT, "L0")
    compare(result, "oracle_w_tgl == 1 + 0.30*|tgl|", t["oracle_w_tgl"], 1.0 + 0.30 * np.abs(tgl), EXACT, "L0")
    ids = [i for i in NOISE_GOLDEN if i in set(t["oracle_row_id"])]
    if len(ids) < 3:
        raise CheckFailure("L0", f"only {len(ids)} golden row ids present")
    got = t.set_index("oracle_row_id").loc[ids, "oracle_noise"]
    compare(result, "noise at golden row ids", got, [NOISE_GOLDEN[i] for i in ids], GOLDEN, "L0")
    noise, n = t["oracle_noise"].to_numpy(float), len(t)
    compare(result, "noise mean", noise.mean(), 0.0, Tol("5 standard errors", 5 * NOISE_SIGMA / n ** 0.5, 0), "L0")
    compare(result, "noise std", noise.std(), NOISE_SIGMA, Tol("5 standard errors", 5 * NOISE_SIGMA / (2 * n) ** 0.5, 0), "L0")


# =============================================================================
# 4. The engine — the only machinery in this file
# =============================================================================
class CheckFailure(AssertionError):
    def __init__(self, level: str, message: str):
        super().__init__(f"{level}/{OWNER[level]}: {message}")
        self.level = level


@dataclasses.dataclass
class CaseResult:
    case_id: str
    status: str = SKIP
    failure_level: str = ""
    comparisons: list = dataclasses.field(default_factory=list)
    detail: str = ""
    page: int | None = None
    wall_time_s: float = 0.0


def compare(result: CaseResult, observable: str, got, want, tol: Tol, level: str,
            diagnostic: bool = False) -> None:
    """Record one numeric comparison (worst |diff|, where, NaN mismatches); raise unless it passes or is diagnostic."""
    a, b = np.atleast_1d(np.asarray(got, dtype=float)), np.atleast_1d(np.asarray(want, dtype=float))
    if a.shape != b.shape:
        raise CheckFailure(level, f"{observable}: shape {a.shape} != {b.shape}")
    both_finite = np.isfinite(a) & np.isfinite(b)
    special = np.flatnonzero(~both_finite & ~(np.isnan(a) & np.isnan(b)) & ~(a == b))   # NaN vs number, inf vs finite, +inf vs -inf
    diff = np.where(both_finite, np.abs(a - b), -1.0)
    worst = float(diff.max()) if diff.size and diff.max() >= 0 else 0.0
    where = str(int(special[0])) if special.size else (str(int(diff.argmax())) if diff.size else "")
    ok = (np.array_equal(a, b, equal_nan=True) if tol is EXACT
          else bool(np.allclose(a, b, atol=tol.atol, rtol=tol.rtol, equal_nan=True)))
    result.comparisons.append(dict(observable=observable, tolerance=tol.name, atol=tol.atol, rtol=tol.rtol,
                                   ok=None if diagnostic else bool(ok), worst_abs=worst, where=where,
                                   nonfinite_mismatches=int(special.size), level=level))
    if not ok and not diagnostic:
        raise CheckFailure(level, f"{observable}: worst |diff| {worst:.3e}, {special.size} NaN/inf mismatch(es), "
                                  f"first at [{where}] ({tol.name})")


def compare_labels(result: CaseResult, observable: str, got: list, want: list, level: str) -> None:
    """Record one exact comparison of group labels, in order."""
    ok = [str(g) for g in got] == [str(w) for w in want]
    result.comparisons.append(dict(observable=observable, tolerance="exact labels", atol=0.0, rtol=0.0, ok=ok,
                                   worst_abs=0.0 if ok else 1.0, where="" if ok else f"{list(got)}",
                                   nonfinite_mismatches=0, level=level))
    if not ok:
        raise CheckFailure(level, f"{observable}: plotted {list(got)} != truth {list(want)}")


def truth_frame(adf, case: Case, result: CaseResult) -> pd.DataFrame:
    """Raw source columns + this file's formulas; each alias step cast to ADF's materialized dtype and checked (L1)."""
    t = adf.df[[c for c in SOURCE_COLUMNS if c in adf.df.columns]].copy()
    if case.fixture is not None:                         # L0 first: the input rows themselves
        case.fixture(result, t)
    for name, formula in case.aliases.items():
        if name not in adf.df.columns:
            raise CheckFailure("L1", f"alias {name!r} was not materialized")
        t[name] = np.asarray(formula(t)).astype(adf.df[name].dtype, copy=False)   # only the dtype comes from ADF
        compare(result, f"alias {name} == formula", adf.df[name], t[name], EXACT, "L1")
    for name, formula in case.derived.items():
        t[name] = formula(t)
    return t


def select(t: pd.DataFrame, rows) -> pd.DataFrame:
    """Rows of the truth frame: `rows` is a pandas query string or a function returning a boolean mask."""
    return t.query(rows) if isinstance(rows, str) else t[rows(t)]


def profile_edges(x: pd.Series, y: pd.Series, spec) -> np.ndarray:
    """Bin edges: the declared range, or "minmax" = min and max of the finite selected (x, y) rows."""
    if spec.range != "minmax":
        return np.linspace(*spec.range, spec.bins + 1)
    finite = np.isfinite(x) & np.isfinite(y)
    return np.linspace(x[finite].min(), x[finite].max(), spec.bins + 1)


def truth_profile(t: pd.DataFrame, spec: Profile) -> dict:
    """{group: (count, sum of weights, weighted float64 mean) per bin} by pandas cut/groupby; bins [a, b), last closed."""
    r = select(t, spec.rows)
    if spec.by_bins:
        r = r[np.isfinite(r[spec.by].astype(float))]
        cut = pd.cut(r[spec.by].astype(float), spec.by_bins)
        names = [f"{iv.left:.2f}-{iv.right:.2f}" for iv in cut.cat.categories]   # dfdraw's group label format
        key = pd.Series(np.asarray(names, dtype=object)[cut.cat.codes], index=r.index)
    else:
        key = r[spec.by].astype(int) if spec.by else pd.Series(0, index=r.index)
    x, y = r[spec.x].astype(float), r[spec.y].astype(float)
    w = r[spec.weights].astype(float) if spec.weights else pd.Series(1.0, index=r.index)
    edges = profile_edges(x, y, spec)
    b = pd.cut(x, edges, right=False, labels=False)
    b[x == edges[-1]] = spec.bins - 1
    keep = b.notna() & np.isfinite(y) & np.isfinite(w)
    if not keep.any():
        raise CheckFailure("L0", f"selection {spec.rows!r} keeps no rows for {spec.y}")
    frame = pd.DataFrame({"g": key, "b": b, "w": w, "y": y})[keep]
    frame["wy"], frame["ww"] = frame.w * frame.y, frame.w * frame.w
    cell = frame.groupby(["g", "b"])
    frame["dev2"] = frame.w * (frame.y - cell.wy.transform("sum") / cell.w.transform("sum")) ** 2   # two-pass
    table = frame.groupby(["g", frame.b.astype(int)]).agg(
        count=("w", "size"), sum_w=("w", "sum"), sum_wy=("wy", "sum"), sum_ww=("ww", "sum"), dev2=("dev2", "sum"))
    labels = names if spec.by_bins else sorted(frame.g.unique())
    table = table.reindex(pd.MultiIndex.from_product([labels, range(spec.bins)])).fillna({"count": 0})
    filled = table["count"] > 0                    # empty bins: count 0, every other quantity NaN
    table["mean"] = table["sum_wy"] / table["sum_w"].where(filled)
    table["std"] = np.sqrt(table["dev2"] / table["sum_w"]).where(table["count"] > 1)
    table["sem"] = table["std"] / np.sqrt(table["sum_w"] ** 2 / table["sum_ww"])
    return {g: (table.loc[g, "count"].to_numpy(int), table.loc[g, "sum_w"].to_numpy(float),
                table.loc[g, "mean"].to_numpy(float), table.loc[g, "std"].to_numpy(float),
                table.loc[g, "sem"].to_numpy(float)) for g in labels}


def check_profile(t: pd.DataFrame, spec: Profile, drawn, result: CaseResult) -> None:
    """Compare the plotted tables with the truth: group labels in order, then count and mean per group (L2)."""
    truth = truth_profile(t, spec)
    plotted = spec.plotted(drawn)
    labels = [str(g) if spec.by_bins else int(float(g)) for g, _ in plotted]
    compare_labels(result, f"{spec.y} groups of {spec.by or 'the whole selection'}", labels, list(truth), "L2")
    for label, (_, table) in zip(labels, plotted):
        count, sum_w, mean, std, sem = truth[label]
        tag = spec.y + (f" {spec.by}={label}" if spec.by else "") + (f" weights={spec.weights}" if spec.weights else "")
        compare(result, f"{tag} count", table["count"], count, EXACT, "L2")
        if spec.weights:
            compare(result, f"{tag} sum of weights", table["sum_weights"], sum_w, spec.tol, "L2")
        compare(result, f"{tag} mean", table["y_mean"], mean, spec.tol, "L2")
        if spec.spread:
            compare(result, f"{tag} std", table["y_std"], std, spec.tol, "L2")
            compare(result, f"{tag} sem", table["y_sem"], sem, spec.tol, "L2")
        if spec.edges_tol is not None:
            r = select(t, spec.rows)
            edges = profile_edges(r[spec.x].astype(float), r[spec.y].astype(float), spec)
            got_edges = np.r_[np.asarray(table["x_low"], float), np.asarray(table["x_high"], float)[-1]]
            compare(result, f"{tag} bin edges", got_edges, edges, spec.edges_tol, "L2")
            compare(result, f"{tag} bin centres", table["x_center"], (edges[1:] + edges[:-1]) / 2, spec.edges_tol, "L2")


def check_delta(t: pd.DataFrame, spec: Delta, drawn, result: CaseResult) -> None:
    """Truth per bin: signal-reference (or signal/reference) of the two independent profiles; undefined where a
    side is empty (or, for a ratio, the reference mean is 0). Centrals, counts and centres where exposed (L2)."""
    sig, ref = truth_profile(t, spec.signal), truth_profile(t, spec.reference)
    plotted = spec.plotted(drawn)
    labels = [int(float(g)) for g, _ in plotted]
    compare_labels(result, f"{spec.kind} groups of {spec.signal.by or 'the whole selection'}", labels,
                   sorted(set(sig) | set(ref)), "L2")
    empty = (np.zeros(spec.signal.bins, int), None, np.full(spec.signal.bins, np.nan))
    edges = np.linspace(*spec.signal.range, spec.signal.bins + 1)
    for label, (_, table) in zip(labels, plotted):
        (n_sig, _, m_sig), (n_ref, _, m_ref) = sig.get(label, empty)[:3], ref.get(label, empty)[:3]
        undefined = (n_sig == 0) | (n_ref == 0)
        want = m_sig - m_ref if spec.kind == "delta" else m_sig / np.where(m_ref == 0, np.nan, m_ref)
        undefined |= ~np.isfinite(want)
        tag = f"{spec.kind} {spec.signal.by}={label}" if spec.signal.by else spec.kind
        compare(result, f"{tag} undefined bins", np.asarray(table["mask_undefined"], bool), undefined, EXACT, "L2")
        values = table["value"] if "value" in table else table["values"]
        compare(result, f"{tag} values", np.where(undefined, np.nan, np.asarray(values, float)),
                np.where(undefined, np.nan, want), spec.tol, "L2")
        centres = table["x_center"] if "x_center" in table else table.get("bin_centers")
        if centres is not None:
            compare(result, f"{tag} bin centres", centres, (edges[1:] + edges[:-1]) / 2, spec.centres_tol, "L2")
        side_tols = (spec.signal_tol or spec.tol, spec.reference_tol or spec.tol)
        for side, n, m, tol in zip(("signal", "reference"), (n_sig, n_ref), (m_sig, m_ref), side_tols):
            if f"{side}_count" in table:
                compare(result, f"{tag} {side} count", table[f"{side}_count"], n, EXACT, "L2")
                compare(result, f"{tag} {side} central", np.where(n > 0, np.asarray(table[f"{side}_central"], float), np.nan),
                        m, tol, "L2")
        if spec.plotted_lines is not None:
            check_overlay_lines(spec.plotted_lines(drawn).get(label, []), edges, m_sig, m_ref, side_tols, tag, result)


def check_overlay_lines(lines: list, edges: np.ndarray, m_sig, m_ref, tols: tuple, tag: str, result: CaseResult) -> None:
    """The two plotted overlay curves, put back on the bin grid (NaN where nothing is drawn), against the truth
    means; a point off the bin centres is a failure of what was plotted (L2)."""
    compare(result, f"{tag} number of overlay lines", len(lines), 2, EXACT, "L2")
    centres = (edges[1:] + edges[:-1]) / 2
    for (x, y), side, want, tol in zip(lines, ("signal", "reference"), (m_sig, m_ref), tols):
        index = np.abs(centres[None, :] - x[:, None]).argmin(axis=1) if x.size else np.zeros(0, int)   # nearest centre
        compare(result, f"{tag} {side} line x on bin centres", x, centres[index], HIST12, "L2")
        on_grid = np.full(len(centres), np.nan)
        on_grid[index] = y
        compare(result, f"{tag} {side} plotted line", on_grid, want, tol, "L2")


def check_row_counts(t: pd.DataFrame, spec: RowCounts, drawn, result: CaseResult) -> None:
    """Per group: rows kept by the selection with finite x, y and x inside the range, against the public n (L2)."""
    r = select(t, spec.rows)
    x, y = r[spec.x].astype(float), r[spec.y].astype(float)
    kept = np.isfinite(x) & np.isfinite(y) & (x >= spec.range[0]) & (x <= spec.range[1])
    want = r[kept].groupby(spec.by).size()
    got = spec.plotted(drawn)
    labels = sorted(int(float(g)) for g in got)
    compare_labels(result, f"groups of {spec.by}", labels, sorted(int(g) for g in want.index), "L2")
    compare(result, f"rows per {spec.by}", [got[str(g)] if str(g) in got else got[g] for g in labels],
            [want[g] for g in labels], EXACT, "L2")


def check_same_across_surfaces(spec: SameAcrossSurfaces, drawn, result: CaseResult) -> None:
    """draw_batch and draw_figures against draw, column by column (L3: the public call contract)."""
    tables = spec.plotted(drawn)
    for surface in ("draw_batch", "draw_figures"):
        for column, tol in spec.columns.items():
            compare(result, f"{surface} {column} == draw", tables[surface][column], tables["draw"][column], tol, "L3")


def check_same_as_call(adf, spec: SameAsCall, drawn, result: CaseResult) -> None:
    """Run the simpler call and compare it value by value with the plotted result (L3)."""
    simpler = spec.call(adf)
    for label, got, want in spec.pairs(drawn, simpler):
        for name in got:
            tol = EXACT if name == "count" else spec.tols.get(name, spec.tol)
            compare(result, f"{spec.name}: {label} {name}", got[name], want[name], tol, "L3")


def check_same_style(spec: SameStyleAcrossFacets, drawn, result: CaseResult) -> None:
    """(colour, line style, marker) of each branch, facet by facet, against the first facet (L3)."""
    styles = {g: [f"{ln.get_color()} {ln.get_linestyle()} {ln.get_marker()}" for ln in profile_lines(ax)[:spec.n_branches]]
              for g, ax in sorted(facet_panels(drawn).items())}
    compare(result, "facets with profile lines", len(styles), spec.n_facets, EXACT, "L3")
    first, *others = sorted(styles)
    for g in others:
        compare_labels(result, f"branch styles side_type={g} == side_type={first}", styles[g], styles[first], "L3")


def check_flag(spec: Flag, drawn, result: CaseResult) -> None:
    compare(result, spec.name, int(bool(spec.plotted(drawn))), 1, EXACT, "L3")


def check_histogram(t: pd.DataFrame, spec: Histogram, drawn, result: CaseResult) -> None:
    """Truth: np.histogram of the selected, finite rows (with weights if declared) (L2)."""
    r = select(t, spec.rows)
    x = r[spec.x].to_numpy(float)
    w = r[spec.weights].to_numpy(float) if spec.weights else None
    keep = np.isfinite(x) & (np.isfinite(w) if w is not None else True)
    if not keep.any():
        raise CheckFailure("L0", f"histogram selection keeps no rows for {spec.x}")
    want = np.histogram(x[keep], bins=np.linspace(*spec.range, spec.bins + 1),
                        weights=None if w is None else w[keep])[0]
    tag = f"histogram {spec.x}" + (f" weights={spec.weights}" if spec.weights else "") + (f" {spec.label}" if spec.label else "")
    compare(result, tag, spec.plotted(drawn), want, spec.tol, "L2")


def check_branches(spec: Branches, drawn, result: CaseResult) -> None:
    compare(result, "number of vector branches", len(drawn[2]), spec.n, EXACT, "L2")


def check_facet_panels(spec: FacetPanels, drawn, result: CaseResult) -> None:
    """Facet values read from the panel titles, sorted; a duplicated or missing title changes the list (L2)."""
    compare_labels(result, "facet panels (by title)", sorted(g for g, _ in titled_facets(spec.axes(drawn))),
                   sorted(spec.want), "L2")


def quantile_label(q: float) -> str:
    return "median" if q == 0.5 else f"q={round(100 * q)}%"


def quantile_line(ax, expr: str, i_expr: int, q: float, i_q: int, group, n_expr: int):
    """The one plotted line for (expression, quantile, group): by dfdraw's semantic tag, or for ungrouped
    panels (which carry no tag) by the branch-major order of the labelled quantile lines."""
    tagged = [ln for ln in ax.lines if (getattr(ln, "_dfdraw_facet_semantic", None) or {}).get("kind") == "quantile"]
    if tagged:
        hits = [ln for ln in tagged if ln._dfdraw_facet_semantic.get("expression") == expr
                and abs(float(ln._dfdraw_facet_semantic.get("quantile", -1)) - q) < 1e-12
                and (group is None or str(ln._dfdraw_facet_semantic.get("group")) == str(group))]
    else:
        names = [quantile_label(v) for v in QUANTILES]
        lines = [ln for ln in ax.lines if str(ln.get_label()) in names]
        if [str(ln.get_label()) for ln in lines] != names * n_expr:
            raise CheckFailure("L3", f"quantile lines out of order: {[str(ln.get_label()) for ln in lines]}")
        hits = [lines[i_expr * len(QUANTILES) + i_q]]
    if len(hits) != 1:
        raise CheckFailure("L3", f"expected one quantile line for {expr} q={q} group={group}, found {len(hits)}")
    return hits[0]


def facet_cells(r: pd.DataFrame, facets: tuple) -> list:
    """Boolean masks of the facet cells, in panel order: pd.cut of each facet column on the selected rows."""
    cuts = [pd.cut(r[column], n) for column, n in facets]
    cells = [np.ones(len(r), bool)]
    for cut in cuts:
        cells = [c & np.asarray(cut == cat) for c in cells for cat in cut.cat.categories]
    return cells


def binned_percentile(x: np.ndarray, y: np.ndarray, edges: np.ndarray, q: float) -> np.ndarray:
    """Linear percentile of y per x bin; bins [a, b), the last closed; NaN where a bin is empty."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    inside = (x >= edges[0]) & (x <= edges[-1])
    out = np.full(len(edges) - 1, np.nan)
    for i in range(len(out)):
        z = y[inside & (b == i)]
        if z.size:
            out[i] = np.percentile(z, 100 * q, method="linear")
    return out


def check_facet_quantiles(t: pd.DataFrame, spec: FacetQuantiles, drawn, result: CaseResult) -> None:
    """np.percentile (linear) per bin of each cell and group, against every plotted quantile line (L2)."""
    r = select(t, spec.rows)
    cells, panels = facet_cells(r, spec.facets), list(np.asarray(drawn[1], dtype=object).flat)
    compare(result, "number of facet panels", len(panels), len(cells), EXACT, "L3")
    edges = np.linspace(*spec.range, spec.bins + 1)
    centers = (edges[1:] + edges[:-1]) / 2
    x = r[spec.x].to_numpy(float)
    for j, cell in enumerate(cells):
        if cell.sum() < spec.bins:
            raise CheckFailure("L0", f"facet cell {j} has only {cell.sum()} rows")
        for group in ((0, 1) if spec.by else (None,)):
            rows = cell & (r[spec.by].to_numpy() == group) if spec.by else cell
            if not rows.any():
                continue
            for i_expr, y_name in enumerate(spec.ys):
                y = r[y_name].to_numpy(float)
                for i_q, q in enumerate(QUANTILES):
                    want = binned_percentile(x[rows], y[rows], edges, q)
                    line = quantile_line(panels[j], y_name, i_expr, q, i_q, group, len(spec.ys))
                    tag = f"cell {j} group {group} {y_name} {quantile_label(q)}"
                    compare(result, f"{tag} x", line.get_xdata(), centers, F16_EDGE, "L2")
                    compare(result, f"{tag} values", line.get_ydata(), want, spec.tol, "L2")


def check_legend(spec: Legend, drawn, result: CaseResult) -> None:
    """One figure legend; each declared label exactly once, nothing else (L3)."""
    fig = drawn[0]
    labels = [t.get_text().split(" vs ", 1)[0] for lg in fig.legends for t in lg.get_texts()]
    compare(result, "number of figure legends", len(fig.legends), 1, EXACT, "L3")
    compare_labels(result, "legend identities", sorted(labels), sorted(spec.labels), "L3")


def check_headings(spec: Headings, drawn, result: CaseResult) -> None:
    """Distinct outer facet headings on the figure (L3)."""
    texts = [t.get_text() for t in drawn[0].texts]
    headings = [h for h in texts if h.startswith(("(", "[")) and "'" not in h and '"' not in h]
    compare(result, "distinct outer facet headings", len(set(headings)), spec.n, EXACT, "L3")


def check_gallery_pages(spec: GalleryPages, result: CaseResult) -> None:
    """The gallery target of this run: run_all_pdf succeeded and the PDF has the declared page count (L3)."""
    gallery = RUN_CONTEXT.get("gallery")
    if gallery is None:
        raise CheckFailure("H", "the visual case needs --target gallery in the same run")
    compare(result, "gallery run_all_pdf succeeded", int(bool(gallery["ok"])), 1, EXACT, "L3")
    compare(result, "gallery pages", gallery["pages"], spec.pages, EXACT, "L3")


def check_gauss_fit(t: pd.DataFrame, spec: GaussFit, drawn, result: CaseResult) -> None:
    """Fit centre within max(bin/4, 5 sigma/sqrt(n)) of the sample mean; width within max(bin/4, 5 sigma/sqrt(2(n-1)))."""
    r = select(t, spec.rows)
    y = r[spec.y].to_numpy(float)
    y = y[np.isfinite(y) & (y >= spec.range[0]) & (y <= spec.range[1])]
    if y.size < 100:
        raise CheckFailure("L0", f"Gaussian sample too small: {y.size} rows")
    record = spec.plotted(drawn)
    if not record or record.get("fit_name") != "gauss" or record.get("fit_status", "ok") != "ok":
        raise CheckFailure("L2", f"no valid Gaussian fit record: {str(record)[:120]}")
    fit = dict(zip(record["param_names"], np.asarray(record["params"], float)))
    mean, sigma, n = y.mean(), y.std(), y.size
    bin_width = (spec.range[1] - spec.range[0]) / spec.bins
    compare(result, "Gaussian fit centre", fit["center"], mean,
            Tol("max(bin/4, 5 sigma/sqrt n)", max(0.25 * bin_width, 5 * sigma / n ** 0.5), 0.0), "L2")
    compare(result, "Gaussian fit width", abs(fit["sigma"]), sigma,
            Tol("max(bin/4, 5 sigma/sqrt(2(n-1)))", max(0.25 * bin_width, 5 * sigma / (2 * (n - 1)) ** 0.5), 0.0), "L2")


def check_refusal(adf, spec: Refuses, result: CaseResult) -> None:
    """The call must raise with the declared message; Figures allocated before refusing are recorded (F-S2-1, D14)."""
    before = set(plt.get_fignums())
    try:
        spec.call(adf)
    except Exception as exc:  # the refusal is the expected outcome
        named = spec.message in str(exc)
        compare(result, f"refusal names {spec.message!r}", int(named), 1, EXACT, "L3")
        compare(result, "Figures allocated before refusal", len(set(plt.get_fignums()) - before), 0,
                EXACT, "L3", diagnostic=True)
        result.detail = f"refused as required: {str(exc)[:100]}"
        return
    raise CheckFailure("L3", "did NOT refuse")


def evaluate(adf, case: Case, drawn, result: CaseResult) -> None:
    """Evaluate every declaration of one case; a case with no gating comparison is a failure of the test (H)."""
    checks = {Profile: check_profile, Delta: check_delta, Histogram: check_histogram, GaussFit: check_gauss_fit,
              FacetQuantiles: check_facet_quantiles, RowCounts: check_row_counts}
    needs_truth = case.aliases or case.fixture or any(type(s) in checks for s in case.expect)
    t = truth_frame(adf, case, result) if needs_truth else None
    for spec in case.expect:
        if type(spec) in checks:
            checks[type(spec)](t, spec, drawn, result)
        elif isinstance(spec, Refuses):
            check_refusal(adf, spec, result)
        elif isinstance(spec, Legend):
            check_legend(spec, drawn, result)
        elif isinstance(spec, Headings):
            check_headings(spec, drawn, result)
        elif isinstance(spec, GalleryPages):
            check_gallery_pages(spec, result)
        elif isinstance(spec, SameAcrossSurfaces):
            check_same_across_surfaces(spec, drawn, result)
        elif isinstance(spec, Flag):
            check_flag(spec, drawn, result)
        elif isinstance(spec, SameAsCall):
            check_same_as_call(adf, spec, drawn, result)
        elif isinstance(spec, SameStyleAcrossFacets):
            check_same_style(spec, drawn, result)
        elif isinstance(spec, Branches):
            check_branches(spec, drawn, result)
        elif isinstance(spec, FacetPanels):
            check_facet_panels(spec, drawn, result)
        else:
            raise CheckFailure("H", f"unknown declaration {type(spec).__name__}")
    gating_levels = {c["level"] for c in result.comparisons if c["ok"] is not None}
    if not gating_levels:
        raise CheckFailure("H", "no gating comparison was executed")
    if drawn is not None and "L2" not in gating_levels:
        raise CheckFailure("H", "a drawn page needs at least one comparison of what was plotted (L2)")


# =============================================================================
# 5. The runner (run_gallery.sh calls main(); no orchestration here)
# =============================================================================
def evidence_page(case: Case, result: CaseResult):
    """A text page after each case: verdict, owner, and the comparisons closest to their tolerance."""
    fig = plt.figure(figsize=(11.0, 8.5))
    rows = sorted(result.comparisons, key=lambda c: -(c["worst_abs"] / max(c["atol"] + c["rtol"], 1e-300)))
    lines = [f"{case.case_id}", case.title, f"verdict: {result.status}"
             + (f"   owner: {result.failure_level}" if result.failure_level else "")
             + f"   comparisons: {len(result.comparisons)}   wall time: {result.wall_time_s} s", result.detail[:150], "",
             f"{'level':5s} {'ok':5s} {'worst |diff|':>13s}  {'tolerance':44s} observable"]
    lines += [f"{c['level']:5s} {str(c['ok']):5s} {c['worst_abs']:13.3e}  {c['tolerance'][:44]:44s} {c['observable'][:58]}"
              for c in rows[:40]]
    if len(rows) > 40:
        lines.append(f"... {len(rows) - 40} more comparisons, all closer to zero than the ones above (manifest has all)")
    fig.text(0.03, 0.97, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.5)
    return fig


def run_case(adf, case: Case, pdf: PdfPages | None, page_counter: list) -> CaseResult:
    """Draw the page once and save it unchanged, evaluate the declarations, close only this case's Figures."""
    result, t0, before = CaseResult(case.case_id), time.perf_counter(), set(plt.get_fignums())
    try:
        try:
            drawn = case.draw(adf) if case.draw else None
        except Exception as exc:  # the public gallery call itself failed
            raise CheckFailure("L3", f"gallery function raised {type(exc).__name__}: {str(exc)[:160]}") from exc
        if drawn is not None and pdf is not None:
            pdf.attach_note(f"{case.case_id}  {case.title}")      # a note labels the NEXT page saved
            pdf.savefig(drawn[0], bbox_inches="tight")
            page_counter[0] += 1
            result.page = page_counter[0]
        evaluate(adf, case, drawn, result)
        result.status = PASS
    except CheckFailure as exc:
        result.status, result.failure_level, result.detail = FAIL, exc.level, str(exc)
    except Exception as exc:  # an error inside this test file: never a silent PASS, never blamed on the product
        result.status, result.failure_level = FAIL, "H"
        result.detail = f"H/{OWNER['H']}: {type(exc).__name__}: {str(exc)[:200]}"
    finally:
        for num in set(plt.get_fignums()) - before:
            plt.close(num)
        result.wall_time_s = round(time.perf_counter() - t0, 3)
        if pdf is not None:
            page = evidence_page(case, result)
            pdf.attach_note(f"{case.case_id}  evidence")
            pdf.savefig(page)
            plt.close(page)
            page_counter[0] += 1
    return result


def run_cases(adf, cases: list, *, pdf_path: str | None, manifest_path: str, provenance: dict,
              selection: list | None = None, input_ok: bool = True, mandatory: tuple = None) -> int:
    """Run the selected cases of the full table once; write PDF and manifest; return the strict exit code.

    `mandatory` is the inventory the table must contain (default MANDATORY_CASE_IDS); a case deleted
    from the table is a failure, not a smaller table."""
    declared = [c.case_id for c in cases]
    mandatory = MANDATORY_CASE_IDS if mandatory is None else mandatory
    missing_mandatory = [i for i in mandatory if i not in declared]
    unknown = sorted(set(selection or []) - set(declared))
    if not declared or unknown or selection == []:
        raise ValueError(f"no case to run: declared={len(declared)}, unknown selection={unknown}")
    chosen = [c for c in cases if selection is None or c.case_id in selection]
    pdf, pages = (PdfPages(pdf_path) if pdf_path else None), [0]
    results = []
    try:
        for c in chosen:
            r = run_case(adf, c, pdf, pages)
            results.append(r)
            print(f"[{_dt.datetime.now():%H:%M:%S}] {r.status:5s} {r.failure_level:3s} {r.case_id}  "
                  f"{r.wall_time_s:.1f} s  {r.detail}", flush=True)
    finally:
        if pdf is not None:
            pdf.close()
    ids = [r.case_id for r in results]
    rec = {"n_declared": len(declared), "missing_mandatory": missing_mandatory,
           "selection": selection or "ALL", "n_selected": len(chosen),
           "n_results": len(results), "not_selected": [i for i in declared if i not in ids],
           "duplicated": sorted({i for i in declared if declared.count(i) > 1}),
           "gating": [r.case_id for r in results if r.status == FAIL],
           "not_executed": [r.case_id for r in results if r.status == SKIP], "input_identity_ok": input_ok}
    parity = historical_parity(results)
    rec["historical_parity_failed"] = parity["failed"]
    rec["exit_code"] = int(bool(rec["duplicated"] or rec["gating"] or rec["not_executed"] or not input_ok
                                or missing_mandatory or rec["n_results"] != rec["n_selected"] or not parity["ok"]))
    with open(manifest_path, "w") as fh:
        json.dump({"schema_version": "13.105-invariancetest-5", "provenance": provenance,
                   "cases": [dataclasses.asdict(r) for r in results], "reconciliation": rec,
                   "historical_parity": parity,
                   "not_yet_migrated": list(NOT_YET_MIGRATED), "targets": provenance.get("targets", {})},
                  fh, indent=1, sort_keys=True, default=str)
    if missing_mandatory:
        print(f"FAIL  mandatory case(s) missing from the table: {missing_mandatory}")
    print(f"HISTORICAL PARITY: {parity['checked']}/{len(parity['rows'])} old gates carried at their old tolerance "
          f"or stricter" + ("" if parity["ok"] else f"  FAILED: {parity['failed']}"))
    gate = ("PASS" if rec["exit_code"] == 0 else "FAIL") + ("" if selection is None else f" (SUBSET {len(chosen)}/{len(declared)})")
    print(f"cases: {len(results)}/{len(declared)}  gating_reds: {len(rec['gating'])}  STRICT GATE: {gate}")
    return rec["exit_code"]


def leaf_digest(a) -> tuple:
    """SHA-256 of a branch's values: numeric arrays by bytes; variable-length arrays by lengths + concatenated values."""
    if a.dtype != object:
        a = np.ascontiguousarray(a)
        return str(a.dtype), hashlib.sha256(a.tobytes()).hexdigest()
    parts = [np.ascontiguousarray(np.asarray(x)) for x in a]
    lengths = np.array([p.size for p in parts], dtype=np.int64)
    values = np.ascontiguousarray(np.concatenate(parts)) if parts else np.zeros(0)
    return f"var*{values.dtype}", hashlib.sha256(lengths.tobytes() + values.tobytes()).hexdigest()


def input_identity(root_path: str, entry_stop: int) -> tuple:
    """Independent of ADF: per-leaf SHA-256 of the first N values via uproot; parent branches are covered by their leaves."""
    import uproot
    t0, digests, raw, parents = time.perf_counter(), {}, {}, []
    with uproot.open(os.path.realpath(root_path)) as f:
        trees = [k for k, v in f.classnames().items() if v == "TTree"]
        if len(trees) != 1:
            raise ValueError(f"expected exactly one TTree, found {trees}")
        tree = f[trees[0].split(";")[0]]
        for name in tree.keys():
            if len(tree[name].branches) > 0:
                parents.append(name)                       # a parent branch: its leaves are digested
                continue
            a = tree[name].array(entry_start=0, entry_stop=entry_stop, library="np")
            digests[name] = leaf_digest(a)
            if name in RAW_COLUMNS:
                raw[name] = a
    combined = hashlib.sha256("".join(f"{k}:{d}:{s}\n" for k, (d, s) in sorted(digests.items())).encode()).hexdigest()
    return {"combined_sha256": combined, "leaf_digests": digests, "parent_branches": parents,
            "n_leaves": len(digests), "rows": int(len(a)),
            "read_seconds": round(time.perf_counter() - t0, 2)}, raw


def identity_matches_s0(identity: dict, s0: dict) -> dict:
    """Every S0 leaf must still exist and no new leaf may appear (numeric or variable-length); numeric leaves
    must also keep dtype and digest. Only parent branches are exempt; S0 values recorded by address are listed."""
    measured, parents = identity["leaf_digests"], set(identity.get("parent_branches", ()))
    missing = sorted(k for k in s0 if k not in parents and k not in measured)
    unexpected = sorted(k for k in measured if k not in s0)
    numeric = {k: v for k, v in s0.items() if v["dtype"] != "object" and k in measured}
    dtype_changed = sorted(k for k, v in numeric.items() if measured[k][0] != v["dtype"])
    differ = sorted(k for k, v in numeric.items() if k not in dtype_changed and measured[k][1] != v["sha256"])
    no_value = sorted(k for k in measured if k in s0 and s0[k]["dtype"] == "object")
    ok = not (missing or unexpected or dtype_changed or differ)
    return {"ok": ok, "s0_branches": len(s0), "parent_branches": sorted(parents), "missing_from_input": missing,
            "unexpected_leaves": unexpected, "dtype_changed": dtype_changed, "differ_from_s0": differ,
            "no_reproducible_s0_value": no_value}


def loaded_equals_raw(adf, raw: dict) -> dict:
    """L0: every raw column the truth uses, as loaded by ADF, equals the ROOT values (cast to the loaded dtype)."""
    return {c: c in raw and bool(np.array_equal(adf.df[c].to_numpy(), raw[c].astype(adf.df[c].dtype), equal_nan=True))
            for c in RAW_COLUMNS}


def code_fingerprints(adf) -> dict:
    """Path and SHA-256 of the files that define the classes actually used; a missing file raises."""
    drawer = inspect.getsourcefile(G.DFDraw)
    files = {"time_series_draw.py": G.__file__, "time_series_draw_invariancetest.py": __file__,
             "AliasDataFrame.py": inspect.getsourcefile(type(adf)), "dfdraw/drawer.py": drawer}
    for p in sorted(glob.glob(os.path.join(os.path.dirname(drawer), "plots", "*.py"))):
        files[f"dfdraw/plots/{os.path.basename(p)}"] = p
    return {k: {"path": v, "sha256": hashlib.sha256(open(v, "rb").read()).hexdigest()} for k, v in files.items()}


# =============================================================================
# 6. The cases
# =============================================================================
def control_one_element_selection(adf):
    """September control: a ONE-element selection_vector (first third in time) must filter every facet."""
    q1 = float(adf.df["time_s"].quantile(1.0 / 3.0))
    return adf.draw("dcar_tpc_vertex:tgl", selection=SIDES_A_C, type="profile", bins=30, range=(-1.5, 1.5),
                    selection_vector=[f"time_s<{q1}"], facet_by="side_type", min_entries=1, auto_title=False)


def control_hist_selection_facets(adf):
    """September control: a histogram with two selection branches must apply both inside every facet."""
    t_mid = float(adf.df["time_s"].median())
    return adf.draw("ncl", selection="(ncl>60)&(side_type<2)", type="hist", bins=40, range=(80.0, 160.0),
                    selection_vector=[f"time_s<{t_mid}", f"time_s>={t_mid}"], facet_by="side_type", auto_title=False)


def first_third(t):
    return t.time_s < t.time_s.quantile(1.0 / 3.0)


# ============================================================================= — the inventory first (what MUST exist), then the table (what each case declares)
# =============================================================================
MANDATORY_CASE_IDS = (
    "I4-REAL-INJECTED-TRUTH-FACET-RESIDUAL-EAGER-20PCT-01",
    "I4-REAL-GROUPING-TRUTH-EAGER-20PCT-01",
    "I2-FACET-DRAW-FIGURES-REFUSAL-01",
    "I4-REAL-INJECTED-TRUTH-VECTOR-OVERLAY-EAGER-20PCT-01",
    "I4-REAL-INJECTED-TRUTH-DIRECT-DELTA-EAGER-20PCT-01",
    "I4-REAL-INJECTED-TRUTH-SELECTION-DELTA-FACET-EAGER-20PCT-01",
    "I4-REAL-INJECTED-TRUTH-WEIGHTS-FACET-EAGER-20PCT-01",
    "I4-REAL-INJECTED-TRUTH-GAUSS-FIT-EAGER-20PCT-01",
    "I4-REAL-HIST-VECTOR-TRUTH-EAGER-20PCT-01",
    "I4-REAL-QUALIFIED-SUBFRAME-CHAIN-TRUTH-EAGER-20PCT-01",
    "ADF99-01",
    "ADF99-02",
    "ADF100-01",
    "A6-VISUAL-GALLERY-FULL-01",
    "I4-REAL-VECTOR-FACET-ONE-ELEMENT-SELECTION-EAGER-20PCT-01",
    "I4-REAL-HIST-VECTOR-FACET-SELECTION-EAGER-20PCT-01",
    "PRECLEAN-C1-HIST-VECTOR-FACET-ACCEPTANCE-EAGER-20PCT-01",
    "PRECLEAN-C2-DRAW-FIGURES-EFFECTIVE-DEFAULTS-EAGER-20PCT-01",
    "PRECLEAN-C3-RATIO-Y-VECTOR-TRUTH-EAGER-20PCT-01",
    "I4-REAL-SELECTION-VECTOR-FACET-DELTA-CORRECTNESS-EAGER-20PCT-01",
    "I4-REAL-PUBLIC-SURFACE-QUALIFIED-VECTOR-EQUIV-EAGER-20PCT-01",
    "I4-REAL-WEIGHTS-VECTOR-FACET-PROFILE-CORRECTNESS-EAGER-20PCT-01",
)

NOT_YET_MIGRATED = (   # historical case IDs still owned by the old harness; each S3 family shortens this list
    "I4-REAL-VECTOR-FACET-SUMMARYFIT-SEMANTICS-EAGER-20PCT-01",
    "I4-REAL-G7-SUBFRAME-EAGER-20PCT-01",
    "I4-REAL-G7-GB-EAGER-20PCT-01",
    "I4-REAL-G7-GB-SECTOR-REUSE-EAGER-20PCT-01",
    "I4-REAL-G7-GB-LOGICAL-STATE-EAGER-20PCT-01",
)

TARGETS = {"gallery": "the readable 59-page gallery (time_series_draw.run_all_pdf) on the loaded data",
           "13_105": "the declared cases of this file, strict"}


def migration_crosswalk(old_case_ids, new_manifest: dict) -> dict:
    """Every old-harness case is either PASS in the new manifest or listed in its not_yet_migrated ledger,
    never both, never neither. The ledger must be present in the manifest, a list of known old IDs, no duplicates."""
    ledger = new_manifest.get("not_yet_migrated")
    ledger_problems = []
    if not isinstance(ledger, list) or not all(isinstance(i, str) for i in ledger):
        ledger_problems.append("not_yet_migrated missing or not a list of case IDs")
        ledger = []
    if len(set(ledger)) != len(ledger):
        ledger_problems.append("not_yet_migrated has duplicates")
    unknown = sorted(set(ledger) - set(old_case_ids))
    if unknown:
        ledger_problems.append(f"not_yet_migrated names unknown cases {unknown}")
    new_status = {c["case_id"]: c["status"] for c in new_manifest["cases"]}
    still_old = set(ledger)
    both = sorted(i for i in old_case_ids if i in still_old and i in new_status)
    neither = sorted(i for i in old_case_ids if i not in still_old and i not in new_status)
    not_passed = sorted(i for i in old_case_ids if i in new_status and new_status[i] != PASS)
    return {"ok": not (both or neither or not_passed or ledger_problems),
            "migrated": sorted(i for i in old_case_ids if i in new_status),
            "still_old": sorted(i for i in old_case_ids if i in still_old), "in_both": both, "in_neither": neither,
            "migrated_but_not_pass": not_passed, "ledger_problems": ledger_problems}


# Every gating assertion of the old harness for a migrated case: (old name, old atol, old rtol, the new observables
# that carry it as a regular expression). Exact = 0, 0. A tolerance given as a rule is the text of that rule.
# "declared": the old Observable declarations (all 66, extracted by script); "internal": raise-checks inside the old
# runner functions, found by reading the code; "review": added by the S3/S4 review (P1-S3-01).
HISTORICAL_GATES = {
    "I4-REAL-INJECTED-TRUTH-FACET-RESIDUAL-EAGER-20PCT-01": (
        ("declared", "count", 0, 0, r"^oracle_residual side_type=\d+ count$"),
        ("declared", "y_mean", 1e-9, 1e-8, r"^oracle_residual side_type=\d+ mean$")),
    "I4-REAL-GROUPING-TRUTH-EAGER-20PCT-01": (
        ("declared", "group_counts", 0, 0, r"^known_delta \S+=\S+ count$"),
        ("declared", "group_means", 1e-9, 1e-8, r"^known_delta \S+=\S+ mean$")),
    "I4-REAL-INJECTED-TRUTH-VECTOR-OVERLAY-EAGER-20PCT-01": (
        ("declared", "counts", 0, 0, r"^(dcar_\w+|known_delta) count$"),
        ("declared", "profile_means", 1e-9, 1e-8, r"^(dcar_\w+|known_delta) mean$"),
        ("internal", "exactly four branch stats", 0, 0, r"^number of vector branches$"),
        ("internal", "vector branch i x_center", 1e-9, 1e-8, r"^(dcar_\w+|known_delta) bin centres$")),
    "I4-REAL-INJECTED-TRUTH-DIRECT-DELTA-EAGER-20PCT-01": (
        ("declared", "delta_values", 1e-9, 1e-8, r"^delta values$"),),
    "I4-REAL-INJECTED-TRUTH-SELECTION-DELTA-FACET-EAGER-20PCT-01": (
        ("declared", "delta_values", 1e-9, 1e-8, r"^delta side_type=\d+ values$"),
        ("declared", "valid_mask", 0, 0, r"^delta side_type=\d+ undefined bins$")),
    "I4-REAL-INJECTED-TRUTH-WEIGHTS-FACET-EAGER-20PCT-01": (
        ("declared", "count", 0, 0, r"^known_delta side_type=\d+ weights=\w+ count$"),
        ("declared", "sum_weights", 1e-8, 1e-7, r"^known_delta side_type=\d+ weights=\w+ sum of weights$"),
        ("declared", "y_mean", 1e-8, 1e-7, r"^known_delta side_type=\d+ weights=\w+ mean$"),
        ("internal", "two weight branches", 0, 0, r"^number of vector branches$")),
    "I4-REAL-INJECTED-TRUTH-GAUSS-FIT-EAGER-20PCT-01": (
        ("declared", "fit_center_delta", "max(bin/4, 5 sigma/sqrt n)", None, r"^Gaussian fit centre$"),
        ("declared", "fit_sigma_delta", "max(bin/4, 5 sigma/sqrt(2(n-1)))", None, r"^Gaussian fit width$")),
    "I4-REAL-HIST-VECTOR-TRUTH-EAGER-20PCT-01": (
        ("declared", "selection_bin_counts", 0, 0, r"^histogram ncl (early|late)$"),
        ("declared", "weighted_bin_sums", 1e-8, 1e-10, r"^histogram ncl weights=\w+$")),
    "I4-REAL-QUALIFIED-SUBFRAME-CHAIN-TRUTH-EAGER-20PCT-01": (
        ("declared", "count", 0, 0, r"^oracle_chain_with_shift side_type=\d+ count$"),
        ("declared", "y_mean", 1e-9, 1e-8, r"^oracle_chain_with_shift side_type=\d+ mean$")),
    "ADF99-01": (
        ("declared", "quantile_bins", 3e-4, 2e-4, r"^cell \d+ group None \w+ (q=\d+%|median) values$"),
        ("declared", "legend_identifiers", 0, 0, r"^legend identities$"),
        ("internal", "quantile x centers", 2e-3, 0, r"^cell \d+ group None \w+ (q=\d+%|median) x$"),
        ("internal", "six panels", 0, 0, r"^number of facet panels$")),
    "ADF99-02": (
        ("declared", "group_quantiles", 3e-4, 2e-4, r"^cell \d+ group [01] \w+ (q=\d+%|median) values$"),
        ("declared", "factored_legend", 0, 0, r"^legend identities$"),
        ("internal", "quantile x centers", 2e-3, 0, r"^cell \d+ group [01] \w+ (q=\d+%|median) x$"),
        ("internal", "six panels", 0, 0, r"^number of facet panels$"),
        ("internal", "five outer headings", 0, 0, r"^distinct outer facet headings$")),
    "ADF100-01": (
        ("declared", "paired_counts", 0, 0, r"^\w+ tgl=\S+ count$"),
        ("declared", "paired_means", 3e-5, 2e-4, r"^\w+ tgl=\S+ mean$"),
        ("declared", "paired_edges", 2e-3, 0, r"^\w+ tgl=\S+ bin edges$"),
        ("internal", "two paired branches", 0, 0, r"^number of vector branches$"),
        ("internal", "six facet identities", 0, 0, r"^\w+ groups of tgl$")),
    "A6-VISUAL-GALLERY-FULL-01": (
        ("internal", "gallery ran without error", 0, 0, r"^gallery run_all_pdf succeeded$"),),
    "I4-REAL-VECTOR-FACET-ONE-ELEMENT-SELECTION-EAGER-20PCT-01": (
        ("declared", "facet_selected_rows", 0, 0, r"^rows per side_type$"),),
    "I4-REAL-HIST-VECTOR-FACET-SELECTION-EAGER-20PCT-01": (
        ("declared", "branch_facet_rows", 0, 0, r"^histogram ncl side_type=\d+ (early|late)$"),
        ("internal", "two histogram facets", 0, 0, r"^facet panels \(by title\)$")),
    "I4-REAL-SELECTION-VECTOR-FACET-DELTA-CORRECTNESS-EAGER-20PCT-01": (
        ("declared", "facet_values", 0, 0, r"^delta groups of side_type$"),
        ("declared", "signal_central", 1e-12, 1e-12, r"^delta side_type=\d+ signal plotted line$"),
        ("declared", "reference_central", 1e-12, 1e-12, r"^delta side_type=\d+ reference plotted line$"),
        ("declared", "delta_values", 1e-12, 1e-12, r"^delta side_type=\d+ values$"),
        ("declared", "valid_bin_mask", 0, 0, r"^delta side_type=\d+ undefined bins$"),
        ("internal", "overlay points on bin centres", 1e-12, 1e-12, r"^delta side_type=\d+ \w+ line x on bin centres$")),
    "I4-REAL-WEIGHTS-VECTOR-FACET-PROFILE-CORRECTNESS-EAGER-20PCT-01": (
        ("declared", "facet_values", 0, 0, r"^dcar_tpc_vertex groups of side_type$"),
        ("declared", "weight_branch_labels", 0, 0, r"^number of vector branches$"),
        ("declared", "count", 0, 0, r"^dcar_tpc_vertex side_type=\d+ weights=\w+ count$"),
        ("declared", "sum_weights", 1e-6, 5e-6, r"^dcar_tpc_vertex side_type=\d+ weights=\w+ sum of weights$"),
        ("declared", "y_mean", 1e-7, 5e-6, r"^dcar_tpc_vertex side_type=\d+ weights=\w+ mean$"),
        ("declared", "y_std", 1e-7, 5e-6, r"^dcar_tpc_vertex side_type=\d+ weights=\w+ std$"),
        ("declared", "y_sem", 1e-7, 5e-6, r"^dcar_tpc_vertex side_type=\d+ weights=\w+ sem$"),
        ("internal", "x_center against the raw grid", 1e-12, 1e-12, r"^dcar_tpc_vertex side_type=\d+ weights=\w+ bin centres$"),
        ("internal", "scalar decomposition count", 0, 0, r"^weights_vector branch \d .* count$"),
        ("internal", "scalar decomposition values", 1e-12, 1e-12,
         r"^weights_vector branch \d .* (x_center|sum_weights|y_mean|y_std|y_sem)$"),
        ("internal", "scalar decomposition pol1 fit", 1e-10, 1e-10, r"^weights_vector branch \d .* pol1 parameters$"),
        ("internal", "same style in both facets", 0, 0, r"^branch styles side_type=\d+ == side_type=\d+$")),
    "PRECLEAN-C1-HIST-VECTOR-FACET-ACCEPTANCE-EAGER-20PCT-01": (
        ("declared", "facet_identities", 0, 0, r"^facet panels \(by title\)$"),
        ("declared", "faceted_bin_counts", 0, 0, r"^histogram ncl side_type=\d+ (early|late)$"),
        ("declared", "control_bin_counts", 0, 0, r"^histogram ncl control (early|late)$"),
        ("declared", "faceted_inrange_rows (= sum of the exact bin counts)", 0, 0, r"^histogram ncl side_type=\d+ (early|late)$"),
        ("declared", "control_inrange_rows (= sum of the exact bin counts)", 0, 0, r"^histogram ncl control (early|late)$")),
    "PRECLEAN-C2-DRAW-FIGURES-EFFECTIVE-DEFAULTS-EAGER-20PCT-01": (
        ("declared", "x_center", 1e-14, 1e-12, r"^delta bin centres$"),
        ("declared", "signal_count", 0, 0, r"^delta signal count$"),
        ("declared", "reference_count", 0, 0, r"^delta reference count$"),
        ("declared", "signal_central", 1e-12, 1e-12, r"^delta signal central$"),
        ("declared", "reference_central", 1e-12, 1e-12, r"^delta reference central$"),
        ("declared", "delta_values", 1e-12, 1e-12, r"^delta values$"),
        ("declared", "valid_bin_mask", 0, 0, r"^delta undefined bins$"),
        ("declared", "caller_unchanged", 0, 0, r"^caller's specs and defaults unchanged$"),
        ("declared", "short_form_preserved", 0, 0, r"^short-form plot string preserved$")),
    "PRECLEAN-C3-RATIO-Y-VECTOR-TRUTH-EAGER-20PCT-01": (
        ("declared", "x_center", 1e-14, 1e-12, r"^ratio bin centres$"),
        ("declared", "signal_count", 0, 0, r"^ratio signal count$"),
        ("declared", "reference_count", 0, 0, r"^ratio reference count$"),
        ("declared", "signal_central", 1e-12, 1e-12, r"^ratio signal central$"),
        ("declared", "reference_central", 2e-12, 1e-12, r"^ratio reference central$"),
        ("declared", "ratio_values", 2e-13, 2e-13, r"^ratio values$"),
        ("declared", "valid_bin_mask", 0, 0, r"^ratio undefined bins$")),
    "I4-REAL-PUBLIC-SURFACE-QUALIFIED-VECTOR-EQUIV-EAGER-20PCT-01": (
        ("declared", "x_center", 1e-14, 1e-12, r"^draw_(batch|figures) x_center == draw$"),
        ("declared", "signal_central", 1e-12, 1e-10, r"^draw_(batch|figures) signal_central == draw$"),
        ("declared", "signal_count", 0, 0, r"^draw_(batch|figures) signal_count == draw$"),
        ("declared", "reference_central", 1e-12, 1e-10, r"^draw_(batch|figures) reference_central == draw$"),
        ("declared", "reference_count", 0, 0, r"^draw_(batch|figures) reference_count == draw$"),
        ("declared", "value", 1e-12, 1e-10, r"^draw_(batch|figures) value == draw$"),
        ("review", "mask_undefined", 0, 0, r"^draw_(batch|figures) mask_undefined == draw$")),
}


def historical_parity(results: list) -> dict:
    """For every historical gate of a case that ran and passed: at least one new gating comparison carries it,
    and each of them is at the old tolerance or stricter (atol and rtol both; a rule must be the same rule)."""
    rows = []
    for r in results:
        gating = [c for c in r.comparisons if c["ok"] is not None]
        for source, old, atol, rtol, pattern in HISTORICAL_GATES.get(r.case_id, ()):
            new = [c for c in gating if re.search(pattern, c["observable"])]
            if r.status != PASS:
                verdict = "case did not pass"
            elif not new:
                verdict = "MISSING"
            elif isinstance(atol, str):
                verdict = "ok" if all(c["tolerance"] == atol for c in new) else "WEAKER"
            else:
                verdict = "ok" if all(c["atol"] <= atol and c["rtol"] <= rtol for c in new) else "WEAKER"
            rows.append(dict(case_id=r.case_id, source=source, old=old, old_atol=atol, old_rtol=rtol, pattern=pattern,
                             n_new=len(new), new_atol=max((c["atol"] for c in new), default=None),
                             new_rtol=max((c["rtol"] for c in new), default=None),
                             new_tolerances=sorted({c["tolerance"] for c in new}), verdict=verdict))
    bad = [row for row in rows if row["verdict"] in ("MISSING", "WEAKER")]
    return {"ok": not bad, "checked": sum(row["verdict"] == "ok" for row in rows), "rows": rows,
            "failed": [f"{row['case_id']} {row['old']}: {row['verdict']}" for row in bad]}


def run_gallery_target(adf, pdf_path: str) -> dict:
    """The gallery as a target: run_all_pdf on the already loaded ADF; record result and page count."""
    import re
    t0 = time.perf_counter()
    ok = bool(G.run_all_pdf(adf, pdf_path))
    data = open(pdf_path, "rb").read() if os.path.exists(pdf_path) else b""
    return {"ok": ok and data.startswith(b"%PDF-"), "pages": len(re.findall(rb"/Type\s*/Page\b", data)),
            "pdf": pdf_path, "wall_time_s": round(time.perf_counter() - t0, 1)}

CASES = [
    Case("I4-REAL-INJECTED-TRUTH-FACET-RESIDUAL-EAGER-20PCT-01", G.fig51_injected_truth_facet_residual,
         "G12.51 injected truth — residual profile per side_type",
         aliases=INJECTED_ALIASES, fixture=injected_noise_is_declared,
         expect=(Profile("oracle_residual", SIDES_A_C, **SECTOR_BINS, by="side_type", plotted=lambda d: facets(d[2])),
                 # physics: the plotted residual mean must equal the mean of the injected noise
                 Profile("oracle_noise", SIDES_A_C, **SECTOR_BINS, by="side_type", plotted=lambda d: facets(d[2]),
                         tol=MIXED_PRECISION))),
    Case("I4-REAL-GROUPING-TRUTH-EAGER-20PCT-01", G.fig53_grouping_truth,
         "G13.53 grouping truth — categorical group_by and numeric group_by_bins",
         aliases={**SECTOR_ALIAS, "known_delta": INJECTED_ALIASES["known_delta"]},
         expect=(Profile("known_delta", SIDES_A_C, **SECTOR_BINS, by="side_type",
                         plotted=lambda d: groups(d[2]["categorical_stats"])),
                 Profile("known_delta", SELECTED, **SECTOR_BINS, by="tgl", by_bins=4,
                         plotted=lambda d: groups(d[2]["binned_stats"])))),
    Case("I2-FACET-DRAW-FIGURES-REFUSAL-01", None,
         "draw_figures refuses facet_by (machine-only contract)",
         expect=(Refuses(lambda adf: adf.draw_figures(
                     [{"plots": [{"expr": "dcar_tpc_vertex:sector", "type": "profile", "bins": 8,
                                  "selection": SELECTED, "facet_by": "side_type"}]}], lazy=True),
                         "facet_by is not supported"),)),
    # ---- batch A: the rest of the injected-truth family, the histogram vector, the qualified-subframe chain
    Case("I4-REAL-INJECTED-TRUTH-VECTOR-OVERLAY-EAGER-20PCT-01", G.fig46_injected_truth_vector_overlay,
         "G12.46 four-branch bracket vector: original, clean-distorted, noisy, known_delta",
         aliases=INJECTED_ALIASES, fixture=injected_noise_is_declared,
         expect=(Branches(4),)
                + tuple(Profile(y, SELECTED, **SECTOR_BINS, plotted=branch(i), edges_tol=SAME_VALUES)
                        for i, y in enumerate(["dcar_tpc_vertex", "dcar_distorted_clean", "dcar_oracle", "known_delta"]))),
    Case("I4-REAL-INJECTED-TRUTH-DIRECT-DELTA-EAGER-20PCT-01", G.fig47_injected_truth_direct_delta,
         "G12.47 normalize=delta of clean-distorted minus original recovers known_delta",
         aliases=INJECTED_ALIASES,
         expect=(Delta(Profile("dcar_distorted_clean", SELECTED, **SECTOR_BINS, plotted=None),
                       Profile("dcar_tpc_vertex", SELECTED, **SECTOR_BINS, plotted=None),
                       plotted=lambda d: [(0, d[2]["normalize_data"])]),)),
    Case("I4-REAL-INJECTED-TRUTH-SELECTION-DELTA-FACET-EAGER-20PCT-01", G.fig48_injected_truth_selection_delta_facet,
         "G12.48 selection_vector delta tgl<0 minus tgl>=0 of known_delta, per side_type",
         aliases=INJECTED_ALIASES,
         expect=(Delta(Profile("known_delta", SIDES_A_C + " & (tgl < 0)", **SECTOR_BINS, by="side_type", plotted=None),
                       Profile("known_delta", SIDES_A_C + " & (tgl >= 0)", **SECTOR_BINS, by="side_type", plotted=None),
                       plotted=lambda d: sorted(d[2]["normalize_data_faceted"].items())),)),
    Case("I4-REAL-INJECTED-TRUTH-WEIGHTS-FACET-EAGER-20PCT-01", G.fig49_injected_truth_weights_facet,
         "G12.49 weights_vector flat / tgl-weighted known_delta profile, per side_type",
         aliases=INJECTED_ALIASES, fixture=injected_noise_is_declared,
         expect=(Branches(2),)
                + tuple(Profile("known_delta", SIDES_A_C, **SECTOR_BINS, by="side_type", weights=w,
                                plotted=branch(i, facets), tol=WEIGHTED)
                        for i, w in enumerate(["oracle_w_flat", "oracle_w_tgl"]))),
    Case("I4-REAL-INJECTED-TRUTH-GAUSS-FIT-EAGER-20PCT-01", G.fig50_injected_truth_gaussian_fit,
         "G12.50 Gaussian fit of the residual recovers the injected noise (centre, width)",
         aliases=INJECTED_ALIASES, fixture=injected_noise_is_declared,
         expect=(GaussFit("oracle_noise", SELECTED, range=(-0.1, 0.1), bins=100, plotted=lambda d: first_fit(d[2])),)),
    Case("I4-REAL-HIST-VECTOR-TRUTH-EAGER-20PCT-01", G.fig52_hist_vector_truth,
         "G13.52 histogram vectors: early / late selection, flat / tgl-weighted",
         aliases=TIME_ALIAS, fixture=injected_noise_is_declared,
         expect=(Histogram("ncl", lambda t: (t.ncl > 60) & (t.time_s < t.time_s.median()), bins=20, range=(80.0, 160.0),
                           plotted=lambda d: bar_heights(d[1][0], 0, 20), label="early"),
                 Histogram("ncl", lambda t: (t.ncl > 60) & (t.time_s >= t.time_s.median()), bins=20, range=(80.0, 160.0),
                           plotted=lambda d: bar_heights(d[1][0], 1, 20), label="late"),
                 Histogram("ncl", "ncl > 60", bins=20, range=(80.0, 160.0), weights="oracle_w_flat",
                           plotted=lambda d: bar_heights(d[1][1], 0, 20), tol=HIST_WEIGHTED),
                 Histogram("ncl", "ncl > 60", bins=20, range=(80.0, 160.0), weights="oracle_w_tgl",
                           plotted=lambda d: bar_heights(d[1][1], 1, 20), tol=HIST_WEIGHTED))),
    Case("I4-REAL-QUALIFIED-SUBFRAME-CHAIN-TRUTH-EAGER-20PCT-01", G.fig54_qualified_subframe_chain_truth,
         "G13.54 chained aliases through the OracleShift subframe, grouped by side_type",
         aliases=CHAIN_ALIASES,
         expect=(Profile("oracle_chain_with_shift", SIDES_A_C, **SECTOR_BINS, by="side_type",
                         plotted=lambda d: groups(d[2])),)),
    # ---- batch C1: the three PHASE_13_101 real-data oracles and the visual gallery case
    Case("ADF99-01", G.fig_adf99_01_quantile_facets,
         "ADF99-01 six tgl facets, two Y, discrete quantiles, one shared legend",
         aliases=SECTOR_ALIAS,
         expect=(FacetQuantiles(("dcar_tpc_vertex", "dcar_tpc"), SELECTED, facets=(("tgl", 6),)),
                 Legend(("dcar_tpc_vertex", "dcar_tpc") + tuple(quantile_label(q) for q in QUANTILES)))),
    Case("ADF99-02", G.fig_adf99_02_quantile_facet_groups,
         "ADF99-02 3x2 tgl x qpt_bin10 facets, global side_type groups, quantiles, outer headings",
         aliases=SECTOR_ALIAS, derived=QPT_BIN,
         expect=(FacetQuantiles(("dcar_tpc_vertex", "dcar_tpc"), SIDES_A_C, facets=(("tgl", 3), ("qpt_bin10", 2)),
                                by="side_type"),
                 Legend(("dcar_tpc_vertex", "dcar_tpc", "0", "1") + tuple(quantile_label(q) for q in QUANTILES)),
                 Headings(5))),
    Case("ADF100-01", G.fig_adf100_01_paired_x,
         "ADF100-01 two positional Y:X pairs, own min-max edges per pair, six tgl facets",
         aliases=SECTOR_ALIAS, derived={"sector_plus_quarter": lambda t: t.sector + 0.25},
         fixture=paired_x_is_sensitive,
         expect=(Branches(2),)
                + tuple(Profile(y, SELECTED, x=x, bins=20, range="minmax", by="tgl", by_bins=6,
                                plotted=branch(i, facets), tol=F16_MEAN, edges_tol=F16_EDGE)
                        for i, (y, x) in enumerate([("dcar_tpc_vertex", "sector"), ("dcar_tpc", "sector_plus_quarter")]))),
    Case("A6-VISUAL-GALLERY-FULL-01", None,
         "the readable gallery itself: run_all_pdf succeeded with all 59 pages",
         expect=(GalleryPages(59),)),
    # ---- batch B: September controls, vector/facet/weights cases, public surfaces
    Case("I4-REAL-VECTOR-FACET-ONE-ELEMENT-SELECTION-EAGER-20PCT-01", control_one_element_selection,
         "control: a one-element selection_vector filters every side_type facet",
         aliases=TIME_ALIAS,
         expect=(RowCounts(lambda t: t.eval(SIDES_A_C) & first_third(t), by="side_type", x="tgl",
                           y="dcar_tpc_vertex", range=(-1.5, 1.5),
                           plotted=lambda d: {g: c["n"] for g, c in d[2]["per_group"].items()}),)),
    Case("I4-REAL-HIST-VECTOR-FACET-SELECTION-EAGER-20PCT-01", control_hist_selection_facets,
         "control: histogram early/late branches applied inside each side_type facet",
         aliases=TIME_ALIAS,
         expect=(FacetPanels((0, 1), axes=lambda d: d[1]),)
                + tuple(Histogram("ncl", lambda t, w=window, s=side: (t.ncl > 60) & (t.side_type == s) & w(t),
                                  bins=40, range=(80.0, 160.0), label=f"side_type={side} {window.__name__}",
                                  plotted=lambda d, i=i, s=side: bar_heights(facet_panel(d[1], s), i, 40))
                        for side in (0, 1) for i, window in enumerate((early, late)))),
    Case("I4-REAL-SELECTION-VECTOR-FACET-DELTA-CORRECTNESS-EAGER-20PCT-01", G.fig22_delta_faceted,
         "G4.22 early minus late delta of dcar_tpc_vertex per sector, per side_type",
         aliases={**SECTOR_ALIAS, **TIME_ALIAS},
         expect=(Delta(Profile("dcar_tpc_vertex", lambda t: t.eval(SIDES_A_C) & early(t), **SECTOR_BINS, by="side_type",
                               plotted=None),
                       Profile("dcar_tpc_vertex", lambda t: t.eval(SIDES_A_C) & late(t), **SECTOR_BINS, by="side_type",
                               plotted=None),
                       plotted=lambda d: sorted(d[2]["normalize_data_faceted"].items()), tol=HIST12,
                       plotted_lines=overlay_lines),)),
    Case("I4-REAL-WEIGHTS-VECTOR-FACET-PROFILE-CORRECTNESS-EAGER-20PCT-01", G.fig44_weights_vector_facet_fit_oracle,
         "G11.44 two weights_vector branches (1, 1+|dcar|), tgl profile per side_type",
         derived=O2_WEIGHTS,
         expect=tuple(Profile("dcar_tpc_vertex", SIDES_A_C, x="tgl", bins=30, range=(-1.5, 1.5), by="side_type",
                              weights=w, plotted=branch(i, facets), tol=O2_VALUE, spread=True, edges_tol=HIST12)
                      for i, w in enumerate(["w_unity", "w_dca"]))
                + tuple(SameAsCall(f"weights_vector branch {i} == draw with weights='{expression}'",
                                   scalar_weight_draw(expression), branch_vs_scalar(i), tols={"pol1 parameters": FIT10})
                        for i, expression in enumerate(O2_WEIGHT_EXPRESSIONS))
                + (SameStyleAcrossFacets(2), Branches(2))),
    Case("PRECLEAN-C1-HIST-VECTOR-FACET-ACCEPTANCE-EAGER-20PCT-01", G.fig55_preclean_hist_vector_facet_acceptance,
         "G14.55 histogram early/late in each of 3 side_type facets, and the unfaceted control",
         aliases=TIME_ALIAS,
         expect=(FacetPanels((0, 1, 2), axes=lambda d: d[1]["facets"]),)
                + tuple(Histogram("ncl", lambda t, w=window, s=side: (t.ncl > 60) & (t.side_type == s) & w(t),
                                  bins=40, range=(80.0, 160.0), label=f"side_type={side} {window.__name__}",
                                  plotted=lambda d, i=i, s=side: bar_heights(facet_panel(d[1]["facets"], s), i, 40))
                        for side in (0, 1, 2) for i, window in enumerate((early, late)))
                + tuple(Histogram("ncl", lambda t, w=window: (t.ncl > 60) & (t.side_type < 3) & w(t),
                                  bins=40, range=(80.0, 160.0), label=f"control {window.__name__}",
                                  plotted=lambda d, i=i: bar_heights(d[1]["control"], i, 40))
                        for i, window in enumerate((early, late)))),
    Case("PRECLEAN-C2-DRAW-FIGURES-EFFECTIVE-DEFAULTS-EAGER-20PCT-01", G.fig56_preclean_draw_figures_effective_defaults,
         "G14.56 draw_figures short form 'known_delta:sector' with defaults: side 0 minus side 1 via the O4 subframe",
         aliases=INJECTED_ALIASES,
         expect=(Delta(Profile("known_delta", SIDES_A_C + " & (side_type == 0)", **SECTOR_BINS, plotted=None),
                       Profile("known_delta", SIDES_A_C + " & (side_type == 1)", **SECTOR_BINS, plotted=None),
                       plotted=lambda d: [(0, d[2]["stats"]["normalize_data"])], tol=HIST12),
                 Flag("caller's specs and defaults unchanged", lambda d: d[2]["caller_unchanged"]),
                 Flag("short-form plot string preserved", lambda d: d[2]["short_form_preserved"]))),
    Case("PRECLEAN-C3-RATIO-Y-VECTOR-TRUTH-EAGER-20PCT-01", G.fig57_preclean_ratio_y_vector_truth,
         "G14.57 normalize=ratio of a two-branch y vector: base / (2 x base) = 0.5 in every filled bin",
         aliases=RATIO_ALIASES,
         expect=(Delta(Profile("preclean_ratio_base", SELECTED, **SECTOR_BINS, plotted=None),
                       Profile("preclean_ratio_scaled", SELECTED, **SECTOR_BINS, plotted=None),
                       plotted=lambda d: [(0, d[2]["normalize_data"])], kind="ratio",
                       tol=C3_RATIO, signal_tol=HIST12, reference_tol=C3_REFERENCE),)),
    Case("I4-REAL-PUBLIC-SURFACE-QUALIFIED-VECTOR-EQUIV-EAGER-20PCT-01", G.fig45_public_surface_equivalence_oracle,
         "G11.45 the same subframe-qualified delta request through draw, draw_batch and draw_figures",
         expect=(SameAcrossSurfaces({"x_center": BIN_CENTRES, "value": O4, "mask_undefined": EXACT,
                                     "signal_central": O4, "signal_count": EXACT,
                                     "reference_central": O4, "reference_count": EXACT},
                                    plotted=lambda d: {k: v["normalize_data"] for k, v in d[2]["surface_stats"].items()}),
                 Delta(Profile("dcar_tpc_vertex", SIDES_A_C + " & (side_type == 0)", x="tgl", bins=30, range=(-1.5, 1.5),
                               plotted=None),
                       Profile("dcar_tpc_vertex", SIDES_A_C + " & (side_type == 1)", x="tgl", bins=30, range=(-1.5, 1.5),
                               plotted=None),
                       plotted=lambda d: [(0, d[2]["surface_stats"]["draw"]["normalize_data"])], tol=O4))),
]


def parse_args(argv: list) -> argparse.Namespace:
    """Command line of run_gallery.sh's Python stage; every inconsistent combination stops with exit 2."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root_path")
    p.add_argument("--bounded-entries", type=int, default=1_000_000, help="first N ROOT entries, unsampled")
    p.add_argument("--manifest", required=True)
    p.add_argument("--pdf", default=None)
    p.add_argument("--strict", action="store_true", help="exit 1 on any FAIL or input-identity mismatch")
    p.add_argument("--case", action="append", default=None, help="run only this case ID (repeatable)")
    p.add_argument("--input-identity", default=None, help="S0 input-identity JSON (per-branch column_digests)")
    p.add_argument("--target", action="append", default=None, help="gallery | 13_105 (repeatable); '?' lists them")
    p.add_argument("--gallery-pdf", default=None, help="PDF of the gallery target")
    args = p.parse_args(argv)
    args.targets = args.target or list(TARGETS)
    unknown_targets = sorted(set(args.targets) - set(TARGETS))
    if unknown_targets:
        p.error(f"unknown --target {unknown_targets}; known: {list(TARGETS)}")
    if "gallery" in args.targets and not args.gallery_pdf:
        p.error("--target gallery requires --gallery-pdf")
    if args.case and "13_105" not in args.targets:
        p.error("--case applies to --target 13_105 only")
    unknown = sorted(set(args.case or []) - {c.case_id for c in CASES})
    if unknown:
        p.error(f"unknown --case {unknown}")
    if args.strict and not args.input_identity:
        p.error("--strict requires --input-identity (the S0 reference the input must match)")
    return args


def check_input(adf, args) -> tuple:
    """L0 for the whole run: measured identity vs S0, loaded columns vs ROOT values, row count."""
    identity, raw = input_identity(args.root_path, args.bounded_entries)
    versus_s0 = identity_matches_s0(identity, json.load(open(args.input_identity))["column_digests"]) \
        if args.input_identity else None
    columns = loaded_equals_raw(adf, raw)
    ok = ((versus_s0 is None or versus_s0["ok"]) and all(columns.values())
          and identity["rows"] == args.bounded_entries)
    print(f"input identity: {identity['n_leaves']} leaves, combined {identity['combined_sha256'][:16]}…, "
          f"vs S0 {versus_s0}, loaded==raw {columns}, read {identity['read_seconds']} s -> {'OK' if ok else 'L0 FAIL'}")
    return ok, {"input_identity_measured": identity, "input_identity_versus_s0": versus_s0, "loaded_equals_raw": columns}


def main(argv: list | None = None) -> int:
    """Load the data once, run the requested targets on it, write the manifest, return the strict exit code."""
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--target" in argv and "?" in argv:                      # `--target ?` lists the targets and exits
        for name, what in TARGETS.items():
            print(f"{name:10s} {what}")
        return 0
    args = parse_args(argv)
    adf = G.build_adf(args.root_path, sample=None, lazy=False, entry_stop=args.bounded_entries)
    if len(adf.df) != args.bounded_entries:
        raise SystemExit(f"fixture contract: expected {args.bounded_entries} rows, loaded {len(adf.df)}")
    input_ok, input_record = check_input(adf, args)
    target_results = {}
    if "gallery" in args.targets:
        target_results["gallery"] = run_gallery_target(adf, args.gallery_pdf)
        print(f"gallery: {'OK' if target_results['gallery']['ok'] else 'FAIL'}, "
              f"{target_results['gallery']['pages']} pages, {target_results['gallery']['wall_time_s']} s")
    provenance = {"targets": target_results, **input_record,
                  "run_utc": _dt.datetime.utcnow().isoformat(timespec="seconds"), "host": platform.node(),
                  "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                  "source_sha256": code_fingerprints(adf),
                  "sample_mode": "FULL", "bounded_fixture_kind": "FIRST_N_NO_FRACTION", "entry_start": 0,
                  "entry_stop": args.bounded_entries, "sample_fraction": None,
                  "selected_rows": int(len(adf.df)), "input_path": os.path.realpath(args.root_path)}
    gallery_ok = target_results.get("gallery", {}).get("ok", True)
    RUN_CONTEXT["gallery"] = target_results.get("gallery")
    selection = args.case
    if selection is None and "gallery" not in args.targets:      # the visual case needs the gallery target
        selection = [c.case_id for c in CASES if not any(isinstance(s, GalleryPages) for s in c.expect)]
    if "13_105" in args.targets:
        code = run_cases(adf, CASES, pdf_path=args.pdf, manifest_path=args.manifest, provenance=provenance,
                         selection=selection, input_ok=input_ok and gallery_ok, mandatory=MANDATORY_CASE_IDS)
    else:   # gallery only: a manifest with no cases, carrying the target result and the input identity
        code = int(not (input_ok and gallery_ok))
        with open(args.manifest, "w") as fh:
            json.dump({"schema_version": "13.105-invariancetest-4", "provenance": provenance, "cases": [],
                       "reconciliation": {"exit_code": code, "selection": "gallery only"},
                       "not_yet_migrated": list(NOT_YET_MIGRATED), "targets": target_results},
                      fh, indent=1, sort_keys=True, default=str)
    return code if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
