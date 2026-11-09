"""Gallery invariance test for the time-series real-data gallery — declarative form.

HOW TO READ THIS FILE (navigation note, ~2 minutes)

  * Start at CASES at the bottom.  Each case names the gallery function that draws the page
    (time_series_draw.py) and DECLARES what must be true about that page:
      aliases   formulas that ADF's materialized aliases must equal exactly          (L1)
      expect    Profile(...): per-bin count and mean the plotted profile must show    (L2)
                Refuses(...): a call that must fail loudly with a named message       (L3)
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

Command line (called by run_gallery.sh; the data is loaded once for all targets):
    python time_series_draw_invariancetest.py time_series_tracks_0.root --bounded-entries 1000000 \
        --target gallery --target 13_105 --input-identity input_identity_first1M.json \
        --manifest out.json --pdf out.pdf --gallery-pdf gallery.pdf --strict
    python time_series_draw_invariancetest.py --target ?        list the targets
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
RAW_COLUMNS = ["phi", "tgl", "dcar_tpc_vertex", "ncl", "side_type"]       # ROOT branches the truth uses
SOURCE_COLUMNS = RAW_COLUMNS + ["oracle_noise", "oracle_row_id"]

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
    tol: Tol = SAME_VALUES


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


def facets(drawn) -> list:
    """Plotted tables of a faceted profile: [(facet value, profile_data), ...]."""
    stats = drawn[2]
    return [(g, stats["per_group"][str(g)]["profile_data"]) for g in stats.get("groups", ())]


def groups(stats) -> list:
    """Plotted tables of a grouped profile: [(group label, rows of profile_data), ...] in plotted order."""
    frame = stats["profile_data"]
    return [(g, frame[frame["group"] == g]) for g in pd.unique(frame["group"])]


def injected_noise_is_declared(result, t: pd.DataFrame) -> None:
    """L0: golden values at fixed row ids; mean 0 and sigma 0.02 within 5 statistical standard errors."""
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
    if case.fixture is not None:
        case.fixture(result, t)
    for name, formula in case.aliases.items():
        if name not in adf.df.columns:
            raise CheckFailure("L1", f"alias {name!r} was not materialized")
        t[name] = np.asarray(formula(t)).astype(adf.df[name].dtype, copy=False)   # only the dtype comes from ADF
        compare(result, f"alias {name} == formula", adf.df[name], t[name], EXACT, "L1")
    return t


def truth_profile(t: pd.DataFrame, spec: Profile) -> dict:
    """{group label: (count per bin, float64 mean per bin)} by pandas query/cut/groupby; bins [a, b), last bin closed."""
    r = t.query(spec.rows)
    if spec.by_bins:
        r = r[np.isfinite(r[spec.by].astype(float))]
        cut = pd.cut(r[spec.by].astype(float), spec.by_bins)
        names = [f"{iv.left:.2f}-{iv.right:.2f}" for iv in cut.cat.categories]   # dfdraw's group label format
        key = pd.Series(np.asarray(names, dtype=object)[cut.cat.codes], index=r.index)
    else:
        key = r[spec.by].astype(int) if spec.by else pd.Series(0, index=r.index)
    x, y = r[spec.x].astype(float), r[spec.y].astype(float)
    lo, hi = spec.range
    b = pd.cut(x, np.linspace(lo, hi, spec.bins + 1), right=False, labels=False)
    b[x == hi] = spec.bins - 1
    keep = b.notna() & np.isfinite(y)
    if not keep.any():
        raise CheckFailure("L0", f"selection {spec.rows!r} keeps no rows for {spec.y}")
    table = y[keep].groupby([key[keep], b[keep].astype(int)]).agg(["count", "mean"])
    labels = names if spec.by_bins else sorted(key[keep].unique())
    table = table.reindex(pd.MultiIndex.from_product([labels, range(spec.bins)])).fillna({"count": 0})
    return {g: (table.loc[g, "count"].to_numpy(int), table.loc[g, "mean"].to_numpy(float)) for g in labels}


def check_profile(t: pd.DataFrame, spec: Profile, drawn, result: CaseResult) -> None:
    """Compare the plotted tables with the truth: group labels in order, then count and mean per group (L2)."""
    truth = truth_profile(t, spec)
    plotted = spec.plotted(drawn)
    labels = [str(g) if spec.by_bins else int(float(g)) for g, _ in plotted]
    compare_labels(result, f"{spec.y} groups of {spec.by}", labels, list(truth), "L2")
    for label, (_, table) in zip(labels, plotted):
        count, mean = truth[label]
        compare(result, f"{spec.y} {spec.by}={label} count", table["count"], count, EXACT, "L2")
        compare(result, f"{spec.y} {spec.by}={label} mean", table["y_mean"], mean, spec.tol, "L2")


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
    needs_truth = case.aliases or case.fixture or any(isinstance(s, Profile) for s in case.expect)
    t = truth_frame(adf, case, result) if needs_truth else None
    for spec in case.expect:
        if isinstance(spec, Profile):
            check_profile(t, spec, drawn, result)
        elif isinstance(spec, Refuses):
            check_refusal(adf, spec, result)
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
    try:
        results = [run_case(adf, c, pdf, pages) for c in chosen]
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
    rec["exit_code"] = int(bool(rec["duplicated"] or rec["gating"] or rec["not_executed"] or not input_ok
                                or missing_mandatory or rec["n_results"] != rec["n_selected"]))
    with open(manifest_path, "w") as fh:
        json.dump({"schema_version": "13.105-invariancetest-4", "provenance": provenance,
                   "cases": [dataclasses.asdict(r) for r in results], "reconciliation": rec,
                   "not_yet_migrated": list(NOT_YET_MIGRATED), "targets": provenance.get("targets", {})},
                  fh, indent=1, sort_keys=True, default=str)
    for r in results:
        print(f"{r.status:5s} {r.failure_level:3s} {r.case_id}  {r.detail}")
    if missing_mandatory:
        print(f"FAIL  mandatory case(s) missing from the table: {missing_mandatory}")
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
    t0, digests, raw = time.perf_counter(), {}, {}
    with uproot.open(os.path.realpath(root_path)) as f:
        trees = [k for k, v in f.classnames().items() if v == "TTree"]
        if len(trees) != 1:
            raise ValueError(f"expected exactly one TTree, found {trees}")
        tree = f[trees[0].split(";")[0]]
        for name in tree.keys():
            if len(tree[name].branches) > 0:
                continue                                   # a parent branch: its leaves are digested
            a = tree[name].array(entry_start=0, entry_stop=entry_stop, library="np")
            digests[name] = leaf_digest(a)
            if name in RAW_COLUMNS:
                raw[name] = a
    combined = hashlib.sha256("".join(f"{k}:{d}:{s}\n" for k, (d, s) in sorted(digests.items())).encode()).hexdigest()
    return {"combined_sha256": combined, "leaf_digests": digests, "n_leaves": len(digests), "rows": int(len(a)),
            "read_seconds": round(time.perf_counter() - t0, 2)}, raw


def identity_matches_s0(identity: dict, s0: dict) -> dict:
    """Both directions: every reproducible S0 branch must be present with the same dtype and digest; nothing new, nothing lost.

    S0 entries recorded with dtype object (parent branches, variable-length leaves) have no reproducible value
    and are only listed; a variable-length leaf measured now is recorded as a new reference."""
    measured = identity["leaf_digests"]
    reference = {k: v for k, v in s0.items() if v["dtype"] != "object"}
    missing = sorted(k for k in reference if k not in measured)
    dtype_changed = sorted(k for k in reference if k in measured and measured[k][0] != reference[k]["dtype"])
    differ = sorted(k for k in reference if k in measured and k not in dtype_changed
                    and measured[k][1] != reference[k]["sha256"])
    unexpected = sorted(k for k in measured if k not in s0 and not measured[k][0].startswith("var*"))
    no_value = sorted(k for k in measured if k in s0 and s0[k]["dtype"] == "object")
    ok = not (missing or dtype_changed or differ or unexpected)
    return {"ok": ok, "reproducible_s0_branches": len(reference), "missing_from_input": missing,
            "dtype_changed": dtype_changed, "differ_from_s0": differ, "unexpected_leaves": unexpected,
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
# 6. The cases — the inventory first (what MUST exist), then the table (what each case declares)
# =============================================================================
MANDATORY_CASE_IDS = (
    "I4-REAL-INJECTED-TRUTH-FACET-RESIDUAL-EAGER-20PCT-01",
    "I4-REAL-GROUPING-TRUTH-EAGER-20PCT-01",
    "I2-FACET-DRAW-FIGURES-REFUSAL-01",
)

NOT_YET_MIGRATED = (   # historical case IDs still owned by the old harness; each S3 family shortens this list
    "I4-REAL-VECTOR-FACET-SUMMARYFIT-SEMANTICS-EAGER-20PCT-01",
    "I4-REAL-VECTOR-FACET-ONE-ELEMENT-SELECTION-EAGER-20PCT-01",
    "I4-REAL-HIST-VECTOR-FACET-SELECTION-EAGER-20PCT-01",
    "PRECLEAN-C1-HIST-VECTOR-FACET-ACCEPTANCE-EAGER-20PCT-01",
    "PRECLEAN-C2-DRAW-FIGURES-EFFECTIVE-DEFAULTS-EAGER-20PCT-01",
    "PRECLEAN-C3-RATIO-Y-VECTOR-TRUTH-EAGER-20PCT-01",
    "I4-REAL-SELECTION-VECTOR-FACET-DELTA-CORRECTNESS-EAGER-20PCT-01",
    "I4-REAL-PUBLIC-SURFACE-QUALIFIED-VECTOR-EQUIV-EAGER-20PCT-01",
    "I4-REAL-WEIGHTS-VECTOR-FACET-PROFILE-CORRECTNESS-EAGER-20PCT-01",
    "I4-REAL-INJECTED-TRUTH-VECTOR-OVERLAY-EAGER-20PCT-01",
    "I4-REAL-INJECTED-TRUTH-DIRECT-DELTA-EAGER-20PCT-01",
    "I4-REAL-INJECTED-TRUTH-SELECTION-DELTA-FACET-EAGER-20PCT-01",
    "I4-REAL-INJECTED-TRUTH-WEIGHTS-FACET-EAGER-20PCT-01",
    "I4-REAL-INJECTED-TRUTH-GAUSS-FIT-EAGER-20PCT-01",
    "I4-REAL-HIST-VECTOR-TRUTH-EAGER-20PCT-01",
    "I4-REAL-QUALIFIED-SUBFRAME-CHAIN-TRUTH-EAGER-20PCT-01",
    "I4-REAL-G7-SUBFRAME-EAGER-20PCT-01",
    "I4-REAL-G7-GB-EAGER-20PCT-01",
    "I4-REAL-G7-GB-SECTOR-REUSE-EAGER-20PCT-01",
    "I4-REAL-G7-GB-LOGICAL-STATE-EAGER-20PCT-01",
    "A6-VISUAL-GALLERY-FULL-01",
    "ADF99-01",
    "ADF99-02",
    "ADF100-01",
)

TARGETS = {"gallery": "the readable 59-page gallery (time_series_draw.run_all_pdf) on the loaded data",
           "13_105": "the declared cases of this file, strict"}


def migration_crosswalk(old_case_ids, new_manifest: dict) -> dict:
    """Every old-harness case must be either PASS in the new manifest or listed in NOT_YET_MIGRATED, never both, never neither."""
    new_status = {c["case_id"]: c["status"] for c in new_manifest["cases"]}
    still_old = set(new_manifest.get("not_yet_migrated", NOT_YET_MIGRATED))
    both = sorted(i for i in old_case_ids if i in still_old and i in new_status)
    neither = sorted(i for i in old_case_ids if i not in still_old and i not in new_status)
    not_passed = sorted(i for i in old_case_ids if i in new_status and new_status[i] != PASS)
    return {"ok": not (both or neither or not_passed), "migrated": sorted(i for i in old_case_ids if i in new_status),
            "still_old": sorted(i for i in old_case_ids if i in still_old), "in_both": both, "in_neither": neither,
            "migrated_but_not_pass": not_passed}


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
         expect=(Profile("oracle_residual", SIDES_A_C, **SECTOR_BINS, by="side_type", plotted=facets),
                 # physics: the plotted residual mean must equal the mean of the injected noise
                 Profile("oracle_noise", SIDES_A_C, **SECTOR_BINS, by="side_type", plotted=facets,
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
    if "13_105" in args.targets:
        code = run_cases(adf, CASES, pdf_path=args.pdf, manifest_path=args.manifest, provenance=provenance,
                         selection=args.case, input_ok=input_ok and gallery_ok, mandatory=MANDATORY_CASE_IDS)
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
