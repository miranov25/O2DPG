"""PHASE_13_101_ADF Slice C: opt-in, bounded, real-ROOT Stage-A oracles.

This is an *additive case family* for the existing PHASE_13_77 harness.  The
harness supplies CaseSpec, CaseResult, write_manifest, validate_registry and
strict_exit_code.  Never use this module as a second acceptance engine.

Commissioned first: ADF99-01, ADF99-02, ADF100-01.  Remaining seven logical
case IDs and CAL99/CAL100 are NOT claimed by this slice.  Never silently
substitute this bounded first-million input for canonical 20%-seed-42 truth.
"""

from __future__ import annotations

import hashlib
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SEL = "(ncl>60)&(abs(dcar_tpc_vertex)<10)"
SEL2 = SEL + "&(side_type<2)"
QUANTILES = (0.15, 0.50, 0.90)
QUANTILE_LABELS = ("q=15%", "median", "q=90%")
CASE_IDS = ("ADF99-01", "ADF99-02", "ADF100-01")
VERSION = "ADF101.TS_BOUNDED.v0.3"
ENTRY_STOP = 1_000_000

# Float16 physical Y columns and float32 facets require source-dtype-aware
# tolerances; counts and discrete membership are always exact.  Provisional
# numbers must be frozen against the first executed ROOT reduction before END.
MEAN_ATOL, MEAN_RTOL = 3e-5, 2e-4
Q_ATOL, Q_RTOL = 3e-4, 2e-4
EDGE_ATOL = 2e-3          # float16 x + literal expression promotion


def _figure_contract(h: Any, cid: str, *, panels: str, traces: str):
    return h.FigureContract(
        expected_panels=panels,
        panel_roles="physical tgl (and qpt_bin10) panels, in source bin order",
        expected_traces=traces,
        expected_group_count="2 physical side groups" if cid == "ADF99-02" else "1 ungrouped identity",
        primary_comparison="independent selected raw rows -> facet/group bins -> profile/quantile artists",
        residual_definition="observed profile mean/quantile minus independent raw reference",
        accepted_envelope="exact membership/counts; source-dtype-aware numerical tolerances",
        case_ids=(cid,), proof_kind="CORRECTNESS")


def cases(h: Any):
    """Declare only the three currently implemented fast real-data cases."""
    common = dict(
        owner_on_failure="dfdraw", purpose="CORRECTNESS",
        gate="CORE_MANDATORY", oracle_kind="CORRECTNESS",
        loading_mode="EAGER", sample_mode="FULL", applicable=True,
        setup_contract="one shared existing build_adf(..., sample=None, entry_stop=1_000_000)",
        preconditions=("one million ROOT entries", "selected rows and all required facets populated",
                       "physical source columns present", "one PDF page per case"),
        surfaces_under_test=("draw",),
        reference_policy="same-process",
        non_claims=("not the canonical 20%-seed-42 full-source Stage-A gate",
                    "not CAL99/CAL100", "no evidence for unimplemented ADF101 IDs"),
    )
    return (
        h.CaseSpec(
            case_id="ADF99-01", claim_id="PHASE_13_101.ADF99-01",
            title="six physical tgl facets, paired Y and discrete quantiles",
            claim="2-expression/6-facet quantiles and factored Figure legend agree with independent ROOT rows",
            failure_means="wrong facet membership, quantile value, Figure panel identity or missing quantile",
            expected_visual="six panels in 2x3 arrangement, one shared 5-identity legend",
            canonical_spec={"expr": "[dcar_tpc_vertex,dcar_tpc]:sector", "type": "profile",
                "selection": SEL, "facet_by": "tgl", "facet_by_bins": 6, "ncols": 3,
                "bins": 36, "range": (0.0, 36.0), "quantiles": list(QUANTILES),
                "quantile_mode": "discrete", "legend": "shared", "auto_title": False,
                "min_entries": 1, "return_data": True},
            figure_contract=_figure_contract(h, "ADF99-01", panels="six panels, 2x3", traces="2 Y branches x 3 quantile lines per facet"),
            observables=(
                h.Observable("quantile_bins", "INDEPENDENT", "ARRAY", "raw.quantile_bins", comparator="close",
                             atol=Q_ATOL, rtol=Q_RTOL, rationale="source float16 and quantile reductions"),
                h.Observable("legend_identifiers", "INDEPENDENT", "ARRAY", "Figure.legend_labels")),
            negative_control="remove q90 Figure-legend identity and require L3 FAIL", **common),
        h.CaseSpec(
            case_id="ADF99-02", claim_id="PHASE_13_101.ADF99-02",
            title="3x2 tgl/qpt facets with global side grouping, vector and quantiles",
            claim="raw facet/group quantiles and 2+2+3 factored legend identities are preserved",
            failure_means="group/facet partition, quantile curves, outer headings or factored legend wrong",
            expected_visual="3x2 facets, five outer headings, one seven-identity shared legend",
            canonical_spec={"expr": "[dcar_tpc_vertex,dcar_tpc]:sector", "type": "profile",
                "selection": SEL2, "facet_by": ["tgl", "qpt_bin10"], "facet_by_bins": [3, 2],
                "group_by": "side_type", "group_by_scope": "global", "bins": 36,
                "range": (0.0, 36.0), "quantiles": list(QUANTILES),
                "quantile_mode": "discrete", "legend": "shared", "facet_labels": "outer",
                "auto_title": False, "min_entries": 1, "return_data": True},
            figure_contract=_figure_contract(h, "ADF99-02", panels="six panels, 3x2", traces="2 Y x 2 groups x 3 quantile lines"),
            observables=(
                h.Observable("group_quantiles", "INDEPENDENT", "ARRAY", "raw.facet.group.quantiles", comparator="close",
                             atol=Q_ATOL, rtol=Q_RTOL, rationale="source float16 and quantile reductions"),
                h.Observable("factored_legend", "INDEPENDENT", "ARRAY", "Figure.legend_labels")),
            negative_control="duplicate group identity in legend and require L3 FAIL", **common),
        h.CaseSpec(
            case_id="ADF100-01", claim_id="PHASE_13_101.ADF100-01",
            title="paired-x Figure-owned bin edges and independent raw profile means",
            claim="two positional X/Y pairs retain independent Figure-domain bin edges and raw counts/means",
            failure_means="wrong paired X substitution, shared edges, per-facet edges or profile contents",
            expected_visual="six facets with two paired X/Y branches and discrete quantiles",
            canonical_spec={"expr": "[dcar_tpc_vertex,dcar_tpc]:[sector,sector+0.25]",
                "type": "profile", "selection": SEL, "facet_by": "tgl", "facet_by_bins": 6,
                "ncols": 3, "bins": 20, "range": "minmax",
                "quantiles": list(QUANTILES), "quantile_mode": "discrete", "legend": "shared",
                "auto_title": False, "min_entries": 1, "return_data": True},
            figure_contract=_figure_contract(h, "ADF100-01", panels="six panels, 2x3", traces="two positional Y:X pairs"),
            observables=(
                h.Observable("paired_counts", "INDEPENDENT", "ARRAY", "raw.facet.pair.count"),
                h.Observable("paired_means", "INDEPENDENT", "ARRAY", "raw.facet.pair.mean", comparator="close",
                             atol=MEAN_ATOL, rtol=MEAN_RTOL, rationale="source float16 Y summation"),
                h.Observable("paired_edges", "INDEPENDENT", "ARRAY", "raw.pair.Figure.edges", comparator="close",
                             atol=EDGE_ATOL, rtol=0, rationale="expression float16 x promotion")),
            negative_control="substitute branch 0 X for branch 1; require L2 FAIL", **common),
    )


def _assert_numbered(name: str, actual: Any, expected: Any, *, atol=0.0, rtol=0.0) -> int:
    got = np.asarray(actual)
    want = np.asarray(expected)
    if got.shape != want.shape:
        raise AssertionError(f"{name}: shape {got.shape} != {want.shape}")
    np.testing.assert_allclose(got, want, atol=atol, rtol=rtol, equal_nan=True, err_msg=name)
    return int(got.size)


def _quantile_raw(x: np.ndarray, y: np.ndarray, edges: np.ndarray, q: float) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    result = np.full(len(edges)-1, np.nan, dtype=np.float64)
    for bin_i in range(len(result)):
        lo, hi = edges[bin_i:bin_i+2]
        inside = (x >= lo) & (x < hi if bin_i < len(result)-1 else x <= hi)
        if inside.any():
            result[bin_i] = np.nanpercentile(y[inside], 100*q, method="linear")
    return result


def _profile_raw(x: np.ndarray, y: np.ndarray, edges: np.ndarray):
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    counts, means = [], []
    for i in range(len(edges) - 1):
        lo, hi = edges[i:i+2]
        inside = (x >= lo) & (x < hi if i < len(edges)-2 else x <= hi)
        z = y[inside]
        counts.append(int(len(z)))
        means.append(float(np.mean(z, dtype=np.float64)) if len(z) else np.nan)
    return np.asarray(counts, dtype=np.int64), np.asarray(means)


def _raw_data(adf):
    """Independent L0 inputs: physical data and source-defined qpt binning.

    ``qpt_bin10`` is a *lazy ADF alias*, not a required ROOT/pandas column.
    Its documented definition in ``time_series.py`` is ``10*qpt`` with
    ``dtype=np.int8``.  Build the expected bin values from the physical
    ``qpt`` branch rather than copying the value returned by dfdraw/ADF.

    A pre-materialized ``qpt_bin10`` is accepted only as a legacy test-double
    fallback when ``qpt`` is absent.  The real TS_ROOT fixture must carry
    ``qpt``; a broken product alias cannot redefine independent raw truth.
    """
    df = adf.df
    physical = ("sector", "tgl", "side_type", "dcar_tpc_vertex", "dcar_tpc", "ncl")
    missing = [name for name in physical if name not in df.columns]
    if missing:
        raise ValueError("INVALID_FIXTURE: physical input columns missing: " + ", ".join(missing))
    raw = {name: np.asarray(df[name]) for name in physical}

    if "qpt" in df.columns:
        qpt = np.asarray(df["qpt"])
        if not np.issubdtype(qpt.dtype, np.number):
            raise ValueError("INVALID_FIXTURE: qpt physical branch is nonnumeric")
        # Match the *declared expression and dtype*, not ADF's evaluated alias.
        # Cast-to-int8 truncation is part of the published TS alias contract.
        with np.errstate(invalid="ignore", over="ignore"):
            raw["qpt_bin10"] = np.asarray(10 * qpt).astype(np.int8)
    elif "qpt_bin10" in df.columns:
        # Supports existing test doubles that provide a fully materialized
        # synthetic column rather than a physical qpt branch.
        raw["qpt_bin10"] = np.asarray(df["qpt_bin10"])
    else:
        raise ValueError(
            "INVALID_FIXTURE: physical qpt branch absent and qpt_bin10 alias "
            "not materialized")
    return raw


def _selected(raw, two_groups: bool):
    keep = ((raw["ncl"] > 60) & (np.abs(raw["dcar_tpc_vertex"]) < 10))
    if two_groups:
        keep &= raw["side_type"] < 2
    if int(keep.sum()) == 0:
        raise ValueError("INVALID_FIXTURE: zero selected rows")
    return keep


def _facets(raw, selected, grouped: bool):
    # Exact pd.cut contracts follow the selected physical dtype, as the live
    # dfdraw facet dispatcher does.  Never cut unselected rows and reuse bins.
    tgl = pd.cut(pd.Series(raw["tgl"][selected]), bins=3 if grouped else 6)
    if grouped:
        qpt = pd.cut(pd.Series(raw["qpt_bin10"][selected]), bins=2)
        labels = [(a,b) for a in tgl.cat.categories for b in qpt.cat.categories]
        masks = [np.asarray((tgl == a) & (qpt == b)) for a,b in labels]
    else:
        labels = list(tgl.cat.categories)
        masks = [np.asarray(tgl == a) for a in labels]
    if len(masks) != 6 or any(int(m.sum()) == 0 for m in masks):
        raise ValueError("INVALID_FIXTURE: fewer than six populated physical facet cells")
    return labels, masks


def _normal_legend_labels(labels):
    # dfdraw can display the public Y expression as "name vs sector" rather
    # than just "name"; those are the same expression identity, not two.
    return [item.split(" vs sector", 1)[0] if item.startswith(
        ("dcar_tpc_vertex vs sector", "dcar_tpc vs sector")) else item
            for item in labels]


def _check_factored_labels(labels, expected):
    labels = _normal_legend_labels(labels)
    if len(labels) != len(expected):
        raise AssertionError(f"L3: factored legend count {labels} != {expected}")
    for name in expected:
        if labels.count(name) != 1:
            raise AssertionError(f"L3: missing/duplicated legend {name}: {labels}")
    return len(labels)


def _figure_legend(fig, expected):
    if len(fig.legends) != 1:
        raise AssertionError(f"L3: expected one Figure shared legend, saw {len(fig.legends)}")
    labels = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
    return _check_factored_labels(labels, expected), labels


def _legend_falsifier(labels, expected, grouped):
    # Prove the same actual comparator rejects a deliberate legend mutation.
    broken = list(labels)
    if grouped:
        # Duplicate group 0, losing group 1 (never a Cartesian identity).
        normalized = _normal_legend_labels(broken)
        at = normalized.index("1")
        broken[at] = "0"
    else:
        normalized = _normal_legend_labels(broken)
        at = normalized.index("q=90%")
        del broken[at]
    try:
        _check_factored_labels(broken, expected)
    except AssertionError:
        return True
    raise AssertionError("L0: deliberate legend falsifier was not detected")


def _public_interval_label(interval: pd.Interval) -> str:
    """Independently reproduce the documented public 1-D facet label grammar.

    The source interval remains the reference for raw membership.  The
    presentation uses width-dependent decimal precision, *not* str(Interval).
    Never infer membership from a rounded display label.
    """
    if not isinstance(interval, pd.Interval):
        raise AssertionError(f"L0: expected physical pandas interval, got {interval!r}")
    width = float(interval.right - interval.left)
    decimals = 0 if width >= 10 else 1 if width >= 1 else 2 if width >= .1 else 3
    return f"{interval.left:.{decimals}f}-{interval.right:.{decimals}f}"


def _legacy_ungrouped_quantile_lines(ax):
    """Identify actual public ungrouped vector-quantile artists, fail closed.

    The scalar ungrouped profile renderer labels Line2D artists q=15%, median,
    q=90%, but historically does not attach the private semantic metadata used
    by group_by.  The public vector dispatcher draws complete Y branches in
    source order.  Require the entire 2*3 quantile sequence, rather than
    selecting arbitrary matching lines or guessing from colors.
    """
    tagged = [line for line in ax.lines
              if isinstance(getattr(line, "_dfdraw_facet_semantic", None), dict)
              and getattr(line, "_dfdraw_facet_semantic").get("kind") == "quantile"]
    if tagged:
        raise AssertionError("L3: partial/mixed semantic quantile metadata on ungrouped panel")
    lines = [line for line in ax.lines if str(line.get_label()) in QUANTILE_LABELS]
    expected_labels = list(QUANTILE_LABELS) * 2
    seen = [str(line.get_label()) for line in lines]
    if seen != expected_labels:
        raise AssertionError(f"L3: ungrouped vector quantile artist order {seen!r} != {expected_labels!r}")
    return lines


def _semantic_quantile_line(ax, expr: str, q: float, group=None):
    matches = []
    for line in ax.lines:
        meta = getattr(line, "_dfdraw_facet_semantic", None)
        if not isinstance(meta, dict) or meta.get("kind") != "quantile":
            continue
        if str(meta.get("expression")) != expr:
            continue
        if not np.isclose(float(meta.get("quantile", -1)), q, rtol=0, atol=1e-12):
            continue
        if group is not None and str(meta.get("group")) != str(group):
            continue
        matches.append(line)
    if not matches and group is None:
        # The 1-D ungrouped public renderer has no private semantic tags;
        # identify its six *real* quantile artists using a complete checked
        # branch-major label sequence. The independent raw quantile arrays
        # are still compared to each selected Line2D below.
        expressions = ("dcar_tpc_vertex", "dcar_tpc")
        if expr not in expressions or q not in QUANTILES:
            raise AssertionError(f"L3: unknown expression/quantile identity: {expr!r}, {q!r}")
        lines = _legacy_ungrouped_quantile_lines(ax)
        matches = [lines[expressions.index(expr) * len(QUANTILES) + QUANTILES.index(q)]]
    if len(matches) != 1:
        raise AssertionError(
            f"L3: exactly one semantic quantile line expected: {expr}, q={q}, "
            f"group={group}; found {len(matches)}")
    return matches[0]


def _one_case(adf, case, raw, pdf):
    """Run one public request and compare independent ROOT rows against artists/stats."""
    import matplotlib.pyplot as plt
    t0 = time.perf_counter()
    fig = None
    result = dict(status="FAIL", comparisons=0, first_layer="L1", owner="ADF")
    try:
        spec = dict(case.canonical_spec)
        expr = spec.pop("expr")
        fig, axes, stats = adf.draw(expr, **spec)
        result["first_layer"] = "L2"
        result["owner"] = "dfdraw"
        panels = list(np.asarray(axes, dtype=object).flat)
        if len(panels) != 6:
            raise AssertionError(f"L3: expected 6 axes, got {len(panels)}")
        is_grouped = case.case_id == "ADF99-02"
        selected = _selected(raw, is_grouped)
        labels, masks = _facets(raw, selected, is_grouped)
        x0 = np.asarray(raw["sector"][selected], dtype=np.float64)
        ys = (np.asarray(raw["dcar_tpc_vertex"][selected], dtype=np.float64),
              np.asarray(raw["dcar_tpc"][selected], dtype=np.float64))
        selected_groups = np.asarray(raw["side_type"][selected])
        comparisons = 0

        if case.case_id == "ADF100-01":
            if not isinstance(stats, list) or len(stats) != 2:
                raise AssertionError("L2: expected 2 paired profile result branches")
            # Figure-domain edges are branch-specific, and the branch-1 X is
            # independently evaluated from the physical ROOT sector column.
            # Preserve the physical sector dtype when evaluating the public
            # scalar expression; float16 sector + 0.25 is generally still
            # float16 in NumPy/pandas, unlike x0.astype(float64) + 0.25.
            # This remains independent of dfdraw's returned statistics.
            typed_shift = np.asarray(raw["sector"][selected] + 0.25, dtype=np.float64)
            x_pairs = (x0, typed_shift)
            edge_pairs = []
            # Exact source intervals own physical membership, while the 1-D
            # public stats API uses a rounded human-readable label as key.
            # Check *each* public key against its independent raw interval;
            # never replace these assertions by a positional-only comparison.
            public_facet_keys = [_public_interval_label(facet) for facet in labels]
            if len(set(public_facet_keys)) != len(labels):
                raise ValueError("INVALID_FIXTURE: public formatted facet labels collide")
            for branch_i, (x, y) in enumerate(zip(x_pairs, ys)):
                finite = np.isfinite(x) & np.isfinite(y)
                if not finite.any():
                    raise ValueError("INVALID_FIXTURE: no finite paired X/Y rows")
                edges = np.linspace(x[finite].min(), x[finite].max(), 21)
                edge_pairs.append(edges)
                branch = stats[branch_i]
                members = list(branch["groups"])
                if len(members) != 6:
                    raise AssertionError("L2: profile branch missing physical facets")
                for j, (facet, local) in enumerate(zip(labels, masks)):
                    member = members[j]
                    if str(member) != public_facet_keys[j]:
                        raise AssertionError(
                            f"L0/L2: facet identity {member!r} != "
                            f"raw {facet!r} displayed as {public_facet_keys[j]!r}")
                    profile = branch["per_group"][member]["profile_data"]
                    expected_n, expected_mean = _profile_raw(x[local], y[local], edges)
                    comparisons += _assert_numbered("L2.count", profile["count"], expected_n)
                    comparisons += _assert_numbered("L2.mean", profile["y_mean"], expected_mean,
                                                    atol=MEAN_ATOL, rtol=MEAN_RTOL)
                    obtained_edges = np.r_[np.asarray(profile["x_low"]), np.asarray(profile["x_high"])[-1]]
                    comparisons += _assert_numbered("L2.Figure-owned edges", obtained_edges, edges,
                                                    atol=EDGE_ATOL)
            if np.allclose(edge_pairs[0], edge_pairs[1], atol=0, rtol=0):
                raise AssertionError("L2: positional branches have erroneously shared X edges")
            # Real source-sensitive falsifier: substitute x0 for the shifted x1
            # and prove a populated-bin difference in a raw distribution.
            wrong_n, wrong_mean = _profile_raw(x_pairs[0], ys[1], edge_pairs[1])
            right_n, right_mean = _profile_raw(x_pairs[1], ys[1], edge_pairs[1])
            if np.array_equal(wrong_n, right_n) and np.allclose(
                    wrong_mean, right_mean, equal_nan=True, atol=MEAN_ATOL, rtol=MEAN_RTOL):
                raise ValueError("INVALID_FIXTURE: wrong-X falsifier is insensitive")
            result["falsifier"] = {"mutation_id": "swap_branch_1_x", "detected": True}

        else:
            edges = np.linspace(0.0, 36.0, 37)
            group_list = (0, 1) if is_grouped else (None,)
            for j, (facet, cell) in enumerate(zip(labels, masks)):
                if np.count_nonzero(cell) < 36:
                    raise ValueError(f"INVALID_FIXTURE: underoccupied cell {facet}")
                for group in group_list:
                    local = cell & (selected_groups == group) if is_grouped else cell
                    if not local.any():
                        # Global-group identity survives even where a physical
                        # cell has no rows for this side.  A sparse cell is
                        # not an INVALID_FIXTURE; compare populated groups.
                        continue
                    for expr_i, expression in enumerate(("dcar_tpc_vertex", "dcar_tpc")):
                        for q in QUANTILES:
                            line = _semantic_quantile_line(panels[j], expression, q, group)
                            expected = _quantile_raw(x0[local], ys[expr_i][local], edges, q)
                            centers = (edges[1:] + edges[:-1]) / 2.0
                            comparisons += _assert_numbered("L2.quantile x centers", line.get_xdata(), centers,
                                                            atol=EDGE_ATOL)
                            comparisons += _assert_numbered("L2.raw quantiles", line.get_ydata(), expected,
                                                            atol=Q_ATOL, rtol=Q_RTOL)
            expected_labels = ["dcar_tpc_vertex", "dcar_tpc"]
            if is_grouped:
                expected_labels.extend(("0", "1"))
            expected_labels.extend(QUANTILE_LABELS)
            result["first_layer"] = "L3"
            leg_count, observed_labels = _figure_legend(fig, expected_labels)
            comparisons += leg_count
            if is_grouped:
                headings = [
                    t.get_text() for t in fig.texts
                    if t.get_text().startswith(("(", "[")) and "'" not in t.get_text() and '"' not in t.get_text()
                ]
                if len(headings) != 5 or len(set(headings)) != 5:
                    raise AssertionError(f"L3: expected 5 distinct outer headings, got {headings}")
                comparisons += 5
            result["falsifier"] = {
                "mutation_id": "remove_q90_identity" if not is_grouped else "duplicate_group_legend_identity",
                "detected": _legend_falsifier(observed_labels, expected_labels, is_grouped),
            }

        result.update(status="PASS", comparisons=comparisons, first_layer="NONE", owner="NONE")
    except Exception as exc:
        if isinstance(exc, ValueError) and "INVALID_FIXTURE" in str(exc):
            result["status"] = "INVALID_FIXTURE"
            result["first_layer"] = "L0"
            result["owner"] = "environment"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc(limit=7)
    finally:
        if fig is not None:
            pdf.savefig(fig)  # Preserve FAILED pages as inspectable evidence.
            plt.close(fig)
        result["wall_time_s"] = round(time.perf_counter() - t0, 4)
    return result


def run_fast(root_path: str, *, manifest_path: str, pdf_path: str, h: Any, gallery_module=None):
    """Use the Stage-A registry/manifest/strict machinery on a first-1M fixture."""
    from matplotlib.backends.backend_pdf import PdfPages

    declared = list(cases(h))
    violations = h.validate_registry(declared)
    if violations:
        raise h.HarnessError(f"ADF101 case registry invalid: {violations}")
    gallery = gallery_module if gallery_module is not None else h._a5_2_import_gallery()
    started = time.perf_counter()
    adf = None
    build_error = None
    try:
        adf = gallery.build_adf(root_path, sample=None, lazy=False, entry_stop=ENTRY_STOP)
        if len(adf.df) != ENTRY_STOP:
            raise ValueError(f"INVALID_FIXTURE: read {len(adf.df)} != {ENTRY_STOP} entries")
        raw = _raw_data(adf)
    except Exception as exc:
        build_error = f"{type(exc).__name__}: {exc}"

    results = []
    with PdfPages(pdf_path) as pdf:
        for c in declared:
            if build_error is not None:
                result = h.CaseResult(case_id=c.case_id, status=h.INVALID_FIXTURE,
                                      detail=build_error, observed={"first_failed_layer": "L0",
                                      "owner_status": "environment", "comparison_count": 0})
            else:
                got = _one_case(adf, c, raw, pdf)
                result = h.CaseResult(
                    case_id=c.case_id, status=got["status"],
                    detail=got.get("error", "independent raw comparisons passed"),
                    observed={"comparison_count": got["comparisons"],
                              "first_failed_layer": got["first_layer"],
                              "owner_status": got["owner"],
                              "falsifier": got.get("falsifier", {}),
                              "contract_version": VERSION},
                    executed_comparisons=int(got["comparisons"]),
                    wall_time_s=float(got["wall_time_s"]),
                    exception=got.get("traceback", ""))
            results.append(result)
            print(f"[ADF101] {c.case_id}: {result.status}, comparisons={result.executed_comparisons}, "
                  f"first={result.observed.get('first_failed_layer')} {result.detail}", flush=True)

    try:
        st = os.stat(root_path)
    except OSError:
        st = None
    pdf_file = Path(pdf_path)
    pdf_hash = hashlib.sha256(pdf_file.read_bytes()).hexdigest() if pdf_file.is_file() else None
    sample_digest = hashlib.sha256(np.arange(ENTRY_STOP, dtype="<i8").tobytes()).hexdigest()
    metadata = {
        "stage_a_gate": "ADF101_FAST_TS_BOUNDED",
        "fixture_kind": "TS_ROOT_BOUNDED", "root_path": os.path.abspath(root_path),
        "root_size_bytes": int(st.st_size) if st is not None else None,
        "root_mtime_ns": int(st.st_mtime_ns) if st is not None else None,
        "entry_start": 0, "entry_stop": ENTRY_STOP,
        "sample_mode": "FULL", "bounded_fixture_kind": "FIRST_N_NO_FRACTION",
        "sampled_source_index_sha256": sample_digest,
        "test_registry_version": VERSION, "declared_ids": list(CASE_IDS),
        "canonical_fraction_20pct_unchanged": True,
        "current_slice_only_not_phase_closure": True,
        "pdf_path": os.path.abspath(pdf_path),
        "pdf_sha256": pdf_hash,
        "pdf_generated": pdf_hash is not None,
        "adf_build_count": 0 if adf is None else 1,
        "elapsed_wall_s": round(time.perf_counter()-started, 3),
    }
    document = h.write_manifest(manifest_path, results, declared, extra=metadata)
    return results, document, h.strict_exit_code(results, declared)
