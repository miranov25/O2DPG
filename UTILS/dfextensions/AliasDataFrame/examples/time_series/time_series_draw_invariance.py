"""PHASE_13_77_ADF Stage A — real-data invariance harness.

A1: surface adapter, shape-aware observable accessor, CaseSpec, registry,
strict runner, manifest writer.

ONE-WAY COMPANION.  This module imports ``time_series_draw`` and never the
other way round.  It changes no ADF, dfdraw or GB production code.

The four functions the governing ruling requires be kept separate:

    adapter                    proves WHERE the public result is
    surface-consistency oracle proves whether public surfaces AGREE
    independent oracle         proves whether the result is CORRECT
    KNOWN_BUG / NOT_APPLICABLE records WHY a surface cannot yet participate

Governing documents
    PHASE_13_77_ADF_..._Proposal_v1_2.md
    PHASE_13_77_ADF_A1_STATS_SCHEMA_SURVEY_a1_v01.md   (measured schema)
    GPT32:ADF consolidated Stage-A decision, 2026-08-25
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Sequence

import numpy as np

SCHEMA_VERSION = "13.77.A4.3.v01"

# ─────────────────────────────────────────────────────────────────────────────
# Enumerations.  Plain strings: they are serialised into the manifest, and a
# str survives a JSON round trip where an Enum member does not.  Same reasoning
# as DTypeOrigin in AliasDataFrame (B3.2b STEP 1).
# ─────────────────────────────────────────────────────────────────────────────

# Terminal statuses.  PASS and INVALID_FIXTURE must never be confused: a
# contaminated or unresolvable fixture is not a product failure.
PASS, FAIL, INVALID_FIXTURE, SKIP, DIAGNOSTIC = (
    "PASS", "FAIL", "INVALID_FIXTURE", "SKIP", "DIAGNOSTIC")

PURPOSE = ("INVARIANCE", "CORRECTNESS", "VISUAL_DIAGNOSTIC",
           "COVERAGE", "ERROR_CONTRACT", "PERFORMANCE_OBSERVATION")
GATE = ("CORE_MANDATORY", "ENVIRONMENT_GATED")
ORACLE_KIND = ("CONSISTENCY", "CORRECTNESS")
ORACLE_SOURCE = ("STATS", "INDEPENDENT", "ARTIST_FALLBACK")
# A1-P1-2: the runner must EXECUTE the source a case DECLARES.  ARTIST_FALLBACK
# is declarable in the schema but no artist extractor exists yet, so a case
# declaring it is refused at registry time rather than silently reading stats.
IMPLEMENTED_SOURCES = ("STATS", "INDEPENDENT")
ACCESS = ("FLAT", "NESTED", "PER_GROUP", "ARRAY")
OBS_STATUS = ("EXECUTED", "NOT_EXTRACTABLE")      # DEFERRED:<stage> also allowed
KNOWN_BUG_STATUS = ("SUPPORTED", "KNOWN_BUG", "EXPECTED_FAIL", "ENVIRONMENT_BLOCKED")
LOADING_MODE = ("EAGER", "LAZY", "BOTH")
SAMPLE_MODE = ("FULL", "FRACTION")

SURFACES = ("draw", "draw_batch", "draw_figures")


class HarnessError(RuntimeError):
    """Harness-side failure.  Never raised for a product defect."""


class AdapterError(HarnessError):
    """The public result did not have the shape this surface must return.

    Raised loudly and never swallowed.  The forbidden behaviour the ruling
    names explicitly is: wrong path -> empty dict -> PASS.  This exception is
    what makes that impossible.
    """


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Surface adapter — a PURE envelope unwrapper.
#
# It normalises WHERE the payload lives.  It must not normalise WHAT the
# payload means: no renaming, no invented fields, no defaults, no flattening
# of faceted stats into flat stats, no reinterpretation of one schema as
# another.  Measured envelopes (survey a1_v01 §2, re-probed 2026-08-25):
#
#     draw          -> (Figure, Axes, stats)          payload = r[2]
#     draw_batch    -> {case: {ax,fig,path,stats}, _errors, _summary}
#                                                     payload = r[case]["stats"]
#     draw_figures  -> {figure_0: {fig, axes, stats}} payload = r[fig]["stats"][i]
#                      stats is a LIST, one entry per plot
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Payload:
    """An unwrapped public result plus the path used to reach it.

    ``path`` is recorded in the manifest so a reviewer can verify the adapter
    itself rather than trusting it.
    """
    surface: str
    stats: Any
    path: tuple


def unwrap(surface: str, result: Any, *,
           case_key: str | None = None,
           figure_key: str = "figure_0",
           plot_index: int = 0) -> Payload:
    """Unwrap a public result to its stats payload.  Pure; raises on mismatch."""
    if surface == "draw":
        if not isinstance(result, tuple) or len(result) < 3:
            raise AdapterError(
                f"draw() must return a 3-tuple (Figure, Axes, stats); got "
                f"{type(result).__name__}"
                + (f" of length {len(result)}" if isinstance(result, tuple) else ""))
        return Payload("draw", result[2], ("[2]",))

    if surface == "draw_batch":
        if not isinstance(result, dict):
            raise AdapterError(
                f"draw_batch() must return a dict; got {type(result).__name__}")
        if case_key is None:
            raise AdapterError("draw_batch unwrap requires case_key")
        if case_key not in result:
            raise AdapterError(
                f"draw_batch result has no case {case_key!r}; "
                f"present: {sorted(k for k in result if not k.startswith('_'))}")
        entry = result[case_key]
        if not isinstance(entry, dict) or "stats" not in entry:
            raise AdapterError(
                f"draw_batch case {case_key!r} has no 'stats'; "
                f"keys: {sorted(entry) if isinstance(entry, dict) else type(entry).__name__}")
        return Payload("draw_batch", entry["stats"], (case_key, "stats"))

    if surface == "draw_figures":
        if not isinstance(result, dict):
            raise AdapterError(
                f"draw_figures() must return a dict; got {type(result).__name__}")
        if figure_key not in result:
            raise AdapterError(
                f"draw_figures result has no {figure_key!r}; present: {sorted(result)}")
        fig = result[figure_key]
        if not isinstance(fig, dict) or "stats" not in fig:
            raise AdapterError(
                f"draw_figures {figure_key!r} has no 'stats'; "
                f"keys: {sorted(fig) if isinstance(fig, dict) else type(fig).__name__}")
        stats_list = fig["stats"]
        if not isinstance(stats_list, (list, tuple)):
            raise AdapterError(
                f"draw_figures stats must be a list (one entry per plot); "
                f"got {type(stats_list).__name__}")
        if not 0 <= plot_index < len(stats_list):
            raise AdapterError(
                f"draw_figures plot_index {plot_index} out of range "
                f"(len={len(stats_list)})")
        return Payload("draw_figures", stats_list[plot_index],
                       (figure_key, "stats", plot_index))

    raise AdapterError(f"unknown surface {surface!r}; known: {SURFACES}")


def batch_errors(result: Any) -> list:
    """draw_batch reports per-case errors out of band; a case must not PASS
    while its own surface recorded an error."""
    if isinstance(result, dict):
        errs = result.get("_errors")
        if errs:
            return list(errs)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Shape-aware observable accessor.
#
# Survey a1_v01 §3: a faceted result carries 7 nested keys and NONE of the 20
# flat statistical keys.  A flat accessor would resolve nothing, the comparator
# would compare two absences, and the case would PASS proving nothing.  The
# accessor therefore RAISES rather than returning a default.
# ─────────────────────────────────────────────────────────────────────────────

_MISSING = object()


# Which oracle source each runner ACTUALLY executes.  A1-v04-P0-2: v04
# validated that a declared source was *implementable* and never checked it
# against the path the runner takes, so an observable could be labelled
# INDEPENDENT and be resolved from public stats.  A label that does not
# describe the executed path is worse than no label.
RUNNER_SOURCE = {
    "run_consistency": ("STATS",),
    "run_correctness": ("INDEPENDENT",),
    "run_slot_symmetry": ("STATS",),
}


def _assert_source_matches(runner: str, o: "Observable") -> None:
    allowed = RUNNER_SOURCE[runner]
    if o.source not in allowed:
        raise HarnessError(
            f"observable {o.name!r} declares source {o.source!r}, but {runner} "
            f"executes {allowed[0]}; the declaration would describe a proof "
            f"this case never performs")


def resolve(stats: Any, path: str, access: str) -> Any:
    """Resolve a dotted observable path against a measured payload.

    Raises HarnessError if the path does not resolve.  Never returns a default:
    an unresolvable observable is INVALID_FIXTURE, not PASS.
    """
    if access not in ACCESS:
        raise HarnessError(f"unknown access {access!r}; known: {ACCESS}")
    cur = stats
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part, _MISSING)
        elif isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                cur = _MISSING
        elif hasattr(cur, "columns") and hasattr(cur, "__getitem__"):
            # A3.6: grouped profile ``return_data=True`` exposes measured
            # per-group/per-bin observables as a pandas DataFrame.  Resolve a
            # named column into a NumPy array so the existing A2 array
            # comparator remains the sole numerical comparison mechanism.
            if part in cur.columns:
                column = cur[part]
                cur = (column.to_numpy(copy=True)
                       if hasattr(column, "to_numpy") else np.asarray(column))
            else:
                cur = _MISSING
        else:
            cur = _MISSING
        if cur is _MISSING:
            raise HarnessError(
                f"observable path {path!r} does not resolve at {part!r}; "
                f"available: "
                f"{sorted(stats) if isinstance(stats, dict) else type(stats).__name__}")
    if access == "ARRAY" and not hasattr(cur, "__len__"):
        raise HarnessError(f"{path!r} declared ARRAY but resolved to "
                           f"{type(cur).__name__}")
    if access in ("FLAT",) and hasattr(cur, "__len__") and not isinstance(cur, str):
        raise HarnessError(f"{path!r} declared FLAT but resolved to "
                           f"{type(cur).__name__}")
    return cur


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Comparators
# ─────────────────────────────────────────────────────────────────────────────

COMPARATORS = ("exact", "close")


@dataclass(frozen=True)
class ComparisonResult:
    """Structured outcome of one A2 numerical comparison."""
    ok: bool
    comparator: str
    detail: str = ""
    atol: float = 0.0
    rtol: float = 0.0
    mismatch_count: int = 0
    mismatch_indices: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class ToleranceSpec:
    """Validated executable tolerance contract for one observable."""
    comparator: str
    atol: float = 0.0
    rtol: float = 0.0
    rationale: str = ""


def _validated_tolerance(comparator: str, atol: float, rtol: float,
                         rationale: str, *, observable_name: str) -> ToleranceSpec:
    try:
        atol_f = float(atol)
        rtol_f = float(rtol)
    except (TypeError, ValueError) as exc:
        raise HarnessError(
            f"observable {observable_name!r}: atol/rtol must be numeric") from exc
    if comparator not in COMPARATORS:
        raise HarnessError(
            f"observable {observable_name!r}: unknown comparator {comparator!r}; "
            f"known: {COMPARATORS}")
    if not np.isfinite(atol_f) or not np.isfinite(rtol_f):
        raise HarnessError(
            f"observable {observable_name!r}: atol/rtol must be finite")
    if atol_f < 0.0 or rtol_f < 0.0:
        raise HarnessError(
            f"observable {observable_name!r}: atol/rtol must be non-negative")
    if comparator == "exact":
        if atol_f != 0.0 or rtol_f != 0.0:
            raise HarnessError(
                f"observable {observable_name!r}: exact comparator cannot carry "
                f"non-zero atol/rtol")
    else:
        if atol_f == 0.0 and rtol_f == 0.0:
            raise HarnessError(
                f"observable {observable_name!r}: 'close' with atol=rtol=0 is "
                f"'exact' in disguise")
        if not rationale:
            raise HarnessError(
                f"observable {observable_name!r}: floating comparator needs a rationale")
    return ToleranceSpec(comparator, atol_f, rtol_f, rationale)


def tolerance_for(o: "Observable") -> ToleranceSpec:
    return _validated_tolerance(
        o.comparator, o.atol, o.rtol, o.rationale, observable_name=o.name)


def tolerance_violations(o: "Observable") -> list[str]:
    try:
        tolerance_for(o)
    except HarnessError as exc:
        return [str(exc)]
    return []


def _real_floating_scalar(value: Any) -> bool:
    """True only for scalar real floating values accepted by ``close``.

    A2 tolerance comparisons are defined for floating observables.  Integer,
    complex and object inputs must use an explicit supported comparator rather
    than being silently narrowed/coerced before comparison.
    """
    if np.ndim(value) != 0:
        return False
    try:
        return bool(np.issubdtype(np.asarray(value).dtype, np.floating))
    except TypeError:
        return False


def compare_scalar(a: Any, b: Any, *, comparator: str,
                   atol: float = 0.0, rtol: float = 0.0) -> ComparisonResult:
    """Compare scalar values with explicit exact or atol/rtol semantics."""
    spec = _validated_tolerance(
        comparator, atol, rtol,
        "" if comparator == "exact" else "runtime scalar comparison",
        observable_name="<scalar>")
    if np.ndim(a) != 0 or np.ndim(b) != 0:
        raise HarnessError("compare_scalar accepts scalar values only")
    if comparator == "exact":
        # Treat paired NaNs as equal for numerical invariance purposes.
        try:
            if bool(np.isnan(a)) and bool(np.isnan(b)):
                ok = True
            else:
                ok = bool(a == b)
        except (TypeError, ValueError):
            ok = bool(a == b)
        return ComparisonResult(
            ok=ok, comparator="exact",
            detail="" if ok else f"exact mismatch: {a!r} != {b!r}")
    if not _real_floating_scalar(a) or not _real_floating_scalar(b):
        raise HarnessError("close comparator requires real floating scalars")
    # Compare the original scalar values.  Do not coerce through Python
    # ``float``: that would narrow np.longdouble/float128 to binary64 and can
    # erase an out-of-tolerance difference before NumPy evaluates it.
    ok = bool(np.isclose(a, b, atol=spec.atol, rtol=spec.rtol,
                         equal_nan=True))
    return ComparisonResult(
        ok=ok, comparator="close", atol=spec.atol, rtol=spec.rtol,
        detail="" if ok else
        f"not close (atol={spec.atol}, rtol={spec.rtol}): {a!r} vs {b!r}")


def compare_array(a: Any, b: Any, *, comparator: str,
                  atol: float = 0.0, rtol: float = 0.0) -> ComparisonResult:
    """Compare arrays element-wise and report mismatch coordinates."""
    spec = _validated_tolerance(
        comparator, atol, rtol,
        "" if comparator == "exact" else "runtime array comparison",
        observable_name="<array>")
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        return ComparisonResult(
            ok=False, comparator=comparator, atol=float(atol), rtol=float(rtol),
            detail=f"shape mismatch: {aa.shape!r} != {bb.shape!r}")
    if comparator == "exact":
        try:
            equal = np.equal(aa, bb) | (np.isnan(aa) & np.isnan(bb))
        except (TypeError, ValueError):
            equal = np.equal(aa, bb)
    elif comparator == "close":
        if not (np.issubdtype(aa.dtype, np.floating) and
                np.issubdtype(bb.dtype, np.floating)):
            raise HarnessError("close comparator requires real floating arrays")
        # Preserve the operands' NumPy floating precision.  Casting to
        # ``float`` here narrows wider dtypes and can create a false-positive
        # comparison pass.  NumPy chooses the common floating dtype directly.
        equal = np.isclose(aa, bb, atol=spec.atol, rtol=spec.rtol,
                           equal_nan=True)
        atol, rtol = spec.atol, spec.rtol
    else:
        raise HarnessError(f"unknown comparator {comparator!r}; known: {COMPARATORS}")
    bad = np.argwhere(~np.asarray(equal, dtype=bool))
    coords = tuple(tuple(int(i) for i in row) for row in bad[:8])
    nbad = int(len(bad))
    return ComparisonResult(
        ok=(nbad == 0), comparator=comparator, atol=float(atol), rtol=float(rtol),
        mismatch_count=nbad, mismatch_indices=coords,
        detail="" if nbad == 0 else f"{nbad} element mismatch(es); first={coords}")


def compare_observable(o: "Observable", reference: Any, candidate: Any) -> ComparisonResult:
    spec = tolerance_for(o)
    if np.ndim(reference) == 0 and np.ndim(candidate) == 0:
        return compare_scalar(reference, candidate, comparator=spec.comparator,
                              atol=spec.atol, rtol=spec.rtol)
    return compare_array(reference, candidate, comparator=spec.comparator,
                         atol=spec.atol, rtol=spec.rtol)


def comparison_evidence(o: "Observable", result: ComparisonResult, *,
                        reference_label: str, candidate_label: str) -> dict:
    """JSON-ready record of the numerical comparison that actually ran."""
    spec = tolerance_for(o)
    return {
        "observable": o.name,
        "reference": reference_label,
        "candidate": candidate_label,
        "comparator": spec.comparator,
        "atol": spec.atol,
        "rtol": spec.rtol,
        "rationale": spec.rationale,
        "ok": bool(result.ok),
        "detail": result.detail,
        "mismatch_count": int(result.mismatch_count),
        "mismatch_indices": [list(x) for x in result.mismatch_indices],
    }


def cmp_exact(a: Any, b: Any) -> tuple[bool, str]:
    result = (compare_scalar(a, b, comparator="exact")
              if np.ndim(a) == 0 and np.ndim(b) == 0
              else compare_array(a, b, comparator="exact"))
    return result.ok, result.detail


def comparator_for(o: "Observable") -> Callable[[Any, Any], tuple[bool, str]]:
    """Resolve the validated comparator contract for one observable."""
    tolerance_for(o)
    def _c(a, b):
        result = compare_observable(o, a, b)
        return result.ok, result.detail
    return _c


def cmp_close(atol: float, rtol: float) -> Callable[[Any, Any], tuple[bool, str]]:
    def _c(a, b):
        result = (compare_scalar(a, b, comparator="close", atol=atol, rtol=rtol)
                  if np.ndim(a) == 0 and np.ndim(b) == 0
                  else compare_array(a, b, comparator="close", atol=atol, rtol=rtol))
        return result.ok, result.detail
    return _c


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CaseSpec
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Observable:
    name: str
    source: str                 # ORACLE_SOURCE
    access: str                 # ACCESS
    path: str
    status: str = "EXECUTED"    # OBS_STATUS or DEFERRED:<stage>
    comparator: str = "exact"   # "exact" | "close"
    atol: float = 0.0
    rtol: float = 0.0
    rationale: str = ""         # REQUIRED when comparator == "close"


@dataclass
class FigureContract:
    """What a machine-gated invariance PAGE must show — v1.2 §6.

    Nine required fields.  This is a STRUCTURED record, not prose: it is what a
    reviewer (human or machine) checks the rendered page against.  It is NOT
    interchangeable with ``expected_visual``, which is the sentence a human
    reads.  A1 v02 omitted it entirely, so a case could be declared complete
    while saying nothing checkable about its figure.
    """
    expected_panels: str = ""
    panel_roles: str = ""
    expected_traces: str = ""
    expected_group_count: str = ""
    primary_comparison: str = ""
    residual_definition: str = ""
    accepted_envelope: str = ""
    case_ids: Sequence[str] = ()
    proof_kind: str = ""            # mirrors oracle_kind, stated on the page

    REQUIRED = ("expected_panels", "panel_roles", "expected_traces",
                "expected_group_count", "primary_comparison",
                "residual_definition", "accepted_envelope", "proof_kind")

    def missing(self) -> list[str]:
        return [f for f in self.REQUIRED if not getattr(self, f)]

    def contradicts(self, case: "CaseSpec") -> list[str]:
        """Where this contract disagrees with the case it claims to describe.

        A1 v04.  NOTE — this rule is an IMPLEMENTATION RULING, not ratified
        text: v1.2 §6 requires the fields `case IDs` and `proof kind` to EXIST
        and nowhere requires them to AGREE with `case_id` / `oracle_kind`.
        v03 therefore accepted a contract naming a different case and a
        different proof kind, which is worse than no contract at all — a page
        that confidently describes something else.  If the architect wants this
        ratified, v1.2 §6 needs one sentence.
        """
        bad: list[str] = []
        if not self.case_ids:
            bad.append("case_ids is empty")
        elif tuple(self.case_ids) != (case.case_id,):
            bad.append(f"case_ids {tuple(self.case_ids)!r} != "
                       f"({case.case_id!r},)")
        if self.proof_kind != case.oracle_kind:
            bad.append(f"proof_kind {self.proof_kind!r} != "
                       f"oracle_kind {case.oracle_kind!r}")
        return bad


@dataclass
class CaseSpec:
    case_id: str
    claim_id: str
    title: str
    claim: str
    failure_means: str
    expected_visual: str
    owner_on_failure: str                       # ADF | dfdraw | GB | harness | environment
    purpose: str                                # PURPOSE
    gate: str                                   # GATE
    oracle_kind: str                            # ORACLE_KIND
    loading_mode: str                           # LOADING_MODE
    sample_mode: str                            # SAMPLE_MODE
    canonical_spec: dict                        # ONE spec, shared by every surface
    # ── the ratified minimum contract (v1.2 §3.1) that A1 v02 omitted ──
    # applicability is grouped with purpose/gate in v1.2 and is ORTHOGONAL to
    # both (§3.2).  A CORE_MANDATORY case is always applicable; an
    # ENVIRONMENT_GATED one may not be, and §3.2 requires "SKIP + explicit
    # applicability reason" — never a bare skip.
    applicable: bool = True
    applicability_reason: str = ""
    # setup_contract  — what the case ESTABLISHES before running: the frame,
    #                   the aliases, the loading mode.
    # preconditions   — what must HOLD before it may run.
    # anti_contamination_preconditions — the narrower slot-exclusivity subset,
    #                   already present since v01.
    # v1.2 lists all three separately; this reading is stated here so a
    # reviewer who disagrees can see the boundary rather than infer it.
    setup_contract: str = ""
    preconditions: Sequence[str] = ()
    figure_contract: "FigureContract | None" = None
    surfaces_under_test: Sequence[str] = ()
    slots_under_test: Sequence[str] = ()
    observables: Sequence[Observable] = ()
    non_claims: Sequence[str] = ()
    anti_contamination_preconditions: Sequence[str] = ()
    not_applicable: dict = field(default_factory=dict)   # surface -> reason
    known_bug_status: str = "SUPPORTED"
    known_bug_id: str = ""
    negative_control: str = ""
    reference_policy: str = "named-immutable"
    # A1-v06-P1-2: there is no per-case schema_version.  v06 defaulted it to
    # the module constant and accepted any override, so a case could declare
    # "1.0.0-ANCIENT" while being validated by 13.77.A1.7 — two schema
    # authorities, one of them wrong.  A field whose only legal value is a
    # module constant carries no information; provenance() records the module
    # version once, for the run.


# ─────────────────────────────────────────────────────────────────────────────
# A3 canonical same-spec cross-surface cases
# ─────────────────────────────────────────────────────────────────────────────

def a3_cases() -> tuple[CaseSpec, ...]:
    """Canonical A3 same-spec cases.

    Every case uses ONE ``canonical_spec`` object for ``draw``, ``draw_batch``
    and ``draw_figures``.  A3.1/A3.2 establish the simplest histogram shape;
    A3.3 adds a two-variable profile; A3.6 adds one production-shaped
    ``group_by`` profile and compares its measured per-group/per-bin data.
    A3.7 adds one canonical ``facet_by`` profile on the two surfaces that
    support faceting plus a separate refusal contract for ``draw_figures``.
    A3.8 adds one production-shaped subframe-qualified profile across all
    three public draw surfaces.  A3.9 adds one production-shaped
    ``selection`` + ``selection_vector`` differential profile across all three
    surfaces.  A3.10 adds explicit profile-bin geometry/statistics coverage and
    records histogram bin edges/counts as ARTIST_FALLBACK/NOT_EXTRACTABLE rather
    than silently upgrading the earlier histogram summary-statistics proof.
    ``weights_vector`` and lazy/eager symmetry remain out of scope.
    """
    hist_id = "I2-HIST-01"
    hist = CaseSpec(
        case_id=hist_id,
        claim_id="I2",
        title="same ncl histogram through all public draw surfaces",
        claim=("draw(), draw_batch() and draw_figures() produce the same "
               "numerical histogram statistics for one canonical request"),
        failure_means=("a public draw surface applies different preparation, "
                       "selection or evaluation semantics to the same request"),
        expected_visual="one ncl histogram with the same selected population",
        owner_on_failure="ADF",
        purpose="INVARIANCE",
        gate="CORE_MANDATORY",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FULL",
        canonical_spec={
            "expr": "ncl",
            "type": "hist",
            "bins": 50,
            "selection": "ncl>30",
            "auto_title": True,
        },
        applicable=True,
        setup_contract=("frame contains numeric ncl; EAGER/FULL; no alias is "
                        "required for this first same-spec case"),
        preconditions=("ncl is present and numeric",),
        figure_contract=FigureContract(
            expected_panels="one panel",
            panel_roles="main: ncl histogram",
            expected_traces="one histogram",
            expected_group_count="1",
            primary_comparison=("n, n_input, n_filtered, mean and std across "
                                "draw/draw_batch/draw_figures"),
            residual_definition="candidate surface minus draw reference",
            accepted_envelope=("counts exact; mean/std within declared "
                               "floating tolerance"),
            case_ids=(hist_id,),
            proof_kind="CONSISTENCY",
        ),
        surfaces_under_test=SURFACES,
        observables=(
            Observable("n", "STATS", "FLAT", "n"),
            Observable("n_input", "STATS", "FLAT", "n_input"),
            Observable("n_filtered", "STATS", "FLAT", "n_filtered"),
            Observable("mean", "STATS", "FLAT", "mean", comparator="close",
                       atol=1e-14, rtol=1e-12,
                       rationale="same selected sample; floating reduction"),
            Observable("std", "STATS", "FLAT", "std", comparator="close",
                       atol=1e-14, rtol=1e-12,
                       rationale="same selected sample; floating reduction"),
            Observable(
                "bin_edges", "ARTIST_FALLBACK", "ARRAY", "artist.bin_edges",
                status="NOT_EXTRACTABLE",
                rationale=("histogram bin edges are absent from the returned public "
                           "stats payload; the current A3 runner has no executable "
                           "ARTIST_FALLBACK extractor"),
            ),
            Observable(
                "bin_counts", "ARTIST_FALLBACK", "ARRAY", "artist.bin_counts",
                status="NOT_EXTRACTABLE",
                rationale=("histogram bin counts are absent from the returned public "
                           "stats payload; the current A3 runner has no executable "
                           "ARTIST_FALLBACK extractor"),
            ),
        ),
        non_claims=("cross-surface agreement is not an independent correctness proof",),
        negative_control="FAMILY_MUTATION:A3-SURFACE-STATS-CORRUPTION",
        reference_policy="named-immutable",
    )

    profile_id = "I2-PROFILE-01"
    profile = CaseSpec(
        case_id=profile_id,
        claim_id="I2",
        title="same y:x profile through all public draw surfaces",
        claim=("draw(), draw_batch() and draw_figures() produce the same "
               "two-variable profile statistics for one canonical request"),
        failure_means=("a public draw surface evaluates, filters or prepares "
                       "the x/y profile population differently"),
        expected_visual="one y:x profile with the same selected population",
        owner_on_failure="ADF",
        purpose="INVARIANCE",
        gate="CORE_MANDATORY",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FULL",
        canonical_spec={
            "expr": "y:x",
            "type": "profile",
            "bins": 25,
            "selection": "(x>-2.0)&(x<2.0)",
            "auto_title": True,
        },
        applicable=True,
        setup_contract=("frame contains finite numeric x and y; EAGER/FULL; "
                        "no grouping/faceting/subframe dependency"),
        preconditions=("x is present and numeric", "y is present and numeric"),
        figure_contract=FigureContract(
            expected_panels="one panel",
            panel_roles="main: y versus x profile",
            expected_traces="one profile",
            expected_group_count="1",
            primary_comparison=("n, n_input, n_filtered, mean_x, mean_y, "
                                "std_x and std_y across all three surfaces"),
            residual_definition="candidate surface minus draw reference",
            accepted_envelope=("counts exact; floating summary statistics "
                               "within declared tolerance"),
            case_ids=(profile_id,),
            proof_kind="CONSISTENCY",
        ),
        surfaces_under_test=SURFACES,
        observables=(
            Observable("n", "STATS", "FLAT", "n"),
            Observable("n_input", "STATS", "FLAT", "n_input"),
            Observable("n_filtered", "STATS", "FLAT", "n_filtered"),
            Observable("mean_x", "STATS", "FLAT", "mean_x", comparator="close",
                       atol=1e-14, rtol=1e-12,
                       rationale="same selected x sample; floating reduction"),
            Observable("mean_y", "STATS", "FLAT", "mean_y", comparator="close",
                       atol=1e-14, rtol=1e-12,
                       rationale="same selected y sample; floating reduction"),
            Observable("std_x", "STATS", "FLAT", "std_x", comparator="close",
                       atol=1e-14, rtol=1e-12,
                       rationale="same selected x sample; floating reduction"),
            Observable("std_y", "STATS", "FLAT", "std_y", comparator="close",
                       atol=1e-14, rtol=1e-12,
                       rationale="same selected y sample; floating reduction"),
        ),
        non_claims=("profile bin contents are not yet independently correctness-anchored",),
        negative_control="FAMILY_MUTATION:A3-PROFILE-SURFACE-STATS-CORRUPTION",
        reference_policy="named-immutable",
    )

    group_id = "I2-GROUPBY-01"
    group = CaseSpec(
        case_id=group_id,
        claim_id="I2",
        title="same side_type grouped profile through all public draw surfaces",
        claim=("draw(), draw_batch() and draw_figures() produce the same "
               "per-group/per-bin profile result for one canonical group_by request"),
        failure_means=("a public draw surface changes group membership, binning, "
                       "selection or grouped numerical reduction semantics"),
        expected_visual="one sector profile with one trace per side_type group",
        owner_on_failure="ADF",
        purpose="INVARIANCE",
        gate="CORE_MANDATORY",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FULL",
        canonical_spec={
            "expr": "dcar_tpc_vertex:sector",
            "type": "profile",
            "bins": 36,
            "selection": "(ncl>60)&(abs(dcar_tpc_vertex)<10)&(side_type<2)",
            "group_by": "side_type",
            "return_data": True,
            "auto_title": True,
        },
        applicable=True,
        setup_contract=("frame contains numeric sector, dcar_tpc_vertex, ncl and "
                        "side_type; EAGER/FULL; grouped profile data requested "
                        "explicitly for numerical observability"),
        preconditions=(
            "sector is present and numeric",
            "dcar_tpc_vertex is present and numeric",
            "ncl is present and numeric",
            "side_type is present, numeric and contains at least two selected groups",
        ),
        figure_contract=FigureContract(
            expected_panels="one panel",
            panel_roles="main: dcar_tpc_vertex versus sector profile",
            expected_traces="one profile trace per side_type group",
            expected_group_count="2",
            primary_comparison=("profile_data group labels, counts, x centers and "
                                "y means across draw/draw_batch/draw_figures"),
            residual_definition=("candidate per-group/per-bin observable minus "
                                 "draw reference at the same profile_data row"),
            accepted_envelope=("group labels and counts exact; floating profile "
                               "coordinates/means within declared tolerance"),
            case_ids=(group_id,),
            proof_kind="CONSISTENCY",
        ),
        surfaces_under_test=SURFACES,
        observables=(
            Observable("group", "STATS", "ARRAY", "profile_data.group"),
            Observable("count", "STATS", "ARRAY", "profile_data.count"),
            Observable("x_center", "STATS", "ARRAY", "profile_data.x_center",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same global profile bins; floating bin centers"),
            Observable("y_mean", "STATS", "ARRAY", "profile_data.y_mean",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same grouped rows; floating per-bin reduction"),
        ),
        non_claims=(
            "cross-surface group_by agreement is not an independent correctness proof",
            "return_data=True is used only to expose grouped numerical observables",
        ),
        negative_control="FAMILY_MUTATION:A3-GROUP-SPECIFIC-PROFILE-CORRUPTION",
        reference_policy="named-immutable",
    )

    facet_id = "I2-FACET-01"
    facet_refusal_id = "I2-FACET-DRAW-FIGURES-REFUSAL-01"
    # One canonical request object is shared by the supported-surface numerical
    # case and the draw_figures refusal case.  Keeping the error contract as a
    # separate CaseSpec is deliberate: each CaseSpec must produce exactly one
    # reconcilable result, and an ERROR_CONTRACT PASS has no numerical
    # comparisons by construction.
    facet_spec = {
        "expr": "dcar_tpc_vertex:sector",
        "type": "profile",
        "bins": 36,
        "selection": "(ncl>60)&(abs(dcar_tpc_vertex)<10)&(side_type<2)",
        "facet_by": "side_type",
        "return_data": True,
        "auto_title": True,
    }
    facet = CaseSpec(
        case_id=facet_id,
        claim_id="I2",
        title="same side_type faceted profile on supported public draw surfaces",
        claim=("draw() and draw_batch() produce the same facet-resolved profile "
               "result for one canonical facet_by request"),
        failure_means=("a supported public draw surface changes facet membership, "
                       "binning, selection or facet-resolved numerical reduction"),
        expected_visual="two side_type facet panels with the same profile definition",
        owner_on_failure="ADF",
        purpose="INVARIANCE",
        gate="CORE_MANDATORY",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FULL",
        canonical_spec=facet_spec,
        applicable=True,
        setup_contract=("frame contains numeric sector, dcar_tpc_vertex, ncl and "
                        "side_type; EAGER/FULL; facet-resolved profile data requested "
                        "explicitly for numerical observability"),
        preconditions=(
            "sector is present and numeric",
            "dcar_tpc_vertex is present and numeric",
            "ncl is present and numeric",
            "side_type is present, numeric and contains selected values 0 and 1",
        ),
        figure_contract=FigureContract(
            expected_panels="two facet panels on supported surfaces",
            panel_roles="side_type=0 facet; side_type=1 facet",
            expected_traces="one profile trace in each facet panel",
            expected_group_count="2 facet populations",
            primary_comparison=("facet keys plus per-facet profile_data counts, x "
                                "centers and y means across draw/draw_batch"),
            residual_definition=("candidate per-facet/per-bin observable minus "
                                 "draw reference at the same facet and profile_data row"),
            accepted_envelope=("facet labels/counts exact; floating profile "
                               "coordinates/means within declared tolerance"),
            case_ids=(facet_id,),
            proof_kind="CONSISTENCY",
        ),
        surfaces_under_test=SURFACES,
        not_applicable={
            "draw_figures": ("facet_by is intentionally refused by the current "
                             "draw_figures panel contract; the separate "
                             f"{facet_refusal_id} error-contract case proves the guard")
        },
        observables=(
            Observable("facet_groups", "STATS", "ARRAY", "groups"),
            Observable("facet0_count", "STATS", "ARRAY",
                       "per_group.0.profile_data.count"),
            Observable("facet0_x_center", "STATS", "ARRAY",
                       "per_group.0.profile_data.x_center",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same global facet profile bins; floating bin centers"),
            Observable("facet0_y_mean", "STATS", "ARRAY",
                       "per_group.0.profile_data.y_mean",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same facet-0 rows; floating per-bin reduction"),
            Observable("facet1_count", "STATS", "ARRAY",
                       "per_group.1.profile_data.count"),
            Observable("facet1_x_center", "STATS", "ARRAY",
                       "per_group.1.profile_data.x_center",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same global facet profile bins; floating bin centers"),
            Observable("facet1_y_mean", "STATS", "ARRAY",
                       "per_group.1.profile_data.y_mean",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same facet-1 rows; floating per-bin reduction"),
        ),
        non_claims=(
            "facet_by agreement is not an independent mathematical correctness proof",
            "draw_figures facet rendering remains unsupported and is not emulated",
            "return_data=True is used only to expose facet-resolved numerical observables",
        ),
        negative_control="FAMILY_MUTATION:A3-FACET-SPECIFIC-PROFILE-CORRUPTION",
        reference_policy="named-immutable",
    )
    facet_refusal = CaseSpec(
        case_id=facet_refusal_id,
        claim_id="I2",
        title="draw_figures explicitly refuses the canonical facet_by request",
        claim=("draw_figures() refuses the exact same canonical facet_by request "
               "until the known faceted-axes limitation is removed deliberately"),
        failure_means=("draw_figures no longer refuses the unsupported facet_by "
                       "request, or refuses without naming facet_by"),
        expected_visual="no figure: the public surface must refuse before rendering",
        owner_on_failure="ADF",
        purpose="ERROR_CONTRACT",
        gate="CORE_MANDATORY",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FULL",
        canonical_spec=facet_spec,
        applicable=True,
        surfaces_under_test=("draw_figures",),
        known_bug_status="KNOWN_BUG",
        known_bug_id="BUG_dfdraw_20260611_facet_by_ax_ignored",
        non_claims=(
            "this case does not claim facet rendering support in draw_figures",
            "when dfdraw gains safe faceted axes ownership this refusal must be revised explicitly",
        ),
        reference_policy="named-immutable",
    )

    subframe_id = "I2-SUBFRAME-01"
    subframe = CaseSpec(
        case_id=subframe_id,
        claim_id="I2",
        title="same CalibVertex subframe profile through all public draw surfaces",
        claim=("draw(), draw_batch() and draw_figures() produce the same "
               "per-bin profile result for one canonical subframe-qualified request"),
        failure_means=("a public draw surface resolves, joins, broadcasts, bins or "
                       "reduces the same subframe-qualified expression differently"),
        expected_visual=("one CalibVertex.vertex_x_intercept versus time_s profile "
                         "with the same keyed subframe values"),
        owner_on_failure="ADF",
        purpose="INVARIANCE",
        gate="CORE_MANDATORY",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FULL",
        canonical_spec={
            "expr": "CalibVertex.vertex_x_intercept:time_s",
            "type": "profile",
            "bins": 12,
            "return_data": True,
            "auto_title": True,
        },
        applicable=True,
        setup_contract=("main frame contains repeated quantile_bin keys and numeric "
                        "time_s; registered CalibVertex subframe contains one row per "
                        "quantile_bin and numeric vertex_x_intercept; EAGER/FULL"),
        preconditions=(
            "quantile_bin is present in the main frame and repeats across rows",
            "CalibVertex is registered on quantile_bin",
            "CalibVertex.vertex_x_intercept exists only on the subframe",
            "time_s is present and numeric",
        ),
        figure_contract=FigureContract(
            expected_panels="one panel",
            panel_roles="main: CalibVertex.vertex_x_intercept versus time_s profile",
            expected_traces="one profile",
            expected_group_count="1",
            primary_comparison=("n plus profile_data counts, x centers and y means "
                                "across draw/draw_batch/draw_figures"),
            residual_definition=("candidate per-bin observable minus draw reference "
                                 "at the same profile_data row"),
            accepted_envelope=("n/count exact; floating profile coordinates/means "
                               "within declared tolerance"),
            case_ids=(subframe_id,),
            proof_kind="CONSISTENCY",
        ),
        surfaces_under_test=SURFACES,
        observables=(
            Observable("n", "STATS", "FLAT", "n"),
            Observable("count", "STATS", "ARRAY", "profile_data.count"),
            Observable("x_center", "STATS", "ARRAY", "profile_data.x_center",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same profile bins; floating bin centers"),
            Observable("y_mean", "STATS", "ARRAY", "profile_data.y_mean",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale=("same keyed subframe values broadcast to the same "
                                  "main-frame rows; floating per-bin reduction")),
        ),
        non_claims=(
            "cross-surface subframe agreement is not an independent correctness proof",
            "this checkpoint does not cover nested subframes, subframe aliases or missing keys",
            "return_data=True is used only to expose per-bin numerical observables",
        ),
        negative_control="FAMILY_MUTATION:A3-SUBFRAME-DERIVED-PROFILE-CORRUPTION",
        reference_policy="named-immutable",
    )

    selection_vector_id = "I2-SELECTION-VECTOR-01"
    selection_vector = CaseSpec(
        case_id=selection_vector_id,
        claim_id="I2",
        title="same selected two-branch differential profile through all public draw surfaces",
        claim=("draw(), draw_batch() and draw_figures() produce the same "
               "branch-resolved and derived differential profile for one canonical "
               "selection + selection_vector request"),
        failure_means=("a public draw surface applies different base-selection, "
                       "vector-branch, binning or normalization semantics to the same request"),
        expected_visual=("one nClITS versus time_s differential profile built from "
                         "the same signal/reference selection-vector branches"),
        owner_on_failure="ADF",
        purpose="INVARIANCE",
        gate="CORE_MANDATORY",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FULL",
        canonical_spec={
            "expr": "nClITS:time_s",
            "type": "profile",
            "bins": 8,
            "selection": "(ncl>60)&(abs(dcar_tpc_vertex)<10)&(hasITSTPC)",
            "selection_vector": [
                "(abs(sector-13)<2)",
                "(abs(sector-13)>=2)&(sector<36)",
            ],
            "normalize": "delta",
            "return_data": True,
            "auto_title": True,
        },
        applicable=True,
        setup_contract=("frame contains numeric nClITS, time_s, ncl, "
                        "dcar_tpc_vertex, hasITSTPC and sector; the base selection "
                        "rejects some rows and the two selection_vector branches "
                        "both contain selected rows; EAGER/FULL"),
        preconditions=(
            "base selection keeps a strict subset of input rows",
            "both selection_vector branches contain selected rows",
            "the two vector branches partition the selected synthetic fixture",
            "nClITS and time_s are present and numeric",
        ),
        figure_contract=FigureContract(
            expected_panels="one main profile plus the differential-normalization view",
            panel_roles=("signal/reference profile comparison with delta normalization"),
            expected_traces="two vector-source profiles plus derived differential",
            expected_group_count="2 vector branches",
            primary_comparison=("normalize_data x_center, signal/reference central values "
                                "and counts, plus the derived delta value across "
                                "draw/draw_batch/draw_figures"),
            residual_definition=("candidate branch-resolved/derived observable minus "
                                 "draw reference at the same normalize_data row"),
            accepted_envelope=("branch counts exact; floating branch central values, "
                               "bin centers and derived delta within declared tolerance"),
            case_ids=(selection_vector_id,),
            proof_kind="CONSISTENCY",
        ),
        surfaces_under_test=SURFACES,
        observables=(
            Observable("x_center", "STATS", "ARRAY", "normalize_data.x_center",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same profile bins; floating bin centers"),
            Observable("signal_central", "STATS", "ARRAY",
                       "normalize_data.signal_central",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same selected signal-vector rows; floating reduction"),
            Observable("signal_count", "STATS", "ARRAY",
                       "normalize_data.signal_count"),
            Observable("reference_central", "STATS", "ARRAY",
                       "normalize_data.reference_central",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same selected reference-vector rows; floating reduction"),
            Observable("reference_count", "STATS", "ARRAY",
                       "normalize_data.reference_count"),
            Observable("value", "STATS", "ARRAY", "normalize_data.value",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same delta transform of the two vector-source profiles"),
        ),
        non_claims=(
            "cross-surface agreement is not an independent mathematical correctness proof",
            "this checkpoint does not cover weights_vector",
            "return_data=True is used only to expose branch-resolved numerical observables",
        ),
        negative_control=("FAMILY_MUTATION:A3-SELECTION-VECTOR-BRANCH-"
                          "PROFILE-CORRUPTION"),
        reference_policy="named-immutable",
    )

    profile_bins_id = "I2-PROFILE-BINS-01"
    profile_bins = CaseSpec(
        case_id=profile_bins_id,
        claim_id="I2",
        title="same sparse explicit-bin profile through all public draw surfaces",
        claim=("draw(), draw_batch() and draw_figures() expose the same explicit "
               "profile-bin geometry, counts, central values, errors and empty-bin "
               "mask for one canonical sparse request"),
        failure_means=("a public draw surface changes explicit profile binning, "
                       "per-bin population, reduction/error semantics or the missing-bin mask"),
        expected_visual=("one sparse y:x profile with populated first/last bins and "
                         "three intentionally empty interior bins"),
        owner_on_failure="ADF",
        purpose="INVARIANCE",
        gate="CORE_MANDATORY",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FULL",
        canonical_spec={
            "expr": "y:x",
            "type": "profile",
            "bins": 5,
            "range": (0.0, 5.0),
            "return_data": True,
            "auto_title": True,
        },
        applicable=True,
        setup_contract=("frame contains finite numeric x/y rows only in the first "
                        "and last of five explicit x bins; EAGER/FULL; return_data=True "
                        "exposes the stable profile_data table"),
        preconditions=(
            "x and y are present and numeric",
            "explicit range is [0, 5] with exactly five bins",
            "first and last bins are populated and three interior bins are empty",
        ),
        figure_contract=FigureContract(
            expected_panels="one panel",
            panel_roles="main: sparse y versus x profile",
            expected_traces="one profile",
            expected_group_count="1",
            primary_comparison=("profile_data x_low/x_high/x_center, count, y_mean, "
                                "y_std, y_sem and y_central across all three surfaces"),
            residual_definition=("candidate per-bin observable minus draw reference "
                                 "at the same explicit bin row"),
            accepted_envelope=("count exact; floating geometry/statistics within "
                               "declared tolerance; NaN empty-bin mask must agree"),
            case_ids=(profile_bins_id,),
            proof_kind="CONSISTENCY",
        ),
        surfaces_under_test=SURFACES,
        observables=(
            Observable("x_low", "STATS", "ARRAY", "profile_data.x_low",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same explicit profile-bin lower edges"),
            Observable("x_high", "STATS", "ARRAY", "profile_data.x_high",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same explicit profile-bin upper edges"),
            Observable("x_center", "STATS", "ARRAY", "profile_data.x_center",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same explicit profile-bin centers"),
            Observable("count", "STATS", "ARRAY", "profile_data.count"),
            Observable("y_mean", "STATS", "ARRAY", "profile_data.y_mean",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same per-bin profile mean; empty bins remain NaN"),
            Observable("y_std", "STATS", "ARRAY", "profile_data.y_std",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same per-bin profile standard deviation; empty bins remain NaN"),
            Observable("y_sem", "STATS", "ARRAY", "profile_data.y_sem",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same per-bin profile standard error; empty bins remain NaN"),
            Observable("y_central", "STATS", "ARRAY", "profile_data.y_central",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same rendered central statistic; empty bins remain NaN"),
        ),
        non_claims=(
            "cross-surface profile-bin agreement is not a full independent raw-row correctness proof",
            "histogram bin edges/counts remain explicitly NOT_EXTRACTABLE from public stats",
            "this checkpoint does not implement ARTIST_FALLBACK extraction",
        ),
        negative_control="FAMILY_MUTATION:A3-PROFILE-BIN-ERROR-CORRUPTION",
        reference_policy="named-immutable",
    )

    return (hist, profile, group, facet, facet_refusal, subframe, selection_vector,
            profile_bins)


# ─────────────────────────────────────────────────────────────────────────────
# A3.11 closure reconciliation
# ─────────────────────────────────────────────────────────────────────────────

A3_REQUIRED_FAMILIES = {
    "histogram": ("I2-HIST-01",),
    "profile": ("I2-PROFILE-01",),
    "group_by": ("I2-GROUPBY-01",),
    "facet_by": ("I2-FACET-01", "I2-FACET-DRAW-FIGURES-REFUSAL-01"),
    "subframe_qualified": ("I2-SUBFRAME-01",),
    "selection_vector": ("I2-SELECTION-VECTOR-01",),
}
A3_CLOSURE_HARDENING = ("I2-PROFILE-BINS-01",)
A3_HISTOGRAM_BIN_NONCLAIMS = ("bin_edges", "bin_counts")

# A3.11-v02: closure is allowed to trust banked proof evidence only if the
# declaration being closed is still the same kind and public-surface scope that
# was reviewed.  IDs alone are not proof identity.  Keep this map deliberately
# narrow: it locks proof semantics, not prose, tolerances, or fixture values.
A3_CLOSURE_CONTRACTS = {
    "I2-HIST-01": {
        "purpose": "INVARIANCE", "oracle_kind": "CONSISTENCY",
        "surfaces_under_test": SURFACES, "not_applicable_surfaces": (),
    },
    "I2-PROFILE-01": {
        "purpose": "INVARIANCE", "oracle_kind": "CONSISTENCY",
        "surfaces_under_test": SURFACES, "not_applicable_surfaces": (),
    },
    "I2-GROUPBY-01": {
        "purpose": "INVARIANCE", "oracle_kind": "CONSISTENCY",
        "surfaces_under_test": SURFACES, "not_applicable_surfaces": (),
    },
    "I2-FACET-01": {
        "purpose": "INVARIANCE", "oracle_kind": "CONSISTENCY",
        "surfaces_under_test": SURFACES,
        "not_applicable_surfaces": ("draw_figures",),
    },
    "I2-FACET-DRAW-FIGURES-REFUSAL-01": {
        "purpose": "ERROR_CONTRACT", "oracle_kind": "CONSISTENCY",
        "surfaces_under_test": ("draw_figures",),
        "not_applicable_surfaces": (),
    },
    "I2-SUBFRAME-01": {
        "purpose": "INVARIANCE", "oracle_kind": "CONSISTENCY",
        "surfaces_under_test": SURFACES, "not_applicable_surfaces": (),
    },
    "I2-SELECTION-VECTOR-01": {
        "purpose": "INVARIANCE", "oracle_kind": "CONSISTENCY",
        "surfaces_under_test": SURFACES, "not_applicable_surfaces": (),
    },
    "I2-PROFILE-BINS-01": {
        "purpose": "INVARIANCE", "oracle_kind": "CONSISTENCY",
        "surfaces_under_test": SURFACES, "not_applicable_surfaces": (),
    },
}


def a3_closure_reconciliation(cases: Sequence[CaseSpec] | None = None) -> dict:
    """Return the machine-readable A3 closure/disposition record.

    A3 is the same-spec cross-surface CONSISTENCY substage.  This record does
    not upgrade a consistency case into an independent correctness proof and it
    does not turn a NOT_EXTRACTABLE observable into VERIFIED evidence.  It
    answers the narrower closure question: is every required A3 family backed
    by an explicit CaseSpec, and is every non-executed A3 observable either
    explicitly dispositioned or an orphan that must block closure?
    """
    cases = tuple(a3_cases() if cases is None else cases)
    by_id = {c.case_id: c for c in cases}
    duplicate_ids = sorted({c.case_id for c in cases
                            if sum(x.case_id == c.case_id for x in cases) > 1})
    required_ids = [cid for ids in A3_REQUIRED_FAMILIES.values() for cid in ids]
    required_ids += list(A3_CLOSURE_HARDENING)
    missing_cases = [cid for cid in required_ids if cid not in by_id]
    orphan_obligations: list[str] = []
    contract_drift: list[dict] = []
    family_records: list[dict] = []

    # Lock the reviewed proof kind and public-surface scope for every required
    # or hardening case.  Generic registry validation only checks that a field
    # is legal; this closure-specific audit checks that it still means what the
    # banked A3 review proved.
    for cid, expected in A3_CLOSURE_CONTRACTS.items():
        c = by_id.get(cid)
        if c is None:
            continue
        observed = {
            "purpose": c.purpose,
            "oracle_kind": c.oracle_kind,
            "surfaces_under_test": tuple(c.surfaces_under_test),
            "not_applicable_surfaces": tuple(sorted(c.not_applicable)),
        }
        for field_name, expected_value in expected.items():
            actual_value = observed[field_name]
            # Preserve strings as strings and surface collections as tuples for
            # comparison; emit JSON-friendly lists below.
            norm_expected = (tuple(expected_value)
                             if field_name in ("surfaces_under_test", "not_applicable_surfaces")
                             else expected_value)
            if actual_value != norm_expected:
                contract_drift.append({
                    "case_id": cid,
                    "field": field_name,
                    "expected": (list(norm_expected)
                                 if isinstance(norm_expected, tuple) else norm_expected),
                    "observed": (list(actual_value)
                                 if isinstance(actual_value, tuple) else actual_value),
                })

    drift_ids = {row["case_id"] for row in contract_drift}
    for family, ids in A3_REQUIRED_FAMILIES.items():
        missing = [cid for cid in ids if cid not in by_id]
        drifted = [cid for cid in ids if cid in drift_ids]
        status = "MISSING" if missing else ("CONTRACT_DRIFT" if drifted else "PROVED")
        family_records.append({
            "family": family,
            "case_ids": list(ids),
            "status": status,
            "evidence_scope": ("banked per-case execution/review evidence; "
                               "this closure record checks declaration/disposition coverage"),
            "missing_case_ids": missing,
            "contract_drift_case_ids": drifted,
        })

    # The facet capability boundary has two proof obligations on one canonical
    # request: numerical invariance where supported and mandatory refusal on
    # draw_figures.  Losing either one blocks an honest A3 closure.
    facet = by_id.get("I2-FACET-01")
    refusal = by_id.get("I2-FACET-DRAW-FIGURES-REFUSAL-01")
    if facet is not None and refusal is not None:
        if facet.canonical_spec is not refusal.canonical_spec:
            orphan_obligations.append("facet_by numerical/refusal CaseSpecs no longer share one canonical request")

    histogram_disposition: list[dict] = []
    hist = by_id.get("I2-HIST-01")
    if hist is not None:
        obs = {o.name: o for o in hist.observables}
        for name in A3_HISTOGRAM_BIN_NONCLAIMS:
            o = obs.get(name)
            if o is None:
                orphan_obligations.append(f"I2-HIST-01/{name}: required explicit non-claim is missing")
                histogram_disposition.append({"observable": name, "status": "MISSING"})
                continue
            record = {
                "observable": name,
                "source": o.source,
                "status": o.status,
                "rationale": o.rationale,
                "a3_disposition": "EXPLICIT_NON_CLAIM",
                "next_owner": "Stage A",
                "future_resolution": ("retain the explicit non-claim unless later "
                                      "PDF/artist review or a future ARTIST_FALLBACK "
                                      "extractor adds evidence"),
            }
            histogram_disposition.append(record)
            if not (o.source == "ARTIST_FALLBACK"
                    and o.status == "NOT_EXTRACTABLE"
                    and bool(o.rationale)):
                orphan_obligations.append(
                    f"I2-HIST-01/{name}: must remain ARTIST_FALLBACK / "
                    "NOT_EXTRACTABLE with rationale for A3 closure")

    # Any other non-executed A3 observable must also carry an explicit
    # disposition.  Silence is an orphan and therefore a closure blocker.
    for c in cases:
        if c.case_id not in required_ids:
            continue
        for o in c.observables:
            if o.status == "EXECUTED":
                continue
            if o.status == "NOT_EXTRACTABLE" and o.rationale:
                continue
            if o.status.startswith("DEFERRED:") and o.status.split(":", 1)[1].strip():
                continue
            orphan_obligations.append(
                f"{c.case_id}/{o.name}: non-executed observable has no complete disposition")

    registry_errors = validate_registry(cases)
    closure_ready = not (duplicate_ids or missing_cases or contract_drift
                         or orphan_obligations or registry_errors)
    return {
        "substage": "A3",
        "record_kind": "closure declaration/disposition reconciliation",
        "scope": "same-spec cross-surface consistency",
        "status": "READY_FOR_CLOSURE" if closure_ready else "BLOCKED",
        "closure_ready": closure_ready,
        "required_families": family_records,
        "closure_hardening_case_ids": list(A3_CLOSURE_HARDENING),
        "histogram_bin_observables": histogram_disposition,
        "missing_case_ids": missing_cases,
        "duplicate_case_ids": duplicate_ids,
        "contract_drift": contract_drift,
        "orphan_obligations": orphan_obligations,
        "registry_errors": registry_errors,
        "execution_context": {
            "fresh_execution_verdict": False,
            "meaning": ("declaration/disposition reconciliation over banked A3 proof "
                        "contracts; ordinary manifest reconciliation remains the "
                        "execution/gate authority"),
        },
        "explicit_non_claims": [
            "histogram bin_edges/bin_counts are not verified by the public stats channel",
            "A3 consistency does not by itself prove independent raw-row mathematical correctness",
            "facet_by_bins/facet_by_quantiles, weights_vector, and lazy/eager composition are not claimed by A3",
        ],
        "later_stage_a_obligations": [
            "independent correctness/reference reconciliation where required",
            "histogram bin-level evidence may be assigned to later Stage-A PDF/artist review or a future extractor",
            "lazy/eager full-stack symmetry belongs to later Stage-A work",
        ],
    }



# ─────────────────────────────────────────────────────────────────────────────
# A4.1 expression-slot symmetry — first bounded slot case
# ─────────────────────────────────────────────────────────────────────────────

EXPRESSION_SLOTS = (
    "expr", "selection", "weights", "group_by", "facet_by",
    "selection_vector", "weights_vector", "subframe_qualified_expression",
    "compound_expression",
)

# A4 does not infer slot ownership from variable names.  This compact execution
# contract is the machine authority for each slot-exclusive fixture.  The
# ``runner`` field is deliberately explicit: the one-surface CORE_MANDATORY
# exception is justified only for cases actually bound to run_slot_symmetry().
A4_SLOT_CONTRACTS = {
    "I3-SELECTION-01": {
        "runner": "run_slot_symmetry",
        "slots_under_test": ("selection",),
        "slot_alias": "slot_keep",
        "required_physical_dependencies": ("dep_selection",),
        "expected_lazy_loaded_after": ("dep_selection", "x", "y"),
        "unrelated_physical_branches": ("decoy",),
        "anti_contamination_preconditions": (
            "slot alias slot_keep is absent from frame columns before each arm",
            "lazy reader begins with no loaded physical branches",
            "selection-only dependency dep_selection is unloaded before the lazy arm",
            "unrelated branch decoy is unloaded before and after the lazy arm",
        ),
    },
    "I3-EXPR-01": {
        "runner": "run_slot_symmetry",
        "slots_under_test": ("expr",),
        "slot_alias": "slot_expr",
        "required_physical_dependencies": ("dep_expr",),
        "expected_lazy_loaded_after": ("dep_expr", "x"),
        "unrelated_physical_branches": ("decoy",),
        "anti_contamination_preconditions": (
            "slot alias slot_expr is absent from frame columns before each arm",
            "lazy reader begins with no loaded physical branches",
            "expr-only dependency dep_expr is unloaded before the lazy arm",
            "unrelated branch decoy is unloaded before and after the lazy arm",
        ),
    },
    "I3-WEIGHTS-01": {
        "runner": "run_slot_symmetry",
        "slots_under_test": ("weights",),
        "slot_alias": "slot_weight",
        "required_physical_dependencies": ("dep_weights",),
        "expected_lazy_loaded_after": ("dep_weights", "x", "y"),
        "unrelated_physical_branches": ("decoy",),
        "anti_contamination_preconditions": (
            "slot alias slot_weight is absent from frame columns before each arm",
            "lazy reader begins with no loaded physical branches",
            "weights-only dependency dep_weights is unloaded before the lazy arm",
            "unrelated branch decoy is unloaded before and after the lazy arm",
        ),
    },
    "I3-GROUP-BY-01": {
        "runner": "run_slot_symmetry",
        "slots_under_test": ("group_by",),
        "slot_alias": "slot_group",
        "required_physical_dependencies": ("dep_group",),
        "expected_lazy_loaded_after": ("dep_group", "x", "y"),
        "unrelated_physical_branches": ("decoy",),
        "anti_contamination_preconditions": (
            "slot alias slot_group is absent from frame columns before each arm",
            "lazy reader begins with no loaded physical branches",
            "group_by-only dependency dep_group is unloaded before the lazy arm",
            "unrelated branch decoy is unloaded before and after the lazy arm",
        ),
    },
    "I3-FACET-BY-01": {
        "runner": "run_slot_symmetry",
        "slots_under_test": ("facet_by",),
        "slot_alias": "slot_facet",
        "required_physical_dependencies": ("dep_facet",),
        "expected_lazy_loaded_after": ("dep_facet", "x", "y"),
        "unrelated_physical_branches": ("decoy",),
        "anti_contamination_preconditions": (
            "slot alias slot_facet is absent from frame columns before each arm",
            "lazy reader begins with no loaded physical branches",
            "facet_by-only dependency dep_facet is unloaded before the lazy arm",
            "unrelated branch decoy is unloaded before and after the lazy arm",
        ),
    },
    "I3-COMPOUND-EXPR-01": {
        "runner": "run_slot_symmetry",
        "slots_under_test": ("compound_expression",),
        "slot_alias": "slot_compound",
        "required_physical_dependencies": ("dep_compound",),
        "expected_lazy_loaded_after": ("dep_compound", "x", "y"),
        "unrelated_physical_branches": ("decoy",),
        "anti_contamination_preconditions": (
            "slot alias slot_compound is absent from frame columns before each arm",
            "lazy reader begins with no loaded physical branches",
            "compound-expression-only dependency dep_compound is unloaded before the lazy arm",
            "unrelated branch decoy is unloaded before and after the lazy arm",
        ),
    },
    "I3-SELECTION-VECTOR-01": {
        "runner": "run_slot_symmetry",
        "slots_under_test": ("selection_vector",),
        "slot_alias": "slot_selection_vector",
        "required_physical_dependencies": ("dep_selection_vector",),
        "expected_lazy_loaded_after": ("dep_selection_vector", "x", "y"),
        "unrelated_physical_branches": ("decoy",),
        "anti_contamination_preconditions": (
            "slot alias slot_selection_vector is absent from frame columns before each arm",
            "lazy reader begins with no loaded physical branches",
            "selection_vector-only dependency dep_selection_vector is unloaded before the lazy arm",
            "unrelated branch decoy is unloaded before and after the lazy arm",
        ),
    },
    "I3-WEIGHTS-VECTOR-01": {
        "runner": "run_slot_symmetry",
        "slots_under_test": ("weights_vector",),
        "slot_alias": "slot_weights_vector",
        "required_physical_dependencies": ("dep_weights_vector",),
        "expected_lazy_loaded_after": ("dep_weights_vector", "x", "y"),
        "unrelated_physical_branches": ("decoy",),
        "anti_contamination_preconditions": (
            "slot alias slot_weights_vector is absent from frame columns before each arm",
            "lazy reader begins with no loaded physical branches",
            "weights_vector-only dependency dep_weights_vector is unloaded before the lazy arm",
            "unrelated branch decoy is unloaded before and after the lazy arm",
        ),
    },
    "I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01": {
        "runner": "run_error_contract",
        "slots_under_test": ("selection_vector",),
        "anti_contamination_preconditions": (
            "registered subframe S is present before the public call",
            "S.count appears only inside selection_vector",
            "the request must refuse before dfdraw evaluates an unresolved subframe reference",
        ),
    },
    "I3-SUBFRAME-WEIGHTS-VECTOR-REFUSAL-01": {
        "runner": "run_error_contract",
        "slots_under_test": ("weights_vector",),
        "anti_contamination_preconditions": (
            "registered subframe S is present before the public call",
            "S.count appears only inside weights_vector",
            "the request must refuse before dfdraw evaluates an unresolved subframe reference",
        ),
    },

}


def _a4_slot_case(case_id: str, *, claim_id: str, title: str, claim: str,
                  failure_means: str, expected_visual: str, canonical_spec: dict,
                  setup_contract: str, preconditions: Sequence[str],
                  observables: Sequence[Observable], primary_comparison: str,
                  accepted_envelope: str, negative_control: str) -> CaseSpec:
    contract = A4_SLOT_CONTRACTS[case_id]
    slot = contract["slots_under_test"][0]
    return CaseSpec(
        case_id=case_id,
        claim_id=claim_id,
        title=title,
        claim=claim,
        failure_means=failure_means,
        expected_visual=expected_visual,
        owner_on_failure="ADF",
        purpose="INVARIANCE",
        gate="CORE_MANDATORY",
        oracle_kind="CONSISTENCY",
        loading_mode="BOTH",
        sample_mode="FULL",
        canonical_spec=canonical_spec,
        applicable=True,
        setup_contract=setup_contract,
        preconditions=preconditions,
        figure_contract=FigureContract(
            expected_panels="one panel",
            panel_roles=f"main: {slot}-exclusive profile",
            expected_traces="one profile or grouped/faceted profile envelope",
            expected_group_count="1 or the declared group/facet count",
            primary_comparison=primary_comparison,
            residual_definition="lazy numerical observable minus eager observable",
            accepted_envelope=accepted_envelope,
            case_ids=(case_id,),
            proof_kind="CONSISTENCY",
        ),
        surfaces_under_test=("draw",),
        slots_under_test=contract["slots_under_test"],
        observables=observables,
        non_claims=(
            "A4.2 covers scalar expression-bearing slots only; vector slots remain A4.3",
            "this synthetic case does not replace the later real-data lazy acceptance run",
            "A4 does not reopen A3 public-surface symmetry",
        ),
        anti_contamination_preconditions=contract["anti_contamination_preconditions"],
        negative_control=negative_control,
        reference_policy="named-immutable",
    )


def _a4_vector_refusal_case(case_id: str, *, slot: str, canonical_spec: dict) -> CaseSpec:
    contract = A4_SLOT_CONTRACTS[case_id]
    return CaseSpec(
        case_id=case_id,
        claim_id=f"I3.{slot}.subframe_refusal.A4.3",
        title=f"subframe-qualified reference in {slot} refuses loudly",
        claim=(f"the currently unsupported subframe-qualified {slot} request refuses "
               "with the registered BUG_20260701 contract instead of falling through "
               "to an opaque dfdraw evaluation error"),
        failure_means=(f"{slot} subframe handling changed without updating the declared "
                       "capability boundary, or the refusal stopped naming its owner"),
        expected_visual="no figure: the public request must refuse before rendering",
        owner_on_failure="ADF",
        purpose="ERROR_CONTRACT",
        gate="CORE_MANDATORY",
        oracle_kind="CONSISTENCY",
        loading_mode="BOTH",
        sample_mode="FULL",
        canonical_spec=canonical_spec,
        applicable=True,
        surfaces_under_test=("draw",),
        slots_under_test=contract["slots_under_test"],
        anti_contamination_preconditions=contract["anti_contamination_preconditions"],
        known_bug_status="KNOWN_BUG",
        known_bug_id="BUG_20260701_ADF_subframe_ref_slot_symmetry",
        negative_control=("FAMILY_MUTATION:A4.3 same subframe-qualified vector request "
                          "must not silently succeed or lose the bug-labelled refusal"),
        reference_policy="named-immutable",
    )


def a4_cases() -> tuple[CaseSpec, ...]:
    """A4 slot-exclusive cases implemented through A4.2.

    A4.1 banks the selection-only case.  A4.2 extends the same BOTH/FULL
    execution contract to the scalar ``expr``, ``weights``, ``group_by``,
    ``facet_by`` and compound-expression slots.  A4.3 adds ordinary alias
    coverage for ``selection_vector`` and ``weights_vector`` and records the
    existing subframe-qualified vector limitation as mandatory ERROR_CONTRACT
    cases carrying BUG_20260701_ADF_subframe_ref_slot_symmetry.  The public draw surface is
    intentionally held fixed: A3 already proved surface symmetry; A4's second
    consistency arm is loading mode, and the lazy arm additionally proves the
    exact physical dependency set attributable to that one slot.
    """
    cases = []
    cases.append(_a4_slot_case(
        "I3-SELECTION-01",
        claim_id="I3.selection.A4.1",
        title="selection-slot alias symmetry in eager and lazy loading modes",
        claim=("a dependency that appears only through selection= is discovered and "
               "materialized in EAGER mode and loads exactly its required physical "
               "branch in LAZY mode, with no unrelated preload"),
        failure_means=("selection-slot dependency discovery is asymmetric, the lazy arm "
                       "loads the wrong physical branch set, or a contaminated fixture "
                       "is allowed to pass"),
        expected_visual="one y:x profile after the slot-only selection removes half the rows",
        canonical_spec={"expr": "y:x", "type": "profile", "bins": 8,
                        "selection": "slot_keep>0", "return_data": True,
                        "auto_title": True},
        setup_contract=("EAGER and tracking-LAZY fixtures contain the same physical "
                        "x/y/dep_selection/decoy data; slot_keep is an alias of "
                        "dep_selection>0 and appears only in selection="),
        preconditions=("slot_keep is registered as an alias and not pre-materialized",
                       "dep_selection is the only selection-only physical dependency",
                       "the lazy reader starts with zero loaded physical branches"),
        observables=(Observable("n", "STATS", "FLAT", "n"),
                     Observable("y_mean", "STATS", "ARRAY", "profile_data.y_mean",
                                comparator="close", atol=1e-14, rtol=1e-12,
                                rationale="same selected rows and profile reduction in EAGER/LAZY")),
        primary_comparison=("EAGER versus LAZY selected-row count and profile values; "
                            "LAZY exact physical branch-load evidence"),
        accepted_envelope=("selected-row count exact; floating profile values within "
                           "declared tolerance; exact lazy loaded branch set"),
        negative_control="GLOBAL_MUTATION:M2 selection-slot preload contamination -> INVALID_FIXTURE",
    ))
    cases.append(_a4_slot_case(
        "I3-EXPR-01", claim_id="I3.expr.A4.2",
        title="expr-slot alias symmetry in eager and lazy loading modes",
        claim="an alias used only as the profile value expression is discovered symmetrically",
        failure_means="expr-slot alias discovery or lazy physical dependency loading is asymmetric",
        expected_visual="one slot_expr:x profile",
        canonical_spec={"expr": "slot_expr:x", "type": "profile", "bins": 8,
                        "return_data": True, "auto_title": True},
        setup_contract="slot_expr aliases dep_expr and appears only in expr",
        preconditions=("slot_expr is registered and not pre-materialized",
                       "dep_expr is unloaded before the lazy arm"),
        observables=(Observable("n", "STATS", "FLAT", "n"),
                     Observable("y_mean", "STATS", "ARRAY", "profile_data.y_mean",
                                comparator="close", atol=1e-14, rtol=1e-12,
                                rationale="same expr-only alias values in EAGER/LAZY")),
        primary_comparison="EAGER versus LAZY profile values plus exact lazy expr dependency load",
        accepted_envelope="n exact; y_mean within tolerance; exact lazy loaded branch set",
        negative_control="GLOBAL_MUTATION:M2 expr-slot preload contamination -> INVALID_FIXTURE",
    ))
    cases.append(_a4_slot_case(
        "I3-WEIGHTS-01", claim_id="I3.weights.A4.2",
        title="weights-slot alias symmetry in eager and lazy loading modes",
        claim="an alias used only through weights= is discovered symmetrically",
        failure_means="weights-slot alias discovery or weighted profile reduction differs by loading mode",
        expected_visual="one weighted y:x profile",
        canonical_spec={"expr": "y:x", "type": "profile", "bins": 8,
                        "weights": "slot_weight", "return_data": True,
                        "auto_title": True},
        setup_contract="slot_weight aliases 1+dep_weights and appears only in weights",
        preconditions=("slot_weight is registered and not pre-materialized",
                       "dep_weights is unloaded before the lazy arm"),
        observables=(Observable("n", "STATS", "FLAT", "n"),
                     Observable("y_mean", "STATS", "ARRAY", "profile_data.y_mean",
                                comparator="close", atol=1e-14, rtol=1e-12,
                                rationale="same weighted per-bin profile reduction in EAGER/LAZY")),
        primary_comparison="EAGER versus LAZY weighted profile plus exact lazy weights dependency load",
        accepted_envelope="n exact; weighted y_mean within tolerance; exact lazy loaded branch set",
        negative_control="GLOBAL_MUTATION:M2 weights-slot preload contamination -> INVALID_FIXTURE",
    ))
    cases.append(_a4_slot_case(
        "I3-GROUP-BY-01", claim_id="I3.group_by.A4.2",
        title="group_by-slot alias symmetry in eager and lazy loading modes",
        claim="an alias used only through group_by= is discovered symmetrically",
        failure_means="group_by-slot dependency discovery or grouped profile reduction differs by loading mode",
        expected_visual="one grouped y:x profile",
        canonical_spec={"expr": "y:x", "type": "profile", "bins": 8,
                        "group_by": "slot_group", "return_data": True,
                        "auto_title": True},
        setup_contract="slot_group aliases dep_group and appears only in group_by",
        preconditions=("slot_group is registered and not pre-materialized",
                       "dep_group is unloaded before the lazy arm"),
        observables=(Observable("n", "STATS", "FLAT", "n"),
                     Observable("y_mean", "STATS", "ARRAY", "profile_data.y_mean",
                                comparator="close", atol=1e-14, rtol=1e-12,
                                rationale="same group-resolved profile data in EAGER/LAZY")),
        primary_comparison="EAGER versus LAZY grouped profile plus exact lazy group_by dependency load",
        accepted_envelope="n exact; grouped y_mean within tolerance; exact lazy loaded branch set",
        negative_control="GLOBAL_MUTATION:M2 group_by-slot preload contamination -> INVALID_FIXTURE",
    ))
    cases.append(_a4_slot_case(
        "I3-FACET-BY-01", claim_id="I3.facet_by.A4.2",
        title="facet_by-slot alias symmetry in eager and lazy loading modes",
        claim="an alias used only through facet_by= is discovered symmetrically",
        failure_means="facet_by-slot dependency discovery or faceted population differs by loading mode",
        expected_visual="one two-facet y:x profile",
        canonical_spec={"expr": "y:x", "type": "profile", "bins": 8,
                        "facet_by": "slot_facet", "return_data": True,
                        "auto_title": True},
        setup_contract="slot_facet aliases dep_facet and appears only in facet_by",
        preconditions=("slot_facet is registered and not pre-materialized",
                       "dep_facet is unloaded before the lazy arm"),
        observables=(Observable("n_total", "STATS", "FLAT", "n_total"),),
        primary_comparison="EAGER versus LAZY faceted population plus exact lazy facet_by dependency load",
        accepted_envelope="n_total exact; exact lazy loaded branch set",
        negative_control="GLOBAL_MUTATION:M2 facet_by-slot preload contamination -> INVALID_FIXTURE",
    ))
    cases.append(_a4_slot_case(
        "I3-COMPOUND-EXPR-01", claim_id="I3.compound_expression.A4.2",
        title="compound-expression slot alias symmetry in eager and lazy loading modes",
        claim="an alias embedded only inside a compound plotted expression is discovered symmetrically",
        failure_means="compound-expression alias discovery or profile reduction differs by loading mode",
        expected_visual="one (y+slot_compound):x profile",
        canonical_spec={"expr": "y + slot_compound:x", "type": "profile", "bins": 8,
                        "return_data": True, "auto_title": True},
        setup_contract="slot_compound aliases dep_compound and appears only inside the compound expr",
        preconditions=("slot_compound is registered and not pre-materialized",
                       "dep_compound is unloaded before the lazy arm"),
        observables=(Observable("n", "STATS", "FLAT", "n"),
                     Observable("y_mean", "STATS", "ARRAY", "profile_data.y_mean",
                                comparator="close", atol=1e-14, rtol=1e-12,
                                rationale="same compound-expression profile reduction in EAGER/LAZY")),
        primary_comparison="EAGER versus LAZY compound profile plus exact lazy compound dependency load",
        accepted_envelope="n exact; y_mean within tolerance; exact lazy loaded branch set",
        negative_control="GLOBAL_MUTATION:M2 compound-expression preload contamination -> INVALID_FIXTURE",
    ))
    cases.append(_a4_slot_case(
        "I3-SELECTION-VECTOR-01", claim_id="I3.selection_vector.A4.3",
        title="selection_vector alias symmetry in eager and lazy loading modes",
        claim="an alias used only inside selection_vector is discovered symmetrically",
        failure_means="selection_vector alias discovery or branch-resolved profile reduction differs by loading mode",
        expected_visual="one normalized signal/reference profile difference",
        canonical_spec={"expr": "y:x", "type": "profile", "bins": 8,
                        "selection_vector": ["slot_selection_vector>0",
                                             "slot_selection_vector<=0"],
                        "normalize": "delta", "return_data": True,
                        "auto_title": True},
        setup_contract="slot_selection_vector aliases dep_selection_vector and appears only in selection_vector",
        preconditions=("slot_selection_vector is registered and not pre-materialized",
                       "dep_selection_vector is unloaded before the lazy arm",
                       "both vector branches select populated rows"),
        observables=(
            Observable("signal_count", "STATS", "ARRAY", "normalize_data.signal_count",
                       comparator="exact", rationale="same signal-branch population in EAGER/LAZY"),
            Observable("reference_count", "STATS", "ARRAY", "normalize_data.reference_count",
                       comparator="exact", rationale="same reference-branch population in EAGER/LAZY"),
            Observable("signal_central", "STATS", "ARRAY", "normalize_data.signal_central",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same signal profile reduction in EAGER/LAZY"),
            Observable("reference_central", "STATS", "ARRAY", "normalize_data.reference_central",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same reference profile reduction in EAGER/LAZY"),
            Observable("value", "STATS", "ARRAY", "normalize_data.value",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same derived delta after identical vector branches"),
        ),
        primary_comparison="EAGER versus LAZY branch-resolved vector profile plus exact lazy dependency load",
        accepted_envelope="branch counts exact; floating branch/delta values within tolerance; exact lazy loaded set",
        negative_control="GLOBAL_MUTATION:M2 selection_vector preload contamination -> INVALID_FIXTURE",
    ))
    cases.append(_a4_slot_case(
        "I3-WEIGHTS-VECTOR-01", claim_id="I3.weights_vector.A4.3",
        title="weights_vector alias symmetry in eager and lazy loading modes",
        claim="an alias used only inside weights_vector is discovered symmetrically",
        failure_means="weights_vector alias discovery or branch-weighted profile reduction differs by loading mode",
        expected_visual="one normalized weighted signal/reference profile difference",
        canonical_spec={"expr": "y:x", "type": "profile", "bins": 8,
                        "weights_vector": ["slot_weights_vector",
                                           "2.5-slot_weights_vector"],
                        "normalize": "delta", "return_data": True,
                        "auto_title": True},
        setup_contract="slot_weights_vector aliases 1+dep_weights_vector and appears only in weights_vector",
        preconditions=("slot_weights_vector is registered and not pre-materialized",
                       "dep_weights_vector is unloaded before the lazy arm",
                       "both vector weight expressions remain positive"),
        observables=(
            Observable("signal_count", "STATS", "ARRAY", "normalize_data.signal_count",
                       comparator="exact", rationale="same signal weighted-profile population in EAGER/LAZY"),
            Observable("reference_count", "STATS", "ARRAY", "normalize_data.reference_count",
                       comparator="exact", rationale="same reference weighted-profile population in EAGER/LAZY"),
            Observable("signal_central", "STATS", "ARRAY", "normalize_data.signal_central",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same signal weighted profile in EAGER/LAZY"),
            Observable("reference_central", "STATS", "ARRAY", "normalize_data.reference_central",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same reference weighted profile in EAGER/LAZY"),
            Observable("value", "STATS", "ARRAY", "normalize_data.value",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same derived weighted delta in EAGER/LAZY"),
        ),
        primary_comparison="EAGER versus LAZY weighted vector branches plus exact lazy dependency load",
        accepted_envelope="branch counts exact; floating weighted branch/delta values within tolerance; exact lazy loaded set",
        negative_control="GLOBAL_MUTATION:M2 weights_vector preload contamination -> INVALID_FIXTURE",
    ))
    cases.append(_a4_vector_refusal_case(
        "I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01",
        slot="selection_vector",
        canonical_spec={"expr": "y:x", "type": "profile", "bins": 8,
                        "selection_vector": ["S.count>0", "S.count<=0"],
                        "normalize": "delta", "return_data": True,
                        "auto_title": True},
    ))
    cases.append(_a4_vector_refusal_case(
        "I3-SUBFRAME-WEIGHTS-VECTOR-REFUSAL-01",
        slot="weights_vector",
        canonical_spec={"expr": "y:x", "type": "profile", "bins": 8,
                        "weights_vector": ["S.count", "2.5-S.count"],
                        "normalize": "delta", "return_data": True,
                        "auto_title": True},
    ))
    return tuple(cases)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Registry validation — §14's self-checks, run BEFORE any case executes.
#
# "Adding a case breaks the build until somebody says what it is."
# ─────────────────────────────────────────────────────────────────────────────

# ── the declared-state audit ────────────────────────────────────────────────
#
# A1 v01 P1-2  oracle_source declared, execution ignored it
# A1 v03 P0-1  applicable    declared, execution ignored it
# A1 v04 P0-2  source        validated as implementable, never checked against
#                            the path the runner takes
# A1 v04 P1-1  slots_under_test / reference_policy declared, nothing reads them
#
# Four instances of ONE class in four increments, each fixed individually.  A
# field that describes state and has no reader is documentation wearing a
# schema's clothes.  This registry makes the class self-detecting: every
# CaseSpec field must be READ, VALIDATED, SERIALISED — or declared future work
# against a named stage.  A new field with no reader fails the suite the day it
# is added, rather than surviving to the next review round.
FUTURE_STAGE_FIELDS = {
    "reference_policy": "A6",                    # named-reference acceptance
}


def _is_casespec_annotation(ann) -> bool:
    """Exact match on the CaseSpec type, never a substring.

    A1-v07-P1-3.  v07 tested `"CaseSpec" in <string annotation>`, which admits
    CaseSpecView, NotACaseSpec, FakeCaseSpec — any name containing the token.
    A quoted forward reference is parsed and matched by identity.
    """
    import ast
    if ann is None:
        return False
    if isinstance(ann, ast.Name):
        return ann.id == "CaseSpec"
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        try:
            inner = ast.parse(ann.value, mode="eval").body
        except SyntaxError:
            return False
        return _is_casespec_annotation(inner)
    if isinstance(ann, ast.Subscript):                # Sequence[CaseSpec]
        return _is_casespec_annotation(ann.slice)
    if isinstance(ann, ast.Tuple):
        return any(_is_casespec_annotation(e) for e in ann.elts)
    if isinstance(ann, ast.BinOp):                    # CaseSpec | None
        return (_is_casespec_annotation(ann.left)
                or _is_casespec_annotation(ann.right))
    return False


def _scope_receivers(scope, spec_body_lines: set) -> set:
    """CaseSpec receiver names proven WITHIN one function scope.

    A1-v07-P1-1.  v07 built a MODULE-GLOBAL set of names, so proving `case` is
    a CaseSpec in one function certified `case.<anything>` in every other
    function.  Executed falsifier: an unannotated `def unrelated(case): return
    case.status` put `status` in the read-set and an unread `CaseSpec.status`
    was reported clean.  A variable name is not a type identity, and it is not
    one across lexical scopes either.

    Evidence recognised, all within this scope only:
        annotated parameter          def f(case: CaseSpec)
        annotated local              case: CaseSpec = ...
        constructor result           c = CaseSpec(...)
        isinstance-narrowed name     isinstance(x, CaseSpec)
        loop over a receiver         for c in cases          (bare Name only)
    """
    import ast
    names = set()

    a = scope.args
    for arg in list(a.args) + list(a.posonlyargs) + list(a.kwonlyargs):
        if _is_casespec_annotation(arg.annotation):
            names.add(arg.arg)

    inner = {n for n in ast.walk(scope)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n is not scope}
    inner_lines = {ln for f in inner for ln in
                   range(f.lineno, getattr(f, "end_lineno", f.lineno) + 1)}

    def _own(node):
        return getattr(node, "lineno", None) not in inner_lines

    for node in ast.walk(scope):
        if not _own(node):
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and _is_casespec_annotation(node.annotation):
            names.add(node.target.id)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) \
                and node.value.func.id == "CaseSpec":
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "isinstance" and len(node.args) == 2
              and isinstance(node.args[0], ast.Name)
              and isinstance(node.args[1], ast.Name)
              and node.args[1].id == "CaseSpec"):
            names.add(node.args[0].id)

    # loop / comprehension over a receiver, bare Name only.  Iterating an
    # ATTRIBUTE of a receiver yields its elements, not CaseSpecs — an earlier
    # v07 draft walked through ast.Attribute and put Observable fields back in.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(scope):
            if not _own(node):
                continue
            if isinstance(node, (ast.For, ast.comprehension)):
                it, tgt = node.iter, node.target
                if isinstance(it, ast.Name) and it.id in names \
                        and isinstance(tgt, ast.Name) and tgt.id not in names:
                    names.add(tgt.id)
                    changed = True
    return names


def _fields_read_in_this_module() -> set:
    """CaseSpec fields genuinely read, resolved PER SCOPE.

    v05 hand-maintained allow-list  -> already false-certified three fields
    v06 derived, receiver-blind     -> any object's same-named attribute
    v07 receiver-narrowed by NAME   -> any same-named receiver, ANY scope
    v08 per-scope symbol table      -> a name proven in one function proves
                                       nothing in another
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(sys.modules[__name__]))

    spec_body_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CaseSpec":
            for sub in ast.walk(node):
                if hasattr(sub, "lineno"):
                    spec_body_lines.add(sub.lineno)

    read = set()
    for scope in ast.walk(tree):
        if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        receivers = _scope_receivers(scope, spec_body_lines)
        if not receivers:
            continue
        for node in ast.walk(scope):
            if getattr(node, "lineno", None) in spec_body_lines:
                continue
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id in receivers:
                read.add(node.attr)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id == "getattr" and len(node.args) >= 2
                  and isinstance(node.args[0], ast.Name)
                  and node.args[0].id in receivers
                  and isinstance(node.args[1], ast.Constant)
                  and isinstance(node.args[1].value, str)):
                read.add(node.args[1].value)
    return read


def audit_declared_state() -> list:
    """Every CaseSpec field is genuinely read, or future-staged with its stage.

    Returns a list of orphans.  Empty list == no declared-but-unread state.

    This is the fourth attempt at closing one class:
        v01 P1-2  oracle_source declared, execution ignored it
        v03 P0-1  applicable    declared, execution ignored it
        v04 P0-2  source        validated, never checked against the runner
        v04 P1-1  slots_under_test / reference_policy, no reader
        v05 P1-1  the audit ITSELF false-certified three fields
    Derivation replaces declaration so the audit cannot be satisfied by editing
    a list.
    """
    from dataclasses import fields as _fields
    read = _fields_read_in_this_module()
    orphans = []
    for f in _fields(CaseSpec):
        if f.name in read or f.name in FUTURE_STAGE_FIELDS:
            continue
        orphans.append(
            f"CaseSpec.{f.name}: declared but never read in this module; add a "
            f"reader, or register it in FUTURE_STAGE_FIELDS against the stage "
            f"that will consume it")
    return orphans


def validate_registry(cases: Sequence[CaseSpec]) -> list[str]:
    """Return a list of violations.  Empty list == registry is admissible."""
    bad: list[str] = []
    # A1-v07-P0-1: an empty registry is a failure at BOTH doors.  The gate
    # refuses it via reconcile(); validation must refuse it too, so a
    # registry-loading failure is caught before anything is executed.
    if not cases:
        bad.append("the declared case set is EMPTY; a harness that declares "
                   "no case cannot report success")
    seen: set[str] = set()
    for c in cases:
        cid = c.case_id
        if cid in seen:
            bad.append(f"{cid}: duplicate case_id")
        seen.add(cid)
        if not c.claim:
            bad.append(f"{cid}: no claim")
        if not c.claim_id:
            bad.append(f"{cid}: no claim_id")
        if not c.owner_on_failure:
            bad.append(f"{cid}: no owner_on_failure")
        if c.purpose not in PURPOSE:
            bad.append(f"{cid}: purpose {c.purpose!r} not in {PURPOSE}")
        if c.gate not in GATE:
            bad.append(f"{cid}: gate {c.gate!r} not in {GATE}")
        if c.oracle_kind not in ORACLE_KIND:
            bad.append(f"{cid}: oracle_kind {c.oracle_kind!r} not in {ORACLE_KIND}")
        if c.loading_mode not in LOADING_MODE:
            bad.append(f"{cid}: loading_mode {c.loading_mode!r} not in {LOADING_MODE}")
        if c.sample_mode not in SAMPLE_MODE:
            bad.append(f"{cid}: sample_mode {c.sample_mode!r} not in {SAMPLE_MODE}")
        if c.known_bug_status not in KNOWN_BUG_STATUS:
            bad.append(f"{cid}: known_bug_status {c.known_bug_status!r} invalid")
        if c.known_bug_status in ("KNOWN_BUG", "EXPECTED_FAIL") and not c.known_bug_id:
            bad.append(f"{cid}: {c.known_bug_status} requires known_bug_id")
        # v1.2 §7.3: sampled-lazy is unsupported and must not be introduced here.
        if c.loading_mode in ("LAZY", "BOTH") and c.sample_mode == "FRACTION":
            bad.append(f"{cid}: LAZY/BOTH with FRACTION — sampled-lazy is not "
                       f"supported by build_adf (v1.2 §7.3)")
        # A4.1: slots_under_test and anti-contamination are no longer
        # future-declared state.  A slot case must bind to an explicit execution
        # contract; the declaration itself is checked against that authority.
        if c.slots_under_test:
            unknown_slots = [slot for slot in c.slots_under_test
                             if slot not in EXPRESSION_SLOTS]
            if unknown_slots:
                bad.append(f"{cid}: unknown slots_under_test {unknown_slots}")
            slot_contract = A4_SLOT_CONTRACTS.get(cid)
            if slot_contract is None:
                bad.append(f"{cid}: slot case has no A4_SLOT_CONTRACTS execution contract")
            else:
                runner = slot_contract.get("runner")
                if c.purpose == "INVARIANCE" and runner != "run_slot_symmetry":
                    bad.append(f"{cid}: A4 slot execution contract is not bound to run_slot_symmetry")
                elif c.purpose == "ERROR_CONTRACT" and runner != "run_error_contract":
                    bad.append(f"{cid}: A4 ERROR_CONTRACT slot execution contract is not bound to run_error_contract")
                elif runner not in ("run_slot_symmetry", "run_error_contract"):
                    bad.append(f"{cid}: A4 slot execution contract has unknown runner {runner!r}")
                if tuple(c.slots_under_test) != tuple(slot_contract["slots_under_test"]):
                    bad.append(f"{cid}: slots_under_test drift from A4 execution contract")
                if tuple(c.anti_contamination_preconditions) != tuple(
                        slot_contract["anti_contamination_preconditions"]):
                    bad.append(f"{cid}: anti_contamination_preconditions drift from A4 execution contract")
                if c.loading_mode != "BOTH":
                    bad.append(f"{cid}: A4 slot contracts require loading_mode='BOTH'")
                if c.sample_mode != "FULL":
                    bad.append(f"{cid}: A4 BOTH slot contract requires sample_mode='FULL'")
                if runner == "run_slot_symmetry" and c.purpose != "INVARIANCE":
                    bad.append(f"{cid}: run_slot_symmetry contract requires purpose='INVARIANCE'")
                if runner == "run_error_contract" and c.purpose != "ERROR_CONTRACT":
                    bad.append(f"{cid}: run_error_contract slot contract requires purpose='ERROR_CONTRACT'")
        elif c.anti_contamination_preconditions:
            bad.append(f"{cid}: anti_contamination_preconditions declared without slots_under_test")
        for s in c.surfaces_under_test:
            if s not in SURFACES:
                bad.append(f"{cid}: unknown surface {s!r}")
        for s in c.not_applicable:
            if s not in SURFACES:
                bad.append(f"{cid}: not_applicable names unknown surface {s!r}")
            if not c.not_applicable[s]:
                bad.append(f"{cid}: not_applicable[{s!r}] has no reason")
        if c.purpose in ("INVARIANCE", "CORRECTNESS") and not c.observables:
            bad.append(f"{cid}: {c.purpose} case declares no observable")
        # ── A1 v03: the ratified minimum contract, validated ──────────────
        # v1.2 §3.1 lists these as separate proof obligations.  v02 accepted a
        # case without any of them, and test_a1_registry_accepts_a_clean_case
        # PROVED validation did not require them — a schema declared complete
        # while four ratified fields were absent.
        if not c.applicable and not c.applicability_reason:
            bad.append(f"{cid}: not applicable but no applicability_reason "
                       f"(v1.2 §3.2 requires SKIP to carry its reason)")
        # A1-v05-P1-2: ENVIRONMENT_BLOCKED asserts the environment is blocked.
        # v05 accepted it alongside applicable=True, a contradiction whose
        # strict semantics were undefined.  The two must agree.
        if c.known_bug_status == "ENVIRONMENT_BLOCKED" and c.applicable:
            bad.append(f"{cid}: known_bug_status ENVIRONMENT_BLOCKED with "
                       f"applicable=True is contradictory; a blocked "
                       f"environment is not applicable")
        if c.applicable is False and c.gate == "CORE_MANDATORY":
            bad.append(f"{cid}: CORE_MANDATORY cases are always applicable; "
                       f"declare gate=ENVIRONMENT_GATED to be inapplicable")
        if c.purpose in ("INVARIANCE", "CORRECTNESS"):
            if not c.setup_contract:
                bad.append(f"{cid}: no setup_contract — what the case "
                           f"establishes before running is unstated")
            if not c.preconditions:
                bad.append(f"{cid}: no preconditions — what must hold before "
                           f"the case may run is unstated")
            if c.figure_contract is None:
                bad.append(f"{cid}: no figure_contract (v1.2 §6) — the page "
                           f"says nothing checkable about what should be seen")
            else:
                miss = c.figure_contract.missing()
                if miss:
                    bad.append(f"{cid}: figure_contract missing required "
                               f"field(s): {miss}")
                # A1-v03-P0-2
                for why in c.figure_contract.contradicts(c):
                    bad.append(f"{cid}: figure_contract contradicts the case: "
                               f"{why}")
            # A1-v03-COMPLETE-3: v1.2 §3.1 lists negative_control /
            # required_mutation in the minimum schema.  The objective is not a
            # mutation per case — it is that no machine-gated case silently has
            # NO falsification owner.
            if not c.negative_control:
                bad.append(f"{cid}: no negative_control / required_mutation — "
                           f"this machine-gated case has no falsification "
                           f"owner; name one, or reference a harness-level "
                           f"falsifier as GLOBAL_MUTATION:<id> or "
                           f"FAMILY_MUTATION:<id>")
            elif c.negative_control.startswith(("GLOBAL_MUTATION:",
                                                "FAMILY_MUTATION:")):
                if not c.negative_control.split(":", 1)[1].strip():
                    bad.append(f"{cid}: negative_control names a shared "
                               f"falsifier but gives no id")
        # A1-P0-1: a machine-gated case must be ABLE to compare something.
        # A registry that only checks `observables` is non-empty accepts a case
        # whose every observable is NOT_EXTRACTABLE, which then PASSES having
        # compared nothing.
        if c.purpose in ("INVARIANCE", "CORRECTNESS"):
            executable = [o for o in c.observables
                          if o.status == "EXECUTED"]
            if not executable:
                bad.append(f"{cid}: {c.purpose} case has no EXECUTED observable "
                           f"— it could only PASS by comparing nothing")
        # A1-P0-3: a consistency case needs >=2 applicable surfaces, or it can
        # only SKIP.  A mandatory case that can only SKIP is invisible to the gate.
        if c.oracle_kind == "CONSISTENCY" and c.purpose == "INVARIANCE":
            applicable = [s_ for s_ in c.surfaces_under_test
                          if s_ not in c.not_applicable]
            # A4 slot symmetry compares EAGER vs LAZY while deliberately holding
            # one public surface fixed; its second consistency arm is loading mode,
            # not another draw surface.
            slot_contract = A4_SLOT_CONTRACTS.get(c.case_id) if c.slots_under_test else None
            a4_both = (c.loading_mode == "BOTH" and slot_contract is not None
                       and slot_contract.get("runner") == "run_slot_symmetry")
            if len(applicable) < 2 and c.gate == "CORE_MANDATORY" and not a4_both:
                bad.append(f"{cid}: CORE_MANDATORY consistency case has "
                           f"{len(applicable)} applicable surface(s); it can only "
                           f"SKIP and would never gate")
        for o in c.observables:
            for violation in tolerance_violations(o):
                bad.append(f"{cid}/{o.name}: {violation}")
            if o.status == "EXECUTED" and o.source not in IMPLEMENTED_SOURCES:
                bad.append(f"{cid}/{o.name}: source {o.source!r} is declarable "
                           f"but not executable by this runner; declare "
                           f"DEFERRED:<stage> instead of claiming it is compared")
            if o.source not in ORACLE_SOURCE:
                bad.append(f"{cid}/{o.name}: source {o.source!r} not in {ORACLE_SOURCE}")
            if o.access not in ACCESS:
                bad.append(f"{cid}/{o.name}: access {o.access!r} not in {ACCESS}")
            if not (o.status in OBS_STATUS or o.status.startswith("DEFERRED:")):
                bad.append(f"{cid}/{o.name}: status {o.status!r} invalid")
            if o.source == "ARTIST_FALLBACK" and not o.rationale:
                bad.append(f"{cid}/{o.name}: ARTIST_FALLBACK needs a stated reason")
        if c.purpose == "INVARIANCE" and c.oracle_kind == "CORRECTNESS":
            bad.append(f"{cid}: INVARIANCE purpose with CORRECTNESS oracle_kind — "
                       f"consistency and correctness must not be conflated")
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Result record + manifest
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    status: str
    detail: str = ""
    payload_paths: dict = field(default_factory=dict)   # surface -> path
    observed: dict = field(default_factory=dict)
    observable_contract: list = field(default_factory=list)   # A1-P1-4
    comparisons: list = field(default_factory=list)            # A2 structured evidence
    executed_comparisons: int = 0                             # A1-P0-1
    skipped_surfaces: dict = field(default_factory=dict)
    wall_time_s: float = 0.0
    exception: str = ""


def machine_oracle_text(case: "CaseSpec") -> str:
    """The MACHINE ORACLE block, generated from the observables themselves.

    v1.2 §5: "The human-readable PDF footer and JSON explanation must be
    generated from the same structured source."  This function is that source
    for both — it is called by ``footer_text`` and by ``write_manifest``, so a
    footer cannot describe a comparison the manifest does not record, and
    neither can drift from the observables actually executed.
    """
    if not case.observables:
        return "no observable declared"
    lines = []
    for o in case.observables:
        if o.status != "EXECUTED":
            lines.append(f"{o.name}: {o.status}")
            continue
        if o.comparator == "exact":
            how = "exact"
        else:
            how = f"atol={o.atol}, rtol={o.rtol}"
        lines.append(f"{o.name} ({o.source}, {o.path}): {how}")
    return "; ".join(lines)


def footer_text(case: "CaseSpec") -> str:
    """The three blocks every invariance page must display (v1.2 §6).

    Generated from the CaseSpec.  Nothing here is separately authored: if the
    footer and the manifest ever disagree, one of them stopped reading the
    CaseSpec, and ``test_a1_v03_footer_and_manifest_share_one_source`` fails.
    """
    fc = case.figure_contract
    expected = case.expected_visual
    if fc is not None and fc.primary_comparison:
        expected = (f"{expected}\n  primary comparison: {fc.primary_comparison}"
                    f"\n  accepted envelope:  {fc.accepted_envelope}")
    return (f"CASE: {case.case_id}\n\n"
            f"EXPECTED:\n  {expected}\n\n"
            f"MACHINE ORACLE:\n  {machine_oracle_text(case)}\n\n"
            f"FAILURE MEANS:\n  {case.failure_means}\n")


def _contract(o: "Observable") -> dict:
    """The full comparison contract for one observable, for the manifest.

    A1-P1-4: v01 serialised only observed VALUES, so a reviewer reading the JSON
    could not reconstruct WHICH comparison was executed, with what comparator or
    tolerance.  The declaration and the outcome must both be in the record.
    """
    spec = tolerance_for(o)
    return {"name": o.name, "source": o.source, "access": o.access,
            "path": o.path, "status": o.status, "comparator": spec.comparator,
            "atol": spec.atol, "rtol": spec.rtol, "rationale": spec.rationale}


def provenance() -> dict:
    import pandas as pd
    prov = {
        "schema_version": SCHEMA_VERSION,
        "run_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for mod, key in (("uproot", "uproot"), ("numba", "numba")):
        try:
            prov[key] = __import__(mod).__version__
        except Exception:
            prov[key] = None
    try:
        import ROOT  # noqa: F401
        prov["pyroot"] = True
    except Exception:
        prov["pyroot"] = False
    return prov


def write_manifest(path: str, results: Sequence[CaseResult],
                   cases: Sequence[CaseSpec], extra: dict | None = None) -> dict:
    """Write the run manifest.  Reconciliation facts come from reconcile().

    A1-v07-P1-2: v07 iterated `results`, so a declared case that produced no
    result was absent from the evidence entirely — the gate knew, the manifest
    did not.  Every declared case now appears, and one that produced no result
    appears with status NO_RESULT rather than silently vanishing.
    """
    by_id = {c.case_id: c for c in cases}
    rec = reconcile(results, cases)
    seen: dict = {}
    for r in results:
        seen.setdefault(r.case_id, r)
    doc = {
        "provenance": {**provenance(), **(extra or {})},
        "reconciliation": rec,
        "cases": [],
    }
    # A3.11-v02: once a manifest is in A3 context, persist the closure record
    # even when it is BLOCKED.  Absence of a required family is itself durable
    # evidence and must not make the governance record disappear.
    a3_known = set(A3_CLOSURE_CONTRACTS)
    if a3_known.intersection(by_id):
        doc["a3_closure"] = a3_closure_reconciliation(cases)
    # a declared case with no result is EVIDENCE, not an omission
    results = list(results) + [
        CaseResult(case_id=cid, status="NO_RESULT",
                   detail="declared but produced no result")
        for cid in rec["missing"]]
    for r in results:
        c = by_id.get(r.case_id)
        rec = asdict(r)
        if c is not None:
            gates, gate_reason = gate_decision(c, r)
            rec["gates"] = gates
            rec["gate_reason"] = gate_reason
            rec.update({
                # A1-v05-P1-1: these two had NO reader and the hand-maintained
                # audit certified them anyway.  A per-case schema_version is
                # what lets a later reader tell which contract a manifest was
                # written under; the title is the human name for the page.
                "title": c.title,
                "claim_id": c.claim_id, "claim": c.claim,
                "non_claims": list(c.non_claims),
                "purpose": c.purpose, "gate": c.gate,
                "oracle_kind": c.oracle_kind,
                "loading_mode": c.loading_mode, "sample_mode": c.sample_mode,
                "expected_visual": c.expected_visual,
                "failure_means": c.failure_means,
                "owner_on_failure": c.owner_on_failure,
                "known_bug_status": c.known_bug_status,
                "known_bug_id": c.known_bug_id,
                "not_applicable": dict(c.not_applicable),
                "declared_observables": [_contract(o) for o in c.observables],
                # generated, never separately authored — same source as the footer
                "machine_oracle": machine_oracle_text(c),
                "footer": footer_text(c),
                "applicable": c.applicable,
                "applicability_reason": c.applicability_reason,
                "setup_contract": c.setup_contract,
                "preconditions": list(c.preconditions),
                # A4.1: these fields are now executable, no longer future-staged.
                "slots_under_test": list(c.slots_under_test),
                "anti_contamination_preconditions": list(c.anti_contamination_preconditions),
                # A1-v04-P1-1: future-staged fields are RECORDED with the
                # stage that owns them, so "declared but unread" is visible in
                # the evidence rather than discoverable only by source audit.
                "future_staged": {
                    name: {"value": getattr(c, name), "owning_stage": stage}
                    for name, stage in FUTURE_STAGE_FIELDS.items()},
                "figure_contract": (asdict(c.figure_contract)
                                    if c.figure_contract is not None else None),
            })
        doc["cases"].append(rec)
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, default=str)
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Runner
# ─────────────────────────────────────────────────────────────────────────────

def _call(adf, surface: str, spec: dict, case_key: str = "c"):
    """Invoke one public surface with the SAME canonical spec.

    Figures are closed by the caller (``_close``): a harness that leaks
    figures trips matplotlib's open-figure warning and, on a full gallery
    run, its memory.
    """
    kw = dict(spec)
    expr = kw.pop("expr")
    if surface == "draw":
        return adf.draw(expr, **kw), {}
    if surface == "draw_batch":
        return adf.draw_batch({case_key: dict(expr=expr, **kw)}), {"case_key": case_key}
    if surface == "draw_figures":
        return adf.draw_figures([{"plots": [dict(expr=expr, **kw)]}]), {}
    raise AdapterError(f"unknown surface {surface!r}")


def _inapplicable(case: "CaseSpec") -> CaseResult | None:
    """A1-v03-P0-1 — applicability decides whether the product runs AT ALL.

    v1.2 §3.2: "If an environment-gated case is unavailable: SKIP + explicit
    applicability reason."  v03 validated `applicable` and SERIALISED it, and
    then every runner ignored it — measured: an `applicable=False` case called
    the frame factory twice and returned PASS.  A field that describes state
    nothing reads is not a contract.

    The product is never invoked.  That matters beyond tidiness: the whole
    point of an unavailable environment is that touching it raises.
    """
    if case.applicable:
        return None
    return CaseResult(case_id=case.case_id, status=SKIP,
                      detail=f"not applicable: {case.applicability_reason}")


def _close() -> None:
    """Release every open matplotlib figure.  Called after each surface."""
    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass



def run_slot_symmetry(case: CaseSpec,
                      make_eager: Callable[[], Any],
                      make_lazy: Callable[[], Any]) -> CaseResult:
    """Execute one A4 slot-exclusive BOTH/FULL case.

    The public surface is held fixed.  EAGER proves slot-specific alias
    discovery/materialization.  LAZY proves the same numerical semantics plus
    the exact physical branch set loaded by that slot.  Anti-contamination is a
    PRECONDITION: if the alias or lazy physical branches are already present,
    the result is INVALID_FIXTURE rather than PASS.
    """
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        contract = A4_SLOT_CONTRACTS.get(case.case_id)
        if contract is None:
            res.status = INVALID_FIXTURE
            res.detail = f"no A4 slot execution contract for {case.case_id}"
            return res
        if contract.get("runner") != "run_slot_symmetry":
            res.status = INVALID_FIXTURE
            res.detail = "A4 slot execution contract is not bound to run_slot_symmetry"
            return res
        if tuple(case.slots_under_test) != tuple(contract["slots_under_test"]):
            res.status = INVALID_FIXTURE
            res.detail = "slots_under_test does not match A4 execution contract"
            return res
        if tuple(case.anti_contamination_preconditions) != tuple(
                contract["anti_contamination_preconditions"]):
            res.status = INVALID_FIXTURE
            res.detail = "anti_contamination_preconditions do not match A4 execution contract"
            return res
        if case.loading_mode != "BOTH" or case.sample_mode != "FULL":
            res.status = INVALID_FIXTURE
            res.detail = "A4 slot runner requires BOTH/FULL"
            return res
        if len(case.surfaces_under_test) != 1:
            res.status = INVALID_FIXTURE
            res.detail = "A4.1 slot runner holds exactly one public surface fixed"
            return res
        surface = case.surfaces_under_test[0]
        if surface != "draw":
            res.status = INVALID_FIXTURE
            res.detail = "A4.1 implements the draw surface only"
            return res

        alias_name = contract["slot_alias"]
        required_physical = set(contract["required_physical_dependencies"])
        expected_after = set(contract["expected_lazy_loaded_after"])
        unrelated = set(contract["unrelated_physical_branches"])

        eager = make_eager()
        lazy = make_lazy()

        if getattr(eager, "_lazy_reader", None) is not None:
            res.status = INVALID_FIXTURE
            res.detail = "EAGER arm factory returned a lazy ADF"
            return res
        lazy_reader = getattr(lazy, "_lazy_reader", None)
        if lazy_reader is None:
            res.status = INVALID_FIXTURE
            res.detail = "LAZY arm factory did not attach a lazy reader"
            return res

        # M2 anti-contamination checks BEFORE any public call.
        if alias_name not in getattr(eager, "aliases", {}):
            res.status = INVALID_FIXTURE
            res.detail = f"slot alias {alias_name!r} is not registered in EAGER arm"
            return res
        if alias_name not in getattr(lazy, "aliases", {}):
            res.status = INVALID_FIXTURE
            res.detail = f"slot alias {alias_name!r} is not registered in LAZY arm"
            return res
        if alias_name in eager.df.columns:
            res.status = INVALID_FIXTURE
            res.detail = f"M2 contamination: EAGER slot alias {alias_name!r} was pre-materialized"
            return res
        if alias_name in lazy.df.columns:
            res.status = INVALID_FIXTURE
            res.detail = f"M2 contamination: LAZY slot alias {alias_name!r} was pre-materialized"
            return res

        lazy_before = set(lazy_reader.loaded_branches)
        if lazy_before:
            res.status = INVALID_FIXTURE
            res.detail = ("M2 contamination: LAZY reader begins with preloaded physical "
                          f"branches {sorted(lazy_before)}")
            return res
        def call_one(adf):
            kw = dict(case.canonical_spec)
            expr = kw.pop("expr")
            # `lazy=True` is the public alias-materialization hook on an eager
            # ADF and the dependency-loading hook on a lazy ADF.  It is an
            # execution control, not part of the logical plot specification.
            raw = adf.draw(expr, lazy=True, keep_materialized=True, **kw)
            return unwrap("draw", raw)

        eager_payload = call_one(eager)
        if alias_name not in eager.df.columns:
            res.status = FAIL
            res.detail = f"EAGER arm did not materialize slot alias {alias_name!r}"
            return res

        lazy_payload = call_one(lazy)
        if alias_name not in lazy.df.columns:
            res.status = FAIL
            res.detail = f"LAZY arm did not materialize slot alias {alias_name!r}"
            return res
        lazy_after = set(lazy_reader.loaded_branches)
        # Deliberately brittle exact-set check: if the loader starts pulling an
        # additional implicit/index branch, or stops loading a required branch,
        # this A4 contract MUST fail and be consciously re-reviewed rather than
        # being widened reflexively.
        if lazy_after != expected_after:
            res.status = FAIL
            res.detail = ("LAZY slot load set mismatch: "
                          f"expected {sorted(expected_after)}, got {sorted(lazy_after)}")
            return res
        if not required_physical.issubset(lazy_after):
            res.status = FAIL
            res.detail = "LAZY arm did not load the slot-only physical dependency"
            return res
        if unrelated & lazy_after:
            res.status = FAIL
            res.detail = f"LAZY arm loaded unrelated physical branch(es) {sorted(unrelated & lazy_after)}"
            return res

        res.payload_paths = {
            "EAGER/draw": list(eager_payload.path),
            "LAZY/draw": list(lazy_payload.path),
        }
        res.observed["slot_evidence"] = {
            "slot": contract["slots_under_test"][0],
            "alias": alias_name,
            "eager_alias_materialized": True,
            "lazy_alias_materialized": True,
            "lazy_loaded_before": sorted(lazy_before),
            "lazy_loaded_after": sorted(lazy_after),
            "required_physical_dependencies": sorted(required_physical),
            "unrelated_physical_branches": sorted(unrelated),
            "anti_contamination_preconditions": list(case.anti_contamination_preconditions),
        }

        for o in case.observables:
            if o.status != "EXECUTED":
                res.observed[o.name] = {"status": o.status}
                continue
            try:
                _assert_source_matches("run_slot_symmetry", o)
            except HarnessError as exc:
                res.status = INVALID_FIXTURE
                res.detail = str(exc)
                return res
            try:
                eager_v = resolve(eager_payload.stats, o.path, o.access)
                lazy_v = resolve(lazy_payload.stats, o.path, o.access)
            except HarnessError as exc:
                res.status = INVALID_FIXTURE
                res.detail = f"declared observable {o.name!r} not extractable: {exc}"
                return res
            res.observed[o.name] = {"EAGER": eager_v, "LAZY": lazy_v}
            res.observable_contract.append(_contract(o))
            result = compare_observable(o, eager_v, lazy_v)
            res.comparisons.append(comparison_evidence(
                o, result, reference_label="EAGER", candidate_label="LAZY"))
            res.executed_comparisons += 1
            if not result.ok:
                res.status = FAIL
                res.detail = f"{o.name}: EAGER vs LAZY: {result.detail}"
                return res

        if res.executed_comparisons == 0:
            res.status = INVALID_FIXTURE
            res.detail = "A4 slot case executed no numerical comparison"
            return res
        res.status = PASS
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"{type(exc).__name__}: {exc}"
        res.exception = traceback.format_exc(limit=4)
        return res
    finally:
        _close()
        res.wall_time_s = round(time.time() - t0, 4)


def run_consistency(case: CaseSpec, make_adf: Callable[[], Any]) -> CaseResult:
    """Execute one CONSISTENCY case across its declared surfaces."""
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    payloads: dict[str, Payload] = {}
    try:
        for surface in case.surfaces_under_test:
            if surface in case.not_applicable:
                res.skipped_surfaces[surface] = case.not_applicable[surface]
                continue
            raw, kw = _call(make_adf(), surface, case.canonical_spec)
            errs = batch_errors(raw)
            if errs:
                res.status, res.detail = FAIL, f"{surface} reported errors: {errs}"
                return res
            payloads[surface] = unwrap(surface, raw, **kw)
            _close()

        if len(payloads) < 2:
            res.status = SKIP
            res.detail = (f"only {len(payloads)} applicable surface(s); "
                          f"a consistency case needs at least two")
            return res

        res.payload_paths = {s: list(p.path) for s, p in payloads.items()}
        ref_name = next(iter(payloads))
        ref = payloads[ref_name]

        for o in case.observables:
            if o.status.startswith("DEFERRED:") or o.status == "NOT_EXTRACTABLE":
                res.observed[o.name] = {"status": o.status}
                continue
            # §14: an observable a case claims to compare MUST resolve, on
            # every applicable surface.  Unresolvable == INVALID_FIXTURE.
            try:
                vals = {s: resolve(p.stats, o.path, o.access)
                        for s, p in payloads.items()}
            except HarnessError as exc:
                res.status = INVALID_FIXTURE
                res.detail = f"declared observable {o.name!r} not extractable: {exc}"
                return res
            res.observed[o.name] = {s: v for s, v in vals.items()}
            try:
                _assert_source_matches("run_consistency", o)
            except HarnessError as exc:
                res.status = INVALID_FIXTURE
                res.detail = str(exc)
                return res
            res.observable_contract.append(_contract(o))
            for sname, v in vals.items():
                if sname == ref_name:
                    continue
                result = compare_observable(o, vals[ref_name], v)
                res.comparisons.append(comparison_evidence(
                    o, result, reference_label=ref_name, candidate_label=sname))
                ok, why = result.ok, result.detail
                res.executed_comparisons += 1
                if not ok:
                    res.status = FAIL
                    res.detail = f"{o.name}: {ref_name} vs {sname}: {why}"
                    return res
        # A1-P0-1: PASS requires that something was actually compared.
        if res.executed_comparisons == 0:
            res.status = INVALID_FIXTURE
            res.detail = ("no observable was executed and compared; a case that "
                          "compares nothing cannot PASS")
            return res
        res.status = PASS
        return res
    except AdapterError as exc:
        res.status, res.detail, res.exception = FAIL, f"adapter: {exc}", repr(exc)
        return res
    except Exception as exc:                       # product exception
        res.status = FAIL
        res.detail = f"{type(exc).__name__}: {exc}"
        res.exception = traceback.format_exc(limit=4)
        return res
    finally:
        res.wall_time_s = round(time.time() - t0, 4)


def run_correctness(case: CaseSpec, make_adf: Callable[[], Any],
                    anchor: Callable[[Any], dict],
                    raw_factory: Callable[[], Any] | None = None) -> CaseResult:
    """Execute one CORRECTNESS case: public result vs an INDEPENDENT anchor.

    ``anchor`` returns {observable_name: value} computed with NumPy/pandas only.
    It must not call ADF or dfdraw.

    A1-P0-2 — the anchor input is taken BEFORE the public surface runs, and is a
    deep copy the product cannot reach.  v01 computed ``anchor(adf.df)`` AFTER
    ``draw()``, so a product that mutated its own frame made the "independent"
    oracle agree with the wrong answer: measured pristine mean 15.0, product
    wrote 0.0, public 0.0, independent 0.0, PASS.  An oracle that reads
    product-mutated state is not independent.

    ``raw_factory`` overrides the snapshot with a frame the harness owns
    outright — preferred where the input can be re-read from source.
    """
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        surface = (case.surfaces_under_test or ("draw",))[0]
        adf = make_adf()
        # PRISTINE input, captured before the product can touch it.
        pristine = raw_factory() if raw_factory is not None else adf.df.copy(deep=True)
        expected = anchor(pristine)
        raw, kw = _call(adf, surface, case.canonical_spec)
        payload = unwrap(surface, raw, **kw)
        _close()
        res.payload_paths = {surface: list(payload.path)}
        for o in case.observables:
            if o.status.startswith("DEFERRED:") or o.status == "NOT_EXTRACTABLE":
                res.observed[o.name] = {"status": o.status}
                continue
            if o.name not in expected:
                res.status = INVALID_FIXTURE
                res.detail = f"anchor supplied no value for {o.name!r}"
                return res
            try:
                got = resolve(payload.stats, o.path, o.access)
            except HarnessError as exc:
                res.status = INVALID_FIXTURE
                res.detail = f"declared observable {o.name!r} not extractable: {exc}"
                return res
            res.observed[o.name] = {"public": got, "independent": expected[o.name]}
            try:
                _assert_source_matches("run_correctness", o)
            except HarnessError as exc:
                res.status = INVALID_FIXTURE
                res.detail = str(exc)
                return res
            res.observable_contract.append(_contract(o))
            result = compare_observable(o, expected[o.name], got)
            res.comparisons.append(comparison_evidence(
                o, result, reference_label="independent", candidate_label=surface))
            ok, why = result.ok, result.detail
            res.executed_comparisons += 1
            if not ok:
                res.status, res.detail = FAIL, f"{o.name}: {why}"
                return res
        if res.executed_comparisons == 0:
            res.status = INVALID_FIXTURE
            res.detail = ("no observable was executed and compared; a case that "
                          "compares nothing cannot PASS")
            return res
        res.status = PASS
        return res
    except AdapterError as exc:
        res.status, res.detail, res.exception = FAIL, f"adapter: {exc}", repr(exc)
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"{type(exc).__name__}: {exc}"
        res.exception = traceback.format_exc(limit=4)
        return res
    finally:
        res.wall_time_s = round(time.time() - t0, 4)


def run_error_contract(case: CaseSpec, make_adf: Callable[[], Any],
                       surface: str, expect_substring: str) -> CaseResult:
    """A refusal is part of the contract.  This proves the refusal happens AND
    that its message names the unsupported condition."""
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        try:
            _call(make_adf(), surface, case.canonical_spec)
            _close()
        except Exception as exc:
            msg = str(exc)
            if expect_substring in msg:
                res.status, res.detail = PASS, f"refused as required: {msg[:120]}"
            else:
                res.status = FAIL
                res.detail = (f"refused, but the message does not name the "
                              f"condition {expect_substring!r}: {msg[:160]}")
            return res
        res.status = FAIL
        res.detail = "did NOT refuse; the guard is gone or the surface changed"
        return res
    finally:
        res.wall_time_s = round(time.time() - t0, 4)


# ── the gate state matrix ───────────────────────────────────────────────────
#
# A1-v05-P0-1 and P1-2 are the same defect from two directions: I kept fixing
# ONE cell of  gate x applicable x known_bug_status x status  instead of
# enumerating the space.  v05 gated CORE_MANDATORY+SKIP, ERROR_CONTRACT+FAIL
# and applicable+SUPPORTED+FAIL, and an applicable ENVIRONMENT_GATED case that
# could not perform its proof returned SKIP -> strict exit 0.
#
# Every reachable combination now has a stated verdict and a reason.
GATE_MATRIX = {
    # (applicable, status)            -> (gates, why)
    (False, SKIP):             (False, "environment unavailable; SKIP is the "
                                       "contracted outcome"),
    (True,  SKIP):             (True,  "the case was applicable and proved "
                                       "nothing; a silent non-proof is the "
                                       "false-green this harness exists to "
                                       "prevent"),
    (True,  INVALID_FIXTURE):  (True,  "the harness could not establish its "
                                       "own claim"),
    (False, INVALID_FIXTURE):  (True,  "a fixture defect is a harness defect "
                                       "whether or not the case applied"),
    (True,  DIAGNOSTIC):       (False, "observational by construction"),
    (False, DIAGNOSTIC):       (False, "observational by construction"),
}


def gate_decision(case: "CaseSpec", result: "CaseResult") -> tuple:
    """(gates, reason) for one result.  Total over the reachable state space."""
    key = (bool(case.applicable), result.status)
    if key in GATE_MATRIX:
        return GATE_MATRIX[key]
    if result.status == PASS:
        if case.purpose in ("INVARIANCE", "CORRECTNESS") \
                and result.executed_comparisons == 0:
            return True, ("PASS with zero executed comparisons; a machine-gated "
                          "numerical case proved no observable")
        return False, "passed"
    if result.status == FAIL:
        # ERROR_CONTRACT proves a refusal still happens; its known_bug_id is
        # PROVENANCE for why the guard exists, not a licence to lose the guard.
        if case.purpose == "ERROR_CONTRACT":
            return True, "an error-contract guard stopped refusing"
        if not case.applicable:
            return False, ("inapplicable cases do not execute; a FAIL here is "
                           "unreachable and non-gating")
        if case.known_bug_status in ("KNOWN_BUG", "EXPECTED_FAIL"):
            return False, f"failure expected: {case.known_bug_id}"
        return True, "an applicable case failed"
    return True, f"unhandled status {result.status!r} — fail closed"


def reconcile(results: Sequence[CaseResult],
              cases: Sequence[CaseSpec]) -> dict:
    """THE single authority on declared-vs-observed reconciliation.

    A1-v07-P1-2 / P2-1.  v07 had `strict_exit_code` and `coverage_gaps`
    computing overlapping facts independently, and `write_manifest` reading
    neither — so a declared case that produced no result gated correctly and
    was ABSENT FROM THE EVIDENCE.  Measured: declared ['A','B_NEVER_RAN'],
    manifest listed ['A'] only.  I named this drift risk in the v07 CRR §7
    item 4 and shipped it.

    The gate, the coverage report and the manifest are now all derived from
    this one function.  Two producers of one fact is how they disagree.

    A1-v07-P0-1.  An EMPTY declared set is a failure, not a vacuous success.
    v07 returned strict exit 0 for zero cases and zero results — a harness
    that ran nothing reporting that everything is fine.  A registry-loading
    failure, a filter matching nothing, or a bad path all produced success.
    """
    declared = [c.case_id for c in cases]
    counts: dict = {}
    for r in results:
        counts[r.case_id] = counts.get(r.case_id, 0) + 1
    by_case = {c.case_id: c for c in cases}
    first: dict = {}
    for r in results:
        first.setdefault(r.case_id, r)

    gaps: list = []
    undeclared: list = []
    missing: list = []
    duplicated: list = []
    gating: list = []

    if not cases:
        gaps.append("the declared case set is EMPTY; a harness that declares "
                    "nothing cannot report success")

    if len(set(declared)) != len(declared):
        dupes = sorted({d for d in declared if declared.count(d) > 1})
        gaps.append(f"duplicate declared case_id(s): {dupes}")

    for cid in counts:
        if cid not in by_case:
            undeclared.append(cid)
            gaps.append(f"{cid}: result for a case that was never declared")

    for c in cases:
        n = counts.get(c.case_id, 0)
        if n == 0:
            missing.append(c.case_id)
            gaps.append(f"{c.case_id}: declared but produced no result")
            continue
        if n > 1:
            duplicated.append(c.case_id)
            gaps.append(f"{c.case_id}: produced {n} results; expected exactly 1")
            continue
        g, why = gate_decision(c, first[c.case_id])
        if g:
            gating.append({"case_id": c.case_id,
                           "status": first[c.case_id].status, "reason": why})

    if undeclared:
        exit_code = 2
    elif gaps or gating:
        exit_code = 1
    else:
        exit_code = 0

    return {"declared": declared, "n_declared": len(declared),
            "n_results": len(results), "missing": missing,
            "duplicated": duplicated, "undeclared": undeclared,
            "gating": gating, "gaps": gaps, "exit_code": exit_code}


def strict_exit_code(results: Sequence[CaseResult],
                     cases: Sequence[CaseSpec]) -> int:
    """Non-zero when reconciliation reports any gap or gating verdict.

    Derived from reconcile(); holds no policy of its own.
    """
    return reconcile(results, cases)["exit_code"]


def coverage_gaps(results: Sequence[CaseResult],
                  cases: Sequence[CaseSpec]) -> list:
    """Human-readable gaps.  Derived from reconcile(); holds no policy."""
    return reconcile(results, cases)["gaps"]
