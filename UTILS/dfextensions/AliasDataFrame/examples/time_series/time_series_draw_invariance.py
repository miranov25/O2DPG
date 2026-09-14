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

import copy
import hashlib
import json
import os
import platform
import re
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

SCHEMA_VERSION = "13.77.A6.4.v04"

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

# A6.1 — reference-policy semantics.  This is deliberately a small, explicit
# policy vocabulary rather than another reference framework.  The structural
# contract follows the tested PHASE_13_79 contract-file pattern; the run
# manifest remains owned by write_manifest() below.
REFERENCE_POLICY = ("named-immutable", "same-process")
REFERENCE_POLICY_SEMANTICS = {
    "named-immutable": {
        "cross_run_reference": True,
        "requires_explicit_reference": True,
        "implicit_latest_allowed": False,
    },
    "same-process": {
        "cross_run_reference": False,
        "requires_explicit_reference": False,
        "implicit_latest_allowed": False,
    },
}

# Existing A5 real-data manifests already record these provenance fields.
# A6.1 derives one reference-identity view from that existing owner instead of
# introducing a second sampling/input-identity implementation.
_REFERENCE_IDENTITY_BASE_KEYS = (
    "input_path", "input_size_bytes", "input_mtime_ns", "loading_mode",
    "sample_mode", "sampling_algorithm", "source_rows", "selected_rows",
)
_REFERENCE_IDENTITY_FRACTION_KEYS = (
    "sample_fraction", "sample_seed", "index_digest_sha256",
)

# A6.1-v02: comparison-ready identity completeness is single-sourced here and
# consumed by derivation, comparison, and manifest emission.  FRACTION is the
# deterministic 20% reference path already owned by A5.  FULL acceptance is a
# later A6 checkpoint, so its minimum comparison-ready identity remains limited
# to the provenance fields the current FULL runner already owns.
_REFERENCE_IDENTITY_FULL_REQUIRED_KEYS = (
    "input_path", "input_size_bytes", "input_mtime_ns", "loading_mode",
    "sample_mode", "source_rows",
)
_REFERENCE_IDENTITY_FRACTION_REQUIRED_KEYS = _REFERENCE_IDENTITY_BASE_KEYS + (
    "sample_fraction", "sample_seed", "index_digest_sha256",
)

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
    "run_subframe_slot_symmetry": ("STATS",),
    "run_a5_full_stack": ("INDEPENDENT",),
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
        "expected_eager_new_aliases": ("slot_keep",),
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
        "expected_eager_new_aliases": ("slot_expr",),
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
        "expected_eager_new_aliases": ("slot_weight",),
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
        "expected_eager_new_aliases": ("slot_group",),
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
        "expected_eager_new_aliases": ("slot_facet",),
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
        "expected_eager_new_aliases": ("slot_compound",),
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
        "expected_eager_new_aliases": ("slot_selection_vector",),
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
        "expected_eager_new_aliases": ("slot_weights_vector",),
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
    "I3-SUBFRAME-EXPR-01": {
        "runner": "run_subframe_slot_symmetry",
        "slots_under_test": ("subframe_qualified_expression",),
        "qualified_reference": "S.count",
        "structural_baseline_physical_dependencies": ("kbin",),
        "expected_lazy_loaded_before": ("kbin",),
        "expected_lazy_loaded_after": ("kbin", "x"),
        "unrelated_physical_branches": ("decoy",),
        "anti_contamination_preconditions": (
            "registered eager subframe S contains kbin/count before each arm",
            "S.count is absent from the parent frame and appears only in expr",
            "lazy parent begins with structural join key kbin loaded and x/decoy unloaded",
            "the public draw call must load x without loading decoy",
        ),
    },
    "I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01": {
        "runner": "run_error_contract",
        "slots_under_test": ("selection_vector",),
        "qualified_reference": "S.count",
        "expected_known_bug_id": "BUG_20260701_ADF_subframe_ref_slot_symmetry",
        "expected_surfaces_under_test": ("draw",),
        "expected_loading_mode": "BOTH",
        "expected_sample_mode": "FULL",
        "bug_id_text_is_deliberate_contract": True,
        "anti_contamination_preconditions": (
            "registered subframe S is present before the public call",
            "S.count appears only inside selection_vector",
            "the request must refuse before dfdraw evaluates an unresolved subframe reference",
        ),
    },
    "I3-SUBFRAME-WEIGHTS-VECTOR-REFUSAL-01": {
        "runner": "run_error_contract",
        "slots_under_test": ("weights_vector",),
        "qualified_reference": "S.count",
        "expected_known_bug_id": "BUG_20260701_ADF_subframe_ref_slot_symmetry",
        "expected_surfaces_under_test": ("draw",),
        "expected_loading_mode": "BOTH",
        "expected_sample_mode": "FULL",
        "bug_id_text_is_deliberate_contract": True,
        "anti_contamination_preconditions": (
            "registered subframe S is present before the public call",
            "S.count appears only inside weights_vector",
            "the request must refuse before dfdraw evaluates an unresolved subframe reference",
        ),
    },

}

A4_POSITIVE_CASE_IDS = (
    "I3-SELECTION-01",
    "I3-EXPR-01",
    "I3-WEIGHTS-01",
    "I3-GROUP-BY-01",
    "I3-FACET-BY-01",
    "I3-COMPOUND-EXPR-01",
    "I3-SELECTION-VECTOR-01",
    "I3-WEIGHTS-VECTOR-01",
    "I3-SUBFRAME-EXPR-01",
)

A4_ERROR_CASE_IDS = (
    "I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01",
    "I3-SUBFRAME-WEIGHTS-VECTOR-REFUSAL-01",
)

A4_CLOSURE_LEDGER_EXPECTATIONS = {
    "slot_exclusivity": {"status": "CLOSED_BY_A4_4", "owner": None},
    "eager_non_target_selectivity": {"status": "CLOSED_BY_A4_4", "owner": None},
    "catalogue_bidirectional": {"status": "CLOSED_BY_A4_4", "owner": None},
    "subframe_scalar_causality": {"status": "CLOSED_BY_A4_4", "owner": None},
    "subframe_vector_boundary": {
        "status": "ACCEPTED_ERROR_CONTRACT",
        "owner": "BUG_20260701_ADF_subframe_ref_slot_symmetry",
    },
    "facet_overlay_real_user_bug": {
        "status": "LATER_NOT_A4",
        "owner": "BUG_dfdraw_20260822_facet_by_overlay_unsupported",
    },
    "interval_label_nan_real_user_bug": {
        "status": "LATER_NOT_A4",
        "owner": "BUG_dfdraw_20260822_format_interval_label_nan_crash",
    },
    "bug_id_text_brittleness": {
        "status": "ACCEPTED_DELIBERATE_CONTRACT",
        "owner": None,
    },
}
A4_REQUIRED_LEDGER_IDS = tuple(A4_CLOSURE_LEDGER_EXPECTATIONS)

A4_CLOSURE_LEDGER = (
    {"id": "slot_exclusivity", "status": "CLOSED_BY_A4_4"},
    {"id": "eager_non_target_selectivity", "status": "CLOSED_BY_A4_4"},
    {"id": "catalogue_bidirectional", "status": "CLOSED_BY_A4_4"},
    {"id": "subframe_scalar_causality", "status": "CLOSED_BY_A4_4"},
    {"id": "subframe_vector_boundary", "status": "ACCEPTED_ERROR_CONTRACT",
     "owner": "BUG_20260701_ADF_subframe_ref_slot_symmetry"},
    {"id": "facet_overlay_real_user_bug", "status": "LATER_NOT_A4",
     "owner": "BUG_dfdraw_20260822_facet_by_overlay_unsupported"},
    {"id": "interval_label_nan_real_user_bug", "status": "LATER_NOT_A4",
     "owner": "BUG_dfdraw_20260822_format_interval_label_nan_crash"},
    {"id": "bug_id_text_brittleness", "status": "ACCEPTED_DELIBERATE_CONTRACT",
     "rationale": "ERROR_CONTRACT cases deliberately require the owning bug ID in exception text"},
)


def _a4_text_contains_target(value: Any, target: str) -> bool:
    """Return whether a canonical-spec value contains one exact target token/ref.

    Qualified references are deliberately token-exact: ``S.count`` matches
    ``S.count:x`` and ``S.count+1`` but not ``S.count2`` or ``XS.count``.
    This brittleness is closure-significant because A4 attributes causality to
    one exact alias/reference identity rather than to a textual prefix.
    """
    if isinstance(value, str):
        boundary = r"[A-Za-z0-9_.]" if "." in target else r"[A-Za-z0-9_]"
        return re.search(
            rf"(?<!{boundary}){re.escape(target)}(?!{boundary})", value) is not None
    if isinstance(value, (list, tuple)):
        return any(_a4_text_contains_target(v, target) for v in value)
    return False


def a4_actual_target_slots(case: CaseSpec, contract: dict) -> tuple[str, ...]:
    """Derive the public A4 slots that actually contain the contract target."""
    target = contract.get("slot_alias") or contract.get("qualified_reference")
    if not target:
        return ()
    primary = tuple(contract.get("slots_under_test", ()))
    primary_slot = primary[0] if len(primary) == 1 else None
    found = []
    for key in ("expr", "selection", "weights", "group_by", "facet_by",
                "selection_vector", "weights_vector"):
        if not _a4_text_contains_target(case.canonical_spec.get(key), target):
            continue
        if key == "expr" and primary_slot in ("compound_expression",
                                               "subframe_qualified_expression"):
            found.append(primary_slot)
        else:
            found.append(key)
    return tuple(sorted(set(found)))


def _a4_contract_reconciliation_errors(case: CaseSpec, contract: dict) -> list[str]:
    """Closure-level proof-identity reconciliation for one A4 case."""
    errors = []
    runner = contract.get("runner")
    if tuple(case.slots_under_test) != tuple(contract.get("slots_under_test", ())):
        errors.append("slots_under_test drift")
    if tuple(case.anti_contamination_preconditions) != tuple(
            contract.get("anti_contamination_preconditions", ())):
        errors.append("anti_contamination_preconditions drift")
    actual_slots = a4_actual_target_slots(case, contract)
    if actual_slots != tuple(sorted(case.slots_under_test)):
        errors.append(
            f"slot-exclusivity drift: declared {tuple(case.slots_under_test)!r}, actual {actual_slots!r}")
    if runner in ("run_slot_symmetry", "run_subframe_slot_symmetry"):
        if case.purpose != "INVARIANCE" or case.oracle_kind != "CONSISTENCY":
            errors.append("positive A4 case lost INVARIANCE/CONSISTENCY proof identity")
        if case.loading_mode != "BOTH" or case.sample_mode != "FULL":
            errors.append("positive A4 case lost BOTH/FULL loading contract")
        if not any(o.status == "EXECUTED" for o in case.observables):
            errors.append("positive A4 case has no executable numerical companion")
        if runner == "run_slot_symmetry":
            alias = contract.get("slot_alias")
            if not alias:
                errors.append("slot-symmetry contract has no slot_alias")
            expected_new = tuple(contract.get("expected_eager_new_aliases", ()))
            if alias and expected_new != (alias,):
                errors.append("expected_eager_new_aliases is not exactly the target alias")
            for key in ("required_physical_dependencies", "expected_lazy_loaded_after",
                        "unrelated_physical_branches"):
                if not contract.get(key):
                    errors.append(f"slot-symmetry contract missing {key}")
        else:
            if not contract.get("qualified_reference"):
                errors.append("subframe slot contract has no qualified_reference")
            if tuple(contract.get("expected_lazy_loaded_before", ())) != tuple(
                    contract.get("structural_baseline_physical_dependencies", ())):
                errors.append("subframe structural baseline/load-before contract drift")
    elif runner == "run_error_contract":
        if case.purpose != "ERROR_CONTRACT":
            errors.append("refusal case lost ERROR_CONTRACT purpose")
        if case.known_bug_status != "KNOWN_BUG" or not case.known_bug_id:
            errors.append("refusal case lost known-bug ownership")
        expected_bug = contract.get("expected_known_bug_id")
        if case.known_bug_id != expected_bug:
            errors.append(
                f"refusal known_bug_id drift: expected {expected_bug!r}, observed {case.known_bug_id!r}")
        if tuple(case.surfaces_under_test) != tuple(contract.get("expected_surfaces_under_test", ())):
            errors.append("refusal surfaces_under_test drift")
        if case.loading_mode != contract.get("expected_loading_mode"):
            errors.append("refusal loading_mode drift")
        if case.sample_mode != contract.get("expected_sample_mode"):
            errors.append("refusal sample_mode drift")
        if not case.negative_control:
            errors.append("refusal case lost negative control")
        if not contract.get("bug_id_text_is_deliberate_contract"):
            errors.append("refusal case does not declare bug-ID text brittleness as deliberate")
    else:
        errors.append(f"unknown A4 runner {runner!r}")
    return errors


def a4_closure_reconciliation(cases: Sequence[CaseSpec]) -> dict:
    """Fail-closed A4 catalogue/contract/ownership reconciliation."""
    by_id = {}
    duplicates = []
    for case in cases:
        if case.case_id in by_id:
            duplicates.append(case.case_id)
        by_id[case.case_id] = case
    expected_positive = set(A4_POSITIVE_CASE_IDS)
    expected_error = set(A4_ERROR_CASE_IDS)
    expected_all = expected_positive | expected_error
    actual_a4 = {cid for cid in by_id if cid.startswith("I3-")}
    contract_ids = set(A4_SLOT_CONTRACTS)
    positive_contract_ids = {cid for cid, c in A4_SLOT_CONTRACTS.items()
                             if c.get("runner") in ("run_slot_symmetry",
                                                    "run_subframe_slot_symmetry")}
    error_contract_ids = {cid for cid, c in A4_SLOT_CONTRACTS.items()
                          if c.get("runner") == "run_error_contract"}

    missing = sorted(expected_all - actual_a4)
    unexpected = sorted(actual_a4 - expected_all)
    stale_contracts = sorted(contract_ids - expected_all)
    uncontracted = sorted(expected_all - contract_ids)
    mapping_errors = []
    if positive_contract_ids != expected_positive:
        mapping_errors.append(
            f"positive case/contract mismatch: expected {sorted(expected_positive)}, contracts {sorted(positive_contract_ids)}")
    if error_contract_ids != expected_error:
        mapping_errors.append(
            f"error case/contract mismatch: expected {sorted(expected_error)}, contracts {sorted(error_contract_ids)}")

    contract_drift = []
    for cid in sorted(expected_all & set(by_id) & contract_ids):
        for error in _a4_contract_reconciliation_errors(by_id[cid], A4_SLOT_CONTRACTS[cid]):
            contract_drift.append({"case_id": cid, "detail": error})

    ledger_ids = [row.get("id") for row in A4_CLOSURE_LEDGER]
    ledger_duplicate_ids = sorted({
        ledger_id for ledger_id in ledger_ids
        if ledger_id is not None and ledger_ids.count(ledger_id) > 1
    })
    ledger_by_id = {row.get("id"): dict(row) for row in A4_CLOSURE_LEDGER}
    ledger_missing = sorted(set(A4_REQUIRED_LEDGER_IDS) - set(ledger_by_id))
    ledger_unexpected = sorted(set(ledger_by_id) - set(A4_REQUIRED_LEDGER_IDS))
    ledger_open = sorted(row["id"] for row in ledger_by_id.values()
                         if row.get("status") == "OPEN_BLOCKER")
    ledger_drift = []
    for ledger_id, expected in A4_CLOSURE_LEDGER_EXPECTATIONS.items():
        observed = ledger_by_id.get(ledger_id)
        if observed is None:
            continue
        for field_name in ("status", "owner"):
            want = expected.get(field_name)
            got = observed.get(field_name)
            if got != want:
                ledger_drift.append({
                    "id": ledger_id,
                    "field": field_name,
                    "expected": want,
                    "observed": got,
                })

    subframe_owned = {
        "I3-SUBFRAME-EXPR-01",
        "I3-SUBFRAME-SELECTION-VECTOR-REFUSAL-01",
        "I3-SUBFRAME-WEIGHTS-VECTOR-REFUSAL-01",
    }
    subframe_missing = sorted(subframe_owned - actual_a4)

    blockers = []
    blockers += [f"missing case {x}" for x in missing]
    blockers += [f"unexpected A4 case {x}" for x in unexpected]
    blockers += [f"stale contract {x}" for x in stale_contracts]
    blockers += [f"case without contract {x}" for x in uncontracted]
    blockers += mapping_errors
    blockers += [f"{row['case_id']}: {row['detail']}" for row in contract_drift]
    blockers += [f"duplicate closure-ledger item {x}" for x in ledger_duplicate_ids]
    blockers += [f"missing closure-ledger item {x}" for x in ledger_missing]
    blockers += [f"unexpected closure-ledger item {x}" for x in ledger_unexpected]
    blockers += [f"open closure-ledger blocker {x}" for x in ledger_open]
    blockers += [
        f"closure-ledger drift {row['id']}.{row['field']}: expected {row['expected']!r}, observed {row['observed']!r}"
        for row in ledger_drift
    ]
    blockers += [f"missing subframe ownership case {x}" for x in subframe_missing]
    if duplicates:
        blockers += [f"duplicate A4 case {x}" for x in sorted(set(duplicates))]

    ready = not blockers
    return {
        "status": "READY_FOR_CLOSURE" if ready else "BLOCKED",
        "closure_ready": ready,
        "evidence_scope": "A4 expression-slot dependency-discovery causality only",
        "execution_context": {
            "fresh_execution_verdict": False,
            "meaning": "declaration/disposition reconciliation over banked A4 execution evidence",
        },
        "positive_case_ids": sorted(expected_positive),
        "error_case_ids": sorted(expected_error),
        "actual_a4_case_ids": sorted(actual_a4),
        "contract_ids": sorted(contract_ids),
        "missing_case_ids": missing,
        "unexpected_case_ids": unexpected,
        "stale_contract_ids": stale_contracts,
        "uncontracted_case_ids": uncontracted,
        "duplicate_case_ids": sorted(set(duplicates)),
        "contract_drift": contract_drift,
        "subframe_ownership_missing": subframe_missing,
        "ledger_duplicate_ids": ledger_duplicate_ids,
        "ledger_missing": ledger_missing,
        "ledger_unexpected": ledger_unexpected,
        "ledger_open_blockers": ledger_open,
        "ledger_drift": ledger_drift,
        "historical_ledger": [dict(row) for row in A4_CLOSURE_LEDGER],
        "accepted_non_claims": [
            "subframe-qualified selection_vector/weights_vector remain explicit ERROR_CONTRACT boundaries",
            "BUG_dfdraw_20260822_facet_by_overlay_unsupported remains later-owner work",
            "BUG_dfdraw_20260822_format_interval_label_nan_crash remains later-owner work",
            "the fast subframe causality fixture uses the established lazy-main/eager-subframe shape with structural join key preloaded",
            "A4 closure does not close Stage A or PHASE_13_77",
        ],
        "blockers": blockers,
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
    """A4 slot-exclusive cases implemented through A4.4 closure hardening.

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
    cases.append(CaseSpec(
        case_id="I3-SUBFRAME-EXPR-01",
        claim_id="I3.subframe_qualified_expression.A4.4",
        title="subframe-qualified expr causality in eager and lazy-main modes",
        claim=("the supported scalar S.count:x request resolves the registered subframe in both arms and "
               "the lazy-main arm loads only x beyond the declared structural join-key baseline"),
        failure_means=("the supported scalar subframe reference no longer resolves, the lazy main frame loads "
                       "the wrong physical set, or the structural join-key precondition is silently bypassed"),
        expected_visual="one S.count:x profile",
        owner_on_failure="ADF",
        purpose="INVARIANCE",
        gate="CORE_MANDATORY",
        oracle_kind="CONSISTENCY",
        loading_mode="BOTH",
        sample_mode="FULL",
        canonical_spec={"expr": "S.count:x", "type": "profile", "bins": 4,
                        "return_data": True, "auto_title": True},
        applicable=True,
        setup_contract=("both arms register the same eager S(kbin,count) subframe; the LAZY main frame preloads "
                        "only structural join key kbin because the established draw-subframe contract requires "
                        "the join key to exist before the temporary merge"),
        preconditions=("S.count is absent from the parent frame",
                       "S.count exists in the registered subframe",
                       "LAZY parent has only kbin loaded before draw",
                       "x and decoy are unloaded before draw"),
        figure_contract=FigureContract(
            expected_panels="one panel",
            panel_roles="main: supported subframe-qualified profile",
            expected_traces="one profile",
            expected_group_count="1",
            primary_comparison="EAGER versus lazy-main subframe profile and exact parent load set",
            residual_definition="lazy-main numerical observable minus eager observable",
            accepted_envelope="n exact; y_mean within tolerance; lazy parent load set exactly {kbin,x}",
            case_ids=("I3-SUBFRAME-EXPR-01",),
            proof_kind="CONSISTENCY",
        ),
        surfaces_under_test=("draw",),
        slots_under_test=("subframe_qualified_expression",),
        observables=(
            Observable("n", "STATS", "FLAT", "n"),
            Observable("y_mean", "STATS", "ARRAY", "profile_data.y_mean",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same supported subframe-qualified profile in both loading arms"),
        ),
        non_claims=(
            "this fast A4 fixture uses an eager registered subframe; lazy-subframe file loading is later Stage-A/A5 work",
            "the structural join key kbin is an explicit setup baseline, not claimed as slot-discovered",
            "subframe-qualified vector forms remain explicit ERROR_CONTRACT boundaries",
        ),
        anti_contamination_preconditions=A4_SLOT_CONTRACTS["I3-SUBFRAME-EXPR-01"]["anti_contamination_preconditions"],
        negative_control="GLOBAL_MUTATION:A4.4 subframe baseline/load-set drift -> INVALID_FIXTURE or FAIL",
        reference_policy="named-immutable",
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
# A5.1 — first lazy/eager + keyed-subframe + group_by full-stack composition
# ─────────────────────────────────────────────────────────────────────────────

A5_1_CASE_ID = "I4-SUBFRAME-GROUPBY-01"
A5_1_CONTRACT = {
    "runner": "run_a5_full_stack",
    "surface": "draw",
    "qualified_reference": "S.count",
    "group_by": "group",
    "structural_baseline_physical_dependencies": ("kbin",),
    "expected_lazy_loaded_before": ("kbin",),
    "expected_lazy_loaded_after": ("group", "kbin", "x"),
    "unrelated_physical_branches": ("decoy",),
}


def a5_cases() -> tuple[CaseSpec, ...]:
    """Return the first bounded A5 full-stack composition case.

    A5.1 deliberately composes dimensions that A3/A4 proved separately:

        loading mode BOTH
        + keyed subframe S.count
        + physical group_by
        + public draw()
        + independent grouped-bin oracle

    This is still a fast synthetic acceptance case.  It does not claim the A6
    real-data/reference run, lazy child-file loading, lifecycle mutation
    coverage, or the known subframe-vector capability boundary.
    """
    cid = A5_1_CASE_ID
    return (CaseSpec(
        case_id=cid,
        claim_id="I4.subframe_groupby.A5.1",
        title="keyed-subframe grouped profile is correct in eager and lazy parent modes",
        claim=("the same keyed S.count:x grouped profile is numerically correct "
               "against an independent grouped-bin oracle in EAGER and LAZY "
               "parent modes"),
        failure_means=("the full-stack composition changes keyed-subframe join, "
                       "group membership, profile binning/reduction, or lazy "
                       "dependency loading semantics"),
        expected_visual=("one S.count:x grouped profile with one trace per "
                         "physical group and four populated x bins per group"),
        owner_on_failure="ADF",
        purpose="CORRECTNESS",
        gate="CORE_MANDATORY",
        oracle_kind="CORRECTNESS",
        loading_mode="BOTH",
        sample_mode="FULL",
        canonical_spec={
            "expr": "S.count:x",
            "type": "profile",
            "bins": 4,
            "range": (0.0, 1.0),
            "group_by": "group",
            "return_data": True,
            "auto_title": True,
        },
        applicable=True,
        setup_contract=("parent contains x, kbin, group and an unrelated decoy; "
                        "registered eager subframe S contains one count per kbin; "
                        "the LAZY parent preloads only structural join key kbin"),
        preconditions=(
            "S.count exists only in the registered keyed subframe",
            "the parent has repeated kbin values and two populated physical groups",
            "every group has a populated row set in each of the four explicit x bins",
            "the LAZY parent begins with only structural join key kbin loaded",
            "decoy is absent from the required dependency set",
        ),
        figure_contract=FigureContract(
            expected_panels="one panel",
            panel_roles="main: S.count versus x grouped profile",
            expected_traces="two group_by profile traces",
            expected_group_count="2",
            primary_comparison=("independent group/count/x_center/y_mean oracle "
                                "versus both EAGER/draw and LAZY/draw"),
            residual_definition=("public grouped-bin observable minus independent "
                                 "NumPy/pandas grouped-bin reference"),
            accepted_envelope=("group/count exact; floating x_center/y_mean within "
                               "declared tolerance; LAZY load set exactly "
                               "{kbin,x,group}"),
            case_ids=(cid,),
            proof_kind="CORRECTNESS",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable("group", "INDEPENDENT", "ARRAY", "profile_data.group"),
            Observable("count", "INDEPENDENT", "ARRAY", "profile_data.count"),
            Observable("x_center", "INDEPENDENT", "ARRAY",
                       "profile_data.x_center", comparator="close",
                       atol=1e-14, rtol=1e-12,
                       rationale="explicit bin centers are floating values"),
            Observable("y_mean", "INDEPENDENT", "ARRAY",
                       "profile_data.y_mean", comparator="close",
                       atol=1e-14, rtol=1e-12,
                       rationale=("independent keyed-subframe grouped-bin mean "
                                  "versus public floating reduction")),
        ),
        non_claims=(
            "A5.1 is a fast synthetic composition case, not the A6 real-data reference run",
            "the registered subframe is eager; lazy child-file loading remains later A5 work",
            "subframe-qualified selection_vector/weights_vector remain owned by BUG_20260701_ADF_subframe_ref_slot_symmetry",
            "dynamic lifecycle defects carried from PHASE_13_76 are not repaired by this case",
        ),
        negative_control="FAMILY_MUTATION:A5.1-INDEPENDENT-GROUP-BIN-CORRUPTION",
        reference_policy="named-immutable",
    ),)


# ─────────────────────────────────────────────────────────────────────────────
# A5.2 — environment-gated real-data CalibVertex/G7.32 acceptance
# ─────────────────────────────────────────────────────────────────────────────

A5_2_CASE_ID = "I4-REAL-G7-SUBFRAME-EAGER-20PCT-01"
A5_2_SAMPLE_FRACTION = 0.20
A5_2_SAMPLE_SEED = 42
A5_2_GALLERY_FUNCTION = "fig32_subframe_vertex"


def _a5_2_import_gallery():
    """Import the trusted time_series_draw gallery only when real data is used.

    Keeping this import delayed preserves the ratified split: the fast pytest
    harness remains ROOT/data independent, while the standalone harness owns
    the real-data execution.
    """
    import importlib
    return importlib.import_module("time_series_draw")


A5_2_ENV_AVAILABLE = "AVAILABLE"
A5_2_ENV_UNAVAILABLE = "UNAVAILABLE"
A5_2_ENV_CONTRACT_ERROR = "CONTRACT_ERROR"

# Missing these modules is a known external-environment condition for the
# trusted time-series gallery.  Everything else fails closed as gallery/code
# contract drift rather than being silently converted to a non-gating SKIP.
A5_2_EXTERNAL_MODULES = frozenset({
    "time_series_draw",
    "perfmonitor",
    "ROOT",
    "uproot",
})


def _a5_2_environment_status(root_path: str, gallery_module=None) -> tuple[str, str]:
    """Classify A5.2 availability without hiding trusted-gallery contract drift."""
    import os

    if not root_path:
        return A5_2_ENV_UNAVAILABLE, "no ROOT input path was supplied"
    if not os.path.isfile(root_path):
        return (A5_2_ENV_UNAVAILABLE,
                f"ROOT input is unavailable: {os.path.abspath(root_path)}")
    try:
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None)
        if missing in A5_2_EXTERNAL_MODULES:
            return (A5_2_ENV_UNAVAILABLE,
                    f"time_series_draw environment unavailable: "
                    f"{type(exc).__name__}: {exc}")
        return (A5_2_ENV_CONTRACT_ERROR,
                f"time_series_draw import contract failure: "
                f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        return (A5_2_ENV_CONTRACT_ERROR,
                f"time_series_draw import contract failure: "
                f"{type(exc).__name__}: {exc}")

    required = ("build_adf", A5_2_GALLERY_FUNCTION)
    missing = [name for name in required if not callable(getattr(gallery, name, None))]
    if missing:
        return (A5_2_ENV_CONTRACT_ERROR,
                f"time_series_draw missing required callable(s): {missing}")
    return A5_2_ENV_AVAILABLE, ""


def a5_2_environment(root_path: str, gallery_module=None) -> tuple[bool, str]:
    """Compatibility view of environment availability.

    ``False`` covers both unavailable environment and contract error.  The
    CaseSpec builder uses the richer classification so only genuine external
    unavailability becomes a non-gating SKIP.
    """
    status, reason = _a5_2_environment_status(
        root_path, gallery_module=gallery_module)
    return status == A5_2_ENV_AVAILABLE, reason


def a5_2_realdata_case(root_path: str, gallery_module=None) -> CaseSpec:
    """Build the bounded environment-gated A5.2 real-data CaseSpec.

    This first real-data increment deliberately proves execution/provenance,
    not independent calibration correctness.  A5.1 already owns an independent
    synthetic full-stack numerical oracle; later A5/A6 work owns real-data
    GB/reference correctness and lazy/full equivalence.
    """
    env_status, reason = _a5_2_environment_status(
        root_path, gallery_module=gallery_module)
    # Only genuine external/data unavailability makes the case inapplicable.
    # Gallery API/code drift remains applicable so the acceptance runner can
    # fail closed instead of silently reporting a SKIP.
    applicable = env_status != A5_2_ENV_UNAVAILABLE
    applicability_reason = reason if not applicable else ""
    return CaseSpec(
        case_id=A5_2_CASE_ID,
        claim_id="I4.real_g7_subframe.A5.2",
        title="real CalibVertex G7.32 executes on deterministic eager 20% data",
        claim=("the trusted time-series G7.32 CalibVertex subframe workflow executes "
               "on the deterministic EAGER 20% sample, returns finite plotted/profile "
               "y evidence from the public draw result, and records the actual "
               "original-row sample identity"),
        failure_means=("an applicable real-data G7.32 workflow silently skipped, "
                       "failed to create the CalibVertex subframe, returned no finite "
                       "plotted/profile y evidence, or the sampled-row provenance "
                       "could not be established"),
        expected_visual=("the existing G7.32 CalibVertex.vertex_x_intercept versus "
                         "time_s profile on the deterministic 20% sample"),
        owner_on_failure="ADF",
        purpose="COVERAGE",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            "gallery_function": A5_2_GALLERY_FUNCTION,
            "expr": "CalibVertex.vertex_x_intercept:time_s",
            "sample_fraction": A5_2_SAMPLE_FRACTION,
            "sample_seed": A5_2_SAMPLE_SEED,
        },
        applicable=applicable,
        applicability_reason=applicability_reason,
        setup_contract=("time_series_draw.build_adf(root_path, sample=0.20, lazy=False) "
                        "using its established random_state=42 sampling path; then the "
                        "existing fig32_subframe_vertex() gallery function executes "
                        "calibVertex(adf) and the public draw()"),
        preconditions=(
            "the ROOT input file is readable by the trusted time-series gallery environment",
            "the trusted build_adf and fig32_subframe_vertex callables are available",
            "sample fraction is exactly 0.20 and the established gallery seed is 42",
        ),
        surfaces_under_test=("draw",),
        non_claims=(
            "A5.2 is execution/provenance coverage, not an independent mathematical oracle for calibVertex",
            "the GB correction figures G7.33/G7.34 remain a later bounded A5 increment",
            "LAZY/FULL and BOTH/FULL real-data acceptance remain later A5/A6 work",
            "the optional gallery may still skip G7.32; only this acceptance harness converts an applicable skip to FAIL",
        ),
        negative_control="FAMILY_MUTATION:A5.2-OPTIONAL-G7-NONE-MUST-FAIL",
        reference_policy="named-immutable",
    )


def _a5_2_index_digest(index) -> str:
    """Stable digest of the actual pre-reset pandas sample row identities."""
    import hashlib
    import pandas as pd

    hashed = pd.util.hash_pandas_object(index, index=False).to_numpy(dtype=np.uint64)
    h = hashlib.sha256()
    h.update(str(getattr(index, "dtype", "unknown")).encode("utf-8"))
    h.update(str(len(index)).encode("ascii"))
    h.update(hashed.tobytes())
    return h.hexdigest()


def _a5_2_numeric_summary(stats: Any) -> dict:
    """Bounded numerical inspectability summary for one public stats payload."""
    numeric_scalars = 0
    numeric_values = 0
    finite_values = 0

    def visit(value):
        nonlocal numeric_scalars, numeric_values, finite_values
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
            return
        if hasattr(value, "columns") and hasattr(value, "__getitem__"):
            for name in value.columns:
                visit(value[name].to_numpy(copy=False))
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
            return
        try:
            arr = np.asarray(value)
        except Exception:
            return
        if not np.issubdtype(arr.dtype, np.number):
            return
        if arr.ndim == 0:
            numeric_scalars += 1
        numeric_values += int(arr.size)
        try:
            finite_values += int(np.isfinite(arr).sum())
        except TypeError:
            pass

    visit(stats)
    return {
        "numeric_scalars": numeric_scalars,
        "numeric_values": numeric_values,
        "finite_values": finite_values,
    }



def _a5_2_profile_numeric_evidence(stats: Any) -> dict:
    """Evidence from the plotted/profile y observable, not bookkeeping counts.

    Prefer the established ``profile_data.count`` + ``profile_data.y_mean``
    contract when the public payload exposes it.  The trusted G7.32 gallery
    currently does not request ``return_data=True``; in that envelope the
    established flat profile summary ``mean_y`` is the bounded equivalent
    plotted-y observable.

    If ``profile_data`` is present but malformed/non-finite, do not fall back
    to ``mean_y``: a broken richer profile payload must fail closed.
    """
    evidence = {
        "source": "",
        "profile_data_present": False,
        "populated_bins": 0,
        "finite_profile_y_values": 0,
    }
    if not isinstance(stats, dict):
        return evidence

    profile_data = stats.get("profile_data")
    if profile_data is not None:
        evidence["source"] = "profile_data.y_mean"
        evidence["profile_data_present"] = True
        if not (hasattr(profile_data, "columns")
                and "count" in profile_data.columns
                and "y_mean" in profile_data.columns):
            return evidence
        try:
            count = np.asarray(profile_data["count"])
            y_mean = np.asarray(profile_data["y_mean"])
            if count.shape != y_mean.shape:
                return evidence
            populated = np.asarray(count > 0, dtype=bool)
            evidence["populated_bins"] = int(populated.sum())
            if not np.issubdtype(y_mean.dtype, np.number):
                return evidence
            finite = populated & np.isfinite(y_mean)
            evidence["finite_profile_y_values"] = int(finite.sum())
            return evidence
        except Exception:
            return evidence

    evidence["source"] = "mean_y"
    mean_y = stats.get("mean_y", _MISSING)
    if mean_y is _MISSING:
        return evidence
    try:
        arr = np.asarray(mean_y)
        if arr.ndim != 0 or not np.issubdtype(arr.dtype, np.number):
            return evidence
        evidence["populated_bins"] = 1
        evidence["finite_profile_y_values"] = int(bool(np.isfinite(arr)))
    except Exception:
        pass
    return evidence


def run_a5_2_realdata(case: CaseSpec, root_path: str, *, gallery_module=None,
                       sample_fraction: float = A5_2_SAMPLE_FRACTION,
                       seed: int = A5_2_SAMPLE_SEED,
                       prepared_adf: Any = None,
                       prepared_provenance: dict | None = None) -> CaseResult:
    """Execute the bounded A5.2 real-data G7.32 acceptance contract.

    The trusted gallery remains unchanged and may keep G7 optional.  This
    runner is the acceptance boundary: once the environment is applicable,
    ``fig32_subframe_vertex() -> None`` is a FAIL, not a silent SKIP.

    The runner instruments the *actual* ``pandas.DataFrame.sample`` call made by
    ``time_series_draw.build_adf``.  It records the original row index selected
    before the gallery resets the sampled frame index, satisfying the v1.2
    requirement that FRACTION runs preserve sample identity rather than merely
    record ``fraction=0.20``.
    """
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    original_sample = None
    try:
        if case.case_id != A5_2_CASE_ID:
            res.status = INVALID_FIXTURE
            res.detail = f"A5.2 runner received unexpected case {case.case_id!r}"
            return res
        if (case.purpose != "COVERAGE" or case.gate != "ENVIRONMENT_GATED"
                or case.loading_mode != "EAGER" or case.sample_mode != "FRACTION"):
            res.status = INVALID_FIXTURE
            res.detail = "A5.2 runner requires COVERAGE/ENVIRONMENT_GATED/EAGER/FRACTION"
            return res
        if sample_fraction != A5_2_SAMPLE_FRACTION or seed != A5_2_SAMPLE_SEED:
            res.status = INVALID_FIXTURE
            res.detail = ("A5.2 canonical sample is fixed at fraction=0.20, seed=42; "
                          f"got fraction={sample_fraction!r}, seed={seed!r}")
            return res

        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        env_status, why = _a5_2_environment_status(
            root_path, gallery_module=gallery)
        if env_status != A5_2_ENV_AVAILABLE:
            # The case reached execution as applicable.  Any subsequent
            # unavailability or trusted-gallery contract drift is evidence
            # drift and must fail closed rather than become a SKIP.
            res.status = INVALID_FIXTURE
            res.detail = (
                f"A5.2 environment/contract changed after CaseSpec creation "
                f"({env_status}): {why}")
            return res

        import os
        sample_calls = []
        if prepared_adf is None:
            import pandas as pd
            original_sample = pd.DataFrame.sample

            def recording_sample(self, *args, **kwargs):
                out = original_sample(self, *args, **kwargs)
                frac = kwargs.get("frac")
                random_state = kwargs.get("random_state")
                if frac == A5_2_SAMPLE_FRACTION and random_state == A5_2_SAMPLE_SEED:
                    sample_calls.append({
                        "source_rows": int(len(self)),
                        "selected_rows": int(len(out)),
                        "index_digest_sha256": _a5_2_index_digest(out.index),
                        "index_dtype": str(out.index.dtype),
                    })
                return out

            pd.DataFrame.sample = recording_sample
            try:
                adf = gallery.build_adf(root_path, sample=sample_fraction, lazy=False)
            finally:
                pd.DataFrame.sample = original_sample
                original_sample = None
        else:
            adf = prepared_adf
            sample_calls.append(_a6_4_prepared_fraction_sample_evidence(
                adf, prepared_provenance, root_path))

        if getattr(adf, "_lazy_reader", None) is not None:
            res.status = INVALID_FIXTURE
            res.detail = "A5.2 EAGER build unexpectedly attached a lazy reader"
            return res
        if len(sample_calls) != 1:
            res.status = INVALID_FIXTURE
            res.detail = ("A5.2 could not identify exactly one canonical pandas "
                          f"sample call; observed {len(sample_calls)}")
            return res
        sample_evidence = sample_calls[0]
        if int(len(adf.df)) != sample_evidence["selected_rows"]:
            res.status = INVALID_FIXTURE
            res.detail = ("A5.2 sampled-row provenance disagrees with the final "
                          f"ADF row count: {sample_evidence['selected_rows']} vs {len(adf.df)}")
            return res
        if sample_evidence["selected_rows"] <= 0:
            res.status = INVALID_FIXTURE
            res.detail = "A5.2 deterministic sample is empty"
            return res

        gallery_fn = getattr(gallery, A5_2_GALLERY_FUNCTION)
        raw = gallery_fn(adf)
        if raw is None:
            res.status = FAIL
            res.detail = ("applicable A5.2 G7.32 returned None: the optional gallery "
                          "skip is not an acceptance PASS")
            return res
        payload = unwrap("draw", raw)
        if not isinstance(payload.stats, dict):
            res.status = FAIL
            res.detail = "A5.2 G7.32 returned a non-dict public stats payload"
            return res

        sf = adf.get_subframe("CalibVertex")
        if sf is None or "vertex_x_intercept" not in sf.df.columns:
            res.status = FAIL
            res.detail = "A5.2 G7.32 did not leave the expected CalibVertex subframe evidence"
            return res
        if "vertex_x_intercept" in adf.df.columns:
            res.status = FAIL
            res.detail = "A5.2 parent was contaminated with subframe-only vertex_x_intercept"
            return res

        n_value = payload.stats.get("n")
        try:
            n_numeric = int(n_value)
        except (TypeError, ValueError):
            res.status = FAIL
            res.detail = f"A5.2 G7.32 stats has no usable n count: {n_value!r}"
            return res
        if n_numeric <= 0:
            res.status = FAIL
            res.detail = f"A5.2 G7.32 produced no selected rows (n={n_numeric})"
            return res
        summary = _a5_2_numeric_summary(payload.stats)
        profile_evidence = _a5_2_profile_numeric_evidence(payload.stats)
        if (profile_evidence["populated_bins"] <= 0
                or profile_evidence["finite_profile_y_values"] <= 0):
            res.status = FAIL
            res.detail = (
                "A5.2 G7.32 has no finite populated plotted/profile y evidence: "
                f"{profile_evidence}; bookkeeping summary={summary}")
            return res

        st = os.stat(root_path)
        res.payload_paths = {"G7.32/draw": list(payload.path)}
        res.observed["realdata_provenance"] = {
            "input_path": os.path.abspath(root_path),
            "input_size_bytes": int(st.st_size),
            "input_mtime_ns": int(st.st_mtime_ns),
            "loading_mode": "EAGER",
            "sample_mode": "FRACTION",
            "sample_fraction": A5_2_SAMPLE_FRACTION,
            "sample_seed": A5_2_SAMPLE_SEED,
            "sampling_algorithm": ("pandas.DataFrame.sample(frac=0.20, "
                                   "random_state=42) observed at runtime"),
            **sample_evidence,
        }
        res.observed["g7_32_evidence"] = {
            "gallery_function": A5_2_GALLERY_FUNCTION,
            "calibvertex_subframe_registered": True,
            "parent_subframe_column_isolated": True,
            "public_n": n_numeric,
            "numeric_summary": summary,
            "profile_numeric_evidence": profile_evidence,
        }
        res.status = PASS
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"{type(exc).__name__}: {exc}"
        res.exception = traceback.format_exc(limit=6)
        return res
    finally:
        if original_sample is not None:
            try:
                import pandas as pd
                pd.DataFrame.sample = original_sample
            except Exception:
                pass
        _close()
        res.wall_time_s = round(time.time() - t0, 4)


def run_a5_2_realdata_gate(root_path: str, *, manifest_path: str,
                           gallery_module=None) -> tuple[CaseResult, dict, int]:
    """Run A5.2 once, write its manifest, and return the strict exit code."""
    case = a5_2_realdata_case(root_path, gallery_module=gallery_module)
    result = run_a5_2_realdata(case, root_path, gallery_module=gallery_module)
    extra = {}
    if isinstance(result.observed.get("realdata_provenance"), dict):
        extra.update(result.observed["realdata_provenance"])
    doc = write_manifest(manifest_path, [result], [case], extra=extra)
    return result, doc, strict_exit_code([result], [case])



# ─────────────────────────────────────────────────────────────────────────────
# A5.3 — environment-gated real-data G7.32 LAZY/FULL acceptance
# ─────────────────────────────────────────────────────────────────────────────

A5_3_CASE_ID = "I4-REAL-G7-SUBFRAME-LAZY-FULL-01"
A5_3_TREE_NAME = "treeTimeSeries"
A5_3_BLOCKER_CASE_ID = "I4-REAL-G7-SUBFRAME-LAZY-FULL-TIMEMS-REFUSAL-01"
A5_3_BLOCKER_BUG_ID = "BUG_time_series_draw_20260901_lazy_build_adf_timeMS_unloaded"


def a5_3_realdata_case(root_path: str, gallery_module=None) -> CaseSpec:
    """Build the bounded real-data LAZY/FULL G7.32 CaseSpec.

    A5.2 established the canonical EAGER/FRACTION real-data bridge.  A5.3
    changes one execution dimension: it reuses the same trusted G7.32 workflow
    on the supported unsampled lazy loader.  It is COVERAGE, not an
    eager-vs-lazy numerical equivalence proof; the BOTH/FULL comparison remains
    a later bounded increment.
    """
    env_status, reason = _a5_2_environment_status(
        root_path, gallery_module=gallery_module)
    applicable = env_status != A5_2_ENV_UNAVAILABLE
    applicability_reason = reason if not applicable else ""
    return CaseSpec(
        case_id=A5_3_CASE_ID,
        claim_id="I4.real_g7_subframe_lazy.A5.3",
        title="real CalibVertex G7.32 executes on unsampled lazy full data",
        claim=("the trusted time-series G7.32 CalibVertex subframe workflow "
               "executes through build_adf(lazy=True, sample=None), remains "
               "genuinely lazy, performs no pandas sampling, expands its "
               "on-demand physical branch set during G7.32, and returns finite "
               "plotted/profile y evidence"),
        failure_means=("an applicable lazy/full G7.32 workflow is eager in "
                       "disguise, samples the input, performs no on-demand "
                       "branch expansion, silently skips, loses CalibVertex "
                       "isolation, or returns no finite plotted/profile y evidence"),
        expected_visual=("the existing G7.32 CalibVertex.vertex_x_intercept "
                         "versus time_s profile on the full lazy input"),
        owner_on_failure="ADF",
        purpose="COVERAGE",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CONSISTENCY",
        loading_mode="LAZY",
        sample_mode="FULL",
        canonical_spec={
            "gallery_function": A5_2_GALLERY_FUNCTION,
            "expr": "CalibVertex.vertex_x_intercept:time_s",
            "sample": None,
            "lazy": True,
            "tree_name": A5_3_TREE_NAME,
        },
        applicable=applicable,
        applicability_reason=applicability_reason,
        setup_contract=("time_series_draw.build_adf(root_path, sample=None, "
                        "lazy=True) using the established read_tree_lazy path; "
                        "then the existing fig32_subframe_vertex() executes "
                        "calibVertex(adf) and public draw()"),
        preconditions=(
            "the ROOT input file is readable by the trusted time-series gallery environment",
            "the trusted build_adf and fig32_subframe_vertex callables are available",
            "sample is None because sampled-lazy is explicitly unsupported",
            "the lazy build exposes a live _lazy_reader with loaded_branches evidence",
        ),
        surfaces_under_test=("draw",),
        non_claims=(
            "A5.3 is lazy/full execution coverage, not independent calibVertex correctness",
            "A5.3 does not compare EAGER/FULL and LAZY/FULL numerical outputs",
            "A5.3 records lazy branch expansion but does not claim an exact minimal branch set",
            "G7.33/G7.34 GB correction coverage remains a later bounded A5 increment",
        ),
        negative_control="FAMILY_MUTATION:A5.3-EAGER-IN-DISGUISE-OR-NO-LAZY-EXPANSION",
        reference_policy="named-immutable",
    )


def _a5_3_loaded_branches(adf) -> tuple[str, ...] | None:
    """Return canonical lazy-reader branch evidence, or None if unavailable."""
    reader = getattr(adf, "_lazy_reader", None)
    if reader is None:
        return None
    loaded = getattr(reader, "loaded_branches", None)
    if loaded is None:
        return None
    try:
        return tuple(sorted(str(x) for x in loaded))
    except TypeError:
        return None




def a5_3_blocker_resolution_record() -> dict:
    """Record the governed A5.3 blocker -> positive-coverage transition.

    The registered refusal is retired only together with the owning caller fix.
    The real FULL+LAZY A6 run supplies the demonstrating execution evidence.
    """
    return {
        "bug_id": A5_3_BLOCKER_BUG_ID,
        "previous_case_id": A5_3_BLOCKER_CASE_ID,
        "previous_state": {
            "purpose": "ERROR_CONTRACT",
            "known_bug_status": "KNOWN_BUG",
            "failure": "KeyError('timeMS') during lazy build_adf/addTimeQuantiles",
            "recorded_owner_on_failure": "ADF",
            "corrected_owner": "time_series/gallery caller",
        },
        "new_case_id": A5_3_CASE_ID,
        "new_state": {
            "purpose": "COVERAGE",
            "known_bug_status": "SUPPORTED",
        },
        "fix_owner": "examples/time_series/time_series.py::addTimeQuantiles",
        "fix_contract": (
            "ensure the requested physical time branch through AliasDataFrame's "
            "lazy-load API before direct pandas-frame access"),
        "demonstrating_case_id": A5_3_CASE_ID,
        "closure_gate": "real public --full --lazy --manifest --pdf --strict run",
    }


def lazy_full_deferral_reconciliation() -> list[dict]:
    """Re-adjudicate A5.4/A5.5/A5.6 deferrals after the timeMS blocker closes."""
    return [
        {
            "case_id": A5_4_CASE_ID,
            "previous_reason": "LAZY/FULL deferred while the A5.3 timeMS blocker was open",
            "disposition": "RE_ADJUDICATED",
            "current_contract": (
                "G7.33 LAZY/FULL execution is covered by the A6 full-lazy gallery gate; "
                "its dedicated machine acceptance remains the deterministic EAGER/FRACTION "
                "case, avoiding duplicate full-data oracle execution."),
        },
        {
            "case_id": A5_5_CASE_ID,
            "previous_reason": "LAZY/FULL deferred while the A5.3 timeMS blocker was open",
            "disposition": "RE_ADJUDICATED",
            "current_contract": (
                "G7.34 LAZY/FULL execution is covered by the A6 full-lazy gallery gate; "
                "the exact prepared-state reuse oracle remains EAGER/FRACTION and targeted "
                "BOTH-mode tests cover loading-mode semantics without duplicating the full-data run."),
        },
        {
            "case_id": A5_6_CASE_ID,
            "previous_reason": "LAZY/FULL deferred while the A5.3 timeMS blocker was open",
            "disposition": "RE_ADJUDICATED",
            "current_contract": (
                "G7.34 logical-state invariance remains an EAGER/FRACTION machine oracle; "
                "the A6 full-lazy gallery proves real LAZY/FULL execution while targeted "
                "EAGER/LAZY invariance tests cover loading-mode equivalence."),
        },
    ]


def run_a5_3_realdata(case: CaseSpec, root_path: str, *, gallery_module=None,
                       prepared_adf: Any = None,
                       prepared_provenance: dict | None = None) -> CaseResult:
    """Execute the bounded A5.3 LAZY/FULL real-data G7.32 contract.

    A6.4 may provide one already-built lazy ADF so the public full-data gate
    reads the ROOT input only once.
    """
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    original_sample = None
    try:
        if case.case_id != A5_3_CASE_ID:
            res.status = INVALID_FIXTURE
            res.detail = f"A5.3 runner received unexpected case {case.case_id!r}"
            return res
        if (case.purpose != "COVERAGE" or case.gate != "ENVIRONMENT_GATED"
                or case.loading_mode != "LAZY" or case.sample_mode != "FULL"):
            res.status = INVALID_FIXTURE
            res.detail = "A5.3 runner requires COVERAGE/ENVIRONMENT_GATED/LAZY/FULL"
            return res

        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        env_status, why = _a5_2_environment_status(
            root_path, gallery_module=gallery)
        if env_status != A5_2_ENV_AVAILABLE:
            res.status = INVALID_FIXTURE
            res.detail = (
                f"A5.3 environment/contract changed after CaseSpec creation "
                f"({env_status}): {why}")
            return res

        import os
        import pandas as pd

        sample_calls = []
        if prepared_adf is None:
            original_sample = pd.DataFrame.sample

            def forbidden_sample(self, *args, **kwargs):
                sample_calls.append({"args": list(args), "kwargs": dict(kwargs)})
                return original_sample(self, *args, **kwargs)

            pd.DataFrame.sample = forbidden_sample
            try:
                adf = gallery.build_adf(
                    root_path, sample=None, lazy=True,
                    tree_name=case.canonical_spec["tree_name"])
            finally:
                pd.DataFrame.sample = original_sample
                original_sample = None

            if sample_calls:
                res.status = INVALID_FIXTURE
                res.detail = (
                    "A5.3 LAZY/FULL build unexpectedly called pandas.DataFrame.sample; "
                    f"observed {len(sample_calls)} call(s)")
                return res
        else:
            adf = prepared_adf
            if not isinstance(prepared_provenance, dict):
                res.status = INVALID_FIXTURE
                res.detail = "A5.3 prepared LAZY/FULL ADF is missing provenance"
                return res
            if (prepared_provenance.get("loading_mode") != "LAZY"
                    or prepared_provenance.get("sample_mode") != "FULL"):
                res.status = INVALID_FIXTURE
                res.detail = "A5.3 prepared ADF has wrong loading/sample mode"
                return res

        loaded_before = _a5_3_loaded_branches(adf)
        if loaded_before is None:
            res.status = INVALID_FIXTURE
            res.detail = "A5.3 lazy build has no usable _lazy_reader.loaded_branches evidence"
            return res

        gallery_fn = getattr(gallery, A5_2_GALLERY_FUNCTION)
        raw = gallery_fn(adf)
        if raw is None:
            res.status = FAIL
            res.detail = (
                "applicable A5.3 G7.32 returned None: the optional gallery skip "
                "is not a lazy/full acceptance PASS")
            return res
        payload = unwrap("draw", raw)
        if not isinstance(payload.stats, dict):
            res.status = FAIL
            res.detail = "A5.3 G7.32 returned a non-dict public stats payload"
            return res

        loaded_after = _a5_3_loaded_branches(adf)
        if loaded_after is None:
            res.status = FAIL
            res.detail = "A5.3 lost lazy-reader branch evidence during G7.32"
            return res
        before_set = set(loaded_before)
        after_set = set(loaded_after)
        if not before_set.issubset(after_set):
            res.status = FAIL
            res.detail = (
                "A5.3 lazy loaded-branch state regressed during G7.32: "
                f"before={loaded_before}, after={loaded_after}")
            return res
        newly_loaded = tuple(sorted(after_set - before_set))
        if not newly_loaded:
            res.status = FAIL
            res.detail = (
                "A5.3 G7.32 produced no on-demand physical branch expansion; "
                "lazy/full acceptance would be eager-in-disguise or non-causal")
            return res

        sf = adf.get_subframe("CalibVertex")
        if sf is None or "vertex_x_intercept" not in sf.df.columns:
            res.status = FAIL
            res.detail = "A5.3 G7.32 did not leave the expected CalibVertex subframe evidence"
            return res
        if "vertex_x_intercept" in adf.df.columns:
            res.status = FAIL
            res.detail = "A5.3 parent was contaminated with subframe-only vertex_x_intercept"
            return res

        n_value = payload.stats.get("n")
        try:
            n_numeric = int(n_value)
        except (TypeError, ValueError):
            res.status = FAIL
            res.detail = f"A5.3 G7.32 stats has no usable n count: {n_value!r}"
            return res
        if n_numeric <= 0:
            res.status = FAIL
            res.detail = f"A5.3 G7.32 produced no rows (n={n_numeric})"
            return res

        summary = _a5_2_numeric_summary(payload.stats)
        profile_evidence = _a5_2_profile_numeric_evidence(payload.stats)
        if (profile_evidence["populated_bins"] <= 0
                or profile_evidence["finite_profile_y_values"] <= 0):
            res.status = FAIL
            res.detail = (
                "A5.3 G7.32 has no finite populated plotted/profile y evidence: "
                f"{profile_evidence}; bookkeeping summary={summary}")
            return res

        st = os.stat(root_path)
        res.payload_paths = {"G7.32/draw": list(payload.path)}
        provenance = {
            "input_path": os.path.abspath(root_path),
            "input_size_bytes": int(st.st_size),
            "input_mtime_ns": int(st.st_mtime_ns),
            "loading_mode": "LAZY",
            "sample_mode": "FULL",
            "sample_fraction": None,
            "sample_seed": None,
            "tree_name": case.canonical_spec["tree_name"],
            "source_rows": int(len(adf.df)),
            "lazy_loaded_before": list(loaded_before),
            "lazy_loaded_after": list(loaded_after),
            "lazy_newly_loaded": list(newly_loaded),
        }
        if isinstance(prepared_provenance, dict):
            provenance.update(dict(prepared_provenance))
            provenance["lazy_loaded_before"] = list(loaded_before)
            provenance["lazy_loaded_after"] = list(loaded_after)
            provenance["lazy_newly_loaded"] = list(newly_loaded)
        res.observed["realdata_provenance"] = provenance
        res.observed["g7_32_evidence"] = {
            "gallery_function": A5_2_GALLERY_FUNCTION,
            "calibvertex_subframe_registered": True,
            "parent_subframe_column_isolated": True,
            "public_n": n_numeric,
            "numeric_summary": summary,
            "profile_numeric_evidence": profile_evidence,
        }
        res.status = PASS
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"{type(exc).__name__}: {exc}"
        res.exception = traceback.format_exc(limit=6)
        return res
    finally:
        if original_sample is not None:
            try:
                import pandas as pd
                pd.DataFrame.sample = original_sample
            except Exception:
                pass
        _close()
        res.wall_time_s = round(time.time() - t0, 4)


def run_a5_3_realdata_gate(root_path: str, *, manifest_path: str,
                           gallery_module=None) -> tuple[CaseResult, dict, int]:
    """Run the positive real-data A5.3 LAZY/FULL regression gate."""
    case = a5_3_realdata_case(root_path, gallery_module=gallery_module)
    result = run_a5_3_realdata(
        case, root_path, gallery_module=gallery_module)
    extra = {
        "a5_3_blocker_transition": a5_3_blocker_resolution_record(),
    }
    if isinstance(result.observed.get("realdata_provenance"), dict):
        extra.update(result.observed["realdata_provenance"])
    doc = write_manifest(manifest_path, [result], [case], extra=extra)
    return result, doc, strict_exit_code([result], [case])

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
# A6.1 activates reference_policy: it is now read, validated, serialized and
# has executable semantics.  Keep the future-stage registry explicit so the
# declared-state audit remains fail-closed for any later deferred field.
FUTURE_STAGE_FIELDS = {}


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


def reference_policy_semantics(case: CaseSpec) -> dict:
    """Return the executable A6 semantics for one CaseSpec reference policy.

    Unknown policies fail loudly rather than being serialized as decorative
    metadata.  ``named-immutable`` participates in cross-run named-reference
    comparison; ``same-process`` is explicitly not eligible for that path.
    """
    try:
        return dict(REFERENCE_POLICY_SEMANTICS[case.reference_policy])
    except KeyError as exc:
        raise HarnessError(
            f"{case.case_id}: unknown reference_policy {case.reference_policy!r}; "
            f"known: {REFERENCE_POLICY}") from exc


def named_reference_case_ids(cases: Sequence[CaseSpec]) -> tuple[str, ...]:
    """Return exactly the cases eligible for cross-run named references.

    This is the first executable use of the per-case A6 policy: same-process
    cases remain recorded in the run manifest but are not silently promoted to
    cross-run reference comparisons.
    """
    out = []
    for case in cases:
        semantics = reference_policy_semantics(case)
        if semantics["cross_run_reference"]:
            out.append(case.case_id)
    return tuple(out)


def reference_identity_required_keys(sample_mode: str) -> tuple[str, ...]:
    """Return the one authoritative required-field set for A6 identity comparison.

    This deliberately reuses A5 provenance fields.  Callers must not maintain
    their own required-field lists; derivation, comparison and manifest
    preparation all pass through this function.
    """
    if sample_mode == "FRACTION":
        return _REFERENCE_IDENTITY_FRACTION_REQUIRED_KEYS
    if sample_mode == "FULL":
        return _REFERENCE_IDENTITY_FULL_REQUIRED_KEYS
    raise HarnessError(
        f"named-reference identity has invalid sample_mode {sample_mode!r}; "
        f"expected one of {SAMPLE_MODE}")


def reference_identity_missing_fields(identity: dict) -> tuple[str, ...]:
    """Return missing/blank fields that make an identity comparison-ineligible."""
    if not isinstance(identity, dict):
        raise HarnessError("reference identity must be a dict")
    mode = identity.get("sample_mode")
    required = reference_identity_required_keys(mode)
    missing = []
    for key in required:
        if key not in identity:
            missing.append(key)
            continue
        value = identity[key]
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(key)
    return tuple(missing)


def _require_complete_reference_identity(identity: dict, *, role: str) -> None:
    """Fail closed unless one explicit identity is comparison-ready."""
    missing = reference_identity_missing_fields(identity)
    if missing:
        raise HarnessError(
            f"{role} named-reference identity is incomplete; missing "
            + ", ".join(missing))


def reference_identity_from_provenance(provenance_doc: dict, *,
                                       require_complete: bool = False) -> dict:
    """Derive the A6 named-reference identity from existing run provenance.

    A5 already owns deterministic sampling and records the original sampled
    index digest.  A6 reuses those fields verbatim.  Completeness is checked by
    the same authoritative rule used at comparison and manifest boundaries.
    """
    if not isinstance(provenance_doc, dict):
        raise HarnessError("reference provenance must be a dict")
    keys = list(_REFERENCE_IDENTITY_BASE_KEYS)
    if provenance_doc.get("sample_mode") == "FRACTION":
        keys.extend(_REFERENCE_IDENTITY_FRACTION_KEYS)
    identity = {k: provenance_doc[k] for k in keys if k in provenance_doc}
    if require_complete:
        _require_complete_reference_identity(identity, role="derived")
    return identity


def compare_reference_identity(current: dict, accepted: dict) -> None:
    """Fail closed unless two complete explicit A6 identities are identical."""
    if not isinstance(current, dict) or not isinstance(accepted, dict):
        raise HarnessError("reference identity comparison requires two dicts")
    _require_complete_reference_identity(current, role="current")
    _require_complete_reference_identity(accepted, role="accepted")
    if current != accepted:
        keys = sorted(set(current) | set(accepted))
        mismatched = [k for k in keys if current.get(k) != accepted.get(k)]
        raise HarnessError(
            "named-reference identity mismatch: " + ", ".join(mismatched))


ACCEPTED_REFERENCE_SCHEMA = "AliasDataFrame.PHASE_13_77.AcceptedReference"
ACCEPTED_REFERENCE_SCHEMA_VERSION = 1
_ACCEPTED_REFERENCE_REQUIRED_FIELDS = (
    "reference_manifest_id",
    "dataset_input_identity",
    "gallery_case_registry_version",
    "sampling_algorithm_version",
    "sample_seed",
    "accepted_code_baseline",
    "schema_oracle_version",
    "approval_identity",
    "approval_date",
    "supersedes",
    "reason_for_update",
)


def accepted_reference_required_fields() -> tuple[str, ...]:
    """Return the authoritative A6 accepted-reference field set."""
    return _ACCEPTED_REFERENCE_REQUIRED_FIELDS


def _json_ready_reference_value(value: Any) -> Any:
    """Convert a reference payload to stable JSON data without stringifying arrays."""
    if isinstance(value, dict):
        return {str(k): _json_ready_reference_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready_reference_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready_reference_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _canonical_json_sha256(payload: dict) -> str:
    data = json.dumps(
        _json_ready_reference_value(payload), sort_keys=True,
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _accepted_reference_id(payload_without_id: dict) -> str:
    return "sha256:" + _canonical_json_sha256(payload_without_id)


def _manifest_reference_identity(run_manifest: dict) -> dict:
    if not isinstance(run_manifest, dict):
        raise HarnessError("run manifest must be a dict")
    status = run_manifest.get("reference_identity_status")
    identity = run_manifest.get("reference_identity")
    if not isinstance(status, dict) or status.get("comparison_ready") is not True:
        missing = status.get("missing_fields", []) if isinstance(status, dict) else []
        suffix = f"; missing {', '.join(missing)}" if missing else ""
        raise HarnessError("run manifest has no comparison-ready reference identity" + suffix)
    if not isinstance(identity, dict):
        raise HarnessError("run manifest comparison-ready identity is missing")
    _require_complete_reference_identity(identity, role="run-manifest")
    return dict(identity)


def _manifest_acceptance_ready(run_manifest: dict) -> None:
    """Fail closed unless a run manifest is eligible for explicit acceptance."""
    if not isinstance(run_manifest, dict):
        raise HarnessError("run manifest must be a dict")
    reconciliation = run_manifest.get("reconciliation")
    if not isinstance(reconciliation, dict):
        raise HarnessError("run manifest has no reconciliation record")
    if reconciliation.get("exit_code") != 0:
        raise HarnessError(
            f"run manifest is not strict-clean; reconciliation exit_code="
            f"{reconciliation.get('exit_code')!r}")
    named = run_manifest.get("named_reference_case_ids")
    if not isinstance(named, list) or not named:
        raise HarnessError("run manifest declares no named-reference cases")
    _manifest_reference_identity(run_manifest)


def _manifest_case_records(run_manifest: dict, *, role: str) -> dict[str, dict]:
    """Return one fail-closed case_id -> record map for a run manifest.

    Acceptance and comparison must share the same duplicate/malformed-record
    semantics.  A dict comprehension is forbidden here because duplicate IDs
    would silently collapse.
    """
    if not isinstance(run_manifest, dict):
        raise HarnessError(f"{role} run manifest must be a dict")
    records = run_manifest.get("cases")
    if not isinstance(records, list):
        raise HarnessError(f"{role} run manifest cases must be a list")
    by_id: dict[str, dict] = {}
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise HarnessError(f"{role} run manifest case record #{i} is not a dict")
        cid = rec.get("case_id")
        if not isinstance(cid, str) or not cid:
            raise HarnessError(f"{role} run manifest case record #{i} has no case_id")
        if cid in by_id:
            raise HarnessError(f"{role} run manifest has duplicate case record {cid!r}")
        by_id[cid] = rec
    return by_id


def _validate_serialized_observable_contract(contract: dict, *, path: str) -> None:
    """Validate one manifest-side observable contract before comparison.

    Malformed evidence is a HarnessError, never an incidental KeyError.
    """
    if not isinstance(contract, dict):
        raise HarnessError(f"{path}: observable contract must be a dict")
    name = contract.get("name")
    if not isinstance(name, str) or not name:
        raise HarnessError(f"{path}: observable contract has no name")
    comparator = contract.get("comparator")
    if comparator not in COMPARATORS:
        raise HarnessError(
            f"{path}/{name}: invalid or missing comparator {comparator!r}; "
            f"known: {COMPARATORS}")
    if comparator == "close":
        for key in ("atol", "rtol"):
            value = contract.get(key)
            if not isinstance(value, (int, float, np.number)):
                raise HarnessError(f"{path}/{name}: {key} must be numeric")


def _reference_case_snapshot(run_manifest: dict) -> list[dict]:
    """Extract the immutable machine-comparison payload for named-reference cases."""
    _manifest_acceptance_ready(run_manifest)
    named = list(run_manifest["named_reference_case_ids"])
    by_id = _manifest_case_records(run_manifest, role="acceptance")
    missing = [cid for cid in named if cid not in by_id]
    if missing:
        raise HarnessError("run manifest is missing named-reference case(s): " + ", ".join(missing))
    out = []
    for cid in named:
        rec = by_id[cid]
        out.append(_json_ready_reference_value({
            "case_id": cid,
            "status": rec.get("status"),
            "oracle_kind": rec.get("oracle_kind"),
            "known_bug_status": rec.get("known_bug_status"),
            "known_bug_id": rec.get("known_bug_id"),
            "declared_observables": rec.get("declared_observables", []),
            "observed": rec.get("observed", {}),
        }))
    return out


def _require_nonblank(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessError(f"accepted reference requires non-empty {field_name}")
    return value.strip()


def accepted_reference_from_manifest(
        run_manifest: dict, *, accepted_code_baseline: str,
        approval_identity: str, approval_date: str,
        reason_for_update: str, supersedes: str | None = None) -> dict:
    """Build one immutable accepted-reference record from a strict-clean run."""
    _manifest_acceptance_ready(run_manifest)
    accepted_code_baseline = _require_nonblank(
        accepted_code_baseline, field_name="accepted_code_baseline")
    approval_identity = _require_nonblank(
        approval_identity, field_name="approval_identity")
    approval_date = _require_nonblank(approval_date, field_name="approval_date")
    reason_for_update = _require_nonblank(
        reason_for_update, field_name="reason_for_update")
    if supersedes is not None:
        supersedes = _require_nonblank(supersedes, field_name="supersedes")

    identity = _manifest_reference_identity(run_manifest)
    provenance = run_manifest.get("provenance", {})
    schema_version = provenance.get("schema_version") or SCHEMA_VERSION
    named_ids = list(run_manifest["named_reference_case_ids"])
    case_registry_digest = _canonical_json_sha256({"case_ids": named_ids})
    payload = {
        "schema": ACCEPTED_REFERENCE_SCHEMA,
        "schema_version": ACCEPTED_REFERENCE_SCHEMA_VERSION,
        "status": "ACCEPTED",
        "dataset_input_identity": _json_ready_reference_value(identity),
        "gallery_case_registry_version": schema_version,
        "case_registry_digest_sha256": case_registry_digest,
        "sampling_algorithm_version": identity.get("sampling_algorithm", "FULL"),
        "sample_seed": identity.get("sample_seed"),
        "accepted_code_baseline": accepted_code_baseline,
        "schema_oracle_version": schema_version,
        "approval_identity": approval_identity,
        "approval_date": approval_date,
        "supersedes": supersedes,
        "reason_for_update": reason_for_update,
        "named_reference_case_ids": named_ids,
        "reference_cases": _reference_case_snapshot(run_manifest),
        "source_run_manifest_sha256": _canonical_json_sha256(run_manifest),
    }
    payload["reference_manifest_id"] = _accepted_reference_id(payload)
    validate_accepted_reference(payload)
    return payload


def validate_accepted_reference(reference: dict) -> None:
    """Validate schema, integrity digest and comparison-critical fields."""
    if not isinstance(reference, dict):
        raise HarnessError("accepted reference must be a dict")
    if reference.get("schema") != ACCEPTED_REFERENCE_SCHEMA:
        raise HarnessError("accepted reference has unknown schema")
    if reference.get("schema_version") != ACCEPTED_REFERENCE_SCHEMA_VERSION:
        raise HarnessError("accepted reference has unsupported schema_version")
    missing = [k for k in accepted_reference_required_fields() if k not in reference]
    if missing:
        raise HarnessError("accepted reference is missing field(s): " + ", ".join(missing))
    _require_complete_reference_identity(
        reference["dataset_input_identity"], role="accepted-reference")
    for key in ("accepted_code_baseline", "approval_identity", "approval_date",
                "reason_for_update", "gallery_case_registry_version",
                "schema_oracle_version"):
        _require_nonblank(reference.get(key), field_name=key)
    if not isinstance(reference.get("named_reference_case_ids"), list) \
            or not reference["named_reference_case_ids"]:
        raise HarnessError("accepted reference has no named_reference_case_ids")
    if not isinstance(reference.get("reference_cases"), list):
        raise HarnessError("accepted reference has no reference_cases")
    case_ids = [r.get("case_id") for r in reference["reference_cases"]
                if isinstance(r, dict)]
    if case_ids != reference["named_reference_case_ids"]:
        raise HarnessError("accepted reference case snapshot does not match named case registry")
    identity = reference["dataset_input_identity"]
    expected_sampling = identity.get("sampling_algorithm", "FULL")
    if reference.get("sampling_algorithm_version") != expected_sampling:
        raise HarnessError(
            "accepted reference sampling_algorithm_version disagrees with dataset_input_identity")
    expected_seed = identity.get("sample_seed")
    if reference.get("sample_seed") != expected_seed:
        raise HarnessError(
            "accepted reference sample_seed disagrees with dataset_input_identity")
    claimed = reference.get("reference_manifest_id")
    without_id = dict(reference)
    without_id.pop("reference_manifest_id", None)
    expected = _accepted_reference_id(without_id)
    if claimed != expected:
        raise HarnessError("accepted reference integrity digest mismatch")


def load_accepted_reference(path: str) -> dict:
    """Load one explicitly named accepted reference; no 'latest' discovery exists."""
    _require_nonblank(path, field_name="reference path")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except FileNotFoundError as exc:
        raise HarnessError(f"accepted reference does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise HarnessError(f"accepted reference is invalid JSON: {path}") from exc
    validate_accepted_reference(doc)
    return doc


def _write_accepted_reference_exclusive(path: str, reference: dict) -> None:
    """Create a new immutable reference path; existing paths are never overwritten."""
    validate_accepted_reference(reference)
    _require_nonblank(path, field_name="reference path")
    parent = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(parent):
        raise HarnessError(f"reference parent directory does not exist: {parent}")
    text = json.dumps(reference, indent=1, sort_keys=False, ensure_ascii=False) + "\n"
    try:
        with open(path, "x", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
    except FileExistsError as exc:
        raise HarnessError(
            f"accepted reference path already exists and is immutable: {path}") from exc
    except Exception:
        # A failed first-time write may leave a partial NEW file.  It is not an
        # accepted reference and must not survive as misleading evidence.
        try:
            if os.path.exists(path):
                os.unlink(path)
        except Exception:
            pass
        raise


def accept_named_reference(
        path: str, run_manifest: dict, *, accepted_code_baseline: str,
        approval_identity: str, approval_date: str,
        reason_for_update: str = "initial acceptance") -> dict:
    """Explicitly create a first immutable named reference at a new path."""
    reference = accepted_reference_from_manifest(
        run_manifest,
        accepted_code_baseline=accepted_code_baseline,
        approval_identity=approval_identity,
        approval_date=approval_date,
        reason_for_update=reason_for_update,
        supersedes=None,
    )
    _write_accepted_reference_exclusive(path, reference)
    return reference


def update_named_reference(
        new_path: str, previous_path: str, run_manifest: dict, *,
        accepted_code_baseline: str, approval_identity: str,
        approval_date: str, reason_for_update: str) -> dict:
    """Write an explicit successor reference; the previous reference stays untouched."""
    previous = load_accepted_reference(previous_path)
    current_identity = _manifest_reference_identity(run_manifest)
    compare_reference_identity(
        current_identity, previous["dataset_input_identity"])
    reference = accepted_reference_from_manifest(
        run_manifest,
        accepted_code_baseline=accepted_code_baseline,
        approval_identity=approval_identity,
        approval_date=approval_date,
        reason_for_update=reason_for_update,
        supersedes=previous["reference_manifest_id"],
    )
    _write_accepted_reference_exclusive(new_path, reference)
    return reference


def _compare_reference_value(reference: Any, candidate: Any, contract: dict,
                             *, path: str) -> None:
    """Recursively compare one JSON-like observable using its declared tolerance."""
    if isinstance(reference, dict) or isinstance(candidate, dict):
        if not isinstance(reference, dict) or not isinstance(candidate, dict):
            raise HarnessError(f"{path}: observed structure type mismatch")
        if set(reference) != set(candidate):
            raise HarnessError(f"{path}: observed mapping keys differ")
        for key in sorted(reference):
            _compare_reference_value(
                reference[key], candidate[key], contract,
                path=f"{path}.{key}")
        return
    if isinstance(reference, (list, tuple)) or isinstance(candidate, (list, tuple)):
        result = compare_array(
            reference, candidate,
            comparator=contract["comparator"],
            atol=contract.get("atol", 0.0), rtol=contract.get("rtol", 0.0))
        if not result.ok:
            raise HarnessError(f"{path}: {result.detail}")
        return
    result = compare_scalar(
        reference, candidate, comparator=contract["comparator"],
        atol=contract.get("atol", 0.0), rtol=contract.get("rtol", 0.0))
    if not result.ok:
        raise HarnessError(f"{path}: {result.detail}")


def compare_run_manifest_to_reference(run_manifest: dict,
                                      accepted_reference: dict) -> dict:
    """Compare one strict-clean run against one explicit immutable reference."""
    _manifest_acceptance_ready(run_manifest)
    validate_accepted_reference(accepted_reference)
    compare_reference_identity(
        _manifest_reference_identity(run_manifest),
        accepted_reference["dataset_input_identity"])
    current_ids = list(run_manifest["named_reference_case_ids"])
    if current_ids != accepted_reference["named_reference_case_ids"]:
        raise HarnessError("named-reference case registry mismatch")
    current_schema = run_manifest.get("provenance", {}).get("schema_version") or SCHEMA_VERSION
    if current_schema != accepted_reference["schema_oracle_version"]:
        raise HarnessError(
            "schema/oracle version mismatch: "
            f"{current_schema!r} != {accepted_reference['schema_oracle_version']!r}")

    current_cases = _manifest_case_records(run_manifest, role="comparison")
    evidence = []
    for stored in accepted_reference["reference_cases"]:
        cid = stored["case_id"]
        current = current_cases.get(cid)
        if current is None:
            raise HarnessError(f"current run has no case record {cid!r}")
        if current.get("status") != stored.get("status"):
            raise HarnessError(
                f"{cid}: status mismatch {current.get('status')!r} != {stored.get('status')!r}")
        current_contracts = current.get("declared_observables", [])
        if current_contracts != stored.get("declared_observables", []):
            raise HarnessError(f"{cid}: declared observable contract changed")
        if not isinstance(current_contracts, list):
            raise HarnessError(f"{cid}: declared observable contract must be a list")
        by_name = {}
        for i, contract in enumerate(current_contracts):
            _validate_serialized_observable_contract(
                contract, path=f"{cid}/declared_observables[{i}]")
            name = contract["name"]
            if name in by_name:
                raise HarnessError(f"{cid}: duplicate observable contract {name!r}")
            by_name[name] = contract
        current_observed = current.get("observed", {})
        stored_observed = stored.get("observed", {})
        for name, contract in by_name.items():
            if contract.get("status") != "EXECUTED":
                continue
            if name not in current_observed or name not in stored_observed:
                raise HarnessError(f"{cid}/{name}: executed observable missing from manifest")
            _compare_reference_value(
                stored_observed[name], current_observed[name], contract,
                path=f"{cid}/{name}")
        evidence.append({"case_id": cid, "status": "MATCH"})
    return {
        "reference_manifest_id": accepted_reference["reference_manifest_id"],
        "case_count": len(evidence),
        "cases": evidence,
        "ok": True,
    }


def compare_manifest_to_named_reference(run_manifest: dict, path: str) -> dict:
    """Load one explicit path and compare; this function never discovers 'latest'."""
    return compare_run_manifest_to_reference(
        run_manifest, load_accepted_reference(path))


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
        if c.reference_policy not in REFERENCE_POLICY:
            bad.append(f"{cid}: reference_policy {c.reference_policy!r} not in "
                       f"{REFERENCE_POLICY}")
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
                if (c.purpose == "INVARIANCE"
                        and runner not in ("run_slot_symmetry", "run_subframe_slot_symmetry")):
                    bad.append(f"{cid}: A4 positive slot execution contract is not bound to run_slot_symmetry or run_subframe_slot_symmetry")
                elif c.purpose == "ERROR_CONTRACT" and runner != "run_error_contract":
                    bad.append(f"{cid}: A4 ERROR_CONTRACT slot execution contract is not bound to run_error_contract")
                elif runner not in ("run_slot_symmetry", "run_subframe_slot_symmetry", "run_error_contract"):
                    bad.append(f"{cid}: A4 slot execution contract has unknown runner {runner!r}")
                if tuple(c.slots_under_test) != tuple(slot_contract["slots_under_test"]):
                    bad.append(f"{cid}: slots_under_test drift from A4 execution contract")
                if tuple(c.anti_contamination_preconditions) != tuple(
                        slot_contract["anti_contamination_preconditions"]):
                    bad.append(f"{cid}: anti_contamination_preconditions drift from A4 execution contract")
                actual_slots = a4_actual_target_slots(c, slot_contract)
                if actual_slots != tuple(sorted(c.slots_under_test)):
                    bad.append(f"{cid}: slot-exclusivity drift; declared {tuple(c.slots_under_test)!r}, actual {actual_slots!r}")
                if c.loading_mode != "BOTH":
                    bad.append(f"{cid}: A4 slot contracts require loading_mode='BOTH'")
                if c.sample_mode != "FULL":
                    bad.append(f"{cid}: A4 BOTH slot contract requires sample_mode='FULL'")
                if runner in ("run_slot_symmetry", "run_subframe_slot_symmetry") and c.purpose != "INVARIANCE":
                    bad.append(f"{cid}: positive A4 runner requires purpose='INVARIANCE'")
                if runner == "run_error_contract" and c.purpose != "ERROR_CONTRACT":
                    bad.append(f"{cid}: run_error_contract slot contract requires purpose='ERROR_CONTRACT'")
                for detail in _a4_contract_reconciliation_errors(c, slot_contract):
                    msg = f"{cid}: {detail}"
                    if msg not in bad:
                        bad.append(msg)
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
                       and slot_contract.get("runner") in ("run_slot_symmetry", "run_subframe_slot_symmetry"))
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


def public_query_text(case: "CaseSpec | None") -> str:
    """Compact faithful public request for page-level human/AI auditing."""
    if case is None:
        return ""
    spec = dict(case.canonical_spec or {})
    explicit = spec.get("public_query")
    if explicit:
        return str(explicit)
    expr = spec.get("expr")
    if not expr:
        return str(spec.get("gallery_function", ""))
    args = [repr(expr)]
    for key in ("type", "selection", "selection_vector", "weights_vector",
                "vector_compose", "normalize", "facet_by", "group_by",
                "group_by_bins", "group_by_quantiles", "fit", "summary_fit",
                "bins", "range"):
        if key in spec and spec[key] is not None:
            args.append(f"{key}={spec[key]!r}")
    return "adf.draw(" + ", ".join(args) + ")"


def footer_text(case: "CaseSpec") -> str:
    """Structured page-review contract shared by PDF and manifest.

    PRIMARY ORACLE pages must be interpretable by a human or multimodal AI
    reviewer without guessing what "validate" means.
    """
    fc = case.figure_contract
    expected = case.expected_visual
    if fc is not None and fc.primary_comparison:
        expected = (f"{expected}\n  primary comparison: {fc.primary_comparison}"
                    f"\n  accepted envelope:  {fc.accepted_envelope}")
    injected_truth = str(case.canonical_spec.get("injected_truth", "")).strip()
    injected_block = (
        f"INJECTED TRUTH:\n  {injected_truth}\n\n"
        if injected_truth else ""
    )
    proof_class = proof_class_for_case(case)
    query = public_query_text(case)
    query_block = f"QUERY:\n  {query}\n\n" if query else ""
    checks = reviewer_checks_for_case(case)
    checks_block = ""
    if checks:
        checks_block = (
            "REVIEWER MUST CHECK (HUMAN / AI):\n"
            + "\n".join(f"  {i}. {text}" for i, text in enumerate(checks, 1))
            + "\n\n"
        )
    return (f"PROOF CLASS: {proof_class}\n"
            f"CASE: {case.case_id}\n\n"
            f"{query_block}"
            f"{injected_block}"
            f"EXPECTED:\n  {expected}\n\n"
            f"{checks_block}"
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
    run_provenance = {**provenance(), **(extra or {})}
    doc = {
        "provenance": run_provenance,
        "reconciliation": rec,
        "named_reference_case_ids": list(named_reference_case_ids(cases)),
        "cases": [],
    }
    reference_identity = reference_identity_from_provenance(run_provenance)
    if reference_identity:
        missing_reference_fields = reference_identity_missing_fields(reference_identity)
        if missing_reference_fields:
            # A6.1-v02: a partial identity may remain visible as ordinary run
            # provenance, but it must never be serialized under the
            # comparison-ready ``reference_identity`` key.
            doc["reference_identity_status"] = {
                "comparison_ready": False,
                "missing_fields": list(missing_reference_fields),
            }
        else:
            doc["reference_identity"] = reference_identity
            doc["reference_identity_status"] = {
                "comparison_ready": True,
                "missing_fields": [],
            }
    # A3.11-v02: once a manifest is in A3 context, persist the closure record
    # even when it is BLOCKED.  Absence of a required family is itself durable
    # evidence and must not make the governance record disappear.
    a3_known = set(A3_CLOSURE_CONTRACTS)
    if a3_known.intersection(by_id):
        doc["a3_closure"] = a3_closure_reconciliation(cases)
    a4_known = set(A4_POSITIVE_CASE_IDS) | set(A4_ERROR_CASE_IDS) | set(A4_SLOT_CONTRACTS)
    if a4_known.intersection(by_id):
        doc["a4_closure"] = a4_closure_reconciliation(cases)
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
                # A6.1: reference_policy is now executable, not future-staged.
                "reference_policy": c.reference_policy,
                "reference_policy_semantics": reference_policy_semantics(c),
                # Future-staged fields remain visible if later stages add any.
                "future_staged": {
                    name: {"value": getattr(c, name), "owning_stage": stage}
                    for name, stage in FUTURE_STAGE_FIELDS.items()},
                "proof_class": proof_class_for_case(c),
                "reviewer_checks": list(reviewer_checks_for_case(c)),
                "figure_contract": (
                    {
                        **asdict(c.figure_contract),
                        **reviewer_contract_for_case(c),
                    }
                    if c.figure_contract is not None else None
                ),
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
        actual_slots = a4_actual_target_slots(case, contract)
        if actual_slots != tuple(sorted(case.slots_under_test)):
            res.status = INVALID_FIXTURE
            res.detail = ("A4 slot-exclusivity drift: "
                          f"declared {tuple(case.slots_under_test)!r}, actual {actual_slots!r}")
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

        eager_registered_aliases = set(getattr(eager, "aliases", {}) or {})
        eager_materialized_before = eager_registered_aliases & set(eager.df.columns)
        expected_eager_new = set(contract.get("expected_eager_new_aliases", (alias_name,)))
        if expected_eager_new != {alias_name}:
            res.status = INVALID_FIXTURE
            res.detail = "A4 contract expected_eager_new_aliases must equal the target alias"
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
        eager_materialized_after = eager_registered_aliases & set(eager.df.columns)
        eager_newly_materialized = eager_materialized_after - eager_materialized_before
        if eager_newly_materialized != expected_eager_new:
            res.status = FAIL
            res.detail = ("EAGER alias materialization set mismatch: "
                          f"expected new {sorted(expected_eager_new)}, got {sorted(eager_newly_materialized)}")
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
            "eager_registered_aliases": sorted(eager_registered_aliases),
            "eager_newly_materialized_aliases": sorted(eager_newly_materialized),
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


def run_subframe_slot_symmetry(case: CaseSpec,
                               make_eager: Callable[[], Any],
                               make_lazy: Callable[[], Any]) -> CaseResult:
    """Execute the supported scalar subframe-qualified A4 causality case.

    The parent join key is an explicit structural baseline.  The qualified
    expression must still resolve the registered child and, in the lazy-main
    arm, load exactly the remaining parent dependency without the decoy.
    """
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        contract = A4_SLOT_CONTRACTS.get(case.case_id)
        if contract is None or contract.get("runner") != "run_subframe_slot_symmetry":
            res.status = INVALID_FIXTURE
            res.detail = "no run_subframe_slot_symmetry A4 execution contract"
            return res
        if tuple(case.slots_under_test) != tuple(contract["slots_under_test"]):
            res.status = INVALID_FIXTURE
            res.detail = "slots_under_test does not match subframe A4 execution contract"
            return res
        if tuple(case.anti_contamination_preconditions) != tuple(contract["anti_contamination_preconditions"]):
            res.status = INVALID_FIXTURE
            res.detail = "anti_contamination_preconditions do not match subframe A4 contract"
            return res
        actual_slots = a4_actual_target_slots(case, contract)
        if actual_slots != tuple(sorted(case.slots_under_test)):
            res.status = INVALID_FIXTURE
            res.detail = ("A4 subframe slot-exclusivity drift: "
                          f"declared {tuple(case.slots_under_test)!r}, actual {actual_slots!r}")
            return res
        if case.loading_mode != "BOTH" or case.sample_mode != "FULL" or tuple(case.surfaces_under_test) != ("draw",):
            res.status = INVALID_FIXTURE
            res.detail = "A4 subframe runner requires BOTH/FULL with draw held fixed"
            return res

        eager = make_eager()
        lazy = make_lazy()
        if getattr(eager, "_lazy_reader", None) is not None:
            res.status = INVALID_FIXTURE
            res.detail = "EAGER subframe arm factory returned a lazy parent"
            return res
        lazy_reader = getattr(lazy, "_lazy_reader", None)
        if lazy_reader is None:
            res.status = INVALID_FIXTURE
            res.detail = "LAZY subframe arm factory did not attach a lazy reader"
            return res
        qualified = contract["qualified_reference"]
        sf_name, sf_col = qualified.split(".", 1)
        for label, adf in (("EAGER", eager), ("LAZY", lazy)):
            if sf_col in adf.df.columns or f"{sf_name}_{sf_col}" in adf.df.columns:
                res.status = INVALID_FIXTURE
                res.detail = f"{label} parent was contaminated with subframe value before draw"
                return res
            sf = adf.get_subframe(sf_name)
            if sf is None or sf_col not in sf.df.columns:
                res.status = INVALID_FIXTURE
                res.detail = f"{label} registered subframe lacks {qualified}"
                return res

        lazy_before = set(lazy_reader.loaded_branches)
        expected_before = set(contract["expected_lazy_loaded_before"])
        expected_after = set(contract["expected_lazy_loaded_after"])
        unrelated = set(contract["unrelated_physical_branches"])
        if lazy_before != expected_before:
            res.status = INVALID_FIXTURE
            res.detail = ("subframe structural baseline mismatch: "
                          f"expected {sorted(expected_before)}, got {sorted(lazy_before)}")
            return res
        if unrelated & lazy_before:
            res.status = INVALID_FIXTURE
            res.detail = "subframe lazy baseline already contains unrelated branch"
            return res

        def call_one(adf):
            kw = dict(case.canonical_spec)
            expr = kw.pop("expr")
            return unwrap("draw", adf.draw(expr, lazy=True, keep_materialized=True, **kw))

        eager_payload = call_one(eager)
        lazy_payload = call_one(lazy)
        lazy_after = set(lazy_reader.loaded_branches)
        if lazy_after != expected_after:
            res.status = FAIL
            res.detail = ("LAZY subframe parent load set mismatch: "
                          f"expected {sorted(expected_after)}, got {sorted(lazy_after)}")
            return res
        if unrelated & lazy_after:
            res.status = FAIL
            res.detail = "LAZY subframe arm loaded unrelated physical branch"
            return res

        res.payload_paths = {"EAGER/draw": list(eager_payload.path),
                             "LAZY/draw": list(lazy_payload.path)}
        res.observed["slot_evidence"] = {
            "slot": case.slots_under_test[0],
            "qualified_reference": qualified,
            "structural_baseline_physical_dependencies": sorted(expected_before),
            "lazy_loaded_before": sorted(lazy_before),
            "lazy_loaded_after": sorted(lazy_after),
            "unrelated_physical_branches": sorted(unrelated),
            "subframe_reference_resolved": True,
        }
        for o in case.observables:
            if o.status != "EXECUTED":
                res.observed[o.name] = {"status": o.status}
                continue
            try:
                _assert_source_matches("run_subframe_slot_symmetry", o)
                eager_v = resolve(eager_payload.stats, o.path, o.access)
                lazy_v = resolve(lazy_payload.stats, o.path, o.access)
            except HarnessError as exc:
                res.status = INVALID_FIXTURE
                res.detail = str(exc)
                return res
            res.observed[o.name] = {"EAGER": eager_v, "LAZY": lazy_v}
            res.observable_contract.append(_contract(o))
            result = compare_observable(o, eager_v, lazy_v)
            res.comparisons.append(comparison_evidence(o, result, reference_label="EAGER", candidate_label="LAZY"))
            res.executed_comparisons += 1
            if not result.ok:
                res.status = FAIL
                res.detail = f"{o.name}: EAGER vs LAZY: {result.detail}"
                return res
        if res.executed_comparisons == 0:
            res.status = INVALID_FIXTURE
            res.detail = "A4 subframe case executed no numerical comparison"
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


def run_a5_full_stack(case: CaseSpec,
                      make_eager: Callable[[], Any],
                      make_lazy: Callable[[], Any],
                      independent_anchor: Callable[[], dict]) -> CaseResult:
    """Execute the first A5 keyed-subframe + group_by full-stack contract.

    The independent anchor is computed before either product fixture is
    constructed, so it cannot read state mutated/materialized by ADF.  Both
    EAGER and LAZY public ``draw`` results are compared independently against
    that anchor and directly against each other.

    The lazy parent uses the same explicit structural join-key baseline as the
    banked A4 scalar-subframe case, then must load exactly ``x`` and ``group``
    for the composed request.  ``decoy`` must remain unloaded.
    """
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        contract = A5_1_CONTRACT if case.case_id == A5_1_CASE_ID else None
        if contract is None or contract.get("runner") != "run_a5_full_stack":
            res.status = INVALID_FIXTURE
            res.detail = f"no A5.1 full-stack execution contract for {case.case_id}"
            return res
        if (case.purpose != "CORRECTNESS"
                or case.oracle_kind != "CORRECTNESS"
                or case.loading_mode != "BOTH"
                or case.sample_mode != "FULL"):
            res.status = INVALID_FIXTURE
            res.detail = "A5.1 runner requires CORRECTNESS/CORRECTNESS/BOTH/FULL"
            return res
        if tuple(case.surfaces_under_test) != (contract["surface"],):
            res.status = INVALID_FIXTURE
            res.detail = "A5.1 runner requires the contract-declared single public surface"
            return res
        if case.canonical_spec.get("group_by") != contract["group_by"]:
            res.status = INVALID_FIXTURE
            res.detail = "A5.1 group_by declaration drift"
            return res
        if not _a4_text_contains_target(
                case.canonical_spec.get("expr"), contract["qualified_reference"]):
            res.status = INVALID_FIXTURE
            res.detail = "A5.1 qualified subframe reference drift"
            return res

        # Freeze the independent truth BEFORE product construction/execution.
        expected = independent_anchor()
        if not isinstance(expected, dict):
            res.status = INVALID_FIXTURE
            res.detail = "A5.1 independent anchor must return a dict"
            return res

        eager = make_eager()
        lazy = make_lazy()
        if getattr(eager, "_lazy_reader", None) is not None:
            res.status = INVALID_FIXTURE
            res.detail = "A5.1 EAGER factory returned a lazy parent"
            return res
        lazy_reader = getattr(lazy, "_lazy_reader", None)
        if lazy_reader is None:
            res.status = INVALID_FIXTURE
            res.detail = "A5.1 LAZY factory did not attach a lazy reader"
            return res

        qualified = contract["qualified_reference"]
        sf_name, sf_col = qualified.split(".", 1)
        for label, adf in (("EAGER", eager), ("LAZY", lazy)):
            if sf_col in adf.df.columns or f"{sf_name}_{sf_col}" in adf.df.columns:
                res.status = INVALID_FIXTURE
                res.detail = f"{label} parent was contaminated with {qualified} before draw"
                return res
            sf = adf.get_subframe(sf_name)
            if sf is None or sf_col not in sf.df.columns:
                res.status = INVALID_FIXTURE
                res.detail = f"{label} registered subframe lacks {qualified}"
                return res

        lazy_before = set(lazy_reader.loaded_branches)
        expected_before = set(contract["expected_lazy_loaded_before"])
        expected_after = set(contract["expected_lazy_loaded_after"])
        unrelated = set(contract["unrelated_physical_branches"])
        if lazy_before != expected_before:
            res.status = INVALID_FIXTURE
            res.detail = ("A5.1 structural baseline mismatch: "
                          f"expected {sorted(expected_before)}, got {sorted(lazy_before)}")
            return res
        if unrelated & lazy_before:
            res.status = INVALID_FIXTURE
            res.detail = "A5.1 LAZY baseline already contains an unrelated branch"
            return res

        def call_one(adf):
            kw = dict(case.canonical_spec)
            expr = kw.pop("expr")
            raw = adf.draw(expr, lazy=True, keep_materialized=True, **kw)
            return unwrap(contract["surface"], raw)

        eager_payload = call_one(eager)
        lazy_payload = call_one(lazy)
        lazy_after = set(lazy_reader.loaded_branches)
        if lazy_after != expected_after:
            res.status = FAIL
            res.detail = ("A5.1 LAZY full-stack load set mismatch: "
                          f"expected {sorted(expected_after)}, got {sorted(lazy_after)}")
            return res
        if unrelated & lazy_after:
            res.status = FAIL
            res.detail = "A5.1 LAZY full-stack request loaded unrelated physical branch"
            return res

        res.payload_paths = {
            "EAGER/draw": list(eager_payload.path),
            "LAZY/draw": list(lazy_payload.path),
        }
        res.observed["full_stack_evidence"] = {
            "qualified_reference": qualified,
            "group_by": contract["group_by"],
            "structural_baseline_physical_dependencies": sorted(expected_before),
            "lazy_loaded_before": sorted(lazy_before),
            "lazy_loaded_after": sorted(lazy_after),
            "unrelated_physical_branches": sorted(unrelated),
            "independent_anchor_computed_before_product": True,
        }

        for o in case.observables:
            if o.status != "EXECUTED":
                res.observed[o.name] = {"status": o.status}
                continue
            try:
                _assert_source_matches("run_a5_full_stack", o)
            except HarnessError as exc:
                res.status = INVALID_FIXTURE
                res.detail = str(exc)
                return res
            if o.name not in expected:
                res.status = INVALID_FIXTURE
                res.detail = f"A5.1 independent anchor supplied no value for {o.name!r}"
                return res
            try:
                eager_v = resolve(eager_payload.stats, o.path, o.access)
                lazy_v = resolve(lazy_payload.stats, o.path, o.access)
            except HarnessError as exc:
                res.status = INVALID_FIXTURE
                res.detail = f"declared A5.1 observable {o.name!r} not extractable: {exc}"
                return res

            independent_v = expected[o.name]
            res.observed[o.name] = {
                "independent": independent_v,
                "EAGER": eager_v,
                "LAZY": lazy_v,
            }
            res.observable_contract.append(_contract(o))
            comparisons = (
                ("independent", "EAGER", independent_v, eager_v),
                ("independent", "LAZY", independent_v, lazy_v),
                ("EAGER", "LAZY", eager_v, lazy_v),
            )
            for reference_label, candidate_label, reference_v, candidate_v in comparisons:
                result = compare_observable(o, reference_v, candidate_v)
                res.comparisons.append(comparison_evidence(
                    o, result, reference_label=reference_label,
                    candidate_label=candidate_label))
                res.executed_comparisons += 1
                if not result.ok:
                    res.status = FAIL
                    res.detail = (
                        f"{o.name}: {reference_label} vs {candidate_label}: "
                        f"{result.detail}")
                    return res

        if res.executed_comparisons == 0:
            res.status = INVALID_FIXTURE
            res.detail = "A5.1 full-stack case executed no numerical comparison"
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

# ─────────────────────────────────────────────────────────────────────────────
# A5.4 — environment-gated real-data GB / G7.33 EAGER 20% acceptance
# ─────────────────────────────────────────────────────────────────────────────

A5_4_CASE_ID = "I4-REAL-G7-GB-EAGER-20PCT-01"
A5_4_GALLERY_FUNCTION = "fig33_gb_correction_tgl"
A5_4_SUBFRAME = "CalibBias1"
A5_4_PREDICTED = "dcar_tpc_vertex_predicted0"


def _a5_4_environment_status(root_path: str, gallery_module=None) -> tuple[str, str]:
    """A5.4 environment status without depending on the G7.32 callable."""
    import os

    if not root_path:
        return A5_2_ENV_UNAVAILABLE, "no ROOT input path was supplied"
    if not os.path.isfile(root_path):
        return (A5_2_ENV_UNAVAILABLE,
                f"ROOT input is unavailable: {os.path.abspath(root_path)}")
    try:
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None)
        if missing in A5_2_EXTERNAL_MODULES:
            return (A5_2_ENV_UNAVAILABLE,
                    f"time_series_draw environment unavailable: "
                    f"{type(exc).__name__}: {exc}")
        return (A5_2_ENV_CONTRACT_ERROR,
                f"time_series_draw import contract failure: "
                f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        return (A5_2_ENV_CONTRACT_ERROR,
                f"time_series_draw import contract failure: "
                f"{type(exc).__name__}: {exc}")

    required = ("build_adf", A5_4_GALLERY_FUNCTION)
    missing = [name for name in required if not callable(getattr(gallery, name, None))]
    if missing:
        return (A5_2_ENV_CONTRACT_ERROR,
                f"time_series_draw missing required callable(s): {missing}")
    return A5_2_ENV_AVAILABLE, ""


def a5_4_realdata_case(root_path: str, gallery_module=None) -> CaseSpec:
    """Bounded real-data G7.33 GB correction coverage on the canonical 20% sample."""
    env_status, reason = _a5_4_environment_status(
        root_path, gallery_module=gallery_module)
    applicable = env_status != A5_2_ENV_UNAVAILABLE
    applicability_reason = reason if not applicable else ""
    return CaseSpec(
        case_id=A5_4_CASE_ID,
        claim_id="I4.real_g7_gb.A5.4",
        title="real G7.33 GB correction executes on deterministic eager 20% data",
        claim=("the trusted time-series G7.33 calibBiasResolution workflow executes "
               "on the canonical deterministic EAGER 20% sample, registers CalibBias1, "
               "materializes finite dcar_tpc_vertex_predicted0 values, and returns "
               "a non-empty public draw result"),
        failure_means=("an applicable real-data G7.33 workflow silently skipped, "
                       "failed to register its coefficient subframe, failed to produce "
                       "finite predicted correction values, or returned no numerical "
                       "public result"),
        expected_visual=("the existing G7.33 normalized raw-versus-predicted "
                         "DCA_r differential profile versus tgl"),
        owner_on_failure="GB",
        purpose="COVERAGE",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            "gallery_function": A5_4_GALLERY_FUNCTION,
            "expr": "[dcar_tpc_vertex, dcar_tpc_vertex_predicted0]:tgl",
            "selection": "(ncl>60)&(abs(dcar_tpc_vertex)<10)",
            "type": "profile",
            "bins": 50,
            "normalize": "delta",
            "sample_fraction": A5_2_SAMPLE_FRACTION,
            "sample_seed": A5_2_SAMPLE_SEED,
            "expected_subframe": A5_4_SUBFRAME,
            "expected_predicted_column": A5_4_PREDICTED,
        },
        applicable=applicable,
        applicability_reason=applicability_reason,
        setup_contract=("reuse time_series_draw.build_adf(root_path, sample=0.20, "
                        "lazy=False), observe the actual pandas sample call, then run "
                        "the unchanged fig33_gb_correction_tgl() workflow"),
        preconditions=(
            "the ROOT input file is readable by the trusted time-series environment",
            "trusted build_adf and fig33_gb_correction_tgl callables are available",
            "sample fraction is exactly 0.20 with the established random_state=42",
        ),
        surfaces_under_test=("draw",),
        non_claims=(
            "A5.4 is execution/GB-composition coverage, not independent calibBiasResolution correctness",
            "G7.34 sector reuse is a later bounded increment",
            "real LAZY/FULL G7.33 execution is re-adjudicated at A6 through the full-lazy gallery gate; this A5.4 machine case remains the deterministic EAGER/FRACTION oracle",
            "the 10% internal calibBiasResolution fit subsample is trusted workflow behavior, not the A5.4 input-sampling contract",
        ),
        negative_control="FAMILY_MUTATION:A5.4-GB-PREDICTED-NONFINITE-MUST-FAIL",
        reference_policy="named-immutable",
    )


def _a5_4_public_numeric_evidence(stats: Any) -> dict:
    """Finite plotted/normalized evidence across the actual G7.33 stats shape.

    G7.33 uses a vector expression and does not request ``return_data=True``.
    Depending on the public draw path, the returned stats payload may therefore
    be a dict *or* a sequence of per-expression/profile stats dictionaries.

    Accept only semantically meaningful plotted quantities:
      * ``normalize_data.value`` when available;
      * populated ``profile_data.y_mean``;
      * finite scalar ``mean_y`` as the established profile-summary fallback.

    Never count arbitrary numeric bookkeeping such as ``n``/``count`` alone.
    """

    records = []

    def one(value: Any, path: str) -> None:
        if isinstance(value, (list, tuple)):
            for i, child in enumerate(value):
                one(child, f"{path}[{i}]")
            return
        if not isinstance(value, dict):
            return

        table = value.get("normalize_data")
        if table is not None and hasattr(table, "columns") and "value" in table.columns:
            vals = np.asarray(table["value"])
            mask = np.ones(vals.shape, dtype=bool)
            if "signal_count" in table.columns and "reference_count" in table.columns:
                sc = np.asarray(table["signal_count"])
                rc = np.asarray(table["reference_count"])
                if sc.shape == vals.shape and rc.shape == vals.shape:
                    mask = (sc > 0) & (rc > 0)
            finite = 0
            if np.issubdtype(vals.dtype, np.number):
                finite = int((mask & np.isfinite(vals)).sum())
            records.append({
                "source": f"{path}.normalize_data.value",
                "populated_bins": int(mask.sum()),
                "finite_values": finite,
            })
            return

        profile = value.get("profile_data")
        if profile is not None:
            finite = 0
            populated_n = 0
            if (hasattr(profile, "columns")
                    and "count" in profile.columns
                    and "y_mean" in profile.columns):
                count = np.asarray(profile["count"])
                y_mean = np.asarray(profile["y_mean"])
                if count.shape == y_mean.shape:
                    populated = np.asarray(count > 0, dtype=bool)
                    populated_n = int(populated.sum())
                    if np.issubdtype(y_mean.dtype, np.number):
                        finite = int((populated & np.isfinite(y_mean)).sum())
            records.append({
                "source": f"{path}.profile_data.y_mean",
                "populated_bins": populated_n,
                "finite_values": finite,
            })
            # Rich profile_data is authoritative.  Do not fall back to mean_y
            # if it is present but malformed/non-finite.
            return

        if "mean_y" in value:
            try:
                arr = np.asarray(value["mean_y"])
                finite = int(
                    arr.ndim == 0
                    and np.issubdtype(arr.dtype, np.number)
                    and bool(np.isfinite(arr)))
            except Exception:
                finite = 0
            records.append({
                "source": f"{path}.mean_y",
                "populated_bins": 1,
                "finite_values": finite,
            })

    one(stats, "stats")
    finite_total = int(sum(r["finite_values"] for r in records))
    populated_total = int(sum(r["populated_bins"] for r in records))
    return {
        "source": "recursive_public_profile_evidence",
        "records": records,
        "populated_bins": populated_total,
        "finite_values": finite_total,
    }

def run_a5_4_realdata(case: CaseSpec, root_path: str, *, gallery_module=None,
                       sample_fraction: float = A5_2_SAMPLE_FRACTION,
                       seed: int = A5_2_SAMPLE_SEED,
                       prepared_adf: Any = None,
                       prepared_provenance: dict | None = None) -> CaseResult:
    """Execute the bounded EAGER/FRACTION real G7.33 GB acceptance case."""
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip

    import os
    import pandas as pd

    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    original_sample = None
    try:
        if case.case_id != A5_4_CASE_ID:
            res.status = INVALID_FIXTURE
            res.detail = f"A5.4 runner received unexpected case {case.case_id!r}"
            return res
        if (case.loading_mode != "EAGER" or case.sample_mode != "FRACTION"
                or sample_fraction != A5_2_SAMPLE_FRACTION
                or seed != A5_2_SAMPLE_SEED):
            res.status = INVALID_FIXTURE
            res.detail = "A5.4 requires exact EAGER fraction=0.20, seed=42"
            return res

        env_status, why = _a5_4_environment_status(
            root_path, gallery_module=gallery_module)
        if env_status == A5_2_ENV_UNAVAILABLE:
            res.status = INVALID_FIXTURE
            res.detail = f"A5.4 environment changed after CaseSpec creation: {why}"
            return res
        if env_status == A5_2_ENV_CONTRACT_ERROR:
            res.status = INVALID_FIXTURE
            res.detail = why
            return res

        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()

        sample_observed = []
        if prepared_adf is None:
            original_sample = pd.DataFrame.sample

            def observed_sample(self, *args, **kwargs):
                frac = kwargs.get("frac")
                random_state = kwargs.get("random_state")
                out = original_sample(self, *args, **kwargs)
                if frac == A5_2_SAMPLE_FRACTION and random_state == A5_2_SAMPLE_SEED:
                    sample_observed.append({
                        "source_rows": int(len(self)),
                        "selected_rows": int(len(out)),
                        "index_digest_sha256": _a5_2_index_digest(out.index),
                        "index_dtype": str(out.index.dtype),
                    })
                return out

            pd.DataFrame.sample = observed_sample
            try:
                adf = gallery.build_adf(
                    root_path, sample=A5_2_SAMPLE_FRACTION, lazy=False)
            finally:
                pd.DataFrame.sample = original_sample
                original_sample = None
        else:
            adf = prepared_adf
            sample_observed.append(_a6_4_prepared_fraction_sample_evidence(
                adf, prepared_provenance, root_path))

        if len(sample_observed) != 1:
            res.status = INVALID_FIXTURE
            res.detail = (
                "A5.4 expected exactly one canonical build_adf sample call; "
                f"observed {len(sample_observed)}")
            return res

        try:
            raw = getattr(gallery, A5_4_GALLERY_FUNCTION)(adf)
        except Exception as exc:
            res.status = FAIL
            res.detail = f"A5.4 G7.33 raised {type(exc).__name__}: {exc}"
            res.exception = traceback.format_exc(limit=8)
            return res
        if raw is None:
            res.status = FAIL
            res.detail = "A5.4 optional gallery skip is not an acceptance PASS"
            return res

        payload = unwrap("draw", raw)

        subframe = (adf.get_subframe(A5_4_SUBFRAME)
                    if hasattr(adf, "get_subframe") else None)
        if subframe is None or not hasattr(subframe, "df") or len(subframe.df) == 0:
            res.status = FAIL
            res.detail = "A5.4 G7.33 did not register a non-empty CalibBias1 subframe"
            return res

        if not hasattr(adf, "df") or A5_4_PREDICTED not in adf.df.columns:
            res.status = FAIL
            res.detail = (
                "A5.4 G7.33 did not materialize "
                "dcar_tpc_vertex_predicted0 in the parent frame")
            return res
        predicted = np.asarray(adf.df[A5_4_PREDICTED])
        if not np.issubdtype(predicted.dtype, np.number):
            res.status = FAIL
            res.detail = "A5.4 predicted correction column is not numeric"
            return res
        finite_pred = int(np.isfinite(predicted).sum())
        if finite_pred <= 0:
            res.status = FAIL
            res.detail = "A5.4 predicted correction column has no finite values"
            return res

        public_evidence = _a5_4_public_numeric_evidence(payload.stats)
        if public_evidence["finite_values"] <= 0:
            res.status = FAIL
            res.detail = "A5.4 G7.33 public result has no finite numerical evidence"
            return res

        sample = sample_observed[0]
        res.observed["realdata_provenance"] = {
            "input_path": os.path.abspath(root_path),
            "input_size_bytes": int(os.path.getsize(root_path)),
            "input_mtime_ns": int(os.stat(root_path).st_mtime_ns),
            "loading_mode": "EAGER",
            "sample_mode": "FRACTION",
            "sample_fraction": A5_2_SAMPLE_FRACTION,
            "sample_seed": A5_2_SAMPLE_SEED,
            "sampling_algorithm": (
                "pandas.DataFrame.sample(frac=0.20, random_state=42) "
                "observed at runtime"),
            **sample,
        }
        res.observed["g7_33_evidence"] = {
            "gallery_function": A5_4_GALLERY_FUNCTION,
            "calibbias1_subframe_registered": True,
            "calibbias1_rows": int(len(subframe.df)),
            "predicted_column": A5_4_PREDICTED,
            "predicted_values": int(predicted.size),
            "finite_predicted_values": finite_pred,
            "public_numeric_evidence": public_evidence,
            "numeric_summary": _a5_2_numeric_summary(payload.stats),
        }
        res.status = PASS
        res.detail = ""
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"{type(exc).__name__}: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        if original_sample is not None:
            try:
                import pandas as pd
                pd.DataFrame.sample = original_sample
            except Exception:
                pass
        _close()
        res.wall_time_s = round(time.time() - t0, 4)


def run_a5_4_realdata_gate(root_path: str, *, manifest_path: str,
                           gallery_module=None) -> tuple[CaseResult, dict, int]:
    case = a5_4_realdata_case(root_path, gallery_module=gallery_module)
    result = run_a5_4_realdata(
        case, root_path, gallery_module=gallery_module)
    extra = {}
    if isinstance(result.observed.get("realdata_provenance"), dict):
        extra.update(result.observed["realdata_provenance"])
    doc = write_manifest(manifest_path, [result], [case], extra=extra)
    return result, doc, strict_exit_code([result], [case])

# ─────────────────────────────────────────────────────────────────────────────
# A5.5 — real-data G7.34 sector reuse after G7.33 preparation
# ─────────────────────────────────────────────────────────────────────────────

A5_5_CASE_ID = "I4-REAL-G7-GB-SECTOR-REUSE-EAGER-20PCT-01"
A5_5_PREP_FUNCTION = "fig33_gb_correction_tgl"
A5_5_REUSE_FUNCTION = "fig34_gb_correction_sector"
A5_5_LOADER_BUG_ID = "ADF-API-TREENAME-1"


def _a5_5_environment_status(root_path: str, gallery_module=None) -> tuple[str, str]:
    status, reason = _a5_4_environment_status(
        root_path, gallery_module=gallery_module)
    if status != A5_2_ENV_AVAILABLE:
        return status, reason
    gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    if not callable(getattr(gallery, A5_5_REUSE_FUNCTION, None)):
        return (A5_2_ENV_CONTRACT_ERROR,
                f"time_series_draw missing required callable {A5_5_REUSE_FUNCTION!r}")
    return A5_2_ENV_AVAILABLE, ""


def _a5_5_array_digest(values: Any) -> str:
    import hashlib
    arr = np.ascontiguousarray(np.asarray(values))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(repr(tuple(arr.shape)).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _a5_5_frame_digest(frame: Any) -> str:
    import hashlib
    import pandas as pd
    if frame is None or not hasattr(frame, "columns"):
        raise HarnessError("A5.5 frame digest requires a pandas-like table")
    h = hashlib.sha256()
    h.update(repr(tuple(str(c) for c in frame.columns)).encode("utf-8"))
    h.update(repr(tuple(str(frame[c].dtype) for c in frame.columns)).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(frame, index=True).to_numpy(
        dtype=np.uint64, copy=False)
    h.update(np.ascontiguousarray(hashed).tobytes())
    return h.hexdigest()


def _a5_5_adf_source_md5() -> str:
    """Hash the production AliasDataFrame.py implementation, not package __init__."""
    import hashlib
    import importlib
    try:
        mod = importlib.import_module(
            "dfextensions.AliasDataFrame.AliasDataFrame")
        path = getattr(mod, "__file__", "")
        if not path:
            return ""
        with open(path, "rb") as stream:
            return hashlib.md5(stream.read()).hexdigest()
    except Exception:
        return ""


def a5_5_realdata_case(root_path: str, gallery_module=None) -> CaseSpec:
    env_status, reason = _a5_5_environment_status(
        root_path, gallery_module=gallery_module)
    applicable = env_status != A5_2_ENV_UNAVAILABLE
    applicability_reason = reason if not applicable else ""
    return CaseSpec(
        case_id=A5_5_CASE_ID,
        claim_id="I4.real_g7_gb_sector_reuse.A5.5",
        title="real G7.34 consumes G7.33-prepared GB state unchanged",
        claim=("after trusted G7.33 prepares CalibBias1 and "
               "dcar_tpc_vertex_predicted0 on the canonical deterministic EAGER "
               "20% sample, trusted G7.34 executes the sector profile while "
               "leaving both measured prepared artifacts unchanged; a poison on "
               "the time_series_draw module-global calibBiasResolution binding "
               "acts as a regression tripwire if a future G7.34 begins refitting "
               "through that seam"),
        failure_means=("G7.34 mutated CalibBias1 or "
                       "dcar_tpc_vertex_predicted0, silently skipped, returned no "
                       "finite public numerical result, or reached the guarded "
                       "module-global calibBiasResolution refit seam"),
        expected_visual=("the existing G7.34 normalized raw-versus-predicted "
                         "DCA_r differential profile versus sector"),
        owner_on_failure="GB",
        purpose="COVERAGE",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            "prepare_gallery_function": A5_5_PREP_FUNCTION,
            "reuse_gallery_function": A5_5_REUSE_FUNCTION,
            "expr": "[dcar_tpc_vertex, dcar_tpc_vertex_predicted0]:sector",
            "selection": "(ncl>60)&(abs(dcar_tpc_vertex)<10)",
            "type": "profile",
            "bins": 36,
            "normalize": "delta",
            "sample_fraction": A5_2_SAMPLE_FRACTION,
            "sample_seed": A5_2_SAMPLE_SEED,
            "expected_subframe": A5_4_SUBFRAME,
            "expected_predicted_column": A5_4_PREDICTED,
        },
        applicable=applicable,
        applicability_reason=applicability_reason,
        setup_contract=("build one canonical EAGER 20% ADF; run trusted G7.33 "
                        "once to prepare GB state; fingerprint CalibBias1 and the "
                        "predicted column; arm the time_series_draw module-global "
                        "calibBiasResolution binding as a future-refit regression "
                        "tripwire; run trusted G7.34; require both fingerprints "
                        "unchanged"),
        preconditions=(
            "ROOT input is readable by the trusted time-series environment",
            "build_adf, fig33_gb_correction_tgl and fig34_gb_correction_sector are available",
            "G7.33 successfully prepares CalibBias1 and dcar_tpc_vertex_predicted0",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable(
                "calibbias1_state_digest", "INDEPENDENT", "FLAT",
                "state.CalibBias1.sha256", comparator="exact",
                rationale="G7.34 must reuse, not mutate/refit, the prepared GB coefficients"),
            Observable(
                "predicted_state_digest", "INDEPENDENT", "FLAT",
                "state.dcar_tpc_vertex_predicted0.sha256", comparator="exact",
                rationale="G7.34 must consume, not rewrite, the prepared predicted correction"),
        ),
        non_claims=(
            "A5.5 is reuse/state-invariance coverage for CalibBias1 and dcar_tpc_vertex_predicted0, not independent GB-fit correctness or whole-ADF non-mutation",
            "the poison covers only the time_series_draw module-global calibBiasResolution binding; a refit routed through a function-local import or lower-level fitting primitive is outside this tripwire",
            "real LAZY/FULL G7.34 execution is re-adjudicated at A6 through the full-lazy gallery gate; this A5.5 machine state-reuse oracle remains EAGER/FRACTION",
            "the known eager root_to_adf/read_tree signature defect is recorded but not fixed here",
        ),
        negative_control="FAMILY_MUTATION:A5.5-REFIT-OR-STATE-MUTATION-MUST-FAIL",
        reference_policy="named-immutable",
    )


def run_a5_5_realdata(case: CaseSpec, root_path: str, *, gallery_module=None,
                       prepared_adf: Any = None,
                       prepared_provenance: dict | None = None) -> CaseResult:
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip

    import os
    import pandas as pd

    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    original_sample = None
    original_calib = None
    gallery = None
    try:
        if case.case_id != A5_5_CASE_ID:
            res.status = INVALID_FIXTURE
            res.detail = f"A5.5 runner received unexpected case {case.case_id!r}"
            return res
        if case.loading_mode != "EAGER" or case.sample_mode != "FRACTION":
            res.status = INVALID_FIXTURE
            res.detail = "A5.5 requires exact EAGER/FRACTION mode"
            return res
        if len(case.observables) != 2:
            res.status = INVALID_FIXTURE
            res.detail = "A5.5 requires exactly two declared state-reuse observables"
            return res

        env_status, why = _a5_5_environment_status(
            root_path, gallery_module=gallery_module)
        if env_status == A5_2_ENV_UNAVAILABLE:
            res.status = INVALID_FIXTURE
            res.detail = f"A5.5 environment changed after CaseSpec creation: {why}"
            return res
        if env_status == A5_2_ENV_CONTRACT_ERROR:
            res.status = INVALID_FIXTURE
            res.detail = why
            return res

        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()

        sample_observed = []
        if prepared_adf is None:
            original_sample = pd.DataFrame.sample

            def observed_sample(self, *args, **kwargs):
                frac = kwargs.get("frac")
                random_state = kwargs.get("random_state")
                out = original_sample(self, *args, **kwargs)
                if frac == A5_2_SAMPLE_FRACTION and random_state == A5_2_SAMPLE_SEED:
                    sample_observed.append({
                        "source_rows": int(len(self)),
                        "selected_rows": int(len(out)),
                        "index_digest_sha256": _a5_2_index_digest(out.index),
                        "index_dtype": str(out.index.dtype),
                    })
                return out

            pd.DataFrame.sample = observed_sample
            try:
                adf = gallery.build_adf(
                    root_path, sample=A5_2_SAMPLE_FRACTION, lazy=False)
            finally:
                pd.DataFrame.sample = original_sample
                original_sample = None
        else:
            adf = prepared_adf
            sample_observed.append(_a6_4_prepared_fraction_sample_evidence(
                adf, prepared_provenance, root_path))

        if len(sample_observed) != 1:
            res.status = INVALID_FIXTURE
            res.detail = (
                "A5.5 expected exactly one canonical build_adf 20% sample; "
                f"observed {len(sample_observed)}")
            return res

        try:
            prepared = getattr(gallery, A5_5_PREP_FUNCTION)(adf)
        except Exception as exc:
            res.status = FAIL
            res.detail = f"A5.5 G7.33 preparation raised {type(exc).__name__}: {exc}"
            res.exception = traceback.format_exc(limit=8)
            return res
        if prepared is None:
            res.status = FAIL
            res.detail = "A5.5 G7.33 preparation silently skipped"
            return res

        subframe_before = (adf.get_subframe(A5_4_SUBFRAME)
                           if hasattr(adf, "get_subframe") else None)
        if (subframe_before is None or not hasattr(subframe_before, "df")
                or len(subframe_before.df) == 0):
            res.status = FAIL
            res.detail = "A5.5 G7.33 did not prepare a non-empty CalibBias1 subframe"
            return res
        if not hasattr(adf, "df") or A5_4_PREDICTED not in adf.df.columns:
            res.status = FAIL
            res.detail = "A5.5 G7.33 did not prepare dcar_tpc_vertex_predicted0"
            return res

        predicted_before = np.asarray(adf.df[A5_4_PREDICTED])
        finite_before = int(np.isfinite(predicted_before).sum())
        if finite_before <= 0:
            res.status = FAIL
            res.detail = "A5.5 prepared predicted column has no finite values"
            return res

        before = {
            "calibbias1_state_digest": _a5_5_frame_digest(subframe_before.df),
            "predicted_state_digest": _a5_5_array_digest(predicted_before),
        }

        # Poison the calibration seam after successful G7.33 preparation.
        # G7.34 is defined as pure reuse and must not call it again.
        original_calib = getattr(gallery, "calibBiasResolution", None)
        if original_calib is None or not callable(original_calib):
            res.status = INVALID_FIXTURE
            res.detail = "A5.5 cannot poison missing calibBiasResolution seam"
            return res

        def forbidden_recalibration(*args, **kwargs):
            raise AssertionError("A5.5 G7.34 reached forbidden calibBiasResolution refit")

        setattr(gallery, "calibBiasResolution", forbidden_recalibration)
        try:
            raw = getattr(gallery, A5_5_REUSE_FUNCTION)(adf)
        finally:
            setattr(gallery, "calibBiasResolution", original_calib)
            original_calib = None

        if raw is None:
            res.status = FAIL
            res.detail = "A5.5 G7.34 optional gallery skip is not an acceptance PASS"
            return res

        payload = unwrap("draw", raw)

        subframe_after = (adf.get_subframe(A5_4_SUBFRAME)
                          if hasattr(adf, "get_subframe") else None)
        if subframe_after is None or not hasattr(subframe_after, "df"):
            res.status = FAIL
            res.detail = "A5.5 G7.34 lost CalibBias1"
            return res
        if A5_4_PREDICTED not in adf.df.columns:
            res.status = FAIL
            res.detail = "A5.5 G7.34 lost dcar_tpc_vertex_predicted0"
            return res

        predicted_after = np.asarray(adf.df[A5_4_PREDICTED])
        after = {
            "calibbias1_state_digest": _a5_5_frame_digest(subframe_after.df),
            "predicted_state_digest": _a5_5_array_digest(predicted_after),
        }

        for observable in case.observables:
            res.observable_contract.append(_contract(observable))
            ref = before[observable.name]
            cand = after[observable.name]
            res.observed[observable.name] = {"before_G7_34": ref, "after_G7_34": cand}
            comparison = compare_observable(observable, ref, cand)
            res.comparisons.append(comparison_evidence(
                observable, comparison,
                reference_label="after_G7_33",
                candidate_label="after_G7_34"))
            res.executed_comparisons += 1
            if not comparison.ok:
                res.status = FAIL
                res.detail = f"{observable.name}: G7.34 mutated prepared state"
                return res

        if res.executed_comparisons != 2:
            res.status = INVALID_FIXTURE
            res.detail = "A5.5 did not execute both declared state comparisons"
            return res

        public_evidence = _a5_4_public_numeric_evidence(payload.stats)
        if public_evidence["finite_values"] <= 0:
            res.status = FAIL
            res.detail = "A5.5 G7.34 public result has no finite profile evidence"
            return res

        sample = sample_observed[0]
        res.observed["realdata_provenance"] = {
            "input_path": os.path.abspath(root_path),
            "input_size_bytes": int(os.path.getsize(root_path)),
            "input_mtime_ns": int(os.stat(root_path).st_mtime_ns),
            "loading_mode": "EAGER",
            "sample_mode": "FRACTION",
            "sample_fraction": A5_2_SAMPLE_FRACTION,
            "sample_seed": A5_2_SAMPLE_SEED,
            "sampling_algorithm": (
                "pandas.DataFrame.sample(frac=0.20, random_state=42) "
                "observed at runtime"),
            "adf_source_md5": _a5_5_adf_source_md5(),
            "adf_source_module": "dfextensions.AliasDataFrame.AliasDataFrame",
            "ingest_entrypoint": "time_series.root_to_adf",
            "known_loader_defect": A5_5_LOADER_BUG_ID,
            **sample,
        }
        res.observed["g7_34_evidence"] = {
            "prepare_gallery_function": A5_5_PREP_FUNCTION,
            "reuse_gallery_function": A5_5_REUSE_FUNCTION,
            "recalibration_poison_active": True,
            "calibbias1_rows_before": int(len(subframe_before.df)),
            "calibbias1_rows_after": int(len(subframe_after.df)),
            "predicted_values": int(predicted_after.size),
            "finite_predicted_values": int(np.isfinite(predicted_after).sum()),
            "public_numeric_evidence": public_evidence,
            "numeric_summary": _a5_2_numeric_summary(payload.stats),
        }
        res.status = PASS
        res.detail = ""
        return res
    except AssertionError as exc:
        res.status = FAIL
        res.detail = str(exc)
        res.exception = traceback.format_exc(limit=6)
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"{type(exc).__name__}: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        if original_calib is not None and gallery is not None:
            try:
                setattr(gallery, "calibBiasResolution", original_calib)
            except Exception:
                pass
        if original_sample is not None:
            try:
                import pandas as pd
                pd.DataFrame.sample = original_sample
            except Exception:
                pass
        _close()
        res.wall_time_s = round(time.time() - t0, 4)


def run_a5_5_realdata_gate(root_path: str, *, manifest_path: str,
                           gallery_module=None) -> tuple[CaseResult, dict, int]:
    case = a5_5_realdata_case(root_path, gallery_module=gallery_module)
    result = run_a5_5_realdata(
        case, root_path, gallery_module=gallery_module)
    extra = {}
    if isinstance(result.observed.get("realdata_provenance"), dict):
        extra.update(result.observed["realdata_provenance"])
    doc = write_manifest(manifest_path, [result], [case], extra=extra)
    return result, doc, strict_exit_code([result], [case])

# ─────────────────────────────────────────────────────────────────────────────
# A5.6 — real G7.34 logical-definition/structure invariance
# ─────────────────────────────────────────────────────────────────────────────

A5_6_CASE_ID = "I4-REAL-G7-GB-LOGICAL-STATE-EAGER-20PCT-01"


A5_6_DEFINITION_EXPORT_KWARGS = {
    "include_precision_stats": False,
    "include_subframes": True,
    "within_group_sort": "schema",
}


def _a5_6_definition_schema(adf: Any) -> dict:
    """Public semantic blueprint used for same-process logical-state comparison."""
    exporter = getattr(adf, "export_definition_schema", None)
    if not callable(exporter):
        raise HarnessError(
            "A5.6 requires public export_definition_schema()")
    schema = exporter(**A5_6_DEFINITION_EXPORT_KWARGS)
    if not isinstance(schema, dict):
        raise HarnessError(
            "A5.6 export_definition_schema() did not return a dict")
    return schema


def _a5_6_json_digest(value: Any) -> str:
    import hashlib
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


A5_6_VOLATILE_DEFINITION_PATHS = ("__meta__.created_at",)


def _a5_6_normalized_definition_schema(adf: Any) -> dict:
    """Return public definition state with export-time-only metadata removed."""
    schema = copy.deepcopy(_a5_6_definition_schema(adf))
    meta = schema.get("__meta__")
    if isinstance(meta, dict):
        meta.pop("created_at", None)
    return schema


def _a5_6_definition_schema_digest(adf: Any) -> str:
    return _a5_6_json_digest(_a5_6_normalized_definition_schema(adf))


def _a5_6_parent_structure(adf: Any) -> dict:
    if not hasattr(adf, "df") or not hasattr(adf.df, "columns"):
        raise HarnessError("A5.6 requires a pandas-like parent df")
    return {
        "columns": [str(c) for c in adf.df.columns],
        "dtypes": [str(adf.df[c].dtype) for c in adf.df.columns],
    }


def _a5_6_parent_structure_digest(adf: Any) -> str:
    return _a5_6_json_digest(_a5_6_parent_structure(adf))


def a5_6_realdata_case(root_path: str, gallery_module=None) -> CaseSpec:
    env_status, reason = _a5_5_environment_status(
        root_path, gallery_module=gallery_module)
    applicable = env_status != A5_2_ENV_UNAVAILABLE
    applicability_reason = reason if not applicable else ""
    return CaseSpec(
        case_id=A5_6_CASE_ID,
        claim_id="I4.real_g7_gb_logical_state.A5.6",
        title="real G7.34 preserves measured GB data state and public ADF definition state",
        claim=("after trusted G7.33 prepares the real GB state on the canonical "
               "deterministic EAGER 20% sample, trusted G7.34 leaves unchanged "
               "the full CalibBias1 frame, the full materialized predicted "
               "column, the public export_definition_schema() blueprint, and "
               "the parent dataframe column/dtype structure"),
        failure_means=("G7.34 changed either measured GB data artifact, altered "
                       "public alias/subframe/schema definitions, added/dropped/"
                       "retyped a parent column, silently skipped, returned no "
                       "finite public numerical evidence, or reached the guarded "
                       "module-global recalibration seam"),
        expected_visual=("the existing G7.34 normalized raw-versus-predicted "
                         "DCA_r differential profile versus sector"),
        owner_on_failure="GB",
        purpose="INVARIANCE",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            "prepare_gallery_function": A5_5_PREP_FUNCTION,
            "reuse_gallery_function": A5_5_REUSE_FUNCTION,
            "sample_fraction": A5_2_SAMPLE_FRACTION,
            "sample_seed": A5_2_SAMPLE_SEED,
            "definition_surface": "normalized export_definition_schema",
            "definition_export_kwargs": dict(A5_6_DEFINITION_EXPORT_KWARGS),
            "parent_structure": "ordered column names + dtypes",
        },
        applicable=applicable,
        applicability_reason=applicability_reason,
        setup_contract=("build one canonical EAGER 20% ADF; run trusted G7.33; "
                        "fingerprint two GB data artifacts plus public definition "
                        "schema and parent column/dtype structure; arm the same "
                        "future-refit tripwire used by A5.5; run trusted G7.34; "
                        "compare all four observables exactly"),
        preconditions=(
            "ROOT input is readable by the trusted time-series environment",
            "public export_definition_schema() is available",
            "G7.33 successfully prepares CalibBias1 and dcar_tpc_vertex_predicted0",
        ),
        figure_contract=FigureContract(
            expected_panels="one panel",
            panel_roles="main: normalized raw-versus-predicted DCA_r profile versus sector",
            expected_traces="two profile traces: raw and predicted",
            expected_group_count="1",
            primary_comparison=(
                "four exact before/after state digests across G7.34, with finite "
                "public profile evidence as a secondary execution guard"),
            residual_definition=(
                "candidate after-G7.34 state digest minus/equality against the "
                "before-G7.34 reference state; visual profile residual is the "
                "trusted normalize='delta' raw-versus-predicted comparison"),
            accepted_envelope=(
                "all four declared state digests exactly equal; public profile "
                "evidence contains at least one finite value"),
            case_ids=(A5_6_CASE_ID,),
            proof_kind="CONSISTENCY",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable(
                "calibbias1_state_digest", "INDEPENDENT", "FLAT",
                "state.CalibBias1.sha256", comparator="exact",
                rationale="retain A5.5 full coefficient-frame invariance"),
            Observable(
                "predicted_state_digest", "INDEPENDENT", "FLAT",
                "state.dcar_tpc_vertex_predicted0.sha256", comparator="exact",
                rationale="retain A5.5 full predicted-array invariance"),
            Observable(
                "parent_structure_digest", "INDEPENDENT", "FLAT",
                "state.parent_columns_dtypes.sha256", comparator="exact",
                rationale="detect persistent parent-column addition/drop/retype"),
            Observable(
                "definition_schema_digest", "INDEPENDENT", "FLAT",
                "state.export_definition_schema.sha256", comparator="exact",
                rationale=("detect alias/subframe/schema-definition drift through "
                           "the public definition export after removing only the "
                           "volatile __meta__.created_at export timestamp")),
        ),
        non_claims=(
            "A5.6 does not hash all parent dataframe values; only the two GB data artifacts are value-fingerprinted",
            "A5.6 deliberately excludes private caches and object identity from the logical-state contract",
            "the public definition digest removes only __meta__.created_at because it is regenerated by each export call and is not logical state",
            "the definition digest is a same-process comparison and is not an A6 cross-version immutable reference",
            "the poison covers only the time_series_draw module-global calibBiasResolution binding",
            "real LAZY/FULL G7.34 execution is re-adjudicated at A6 through the full-lazy gallery gate; this A5.6 logical-state oracle remains EAGER/FRACTION",
        ),
        negative_control="FAMILY_MUTATION:A5.6-LOGICAL-DEFINITION-OR-STRUCTURE-MUTATION-MUST-FAIL",
        reference_policy="same-process",
    )


def run_a5_6_realdata(case: CaseSpec, root_path: str, *, gallery_module=None,
                       prepared_adf: Any = None,
                       prepared_provenance: dict | None = None) -> CaseResult:
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip

    import os
    import pandas as pd

    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    original_sample = None
    original_calib = None
    gallery = None
    try:
        if case.case_id != A5_6_CASE_ID:
            res.status = INVALID_FIXTURE
            res.detail = f"A5.6 runner received unexpected case {case.case_id!r}"
            return res
        if case.loading_mode != "EAGER" or case.sample_mode != "FRACTION":
            res.status = INVALID_FIXTURE
            res.detail = "A5.6 requires exact EAGER/FRACTION mode"
            return res
        if len(case.observables) != 4:
            res.status = INVALID_FIXTURE
            res.detail = "A5.6 requires exactly four declared observables"
            return res

        env_status, why = _a5_5_environment_status(
            root_path, gallery_module=gallery_module)
        if env_status == A5_2_ENV_UNAVAILABLE:
            res.status = INVALID_FIXTURE
            res.detail = f"A5.6 environment changed after CaseSpec creation: {why}"
            return res
        if env_status == A5_2_ENV_CONTRACT_ERROR:
            res.status = INVALID_FIXTURE
            res.detail = why
            return res

        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()

        sample_observed = []
        if prepared_adf is None:
            original_sample = pd.DataFrame.sample

            def observed_sample(self, *args, **kwargs):
                frac = kwargs.get("frac")
                random_state = kwargs.get("random_state")
                out = original_sample(self, *args, **kwargs)
                if frac == A5_2_SAMPLE_FRACTION and random_state == A5_2_SAMPLE_SEED:
                    sample_observed.append({
                        "source_rows": int(len(self)),
                        "selected_rows": int(len(out)),
                        "index_digest_sha256": _a5_2_index_digest(out.index),
                        "index_dtype": str(out.index.dtype),
                    })
                return out

            pd.DataFrame.sample = observed_sample
            try:
                adf = gallery.build_adf(
                    root_path, sample=A5_2_SAMPLE_FRACTION, lazy=False)
            finally:
                pd.DataFrame.sample = original_sample
                original_sample = None
        else:
            adf = prepared_adf
            sample_observed.append(_a6_4_prepared_fraction_sample_evidence(
                adf, prepared_provenance, root_path))

        if len(sample_observed) != 1:
            res.status = INVALID_FIXTURE
            res.detail = (
                "A5.6 expected exactly one canonical build_adf 20% sample; "
                f"observed {len(sample_observed)}")
            return res

        try:
            prepared = getattr(gallery, A5_5_PREP_FUNCTION)(adf)
        except Exception as exc:
            res.status = FAIL
            res.detail = f"A5.6 G7.33 preparation raised {type(exc).__name__}: {exc}"
            res.exception = traceback.format_exc(limit=8)
            return res
        if prepared is None:
            res.status = FAIL
            res.detail = "A5.6 G7.33 preparation silently skipped"
            return res

        subframe_before = (adf.get_subframe(A5_4_SUBFRAME)
                           if hasattr(adf, "get_subframe") else None)
        if (subframe_before is None or not hasattr(subframe_before, "df")
                or len(subframe_before.df) == 0):
            res.status = FAIL
            res.detail = "A5.6 G7.33 did not prepare non-empty CalibBias1"
            return res
        if not hasattr(adf, "df") or A5_4_PREDICTED not in adf.df.columns:
            res.status = FAIL
            res.detail = "A5.6 G7.33 did not prepare dcar_tpc_vertex_predicted0"
            return res

        predicted_before = np.asarray(adf.df[A5_4_PREDICTED])
        if int(np.isfinite(predicted_before).sum()) <= 0:
            res.status = FAIL
            res.detail = "A5.6 reference predicted column has no finite values"
            return res

        before = {
            "calibbias1_state_digest": _a5_5_frame_digest(subframe_before.df),
            "predicted_state_digest": _a5_5_array_digest(predicted_before),
            "definition_schema_digest": _a5_6_definition_schema_digest(adf),
            "parent_structure_digest": _a5_6_parent_structure_digest(adf),
        }

        original_calib = getattr(gallery, "calibBiasResolution", None)
        if original_calib is None or not callable(original_calib):
            res.status = INVALID_FIXTURE
            res.detail = "A5.6 cannot arm missing calibBiasResolution tripwire"
            return res

        def forbidden_recalibration(*args, **kwargs):
            raise AssertionError(
                "A5.6 G7.34 reached guarded calibBiasResolution refit seam")

        setattr(gallery, "calibBiasResolution", forbidden_recalibration)
        try:
            raw = getattr(gallery, A5_5_REUSE_FUNCTION)(adf)
        finally:
            setattr(gallery, "calibBiasResolution", original_calib)
            original_calib = None

        if raw is None:
            res.status = FAIL
            res.detail = "A5.6 G7.34 optional gallery skip is not an acceptance PASS"
            return res

        payload = unwrap("draw", raw)

        subframe_after = (adf.get_subframe(A5_4_SUBFRAME)
                          if hasattr(adf, "get_subframe") else None)
        if subframe_after is None or not hasattr(subframe_after, "df"):
            res.status = FAIL
            res.detail = "A5.6 G7.34 lost CalibBias1"
            return res
        if A5_4_PREDICTED not in adf.df.columns:
            res.status = FAIL
            res.detail = "A5.6 G7.34 lost dcar_tpc_vertex_predicted0"
            return res

        predicted_after = np.asarray(adf.df[A5_4_PREDICTED])
        after = {
            "calibbias1_state_digest": _a5_5_frame_digest(subframe_after.df),
            "predicted_state_digest": _a5_5_array_digest(predicted_after),
            "definition_schema_digest": _a5_6_definition_schema_digest(adf),
            "parent_structure_digest": _a5_6_parent_structure_digest(adf),
        }

        for observable in case.observables:
            res.observable_contract.append(_contract(observable))
            ref = before[observable.name]
            cand = after[observable.name]
            res.observed[observable.name] = {
                "before_G7_34": ref,
                "after_G7_34": cand,
            }
            comparison = compare_observable(observable, ref, cand)
            res.comparisons.append(comparison_evidence(
                observable, comparison,
                reference_label="before_G7_34",
                candidate_label="after_G7_34"))
            res.executed_comparisons += 1
            if not comparison.ok:
                res.status = FAIL
                res.detail = f"{observable.name}: G7.34 mutated declared state"
                return res

        if res.executed_comparisons != 4:
            res.status = INVALID_FIXTURE
            res.detail = "A5.6 did not execute all four declared comparisons"
            return res

        public_evidence = _a5_4_public_numeric_evidence(payload.stats)
        if public_evidence["finite_values"] <= 0:
            res.status = FAIL
            res.detail = "A5.6 G7.34 public result has no finite profile evidence"
            return res

        sample = sample_observed[0]
        res.observed["realdata_provenance"] = {
            "input_path": os.path.abspath(root_path),
            "input_size_bytes": int(os.path.getsize(root_path)),
            "input_mtime_ns": int(os.stat(root_path).st_mtime_ns),
            "loading_mode": "EAGER",
            "sample_mode": "FRACTION",
            "sample_fraction": A5_2_SAMPLE_FRACTION,
            "sample_seed": A5_2_SAMPLE_SEED,
            "sampling_algorithm": (
                "pandas.DataFrame.sample(frac=0.20, random_state=42) "
                "observed at runtime"),
            "adf_source_md5": _a5_5_adf_source_md5(),
            "adf_source_module": "dfextensions.AliasDataFrame.AliasDataFrame",
            "ingest_entrypoint": "time_series.root_to_adf",
            "known_loader_defect": A5_5_LOADER_BUG_ID,
            **sample,
        }
        res.observed["g7_34_logical_state_evidence"] = {
            "definition_surface": "normalized export_definition_schema",
            "definition_export_kwargs": dict(A5_6_DEFINITION_EXPORT_KWARGS),
            "definition_ignored_volatile_paths": list(A5_6_VOLATILE_DEFINITION_PATHS),
            "parent_structure_surface": "ordered column names + dtypes",
            "declared_observables": 4,
            "executed_comparisons": res.executed_comparisons,
            "calibbias1_rows_before": int(len(subframe_before.df)),
            "calibbias1_rows_after": int(len(subframe_after.df)),
            "predicted_values": int(predicted_after.size),
            "finite_predicted_values": int(np.isfinite(predicted_after).sum()),
            "public_numeric_evidence": public_evidence,
        }
        res.status = PASS
        res.detail = ""
        return res
    except AssertionError as exc:
        res.status = FAIL
        res.detail = str(exc)
        res.exception = traceback.format_exc(limit=6)
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"{type(exc).__name__}: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        if original_calib is not None and gallery is not None:
            try:
                setattr(gallery, "calibBiasResolution", original_calib)
            except Exception:
                pass
        if original_sample is not None:
            try:
                import pandas as pd
                pd.DataFrame.sample = original_sample
            except Exception:
                pass
        _close()
        res.wall_time_s = round(time.time() - t0, 4)


def run_a5_6_realdata_gate(root_path: str, *, manifest_path: str,
                           gallery_module=None) -> tuple[CaseResult, dict, int]:
    case = a5_6_realdata_case(root_path, gallery_module=gallery_module)
    result = run_a5_6_realdata(
        case, root_path, gallery_module=gallery_module)
    extra = {}
    if isinstance(result.observed.get("realdata_provenance"), dict):
        extra.update(result.observed["realdata_provenance"])
    doc = write_manifest(manifest_path, [result], [case], extra=extra)
    return result, doc, strict_exit_code([result], [case])


# ─────────────────────────────────────────────────────────────────────────────
# A7 — post-Stage-A correctness hardening:
#      vector × facet × fit × summary_fit semantic-coordinate oracle
# ─────────────────────────────────────────────────────────────────────────────

A7_1_CASE_ID = "I4-REAL-VECTOR-FACET-SUMMARYFIT-SEMANTICS-EAGER-20PCT-01"
A7_1_GALLERY_FUNCTION = "fig43_vector_facet_summary_fit"
A7_1_BRANCH_LABELS = ("early", "middle", "late")
A7_1_FACET_VALUES = (0, 1)
A7_1_BINS = 30
A7_1_RANGE = (-1.5, 1.5)
A7_1_MIN_ENTRIES = 50


def _a7_1_environment_status(root_path: str, gallery_module=None) -> tuple[str, str]:
    """A7 availability: real input + the existing gallery owners only."""
    if not root_path:
        return A5_2_ENV_UNAVAILABLE, "no ROOT input path was supplied"
    if not os.path.isfile(root_path):
        return (A5_2_ENV_UNAVAILABLE,
                f"ROOT input is unavailable: {os.path.abspath(root_path)}")
    try:
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None)
        if missing in A5_2_EXTERNAL_MODULES:
            return (A5_2_ENV_UNAVAILABLE,
                    f"time_series_draw environment unavailable: "
                    f"{type(exc).__name__}: {exc}")
        return (A5_2_ENV_CONTRACT_ERROR,
                f"time_series_draw import contract failure: "
                f"{type(exc).__name__}: {exc}")
    except Exception as exc:
        return (A5_2_ENV_CONTRACT_ERROR,
                f"time_series_draw import contract failure: "
                f"{type(exc).__name__}: {exc}")

    required = ("build_adf", A7_1_GALLERY_FUNCTION)
    missing = [name for name in required if not callable(getattr(gallery, name, None))]
    if missing:
        return (A5_2_ENV_CONTRACT_ERROR,
                f"time_series_draw missing required callable(s): {missing}")
    return A5_2_ENV_AVAILABLE, ""


def a7_1_realdata_case(root_path: str, gallery_module=None) -> CaseSpec:
    """Dedicated real-data correctness CaseSpec for the composed seam."""
    env_status, reason = _a7_1_environment_status(
        root_path, gallery_module=gallery_module)
    applicable = env_status != A5_2_ENV_UNAVAILABLE
    applicability_reason = reason if not applicable else ""
    case = CaseSpec(
        case_id=A7_1_CASE_ID,
        claim_id="I4.real_vector_facet_summaryfit_semantics.A7.1",
        title="real vector×facet×fit×summary_fit preserves semantic coordinates",
        claim=("the public ADF draw() result for three time-selection branches across "
               "two side_type facets returns exactly the independently expected "
               "branch×facet semantic product, no invented group dimension, and a "
               "rendered summary table whose identity cells match the validated rows"),
        failure_means=("dfdraw attached a fit/summary row to the wrong vector branch or "
                       "facet, invented a group coordinate, omitted/duplicated a semantic "
                       "cell, degraded summary_fit to a diagnostic note, or rendered a "
                       "table whose identities disagree with the validated data"),
        expected_visual=("two side_type facet panels, each carrying the same three "
                         "time-branch profile families, plus a six-row summary-fit table"),
        owner_on_failure="dfdraw",
        purpose="CORRECTNESS",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            "gallery_function": A7_1_GALLERY_FUNCTION,
            "surface": "draw",
            "expr": "dcar_tpc_vertex:tgl",
            "selection": "BASE_SEL&(side_type<2)",
            "selection_labels": list(A7_1_BRANCH_LABELS),
            "facet_by": "side_type",
            "fit": "pol1",
            "summary_fit": "table",
            "bins": A7_1_BINS,
            "range": list(A7_1_RANGE),
            "min_entries": A7_1_MIN_ENTRIES,
            "sample_fraction": A5_2_SAMPLE_FRACTION,
            "sample_seed": A5_2_SAMPLE_SEED,
        },
        applicable=applicable,
        applicability_reason=applicability_reason,
        setup_contract=("reuse the canonical EAGER 20% Stage-A ADF build; derive the "
                        "three time quantile branches independently from adf.df; then "
                        "execute the existing public fig43/adf.draw() call without "
                        "consulting returned fit nesting to build expected coordinates"),
        preconditions=(
            "the ROOT input file is readable by the trusted time-series gallery environment",
            "time_s, side_type, ncl, dcar_tpc_vertex and tgl are present in the prepared ADF",
            "all six branch×facet cells contain enough data to exercise pol1 fitting",
        ),
        figure_contract=FigureContract(
            expected_panels="two side_type facet panels plus one separate summary-fit table page",
            panel_roles="side_type=0 and side_type=1; the table reports branch×facet identities",
            expected_traces="three branch-resolved profile families with pol1 fits in each facet panel",
            expected_group_count="zero group_by dimensions; summary rows must report group=None",
            primary_comparison=("raw/spec Cartesian branch×facet product -> summary_fit data "
                                "coordinates -> rendered table identity cells"),
            residual_definition="not applicable; this case proves semantic coordinate identity",
            accepted_envelope="exact identity/cardinality; six unique cells; no diagnostic fallback",
            case_ids=(A7_1_CASE_ID,),
            proof_kind="CORRECTNESS",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable(
                name="semantic_coordinates",
                source="INDEPENDENT", access="ARRAY",
                path="raw/spec branch×facet Cartesian product",
                comparator="exact"),
            Observable(
                name="branch_selected_rows",
                source="INDEPENDENT", access="ARRAY",
                path="raw selected-row counts by declared branch order",
                comparator="exact"),
            Observable(
                name="rendered_table_identities",
                source="INDEPENDENT", access="ARRAY",
                path="Matplotlib summary table identity column",
                comparator="exact"),
        ),
        non_claims=(
            "this ADF real-data oracle does not independently prove analytic pol1 coefficients",
            "legend text is not used as branch truth while selection_labels legend handling is a separate dfdraw bug",
        ),
        negative_control="FAMILY_MUTATION:A7_VECTOR_FACET_SUMMARY_SEMANTICS",
        reference_policy="named-immutable",
    )
    return case


def _a7_1_expected_model(adf: Any) -> dict:
    """Independent raw-frame truth for the fig43 call specification."""
    if not hasattr(adf, "df"):
        raise HarnessError("A7 prepared object has no .df raw-frame owner")
    df = adf.df
    required = ("time_s", "side_type", "ncl", "dcar_tpc_vertex", "tgl")
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise HarnessError(f"A7 raw frame missing required columns: {missing}")
    if len(df) == 0:
        raise HarnessError("A7 raw frame is empty")

    q1, q2 = df["time_s"].quantile([1.0 / 3.0, 2.0 / 3.0]).to_numpy()
    base = ((df["ncl"] > 60)
            & (np.abs(np.asarray(df["dcar_tpc_vertex"], dtype=float)) < 10)
            & (df["side_type"] < 2))
    branch_masks = (
        df["time_s"] < q1,
        (df["time_s"] >= q1) & (df["time_s"] < q2),
        df["time_s"] >= q2,
    )

    cell_counts: dict[str, int] = {}
    eligible_bins: dict[str, int] = {}
    branch_totals = []
    coordinates = []
    summary_identities = []
    for branch_index, (label, branch_mask) in enumerate(
            zip(A7_1_BRANCH_LABELS, branch_masks)):
        branch_total = 0
        for facet in A7_1_FACET_VALUES:
            mask = base & branch_mask & (df["side_type"] == facet)
            count = int(np.count_nonzero(np.asarray(mask, dtype=bool)))
            key = f"{label}|side_type={facet}"
            cell_counts[key] = count
            branch_total += count
            coordinates.append(key)
            summary_identities.append(f"side_type={facet} branch={branch_index}")

            values = np.asarray(df.loc[mask, "tgl"], dtype=float)
            values = values[np.isfinite(values)]
            hist, _ = np.histogram(values, bins=A7_1_BINS, range=A7_1_RANGE)
            n_fit_bins = int(np.count_nonzero(hist >= A7_1_MIN_ENTRIES))
            eligible_bins[key] = n_fit_bins
            if count < A7_1_MIN_ENTRIES or n_fit_bins < 2:
                raise HarnessError(
                    f"A7 semantic cell {key} cannot exercise pol1 robustly: "
                    f"selected_rows={count}, bins_with_>={A7_1_MIN_ENTRIES}={n_fit_bins}")
        branch_totals.append(branch_total)

    return {
        "q1": float(q1),
        "q2": float(q2),
        "coordinates": sorted(coordinates),
        "summary_identities": sorted(summary_identities),
        "branch_totals": branch_totals,
        "cell_counts": cell_counts,
        "eligible_fit_bins": eligible_bins,
    }


def _a7_1_parse_summary_identity(row: dict) -> tuple[str, str]:
    """Map public composite facet label to ADF-owned branch label + facet."""
    text = str(row.get("facet"))
    facet_match = re.search(r"(?:^|\s)side_type=([^\s]+)", text)
    branch_match = re.search(r"(?:^|\s)branch=([^\s]+)", text)
    if facet_match is None or branch_match is None:
        raise HarnessError(
            f"A7 summary row lacks side_type/branch identity: facet={text!r}")
    try:
        facet = int(float(facet_match.group(1)))
        branch_index = int(float(branch_match.group(1)))
    except ValueError as exc:
        raise HarnessError(f"A7 summary identity is not numeric: {text!r}") from exc
    if facet not in A7_1_FACET_VALUES:
        raise HarnessError(f"A7 unexpected facet side_type={facet}: {text!r}")
    if not 0 <= branch_index < len(A7_1_BRANCH_LABELS):
        raise HarnessError(f"A7 unexpected branch index {branch_index}: {text!r}")
    return A7_1_BRANCH_LABELS[branch_index], f"side_type={facet}"


def _a7_1_validate_summary_rows(rows: Sequence[dict], expected: dict) -> dict:
    """Fail-loud semantic-coordinate oracle; order is deliberately non-semantic."""
    if len(rows) != 6:
        raise HarnessError(f"A7 expected exactly 6 summary rows, got {len(rows)}")
    observed_coordinates = []
    observed_identities = []
    for row in rows:
        if not isinstance(row, dict):
            raise HarnessError(f"A7 summary row is not a dict: {type(row).__name__}")
        if row.get("group") is not None:
            raise HarnessError(
                f"A7 no group_by was requested but summary row reports group={row.get('group')!r}")
        label, facet = _a7_1_parse_summary_identity(row)
        observed_coordinates.append(f"{label}|{facet}")
        observed_identities.append(str(row.get("facet")))
        if row.get("fit_name") not in (None, "pol1"):
            raise HarnessError(f"A7 unexpected fit_name={row.get('fit_name')!r}")
        if "fit_status" in row and row.get("fit_status") != "ok":
            raise HarnessError(
                f"A7 fit failed for {label}|{facet}: status={row.get('fit_status')!r}")

    if len(set(observed_coordinates)) != len(observed_coordinates):
        raise HarnessError(f"A7 duplicate semantic coordinates: {sorted(observed_coordinates)}")
    if sorted(observed_coordinates) != list(expected["coordinates"]):
        raise HarnessError(
            "A7 semantic-coordinate mismatch; "
            f"expected={expected['coordinates']}, observed={sorted(observed_coordinates)}")
    if sorted(observed_identities) != list(expected["summary_identities"]):
        raise HarnessError(
            "A7 public summary identity mismatch; "
            f"expected={expected['summary_identities']}, "
            f"observed={sorted(observed_identities)}")
    return {
        "coordinates": sorted(observed_coordinates),
        "summary_identities": sorted(observed_identities),
    }


def _a7_1_rendered_identity_cells(table_fig: Any) -> list[str]:
    """Read the rendered summary identity column; presentation order is ignored."""
    if table_fig is None:
        raise HarnessError("A7 summary_fit carries no rendered table figure")
    cells = None
    for ax in getattr(table_fig, "axes", ()):
        for table in getattr(ax, "tables", ()):
            cells = table.get_celld()
            break
        if cells:
            break
    if not cells:
        raise HarnessError("A7 rendered summary figure contains no Matplotlib Table")
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})
    if not rows or not cols or 0 not in rows:
        raise HarnessError("A7 rendered summary table has no header row")
    header_by_col = {
        c: str(cells[(0, c)].get_text().get_text())
        for c in cols if (0, c) in cells
    }
    identity_cols = [c for c, text in header_by_col.items() if text == "facet"]
    if len(identity_cols) != 1:
        raise HarnessError(
            f"A7 rendered table expected one 'facet' identity column, got {header_by_col}")
    col = identity_cols[0]
    values = [
        str(cells[(r, col)].get_text().get_text())
        for r in rows if r != 0 and (r, col) in cells
    ]
    if len(values) != 6:
        raise HarnessError(f"A7 rendered table expected 6 identity rows, got {len(values)}")
    return sorted(values)


def _a7_1_primary_figure_evidence(axes: Any) -> dict:
    """Secondary structural proof for the two facet panels."""
    try:
        flat = list(np.asarray(axes, dtype=object).reshape(-1))
    except Exception as exc:
        raise HarnessError(f"A7 could not normalize returned axes: {exc}") from exc
    visible = [ax for ax in flat if ax is not None and getattr(ax, "get_visible", lambda: True)()]
    if len(visible) != 2:
        raise HarnessError(f"A7 expected exactly 2 visible facet axes, got {len(visible)}")
    titles = [str(getattr(ax, "get_title", lambda: "")()) for ax in visible]
    facets = set()
    line_counts = []
    for ax, title in zip(visible, titles):
        match = re.search(r"side_type=([^\s]+)", title)
        if match is None:
            raise HarnessError(f"A7 facet panel has no side_type title: {title!r}")
        facets.add(int(float(match.group(1))))
        line_counts.append(len(getattr(ax, "lines", ())))
    if facets != set(A7_1_FACET_VALUES):
        raise HarnessError(f"A7 primary figure facets {sorted(facets)} != {list(A7_1_FACET_VALUES)}")
    if any(n < len(A7_1_BRANCH_LABELS) for n in line_counts):
        raise HarnessError(
            f"A7 primary figure does not visibly carry all three branches: line_counts={line_counts}")
    return {"facet_titles": titles, "line_counts": line_counts}


def run_a7_1_realdata(case: CaseSpec, root_path: str, *, gallery_module=None,
                       prepared_adf: Any = None,
                       prepared_provenance: dict | None = None) -> CaseResult:
    """Execute the A7 real-data semantic alarm on the public fig43/draw path."""
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        if case.case_id != A7_1_CASE_ID:
            res.status = INVALID_FIXTURE
            res.detail = f"A7 runner received unexpected case {case.case_id!r}"
            return res
        if (case.purpose != "CORRECTNESS" or case.gate != "ENVIRONMENT_GATED"
                or case.oracle_kind != "CORRECTNESS"
                or case.loading_mode != "EAGER" or case.sample_mode != "FRACTION"):
            res.status = INVALID_FIXTURE
            res.detail = "A7 runner requires CORRECTNESS/ENVIRONMENT_GATED/EAGER/FRACTION"
            return res

        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        env_status, why = _a7_1_environment_status(root_path, gallery_module=gallery)
        if env_status != A5_2_ENV_AVAILABLE:
            res.status = INVALID_FIXTURE
            res.detail = f"A7 environment/contract changed after CaseSpec creation ({env_status}): {why}"
            return res

        if prepared_adf is None:
            adf, provenance_doc = _a6_4_build_fraction_adf_once(
                root_path, gallery_module=gallery)
        else:
            adf = prepared_adf
            _a6_4_prepared_fraction_sample_evidence(adf, prepared_provenance, root_path)
            provenance_doc = dict(prepared_provenance or {})

        expected = _a7_1_expected_model(adf)
        raw = getattr(gallery, A7_1_GALLERY_FUNCTION)(adf)
        if raw is None or not isinstance(raw, tuple) or len(raw) < 3:
            res.status = FAIL
            res.detail = "A7 fig43 did not return the public (fig, axes, stats) tuple"
            return res
        fig, axes, stats = raw[0], raw[1], raw[2]
        if not isinstance(stats, list) or len(stats) != len(A7_1_BRANCH_LABELS):
            res.status = FAIL
            res.detail = (
                f"A7 expected {len(A7_1_BRANCH_LABELS)} vector branch stats, "
                f"got {type(stats).__name__} len={len(stats) if isinstance(stats, list) else 'n/a'}")
            return res

        actual_branch_totals = []
        for i, branch_stats in enumerate(stats):
            if not isinstance(branch_stats, dict):
                raise HarnessError(f"A7 branch {i} stats is not a dict")
            try:
                actual_branch_totals.append(int(branch_stats["n_total"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise HarnessError(f"A7 branch {i} has no usable n_total") from exc

        if stats[0].get("summary_fit_note") is not None:
            raise HarnessError(
                f"A7 summary_fit degraded to diagnostic note: {stats[0].get('summary_fit_note')}")
        payload = stats[0].get("summary_fit")
        if not isinstance(payload, dict) or not payload:
            raise HarnessError("A7 summary_fit payload is absent or empty")
        rows = payload.get("data")
        if not isinstance(rows, list) or not rows:
            raise HarnessError("A7 summary_fit['data'] is absent or empty")

        semantic = _a7_1_validate_summary_rows(rows, expected)
        rendered_identities = _a7_1_rendered_identity_cells(payload.get("table"))
        if rendered_identities != semantic["summary_identities"]:
            raise HarnessError(
                "A7 rendered table identity cells disagree with validated summary rows; "
                f"rendered={rendered_identities}, data={semantic['summary_identities']}")
        primary = _a7_1_primary_figure_evidence(axes)
        scalar_fit = _a7_1_scalar_fit_decomposition(adf, rows, expected)
        style_invariance = _a7_1_style_invariance(axes)

        reference_values = {
            "semantic_coordinates": list(expected["coordinates"]),
            "branch_selected_rows": list(expected["branch_totals"]),
            "rendered_table_identities": list(expected["summary_identities"]),
        }
        candidate_values = {
            "semantic_coordinates": list(semantic["coordinates"]),
            "branch_selected_rows": list(actual_branch_totals),
            "rendered_table_identities": list(rendered_identities),
        }
        for observable in case.observables:
            res.observable_contract.append(_contract(observable))
            ref = reference_values[observable.name]
            cand = candidate_values[observable.name]
            res.observed[observable.name] = cand
            comparison = compare_observable(observable, ref, cand)
            res.comparisons.append(comparison_evidence(
                observable, comparison,
                reference_label="raw/spec independent oracle",
                candidate_label="public draw/summary/table"))
            res.executed_comparisons += 1
            if not comparison.ok:
                res.status = FAIL
                res.detail = f"A7 {observable.name} mismatch: {comparison.detail}"
                return res

        if res.executed_comparisons != len(case.observables):
            res.status = INVALID_FIXTURE
            res.detail = "A7 did not execute every declared comparison"
            return res

        res.observed["realdata_provenance"] = provenance_doc
        res.observed["semantic_oracle"] = {
            "branch_labels": list(A7_1_BRANCH_LABELS),
            "facet_values": list(A7_1_FACET_VALUES),
            "q1": expected["q1"],
            "q2": expected["q2"],
            "cell_counts": dict(expected["cell_counts"]),
            "eligible_fit_bins": dict(expected["eligible_fit_bins"]),
            "coordinates": list(expected["coordinates"]),
            "summary_identities": list(expected["summary_identities"]),
            "group_expected": None,
        }
        res.observed["figure_structure"] = primary
        res.observed["scalar_fit_decomposition"] = scalar_fit
        res.observed["style_invariance"] = style_invariance
        res.observed["dfdraw_red_green_custody"] = {
            "red": dict(O1_RED_DFDRAW),
            "green": dict(O1_GREEN_DFDRAW),
        }
        res.payload_paths = {
            "draw": ["tuple", 2],
            "summary_fit": ["tuple", 2, 0, "summary_fit"],
        }
        res.status = PASS
        res.detail = ""
        return res
    except HarnessError as exc:
        res.status = FAIL
        res.detail = f"VECTOR_FACET_SUMMARY_SEMANTICS FAIL: {exc}"
        res.exception = traceback.format_exc(limit=6)
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"{type(exc).__name__}: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        _close()
        res.wall_time_s = round(time.time() - t0, 4)


def run_a7_1_realdata_gate(root_path: str, *, manifest_path: str,
                           gallery_module=None) -> tuple[CaseResult, dict, int]:
    case = a7_1_realdata_case(root_path, gallery_module=gallery_module)
    result = run_a7_1_realdata(case, root_path, gallery_module=gallery_module)
    extra = {}
    if isinstance(result.observed.get("realdata_provenance"), dict):
        extra.update(result.observed["realdata_provenance"])
    doc = write_manifest(manifest_path, [result], [case], extra=extra)
    return result, doc, strict_exit_code([result], [case])


# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_77 post-Stage-A gallery-oracle hardening v0.2 — STEP 1 / O1
# ─────────────────────────────────────────────────────────────────────────────

HARDENING_O1_NEG_A_CASE_ID = "I4-REAL-VECTOR-FACET-ONE-ELEMENT-SELECTION-EAGER-20PCT-01"
HARDENING_O1_NEG_B_CASE_ID = "I4-REAL-HIST-VECTOR-FACET-SELECTION-EAGER-20PCT-01"
HARDENING_FACETS = (0, 1)
HARDENING_BASE_SEL = "(ncl>60)&(abs(dcar_tpc_vertex)<10)"
HARDENING_PROFILE_BINS = 30
HARDENING_PROFILE_RANGE = (-1.5, 1.5)

O1_RED_DFDRAW = {
    "md5": "119fac5392b626f82d1bd9ac4d630683",
    "sha256": "7d1a87e7af9564f5a0298426b736418eeef918a9a837993cb444cd999fd4a96a",
}
O1_GREEN_DFDRAW = {
    "md5": "3b7a8b7537745620d1a4b24c61802414",
    "sha256": "930e027acb42b1a0226c99d3cdd5a88b777679fcc007f24ceb2f4c067cbcb6d3",
}

def _first_fit_record(stats: Any) -> dict:
    """Return the first concrete fit record from the public nested fit payload."""
    if isinstance(stats, dict):
        if "fit_name" in stats and ("params" in stats or "slope" in stats):
            return stats
        if "fit" in stats:
            found = _first_fit_record(stats["fit"])
            if found:
                return found
        for value in stats.values():
            found = _first_fit_record(value)
            if found:
                return found
    elif isinstance(stats, (list, tuple)):
        for value in stats:
            found = _first_fit_record(value)
            if found:
                return found
    return {}


def _fit_named_values(record: dict) -> dict:
    if not isinstance(record, dict):
        return {}
    if "params" in record and "param_names" in record:
        names = list(record.get("param_names") or ())
        values = np.asarray(record.get("params"), dtype=float).reshape(-1)
        return {str(name): float(value) for name, value in zip(names, values)}
    out = {}
    for name in ("slope", "intercept", "c0", "c1"):
        if name in record:
            try:
                out[name] = float(record[name])
            except (TypeError, ValueError):
                pass
    return out


def _a7_1_scalar_fit_decomposition(adf: Any, summary_rows: Sequence[dict], expected: dict) -> dict:
    """Compare composed O1 fits with six independent scalar public calls."""
    by_coordinate = {}
    for row in summary_rows:
        label, facet_name = _a7_1_parse_summary_identity(row)
        by_coordinate[f"{label}|{facet_name}"] = row
    q1, q2 = expected["q1"], expected["q2"]
    branch_selections = (
        f"time_s<{q1}",
        f"(time_s>={q1})&(time_s<{q2})",
        f"time_s>={q2}",
    )
    records = []
    max_abs_delta = 0.0
    try:
        for branch_index, (label, branch_selection) in enumerate(
                zip(A7_1_BRANCH_LABELS, branch_selections)):
            for facet in A7_1_FACET_VALUES:
                coordinate = f"{label}|side_type={facet}"
                raw = adf.draw(
                    "dcar_tpc_vertex:tgl",
                    selection=f"{HARDENING_BASE_SEL}&(side_type=={facet})&({branch_selection})",
                    type="profile", bins=A7_1_BINS, range=A7_1_RANGE,
                    fit="pol1", min_entries=A7_1_MIN_ENTRIES, auto_title=False)
                fit_record = _first_fit_record(raw[2])
                scalar = _fit_named_values(fit_record)
                composed_row = by_coordinate.get(coordinate)
                if composed_row is None:
                    raise HarnessError(f"O1 scalar decomposition missing composed row {coordinate}")
                composed = {
                    "slope": float(composed_row["slope"]),
                    "intercept": float(composed_row["intercept"]),
                }
                for name in ("slope", "intercept"):
                    if name not in scalar:
                        raise HarnessError(f"O1 scalar {coordinate} has no {name} fit parameter")
                    delta = abs(float(scalar[name]) - float(composed[name]))
                    max_abs_delta = max(max_abs_delta, delta)
                    if not np.isclose(scalar[name], composed[name], rtol=1e-10, atol=1e-12):
                        raise HarnessError(
                            f"O1 scalar fit mismatch {coordinate}/{name}: "
                            f"scalar={scalar[name]}, composed={composed[name]}")
                records.append({
                    "coordinate": coordinate,
                    "branch_index": branch_index,
                    "facet": facet,
                    "scalar": {k: scalar[k] for k in ("slope", "intercept")},
                    "composed": composed,
                })
                _close()
    finally:
        _close()
    return {"records": records, "max_abs_parameter_delta": float(max_abs_delta)}


def _profile_branch_styles_from_axes(axes: Any, *, n_branches: int) -> dict:
    """Extract the primary profile-line channel tuple for each facet/branch."""
    flat = list(np.asarray(axes, dtype=object).reshape(-1))
    evidence = {}
    for ax in flat:
        if ax is None or not getattr(ax, "get_visible", lambda: True)():
            continue
        title = str(getattr(ax, "get_title", lambda: "")())
        match = re.search(r"side_type=([^\s]+)", title)
        if match is None:
            continue
        facet = int(float(match.group(1)))
        data_lines = [
            line for line in getattr(ax, "lines", ())
            if str(line.get_marker()) == "o" and str(line.get_linestyle()).lower() not in ("none", "")
        ]
        if len(data_lines) < n_branches:
            raise HarnessError(
                f"style oracle facet {facet} found {len(data_lines)} profile lines; expected {n_branches}")
        evidence[facet] = [
            {
                "color": str(line.get_color()),
                "linestyle": str(line.get_linestyle()),
                "marker": str(line.get_marker()),
            }
            for line in data_lines[:n_branches]
        ]
    return evidence


def _a7_1_style_invariance(axes: Any) -> dict:
    styles = _profile_branch_styles_from_axes(axes, n_branches=len(A7_1_BRANCH_LABELS))
    if set(styles) != set(A7_1_FACET_VALUES):
        raise HarnessError(f"O1 style oracle facet set mismatch: {sorted(styles)}")
    reference = styles[A7_1_FACET_VALUES[0]]
    mismatches = []
    for facet in A7_1_FACET_VALUES[1:]:
        for branch, (ref_style, got_style) in enumerate(zip(reference, styles[facet])):
            if ref_style != got_style:
                mismatches.append({
                    "facet": facet, "branch": branch,
                    "reference": ref_style, "observed": got_style,
                })
    if mismatches:
        raise HarnessError(f"O1 branch channel/style mismatch across facets: {mismatches}")
    return {"styles_by_facet": styles, "mismatches": []}


def _hardening_fraction_environment(root_path: str, gallery_module=None,
                                    required=()) -> tuple[str, str]:
    status, reason = _a7_1_environment_status(root_path, gallery_module=gallery_module)
    if status != A5_2_ENV_AVAILABLE:
        return status, reason
    gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    missing = [name for name in required if not callable(getattr(gallery, name, None))]
    if missing:
        return A5_2_ENV_CONTRACT_ERROR, f"time_series_draw missing required callable(s): {missing}"
    return A5_2_ENV_AVAILABLE, ""


def _hardening_case(*, case_id: str, claim_id: str, title: str, claim: str,
                    failure_means: str, expected_visual: str, purpose: str,
                    oracle_kind: str, owner: str, canonical_spec: dict,
                    observables: tuple[Observable, ...], root_path: str,
                    gallery_module=None, gallery_function: str | None = None,
                    loading_mode: str = "EAGER", sample_mode: str = "FRACTION",
                    setup_contract: str = "", preconditions: tuple[str, ...] = (),
                    negative_control: str = "") -> CaseSpec:
    required = (gallery_function,) if gallery_function else ()
    status, reason = _hardening_fraction_environment(
        root_path, gallery_module=gallery_module, required=required)
    return CaseSpec(
        case_id=case_id, claim_id=claim_id, title=title, claim=claim,
        failure_means=failure_means, expected_visual=expected_visual,
        owner_on_failure=owner, purpose=purpose, gate="ENVIRONMENT_GATED",
        oracle_kind=oracle_kind, loading_mode=loading_mode, sample_mode=sample_mode,
        canonical_spec=dict(canonical_spec),
        applicable=status != A5_2_ENV_UNAVAILABLE,
        applicability_reason=reason if status == A5_2_ENV_UNAVAILABLE else "",
        setup_contract=setup_contract, preconditions=preconditions,
        observables=observables, negative_control=negative_control,
        reference_policy="named-immutable" if sample_mode == "FRACTION" else "same-process",
        surfaces_under_test=tuple(canonical_spec.get("surfaces", ("draw",))),
    )


def o1_neg_a_case(root_path: str, gallery_module=None) -> CaseSpec:
    return _hardening_case(
        case_id=HARDENING_O1_NEG_A_CASE_ID,
        claim_id="I4.real_vector_facet_one_element_selection.HARDENING.O1A",
        title="one-element faceted selection_vector is applied, never silently discarded",
        claim="a one-element selection_vector filters every facet exactly as the raw selection requests",
        failure_means="dfdraw silently discarded a one-element selection_vector or returned the wrong per-facet population",
        expected_visual="no dedicated page; machine-only one-element vector control",
        purpose="CORRECTNESS", oracle_kind="CORRECTNESS", owner="dfdraw",
        canonical_spec={"expr": "dcar_tpc_vertex:tgl", "type": "profile",
                        "facet_by": "side_type", "selection_vector": ["early"],
                        "bins": HARDENING_PROFILE_BINS, "range": list(HARDENING_PROFILE_RANGE)},
        observables=(Observable("facet_selected_rows", "INDEPENDENT", "ARRAY",
                                "raw per-facet selected row counts", comparator="exact"),),
        root_path=root_path, gallery_module=gallery_module,
        setup_contract="reuse the shared EAGER 20% ADF and compare public per-facet n with raw boolean masks",
        preconditions=("side_type facets 0 and 1 are populated", "the early time branch is non-empty"),
        negative_control="selection omission must produce a count mismatch",
    )


def o1_neg_b_case(root_path: str, gallery_module=None) -> CaseSpec:
    return _hardening_case(
        case_id=HARDENING_O1_NEG_B_CASE_ID,
        claim_id="I4.real_hist_vector_facet_selection.HARDENING.O1B",
        title="supported faceted histogram selection_vector is applied",
        claim="the canonically supported non-profile vector path applies both selection branches inside each facet",
        failure_means="dfdraw silently rendered unselected histogram data, dropped branch identity, or refused a supported path",
        expected_visual="no dedicated page; machine-only non-profile vector control",
        purpose="CORRECTNESS", oracle_kind="CORRECTNESS", owner="dfdraw",
        canonical_spec={"expr": "ncl", "type": "hist", "facet_by": "side_type",
                        "selection_vector": ["early", "late"], "bins": 40,
                        "range": [80.0, 160.0]},
        observables=(Observable("branch_facet_rows", "INDEPENDENT", "ARRAY",
                                "raw branch×facet selected row counts", comparator="exact"),),
        root_path=root_path, gallery_module=gallery_module,
        setup_contract="reuse the shared EAGER 20% ADF; supported hist×selection_vector×facet_by must preserve two branches",
        preconditions=("both time branches and both side facets are populated",),
        negative_control="silent vector discard must be detected as cardinality/count mismatch",
    )


def _vector_faceted_counts(stats: Any, *, n_branches: int) -> list[int]:
    """Extract branch-major facet counts; supports the intended vector-facet shape only."""
    if not isinstance(stats, list) or len(stats) != n_branches:
        raise HarnessError(
            f"expected {n_branches} vector branches, got {type(stats).__name__} "
            f"len={len(stats) if isinstance(stats, list) else 'n/a'}")
    out = []
    for branch, branch_stats in enumerate(stats):
        per_group = branch_stats.get("per_group") if isinstance(branch_stats, dict) else None
        if not isinstance(per_group, dict):
            raise HarnessError(f"vector branch {branch} has no faceted per_group payload")
        for facet in HARDENING_FACETS:
            cell = per_group.get(str(facet))
            if not isinstance(cell, dict) or "n" not in cell:
                raise HarnessError(f"vector branch {branch} facet {facet} has no n")
            out.append(int(cell["n"]))
    return out


def run_o1_neg_a(case: CaseSpec, root_path: str, *, gallery_module=None,
                   prepared_adf=None, prepared_provenance=None) -> CaseResult:
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time(); res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        adf = prepared_adf
        if adf is None:
            adf, provenance = _a6_4_build_fraction_adf_once(root_path, gallery_module=gallery_module)
        else:
            _a6_4_prepared_fraction_sample_evidence(adf, prepared_provenance, root_path)
            provenance = dict(prepared_provenance or {})
        df = adf.df
        q1 = float(df["time_s"].quantile(1.0 / 3.0))
        tgl = np.asarray(df["tgl"])
        dcar = np.asarray(df["dcar_tpc_vertex"])
        base = ((np.asarray(df["ncl"]) > 60) & (np.abs(dcar) < 10)
                & (np.asarray(df["side_type"]) < 2) & (np.asarray(df["time_s"]) < q1)
                & np.isfinite(tgl) & np.isfinite(dcar)
                & (tgl >= HARDENING_PROFILE_RANGE[0]) & (tgl <= HARDENING_PROFILE_RANGE[1]))
        expected = [int(np.count_nonzero(base & (np.asarray(df["side_type"]) == facet)))
                    for facet in HARDENING_FACETS]
        raw = adf.draw("dcar_tpc_vertex:tgl", selection=f"{HARDENING_BASE_SEL}&(side_type<2)",
                       type="profile", bins=HARDENING_PROFILE_BINS,
                       range=HARDENING_PROFILE_RANGE,
                       selection_vector=[f"time_s<{q1}"], facet_by="side_type",
                       min_entries=1, auto_title=False)
        stats = raw[2]
        # The semantic contract is selection application, not return-container
        # shape.  A one-element vector may legally degrade to the scalar faceted
        # envelope; prove the supplied selection survived by comparing the public
        # per-facet populations with independent raw masks.
        branch_stats = stats[0] if isinstance(stats, list) and len(stats) == 1 else stats
        per_group = branch_stats.get("per_group") if isinstance(branch_stats, dict) else None
        if not isinstance(per_group, dict):
            raise HarnessError(
                f"one-element vector result has no faceted per_group payload: {type(stats).__name__}")
        observed = []
        for facet in HARDENING_FACETS:
            cell = per_group.get(str(facet))
            if not isinstance(cell, dict) or "n" not in cell:
                raise HarnessError(f"one-element vector facet {facet} has no public n")
            observed.append(int(cell["n"]))
        if observed != expected:
            res.observed["ownership_ladder"] = _o1_neg_a_ownership_ladder(
                adf, q1=q1, expected=expected, adf_observed=observed)
            raise HarnessError(f"O1-neg-A selected counts mismatch: expected={expected}, observed={observed}")
        res.observable_contract.append(_contract(case.observables[0]))
        res.observed["facet_selected_rows"] = observed
        cmp = compare_observable(case.observables[0], expected, observed)
        res.comparisons.append(comparison_evidence(case.observables[0], cmp,
                                                   reference_label="raw masks", candidate_label="public draw"))
        res.executed_comparisons = 1
        res.observed["realdata_provenance"] = provenance
        res.observed["ownership_ladder"] = {
            "L0": "raw boolean masks",
            "L3": "adf.draw one-element selection_vector per-facet counts",
            "first_disagreement_layer": "NONE",
            "derived_owner": "NONE",
        }
        res.status = PASS; res.detail = ""
        return res
    except Exception as exc:
        res.status = FAIL; res.detail = f"O1_NEG_A selection-vector discard control FAIL: {exc}"
        res.exception = traceback.format_exc(limit=6); return res
    finally:
        _close(); res.wall_time_s = round(time.time() - t0, 4)


def run_o1_neg_b(case: CaseSpec, root_path: str, *, gallery_module=None,
                   prepared_adf=None, prepared_provenance=None) -> CaseResult:
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time(); res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        adf = prepared_adf
        if adf is None:
            adf, provenance = _a6_4_build_fraction_adf_once(root_path, gallery_module=gallery_module)
        else:
            _a6_4_prepared_fraction_sample_evidence(adf, prepared_provenance, root_path)
            provenance = dict(prepared_provenance or {})
        df = adf.df; t_mid = float(df["time_s"].median())
        ncl = np.asarray(df["ncl"], dtype=float)
        side = np.asarray(df["side_type"])
        ts = np.asarray(df["time_s"], dtype=float)
        base = (ncl > 60) & (side < 2) & np.isfinite(ncl) & np.isfinite(ts)
        branch_masks = (ts < t_mid, ts >= t_mid)
        edges = np.linspace(80.0, 160.0, 41)
        expected_cells = {}
        for branch_index, branch_mask in enumerate(branch_masks):
            for facet in HARDENING_FACETS:
                mask = base & branch_mask & (side == facet)
                expected_cells[(branch_index, facet)] = np.histogram(
                    ncl[mask], bins=edges)[0].astype(float)
        raw = adf.draw("ncl", selection="(ncl>60)&(side_type<2)", type="hist", bins=40,
                       range=(80.0, 160.0),
                       selection_vector=[f"time_s<{t_mid}", f"time_s>={t_mid}"],
                       facet_by="side_type", auto_title=False)
        axes = np.asarray(raw[1], dtype=object).reshape(-1)
        observed_cells = {}
        try:
            if len(axes) != len(HARDENING_FACETS):
                raise HarnessError(f"expected {len(HARDENING_FACETS)} histogram facets, got {len(axes)}")
            for facet, ax in zip(HARDENING_FACETS, axes):
                rendered = _m1_bar_heights(ax, n_series=2, n_bins=40)
                for branch_index, heights in enumerate(rendered):
                    observed_cells[(branch_index, facet)] = np.asarray(heights, dtype=float)
        except Exception:
            res.observed["ownership_ladder"] = _o1_neg_b_ownership_ladder(
                adf, t_mid=t_mid, expected_cells=expected_cells,
                adf_observed_cells=observed_cells)
            raise
        mismatches = []
        for key, exp in expected_cells.items():
            got = observed_cells.get(key)
            if got is None or not np.array_equal(exp, got):
                mismatches.append(key)
        if mismatches:
            res.observed["ownership_ladder"] = _o1_neg_b_ownership_ladder(
                adf, t_mid=t_mid, expected_cells=expected_cells,
                adf_observed_cells=observed_cells)
            raise HarnessError(f"O1-neg-B rendered branch×facet histogram mismatch: {mismatches}")
        expected = np.concatenate([expected_cells[(b, f)] for b in range(2) for f in HARDENING_FACETS]).tolist()
        observed = np.concatenate([observed_cells[(b, f)] for b in range(2) for f in HARDENING_FACETS]).tolist()
        res.observable_contract.append(_contract(case.observables[0]))
        res.observed["branch_facet_rows"] = observed
        cmp = compare_observable(case.observables[0], expected, observed)
        res.comparisons.append(comparison_evidence(case.observables[0], cmp,
                                                   reference_label="raw np.histogram branch×facet bins", candidate_label="rendered public histogram bars"))
        res.executed_comparisons = 1
        res.observed["realdata_provenance"] = provenance
        res.observed["ownership_ladder"] = {
            "L0": "raw np.histogram branch×facet bins",
            "L3": "adf.draw rendered histogram branch×facet bins",
            "first_disagreement_layer": "NONE",
            "derived_owner": "NONE",
        }
        res.status = PASS; res.detail = ""
        return res
    except Exception as exc:
        res.status = FAIL; res.detail = f"O1_NEG_B non-profile vector control FAIL: {exc}"
        res.exception = traceback.format_exc(limit=6); return res
    finally:
        _close(); res.wall_time_s = round(time.time() - t0, 4)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_76 pre-cleaning oracle addendum — C1
# ORACLE-02 positive acceptance: hist × selection_vector × facet_by
# ─────────────────────────────────────────────────────────────────────────────

PRECLEAN_C1_CASE_ID = "PRECLEAN-C1-HIST-VECTOR-FACET-ACCEPTANCE-EAGER-20PCT-01"
PRECLEAN_C1_GALLERY_FUNCTION = "fig55_preclean_hist_vector_facet_acceptance"
PRECLEAN_C1_FACETS = (0, 1, 2)
PRECLEAN_C1_BINS = 40
PRECLEAN_C1_RANGE = (80.0, 160.0)


def preclean_c1_case(root_path: str, gallery_module=None) -> CaseSpec:
    """Reviewable positive acceptance oracle for the repaired ORACLE-02 path."""
    case = _hardening_case(
        case_id=PRECLEAN_C1_CASE_ID,
        claim_id="I4.preclean.hist_vector_facet_acceptance.C1",
        title="hist selection_vector×facet preserves complete 3×2 semantic product",
        claim=("supported hist×selection_vector×facet_by renders every branch in every facet "
               "and agrees with independent raw histogram truth"),
        failure_means=("the repaired non-profile vector/facet route dropped, duplicated, swapped, "
                       "or numerically changed a branch/facet histogram, or the non-faceted "
                       "positive control regressed"),
        expected_visual=("three side_type facet panels, each containing early and late histograms, "
                         "plus a same-page non-faceted early/late positive control"),
        purpose="CORRECTNESS", oracle_kind="CORRECTNESS", owner="dfdraw",
        canonical_spec={
            "gallery_function": PRECLEAN_C1_GALLERY_FUNCTION,
            "expr": "ncl", "type": "hist",
            "selection": "(ncl>60)&(side_type<3)",
            "selection_vector": ["early", "late"],
            "vector_compose": "outer", "facet_by": "side_type",
            "facets": list(PRECLEAN_C1_FACETS),
            "bins": PRECLEAN_C1_BINS, "range": list(PRECLEAN_C1_RANGE),
            "public_query": (
                "adf.draw('ncl', type='hist', selection='(ncl>60)&(side_type<3)', "
                "selection_vector=['time_s<median(time_s)','time_s>=median(time_s)'], "
                "vector_compose='outer', facet_by='side_type', bins=40, range=(80,160))"
            ),
        },
        observables=(
            Observable("facet_identities", "INDEPENDENT", "ARRAY",
                       "raw side_type facet identities", comparator="exact"),
            Observable("faceted_bin_counts", "INDEPENDENT", "ARRAY",
                       "raw np.histogram branch×facet bins", comparator="exact"),
            Observable("control_bin_counts", "INDEPENDENT", "ARRAY",
                       "raw np.histogram non-faceted branch bins", comparator="exact"),
            Observable("faceted_inrange_rows", "INDEPENDENT", "ARRAY",
                       "raw in-range branch×facet row totals", comparator="exact"),
            Observable("control_inrange_rows", "INDEPENDENT", "ARRAY",
                       "raw in-range branch row totals", comparator="exact"),
        ),
        root_path=root_path, gallery_module=gallery_module,
        gallery_function=PRECLEAN_C1_GALLERY_FUNCTION,
        setup_contract=("reuse the shared EAGER 20% ADF and the exact gallery C1 request; "
                        "three facets and two selection branches must be populated"),
        preconditions=("side_type values 0, 1 and 2 are populated", "both time branches are populated"),
        negative_control=("same-page no-facet early/late histogram must remain correct while the "
                          "3-facet route is exercised"),
    )
    case.figure_contract = FigureContract(
        expected_panels="three side_type facet panels plus one non-faceted positive-control inset",
        panel_roles="facets side_type=0/1/2; inset is identical early/late request without facet_by",
        expected_traces="early and late histogram branches in every facet and in the control",
        expected_group_count="3 facets × 2 branches; control has 2 branches",
        primary_comparison=("raw NumPy branch×facet histogram bins and in-range row totals -> "
                            "rendered public histogram artists"),
        residual_definition="rendered bin height - independent np.histogram bin count",
        accepted_envelope="facet identities exact; all bin counts and in-range row totals exact",
        case_ids=(PRECLEAN_C1_CASE_ID,), proof_kind="CORRECTNESS",
    )
    return case


def _preclean_c1_axes_by_facet(axes: Any) -> dict[int, Any]:
    """Map public facet axes to side_type identities; never trust subplot order."""
    out = {}
    for ax in np.asarray(axes, dtype=object).reshape(-1):
        if ax is None or not getattr(ax, "get_visible", lambda: True)():
            continue
        title = str(getattr(ax, "get_title", lambda: "")())
        match = re.search(r"side_type=([^\s]+)", title)
        if match is None:
            continue
        try:
            facet = int(float(match.group(1)))
        except (TypeError, ValueError):
            continue
        if facet in out:
            raise HarnessError(f"C1 duplicate rendered facet identity {facet}")
        out[facet] = ax
    return out


def _preclean_c1_expected(df: Any, *, t_mid: float) -> dict:
    ncl = np.asarray(df["ncl"], dtype=float)
    side = np.asarray(df["side_type"])
    ts = np.asarray(df["time_s"], dtype=float)
    finite = np.isfinite(ncl) & np.isfinite(ts)
    edges = np.linspace(PRECLEAN_C1_RANGE[0], PRECLEAN_C1_RANGE[1], PRECLEAN_C1_BINS + 1)
    branch_masks = (ts < t_mid, ts >= t_mid)

    faceted_cells = {}
    faceted_rows = []
    for branch_index, branch_mask in enumerate(branch_masks):
        for facet in PRECLEAN_C1_FACETS:
            mask = (ncl > 60) & (side < 3) & finite & branch_mask & (side == facet)
            hist = np.histogram(ncl[mask], bins=edges)[0].astype(float)
            faceted_cells[(branch_index, facet)] = hist
            faceted_rows.append(int(np.sum(hist)))

    control_cells = {}
    control_rows = []
    for branch_index, branch_mask in enumerate(branch_masks):
        mask = (ncl > 60) & (side < 3) & finite & branch_mask
        hist = np.histogram(ncl[mask], bins=edges)[0].astype(float)
        control_cells[branch_index] = hist
        control_rows.append(int(np.sum(hist)))

    return {
        "facet_identities": list(PRECLEAN_C1_FACETS),
        "faceted_cells": faceted_cells,
        "control_cells": control_cells,
        "faceted_inrange_rows": faceted_rows,
        "control_inrange_rows": control_rows,
    }


def _preclean_c1_direct_dfdraw(adf: Any, *, t_mid: float, expected: dict) -> dict:
    """L2 direct-dfdraw attribution for C1; reuses the established direct owner."""
    owner = _DirectDFDrawOwner(adf.df)
    raw = owner.draw(
        "ncl", type="hist", bins=PRECLEAN_C1_BINS, range=PRECLEAN_C1_RANGE,
        selection="(ncl>60)&(side_type<3)",
        selection_vector=[f"time_s<{t_mid}", f"time_s>={t_mid}"],
        vector_compose="outer", facet_by="side_type", auto_title=False)
    try:
        by_facet = _preclean_c1_axes_by_facet(raw[1])
        if set(by_facet) != set(PRECLEAN_C1_FACETS):
            raise HarnessError(f"C1 direct facet identities {sorted(by_facet)}")
        cells = {}
        for facet in PRECLEAN_C1_FACETS:
            rendered = _m1_bar_heights(by_facet[facet], n_series=2, n_bins=PRECLEAN_C1_BINS)
            for branch_index, heights in enumerate(rendered):
                cells[(branch_index, facet)] = np.asarray(heights, dtype=float)
        ok = set(cells) == set(expected["faceted_cells"]) and all(
            np.array_equal(cells[k], expected["faceted_cells"][k]) for k in cells)
        return {"ok": bool(ok), "complete_cells": sorted(str(k) for k in cells)}
    finally:
        try:
            plt.close(raw[0])
        except Exception:
            pass


def run_preclean_c1(case: CaseSpec, root_path: str, *, gallery_module=None,
                    prepared_adf=None, prepared_provenance=None) -> CaseResult:
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time(); res = CaseResult(case_id=case.case_id, status=SKIP)
    adf = prepared_adf
    try:
        if adf is None:
            raise HarnessError("C1 requires shared FAST prepared_adf")
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        provenance = _a6_4_prepared_fraction_sample_evidence(
            adf, prepared_provenance, root_path)
        raw = getattr(gallery, PRECLEAN_C1_GALLERY_FUNCTION)(adf)
        fig, axes_payload, meta = raw
        t_mid = float(meta["t_mid"])
        expected_model = _preclean_c1_expected(adf.df, t_mid=t_mid)

        facet_axes = axes_payload.get("facets") if isinstance(axes_payload, dict) else None
        control_ax = axes_payload.get("control") if isinstance(axes_payload, dict) else None
        by_facet = _preclean_c1_axes_by_facet(facet_axes)
        if set(by_facet) != set(PRECLEAN_C1_FACETS):
            raise HarnessError(
                f"C1 rendered facet identities mismatch: expected={list(PRECLEAN_C1_FACETS)} "
                f"observed={sorted(by_facet)}")
        if control_ax is None:
            raise HarnessError("C1 gallery exposes no non-faceted positive-control axis")

        observed_cells = {}
        observed_rows = []
        for branch_index in range(2):
            for facet in PRECLEAN_C1_FACETS:
                rendered = _m1_bar_heights(
                    by_facet[facet], n_series=2, n_bins=PRECLEAN_C1_BINS)
                heights = np.asarray(rendered[branch_index], dtype=float)
                observed_cells[(branch_index, facet)] = heights
                observed_rows.append(int(np.sum(heights)))

        control_rendered = _m1_bar_heights(
            control_ax, n_series=2, n_bins=PRECLEAN_C1_BINS)
        control_cells = {i: np.asarray(h, dtype=float) for i, h in enumerate(control_rendered)}
        control_rows = [int(np.sum(control_cells[i])) for i in range(2)]

        expected = {
            "facet_identities": expected_model["facet_identities"],
            "faceted_bin_counts": np.concatenate([
                expected_model["faceted_cells"][(b, f)]
                for b in range(2) for f in PRECLEAN_C1_FACETS]).tolist(),
            "control_bin_counts": np.concatenate([
                expected_model["control_cells"][b] for b in range(2)]).tolist(),
            "faceted_inrange_rows": expected_model["faceted_inrange_rows"],
            "control_inrange_rows": expected_model["control_inrange_rows"],
        }
        observed = {
            "facet_identities": sorted(by_facet),
            "faceted_bin_counts": np.concatenate([
                observed_cells[(b, f)] for b in range(2) for f in PRECLEAN_C1_FACETS]).tolist(),
            "control_bin_counts": np.concatenate([
                control_cells[b] for b in range(2)]).tolist(),
            "faceted_inrange_rows": observed_rows,
            "control_inrange_rows": control_rows,
        }

        first_failure = None
        for obs in case.observables:
            res.observable_contract.append(_contract(obs))
            cmp = compare_observable(obs, expected[obs.name], observed[obs.name])
            res.comparisons.append(comparison_evidence(
                obs, cmp, reference_label="raw NumPy/pandas C1 truth",
                candidate_label="public C1 rendered histograms"))
            if not cmp.ok and first_failure is None:
                first_failure = f"{obs.name}: {cmp.detail}"
        res.executed_comparisons = len(case.observables)
        res.observed.update({
            "realdata_provenance": dict(prepared_provenance or provenance),
            "facet_identities": observed["facet_identities"],
            "branch_facet_cardinality": len(observed_cells),
            "positive_control_branch_count": len(control_cells),
        })
        if first_failure is not None:
            direct = _preclean_c1_direct_dfdraw(adf, t_mid=t_mid, expected=expected_model)
            res.observed["ownership_ladder"] = {
                "L0": "raw NumPy branch×facet histograms",
                "L2": "direct DFDraw branch×facet histograms",
                "L2_matches_truth": bool(direct.get("ok")),
                "L2_complete_cells": direct.get("complete_cells", []),
                "L3": "ADF public C1 gallery request",
                "first_disagreement_layer": "L3" if direct.get("ok") else "L2",
                "contract_reference_status": "VERIFIED",
                "owner_status": "ADF" if direct.get("ok") else "DFDRAW",
                "derived_owner": "ADF_DRAW_BRIDGE" if direct.get("ok") else "dfdraw",
            }
            raise HarnessError(first_failure)

        res.observed["ownership_ladder"] = {
            "L0": "raw NumPy branch×facet histograms",
            "L2": "not required: public correctness oracle is green",
            "L3": "ADF public C1 gallery request matches truth",
            "first_disagreement_layer": "NONE",
            "contract_reference_status": "VERIFIED",
            "owner_status": "NONE",
            "derived_owner": "NONE",
        }
        res.status = PASS; res.detail = ""
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"PRECLEAN_C1 HIST_VECTOR_FACET FAIL: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        res.wall_time_s = round(time.time() - t0, 4)
        if adf is not None:
            _record_stage_a_machine_status(adf, res)
        _close()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_76 pre-cleaning oracle addendum — C2
# draw_figures short-form + per-figure effective defaults + independent truth
# ─────────────────────────────────────────────────────────────────────────────

PRECLEAN_C2_CASE_ID = "PRECLEAN-C2-DRAW-FIGURES-EFFECTIVE-DEFAULTS-EAGER-20PCT-01"
PRECLEAN_C2_GALLERY_FUNCTION = "fig56_preclean_draw_figures_effective_defaults"
PRECLEAN_C2_NAME = "preclean_c2_effective_defaults"
PRECLEAN_C2_SELECTIONS = (
    "O4Surface.selector==0",
    "O4Surface.selector==1",
)


def preclean_c2_case(root_path: str, gallery_module=None) -> CaseSpec:
    """Independent correctness oracle for the repaired draw_figures public form."""
    case = _hardening_case(
        case_id=PRECLEAN_C2_CASE_ID,
        claim_id="I4.preclean.draw_figures_effective_defaults.C2",
        title="draw_figures short-form figure defaults preserve qualified delta truth",
        claim=("draw_figures() short-form plot strings use the same effective-default/vector path "
               "as dictionary plots, preserve caller-owned specs, and reproduce independent "
               "injected-truth profile arithmetic"),
        failure_means=("effective-default precedence, short-form normalization, qualified vector "
                       "preparation, vector_compose inference, caller-copy semantics, or final "
                       "draw_figures rendering changed"),
        expected_visual=("one draw_figures profile panel showing the two qualified side branches "
                         "and their normalized delta over sector"),
        purpose="CORRECTNESS", oracle_kind="CORRECTNESS", owner="ADF",
        canonical_spec={
            "gallery_function": PRECLEAN_C2_GALLERY_FUNCTION,
            "surface": "draw_figures",
            "surfaces": ["draw_figures"],
            "figure_name": PRECLEAN_C2_NAME,
            "short_form_plot": "known_delta:sector",
            "top_defaults": {"type": "profile", "bins": 18, "selection": "ncl>0"},
            "figure_defaults": {
                "type": "profile",
                "bins": INJECTED_TRUTH_BINS,
                "range": list(INJECTED_TRUTH_RANGE),
                "selection": f"{HARDENING_BASE_SEL}&(side_type<2)",
                "selection_vector": list(PRECLEAN_C2_SELECTIONS),
                "normalize": "delta",
            },
            "public_query": (
                "adf.draw_figures([{'name':'preclean_c2_effective_defaults', "
                "'defaults':{'type':'profile','bins':36,'range':[-0.5,35.5],"
                "'selection':BASE_SEL+'&(side_type<2)',"
                "'selection_vector':['O4Surface.selector==0','O4Surface.selector==1'],"
                "'normalize':'delta'}, 'plots':['known_delta:sector']}], "
                "defaults={'type':'profile','bins':18,'selection':'ncl>0'})"
            ),
            "injected_truth": (
                "known_delta = 0.08*sin(2*pi*sector/36) + 0.03*tgl + 0.015*tgl^2; "
                "qualified selector maps exactly to side_type"
            ),
        },
        observables=(
            Observable("x_center", "INDEPENDENT", "ARRAY", "raw sector bin centers",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same explicit 36-bin sector geometry"),
            Observable("signal_count", "INDEPENDENT", "ARRAY", "raw side_type==0 bin counts",
                       comparator="exact"),
            Observable("reference_count", "INDEPENDENT", "ARRAY", "raw side_type==1 bin counts",
                       comparator="exact"),
            Observable("signal_central", "INDEPENDENT", "ARRAY", "raw side_type==0 known-delta means",
                       comparator="close", atol=1e-12, rtol=1e-12,
                       rationale="same float64 arithmetic mean on identical selected rows"),
            Observable("reference_central", "INDEPENDENT", "ARRAY", "raw side_type==1 known-delta means",
                       comparator="close", atol=1e-12, rtol=1e-12,
                       rationale="same float64 arithmetic mean on identical selected rows"),
            Observable("delta_values", "INDEPENDENT", "ARRAY", "raw signal-reference delta",
                       comparator="close", atol=1e-12, rtol=1e-12,
                       rationale="delta is direct subtraction of independent branch means"),
            Observable("valid_bin_mask", "INDEPENDENT", "ARRAY", "raw bins populated in both branches",
                       comparator="exact"),
            Observable("caller_unchanged", "INDEPENDENT", "FLAT", "caller containers unchanged",
                       comparator="exact"),
            Observable("short_form_preserved", "INDEPENDENT", "FLAT", "caller plot remains a string",
                       comparator="exact"),
        ),
        root_path=root_path, gallery_module=gallery_module,
        gallery_function=PRECLEAN_C2_GALLERY_FUNCTION,
        setup_contract=("reuse shared injected truth + O4Surface qualified selector; execute one "
                        "short-form draw_figures request with conflicting top/figure defaults"),
        preconditions=("known_delta injected truth is materialized", "side_type 0 and 1 are populated",),
        negative_control=("top-level bins=18 and selection='ncl>0' must not override figure bins=36 "
                          "and BASE_SEL; caller specs must not gain vector_compose or dict normalization"),
    )
    case.figure_contract = FigureContract(
        expected_panels="one draw_figures profile panel with normalized delta evidence",
        panel_roles="single short-form plot resolved entirely through figure defaults",
        expected_traces="qualified signal/reference branch profiles and normalized delta",
        expected_group_count="two qualified selection_vector branches",
        primary_comparison=("raw NumPy/pandas per-sector side0/side1 means and counts -> "
                            "draw_figures normalize_data payload"),
        residual_definition="expected delta = side_type==0 mean - side_type==1 mean",
        accepted_envelope=("36-bin geometry/count/mask identity exact; floating means/delta within "
                           "declared tolerance; caller containers exactly unchanged"),
        case_ids=(PRECLEAN_C2_CASE_ID,), proof_kind="CORRECTNESS",
    )
    return case


def _preclean_c2_expected(adf: Any, gallery: Any) -> dict:
    getattr(gallery, "_ensure_injected_truth")(adf)
    _ensure_o4_surface_subframe(adf)
    model = _it_semantic_model(adf)
    side = np.asarray(model["side"])
    signal = _it_profile(model["sector"], model["delta"], model["base"] & (side == 0))
    reference = _it_profile(model["sector"], model["delta"], model["base"] & (side == 1))
    valid = (signal["count"] > 0) & (reference["count"] > 0)
    delta = np.asarray(signal["y_mean"] - reference["y_mean"], dtype=float)
    delta[~valid] = np.nan
    return {
        "x_center": np.asarray(signal["x_center"], dtype=float),
        "signal_count": np.asarray(signal["count"], dtype=int),
        "reference_count": np.asarray(reference["count"], dtype=int),
        "signal_central": np.asarray(signal["y_mean"], dtype=float),
        "reference_central": np.asarray(reference["y_mean"], dtype=float),
        "delta_values": delta,
        "valid_bin_mask": np.asarray(valid, dtype=bool),
        "caller_unchanged": True,
        "short_form_preserved": True,
    }


def _preclean_c2_normalize_payload(stats: Any) -> dict:
    if not isinstance(stats, dict):
        raise HarnessError(f"C2 expected dict stats, got {type(stats).__name__}")
    if stats.get("normalize_mode") != "delta":
        raise HarnessError(f"C2 expected normalize_mode='delta', got {stats.get('normalize_mode')!r}")
    nd = stats.get("normalize_data")
    if nd is None:
        raise HarnessError("C2 draw_figures stats expose no normalize_data")

    def column(name: str):
        if isinstance(nd, dict):
            if name not in nd:
                raise HarnessError(f"C2 normalize_data missing {name!r}")
            return np.asarray(nd[name])
        columns = getattr(nd, "columns", ())
        if name not in columns:
            raise HarnessError(f"C2 normalize_data frame missing {name!r}")
        return np.asarray(nd[name])

    mask_undefined = np.asarray(column("mask_undefined"), dtype=bool)
    return {
        "x_center": np.asarray(column("x_center"), dtype=float),
        "signal_count": np.asarray(column("signal_count"), dtype=int),
        "reference_count": np.asarray(column("reference_count"), dtype=int),
        "signal_central": np.asarray(column("signal_central"), dtype=float),
        "reference_central": np.asarray(column("reference_central"), dtype=float),
        "delta_values": np.asarray(column("value"), dtype=float),
        "valid_bin_mask": ~mask_undefined,
    }


def _preclean_c2_direct_dfdraw(adf: Any, expected: dict) -> dict:
    """L2 diagnostic using materialized columns and equivalent unqualified selectors."""
    fig = None
    try:
        owner = _DirectDFDrawOwner(adf.df)
        raw = owner.draw(
            "known_delta:sector", type="profile",
            bins=INJECTED_TRUTH_BINS, range=INJECTED_TRUTH_RANGE,
            selection=f"{HARDENING_BASE_SEL}&(side_type<2)",
            selection_vector=["side_type==0", "side_type==1"],
            normalize="delta", return_data=True, auto_title=False)
        fig = raw[0]
        observed = _preclean_c2_normalize_payload(raw[2])
        names = ("x_center", "signal_count", "reference_count", "signal_central",
                 "reference_central", "delta_values", "valid_bin_mask")
        ok = True
        for name in names:
            exp = expected[name]
            got = observed[name]
            if name in {"signal_count", "reference_count", "valid_bin_mask"}:
                ok &= bool(np.array_equal(np.asarray(exp), np.asarray(got)))
            else:
                ok &= bool(np.allclose(np.asarray(exp, float), np.asarray(got, float),
                                       atol=1e-12, rtol=1e-12, equal_nan=True))
        return {"ok": bool(ok)}
    finally:
        try:
            if fig is not None:
                plt.close(fig)
        except Exception:
            pass


def run_preclean_c2(case: CaseSpec, root_path: str, *, gallery_module=None,
                    prepared_adf=None, prepared_provenance=None) -> CaseResult:
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time(); res = CaseResult(case_id=case.case_id, status=SKIP)
    adf = prepared_adf
    try:
        if adf is None:
            raise HarnessError("C2 requires shared FAST prepared_adf")
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        provenance = _a6_4_prepared_fraction_sample_evidence(
            adf, prepared_provenance, root_path)
        expected = _preclean_c2_expected(adf, gallery)
        raw = getattr(gallery, PRECLEAN_C2_GALLERY_FUNCTION)(adf)
        if not isinstance(raw, tuple) or len(raw) < 3 or not isinstance(raw[2], dict):
            raise HarnessError("C2 gallery result does not expose metadata")
        meta = raw[2]
        observed = _preclean_c2_normalize_payload(meta.get("stats"))
        observed.update({
            "caller_unchanged": bool(meta.get("caller_unchanged")),
            "short_form_preserved": bool(meta.get("short_form_preserved")),
        })

        first_failure = None
        for obs in case.observables:
            res.observable_contract.append(_contract(obs))
            cmp = compare_observable(obs, expected[obs.name], observed[obs.name])
            res.comparisons.append(comparison_evidence(
                obs, cmp, reference_label="raw NumPy/pandas C2 truth",
                candidate_label="draw_figures short-form/effective-default result"))
            if not cmp.ok and first_failure is None:
                first_failure = f"{obs.name}: {cmp.detail}"
        res.executed_comparisons = len(case.observables)
        res.observed.update({
            "realdata_provenance": dict(prepared_provenance or provenance),
            "caller_unchanged": observed["caller_unchanged"],
            "short_form_preserved": observed["short_form_preserved"],
            "caller_specs_before": meta.get("specs_before"),
            "caller_specs_after": meta.get("specs_after"),
            "caller_defaults_before": meta.get("defaults_before"),
            "caller_defaults_after": meta.get("defaults_after"),
        })
        if first_failure is not None:
            direct = _preclean_c2_direct_dfdraw(adf, expected)
            res.observed["ownership_ladder"] = {
                "L0": "raw NumPy/pandas injected truth + declared qualified selector mapping",
                "L1": "ADF materialized injected truth / qualified-subframe preparation",
                "L2": "direct DFDraw equivalent materialized-column request",
                "L2_matches_truth": bool(direct.get("ok")),
                "L3": "ADF draw_figures short-form/effective-default route",
                "first_disagreement_layer": "L3" if direct.get("ok") else "L2",
                "contract_reference_status": "VERIFIED",
                "owner_status": "ADF" if direct.get("ok") else "DFDRAW",
                "derived_owner": "ADF_DRAW_FIGURES" if direct.get("ok") else "dfdraw",
            }
            raise HarnessError(first_failure)

        res.observed["ownership_ladder"] = {
            "L0": "raw NumPy/pandas injected truth + declared qualified selector mapping",
            "L1": "ADF materialized injected truth / qualified-subframe preparation",
            "L2": "not required: public correctness oracle is green",
            "L3": "draw_figures short-form/effective-default result matches truth",
            "first_disagreement_layer": "NONE",
            "contract_reference_status": "VERIFIED",
            "owner_status": "NONE",
            "derived_owner": "NONE",
        }
        res.status = PASS; res.detail = ""
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"PRECLEAN_C2 DRAW_FIGURES_DEFAULTS FAIL: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        res.wall_time_s = round(time.time() - t0, 4)
        if adf is not None:
            _record_stage_a_machine_status(adf, res)
        _close()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_76 pre-cleaning oracle addendum — C3
# non-delta y-vector normalize=ratio with predeclared exact scaled truth
# ─────────────────────────────────────────────────────────────────────────────

PRECLEAN_C3_CASE_ID = "PRECLEAN-C3-RATIO-Y-VECTOR-TRUTH-EAGER-20PCT-01"
PRECLEAN_C3_GALLERY_FUNCTION = "fig57_preclean_ratio_y_vector_truth"
PRECLEAN_C3_BASE_EXPR = "1.0 + 0.02*sector + 0.01*tgl*tgl"
PRECLEAN_C3_SCALE = 2.0
PRECLEAN_C3_EXPECTED_RATIO = 1.0 / PRECLEAN_C3_SCALE


def preclean_c3_case(root_path: str, gallery_module=None) -> CaseSpec:
    """Independent numerical oracle for supported y-vector normalize='ratio'."""
    case = _hardening_case(
        case_id=PRECLEAN_C3_CASE_ID,
        claim_id="I4.preclean.ratio_y_vector_truth.C3",
        title="y-vector normalize=ratio recovers predeclared exact 0.5 truth",
        claim=("a supported two-branch y-vector profile normalized with ratio preserves branch "
               "order and returns the independently known signal/reference ratio"),
        failure_means=("y-vector normalization routing was ignored, branch order changed, "
                       "profile reduction changed, undefined-bin masking drifted, or ratio "
                       "arithmetic differs from independent raw truth"),
        expected_visual=("two strictly-positive source profiles with the second exactly 2× the "
                         "first, plus a normalized ratio trace at 0.5 in every populated bin"),
        purpose="CORRECTNESS", oracle_kind="CORRECTNESS", owner="dfdraw",
        canonical_spec={
            "gallery_function": PRECLEAN_C3_GALLERY_FUNCTION,
            "surface": "draw",
            "slot": "y_vector",
            "expr": "[preclean_ratio_base,preclean_ratio_scaled]:sector",
            "type": "profile",
            "selection": HARDENING_BASE_SEL,
            "bins": INJECTED_TRUTH_BINS,
            "range": list(INJECTED_TRUTH_RANGE),
            "normalize": "ratio",
            "ratio_definition": "signal/reference = first y-vector branch / second y-vector branch",
            "base_formula": PRECLEAN_C3_BASE_EXPR,
            "scale": PRECLEAN_C3_SCALE,
            "expected_ratio": PRECLEAN_C3_EXPECTED_RATIO,
            "public_query": (
                "adf.draw('[preclean_ratio_base,preclean_ratio_scaled]:sector', "
                "type='profile', selection=BASE_SEL, bins=36, range=(-0.5,35.5), "
                "normalize='ratio', return_data=True)"
            ),
        },
        observables=(
            Observable("x_center", "INDEPENDENT", "ARRAY", "raw sector bin centers",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="same explicit 36-bin sector geometry"),
            Observable("signal_count", "INDEPENDENT", "ARRAY", "raw first-branch bin counts",
                       comparator="exact"),
            Observable("reference_count", "INDEPENDENT", "ARRAY", "raw second-branch bin counts",
                       comparator="exact"),
            Observable("signal_central", "INDEPENDENT", "ARRAY", "raw base-formula bin means",
                       comparator="close", atol=1e-12, rtol=1e-12,
                       rationale="same float64 arithmetic mean on identical selected rows"),
            Observable("reference_central", "INDEPENDENT", "ARRAY", "raw 2×base bin means",
                       comparator="close", atol=2e-12, rtol=1e-12,
                       rationale="same float64 arithmetic mean on exact scaled rows"),
            Observable("ratio_values", "INDEPENDENT", "ARRAY", "raw signal/reference ratio",
                       comparator="close", atol=2e-13, rtol=2e-13,
                       rationale="ratio of independently reduced strictly-positive scaled branches"),
            Observable("valid_bin_mask", "INDEPENDENT", "ARRAY", "raw jointly populated/nonzero bins",
                       comparator="exact"),
        ),
        root_path=root_path, gallery_module=gallery_module,
        gallery_function=PRECLEAN_C3_GALLERY_FUNCTION,
        setup_contract=("construct a strictly-positive base formula from real sector/tgl rows, "
                        "materialize an exact 2× y-vector companion, and normalize first/second"),
        preconditions=("selected rows populate the sector profile", "reference branch is strictly positive"),
        negative_control=("a swapped y-vector branch order must produce ratio 2.0 rather than 0.5 "
                          "and therefore fail the declared oracle"),
    )
    case.figure_contract = FigureContract(
        expected_panels="one profile normalization page with source overlay and ratio evidence",
        panel_roles="source branches are base and exact 2×base; normalized result is base/scaled",
        expected_traces="two positive source profiles and ratio=0.5 on populated bins",
        expected_group_count="two ordered y-vector branches",
        primary_comparison=("raw NumPy/pandas branch means/counts and predeclared 0.5 ratio -> "
                            "public normalize_data payload"),
        residual_definition="public ratio - 0.5 on independently valid bins",
        accepted_envelope=("bin geometry/count/mask identity exact; source means and ratio within "
                           "declared pre-set floating tolerances"),
        case_ids=(PRECLEAN_C3_CASE_ID,), proof_kind="CORRECTNESS",
    )
    return case


def _preclean_c3_expected(adf: Any) -> dict:
    df = adf.df
    sector = np.asarray(df["sector"], dtype=float)
    tgl = np.asarray(df["tgl"], dtype=float)
    ncl = np.asarray(df["ncl"], dtype=float)
    dcar = np.asarray(df["dcar_tpc_vertex"], dtype=float)
    base_mask = (ncl > 60) & (np.abs(dcar) < 10)
    base_values = 1.0 + 0.02 * sector + 0.01 * tgl * tgl
    scaled_values = PRECLEAN_C3_SCALE * base_values
    # When the product-side aliases have already been materialized, use only
    # their authoritative dtype as part of the contract reference.  Values are
    # still reconstructed independently from raw sector/tgl; no ADF result is
    # reused as expected data.
    if "preclean_ratio_base" in df.columns:
        base_values = _it_cast_expected_to_authoritative_dtype(
            df, "preclean_ratio_base", base_values)
    if "preclean_ratio_scaled" in df.columns:
        scaled_values = _it_cast_expected_to_authoritative_dtype(
            df, "preclean_ratio_scaled", scaled_values)
    signal = _it_profile(sector, base_values, base_mask)
    reference = _it_profile(sector, scaled_values, base_mask)
    valid = ((signal["count"] > 0) & (reference["count"] > 0)
             & np.isfinite(signal["y_mean"]) & np.isfinite(reference["y_mean"])
             & (reference["y_mean"] != 0.0))
    ratio = np.full_like(signal["y_mean"], np.nan, dtype=float)
    ratio[valid] = signal["y_mean"][valid] / reference["y_mean"][valid]
    if np.any(valid) and not np.allclose(
            ratio[valid], PRECLEAN_C3_EXPECTED_RATIO, rtol=2e-13, atol=2e-13):
        raise HarnessError("C3 independent construction does not produce the predeclared 0.5 ratio")
    return {
        "x_center": np.asarray(signal["x_center"], dtype=float),
        "signal_count": np.asarray(signal["count"], dtype=int),
        "reference_count": np.asarray(reference["count"], dtype=int),
        "signal_central": np.asarray(signal["y_mean"], dtype=float),
        "reference_central": np.asarray(reference["y_mean"], dtype=float),
        "ratio_values": ratio,
        "valid_bin_mask": np.asarray(valid, dtype=bool),
    }


def _preclean_c3_normalize_payload(stats: Any) -> dict:
    if not isinstance(stats, dict):
        raise HarnessError(f"C3 expected dict stats, got {type(stats).__name__}")
    if stats.get("normalize_mode") != "ratio":
        raise HarnessError(f"C3 expected normalize_mode='ratio', got {stats.get('normalize_mode')!r}")
    nd = stats.get("normalize_data")
    if nd is None:
        raise HarnessError("C3 y-vector result exposes no normalize_data")

    def column(name: str):
        if isinstance(nd, dict):
            if name not in nd:
                raise HarnessError(f"C3 normalize_data missing {name!r}")
            return np.asarray(nd[name])
        columns = getattr(nd, "columns", ())
        if name not in columns:
            raise HarnessError(f"C3 normalize_data frame missing {name!r}")
        return np.asarray(nd[name])

    mask_undefined = np.asarray(column("mask_undefined"), dtype=bool)
    return {
        "x_center": np.asarray(column("x_center"), dtype=float),
        "signal_count": np.asarray(column("signal_count"), dtype=int),
        "reference_count": np.asarray(column("reference_count"), dtype=int),
        "signal_central": np.asarray(column("signal_central"), dtype=float),
        "reference_central": np.asarray(column("reference_central"), dtype=float),
        "ratio_values": np.asarray(column("value"), dtype=float),
        "valid_bin_mask": ~mask_undefined,
    }


def _preclean_c3_l1_materialization(adf: Any) -> dict:
    df = adf.df
    sector = np.asarray(df["sector"], dtype=float)
    tgl = np.asarray(df["tgl"], dtype=float)
    expected_base = 1.0 + 0.02 * sector + 0.01 * tgl * tgl
    expected_scaled = PRECLEAN_C3_SCALE * expected_base
    expected_base = _it_cast_expected_to_authoritative_dtype(
        df, "preclean_ratio_base", expected_base)
    expected_scaled = _it_cast_expected_to_authoritative_dtype(
        df, "preclean_ratio_scaled", expected_scaled)
    base = np.asarray(df["preclean_ratio_base"])
    scaled = np.asarray(df["preclean_ratio_scaled"])
    base_ok = bool(np.array_equal(base, expected_base, equal_nan=True))
    scaled_ok = bool(np.array_equal(scaled, expected_scaled, equal_nan=True))
    return {"base_ok": base_ok, "scaled_ok": scaled_ok, "all_ok": bool(base_ok and scaled_ok)}


def _preclean_c3_direct_dfdraw(adf: Any, expected: dict) -> dict:
    fig = None
    try:
        owner = _DirectDFDrawOwner(adf.df)
        raw = owner.draw(
            "[preclean_ratio_base,preclean_ratio_scaled]:sector",
            type="profile", bins=INJECTED_TRUTH_BINS, range=INJECTED_TRUTH_RANGE,
            selection=HARDENING_BASE_SEL, normalize="ratio",
            return_data=True, auto_title=False)
        fig = raw[0]
        observed = _preclean_c3_normalize_payload(raw[2])
        specs = {o.name: o for o in preclean_c3_case("synthetic.root").observables}
        checks = [compare_observable(specs[name], expected[name], observed[name]).ok
                  for name in ("x_center", "signal_count", "reference_count", "signal_central",
                               "reference_central", "ratio_values", "valid_bin_mask")]
        return {"ok": bool(all(checks)), "observed": observed}
    except Exception as exc:
        return {"ok": False, "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        if fig is not None:
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass


def run_preclean_c3(case: CaseSpec, root_path: str, *, gallery_module=None,
                    prepared_adf=None, prepared_provenance=None) -> CaseResult:
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time(); res = CaseResult(case_id=case.case_id, status=SKIP)
    adf = None
    try:
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        adf = prepared_adf
        if adf is None:
            adf, provenance = _a6_4_build_fraction_adf_once(root_path, gallery_module=gallery)
        else:
            _a6_4_prepared_fraction_sample_evidence(adf, prepared_provenance, root_path)
            provenance = dict(prepared_provenance or {})
        ensure_truth = getattr(gallery, "_ensure_preclean_c3_ratio_truth", None)
        if not callable(ensure_truth):
            raise HarnessError("gallery has no _ensure_preclean_c3_ratio_truth owner")
        ensure_truth(adf)
        expected = _preclean_c3_expected(adf)
        raw = getattr(gallery, PRECLEAN_C3_GALLERY_FUNCTION)(adf)
        try:
            observed = _preclean_c3_normalize_payload(raw[2])
        finally:
            try:
                import matplotlib.pyplot as plt
                fig = raw[0]
                if isinstance(fig, (list, tuple)):
                    for item in fig:
                        if item is not None:
                            plt.close(item)
                elif fig is not None:
                    plt.close(fig)
            except Exception:
                pass

        first_failure = None
        for obs in case.observables:
            res.observable_contract.append(_contract(obs))
            cmp = compare_observable(obs, expected[obs.name], observed[obs.name])
            res.comparisons.append(comparison_evidence(
                obs, cmp, reference_label="raw NumPy/pandas exact-scaled C3 truth",
                candidate_label="public y-vector normalize=ratio result"))
            if not cmp.ok and first_failure is None:
                first_failure = f"{obs.name}: {cmp.detail}"
        res.executed_comparisons = len(case.observables)
        l1 = _preclean_c3_l1_materialization(adf)
        res.observed.update({
            "realdata_provenance": dict(prepared_provenance or provenance),
            "expected_ratio": PRECLEAN_C3_EXPECTED_RATIO,
            "L1_materialization": l1,
        })
        if not l1["all_ok"]:
            res.observed["ownership_ladder"] = {
                "L0": "raw formula from sector/tgl",
                "L1": "ADF materialized preclean_ratio_base / preclean_ratio_scaled",
                "first_disagreement_layer": "L1",
                "contract_reference_status": "VERIFIED",
                "owner_status": "ADF",
                "derived_owner": "ADF_MATERIALIZATION",
            }
            raise HarnessError("C3 ADF alias materialization differs from declared raw formula")
        if first_failure is not None:
            direct = _preclean_c3_direct_dfdraw(adf, expected)
            res.observed["ownership_ladder"] = {
                "L0": "raw NumPy/pandas exact-scaled ratio truth",
                "L1": "ADF materialized ratio source columns match raw formula",
                "L2": "direct DFDraw y-vector normalize=ratio on materialized columns",
                "L2_matches_truth": bool(direct.get("ok")),
                "L3": "ADF adf.draw y-vector normalize=ratio",
                "first_disagreement_layer": "L3" if direct.get("ok") else "L2",
                "contract_reference_status": "VERIFIED",
                "owner_status": "ADF" if direct.get("ok") else "DFDRAW",
                "derived_owner": "ADF_DRAW" if direct.get("ok") else "dfdraw",
            }
            raise HarnessError(first_failure)
        res.observed["ownership_ladder"] = {
            "L0": "raw NumPy/pandas exact-scaled ratio truth",
            "L1": "ADF materialized ratio source columns match raw formula",
            "L2": "not required: public correctness oracle is green",
            "L3": "ADF adf.draw y-vector normalize=ratio matches truth",
            "first_disagreement_layer": "NONE",
            "contract_reference_status": "VERIFIED",
            "owner_status": "NONE",
            "derived_owner": "NONE",
        }
        res.status = PASS; res.detail = ""
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"PRECLEAN_C3 RATIO_Y_VECTOR FAIL: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        res.wall_time_s = round(time.time() - t0, 4)
        if adf is not None:
            _record_stage_a_machine_status(adf, res)
        _close()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_76 pre-cleaning oracle addendum — C4
# FULL EAGER↔LAZY same-truth qualified/chained-alias numerical oracle
#
# C4 deliberately reuses the existing M3 scientific fixture instead of adding
# another subframe/oracle framework.  The public request is the normal example
# ``fig54_qualified_subframe_chain_truth``: a two-level alias chain plus the
# qualified ``OracleShift.oracle_offset`` projection, grouped by side_type.
# The only new responsibility here is to execute that same scientific request
# once in FULL EAGER and once in FULL LAZY mode and compare both to the same
# independently reconstructed NumPy/pandas truth.
# ─────────────────────────────────────────────────────────────────────────────

PRECLEAN_C4_CASE_ID = "PRECLEAN-C4-QUALIFIED-CHAIN-EAGER-LAZY-BOTH-FULL-01"
PRECLEAN_C4_REUSED_GALLERY_FUNCTION = "fig54_qualified_subframe_chain_truth"
PRECLEAN_C4_TREE_NAME = A5_3_TREE_NAME


def preclean_c4_case(root_path: str, gallery_module=None) -> CaseSpec:
    """PRIMARY ORACLE: the same qualified injected truth in FULL EAGER/LAZY."""
    return CaseSpec(
        case_id=PRECLEAN_C4_CASE_ID,
        claim_id="I4.preclean.qualified_chain_eager_lazy_truth.C4",
        title="FULL EAGER and LAZY resolve the same qualified chained-alias truth",
        claim=("the existing M3 chained injected-truth expression plus qualified OracleShift "
               "subframe produces the same independently known grouped sector profile in FULL "
               "EAGER and FULL LAZY loading modes"),
        failure_means=("loading-mode dependency discovery/materialization, chained alias evaluation, "
                       "qualified subframe projection, grouping, row population or floating reduction "
                       "changed between EAGER and LAZY or disagrees with independent raw truth"),
        expected_visual=("for side_type 0 and 1, EAGER and LAZY profiles overlay the same independently "
                         "known chained+qualified truth; the lower LAZY-EAGER residual is exactly zero "
                         "on every jointly populated bin"),
        owner_on_failure="ADF",
        purpose="CORRECTNESS",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS",
        loading_mode="BOTH",
        sample_mode="FULL",
        canonical_spec={
            "gallery_function": PRECLEAN_C4_REUSED_GALLERY_FUNCTION,
            "expr": "oracle_chain_with_shift:sector",
            "type": "profile",
            "group_by": "side_type",
            "selection": INJECTED_TRUTH_SIDE_SEL,
            "bins": INJECTED_TRUTH_BINS,
            "range": list(INJECTED_TRUTH_RANGE),
            "tree_name": PRECLEAN_C4_TREE_NAME,
            "sample": None,
            "execution_legs": ["EAGER_FULL", "LAZY_FULL"],
            "public_query": (
                "fig54_qualified_subframe_chain_truth(adf)  # same request in EAGER FULL and LAZY FULL"
            ),
        },
        applicable=bool(root_path),
        applicability_reason="" if root_path else "ROOT input path is empty",
        setup_contract=("reuse the existing M3 injected-truth alias/subframe fixture; build exactly one "
                        "FULL EAGER ADF and one FULL LAZY ADF from the same ROOT identity; reconstruct "
                        "the M3 formula independently in each mode and compare each public result to truth"),
        preconditions=(
            "the trusted time-series ROOT input is readable",
            "build_adf supports sample=None in both eager and lazy modes",
            "fig54_qualified_subframe_chain_truth and _ensure_m3_subframe_chain are available",
            "side_type 0 and 1 are populated on the selected scientific domain",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable("eager_group_values", "INDEPENDENT", "ARRAY", "raw side_type group identities",
                       comparator="exact"),
            Observable("lazy_group_values", "INDEPENDENT", "ARRAY", "raw side_type group identities",
                       comparator="exact"),
            Observable("eager_x_center", "INDEPENDENT", "ARRAY", "explicit sector-bin centers",
                       comparator="exact"),
            Observable("lazy_x_center", "INDEPENDENT", "ARRAY", "explicit sector-bin centers",
                       comparator="exact"),
            Observable("eager_count", "INDEPENDENT", "ARRAY", "raw side/sector selected counts",
                       comparator="exact"),
            Observable("lazy_count", "INDEPENDENT", "ARRAY", "raw side/sector selected counts",
                       comparator="exact"),
            Observable("eager_valid_mask", "INDEPENDENT", "ARRAY", "raw populated finite-bin mask",
                       comparator="exact"),
            Observable("lazy_valid_mask", "INDEPENDENT", "ARRAY", "raw populated finite-bin mask",
                       comparator="exact"),
            Observable("eager_y_mean", "INDEPENDENT", "ARRAY", "raw chained+qualified group means",
                       comparator="close", rtol=1e-8, atol=1e-9,
                       rationale="same M3 independent float64/bin reduction contract"),
            Observable("lazy_y_mean", "INDEPENDENT", "ARRAY", "raw chained+qualified group means",
                       comparator="close", rtol=1e-8, atol=1e-9,
                       rationale="same M3 independent float64/bin reduction contract"),
            Observable("mode_count_delta", "INDEPENDENT", "ARRAY", "LAZY count - EAGER count",
                       comparator="exact"),
            Observable("mode_y_mean_delta", "INDEPENDENT", "ARRAY", "LAZY mean - EAGER mean on valid bins",
                       comparator="exact"),
        ),
        figure_contract=FigureContract(
            expected_panels="two panels in a separate C4 FULL slow-gate PDF",
            panel_roles="top: EAGER/LAZY grouped scientific profiles; bottom: LAZY-EAGER residual",
            expected_traces="two side_type groups for each loading mode plus zero residual traces",
            expected_group_count="two side_type groups in EAGER and two in LAZY",
            primary_comparison=("independent raw chained+qualified M3 truth -> EAGER and -> LAZY; "
                                "then exact EAGER↔LAZY identity on count/mask/bin/result arrays"),
            residual_definition="LAZY public y_mean - EAGER public y_mean on jointly valid bins",
            accepted_envelope=("group/bin/count/mask identities exact; each mode matches independent "
                               "truth within existing M3 tolerance; EAGER↔LAZY public floating result "
                               "must be exactly identical in this first measurement"),
            case_ids=(PRECLEAN_C4_CASE_ID,), proof_kind="CORRECTNESS",
        ),
        non_claims=(
            "C4 does not replace O3 dependency-sparsity/decoy instrumentation",
            "C4 is not part of the routine sampled FAST gallery",
            "C4 introduces no sampled-lazy semantics",
        ),
        negative_control="PRECLEAN_C4:LAZY_RESULT_OR_ROW_POPULATION_MUTATION",
        reference_policy="same-process",
    )


def _preclean_c4_grouped_truth_and_public(adf: Any, gallery: Any) -> dict:
    """Run the existing M3 example and return independent/public comparison arrays.

    The expected values are reconstructed from physical source columns and the
    declared injected-truth/subframe formulas.  Public aggregation is used only
    for the candidate arrays.  This keeps the example function readable while
    the detailed machine-oracle bookkeeping remains in the harness.
    """
    ensure = getattr(gallery, "_ensure_m3_subframe_chain", None)
    if not callable(ensure):
        raise HarnessError("C4 gallery has no _ensure_m3_subframe_chain owner")
    ensure(adf)
    model = _it_semantic_model(adf)
    chain = _m3_independent_semantic_values(adf)
    l1 = _m3_materialization_diagnostics(adf, chain)
    membership = _preclean_c4_membership_rows(adf, model, chain)
    source_dtypes = {
        name: str(np.asarray(adf.df[name]).dtype)
        for name in (
            "oracle_row_id", "sector", "side_type", "ncl", "dcar_tpc_vertex",
            "tgl", "oracle_chain_l1", "oracle_chain_l2", "oracle_chain_with_shift"
        )
        if name in adf.df.columns
    }

    raw = getattr(gallery, PRECLEAN_C4_REUSED_GALLERY_FUNCTION)(adf)
    try:
        stats = raw[2]
        groups = _m2_profile_groups(stats)
        group_ids = [int(float(g)) for g, _ in groups]
        if group_ids != [0, 1]:
            raise HarnessError(f"C4 group identity/order mismatch: {group_ids}")

        exp_group = []
        got_group = []
        exp_x = []
        got_x = []
        exp_count = []
        got_count = []
        exp_mean = []
        got_mean = []
        exp_valid = []
        got_valid = []

        for (g, frame), side in zip(groups, (0, 1)):
            ref = _it_profile(
                model["sector"], chain["final"],
                model["side_sel"] & (model["side"] == side),
                accumulator="float64")
            x = np.asarray(frame["x_center"], dtype=float)
            count = np.asarray(frame["count"], dtype=int)
            mean = np.asarray(frame["y_mean"], dtype=float)
            if not (len(x) == len(count) == len(mean) == INJECTED_TRUTH_BINS):
                raise HarnessError(
                    f"C4 side_type={side} profile shape drift: "
                    f"x={len(x)}, count={len(count)}, mean={len(mean)}")
            valid_ref = (np.asarray(ref["count"], dtype=int) > 0) & np.isfinite(ref["y_mean"])
            valid_got = (count > 0) & np.isfinite(mean)

            exp_group.extend([side] * INJECTED_TRUTH_BINS)
            got_group.extend([int(float(g))] * INJECTED_TRUTH_BINS)
            exp_x.extend(np.asarray(ref["x_center"], dtype=float).tolist())
            got_x.extend(x.tolist())
            exp_count.extend(np.asarray(ref["count"], dtype=int).tolist())
            got_count.extend(count.tolist())
            exp_mean.extend(np.asarray(ref["y_mean"], dtype=float).tolist())
            got_mean.extend(mean.tolist())
            exp_valid.extend(valid_ref.tolist())
            got_valid.extend(valid_got.tolist())

        return {
            "expected": {
                "group_values": np.asarray(exp_group, dtype=int),
                "x_center": np.asarray(exp_x, dtype=float),
                "count": np.asarray(exp_count, dtype=int),
                "y_mean": np.asarray(exp_mean, dtype=float),
                "valid_mask": np.asarray(exp_valid, dtype=bool),
            },
            "observed": {
                "group_values": np.asarray(got_group, dtype=int),
                "x_center": np.asarray(got_x, dtype=float),
                "count": np.asarray(got_count, dtype=int),
                "y_mean": np.asarray(got_mean, dtype=float),
                "valid_mask": np.asarray(got_valid, dtype=bool),
            },
            "l1": l1,
            "membership": membership,
            "source_dtypes": source_dtypes,
        }
    finally:
        try:
            import matplotlib.pyplot as plt
            plt.close(raw[0])
        except Exception:
            pass


def _preclean_c4_evaluate_prepared(adf: Any, gallery: Any, *, mode: str) -> dict:
    """Evaluate one already-built FULL loading-mode leg against independent truth."""
    payload = _preclean_c4_grouped_truth_and_public(adf, gallery)
    payload["mode"] = mode
    payload["source_rows"] = int(len(adf.df))
    payload["lazy_reader_present"] = bool(getattr(adf, "_lazy_reader", None) is not None)
    return payload



def _preclean_c4_membership_rows(adf: Any, model: dict, chain: dict) -> dict:
    """Compact stable-row evidence for rows entering the C4 independent profile.

    This is diagnostic data, not a second oracle.  It lets a failed FULL
    EAGER/LAZY comparison identify the exact row IDs and scientific values
    behind a count mismatch after the memory-heavy EAGER ADF is released.
    """
    df = adf.df
    row_id = np.asarray(df["oracle_row_id"], dtype=np.int64)
    x = np.asarray(model["sector"], dtype=float)
    y = np.asarray(chain["final"], dtype=float)
    side = np.asarray(model["side"])
    ncl = np.asarray(df["ncl"], dtype=float)
    dcar = np.asarray(df["dcar_tpc_vertex"], dtype=float)
    tgl = np.asarray(df["tgl"], dtype=float)

    lo, hi = INJECTED_TRUTH_RANGE
    edges = np.linspace(lo, hi, INJECTED_TRUTH_BINS + 1)
    bin_idx = np.searchsorted(edges, x, side="right") - 1
    bin_idx[x == hi] = INJECTED_TRUTH_BINS - 1
    selected = (
        np.asarray(model["side_sel"], dtype=bool)
        & np.isfinite(x)
        & np.isfinite(y)
        & ((side == 0) | (side == 1))
        & (bin_idx >= 0)
        & (bin_idx < INJECTED_TRUTH_BINS)
    )
    flat_bin = (
        np.asarray(side, dtype=np.int64) * INJECTED_TRUTH_BINS
        + np.asarray(bin_idx, dtype=np.int64)
    )
    return {
        "row_id": row_id[selected],
        "flat_bin": flat_bin[selected].astype(np.int16, copy=False),
        "sector": x[selected],
        "side_type": np.asarray(side[selected]),
        "ncl": ncl[selected],
        "dcar_tpc_vertex": dcar[selected],
        "tgl": tgl[selected],
        "final": np.asarray(y[selected], dtype=float),
    }


def _preclean_c4_membership_records(payload: dict, row_ids: Any, *, limit: int = 12) -> list[dict]:
    """Return small JSON-safe source records for selected row IDs."""
    wanted = np.asarray(row_ids, dtype=np.int64)
    if wanted.size == 0:
        return []
    rows = np.asarray(payload["row_id"], dtype=np.int64)
    pos_by_id = {int(rid): i for i, rid in enumerate(rows)}
    records = []
    for rid in wanted[:limit]:
        pos = pos_by_id.get(int(rid))
        if pos is None:
            continue
        flat = int(np.asarray(payload["flat_bin"])[pos])
        records.append({
            "oracle_row_id": int(rid),
            "flat_index": flat,
            "side_type": int(flat // INJECTED_TRUTH_BINS),
            "sector_bin": int(flat % INJECTED_TRUTH_BINS),
            "sector": float(np.asarray(payload["sector"])[pos]),
            "ncl": float(np.asarray(payload["ncl"])[pos]),
            "dcar_tpc_vertex": float(np.asarray(payload["dcar_tpc_vertex"])[pos]),
            "tgl": float(np.asarray(payload["tgl"])[pos]),
            "final": float(np.asarray(payload["final"])[pos]),
        })
    return records


def _preclean_c4_pair_diagnostics(eager: dict, lazy: dict) -> dict:
    """Locate independent EAGER/LAZY membership disagreement before tolerance logic."""
    ecount = np.asarray(eager["expected"]["count"], dtype=np.int64)
    lcount = np.asarray(lazy["expected"]["count"], dtype=np.int64)
    mismatch = np.flatnonzero(ecount != lcount)

    em = eager.get("membership", {})
    lm = lazy.get("membership", {})
    erows = np.asarray(em.get("row_id", ()), dtype=np.int64)
    lrows = np.asarray(lm.get("row_id", ()), dtype=np.int64)
    eflat = np.asarray(em.get("flat_bin", ()), dtype=np.int64)
    lflat = np.asarray(lm.get("flat_bin", ()), dtype=np.int64)

    bins = []
    for flat in mismatch:
        eids = np.sort(erows[eflat == flat])
        lids = np.sort(lrows[lflat == flat])
        only_e = np.setdiff1d(eids, lids, assume_unique=False)
        only_l = np.setdiff1d(lids, eids, assume_unique=False)
        bins.append({
            "flat_index": int(flat),
            "side_type": int(flat // INJECTED_TRUTH_BINS),
            "sector_bin": int(flat % INJECTED_TRUTH_BINS),
            "eager_count": int(ecount[flat]),
            "lazy_count": int(lcount[flat]),
            "count_delta_lazy_minus_eager": int(lcount[flat] - ecount[flat]),
            "only_eager_row_count": int(only_e.size),
            "only_lazy_row_count": int(only_l.size),
            "only_eager_row_ids": [int(x) for x in only_e[:20]],
            "only_lazy_row_ids": [int(x) for x in only_l[:20]],
            "only_eager_rows": _preclean_c4_membership_records(em, only_e),
            "only_lazy_rows": _preclean_c4_membership_records(lm, only_l),
        })

    global_only_e = np.setdiff1d(np.sort(erows), np.sort(lrows), assume_unique=False)
    global_only_l = np.setdiff1d(np.sort(lrows), np.sort(erows), assume_unique=False)
    return {
        "independent_count_mismatch_flat_indices": [int(x) for x in mismatch],
        "independent_count_mismatch_bins": bins,
        "selected_row_count_eager": int(erows.size),
        "selected_row_count_lazy": int(lrows.size),
        "selected_row_ids_equal_globally": bool(
            erows.size == lrows.size and np.array_equal(np.sort(erows), np.sort(lrows))
        ),
        "global_only_eager_row_count": int(global_only_e.size),
        "global_only_lazy_row_count": int(global_only_l.size),
        "global_only_eager_row_ids": [int(x) for x in global_only_e[:20]],
        "global_only_lazy_row_ids": [int(x) for x in global_only_l[:20]],
        "eager_source_dtypes": dict(eager.get("source_dtypes", {})),
        "lazy_source_dtypes": dict(lazy.get("source_dtypes", {})),
        "source_dtypes_equal": bool(
            eager.get("source_dtypes", {}) == lazy.get("source_dtypes", {})
        ),
        "first_disagreement_layer": "L0_REFERENCE_MEMBERSHIP" if mismatch.size else "NONE",
        "derived_owner": "UNRESOLVED_REFERENCE_INPUT" if mismatch.size else "NONE",
        "diagnostic_contract": (
            "exact stable-row membership before public-result or tolerance comparison"
        ),
    }


def _preclean_c4_failure_annotation(case: CaseSpec, result: CaseResult,
                                    diagnostics: Mapping[str, Any] | None = None) -> str:
    """Add compact observed failure evidence to the C4 PRIMARY ORACLE PDF."""
    lines = [
        footer_text(case),
        "",
        "OBSERVED:",
        f"  STATUS: {result.status}",
        f"  {result.detail or 'public result matched declared truth'}",
    ]
    if isinstance(diagnostics, Mapping):
        bins = diagnostics.get("independent_count_mismatch_bins", [])
        if bins:
            lines.append("  independent membership mismatches:")
            for row in bins[:6]:
                lines.append(
                    "    side_type={side_type} sector_bin={sector_bin}: "
                    "EAGER={eager_count} LAZY={lazy_count} "
                    "only_E={only_eager_row_count} only_L={only_lazy_row_count}".format(**row)
                )
    return "\n".join(lines)


def _preclean_c4_compare_pair(case: CaseSpec, eager: dict, lazy: dict) -> tuple[list[dict], dict]:
    """Compare EAGER and LAZY to truth and pin the first measured mode parity."""
    eexp, eobs = eager["expected"], eager["observed"]
    lexp, lobs = lazy["expected"], lazy["observed"]

    # The independent truth itself must describe the same FULL row population
    # in both loading modes before any product-result comparison is accepted.
    for name in ("group_values", "x_center", "count", "valid_mask"):
        cmp = compare_array(eexp[name], lexp[name], comparator="exact")
        if not cmp.ok:
            raise HarnessError(f"C4 independent EAGER/LAZY {name} differs: {cmp.detail}")
    truth_mean_cmp = compare_array(eexp["y_mean"], lexp["y_mean"], comparator="exact")
    if not truth_mean_cmp.ok:
        raise HarnessError(
            "C4 independent EAGER/LAZY y_mean is not bit-identical; "
            "the mechanism must be adjudicated before any tolerance is introduced: "
            + truth_mean_cmp.detail)

    valid = eobs["valid_mask"] & lobs["valid_mask"]
    mode_mean_delta = np.zeros_like(eobs["y_mean"], dtype=float)
    mode_mean_delta[valid] = lobs["y_mean"][valid] - eobs["y_mean"][valid]
    mode_count_delta = np.asarray(lobs["count"], dtype=np.int64) - np.asarray(eobs["count"], dtype=np.int64)

    expected = {
        "eager_group_values": eexp["group_values"],
        "lazy_group_values": eexp["group_values"],
        "eager_x_center": eexp["x_center"],
        "lazy_x_center": eexp["x_center"],
        "eager_count": eexp["count"],
        "lazy_count": eexp["count"],
        "eager_valid_mask": eexp["valid_mask"],
        "lazy_valid_mask": eexp["valid_mask"],
        "eager_y_mean": eexp["y_mean"],
        "lazy_y_mean": eexp["y_mean"],
        "mode_count_delta": np.zeros_like(mode_count_delta),
        "mode_y_mean_delta": np.zeros_like(mode_mean_delta),
    }
    observed = {
        "eager_group_values": eobs["group_values"],
        "lazy_group_values": lobs["group_values"],
        "eager_x_center": eobs["x_center"],
        "lazy_x_center": lobs["x_center"],
        "eager_count": eobs["count"],
        "lazy_count": lobs["count"],
        "eager_valid_mask": eobs["valid_mask"],
        "lazy_valid_mask": lobs["valid_mask"],
        "eager_y_mean": eobs["y_mean"],
        "lazy_y_mean": lobs["y_mean"],
        "mode_count_delta": mode_count_delta,
        "mode_y_mean_delta": mode_mean_delta,
    }

    evidence = []
    first_failure = None
    for obs in case.observables:
        cmp = compare_observable(obs, expected[obs.name], observed[obs.name])
        evidence.append(comparison_evidence(
            obs, cmp,
            reference_label="same independent raw M3 chained+qualified truth",
            candidate_label="FULL EAGER/LAZY public grouped profile"))
        if not cmp.ok and first_failure is None:
            first_failure = f"{obs.name}: {cmp.detail}"
    if first_failure is not None:
        raise HarnessError(first_failure)

    max_abs = 0.0
    if np.any(valid):
        max_abs = float(np.max(np.abs(mode_mean_delta[valid])))
    diagnostics = {
        "joint_valid_bins": int(np.count_nonzero(valid)),
        "max_abs_eager_lazy_y_mean_delta": max_abs,
        "mode_y_mean_exact": bool(np.array_equal(
            eobs["y_mean"], lobs["y_mean"], equal_nan=True)),
        "independent_truth_y_mean_exact": bool(np.array_equal(
            eexp["y_mean"], lexp["y_mean"], equal_nan=True)),
    }
    return evidence, diagnostics


def _preclean_c4_build_full(root_path: str, gallery: Any, *, lazy: bool) -> tuple[Any, dict]:
    """Build one unsampled FULL ADF leg and prove sampling was not invoked."""
    original_sample = pd.DataFrame.sample
    calls = []

    def forbidden_sample(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original_sample(self, *args, **kwargs)

    pd.DataFrame.sample = forbidden_sample
    try:
        adf = gallery.build_adf(
            root_path, sample=None, lazy=lazy, tree_name=PRECLEAN_C4_TREE_NAME)
    finally:
        pd.DataFrame.sample = original_sample
    if calls:
        raise HarnessError(f"C4 FULL {'LAZY' if lazy else 'EAGER'} unexpectedly sampled rows")
    reader = getattr(adf, "_lazy_reader", None)
    if lazy and reader is None:
        raise HarnessError("C4 LAZY_FULL unexpectedly fell back to eager loading")
    if not lazy and reader is not None:
        raise HarnessError("C4 EAGER_FULL unexpectedly exposes a lazy reader")
    st = os.stat(root_path)
    provenance = {
        "input_path": os.path.abspath(root_path),
        "input_size_bytes": int(st.st_size),
        "input_mtime_ns": int(st.st_mtime_ns),
        "tree_name": PRECLEAN_C4_TREE_NAME,
        "loading_mode": "LAZY" if lazy else "EAGER",
        "sample_mode": "FULL",
        "sample_fraction": None,
        "sample_seed": None,
        "source_rows": int(len(adf.df)),
    }
    return adf, provenance


def _preclean_c4_render_pdf(path: str, case: CaseSpec, eager: dict, lazy: dict,
                            result: CaseResult) -> None:
    """Render the one approved C4 PRIMARY ORACLE slow-gate page."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    e = eager["observed"]
    l = lazy["observed"]
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    for side in (0, 1):
        mask_e = (e["group_values"] == side) & e["valid_mask"]
        mask_l = (l["group_values"] == side) & l["valid_mask"]
        axes[0].plot(e["x_center"][mask_e], e["y_mean"][mask_e],
                     marker="o", ms=2.5, lw=1.0, label=f"EAGER side={side}")
        axes[0].plot(l["x_center"][mask_l], l["y_mean"][mask_l],
                     marker=".", ms=2.5, lw=1.0, label=f"LAZY side={side}")
        joint = (e["group_values"] == side) & e["valid_mask"] & l["valid_mask"]
        residual = l["y_mean"][joint] - e["y_mean"][joint]
        axes[1].plot(e["x_center"][joint], residual,
                     marker=".", ms=2.5, lw=0.9, label=f"side={side}")

    axes[0].set_ylabel("oracle_chain_with_shift mean")
    axes[0].set_title("C4 PRIMARY ORACLE — FULL EAGER ↔ LAZY qualified-chain truth")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(True, alpha=0.25)
    axes[1].axhline(0.0, lw=0.8)
    axes[1].set_xlabel("sector")
    axes[1].set_ylabel("LAZY - EAGER")
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(True, alpha=0.25)
    diagnostics = None
    if isinstance(result.observed, Mapping):
        diagnostics = result.observed.get("pair_diagnostics")
    _annotate_stage_a_figure(
        fig, _preclean_c4_failure_annotation(case, result, diagnostics))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def run_preclean_c4_full_gate(root_path: str, *, manifest_path: str, pdf_path: str,
                              gallery_module=None) -> tuple[list[CaseResult], dict, int]:
    """Execute C4 FULL EAGER then FULL LAZY, compare both to one truth contract."""
    case = preclean_c4_case(root_path, gallery_module=gallery_module)
    result = CaseResult(case_id=case.case_id, status=FAIL)
    t0 = time.time()
    eager_adf = lazy_adf = None
    eager = lazy = None
    eager_prov = lazy_prov = None
    try:
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        if not case.applicable:
            result.status = SKIP
            result.detail = case.applicability_reason
        else:
            print(f"[C4 EAGER_FULL] {_o3_timestamp()} BUILD BEGIN")
            eager_adf, eager_prov = _preclean_c4_build_full(root_path, gallery, lazy=False)
            eager = _preclean_c4_evaluate_prepared(eager_adf, gallery, mode="EAGER_FULL")
            print(f"[C4 EAGER_FULL] {_o3_timestamp()} END rows={eager['source_rows']}")

            # Keep only small comparison arrays before constructing the LAZY leg.
            del eager_adf
            eager_adf = None
            _close()
            try:
                import gc
                gc.collect()
            except Exception:
                pass

            print(f"[C4 LAZY_FULL] {_o3_timestamp()} BUILD BEGIN")
            lazy_adf, lazy_prov = _preclean_c4_build_full(root_path, gallery, lazy=True)
            lazy = _preclean_c4_evaluate_prepared(lazy_adf, gallery, mode="LAZY_FULL")
            print(f"[C4 LAZY_FULL] {_o3_timestamp()} END rows={lazy['source_rows']}")

            identity_fields = ("input_path", "input_size_bytes", "input_mtime_ns", "tree_name", "source_rows")
            mismatch = {k: {"eager": eager_prov[k], "lazy": lazy_prov[k]}
                        for k in identity_fields if eager_prov[k] != lazy_prov[k]}
            if mismatch:
                raise HarnessError(f"C4 FULL source identity mismatch: {mismatch}")
            if not eager["l1"].get("L1_all_within_tolerance", False):
                raise HarnessError("C4 EAGER_FULL ADF materialized chain differs from independent formula")
            if not lazy["l1"].get("L1_all_within_tolerance", False):
                raise HarnessError("C4 LAZY_FULL ADF materialized chain differs from independent formula")

            pair_diag = _preclean_c4_pair_diagnostics(eager, lazy)
            result.observed = {
                "realdata_provenance": {"eager": eager_prov, "lazy": lazy_prov},
                "L1_materialization": {"eager": eager["l1"], "lazy": lazy["l1"]},
                "pair_diagnostics": pair_diag,
            }
            comparisons, diag = _preclean_c4_compare_pair(case, eager, lazy)
            result.comparisons = comparisons
            result.executed_comparisons = len(comparisons)
            result.observable_contract = [_contract(obs) for obs in case.observables]
            result.observed.update({
                "mode_parity": diag,
                "ownership_ladder": {
                    "L0": "same independently reconstructed M3 raw/chained/qualified truth",
                    "L1": "ADF materialized chain matches independent truth in EAGER and LAZY",
                    "L2": "public grouped profile in each mode matches the same independent truth",
                    "L3": "EAGER and LAZY public counts/masks/binning/y_mean are exactly identical",
                    "first_disagreement_layer": "NONE",
                    "contract_reference_status": "VERIFIED",
                    "owner_status": "NONE",
                    "derived_owner": "NONE",
                },
            })
            result.status = PASS
            result.detail = ""
            _preclean_c4_render_pdf(pdf_path, case, eager, lazy, result)
            result.payload_paths["C4/slow_pdf"] = os.path.abspath(pdf_path)
    except Exception as exc:
        result.status = FAIL
        result.detail = f"PRECLEAN_C4 FULL EAGER_LAZY FAIL: {exc}"
        result.exception = traceback.format_exc(limit=8)
        pair_diag = result.observed.get("pair_diagnostics", {}) if isinstance(result.observed, Mapping) else {}
        first = pair_diag.get("first_disagreement_layer", "UNRESOLVED")
        owner = pair_diag.get("derived_owner", "UNRESOLVED")
        if not result.observed.get("ownership_ladder"):
            result.observed["ownership_ladder"] = {
                "L0": "same ROOT identity plus exact stable-row independent membership",
                "L1": "ADF materialization checked independently in each loading mode",
                "L2": "public grouped profiles compared independently to truth",
                "L3": "exact EAGER↔LAZY parity",
                "first_disagreement_layer": first,
                "contract_reference_status": "VERIFIED" if eager is not None and lazy is not None else "UNRESOLVED",
                "owner_status": owner,
                "derived_owner": owner,
            }
        if eager is not None and lazy is not None:
            try:
                _preclean_c4_render_pdf(pdf_path, case, eager, lazy, result)
                result.payload_paths["C4/slow_pdf"] = os.path.abspath(pdf_path)
            except Exception as pdf_exc:
                result.observed["failure_pdf_error"] = f"{type(pdf_exc).__name__}: {pdf_exc}"
    finally:
        eager_adf = None
        lazy_adf = None
        _close()
        result.wall_time_s = round(time.time() - t0, 4)

    extra = {
        "stage_a_gate": "PRECLEAN_C4_FULL_EAGER_LAZY_PRIMARY_ORACLE",
        "preclean_c4": {
            "loading_mode": "BOTH",
            "sample_mode": "FULL",
            "execution_legs": ["EAGER_FULL", "LAZY_FULL"],
            "reused_gallery_function": PRECLEAN_C4_REUSED_GALLERY_FUNCTION,
            "fast_gallery_page_count_effect": 0,
            "slow_primary_oracle_pages": 1,
            "tolerance_policy": (
                "each mode uses existing M3 independent-truth tolerance; first measured "
                "EAGER↔LAZY y_mean parity is pinned exact and must be adjudicated before any relaxation"
            ),
            "diagnostic_revision": (
                "v06 records exact stable-row membership deltas by side_type/sector bin "
                "and writes the PRIMARY ORACLE PDF even when C4 is red"
            ),
        },
        "stage_a_closure": stage_a_closure_metadata(gallery_module),
    }
    manifest = write_manifest(manifest_path, [result], [case], extra=extra)
    return [result], manifest, strict_exit_code([result], [case])


# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_77 post-Stage-A gallery-oracle hardening v0.2 — STEP 2 / O5
# selection_vector × facet_by × normalize=delta correctness
# ─────────────────────────────────────────────────────────────────────────────

HARDENING_O5_CASE_ID = "I4-REAL-SELECTION-VECTOR-FACET-DELTA-CORRECTNESS-EAGER-20PCT-01"
HARDENING_O5_GALLERY_FUNCTION = "fig22_delta_faceted"
HARDENING_O5_FACETS = (0, 1)
HARDENING_O5_BINS = 36
HARDENING_O5_RANGE = (-0.5, 35.5)


def o5_realdata_case(root_path: str, gallery_module=None) -> CaseSpec:
    """Real-data correctness case upgrading the existing G4.22 gallery page."""
    try:
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        callable_ok = callable(getattr(gallery, HARDENING_O5_GALLERY_FUNCTION, None))
    except Exception:
        callable_ok = False
    applicable = bool(root_path and os.path.isfile(root_path) and callable_ok)
    reason = "" if applicable else "ROOT input or fig22_delta_faceted is unavailable"
    return CaseSpec(
        case_id=HARDENING_O5_CASE_ID,
        claim_id="I4.real_selection_vector_facet_delta.HARDENING.O5",
        title="real selection_vector×facet_by normalize=delta matches raw profile arithmetic",
        claim=("the existing G4.22 early/late selection-vector delta for side_type 0/1 "
               "matches an independent raw NumPy/pandas profile calculation using explicit "
               "sector-bin geometry"),
        failure_means=("the composed selection_vector×facet_by normalization changed bin "
                       "membership, branch/facet identity, profile central values, undefined-bin "
                       "masking, or delta arithmetic"),
        expected_visual=("the existing G4.22 early-minus-late DCA_r profile in two side_type "
                         "facet panels"),
        owner_on_failure="dfdraw",
        purpose="CORRECTNESS",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            "gallery_function": HARDENING_O5_GALLERY_FUNCTION,
            "surface": "draw",
            "expr": "dcar_tpc_vertex:sector",
            "selection": f"{HARDENING_BASE_SEL}&(side_type<2)",
            "selection_vector": ["time_s<median(time_s)", "time_s>=median(time_s)"],
            "normalize": "delta",
            "facet_by": "side_type",
            "facets": list(HARDENING_O5_FACETS),
            "bins": HARDENING_O5_BINS,
            "range": list(HARDENING_O5_RANGE),
            "sample_fraction": A5_2_SAMPLE_FRACTION,
            "sample_seed": A5_2_SAMPLE_SEED,
        },
        applicable=applicable,
        applicability_reason=reason,
        setup_contract=("reuse the single prepared EAGER 20% ADF; derive t_mid and all sector "
                        "bins directly from the raw frame and explicit CaseSpec range; execute "
                        "the unchanged public fig22_delta_faceted() request"),
        preconditions=(
            "time_s, sector, side_type, ncl and dcar_tpc_vertex exist in the prepared ADF",
            "the stable facet domain is explicitly side_type in {0,1}",
            "sector bins are fixed by CaseSpec to [-0.5,35.5] with 36 bins",
        ),
        figure_contract=FigureContract(
            expected_panels="two side_type facet panels",
            panel_roles="side_type=0 and side_type=1",
            expected_traces="signal/reference profiles plus their normalized delta in each facet",
            expected_group_count="no group_by dimension",
            primary_comparison=("raw NumPy/pandas per-bin signal/reference means and counts -> "
                                "public normalize_data_faceted delta payload"),
            residual_definition="expected delta = early profile mean - late profile mean",
            accepted_envelope=("exact facet/bin/count/mask identity; floating profile means and "
                               "delta within declared numerical tolerance"),
            case_ids=(HARDENING_O5_CASE_ID,),
            proof_kind="CORRECTNESS",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable("facet_values", "INDEPENDENT", "ARRAY", "raw side_type domain", comparator="exact"),
            Observable("signal_central", "INDEPENDENT", "ARRAY", "raw early per-bin means",
                       comparator="close", rtol=1e-12, atol=1e-12,
                       rationale="same arithmetic mean over identical explicitly defined bins"),
            Observable("reference_central", "INDEPENDENT", "ARRAY", "raw late per-bin means",
                       comparator="close", rtol=1e-12, atol=1e-12,
                       rationale="same arithmetic mean over identical explicitly defined bins"),
            Observable("delta_values", "INDEPENDENT", "ARRAY", "raw signal-reference delta",
                       comparator="close", rtol=1e-12, atol=1e-12,
                       rationale="delta is direct subtraction of the two independently calculated profile means"),
            Observable("valid_bin_mask", "INDEPENDENT", "ARRAY", "raw bins populated in both branches",
                       comparator="exact"),
        ),
        non_claims=(
            "O5 does not close y-vector normalization gaps",
            "O5 does not close non-profile normalization/refusal gaps",
        ),
        negative_control="FAMILY_MUTATION:O5_DELTA_NUMERICAL_MISMATCH",
        reference_policy="same-process",
    )


def _o5_profile_reference(df: Any, mask: Any, *, bins: int, value_range: tuple[float, float]) -> dict:
    """Independent raw-row profile reduction for one O5 branch/facet."""
    x = np.asarray(df["sector"], dtype=float)
    y = np.asarray(df["dcar_tpc_vertex"], dtype=float)
    take = np.asarray(mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    lo, hi = map(float, value_range)
    edges = np.linspace(lo, hi, int(bins) + 1)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx[x == hi] = int(bins) - 1
    inside = take & (idx >= 0) & (idx < int(bins))
    counts = np.zeros(int(bins), dtype=int)
    means = np.full(int(bins), np.nan, dtype=float)
    for b in range(int(bins)):
        yy = y[inside & (idx == b)]
        counts[b] = int(len(yy))
        if len(yy):
            means[b] = float(np.mean(yy))
    return {
        "bin_centers": (edges[:-1] + edges[1:]) / 2.0,
        "count": counts,
        "central": means,
    }


def _o5_expected_model(adf: Any, case: CaseSpec) -> dict:
    """Build O5 truth exclusively from raw rows + explicit CaseSpec geometry."""
    if not hasattr(adf, "df"):
        raise HarnessError("O5 prepared object has no .df raw-frame owner")
    df = adf.df
    required = ("time_s", "sector", "side_type", "ncl", "dcar_tpc_vertex")
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise HarnessError(f"O5 raw frame missing required columns: {missing}")
    spec = dict(case.canonical_spec)
    bins = int(spec["bins"])
    value_range = tuple(float(x) for x in spec["range"])
    facets = tuple(int(x) for x in spec["facets"])
    t_mid = float(df["time_s"].median())
    base = ((np.asarray(df["ncl"], dtype=float) > 60)
            & (np.abs(np.asarray(df["dcar_tpc_vertex"], dtype=float)) < 10)
            & (np.asarray(df["side_type"]) < 2))
    early = np.asarray(df["time_s"], dtype=float) < t_mid
    late = ~early
    by_facet = {}
    for facet in facets:
        fmask = np.asarray(df["side_type"] == facet, dtype=bool)
        signal = _o5_profile_reference(df, base & fmask & early, bins=bins, value_range=value_range)
        reference = _o5_profile_reference(df, base & fmask & late, bins=bins, value_range=value_range)
        valid = (signal["count"] > 0) & (reference["count"] > 0)
        delta = np.asarray(signal["central"] - reference["central"], dtype=float)
        delta[~valid] = np.nan
        by_facet[facet] = {
            "bin_centers": signal["bin_centers"],
            "signal_count": signal["count"],
            "reference_count": reference["count"],
            "signal_central": signal["central"],
            "reference_central": reference["central"],
            "valid_bin_mask": valid,
            "delta_values": delta,
        }
    return {
        "t_mid": t_mid,
        "facets": list(facets),
        "bins": bins,
        "range": list(value_range),
        "by_facet": by_facet,
    }


def _o5_normalize_facets(stats: Any) -> dict[int, dict]:
    if not isinstance(stats, dict):
        raise HarnessError(f"O5 expected dict stats, got {type(stats).__name__}")
    if stats.get("normalize_mode") != "delta":
        raise HarnessError(f"O5 expected normalize_mode='delta', got {stats.get('normalize_mode')!r}")
    if stats.get("facet_by") != "side_type":
        raise HarnessError(f"O5 expected facet_by='side_type', got {stats.get('facet_by')!r}")
    raw = stats.get("normalize_data_faceted")
    if not isinstance(raw, dict):
        raise HarnessError("O5 stats are missing normalize_data_faceted")
    out = {}
    for key, value in raw.items():
        try:
            facet = int(float(key))
        except (TypeError, ValueError) as exc:
            raise HarnessError(f"O5 facet key is not numeric: {key!r}") from exc
        if not isinstance(value, dict):
            raise HarnessError(f"O5 facet {facet} payload is not a dict")
        out[facet] = value
    return out


def _o5_compare_float(name: str, expected: Any, observed: Any, *, facet: int) -> float:
    e = np.asarray(expected, dtype=float)
    o = np.asarray(observed, dtype=float)
    if e.shape != o.shape:
        raise HarnessError(f"O5 numerical mismatch {name} facet={facet}: shape {e.shape} != {o.shape}")
    if not np.allclose(e, o, rtol=1e-12, atol=1e-12, equal_nan=True):
        finite = np.isfinite(e) & np.isfinite(o)
        max_delta = float(np.max(np.abs(e[finite] - o[finite]))) if np.any(finite) else float("nan")
        raise HarnessError(
            f"O5 numerical mismatch {name} facet={facet}: max_abs_delta={max_delta}")
    finite = np.isfinite(e) & np.isfinite(o)
    return float(np.max(np.abs(e[finite] - o[finite]))) if np.any(finite) else 0.0


def _o5_overlay_profiles(fig: Any, expected: dict) -> dict[int, dict[str, np.ndarray]]:
    """Extract signal/reference profile central values from the rendered facet axes."""
    centers = np.asarray(expected["by_facet"][expected["facets"][0]]["bin_centers"], dtype=float)
    out: dict[int, dict[str, np.ndarray]] = {}
    for ax in getattr(fig, "axes", ()):
        title = str(getattr(ax, "get_title", lambda: "")())
        m = re.search(r"side_type=([^\s]+)", title)
        if m is None:
            continue
        try:
            facet = int(float(m.group(1)))
        except ValueError:
            continue
        if facet not in expected["facets"]:
            continue
        primary = [
            line for line in getattr(ax, "lines", ())
            if str(line.get_marker()) == "o"
            and str(line.get_linestyle()).lower() not in ("none", "")
        ]
        if len(primary) != 2:
            raise HarnessError(
                f"O5 facet={facet} expected exactly two primary profile lines, got {len(primary)}")
        branch_arrays = []
        for line in primary:
            full = np.full(len(centers), np.nan, dtype=float)
            xs = np.asarray(line.get_xdata(), dtype=float)
            ys = np.asarray(line.get_ydata(), dtype=float)
            if xs.shape != ys.shape:
                raise HarnessError(f"O5 facet={facet} profile x/y shape mismatch")
            for x, y in zip(xs, ys):
                idx = int(np.argmin(np.abs(centers - x)))
                if not np.isclose(centers[idx], x, rtol=0, atol=1e-12):
                    raise HarnessError(
                        f"O5 facet={facet} rendered profile x={x} does not match explicit bin centers")
                full[idx] = float(y)
            branch_arrays.append(full)
        out[facet] = {"signal_central": branch_arrays[0], "reference_central": branch_arrays[1]}
    if sorted(out) != sorted(expected["facets"]):
        raise HarnessError(
            f"O5 rendered facet identity mismatch: expected={expected['facets']}, observed={sorted(out)}")
    return out


def _o5_validate_stats(stats: Any, expected: dict, *, fig: Any = None) -> dict:
    """Compare public normalized facet payload with independent raw truth."""
    faceted = _o5_normalize_facets(stats)
    expected_facets = list(expected["facets"])
    if sorted(faceted) != sorted(expected_facets):
        raise HarnessError(
            f"O5 facet identity mismatch: expected={expected_facets}, observed={sorted(faceted)}")
    profiles = _o5_overlay_profiles(fig, expected) if fig is not None else None
    flattened = {
        "facet_values": [], "signal_central": [], "reference_central": [],
        "delta_values": [], "valid_bin_mask": [],
    }
    max_delta = 0.0
    for facet in expected_facets:
        exp = expected["by_facet"][facet]
        got = faceted[facet]
        max_delta = max(max_delta, _o5_compare_float(
            "bin_centers", exp["bin_centers"], got.get("bin_centers"), facet=facet))
        observed_mask_undefined = np.asarray(got.get("mask_undefined"), dtype=bool)
        observed_valid = ~observed_mask_undefined
        if not np.array_equal(np.asarray(exp["valid_bin_mask"], dtype=bool), observed_valid):
            raise HarnessError(f"O5 valid-bin mask mismatch facet={facet}")
        max_delta = max(max_delta, _o5_compare_float(
            "delta_values", exp["delta_values"], got.get("values"), facet=facet))
        if profiles is not None:
            for name in ("signal_central", "reference_central"):
                max_delta = max(max_delta, _o5_compare_float(
                    name, exp[name], profiles[facet][name], facet=facet))
        flattened["facet_values"].append(facet)
        if profiles is not None:
            flattened["signal_central"].extend(
                np.asarray(profiles[facet]["signal_central"], dtype=float).tolist())
            flattened["reference_central"].extend(
                np.asarray(profiles[facet]["reference_central"], dtype=float).tolist())
        else:
            flattened["signal_central"].extend(
                np.asarray(exp["signal_central"], dtype=float).tolist())
            flattened["reference_central"].extend(
                np.asarray(exp["reference_central"], dtype=float).tolist())
        flattened["delta_values"].extend(
            np.asarray(got.get("values"), dtype=float).tolist())
        flattened["valid_bin_mask"].extend(observed_valid.tolist())
    flattened["max_abs_numerical_delta"] = float(max_delta)
    return flattened


def run_o5_realdata(case: CaseSpec, root_path: str, *, gallery_module=None,
                    prepared_adf=None, prepared_provenance=None) -> CaseResult:
    _skip = _inapplicable(case)
    if _skip is not None:
        return _skip
    t0 = time.time(); res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        adf = prepared_adf
        if adf is None:
            adf, provenance = _a6_4_build_fraction_adf_once(root_path, gallery_module=gallery)
        else:
            _a6_4_prepared_fraction_sample_evidence(adf, prepared_provenance, root_path)
            provenance = dict(prepared_provenance or {})
        expected = _o5_expected_model(adf, case)
        result = getattr(gallery, HARDENING_O5_GALLERY_FUNCTION)(adf)
        if not isinstance(result, tuple) or len(result) < 3:
            raise HarnessError("O5 gallery result does not expose public stats")
        actual = _o5_validate_stats(result[2], expected, fig=result[0])
        expected_flat = {
            "facet_values": list(expected["facets"]),
            "signal_central": [], "reference_central": [],
            "delta_values": [], "valid_bin_mask": [],
        }
        for facet in expected["facets"]:
            row = expected["by_facet"][facet]
            for name in ("signal_central", "reference_central", "delta_values"):
                expected_flat[name].extend(np.asarray(row[name]).tolist())
            expected_flat["valid_bin_mask"].extend(np.asarray(row["valid_bin_mask"], dtype=bool).tolist())
        for obs in case.observables:
            res.observable_contract.append(_contract(obs))
            exp_value = expected_flat[obs.name]
            got_value = actual[obs.name]
            cmp = compare_observable(obs, exp_value, got_value)
            res.comparisons.append(comparison_evidence(
                obs, cmp, reference_label="raw NumPy/pandas", candidate_label="public G4.22"))
        res.executed_comparisons = len(case.observables)
        res.observed.update({
            "realdata_provenance": provenance,
            "o5_oracle": {
                "t_mid": expected["t_mid"],
                "facets": expected["facets"],
                "bins": expected["bins"],
                "range": expected["range"],
                "max_abs_numerical_delta": actual["max_abs_numerical_delta"],
                "valid_bins_by_facet": {
                    str(f): int(np.count_nonzero(expected["by_facet"][f]["valid_bin_mask"]))
                    for f in expected["facets"]
                },
            },
        })
        res.status = PASS; res.detail = ""
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"O5_DELTA_CORRECTNESS FAIL: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        _close(); res.wall_time_s = round(time.time() - t0, 4)



# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_77 hardening v0.2 — STEP 3 / O4 public-surface equivalence
# ─────────────────────────────────────────────────────────────────────────────

HARDENING_O4_CASE_ID = "I4-REAL-PUBLIC-SURFACE-QUALIFIED-VECTOR-EQUIV-EAGER-20PCT-01"
HARDENING_O4_GALLERY_FUNCTION = "fig45_public_surface_equivalence_oracle"
HARDENING_O4_SUBFRAME = "O4Surface"
HARDENING_O4_SELECTIONS = (
    f"{HARDENING_O4_SUBFRAME}.selector==0",
    f"{HARDENING_O4_SUBFRAME}.selector==1",
)
HARDENING_O4_A7_COMMIT = "f730b0cfd4d6874e5f93331c35b3cc188470f7fc"
HARDENING_O4_ADF_SOURCE_MD5 = "698410cf846183d8f8f362be1516409c"
HARDENING_O4_ADF_SOURCE_SHA256 = "368853cbdb6b7418eaaa1ef92004de58fe25840f459f4385e4a9113ba1044ce0"


def current_stage_a_contract_amendment() -> dict:
    """Persistent current-state reconciliation without rewriting Stage-A history."""
    return {
        "owner": "PHASE_13_77 hardening v0.2 implementation manifest",
        "historical_stage_a_gallery_pages": 43,
        "historical_stage_a_closure_immutable": True,
        "pre_injected_checkpoint_gallery_pages": 47,
        "previous_checkpoint_gallery_pages": 53,
        "injected_truth_primary_pages": 6,
        "commissioning_addition_pages": 3,
        "preclean_c1_primary_pages": 1,
        "preclean_c2_primary_pages": 1,
        "preclean_c3_primary_pages": 1,
        "preclean_c4_slow_primary_pages": 1,
        "superseded_nodes": [
            {
                "node": ("tests/test_phase_13_77_realdata_invariance_harness.py::"
                         "test_a4_11_vector_catalogue_and_refusal_contracts_are_machine_visible"),
                "old_contract": "subframe-qualified selection_vector refuses with BUG_20260701",
                "current_contract": "PHASE_13_76 B3.3b supports qualified selection_vector",
            },
            {
                "node": ("tests/test_phase_13_77_realdata_invariance_harness.py::"
                         "test_a4_14_15_subframe_vector_refusals_hold_in_eager_and_lazy_modes"),
                "old_contract": "subframe-qualified vector requests refuse in EAGER and LAZY",
                "current_contract": "qualified vector requests execute in EAGER and LAZY",
            },
        ],
        "replacement_evidence": [
            ("tests/test_phase_13_77_realdata_invariance_harness.py::"
             "test_a4_11_vector_catalogue_and_historical_refusal_contracts_are_machine_visible"),
            ("tests/test_phase_13_77_realdata_invariance_harness.py::"
             "test_a4_14_15_b33b_supersedes_historical_subframe_vector_refusals_in_eager_and_lazy_modes"),
            ("tests/test_phase_13_77_realdata_invariance_harness.py::"
             "test_a7_21_o4_three_public_surfaces_match_on_qualified_vector_request"),
        ],
        "target_a7_commit": HARDENING_O4_A7_COMMIT,
        "target_aliasdataframe_md5": HARDENING_O4_ADF_SOURCE_MD5,
        "target_aliasdataframe_sha256": HARDENING_O4_ADF_SOURCE_SHA256,
        "current_step_gallery_pages": 59,
        "approved_final_fast_gallery_pages": 59,
    }


def _ensure_o4_surface_subframe(adf: Any) -> None:
    """Register one deterministic subframe over the complete parent key domain.

    Subframe projection precedes the request selection.  Covering every observed
    ``side_type`` key avoids unrelated AD-19 row-level-missingness refusal while
    still testing only selector==0 and selector==1 in the qualified vector slot.
    """
    import pandas as pd
    registry = getattr(getattr(adf, "_subframes", None), "subframes", {}) or {}
    if HARDENING_O4_SUBFRAME in registry:
        return
    if not hasattr(adf, "df") or "side_type" not in adf.df.columns:
        raise HarnessError("O4 fixture requires parent column side_type")
    keys = np.asarray(pd.unique(adf.df["side_type"].dropna()))
    if keys.size == 0:
        raise HarnessError("O4 fixture found no side_type keys")
    keys = np.sort(keys)
    sub = type(adf)(pd.DataFrame({
        "side_type": keys,
        "selector": keys.astype(np.int64, copy=False),
    }))
    adf.register_subframe(HARDENING_O4_SUBFRAME, sub, index_columns="side_type")


def _o4_canonical_spec() -> dict:
    return {
        "expr": "dcar_tpc_vertex:tgl",
        "selection": f"{HARDENING_BASE_SEL}&(side_type<2)",
        "type": "profile",
        "bins": 30,
        "range": (-1.5, 1.5),
        "selection_vector": list(HARDENING_O4_SELECTIONS),
        "normalize": "delta",
        "return_data": True,
        "auto_title": True,
    }


def o4_realdata_case(root_path: str, gallery_module=None) -> CaseSpec:
    gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    applicable = callable(getattr(gallery, HARDENING_O4_GALLERY_FUNCTION, None))
    return CaseSpec(
        case_id=HARDENING_O4_CASE_ID,
        claim_id="I4.real_public_surface_qualified_vector.HARDENING.O4",
        title="draw/draw_batch/draw_figures agree on one B3.3-qualified vector request",
        claim=("the same subframe-qualified selection_vector request has identical branch "
               "identity, binning, counts and profile values on draw, draw_batch and draw_figures"),
        failure_means=("one public surface dropped/reinterpreted the B3.3-qualified vector slot, "
                       "changed branch order/selection, or returned numerically different profile data"),
        expected_visual=("top: draw/draw_batch/draw_figures overlaid; bottom: candidate minus draw residuals"),
        owner_on_failure="ADF",
        purpose="INVARIANCE",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CONSISTENCY",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec=_o4_canonical_spec(),
        applicable=applicable,
        applicability_reason=("" if applicable else f"missing {HARDENING_O4_GALLERY_FUNCTION}"),
        setup_contract=("reuse the one canonical EAGER 20% ADF; register deterministic O4Surface over the "
                        "complete observed side_type key domain; execute the exact same qualified selection_vector request "
                        "through all three public surfaces"),
        preconditions=(
            "the canonical Stage-A EAGER 20% ADF is already built",
            "side_type, sector and dcar_tpc_vertex are present",
            "O4Surface.selector is used only through selection_vector",
        ),
        surfaces_under_test=SURFACES,
        slots_under_test=(),
        observables=(
            Observable("x_center", "STATS", "ARRAY", "normalize_data.x_center",
                       comparator="close", atol=1e-14, rtol=1e-12,
                       rationale="identical explicit tgl-bin coordinates across public surfaces"),
            Observable("signal_central", "STATS", "ARRAY", "normalize_data.signal_central",
                       comparator="close", atol=1e-12, rtol=1e-10,
                       rationale="same qualified signal branch profile across public surfaces"),
            Observable("signal_count", "STATS", "ARRAY", "normalize_data.signal_count"),
            Observable("reference_central", "STATS", "ARRAY", "normalize_data.reference_central",
                       comparator="close", atol=1e-12, rtol=1e-10,
                       rationale="same qualified reference branch profile across public surfaces"),
            Observable("reference_count", "STATS", "ARRAY", "normalize_data.reference_count"),
            Observable("value", "STATS", "ARRAY", "normalize_data.value",
                       comparator="close", atol=1e-12, rtol=1e-10,
                       rationale="same derived delta after branch-level equality is proven"),
        ),
        non_claims=(
            "O4 is public-surface consistency, not an independent correctness oracle",
            "O4 does not claim unsupported nested faceting",
        ),
        negative_control="FAMILY_MUTATION:O4_SURFACE_NUMERICAL_MISMATCH",
        reference_policy="same-process",
        figure_contract=FigureContract(
            expected_panels="two stacked panels",
            panel_roles="top: three public-surface delta profiles; bottom: candidate-minus-draw residuals",
            expected_traces="three top traces and two residual traces",
            expected_group_count="two qualified selection_vector branches",
            primary_comparison="draw versus draw_batch versus draw_figures returned numerical data",
            residual_definition="draw_batch-draw and draw_figures-draw",
            accepted_envelope="all declared observables equal within their explicit tolerances",
            case_ids=(HARDENING_O4_CASE_ID,),
            proof_kind="CONSISTENCY",
        ),
    )


def _o4_compare_stats(case: CaseSpec, stats_by_surface: dict[str, Any]) -> tuple[list[dict], dict]:
    if set(stats_by_surface) != set(SURFACES):
        raise HarnessError(
            f"O4 surface set mismatch: expected={list(SURFACES)}, observed={sorted(stats_by_surface)}")
    comparisons = []
    observed = {}
    reference = stats_by_surface["draw"]
    for obs in case.observables:
        try:
            values = {surface: resolve(stats, obs.path, obs.access)
                      for surface, stats in stats_by_surface.items()}
        except HarnessError as exc:
            raise HarnessError(f"O4 missing observable {obs.name}: {exc}") from exc
        observed[obs.name] = values
        for surface in ("draw_batch", "draw_figures"):
            cmp = compare_observable(obs, values["draw"], values[surface])
            rec = comparison_evidence(
                obs, cmp, reference_label="draw", candidate_label=surface)
            comparisons.append(rec)
            if not cmp.ok:
                raise HarnessError(
                    f"O4 numerical mismatch {obs.name}: draw vs {surface}: {cmp.detail}")
    return comparisons, observed


def run_o4_surface_equivalence(case: CaseSpec, root_path: str, *, gallery_module=None,
                               prepared_adf: Any = None,
                               prepared_provenance: dict | None = None) -> CaseResult:
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        if prepared_adf is None:
            raise HarnessError("O4 requires the shared FAST prepared_adf; rebuilding is forbidden")
        provenance = _a6_4_prepared_fraction_sample_evidence(
            prepared_adf, prepared_provenance, root_path)
        _ensure_o4_surface_subframe(prepared_adf)
        stats_by_surface = {}
        for surface in SURFACES:
            raw, kw = _call(prepared_adf, surface, case.canonical_spec, case_key="o4")
            errs = batch_errors(raw)
            if errs:
                raise HarnessError(f"O4 {surface} reported errors: {errs}")
            payload = unwrap(surface, raw, **kw)
            stats_by_surface[surface] = payload.stats
            res.payload_paths[surface] = list(payload.path)
            _close()
        comparisons, observed = _o4_compare_stats(case, stats_by_surface)
        res.observable_contract = [_contract(o) for o in case.observables]
        res.comparisons = comparisons
        res.executed_comparisons = len(comparisons)
        res.observed.update(observed)
        res.observed.update({
            "realdata_provenance": dict(prepared_provenance or provenance),
            "current_state_amendment": current_stage_a_contract_amendment(),
            "qualified_vector_slot": {
                "subframe": HARDENING_O4_SUBFRAME,
                "selection_vector": list(HARDENING_O4_SELECTIONS),
                "surface_order": list(SURFACES),
            },
        })
        if res.executed_comparisons != len(case.observables) * 2:
            raise HarnessError("O4 comparison cardinality drift")
        res.status = PASS
        res.detail = ""
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"O4_PUBLIC_SURFACE_EQUIVALENCE FAIL: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        _close()
        res.wall_time_s = round(time.time() - t0, 4)


# ─────────────────────────────────────────────────────────────────────────────
# A6.3 — final Stage-A orchestration, PDF evidence, and closure reconciliation
# ─────────────────────────────────────────────────────────────────────────────

GALLERY_DISPOSITION_ALLOWED = (
    "REUSED_CORE", "REUSED_VISUAL", "ENVIRONMENT_GATED", "KNOWN_BUG",
    "NOT_IN_STAGE_A",
)

# Explicit disposition of every trusted-gallery figNN_* function known at A6.3.
# The discovery check below makes this list fail closed when the gallery grows.
_GALLERY_DISPOSITION = {
    "fig01_hist_ncl": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig02_hist_time": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig03_hist_cumulative": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig04_scatter_dca_tgl": ("REUSED_VISUAL", "trusted mandatory gallery page; scatter auto-title workaround remains in gallery"),
    "fig05_hist2d_dca_sector": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig06_hexbin_dca_sector": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig07_profile_dca_sector": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig08_profile2d_dca_tgl_sector": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig09_scatter3d": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig10_profile_groupby_side": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig11_hist_groupby_side": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig12_profile_facet_side": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig13_profile_facet_nd": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig14_group_and_facet": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig15_profile_time": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig16_hist2d_time": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig17_profile_facet_time": ("KNOWN_BUG", "trusted page retained; facet time-format panel ticks remain the registered S-4 limitation"),
    "fig18_delta_sector13": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig19_ratio_time": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig20_pull_time": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig21_delta_side": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig22_delta_faceted": (
        "REUSED_CORE",
        "O5 real-data correctness oracle: selection_vector×facet_by normalize=delta"),
    "fig23_hist_fit": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig24_profile_fit": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig25_profile_fit_median": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig26_summary_fit": ("REUSED_VISUAL", "trusted mandatory gallery page including fit-summary table"),
    "fig27_vector": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig28_quantile_band": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig29_central_median": ("REUSED_VISUAL", "trusted mandatory non-grouped median page"),
    "fig30_overlay": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig31_selection_delta_ncl": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig32_subframe_vertex": ("REUSED_CORE", "A5.2 real-data G7.32 acceptance reuses this exact gallery function"),
    "fig33_gb_correction_tgl": ("REUSED_CORE", "A5.4/A5.5 real-data G7.33 preparation reuses this exact gallery function"),
    "fig34_gb_correction_sector": ("REUSED_CORE", "A5.5/A5.6 real-data G7.34 acceptance reuses this exact gallery function"),
    "fig35_batch_profile2d": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig36_batch_overlay": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig37_adf_draw_overlay": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig38_adf_draw_histo_alias": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig39_figures_overlay_in_spec": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig40_weights_alias": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig41_on_error_skip_placeholder": ("REUSED_VISUAL", "trusted visual-only error-placeholder demonstration"),
    "fig42_entry_window": ("REUSED_VISUAL", "trusted mandatory gallery page"),
    "fig43_vector_facet_summary_fit": (
        "REUSED_CORE",
        "A7 real-data correctness oracle: vector×facet×fit×summary_fit semantic coordinates"),
    "fig44_weights_vector_facet_fit_oracle": (
        "REUSED_CORE",
        "O2 correctness oracle: weights_vector×facet weighted profiles and pol1 fits"),
    "fig45_public_surface_equivalence_oracle": (
        "REUSED_CORE",
        "O4 consistency oracle: draw/draw_batch/draw_figures on B3.3-qualified selection_vector"),
    "fig46_injected_truth_vector_overlay": (
        "REUSED_CORE",
        "Injected-truth Rank 1: native vector overlay of original/clean/noisy/known delta"),
    "fig47_injected_truth_direct_delta": (
        "REUSED_CORE",
        "Injected-truth Rank 2: vector normalize=delta must recover known injected bias"),
    "fig48_injected_truth_selection_delta_facet": (
        "REUSED_CORE",
        "Injected-truth Rank 3: non-null selection_vector×normalize×facet correctness"),
    "fig49_injected_truth_weights_facet": (
        "REUSED_CORE",
        "Injected-truth Rank 4: weights_vector×facet known weighted-profile truth"),
    "fig50_injected_truth_gaussian_fit": (
        "REUSED_CORE",
        "Injected-truth Rank 5: residual Gaussian fit against known injected noise"),
    "fig51_injected_truth_facet_residual": (
        "REUSED_CORE",
        "Injected-truth Rank 6: simple facet residual truth for failure localization"),
    "fig52_hist_vector_truth": ("REUSED_CORE", "M1 histogram selection/weight vector independent truth"),
    "fig53_grouping_truth": ("REUSED_CORE", "M2 categorical and binned grouping independent truth"),
    "fig54_qualified_subframe_chain_truth": ("REUSED_CORE", "M3 qualified-subframe/chained-alias independent truth"),
    "fig55_preclean_hist_vector_facet_acceptance": ("REUSED_CORE", "C1 repaired ORACLE-02 hist×selection_vector×facet acceptance truth"),
    "fig56_preclean_draw_figures_effective_defaults": ("REUSED_CORE", "C2 draw_figures short-form effective-defaults injected-truth correctness"),
    "fig57_preclean_ratio_y_vector_truth": ("REUSED_CORE", "C3 y-vector normalize=ratio predeclared exact-scaled numerical truth"),
}

_NUMERICAL_CORRECTNESS_ANCHORS = (
    {
        "family": "histogram",
        "evidence": "test_phase_13_77_realdata_invariance_harness.py::test_a1_correctness_case_agrees_with_numpy",
        "meaning": "independent NumPy histogram/count anchor",
    },
    {
        "family": "hist_selection_vector_facet_acceptance",
        "evidence": "test_phase_13_77_realdata_invariance_harness.py::test_preclean_c1_runner_matches_3x2_histogram_truth",
        "meaning": "independent 3-facet × 2-branch NumPy histogram/count acceptance anchor for repaired ORACLE-02",
    },
    {
        "family": "y_vector_ratio_exact_scaled_truth",
        "evidence": "test_phase_13_77_realdata_invariance_harness.py::test_preclean_c3_runner_matches_exact_ratio_truth",
        "meaning": "independent raw positive/scaled y-vector profile ratio oracle with predeclared 0.5 truth",
    },
    {
        "family": "grouped_profile_keyed_subframe",
        "evidence": "test_phase_13_77_realdata_invariance_harness.py::test_a5_02_full_stack_matches_independent_oracle_in_eager_and_lazy_modes",
        "meaning": "independent grouped-bin NumPy/pandas anchor in EAGER and LAZY modes",
    },
    {
        "family": "vector_facet_summary_fit_semantics",
        "evidence": "test_phase_13_77_realdata_invariance_harness.py::test_a7_03_semantic_oracle_accepts_exact_product_and_row_order_is_nonsemantic",
        "meaning": "independent real-data branch×facet coordinate and rendered-table identity anchor",
    },
    {
        "family": "selection_vector_facet_delta",
        "evidence": "test_phase_13_77_realdata_invariance_harness.py::test_a7_16_o5_raw_oracle_matches_current_product_on_synthetic_data",
        "meaning": "independent raw NumPy/pandas per-bin early-minus-late profile oracle for G4.22",
    },
    {
        "family": "weights_vector_facet_profile",
        "evidence": "test_phase_13_77_realdata_invariance_harness.py::test_a7_29_o2_public_weights_vector_facet_matches_raw_correctness_oracle",
        "meaning": "independent raw NumPy/pandas weighted-profile oracle for O2 weights_vector×facet",
    },
    {
        "family": "realdata_injected_truth",
        "evidence": "test_phase_13_77_realdata_invariance_harness.py::test_a8_03_injected_vector_overlay_matches_independent_raw_truth",
        "meaning": "real data plus deterministic known bias and stable-row Gaussian noise; six normal ADF/dfdraw workflows checked against raw known truth",
    },
)


def gallery_disposition_table(gallery_module=None) -> list[dict]:
    """Return the complete, fail-closed §11 disposition of trusted figNN_* callables."""
    gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    discovered = sorted(
        name for name in dir(gallery)
        if re.fullmatch(r"fig\d{2}_.+", name) and callable(getattr(gallery, name, None))
    )
    expected = sorted(_GALLERY_DISPOSITION)
    missing = sorted(set(discovered) - set(expected))
    stale = sorted(set(expected) - set(discovered))
    if missing or stale:
        raise HarnessError(
            "gallery disposition drift: "
            f"unclassified={missing}, stale={stale}")
    rows = []
    for name in discovered:
        disposition, reason = _GALLERY_DISPOSITION[name]
        if disposition not in GALLERY_DISPOSITION_ALLOWED:
            raise HarnessError(
                f"{name}: invalid gallery disposition {disposition!r}")
        rows.append({
            "gallery_function": name,
            "disposition": disposition,
            "reason": reason,
        })
    return rows


def numerical_oracle_closure_record() -> dict:
    """Machine reconciliation of the Stage-A numerical-oracle closure criterion."""
    same_spec = [
        c.case_id for c in a3_cases()
        if c.purpose == "INVARIANCE" and len(tuple(c.surfaces_under_test)) >= 2
    ]
    missing_rationale = []
    for case in tuple(a3_cases()) + tuple(a4_cases()) + tuple(a5_cases()):
        for obs in case.observables:
            if obs.status == "EXECUTED" and obs.comparator == "close" \
                    and not str(obs.rationale).strip():
                missing_rationale.append(f"{case.case_id}/{obs.name}")
    blockers = []
    if not same_spec:
        blockers.append("no same-spec cross-surface numerical case")
    if len(_NUMERICAL_CORRECTNESS_ANCHORS) < 2:
        blockers.append("insufficient independent correctness-anchor families")
    if missing_rationale:
        blockers.append("floating tolerances without rationale: " + ", ".join(missing_rationale))
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "same_spec_cross_surface_case_ids": same_spec,
        "independent_correctness_anchors": [dict(x) for x in _NUMERICAL_CORRECTNESS_ANCHORS],
        "tolerance_rationale_missing": missing_rationale,
        "blockers": blockers,
    }


def stage_a_closure_metadata(gallery_module=None) -> dict:
    """Closure metadata shared by manifest and final review evidence."""
    return {
        "gallery_dispositions": gallery_disposition_table(gallery_module),
        "numerical_oracle": numerical_oracle_closure_record(),
        "a5_3_blocker_transition": a5_3_blocker_resolution_record(),
        "lazy_full_deferral_reconciliation": lazy_full_deferral_reconciliation(),
        "current_state_amendment": current_stage_a_contract_amendment(),
    }


def _stage_a_case_for_gallery_function(name: str, root_path: str, gallery_module=None):
    """Return the CaseSpec that owns a load-bearing real-data gallery page, if any."""
    if name == A5_2_GALLERY_FUNCTION:
        return a5_2_realdata_case(root_path, gallery_module=gallery_module)
    if name == A5_4_GALLERY_FUNCTION:
        return a5_4_realdata_case(root_path, gallery_module=gallery_module)
    if name == A5_5_REUSE_FUNCTION:
        return a5_5_realdata_case(root_path, gallery_module=gallery_module)
    if name == A7_1_GALLERY_FUNCTION:
        return a7_1_realdata_case(root_path, gallery_module=gallery_module)
    if name == HARDENING_O5_GALLERY_FUNCTION:
        return o5_realdata_case(root_path, gallery_module=gallery_module)
    if name == HARDENING_O4_GALLERY_FUNCTION:
        return o4_realdata_case(root_path, gallery_module=gallery_module)
    if name == HARDENING_O2_GALLERY_FUNCTION:
        return o2_oracle_case(gallery_module=gallery_module)
    injected = globals().get("INJECTED_TRUTH_GALLERY_CASES", {})
    case_factory = injected.get(name) if isinstance(injected, dict) else None
    if callable(case_factory):
        return case_factory(gallery_module=gallery_module)
    commissioning = {
        M1_HIST_GALLERY_FUNCTION: m1_hist_vector_case,
        M2_GROUP_GALLERY_FUNCTION: m2_grouping_case,
        M3_SUBFRAME_GALLERY_FUNCTION: m3_subframe_chain_case,
        PRECLEAN_C1_GALLERY_FUNCTION: lambda **_: preclean_c1_case(root_path, gallery_module=gallery_module),
        PRECLEAN_C2_GALLERY_FUNCTION: lambda **_: preclean_c2_case(root_path, gallery_module=gallery_module),
        PRECLEAN_C3_GALLERY_FUNCTION: lambda **_: preclean_c3_case(root_path, gallery_module=gallery_module),
    }
    factory = commissioning.get(name)
    if callable(factory):
        return factory(gallery_module=gallery_module)
    return None


def _annotate_stage_a_figure(fig: Any, text: str) -> None:
    """Add visible proof class and structured review instructions."""
    if fig is None:
        return
    is_primary = "PROOF CLASS: PRIMARY ORACLE" in text
    is_consistency = "PROOF CLASS: CONSISTENCY ORACLE" in text
    text_fn = getattr(fig, "text", None)
    if callable(text_fn):
        # Badge is intentionally large enough to survive PDF thumbnail review.
        badge = (PRIMARY_ORACLE if is_primary else
                 CONSISTENCY_ORACLE if is_consistency else SMOKE_COVERAGE)
        text_fn(
            0.99, 0.992, badge, ha="right", va="top",
            fontsize=8.0 if is_primary else 6.5,
            fontweight="bold", family="sans-serif",
        )
        text_fn(
            0.01, 0.012, text, ha="left", va="bottom",
            fontsize=3.7 if is_primary else 4.1,
            family="monospace", wrap=True,
        )
    adjust = getattr(fig, "subplots_adjust", None)
    if callable(adjust):
        try:
            adjust(bottom=0.40 if is_primary else 0.30)
        except Exception:
            pass


def write_stage_a_pdf(adf: Any, path: str, *, root_path: str,
                      gallery_module=None) -> dict:
    """Render Stage-A human evidence using the trusted gallery PDF primitives.

    The gallery remains independently runnable.  This wrapper reuses its exact
    figure callables, ``PdfPages`` and ``_add`` saver, adding only structured
    Stage-A footer/disposition annotations.
    """
    gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    for attr in ("PdfPages", "_add", "FIGURES_MANDATORY", "FIGURES_OPTIONAL",
                 "FIGURE_EXTRA_PAGE_SPECS", "declared_extra_page_count",
                 "extract_declared_extra_pages"):
        if not hasattr(gallery, attr):
            raise HarnessError(f"time_series_draw missing PDF reuse owner {attr!r}")
    dispositions = {r["gallery_function"]: r for r in gallery_disposition_table(gallery)}
    errors = []
    skipped = []
    n_pages = 0
    perf_logger = getattr(gallery, "logger", None)
    perf_log = getattr(perf_logger, "log", None)
    required_names = {fn.__name__ for fn in gallery.FIGURES_MANDATORY}
    required_names.update(
        fn.__name__ for fn in gallery.FIGURES_OPTIONAL
        if dispositions[fn.__name__]["disposition"] == "REUSED_CORE")
    # Extra generated pages are declared by the trusted gallery owner.  The
    # same declaration drives rendering and strict expected-page accounting.
    expected_page_count = len(required_names) + sum(
        int(gallery.declared_extra_page_count(name)) for name in required_names)

    with gallery.PdfPages(path) as pdf:
        for mandatory, funcs in ((True, gallery.FIGURES_MANDATORY),
                                 (False, gallery.FIGURES_OPTIONAL)):
            for fn in funcs:
                name = fn.__name__
                title = (fn.__doc__ or name).splitlines()[0].strip()
                disposition = dispositions[name]["disposition"]
                required = bool(mandatory or disposition == "REUSED_CORE")
                if callable(perf_log):
                    perf_log(f"{name} : BEGIN")
                try:
                    result = fn(adf)
                    if result is None:
                        if required:
                            raise HarnessError(
                                f"required gallery function {name} returned None "
                                f"(disposition={disposition})")
                        skipped.append({"gallery_function": name, "reason": "returned None"})
                        continue
                    fig = result[0] if isinstance(result, tuple) else result
                    case = _stage_a_case_for_gallery_function(
                        name, root_path, gallery_module=gallery)
                    if case is not None:
                        annotation = footer_text(case)
                        status_map = getattr(adf, "_stage_a_machine_status", {})
                        status_row = status_map.get(case.case_id) if isinstance(status_map, dict) else None
                        if isinstance(status_row, dict):
                            annotation += (
                                "\nOBSERVED:\n  "
                                + str(status_row.get("detail", "") or "public result matched declared truth")
                                + "\n\nMACHINE ORACLE STATUS: "
                                + str(status_row.get("status", "UNKNOWN")) + "\n"
                            )
                    else:
                        row = dispositions[name]
                        annotation = (
                            f"PROOF CLASS: {SMOKE_COVERAGE}\n"
                            f"GALLERY DISPOSITION: {row['disposition']}\n"
                            f"REASON: {row['reason']}")
                    _annotate_stage_a_figure(fig, annotation)
                    gallery._add(pdf, fig, title)
                    n_pages += 1
                    extra_pages = gallery.extract_declared_extra_pages(name, result)
                    declared_extra = int(gallery.declared_extra_page_count(name))
                    if len(extra_pages) != declared_extra:
                        raise HarnessError(
                            f"{name}: declared/produced extra-page mismatch "
                            f"{declared_extra}!={len(extra_pages)}")
                    for extra_fig, suffix in extra_pages:
                        if case is not None:
                            extra_annotation = footer_text(case)
                            status_map = getattr(adf, "_stage_a_machine_status", {})
                            status_row = status_map.get(case.case_id) if isinstance(status_map, dict) else None
                            if isinstance(status_row, dict):
                                extra_annotation += (
                                    "\nOBSERVED:\n  "
                                    + str(status_row.get("detail", "") or "public result matched declared truth")
                                    + "\n\nMACHINE ORACLE STATUS: "
                                    + str(status_row.get("status", "UNKNOWN")) + "\n"
                                )
                        else:
                            row = dispositions[name]
                            extra_annotation = (
                                f"PROOF CLASS: {SMOKE_COVERAGE}\n"
                                f"GALLERY DISPOSITION: {row['disposition']}\n"
                                f"REASON: declared extra page for {name}")
                        _annotate_stage_a_figure(extra_fig, extra_annotation)
                        gallery._add(pdf, extra_fig, title + suffix)
                        n_pages += 1
                except Exception as exc:
                    if required:
                        errors.append({"gallery_function": name,
                                       "error": f"{type(exc).__name__}: {exc}"})
                    else:
                        skipped.append({"gallery_function": name,
                                        "reason": f"{type(exc).__name__}: {exc}"})
                    try:
                        import matplotlib.pyplot as plt
                        plt.close("all")
                    except Exception:
                        pass
                finally:
                    if callable(perf_log):
                        perf_log(f"{name} : END")

    if n_pages != expected_page_count:
        errors.append({
            "gallery_function": "__page_count__",
            "error": (f"Stage-A gallery produced {n_pages} pages; "
                      f"expected exactly {expected_page_count}"),
        })
    return {
        "ok": not errors,
        "pdf_path": os.path.abspath(path),
        "page_count": n_pages,
        "expected_page_count": expected_page_count,
        "required_gallery_functions": sorted(required_names),
        "errors": errors,
        "skipped": skipped,
        "gallery_dispositions": list(dispositions.values()),
    }


def _a6_3_visual_case(*, sample_mode: str, loading_mode: str = "EAGER") -> CaseSpec:
    label = "20PCT" if sample_mode == "FRACTION" else "FULL"
    return CaseSpec(
        case_id=f"A6-VISUAL-GALLERY-{label}-01",
        claim_id="A6.visual.gallery",
        title=f"trusted Stage-A gallery renders in {sample_mode} mode",
        claim=("the unchanged trusted time_series_draw gallery renders every mandatory "
               "page and records an explicit disposition for every figNN_* function"),
        failure_means=("a mandatory trusted gallery page stopped rendering, a gallery "
                       "function disappeared from disposition coverage, or PDF evidence "
                       "cannot be produced"),
        expected_visual="one reviewed PDF containing all mandatory trusted-gallery pages",
        owner_on_failure="ADF",
        purpose="COVERAGE",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CONSISTENCY",
        loading_mode=loading_mode,
        sample_mode=sample_mode,
        canonical_spec={"gallery": "time_series_draw", "pdf": True},
        applicable=True,
        setup_contract="reuse time_series_draw.build_adf and trusted gallery figure lists",
        preconditions=("ROOT input exists", "trusted gallery is importable"),
        surfaces_under_test=("draw",),
        observables=(),
        non_claims=("PDF evidence is not a substitute for machine numerical oracles",),
        reference_policy="same-process",
    )


def _run_a6_3_visual_case(case: CaseSpec, root_path: str, *, pdf_path: str,
                          gallery_module=None, sample_fraction: float | None = None,
                          lazy: bool = False, prepared_adf: Any = None) -> CaseResult:
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        build = getattr(gallery, "build_adf", None)
        if not callable(build):
            raise HarnessError("time_series_draw missing callable build_adf")
        if prepared_adf is None:
            build_kwargs = {"sample": sample_fraction, "lazy": bool(lazy)}
            if lazy:
                build_kwargs["tree_name"] = A5_3_TREE_NAME
            adf = build(root_path, **build_kwargs)
        else:
            adf = prepared_adf
        lazy_before = _a5_3_loaded_branches(adf) if lazy else None
        evidence = write_stage_a_pdf(
            adf, pdf_path, root_path=root_path, gallery_module=gallery)
        if lazy:
            lazy_after = _a5_3_loaded_branches(adf)
            evidence["lazy_loaded_branches_before_gallery"] = list(lazy_before or ())
            evidence["lazy_loaded_branches_after_gallery"] = list(lazy_after or ())
            evidence["lazy_loaded_branch_count"] = len(lazy_after or ())
        res.observed["visual_evidence"] = evidence
        if not evidence["ok"]:
            res.status = FAIL
            res.detail = f"mandatory PDF gallery failures: {evidence['errors']}"
        else:
            res.status = PASS
        return res
    except Exception as exc:
        res.status = FAIL
        res.detail = f"{type(exc).__name__}: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        res.wall_time_s = round(time.time() - t0, 4)



def _a6_4_prepared_fraction_sample_evidence(
        adf: Any, provenance: dict | None, root_path: str) -> dict:
    """Validate one shared EAGER/FRACTION build before an A5 runner reuses it."""
    if not isinstance(provenance, dict):
        raise HarnessError("shared Stage-A FRACTION build is missing provenance")
    required = (
        "input_path", "input_size_bytes", "input_mtime_ns", "loading_mode",
        "sample_mode", "sample_fraction", "sample_seed", "sampling_algorithm",
        "source_rows", "selected_rows", "index_digest_sha256", "index_dtype",
    )
    missing = [key for key in required if key not in provenance]
    if missing:
        raise HarnessError(
            "shared Stage-A FRACTION provenance missing: " + ", ".join(missing))
    if os.path.abspath(str(provenance["input_path"])) != os.path.abspath(root_path):
        raise HarnessError("shared Stage-A FRACTION input_path mismatch")
    if provenance["loading_mode"] != "EAGER" or provenance["sample_mode"] != "FRACTION":
        raise HarnessError("shared Stage-A FRACTION build has wrong loading/sample mode")
    if (float(provenance["sample_fraction"]) != A5_2_SAMPLE_FRACTION
            or int(provenance["sample_seed"]) != A5_2_SAMPLE_SEED):
        raise HarnessError("shared Stage-A FRACTION build has wrong fraction/seed")
    if not hasattr(adf, "df") or int(len(adf.df)) != int(provenance["selected_rows"]):
        raise HarnessError("shared Stage-A FRACTION row count disagrees with provenance")
    if getattr(adf, "_lazy_reader", None) is not None:
        raise HarnessError("shared Stage-A FRACTION build unexpectedly has a lazy reader")
    if not str(provenance["index_digest_sha256"]):
        raise HarnessError("shared Stage-A FRACTION provenance has empty index digest")
    return {
        "source_rows": int(provenance["source_rows"]),
        "selected_rows": int(provenance["selected_rows"]),
        "index_digest_sha256": str(provenance["index_digest_sha256"]),
        "index_dtype": str(provenance["index_dtype"]),
    }


def _a6_4_build_fraction_adf_once(root_path: str, *, gallery_module=None) -> tuple[Any, dict]:
    """Build the canonical EAGER 20% ADF exactly once for the whole A6 gate."""
    import pandas as pd

    gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    build = getattr(gallery, "build_adf", None)
    if not callable(build):
        raise HarnessError("time_series_draw missing callable build_adf")

    sample_calls = []
    original_sample = pd.DataFrame.sample

    def recording_sample(self, *args, **kwargs):
        out = original_sample(self, *args, **kwargs)
        if (kwargs.get("frac") == A5_2_SAMPLE_FRACTION
                and kwargs.get("random_state") == A5_2_SAMPLE_SEED):
            sample_calls.append({
                "source_rows": int(len(self)),
                "selected_rows": int(len(out)),
                "index_digest_sha256": _a5_2_index_digest(out.index),
                "index_dtype": str(out.index.dtype),
            })
        return out

    pd.DataFrame.sample = recording_sample
    try:
        adf = build(root_path, sample=A5_2_SAMPLE_FRACTION, lazy=False)
    finally:
        pd.DataFrame.sample = original_sample

    if len(sample_calls) != 1:
        raise HarnessError(
            "shared Stage-A FRACTION build expected exactly one canonical sample call; "
            f"observed {len(sample_calls)}")
    sample = sample_calls[0]
    if not hasattr(adf, "df") or len(adf.df) != sample["selected_rows"]:
        raise HarnessError("shared Stage-A FRACTION build row count mismatch")
    st = os.stat(root_path)
    provenance = {
        "input_path": os.path.abspath(root_path),
        "input_size_bytes": int(st.st_size),
        "input_mtime_ns": int(st.st_mtime_ns),
        "loading_mode": "EAGER",
        "sample_mode": "FRACTION",
        "sample_fraction": A5_2_SAMPLE_FRACTION,
        "sample_seed": A5_2_SAMPLE_SEED,
        "sampling_algorithm": (
            "pandas.DataFrame.sample(frac=0.20, random_state=42) observed at runtime"),
        **sample,
    }
    return adf, provenance


def _a6_3_fraction_cases(root_path: str, gallery_module=None) -> tuple[CaseSpec, ...]:
    # v0.2 STEP 1: O1 controls run first and reuse the same prepared EAGER 20% ADF.
    return (
        a7_1_realdata_case(root_path, gallery_module=gallery_module),
        o1_neg_a_case(root_path, gallery_module=gallery_module),
        o1_neg_b_case(root_path, gallery_module=gallery_module),
        preclean_c1_case(root_path, gallery_module=gallery_module),
        preclean_c2_case(root_path, gallery_module=gallery_module),
        preclean_c3_case(root_path, gallery_module=gallery_module),
        o5_realdata_case(root_path, gallery_module=gallery_module),
        o4_realdata_case(root_path, gallery_module=gallery_module),
        o2_oracle_case(gallery_module=gallery_module),
        *injected_truth_cases(gallery_module=gallery_module),
        *commissioning_addition_cases(gallery_module=gallery_module),
        a5_2_realdata_case(root_path, gallery_module=gallery_module),
        a5_4_realdata_case(root_path, gallery_module=gallery_module),
        a5_5_realdata_case(root_path, gallery_module=gallery_module),
        a5_6_realdata_case(root_path, gallery_module=gallery_module),
    )


def _a6_3_fraction_runners():
    return (
        run_a7_1_realdata, run_o1_neg_a, run_o1_neg_b, run_preclean_c1, run_preclean_c2, run_preclean_c3, run_o5_realdata,
        run_o4_surface_equivalence, run_o2_realdata,
        run_injected_truth_case, run_injected_truth_case, run_injected_truth_case,
        run_injected_truth_case, run_injected_truth_case, run_injected_truth_case,
        run_m1_hist_vector, run_m2_grouping, run_m3_subframe_chain,
        run_a5_2_realdata, run_a5_4_realdata, run_a5_5_realdata, run_a5_6_realdata,
    )


def _shared_fraction_provenance(results: Sequence[CaseResult]) -> dict:
    identities = []
    provenance_docs = []
    for result in results:
        prov = result.observed.get("realdata_provenance")
        if isinstance(prov, dict):
            provenance_docs.append(dict(prov))
            identities.append(reference_identity_from_provenance(prov, require_complete=True))
    if not identities:
        raise HarnessError("Stage-A fraction gate produced no comparison-ready real-data provenance")
    first = identities[0]
    for identity in identities[1:]:
        compare_reference_identity(first, identity)
    return provenance_docs[0]


def run_stage_a_fraction_gate(root_path: str, *, manifest_path: str, pdf_path: str,
                              gallery_module=None) -> tuple[list[CaseResult], dict, int]:
    """Execute the deterministic 20% A6 gate from one shared ADF construction."""
    gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    adf, provenance_doc = _a6_4_build_fraction_adf_once(
        root_path, gallery_module=gallery)
    cases = list(_a6_3_fraction_cases(root_path, gallery_module=gallery))
    runners = _a6_3_fraction_runners()
    results = [
        runner(
            case, root_path, gallery_module=gallery,
            prepared_adf=adf, prepared_provenance=provenance_doc)
        for runner, case in zip(runners, cases)
    ]
    # Every case must report the same shared input/sample identity.
    _shared_fraction_provenance(results)
    visual_case = _a6_3_visual_case(sample_mode="FRACTION", loading_mode="EAGER")
    visual_result = _run_a6_3_visual_case(
        visual_case, root_path, pdf_path=pdf_path, gallery_module=gallery,
        sample_fraction=A5_2_SAMPLE_FRACTION, prepared_adf=adf)
    cases.append(visual_case)
    results.append(visual_result)
    extra = {
        **provenance_doc,
        "stage_a_gate": "FRACTION_20PCT",
        "stage_a_execution": {
            "adf_build_count": 1,
            "adf_reused_across_cases": True,
        },
        "stage_a_closure": stage_a_closure_metadata(gallery),
    }
    doc = write_manifest(manifest_path, results, cases, extra=extra)
    return results, doc, strict_exit_code(results, cases)



def numeric_oracle_recheck_cases(root_path: str, gallery_module=None) -> tuple[CaseSpec, ...]:
    """Bounded dfdraw-feedback rerun: four numerical cases + unchanged controls."""
    return (
        o2_oracle_case(gallery_module=gallery_module),
        injected_truth_vector_case(gallery_module=gallery_module),
        injected_truth_selection_case(gallery_module=gallery_module),
        m2_grouping_case(gallery_module=gallery_module),
        # Positive controls remain unchanged and guard semantic membership.
        o5_realdata_case(root_path, gallery_module=gallery_module),
        injected_truth_weights_case(gallery_module=gallery_module),
        m1_hist_vector_case(gallery_module=gallery_module),
        m3_subframe_chain_case(gallery_module=gallery_module),
    )


def numeric_oracle_recheck_runners():
    return (
        run_o2_realdata,
        run_injected_truth_case,
        run_injected_truth_case,
        run_m2_grouping,
        run_o5_realdata,
        run_injected_truth_case,
        run_m1_hist_vector,
        run_m3_subframe_chain,
    )


def _numeric_recheck_ownership_table(results: Sequence[CaseResult],
                                     requested_case_ids: Sequence[str]) -> list[dict]:
    """Compact final ownership ledger for the four numerical recheck cases."""
    requested = set(requested_case_ids)
    rows = []
    for result in results:
        if result.case_id not in requested:
            continue
        observed = result.observed if isinstance(result.observed, Mapping) else {}
        ladder = observed.get("ownership_ladder", {}) if isinstance(observed, Mapping) else {}
        if not isinstance(ladder, Mapping):
            ladder = {}
        numeric = ladder.get("numerical_recheck", {})
        comparisons = numeric.get("comparisons", []) if isinstance(numeric, Mapping) else []
        high_ok = bool(comparisons) and all(
            bool(c.get("reference_high_precision", {}).get("ok"))
            for c in comparisons if isinstance(c, Mapping))
        native_ok = bool(comparisons) and all(
            bool(c.get("reference_native", {}).get("ok"))
            for c in comparisons if isinstance(c, Mapping))
        rows.append({
            "case_id": result.case_id,
            "status": result.status,
            "first_disagreement_layer": ladder.get("first_disagreement_layer", "UNRESOLVED"),
            "contract_reference_status": ladder.get("contract_reference_status", "UNRESOLVED"),
            "owner_status": ladder.get("owner_status", ladder.get("derived_owner", "UNRESOLVED")),
            "resolution": ladder.get("resolution", ""),
            "reference_native_all_ok": native_ok,
            "reference_high_precision_all_ok": high_ok,
        })
    return rows


def run_numeric_oracle_recheck(root_path: str, *, manifest_path: str,
                               gallery_module=None) -> tuple[list[CaseResult], dict, int]:
    """Run only ORACLE-03/04/06/07 plus stable positive controls.

    No gallery PDF is generated: this is the narrow evidence revision requested
    by the dfdraw panel.  The same canonical EAGER 20% fixture is built once.
    """
    gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    adf, provenance_doc = _a6_4_build_fraction_adf_once(root_path, gallery_module=gallery)
    cases = list(numeric_oracle_recheck_cases(root_path, gallery_module=gallery))
    results = [
        runner(
            case, root_path, gallery_module=gallery,
            prepared_adf=adf, prepared_provenance=provenance_doc)
        for runner, case in zip(numeric_oracle_recheck_runners(), cases)
    ]
    _shared_fraction_provenance(results)
    requested_cases = [
        HARDENING_O2_CASE_ID,
        INJECTED_TRUTH_VECTOR_CASE_ID,
        INJECTED_TRUTH_SELECTION_CASE_ID,
        M2_GROUP_CASE_ID,
    ]
    ownership_table = _numeric_recheck_ownership_table(results, requested_cases)
    extra = {
        **provenance_doc,
        "stage_a_gate": "NUMERIC_ORACLE_RECHECK",
        "numeric_oracle_recheck": {
            "requested_cases": requested_cases,
            "positive_controls": [
                HARDENING_O5_CASE_ID,
                INJECTED_TRUTH_WEIGHTS_CASE_ID,
                M1_HIST_CASE_ID,
                M3_SUBFRAME_CASE_ID,
            ],
            "contract": (
                "same materialized values and semantic membership; independent float64 "
                "statistical accumulation; no tolerance widening"),
            "final_ownership_table": ownership_table,
        },
    }
    doc = write_manifest(manifest_path, results, cases, extra=extra)
    return results, doc, strict_exit_code(results, cases)


def _a6_4_build_lazy_adf_once(root_path: str, *, gallery_module=None) -> tuple[Any, dict]:
    """Build the canonical real-data FULL+LAZY ADF once and prove it stayed lazy.

    This is the public-path guard against an eager/full-column fallback.  It does
    not require an exact minimal branch set, only a live lazy reader, no pandas
    sampling, the required ``timeMS`` setup branch loaded through the lazy API,
    and a strict subset of the available physical branches resident after setup.
    """
    import pandas as pd

    gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    build = getattr(gallery, "build_adf", None)
    if not callable(build):
        raise HarnessError("time_series_draw missing callable build_adf")

    sample_calls = []
    original_sample = pd.DataFrame.sample

    def forbidden_sample(self, *args, **kwargs):
        sample_calls.append({"args": list(args), "kwargs": dict(kwargs)})
        return original_sample(self, *args, **kwargs)

    pd.DataFrame.sample = forbidden_sample
    try:
        adf = build(root_path, sample=None, lazy=True, tree_name=A5_3_TREE_NAME)
    finally:
        pd.DataFrame.sample = original_sample

    if sample_calls:
        raise HarnessError(
            "FULL+LAZY build unexpectedly called pandas.DataFrame.sample")

    reader = getattr(adf, "_lazy_reader", None)
    if reader is None:
        raise HarnessError("FULL+LAZY build has no live _lazy_reader (eager fallback)")
    loaded = set(getattr(reader, "loaded_branches", ()) or ())
    available = set(getattr(reader, "available_branches", ()) or ())
    if not available:
        raise HarnessError("FULL+LAZY lazy reader exposes no available_branches evidence")
    if "timeMS" not in loaded:
        raise HarnessError(
            "FULL+LAZY setup did not materialize required timeMS through the lazy reader")
    if "ncl" in loaded:
        raise HarnessError(
            "FULL+LAZY setup preloaded figure-only branch ncl; lazy setup is too eager")
    if len(loaded) >= len(available):
        raise HarnessError(
            "FULL+LAZY setup materialized every available branch (eager/full-column fallback)")

    st = os.stat(root_path)
    provenance = {
        "input_path": os.path.abspath(root_path),
        "input_size_bytes": int(st.st_size),
        "input_mtime_ns": int(st.st_mtime_ns),
        "loading_mode": "LAZY",
        "sample_mode": "FULL",
        "sample_fraction": None,
        "sample_seed": None,
        "tree_name": A5_3_TREE_NAME,
        "source_rows": int(len(adf.df)),
        "lazy_reader_present": True,
        "eager_fallback": False,
        "lazy_available_branch_count": len(available),
        "lazy_loaded_branch_count_after_setup": len(loaded),
        "lazy_loaded_branches_after_setup": sorted(str(x) for x in loaded),
    }
    return adf, provenance


def run_stage_a_full_gallery_gate(root_path: str, *, manifest_path: str, pdf_path: str,
                                  gallery_module=None, lazy: bool = False
                                  ) -> tuple[list[CaseResult], dict, int]:
    """Run the trusted full-data gallery from one ADF construction.

    The closure-critical real-data full path is ``lazy=True``.  It proves a
    live lazy reader and rejects eager/full-column fallback before rendering.
    Real-data EAGER+FULL remains available as a diagnostic but is not part of
    the required A6 acceptance matrix.
    """
    gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
    loading_mode = "LAZY" if lazy else "EAGER"
    provenance = {}
    prepared_adf = None
    case = _a6_3_visual_case(sample_mode="FULL", loading_mode=loading_mode)
    if lazy:
        try:
            prepared_adf, provenance = _a6_4_build_lazy_adf_once(
                root_path, gallery_module=gallery)
        except Exception as exc:
            result = CaseResult(case_id=case.case_id, status=FAIL)
            result.detail = f"{type(exc).__name__}: {exc}"
            result.exception = traceback.format_exc(limit=8)
            doc = write_manifest(
                manifest_path, [result], [case],
                extra={
                    "loading_mode": "LAZY",
                    "sample_mode": "FULL",
                    "stage_a_gate": "FULL_GALLERY_LAZY",
                    "stage_a_execution": {
                        "adf_build_count": 1,
                        "adf_reused_across_cases": True,
                        "public_entrypoint": (
                            "examples/time_series/time_series_draw_invariance.py "
                            "--full --lazy --manifest <path> --pdf <path> --strict"),
                        "failure_manifest_persisted": True,
                    },
                    "a5_3_blocker_transition": a5_3_blocker_resolution_record(),
                })
            return [result], doc, strict_exit_code([result], [case])

    result = _run_a6_3_visual_case(
        case, root_path, pdf_path=pdf_path, gallery_module=gallery,
        sample_fraction=None, lazy=lazy, prepared_adf=prepared_adf)
    if lazy and isinstance(result.observed.get("visual_evidence"), dict):
        result.observed["visual_evidence"]["lazy_setup_provenance"] = dict(provenance)

    extra = {
        **provenance,
        "loading_mode": loading_mode,
        "sample_mode": "FULL",
        "stage_a_gate": f"FULL_GALLERY_{loading_mode}",
        "stage_a_execution": {
            "adf_build_count": 1,
            "adf_reused_across_cases": True,
            "public_entrypoint": (
                "examples/time_series/time_series_draw_invariance.py "
                "--full --lazy --manifest <path> --pdf <path> --strict"
                if lazy else
                "examples/time_series/time_series_draw_invariance.py --full ..."),
        },
        "stage_a_closure": stage_a_closure_metadata(gallery),
    }
    doc = write_manifest(manifest_path, [result], [case], extra=extra)
    return [result], doc, strict_exit_code([result], [case])


def build_stage_a_cli_parser():
    """Build the one PHASE_13_77 Stage-A CLI defined by proposal v1.2 §17."""
    import argparse
    parser = argparse.ArgumentParser(
        prog="time_series_draw_invariance.py",
        description="PHASE_13_77 Stage-A real-data acceptance runner")
    parser.add_argument("root_path")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--sample", type=float)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--validate-lazy-eager", action="store_true")
    mode.add_argument(
        "--numeric-recheck", action="store_true",
        help="rerun ORACLE-03/04/06/07 with float64 reference diagnostics plus positive controls")
    mode.add_argument(
        "--preclean-c4", action="store_true",
        help="run the FULL EAGER↔LAZY C4 qualified-chain PRIMARY ORACLE slow gate")
    parser.add_argument(
        "--lazy", action="store_true",
        help="with --full, use the existing unsampled lazy loader (read branches on demand)")
    parser.add_argument("--seed", type=int, default=A5_2_SAMPLE_SEED)
    parser.add_argument("--manifest")
    parser.add_argument("--pdf")
    parser.add_argument("--compare", metavar="REFERENCE")
    parser.add_argument("--accept-reference", metavar="PATH")
    parser.add_argument("--update-reference", metavar="NEW_PATH")
    parser.add_argument("--previous-reference", metavar="PATH")
    parser.add_argument("--accepted-code-baseline")
    parser.add_argument("--approval-identity")
    parser.add_argument("--approval-date")
    parser.add_argument("--reason-for-update", default="initial reviewed acceptance")
    parser.add_argument("--strict", action="store_true")
    return parser


def _require_cli_evidence_paths(args) -> None:
    if args.validate_lazy_eager:
        return
    if not args.manifest:
        raise HarnessError("Stage-A gate requires --manifest")
    if args.numeric_recheck:
        return
    if not args.pdf:
        raise HarnessError("Stage-A sample/full gate requires --pdf")


def _require_reference_approval_args(args) -> None:
    for attr in ("accepted_code_baseline", "approval_identity", "approval_date"):
        if not getattr(args, attr):
            raise HarnessError(
                f"reference acceptance/update requires --{attr.replace('_', '-')}")



def print_stage_a_console_summary(manifest: Mapping[str, Any]) -> None:
    """Print a compact production-facing Stage-A result/ownership summary.

    The manifest remains authoritative.  This is intentionally only a concise
    terminal view so operators do not need ad-hoc JSON/heredoc parsing.
    """
    reconciliation = manifest.get("reconciliation", {}) if isinstance(manifest, Mapping) else {}
    cases = manifest.get("cases", []) if isinstance(manifest, Mapping) else []
    declared = int(reconciliation.get("n_declared", len(cases)) or 0)
    n_results = int(reconciliation.get("n_results", len(cases)) or 0)
    gating = list(reconciliation.get("gating", []) or [])
    by_id = {
        row.get("case_id"): row
        for row in cases
        if isinstance(row, Mapping) and row.get("case_id")
    }

    print()
    print("=== PHASE_13_77 STAGE-A SUMMARY ===")
    print(f"cases: {n_results}/{declared}  gating_reds: {len(gating)}")

    unknown = 0
    for gate in gating:
        case_id = gate.get("case_id", "<missing-case-id>")
        row = by_id.get(case_id, {})
        observed = row.get("observed", {}) if isinstance(row, Mapping) else {}
        ladder = observed.get("ownership_ladder", {}) if isinstance(observed, Mapping) else {}
        owner = (ladder.get("owner_status") or ladder.get("derived_owner")) if isinstance(ladder, Mapping) else None
        contract = ladder.get("contract_reference_status") if isinstance(ladder, Mapping) else None
        first = ladder.get("first_disagreement_layer") if isinstance(ladder, Mapping) else None
        if not owner or owner in {"NONE", "UNKNOWN", "UNRESOLVED"}:
            owner = "UNRESOLVED"
            unknown += 1
        if not first or first == "NONE":
            first = "UNKNOWN"
        contract = contract or "UNRESOLVED"
        detail = str(row.get("detail", gate.get("reason", ""))).replace("\n", " ").strip()
        if len(detail) > 180:
            detail = detail[:177] + "..."
        print(f"RED  owner={owner:<12} contract={contract:<12} first={first:<7} {case_id}")
        if detail:
            print(f"     {detail}")

    if not gating:
        print("STRICT GATE: PASS")
    else:
        print(f"STRICT GATE: FAIL  UNKNOWN={unknown}")

    provenance_doc = manifest.get("provenance", {}) if isinstance(manifest, Mapping) else {}
    recheck = provenance_doc.get("numeric_oracle_recheck", {}) if isinstance(provenance_doc, Mapping) else {}
    ownership_table = recheck.get("final_ownership_table", []) if isinstance(recheck, Mapping) else []
    if ownership_table:
        print("--- NUMERIC ORACLE RECHECK OWNERSHIP ---")
        for row in ownership_table:
            print(
                f"{row.get('case_id')}  status={row.get('status')} "
                f"first={row.get('first_disagreement_layer')} "
                f"contract={row.get('contract_reference_status')} "
                f"owner={row.get('owner_status')} "
                f"native_ok={row.get('reference_native_all_ok')} "
                f"float64_ok={row.get('reference_high_precision_all_ok')}")
            if row.get("resolution"):
                print(f"     resolution={row.get('resolution')}")
    print("Manifest is authoritative; this summary is operator convenience only.")
    print("=== END STAGE-A SUMMARY ===")

def stage_a_cli_main(argv: Sequence[str] | None = None, *, gallery_module=None) -> int:
    """Execute the single Stage-A CLI; returns a process-style exit code."""
    parser = build_stage_a_cli_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        if args.lazy and not args.full:
            raise HarnessError("--lazy is supported only together with --full")
        if args.validate_lazy_eager:
            if any((args.compare, args.accept_reference, args.update_reference,
                    args.previous_reference)):
                raise HarnessError("lazy/eager validation cannot perform reference mutation/comparison")
            validator = getattr(gallery, "validate_lazy_vs_eager", None)
            if not callable(validator):
                raise HarnessError("time_series_draw missing validate_lazy_vs_eager")
            validator(args.root_path)
            return 0

        _require_cli_evidence_paths(args)
        if args.preclean_c4:
            if any((args.compare, args.accept_reference, args.update_reference,
                    args.previous_reference)):
                raise HarnessError(
                    "C4 FULL slow gate cannot perform reference mutation/comparison")
            results, manifest, gate_code = run_preclean_c4_full_gate(
                args.root_path, manifest_path=args.manifest, pdf_path=args.pdf,
                gallery_module=gallery)
        elif args.numeric_recheck:
            if any((args.compare, args.accept_reference, args.update_reference,
                    args.previous_reference)):
                raise HarnessError(
                    "numeric oracle recheck cannot perform reference mutation/comparison")
            results, manifest, gate_code = run_numeric_oracle_recheck(
                args.root_path, manifest_path=args.manifest, gallery_module=gallery)
        elif args.sample is not None:
            if args.sample != A5_2_SAMPLE_FRACTION or args.seed != A5_2_SAMPLE_SEED:
                raise HarnessError(
                    "canonical Stage-A FRACTION gate is fixed at --sample 0.20 --seed 42")
            results, manifest, gate_code = run_stage_a_fraction_gate(
                args.root_path, manifest_path=args.manifest, pdf_path=args.pdf,
                gallery_module=gallery)
        else:
            if any((args.compare, args.accept_reference, args.update_reference,
                    args.previous_reference)):
                raise HarnessError(
                    "persistent named-reference operations require the canonical 20% FRACTION gate")
            results, manifest, gate_code = run_stage_a_full_gallery_gate(
                args.root_path, manifest_path=args.manifest, pdf_path=args.pdf,
                gallery_module=gallery, lazy=args.lazy)

        if args.compare:
            compare_manifest_to_named_reference(manifest, args.compare)
        if args.accept_reference:
            _require_reference_approval_args(args)
            accept_named_reference(
                args.accept_reference, manifest,
                accepted_code_baseline=args.accepted_code_baseline,
                approval_identity=args.approval_identity,
                approval_date=args.approval_date,
                reason_for_update=args.reason_for_update)
        if args.update_reference:
            _require_reference_approval_args(args)
            if not args.previous_reference:
                raise HarnessError("--update-reference requires --previous-reference")
            update_named_reference(
                args.update_reference, args.previous_reference, manifest,
                accepted_code_baseline=args.accepted_code_baseline,
                approval_identity=args.approval_identity,
                approval_date=args.approval_date,
                reason_for_update=args.reason_for_update)
        print_stage_a_console_summary(manifest)
        return gate_code if args.strict else 0
    except HarnessError as exc:
        print(f"A6 Stage-A gate REFUSED: {exc}", file=sys.stderr)
        return 2



# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_77 post-Stage-A gallery-oracle hardening v0.2 — STEP 4a / O2
# weights_vector × facet correctness contract + independent raw oracle only.
# Public-product composition and fig44 are intentionally deferred to STEP 4b/4c.
# ─────────────────────────────────────────────────────────────────────────────

HARDENING_O2_CASE_ID = "I4-REAL-WEIGHTS-VECTOR-FACET-PROFILE-CORRECTNESS-EAGER-20PCT-01"
HARDENING_O2_GALLERY_FUNCTION = "fig44_weights_vector_facet_fit_oracle"
HARDENING_O2_FACETS = (0, 1)
HARDENING_O2_BINS = 30
HARDENING_O2_RANGE = (-1.5, 1.5)
HARDENING_O2_WEIGHT_BRANCHES = (
    ("unity", "1.0 + 0.0*abs(dcar_tpc_vertex)"),
    ("w_dca", "1.0 + abs(dcar_tpc_vertex)"),
)

# Numerical contract: the independent oracle uses the exact materialized/input
# values delivered to dfdraw, but performs statistical reductions in explicit
# float64.  This matches current dfdraw's source-reviewed promotion contract
# without calling dfdraw reducers.  The existing tolerances are retained; they
# are not widened by the numerical-oracle recheck.
HARDENING_O2_SUMW_RTOL = 5e-6
HARDENING_O2_SUMW_ATOL = 1e-6
HARDENING_O2_VALUE_RTOL = 5e-6
HARDENING_O2_VALUE_ATOL = 1e-7


def o2_oracle_case(gallery_module=None) -> CaseSpec:
    """Declare the O2 real-data correctness case owned by gallery fig44."""
    return CaseSpec(
        case_id=HARDENING_O2_CASE_ID,
        claim_id="I4.real_weights_vector_facet_profile.HARDENING.O2",
        title="real weights_vector×facet weighted profile matches raw weighted arithmetic",
        claim=("two deterministic weight branches across side_type 0/1 use explicit tgl bins "
               "and agree with an independent raw NumPy/pandas weighted-profile reference"),
        failure_means=("weight-branch or facet identity changed, bin membership changed, weighted "
                       "support changed, or weighted profile central/error values disagree with "
                       "the independent raw-row calculation"),
        expected_visual=("two side_type facet panels with two deterministic weighted DCA_r:tgl "
                         "profile branches and pol1 fits"),
        owner_on_failure="dfdraw",
        purpose="CORRECTNESS",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            "surface": "draw",
            "gallery_function": HARDENING_O2_GALLERY_FUNCTION,
            "expr": "dcar_tpc_vertex:tgl",
            "selection": f"{HARDENING_BASE_SEL}&(side_type<2)",
            "weights_vector": [expr for _, expr in HARDENING_O2_WEIGHT_BRANCHES],
            "weights_labels": [label for label, _ in HARDENING_O2_WEIGHT_BRANCHES],
            "vector_compose": "outer",
            "facet_by": "side_type",
            "facets": list(HARDENING_O2_FACETS),
            "bins": HARDENING_O2_BINS,
            "range": list(HARDENING_O2_RANGE),
            "fit": "pol1",
            "sample_fraction": A5_2_SAMPLE_FRACTION,
            "sample_seed": A5_2_SAMPLE_SEED,
        },
        applicable=True,
        applicability_reason="",
        setup_contract=("reuse the single prepared EAGER 20% ADF; compute all O2 reference arrays "
                        "directly from adf.df using the explicit CaseSpec bin edges and declared "
                        "weight formulas; execute the exact gallery fig44 public composed call"),
        preconditions=(
            "tgl, side_type, ncl and dcar_tpc_vertex exist in the prepared ADF",
            "side_type facet domain is explicitly {0,1}",
            "all O2 weights are finite and strictly positive on selected rows",
            "tgl bins are fixed to [-1.5,1.5] with 30 bins",
        ),
        figure_contract=FigureContract(
            expected_panels="two side_type facet panels",
            panel_roles="side_type=0 and side_type=1",
            expected_traces="two weight-branch weighted profiles with pol1 fits per facet",
            expected_group_count="no group_by dimension",
            primary_comparison=("raw NumPy/pandas weighted counts, sum_weights, effective support, "
                                "weighted means/std/sem -> public weights_vector×facet payload"),
            residual_definition="public weighted observable minus raw weighted reference",
            accepted_envelope=("exact facet/branch/bin/count identity; weighted numerical values "
                               "within declared tolerance"),
            case_ids=(HARDENING_O2_CASE_ID,),
            proof_kind="CORRECTNESS",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable("facet_values", "INDEPENDENT", "ARRAY", "raw side_type domain", comparator="exact"),
            Observable("weight_branch_labels", "INDEPENDENT", "ARRAY", "CaseSpec declared weights", comparator="exact"),
            Observable("count", "INDEPENDENT", "ARRAY", "raw per-bin selected-row count", comparator="exact"),
            Observable("sum_weights", "INDEPENDENT", "ARRAY", "raw per-bin sum(weights)",
                       comparator="close", rtol=HARDENING_O2_SUMW_RTOL,
                       atol=HARDENING_O2_SUMW_ATOL,
                       rationale=("independent float64 reference versus dfdraw source-dtype "
                                  "(float32 ROOT) accumulation")),
            Observable("y_mean", "INDEPENDENT", "ARRAY", "raw weighted profile mean",
                       comparator="close", rtol=HARDENING_O2_VALUE_RTOL,
                       atol=HARDENING_O2_VALUE_ATOL,
                       rationale=("sum(w*y)/sum(w) on identical rows; tolerance covers only "
                                  "source-dtype accumulation rounding")),
            Observable("y_std", "INDEPENDENT", "ARRAY", "raw weighted population std",
                       comparator="close", rtol=HARDENING_O2_VALUE_RTOL,
                       atol=HARDENING_O2_VALUE_ATOL,
                       rationale=("same weighted population formula; tolerance covers only "
                                  "source-dtype accumulation rounding")),
            Observable("y_sem", "INDEPENDENT", "ARRAY", "raw weighted SEM via effective sample size",
                       comparator="close", rtol=HARDENING_O2_VALUE_RTOL,
                       atol=HARDENING_O2_VALUE_ATOL,
                       rationale=("same n_eff formula; tolerance covers only source-dtype "
                                  "accumulation rounding")),
        ),
        non_claims=(
            "O2 does not introduce an independent polynomial fitter; fit parameters are checked by scalar decomposition",
            "O2 does not change dfdraw weighting semantics or production code",
        ),
        negative_control="FAMILY_MUTATION:O2_RAW_WEIGHTED_NUMERICAL_MISMATCH",
        reference_policy="same-process",
    )


def _o2_weight_values(df: Any, branch_label: str) -> np.ndarray:
    """Independent implementation of the two approved O2 weight definitions.

    Preserve the authoritative source dtype.  dfdraw evaluates these expressions
    on the physical ROOT-derived column without an implicit float64 promotion;
    the oracle must test the same declared arithmetic contract rather than create
    an artificial source-dtype-vs-float64 disagreement.
    """
    dcar = np.asarray(df["dcar_tpc_vertex"])
    if branch_label == "unity":
        return 1.0 + 0.0 * np.abs(dcar)
    if branch_label == "w_dca":
        return 1.0 + np.abs(dcar)
    raise HarnessError(f"O2 unknown weight branch {branch_label!r}")


def _o2_weighted_profile_reference_arrays(
        x: Any, y: Any, weights: Any, *, bins: int,
        value_range: tuple[float, float], accumulator: str = "float64") -> dict:
    """Independent weighted-profile reduction over the exact input values.

    ``accumulator='float64'`` is the contract reference used for gating: the
    materialized/evaluated values are unchanged, while all statistical sums are
    accumulated in explicit float64, matching current dfdraw's documented
    promotion before reduction.  ``accumulator='native'`` is retained only as a
    diagnostic calibration path so we can detect old false reds caused by
    low-precision NumPy accumulation.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(weights)
    if not (x.shape == y.shape == w.shape):
        raise HarnessError(f"O2 raw arrays have incompatible shapes: {x.shape}, {y.shape}, {w.shape}")
    if accumulator not in {"float64", "native"}:
        raise HarnessError(f"unknown O2 accumulator {accumulator!r}")

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[finite], y[finite], w[finite]
    lo, hi = map(float, value_range)
    nbins = int(bins)
    edges = np.linspace(lo, hi, nbins + 1)
    x_for_bins = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(edges, x_for_bins, side="right") - 1
    idx[x_for_bins == hi] = nbins - 1
    inside = (idx >= 0) & (idx < nbins)
    idx, y, w = idx[inside], y[inside], w[inside]

    count = np.zeros(nbins, dtype=int)
    sum_weights = np.full(nbins, np.nan, dtype=float)
    n_eff = np.full(nbins, np.nan, dtype=float)
    y_mean = np.full(nbins, np.nan, dtype=float)
    y_std = np.full(nbins, np.nan, dtype=float)
    y_sem = np.full(nbins, np.nan, dtype=float)
    for b in range(nbins):
        take = idx == b
        yy_native = y[take]
        ww_native = w[take]
        n = int(len(yy_native))
        count[b] = n
        if not n:
            continue
        if accumulator == "float64":
            yy = np.asarray(yy_native, dtype=np.float64)
            ww = np.asarray(ww_native, dtype=np.float64)
            sw = float(np.sum(ww, dtype=np.float64))
            sw2 = float(np.sum(ww * ww, dtype=np.float64))
            numerator = float(np.sum(ww * yy, dtype=np.float64))
        else:
            yy = yy_native
            ww = ww_native
            sw_native = np.sum(ww)
            sw = float(sw_native)
            sw2 = float(np.sum(ww * ww))
            numerator = float(np.sum(ww * yy))
        sum_weights[b] = sw
        if sw <= 0:
            continue
        mean = numerator / sw
        y_mean[b] = mean
        if sw2 > 0:
            n_eff[b] = (sw * sw) / sw2
        if n > 1:
            if accumulator == "float64":
                variance = float(np.sum(ww * (yy - mean) ** 2, dtype=np.float64) / sw)
            else:
                variance = float(np.sum(ww * (yy - mean) ** 2) / sw)
            y_std[b] = float(np.sqrt(variance))
            if np.isfinite(n_eff[b]) and n_eff[b] > 0:
                y_sem[b] = float(y_std[b] / np.sqrt(n_eff[b]))
    return {
        "x_center": (edges[:-1] + edges[1:]) / 2.0,
        "x_low": edges[:-1],
        "x_high": edges[1:],
        "count": count,
        "sum_weights": sum_weights,
        "n_eff": n_eff,
        "y_mean": y_mean,
        "y_std": y_std,
        "y_sem": y_sem,
        "accumulator": accumulator,
    }


def _o2_expected_model(adf: Any, case: CaseSpec | None = None, *, accumulator: str = "float64") -> dict:
    """Build the complete O2 branch×facet weighted reference from raw rows."""
    case = case or o2_oracle_case()
    if not hasattr(adf, "df"):
        raise HarnessError("O2 prepared object has no .df raw-frame owner")
    df = adf.df
    required = ("tgl", "side_type", "ncl", "dcar_tpc_vertex")
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise HarnessError(f"O2 raw frame missing required columns: {missing}")
    spec = dict(case.canonical_spec)
    bins = int(spec["bins"])
    value_range = tuple(float(v) for v in spec["range"])
    facets = tuple(int(v) for v in spec["facets"])
    x = np.asarray(df["tgl"])
    y = np.asarray(df["dcar_tpc_vertex"])
    base = ((np.asarray(df["ncl"], dtype=float) > 60)
            & (np.abs(y) < 10)
            & (np.asarray(df["side_type"]) < 2))
    by_branch_facet = {}
    for branch_label, branch_expr in HARDENING_O2_WEIGHT_BRANCHES:
        weights = _o2_weight_values(df, branch_label)
        selected_w = weights[base]
        if selected_w.size == 0 or not np.all(np.isfinite(selected_w)) or np.any(selected_w <= 0):
            raise HarnessError(f"O2 branch {branch_label} has invalid/non-positive selected weights")
        for facet in facets:
            mask = base & (np.asarray(df["side_type"]) == facet)
            key = f"{branch_label}|side_type={facet}"
            by_branch_facet[key] = {
                "branch_label": branch_label,
                "weight_expression": branch_expr,
                "facet": facet,
                **_o2_weighted_profile_reference_arrays(
                    x[mask], y[mask], weights[mask], bins=bins, value_range=value_range,
                    accumulator=accumulator),
            }
    return {
        "facets": list(facets),
        "weight_branches": [label for label, _ in HARDENING_O2_WEIGHT_BRANCHES],
        "weight_expressions": [expr for _, expr in HARDENING_O2_WEIGHT_BRANCHES],
        "bins": bins,
        "range": list(value_range),
        "accumulator": accumulator,
        "by_branch_facet": by_branch_facet,
    }


def _o2_flatten_reference(expected: dict) -> dict:
    """Flatten in declaration-major order for future product comparisons."""
    out = {
        "facet_values": [], "weight_branch_labels": [], "count": [],
        "sum_weights": [], "y_mean": [], "y_std": [], "y_sem": [],
    }
    for branch in expected["weight_branches"]:
        for facet in expected["facets"]:
            row = expected["by_branch_facet"][f"{branch}|side_type={facet}"]
            out["facet_values"].append(facet)
            out["weight_branch_labels"].append(branch)
            for name in ("count", "sum_weights", "y_mean", "y_std", "y_sem"):
                out[name].extend(np.asarray(row[name]).tolist())
    return out


def _o2_assert_reference_consistent(case: CaseSpec, expected: dict) -> dict:
    """Fail for the intended O2 reference dimension, not an unrelated assertion."""
    flat = _o2_flatten_reference(expected)
    n_cells = len(expected["facets"]) * len(expected["weight_branches"])
    if n_cells != 4:
        raise HarnessError(f"O2 identity mismatch expected four branch×facet cells, got {n_cells}")
    if flat["facet_values"] != [0, 1, 0, 1]:
        raise HarnessError(f"O2 facet identity mismatch: {flat['facet_values']}")
    if flat["weight_branch_labels"] != ["unity", "unity", "w_dca", "w_dca"]:
        raise HarnessError(f"O2 weight identity mismatch: {flat['weight_branch_labels']}")
    counts = np.asarray(flat["count"], dtype=int)
    sumw = np.asarray(flat["sum_weights"], dtype=float)
    means = np.asarray(flat["y_mean"], dtype=float)
    populated = counts > 0
    if not np.any(populated):
        raise HarnessError("O2 numerical mismatch count: no populated weighted bins")
    if np.any(~np.isfinite(sumw[populated])) or np.any(sumw[populated] <= 0):
        raise HarnessError("O2 numerical mismatch sum_weights: populated bin has invalid support")
    if np.any(~np.isfinite(means[populated])):
        raise HarnessError("O2 numerical mismatch y_mean: populated bin has non-finite weighted mean")
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_77 hardening v0.2 — STEP 4b / O2 public-product integration
# ─────────────────────────────────────────────────────────────────────────────

def _o2_product_draw(adf: Any, case: CaseSpec | None = None):
    """Execute the approved public weights_vector×facet request."""
    case = case or o2_oracle_case()
    spec = case.canonical_spec
    return adf.draw(
        spec["expr"],
        selection=spec["selection"],
        type="profile",
        bins=int(spec["bins"]),
        range=tuple(spec["range"]),
        weights_vector=list(spec["weights_vector"]),
        facet_by="side_type",
        vector_compose="outer",
        fit="pol1",
        min_entries=1,
        auto_title=False,
        return_data=True,
    )


def _o2_scalar_draw(adf: Any, weight_expr: str, case: CaseSpec | None = None):
    """Independent scalar decomposition for one weight branch."""
    case = case or o2_oracle_case()
    spec = case.canonical_spec
    return adf.draw(
        spec["expr"],
        selection=spec["selection"],
        type="profile",
        bins=int(spec["bins"]),
        range=tuple(spec["range"]),
        weights=weight_expr,
        facet_by="side_type",
        fit="pol1",
        min_entries=1,
        auto_title=False,
        return_data=True,
    )


def _o2_profile_frame(branch_stats: Any, facet: int):
    if not isinstance(branch_stats, dict):
        raise HarnessError(f"O2 branch payload is {type(branch_stats).__name__}, expected dict")
    groups = list(branch_stats.get("groups", ()))
    if groups != list(HARDENING_O2_FACETS):
        raise HarnessError(f"O2 facet identity mismatch: observed groups={groups!r}")
    per_group = branch_stats.get("per_group")
    if not isinstance(per_group, dict):
        raise HarnessError("O2 branch has no faceted per_group payload")
    cell = per_group.get(str(facet))
    if not isinstance(cell, dict):
        raise HarnessError(f"O2 missing facet payload side_type={facet}")
    frame = cell.get("profile_data")
    if frame is None:
        raise HarnessError(f"O2 missing profile_data for side_type={facet}")
    required = ("x_center", "count", "sum_weights", "y_mean", "y_std", "y_sem")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise HarnessError(f"O2 profile_data missing columns {missing}")
    return frame


def _o2_fit_record(branch_stats: Any, facet: int) -> dict:
    fit_block = branch_stats.get("fit") if isinstance(branch_stats, dict) else None
    if not isinstance(fit_block, dict):
        raise HarnessError(f"O2 missing fit block for facet {facet}")
    candidates = ((str(facet),), (facet,), str(facet), facet)
    cell = None
    for key in candidates:
        if key in fit_block:
            cell = fit_block[key]
            break
    if cell is None:
        raise HarnessError(f"O2 fit block missing facet {facet}: keys={list(fit_block)}")
    try:
        record = cell[0][0]
    except Exception as exc:
        raise HarnessError(f"O2 malformed fit payload for facet {facet}: {cell!r}") from exc
    if record.get("fit_name") != "pol1" or record.get("fit_status") != "ok":
        raise HarnessError(f"O2 invalid pol1 fit for facet {facet}: {record}")
    params = np.asarray(record.get("params"), dtype=float)
    if params.shape != (2,) or not np.all(np.isfinite(params)):
        raise HarnessError(f"O2 invalid pol1 parameters for facet {facet}: {params}")
    return {"fit_name": "pol1", "params": params, "n_data": int(record.get("n_data", -1))}


def _o2_public_model(raw: Any, case: CaseSpec | None = None) -> dict:
    """Normalize public vector×facet stats into branch/facet keyed evidence."""
    case = case or o2_oracle_case()
    stats = raw[2] if isinstance(raw, tuple) and len(raw) >= 3 else None
    expected_branches = list(case.canonical_spec["weights_labels"] or [])
    if not isinstance(stats, list) or len(stats) != len(expected_branches):
        raise HarnessError(
            f"O2 product branch identity mismatch: expected {len(expected_branches)} branches, "
            f"got {type(stats).__name__} len={len(stats) if isinstance(stats, list) else 'n/a'}")
    cells = {}
    for branch_index, (branch_label, branch_stats) in enumerate(zip(expected_branches, stats)):
        for facet in HARDENING_O2_FACETS:
            frame = _o2_profile_frame(branch_stats, facet)
            cells[f"{branch_label}|side_type={facet}"] = {
                "branch_label": branch_label,
                "branch_index": branch_index,
                "facet": facet,
                "x_center": np.asarray(frame["x_center"], dtype=float),
                "count": np.asarray(frame["count"], dtype=int),
                "sum_weights": np.asarray(frame["sum_weights"], dtype=float),
                "y_mean": np.asarray(frame["y_mean"], dtype=float),
                "y_std": np.asarray(frame["y_std"], dtype=float),
                "y_sem": np.asarray(frame["y_sem"], dtype=float),
                "fit": _o2_fit_record(branch_stats, facet),
            }
    return {"cells": cells, "stats": stats, "axes": raw[1]}


def _o2_raw_tolerance(name: str) -> tuple[float, float]:
    """Tolerance for independent float64 reference versus source-dtype product."""
    if name == "sum_weights":
        return HARDENING_O2_SUMW_ATOL, HARDENING_O2_SUMW_RTOL
    if name in ("y_mean", "y_std", "y_sem"):
        return HARDENING_O2_VALUE_ATOL, HARDENING_O2_VALUE_RTOL
    return 1e-12, 1e-12


def _finite_summary(values: Any) -> dict:
    arr = np.asarray(values)
    try:
        numeric = np.asarray(arr, dtype=np.float64)
    except Exception:
        return {
            "dtype": str(arr.dtype),
            "size": int(arr.size),
            "n_nan": None,
            "n_posinf": None,
            "n_neginf": None,
        }
    return {
        "dtype": str(arr.dtype),
        "size": int(arr.size),
        "n_nan": int(np.count_nonzero(np.isnan(numeric))),
        "n_posinf": int(np.count_nonzero(np.isposinf(numeric))),
        "n_neginf": int(np.count_nonzero(np.isneginf(numeric))),
    }


def _numeric_reference_diagnostic(
        *, observable: str, reference_native: Any, reference_high_precision: Any,
        observed: Any, coordinates: Sequence[Any] | None = None,
        atol: float, rtol: float, dtype_metadata: Mapping[str, Any] | None = None,
        input_finiteness: Mapping[str, Any] | None = None) -> dict:
    """Serialize the numerical evidence requested by the dfdraw review panel.

    The two references use identical materialized values and semantic membership;
    only the reduction accumulator differs.  Ownership is never inferred merely
    from the fact that one comparison is red.
    """
    native = np.asarray(reference_native)
    high = np.asarray(reference_high_precision)
    got = np.asarray(observed)
    doc = {
        "observable": observable,
        "expected_shape": list(high.shape),
        "observed_shape": list(got.shape),
        "number_compared": int(high.size) if high.shape == got.shape else 0,
        "atol": float(atol),
        "rtol": float(rtol),
        "dtypes": dict(dtype_metadata or {}),
        "finiteness": dict(input_finiteness or {}),
    }
    if high.shape != got.shape or native.shape != got.shape:
        doc.update({
            "reference_native": {"ok": False, "reason": "shape mismatch"},
            "reference_high_precision": {"ok": False, "reason": "shape mismatch"},
            "number_finite": 0,
            "number_mismatched": int(max(high.size, got.size)),
            "first_mismatches": [],
        })
        return doc

    def _one(ref: np.ndarray) -> dict:
        r = np.asarray(ref, dtype=np.float64).reshape(-1)
        g = np.asarray(got, dtype=np.float64).reshape(-1)
        finite = np.isfinite(r) & np.isfinite(g)
        equal_nonfinite = ((np.isnan(r) & np.isnan(g)) |
                           (np.isposinf(r) & np.isposinf(g)) |
                           (np.isneginf(r) & np.isneginf(g)))
        close = np.isclose(r, g, rtol=rtol, atol=atol, equal_nan=True)
        mismatch = ~close
        diff = np.abs(r - g)
        denom = np.maximum(np.abs(r), np.finfo(np.float64).tiny)
        rel = diff / denom
        finite_diff = diff[finite]
        finite_rel = rel[finite]
        return {
            "ok": bool(np.all(close)),
            "number_finite": int(np.count_nonzero(finite)),
            "number_mismatched": int(np.count_nonzero(mismatch)),
            "max_abs": float(np.max(finite_diff)) if finite_diff.size else 0.0,
            "max_rel": float(np.max(finite_rel)) if finite_rel.size else 0.0,
            "median_abs": float(np.median(finite_diff)) if finite_diff.size else 0.0,
            "mismatch_mask": mismatch,
            "diff": diff,
            "rel": rel,
            "ref": r,
            "got": g,
        }

    native_doc = _one(native)
    high_doc = _one(high)
    mismatch_idx = np.flatnonzero(high_doc["mismatch_mask"])
    coords = list(coordinates or ())
    first = []
    for flat_index in mismatch_idx[:10]:
        coordinate = coords[int(flat_index)] if int(flat_index) < len(coords) else int(flat_index)
        first.append({
            "flat_index": int(flat_index),
            "coordinate": coordinate,
            "reference_native": float(native_doc["ref"][flat_index]),
            "reference_high_precision": float(high_doc["ref"][flat_index]),
            "observed": float(high_doc["got"][flat_index]),
            "abs_difference": float(high_doc["diff"][flat_index]),
            "relative_difference": float(high_doc["rel"][flat_index]),
        })
    for d in (native_doc, high_doc):
        d.pop("mismatch_mask", None)
        d.pop("diff", None)
        d.pop("rel", None)
        d.pop("ref", None)
        d.pop("got", None)
    doc.update({
        "number_finite": int(high_doc["number_finite"]),
        "number_mismatched": int(high_doc["number_mismatched"]),
        "reference_native": native_doc,
        "reference_high_precision": high_doc,
        "first_5_expected_high_precision": np.asarray(high, dtype=np.float64).reshape(-1)[:5].tolist(),
        "first_5_observed": np.asarray(got, dtype=np.float64).reshape(-1)[:5].tolist(),
        "first_5_absolute_differences": np.abs(
            np.asarray(high, dtype=np.float64).reshape(-1)[:5]
            - np.asarray(got, dtype=np.float64).reshape(-1)[:5]).tolist(),
        "first_mismatches": first,
    })
    return doc


def _numeric_owner_from_diagnostics(*, l1_ok: bool, diagnostics: Sequence[Mapping[str, Any]]) -> dict:
    """Separate divergence layer from ownership/reference-contract adjudication."""
    if not l1_ok:
        return {
            "first_disagreement_layer": "L1",
            "contract_reference_status": "VERIFIED",
            "owner_status": "ADF",
            "derived_owner": "ADF",
        }
    hp_ok = all(bool(d.get("reference_high_precision", {}).get("ok")) for d in diagnostics)
    native_ok = all(bool(d.get("reference_native", {}).get("ok")) for d in diagnostics)
    if hp_ok:
        owner = "NONE" if native_ok else "ORACLE"
        out = {
            "first_disagreement_layer": "NONE",
            "contract_reference_status": "VERIFIED",
            "owner_status": owner,
            "derived_owner": owner,
        }
        if owner == "ORACLE":
            out["resolution"] = "numeric reference accumulator corrected"
        return out
    return {
        "first_disagreement_layer": "L2",
        "contract_reference_status": "VERIFIED",
        "owner_status": "DFDRAW",
        "derived_owner": "dfdraw",
    }


def _o2_close_array(label: str, expected: Any, observed: Any, *, atol=1e-12, rtol=1e-12):
    a = np.asarray(expected)
    b = np.asarray(observed)
    if a.shape != b.shape:
        raise HarnessError(f"O2 {label} shape mismatch: {a.shape} != {b.shape}")
    if np.issubdtype(a.dtype, np.integer):
        if not np.array_equal(a, b):
            raise HarnessError(f"O2 {label} exact mismatch")
        return
    if not np.allclose(a.astype(float), b.astype(float), rtol=rtol, atol=atol, equal_nan=True):
        delta = np.nanmax(np.abs(a.astype(float) - b.astype(float)))
        raise HarnessError(f"O2 {label} numerical mismatch max_abs={delta}")


def _o2_assert_product_matches_raw(case: CaseSpec, expected: dict, product: dict) -> dict:
    """Primary O2 correctness check against independent raw arithmetic."""
    expected_keys = [
        f"{branch}|side_type={facet}"
        for branch in expected["weight_branches"] for facet in expected["facets"]
    ]
    if list(product["cells"].keys()) != expected_keys:
        raise HarnessError(
            f"O2 branch/facet identity mismatch expected={expected_keys} "
            f"observed={list(product['cells'])}")
    maxima = {}
    for key in expected_keys:
        ref = expected["by_branch_facet"][key]
        got = product["cells"][key]
        for name in ("x_center", "count", "sum_weights", "y_mean", "y_std", "y_sem"):
            atol, rtol = _o2_raw_tolerance(name)
            _o2_close_array(
                f"{key}.{name}", ref[name], got[name], atol=atol, rtol=rtol)
        diff = np.abs(np.asarray(ref["y_mean"], float) - np.asarray(got["y_mean"], float))
        maxima[key] = float(np.nanmax(diff)) if np.any(np.isfinite(diff)) else 0.0
    return {"cells": expected_keys, "max_abs_y_mean": maxima}


def _o2_assert_scalar_decomposition(adf: Any, case: CaseSpec, product: dict) -> dict:
    """Secondary composition invariant: vector branch == scalar weighted facet request."""
    records = []
    for branch_label, weight_expr in HARDENING_O2_WEIGHT_BRANCHES:
        scalar = _o2_scalar_draw(adf, weight_expr, case)
        scalar_stats = scalar[2]
        try:
            for facet in HARDENING_O2_FACETS:
                scalar_frame = _o2_profile_frame(scalar_stats, facet)
                got = product["cells"][f"{branch_label}|side_type={facet}"]
                for name in ("x_center", "count", "sum_weights", "y_mean", "y_std", "y_sem"):
                    _o2_close_array(
                        f"scalar {branch_label}|side_type={facet}.{name}",
                        scalar_frame[name].to_numpy(), got[name])
                scalar_fit = _o2_fit_record(scalar_stats, facet)
                _o2_close_array(
                    f"scalar fit {branch_label}|side_type={facet}",
                    scalar_fit["params"], got["fit"]["params"], atol=1e-10, rtol=1e-10)
                records.append({
                    "branch": branch_label, "facet": facet,
                    "fit_params": np.asarray(got["fit"]["params"], float).tolist(),
                })
        finally:
            try:
                plt.close(scalar[0])
            except Exception:
                pass
    return {"records": records}


def _o2_style_invariance(axes: Any) -> dict:
    styles = _profile_branch_styles_from_axes(axes, n_branches=len(HARDENING_O2_WEIGHT_BRANCHES))
    if set(styles) != set(HARDENING_O2_FACETS):
        raise HarnessError(f"O2 style facet identity mismatch: {sorted(styles)}")
    ref = styles[HARDENING_O2_FACETS[0]]
    mismatches = []
    for facet in HARDENING_O2_FACETS[1:]:
        for branch_index, (a, b) in enumerate(zip(ref, styles[facet])):
            if a != b:
                mismatches.append({
                    "branch": HARDENING_O2_WEIGHT_BRANCHES[branch_index][0],
                    "facet": facet, "reference": a, "observed": b,
                })
    if mismatches:
        raise HarnessError(f"O2 branch channel/style mismatch across facets: {mismatches}")
    return {"styles_by_facet": styles, "mismatches": []}


def _o2_execute_public_checks(adf: Any, case: CaseSpec | None = None) -> dict:
    """Execute O2 primary correctness + scalar + style checks on one prepared ADF."""
    case = case or o2_oracle_case()
    expected = _o2_expected_model(adf, case)
    raw = _o2_product_draw(adf, case)
    try:
        product = _o2_public_model(raw, case)
        primary = _o2_assert_product_matches_raw(case, expected, product)
        scalar = _o2_assert_scalar_decomposition(adf, case, product)
        style = _o2_style_invariance(product["axes"])
        return {"primary": primary, "scalar": scalar, "style": style, "product": product}
    finally:
        try:
            plt.close(raw[0])
        except Exception:
            pass


def _o2_flatten_for_observables(expected: dict, product: dict) -> tuple[dict, dict]:
    expected_flat = {
        "facet_values": list(expected["facets"]),
        "weight_branch_labels": list(expected["weight_branches"]),
        "count": [], "sum_weights": [], "y_mean": [], "y_std": [], "y_sem": [],
    }
    observed_flat = {
        "facet_values": list(expected["facets"]),
        "weight_branch_labels": list(expected["weight_branches"]),
        "count": [], "sum_weights": [], "y_mean": [], "y_std": [], "y_sem": [],
    }
    for branch in expected["weight_branches"]:
        for facet in expected["facets"]:
            key = f"{branch}|side_type={facet}"
            ref = expected["by_branch_facet"][key]
            got = product["cells"][key]
            for name in ("count", "sum_weights", "y_mean", "y_std", "y_sem"):
                expected_flat[name].extend(np.asarray(ref[name]).tolist())
                observed_flat[name].extend(np.asarray(got[name]).tolist())
    return expected_flat, observed_flat

def _o2_direct_product(adf: Any, case: CaseSpec) -> dict:
    owner=_DirectDFDrawOwner(adf.df)
    spec=case.canonical_spec
    raw=owner.draw(
        spec["expr"], selection=spec["selection"], type="profile",
        bins=int(spec["bins"]), range=tuple(spec["range"]),
        weights_vector=list(spec["weights_vector"]),
        weights_labels=list(spec.get("weights_labels") or ()),
        facet_by="side_type", vector_compose="outer", fit="pol1",
        min_entries=1, auto_title=False, return_data=True)
    try:
        return _o2_public_model(raw,case)
    finally:
        try: plt.close(raw[0])
        except Exception: pass


def _o2_numeric_recheck_diagnostics(adf: Any, case: CaseSpec, product: dict) -> dict:
    """Native-vs-float64 reduction diagnostics for ORACLE-03."""
    expected_native = _o2_expected_model(adf, case, accumulator="native")
    expected_high = _o2_expected_model(adf, case, accumulator="float64")
    df = adf.df
    records = []
    coordinates_by_observable = {name: [] for name in ("sum_weights", "y_mean", "y_std", "y_sem")}
    native_flat = {name: [] for name in coordinates_by_observable}
    high_flat = {name: [] for name in coordinates_by_observable}
    got_flat = {name: [] for name in coordinates_by_observable}

    for branch in expected_high["weight_branches"]:
        weights = _o2_weight_values(df, branch)
        for facet in expected_high["facets"]:
            key = f"{branch}|side_type={facet}"
            ref_n = expected_native["by_branch_facet"][key]
            ref_h = expected_high["by_branch_facet"][key]
            got = product["cells"][key]
            for b, center in enumerate(np.asarray(ref_h["x_center"], dtype=float)):
                coord = {
                    "branch": branch,
                    "facet": int(facet),
                    "bin_index": int(b),
                    "x_center": float(center),
                    "row_count": int(np.asarray(ref_h["count"])[b]),
                }
                for name in coordinates_by_observable:
                    coordinates_by_observable[name].append(coord)
                    native_flat[name].append(np.asarray(ref_n[name])[b])
                    high_flat[name].append(np.asarray(ref_h[name])[b])
                    got_flat[name].append(np.asarray(got[name])[b])

            # Branch/facet-level admission diagnostics requested by dfdraw.
            x = np.asarray(df["tgl"])
            y = np.asarray(df["dcar_tpc_vertex"])
            side = np.asarray(df["side_type"])
            ncl = np.asarray(df["ncl"])
            base = (ncl > 60) & (np.abs(y.astype(np.float64)) < 10) & (side < 2) & (side == facet)
            finite = base & np.isfinite(x) & np.isfinite(y) & np.isfinite(weights)
            in_range = finite & (x.astype(np.float64) >= HARDENING_O2_RANGE[0]) & (x.astype(np.float64) <= HARDENING_O2_RANGE[1])
            ww = weights[in_range]
            public_count = int(np.nansum(np.asarray(got["count"], dtype=np.float64)))
            public_sumw = float(np.nansum(np.asarray(got["sum_weights"], dtype=np.float64)))
            nonfinite_weight_rows = np.flatnonzero(base & np.isfinite(x) & np.isfinite(y) & ~np.isfinite(weights))[:20]
            edges = np.linspace(HARDENING_O2_RANGE[0], HARDENING_O2_RANGE[1], HARDENING_O2_BINS + 1)
            x64 = np.asarray(x, dtype=np.float64)
            bin_idx = np.searchsorted(edges, x64, side="right") - 1
            bin_idx[x64 == HARDENING_O2_RANGE[1]] = HARDENING_O2_BINS - 1
            got_counts = np.asarray(got["count"], dtype=np.float64)
            got_sumw = np.asarray(got["sum_weights"], dtype=np.float64)
            mismatch_bins = np.flatnonzero(
                np.isfinite(got_sumw) & ~np.isclose(got_counts, got_sumw, rtol=HARDENING_O2_SUMW_RTOL, atol=HARDENING_O2_SUMW_ATOL))
            mismatch_bin_rows = []
            for b in mismatch_bins[:10]:
                rows = np.flatnonzero(in_range & (bin_idx == b))[:20]
                mismatch_bin_rows.append({
                    "bin_index": int(b),
                    "x_center": float((edges[b] + edges[b + 1]) / 2.0),
                    "dfdraw_count": float(got_counts[b]),
                    "dfdraw_sum_weights": float(got_sumw[b]),
                    "count_minus_dfdraw_sum_weights": float(got_counts[b] - got_sumw[b]),
                    "affected_rows": [
                        {
                            "row_index": int(i),
                            "dcar_tpc_vertex": float(np.asarray(y, dtype=np.float64)[i]),
                            "evaluated_weight": float(np.asarray(weights, dtype=np.float64)[i]),
                            "isfinite_dcar": bool(np.isfinite(np.asarray(y, dtype=np.float64)[i])),
                            "isfinite_weight": bool(np.isfinite(np.asarray(weights, dtype=np.float64)[i])),
                        }
                        for i in rows
                    ],
                })
            records.append({
                "branch": branch,
                "facet": int(facet),
                "selected_row_count_in_profile_range": int(np.count_nonzero(in_range)),
                "finite_evaluated_weight_count_in_profile_range": int(np.count_nonzero(in_range & np.isfinite(weights))),
                "native_sum_weights": float(np.sum(ww)) if ww.size else 0.0,
                "float64_sum_weights": float(np.sum(np.asarray(ww, dtype=np.float64), dtype=np.float64)) if ww.size else 0.0,
                "dfdraw_count_total": public_count,
                "dfdraw_sum_weights_total": public_sumw,
                "count_minus_dfdraw_sum_weights": float(public_count - public_sumw),
                "weight_dtype": str(np.asarray(weights).dtype),
                "nonfinite_weight_row_indices": [int(i) for i in nonfinite_weight_rows],
                "nonfinite_weight_rows": [
                    {
                        "row_index": int(i),
                        "dcar_tpc_vertex": float(np.asarray(y, dtype=np.float64)[i]),
                        "evaluated_weight": float(np.asarray(weights, dtype=np.float64)[i]),
                        "isfinite_dcar": bool(np.isfinite(np.asarray(y, dtype=np.float64)[i])),
                        "isfinite_weight": bool(np.isfinite(np.asarray(weights, dtype=np.float64)[i])),
                    }
                    for i in nonfinite_weight_rows
                ],
                "mismatching_sumweight_bins": mismatch_bin_rows,
            })

    diagnostics = []
    for name in coordinates_by_observable:
        atol, rtol = _o2_raw_tolerance(name)
        diagnostics.append(_numeric_reference_diagnostic(
            observable=name,
            reference_native=native_flat[name],
            reference_high_precision=high_flat[name],
            observed=got_flat[name],
            coordinates=coordinates_by_observable[name],
            atol=atol,
            rtol=rtol,
            dtype_metadata={
                "raw_source_dtype_tgl": str(np.asarray(df["tgl"]).dtype),
                "raw_source_dtype_dcar_tpc_vertex": str(np.asarray(df["dcar_tpc_vertex"]).dtype),
                "adf_materialized_dtype_tgl": str(np.asarray(df["tgl"]).dtype),
                "adf_materialized_dtype_dcar_tpc_vertex": str(np.asarray(df["dcar_tpc_vertex"]).dtype),
                "oracle_accumulator_dtype": "float64",
                "dfdraw_effective_reduction_dtype": "float64 (source-reviewed contract)",
                "returned_statistic_dtype": str(np.asarray(got_flat[name]).dtype),
            },
            input_finiteness={
                "tgl": _finite_summary(df["tgl"]),
                "dcar_tpc_vertex": _finite_summary(df["dcar_tpc_vertex"]),
            },
        ))
    return {
        "case_id": case.case_id,
        "contract_reference_status": "VERIFIED",
        "reference_policy": "same materialized/evaluated values; independent float64 statistical accumulation",
        "comparisons": diagnostics,
        "branch_facet_admission": records,
    }


def _o2_ownership_ladder(adf: Any, case: CaseSpec, expected: dict) -> dict:
    df = adf.df
    diag = {
        "L0": "independent weighted profile over exact materialized/input values",
        "L1": "ADF physical tgl/dcar/side/ncl columns; no alias materialization in O2",
        "L1_all_within_tolerance": True,
        "source_dtypes": {
            "tgl": str(np.asarray(df["tgl"]).dtype),
            "dcar_tpc_vertex": str(np.asarray(df["dcar_tpc_vertex"]).dtype),
        },
    }
    try:
        direct = _o2_direct_product(adf, case)
        numeric = _o2_numeric_recheck_diagnostics(adf, case, direct)
        diag["numerical_recheck"] = numeric
        owner = _numeric_owner_from_diagnostics(
            l1_ok=True, diagnostics=numeric["comparisons"])
        diag.update({
            "L2": "direct DFDraw weights_vector×facet",
            "L2_matches_high_precision_truth": bool(
                all(d["reference_high_precision"]["ok"] for d in numeric["comparisons"])),
            "L3": "adf.draw numerical result evaluated against the same corrected contract",
            **owner,
        })
    except Exception as exc:
        diag.update({
            "L2": "direct DFDraw weights_vector×facet",
            "L2_matches_high_precision_truth": False,
            "L2_detail": f"{type(exc).__name__}: {exc}",
            "L3": "adf.draw numerical result requires adjudication",
            "first_disagreement_layer": "L2",
            "contract_reference_status": "UNRESOLVED",
            "owner_status": "UNRESOLVED",
            "derived_owner": "UNKNOWN",
        })
    return diag


def run_o2_realdata(case: CaseSpec, root_path: str, *, gallery_module=None,
                    prepared_adf=None, prepared_provenance=None) -> CaseResult:
    """Run O2 on the one shared FAST ADF; rebuilding the source is forbidden."""
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    try:
        if prepared_adf is None:
            raise HarnessError("O2 requires the shared FAST prepared_adf; rebuilding is forbidden")
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        provenance = _a6_4_prepared_fraction_sample_evidence(
            prepared_adf, prepared_provenance, root_path)
        expected = _o2_expected_model(prepared_adf, case)
        fig_fn = getattr(gallery, HARDENING_O2_GALLERY_FUNCTION, None)
        if not callable(fig_fn):
            raise HarnessError(f"O2 missing gallery owner {HARDENING_O2_GALLERY_FUNCTION}")
        raw = fig_fn(prepared_adf)
        try:
            product = _o2_public_model(raw, case)
            primary = _o2_assert_product_matches_raw(case, expected, product)
            scalar = _o2_assert_scalar_decomposition(prepared_adf, case, product)
            style = _o2_style_invariance(product["axes"])
            expected_flat, observed_flat = _o2_flatten_for_observables(expected, product)
            numeric_recheck = _o2_numeric_recheck_diagnostics(prepared_adf, case, product)
            numeric_owner = _numeric_owner_from_diagnostics(
                l1_ok=True, diagnostics=numeric_recheck["comparisons"])
            for obs in case.observables:
                res.observable_contract.append(_contract(obs))
                cmp = compare_observable(obs, expected_flat[obs.name], observed_flat[obs.name])
                res.comparisons.append(comparison_evidence(
                    obs, cmp, reference_label="raw NumPy/pandas", candidate_label="public fig44"))
                if not cmp.ok:
                    raise HarnessError(f"O2 observable mismatch {obs.name}: {cmp.detail}")
            res.executed_comparisons = len(case.observables)
            res.observed.update({
                "realdata_provenance": dict(prepared_provenance or provenance),
                "o2_oracle": {
                    "weight_branches": list(expected["weight_branches"]),
                    "facets": list(expected["facets"]),
                    "bins": int(expected["bins"]),
                    "range": list(expected["range"]),
                    "primary": primary,
                    "scalar_fit_records": scalar["records"],
                    "style_mismatches": style["mismatches"],
                },
                "ownership_ladder": {
                    "L0":"independent weighted profile over exact materialized/input values",
                    "L1":"ADF physical columns; no alias materialization in O2",
                    "L1_all_within_tolerance": True,
                    "L2":"public/direct numerical result matches corrected float64 contract",
                    "L3":"adf.draw matches corrected truth",
                    "numerical_recheck": numeric_recheck,
                    **numeric_owner,
                },
            })
            res.status = PASS
            res.detail = ""
            return res
        finally:
            try:
                plt.close(raw[0])
            except Exception:
                pass
    except Exception as exc:
        try:
            if prepared_adf is not None and 'expected' in locals():
                res.observed["ownership_ladder"]=_o2_ownership_ladder(prepared_adf,case,expected)
        except Exception as diag_exc:
            res.observed["ownership_ladder"]={
                "first_disagreement_layer":"UNKNOWN",
                "derived_owner":"UNKNOWN",
                "diagnostic_error":f"{type(diag_exc).__name__}: {diag_exc}",
            }
        res.status = FAIL
        res.detail = f"O2_WEIGHTS_VECTOR_FACET_CORRECTNESS FAIL: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        _close()
        res.wall_time_s = round(time.time() - t0, 4)



# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_77 injected-truth scientific oracle — primary six user workflows
# ─────────────────────────────────────────────────────────────────────────────

INJECTED_TRUTH_NOISE_SEED = 137
INJECTED_TRUTH_NOISE_SIGMA = 0.02
INJECTED_TRUTH_BINS = 36
INJECTED_TRUTH_RANGE = (-0.5, 35.5)
INJECTED_TRUTH_BASE_SEL = "(ncl>60)&(abs(dcar_tpc_vertex)<10)"
INJECTED_TRUTH_SIDE_SEL = INJECTED_TRUTH_BASE_SEL + "&(side_type<2)"
INJECTED_TRUTH_DELTA_EXPR = (
    "0.08*sin(2*np.pi*sector/36) + 0.03*tgl + 0.015*tgl*tgl"
)

INJECTED_TRUTH_VECTOR_CASE_ID = "I4-REAL-INJECTED-TRUTH-VECTOR-OVERLAY-EAGER-20PCT-01"
INJECTED_TRUTH_DELTA_CASE_ID = "I4-REAL-INJECTED-TRUTH-DIRECT-DELTA-EAGER-20PCT-01"
INJECTED_TRUTH_SELECTION_CASE_ID = "I4-REAL-INJECTED-TRUTH-SELECTION-DELTA-FACET-EAGER-20PCT-01"
INJECTED_TRUTH_WEIGHTS_CASE_ID = "I4-REAL-INJECTED-TRUTH-WEIGHTS-FACET-EAGER-20PCT-01"
INJECTED_TRUTH_GAUSS_CASE_ID = "I4-REAL-INJECTED-TRUTH-GAUSS-FIT-EAGER-20PCT-01"
INJECTED_TRUTH_FACET_CASE_ID = "I4-REAL-INJECTED-TRUTH-FACET-RESIDUAL-EAGER-20PCT-01"

INJECTED_TRUTH_VECTOR_GALLERY = "fig46_injected_truth_vector_overlay"
INJECTED_TRUTH_DELTA_GALLERY = "fig47_injected_truth_direct_delta"
INJECTED_TRUTH_SELECTION_GALLERY = "fig48_injected_truth_selection_delta_facet"
INJECTED_TRUTH_WEIGHTS_GALLERY = "fig49_injected_truth_weights_facet"
INJECTED_TRUTH_GAUSS_GALLERY = "fig50_injected_truth_gaussian_fit"
INJECTED_TRUTH_FACET_GALLERY = "fig51_injected_truth_facet_residual"

M1_HIST_CASE_ID = "I4-REAL-HIST-VECTOR-TRUTH-EAGER-20PCT-01"
M1_HIST_NORMALIZE_CASE_ID = "I4-HIST-NORMALIZE-EXPLICIT-CONTRACT-01"
M1_HIST_GALLERY_FUNCTION = "fig52_hist_vector_truth"
M2_GROUP_CASE_ID = "I4-REAL-GROUPING-TRUTH-EAGER-20PCT-01"
M2_GROUP_GALLERY_FUNCTION = "fig53_grouping_truth"
M3_SUBFRAME_CASE_ID = "I4-REAL-QUALIFIED-SUBFRAME-CHAIN-TRUTH-EAGER-20PCT-01"
M3_SUBFRAME_GALLERY_FUNCTION = "fig54_qualified_subframe_chain_truth"


# ── Human/AI review contract for scientific gallery pages ────────────────────
#
# The gallery has three proof classes:
#
#   PRIMARY ORACLE       independently known scientific truth; reviewer must
#                        actively validate the plotted relation.
#   CONSISTENCY ORACLE   agreement/invariance between public routes/states.
#   SMOKE / COVERAGE     execution/rendering coverage only.
#
# The lists below are deliberately concrete.  A generic instruction such as
# "validate the figure" is not sufficient evidence.

PRIMARY_ORACLE = "PRIMARY ORACLE"
CONSISTENCY_ORACLE = "CONSISTENCY ORACLE"
SMOKE_COVERAGE = "SMOKE / COVERAGE"

_PRIMARY_ORACLE_REVIEW_CHECKS = {
    HARDENING_O5_CASE_ID: (
        "In each side_type panel, compare the vertical early/late profile separation with the lower delta points; the sign and approximate magnitude must agree bin-by-bin.",
        "Confirm both side facets are present and the delta is not produced by swapping signal/reference branch order.",
        "Confirm the machine oracle uses raw NumPy/pandas early/late means and masks, not another ADF/dfdraw result.",
    ),
    A7_1_CASE_ID: (
        "Confirm two side_type panels are present and each contains exactly three time branches with their own pol1 fit.",
        "Confirm the six-row summary table has exactly one row for every branch×facet coordinate and that the fit row identity matches the visible branch/facet.",
        "Confirm the machine oracle checks independently constructed semantic coordinates/cardinality rather than only table formatting.",
    ),
    HARDENING_O2_CASE_ID: (
        "In each side_type panel, identify both weight branches and check that the weighted profile/fit shift is plausible rather than a duplicated unweighted curve.",
        "Confirm both weight branches exist in both facets and no branch/facet coordinate is missing or duplicated.",
        "Confirm the machine oracle compares count, sum(weights), weighted mean and fit decomposition against raw weighted NumPy/pandas arithmetic.",
    ),
    INJECTED_TRUTH_VECTOR_CASE_ID: (
        "Identify all four traces: original real DCA_r, clean-distorted, noisy-distorted and known_delta; none may be missing or silently merged.",
        "Check that clean-distorted minus original follows the stated known_delta trend, while noisy-distorted stays close to clean-distorted after sector averaging because the injected Gaussian has zero mean.",
        "Check the known_delta trace itself has the expected sector/tgl-driven structure and that the machine oracle compares every branch to independently binned raw truth.",
    ),
    INJECTED_TRUTH_DELTA_CASE_ID: (
        "The normalized lower/delta result must exist; absence of a normalized payload/panel is itself a FAIL.",
        "Check clean-distorted minus original follows the independently known_delta sector profile with the correct sign and branch order.",
        "Confirm the machine oracle compares the public normalized delta directly with raw mean(known_delta), not with another public profile.",
    ),
    INJECTED_TRUTH_SELECTION_CASE_ID: (
        "Confirm two side_type facets and both tgl<0 / tgl>=0 source branches are present.",
        "The normalized delta must be deliberately non-zero; its sign and sector dependence must match the visible separation of the two source profiles in each facet.",
        "Confirm the machine oracle independently recomputes branch/facet sector means and valid-bin masks from raw rows.",
    ),
    INJECTED_TRUTH_WEIGHTS_CASE_ID: (
        "Confirm each side_type facet contains both flat and tgl-dependent weight branches; a missing or duplicated branch is a FAIL.",
        "Check that the tgl-weighted profile differs from the flat profile where the known_delta/tgl correlation predicts a shift; this is not intended as a null test.",
        "Confirm count, sum(weights) and weighted means are checked against independent raw float arithmetic for every branch×facet×sector bin.",
    ),
    INJECTED_TRUTH_GAUSS_CASE_ID: (
        "Check the residual histogram is centered at approximately zero and the fitted Gaussian sigma is approximately 0.02.",
        "Read the fitted center and sigma from the page and compare them with the stated injected Gaussian truth; a visually good histogram without the expected fit parameters is insufficient.",
        "Confirm the machine oracle compares fitted center/sigma with raw selected injected-noise moments using the declared acceptance band.",
    ),
    INJECTED_TRUTH_FACET_CASE_ID: (
        "Confirm both side_type panels are present and residual means fluctuate around zero rather than showing a coherent sector-dependent bias.",
        "Check there is no systematic A/C-side offset or repeated sector structure larger than the expected statistical fluctuations.",
        "Confirm per-facet/per-sector counts and means are compared with the raw stable-row oracle_noise partition.",
    ),
    PRECLEAN_C1_CASE_ID: (
        "Confirm three side_type facets are visible and every facet contains both early and late histogram branches; 3×2 cardinality is load-bearing.",
        "Confirm the same-page non-faceted positive control contains the same two early/late branches and has not regressed while facet composition is exercised.",
        "Confirm the machine oracle compares facet identities, exact branch×facet bin counts and in-range row totals against independent raw NumPy histograms.",
    ),
    PRECLEAN_C2_CASE_ID: (
        "Confirm the page is the draw_figures short-form request and that figure-level defaults, not the conflicting top-level defaults, determine the 36-bin normalized result.",
        "Confirm the two qualified O4Surface selector branches are present with the declared signal/reference ordering and that the visible delta follows the independently computed side_type=0 minus side_type=1 known-delta profile.",
        "Confirm the machine oracle checks raw counts, branch means, delta, valid-bin mask, caller non-mutation, and preservation of the caller short-form string rather than only successful rendering.",
    ),
    PRECLEAN_C3_CASE_ID: (
        "Confirm the two y-vector source profiles are strictly positive and the second trace is visibly the declared 2× scaled companion of the first.",
        "Confirm the normalized ratio result exists and stays at 0.5 on every independently valid sector bin; absence of normalize_data is itself a FAIL.",
        "Confirm the machine oracle checks branch order, exact counts/mask, independent raw branch means, and public ratio values rather than only successful rendering.",
    ),
    PRECLEAN_C4_CASE_ID: (
        "Confirm EAGER and LAZY show both side_type groups for the same oracle_chain_with_shift:sector scientific request; no group may disappear or swap identity.",
        "Confirm the lower LAZY-EAGER residual is exactly zero on every jointly populated sector bin; any non-zero residual requires adjudication before introducing a tolerance.",
        "Confirm each loading mode is independently compared with the raw chained-alias plus qualified OracleShift truth, not only with the other mode.",
    ),
    M1_HIST_CASE_ID: (
        "On the left, identify early and late histogram branches; on the right, identify flat and tgl-weighted branches. No branch may silently disappear.",
        "Check that visibly different branch populations/weights produce correspondingly different bin heights rather than duplicated histograms.",
        "Confirm the machine oracle compares rendered histogram bin heights against independent np.histogram counts and weighted sums.",
    ),
    M2_GROUP_CASE_ID: (
        "On the left, confirm side_type groups are distinct; on the right, confirm four ordered tgl bins are present and produce distinct profile families.",
        "Check group labels/order correspond to the declared grouping rather than accidental row order.",
        "Confirm the machine oracle reconstructs group membership with raw pandas and compares per-group per-sector counts and means.",
    ),
    M3_SUBFRAME_CASE_ID: (
        "Confirm side_type groups show the known offset separation on top of the chained known_delta trend.",
        "Check the query explicitly contains both the chained alias and the qualified OracleShift.oracle_offset reference.",
        "Confirm the machine oracle computes the chained expression and subframe projection directly from raw rows without using ADF/dfdraw aggregation.",
    ),
}


def proof_class_for_case(case: "CaseSpec | None") -> str:
    """Return the review meaning of a gallery page, not merely its purpose."""
    if case is None:
        return SMOKE_COVERAGE
    if case.case_id in _PRIMARY_ORACLE_REVIEW_CHECKS:
        return PRIMARY_ORACLE
    if case.oracle_kind == "CONSISTENCY" or case.purpose == "INVARIANCE":
        return CONSISTENCY_ORACLE
    return SMOKE_COVERAGE


def reviewer_checks_for_case(case: "CaseSpec | None") -> tuple[str, ...]:
    """Concrete checks a human or multimodal AI reviewer must perform."""
    if case is None:
        return ()
    return tuple(_PRIMARY_ORACLE_REVIEW_CHECKS.get(case.case_id, ()))


def reviewer_contract_for_case(case: "CaseSpec | None") -> dict:
    return {
        "proof_class": proof_class_for_case(case),
        "reviewer_checks": list(reviewer_checks_for_case(case)),
    }


def _injected_truth_common_spec(gallery_function: str, injected_truth: str) -> dict:
    return {
        "gallery_function": gallery_function,
        "sample_fraction": A5_2_SAMPLE_FRACTION,
        "sample_seed": A5_2_SAMPLE_SEED,
        "noise_seed": INJECTED_TRUTH_NOISE_SEED,
        "noise_sigma": INJECTED_TRUTH_NOISE_SIGMA,
        "bins": INJECTED_TRUTH_BINS,
        "range": INJECTED_TRUTH_RANGE,
        "injected_truth": injected_truth,
    }


def injected_truth_vector_case(*, gallery_module=None) -> CaseSpec:
    cid = INJECTED_TRUTH_VECTOR_CASE_ID
    return CaseSpec(
        case_id=cid,
        claim_id="I4.injected_truth.vector_overlay",
        title="injected truth: native vector overlay matches independent raw known truth",
        claim=("the normal bracket-vector ADF/dfdraw workflow renders original, clean-distorted, "
               "noisy-distorted and known injected delta with the same per-sector means and "
               "populations computed independently from raw rows"),
        failure_means=("vector expression routing, alias evaluation, selection, binning or profile "
                       "reduction changed one of the four known-truth branches"),
        expected_visual=("four profile traces: original real DCA_r, noiseless distorted DCA_r, "
                         "noisy distorted DCA_r, and the analytically known injected delta"),
        owner_on_failure="ADF/dfdraw",
        purpose="CORRECTNESS",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            **_injected_truth_common_spec(
                INJECTED_TRUTH_VECTOR_GALLERY,
                "dcar_distorted_clean = dcar_tpc_vertex + known_delta; "
                "dcar_oracle = dcar_distorted_clean + stable-row Gaussian(0,0.02)"),
            "expr": "[dcar_tpc_vertex,dcar_distorted_clean,dcar_oracle,known_delta]:sector",
            "selection": INJECTED_TRUTH_BASE_SEL,
            "type": "profile",
        },
        applicable=True,
        setup_contract=("reuse the shared EAGER 20% ADF; install stable-row Gaussian noise and "
                        "reviewed aliases; execute the exact fig46 native vector expression"),
        preconditions=("oracle_row_id is unique", "sector/tgl/dcar/ncl are available"),
        figure_contract=FigureContract(
            expected_panels="one vector-overlay profile panel",
            panel_roles="four known-truth branches on the same sector axis",
            expected_traces="original, clean-distorted, noisy-distorted, known delta",
            expected_group_count="4 vector branches",
            primary_comparison="independent raw per-sector means/counts -> public vector profile_data",
            residual_definition="public branch profile mean - independently binned raw truth",
            accepted_envelope="counts exact; profile means within declared floating tolerance",
            case_ids=(cid,),
            proof_kind="CORRECTNESS",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable("counts", "INDEPENDENT", "ARRAY", "raw four-branch per-sector counts",
                       comparator="exact"),
            Observable("profile_means", "INDEPENDENT", "ARRAY", "raw four-branch per-sector means",
                       comparator="close", rtol=1e-8, atol=1e-9,
                       rationale="same selected rows and explicit sector bins; float reduction"),
        ),
        non_claims=("this case does not test normalization; Rank 2 owns that workflow",),
        negative_control="INJECTED_TRUTH:SWAPPED_VECTOR_BRANCH_OR_WRONG_DELTA_AMPLITUDE",
        reference_policy="same-process",
    )


def injected_truth_delta_case(*, gallery_module=None) -> CaseSpec:
    cid = INJECTED_TRUTH_DELTA_CASE_ID
    return CaseSpec(
        case_id=cid,
        claim_id="I4.injected_truth.direct_delta",
        title="injected truth: top-level draw() bracket-vector normalize=delta dispatch",
        claim=("top-level draw() bracket-vector profile must preserve typed-profile normalization; "
               "the clean-distorted minus original per-sector delta is independently known"),
        failure_means=("top-level draw() bracket-vector profile bypassed typed-profile normalization, "
                       "normalize='delta' was silently lost, branch order changed, or public delta "
                       "values differ from the independently known injected bias"),
        expected_visual="clean-distorted minus original follows the known_delta sector profile",
        owner_on_failure="dfdraw",
        purpose="CORRECTNESS",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            **_injected_truth_common_spec(
                INJECTED_TRUTH_DELTA_GALLERY,
                "dcar_distorted_clean - dcar_tpc_vertex = known_delta exactly row-by-row"),
            "expr": "[dcar_distorted_clean,dcar_tpc_vertex]:sector",
            "selection": INJECTED_TRUTH_BASE_SEL,
            "type": "profile",
            "normalize": "delta",
        },
        applicable=True,
        setup_contract="reuse the shared injected-truth ADF and execute exact fig47",
        preconditions=("the clean and original branches share identical selected rows",),
        figure_contract=FigureContract(
            expected_panels="public vector profile with a delta normalization result",
            panel_roles="clean/original source profiles plus their known delta",
            expected_traces="clean-distorted, original, and/or an explicit normalized delta",
            expected_group_count="2 vector branches",
            primary_comparison="raw mean(clean)-mean(original) -> public normalized delta",
            residual_definition="observed public delta - raw mean(known_delta)",
            accepted_envelope="sector-bin identity exact; floating delta within declared tolerance",
            case_ids=(cid,),
            proof_kind="CORRECTNESS",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable("delta_values", "INDEPENDENT", "ARRAY", "raw mean(known_delta) per sector",
                       comparator="close", rtol=1e-8, atol=1e-9,
                       rationale="known row-level identity reduced over explicit sector bins"),
        ),
        non_claims=("source-profile correctness is independently covered by Rank 1",),
        negative_control="INJECTED_TRUTH:NORMALIZE_IGNORED_OR_WRONG_DELTA_AMPLITUDE",
        reference_policy="same-process",
    )


def injected_truth_selection_case(*, gallery_module=None) -> CaseSpec:
    cid = INJECTED_TRUTH_SELECTION_CASE_ID
    return CaseSpec(
        case_id=cid,
        claim_id="I4.injected_truth.selection_delta_facet",
        title="injected truth: non-null selection_vector delta survives facet composition",
        claim=("tgl<0 minus tgl>=0 known_delta is recovered independently in each side_type facet"),
        failure_means=("selection_vector branches were dropped/swapped, facet identity changed, "
                       "or normalize='delta' produced the wrong non-zero answer"),
        expected_visual=("two side_type panels with a deliberately non-zero tgl<0 minus tgl>=0 "
                         "known_delta sector profile"),
        owner_on_failure="dfdraw",
        purpose="CORRECTNESS",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            **_injected_truth_common_spec(
                INJECTED_TRUTH_SELECTION_GALLERY,
                "known_delta depends on tgl, so mean(tgl<0)-mean(tgl>=0) is intentionally non-zero"),
            "expr": "known_delta:sector",
            "selection": INJECTED_TRUTH_SIDE_SEL,
            "selection_vector": ["tgl<0", "tgl>=0"],
            "normalize": "delta",
            "facet_by": "side_type",
            "type": "profile",
        },
        applicable=True,
        setup_contract="reuse shared injected truth; execute exact fig48 selection_vector+delta+facet",
        preconditions=("side_type 0/1 and both tgl-sign branches are populated",),
        figure_contract=FigureContract(
            expected_panels="two side_type facets with source branches and delta",
            panel_roles="side_type=0 and side_type=1",
            expected_traces="tgl<0, tgl>=0 and non-zero normalized delta",
            expected_group_count="2 selection branches × 2 facets",
            primary_comparison="raw branch/facet per-sector means -> normalize_data_faceted delta",
            residual_definition="public delta - independently computed branch-mean difference",
            accepted_envelope="facet/bin/mask identity exact; numerical delta within tolerance",
            case_ids=(cid,),
            proof_kind="CORRECTNESS",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable("delta_values", "INDEPENDENT", "ARRAY",
                       "raw tgl<0 minus tgl>=0 known_delta by facet/sector",
                       comparator="close", rtol=1e-8, atol=1e-9,
                       rationale="same explicitly selected rows and arithmetic means"),
            Observable("valid_mask", "INDEPENDENT", "ARRAY",
                       "raw bins populated by both vector branches", comparator="exact"),
        ),
        non_claims=("zero-valued differential tests are intentionally not used here",),
        negative_control="INJECTED_TRUTH:SWAPPED_SELECTION_BRANCH",
        reference_policy="same-process",
    )


def injected_truth_weights_case(*, gallery_module=None) -> CaseSpec:
    cid = INJECTED_TRUTH_WEIGHTS_CASE_ID
    return CaseSpec(
        case_id=cid,
        claim_id="I4.injected_truth.weights_facet",
        title="injected truth: weights_vector known weighted profile in side facets",
        claim=("flat and |tgl|-modulated weights recover independently calculated weighted "
               "known_delta profiles in each side_type facet"),
        failure_means=("weights_vector identity, facet routing, weighted numerator or weighted "
                       "normalization differs from raw sum(w*y)/sum(w) truth"),
        expected_visual="two facets with flat and tgl-weighted known_delta profile branches",
        owner_on_failure="dfdraw",
        purpose="CORRECTNESS",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            **_injected_truth_common_spec(
                INJECTED_TRUTH_WEIGHTS_GALLERY,
                "oracle_w_flat=1; oracle_w_tgl=1+0.30*abs(tgl); truth=sum(w*known_delta)/sum(w)"),
            "expr": "known_delta:sector",
            "selection": INJECTED_TRUTH_SIDE_SEL,
            "weights_vector": ["oracle_w_flat", "oracle_w_tgl"],
            "facet_by": "side_type",
            "type": "profile",
        },
        applicable=True,
        setup_contract="reuse shared injected truth; execute exact fig49 weights_vector+facet",
        preconditions=("weights are positive", "side_type 0/1 are populated"),
        figure_contract=FigureContract(
            expected_panels="two side_type facets",
            panel_roles="side_type=0 and side_type=1",
            expected_traces="flat-weight and tgl-weight known_delta profiles",
            expected_group_count="2 weight branches × 2 facets",
            primary_comparison="raw sum(w*y)/sum(w) and sum(w) -> public weighted profile_data",
            residual_definition="public weighted mean - independently computed weighted mean",
            accepted_envelope=("counts exact; sum_weights and weighted means compared with tight "
                               "declared tolerance; do not widen merely to obtain green"),
            case_ids=(cid,),
            proof_kind="CORRECTNESS",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable("count", "INDEPENDENT", "ARRAY", "raw weighted-cell row counts",
                       comparator="exact"),
            Observable("sum_weights", "INDEPENDENT", "ARRAY", "raw per-bin sum(weights)",
                       comparator="close", rtol=1e-7, atol=1e-8,
                       rationale="independent float64 weighted accumulation"),
            Observable("y_mean", "INDEPENDENT", "ARRAY", "raw sum(w*known_delta)/sum(w)",
                       comparator="close", rtol=1e-7, atol=1e-8,
                       rationale="independent float64 weighted mean"),
        ),
        non_claims=("a red result is evidence to investigate, not a reason to widen tolerance",),
        negative_control="INJECTED_TRUTH:WEIGHTED_NUMERATOR_MUTATION",
        reference_policy="same-process",
    )


def injected_truth_gauss_case(*, gallery_module=None) -> CaseSpec:
    cid = INJECTED_TRUTH_GAUSS_CASE_ID
    return CaseSpec(
        case_id=cid,
        claim_id="I4.injected_truth.gaussian_fit",
        title="injected truth: Gaussian residual fit recovers selected known noise sample",
        claim=("oracle_residual is exactly the injected stable-row Gaussian noise and the public "
               "histogram fit recovers its selected-sample center and width"),
        failure_means=("alias arithmetic, histogram selection/range or Gaussian fit extraction "
                       "is inconsistent with the known injected residual distribution"),
        expected_visual="Gaussian residual centered near zero with sigma near 0.02",
        owner_on_failure="ADF/dfdraw",
        purpose="CORRECTNESS",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            **_injected_truth_common_spec(
                INJECTED_TRUTH_GAUSS_GALLERY,
                "oracle_residual = oracle_noise; stable-row Gaussian seed=137 sigma=0.02"),
            "expr": "oracle_residual",
            "selection": INJECTED_TRUTH_BASE_SEL,
            "type": "hist",
            "fit": "gauss",
            "bins": 100,
            "range": (-0.1, 0.1),
        },
        applicable=True,
        setup_contract="reuse shared injected truth; execute exact fig50 residual histogram+gauss",
        preconditions=("selected residual sample is non-empty",),
        figure_contract=FigureContract(
            expected_panels="one residual histogram with Gaussian fit",
            panel_roles="selected stable-row injected noise distribution",
            expected_traces="histogram plus gauss fit",
            expected_group_count="1",
            primary_comparison="public fitted center/sigma -> raw selected injected-noise mean/std",
            residual_definition="fit parameter - raw selected-noise moment",
            accepted_envelope=("center and sigma must lie inside a binning/statistics-derived "
                               "acceptance band recorded by the runner"),
            case_ids=(cid,),
            proof_kind="CORRECTNESS",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable("fit_center_delta", "INDEPENDENT", "FLAT",
                       "abs(public gauss center - raw selected noise mean) / allowed tolerance",
                       comparator="close", rtol=0.0, atol=1.0,
                       rationale="dimensionless deviation must be <=1 derived tolerance unit"),
            Observable("fit_sigma_delta", "INDEPENDENT", "FLAT",
                       "abs(public gauss sigma - raw selected noise std) / allowed tolerance",
                       comparator="close", rtol=0.0, atol=1.0,
                       rationale="dimensionless deviation must be <=1 derived tolerance unit"),
        ),
        non_claims=("the generator's nominal 0/0.02 values are human sanity; selected raw moments are strict reference",),
        negative_control="INJECTED_TRUTH:WRONG_EXPECTED_SIGMA",
        reference_policy="same-process",
    )


def injected_truth_facet_case(*, gallery_module=None) -> CaseSpec:
    cid = INJECTED_TRUTH_FACET_CASE_ID
    return CaseSpec(
        case_id=cid,
        claim_id="I4.injected_truth.facet_residual",
        title="injected truth: simple side-facet residual profile matches raw injected noise",
        claim=("basic facet partitioning preserves the independently known oracle_residual "
               "profile in each side_type×sector cell"),
        failure_means=("facet partitioning or profile reduction differs from raw injected-noise truth"),
        expected_visual="two side_type panels with residual means fluctuating around zero",
        owner_on_failure="dfdraw",
        purpose="CORRECTNESS",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS",
        loading_mode="EAGER",
        sample_mode="FRACTION",
        canonical_spec={
            **_injected_truth_common_spec(
                INJECTED_TRUTH_FACET_GALLERY,
                "oracle_residual = stable-row injected Gaussian noise exactly"),
            "expr": "oracle_residual:sector",
            "selection": INJECTED_TRUTH_SIDE_SEL,
            "facet_by": "side_type",
            "type": "profile",
        },
        applicable=True,
        setup_contract="reuse shared injected truth; execute exact fig51 simple facet profile",
        preconditions=("side_type 0/1 are populated",),
        figure_contract=FigureContract(
            expected_panels="two side_type facets",
            panel_roles="side_type=0 and side_type=1",
            expected_traces="one residual profile per facet",
            expected_group_count="2 facets",
            primary_comparison="raw oracle_noise profile -> public facet profile_data",
            residual_definition="public facet-bin residual mean - raw injected-noise mean",
            accepted_envelope="counts exact; means within declared numerical tolerance",
            case_ids=(cid,),
            proof_kind="CORRECTNESS",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable("count", "INDEPENDENT", "ARRAY", "raw facet/sector counts",
                       comparator="exact"),
            Observable("y_mean", "INDEPENDENT", "ARRAY", "raw facet/sector oracle_noise means",
                       comparator="close", rtol=1e-8, atol=1e-9,
                       rationale="same explicitly selected injected noise rows"),
        ),
        non_claims=("complex vector/facet composition is owned by Ranks 3 and 4",),
        negative_control="INJECTED_TRUTH:FACET_IDENTITY_MUTATION",
        reference_policy="same-process",
    )


INJECTED_TRUTH_GALLERY_CASES = {
    INJECTED_TRUTH_VECTOR_GALLERY: injected_truth_vector_case,
    INJECTED_TRUTH_DELTA_GALLERY: injected_truth_delta_case,
    INJECTED_TRUTH_SELECTION_GALLERY: injected_truth_selection_case,
    INJECTED_TRUTH_WEIGHTS_GALLERY: injected_truth_weights_case,
    INJECTED_TRUTH_GAUSS_GALLERY: injected_truth_gauss_case,
    INJECTED_TRUTH_FACET_GALLERY: injected_truth_facet_case,
}


def m1_hist_vector_case(*, gallery_module=None) -> CaseSpec:
    cid = M1_HIST_CASE_ID
    return CaseSpec(
        case_id=cid,
        claim_id="I4.commissioning.hist_vector_truth",
        title="hist selection_vector and weights_vector match independent NumPy histograms",
        claim=("the supported non-profile vector dispatch preserves selection and weight branch "
               "semantics down to rendered histogram bin contents"),
        failure_means=("hist vector dispatch dropped/swapped a branch, changed selection membership, "
                       "or changed weighted bin sums"),
        expected_visual="left: early/late ncl histograms; right: flat/tgl-weighted ncl histograms",
        owner_on_failure="dfdraw",
        purpose="CORRECTNESS", gate="ENVIRONMENT_GATED", oracle_kind="CORRECTNESS",
        loading_mode="EAGER", sample_mode="FRACTION",
        canonical_spec={
            "gallery_function": M1_HIST_GALLERY_FUNCTION,
            "public_query": (
                "left adf.draw('ncl', type='hist', selection_vector=['early','late'], vector_compose='outer'); "
                "right adf.draw('ncl', type='hist', weights_vector=['flat','tgl-weighted'], vector_compose='outer')"),
            "type": "hist", "bins": 20, "range": (80.0, 160.0),
        },
        setup_contract="reuse shared EAGER 20% ADF and injected deterministic weights",
        preconditions=("ncl/time_s/tgl finite support is non-empty",),
        figure_contract=FigureContract(
            expected_panels="two histogram panels", panel_roles="selection vector; weight vector",
            expected_traces="early/late; flat/tgl-weighted", expected_group_count="2 + 2 branches",
            primary_comparison="np.histogram raw counts/weighted sums -> rendered bar heights",
            residual_definition="rendered bin height - independent NumPy bin content",
            accepted_envelope="bin edges fixed; unweighted counts exact; weighted sums tight float tolerance",
            case_ids=(cid,), proof_kind="CORRECTNESS"),
        surfaces_under_test=("draw",),
        observables=(
            Observable("selection_bin_counts", "INDEPENDENT", "ARRAY", "np.histogram selected counts", comparator="exact"),
            Observable("weighted_bin_sums", "INDEPENDENT", "ARRAY", "np.histogram weighted sums",
                       comparator="close", rtol=1e-10, atol=1e-8,
                       rationale="matplotlib histogram heights from float64 weighted accumulation"),
        ),
        non_claims=("hist normalize is a separate explicit contract",),
        negative_control="M1:SWAP_SELECTION_OR_DROP_WEIGHT_BRANCH", reference_policy="same-process",
    )


def m1_hist_normalize_case() -> CaseSpec:
    cid = M1_HIST_NORMALIZE_CASE_ID
    return CaseSpec(
        case_id=cid, claim_id="I4.commissioning.hist_normalize_contract",
        title="hist normalize semantic is explicit: supported numerically or loud refusal",
        claim=("profile-style normalize= must never be silently accepted and ignored by hist; "
               "the current commissioning contract requires deterministic loud refusal unless support is implemented"),
        failure_means="hist normalize was silently dropped or ambiguously accepted",
        expected_visual="no gallery page; machine-only semantic contract",
        owner_on_failure="dfdraw", purpose="ERROR_CONTRACT", gate="CORE_MANDATORY",
        oracle_kind="ERROR_CONTRACT", loading_mode="EAGER", sample_mode="SYNTHETIC",
        canonical_spec={"expr":"ncl", "type":"hist", "normalize":"delta"},
        setup_contract="execute one histogram request with profile-style normalize='delta'",
        preconditions=("histogram dispatch available",),
        figure_contract=FigureContract(
            expected_panels="none", panel_roles="machine-only refusal contract",
            expected_traces="none", expected_group_count="0",
            primary_comparison="public call -> deterministic explicit refusal",
            residual_definition="not applicable",
            accepted_envelope="exception must explicitly mention normalize/hist unsupported semantics",
            case_ids=(cid,), proof_kind="ERROR_CONTRACT"),
        surfaces_under_test=("draw",), observables=(
            Observable("loud_refusal", "INDEPENDENT", "FLAT", "explicit normalize refusal", comparator="exact"),
        ),
        non_claims=("this does not claim histogram normalize is supported",),
        known_bug_status="KNOWN_BUG",
        known_bug_id="BUG_dfdraw_20260912_hist_normalize_ignored",
        negative_control="M1:NORMALIZE_SILENTLY_IGNORED", reference_policy="same-process",
    )


def m2_grouping_case(*, gallery_module=None) -> CaseSpec:
    cid=M2_GROUP_CASE_ID
    return CaseSpec(
        case_id=cid, claim_id="I4.commissioning.grouping_truth",
        title="categorical group_by and numeric group_by_bins match raw pandas grouping truth",
        claim="production-shaped grouping preserves independently reconstructed membership, counts and means",
        failure_means="group membership/order, bin membership, counts or profile means differ from raw pandas truth",
        expected_visual="left side_type groups; right four ordered tgl bins",
        owner_on_failure="dfdraw", purpose="CORRECTNESS", gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS", loading_mode="EAGER", sample_mode="FRACTION",
        canonical_spec={"gallery_function":M2_GROUP_GALLERY_FUNCTION,
                        "public_query":"adf.draw('known_delta:sector', type='profile', group_by='side_type'); adf.draw(..., group_by='tgl', group_by_bins=4)",
                        "expr":"known_delta:sector", "type":"profile", "group_by":"side_type / tgl", "group_by_bins":4},
        setup_contract="reuse injected-truth ADF; execute categorical and binned grouping on one page",
        preconditions=("side_type 0/1 populated", "tgl selected domain has four non-empty bins"),
        figure_contract=FigureContract(
            expected_panels="two grouping panels", panel_roles="categorical groups; numeric bins",
            expected_traces="2 side groups; 4 tgl-bin groups", expected_group_count="2 + 4 groups",
            primary_comparison="raw pandas group membership and per-sector means -> public profile_data",
            residual_definition="public group-bin count/mean - raw grouped truth",
            accepted_envelope="group cardinality/order and counts exact; means tight float tolerance",
            case_ids=(cid,), proof_kind="CORRECTNESS"),
        surfaces_under_test=("draw",),
        observables=(
            Observable("group_counts", "INDEPENDENT", "ARRAY", "raw grouped per-sector counts", comparator="exact"),
            Observable("group_means", "INDEPENDENT", "ARRAY", "raw grouped per-sector known_delta means",
                       comparator="close", rtol=1e-8, atol=1e-9,
                       rationale="same selected rows and explicit sector bins"),
        ),
        non_claims=("group_by_quantiles is not additionally required because Round-2 permits bins OR quantiles",),
        negative_control="M2:GROUP_MEMBERSHIP_MUTATION", reference_policy="same-process")


def m3_subframe_chain_case(*, gallery_module=None) -> CaseSpec:
    cid=M3_SUBFRAME_CASE_ID
    return CaseSpec(
        case_id=cid, claim_id="I4.commissioning.qualified_subframe_chain_truth",
        title="qualified subframe plus chained aliases matches independent raw truth",
        claim=("one production-shaped public draw resolves a two-level parent alias chain and a qualified "
               "subframe projection without changing the independently known expression"),
        failure_means="alias chaining, subframe projection, qualified-reference resolution or grouping changed the known truth",
        expected_visual="two side_type groups separated by deterministic OracleShift offsets on the chained known_delta trend",
        owner_on_failure="ADF", purpose="CORRECTNESS", gate="ENVIRONMENT_GATED",
        oracle_kind="CORRECTNESS", loading_mode="EAGER", sample_mode="FRACTION",
        canonical_spec={"gallery_function":M3_SUBFRAME_GALLERY_FUNCTION,
                        "public_query":"adf.draw('oracle_chain_l2+OracleShift.oracle_offset:sector', type='profile', group_by='side_type', ...)",
                        "expr":"oracle_chain_l2+OracleShift.oracle_offset:sector", "type":"profile", "group_by":"side_type"},
        setup_contract="register OracleShift(side_type->offset), add two chained aliases, execute ordinary draw",
        preconditions=("side_type key domain fully covered by subframe",),
        figure_contract=FigureContract(
            expected_panels="one grouped profile panel", panel_roles="side_type groups with qualified offset",
            expected_traces="side_type=0 and side_type=1", expected_group_count="2",
            primary_comparison="raw chained formula + raw side offset -> public grouped profile_data",
            residual_definition="public grouped mean - independently computed row formula",
            accepted_envelope="counts exact; means tight float tolerance",
            case_ids=(cid,), proof_kind="CORRECTNESS"),
        surfaces_under_test=("draw",),
        observables=(
            Observable("count", "INDEPENDENT", "ARRAY", "raw side/sector selected counts", comparator="exact"),
            Observable("y_mean", "INDEPENDENT", "ARRAY", "raw chained+qualified expression mean",
                       comparator="close", rtol=1e-8, atol=1e-9,
                       rationale="same selected rows and deterministic subframe projection"),
        ),
        non_claims=("draw_figures orchestration is already covered by consistency tests; this case is correctness-first",),
        negative_control="M3:QUALIFIED_OFFSET_OR_ALIAS_CHAIN_MUTATION", reference_policy="same-process")


def commissioning_addition_cases(*, gallery_module=None) -> tuple[CaseSpec, ...]:
    return (m1_hist_vector_case(gallery_module=gallery_module),
            m2_grouping_case(gallery_module=gallery_module),
            m3_subframe_chain_case(gallery_module=gallery_module))


def injected_truth_cases(*, gallery_module=None) -> tuple[CaseSpec, ...]:
    return (
        injected_truth_vector_case(gallery_module=gallery_module),
        injected_truth_delta_case(gallery_module=gallery_module),
        injected_truth_selection_case(gallery_module=gallery_module),
        injected_truth_weights_case(gallery_module=gallery_module),
        injected_truth_gauss_case(gallery_module=gallery_module),
        injected_truth_facet_case(gallery_module=gallery_module),
    )


def _it_stable_noise(row_ids: Any) -> np.ndarray:
    """Independent implementation of the declared stable-row Gaussian generator."""
    row_ids = np.asarray(row_ids, dtype=np.int64)
    if row_ids.ndim != 1 or len(np.unique(row_ids)) != len(row_ids):
        raise HarnessError("injected-truth oracle_row_id must be unique and one-dimensional")

    def mix64(x):
        x = np.asarray(x, dtype=np.uint64)
        with np.errstate(over="ignore"):
            x = x + np.uint64(0x9E3779B97F4A7C15)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))

    keys = row_ids.astype(np.uint64, copy=False) ^ np.uint64(INJECTED_TRUTH_NOISE_SEED)
    h1 = mix64(keys)
    h2 = mix64(keys ^ np.uint64(0xD1B54A32D192ED03))
    u1 = ((h1 >> np.uint64(11)).astype(np.float64) + 0.5) / float(2**53)
    u2 = ((h2 >> np.uint64(11)).astype(np.float64) + 0.5) / float(2**53)
    return INJECTED_TRUTH_NOISE_SIGMA * (
        np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2))


def _it_raw_model(adf: Any) -> dict:
    if not hasattr(adf, "df"):
        raise HarnessError("injected-truth prepared object has no raw dataframe")
    df = adf.df
    required = ("oracle_row_id", "oracle_noise", "sector", "tgl",
                "dcar_tpc_vertex", "ncl", "side_type")
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise HarnessError(f"injected-truth raw frame missing {missing}")
    row_id = np.asarray(df["oracle_row_id"], dtype=np.int64)
    expected_noise = _it_stable_noise(row_id)
    noise = np.asarray(df["oracle_noise"], dtype=float)
    if not np.array_equal(noise, expected_noise):
        raise HarnessError("oracle_noise is not the declared stable-row Gaussian sequence")
    sector = np.asarray(df["sector"], dtype=float)
    tgl = np.asarray(df["tgl"], dtype=float)
    dcar = np.asarray(df["dcar_tpc_vertex"], dtype=float)
    ncl = np.asarray(df["ncl"], dtype=float)
    side = np.asarray(df["side_type"])
    delta = 0.08 * np.sin(2.0 * np.pi * sector / 36.0) + 0.03 * tgl + 0.015 * tgl * tgl
    clean = dcar + delta
    noisy = clean + noise

    # Real detector frames legitimately contain non-finite/out-of-domain rows.
    # The public profile/histogram paths drop non-finite x/y values after the
    # user selection, so the independent row-level identity must be checked on
    # the same finite scientific domain rather than across every source row.
    base = (ncl > 60) & (np.abs(dcar) < 10)
    truth_domain = (
        base
        & np.isfinite(sector)
        & np.isfinite(tgl)
        & np.isfinite(dcar)
        & np.isfinite(delta)
        & np.isfinite(clean)
        & np.isfinite(noisy)
        & np.isfinite(noise)
    )
    if not np.any(truth_domain):
        raise HarnessError("injected-truth finite selected domain is empty")

    residual = noisy - dcar - delta
    if not np.allclose(
            residual[truth_domain], noise[truth_domain],
            rtol=0.0, atol=1e-14):
        finite_delta = np.abs(residual[truth_domain] - noise[truth_domain])
        raise HarnessError(
            "independent injected-truth row algebra is inconsistent on finite "
            f"selected rows; max_abs={float(np.max(finite_delta))}")

    side_sel = base & (side < 2)
    return {
        "df": df, "sector": sector, "tgl": tgl, "dcar": dcar, "side": side,
        "delta": delta, "clean": clean, "noisy": noisy, "noise": noise,
        "residual": residual, "base": base, "side_sel": side_sel,
        "truth_domain": truth_domain,
        "truth_domain_rows": int(np.count_nonzero(truth_domain)),
        "w_flat": np.ones(len(df), dtype=float),
        "w_tgl": 1.0 + 0.30 * np.abs(tgl),
    }


def _it_cast_expected_to_authoritative_dtype(df: Any, column: str, values: Any) -> np.ndarray:
    """Cast independent values to an already-materialized authoritative dtype.

    AD-19/13.76.ADF makes the first successful materialization dtype authoritative
    for aliases without an explicit dtype.  Using that dtype metadata is not using
    the ADF values as truth: the numerical values below are still built independently
    from physical source columns and the declared formula.
    """
    if column not in df.columns:
        raise HarnessError(f"authoritative materialized column {column!r} is absent")
    target = np.asarray(df[column]).dtype
    try:
        return np.asarray(values).astype(target, copy=False)
    except Exception as exc:
        raise HarnessError(
            f"cannot cast independent {column!r} truth to authoritative dtype {target}") from exc


def _it_semantic_model(adf: Any) -> dict:
    """Independent injected-truth model with authoritative stepwise dtypes.

    `_it_raw_model` remains the ideal float64 analytic L0 truth.  This model is
    the independent reference for public ADF/dfdraw semantics after alias
    materialization: every alias step is recomputed from source columns and then
    cast only to the authoritative materialized dtype required by AD-19.
    """
    raw = _it_raw_model(adf)
    df = adf.df
    sector = np.asarray(df["sector"])
    tgl = np.asarray(df["tgl"])
    dcar = np.asarray(df["dcar_tpc_vertex"])
    noise = np.asarray(df["oracle_noise"])

    delta_formula = (
        0.08 * np.sin(2.0 * np.pi * sector / 36.0)
        + 0.03 * tgl + 0.015 * tgl * tgl)
    delta = _it_cast_expected_to_authoritative_dtype(
        df, "known_delta", delta_formula)
    clean = _it_cast_expected_to_authoritative_dtype(
        df, "dcar_distorted_clean", dcar + delta)
    noisy = _it_cast_expected_to_authoritative_dtype(
        df, "dcar_oracle", clean + noise)
    residual = _it_cast_expected_to_authoritative_dtype(
        df, "oracle_residual", noisy - dcar - delta)

    out = dict(raw)
    out.update({
        "sector": sector, "tgl": tgl, "dcar": dcar,
        "delta": delta, "clean": clean, "noisy": noisy,
        "noise": noise, "residual": residual,
    })
    return out


def _it_profile(x: Any, y: Any, mask: Any, *, weights: Any = None,
                accumulator: str = "float64") -> dict:
    """Independent profile reduction over unchanged materialized values.

    ``float64`` is the corrected contract reference.  ``native`` is retained
    only as diagnostic evidence showing whether the historical oracle red came
    from accumulator precision rather than semantic membership.
    """
    if accumulator not in {"float64", "native"}:
        raise HarnessError(f"unknown injected-truth accumulator {accumulator!r}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(x) & np.isfinite(y)
    lo, hi = INJECTED_TRUTH_RANGE
    edges = np.linspace(lo, hi, INJECTED_TRUTH_BINS + 1)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx[x == hi] = INJECTED_TRUTH_BINS - 1
    inside = mask & (idx >= 0) & (idx < INJECTED_TRUTH_BINS)
    centers = (edges[:-1] + edges[1:]) / 2.0
    count = np.zeros(INJECTED_TRUTH_BINS, dtype=int)
    mean = np.full(INJECTED_TRUTH_BINS, np.nan, dtype=float)
    sum_weights = np.full(INJECTED_TRUTH_BINS, np.nan, dtype=float)
    w = None if weights is None else np.asarray(weights)
    if w is not None:
        inside &= np.isfinite(w)
    for b in range(INJECTED_TRUTH_BINS):
        take = inside & (idx == b)
        yy_native = y[take]
        count[b] = int(len(yy_native))
        if w is None:
            sum_weights[b] = float(len(yy_native))
            if len(yy_native):
                if accumulator == "float64":
                    yy = np.asarray(yy_native, dtype=np.float64)
                    mean[b] = float(np.mean(yy, dtype=np.float64))
                else:
                    mean[b] = float(np.mean(yy_native))
        else:
            ww_native = w[take]
            if len(ww_native):
                if accumulator == "float64":
                    ww = np.asarray(ww_native, dtype=np.float64)
                    yy = np.asarray(yy_native, dtype=np.float64)
                    sw = float(np.sum(ww, dtype=np.float64))
                    numerator = float(np.sum(ww * yy, dtype=np.float64))
                else:
                    ww = ww_native
                    yy = yy_native
                    sw_native = np.sum(ww)
                    sw = float(sw_native)
                    numerator = float(np.sum(ww * yy))
                sum_weights[b] = sw
                if sw > 0.0:
                    mean[b] = numerator / sw
    return {
        "x_center": centers,
        "count": count,
        "sum_weights": sum_weights,
        "y_mean": mean,
        "accumulator": accumulator,
    }


def _it_profile_frame(stats: Any):
    if not isinstance(stats, dict):
        raise HarnessError(f"injected-truth profile stats are {type(stats).__name__}, expected dict")
    frame = stats.get("profile_data")
    if frame is None:
        raise HarnessError("injected-truth public profile_data is absent")
    required = ("x_center", "count", "y_mean")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise HarnessError(f"injected-truth profile_data missing {missing}")
    return frame


def _it_faceted_profile_frame(stats: Any, facet: int):
    if not isinstance(stats, dict):
        raise HarnessError(f"injected-truth faceted stats are {type(stats).__name__}, expected dict")
    groups = [int(float(x)) for x in stats.get("groups", ())]
    if sorted(groups) != [0, 1]:
        raise HarnessError(f"injected-truth facet identity mismatch: {groups}")
    per_group = stats.get("per_group")
    if not isinstance(per_group, dict):
        raise HarnessError("injected-truth faceted stats missing per_group")
    cell = per_group.get(str(facet))
    if not isinstance(cell, dict):
        raise HarnessError(f"injected-truth missing side_type={facet}")
    return _it_profile_frame(cell)


def _it_close(label: str, expected: Any, observed: Any, *, rtol=1e-8, atol=1e-9):
    a = np.asarray(expected)
    b = np.asarray(observed)
    if a.shape != b.shape:
        raise HarnessError(f"{label} shape mismatch {a.shape}!={b.shape}")
    if np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_):
        if not np.array_equal(a, b):
            raise HarnessError(f"{label} exact mismatch")
        return
    if not np.allclose(a.astype(float), b.astype(float), rtol=rtol, atol=atol, equal_nan=True):
        finite = np.isfinite(a.astype(float)) & np.isfinite(b.astype(float))
        max_abs = float(np.max(np.abs(a.astype(float)[finite] - b.astype(float)[finite]))) if np.any(finite) else float("nan")
        raise HarnessError(f"{label} numerical mismatch max_abs={max_abs}")


def _it_prepare_public(adf: Any, gallery: Any):
    helper = getattr(gallery, "_ensure_injected_truth", None)
    if not callable(helper):
        raise HarnessError("gallery has no _ensure_injected_truth owner")
    helper(adf)
    return _it_semantic_model(adf)


def _it_vector_overlay(adf: Any, gallery: Any) -> tuple[dict, dict]:
    model = _it_prepare_public(adf, gallery)
    raw = getattr(gallery, INJECTED_TRUTH_VECTOR_GALLERY)(adf)
    try:
        stats = raw[2] if isinstance(raw, tuple) and len(raw) >= 3 else None
        if not isinstance(stats, list) or len(stats) != 4:
            raise HarnessError("vector overlay must expose exactly four branch stats")
        ys = (model["dcar"], model["clean"], model["noisy"], model["delta"])
        exp_counts, got_counts, exp_means, got_means = [], [], [], []
        for i, y in enumerate(ys):
            ref = _it_profile(model["sector"], y, model["base"])
            frame = _it_profile_frame(stats[i])
            _it_close(f"vector branch {i} x_center", ref["x_center"], frame["x_center"].to_numpy())
            exp_counts.extend(ref["count"].tolist())
            got_counts.extend(np.asarray(frame["count"], dtype=int).tolist())
            exp_means.extend(ref["y_mean"].tolist())
            got_means.extend(np.asarray(frame["y_mean"], dtype=float).tolist())
        return (
            {"counts": exp_counts, "profile_means": exp_means},
            {"counts": got_counts, "profile_means": got_means},
        )
    finally:
        try:
            plt.close(raw[0])
        except Exception:
            pass


def _it_direct_delta(adf: Any, gallery: Any) -> tuple[dict, dict]:
    model = _it_prepare_public(adf, gallery)
    raw = getattr(gallery, INJECTED_TRUTH_DELTA_GALLERY)(adf)
    try:
        ref = _it_profile(model["sector"], model["delta"], model["base"])
        stats = raw[2] if isinstance(raw, tuple) and len(raw) >= 3 else None

        observed = None

        def _normalized_value(payload: Any) -> np.ndarray | None:
            """Extract public normalize_data.value from supported stats shapes.

            dfdraw's established normalize payload is normally a pandas
            DataFrame, while some compatibility paths expose a dict.  Traverse
            vector/list wrappers but do not infer normalization from unrelated
            profile payloads.
            """
            if isinstance(payload, (list, tuple)):
                for child in payload:
                    value = _normalized_value(child)
                    if value is not None:
                        return value
                return None
            if not isinstance(payload, dict):
                return None

            nd = payload.get("normalize_data")
            if isinstance(nd, dict) and "value" in nd:
                return np.asarray(nd["value"], dtype=float)
            if nd is not None and hasattr(nd, "columns") and "value" in nd.columns:
                return np.asarray(nd["value"], dtype=float)
            return None

        observed = _normalized_value(stats)

        if observed is None:
            fig = raw[0] if isinstance(raw, tuple) else None
            candidate_lines = []
            for ax in getattr(fig, "axes", ()):
                ylabel = str(getattr(ax, "get_ylabel", lambda: "")()).strip().lower()
                title = str(getattr(ax, "get_title", lambda: "")()).strip().lower()
                if ("δ" in ylabel or "delta" in ylabel or "δ" in title or "delta" in title):
                    for line in getattr(ax, "lines", ()):
                        xs = np.asarray(line.get_xdata(), dtype=float)
                        ys = np.asarray(line.get_ydata(), dtype=float)
                        if len(xs) == INJECTED_TRUTH_BINS and len(ys) == INJECTED_TRUTH_BINS:
                            candidate_lines.append(ys)
            if len(candidate_lines) == 1:
                observed = candidate_lines[0]

        if observed is None:
            raise HarnessError(
                "normalize='delta' request exposed no normalized public payload/artist; "
                "normalization may have been ignored for vector expressions")
        return ({"delta_values": ref["y_mean"].tolist()},
                {"delta_values": np.asarray(observed, float).tolist()})
    finally:
        try:
            plt.close(raw[0])
        except Exception:
            pass


def _it_selection_delta(adf: Any, gallery: Any) -> tuple[dict, dict]:
    model = _it_prepare_public(adf, gallery)
    raw = getattr(gallery, INJECTED_TRUTH_SELECTION_GALLERY)(adf)
    try:
        stats = raw[2]
        if not isinstance(stats, dict) or not isinstance(stats.get("normalize_data_faceted"), dict):
            raise HarnessError("selection injected-truth case missing normalize_data_faceted")
        exp_delta, got_delta, exp_mask, got_mask = [], [], [], []
        for facet in (0, 1):
            fmask = model["side_sel"] & (model["side"] == facet)
            left = _it_profile(model["sector"], model["delta"], fmask & (model["tgl"] < 0))
            right = _it_profile(model["sector"], model["delta"], fmask & (model["tgl"] >= 0))
            valid = (left["count"] > 0) & (right["count"] > 0)
            delta = left["y_mean"] - right["y_mean"]
            delta[~valid] = np.nan
            cell = stats["normalize_data_faceted"].get(str(facet))
            if not isinstance(cell, dict):
                raise HarnessError(f"selection injected-truth missing facet={facet}")
            observed = np.asarray(cell.get("values"), dtype=float)
            observed_valid = ~np.asarray(cell.get("mask_undefined"), dtype=bool)
            exp_delta.extend(delta.tolist()); got_delta.extend(observed.tolist())
            exp_mask.extend(valid.tolist()); got_mask.extend(observed_valid.tolist())
        return ({"delta_values": exp_delta, "valid_mask": exp_mask},
                {"delta_values": got_delta, "valid_mask": got_mask})
    finally:
        try:
            plt.close(raw[0])
        except Exception:
            pass


def _it_weights(adf: Any, gallery: Any) -> tuple[dict, dict]:
    model = _it_prepare_public(adf, gallery)
    raw = getattr(gallery, INJECTED_TRUTH_WEIGHTS_GALLERY)(adf)
    try:
        stats = raw[2]
        if not isinstance(stats, list) or len(stats) != 2:
            raise HarnessError("weights injected-truth case expected two weight branches")
        exp = {"count": [], "sum_weights": [], "y_mean": []}
        got = {"count": [], "sum_weights": [], "y_mean": []}
        for branch_index, weights in enumerate((model["w_flat"], model["w_tgl"])):
            branch = stats[branch_index]
            for facet in (0, 1):
                mask = model["side_sel"] & (model["side"] == facet)
                ref = _it_profile(model["sector"], model["delta"], mask, weights=weights)
                frame = _it_faceted_profile_frame(branch, facet)
                exp["count"].extend(ref["count"].tolist())
                got["count"].extend(np.asarray(frame["count"], dtype=int).tolist())
                exp["sum_weights"].extend(ref["sum_weights"].tolist())
                got["sum_weights"].extend(np.asarray(frame["sum_weights"], dtype=float).tolist())
                exp["y_mean"].extend(ref["y_mean"].tolist())
                got["y_mean"].extend(np.asarray(frame["y_mean"], dtype=float).tolist())
        return exp, got
    finally:
        try:
            plt.close(raw[0])
        except Exception:
            pass


def _it_vector_reference(model: Mapping[str, Any], *, accumulator: str) -> tuple[dict, list[dict]]:
    ys = (
        ("dcar_tpc_vertex", model["dcar"]),
        ("dcar_distorted_clean", model["clean"]),
        ("dcar_oracle", model["noisy"]),
        ("known_delta", model["delta"]),
    )
    counts, means, coords = [], [], []
    for branch, y in ys:
        ref = _it_profile(model["sector"], y, model["base"], accumulator=accumulator)
        counts.extend(ref["count"].tolist())
        means.extend(ref["y_mean"].tolist())
        sector64 = np.asarray(model["sector"], dtype=np.float64)
        y_native = np.asarray(y)
        edges = np.linspace(INJECTED_TRUTH_RANGE[0], INJECTED_TRUTH_RANGE[1], INJECTED_TRUTH_BINS + 1)
        idx = np.searchsorted(edges, sector64, side="right") - 1
        idx[sector64 == INJECTED_TRUTH_RANGE[1]] = INJECTED_TRUTH_BINS - 1
        valid = np.asarray(model["base"], dtype=bool) & np.isfinite(sector64) & np.isfinite(y_native)
        for b, center in enumerate(ref["x_center"]):
            values = np.asarray(y_native[valid & (idx == b)], dtype=np.float64)
            coords.append({
                "branch": branch,
                "bin_index": int(b),
                "sector_center": float(center),
                "row_count": int(ref["count"][b]),
                "y_min": float(np.min(values)) if values.size else float("nan"),
                "y_max": float(np.max(values)) if values.size else float("nan"),
                "materialized_dtype": str(y_native.dtype),
            })
    return {"counts": counts, "profile_means": means}, coords


def _it_selection_reference(model: Mapping[str, Any], *, accumulator: str) -> tuple[dict, list[dict], list[dict]]:
    delta_values, valid_mask, coords, branch_context = [], [], [], []
    for facet in (0, 1):
        fmask = model["side_sel"] & (model["side"] == facet)
        left = _it_profile(
            model["sector"], model["delta"], fmask & (model["tgl"] < 0),
            accumulator=accumulator)
        right = _it_profile(
            model["sector"], model["delta"], fmask & (model["tgl"] >= 0),
            accumulator=accumulator)
        valid = (left["count"] > 0) & (right["count"] > 0)
        delta = left["y_mean"] - right["y_mean"]
        delta[~valid] = np.nan
        delta_values.extend(delta.tolist())
        valid_mask.extend(valid.tolist())
        for b, center in enumerate(left["x_center"]):
            coords.append({
                "facet": int(facet),
                "bin_index": int(b),
                "sector_center": float(center),
                "branch0_row_count": int(left["count"][b]),
                "branch1_row_count": int(right["count"][b]),
            })
            branch_context.append({
                "facet": int(facet),
                "bin_index": int(b),
                "sector_center": float(center),
                "branch0_row_count": int(left["count"][b]),
                "branch1_row_count": int(right["count"][b]),
                "branch0_mean": float(left["y_mean"][b]),
                "branch1_mean": float(right["y_mean"][b]),
                "independent_delta": float(delta[b]),
            })
    return {"delta_values": delta_values, "valid_mask": valid_mask}, coords, branch_context


def _it_numeric_recheck_diagnostics(case_id: str, adf: Any, observed: Mapping[str, Any]) -> dict:
    """Numerical calibration evidence for ORACLE-04 and ORACLE-06."""
    model = _it_semantic_model(adf)
    df = adf.df
    if case_id == INJECTED_TRUTH_VECTOR_CASE_ID:
        native, coords = _it_vector_reference(model, accumulator="native")
        high, _ = _it_vector_reference(model, accumulator="float64")
        diag = _numeric_reference_diagnostic(
            observable="profile_means",
            reference_native=native["profile_means"],
            reference_high_precision=high["profile_means"],
            observed=observed["profile_means"],
            coordinates=coords,
            atol=1e-9,
            rtol=1e-8,
            dtype_metadata={
                "raw_source_dtype_dcar_tpc_vertex": str(np.asarray(df["dcar_tpc_vertex"]).dtype),
                "adf_materialized_dtype_known_delta": str(np.asarray(df["known_delta"]).dtype),
                "adf_materialized_dtype_dcar_distorted_clean": str(np.asarray(df["dcar_distorted_clean"]).dtype),
                "adf_materialized_dtype_dcar_oracle": str(np.asarray(df["dcar_oracle"]).dtype),
                "expression_result_dtypes": {
                    "dcar_tpc_vertex": str(np.asarray(model["dcar"]).dtype),
                    "dcar_distorted_clean": str(np.asarray(model["clean"]).dtype),
                    "dcar_oracle": str(np.asarray(model["noisy"]).dtype),
                    "known_delta": str(np.asarray(model["delta"]).dtype),
                },
                "oracle_accumulator_dtype": "float64",
                "dfdraw_effective_reduction_dtype": "float64 (source-reviewed contract)",
                "returned_statistic_dtype": str(np.asarray(observed["profile_means"]).dtype),
            },
            input_finiteness={
                "dcar_tpc_vertex": _finite_summary(model["dcar"]),
                "dcar_distorted_clean": _finite_summary(model["clean"]),
                "dcar_oracle": _finite_summary(model["noisy"]),
                "known_delta": _finite_summary(model["delta"]),
            },
        )
        return {
            "case_id": case_id,
            "contract_reference_status": "VERIFIED",
            "reference_policy": "identical materialized values; independent float64 mean",
            "comparisons": [diag],
        }

    if case_id == INJECTED_TRUTH_SELECTION_CASE_ID:
        native, coords, _ = _it_selection_reference(model, accumulator="native")
        high, _, branch_context = _it_selection_reference(model, accumulator="float64")
        diag = _numeric_reference_diagnostic(
            observable="delta_values",
            reference_native=native["delta_values"],
            reference_high_precision=high["delta_values"],
            observed=observed["delta_values"],
            coordinates=coords,
            atol=1e-9,
            rtol=1e-8,
            dtype_metadata={
                "raw_source_dtype_tgl": str(np.asarray(df["tgl"]).dtype),
                "adf_materialized_dtype_known_delta": str(np.asarray(df["known_delta"]).dtype),
                "expression_result_dtype": str(np.asarray(model["delta"]).dtype),
                "oracle_accumulator_dtype": "float64",
                "dfdraw_effective_reduction_dtype": "float64 (source-reviewed contract)",
                "returned_statistic_dtype": str(np.asarray(observed["delta_values"]).dtype),
            },
            input_finiteness={
                "tgl": _finite_summary(model["tgl"]),
                "known_delta": _finite_summary(model["delta"]),
            },
        )
        mismatch_indices = [row["flat_index"] for row in diag.get("first_mismatches", [])]
        diag["first_failing_branch_context"] = [
            branch_context[i] for i in mismatch_indices if i < len(branch_context)
        ]
        return {
            "case_id": case_id,
            "contract_reference_status": "VERIFIED",
            "reference_policy": "identical branch/facet/bin membership; independent float64 branch means",
            "comparisons": [diag],
        }

    raise HarnessError(f"no numerical recheck diagnostics for {case_id}")


def _it_gauss(adf: Any, gallery: Any) -> tuple[dict, dict, dict]:
    model = _it_prepare_public(adf, gallery)
    raw = getattr(gallery, INJECTED_TRUTH_GAUSS_GALLERY)(adf)
    try:
        selected = np.asarray(model["noise"][model["base"]], dtype=float)
        selected = selected[np.isfinite(selected) & (selected >= -0.1) & (selected <= 0.1)]
        if selected.size < 100:
            raise HarnessError("Gaussian injected-truth selected sample is too small")
        raw_mean = float(np.mean(selected, dtype=np.float64))
        raw_sigma = float(np.std(selected, ddof=0, dtype=np.float64))
        record = _first_fit_record(raw[2] if isinstance(raw, tuple) and len(raw) >= 3 else None)
        if not record or record.get("fit_name") != "gauss" or record.get("fit_status") != "ok":
            raise HarnessError(f"Gaussian injected-truth fit missing/invalid: {record}")
        named = _fit_named_values(record)
        center = named.get("center")
        sigma = named.get("sigma")
        if center is None or sigma is None:
            raise HarnessError(f"Gaussian injected-truth fit lacks named center/sigma: {named}")

        bin_width = 0.2 / 100.0
        center_tol = max(0.25 * bin_width, 5.0 * raw_sigma / np.sqrt(selected.size))
        sigma_tol = max(
            0.25 * bin_width,
            5.0 * raw_sigma / np.sqrt(max(2.0 * (selected.size - 1), 1.0)),
        )
        center_ratio = abs(float(center) - raw_mean) / center_tol
        sigma_ratio = abs(abs(float(sigma)) - raw_sigma) / sigma_tol
        expected = {"fit_center_delta": 0.0, "fit_sigma_delta": 0.0}
        observed = {
            "fit_center_delta": float(center_ratio),
            "fit_sigma_delta": float(sigma_ratio),
        }
        diagnostics = {
            "raw_selected_mean": raw_mean,
            "raw_selected_sigma": raw_sigma,
            "fit_center": float(center),
            "fit_sigma": float(sigma),
            "center_tolerance": float(center_tol),
            "sigma_tolerance": float(sigma_tol),
            "selected_rows": int(selected.size),
        }
        return expected, observed, diagnostics
    finally:
        try:
            plt.close(raw[0])
        except Exception:
            pass


def _it_facet(adf: Any, gallery: Any) -> tuple[dict, dict]:
    model = _it_prepare_public(adf, gallery)
    raw = getattr(gallery, INJECTED_TRUTH_FACET_GALLERY)(adf)
    try:
        stats = raw[2]
        exp = {"count": [], "y_mean": []}
        got = {"count": [], "y_mean": []}
        for facet in (0, 1):
            mask = model["side_sel"] & (model["side"] == facet)
            ref = _it_profile(model["sector"], model["noise"], mask)
            frame = _it_faceted_profile_frame(stats, facet)
            exp["count"].extend(ref["count"].tolist())
            got["count"].extend(np.asarray(frame["count"], dtype=int).tolist())
            exp["y_mean"].extend(ref["y_mean"].tolist())
            got["y_mean"].extend(np.asarray(frame["y_mean"], dtype=float).tolist())
        return exp, got
    finally:
        try:
            plt.close(raw[0])
        except Exception:
            pass


def _m1_bar_heights(ax: Any, *, n_series: int, n_bins: int) -> list[np.ndarray]:
    """Extract rendered histogram heights for bar or stepfilled artists.

    dfdraw currently renders these vector histograms as filled step polygons,
    not BarContainers.  Keep both shapes supported so the oracle follows the
    public rendered result without depending on one Matplotlib histtype.
    """
    series: list[np.ndarray] = []

    # Ordinary bar histograms.
    for container in getattr(ax, "containers", ()):
        patches = getattr(container, "patches", None)
        if patches is not None and len(patches) == n_bins:
            series.append(np.asarray(
                [float(p.get_height()) for p in patches], dtype=float))

    # Filled/step histograms.  The forward half of Matplotlib's closed Polygon
    # path has two vertices per bin; the first of each pair carries that bin's
    # top height.  A 20-bin filled step therefore has 4*20+1 vertices.
    if len(series) < n_series:
        for patch in getattr(ax, "patches", ()):
            get_xy = getattr(patch, "get_xy", None)
            if not callable(get_xy):
                continue
            xy = np.asarray(get_xy(), dtype=float)
            if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 * n_bins + 1:
                continue
            heights = xy[1:2 * n_bins:2, 1]
            if len(heights) == n_bins:
                series.append(np.asarray(heights, dtype=float))

    if len(series) < n_series:
        raise HarnessError(
            f"histogram artist exposes {len(series)} extractable {n_bins}-bin "
            f"series; expected >= {n_series}")
    return series[-n_series:]


def run_m1_hist_vector(case: CaseSpec, root_path: str, *, gallery_module=None,
                       prepared_adf=None, prepared_provenance=None) -> CaseResult:
    t0=time.time(); res=CaseResult(case_id=case.case_id, status=SKIP); adf=prepared_adf
    try:
        if adf is None: raise HarnessError("M1 requires shared FAST prepared_adf")
        gallery=gallery_module if gallery_module is not None else _a5_2_import_gallery()
        provenance=_a6_4_prepared_fraction_sample_evidence(adf, prepared_provenance, root_path)
        getattr(gallery, "_ensure_injected_truth")(adf)
        raw=getattr(gallery, M1_HIST_GALLERY_FUNCTION)(adf)
        fig, axes, meta=raw
        df=adf.df
        ncl=np.asarray(df["ncl"], dtype=float); ts=np.asarray(df["time_s"], dtype=float)
        base=(ncl>60)&np.isfinite(ncl); mid=float(meta["t_mid"])
        edges=np.linspace(80.0,160.0,21)
        exp_sel=[]
        for mask in (base&(ts<mid), base&(ts>=mid)):
            exp_sel.extend(np.histogram(ncl[mask], bins=edges)[0].tolist())
        got_sel=np.concatenate(_m1_bar_heights(axes[0], n_series=2, n_bins=20)).astype(int).tolist()
        exp_w=[]
        for w in (np.ones(len(df), dtype=float), 1.0+0.30*np.abs(np.asarray(df["tgl"],dtype=float))):
            finite=base&np.isfinite(w)
            exp_w.extend(np.histogram(ncl[finite], bins=edges, weights=w[finite])[0].tolist())
        got_w=np.concatenate(_m1_bar_heights(axes[1], n_series=2, n_bins=20)).tolist()
        expected={"selection_bin_counts":exp_sel,"weighted_bin_sums":exp_w}
        observed={"selection_bin_counts":got_sel,"weighted_bin_sums":got_w}
        for obs in case.observables:
            res.observable_contract.append(_contract(obs)); cmp=compare_observable(obs, expected[obs.name], observed[obs.name])
            res.comparisons.append(comparison_evidence(obs,cmp,reference_label="raw np.histogram",candidate_label="rendered histogram bars"))
            if not cmp.ok: raise HarnessError(f"{obs.name}: {cmp.detail}")
        res.executed_comparisons=len(case.observables); res.observed["realdata_provenance"]=dict(prepared_provenance or provenance)
        res.status=PASS; res.detail=""; return res
    except Exception as exc:
        res.status=FAIL; res.detail=f"M1 HIST FAIL: {exc}"; res.exception=traceback.format_exc(limit=8); return res
    finally:
        res.wall_time_s=round(time.time()-t0,4)
        if adf is not None: _record_stage_a_machine_status(adf,res)
        try: plt.close('all')
        except Exception: pass


def run_m1_hist_normalize_contract(case: CaseSpec, make_adf: Callable[[], Any] | None = None) -> CaseResult:
    t0=time.time(); res=CaseResult(case_id=case.case_id,status=SKIP)
    try:
        if make_adf is None: raise HarnessError("M1 hist normalize contract requires make_adf")
        adf=make_adf(); refused=False; detail=""
        try:
            adf.draw("ncl", type="hist", bins=10, normalize="delta", auto_title=False)
        except Exception as exc:
            detail=f"{type(exc).__name__}: {exc}"
            low=detail.lower()
            refused=("normalize" in low and ("hist" in low or "unsupported" in low or "not" in low))
        expected=True; observed=bool(refused)
        obs=case.observables[0]; res.observable_contract.append(_contract(obs))
        cmp=compare_observable(obs, expected, observed)
        res.comparisons.append(comparison_evidence(obs,cmp,reference_label="explicit loud-refusal contract",candidate_label=detail or "call returned normally"))
        if not cmp.ok: raise HarnessError("hist normalize did not refuse explicitly; silent semantic drop/ambiguous acceptance")
        res.executed_comparisons=1; res.status=PASS; res.detail=detail; return res
    except Exception as exc:
        res.status=FAIL; res.detail=f"M1 HIST NORMALIZE FAIL: {exc}"; res.exception=traceback.format_exc(limit=8); return res
    finally: res.wall_time_s=round(time.time()-t0,4)


def _m2_profile_groups(stats: Any) -> list[tuple[Any, Any]]:
    if not isinstance(stats, dict) or not hasattr(stats.get("profile_data"), "columns"):
        raise HarnessError("grouping oracle requires public profile_data DataFrame")
    frame=stats["profile_data"]
    if "group" not in frame.columns: raise HarnessError("grouping profile_data lacks group column")
    return [(g, frame[frame["group"]==g].copy()) for g in pd.unique(frame["group"])]


def _m2_grouping_arrays(model: dict, meta: dict, *, accumulator: str = "float64",
                        return_coordinates: bool = False):
    exp_c=[]; got_c=[]; exp_m=[]; got_m=[]; coords=[]
    groups=_m2_profile_groups(meta["categorical_stats"])
    if [int(float(g)) for g,_ in groups] != [0,1]:
        raise HarnessError(f"categorical group identity mismatch {[g for g,_ in groups]}")
    for (g,frame),side in zip(groups,(0,1)):
        ref=_it_profile(
            model["sector"],model["delta"],model["side_sel"]&(model["side"]==side),
            accumulator=accumulator)
        exp_c.extend(ref["count"].tolist()); got_c.extend(np.asarray(frame["count"],dtype=int).tolist())
        exp_m.extend(ref["y_mean"].tolist()); got_m.extend(np.asarray(frame["y_mean"],dtype=float).tolist())
        for b,center in enumerate(ref["x_center"]):
            coords.append({
                "group_kind":"categorical",
                "group":"side_type",
                "group_identity":int(side),
                "bin_index":int(b),
                "sector_center":float(center),
                "row_count":int(ref["count"][b]),
            })

    base=model["base"] & np.isfinite(model["tgl"])
    selected_tgl=pd.Series(model["tgl"][base]); cut=pd.cut(selected_tgl,bins=4)
    categories=list(cut.cat.categories); groups2=_m2_profile_groups(meta["binned_stats"])
    if len(groups2)!=len(categories):
        raise HarnessError(f"group_by_bins cardinality {len(groups2)}!={len(categories)}")
    selected_indices=np.flatnonzero(base)
    for j,(_,frame) in enumerate(groups2):
        member=np.zeros(len(model["df"]),dtype=bool)
        member[selected_indices[np.asarray(cut==categories[j])]]=True
        ref=_it_profile(model["sector"],model["delta"],member,accumulator=accumulator)
        exp_c.extend(ref["count"].tolist()); got_c.extend(np.asarray(frame["count"],dtype=int).tolist())
        exp_m.extend(ref["y_mean"].tolist()); got_m.extend(np.asarray(frame["y_mean"],dtype=float).tolist())
        for b,center in enumerate(ref["x_center"]):
            coords.append({
                "group_kind":"binned",
                "group":"tgl",
                "group_index":int(j),
                "group_identity":str(categories[j]),
                "bin_index":int(b),
                "sector_center":float(center),
                "row_count":int(ref["count"][b]),
            })
    expected={"group_counts":exp_c,"group_means":exp_m}
    observed={"group_counts":got_c,"group_means":got_m}
    if return_coordinates:
        return expected, observed, coords
    return expected, observed


def _m2_numeric_recheck_diagnostics(adf: Any, model: dict, meta: dict, observed: dict) -> dict:
    native, _, coords = _m2_grouping_arrays(
        model, meta, accumulator="native", return_coordinates=True)
    high, _, _ = _m2_grouping_arrays(
        model, meta, accumulator="float64", return_coordinates=True)
    diag = _numeric_reference_diagnostic(
        observable="group_means",
        reference_native=native["group_means"],
        reference_high_precision=high["group_means"],
        observed=observed["group_means"],
        coordinates=coords,
        atol=1e-9,
        rtol=1e-8,
        dtype_metadata={
            "raw_source_dtype_sector": str(np.asarray(adf.df["sector"]).dtype),
            "raw_source_dtype_tgl": str(np.asarray(adf.df["tgl"]).dtype),
            "adf_materialized_dtype_known_delta": str(np.asarray(adf.df["known_delta"]).dtype),
            "expression_result_dtype": str(np.asarray(model["delta"]).dtype),
            "oracle_accumulator_dtype": "float64",
            "dfdraw_effective_reduction_dtype": "float64 (source-reviewed contract)",
            "returned_statistic_dtype": str(np.asarray(observed["group_means"]).dtype),
        },
        input_finiteness={
            "sector": _finite_summary(model["sector"]),
            "tgl": _finite_summary(model["tgl"]),
            "known_delta": _finite_summary(model["delta"]),
        },
    )
    return {
        "case_id": M2_GROUP_CASE_ID,
        "contract_reference_status": "VERIFIED",
        "reference_policy": "identical group/bin membership; independent float64 group means",
        "comparisons": [diag],
    }


def _m2_direct_meta(adf: Any) -> tuple[dict, list[Any]]:
    owner=_DirectDFDrawOwner(adf.df); figures=[]
    cat=owner.draw(
        "known_delta:sector", type="profile", bins=INJECTED_TRUTH_BINS,
        range=INJECTED_TRUTH_RANGE, selection=INJECTED_TRUTH_SIDE_SEL,
        group_by="side_type", min_entries=1, return_data=True, auto_title=False)
    figures.append(cat[0])
    binned=owner.draw(
        "known_delta:sector", type="profile", bins=INJECTED_TRUTH_BINS,
        range=INJECTED_TRUTH_RANGE, selection=INJECTED_TRUTH_BASE_SEL,
        group_by="tgl", group_by_bins=4, min_entries=1,
        return_data=True, auto_title=False)
    figures.append(binned[0])
    return {"categorical_stats":cat[2],"binned_stats":binned[2]}, figures


def _m2_classify_failure(adf: Any, case: CaseSpec, model: dict, expected: dict,
                         semantic_diag: dict) -> dict:
    diag=copy.deepcopy(semantic_diag)
    if not diag.get("L1_all_within_tolerance",False):
        diag.update({
            "first_disagreement_layer":"L1",
            "contract_reference_status":"VERIFIED",
            "owner_status":"ADF",
            "derived_owner":"ADF",
        })
        return diag
    figures=[]
    try:
        meta,figures=_m2_direct_meta(adf)
        _,direct_observed=_m2_grouping_arrays(model,meta,accumulator="float64")
        numeric=_m2_numeric_recheck_diagnostics(adf,model,meta,direct_observed)
        owner=_numeric_owner_from_diagnostics(
            l1_ok=True,diagnostics=numeric["comparisons"])
        diag.update({
            "L2":"direct DFDraw grouped profiles on materialized known_delta",
            "L2_matches_high_precision_truth":bool(
                all(d["reference_high_precision"]["ok"] for d in numeric["comparisons"])),
            "L3":"adf.draw grouped profile evaluated against corrected contract",
            "numerical_recheck":numeric,
            **owner,
        })
        return diag
    except Exception as exc:
        diag.update({
            "L2":"direct DFDraw grouped profiles on materialized known_delta",
            "L2_matches_high_precision_truth":False,
            "L2_detail":f"{type(exc).__name__}: {exc}",
            "L3":"adf.draw grouped profile requires adjudication",
            "first_disagreement_layer":"L2",
            "contract_reference_status":"UNRESOLVED",
            "owner_status":"UNRESOLVED",
            "derived_owner":"UNKNOWN",
        })
        return diag
    finally:
        for fig in figures:
            try: plt.close(fig)
            except Exception: pass


def run_m2_grouping(case: CaseSpec, root_path: str, *, gallery_module=None,
                    prepared_adf=None, prepared_provenance=None) -> CaseResult:
    t0=time.time(); res=CaseResult(case_id=case.case_id,status=SKIP); adf=prepared_adf
    try:
        if adf is None: raise HarnessError("M2 requires shared FAST prepared_adf")
        gallery=gallery_module if gallery_module is not None else _a5_2_import_gallery(); getattr(gallery,"_ensure_injected_truth")(adf)
        provenance=_a6_4_prepared_fraction_sample_evidence(adf,prepared_provenance,root_path)
        raw=getattr(gallery,M2_GROUP_GALLERY_FUNCTION)(adf); meta=raw[2]
        semantic_diag=_it_semantic_dtype_diagnostics(adf,gallery)
        model=_it_semantic_model(adf)
        expected,observed=_m2_grouping_arrays(model,meta,accumulator="float64")
        numeric_recheck=_m2_numeric_recheck_diagnostics(adf,model,meta,observed)
        first_failure=None
        for obs in case.observables:
            res.observable_contract.append(_contract(obs)); cmp=compare_observable(obs,expected[obs.name],observed[obs.name])
            res.comparisons.append(comparison_evidence(obs,cmp,reference_label="raw pandas grouping with authoritative source dtype",candidate_label="public grouped profile_data"))
            if not cmp.ok and first_failure is None: first_failure=f"{obs.name}: {cmp.detail}"
        if first_failure is not None:
            res.observed["ownership_ladder"]=_m2_classify_failure(adf,case,model,expected,semantic_diag)
            raise HarnessError(first_failure)
        res.executed_comparisons=len(case.observables)
        res.observed["realdata_provenance"]=dict(prepared_provenance or provenance)
        semantic_diag["numerical_recheck"]=numeric_recheck
        semantic_diag.update(_numeric_owner_from_diagnostics(
            l1_ok=True,diagnostics=numeric_recheck["comparisons"]))
        semantic_diag.update({
            "L2":"public grouped result matches corrected float64 contract",
            "L3":"adf.draw grouped result matches corrected truth",
        })
        res.observed["ownership_ladder"]=semantic_diag
        res.status=PASS; res.detail=""; return res
    except Exception as exc:
        res.status=FAIL; res.detail=f"M2 GROUPING FAIL: {exc}"; res.exception=traceback.format_exc(limit=8); return res
    finally:
        res.wall_time_s=round(time.time()-t0,4)
        if adf is not None: _record_stage_a_machine_status(adf,res)
        try: plt.close('all')
        except Exception: pass


def _m3_independent_semantic_values(adf: Any) -> dict:
    """Recompute the M3 chain independently, honoring each materialized dtype."""
    df = adf.df
    semantic = _it_semantic_model(adf)
    tgl = np.asarray(df["tgl"])
    sector = np.asarray(df["sector"])
    side = np.asarray(df["side_type"])
    l1 = _it_cast_expected_to_authoritative_dtype(
        df, "oracle_chain_l1", semantic["delta"] + 0.005 * tgl)
    l2 = _it_cast_expected_to_authoritative_dtype(
        df, "oracle_chain_l2", l1 - 0.002 * sector / 36.0)
    offset = 0.04 - 0.035 * side.astype(float)
    final = _it_cast_expected_to_authoritative_dtype(
        df, "oracle_chain_with_shift", l2 + offset)
    return {"chain_l1": l1, "chain_l2": l2, "offset": offset, "final": final}


def _m3_materialization_diagnostics(adf: Any, expected: dict) -> dict:
    rows = {}
    ok = True
    for key, column in (("chain_l1", "oracle_chain_l1"),
                        ("chain_l2", "oracle_chain_l2"),
                        ("final", "oracle_chain_with_shift")):
        exp = np.asarray(expected[key])
        got = np.asarray(adf.df[column]).astype(exp.dtype, copy=False)
        exact = bool(np.array_equal(exp, got, equal_nan=True))
        ok &= exact
        finite = np.isfinite(exp.astype(float)) & np.isfinite(got.astype(float))
        delta = np.abs(exp.astype(float)[finite] - got.astype(float)[finite]) if np.any(finite) else np.asarray([])
        rows[column] = {
            "expected_dtype": str(exp.dtype),
            "adf_dtype": str(np.asarray(adf.df[column]).dtype),
            "max_abs_difference": float(np.max(delta)) if delta.size else 0.0,
            "exact": exact,
        }
    return {
        "L0": "raw source columns + declared chain/subframe formula",
        "L1": "independent stepwise authoritative-dtype chain -> ADF materialized chain",
        "L1_all_within_tolerance": bool(ok),
        "first_disagreement_layer": ">=L2" if ok else "L1",
        "derived_owner": "UNRESOLVED_BEYOND_L1" if ok else "ADF",
        "variables": rows,
    }


def _m3_expected_observed(model: dict, chain: dict, stats: Any) -> tuple[dict,dict]:
    groups=_m2_profile_groups(stats)
    if [int(float(g)) for g,_ in groups] != [0,1]:
        raise HarnessError(f"M3 group identity mismatch {[g for g,_ in groups]}")
    exp_c=[];got_c=[];exp_m=[];got_m=[]
    for (g,frame),side in zip(groups,(0,1)):
        ref=_it_profile(model["sector"],chain["final"],
                        model["side_sel"]&(model["side"]==side))
        exp_c.extend(ref["count"].tolist()); got_c.extend(np.asarray(frame["count"],dtype=int).tolist())
        exp_m.extend(ref["y_mean"].tolist()); got_m.extend(np.asarray(frame["y_mean"],dtype=float).tolist())
    return ({"count":exp_c,"y_mean":exp_m},{"count":got_c,"y_mean":got_m})


def _m3_direct_stats(adf: Any) -> tuple[Any,Any]:
    owner=_DirectDFDrawOwner(adf.df)
    raw=owner.draw(
        "oracle_chain_with_shift:sector", type="profile",
        bins=INJECTED_TRUTH_BINS, range=INJECTED_TRUTH_RANGE,
        selection=INJECTED_TRUTH_SIDE_SEL, group_by="side_type",
        min_entries=1, return_data=True, auto_title=False)
    return raw[2],raw[0]


def _m3_classify_failure(adf: Any, case: CaseSpec, model: dict, chain: dict,
                         expected: dict, ownership: dict) -> dict:
    diag=copy.deepcopy(ownership)
    if not diag.get("L1_all_within_tolerance",False):
        diag.update({"first_disagreement_layer":"L1","derived_owner":"ADF"})
        return diag
    fig=None
    try:
        stats,fig=_m3_direct_stats(adf)
        _,direct_observed=_m3_expected_observed(model,chain,stats)
        ok,detail=_case_observables_match(case,expected,direct_observed)
        diag.update({
            "L2":"direct DFDraw on materialized oracle_chain_with_shift",
            "L2_matches_truth":bool(ok),"L2_detail":detail,
            "L3":"adf.draw qualified-chain profile is red",
            "first_disagreement_layer":"L3" if ok else "L2",
            "derived_owner":"ADF_DRAW_BRIDGE" if ok else "dfdraw",
        })
        return diag
    except Exception as exc:
        diag.update({
            "L2":"direct DFDraw on materialized oracle_chain_with_shift",
            "L2_matches_truth":False,"L2_detail":f"{type(exc).__name__}: {exc}",
            "L3":"adf.draw qualified-chain profile is red",
            "first_disagreement_layer":"L2","derived_owner":"dfdraw",
        })
        return diag
    finally:
        try:
            if fig is not None: plt.close(fig)
        except Exception: pass


def run_m3_subframe_chain(case: CaseSpec, root_path: str, *, gallery_module=None,
                          prepared_adf=None, prepared_provenance=None) -> CaseResult:
    t0=time.time(); res=CaseResult(case_id=case.case_id,status=SKIP); adf=prepared_adf
    try:
        if adf is None: raise HarnessError("M3 requires shared FAST prepared_adf")
        gallery=gallery_module if gallery_module is not None else _a5_2_import_gallery(); getattr(gallery,"_ensure_m3_subframe_chain")(adf)
        provenance=_a6_4_prepared_fraction_sample_evidence(adf,prepared_provenance,root_path)
        raw=getattr(gallery,M3_SUBFRAME_GALLERY_FUNCTION)(adf); stats=raw[2]
        model=_it_semantic_model(adf)
        chain=_m3_independent_semantic_values(adf)
        ownership=_m3_materialization_diagnostics(adf,chain)
        expected,observed=_m3_expected_observed(model,chain,stats)
        first_failure=None
        for obs in case.observables:
            res.observable_contract.append(_contract(obs)); cmp=compare_observable(obs,expected[obs.name],observed[obs.name])
            res.comparisons.append(comparison_evidence(obs,cmp,reference_label="raw chained formula + authoritative materialization dtype + raw subframe offset",candidate_label="public qualified grouped profile_data"))
            if not cmp.ok and first_failure is None: first_failure=f"{obs.name}: {cmp.detail}"
        if first_failure is not None:
            res.observed["ownership_ladder"]=_m3_classify_failure(adf,case,model,chain,expected,ownership)
            raise HarnessError(first_failure)
        res.executed_comparisons=len(case.observables)
        res.observed["realdata_provenance"]=dict(prepared_provenance or provenance)
        ownership.update({"first_disagreement_layer":"NONE","derived_owner":"NONE"})
        res.observed["ownership_ladder"]=ownership
        res.status=PASS; res.detail=""; return res
    except Exception as exc:
        res.status=FAIL; res.detail=f"M3 SUBFRAME FAIL: {exc}"; res.exception=traceback.format_exc(limit=8); return res
    finally:
        res.wall_time_s=round(time.time()-t0,4)
        if adf is not None: _record_stage_a_machine_status(adf,res)
        try: plt.close('all')
        except Exception: pass


def _it_semantic_dtype_diagnostics(adf: Any, gallery: Any) -> dict:
    """L0/L1 ownership evidence with AD-19 authoritative dtype semantics."""
    getattr(gallery, "_ensure_injected_truth")(adf)
    raw = _it_raw_model(adf)
    semantic = _it_semantic_model(adf)
    df = adf.df
    raw_expected = {
        "known_delta": raw["delta"],
        "dcar_distorted_clean": raw["clean"],
        "dcar_oracle": raw["noisy"],
        "oracle_residual": raw["residual"],
    }
    semantic_expected = {
        "known_delta": semantic["delta"],
        "dcar_distorted_clean": semantic["clean"],
        "dcar_oracle": semantic["noisy"],
        "oracle_residual": semantic["residual"],
    }
    rows = {}
    l1_ok = True
    for name in raw_expected:
        raw_values = np.asarray(raw_expected[name], dtype=np.float64)
        expected = np.asarray(semantic_expected[name])
        got_native = np.asarray(df[name])
        got = got_native.astype(expected.dtype, copy=False)
        finite_raw = np.isfinite(raw_values) & np.isfinite(expected.astype(np.float64))
        precision_delta = (np.abs(raw_values[finite_raw] - expected.astype(np.float64)[finite_raw])
                           if np.any(finite_raw) else np.asarray([]))
        finite = np.isfinite(expected.astype(np.float64)) & np.isfinite(got.astype(np.float64))
        diff = (np.abs(expected.astype(np.float64)[finite] - got.astype(np.float64)[finite])
                if np.any(finite) else np.asarray([]))
        denom = (np.maximum(np.abs(expected.astype(np.float64)[finite]), 1e-30)
                 if np.any(finite) else np.asarray([]))
        max_abs = float(np.max(diff)) if diff.size else 0.0
        max_rel = float(np.max(diff / denom)) if diff.size else 0.0
        exact = bool(np.array_equal(expected, got, equal_nan=True))
        l1_ok &= exact
        rows[name] = {
            "raw_formula_dtype": str(raw_values.dtype),
            "authoritative_dtype": str(expected.dtype),
            "adf_dtype": str(got_native.dtype),
            "max_abs_raw_to_authoritative": (
                float(np.max(precision_delta)) if precision_delta.size else 0.0),
            "max_abs_difference": max_abs,
            "max_relative_difference": max_rel,
            "exact_after_authoritative_cast": exact,
            "within_current_oracle_tolerance": exact,
        }
    return {
        "L0": "raw NumPy/pandas formula",
        "L1": "independent formula with AD-19 authoritative materialized dtype -> ADF materialized values",
        "L1_all_within_tolerance": bool(l1_ok),
        "first_disagreement_layer": ">=L2" if l1_ok else "L1",
        "derived_owner": "UNRESOLVED_BEYOND_L1" if l1_ok else "ADF",
        "variables": rows,
    }


class _DirectDFDrawOwner:
    """Minimal direct-dfdraw owner used only for M4 attribution.

    It deliberately bypasses AliasDataFrame.draw while consuming the already
    materialized plain DataFrame.  This is the L2 leg of the approved ownership
    ladder, not a second implementation of plotting semantics.
    """
    def __init__(self, df: Any):
        self.df = df
        try:
            from dfextensions.dfdraw.drawer import DFDraw
        except Exception:
            from dfdraw.drawer import DFDraw
        self._plotter = DFDraw(df)

    def draw(self, *args, **kwargs):
        return self._plotter.draw(*args, **kwargs)

    def profile(self, *args, **kwargs):
        return self._plotter.profile(*args, **kwargs)


def _o1_scalar_faceted_counts(stats: Any) -> list[int]:
    """Extract per-facet population from either scalar or one-vector envelope."""
    branch_stats = stats[0] if isinstance(stats, list) and len(stats) == 1 else stats
    per_group = branch_stats.get("per_group") if isinstance(branch_stats, dict) else None
    if not isinstance(per_group, dict):
        raise HarnessError(f"faceted result has no per_group payload: {type(stats).__name__}")
    out = []
    for facet in HARDENING_FACETS:
        cell = per_group.get(str(facet))
        if not isinstance(cell, dict) or "n" not in cell:
            raise HarnessError(f"facet {facet} has no public n")
        out.append(int(cell["n"]))
    return out


def _o1_neg_a_ownership_ladder(adf: Any, *, q1: float, expected: list[int],
                                  adf_observed: list[int]) -> dict:
    diag = {
        "L0": "raw boolean masks",
        "L0_expected": list(expected),
        "L3": "adf.draw one-element selection_vector per-facet counts",
        "L3_observed": list(adf_observed),
    }
    fig = None
    try:
        owner = _DirectDFDrawOwner(adf.df)
        raw = owner.draw(
            "dcar_tpc_vertex:tgl", selection=f"{HARDENING_BASE_SEL}&(side_type<2)",
            type="profile", bins=HARDENING_PROFILE_BINS, range=HARDENING_PROFILE_RANGE,
            selection_vector=[f"time_s<{q1}"], facet_by="side_type",
            min_entries=1, auto_title=False)
        fig = raw[0]
        direct = _o1_scalar_faceted_counts(raw[2])
        direct_ok = direct == list(expected)
        diag.update({
            "L2": "direct DFDraw one-element selection_vector per-facet counts",
            "L2_observed": direct,
            "L2_matches_truth": direct_ok,
            "first_disagreement_layer": "L3" if direct_ok else "L2",
            "contract_reference_status": "VERIFIED",
            "owner_status": "ADF" if direct_ok else "DFDRAW",
            "derived_owner": "ADF_DRAW_BRIDGE" if direct_ok else "dfdraw",
        })
    except Exception as exc:
        diag.update({
            "L2": "direct DFDraw one-element selection_vector per-facet counts",
            "L2_matches_truth": False,
            "L2_detail": f"{type(exc).__name__}: {exc}",
            "first_disagreement_layer": "L2",
            "contract_reference_status": "VERIFIED",
            "owner_status": "DFDRAW",
            "derived_owner": "dfdraw",
        })
    finally:
        try:
            if fig is not None: plt.close(fig)
        except Exception:
            pass
    return diag


def _o1_neg_b_direct_cells(adf: Any, *, t_mid: float) -> tuple[dict, Any]:
    owner = _DirectDFDrawOwner(adf.df)
    raw = owner.draw(
        "ncl", selection="(ncl>60)&(side_type<2)", type="hist", bins=40,
        range=(80.0, 160.0),
        selection_vector=[f"time_s<{t_mid}", f"time_s>={t_mid}"],
        facet_by="side_type", auto_title=False)
    axes = np.asarray(raw[1], dtype=object).reshape(-1)
    if len(axes) != len(HARDENING_FACETS):
        raise HarnessError(f"direct DFDraw expected {len(HARDENING_FACETS)} facets, got {len(axes)}")
    cells = {}
    for facet, ax in zip(HARDENING_FACETS, axes):
        rendered = _m1_bar_heights(ax, n_series=2, n_bins=40)
        for branch_index, heights in enumerate(rendered):
            cells[(branch_index, facet)] = np.asarray(heights, dtype=float)
    return cells, raw[0]


def _o1_neg_b_ownership_ladder(adf: Any, *, t_mid: float, expected_cells: dict,
                                  adf_observed_cells: dict) -> dict:
    diag = {
        "L0": "raw np.histogram branch×facet bins",
        "L3": "adf.draw rendered histogram branch×facet bins",
        "L3_complete_cells": sorted(str(k) for k in adf_observed_cells),
    }
    fig = None
    try:
        direct, fig = _o1_neg_b_direct_cells(adf, t_mid=t_mid)
        direct_ok = (set(direct) == set(expected_cells) and all(
            np.array_equal(np.asarray(expected_cells[k]), np.asarray(direct[k]))
            for k in expected_cells))
        diag.update({
            "L2": "direct DFDraw rendered histogram branch×facet bins",
            "L2_complete_cells": sorted(str(k) for k in direct),
            "L2_matches_truth": bool(direct_ok),
            "first_disagreement_layer": "L3" if direct_ok else "L2",
            "contract_reference_status": "VERIFIED",
            "owner_status": "ADF" if direct_ok else "DFDRAW",
            "derived_owner": "ADF_DRAW_BRIDGE" if direct_ok else "dfdraw",
        })
    except Exception as exc:
        diag.update({
            "L2": "direct DFDraw rendered histogram branch×facet bins",
            "L2_matches_truth": False,
            "L2_detail": f"{type(exc).__name__}: {exc}",
            "first_disagreement_layer": "L2",
            "contract_reference_status": "VERIFIED",
            "owner_status": "DFDRAW",
            "derived_owner": "dfdraw",
        })
    finally:
        try:
            if fig is not None: plt.close(fig)
        except Exception:
            pass
    return diag


class _DirectInjectedGallery:
    """Exact G12 public requests routed directly to DFDraw for M4 only."""
    @staticmethod
    def _ensure_injected_truth(owner):
        return owner

    @staticmethod
    def fig46_injected_truth_vector_overlay(owner):
        return owner.draw(
            "[dcar_tpc_vertex,dcar_distorted_clean,dcar_oracle,known_delta]:sector",
            type="profile", bins=INJECTED_TRUTH_BINS, range=INJECTED_TRUTH_RANGE,
            selection=INJECTED_TRUTH_BASE_SEL, auto_title=False, return_data=True)

    @staticmethod
    def fig47_injected_truth_direct_delta(owner):
        return owner.draw(
            "[dcar_distorted_clean,dcar_tpc_vertex]:sector",
            type="profile", bins=INJECTED_TRUTH_BINS, range=INJECTED_TRUTH_RANGE,
            selection=INJECTED_TRUTH_BASE_SEL, normalize="delta",
            auto_title=False, return_data=True)

    @staticmethod
    def fig48_injected_truth_selection_delta_facet(owner):
        return owner.draw(
            "known_delta:sector", type="profile", bins=INJECTED_TRUTH_BINS,
            range=INJECTED_TRUTH_RANGE, selection=INJECTED_TRUTH_SIDE_SEL,
            selection_vector=["tgl<0", "tgl>=0"], normalize="delta",
            facet_by="side_type", auto_title=False, return_data=True)

    @staticmethod
    def fig49_injected_truth_weights_facet(owner):
        return owner.draw(
            "known_delta:sector", type="profile", bins=INJECTED_TRUTH_BINS,
            range=INJECTED_TRUTH_RANGE, selection=INJECTED_TRUTH_SIDE_SEL,
            weights_vector=["oracle_w_flat", "oracle_w_tgl"],
            weights_labels=["flat", "tgl-weighted"], vector_compose="outer",
            facet_by="side_type", auto_title=False, return_data=True)

    @staticmethod
    def fig50_injected_truth_gaussian_fit(owner):
        return owner.draw(
            "oracle_residual", type="hist", bins=100, range=(-0.1, 0.1),
            fit="gauss", selection=INJECTED_TRUTH_BASE_SEL, auto_title=False)

    @staticmethod
    def fig51_injected_truth_facet_residual(owner):
        return owner.draw(
            "oracle_residual:sector", type="profile", bins=INJECTED_TRUTH_BINS,
            range=INJECTED_TRUTH_RANGE, selection=INJECTED_TRUTH_SIDE_SEL,
            facet_by="side_type", auto_title=False, return_data=True)


def _it_typed_profile_delta_positive_control(adf: Any) -> dict:
    """Positive neighbor for ORACLE-05: typed profile() normalization works."""
    owner = _DirectDFDrawOwner(adf.df)
    raw = owner.profile(
        "[dcar_distorted_clean,dcar_tpc_vertex]:sector",
        bins=INJECTED_TRUTH_BINS,
        range=INJECTED_TRUTH_RANGE,
        selection=INJECTED_TRUTH_BASE_SEL,
        normalize="delta",
        auto_title=False,
        return_data=True,
    )
    try:
        stats = raw[2] if isinstance(raw, tuple) and len(raw) >= 3 else None
        nd = stats.get("normalize_data") if isinstance(stats, dict) else None
        if nd is None or not hasattr(nd, "columns") or "value" not in nd.columns:
            return {"status": "FAIL", "detail": "typed profile() returned no normalize_data.value"}
        observed = np.asarray(nd["value"], dtype=float)
        model = _it_semantic_model(adf)
        ref = _it_profile(model["sector"], model["delta"], model["base"], accumulator="float64")
        cmp = np.allclose(
            np.asarray(ref["y_mean"], dtype=float), observed,
            rtol=1e-8, atol=1e-9, equal_nan=True)
        return {
            "status": "PASS",
            "detail": "typed DFDraw.profile bracket-vector exposes normalize_data.value",
            "normalize_payload_present": True,
            "numerically_within_current_oracle_tolerance": bool(cmp),
            "max_abs": float(np.nanmax(np.abs(np.asarray(ref["y_mean"], dtype=float) - observed)))
                       if np.any(np.isfinite(observed)) else float("nan"),
            "atol": 1e-9,
            "rtol": 1e-8,
        }
    except Exception as exc:
        return {"status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            plt.close(raw[0])
        except Exception:
            pass


def _it_direct_expected_observed(case_id: str, adf: Any) -> tuple[dict, dict]:
    owner = _DirectDFDrawOwner(adf.df)
    gallery = _DirectInjectedGallery()
    if case_id == INJECTED_TRUTH_VECTOR_CASE_ID:
        return _it_vector_overlay(owner, gallery)
    if case_id == INJECTED_TRUTH_DELTA_CASE_ID:
        return _it_direct_delta(owner, gallery)
    if case_id == INJECTED_TRUTH_SELECTION_CASE_ID:
        return _it_selection_delta(owner, gallery)
    if case_id == INJECTED_TRUTH_WEIGHTS_CASE_ID:
        return _it_weights(owner, gallery)
    if case_id == INJECTED_TRUTH_GAUSS_CASE_ID:
        exp, got, _ = _it_gauss(owner, gallery)
        return exp, got
    if case_id == INJECTED_TRUTH_FACET_CASE_ID:
        return _it_facet(owner, gallery)
    raise HarnessError(f"no direct injected-truth route for {case_id}")


def _case_observables_match(case: CaseSpec, expected: dict, observed: dict) -> tuple[bool, str]:
    failures = []
    for obs in case.observables:
        cmp = compare_observable(obs, expected[obs.name], observed[obs.name])
        if not cmp.ok:
            failures.append(f"{obs.name}: {cmp.detail}")
    return (not failures), "; ".join(failures)


def _classify_injected_failure(case: CaseSpec, adf: Any, semantic_diag: dict) -> dict:
    """Complete L0/L1/L2/L3 attribution without conflating divergence and ownership."""
    diag = copy.deepcopy(semantic_diag)
    if not diag.get("L1_all_within_tolerance", False):
        diag.update({
            "first_disagreement_layer": "L1",
            "contract_reference_status": "VERIFIED",
            "owner_status": "ADF",
            "derived_owner": "ADF",
            "L2": "not executed because L1 already disagrees",
            "L3": "adf.draw red",
        })
        return diag
    try:
        expected, observed = _it_direct_expected_observed(case.case_id, adf)
        direct_ok, detail = _case_observables_match(case, expected, observed)
        diag["L2"] = "direct DFDraw on materialized plain DataFrame"
        diag["L2_matches_truth"] = bool(direct_ok)
        diag["L2_detail"] = detail
        diag["L3"] = "adf.draw red"

        if case.case_id in {INJECTED_TRUTH_VECTOR_CASE_ID, INJECTED_TRUTH_SELECTION_CASE_ID}:
            numeric = _it_numeric_recheck_diagnostics(case.case_id, adf, observed)
            diag["numerical_recheck"] = numeric
            diag.update(_numeric_owner_from_diagnostics(
                l1_ok=True, diagnostics=numeric["comparisons"]))
            return diag

        if case.case_id == INJECTED_TRUTH_DELTA_CASE_ID:
            diag.update({
                "first_disagreement_layer": "L2" if not direct_ok else "L3",
                "contract_reference_status": "VERIFIED",
                "owner_status": "DFDRAW" if not direct_ok else "ADF",
                "derived_owner": "dfdraw" if not direct_ok else "ADF_DRAW_BRIDGE",
                "typed_profile_positive_control": _it_typed_profile_delta_positive_control(adf),
                "confirmed_bug_scope": (
                    "top-level draw() bracket-vector profile bypasses typed-profile normalization; "
                    "normalize='delta' is silently lost"),
            })
            return diag

        if direct_ok:
            diag.update({
                "first_disagreement_layer": "L3",
                "contract_reference_status": "VERIFIED",
                "owner_status": "ADF",
                "derived_owner": "ADF_DRAW_BRIDGE",
            })
        else:
            diag.update({
                "first_disagreement_layer": "L2",
                "contract_reference_status": "VERIFIED",
                "owner_status": "DFDRAW",
                "derived_owner": "dfdraw",
            })
    except Exception as exc:
        diag.update({
            "L2": "direct DFDraw on materialized plain DataFrame",
            "L2_matches_truth": False,
            "L2_detail": f"{type(exc).__name__}: {exc}",
            "L3": "adf.draw red",
            "first_disagreement_layer": "L2",
        })
        if case.case_id == INJECTED_TRUTH_DELTA_CASE_ID:
            # ORACLE-05 is structural: direct top-level draw() itself exposes no
            # normalized payload.  That is a verified dispatch-door divergence,
            # not a numerical-reference ambiguity.
            diag.update({
                "contract_reference_status": "VERIFIED",
                "owner_status": "DFDRAW",
                "derived_owner": "dfdraw",
                "confirmed_bug_scope": (
                    "top-level draw() bracket-vector profile bypasses typed-profile normalization; "
                    "normalize='delta' is silently lost"),
            })
            try:
                diag["typed_profile_positive_control"] = _it_typed_profile_delta_positive_control(adf)
            except Exception as control_exc:
                diag["typed_profile_positive_control"] = {
                    "status": "FAIL",
                    "detail": f"{type(control_exc).__name__}: {control_exc}",
                }
        else:
            diag.update({
                "contract_reference_status": "UNRESOLVED",
                "owner_status": "UNRESOLVED",
                "derived_owner": "UNKNOWN",
            })
    return diag


def _record_stage_a_machine_status(adf: Any, result: CaseResult) -> None:
    status_map = getattr(adf, "_stage_a_machine_status", None)
    if not isinstance(status_map, dict):
        status_map = {}
        setattr(adf, "_stage_a_machine_status", status_map)
    status_map[result.case_id] = {
        "status": result.status,
        "detail": result.detail,
    }


def run_injected_truth_case(case: CaseSpec, root_path: str, *, gallery_module=None,
                            prepared_adf=None, prepared_provenance=None) -> CaseResult:
    """Execute one approved injected-truth workflow on the shared EAGER 20% ADF."""
    t0 = time.time()
    res = CaseResult(case_id=case.case_id, status=SKIP)
    adf = prepared_adf
    try:
        if adf is None:
            raise HarnessError(
                "injected-truth cases require the shared FAST prepared_adf; rebuilding is forbidden")
        gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
        provenance = _a6_4_prepared_fraction_sample_evidence(
            adf, prepared_provenance, root_path)
        semantic_diag = _it_semantic_dtype_diagnostics(adf, gallery)
        res.observed["ownership_ladder"] = semantic_diag

        if case.case_id == INJECTED_TRUTH_VECTOR_CASE_ID:
            expected, observed = _it_vector_overlay(adf, gallery)
            diagnostics = {}
        elif case.case_id == INJECTED_TRUTH_DELTA_CASE_ID:
            expected, observed = _it_direct_delta(adf, gallery)
            diagnostics = {}
        elif case.case_id == INJECTED_TRUTH_SELECTION_CASE_ID:
            expected, observed = _it_selection_delta(adf, gallery)
            diagnostics = {}
        elif case.case_id == INJECTED_TRUTH_WEIGHTS_CASE_ID:
            expected, observed = _it_weights(adf, gallery)
            diagnostics = {}
        elif case.case_id == INJECTED_TRUTH_GAUSS_CASE_ID:
            expected, observed, diagnostics = _it_gauss(adf, gallery)
        elif case.case_id == INJECTED_TRUTH_FACET_CASE_ID:
            expected, observed = _it_facet(adf, gallery)
            diagnostics = {}
        else:
            raise HarnessError(f"unknown injected-truth case {case.case_id}")

        first_failure = None
        for obs in case.observables:
            res.observable_contract.append(_contract(obs))
            cmp = compare_observable(obs, expected[obs.name], observed[obs.name])
            res.comparisons.append(comparison_evidence(
                obs, cmp, reference_label="raw NumPy/pandas known injected truth",
                candidate_label=f"public {case.canonical_spec['gallery_function']}"))
            if not cmp.ok and first_failure is None:
                first_failure = f"{obs.name}: {cmp.detail}"
        res.executed_comparisons = len(case.observables)
        res.observed.update({
            "realdata_provenance": dict(prepared_provenance or provenance),
            "injected_truth": {
                "noise_seed": INJECTED_TRUTH_NOISE_SEED,
                "noise_sigma": INJECTED_TRUTH_NOISE_SIGMA,
                "known_delta_expression": INJECTED_TRUTH_DELTA_EXPR,
                "gallery_function": case.canonical_spec["gallery_function"],
                "diagnostics": diagnostics,
            },
        })
        if first_failure is not None:
            res.observed["ownership_ladder"] = _classify_injected_failure(
                case, adf, semantic_diag)
            raise HarnessError(first_failure)
        if case.case_id in {INJECTED_TRUTH_VECTOR_CASE_ID, INJECTED_TRUTH_SELECTION_CASE_ID}:
            numeric = _it_numeric_recheck_diagnostics(case.case_id, adf, observed)
            semantic_diag["numerical_recheck"] = numeric
            semantic_diag.update(_numeric_owner_from_diagnostics(
                l1_ok=True, diagnostics=numeric["comparisons"]))
            semantic_diag.update({
                "L2": "public result matches corrected float64 contract",
                "L3": "adf.draw matches corrected truth",
            })
        else:
            semantic_diag.update({
                "first_disagreement_layer": "NONE",
                "contract_reference_status": "VERIFIED",
                "owner_status": "NONE",
                "derived_owner": "NONE",
                "L2": "not required: public correctness oracle is green",
                "L3": "adf.draw matches truth",
            })
        res.observed["ownership_ladder"] = semantic_diag
        res.status = PASS
        res.detail = ""
        return res
    except Exception as exc:
        # A helper may fail before the observable-comparison loop (for example a
        # missing normalize payload).  In that case the initial L1 diagnostic is
        # not yet an ownership result: complete the L2/L3 ladder here.
        if adf is not None:
            current = res.observed.get("ownership_ladder", {})
            layer = current.get("first_disagreement_layer") if isinstance(current, dict) else None
            if layer in {None, ">=L2", "UNKNOWN"}:
                try:
                    res.observed["ownership_ladder"] = _classify_injected_failure(
                        case, adf, semantic_diag)
                except Exception as diag_exc:
                    res.observed["ownership_ladder"] = {
                        "first_disagreement_layer": "UNKNOWN",
                        "derived_owner": "UNKNOWN",
                        "diagnostic_error": f"{type(diag_exc).__name__}: {diag_exc}",
                    }
        res.status = FAIL
        res.detail = f"INJECTED_TRUTH FAIL: {exc}"
        res.exception = traceback.format_exc(limit=8)
        return res
    finally:
        res.wall_time_s = round(time.time() - t0, 4)
        if adf is not None:
            _record_stage_a_machine_status(adf, res)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_77 hardening v0.2 — STEP 5a / O3 contract + dependency instrumentation
# ─────────────────────────────────────────────────────────────────────────────

HARDENING_O3_CASE_ID = "I4-REAL-G7-SUBFRAME-EAGER-LAZY-BOTH-FULL-01"
HARDENING_O3_REUSED_GALLERY_FUNCTION = A5_2_GALLERY_FUNCTION  # fig32_subframe_vertex
HARDENING_O3_TREE_NAME = A5_3_TREE_NAME
HARDENING_O3_SETUP_REQUIRED = ("timeMS",)
HARDENING_O3_REQUIRED_ON_DEMAND = (
    "vertex_x", "vertex_y", "vertex_z", "vertex_nContributors",
)
HARDENING_O3_DECOY_BRANCH = "ncl"


def o3_fullstack_case(root_path: str, gallery_module=None) -> CaseSpec:
    """Declare the SLOW BOTH+FULL G7.32 eager/lazy equivalence gate.

    Step 5a defines the contract only.  The expensive FULL EAGER and FULL LAZY
    legs are executed separately in later steps so one gate never performs a
    redundant third source read merely to compute combined timing.
    """
    status, reason = _a5_2_environment_status(root_path, gallery_module=gallery_module)
    applicable = status != A5_2_ENV_UNAVAILABLE
    return CaseSpec(
        case_id=HARDENING_O3_CASE_ID,
        claim_id="I4.real_g7_subframe_eager_lazy.HARDENING.O3",
        title="FULL G7.32 CalibVertex profile is numerically equivalent in eager and lazy modes",
        claim=("the exact trusted G7.32 CalibVertex.vertex_x_intercept:time_s profile "
               "produces the same selected/bin profile observables in one FULL EAGER and "
               "one FULL LAZY execution, while the lazy leg loads required vertex branches "
               "on demand and leaves the available ncl decoy unloaded"),
        failure_means=("EAGER and LAZY full-data G7.32 differ numerically, the lazy leg falls "
                       "back to eager/full-column loading, a required dependency is not loaded, "
                       "or the unrelated ncl decoy is loaded"),
        expected_visual=("optional separate G7.32 EAGER/LAZY overlay and residual artifact; "
                         "not part of the 47-page FAST gallery"),
        owner_on_failure="ADF",
        purpose="INVARIANCE",
        gate="ENVIRONMENT_GATED",
        oracle_kind="CONSISTENCY",
        loading_mode="BOTH",
        sample_mode="FULL",
        canonical_spec={
            "gallery_function": HARDENING_O3_REUSED_GALLERY_FUNCTION,
            "expr": "CalibVertex.vertex_x_intercept:time_s",
            "type": "profile",
            "bins": 100,
            "time_format": "%H:%M",
            "sample": None,
            "tree_name": HARDENING_O3_TREE_NAME,
            "reused_case_id": A5_3_CASE_ID,
        },
        applicable=applicable,
        applicability_reason=reason if not applicable else "",
        setup_contract=("reuse the existing G7.32 gallery owner and Stage-A lazy-reader "
                        "loaded_branches instrumentation; execute exactly one FULL EAGER leg "
                        "and one FULL LAZY leg in separate measured steps; compare saved machine "
                        "observables without a third ROOT read"),
        preconditions=(
            "the ROOT input is readable by the trusted time-series environment",
            "fig32_subframe_vertex and build_adf are available",
            "sample is None in both FULL legs",
            "the LAZY leg exposes available_branches and loaded_branches",
            "ncl exists as an available but unrelated physical branch",
        ),
        surfaces_under_test=("draw",),
        observables=(
            Observable("count", "STATS", "ARRAY", "profile_data.count"),
            Observable("x_center", "STATS", "ARRAY", "profile_data.x_center",
                       comparator="close", atol=1e-12, rtol=1e-10,
                       rationale="identical explicit profile-bin coordinates across loading modes"),
            Observable("y_mean", "STATS", "ARRAY", "profile_data.y_mean",
                       comparator="close", atol=1e-10, rtol=1e-8,
                       rationale="same full-data CalibVertex profile central values across loading modes"),
        ),
        figure_contract=FigureContract(
            expected_panels="optional two-panel slow-gate artifact, separate from FAST gallery",
            panel_roles="top: EAGER and LAZY G7.32 profiles; bottom: LAZY minus EAGER residual",
            expected_traces="two profile traces plus one residual trace when artifact rendering is enabled",
            expected_group_count="one G7.32 profile in each loading mode",
            primary_comparison="saved EAGER_FULL versus LAZY_FULL profile_data arrays",
            residual_definition="LAZY y_mean - EAGER y_mean on matched populated bins",
            accepted_envelope="declared observables agree within explicit tolerances and dependency evidence is valid",
            case_ids=(HARDENING_O3_CASE_ID,),
            proof_kind="CONSISTENCY",
        ),
        non_claims=(
            "O3 is loading-mode consistency, not independent calibVertex physics correctness",
            "O3 is not part of the routine 47-page FAST gallery",
            "O3 does not permit sampled-lazy execution",
        ),
        negative_control="FAMILY_MUTATION:O3_REQUIRED_DEPENDENCY_OR_DECOY_OR_NUMERICAL_MISMATCH",
        reference_policy="same-process",
    )


def o3_execution_contract() -> dict:
    """Persistent SLOW-gate execution plan used by later FULL leg runners."""
    return {
        "case_id": HARDENING_O3_CASE_ID,
        "reused_gallery_function": HARDENING_O3_REUSED_GALLERY_FUNCTION,
        "reused_stage_a_case_id": A5_3_CASE_ID,
        "loading_mode": "BOTH",
        "sample_mode": "FULL",
        "execution_legs": ("EAGER_FULL", "LAZY_FULL"),
        "max_full_source_constructions": 2,
        "redundant_combined_third_read_forbidden": True,
        "combined_wall_time_definition": "EAGER_FULL elapsed + LAZY_FULL elapsed from the same two measured legs",
        "dependency_instrumentation_owner": "_a5_3_loaded_branches / lazy_reader.loaded_branches",
        "setup_required_physical_branches": HARDENING_O3_SETUP_REQUIRED,
        "required_on_demand_physical_branches": HARDENING_O3_REQUIRED_ON_DEMAND,
        "decoy_available_but_unrequired_branch": HARDENING_O3_DECOY_BRANCH,
        "fast_gallery_page_count_effect": 0,
    }


def _o3_lazy_dependency_snapshot(adf: Any) -> dict:
    """Reuse the Stage-A lazy-reader branch evidence without new instrumentation."""
    reader = getattr(adf, "_lazy_reader", None)
    if reader is None:
        raise HarnessError("O3 lazy dependency evidence has no live _lazy_reader (eager fallback)")
    loaded = set(getattr(reader, "loaded_branches", ()) or ())
    available = set(getattr(reader, "available_branches", ()) or ())
    if not available:
        raise HarnessError("O3 lazy dependency evidence has no available_branches inventory")
    decoy = HARDENING_O3_DECOY_BRANCH
    if decoy not in available:
        raise HarnessError(f"O3 declared decoy {decoy!r} is not available in the lazy source")
    return {
        "available": tuple(sorted(str(x) for x in available)),
        "loaded": tuple(sorted(str(x) for x in loaded)),
        "decoy": decoy,
    }


def _o3_validate_lazy_dependency_transition(before: dict, after: dict) -> dict:
    """Validate required G7.32 expansion and the available-but-unrequired decoy."""
    before_loaded = set(before.get("loaded", ()))
    after_loaded = set(after.get("loaded", ()))
    available = set(after.get("available", ()))
    required = set(HARDENING_O3_REQUIRED_ON_DEMAND)
    decoy = HARDENING_O3_DECOY_BRANCH
    missing = sorted(required - after_loaded)
    if missing:
        raise HarnessError(f"O3 required dependency missing after G7.32: {missing}")
    if required & before_loaded:
        raise HarnessError(
            f"O3 required on-demand dependency was preloaded before G7.32: "
            f"{sorted(required & before_loaded)}")
    if decoy not in available:
        raise HarnessError(f"O3 decoy {decoy!r} disappeared from available branch inventory")
    if decoy in after_loaded:
        raise HarnessError(f"O3 unrelated decoy branch {decoy!r} was loaded")
    newly_loaded = sorted(after_loaded - before_loaded)
    return {
        "required_on_demand": sorted(required),
        "newly_loaded": newly_loaded,
        "decoy": decoy,
        "decoy_remained_unloaded": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_77 hardening v0.2 — STEP 5b / O3 EAGER FULL leg
# ─────────────────────────────────────────────────────────────────────────────


def _o3_timestamp() -> str:
    """Local timestamp for long-running O3 debug/evidence messages."""
    return time.strftime("%Y-%m-%d %H:%M:%S %z", time.localtime())


def _o3_peak_rss_mb() -> float:
    """Best-effort process peak RSS in MiB (portable Linux/macOS units)."""
    try:
        import resource
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        if sys.platform == "darwin":
            return value / (1024.0 * 1024.0)
        return value / 1024.0
    except Exception:
        return float("nan")


def _o3_profile_arrays_from_stats(stats: Any) -> dict:
    """Extract comparison-ready profile arrays from return_data=True stats.

    JSON-facing y values use ``None`` for non-finite/missing bins.  ``missing_mask``
    is authoritative for restoring NaNs before the final EAGER↔LAZY comparison.
    """
    if not isinstance(stats, dict):
        raise HarnessError("O3 profile stats payload is not a dict")
    profile_data = stats.get("profile_data")
    if not (hasattr(profile_data, "columns") and hasattr(profile_data, "__getitem__")):
        raise HarnessError("O3 profile stats has no DataFrame-like profile_data")
    required = ("count", "x_center", "y_mean")
    missing_cols = [name for name in required if name not in profile_data.columns]
    if missing_cols:
        raise HarnessError(f"O3 profile_data missing columns: {missing_cols}")

    count = np.asarray(profile_data["count"])
    x_center = np.asarray(profile_data["x_center"], dtype=float)
    y_mean = np.asarray(profile_data["y_mean"], dtype=float)
    if not (count.shape == x_center.shape == y_mean.shape):
        raise HarnessError(
            "O3 profile_data shape mismatch: "
            f"count={count.shape}, x_center={x_center.shape}, y_mean={y_mean.shape}")
    if count.ndim != 1:
        raise HarnessError(f"O3 profile_data must be one-dimensional; got {count.shape}")
    if count.size == 0:
        raise HarnessError("O3 profile_data is empty")
    if not np.all(np.isfinite(x_center)):
        raise HarnessError("O3 x_center contains non-finite values")

    missing = (np.asarray(count, dtype=float) <= 0.0) | ~np.isfinite(y_mean)
    if int((~missing).sum()) <= 0:
        raise HarnessError("O3 profile_data has no populated finite y_mean bins")

    return {
        "count": [int(x) for x in np.asarray(count, dtype=np.int64)],
        "x_center": [float(x) for x in x_center],
        "y_mean": [None if bad else float(y) for y, bad in zip(y_mean, missing)],
        "missing_mask": [bool(x) for x in missing],
        "populated_bins": int((~missing).sum()),
    }


def run_o3_eager_full_leg(root_path: str, *, manifest_path: str,
                          gallery_module=None) -> tuple[CaseResult, dict]:
    """Execute exactly one FULL EAGER O3 leg and persist comparison-ready evidence.

    This is deliberately *not* the final O3 PASS gate.  It writes an ordinary
    Stage-A manifest with a DIAGNOSTIC partial-leg result.  Step 5c writes the
    corresponding LAZY_FULL leg; Step 5d compares the two saved legs without a
    third ROOT construction.
    """
    case = o3_fullstack_case(root_path, gallery_module=gallery_module)
    result = CaseResult(case_id=case.case_id, status=DIAGNOSTIC)
    started = _o3_timestamp()
    t_leg = time.time()
    rss_before = _o3_peak_rss_mb()
    raw_gallery = None
    raw_machine = None
    original_sample = None

    print(f"[O3 EAGER_FULL] {started} START")
    try:
        if not case.applicable:
            result.status = SKIP
            result.detail = case.applicability_reason
        else:
            gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()
            import pandas as pd

            sample_calls = []
            original_sample = pd.DataFrame.sample

            def forbidden_sample(self, *args, **kwargs):
                sample_calls.append({"args": list(args), "kwargs": dict(kwargs)})
                return original_sample(self, *args, **kwargs)

            print(f"[O3 EAGER_FULL] {_o3_timestamp()} BUILD BEGIN")
            t_build = time.time()
            pd.DataFrame.sample = forbidden_sample
            try:
                adf = gallery.build_adf(
                    root_path, sample=None, lazy=False,
                    tree_name=case.canonical_spec["tree_name"])
            finally:
                pd.DataFrame.sample = original_sample
                original_sample = None
            build_wall = time.time() - t_build
            print(f"[O3 EAGER_FULL] {_o3_timestamp()} BUILD END wall={build_wall:.3f}s")

            if sample_calls:
                raise HarnessError(
                    f"O3 EAGER_FULL unexpectedly called DataFrame.sample {len(sample_calls)} time(s)")
            if getattr(adf, "_lazy_reader", None) is not None:
                raise HarnessError("O3 EAGER_FULL unexpectedly exposes a lazy reader")

            print(f"[O3 EAGER_FULL] {_o3_timestamp()} G7.32 BEGIN")
            t_exec = time.time()
            gallery_fn = getattr(gallery, HARDENING_O3_REUSED_GALLERY_FUNCTION)
            raw_gallery = gallery_fn(adf)
            if raw_gallery is None:
                raise HarnessError("O3 EAGER_FULL trusted G7.32 returned None")
            gallery_payload = unwrap("draw", raw_gallery)
            if not isinstance(gallery_payload.stats, dict):
                raise HarnessError("O3 EAGER_FULL G7.32 returned non-dict stats")

            sf = adf.get_subframe("CalibVertex")
            if sf is None or "vertex_x_intercept" not in sf.df.columns:
                raise HarnessError("O3 EAGER_FULL did not create CalibVertex.vertex_x_intercept")
            if "vertex_x_intercept" in adf.df.columns:
                raise HarnessError("O3 EAGER_FULL contaminated parent with subframe-only column")

            # Reuse the exact G7.32 logical request, adding only return_data=True
            # to expose machine-comparison arrays.  The expensive source build is
            # not repeated; CalibVertex is already materialized by fig32.
            raw_machine = adf.draw(
                case.canonical_spec["expr"], type="profile",
                bins=int(case.canonical_spec["bins"]),
                time_format=case.canonical_spec["time_format"],
                auto_title=True, return_data=True)
            machine_payload = unwrap("draw", raw_machine)
            arrays = _o3_profile_arrays_from_stats(machine_payload.stats)
            execute_wall = time.time() - t_exec
            print(f"[O3 EAGER_FULL] {_o3_timestamp()} G7.32 END wall={execute_wall:.3f}s")

            st = os.stat(root_path)
            rss_after = _o3_peak_rss_mb()
            result.observed = {
                "o3_partial_leg": "EAGER_FULL",
                "o3_final_gate": False,
                "profile_data": arrays,
                "realdata_provenance": {
                    "input_path": os.path.abspath(root_path),
                    "input_size_bytes": int(st.st_size),
                    "input_mtime_ns": int(st.st_mtime_ns),
                    "tree_name": case.canonical_spec["tree_name"],
                    "loading_mode": "EAGER",
                    "sample_mode": "FULL",
                    "sample_fraction": None,
                    "sample_seed": None,
                    "source_rows": int(len(adf.df)),
                    "full_source_constructions_this_leg": 1,
                },
                "performance": {
                    "build_wall_time_s": round(build_wall, 6),
                    "execute_wall_time_s": round(execute_wall, 6),
                    "leg_wall_time_s": round(time.time() - t_leg, 6),
                    "peak_rss_mb_before": None if not np.isfinite(rss_before) else round(rss_before, 3),
                    "peak_rss_mb_after": None if not np.isfinite(rss_after) else round(rss_after, 3),
                },
            }
            result.payload_paths = {
                "G7.32/gallery": list(gallery_payload.path),
                "G7.32/return_data": list(machine_payload.path),
            }
            result.detail = (
                "O3 EAGER_FULL leg complete; final O3 PASS requires the separately "
                "measured LAZY_FULL leg and Step-5d comparison")
            result.status = DIAGNOSTIC
    except Exception as exc:
        result.status = FAIL
        result.detail = f"O3_EAGER_FULL FAIL: {type(exc).__name__}: {exc}"
        result.exception = traceback.format_exc(limit=8)
    finally:
        if original_sample is not None:
            try:
                import pandas as pd
                pd.DataFrame.sample = original_sample
            except Exception:
                pass
        _close()
        result.wall_time_s = round(time.time() - t_leg, 4)

    finished = _o3_timestamp()
    extra = {
        "stage_a_gate": "O3_EAGER_FULL_PARTIAL",
        "o3_execution_contract": o3_execution_contract(),
        "o3_leg": "EAGER_FULL",
        "o3_partial_evidence": True,
        "o3_started_local": started,
        "o3_finished_local": finished,
        "o3_full_source_constructions_this_manifest": 1 if result.status != SKIP else 0,
    }
    doc = write_manifest(manifest_path, [result], [case], extra=extra)
    print(f"[O3 EAGER_FULL] {finished} END status={result.status} wall={result.wall_time_s:.3f}s")
    return result, doc


# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_77 hardening v0.2 — STEP 5c / O3 LAZY FULL leg
# ─────────────────────────────────────────────────────────────────────────────


def run_o3_lazy_full_leg(root_path: str, *, manifest_path: str,
                         gallery_module=None) -> tuple[CaseResult, dict]:
    """Execute exactly one FULL LAZY O3 leg and persist comparison-ready evidence.

    The leg reuses the canonical Stage-A FULL+LAZY builder and its lazy-reader
    instrumentation.  Required CalibVertex physical branches must be absent
    before G7.32 and loaded on demand afterwards; the available-but-unrequired
    ``ncl`` decoy must remain unloaded.  This is partial evidence only: Step 5d
    compares this saved leg with the separately measured EAGER_FULL manifest,
    without constructing the ROOT source a third time.
    """
    case = o3_fullstack_case(root_path, gallery_module=gallery_module)
    result = CaseResult(case_id=case.case_id, status=DIAGNOSTIC)
    started = _o3_timestamp()
    t_leg = time.time()
    rss_before = _o3_peak_rss_mb()

    print(f"[O3 LAZY_FULL] {started} START")
    try:
        if not case.applicable:
            result.status = SKIP
            result.detail = case.applicability_reason
        else:
            gallery = gallery_module if gallery_module is not None else _a5_2_import_gallery()

            print(f"[O3 LAZY_FULL] {_o3_timestamp()} BUILD BEGIN")
            t_build = time.time()
            adf, setup_provenance = _a6_4_build_lazy_adf_once(
                root_path, gallery_module=gallery)
            build_wall = time.time() - t_build
            print(f"[O3 LAZY_FULL] {_o3_timestamp()} BUILD END wall={build_wall:.3f}s")

            before = _o3_lazy_dependency_snapshot(adf)
            print(
                f"[O3 LAZY_FULL] {_o3_timestamp()} DEPENDENCY BEFORE "
                f"loaded={len(before['loaded'])} available={len(before['available'])}")

            print(f"[O3 LAZY_FULL] {_o3_timestamp()} G7.32 BEGIN")
            t_exec = time.time()
            gallery_fn = getattr(gallery, HARDENING_O3_REUSED_GALLERY_FUNCTION)
            raw_gallery = gallery_fn(adf)
            if raw_gallery is None:
                raise HarnessError("O3 LAZY_FULL trusted G7.32 returned None")
            gallery_payload = unwrap("draw", raw_gallery)
            if not isinstance(gallery_payload.stats, dict):
                raise HarnessError("O3 LAZY_FULL G7.32 returned non-dict stats")

            after_gallery = _o3_lazy_dependency_snapshot(adf)
            dependency_evidence = _o3_validate_lazy_dependency_transition(
                before, after_gallery)

            sf = adf.get_subframe("CalibVertex")
            if sf is None or "vertex_x_intercept" not in sf.df.columns:
                raise HarnessError("O3 LAZY_FULL did not create CalibVertex.vertex_x_intercept")
            if "vertex_x_intercept" in adf.df.columns:
                raise HarnessError("O3 LAZY_FULL contaminated parent with subframe-only column")

            # Reuse the exact G7.32 request and the already-built lazy ADF.  This
            # exposes machine arrays without a second source construction.
            raw_machine = adf.draw(
                case.canonical_spec["expr"], type="profile",
                bins=int(case.canonical_spec["bins"]),
                time_format=case.canonical_spec["time_format"],
                auto_title=True, return_data=True)
            machine_payload = unwrap("draw", raw_machine)
            arrays = _o3_profile_arrays_from_stats(machine_payload.stats)

            # The machine-observable extraction itself must not pull the decoy.
            after_machine = _o3_lazy_dependency_snapshot(adf)
            final_dependency_evidence = _o3_validate_lazy_dependency_transition(
                before, after_machine)
            execute_wall = time.time() - t_exec
            print(f"[O3 LAZY_FULL] {_o3_timestamp()} G7.32 END wall={execute_wall:.3f}s")

            st = os.stat(root_path)
            rss_after = _o3_peak_rss_mb()
            result.observed = {
                "o3_partial_leg": "LAZY_FULL",
                "o3_final_gate": False,
                "profile_data": arrays,
                "lazy_dependency_evidence": {
                    "before": before,
                    "after_gallery": after_gallery,
                    "after_machine": after_machine,
                    "transition_after_gallery": dependency_evidence,
                    "transition_after_machine": final_dependency_evidence,
                },
                "realdata_provenance": {
                    **dict(setup_provenance),
                    "input_path": os.path.abspath(root_path),
                    "input_size_bytes": int(st.st_size),
                    "input_mtime_ns": int(st.st_mtime_ns),
                    "tree_name": case.canonical_spec["tree_name"],
                    "loading_mode": "LAZY",
                    "sample_mode": "FULL",
                    "sample_fraction": None,
                    "sample_seed": None,
                    "source_rows": int(len(adf.df)),
                    "full_source_constructions_this_leg": 1,
                    "eager_fallback": False,
                },
                "performance": {
                    "build_wall_time_s": round(build_wall, 6),
                    "execute_wall_time_s": round(execute_wall, 6),
                    "leg_wall_time_s": round(time.time() - t_leg, 6),
                    "peak_rss_mb_before": None if not np.isfinite(rss_before) else round(rss_before, 3),
                    "peak_rss_mb_after": None if not np.isfinite(rss_after) else round(rss_after, 3),
                },
            }
            result.payload_paths = {
                "G7.32/gallery": list(gallery_payload.path),
                "G7.32/return_data": list(machine_payload.path),
            }
            result.detail = (
                "O3 LAZY_FULL leg complete with required dependency expansion and "
                "decoy-unloaded evidence; final O3 PASS requires Step-5d comparison "
                "with the separately measured EAGER_FULL leg")
            result.status = DIAGNOSTIC
    except Exception as exc:
        result.status = FAIL
        result.detail = f"O3_LAZY_FULL FAIL: {type(exc).__name__}: {exc}"
        result.exception = traceback.format_exc(limit=8)
    finally:
        _close()
        result.wall_time_s = round(time.time() - t_leg, 4)

    finished = _o3_timestamp()
    extra = {
        "stage_a_gate": "O3_LAZY_FULL_PARTIAL",
        "o3_execution_contract": o3_execution_contract(),
        "o3_leg": "LAZY_FULL",
        "o3_partial_evidence": True,
        "o3_started_local": started,
        "o3_finished_local": finished,
        "o3_full_source_constructions_this_manifest": 1 if result.status != SKIP else 0,
    }
    doc = write_manifest(manifest_path, [result], [case], extra=extra)
    print(f"[O3 LAZY_FULL] {finished} END status={result.status} wall={result.wall_time_s:.3f}s")
    return result, doc


# ─────────────────────────────────────────────────────────────────────────────
# PHASE_13_77 hardening v0.2 — STEP 5d / O3 saved-leg adjudication + PDF
# ─────────────────────────────────────────────────────────────────────────────


def _o3_load_saved_leg_manifest(path: str, expected_leg: str) -> tuple[dict, dict]:
    """Load one Step-5b/5c partial manifest without touching the ROOT source."""
    with open(path) as fh:
        doc = json.load(fh)
    prov = doc.get("provenance", {})
    if prov.get("o3_leg") != expected_leg:
        raise HarnessError(
            f"O3 saved manifest leg mismatch: expected {expected_leg}, got {prov.get('o3_leg')!r}")
    if prov.get("o3_partial_evidence") is not True:
        raise HarnessError(f"O3 {expected_leg} manifest is not marked partial evidence")
    if int(prov.get("o3_full_source_constructions_this_manifest", -1)) != 1:
        raise HarnessError(
            f"O3 {expected_leg} must record exactly one FULL source construction")
    records = [row for row in doc.get("cases", [])
               if row.get("case_id") == HARDENING_O3_CASE_ID]
    if len(records) != 1:
        raise HarnessError(
            f"O3 {expected_leg} manifest must contain exactly one {HARDENING_O3_CASE_ID} record")
    rec = records[0]
    if rec.get("status") != DIAGNOSTIC:
        raise HarnessError(
            f"O3 {expected_leg} partial record must be DIAGNOSTIC; got {rec.get('status')!r}")
    observed = rec.get("observed", {})
    if observed.get("o3_partial_leg") != expected_leg:
        raise HarnessError(
            f"O3 {expected_leg} observed partial-leg marker is missing/wrong")
    return doc, rec


def _o3_restore_saved_profile_arrays(record: dict, *, role: str) -> dict[str, np.ndarray]:
    data = record.get("observed", {}).get("profile_data")
    if not isinstance(data, dict):
        raise HarnessError(f"O3 {role} saved record has no profile_data")
    required = ("count", "x_center", "y_mean", "missing_mask")
    missing = [name for name in required if name not in data]
    if missing:
        raise HarnessError(f"O3 {role} profile_data missing {missing}")
    count = np.asarray(data["count"], dtype=np.int64)
    x_center = np.asarray(data["x_center"], dtype=float)
    missing_mask = np.asarray(data["missing_mask"], dtype=bool)
    y_values = np.asarray([
        np.nan if value is None else float(value) for value in data["y_mean"]
    ], dtype=float)
    if not (count.shape == x_center.shape == missing_mask.shape == y_values.shape):
        raise HarnessError(
            f"O3 {role} saved profile shape mismatch: count={count.shape}, "
            f"x={x_center.shape}, missing={missing_mask.shape}, y={y_values.shape}")
    if count.ndim != 1 or count.size == 0:
        raise HarnessError(f"O3 {role} saved profile must be a non-empty 1-D array")
    reconstructed_missing = (count <= 0) | ~np.isfinite(y_values)
    if not np.array_equal(reconstructed_missing, missing_mask):
        raise HarnessError(f"O3 {role} missing_mask disagrees with count/y_mean")
    return {
        "count": count,
        "x_center": x_center,
        "y_mean": y_values,
        "missing_mask": missing_mask,
    }


def _o3_saved_source_identity(record: dict, *, role: str) -> dict:
    prov = record.get("observed", {}).get("realdata_provenance", {})
    required = (
        "input_path", "input_size_bytes", "input_mtime_ns", "tree_name",
        "sample_mode", "source_rows", "full_source_constructions_this_leg",
    )
    missing = [name for name in required if name not in prov]
    if missing:
        raise HarnessError(f"O3 {role} provenance missing {missing}")
    if prov.get("sample_mode") != "FULL":
        raise HarnessError(f"O3 {role} is not FULL data")
    if int(prov.get("full_source_constructions_this_leg", -1)) != 1:
        raise HarnessError(f"O3 {role} did not record exactly one source construction")
    return {name: prov[name] for name in required}


def _o3_final_dependency_check(lazy_record: dict) -> dict:
    evidence = lazy_record.get("observed", {}).get("lazy_dependency_evidence")
    if not isinstance(evidence, dict):
        raise HarnessError("O3 LAZY_FULL final adjudication has no dependency evidence")
    transition = evidence.get("transition_after_machine")
    if not isinstance(transition, dict):
        raise HarnessError("O3 LAZY_FULL has no final dependency transition")
    newly = set(transition.get("newly_loaded", ()))
    required = set(HARDENING_O3_REQUIRED_ON_DEMAND)
    missing = sorted(required - newly)
    if missing:
        raise HarnessError(f"O3 LAZY_FULL missing required on-demand branches: {missing}")
    if transition.get("decoy") != HARDENING_O3_DECOY_BRANCH:
        raise HarnessError("O3 LAZY_FULL decoy identity drifted")
    if transition.get("decoy_remained_unloaded") is not True:
        raise HarnessError(
            f"O3 unrelated decoy branch {HARDENING_O3_DECOY_BRANCH!r} was loaded")
    return {
        "required_on_demand": sorted(required),
        "newly_loaded": sorted(newly),
        "decoy": HARDENING_O3_DECOY_BRANCH,
        "decoy_remained_unloaded": True,
    }


def _o3_render_saved_comparison_pdf(pdf_path: str, case: CaseSpec,
                                    eager: dict[str, np.ndarray],
                                    lazy: dict[str, np.ndarray],
                                    result: CaseResult) -> None:
    """Render the separate O3 overlay/residual artifact from saved arrays only."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    x = eager["x_center"]
    ey = eager["y_mean"]
    ly = lazy["y_mean"]
    valid = ~(eager["missing_mask"] | lazy["missing_mask"])
    residual = np.full_like(ey, np.nan, dtype=float)
    residual[valid] = ly[valid] - ey[valid]

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.0]})
    axes[0].plot(x[valid], ey[valid], marker="o", ms=2.5, lw=1.0, label="EAGER FULL")
    axes[0].plot(x[valid], ly[valid], marker=".", ms=2.5, lw=1.0, label="LAZY FULL")
    axes[0].set_ylabel("vertex_x_intercept")
    axes[0].set_title("O3 FULL G7.32 EAGER ↔ LAZY equivalence")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.25)

    axes[1].axhline(0.0, lw=0.8)
    axes[1].plot(x[valid], residual[valid], marker=".", ms=2.5, lw=0.9)
    axes[1].set_xlabel("time_s")
    axes[1].set_ylabel("LAZY - EAGER")
    axes[1].grid(True, alpha=0.25)

    max_abs = float(np.nanmax(np.abs(residual[valid]))) if np.any(valid) else float("nan")
    summary = (
        f"STATUS: {result.status}    populated bins: {int(np.count_nonzero(valid))}    "
        f"max |LAZY-EAGER|: {max_abs:.6g}\n"
        f"EXPECTED: {case.expected_visual}\n"
        f"MACHINE ORACLE: count exact; missing mask exact; x_center atol=1e-12 rtol=1e-10; "
        f"y_mean atol=1e-10 rtol=1e-8\n"
        f"FAILURE MEANS: {case.failure_means}"
    )
    fig.subplots_adjust(bottom=0.25, hspace=0.12)
    fig.text(0.02, 0.02, summary, ha="left", va="bottom", fontsize=7,
             family="monospace", wrap=True)
    os.makedirs(os.path.dirname(os.path.abspath(pdf_path)), exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)


def run_o3_saved_leg_comparison(eager_manifest_path: str, lazy_manifest_path: str, *,
                                manifest_path: str, pdf_path: str | None = None) -> tuple[CaseResult, dict]:
    """Final O3 adjudication from saved Step-5b/5c manifests — zero ROOT reads."""
    started = _o3_timestamp()
    t0 = time.time()
    print(f"[O3 FINAL] {started} START — saved-manifest comparison; no ROOT read")

    eager_doc = lazy_doc = eager_rec = lazy_rec = None
    # The path exists in real runs and synthetic tests; o3_fullstack_case uses it
    # only for environment applicability metadata and never reads it here.
    root_hint = ""
    result = CaseResult(case_id=HARDENING_O3_CASE_ID, status=FAIL)
    try:
        eager_doc, eager_rec = _o3_load_saved_leg_manifest(eager_manifest_path, "EAGER_FULL")
        lazy_doc, lazy_rec = _o3_load_saved_leg_manifest(lazy_manifest_path, "LAZY_FULL")
        root_hint = eager_rec["observed"]["realdata_provenance"]["input_path"]
        case = o3_fullstack_case(root_hint)

        eager_id = _o3_saved_source_identity(eager_rec, role="EAGER_FULL")
        lazy_id = _o3_saved_source_identity(lazy_rec, role="LAZY_FULL")
        identity_fields = (
            "input_path", "input_size_bytes", "input_mtime_ns", "tree_name",
            "sample_mode", "source_rows",
        )
        identity_mismatch = {
            key: {"eager": eager_id[key], "lazy": lazy_id[key]}
            for key in identity_fields if eager_id[key] != lazy_id[key]
        }
        if identity_mismatch:
            raise HarnessError(f"O3 source identity mismatch: {identity_mismatch}")

        eager = _o3_restore_saved_profile_arrays(eager_rec, role="EAGER_FULL")
        lazy = _o3_restore_saved_profile_arrays(lazy_rec, role="LAZY_FULL")

        count_cmp = compare_array(eager["count"], lazy["count"], comparator="exact")
        if not count_cmp.ok:
            raise HarnessError(f"O3 count mismatch: {count_cmp.detail}")
        missing_cmp = compare_array(
            eager["missing_mask"], lazy["missing_mask"], comparator="exact")
        if not missing_cmp.ok:
            raise HarnessError(f"O3 missing-mask mismatch: {missing_cmp.detail}")
        x_cmp = compare_array(eager["x_center"], lazy["x_center"], comparator="close",
                              atol=1e-12, rtol=1e-10)
        if not x_cmp.ok:
            raise HarnessError(f"O3 x_center numerical mismatch: {x_cmp.detail}")
        y_cmp = compare_array(eager["y_mean"], lazy["y_mean"], comparator="close",
                              atol=1e-10, rtol=1e-8)
        if not y_cmp.ok:
            raise HarnessError(f"O3 y_mean numerical mismatch: {y_cmp.detail}")

        dependency = _o3_final_dependency_check(lazy_rec)
        eager_perf = eager_rec["observed"].get("performance", {})
        lazy_perf = lazy_rec["observed"].get("performance", {})
        eager_wall = float(eager_perf.get("leg_wall_time_s", eager_rec.get("wall_time_s", 0.0)))
        lazy_wall = float(lazy_perf.get("leg_wall_time_s", lazy_rec.get("wall_time_s", 0.0)))
        rss_values = [
            x for x in (
                eager_perf.get("peak_rss_mb_after"), lazy_perf.get("peak_rss_mb_after")
            ) if x is not None
        ]
        max_rss = max(float(x) for x in rss_values) if rss_values else None

        result.status = PASS
        result.detail = (
            "O3 PASS: saved FULL EAGER and FULL LAZY G7.32 profiles agree; lazy required "
            "vertex dependencies loaded on demand, ncl decoy remained unloaded; final "
            "comparison performed without a third ROOT read")
        result.executed_comparisons = 4
        result.comparisons = [
            {"observable": "count", "ok": True, "comparator": "exact", "detail": ""},
            {"observable": "missing_mask", "ok": True, "comparator": "exact", "detail": ""},
            {"observable": "x_center", "ok": True, "comparator": "close",
             "atol": 1e-12, "rtol": 1e-10, "detail": ""},
            {"observable": "y_mean", "ok": True, "comparator": "close",
             "atol": 1e-10, "rtol": 1e-8, "detail": ""},
        ]
        result.observed = {
            "o3_final_gate": True,
            "profile_data": {
                "populated_bins": int(np.count_nonzero(~eager["missing_mask"])),
                "max_abs_y_mean_delta": float(np.nanmax(np.abs(lazy["y_mean"] - eager["y_mean"]))),
            },
            "source_identity": {key: eager_id[key] for key in identity_fields},
            "lazy_dependency_evidence": dependency,
            "performance": {
                "eager_full_wall_time_s": eager_wall,
                "lazy_full_wall_time_s": lazy_wall,
                "combined_two_leg_wall_time_s": eager_wall + lazy_wall,
                "peak_rss_mb_max": max_rss,
                "full_source_constructions_total": 2,
                "comparison_root_reads": 0,
                "redundant_third_full_read": False,
            },
            "saved_manifest_paths": {
                "eager": os.path.abspath(eager_manifest_path),
                "lazy": os.path.abspath(lazy_manifest_path),
            },
        }
        if pdf_path:
            _o3_render_saved_comparison_pdf(pdf_path, case, eager, lazy, result)
            result.payload_paths["O3/comparison_pdf"] = os.path.abspath(pdf_path)
    except Exception as exc:
        case = o3_fullstack_case(root_hint) if root_hint else CaseSpec(
            case_id=HARDENING_O3_CASE_ID,
            claim_id="I4.real_g7_subframe_eager_lazy.HARDENING.O3",
            title="FULL G7.32 CalibVertex profile is numerically equivalent in eager and lazy modes",
            claim="saved FULL EAGER and FULL LAZY G7.32 evidence agrees",
            failure_means="saved-leg comparison or dependency contract failed",
            expected_visual="separate EAGER/LAZY comparison artifact",
            owner_on_failure="ADF", purpose="INVARIANCE", gate="ENVIRONMENT_GATED",
            oracle_kind="CONSISTENCY", loading_mode="BOTH", sample_mode="FULL")
        result.status = FAIL
        result.detail = f"O3_FINAL FAIL: {type(exc).__name__}: {exc}"
        result.exception = traceback.format_exc(limit=8)
    finally:
        result.wall_time_s = round(time.time() - t0, 4)

    finished = _o3_timestamp()
    extra = {
        "stage_a_gate": "O3_FINAL_SAVED_LEG_COMPARISON",
        "o3_execution_contract": o3_execution_contract(),
        "o3_final_gate": True,
        "o3_comparison_uses_saved_manifests_only": True,
        "o3_comparison_root_reads": 0,
        "o3_full_source_constructions_total": 2,
        "o3_redundant_third_full_read": False,
        "o3_eager_manifest": os.path.abspath(eager_manifest_path),
        "o3_lazy_manifest": os.path.abspath(lazy_manifest_path),
        "o3_started_local": started,
        "o3_finished_local": finished,
    }
    doc = write_manifest(manifest_path, [result], [case], extra=extra)
    print(f"[O3 FINAL] {finished} END status={result.status} wall={result.wall_time_s:.3f}s")
    return result, doc

if __name__ == "__main__":
    raise SystemExit(stage_a_cli_main())

